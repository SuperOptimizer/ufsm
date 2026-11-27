/* ufsm ink-strat (inkeval 2026-11-27): per-crop ink scores (devscore / cleanscore / pair rows: TAG CROP .. AUC) stratified by scroll and
   by scan recipe, so cross-scroll generalisation is visible instead of hidden in a pooled mean. Every scroll weighs the same (the mean
   of the scroll means), the worst scroll is named, and with --leaks every cell carries the model's training exposure to that scroll, so
   the worst *clean* scroll is reported apart from the leaked ones. With --ref the rows are paired by crop (A - REF per scroll ± SE).
   The recipe comes from the CT key ("20250521151220-8.640um-1.2m-116keV-masked.zarr" -> 8.64um/1.2m/116keV): pixel size, propagation
   distance and energy are the scan's generalisation axes (arXiv 2606.29085 Ext. Data Table 1; Paganin delta/beta is not in our metadata).
   No GPU, no maps: it reads the rows the scorers already wrote. */
#include "optcheck.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { char **s; int n, cap; } strtab;
static int st_find(const strtab *t, const char *s) { for (int i = 0; i < t->n; i++) if (!strcmp(t->s[i], s)) return i; return -1; }
static int st_add(strtab *t, const char *s) {
    const int i = st_find(t, s); if (i >= 0) return i;
    if (t->n == t->cap) { t->cap = t->cap ? 2 * t->cap : 64; t->s = realloc(t->s, (size_t)t->cap * sizeof *t->s); }
    t->s[t->n] = strdup(s); return t->n++;
}
typedef struct { int tag, crop; double v; } srow;
typedef struct { char tag[128]; char scroll[64]; char flag; int wild; } sleak;

static const char *sopt_(int argc, char **argv, const char *name, const char *def) {
    for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(name))) return argv[i + 1];
    return def;
}
/* the scroll of a crop name: its leading [A-Za-z0-9] run ("PHerc0009B-h116_s00_20_7" -> PHerc0009B, "PHerc1447_w058_0" -> PHerc1447) */
static void crop_scroll(const char *c, char *out, size_t n) {
    size_t k = 0; while (c[k] && k + 1 < n && ((c[k] >= 'A' && c[k] <= 'Z') || (c[k] >= 'a' && c[k] <= 'z') || (c[k] >= '0' && c[k] <= '9'))) k++;
    memcpy(out, c, k); out[k] = 0;
}
/* "PHerc0343/20250521140437-8.640um-1.2m-116keV-masked.zarr" -> "8.64um/1.2m/116keV"; 0 when no "um" token */
static int ct_recipe(const char *key, char *out, size_t n) {
    const char *b = strrchr(key, '/'); b = b ? b + 1 : key;
    char um[32] = "", dist[32] = "-", kev[32] = "-", tok[64]; int have = 0;
    for (const char *p = b; *p; ) {
        size_t l = strcspn(p, "-."); if (p[l] == '.' && p[l + 1] >= '0' && p[l + 1] <= '9') l += 1 + strcspn(p + l + 1, "-.");   /* keep 8.640 / 1.2 whole */
        if (l && l < sizeof tok) { memcpy(tok, p, l); tok[l] = 0;
            char *e; const double x = strtod(tok, &e);
            if (e != tok && !strcmp(e, "um")) { snprintf(um, sizeof um, "%gum", x); have = 1; }
            else if (e != tok && !strcmp(e, "m")) snprintf(dist, sizeof dist, "%gm", x);
            else if (e != tok && !strcmp(e, "keV")) snprintf(kev, sizeof kev, "%gkeV", x); }
        p += l; if (*p) p++;
    }
    if (have) snprintf(out, n, "%s/%s/%s", um, dist, kev);
    return have;
}
static int parse_cols(const char *s, int *c, int n) {   /* "1,2,4[,3]" */
    int k = 0; for (const char *p = s; *p && k < n; ) { char *e; const long v = strtol(p, &e, 10); if (e == p || v < 1) return -1; c[k++] = (int)v; p = *e == ',' ? e + 1 : e; }
    return k;
}
static int split_tab(char *ln, char **f, int nf) { int k = 0; char *p = ln; while (k < nf) { f[k++] = p; char *t = strchr(p, '\t'); if (!t) break; *t = 0; p = t + 1; } return k; }

static char leak_flag(const sleak *L, int n, const char *tag, const char *scroll) {
    char f = '?';
    for (int i = 0; i < n; i++) {
        const int tm = L[i].wild ? !strncmp(tag, L[i].tag, strlen(L[i].tag)) : !strcmp(tag, L[i].tag);
        if (tm && (!strcmp(L[i].scroll, "*") || !strcmp(L[i].scroll, scroll))) f = L[i].flag;   /* the last match wins */
    }
    return f;
}
static double cell_stats(const double *v, int n, double *se) {
    double s = 0, ss = 0; for (int i = 0; i < n; i++) { s += v[i]; ss += v[i] * v[i]; }
    const double m = n ? s / n : NAN; *se = n > 1 ? sqrt(fmax(0, (ss - s * m) / (n - 1)) / n) : NAN; return m;
}

int cmd_ink_strat(int argc, char **argv) {
    int nrows_opt = 0; for (int i = 2; i + 1 < argc; i++) nrows_opt += !strcmp(argv[i], opt_n("--rows"));
    if (!nrows_opt) {
        fprintf(stderr, "usage: ufsm ink-strat --rows FILE[:TAG,CROP,AUC[,SCROLL]] [--rows ..] [--crops TSV ..] [--scan SCROLL=CTKEY ..] [--alias OLD=NEW ..]\n"
                        "         [--tags A,B,..] [--metric SUFFIX] [--leaks TSV] [--ref TAG] [--min-n N] [--worst-n 5] [--tsv OUT]\n"
                        "  --worst-n N: the worst-scroll lines only consider scrolls with >= N crops (default 5); means use every scroll with >= --min-n.\n"
                        "  rows: one score per (model, crop); columns 1-based, default 1,2,4 (devscore's TAG CROP SCROLL AUC SUP), the scroll from the crop name\n"
                        "  unless SCROLL is given. Lines starting with # and non-numeric / nan scores are skipped (counted). A second row of the same (model, crop)\n"
                        "  is an error unless both scores agree to 1e-4.\n"
                        "  --metric SUFFIX: only rows whose TAG ends in SUFFIX, which is cut off (pair rows NAME:band2 etc.); default every row.\n"
                        "  --crops TSV: crop in column 1, its CT key = the first field holding \"um-\" (devscore's crops.tsv); --scan SCROLL=KEY for the rest.\n"
                        "  --leaks TSV: TAG SCROLL FLAG [note] lines (TAG may end in *, SCROLL may be *; the last match wins); FLAG C clean, S scroll seen,\n"
                        "  X cross-scan leak (same region, another scan), L leak, U unknown. Unlisted cells are '?' and count as not clean.\n");
        return 2;
    }
    strtab tags = {0}, crops = {0}, scrolls = {0};
    int *crop_scroll_ix = nullptr; int ccap = 0;
    srow *R = nullptr; int nr = 0, rcap = 0, skipped = 0, rc = 0;
    enum { MAXAL = 64 }; char alo[MAXAL][128], aln[MAXAL][128]; int nal = 0;   /* --alias OLD=NEW: one model name over rows from several scorers */
    for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n("--alias"))) {
        const char *e = strchr(argv[i + 1], '='); if (!e || nal == MAXAL) { fprintf(stderr, "ink-strat: --alias OLD=NEW (at most %d)\n", MAXAL); return 2; }
        snprintf(alo[nal], 128, "%.*s", (int)(e - argv[i + 1]), argv[i + 1]); snprintf(aln[nal], 128, "%s", e + 1); nal++; }
    const char *metric = sopt_(argc, argv, "--metric", nullptr); const size_t ml = metric ? strlen(metric) : 0;
    const char *tagsel = sopt_(argc, argv, "--tags", nullptr);
    for (int a = 2; a + 1 < argc; a++) {
        if (strcmp(argv[a], opt_n("--rows"))) continue;
        char path[1024]; snprintf(path, sizeof path, "%s", argv[a + 1]); int col[4] = {1, 2, 4, 0};
        char *cs = strrchr(path, ':');
        if (cs && cs[1] >= '0' && cs[1] <= '9') { int c4[4] = {0}; const int k = parse_cols(cs + 1, c4, 4);
            if (k < 3) { fprintf(stderr, "ink-strat: %s: columns TAG,CROP,AUC[,SCROLL]\n", argv[a + 1]); return 2; }
            memcpy(col, c4, sizeof col); if (k == 3) col[3] = 0; *cs = 0; }
        FILE *f = fopen(path, "r"); if (!f) { fprintf(stderr, "ink-strat: cannot read %s\n", path); return 1; }
        char ln[8192]; int nl = 0, nk = 0, ns = 0;
        while (fgets(ln, sizeof ln, f)) {
            nl++; ln[strcspn(ln, "\r\n")] = 0; if (!ln[0] || ln[0] == '#') continue;
            char *fl[64]; const int nf = split_tab(ln, fl, 64);
            int mx = col[0]; for (int k = 1; k < 4; k++) if (col[k] > mx) mx = col[k];
            if (nf < mx) { ns++; continue; }
            char tg[256]; snprintf(tg, sizeof tg, "%s", fl[col[0] - 1]);
            if (metric) { const size_t tl = strlen(tg); if (tl <= ml || strcmp(tg + tl - ml, metric)) continue; tg[tl - ml] = 0; }
            for (int k = 0; k < nal; k++) if (!strcmp(tg, alo[k])) { snprintf(tg, sizeof tg, "%s", aln[k]); break; }
            char *e; const double v = strtod(fl[col[2] - 1], &e);
            if (e == fl[col[2] - 1] || *e || !isfinite(v)) { ns++; continue; }
            const char *cr = fl[col[1] - 1]; char sc[64];
            if (col[3]) snprintf(sc, sizeof sc, "%s", fl[col[3] - 1]); else crop_scroll(cr, sc, sizeof sc);
            const int ti = st_add(&tags, tg), si = st_add(&scrolls, sc), cnew = st_find(&crops, cr) < 0, ci = st_add(&crops, cr);
            if (ci >= ccap) { ccap = ci + 256; crop_scroll_ix = realloc(crop_scroll_ix, (size_t)ccap * sizeof *crop_scroll_ix); }
            if (cnew) crop_scroll_ix[ci] = si;
            if (crop_scroll_ix[ci] != si) { fprintf(stderr, "ink-strat: %s:%d: crop %s has scroll %s here but %s before\n", path, nl, cr, sc, scrolls.s[crop_scroll_ix[ci]]); rc = 1; continue; }
            int dup = -1; for (int r = 0; r < nr && dup < 0; r++) if (R[r].tag == ti && R[r].crop == ci) dup = r;
            if (dup >= 0) { if (fabs(R[dup].v - v) > 1e-4) { fprintf(stderr, "ink-strat: %s:%d: %s / %s scored twice (%.4f, %.4f): one row per model and crop\n", path, nl, tg, cr, R[dup].v, v); rc = 1; } continue; }
            if (nr == rcap) { rcap = rcap ? 2 * rcap : 1024; R = realloc(R, (size_t)rcap * sizeof *R); }
            R[nr++] = (srow){ti, ci, v}; nk++;
        }
        fclose(f); skipped += ns;
        fprintf(stderr, "ink-strat: %s: %d rows kept, %d skipped (non-numeric / nan / short)\n", path, nk, ns);
    }
    if (rc) return rc;
    if (!nr) { fprintf(stderr, "ink-strat: no rows\n"); return 1; }
    /* scan recipe per scroll: --crops (crop -> CT key), then --scan SCROLL=KEY */
    char (*rec)[48] = calloc((size_t)scrolls.n, sizeof *rec);
    for (int a = 2; a + 1 < argc; a++) {
        if (!strcmp(argv[a], opt_n("--crops"))) {
            FILE *f = fopen(argv[a + 1], "r"); if (!f) { fprintf(stderr, "ink-strat: cannot read %s\n", argv[a + 1]); return 1; }
            char ln[8192];
            while (fgets(ln, sizeof ln, f)) { ln[strcspn(ln, "\r\n")] = 0; if (ln[0] == '#') continue; char *fl[64]; const int nf = split_tab(ln, fl, 64);
                const int ci = st_find(&crops, fl[0]); if (ci < 0) continue;
                for (int k = 1; k < nf; k++) if (strstr(fl[k], "um-")) { char r[48]; if (ct_recipe(fl[k], r, sizeof r)) {
                    char *cur = rec[crop_scroll_ix[ci]];
                    if (cur[0] && strcmp(cur, r)) { fprintf(stderr, "ink-strat: scroll %s has crops on two recipes (%s, %s): split it by scan (--rows ..:SCROLL with a scan column)\n", scrolls.s[crop_scroll_ix[ci]], cur, r); return 1; }
                    snprintf(cur, 48, "%s", r); } break; } }
            fclose(f);
        } else if (!strcmp(argv[a], opt_n("--scan"))) {
            const char *e = strchr(argv[a + 1], '='); if (!e) { fprintf(stderr, "ink-strat: --scan SCROLL=CTKEY\n"); return 2; }
            char s[64]; snprintf(s, sizeof s, "%.*s", (int)(e - argv[a + 1]), argv[a + 1]); const int si = st_find(&scrolls, s);
            char r[48]; if (!ct_recipe(e + 1, r, sizeof r)) { fprintf(stderr, "ink-strat: --scan %s: no <um>um token in the key\n", argv[a + 1]); return 2; }
            if (si >= 0 && !rec[si][0]) snprintf(rec[si], 48, "%s", r);
        }
    }
    for (int s = 0; s < scrolls.n; s++) if (!rec[s][0]) { fprintf(stderr, "ink-strat: %s: no scan given (--crops / --scan): recipe \"?\"\n", scrolls.s[s]); snprintf(rec[s], 48, "?"); }
    /* the models shown, in --tags order or first appearance */
    int *T = malloc((size_t)tags.n * sizeof *T), nt = 0;
    if (tagsel) { char b[4096]; snprintf(b, sizeof b, "%s", tagsel);
        for (char *p = strtok(b, ","); p; p = strtok(nullptr, ",")) { const int ti = st_find(&tags, p); if (ti < 0) { fprintf(stderr, "ink-strat: --tags %s: no rows\n", p); return 1; } T[nt++] = ti; } }
    else for (int t = 0; t < tags.n; t++) T[nt++] = t;
    const char *reft = sopt_(argc, argv, "--ref", nullptr); const int ref = reft ? st_find(&tags, reft) : -1;
    if (reft && ref < 0) { fprintf(stderr, "ink-strat: --ref %s: no rows\n", reft); return 1; }
    const int minn = opt_atoi("--min-n", sopt_(argc, argv, "--min-n", "1"));
    const int wn = opt_atoi("--worst-n", sopt_(argc, argv, "--worst-n", "5"));   /* a 1-crop scroll is noise, not a worst scroll */
    /* leak flags */
    sleak *LK = nullptr; int nlk = 0;
    const char *lkf = sopt_(argc, argv, "--leaks", nullptr);
    if (lkf) { FILE *f = fopen(lkf, "r"); if (!f) { fprintf(stderr, "ink-strat: cannot read %s\n", lkf); return 1; }
        char ln[4096]; int nl = 0;
        while (fgets(ln, sizeof ln, f)) { nl++; ln[strcspn(ln, "\r\n")] = 0; if (!ln[0] || ln[0] == '#') continue; char *fl[8]; const int nf = split_tab(ln, fl, 8);
            if (nf < 3 || !strchr("CSXLU", fl[2][0]) || fl[2][1]) { fprintf(stderr, "ink-strat: %s:%d: TAG SCROLL C|S|X|L|U [note]\n", lkf, nl); fclose(f); return 2; }
            LK = realloc(LK, (size_t)(nlk + 1) * sizeof *LK); sleak *l = &LK[nlk++]; memset(l, 0, sizeof *l);
            snprintf(l->tag, sizeof l->tag, "%s", fl[0]); snprintf(l->scroll, sizeof l->scroll, "%s", fl[1]); l->flag = fl[2][0];
            const size_t tl = strlen(l->tag); if (tl && l->tag[tl - 1] == '*') { l->tag[tl - 1] = 0; l->wild = 1; } }
        fclose(f); }
    #define FLAG(ti_, si_) leak_flag(LK, nlk, tags.s[ti_], scrolls.s[si_])
    /* scroll order: by recipe, then name */
    int *SO = malloc((size_t)scrolls.n * sizeof *SO); for (int s = 0; s < scrolls.n; s++) SO[s] = s;
    for (int i = 1; i < scrolls.n; i++) for (int j = i; j > 0; j--) { const int a = SO[j - 1], b = SO[j]; const int c = strcmp(rec[a], rec[b]);
        if (c > 0 || (c == 0 && strcmp(scrolls.s[a], scrolls.s[b]) > 0)) { SO[j - 1] = b; SO[j] = a; } else break; }
    /* per (model, scroll) values */
    double *buf = malloc((size_t)nr * sizeof *buf);
    double *M = malloc((size_t)tags.n * scrolls.n * sizeof *M); int *N = calloc((size_t)tags.n * scrolls.n, sizeof *N);
    for (int t = 0; t < tags.n; t++) for (int s = 0; s < scrolls.n; s++) { int n = 0; for (int r = 0; r < nr; r++) if (R[r].tag == t && crop_scroll_ix[R[r].crop] == s) buf[n++] = R[r].v;
        double se; M[(size_t)t * scrolls.n + s] = cell_stats(buf, n, &se); N[(size_t)t * scrolls.n + s] = n; }
    FILE *tsv = nullptr; const char *tsvp = sopt_(argc, argv, "--tsv", nullptr);
    if (tsvp && !(tsv = fopen(tsvp, "w"))) { fprintf(stderr, "ink-strat: cannot write %s\n", tsvp); return 1; }
    if (tsv) fprintf(tsv, "#model\tscroll\trecipe\tflag\tn\tmean\tse\n");
    printf("ink-strat: %d models, %d scrolls, %d crops, %d rows (%d skipped); cell = mean crop AUC (n crops) + training exposure%s\n", nt, scrolls.n, crops.n, nr, skipped,
           lkf ? " (C clean, S scroll seen, X cross-scan, L leak, U unknown, ? unlisted)" : " (no --leaks: every cell '?')");
    printf("%-12s %-20s", "scroll", "recipe"); for (int k = 0; k < nt; k++) printf(" %18.18s", tags.s[T[k]]); printf("\n");
    for (int q = 0; q < scrolls.n; q++) { const int s = SO[q];
        printf("%-12s %-20s", scrolls.s[s], rec[s]);
        for (int k = 0; k < nt; k++) { const int t = T[k]; const int n = N[(size_t)t * scrolls.n + s];
            if (!n) { printf(" %18s", "-"); continue; }
            char c[40]; snprintf(c, sizeof c, "%.4f (%d) %c", M[(size_t)t * scrolls.n + s], n, FLAG(t, s)); printf(" %18s", c);
            if (tsv) { int m = 0; for (int r = 0; r < nr; r++) if (R[r].tag == t && crop_scroll_ix[R[r].crop] == s) buf[m++] = R[r].v; double se; cell_stats(buf, m, &se);
                fprintf(tsv, "%s\t%s\t%s\t%c\t%d\t%.4f\t%.4f\n", tags.s[t], scrolls.s[s], rec[s], FLAG(t, s), n, M[(size_t)t * scrolls.n + s], se); } }
        printf("\n"); }
    /* per recipe: the mean of its scroll means */
    printf("%-33s", "per recipe (mean of scrolls)"); for (int k = 0; k < nt; k++) printf(" %18.18s", tags.s[T[k]]); printf("\n");
    for (int q = 0; q < scrolls.n; ) { const int s0 = SO[q]; int q1 = q; while (q1 < scrolls.n && !strcmp(rec[SO[q1]], rec[s0])) q1++;
        printf("%-33s", rec[s0]);
        for (int k = 0; k < nt; k++) { const int t = T[k]; double a = 0; int ns = 0, nc = 0;
            for (int j = q; j < q1; j++) { const int s = SO[j], n = N[(size_t)t * scrolls.n + s]; if (n >= minn) { a += M[(size_t)t * scrolls.n + s]; ns++; nc += n; } }
            if (!ns) printf(" %18s", "-"); else { char c[40]; snprintf(c, sizeof c, "%.4f (%ds %d)", a / ns, ns, nc); printf(" %18s", c); } }
        printf("\n"); q = q1; }
    /* the summary lines: mean of scrolls, worst scroll, worst clean scroll, pooled crop mean */
    for (int k = 0; k < nt; k++) { const int t = T[k];
        double a = 0, w = INFINITY, wc = INFINITY, ac = 0, cm = 0; int ns = 0, ws = -1, wcs = -1, nsc = 0, nc = 0;
        for (int s = 0; s < scrolls.n; s++) { const int n = N[(size_t)t * scrolls.n + s]; if (n < minn) continue; const double m = M[(size_t)t * scrolls.n + s];
            a += m; ns++; cm += m * n; nc += n; if (n >= wn && m < w) { w = m; ws = s; }
            if (FLAG(t, s) == 'C') { ac += m; nsc++; if (n >= wn && m < wc) { wc = m; wcs = s; } } }
        if (!ns) continue;
        if (ws >= 0) printf("worst-scroll %s: %s %.4f (%d crops, %s, %c)", tags.s[t], scrolls.s[ws], w, N[(size_t)t * scrolls.n + ws], rec[ws], FLAG(t, ws));
        else printf("worst-scroll %s: none with >= %d crops", tags.s[t], wn);
        printf(" | mean of %d scrolls %.4f | crop mean %.4f (%d)", ns, a / ns, cm / nc, nc);
        if (lkf) { if (wcs >= 0) printf(" | clean: worst %s %.4f, mean of %d %.4f", scrolls.s[wcs], wc, nsc, ac / nsc);
                   else if (nsc) printf(" | clean: mean of %d %.4f (none with >= %d crops)", nsc, ac / nsc, wn); else printf(" | clean: none"); }
        printf("\n"); }
    if (ref >= 0) {   /* paired by crop against REF */
        printf("vs %s, paired by crop (A - REF, mean ± SE, crops up / n):\n%-12s %-20s", reft, "scroll", "recipe");
        for (int k = 0; k < nt; k++) if (T[k] != ref) printf(" %22.22s", tags.s[T[k]]); printf("\n");
        double *ds = calloc((size_t)tags.n, sizeof *ds), *dv = calloc((size_t)tags.n, sizeof *dv), *dw = malloc((size_t)tags.n * sizeof *dw); int *dn = calloc((size_t)tags.n, sizeof *dn), *dws = malloc((size_t)tags.n * sizeof *dws);
        for (int t = 0; t < tags.n; t++) { dw[t] = INFINITY; dws[t] = -1; }
        double *rv = malloc((size_t)crops.n * sizeof *rv); uint8_t *rh = malloc((size_t)crops.n);
        memset(rh, 0, (size_t)crops.n); for (int r = 0; r < nr; r++) if (R[r].tag == ref) { rv[R[r].crop] = R[r].v; rh[R[r].crop] = 1; }
        for (int q = 0; q < scrolls.n; q++) { const int s = SO[q]; printf("%-12s %-20s", scrolls.s[s], rec[s]);
            for (int k = 0; k < nt; k++) { const int t = T[k]; if (t == ref) continue; int n = 0, up = 0;
                for (int r = 0; r < nr; r++) if (R[r].tag == t && crop_scroll_ix[R[r].crop] == s && rh[R[r].crop]) { buf[n] = R[r].v - rv[R[r].crop]; up += buf[n] > 0; n++; }
                if (n < minn || !n) { printf(" %22s", "-"); continue; }
                double se; const double m = cell_stats(buf, n, &se); char c[48]; snprintf(c, sizeof c, "%+.4f±%.4f %d/%d", m, isnan(se) ? 0 : se, up, n); printf(" %22s", c);
                ds[t] += m; dv[t] += isnan(se) ? 0 : se * se; dn[t]++; if (n >= wn && m < dw[t]) { dw[t] = m; dws[t] = s; } }
            printf("\n"); }
        for (int k = 0; k < nt; k++) { const int t = T[k]; if (t == ref || !dn[t]) continue;
            printf("vs-%s %s: mean of %d scroll deltas %+.4f ± %.4f", reft, tags.s[t], dn[t], ds[t] / dn[t], sqrt(dv[t]) / dn[t]);
            if (dws[t] >= 0) printf(" | worst scroll %s %+.4f\n", scrolls.s[dws[t]], dw[t]); else printf(" | worst scroll: none with >= %d crops\n", wn); }
        free(ds); free(dv); free(dw); free(dn); free(dws); free(rv); free(rh);
    }
    #undef FLAG
    if (tsv) fclose(tsv);
    free(R); free(buf); free(M); free(N); free(T); free(SO); free(rec); free(LK); free(crop_scroll_ix);
    return 0;
}
