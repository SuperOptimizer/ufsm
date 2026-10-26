/* ufsm wbench-stats A.tsv [B.tsv] [--metric ids,p1x,...] [--set clean|all] [--ceil-min 0.8] [--n 10000] [--seed 1] [--boxes 0|1]
   Statistics of the winding+fitter bench (scripts/wbench/wbench.sh rows: a header line naming the columns, one row per box; columns
   used: "box", "scroll", the metrics, "ceil" for the clean set; "na" = missing). For each metric: the mean over boxes with a 95 %
   percentile bootstrap CI, resampling boxes WITH replacement inside each scroll (stratified: every scroll keeps its box count), the
   per-scroll means; with B: B's mean, the paired difference A - B (same resample for both: only boxes scored in both), its CI,
   the bootstrap share of A > B, and the minimal detectable difference MDE = 2.8 x the bootstrap SE of the difference (80 % power at
   two-sided 5 %). --set clean (default) keeps boxes with ceil >= --ceil-min (A's ceil; boxes without a ceil are kept; the others are summarised on a "label-noise" line); --boxes 1
   prints the per-box values. Several rows of one box (replicates) are averaged first. --scroll a,b keeps only the boxes of those
   scrolls (e.g. the wgate primary set paris4,paris4j); --tsv 1 prints one tab-separated line per metric instead (scripts/wbench/wgate.sh). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { WB_MAXBOX = 1024, WB_MAXCOL = 128, WB_MAXSC = 32 };
typedef struct { char box[64], scroll[64]; double v[WB_MAXCOL]; int nv[WB_MAXCOL]; } wb_row;
typedef struct { int ncol; char col[WB_MAXCOL][48]; int nrow; wb_row *row; } wb_tab;

static const char *wb_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
static int wb_colix(const wb_tab *t, const char *c) { for (int i = 0; i < t->ncol; i++) if (!strcmp(t->col[i], c)) return i; return -1; }
static int wb_split(char *s, char **f, int max) {   /* tab or space separated */
    int n = 0; char *p = s;
    while (*p && n < max) { while (*p == ' ' || *p == '\t') p++; if (!*p || *p == '\n') break; f[n++] = p; while (*p && *p != ' ' && *p != '\t' && *p != '\n') p++; if (*p) *p++ = 0; }
    return n;
}
static int wb_load(const char *path, wb_tab *t) {
    FILE *f = fopen(path, "r"); if (!f) { fprintf(stderr, "wbench-stats: cannot open %s\n", path); return 1; }
    memset(t, 0, sizeof *t); t->row = calloc(WB_MAXBOX, sizeof *t->row);
    char line[16384]; int hdr = 0, ib = -1, is = -1;
    while (fgets(line, sizeof line, f)) {
        char *fl[WB_MAXCOL + 8]; if (line[0] == '#' && !hdr) memmove(line, line + 1, strlen(line));
        else if (line[0] == '#') continue;
        const int n = wb_split(line, fl, WB_MAXCOL);
        if (!n) continue;
        if (!hdr) {
            hdr = 1; t->ncol = n; for (int i = 0; i < n; i++) snprintf(t->col[i], sizeof t->col[i], "%s", fl[i]);
            ib = wb_colix(t, "box"); is = wb_colix(t, "scroll");
            if (ib < 0 || is < 0) { fprintf(stderr, "wbench-stats: %s: the header needs \"box\" and \"scroll\" columns\n", path); fclose(f); return 1; }
            continue;
        }
        if (!strcmp(fl[0], t->col[0])) continue;   /* a repeated header */
        if (n <= ib || n <= is) continue;
        int r = 0; while (r < t->nrow && strcmp(t->row[r].box, fl[ib])) r++;
        if (r == t->nrow) { if (t->nrow == WB_MAXBOX) continue; t->nrow++; snprintf(t->row[r].box, 64, "%s", fl[ib]); snprintf(t->row[r].scroll, 64, "%s", fl[is]); }
        for (int c = 0; c < n && c < t->ncol; c++) { char *e; const double v = strtod(fl[c], &e); if (e != fl[c] && !*e && isfinite(v)) { t->row[r].v[c] += v; t->row[r].nv[c]++; } }
    }
    fclose(f);
    for (int r = 0; r < t->nrow; r++) for (int c = 0; c < t->ncol; c++) t->row[r].v[c] = t->row[r].nv[c] ? t->row[r].v[c] / t->row[r].nv[c] : NAN;
    return hdr ? 0 : 1;
}
static uint64_t wb_s;
static inline uint64_t wb_rnd(void) { uint64_t x = (wb_s += 0x9e3779b97f4a7c15ull); x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull; x = (x ^ (x >> 27)) * 0x94d049bb133111ebull; return x ^ (x >> 31); }
static int wb_dcmp(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
static double wb_q(double *v, int n, double q) { qsort(v, (size_t)n, sizeof *v, wb_dcmp); double p = q * (n - 1); int i = (int)p; return i + 1 < n ? v[i] + (p - i) * (v[i + 1] - v[i]) : v[n - 1]; }

int cmd_wbench_stats(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: ufsm wbench-stats A.tsv [B.tsv] [--metric ids,p1x] [--set clean|all] [--ceil-min 0.8] [--n 10000] [--seed 1] [--boxes 0|1]\n"); return 2; }
    const char *pa = argv[2], *pb = argc > 3 && argv[3][0] != '-' ? argv[3] : nullptr;
    static wb_tab A, B; if (wb_load(pa, &A) || (pb && wb_load(pb, &B))) return 1;
    char ml[1024]; snprintf(ml, sizeof ml, "%s", wb_opt(argc, argv, "--metric", "ids"));
    const int clean = strcmp(wb_opt(argc, argv, "--set", "clean"), "all") != 0, nb = atoi(wb_opt(argc, argv, "--n", "10000")), pbox = atoi(wb_opt(argc, argv, "--boxes", "0"));
    const double cmin = atof(wb_opt(argc, argv, "--ceil-min", "0.8")); wb_s = (uint64_t)atoll(wb_opt(argc, argv, "--seed", "1"));
    const int ic = wb_colix(&A, "ceil"), tsv = atoi(wb_opt(argc, argv, "--tsv", "0"));
    const char *only = wb_opt(argc, argv, "--scroll", nullptr);   /* comma list: only boxes of these scrolls */
    if (tsv) printf("metric\tscrolls\tn\tA\tA_lo\tA_hi\tB\tdiff\tdiff_lo\tdiff_hi\tP_A_gt_B\tmde\n");
    else printf("wbench-stats A = %s%s%s, set %s%s%s%s%s\n", pa, pb ? ", B = " : "", pb ? pb : "", clean ? "clean (ceil >= " : "all", clean ? wb_opt(argc, argv, "--ceil-min", "0.8") : "", clean ? "; boxes without a ceil kept)" : "",
                only ? ", scrolls " : "", only ? only : "");
    for (char *m = strtok(ml, ","); m; m = strtok(nullptr, ",")) {
        const int ia = wb_colix(&A, m), ibb = pb ? wb_colix(&B, m) : -1;
        if (ia < 0 || (pb && ibb < 0)) { printf("metric %s: not a column of %s\n", m, ia < 0 ? pa : pb); continue; }
        /* boxes: A's rows (in the set) with a value; paired: also scored in B */
        int idx[WB_MAXBOX], jb[WB_MAXBOX], sc[WB_MAXBOX], n = 0, ns = 0; int nx = 0; double sx = 0, sxc = 0; char sname[WB_MAXSC][64];
        for (int r = 0; r < A.nrow; r++) {
            const double v = A.row[r].v[ia]; if (!isfinite(v)) continue;
            if (only) { const char *s = strstr(only, A.row[r].scroll); const size_t l = strlen(A.row[r].scroll);   /* whole comma-list items only */
                while (s && !((s == only || s[-1] == ',') && (s[l] == 0 || s[l] == ','))) s = strstr(s + 1, A.row[r].scroll);
                if (!s) continue; }
            if (clean && ic >= 0 && isfinite(A.row[r].v[ic]) && A.row[r].v[ic] < cmin) { nx++; sx += v; sxc += A.row[r].v[ic]; continue; }
            int j = -1; if (pb) { for (int q = 0; q < B.nrow; q++) if (!strcmp(B.row[q].box, A.row[r].box)) j = q; if (j < 0 || !isfinite(B.row[j].v[ibb])) continue; }
            int s = 0; while (s < ns && strcmp(sname[s], A.row[r].scroll)) s++;
            if (s == ns) { if (ns == WB_MAXSC) continue; snprintf(sname[ns++], 64, "%s", A.row[r].scroll); }
            idx[n] = r; jb[n] = j; sc[n] = s; n++;
        }
        if (!n) { if (tsv) printf("%s\t%s\t0\tna\tna\tna\tna\tna\tna\tna\tna\tna\n", m, only ? only : "all"); else printf("metric %s: no boxes\n", m); continue; }
        double ma = 0, mb = 0; for (int i = 0; i < n; i++) { ma += A.row[idx[i]].v[ia]; if (pb) mb += B.row[jb[i]].v[ibb]; } ma /= n; mb /= n;
        int cnt[WB_MAXSC] = {0}, mem[WB_MAXSC][WB_MAXBOX]; for (int i = 0; i < n; i++) mem[sc[i]][cnt[sc[i]]++] = i;
        double *ba = malloc((size_t)nb * sizeof(double)), *bb = malloc((size_t)nb * sizeof(double)), *bd = malloc((size_t)nb * sizeof(double)); int gt = 0;
        for (int k = 0; k < nb; k++) {
            double sa = 0, sb = 0;
            for (int s = 0; s < ns; s++) for (int q = 0; q < cnt[s]; q++) { const int i = mem[s][wb_rnd() % (uint64_t)cnt[s]]; sa += A.row[idx[i]].v[ia]; if (pb) sb += B.row[jb[i]].v[ibb]; }
            ba[k] = sa / n; bb[k] = sb / n; bd[k] = (sa - sb) / n; gt += sa > sb;
        }
        double sd = 0, mdm = 0; for (int k = 0; k < nb; k++) mdm += bd[k]; mdm /= nb; for (int k = 0; k < nb; k++) sd += (bd[k] - mdm) * (bd[k] - mdm); sd = sqrt(sd / (nb > 1 ? nb - 1 : 1));
        if (tsv) {   /* one machine-readable line per metric (wgate.sh) */
            const double alo = wb_q(ba, nb, 0.025), ahi = wb_q(ba, nb, 0.975);
            printf("%s\t%s\t%d\t%.4f\t%.4f\t%.4f", m, only ? only : "all", n, ma, alo, ahi);
            if (pb) { const double dlo = wb_q(bd, nb, 0.025), dhi = wb_q(bd, nb, 0.975); printf("\t%.4f\t%+.4f\t%+.4f\t%+.4f\t%.3f\t%.4f\n", mb, ma - mb, dlo, dhi, (double)gt / nb, 2.8 * sd); }
            else printf("\tna\tna\tna\tna\tna\tna\n");
            free(ba); free(bb); free(bd); continue;
        }
        printf("%-10s n %3d | A %.4f [%.4f, %.4f]", m, n, ma, wb_q(ba, nb, 0.025), wb_q(ba, nb, 0.975));
        if (pb) printf(" | B %.4f [%.4f, %.4f] | A-B %+.4f [%+.4f, %+.4f] P(A>B) %.3f MDE %.4f", mb, wb_q(bb, nb, 0.025), wb_q(bb, nb, 0.975), ma - mb, wb_q(bd, nb, 0.025), wb_q(bd, nb, 0.975), (double)gt / nb, 2.8 * sd);
        printf("\n");
        if (nx) printf("    label-noise  %3d boxes  A %.4f  (ceil %.4f; not in the mean)\n", nx, sx / nx, sxc / nx);
        for (int s = 0; s < ns; s++) {
            double a = 0, b = 0; for (int q = 0; q < cnt[s]; q++) { a += A.row[idx[mem[s][q]]].v[ia]; if (pb) b += B.row[jb[mem[s][q]]].v[ibb]; }
            printf("    %-12s %3d boxes  A %.4f", sname[s], cnt[s], a / cnt[s]); if (pb) printf("  B %.4f  A-B %+.4f", b / cnt[s], (a - b) / cnt[s]); printf("\n");
        }
        if (pbox) for (int i = 0; i < n; i++) { printf("      %-10s %-12s A %.4f", A.row[idx[i]].box, A.row[idx[i]].scroll, A.row[idx[i]].v[ia]); if (pb) printf("  B %.4f  A-B %+.4f", B.row[jb[i]].v[ibb], A.row[idx[i]].v[ia] - B.row[jb[i]].v[ibb]); printf("\n"); }
        free(ba); free(bb); free(bd);
    }
    return 0;
}

/* ufsm ckpt-avg OUT.ckpt IN1.ckpt IN2.ckpt ... [--arrays ema|both]
   Uniform average of the weights of several snapshots of ONE run (same network: identical header up to "nparams" and equal file
   size): the EMA weights (array 1, what predict --ema 1 reads) and, with --arrays both (default), the live weights (array 0).
   The optimiser state (m, v, Muon momentum) and the header (step, Hyperball radii, level-0 running statistics) are the LAST
   input's, so the output loads and resumes like that snapshot. A late-schedule average stands in for a cooled checkpoint when a
   gate has to read a run mid-schedule (peak LR). Host only (no CUDA), streamed in 4 MB chunks, exact in double. */
static long wb_hdr(FILE *f, char *line, size_t cap) {   /* reads "UFSM" + the header line; returns the payload offset or -1 */
    char mg[4]; if (fread(mg, 1, 4, f) != 4 || memcmp(mg, "UFSM", 4) || !fgets(line, (int)cap, f) || !strchr(line, '\n')) return -1;
    return ftell(f);
}
int cmd_ckpt_avg(int argc, char **argv) {
    int nin = 0; const char *in[256]; const char *out = argc > 2 ? argv[2] : nullptr;
    for (int i = 3; i < argc; i++) { if (!strcmp(argv[i], "--arrays")) { i++; continue; } if (nin < 256) in[nin++] = argv[i]; }
    const char *ar = wb_opt(argc, argv, "--arrays", "both"); const int both = strcmp(ar, "ema") != 0;
    if (!out || nin < 1 || (strcmp(ar, "ema") && strcmp(ar, "both"))) { fprintf(stderr, "usage: ufsm ckpt-avg OUT.ckpt IN1.ckpt [IN2.ckpt ...] [--arrays ema|both]\n"); return 2; }
    enum { HL = 16384 }; static char h0[HL], h1[HL]; FILE *f[256]; long off[256]; size_t np = 0; long long sz = -1;
    for (int k = 0; k < nin; k++) {
        f[k] = fopen(in[k], "rb"); if (!f[k]) { fprintf(stderr, "ckpt-avg: cannot open %s\n", in[k]); return 1; }
        char *h = k ? h1 : h0; off[k] = wb_hdr(f[k], h, HL); if (off[k] < 0) { fprintf(stderr, "ckpt-avg: %s is not a ufsm checkpoint\n", in[k]); return 1; }
        const char *p = strstr(h, "\"nparams\":"); if (!p) { fprintf(stderr, "ckpt-avg: %s has no nparams\n", in[k]); return 1; }
        const size_t n = (size_t)atoll(p + 10), pre = (size_t)(p - h);
        fseek(f[k], 0, SEEK_END); const long long s = ftell(f[k]) - off[k]; fseek(f[k], off[k], SEEK_SET);
        if (!k) { np = n; sz = s; }
        else if (n != np || s != sz || strncmp(h0, h1, pre)) { fprintf(stderr, "ckpt-avg: %s is not the same network as %s\n", in[k], in[0]); return 1; }
        if (k == nin - 1 && k) memcpy(h0, h1, HL);   /* the header of the last input */
    }
    if (sz < (long long)(2 * np * 4)) { fprintf(stderr, "ckpt-avg: truncated checkpoint\n"); return 1; }
    char tmp[4096]; snprintf(tmp, sizeof tmp, "%s.tmp", out);
    FILE *o = fopen(tmp, "wb"); if (!o) { fprintf(stderr, "ckpt-avg: cannot write %s\n", tmp); return 1; }
    fputs("UFSM", o); fputs(h0, o);
    enum { CH = 1 << 20 }; float *b = malloc(CH * 4); double *acc = malloc(CH * sizeof(double)); int rc = 0;
    for (int a = 0; a < 2 && !rc; a++) {   /* arrays 0 (live) and 1 (EMA) */
        const int avg = a == 1 || both;
        for (size_t s0 = 0; s0 < np && !rc; s0 += CH) {
            const size_t c = np - s0 < CH ? np - s0 : CH;
            if (avg) {
                for (size_t i = 0; i < c; i++) acc[i] = 0;
                for (int k = 0; k < nin && !rc; k++) {
                    if (fseek(f[k], off[k] + (long)((a * np + s0) * 4), SEEK_SET) || fread(b, 4, c, f[k]) != c) { rc = 1; break; }
                    for (size_t i = 0; i < c; i++) acc[i] += b[i];
                }
                for (size_t i = 0; i < c; i++) b[i] = (float)(acc[i] / nin);
            } else if (fseek(f[nin - 1], off[nin - 1] + (long)((a * np + s0) * 4), SEEK_SET) || fread(b, 4, c, f[nin - 1]) != c) rc = 1;
            if (!rc && fwrite(b, 4, c, o) != c) rc = 1;
        }
    }
    /* the rest (optimiser state, Muon momentum) from the last input */
    if (!rc && fseek(f[nin - 1], off[nin - 1] + (long)(2 * np * 4), SEEK_SET)) rc = 1;
    for (size_t r; !rc && (r = fread(b, 1, CH * 4, f[nin - 1])) > 0;) if (fwrite(b, 1, r, o) != r) rc = 1;
    for (int k = 0; k < nin; k++) fclose(f[k]);
    free(b); free(acc);
    if (fclose(o) || rc) { fprintf(stderr, "ckpt-avg: read / write error\n"); remove(tmp); return 1; }
    if (rename(tmp, out)) { perror("ckpt-avg: rename"); return 1; }
    fprintf(stderr, "ckpt-avg: %d checkpoint(s), %zu parameters, %s averaged -> %s\n", nin, np, both ? "live + EMA weights" : "EMA weights", out);
    return 0;
}
