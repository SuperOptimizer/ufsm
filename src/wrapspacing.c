/* ufsm wrap-spacing <intervals.tsv> [--win 3] [--dth 0.06] [--patch-deg 10] [--patch-z 4] [--gap-min 150] [--thr 0] [--out OUT.tsv]
   The layer-spacing diagnostic (physics.md #2, lead 2026-11-27): carbonised papyrus is close to constant thickness, so the normal gap
   between consecutive wraps should vary slowly along each wrap; a jump marks a wrap that left its sheet (a duplicate or a skipped
   sheet). Input: wrap-check --intervals (one line per interval k -> k + 1 on a ray: z y x r_mm k npk rct gap_um gtq; gtq the GT
   turns with --codes). Per interval, |ln| ratios of its gap against
     st  the median gap of the same wrap pair k on the --win neighbouring rays either side (same z level, within --dth rad),
     sz  the same pair on the nearest ray of the adjacent z levels,
     sr  the pairs k - 1 and k + 1 on the same ray (radial: the neighbouring layers),
     sct the local CT peak spacing rct,
   and the scores along = max(st, sz) (the along-sheet jump) and all = max(st, sz, sr). The rays' angle is taken about the z level's
   centroid of interval midpoints (wrap-check casts its rays from the umbilicus over full turns, so the centroid is close to it).
   With GT (gtq finite): an interval is wrong when |gtq - 1| >= 0.5 (0: two wraps on one sheet, 2+: a sheet skipped); prints the AUC
   of each score (and of wrap-check's own CT peak flag npk != 1) for wrong intervals, and the same per patch (wrap pair k x
   --patch-deg of angle x --patch-z z levels: score = the patch mean, wrong = more than half of its GT intervals wrong).
   Measured on the 1667 arm-C unroll (DESIGN 2026-11-27): the jump prior is weak once the gap itself is known (st patch AUC 0.80, but
   0.58 among gaps < 150 um); the small fitted gap is the predictor (interval AUC 0.89, patch 0.91: the GT errors sit in packed
   regions, where one fitted turn spans two sheets or two wraps share one). So the QC flag: gap < --gap-min um (150), or with --thr T
   also along > T. --out: per interval its scores and flag. CPU, single thread (1e5..1e6 intervals: seconds). */
#include "optcheck.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { float z, y, x, r, rct, gap, gtq; int k, npk; double th; float st, sz, sr, sct; int zi; } ws_iv;

static const char *ws_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 3; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(k))) return argv[i + 1]; return d; }
static int cmp_f(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }
static float med(float *v, int n) { if (n <= 0) return NAN; qsort(v, (size_t)n, sizeof *v, cmp_f); return n & 1 ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]); }
static float lr(float a, float b) { return a > 0 && b > 0 ? fabsf(logf(a / b)) : NAN; }
/* order: (zi, k, th) */
static int cmp_iv(const void *a, const void *b) {
    const ws_iv *p = a, *q = b;
    if (p->zi != q->zi) return p->zi - q->zi;
    if (p->k != q->k) return p->k - q->k;
    return (p->th > q->th) - (p->th < q->th);
}
typedef struct { size_t a, b; } span;   /* [a, b) of one (zi, k) group in the sorted array */
/* the group of (zi, k): binary search over the group table (sorted the same way) */
typedef struct { int zi, k; span s; } grp;
static int cmp_grp(const void *a, const void *b) { const grp *p = a, *q = b; return p->zi != q->zi ? p->zi - q->zi : p->k - q->k; }
static const grp *find_grp(const grp *G, size_t ng, int zi, int k) { grp key = {zi, k, {0, 0}}; return bsearch(&key, G, ng, sizeof *G, cmp_grp); }
/* the interval of group g nearest in angle to th (within dth; circular), or -1 */
static long nearest(const ws_iv *V, const grp *g, double th, double dth) {
    if (!g || g->s.b <= g->s.a) return -1;
    size_t lo = g->s.a, hi = g->s.b;
    while (lo < hi) { const size_t m = (lo + hi) / 2; if (V[m].th < th) lo = m + 1; else hi = m; }
    long best = -1; double bd = dth;
    const size_t c[4] = {lo, lo > g->s.a ? lo - 1 : g->s.b - 1, lo < g->s.b ? lo : g->s.a, g->s.a};
    for (int i = 0; i < 4; i++) { const size_t j = c[i] >= g->s.b ? g->s.b - 1 : c[i]; double d = fabs(V[j].th - th); if (d > M_PI) d = 2 * M_PI - d; if (d <= bd) { bd = d; best = (long)j; } }
    return best;
}
typedef struct { float s; int y; } sl;
static int cmp_sl(const void *a, const void *b) { const float x = ((const sl *)a)->s, y = ((const sl *)b)->s; return (x > y) - (x < y); }
/* AUC of score s for label y (Mann-Whitney, average ranks over ties); NAN scores skipped. n pos / neg out */
static double auc(sl *v, size_t n, long *np, long *nn) {
    size_t m = 0; for (size_t i = 0; i < n; i++) if (!isnan(v[i].s)) v[m++] = v[i];
    qsort(v, m, sizeof *v, cmp_sl);
    double rs = 0; long p = 0, q = 0;
    for (size_t i = 0; i < m;) { size_t j = i; while (j < m && v[j].s == v[i].s) j++; const double r = 0.5 * (double)(i + 1 + j); for (size_t t = i; t < j; t++) if (v[t].y) { rs += r; p++; } else q++; i = j; }
    *np = p; *nn = q; return p && q ? (rs - (double)p * (p + 1) / 2) / ((double)p * q) : NAN;
}

int cmd_wrap_spacing(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: ufsm wrap-spacing <intervals.tsv> [--win 3] [--dth 0.06] [--patch-deg 10] [--patch-z 4] [--gap-min 150] [--thr 0] [--out OUT.tsv]\n"); return 2; }
    const int win = opt_atoi("--win", ws_opt(argc, argv, "--win", "3")), pz = opt_atoi("--patch-z", ws_opt(argc, argv, "--patch-z", "4"));
    const double dth = opt_atof("--dth", ws_opt(argc, argv, "--dth", "0.06")), pdeg = opt_atof("--patch-deg", ws_opt(argc, argv, "--patch-deg", "10"));
    const double thr = opt_atof("--thr", ws_opt(argc, argv, "--thr", "0")), gmin = opt_atof("--gap-min", ws_opt(argc, argv, "--gap-min", "150"));
    const char *out = ws_opt(argc, argv, "--out", nullptr);
    if (win < 1 || pz < 1 || !(dth > 0) || !(pdeg > 0)) { fprintf(stderr, "wrap-spacing: --win, --patch-z >= 1, --dth, --patch-deg > 0\n"); return 2; }
    FILE *f = fopen(argv[2], "r"); if (!f) { fprintf(stderr, "wrap-spacing: cannot read %s\n", argv[2]); return 1; }
    char line[512]; if (!fgets(line, sizeof line, f) || strncmp(line, "z\ty\tx\tr_mm\tk\tnpk\trct\tgap_um\tgtq", 31)) { fprintf(stderr, "wrap-spacing: %s is not a wrap-check --intervals table\n", argv[2]); fclose(f); return 1; }
    size_t n = 0, cap = 1 << 16; ws_iv *V = malloc(cap * sizeof *V);
    while (fgets(line, sizeof line, f)) {
        ws_iv v = {0}; char g[32];
        if (sscanf(line, "%f %f %f %f %d %d %f %f %31s", &v.z, &v.y, &v.x, &v.r, &v.k, &v.npk, &v.rct, &v.gap, g) != 9) continue;
        v.gtq = strcmp(g, "nan") ? (float)atof(g) : NAN;
        if (n == cap) { cap *= 2; V = realloc(V, cap * sizeof *V); }
        V[n++] = v;
    }
    fclose(f);
    if (!n) { fprintf(stderr, "wrap-spacing: no intervals in %s\n", argv[2]); free(V); return 1; }
    /* z levels: the distinct z values, in order */
    float *zs = malloc(n * sizeof *zs); for (size_t i = 0; i < n; i++) zs[i] = V[i].z;
    qsort(zs, n, sizeof *zs, cmp_f); size_t nz = 0; for (size_t i = 0; i < n; i++) if (!nz || zs[i] != zs[nz - 1]) zs[nz++] = zs[i];
    double *cy = calloc(nz, sizeof *cy), *cx = calloc(nz, sizeof *cx); long *cn = calloc(nz, sizeof *cn);
    for (size_t i = 0; i < n; i++) {
        size_t lo = 0, hi = nz; while (lo + 1 < hi) { const size_t m = (lo + hi) / 2; if (zs[m] <= V[i].z) lo = m; else hi = m; }
        V[i].zi = (int)lo; cy[lo] += V[i].y; cx[lo] += V[i].x; cn[lo]++;
    }
    for (size_t i = 0; i < n; i++) { const int z = V[i].zi; V[i].th = atan2(V[i].y - cy[z] / cn[z], V[i].x - cx[z] / cn[z]); }
    qsort(V, n, sizeof *V, cmp_iv);
    size_t ng = 0; grp *G = malloc(n * sizeof *G);
    for (size_t i = 0; i < n;) { size_t j = i; while (j < n && V[j].zi == V[i].zi && V[j].k == V[i].k) j++; G[ng++] = (grp){V[i].zi, V[i].k, {i, j}}; i = j; }
    float nb[64];
    for (size_t gi = 0; gi < ng; gi++) {
        const grp *g = &G[gi]; const size_t a = g->s.a, b = g->s.b, m = b - a;
        for (size_t i = a; i < b; i++) {
            ws_iv *v = &V[i]; int c = 0;
            for (int d = -win; d <= win; d++) {   /* the neighbouring rays of the same pair (circular within the group) */
                if (!d || (size_t)(2 * win) >= m) continue;
                const size_t j = a + (size_t)(((long)(i - a) + d + (long)m) % (long)m);
                double dd = fabs(V[j].th - v->th); if (dd > M_PI) dd = 2 * M_PI - dd;
                if (dd <= dth * win) nb[c++] = V[j].gap;
            }
            v->st = c >= 2 ? lr(v->gap, med(nb, c)) : NAN;
            c = 0;
            for (int dz = -1; dz <= 1; dz += 2) { const long j = nearest(V, find_grp(G, ng, v->zi + dz, v->k), v->th, dth / 3); if (j >= 0) nb[c++] = V[j].gap; }
            v->sz = c ? lr(v->gap, med(nb, c)) : NAN;
            c = 0;
            for (int dk = -1; dk <= 1; dk += 2) { const long j = nearest(V, find_grp(G, ng, v->zi, v->k + dk), v->th, dth / 3); if (j >= 0) nb[c++] = V[j].gap; }
            v->sr = c ? lr(v->gap, med(nb, c)) : NAN;
            v->sct = lr(v->gap, v->rct);
        }
    }
    /* AUCs over the GT intervals */
    enum { NS = 9 };   /* (+ the raw gap both ways: does a score only restate small / large gaps?) */
    const char *nm[NS] = {"st (along, same z)", "sz (along, adjacent z)", "sr (radial, k +- 1)", "sct (gap / CT spacing)", "along = max(st, sz)", "all = max(st, sz, sr)", "npk != 1 (wrap-check CT)",
                          "gap (large)", "-gap (small)"};
    sl *S = malloc(n * sizeof *S); size_t ngt = 0; long nw = 0;
    for (size_t i = 0; i < n; i++) if (!isnan(V[i].gtq)) { ngt++; nw += fabsf(V[i].gtq - 1) >= 0.5f; }
    printf("wrap-spacing: %zu intervals, %zu z levels, %zu (z, k) groups; %zu with GT, %ld wrong (%.4f)\n", n, nz, ng, ngt, nw, ngt ? (double)nw / ngt : 0);
    double ap[NS] = {0};
    for (int s = 0; s < NS; s++) {
        size_t m = 0;
        for (size_t i = 0; i < n; i++) {
            const ws_iv *v = &V[i]; if (isnan(v->gtq)) continue;
            const float al = isnan(v->st) ? v->sz : isnan(v->sz) ? v->st : fmaxf(v->st, v->sz), all = isnan(v->sr) ? al : isnan(al) ? v->sr : fmaxf(al, v->sr);
            const float sc[NS] = {v->st, v->sz, v->sr, v->sct, al, all, (float)(v->npk != 1), v->gap, -v->gap};
            S[m++] = (sl){sc[s], fabsf(v->gtq - 1) >= 0.5f};
        }
        long p, q; ap[s] = auc(S, m, &p, &q);
        printf("  interval AUC %-26s %.4f  (wrong %ld, right %ld)\n", nm[s], ap[s], p, q);
    }
    /* patches: (k, z band, angle bin) */
    const int nb_th = (int)ceil(360 / pdeg);
    typedef struct { long key; double s[NS]; int c[NS]; int ngt, nwr; } patch;
    patch *P = calloc(n, sizeof *P); size_t np = 0;
    long *key = malloc(n * sizeof *key);
    for (size_t i = 0; i < n; i++) { const ws_iv *v = &V[i]; int t = (int)floor((v->th + M_PI) / (2 * M_PI) * nb_th); if (t >= nb_th) t = nb_th - 1; key[i] = (((long)v->k + 4096) * 4096 + v->zi / pz) * 1024 + t; }
    /* (patch per key: a sort of indices by key) */
    size_t *ix = malloc(n * sizeof *ix); for (size_t i = 0; i < n; i++) ix[i] = i;
    for (size_t gap = n / 2; gap > 0; gap /= 2) for (size_t i = gap; i < n; i++) { const size_t t = ix[i]; size_t j = i; while (j >= gap && key[ix[j - gap]] > key[t]) { ix[j] = ix[j - gap]; j -= gap; } ix[j] = t; }
    for (size_t r = 0; r < n;) {
        size_t e = r; patch *pp = &P[np++]; pp->key = key[ix[r]];
        while (e < n && key[ix[e]] == pp->key) {
            const ws_iv *v = &V[ix[e]];
            const float al = isnan(v->st) ? v->sz : isnan(v->sz) ? v->st : fmaxf(v->st, v->sz), all = isnan(v->sr) ? al : isnan(al) ? v->sr : fmaxf(al, v->sr);
            const float sc[NS] = {v->st, v->sz, v->sr, v->sct, al, all, (float)(v->npk != 1), v->gap, -v->gap};
            for (int s = 0; s < NS; s++) if (!isnan(sc[s])) { pp->s[s] += sc[s]; pp->c[s]++; }
            if (!isnan(v->gtq)) { pp->ngt++; pp->nwr += fabsf(v->gtq - 1) >= 0.5f; }
            e++;
        }
        r = e;
    }
    size_t pg = 0, pw = 0; for (size_t i = 0; i < np; i++) if (P[i].ngt >= 3) { pg++; pw += 2 * P[i].nwr > P[i].ngt; }
    printf("  patches (k x %.0f deg x %d z levels): %zu, %zu with >= 3 GT intervals, %zu of them wrong (> half)\n", pdeg, pz, np, pg, pw);
    for (int s = 0; s < NS; s++) {
        size_t m = 0;
        for (size_t i = 0; i < np; i++) if (P[i].ngt >= 3) S[m++] = (sl){P[i].c[s] ? (float)(P[i].s[s] / P[i].c[s]) : NAN, 2 * P[i].nwr > P[i].ngt};
        long p, q; const double a = auc(S, m, &p, &q);
        printf("  patch AUC    %-26s %.4f  (wrong %ld, right %ld)\n", nm[s], a, p, q);
    }
    {   /* the flag: a fitted gap below --gap-min (the measured predictor: 1667 arm C, DESIGN 2026-11-27) or an along-sheet jump above --thr */
        FILE *o = out ? fopen(out, "w") : nullptr; if (out && !o) { fprintf(stderr, "wrap-spacing: cannot write %s\n", out); return 1; }
        if (o) fprintf(o, "z\ty\tx\tk\tgap_um\tst\tsz\tsr\tsct\talong\tflag\tgtq\n");
        long fl = 0, tp = 0, fp = 0, fn = 0;
        for (size_t i = 0; i < n; i++) {
            const ws_iv *v = &V[i]; const float al = isnan(v->st) ? v->sz : isnan(v->sz) ? v->st : fmaxf(v->st, v->sz);
            const int fg = v->gap < gmin || (thr > 0 && al > thr); fl += fg;
            if (!isnan(v->gtq)) { const int w = fabsf(v->gtq - 1) >= 0.5f; tp += fg && w; fp += fg && !w; fn += !fg && w; }
            if (o) fprintf(o, "%.0f\t%.1f\t%.1f\t%d\t%.1f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%d\t%.3f\n", v->z, v->y, v->x, v->k, v->gap, v->st, v->sz, v->sr, v->sct, al, fg, v->gtq);
        }
        if (o && fclose(o)) { fprintf(stderr, "wrap-spacing: cannot write %s\n", out); return 1; }
        printf("wrap-spacing: flagged (gap < %.0f um%s) %.4f of the intervals%s%s", gmin, thr > 0 ? " or along jump > --thr" : "", (double)fl / n, out ? " -> " : "", out ? out : "");
        if (tp + fn) printf("; vs GT: recall %.3f precision %.3f (base rate %.4f)", (double)tp / (tp + fn), tp + fp ? (double)tp / (tp + fp) : 0, ngt ? (double)nw / ngt : 0);
        printf("\n");
    }
    free(V); free(zs); free(cy); free(cx); free(cn); free(G); free(S); free(P); free(key); free(ix);
    return 0;
}
