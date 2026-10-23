/* ufsm reg inkstat <render-dir>... [-o out.json] : model-free ink separability of multi-scan renders (reg render with "labels").
   Each render dir holds samples.bin (labelled pixels, per variant / scan / layer raw values) and config.json (rgb / diff / baseline names).
   Per segment, values are normalised per variant, scan and layer by the 1st / 99th percentiles of the sampled raw values; channels are the
   energies (config rgb), the energy differences (config diff, a - b of normalised values) and the same-energy baseline difference.
   Reported per segment and pooled (all segments' samples), per variant and channel:
     sheet band (layers -2 .. 2 mean): AUC = P(ink > no-ink) and Cohen's d (ink minus no-ink, pooled sd);
     the depth profile (AUC per layer) and the best layer window (widths 1 3 5 9 15, max |AUC - 0.5|);
   and leave-one-segment-out linear discriminants: LDA on the energies x 5 depth bins (15 features) and on the 3 energies at the sheet
   band, against the best single (channel, depth bin) feature chosen on the training segments, all scored by AUC on the held-out one. */
#include "json.h"
#include <pthread.h>
#include <stdatomic.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_all;   /* --all: also the uncertified samples */
typedef struct { char dir[1024], name[256]; int ns, nl, d0; char scan[8][64]; size_t n; uint8_t *rec; size_t rl; float *nrm; /* [n][2][ns][nl], NAN missing */ } iseg;
enum { MAXCHN = 8 };
typedef struct { char name[96]; int a, b; int kind; } ichan;   /* kind 0 energy (a), 1 diff a - b, 2 baseline a - b */

static char *slurp(const char *p, size_t *n) { FILE *f = fopen(p, "rb"); if (!f) return nullptr; fseek(f, 0, SEEK_END); long L = ftell(f); fseek(f, 0, SEEK_SET); char *b = malloc(L + 1); if (fread(b, 1, L, f) != (size_t)L) { fclose(f); free(b); return nullptr; } b[L] = 0; fclose(f); if (n) *n = L; return b; }

static int seg_load(iseg *g, const char *dir) {
    memset(g, 0, sizeof *g); snprintf(g->dir, sizeof g->dir, "%s", dir);
    {   /* the segment: the render dir's parent when the dir itself is a render variant (labelled4, crop_*) */
        char t2[1024]; snprintf(t2, sizeof t2, "%s", dir); size_t l = strlen(t2); while (l > 1 && t2[l - 1] == '/') t2[--l] = 0;
        char *b = strrchr(t2, '/'); const char *last = b ? b + 1 : t2;
        if (b && (!strncmp(last, "labelled", 8) || !strncmp(last, "crop", 4) || !strcmp(last, "full"))) { *b = 0; char *b2 = strrchr(t2, '/'); snprintf(g->name, sizeof g->name, "%s", b2 ? b2 + 1 : t2); }
        else snprintf(g->name, sizeof g->name, "%s", last);
    }
    char p[1200]; snprintf(p, sizeof p, "%s/samples.bin", dir);
    size_t n; char *t = slurp(p, &n); if (!t) { fprintf(stderr, "reg inkstat: cannot read %s\n", p); return -1; }
    char *nl = memchr(t, '\n', n); if (!nl) { free(t); return -1; }
    json *h = json_parse(t, nl - t); if (!h) { free(t); return -1; }
    const json *sc = json_get(h, "scans"); g->ns = (int)sc->n;
    for (int s = 0; s < g->ns; s++) snprintf(g->scan[s], 64, "%s", json_str(json_at(sc, s), ""));
    g->d0 = (int)json_num(json_at(json_get(h, "depth"), 0), -10); g->nl = (int)json_num(json_at(json_get(h, "depth"), 1), 40) - g->d0 + 1;
    g->n = (size_t)json_num(json_get(h, "samples"), 0); json_free(h);
    g->rl = 1 + g->ns + (size_t)2 * g->ns * g->nl;
    if ((size_t)(n - (nl + 1 - t)) < g->n * g->rl) { fprintf(stderr, "reg inkstat: %s truncated\n", p); free(t); return -1; }
    g->rec = malloc(g->n * g->rl); memcpy(g->rec, nl + 1, g->n * g->rl); free(t);
    /* normalisation per variant, scan, layer */
    g->nrm = malloc(g->n * 2 * g->ns * g->nl * sizeof(float));
    for (int v = 0; v < 2; v++) for (int s = 0; s < g->ns; s++) for (int l = 0; l < g->nl; l++) {
        size_t h2[256] = {0}, tot = 0; const size_t off = 1 + g->ns + ((size_t)v * g->ns + s) * g->nl + l;
        for (size_t i = 0; i < g->n; i++) { const uint8_t x = g->rec[i * g->rl + off]; if (x && (g_all || g->rec[i * g->rl + 1 + s])) { h2[x]++; tot++; } }
        double lo = 0, hi = 255; size_t c = 0; int k = 1;
        if (tot) { for (; k < 256 && c + h2[k] <= 0.01 * tot; k++) c += h2[k]; lo = k; c = 0; for (k = 1; k < 256 && c + h2[k] < 0.99 * tot; k++) c += h2[k]; hi = k; if (hi <= lo) hi = lo + 1; }
        for (size_t i = 0; i < g->n; i++) { const uint8_t x = g->rec[i * g->rl + off]; g->nrm[((i * 2 + v) * g->ns + s) * g->nl + l] = x && (g_all || g->rec[i * g->rl + 1 + s]) ? (float)((x - lo) / (hi - lo)) : NAN; }
    }
    return 0;
}
static inline int cls_of(const iseg *g, size_t i) { return g->rec[i * g->rl]; }
static inline float nv(const iseg *g, size_t i, int v, int s, int l) { return g->nrm[((i * 2 + v) * g->ns + s) * g->nl + l]; }
static float chv(const iseg *g, size_t i, int v, const ichan *c, int l0, int l1) {   /* band mean of a channel over layers l0 .. l1 (indices) */
    double a = 0;
    for (int l = l0; l <= l1; l++) {
        const float x = nv(g, i, v, c->a, l); if (isnan(x)) return NAN;
        if (c->kind) { const float y = nv(g, i, v, c->b, l); if (isnan(y)) return NAN; a += x - y; } else a += x;
    }
    return (float)(a / (l1 - l0 + 1));
}

/* AUC = P(ink > no-ink) (ties half) and Cohen's d from (value, class) pairs */
typedef struct { float v; uint8_t c; } vc;
static int vc_cmp(const void *a, const void *b) { const float x = ((const vc *)a)->v, y = ((const vc *)b)->v; return x < y ? -1 : x > y; }
static void metrics(vc *a, size_t n, double *auc, double *d, size_t *n1, size_t *n2) {
    double s1 = 0, s2 = 0, q1 = 0, q2 = 0; size_t c1 = 0, c2 = 0;
    for (size_t i = 0; i < n; i++) { if (a[i].c == 2) { s2 += a[i].v; q2 += (double)a[i].v * a[i].v; c2++; } else { s1 += a[i].v; q1 += (double)a[i].v * a[i].v; c1++; } }
    *n1 = c1; *n2 = c2; *auc = NAN; *d = NAN;
    if (c1 < 10 || c2 < 10) return;
    const double m1 = s1 / c1, m2 = s2 / c2, v1 = q1 / c1 - m1 * m1, v2 = q2 / c2 - m2 * m2, sp = sqrt(((c1 - 1) * v1 + (c2 - 1) * v2) / (c1 + c2 - 2));
    *d = sp > 0 ? (m2 - m1) / sp : 0;
    qsort(a, n, sizeof *a, vc_cmp);
    double rs = 0; size_t i = 0;
    while (i < n) { size_t j = i; while (j < n && a[j].v == a[i].v) j++; const double r = (i + 1 + j) / 2.0; for (size_t k = i; k < j; k++) if (a[k].c == 2) rs += r; i = j; }
    *auc = (rs - c2 * (c2 + 1) / 2.0) / ((double)c1 * c2);
}
/* metrics of channel c over layers [l0, l1] on segments segs (mask) */
static void ch_metrics(iseg *G, int ng, const int *use, int v, const ichan *c, int l0, int l1, double *auc, double *d, size_t *n1, size_t *n2) {
    size_t tot = 0; for (int k = 0; k < ng; k++) if (use[k]) tot += G[k].n;
    vc *a = malloc((tot + 1) * sizeof *a); size_t n = 0;
    for (int k = 0; k < ng; k++) if (use[k]) for (size_t i = 0; i < G[k].n; i++) { const float x = chv(&G[k], i, v, c, l0, l1); if (!isnan(x)) { a[n].v = x; a[n].c = (uint8_t)cls_of(&G[k], i); n++; } }
    metrics(a, n, auc, d, n1, n2); free(a);
}

/* LDA: w = (S + ridge)^-1 (mu_ink - mu_no) on rows with all features present */
static int solve(int n, double *A, double *b) {
    for (int c = 0; c < n; c++) { int p = c; for (int r = c + 1; r < n; r++) if (fabs(A[r * n + c]) > fabs(A[p * n + c])) p = r; if (fabs(A[p * n + c]) < 1e-300) return -1;
        if (p != c) { for (int k = 0; k < n; k++) { const double t = A[c * n + k]; A[c * n + k] = A[p * n + k]; A[p * n + k] = t; } const double t = b[c]; b[c] = b[p]; b[p] = t; }
        for (int r = c + 1; r < n; r++) { const double f = A[r * n + c] / A[c * n + c]; for (int k = c; k < n; k++) A[r * n + k] -= f * A[c * n + k]; b[r] -= f * b[c]; } }
    for (int c = n - 1; c >= 0; c--) { double s = b[c]; for (int k = c + 1; k < n; k++) s -= A[c * n + k] * b[k]; b[c] = s / A[c * n + c]; }
    return 0;
}
typedef struct { int nf; int s[32], kind[32], b2[32], l0[32], l1[32]; } fset;   /* feature: channel (scan s / diff with b2) over layers */
static int feats(const iseg *g, size_t i, int v, const fset *F, double *x) {
    for (int f = 0; f < F->nf; f++) { const ichan c = {"", F->s[f], F->b2[f], F->kind[f]}; const float y = chv(g, i, v, &c, F->l0[f], F->l1[f]); if (isnan(y)) return 0; x[f] = y; }
    return 1;
}
static double lda_auc2(iseg *G, int ng, int test, int v, const fset *F) {
    const int m = F->nf; double mu[2][32] = {{0}}, S[32 * 32] = {0}, x[32]; size_t cnt[2] = {0};
    if (m < 1) return NAN;
    for (int k = 0; k < ng; k++) if (k != test) for (size_t i = 0; i < G[k].n; i++) { if (!feats(&G[k], i, v, F, x)) continue; const int c = cls_of(&G[k], i) == 2; for (int f = 0; f < m; f++) mu[c][f] += x[f]; cnt[c]++; }
    if (cnt[0] < 20 || cnt[1] < 20) return NAN;
    for (int c = 0; c < 2; c++) for (int f = 0; f < m; f++) mu[c][f] /= cnt[c];
    for (int k = 0; k < ng; k++) if (k != test) for (size_t i = 0; i < G[k].n; i++) { if (!feats(&G[k], i, v, F, x)) continue; const int c = cls_of(&G[k], i) == 2;
        for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) S[a * m + b] += (x[a] - mu[c][a]) * (x[b] - mu[c][b]); }
    double tr = 0; for (int a = 0; a < m; a++) tr += S[a * m + a];
    for (int a = 0; a < m; a++) S[a * m + a] += 1e-3 * tr / m;
    double w[32]; for (int f = 0; f < m; f++) w[f] = mu[1][f] - mu[0][f];
    if (solve(m, S, w)) return NAN;
    vc *a = malloc((G[test].n + 1) * sizeof *a); size_t n = 0;
    for (size_t i = 0; i < G[test].n; i++) { if (!feats(&G[test], i, v, F, x)) continue; double s = 0; for (int f = 0; f < m; f++) s += w[f] * x[f]; a[n].v = (float)s; a[n].c = (uint8_t)cls_of(&G[test], i); n++; }
    double auc, d; size_t n1, n2; metrics(a, n, &auc, &d, &n1, &n2); free(a);
    return auc;
}

/* ---------- depth composites (modes) ---------- */
enum { M_MEAN, M_MAX, M_TOP3, M_P90, M_ARGMAX, E_COH, D_MAX, D_MIN, D_OFMAX, D_COH, LR_MAX, LR_P90, M_N };
static const char *const mode_name[M_N] = {"mean", "max", "top3-mean", "p90", "depth-of-peak", "at-78keV-peak", "max-of-layer-diff", "min-of-layer-diff", "diff-of-max",
                                           "diff-at-78keV-peak", "max-log-ratio", "p90-log-ratio"};
typedef struct { int op, a, b, l0, l1, v; char name[160]; } mspec;
static inline int rawv(const iseg *g, size_t i, int v, int s, int l) {
    if (!g_all && !g->rec[i * g->rl + 1 + s]) return 0;
    return g->rec[i * g->rl + 1 + g->ns + ((size_t)v * g->ns + s) * g->nl + l];
}
static int f_cmp(const void *x, const void *y) { const float a = *(const float *)x, b = *(const float *)y; return a < b ? -1 : a > b; }
static float stat(float *x, int n, int op) {   /* x is reordered; n >= 1 */
    if (n < 1) return NAN;
    if (op == M_MEAN) { double s = 0; for (int i = 0; i < n; i++) s += x[i]; return (float)(s / n); }
    if (op == M_MAX) { float m = x[0]; for (int i = 1; i < n; i++) if (x[i] > m) m = x[i]; return m; }
    qsort(x, n, sizeof *x, f_cmp);
    if (op == M_TOP3) { const int k = n < 3 ? n : 3; double s = 0; for (int i = n - k; i < n; i++) s += x[i]; return (float)(s / k); }
    return x[(int)floor(0.9 * (n - 1))];   /* p90 */
}
static float mode_value(const iseg *g, size_t i, const mspec *m, int base) {
    const int n = m->l1 - m->l0 + 1; float x[128];
    switch (m->op) {
    case M_MEAN: case M_MAX: case M_TOP3: case M_P90: case M_ARGMAX: {
        int am = 0;
        for (int l = 0; l < n; l++) { x[l] = nv(g, i, m->v, m->a, m->l0 + l); if (isnan(x[l])) return NAN; if (x[l] > x[am]) am = l; }
        if (m->op == M_ARGMAX) return (float)(m->l0 + am + g->d0);
        return stat(x, n, m->op);
    }
    case E_COH: case D_COH: {
        int am = -1; float bm = -1e30f;
        for (int l = 0; l < n; l++) { const float y = nv(g, i, m->v, base, m->l0 + l); if (isnan(y)) return NAN; if (y > bm) { bm = y; am = l; } }
        const float a = nv(g, i, m->v, m->a, m->l0 + am); if (m->op == E_COH) return a;
        const float b = nv(g, i, m->v, m->b, m->l0 + am); return isnan(a) || isnan(b) ? NAN : a - b;
    }
    case D_MAX: case D_MIN: {
        float r = m->op == D_MAX ? -1e30f : 1e30f;
        for (int l = 0; l < n; l++) { const float a = nv(g, i, m->v, m->a, m->l0 + l), b = nv(g, i, m->v, m->b, m->l0 + l); if (isnan(a) || isnan(b)) return NAN; const float d = a - b; if (m->op == D_MAX ? d > r : d < r) r = d; }
        return r;
    }
    case D_OFMAX: {
        float ma = -1e30f, mb = -1e30f;
        for (int l = 0; l < n; l++) { const float a = nv(g, i, m->v, m->a, m->l0 + l), b = nv(g, i, m->v, m->b, m->l0 + l); if (isnan(a) || isnan(b)) return NAN; if (a > ma) ma = a; if (b > mb) mb = b; }
        return ma - mb;
    }
    case LR_MAX: case LR_P90: {
        for (int l = 0; l < n; l++) { const int a = rawv(g, i, m->v, m->a, m->l0 + l), b = rawv(g, i, m->v, m->b, m->l0 + l); if (!a || !b) return NAN; x[l] = logf((float)a) - logf((float)b); }
        return stat(x, n, m->op == LR_MAX ? M_MAX : M_P90);
    }
    }
    return NAN;
}
static void mode_metrics(iseg *G, int ng, const int *use, const mspec *m, int base, double *auc, double *d, size_t *n1, size_t *n2) {
    size_t tot = 0; for (int k = 0; k < ng; k++) if (use[k]) tot += G[k].n;
    vc *a = malloc((tot + 1) * sizeof *a); size_t n = 0;
    for (int k = 0; k < ng; k++) if (use[k]) for (size_t i = 0; i < G[k].n; i++) { const float x = mode_value(&G[k], i, m, base); if (!isnan(x)) { a[n].v = x; a[n].c = (uint8_t)cls_of(&G[k], i); n++; } }
    metrics(a, n, auc, d, n1, n2); free(a);
}
typedef struct { int k; double sep; } rk;
/* modes evaluated on threads */
typedef struct { iseg *G; int ng; const int *use; const mspec *M; int nm, base; double *A, *D; size_t *N2; atomic_int next; } mjob;
static void *mode_worker(void *arg) {
    mjob *q = arg;
    for (;;) { const int m = atomic_fetch_add(&q->next, 1); if (m >= q->nm) break; size_t n1; mode_metrics(q->G, q->ng, q->use, &q->M[m], q->base, &q->A[m], &q->D[m], &n1, &q->N2[m]); }
    return nullptr;
}
static int rk_cmp(const void *x, const void *y) { const double a = ((const rk *)x)->sep, b = ((const rk *)y)->sep; return a > b ? -1 : a < b; }

int cmd_reg_inkstat(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm reg inkstat <render-dir>... [-o out.json] [--all (uncertified samples too)]\n"); return 2; }
    const char *out = nullptr; int ng = 0; iseg *G = calloc(argc, sizeof *G);
    for (int i = 3; i < argc; i++) if (!strcmp(argv[i], "--all")) g_all = 1;
    for (int i = 3; i < argc; i++) { if (!strcmp(argv[i], "--all")) continue; if (!strcmp(argv[i], "-o") && i + 1 < argc) { out = argv[++i]; continue; } if (!seg_load(&G[ng], argv[i])) ng++; }
    if (!ng) return 1;
    /* channels from the first dir's config */
    char p[1200]; snprintf(p, sizeof p, "%s/config.json", G[0].dir); size_t n; char *t = slurp(p, &n); json *cfg = t ? json_parse(t, n) : nullptr; free(t);
    if (!cfg) { fprintf(stderr, "reg inkstat: no config.json in %s\n", G[0].dir); return 1; }
    ichan C[MAXCHN]; int nc = 0;
    #define SIDX(nm) ({ int _r = -1; for (int _s = 0; _s < G[0].ns; _s++) if (!strcmp(G[0].scan[_s], nm)) _r = _s; _r; })
    const json *rgb = json_get(cfg, "rgb"), *dif = json_get(cfg, "diff"), *bl = json_get(cfg, "baseline");
    for (size_t k = 0; rgb && k < rgb->n; k++) { const int a = SIDX(json_str(json_at(rgb, k), "")); if (a >= 0) { C[nc] = (ichan){"", a, -1, 0}; snprintf(C[nc].name, 96, "%s", G[0].scan[a]); nc++; } }
    for (size_t k = 0; dif && k < dif->n; k++) { const int a = SIDX(json_str(json_at(json_at(dif, k), 0), "")), b = SIDX(json_str(json_at(json_at(dif, k), 1), ""));
        if (a >= 0 && b >= 0) { C[nc] = (ichan){"", a, b, 1}; snprintf(C[nc].name, 96, "%s - %s", G[0].scan[a], G[0].scan[b]); nc++; } }
    if (bl && bl->n == 2) { const int a = SIDX(json_str(json_at(bl, 0), "")), b = SIDX(json_str(json_at(bl, 1), "")); if (a >= 0 && b >= 0) { C[nc] = (ichan){"", a, b, 2}; snprintf(C[nc].name, 96, "baseline %s - %s", G[0].scan[a], G[0].scan[b]); nc++; } }
    const int d0 = G[0].d0, nl = G[0].nl, s0 = -2 - d0, s1 = 2 - d0;
    const char *vn[2] = {"native", "matched"};
    FILE *o = out ? fopen(out, "w") : nullptr;
    if (o) fprintf(o, "{\"segments\":[");
    int *use = calloc(ng + 1, sizeof *use);
    printf("%s samples; ", g_all ? "all" : "certified");
    printf("model-free ink separability (AUC = P(ink > no-ink); d = Cohen's d ink - no-ink); sheet band = layers -2..2 (2.4 um layers along +N)\n");
    for (int k = 0; k <= ng; k++) {   /* each segment, then pooled */
        for (int q = 0; q < ng; q++) use[q] = k == ng || q == k;
        const char *nm = k < ng ? G[k].name : "POOLED";
        if (k == ng && ng < 2) break;
        if (o) fprintf(o, "%s{\"segment\":\"%s\",\"channels\":[", k ? "," : "", nm);
        printf("\n== %s\n%-8s %-34s %8s %7s %7s | %-14s %7s %7s | %s\n", nm, "variant", "channel", "n_ink", "AUC", "d", "best window", "AUC", "d", "depth profile AUC (every 5th layer from -10)");
        for (int v = 0; v < 2; v++) for (int c = 0; c < nc; c++) {
            double auc, d; size_t n1, n2; ch_metrics(G, ng, use, v, &C[c], s0, s1, &auc, &d, &n1, &n2);
            double prof[128]; for (int l = 0; l < nl && l < 128; l++) { double a2, d2; size_t m1, m2; ch_metrics(G, ng, use, v, &C[c], l, l, &a2, &d2, &m1, &m2); prof[l] = a2; }
            const int ws[5] = {1, 3, 5, 9, 15}; double ba = 0.5, bd = 0; int bl0 = 0, bl1 = 0;
            for (int wi = 0; wi < 5; wi++) for (int l = 0; l + ws[wi] <= nl; l++) {
                double a2, d2; size_t m1, m2; ch_metrics(G, ng, use, v, &C[c], l, l + ws[wi] - 1, &a2, &d2, &m1, &m2);
                if (!isnan(a2) && fabs(a2 - 0.5) > fabs(ba - 0.5) + 1e-9) { ba = a2; bd = d2; bl0 = l; bl1 = l + ws[wi] - 1; }
            }
            char win[32]; snprintf(win, sizeof win, "[%+d,%+d]", bl0 + d0, bl1 + d0);
            printf("%-8s %-34s %8zu %7.3f %+7.3f | %-14s %7.3f %+7.3f |", vn[v], C[c].name, n2, auc, d, win, ba, bd);
            for (int l = 0; l < nl; l += 5) printf(" %.2f", prof[l]);
            printf("\n");
            if (o) { fprintf(o, "%s{\"variant\":\"%s\",\"channel\":\"%s\",\"n_ink\":%zu,\"n_noink\":%zu,\"sheet_auc\":%.4f,\"sheet_d\":%.4f,\"best_window\":[%d,%d],\"best_auc\":%.4f,\"best_d\":%.4f,\"profile_auc\":[",
                             v || c ? "," : "", vn[v], C[c].name, n2, n1, auc, d, bl0 + d0, bl1 + d0, ba, bd);
                     for (int l = 0; l < nl; l++) fprintf(o, "%s%.4f", l ? "," : "", prof[l]);
                     fprintf(o, "]}"); }
        }
        if (o) fprintf(o, "]}");
    }
    if (o) fprintf(o, "]");
    /* leave-one-segment-out discriminants */
    if (ng >= 2) {
        printf("\n== leave-one-segment-out (train on the others, AUC on the held-out segment)\n%-8s %-28s %10s %10s %10s %s\n", "variant", "held out", "LDA 15", "LDA 3", "best 1", "(best single: chosen on train)");
        if (o) fprintf(o, ",\"loso\":[");
        const int bins[5][2] = {{-10, -1}, {0, 9}, {10, 19}, {20, 29}, {30, 40}};
        int ne = 0, es[8]; for (int c = 0; c < nc; c++) if (!C[c].kind) es[ne++] = C[c].a;
        fset F15 = {0}, F3 = {0};
        for (int e = 0; e < ne; e++) for (int b = 0; b < 5; b++) { const int l0 = bins[b][0] - d0, l1 = bins[b][1] - d0; if (l0 < 0 || l1 >= nl) continue; const int f = F15.nf++; F15.s[f] = es[e]; F15.kind[f] = 0; F15.b2[f] = -1; F15.l0[f] = l0; F15.l1[f] = l1; }
        for (int e = 0; e < ne; e++) { const int f = F3.nf++; F3.s[f] = es[e]; F3.kind[f] = 0; F3.b2[f] = -1; F3.l0[f] = s0; F3.l1[f] = s1; }
        int first = 1;
        for (int v = 0; v < 2; v++) for (int k = 0; k < ng; k++) {
            const double a15 = lda_auc2(G, ng, k, v, &F15), a3 = lda_auc2(G, ng, k, v, &F3);
            /* best single (channel incl. diffs, depth bin) on the training segments */
            for (int q = 0; q < ng; q++) use[q] = q != k;
            double best = 0.5; int bc = -1, bb = -1;
            for (int c = 0; c < nc; c++) { if (C[c].kind == 2) continue; for (int b = 0; b < 5; b++) { const int l0 = bins[b][0] - d0, l1 = bins[b][1] - d0; if (l0 < 0 || l1 >= nl) continue;
                double a2, d2; size_t m1, m2; ch_metrics(G, ng, use, v, &C[c], l0, l1, &a2, &d2, &m1, &m2); if (!isnan(a2) && fabs(a2 - 0.5) > fabs(best - 0.5)) { best = a2; bc = c; bb = b; } } }
            double at = NAN;
            if (bc >= 0) { for (int q = 0; q < ng; q++) use[q] = q == k; double a2, d2; size_t m1, m2; ch_metrics(G, ng, use, v, &C[bc], bins[bb][0] - d0, bins[bb][1] - d0, &a2, &d2, &m1, &m2); at = best >= 0.5 ? a2 : 1 - a2; }
            printf("%-8s %-28s %10.3f %10.3f %10.3f (%s, layers %d..%d)\n", vn[v], G[k].name, a15, a3, at, bc >= 0 ? C[bc].name : "-", bc >= 0 ? bins[bb][0] : 0, bc >= 0 ? bins[bb][1] : 0);
            if (o) fprintf(o, "%s{\"variant\":\"%s\",\"held_out\":\"%s\",\"lda15_auc\":%.4f,\"lda3_auc\":%.4f,\"best_single_auc\":%.4f,\"best_single\":\"%s\",\"best_single_layers\":[%d,%d]}", first ? "" : ",", vn[v], G[k].name, a15, a3, at,
                           bc >= 0 ? C[bc].name : "", bc >= 0 ? bins[bb][0] : 0, bc >= 0 ? bins[bb][1] : 0);
            first = 0;
        }
        if (o) fprintf(o, "]");
    }
    /* depth composites: every mode x channel x window x variant, per segment and pooled, ranked by |AUC - 0.5| */
    {
        const int wins[3][2] = {{-4, 17}, {-10, 10}, {4, 17}}; const char *wn[3] = {"+N -10..+40um", "both +-24um", "off-sheet +10..+40um"};
        int ne = 0, es[8]; for (int c = 0; c < nc; c++) if (!C[c].kind) es[ne++] = C[c].a;
        const int base = ne >= 2 ? es[1] : 0;   /* the rgb G scan (78 keV) */
        int np = 0, pa[8], pb[8], pk[8]; for (int c = 0; c < nc; c++) if (C[c].kind) { pa[np] = C[c].a; pb[np] = C[c].b; pk[np] = C[c].kind; np++; }
        mspec *M = malloc(4096 * sizeof *M); int nm = 0;
        for (int v = 0; v < 2; v++) for (int w = 0; w < 3; w++) {
            const int l0 = wins[w][0] - d0, l1 = wins[w][1] - d0; if (l0 < 0 || l1 >= nl) continue;
            for (int e = 0; e < ne; e++) for (int op = M_MEAN; op <= E_COH; op++) {
                if (op == E_COH && es[e] == base) continue;
                M[nm] = (mspec){op, es[e], -1, l0, l1, v, ""}; snprintf(M[nm].name, 160, "%s %s %s [%s]", vn[v], G[0].scan[es[e]], mode_name[op], wn[w]); nm++;
            }
            for (int q = 0; q < np; q++) for (int op = D_MAX; op <= LR_P90; op++) {
                M[nm] = (mspec){op, pa[q], pb[q], l0, l1, v, ""}; snprintf(M[nm].name, 160, "%s %s%s - %s %s [%s]", vn[v], pk[q] == 2 ? "BASELINE " : "", G[0].scan[pa[q]], G[0].scan[pb[q]], mode_name[op], wn[w]); nm++;
            }
        }
        if (o) fprintf(o, ",\"modes\":[");
        for (int k = 0; k <= ng; k++) {
            if (k == ng && ng < 2) break;
            for (int q = 0; q < ng; q++) use[q] = k == ng || q == k;
            const char *nm2 = k < ng ? G[k].name : "POOLED";
            rk *r = malloc(nm * sizeof *r); double *A = malloc(nm * sizeof *A), *Dd = malloc(nm * sizeof *Dd); size_t *N2 = malloc(nm * sizeof *N2);
            { mjob q = {.G = G, .ng = ng, .use = use, .M = M, .nm = nm, .base = base, .A = A, .D = Dd, .N2 = N2}; atomic_store(&q.next, 0); pthread_t th[16];
              for (int t = 0; t < 12; t++) pthread_create(&th[t], nullptr, mode_worker, &q);
              for (int t = 0; t < 12; t++) pthread_join(th[t], nullptr); }
            for (int m = 0; m < nm; m++) { r[m].k = m; r[m].sep = isnan(A[m]) ? 0 : fabs(A[m] - 0.5); }
            qsort(r, nm, sizeof *r, rk_cmp);
            printf("\n== depth composites, %s: top 15 of %d (AUC = P(ink > no-ink); separability |AUC - 0.5|)\n", nm2, nm);
            for (int t2 = 0; t2 < 15 && t2 < nm; t2++) { const int m = r[t2].k; printf("%2d  AUC %.3f  d %+.3f  n_ink %7zu  %s\n", t2 + 1, A[m], Dd[m], N2[m], M[m].name); }
            /* the same-energy baseline: its best mode, for reference */
            for (int t2 = 0; t2 < nm; t2++) { const int m = r[t2].k; if (strstr(M[m].name, "BASELINE")) { printf("    best BASELINE mode (rank %d): AUC %.3f d %+.3f  %s\n", t2 + 1, A[m], Dd[m], M[m].name); break; } }
            if (o) { fprintf(o, "%s{\"segment\":\"%s\",\"ranked\":[", k ? "," : "", nm2);
                     for (int t2 = 0; t2 < nm; t2++) { const int m = r[t2].k; fprintf(o, "%s{\"mode\":\"%s\",\"auc\":%.4f,\"d\":%.4f,\"n_ink\":%zu}", t2 ? "," : "", M[m].name, A[m], Dd[m], N2[m]); }
                     fprintf(o, "]}"); }
            free(r); free(A); free(Dd); free(N2);
        }
        if (o) fprintf(o, "]");
        free(M);
    }
    if (o) { fprintf(o, "}\n"); fclose(o); }
    return 0;
}
