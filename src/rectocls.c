/* ufsm recto-cls train OUT DUMP[,DUMP..] [--test DUMP,..] [--holdout-z ZB] [--trees 200] [--depth 4] [--lr 0.1] [--lambda 1] [--min-leaf 40]
                     [--bins 64] [--drop F,F..] [--sub 1]
   ufsm recto-cls eval MODEL DUMP[,DUMP..]
   A transparent interval classifier for recto-ruler dumps: per interval between consecutive recto hits on a ray, how many turns it spans
   (classes 0 = a spurious double hit, 1, 2, 3+ = merged sheets), gradient-boosted depth-limited trees (softmax over the classes,
   histogram splits on per-feature quantile bins, missing values in bin 0). Train / test split: --test dumps, or z blocks of --holdout-z
   native voxels (odd blocks test). Prints confusion matrices (rows GT class, columns predicted), accuracy, turns per GT turn and the share
   off by >= 0.5 turn for the classifier (argmax and expectation) and the baselines (every interval 1 turn; rounded density; rounded
   width / median neighbour width), the feature gains. Model file: text (header RRCLS1). */
#include "rectoruler.h"
#include "fit.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int f, thr, l, r; float v; } rc_node;   /* f < 0: leaf with value v; else bin <= thr -> l */
struct rr_cls { int K, NF, nt, nb[RR_NF]; float *edge[RR_NF]; float f0[RR_K]; int *tn; rc_node **tree; };

static int rc_bin(const rr_cls *c, int f, float x) {
    if (!(x == x)) return 0;
    int lo = 0, hi = c->nb[f];   /* first edge > x */
    while (lo < hi) { const int m = (lo + hi) / 2; if (c->edge[f][m] > x) hi = m; else lo = m + 1; }
    return 1 + lo;
}
void rr_cls_predict(const rr_cls *c, const float *x, float *p) {
    double F[RR_K]; for (int k = 0; k < c->K; k++) F[k] = c->f0[k];
    for (int t = 0; t < c->nt; t++) {
        const rc_node *nd = c->tree[t]; int i = 0;
        while (nd[i].f >= 0) i = rc_bin(c, nd[i].f, x[nd[i].f]) <= nd[i].thr ? nd[i].l : nd[i].r;
        F[t % c->K] += nd[i].v;
    }
    double mx = -1e300, s = 0; for (int k = 0; k < c->K; k++) if (F[k] > mx) mx = F[k];
    for (int k = 0; k < c->K; k++) { p[k] = (float)exp(F[k] - mx); s += p[k]; }
    for (int k = 0; k < c->K; k++) p[k] = (float)(p[k] / s);
}
void rr_cls_free(rr_cls *c) {
    if (!c) return;
    for (int f = 0; f < c->NF; f++) free(c->edge[f]);
    for (int t = 0; t < c->nt; t++) free(c->tree[t]);
    free(c->tree); free(c->tn); free(c);
}
rr_cls *rr_cls_load(const char *path) {
    FILE *f = fopen(path, "r"); if (!f) return nullptr;
    rr_cls *c = calloc(1, sizeof *c); char hd[16];
    if (fscanf(f, "%15s %d %d %d", hd, &c->K, &c->NF, &c->nt) != 4 || strcmp(hd, "RRCLS1") || c->K != RR_K || c->NF != RR_NF) { fclose(f); free(c); return nullptr; }
    for (int k = 0; k < c->K; k++) if (fscanf(f, "%f", &c->f0[k]) != 1) goto bad;
    for (int ft = 0; ft < c->NF; ft++) { if (fscanf(f, "%d", &c->nb[ft]) != 1) goto bad; c->edge[ft] = malloc(sizeof(float) * (c->nb[ft] + 1)); for (int b = 0; b < c->nb[ft]; b++) if (fscanf(f, "%f", &c->edge[ft][b]) != 1) goto bad; }
    c->tn = calloc(c->nt, sizeof(int)); c->tree = calloc(c->nt, sizeof(rc_node *));
    for (int t = 0; t < c->nt; t++) { if (fscanf(f, "%d", &c->tn[t]) != 1) goto bad; c->tree[t] = malloc(sizeof(rc_node) * c->tn[t]);
        for (int i = 0; i < c->tn[t]; i++) { rc_node *n = &c->tree[t][i]; if (fscanf(f, "%d %d %d %d %f", &n->f, &n->thr, &n->l, &n->r, &n->v) != 5) goto bad; } }
    fclose(f); return c;
bad:
    fclose(f); c->nt = 0; rr_cls_free(c); return nullptr;
}

/* --- data --- */
typedef struct { float *x; float *t; float *z; long long n; } rc_data;
static int rc_load(const char *list, rc_data *d) {
    char buf[8192]; snprintf(buf, sizeof buf, "%s", list);
    for (char *p = strtok(buf, ","); p; p = strtok(nullptr, ",")) {
        FILE *f = fopen(p, "rb"); if (!f) { fprintf(stderr, "recto-cls: cannot read %s\n", p); return -1; }
        int hd[4]; if (fread(hd, 4, 4, f) != 4 || hd[0] != 0x31445252 || hd[1] != RR_NF || hd[2] != RR_REC) { fprintf(stderr, "recto-cls: %s: not a recto-ruler dump of this version\n", p); fclose(f); return -1; }
        float r[RR_REC];
        while (fread(r, sizeof(float), RR_REC, f) == RR_REC) {
            if (!(r[RR_NF] >= -0.25f)) continue;
            if ((d->n & 4095) == 0) { d->x = realloc(d->x, sizeof(float) * RR_NF * (d->n + 4096)); d->t = realloc(d->t, sizeof(float) * (d->n + 4096)); d->z = realloc(d->z, sizeof(float) * (d->n + 4096)); }
            memcpy(d->x + d->n * RR_NF, r, sizeof(float) * RR_NF); d->t[d->n] = r[RR_NF]; d->z[d->n] = r[RR_NF + 1]; d->n++;
        }
        fclose(f);
    }
    return 0;
}
static int rc_cmpf(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }
static int rc_class(float t) { return t < 0.5f ? 0 : t < 1.5f ? 1 : t < 2.5f ? 2 : 3; }

/* --- training --- */
typedef struct { const uint8_t *X; const float *g, *h; const int *idx; int n, nbin; double (*hist)[2]; const int *use; } rc_hctx;
static void rc_hist_job(void *ctx, int f0, int f1, int t) {
    (void)t; rc_hctx *H = ctx;
    for (int f = f0; f < f1; f++) {
        double (*hs)[2] = H->hist + (size_t)f * H->nbin; memset(hs, 0, sizeof(double) * 2 * H->nbin);
        if (!H->use[f]) continue;
        for (int i = 0; i < H->n; i++) { const int r = H->idx[i]; const int b = H->X[(size_t)r * RR_NF + f]; hs[b][0] += H->g[r]; hs[b][1] += H->h[r]; }
    }
}

static void rc_eval(const char *tag, const rr_cls *c, const rc_data *d, const char *mask) {
    long long cm[RR_K][RR_K] = {{0}}, n = 0; double st = 0, sa = 0, se = 0, s1 = 0, sd = 0, sw = 0; long long oa = 0, oe = 0, o1 = 0, od = 0, ow = 0, oc = 0; double sc = 0;
    long long cmd[RR_K][RR_K] = {{0}};
    for (long long i = 0; i < d->n; i++) {
        if (mask && !mask[i]) continue;
        const float *x = d->x + i * RR_NF; float p[RR_K]; int am = 0; double E = 0;
        if (c) { rr_cls_predict(c, x, p); for (int k = 0; k < RR_K; k++) { E += k * p[k]; if (p[k] > p[am]) am = k; } }
        const int y = rc_class(d->t[i]); const double t = d->t[i];
        const double dv = x[3] == x[3] ? fmin(3, fmax(0, round(x[3]))) : 1, wv = x[2] == x[2] ? fmin(3, fmax(0, round(x[2]))) : 1;
        cm[y][am]++; cmd[y][(int)dv]++; n++; st += t; sa += am; se += E; s1 += 1; sd += dv; sw += wv;
        { const double dc = x[3] == x[3] ? x[3] : 1; sc += dc; oc += fabs(dc - t) >= 0.5; } oa += fabs(am - t) >= 0.5; oe += fabs(E - t) >= 0.5; o1 += fabs(1 - t) >= 0.5; od += fabs(dv - t) >= 0.5; ow += fabs(wv - t) >= 0.5;
    }
    if (!n) { printf("  %s: no intervals\n", tag); return; }
    long long gt[RR_K] = {0}; for (int a = 0; a < RR_K; a++) for (int b = 0; b < RR_K; b++) gt[a] += cm[a][b];
    printf("  %s: %lld intervals, GT classes 0 / 1 / 2 / 3+: %.3f / %.3f / %.3f / %.3f\n", tag, n, (double)gt[0] / n, (double)gt[1] / n, (double)gt[2] / n, (double)gt[3] / n);
    printf("    turns per GT turn / off >= 0.5: density %.3f/%.3f  all-ones %.3f/%.3f  density-round %.3f/%.3f  width-round %.3f/%.3f", sc / st, (double)oc / n, s1 / st, (double)o1 / n, sd / st, (double)od / n, sw / st, (double)ow / n);
    if (c) printf("  cls %.3f/%.3f  clsE %.3f/%.3f", sa / st, (double)oa / n, se / st, (double)oe / n);
    printf("\n");
    if (c) { long long ok = 0; for (int k = 0; k < RR_K; k++) ok += cm[k][k];
        printf("    classifier accuracy %.4f; confusion (rows GT 0 1 2 3+, columns predicted):\n", (double)ok / n);
        for (int a = 0; a < RR_K; a++) { printf("      GT %d%s", a, a == 3 ? "+" : " "); for (int b = 0; b < RR_K; b++) printf(" %9lld", cm[a][b]); printf("   recall %.3f\n", gt[a] ? (double)cm[a][a] / gt[a] : 0); } }
    { long long ok = 0; for (int k = 0; k < RR_K; k++) ok += cmd[k][k];
      printf("    density-round accuracy %.4f; confusion:\n", (double)ok / n);
      for (int a = 0; a < RR_K; a++) { printf("      GT %d%s", a, a == 3 ? "+" : " "); for (int b = 0; b < RR_K; b++) printf(" %9lld", cmd[a][b]); printf("\n"); } }
}

int cmd_recto_cls(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm recto-cls train OUT DUMP[,..] [--test DUMP,..] [--holdout-z ZB] [--trees 200] [--depth 4] [--lr 0.1] [--lambda 1] [--min-leaf 40] [--bins 64] [--drop F,..]\n"
                                    "       ufsm recto-cls eval MODEL DUMP[,..]\n"); return 2; }
    const char *opt_test = nullptr; double ZB = 0, lr = 0.1, lam = 1; int ntree = 200, depth = 4, minleaf = 40, nbins = 64; int use[RR_NF]; for (int f = 0; f < RR_NF; f++) use[f] = 1;
    for (int i = 4; i + 1 < argc; i++) {
        if (!strcmp(argv[i], "--test")) opt_test = argv[++i];
        else if (!strcmp(argv[i], "--holdout-z")) ZB = atof(argv[++i]);
        else if (!strcmp(argv[i], "--trees")) ntree = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--depth")) depth = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lr")) lr = atof(argv[++i]);
        else if (!strcmp(argv[i], "--lambda")) lam = atof(argv[++i]);
        else if (!strcmp(argv[i], "--min-leaf")) minleaf = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bins")) nbins = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--drop")) { char b[512]; snprintf(b, sizeof b, "%s", argv[++i]); for (char *p = strtok(b, ","); p; p = strtok(nullptr, ",")) { const int f = atoi(p); if (f >= 0 && f < RR_NF) use[f] = 0; } }
    }
    if (nbins > 255) nbins = 255;
    if (nbins < 4) nbins = 4;
    if (depth < 1) depth = 1;
    if (depth > 8) depth = 8;
    if (!strcmp(argv[2], "eval")) {
        rr_cls *c = rr_cls_load(argv[3]); if (!c) { fprintf(stderr, "recto-cls: cannot load %s\n", argv[3]); return 1; }
        rc_data d = {0}; if (argc < 5 || rc_load(argv[4], &d)) return 1;
        rc_eval("eval", c, &d, nullptr); rr_cls_free(c); free(d.x); free(d.t); free(d.z); return 0;
    }
    if (strcmp(argv[2], "train") || argc < 5) { fprintf(stderr, "recto-cls: train OUT DUMPS | eval MODEL DUMPS\n"); return 2; }
    rc_data d = {0}, te = {0}; if (rc_load(argv[4], &d)) return 1;
    char *mtr = malloc(d.n), *mte = calloc(d.n, 1);
    for (long long i = 0; i < d.n; i++) { const int blk = ZB > 0 ? (int)floor(d.z[i] / ZB) : 0; mtr[i] = !(ZB > 0 && (blk & 1)); mte[i] = !mtr[i]; }
    if (opt_test && rc_load(opt_test, &te)) return 1;
    long long ntr = 0; for (long long i = 0; i < d.n; i++) ntr += mtr[i];
    printf("recto-cls: %lld intervals (%lld train)%s, %d trees depth %d lr %g\n", d.n, ntr, opt_test ? " + test dumps" : ZB > 0 ? " z-block holdout" : "", ntree, depth, lr);
    /* bins: quantile edges of the training rows (a sorted subsample), bin 0 = missing */
    rr_cls *c = calloc(1, sizeof *c); c->K = RR_K; c->NF = RR_NF;
    long long nsub = ntr < 400000 ? ntr : 400000; float *col = malloc(sizeof(float) * (nsub + 1));
    for (int f = 0; f < RR_NF; f++) {
        long long m = 0, seen = 0;
        for (long long i = 0; i < d.n && m < nsub; i++) { if (!mtr[i]) continue; if ((seen++ * nsub) / ntr != m) continue; const float v = d.x[i * RR_NF + f]; if (v == v) col[m++] = v; }
        qsort(col, m, sizeof(float), rc_cmpf);
        c->edge[f] = malloc(sizeof(float) * nbins); int ne = 0;
        for (int b = 1; b < nbins - 1 && m > 0; b++) { const float v = col[(long long)((double)b * m / (nbins - 1))]; if (ne == 0 || v > c->edge[f][ne - 1]) c->edge[f][ne++] = v; }
        c->nb[f] = ne;
    }
    free(col);
    const int NB = nbins + 1;
    uint8_t *X = malloc((size_t)d.n * RR_NF);
    for (long long i = 0; i < d.n; i++) for (int f = 0; f < RR_NF; f++) X[i * RR_NF + f] = (uint8_t)rc_bin(c, f, d.x[i * RR_NF + f]);
    /* priors */
    double pri[RR_K] = {0}; for (long long i = 0; i < d.n; i++) if (mtr[i]) pri[rc_class(d.t[i])] += 1;
    for (int k = 0; k < RR_K; k++) c->f0[k] = (float)log((pri[k] + 1) / (ntr + RR_K));
    float *Fs = malloc(sizeof(float) * d.n * RR_K); for (long long i = 0; i < d.n; i++) for (int k = 0; k < RR_K; k++) Fs[i * RR_K + k] = c->f0[k];
    float *g = malloc(sizeof(float) * d.n), *hh = malloc(sizeof(float) * d.n), *P = malloc(sizeof(float) * d.n * RR_K);
    int *tri = malloc(sizeof(int) * ntr); { long long m = 0; for (long long i = 0; i < d.n; i++) if (mtr[i]) tri[m++] = (int)i; }
    c->nt = ntree * RR_K; c->tn = calloc(c->nt, sizeof(int)); c->tree = calloc(c->nt, sizeof(rc_node *));
    double (*hist)[2] = malloc(sizeof(double) * 2 * (size_t)RR_NF * NB); double gain[RR_NF] = {0};
    int *idx = malloc(sizeof(int) * ntr), *tmpi = malloc(sizeof(int) * ntr);
    for (int it = 0; it < ntree; it++) {
        for (long long i = 0; i < d.n; i++) {   /* softmax */
            double mx = -1e300, s = 0; for (int k = 0; k < RR_K; k++) if (Fs[i * RR_K + k] > mx) mx = Fs[i * RR_K + k];
            for (int k = 0; k < RR_K; k++) { P[i * RR_K + k] = (float)exp(Fs[i * RR_K + k] - mx); s += P[i * RR_K + k]; }
            for (int k = 0; k < RR_K; k++) P[i * RR_K + k] = (float)(P[i * RR_K + k] / s);
        }
        for (int k = 0; k < RR_K; k++) {
            for (long long i = 0; i < d.n; i++) { const float p = P[i * RR_K + k]; g[i] = p - (rc_class(d.t[i]) == k); hh[i] = fmaxf(p * (1 - p), 1e-6f); }
            /* grow a tree depth-first by node; nodes as tasks over a shared index array */
            const int maxn = (1 << (depth + 1)); rc_node *nd = calloc(maxn, sizeof(rc_node)); int nn = 1;
            memcpy(idx, tri, sizeof(int) * ntr);
            typedef struct { int a, n, node, dep; } rc_st; rc_st stk[64]; int sp = 0; stk[sp++] = (rc_st){0, (int)ntr, 0, 0};
            while (sp) {
                const rc_st S = stk[--sp]; double G = 0, Hs = 0; for (int i = 0; i < S.n; i++) { G += g[idx[S.a + i]]; Hs += hh[idx[S.a + i]]; }
                nd[S.node].f = -1; nd[S.node].v = (float)(-lr * G / (Hs + lam));
                if (S.dep >= depth || S.n < 2 * minleaf) continue;
                rc_hctx H = {X, g, hh, idx + S.a, S.n, NB, hist, use}; fit_par(RR_NF, rc_hist_job, &H);
                double best = 0; int bf = -1, bt = -1;
                for (int f = 0; f < RR_NF; f++) { if (!use[f]) continue;
                    double gl = 0, hl = 0; const double (*hs)[2] = (const double (*)[2])(hist + (size_t)f * NB);
                    for (int b = 0; b + 1 < NB; b++) { gl += hs[b][0]; hl += hs[b][1]; if (hl < 1e-3 || Hs - hl < 1e-3) continue;
                        const double gn = gl * gl / (hl + lam) + (G - gl) * (G - gl) / (Hs - hl + lam) - G * G / (Hs + lam);
                        if (gn > best) { best = gn; bf = f; bt = b; } } }
                if (bf < 0) continue;
                int nl = 0, nr = 0; for (int i = 0; i < S.n; i++) { const int r = idx[S.a + i]; if (X[(size_t)r * RR_NF + bf] <= bt) idx[S.a + nl++] = r; else tmpi[nr++] = r; }
                if (nl < minleaf || nr < minleaf) { memcpy(idx + S.a + nl, tmpi, sizeof(int) * nr); continue; }
                memcpy(idx + S.a + nl, tmpi, sizeof(int) * nr);
                gain[bf] += best;
                nd[S.node].f = bf; nd[S.node].thr = bt; nd[S.node].l = nn; nd[S.node].r = nn + 1; nn += 2;
                stk[sp++] = (rc_st){S.a, nl, nd[S.node].l, S.dep + 1}; stk[sp++] = (rc_st){S.a + nl, nr, nd[S.node].r, S.dep + 1};
            }
            const int t = it * RR_K + k; c->tree[t] = nd; c->tn[t] = nn;
            for (long long i = 0; i < d.n; i++) { int q = 0; while (nd[q].f >= 0) q = X[i * RR_NF + nd[q].f] <= nd[q].thr ? nd[q].l : nd[q].r; Fs[i * RR_K + k] += nd[q].v; }
        }
        if ((it + 1) % 50 == 0 || it + 1 == ntree) {
            double ll = 0; long long nte = 0; double llt = 0;
            for (long long i = 0; i < d.n; i++) { double mx = -1e300, s = 0; for (int k = 0; k < RR_K; k++) if (Fs[i * RR_K + k] > mx) mx = Fs[i * RR_K + k];
                for (int k = 0; k < RR_K; k++) s += exp(Fs[i * RR_K + k] - mx); const double lp = Fs[i * RR_K + rc_class(d.t[i])] - mx - log(s);
                if (mtr[i]) ll -= lp; else { llt -= lp; nte++; } }
            fprintf(stderr, "recto-cls: tree %d: train logloss %.4f%s", it + 1, ll / ntr, nte ? "" : "\n"); if (nte) fprintf(stderr, " holdout %.4f\n", llt / nte);
        }
    }
    {   FILE *f = fopen(argv[3], "w"); if (!f) { fprintf(stderr, "recto-cls: cannot write %s\n", argv[3]); return 1; }
        fprintf(f, "RRCLS1 %d %d %d\n", RR_K, RR_NF, c->nt); for (int k = 0; k < RR_K; k++) fprintf(f, "%.9g ", c->f0[k]); fprintf(f, "\n");
        for (int ft = 0; ft < RR_NF; ft++) { fprintf(f, "%d", c->nb[ft]); for (int b = 0; b < c->nb[ft]; b++) fprintf(f, " %.9g", c->edge[ft][b]); fprintf(f, "\n"); }
        for (int t = 0; t < c->nt; t++) { fprintf(f, "%d\n", c->tn[t]); for (int i = 0; i < c->tn[t]; i++) fprintf(f, "%d %d %d %d %.9g\n", c->tree[t][i].f, c->tree[t][i].thr, c->tree[t][i].l, c->tree[t][i].r, c->tree[t][i].v); }
        fclose(f); }
    rr_cls *cl = rr_cls_load(argv[3]); if (!cl) { fprintf(stderr, "recto-cls: reload of %s failed\n", argv[3]); return 1; }
    rc_eval("train", cl, &d, mtr);
    if (ZB > 0) rc_eval("holdout z blocks", cl, &d, mte);
    if (opt_test) rc_eval("test dumps", cl, &te, nullptr);
    {   double gs = 0; for (int f = 0; f < RR_NF; f++) gs += gain[f];
        printf("  feature gain shares:"); for (int f = 0; f < RR_NF; f++) if (gain[f] > 0.005 * gs) printf(" f%d %.3f", f, gain[f] / gs); printf("\n"); }
    rr_cls_free(cl); rr_cls_free(c); free(X); free(Fs); free(g); free(hh); free(P); free(tri); free(hist); free(idx); free(tmpi); free(mtr); free(mte);
    free(d.x); free(d.t); free(d.z); free(te.x); free(te.t); free(te.z);
    return 0;
}
