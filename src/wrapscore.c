/* ufsm ink-wrapscore WRAPS SEGS.JSON OUT --ct-key K --maps name=DIR[,name=DIR ..] [--ref NAME] [--nonzero a,b] [--split heldout,hold,valonly]
       [--tol 40] [--step 8] [--holdouts H.json] [--um-label U] [--table FILE] [--need w035-w051] [--min-n 1000] [--draws 1000] [--threads 6]
   (2026-11-20, fullvram; the scorer of scripts/inkgt/wrapscore.py + inksdk's pairwrap2.py in C, lead: "an ink AUC per checkpoint in
   < 10 min, a hard failure on any missing input"). Several ink maps rendered on one unroll's wraps (DIR/<w>.tif or .jpg, any size
   proportional to the wrap canvas) scored on the same held-out labelled samples, paired against --ref.

   The samples are wrapscore.py's: each published segment of SEGS (the ink-onfly list: tifxyz, labels, mask, validation, crop) sampled
   every --step label px inside its supervision mask, the label px -> canvas ((r step + 0.5) F - 0.5) -> tifxyz grid (x scale), a 3-D
   point on the published mesh (bilinear), its label, validation, ink-holdout box (H.json boxes of this --ct-key's scan); the nearest
   wrap vertex (a kd tree over the wraps' vertices within the labels' box + 200, scaled by the wraps' voxel_size_um / U), the foot point
   on that wrap's bilinear patch (6 Gauss-Newton steps), the miss. Everything up to the maps depends on (WRAPS, SEGS, step, ct key,
   holdouts) only, never on a checkpoint: --table FILE keeps it (built when missing or when its inputs changed), so a score is the maps'
   lookups plus the AUCs: seconds instead of pairwrap2.py's 5-10 min per split.

   Per --split (heldout = validation | holdout, hold, valonly = validation and not holdout, rest, all): the samples with miss <= --tol um
   and every map valid (> 0 for --nonzero maps, default all; a map without a file for a wrap is invalid there), AUC of each map
   (8-bit histogram ranks), its difference to --ref with a 95 % bootstrap interval over 1024-label-px blocks per segment (--draws draws;
   our RNG, not numpy's: the intervals agree with pairwrap2.py's within the bootstrap noise, the AUCs exactly), ink share at >= 254 and
   its recall. Out: OUT.<split>.tsv, pairwrap2.py's columns.

   Hard failures (exit 1, no tsv), never an n = 0 table: an unreadable segment / raster / wrap, a map dir missing or without one file of
   --need (wNNN-wMMM or a comma list), an unreadable or non-proportional map, fewer than --min-n samples in a split, a nan AUC. */
#pragma clang fp contract(off) reassociate(off)   /* (the samples' points as numpy computes them: a contracted bil() moved one sample across a holdout box face) */
#include "imgops.h"
#include "inksplat.h"
#include "json.h"
#include "optcheck.h"
#include "render_core.h"
#include "store.h"
#include "zarr3.h"
#include <dirent.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static const char *wopt(int argc, char **argv, const char *name, const char *dflt) {
    opt_n(name);
    for (int i = 5; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}
static int cmp_dbl(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static int cmp_u64(const void *a, const void *b) { const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return (x > y) - (x < y); }
static double wnow(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

typedef struct { char name[256]; double sc; int w, h; float *X, *Y, *Z; } wwrap;
typedef struct { int32_t k; uint8_t lab, val, hold, pad; uint64_t blk; double fr, fc, miss; double P[3]; } wsamp;   /* miss: level-0 (label) voxels */

static double bil(const float *G, int H, int W, double r, double c) {   /* wrapscore.py bil() */
    double rf = floor(r), cf = floor(c); int64_t r0 = (int64_t)rf, c0 = (int64_t)cf;
    if (r0 < 0) r0 = 0; if (r0 > H - 2) r0 = H - 2; if (c0 < 0) c0 = 0; if (c0 > W - 2) c0 = W - 2;
    const double fr = r - r0, fc = c - c0; const size_t i = (size_t)r0 * W + c0;
    return (1 - fr) * ((1 - fc) * (double)G[i] + fc * (double)G[i + 1]) + fr * ((1 - fc) * (double)G[i + W] + fc * (double)G[i + W + 1]);
}
static char *slurp(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb"); if (!f) return nullptr;
    fseek(f, 0, SEEK_END); long l = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)l + 1); if (b && fread(b, 1, (size_t)l, f) != (size_t)l) { free(b); b = nullptr; }
    fclose(f); if (b) { b[l] = 0; if (n) *n = (size_t)l; } return b;
}
static int meta_scale(const char *dir, double *sc, double *vum) {   /* tifxyz meta.json: scale[0], voxel_size_um (0 when absent) */
    char p[1400]; snprintf(p, sizeof p, "%s/meta.json", dir); size_t n; char *t = slurp(p, &n); if (!t) return -1;
    json *j = json_parse(t, n); free(t); if (!j) return -1;
    *sc = json_num(json_at(json_get(j, "scale"), 0), 0); *vum = json_num(json_get(j, "voxel_size_um"), 0); json_free(j);
    return *sc > 0 ? 0 : -1;
}
static int load_xyz(const char *dir, float **X, float **Y, float **Z, int *w, int *h) {
    char p[1400]; int w2, h2, w3, h3;
    snprintf(p, sizeof p, "%s/x.tif", dir); *X = img_read_f32(p, w, h);
    snprintf(p, sizeof p, "%s/y.tif", dir); *Y = img_read_f32(p, &w2, &h2);
    snprintf(p, sizeof p, "%s/z.tif", dir); *Z = img_read_f32(p, &w3, &h3);
    if (!*X || !*Y || !*Z || *w != w2 || *w != w3 || *h != h2 || *h != h3 || *w < 2 || *h < 2) { free(*X); free(*Y); free(*Z); return -1; }
    return 0;
}

/* ---- a 3-D kd tree over float points (exact nearest neighbour, double distances as scipy's cKDTree on float32 input) ---- */
typedef struct { float p[3]; uint32_t k, r, c; } kpt;
typedef struct { int64_t lo, hi; int ax; float split; int64_t left, right; float bmin[3], bmax[3]; } knode;
typedef struct { kpt *pt; knode *nd; int64_t nn, cap; } ktree;
static void kselect(kpt *a, int64_t n, int64_t k, int ax) {   /* quickselect: a[k] the k-th along ax, smaller left */
    int64_t lo = 0, hi = n - 1;
    while (lo < hi) {
        const float pv = a[lo + (hi - lo) / 2].p[ax]; int64_t i = lo, j = hi;
        while (i <= j) { while (a[i].p[ax] < pv) i++; while (a[j].p[ax] > pv) j--; if (i <= j) { kpt t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; } }
        if (k <= j) hi = j; else if (k >= i) lo = i; else break;
    }
}
static int64_t kbuild(ktree *T, int64_t lo, int64_t hi) {
    if (T->nn == T->cap) { T->cap = T->cap ? T->cap * 2 : 1024; T->nd = realloc(T->nd, (size_t)T->cap * sizeof *T->nd); }
    const int64_t id = T->nn++; knode *N = &T->nd[id]; N->lo = lo; N->hi = hi; N->left = N->right = -1;
    for (int d = 0; d < 3; d++) { N->bmin[d] = INFINITY; N->bmax[d] = -INFINITY; }
    for (int64_t i = lo; i < hi; i++) for (int d = 0; d < 3; d++) { const float v = T->pt[i].p[d]; if (v < N->bmin[d]) N->bmin[d] = v; if (v > N->bmax[d]) N->bmax[d] = v; }
    if (hi - lo <= 16) return id;
    int ax = 0; for (int d = 1; d < 3; d++) if (N->bmax[d] - N->bmin[d] > N->bmax[ax] - N->bmin[ax]) ax = d;
    const int64_t m = lo + (hi - lo) / 2; kselect(T->pt + lo, hi - lo, m - lo, ax);
    N->ax = ax; N->split = T->pt[m].p[ax];
    const int64_t l = kbuild(T, lo, m), r = kbuild(T, m, hi); T->nd[id].left = l; T->nd[id].right = r;
    return id;
}
static double kbox_d2(const knode *N, const double q[3]) {
    double s = 0; for (int d = 0; d < 3; d++) { const double v = q[d] < N->bmin[d] ? N->bmin[d] - q[d] : q[d] > N->bmax[d] ? q[d] - N->bmax[d] : 0; s += v * v; }
    return s;
}
static int64_t knearest(const ktree *T, const double q[3], double *bd2) {
    int64_t st[128]; int sp = 0, best = -1; double b = INFINITY; st[sp++] = 0;
    while (sp) {
        const knode *N = &T->nd[st[--sp]]; if (kbox_d2(N, q) >= b) continue;
        if (N->left < 0) {
            for (int64_t i = N->lo; i < N->hi; i++) { double s = 0; for (int d = 0; d < 3; d++) { const double v = q[d] - (double)T->pt[i].p[d]; s += v * v; }
                if (s < b) { b = s; best = i; } }
            continue;
        }
        const int near_left = q[N->ax] < N->split;   /* push the far child first */
        st[sp++] = near_left ? N->right : N->left; st[sp++] = near_left ? N->left : N->right;
    }
    *bd2 = b; return best;
}

/* ---- the sample table ---- */
typedef struct { int nw; wwrap *W; size_t ns; wsamp *S; } wtable;
typedef struct { const ktree *T; wtable *t; atomic_size_t next; } fjob;
static void *foot_worker(void *arg) {   /* nearest vertex, then wrapscore.py's Gauss-Newton foot point */
    fjob *J = arg; wtable *t = J->t;
    for (;;) {
        const size_t s0 = atomic_fetch_add(&J->next, 4096); if (s0 >= t->ns) break;
        const size_t s1 = s0 + 4096 < t->ns ? s0 + 4096 : t->ns;
        for (size_t s = s0; s < s1; s++) {
            wsamp *x = &t->S[s]; double d2; const int64_t vi = knearest(J->T, x->P, &d2); const kpt *v = &J->T->pt[vi];
            const double dist0 = sqrt(d2); const wwrap *w = &t->W[v->k]; const float *G[3] = {w->Z, w->Y, w->X}; const int H = w->h, Wd = w->w;
            double r = v->r, c = v->c;
            for (int it = 0; it < 6; it++) {
                double f0[3], fr[3], fc[3], e[3];
                for (int d = 0; d < 3; d++) { f0[d] = bil(G[d], H, Wd, r, c); fr[d] = (bil(G[d], H, Wd, r + .5, c) - f0[d]) * 2; fc[d] = (bil(G[d], H, Wd, r, c + .5) - f0[d]) * 2; e[d] = x->P[d] - f0[d]; }
                double a = 0, b = 0, dd = 0, j0 = 0, j1 = 0;
                for (int d = 0; d < 3; d++) { a += fr[d] * fr[d]; b += fr[d] * fc[d]; dd += fc[d] * fc[d]; j0 += fr[d] * e[d]; j1 += fc[d] * e[d]; }
                a += 1e-6; dd += 1e-6; const double det = a * dd - b * b;
                double l0 = (dd * j0 - b * j1) / det, l1 = (a * j1 - b * j0) / det;
                if (!isfinite(l0)) l0 = 0; if (!isfinite(l1)) l1 = 0;
                l0 = l0 < -1 ? -1 : l0 > 1 ? 1 : l0; l1 = l1 < -1 ? -1 : l1 > 1 ? 1 : l1;
                r += l0; c += l1; if (r < 0) r = 0; if (r > H - 1.001) r = H - 1.001; if (c < 0) c = 0; if (c > Wd - 1.001) c = Wd - 1.001;
            }
            double f0[3], s2 = 0; for (int d = 0; d < 3; d++) { f0[d] = bil(G[d], H, Wd, r, c); const double q = x->P[d] - f0[d]; s2 += q * q; }
            int64_t r0 = (int64_t)floor(r), c0 = (int64_t)floor(c); if (r0 > H - 2) r0 = H - 2; if (c0 > Wd - 2) c0 = Wd - 2; if (r0 < 0) r0 = 0; if (c0 < 0) c0 = 0;
            const size_t i = (size_t)r0 * Wd + c0; const int okc = w->Z[i] > 0 && w->Z[i + Wd] > 0 && w->Z[i + 1] > 0 && w->Z[i + Wd + 1] > 0;
            double d = sqrt(s2); if (!okc && dist0 < d) d = dist0;
            x->k = (int32_t)v->k; x->fr = r; x->fc = c; x->miss = d;
        }
    }
    return nullptr;
}

/* the label raster thresholded as rd2(): >= 1 when the strided raster's max is <= 1, else >= 128 (the max over the WHOLE strided raster:
   read again only when the window's max is exactly 1) */
static uint8_t *raster_win(const char *path, int step, int64_t i0, int64_t i1, int64_t j0, int64_t j1, int64_t *H, int64_t *W, int nth) {
    void *h = inksplat_label_open(path, H, W); if (!h) { fprintf(stderr, "ink-wrapscore: cannot read %s\n", path); return nullptr; }
    uint8_t *o = malloc((size_t)((i1 - i0) > 0 ? i1 - i0 : 1) * ((j1 - j0) > 0 ? j1 - j0 : 1)); int mx = 0;
    if (!o || inksplat_label_strided(h, step, i0, i1, j0, j1, o, &mx, nth)) { fprintf(stderr, "ink-wrapscore: read failed %s\n", path); free(o); inksplat_label_close(h); return nullptr; }
    int gmx = mx;
    if (mx == 1) {   /* the whole raster's strided max decides the threshold */
        const int64_t hs = (*H + step - 1) / step, ws = (*W + step - 1) / step; uint8_t *g = malloc((size_t)hs * ws);
        if (!g || inksplat_label_strided(h, step, 0, hs, 0, ws, g, &gmx, nth)) { free(g); free(o); inksplat_label_close(h); return nullptr; }
        free(g);
    }
    inksplat_label_close(h);
    const int thr = gmx <= 1 ? 1 : 128; const size_t n = (size_t)(i1 - i0) * (j1 - j0);
    for (size_t k = 0; k < n; k++) o[k] = o[k] >= thr;
    return o;
}
static int cmp_name(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }

static int table_build(wtable *t, const char *wraps, const char *segs, const char *ctkey, const char *holdf, int step, double umL, int nth) {
    const double t0 = wnow(); size_t n; char *txt = slurp(segs, &n); json *J = txt ? json_parse(txt, n) : nullptr; free(txt);
    const json *SG = json_get(J, "segments"); if (!SG || !SG->n) { fprintf(stderr, "ink-wrapscore: %s: no segments\n", segs); json_free(J); return -1; }
    /* the holdout boxes of this scan */
    int nh = 0; int64_t hb[256][6]; const char *ckb = strrchr(ctkey, '/'); ckb = ckb ? ckb + 1 : ctkey;
    if (holdf && *holdf) {
        char *ht = slurp(holdf, &n); json *HJ = ht ? json_parse(ht, n) : nullptr; free(ht);
        if (!HJ) { fprintf(stderr, "ink-wrapscore: cannot read holdouts %s\n", holdf); json_free(J); return -1; }
        const json *B = json_get(HJ, "boxes");
        for (size_t i = 0; B && i < B->n && nh < 256; i++) { const char *c = json_str(json_get(json_at(B, i), "ct"), ""); const char *cb = strrchr(c, '/'); cb = cb ? cb + 1 : c;
            if (strcmp(cb, ckb)) continue; const json *l = json_get(json_at(B, i), "l0"); for (int d = 0; d < 6; d++) hb[nh][d] = (int64_t)json_num(json_at(l, d), 0); nh++; }
        json_free(HJ);
    }
    size_t cap = 1 << 20; t->S = malloc(cap * sizeof *t->S); t->ns = 0;
    char **seen = calloc(SG->n, sizeof *seen); char **names = calloc(SG->n, sizeof *names); int nnames = 0, rc = -1;
    const int64_t bs = 1024 / step;
    for (size_t gi = 0; gi < SG->n; gi++) {
        const json *g = json_at(SG, gi); const char *nm = json_str(json_get(g, "name"), ""), *tx = json_str(json_get(g, "tifxyz"), nullptr), *lb = json_str(json_get(g, "labels"), nullptr),
                   *mk = json_str(json_get(g, "mask"), nullptr), *vl = json_str(json_get(g, "validation"), nullptr); const json *cr = json_get(g, "crop");
        if (!tx || !lb || !mk) { fprintf(stderr, "ink-wrapscore: segment %zu (%s): needs tifxyz, labels, mask\n", gi, nm); goto out; }
        char key[2400]; snprintf(key, sizeof key, "%s|", lb);
        for (size_t d = 0; cr && d < cr->n; d++) snprintf(key + strlen(key), sizeof key - strlen(key), "%lld,", (long long)json_num(json_at(cr, d), 0));
        int dup = 0; for (size_t q = 0; q < gi; q++) if (seen[q] && !strcmp(seen[q], key)) dup = 1;
        seen[gi] = strdup(key); if (dup) continue;
        int ni = -1; for (int q = 0; q < nnames; q++) if (!strcmp(names[q], nm)) ni = q;
        if (ni < 0) { ni = nnames; names[nnames++] = strdup(nm); }
        double sc, vum; float *X, *Y, *Z; int zw, zh;
        if (meta_scale(tx, &sc, &vum) || load_xyz(tx, &X, &Y, &Z, &zw, &zh)) { fprintf(stderr, "ink-wrapscore: cannot read the tifxyz %s\n", tx); goto out; }
        int64_t LH, LW, MH, MW, VH = 0, VW = 0;
        { void *h; if (!(h = inksplat_label_open(lb, &LH, &LW))) { fprintf(stderr, "ink-wrapscore: cannot read %s\n", lb); goto out; } inksplat_label_close(h);
          if (!(h = inksplat_label_open(mk, &MH, &MW))) { fprintf(stderr, "ink-wrapscore: cannot read %s\n", mk); goto out; } inksplat_label_close(h);
          if (vl) { if (!(h = inksplat_label_open(vl, &VH, &VW))) { fprintf(stderr, "ink-wrapscore: cannot read %s\n", vl); goto out; } inksplat_label_close(h); } }
        int64_t hs = (MH + step - 1) / step, ws = (MW + step - 1) / step;
        { const int64_t a = (LH + step - 1) / step, b = (LW + step - 1) / step; if (a < hs) hs = a; if (b < ws) ws = b; }
        if (vl) { const int64_t a = (VH + step - 1) / step, b = (VW + step - 1) / step; if (a < hs) hs = a; if (b < ws) ws = b; }
        int64_t i0 = 0, i1 = hs, j0 = 0, j1 = ws;
        if (cr && cr->n == 4) { const int64_t c0 = (int64_t)json_num(json_at(cr, 0), 0), r0 = (int64_t)json_num(json_at(cr, 1), 0), cw = (int64_t)json_num(json_at(cr, 2), 0), ch = (int64_t)json_num(json_at(cr, 3), 0);
            i0 = r0 / step; i1 = (r0 + ch) / step; j0 = c0 / step; j1 = (c0 + cw) / step;
            if (i0 < 0) i0 = 0; if (j0 < 0) j0 = 0; if (i1 > hs) i1 = hs; if (j1 > ws) j1 = ws; }
        if (i1 <= i0 || j1 <= j0) { free(X); free(Y); free(Z); continue; }
        int64_t dH, dW; uint8_t *M = raster_win(mk, step, i0, i1, j0, j1, &dH, &dW, nth), *L = M ? raster_win(lb, step, i0, i1, j0, j1, &dH, &dW, nth) : nullptr,
            *V = L && vl ? raster_win(vl, step, i0, i1, j0, j1, &dH, &dW, nth) : nullptr;
        if (!M || !L || (vl && !V)) { free(M); free(L); free(V); free(X); free(Y); free(Z); goto out; }
        const double ew = zw / sc, eh = zh / sc; double F = 0.5 * (ew / LW + eh / LH); const double p2 = pow(2.0, nearbyint(log2(F)));
        if (fabs(F - p2) < 0.002 * p2) F = p2;
        if (fabs(F - 1) > 1e-3) printf("labels: %-44.44s F %.5f canvas px per label px (labels %lld x %lld, canvas %.0f x %.0f)\n", nm, F, (long long)LW, (long long)LH, ew, eh);
        size_t cnt = 0, ink = 0, nv = 0, nho = 0; const int64_t nj = j1 - j0;
        for (int64_t i = i0; i < i1; i++) for (int64_t j = j0; j < j1; j++) {
            const size_t q = (size_t)(i - i0) * nj + (j - j0); if (!M[q]) continue;
            const double gr = ((double)(i * step) + 0.5) * F - 0.5, gr2 = gr * sc, gc2 = (((double)(j * step) + 0.5) * F - 0.5) * sc;
            if (!(gr2 >= 0 && gc2 >= 0 && gr2 <= zh - 1 && gc2 <= zw - 1)) continue;
            int64_t r0 = (int64_t)floor(gr2), c0 = (int64_t)floor(gc2); if (r0 > zh - 2) r0 = zh - 2; if (c0 > zw - 2) c0 = zw - 2; if (r0 < 0) r0 = 0; if (c0 < 0) c0 = 0;
            const size_t zi = (size_t)r0 * zw + c0; if (!(Z[zi] > 0 && Z[zi + zw] > 0 && Z[zi + 1] > 0 && Z[zi + zw + 1] > 0)) continue;
            if (t->ns == cap) { cap *= 2; t->S = realloc(t->S, cap * sizeof *t->S); }
            wsamp *x = &t->S[t->ns++]; memset(x, 0, sizeof *x);
            x->P[0] = bil(Z, zh, zw, gr2, gc2); x->P[1] = bil(Y, zh, zw, gr2, gc2); x->P[2] = bil(X, zh, zw, gr2, gc2);
            x->lab = L[q]; x->val = V ? V[q] : 0;
            for (int b = 0; b < nh && !x->hold; b++) { int in = 1; for (int d = 0; d < 3; d++) in &= x->P[d] >= hb[b][d] && x->P[d] < hb[b][d] + hb[b][3 + d]; x->hold = (uint8_t)in; }
            x->blk = (uint64_t)ni << 42 | (uint64_t)(i / bs) << 21 | (uint64_t)(j / bs);
            cnt++; ink += x->lab; nv += x->val; nho += x->hold;
        }
        printf("labels: %-44.44s %8zu samples (ink %.3f, validation %.3f, holdout %.3f)\n", nm, cnt, cnt ? (double)ink / cnt : NAN, cnt ? (double)nv / cnt : NAN, cnt ? (double)nho / cnt : NAN);
        fflush(stdout); free(M); free(L); free(V); free(X); free(Y); free(Z);
    }
    if (!t->ns) { fprintf(stderr, "ink-wrapscore: no labelled samples in %s\n", segs); goto out; }
    { size_t ink = 0; for (size_t s = 0; s < t->ns; s++) ink += t->S[s].lab; printf("labels: %zu samples, ink %.3f (%.0f s)\n", t->ns, (double)ink / t->ns, wnow() - t0); }
    /* the wraps */
    double lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (size_t s = 0; s < t->ns; s++) for (int d = 0; d < 3; d++) { if (t->S[s].P[d] < lo[d]) lo[d] = t->S[s].P[d]; if (t->S[s].P[d] > hi[d]) hi[d] = t->S[s].P[d]; }
    for (int d = 0; d < 3; d++) { lo[d] -= 200; hi[d] += 200; }
    DIR *dd = opendir(wraps); if (!dd) { fprintf(stderr, "ink-wrapscore: cannot open %s\n", wraps); goto out; }
    char **wl = nullptr; int nwl = 0; struct dirent *de;
    while ((de = readdir(dd))) { if (de->d_name[0] == '.') continue; char p[1400]; struct stat st; snprintf(p, sizeof p, "%s/%s/meta.json", wraps, de->d_name);
        if (stat(p, &st)) continue; wl = realloc(wl, (nwl + 1) * sizeof *wl); wl[nwl++] = strdup(de->d_name); }
    closedir(dd); qsort(wl, nwl, sizeof *wl, cmp_name);
    if (!nwl) { fprintf(stderr, "ink-wrapscore: no wraps (*/meta.json) in %s\n", wraps); free(wl); goto out; }
    t->nw = nwl; t->W = calloc(nwl, sizeof *t->W); ktree T = {0}; size_t tcap = 1 << 20, np = 0; T.pt = malloc(tcap * sizeof *T.pt);
    for (int k = 0; k < nwl; k++) {
        wwrap *w = &t->W[k]; char p[1400]; snprintf(p, sizeof p, "%s/%s", wraps, wl[k]); snprintf(w->name, sizeof w->name, "%s", wl[k]); double vum;
        if (meta_scale(p, &w->sc, &vum) || load_xyz(p, &w->X, &w->Y, &w->Z, &w->w, &w->h)) { fprintf(stderr, "ink-wrapscore: cannot read the wrap %s\n", p); goto out; }
        const float f = (float)((vum > 0 ? vum : umL) / umL); const size_t nn = (size_t)w->w * w->h; size_t nk = 0;
        for (size_t i = 0; i < nn; i++) { w->X[i] *= f; w->Y[i] *= f; w->Z[i] *= f; if (w->Z[i] <= 0) w->Z[i] = -1; }
        for (size_t i = 0; i < nn; i++) {
            const float z = w->Z[i], y = w->Y[i], x = w->X[i];
            if (!(z > 0 && z >= lo[0] && z <= hi[0] && y >= lo[1] && y <= hi[1] && x >= lo[2] && x <= hi[2])) continue;
            if (np == tcap) { tcap *= 2; T.pt = realloc(T.pt, tcap * sizeof *T.pt); }
            T.pt[np++] = (kpt){{z, y, x}, (uint32_t)k, (uint32_t)(i / w->w), (uint32_t)(i % w->w)}; nk++;
        }
        printf("wraps: %s scale %g um %g -> x%.4f: %zu vertices near the labels\n", w->name, w->sc, vum, (double)f, nk);
        free(wl[k]);
    }
    free(wl);
    if (!np) { fprintf(stderr, "ink-wrapscore: no wrap vertex near the labelled meshes (coordinates / scale?)\n"); free(T.pt); goto out; }
    const double t1 = wnow(); kbuild(&T, 0, (int64_t)np);
    fjob FJ = {.T = &T, .t = t}; atomic_init(&FJ.next, 0); pthread_t th[64]; const int nt = nth < 1 ? 1 : nth > 64 ? 64 : nth;
    for (int i = 0; i < nt; i++) pthread_create(&th[i], nullptr, foot_worker, &FJ);
    for (int i = 0; i < nt; i++) pthread_join(th[i], nullptr);
    free(T.pt); free(T.nd);
    { double *m = malloc(t->ns * sizeof *m); for (size_t s = 0; s < t->ns; s++) m[s] = t->S[s].miss * umL;
      qsort(m, t->ns, sizeof *m, cmp_dbl); const double md = t->ns % 2 ? m[t->ns / 2] : 0.5 * (m[t->ns / 2 - 1] + m[t->ns / 2]), p9 = m[(size_t)(0.9 * (t->ns - 1))];
      printf("miss: median %.1f um, p90 %.1f um (%zu vertices, foot points %.0f s, table %.0f s)\n", md, p9, np, wnow() - t1, wnow() - t0); free(m); }
    rc = 0;
out:
    for (size_t q = 0; q < SG->n; q++) free(seen[q]);
    for (int q = 0; q < nnames; q++) free(names[q]);
    free(seen); free(names); json_free(J); return rc;
}

/* the table cache: "UWS1", the key (inputs + their sizes / mtimes), the wraps (name, sc, w, h), the samples (the grids are not kept) */
static void file_sig(const char *p, char *o, size_t n) { struct stat st; if (stat(p, &st)) snprintf(o, n, "%s:missing;", p); else snprintf(o, n, "%s:%lld:%lld;", p, (long long)st.st_size, (long long)st.st_mtime); }
static void table_key(char *k, size_t n, const char *wraps, const char *segs, const char *ctkey, const char *holdf, int step, double umL) {
    snprintf(k, n, "v1;wraps=%s;segs=%s;ct=%s;hold=%s;step=%d;um=%.6g;", wraps, segs, ctkey, holdf ? holdf : "", step, umL);
    char s[1600]; file_sig(segs, s, sizeof s); snprintf(k + strlen(k), n - strlen(k), "%s", s);
    if (holdf && *holdf) { file_sig(holdf, s, sizeof s); snprintf(k + strlen(k), n - strlen(k), "%s", s); }
    DIR *d = opendir(wraps); struct dirent *de; long long mt = 0, sz = 0; int nw = 0;
    while (d && (de = readdir(d))) { if (de->d_name[0] == '.') continue; for (int c = 0; c < 4; c++) { char p[1400]; struct stat st;
        snprintf(p, sizeof p, "%s/%s/%s", wraps, de->d_name, (const char *[]){"x.tif", "y.tif", "z.tif", "meta.json"}[c]);
        if (!stat(p, &st)) { if (st.st_mtime > mt) mt = st.st_mtime; sz += st.st_size; if (c == 3) nw++; } } }
    if (d) closedir(d);
    snprintf(k + strlen(k), n - strlen(k), "wrapfiles=%d:%lld:%lld;", nw, sz, mt);
}
enum { KEYLEN = 8192 };
static int table_save(const wtable *t, const char *path, const char *key) {
    char tmp[1400]; snprintf(tmp, sizeof tmp, "%s.tmp%d", path, (int)getpid()); FILE *f = fopen(tmp, "wb"); if (!f) return -1;
    char k[KEYLEN] = {0}; snprintf(k, sizeof k, "%s", key); const uint64_t ns = t->ns; const uint32_t nw = (uint32_t)t->nw;
    int ok = fwrite("UWS1", 1, 4, f) == 4 && fwrite(k, 1, KEYLEN, f) == KEYLEN && fwrite(&nw, 4, 1, f) == 1 && fwrite(&ns, 8, 1, f) == 1;
    for (int i = 0; ok && i < t->nw; i++) { wwrap w = t->W[i]; w.X = w.Y = w.Z = nullptr; ok = fwrite(&w, sizeof w, 1, f) == 1; }
    ok = ok && fwrite(t->S, sizeof *t->S, t->ns, f) == t->ns; ok = !fclose(f) && ok;
    if (!ok || rename(tmp, path)) { remove(tmp); return -1; }
    return 0;
}
static int table_load(wtable *t, const char *path, const char *key) {
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    char m[4], k[KEYLEN]; uint32_t nw; uint64_t ns; int ok = fread(m, 1, 4, f) == 4 && !memcmp(m, "UWS1", 4) && fread(k, 1, KEYLEN, f) == KEYLEN && fread(&nw, 4, 1, f) == 1 && fread(&ns, 8, 1, f) == 1;
    k[KEYLEN - 1] = 0;
    if (!ok || strcmp(k, key)) { fclose(f); if (ok) printf("table: %s is for other inputs: rebuilding\n", path); return -1; }
    t->nw = (int)nw; t->ns = ns; t->W = calloc(nw, sizeof *t->W); t->S = malloc(ns * sizeof *t->S);
    ok = t->W && t->S && fread(t->W, sizeof *t->W, nw, f) == nw && fread(t->S, sizeof *t->S, ns, f) == ns; fclose(f);
    for (uint32_t i = 0; i < nw; i++) t->W[i].X = t->W[i].Y = t->W[i].Z = nullptr;
    if (!ok) { free(t->W); free(t->S); t->W = nullptr; t->S = nullptr; return -1; }
    return 0;
}

/* ---- the maps and the scores ---- */
static uint8_t *map_read(const char *p, int *w, int *h) {   /* uint8 TIFF (first channel of a multi-sample one) or JPEG as PIL convert('L') */
    const size_t l = strlen(p);
    if (l > 4 && !strcmp(p + l - 4, ".jpg")) {
        uint8_t *rgb = jpeg_read_rgb(p, w, h); if (!rgb) return nullptr; const size_t n = (size_t)*w * *h; uint8_t *o = malloc(n);
        for (size_t i = 0; o && i < n; i++) o[i] = (uint8_t)((rgb[3 * i] * 19595u + rgb[3 * i + 1] * 38470u + rgb[3 * i + 2] * 7471u + 0x8000u) >> 16);
        free(rgb); return o;
    }
    return img_read_u8(p, w, h);
}
static double wauc(const double *hp, const double *hn) {   /* pairwrap2.py wauc: 8-bit histograms */
    double sp = 0, sn = 0, s = 0, cn = 0; for (int b = 0; b < 256; b++) { sp += hp[b]; sn += hn[b]; }
    for (int b = 0; b < 256; b++) { s += hp[b] * (cn + 0.5 * hn[b]); cn += hn[b]; }
    return s / (sp * sn);
}
static double pctl(double *v, int n, double q) {   /* numpy percentile, linear */
    qsort(v, n, sizeof *v, cmp_dbl); const double idx = q / 100 * (n - 1); const int a = (int)floor(idx), b = a + 1 < n ? a + 1 : a;
    return v[a] + (v[b] - v[a]) * (idx - a);
}
static uint64_t sm64(uint64_t *s) { uint64_t z = (*s += 0x9e3779b97f4a7c15ull); z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull; return z ^ (z >> 31); }
typedef struct { int nm, nb, ref, draws, lo, hi; const uint32_t *bh; double *dl; } bjob;   /* bh[(b nm + m) 512 + bin]: block histograms (pos 0..255, neg 256..511) */
static void *boot_worker(void *arg) {
    bjob *J = arg; uint32_t *cnt = malloc((size_t)J->nb * sizeof *cnt); double *hp = malloc((size_t)J->nm * 512 * sizeof *hp);
    for (int d = J->lo; d < J->hi; d++) {
        uint64_t s = 0x5eed0000ull + (uint64_t)d; memset(cnt, 0, (size_t)J->nb * sizeof *cnt);
        for (int i = 0; i < J->nb; i++) cnt[(uint32_t)(((unsigned __int128)sm64(&s) * (uint64_t)J->nb) >> 64)]++;
        memset(hp, 0, (size_t)J->nm * 512 * sizeof *hp);
        for (int b = 0; b < J->nb; b++) { if (!cnt[b]) continue; const double w = cnt[b];
            for (int m = 0; m < J->nm; m++) { const uint32_t *h = J->bh + ((size_t)b * J->nm + m) * 512; double *o = hp + (size_t)m * 512; for (int k = 0; k < 512; k++) if (h[k]) o[k] += w * h[k]; } }
        const double ar = wauc(hp + (size_t)J->ref * 512, hp + (size_t)J->ref * 512 + 256);
        for (int m = 0; m < J->nm; m++) J->dl[(size_t)m * J->draws + d] = wauc(hp + (size_t)m * 512, hp + (size_t)m * 512 + 256) - ar;
    }
    free(cnt); free(hp); return nullptr;
}


/* ---- the volume maps (--vmap): render's pixel (render_core.h rc_pixel, depth D, scale 1) at each sample's map pixel, gated as img-gate:
   the ink volume x max(c12, c13) / 255, without rendering or writing the wraps; and the 512-tile lists a cropped predict needs ---- */
static int need_set(const char *need, const wtable *t, uint8_t *in) {   /* "wNNN-wMMM" or "a,b,c" -> in[k]; -1 when a name is not a wrap */
    memset(in, 0, (size_t)t->nw); int a, b, n = 0; char nb[2048]; snprintf(nb, sizeof nb, "%s", need);
    if (sscanf(nb, "w%d-w%d", &a, &b) == 2) {
        for (int q = a; q <= b; q++) { char nm[32]; snprintf(nm, sizeof nm, "w%03d", q); int f = 0;
            for (int k = 0; k < t->nw; k++) if (!strcmp(t->W[k].name, nm)) { in[k] = 1; f = 1; n++; }
            if (!f) { fprintf(stderr, "ink-wrapscore: --need %s: no wrap %s in the table\n", need, nm); return -1; } }
        return n;
    }
    for (char *sv = nullptr, *x = strtok_r(nb, ",", &sv); x; x = strtok_r(nullptr, ",", &sv)) { int f = 0;
        for (int k = 0; k < t->nw; k++) if (!strcmp(t->W[k].name, x)) { in[k] = 1; f = 1; n++; }
        if (!f) { fprintf(stderr, "ink-wrapscore: --need: no wrap %s in the table\n", x); return -1; } }
    return n;
}
static void samp_px(const wwrap *w, const wsamp *x, int *px, int *py) {   /* the sample's pixel of a scale-1 render (pairwrap2.py's lookup) */
    const double gs = 1 / w->sc; const int ow = render_out_px(w->w, gs), oh = render_out_px(w->h, gs);
    const double cw = w->w / w->sc, ch = w->h / w->sc, fx = ow / cw, fy = oh / ch;
    int64_t xi = (int64_t)nearbyint(x->fc / w->sc * fx), yi = (int64_t)nearbyint(x->fr / w->sc * fy);
    *px = (int)(xi < 0 ? 0 : xi > ow - 1 ? ow - 1 : xi); *py = (int)(yi < 0 ? 0 : yi > oh - 1 ? oh - 1 : yi);
}
static int px_point(const float *Z, const float *Y, const float *X, int w, int h, double gs, int px, int py, double P[3]) {   /* rc_pixel's point */
    const double gu = px / gs, gv = py / gs; int c0 = (int)gu, r0 = (int)gv; if (c0 >= w - 1) c0 = w - 2; if (r0 >= h - 1) r0 = h - 2; if (c0 < 0 || r0 < 0) return 0;
    const double fu = gu - c0, fv = gv - r0; const size_t i00 = (size_t)r0 * w + c0, i01 = i00 + 1, i10 = i00 + w, i11 = i10 + 1;
    if (!(Z[i00] >= 0 && Z[i01] >= 0 && Z[i10] >= 0 && Z[i11] >= 0)) return 0;
    const float *A[3] = {Z, Y, X};
    for (int d = 0; d < 3; d++) { const double a = A[d][i00], b = A[d][i01], c = A[d][i10], e = A[d][i11]; P[d] = (1 - fv) * ((1 - fu) * a + fu * b) + fv * ((1 - fu) * c + fu * e); }
    return 1;
}
typedef struct { float *X, *Y, *Z; } rgrid;
static int cmp_u64b(const void *a, const void *b) { const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return (x > y) - (x < y); }
enum { VMARG = 6 };   /* voxels around a pixel's point: depth 4 + the trilinear corner + rounding */
typedef struct { z3 *z; int sh; const int64_t *shape; const uint64_t *keys; uint8_t **bufs; size_t nk; atomic_size_t next; int fail; } djob;
static void *dec_worker(void *arg) {
    djob *J = arg; const int64_t C = (int64_t)1 << J->sh; uint8_t *tmp = malloc((size_t)(C * C * C));
    for (;;) {
        const size_t i = atomic_fetch_add(&J->next, 1); if (i >= J->nk) break;
        const uint64_t k = J->keys[i]; const int64_t c[3] = {(int64_t)(k >> 42), (int64_t)(k >> 21 & 0x1fffff), (int64_t)(k & 0x1fffff)};
        int64_t o[3], n[3]; int in = 1; for (int d = 0; d < 3; d++) { o[d] = c[d] * C; n[d] = J->shape[d] - o[d] < C ? J->shape[d] - o[d] : C; in &= n[d] > 0; }
        if (!in) { J->bufs[i] = nullptr; continue; }
        uint8_t *b = calloc((size_t)(C * C * C), 1); if (!b || z3_read(J->z, o, n, tmp, 1)) { free(b); J->fail = 1; continue; }
        for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) memcpy(b + ((size_t)z * C + y) * C, tmp + ((size_t)z * n[1] + y) * n[2], (size_t)n[2]);
        J->bufs[i] = b;
    }
    free(tmp); return nullptr;
}
typedef struct { const wwrap *w; const rgrid *g; const size_t *idx; size_t n; const wsamp *S; int sh; const uint64_t *keys; uint8_t *const *bufs; size_t nk; int depth; uint8_t *out; atomic_size_t next; } ejob;
static void *eval_worker(void *arg) {
    ejob *J = arg; const double gs = 1 / J->w->sc; const int64_t C = (int64_t)1 << J->sh;
    for (;;) {
        const size_t i = atomic_fetch_add(&J->next, 1); if (i >= J->n) break;
        const wsamp *x = &J->S[J->idx[i]]; int px, py; samp_px(J->w, x, &px, &py); double P[3]; J->out[i] = 0;
        if (!px_point(J->g->Z, J->g->Y, J->g->X, J->w->w, J->w->h, gs, px, py, P)) continue;
        rc_vol V; V.sh = J->sh; const uint8_t *ptr[27];
        for (int d = 0; d < 3; d++) { int64_t a = (int64_t)floor((P[d] - VMARG) / C), b = (int64_t)floor((P[d] + VMARG) / C); if (a < 0) a = 0; if (b < a) b = a; V.c0[d] = a; V.cn[d] = b - a + 1; }
        if (V.cn[0] * V.cn[1] * V.cn[2] > 27) continue;
        for (int64_t a = 0; a < V.cn[0]; a++) for (int64_t b = 0; b < V.cn[1]; b++) for (int64_t c = 0; c < V.cn[2]; c++) {
            const uint64_t k = (uint64_t)(V.c0[0] + a) << 42 | (uint64_t)(V.c0[1] + b) << 21 | (uint64_t)(V.c0[2] + c);
            const uint64_t *f = bsearch(&k, J->keys, J->nk, sizeof k, cmp_u64b); ptr[(a * V.cn[1] + b) * V.cn[2] + c] = f ? J->bufs[f - J->keys] : nullptr; }
        V.ptr = ptr; uint8_t val = 0;
        if (rc_pixel(J->g->Z, J->g->Y, J->g->X, J->w->w, J->w->h, gs, px, py, &V, 1, &J->depth, nullptr, nullptr, &val, nullptr)) J->out[i] = val;
    }
    return nullptr;
}
/* one volume at the samples idx[0..n) of wrap w -> out[]; *absent: samples whose point's shard is not in the store */
static int vol_eval(z3 *z, const wwrap *w, const rgrid *g, const wsamp *S, const size_t *idx, size_t n, int depth, int nth, uint8_t *out, size_t *absent) {
    const z3_meta *m = z3_meta_of(z); const int C = m->chunk[0]; int sh = 0; while ((1 << sh) < C) sh++;
    if (C != m->chunk[1] || C != m->chunk[2] || (1 << sh) != C) { fprintf(stderr, "ink-wrapscore: volume chunks must be cubes of a power of 2\n"); return -1; }
    const double gs = 1 / w->sc; uint64_t *keys = malloc(n * 27 * sizeof *keys), *sk = malloc(n * sizeof *sk); size_t nk = 0, ns = 0;
    for (size_t i = 0; i < n; i++) { int px, py; samp_px(w, &S[idx[i]], &px, &py); double P[3]; if (!px_point(g->Z, g->Y, g->X, w->w, w->h, gs, px, py, P)) continue;
        int64_t a[3], b[3]; for (int d = 0; d < 3; d++) { a[d] = (int64_t)floor((P[d] - VMARG) / C); b[d] = (int64_t)floor((P[d] + VMARG) / C); if (a[d] < 0) a[d] = 0; if (b[d] < a[d]) b[d] = a[d]; }
        for (int64_t p = a[0]; p <= b[0]; p++) for (int64_t q = a[1]; q <= b[1]; q++) for (int64_t r = a[2]; r <= b[2]; r++) if (nk < n * 27) keys[nk++] = (uint64_t)p << 42 | (uint64_t)q << 21 | (uint64_t)r;
        int64_t s3[3]; int ok = 1; for (int d = 0; d < 3; d++) { const int64_t v = (int64_t)floor(P[d]); ok &= v >= 0 && v < m->shape[d]; s3[d] = v / m->shard[d]; }
        sk[ns++] = ok ? ((uint64_t)s3[0] << 42 | (uint64_t)s3[1] << 21 | (uint64_t)s3[2]) : UINT64_MAX; }
    qsort(keys, nk, sizeof *keys, cmp_u64b); size_t u = 0; for (size_t i = 0; i < nk; i++) if (!i || keys[i] != keys[i - 1]) keys[u++] = keys[i]; nk = u;
    qsort(sk, ns, sizeof *sk, cmp_u64b); *absent = 0;
    for (size_t i = 0; i < ns;) { size_t j = i; while (j < ns && sk[j] == sk[i]) j++;
        if (sk[i] == UINT64_MAX || z3_shard_present(z, (int64_t)(sk[i] >> 42), (int64_t)(sk[i] >> 21 & 0x1fffff), (int64_t)(sk[i] & 0x1fffff)) != 1) *absent += j - i; i = j; }
    uint8_t **bufs = calloc(nk ? nk : 1, sizeof *bufs); djob D = {.z = z, .sh = sh, .shape = m->shape, .keys = keys, .bufs = bufs, .nk = nk}; atomic_init(&D.next, 0);
    const int nt = nth > 32 ? 32 : nth; pthread_t th[32];
    for (int i = 0; i < nt; i++) pthread_create(&th[i], nullptr, dec_worker, &D);
    for (int i = 0; i < nt; i++) pthread_join(th[i], nullptr);
    int rc = D.fail ? -1 : 0;
    if (!rc) { ejob E = {.w = w, .g = g, .idx = idx, .n = n, .S = S, .sh = sh, .keys = keys, .bufs = bufs, .nk = nk, .depth = depth, .out = out}; atomic_init(&E.next, 0);
        for (int i = 0; i < nt; i++) pthread_create(&th[i], nullptr, eval_worker, &E);
        for (int i = 0; i < nt; i++) pthread_join(th[i], nullptr); }
    else fprintf(stderr, "ink-wrapscore: volume read failed: %s\n", z3_error());
    for (size_t i = 0; i < nk; i++) free(bufs[i]);
    free(bufs); free(keys); free(sk); return rc;
}
static int64_t fdiv512(int64_t v) { return v >= 0 ? v / 512 : -((-v + 511) / 512); }   /* floor(v / 512) */
void img_gate_lut(uint8_t lut[256][256]);   /* src/imggate.c */
static z3 *vol_open(const char *dir, const char *name, const char *key) {   /* DIR/NAME.zarr, level KEY */
    char p[1400]; snprintf(p, sizeof p, "%s/%s.zarr", dir, name); struct stat st; if (stat(p, &st)) return nullptr;
    store *s = store_open(p); z3 *z = s ? z3_open(s, key, nullptr) : nullptr;
    if (!z) fprintf(stderr, "ink-wrapscore: %s level %s: %s\n", p, key, z3_error());
    return z;
}

int cmd_ink_wrapscore(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: ufsm ink-wrapscore WRAPS SEGS.JSON OUT --ct-key K --maps name=DIR[,name=DIR ..] [--ref NAME] [--nonzero a,b] [--split heldout,hold,valonly]\n"
                        "  [--tol 40] [--step 8] [--holdouts H.json] [--um-label 2.399] [--table FILE] [--need w035-w051] [--min-n 1000] [--draws 1000] [--threads 6]\n"
                        "  the held-out ink AUC of rendered maps on an unroll's wraps against the published human labels (wrapscore.py / pairwrap2.py in C);\n"
                        "  --table caches the checkpoint-independent samples and foot points; OUT.<split>.tsv per split; any missing input fails (exit 1)\n"
                        "  [--vmap NAME=DIR] [--tiles-out DIR [--tiles-sub S]]: volume maps; the 512 tiles (grids 0 / 256) the samples read, S x finer (2: a level-1 predict of a level-2 --vkey)\n");
        return 2;
    }
    const char *wraps = argv[2], *segs = argv[3], *out = argv[4];
    const char *ctkey = wopt(argc, argv, "--ct-key", nullptr), *maps = wopt(argc, argv, "--maps", nullptr), *ref = wopt(argc, argv, "--ref", nullptr),
               *nzs = wopt(argc, argv, "--nonzero", nullptr), *splits = wopt(argc, argv, "--split", "heldout"), *tabf = wopt(argc, argv, "--table", nullptr),
               *holdf = wopt(argc, argv, "--holdouts", "/vesuvius/ufsm/inkgt/holdouts-ink.json"), *need = wopt(argc, argv, "--need", nullptr);
    const double tol = opt_atof("--tol", wopt(argc, argv, "--tol", "40")), umL = opt_atof("--um-label", wopt(argc, argv, "--um-label", "2.399"));
    const int step = opt_atoi("--step", wopt(argc, argv, "--step", "8")), minn = opt_atoi("--min-n", wopt(argc, argv, "--min-n", "1000")),
              draws = opt_atoi("--draws", wopt(argc, argv, "--draws", "1000")), nth = opt_atoi("--threads", wopt(argc, argv, "--threads", "6"));
    const char *vmaps = wopt(argc, argv, "--vmap", nullptr), *vkey = wopt(argc, argv, "--vkey", "9.596"), *tiles = wopt(argc, argv, "--tiles-out", nullptr);
    const int depth = opt_atoi("--depth", wopt(argc, argv, "--depth", "4")), gate = opt_atoi("--gate", wopt(argc, argv, "--gate", "1"));
    const double maxabs = opt_atof("--max-absent", wopt(argc, argv, "--max-absent", "0.001"));
    if (!ctkey || (!maps && !vmaps && !tiles) || step < 1 || draws < 10 || nth < 1 || depth < 0 || depth > 64) {
        fprintf(stderr, "ink-wrapscore: --ct-key and --maps / --vmap / --tiles-out needed (--step >= 1, --draws >= 10, --depth 0..64)\n"); return 2; }
    if ((vmaps || tiles) && !need) { fprintf(stderr, "ink-wrapscore: --vmap / --tiles-out need --need (the wraps the volume maps cover)\n"); return 2; }
    if (holdf && *holdf) { struct stat st; if (stat(holdf, &st)) { fprintf(stderr, "ink-wrapscore: no holdouts file %s (--holdouts '' for none)\n", holdf); return 1; } }
    const double t0 = wnow(); wtable t = {0}; char key[KEYLEN]; table_key(key, sizeof key, wraps, segs, ctkey, holdf, step, umL);
    if (!tabf || table_load(&t, tabf, key)) {
        if (table_build(&t, wraps, segs, ctkey, holdf, step, umL, nth)) return 1;
        for (int k = 0; k < t.nw; k++) { free(t.W[k].X); free(t.W[k].Y); free(t.W[k].Z); t.W[k].X = t.W[k].Y = t.W[k].Z = nullptr; }
        if (tabf && table_save(&t, tabf, key)) fprintf(stderr, "ink-wrapscore: warning: cannot write the table %s\n", tabf);
    } else printf("table: %s: %zu samples on %d wraps\n", tabf, t.ns, t.nw);
    /* the samples to score: the requested splits within --tol */
    uint8_t *ev = calloc(t.ns, 1); char *scp0 = strdup(splits);
    for (char *sv = nullptr, *sp = strtok_r(scp0, ",", &sv); sp; sp = strtok_r(nullptr, ",", &sv)) {
        const int mode = !strcmp(sp, "heldout") ? 0 : !strcmp(sp, "rest") ? 1 : !strcmp(sp, "all") ? 2 : !strcmp(sp, "hold") ? 3 : !strcmp(sp, "valonly") ? 4 : -1;
        if (mode < 0) { fprintf(stderr, "ink-wrapscore: --split heldout | rest | all | hold | valonly\n"); return 2; }
        for (size_t s = 0; s < t.ns; s++) { const wsamp *x = &t.S[s];
            const int in = mode == 0 ? (x->val | x->hold) : mode == 1 ? (!x->val && !x->hold) : mode == 2 ? 1 : mode == 3 ? x->hold : (x->val && !x->hold);
            if (in && x->miss * umL <= tol) ev[s] = 1; } }
    free(scp0);
    /* the --need wraps' grids (render coordinates) for the volume maps / tile lists */
    uint8_t *inw = calloc(t.nw ? t.nw : 1, 1); rgrid *G = calloc(t.nw ? t.nw : 1, sizeof *G);
    if (vmaps || tiles) {
        if (need_set(need, &t, inw) <= 0) return 1;
        for (int k = 0; k < t.nw; k++) { if (!inw[k]) continue; char p[1400]; int w, h; snprintf(p, sizeof p, "%s/%s", wraps, t.W[k].name);
            if (load_xyz(p, &G[k].X, &G[k].Y, &G[k].Z, &w, &h) || w != t.W[k].w || h != t.W[k].h) { fprintf(stderr, "ink-wrapscore: cannot read the wrap %s (or its grid changed)\n", p); return 1; } }
    }
    if (tiles) {   /* latink.sh LATINK_TILES: the 512 tiles of grids 0 and 256 that hold a scored sample's pixel point (+- VMARG). --tiles-sub S
                      (2026-11-21, inkperf): the tiles of a predict S times finer than --vkey (S 2: score.sh FASTLV=1, level-1 tiles whose 2^3
                      mean is the scored level): exactly the children of the voxels read, not every child of the coarse tiles */
        const int tsub = opt_atoi("--tiles-sub", wopt(argc, argv, "--tiles-sub", "1"));
        if (tsub < 1 || tsub > 8 || (tsub & (tsub - 1))) { fprintf(stderr, "ink-wrapscore: --tiles-sub 1, 2, 4 or 8\n"); return 2; }
        size_t cap = 1 << 16, nt = 0; uint64_t *tk = malloc(cap * sizeof *tk); size_t npt = 0;
        for (size_t s = 0; s < t.ns; s++) { if (!ev[s] || !inw[t.S[s].k]) continue; const wwrap *w = &t.W[t.S[s].k]; int px, py; double P[3]; samp_px(w, &t.S[s], &px, &py);
            if (!px_point(G[t.S[s].k].Z, G[t.S[s].k].Y, G[t.S[s].k].X, w->w, w->h, 1 / w->sc, px, py, P)) continue; npt++;
            for (int g = 0; g < 2; g++) { const int64_t o = g ? 256 : 0; int64_t a[3], b[3];
                for (int d = 0; d < 3; d++) {   /* the --vkey voxels [lo, hi] read around P -> their tsub^3 children on the predict's level */
                    int64_t lo = (int64_t)floor(P[d] - VMARG), hi = (int64_t)floor(P[d] + VMARG);
                    if (tsub > 1) { lo = lo >> 4 << 4; hi = (hi >> 4 << 4) + 15; }   /* (whole 16^3 volcomp blocks: at q > 0 a voxel's decoded value depends on its
                                                                                       block, so every voxel of a read block keeps both grids; tsub 1 keeps the
                                                                                       lists the FAST scores were made with) */
                    lo *= tsub; hi = hi * tsub + tsub - 1;
                    a[d] = fdiv512(lo - o); b[d] = fdiv512(hi - o); }
                for (int64_t z = a[0]; z <= b[0]; z++) for (int64_t y = a[1]; y <= b[1]; y++) for (int64_t x = a[2]; x <= b[2]; x++) {
                    if (nt + 1 >= cap) { cap *= 2; tk = realloc(tk, cap * sizeof *tk); }
                    tk[nt++] = (uint64_t)g << 63 | (uint64_t)(z + 1024) << 42 | (uint64_t)(y + 1024) << 21 | (uint64_t)(x + 1024); } } }
        qsort(tk, nt, sizeof *tk, cmp_u64b); size_t u = 0; for (size_t i = 0; i < nt; i++) if (!i || tk[i] != tk[i - 1]) tk[u++] = tk[i];
        char cmd[1500]; snprintf(cmd, sizeof cmd, "%s", tiles); mkdir(cmd, 0775); int cnt[2] = {0, 0}; int bad = 0;
        for (int g = 0; g < 2; g++) { char p[1500], q[1600]; snprintf(p, sizeof p, "%s/tiles%d.txt", tiles, g ? 256 : 0); snprintf(q, sizeof q, "%s.tmp", p); FILE *f = fopen(q, "w"); if (!f) { bad = 1; continue; }
            for (size_t i = 0; i < u; i++) { if ((int)(tk[i] >> 63) != g) continue; fprintf(f, "%lld %lld %lld\n", (long long)(tk[i] >> 42 & 0x1fffff) - 1024, (long long)(tk[i] >> 21 & 0x1fffff) - 1024, (long long)(tk[i] & 0x1fffff) - 1024); cnt[g]++; }
            if (fclose(f) || rename(q, p)) bad = 1; }
        free(tk);
        if (bad || !npt) { fprintf(stderr, "ink-wrapscore: FAILED: tile lists in %s (%zu sample points)\n", tiles, npt); return 1; }
        printf("tiles: %zu sample points on the --need wraps -> %d tiles of grid 0, %d of grid 256 in %s\n", npt, cnt[0], cnt[1], tiles);
        if (!maps && !vmaps) return 0;
    }
    /* the maps */
    char *mcp = strdup(maps ? maps : ""); int nm = 0, nfm = 0; char *mn[16], *md[16];
    for (char *s = strtok(mcp, ","); s; s = strtok(nullptr, ",")) { char *e = strchr(s, '='); if (!e || nm == 16) { fprintf(stderr, "ink-wrapscore: --maps name=DIR,..\n"); return 2; } *e = 0; mn[nm] = s; md[nm++] = e + 1; }
    nfm = nm; char *vcp = strdup(vmaps ? vmaps : "");
    for (char *s = strtok(vcp, ","); s; s = strtok(nullptr, ",")) { char *e = strchr(s, '='); if (!e || nm == 16) { fprintf(stderr, "ink-wrapscore: --vmap name=DIR,..\n"); return 2; } *e = 0; mn[nm] = s; md[nm++] = e + 1; }
    int ri = -1; for (int m = 0; m < nm; m++) if (ref && !strcmp(mn[m], ref)) ri = m;
    if (ref && ri < 0) { fprintf(stderr, "ink-wrapscore: --ref %s is not one of --maps\n", ref); return 2; }
    int nzm[16]; for (int m = 0; m < nm; m++) { nzm[m] = !nzs; if (nzs) { char b[1024]; snprintf(b, sizeof b, ",%s,", nzs); char q[300]; snprintf(q, sizeof q, ",%s,", mn[m]); nzm[m] = strstr(b, q) != nullptr; } }
    int16_t *V = malloc((size_t)nm * t.ns * sizeof *V); for (size_t i = 0; i < (size_t)nm * t.ns; i++) V[i] = -1;
    int fail = 0;
    for (int m = nfm; m < nm && !fail; m++) {   /* the volume maps */
        const double tv = wnow(); z3 *zi = vol_open(md[m], "ink", vkey), *zg[2] = {gate ? vol_open(md[m], "c12", vkey) : nullptr, gate ? vol_open(md[m], "c13", vkey) : nullptr};
        if (!zi || (gate && (!zg[0] || !zg[1]))) { fprintf(stderr, "ink-wrapscore: vmap %s: need %s/ink.zarr%s (level %s)\n", mn[m], md[m], gate ? ", c12.zarr, c13.zarr" : "", vkey); fail = 1; break; }
        static uint8_t glut[256][256]; static int glut_ok; if (!glut_ok) { img_gate_lut(glut); glut_ok = 1; }   /* mgpu: main */
        size_t nall = 0, nabs = 0;
        for (int k = 0; k < t.nw && !fail; k++) {
            if (!inw[k]) continue; size_t n = 0; size_t *idx = malloc(t.ns * sizeof *idx);
            for (size_t s = 0; s < t.ns; s++) if (ev[s] && t.S[s].k == k) idx[n++] = s;
            uint8_t *o[3] = {malloc(n + 1), malloc(n + 1), malloc(n + 1)};
            for (int v = 0; v < (gate ? 3 : 1) && !fail; v++) { size_t ab = 0;
                if (vol_eval(v ? zg[v - 1] : zi, &t.W[k], &G[k], t.S, idx, n, depth, nth, o[v], &ab)) fail = 1;
                if (!v) nabs += ab; }
            for (size_t i = 0; i < n && !fail; i++) V[(size_t)m * t.ns + idx[i]] = gate ? glut[o[0][i]][o[1][i] > o[2][i] ? o[1][i] : o[2][i]] : o[0][i];
            nall += n; free(idx); free(o[0]); free(o[1]); free(o[2]);
        }
        z3_close(zi); if (zg[0]) z3_close(zg[0]); if (zg[1]) z3_close(zg[1]);
        printf("vmap: %s: %zu samples on the --need wraps from %s (depth %d%s), %zu in shards the ink store lacks (%.0f s)\n", mn[m], nall, md[m], depth, gate ? ", x max(c12, c13)" : "", nabs, wnow() - tv);
        if (!fail && (double)nabs > maxabs * (double)(nall ? nall : 1)) { fprintf(stderr, "ink-wrapscore: FAILED: vmap %s: %zu of %zu samples lie in shards missing from %s/ink.zarr (> --max-absent %g): a cropped / failed predict?\n", mn[m], nabs, nall, md[m], maxabs); fail = 1; }
    }
    for (int m = 0; m < nfm; m++) {
        struct stat st; if (stat(md[m], &st) || !S_ISDIR(st.st_mode)) { fprintf(stderr, "ink-wrapscore: map %s: no directory %s\n", mn[m], md[m]); fail = 1; continue; }
        int got = 0;
        for (int k = 0; k < t.nw; k++) {
            const wwrap *w = &t.W[k]; char p[1600] = ""; int have = 0;
            for (int e = 0; e < 2 && !have; e++) { snprintf(p, sizeof p, "%s/%s%s", md[m], w->name, e ? ".jpg" : ".tif"); have = !stat(p, &st); }
            if (!have) continue;
            int iw, ih; uint8_t *im = map_read(p, &iw, &ih); if (!im) { fprintf(stderr, "ink-wrapscore: map %s: cannot read %s\n", mn[m], p); fail = 1; continue; }
            const double cw = w->w / w->sc, ch = w->h / w->sc, fx = iw / cw, fy = ih / ch;
            if (fabs(fx - fy) > 0.02 * fx) { fprintf(stderr, "ink-wrapscore: map %s: %s %d x %d is not proportional to the canvas %.0f x %.0f\n", mn[m], p, iw, ih, cw, ch); free(im); fail = 1; continue; }
            for (size_t s = 0; s < t.ns; s++) { const wsamp *x = &t.S[s]; if (x->k != k) continue;
                int64_t xi = (int64_t)nearbyint(x->fc / w->sc * fx), yi = (int64_t)nearbyint(x->fr / w->sc * fy);
                xi = xi < 0 ? 0 : xi > iw - 1 ? iw - 1 : xi; yi = yi < 0 ? 0 : yi > ih - 1 ? ih - 1 : yi;
                V[(size_t)m * t.ns + s] = im[(size_t)yi * iw + xi]; }
            free(im); got++;
        }
        if (!got) { fprintf(stderr, "ink-wrapscore: map %s: no map of any wrap in %s\n", mn[m], md[m]); fail = 1; }
        if (need) {   /* every listed wrap must have a map */
            char nb[1024]; snprintf(nb, sizeof nb, "%s", need); int a, b, miss = 0;
            if (sscanf(nb, "w%d-w%d", &a, &b) == 2) { for (int q = a; q <= b; q++) { char p[1600]; snprintf(p, sizeof p, "%s/w%03d.tif", md[m], q); char p2[1600]; snprintf(p2, sizeof p2, "%s/w%03d.jpg", md[m], q);
                    if (stat(p, &st) && stat(p2, &st)) { fprintf(stderr, "ink-wrapscore: map %s: no w%03d in %s\n", mn[m], q, md[m]); miss = 1; } } }
            else for (char *s = strtok(nb, ","); s; s = strtok(nullptr, ",")) { char p[1600], p2[1600]; snprintf(p, sizeof p, "%s/%s.tif", md[m], s); snprintf(p2, sizeof p2, "%s/%s.jpg", md[m], s);
                    if (stat(p, &st) && stat(p2, &st)) { fprintf(stderr, "ink-wrapscore: map %s: no %s in %s\n", mn[m], s, md[m]); miss = 1; } }
            fail |= miss;
        }
        printf("maps: %s: %d wraps from %s\n", mn[m], got, md[m]);
    }
    if (fail) { fprintf(stderr, "ink-wrapscore: FAILED (missing / unreadable maps): no score written\n"); return 1; }
    /* the splits */
    char *scp = strdup(splits); int rc = 0;
    for (char *sp = strtok(scp, ","); sp; sp = strtok(nullptr, ",")) {
        int mode = !strcmp(sp, "heldout") ? 0 : !strcmp(sp, "rest") ? 1 : !strcmp(sp, "all") ? 2 : !strcmp(sp, "hold") ? 3 : !strcmp(sp, "valonly") ? 4 : -1;
        if (mode < 0) { fprintf(stderr, "ink-wrapscore: --split heldout | rest | all | hold | valonly\n"); return 2; }
        size_t nheld = 0, nok = 0, nink = 0; uint8_t *ok = calloc(t.ns, 1);
        for (size_t s = 0; s < t.ns; s++) { const wsamp *x = &t.S[s];
            const int in = mode == 0 ? (x->val | x->hold) : mode == 1 ? (!x->val && !x->hold) : mode == 2 ? 1 : mode == 3 ? x->hold : (x->val && !x->hold);
            if (!in || !(x->miss * umL <= tol)) continue; nheld++;
            int o = 1; for (int m = 0; m < nm; m++) { const int v = V[(size_t)m * t.ns + s]; o &= nzm[m] ? v > 0 : v >= 0; }
            if (o) { ok[s] = 1; nok++; nink += x->lab; } }
        printf("%s samples within %g um: %zu, with every map valid: %zu (%.3f), ink %.3f\n", sp, tol, nheld, nok, (double)nok / (nheld ? nheld : 1), nok ? (double)nink / nok : NAN);
        if ((int64_t)nok < minn || !nink || nink == nok) { fprintf(stderr, "ink-wrapscore: FAILED: split %s has %zu valid samples (%zu ink; --min-n %d): no score written\n", sp, nok, nink, minn); free(ok); rc = 1; continue; }
        /* blocks -> dense ids, per block and map the 8-bit histograms */
        uint64_t *bk = malloc(nok * sizeof *bk); size_t q = 0; for (size_t s = 0; s < t.ns; s++) if (ok[s]) bk[q++] = t.S[s].blk;
        qsort(bk, nok, sizeof *bk, cmp_u64); size_t nb = 0; for (size_t i = 0; i < nok; i++) if (!i || bk[i] != bk[i - 1]) bk[nb++] = bk[i];
        uint32_t *bh = calloc(nb * nm * 512, sizeof *bh); double base[16], hits[16], hink[16], tink[16]; double *H = calloc((size_t)nm * 512, sizeof *H);
        for (int m = 0; m < nm; m++) hits[m] = hink[m] = tink[m] = 0;
        for (size_t s = 0; s < t.ns; s++) { if (!ok[s]) continue; const wsamp *x = &t.S[s]; size_t lo = 0, hi = nb; while (lo + 1 < hi) { const size_t mid = (lo + hi) / 2; if (bk[mid] <= x->blk) lo = mid; else hi = mid; }
            for (int m = 0; m < nm; m++) { const int v = V[(size_t)m * t.ns + s]; bh[(lo * nm + m) * 512 + (x->lab ? 0 : 256) + v]++; H[(size_t)m * 512 + (x->lab ? 0 : 256) + v] += 1;
                if (v >= 254) { hits[m]++; hink[m] += x->lab; } if (x->lab && v >= 254) tink[m]++; } }
        for (int m = 0; m < nm; m++) { base[m] = wauc(H + (size_t)m * 512, H + (size_t)m * 512 + 256); if (!isfinite(base[m])) { fprintf(stderr, "ink-wrapscore: FAILED: %s AUC nan on %s\n", mn[m], sp); rc = 1; } }
        double *dl = nullptr;
        if (ri >= 0 && !rc) {
            dl = malloc((size_t)nm * draws * sizeof *dl); const int nt = nth > 32 ? 32 : nth; pthread_t th[32]; bjob J[32];
            for (int i = 0; i < nt; i++) { J[i] = (bjob){nm, (int)nb, ri, draws, draws * i / nt, draws * (i + 1) / nt, bh, dl}; pthread_create(&th[i], nullptr, boot_worker, &J[i]); }
            for (int i = 0; i < nt; i++) pthread_join(th[i], nullptr);
        }
        if (!rc) {
            char path[1400]; snprintf(path, sizeof path, "%s.%s.tsv", out, sp); char tmp[1500]; snprintf(tmp, sizeof tmp, "%s.tmp", path); FILE *f = fopen(tmp, "w");
            if (!f) { fprintf(stderr, "ink-wrapscore: cannot write %s\n", path); rc = 1; }
            else {
                char row[512]; snprintf(row, sizeof row, "model\tn\tauc\tdelta_vs_ref\tlo95\thi95\tink_share_at_254\trecall_254\n"); fputs(row, f); fputs(row, stdout);
                for (int m = 0; m < nm; m++) {
                    double d = NAN, lo = NAN, hi = NAN;
                    if (ri >= 0 && m != ri) { d = base[m] - base[ri]; double *v = dl + (size_t)m * draws; lo = pctl(v, draws, 2.5); hi = pctl(v, draws, 97.5); }
                    snprintf(row, sizeof row, "%s\t%zu\t%.4f\t%+.4f\t%+.4f\t%+.4f\t%.3f\t%.3f\n", mn[m], nok, base[m], d, lo, hi, hits[m] ? hink[m] / hits[m] : NAN, (double)tink[m] / nink);
                    fputs(row, f); fputs(row, stdout);
                }
                if (fclose(f) || rename(tmp, path)) { fprintf(stderr, "ink-wrapscore: cannot write %s\n", path); rc = 1; }
            }
        }
        free(dl); free(bh); free(H); free(bk); free(ok);
    }
    printf("ink-wrapscore: %.1f s\n", wnow() - t0);
    for (int k = 0; k < t.nw; k++) { free(G[k].X); free(G[k].Y); free(G[k].Z); }
    free(scp); free(mcp); free(vcp); free(V); free(t.W); free(t.S); free(ev); free(inw); free(G);
    return rc;
}
