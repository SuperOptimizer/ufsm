/* ufsm recto-ruler <model-or-set> <fields-dir> --codes ROOT KEY [--src NAME ROOT KEY UM]... [--map NAME REG.json] [--ct ROOT KEY --ct-um U]
                    [--nz 16 | --zs Z,Z,...] [--zr Z0,Z1] [--nth 180] [--step-um U] [--sigma F] [--prom P] [--sep-um S] [--win-um 60]
                    [--thr 128] [--ens NAME OP A,B,..] [--dump FILE --dump-src NAME] [--cls MODEL] [--threads N] [--cache DIR]
   Recto (sheet surface) predictions as a winding ruler, measured against GT (winding_mod14 band codes) on radial rays from the
   umbilicus at native z levels (wrap-check's geometry). Every source is sampled at its own resolution (the source rows covering the
   native z averaged, bilinear in the plane) on rays stepped at --step-um (default: the finest source's voxel), then
     - pk: peaks of the profile smoothed by a gaussian of --sigma x the source voxel: local maxima with prominence >= --prom (0..255
       units; the deeper of the two side minima within +-win-um decides, as topographic prominence), closer than --sep-um merged
       (higher kept), sub-sample (parabola) positions; binary masks give their run centres the same way;
     - run: runs of the raw profile >= --thr (the TRACE ruler);
   per GT interval (consecutive GT sheets on a ray, dq turns, inside material): hits in [a + g/2, b + g/2) (the sheet at b), turns per GT
   turn and the share off by >= 0.5 turn, by radius, normal gap per GT turn and network spacing; beside the fit (model W), the network
   density (integral |n.r| / sp) and, with --ct, CT peaks (wrap-check's detector at native resolution).
   --map NAME REG.json: the source NAME is on another scan: ufsm reg map (fine = that scan, coarse = the fields' scan at its L0 um
   "coarse.um"); its samples are read through fine(c) = A^-1 (c + D(c)).
   --src-box NAME Z0,Y0,X0,Z1,Y1,X1: the source's valid box (its voxels; partial predictions): NaN outside, its columns only on GT
   intervals sampled inside it.
   --ens NAME OP A,B: a source from the samples of A, B (OP mean | max | gmean | norm = each member scaled to its 99th percentile along
   the ray, then the mean), NaN where any member is.
   --vote A,B,..: column vote = the median over those sources' pk turns per GT interval (a per-interval consensus).
   --dump FILE --dump-src NAME: per interval between consecutive pk hits of NAME inside a GT-covered stretch, the feature vector
   (rr_feat) and the GT turns (Q(b) - Q(a), Q piecewise linear between GT sheet centres) -> FILE (float32 records, header "RRD1").
   --cls MODEL (ufsm recto-cls train): the classifier's turns per hit interval of --dump-src summed along the ray; columns cls (argmax)
   and clsE (expectation), Q from linear interpolation between hits. CPU, rays on --threads threads. */
#include "fit.h"
#include "rectoruler.h"
#include "sysinfo.h"
#include "zarr2.h"
#include "zarr3.h"
#include "store.h"
#include "reg.h"
/* (weak: the test binary links neither src/reg_cli.c nor src/imgops.c; ufsm has both) */
__attribute__((weak)) const rmap *reg_map_open(const char *path, const char *cache, rvol **fv, double *coarse_um, double *fine_um);
__attribute__((weak)) int jpeg_write_rgb(const char *path, int w, int h, const uint8_t *rgb, int quality);
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdatomic.h>

enum { RR_MAXS = 8 };
static const char *rr_opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 3; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}
static double rr_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int cmp_f(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }

typedef struct {
    char name[64]; z3 *z; z2 *z2v; double um; int64_t shape[3];
    const rmap *map; double cum, lf;   /* registered source (another scan): the map, its coarse L0 um, fine L0 voxels per source voxel */
    int ens; int ea[RR_MAXS], ne;      /* --ens: op (1 mean, 2 max), the member sources */
    /* the plane for this z level: uint8 averaged rows; origin (source voxels, global yx) and size; registered: a volume block */
    uint8_t *pl; int64_t po[3], pn[3];
    int hasbox; double vb[6];   /* --src-box NAME Z0,Y0,X0,Z1,Y1,X1: the source's valid box (source voxels); NaN outside */
} rr_src;

typedef struct {
    const fit_set *fs; const fit_model *m; double zl, step, um, h, cf, morg[3];
    const uint8_t *w, *sp, *pc, *ps, *tyy, *txx, *tyx; int FH, FW; double fo[2];
    const float *ct; int64_t co[2], cn[2]; double F;
    const uint8_t *cd; int64_t qo[2], qn[2];
    rr_src *S; int ns; double zg;   /* global native z */
    int nth; double sigma, prom, sep, win, thr, ctsig, ctprom;
    int dsrc; const rr_cls *cls; int vote[RR_MAXS], nvote;
    rr_iv **out; int *nout; float **dump; int *ndump;
} rr_job;

static double rr_plane(const uint8_t *p, const int64_t o[2], const int64_t n[2], double cy, double cx) {   /* source voxel coords (global) */
    cy -= o[0]; cx -= o[1];
    const int64_t y0 = (int64_t)floor(cy), x0 = (int64_t)floor(cx); const double ty = cy - y0, tx = cx - x0;
    if (y0 < 0 || x0 < 0 || y0 + 1 >= n[0] || x0 + 1 >= n[1]) return NAN;
    const uint8_t *q = p + y0 * n[1] + x0;
    return (1 - ty) * ((1 - tx) * q[0] + tx * q[1]) + ty * ((1 - tx) * q[n[1]] + tx * q[n[1] + 1]);
}
static double rr_plane8f(const float *p, const int64_t o[2], const int64_t n[2], double cy, double cx) {
    cy -= o[0]; cx -= o[1];
    const int64_t y0 = (int64_t)floor(cy), x0 = (int64_t)floor(cx); const double ty = cy - y0, tx = cx - x0;
    if (y0 < 0 || x0 < 0 || y0 + 1 >= n[0] || x0 + 1 >= n[1]) return NAN;
    const float *q = p + y0 * n[1] + x0;
    return (1 - ty) * ((1 - tx) * q[0] + tx * q[1]) + ty * ((1 - tx) * q[n[1]] + tx * q[n[1] + 1]);
}
static double rr_sample(const rr_job *J, const rr_src *s, double y, double x) {   /* box-local native (y, x) at J->zg */
    const double gy = y + J->morg[1], gx = x + J->morg[2];
    if (!s->map) { const double RF = J->um / s->um; const int64_t o2[2] = {s->po[1], s->po[2]}, n2[2] = {s->pn[1], s->pn[2]};
                   const double sy = (gy + 0.5) * RF - 0.5, sx = (gx + 0.5) * RF - 0.5;
                   if (s->hasbox) { const double sz = (J->zg + 0.5) * RF - 0.5; if (sz < s->vb[0] || sy < s->vb[1] || sx < s->vb[2] || sz >= s->vb[3] || sy >= s->vb[4] || sx >= s->vb[5]) return NAN; }
                   return rr_plane(s->pl, o2, n2, sy, sx); }
    const double K = J->um / s->cum;   /* native -> coarse L0 voxels */
    const double c[3] = {(gx + 0.5) * K - 0.5, (gy + 0.5) * K - 0.5, (J->zg + 0.5) * K - 0.5}; double f[3]; rmap_c2f(s->map, c, f);
    const double L = s->lf;
    if (s->hasbox) { const double gz = (f[2] + 0.5) / L - 0.5, gyy = (f[1] + 0.5) / L - 0.5, gxx = (f[0] + 0.5) / L - 0.5;
                     if (gz < s->vb[0] || gyy < s->vb[1] || gxx < s->vb[2] || gz >= s->vb[3] || gyy >= s->vb[4] || gxx >= s->vb[5]) return NAN; }
    const double fz = (f[2] + 0.5) / L - 0.5 - s->po[0], fy = (f[1] + 0.5) / L - 0.5 - s->po[1], fx = (f[0] + 0.5) / L - 0.5 - s->po[2];
    const int64_t z0 = (int64_t)floor(fz), y0 = (int64_t)floor(fy), x0 = (int64_t)floor(fx); const double tz = fz - z0, ty = fy - y0, tx = fx - x0;
    if (z0 < 0 || y0 < 0 || x0 < 0 || z0 + 1 >= s->pn[0] || y0 + 1 >= s->pn[1] || x0 + 1 >= s->pn[2]) return NAN;
    const size_t sy = s->pn[2], sz = (size_t)s->pn[1] * s->pn[2]; const uint8_t *q = s->pl + z0 * sz + y0 * sy + x0;
    const double a = (1 - ty) * ((1 - tx) * q[0] + tx * q[1]) + ty * ((1 - tx) * q[sy] + tx * q[sy + 1]);
    const double b = (1 - ty) * ((1 - tx) * q[sz] + tx * q[sz + 1]) + ty * ((1 - tx) * q[sz + sy] + tx * q[sz + sy + 1]);
    return (1 - tz) * a + tz * b;
}

/* peaks of v (n samples, step st native) inside material: smoothed by a gaussian of sig samples, prominence >= prom within +-W
   samples, merged closer than sep samples; positions (samples, sub-sample), heights, prominences, half-prominence widths (samples) */
int rr_peaks(const float *v, const uint8_t *mat, int n, double sig, double prom, int W, double sep, rr_hit *out, float *tmp) {
    const int R = sig > 0 ? (int)ceil(3 * sig) : 0;
    for (int i = 0; i < n; i++) {
        if (!R) { tmp[i] = v[i]; continue; }
        double s = 0, ws = 0;
        for (int d = -R; d <= R; d++) { const int k = i + d; if (k < 0 || k >= n || !(v[k] == v[k])) continue; const double g = exp(-0.5 * (d / sig) * (d / sig)); s += g * v[k]; ws += g; }
        tmp[i] = ws > 0 && v[i] == v[i] ? (float)(s / ws) : NAN;
    }
    int k = 0;
    for (int i = 1; i + 1 < n; i++) {
        if (!mat[i] || !(tmp[i] == tmp[i]) || !(tmp[i] > tmp[i - 1])) continue;
        int e = i; while (e + 1 < n && tmp[e + 1] == tmp[i]) e++;   /* plateau [i, e] */
        if (e + 1 >= n || !(tmp[e + 1] < tmp[i])) { i = e; continue; }
        double ml = tmp[i], mr = tmp[i];
        for (int q = i - 1; q >= 0 && q >= i - W; q--) { if (!(tmp[q] == tmp[q])) break; if (tmp[q] > tmp[i]) break; if (tmp[q] < ml) ml = tmp[q]; }
        for (int q = e + 1; q < n && q <= e + W; q++) { if (!(tmp[q] == tmp[q])) break; if (tmp[q] > tmp[i]) break; if (tmp[q] < mr) mr = tmp[q]; }
        const double pr = tmp[i] - fmax(ml, mr);
        if (pr >= prom) {
            double pos = 0.5 * (i + e);
            if (e == i) { const double a = tmp[i - 1], b = tmp[i], c = tmp[i + 1], den = a - 2 * b + c; if (den < 0) { const double d = 0.5 * (a - c) / den; if (fabs(d) < 1) pos = i + d; } }
            const double half = tmp[i] - 0.5 * pr; int l = i, r = e;
            while (l > 0 && tmp[l - 1] == tmp[l - 1] && tmp[l - 1] > half) l--;
            while (r + 1 < n && tmp[r + 1] == tmp[r + 1] && tmp[r + 1] > half) r++;
            out[k++] = (rr_hit){(float)pos, tmp[i], (float)pr, (float)(r - l + 1)};
        }
        i = e;
    }
    /* merge closer than sep: keep the higher of two neighbours, repeatedly */
    if (sep > 0) {
        int again = 1;
        while (again && k > 1) {
            again = 0; int w = 0;
            for (int i = 0; i < k; i++) {
                if (w > 0 && out[i].pos - out[w - 1].pos < sep) { if (out[i].h > out[w - 1].h) out[w - 1] = out[i]; again = 1; continue; }
                out[w++] = out[i];
            }
            k = w;
        }
    }
    return k;
}
static int rr_runs(const float *v, const uint8_t *mat, int n, double thr, float *out) {
    int k = 0;
    for (int i = 0; i < n;) {
        if (!(v[i] >= thr) || !mat[i]) { i++; continue; }
        int e = i; while (e < n && v[e] >= thr && mat[e]) e++;
        out[k++] = (float)(0.5 * (i + e - 1)); i = e;
    }
    return k;
}
static int rr_count(const float *c, int nc, double a, double b) { int k = 0; for (int i = 0; i < nc; i++) k += c[i] >= a && c[i] < b; return k; }
static int rr_count_hits(const rr_hit *c, int nc, double a, double b) { int k = 0; for (int i = 0; i < nc; i++) k += c[i].pos >= a && c[i].pos < b; return k; }
/* hits as a ruler, offset-free: the cumulative hit index, linear between consecutive hits; turns over [a, b] = C(b) - C(a) (NaN outside the hits) */
static double rr_lin(const float *p, int stride, int n, double a, double b) {
    double C[2]; const double r[2] = {a, b};
    for (int e = 0; e < 2; e++) {
        int lo = 0, hi = n;   /* first hit > r */
        while (lo < hi) { const int m = (lo + hi) / 2; if (p[(size_t)m * stride] > r[e]) hi = m; else lo = m + 1; }
        if (lo == 0 || lo >= n) return NAN;
        const double pa = p[(size_t)(lo - 1) * stride], pb = p[(size_t)lo * stride]; C[e] = lo - 1 + (r[e] - pa) / (pb - pa);
    }
    return C[1] - C[0];
}

/* the feature vector of the interval between hits j and j + 1 of the dump source (positions in samples) */
static void rr_feat(const rr_job *J, const rr_hit *hp, int nh, int j, const float *V, const float *spn, const float *nr, const float *dn,
                    const float *ph, const float *C, const int *ctpk, int nct, rr_hit *const *oh, const int *noh, double rad0, float *f) {
    const double st = J->step, a = hp[j].pos, b = hp[j + 1].pos; const int ia = (int)ceil(a), ib = (int)floor(b), im = (int)lround(0.5 * (a + b));
    for (int k = 0; k < RR_NF; k++) f[k] = NAN;
    const double nrm = nr[im] == nr[im] && nr[im] > 0.05 ? nr[im] : NAN;
    const double wu = (b - a) * st * J->um * (nrm == nrm ? nrm : 1);
    f[0] = (float)wu;
    f[1] = spn[im] > 0 ? (float)(wu / (spn[im] * J->um)) : NAN;
    {   float wd[9]; int nw = 0;
        for (int q = j - 4; q <= j + 4; q++) { if (q < 0 || q + 1 >= nh || q == j) continue; const int qm = (int)lround(0.5 * (hp[q].pos + hp[q + 1].pos)); const double nq = nr[qm] == nr[qm] && nr[qm] > 0.05 ? nr[qm] : 1; wd[nw++] = (float)((hp[q + 1].pos - hp[q].pos) * st * J->um * nq); }
        if (nw >= 2) { qsort(wd, nw, sizeof(float), cmp_f); f[2] = (float)(wu / wd[nw / 2]); } }
    {   double sdn = 0; int ndn = 0, n = 0; double sph = 0; int nph = 0, lp = -1; double vmin = 1e9, vs = 0; int nv = 0;
        for (int i = ia; i <= ib; i++) { n++; if (dn[i] == dn[i]) { sdn += dn[i] * st; ndn++; }
            if (ph[i] == ph[i]) { if (lp >= 0 && i - lp <= 3) { double d = ph[i] - ph[lp]; d -= floor(d + 0.5); sph += d; nph += i - lp; } lp = i; }
            if (V[i] == V[i]) { if (V[i] < vmin) vmin = V[i]; vs += V[i]; nv++; } }
        if (n > 0 && ndn >= 0.8 * n) f[3] = (float)(sdn * (b - a) / (ndn * st) * st);
        if (n > 1 && nph >= 0.8 * (n - 1)) f[4] = (float)(sph * (b - a) / nph);
        if (nv) { f[5] = (float)(vmin / 255); f[6] = (float)(vs / nv / 255); } }
    f[7] = hp[j].h / 255; f[8] = hp[j + 1].h / 255; f[9] = hp[j].prom / 255; f[10] = hp[j + 1].prom / 255;
    f[11] = (float)(hp[j].wid * st * J->um * (nrm == nrm ? nrm : 1)); f[12] = (float)(hp[j + 1].wid * st * J->um * (nrm == nrm ? nrm : 1));
    if (C) { double s = 0, mn = 1e9, mx = -1e9; int c = 0; for (int i = ia; i <= ib; i++) if (C[i] == C[i]) { s += C[i]; c++; if (C[i] < mn) mn = C[i]; if (C[i] > mx) mx = C[i]; }
             if (c) { f[13] = (float)(s / c / 255); f[14] = (float)(mn / 255); f[15] = (float)(mx / 255); }
             int c0 = 0; for (int p = 0; p < nct; p++) c0 += ctpk[p] > a && ctpk[p] < b; f[16] = (float)c0; }
    if (j > 0) { const int qm = (int)lround(0.5 * (hp[j - 1].pos + hp[j].pos)); if (spn[qm] > 0) f[17] = (float)((hp[j].pos - hp[j - 1].pos) * st * (nr[qm] == nr[qm] && nr[qm] > .05 ? nr[qm] : 1) / spn[qm]); }
    if (j + 2 < nh) { const int qm = (int)lround(0.5 * (hp[j + 1].pos + hp[j + 2].pos)); if (spn[qm] > 0) f[18] = (float)((hp[j + 2].pos - hp[j + 1].pos) * st * (nr[qm] == nr[qm] && nr[qm] > .05 ? nr[qm] : 1) / spn[qm]); }
    f[19] = (float)((rad0 + 0.5 * (a + b) * st) * J->um / 1000);
    f[20] = spn[im] > 0 ? (float)(spn[im] * J->um) : NAN;
    f[21] = (float)(nrm == nrm ? nrm : NAN);
    for (int s = 0, o = 0; s < J->ns && o < 3; s++) { if (s == J->dsrc) continue;   /* other sources' hits strictly inside (a + sep/2, b - sep/2) */
        if (oh[s]) { int c0 = 0; const double m = 0.25 * J->sep; for (int p = 0; p < noh[s]; p++) c0 += oh[s][p].pos > a + m && oh[s][p].pos < b - m; f[22 + o] = (float)c0; }
        o++; }
}

static void rr_rays(void *ctx, int a0, int a1, int t) {
    rr_job *J = ctx; (void)t;
    const fit_model *m = J->m; const int na = (int)(m->D * m->h) + 1; const int iz = (int)fmin(J->zl, na - 1);
    const double cy = m->cy[iz], cx = m->cx[iz], Y = m->H * m->h, X = m->W * m->h, st = J->step;
    for (int j = a0; j < a1; j++) {
        const double th = 2 * M_PI * (j + 0.5) / J->nth, sn = sin(th), cs = cos(th);
        double rmax = 0; for (double r = 0;; r += 16) { const double y = cy + r * sn, x = cx + r * cs; if (r > 0 && (y < 0 || x < 0 || y >= Y || x >= X)) break; rmax = r + 16; if (r > 4 * (Y + X)) break; }
        const int n = (int)(rmax / st) + 1;
        float *W = malloc(sizeof(float) * n), *spn = malloc(sizeof(float) * n), *nr = malloc(sizeof(float) * n), *dn = malloc(sizeof(float) * n), *ph = malloc(sizeof(float) * n), *C = J->ct ? malloc(sizeof(float) * n) : nullptr;
        uint8_t *mat = calloc(n, 1); short *q = calloc(n, sizeof(short));
        float *V[RR_MAXS] = {0}; for (int s = 0; s < J->ns; s++) V[s] = malloc(sizeof(float) * n);
        float *tmp = malloc(sizeof(float) * n);
        for (int i = 0; i < n; i++) {
            const double r = i * st, y = cy + r * sn, x = cx + r * cs;
            W[i] = NAN; spn[i] = 0; nr[i] = NAN; dn[i] = NAN; ph[i] = NAN; if (C) C[i] = NAN; for (int s = 0; s < J->ns; s++) V[s][i] = NAN;
            if (y < 0 || x < 0 || y >= Y || x >= X) continue;
            const int fy = (int)floor((y + J->fo[0]) / J->h), fx = (int)floor((x + J->fo[1]) / J->h);
            if (fy < 0 || fx < 0 || fy >= J->FH || fx >= J->FW) continue;
            const size_t ci = (size_t)fy * J->FW + fx;
            mat[i] = J->w[ci] != 0; spn[i] = J->sp[ci] * J->h / 8.0f;
            { const int a = J->pc[ci] - 128, b = J->ps[ci] - 128; if (a * a + b * b >= 16 * 16) ph[i] = (float)(atan2(b, a) / (2 * M_PI)); }
            { const double nyy = J->tyy[ci] / 255.0, nxx = J->txx[ci] / 255.0, nyx = J->tyx[ci] / 255.0 - 0.5;
              const double nr2 = sn * sn * nyy + cs * cs * nxx + 2 * sn * cs * nyx; nr[i] = (float)sqrt(fmax(0, nr2)); if (spn[i] > 0) dn[i] = (float)(nr[i] / spn[i]); }
            W[i] = (float)fit_set_W(J->fs, J->zl, y, x);
            if (J->cd) { const int64_t qy = (int64_t)floor((y + J->morg[1]) / J->cf) - J->qo[0], qx = (int64_t)floor((x + J->morg[2]) / J->cf) - J->qo[1];
                         if (qy >= 0 && qx >= 0 && qy < J->qn[0] && qx < J->qn[1]) { const int c = J->cd[qy * J->qn[1] + qx]; q[i] = (short)(c >= 1 && c <= 252 ? c : 0); } }
            for (int s = 0; s < J->ns; s++) if (!J->S[s].ens) V[s][i] = (float)rr_sample(J, &J->S[s], y, x);
        }
        if (C) for (int i = 0; i < n; i++) {   /* CT (float plane) bilinear */
            const double r = i * st, y = cy + r * sn, x = cx + r * cs;
            const double ccy = (y + J->morg[1] + 0.5) * J->F - 0.5 - J->co[0], ccx = (x + J->morg[2] + 0.5) * J->F - 0.5 - J->co[1];
            const int64_t y0 = (int64_t)floor(ccy), x0 = (int64_t)floor(ccx); const double ty = ccy - y0, tx = ccx - x0;
            if (y0 < 0 || x0 < 0 || y0 + 1 >= J->cn[0] || x0 + 1 >= J->cn[1]) { C[i] = NAN; continue; }
            const float *p = J->ct + y0 * J->cn[1] + x0;
            C[i] = (float)((1 - ty) * ((1 - tx) * p[0] + tx * p[1]) + ty * ((1 - tx) * p[J->cn[1]] + tx * p[J->cn[1] + 1]));
        }
        for (int s = 0; s < J->ns; s++) if (J->S[s].ens) {   /* ensembles: 1 mean, 2 max, 3 geometric mean, 4 mean of the members each scaled
                                                                  to its own 99th percentile along the ray (rank-free normalisation); NaN when any member is NaN */
            double sc[RR_MAXS]; for (int e = 0; e < J->S[s].ne; e++) sc[e] = 1;
            if (J->S[s].ens == 4) for (int e = 0; e < J->S[s].ne; e++) {
                const float *v = V[J->S[s].ea[e]]; int m = 0; for (int i = 0; i < n; i++) if (v[i] == v[i] && mat[i]) tmp[m++] = v[i];
                if (m > 10) { qsort(tmp, m, sizeof(float), cmp_f); const double q = tmp[(int)(0.99 * (m - 1))]; sc[e] = q > 1 ? 255.0 / q : 1; } }
            for (int i = 0; i < n; i++) {
                double acc = J->S[s].ens == 2 ? -1 : J->S[s].ens == 3 ? 0 : 0; int c = 0, bad = 0;
                for (int e = 0; e < J->S[s].ne; e++) { const float v = V[J->S[s].ea[e]][i]; if (!(v == v)) { bad = 1; continue; }
                    const double w = fmin(255, v * sc[e]);
                    if (J->S[s].ens == 2) acc = fmax(acc, w); else if (J->S[s].ens == 3) acc += log(w + 1); else acc += w; c++; }
                V[s][i] = c && !bad ? (float)(J->S[s].ens == 2 ? acc : J->S[s].ens == 3 ? exp(acc / c) - 1 : acc / c) : NAN;
            }
        }
        /* hits per source */
        rr_hit *H[RR_MAXS] = {0}; int nH[RR_MAXS] = {0}; float *R[RR_MAXS] = {0}; int nR[RR_MAXS] = {0};
        const int Wn = (int)lround(J->win / (st * J->um));
        for (int s = 0; s < J->ns; s++) {
            H[s] = malloc(sizeof(rr_hit) * (n / 2 + 1)); R[s] = malloc(sizeof(float) * (n / 2 + 1));
            const double sig = J->sigma * J->S[s].um / (st * J->um);
            nH[s] = rr_peaks(V[s], mat, n, sig, J->prom, Wn, J->sep / (st * J->um), H[s], tmp);
            nR[s] = rr_runs(V[s], mat, n, J->thr, R[s]);
        }
        /* CT peaks at native resolution (wrap-check's detector) */
        int *ctpk = nullptr, nct = 0;
        if (C) {
            const double sg = J->ctsig / st; const int RR = (int)ceil(3 * sg); float *Cs = tmp;
            for (int i = 0; i < n; i++) { double s = 0, ws = 0; for (int d = -RR; d <= RR; d++) { const int k = i + d; if (k < 0 || k >= n || !(C[k] == C[k])) continue; const double g = exp(-0.5 * (d / sg) * (d / sg)); s += g * C[k]; ws += g; } Cs[i] = ws > 0 ? (float)(s / ws) : NAN; }
            const int L = (int)lround(12 / st), P = (int)lround(24 / st); ctpk = malloc(sizeof(int) * (n / 2 + 1));
            for (int i = 1; i + 1 < n; i++) {
                if (!mat[i] || !(Cs[i] == Cs[i]) || !(Cs[i] > Cs[i - 1]) || Cs[i] < Cs[i + 1]) continue;
                const int a = i - L < 0 ? 0 : i - L, b = i + L + 1 > n ? n : i + L + 1; double s1 = 0, s2 = 0; int c = 0;
                for (int k = a; k < b; k++) if (Cs[k] == Cs[k]) { s1 += Cs[k]; s2 += (double)Cs[k] * Cs[k]; c++; }
                if (c < 4) continue;
                const double mu = s1 / c, sd = sqrt(fmax(0, s2 / c - mu * mu)); if (!(sd > 0)) continue;
                double ml = Cs[i], mr = Cs[i];
                for (int k = i - 1; k >= 0 && k >= i - P; k--) { if (!(Cs[k] == Cs[k]) || Cs[k] > Cs[i]) break; if (Cs[k] < ml) ml = Cs[k]; }
                for (int k = i + 1; k < n && k <= i + P; k++) { if (!(Cs[k] == Cs[k]) || Cs[k] >= Cs[i]) break; if (Cs[k] < mr) mr = Cs[k]; }
                if (Cs[i] - fmax(ml, mr) >= J->ctprom * sd) ctpk[nct++] = i;
            }
        }
        /* GT sheets: runs of labelled samples (codes within half a turn of the run's first) */
        int ng = 0, capg = 64; double *gr = malloc(sizeof(double) * capg); int *gq = malloc(sizeof(int) * capg);
        for (int i = 0; i < n;) {
            if (!q[i]) { i++; continue; }
            int e = i, last = i;
            while (e < n && e - last <= (int)lround(2 / st)) { if (q[e]) { int dc = (q[e] - q[i]) % 252; if (dc < 0) dc += 252; if (dc > 126) dc -= 252; if (abs(dc) > 9) break; last = e; } e++; }
            if (e < n && e - last <= (int)lround(2 / st)) e = last + 1;
            const double c0 = 0.5 * (i + last); const int qc = q[(int)lround(c0)] ? q[(int)lround(c0)] : q[i];
            if (ng == capg) { capg *= 2; gr = realloc(gr, sizeof(double) * capg); gq = realloc(gq, sizeof(int) * capg); }
            gr[ng] = c0; gq[ng] = qc; ng++; i = e > i ? e : i + 1;
        }
        /* consecutive GT sheets -> intervals; stretches with cumulative Q */
        double *Q = malloc(sizeof(double) * (ng + 1)); int *sid = malloc(sizeof(int) * (ng + 1)); int cur = -1;
        int no = 0, capo = 64; rr_iv *o = malloc(sizeof(rr_iv) * capo);
        for (int g = 0; g < ng; g++) {
            int okp = 0; double dq = 0;
            if (g > 0) {
                dq = fmod((gq[g] - gq[g - 1]) / 18.0, 14.0); if (dq < 0) dq += 14; if (dq > 7) dq -= 14;
                const int ia = (int)lround(gr[g - 1]), ib = (int)lround(gr[g]);
                okp = dq > 0.5 && dq < 2.5 && ib > ia; for (int k = ia; okp && k <= ib; k++) if (!mat[k]) okp = 0;
                if (okp && !(W[ia] == W[ia] && W[ib] == W[ib])) okp = 0;
                if (okp) {
                    const double ra = gr[g - 1], rb = gr[g], rm = 0.5 * (ra + rb); const int im = (int)lround(rm);
                    rr_iv v; memset(&v, 0, sizeof v); v.z = (float)J->zl; v.ray = j; v.r = (float)(rm * st); v.gtq = (float)dq;
                    v.gapn = (float)((rb - ra) * st * (nr[im] == nr[im] ? nr[im] : 1) * J->um / dq); v.sp = spn[im] > 0 ? (float)(spn[im] * J->um) : NAN;
                    for (int k = 0; k < RR_NCOL; k++) v.t[k] = NAN;
                    v.t[0] = W[ib] - W[ia];
                    { double sdn = 0; int ndn = 0, span = 0; for (int k = ia; k <= ib; k++) { span++; if (dn[k] == dn[k]) { sdn += dn[k] * st; ndn++; } } if (ndn >= 0.8 * span) v.t[1] = (float)(sdn * span / ndn); }
                    const double g2 = 0.5 * (rb - ra); const int ie = (int)ceil(rb + g2);
                    int ok = ie < n; for (int k = ib; ok && k <= ie; k++) if (!mat[k]) ok = 0;
                    if (ok && C) { int c0 = 0; for (int p = 0; p < nct; p++) c0 += ctpk[p] >= ra + g2 && ctpk[p] < rb + g2; v.t[2] = (float)c0; }
                    for (int s = 0; s < J->ns; s++) {
                        if (J->S[s].hasbox) { int fin = 1; const int k0 = (int)fmax(0, ra - g2), k1 = (int)fmin(n - 1, rb + g2); for (int k = k0; k <= k1 && fin; k++) if (!(V[s][k] == V[s][k])) fin = 0; if (!fin) continue; }
                        if (ok) v.t[3 + 3 * s] = (float)rr_count_hits(H[s], nH[s], ra + g2, rb + g2);   /* (vote: below) */
                        v.t[4 + 3 * s] = (float)rr_lin(&H[s][0].pos, 4, nH[s], ra, rb); v.t[5 + 3 * s] = (float)rr_lin(R[s], 1, nR[s], ra, rb); }
                    if (J->nvote) { float m[RR_MAXS]; int c = 0; for (int e = 0; e < J->nvote; e++) { const float t = v.t[4 + 3 * J->vote[e]]; if (t == t) m[c++] = t; }
                        if (c == J->nvote) { qsort(m, c, sizeof(float), cmp_f); v.t[5 + 3 * J->ns] = c & 1 ? m[c / 2] : 0.5f * (m[c / 2 - 1] + m[c / 2]); } }
                    v.ia = ia; v.ib = ib;
                    if (no == capo) { capo *= 2; o = realloc(o, sizeof(rr_iv) * capo); }
                    o[no++] = v;
                }
            }
            if (okp) { Q[g] = Q[g - 1] + lround(dq); sid[g] = cur; } else { cur++; Q[g] = 0; sid[g] = cur; }
        }
        /* the dump source's hit intervals: features, GT turns (Q linear between sheet centres of one stretch), classifier turns */
        if (J->dsrc >= 0) {
            const int s = J->dsrc; const rr_hit *hp = H[s]; const int nh = nH[s];
            double *cum = malloc(sizeof(double) * (nh + 1)), *cumE = malloc(sizeof(double) * (nh + 1)); cum[0] = cumE[0] = 0;
            int nd = 0, capd = 0; float *d = nullptr; int g = 0;
            for (int h = 0; h + 1 < nh; h++) {
                float f[RR_NF]; rr_feat(J, hp, nh, h, V[s], spn, nr, dn, ph, C, ctpk, nct, H, nH, 0, f);
                int allm = 1; for (int k = (int)floor(hp[h].pos); k <= (int)ceil(hp[h + 1].pos) && k < n; k++) if (!mat[k]) { allm = 0; break; }
                double tc = 1, tE = 1;
                if (J->cls && allm) { float p[RR_K]; rr_cls_predict(J->cls, f, p); int am = 0; tE = 0; for (int k = 0; k < RR_K; k++) { tE += k * p[k]; if (p[k] > p[am]) am = k; } tc = am; }
                if (!allm) { tc = tE = NAN; }
                cum[h + 1] = cum[h] + tc; cumE[h + 1] = cumE[h] + tE;
                if (J->dump && allm) {   /* GT turns: both hits inside one stretch's span */
                    const double a = hp[h].pos, b = hp[h + 1].pos;
                    while (g + 1 < ng && gr[g + 1] <= a) g++;
                    if (g + 1 < ng && gr[g] <= a) {
                        int gb = g; while (gb + 1 < ng && gr[gb + 1] < b) gb++;
                        if (gb + 1 < ng && sid[gb + 1] == sid[g] && sid[g + 1] == sid[g]) {
                            const double Qa = Q[g] + (Q[g + 1] - Q[g]) * (a - gr[g]) / (gr[g + 1] - gr[g]), Qb = Q[gb] + (Q[gb + 1] - Q[gb]) * (b - gr[gb]) / (gr[gb + 1] - gr[gb]);
                            if (nd + RR_REC > capd) { capd = capd ? 2 * capd : 64 * RR_REC; d = realloc(d, sizeof(float) * capd); }
                            memcpy(d + nd, f, sizeof f); d[nd + RR_NF] = (float)(Qb - Qa); d[nd + RR_NF + 1] = (float)J->zg; d[nd + RR_NF + 2] = (float)j; d[nd + RR_NF + 3] = (float)(a * st); nd += RR_REC;
                        }
                    }
                }
            }
            if (J->dump) { J->dump[j] = d; J->ndump[j] = nd; } else free(d);
            if (J->cls) for (int v = 0; v < no; v++) {   /* cumulative turns at the GT sheet centres: linear between hits */
                double ce[2], cc[2]; int okc = 1;
                for (int e = 0; e < 2 && okc; e++) {
                    const double r = e ? o[v].ib : o[v].ia; int hh = -1; for (int k = 0; k + 1 < nh; k++) if (hp[k].pos <= r && hp[k + 1].pos > r) { hh = k; break; }
                    if (hh < 0 || !(cum[hh + 1] == cum[hh + 1]) || !(cum[hh] == cum[hh])) { okc = 0; break; }
                    const double tt = (r - hp[hh].pos) / (hp[hh + 1].pos - hp[hh].pos);
                    cc[e] = cum[hh] + tt * (cum[hh + 1] - cum[hh]); ce[e] = cumE[hh] + tt * (cumE[hh + 1] - cumE[hh]);
                }
                /* (NaN segments: the cumulative breaks; differences across them are invalid) */
                if (okc) { int hb = 0; for (int k = 0; k + 1 < nh; k++) if (hp[k].pos >= o[v].ia && hp[k].pos <= o[v].ib && !(cum[k + 1] == cum[k + 1])) hb = 1;
                           if (!hb) { o[v].t[3 + 3 * J->ns] = (float)(cc[1] - cc[0]); o[v].t[4 + 3 * J->ns] = (float)(ce[1] - ce[0]); } }
            }
            free(cum); free(cumE);
        }
        J->out[j] = o; J->nout[j] = no;
        free(W); free(spn); free(nr); free(dn); free(ph); free(C); free(mat); free(q); free(tmp); free(gr); free(gq); free(Q); free(sid); free(ctpk);
        for (int s = 0; s < J->ns; s++) { free(V[s]); free(H[s]); free(R[s]); }
    }
}

/* --img: a crop of the z level (box-local native centre cy, cx, half size hs) at the finest source's voxel: CT grey; red = source 0,
   green = source 1 (x 0.6 + CT x 0.4); blue = GT labelled (alternating sheet shades by code parity) */
static int rr_image(const rr_job *J, const char *path, double cy, double cx, double hs, double px) {
    const int n = (int)(2 * hs / px); if (n < 8 || n > 8000) return -1;
    uint8_t *rgb = malloc((size_t)n * n * 3);
    for (int a = 0; a < n; a++) for (int b = 0; b < n; b++) {
        const double y = cy - hs + (a + 0.5) * px, x = cx - hs + (b + 0.5) * px; uint8_t *o = rgb + ((size_t)a * n + b) * 3;
        double g = 0; if (J->ct) { const double ccy = (y + J->morg[1] + 0.5) * J->F - 0.5, ccx = (x + J->morg[2] + 0.5) * J->F - 0.5; const double v = rr_plane8f(J->ct, J->co, J->cn, ccy, ccx); g = v == v ? v : 0; }
        const double s0 = J->ns > 0 ? rr_sample(J, &J->S[0], y, x) : NAN, s1 = J->ns > 1 ? rr_sample(J, &J->S[1], y, x) : NAN;
        int c = 0; if (J->cd) { const int64_t qy = (int64_t)floor((y + J->morg[1]) / J->cf) - J->qo[0], qx = (int64_t)floor((x + J->morg[2]) / J->cf) - J->qo[1];
            if (qy >= 0 && qx >= 0 && qy < J->qn[0] && qx < J->qn[1]) { const int v = J->cd[qy * J->qn[1] + qx]; c = v >= 1 && v <= 252 ? v : 0; } }
        double r = 0.55 * g, gg = 0.55 * g, bb = 0.55 * g;
        if (s0 == s0) r += 0.45 * s0;
        if (s1 == s1) gg += 0.45 * s1;
        if (c) bb += ((c + 9) / 18) % 2 ? 115 : 60;
        o[0] = (uint8_t)fmin(255, r); o[1] = (uint8_t)fmin(255, gg); o[2] = (uint8_t)fmin(255, bb);
    }
    const int rc = jpeg_write_rgb ? jpeg_write_rgb(path, n, n, rgb, 92) : -1; free(rgb); return rc;
}

/* read the source rows covering native z [zg - 0.5, zg + 0.5) (averaged), the box's y / x range + 2 */
static int rr_read_plane(rr_src *s, double zg, double um, const double morg[3], double Y, double X, int nt) {
    free(s->pl); s->pl = nullptr;
    if (s->ens) return 0;
    if (s->map) {   /* a block of the source level covering the mapped plane */
        double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30}; const double K = um / s->cum;
        for (int a = 0; a <= 16; a++) for (int b = 0; b <= 16; b++) {
            const double gy = morg[1] + Y * a / 16, gx = morg[2] + X * b / 16;
            const double c[3] = {(gx + 0.5) * K - 0.5, (gy + 0.5) * K - 0.5, (zg + 0.5) * K - 0.5}; double f[3]; rmap_c2f(s->map, c, f);
            const double L = s->lf; const double v[3] = {(f[2] + 0.5) / L - 0.5, (f[1] + 0.5) / L - 0.5, (f[0] + 0.5) / L - 0.5};
            for (int d = 0; d < 3; d++) { if (v[d] < lo[d]) lo[d] = v[d]; if (v[d] > hi[d]) hi[d] = v[d]; }
        }
        for (int d = 0; d < 3; d++) {
            int64_t a = (int64_t)floor(lo[d]) - 3, b = (int64_t)ceil(hi[d]) + 4;
            if (s->hasbox) { if (a < (int64_t)s->vb[d] - 1) a = (int64_t)s->vb[d] - 1; if (b > (int64_t)s->vb[d + 3] + 2) b = (int64_t)s->vb[d + 3] + 2; }   /* only the --src-box part */
            s->po[d] = a < 0 ? 0 : a; int64_t e = b > s->shape[d] ? s->shape[d] : b; s->pn[d] = e - s->po[d];
            if (s->pn[d] < 2) { s->pn[0] = 0; return 0; }
        }
        s->pl = malloc((size_t)s->pn[0] * s->pn[1] * s->pn[2]);
        if (!s->pl) return -1;
        return s->z ? z3_read(s->z, s->po, s->pn, s->pl, nt) : z2_read(s->z2v, s->po, s->pn, s->pl, nt);
    }
    const double RF = um / s->um;
    int64_t c0 = (int64_t)floor((zg - 0.5) * RF + 1e-9), c1 = (int64_t)ceil((zg + 0.5) * RF - 1e-9); if (c1 <= c0) c1 = c0 + 1;
    if (c0 < 0) c0 = 0;
    if (c1 > s->shape[0]) c1 = s->shape[0];
    int64_t o3[3] = {c0, (int64_t)floor(morg[1] * RF) - 2, (int64_t)floor(morg[2] * RF) - 2}, n3[3] = {c1 - c0, (int64_t)ceil(Y * RF) + 4, (int64_t)ceil(X * RF) + 4};
    for (int d = 1; d < 3; d++) { if (o3[d] < 0) { n3[d] += o3[d]; o3[d] = 0; } if (o3[d] + n3[d] > s->shape[d]) n3[d] = s->shape[d] - o3[d]; }
    s->pn[0] = 1; s->pn[1] = 0; s->pn[2] = 0;
    if (n3[0] <= 0 || n3[1] <= 1 || n3[2] <= 1) return 0;
    const size_t pl = (size_t)n3[1] * n3[2]; uint8_t *raw = malloc(pl * n3[0]); if (!raw) return -1;
    if (s->z ? z3_read(s->z, o3, n3, raw, nt) : z2_read(s->z2v, o3, n3, raw, nt)) { free(raw); return -1; }
    if (n3[0] > 1) { for (size_t i = 0; i < pl; i++) { unsigned acc = 0; for (int64_t a = 0; a < n3[0]; a++) acc += raw[a * pl + i]; raw[i] = (uint8_t)((acc + n3[0] / 2) / n3[0]); } raw = realloc(raw, pl); }
    s->pl = raw; s->po[0] = c0; s->po[1] = o3[1]; s->po[2] = o3[2]; s->pn[1] = n3[1]; s->pn[2] = n3[2];
    return 0;
}

int cmd_recto_ruler(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: ufsm recto-ruler <model-or-set> <fields-dir> --codes ROOT KEY [--src NAME ROOT KEY UM]... [--map NAME REG.json] [--ct ROOT KEY --ct-um U]\n"
                        "       [--nz 16 | --zs Z,..] [--zr Z0,Z1] [--nth 180] [--step-um U] [--sigma 0.5] [--prom 25] [--sep-um 20] [--win-um 60] [--thr 128]\n"
                        "       [--ens NAME mean|max A,B] [--dump FILE --dump-src NAME] [--cls MODEL] [--threads N] [--cache DIR]\n"
                        "  recto predictions as a winding ruler vs GT codes (peaks / runs per GT turn), interval features for ufsm recto-cls\n");
        return 2;
    }
    fit_set *fs = fit_set_load(argv[2]); if (!fs) { fprintf(stderr, "recto-ruler: cannot load %s\n", argv[2]); return 1; }
    const fit_model *m = fs->global;
    fit_fields_skip_planes(1 | 2); fit_fields_ct_planes(0);
    fit_fields_buf g; if (fit_fields_load(argv[3], 0, 0, 0, &g)) { fprintf(stderr, "recto-ruler: cannot read the fields in %s\n", argv[3]); return 1; }
    const double h = g.h, um = g.um; long long forg[3]; int fsh[3]; for (int d = 0; d < 3; d++) { forg[d] = g.origin[d]; fsh[d] = g.shape[d]; } fit_fields_free(&g);
    if (m->h != h) { fprintf(stderr, "recto-ruler: the model's cell h %g differs from the fields' %g\n", m->h, h); return 1; }
    const int nth = atoi(rr_opt(argc, argv, "--nth", "180")), nt = ufsm_threads(atoi(rr_opt(argc, argv, "--threads", "0")), 16);
    const char *cache = rr_opt(argc, argv, "--cache", nullptr);
    fit_set_threads(nt);
    rr_src S[RR_MAXS]; int ns = 0; memset(S, 0, sizeof S);
    double finest = um;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--src")) {
            if (i + 4 >= argc || ns == RR_MAXS) { fprintf(stderr, "recto-ruler: --src NAME ROOT KEY UM (at most %d)\n", RR_MAXS); return 2; }
            rr_src *s = &S[ns]; snprintf(s->name, sizeof s->name, "%s", argv[i + 1]); s->um = atof(argv[i + 4]);
            store *st = store_open(argv[i + 2]);
            if (!(s->z = z3_open(st, argv[i + 3], cache)) && !(s->z2v = z2_open(store_open(argv[i + 2]), argv[i + 3], cache))) { fprintf(stderr, "recto-ruler: --src %s: %s / %s\n", s->name, z3_error(), z2_error()); return 1; }
            if (s->z2v) z2_assume_present(s->z2v, 1);
            const int64_t *sh = s->z ? z3_meta_of(s->z)->shape : z2_meta_of(s->z2v)->shape; for (int d = 0; d < 3; d++) s->shape[d] = sh[d];
            if (!(s->um > 0)) { fprintf(stderr, "recto-ruler: --src %s: UM > 0\n", s->name); return 2; }
            ns++; i += 4;
        } else if (!strcmp(argv[i], "--ens")) {
            if (i + 3 >= argc || ns == RR_MAXS) { fprintf(stderr, "recto-ruler: --ens NAME mean|max A,B\n"); return 2; }
            rr_src *s = &S[ns]; snprintf(s->name, sizeof s->name, "%s", argv[i + 1]); s->ens = !strcmp(argv[i + 2], "max") ? 2 : !strcmp(argv[i + 2], "gmean") ? 3 : !strcmp(argv[i + 2], "norm") ? 4 : 1; s->um = um;
            char buf[512]; snprintf(buf, sizeof buf, "%s", argv[i + 3]);
            for (char *tok = strtok(buf, ","); tok; tok = strtok(nullptr, ",")) { int f = -1; for (int k = 0; k < ns; k++) if (!strcmp(S[k].name, tok)) f = k;
                if (f < 0) { fprintf(stderr, "recto-ruler: --ens: no source %s (sources first)\n", tok); return 2; } s->ea[s->ne++] = f; if (S[f].um < s->um) s->um = S[f].um; }
            ns++; i += 3;
        }
    }
    for (int i = 3; i + 2 < argc; i++) if (!strcmp(argv[i], "--src-box")) {
        int f = -1; for (int k = 0; k < ns; k++) if (!strcmp(S[k].name, argv[i + 1])) f = k;
        if (f < 0 || sscanf(argv[i + 2], "%lf,%lf,%lf,%lf,%lf,%lf", &S[f].vb[0], &S[f].vb[1], &S[f].vb[2], &S[f].vb[3], &S[f].vb[4], &S[f].vb[5]) != 6) { fprintf(stderr, "recto-ruler: --src-box NAME Z0,Y0,X0,Z1,Y1,X1\n"); return 2; }
        S[f].hasbox = 1; i += 2;
    }
    for (int i = 3; i + 2 < argc; i++) if (!strcmp(argv[i], "--map")) {   /* --map NAME REG.json: NAME's UM is its own voxel um (fine scan level) */
        int f = -1; for (int k = 0; k < ns; k++) if (!strcmp(S[k].name, argv[i + 1])) f = k;
        if (f < 0) { fprintf(stderr, "recto-ruler: --map: no source %s\n", argv[i + 1]); return 2; }
        rvol *fv = nullptr; double fum = 0; if (!reg_map_open || !(S[f].map = reg_map_open(argv[i + 2], cache, &fv, &S[f].cum, &fum)) || !(fum > 0)) { fprintf(stderr, "recto-ruler: cannot open the map %s\n", argv[i + 2]); return 1; }
        S[f].lf = S[f].um / fum;
        i += 2;
    }
    for (int s = 0; s < ns; s++) if (!S[s].ens && S[s].um < finest) finest = S[s].um;
    const double step_um = atof(rr_opt(argc, argv, "--step-um", "0")) > 0 ? atof(rr_opt(argc, argv, "--step-um", "0")) : finest / 2, step = step_um / um;
    const double sigma = atof(rr_opt(argc, argv, "--sigma", "0.5")), prom = atof(rr_opt(argc, argv, "--prom", "25")), sep = atof(rr_opt(argc, argv, "--sep-um", "20")), win = atof(rr_opt(argc, argv, "--win-um", "60")), thr = atof(rr_opt(argc, argv, "--thr", "128"));
    z3 *cz = nullptr, *qz = nullptr; double ctum = 0, cf = 1;
    for (int i = 3; i + 2 < argc; i++) {
        if (!strcmp(argv[i], "--ct")) { ctum = atof(rr_opt(argc, argv, "--ct-um", "0")); if (!(ctum > 0)) { fprintf(stderr, "recto-ruler: --ct needs --ct-um\n"); return 2; }
                                       if (!(cz = z3_open(store_open(argv[i + 1]), argv[i + 2], cache))) { fprintf(stderr, "recto-ruler: %s\n", z3_error()); return 1; } }
        if (!strcmp(argv[i], "--codes")) { if (!(qz = z3_open(store_open(argv[i + 1]), argv[i + 2], nullptr))) { fprintf(stderr, "recto-ruler: %s\n", z3_error()); return 1; }
                                          cf = atof(argv[i + 2]) > 0 ? atof(argv[i + 2]) / um : 1; }
    }
    if (!qz) { fprintf(stderr, "recto-ruler: --codes ROOT KEY needed\n"); return 2; }
    int vote[RR_MAXS], nvote = 0;
    { const char *vl = rr_opt(argc, argv, "--vote", nullptr);
      if (vl) { char buf[512]; snprintf(buf, sizeof buf, "%s", vl); for (char *t = strtok(buf, ","); t; t = strtok(nullptr, ",")) { int f = -1; for (int k = 0; k < ns; k++) if (!strcmp(S[k].name, t)) f = k;
                if (f < 0 || nvote == RR_MAXS) { fprintf(stderr, "recto-ruler: --vote: no source %s\n", t); return 2; } vote[nvote++] = f; } } }
    int dsrc = -1; const char *dn = rr_opt(argc, argv, "--dump-src", nullptr);
    if (dn) { for (int k = 0; k < ns; k++) if (!strcmp(S[k].name, dn)) dsrc = k; if (dsrc < 0) { fprintf(stderr, "recto-ruler: --dump-src %s: no such source\n", dn); return 2; } }
    const char *dumpf = rr_opt(argc, argv, "--dump", nullptr), *clsf = rr_opt(argc, argv, "--cls", nullptr);
    if ((dumpf || clsf) && dsrc < 0) { fprintf(stderr, "recto-ruler: --dump / --cls need --dump-src\n"); return 2; }
    rr_cls *cls = nullptr; if (clsf && !(cls = rr_cls_load(clsf))) { fprintf(stderr, "recto-ruler: cannot load --cls %s\n", clsf); return 1; }
    FILE *df = nullptr; if (dumpf) { if (!(df = fopen(dumpf, "wb"))) { fprintf(stderr, "recto-ruler: cannot write %s\n", dumpf); return 1; } const int hd[4] = {0x31445252, RR_NF, RR_REC, 0}; fwrite(hd, 4, 4, df); }
    const double Z = m->D * m->h; double zr0 = 0.05 * Z, zr1 = 0.95 * Z;
    const char *zr = rr_opt(argc, argv, "--zr", nullptr); if (zr && sscanf(zr, "%lf,%lf", &zr0, &zr1) != 2) { fprintf(stderr, "recto-ruler: --zr Z0,Z1\n"); return 2; }
    double zs[4096]; int nzs = 0;
    { const char *zl = rr_opt(argc, argv, "--zs", nullptr);
      if (zl) { char buf[65536]; snprintf(buf, sizeof buf, "%s", zl); for (char *t = strtok(buf, ","); t && nzs < 4096; t = strtok(nullptr, ",")) zs[nzs++] = floor(atof(t)) + 0.5; }
      else { const int nz = atoi(rr_opt(argc, argv, "--nz", "16")); for (int i = 0; i < nz && i < 4096; i++) zs[nzs++] = floor(zr0 + (i + 0.5) * (zr1 - zr0) / nz) + 0.5; } }
    const double morg[3] = {(double)m->origin[0], (double)m->origin[1], (double)m->origin[2]};
    const double Y = m->H * h, X = m->W * h, F = cz ? um / ctum : 1;
    printf("recto-ruler: %s: box %dx%dx%d cells h %g (%.3f um), %d z levels x %d rays, step %.2f um, sources", argv[2], m->D, m->H, m->W, h, um, nzs, nth, step_um);
    for (int s = 0; s < ns; s++) printf(" %s(%s%.3g)", S[s].name, S[s].ens ? "ens " : S[s].map ? "reg " : "", S[s].um);
    printf("; peaks sigma %.2f vox prom %.0f sep %.0f um win %.0f um; runs >= %.0f\n", sigma, prom, sep, win, thr);
    const double t0 = rr_now(); size_t ni = 0, capi = 1 << 16; rr_iv *all = malloc(sizeof(rr_iv) * capi); size_t ndrec = 0;
    for (int zi = 0; zi < nzs; zi++) {
        const double zl = zs[zi], zg = zl + morg[0]; const int fz = (int)floor((zg - forg[0]) / h);
        if (fz < 0 || fz >= fsh[0]) continue;
        { double zb = 0; int par = 1; if (sscanf(rr_opt(argc, argv, "--zblock", ""), "%lf,%d", &zb, &par) == 2 && zb > 0 && (((int)floor(zg / zb)) & 1) != par) continue; }   /* --zblock ZB,P: only z levels in blocks of parity P (recto-cls --holdout-z ZB: 1 = the holdout) */
        fit_fields_buf b; if (fit_fields_load(argv[3], fz, fz + 1, 1, &b)) { fprintf(stderr, "recto-ruler: cannot load field row %d\n", fz); return 1; }
        rr_job J = {.fs = fs, .m = m, .zl = zl, .step = step, .um = um, .h = h, .cf = cf, .morg = {morg[0], morg[1], morg[2]}, .w = b.f.w, .sp = b.f.sp, .pc = b.f.pc, .ps = b.f.ps,
                    .tyy = b.f.t[1], .txx = b.f.t[2], .tyx = b.f.t[5], .FH = fsh[1], .FW = fsh[2], .fo = {morg[1] - forg[1], morg[2] - forg[2]}, .F = F, .S = S, .ns = ns, .zg = zg,
                    .nth = nth, .sigma = sigma, .prom = prom, .sep = sep, .win = win, .thr = thr, .ctsig = 1.2, .ctprom = 1, .dsrc = dsrc, .cls = cls, .nvote = nvote};
        memcpy(J.vote, vote, sizeof vote);
        float *ctp = nullptr; uint8_t *cdp = nullptr;
        for (int s = 0; s < ns; s++) if (rr_read_plane(&S[s], zg, um, morg, Y, X, nt)) { fprintf(stderr, "recto-ruler: %s read failed: %s %s\n", S[s].name, z3_error(), z2_error()); return 1; }
        if (cz) {
            const z3_meta *zm = z3_meta_of(cz);
            int64_t c0 = (int64_t)floor((zg - 0.5) * F + 1e-9), c1 = (int64_t)ceil((zg + 0.5) * F - 1e-9); if (c1 <= c0) c1 = c0 + 1;
            if (c0 < 0) c0 = 0;
            if (c1 > zm->shape[0]) c1 = zm->shape[0];
            int64_t o3[3] = {c0, (int64_t)floor(morg[1] * F) - 2, (int64_t)floor(morg[2] * F) - 2}, n3[3] = {c1 - c0, (int64_t)ceil(Y * F) + 4, (int64_t)ceil(X * F) + 4};
            for (int d = 1; d < 3; d++) { if (o3[d] < 0) { n3[d] += o3[d]; o3[d] = 0; } if (o3[d] + n3[d] > zm->shape[d]) n3[d] = zm->shape[d] - o3[d]; }
            if (n3[0] > 0 && n3[1] > 1 && n3[2] > 1) {
                uint8_t *raw = malloc((size_t)n3[0] * n3[1] * n3[2]);
                if (!raw || z3_read(cz, o3, n3, raw, nt)) { fprintf(stderr, "recto-ruler: CT read failed: %s\n", z3_error()); return 1; }
                const size_t pl = (size_t)n3[1] * n3[2]; ctp = calloc(pl, sizeof(float));
                for (int64_t a = 0; a < n3[0]; a++) for (size_t i = 0; i < pl; i++) ctp[i] += raw[a * pl + i];
                for (size_t i = 0; i < pl; i++) ctp[i] /= (float)n3[0];
                free(raw); J.ct = ctp; J.co[0] = o3[1]; J.co[1] = o3[2]; J.cn[0] = n3[1]; J.cn[1] = n3[2];
            }
        }
        {
            const z3_meta *zm = z3_meta_of(qz);
            int64_t o3[3] = {(int64_t)floor(zg / cf), (int64_t)floor(morg[1] / cf), (int64_t)floor(morg[2] / cf)}, n3[3] = {1, (int64_t)ceil(Y / cf) + 1, (int64_t)ceil(X / cf) + 1};
            for (int d = 1; d < 3; d++) if (o3[d] + n3[d] > zm->shape[d]) n3[d] = zm->shape[d] - o3[d];
            if (o3[0] < zm->shape[0] && n3[1] > 0 && n3[2] > 0) {
                cdp = malloc((size_t)n3[1] * n3[2]);
                if (z3_read(qz, o3, n3, cdp, nt)) { fprintf(stderr, "recto-ruler: code read failed: %s\n", z3_error()); return 1; }
                J.cd = cdp; J.qo[0] = o3[1]; J.qo[1] = o3[2]; J.qn[0] = n3[1]; J.qn[1] = n3[2];
            }
        }
        if (zi == 0 && rr_opt(argc, argv, "--img", nullptr)) {   /* --img FILE --img-box DY,DX,HALF (native, centre offset from the umbilicus) */
            double ib[3] = {0, 0, 0}; if (sscanf(rr_opt(argc, argv, "--img-box", ""), "%lf,%lf,%lf", &ib[0], &ib[1], &ib[2]) != 3) { fprintf(stderr, "recto-ruler: --img needs --img-box CY,CX,HALF\n"); return 2; }
            { const int iz = (int)fmin(zl, m->D * m->h); ib[0] += m->cy[iz]; ib[1] += m->cx[iz]; }   /* (CY, CX: offsets from the umbilicus) */
            if (rr_image(&J, rr_opt(argc, argv, "--img", nullptr), ib[0], ib[1], ib[2], finest / um)) fprintf(stderr, "recto-ruler: --img failed\n");
            if (atoi(rr_opt(argc, argv, "--img-only", "0"))) return 0;
        }
        J.out = calloc(nth, sizeof(rr_iv *)); J.nout = calloc(nth, sizeof(int));
        if (df) { J.dump = calloc(nth, sizeof(float *)); J.ndump = calloc(nth, sizeof(int)); }
        if (J.cd) fit_par(nth, rr_rays, &J);
        for (int j = 0; j < nth; j++) {
            if (ni + J.nout[j] > capi) { while (ni + J.nout[j] > capi) capi *= 2; all = realloc(all, sizeof(rr_iv) * capi); }
            if (J.nout[j]) memcpy(all + ni, J.out[j], sizeof(rr_iv) * J.nout[j]); ni += J.nout[j]; free(J.out[j]);
            if (df && J.ndump[j]) { fwrite(J.dump[j], sizeof(float), J.ndump[j], df); ndrec += J.ndump[j] / RR_REC; }
            if (df) free(J.dump[j]);
        }
        free(J.out); free(J.nout); free(J.dump); free(J.ndump); free(ctp); free(cdp); fit_fields_free(&b);
        fprintf(stderr, "\rrecto-ruler: z %d / %d (%zu GT intervals, %zu dump records, %.0f s)", zi + 1, nzs, ni, ndrec, rr_now() - t0);
    }
    fprintf(stderr, "\n");
    if (df) fclose(df);
    if (!ni) { printf("recto-ruler: no GT intervals\n"); return 1; }
    /* tables */
    const int NS = 3 + 3 * ns + 3; char nm[3 + 3 * RR_MAXS + 3][80];
    snprintf(nm[0], 80, "fit"); snprintf(nm[1], 80, "density"); snprintf(nm[2], 80, "CT");
    for (int s = 0; s < ns; s++) { snprintf(nm[3 + 3 * s], 80, "%s.pkw", S[s].name); snprintf(nm[4 + 3 * s], 80, "%s.pk", S[s].name); snprintf(nm[5 + 3 * s], 80, "%s.run", S[s].name); }
    snprintf(nm[3 + 3 * ns], 80, "cls"); snprintf(nm[4 + 3 * ns], 80, "clsE"); snprintf(nm[5 + 3 * ns], 80, "vote");
    const double gb[] = {0, 70, 85, 100, 115, 130, 150, 180, 230, 1e9}; const int ngb = 9;
    FILE *tf = nullptr; const char *tsv = rr_opt(argc, argv, "--tsv", nullptr); if (tsv && !(tf = fopen(tsv, "w"))) { fprintf(stderr, "recto-ruler: cannot write %s\n", tsv); return 1; }
    if (tf) fprintf(tf, "table\tbin\tsource\tn\tper_gt_turn\toff\n");
    FILE *tf2 = tf;
    for (int tab = 0; tab < 3; tab++) {
        const int nbin = tab ? ngb : 20;
        double (*s)[3 + 3 * RR_MAXS + 3][4] = calloc(nbin + 1, sizeof *s); long long *cnt = calloc(nbin + 1, sizeof(long long));
        for (size_t i = 0; i < ni; i++) {
            const rr_iv *v = &all[i]; int b;
            if (tab) { const double gv = tab == 1 ? v->gapn : v->sp; if (!(gv == gv)) continue; b = 0; while (b + 1 < ngb && gv >= gb[b + 1]) b++; } else { b = (int)(v->r * um / 2000); if (b >= nbin) b = nbin - 1; }
            cnt[b]++; cnt[nbin]++;
            for (int k = 0; k < NS; k++) { if (!(v->t[k] == v->t[k])) continue;
                for (int bb = 0; bb < 2; bb++) { double *a = s[bb ? nbin : b][k]; a[0] += v->t[k]; a[1] += v->gtq; a[2]++; a[3] += fabs(v->t[k] - v->gtq) >= 0.5; } }
        }
        printf("  GT intervals by %s: turns per GT turn / share off >= 0.5 turn\n  %-11s %8s", tab == 1 ? "normal gap per GT turn (um)" : tab == 2 ? "network spacing sp (um)" : "radius (mm)", tab ? "gap(um)" : "radius", "n");
        for (int k = 0; k < NS; k++) { if ((k == 3 + 3 * ns || k == 4 + 3 * ns) && !cls) continue; if (k == 5 + 3 * ns && !nvote) continue; printf("  %13s", nm[k]); }
        printf("\n");
        for (int b = 0; b <= nbin; b++) { if (cnt[b] < 50) continue;
            char bl[32]; if (b == nbin) snprintf(bl, sizeof bl, "all"); else if (tab) snprintf(bl, sizeof bl, "%.0f-%.0f", gb[b], b + 1 < ngb ? gb[b + 1] : 9999.0); else snprintf(bl, sizeof bl, "%d-%d", 2 * b, 2 * b + 2);
            printf("  %-11s %8lld", bl, cnt[b]);
            for (int k = 0; k < NS; k++) { if ((k == 3 + 3 * ns || k == 4 + 3 * ns) && !cls) continue; if (k == 5 + 3 * ns && !nvote) continue; const double *a = s[b][k]; if (a[2] < 20) printf("  %13s", "-"); else printf("  %6.3f/%.3f", a[0] / a[1], a[3] / a[2]);
                if (tf && a[2] >= 20) fprintf(tf, "%s\t%s\t%s\t%.0f\t%.4f\t%.4f\n", tab == 0 ? "radius" : tab == 1 ? "gap" : "sp", bl, nm[k], a[2], a[0] / a[1], a[3] / a[2]); }
            printf("\n"); }
        free(s); free(cnt);
    }
    {   /* long spans: L consecutive GT intervals on one ray (contiguous: same z / ray, sheet shared), each source's turns summed vs the GT turns */
        const int Ls[3] = {5, 10, 20};
        printf("  spans of L consecutive GT intervals: turns per GT turn / share |error| >= 1 turn / mean |error| (turns)\n  %-11s %8s", "L", "n");
        for (int k = 0; k < NS; k++) { if ((k == 3 + 3 * ns || k == 4 + 3 * ns) && !cls) continue; if (k == 5 + 3 * ns && !nvote) continue; if (k >= 3 && k < 3 + 3 * ns && (k - 3) % 3 != 1) continue; printf("  %19s", nm[k]); }
        printf("\n");
        for (int li = 0; li < 3; li++) {
            const int L = Ls[li]; double a[3 + 3 * RR_MAXS + 3][4]; memset(a, 0, sizeof a); long long nsp = 0;
            for (size_t i = 0; i + L <= ni; i += 1) {
                int ok = 1; for (int q = 1; q < L && ok; q++) { const rr_iv *u = &all[i + q - 1], *v = &all[i + q]; if (v->z != u->z || v->ray != u->ray || v->ia != u->ib) ok = 0; }
                if (!ok) continue;
                nsp++;
                for (int k = 0; k < NS; k++) { double st = 0, sg = 0; int okk = 1; for (int q = 0; q < L; q++) { const float t = all[i + q].t[k]; if (!(t == t)) { okk = 0; break; } st += t; sg += all[i + q].gtq; }
                    if (!okk) continue; a[k][0] += st; a[k][1] += sg; a[k][2]++; a[k][3] += fabs(st - sg) >= 1; a[k][3] += 0; }
            }
            double ae[3 + 3 * RR_MAXS + 3] = {0};
            for (size_t i = 0; i + L <= ni; i++) {
                int ok = 1; for (int q = 1; q < L && ok; q++) { const rr_iv *u = &all[i + q - 1], *v = &all[i + q]; if (v->z != u->z || v->ray != u->ray || v->ia != u->ib) ok = 0; }
                if (!ok) continue;
                for (int k = 0; k < NS; k++) { double st = 0, sg = 0; int okk = 1; for (int q = 0; q < L; q++) { const float t = all[i + q].t[k]; if (!(t == t)) { okk = 0; break; } st += t; sg += all[i + q].gtq; } if (okk) ae[k] += fabs(st - sg); }
            }
            printf("  %-11d %8lld", L, nsp);
            for (int k = 0; k < NS; k++) { if ((k == 3 + 3 * ns || k == 4 + 3 * ns) && !cls) continue; if (k == 5 + 3 * ns && !nvote) continue; if (k >= 3 && k < 3 + 3 * ns && (k - 3) % 3 != 1) continue;
                if (a[k][2] < 20) printf("  %19s", "-"); else printf("  %5.3f/%.3f/%5.2f", a[k][0] / a[k][1], a[k][3] / a[k][2], ae[k] / a[k][2]);
                if (tf2 && a[k][2] >= 20) fprintf(tf2, "span\t%d\t%s\t%.0f\t%.4f\t%.4f\t%.4f\n", L, nm[k], a[k][2], a[k][0] / a[k][1], a[k][3] / a[k][2], ae[k] / a[k][2]); }
            printf("\n");
        }
    }
    if (tf2) fclose(tf2);
    printf("recto-ruler: %zu GT intervals%s in %.0f s\n", ni, df ? ", dump written" : "", rr_now() - t0);
    if (df) printf("recto-ruler: %zu dump records (%d features) -> %s\n", ndrec, RR_NF, dumpf);
    free(all); for (int s = 0; s < ns; s++) { free(S[s].pl); if (S[s].z) z3_close(S[s].z); if (S[s].z2v) z2_close(S[s].z2v); }
    if (cz) z3_close(cz); z3_close(qz); fit_set_free(fs); rr_cls_free(cls);
    return 0;
}
