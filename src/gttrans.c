/* ufsm gt-trans ROOT CTKEY CTUM CODEROOT CODEKEY CODEUM CENTREKEY CENTREUM BOXES_TSV SCROLL BOXUM OUT.tsv [--nz 8] [--nth 360]
                  [--rv ROOT KEY] [--fh ROOT KEY --fv ROOT KEY] [--rv-thr 8] [--fib-thr 8] [--fib-margin 4] [--threads N]
   Sheet-boundary transitions per GT turn by normal gap (rvsep 2026-11-18): can recto / verso or fibre direction count TOUCHING sheets
   where CT peaks cannot? A sheet is a recto layer (horizontal fibres) on a verso layer (vertical fibres), so a ray through a stack
   crosses R V | R V | R V: every sheet boundary is a V -> R (V-fibre -> H-fibre) step even at zero gap.
   The rays, GT intervals and CT detector are pseudolab's / fix0332's ctgt.py (DESIGN "Packed sheets are not resolved by finer CT"):
   per box of BOXES_TSV (scripts/wbench/boxes.tsv; rows of SCROLL, box in BOXUM voxels) --nz native z planes; rays every 360 / --nth
   degrees from the scroll's CT centroid at that z (plane CENTREKEY, CENTREUM um, CT > 0), unit steps in CTKEY's voxels (CTUM um);
   GT codes CODEROOT / CODEKEY (CODEUM um; 1..252 = 18 codes a turn) sampled nearest; a GT interval = consecutive GT sheet crossings
   (runs of codes within 9 codes, gaps <= 2 samples) 0.5 .. 2.5 turns apart with material (CT > 0) all along; its normal gap = its
   length x |cos(ray, sheet normal)| / turns, the normal from the structure tensor of the GT label mask (gaussian 1, sobel, gaussian 3).
   Per interval [ia, ib): (c) CT papyrus peaks (wrap-check's detector: gaussian --ct-sig 1.2, local sd over +-12, prominence 1 sd within
   +-24 samples); (a) with --rv (predict --heads rv: 1 + 254 (1 + p_recto - p_verso) / 2, 0 = no CT) the side state along the ray with
   hysteresis (R above 128 + --rv-thr, V below 128 - --rv-thr, else the last state; try9's rv bytes sit mostly in 96..160) and its V -> R / R -> V flips going outward;
   (b) with --fh / --fv (predict --heads fib: fib/H, fib/V probabilities 0..255) the fibre state (H where H >= --fib-thr and
   H >= V + --fib-margin, V likewise) and its V -> H / H -> V flips. Prediction stores on CTKEY's grid (the same level), global
   coordinates; a ROOT with {box} names one store per box (predict --box: box-local coordinates). A sheet's GT centre lies at its R | V interface, so one interval holds exactly one sheet boundary (V -> R / V -> H,
   outward) and the two intra-sheet flips at its ends: the boundary direction should count 1 per GT turn.
   OUT.tsv: one row per interval; stdout: per gap bucket (0-70 / 70-85 / 85-100 / 100-115 / ... um) turns, CT peaks, flips per GT
   turn and the share off by >= 0.5 turn, as the ctgt.py table (and the CT peaks again on the intervals with predictions: the same rays for both); also the peaks of the rv signal and of H - V (the CT detector on those
   signals: recto-ruler's "peaks of the fused signal", one per sheet where the side runs alternate).
   --ct-l / --ct-p: the detector's local-sd and prominence windows (samples; 12 / 24 at 9.6 um, scale with the voxel for equal physics).
   (d) --orient SIG,RHO: the layers' fibre orientation straight from the CT (no net): the 3-D structure tensor (gaussian derivative SIG,
   integration RHO, CT voxels) of the CT (--orient-ct: another store on its grid) projected on the plane spanned by z (the scroll axis,
   taken as z) and the sheet's in-slice tangent (from the GT label's normal); a sample is H (fibres perpendicular to the axis: the
   gradient along z) where its coherence >= --orient-coh and cos 2a > --orient-margin, V where < -margin, else it keeps the last state
   (also where CT < --orient-iso); --orient-persist UM: a new state must hold over UM of classified samples first; the V -> H / H -> V flips going outward per GT turn as (b). */
#include "fit.h"
#include "latfit.h"
#include "lcodec.h"
#include "volcomp.h"
#include "optcheck.h"
#include "sysinfo.h"
#include "store.h"
#include "zarr3.h"
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static const char *gt_opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 13; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(name))) return argv[i + 1];
    return dflt;
}
static const char *gt_opt2(int argc, char **argv, const char *name, int k) {   /* the k-th value after --name (two-value options) */
    for (int i = 13; i + 2 < argc; i++) if (!strcmp(argv[i], opt_n(name))) return argv[i + 1 + k];
    return nullptr;
}

typedef struct { char name[32]; int64_t b[6]; } gt_box;
typedef struct { int box; int z; float rmm, dq, gap; int npk, rvf, rvb, fbf, fbb, rvok, fbok, rvpk, fhpk, orf, orb, orok; } gt_row;
typedef struct {
    z3 *ct, *cd, *cl, **rvz, **fhz, **fvz; int plocal; double ctum, cum, clum, bum, rvthr, fthr, fmar, sig;
    const gt_box *bx; int nbox, nz, nth, pkL, pkP;
    z3 *oct; int orient, opers; double osig, orho, ocoh, omar, oiso;   /* --orient: the in-sheet fibre orientation from the CT structure tensor */
    const lat_basis *lb; int lc; double lstep, lcen, lsc;   /* --lat-rt: the CT through the latent cell codec first */
    const lcodec *lcd; double vcq;   /* --lat-codec: the learned codec instead (on --lat-rt's basis); --vc-rt Q: plain volcomp at quality Q instead */
    gt_row **rows; int *nrow, *cap;   /* per job */
    pthread_mutex_t mu; int done; char err[256];
} gt_job;

static int rd(z3 *z, int64_t zz, int64_t y, int64_t x, int64_t ny, int64_t nx, uint8_t *out) {   /* one plane, zeros outside the array */
    const z3_meta *m = z3_meta_of(z); memset(out, 0, (size_t)ny * nx);
    if (zz < 0 || zz >= m->shape[0]) return 0;
    int64_t y0 = y < 0 ? 0 : y, x0 = x < 0 ? 0 : x, y1 = y + ny > m->shape[1] ? m->shape[1] : y + ny, x1 = x + nx > m->shape[2] ? m->shape[2] : x + nx;
    if (y1 <= y0 || x1 <= x0) return 0;
    uint8_t *t = malloc((size_t)(y1 - y0) * (x1 - x0)); if (!t) return 1;
    const int64_t o[3] = {zz, y0, x0}, n[3] = {1, y1 - y0, x1 - x0};
    if (z3_read(z, o, n, t, 1)) { free(t); return 1; }
    for (int64_t r = y0; r < y1; r++) memcpy(out + (size_t)(r - y) * nx + (x0 - x), t + (size_t)(r - y0) * (x1 - x0), (size_t)(x1 - x0));
    free(t); return 0;
}
/* scipy.ndimage-style separable gaussian (truncate 4, mode reflect) and sobel, in place / out of place on H x W floats */
static int refl(int i, int n) { while (i < 0 || i >= n) i = i < 0 ? -i - 1 : 2 * n - i - 1; return i; }
static void gauss2(float *a, int H, int W, double s) {
    const int R = (int)(4 * s + 0.5); double *k = malloc(sizeof(double) * (2 * R + 1)), sum = 0;
    for (int i = -R; i <= R; i++) sum += k[i + R] = exp(-0.5 * i * i / (s * s));
    for (int i = 0; i <= 2 * R; i++) k[i] /= sum;
    float *t = malloc(sizeof(float) * (size_t)(H > W ? H : W));
    for (int y = 0; y < H; y++) { for (int x = 0; x < W; x++) { double v = 0; for (int i = -R; i <= R; i++) v += k[i + R] * a[(size_t)y * W + refl(x + i, W)]; t[x] = (float)v; }
                                  memcpy(a + (size_t)y * W, t, sizeof(float) * W); }
    for (int x = 0; x < W; x++) { for (int y = 0; y < H; y++) { double v = 0; for (int i = -R; i <= R; i++) v += k[i + R] * a[(size_t)refl(y + i, H) * W + x]; t[y] = (float)v; }
                                  for (int y = 0; y < H; y++) a[(size_t)y * W + x] = t[y]; }
    free(t); free(k);
}
static void sobel(const float *a, int H, int W, int ax, float *o) {   /* ax 0: d/dy ([-1 0 1] along y, [1 2 1] along x); 1: d/dx */
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        double v = 0;
        for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
            const int dd = ax ? i : j, ss = ax ? j : i; const double w = dd * (ss ? 1 : 2);
            if (w != 0) v += w * a[(size_t)refl(y + j, H) * W + refl(x + i, W)];
        }
        o[(size_t)y * W + x] = (float)v;
    }
}
static double bil(const float *img, int H, int W, double y, double x) {
    const int y0 = (int)floor(y), x0 = (int)floor(x); if (y0 < 0 || x0 < 0 || y0 + 1 >= H || x0 + 1 >= W) return NAN;
    const double ty = y - y0, tx = x - x0; const float *p = img + (size_t)y0 * W + x0;
    return (1 - ty) * ((1 - tx) * p[0] + tx * p[1]) + ty * ((1 - tx) * p[W] + tx * p[W + 1]);
}
/* ctgt.py's peaks(): wrap-check's detector, step 1 native; pk[i] = 1 at a peak */
static void peaks(const double *C, const uint8_t *mat, int n, double sig, int L, int P, uint8_t *pk) {
    const int R = (int)ceil(3 * sig); double *gk = malloc(sizeof(double) * (2 * R + 1)); for (int d = -R; d <= R; d++) gk[d + R] = exp(-0.5 * (d / sig) * (d / sig));
    double *Cs = malloc(sizeof(double) * n), *ps = malloc(sizeof(double) * (n + 1)), *ps2 = malloc(sizeof(double) * (n + 1)); int *pc = malloc(sizeof(int) * (n + 1));
    for (int i = 0; i < n; i++) { double num = 0, den = 0; for (int d = -R; d <= R; d++) { const int j = i + d; if (j < 0 || j >= n) continue; if (!isnan(C[j])) { num += gk[d + R] * C[j]; den += gk[d + R]; } }
                                  Cs[i] = den > 0 ? num / fmax(den, 1e-12) : NAN; }
    ps[0] = ps2[0] = 0; pc[0] = 0;
    for (int i = 0; i < n; i++) { const int ok = !isnan(Cs[i]); const double v = ok ? Cs[i] : 0; ps[i + 1] = ps[i] + v; ps2[i + 1] = ps2[i] + v * v; pc[i + 1] = pc[i] + ok; }
    memset(pk, 0, n);
    for (int i = 1; i < n - 1; i++) {
        if (!mat[i] || isnan(Cs[i]) || !(Cs[i] > Cs[i - 1]) || Cs[i] < Cs[i + 1]) continue;
        const int lo = i - L > 0 ? i - L : 0, hi = i + L + 1 < n ? i + L + 1 : n, c = pc[hi] - pc[lo]; if (c < 4) continue;
        const double mu = (ps[hi] - ps[lo]) / c, sd = sqrt(fmax(0, (ps2[hi] - ps2[lo]) / c - mu * mu)); if (!(sd > 0)) continue;
        double ml = Cs[i], mr = Cs[i];
        for (int k = i - 1; k > (i - P - 1 > -1 ? i - P - 1 : -1); k--) { if (isnan(Cs[k]) || Cs[k] > Cs[i]) break; ml = fmin(ml, Cs[k]); }
        for (int k = i + 1; k < (i + P + 1 < n ? i + P + 1 : n); k++) { if (isnan(Cs[k]) || Cs[k] >= Cs[i]) break; mr = fmin(mr, Cs[k]); }
        if (Cs[i] - fmax(ml, mr) >= sd) pk[i] = 1;
    }
    free(gk); free(Cs); free(ps); free(ps2); free(pc);
}
/* the 2-state sequence with hysteresis: st[i] in {0 unknown, 1, 2}; flips: f12[i] = 1 where 1 -> 2 at i, f21 where 2 -> 1 */
static void flips(const int8_t *raw, int n, uint8_t *f12, uint8_t *f21) {
    int s = 0; memset(f12, 0, n); memset(f21, 0, n);
    for (int i = 0; i < n; i++) { const int r = raw[i]; if (!r || r == s) continue; if (s == 1 && r == 2) f12[i] = 1; else if (s == 2 && r == 1) f21[i] = 1; s = r; }
}

/* flips() with persistence: a new state counts only once it holds for P classified samples (unknown samples neither count nor
   break the run); the flip sits where that run started */
static void flips_p(const int8_t *raw, int n, int P, uint8_t *f12, uint8_t *f21) {
    int s = 0, c = 0, c0 = 0, cl = 0; memset(f12, 0, n); memset(f21, 0, n);
    for (int i = 0; i < n; i++) {
        const int r = raw[i]; if (!r) continue;
        if (r != c) { c = r; c0 = i; cl = 0; }
        if (++cl >= P && c != s) { if (s == 1 && c == 2) f12[c0] = 1; else if (s == 2 && c == 1) f21[c0] = 1; s = c; }
    }
}

/* --lat-rt BASIS: replace plane zn of the CT by what a latent-grid net reads of it: each 4^3 cell (global 4-aligned grid) of the
   normalised CT (v - centre) / scale on the first lc basis rows, quantised to lstep (0: not; |q| <= 127, the mean 32767) as
   input_latc_k (lp_mx_cvt.cu) does, then back to voxels (orthonormal basis: the sum of q_c b_c) and to uint8 CT */
static int lat_rt(const gt_job *J, int64_t zn, int64_t Y0, int64_t X0, int H, int W, float *ct) {
    const int64_t zc = zn & ~(int64_t)3, yc = Y0 & ~(int64_t)3, xc = X0 & ~(int64_t)3; const int dz = (int)(zn - zc);
    const int HH = (int)(((Y0 + H + 3) & ~(int64_t)3) - yc), WW = (int)(((X0 + W + 3) & ~(int64_t)3) - xc);
    uint8_t *b = malloc((size_t)4 * HH * WW); if (!b) return 1;
    for (int k = 0; k < 4; k++) if (rd(J->ct, zc + k, yc, xc, HH, WW, b + (size_t)k * HH * WW)) { free(b); return 1; }
    for (int cy = 0; cy < HH; cy += 4) for (int cx = 0; cx < WW; cx += 4) {
        float v[64], r[64] = {0};
        for (int k = 0; k < 64; k++) v[k] = (float)((b[((size_t)(k >> 4) * HH + cy + (k >> 2 & 3)) * WW + cx + (k & 3)] - J->lcen) / J->lsc);
        for (int c = 0; c < J->lc; c++) {
            float s = 0.f; for (int k = 0; k < 64; k++) s += J->lb->b[c][k] * v[k];
            if (J->lstep > 0) { const float lim = c ? 127.f : 32767.f, st = (float)J->lstep; s = fminf(lim, fmaxf(-lim, rintf(s / st))) * st; }
            for (int k = 0; k < 64; k++) r[k] += s * J->lb->b[c][k];
        }
        for (int k = 16 * dz; k < 16 * dz + 16; k++) {
            const int64_t y = yc + cy + (k >> 2 & 3) - Y0, x = xc + cx + (k & 3) - X0;
            if (y >= 0 && y < H && x >= 0 && x < W) ct[y * W + x] = fminf(255.f, fmaxf(0.f, rintf((float)(r[k] * J->lsc + J->lcen))));   /* a uint8 decode (float residue broke flat masked runs' ties: 1667 64-coefficient check 0.37 vs raw 0.45) */
        }
    }
    free(b); return 0;
}

/* --lat-codec FILE: plane zn through the learned codec (lcodec.h): every 4^3 cell's indices from its 3^3 neighbourhood (the encoder side
   reads the planes around), rebuilt by the decoder (tier a: the linear rows; tier b: + the nonlinear term) */
static int lcd_rt(const gt_job *J, int64_t zn, int64_t Y0, int64_t X0, int H, int W, float *ct) {
    const lcodec *L = J->lcd; const int64_t zc = (zn & ~(int64_t)3) - 4, yc = (Y0 & ~(int64_t)3) - 4, xc = (X0 & ~(int64_t)3) - 4; const int dz = (int)(zn - zc - 4);
    const int HH = (int)(((Y0 + H + 3) & ~(int64_t)3) + 4 - yc), WW = (int)(((X0 + W + 3) & ~(int64_t)3) + 4 - xc), ny = HH / 4, nx = WW / 4;
    uint8_t *b = malloc((size_t)12 * HH * WW); float *xg = malloc(sizeof(float) * 64 * 3 * (size_t)ny * nx); uint8_t *anyc = malloc((size_t)3 * ny * nx);
    if (!b || !xg || !anyc) { free(b); free(xg); free(anyc); return 1; }
    for (int k = 0; k < 12; k++) if (rd(J->ct, zc + k, yc, xc, HH, WW, b + (size_t)k * HH * WW)) { free(b); free(xg); free(anyc); return 1; }
    for (int cz = 0; cz < 3; cz++) for (int cy = 0; cy < ny; cy++) for (int cx = 0; cx < nx; cx++) {
        float v[64]; int a = 0; const size_t ci = ((size_t)cz * ny + cy) * nx + cx;
        for (int k = 0; k < 64; k++) { const uint8_t u = b[((size_t)(4 * cz + (k >> 4)) * HH + 4 * cy + (k >> 2 & 3)) * WW + 4 * cx + (k & 3)]; v[k] = u; a |= u; }
        anyc[ci] = a != 0;
        for (int j = 0; j < 64; j++) { float s = 0.f; for (int k = 0; k < 64; k++) s += J->lb->b[j][k] * v[k]; xg[ci * 64 + j] = s / L->sd_ref; }
    }
    float f[LCD_NF], cx_[LCD_NCX]; int16_t q[64]; uint8_t r[64];
    for (int cy = 1; cy < ny - 1; cy++) for (int cx = 1; cx < nx - 1; cx++) {
        const size_t ci = ((size_t)ny + cy) * nx + cx;
        lcodec_feat(L, xg, 3, ny, nx, 1, cy, cx, f); lcodec_encode_cell(L, f, xg + ci * 64, anyc[ci], q);
        lcodec_ctx(L, f, xg[ci * 64], cx_); lcodec_rebuild(L, J->lb, q, cx_, r);
        for (int k = 16 * dz; k < 16 * dz + 16; k++) {
            const int64_t y = yc + 4 * cy + (k >> 2 & 3) - Y0, x = xc + 4 * cx + (k & 3) - X0;
            if (y >= 0 && y < H && x >= 0 && x < W) ct[y * W + x] = r[k];
        }
    }
    free(b); free(xg); free(anyc); return 0;
}
/* --vc-rt Q: plane zn as plain volcomp at quality Q stores it: the 128^3 chunks (global grid) under the plane encoded and decoded */
static int vc_rt(const gt_job *J, int64_t zn, int64_t Y0, int64_t X0, int H, int W, float *ct) {
    const z3_meta *m = z3_meta_of(J->ct); const int64_t z0 = zn & ~(int64_t)127;
    uint8_t *ch = malloc(1 << 21), *rb = malloc(1 << 21), *enc = malloc(VOLCOMP_ENCODE_BOUND); int rc = 0;
    if (!ch || !rb || !enc) { free(ch); free(rb); free(enc); return 1; }
    for (int64_t y0 = Y0 & ~(int64_t)127; y0 < Y0 + H && !rc; y0 += 128) for (int64_t x0 = X0 & ~(int64_t)127; x0 < X0 + W && !rc; x0 += 128) {
        memset(ch, 0, 1 << 21);
        const int64_t o[3] = {z0, y0 < 0 ? 0 : y0, x0 < 0 ? 0 : x0}, n[3] = {z0 + 128 > m->shape[0] ? m->shape[0] - z0 : 128,
                          (y0 + 128 > m->shape[1] ? m->shape[1] : y0 + 128) - o[1], (x0 + 128 > m->shape[2] ? m->shape[2] : x0 + 128) - o[2]};
        if (n[0] > 0 && n[1] > 0 && n[2] > 0) {
            uint8_t *t = malloc((size_t)(n[0] * n[1] * n[2])); if (!t || z3_read(J->ct, o, n, t, 1)) { free(t); rc = 1; break; }
            for (int64_t a = 0; a < n[0]; a++) for (int64_t b = 0; b < n[1]; b++) memcpy(ch + ((size_t)a * 128 + (size_t)(o[1] - y0 + b)) * 128 + (size_t)(o[2] - x0), t + ((size_t)a * n[1] + b) * n[2], (size_t)n[2]);
            free(t);
        }
        size_t len = 0;
        if (volcomp_encode(ch, (float)J->vcq, enc, VOLCOMP_ENCODE_BOUND, &len) != VOLCOMP_OK || volcomp_decode(enc, len, rb, 1 << 21) != VOLCOMP_OK) { rc = 1; break; }
        const uint8_t *pl = rb + (size_t)(zn - z0) * 128 * 128;
        for (int yy = 0; yy < 128; yy++) for (int xx = 0; xx < 128; xx++) { const int64_t y = y0 + yy - Y0, x = x0 + xx - X0; if (y >= 0 && y < H && x >= 0 && x < W) ct[y * W + x] = pl[yy * 128 + xx]; }
    }
    free(ch); free(rb); free(enc); return rc;
}

/* --orient SIG,RHO: the 3-D structure tensor of the CT (gaussian derivative SIG, integration RHO, CT voxels; truncated at 3 sigma) on
   plane zn of the box, its 6 components (zz yy xx zy zx yx) into T[6][H][W]. Tiles of 128^2 with their halo read from one z stack
   (2K + 1 planes, K = Rs + Rr + 1); a tile whose CT plane or GT codes are empty stays 0 (no orientation there). */
static int rd3(z3 *z, int64_t z0, int64_t y0, int64_t x0, int nz, int ny, int nx, uint8_t *out) {   /* a block, zeros outside the array */
    const z3_meta *m = z3_meta_of(z); memset(out, 0, (size_t)nz * ny * nx);
    const int64_t lo[3] = {z0, y0, x0}, n[3] = {nz, ny, nx}; int64_t a[3], b[3];
    for (int k = 0; k < 3; k++) { a[k] = lo[k] < 0 ? 0 : lo[k]; b[k] = lo[k] + n[k] > m->shape[k] ? m->shape[k] : lo[k] + n[k]; if (b[k] <= a[k]) return 0; }
    const int64_t o2[3] = {a[0], a[1], a[2]}, n2[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    uint8_t *t = malloc((size_t)(n2[0] * n2[1] * n2[2])); if (!t) return 1;
    if (z3_read(z, o2, n2, t, 1)) { free(t); return 1; }
    for (int64_t zz = 0; zz < n2[0]; zz++) for (int64_t y = 0; y < n2[1]; y++)
        memcpy(out + ((size_t)(zz + a[0] - z0) * ny + (y + a[1] - y0)) * nx + (a[2] - x0), t + ((size_t)zz * n2[1] + y) * n2[2], (size_t)n2[2]);
    free(t); return 0;
}
static float *gk1(double s, int *R) { *R = (int)ceil(3 * s); float *k = malloc(sizeof(float) * (2 * *R + 1)); double sum = 0;
    for (int i = -*R; i <= *R; i++) sum += k[i + *R] = (float)exp(-0.5 * i * i / (s * s));
    for (int i = 0; i <= 2 * *R; i++) k[i] = (float)(k[i] / sum); return k; }
static int orient_tensor(const gt_job *J, int64_t zn, int64_t Y0, int64_t X0, int H, int W, const uint8_t *cdb, int QH, int QW, int64_t qy0, int64_t qx0, double f, float *T) {
    int Rs, Rr; float *ks = gk1(J->osig, &Rs), *kr = gk1(J->orho, &Rr); const int K = Rs + Rr + 1, NS = 2 * K + 1, SH = H + 2 * K, SW = W + 2 * K, TS = 128;
    uint8_t *st = malloc((size_t)NS * SH * SW); memset(T, 0, sizeof(float) * 6 * (size_t)H * W);
    if (!st || rd3(J->oct, zn - K, Y0 - K, X0 - K, NS, SH, SW, st)) { free(st); free(ks); free(kr); return 1; }
    const int M = Rr + 1, AYm = TS + 2 * (M + Rs), AXm = TS + 2 * M, NZ = 2 * Rr + 3, PYm = TS + 2 * Rr;
    float *A = malloc(sizeof(float) * (size_t)NS * AYm * AXm), *B = malloc(sizeof(float) * (size_t)NS * AXm * AXm), *G = malloc(sizeof(float) * (size_t)NZ * AXm * AXm),
          *P = malloc(sizeof(float) * 6 * (size_t)PYm * PYm), *Q = malloc(sizeof(float) * 6 * (size_t)PYm * TS);
    for (int ty = 0; ty < H; ty += TS) for (int tx = 0; tx < W; tx += TS) {
        const int th = ty + TS > H ? H - ty : TS, tw = tx + TS > W ? W - tx : TS;
        int any = 0; for (int y = ty; y < ty + th && !any; y++) for (int x = tx; x < tx + tw; x++) if (st[((size_t)K * SH + y + K) * SW + x + K]) { any = 1; break; }
        if (!any) continue;
        any = 0; { const int qa = (int)floor((Y0 + ty) * f) - (int)qy0, qb = (int)floor((Y0 + ty + th - 1) * f) - (int)qy0 + 1, xa = (int)floor((X0 + tx) * f) - (int)qx0, xb = (int)floor((X0 + tx + tw - 1) * f) - (int)qx0 + 1;
            for (int y = qa < 0 ? 0 : qa; y < (qb > QH ? QH : qb) && !any; y++) for (int x = xa < 0 ? 0 : xa; x < (xb > QW ? QW : xb); x++) if (cdb[(size_t)y * QW + x]) { any = 1; break; } }
        if (!any) continue;
        const int AY = th + 2 * (M + Rs), AX = tw + 2 * M, HY = th + 2 * M, PY = th + 2 * Rr, PX = tw + 2 * Rr;
        /* A: x-smoothed, rows ty - M - Rs .. , cols tx - M ..; stack index = plane coordinate + K */
        for (int z = 0; z < NS; z++) for (int y = 0; y < AY; y++) {
            const uint8_t *row = st + ((size_t)z * SH + (ty - M - Rs + y + K)) * SW; float *o = A + ((size_t)z * AY + y) * AX;
            for (int x = 0; x < AX; x++) { const int c = tx - M + x + K; float v = 0; for (int i = -Rs; i <= Rs; i++) v += ks[i + Rs] * row[c + i]; o[x] = v; }
        }
        for (int z = 0; z < NS; z++) for (int y = 0; y < HY; y++) { float *o = B + ((size_t)z * HY + y) * AX;
            for (int x = 0; x < AX; x++) { float v = 0; for (int i = -Rs; i <= Rs; i++) v += ks[i + Rs] * A[((size_t)z * AY + y + Rs + i) * AX + x]; o[x] = v; } }
        for (int p = 0; p < NZ; p++) { const int zc = K - Rr - 1 + p; float *o = G + (size_t)p * HY * AX;
            for (size_t i = 0; i < (size_t)HY * AX; i++) { float v = 0; for (int d = -Rs; d <= Rs; d++) v += ks[d + Rs] * B[(size_t)(zc + d) * HY * AX + i]; o[i] = v; } }
        /* gradients (central differences) at planes 1 .. NZ - 2, z-integrated with the RHO kernel into P (PY x PX) */
        memset(P, 0, sizeof(float) * 6 * (size_t)PY * PX);
        for (int p = 1; p <= NZ - 2; p++) { const float wz = kr[p - 1]; const float *g = G + (size_t)p * HY * AX;
            for (int y = 0; y < PY; y++) for (int x = 0; x < PX; x++) {
                const size_t i = (size_t)(y + 1) * AX + x + 1;
                const float gz = 0.5f * (g[i + (size_t)HY * AX] - g[i - (size_t)HY * AX]), gy = 0.5f * (g[i + AX] - g[i - AX]), gx = 0.5f * (g[i + 1] - g[i - 1]);
                float *q = P + (size_t)y * PX + x; const size_t sP = (size_t)PY * PX;
                q[0] += wz * gz * gz; q[sP] += wz * gy * gy; q[2 * sP] += wz * gx * gx; q[3 * sP] += wz * gz * gy; q[4 * sP] += wz * gz * gx; q[5 * sP] += wz * gy * gx;
            } }
        for (int c = 0; c < 6; c++) {   /* in-plane RHO integration: x then y, to the tile core */
            const float *pc = P + (size_t)c * PY * PX; float *qc = Q + (size_t)c * PY * tw;
            for (int y = 0; y < PY; y++) for (int x = 0; x < tw; x++) { float v = 0; for (int i = -Rr; i <= Rr; i++) v += kr[i + Rr] * pc[(size_t)y * PX + x + Rr + i]; qc[(size_t)y * tw + x] = v; }
            for (int y = 0; y < th; y++) for (int x = 0; x < tw; x++) { float v = 0; for (int i = -Rr; i <= Rr; i++) v += kr[i + Rr] * qc[(size_t)(y + Rr + i) * tw + x];
                T[(size_t)c * H * W + (size_t)(ty + y) * W + tx + x] = v; }
        }
    }
    free(st); free(A); free(B); free(G); free(P); free(Q); free(ks); free(kr); return 0;
}

static void gt_plane(void *ctx, int a0, int a1, int t) {
    gt_job *J = ctx; (void)t;
    for (int job = a0; job < a1; job++) {
        const gt_box *B = &J->bx[job / J->nz]; const int zi = job % J->nz; gt_row **R = &J->rows[job]; int *nr = &J->nrow[job], *cp = &J->cap[job];
        const double s = J->bum / J->ctum, z0 = B->b[0] * s, y0 = B->b[1] * s, x0 = B->b[2] * s, nz = B->b[3] * s, ny = B->b[4] * s, nx = B->b[5] * s;
        const int64_t zn = (int64_t)(z0 + (zi + 0.5) * nz / J->nz), Y0 = (int64_t)y0, X0 = (int64_t)x0; const int H = (int)ny, W = (int)nx;
        uint8_t *ctb = malloc((size_t)H * W), *rvb = J->rvz[B - J->bx] ? malloc((size_t)H * W) : nullptr, *fhb = J->fhz[B - J->bx] ? malloc((size_t)H * W) : nullptr, *fvb = J->fhz[B - J->bx] ? malloc((size_t)H * W) : nullptr;
        const double f = J->ctum / J->cum; const int64_t qz = (int64_t)(zn * f), qy0 = (int64_t)floor(y0 * f), qx0 = (int64_t)floor(x0 * f);
        const int QH = (int)ceil(ny * f) + 2, QW = (int)ceil(nx * f) + 2; uint8_t *cdb = malloc((size_t)QH * QW);
        const z3_meta *cm = z3_meta_of(J->cl); const double f2 = J->ctum / J->clum; const int CH = (int)cm->shape[1], CW = (int)cm->shape[2]; uint8_t *clb = malloc((size_t)CH * CW);
        int bad = rd(J->ct, zn, Y0, X0, H, W, ctb) || rd(J->cd, qz, qy0, qx0, QH, QW, cdb) || rd(J->cl, (int64_t)(zn * f2), 0, 0, CH, CW, clb);
        /* the prediction stores: per box ({box} in the root: box-local coordinates, predict --box writes the box alone) or global */
        const int64_t pz = J->plocal ? (int64_t)z0 : 0, py = J->plocal ? Y0 : 0, px = J->plocal ? X0 : 0; const int bi = (int)(B - J->bx);
        if (!bad && rvb) bad = rd(J->rvz[bi], zn - pz, Y0 - py, X0 - px, H, W, rvb);
        if (!bad && fhb) bad = rd(J->fhz[bi], zn - pz, Y0 - py, X0 - px, H, W, fhb) || rd(J->fvz[bi], zn - pz, Y0 - py, X0 - px, H, W, fvb);
        if (bad) { pthread_mutex_lock(&J->mu); snprintf(J->err, sizeof J->err, "box %s z %lld: %s", B->name, (long long)zn, z3_error()); pthread_mutex_unlock(&J->mu); goto next; }
        int any = 0; for (size_t i = 0; i < (size_t)QH * QW; i++) { if (cdb[i] < 1 || cdb[i] > 252) cdb[i] = 0; any |= cdb[i]; }
        if (!any) goto next;
        {
        double sy = 0, sx = 0; long nm = 0;
        for (int y = 0; y < CH; y++) for (int x = 0; x < CW; x++) if (clb[(size_t)y * CW + x]) { sy += y; sx += x; nm++; }
        if (!nm) goto next;
        const double cy = (sy / nm + 0.5) / f2 - 0.5, cx = (sx / nm + 0.5) / f2 - 0.5, cyl = cy - y0, cxl = cx - x0;
        float *ct = malloc(sizeof(float) * (size_t)H * W), *rvf = rvb ? malloc(sizeof(float) * (size_t)H * W) : nullptr, *fhf = fhb ? malloc(sizeof(float) * (size_t)H * W) : nullptr, *fvf = fhb ? malloc(sizeof(float) * (size_t)H * W) : nullptr;
        for (size_t i = 0; i < (size_t)H * W; i++) { ct[i] = ctb[i]; if (rvf) rvf[i] = rvb[i]; if (fhf) { fhf[i] = fhb[i]; fvf[i] = fvb[i]; } }
        if ((J->vcq > 0 ? vc_rt(J, zn, Y0, X0, H, W, ct) : J->lcd ? lcd_rt(J, zn, Y0, X0, H, W, ct) : 0) ) { pthread_mutex_lock(&J->mu); snprintf(J->err, sizeof J->err, "box %s z %lld: --lat-codec / --vc-rt read", B->name, (long long)zn); pthread_mutex_unlock(&J->mu); }
        else if (!J->lcd && J->vcq <= 0 && J->lb && lat_rt(J, zn, Y0, X0, H, W, ct)) { pthread_mutex_lock(&J->mu); snprintf(J->err, sizeof J->err, "box %s z %lld: --lat-rt read: %s", B->name, (long long)zn, z3_error()); pthread_mutex_unlock(&J->mu); }
        const size_t qn = (size_t)QH * QW; float *lab = malloc(sizeof(float) * qn), *gy = malloc(sizeof(float) * qn), *gx = malloc(sizeof(float) * qn), *jyy = malloc(sizeof(float) * qn), *jxx = malloc(sizeof(float) * qn), *jyx = malloc(sizeof(float) * qn);
        for (size_t i = 0; i < qn; i++) lab[i] = cdb[i] > 0;
        gauss2(lab, QH, QW, 1.0); sobel(lab, QH, QW, 0, gy); sobel(lab, QH, QW, 1, gx);
        for (size_t i = 0; i < qn; i++) { jyy[i] = gy[i] * gy[i]; jxx[i] = gx[i] * gx[i]; jyx[i] = gy[i] * gx[i]; }
        gauss2(jyy, QH, QW, 3); gauss2(jxx, QH, QW, 3); gauss2(jyx, QH, QW, 3);
        float *OT = nullptr;
        if (J->orient) {
            OT = malloc(sizeof(float) * 6 * (size_t)H * W);
            if (!OT || orient_tensor(J, zn, Y0, X0, H, W, cdb, QH, QW, qy0, qx0, f, OT)) { pthread_mutex_lock(&J->mu); snprintf(J->err, sizeof J->err, "box %s z %lld: --orient read: %s", B->name, (long long)zn, z3_error()); pthread_mutex_unlock(&J->mu); free(OT); OT = nullptr; }
        }
        const int inside = cyl >= 0 && cyl < ny && cxl >= 0 && cxl < nx;
        const double rmin = inside ? 0 : hypot(fmin(fmax(cyl, 0), ny) - cyl, fmin(fmax(cxl, 0), nx) - cxl);
        double rmax = 0; { const double c[4][2] = {{0, 0}, {0, nx}, {ny, 0}, {ny, nx}}; for (int k = 0; k < 4; k++) rmax = fmax(rmax, hypot(c[k][0] - cyl, c[k][1] - cxl)); }
        const int nr0 = (int)rmin, nr1 = (int)rmax + 2, NR = nr1 - nr0; if (NR <= 0) goto freeall;
        double *C = malloc(sizeof(double) * NR), *ry = malloc(sizeof(double) * NR), *rx = malloc(sizeof(double) * NR), *rr = malloc(sizeof(double) * NR);
        uint8_t *mat = malloc(NR), *pk = malloc(NR), *q = malloc(NR), *a12 = malloc(NR), *a21 = malloc(NR), *b12 = malloc(NR), *b21 = malloc(NR), *rok = malloc(NR), *fok = malloc(NR);
        int8_t *rs = malloc(NR), *fs = malloc(NR), *os = malloc(NR); uint8_t *o12 = malloc(NR), *o21 = malloc(NR); double *RV = malloc(sizeof(double) * NR), *FH = malloc(sizeof(double) * NR); uint8_t *rpk = malloc(NR), *hpk = malloc(NR), *rmat = malloc(NR), *hmat = malloc(NR); int *qyv = malloc(sizeof(int) * NR), *qxv = malloc(sizeof(int) * NR);
        for (int k = 0; k < J->nth; k++) {
            const double th = k * 2 * M_PI / J->nth + M_PI / J->nth, sn = sin(th), cs = cos(th);
            int i0 = -1, i1 = -1, ninb = 0;
            for (int i = 0; i < NR; i++) { const double r = nr0 + i, y = cyl + r * sn, x = cxl + r * cs; if (y >= 0 && x >= 0 && y < ny && x < nx) { if (i0 < 0) i0 = i; i1 = i; ninb++; } }
            if (ninb < 32) continue;
            const int n = i1 - i0 + 1;
            for (int j = 0; j < n; j++) {
                const double r = nr0 + i0 + j, y = cyl + r * sn, x = cxl + r * cs; rr[j] = r; ry[j] = y; rx[j] = x;
                C[j] = bil(ct, H, W, y, x); mat[j] = !isnan(C[j]) && C[j] > 0;
                const int qy = (int)floor((y + y0) * f) - (int)qy0, qx = (int)floor((x + x0) * f) - (int)qx0; qyv[j] = qy; qxv[j] = qx;
                q[j] = qy >= 0 && qx >= 0 && qy < QH && qx < QW ? cdb[(size_t)qy * QW + qx] : 0;
                rs[j] = 0; rok[j] = 0; fs[j] = 0; fok[j] = 0; RV[j] = FH[j] = NAN; rmat[j] = hmat[j] = 0;
                if (rvf) { const double v = bil(rvf, H, W, y, x); if (!isnan(v) && v >= 1) { rok[j] = 1; RV[j] = v; rmat[j] = 1; rs[j] = v > 128 + J->rvthr ? 2 : v < 128 - J->rvthr ? 1 : 0; } }   /* 2 = R, 1 = V */
                if (fhf) { const double h = bil(fhf, H, W, y, x), v = bil(fvf, H, W, y, x);
                           if (!isnan(h) && !isnan(v)) { fok[j] = mat[j]; FH[j] = h - v; hmat[j] = mat[j]; fs[j] = h >= J->fthr && h >= v + J->fmar ? 2 : v >= J->fthr && v >= h + J->fmar ? 1 : 0; } }   /* 2 = H, 1 = V */
                os[j] = 0;
                if (OT && !isnan(C[j]) && C[j] >= J->oiso) {
                    /* the in-sheet tangent from the GT label's in-slice normal (not the ray: oblique sheets would leak their own faces into
                       the tangential gradient); the gradient tensor on the plane (z, tangent); its dominant gradient along z = fibres along
                       the tangent = H (perpendicular to the scroll axis, taken as z), along the tangent = V */
                    const int yy = qy < 0 ? 0 : qy >= QH ? QH - 1 : qy, xx = qx < 0 ? 0 : qx >= QW ? QW - 1 : qx; const size_t qi = (size_t)yy * QW + xx;
                    const double a2 = 0.5 * atan2(2 * jyx[qi], jyy[qi] - jxx[qi]), ty_ = -sin(a2), tx_ = cos(a2); double t6[6]; int okt = 1;
                    for (int c = 0; c < 6 && okt; c++) { t6[c] = bil(OT + (size_t)c * H * W, H, W, y, x); okt = !isnan(t6[c]); }
                    if (okt) { const double tzz = t6[0], ttt = ty_ * ty_ * t6[1] + 2 * ty_ * tx_ * t6[5] + tx_ * tx_ * t6[2], tzt = ty_ * t6[3] + tx_ * t6[4];
                               const double d = tzz - ttt, e = sqrt(d * d + 4 * tzt * tzt), tr = tzz + ttt;
                               if (tr > 0 && e / tr >= J->ocoh) os[j] = d / e > J->omar ? 2 : d / e < -J->omar ? 1 : 0; }   /* 2 = H, 1 = V */
                }
            }
            peaks(C, mat, n, J->sig, J->pkL, J->pkP, pk); peaks(RV, rmat, n, J->sig, J->pkL, J->pkP, rpk); peaks(FH, hmat, n, J->sig, J->pkL, J->pkP, hpk);   /* (rv / H - V peaks: recto-ruler's "peaks of the fused signal") */
            flips(rs, n, a12, a21); flips(fs, n, b12, b21); flips_p(os, n, J->opers, o12, o21);   /* a12: V -> R, a21: R -> V; b12: V -> H, b21: H -> V (outward) */
            double pr = -1; int pq = 0;
            for (int i = 0; i < n;) {
                if (!q[i]) { i++; continue; }
                int e = i, last = i;
                while (e < n && e - last <= 2) {
                    if (q[e]) { int dc = ((int)q[e] - (int)q[i]) % 252; if (dc < 0) dc += 252; if (dc > 126) dc -= 252; if (abs(dc) > 9) break; last = e; }
                    e++;
                }
                if (e < n && e - last <= 2) e = last + 1;
                const double cr0 = 0.5 * (i + last); const int ic = (int)lround(cr0); const int qc = q[ic] ? q[ic] : q[i];
                if (pr >= 0) {
                    double dq = fmod((qc - pq) / 18.0, 14.0); if (dq < 0) dq += 14; if (dq > 7) dq -= 14;
                    const int ia = (int)lround(pr), ib = ic; int allm = ib > ia; for (int j = ia; j <= ib && allm; j++) allm = mat[j];
                    if (allm && dq > 0.5 && dq < 2.5) {
                        double ns = 0;
                        for (int s2 = 0; s2 < 2; s2++) { const int kk = s2 ? ib : ia, yy = qyv[kk] < 0 ? 0 : qyv[kk] >= QH ? QH - 1 : qyv[kk], xx = qxv[kk] < 0 ? 0 : qxv[kk] >= QW ? QW - 1 : qxv[kk];
                            const size_t qi = (size_t)yy * QW + xx; const double a2 = 0.5 * atan2(2 * jyx[qi], jyy[qi] - jxx[qi]); ns += fabs(cos(a2) * sn + sin(a2) * cs); }
                        gt_row w = {.box = (int)(B - J->bx), .z = (int)zn, .rmm = (float)(0.5 * (rr[ia] + rr[ib]) * J->ctum / 1000), .dq = (float)dq, .gap = (float)((ib - ia) * 0.5 * ns * J->ctum / dq), .rvok = 1, .fbok = 1, .orok = OT != nullptr};
                        for (int j = ia; j < ib; j++) { w.npk += pk[j]; w.rvpk += rpk[j]; w.fhpk += hpk[j]; w.rvf += a12[j]; w.rvb += a21[j]; w.fbf += b12[j]; w.fbb += b21[j]; w.orf += o12[j]; w.orb += o21[j]; if (!rok[j]) w.rvok = 0; if (!fok[j]) w.fbok = 0; }
                        if (!rvf) w.rvok = 0; if (!fhf) w.fbok = 0;
                        if (*nr == *cp) { *cp = *cp ? 2 * *cp : 256; *R = realloc(*R, sizeof(gt_row) * *cp); }
                        (*R)[(*nr)++] = w;
                    }
                }
                pr = cr0; pq = qc; i = e;
            }
        }
        free(C); free(ry); free(rx); free(rr); free(mat); free(pk); free(q); free(a12); free(a21); free(b12); free(b21); free(rok); free(fok); free(rs); free(fs); free(RV); free(FH); free(rpk); free(hpk); free(rmat); free(hmat); free(qyv); free(qxv); free(os); free(o12); free(o21);
        freeall:
        free(OT); free(ct); free(rvf); free(fhf); free(fvf); free(lab); free(gy); free(gx); free(jyy); free(jxx); free(jyx);
        }
        next:
        free(ctb); free(rvb); free(fhb); free(fvb); free(cdb); free(clb);
        pthread_mutex_lock(&J->mu); J->done++; pthread_mutex_unlock(&J->mu);
    }
}

static z3 *gt_open(const char *root, const char *key) {
    store *s = store_open(root); if (!s) { fprintf(stderr, "gt-trans: cannot open %s\n", root); return nullptr; }
    z3 *z = z3_open(s, key, nullptr); if (!z) fprintf(stderr, "gt-trans: %s / %s: %s\n", root, key, z3_error());
    return z;
}

int cmd_gt_trans(int argc, char **argv) {
    if (argc < 14) {
        fprintf(stderr, "usage: ufsm gt-trans ROOT CTKEY CTUM CODEROOT CODEKEY CODEUM CENTREKEY CENTREUM BOXES_TSV SCROLL BOXUM OUT.tsv [--nz 8] [--nth 360]\n"
                        "                     [--rv ROOT KEY] [--fh ROOT KEY --fv ROOT KEY] [--rv-thr 8] [--fib-thr 8] [--fib-margin 4] [--ct-sig 1.2] [--threads N]\n"
                        "                     [--lat-rt BASIS.lcb --lat-c 29 --lat-step 0.25 --lat-norm CENTRE,SCALE [--lat-codec CODEC.lcd]] [--vc-rt Q] [--ct-l 12 --ct-p 24]\n"
                        "                     [--orient SIG,RHO [--orient-coh 0.2] [--orient-margin 0.2] [--orient-iso 1] [--orient-persist UM] [--orient-ct ROOT KEY]]\n"
                        "  CT peaks, recto / verso and fibre-direction flips per GT turn by normal gap (src/gttrans.c)\n");
        return 2;
    }
    gt_job J = {.ctum = atof(argv[4]), .cum = atof(argv[7]), .clum = atof(argv[9]), .bum = atof(argv[12]), .nz = opt_atoi("--nz", gt_opt(argc, argv, "--nz", "8")),
                .nth = opt_atoi("--nth", gt_opt(argc, argv, "--nth", "360")), .rvthr = opt_atof("--rv-thr", gt_opt(argc, argv, "--rv-thr", "8")),
                .fthr = opt_atof("--fib-thr", gt_opt(argc, argv, "--fib-thr", "8")), .fmar = opt_atof("--fib-margin", gt_opt(argc, argv, "--fib-margin", "4")),
                .sig = opt_atof("--ct-sig", gt_opt(argc, argv, "--ct-sig", "1.2")), .pkL = opt_atoi("--ct-l", gt_opt(argc, argv, "--ct-l", "12")),
                .pkP = opt_atoi("--ct-p", gt_opt(argc, argv, "--ct-p", "24")), .ocoh = opt_atof("--orient-coh", gt_opt(argc, argv, "--orient-coh", "0.2")),
                .omar = opt_atof("--orient-margin", gt_opt(argc, argv, "--orient-margin", "0.2")), .oiso = opt_atof("--orient-iso", gt_opt(argc, argv, "--orient-iso", "1"))};
    { const double pu = opt_atof("--orient-persist", gt_opt(argc, argv, "--orient-persist", "0")); J.opers = pu > 0 ? (int)ceil(pu / J.ctum) : 1; }
    { const char *o = gt_opt(argc, argv, "--orient", nullptr);
      if (o) { if (sscanf(o, "%lf,%lf", &J.osig, &J.orho) != 2 || J.osig <= 0 || J.orho <= 0) { fprintf(stderr, "gt-trans: --orient SIG,RHO (CT voxels)\n"); return 1; } J.orient = 1; } }
    pthread_mutex_init(&J.mu, nullptr);
    const char *lbp = gt_opt(argc, argv, "--lat-rt", nullptr);
    if (lbp) {
        lat_basis *lb = malloc(sizeof *lb); if (lat_basis_load(lbp, lb)) { fprintf(stderr, "gt-trans: --lat-rt: cannot read %s\n", lbp); return 1; }
        J.lb = lb; J.lc = opt_atoi("--lat-c", gt_opt(argc, argv, "--lat-c", "29")); J.lstep = opt_atof("--lat-step", gt_opt(argc, argv, "--lat-step", "0.25"));
        if (J.lc < 1 || J.lc > 64) { fprintf(stderr, "gt-trans: --lat-c 1..64\n"); return 1; }
        const char *nm = gt_opt(argc, argv, "--lat-norm", nullptr);
        if (!nm || sscanf(nm, "%lf,%lf", &J.lcen, &J.lsc) != 2 || J.lsc <= 0) { fprintf(stderr, "gt-trans: --lat-rt needs --lat-norm CENTRE,SCALE (predict's --scan-norm line)\n"); return 1; }
        fprintf(stderr, "gt-trans: CT through the latent codec: %d coefficients of %s, step %g, norm (v - %g) / %g\n", J.lc, lbp, J.lstep, J.lcen, J.lsc);
    }
    const char *lcp = gt_opt(argc, argv, "--lat-codec", nullptr); J.vcq = opt_atof("--vc-rt", gt_opt(argc, argv, "--vc-rt", "0"));
    if (lcp) {
        if (!J.lb) { fprintf(stderr, "gt-trans: --lat-codec needs --lat-rt BASIS (its KLT frame)\n"); return 1; }
        lcodec *L = malloc(sizeof *L); if (lcodec_read(L, lcp)) { fprintf(stderr, "gt-trans: --lat-codec: cannot read %s\n", lcp); return 1; }
        J.lcd = L; fprintf(stderr, "gt-trans: CT through the learned codec %s (cl %d, hid2 %d, mean step %.3f x sd %.2f)\n", lcp, L->cl, L->hid2, L->mstep, L->sd_ref);
    }
    if (J.vcq > 0) fprintf(stderr, "gt-trans: CT through volcomp q %g\n", J.vcq);
    if (!(J.ct = gt_open(argv[2], argv[3])) || !(J.cd = gt_open(argv[5], argv[6])) || !(J.cl = gt_open(argv[2], argv[8]))) return 1;
    J.oct = J.ct;
    if (J.orient && gt_opt2(argc, argv, "--orient-ct", 1) && !(J.oct = gt_open(gt_opt2(argc, argv, "--orient-ct", 0), gt_opt2(argc, argv, "--orient-ct", 1)))) return 1;
    if (J.orient) fprintf(stderr, "gt-trans: fibre orientation: structure tensor sigma %g, rho %g CT voxels, coherence >= %g, margin %g, CT >= %g, persistence %d samples\n", J.osig, J.orho, J.ocoh, J.omar, J.oiso, J.opers);
    const char *rr = gt_opt2(argc, argv, "--rv", 0), *rk = gt_opt2(argc, argv, "--rv", 1), *hr = gt_opt2(argc, argv, "--fh", 0), *hk = gt_opt2(argc, argv, "--fh", 1),
               *vr = gt_opt2(argc, argv, "--fv", 0), *vk = gt_opt2(argc, argv, "--fv", 1);
    if (hr && !vr) { fprintf(stderr, "gt-trans: --fh needs --fv\n"); return 1; }
    /* the boxes of SCROLL */
    FILE *bf = fopen(argv[10], "r"); if (!bf) { fprintf(stderr, "gt-trans: cannot read %s\n", argv[10]); return 1; }
    gt_box *bx = nullptr; int nb = 0, cb = 0; char line[1024];
    while (fgets(line, sizeof line, bf)) {
        if (line[0] == '#') continue;
        char nm[64], sc[64]; long long v[6];
        if (sscanf(line, "%63s %63s %lld %lld %lld %lld %lld %lld", nm, sc, &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) != 8 || strcmp(sc, argv[11])) continue;
        if (nb == cb) bx = realloc(bx, sizeof *bx * (cb = cb ? 2 * cb : 64));
        snprintf(bx[nb].name, sizeof bx[nb].name, "%s", nm); for (int k = 0; k < 6; k++) bx[nb].b[k] = v[k]; nb++;
    }
    fclose(bf);
    if (!nb) { fprintf(stderr, "gt-trans: no boxes of %s in %s\n", argv[11], argv[10]); return 1; }
    /* prediction stores: ROOT with {box} = one store per box (box-local coordinates; a missing one: that box without the signal) */
    J.rvz = calloc(nb, sizeof(z3 *)); J.fhz = calloc(nb, sizeof(z3 *)); J.fvz = calloc(nb, sizeof(z3 *)); J.plocal = (rr && strstr(rr, "{box}")) || (hr && strstr(hr, "{box}"));
    int nrv = 0, nfb = 0;
    for (int b = 0; b < nb; b++) {
        char p[1024];
        #define GT_SUB(t) do { const char *c = strstr(t, "{box}"); if (c) snprintf(p, sizeof p, "%.*s%s%s", (int)(c - t), t, bx[b].name, c + 5); else snprintf(p, sizeof p, "%s", t); } while (0)
        if (rr) { GT_SUB(rr); struct stat st; if (!J.plocal || !stat(p, &st)) { if (!(J.rvz[b] = gt_open(p, rk))) return 1; nrv++; } }
        if (hr) { GT_SUB(hr); struct stat st; if (!J.plocal || !stat(p, &st)) { if (!(J.fhz[b] = gt_open(p, hk))) return 1; GT_SUB(vr); if (!(J.fvz[b] = gt_open(p, vk))) return 1; nfb++; } }
        #undef GT_SUB
    }
    if (rr || hr) fprintf(stderr, "gt-trans: rv stores for %d, fibre stores for %d of %d boxes\n", nrv, nfb, nb);
    const int nj = nb * J.nz; J.bx = bx; J.nbox = nb; J.rows = calloc(nj, sizeof *J.rows); J.nrow = calloc(nj, sizeof(int)); J.cap = calloc(nj, sizeof(int));
    fit_set_threads(opt_atoi("--threads", gt_opt(argc, argv, "--threads", "0")));
    fit_par(nj, gt_plane, &J);
    if (*J.err) { fprintf(stderr, "gt-trans: %s\n", J.err); return 1; }
    FILE *fo = fopen(argv[13], "w"); if (!fo) { fprintf(stderr, "gt-trans: cannot write %s\n", argv[13]); return 1; }
    fprintf(fo, "box\tz\tr_mm\tdq\tgap_um\tnpk\trv_VR\trv_RV\trv_ok\tfib_VH\tfib_HV\tfib_ok\trv_pk\tfib_HVpk\tor_VH\tor_HV\tor_ok\n");
    enum { NG = 9 }; const double GB[NG + 1] = {0, 70, 85, 100, 115, 130, 150, 180, 230, 1e9};
    double sdq[NG + 1] = {0}, spk[NG + 1] = {0}, off[NG + 1] = {0}, rvdq[NG + 1] = {0}, rvf[NG + 1] = {0}, rvb[NG + 1] = {0}, rvoff[NG + 1] = {0}, fbdq[NG + 1] = {0}, fbf[NG + 1] = {0}, fbb[NG + 1] = {0}, fboff[NG + 1] = {0}, rvp[NG + 1] = {0}, rvpoff[NG + 1] = {0}, rvct[NG + 1] = {0}, rvctoff[NG + 1] = {0}, fhp[NG + 1] = {0}, fhpoff[NG + 1] = {0}, orf[NG + 1] = {0}, orb[NG + 1] = {0}, oroff[NG + 1] = {0};
    long cnt[NG + 1] = {0}, rvn[NG + 1] = {0}, fbn[NG + 1] = {0}, tot = 0;
    for (int j = 0; j < nj; j++) for (int i = 0; i < J.nrow[j]; i++) {
        const gt_row *w = &J.rows[j][i]; tot++;
        fprintf(fo, "%s\t%d\t%.3f\t%.3f\t%.1f\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", bx[w->box].name, w->z, w->rmm, w->dq, w->gap, w->npk, w->rvf, w->rvb, w->rvok, w->fbf, w->fbb, w->fbok, w->rvpk, w->fhpk, w->orf, w->orb, w->orok);
        int g = 0; while (g < NG - 1 && w->gap >= GB[g + 1]) g++;
        for (int a = 0; a < 2; a++) { const int k = a ? NG : g;   /* (bucket g and "all") */
            cnt[k]++; sdq[k] += w->dq; spk[k] += w->npk; off[k] += fabs(w->npk - w->dq) >= 0.5;
            if (w->rvok) { rvn[k]++; rvdq[k] += w->dq; rvf[k] += w->rvf; rvb[k] += w->rvb; rvoff[k] += fabs(w->rvf - w->dq) >= 0.5; rvp[k] += w->rvpk; rvpoff[k] += fabs(w->rvpk - w->dq) >= 0.5; rvct[k] += w->npk; rvctoff[k] += fabs(w->npk - w->dq) >= 0.5; }
            if (w->orok) { orf[k] += w->orf; orb[k] += w->orb; oroff[k] += fabs(w->orf - w->dq) >= 0.5; }
            if (w->fbok) { fbn[k]++; fbdq[k] += w->dq; fbf[k] += w->fbf; fbb[k] += w->fbb; fboff[k] += fabs(w->fbf - w->dq) >= 0.5; fhp[k] += w->fhpk; fhpoff[k] += fabs(w->fhpk - w->dq) >= 0.5; } }
    }
    fclose(fo);
    printf("gt-trans %s: %ld GT intervals (%d boxes x %d z x %d rays)\n", argv[11], tot, nb, J.nz, J.nth);
    printf("  gap (um)      n   CT/turn  off   |  rv n    CT/turn off    V>R/turn R>V/turn  off    rvpk/turn off   |  fib n   V>H/turn H>V/turn  off    (H-V)pk/turn off%s\n", J.orient ? "   | CT orientation V>H/turn H>V/turn off" : "");
    for (int k = 0; k <= NG; k++) {
        if (cnt[k] < 50) continue;
        char lb[32]; if (k == NG) snprintf(lb, sizeof lb, "all"); else if (GB[k + 1] < 1e8) snprintf(lb, sizeof lb, "%.0f-%.0f", GB[k], GB[k + 1]); else snprintf(lb, sizeof lb, "%.0f-inf", GB[k]);
        printf("  %-9s %7ld   %.3f  %.3f", lb, cnt[k], spk[k] / sdq[k], off[k] / cnt[k]);
        if (rvn[k] >= 50) printf("  | %7ld   %.3f   %.3f   %.3f    %.3f    %.3f    %.3f   %.3f", rvn[k], rvct[k] / rvdq[k], rvctoff[k] / rvn[k], rvf[k] / rvdq[k], rvb[k] / rvdq[k], rvoff[k] / rvn[k], rvp[k] / rvdq[k], rvpoff[k] / rvn[k]); else printf("  |      -");
        if (fbn[k] >= 50) printf("   | %7ld   %.3f    %.3f    %.3f    %.3f   %.3f", fbn[k], fbf[k] / fbdq[k], fbb[k] / fbdq[k], fboff[k] / fbn[k], fhp[k] / fbdq[k], fhpoff[k] / fbn[k]); else printf("   |      -");
        if (J.orient) printf("   | %.3f    %.3f    %.3f", orf[k] / sdq[k], orb[k] / sdq[k], oroff[k] / cnt[k]);
        printf("\n");
    }
    for (int j = 0; j < nj; j++) free(J.rows[j]); free(J.rows); free(J.nrow); free(J.cap); free(bx);
    z3_close(J.ct); z3_close(J.cd); z3_close(J.cl); for (int b = 0; b < nb; b++) { if (J.rvz[b]) z3_close(J.rvz[b]); if (J.fhz[b]) z3_close(J.fhz[b]); if (J.fvz[b]) z3_close(J.fvz[b]); }
    free(J.rvz); free(J.fhz); free(J.fvz);
    return 0;
}
