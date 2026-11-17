/* ufsm unroll-qa: label-free sheet-identity QA of an unroll's fitted winding field (DESIGN "Label-free geometry gates", 2026-11-14).

   The fitter's own gates (fit-join agreement, wrap-check, slab coherence) measure self-consistency with the count-field reference and
   pass runs whose sheet identity is near random (Paris 4 PH1-3: GT ids 0.12-0.21). unroll-qa measures the winding W of a fit set against
   evidence the reference never integrated, per box (--box, the GT-box validation) or per tile over the whole fields box (default):

   - fibres (--fibres, UFIB of level-0 points, e.g. fiber-trace output): a fibre lies in one sheet, so W along it must stay within a turn.
     The fiber-judge measure: W sampled every native voxel along the polyline, steps unwrapped, 9-sample median, hysteresis --hyst turn;
     "persistent" = |net sheet change| per stretch, per metre of fibre (H / V family). The validated gate (2026-11-14, 44 GT boxes):
     H persistent > 130 /m flags 24 / 27 Paris 4 boxes with ids < 0.5 (PH1-3 at 552-1090 /m), 0 / 7 good 1667 boxes, 3 / 5 good Paris 4 boxes.
   - wind4: the network's winding-mod-4 planes (dz dy dx: W(x + d e_a) - W(x) mod 4 inside one inference window, a head count-field does
     not read) against the model's W difference over the same d: share within half a turn, per axis;
   - w4 loop closure: the net's own mod-4 differences summed around d x d squares (input consistency, model-free);
   - tilt: the angle between grad W and the network's sheet normal (W must be constant along a sheet); tangential |grad W| in turns / mm;
   - density ratio: |grad W| x the network's sheet spacing (1 = one turn per sheet);
   - phase: the share of material with phase confidence < 0.3, and of 16^3 blocks whose local coherence |<c e^{i(phi - 2 pi W)}>| < 0.5;
   - phase-unwrap witness: the net phase unwrapped along each axis over 16 / 48 cells against the model's W difference (integer misses);
   - CT (--ct ROOT KEY, the fields' native grid): the zero-CT (masked) share of material, and the share of 16^3 blocks whose CT ridge
     contrast |<c e^{-2 pi i W}>| / <|c|> (c = CT minus its 9^3 mean) is < 0.2.
   Gated (warn lines; exit 3 under --strict): conf_lo, pw48, w4 and, with fibres, fib_pers_h (thresholds at uq_gates); the others are
   reported for the record (tang_tpm and lcoh_lo track identity as well but add nothing to the gates; tilt, density ratio, w4 loop
   closure and the zero-CT share do not track it within a scroll).
   CPU only. */
#include "afv.h"
#include "optcheck.h"
#include "fit.h"
#include "json.h"
#include "store.h"
#include "sysinfo.h"
#include "zarr3.h"
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char *uq_opt(int argc, char **argv, const char *name, const char *def) { for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(name))) return argv[i + 1]; return def; }
static int uq_flag(int argc, char **argv, const char *name) { for (int i = 2; i < argc; i++) if (!strcmp(argv[i], opt_n(name))) return 1; return 0; }
static double uq_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static inline double uq_wrap(double d, double p) { return d - p * floor(d / p + 0.5); }   /* into [-p/2, p/2) */

/* per-box tallies (sums; a box's metrics are ratios of these) */
enum { A_MAT, A_RAD, A_W4N0, A_W4N1, A_W4N2, A_W4K0, A_W4K1, A_W4K2, A_LPN, A_LPB, A_TN, A_T20, A_TG, A_DRN, A_DRO, A_CONFLO, A_LCN, A_LCLO,
       A_PW16N, A_PW16B, A_PW48N, A_PW48B, A_CTN, A_CT0, A_CRN, A_CRLO, A_NACC };
enum { UQ_TH = 90 };   /* tilt histogram: 1 degree bins */
typedef struct { double a[A_NACC]; double th[UQ_TH]; } uq_acc;

/* the metrics of a box (NAN: not measured) */
enum { M_MAT, M_RAD, M_W4Z, M_W4Y, M_W4X, M_W4, M_LOOP, M_TILT, M_T20, M_TG, M_DRO, M_CONFLO, M_LCLO, M_PW16, M_PW48, M_CT0, M_CRLO,
       M_FARC, M_FPH, M_FPV, M_FP, M_FEH, M_NM };
static const char *uq_mname[M_NM] =   /* mgpu: process (constant names) */ {"mat", "rad_mm", "w4_z", "w4_y", "w4_x", "w4", "w4_loop", "tilt_med", "tilt20", "tang_tpm", "dr_out", "conf_lo",
                                     "lcoh_lo", "pw16", "pw48", "ct0", "ctr_lo", "fib_arc_m", "fib_pers_h", "fib_pers_v", "fib_pers", "fib_exc_h"};

typedef struct {
    int D, H, W; double h, um;
    const uint8_t *pl[14];          /* pc ps t0..t5 sp w dz dy dx dq */
    int w4k;                        /* the wind4 offset in cells (0: no planes) */
    const float *Wv, *T;            /* W and T = dir theta / 2 pi (turns) per cell */
    const float *ct, *ctm;          /* mean CT per cell, its 9^3 mean (null: no CT) */
    const float *cy, *cx; double oy, ox;   /* the axis (native, slab-local z; box-local y x) and the crop's native y x offset */
    uq_acc *acc;                    /* per thread */
} uq_box;

static inline size_t uq_ix(const uq_box *B, int z, int y, int x) { return ((size_t)z * B->H + y) * B->W + x; }
/* the continuous W difference W(b) - W(a): T is cut at +-1/2, W is not (it is U - T with U smooth) */
static inline double uq_dW(const uq_box *B, size_t a, size_t b) { const double dt = (double)B->T[b] - B->T[a]; return (double)B->Wv[b] - B->Wv[a] + dt - uq_wrap(dt, 1.0); }

/* block rows [b0, b1) of 16-cell z blocks (cells z in [16 b0, 16 b1)) */
enum { UQ_BL = 16 };
static void uq_rows(void *ctx, int b0, int b1, int t) {
    const uq_box *B = ctx; uq_acc *A = &B->acc[t]; const int D = B->D, H = B->H, W = B->W; const double h = B->h;
    const uint8_t *pc = B->pl[0], *ps = B->pl[1], *sp = B->pl[8], *wm = B->pl[9];
    const int z0 = b0 * UQ_BL, z1 = b1 * UQ_BL < D ? b1 * UQ_BL : D;
    for (int z = z0; z < z1; z++) {
        const double zn = (z + 0.5) * h; const int ai = (int)zn;
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
            const size_t i = uq_ix(B, z, y, x); if (wm[i] <= 64) continue;
            A->a[A_MAT]++; A->a[A_RAD] += hypot(B->oy + (y + 0.5) * h - B->cy[ai], B->ox + (x + 0.5) * h - B->cx[ai]) * B->um / 1000;
            const double cr = pc[i] - 128.0, si = ps[i] - 128.0, cf = hypot(cr, si) / 127;
            if (cf < 0.3) A->a[A_CONFLO]++;
            /* wind4 against the model, and the net's loop closure */
            const int k = B->w4k;
            if (k > 0) {
                const int nb[3] = {z + k < D, y + k < H, x + k < W}; const size_t st[3] = {(size_t)k * H * W, (size_t)k * W, (size_t)k};
                for (int a = 0; a < 3; a++) {
                    if (!nb[a] || B->pl[13][i] <= 76 || wm[i + st[a]] <= 64) continue;
                    const double e = fabs(uq_wrap(uq_dW(B, i, i + st[a]) - B->pl[10 + a][i] / 64.0, 4.0));
                    A->a[A_W4N0 + a]++; if (e < 0.5) A->a[A_W4K0 + a]++;
                }
                if (((z ^ y ^ x) & 3) == 0) for (int a = 0; a < 3; a++) for (int b = a + 1; b < 3; b++) {   /* every 4th cell */
                    if (!nb[a] || !nb[b]) continue;
                    const size_t ia = i + st[a], ib = i + st[b];
                    if (B->pl[13][i] <= 76 || B->pl[13][ia] <= 76 || B->pl[13][ib] <= 76) continue;
                    const double c = B->pl[10 + a][i] / 64.0 + B->pl[10 + b][ia] / 64.0 - B->pl[10 + a][ib] / 64.0 - B->pl[10 + b][i] / 64.0;
                    A->a[A_LPN]++; if (fabs(uq_wrap(c, 4.0)) > 0.5) A->a[A_LPB]++;
                }
            }
            /* grad W (central differences; one-sided at the box faces) against the net normal n n^T and spacing */
            if (z > 0 && z + 1 < D && y > 0 && y + 1 < H && x > 0 && x + 1 < W) {
                const double g[3] = {uq_dW(B, i - (size_t)H * W, i + (size_t)H * W) / (2 * h), uq_dW(B, i - W, i + W) / (2 * h), uq_dW(B, i - 1, i + 1) / (2 * h)};
                const double gn = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]) + 1e-12;
                const double t0 = B->pl[2][i] / 255.0, t1 = B->pl[3][i] / 255.0, t2 = B->pl[4][i] / 255.0, tr = t0 + t1 + t2;
                if (tr > 0.5) {
                    const double t3 = B->pl[5][i] / 255.0 - 0.5, t4 = B->pl[6][i] / 255.0 - 0.5, t5 = B->pl[7][i] / 255.0 - 0.5;
                    double c2 = (g[0] * g[0] * t0 + g[1] * g[1] * t1 + g[2] * g[2] * t2 + 2 * (g[0] * g[1] * t3 + g[0] * g[2] * t4 + g[1] * g[2] * t5)) / (gn * gn * tr);
                    c2 = c2 < 0 ? 0 : c2 > 1 ? 1 : c2;
                    const double ang = acos(sqrt(c2)) * 180 / M_PI; A->a[A_TN]++; if (ang > 20) A->a[A_T20]++;
                    A->th[(int)fmin(UQ_TH - 1, ang)]++; A->a[A_TG] += gn * sqrt(1 - c2) * 1000 / B->um;
                }
                if (sp[i]) { const double r = gn * sp[i] * h / 8; A->a[A_DRN]++; if (r < 0.7 || r > 1.4) A->a[A_DRO]++; }
            }
            if (B->ct) { A->a[A_CTN]++; if (B->ct[i] <= 0) A->a[A_CT0]++; }
        }
    }
    /* 16^3 blocks: local phase coherence and CT ridge contrast */
    for (int bz = z0; bz + UQ_BL <= z1; bz += UQ_BL) for (int by = 0; by + UQ_BL <= H; by += UQ_BL) for (int bx = 0; bx + UQ_BL <= W; bx += UQ_BL) {
        double zr = 0, zi = 0, cs = 0, cr = 0, ci = 0, ca = 0, nc = 0;
        for (int z = bz; z < bz + UQ_BL; z++) for (int y = by; y < by + UQ_BL; y++) for (int x = bx; x < bx + UQ_BL; x++) {
            const size_t i = uq_ix(B, z, y, x); if (wm[i] <= 64) continue;
            const double c = pc[i] - 128.0, s = ps[i] - 128.0, cf = hypot(c, s) / 127, ph = 2 * M_PI * B->Wv[i];
            if (cf >= 0.3) { const double p = atan2(s, c) - ph; zr += cf * cos(p); zi += cf * sin(p); cs += cf; }
            if (B->ct && B->ct[i] > 0) { const double v = B->ct[i] - B->ctm[i]; cr += v * cos(ph); ci -= v * sin(ph); ca += fabs(v); nc++; }
        }
        if (cs > UQ_BL * UQ_BL * UQ_BL * 0.05) { A->a[A_LCN]++; if (hypot(zr, zi) / cs < 0.5) A->a[A_LCLO]++; }
        if (nc > UQ_BL * UQ_BL * UQ_BL * 0.2 && ca > 0) { A->a[A_CRN]++; if (hypot(cr, ci) / ca < 0.2) A->a[A_CRLO]++; }
    }
    /* phase-unwrap witness: lines along each axis from every 2nd cell (start z in this range), L = 16 / 48, every cell confident */
    for (int z = z0; z < z1; z += 2) for (int y = 0; y < H; y += 2) for (int x = 0; x < W; x += 2) for (int a = 0; a < 3; a++) {
        const int n_a = a == 0 ? D : a == 1 ? H : W, p_a = a == 0 ? z : a == 1 ? y : x; const size_t st = a == 0 ? (size_t)H * W : a == 1 ? (size_t)W : 1;
        if (p_a + 16 >= n_a) continue;
        size_t i = uq_ix(B, z, y, x); double uph = 0; int ok = 1;
        for (int s = 0; s < 48 && p_a + s + 1 < n_a; s++) {
            const size_t j = i + st;
            const double c0 = pc[i] - 128.0, s0 = ps[i] - 128.0, c1 = pc[j] - 128.0, s1 = ps[j] - 128.0;
            if (wm[i] <= 64 || wm[j] <= 64 || hypot(c0, s0) < 0.3 * 127 || hypot(c1, s1) < 0.3 * 127) { ok = 0; break; }
            uph += uq_wrap(atan2(s1, c1) - atan2(s0, c0), 2 * M_PI) / (2 * M_PI); i = j;
            if (s == 15 || s == 47) {
                const double e = fabs(uph - uq_dW(B, uq_ix(B, z, y, x), j));
                if (s == 15) { A->a[A_PW16N]++; if (e > 0.5) A->a[A_PW16B]++; } else { A->a[A_PW48N]++; if (e > 0.5) A->a[A_PW48B]++; }
            }
        }
        (void)ok;
    }
}

/* fibres: per run, W along the polyline every native voxel inside the box [lo, hi) (set-local native), unwrapped, 9-sample median,
   hysteresis; arc / excursions / persistent per family (0 H, 1 V, 2 other) */
typedef struct { double arc[3], exc[3], pers[3]; } uq_fib;
static int uq_cmpd(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static void uq_fib_stretch(const double *u, int n, double hyst, double um, int fam, uq_fib *F) {
    if (n < 9) return;
    double *mf = malloc(n * sizeof *mf), w[9];
    for (int i = 0; i < n; i++) { int a = i - 4 < 0 ? 0 : i - 4, b = i + 4 >= n ? n - 1 : i + 4, m = 0; for (int j = a; j <= b; j++) w[m++] = u[j]; qsort(w, m, sizeof *w, uq_cmpd); mf[i] = w[m / 2]; }
    double base = mf[0], net = 0, exc = 0;
    for (int i = 1; i < n; i++) { const double d = mf[i] - base; if (fabs(d) > hyst) { double k = round(d); if (k == 0) k = d > 0 ? 1 : -1; exc += fabs(k); net += k; base += k; } }
    F->arc[fam] += (n - 1) * um * 1e-6; F->exc[fam] += exc; F->pers[fam] += fabs(net);
    free(mf);
}
static void uq_fibres(const fib_set *Fs, const fit_set *S, double sc, const double lo[3], const double hi[3], double hyst, double um, uq_fib *F) {
    double *u = nullptr; int cap = 0;
    for (int r = 0; r < Fs->n; r++) {   /* samples every native voxel of arc length (carried across the polyline's segments) */
        const int fam = Fs->fam[r] == 'H' ? 0 : Fs->fam[r] == 'V' ? 1 : 2; int n = 0; double prevW = 0, acc = 0, next = 0, s0 = 0;
        for (int i = Fs->off[r]; i + 1 < Fs->off[r + 1]; i++) {
            const float *p = Fs->pts + 3 * (size_t)i, *q = p + 3; double a[3], b[3];
            for (int d = 0; d < 3; d++) { a[d] = p[d] / sc; b[d] = q[d] / sc; }
            const double len = sqrt((b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1]) + (b[2] - a[2]) * (b[2] - a[2]));
            for (; next <= s0 + len; next += 1) {
                const double t = len > 0 ? (next - s0) / len : 0, x[3] = {a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])};
                if (x[0] < lo[0] || x[0] >= hi[0] || x[1] < lo[1] || x[1] >= hi[1] || x[2] < lo[2] || x[2] >= hi[2]) {   /* outside: end the stretch */
                    uq_fib_stretch(u, n, hyst, um, fam, F); n = 0; continue; }
                const double Wv = fit_set_W(S, x[0] - S->global->origin[0], x[1] - S->global->origin[1], x[2] - S->global->origin[2]);
                if (n == cap) { cap = cap ? 2 * cap : 1024; u = realloc(u, cap * sizeof *u); }
                acc = n ? acc + uq_wrap(Wv - prevW, 1.0) : Wv; u[n++] = acc; prevW = Wv;
            }
            s0 += len;
        }
        uq_fib_stretch(u, n, hyst, um, fam, F);
    }
    free(u);
}

/* separable 9-cell running mean (CT minus local mean for the ridge contrast); edges: the mean over the cells inside */
static void uq_mean_axis(const float *src, float *dst, int D, int H, int W, int a) {
    const int L = a == 0 ? W : a == 1 ? H : D; const size_t st = a == 0 ? 1 : a == 1 ? (size_t)W : (size_t)H * W;
    const size_t nl = (size_t)D * H * W / L;
    for (size_t l = 0; l < nl; l++) {
        const size_t base = a == 0 ? l * W : a == 1 ? (l / W) * (size_t)H * W + l % W : l;
        double s = 0; int c = 0; for (int k = 0; k <= 4 && k < L; k++) { s += src[base + k * st]; c++; }
        for (int k = 0; k < L; k++) {
            dst[base + k * st] = (float)(s / c);
            if (k + 5 < L) { s += src[base + (k + 5) * st]; c++; }
            if (k - 4 >= 0) { s -= src[base + (k - 4) * st]; c--; }
        }
    }
}
static void uq_boxmean(const float *in, float *out, int D, int H, int W) {   /* 9^3 running mean (the cells inside at the faces) */
    float *t = malloc((size_t)D * H * W * sizeof *t);
    uq_mean_axis(in, t, D, H, W, 0); uq_mean_axis(t, out, D, H, W, 1); uq_mean_axis(out, t, D, H, W, 2);
    memcpy(out, t, (size_t)D * H * W * sizeof *out); free(t);
}

/* one box: the fields' cells [fz0, fz0 + nz) x [oy, oy + ny) x [ox, ox + nx) -> metrics m[M_NM] */
typedef struct { const char *fdir; const fit_set *S; z3 *ct; const fib_set *fib; double fib_sc, fib_um, hyst; int nt; } uq_ctx;
static int uq_measure(const uq_ctx *C, int fz0, int nz, int oy, int ny, int ox, int nx, double m[M_NM]) {
    for (int k = 0; k < M_NM; k++) m[k] = NAN;
    fit_fields_buf b; memset(&b, 0, sizeof b);
    fit_fields_crop(1, oy, ox, ny, nx, nullptr);
    const int rc = fit_fields_load(C->fdir, fz0, fz0 + nz, 1, &b);
    fit_fields_crop(0, 0, 0, 0, 0, nullptr);
    if (rc) return -1;
    uq_box B = {.D = b.shape[0], .H = b.shape[1], .W = b.shape[2], .h = b.h, .um = b.um, .cy = b.cy, .cx = b.cx, .oy = (double)oy * b.h, .ox = (double)ox * b.h};
    for (int k = 0; k < 14; k++) B.pl[k] = b.a[k];
    B.w4k = b.a[10] && b.a[13] && b.f.w4_d > 0 ? (int)lround(b.f.w4_d / b.h) : 0;
    const size_t n = (size_t)B.D * B.H * B.W;
    float *Wv = malloc(n * sizeof *Wv), *U = malloc(n * sizeof *U);
    const fit_model *g = C->S->global;
    const double o[3] = {(double)(b.origin[0] - g->origin[0]), (double)(b.origin[1] - g->origin[1]) + B.oy, (double)(b.origin[2] - g->origin[2]) + B.ox};
    const int64_t n3[3] = {B.D, B.H, B.W};
    fit_set_grid_cpu(C->S, o, n3, b.h, 0, nullptr, Wv); fit_set_grid_cpu(C->S, o, n3, b.h, 1, nullptr, U);
    for (size_t i = 0; i < n; i++) U[i] -= Wv[i];   /* T = dir theta / 2 pi */
    B.Wv = Wv; B.T = U;
    float *ct = nullptr, *ctm = nullptr;
    if (C->ct) {   /* the mean of the h^3 native voxels per cell */
        const int h = (int)b.h; const int64_t co[3] = {b.origin[0], b.origin[1] + (int64_t)oy * h, b.origin[2] + (int64_t)ox * h}, cn[3] = {(int64_t)B.D * h, (int64_t)B.H * h, (int64_t)B.W * h};
        uint8_t *raw = malloc((size_t)cn[0] * cn[1] * cn[2]); ct = calloc(n, sizeof *ct); ctm = malloc(n * sizeof *ctm);
        if (!raw || z3_read(C->ct, co, cn, raw, C->nt)) { fprintf(stderr, "unroll-qa: CT read failed: %s\n", z3_error()); free(raw); free(ct); free(ctm); ct = ctm = nullptr; }
        else {
            for (int64_t z = 0; z < cn[0]; z++) for (int64_t y = 0; y < cn[1]; y++) for (int64_t x = 0; x < cn[2]; x++)
                ct[uq_ix(&B, (int)(z / h), (int)(y / h), (int)(x / h))] += raw[(z * cn[1] + y) * cn[2] + x];
            for (size_t i = 0; i < n; i++) ct[i] /= (float)(h * h * h);
            free(raw); uq_boxmean(ct, ctm, B.D, B.H, B.W);
        }
    }
    B.ct = ct; B.ctm = ctm;
    const int nb = (B.D + UQ_BL - 1) / UQ_BL; fit_set_threads(C->nt);
    B.acc = calloc(64, sizeof *B.acc);
    fit_par(nb, uq_rows, &B);
    uq_acc A = {0}; for (int t = 0; t < 64; t++) { for (int k = 0; k < A_NACC; k++) A.a[k] += B.acc[t].a[k]; for (int k = 0; k < UQ_TH; k++) A.th[k] += B.acc[t].th[k]; }
    const double nm = A.a[A_MAT];
    m[M_MAT] = nm / n;
    if (nm > 0) {
        m[M_RAD] = A.a[A_RAD] / nm; m[M_CONFLO] = A.a[A_CONFLO] / nm;
        double s = 0, c = 0; for (int a = 0; a < 3; a++) { m[M_W4Z + a] = A.a[A_W4N0 + a] ? A.a[A_W4K0 + a] / A.a[A_W4N0 + a] : NAN; s += A.a[A_W4K0 + a]; c += A.a[A_W4N0 + a]; }
        m[M_W4] = c ? s / c : NAN; m[M_LOOP] = A.a[A_LPN] ? A.a[A_LPB] / A.a[A_LPN] : NAN;
        if (A.a[A_TN]) { double cs = 0; int k = 0; for (; k < UQ_TH; k++) { cs += A.th[k]; if (cs >= A.a[A_TN] / 2) break; } m[M_TILT] = k + 0.5; m[M_T20] = A.a[A_T20] / A.a[A_TN]; m[M_TG] = A.a[A_TG] / A.a[A_TN]; }
        m[M_DRO] = A.a[A_DRN] ? A.a[A_DRO] / A.a[A_DRN] : NAN; m[M_LCLO] = A.a[A_LCN] ? A.a[A_LCLO] / A.a[A_LCN] : NAN;
        m[M_PW16] = A.a[A_PW16N] ? A.a[A_PW16B] / A.a[A_PW16N] : NAN; m[M_PW48] = A.a[A_PW48N] ? A.a[A_PW48B] / A.a[A_PW48N] : NAN;
        if (ct) { m[M_CT0] = A.a[A_CTN] ? A.a[A_CT0] / A.a[A_CTN] : NAN; m[M_CRLO] = A.a[A_CRN] ? A.a[A_CRLO] / A.a[A_CRN] : NAN; }
    }
    if (C->fib) {
        const double lo[3] = {(double)b.origin[0], (double)b.origin[1] + B.oy, (double)b.origin[2] + B.ox}, hi[3] = {lo[0] + B.D * b.h, lo[1] + B.H * b.h, lo[2] + B.W * b.h};
        uq_fib F = {0}; uq_fibres(C->fib, C->S, C->fib_sc, lo, hi, C->hyst, b.um, &F);
        const double arc = F.arc[0] + F.arc[1] + F.arc[2];
        m[M_FARC] = arc;
        if (F.arc[0] > 0.005) { m[M_FPH] = F.pers[0] / F.arc[0]; m[M_FEH] = F.exc[0] / F.arc[0]; }
        if (F.arc[1] > 0.005) m[M_FPV] = F.pers[1] / F.arc[1];
        if (arc > 0.005) m[M_FP] = (F.pers[0] + F.pers[1] + F.pers[2]) / arc;
    }
    free(B.acc); free(Wv); free(U); free(ct); free(ctm); fit_fields_free(&b);
    return 0;
}

/* ---- gates (DESIGN 2026-11-14; thresholds from 92 clean GT boxes: Paris 4 PH1-3 / HOLD / 34 July J boxes on ref-cf, 54 clean 1667 C boxes on
   ref-cf and on joined): conf_lo > 0.40 OR pw48 > 0.10 flags PH1-3, 30 / 33 Paris 4 boxes with ids < 0.5 and 8 / 9 1667 joined boxes
   with ids < 0.7, against 1 / 79 good 1667 box fits (ids >= 0.8). Tiles: a gate warns when more than --max-bad-share (0.35) of the tiles
   are over it (whole scrolls: 1667 joined 0.11 / 0.29 / 0.10 of 600 tiles for conf_lo / pw48 / w4, Paris 4 ref-cf 0.74 / 0.42 / 0.57). */
typedef struct { int m, sign; double thr; const char *opt, *what; } uq_gate;
static uq_gate uq_gates[] = {
    {M_CONFLO, +1, 0.40, "--max-conf-lo", "the network's phase is unsure (confidence < 0.3) on much of the material: the evidence cannot fix sheet identity"},
    {M_PW48, +1, 0.10, "--max-pw48", "the net phase unwrapped over 48 cells counts a different number of sheets than W (integer jumps of the fit)"},
    {M_W4, -1, 0.80, "--min-w4", "W disagrees with the network's winding-mod-4 head (the sheet count over 32 voxels)"},
    {M_FPH, +1, 130, "--max-fib-pers", "fibres change sheet in W (H family persistent changes per m; a fibre lies in one sheet)"},
};
enum { UQ_NG = sizeof uq_gates / sizeof uq_gates[0] };

int cmd_unroll_qa(int argc, char **argv) {
    if (argc < 3 || !uq_opt(argc, argv, "--fields", nullptr) || !uq_opt(argc, argv, "--model", nullptr)) {
        fprintf(stderr, "usage: ufsm unroll-qa --fields DIR --model SET [--ct ROOT KEY] [--fibres F.fib [--fib-um 2.4] [--hyst 0.75]]\n"
                        "         [--box Z,Y,X,NZ,NY,NX [--name N]]... | [--tile 128 --tiles 48 --min-mat 0.2]\n"
                        "         [--out DIR] [--threads 8] [--strict] [--max-conf-lo 0.40] [--max-pw48 0.10] [--min-w4 0.80] [--max-fib-pers 130] [--max-bad-share 0.35]\n"
                        "         [--gate-boxes FILE [--gate-scale 1]]   (per-box verdicts as a phase-coh --box-scale file: s 0 over a gate, 1 passing)\n"
                        "  label-free sheet-identity QA of a fitted winding (src/unrollqa.c): per box / tile the fibre sheet changes, the net's\n"
                        "  winding-mod-4 agreement, normal tilt, density ratio, phase confidence / coherence, phase-unwrap witness, CT; warn lines,\n"
                        "  exit 3 under --strict when a gate fails (a box over its threshold; tiles: more than --max-bad-share of them; a threshold 0 turns a gate off)\n"); return 2;
    }
    const double t0 = uq_now();
    const char *fdir = uq_opt(argc, argv, "--fields", nullptr), *md = uq_opt(argc, argv, "--model", nullptr), *out = uq_opt(argc, argv, "--out", nullptr);
    const int nt = ufsm_threads(opt_atoi("--threads", uq_opt(argc, argv, "--threads", "0")), 8), strict = uq_flag(argc, argv, "--strict");
    for (int k = 0; k < UQ_NG; k++) { const char *v = uq_opt(argc, argv, uq_gates[k].opt, nullptr); if (v) uq_gates[k].thr = atof(v); }   /* 0: that gate off */
    const double max_bad = opt_atof("--max-bad-share", uq_opt(argc, argv, "--max-bad-share", "0.35"));
    fit_set *S = fit_set_load(md); if (!S) { fprintf(stderr, "unroll-qa: cannot load the fit set %s\n", md); return 1; }
    uq_ctx C = {.fdir = fdir, .S = S, .nt = nt, .hyst = opt_atof("--hyst", uq_opt(argc, argv, "--hyst", "0.75"))};
    for (int i = 2; i + 2 < argc; i++) if (!strcmp(argv[i], opt_n("--ct"))) {
        store *st = store_open(argv[i + 1]); C.ct = st ? z3_open(st, argv[i + 2], nullptr) : nullptr;
        if (!C.ct) { fprintf(stderr, "unroll-qa: cannot open the CT %s %s: %s\n", argv[i + 1], argv[i + 2], z3_error()); return 1; }
    }
    fib_set fib = {0};
    if (uq_opt(argc, argv, "--fibres", nullptr)) {
        if (fib_load(uq_opt(argc, argv, "--fibres", ""), &fib)) { fprintf(stderr, "unroll-qa: cannot read the fibres %s\n", uq_opt(argc, argv, "--fibres", "")); return 1; }
        const double fum = opt_atof("--fib-um", uq_opt(argc, argv, "--fib-um", "2.4")), sum = S->global->um > 0 ? S->global->um : 2.4;
        C.fib = &fib; C.fib_sc = sum / fum;   /* level-0 point -> the set's native voxels */
    }
    long long fo[3]; int fshape[3], h;
    { char p[1400]; snprintf(p, sizeof p, "%s/fields.json", fdir); FILE *f = fopen(p, "rb"); if (!f) { fprintf(stderr, "unroll-qa: cannot read %s\n", p); return 1; }
      fseek(f, 0, SEEK_END); const long len = ftell(f); fseek(f, 0, SEEK_SET); char *txt = malloc(len + 1); const size_t r = fread(txt, 1, len, f); fclose(f);
      json *j = json_parse(txt, r); free(txt); if (!j) { fprintf(stderr, "unroll-qa: bad %s\n", p); return 1; }
      for (int d = 0; d < 3; d++) { fo[d] = (long long)json_num(json_at(json_get(j, "origin"), d), 0); fshape[d] = (int)json_num(json_at(json_get(j, "shape"), d), 0); }
      h = (int)json_num(json_get(j, "h"), 8); json_free(j); }
    /* the boxes: --box (global native) or tiles over the fields box with material */
    int nbox = 0, cap = 16; int (*bx)[6] = malloc(cap * sizeof *bx); char (*bn)[32] = malloc(cap * sizeof *bn);
    for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n("--box"))) {
        long long v[6]; if (sscanf(argv[i + 1], "%lld,%lld,%lld,%lld,%lld,%lld", v, v + 1, v + 2, v + 3, v + 4, v + 5) != 6) { fprintf(stderr, "unroll-qa: bad --box %s\n", argv[i + 1]); return 2; }
        if (nbox == cap) { cap *= 2; bx = realloc(bx, cap * sizeof *bx); bn = realloc(bn, cap * sizeof *bn); }
        int *c = bx[nbox]; for (int d = 0; d < 3; d++) { c[d] = (int)((v[d] - fo[d]) / h); c[3 + d] = (int)(v[3 + d] / h); if (c[d] < 0) { c[3 + d] += c[d]; c[d] = 0; } if (c[d] + c[3 + d] > fshape[d]) c[3 + d] = fshape[d] - c[d]; }
        const char *nmv = (i + 3 < argc && !strcmp(argv[i + 2], opt_n("--name"))) ? argv[i + 3] : nullptr;
        if (nmv) snprintf(bn[nbox], 32, "%s", nmv); else snprintf(bn[nbox], 32, "box%d", nbox);
        if (c[3] > 0 && c[4] > 0 && c[5] > 0) nbox++;
    }
    const int tiled = nbox == 0;
    if (tiled) {   /* tiles: a regular grid; the ones with material, an even pick of --tiles over them (z-major order: every z band) */
        const int T = opt_atoi("--tile", uq_opt(argc, argv, "--tile", "128")), want = opt_atoi("--tiles", uq_opt(argc, argv, "--tiles", "48")); const double minm = opt_atof("--min-mat", uq_opt(argc, argv, "--min-mat", "0.2"));
        int sh[3]; uint8_t *wpl = fit_field_array(fdir, "w", 8, sh);
        if (!wpl) { fprintf(stderr, "unroll-qa: cannot read the material plane of %s\n", fdir); return 1; }
        const int P = T / 8; int ncand = 0, ccap = 64; int (*cand)[3] = malloc(ccap * sizeof *cand);
        for (int z = 0; z + T <= fshape[0]; z += T) for (int y = 0; y + T <= fshape[1]; y += T) for (int x = 0; x + T <= fshape[2]; x += T) {
            double s = 0; for (int a = 0; a < P; a++) for (int b2 = 0; b2 < P; b2++) for (int c = 0; c < P; c++) {
                const int zz = z / 8 + a, yy = y / 8 + b2, xx = x / 8 + c; if (zz < sh[0] && yy < sh[1] && xx < sh[2]) s += wpl[((size_t)zz * sh[1] + yy) * sh[2] + xx] > 64; }
            if (s / ((double)P * P * P) < minm) continue;
            if (ncand == ccap) { ccap *= 2; cand = realloc(cand, ccap * sizeof *cand); }
            cand[ncand][0] = z; cand[ncand][1] = y; cand[ncand][2] = x; ncand++;
        }
        free(wpl);
        const int pick = want < ncand ? want : ncand;
        for (int k = 0; k < pick; k++) {
            const int q = (int)((k + 0.5) * ncand / pick);
            if (nbox == cap) { cap *= 2; bx = realloc(bx, cap * sizeof *bx); bn = realloc(bn, cap * sizeof *bn); }
            for (int d = 0; d < 3; d++) { bx[nbox][d] = cand[q][d]; bx[nbox][3 + d] = T; }
            snprintf(bn[nbox], 32, "t%lld_%lld_%lld", fo[0] + (long long)cand[q][0] * h, fo[1] + (long long)cand[q][1] * h, fo[2] + (long long)cand[q][2] * h); nbox++;
        }
        printf("unroll-qa: %d of %d tiles of %d^3 cells with material >= %.2f\n", nbox, ncand, T, minm);
        free(cand);
    }
    FILE *ft = nullptr;
    if (out) { char p[1400]; snprintf(p, sizeof p, "%s/boxes.tsv", out); ft = fopen(p, "w");
               if (!ft) { fprintf(stderr, "unroll-qa: cannot write %s (does --out exist?)\n", p); return 1; }
               fprintf(ft, "name\tz\ty\tx\tnz\tny\tnx"); for (int k = 0; k < M_NM; k++) fprintf(ft, "\t%s", uq_mname[k]); fprintf(ft, "\n"); }
    double (*M)[M_NM] = malloc((size_t)(nbox ? nbox : 1) * sizeof *M); int nok = 0;
    for (int k = 0; k < nbox; k++) {
        const int *c = bx[k];
        if (uq_measure(&C, c[0], c[3], c[1], c[4], c[2], c[5], M[k])) { fprintf(stderr, "unroll-qa: %s: cannot load the fields\n", bn[k]); for (int j = 0; j < M_NM; j++) M[k][j] = NAN; continue; }
        nok++;
        printf("%-12s", bn[k]); for (int j = 0; j < M_NM; j++) if (M[k][j] == M[k][j]) printf(" %s %.4g", uq_mname[j], M[k][j]); printf("\n");
        if (ft) { fprintf(ft, "%s\t%lld\t%lld\t%lld\t%d\t%d\t%d", bn[k], fo[0] + (long long)c[0] * h, fo[1] + (long long)c[1] * h, fo[2] + (long long)c[2] * h, c[3] * h, c[4] * h, c[5] * h);
                  for (int j = 0; j < M_NM; j++) fprintf(ft, "\t%.5g", M[k][j]); fprintf(ft, "\n"); fflush(ft); }
    }
    if (ft) fclose(ft);
    /* gates (also --out/gates.tsv: gate, threshold, boxes measured, boxes over, verdict) */
    int fail = 0; FILE *fg = nullptr;
    if (out) { char p[1400]; snprintf(p, sizeof p, "%s/gates.tsv", out); fg = fopen(p, "w"); if (fg) fprintf(fg, "gate\tthreshold\tmeasured\tover\tverdict\n"); }
    for (int gi = 0; gi < UQ_NG; gi++) {
        const uq_gate *G = &uq_gates[gi]; if (!(G->thr > 0)) continue;
        int nm = 0, nbad = 0; char worst[256] = ""; size_t wl = 0;
        for (int k = 0; k < nbox; k++) { const double v = M[k][G->m]; if (!(v == v)) continue; nm++;
            if (G->sign > 0 ? v > G->thr : v < G->thr) { nbad++; if (wl < sizeof worst - 40) wl += snprintf(worst + wl, sizeof worst - wl, " %s %.3g", bn[k], v); } }
        if (!nm) { printf("unroll-qa: gate %s: not measured (%s)\n", uq_mname[G->m], G->m >= M_FARC ? "no --fibres or under 5 mm of fibre per box" : "no data"); if (fg) fprintf(fg, "%s\t%g\t0\t0\tunmeasured\n", uq_mname[G->m], G->thr); continue; }
        const int bad = tiled ? nbad > max_bad * nm : nbad > 0;
        printf("unroll-qa: %s gate %s %s %.4g: %d of %d %s over%s%s\n", bad ? "WARN" : "ok", uq_mname[G->m], G->sign > 0 ? "<=" : ">=", G->thr, nbad, nm, tiled ? "tiles" : "boxes", nbad ? ":" : "", worst);
        if (bad) printf("unroll-qa: WARN %s: %s%s\n", G->what, tiled ? "" : "a box fails", strict ? " (--strict: exit 3)" : " (warn only; --strict makes it fatal)");
        if (fg) fprintf(fg, "%s\t%g\t%d\t%d\t%s\n", uq_mname[G->m], G->thr, nm, nbad, bad ? "warn" : "ok");
        fail |= bad;
    }
    if (fg) fclose(fg);
    /* --gate-boxes FILE [--gate-scale S]: the per-box verdicts as a `phase-coh --box-scale` file for gated pseudo labels (DESIGN 2026-11-17
       "Fit-derived pseudo labels"): every box as "z y x nz ny nx s", global native voxels x S (S: label grid / fields grid, e.g. 2 for
       labels at half the fields' um), s 0 when any active gate is over (a gate unmeasured on the box, e.g. fibres, does not count) or
       the box could not be measured, else 1. Passing boxes are written too, so the build can run phase-coh with --box-default 0 and
       untiled material (tiles under --min-mat, outside the fields box) stays unlabelled. */
    if (uq_opt(argc, argv, "--gate-boxes", nullptr)) {
        const char *gp = uq_opt(argc, argv, "--gate-boxes", ""); const double gs = opt_atof("--gate-scale", uq_opt(argc, argv, "--gate-scale", "1"));
        FILE *fb = gs > 0 ? fopen(gp, "w") : nullptr;
        if (!fb) { fprintf(stderr, "unroll-qa: cannot write --gate-boxes %s (or --gate-scale %g <= 0)\n", gp, gs); free(M); free(bx); free(bn); if (C.fib) fib_free(&fib); if (C.ct) z3_close(C.ct); fit_set_free(S); return 1; }
        fprintf(fb, "# unroll-qa gate boxes: z y x nz ny nx s (s 0 = a gate over or unmeasured box); scale %g; gates", gs);
        for (int gi = 0; gi < UQ_NG; gi++) if (uq_gates[gi].thr > 0) fprintf(fb, " %s %s %g", uq_mname[uq_gates[gi].m], uq_gates[gi].sign > 0 ? "<=" : ">=", uq_gates[gi].thr);
        fprintf(fb, "\n");
        int npass = 0; double mp = 0, mt = 0;
        for (int k = 0; k < nbox; k++) {
            const int *c = bx[k]; int ok = 1, any = 0;
            for (int gi = 0; gi < UQ_NG; gi++) { const uq_gate *G = &uq_gates[gi]; if (!(G->thr > 0)) continue; const double v = M[k][G->m]; if (!(v == v)) continue;
                any = 1; if (G->sign > 0 ? v > G->thr : v < G->thr) ok = 0; }
            ok &= any; const double mat = M[k][M_MAT] == M[k][M_MAT] ? M[k][M_MAT] : 0; npass += ok; mt += mat; mp += ok * mat;
            long long g[6]; for (int d = 0; d < 3; d++) { g[d] = llround((double)(fo[d] + (long long)c[d] * h) * gs); g[3 + d] = llround((double)c[3 + d] * h * gs); }
            fprintf(fb, "%lld %lld %lld %lld %lld %lld %d\n", g[0], g[1], g[2], g[3], g[4], g[5], ok);
        }
        fclose(fb);
        printf("unroll-qa: gate boxes %s: %d of %d boxes pass (%.4f; material-weighted %.4f)\n", gp, npass, nbox, nbox ? (double)npass / nbox : 0, mt > 0 ? mp / mt : 0);
    }
    printf("unroll-qa: %d boxes measured in %.1f s\n", nok, uq_now() - t0);
    free(M); free(bx); free(bn); if (C.fib) fib_free(&fib); if (C.ct) z3_close(C.ct); fit_set_free(S);
    return fail && strict ? 3 : 0;
}
