/* ufsm phase-near (src/phasenear.c) on synthetic sheets with a known offset (lead 11-27: the offset recovered within 0.5 voxel in both
   directions before its numbers count). GT: planes z = k S + t (tilted by t = 0.2 y: not grid-aligned), the GT phase rasterised as the
   pair rule would (0 on the voxels nearest a plane, the fraction between). Prediction: the same planes moved by +off voxels along z,
   (a) on the level-0 grid (h 1) and (b) as 4x coarser fields (h 4, a 9.6 um net on a 2.4 um grid: upsampled by phase_near_sample).
   A plane moved by off along z lies off / sqrt(1 + 0.2^2) away along its normal; both directions' p50 must be within 0.5 voxel of that.
   Also: offset 0 gives p50 <= 0.5 both ways, and a prediction with every other sheet missing gives a GT -> pred p50 of about S / 2 while
   pred -> GT stays small (the two directions separate missed from misplaced sheets). */
#include "../src/band.h"
#include "../src/fit.h"
#include "../src/phasenear.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { NZ = 96, NY = 64, NX = 64 };
static const double S = 12.0, TY = 0.2;
static double wpos(double z, double y, double off) { return (z - TY * y - off) / S; }   /* the winding (turns) at a point */
static void gt_phase(uint8_t *g) {
    for (int z = 0; z < NZ; z++) for (int y = 0; y < NY; y++) for (int x = 0; x < NX; x++) {
        const double w = wpos(z + 0.5, y + 0.5, 0), f = w - floor(w);
        /* the sheet voxels: within half a voxel (along z) of a plane */
        const int on = fabs(f) * S < 0.5 || fabs(1 - f) * S < 0.5;
        int q = on ? 0 : (int)lrint(f * PHASE_PERIOD); if (q >= PHASE_PERIOD) q -= PHASE_PERIOD; if (!on && q == 0) q = 1;
        g[((size_t)z * NY + y) * NX + x] = (uint8_t)q;
    }
}
/* fields of h native voxels per cell (cell i centred at h (i + 0.5)), origin 0, um 2.4 * h / hscale ... here the native grid is the
   level-0 grid itself (um = um0) and h is the cell size; skip: every other sheet dropped (the phase stretched over 2 S) */
static void mk_fields(fit_fields_buf *F, int h, double off, int skip) {
    memset(F, 0, sizeof *F);
    const int D = NZ / h, H = NY / h, W = NX / h; const size_t M = (size_t)D * H * W;
    uint8_t *pc = malloc(M), *ps = malloc(M);
    for (int i = 0; i < D; i++) for (int j = 0; j < H; j++) for (int k = 0; k < W; k++) {
        const double w = wpos(h * (i + 0.5), h * (j + 0.5), off) / (skip ? 2 : 1), a = 2 * M_PI * w;
        const size_t q = ((size_t)i * H + j) * W + k;
        pc[q] = (uint8_t)lrint(128 + 127 * cos(a)); ps[q] = (uint8_t)lrint(128 + 127 * sin(a));
    }
    F->a[0] = pc; F->a[1] = ps; F->f.pc = pc; F->f.ps = ps; F->f.D = D; F->f.H = H; F->f.W = W; F->f.h = h; F->h = h; F->um = 2.4;
    F->shape[0] = D; F->shape[1] = H; F->shape[2] = W;
}
static int run(const char *what, int h, double off, int skip, double want_pg, double want_gp, double tol) {
    const int64_t n[3] = {NZ, NY, NX}, o[3] = {0, 0, 0}; const size_t N = (size_t)NZ * NY * NX;
    uint8_t *g = malloc(N); float *ph = malloc(N * 4), *cf = malloc(N * 4); fit_fields_buf F; phase_near_res R;
    gt_phase(g); mk_fields(&F, h, off, skip); phase_near_sample(&F, o, n, 2.4, ph, cf);
    if (phase_near_measure(g, ph, cf, n, 8, 0.f, 2, &R)) { printf("FAIL %s: measure\n", what); return 1; }
    const int ok = fabs(R.pg50 - want_pg) <= tol && fabs(R.gp50 - want_gp) <= tol;
    printf("%s %s: h %d off %.2f%s: pred -> GT p50 %.2f (want %.2f), GT -> pred p50 %.2f (want %.2f), sheets GT %zu pred %zu\n", ok ? "ok  " : "FAIL", what, h, off,
           skip ? " (every other sheet missing)" : "", R.pg50, want_pg, R.gp50, want_gp, R.gt_sheet, R.pred_sheet);
    free(g); free(ph); free(cf); free(F.a[0]); free(F.a[1]);
    return !ok;
}
int main(void) {
    const double c = 1 / sqrt(1 + TY * TY);   /* a z shift's normal component */
    int fails = 0;
    fails += run("zero offset", 1, 0.0, 0, 0.0, 0.0, 0.5);
    fails += run("known offset", 1, 2.0, 0, 2.0 * c, 2.0 * c, 0.5);
    fails += run("known offset", 1, 3.5, 0, 3.5 * c, 3.5 * c, 0.5);
    fails += run("known offset, 4x coarser fields", 4, 2.0, 0, 2.0 * c, 2.0 * c, 0.5);
    fails += run("known offset, 4x coarser fields", 4, -3.0, 0, 3.0 * c, 3.0 * c, 0.5);
    /* missed sheets: half the GT sheets have no predicted sheet within ~S / 2 (their GT -> pred distance ~ S c), so the GT -> pred p50 sits
       between 0 and S c; pred -> GT stays near 0 (every predicted sheet is a GT one) */
    {
        const int64_t n[3] = {NZ, NY, NX}, o[3] = {0, 0, 0}; const size_t N = (size_t)NZ * NY * NX;
        uint8_t *g = malloc(N); float *ph = malloc(N * 4), *cf = malloc(N * 4); fit_fields_buf F; phase_near_res R;
        gt_phase(g); mk_fields(&F, 1, 0.0, 1); phase_near_sample(&F, o, n, 2.4, ph, cf);
        phase_near_measure(g, ph, cf, n, 8, 0.f, 2, &R);
        const int ok = R.pg50 <= 0.5 && R.gp90 >= 0.8 * S * c && R.gp_within[2] < 0.6;
        printf("%s missed sheets: pred -> GT p50 %.2f, GT -> pred p90 %.2f (>= %.1f), GT within 4 voxels %.3f (< 0.6)\n", ok ? "ok  " : "FAIL", R.pg50, R.gp90, 0.8 * S * c, R.gp_within[2]);
        fails += !ok; free(g); free(ph); free(cf); free(F.a[0]); free(F.a[1]);
    }
    /* --min-comp (phase_near_set_min_comp): a small spurious wrap (a 5^3 blob of phase 0.98 | 0.02 between two sheets) adds predicted sheet voxels
       far from any GT sheet (the share within 1 voxel drops); dropping components < 200 voxels removes exactly it (pred count back to the clean run's), the planes stay */
    {
        const int64_t n[3] = {NZ, NY, NX}, o[3] = {0, 0, 0}; const size_t N = (size_t)NZ * NY * NX;
        uint8_t *g = malloc(N); float *ph = malloc(N * 4), *cf = malloc(N * 4); fit_fields_buf F; phase_near_res R0, R1, R2;
        gt_phase(g); mk_fields(&F, 1, 0.0, 0); phase_near_sample(&F, o, n, 2.4, ph, cf);
        phase_near_measure(g, ph, cf, n, 8, 0.f, 2, &R0);
        for (int z = 46; z < 51; z++) for (int y = 30; y < 35; y++) for (int x = 30; x < 35; x++) ph[((size_t)z * NY + y) * NX + x] = x < 32 ? 0.98f : 0.02f;   /* (mid-gap, ~4 voxels from either plane: a wrap inside the blob) */
        phase_near_measure(g, ph, cf, n, 8, 0.f, 2, &R1);
        phase_near_set_min_comp(200); phase_near_measure(g, ph, cf, n, 8, 0.f, 2, &R2); phase_near_set_min_comp(0);
        const int ok = R1.pred_sheet > R0.pred_sheet && R2.pred_sheet == R0.pred_sheet && R1.pg_within[0] < R0.pg_within[0] && R2.pg_within[0] == R0.pg_within[0];
        printf("%s --min-comp: pred sheets clean %zu, + blob %zu, filtered %zu; pred -> GT within 1 voxel %.4f / %.4f / %.4f\n", ok ? "ok  " : "FAIL", R0.pred_sheet, R1.pred_sheet, R2.pred_sheet, R0.pg_within[0], R1.pg_within[0], R2.pg_within[0]);
        fails += !ok; free(g); free(ph); free(cf); free(F.a[0]); free(F.a[1]);
    }
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
