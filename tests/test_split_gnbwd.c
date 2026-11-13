/* --split z: the MX GroupNorm + SiLU backward (nn_gn_silu_bwd) of a window cut into two z halves (each with h halo planes, gy zero
   there, the per-channel sums added across the halves by the split reduce) against the whole window on one GPU: the owned planes'
   gx must be bit-identical (one gradient voxel per channel and half, so the statistics are exact sums on both runs). 2026-11-13 regression: the halves scaled the summed group sums by D_local / D_whole and the apply
   kernels divided by the local element count, equal in exact arithmetic but a last-bit difference in fp32 (10 / 16 planes); MX
   rounding flipped some of those gradients and the fp4 encoder amplified them to 0.29 relative at enc0.c1 in the trainer A/B. Now
   gn_group_sums_k divides by the whole window's count on both. Cases: C 32 / 64, x MX-fp4 / MX-fp8, gy / gx fp32 (every last bit
   visible) or MX-fp8 (the trainer's gradient storage); the GroupNorm parameter gradients (local shares, summed) within 1e-5.
   Both halves run on GPU 0 (split thread slots). */
#include "nn.h"
#include "nn_lp.h"
#include "split.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { DW = 16, HH = 2, HW = 6, G = 8 };
static int C, XB, GB;   /* channels, x MX bits, gradient bits (0 fp32, 8 MX-fp8) */
static float *hx, *hgy, *hmean, *hrstd, *hgam, *hbet;   /* whole-window host tensors [C][DW][HW][HW], per group / channel */
static float *res_gx[3], *res_gg[3], *res_gb[3];   /* [0] whole, [1] / [2] the halves (local planes) */

static void *mk(shape5 s, int bits) {
    const size_t b = bits ? nn_mx_bytes(s, bits) : shape_numel(s) * 4; void *p = nn_malloc(b); nn_zero(p, b);
    if (bits) nn_set_storage(p, b, bits);
    return p;
}
static void to_dev(void *dst, shape5 s, int bits, const float *h) {   /* host fp32 -> fp32 or MX storage */
    const size_t n = shape_numel(s);
    if (!bits) { nn_h2d(dst, h, n * 4); return; }
    float *f = nn_malloc(n * 4); nn_h2d(f, h, n * 4); nn_set_prec(bits == 4 ? 3 : 2); nn_f32_to_act(f, s, dst); nn_sync(); nn_free(f);
}
static void to_host(const void *src, shape5 s, int bits, float *h) {
    const size_t n = shape_numel(s);
    if (!bits) { nn_d2h(h, src, n * 4); return; }
    float *f = nn_malloc(n * 4);
    if (bits == 4) lp_mx4_to_f32(src, s.n, s.c, shape_spatial(s), f); else lp_mx8_to_f32(src, s.n, s.c, shape_spatial(s), f);
    nn_sync(); nn_d2h(h, f, n * 4); nn_free(f);
}
/* one run: z0 the first window plane of the local tensor, D its depth, [o0, o1) the owned local planes; slot 0 whole, 1 / 2 halves */
static void run_one(int slot, int z0, int D, int o0, int o1) {
    const shape5 s = {1, C, D, HW, HW}; const size_t pl = (size_t)HW * HW, n = shape_numel(s);
    float *lx = malloc(n * 4), *lg = calloc(n, 4);
    for (int c = 0; c < C; c++) for (int z = 0; z < D; z++) {
        memcpy(lx + ((size_t)c * D + z) * pl, hx + ((size_t)c * DW + z0 + z) * pl, pl * 4);
        if (z >= o0 && z < o1) memcpy(lg + ((size_t)c * D + z) * pl, hgy + ((size_t)c * DW + z0 + z) * pl, pl * 4);   /* (halo planes: no gradient) */
    }
    void *x = mk(s, XB), *gy = mk(s, GB), *gx = mk(s, GB);
    to_dev(x, s, XB, lx); to_dev(gy, s, GB, lg);
    float *mean = nn_malloc(G * 4), *rstd = nn_malloc(G * 4), *gam = nn_malloc(C * 4), *bet = nn_malloc(C * 4), *gg = nn_malloc(C * 4), *gb = nn_malloc(C * 4);
    float *scr = nn_malloc(nn_gn_scratch(s));
    nn_h2d(mean, hmean, G * 4); nn_h2d(rstd, hrstd, G * 4); nn_h2d(gam, hgam, C * 4); nn_h2d(bet, hbet, C * 4); nn_zero(gg, C * 4); nn_zero(gb, C * 4);
    nn_set_grad_bf16(0);   /* (fp32 gradients unless gy / gx are MX-registered) */
    nn_gn_silu_bwd(x, s, G, gam, bet, mean, rstd, gy, gx, gg, gb, scr); nn_sync();
    res_gx[slot] = malloc(n * 4); to_host(gx, s, GB, res_gx[slot]);
    res_gg[slot] = malloc(C * 4); res_gb[slot] = malloc(C * 4); nn_d2h(res_gg[slot], gg, C * 4); nn_d2h(res_gb[slot], gb, C * 4);
    nn_free(x); nn_free(gy); nn_free(gx); nn_free(mean); nn_free(rstd); nn_free(gam); nn_free(bet); nn_free(gg); nn_free(gb); nn_free(scr);
    free(lx); free(lg);
}
static void job(int side, void *a) {
    (void)a; const int D = DW / 2 + HH;
    nn_split_cfg(side ? HH : 0, side ? 0 : HH, D, DW);
    run_one(1 + side, side ? DW / 2 - HH : 0, D, side ? HH : 0, side ? D : DW / 2);
    nn_split_cfg(0, 0, 0, 0);
}
static uint64_t rs = 0x2545f4914f6cdd1dull;
static float ru(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 11) * (1.0 / 9007199254740992.0)) * 2.f - 1.f; }
static int one_case(split_ctx *ctx) {
    const size_t pl = (size_t)HW * HW, nw = (size_t)C * DW * pl;
    hx = malloc(nw * 4); hgy = malloc(nw * 4); hmean = malloc(G * 4); hrstd = malloc(G * 4); hgam = malloc(C * 4); hbet = malloc(C * 4);
    for (size_t i = 0; i < nw; i++) { hx[i] = 1.5f * ru() + 0.2f; hgy[i] = 0.f; }
    for (int c = 0; c < C; c++) for (int side = 0; side < 2; side++) {   /* one gradient voxel per channel and half: every partial sum is a single term, so the
                                                                            summed statistics are exact on both runs and gx (non-zero at every voxel through the
                                                                            group terms) shows the coefficients' bits (the halves' local float sums of many
                                                                            terms differ from the whole run's in the last bit, by design) */
        const int z = side * DW / 2 + (c * 3 + side) % (DW / 2), y = (c * 5) % HW, x = (c * 7 + side) % HW;
        hgy[(((size_t)c * DW + z) * HW + y) * HW + x] = 1e-2f * ru();
    }
    for (int g = 0; g < G; g++) { hmean[g] = 0.2f + 0.05f * ru(); hrstd[g] = 0.9f + 0.1f * ru(); }
    for (int c = 0; c < C; c++) { hgam[c] = 1.f + 0.3f * ru(); hbet[c] = 0.1f * ru(); }
    run_one(0, 0, DW, 0, DW);
    split_run(ctx, job, nullptr);
    const int D = DW / 2 + HH; size_t ndiff = 0, nown = 0; double gmax = 0, gref = 0;
    for (int c = 0; c < C; c++) for (int z = 0; z < DW; z++) {
        const int side = z >= DW / 2, lz = side ? z - (DW / 2 - HH) : z;
        const float *w = res_gx[0] + ((size_t)c * DW + z) * pl, *h = res_gx[1 + side] + ((size_t)c * D + lz) * pl;
        for (size_t k = 0; k < pl; k++, nown++) ndiff += memcmp(&w[k], &h[k], 4) != 0;
    }
    for (int c = 0; c < C; c++) {
        gmax = fmax(gmax, fabs((double)res_gg[1][c] + res_gg[2][c] - res_gg[0][c])); gmax = fmax(gmax, fabs((double)res_gb[1][c] + res_gb[2][c] - res_gb[0][c]));
        gref = fmax(gref, fmax(fabs(res_gg[0][c]), fabs(res_gb[0][c])));
    }
    const int ok = ndiff == 0 && gmax <= 1e-5 * gref + 1e-12;
    printf("  C %2d x MX-fp%d gy/gx %-6s: owned gx %zu / %zu bits differ, gamma / beta grads %.2e of %.2e  %s\n", C, XB, GB ? "MX-fp8" : "fp32", ndiff, nown, gmax, gref, ok ? "ok" : "FAIL");
    for (int k = 0; k < 3; k++) { free(res_gx[k]); free(res_gg[k]); free(res_gb[k]); }
    free(hx); free(hgy); free(hmean); free(hrstd); free(hgam); free(hbet);
    return !ok;
}
int main(void) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (nn_init(0)) { printf("test_split_gnbwd: no GPU, skipped\n"); return 0; }
    nn_set_f16(1); nn_set_act_bf16(1); nn_set_sr(0); nn_set_prec_policy("");
    split_ctx *ctx = split_create(0, 0);
    printf("test_split_gnbwd: window %d planes, halves %d + %d halo planes, G %d\n", DW, DW / 2, HH, G);
    int fails = 0;
    static const int cs[2] = {32, 64}, xb[2] = {4, 8}, gb[2] = {0, 8};
    for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) for (int g = 0; g < 2; g++) { C = cs[a]; XB = xb[b]; GB = gb[g]; fails += one_case(ctx); }
    split_free(ctx);
    printf("test_split_gnbwd: %s\n", fails ? "FAIL" : "ok");
    return fails != 0;
}
