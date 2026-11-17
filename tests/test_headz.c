/* unet_set_head_infer_z (predict --fit-zarr's per-chunk head, 2026-11-17): an inference build without whole-window logits, whose head runs
   per level-0 z range in unet_head_fwd_z32, against the whole-window fp32 logits of the same net, weights and input. MX-fp4 activations and
   fp16 compute as predict's embedded precision (prec 1, f16, act_mx4). Heads of 32 outputs (two passes of 16), 11 (one pass) and 144 with
   head channels set (the fields predict's u1 shape: nine passes). Pass: every range (one plane, a middle run, the last planes, the whole
   window) bit for bit; the per-z build reports on and its forward returns no logits. Usage: test_headz */
#include "nn.h"
#include "test_gpu.h"
#include "unet.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int run(int cout, int hc0, int hnc, int D) {
    unet_cfg cfg = {3, {16, 16, 32}, 2, cout, 8};
    unet *u = unet_create(&cfg); unet_init(u, 11);
    if (hnc) unet_set_head_channels(u, hc0, hnc);
    const int rows = hnc ? hnc : cout;
    const shape5 xs = {1, 2, D, 32, 40}; const size_t nx = shape_numel(xs), S = (size_t)D * 32 * 40;
    _Float16 *xh = malloc(nx * 2); uint64_t s = 9;
    for (size_t i = 0; i < nx; i++) { s = s * 6364136223846793005ull + 1442695040888963407ull; xh[i] = (_Float16)((double)(s >> 11) * 0x1p-53 * 4 - 2); }
    void *xd = nn_malloc(nx * 2); nn_h2d(xd, xh, nx * 2);
    float *ref = malloc((size_t)rows * S * 4), *got = malloc((size_t)rows * S * 4);
    const float *lg = unet_forward_x(u, xd, xs, 0, 1);
    if (!lg) { fprintf(stderr, "test_headz: the whole-window build returned no logits\n"); return 1; }
    nn_sync(); nn_d2h(ref, lg, (size_t)rows * S * 4);
    unet_set_head_infer_z(u, 1);
    const float *l2 = unet_forward_x(u, xd, xs, 0, 1);
    int fails = 0;
    if (!unet_head_infer_z_on(u)) { fprintf(stderr, "test_headz: cout %d: the per-z build is off\n", cout); fails++; }
    if (l2) { fprintf(stderr, "test_headz: cout %d: the per-z build returned whole-window logits\n", cout); fails++; }
    const int rg[5][2] = {{0, 1}, {3, 7}, {D - 2, D}, {1, D - 1}, {0, D}};
    float *yd = nn_malloc((size_t)rows * S * 4);
    for (int r = 0; r < 5 && !fails; r++) {
        const int z0 = rg[r][0], z1 = rg[r][1]; const size_t pl = (size_t)32 * 40, nv = (size_t)(z1 - z0) * pl;
        if (unet_head_fwd_z32(u, z0, z1, yd)) { fprintf(stderr, "test_headz: cout %d: unet_head_fwd_z32 [%d, %d) failed\n", cout, z0, z1); fails++; break; }
        nn_sync(); nn_d2h(got, yd, (size_t)rows * nv * 4);
        size_t bad = 0;
        for (int c = 0; c < rows; c++) bad += memcmp(got + (size_t)c * nv, ref + (size_t)c * S + (size_t)z0 * pl, nv * 4) != 0;
        if (bad) { fprintf(stderr, "test_headz: cout %d rows %d: planes [%d, %d): %zu of %d channels differ from the whole-window logits\n", cout, rows, z0, z1, bad, rows); fails++; }
    }
    const char *e = nn_check(); if (e) { fprintf(stderr, "test_headz: cuda: %s\n", e); fails++; }
    printf("test_headz: cout %d (head rows %d from %d), %d x 32 x 40 level 0: %s\n", cout, rows, hc0, D, fails ? "FAIL" : "bit-identical over 5 ranges");
    nn_free(xd); nn_free(yd); free(xh); free(ref); free(got); unet_free(u);
    return fails;
}
int main(void) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (nn_init(0)) return test_no_gpu("test_headz", "nn_init(0) failed");
    nn_set_prec(1); nn_set_f16(1); unet_set_act_mx4(1); unet_set_act_mx8(0);
    int fails = 0;
    fails += run(32, 0, 0, 24);
    fails += run(11, 0, 0, 16);
    fails += run(144, 0, 144, 16);
    printf("test_headz: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
