/* bench_s2: one desk card's share of the scratch2 trunk (train --lat 4 --split z --split-top 3 at P 1024: a 544 x 1024^2-voxel slab = 136 x 256^2
   cells, widths 256,320,384,448,288,288, 4 attention blocks, the --fp4 2 preset: MX-fp4 activations / gradients / input, stored GN statistics,
   stochastic rounding, chunked, lean 2, shared encoder a1, recompute 1) on one GPU without the split collectives: training step time and the
   device bytes, for the memory / recompute / host-offload modes the unet env switches select (UFSM_XIN_OFFLOAD, UFSM_SKIP_OFFLOAD,
   UFSM_DOWN_OFFLOAD, UFSM_COARSE_GRAD_SHARE, UFSM_A0_SHARE, UFSM_A1_OFFLOAD, UFSM_RECOMPUTE, UFSM_LEAN, ...). Synthetic input (a smooth CT
   field), a fixed cell-logit gradient. No data, no sampler: the network's GPU time only (the trainer adds the head slabs, losses, data path).
   env: BS2_D (136 cells deep), BS2_HW (256 cells), BS2_IT (6 timed steps after 2 warm-up), BS2_COUT (160), BS2_CAP_GB (ballast: leave only this
   much device memory, e.g. 16.3 for a desk card), UFSM_PROF=layers for the per-layer table. */
#include "unet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int ienv(const char *n, int d) { const char *e = getenv(n); return e ? atoi(e) : d; }

int main(void) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (nn_init(0)) { printf("bench_s2: no GPU\n"); return 1; }
    const int D = ienv("BS2_D", 136), HW = ienv("BS2_HW", 256), IT = ienv("BS2_IT", 6), LC = 29, COUT = ienv("BS2_COUT", 160);
    void *ballast = nullptr;
    if (getenv("BS2_CAP_GB")) {   /* emulate a smaller card: hold everything above the cap */
        const size_t fr = nn_mem_free(), cap = (size_t)(atof(getenv("BS2_CAP_GB")) * 1e9), total = nn_mem_total(), used = total - fr;   /* (keep cap - used free) */
        if (total > cap && fr > cap - used) { const size_t b = fr - (cap - used); ballast = nn_malloc(b); printf("bench_s2: ballast %.2f GB (card %.2f GB, cap %.2f GB)\n", b / 1e9, total / 1e9, cap / 1e9); }
    }
    nn_set_f16(1); nn_set_grad_scale(1024.f); nn_set_loss_grad_h16(1); nn_set_tf32(1);
    setenv("UFSM_F4_WGRAD", "1", 0);
    unet_set_act_mx4(1); nn_set_sr(1); if (nn_set_prec_policy("all=fp4:fp4:fp4")) return 2;
    unet_set_chunk_up(2); unet_set_grad_mx8(1); unet_set_lean(ienv("UFSM_LEAN", 2)); unet_set_grad_mx4(1); unet_set_share_enc_a1(1); unet_set_recompute(ienv("UFSM_RECOMPUTE", 1));
    unet_set_input_mx(1); unet_set_input_prec(4); nn_set_gn_stored(1);
    unet_cfg cfg = {6, {256, 320, 384, 448, 288, 288}, LC + 3, COUT, 8, 1}; cfg.half_res = 3; cfg.attn_blocks = 4; cfg.attn_pe = 1;
    unet *u = unet_create(&cfg); unet_init(u, 7);
    static float bas[29 * 64], gain[29];
    for (int c = 0; c < LC; c++) { gain[c] = 1.f; double n = 0; for (int k = 0; k < 64; k++) { bas[c * 64 + k] = c ? (float)sin(0.37 * (c * 64 + k)) : 0.125f; n += (double)bas[c * 64 + k] * bas[c * 64 + k]; }
                                   for (int k = 0; k < 64; k++) bas[c * 64 + k] = (float)(bas[c * 64 + k] / sqrt(n)); }
    unet_set_lat_basis(u, bas, gain, LC, 0.f);
    const int Dv = 4 * D, Pv = 4 * HW;
    const size_t nv = (size_t)Dv * Pv * Pv, nc = (size_t)D * HW * HW;
    const size_t RB = NN_RAD_REC(Pv);
    uint8_t *rec = calloc(1, RB); { int32_t h[8] = {Pv / 3, -Pv / 5, Dv, Pv, 1, 0 | 1 << 2 | 2 << 4, 0, 0}; memcpy(rec, h, sizeof h); }
    void *drec = nn_malloc(RB); nn_h2d(drec, rec, RB); unet_set_input_rad(u, drec);
    _Float16 *h = malloc(nv * 2);
    for (size_t i = 0; i < nv; i++) { const size_t z = i / ((size_t)Pv * Pv), y = i / Pv % Pv, x = i % Pv; h[i] = (_Float16)(0.6 * sin(0.05 * z + 0.031 * y + 0.023 * x) + 0.3 * sin(0.4 * x + 0.2 * z)); }
    void *ct = nn_malloc(nv * 2); nn_h2d(ct, h, nv * 2); free(h);
    _Float16 *g = malloc((size_t)COUT * nc * 2);
    for (size_t i = 0; i < (size_t)COUT * nc; i++) g[i] = (_Float16)(1e-3 * sin(0.001 * i));
    void *gl = nn_malloc((size_t)COUT * nc * 2); nn_h2d(gl, g, (size_t)COUT * nc * 2); free(g);
    const shape5 xs = {1, cfg.cin, D, HW, HW};
    nn_sync(); const size_t f0 = nn_mem_free(); size_t fmin = f0;
    double t = 0;
    for (int k = 0; k < 2 + IT; k++) {
        if (k == 2) { nn_sync(); t = now(); if (getenv("UFSM_PROF")) unet_prof_report(); }
        unet_forward_x(u, ct, xs, 1, 1);
        unet_zero_grad(u); unet_backward_x(u, gl, 1);
        nn_sync(); const size_t f = nn_mem_free(); if (f < fmin) fmin = f;
        if (nn_check()) { printf("bench_s2: cuda error %s\n", nn_check()); return 1; }
    }
    const double st = (now() - t) / IT;
    printf("bench_s2: %d x %d^2 cells, widths 256..448,288,288 + 4 attention blocks, cout %d: %.3f s per training step (%d steps), device %.2f GB "
           "after build + step (net %.2f GB activations / gradients, host offloads %.2f GB each way)\n",
           D, HW, COUT, st, IT, (f0 - fmin) / 1e9, unet_activation_bytes(u) / 1e9, unet_last_offload_bytes() / 1e9);
    if (getenv("UFSM_PROF")) unet_prof_report();
    unet_free(u); nn_free(ct); nn_free(gl); nn_free(drec); if (ballast) nn_free(ballast);
    return 0;
}
