/* The production unwrap head over z-slabs (train.c head_slab_pass): the 1^3 head from 32 MX-fp4 channels with GroupNorm + SiLU in
   staging to 11 fp16 logit planes per slab (lp_conv1_fwd_mx_h16_r), and the fused head backward (weight / bias gradient and the
   MX-fp4 input gradient with SR, lp_head_bwd_mx_r) from an fp16 logit gradient, over every slab of a P^3 window.
   Usage: bench_head2 [P] [Z]   (default 512, 32 planes per slab; prints ms per window pass, the minimum over repetitions, and
   checksums of the logits and of the outputs to compare kernel variants bit for bit) */
#include "nn.h"
#include "nn_lp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t rs = 88172645463325252ull;
static double urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 11) * 0x1.0p-53; }
static uint64_t fnv(const uint8_t *p, size_t n, uint64_t h) { for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; } return h; }
int main(int argc, char **argv) {
    const int P = argc > 1 ? atoi(argv[1]) : 512, Z = argc > 2 ? atoi(argv[2]) : 32, Ci = 32, Co = 11, G = 8;
    if (nn_init(0)) return 1;
    nn_set_f16(1); nn_set_loss_grad_h16(1); nn_set_sr(1);
    const shape5 xs = {1, Ci, P, P, P}, ys = {1, Co, P, P, P};
    const size_t S = (size_t)P * P * P, hw = (size_t)P * P, ns = (size_t)Z * hw, xb = nn_mx4_bytes(xs);
    void *xq = nn_malloc(xb), *gx = nn_malloc(xb);
    nn_set_storage(xq, xb, 4); nn_set_storage(gx, xb, 4);
    {   /* x: N(0, 1) channels -> MX-fp4, a z-chunk at a time through a float staging buffer */
        const size_t ch = (size_t)P * P * 8; float *xf = nn_malloc(ch * Ci * 4), *h = malloc(ch * 4);
        void *tq = nn_malloc(nn_mx4_bytes((shape5){1, Ci, 8, P, P}));
        uint8_t *hq = malloc(nn_mx4_bytes((shape5){1, Ci, 8, P, P})), *all = malloc(xb);
        const size_t rows = ch, rb = 16;   /* (32-channel MX-fp4: 16 data bytes per voxel, the scales after all rows) */
        for (int z0 = 0; z0 < P; z0 += 8) {
            for (int c = 0; c < Ci; c++) { for (size_t i = 0; i < ch; i++) h[i] = (float)(sqrt(-2 * log(urand() + 1e-300)) * cos(6.283185307 * urand())); nn_h2d(xf + (size_t)c * ch, h, ch * 4); }
            lp_f32_to_mx4(xf, 1, Ci, ch, tq); nn_d2h(hq, tq, nn_mx4_bytes((shape5){1, Ci, 8, P, P}));
            memcpy(all + (size_t)z0 * hw * rb, hq, rows * rb); memcpy(all + S * rb + (size_t)z0 * hw, hq + rows * rb, rows);
        }
        nn_h2d(xq, all, xb); nn_free(xf); nn_free(tq); free(h); free(hq); free(all);
    }
    float gam[32], bet[32], mean[8], rstd[8], w[11 * 32], bb[11];
    for (int c = 0; c < Ci; c++) { gam[c] = 1.f + 0.1f * c / Ci; bet[c] = 0.05f * c / Ci; }
    for (int g = 0; g < G; g++) { mean[g] = 0.01f * g; rstd[g] = 1.f; }
    for (int i = 0; i < Co * Ci; i++) w[i] = (float)(urand() - 0.5) * 0.5f;
    for (int i = 0; i < Co; i++) bb[i] = -0.5f;
    float *dg = nn_malloc(sizeof gam), *db_ = nn_malloc(sizeof bet), *dm = nn_malloc(sizeof mean), *dr = nn_malloc(sizeof rstd);
    nn_h2d(dg, gam, sizeof gam); nn_h2d(db_, bet, sizeof bet); nn_h2d(dm, mean, sizeof mean); nn_h2d(dr, rstd, sizeof rstd);
    const gnp_t gp = gnp_mk(dg, db_, dm, dr, G);
    float *dw = nn_malloc(sizeof w), *dbias = nn_malloc(sizeof bb), *gw = nn_malloc(sizeof w), *gb = nn_malloc(sizeof bb);
    nn_h2d(dw, w, sizeof w); nn_h2d(dbias, bb, sizeof bb);
    void *lg = nn_malloc(ns * Co * 2), *gyh = nn_malloc(ns * Co * 2);
    {   /* a fixed fp16 logit gradient for every slab */
        uint16_t *h = malloc(ns * Co * 2);
        for (size_t i = 0; i < ns * Co; i++) { _Float16 v = (_Float16)((urand() - 0.5) * 0.02); memcpy(&h[i], &v, 2); }
        nn_h2d(gyh, h, ns * Co * 2); free(h);
    }
    double bf = 1e30, bbk = 1e30; uint64_t hl = 1469598103934665603ull;
    uint8_t *hlg = malloc(ns * Co * 2);
    for (int r = 0; r < 4; r++) {
        nn_sync(); double t0 = now();
        for (int z0 = 0; z0 < P; z0 += Z) {
            lp_conv1_fwd_mx_h16_r(xq, 4, xs, dw, dbias, Co, lg, gp, (size_t)z0 * hw, (size_t)Z * hw);
            if (r == 0) { nn_sync(); nn_d2h(hlg, lg, ns * Co * 2); hl = fnv(hlg, ns * Co * 2, hl); }
        }
        nn_sync(); double t1 = now();
        if (r) bf = fmin(bf, t1 - t0);
        nn_zero(gw, sizeof w); nn_zero(gb, sizeof bb);
        nn_sync(); t0 = now();
        for (int z0 = 0; z0 < P; z0 += Z)
            if (!lp_head_bwd_mx_r(xq, 4, xs, gyh, 2, ys, dw, gw, gb, gx, 4, gp, 0x1234u, (size_t)z0 * hw, (size_t)Z * hw)) { fprintf(stderr, "head backward not covered\n"); return 1; }
        nn_sync(); t1 = now();
        if (r) bbk = fmin(bbk, t1 - t0);
    }
    const char *e = nn_check(); if (e) { printf("cuda: %s\n", e); return 1; }
    uint8_t *q = malloc(xb); nn_d2h(q, gx, xb);
    float hgw[11 * 32], hgb[11]; nn_d2h(hgw, gw, sizeof hgw); nn_d2h(hgb, gb, sizeof hgb);
    double a1 = 0; for (int i = 0; i < Co * Ci; i++) a1 += fabs(hgw[i]);
    for (int i = 0; i < Co; i++) a1 += fabs(hgb[i]);
    printf("checksums: logits fnv %016llx | gx fnv %016llx | |gw| + |gb| %.9g\n", (unsigned long long)hl, (unsigned long long)fnv(q, xb, 1469598103934665603ull), a1);
    printf("P %d, %d planes per slab: head forward %.2f ms, fused head backward %.2f ms per window (%.3f / %.3f ms per slab)\n",
           P, Z, bf * 1e3, bbk * 1e3, bf * 1e3 * Z / P, bbk * 1e3 * Z / P);
    return 0;
}
