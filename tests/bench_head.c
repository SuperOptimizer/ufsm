/* Level-0 head and loss kernels of the band_affinity task at a training-window size: the 1^3 head (16 MX-fp4 channels with
   GroupNorm + SiLU in staging -> 7 fp32 logits), its weight / bias gradient from the 16-bit logit gradient, its backward-data
   into the MX-fp8 gradient of dec0, and the affinity loss (6 offsets) on band labels.
   Usage: bench_head [P]   (default 384; prints ms per call, the minimum over repetitions) */
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
typedef struct { void *xq, *gyh, *gout; float *scr, *w, *b, *y, *gw, *gb, *fin; uint8_t *band; shape5 xs, os; nn_gn_t gp; aff_offsets_t off; } B;
static void f_fwd(void *p) { B *a = p; if (nn_conv3d_fwd_x(a->xq, &a->gp, nullptr, nullptr, 0, 0, a->xs, a->w, a->b, a->os.c, 1, 1, a->y, 0, 0.f, nullptr, nullptr)) abort(); }
static void f_wgrad(void *p) { B *a = p; if (nn_conv3d_bwd_weight_x(a->xq, &a->gp, nullptr, nullptr, 0, 0, a->xs, a->gyh, a->os, 1, 1, a->gw, a->gb)) abort(); }
static void f_bdata(void *p) { B *a = p; nn_conv3d_bwd_data(a->gyh, a->os, a->w, a->xs, 1, 1, a->gout, a->scr); }
static void f_aff(void *p) { B *a = p; nn_aff_loss_async(a->y, a->os, 1, a->band, a->off, 0.5f, 1.f, a->gyh, a->fin); }
static uint8_t *g_t, *g_m, *g_w; static float *g_lscr;
static void f_recto(void *p) { B *a = p; shape5 ts = a->os; ts.c = 1; nn_loss_async_tol(a->y, g_t, g_m, g_w, ts, 0.5f, a->gyh, g_lscr, nullptr); }   /* channel 0 of the logits */
static double best(void (*f)(void *), void *a) {
    f(a); nn_sync();
    double m = 1e30;
    for (int r = 0; r < 7; r++) { double t0 = now(); f(a); nn_sync(); double t = now() - t0; if (t < m) m = t; }
    return m * 1e3;
}
int main(int argc, char **argv) {
    const int P = argc > 1 ? atoi(argv[1]) : 384, Ci = 16, Co = 7, G = 8;
    if (nn_init(0)) return 1;
    nn_set_f16(1); nn_set_loss_grad_h16(1);
    B a = {0};
    a.xs = (shape5){1, Ci, P, P, P}; a.os = (shape5){1, Co, P, P, P};
    const size_t S = (size_t)P * P * P;
    /* x: N(0, 1) channels -> MX-fp4 (registered), via a float staging buffer one channel slab at a time */
    a.xq = nn_malloc(nn_mx4_bytes(a.xs)); nn_set_storage(a.xq, nn_mx4_bytes(a.xs), 4);
    {
        float *xf = nn_malloc(S * Ci * 4), *h = malloc(S * 4);
        for (int c = 0; c < Ci; c++) { for (size_t i = 0; i < S; i++) h[i] = (float)(sqrt(-2 * log(urand() + 1e-300)) * cos(6.283185307 * urand())); nn_h2d(xf + (size_t)c * S, h, S * 4); }
        lp_f32_to_mx4(xf, 1, Ci, S, a.xq); nn_free(xf); free(h);
    }
    float gam[16], bet[16], mean[8], rstd[8], w[7 * 16], bb[7];
    for (int c = 0; c < Ci; c++) { gam[c] = 1.f + 0.1f * c / Ci; bet[c] = 0.05f * c / Ci; }
    for (int g = 0; g < G; g++) { mean[g] = 0.01f * g; rstd[g] = 1.f; }
    for (int i = 0; i < Co * Ci; i++) w[i] = (float)(urand() - 0.5) * 0.5f;
    for (int i = 0; i < Co; i++) bb[i] = -2.f;
    float *dg = nn_malloc(sizeof gam), *db_ = nn_malloc(sizeof bet), *dm = nn_malloc(sizeof mean), *dr = nn_malloc(sizeof rstd);
    nn_h2d(dg, gam, sizeof gam); nn_h2d(db_, bet, sizeof bet); nn_h2d(dm, mean, sizeof mean); nn_h2d(dr, rstd, sizeof rstd);
    a.gp = (nn_gn_t){dg, db_, dm, dr, G};
    a.w = nn_malloc(sizeof w); nn_h2d(a.w, w, sizeof w); a.b = nn_malloc(sizeof bb); nn_h2d(a.b, bb, sizeof bb);
    a.y = nn_malloc(S * Co * 4); a.gw = nn_malloc(sizeof w); a.gb = nn_malloc(sizeof bb);
    a.gyh = nn_malloc(S * Co * 2); nn_zero(a.gyh, S * Co * 2);
    a.gout = nn_malloc(nn_mx8_bytes(a.xs)); nn_set_storage(a.gout, nn_mx8_bytes(a.xs), 8);
    /* band: smooth winding in z (one band every ~20 planes) with an unknown slab */
    {
        uint8_t *h = malloc(S);
        for (size_t i = 0; i < S; i++) { int z = (int)(i / ((size_t)P * P)), x = (int)(i % P); h[i] = x < P / 8 ? 255 : (uint8_t)(1 + (z * 18 / 20 + x / 64) % 252); }
        a.band = nn_malloc(S); nn_h2d(a.band, h, S); free(h);
    }
    a.off = (aff_offsets_t){6, {0, 1, 2, 0, 1, 2}, {1, 1, 1, 8, 8, 8}};
    a.fin = nn_malloc(nn_aff_scratch(6)); a.scr = nn_malloc(1 << 16);
    {   /* recto loss inputs: soft targets, mask (air excluded), active (n, c) */
        uint8_t *h = malloc(S);
        for (size_t i = 0; i < S; i++) h[i] = (uint8_t)((i * 2654435761u >> 13) & 0xff);
        g_t = nn_malloc(S); nn_h2d(g_t, h, S);
        for (size_t i = 0; i < S; i++) h[i] = (i % 97) != 0;
        g_m = nn_malloc(S); nn_h2d(g_m, h, S); free(h);
        uint8_t one = 1; g_w = nn_malloc(64); nn_h2d(g_w, &one, 1);
        shape5 ts = a.os; ts.c = 1; g_lscr = nn_malloc(nn_loss_scratch(ts) + 64);
    }
    double t_fwd = best(f_fwd, &a), t_aff = best(f_aff, &a), t_w = best(f_wgrad, &a), t_d = best(f_bdata, &a), t_rl = best(f_recto, &a);
    const char *e = nn_check(); if (e) { printf("cuda: %s\n", e); return 1; }
    {   /* checksums of the outputs, to compare kernel variants */
        float *h = malloc(S * Co * 4); f_fwd(&a); nn_sync(); nn_d2h(h, a.y, S * Co * 4);
        double s1 = 0, s2 = 0; for (size_t i = 0; i < S * Co; i++) { s1 += h[i]; s2 += (double)h[i] * h[i]; }
        float gw[7 * 16], gb[7]; nn_zero(a.gw, sizeof gw); nn_zero(a.gb, sizeof gb); f_aff(&a); f_wgrad(&a); nn_sync(); nn_d2h(gw, a.gw, sizeof gw); nn_d2h(gb, a.gb, sizeof gb);
        double w1 = 0, b1 = 0; for (int i = 0; i < 7 * 16; i++) w1 += fabs(gw[i]); for (int i = 0; i < 7; i++) b1 += fabs(gb[i]);
        uint8_t *q = malloc(nn_mx8_bytes(a.xs)); f_bdata(&a); nn_sync(); nn_d2h(q, a.gout, nn_mx8_bytes(a.xs));
        uint64_t hq = 1469598103934665603ull; for (size_t i = 0; i < nn_mx8_bytes(a.xs); i++) { hq ^= q[i]; hq *= 1099511628211ull; }
        printf("checksums: logits sum %.9g sumsq %.9g | |gw| %.9g |gb| %.9g | gout fnv %016llx\n", s1, s2, w1, b1, (unsigned long long)hq);
        free(h); free(q);
    }
    printf("P %d (%.1f M voxels), head %d -> %d:\n  fwd (gn+silu, MX4 -> fp32)   %7.2f ms\n  weight + bias gradient        %7.2f ms\n"
           "  backward-data (-> MX8)       %7.2f ms\n  affinity loss + gradient     %7.2f ms\n  total                        %7.2f ms\n",
           P, S / 1e6, Ci, Co, t_fwd, t_w, t_d, t_aff, t_fwd + t_w + t_d + t_aff);
    printf("  recto loss + gradient        %7.2f ms\n", t_rl);
    return 0;
}
