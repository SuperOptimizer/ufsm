/* the fp4 weight gradient at the scratch2 level-0 shapes (perf2 2026-11-13): MX-fp4 x (256 channels, GN+SiLU fused), MX-fp8 gy, SR on,
   one sample of S^3 cells. usage: bench_wgl0 [S 128] [CI 256] [CO 256] [iters 5] */
#include "nn.h"
#include "nn_lp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t rs = 88172645463325252ull;
static float *dev_vals(size_t n, float sc, float off) { float *h = malloc(n * 4); for (size_t i = 0; i < n; i++) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; h[i] = off + sc * (float)((rs >> 40) / 16777216.0 - 0.5); } float *d = nn_malloc(n * 4); nn_h2d(d, h, n * 4); free(h); return d; }
int main(int argc, char **argv) {
    const int S = argc > 1 ? atoi(argv[1]) : 128, CI = argc > 2 ? atoi(argv[2]) : 256, CO = argc > 3 ? atoi(argv[3]) : 256, it = argc > 4 ? atoi(argv[4]) : 5, G = 32;
    if (nn_init(0)) return 1;
    shape5 xs = {1, CI, S, S, S}, ys = {1, CO, S, S, S};
    void *x = nn_malloc(nn_mx4_bytes(xs)); nn_set_storage(x, nn_mx4_bytes(xs), 4);
    { float *f = dev_vals(shape_numel(xs), 3.f, 0.f); lp_f32_to_mx4(f, 1, CI, shape_spatial(xs), x); nn_free(f); }
    const int G4 = getenv("GY8") ? 0 : 1;   /* gy MX-fp4 (the fp4-everywhere training) or MX-fp8 (GY8=1) */
    const size_t gyb = G4 ? nn_mx4_bytes(ys) : nn_mx8_bytes(ys);
    void *gy = nn_malloc(gyb); nn_set_storage(gy, gyb, G4 ? 4 : 8);
    { float *f = dev_vals(shape_numel(ys), 1e-3f, 0.f); if (G4) lp_f32_to_mx4(f, 1, CO, shape_spatial(ys), gy); else lp_f32_to_mx8(f, 1, CO, shape_spatial(ys), gy); nn_free(f); }
    float *gam = dev_vals(CI, 0.5f, 1.f), *bet = dev_vals(CI, 0.4f, 0.f), *mean = dev_vals(G, 0.4f, 0.f), *rstd = dev_vals(G, 0.4f, 0.9f);
    const gnp_t gp = gnp_mk(gam, bet, mean, rstd, G);
    float *gw = nn_malloc((size_t)CO * CI * 27 * 4), *gb = nn_malloc((size_t)CO * 4);
    split_t sp; memset(&sp, 0, sizeof sp); sp.sr = 12345u;
    double best = 1e9;
    for (int r = 0; r < it + 1; r++) { nn_zero(gw, (size_t)CO * CI * 27 * 4); nn_zero(gb, CO * 4); nn_sync(); const double t = now();
        lp_bwd_w_f4(x, 4, xs, gy, G4 ? 4 : 3, ys, gw, gb, gp, sp, 0); nn_sync(); if (r && now() - t < best) best = now() - t; }
    const double fl = 2.0 * shape_spatial(xs) * 27.0 * CI * CO;
    float *h = malloc((size_t)CO * CI * 27 * 4); nn_d2h(h, gw, (size_t)CO * CI * 27 * 4); double s2 = 0; for (size_t i = 0; i < (size_t)CO * CI * 27; i++) s2 += (double)h[i] * h[i];
    printf("wgrad fp4 (gy %s) %d^3 cells, %d -> %d: %.2f ms (%.1f TFLOP/s), |gw| %.6g\n", G4 ? "fp4" : "fp8", S, CI, CO, best * 1e3, fl / best / 1e12, s2);
    const char *e = nn_check(); if (e) { printf("CUDA: %s\n", e); return 1; }
    return 0;
}
