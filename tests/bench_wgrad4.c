/* Weight-gradient kernels in isolation at the 96^3 B2 training shapes: fp8 (lp_bwd_w_f8) vs fp4 (lp_bwd_w_f4, plain /
   Hadamard), fp16 x and gy, SR on. Min over iterations (robust to a shared GPU). usage: bench_wgrad4 [iters] */
#include "nn.h"
#include "nn_lp.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
static void fill_rand(float *d, size_t n, float sc) {
    float *h = malloc(n * 4); uint64_t r = 88172645463325252ull;
    for (size_t i = 0; i < n; i++) { r ^= r << 13; r ^= r >> 7; r ^= r << 17; h[i] = ((float)(r >> 40) * (1.f / 16777216.f) * 2.f - 1.f) * sc; }
    nn_h2d(d, h, n * 4); free(h);
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
    const int it = argc > 1 ? atoi(argv[1]) : 40;
    nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0);
    struct { const char *nm; int ci, co, s; } L[] = {{"16->16 @96", 16, 16, 96}, {"32->32 @48", 32, 32, 48}, {"64->64 @24", 64, 64, 24}, {"80->80 @12", 80, 80, 12}, {"48->16 @96", 48, 16, 96}, {"96->32 @48", 96, 32, 48}};
    printf("%-12s %8s %8s %8s  ms (min of %d)\n", "conv", "fp8", "fp4", "fp4+Had", it);
    for (int l = 0; l < (int)(sizeof L / sizeof L[0]); l++) {
        if (getenv("BW_LAYER") && l != atoi(getenv("BW_LAYER"))) continue;
        int P = getenv("BW_P") ? atoi(getenv("BW_P")) : 96;
        int S = L[l].s * P / 96, B = getenv("BW_B") ? atoi(getenv("BW_B")) : 2;
        shape5 xs = {B, L[l].ci, S, S, S}, ys = xs; ys.c = L[l].co;
        /* BW_MX=1: the training storage of --fp4: mx4 x, mx8 gy (random data, so the decode does real work) */
        const int mx = getenv("BW_MX") ? atoi(getenv("BW_MX")) : 0;
        const size_t bx = (mx & 1) ? lp_mx4_bytes(xs.n, xs.c, shape_spatial(xs)) : shape_numel(xs) * 2, bg = (mx & 2) ? lp_mx8_bytes(ys.n, ys.c, shape_spatial(ys)) : shape_numel(ys) * 2;
        void *x = nn_malloc(bx), *g = nn_malloc(bg);
        nn_zero(x, bx); nn_zero(g, bg);
        if (mx & 1) { float *t = nn_malloc(shape_numel(xs) * 4); fill_rand(t, shape_numel(xs), 1.f); lp_f32_to_mx4(t, xs.n, xs.c, shape_spatial(xs), x); nn_free(t); }
        if (mx & 2) { float *t = nn_malloc(shape_numel(ys) * 4); fill_rand(t, shape_numel(ys), 1e-3f); lp_f32_to_mx8(t, ys.n, ys.c, shape_spatial(ys), g); nn_free(t); }
        const int xt = (mx & 1) ? 4 : 2, gt = (mx & 2) ? 3 : 2;   /* BW_MX bit 0: mx4 x, bit 1: mx8 gy (3 = the --fp4 training storage) */
        float *gw = nn_malloc((size_t)L[l].co * L[l].ci * 27 * 4), *gb = nn_malloc(L[l].co * 4);
        gnp_t none = {0}; split_t sp = {0}; sp.sr = getenv("BW_SR") ? (unsigned)atoi(getenv("BW_SR")) : 12345u;
        double best[3] = {1e9, 1e9, 1e9};
        const int hadm = getenv("BW_HAD") ? atoi(getenv("BW_HAD")) : 1;   /* third column: 1 Hadamard, 3 Hadamard + x SR */
        for (int v = 0; v < 3; v++) for (int i = 0; i < it + 3; i++) {
            nn_sync(); double t0 = now();
            if (v == 0) lp_bwd_w_f8(x, xt, xs, g, gt, ys, gw, gb, none, sp); else lp_bwd_w_f4(x, xt, xs, g, gt, ys, gw, gb, none, sp, v == 2 ? hadm : 0);
            nn_sync(); double dt = (now() - t0) * 1e3;
            if (i >= 3 && dt < best[v]) best[v] = dt;
        }
        printf("%-12s %8.3f %8.3f %8.3f\n", L[l].nm, best[0], best[1], best[2]);
        nn_free(x); nn_free(g); nn_free(gw); nn_free(gb);
    }
    return 0;
}
