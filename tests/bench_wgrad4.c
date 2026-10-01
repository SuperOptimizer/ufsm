/* Weight-gradient kernels in isolation at the 96^3 B2 training shapes: fp8 (lp_bwd_w_f8) vs fp4 (lp_bwd_w_f4, plain /
   Hadamard), fp16 x and gy, SR on. Min over iterations (robust to a shared GPU). usage: bench_wgrad4 [iters] */
#include "nn.h"
#include "nn_lp.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
    const int it = argc > 1 ? atoi(argv[1]) : 40;
    nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0);
    struct { const char *nm; int ci, co, s; } L[] = {{"16->16 @96", 16, 16, 96}, {"32->32 @48", 32, 32, 48}, {"64->64 @24", 64, 64, 24}, {"80->80 @12", 80, 80, 12}, {"48->16 @96", 48, 16, 96}, {"96->32 @48", 96, 32, 48}};
    printf("%-12s %8s %8s %8s  ms (min of %d)\n", "conv", "fp8", "fp4", "fp4+Had", it);
    for (int l = 0; l < (int)(sizeof L / sizeof L[0]); l++) {
        shape5 xs = {2, L[l].ci, L[l].s, L[l].s, L[l].s}, ys = xs; ys.c = L[l].co;
        void *x = nn_malloc(shape_numel(xs) * 2), *g = nn_malloc(shape_numel(ys) * 2);
        nn_zero(x, shape_numel(xs) * 2); nn_zero(g, shape_numel(ys) * 2);
        float *gw = nn_malloc((size_t)L[l].co * L[l].ci * 27 * 4), *gb = nn_malloc(L[l].co * 4);
        gnp_t none = {0}; split_t sp = {0}; sp.sr = getenv("BW_SR") ? (unsigned)atoi(getenv("BW_SR")) : 12345u;
        double best[3] = {1e9, 1e9, 1e9};
        for (int v = 0; v < 3; v++) for (int i = 0; i < it + 3; i++) {
            nn_sync(); double t0 = now();
            if (v == 0) lp_bwd_w_f8(x, 2, xs, g, 2, ys, gw, gb, none, sp); else lp_bwd_w_f4(x, 2, xs, g, 2, ys, gw, gb, none, sp, v == 2);
            nn_sync(); double dt = (now() - t0) * 1e3;
            if (i >= 3 && dt < best[v]) best[v] = dt;
        }
        printf("%-12s %8.3f %8.3f %8.3f\n", L[l].nm, best[0], best[1], best[2]);
        nn_free(x); nn_free(g); nn_free(gw); nn_free(gb);
    }
    return 0;
}
