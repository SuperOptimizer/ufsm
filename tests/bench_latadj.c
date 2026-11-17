/* bench_latadj: the head slab's adjoint gather (nn_lat_up_adj_c) at try9's P 768 shape (192^2 cells, 16-plane slabs, 14 + 2 channels),
   per kernel mode (UFSM_LAT_ADJ_TILE 0 / 1 / 2); min of 7 (trainperf-20261117) */
#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
    const int d = 192, h = 192, w = 192, nz = 16, z0 = 64, C = argc > 1 ? atoi(argv[1]) : 14;
    nn_init(0);
    const size_t ns = (size_t)C * nz * 16 * h * w;
    void *g = nn_malloc(ns * 2); float *s = nn_malloc(ns * 4), *gc = nn_malloc((size_t)C * d * h * w * 4);
    nn_zero(g, ns * 2); nn_zero(gc, (size_t)C * d * h * w * 4);
    for (int m = 0; m <= 2; m++) {
        nn_lat_set_adj_tile(m); double best = 1e9;
        for (int r = 0; r < 7; r++) { nn_sync(); const double t0 = now(); nn_lat_up_adj_c(g, C, d, h, w, z0, nz, gc, s, 0, d); nn_sync(); const double t = now() - t0; if (t < best) best = t; }
        printf("mode %d (%s): %.2f ms, %.0f GB/s of fp16 gradient\n", m, m == 0 ? "per cell" : m == 1 ? "row strips" : "64-cell tiles", 1e3 * best, ns * 2 / best / 1e9);
    }
    const char *e = nn_check(); if (e) { fprintf(stderr, "CUDA error %s\n", e); return 1; }
    return 0;
}
