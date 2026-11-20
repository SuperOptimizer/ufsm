/* the learned sub-cell decoder's cost per training window (train --lat-dec H): a P^3 window = (P/4)^3 cells, head slabs of Z voxel planes.
   The streamed head runs the decoder forward on Z/4 + 2 cell planes per slab in both passes (statistics and gradient) and the backward on
   Z/4 planes per slab; the per-window time is 2 fwd(Z/4 + 2) + bwd(Z/4) times P / Z slabs (cells computed, not the cells kernels' latent
   inputs only). usage: bench_latdec [P 768] [Z 16] [H 64] [Cl 29] [NG 2] */
#include "nn.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
    const int P = argc > 1 ? atoi(argv[1]) : 768, Z = argc > 2 ? atoi(argv[2]) : 16, Hd = argc > 3 ? atoi(argv[3]) : 64, Cl = argc > 4 ? atoi(argv[4]) : 29, NG = argc > 5 ? atoi(argv[5]) : 2;
    if (nn_init(0)) return 1;
    const int c = P / 4, qf = Z / 4 + 2, qb = Z / 4, no = 64 * NG; const size_t pl = (size_t)c * c, nf = qf * pl, nb = qb * pl, np = (size_t)Cl * 27 * Hd + (size_t)no * Hd;
    uint64_t rs = 88172645463325252ull;
#define RND() (rs ^= rs << 13, rs ^= rs >> 7, rs ^= rs << 17, (float)((rs >> 40) / 16777216.0 - 0.5))
    const int D = 4 * (qf + 2); const size_t nx = (size_t)D * 16 * pl;   /* (the CT planes of the cells kernel: Z/4 + 4 cell planes) */
    float *hf = malloc((nx > np ? nx : np) * 4);
    for (size_t i = 0; i < np; i++) hf[i] = 0.05f * RND();
    float *p = nn_malloc(np * 4); nn_h2d(p, hf, np * 4);
    float *bas = nn_malloc(64 * 64 * 4), *gain = nn_malloc(64 * 4); for (int i = 0; i < 64 * 64; i++) hf[i] = 0.1f * RND(); nn_h2d(bas, hf, 64 * 64 * 4); for (int i = 0; i < 64; i++) hf[i] = 1.f; nn_h2d(gain, hf, 64 * 4);
    void *x = nn_malloc(nx * 2); { float *d = nn_malloc(nx * 4); for (size_t i = 0; i < nx; i++) hf[i] = RND(); nn_h2d(d, hf, nx * 4); nn_f32_to_h16(d, nx, x, 1.f); nn_free(d); }
    void *L = nn_malloc((size_t)Cl * (qf + 2) * pl * 2);
    void *rows = nn_malloc((size_t)(no + Hd) * nf * 2), *pre = nn_malloc((size_t)Hd * nf * 2); nn_zero(rows, (size_t)(no + Hd) * nf * 2); nn_zero(pre, (size_t)Hd * nf * 2);
    float *gp = nn_malloc(np * 4); nn_zero(gp, np * 4);
    void *scr = nn_malloc(nn_latdec_scratch(nb, Hd, Cl, NG));
    double tc = 1e9, tf = 1e9, tb = 1e9;
    for (int r = 0; r < 5; r++) {
        double t = now(); nn_latdec_cells(x, D, 4 * c, 4 * c, 1, qf + 2, Cl == 64 ? nullptr : bas, gain, Cl, Cl == 64 ? 0.f : 0.25f, L); nn_sync(); if (now() - t < tc) tc = now() - t;
        t = now(); if (nn_latdec_fwd(rows, 0, nf, pre, 0, nb, L, qb, c, c, p, Hd, Cl, NG)) { fprintf(stderr, "fwd refused\n"); return 1; } nn_sync(); if (now() - t < tf) tf = now() - t;
        t = now(); if (nn_latdec_bwd(rows, nb, (char *)rows + (size_t)no * nb * 2, pre, nb, L, qb, c, c, p, Hd, Cl, NG, gp, scr)) { fprintf(stderr, "bwd refused\n"); return 1; } nn_sync(); if (now() - t < tb) tb = now() - t;
    }
    const int ns = P / Z;
    const double win = ns * (3 * tc + 2 * tf + tb);
    printf("bench_latdec P %d Z %d H %d Cl %d ng %d: per slab cells %.3f ms, fwd (%d own planes) %.3f ms, bwd (%d planes) %.3f ms; per window (%d slabs, 3 cells + 2 fwd + 1 bwd) %.1f ms; "
           "scratch %.1f MB, L %.1f MB\n", P, Z, Hd, Cl, NG, tc * 1e3, qb, tf * 1e3, qb, tb * 1e3, ns, win * 1e3, nn_latdec_scratch(nb, Hd, Cl, NG) / 1e6, (double)Cl * (qf + 2) * pl * 4 / 1e6);
    return 0;
}
