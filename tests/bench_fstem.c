/* the fine stem's forward / backward timing at a P 512 head slab (perf2 2026-11-13): D = H = W = P, a slab of NZ planes, C 16, NOUT outputs.
   usage: bench_fstem [P 512] [NZ 32] [NOUT 11] */
#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
    const int P = argc > 1 ? atoi(argv[1]) : 512, NZ = argc > 2 ? atoi(argv[2]) : 32, NO = argc > 3 ? atoi(argv[3]) : 11, C = 16;
    if (nn_init(0)) return 1;
    const size_t vol = (size_t)P * P * P, ns = (size_t)NZ * P * P, np = (size_t)C * 64 + C + (size_t)NO * C;
    uint64_t rs = 88172645463325252ull; float *hf = malloc((vol > ns * (NO + 4) ? vol : ns * (NO + 4)) * 4);
#define RND() (rs ^= rs << 13, rs ^= rs >> 7, rs ^= rs << 17, (float)((rs >> 40) / 16777216.0 - 0.5))
    void *x = nn_malloc(vol * 2); for (size_t i = 0; i < vol; i++) hf[i] = 2 * RND(); { float *d = nn_malloc(vol * 4); nn_h2d(d, hf, vol * 4); nn_f32_to_h16(d, vol, x, 1.f); nn_free(d); }
    float *p = nn_malloc(np * 4); for (size_t i = 0; i < np; i++) hf[i] = 0.3f * RND(); nn_h2d(p, hf, np * 4);
    void *lg = nn_malloc(ns * 2 * (NO + 4)); const size_t nl = ns * (NO + 4); for (size_t i = 0; i < nl; i++) hf[i] = RND(); { float *d = nn_malloc(nl * 4); nn_h2d(d, hf, nl * 4); nn_f32_to_h16(d, nl, lg, 1.f); nn_free(d); }
    uint16_t *hl = malloc(nl * 2);
    float *gp = nn_malloc(np * 4); nn_zero(gp, np * 4);
    int chs[16]; for (int o = 0; o < NO; o++) chs[o] = o;
    nn_fstem_fwd(x, P, P, P, 64, NZ, p, C, NO, chs, lg, ns); nn_sync(); nn_d2h(hl, lg, nl * 2);
    { uint64_t h = 1469598103934665603ull; for (size_t i = 0; i < nl; i++) h = (h ^ hl[i]) * 1099511628211ull; printf("fstem fwd output hash %016llx (one call on random inputs)\n", (unsigned long long)h); }
    nn_sync(); double best = 1e9;
    for (int r = 0; r < 8; r++) { const double t = now(); nn_fstem_fwd(x, P, P, P, 64, NZ, p, C, NO, chs, lg, ns); nn_sync(); if (now() - t < best) best = now() - t; }
    const double by = ns * 2.0 * NO * 2 + ns * 2.0;
    printf("fstem fwd P %d slab %d planes, %d outputs: %.3f ms (%.0f GB/s of logits RMW + x)\n", P, NZ, NO, best * 1e3, by / best / 1e9);
    best = 1e9;
    for (int r = 0; r < 8; r++) { const double t = now(); nn_fstem_bwd(x, P, P, P, 64, NZ, p, C, NO, chs, lg, ns, gp); nn_sync(); if (now() - t < best) best = now() - t; }
    printf("fstem bwd: %.3f ms\n", best * 1e3);
    const char *e = nn_check(); if (e) { printf("CUDA: %s\n", e); return 1; }
    return 0;
}
