/* The level-0 fp4 3^3 convolutions of training in their storage formats, timed in isolation (min over repetitions) with an
   FNV hash of the output bytes (equal hashes = identical results across builds / variants):
     fwd   MX-fp4 in -> MX-fp4 out, GN+SiLU of the input in staging, GN statistics of the output (enc0.c2 / dec0.c2, dec0.c1)
     bdata MX-fp8 gy -> MX-fp8 gx with stochastic rounding of gy (backward-data), 16 -> 16 and 16 -> 32 + 16 (dec0.c1 split)
   usage: bench_f4conv [P] */
#include "nn.h"
#include "nn_lp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static float *rnd(size_t n, float sc, uint64_t r) {
    float *h = malloc(n * 4);
    for (size_t i = 0; i < n; i++) { r ^= r << 13; r ^= r >> 7; r ^= r << 17; h[i] = ((float)(r >> 40) / 16777216.f - 0.5f) * sc; }
    float *d = nn_malloc(n * 4); nn_h2d(d, h, n * 4); free(h); return d;
}
static void *mx(int bits, int C, size_t S, float sc, uint64_t seed) {
    float *f = rnd(S * C, sc, seed);
    void *q = nn_malloc(bits == 4 ? lp_mx4_bytes(1, C, S) : lp_mx8_bytes(1, C, S));
    if (bits == 4) lp_f32_to_mx4(f, 1, C, S, q); else lp_f32_to_mx8(f, 1, C, S, q);
    nn_free(f); return q;
}
static uint64_t fnv(const void *d, size_t n) {
    uint8_t *h = malloc(n); nn_d2h(h, d, n);
    uint64_t x = 1469598103934665603ull; for (size_t i = 0; i < n; i++) { x ^= h[i]; x *= 1099511628211ull; }
    free(h); return x;
}
typedef struct { const void *x; void *y, *y2; int xb, ci, co, osplit, sr; shape5 xs; const float *w, *b; gnp_t gp; double *osum; int Go; } job;
static void run(job *j, unsigned it) {
    split_t sp = {0};
    if (j->y2) { sp.y2 = j->y2; sp.o_split = j->osplit; }
    if (j->sr) sp.sr = 0x2545f491u + it;
    if (j->osum) nn_zero(j->osum, 2 * 64 * sizeof(double));
    lp_conv_fwd_f4(j->x, j->xb, j->xs, j->w, j->b, j->co, j->y, j->xb, j->gp, j->osum, j->Go, sp);
}
int main(int argc, char **argv) {
    const int P = argc > 1 ? atoi(argv[1]) : 384, G = 8;
    nn_init(0); nn_set_f16(1);
    const size_t S = (size_t)P * P * P;
    float *gam = rnd(48, 1.f, 4), *bet = rnd(48, 0.5f, 5), *mean = rnd(G, 0.1f, 6), *rstd = rnd(G, 0.1f, 7);
    double *osum = nn_malloc(2 * 64 * sizeof(double));
    struct { const char *nm; int xb, ci, co, osplit, gn, stats, sr; } L[] = {
        {"fwd 16 -> 16 (gn+silu, stats)", 4, 16, 16, 0, 1, 1, 0}, {"fwd 48 -> 16 (gn+silu, stats)", 4, 48, 16, 0, 1, 1, 0},
        {"bdata 16 -> 16 (mx8, SR)", 3, 16, 16, 0, 0, 0, 1}, {"bdata 16 -> 32 + 16 (mx8, SR)", 3, 16, 48, 32, 0, 0, 1}};
    for (int l = 0; l < 4; l++) {
        shape5 xs = {1, L[l].ci, P, P, P};
        job j = {0};
        j.x = mx(L[l].xb == 4 ? 4 : 8, L[l].ci, S, L[l].xb == 4 ? 2.f : 1e-3f, 11 + l); j.xb = L[l].xb; j.ci = L[l].ci; j.co = L[l].co; j.xs = xs; j.sr = L[l].sr;
        const int c1 = L[l].osplit ? L[l].osplit : L[l].co, c2 = L[l].co - c1, bits = L[l].xb == 4 ? 4 : 8;
        const size_t by1 = bits == 4 ? lp_mx4_bytes(1, c1, S) : lp_mx8_bytes(1, c1, S), by2 = c2 ? (bits == 4 ? lp_mx4_bytes(1, c2, S) : lp_mx8_bytes(1, c2, S)) : 0;
        j.y = nn_malloc(by1); j.y2 = c2 ? nn_malloc(by2) : nullptr; j.osplit = L[l].osplit;
        float *w = rnd((size_t)L[l].co * L[l].ci * 27, 0.1f, 2), *b = L[l].osplit || L[l].sr ? nullptr : rnd(L[l].co, 0.1f, 3);
        j.w = w; j.b = b;
        if (L[l].gn) j.gp = (gnp_t){gam, bet, mean, rstd, G};
        if (L[l].stats) { j.osum = osum; j.Go = G; }
        double best = 1e9;
        for (unsigned it = 0; it < 12; it++) {
            nn_sync(); double t0 = now(); run(&j, it); nn_sync(); double t = (now() - t0) * 1e3;
            if (it >= 2 && t < best) best = t;
        }
        run(&j, 0); nn_sync();
        const uint64_t h = fnv(j.y, by1) ^ (c2 ? fnv(j.y2, by2) * 31 : 0);
        double hs[2] = {0, 0}; if (L[l].stats) nn_d2h(hs, osum, sizeof hs);
        printf("%-32s @%d %8.3f ms  out %016llx", L[l].nm, P, best, (unsigned long long)h);
        if (L[l].stats) printf("  stats %.9g %.9g", hs[0], hs[1]);
        printf("\n");
        nn_free((void *)j.x); nn_free(j.y); if (j.y2) nn_free(j.y2); nn_free(w); if (b) nn_free(b);
    }
    const char *e = nn_check(); if (e) { printf("cuda: %s\n", e); return 1; }
    return 0;
}
