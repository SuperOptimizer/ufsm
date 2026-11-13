/* dense vs 2:4 sparse (pair-wise 4:8) MX-fp4 3^3 convs at the scratch2 widths (92 M net, P 1024 two-card split: per-card half
   window; one 32-plane slab per level, 16 / 8 / 4 planes below level 3), timed in isolation (min of BENCH_REPS, prepared-weight memo
   on, so the weight prep is outside the timing as in training):
     fwd    MX-fp4 x with GN + SiLU in staging -> MX-fp4 y + GN statistics (c2 convs; c1 convs read the raw block output: plain)
     bdata  MX-fp4 gy -> MX-fp4 gx, output SR (the backward-data through flipped weights, as nn_bwd.cu)
   The output hashes of dense and sparse differ (different MMA grouping); test_sp24_conv checks equality on exact operands.
   usage: bench_sp24 [level mask, default all]   (DESIGN.md 2026-11-13 "2:4 sparse fp4 convs") */
#include "nn.h"
#include "nn_lp.h"
#include "lp_sp24.h"
#include <math.h>
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
/* an MX tensor [1][C][S] holding the values of rnd(S C, sc, seed) (the same stream, so the same bytes at any P); quantised in voxel
   chunks of <= 256 MB fp32, so the fp32 tensor never sits on the device whole (P 384 did not fit 16 GB) */
static void *mx(int bits, int C, size_t S, float sc, uint64_t seed) {
    const int bw = C <= 8 ? 8 : C <= 16 ? 16 : 32, nb = (C + bw - 1) / bw, rb = bw * bits / 8;   /* lp_common.cuh mx_bw / mx_nb / mx_rb */
    void *q = nn_malloc(bits == 4 ? lp_mx4_bytes(1, C, S) : lp_mx8_bytes(1, C, S));
    size_t L = S;
    while (L > 1 && (size_t)C * L * 4 > ((size_t)256 << 20)) L = (L + 1) / 2;
    if (L == S) {
        float *f = rnd(S * C, sc, seed);
        if (bits == 4) lp_f32_to_mx4(f, 1, C, S, q); else lp_f32_to_mx8(f, 1, C, S, q);
        nn_free(f); return q;
    }
    float *h = malloc(S * C * 4), *hc = malloc(L * C * 4), *d = nn_malloc(L * C * 4);
    uint64_t r = seed;
    for (size_t i = 0; i < S * C; i++) { r ^= r << 13; r ^= r >> 7; r ^= r << 17; h[i] = ((float)(r >> 40) / 16777216.f - 0.5f) * sc; }
    uint8_t *t = nn_malloc(bits == 4 ? lp_mx4_bytes(1, C, L) : lp_mx8_bytes(1, C, L)), *qb = q;
    for (size_t s0 = 0; s0 < S; s0 += L) {
        const size_t l = S - s0 < L ? S - s0 : L;
        for (int c = 0; c < C; c++) memcpy(hc + (size_t)c * l, h + (size_t)c * S + s0, l * 4);
        nn_h2d(d, hc, l * C * 4);
        if (bits == 4) lp_f32_to_mx4(d, 1, C, l, t); else lp_f32_to_mx8(d, 1, C, l, t);
        for (int b = 0; b < nb; b++) {   /* rows [nb][S][rb], then the scale bytes [nb][S] */
            nn_d2d(qb + ((size_t)b * S + s0) * rb, t + (size_t)b * l * rb, l * rb);
            nn_d2d(qb + (size_t)nb * S * rb + (size_t)b * S + s0, t + (size_t)nb * l * rb + (size_t)b * l, l);
        }
    }
    free(h); free(hc); nn_free(d); nn_free(t);
    return q;
}
static void mask48(float *w, int Co, int Ci) {   /* host pair-wise 4:8 (tests/test_sp24_conv.c host_mask48) */
    for (int co = 0; co < Co; co++) for (int t = 0; t < 27; t++) for (int c0 = 0; c0 + 8 <= Ci; c0 += 8) {
        float m[4]; for (int b = 0; b < 4; b++) m[b] = fabsf(w[((size_t)co * Ci + c0 + 2 * b) * 27 + t]) + fabsf(w[((size_t)co * Ci + c0 + 2 * b + 1) * 27 + t]);
        int i0 = m[1] > m[0] ? 1 : 0, i1 = 1 - i0;
        for (int b = 2; b < 4; b++) { if (m[b] > m[i0]) { i1 = i0; i0 = b; } else if (m[b] > m[i1]) i1 = b; }
        for (int b = 0; b < 4; b++) if (b != i0 && b != i1) { w[((size_t)co * Ci + c0 + 2 * b) * 27 + t] = 0.f; w[((size_t)co * Ci + c0 + 2 * b + 1) * 27 + t] = 0.f; }
    }
}
int main(int argc, char **argv) {
    nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0);
    const int lmask = argc > 1 ? atoi(argv[1]) : 63, reps = getenv("BENCH_REPS") ? atoi(getenv("BENCH_REPS")) : 5;
    /* level: width, slab planes, in-plane size (P 1024 latent rungs: 256^2 at level 0) */
    const int Wd[6] = {256, 320, 384, 448, 288, 288}, Dz[6] = {32, 32, 32, 16, 8, 4}, HW[6] = {256, 128, 64, 32, 16, 8};
    lp_wmemo_step(1);
    printf("%-6s %-6s %22s %10s %10s %8s %8s\n", "level", "pass", "shape", "dense ms", "sparse ms", "speedup", "TF/s sp");
    for (int l = 0; l < 6; l++) {
        if (!((lmask >> l) & 1)) continue;
        const int C = Wd[l]; shape5 xs = {1, C, Dz[l], HW[l], HW[l]};
        const size_t S = shape_spatial(xs), nw = (size_t)C * C * 27;
        float *hw = malloc(nw * 4); uint64_t r = 77 + l;
        for (size_t i = 0; i < nw; i++) { r ^= r << 13; r ^= r >> 7; r ^= r << 17; hw[i] = ((float)(r >> 40) / 16777216.f - 0.5f) * 0.05f; }
        mask48(hw, C, C);
        float *w = nn_malloc(nw * 4); nn_h2d(w, hw, nw * 4); free(hw);
        float *b = rnd(C, 0.1f, 5);
        void *x = mx(4, C, S, 4.f, 11 + l), *y = nn_malloc(lp_mx4_bytes(1, C, S));
        const int G = 32;
        float *ga = rnd(C, 0.2f, 21), *be = rnd(C, 0.1f, 22), *mu = rnd(G, 0.1f, 23), *rsd = rnd(G, 0.1f, 24);
        { float *h = malloc(C * 4); nn_d2h(h, ga, C * 4); for (int c = 0; c < C; c++) h[c] += 1.f; nn_h2d(ga, h, C * 4); nn_d2h(h, rsd, G * 4); for (int g = 0; g < G; g++) h[g] += 1.f; nn_h2d(rsd, h, G * 4); free(h); }
        double *osum = nn_malloc(2 * 64 * sizeof(double) * 64);
        const double flop = 2.0 * C * C * 27 * (double)S;
        for (int pass = 0; pass < 2; pass++) {
            double best[2] = {1e30, 1e30};
            for (int sv = 0; sv < 2; sv++) {
                lp_sp24_force(sv);
                for (int it = 0; it <= reps; it++) {
                    split_t sp = {0}; sp.wkey = 100 + 2 * l + pass;
                    gnp_t gp = {0};
                    if (pass == 0) gp = gnp_mk(ga, be, mu, rsd, G); else sp.osr = 0x6a09e667u + it;
                    nn_sync(); const double t0 = now();
                    if (!(sv && lp_sp24_conv_fwd(x, 4, xs, w, b, C, y, 4, gp, pass == 0 ? osum : nullptr, pass == 0 ? G : 0, sp)))
                        lp_conv_fwd_f4(x, 4, xs, w, b, C, y, 4, gp, pass == 0 ? osum : nullptr, pass == 0 ? G : 0, sp);
                    nn_sync(); const double dt = now() - t0;
                    if (it && dt < best[sv]) best[sv] = dt;
                }
            }
            lp_sp24_force(-1);
            const char *e = lp_check();
            char shp[32]; snprintf(shp, sizeof shp, "%dx%d^2x%d", Dz[l], HW[l], C);
            printf("L%-5d %-6s %22s %10.2f %10.2f %7.2fx %8.0f%s%s\n", l, pass ? "bdata" : "fwd", shp, best[0] * 1e3, best[1] * 1e3, best[0] / best[1], flop / best[1] * 1e-12,
                   e ? "  ERROR " : "", e ? e : "");
        }
        nn_free(w); nn_free(b); nn_free(x); nn_free(y); nn_free(ga); nn_free(be); nn_free(mu); nn_free(rsd); nn_free(osum);
    }
    return 0;
}
