/* conv_fwd_f4v_k's training shapes in isolation (DESIGN 2026-11-17 "convk"): the decoder conv1 [up2 | skip] with both GroupNorms and
   the GN-input trunk conv, one 32-plane slab of try9's level 0 (192^2, P 768) by default. Time each arm under ncu (the sum of a call's
   kernels) or by the wall clock printed here (min of BENCH_REPS). UFSM_F4V=0 / 1 picks the kernel.
   usage: bench_f4v [case 0 dec c1 | 1 GN trunk | 2 bdata] [D 32] [HW 192] [Cx 320] [Cs 256] [Co 256] */
#include "nn.h"
#include "nn_lp.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t rs = 0x1234567ull;
static float *dev_rand(size_t n, float sc, float add) { float *h = malloc(n * 4); for (size_t i = 0; i < n; i++) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; h[i] = ((rs >> 40) / 16777216.f - 0.5f) * 2.f * sc + add; } float *d = nn_malloc(n * 4); nn_h2d(d, h, n * 4); free(h); return d; }
static void *mx4_rand(shape5 s) { const size_t n = shape_numel(s); float *d = dev_rand(n, 2.f, 0.f); const size_t b = nn_mx4_bytes(s); void *p = nn_malloc(b); nn_set_storage(p, b, 4); lp_f32_to_mx4(d, s.n, s.c, shape_spatial(s), p); nn_free(d); return p; }
int main(int argc, char **argv) {
    nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0);
    const int cs = argc > 1 ? atoi(argv[1]) : 0, D = argc > 2 ? atoi(argv[2]) : 32, HW = argc > 3 ? atoi(argv[3]) : 192, Cx = argc > 4 ? atoi(argv[4]) : 320,
              Cs = argc > 5 ? atoi(argv[5]) : 256, Co = argc > 6 ? atoi(argv[6]) : 256, reps = getenv("BENCH_REPS") ? atoi(getenv("BENCH_REPS")) : 3, G = 8;
    lp_wmemo_step(1);
    const int Ci = cs == 0 ? Cx + Cs : Cx;
    shape5 fs = {1, Ci, D, HW, HW}, ys = {1, Co, D, HW, HW};
    float *w = dev_rand((size_t)Co * Ci * 27, 0.04f, 0.f), *b = dev_rand(Co, 0.1f, 0.f);
    gnp_t gp = gnp_mk(dev_rand(Cx, 0.2f, 1.f), dev_rand(Cx, 0.1f, 0.f), dev_rand(G, 0.1f, 0.f), dev_rand(G, 0.1f, 1.f), G);
    split_t sp = {0};
    void *x;
    if (cs == 0) { shape5 c = {1, Cx, D / 2, HW / 2, HW / 2}, k = {1, Cs, D, HW, HW}; x = mx4_rand(c); sp.x2 = mx4_rand(k); sp.c_split = Cx; sp.up = 1;
                   sp.gp2 = gnp_mk(dev_rand(Cs, 0.3f, 1.f), dev_rand(Cs, 0.2f, 0.f), dev_rand(G, 0.2f, 0.f), dev_rand(G, 0.1f, 1.f), G); }
    else { x = mx4_rand(fs); if (cs == 2) { gp = (gnp_t){0}; sp.osr = 0x9e3779b9u; } }
    const size_t yb = nn_mx4_bytes(ys); void *y = nn_malloc(yb); nn_set_storage(y, yb, 4);
    double *os = nn_malloc(2 * 64 * 3 * 8);
    double best = 1e30;
    for (int it = 0; it <= reps; it++) {
        nn_zero(os, 2 * 64 * 3 * 8); nn_sync(); const double t0 = now();
        lp_conv_fwd_f4(x, 4, fs, w, b, Co, y, 4, gp, cs == 2 ? nullptr : os, cs == 2 ? 0 : G, sp);
        nn_sync(); const double dt = now() - t0; if (it && dt < best) best = dt;
    }
    const double fl = 2.0 * Co * Ci * 27 * (double)D * HW * HW;
    printf("bench_f4v case %d %d->%d %dx%d^2: %.2f ms (%.0f TFLOP/s)%s\n", cs, Ci, Co, D, HW, best * 1e3, fl / best * 1e-12, lp_check() ? lp_check() : "");
    return 0;
}
