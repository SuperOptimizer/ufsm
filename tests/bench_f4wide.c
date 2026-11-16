/* bench_f4wide: the wide MX-fp4 3^3 forward (conv_fwd_f4_k, Cip >= 96: every trunk conv of the production net above level 0) with one
   vs two weight-stage buffers (UFSM_F4_WDB, lp_f4_wdb), timed in isolation (min over repetitions), MX-fp4 in -> MX-fp4 out, plain and
   with a GroupNorm + SiLU input, plus an FNV hash of the output bytes (the two variants issue the same MMAs in the same order: equal
   hashes). Shapes: the try6 widths at a 96^3 / 48^3 grid (levels 1 / 2 of a P 768 lat-4 window).
   usage: bench_f4wide [REPS] */
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
static uint64_t fnv(const void *dp, size_t n) {
    uint8_t *h = malloc(n); nn_d2h(h, dp, n); uint64_t f = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { f ^= h[i]; f *= 1099511628211ull; }
    free(h); return f;
}
int main(int argc, char **argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 10;
    if (nn_init(0)) { printf("no GPU\n"); return 1; }
    static const int cs[][4] = {{256, 256, 96, 0}, {256, 320, 96, 0}, {448, 288, 48, 0}, {288, 288, 48, 0}, {256, 256, 96, 1}, {448, 448, 48, 1}};
    int bad = 0;
    for (size_t k = 0; k < sizeof cs / sizeof cs[0]; k++) {
        const int Ci = cs[k][0], Co = cs[k][1], P = cs[k][2], gn = cs[k][3];
        const shape5 xs = {1, Ci, P, P, P}; const size_t S = (size_t)P * P * P;
        float *xf = rnd(S * Ci, 4.f, 7 + k), *w = rnd((size_t)Co * Ci * 27, 0.1f, 11 + k), *b = rnd(Co, 0.1f, 13 + k);
        void *x4 = nn_malloc(lp_mx4_bytes(1, Ci, S)), *y4 = nn_malloc(lp_mx4_bytes(1, Co, S)); lp_f32_to_mx4(xf, 1, Ci, S, x4); nn_free(xf);
        float *gam = rnd(Ci, 0.5f, 17), *bet = rnd(Ci, 0.5f, 19), *mu = rnd(8, 0.1f, 23), *rs = rnd(8, 0.1f, 29);
        gnp_t gp = {}; if (gn) { gp.gamma = gam; gp.beta = bet; gp.mean = mu; gp.rstd = rs; gp.G = 8; }
        const split_t s0 = {};
        double t[2]; uint64_t h[2];
        for (int v = 0; v < 2; v++) {
            lp_f4_wdb(v); double best = 1e30;
            for (int r = 0; r < reps + 1; r++) {
                nn_sync(); const double t0 = now();
                lp_conv_fwd_f4(x4, 4, xs, w, b, Co, y4, 4, gp, nullptr, 0, s0);
                nn_sync(); const double dt = now() - t0; if (r && dt < best) best = dt;   /* (rep 0: warm-up, weight prep) */
            }
            t[v] = best; h[v] = fnv(y4, lp_mx4_bytes(1, Co, S));
        }
        const char *e = nn_check(), *e2 = lp_check();
        printf("fwd%s %3d -> %3d @ %d^3: one buffer %7.3f ms, two %7.3f ms (%.3fx), hashes %016llx %016llx %s%s%s\n", gn ? " gn" : "   ", Ci, Co, P, 1e3 * t[0], 1e3 * t[1], t[0] / t[1],
               (unsigned long long)h[0], (unsigned long long)h[1], h[0] == h[1] ? "equal" : "DIFFER", e ? " " : "", e ? e : e2 ? e2 : "");
        bad += h[0] != h[1] || e || e2;
        nn_free(x4); nn_free(y4); nn_free(w); nn_free(b); nn_free(gam); nn_free(bet); nn_free(mu); nn_free(rs);
    }
    lp_f4_wdb(-1);
    return bad ? 1 : 0;
}
