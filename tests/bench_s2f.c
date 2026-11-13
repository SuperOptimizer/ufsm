/* The 92 M net's stride-2 forward (down0..down3: 256..448 channels in and out, MX-fp4 x -> MX-fp4 y) timed in isolation: the fp8 kernel
   (today's default) against conv_fwd_s2_f4_k GEN (UFSM_S2_F4G=1), min over repetitions, with an FNV hash of the output bytes.
   usage: bench_s2f [D H W C]... (x shape; default 32-plane slabs of the P 1024 split half's down0 / down1 / down2 and down3 whole) */
#include "nn.h"
#include "nn_lp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t rs = 88172645463325252ull;
static float frand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 40) / 16777216.0 - 0.5); }
static void *mx4_rand(shape5 s, float sc) {
    const size_t S = shape_spatial(s), b = nn_mx4_bytes(s); void *p = nn_malloc(b); nn_zero(p, b); nn_set_storage(p, b, 4);
    float *h = malloc((size_t)s.c * S * 4); for (size_t i = 0; i < (size_t)s.c * S; i++) h[i] = frand() * sc;
    float *d = nn_malloc((size_t)s.c * S * 4); nn_h2d(d, h, (size_t)s.c * S * 4); lp_f32_to_mx4(d, 1, s.c, S, p); nn_free(d); free(h); return p;
}
static uint64_t fnv(const void *d, size_t n) { uint8_t *h = malloc(n); nn_d2h(h, d, n); uint64_t x = 1469598103934665603ull; for (size_t i = 0; i < n; i++) x = (x ^ h[i]) * 1099511628211ull; free(h); return x; }
int main(int argc, char **argv) {
    nn_init(0); nn_set_prec(3);
    int sh[4][4] = {{32, 256, 256, 256}, {32, 128, 128, 320}, {32, 64, 64, 384}, {16, 32, 32, 448}}, ns = 4;
    if (argc >= 5) { ns = 0; for (int a = 1; a + 3 < argc && ns < 4; a += 4, ns++) for (int k = 0; k < 4; k++) sh[ns][k] = atoi(argv[a + k]); }
    for (int i = 0; i < ns; i++) {
        const int C = sh[i][3];
        shape5 xs = {1, C, sh[i][0], sh[i][1], sh[i][2]}, ys = {1, C, xs.d / 2, xs.h / 2, xs.w / 2};
        float *hw = malloc((size_t)C * C * 27 * 4); for (size_t k = 0; k < (size_t)C * C * 27; k++) hw[k] = frand() * 0.05f;
        float *w = nn_malloc((size_t)C * C * 27 * 4); nn_h2d(w, hw, (size_t)C * C * 27 * 4); free(hw);
        float *b = nn_malloc((size_t)C * 4); nn_zero(b, (size_t)C * 4);
        void *x = mx4_rand(xs, 1.f), *y = nn_malloc(nn_mx4_bytes(ys)); nn_set_storage(y, nn_mx4_bytes(ys), 4);
        const double fl = 2.0 * shape_spatial(ys) * 27 * C * C;
        for (int g = 0; g < 2; g++) {
            lp_set_s2_f4g(g);
            double best = 1e9; uint64_t h = 0; int rc = 0;
            for (int r = 0; r < 4; r++) {
                nn_sync(); const double t = now(); rc = lp_conv_fwd_s2_f8(x, 4, xs, w, b, C, y, 4, ys, (gnp_t){0}); nn_sync(); const double e = now() - t;
                if (e < best) best = e;
                if (!r) h = fnv(y, nn_mx4_bytes(ys));
            }
            printf("x %dx%dx%d C %d  %-16s rc %d %8.2f ms  %6.1f TFLOP/s  hash %016llx\n", xs.d, xs.h, xs.w, C, g ? "fp4 GEN" : "fp8 kernel", rc, best * 1e3, fl / best / 1e12, (unsigned long long)h);
        }
        lp_set_s2_f4g(-1);
        nn_free(w); nn_free(b); nn_free(x); nn_free(y);
    }
    return 0;
}
