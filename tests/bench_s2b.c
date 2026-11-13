/* The 92 M net's stride-2 backward-data (down0..down3: 256..448 channels, MX-fp4 gy -> MX-fp4 gx accumulated) timed in isolation: the
   weight-streaming fp4 kernel (UFSM_S2B_F4 3) against the dilated path it replaces (0), min over repetitions, with an FNV hash of the
   output bytes. usage: bench_s2b [D H W C]... (gx shape; default 32-plane slabs of the P 1024 split half's down0 / down1 / down2 and down3 whole:
   32 x 256 x 256 x 256, 32 x 128 x 128 x 320, 32 x 64 x 64 x 384, 16 x 32 x 32 x 448; the half is 136 / 68 / 34 / 18 planes deep) */
#include "nn.h"
#include "nn_lp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t rs = 88172645463325252ull;
static float frand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 40) / 16777216.0 - 0.5); }
static void *mx4_rand(shape5 s, float sc) {   /* quantised in chunks so the fp32 copy stays small */
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
        void *gy = mx4_rand(ys, 0.02f), *gx0 = mx4_rand(xs, 0.02f), *gx = nn_malloc(nn_mx4_bytes(xs)); nn_set_storage(gx, nn_mx4_bytes(xs), 4);
        float *scr = nn_malloc(nn_conv3d_scratch(xs, C, 3) * 2 + shape_numel(xs) * 4);   /* (as test_mx4: the dilated path's) */
        const double fl = 2.0 * shape_spatial(xs) / 8 * 27 * C;   /* 2 x MACs / C: each gy voxel x 27 taps x C x C */
        for (int f4 = 0; f4 < 2; f4++) {
            lp_set_s2b_f4(f4 ? 3 : 0);
            double best = 1e9; uint64_t h = 0;
            for (int r = 0; r < 4; r++) {
                nn_copy_async(gx, gx0, nn_mx4_bytes(xs)); nn_sync();
                const double t = now(); nn_conv3d_bwd_data_acc(gy, ys, w, xs, 3, 2, (float *)gx, scr); nn_sync(); const double e = now() - t;
                if (e < best) best = e;
                if (!r) h = fnv(gx, nn_mx4_bytes(xs));
            }
            printf("gx %dx%dx%d C %d  %-22s %8.2f ms  %6.1f TFLOP/s  hash %016llx\n", xs.d, xs.h, xs.w, C, f4 ? "fp4 streaming kernel" : "dilated path", best * 1e3,
                   fl * C / best / 1e12, (unsigned long long)h);
        }
        lp_set_s2b_f4(-1);
        nn_free(w); nn_free(gy); nn_free(gx0); nn_free(gx); nn_free(scr);
    }
    return 0;
}
