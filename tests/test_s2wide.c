/* conv_fwd_s2_f8_k with every output channel in one block (UFSM_S2_WIDE: MT 6 for 96 / 192 outputs, MT 8 for multiples of 128, MT 10 for 160) against
   its 32 / 64-row output tiles (lp_set_s2_wide 0 / 1 on the same operands): the same MMAs in the same order per accumulator, so every
   output byte must be identical. MX-fp4 inputs of 96 / 128 / 192 channels (the latent nets' level-0 -> 1 and 1 -> 2 down convs), with and
   without the input's GroupNorm + SiLU in staging, partial tiles (MX-fp4 in and out: the kernel takes one activation type). Usage: test_s2wide */
#include "nn.h"
#include "nn_lp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void lp_set_s2_wide(int on);
static uint64_t rs = 0x52a1d3ull;
static unsigned rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (unsigned)(rs >> 32); }
static float frand(void) { return (rnd() >> 8) * (1.f / 16777216.f) * 2.f - 1.f; }
static int bad;
static float *dev_rand(size_t n, float sc, float add) { float *h = malloc(n * 4); for (size_t i = 0; i < n; i++) h[i] = frand() * sc + add; float *d = nn_malloc(n * 4); nn_h2d(d, h, n * 4); free(h); return d; }
static void *mx_rand(shape5 s, float sc, int dt) {
    const size_t n = shape_numel(s); float *d = dev_rand(n, sc, 0.f);
    const size_t b = nn_mx_bytes(s, dt); void *p = nn_malloc(b); nn_zero(p, b); nn_set_storage(p, b, dt);
    if (dt == 4) lp_f32_to_mx4(d, s.n, s.c, shape_spatial(s), p); else lp_f32_to_mx8(d, s.n, s.c, shape_spatial(s), p);
    nn_free(d); return p;
}
static void run(int Ci, int Co, int D, int H, int W, int gn, int odt) {
    const shape5 xs = {1, Ci, D, H, W}, ys = {1, Co, D / 2, H / 2, W / 2};
    void *x = mx_rand(xs, 2.f, 4);
    float *w = dev_rand((size_t)Co * Ci * 27, 0.05f, 0.f), *b = dev_rand(Co, 0.1f, 0.f);
    const int G = 8;
    float *ga = dev_rand(Ci, 0.2f, 1.f), *be = dev_rand(Ci, 0.1f, 0.f), *mu = dev_rand(G, 0.1f, 0.f), *rsd = dev_rand(G, 0.1f, 1.f);
    const gnp_t gp = gn ? gnp_mk(ga, be, mu, rsd, G) : (gnp_t){0};
    const size_t yb = nn_mx_bytes(ys, odt);
    void *y[2]; unsigned char *h[2];
    for (int k = 0; k < 2; k++) {
        y[k] = nn_malloc(yb); nn_zero(y[k], yb); nn_set_storage(y[k], yb, odt);
        lp_set_s2_wide(k);
        lp_conv_fwd_s2_f8(x, 4, xs, w, b, Co, y[k], odt, ys, gp);
        nn_sync(); h[k] = malloc(yb); nn_d2h(h[k], y[k], yb);
    }
    lp_set_s2_wide(1);
    size_t nd = 0, nz = 0; for (size_t i = 0; i < yb; i++) { nd += h[0][i] != h[1][i]; nz += h[0][i] != 0; }
    const char *e = nn_check();
    const int ok = !e && nd == 0 && nz > yb / 4;
    printf("  %3d -> %3d %dx%dx%d gn %d out mx%d: %zu of %zu bytes differ%s %s\n", Ci, Co, D, H, W, gn, odt, nd, yb, e ? e : "", ok ? "ok" : "FAIL");
    bad += !ok;
    for (int k = 0; k < 2; k++) { nn_free(y[k]); free(h[k]); }
    nn_free(x); nn_free(w); nn_free(b); nn_free(ga); nn_free(be); nn_free(mu); nn_free(rsd);
}
int main(void) {
    nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0);
    run(96, 96, 32, 32, 48, 1, 4);    /* the level-0 -> 1 down conv (MT 6) */
    run(96, 96, 20, 18, 34, 0, 4);    /* partial tiles, plain input */
    run(128, 128, 24, 20, 40, 1, 4);  /* MT 8 */
    run(96, 192, 16, 16, 32, 1, 4);   /* MT 6, two output tiles */
    run(128, 256, 12, 16, 24, 0, 4);  /* MT 8, two output tiles */
    run(128, 160, 16, 20, 32, 1, 4);  /* MT 10 (the level-2 -> 3 down conv) */
    run(160, 160, 12, 12, 24, 1, 4);
    printf("%s\n", bad ? "test_s2wide: FAIL" : "test_s2wide: ok");
    return bad != 0;
}
