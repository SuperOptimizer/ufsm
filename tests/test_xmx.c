/* test_xmx: xnet's MX-fp4 implicit-GEMM convolution (UFSM_XNET_MX 6 / 5, src/nn_xnet.cu conv_mx) against a host reference with the same
   quantisation, written here from the format's definition: activations per voxel in blocks of 16 channels, weights per output row in
   blocks of 16 K (K = tap-major, channels inner); 6: power-of-two scale 2^e, e = ceil(log2(amax / 6)); 5: scale e4m3(amax / 6) (times a
   per-row fp32 scale amax_row / 2688 for weights); values round to nearest even on the e2m1 grid, saturating. The host sums in double;
   the GPU in fp32 then rounds to fp16, so the bound is relative to the output's magnitude. Shapes: 3x3x3 stride 1 / 2 with padding at
   odd edges, 1x1, widths 32 .. 320, two samples, and tile edges (M not a multiple of the tile). */
#include "xnet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float e2m1(float v) {
    static const float g[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float a = fabsf(v); int b = 0; float bd = 1e30f;
    for (int i = 0; i < 8; i++) { const float d = fabsf(a - g[i]); if (d < bd || (d == bd && !(i & 1))) { bd = d; b = i; } }
    return copysignf(g[b], v);
}
static float e4m3(float v) {
    const float a = fabsf(v); if (a >= 448.f) return 448.f; if (a == 0.f) return 0.f;
    int ex; frexpf(a, &ex); ex -= 1; if (ex < -6) ex = -6;
    const float ulp = ldexpf(1.f, ex - 3); return nearbyintf(a / ulp) * ulp;
}
static void q16(float *v, int sf, float gs) {   /* one block of 16 in place */
    float am = 0.f; for (int i = 0; i < 16; i++) am = fmaxf(am, fabsf(v[i]));
    if (am == 0.f) return;
    float d;
    if (sf == 6) { int e; frexpf(am / 6.f, &e); if (ldexpf(1.f, e - 1) >= am / 6.f) e--; d = ldexpf(1.f, e); }
    else d = e4m3(am / 6.f / gs) * gs;
    for (int i = 0; i < 16; i++) v[i] = d > 0.f ? e2m1(v[i] / d) * d : 0.f;
}
static float h2f(uint16_t h) {   /* fp16 bits -> float */
    const int s = h >> 15, e = (h >> 10) & 31, m = h & 1023; float v;
    if (e == 0) v = ldexpf((float)m, -24); else if (e == 31) v = m ? NAN : INFINITY; else v = ldexpf((float)(m | 1024), e - 25);
    return s ? -v : v;
}
static uint16_t f2h(float f) {   /* round to nearest even */
    uint32_t x; memcpy(&x, &f, 4); const uint32_t s = (x >> 16) & 0x8000; float a = fabsf(f);
    if (a >= 65520.f) return (uint16_t)(s | 0x7c00);
    if (a < ldexpf(1.f, -14)) { const float m = nearbyintf(a / ldexpf(1.f, -24)); return (uint16_t)(s | (uint32_t)m); }
    int e; float fr = frexpf(a, &e); float m = nearbyintf(fr * 2048.f); if (m >= 2048.f) { m = 1024.f; e++; }
    return (uint16_t)(s | (uint32_t)(e + 14) << 10 | ((uint32_t)m - 1024));
}
static uint32_t rs = 12345; static float rnd(void) { rs = rs * 1664525u + 1013904223u; return (float)(rs >> 8) / 16777216.f * 2.f - 1.f; }

static int run(int sf, int N, int D, int H, int W, int ci, int co, int k, int st) {
    const int pd = k / 2, Do = (D + 2 * pd - k) / st + 1, Ho = (H + 2 * pd - k) / st + 1, Wo = (W + 2 * pd - k) / st + 1, K = k * k * k * ci;
    const size_t nx = (size_t)N * D * H * W * ci, ny = (size_t)N * Do * Ho * Wo * co;
    uint16_t *xh = malloc(nx * 2), *yh = malloc(ny * 2); float *x = malloc(nx * 4), *w = malloc((size_t)co * ci * k * k * k * 4), *wk = malloc((size_t)co * K * 4), *b = malloc(co * 4);
    for (size_t i = 0; i < nx; i++) { float v = rnd() * (i % 7 == 0 ? 8.f : 1.f); if (i % 97 == 0) v = 0.f; xh[i] = f2h(v); x[i] = h2f(xh[i]); }
    for (size_t i = 0; i < (size_t)co * ci * k * k * k; i++) w[i] = rnd() * 0.05f * (i % 13 == 0 ? 4.f : 1.f);
    for (int o = 0; o < co; o++) b[o] = rnd();
    /* the reference: quantised x and weights (GEMM order), double sums */
    for (size_t v = 0; v < nx / ci; v++) for (int c = 0; c < ci; c += 16) q16(x + v * ci + c, sf, 1.f);
    for (int o = 0; o < co; o++) {
        float am = 0.f;
        for (int i = 0; i < ci; i++) for (int t = 0; t < k * k * k; t++) { const float v = w[((size_t)o * ci + i) * k * k * k + t]; wk[(size_t)o * K + t * ci + i] = v; am = fmaxf(am, fabsf(v)); }
        for (int kk = 0; kk < K; kk += 16) q16(wk + (size_t)o * K + kk, sf, sf == 5 && am > 0.f ? am / 2688.f : 1.f);
    }
    xnet_set_mx(sf);
    xconv c; if (xconv_init(&c, w, co, ci, k, k, k, st, st, st, pd, pd, pd, nullptr, b)) { fprintf(stderr, "xconv_init failed\n"); return 1; }
    xt xd = xt_new(N, D, H, W, ci); xh2d(xd.p, xh, nx * 2);
    xt yd = xconv_fwd(&c, xd, nullptr, XA_NONE); xd2h(yh, yd.p, ny * 2);
    if (xnet_check("test_xmx")) return 1;
    double emax = 0, ymax = 0; size_t bad = 0;
    for (int n = 0; n < N; n++) for (int z = 0; z < Do; z++) for (int y = 0; y < Ho; y++) for (int xx = 0; xx < Wo; xx++) for (int o = 0; o < co; o++) {
        double s = b[o];
        for (int a = 0; a < k; a++) for (int bb = 0; bb < k; bb++) for (int e = 0; e < k; e++) {
            const int zi = z * st - pd + a, yi = y * st - pd + bb, xi = xx * st - pd + e;
            if (zi < 0 || zi >= D || yi < 0 || yi >= H || xi < 0 || xi >= W) continue;
            const float *xr = x + ((((size_t)n * D + zi) * H + yi) * W + xi) * ci, *wr = wk + (size_t)o * K + ((a * k + bb) * k + e) * ci;
            for (int i = 0; i < ci; i++) s += (double)xr[i] * wr[i];
        }
        const size_t yi = ((((size_t)n * Do + z) * Ho + y) * Wo + xx) * co + o; const double g = h2f(yh[yi]), d = fabs(g - s);
        if (d > emax) emax = d; if (fabs(s) > ymax) ymax = fabs(s);
        if (d > 1e-3 * fabs(s) + 2e-4 * ymax + 1e-3) bad++;
    }
    printf("sf %d N %d %dx%dx%d ci %3d co %3d k %d s %d: max |d| %.3g (max |y| %.3g), %zu of %zu over the bound\n", sf, N, D, H, W, ci, co, k, st, emax, ymax, bad, ny);
    xt_free(&xd); xt_free(&yd); xconv_free(&c); free(xh); free(yh); free(x); free(w); free(wk); free(b);
    return bad != 0;
}

#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
/* XMX_BENCH=1: time the gambiarra-sized convs (16 crops) per format (0 fp16, 6, 5): TMAC/s */
static void bench(void) {
    static const int S[][5] = {{64, 32, 32, 3, 1}, {64, 64, 32, 3, 1}, {64, 32, 64, 3, 2}, {32, 64, 64, 3, 1}, {32, 128, 64, 3, 1}, {16, 128, 128, 3, 1}, {16, 256, 128, 3, 1}, {8, 256, 256, 3, 1}, {32, 64, 256, 1, 1}};
    for (int f = 0; f < 3; f++) {
        const int sf = f == 0 ? 0 : f == 1 ? 6 : 5; xnet_set_mx(sf);
        for (size_t i = 0; i < sizeof S / sizeof S[0]; i++) {
            const int D = S[i][0], ci = S[i][1], co = S[i][2], k = S[i][3], st = S[i][4], N = 16;
            float *w = calloc((size_t)co * ci * k * k * k, 4); for (size_t j = 0; j < (size_t)co * ci * k * k * k; j++) w[j] = rnd() * 0.05f;
            xconv c; xconv_init(&c, w, co, ci, k, k, k, st, st, st, k / 2, k / 2, k / 2, nullptr, nullptr);
            xt x = xt_new(N, D, D, D, ci); xt y = xconv_fwd(&c, x, nullptr, XA_NONE); xt_free(&y); xnet_sync();
            const double t0 = now(); const int R = 10; for (int r = 0; r < R; r++) { y = xconv_fwd(&c, x, nullptr, XA_NONE); xt_free(&y); } xnet_sync(); const double t = (now() - t0) / R;
            const int Do = (D + 2 * (k / 2) - k) / st + 1; const double mac = (double)N * Do * Do * Do * co * ci * k * k * k;
            printf("bench fmt %d %3d^3 ci %3d co %3d k %d s %d: %8.3f ms %6.1f TMAC/s\n", sf, D, ci, co, k, st, t * 1e3, mac / t / 1e12);
            xt_free(&x); xconv_free(&c); free(w);
        }
    }
}
int main(void) {
    if (xnet_init(0)) { printf("SKIP: no GPU\n"); return 0; }
    if (getenv("XMX_BENCH")) { bench(); return 0; }
    int fails = 0;
    for (int sf = 6; sf >= 5; sf--) {
        fails += run(sf, 2, 9, 7, 11, 32, 32, 3, 1);
        fails += run(sf, 1, 16, 16, 16, 64, 64, 3, 1);
        fails += run(sf, 2, 10, 9, 8, 32, 64, 3, 2);
        fails += run(sf, 1, 6, 5, 7, 128, 256, 1, 1);
        fails += run(sf, 2, 4, 4, 4, 320, 320, 3, 1);
        fails += run(sf, 1, 5, 6, 4, 96, 128, 3, 1);
        fails += run(sf, 2, 5, 10, 40, 32, 32, 3, 1);    /* halo kernel: TX 32 tiles with ragged edges */
        fails += run(sf, 1, 7, 9, 36, 128, 64, 3, 1);    /* halo kernel: two chunk pairs, ragged */
        fails += run(sf, 1, 6, 6, 20, 64, 96, 3, 1);     /* halo kernel: TX 16, outputs not a multiple of 32 */
    }
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
