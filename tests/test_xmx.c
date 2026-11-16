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
static float e2m1_div(float v, float d) {   /* e2m1(v / d) without dividing: exact under fast math (x / d -> x * (1 / d) flipped ties under mcc) */
    static const double M[7] = {.25, .75, 1.25, 1.75, 2.5, 3.5, 5.}; static const float G[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const double a = fabs((double)v); int b = 0;
    for (int i = 0; i < 7; i++) { const double t = M[i] * (double)d; if (a > t) { b = i + 1; continue; } if (a == t) b = (i & 1) ? i + 1 : i; break; }
    return copysignf(G[b], v);
}
static void q16(float *v, int sf, float gs) {   /* one block of 16 in place */
    float am = 0.f; for (int i = 0; i < 16; i++) am = fmaxf(am, fabsf(v[i]));
    if (am == 0.f) return;
    float d;
    if (sf == 6) { int e; frexpf(am / 6.f, &e); if (ldexpf(1.f, e - 1) >= am / 6.f) e--; d = ldexpf(1.f, e); }
    else d = e4m3(gs == 1.f ? (float)((double)am / 6.0) : am / 6.f / gs) * gs;
    for (int i = 0; i < 16; i++) v[i] = d > 0.f ? (sf == 6 ? e2m1(v[i] / d) : e2m1_div(v[i], d)) * d : 0.f;
}
/* the least-squares weight quantiser (UFSM_XNET_MXQ 1 / 2), from its definition: per 16-block the scale of least squared error among
   the e4m3 values 4 codes below .. 2 above e4m3(amax / 6 / gs) (5; in double; the first of equal errors: the amax code, then ascending) or the
   exponents e, e - 1 (6); mode 2 (5 only): the row scale gs = amax_row / 2688 (double) 2^(j / 8), j 0..7, of least row error (the first of equal: a later candidate wins only when 1e-9 better) */
static float e4v(int cc) { const int E = (cc >> 3) & 15, M = cc & 7; return E ? ldexpf(1.f + M / 8.f, E - 7) : ldexpf(M / 8.f, -6); }
static double qerr(const float *v, float d, int sf, float *out) {
    double e = 0.0;
    for (int i = 0; i < 16; i++) {   /* the error in double (G d is not always a float) */
        const double q = (double)(d > 0.f ? (sf == 6 ? e2m1(v[i] / d) : e2m1_div(v[i], d)) : 0.f) * d; if (out) out[i] = (float)q; e += (q - v[i]) * (q - v[i]);
    }
    return e;
}
static double q16s(float *v, int sf, float gs, int apply) {   /* least-squares block scale; returns the block's error */
    float am = 0.f; for (int i = 0; i < 16; i++) am = fmaxf(am, fabsf(v[i]));
    if (am == 0.f) return 0.0;
    float bd;
    if (sf == 6) {
        int e; frexpf(am / 6.f, &e); if (ldexpf(1.f, e - 1) >= am / 6.f) e--;
        bd = ldexpf(1.f, e); if (qerr(v, ldexpf(1.f, e - 1), 6, nullptr) < qerr(v, bd, 6, nullptr) * (1.0 - 1e-9)) bd = ldexpf(1.f, e - 1);
    } else {
        const float sv = e4m3((float)((double)am / 6.0 / gs)); int c0 = 0; while (c0 < 126 && e4v(c0) != sv) c0++;
        bd = sv * gs; double be = qerr(v, bd, 5, nullptr);
        for (int cc = c0 - 4; cc <= c0 + 2; cc++) { if (cc < 1 || cc > 126 || cc == c0) continue; const double t = qerr(v, e4v(cc) * gs, 5, nullptr); if (t < be * (1.0 - 1e-9)) { be = t; bd = e4v(cc) * gs; } }
    }
    return qerr(v, bd, sf, apply ? v : nullptr);
}
static void qrow_s(float *r, int K, int sf, int mode, float am) {   /* K % 16 == 0 */
    float gs = sf == 5 ? (float)((double)am / 2688.0) : 1.f;
    if (mode >= 2 && sf == 5) {
        double best = -1.0; const float g0 = gs; float tmp[16];
        for (int j = 0; j < 8; j++) {
            static const float R8[8] = {1.f, 1.0905077f, 1.1892071f, 1.2968396f, 1.4142135f, 1.5422108f, 1.6817929f, 1.8340081f};
            const float gj = g0 * R8[j]; double t = 0.0;
            for (int k = 0; k < K; k += 16) { memcpy(tmp, r + k, sizeof tmp); t += q16s(tmp, 5, gj, 0); }
            if (best < 0.0 || t < best * (1.0 - 1e-9)) { best = t; gs = gj; }
        }
    }
    for (int k = 0; k < K; k += 16) q16s(r + k, sf, gs, 1);
}
static int g_rot = 0, g_mxa = 0;   /* g_mxa (UFSM_XNET_MXA): least-squares activation scales, a later candidate only when 1e-5 better */
static void q16a(float *v, int sf) {   /* one activation block (k_xq MODE bit 1) */
    float am = 0.f; for (int i = 0; i < 16; i++) am = fmaxf(am, fabsf(v[i]));
    if (am == 0.f || !g_mxa) { q16(v, sf, 1.f); return; }
    float bd;
    if (sf == 6) { int e; frexpf(am / 6.f, &e); if (ldexpf(1.f, e - 1) >= am / 6.f) e--; bd = ldexpf(1.f, e); if (qerr(v, ldexpf(1.f, e - 1), 6, nullptr) < qerr(v, bd, 6, nullptr) * (1.0 - 1e-5)) bd = ldexpf(1.f, e - 1); }
    else {
        const float sv = e4m3((float)((double)am / 6.0)); int c0 = 0; while (c0 < 126 && e4v(c0) != sv) c0++;
        bd = sv; double be = qerr(v, bd, 5, nullptr);
        for (int cc = c0 - 4; cc <= c0 + 2; cc++) { if (cc < 1 || cc > 126 || cc == c0) continue; const double t = qerr(v, e4v(cc), 5, nullptr); if (t < be * (1.0 - 1e-5)) { be = t; bd = e4v(cc); } }
    }
    qerr(v, bd, sf, v);
}   /* UFSM_XNET_MXROT: x and w 16-blocks rotated by H16 / 4 before the quantisation */
static void wht16h(float *u) { for (int len = 1; len < 16; len <<= 1) for (int i = 0; i < 16; i += 2 * len) for (int j = i; j < i + len; j++) { const float a = u[j], c = u[j + len]; u[j] = a + c; u[j + len] = a - c; } for (int i = 0; i < 16; i++) u[i] *= .25f; }
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

static int run(int sf, int N, int D, int H, int W, int ci, int co, int k, int st, int mode) {
    const int pd = k / 2, Do = (D + 2 * pd - k) / st + 1, Ho = (H + 2 * pd - k) / st + 1, Wo = (W + 2 * pd - k) / st + 1, K = k * k * k * ci;
    const size_t nx = (size_t)N * D * H * W * ci, ny = (size_t)N * Do * Ho * Wo * co;
    uint16_t *xh = malloc(nx * 2), *yh = malloc(ny * 2); float *x = malloc(nx * 4), *w = malloc((size_t)co * ci * k * k * k * 4), *wk = malloc((size_t)co * K * 4), *b = malloc(co * 4);
    for (size_t i = 0; i < nx; i++) { float v = rnd() * (i % 7 == 0 ? 8.f : 1.f); if (i % 97 == 0) v = 0.f; xh[i] = f2h(v); x[i] = h2f(xh[i]); }
    for (size_t i = 0; i < (size_t)co * ci * k * k * k; i++) w[i] = rnd() * 0.05f * (i % 13 == 0 ? 4.f : 1.f);
    for (int o = 0; o < co; o++) b[o] = rnd();
    /* the reference: quantised x and weights (GEMM order), double sums */
    if (sf) for (size_t v = 0; v < nx / ci; v++) for (int c = 0; c < ci; c += 16) { if (g_rot) wht16h(x + v * ci + c); q16a(x + v * ci + c, sf); }
    for (int o = 0; o < co; o++) {
        float am = 0.f;
        for (int i = 0; i < ci; i++) for (int t = 0; t < k * k * k; t++) { const float v = w[((size_t)o * ci + i) * k * k * k + t]; wk[(size_t)o * K + t * ci + i] = v; am = fmaxf(am, fabsf(v)); }
        if (g_rot) { am = 0.f; for (int kk = 0; kk < K; kk += 16) wht16h(wk + (size_t)o * K + kk); for (int kk = 0; kk < K; kk++) am = fmaxf(am, fabsf(wk[(size_t)o * K + kk])); }
        if (!sf) for (int kk = 0; kk < K; kk++) wk[(size_t)o * K + kk] = h2f(f2h(wk[(size_t)o * K + kk]));   /* fp16 path: fp16 weights */
        else if (mode && am > 0.f) qrow_s(wk + (size_t)o * K, K, sf, mode, am);
        else for (int kk = 0; kk < K; kk += 16) q16(wk + (size_t)o * K + kk, sf, sf == 5 && am > 0.f ? am / 2688.f : 1.f);
    }
    xnet_set_mx(sf); xnet_set_mxq(mode); xnet_set_mxrot(g_rot); xnet_set_mxa(g_mxa);
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
    printf("sf %d q %d%s%s N %d %dx%dx%d ci %3d co %3d k %d s %d: max |d| %.3g (max |y| %.3g), %zu of %zu over the bound\n", sf, mode, g_rot ? " rot" : "", g_mxa ? " lsa" : "", N, D, H, W, ci, co, k, st, emax, ymax, bad, ny);
    xt_free(&xd); xt_free(&yd); xconv_free(&c); free(xh); free(yh); free(x); free(w); free(wk); free(b);
    return bad != 0;
}

/* xgn (GroupNorm + residual + LReLU) against a double host reference, and run to run byte equality (the stats are deterministic) */
static int run_gn(int N, int D, int H, int W, int C, int G, float off) {
    const size_t S = (size_t)D * H * W, ne = (size_t)N * S * C;
    uint16_t *xh = malloc(ne * 2), *rh = malloc(ne * 2), *y1 = malloc(ne * 2), *y2 = malloc(ne * 2); float *ga = malloc(C * 4), *be = malloc(C * 4);
    for (size_t i = 0; i < ne; i++) { xh[i] = f2h(off * (1.f + 0.1f * (float)(i % C) / C) + rnd() * 2.f); rh[i] = f2h(rnd()); }
    for (int c = 0; c < C; c++) { ga[c] = 1.f + 0.5f * rnd(); be[c] = 0.3f * rnd(); }
    float *gd = xdev_floats(ga, C), *bd = xdev_floats(be, C);
    xt x = xt_new(N, D, H, W, C), r = xt_new(N, D, H, W, C), y = xt_new(N, D, H, W, C); xh2d(r.p, rh, ne * 2);
    for (int rep = 0; rep < 2; rep++) { xh2d(x.p, xh, ne * 2); xgn(x, G, gd, bd, 1e-5f, &r, XA_LRELU, y); xd2h(rep ? y2 : y1, y.p, ne * 2); }
    if (xnet_check("test_xmx gn")) return 1;
    double emax = 0; size_t bad = 0; const int cpg = C / G;
    for (int n = 0; n < N; n++) for (int g = 0; g < G; g++) {
        double s = 0, q = 0;
        for (size_t v = 0; v < S; v++) for (int c = g * cpg; c < (g + 1) * cpg; c++) { const double f = h2f(xh[((size_t)n * S + v) * C + c]); s += f; q += f * f; }
        const double m = s / ((double)S * cpg), rs = 1.0 / sqrt(q / ((double)S * cpg) - m * m + 1e-5);
        for (size_t v = 0; v < S; v++) for (int c = g * cpg; c < (g + 1) * cpg; c++) {
            const size_t i = ((size_t)n * S + v) * C + c; double t = (h2f(xh[i]) - m) * rs * ga[c] + be[c] + h2f(rh[i]); if (t < 0) t *= 0.01;
            const double d = fabs(h2f(y1[i]) - t); if (d > emax) emax = d; if (d > 2e-3 * fabs(t) + 2e-3) bad++;
        }
    }
    const int same = !memcmp(y1, y2, ne * 2);
    printf("gn N %d %dx%dx%d C %3d G %2d offset %g: max |d| %.3g, %zu of %zu over the bound, runs %s\n", N, D, H, W, C, G, off, emax, bad, ne, same ? "equal" : "DIFFER");
    xt_free(&x); xt_free(&r); xt_free(&y); xfree_dev(gd); xfree_dev(bd); free(xh); free(rh); free(y1); free(y2); free(ga); free(be);
    return bad != 0 || !same;
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
        fails += run(sf, 2, 9, 7, 11, 32, 32, 3, 1, 0);
        fails += run(sf, 1, 16, 16, 16, 64, 64, 3, 1, 0);
        fails += run(sf, 2, 10, 9, 8, 32, 64, 3, 2, 0);
        fails += run(sf, 1, 6, 5, 7, 128, 256, 1, 1, 0);
        fails += run(sf, 2, 4, 4, 4, 320, 320, 3, 1, 0);
        fails += run(sf, 1, 5, 6, 4, 96, 128, 3, 1, 0);
        fails += run(sf, 2, 5, 10, 40, 32, 32, 3, 1, 0);    /* halo kernel: TX 32 tiles with ragged edges */
        fails += run(sf, 1, 7, 9, 36, 128, 64, 3, 1, 0);    /* halo kernel: two chunk pairs, ragged */
        fails += run(sf, 1, 6, 6, 20, 64, 96, 3, 1, 0);     /* halo kernel: TX 16, outputs not a multiple of 32 */
        for (int mode = 1; mode <= 2; mode++) {            /* the least-squares weight scales (UFSM_XNET_MXQ) */
            fails += run(sf, 2, 9, 7, 11, 32, 32, 3, 1, mode);
            fails += run(sf, 1, 6, 5, 7, 128, 256, 1, 1, mode);
            fails += run(sf, 1, 7, 9, 36, 128, 64, 3, 1, mode);
        }
    }
    fails += run(0, 2, 7, 33, 29, 16, 16, 3, 1, 0);   /* fp16 igemm, 16 outputs (BN 16 tile; the Reader depth fusion) */
    fails += run(0, 1, 5, 20, 24, 32, 32, 3, 1, 0);
    g_rot = 1;   /* H16-rotated blocks (UFSM_XNET_MXROT) */
    for (int sf = 6; sf >= 5; sf--) { fails += run(sf, 2, 9, 7, 11, 32, 32, 3, 1, 2); fails += run(sf, 1, 7, 9, 36, 128, 64, 3, 1, 0); fails += run(sf, 1, 6, 5, 7, 128, 256, 1, 1, 2); }
    g_mxa = 1;   /* + least-squares activation scales (UFSM_XNET_MXA) */
    for (int sf = 6; sf >= 5; sf--) { fails += run(sf, 2, 9, 7, 11, 32, 32, 3, 1, 2); fails += run(sf, 1, 7, 9, 36, 128, 64, 3, 1, 0); }
    g_rot = 0;
    for (int sf = 6; sf >= 5; sf--) { fails += run(sf, 2, 9, 7, 11, 32, 32, 3, 1, 2); fails += run(sf, 1, 6, 5, 7, 128, 256, 1, 1, 0); }
    g_mxa = 0; xnet_set_mxq(0); xnet_set_mxrot(0); xnet_set_mxa(0);
    fails += run_gn(2, 17, 64, 48, 16, 16, 0.f);     /* the depth-fusion shape (InstanceNorm: G = C) */
    fails += run_gn(3, 1, 40, 36, 32, 32, 20.f);     /* mean >> std: the shifted sums */
    fails += run_gn(2, 1, 12, 12, 320, 32, 1.f);     /* deep level, 10 channels a group */
    fails += run_gn(1, 1, 5, 7, 640, 32, 1.f);       /* more pairs than threads */
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
