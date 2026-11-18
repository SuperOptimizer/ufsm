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

/* xdepth_collapse16 (fused GN apply + head + depth softmax / max) against the separate xgn + xhead + xdepth_attn_max */
static int run_dattn(int N, int D, int H, int W) {
    const size_t S = (size_t)D * H * W, ne = (size_t)N * S * 16, no = (size_t)N * H * W * 32;
    uint16_t *fh = malloc(ne * 2), *y1 = malloc(no * 2), *y2 = malloc(no * 2); float ga[16], be[16], w[16];
    for (size_t i = 0; i < ne; i++) fh[i] = f2h(rnd() * 3.f + 0.5f * (float)(i % 16) / 16);
    for (int c = 0; c < 16; c++) { ga[c] = 1.f + 0.5f * rnd(); be[c] = 0.3f * rnd(); w[c] = rnd(); }
    float *gd = xdev_floats(ga, 16), *bd = xdev_floats(be, 16), *wd = xdev_floats(w, 16);
    xt f = xt_new(N, D, H, W, 16); xh2d(f.p, fh, ne * 2);
    xt z = xdepth_collapse16(f, 16, gd, bd, 1e-5f, XA_LRELU, wd, 0.2f); xd2h(y1, z.p, no * 2); xt_free(&z);
    xgn(f, 16, gd, bd, 1e-5f, nullptr, XA_LRELU, f); float *at = xmalloc_dev(N * S * 4); xhead(f, wd, 0.2f, at);
    z = xdepth_attn_max(f, at); xd2h(y2, z.p, no * 2); xt_free(&z); xfree_dev(at);
    if (xnet_check("test_xmx dattn")) return 1;
    double emax = 0; size_t bad = 0;
    for (size_t i = 0; i < no; i++) { const double a = h2f(y1[i]), b = h2f(y2[i]), d = fabs(a - b); if (d > emax) emax = d; if (d > 2e-3 * fabs(b) + 1e-3) bad++; }
    printf("dattn N %d D %d %dx%d: max |d| %.3g vs the separate kernels, %zu of %zu over the bound\n", N, D, H, W, emax, bad, no);
    xt_free(&f); xfree_dev(gd); xfree_dev(bd); xfree_dev(wd); free(fh); free(y1); free(y2);
    return bad != 0;
}

/* xconv_fwd_gnin (GroupNorm + LReLU applied in the halo conv's load) against xgn then xconv_fwd */
static int run_gnin(int N, int D, int H, int W) {
    const size_t ne = (size_t)N * D * H * W * 16; uint16_t *fh = malloc(ne * 2), *y1 = malloc(ne * 2), *y2 = malloc(ne * 2); float ga[16], be[16], *w = malloc(16 * 16 * 27 * 4);
    for (size_t i = 0; i < ne; i++) fh[i] = f2h(rnd() * 2.f + 1.f);
    for (int c = 0; c < 16; c++) { ga[c] = 1.f + 0.5f * rnd(); be[c] = 0.3f * rnd(); }
    for (int i = 0; i < 16 * 16 * 27; i++) w[i] = rnd() * 0.1f;
    xnet_set_mx(0); xconv c; xconv_init(&c, w, 16, 16, 3, 3, 3, 1, 1, 1, 1, 1, 1, nullptr, nullptr);
    float *gd = xdev_floats(ga, 16), *bd = xdev_floats(be, 16); xt f = xt_new(N, D, H, W, 16); xh2d(f.p, fh, ne * 2);
    float *g = xgn_stats(f, 16, 1e-5f); xt y = xconv_fwd_gnin(&c, f, g, g + N * 16, gd, bd, XA_LRELU); xd2h(y1, y.p, ne * 2); xt_free(&y); xfree_dev(g);
    xgn(f, 16, gd, bd, 1e-5f, nullptr, XA_LRELU, f); y = xconv_fwd(&c, f, nullptr, XA_NONE); xd2h(y2, y.p, ne * 2); xt_free(&y);
    if (xnet_check("test_xmx gnin")) return 1;
    double emax = 0; size_t bad = 0;
    for (size_t i = 0; i < ne; i++) { const double a = h2f(y1[i]), b = h2f(y2[i]), d = fabs(a - b); if (d > emax) emax = d; if (d > 2e-3 * fabs(b) + 1e-3) bad++; }
    printf("gnin N %d %dx%dx%d: max |d| %.3g vs xgn + conv, %zu of %zu over the bound\n", N, D, H, W, emax, bad, ne);
    xt_free(&f); xconv_free(&c); xfree_dev(gd); xfree_dev(bd); free(fh); free(y1); free(y2); free(w);
    return bad != 0;
}

/* k_dfc16s (UFSM_XNET_DF16 10..16, depth-streaming; 10 the default) against k_dfc16 (1): byte-identical, plain and with the GroupNorm applied on load */
static int run_dfs(int N, int D, int H, int W) {
    const size_t ne = (size_t)N * D * H * W * 16; uint16_t *fh = malloc(ne * 2), *y0 = malloc(ne * 2), *y1 = malloc(ne * 2); float ga[16], be[16], *w = malloc(16 * 16 * 27 * 4);
    for (size_t i = 0; i < ne; i++) fh[i] = f2h(rnd() * 2.f + 0.5f);
    for (int c = 0; c < 16; c++) { ga[c] = 1.f + 0.5f * rnd(); be[c] = 0.3f * rnd(); }
    for (int i = 0; i < 16 * 16 * 27; i++) w[i] = rnd() * 0.1f;
    xnet_set_mx(0); xconv c; xconv_init(&c, w, 16, 16, 3, 3, 3, 1, 1, 1, 1, 1, 1, nullptr, nullptr);
    float *gd = xdev_floats(ga, 16), *bd = xdev_floats(be, 16); xt f = xt_new(N, D, H, W, 16); xh2d(f.p, fh, ne * 2); float *g = xgn_stats(f, 16, 1e-5f);
    int bad = 0;
    for (int gn = 0; gn < 2; gn++)
        for (int v = 10; v <= 16; v++) {
            xt y;
            xnet_set_df16(1); y = gn ? xconv_fwd_gnin(&c, f, g, g + N * 16, gd, bd, XA_LRELU) : xconv_fwd(&c, f, nullptr, XA_NONE); xd2h(y0, y.p, ne * 2); xt_free(&y);
            xnet_set_df16(v); y = gn ? xconv_fwd_gnin(&c, f, g, g + N * 16, gd, bd, XA_LRELU) : xconv_fwd(&c, f, nullptr, XA_NONE); xd2h(y1, y.p, ne * 2); xt_free(&y);
            if (xnet_check("test_xmx dfs")) return 1;
            size_t nd = 0; for (size_t i = 0; i < ne; i++) nd += y0[i] != y1[i];
            printf("dfs N %d %dx%dx%d gn %d variant %d: %zu of %zu differ\n", N, D, H, W, gn, v, nd, ne); bad += nd != 0;
        }
    xnet_set_df16(10); xfree_dev(g); xt_free(&f); xconv_free(&c); xfree_dev(gd); xfree_dev(bd); free(fh); free(y0); free(y1); free(w);
    return bad != 0;
}

/* the GroupNorm statistics are batch-independent: each sample's gm / rs bit-equal whether it is computed alone or inside a batch (any
   size, any position), and within 1e-5 (relative) of the old batch-dependent chunking (UFSM_XNET_GNOLD=1) */
static int run_gnx(int N, int D, int H, int W, int C, int G, float off) {
    const size_t S = (size_t)D * H * W, ne = (size_t)N * S * C; uint16_t *xh = malloc(ne * 2); int bad = 0;
    for (size_t i = 0; i < ne; i++) xh[i] = f2h(off * (1.f + 0.1f * (float)(i % C) / C) + rnd() * 2.f);
    xt x = xt_new(N, D, H, W, C); xh2d(x.p, xh, ne * 2);
    float *gb = xgn_stats(x, G, 1e-5f), *hb = malloc((size_t)N * C * 2 * 4), *h1 = malloc((size_t)C * 2 * 4), *ho = malloc((size_t)N * C * 2 * 4); xd2h(hb, gb, (size_t)N * C * 2 * 4); xfree_dev(gb);
    xnet_set_gnold(1); gb = xgn_stats(x, G, 1e-5f); xd2h(ho, gb, (size_t)N * C * 2 * 4); xfree_dev(gb); xnet_set_gnold(0);
    double dmax = 0;
    for (size_t i = 0; i < (size_t)N * C * 2; i++) { const double d = fabs((double)hb[i] - ho[i]) / (fabs((double)ho[i]) + 1e-3); if (d > dmax) dmax = d; }
    size_t nd = 0;
    for (int n = 0; n < N; n++) {   /* sample n alone */
        xt xs = {(char *)x.p + (size_t)n * S * C * 2, 1, D, H, W, C, nullptr}; float *g1 = xgn_stats(xs, G, 1e-5f); xd2h(h1, g1, (size_t)C * 2 * 4); xfree_dev(g1);
        for (int c = 0; c < C; c++) { nd += memcmp(&h1[c], &hb[(size_t)n * C + c], 4) != 0; nd += memcmp(&h1[C + c], &hb[(size_t)N * C + (size_t)n * C + c], 4) != 0; }
    }
    if (xnet_check("test_xmx gnx")) return 1;
    printf("gnx N %d %dx%dx%d C %3d G %2d offset %g: %zu of %d stats differ alone vs in the batch; max rel diff vs the old path %.2g\n", N, D, H, W, C, G, off, nd, N * C * 2, dmax);
    bad = nd != 0 || dmax > 1e-5;
    xt_free(&x); free(xh); free(hb); free(h1); free(ho);
    return bad;
}

/* the 1 -> 16 stem conv (k_dfc1) against a double host reference with fp16 weights */
static int run_c1(int N, int D, int H, int W) {
    const size_t nx = (size_t)N * D * H * W; uint16_t *xh = malloc(nx * 2), *yh = malloc(nx * 32); float *x = malloc(nx * 4), *w = malloc(16 * 27 * 4), b[16];
    for (size_t i = 0; i < nx; i++) { xh[i] = f2h(rnd() * 3.f); x[i] = h2f(xh[i]); }
    for (int i = 0; i < 16 * 27; i++) w[i] = rnd() * 0.3f;
    for (int c = 0; c < 16; c++) b[c] = rnd();
    xnet_set_mx(0); xconv c; xconv_init(&c, w, 16, 1, 3, 3, 3, 1, 1, 1, 1, 1, 1, nullptr, b);
    xt xd = xt_new(N, D, H, W, 1); xh2d(xd.p, xh, nx * 2); xt y = xconv_fwd(&c, xd, nullptr, XA_NONE); xd2h(yh, y.p, nx * 32);
    if (xnet_check("test_xmx c1")) return 1;
    double emax = 0; size_t bad = 0;
    for (int n = 0; n < N; n++) for (int z = 0; z < D; z++) for (int yy = 0; yy < H; yy++) for (int xx = 0; xx < W; xx++) for (int o = 0; o < 16; o++) {
        double s = b[o];
        for (int a = 0; a < 3; a++) for (int bb = 0; bb < 3; bb++) for (int e = 0; e < 3; e++) {
            const int zi = z + a - 1, yi = yy + bb - 1, xi = xx + e - 1; if (zi < 0 || zi >= D || yi < 0 || yi >= H || xi < 0 || xi >= W) continue;
            s += (double)x[(((size_t)n * D + zi) * H + yi) * W + xi] * h2f(f2h(w[o * 27 + (a * 3 + bb) * 3 + e]));
        }
        const double g = h2f(yh[((((size_t)n * D + z) * H + yy) * W + xx) * 16 + o]), d = fabs(g - s); if (d > emax) emax = d; if (d > 1e-3 * fabs(s) + 2e-3) bad++;
    }
    printf("c1 N %d %dx%dx%d: max |d| %.3g, %zu of %zu over the bound\n", N, D, H, W, emax, bad, nx * 16);
    uint16_t *y0 = malloc(nx * 32);   /* k_dfc1 (1) against the x-blocked k_dfc1x (2, 4): byte-equal */
    for (int v = 1; v <= 4; v++) {
        if (v == 3) continue;
        xnet_set_df1(v); xt yv = xconv_fwd(&c, xd, nullptr, XA_NONE); xd2h(v == 1 ? y0 : yh, yv.p, nx * 32); xt_free(&yv);
        if (v > 1) { size_t nd = 0; for (size_t i = 0; i < nx * 16; i++) nd += y0[i] != yh[i]; printf("c1 variant %d: %zu of %zu differ from k_dfc1\n", v, nd, nx * 16); bad += nd; }
    }
    xnet_set_df1(-1); free(y0);
    xt_free(&xd); xt_free(&y); xconv_free(&c); free(xh); free(yh); free(x); free(w);
    return bad != 0;
}

/* xgn_q (k_gnq: the GroupNorm apply writing the MX copy of its output) against xgn then xq_ensure (k_gn_apply, k_xq): y and q byte-equal */
static int run_gnq(int sf, int N, int H, int W, int C, int G, int withres, int act) {
    const size_t ne = (size_t)N * H * W * C; uint16_t *xh = malloc(ne * 2), *rh = malloc(ne * 2); float ga[640], be[640]; int bad = 0;
    for (size_t i = 0; i < ne; i++) { xh[i] = f2h(rnd() * 3.f + 0.2f * (float)(i % C) / C); rh[i] = f2h(rnd()); }
    for (int c = 0; c < C; c++) { ga[c] = 1.f + 0.5f * rnd(); be[c] = 0.3f * rnd(); }
    float *gd = xdev_floats(ga, C), *bd = xdev_floats(be, C);
    xnet_set_mx(sf); xnet_set_gnq(1);
    xt x = xt_new(N, 1, H, W, C), r = xt_new(N, 1, H, W, C); xh2d(r.p, rh, ne * 2);
    xh2d(x.p, xh, ne * 2); xt a = xt_new(N, 1, H, W, C); xgn(x, G, gd, bd, 1e-5f, withres ? &r : nullptr, act, a); xq_ensure(&a);
    xt b = x; b.q = nullptr; xh2d(x.p, xh, ne * 2); xgn_q(x, G, gd, bd, 1e-5f, withres ? &r : nullptr, act, 1, &b);   /* in place, as reader3 */
    const size_t nq = (ne / 2 + 15) / 16 * 16 + ne / 8;   /* nibbles, then 4 bytes a 32-channel chunk of which 2 written: compare the written ones */
    uint8_t *qa = malloc(nq + 64), *qb = malloc(nq + 64); uint16_t *ya = malloc(ne * 2), *yb = malloc(ne * 2);
    xd2h(qa, a.q, nq); xd2h(qb, b.q, nq); xd2h(ya, a.p, ne * 2); xd2h(yb, b.p, ne * 2);
    if (xnet_check("test_xmx gnq")) return 1;
    size_t dy = 0, dq = 0; const size_t so = (ne / 2 + 15) / 16 * 16;
    for (size_t i = 0; i < ne; i++) dy += ya[i] != yb[i];
    for (size_t i = 0; i < so; i++) dq += qa[i] != qb[i];
    for (size_t k = 0; k < ne / 32; k++) for (int j = 0; j < 2; j++) dq += qa[so + k * 4 + j] != qb[so + k * 4 + j];
    printf("gnq sf %d N %d %dx%d C %d G %d res %d act %d: %zu of %zu y, %zu q bytes differ from xgn + k_xq\n", sf, N, H, W, C, G, withres, act, dy, ne, dq);
    bad = dy != 0 || dq != 0;
    xnet_set_gnq(-1); xnet_set_mx(0);
    xt_free(&a); xt_free(&b); xt_free(&r); xfree_dev(gd); xfree_dev(bd); free(xh); free(rh); free(qa); free(qb); free(ya); free(yb);
    return bad;
}
/* the 2-D transposed conv (k2 s2) shuffle: k_tshuffle8 (8 channels a thread, the default) byte-equal to the per-element k_tshuffle */
static int run_ts(int N, int D, int H, int W, int ci, int co) {
    const size_t nx = (size_t)N * D * H * W * ci, ny = (size_t)N * D * 4 * H * W * co; uint16_t *xh = malloc(nx * 2), *y0 = malloc(ny * 2), *y1 = malloc(ny * 2);
    float *w = malloc((size_t)ci * co * 4 * 4), *b = malloc((size_t)co * 4);
    for (size_t i = 0; i < nx; i++) xh[i] = f2h(rnd() * 3.f);
    for (size_t i = 0; i < (size_t)ci * co * 4; i++) w[i] = rnd() * 0.1f;
    for (int c = 0; c < co; c++) b[c] = rnd();
    xnet_set_mx(0); xconv c; xtconv_init(&c, w, ci, co, b); xt xd = xt_new(N, D, H, W, ci); xh2d(xd.p, xh, nx * 2);
    xnet_set_ts8(0); xt ya = xconv_fwd(&c, xd, nullptr, XA_NONE); xd2h(y0, ya.p, ny * 2);
    xnet_set_ts8(1); xt yb = xconv_fwd(&c, xd, nullptr, XA_NONE); xd2h(y1, yb.p, ny * 2); xnet_set_ts8(-1);
    if (xnet_check("test_xmx ts")) return 1;
    size_t nd = 0; for (size_t i = 0; i < ny; i++) nd += y0[i] != y1[i];
    printf("ts N %d %dx%dx%d %d -> %d: %zu of %zu differ from k_tshuffle\n", N, D, H, W, ci, co, nd, ny);
    xt_free(&xd); xt_free(&ya); xt_free(&yb); xconv_free(&c); free(xh); free(y0); free(y1); free(w); free(b);
    return nd != 0;
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
    if (getenv("XMX_IGBENCH") || getenv("XMX_IGTEST")) {   /* the Reader's 2-D 3x3 convs (N 8, d 1): UFSM_XNET_CFG 5..8 forced tiles vs the wave pick (0); byte-equal */
        static const int L[][3] = {{48, 256, 256}, {24, 320, 320}, {12, 320, 320}, {96, 128, 128}, {24, 640, 320}, {12, 320, 1280}, {20, 96, 160}};
        const int bench = getenv("XMX_IGBENCH") != nullptr; int bad = 0; xnet_set_mx(0);
        for (size_t li = 0; li < sizeof L / sizeof L[0]; li++) {
            const int Hs = L[li][0], ci = L[li][1], co = L[li][2], k = co == 1280 ? 1 : 3, N = 8;
            float *w = calloc((size_t)co * ci * k * k, 4); for (size_t j = 0; j < (size_t)co * ci * k * k; j++) w[j] = rnd() * 0.05f;
            xconv c; xconv_init(&c, w, co, ci, 1, k, k, 1, 1, 1, 0, k / 2, k / 2, nullptr, nullptr);
            const size_t nx = (size_t)N * Hs * Hs * ci, ny = (size_t)N * Hs * Hs * co; uint16_t *xh = malloc(nx * 2), *y0 = malloc(ny * 2), *y1 = malloc(ny * 2);
            for (size_t j = 0; j < nx; j++) xh[j] = f2h(rnd()); xt x = xt_new(N, 1, Hs, Hs, ci); xh2d(x.p, xh, nx * 2);
            printf("ig %3d^2 ci %4d co %4d:", Hs, ci, co);
            for (int v = 5; v <= 9; v++) {
                const int cfg = v == 9 ? 0 : v; xnet_set_cfg(cfg); double best = 1e9;
                for (int r = 0; r < (bench ? 10 : 1); r++) { xnet_sync(); const double t0 = now(); xt y = xconv_fwd(&c, x, nullptr, XA_NONE); xnet_sync(); const double t = now() - t0; if (t < best) best = t; xd2h(v == 5 ? y0 : y1, y.p, ny * 2); xt_free(&y); }
                size_t nd = 0; if (v > 5) for (size_t j = 0; j < ny; j++) nd += y0[j] != y1[j];
                bad += nd != 0; printf("  cfg %d %.3f ms%s", cfg, best * 1e3, nd ? " DIFFERS" : "");
            }
            printf("\n"); xnet_set_cfg(0); xt_free(&x); xconv_free(&c); free(w); free(xh); free(y0); free(y1);
        }
        printf("%s\n", bad ? "FAIL" : "PASS"); return bad != 0;
    }
    if (getenv("XMX_DFBENCH")) {   /* the Reader depth-fusion conv, 4 x 17 x 384^2 x 16: implicit GEMM (0) vs the halo-kernel variants; min of 10 */
        xnet_set_mx(0); float *w = calloc(16 * 16 * 27, 4); for (int j = 0; j < 16 * 16 * 27; j++) w[j] = rnd() * 0.05f;
        xconv c; xconv_init(&c, w, 16, 16, 3, 3, 3, 1, 1, 1, 1, 1, 1, nullptr, nullptr); xt x = xt_new(4, 17, 384, 384, 16);
        for (int v = 0; v <= 16; v++) {
            xnet_set_df16(v); double best = 1e9;
            for (int r = 0; r < 10; r++) { xnet_sync(); const double t0 = now(); xt y = xconv_fwd(&c, x, nullptr, XA_NONE); xnet_sync(); const double t = now() - t0; xt_free(&y); if (t < best) best = t; }
            printf("dfbench variant %d: %.3f ms\n", v, best * 1e3);
        }
        float ga[16], be[16]; for (int k = 0; k < 16; k++) { ga[k] = 1.f; be[k] = 0.f; } float *gd = xdev_floats(ga, 16), *bd = xdev_floats(be, 16), *g = xgn_stats(x, 16, 1e-5f);
        for (int v = 1; v <= 16; v += v == 1 ? 9 : 1) {   /* with the GroupNorm applied on load (the Reader's f2) */
            xnet_set_df16(v); double best = 1e9;
            for (int r = 0; r < 10; r++) { xnet_sync(); const double t0 = now(); xt y = xconv_fwd_gnin(&c, x, g, g + 4 * 16, gd, bd, XA_LRELU); xnet_sync(); const double t = now() - t0; xt_free(&y); if (t < best) best = t; }
            printf("dfbench gnin variant %d: %.3f ms\n", v, best * 1e3);
        }
        { float w1[16 * 27]; for (int j = 0; j < 16 * 27; j++) w1[j] = rnd() * 0.3f; xconv c1; xconv_init(&c1, w1, 16, 1, 3, 3, 3, 1, 1, 1, 1, 1, 1, nullptr, nullptr); xt x1 = xt_new(4, 17, 384, 384, 1);
          for (int v = 1; v <= 4; v++) { if (v == 3) continue; xnet_set_df1(v); double best = 1e9;
              for (int r = 0; r < 10; r++) { xnet_sync(); const double t0 = now(); xt y = xconv_fwd(&c1, x1, nullptr, XA_NONE); xnet_sync(); const double t = now() - t0; xt_free(&y); if (t < best) best = t; }
              printf("dfbench c1 variant %d: %.3f ms\n", v, best * 1e3); }
          xnet_set_df1(-1); xt_free(&x1); xconv_free(&c1); }
        return 0;
    }
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
    fails += run(0, 2, 7, 33, 29, 16, 16, 3, 1, 0);   /* fp16 16 -> 16 3x3x3: the halo kernel k_dfc16 (UFSM_XNET_DF16=0: the BN 16 igemm tile) */
    fails += run(0, 1, 17, 20, 64, 16, 16, 3, 1, 0);  /* the Reader depth-fusion shape (17 layers), ragged z / y tiles */
    fails += run(0, 1, 5, 20, 24, 32, 32, 3, 1, 0);
    g_rot = 1;   /* H16-rotated blocks (UFSM_XNET_MXROT) */
    for (int sf = 6; sf >= 5; sf--) { fails += run(sf, 2, 9, 7, 11, 32, 32, 3, 1, 2); fails += run(sf, 1, 7, 9, 36, 128, 64, 3, 1, 0); fails += run(sf, 1, 6, 5, 7, 128, 256, 1, 1, 2); }
    g_mxa = 1;   /* + least-squares activation scales (UFSM_XNET_MXA) */
    for (int sf = 6; sf >= 5; sf--) { fails += run(sf, 2, 9, 7, 11, 32, 32, 3, 1, 2); fails += run(sf, 1, 7, 9, 36, 128, 64, 3, 1, 0); }
    g_rot = 0;
    for (int sf = 6; sf >= 5; sf--) { fails += run(sf, 2, 9, 7, 11, 32, 32, 3, 1, 2); fails += run(sf, 1, 6, 5, 7, 128, 256, 1, 1, 0); }
    g_mxa = 0; xnet_set_mxq(0); xnet_set_mxrot(0); xnet_set_mxa(0);
    fails += run_dattn(2, 17, 40, 36);
    for (int sf = 5; sf <= 6; sf++) { fails += run_gnq(sf, 2, 13, 11, 64, 32, 0, XA_LRELU); fails += run_gnq(sf, 3, 6, 5, 320, 32, 1, XA_LRELU); fails += run_gnq(sf, 1, 9, 7, 32, 32, 1, XA_RELU); }
    fails += run_ts(2, 1, 7, 9, 32, 16);     /* ragged, two samples */
    fails += run_ts(1, 1, 24, 24, 64, 8);
    fails += run_ts(3, 1, 5, 13, 16, 64);
    fails += run_gnin(2, 17, 20, 70);
    fails += run_c1(2, 17, 30, 41);
    fails += run_c1(1, 3, 9, 23);     /* W % 4 ragged */
    fails += run_dfs(2, 17, 20, 70);   /* ragged y / x tiles */
    fails += run_dfs(1, 3, 37, 33);
    fails += run_dfs(1, 1, 9, 16);
    fails += run_gn(2, 17, 64, 48, 16, 16, 0.f);     /* the depth-fusion shape (InstanceNorm: G = C) */
    fails += run_gn(3, 1, 40, 36, 32, 32, 20.f);     /* mean >> std: the shifted sums */
    fails += run_gn(2, 1, 12, 12, 320, 32, 1.f);     /* deep level, 10 channels a group */
    fails += run_gn(1, 1, 5, 7, 640, 32, 1.f);       /* more pairs than threads */
    fails += run_gnx(3, 17, 40, 36, 16, 16, 0.f);
    fails += run_gnx(5, 1, 24, 24, 320, 32, 20.f);
    fails += run_gnx(2, 1, 96, 96, 128, 32, 1.f);
    fails += run_gnx(4, 1, 12, 12, 640, 32, 1.f);
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
