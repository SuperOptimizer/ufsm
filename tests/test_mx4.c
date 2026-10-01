/* MX-fp4 (packed e2m1 + ue8m0 per 32) storage: format round trips, stochastic rounding, NaN policy, and the ops that read
   or write mx4 tensors against the exact fp32 kernels. As test_mx: every MX input is dequantised and fed to the fp32
   reference, so the error is the op's own compute / output quantisation (fp4 compute ~ amax/24 per element -> 0.2;
   dequantise-only ops 0.08; a layout bug shows as ~100%). */
#include "nn.h"
#include "nn_lp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint64_t rs = 12345;
static float frand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 11) * 0x1.0p-53 * 2 - 1); }
static float nrand(void) { float u = 0.5f * frand() + 0.5f + 1e-7f, v = 0.5f * frand() + 0.5f; return sqrtf(-2.f * logf(u)) * cosf(6.2831853f * v); }
static int bad;
static float *dev_rand(size_t n, float scale) { float *h = malloc(n * 4); for (size_t i = 0; i < n; i++) h[i] = frand() * scale; float *d = nn_malloc(n * 4); nn_h2d(d, h, n * 4); free(h); return d; }
static float *dev_zero(size_t n) { float *d = nn_malloc(n * 4); nn_zero(d, n * 4); return d; }
static void *mx4_new(shape5 s) { size_t b = nn_mx4_bytes(s); void *p = nn_malloc(b); nn_zero(p, b); nn_set_storage(p, b, 4); return p; }
static void *mx4_from(const float *x, shape5 s) { void *p = mx4_new(s); lp_f32_to_mx4(x, s.n, s.c, shape_spatial(s), p); return p; }
static float *deq4(const void *p, shape5 s) { float *d = dev_zero(shape_numel(s)); lp_mx4_to_f32(p, s.n, s.c, shape_spatial(s), d); return d; }
static void *mx8_new(shape5 s) { size_t b = nn_mx8_bytes(s); void *p = nn_malloc(b); nn_zero(p, b); nn_set_storage(p, b, 8); return p; }
static void *mx8_from(const float *x, shape5 s) { void *p = mx8_new(s); lp_f32_to_mx8(x, s.n, s.c, shape_spatial(s), p); return p; }
static float *deq8(const void *p, shape5 s) { float *d = dev_zero(shape_numel(s)); lp_mx8_to_f32(p, s.n, s.c, shape_spatial(s), d); return d; }
static double cmp(const char *what, const float *a, const float *r, size_t n, double tol) {
    float *ha = malloc(n * 4), *hr = malloc(n * 4);
    nn_d2h(ha, a, n * 4); nn_d2h(hr, r, n * 4);
    double d2 = 0, r2 = 0;
    for (size_t i = 0; i < n; i++) { double d = (double)ha[i] - hr[i]; d2 += d * d; r2 += (double)hr[i] * hr[i]; }
    double e = sqrt(d2 / (r2 > 0 ? r2 : 1e-300));
    printf("  %-48s rel err %.3g%s\n", what, e, e < tol ? "" : "  FAIL");
    if (!(e < tol)) bad++;
    free(ha); free(hr);
    return e;
}
static void check(const char *what, int ok) { printf("  %-48s %s\n", what, ok ? "ok" : "FAIL"); if (!ok) bad++; }
/* host reference of the mx4 quantiser: per (n, block, voxel) amax over the block, e = ceil(log2(amax / 6)) clamped to
   [-126, 126], round-to-nearest-even onto {0, .5, 1, 1.5, 2, 3, 4, 6} * 2^e */
static float e2m1_rne(float a) {   /* a >= 0 in grid units */
    static const float g[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    if (a >= 6.f) return 6.f;
    int i = 0; while (i < 7 && g[i + 1] <= a) i++;
    float lo = g[i], hi = g[i + 1], d = a - lo, u = hi - a;
    if (d < u) return lo; if (u < d) return hi;
    return (i & 1) ? hi : lo;   /* tie: even code */
}
static void host_mx4(const float *x, shape5 s, float *y) {
    const int bw = s.c <= 16 ? 16 : 32, nb = (s.c + bw - 1) / bw; const size_t S = shape_spatial(s);
    for (int n = 0; n < s.n; n++) for (int b = 0; b < nb; b++) for (size_t v = 0; v < S; v++) {
        float am = 0.f;
        for (int k = 0; k < bw; k++) { int c = b * bw + k; if (c < s.c) am = fmaxf(am, fabsf(x[((size_t)n * s.c + c) * S + v])); }
        int e; if (am == 0.f) e = -126; else { e = (int)ceilf(log2f(am / 6.f)); float t = ldexpf(am, -e); if (t > 6.f) e++; if (e < -126) e = -126; if (e > 126) e = 126; }
        for (int k = 0; k < bw; k++) { int c = b * bw + k; if (c >= s.c) continue; size_t i = ((size_t)n * s.c + c) * S + v; float a = fabsf(x[i]) * ldexpf(1.f, -e); y[i] = copysignf(e2m1_rne(a) * ldexpf(1.f, e), x[i]); }
    }
}
/* host reference of prep_w4_k: w[co][ci][27] snapped to the e2m1 grid with one ue8m0 per (tap, co, 32 ci); a kernel that
   requantises the snapped weights is lossless, so the fp4 kernel on exactly representable inputs must match fp32 */
static void host_w4(float *w, int Co, int Ci) {
    for (int t = 0; t < 27; t++) for (int co = 0; co < Co; co++) for (int c0 = 0; c0 < Ci; c0 += 32) {
        float am = 0.f;
        for (int ci = c0; ci < c0 + 32 && ci < Ci; ci++) am = fmaxf(am, fabsf(w[((size_t)co * Ci + ci) * 27 + t]));
        int e; if (am == 0.f) e = -126; else { e = (int)ceilf(log2f(am / 6.f)); float tt = ldexpf(am, -e); if (tt > 6.f) e++; if (e < -126) e = -126; if (e > 126) e = 126; }
        for (int ci = c0; ci < c0 + 32 && ci < Ci; ci++) { size_t i = ((size_t)co * Ci + ci) * 27 + t; w[i] = copysignf(e2m1_rne(fabsf(w[i]) * ldexpf(1.f, -e)) * ldexpf(1.f, e), w[i]); }
    }
}
static void mode_ref(void) { nn_set_tf32(0); }
static void mode_mx(void) { nn_set_prec(3); }
int main(void) {
    nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0);
    nn_set_act_bf16(0); nn_set_grad_bf16(0);   /* non-MX operands are fp32 */
    /* TOL4: fp4 compute with mx4 storage = three e2m1 quantisations (input, weights, output; ~0.13 each on these inputs ->
       ~0.2 combined); the exact tests below separate kernel correctness from this format error. */
    const double TOL4 = 0.25, TOL = 0.08;
    const int N = 2, G = 8;
    {   /* e2m1 storage round trip of post-GN-SiLU-like data (|x| up to ~4): quantisation error of the format alone */
        shape5 s = {N, 32, 8, 8, 8}, s16 = {N, 16, 8, 8, 8};
        size_t n = shape_numel(s), n16 = shape_numel(s16);
        float *h = malloc(n * 4);
        for (size_t i = 0; i < n; i++) { float v = nrand(); h[i] = v / (1.f + expf(-v)); }   /* silu of N(0, 1) */
        float *x = nn_malloc(n * 4); nn_h2d(x, h, n * 4);
        float *ref = malloc(n * 4), *dref = nn_malloc(n * 4);
        host_mx4(h, s, ref); nn_h2d(dref, ref, n * 4);
        cmp("mx4 round trip silu(N(0,1)) == host quantiser", deq4(mx4_from(x, s), s), dref, n, 1e-6);
        cmp("mx4 round trip silu(N(0,1)), 32 ch, format error", deq4(mx4_from(x, s), s), x, n, 0.16);   /* e2m1 RN: ~0.13 on this data (the plan's 0.03 is below what the grid can do) */
        host_mx4(h, s16, ref); nn_h2d(dref, ref, n16 * 4);
        cmp("mx4 round trip, 16 ch (bw 16) == host quantiser", deq4(mx4_from(x, s16), s16), dref, n16, 1e-6);
        cmp("mx8 round trip (unchanged helper)", deq8(mx8_from(x, s), s), x, n, 0.03);
        for (size_t i = 0; i < n; i++) h[i] = 4.f * nrand();
        nn_h2d(x, h, n * 4); host_mx4(h, s, ref); nn_h2d(dref, ref, n * 4);
        cmp("mx4 round trip 4 N(0,1) == host quantiser", deq4(mx4_from(x, s), s), dref, n, 1e-6);
        cmp("mx4 round trip 4 N(0,1), format error", deq4(mx4_from(x, s), s), x, n, 0.14);
        /* 16-bit source: fp16 rounding moves ~1% of the values across a rounding boundary */
        nn_set_f16(1); void *hx = nn_malloc(n * 2); nn_f32_to_h16(x, n, hx, 1.f);
        void *p4 = mx4_new(s); lp_h16_to_mx4(hx, 2, s.n, s.c, shape_spatial(s), p4);
        cmp("mx4 from fp16 vs mx4 from fp32", deq4(p4, s), deq4(mx4_from(x, s), s), n, 0.03);
        nn_set_f16(0);
        free(h); free(ref);
    }
    {   /* stochastic rounding: unbiased on the non-uniform grid */
        const float vals[] = {1.3f, 0.2f, 2.7f, 4.5f, 5.9f, -1.3f};
        for (int i = 0; i < 6; i++) {
            double m = lp_sr_e2m1_mean(vals[i], 4000000);   /* the mean of 4e6 roundings has sigma <= 2.5e-4 (step 1, p = 1/2) */
            char what[64]; snprintf(what, sizeof what, "sr_e2m1 mean of 4e6 roundings of %.1f (%.4f)", vals[i], m);
            check(what, fabs(m - vals[i]) < 1e-3);
        }
        check("sr_e2m1 clamps at 6", fabs(lp_sr_e2m1_mean(9.f, 1000) - 6.0) < 1e-9);
    }
    {   /* NaN policy: cvt.rn.satfinite.e2m1x2 of NaN / inf (documented), and a NaN in a block makes the decoded block non-finite */
        const float hv[4] = {NAN, INFINITY, -INFINITY, 7.f}; unsigned char ho[4];
        lp_cvt_e2m1_probe(hv, ho, 4);
        printf("  cvt.rn.satfinite.e2m1x2: NaN -> 0x%x, +inf -> 0x%x, -inf -> 0x%x, 7 -> 0x%x\n", ho[0], ho[1], ho[2], ho[3]);
        check("cvt saturates +inf to +6 (0x7), -inf to -6 (0xf)", ho[1] == 0x7 && ho[2] == 0xf);
        shape5 s = {1, 32, 2, 2, 2};
        size_t n = shape_numel(s);
        float *h = calloc(n, 4);
        for (size_t i = 0; i < n; i++) h[i] = 0.5f * (float)(i % 7);
        h[5 * 8 + 0] = NAN;   /* channel 5 of voxel 0 (index c * S + v) */
        float *x = nn_malloc(n * 4); nn_h2d(x, h, n * 4);
        float *d = deq4(mx4_from(x, s), s);
        float *hd = malloc(n * 4), *ref = malloc(n * 4); nn_d2h(hd, d, n * 4);
        h[5 * 8 + 0] = 0.f; host_mx4(h, s, ref);
        int nonfin = 0, other_ok = 1;
        for (int c = 0; c < 32; c++) { float v = hd[(size_t)c * 8]; if (!isfinite(v)) nonfin++; }
        for (size_t i = 0; i < n; i++) if (i % 8) { if (hd[i] != ref[i]) other_ok = 0; }
        check("NaN in a block: that voxel's block decodes non-finite", nonfin == 32 && !isfinite(hd[5 * 8]));
        check("NaN in a block: other voxels == host quantiser", other_ok);
        free(h); free(hd); free(ref);
    }
    {   /* fp4 conv forward on mx4 I/O: GN+SiLU input (32 ch, D = 16 -> TZ 4), output GN statistics from the unquantised accumulators */
        shape5 xs = {N, 32, 16, 12, 16}, ys = xs;
        size_t nx = shape_numel(xs), ny = shape_numel(ys), NG = (size_t)N * G;
        float *x = dev_rand(nx, 2.f), *w = dev_rand((size_t)32 * 32 * 27, 0.1f), *b = dev_rand(32, 0.1f);
        float *gam = dev_rand(32, 1.f), *bet = dev_rand(32, 0.5f), *mean = dev_rand(NG, 0.2f), *rstd = dev_rand(NG, 0.3f);
        { float h[64]; nn_d2h(h, rstd, NG * 4); for (size_t i = 0; i < NG; i++) h[i] = 0.8f + fabsf(h[i]); nn_h2d(rstd, h, NG * 4); }
        void *xm = mx4_from(x, xs); float *xd = deq4(xm, xs);
        float *yr = dev_zero(ny), *m1 = dev_zero(NG), *r1 = dev_zero(NG), *m2 = dev_zero(NG), *r2 = dev_zero(NG);
        void *ym = mx4_new(ys);
        mode_ref(); { float *t = dev_zero(nx); nn_gn_silu_apply(xd, xs, G, gam, bet, mean, rstd, t); nn_conv3d_fwd(t, xs, w, b, 32, 3, 1, yr); nn_gn_fwd(yr, ys, G, 1e-5f, nullptr, nullptr, nullptr, m1, r1); }
        mode_mx(); nn_conv3d_fwd_gn_stats(xm, xs, G, gam, bet, mean, rstd, w, b, 32, (float *)ym, G, 1e-5f, m2, r2);
        cmp("fp4 conv fwd gn+silu input (mx4 in/out, TZ 4)", deq4(ym, ys), yr, ny, TOL4);
        cmp("fp4 conv fwd output GN mean", m2, m1, NG, TOL4);
        cmp("fp4 conv fwd output GN rstd", r2, r1, NG, TOL4);
        /* exact check: mx4 input (copy staging, no requantisation), weights on the e2m1 grid (host snap), fp32 output: the
           kernel's only rounding is the fp32 accumulation order */
        {
            float *hw = malloc((size_t)32 * 32 * 27 * 4); nn_d2h(hw, w, (size_t)32 * 32 * 27 * 4); host_w4(hw, 32, 32);
            float *wq = nn_malloc((size_t)32 * 32 * 27 * 4); nn_h2d(wq, hw, (size_t)32 * 32 * 27 * 4); free(hw);
            float *yx = dev_zero(ny), *y4 = dev_zero(ny);
            mode_ref(); nn_conv3d_fwd(xd, xs, wq, b, 32, 3, 1, yx);
            mode_mx(); nn_conv3d_fwd(xm, xs, wq, b, 32, 3, 1, y4);
            cmp("fp4 conv fwd exact (mx4 in, grid weights, fp32 out)", y4, yx, ny, 2e-3);
            shape5 xo = {N, 32, 16, 12, 16}; shape5 yo = xo; yo.c = 64;   /* MT 4 / TZ 2 tile */
            float *w64 = dev_rand((size_t)64 * 32 * 27, 0.1f); hw = malloc((size_t)64 * 32 * 27 * 4); nn_d2h(hw, w64, (size_t)64 * 32 * 27 * 4); host_w4(hw, 64, 32); nn_h2d(w64, hw, (size_t)64 * 32 * 27 * 4); free(hw);
            float *yx2 = dev_zero(shape_numel(yo)), *y42 = dev_zero(shape_numel(yo));
            mode_ref(); nn_conv3d_fwd(xd, xo, w64, nullptr, 64, 3, 1, yx2);
            mode_mx(); nn_conv3d_fwd(xm, xo, w64, nullptr, 64, 3, 1, y42);
            cmp("fp4 conv fwd exact, 64 out (MT 4)", y42, yx2, shape_numel(yo), 2e-3);
        }
        /* untransformed mx4 input (copy staging) */
        float *yr2 = dev_zero(ny); void *ym2 = mx4_new(ys);
        mode_ref(); nn_conv3d_fwd(xd, xs, w, b, 32, 3, 1, yr2);
        mode_mx(); nn_conv3d_fwd(xm, xs, w, b, 32, 3, 1, ym2);
        cmp("fp4 conv fwd plain mx4 input (copy staging)", deq4(ym2, ys), yr2, ny, TOL4);
        /* the same on mx8 storage under prec 3 (the former dt-3 misroute): fp4 kernel with mx8 I/O */
        void *x8 = mx8_from(x, xs); float *x8d = deq8(x8, xs); float *yr3 = dev_zero(ny); void *y8 = mx8_new(ys);
        mode_ref(); { float *t = dev_zero(nx); nn_gn_silu_apply(x8d, xs, G, gam, bet, mean, rstd, t); nn_conv3d_fwd(t, xs, w, b, 32, 3, 1, yr3); }
        nn_gn_t gx = {gam, bet, mean, rstd, G};
        mode_mx(); nn_conv3d_fwd_x(x8, &gx, nullptr, nullptr, 0, 0, xs, w, b, 32, 3, 1, (float *)y8, 0, 1e-5f, nullptr, nullptr);
        cmp("fp4 conv fwd gn+silu input (mx8 in/out, prec 3)", deq8(y8, ys), yr3, ny, TOL4);
        /* memo: same conv id, weights changed in place without a step -> stale until nn_wmemo_clear */
        float *w2 = dev_rand((size_t)32 * 32 * 27, 0.1f), *yr4 = dev_zero(ny); void *ym4 = mx4_new(ys);
        nn_set_layer(1); nn_set_conv(0);
        mode_mx(); nn_conv3d_fwd(xm, xs, w, b, 32, 3, 1, ym4);
        nn_d2d(w, w2, (size_t)32 * 32 * 27 * 4);
        nn_conv3d_fwd(xm, xs, w, b, 32, 3, 1, ym4);
        double stale = cmp("fp4 weight memo: stale after in-place change (info)", deq4(ym4, ys), deq4(ym2, ys), ny, 1e9);
        nn_wmemo_clear(); nn_conv3d_fwd(xm, xs, w, b, 32, 3, 1, ym4);
        mode_ref(); nn_conv3d_fwd(xd, xs, w, b, 32, 3, 1, yr4);
        check("fp4 weight memo: hit while unchanged", stale < 1e-6);
        cmp("fp4 weight memo: refreshed by nn_wmemo_clear", deq4(ym4, ys), yr4, ny, TOL4);
        nn_set_layer(-1);
    }
    {   /* 16-channel mx4 input: the fp8 kernel with nibble staging and an mx4 output (gn+silu input, bw 16 rows) */
        shape5 xs = {N, 16, 12, 12, 16}, ys = xs; ys.c = 32;
        size_t nx = shape_numel(xs), ny = shape_numel(ys), NG = (size_t)N * G;
        float *x = dev_rand(nx, 2.f), *w = dev_rand((size_t)32 * 16 * 27, 0.1f), *b = dev_rand(32, 0.1f);
        float *gam = dev_rand(16, 1.f), *bet = dev_rand(16, 0.5f), *mean = dev_rand(NG, 0.2f), *rstd = dev_rand(NG, 0.3f);
        { float h[64]; nn_d2h(h, rstd, NG * 4); for (size_t i = 0; i < NG; i++) h[i] = 0.8f + fabsf(h[i]); nn_h2d(rstd, h, NG * 4); }
        void *xm = mx4_from(x, xs); float *xd = deq4(xm, xs);
        float *yr = dev_zero(ny); void *ym = mx4_new(ys);
        mode_ref(); { float *t = dev_zero(nx); nn_gn_silu_apply(xd, xs, G, gam, bet, mean, rstd, t); nn_conv3d_fwd(t, xs, w, b, 32, 3, 1, yr); }
        nn_gn_t gx = {gam, bet, mean, rstd, G};
        mode_mx(); nn_conv3d_fwd_x(xm, &gx, nullptr, nullptr, 0, 0, xs, w, b, 32, 3, 1, (float *)ym, 0, 1e-5f, nullptr, nullptr);
        cmp("fp8 conv fwd 16-ch mx4 input, mx4 output", deq4(ym, ys), yr, ny, TOL4);   /* fp8 compute + one mx4 output quantisation */
        /* split forward: x (32 ch) + x2 (16 ch) -> 16 ch, fp4 kernel */
        shape5 cs = xs; cs.c = 48; shape5 c1 = xs; c1.c = 32; shape5 o16 = xs; o16.c = 16;
        float *xa = dev_rand(shape_numel(c1), 1.f); void *xam = mx4_from(xa, c1); float *xad = deq4(xam, c1);
        float *w48 = dev_rand((size_t)16 * 48 * 27, 0.1f), *cat = dev_zero(shape_numel(cs)), *ysr = dev_zero(shape_numel(o16));
        void *ysm = mx4_new(o16);
        mode_ref(); nn_concat_fwd(xad, 32, xd, 16, cs, cat); nn_conv3d_fwd(cat, cs, w48, nullptr, 16, 3, 1, ysr);
        mode_mx(); nn_conv3d_fwd_split(xam, xm, 32, cs, 0, nullptr, nullptr, nullptr, nullptr, w48, nullptr, 16, (float *)ysm, 0, 1e-5f, nullptr, nullptr);
        cmp("fp4 conv fwd split input (32 + 16, mx4)", deq4(ysm, o16), ysr, shape_numel(o16), TOL4);
        /* split with gn+silu on both segments (the decoder's dec0.c1 without the fused upsample) */
        float *gam2 = dev_rand(32, 1.f), *bet2 = dev_rand(32, 0.5f), *mean2 = dev_rand(NG, 0.2f), *rstd2 = dev_rand(NG, 0.3f);
        { float h[64]; nn_d2h(h, rstd2, NG * 4); for (size_t i = 0; i < NG; i++) h[i] = 0.8f + fabsf(h[i]); nn_h2d(rstd2, h, NG * 4); }
        float *ysr2 = dev_zero(shape_numel(o16)); void *ysm2 = mx4_new(o16);
        mode_ref(); { float *ta = dev_zero(shape_numel(c1)), *tb = dev_zero(nx); nn_gn_silu_apply(xad, c1, G, gam2, bet2, mean2, rstd2, ta); nn_gn_silu_apply(xd, xs, G, gam, bet, mean, rstd, tb); nn_concat_fwd(ta, 32, tb, 16, cs, cat); nn_conv3d_fwd(cat, cs, w48, nullptr, 16, 3, 1, ysr2); }
        nn_gn_t ga = {gam2, bet2, mean2, rstd2, G}, gb = {gam, bet, mean, rstd, G};
        mode_mx(); nn_conv3d_fwd_x(xam, &ga, xm, &gb, 32, 0, cs, w48, nullptr, 16, 3, 1, (float *)ysm2, 0, 1e-5f, nullptr, nullptr);
        cmp("fp4 conv fwd split input, gn+silu both (mx4)", deq4(ysm2, o16), ysr2, shape_numel(o16), TOL4);
    }
    {   /* split with a segment that is not a multiple of 32 channels (dec2: 80 + 64 -> 64), mx4 */
        shape5 cs = {N, 144, 8, 8, 12}, c1 = cs, c2 = cs, ys = cs; c1.c = 80; c2.c = 64; ys.c = 64;
        float *xa = dev_rand(shape_numel(c1), 1.f), *xb = dev_rand(shape_numel(c2), 1.f);
        void *xam = mx4_from(xa, c1), *xbm = mx4_from(xb, c2); float *xad = deq4(xam, c1), *xbd = deq4(xbm, c2);
        float *w = dev_rand((size_t)64 * 144 * 27, 0.05f), *cat = dev_zero(shape_numel(cs)), *yr = dev_zero(shape_numel(ys));
        void *ym = mx4_new(ys);
        mode_ref(); nn_concat_fwd(xad, 80, xbd, 64, cs, cat); nn_conv3d_fwd(cat, cs, w, NULL, 64, 3, 1, yr);
        mode_mx(); nn_conv3d_fwd_split(xam, xbm, 80, cs, 0, NULL, NULL, NULL, NULL, w, NULL, 64, (float *)ym, 0, 1e-5f, NULL, NULL);
        cmp("fp4 conv fwd split 80 + 64 (mx4)", deq4(ym, ys), yr, shape_numel(ys), TOL4);
    }
    {   /* fp32 I/O through the fp4 kernel (the direct entry), D = 16 -> TZ 4 */
        shape5 xs = {N, 32, 16, 12, 16}, ys = xs; ys.c = 64;
        size_t nx = shape_numel(xs), ny = shape_numel(ys);
        float *x = dev_rand(nx, 1.f), *w = dev_rand((size_t)64 * 32 * 27, 0.1f), *b = dev_rand(64, 0.1f), *yr = dev_zero(ny), *y4 = dev_zero(ny);
        mode_ref(); nn_conv3d_fwd(x, xs, w, b, 64, 3, 1, yr);
        nn_conv3d_fwd_fp4(x, xs, w, b, 64, y4);
        cmp("fp4 conv fwd fp32 I/O (TZ 4, MT 4 -> TZ 2)", y4, yr, ny, TOL4);
    }
    /* ---- step 4: every consumer of an mx4 activation (training and inference). Inputs mx4 (dequantised for the fp32
       reference); TOL4O: one e2m1 output quantisation of the op's result; TOL: fp32 / mx8 outputs ---- */
    const double TOL4O = 0.16;
    {   /* upsample forward (mx4 -> mx4, plain and with gn+silu of the coarse input) */
        shape5 xs = {N, 32, 6, 6, 6}, ys = {N, 32, 12, 12, 12};
        size_t NG = (size_t)N * G;
        float *x = dev_rand(shape_numel(xs), 1.f); void *xm = mx4_from(x, xs); float *xd = deq4(xm, xs);
        float *yr = dev_zero(shape_numel(ys)); void *ym = mx4_new(ys);
        mode_ref(); nn_up2_fwd_into(xd, xs, yr, 32, 0);
        mode_mx(); nn_up2_fwd_into(xm, xs, ym, 32, 0);
        cmp("up2 fwd (mx4 -> mx4)", deq4(ym, ys), yr, shape_numel(ys), TOL4O);
        float *gam = dev_rand(32, 1.f), *bet = dev_rand(32, 0.5f), *mean = dev_rand(NG, 0.2f), *rstd = dev_rand(NG, 0.3f);
        { float h[64]; nn_d2h(h, rstd, NG * 4); for (size_t i = 0; i < NG; i++) h[i] = 0.8f + fabsf(h[i]); nn_h2d(rstd, h, NG * 4); }
        nn_gn_t g = {gam, bet, mean, rstd, G};
        float *t = dev_zero(shape_numel(xs)), *yr2 = dev_zero(shape_numel(ys)); void *ym2 = mx4_new(ys);
        mode_ref(); nn_gn_silu_apply(xd, xs, G, gam, bet, mean, rstd, t); nn_up2_fwd_into(t, xs, yr2, 32, 0);
        mode_mx(); nn_up2_fwd_gn_into(xm, xs, &g, ym2, 32, 0);
        cmp("up2 fwd gn+silu input (mx4 -> mx4)", deq4(ym2, ys), yr2, shape_numel(ys), TOL4O);
        void *ym8 = mx8_new(ys);   /* mx4 coarse -> mx8 transient */
        mode_mx(); nn_up2_fwd_into(xm, xs, ym8, 32, 0);
        cmp("up2 fwd (mx4 -> mx8)", deq8(ym8, ys), yr, shape_numel(ys), TOL);
    }
    {   /* GroupNorm + SiLU apply (mx4 / mx8 / fp32 in -> mx4 out), GroupNorm statistics, backward (mx4 x; fp32 or mx8 gradients) */
        shape5 s = {N, 16, 10, 10, 10}, s48 = {N, 64, 6, 6, 6};
        for (int pass = 0; pass < 2; pass++) {
            const shape5 ss = pass ? s48 : s;
            const int C = ss.c;
            size_t n = shape_numel(ss), NG = (size_t)N * G;
            float *x = dev_rand(n, 2.f), *gam = dev_rand(C, 1.f), *bet = dev_rand(C, 0.5f), *mean = dev_rand(NG, 0.2f), *rstd = dev_rand(NG, 0.3f);
            { float h[64]; nn_d2h(h, rstd, NG * 4); for (size_t i = 0; i < NG; i++) h[i] = 0.8f + fabsf(h[i]); nn_h2d(rstd, h, NG * 4); }
            void *xm = mx4_from(x, ss); float *xd = deq4(xm, ss);
            float *yr = dev_zero(n); void *ym = mx4_new(ss);
            char nm[96];
            mode_ref(); nn_gn_silu_apply(xd, ss, G, gam, bet, mean, rstd, yr);
            mode_mx(); nn_gn_silu_apply(xm, ss, G, gam, bet, mean, rstd, ym);
            snprintf(nm, sizeof nm, "gn+silu apply mx4 -> mx4 (%d ch)", C); cmp(nm, deq4(ym, ss), yr, n, TOL4O);
            void *x8 = mx8_from(x, ss); float *x8d = deq8(x8, ss), *yr8 = dev_zero(n); void *ym48 = mx4_new(ss);
            mode_ref(); nn_gn_silu_apply(x8d, ss, G, gam, bet, mean, rstd, yr8);
            mode_mx(); nn_gn_silu_apply(x8, ss, G, gam, bet, mean, rstd, ym48);
            snprintf(nm, sizeof nm, "gn+silu apply mx8 -> mx4 (%d ch)", C); cmp(nm, deq4(ym48, ss), yr8, n, TOL4O);
            float *yr32 = dev_zero(n); void *ymf = mx4_new(ss);
            mode_ref(); nn_gn_silu_apply(x, ss, G, gam, bet, mean, rstd, yr32);
            mode_mx(); nn_gn_silu_apply(x, ss, G, gam, bet, mean, rstd, ymf);
            snprintf(nm, sizeof nm, "gn+silu apply fp32 -> mx4 (%d ch)", C); cmp(nm, deq4(ymf, ss), yr32, n, TOL4O);
            float *m1 = dev_zero(NG), *r1 = dev_zero(NG), *m2 = dev_zero(NG), *r2 = dev_zero(NG);
            mode_ref(); nn_gn_stats(xd, ss, G, 1e-5f, m1, r1);
            mode_mx(); nn_gn_stats(xm, ss, G, 1e-5f, m2, r2);
            snprintf(nm, sizeof nm, "gn stats mean (mx4 x, %d ch)", C); cmp(nm, m2, m1, NG, 1e-4);
            snprintf(nm, sizeof nm, "gn stats rstd (mx4 x, %d ch)", C); cmp(nm, r2, r1, NG, 1e-4);
            float *gy = dev_rand(n, 1e-3f), *gxr = dev_zero(n), *gx2 = dev_zero(n), *gg1 = dev_zero(C), *gb1 = dev_zero(C), *gg2 = dev_zero(C), *gb2 = dev_zero(C), *scr = nn_malloc(nn_gn_scratch(ss) + 4096);
            mode_ref(); nn_gn_silu_bwd(xd, ss, G, gam, bet, mean, rstd, gy, gxr, gg1, gb1, scr);
            mode_mx(); nn_gn_silu_bwd(xm, ss, G, gam, bet, mean, rstd, gy, gx2, gg2, gb2, scr);
            snprintf(nm, sizeof nm, "gn+silu bwd gx (mx4 x, fp32 gy/gx, %d ch)", C); cmp(nm, gx2, gxr, n, 1e-4);
            snprintf(nm, sizeof nm, "gn+silu bwd ggamma (mx4 x, %d ch)", C); cmp(nm, gg2, gg1, C, 1e-4);
            snprintf(nm, sizeof nm, "gn+silu bwd gbeta (mx4 x, %d ch)", C); cmp(nm, gb2, gb1, C, 1e-4);
            void *gym = mx8_from(gy, ss); float *gyd = deq8(gym, ss), *gxr8 = dev_zero(n); void *gxm = mx8_new(ss);
            mode_ref(); nn_gn_silu_bwd(xd, ss, G, gam, bet, mean, rstd, gyd, gxr8, gg1, gb1, scr);
            mode_mx(); nn_gn_silu_bwd(xm, ss, G, gam, bet, mean, rstd, gym, gxm, gg2, gb2, scr);
            snprintf(nm, sizeof nm, "gn+silu bwd gx (mx4 x, mx8 gy/gx, %d ch)", C); cmp(nm, deq8(gxm, ss), gxr8, n, TOL);
        }
    }
    {   /* stride-1 weight gradients with an mx4 x: gn input (fp32 / mx8 gy), split input (dec0.c1 shape, 32 + 16), plain 16 ch */
        shape5 xs = {N, 16, 12, 12, 16}, ys = xs; ys.c = 32;
        size_t nx = shape_numel(xs), ny = shape_numel(ys), NG = (size_t)N * G;
        float *x = dev_rand(nx, 2.f), *gam = dev_rand(16, 1.f), *bet = dev_rand(16, 0.5f), *mean = dev_rand(NG, 0.2f), *rstd = dev_rand(NG, 0.3f);
        { float h[64]; nn_d2h(h, rstd, NG * 4); for (size_t i = 0; i < NG; i++) h[i] = 0.8f + fabsf(h[i]); nn_h2d(rstd, h, NG * 4); }
        void *xm = mx4_from(x, xs); float *xd = deq4(xm, xs);
        float *gy32 = dev_rand(ny, 1e-3f), *t = dev_zero(nx);
        float *gw1 = dev_zero((size_t)32 * 16 * 27), *gw2 = dev_zero((size_t)32 * 16 * 27), *gb1 = dev_zero(32), *gb2 = dev_zero(32);
        mode_ref(); nn_gn_silu_apply(xd, xs, G, gam, bet, mean, rstd, t); nn_conv3d_bwd_weight(t, xs, gy32, ys, 3, 1, gw1, gb1);
        mode_mx(); nn_conv3d_bwd_weight_gn(xm, xs, G, gam, bet, mean, rstd, gy32, ys, gw2, gb2);
        cmp("bwd_weight gn input (mx4 x, fp32 gy)", gw2, gw1, (size_t)32 * 16 * 27, TOL);
        cmp("bwd_weight bias (mx4 x, fp32 gy)", gb2, gb1, 32, TOL);
        void *gy32m = mx8_from(gy32, ys); float *gy32d = deq8(gy32m, ys);
        nn_zero(gw1, (size_t)32 * 16 * 27 * 4); nn_zero(gw2, (size_t)32 * 16 * 27 * 4);
        mode_ref(); nn_conv3d_bwd_weight(t, xs, gy32d, ys, 3, 1, gw1, nullptr);
        mode_mx(); nn_conv3d_bwd_weight_gn(xm, xs, G, gam, bet, mean, rstd, gy32m, ys, gw2, nullptr);
        cmp("bwd_weight gn input (mx4 x, mx8 gy)", gw2, gw1, (size_t)32 * 16 * 27, TOL);
        float *w3 = dev_zero((size_t)32 * 16 * 27), *w4 = dev_zero((size_t)32 * 16 * 27);
        mode_ref(); nn_conv3d_bwd_weight(xd, xs, gy32, ys, 3, 1, w3, nullptr);
        mode_mx(); nn_conv3d_bwd_weight(xm, xs, gy32, ys, 3, 1, w4, nullptr);
        cmp("bwd_weight plain (mx4 x, 16 ch)", w4, w3, (size_t)32 * 16 * 27, TOL);
        shape5 cs = xs; cs.c = 48; shape5 c1 = xs; c1.c = 32; shape5 o16 = xs; o16.c = 16;
        float *xa = dev_rand(shape_numel(c1), 1.f); void *xam = mx4_from(xa, c1); float *xad = deq4(xam, c1);
        float *cat = dev_zero(shape_numel(cs)), *gy16 = dev_rand(shape_numel(o16), 1e-3f);
        float *gw5 = dev_zero((size_t)16 * 48 * 27), *gw6 = dev_zero((size_t)16 * 48 * 27);
        mode_ref(); nn_concat_fwd(xad, 32, xd, 16, cs, cat); nn_conv3d_bwd_weight(cat, cs, gy16, o16, 3, 1, gw5, nullptr);
        mode_mx(); nn_conv3d_bwd_weight_split(xam, xm, 32, cs, 0, nullptr, nullptr, nullptr, nullptr, gy16, o16, gw6, nullptr);
        cmp("bwd_weight split input (mx4, 32 + 16)", gw6, gw5, (size_t)16 * 48 * 27, TOL);
    }
    {   /* stride 2: forward mx4 -> mx4 (plain, and with the gn+silu input of recompute mode), weight gradient (mx4 x; fp32 / mx8 gy) */
        shape5 xs = {N, 32, 16, 16, 16}, ys = {N, 32, 8, 8, 8}, x16 = {N, 16, 16, 16, 16}, y16 = {N, 16, 8, 8, 8};
        size_t nx = shape_numel(xs), ny = shape_numel(ys), NG = (size_t)N * G;
        float *x = dev_rand(nx, 1.f), *w = dev_rand((size_t)32 * 32 * 27, 0.1f), *b = dev_rand(32, 0.1f);
        void *xm = mx4_from(x, xs); float *xd = deq4(xm, xs);
        float *yr = dev_zero(ny); void *ym = mx4_new(ys);
        mode_ref(); nn_conv3d_fwd(xd, xs, w, b, 32, 3, 2, yr);
        mode_mx(); nn_conv3d_fwd(xm, xs, w, b, 32, 3, 2, ym);
        cmp("s2 fwd (mx4 -> mx4, fp8 compute)", deq4(ym, ys), yr, ny, TOL4O);
        float *gam = dev_rand(32, 1.f), *bet = dev_rand(32, 0.5f), *mean = dev_rand(NG, 0.2f), *rstd = dev_rand(NG, 0.3f);
        { float h[64]; nn_d2h(h, rstd, NG * 4); for (size_t i = 0; i < NG; i++) h[i] = 0.8f + fabsf(h[i]); nn_h2d(rstd, h, NG * 4); }
        nn_gn_t g = {gam, bet, mean, rstd, G};
        float *t = dev_zero(nx), *yr2 = dev_zero(ny); void *ym2 = mx4_new(ys);
        mode_ref(); nn_gn_silu_apply(xd, xs, G, gam, bet, mean, rstd, t); nn_conv3d_fwd(t, xs, w, b, 32, 3, 2, yr2);
        mode_mx(); nn_conv3d_fwd_x(xm, &g, nullptr, nullptr, 0, 0, xs, w, b, 32, 3, 2, (float *)ym2, 0, 0.f, nullptr, nullptr);
        cmp("s2 fwd gn+silu input (mx4 -> mx4)", deq4(ym2, ys), yr2, ny, TOL4O);
        float *x6 = dev_rand(shape_numel(x16), 1.f), *w6 = dev_rand((size_t)16 * 16 * 27, 0.1f); void *x6m = mx4_from(x6, x16); float *x6d = deq4(x6m, x16);
        float *y6r = dev_zero(shape_numel(y16)); void *y6m = mx4_new(y16);
        mode_ref(); nn_conv3d_fwd(x6d, x16, w6, nullptr, 16, 3, 2, y6r);
        mode_mx(); nn_conv3d_fwd(x6m, x16, w6, nullptr, 16, 3, 2, y6m);
        cmp("s2 fwd 16 ch (mx4 bw 16 -> mx4)", deq4(y6m, y16), y6r, shape_numel(y16), TOL4O);
        float *gy = dev_rand(ny, 1e-3f), *gw1 = dev_zero((size_t)32 * 32 * 27), *gw2 = dev_zero((size_t)32 * 32 * 27), *gb1 = dev_zero(32), *gb2 = dev_zero(32);
        mode_ref(); nn_conv3d_bwd_weight(xd, xs, gy, ys, 3, 2, gw1, gb1);
        mode_mx(); nn_conv3d_bwd_weight(xm, xs, gy, ys, 3, 2, gw2, gb2);
        cmp("s2 bwd_weight (mx4 x, fp32 gy)", gw2, gw1, (size_t)32 * 32 * 27, TOL);
        cmp("s2 bias grad (mx4 x)", gb2, gb1, 32, TOL);
        void *gym = mx8_from(gy, ys); float *gyd = deq8(gym, ys);
        nn_zero(gw1, (size_t)32 * 32 * 27 * 4); nn_zero(gw2, (size_t)32 * 32 * 27 * 4);
        mode_ref(); nn_conv3d_bwd_weight(t, xs, gyd, ys, 3, 2, gw1, nullptr);
        mode_mx(); nn_conv3d_bwd_weight_x(xm, &g, nullptr, nullptr, 0, 0, xs, gym, ys, 3, 2, gw2, nullptr);
        cmp("s2 bwd_weight gn+silu input (mx4 x, mx8 gy)", gw2, gw1, (size_t)32 * 32 * 27, TOL);
    }
    {   /* head: 1^3 forward (fp32 logits) and weight gradient from an mx4 x, plain and with gn+silu (recompute mode) */
        shape5 xs = {N, 16, 10, 10, 10}, ys = xs; ys.c = 1;
        size_t nx = shape_numel(xs), ny = shape_numel(ys), NG = (size_t)N * G;
        float *x = dev_rand(nx, 1.f), *w = dev_rand(16, 0.3f), *b = dev_rand(1, 0.1f);
        void *xm = mx4_from(x, xs); float *xd = deq4(xm, xs);
        float *yr = dev_zero(ny), *ym = dev_zero(ny);
        mode_ref(); nn_conv3d_fwd(xd, xs, w, b, 1, 1, 1, yr);
        mode_mx(); nn_conv3d_fwd(xm, xs, w, b, 1, 1, 1, ym);
        cmp("head fwd (mx4 x, fp32 logits)", ym, yr, ny, 1e-5);
        float *gam = dev_rand(16, 1.f), *bet = dev_rand(16, 0.5f), *mean = dev_rand(NG, 0.2f), *rstd = dev_rand(NG, 0.3f);
        { float h[64]; nn_d2h(h, rstd, NG * 4); for (size_t i = 0; i < NG; i++) h[i] = 0.8f + fabsf(h[i]); nn_h2d(rstd, h, NG * 4); }
        nn_gn_t g = {gam, bet, mean, rstd, G};
        float *t = dev_zero(nx), *yr2 = dev_zero(ny), *ym2 = dev_zero(ny);
        mode_ref(); nn_gn_silu_apply(xd, xs, G, gam, bet, mean, rstd, t); nn_conv3d_fwd(t, xs, w, b, 1, 1, 1, yr2);
        mode_mx(); nn_conv3d_fwd_x(xm, &g, nullptr, nullptr, 0, 0, xs, w, b, 1, 1, 1, ym2, 0, 0.f, nullptr, nullptr);
        cmp("head fwd gn+silu input (mx4 x)", ym2, yr2, ny, 1e-5);
        float *gy = dev_rand(ny, 1e-3f), *gw1 = dev_zero(16), *gw2 = dev_zero(16), *gb1 = dev_zero(1), *gb2 = dev_zero(1);
        mode_ref(); nn_conv3d_bwd_weight(xd, xs, gy, ys, 1, 1, gw1, gb1);
        mode_mx(); nn_conv3d_bwd_weight(xm, xs, gy, ys, 1, 1, gw2, gb2);
        cmp("head bwd_weight (mx4 x)", gw2, gw1, 16, 1e-4);
        cmp("head bias grad (mx4 x)", gb2, gb1, 1, 1e-4);
        nn_zero(gw1, 64); nn_zero(gw2, 64);
        mode_ref(); nn_conv3d_bwd_weight(t, xs, gy, ys, 1, 1, gw1, nullptr);
        mode_mx(); nn_conv3d_bwd_weight_x(xm, &g, nullptr, nullptr, 0, 0, xs, gy, ys, 1, 1, gw2, nullptr);
        cmp("head bwd_weight gn+silu input (mx4 x)", gw2, gw1, 16, 1e-4);
    }
    {   /* network input: fp32 / fp16 -> mx4 through the registry entry points */
        shape5 s = {N, 4, 16, 16, 16};
        size_t n = shape_numel(s);
        float *x = dev_rand(n, 1.f); void *a = mx4_new(s), *b = mx4_new(s);
        nn_f32_to_act(x, s, a);
        cmp("nn_f32_to_act -> mx4 == lp_f32_to_mx4", deq4(a, s), deq4(mx4_from(x, s), s), n, 1e-6);
        nn_set_f16(1); void *hx = nn_malloc(n * 2); nn_f32_to_h16(x, n, hx, 1.f); nn_h16_to_mx(hx, s, b); nn_set_f16(0);
        cmp("nn_h16_to_mx -> mx4 vs fp32 source", deq4(b, s), deq4(a, s), n, 0.03);
    }
    const char *e = nn_check(); if (e) { printf("cuda: %s\n", e); bad++; }
    e = lp_check(); if (e) { printf("cuda (lp): %s\n", e); bad++; }
    printf(bad ? "mx4 FAIL (%d)\n" : "mx4 ok\n", bad);
    (void)mode_ref; (void)mode_mx; (void)TOL4; (void)TOL; (void)G; (void)dev_rand;
    return bad != 0;
}
