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
    if (r2 == 0) e = 1e9;   /* an all-zero reference proves nothing */
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
    const int bw = s.c <= 8 ? 8 : s.c <= 16 ? 16 : 32, nb = (s.c + bw - 1) / bw; const size_t S = shape_spatial(s);
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
/* host snap of the packed 16-channel fp4 weights: one ue8m0 per (tap row, co, block), block 0 = [kx0 | kx1] x 16 ci, block 1 = kx2 */
static void host_w4p(float *w, int Co, int Ci) {
    for (int r = 0; r < 9; r++) for (int co = 0; co < Co; co++) for (int blk = 0; blk < 2; blk++) {
        float am = 0.f;
        for (int kx = 2 * blk; kx < 3 && kx < 2 * blk + 2; kx++) for (int ci = 0; ci < Ci; ci++) am = fmaxf(am, fabsf(w[((size_t)co * Ci + ci) * 27 + r * 3 + kx]));
        int e; if (am == 0.f) e = -126; else { e = (int)ceilf(log2f(am / 6.f)); float tt = ldexpf(am, -e); if (tt > 6.f) e++; if (e < -126) e = -126; if (e > 126) e = 126; }
        for (int kx = 2 * blk; kx < 3 && kx < 2 * blk + 2; kx++) for (int ci = 0; ci < Ci; ci++) { size_t i = ((size_t)co * Ci + ci) * 27 + r * 3 + kx; w[i] = copysignf(e2m1_rne(fabsf(w[i]) * ldexpf(1.f, -e)) * ldexpf(1.f, e), w[i]); }
    }
}
static void mode_ref(void) { nn_set_tf32(0); }
static void mode_mx(void) { nn_set_prec(3); }
/* host reference of a 1^3 weight gradient (and bias): gw[co][ci] = sum_n,v x[n][ci][v] gy[n][co][v] */
static void w1_ref(const float *x, shape5 xs, const float *gy, int co, float *gw, float *gb) {
    const size_t S = shape_spatial(xs), nx = shape_numel(xs), ng = (size_t)xs.n * co * S;
    float *hx = malloc(nx * 4), *hg = malloc(ng * 4), *hw = malloc((size_t)co * xs.c * 4), *hb = malloc((size_t)co * 4);
    nn_d2h(hx, x, nx * 4); nn_d2h(hg, gy, ng * 4);
    for (int o = 0; o < co; o++) {
        double b = 0;
        for (int n = 0; n < xs.n; n++) for (size_t v = 0; v < S; v++) b += hg[((size_t)n * co + o) * S + v];
        hb[o] = (float)b;
        for (int c = 0; c < xs.c; c++) {
            double a = 0;
            for (int n = 0; n < xs.n; n++) for (size_t v = 0; v < S; v++) a += (double)hx[((size_t)n * xs.c + c) * S + v] * hg[((size_t)n * co + o) * S + v];
            hw[(size_t)o * xs.c + c] = (float)a;
        }
    }
    nn_h2d(gw, hw, (size_t)co * xs.c * 4); if (gb) nn_h2d(gb, hb, (size_t)co * 4);
    free(hx); free(hg); free(hw); free(hb);
}
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
        for (int i = 0; i < 6; i++) {
            double m = lp_sr_e2m1_nib_mean(vals[i], 4000000);
            char what[64]; snprintf(what, sizeof what, "sr_e2m1_nib mean of 4e6 roundings of %.1f (%.4f)", vals[i], m);
            check(what, fabs(m - vals[i]) < 1e-3);
        }
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
        cmp("16-ch mx4 input, mx4 output (fp4 packed kernel)", deq4(ym, ys), yr, ny, TOL4);   /* fp8 compute + one mx4 output quantisation */
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
    {   /* decoder conv1 on mx4 storage without the full-resolution transient: the up part (32 ch) is read from the coarse
           tensor and upsampled while staging, the skip (16 ch) gets gn+silu (dec conv1 up segment from coarse mx4 (fp4)) */
        shape5 fs = {N, 48, 12, 12, 16}, co = {N, 32, 6, 6, 8}, fu = fs, sk = fs, o16 = fs; fu.c = 32; sk.c = 16; o16.c = 16;
        size_t NG = (size_t)N * G;
        float *xc = dev_rand(shape_numel(co), 1.f), *xs_ = dev_rand(shape_numel(sk), 2.f);
        void *xcm = mx4_from(xc, co), *xsm = mx4_from(xs_, sk); float *xcd = deq4(xcm, co), *xsd = deq4(xsm, sk);
        float *gam = dev_rand(16, 1.f), *bet = dev_rand(16, 0.5f), *mean = dev_rand(NG, 0.2f), *rstd = dev_rand(NG, 0.3f);
        { float h[64]; nn_d2h(h, rstd, NG * 4); for (size_t i = 0; i < NG; i++) h[i] = 0.8f + fabsf(h[i]); nn_h2d(rstd, h, NG * 4); }
        float *w = dev_rand((size_t)16 * 48 * 27, 0.1f), *b = dev_rand(16, 0.1f);
        float *up = dev_zero(shape_numel(fu)), *t = dev_zero(shape_numel(sk)), *cat = dev_zero(shape_numel(fs)), *yr = dev_zero(shape_numel(o16));
        mode_ref(); nn_up2_fwd_into(xcd, co, up, 32, 0); nn_gn_silu_apply(xsd, sk, G, gam, bet, mean, rstd, t); nn_concat_fwd(up, 32, t, 16, fs, cat); nn_conv3d_fwd(cat, fs, w, b, 16, 3, 1, yr);
        nn_gn_t g2 = {gam, bet, mean, rstd, G};
        void *ym = mx4_new(o16);
        mode_mx(); int rv = nn_conv3d_fwd_x(xcm, NULL, xsm, &g2, 32, 1, fs, w, b, 16, 3, 1, (float *)ym, 0, 1e-5f, NULL, NULL);
        if (rv) { printf("  dec conv1 with the up segment: unsupported (rv %d)  FAIL\n", rv); bad++; }
        else cmp("dec conv1 up segment from coarse mx4 (fp4)", deq4(ym, o16), yr, shape_numel(o16), TOL4);
    }
    {   /* step 5b: the packed 16-channel fp4 kernel (K = [kx0 | kx1 | kx2 | 0] x 16 ch, one scale per staged row) */
        /* exact: inputs on the e2m1 grid with a 6 in channel 0 of every voxel (every voxel and row scale = 1), weights on the
           packed grid, fp32 output -> the only rounding is the fp32 accumulation */
        const int cos_[2] = {16, 32};
        for (int k = 0; k < 2; k++) {
            shape5 xs = {N, 16, 16, 12, 20}, ys = xs; ys.c = cos_[k];
            size_t nx = shape_numel(xs), ny = shape_numel(ys), S = shape_spatial(xs);
            static const float grid[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
            float *h = malloc(nx * 4);
            for (size_t i = 0; i < nx; i++) { int c = (int)((i / S) % 16); h[i] = c == 0 ? 6.f : grid[(int)(fabsf(frand()) * 7.99f)] * (frand() < 0 ? -1.f : 1.f); }
            float *xg = nn_malloc(nx * 4); nn_h2d(xg, h, nx * 4); free(h);
            void *xm = mx4_from(xg, xs);
            float *hw = malloc((size_t)cos_[k] * 16 * 27 * 4), *w = dev_rand((size_t)cos_[k] * 16 * 27, 0.1f);
            nn_d2h(hw, w, (size_t)cos_[k] * 16 * 27 * 4); host_w4p(hw, cos_[k], 16); nn_h2d(w, hw, (size_t)cos_[k] * 16 * 27 * 4); free(hw);
            float *b = dev_rand(cos_[k], 0.1f), *yr = dev_zero(ny), *y4 = dev_zero(ny);
            mode_ref(); nn_conv3d_fwd(xg, xs, w, b, cos_[k], 3, 1, yr);
            mode_mx(); nn_conv3d_fwd(xm, xs, w, b, cos_[k], 3, 1, y4);
            char nm[96]; snprintf(nm, sizeof nm, "fp4 16-ch packed exact (mx4 in, grid w, fp32 out, 16 -> %d)", cos_[k]);
            cmp(nm, y4, yr, ny, 2e-3);
        }
        /* gn+silu input, mx4 in / out (enc0.c2 / dec0.c2), and 16 -> 32 (enc1.c1) */
        for (int k = 0; k < 2; k++) {
            shape5 xs = {N, 16, 16, 12, 16}, ys = xs; ys.c = cos_[k];
            size_t nx = shape_numel(xs), ny = shape_numel(ys), NG = (size_t)N * G;
            float *x = dev_rand(nx, 2.f), *w = dev_rand((size_t)cos_[k] * 16 * 27, 0.1f), *b = dev_rand(cos_[k], 0.1f);
            float *gam = dev_rand(16, 1.f), *bet = dev_rand(16, 0.5f), *mean = dev_rand(NG, 0.2f), *rstd = dev_rand(NG, 0.3f);
            { float hh[64]; nn_d2h(hh, rstd, NG * 4); for (size_t i = 0; i < NG; i++) hh[i] = 0.8f + fabsf(hh[i]); nn_h2d(rstd, hh, NG * 4); }
            void *xm = mx4_from(x, xs); float *xd = deq4(xm, xs);
            float *yr = dev_zero(ny), *t = dev_zero(nx), *m1 = dev_zero(NG), *r1 = dev_zero(NG), *m2 = dev_zero(NG), *r2 = dev_zero(NG); void *ym = mx4_new(ys);
            mode_ref(); nn_gn_silu_apply(xd, xs, G, gam, bet, mean, rstd, t); nn_conv3d_fwd(t, xs, w, b, cos_[k], 3, 1, yr); nn_gn_fwd(yr, ys, G, 1e-5f, nullptr, nullptr, nullptr, m1, r1);
            mode_mx(); nn_conv3d_fwd_gn_stats(xm, xs, G, gam, bet, mean, rstd, w, b, cos_[k], (float *)ym, G, 1e-5f, m2, r2);
            char nm[96]; snprintf(nm, sizeof nm, "fp4 16-ch packed gn+silu input (mx4 in/out, 16 -> %d)", cos_[k]);
            cmp(nm, deq4(ym, ys), yr, ny, TOL4);
            snprintf(nm, sizeof nm, "fp4 16-ch packed output GN rstd (16 -> %d)", cos_[k]); cmp(nm, r2, r1, NG, TOL4);
        }
        {   /* fp32 I/O through the direct entry (12 channels: padding inside the 16-wide slots) */
            shape5 xs = {N, 12, 8, 12, 16}, ys = xs; ys.c = 16;
            size_t nx = shape_numel(xs), ny = shape_numel(ys);
            float *x = dev_rand(nx, 1.f), *w = dev_rand((size_t)16 * 12 * 27, 0.1f), *yr = dev_zero(ny), *y4 = dev_zero(ny);
            mode_ref(); nn_conv3d_fwd(x, xs, w, nullptr, 16, 3, 1, yr);
            nn_conv3d_fwd_fp4(x, xs, w, nullptr, 16, y4);
            cmp("fp4 16-ch packed fp32 I/O (12 -> 16)", y4, yr, ny, TOL4);
        }
    }
    {   /* GroupNorm with channels per group not dividing the block width (deeper nets: 96 / 112 channels, G = 8 -> 12 / 14) */
        const int Cs[2] = {96, 112};
        for (int k = 0; k < 2; k++) {
            shape5 s = {N, Cs[k], 6, 6, 8};
            size_t n = shape_numel(s), NG = (size_t)N * G;
            float *x = dev_rand(n, 2.f), *gam = dev_rand(Cs[k], 1.f), *bet = dev_rand(Cs[k], 0.5f);
            void *xm = mx4_from(x, s); float *xd = deq4(xm, s);
            float *m1 = dev_zero(NG), *r1 = dev_zero(NG), *m2 = dev_zero(NG), *r2 = dev_zero(NG);
            mode_ref(); nn_gn_stats(xd, s, G, 1e-5f, m1, r1);
            mode_mx(); int rv = nn_gn_stats(xm, s, G, 1e-5f, m2, r2);
            char nm[96];
            snprintf(nm, sizeof nm, "gn stats mean, %d ch / 8 groups (mx4)", Cs[k]); if (rv) { printf("  %s unsupported  FAIL\n", nm); bad++; } else cmp(nm, m2, m1, NG, 1e-4);
            snprintf(nm, sizeof nm, "gn stats rstd, %d ch / 8 groups (mx4)", Cs[k]); cmp(nm, r2, r1, NG, 1e-4);
            float *yr = dev_zero(n); void *ym = mx4_new(s);
            mode_ref(); nn_gn_silu_apply(xd, s, G, gam, bet, m1, r1, yr);
            mode_mx(); nn_gn_silu_apply(xm, s, G, gam, bet, m1, r1, ym);
            snprintf(nm, sizeof nm, "gn+silu apply, %d ch / 8 groups (mx4)", Cs[k]); cmp(nm, deq4(ym, s), yr, n, 0.16);
            float *gy = dev_rand(n, 1e-3f), *gxr = dev_zero(n), *gx2 = dev_zero(n), *gg1 = dev_zero(Cs[k]), *gb1 = dev_zero(Cs[k]), *gg2 = dev_zero(Cs[k]), *gb2 = dev_zero(Cs[k]), *scr = nn_malloc(nn_gn_scratch(s) + 4096);
            mode_ref(); nn_gn_silu_bwd(xd, s, G, gam, bet, m1, r1, gy, gxr, gg1, gb1, scr);
            mode_mx(); nn_gn_silu_bwd(xm, s, G, gam, bet, m1, r1, gy, gx2, gg2, gb2, scr);
            snprintf(nm, sizeof nm, "gn+silu bwd gx, %d ch / 8 groups (mx4 x)", Cs[k]); cmp(nm, gx2, gxr, n, 1e-4);
            snprintf(nm, sizeof nm, "gn+silu bwd ggamma, %d ch / 8 groups (mx4 x)", Cs[k]); cmp(nm, gg2, gg1, Cs[k], 1e-4);
        }
    }
    {   /* fp4 weight gradient (lp_bwd_w_f4): (i) inputs on the e2m1 grid with round to nearest -> requantisation is lossless and
           the kernel must equal fp32 (layout / shift / scale-pairing check for every storage type), (ii) random inputs: format
           error of one call, stochastic rounding of gy averaged over 32 seeds (unbiased: the gy part of the error falls ~1/sqrt(32)),
           Hadamard on / off, heavy-tailed gy, gn+silu input, split input */
        static const float g8[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
        const int NS = 32;
        shape5 xs = {N, 48, 6, 12, 20}, ys = xs; ys.c = 32;   /* W = 20: partial x tile; Ci = 48: 3 channel tiles; Co = 32: MT 2 */
        size_t nx = shape_numel(xs), ny = shape_numel(ys), nw = (size_t)32 * 48 * 27;
        float *hx = malloc(nx * 4), *hg = malloc(ny * 4);
        for (size_t i = 0; i < nx; i++) hx[i] = (frand() < 0 ? -1.f : 1.f) * g8[(int)((frand() * 0.5f + 0.5f) * 7.999f)];
        for (size_t i = 0; i < ny; i++) hg[i] = (frand() < 0 ? -1.f : 1.f) * g8[(int)((frand() * 0.5f + 0.5f) * 7.999f)] * 0x1p-10f;
        float *xg = nn_malloc(nx * 4), *gg = nn_malloc(ny * 4); nn_h2d(xg, hx, nx * 4); nn_h2d(gg, hg, ny * 4);
        gnp_t none = {0}; split_t s0 = {0};
        float *gwr = dev_zero(nw), *gbr = dev_zero(32), *gw = dev_zero(nw), *gb = dev_zero(32), *acc = dev_zero(nw);
        mode_ref(); nn_conv3d_bwd_weight(xg, xs, gg, ys, 3, 1, gwr, gbr);
        nn_set_f16(1); void *xh = nn_malloc(nx * 2), *gh = nn_malloc(ny * 2); nn_f32_to_h16(xg, nx, xh, 1.f); nn_f32_to_h16(gg, ny, gh, 1.f); nn_set_f16(0);
        void *xb = nn_malloc(nx * 2), *gbf = nn_malloc(ny * 2); lp_f32_to_bf16(xg, nx, xb); lp_f32_to_bf16(gg, ny, gbf);
        void *x8 = mx8_from(xg, xs), *g8m = mx8_from(gg, ys), *x4 = mx4_from(xg, xs);
        struct { const char *nm; const void *x; int xt; const void *g; int gt; } ty[] = {
            {"fp32 x, fp32 gy", xg, 0, gg, 0}, {"fp16 x, fp32 gy", xh, 2, gg, 0}, {"fp16 x, fp16 gy", xh, 2, gh, 2}, {"bf16 x, bf16 gy", xb, 1, gbf, 1},
            {"mx8 x, mx8 gy", x8, 3, g8m, 3}, {"mx8 x, fp16 gy", x8, 3, gh, 2}, {"mx4 x, fp32 gy", x4, 4, gg, 0}, {"mx4 x, mx8 gy", x4, 4, g8m, 3}, {"mx4 x, bf16 gy", x4, 4, gbf, 1}};
        char nm[96];
        for (int i = 0; i < (int)(sizeof ty / sizeof ty[0]); i++) {
            nn_zero(gw, nw * 4); nn_zero(gb, 32 * 4);
            lp_bwd_w_f4(ty[i].x, ty[i].xt, xs, ty[i].g, ty[i].gt, ys, gw, gb, none, s0, 0);
            snprintf(nm, sizeof nm, "f4 wgrad exact on grid (%s)", ty[i].nm); cmp(nm, gw, gwr, nw, 1e-5);
            if (i == 0) cmp("f4 wgrad bias (fp32 gy)", gb, gbr, 32, 1e-5);
        }
        nn_zero(gw, nw * 4); lp_bwd_w_f4(xg, 0, xs, gg, 0, ys, gw, nullptr, none, s0, 1);
        cmp("f4 wgrad Hadamard, grid inputs (RN)", gw, gwr, nw, 0.25);
        nn_zero(gw, nw * 4); lp_bwd_w_f4(xg, 0, xs, gg, 0, ys, gw, nullptr, none, s0, 5);
        cmp("f4 wgrad H16 variant, grid inputs (RN)", gw, gwr, nw, 0.25);
        {   /* odd sizes: W = 18 (scalar loads), Ci = 8 (NT 1), Co = 16 (MT 1), D odd */
            shape5 xo = {1, 8, 5, 9, 18}, yo = xo; yo.c = 16;
            size_t a = shape_numel(xo), b = shape_numel(yo);
            float *x1 = nn_malloc(a * 4), *g1 = nn_malloc(b * 4); nn_h2d(x1, hx, a * 4); nn_h2d(g1, hg, b * 4);
            float *r1 = dev_zero((size_t)16 * 8 * 27), *o1 = dev_zero((size_t)16 * 8 * 27), *rb = dev_zero(16), *ob = dev_zero(16);
            mode_ref(); nn_conv3d_bwd_weight(x1, xo, g1, yo, 3, 1, r1, rb);
            lp_bwd_w_f4(x1, 0, xo, g1, 0, yo, o1, ob, none, s0, 0);
            cmp("f4 wgrad exact on grid (W 18, Ci 8, Co 16, D 5)", o1, r1, (size_t)16 * 8 * 27, 1e-5);
            cmp("f4 wgrad bias (W 18, Co 16)", ob, rb, 16, 1e-5);
        }
        {   /* W 16 (vector scale loads of the MX reads: 4 x scales, 8 gy scales per load), mx4 x / mx8 gy, grid values: exact */
            shape5 xw = {2, 32, 4, 8, 16}, yw = xw;
            size_t a = shape_numel(xw);
            float *x1 = nn_malloc(a * 4), *g1 = nn_malloc(a * 4); nn_h2d(x1, hx, a * 4); nn_h2d(g1, hg, a * 4);
            float *r1 = dev_zero((size_t)32 * 32 * 27), *o1 = dev_zero((size_t)32 * 32 * 27);
            void *x1m = mx4_from(x1, xw), *g1m = mx8_from(g1, yw);
            mode_ref(); nn_conv3d_bwd_weight(deq4(x1m, xw), xw, deq8(g1m, yw), yw, 3, 1, r1, nullptr);
            lp_bwd_w_f4(x1m, 4, xw, g1m, 3, yw, o1, nullptr, none, s0, 0);
            cmp("f4 wgrad exact on grid (W 16, mx4 x, mx8 gy)", o1, r1, (size_t)32 * 32 * 27, 1e-5);
        }
        {   /* 48 -> 16 (dec0.c1 shape: NT 3 under UFSM_F4W_LAYOUT=2), grid values: exact */
            shape5 yo = xs; yo.c = 16;
            float *r1 = dev_zero((size_t)16 * 48 * 27), *o1 = dev_zero((size_t)16 * 48 * 27);
            mode_ref(); nn_conv3d_bwd_weight(xg, xs, gg, yo, 3, 1, r1, nullptr);
            lp_bwd_w_f4(xg, 0, xs, gg, 0, yo, o1, nullptr, none, s0, 0);
            cmp("f4 wgrad exact on grid (48 -> 16)", o1, r1, (size_t)16 * 48 * 27, 1e-5);
        }
        {   /* split input (x2 segment, dec0.c1-like 32 + 16), grid values: exact */
            shape5 c1 = xs, c2 = xs; c1.c = 32; c2.c = 16;
            float *xa = nn_malloc(shape_numel(c1) * 4), *xc = nn_malloc(shape_numel(c2) * 4);
            nn_h2d(xa, hx, shape_numel(c1) * 4); nn_h2d(xc, hx + shape_numel(c1), shape_numel(c2) * 4);
            float *cat = dev_zero(nx); nn_concat_fwd(xa, 32, xc, 16, xs, cat);
            float *r = dev_zero(nw); mode_ref(); nn_conv3d_bwd_weight(cat, xs, gg, ys, 3, 1, r, nullptr);
            split_t sx = {0}; sx.x2 = xc; sx.c_split = 32;
            nn_zero(gw, nw * 4); lp_bwd_w_f4(xa, 0, xs, gg, 0, ys, gw, nullptr, none, sx, 0);
            cmp("f4 wgrad exact on grid (split 32 + 16)", gw, r, nw, 1e-5);
            void *xam = mx4_from(xa, c1), *xcm = mx4_from(xc, c2); sx.x2 = xcm;
            nn_zero(gw, nw * 4); lp_bwd_w_f4(xam, 4, xs, gg, 0, ys, gw, nullptr, none, sx, 0);
            cmp("f4 wgrad exact on grid (split, mx4 32 + 16)", gw, r, nw, 1e-5);
        }
        /* SR unbiasedness: grid x (lossless), random gy: single-seed error vs the 32-seed mean */
        for (size_t i = 0; i < ny; i++) hg[i] = nrand() * 1e-3f;
        nn_h2d(gg, hg, ny * 4); nn_zero(gwr, nw * 4); mode_ref(); nn_conv3d_bwd_weight(xg, xs, gg, ys, 3, 1, gwr, nullptr);
        for (int had = 0; had < 2; had++) {
            split_t sr = {0}; double e1 = 0, ea;
            nn_zero(acc, nw * 4);
            for (int k = 0; k < NS; k++) {
                sr.sr = (0x9e3779b9u * (unsigned)(k + 1)) | 1u;
                if (k == 0) { nn_zero(gw, nw * 4); lp_bwd_w_f4(xg, 0, xs, gg, 0, ys, gw, nullptr, none, sr, had); snprintf(nm, sizeof nm, "f4 wgrad SR one seed, grid x, N(0,1) gy%s", had ? ", Had" : ""); e1 = cmp(nm, gw, gwr, nw, 0.25); }
                lp_bwd_w_f4(xg, 0, xs, gg, 0, ys, acc, nullptr, none, sr, had);
            }
            nn_scale(acc, 1.f / NS, nw);
            snprintf(nm, sizeof nm, "f4 wgrad SR mean of %d seeds%s", NS, had ? ", Had" : ""); ea = cmp(nm, acc, gwr, nw, 0.25);
            /* plain: x is lossless, so only the zero-mean gy rounding is left and the mean must fall ~1/sqrt(32); Hadamard:
               the transformed x is rounded to nearest (a fixed error ~0.11 that no seed average removes), report only */
            if (!had) { snprintf(nm, sizeof nm, "  SR mean error < 0.35 x single (%.3g / %.3g)", ea, e1); check(nm, ea < 0.35 * e1); }
            split_t rn = {0}; nn_zero(gw, nw * 4); lp_bwd_w_f4(xg, 0, xs, gg, 0, ys, gw, nullptr, none, rn, had);
            snprintf(nm, sizeof nm, "f4 wgrad RN, grid x, N(0,1) gy%s", had ? ", Had" : ""); cmp(nm, gw, gwr, nw, 0.25);
        }
        /* random silu(N(0,1)) x and N(0,1) gy, then heavy-tailed gy (1% of the voxels x 30): format error RN / SR, Hadamard on / off */
        for (size_t i = 0; i < nx; i++) { float v = nrand(); hx[i] = v / (1.f + expf(-v)); }
        nn_h2d(xg, hx, nx * 4);
        for (int heavy = 0; heavy < 2; heavy++) {
            if (heavy) { for (size_t i = 0; i < ny; i++) if (frand() > 0.98f) hg[i] *= 30.f; nn_h2d(gg, hg, ny * 4); }
            nn_zero(gwr, nw * 4); mode_ref(); nn_conv3d_bwd_weight(xg, xs, gg, ys, 3, 1, gwr, nullptr);
            for (int had = 0; had < 6; had += had == 1 ? 4 : 1) for (int srm = 0; srm < 2; srm++) {   /* had 0, 1 (H32), 5 (H16) */
                split_t sr = {0}; sr.sr = srm ? 0x2545f491u : 0u;
                nn_zero(gw, nw * 4); lp_bwd_w_f4(xg, 0, xs, gg, 0, ys, gw, nullptr, none, sr, had);
                snprintf(nm, sizeof nm, "f4 wgrad %s gy, %s%s", heavy ? "heavy-tailed" : "N(0,1)", srm ? "SR" : "RN", had == 5 ? ", H16" : had ? ", Had" : ""); cmp(nm, gw, gwr, nw, srm ? 0.3 : 0.25);   /* one SR draw on heavy-tailed gy without Hadamard: ~0.28 */
            }
        }
        {   /* diagnostic x SR (had bit 1): both operands unbiased -> the 32-seed mean of random x / gy falls ~1/sqrt(32) */
            nn_zero(gwr, nw * 4); mode_ref(); nn_conv3d_bwd_weight(xg, xs, gg, ys, 3, 1, gwr, nullptr);
            for (int had = 2; had < 8; had += had == 3 ? 4 : 1) {   /* 2: x SR, 3: + H32, 7: + H16 */
                split_t sr = {0}; double e1 = 0, ea;
                nn_zero(acc, nw * 4);
                for (int k = 0; k < NS; k++) {
                    sr.sr = (0x7f4a7c15u * (unsigned)(k + 1)) | 1u;
                    if (k == 0) { nn_zero(gw, nw * 4); lp_bwd_w_f4(xg, 0, xs, gg, 0, ys, gw, nullptr, none, sr, had); snprintf(nm, sizeof nm, "f4 wgrad SR x and gy, one seed%s", had & 4 ? ", H16" : had & 1 ? ", Had" : ""); e1 = cmp(nm, gw, gwr, nw, 0.35); }
                    lp_bwd_w_f4(xg, 0, xs, gg, 0, ys, acc, nullptr, none, sr, had);
                }
                nn_scale(acc, 1.f / NS, nw);
                snprintf(nm, sizeof nm, "f4 wgrad SR x and gy, mean of %d%s", NS, had & 4 ? ", H16" : had & 1 ? ", Had" : ""); ea = cmp(nm, acc, gwr, nw, 0.25);
                snprintf(nm, sizeof nm, "  SR x+gy mean < 0.35 x single (%.3g / %.3g)", ea, e1); check(nm, ea < 0.35 * e1);
            }
        }
        {   /* gn+silu input (fused affine + SiLU while staging x), mx4 x */
            const int Gn = 8; size_t NG = (size_t)N * Gn;
            float *gam = dev_rand(48, 1.f), *bet = dev_rand(48, 0.5f), *mean = dev_rand(NG, 0.2f), *rstd = dev_rand(NG, 0.3f);
            { float h[64]; nn_d2h(h, rstd, NG * 4); for (size_t i = 0; i < NG; i++) h[i] = 0.8f + fabsf(h[i]); nn_h2d(rstd, h, NG * 4); }
            float *xr = dev_rand(nx, 2.f); void *xm = mx4_from(xr, xs); float *xd = deq4(xm, xs), *t = dev_zero(nx);
            mode_ref(); nn_gn_silu_apply(xd, xs, Gn, gam, bet, mean, rstd, t); nn_zero(gwr, nw * 4); nn_conv3d_bwd_weight(t, xs, gg, ys, 3, 1, gwr, nullptr);
            gnp_t gp = {gam, bet, mean, rstd, Gn};
            split_t sr = {0}; sr.sr = 0x51ed270bu;
            nn_zero(gw, nw * 4); lp_bwd_w_f4(xm, 4, xs, gg, 0, ys, gw, nullptr, gp, sr, 0);
            cmp("f4 wgrad gn+silu input (mx4 x, SR)", gw, gwr, nw, 0.3);
            nn_zero(gw, nw * 4); lp_bwd_w_f4(xm, 4, xs, g8m, 3, ys, gw, nullptr, gp, sr, 1);
            nn_zero(gwr, nw * 4); nn_conv3d_bwd_weight(t, xs, deq8(g8m, ys), ys, 3, 1, gwr, nullptr);
            cmp("f4 wgrad gn+silu input (mx4 x, mx8 gy, SR, Had)", gw, gwr, nw, 0.25);
        }
        free(hx); free(hg);
    }
    {   /* step 6: fp4 backward-data is the adjoint of the fp4 forward when both quantise the weights alike (2D tile scales):
           <conv_q(x), gy> == <x, conv_q^T(gy)> with x, gy on the e2m1 grid (a 6 in channel 0 of every voxel: lossless staging) */
        shape5 xs = {N, 32, 8, 12, 16}, ys = xs;
        size_t nx = shape_numel(xs), S = shape_spatial(xs);
        static const float grid[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
        float *h = malloc(nx * 4), *x = nn_malloc(nx * 4), *gy = nn_malloc(nx * 4);
        for (int pass = 0; pass < 2; pass++) {
            for (size_t i = 0; i < nx; i++) { int c = (int)((i / S) % 32); h[i] = c == 0 ? 6.f : grid[(int)(fabsf(frand()) * 7.99f)] * (frand() < 0 ? -1.f : 1.f); }
            nn_h2d(pass ? gy : x, h, nx * 4);
        }
        float *w = dev_rand((size_t)32 * 32 * 27, 0.1f), *y = dev_zero(nx), *gx = dev_zero(nx), *scr = nn_malloc(nn_conv3d_scratch(xs, 32, 3) + 4096);
        float *hy = malloc(nx * 4), *hg = malloc(nx * 4);
        double rel[2];
        for (int d2 = 0; d2 < 2; d2++) {
            lp_set_w4_2d(d2);
            mode_mx(); nn_conv3d_fwd(x, xs, w, nullptr, 32, 3, 1, y); nn_conv3d_bwd_data(gy, ys, w, xs, 3, 1, gx, scr);
            double a = 0, b = 0;
            nn_d2h(hy, y, nx * 4); nn_d2h(h, gy, nx * 4); for (size_t i = 0; i < nx; i++) a += (double)hy[i] * h[i];
            nn_d2h(hg, gx, nx * 4); nn_d2h(h, x, nx * 4); for (size_t i = 0; i < nx; i++) b += (double)hg[i] * h[i];
            rel[d2] = fabs(a - b) / fabs(a);
        }
        lp_set_w4_2d(-1);
        printf("  fp4 adjoint <Wx, gy> vs <x, W^T gy>: per-row scales %.3g (info), 2D tile scales %.3g\n", rel[0], rel[1]);
        check("fp4 backward-data = adjoint of the forward with 2D weight scales", rel[1] < 1e-5);
        free(h); free(hy); free(hg);
    }
    {   /* SR of the fp4 conv operand (the backward-data gy, MX-fp8): unbiased on both staging paths (16 channels: packed pair
           scales; 32: one scale per position). Weights in {-c, 0, c} with c = 6 / 16 are lossless in e2m1 whatever the block
           scales, the output is fp16, so the 32-seed mean must approach the fp32 conv of the stored gy ~1/sqrt(32) */
        for (int ci = 16; ci <= 32; ci += 16) {
            shape5 xs = {N, ci, 8, 10, 20};
            const int co = 16; shape5 ys = xs; ys.c = co;
            const size_t nx = shape_numel(xs), ny = shape_numel(ys), nw = (size_t)co * ci * 27;
            float *hx = malloc(nx * 4), *hw = malloc(nw * 4);
            for (size_t i = 0; i < nx; i++) hx[i] = nrand() * 1e-3f;
            for (size_t i = 0; i < nw; i++) { const float r = frand(); hw[i] = r < -0.33f ? -0.375f : r > 0.33f ? 0.375f : 0.f; }
            float *xf = nn_malloc(nx * 4), *w = nn_malloc(nw * 4); nn_h2d(xf, hx, nx * 4); nn_h2d(w, hw, nw * 4);
            void *xm = mx8_from(xf, xs); float *xd = deq8(xm, xs), *yr = dev_zero(ny);
            mode_ref(); nn_conv3d_fwd(xd, xs, w, nullptr, co, 3, 1, yr);
            float *hr = malloc(ny * 4), *acc = calloc(ny, 4), *one = malloc(ny * 4); nn_d2h(hr, yr, ny * 4);
            _Float16 *hh = malloc(ny * 2); void *yh = nn_malloc(ny * 2);
            const int NSD = 32; double e1 = 0, ea = 0, rr = 0;
            mode_mx();
            for (int k = 0; k < NSD; k++) {
                gnp_t none = {0}; split_t sr = {0}; sr.sr = (0x9e3779b9u * (unsigned)(k + 1)) | 1u;
                lp_conv_fwd_f4(xm, 3, xs, w, nullptr, co, yh, 2, none, nullptr, 0, sr);
                nn_d2h(hh, yh, ny * 2);
                for (size_t i = 0; i < ny; i++) { acc[i] += (float)hh[i]; if (k == 0) one[i] = (float)hh[i]; }
            }
            for (size_t i = 0; i < ny; i++) { const double r = hr[i], a = acc[i] / NSD; e1 += (one[i] - r) * (one[i] - r); ea += (a - r) * (a - r); rr += r * r; }
            e1 = sqrt(e1 / rr); ea = sqrt(ea / rr);
            char nm[112]; snprintf(nm, sizeof nm, "fp4 conv SR operand (%d ch, mx8): one seed %.3g, %d-seed mean %.3g", ci, e1, NSD, ea);
            check(nm, e1 > 1e-3 && ea < 0.35 * e1);
            free(hx); free(hw); free(hr); free(acc); free(one); free(hh); nn_free(xf); nn_free(w); nn_free(xm); nn_free(xd); nn_free(yr); nn_free(yh);
        }
    }
    {   /* dec0.c1 backward-data shape on the packed fp4 kernel: gy 16 ch -> 48 outputs split 32 + 16 (one 48-row tile, MT 3) */
        shape5 o16 = {N, 16, 16, 12, 16}, cs = o16, c1 = o16, c2 = o16; cs.c = 48; c1.c = 32;
        float *gy = dev_rand(shape_numel(o16), 1.f), *w = dev_rand((size_t)16 * 48 * 27, 0.1f), *scr = nn_malloc(nn_conv3d_scratch(cs, 16, 3) + 4096);
        float *gxr = dev_zero(shape_numel(cs)), *ga = dev_zero(shape_numel(c1)), *gb_ = dev_zero(shape_numel(c2)), *g1 = dev_zero(shape_numel(c1)), *g2 = dev_zero(shape_numel(c2));
        mode_ref(); nn_conv3d_bwd_data(gy, o16, w, cs, 3, 1, gxr, scr); nn_concat_bwd(gxr, 32, 16, cs, ga, gb_);
        mode_mx(); nn_conv3d_bwd_data_split(gy, o16, w, cs, g1, g2, 32, scr);
        cmp("fp4 bwd_data 16 -> 32 + 16 (packed, 48-row tile), part 1", g1, ga, shape_numel(c1), TOL4);
        cmp("fp4 bwd_data 16 -> 32 + 16 (packed, 48-row tile), part 2", g2, gb_, shape_numel(c2), TOL4);
    }
    {   /* fp4 weight gradient on MX x: the decoded-tile staging (raw decoded plane in bf16, exact; one row load per voxel) must equal
           the per-element path (had bit 3) to atomic-order noise, per layout mode, gn + SR included */
        const int Gn = 8;
        shape5 xs = {2, 32, 6, 10, 16}, ys = xs;
        size_t nx = shape_numel(xs), nw = (size_t)32 * 32 * 27, NG = (size_t)2 * Gn;
        float *xr = dev_rand(nx, 2.f), *gr = dev_rand(nx, 1e-3f);
        float *gam = dev_rand(32, 1.f), *bet = dev_rand(32, 0.5f), *mean = dev_rand(NG, 0.2f), *rstd = dev_rand(NG, 0.3f);
        { float h[64]; nn_d2h(h, rstd, NG * 4); for (size_t j = 0; j < NG; j++) h[j] = 0.8f + fabsf(h[j]); nn_h2d(rstd, h, NG * 4); }
        void *xm = mx4_from(xr, xs), *gm = mx8_from(gr, ys);
        gnp_t gp = {gam, bet, mean, rstd, Gn}; split_t sr = {0}; sr.sr = 0x1b873593u;
        float *a = dev_zero(nw), *b = dev_zero(nw);
        const int modes[] = {0, 1, 3, 5, 7}; const char *mn[] = {"plain", "H32", "H32 + x SR", "H16", "H16 + x SR"};
        for (int i = 0; i < 5; i++) {
            nn_zero(a, nw * 4); nn_zero(b, nw * 4);
            lp_bwd_w_f4(xm, 4, xs, gm, 3, ys, a, nullptr, gp, sr, modes[i]);
            lp_bwd_w_f4(xm, 4, xs, gm, 3, ys, b, nullptr, gp, sr, modes[i] | 8);
            char nm[96]; snprintf(nm, sizeof nm, "f4 wgrad tile == per-element (mx4 x, gn, SR, %s)", mn[i]);
            cmp(nm, a, b, nw, 1e-6);
        }
    }
    {   /* fp4 weight gradient with the gy pre-pass (pre-rounded A blocks, z slabs; bias from the pre-pass) == in-kernel
           rounding bit for bit (fp32 reduction order aside), SR and gn+silu included, ragged W / H / D and slab edges */
        const int Gn = 8;
        struct { int ci, co, d, h, w; const char *nm; } cs[] = {{16, 16, 9, 11, 20, "16 -> 16, 9 x 11 x 20"}, {32, 32, 6, 10, 16, "32 -> 32"},
                                                              {48, 16, 7, 8, 24, "48 -> 16"}, {32, 48, 5, 9, 18, "32 -> 48 (MT 2)"}};
        for (int i = 0; i < 4; i++) for (int kb = 1; kb <= 2; kb++) {
            shape5 xs = {2, cs[i].ci, cs[i].d, cs[i].h, cs[i].w}, ys = xs; ys.c = cs[i].co;
            size_t nx = shape_numel(xs), ny = shape_numel(ys), nw = (size_t)cs[i].co * cs[i].ci * 27, NG = (size_t)2 * Gn;
            float *xr = dev_rand(nx, 2.f), *gr = dev_rand(ny, 1e-3f);
            float *gam = dev_rand(cs[i].ci, 1.f), *bet = dev_rand(cs[i].ci, 0.5f), *mean = dev_rand(NG, 0.2f), *rstd = dev_rand(NG, 0.3f);
            { float h[64]; nn_d2h(h, rstd, NG * 4); for (size_t j = 0; j < NG; j++) h[j] = 0.8f + fabsf(h[j]); nn_h2d(rstd, h, NG * 4); }
            void *xm = mx4_from(xr, xs), *gm = mx8_from(gr, ys);
            gnp_t gp = {gam, bet, mean, rstd, Gn}; split_t sr = {0}; sr.sr = 0x3c6ef372u;
            float *a = dev_zero(nw), *b = dev_zero(nw), *ab = dev_zero(cs[i].co), *bb = dev_zero(cs[i].co);
            lp_set_f4w_gypre_kb(kb == 1 ? 1 : 1 << 20);   /* 1 KiB: one z-step per slab; 1 GiB: one slab */
            lp_bwd_w_f4(xm, 4, xs, gm, 3, ys, a, ab, gp, sr, 0);
            lp_set_f4w_gypre_kb(0);
            lp_bwd_w_f4(xm, 4, xs, gm, 3, ys, b, bb, gp, sr, 0);
            lp_set_f4w_gypre_kb(-1);
            char nm[112]; snprintf(nm, sizeof nm, "f4 wgrad gy pre-pass == in-kernel (%s, %s)", cs[i].nm, kb == 1 ? "1 z-step slabs" : "one slab");
            cmp(nm, a, b, nw, 1e-6);
            snprintf(nm, sizeof nm, "f4 wgrad gy pre-pass bias (%s)", cs[i].nm); cmp(nm, ab, bb, cs[i].co, 1e-5);
            nn_free(xr); nn_free(gr); nn_free(gam); nn_free(bet); nn_free(mean); nn_free(rstd); nn_free(xm); nn_free(gm); nn_free(a); nn_free(b); nn_free(ab); nn_free(bb);
        }
    }
    {   /* fp8 weight gradient on MX inputs: the cooperative staging (x row decoded once per voxel for all channels, gy per
           voxel for all outputs) must reproduce the per-element staging bit for bit, SR and gn+silu included */
        const int Gn = 8;
        struct { int ci, co, w; const char *nm; } cs[] = {{32, 32, 16, "32 -> 32, W 16"}, {16, 16, 20, "16 -> 16, W 20"}, {48, 16, 16, "48 -> 16 (NT 3)"}, {80, 80, 12, "80 -> 80, W 12"},
                                                          {4, 16, 16, "4 -> 16 (network input), W 16"}, {4, 32, 16, "4 -> 32 (network input), W 16"}};
        for (int i = 0; i < 6; i++) {
            shape5 xs = {2, cs[i].ci, 6, 10, cs[i].w}, ys = xs; ys.c = cs[i].co;
            size_t nx = shape_numel(xs), ny = shape_numel(ys), nw = (size_t)cs[i].co * cs[i].ci * 27, NG = (size_t)2 * Gn;
            float *xr = dev_rand(nx, 2.f), *gr = dev_rand(ny, 1e-3f);
            float *gam = dev_rand(cs[i].ci, 1.f), *bet = dev_rand(cs[i].ci, 0.5f), *mean = dev_rand(NG, 0.2f), *rstd = dev_rand(NG, 0.3f);
            { float h[64]; nn_d2h(h, rstd, NG * 4); for (size_t j = 0; j < NG; j++) h[j] = 0.8f + fabsf(h[j]); nn_h2d(rstd, h, NG * 4); }
            void *xm = mx4_from(xr, xs), *x8 = mx8_from(xr, xs), *gm = mx8_from(gr, ys);
            gnp_t gp = {gam, bet, mean, rstd, Gn}; split_t sr = {0}; sr.sr = 0x3c6ef372u;
            float *a = dev_zero(nw), *b = dev_zero(nw), *ab = dev_zero(cs[i].co), *bb = dev_zero(cs[i].co);
            for (int xt = 3; xt <= 4; xt++) {
                nn_zero(a, nw * 4); nn_zero(b, nw * 4); nn_zero(ab, cs[i].co * 4); nn_zero(bb, cs[i].co * 4);
                lp_set_f8w_coop(3); lp_bwd_w_f8(xt == 4 ? xm : x8, xt, xs, gm, 3, ys, a, ab, gp, sr);   /* both cooperative paths */
                lp_set_f8w_coop(0); lp_bwd_w_f8(xt == 4 ? xm : x8, xt, xs, gm, 3, ys, b, bb, gp, sr);
                lp_set_f8w_coop(1 | 4);   /* the default: x cooperative, gy cooperative for the network input conv */
                char nm[96]; snprintf(nm, sizeof nm, "f8 wgrad coop == per-element (%s, %s x, SR, gn)", cs[i].nm, xt == 4 ? "mx4" : "mx8");
                cmp(nm, a, b, nw, 1e-6);
                snprintf(nm, sizeof nm, "f8 wgrad coop bias (%s)", cs[i].nm); cmp(nm, ab, bb, cs[i].co, 1e-6);
            }
        }
    }
    {   /* decoder conv1 weight gradient with the fused upsample on MX storage (no full-resolution transient): the up part (32 ch)
           is interpolated from the coarse mx4 tensor in the kernels' decode, the skip (16 ch) gets gn+silu. vs the fp32
           reference, and vs the same kernel on an fp32 materialised upsample (same values up to summation order) */
        shape5 fs = {N, 48, 12, 12, 16}, co = {N, 32, 6, 6, 8}, fu = fs, sk = fs, o16 = fs; fu.c = 32; sk.c = 16; o16.c = 16;
        const size_t NG = (size_t)N * G, nw = (size_t)16 * 48 * 27;
        float *xc = dev_rand(shape_numel(co), 1.f), *xs_ = dev_rand(shape_numel(sk), 2.f), *gy = dev_rand(shape_numel(o16), 1e-3f);
        void *xcm = mx4_from(xc, co), *xsm = mx4_from(xs_, sk); float *xcd = deq4(xcm, co), *xsd = deq4(xsm, sk);
        float *gam = dev_rand(16, 1.f), *bet = dev_rand(16, 0.5f), *mean = dev_rand(NG, 0.2f), *rstd = dev_rand(NG, 0.3f);
        { float h[64]; nn_d2h(h, rstd, NG * 4); for (size_t i = 0; i < NG; i++) h[i] = 0.8f + fabsf(h[i]); nn_h2d(rstd, h, NG * 4); }
        float *up = dev_zero(shape_numel(fu)), *t = dev_zero(shape_numel(sk)), *cat = dev_zero(shape_numel(fs)), *gr = dev_zero(nw), *gw = dev_zero(nw), *gw2 = dev_zero(nw);
        mode_ref(); nn_up2_fwd_into(xcd, co, up, 32, 0); nn_gn_silu_apply(xsd, sk, G, gam, bet, mean, rstd, t); nn_concat_fwd(up, 32, t, 16, fs, cat);
        nn_conv3d_bwd_weight(cat, fs, gy, o16, 3, 1, gr, nullptr);
        nn_gn_t g2 = {gam, bet, mean, rstd, G};
        mode_mx(); int rv = nn_conv3d_bwd_weight_x(xcm, NULL, xsm, &g2, 32, 1, fs, gy, o16, 3, 1, gw, NULL);
        if (rv) { printf("  dec conv1 wgrad fused up (MX): unsupported (rv %d)  FAIL\n", rv); bad++; }
        else cmp("dec conv1 wgrad fused up from coarse mx4 (fp8)", gw, gr, nw, TOL);
        gnp_t gpn = {0}, gq = {gam, bet, mean, rstd, G};
        split_t su = {0}; su.x2 = xsm; su.c_split = 32; su.gp2 = gq; su.up = 1;
        split_t sf = {0}; sf.x2 = xsd; sf.c_split = 32; sf.gp2 = gq;
        nn_zero(gw, nw * 4); nn_zero(gw2, nw * 4);
        lp_bwd_w_f8(xcm, 4, fs, gy, 0, o16, gw, nullptr, gpn, su);
        lp_bwd_w_f8(up, 0, fs, gy, 0, o16, gw2, nullptr, gpn, sf);
        cmp("dec conv1 wgrad fused up (fp8) vs materialised fp32 up", gw, gw2, nw, 5e-3);
        for (int had = 0; had < 6; had += 5) {   /* fp4 plain (default layout), H16 (4|1 = 5) */
            nn_zero(gw, nw * 4); nn_zero(gw2, nw * 4);
            lp_bwd_w_f4(xcm, 4, fs, gy, 0, o16, gw, nullptr, gpn, su, had);
            lp_bwd_w_f4(up, 0, fs, gy, 0, o16, gw2, nullptr, gpn, sf, had);
            char nm[96]; snprintf(nm, sizeof nm, "dec conv1 wgrad fused up (fp4%s) vs fp32 reference", had ? ", H16" : "");
            cmp(nm, gw, gr, nw, 0.25);
            snprintf(nm, sizeof nm, "dec conv1 wgrad fused up (fp4%s) vs materialised fp32 up", had ? ", H16" : "");
            cmp(nm, gw, gw2, nw, 0.03);   /* bf16 tile rounding of the interpolated values (rare e2m1 flips) */
        }
    }
    {   /* MX activation dispatch must preserve SR for an off-grid weight-gradient operand.
           Stored MX-fp8 gy can already be exactly representable, so it need not vary with the seed. */
        shape5 s = {1, 32, 8, 8, 16};
        const size_t nw = (size_t)32 * 32 * 27;
        float *x = dev_rand(shape_numel(s), 1.f), *gy = dev_rand(shape_numel(s), 1.f);
        void *mx = mx4_from(x, s);
        float *gw = dev_zero(nw), *h1 = malloc(nw * 4), *h2 = malloc(nw * 4);
        nn_gn_t none = {0};
        nn_set_tf32(1); nn_set_layer(0); nn_set_conv(0); nn_set_conv_prec(0, 0, 3, 3, 2); nn_set_sr(1);
        double repeat = 0, changed = 0, norm = 0;
        for (int k = 0; k < 3; k++) {
            nn_set_sr_step(k == 2 ? 2 : 1); nn_zero(gw, nw * 4);
            check("MX SR wgrad dispatch", !nn_conv3d_bwd_weight_x(mx, &none, nullptr, nullptr, 0, 0, s, gy, s, 3, 1, gw, nullptr));
            nn_d2h(k ? h2 : h1, gw, nw * 4);
            if (k) for (size_t j = 0; j < nw; j++) {
                double d = (double)h1[j] - h2[j];
                if (k == 1) repeat += d*d; else changed += d*d;
            }
        }
        for (size_t j = 0; j < nw; j++) norm += (double)h1[j]*h1[j];
        printf("  MX SR wgrad: repeat %.3g, different step %.3g\n", sqrt(repeat/norm), sqrt(changed/norm));
        check("MX SR same step reproducible", repeat < norm*1e-8);
        check("MX SR different steps change off-grid gy", changed > norm*1e-6);
        nn_set_conv_prec(0, 0, 0, 0, 0); nn_set_layer(-1); nn_set_conv(-1); nn_set_sr(0);
        nn_free(mx); nn_free(x); nn_free(gy); nn_free(gw); free(h1); free(h2);
    }
    {   /* decoder conv1 with the up segment read from the coarse PRE-GN tensor: GN+SiLU of the coarse rows inside the up staging (no kept s2) */
        shape5 fs = {N, 48, 12, 12, 16}, co = {N, 32, 6, 6, 8}, fu = fs, sk = fs, o16 = fs; fu.c = 32; sk.c = 16; o16.c = 16;
        size_t NG = (size_t)N * G;
        float *xc = dev_rand(shape_numel(co), 2.f), *xs_ = dev_rand(shape_numel(sk), 2.f);
        void *xcm = mx4_from(xc, co), *xsm = mx4_from(xs_, sk); float *xcd = deq4(xcm, co), *xsd = deq4(xsm, sk);
        float *gam = dev_rand(16, 1.f), *bet = dev_rand(16, 0.5f), *mean = dev_rand(NG, 0.2f), *rstd = dev_rand(NG, 0.3f);
        float *gac = dev_rand(32, 1.f), *bec = dev_rand(32, 0.5f), *mec = dev_rand(NG, 0.2f), *rsc = dev_rand(NG, 0.3f);
        { float h[64]; nn_d2h(h, rstd, NG * 4); for (size_t i = 0; i < NG; i++) h[i] = 0.8f + fabsf(h[i]); nn_h2d(rstd, h, NG * 4); nn_d2h(h, rsc, NG * 4); for (size_t i = 0; i < NG; i++) h[i] = 0.8f + fabsf(h[i]); nn_h2d(rsc, h, NG * 4); }
        float *w = dev_rand((size_t)16 * 48 * 27, 0.1f), *b = dev_rand(16, 0.1f);
        float *tc = dev_zero(shape_numel(co)), *up = dev_zero(shape_numel(fu)), *t = dev_zero(shape_numel(sk)), *cat = dev_zero(shape_numel(fs)), *yr = dev_zero(shape_numel(o16));
        mode_ref(); nn_gn_silu_apply(xcd, co, G, gac, bec, mec, rsc, tc); nn_up2_fwd_into(tc, co, up, 32, 0); nn_gn_silu_apply(xsd, sk, G, gam, bet, mean, rstd, t); nn_concat_fwd(up, 32, t, 16, fs, cat); nn_conv3d_fwd(cat, fs, w, b, 16, 3, 1, yr);
        nn_gn_t gc = {gac, bec, mec, rsc, G}, g2 = {gam, bet, mean, rstd, G};
        void *ym = mx4_new(o16);
        mode_mx(); int rv = nn_conv3d_fwd_x(xcm, &gc, xsm, &g2, 32, 1, fs, w, b, 16, 3, 1, (float *)ym, 0, 1e-5f, NULL, NULL);
        if (rv) { printf("  dec conv1 up segment with GN: unsupported (rv %d)  FAIL\n", rv); bad++; }
        else cmp("dec conv1 up from coarse pre-GN mx4 (fp4)", deq4(ym, o16), yr, shape_numel(o16), TOL4);
    }
    const char *e = nn_check(); if (e) { printf("cuda: %s\n", e); bad++; }
    e = lp_check(); if (e) { printf("cuda (lp): %s\n", e); bad++; }
    printf(bad ? "mx4 FAIL (%d)\n" : "mx4 ok\n", bad);
    (void)mode_ref; (void)mode_mx; (void)TOL4; (void)TOL; (void)G; (void)dev_rand;
    for (int CI = 16; CI <= 32; CI += 16) {   /* wide heads (band affinity outputs): 1^3 weight gradient, 16 / 32 MX inputs (optionally
                                                 GN+SiLU) x 3..8 outputs, vs fp32 */
        shape5 xs = {2, CI, 6, 10, 12};
        const size_t S = shape_spatial(xs);
        for (int dt = 4; dt <= 8; dt += 4) for (int co = 3; co <= 8; co += 4) for (int G = 0; G <= 4; G += 4) {
            shape5 ys = xs; ys.c = co;
            float *x = dev_rand(shape_numel(xs), 2.f), *gy = dev_rand(shape_numel(ys), 1.f);
            void *xm = dt == 4 ? mx4_from(x, xs) : mx8_from(x, xs); float *xd = dt == 4 ? deq4(xm, xs) : deq8(xm, xs);
            float *gam = dev_rand(CI, 1.f), *bet = dev_rand(CI, 0.5f), *mean = dev_rand((size_t)xs.n * 4, 0.2f), *rstd = dev_rand((size_t)xs.n * 4, 0.3f);
            { float h[8]; nn_d2h(h, rstd, (size_t)xs.n * 4 * 4); for (int k = 0; k < xs.n * 4; k++) h[k] = 0.8f + fabsf(h[k]); nn_h2d(rstd, h, (size_t)xs.n * 4 * 4); }
            float *xa = dev_zero(shape_numel(xs)), *gr = dev_zero((size_t)co * CI), *gw = dev_zero((size_t)co * CI);
            mode_ref();
            if (G) nn_gn_silu_apply(xd, xs, G, gam, bet, mean, rstd, xa); else nn_d2d(xa, xd, shape_numel(xs) * 4);
            if (CI == 16) nn_conv3d_bwd_weight(xa, xs, gy, ys, 1, 1, gr, nullptr); else w1_ref(xa, xs, gy, co, gr, nullptr);
            gnp_t gp = {G ? gam : nullptr, G ? bet : nullptr, G ? mean : nullptr, G ? rstd : nullptr, G};
            lp_bwd_w1_mx(xm, dt == 4 ? 4 : 3, xs, gy, 0, ys, gw, gp);
            char nm[96]; snprintf(nm, sizeof nm, "head wgrad %d -> %d, mx%d%s", CI, co, dt, G ? ", gn+silu" : "");
            cmp(nm, gw, gr, (size_t)co * CI, 1e-4);
            {   /* fp16 gy (the training storage): tensor-core kernel with the bias gradient in the same pass; fp16 rounding of
                   gy and of the activations bounds the agreement */
                const int f16 = nn_get_f16(); nn_set_f16(1);
                void *gyh = nn_malloc(shape_numel(ys) * 2); nn_f32_to_h16(gy, shape_numel(ys), gyh, 1.f);
                float *grb = dev_zero(co), *gw2 = dev_zero((size_t)co * CI), *gb2 = dev_zero(co), *gr2 = dev_zero((size_t)co * CI);
                mode_ref(); if (CI == 16) nn_conv3d_bwd_weight(xa, xs, gy, ys, 1, 1, gr2, grb); else w1_ref(xa, xs, gy, co, gr2, grb);
                const int bd = lp_bwd_w1_mx_b(xm, dt == 4 ? 4 : 3, xs, gyh, 2, ys, gw2, gb2, gp);
                char nm2[112]; snprintf(nm2, sizeof nm2, "head wgrad %d -> %d, mx%d%s, fp16 gy (tensor cores)", CI, co, dt, G ? ", gn+silu" : "");
                cmp(nm2, gw2, gr2, (size_t)co * CI, 3e-3);
                if (bd) { snprintf(nm2, sizeof nm2, "head bias grad -> %d, fp16 gy, same pass", co); cmp(nm2, gb2, grb, co, 3e-3); }
                else { printf("  head wgrad fp16 gy: tensor-core path not taken  FAIL\n"); bad++; }
                nn_set_f16(f16); nn_free(gyh); nn_free(grb); nn_free(gw2); nn_free(gb2); nn_free(gr2);
            }
            nn_free(x); nn_free(gy); nn_free(xm); nn_free(xd); nn_free(xa); nn_free(gr); nn_free(gw); nn_free(gam); nn_free(bet); nn_free(mean); nn_free(rstd);
            (void)S;
        }
    }
    return bad != 0;
}
