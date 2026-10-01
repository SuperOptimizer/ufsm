/* MX-fp8 storage ops against the exact fp32 kernels: every MX input is dequantized and fed to the fp32 reference, so the
   error is that of the op's own fp8 compute / output quantization (expected a few percent; a layout bug shows as ~100%). */
#include "nn.h"
#include "nn_lp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint64_t rs = 12345;
static float frand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 11) * 0x1.0p-53 * 2 - 1); }
static int bad;
static float *dev_rand(size_t n, float scale) { float *h = malloc(n * 4); for (size_t i = 0; i < n; i++) h[i] = frand() * scale; float *d = nn_malloc(n * 4); nn_h2d(d, h, n * 4); free(h); return d; }
static float *dev_zero(size_t n) { float *d = nn_malloc(n * 4); nn_zero(d, n * 4); return d; }
static void *mx_new(shape5 s) { size_t b = nn_mx8_bytes(s); void *p = nn_malloc(b); nn_zero(p, b); nn_set_storage(p, b, 8); return p; }
static void *mx_from(const float *x, shape5 s) { void *p = mx_new(s); lp_f32_to_mx8(x, s.n, s.c, shape_spatial(s), p); return p; }
static float *deq(const void *p, shape5 s) { float *d = dev_zero(shape_numel(s)); lp_mx8_to_f32(p, s.n, s.c, shape_spatial(s), d); return d; }
static void cmp(const char *what, const float *a, const float *r, size_t n, double tol) {
    float *ha = malloc(n * 4), *hr = malloc(n * 4);
    nn_d2h(ha, a, n * 4); nn_d2h(hr, r, n * 4);
    double d2 = 0, r2 = 0;
    for (size_t i = 0; i < n; i++) { double d = (double)ha[i] - hr[i]; d2 += d * d; r2 += (double)hr[i] * hr[i]; }
    double e = sqrt(d2 / (r2 > 0 ? r2 : 1e-300));
    printf("  %-44s rel err %.3g%s\n", what, e, e < tol ? "" : "  FAIL");
    if (!(e < tol)) bad++;
    free(ha); free(hr);
}
static void mode_ref(void) { nn_set_tf32(0); }
static void mode_mx(void) { nn_set_prec(2); }
int main(void) {
    nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0);
    nn_set_act_bf16(0); nn_set_grad_bf16(0);   /* non-MX operands are fp32 */
    const double TOL = 0.08;
    const int N = 2, G = 8;
    {   /* upsample forward / backward */
        shape5 xs = {N, 32, 6, 6, 6}, ys = {N, 32, 12, 12, 12};
        float *x = dev_rand(shape_numel(xs), 1.f); void *xm = mx_from(x, xs); float *xd = deq(xm, xs);
        float *yr = dev_zero(shape_numel(ys)); void *ym = mx_new(ys);
        mode_ref(); nn_up2_fwd_into(xd, xs, yr, 32, 0);
        mode_mx(); nn_up2_fwd_into(xm, xs, ym, 32, 0);
        cmp("up2 fwd", deq(ym, ys), yr, shape_numel(ys), TOL);
        float *gy = dev_rand(shape_numel(ys), 1e-3f); void *gym = mx_from(gy, ys); float *gyd = deq(gym, ys);
        float *gxr = dev_zero(shape_numel(xs)); void *gxm = mx_new(xs);
        mode_ref(); nn_up2_bwd(gyd, xs, gxr);
        mode_mx(); nn_up2_bwd(gym, xs, gxm);
        cmp("up2 bwd", deq(gxm, xs), gxr, shape_numel(xs), TOL);
    }
    {   /* GroupNorm + SiLU apply / backward */
        shape5 s = {N, 16, 10, 10, 10};
        size_t n = shape_numel(s), NG = (size_t)N * G;
        float *x = dev_rand(n, 2.f), *gam = dev_rand(16, 1.f), *bet = dev_rand(16, 0.5f), *mean = dev_rand(NG, 0.2f), *rstd = dev_rand(NG, 0.3f);
        { float h[64]; nn_d2h(h, rstd, NG * 4); for (size_t i = 0; i < NG; i++) h[i] = 0.8f + fabsf(h[i]); nn_h2d(rstd, h, NG * 4); }
        void *xm = mx_from(x, s); float *xd = deq(xm, s);
        float *yr = dev_zero(n); void *ym = mx_new(s);
        mode_ref(); nn_gn_silu_apply(xd, s, G, gam, bet, mean, rstd, yr);
        mode_mx(); nn_gn_silu_apply(xm, s, G, gam, bet, mean, rstd, ym);
        cmp("gn+silu apply", deq(ym, s), yr, n, TOL);
        float *gy = dev_rand(n, 1e-3f); void *gym = mx_from(gy, s); float *gyd = deq(gym, s);
        float *gxr = dev_zero(n), *gg1 = dev_zero(16), *gb1 = dev_zero(16), *gg2 = dev_zero(16), *gb2 = dev_zero(16), *scr = nn_malloc(nn_gn_scratch(s) + 4096);
        void *gxm = mx_new(s);
        mode_ref(); nn_gn_silu_bwd(xd, s, G, gam, bet, mean, rstd, gyd, gxr, gg1, gb1, scr);
        mode_mx(); nn_gn_silu_bwd(xm, s, G, gam, bet, mean, rstd, gym, gxm, gg2, gb2, scr);
        cmp("gn+silu bwd gx (MX x, MX gy, MX gx)", deq(gxm, s), gxr, n, TOL);
        cmp("gn+silu bwd ggamma", gg2, gg1, 16, 1e-4);
        cmp("gn+silu bwd gbeta", gb2, gb1, 16, 1e-4);
    }
    {   /* stride-1 convs: fused gn forward with output stats, split forward, backward-data (+ split), weight gradients */
        shape5 xs = {N, 16, 12, 12, 16}, ys = xs; ys.c = 32;
        size_t nx = shape_numel(xs), ny = shape_numel(ys), NG = (size_t)N * G;
        float *x = dev_rand(nx, 2.f), *w = dev_rand((size_t)32 * 16 * 27, 0.1f), *b = dev_rand(32, 0.1f);
        float *gam = dev_rand(16, 1.f), *bet = dev_rand(16, 0.5f), *mean = dev_rand(NG, 0.2f), *rstd = dev_rand(NG, 0.3f);
        { float h[64]; nn_d2h(h, rstd, NG * 4); for (size_t i = 0; i < NG; i++) h[i] = 0.8f + fabsf(h[i]); nn_h2d(rstd, h, NG * 4); }
        void *xm = mx_from(x, xs); float *xd = deq(xm, xs);
        float *yr = dev_zero(ny), *m1 = dev_zero(NG), *r1 = dev_zero(NG), *m2 = dev_zero(NG), *r2 = dev_zero(NG);
        void *ym = mx_new(ys);
        mode_ref(); { float *t = dev_zero(nx); nn_gn_silu_apply(xd, xs, G, gam, bet, mean, rstd, t); nn_conv3d_fwd(t, xs, w, b, 32, 3, 1, yr); nn_gn_fwd(yr, ys, G, 1e-5f, nullptr, nullptr, nullptr, m1, r1); }
        mode_mx(); nn_conv3d_fwd_gn_stats(xm, xs, G, gam, bet, mean, rstd, w, b, 32, (float *)ym, G, 1e-5f, m2, r2);
        cmp("conv fwd gn+silu input (MX in/out)", deq(ym, ys), yr, ny, TOL);
        cmp("conv fwd output GN mean", m2, m1, NG, TOL);
        cmp("conv fwd output GN rstd", r2, r1, NG, TOL);
        /* split forward: x (32 ch MX) + x2 (16 ch MX) -> 16 ch */
        shape5 cs = xs; cs.c = 48; shape5 c1 = xs; c1.c = 32; shape5 o16 = xs; o16.c = 16;
        float *xa = dev_rand(shape_numel(c1), 1.f); void *xam = mx_from(xa, c1); float *xad = deq(xam, c1);
        float *w48 = dev_rand((size_t)16 * 48 * 27, 0.1f), *cat = dev_zero(shape_numel(cs)), *ysr = dev_zero(shape_numel(o16));
        void *ysm = mx_new(o16);
        mode_ref(); nn_concat_fwd(xad, 32, xd, 16, cs, cat); nn_conv3d_fwd(cat, cs, w48, nullptr, 16, 3, 1, ysr);
        mode_mx(); nn_conv3d_fwd_split(xam, xm, 32, cs, 0, nullptr, nullptr, nullptr, nullptr, w48, nullptr, 16, (float *)ysm, 0, 1e-5f, nullptr, nullptr);
        cmp("conv fwd split input (32 + 16, MX)", deq(ysm, o16), ysr, shape_numel(o16), TOL);
        /* backward-data split: gy (16 MX) -> gx (32 MX) + gx2 (16 MX) */
        float *gy = dev_rand(shape_numel(o16), 1e-3f); void *gym = mx_from(gy, o16); float *gyd = deq(gym, o16);
        float *gxr = dev_zero(shape_numel(cs)), *scr = nn_malloc(nn_conv3d_scratch(cs, 16, 3) + 4096), *ga = dev_zero(shape_numel(c1)), *gb_ = dev_zero(nx);
        void *g1m = mx_new(c1), *g2m = mx_new(xs);
        mode_ref(); nn_conv3d_bwd_data(gyd, o16, w48, cs, 3, 1, gxr, scr); nn_concat_bwd(gxr, 32, 16, cs, ga, gb_);
        mode_mx(); nn_conv3d_bwd_data_split(gym, o16, w48, cs, (float *)g1m, (float *)g2m, 32, scr);
        cmp("bwd_data split part 1 (MX)", deq(g1m, c1), ga, shape_numel(c1), TOL);
        cmp("bwd_data split part 2 (MX)", deq(g2m, xs), gb_, nx, TOL);
        /* weight gradients: gn input (MX x, MX gy), split input */
        float *gw1 = dev_zero((size_t)32 * 16 * 27), *gw2 = dev_zero((size_t)32 * 16 * 27), *gb1 = dev_zero(32), *gb2 = dev_zero(32);
        float *gy32 = dev_rand(ny, 1e-3f); void *gy32m = mx_from(gy32, ys); float *gy32d = deq(gy32m, ys);
        mode_ref(); { float *t = dev_zero(nx); nn_gn_silu_apply(xd, xs, G, gam, bet, mean, rstd, t); nn_conv3d_bwd_weight(t, xs, gy32d, ys, 3, 1, gw1, gb1); }
        mode_mx(); nn_conv3d_bwd_weight_gn(xm, xs, G, gam, bet, mean, rstd, gy32m, ys, gw2, gb2);
        cmp("bwd_weight gn input (MX x, MX gy)", gw2, gw1, (size_t)32 * 16 * 27, TOL);
        cmp("bwd_weight bias", gb2, gb1, 32, TOL);
        float *gw3 = dev_zero((size_t)16 * 48 * 27), *gw4 = dev_zero((size_t)16 * 48 * 27);
        mode_ref(); nn_conv3d_bwd_weight(cat, cs, gyd, o16, 3, 1, gw3, nullptr);
        mode_mx(); nn_conv3d_bwd_weight_split(xam, xm, 32, cs, 0, nullptr, nullptr, nullptr, nullptr, gym, o16, gw4, nullptr);
        cmp("bwd_weight split input (MX)", gw4, gw3, (size_t)16 * 48 * 27, TOL);
    }
    {   /* split with a segment that is not a multiple of 32 channels (the network's dec2: 80 + 64 -> 64) */
        shape5 cs = {N, 144, 8, 8, 12}, c1 = cs, c2 = cs, ys = cs; c1.c = 80; c2.c = 64; ys.c = 64;
        float *xa = dev_rand(shape_numel(c1), 1.f), *xb = dev_rand(shape_numel(c2), 1.f);
        void *xam = mx_from(xa, c1), *xbm = mx_from(xb, c2); float *xad = deq(xam, c1), *xbd = deq(xbm, c2);
        float *w = dev_rand((size_t)64 * 144 * 27, 0.05f), *cat = dev_zero(shape_numel(cs)), *yr = dev_zero(shape_numel(ys));
        void *ym = mx_new(ys);
        mode_ref(); nn_concat_fwd(xad, 80, xbd, 64, cs, cat); nn_conv3d_fwd(cat, cs, w, NULL, 64, 3, 1, yr);
        mode_mx(); nn_conv3d_fwd_split(xam, xbm, 80, cs, 0, NULL, NULL, NULL, NULL, w, NULL, 64, (float *)ym, 0, 1e-5f, NULL, NULL);
        cmp("conv fwd split 80 + 64 (MX)", deq(ym, ys), yr, shape_numel(ys), TOL);
        float *gy = dev_rand(shape_numel(ys), 1e-3f); void *gym = mx_from(gy, ys); float *gyd = deq(gym, ys);
        float *gxr = dev_zero(shape_numel(cs)), *scr = nn_malloc(nn_conv3d_scratch(cs, 64, 3) + 4096), *ga = dev_zero(shape_numel(c1)), *gb_ = dev_zero(shape_numel(c2));
        void *g1m = mx_new(c1), *g2m = mx_new(c2);
        mode_ref(); nn_conv3d_bwd_data(gyd, ys, w, cs, 3, 1, gxr, scr); nn_concat_bwd(gxr, 80, 64, cs, ga, gb_);
        mode_mx(); nn_conv3d_bwd_data_split(gym, ys, w, cs, (float *)g1m, (float *)g2m, 80, scr);
        cmp("bwd_data split 80 + 64, part 1 (MX)", deq(g1m, c1), ga, shape_numel(c1), TOL);
        cmp("bwd_data split 80 + 64, part 2 (MX)", deq(g2m, c2), gb_, shape_numel(c2), TOL);
        float *gw3 = dev_zero((size_t)64 * 144 * 27), *gw4 = dev_zero((size_t)64 * 144 * 27);
        mode_ref(); nn_conv3d_bwd_weight(cat, cs, gyd, ys, 3, 1, gw3, NULL);
        mode_mx(); nn_conv3d_bwd_weight_split(xam, xbm, 80, cs, 0, NULL, NULL, NULL, NULL, gym, ys, gw4, NULL);
        cmp("bwd_weight split 80 + 64 (MX)", gw4, gw3, (size_t)64 * 144 * 27, TOL);
    }
    {   /* stride 2: forward, backward-data (set and accumulate), weight gradient */
        shape5 xs = {N, 32, 16, 16, 16}, ys = {N, 32, 8, 8, 8};
        size_t nx = shape_numel(xs), ny = shape_numel(ys);
        float *x = dev_rand(nx, 1.f), *w = dev_rand((size_t)32 * 32 * 27, 0.1f), *b = dev_rand(32, 0.1f);
        void *xm = mx_from(x, xs); float *xd = deq(xm, xs);
        float *yr = dev_zero(ny); void *ym = mx_new(ys);
        mode_ref(); nn_conv3d_fwd(xd, xs, w, b, 32, 3, 2, yr);
        mode_mx(); nn_conv3d_fwd(xm, xs, w, b, 32, 3, 2, ym);
        cmp("s2 fwd (MX)", deq(ym, ys), yr, ny, TOL);
        float *gy = dev_rand(ny, 1e-3f); void *gym = mx_from(gy, ys); float *gyd = deq(gym, ys);
        float *scr = nn_malloc(nn_conv3d_scratch(xs, 32, 3) * 2 + (size_t)xs.n * 32 * shape_spatial(xs) * 4), *gxr = dev_zero(nx);
        void *gxm = mx_new(xs);
        mode_ref(); nn_conv3d_bwd_data(gyd, ys, w, xs, 3, 2, gxr, scr);
        mode_mx(); nn_conv3d_bwd_data(gym, ys, w, xs, 3, 2, (float *)gxm, scr);
        cmp("s2 bwd_data (MX)", deq(gxm, xs), gxr, nx, TOL);
        float *gx0 = dev_rand(nx, 1e-3f); void *gx0m = mx_from(gx0, xs); float *gx0d = deq(gx0m, xs);
        mode_ref(); nn_axpy(gxr, 1.f, gx0d, nx);
        mode_mx(); nn_conv3d_bwd_data_acc(gym, ys, w, xs, 3, 2, (float *)gx0m, scr);
        cmp("s2 bwd_data accumulate (MX)", deq(gx0m, xs), gxr, nx, TOL);
        float *gw1 = dev_zero((size_t)32 * 32 * 27), *gw2 = dev_zero((size_t)32 * 32 * 27), *gb1 = dev_zero(32), *gb2 = dev_zero(32);
        mode_ref(); nn_conv3d_bwd_weight(xd, xs, gyd, ys, 3, 2, gw1, gb1);
        mode_mx(); nn_conv3d_bwd_weight(xm, xs, gym, ys, 3, 2, gw2, gb2);
        cmp("s2 bwd_weight (MX x, MX gy)", gw2, gw1, (size_t)32 * 32 * 27, TOL);
        cmp("s2 bias grad", gb2, gb1, 32, TOL);
    }
    {   /* head: 1^3 conv forward, weight gradient, backward-data into MX */
        shape5 xs = {N, 16, 10, 10, 10}, ys = xs; ys.c = 1;
        size_t nx = shape_numel(xs), ny = shape_numel(ys);
        float *x = dev_rand(nx, 1.f), *w = dev_rand(16, 0.3f), *b = dev_rand(1, 0.1f);
        void *xm = mx_from(x, xs); float *xd = deq(xm, xs);
        float *yr = dev_zero(ny), *ym = dev_zero(ny);
        mode_ref(); nn_conv3d_fwd(xd, xs, w, b, 1, 1, 1, yr);
        mode_mx(); nn_conv3d_fwd(xm, xs, w, b, 1, 1, 1, ym);
        cmp("head fwd (MX x, fp32 logits)", ym, yr, ny, 1e-5);
        float *gy = dev_rand(ny, 1e-3f), *gw1 = dev_zero(16), *gw2 = dev_zero(16), *gb1 = dev_zero(1), *gb2 = dev_zero(1);
        mode_ref(); nn_conv3d_bwd_weight(xd, xs, gy, ys, 1, 1, gw1, gb1);
        mode_mx(); nn_conv3d_bwd_weight(xm, xs, gy, ys, 1, 1, gw2, gb2);
        cmp("head bwd_weight (MX x)", gw2, gw1, 16, 1e-4);
        float *scr = nn_malloc(4096), *gxr = dev_zero(nx); void *gxm = mx_new(xs);
        mode_ref(); nn_conv3d_bwd_data(gy, ys, w, xs, 1, 1, gxr, scr);
        mode_mx(); nn_conv3d_bwd_data(gy, ys, w, xs, 1, 1, (float *)gxm, scr);
        cmp("head bwd_data (fp32 gy -> MX gx)", deq(gxm, xs), gxr, nx, TOL);
    }
    const char *e = nn_check(); if (e) { printf("cuda: %s\n", e); bad++; }
    printf(bad ? "mx FAIL (%d)\n" : "mx ok\n", bad);
    return bad != 0;
}
