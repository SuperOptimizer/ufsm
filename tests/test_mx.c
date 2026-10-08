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
    {   /* the two-taps-per-k 16-channel fp8 kernel on fp32 I/O (the fp32 tap-packed small kernel is switched off for it) */
        setenv("UFSM_F8_NOSMALL", "1", 1);
        const int cis[3] = {16, 12, 16}, cos_[3] = {16, 16, 64}, ds[3] = {16, 12, 8};
        for (int k = 0; k < 3; k++) {
            shape5 xs = {N, cis[k], ds[k], 12, 20}, ys = xs; ys.c = cos_[k];
            size_t nx = shape_numel(xs), ny = shape_numel(ys);
            float *x = dev_rand(nx, 1.f), *w = dev_rand((size_t)cos_[k] * cis[k] * 27, 0.1f), *b = dev_rand(cos_[k], 0.1f), *yr = dev_zero(ny), *y8 = dev_zero(ny);
            mode_ref(); nn_conv3d_fwd(x, xs, w, b, cos_[k], 3, 1, yr);
            nn_conv3d_fwd_fp8(x, xs, w, b, cos_[k], y8);
            char nm[80]; snprintf(nm, sizeof nm, "fp8 conv fwd %d -> %d ch, two taps per k, D %d", cis[k], cos_[k], ds[k]);
            cmp(nm, y8, yr, ny, 0.06);
        }
    }
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
    for (int C = 192; C <= 256; C += 64) {   /* GroupNorm + SiLU backward on wide MX tensors (6-level nets: 192 / 256 channels) */
        shape5 s = {N, C, 6, 6, 8};
        size_t n = shape_numel(s), NG = (size_t)N * G;
        float *x = dev_rand(n, 2.f), *gam = dev_rand(C, 1.f), *bet = dev_rand(C, 0.5f), *mean = dev_rand(NG, 0.2f), *rstd = dev_rand(NG, 0.3f);
        { float h[64]; nn_d2h(h, rstd, NG * 4); for (size_t i = 0; i < NG; i++) h[i] = 0.8f + fabsf(h[i]); nn_h2d(rstd, h, NG * 4); }
        void *xm = mx_from(x, s); float *xd = deq(xm, s);
        float *gy = dev_rand(n, 1e-3f); void *gym = mx_from(gy, s); float *gyd = deq(gym, s);
        float *gxr = dev_zero(n), *gg1 = dev_zero(C), *gb1 = dev_zero(C), *gg2 = dev_zero(C), *gb2 = dev_zero(C), *scr = nn_malloc(nn_gn_scratch(s) + 4096);
        void *gxm = mx_new(s);
        mode_ref(); nn_gn_silu_bwd(xd, s, G, gam, bet, mean, rstd, gyd, gxr, gg1, gb1, scr);
        mode_mx(); nn_gn_silu_bwd(xm, s, G, gam, bet, mean, rstd, gym, gxm, gg2, gb2, scr);
        char nm[64];
        snprintf(nm, sizeof nm, "gn+silu bwd gx, %d channels", C); cmp(nm, deq(gxm, s), gxr, n, TOL);
        snprintf(nm, sizeof nm, "gn+silu bwd ggamma, %d channels", C); cmp(nm, gg2, gg1, C, 1e-4);
        snprintf(nm, sizeof nm, "gn+silu bwd gbeta, %d channels", C); cmp(nm, gb2, gb1, C, 1e-4);
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
    for (int wc = 0; wc < 2; wc++) {   /* stride-2 backward-data of wide layers (dilated fp8 tensor-core path), set and accumulate */
        const int Ci = wc ? 160 : 96, Co = wc ? 128 : 96;
        shape5 xs = {N, Ci, 12, 10, 14}, ys = {N, Co, 6, 5, 7};
        size_t nx = shape_numel(xs), ny = shape_numel(ys);
        float *w = dev_rand((size_t)Co * Ci * 27, 0.05f), *gy = dev_rand(ny, 1e-3f);
        void *gym = mx_from(gy, ys); float *gyd = deq(gym, ys);
        float *scr = nn_malloc(nn_conv3d_scratch(xs, Co, 3) * 2 + (size_t)xs.n * Co * shape_spatial(xs) * 4), *gxr = dev_zero(nx);
        void *gxm = mx_new(xs);
        mode_ref(); nn_conv3d_bwd_data(gyd, ys, w, xs, 3, 2, gxr, scr);
        mode_mx(); nn_conv3d_bwd_data(gym, ys, w, xs, 3, 2, (float *)gxm, scr);
        char nm[96]; snprintf(nm, sizeof nm, "s2 bwd_data %d -> %d (MX, dilated fp8)", Co, Ci);
        cmp(nm, deq(gxm, xs), gxr, nx, TOL);
        float *gx0 = dev_rand(nx, 1e-3f); void *gx0m = mx_from(gx0, xs); float *gx0d = deq(gx0m, xs);
        mode_ref(); nn_axpy(gxr, 1.f, gx0d, nx);
        mode_mx(); nn_conv3d_bwd_data_acc(gym, ys, w, xs, 3, 2, (float *)gx0m, scr);
        snprintf(nm, sizeof nm, "s2 bwd_data %d -> %d accumulate (MX, dilated fp8)", Co, Ci);
        cmp(nm, deq(gx0m, xs), gxr, nx, TOL);
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
    {   /* decoder conv1 on MX-fp8 storage without the full-resolution transient: the up part (32 ch) is read from the coarse
           tensor and upsampled while staging, the skip (16 ch) gets gn+silu (dec conv1 up segment from coarse MX (no transient)) */
        shape5 fs = {N, 48, 12, 12, 16}, co = {N, 32, 6, 6, 8}, fu = fs, sk = fs, o16 = fs; fu.c = 32; sk.c = 16; o16.c = 16;
        size_t NG = (size_t)N * G;
        float *xc = dev_rand(shape_numel(co), 1.f), *xs_ = dev_rand(shape_numel(sk), 2.f);
        void *xcm = mx_from(xc, co), *xsm = mx_from(xs_, sk); float *xcd = deq(xcm, co), *xsd = deq(xsm, sk);
        float *gam = dev_rand(16, 1.f), *bet = dev_rand(16, 0.5f), *mean = dev_rand(NG, 0.2f), *rstd = dev_rand(NG, 0.3f);
        { float h[64]; nn_d2h(h, rstd, NG * 4); for (size_t i = 0; i < NG; i++) h[i] = 0.8f + fabsf(h[i]); nn_h2d(rstd, h, NG * 4); }
        float *w = dev_rand((size_t)16 * 48 * 27, 0.1f), *b = dev_rand(16, 0.1f);
        float *up = dev_zero(shape_numel(fu)), *t = dev_zero(shape_numel(sk)), *cat = dev_zero(shape_numel(fs)), *yr = dev_zero(shape_numel(o16));
        mode_ref(); nn_up2_fwd_into(xcd, co, up, 32, 0); nn_gn_silu_apply(xsd, sk, G, gam, bet, mean, rstd, t); nn_concat_fwd(up, 32, t, 16, fs, cat); nn_conv3d_fwd(cat, fs, w, b, 16, 3, 1, yr);
        nn_gn_t g2 = {gam, bet, mean, rstd, G};
        void *ym = mx_new(o16);
        mode_mx(); int rv = nn_conv3d_fwd_x(xcm, NULL, xsm, &g2, 32, 1, fs, w, b, 16, 3, 1, (float *)ym, 0, 1e-5f, NULL, NULL);
        if (rv) { printf("  dec conv1 with the up segment: unsupported (rv %d)  FAIL\n", rv); bad++; }
        else cmp("dec conv1 up segment from coarse MX (no transient)", deq(ym, o16), yr, shape_numel(o16), TOL);
    }
    {   /* GroupNorm with channels per group not dividing the block width (deeper nets: 96 / 112 channels, G = 8 -> 12 / 14) */
        const int Cs[2] = {96, 112};
        for (int k = 0; k < 2; k++) {
            shape5 s = {N, Cs[k], 6, 6, 8};
            size_t n = shape_numel(s), NG = (size_t)N * G;
            float *x = dev_rand(n, 2.f), *gam = dev_rand(Cs[k], 1.f), *bet = dev_rand(Cs[k], 0.5f);
            void *xm = mx_from(x, s); float *xd = deq(xm, s);
            float *m1 = dev_zero(NG), *r1 = dev_zero(NG), *m2 = dev_zero(NG), *r2 = dev_zero(NG);
            mode_ref(); nn_gn_stats(xd, s, G, 1e-5f, m1, r1);
            mode_mx(); int rv = nn_gn_stats(xm, s, G, 1e-5f, m2, r2);
            char nm[96];
            snprintf(nm, sizeof nm, "gn stats mean, %d ch / 8 groups (mx8)", Cs[k]); if (rv) { printf("  %s unsupported  FAIL\n", nm); bad++; } else cmp(nm, m2, m1, NG, 1e-4);
            snprintf(nm, sizeof nm, "gn stats rstd, %d ch / 8 groups (mx8)", Cs[k]); cmp(nm, r2, r1, NG, 1e-4);
            float *yr = dev_zero(n); void *ym = mx_new(s);
            mode_ref(); nn_gn_silu_apply(xd, s, G, gam, bet, m1, r1, yr);
            mode_mx(); nn_gn_silu_apply(xm, s, G, gam, bet, m1, r1, ym);
            snprintf(nm, sizeof nm, "gn+silu apply, %d ch / 8 groups (mx8)", Cs[k]); cmp(nm, deq(ym, s), yr, n, TOL);
            float *gy = dev_rand(n, 1e-3f), *gxr = dev_zero(n), *gx2 = dev_zero(n), *gg1 = dev_zero(Cs[k]), *gb1 = dev_zero(Cs[k]), *gg2 = dev_zero(Cs[k]), *gb2 = dev_zero(Cs[k]), *scr = nn_malloc(nn_gn_scratch(s) + 4096);
            mode_ref(); nn_gn_silu_bwd(xd, s, G, gam, bet, m1, r1, gy, gxr, gg1, gb1, scr);
            mode_mx(); nn_gn_silu_bwd(xm, s, G, gam, bet, m1, r1, gy, gx2, gg2, gb2, scr);
            snprintf(nm, sizeof nm, "gn+silu bwd gx, %d ch / 8 groups (mx8 x)", Cs[k]); cmp(nm, gx2, gxr, n, 1e-4);
            snprintf(nm, sizeof nm, "gn+silu bwd ggamma, %d ch / 8 groups (mx8 x)", Cs[k]); cmp(nm, gg2, gg1, Cs[k], 1e-4);
        }
    }
    {   /* stride 2 with 16 input channels (down0): two taps per K block, plain and gn+silu input (mx8 in / out) */
        shape5 xs = {N, 16, 16, 16, 24}, ys = {N, 16, 8, 8, 12};
        size_t nx = shape_numel(xs), ny = shape_numel(ys), NG = (size_t)N * G;
        float *x = dev_rand(nx, 1.f), *w = dev_rand((size_t)16 * 16 * 27, 0.1f), *b = dev_rand(16, 0.1f);
        void *xm = mx_from(x, xs); float *xd = deq(xm, xs);
        float *yr = dev_zero(ny); void *ym = mx_new(ys);
        mode_ref(); nn_conv3d_fwd(xd, xs, w, b, 16, 3, 2, yr);
        mode_mx(); nn_conv3d_fwd(xm, xs, w, b, 16, 3, 2, ym);
        cmp("s2 fwd 16 ch, two taps per k (MX)", deq(ym, ys), yr, ny, TOL);
        float *gam = dev_rand(16, 1.f), *bet = dev_rand(16, 0.5f), *mean = dev_rand(NG, 0.2f), *rstd = dev_rand(NG, 0.3f);
        { float h[64]; nn_d2h(h, rstd, NG * 4); for (size_t i = 0; i < NG; i++) h[i] = 0.8f + fabsf(h[i]); nn_h2d(rstd, h, NG * 4); }
        nn_gn_t g = {gam, bet, mean, rstd, G};
        float *t = dev_zero(nx), *yr2 = dev_zero(ny); void *ym2 = mx_new(ys);
        mode_ref(); nn_gn_silu_apply(xd, xs, G, gam, bet, mean, rstd, t); nn_conv3d_fwd(t, xs, w, b, 16, 3, 2, yr2);
        mode_mx(); nn_conv3d_fwd_x(xm, &g, NULL, NULL, 0, 0, xs, w, b, 16, 3, 2, (float *)ym2, 0, 0.f, NULL, NULL);
        cmp("s2 fwd 16 ch, gn+silu input (MX)", deq(ym2, ys), yr2, ny, TOL);
        float *y3 = dev_zero(ny), *x3 = dev_rand((size_t)N * 12 * 16 * 16 * 24, 1.f); shape5 x12 = xs; x12.c = 12;
        float *w3 = dev_rand((size_t)16 * 12 * 27, 0.1f), *yr3 = dev_zero(ny);
        mode_ref(); nn_conv3d_fwd(x3, x12, w3, NULL, 16, 3, 2, yr3);
        mode_mx(); lp_conv_fwd_s2_f8(x3, 0, x12, w3, NULL, 16, y3, 0, ys, (gnp_t){0});
        cmp("s2 fwd 12 ch fp32 I/O, two taps per k", y3, yr3, ny, TOL);
    }
    {   /* stride-2 weight gradient (down0 16 -> 16 / down2-like 64 -> 64): odd sizes, plain and gn+silu input (down_norm) */
        const int Cs[4] = {16, 64, 32, 16}, Ws[4] = {22, 22, 40, 24};   /* output width % 4 == 0: the kernel's FAST staging */
        for (int k = 0; k < 4; k++) {
            const int C = Cs[k];
            shape5 xs = {N, C, 13, 10, Ws[k]}, ys = nn_conv3d_out_shape(xs, C, 3, 2);
            size_t nx = shape_numel(xs), ny = shape_numel(ys), NG = (size_t)N * G, nw = (size_t)C * C * 27;
            float *x = dev_rand(nx, 1.f), *gy = dev_rand(ny, 1e-3f);
            void *xm = mx_from(x, xs), *gym = mx_from(gy, ys); float *xd = deq(xm, xs), *gyd = deq(gym, ys);
            float *gam = dev_rand(C, 1.f), *bet = dev_rand(C, 0.5f), *mean = dev_rand(NG, 0.2f), *rstd = dev_rand(NG, 0.3f);
            { float h[64]; nn_d2h(h, rstd, NG * 4); for (size_t i = 0; i < NG; i++) h[i] = 0.8f + fabsf(h[i]); nn_h2d(rstd, h, NG * 4); }
            nn_gn_t g = {gam, bet, mean, rstd, G};
            for (int gn = 0; gn < 2; gn++) {
                float *gw1 = dev_zero(nw), *gw2 = dev_zero(nw), *gb1 = dev_zero(C), *gb2 = dev_zero(C), *t = dev_zero(nx);
                mode_ref(); if (gn) nn_gn_silu_apply(xd, xs, G, gam, bet, mean, rstd, t); nn_conv3d_bwd_weight(gn ? t : xd, xs, gyd, ys, 3, 2, gw1, gb1);
                mode_mx(); nn_conv3d_bwd_weight_x(xm, gn ? &g : NULL, NULL, NULL, 0, 0, xs, gym, ys, 3, 2, gw2, gb2);
                char nm[96]; snprintf(nm, sizeof nm, "s2 bwd_weight %d -> %d, odd sizes, W %d%s (MX)", C, C, Ws[k], gn ? ", gn+silu input" : "");
                cmp(nm, gw2, gw1, nw, TOL);
                snprintf(nm, sizeof nm, "s2 bias grad %d, odd sizes, W %d%s (MX)", C, Ws[k], gn ? ", gn+silu input" : "");
                cmp(nm, gb2, gb1, C, TOL);
            }
        }
    }
    {   /* stride-2 backward-data on MX-fp8 gradients, other widths (down0 16 -> 16, odd sizes; down2-like 64 -> 64, accumulate) */
        const int cis[2] = {16, 64}, cos_[2] = {16, 64}, Ds[2] = {14, 8};
        for (int k = 0; k < 2; k++) {
            shape5 xs = {N, cis[k], Ds[k], 10, 18}, ys = nn_conv3d_out_shape(xs, cos_[k], 3, 2);
            size_t nx = shape_numel(xs), ny = shape_numel(ys);
            float *w = dev_rand((size_t)cos_[k] * cis[k] * 27, 0.1f), *gy = dev_rand(ny, 1e-3f); void *gym = mx_from(gy, ys); float *gyd = deq(gym, ys);
            float *scr = nn_malloc(nn_conv3d_scratch(xs, cos_[k], 3) * 2 + (size_t)xs.n * cos_[k] * shape_spatial(xs) * 4), *gxr = dev_zero(nx);
            float *gx0 = dev_rand(nx, 1e-3f); void *gxm = mx_from(gx0, xs); float *gx0d = deq(gxm, xs);
            mode_ref(); nn_conv3d_bwd_data(gyd, ys, w, xs, 3, 2, gxr, scr); if (k) nn_axpy(gxr, 1.f, gx0d, nx);
            mode_mx(); if (k) nn_conv3d_bwd_data_acc(gym, ys, w, xs, 3, 2, (float *)gxm, scr); else nn_conv3d_bwd_data(gym, ys, w, xs, 3, 2, (float *)gxm, scr);
            char nm[80]; snprintf(nm, sizeof nm, "s2 bwd_data MX %d -> %d%s", cos_[k], cis[k], k ? " (accumulate)" : " (odd D)");
            cmp(nm, deq(gxm, xs), gxr, nx, TOL);
        }
    }
    {   /* decoder conv1 with the up segment read from the coarse PRE-GN tensor: GN+SiLU of the coarse rows inside the up staging (no kept s2) */
        shape5 fs = {N, 48, 12, 12, 16}, co = {N, 32, 6, 6, 8}, fu = fs, sk = fs, o16 = fs; fu.c = 32; sk.c = 16; o16.c = 16;
        size_t NG = (size_t)N * G;
        float *xc = dev_rand(shape_numel(co), 2.f), *xs_ = dev_rand(shape_numel(sk), 2.f);
        void *xcm = mx_from(xc, co), *xsm = mx_from(xs_, sk); float *xcd = deq(xcm, co), *xsd = deq(xsm, sk);
        float *gam = dev_rand(16, 1.f), *bet = dev_rand(16, 0.5f), *mean = dev_rand(NG, 0.2f), *rstd = dev_rand(NG, 0.3f);
        float *gac = dev_rand(32, 1.f), *bec = dev_rand(32, 0.5f), *mec = dev_rand(NG, 0.2f), *rsc = dev_rand(NG, 0.3f);
        { float h[64]; nn_d2h(h, rstd, NG * 4); for (size_t i = 0; i < NG; i++) h[i] = 0.8f + fabsf(h[i]); nn_h2d(rstd, h, NG * 4); nn_d2h(h, rsc, NG * 4); for (size_t i = 0; i < NG; i++) h[i] = 0.8f + fabsf(h[i]); nn_h2d(rsc, h, NG * 4); }
        float *w = dev_rand((size_t)16 * 48 * 27, 0.1f), *b = dev_rand(16, 0.1f);
        float *tc = dev_zero(shape_numel(co)), *up = dev_zero(shape_numel(fu)), *t = dev_zero(shape_numel(sk)), *cat = dev_zero(shape_numel(fs)), *yr = dev_zero(shape_numel(o16));
        mode_ref(); nn_gn_silu_apply(xcd, co, G, gac, bec, mec, rsc, tc); nn_up2_fwd_into(tc, co, up, 32, 0); nn_gn_silu_apply(xsd, sk, G, gam, bet, mean, rstd, t); nn_concat_fwd(up, 32, t, 16, fs, cat); nn_conv3d_fwd(cat, fs, w, b, 16, 3, 1, yr);
        nn_gn_t gc = {gac, bec, mec, rsc, G}, g2 = {gam, bet, mean, rstd, G};
        void *ym = mx_new(o16);
        mode_mx(); int rv = nn_conv3d_fwd_x(xcm, &gc, xsm, &g2, 32, 1, fs, w, b, 16, 3, 1, (float *)ym, 0, 1e-5f, NULL, NULL);
        if (rv) { printf("  dec conv1 up segment with GN: unsupported (rv %d)  FAIL\n", rv); bad++; }
        else cmp("dec conv1 up segment from coarse pre-GN MX", deq(ym, o16), yr, shape_numel(o16), TOL);
    }
    const char *e = nn_check(); if (e) { printf("cuda: %s\n", e); bad++; }
    printf(bad ? "mx FAIL (%d)\n" : "mx ok\n", bad);
    return bad != 0;
}
