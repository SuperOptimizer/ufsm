/* input-side recompute ops (nn_conv3d_fwd_x / nn_conv3d_bwd_weight_x) against the same convs on materialised inputs
   (gn+silu applied, upsampled, concatenated). Run per compute precision on the tensor-core path with fp32 storage. */
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
static uint64_t rs = 777;
static float frand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 11) * 0x1.0p-53 * 2 - 1); }
static int bad;
static float *dev_rand(size_t n, float sc, float off) { float *h = malloc(n * 4); for (size_t i = 0; i < n; i++) h[i] = off + frand() * sc; float *d = nn_malloc(n * 4); nn_h2d(d, h, n * 4); free(h); return d; }
static float *dev_zero(size_t n) { float *d = nn_malloc(n * 4); nn_zero(d, n * 4); return d; }
static void cmp(const char *what, const float *a, const float *r, size_t n, double tol) {
    float *ha = malloc(n * 4), *hr = malloc(n * 4);
    nn_d2h(ha, a, n * 4); nn_d2h(hr, r, n * 4);
    double d2 = 0, r2 = 0;
    for (size_t i = 0; i < n; i++) { double d = (double)ha[i] - hr[i]; d2 += d * d; r2 += (double)hr[i] * hr[i]; }
    double e = sqrt(d2 / (r2 > 0 ? r2 : 1e-300));
    printf("  %-40s rel err %.3g%s\n", what, e, e < tol ? "" : "  FAIL");
    if (!(e < tol)) bad++;
    free(ha); free(hr);
}
static nn_gn_t mkgn(int N, int C, int G) { nn_gn_t g = {dev_rand(C, 0.5f, 1.f), dev_rand(C, 0.5f, 0.f), dev_rand((size_t)N * G, 0.3f, 0.f), dev_rand((size_t)N * G, 0.2f, 1.f), G}; return g; }
static float *mat(const float *x, shape5 s, const nn_gn_t *g) { float *y = dev_zero(shape_numel(s)); nn_gn_silu_apply(x, s, g->G, g->gamma, g->beta, g->mean, g->rstd, y); return y; }
int main(void) {
    nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0);
    nn_set_tf32(1); nn_set_act_bf16(0); nn_set_grad_bf16(0);
    const int precs[] = {1, 4, 2};
    for (int pi = 0; pi < 3; pi++) {
        nn_set_prec(precs[pi]);
        const double tol = precs[pi] == 2 ? 0.08 : 0.02;
        printf("prec %d\n", precs[pi]);
        const int N = 2, G = 8;
        /* decoder conv1: [up2(silu(gn(xc))) (32 ch, half res) | silu(gn(xs)) (16 ch)] -> 16 */
        shape5 fs = {N, 48, 16, 12, 20}, cs = {N, 32, 8, 6, 10}, ss = {N, 16, 16, 12, 20}, us = {N, 32, 16, 12, 20}, ys = {N, 16, 16, 12, 20};
        float *xc = dev_rand(shape_numel(cs), 2.f, 0.f), *xk = dev_rand(shape_numel(ss), 2.f, 0.f);
        nn_gn_t g1 = mkgn(N, 32, G), g2 = mkgn(N, 16, G);
        float *w = dev_rand((size_t)16 * 48 * 27, 0.1f, 0.f), *b = dev_rand(16, 0.1f, 0.f);
        float *up = dev_zero(shape_numel(us)), *t1 = mat(xc, cs, &g1), *t2 = mat(xk, ss, &g2);
        nn_up2_fwd_into(t1, cs, up, 32, 0);
        float *yr = dev_zero(shape_numel(ys)), *yx = dev_zero(shape_numel(ys));
        float *m1 = dev_zero(N * G), *r1 = dev_zero(N * G), *m2 = dev_zero(N * G), *r2 = dev_zero(N * G);
        nn_conv3d_fwd_split(up, t2, 32, fs, 0, NULL, NULL, NULL, NULL, w, b, 16, yr, G, 1e-5f, m1, r1);
        float *upx = dev_zero(shape_numel(us));
        nn_up2_fwd_gn_into(xc, cs, &g1, upx, 32, 0);
        cmp("up2 of silu(gn(x))", upx, up, shape_numel(us), 1e-5);
        if (nn_conv3d_fwd_x(upx, NULL, xk, &g2, 32, 0, fs, w, b, 16, 3, 1, yx, G, 1e-5f, m2, r2)) { printf("fwd_x unsupported\n"); return 1; }
        cmp("split fwd, gn on x2", yx, yr, shape_numel(ys), tol);
        cmp("  output GN mean", m2, m1, N * G, tol);
        float *gy = dev_rand(shape_numel(ys), 1.f, 0.f), *gwr = dev_zero((size_t)16 * 48 * 27), *gwx = dev_zero((size_t)16 * 48 * 27), *gbr = dev_zero(16), *gbx = dev_zero(16);
        nn_conv3d_bwd_weight_split(up, t2, 32, fs, 0, NULL, NULL, NULL, NULL, gy, ys, gwr, gbr);
        nn_conv3d_bwd_weight_x(upx, NULL, xk, &g2, 32, 0, fs, gy, ys, 3, 1, gwx, gbx);
        cmp("split wgrad, gn on x2", gwx, gwr, (size_t)16 * 48 * 27, tol);
        {   /* fused upsample: x = coarse tensor read as its trilinear upsample, x2 with gn */
            float *upr = dev_zero(shape_numel(us)), *y5 = dev_zero(shape_numel(ys)), *y6 = dev_zero(shape_numel(ys));
            float *m5 = dev_zero(N * G), *r5 = dev_zero(N * G), *m6 = dev_zero(N * G), *r6 = dev_zero(N * G);
            nn_up2_fwd_into(xc, cs, upr, 32, 0);
            const double ftol = precs[pi] == 2 ? tol : tol / 4;   /* the fused path always runs the 16-bit kernels; the reference at prec 2 is fp8 */
            nn_conv3d_fwd_x(upr, NULL, xk, &g2, 32, 0, fs, w, b, 16, 3, 1, y5, G, 1e-5f, m5, r5);
            if (nn_conv3d_fwd_x(xc, NULL, xk, &g2, 32, 1, fs, w, b, 16, 3, 1, y6, G, 1e-5f, m6, r6)) {
                printf("  fused upsample unsupported  FAIL\n"); bad++;
            } else {
            cmp("split fwd, fused upsample", y6, y5, shape_numel(ys), ftol);
            cmp("  output GN rstd", r6, r5, N * G, ftol);
            float *w5 = dev_zero((size_t)16 * 48 * 27), *w6 = dev_zero((size_t)16 * 48 * 27), *b5 = dev_zero(16), *b6 = dev_zero(16);
            nn_conv3d_bwd_weight_x(upr, NULL, xk, &g2, 32, 0, fs, gy, ys, 3, 1, w5, b5);
            nn_conv3d_bwd_weight_x(xc, NULL, xk, &g2, 32, 1, fs, gy, ys, 3, 1, w6, b6);
            cmp("split wgrad, fused upsample", w6, w5, (size_t)16 * 48 * 27, ftol);
            }
        }
        {   /* both segments transformed (x at full resolution) */
            float *xf = dev_rand(shape_numel(us), 2.f, 0.f), *tf = mat(xf, us, &g1), *y3 = dev_zero(shape_numel(ys)), *y4 = dev_zero(shape_numel(ys));
            nn_conv3d_fwd_split(tf, t2, 32, fs, 0, NULL, NULL, NULL, NULL, w, b, 16, y3, 0, 0.f, NULL, NULL);
            nn_conv3d_fwd_x(xf, &g1, xk, &g2, 32, 0, fs, w, b, 16, 3, 1, y4, 0, 0.f, NULL, NULL);
            cmp("split fwd, gn both", y4, y3, shape_numel(ys), tol);
            float *w3 = dev_zero((size_t)16 * 48 * 27), *w4 = dev_zero((size_t)16 * 48 * 27);
            nn_conv3d_bwd_weight_split(tf, t2, 32, fs, 0, NULL, NULL, NULL, NULL, gy, ys, w3, NULL);
            nn_conv3d_bwd_weight_x(xf, &g1, xk, &g2, 32, 0, fs, gy, ys, 3, 1, w4, NULL);
            cmp("split wgrad, gn both", w4, w3, (size_t)16 * 48 * 27, tol);
        }
        cmp("  bias", gbx, gbr, 16, 1e-4);
        /* down conv: s2 of silu(gn(x)) */
        shape5 xs = {N, 32, 16, 12, 20};
        float *x = dev_rand(shape_numel(xs), 2.f, 0.f); nn_gn_t g3 = mkgn(N, 32, G); float *t3 = mat(x, xs, &g3);
        float *w2 = dev_rand((size_t)32 * 32 * 27, 0.1f, 0.f), *b2 = dev_rand(32, 0.1f, 0.f);
        shape5 ds = nn_conv3d_out_shape(xs, 32, 3, 2);
        float *dr = dev_zero(shape_numel(ds)), *dx = dev_zero(shape_numel(ds));
        nn_conv3d_fwd(t3, xs, w2, b2, 32, 3, 2, dr);
        nn_conv3d_fwd_x(x, &g3, NULL, NULL, 0, 0, xs, w2, b2, 32, 3, 2, dx, 0, 0.f, NULL, NULL);
        cmp("s2 fwd, gn", dx, dr, shape_numel(ds), tol);
        float *gd = dev_rand(shape_numel(ds), 1.f, 0.f), *gw2r = dev_zero((size_t)32 * 32 * 27), *gw2x = dev_zero((size_t)32 * 32 * 27);
        nn_conv3d_bwd_weight(t3, xs, gd, ds, 3, 2, gw2r, NULL);
        nn_conv3d_bwd_weight_x(x, &g3, NULL, NULL, 0, 0, xs, gd, ds, 3, 2, gw2x, NULL);
        cmp("s2 wgrad, gn", gw2x, gw2r, (size_t)32 * 32 * 27, tol);
        /* head */
        shape5 hs = {N, 16, 16, 12, 20}, os = hs; os.c = 1;
        float *xh = dev_rand(shape_numel(hs), 2.f, 0.f); nn_gn_t g4 = mkgn(N, 16, G); float *t4 = mat(xh, hs, &g4);
        float *wh = dev_rand(16, 0.3f, 0.f), *bh = dev_rand(1, 0.1f, 0.f), *hr = dev_zero(shape_numel(os)), *hx = dev_zero(shape_numel(os));
        nn_conv3d_fwd(t4, hs, wh, bh, 1, 1, 1, hr);
        nn_conv3d_fwd_x(xh, &g4, NULL, NULL, 0, 0, hs, wh, bh, 1, 1, 1, hx, 0, 0.f, NULL, NULL);
        cmp("head fwd, gn", hx, hr, shape_numel(os), 1e-4);
        float *go = dev_rand(shape_numel(os), 1.f, 0.f), *ghr = dev_zero(16), *ghx = dev_zero(16);
        nn_conv3d_bwd_weight(t4, hs, go, os, 1, 1, ghr, NULL);
        nn_conv3d_bwd_weight_x(xh, &g4, NULL, NULL, 0, 0, hs, go, os, 1, 1, ghx, NULL);
        cmp("head wgrad, gn", ghx, ghr, 16, 1e-4);
    }
    const char *e = nn_check(); if (e) { printf("cuda: %s\n", e); bad++; }
    printf(bad ? "rc FAIL (%d)\n" : "rc ok\n", bad);
    return bad != 0;
}
