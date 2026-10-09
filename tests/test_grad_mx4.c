/* MX-fp4 gradient storage (opt-in): every backward op that reads or writes an activation gradient must accept MX-fp4
   gradients and store them with exact stochastic rounding. Checks per op: the error against the fp32 reference on the
   dequantised inputs (fp4 output rounding: tens of percent per element is expected), and unbiasedness: the mean of K
   independently rounded outputs must approach the reference ~1 / sqrt(K) (a biased rounding plateaus). */
#include "nn.h"
#include "nn_lp.h"
#include "unet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint64_t rs = 12345;
static float frand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 11) * 0x1.0p-53 * 2 - 1); }
static int bad;
static float *dev_rand(size_t n, float scale) { float *h = malloc(n * 4); for (size_t i = 0; i < n; i++) h[i] = frand() * scale; float *d = nn_malloc(n * 4); nn_h2d(d, h, n * 4); free(h); return d; }
static float *dev_rand_wide(size_t n, float scale) {   /* magnitudes spread over 2^-4 .. 2^4: MX block scales differ per voxel */
    float *h = malloc(n * 4); for (size_t i = 0; i < n; i++) h[i] = frand() * scale * exp2f(rintf(4.f * frand()));
    float *d = nn_malloc(n * 4); nn_h2d(d, h, n * 4); free(h); return d;
}
static float *dev_zero(size_t n) { float *d = nn_malloc(n * 4); nn_zero(d, n * 4); return d; }
static void *mx4_new(shape5 s) { size_t b = nn_mx4_bytes(s); void *p = nn_malloc(b); nn_zero(p, b); nn_set_storage(p, b, 4); return p; }
static void *mx4_from(const float *x, shape5 s) { void *p = mx4_new(s); lp_f32_to_mx4(x, s.n, s.c, shape_spatial(s), p); return p; }
static void deq4_into(const void *p, shape5 s, float *d) { lp_mx4_to_f32(p, s.n, s.c, shape_spatial(s), d); }
static float *deq4(const void *p, shape5 s) { float *d = dev_zero(shape_numel(s)); deq4_into(p, s, d); return d; }
static double relerr(const float *a, const float *r, size_t n) {
    float *ha = malloc(n * 4), *hr = malloc(n * 4);
    nn_d2h(ha, a, n * 4); nn_d2h(hr, r, n * 4);
    double d2 = 0, r2 = 0;
    for (size_t i = 0; i < n; i++) { double d = (double)ha[i] - hr[i]; d2 += d * d; r2 += (double)hr[i] * hr[i]; }
    free(ha); free(hr);
    return sqrt(d2 / (r2 > 0 ? r2 : 1e-300));
}
static void report(const char *what, double one, double avg, int K, double tol1) {
    /* unbiased: avg ~ one / sqrt(K) (+ the op's own deterministic error); require at least half the ideal reduction */
    const int ok = one < tol1 && avg < one * 2.0 / sqrt((double)K) + 0.01;
    printf("  %-46s one %.3g, mean of %d %.3g%s\n", what, one, K, avg, ok ? "" : "  FAIL");
    if (!ok) bad++;
}
static void mode_ref(void) { nn_set_tf32(0); }
static void mode_mx(void) { nn_set_prec(3); nn_set_tf32(1); }
int main(void) {
    nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0);
    nn_set_act_bf16(0); nn_set_grad_bf16(0);
    const int N = 2, G = 8, K = 16;
    for (int C = 16; C <= 64; C *= 2) {   /* GroupNorm + SiLU backward: gy and gx MX-fp4 */
        shape5 s = {N, C, 8, 10, 12};
        size_t n = shape_numel(s), NG = (size_t)N * G;
        float *x = dev_rand(n, 2.f), *gam = dev_rand(C, 1.f), *bet = dev_rand(C, 0.5f), *mean = dev_rand(NG, 0.2f), *rstd = dev_rand(NG, 0.3f);
        { float h[64]; nn_d2h(h, rstd, NG * 4); for (size_t i = 0; i < NG; i++) h[i] = 0.8f + fabsf(h[i]); nn_h2d(rstd, h, NG * 4); }
        void *xm = mx4_from(x, s); float *xd = deq4(xm, s);
        float *gy = dev_rand(n, 1e-3f); void *gym = mx4_from(gy, s); float *gyd = deq4(gym, s);
        float *gxr = dev_zero(n), *gg1 = dev_zero(C), *gb1 = dev_zero(C), *gg2 = dev_zero(C), *gb2 = dev_zero(C), *scr = nn_malloc(nn_gn_scratch(s) + 4096);
        mode_ref(); nn_gn_silu_bwd(xd, s, G, gam, bet, mean, rstd, gyd, gxr, gg1, gb1, scr);
        void *gxm = mx4_new(s); float *acc = dev_zero(n), *one = dev_zero(n), *tmp = dev_zero(n);
        mode_mx();
        for (int k = 0; k < K; k++) {
            nn_set_sr_step(1000 + k);
            nn_zero(gg2, C * 4); nn_zero(gb2, C * 4);
            nn_gn_silu_bwd(xm, s, G, gam, bet, mean, rstd, gym, gxm, gg2, gb2, scr);
            deq4_into(gxm, s, tmp);
            if (!k) nn_d2d(one, tmp, n * 4);
            nn_axpy(acc, 1.f / K, tmp, n);
        }
        char nm[96];
        snprintf(nm, sizeof nm, "gn+silu bwd gx MX-fp4, %d ch", C); report(nm, relerr(one, gxr, n), relerr(acc, gxr, n), K, 0.35);
        snprintf(nm, sizeof nm, "gn+silu bwd ggamma, %d ch", C);
        { double e = relerr(gg2, gg1, C); printf("  %-46s rel err %.3g%s\n", nm, e, e < 1e-3 ? "" : "  FAIL"); if (!(e < 1e-3)) bad++; }
    }
    {   /* stride-1 backward-data: gy MX-fp4 -> gx MX-fp4 (the fp4 conv kernels; gx with exact SR) */
        const int cis[3] = {32, 64, 32}, cos_[3] = {32, 32, 64};
        for (int k3 = 0; k3 < 3; k3++) {
            shape5 xs = {1, cis[k3], 8, 16, 32}, ys = xs; ys.c = cos_[k3];
            size_t nx = shape_numel(xs), ny = shape_numel(ys);
            float *w = dev_rand((size_t)cos_[k3] * cis[k3] * 27, 0.1f), *gy = dev_rand(ny, 1e-3f);
            void *gym = mx4_from(gy, ys); float *gyd = deq4(gym, ys);
            mode_ref();
            float *gxr = dev_zero(nx), *scr = nn_malloc(nn_conv3d_scratch(xs, cos_[k3], 3) + 4096); nn_conv3d_bwd_data(gyd, ys, w, xs, 3, 1, gxr, scr);
            void *gxm = mx4_new(xs); float *acc = dev_zero(nx), *one = dev_zero(nx), *tmp = dev_zero(nx);
            mode_mx();
            for (int k = 0; k < K; k++) {
                nn_set_sr_step(2000 + k);
                nn_conv3d_bwd_data(gym, ys, w, xs, 3, 1, (float *)gxm, scr);
                deq4_into(gxm, xs, tmp);
                if (!k) nn_d2d(one, tmp, nx * 4);
                nn_axpy(acc, 1.f / K, tmp, nx);
            }
            /* the same kernel with an fp32 gx (the fp4 weights' own deterministic error stays): the SR output must average to it */
            float *gxd = dev_zero(nx); nn_conv3d_bwd_data(gym, ys, w, xs, 3, 1, gxd, scr);
            char nm[96]; snprintf(nm, sizeof nm, "bwd_data %d -> %d MX-fp4 (vs fp32-out kernel)", cos_[k3], cis[k3]);
            report(nm, relerr(one, gxd, nx), relerr(acc, gxd, nx), K, 0.35);
            const double ek = relerr(gxd, gxr, nx);
            printf("  %-46s rel err %.3g%s\n", "  kernel (fp32 out) vs fp32 reference", ek, ek < 0.15 ? "" : "  FAIL"); if (!(ek < 0.15)) bad++;
        }
    }
    {   /* stride-1 weight gradient (fp4 kernels, gy pre-pass): MX-fp4 gy vs MX-fp8 gy vs fp32 */
        const int cis[2] = {32, 64}, cos_[2] = {32, 32};
        nn_set_sr(1); setenv("UFSM_F4_WGRAD", "1", 1);
        for (int k3 = 0; k3 < 2; k3++) {
            shape5 xs = {1, cis[k3], 16, 16, 32}, ys = xs; ys.c = cos_[k3];
            size_t nx = shape_numel(xs), ny = shape_numel(ys), nw = (size_t)cos_[k3] * cis[k3] * 27;
            float *x = dev_rand_wide(nx, 2.f), *gy = dev_rand_wide(ny, 1e-3f);
            void *xm = mx4_from(x, xs); float *xd = deq4(xm, xs);
            void *gy4 = mx4_from(gy, ys); float *gyd4 = deq4(gy4, ys);
            size_t b8 = nn_mx8_bytes(ys); void *gy8 = nn_malloc(b8); nn_set_storage(gy8, b8, 8); lp_f32_to_mx8(gyd4, ys.n, ys.c, shape_spatial(ys), gy8);   /* the same fp4 values in MX-fp8 (exact) */
            float *gwr = dev_zero(nw), *gbr = dev_zero(cos_[k3]), *gw4 = dev_zero(nw), *gb4 = dev_zero(cos_[k3]), *gw8 = dev_zero(nw), *gb8 = dev_zero(cos_[k3]);
            mode_ref(); nn_conv3d_bwd_weight(xd, xs, gyd4, ys, 3, 1, gwr, gbr);
            mode_mx();
            split_t sr = {0}; sr.sr = 0x3c6ef372u;
            lp_bwd_w_f4(xm, 4, xs, gy4, 4, ys, gw4, gb4, (gnp_t){0}, sr, 0);
            lp_bwd_w_f4(xm, 4, xs, gy8, 3, ys, gw8, gb8, (gnp_t){0}, sr, 0);
            const double e4 = relerr(gw4, gwr, nw), e8 = relerr(gw8, gwr, nw), eb = relerr(gb4, gbr, cos_[k3]), d48 = relerr(gw4, gw8, nw);
            { float *h = malloc(nw * 4); double a = 0, c = 0; nn_d2h(h, gwr, nw * 4); for (size_t i = 0; i < nw; i++) a += fabs(h[i]); nn_d2h(h, gw4, nw * 4); for (size_t i = 0; i < nw; i++) c += fabs(h[i]); free(h);
              printf("  |gw| reference %.6g, fp4 gy %.6g\n", a, c); if (!(a > 0) || !(c > 0)) bad++; }
            const int ok = e4 < 0.1 && d48 < 1e-6 && eb < 1e-3 && e4 > 0;
            printf("  wgrad %d -> %d: gy MX-fp4 %.3g, gy MX-fp8 (same values) %.3g, fp4 vs fp8 %.3g, bias %.3g%s\n", cis[k3], cos_[k3], e4, e8, d48, eb, ok ? "" : "  FAIL");
            if (!ok) bad++;
        }
    }
    for (int which = 0; which < 2; which++) {   /* fp8 weight gradients reading gy: the stem 4 -> 32 and the stride-2 down conv 32 -> 32 */
        const int ci = which ? 32 : 4, co = 32, Pf = 32;
        shape5 xs = {1, ci, Pf, Pf, Pf}, ys = {1, co, which ? Pf / 2 : Pf, which ? Pf / 2 : Pf, which ? Pf / 2 : Pf};
        size_t nx = shape_numel(xs), ny = shape_numel(ys), nw = (size_t)co * ci * 27;
        float *x = dev_rand_wide(nx, 2.f), *gy = dev_rand_wide(ny, 1e-3f);
        size_t bx = which ? nn_mx4_bytes(xs) : nn_mx8_bytes(xs); void *xm = nn_malloc(bx); nn_set_storage(xm, bx, which ? 4 : 8);
        if (which) lp_f32_to_mx4(x, xs.n, xs.c, shape_spatial(xs), xm); else lp_f32_to_mx8(x, xs.n, xs.c, shape_spatial(xs), xm);
        void *gy4 = mx4_from(gy, ys); float *gyd4 = deq4(gy4, ys);
        size_t b8 = nn_mx8_bytes(ys); void *gy8 = nn_malloc(b8); nn_set_storage(gy8, b8, 8); lp_f32_to_mx8(gyd4, ys.n, ys.c, shape_spatial(ys), gy8);
        float *gw4 = dev_zero(nw), *gb4 = dev_zero(co), *gw8 = dev_zero(nw), *gb8 = dev_zero(co);
        split_t sr = {0}; sr.sr = 0x6d2b79f5u;
        if (which) { lp_bwd_w_s2_f8(xm, 4, xs, gy4, 4, ys, gw4, gb4, (gnp_t){0}); lp_bwd_w_s2_f8(xm, 4, xs, gy8, 3, ys, gw8, gb8, (gnp_t){0}); }
        else { lp_bwd_w_f8(xm, 3, xs, gy4, 4, ys, gw4, gb4, (gnp_t){0}, sr); lp_bwd_w_f8(xm, 3, xs, gy8, 3, ys, gw8, gb8, (gnp_t){0}, sr); }
        const double d = relerr(gw4, gw8, nw), db = relerr(gb4, gb8, co);
        float hg[4]; nn_d2h(hg, gw4, 16);
        const int ok = d < 1e-6 && db < 1e-5 && (hg[0] != 0.f || hg[1] != 0.f || hg[2] != 0.f || hg[3] != 0.f);
        printf("  %-46s gw fp4 vs fp8 gy %.3g, bias %.3g%s\n", which ? "wgrad s2 32 -> 32, gy MX-fp4 (FAST)" : "wgrad stem 4 -> 32, gy MX-fp4 (FAST)", d, db, ok ? "" : "  FAIL");
        if (!ok) bad++;
    }
    for (int C = 32; C <= 96; C += 32) for (int accum = 0; accum < 2; accum++) {   /* stride-2 backward-data, MX-fp4 gy / gx (tensor cores; 96: the dilated fp8 path via MX-fp8 copies) */
        shape5 xs = {1, C, 16, 16, 32}, ys = {1, C, 8, 8, 16};
        size_t nx = shape_numel(xs), ny = shape_numel(ys);
        float *w = dev_rand((size_t)C * C * 27, 0.1f), *gy = dev_rand_wide(ny, 1e-3f), *g0 = dev_rand(nx, 1e-3f);
        void *gym = mx4_from(gy, ys); float *gyd = deq4(gym, ys);
        void *g0m = mx4_from(g0, xs); float *g0d = deq4(g0m, xs);
        mode_ref();   /* the scratch sized for the fp32 reference (it needs the whole output there) */
        float *gxr = dev_zero(nx), *scr = nn_malloc(nn_conv3d_scratch(xs, C, 3) + 4096); nn_conv3d_bwd_data(gyd, ys, w, xs, 3, 2, gxr, scr);   /* fp32: no accumulate variant; add the old gradient */
        if (accum) nn_axpy(gxr, 1.f, g0d, nx);
        void *gxm = mx4_new(xs); float *acc = dev_zero(nx), *one = dev_zero(nx), *tmp = dev_zero(nx);
        mode_mx();
        for (int k = 0; k < K; k++) {
            nn_set_sr_step(4000 + k);
            nn_d2d(gxm, g0m, nn_mx4_bytes(xs));
            if (accum) { if (nn_conv3d_bwd_data_acc(gym, ys, w, xs, 3, 2, (float *)gxm, scr)) { printf("  accumulate unsupported  FAIL\n"); bad++; } }
            else nn_conv3d_bwd_data(gym, ys, w, xs, 3, 2, (float *)gxm, scr);
            deq4_into(gxm, xs, tmp);
            if (!k) nn_d2d(one, tmp, nx * 4);
            nn_axpy(acc, 1.f / K, tmp, nx);
        }
        char nm[96]; snprintf(nm, sizeof nm, "s2 bwd_data %d -> %d MX-fp4%s", C, C, accum ? " (accumulate)" : "");
        report(nm, relerr(one, gxr, nx), relerr(acc, gxr, nx), K, 0.35);
    }
    for (int C = 32; C <= 64; C *= 2) {   /* upsample backward (tiled): fine MX-fp4 gy -> coarse MX-fp4 gx */
        shape5 cs = {1, C, 6, 8, 10}, fs = {1, C, 12, 16, 20};
        size_t nc = shape_numel(cs), nf = shape_numel(fs);
        float *gy = dev_rand_wide(nf, 1e-3f); void *gym = mx4_from(gy, fs); float *gyd = deq4(gym, fs);
        float *gxr = dev_zero(nc);
        mode_ref(); nn_up2_bwd(gyd, cs, gxr);
        void *gxm = mx4_new(cs); float *acc = dev_zero(nc), *one = dev_zero(nc), *tmp = dev_zero(nc);
        mode_mx();
        for (int k = 0; k < K; k++) {
            nn_set_sr_step(5000 + k);
            nn_up2_bwd(gym, cs, (float *)gxm);
            deq4_into(gxm, cs, tmp);
            if (!k) nn_d2d(one, tmp, nc * 4);
            nn_axpy(acc, 1.f / K, tmp, nc);
        }
        char nm[96]; snprintf(nm, sizeof nm, "up2 bwd %d ch MX-fp4", C);
        report(nm, relerr(one, gxr, nc), relerr(acc, gxr, nc), K, 0.35);
    }
    {   /* end to end: a small MX net's parameter gradients with MX-fp4 activation gradients vs MX-fp8 (fp4 storage noise is
           unbiased, so the mean of K differently-rounded backwards approaches the fp8 gradients) */
        nn_set_tf32(1); nn_set_act_bf16(1); nn_set_grad_bf16(1); nn_set_f16(1); nn_set_grad_scale(1024); nn_set_sr(1); nn_set_gn_stored(1);
        nn_set_prec(3); if (nn_set_prec_policy("all=fp4:fp4:fp4,enc0.c1=fp16")) bad++;
        setenv("UFSM_F4_WGRAD", "1", 1);
        unet_set_act_mx4(1); unet_set_grad_mx8(1); unet_set_input_prec(8); unet_set_recompute(1); unet_set_chunk_up(2); unet_set_lean(2);
        unet_cfg cfg = {3, {32, 64, 96}, 4, 1, 8, 1};
        shape5 xs = {1, 4, 32, 32, 32}; size_t nx = shape_numel(xs), no = nx / 4;
        float *x = dev_rand(nx, 1.f), *gyo = dev_rand(no, 1.f);
        unet *u = unet_create(&cfg); unet_init(u, 23); size_t np = unet_nparams(u);
        float *gr = malloc(np * 4), *g8 = malloc(np * 4), *g4 = malloc(np * 4), *dm = calloc(np, 4), *rm = calloc(np, 4), *d8 = calloc(np, 4);
        {   /* the sliced decoder conv1 weight gradient == the whole up transient's (no SR: same operands, other summation order) */
            nn_set_sr(0); unet_set_grad_mx4(0);
            unet_set_up_wg_chunk(0); unet_forward(u, x, xs, 1); unet_zero_grad(u); unet_backward(u, gyo); unet_grad_d2h(u, gr);
            unet_set_up_wg_chunk(1); unet_forward(u, x, xs, 1); unet_zero_grad(u); unet_backward(u, gyo); unet_grad_d2h(u, g8);
            unet_set_up_wg_chunk(-1); nn_set_sr(1);
            double d2 = 0, r2 = 0;
            for (size_t i = 0; i < np; i++) { double d = (double)g8[i] - gr[i]; d2 += d * d; r2 += (double)gr[i] * gr[i]; }
            const double rel = sqrt(d2 / r2);
            printf("  %-46s rel diff %.3g%s\n", "unet param. grads, sliced up wgrad vs whole", rel, rel < 1e-4 ? "" : "  FAIL");
            if (!(rel < 1e-4)) bad++;
        }
        /* per k one forward seed for all three runs: 16-bit gradients (reference), MX-fp8, MX-fp4. MX-fp8 stores round to nearest,
           so its error is deterministic; MX-fp4 is SR throughout, so its mean over k must approach the reference */
        double one = 0, one8 = 0;
        for (int k = 0; k < K; k++) {
            unet_set_grad_mx8(0); nn_set_sr_step(6000 + k); unet_forward(u, x, xs, 1); unet_zero_grad(u); unet_backward(u, gyo); unet_grad_d2h(u, gr);
            unet_set_grad_mx8(1); unet_set_grad_mx4(0); nn_set_sr_step(6000 + k); unet_forward(u, x, xs, 1); unet_zero_grad(u); unet_backward(u, gyo); unet_grad_d2h(u, g8);
            unet_set_grad_mx4(1); nn_set_sr_step(6000 + k); unet_forward(u, x, xs, 1); unet_zero_grad(u); unet_backward(u, gyo); unet_grad_d2h(u, g4);
            double a2 = 0, b2 = 0, r2 = 0;
            for (size_t i = 0; i < np; i++) {
                dm[i] += (g4[i] - gr[i]) / K; d8[i] += (g8[i] - gr[i]) / K; rm[i] += gr[i] / K;
                double d = (double)g4[i] - gr[i], e = (double)g8[i] - gr[i]; a2 += d * d; b2 += e * e; r2 += (double)gr[i] * gr[i];
            }
            if (!k) { one = sqrt(a2 / r2); one8 = sqrt(b2 / r2); }
        }
        double a2 = 0, b2 = 0, r2 = 0;
        for (size_t i = 0; i < np; i++) { a2 += (double)dm[i] * dm[i]; b2 += (double)d8[i] * d8[i]; r2 += (double)rm[i] * rm[i]; }
        const double avg = sqrt(a2 / r2), avg8 = sqrt(b2 / r2);
        const int ok = isfinite(one) && avg < one * 2.0 / sqrt((double)K) + 0.01;
        printf("  %-46s one %.3g, mean of %d %.3g\n", "unet param. grads, MX-fp8 act. grads vs 16-bit", one8, K, avg8);
        printf("  %-46s one %.3g, mean of %d %.3g%s\n", "unet param. grads, MX-fp4 act. grads vs 16-bit", one, K, avg, ok ? "" : "  FAIL");
        if (!ok) bad++;
        if (!ok || getenv("GM4_DETAIL")) {   /* per tensor (unet_create's layout): where the residual comes from */
            const char *nm[64]; size_t sz[64]; int nt = 0; char names[64][24];
            const int *w = cfg.widths, L = cfg.nlev;
#define T(fmt, a, n) do { snprintf(names[nt], 24, fmt, a); nm[nt] = names[nt]; sz[nt++] = (n); } while (0)
            for (int i = 0; i < L; i++) { int ci = i ? w[i - 1] : cfg.cin, co = w[i];
                T("enc%d.c1.w", i, (size_t)co * ci * 27); T("enc%d.c1.b", i, co); T("enc%d.n1", i, 2 * co);
                T("enc%d.c2.w", i, (size_t)co * co * 27); T("enc%d.c2.b", i, co); T("enc%d.n2", i, 2 * co); }
            for (int i = 0; i < L - 1; i++) { T("down%d.w", i, (size_t)w[i] * w[i] * 27); T("down%d.b", i, w[i]); }
            for (int i = L - 2; i >= 0; i--) { int ci = w[i] + w[i + 1], co = w[i];
                T("dec%d.c1.w", i, (size_t)co * ci * 27); T("dec%d.c1.b", i, co); T("dec%d.n1", i, 2 * co);
                T("dec%d.c2.w", i, (size_t)co * co * 27); T("dec%d.c2.b", i, co); T("dec%d.n2", i, 2 * co); }
            T("head.w%s", "", (size_t)cfg.cout * w[0]); T("head.b%s", "", cfg.cout);
            for (int i = 0; i < L - 1; i++) T("dn%d", i, 2 * w[i]);
#undef T
            size_t o = 0;
            for (int t = 0; t < nt; t++) {
                double a2 = 0, b2 = 0, c2 = 0;
                for (size_t i = o; i < o + sz[t]; i++) { double d = (double)g4[i] - gr[i]; a2 += d * d; b2 += (double)dm[i] * dm[i]; c2 += (double)rm[i] * rm[i]; }
                double e2 = 0; for (size_t i = o; i < o + sz[t]; i++) e2 += (double)d8[i] * d8[i];
                printf("    %-12s %7zu  fp4 last %.3g mean %.3g   fp8 mean %.3g\n", nm[t], sz[t], sqrt(a2 / c2), sqrt(b2 / c2), sqrt(e2 / c2));
                o += sz[t];
            }
            if (o != np) printf("    layout mismatch: %zu of %zu\n", o, np);
        }
        unet_set_grad_mx4(0); unet_free(u);
        {   /* level-0 A in dec[0].a2's buffer with the skip on the host and the logit gradient in its buffer (the trainer's path:
               unet_logit_grad_scratch), with the shared encoder a1 / input offloads: the same gradients as separate buffers;
               a repeated backward (no forward) the same as the first */
            unet_set_grad_mx4(1); unet_set_share_enc_a1(1);
            unet *v = unet_create(&cfg); unet_init(v, 23);
            float *ga = malloc(np * 4), *gb2 = malloc(np * 4), *gc = malloc(np * 4);
            size_t mem[2];
            for (int on = 1; on >= 0; on--) {
                unet_set_a0_share(on);
                mem[on] = unet_train_bytes(v, xs);
                nn_set_sr_step(9000); unet_forward(v, x, xs, 1); unet_zero_grad(v);
                void *gl = unet_logit_grad_scratch(v, no * 2);
                if (on && !gl) { printf("  a0 share: no logit gradient scratch  FAIL\n"); bad++; }
                if (!gl) { static void *own; if (!own) own = nn_malloc(no * 2); gl = own; }
                nn_f32_to_h16(gyo, no, gl, nn_get_grad_scale());
                nn_set_sr_step(9100); unet_backward_x(v, gl, 1); unet_grad_d2h(v, on ? ga : gb2);
                if (on) { unet_zero_grad(v); nn_set_sr_step(9100); unet_backward(v, gyo); unet_grad_d2h(v, gc); }   /* repeated backward (fp32 gradient: its own buffer) */
            }
            double d2 = 0, e2 = 0, r2 = 0;
            for (size_t i = 0; i < np; i++) { double d = (double)ga[i] - gb2[i]; d2 += d * d; r2 += (double)gb2[i] * gb2[i]; }
            for (int what = 0; what < 2; what++) {   /* the coarse gradient buffers / the down-conv outputs in dec[0]'s buffers (on by default
                                                       above) against their own buffers */
                float *gd = malloc(np * 4); unet_set_a0_share(1); if (what) unet_set_down_offload(0); else unet_set_coarse_grad_share(0);
                const size_t m2 = unet_train_bytes(v, xs);
                nn_set_sr_step(9000); unet_forward(v, x, xs, 1); unet_zero_grad(v);
                void *gl = unet_logit_grad_scratch(v, no * 2);
                nn_f32_to_h16(gyo, no, gl, nn_get_grad_scale());
                nn_set_sr_step(9100); unet_backward_x(v, gl, 1); unet_grad_d2h(v, gd);
                unet_set_coarse_grad_share(-1); unet_set_down_offload(-1);
                double q2 = 0, s2 = 0; for (size_t i = 0; i < np; i++) { double d = (double)ga[i] - gd[i]; q2 += d * d; s2 += (double)gd[i] * gd[i]; }
                const int okc = sqrt(q2 / s2) < 1e-6 && mem[1] < m2;
                printf("  %-46s rel diff %.3g, train bytes %zu -> %zu%s\n", what ? "down-conv outputs offloaded vs own" : "coarse gout in dec[0]'s buffers vs own", sqrt(q2 / s2), m2, mem[1], okc ? "" : "  FAIL");
                if (!okc) bad++;
                free(gd);
            }
            float *gr2 = malloc(np * 4);   /* the repeated backward against a fresh one with the fp32 gradient */
            unet_set_a0_share(0); nn_set_sr_step(9000); unet_forward(v, x, xs, 1); unet_zero_grad(v); nn_set_sr_step(9100); unet_backward(v, gyo); unet_grad_d2h(v, gr2);
            for (size_t i = 0; i < np; i++) { double d = (double)gc[i] - gr2[i]; e2 += d * d; }
            const double rel = sqrt(d2 / r2), rel2 = sqrt(e2 / r2);
            const int ok = rel < 1e-6 && rel2 < 1e-6 && mem[1] < mem[0];
            printf("  %-46s rel diff %.3g, repeated %.3g, train bytes %zu -> %zu%s\n", "level-0 A share (skip offload) vs separate", rel, rel2, mem[0], mem[1], ok ? "" : "  FAIL");
            if (!ok) bad++;
            unet_set_a0_share(-1); unet_set_share_enc_a1(0); unet_set_grad_mx4(0); unet_free(v); free(ga); free(gb2); free(gc); free(gr2);
        } free(gr); free(g8); free(g4); free(dm); free(rm); free(d8);
    }
    const char *e = nn_check(); if (e) { printf("cuda: %s\n", e); bad++; }
    printf("gradient MX-fp4: %s\n", bad ? "FAIL" : "ok");
    return bad != 0;
}
