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
/* the weights as the fp4 stride-2 backward-data kernel sees them (lp_mx_s2bwd_tc.cu s2b_f4: w transposed, prep_w4_k): per (ci, tap)
   an MX block over the 32 output channels, scale 2^ceil(log2(amax / 6)), e2m1 round-to-nearest-even. Its deterministic error then
   drops out of the reference, leaving the stochastic rounding of the stored gradient */
static float *w_s2b_e2m1(const float *wd, int C) {
    static const float grid[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const size_t nw = (size_t)C * C * 27; float *h = malloc(nw * 4); nn_d2h(h, wd, nw * 4);
    for (int ci = 0; ci < C; ci++) for (int t = 0; t < 27; t++) for (int c0 = 0; c0 < C; c0 += 32) {
        float am = 0.f;
        for (int k = 0; k < 32; k++) am = fmaxf(am, fabsf(h[((size_t)(c0 + k) * C + ci) * 27 + t]));
        float q = am * (1.f / 6.f); unsigned u; memcpy(&u, &q, 4);
        int e = (int)((u >> 23) & 255) - 127 + ((u & 0x7fffffu) != 0); e = e < -126 ? -126 : e > 126 ? 126 : e;
        for (int k = 0; k < 32; k++) {
            float *v = &h[((size_t)(c0 + k) * C + ci) * 27 + t];
            const float a = fminf(fabsf(*v) * ldexpf(1.f, -e), 6.f); int bi = 0;
            for (int j = 1; j < 8; j++) { const float dj = fabsf(a - grid[j]), db = fabsf(a - grid[bi]); if (dj < db || (dj == db && !(j & 1))) bi = j; }
            *v = copysignf(grid[bi] * ldexpf(1.f, e), *v);
        }
    }
    float *d = nn_malloc(nw * 4); nn_h2d(d, h, nw * 4); free(h); return d;
}
/* head-slab test callback: the backward's head step per slab of Z planes, the logit gradient copied from the whole-layout gl */
typedef struct { unet *v; void *gl; void *sb; size_t S, hw; int D, Z, bad; } hs_ctx;
static void hs_cb(void *p) {
    hs_ctx *h = p;
    for (int z0 = 0; z0 < h->D; z0 += h->Z) {
        const int z1 = z0 + h->Z < h->D ? z0 + h->Z : h->D; const size_t ns = (size_t)(z1 - z0) * h->hw;
        for (int c = 0; c < 11; c++) nn_d2d((char *)h->sb + c * ns * 2, (const char *)h->gl + (c * h->S + z0 * h->hw) * 2, ns * 2);
        if (unet_head_bwd_z(h->v, z0, z1, h->sb, z0 == 0)) h->bad++;
    }
}
static void mode_ref(void) { nn_set_tf32(0); }
static void mode_mx(void) { nn_set_prec(3); nn_set_tf32(1); }
int main(void) {
    nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0);
    nn_set_act_bf16(0); nn_set_grad_bf16(0);
    {   /* e4m3 stochastic rounding of the fp8 weight gradient's gy (sr_e4m3x): unbiased, grid values exact (sr_e4m3, the dither then
           round-to-nearest, gives 1 -> 0.98438: -1/64 on powers of two). 4e6 roundings: sigma <= 0.5 ulp / 2000 */
        const float pv[] = {1.f, 1.5f, 0.25f, 3.f, 0.3f, 1.4f, -1.4f, 1.5f / 512.f, 0.7f / 512.f, 448.f, 430.f};
        for (int i = 0; i < 11; i++) {
            const double m = lp_sr_e4m3x_mean(pv[i], 4000000), ulp = ldexp(1.0, (int)fmax(floor(log2(fabs((double)pv[i]))), -6.0) - 3);
            const int ok = fabs(m - pv[i]) < 1.5e-3 * ulp + 1e-12;
            printf("  sr_e4m3x mean of 4e6 roundings of %-10.6g %.8g (%+.2g ulp)%s\n", pv[i], m, (m - pv[i]) / ulp, ok ? "" : "  FAIL");
            if (!ok) bad++;
            if (getenv("UFSM_SR_E4M3_PROBE")) { const double m0 = lp_sr_e4m3_mean(pv[i], 4000000); printf("    (sr_e4m3: %.8g, %+.2g ulp)\n", m0, (m0 - pv[i]) / ulp); }
        }
    }
    const int N = 2, G = 8, K = 16;
    for (int C = 8; C <= 64; C *= 2) {   /* GroupNorm + SiLU backward: gy and gx MX-fp4 */
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
    for (int C = 8; C <= 32; C *= 2) {   /* the apply that also writes the weight gradient's gy records (N 1): its gx bit-identical to the plain apply's
                                            (same folded coefficients, same SR keys); 8 / 16 channels only with the narrow records on */
        shape5 s = {1, C, 8, 10, 12};
        size_t n = shape_numel(s);
        float *x = dev_rand(n, 2.f), *gam = dev_rand(C, 1.f), *bet = dev_rand(C, 0.5f), *mean = dev_rand(G, 0.2f), *rstd = dev_rand(G, 0.3f);
        { float h[64]; nn_d2h(h, rstd, G * 4); for (int i = 0; i < G; i++) h[i] = 0.8f + fabsf(h[i]); nn_h2d(rstd, h, G * 4); }
        void *xm = mx4_from(x, s); float *gy = dev_rand(n, 1e-3f); void *gym = mx4_from(gy, s);
        float *gg = dev_zero(C), *gb = dev_zero(C), *scr = nn_malloc(nn_gn_scratch(s) + 4096);
        void *gxa = mx4_new(s), *gxb = mx4_new(s);
        const size_t bytes = nn_mx4_bytes(s);
        mode_mx();
        lp_set_gyrec_narrow(0); lp_set_gyrec_off(1); nn_set_sr_step(1500); nn_gn_silu_bwd(xm, s, G, gam, bet, mean, rstd, gym, (float *)gxa, gg, gb, scr);
        lp_set_gyrec_narrow(1); lp_set_gyrec_off(0); nn_set_sr_step(1500); nn_gn_silu_bwd(xm, s, G, gam, bet, mean, rstd, gym, (float *)gxb, gg, gb, scr);
        lp_set_gyrec_narrow(-1); lp_set_gyrec_off(-1);
        uint8_t *ha = malloc(bytes), *hb = malloc(bytes); nn_d2h(ha, gxa, bytes); nn_d2h(hb, gxb, bytes);
        const int same = !memcmp(ha, hb, bytes);
        printf("  gn+silu bwd with gy records, %d ch: gx bytes %s the plain apply's%s\n", C, same ? "equal" : "DIFFER from", same ? "" : "  FAIL");
        if (!same) bad++;
        free(ha); free(hb);
    }
    /* dec[0]'s GN+SiLU backward with the head's input gradient recomputed from the fp16 logit gradient (unet's three level-0 buffers):
       the same bits as the head's backward-data into an MX-fp4 buffer followed by the GN backward from it (gx, the GroupNorm parameter
       gradients, the seed sequence); gx in place over x */
    const uint64_t rs_hg = rs;   /* (the tests below draw the random inputs they drew before this one was added) */
    for (int C = 8; C <= 16; C *= 2) for (int co = 1; co <= 3; co++) for (int sh = 0; sh < 2; sh++) {
        nn_set_tf32(1); nn_set_prec(3); nn_set_act_bf16(1); nn_set_grad_bf16(1); nn_set_f16(1); nn_set_grad_scale(1024); nn_set_sr(1);
        const shape5 s = sh ? (shape5){2, C, 6, 10, 14} : (shape5){1, C, 16, 32, 32}, os = {s.n, co, s.d, s.h, s.w};
        const int Gh = C / 4;
        const size_t n = shape_numel(s), no = shape_numel(os), NG = (size_t)s.n * Gh, xb = nn_mx4_bytes(s);
        float *x = dev_rand_wide(n, 1.f), *gam = dev_rand(C, 1.f), *bet = dev_rand(C, 0.5f), *mean = dev_zero(NG), *rstd = dev_zero(NG), *wh = dev_rand((size_t)co * C, 0.5f);
        void *xm = mx4_from(x, s);
        if (nn_gn_stats(xm, s, Gh, 1e-5f, mean, rstd)) { printf("  head-recomputing gn bwd: no MX-fp4 stats  FAIL\n"); bad++; }
        float *glf = dev_rand_wide(no, 1e-2f); void *gl = nn_malloc(no * 2); nn_f32_to_h16(glf, no, gl, 1024.f);
        void *gout = mx4_new(s), *xf = mx4_new(s);
        float *gg1 = dev_zero(C), *gb1 = dev_zero(C), *gg2 = dev_zero(C), *gb2 = dev_zero(C), *scr = nn_malloc(nn_gn_scratch(s) + 4096), *cs = nn_malloc(nn_conv3d_scratch(s, co, 1) + 4096);
        nn_set_sr_step(4242); nn_set_layer(1);
        nn_conv3d_bwd_data(gl, os, wh, s, 1, 1, gout, cs);
        nn_gn_silu_bwd(xm, s, Gh, gam, bet, mean, rstd, gout, gout, gg1, gb1, scr);
        const unsigned c1 = nn_sr_ctr();
        nn_set_sr_step(4242);
        nn_d2d(xf, xm, xb);
        const unsigned hs = nn_sr_seed();
        const int rc = nn_gn_silu_bwd_head(xf, s, Gh, gam, bet, mean, rstd, gl, co, wh, hs, gg2, gb2, scr);
        const unsigned c2 = nn_sr_ctr();
        nn_set_layer(-1);
        uint8_t *ha = malloc(xb), *hb = malloc(xb); float hg[4][16];
        nn_sync(); nn_d2h(ha, gout, xb); nn_d2h(hb, xf, xb);
        nn_d2h(hg[0], gg1, C * 4); nn_d2h(hg[1], gg2, C * 4); nn_d2h(hg[2], gb1, C * 4); nn_d2h(hg[3], gb2, C * 4);
        size_t nd = 0; for (size_t i = 0; i < xb; i++) nd += ha[i] != hb[i];
        const int pe = !memcmp(hg[0], hg[1], C * 4) && !memcmp(hg[2], hg[3], C * 4);
        const char *e = nn_check();
        const int ok = !rc && !nd && pe && c1 == c2 && !e;
        printf("  head-recomputing gn bwd %2d ch, %d logits, %s  %zu of %zu bytes differ, param grads %s, seeds %u / %u%s%s\n", C, co, sh ? "odd" : "even",
               nd, xb, pe ? "equal" : "DIFFER", c1, c2, e ? e : "", ok ? "" : "  FAIL");
        if (!ok) bad++;
        free(ha); free(hb);
        nn_free(x); nn_free(gam); nn_free(bet); nn_free(mean); nn_free(rstd); nn_free(wh); nn_free(xm); nn_free(glf); nn_free(gl); nn_free(gout); nn_free(xf);
        nn_free(gg1); nn_free(gb1); nn_free(gg2); nn_free(gb2); nn_free(scr); nn_free(cs);
        nn_set_act_bf16(0); nn_set_grad_bf16(0); nn_set_f16(0); nn_set_sr(0);
    }
    rs = rs_hg;
    {   /* stride-1 backward-data with narrow (16-channel) MX-fp4 gy / gx and weights +-6 * 2^-5 (exact in e2m1 / e4m3 / bf16 under any
           block layout): the fp32 reference is then the exact expectation of the SR'd output, so the mean of K converges to it unless
           the gy staging re-rounds (the p16 kernel re-rounded paired voxels to their shared scale to nearest: a bias floor) */
        const int cis[7] = {16, 16, 32, 32, 8, 8, 16}, cos_[7] = {16, 32, 16, 32, 8, 16, 8};
        for (int k4 = 0; k4 < 7; k4++) {
            shape5 xs = {1, cis[k4], 8, 16, 32}, ys = xs; ys.c = cos_[k4];
            size_t nx = shape_numel(xs), ny = shape_numel(ys), nw = (size_t)cos_[k4] * cis[k4] * 27;
            float *hw = malloc(nw * 4); for (size_t i = 0; i < nw; i++) hw[i] = (frand() < 0.f ? -6.f : 6.f) / 32.f;
            float *w = nn_malloc(nw * 4); nn_h2d(w, hw, nw * 4); free(hw);
            float *gy = dev_rand_wide(ny, 1e-3f);
            void *gym = mx4_from(gy, ys); float *gyd = deq4(gym, ys);
            mode_ref();
            float *gxr = dev_zero(nx), *scr = nn_malloc(nn_conv3d_scratch(xs, cos_[k4], 3) + 4096); nn_conv3d_bwd_data(gyd, ys, w, xs, 3, 1, gxr, scr);
            void *gxm = mx4_new(xs); float *acc = dev_zero(nx), *one = dev_zero(nx), *tmp = dev_zero(nx);
            mode_mx();
            for (int k = 0; k < K; k++) {
                nn_set_sr_step(2500 + k);
                nn_conv3d_bwd_data(gym, ys, w, xs, 3, 1, (float *)gxm, scr);
                deq4_into(gxm, xs, tmp);
                if (!k) nn_d2d(one, tmp, nx * 4);
                nn_axpy(acc, 1.f / K, tmp, nx);
            }
            char nm[96]; snprintf(nm, sizeof nm, "bwd_data %d -> %d MX-fp4, exact weights", cos_[k4], cis[k4]);
            const double e1 = relerr(one, gxr, nx), ea = relerr(acc, gxr, nx);
            report(nm, e1, ea, K, 0.35);
            if (!(ea < e1 * 1.5 / sqrt((double)K))) { printf("  %-46s mean of %d %.3g > 1.5 one / sqrt(K): staging bias  FAIL\n", nm, K, ea); bad++; }
            nn_free(w); nn_free(gy); nn_free(gym); nn_free(gyd); nn_free(gxr); nn_free(scr); nn_free(gxm); nn_free(acc); nn_free(one); nn_free(tmp);
        }
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
    {   /* stride-1 weight gradient (fp4 kernels, gy pre-pass): MX-fp4 gy vs MX-fp8 gy vs fp32; narrow levels (16 / 8 channels in
           or out) also the warp-specialised kernel vs conv_bwd_w_f4_k (the same records and products: only the gw sums group
           differently) */
        const int cis[6] = {32, 64, 16, 32, 16, 8}, cos_[6] = {32, 32, 16, 16, 8, 8};
        nn_set_sr(1); setenv("UFSM_F4_WGRAD", "1", 1);
        for (int k3 = 0; k3 < 6; k3++) {
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
            const double tol4 = cis[k3] >= 32 && cos_[k3] >= 32 ? 0.1 : 0.16;   /* one SR draw of gy (narrow levels: 0.10 .. 0.13) */
            const int ok = e4 < tol4 && d48 < 1e-6 && eb < 1e-3 && e4 > 0;
            printf("  wgrad %d -> %d: gy MX-fp4 %.3g, gy MX-fp8 (same values) %.3g, fp4 vs fp8 %.3g, bias %.3g%s\n", cis[k3], cos_[k3], e4, e8, d48, eb, ok ? "" : "  FAIL");
            if (!ok) bad++;
            {   /* the same call on conv_bwd_w_f4_k (warp-specialised kernel off): equal up to the fp32 grouping of the sums */
                float *gwp = dev_zero(nw), *gbp = dev_zero(cos_[k3]);
                lp_set_f4w_ws(0); lp_bwd_w_f4(xm, 4, xs, gy4, 4, ys, gwp, gbp, (gnp_t){0}, sr, 0); lp_set_f4w_ws(-1);
                const double dws = relerr(gw4, gwp, nw);
                printf("  wgrad %d -> %d: warp-specialised vs plain kernel rel diff %.3g%s\n", cis[k3], cos_[k3], dws, dws < 1e-6 ? "" : "  FAIL");
                if (!(dws < 1e-6)) bad++;
                nn_free(gwp); nn_free(gbp);
            }
        }
        for (int k4 = 0; k4 < 4; k4++) {   /* split inputs (decoder inputs): 8-channel x + 16-channel x2 (8-channel tiles, the warp-specialised
                                              kernel's NT 1) and 16 + 8 (a 16-channel tile over the 8-channel segment and padding), vs fp32 and
                                              vs the plain kernel */
            const int c1 = k4 < 2 ? 8 : 16, c2 = 24 - c1, co = k4 & 1 ? 16 : 8;
            shape5 x1s = {1, c1, 16, 16, 32}, x2s = x1s, xs = x1s, ys = x1s; x2s.c = c2; xs.c = 24; ys.c = co;
            const size_t n1 = shape_numel(x1s), n2 = shape_numel(x2s), nw = (size_t)co * 24 * 27;
            float *x1 = dev_rand_wide(n1, 2.f), *x2 = dev_rand_wide(n2, 2.f), *gy = dev_rand_wide(shape_numel(ys), 1e-3f);
            void *x1m = mx4_from(x1, x1s), *x2m = mx4_from(x2, x2s); float *x1d = deq4(x1m, x1s), *x2d = deq4(x2m, x2s);
            float *xd = dev_zero(n1 + n2); nn_d2d(xd, x1d, n1 * 4); nn_d2d(xd + n1, x2d, n2 * 4);   /* N 1: channel concatenation */
            void *gy4 = mx4_from(gy, ys); float *gyd4 = deq4(gy4, ys);
            float *gwr = dev_zero(nw), *gbr = dev_zero(co), *gw4 = dev_zero(nw), *gb4 = dev_zero(co), *gwp = dev_zero(nw), *gbp = dev_zero(co);
            mode_ref(); nn_conv3d_bwd_weight(xd, xs, gyd4, ys, 3, 1, gwr, gbr);
            mode_mx();
            split_t sr = {0}; sr.sr = 0x3c6ef372u; sr.x2 = x2m; sr.c_split = c1;
            lp_bwd_w_f4(x1m, 4, xs, gy4, 4, ys, gw4, gb4, (gnp_t){0}, sr, 0);
            lp_set_f4w_ws(0); lp_bwd_w_f4(x1m, 4, xs, gy4, 4, ys, gwp, gbp, (gnp_t){0}, sr, 0); lp_set_f4w_ws(-1);
            const double e4 = relerr(gw4, gwr, nw), dws = relerr(gw4, gwp, nw), eb = relerr(gb4, gbr, co);
            const int ok = e4 < 0.16 && e4 > 0 && dws < 1e-6 && eb < 1e-3;
            printf("  wgrad split %d + %d -> %d: gy MX-fp4 %.3g, warp-specialised vs plain %.3g, bias %.3g%s\n", c1, c2, co, e4, dws, eb, ok ? "" : "  FAIL");
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
    for (int C = 8; C <= 96; C += C <= 16 ? C : 32) for (int accum = 0; accum < 2; accum++) {   /* stride-2 backward-data, MX-fp4 gy / gx (tensor cores; 96: the dilated fp8 path via MX-fp8 copies) */
        shape5 xs = {1, C, 16, 16, 32}, ys = {1, C, 8, 8, 16};
        size_t nx = shape_numel(xs), ny = shape_numel(ys);
        float *w = dev_rand((size_t)C * C * 27, 0.1f), *gy = dev_rand_wide(ny, 1e-3f), *g0 = dev_rand(nx, 1e-3f);
        void *gym = mx4_from(gy, ys); float *gyd = deq4(gym, ys);
        void *g0m = mx4_from(g0, xs); float *g0d = deq4(g0m, xs);
        mode_ref();   /* the scratch sized for the fp32 reference (it needs the whole output there) */
        float *wr = C == 32 || C == 64 ? w_s2b_e2m1(w, C) : w;   /* 32 / 64 channels: the fp4 kernel (its weights in e2m1); 16: bf16 tensor cores */
        float *gxr = dev_zero(nx), *scr = nn_malloc(nn_conv3d_scratch(xs, C, 3) + 4096); nn_conv3d_bwd_data(gyd, ys, wr, xs, 3, 2, gxr, scr);   /* fp32: no accumulate variant; add the old gradient */
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
    for (int C = 8; C <= 64; C *= 2) {   /* upsample backward (tiled, block widths 8 / 16 / 32): fine MX-fp4 gy -> coarse MX-fp4 gx */
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
    for (int wc = 0; wc < 3; wc++) {   /* end to end, narrow levels (8 / 16-channel MX-fp4 gradients): parameter gradients vs 16-bit (the mean of K
                                          must approach them), fp16 logits (the level-0 chain engages), fewer training bytes than MX-fp8 */
        nn_set_tf32(1); nn_set_act_bf16(1); nn_set_grad_bf16(1); nn_set_f16(1); nn_set_grad_scale(1024); nn_set_sr(1); nn_set_gn_stored(1);
        nn_set_prec(3); if (nn_set_prec_policy("all=fp4:fp4:fp4,enc0.c1=fp16")) bad++;
        setenv("UFSM_F4_WGRAD", "1", 1);
        unet_set_act_mx4(1); unet_set_grad_mx8(1); unet_set_input_prec(8); unet_set_recompute(1); unet_set_chunk_up(2); unet_set_lean(2);
        unet_cfg cfg = wc == 0 ? (unet_cfg){3, {16, 32, 64}, 4, 2, 8, 1} : wc == 1 ? (unet_cfg){3, {16, 16, 32}, 4, 2, 8, 1} : (unet_cfg){3, {8, 16, 32}, 4, 2, 8, 1};
        shape5 xs = {1, 4, 32, 32, 32}; size_t nx = shape_numel(xs), no = nx / 4 * 2;
        float *x = dev_rand(nx, 1.f), *gyo = dev_rand(no, 1.f);
        unet *u = unet_create(&cfg); unet_init(u, 29); size_t np = unet_nparams(u);
        unet_set_grad_mx4(0); const size_t b8 = unet_train_bytes(u, xs);
        unet_set_grad_mx4(1); const size_t b4 = unet_train_bytes(u, xs);
        float *gr = malloc(np * 4), *g4 = malloc(np * 4), *dm = calloc(np, 4), *rm = calloc(np, 4);
        double one = 0; int h16 = 0;
        for (int k = 0; k < K; k++) {
            unet_set_grad_mx8(0); nn_set_sr_step(7000 + k); unet_forward(u, x, xs, 1); unet_zero_grad(u); unet_backward(u, gyo); unet_grad_d2h(u, gr);
            unet_set_grad_mx8(1); unet_set_grad_mx4(1); nn_set_sr_step(7000 + k); unet_forward(u, x, xs, 1); h16 |= unet_logits_h16(u); unet_zero_grad(u); unet_backward(u, gyo); unet_grad_d2h(u, g4);
            double a2 = 0, r2 = 0;
            for (size_t i = 0; i < np; i++) { dm[i] += (g4[i] - gr[i]) / K; rm[i] += gr[i] / K; double d = (double)g4[i] - gr[i]; a2 += d * d; r2 += (double)gr[i] * gr[i]; }
            if (!k) one = sqrt(a2 / r2);
        }
        { const char *e = nn_check(); if (e) { printf("  narrow end to end: CUDA error %s  FAIL\n", e); bad++; } }
        double a2 = 0, r2 = 0;
        for (size_t i = 0; i < np; i++) { a2 += (double)dm[i] * dm[i]; r2 += (double)rm[i] * rm[i]; }
        const double avg = sqrt(a2 / r2);
        char nm[96]; snprintf(nm, sizeof nm, "unet {%d,%d,%d} MX-fp4 act. grads vs 16-bit", cfg.widths[0], cfg.widths[1], cfg.widths[2]);
        const int ok = isfinite(one) && avg < one * 2.0 / sqrt((double)K) + 0.01;
        printf("  %-46s one %.3g, mean of %d %.3g%s\n", nm, one, K, avg, ok ? "" : "  FAIL"); if (!ok) bad++;
        snprintf(nm, sizeof nm, "unet {%d,%d,%d} train bytes fp8 -> fp4 grads", cfg.widths[0], cfg.widths[1], cfg.widths[2]);
        printf("  %-46s %zu -> %zu, fp16 logits %d%s\n", nm, b8, b4, h16, b4 < b8 && h16 ? "" : "  FAIL"); if (!(b4 < b8 && h16)) bad++;
        unet_set_grad_mx4(0);
        unet_free(u); nn_free(x); nn_free(gyo); free(gr); free(g4); free(dm); free(rm);
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
            for (int what = 0; what < 3; what++) {   /* the coarse gradient buffers / the down-conv outputs in dec[0]'s buffers (on by default
                                                       above) against their own buffers */
                float *gd = malloc(np * 4); unet_set_a0_share(1); if (what == 2) unet_set_skip_offload(0); else if (what) unet_set_down_offload(0); else unet_set_coarse_grad_share(0);
                const size_t m2 = unet_train_bytes(v, xs);
                nn_set_sr_step(9000); unet_forward(v, x, xs, 1); unet_zero_grad(v);
                void *gl = unet_logit_grad_scratch(v, no * 2);
                nn_f32_to_h16(gyo, no, gl, nn_get_grad_scale());
                nn_set_sr_step(9100); unet_backward_x(v, gl, 1); unet_grad_d2h(v, gd);
                unet_set_coarse_grad_share(-1); unet_set_down_offload(-1); unet_set_skip_offload(-1);
                double q2 = 0, s2 = 0; for (size_t i = 0; i < np; i++) { double d = (double)ga[i] - gd[i]; q2 += d * d; s2 += (double)gd[i] * gd[i]; }
                const int okc = sqrt(q2 / s2) < 1e-6 && mem[1] < m2;
                printf("  %-46s rel diff %.3g, train bytes %zu -> %zu%s\n", what == 2 ? "coarse skips offloaded vs own" : what ? "down-conv outputs offloaded vs own" : "coarse gout in dec[0]'s buffers vs own", sqrt(q2 / s2), m2, mem[1], okc ? "" : "  FAIL");
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
        }
        {   /* the same chain with an 11-output head (unwrap): the fp16 logits (22 B per voxel) and their gradient do not fit the level-0
               buffers, so they keep buffers of their own while A / the skip / the coarse buffers still share and offload */
            unet_set_grad_mx4(1); unet_set_share_enc_a1(1);
            unet_cfg c11 = cfg; c11.cout = 11;
            unet *v = unet_create(&c11); unet_init(v, 29); unet_set_fp16_logits(v, 1);
            const size_t np11 = unet_nparams(v), no11 = no * 11;
            float *ga = malloc(np11 * 4), *gb2 = malloc(np11 * 4), *gy11 = dev_rand(no11, 1.f);
            void *gl11 = nn_malloc(no11 * 2);
            size_t mem[2];
            for (int on = 1; on >= 0; on--) {
                unet_set_a0_share(on);
                mem[on] = unet_train_bytes(v, xs);
                nn_set_sr_step(9000); unet_forward(v, x, xs, 1); unet_zero_grad(v);
                if (!unet_logits_h16(v)) { printf("  11-output head: logits not fp16  FAIL\n"); bad++; }
                if (unet_logit_grad_scratch(v, no11 * 2)) { printf("  11-output head: logit gradient placed in a level-0 buffer  FAIL\n"); bad++; }
                nn_f32_to_h16(gy11, no11, gl11, nn_get_grad_scale());
                nn_set_sr_step(9100); unet_backward_x(v, gl11, 1); unet_grad_d2h(v, on ? ga : gb2);
            }
            double d2 = 0, r2 = 0;
            for (size_t i = 0; i < np11; i++) { double d = (double)ga[i] - gb2[i]; d2 += d * d; r2 += (double)gb2[i] * gb2[i]; }
            const double rel = sqrt(d2 / r2);
            const int ok = rel < 1e-6 && mem[1] < mem[0];
            printf("  %-46s rel diff %.3g, train bytes %zu -> %zu%s\n", "11-output head: level-0 chain vs separate", rel, mem[0], mem[1], ok ? "" : "  FAIL");
            if (!ok) bad++;
            unet_set_a0_share(-1); unet_set_share_enc_a1(0); unet_set_grad_mx4(0); unet_free(v); free(ga); free(gb2); nn_free(gy11); nn_free(gl11);
        }
        {   /* head slabs (train.c head_slab_pass): the 11-output head per z-slab (3 planes, a partial last slab) through unet_head_fwd_z
               and, inside the backward, unet_head_bwd_z from the same logit gradient; the slab logits equal the whole head's bit for
               bit, the parameter gradients up to summation order, and the build holds no whole-window logits */
            unet_set_grad_mx4(1); unet_set_share_enc_a1(1); unet_set_a0_share(1);
            unet_cfg c11 = cfg; c11.cout = 11;
            unet *v = unet_create(&c11); unet_init(v, 31); unet_set_fp16_logits(v, 1);
            const size_t np11 = unet_nparams(v), no11 = no * 11, S = no, hw = (size_t)xs.h * xs.w;
            float *ga = malloc(np11 * 4), *gb2 = malloc(np11 * 4), *gy11 = dev_rand(no11, 1.f);
            void *gl11 = nn_malloc(no11 * 2);
            uint16_t *lw = malloc(no11 * 2), *ls = malloc(no11 * 2);
            size_t mem[2];
            nn_f32_to_h16(gy11, no11, gl11, nn_get_grad_scale());
            mem[0] = unet_train_bytes(v, xs);
            nn_set_sr_step(9000); const float *lg = unet_forward(v, x, xs, 1); unet_zero_grad(v);
            nn_d2h(lw, lg, no11 * 2);
            nn_set_sr_step(9100); unet_backward_x(v, gl11, 1); unet_grad_d2h(v, ga);
            hs_ctx hc = {v, gl11, nn_malloc(11 * 3 * hw * 2), S, hw, xs.d, 3, 0};
            unet_set_head_ext(v, 1); unet_set_head_cb(v, hs_cb, &hc);
            mem[1] = unet_train_bytes(v, xs);
            nn_set_sr_step(9000); const float *lg2 = unet_forward(v, x, xs, 1); unet_zero_grad(v);
            for (int z0 = 0; z0 < xs.d; z0 += 3) {   /* the slab logits, gathered into the whole layout */
                const int z1 = z0 + 3 < xs.d ? z0 + 3 : xs.d; const size_t ns = (size_t)(z1 - z0) * hw;
                if (unet_head_fwd_z(v, z0, z1, hc.sb)) { hc.bad++; break; }
                for (int c = 0; c < 11; c++) nn_d2h(ls + c * S + z0 * hw, (const char *)hc.sb + c * ns * 2, ns * 2);
            }
            nn_set_sr_step(9100); unet_backward_x(v, nullptr, 1); unet_grad_d2h(v, gb2);
            size_t lneq = 0; for (size_t i = 0; i < no11; i++) lneq += lw[i] != ls[i];
            double d2 = 0, r2 = 0;
            for (size_t i = 0; i < np11; i++) { double d = (double)ga[i] - gb2[i]; d2 += d * d; r2 += (double)ga[i] * ga[i]; }
            const double rel = sqrt(d2 / r2);
            const int ok = !lg2 && !hc.bad && !lneq && rel < 1e-6 && mem[1] + no11 * 2 <= mem[0];
            printf("  %-46s logits differ %zu, grads rel diff %.3g, train bytes %zu -> %zu%s\n", "11-output head slabs vs whole head", lneq, rel, mem[0], mem[1], ok ? "" : "  FAIL");
            if (!ok) bad++;
            unet_set_a0_share(-1); unet_set_share_enc_a1(0); unet_set_grad_mx4(0); unet_free(v); free(ga); free(gb2); free(lw); free(ls); nn_free(gy11); nn_free(gl11); nn_free(hc.sb);
        } free(gr); free(g8); free(g4); free(dm); free(rm); free(d8);
    }
    {   /* three level-0 buffers (UFSM_L0_3BUF, unet.l3b) against the four-buffer chain, the trainer's way (the logit gradient over the fp16
           logits / in the skip's buffer): two training steps with a validation forward between them, 16- and 8-channel level 0, down_norm on
           and off. The logits bit for bit, the parameter gradients up to the weight gradients' summation order, the same seed count, fewer
           training bytes, the batch tail at X3's end untouched by the forwards */
        nn_set_tf32(1); nn_set_act_bf16(1); nn_set_grad_bf16(1); nn_set_f16(1); nn_set_grad_scale(1024); nn_set_sr(1); nn_set_gn_stored(1);
        nn_set_prec(3); if (nn_set_prec_policy("all=fp4:fp4:fp4,enc0.c1=fp16")) bad++;
        setenv("UFSM_F4_WGRAD", "1", 1);
        unet_set_act_mx4(1); unet_set_grad_mx8(1); unet_set_grad_mx4(1); unet_set_input_prec(4); unet_set_recompute(1); unet_set_chunk_up(2); unet_set_lean(2);
        unet_set_share_enc_a1(1); unet_set_a0_share(1);
        for (int wc = 0; wc < 3; wc++) {
            unet_cfg cfg = wc == 0 ? (unet_cfg){4, {16, 16, 32, 64}, 4, 2, 8, 1} : wc == 1 ? (unet_cfg){4, {16, 16, 32, 64}, 4, 2, 8, 0} : (unet_cfg){4, {8, 8, 16, 32}, 4, 2, 8, 1};
            const shape5 xs = {1, 4, 32, 32, 32};
            const size_t nx = shape_numel(xs), S = nx / 4, no = S * 2, tailb = 3 * S;
            float *xi[2] = {dev_rand(nx, 1.f), dev_rand(nx, 1.f)}, *gy[2] = {dev_rand(no, 1.f), dev_rand(no, 1.f)};
            unet *v = unet_create(&cfg); unet_init(v, 37); const size_t np = unet_nparams(v);
            float *gg[2][2]; uint16_t *lgs[2][3]; size_t mem[2]; unsigned ctr[2][2]; int l3[2] = {0, 0}, tail_ok = 1;
            uint8_t *pat = malloc(tailb), *chk = malloc(tailb);
            for (size_t i = 0; i < tailb; i++) pat[i] = (uint8_t)(i * 2654435761u >> 13);
            void *own = nn_malloc(no * 2);
            for (int on = 0; on < 2; on++) {
                unet_set_l0_3buf(on);
                unet_set_batch_bytes(v, 2 * S, tailb);
                mem[on] = unet_train_bytes(v, xs);
                unet_build(v, xs, 1);
                l3[on] = unet_l0_3buf(v);
                void *tp = nullptr; void *hp = unet_batch_scratch2(v, 2 * S, tailb, &tp);
                if (on && (!hp || !tp)) tail_ok = 0;
                for (int st = 0; st < 2; st++) {
                    if (tp) nn_h2d(tp, pat, tailb);
                    nn_set_sr_step(9000 + 200 * st); const float *lg = unet_forward(v, xi[st], xs, 1); unet_zero_grad(v);
                    lgs[on][st] = malloc(no * 2); nn_sync(); nn_d2h(lgs[on][st], lg, no * 2);
                    if (tp) { nn_d2h(chk, tp, tailb); if (memcmp(chk, pat, tailb)) tail_ok = 0; }
                    void *gl = on ? (void *)lg : unet_logit_grad_scratch(v, no * 2);
                    if (!gl) gl = own;
                    nn_f32_to_h16(gy[st], no, gl, nn_get_grad_scale());
                    nn_set_sr_step(9100 + 200 * st); unet_backward_x(v, gl, 1); ctr[on][st] = nn_sr_ctr();
                    gg[on][st] = malloc(np * 4); unet_grad_d2h(v, gg[on][st]);
                    if (!st) {   /* the trainer's validation forward between steps */
                        if (tp) nn_h2d(tp, pat, tailb);
                        nn_set_sr_step(9050); const float *lv = unet_forward(v, xi[1], xs, 0);
                        lgs[on][2] = malloc(no * 2); nn_sync(); nn_d2h(lgs[on][2], lv, no * 2);
                        if (tp) { nn_d2h(chk, tp, tailb); if (memcmp(chk, pat, tailb)) tail_ok = 0; }
                    }
                }
            }
            { const char *e = nn_check(); if (e) { printf("  three level-0 buffers: CUDA error %s  FAIL\n", e); bad++; } }
            size_t lneq = 0; for (int k = 0; k < 3; k++) for (size_t i = 0; i < no; i++) lneq += lgs[0][k][i] != lgs[1][k][i];
            double rel[2];
            for (int st = 0; st < 2; st++) {
                double d2 = 0, r2 = 0;
                for (size_t i = 0; i < np; i++) { double d = (double)gg[1][st][i] - gg[0][st][i]; d2 += d * d; r2 += (double)gg[0][st][i] * gg[0][st][i]; }
                rel[st] = sqrt(d2 / r2);
            }
            const int ok = !l3[0] && l3[1] && !lneq && rel[0] < 1e-6 && rel[1] < 1e-6 && ctr[0][0] == ctr[1][0] && ctr[0][1] == ctr[1][1] && mem[1] < mem[0] && tail_ok;
            char nm[96]; snprintf(nm, sizeof nm, "three level-0 buffers {%d,%d,%d,%d}%s vs four", cfg.widths[0], cfg.widths[1], cfg.widths[2], cfg.widths[3], cfg.down_norm ? " dn" : "");
            printf("  %-46s layout %d, logits differ %zu, grads rel diff %.3g / %.3g, seeds %u %u / %u %u, train bytes %zu -> %zu (%.2f B / voxel less), batch tail %s%s\n",
                   nm, l3[1], lneq, rel[0], rel[1], ctr[0][0], ctr[0][1], ctr[1][0], ctr[1][1], mem[0], mem[1], ((double)mem[0] - (double)mem[1]) / S, tail_ok ? "kept" : "CLOBBERED", ok ? "" : "  FAIL");
            if (!ok) bad++;
            if (wc == 0) {   /* dec[0]'s conv1 weight gradient split around the skip's copy (UFSM_L0_3BUF_SPLIT) and enc[0]'s conv2 backward-data
                                before its weight gradient off (UFSM_L0_3BUF_REORDER=0): the same gradients and seeds as the default l3b step */
                for (int var = 0; var < 2; var++) {
                    unet_set_l0_3buf(1); if (var) unet_set_l0_3buf_reorder(0); else unet_set_l0_3buf_split(1);
                    void *tp = nullptr; unet_batch_scratch2(v, 2 * S, tailb, &tp);
                    nn_set_sr_step(9000); const float *lg = unet_forward(v, xi[0], xs, 1); unet_zero_grad(v);
                    nn_f32_to_h16(gy[0], no, (void *)lg, nn_get_grad_scale());
                    nn_set_sr_step(9100); unet_backward_x(v, lg, 1);
                    const unsigned c = nn_sr_ctr(); float *gv = malloc(np * 4); unet_grad_d2h(v, gv);
                    double d2 = 0, r2 = 0; for (size_t i = 0; i < np; i++) { double d = (double)gv[i] - gg[1][0][i]; d2 += d * d; r2 += (double)gg[1][0][i] * gg[1][0][i]; }
                    const double r = sqrt(d2 / r2); const int okv = unet_l0_3buf(v) && r < 1e-6 && c == ctr[1][0];
                    printf("  %-46s grads rel diff %.3g, seeds %u / %u%s\n", var ? "three buffers: enc[0] conv2 in the usual order" : "three buffers: split dec[0] conv1 weight grad.", r, c, ctr[1][0], okv ? "" : "  FAIL");
                    if (!okv) bad++;
                    free(gv); unet_set_l0_3buf_split(-1); unet_set_l0_3buf_reorder(-1);
                }
            }
            for (int on = 0; on < 2; on++) { for (int k = 0; k < 3; k++) free(lgs[on][k]); for (int st = 0; st < 2; st++) free(gg[on][st]); }
            unet_set_l0_3buf(-1); unet_free(v); nn_free(own); free(pat); free(chk);
            for (int k = 0; k < 2; k++) { nn_free(xi[k]); nn_free(gy[k]); }
        }
        unet_set_a0_share(-1); unet_set_share_enc_a1(0); unet_set_grad_mx4(0); unet_set_grad_mx8(0); unet_set_lean(0);
    }
    const char *e = nn_check(); if (e) { printf("cuda: %s\n", e); bad++; }
    printf("gradient MX-fp4: %s\n", bad ? "FAIL" : "ok");
    return bad != 0;
}
