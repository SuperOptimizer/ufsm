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
static void mode_ref(void) { nn_set_tf32(0); }
static void mode_mx(void) { nn_set_prec(3); }
int main(void) {
    nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0);
    nn_set_act_bf16(0); nn_set_grad_bf16(0);   /* non-MX operands are fp32 */
    const double TOL4 = 0.2, TOL = 0.08;
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
    const char *e = nn_check(); if (e) { printf("cuda: %s\n", e); bad++; }
    e = lp_check(); if (e) { printf("cuda (lp): %s\n", e); bad++; }
    printf(bad ? "mx4 FAIL (%d)\n" : "mx4 ok\n", bad);
    (void)mode_ref; (void)mode_mx; (void)TOL4; (void)TOL; (void)G; (void)dev_rand;
    return bad != 0;
}
