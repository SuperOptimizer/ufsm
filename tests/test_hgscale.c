/* the head-gradient plane scale (nn_set_head_gscale, audit 2026-11-14 kernels.md #1 / #2, net.md #1): the loss-gradient producers write
   their fp16 planes at gradient scale x s (s a power of two), the head-plane consumers divide s out. Checks:
   (1) nn_wind4_grad at try6's scale (lambda 0.25, gscale 1024, N 1e8 supervised voxels: 128 / N = 1.3e-6 a voxel, fp16 subnormal): at s 1
       a dead zone (exact zeros where the fp32 gradient is not) is there; at s 2^20 every gradient is within fp16's normal rounding of the
       fp32 reference and no nonzero one is lost; where the s 1 value was normal, the s value is exactly s times it.
   (2) nn_fstem_bwd on planes G x s with s set == on G with s 1, byte for byte (the fp32 parameter gradients).
   (3) nn_head_bwd_z (the latent head's backward: MX-fp4 x 256 ch, 160 outputs; and the fused 32 -> 16 kernel) on gy x s with s set
       against gy with s 1: gw / gb (pre-filled: the accumulation onto earlier windows' sums) and the MX-fp4 gx, byte for byte when the
       s 1 run repeats its own bytes (else rel 1e-6), two slab ranges (first and later slab: the transposed weights' reuse).
   Usage: test_hgscale */
#include "nn.h"
#include "nn_lp.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int fails;
static void check(const char *what, int ok) { printf("  %-74s %s\n", what, ok ? "ok" : "FAIL"); if (!ok) fails++; }
static uint64_t rs = 88172645463325252ull;
static float frand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 40) / 16777216.0 - 0.5); }   /* [-0.5, 0.5) */
static float h2f(uint16_t h) { _Float16 f; memcpy(&f, &h, 2); return (float)f; }
static uint16_t f2h(float v) { _Float16 f = (_Float16)v; uint16_t h; memcpy(&h, &f, 2); return h; }
static float *dev_vals(size_t n, float sc, float off) { float *h = malloc(n * 4); for (size_t i = 0; i < n; i++) h[i] = off + frand() * sc; float *d = nn_malloc(n * 4); nn_h2d(d, h, n * 4); free(h); return d; }
static void *mx4_of(shape5 s, float sc) { float *f = dev_vals(shape_numel(s), sc, 0.f); size_t b = nn_mx4_bytes(s); void *p = nn_malloc(b); nn_zero(p, b); nn_set_storage(p, b, 4); lp_f32_to_mx4(f, s.n, s.c, shape_spatial(s), p); nn_free(f); return p; }
static int same_or_close(const void *a, const void *b, size_t nfloat, int bytes_expected) {   /* fp32 arrays: equal bytes, or (non-repeating reductions) rel 1e-6 */
    if (!memcmp(a, b, nfloat * 4)) return 1;
    if (bytes_expected) return 0;
    const float *x = a, *y = b; double d = 0, r = 0; for (size_t i = 0; i < nfloat; i++) { d += ((double)x[i] - y[i]) * ((double)x[i] - y[i]); r += (double)y[i] * y[i]; }
    return sqrt(d / (r > 0 ? r : 1)) < 1e-6;
}

static void wind4_case(float N, float S) {   /* N: the supervised count (1e8: P 768; 1e3: every value normal at both scales), S: the plane scale */
    printf("(1) nn_wind4_grad: lambda 0.25, gscale 1024, N %.0e, plane scale %.0f\n", N, S);
    const size_t n = 1 << 18; const float lam = 0.25f, gs = 1024.f;
    nn_set_grad_scale(gs); nn_set_logits_h16(1); nn_set_loss_grad_h16(1);
    uint16_t *la = malloc(n * 2), *lb = malloc(n * 2); uint8_t *t = malloc(n), *cp = malloc(n), *m = malloc(n);
    for (size_t i = 0; i < n; i++) {   /* logits near their targets (a trained head: small residuals), some far */
        t[i] = (uint8_t)(i * 7 % 252); cp[i] = 0; m[i] = (uint8_t)(1 + (i % 5 == 0) * 7);   /* mask weights 1 and 8 (packed sheets) */
        const double tg = 6.283185307179586 * (t[i] / 252.0), ta = 0.5 + 0.5 * cos(tg), tb = 0.5 + 0.5 * sin(tg);   /* (component 0, gauge fin[8] = 0) */
        const double e = (i % 3 == 0) ? 0.3 * frand() : 0.003 * frand();
        const double pa = fmin(fmax(ta + e, 1e-3), 1 - 1e-3), pb = fmin(fmax(tb - e, 1e-3), 1 - 1e-3);
        la[i] = f2h((float)log(pa / (1 - pa))); lb[i] = f2h((float)log(pb / (1 - pb)));
    }
    const size_t nf = nn_wind4_scratch() / 4; float *fin = calloc(nf, 4); fin[3] = 1.f / N;   /* fin[8 + c] = 0 .. 3: the gauge; component 0 only */
    for (size_t c = 8; c < nf; c++) fin[c] = (float)((c - 8) % 4);
    void *dla = nn_malloc(n * 2), *dlb = nn_malloc(n * 2), *ga = nn_malloc(n * 2), *gb = nn_malloc(n * 2); uint8_t *dt = nn_malloc(n), *dc = nn_malloc(n), *dm = nn_malloc(n); float *dfin = nn_malloc(nf * 4);
    nn_h2d(dla, la, n * 2); nn_h2d(dlb, lb, n * 2); nn_h2d(dt, t, n); nn_h2d(dc, cp, n); nn_h2d(dm, m, n); nn_h2d(dfin, fin, nf * 4);
    uint16_t *g1a = malloc(n * 2), *g1b = malloc(n * 2), *g2a = malloc(n * 2), *g2b = malloc(n * 2);
    nn_set_head_gscale(1.f); nn_wind4_grad(dla, dlb, dt, dc, dm, n, dfin, lam, ga, gb); nn_sync(); nn_d2h(g1a, ga, n * 2); nn_d2h(g1b, gb, n * 2);
    nn_set_head_gscale(S); nn_wind4_grad(dla, dlb, dt, dc, dm, n, dfin, lam, ga, gb); nn_sync(); nn_d2h(g2a, ga, n * 2); nn_d2h(g2b, gb, n * 2);
    nn_set_head_gscale(1.f);
    size_t dead1 = 0, dead2 = 0, nz = 0, bad = 0, exact_bad = 0, nnorm = 0; double worst = 0;
    for (size_t i = 0; i < n; i++) for (int ab = 0; ab < 2; ab++) {
        const double tg = 6.283185307179586 * (t[i] / 252.0 + 0.25 * fin[8 + cp[i]]), tv = ab ? 0.5 + 0.5 * sin(tg) : 0.5 + 0.5 * cos(tg);   /* (the kernel's gauge: fin[8 + component]; the reference had i % 4, a target the kernel never uses: 1e4 relative "errors", sweep 11-14) */
        const double x = h2f(ab ? lb[i] : la[i]), want = m[i] * 0.5 * (1.0 / N) * lam * gs * (1 / (1 + exp(-x)) - tv);   /* (the s 1 scale) */
        const float v1 = h2f(ab ? g1b[i] : g1a[i]), v2 = h2f(ab ? g2b[i] : g2a[i]) / S;
        if (fabs(want) * S > 1e-3) { nz++; dead1 += v1 == 0.f; dead2 += v2 == 0.f; }   /* (gradients s puts well inside fp16's normal range) */
        const double e = fabs(v2 - want), tol = 1e-3 * fabs(want) + 2e-3 * m[i] * 0.5 * (1.0 / N) * lam * gs;   /* (fp16 rounding; __expf / sincosf in the kernel) */
        if (e > tol) bad++;
        if (e / (fabs(want) + 1e-30) > worst && fabs(want) > 1e-9) worst = e / fabs(want);
        if (fabsf(v1) >= 6.103515625e-05f) { nnorm++; if (v2 != v1) exact_bad++; }   /* normal at s 1: the s value is s times it, exactly */
    }
    printf("  %zu nonzero gradients: exact zeros at s 1 %zu (%.1f%%), at s %.0f %zu; s 1 normal values %zu; worst relative error %.2e\n", nz, dead1, 100.0 * dead1 / (nz ? nz : 1), S, dead2, nnorm, worst);
    if (N > 1e6f) check("s 1: the fp16 dead zone is there (the bug this fixes)", dead1 > 0);
    check("plane scale: no nonzero gradient lost", dead2 == 0);
    check("plane scale: every gradient within fp16 rounding of the fp32 reference", bad == 0);
    if (N < 1e6f) check("s 1 normal values: the scaled value / s equals them exactly", nnorm > 0 && exact_bad == 0);
    free(la); free(lb); free(t); free(cp); free(m); free(fin); free(g1a); free(g1b); free(g2a); free(g2b);
    nn_free(dla); nn_free(dlb); nn_free(ga); nn_free(gb); nn_free(dt); nn_free(dc); nn_free(dm); nn_free(dfin);
}

static void fstem_case(void) {
    printf("(2) nn_fstem_bwd: planes G x s with s set == G with s 1\n");
    enum { D = 10, H = 12, W = 14, C = 16, NO = 7, NCH = 9, T = 64 };
    static const int chs[NO] = {0, 1, 2, 5, 6, 7, 8};
    const size_t V = (size_t)D * H * W, NP = C * T + C + NO * C, ns = V;
    uint16_t *x = malloc(V * 2), *G = malloc(NCH * V * 2), *Gs = malloc(NCH * V * 2); float *pf = malloc(NP * 4), *g1 = malloc(NP * 4), *g2 = malloc(NP * 4), *g3 = malloc(NP * 4);
    const float S = 4096.f;
    for (size_t i = 0; i < V; i++) x[i] = f2h(2.f * frand());
    for (size_t i = 0; i < NCH * V; i++) { G[i] = f2h(0.01f * frand()); Gs[i] = f2h(h2f(G[i]) * S); }   /* (|G| <= 5e-3 x 4096 = 20: exact in fp16, normal unless tiny) */
    for (size_t i = 0; i < NP; i++) pf[i] = 0.4f * frand();
    void *xd = nn_malloc(V * 2), *gd = nn_malloc(NCH * V * 2), *gsd = nn_malloc(NCH * V * 2); float *pd = nn_malloc(NP * 4), *gpd = nn_malloc(NP * 4);
    nn_h2d(xd, x, V * 2); nn_h2d(gd, G, NCH * V * 2); nn_h2d(gsd, Gs, NCH * V * 2); nn_h2d(pd, pf, NP * 4);
    float *pre = dev_vals(NP, 0.02f, 0.f); float *hpre = malloc(NP * 4); nn_d2h(hpre, pre, NP * 4);   /* (a gradient already holding earlier sums) */
    int rc = 0;
    nn_h2d(gpd, hpre, NP * 4); nn_set_head_gscale(1.f); rc |= nn_fstem_bwd(xd, D, H, W, 0, D, pd, C, NO, chs, gd, ns, gpd); nn_sync(); nn_d2h(g1, gpd, NP * 4);
    nn_h2d(gpd, hpre, NP * 4); rc |= nn_fstem_bwd(xd, D, H, W, 0, D, pd, C, NO, chs, gd, ns, gpd); nn_sync(); nn_d2h(g3, gpd, NP * 4);
    nn_h2d(gpd, hpre, NP * 4); nn_set_head_gscale(S); rc |= nn_fstem_bwd(xd, D, H, W, 0, D, pd, C, NO, chs, gsd, ns, gpd); nn_sync(); nn_d2h(g2, gpd, NP * 4);
    nn_set_head_gscale(1.f);
    check("nn_fstem_bwd accepted the shapes", rc == 0);
    const int rep = !memcmp(g1, g3, NP * 4);
    printf("  s 1 repeats its own bytes: %s\n", rep ? "yes" : "no (rel 1e-6 below)");
    check("fstem parameter gradients: s 2^12 planes == s 1 planes", same_or_close(g2, g1, NP, rep));
    free(x); free(G); free(Gs); free(pf); free(g1); free(g2); free(g3); free(hpre); nn_free(xd); nn_free(gd); nn_free(gsd); nn_free(pd); nn_free(gpd); nn_free(pre);
}

static void head_case(int C, int K, int D, int H, int W) {
    printf("(3) nn_head_bwd_z: %d MX-fp4 channels -> %d outputs, %d x %d x %d\n", C, K, D, H, W);
    shape5 xs = {1, C, D, H, W}; const size_t hw = (size_t)H * W, S = (size_t)D * hw;
    const float sc = 1024.f;
    void *x = mx4_of(xs, 3.f);
    float *w = dev_vals((size_t)K * C, 0.1f, 0.f);
    uint16_t *gy = malloc((size_t)K * S * 2), *gys = malloc((size_t)K * S * 2);
    for (size_t i = 0; i < (size_t)K * S; i++) { gy[i] = f2h(4e-3f * frand()); gys[i] = f2h(h2f(gy[i]) * sc); }
    void *gyd = nn_malloc((size_t)K * S * 2), *gysd = nn_malloc((size_t)K * S * 2); nn_h2d(gyd, gy, (size_t)K * S * 2); nn_h2d(gysd, gys, (size_t)K * S * 2);
    const size_t nw = (size_t)K * C, gxb = nn_mx4_bytes(xs);
    float *pre = dev_vals(nw + K, 0.05f, 0.f); float *hpre = malloc((nw + K) * 4); nn_d2h(hpre, pre, (nw + K) * 4);
    float *gw = nn_malloc(nw * 4), *gb = nn_malloc((size_t)K * 4); void *gx = nn_malloc(gxb);
    float *rw[4], *rb[4]; uint8_t *rx[4]; int rc = 0;
    const unsigned c0 = nn_sr_ctr();
    for (int r = 0; r < 4; r++) {   /* r 0, 1: s 1 twice (does it repeat its bytes?), r 2: gy x 1024 at s 1024, r 3: s 1 with one output tile per block (lp_set_w1_nt 0) */
        rw[r] = malloc(nw * 4); rb[r] = malloc((size_t)K * 4); rx[r] = malloc(gxb);
        nn_h2d(gw, hpre, nw * 4); nn_h2d(gb, hpre + nw, (size_t)K * 4); nn_zero(gx, gxb); nn_set_storage(gx, gxb, 4);
        nn_sr_set_ctr(c0); nn_set_head_gscale(r == 2 ? sc : 1.f); const int nt_was = lp_set_w1_nt(r == 3 ? 0 : -1);
        const void *g = r == 2 ? gysd : gyd;
        const int z1 = D / 2;   /* two slab ranges: planes [0, z1) (first: the transposed weights built) and [z1, D) (reused) */
        rc |= nn_head_bwd_z((const float *)x, nullptr, xs, g, K, w, gw, gb, (float *)gx, 0, (size_t)z1 * hw, 1);
        rc |= nn_head_bwd_z((const float *)x, nullptr, xs, g, K, w, gw, gb, (float *)gx, (size_t)z1 * hw, (size_t)(D - z1) * hw, 0);
        nn_sync(); nn_d2h(rw[r], gw, nw * 4); nn_d2h(rb[r], gb, (size_t)K * 4); nn_d2h(rx[r], gx, gxb); lp_set_w1_nt(nt_was);
    }
    nn_set_head_gscale(1.f);
    check("nn_head_bwd_z covered the shapes", rc == 0);
    const int repw = !memcmp(rw[0], rw[1], nw * 4) && !memcmp(rb[0], rb[1], (size_t)K * 4), repx = !memcmp(rx[0], rx[1], gxb);
    printf("  s 1 repeats its own bytes: weights %s, gx %s\n", repw ? "yes" : "no (rel 1e-6 below)", repx ? "yes" : "no");
    char nm[128];
    snprintf(nm, sizeof nm, "head weight / bias gradients: s 1024 == s 1 (%d -> %d)", C, K);
    check(nm, same_or_close(rw[2], rw[0], nw, repw) && same_or_close(rb[2], rb[0], (size_t)K, repw));
    snprintf(nm, sizeof nm, "head gx (MX-fp4 bytes): s 1024 == s 1 (%d -> %d)", C, K);
    check(nm, repx ? !memcmp(rx[2], rx[0], gxb) : 1);
    snprintf(nm, sizeof nm, "head weight / bias gradients: 4 output tiles a block == 1 (%d -> %d)", C, K);
    check(nm, same_or_close(rw[3], rw[0], nw, repw) && same_or_close(rb[3], rb[0], (size_t)K, repw));
    for (int r = 0; r < 4; r++) { free(rw[r]); free(rb[r]); free(rx[r]); }
    free(gy); free(gys); free(hpre); nn_free(x); nn_free(w); nn_free(gyd); nn_free(gysd); nn_free(pre); nn_free(gw); nn_free(gb); nn_free(gx);
}

int main(void) {
    nn_init(0); nn_set_prec(3); nn_set_tf32(1); nn_set_f16(1);
    wind4_case(1e8f, 1048576.f);   /* P 768: 128 / N = 1.3e-6, s 2^20 */
    wind4_case(1e3f, 16.f);        /* every value normal at both scales: exact x s */
    fstem_case();
    head_case(256, 160, 8, 24, 24);   /* the latent head's path (> 16 outputs: lp_bwd_w1_mx_b_r + lp_conv1_to_mx_r on the transposed weights) */
    head_case(32, 16, 8, 24, 24);     /* the fused kernel (lp_head_bwd_mx_r) */
    printf(fails ? "test_hgscale: %d FAILED\n" : "test_hgscale: ok\n", fails);
    return fails != 0;
}
