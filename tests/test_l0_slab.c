/* Slab-local GroupNorm readers (level-0 slab streaming plan stage S1; gnp_t ns > 0 via lp_gn_slab_reg): statistics per (sample,
   z-slab), each plane normalised with its own slab's. The readers that apply them (so far: the head's forward / weight gradient /
   fused backward, lp_mx_conv1.cu; the skip half of the 16-wide decoder conv1, lp_f4up.cu) against fp32 references computed with the
   right slab per plane on the dequantised inputs; slabs are given very different statistics, so a reader using another slab's (or
   the sample's) is off by ~100%. Then the unet kernel path (UFSM_L0_SLAB_KERN=1) with one slab per window is GroupNorm bit for bit. */
#include "nn.h"
#include "nn_lp.h"
#include "unet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int bad;
static uint64_t rs = 4242;
static float frand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 11) * 0x1.0p-53 * 2 - 1); }
static float *dev_from(const float *h, size_t n) { float *d = nn_malloc(n * 4); nn_h2d(d, h, n * 4); return d; }
static float *dev_rand(size_t n, float scale) { float *h = malloc(n * 4); for (size_t i = 0; i < n; i++) h[i] = frand() * scale; float *d = dev_from(h, n); free(h); return d; }
static float *dev_zero(size_t n) { float *d = nn_malloc(n * 4); nn_zero(d, n * 4); return d; }
static void *mx4_new(shape5 s) { size_t b = nn_mx4_bytes(s); void *p = nn_malloc(b); nn_zero(p, b); nn_set_storage(p, b, 4); return p; }
static void *mx4_from(const float *x, shape5 s) { void *p = mx4_new(s); lp_f32_to_mx4(x, s.n, s.c, shape_spatial(s), p); return p; }
static float *deq4_h(const void *p, shape5 s) {   /* host copy of a dequantised mx4 tensor */
    const size_t n = shape_numel(s); float *d = dev_zero(n), *h = malloc(n * 4);
    lp_mx4_to_f32(p, s.n, s.c, shape_spatial(s), d); nn_d2h(h, d, n * 4); nn_free(d); return h;
}
static float h2f(uint16_t h) {
    const uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023;
    uint32_t f;
    if (e == 0) { if (!m) f = s; else { int k = 0; uint32_t mm = m; while (!(mm & 1024)) { mm <<= 1; k++; } f = s | ((uint32_t)(113 - k) << 23) | ((mm & 1023) << 13); } }
    else if (e == 31) f = s | 0x7f800000u | (m << 13);
    else f = s | ((e + 112) << 23) | (m << 13);
    float r; memcpy(&r, &f, 4); return r;
}
static double rel(const float *a, const float *r, size_t n) {
    double d2 = 0, r2 = 0; for (size_t i = 0; i < n; i++) { const double d = (double)a[i] - r[i]; d2 += d * d; r2 += (double)r[i] * r[i]; }
    return r2 > 0 ? sqrt(d2 / r2) : 1e9;
}
static void check(const char *what, double e, double tol) { printf("  %-62s rel err %.3g%s\n", what, e, e < tol ? "" : "  FAIL"); if (!(e < tol)) bad++; }
static float silu(float v) { return v / (1.f + expf(-v)); }

/* a level-0 tensor whose slab s has mean ~ 2 s and scale ~ 1 + s, and its statistics per (slab, group) or per sample (ns 0) */
static float *slab_input(shape5 s, int S) {
    const size_t n = shape_numel(s), P = (size_t)s.h * s.w; float *h = malloc(n * 4);
    for (int c = 0; c < s.c; c++) for (int z = 0; z < s.d; z++) for (size_t v = 0; v < P; v++) {
        const int sl = z / S; h[((size_t)c * s.d + z) * P + v] = 2.f * sl + (1.f + sl) * (frand() + 0.3f * (float)(c % 3));
    }
    return h;
}
static void stats(const float *x, shape5 s, int G, int S, int ns, float *mean, float *rstd) {   /* ns 0: per sample */
    const size_t P = (size_t)s.h * s.w; const int cpg = s.c / G, nr = ns ? ns : 1, zs = ns ? S : s.d;
    for (int r = 0; r < nr; r++) for (int g = 0; g < G; g++) {
        double a = 0, q = 0, cnt = 0;
        for (int c = g * cpg; c < (g + 1) * cpg; c++) for (int z = r * zs; z < (r + 1) * zs && z < s.d; z++) for (size_t v = 0; v < P; v++) {
            const double t = x[((size_t)c * s.d + z) * P + v]; a += t; q += t * t; cnt++;
        }
        const double m = a / cnt, var = q / cnt - m * m;
        mean[r * G + g] = (float)m; rstd[r * G + g] = (float)(1.0 / sqrt(var + 1e-5));
    }
}
/* silu(gn(x)) with each plane's slab (ns 0: the sample's statistics) */
static void act(const float *x, shape5 s, int G, int S, int ns, const float *gam, const float *bet, const float *mean, const float *rstd, float *y) {
    const size_t P = (size_t)s.h * s.w; const int cpg = s.c / G;
    for (int c = 0; c < s.c; c++) for (int z = 0; z < s.d; z++) {
        const int r = ns ? z / S : 0, k = r * G + c / cpg;
        for (size_t v = 0; v < P; v++) { const size_t i = ((size_t)c * s.d + z) * P + v; y[i] = silu((x[i] - mean[k]) * rstd[k] * gam[c] + bet[c]); }
    }
}

static void test_head(int C) {
    const int S = 8, G = C / 4, cout = 2;
    const shape5 xs = {1, C, 32, 12, 16}, ys = {1, cout, 32, 12, 16};
    const int ns = xs.d / S;
    const size_t n = shape_numel(xs), V = shape_spatial(xs), P = (size_t)xs.h * xs.w;
    float *hx = slab_input(xs, S), *hg = malloc(C * 4), *hb = malloc(C * 4), *hw = malloc((size_t)cout * C * 4), *hbias = malloc(cout * 4);
    for (int c = 0; c < C; c++) { hg[c] = 0.7f + 0.5f * frand(); hb[c] = 0.3f * frand(); }
    for (int i = 0; i < cout * C; i++) hw[i] = 0.3f * frand();
    for (int i = 0; i < cout; i++) hbias[i] = 0.1f * frand();
    float *dx = dev_from(hx, n); void *xm = mx4_from(dx, xs); float *xd = deq4_h(xm, xs);
    float *hm = malloc((size_t)ns * G * 4), *hr = malloc((size_t)ns * G * 4), *hm0 = malloc(G * 4), *hr0 = malloc(G * 4);
    stats(xd, xs, G, S, ns, hm, hr); stats(xd, xs, G, S, 0, hm0, hr0);
    float *gam = dev_from(hg, C), *bet = dev_from(hb, C), *mean = dev_from(hm, (size_t)ns * G), *rstd = dev_from(hr, (size_t)ns * G);
    float *w = dev_from(hw, (size_t)cout * C), *bias = dev_from(hbias, cout);
    float *a = malloc(n * 4), *a0 = malloc(n * 4);
    act(xd, xs, G, S, ns, hg, hb, hm, hr, a); act(xd, xs, G, S, 0, hg, hb, hm0, hr0, a0);
    lp_gn_slab_reg(mean, S, 0, ns, (int)P);
    const gnp_t gp = gnp_mk(gam, bet, mean, rstd, G);
    if (gp.ns != ns || gp.zhw != (int)P) { printf("  gnp_mk: registry geometry not applied  FAIL\n"); bad++; }
    /* forward, fp16 logits: the whole window and a head slab [vb, vb + nv) not aligned to planes */
    const size_t ranges[2][2] = {{0, 0}, {1000, 3000}};
    for (int q = 0; q < 2; q++) {
        const size_t vb = ranges[q][0], nv = ranges[q][1] ? ranges[q][1] : V;
        uint16_t *y16 = nn_malloc((size_t)cout * nv * 2), *hy = malloc((size_t)cout * nv * 2);
        lp_conv1_fwd_mx_h16_r(xm, 4, xs, w, bias, cout, y16, gp, ranges[q][1] ? vb : 0, ranges[q][1]);
        nn_sync(); nn_d2h(hy, y16, (size_t)cout * nv * 2);
        float *yk = malloc((size_t)cout * nv * 4), *yr = malloc((size_t)cout * nv * 4), *yw = malloc((size_t)cout * nv * 4);
        for (int co = 0; co < cout; co++) for (size_t v = 0; v < nv; v++) {
            double s = hbias[co], s0 = hbias[co];
            for (int c = 0; c < C; c++) { s += (double)hw[co * C + c] * a[(size_t)c * V + vb + v]; s0 += (double)hw[co * C + c] * a0[(size_t)c * V + vb + v]; }
            yk[co * nv + v] = h2f(hy[co * nv + v]); yr[co * nv + v] = (float)s; yw[co * nv + v] = (float)s0;
        }
        char nm[96];
        snprintf(nm, sizeof nm, "head forward %d ch, %d slabs, voxels [%zu, %zu)", C, ns, vb, vb + nv); check(nm, rel(yk, yr, (size_t)cout * nv), 2e-3);
        snprintf(nm, sizeof nm, "  (the same vs the sample's statistics: must be far)"); { const double e = rel(yk, yw, (size_t)cout * nv); printf("  %-62s rel err %.3g%s\n", nm, e, e > 0.1 ? "" : "  FAIL"); if (!(e > 0.1)) bad++; }
        nn_free(y16); free(hy); free(yk); free(yr); free(yw);
    }
    /* weight gradient (fp16 gy): gw[co][ci] = sum_v act(x)[ci][v] gy[co][v]; the 16-channel tensor-core kernel / the 32-channel one */
    uint16_t *hgy = malloc((size_t)cout * V * 2); float *gyf = malloc((size_t)cout * V * 4);
    for (size_t i = 0; i < (size_t)cout * V; i++) {   /* k / 1024, |k| <= 512: exact in fp16 */
        const int k = (int)lrintf(512.f * frand()); const float f = (float)k / 1024.f;
        uint32_t u; memcpy(&u, &f, 4);
        hgy[i] = k ? (uint16_t)(((u >> 16) & 0x8000) | ((((u >> 23) & 255) - 112) << 10) | ((u >> 13) & 1023)) : 0;
        gyf[i] = h2f(hgy[i]);
        if (gyf[i] != f) { printf("  fp16 encoding of the test gradient  FAIL\n"); bad++; break; }
    }
    uint16_t *gy16 = nn_malloc((size_t)cout * V * 2); nn_h2d(gy16, hgy, (size_t)cout * V * 2);
    double *rw = calloc((size_t)cout * C, sizeof(double)), *rb = calloc(cout, sizeof(double));
    for (int co = 0; co < cout; co++) for (size_t v = 0; v < V; v++) { const double g = gyf[co * V + v]; rb[co] += g; for (int c = 0; c < C; c++) rw[co * C + c] += g * a[(size_t)c * V + v]; }
    float *rwf = malloc((size_t)cout * C * 4), *hgw = malloc((size_t)cout * C * 4); for (int i = 0; i < cout * C; i++) rwf[i] = (float)rw[i];
    float *gw = dev_zero((size_t)cout * C), *gb = dev_zero(cout);
    (void)lp_bwd_w1_mx_b(xm, 4, xs, gy16, 2, ys, gw, gb, gp);
    nn_sync(); nn_d2h(hgw, gw, (size_t)cout * C * 4);
    char nm[96]; snprintf(nm, sizeof nm, "head weight gradient %d ch, %d slabs", C, ns); check(nm, rel(hgw, rwf, (size_t)cout * C), 5e-3);
    if (C == 32) {   /* the fused head backward (weight, bias, input gradient in one pass) */
        void *gx = mx4_new(xs); nn_zero(gw, (size_t)cout * C * 4); nn_zero(gb, cout * 4);
        const int done = lp_head_bwd_mx_r(xm, 4, xs, gy16, 2, ys, w, gw, gb, gx, 4, gp, 0u, 0, 0);
        nn_sync(); nn_d2h(hgw, gw, (size_t)cout * C * 4);
        if (!done) { printf("  fused head backward not taken  FAIL\n"); bad++; }
        else { snprintf(nm, sizeof nm, "fused head backward weight gradient %d ch, %d slabs", C, ns); check(nm, rel(hgw, rwf, (size_t)cout * C), 5e-3); }
        nn_free(gx);
    }
    lp_gn_slab_reg(mean, 0, 0, 0, 0);
    { const gnp_t g0 = gnp_mk(gam, bet, mean, rstd, G); if (g0.ns) { printf("  lp_gn_slab_reg: entry not removed  FAIL\n"); bad++; } }
    nn_free(dx); nn_free(xm); nn_free(gam); nn_free(bet); nn_free(mean); nn_free(rstd); nn_free(w); nn_free(bias); nn_free(gy16); nn_free(gw); nn_free(gb);
    free(hx); free(hg); free(hb); free(hw); free(hbias); free(xd); free(hm); free(hr); free(hm0); free(hr0); free(a); free(a0); free(hgy); free(gyf); free(rw); free(rb); free(rwf); free(hgw);
}

/* the 16-wide decoder conv1 (lp_f4up.cu): [up2(coarse) | silu(gn(skip))] with the skip's statistics per slab */
static void test_f4u(int S) {
    const int G = 4;
    nn_wmemo_clear();   /* (the prepared-weight memo is keyed by pointer: the previous call's freed weights) */
    const shape5 fs = {1, 32, 32, 12, 16}, co = {1, 16, 16, 6, 8}, sk = {1, 16, 32, 12, 16}, o16 = {1, 16, 32, 12, 16};
    const int ns = sk.d / S;
    const size_t nsk = shape_numel(sk), P = (size_t)sk.h * sk.w;
    float *hs = slab_input(sk, S);
    float *dsk = dev_from(hs, nsk); void *xsm = mx4_from(dsk, sk); float *xsd = deq4_h(xsm, sk);
    float *xc = dev_rand(shape_numel(co), 1.f); void *xcm = mx4_from(xc, co);
    float *xcd = dev_zero(shape_numel(co)); lp_mx4_to_f32(xcm, co.n, co.c, shape_spatial(co), xcd);
    float *hg = malloc(64), *hb = malloc(64); for (int c = 0; c < 16; c++) { hg[c] = 0.7f + 0.5f * frand(); hb[c] = 0.3f * frand(); }
    float *hm = malloc((size_t)ns * G * 4), *hr = malloc((size_t)ns * G * 4), *hm0 = malloc(G * 4), *hr0 = malloc(G * 4);
    stats(xsd, sk, G, S, ns, hm, hr); stats(xsd, sk, G, S, 0, hm0, hr0);
    float *gam = dev_from(hg, 16), *bet = dev_from(hb, 16), *mean = dev_from(hm, (size_t)ns * G), *rstd = dev_from(hr, (size_t)ns * G);
    float *mean0 = dev_from(hm0, G), *rstd0 = dev_from(hr0, G);
    float *w = dev_rand((size_t)16 * 32 * 27, 0.1f), *b = dev_rand(16, 0.1f);
    float *t = malloc(nsk * 4), *ref[2], *out[2];
    for (int k = 0; k < 2; k++) {   /* 0: slabs, 1: per sample (the baseline error of the fp4 kernel against fp32) */
        act(xsd, sk, G, S, k ? 0 : ns, hg, hb, k ? hm0 : hm, k ? hr0 : hr, t);
        float *dt = dev_from(t, nsk), *up = dev_zero(nsk), *cat = dev_zero(shape_numel(fs)), *yr = dev_zero(shape_numel(o16));
        nn_set_tf32(0); nn_up2_fwd_into(xcd, co, up, 16, 0); nn_concat_fwd(up, 16, dt, 16, fs, cat); nn_conv3d_fwd(cat, fs, w, b, 16, 3, 1, yr);
        ref[k] = malloc(shape_numel(o16) * 4); nn_d2h(ref[k], yr, shape_numel(o16) * 4);
        if (!k) lp_gn_slab_reg(mean, S, 0, ns, (int)P);
        nn_gn_t g2 = {gam, bet, k ? mean0 : mean, k ? rstd0 : rstd, G};
        void *ym = mx4_new(o16);
        nn_set_prec(3); nn_set_tf32(1);
        const int rv = nn_conv3d_fwd_x(xcm, NULL, xsm, &g2, 16, 1, fs, w, b, 16, 3, 1, (float *)ym, 0, 1e-5f, NULL, NULL);
        if (rv) { printf("  f4u: unsupported (rv %d)  FAIL\n", rv); bad++; }
        out[k] = deq4_h(ym, o16);
        if (!k) lp_gn_slab_reg(mean, 0, 0, 0, 0);
        nn_free(dt); nn_free(up); nn_free(cat); nn_free(yr); nn_free(ym);
    }
    const size_t no = shape_numel(o16);
    const double es = rel(out[0], ref[0], no), e0 = rel(out[1], ref[1], no), ex = rel(out[0], ref[1], no);
    char nm[96];
    snprintf(nm, sizeof nm, "dec0.c1 (f4u) skip slab GN, %d-plane slabs (%d): vs fp32", S, ns); check(nm, es, e0 + 0.02);
    printf("  %-62s rel err %.3g (baseline for the line above)\n", "dec0.c1 (f4u) per-sample GN: vs fp32", e0);
    { const double e = ex; printf("  %-62s rel err %.3g%s\n", "  (slab output vs the per-sample reference: must be far)", e, e > 0.1 ? "" : "  FAIL"); if (!(e > 0.1)) bad++; }
    for (int k = 0; k < 2; k++) { free(ref[k]); free(out[k]); }
    nn_free(dsk); nn_free(xsm); nn_free(xc); nn_free(xcm); nn_free(xcd); nn_free(gam); nn_free(bet); nn_free(mean); nn_free(rstd); nn_free(mean0); nn_free(rstd0); nn_free(w); nn_free(b);
    free(hs); free(xsd); free(hg); free(hb); free(hm); free(hr); free(hm0); free(hr0); free(t);
}

/* unet, kernel path with one slab per window: GroupNorm (cfg.l0_norm 0) bit for bit -- logits and parameter gradients */
static void test_unet_one_slab(void) {
    const shape5 xs = {1, 4, 32, 32, 32};
    const size_t nx = shape_numel(xs), no = nx / 4 * 2;
    float *h = malloc(nx * 4); for (size_t i = 0; i < nx; i++) h[i] = frand();
    float *x = dev_from(h, nx); free(h);
    float *gy = dev_rand(no, 1.f);
    float *lg[3], *gr[3]; size_t np = 0;
    for (int arm = 0; arm < 3; arm++) {   /* gn, gn again (run-to-run), slab kernel path S = D */
        unet_cfg cfg = {3, {16, 16, 32}, 4, 2, 8}; cfg.l0_norm = arm == 2 ? 4 : 0; cfg.l0_slab = 32;
        unet *u = unet_create(&cfg); unet_init(u, 41);
        np = unet_nparams(u);
        lg[arm] = malloc(no * 4); gr[arm] = malloc(np * 4);
        nn_set_sr_step(7); const float *l = unet_forward(u, x, xs, 1); nn_sync(); nn_d2h(lg[arm], l, no * 4);
        unet_zero_grad(u); nn_set_sr_step(8); unet_backward(u, gy); unet_grad_d2h(u, gr[arm]);
        unet_free(u);
    }
    /* (GroupNorm itself is not bit-reproducible run to run: atomic summation order in the statistics; the slab path must stay within
       twice the run-to-run spread, or be equal) */
    const double el = rel(lg[2], lg[0], no), el0 = rel(lg[1], lg[0], no), eg = rel(gr[2], gr[0], np), eg0 = rel(gr[1], gr[0], np);
    const int ok = el <= 2 * el0 + 1e-7 && eg <= 2 * eg0 + 1e-7;
    printf("  %-62s logits rel %.3g, gradients rel %.3g (GroupNorm run to run: %.3g, %.3g)%s\n", "unet slab kernel path, one slab vs GroupNorm",
           el, eg, el0, eg0, ok ? "" : "  FAIL");
    if (!ok) bad++;
    for (int k = 0; k < 3; k++) { free(lg[k]); free(gr[k]); }
    nn_free(x); nn_free(gy);
}

int main(void) {
    setenv("UFSM_L0_SLAB_KERN", "1", 0);
    if (nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0)) { printf("no cuda\n"); return 1; }
    nn_set_tf32(1); nn_set_f16(1); nn_set_gn_stored(1);
    test_unet_one_slab();
    nn_set_prec(3); nn_set_tf32(1);
    test_head(16); test_head(32);
    test_f4u(8); test_f4u(16);
    printf("test_l0_slab: %s\n", bad ? "FAIL" : "PASS");
    return bad != 0;
}
