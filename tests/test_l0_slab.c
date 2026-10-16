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

/* ---- the 3^3 readers (stride 1 and 2, forward and weight gradient) of a slab-normalised MX-fp4 input: each kernel family against
   an fp32 reference with the right slab per plane, its error no larger than the same kernel's with per-sample statistics (its fp4 /
   fp8 baseline), and far from the per-sample reference. kern: what nn_conv3d_fwd_x / nn_conv3d_bwd_weight_x pick at this precision
   (named in the line) */
static void *mx8_from_l(const float *x, shape5 s) { size_t b = nn_mx8_bytes(s); void *p = nn_malloc(b); nn_zero(p, b); nn_set_storage(p, b, 8); lp_f32_to_mx8(x, s.n, s.c, shape_spatial(s), p); return p; }
static float *deq8_h(const void *p, shape5 s) {
    const size_t n = shape_numel(s); float *d = dev_zero(n), *h = malloc(n * 4);
    lp_mx8_to_f32(p, s.n, s.c, shape_spatial(s), d); nn_d2h(h, d, n * 4); nn_free(d); return h;
}
typedef struct { shape5 s; int S, G, ns; void *xm; float *xd, *hg, *hb, *hm, *hr, *hm0, *hr0, *a, *a0, *gam, *bet, *mean, *rstd, *mean0, *rstd0; } sx_t;
static sx_t sx_make(int C, int S) {
    sx_t q = {0}; q.s = (shape5){1, C, 32, 12, 16}; q.S = S; q.G = C / 4; q.ns = q.s.d / S;
    const size_t n = shape_numel(q.s);
    float *hx = slab_input(q.s, S), *dx = dev_from(hx, n); q.xm = mx4_from(dx, q.s); q.xd = deq4_h(q.xm, q.s); free(hx); nn_free(dx);
    q.hg = malloc(C * 4); q.hb = malloc(C * 4); for (int c = 0; c < C; c++) { q.hg[c] = 0.7f + 0.5f * frand(); q.hb[c] = 0.3f * frand(); }
    q.hm = malloc((size_t)q.ns * q.G * 4); q.hr = malloc((size_t)q.ns * q.G * 4); q.hm0 = malloc(q.G * 4); q.hr0 = malloc(q.G * 4);
    stats(q.xd, q.s, q.G, S, q.ns, q.hm, q.hr); stats(q.xd, q.s, q.G, S, 0, q.hm0, q.hr0);
    q.a = malloc(n * 4); q.a0 = malloc(n * 4);
    act(q.xd, q.s, q.G, S, q.ns, q.hg, q.hb, q.hm, q.hr, q.a); act(q.xd, q.s, q.G, S, 0, q.hg, q.hb, q.hm0, q.hr0, q.a0);
    q.gam = dev_from(q.hg, C); q.bet = dev_from(q.hb, C); q.mean = dev_from(q.hm, (size_t)q.ns * q.G); q.rstd = dev_from(q.hr, (size_t)q.ns * q.G);
    q.mean0 = dev_from(q.hm0, q.G); q.rstd0 = dev_from(q.hr0, q.G);
    return q;
}
static void sx_free(sx_t *q) {
    nn_free(q->xm); nn_free(q->gam); nn_free(q->bet); nn_free(q->mean); nn_free(q->rstd); nn_free(q->mean0); nn_free(q->rstd0);
    free(q->xd); free(q->hg); free(q->hb); free(q->hm); free(q->hr); free(q->hm0); free(q->hr0); free(q->a); free(q->a0);
}
/* the slab check of one reader: out[0] / ref[0] slab, out[1] / ref[1] per sample */
static void slab_cmp(const char *what, float *const out[2], float *const ref[2], size_t n) {
    const double es = rel(out[0], ref[0], n), e0 = rel(out[1], ref[1], n), ex = rel(out[0], ref[1], n);
    const int ok = es < 1.5 * e0 + 0.02 && ex > 0.1;
    printf("  %-58s slab %.3g, per-sample baseline %.3g, slab vs per-sample ref %.3g%s\n", what, es, e0, ex, ok ? "" : "  FAIL");
    if (!ok) bad++;
}
/* forward 3^3, stride st, C -> cout, MX-fp4 output; prec 3 fp4 / 2 fp8; s2f4: lp_set_s2_f4 (-1 default) */
static void test_conv_fwd(int C, int cout, int S, int st, int prec, int s2f4, const char *kern) {
    nn_wmemo_clear();
    sx_t q = sx_make(C, S);
    const shape5 ys = nn_conv3d_out_shape(q.s, cout, 3, st);
    const size_t n = shape_numel(q.s), no = shape_numel(ys);
    float *w = dev_rand((size_t)cout * C * 27, 0.1f), *b = dev_rand(cout, 0.1f), *ref[2], *out[2];
    for (int k = 0; k < 2; k++) {   /* 0: slabs, 1: per sample */
        float *dt = dev_from(k ? q.a0 : q.a, n), *yr = dev_zero(no);
        nn_set_tf32(0); nn_conv3d_fwd(dt, q.s, w, b, cout, 3, st, yr);
        ref[k] = malloc(no * 4); nn_d2h(ref[k], yr, no * 4); nn_free(dt); nn_free(yr);
        if (!k) lp_gn_slab_reg(q.mean, S, 0, q.ns, q.s.h * q.s.w);
        nn_gn_t g = {q.gam, q.bet, k ? q.mean0 : q.mean, k ? q.rstd0 : q.rstd, q.G};
        void *ym = mx4_new(ys);
        nn_set_tf32(1); nn_set_prec(prec); lp_set_s2_f4(s2f4);
        const int rv = nn_conv3d_fwd_x(q.xm, &g, NULL, NULL, 0, 0, q.s, w, b, cout, 3, st, (float *)ym, 0, 1e-5f, NULL, NULL);
        lp_set_s2_f4(-1);
        if (rv) { printf("  %s: unsupported (rv %d)  FAIL\n", kern, rv); bad++; }
        out[k] = deq4_h(ym, ys); nn_free(ym);
        if (!k) lp_gn_slab_reg(q.mean, 0, 0, 0, 0);
    }
    char nm[96]; snprintf(nm, sizeof nm, "fwd s%d %d->%d, %d-plane slabs (%d): %s", st, C, cout, S, q.ns, kern);
    slab_cmp(nm, out, ref, no);
    for (int k = 0; k < 2; k++) { free(ref[k]); free(out[k]); }
    nn_free(w); nn_free(b); sx_free(&q);
}
/* weight gradient 3^3, stride st, MX-fp8 gy (sr: stochastic rounding of the gradient operands on) */
static void test_conv_wgrad(int C, int cout, int S, int st, int prec, int sr, const char *kern) {
    nn_wmemo_clear();
    sx_t q = sx_make(C, S);
    const shape5 ys = nn_conv3d_out_shape(q.s, cout, 3, st);
    const size_t n = shape_numel(q.s), no = shape_numel(ys), nw = (size_t)cout * C * 27;
    float *gy = dev_rand(no, 1.f); void *gym = mx8_from_l(gy, ys); float *hgyd = deq8_h(gym, ys), *gyd = dev_from(hgyd, no);
    float *ref[2], *out[2];
    for (int k = 0; k < 2; k++) {
        float *dt = dev_from(k ? q.a0 : q.a, n), *gw = dev_zero(nw);
        nn_set_tf32(0); nn_conv3d_bwd_weight(dt, q.s, gyd, ys, 3, st, gw, NULL);
        ref[k] = malloc(nw * 4); nn_d2h(ref[k], gw, nw * 4); nn_free(dt);
        if (!k) lp_gn_slab_reg(q.mean, S, 0, q.ns, q.s.h * q.s.w);
        nn_gn_t g = {q.gam, q.bet, k ? q.mean0 : q.mean, k ? q.rstd0 : q.rstd, q.G};
        nn_zero(gw, nw * 4);
        nn_set_tf32(1); nn_set_prec(prec); nn_set_sr(sr); nn_set_sr_step(31);
        const int rv = nn_conv3d_bwd_weight_x(q.xm, &g, NULL, NULL, 0, 0, q.s, gym, ys, 3, st, gw, NULL);
        nn_set_sr(0);
        if (rv) { printf("  %s: unsupported (rv %d)  FAIL\n", kern, rv); bad++; }
        out[k] = malloc(nw * 4); nn_sync(); nn_d2h(out[k], gw, nw * 4); nn_free(gw);
        if (!k) lp_gn_slab_reg(q.mean, 0, 0, 0, 0);
    }
    char nm[96]; snprintf(nm, sizeof nm, "wgrad s%d %d->%d, %d-plane slabs (%d): %s", st, C, cout, S, q.ns, kern);
    slab_cmp(nm, out, ref, nw);
    for (int k = 0; k < 2; k++) { free(ref[k]); free(out[k]); }
    nn_free(gy); nn_free(gym); nn_free(gyd); free(hgyd); sx_free(&q);
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

/* unet with several slabs, MX-fp8 activations and gradients on the fp8 kernels without stochastic rounding (fp4 + SR: the two paths
   drift apart by ~0.25 even with one slab, too much to see a slab error): the kernel path against D's prototype (normalised copies:
   the reference arithmetic up to its extra roundings) at the same slab size, their distance no larger than the two paths' with one
   slab (both GroupNorm then, the prototype's roundings the only difference), and far from GroupNorm (the input's statistics change by
   slab: per-sample statistics are another net). The production chain's three level-0 buffers (MX-fp4, the head-recomputing GN
   backward over the pseudo-sample shape) against four: test_grad_mx4 with UFSM_L0_NORM=4 UFSM_L0_SLAB=8 UFSM_L0_SLAB_KERN=1 */
static void test_unet_multi(int S) {
    nn_set_tf32(1); nn_set_act_bf16(1); nn_set_grad_bf16(1); nn_set_f16(1); nn_set_sr(0); nn_set_gn_stored(1);
    nn_set_prec(2);
    unet_set_act_mx4(0); unet_set_act_mx8(1); unet_set_grad_mx8(1); unet_set_grad_mx4(0);
    const shape5 xs = {1, 4, 32, 32, 32};
    const size_t nx = shape_numel(xs), P = (size_t)xs.h * xs.w, no = nx / 4 * 2;
    float *h = malloc(nx * 4);
    for (int c = 0; c < 4; c++) for (int z = 0; z < xs.d; z++) for (size_t v = 0; v < P; v++) {   /* statistics by slab: mean ~ slab, scale 1 + slab */
        const int sl = z / S; h[((size_t)c * xs.d + z) * P + v] = (1.f + sl) * frand() + 0.8f * sl;
    }
    float *x = dev_from(h, nx), *gy = dev_rand(no, 1.f); free(h);
    float *lf[4], *gr[4]; size_t np = 0; int l3[4];
    for (int arm = 0; arm < 4; arm++) {   /* kernel S, prototype S, kernel one slab (GroupNorm), prototype one slab */
        unet_set_l0_slab_kern(arm == 0 || arm == 2);
        unet_cfg cfg = {3, {16, 16, 32}, 4, 2, 8, 1}; cfg.l0_norm = 4; cfg.l0_slab = arm < 2 ? S : xs.d;
        unet *u = unet_create(&cfg); unet_init(u, 43); np = unet_nparams(u);
        unet_build(u, xs, 1); l3[arm] = unet_l0_3buf(u);
        lf[arm] = malloc(no * 4); gr[arm] = malloc(np * 4);
        nn_set_sr_step(11); const float *l = unet_forward(u, x, xs, 1); nn_sync();
        if (unet_logits_h16(u)) { uint16_t *t = malloc(no * 2); nn_d2h(t, l, no * 2); for (size_t i = 0; i < no; i++) lf[arm][i] = h2f(t[i]); free(t); }
        else nn_d2h(lf[arm], l, no * 4);
        unet_zero_grad(u); nn_set_sr_step(12); unet_backward(u, gy); unet_grad_d2h(u, gr[arm]);
        unet_free(u);
    }
    unet_set_l0_slab_kern(-1); unet_set_act_mx8(0);
    const double el = rel(lf[0], lf[1], no), el1 = rel(lf[2], lf[3], no), elg = rel(lf[0], lf[2], no);
    const double eg = rel(gr[0], gr[1], np), eg1 = rel(gr[2], gr[3], np), egg = rel(gr[0], gr[2], np);
    const int ok = el <= 2 * el1 + 0.01 && eg <= 2 * eg1 + 0.02 && elg > 3 * el1 + 0.02;
    printf("  unet %d-plane slabs (%d): kernel vs prototype logits %.3g grads %.3g; one slab: %.3g / %.3g; kernel slabs vs GroupNorm %.3g / %.3g; "
           "three buffers %d %d%s\n", S, xs.d / S, el, eg, el1, eg1, elg, egg, l3[0], l3[2], ok ? "" : "  FAIL");
    if (!ok) bad++;
    for (int a = 0; a < 4; a++) { free(gr[a]); free(lf[a]); }
    nn_free(x); nn_free(gy);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setenv("UFSM_L0_SLAB_KERN", "1", 0);
    setenv("UFSM_F4_WGRAD", "1", 1);   /* (read once: prec 3 weight gradients on the fp4 kernels, prec 2 on the fp8 ones) */
    if (nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0)) { printf("no cuda\n"); return 1; }
    nn_set_tf32(1); nn_set_f16(1); nn_set_gn_stored(1);
    test_unet_one_slab();
    nn_set_prec(3); nn_set_tf32(1);
    test_head(16); test_head(32);
    test_f4u(8); test_f4u(16);
    /* the 3^3 readers: stride 1 (fp4: the chunk kernels f4r / f4 -- conv_fwd_f4p_k is not taken with slab statistics; fp8: f8p /
       f8; <= 8 channels: the stem kernel f8s), stride 2 (fp4: s2_f4p / s2_f4; fp8: s2_f8 P16 / generic), weight gradients (fp4:
       conv_bwd_w_f4_k, also with the gy pre-pass (SR); fp8: conv_bwd_w_f8_k; stride 2: conv_bwd_w_s2_f8_k) */
    test_conv_fwd(16, 16, 8, 1, 3, -1, "fp4");
    test_conv_fwd(16, 16, 16, 1, 3, -1, "fp4");
    test_conv_fwd(32, 32, 8, 1, 3, -1, "fp4");
    test_conv_fwd(16, 16, 8, 1, 2, -1, "fp8");
    test_conv_fwd(32, 32, 8, 1, 2, -1, "fp8");
    test_conv_fwd(4, 16, 8, 1, 3, -1, "network-input kernel (f8s)");
    test_conv_fwd(16, 16, 8, 2, 3, -1, "fp4 down (s2_f4p)");
    test_conv_fwd(16, 16, 8, 2, 3, 0, "fp8 down (s2_f8 P16)");
    test_conv_fwd(32, 32, 8, 2, 3, -1, "fp4 down (s2_f4)");
    test_conv_fwd(32, 32, 8, 2, 3, 0, "fp8 down (s2_f8)");
    test_conv_wgrad(16, 16, 8, 1, 3, 0, "fp4");
    test_conv_wgrad(16, 16, 8, 1, 3, 1, "fp4, SR (gy pre-pass)");
    test_conv_wgrad(32, 32, 8, 1, 3, 0, "fp4");
    test_conv_wgrad(16, 16, 8, 1, 2, 0, "fp8");
    test_conv_wgrad(16, 16, 8, 1, 2, 1, "fp8, SR");
    test_conv_wgrad(32, 32, 8, 1, 2, 0, "fp8");
    test_conv_wgrad(4, 16, 8, 1, 2, 0, "fp8, network-input tile");
    test_conv_wgrad(16, 16, 8, 2, 3, 0, "down (s2)");
    test_conv_wgrad(16, 16, 8, 2, 3, 1, "down (s2), SR");
    test_conv_wgrad(32, 32, 8, 2, 3, 0, "down (s2)");
    test_unet_multi(8); test_unet_multi(16);
    printf("test_l0_slab: %s\n", bad ? "FAIL" : "PASS");
    return bad != 0;
}
