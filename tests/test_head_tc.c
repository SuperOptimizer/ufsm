/* the wide MX-fp4 head forward on the tensor cores (lp_mx_conv1.cu conv1_mx_tc_k, perf2 2026-11-13) against the FFMA passes
   (conv1_mx_t_k, fp32 accumulation: the reference) at the latent cell head's shapes: 256 MX-fp4 channels, 160 / 35 / 17 outputs, GN+SiLU
   per sample and per z-slab, a voxel range (head slabs) with tails that are no multiple of the 128-cell tile. Pass: the fp16 logits' rel L2
   difference < 2e-3 and max |difference| < 1e-2 of max |logit| (the fp16 rounding of the activations and weights; the logits themselves
   are fp16), and the tensor-core path's bytes repeat run to run. Also times both at a P 1024 split-half slab (4 cell planes of 256^2). */
#include "nn.h"
#include "nn_lp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
void lp_gn_slab_reg(const float *mean, int zs, int zo, int ns, int zhw);
static uint64_t rs = 88172645463325252ull;
static float frand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 40) / 16777216.0 - 0.5); }
static float *dev_vals(size_t n, float sc, float off) { float *h = malloc(n * 4); for (size_t i = 0; i < n; i++) h[i] = off + frand() * sc; float *d = nn_malloc(n * 4); nn_h2d(d, h, n * 4); free(h); return d; }
static void *mx4_of(shape5 s, float sc) { float *f = dev_vals(shape_numel(s), sc, 0.f); size_t b = nn_mx4_bytes(s); void *p = nn_malloc(b); nn_zero(p, b); nn_set_storage(p, b, 4); lp_f32_to_mx4(f, s.n, s.c, shape_spatial(s), p); nn_free(f); return p; }
static int fails;
static void cmp16(const char *nm, const void *a, const void *r, size_t n) {
    _Float16 *ha = malloc(n * 2), *hr = malloc(n * 2); nn_d2h(ha, a, n * 2); nn_d2h(hr, r, n * 2);
    double d2 = 0, r2 = 0, md = 0, mr = 0;
    for (size_t i = 0; i < n; i++) { const double x = (double)ha[i], y = (double)hr[i]; d2 += (x - y) * (x - y); r2 += y * y; if (fabs(x - y) > md) md = fabs(x - y); if (fabs(y) > mr) mr = fabs(y); }
    const double rel = sqrt(d2 / (r2 > 0 ? r2 : 1)), mx = md / (mr > 0 ? mr : 1); const int ok = rel < 2e-3 && mx < 1e-2 && isfinite(rel);
    printf("  %-58s rel L2 %.2e  max |d| / max |y| %.2e  %s\n", nm, rel, mx, ok ? "ok" : "FAIL"); if (!ok) fails++;
    free(ha); free(hr);
}
int main(void) {
    nn_init(0); nn_set_prec(3);
    const int C = 256, G = 32;
    for (int sl = 0; sl < 2; sl++) for (int oc = 0; oc < 3; oc++) {
        const int cout = oc == 0 ? 160 : oc == 1 ? 35 : 17, D = 12, H = 40, W = 40;   /* (40^2 = 1600 cells a plane: 12.5 tiles) */
        shape5 xs = {1, C, D, H, W}; const size_t hw = (size_t)H * W, S = D * hw;
        void *x = mx4_of(xs, 3.f);
        float *w = dev_vals((size_t)cout * C, 0.1f, 0.f), *b = dev_vals(cout, 0.2f, 0.f), *gam = dev_vals(C, 0.5f, 1.f), *bet = dev_vals(C, 0.4f, 0.f);
        const int ns = sl ? 3 : 0, zs = 4;   /* slab GroupNorm: 3 slabs of 4 planes */
        float *mean = dev_vals((size_t)(ns ? ns : 1) * G, 0.4f, 0.f), *rstd = dev_vals((size_t)(ns ? ns : 1) * G, 0.4f, 0.9f);
        if (ns) lp_gn_slab_reg(mean, zs, 0, ns, (int)hw);
        const gnp_t gp = gnp_mk(gam, bet, mean, rstd, G);
        for (int rg = 0; rg < 2; rg++) {   /* the whole tensor; a head-slab range: planes 3 .. 8 (crosses a slab border) */
            const size_t vb = rg ? 3 * hw : 0, nv = rg ? 5 * hw + 77 : 0, Sy = nv ? nv : S;
            void *y0 = nn_malloc((size_t)cout * Sy * 2), *y1 = nn_malloc((size_t)cout * Sy * 2), *y2 = nn_malloc((size_t)cout * Sy * 2);
            lp_set_head_tc(0); lp_conv1_fwd_mx_h16_r(x, 4, xs, w, b, cout, y0, gp, vb, nv);
            lp_set_head_tc(1); lp_conv1_fwd_mx_h16_r(x, 4, xs, w, b, cout, y1, gp, vb, nv); lp_conv1_fwd_mx_h16_r(x, 4, xs, w, b, cout, y2, gp, vb, nv);
            nn_sync();
            char nm[128]; snprintf(nm, sizeof nm, "head tc %d outputs, %s GN, %s", cout, ns ? "slab" : "sample", rg ? "range (planes 3..8 + 77)" : "whole");
            cmp16(nm, y1, y0, (size_t)cout * Sy);
            { const size_t nb = (size_t)cout * Sy * 2; uint8_t *a = malloc(nb), *c = malloc(nb); nn_d2h(a, y1, nb); nn_d2h(c, y2, nb); const int same = !memcmp(a, c, nb);
              printf("  %-58s %s\n", "  tensor-core bytes repeat", same ? "ok" : "FAIL"); fails += !same; free(a); free(c); }
            nn_free(y0); nn_free(y1); nn_free(y2);
        }
        if (ns) lp_gn_slab_reg(mean, 0, 0, 0, 0);
        nn_free(x); nn_free(w); nn_free(b); nn_free(gam); nn_free(bet); nn_free(mean); nn_free(rstd);
    }
    for (int ci = 0; ci < 3; ci++) for (int gm = 0; gm < 3; gm++) {   /* train --ink-surf's feature planes (lp_feat_mx_h16_r, 2026-11-25): byte-equal to the
                                                                          identity head on the tensor cores (32 / 64 / 160 channels; no GN (-0 inputs), sample GN, slab GN) */
        const int Cf = ci == 0 ? 32 : ci == 1 ? 64 : 160, D = 12, H = 40, W = 40, Gf = Cf / 4 < 32 ? Cf / 4 : 32;
        shape5 xs = {1, Cf, D, H, W}; const size_t hw = (size_t)H * W, S = D * hw;
        void *x = mx4_of(xs, gm == 0 ? 0.02f : 3.f);
        float *idw = calloc((size_t)Cf * (Cf + 1), 4); for (int c = 0; c < Cf; c++) idw[(size_t)c * Cf + c] = 1.f;
        float *w = nn_malloc((size_t)Cf * (Cf + 1) * 4); nn_h2d(w, idw, (size_t)Cf * (Cf + 1) * 4); free(idw);
        float *gam = dev_vals(Cf, 0.5f, 1.f), *bet = dev_vals(Cf, 0.4f, 0.f);
        const int ns = gm == 2 ? 3 : 0, zs = 4;
        float *mean = dev_vals((size_t)(ns ? ns : 1) * Gf, 0.4f, 0.f), *rstd = dev_vals((size_t)(ns ? ns : 1) * Gf, 0.4f, 0.9f);
        if (ns) lp_gn_slab_reg(mean, zs, 0, ns, (int)hw);
        gnp_t gp = gnp_mk(gam, bet, mean, rstd, Gf); if (gm == 0) memset(&gp, 0, sizeof gp);
        for (int rg = 0; rg < 2; rg++) {
            const size_t vb = rg ? 3 * hw : 0, nv = rg ? 5 * hw + 77 : 0, Sy = nv ? nv : S, nb = (size_t)Cf * Sy * 2;
            void *y0 = nn_malloc(nb), *y1 = nn_malloc(nb); nn_zero(y0, nb); nn_zero(y1, nb);
            lp_set_head_tc(1); lp_conv1_fwd_mx_h16_r(x, 4, xs, w, w + (size_t)Cf * Cf, Cf, y0, gp, vb, nv);
            const bool cov = lp_feat_mx_h16_r(x, 4, xs, y1, gp, vb, nv); nn_sync();
            uint8_t *a = malloc(nb), *c = malloc(nb); nn_d2h(a, y0, nb); nn_d2h(c, y1, nb);
            size_t nd = 0, nz = 0; for (size_t i = 0; i < nb; i++) nd += a[i] != c[i];
            for (size_t i = 0; i + 1 < nb; i += 2) nz += a[i] == 0 && a[i + 1] == 0;
            const int ok = cov && nd == 0;
            printf("  feature planes %3d ch, %-6s GN, %-5s: %zu of %zu bytes differ from the identity head (%zu zero halves)  %s\n", Cf, gm == 0 ? "no" : gm == 1 ? "sample" : "slab",
                   rg ? "range" : "whole", nd, nb, nz, ok ? "ok" : "FAIL"); fails += !ok;
            free(a); free(c); nn_free(y0); nn_free(y1);
        }
        if (ns) lp_gn_slab_reg(mean, 0, 0, 0, 0);
        lp_set_head_tc(-1);
        nn_free(x); nn_free(w); nn_free(gam); nn_free(bet); nn_free(mean); nn_free(rstd);
    }
    for (int oc = 0; oc < 8; oc++) {   /* the backward-data: K fp16 logit gradients -> C MX-fp4 channels (SR), a slab range or whole, against the FFMA kernel
                                          (oc 3..5: test_lat_head's 32 / 96 channel trunks, 144 outputs, two samples; oc 6: weights ~3e-7, below fp16's
                                          normal range: the row scales of c1b_wh_k, 2026-11-14; oc 7: the 256-channel trunk's 144-output head, one
                                          sample, a slab range: audit 2026-11-14 #18e) */
        const int K = oc == 0 ? 160 : oc == 1 ? 35 : oc == 2 ? 17 : 144, C = oc < 3 || oc == 7 ? 256 : oc == 3 ? 32 : 96, NS = oc < 3 || oc == 7 ? 1 : 2, D = 12, H = 40, W = 40;
        const size_t hw = (size_t)H * W, S = D * hw, vb = oc == 5 ? 0 : 3 * hw, nv = oc == 5 ? 0 : 5 * hw + 77, nvv = nv ? nv : S;
        shape5 xs = {NS, C, D, H, W};
        void *gy = nn_malloc((size_t)NS * K * nvv * 2); { float *f = dev_vals((size_t)NS * K * nvv, 2e-3f, 0.f); nn_f32_to_h16(f, (size_t)NS * K * nvv, gy, 1.f); nn_free(f); }
        float *wt = dev_vals((size_t)K * C, oc == 6 ? 3e-7f : 0.1f, 0.f);
        void *g0 = nn_malloc(nn_mx4_bytes(xs)), *g1 = nn_malloc(nn_mx4_bytes(xs)), *g2 = nn_malloc(nn_mx4_bytes(xs));
        for (int q = 0; q < 3; q++) { void *gq = q == 0 ? g0 : q == 1 ? g1 : g2; nn_zero(gq, nn_mx4_bytes(xs)); nn_set_storage(gq, nn_mx4_bytes(xs), 4); }
        lp_set_head_btc(0); lp_conv1_to_mx_r(gy, 2, NS, K, S, wt, C, g0, 4, 1234u, vb, nv);
        lp_set_head_btc(1); lp_conv1_to_mx_r(gy, 2, NS, K, S, wt, C, g1, 4, 1234u, vb, nv); lp_conv1_to_mx_r(gy, 2, NS, K, S, wt, C, g2, 4, 1234u, vb, nv);
        float *d0 = nn_malloc(shape_numel(xs) * 4), *d1 = nn_malloc(shape_numel(xs) * 4);
        lp_mx4_to_f32(g0, NS, C, S, d0); lp_mx4_to_f32(g1, NS, C, S, d1); nn_sync();
        float *h0 = malloc(shape_numel(xs) * 4), *h1 = malloc(shape_numel(xs) * 4); nn_d2h(h0, d0, shape_numel(xs) * 4); nn_d2h(h1, d1, shape_numel(xs) * 4);
        double dd = 0, rr = 0; size_t neq = 0, nn_ = 0;
        for (size_t i = 0; i < shape_numel(xs); i++) { const double a = h1[i], r = h0[i]; dd += (a - r) * (a - r); rr += r * r; neq += a != r; nn_++; }
        const double rel = sqrt(dd / (rr > 0 ? rr : 1)); const int ok = rel < 0.05 && rr > 0;   /* (MX-fp4 with SR: an element whose sum moved across a rounding step differs by a step) */
        printf("  head bwd-data tc %3d -> %d x %d%s, MX-fp4 SR: rel L2 vs the FFMA kernel %.3e, %.4f of the elements differ  %s\n", K, C, NS, nv ? "" : " whole", rel, (double)neq / nn_, ok ? "ok" : "FAIL"); fails += !ok;
        { const size_t nb = nn_mx4_bytes(xs); uint8_t *a = malloc(nb), *c = malloc(nb); nn_d2h(a, g1, nb); nn_d2h(c, g2, nb); const int same = !memcmp(a, c, nb);
          printf("  %-58s %s\n", "  tensor-core bytes repeat", same ? "ok" : "FAIL"); fails += !same; free(a); free(c); }
        lp_set_head_btc(-1);
        free(h0); free(h1); nn_free(d0); nn_free(d1); nn_free(g0); nn_free(g1); nn_free(g2); nn_free(gy); nn_free(wt);
    }
    if (getenv("UFSM_TEST_QUICK") == nullptr) {   /* timing: one P 1024 split-half head slab (4 cell planes of 256^2), 160 outputs */
        shape5 xs = {1, C, 8, 256, 256}; const size_t hw = 256 * 256, nv = 4 * hw;
        void *x = mx4_of(xs, 3.f); float *w = dev_vals((size_t)160 * C, 0.1f, 0.f), *b = dev_vals(160, 0.2f, 0.f), *gam = dev_vals(C, 0.5f, 1.f), *bet = dev_vals(C, 0.4f, 0.f);
        float *mean = dev_vals(G, 0.4f, 0.f), *rstd = dev_vals(G, 0.4f, 0.9f); const gnp_t gp = gnp_mk(gam, bet, mean, rstd, G);
        void *y = nn_malloc((size_t)160 * nv * 2);
        for (int tc = 0; tc < 2; tc++) {
            lp_set_head_tc(tc); double best = 1e9;
            for (int r = 0; r < 5; r++) { nn_sync(); const double t = now(); lp_conv1_fwd_mx_h16_r(x, 4, xs, w, b, 160, y, gp, 2 * hw, nv); nn_sync(); if (now() - t < best) best = now() - t; }
            printf("  timing: 4 x 256^2 cells, 256 -> 160, %-14s %7.3f ms (%.1f TFLOP/s)\n", tc ? "tensor cores" : "FFMA passes", best * 1e3, 2.0 * nv * C * 160 / best / 1e12);
        }
        lp_set_head_tc(-1);
        {   /* the feature planes of a P 384 head slab (32 planes of 384^2, 32 channels, sample GN): the identity head vs lp_feat_mx_h16_r */
            shape5 fs = {1, 32, 64, 384, 384}; const size_t fhw = 384 * 384, fnv = 32 * fhw;
            void *fx = mx4_of(fs, 3.f); float *idw = calloc(32 * 33, 4); for (int c = 0; c < 32; c++) idw[c * 32 + c] = 1.f;
            float *fw = nn_malloc(32 * 33 * 4); nn_h2d(fw, idw, 32 * 33 * 4); free(idw);
            float *fg = dev_vals(32, 0.5f, 1.f), *fb = dev_vals(32, 0.4f, 0.f), *fm = dev_vals(8, 0.4f, 0.f), *fr = dev_vals(8, 0.4f, 0.9f);
            const gnp_t fgp = gnp_mk(fg, fb, fm, fr, 8);
            void *fy = nn_malloc((size_t)32 * fnv * 2);
            for (int k = 0; k < 2; k++) {
                double best = 1e9;
                for (int r = 0; r < 5; r++) { nn_sync(); const double t = now();
                    if (k) lp_feat_mx_h16_r(fx, 4, fs, fy, fgp, 16 * fhw, fnv); else lp_conv1_fwd_mx_h16_r(fx, 4, fs, fw, fw + 32 * 32, 32, fy, fgp, 16 * fhw, fnv);
                    nn_sync(); if (now() - t < best) best = now() - t; }
                printf("  timing: feature planes 32 x 384^2 x 32 ch, %-16s %7.3f ms (%.0f GB/s)\n", k ? "lp_feat_mx_h16_r" : "identity head", best * 1e3, (double)fnv * (17 + 64) / best / 1e9);
            }
            nn_free(fx); nn_free(fw); nn_free(fg); nn_free(fb); nn_free(fm); nn_free(fr); nn_free(fy);
        }
        {   /* the head's backward-data at the same slab: 160 fp16 logit gradients -> 256 MX-fp4 channels (SR) */
            void *gy = nn_malloc((size_t)160 * nv * 2); { float *f = dev_vals((size_t)160 * nv, 1e-3f, 0.f); nn_f32_to_h16(f, (size_t)160 * nv, gy, 1.f); nn_free(f); }
            float *wt = dev_vals((size_t)160 * C, 0.1f, 0.f);
            void *gx = nn_malloc(nn_mx4_bytes(xs)); nn_set_storage(gx, nn_mx4_bytes(xs), 4);
            for (int tc = 0; tc < 2; tc++) {
                lp_set_head_btc(tc); double best = 1e9;
                for (int r = 0; r < 5; r++) { nn_sync(); const double t = now(); lp_conv1_to_mx_r(gy, 2, 1, 160, shape_spatial(xs), wt, C, gx, 4, 77u, 2 * hw, nv); nn_sync(); if (now() - t < best) best = now() - t; }
                printf("  timing: head backward-data 160 -> 256, %-14s %7.3f ms (%.1f TFLOP/s)\n", tc ? "tensor cores" : "FFMA kernel", best * 1e3, 2.0 * nv * C * 160 / best / 1e12);
            }
            lp_set_head_btc(-1);
            nn_free(gy); nn_free(wt); nn_free(gx);
        }
    }
    printf(fails ? "FAIL: %d checks\n" : "PASS\n", fails);
    return fails != 0;
}
