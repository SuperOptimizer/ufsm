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
    }
    printf(fails ? "FAIL: %d checks\n" : "PASS\n", fails);
    return fails != 0;
}
