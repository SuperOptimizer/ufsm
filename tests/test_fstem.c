/* nn_fstem_fwd / nn_fstem_bwd (train --fstem) against a double host reference on a 10 x 12 x 14 window (two slabs, z 0..5 and 6..9),
   C 16, nout 7 channels of 9; the reference's own gradient checked by central differences on a few parameters. Then nn_fstem_up_fwd (the head
   slab's upsample + sub-cell shuffle + stem in one kernel) byte for byte against nn_lat_up_slab_c + nn_lat_shuffle_c + nn_fstem_fwd on a
   12 x 8 x 264 window (two row tiles, the clamped edges), slabs out of a rolling cell-plane buffer, with and without the stem. Usage: test_fstem */
#include "nn.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
enum { D = 10, H = 12, W = 14, C = 16, NO = 7, NCH = 9, T = 64 };
static const int chs[NO] = {0, 1, 2, 5, 6, 7, 8};
static double rnd(uint64_t *s) { *s = *s * 6364136223846793005ull + 1442695040888963407ull; return (double)(*s >> 11) * 0x1p-53 - 0.5; }
static float h2f(uint16_t h) { _Float16 f; memcpy(&f, &h, 2); return (float)f; }
static uint16_t f2h(float v) { _Float16 f = (_Float16)v; uint16_t h; memcpy(&h, &f, 2); return h; }
static double xv(const uint16_t *x, int z, int y, int xx) { return z >= 0 && z < D && y >= 0 && y < H && xx >= 0 && xx < W ? h2f(x[((size_t)z * H + y) * W + xx]) : 0.0; }
/* out[o][voxel] and L = sum G . out (double) */
static double ref(const uint16_t *x, const double *p, const uint16_t *G, double *out) {
    const double *W1 = p, *b1 = p + C * T, *W2 = b1 + C; double L = 0;
    for (int bz = 0; bz < D / 2; bz++) for (int by = 0; by < H / 2; by++) for (int bx = 0; bx < W / 2; bx++) {
        double h[C];
        for (int c = 0; c < C; c++) { double a = b1[c]; for (int k = 0; k < T; k++) a += W1[c * T + k] * xv(x, 2 * bz - 1 + k / 16, 2 * by - 1 + k / 4 % 4, 2 * bx - 1 + k % 4); h[c] = a / (1 + exp(-a)); }
        for (int o = 0; o < NO; o++) {
            double v = 0; for (int c = 0; c < C; c++) v += W2[o * C + c] * h[c];
            for (int q = 0; q < 8; q++) { const size_t vi = ((size_t)(2 * bz + q / 4) * H + 2 * by + q / 2 % 2) * W + 2 * bx + q % 2; if (out) out[(size_t)o * D * H * W + vi] = v; L += v * h2f(G[(size_t)chs[o] * D * H * W + vi]); }
        }
    }
    return L;
}
static int up_check(void) {
    enum { UD = 12, UH = 8, UW = 264, UC = 9, BASE = 9, LC = BASE + 128 };
    const int d = UD / 4, h = UH / 4, w = UW / 4; const size_t V = (size_t)UD * UH * UW, cpl = (size_t)h * w, NP = C * T + C + NO * C;
    uint64_t s = 11; int fails = 0;
    uint16_t *x = malloc(V * 2), *lc = malloc((size_t)LC * d * cpl * 2); float *pf = malloc(NP * 4);
    for (size_t i = 0; i < V; i++) x[i] = f2h((float)(2 * rnd(&s)));
    for (size_t i = 0; i < (size_t)LC * d * cpl; i++) lc[i] = f2h((float)(4 * rnd(&s)));
    for (size_t i = 0; i < NP; i++) pf[i] = (float)(0.4 * rnd(&s));
    void *xd = nn_malloc(V * 2); float *pd = nn_malloc(NP * 4); nn_h2d(xd, x, V * 2); nn_h2d(pd, pf, NP * 4);
    const int sl[3][2] = {{0, 4}, {4, 4}, {8, 4}};
    for (int stem = 0; stem < 2; stem++) for (int k = 0; k < 3; k++) {
        const int z0 = sl[k][0], nz = sl[k][1], qa = z0 / 4 - 1 > 0 ? z0 / 4 - 1 : 0, qb = (z0 + nz) / 4 + 1 < d ? (z0 + nz) / 4 + 1 : d, qd = qb - qa;
        const size_t ns = (size_t)nz * UH * UW;
        uint16_t *hc = malloc((size_t)LC * qd * cpl * 2), *ya = malloc(UC * ns * 2), *yb = malloc(UC * ns * 2);   /* the rolling buffer: planes [qa, qb) */
        for (int c = 0; c < LC; c++) memcpy(hc + (size_t)c * qd * cpl, lc + ((size_t)c * d + qa) * cpl, qd * cpl * 2);
        void *cd = nn_malloc((size_t)LC * qd * cpl * 2), *da = nn_malloc(UC * ns * 2), *db = nn_malloc(UC * ns * 2);
        nn_h2d(cd, hc, (size_t)LC * qd * cpl * 2);
        for (size_t i = 0; i < UC * ns; i++) ya[i] = 0x7e00;   /* (NaN: what the fused kernel must not touch, or must overwrite) */
        nn_h2d(db, ya, UC * ns * 2);
        nn_lat_up_slab_c(cd, 1, UC, d, h, w, z0, nz, da, qa, qd);
        nn_lat_shuffle_c(cd, 1, d, h, w, BASE, 2, z0, nz, da, qa, qd);
        if (stem && nn_fstem_fwd(xd, UD, UH, UW, z0, nz, pd, C, NO, chs, da, ns)) { fprintf(stderr, "FAIL: fstem shape\n"); return 1; }
        if (nn_fstem_up_fwd(xd, UD, UH, UW, z0, nz, pd, stem ? C : 0, NO, chs, db, ns, cd, qa, qd, 0, BASE)) { fprintf(stderr, "FAIL: fused shape refused\n"); return 1; }
        nn_sync(); nn_d2h(ya, da, UC * ns * 2); nn_d2h(yb, db, UC * ns * 2);
        size_t nd = 0, nu = 0;
        for (int c = 0; c < UC; c++) {
            int o = -1; for (int q = 0; q < NO; q++) if (chs[q] == c) o = q;
            for (size_t i = 0; i < ns; i++) { if (o >= 0) nd += ya[c * ns + i] != yb[c * ns + i]; else nu += yb[c * ns + i] != 0x7e00; }
        }
        printf("fstem up fwd (stem %d, slab z %d + %d, cell planes %d..%d): %zu of %zu logits differ from the 3-kernel chain, %zu other values touched\n", stem, z0, nz, qa, qb, nd, (size_t)NO * ns, nu);
        if (nd || nu) { fprintf(stderr, "FAIL: fused up\n"); fails++; }
        nn_free(cd); nn_free(da); nn_free(db); free(hc); free(ya); free(yb);
    }
    if (nn_fstem_up_fwd(xd, UD, UH, UW, 4, 4, pd, C, NO, chs, xd, 4 * UH * UW, xd, 1, 1, 0, BASE) != 1) { fprintf(stderr, "FAIL: a short cell buffer accepted\n"); fails++; }
    nn_free(xd); nn_free(pd); free(x); free(lc); free(pf);
    return fails;
}
int main(void) {
    nn_init(0); int fails = 0; uint64_t s = 5;
    const size_t V = (size_t)D * H * W, NP = C * T + C + NO * C;
    uint16_t *x = malloc(V * 2), *G = malloc(NCH * V * 2), *lg0 = malloc(NCH * V * 2), *lg1 = malloc(NCH * V * 2);
    double *p = malloc(NP * 8), *out = malloc(NO * V * 8); float *pf = malloc(NP * 4), *gp = malloc(NP * 4);
    for (size_t i = 0; i < V; i++) x[i] = f2h((float)(2 * rnd(&s)));
    for (size_t i = 0; i < NCH * V; i++) { G[i] = f2h((float)rnd(&s)); lg0[i] = f2h((float)rnd(&s)); }
    for (size_t i = 0; i < NP; i++) { pf[i] = (float)(0.4 * rnd(&s)); p[i] = pf[i]; }
    void *xd = nn_malloc(V * 2), *ld = nn_malloc(NCH * V * 2), *gd = nn_malloc(NCH * V * 2); float *pd = nn_malloc(NP * 4), *gpd = nn_malloc(NP * 4);
    nn_h2d(xd, x, V * 2); nn_h2d(ld, lg0, NCH * V * 2); nn_h2d(gd, G, NCH * V * 2); nn_h2d(pd, pf, NP * 4); nn_zero(gpd, NP * 4);
    const int sl[2][2] = {{0, 6}, {6, 4}};
    for (int k = 0; k < 2; k++) {   /* slab planes: the slab buffers of train's head slabs (planes of ns = nz H W voxels) */
        const int z0 = sl[k][0], nz = sl[k][1]; const size_t ns = (size_t)nz * H * W;
        uint16_t *hb = malloc(NCH * ns * 2), *hg = malloc(NCH * ns * 2);
        for (int c = 0; c < NCH; c++) { memcpy(hb + c * ns, lg0 + c * V + (size_t)z0 * H * W, ns * 2); memcpy(hg + c * ns, G + c * V + (size_t)z0 * H * W, ns * 2); }
        void *sb = nn_malloc(NCH * ns * 2), *sg = nn_malloc(NCH * ns * 2); nn_h2d(sb, hb, NCH * ns * 2); nn_h2d(sg, hg, NCH * ns * 2);
        if (nn_fstem_fwd(xd, D, H, W, z0, nz, pd, C, NO, chs, sb, ns) || nn_fstem_bwd(xd, D, H, W, z0, nz, pd, C, NO, chs, sg, ns, gpd)) { fprintf(stderr, "FAIL: shape refused\n"); return 1; }
        nn_sync(); nn_d2h(hb, sb, NCH * ns * 2);
        for (int c = 0; c < NCH; c++) memcpy(lg1 + c * V + (size_t)z0 * H * W, hb + c * ns, ns * 2);
        nn_free(sb); nn_free(sg); free(hb); free(hg);
    }
    ref(x, p, G, out);
    double we = 0; int touched_other = 0;
    for (int c = 0; c < NCH; c++) {
        int o = -1; for (int q = 0; q < NO; q++) if (chs[q] == c) o = q;
        for (size_t i = 0; i < V; i++) {
            const double want = h2f(lg0[c * V + i]) + (o >= 0 ? out[o * V + i] : 0), got = h2f(lg1[c * V + i]);
            if (o < 0) touched_other += lg1[c * V + i] != lg0[c * V + i]; else { const double e = fabs(got - want) / (1 + fabs(want)); we = e > we ? e : we; }
        }
    }
    printf("fstem fwd: worst |error| / (1 + |want|) %.2e, other channels changed %d\n", we, touched_other);
    if (!(we < 2e-3) || touched_other) { fprintf(stderr, "FAIL: fwd\n"); fails++; }
    nn_d2h(gp, gpd, NP * 4);
    double gmax = 0, ge = 0; double *ga = malloc(NP * 8);
    for (size_t i = 0; i < NP; i++) {   /* the analytic gradient of the reference, by central differences on every parameter (double) */
        const double h = 1e-5, p0 = p[i]; p[i] = p0 + h; const double lp = ref(x, p, G, nullptr); p[i] = p0 - h; const double lm = ref(x, p, G, nullptr); p[i] = p0;
        ga[i] = (lp - lm) / (2 * h); gmax = fabs(ga[i]) > gmax ? fabs(ga[i]) : gmax;
    }
    for (size_t i = 0; i < NP; i++) { const double e = fabs(gp[i] - ga[i]); ge = e > ge ? e : ge; }
    printf("fstem bwd: worst |error| %.2e over %zu parameters (max |gradient| %.3g): %.2e relative\n", ge, NP, gmax, ge / gmax);
    if (!(ge / gmax < 1e-3))   /* (the recompute's W1 in fp16, as the forward's tensor-core product: 1.8e-4 measured) */ { fprintf(stderr, "FAIL: bwd\n"); fails++; }
    if (nn_fstem_fwd(xd, D, H, W, 1, 4, pd, C, NO, chs, ld, 4 * H * W) != 1) { fprintf(stderr, "FAIL: odd z0 accepted\n"); fails++; }
    if (fails) return 1;
    if (up_check()) return 1;
    printf("test_fstem: ok\n"); return 0;
}
