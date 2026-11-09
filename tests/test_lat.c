/* train --lat 4: the slab-wise latent logit upsampling (nn_lat_up_slab) against the whole-window pool4 chain (nn_lowres_output div 4
   mode 1, fp32), for every slab split; its adjoint (nn_lat_up_adj, accumulated over the slabs) against nn_lowres_grad on an fp32 gradient
   and the dot-product identity <up(l), g> = <l, adj(g)>. Odd and even grid sizes, slab heights 1, 3, 7, 32, coarse fp16 and fp32 logits. */
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rs = 0x2545f4914f6cdd1dull;
static float ru(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 11) * (1.0 / 9007199254740992.0)) * 2.f - 1.f; }
static float h2f(uint16_t v) { const int e = (v >> 10) & 31, m = v & 1023; const float f = e == 0 ? ldexpf((float)m, -24) : ldexpf((float)(m | 1024), e - 25); return (v >> 15) ? -f : f; }
static uint16_t f2h(float f) { _Float16 h = (_Float16)f; uint16_t u; memcpy(&u, &h, 2); return u; }

int main(void) {
    nn_init(0);
    int fails = 0, cases = 0;
    static const int dims[6][3] = {{5, 4, 4}, {3, 6, 6}, {8, 5, 5}, {2, 3, 3}, {1, 1, 2}, {20, 9, 7}};
    static const int slabs[6] = {4, 1, 8, 3, 32, 7};   /* (multiples of 4: the separable fast paths) */
    for (int t = 0; t < 6; t++) for (int lh = 0; lh < 2; lh++) {
        const int C = 3, d = dims[t][0], h = dims[t][1], w = dims[t][2], D = 4 * d, H = 4 * h, W = 4 * w;
        const size_t cv = (size_t)d * h * w, fv = (size_t)D * H * W;
        float *lc = malloc(C * cv * 4), *ref = malloc(C * fv * 4), *g = malloc(C * fv * 4), *gref = malloc(C * cv * 4), *gc = malloc(C * cv * 4);
        uint16_t *lc16 = malloc(C * cv * 2), *fine = malloc(C * fv * 2), *g16 = malloc(C * fv * 2);
        for (size_t k = 0; k < C * cv; k++) { lc[k] = 4.f * ru(); lc16[k] = f2h(lc[k]); if (lh) lc[k] = h2f(lc16[k]); }
        for (size_t k = 0; k < C * fv; k++) { g16[k] = f2h(ru()); g[k] = h2f(g16[k]); }
        void *d_lc = nn_malloc(C * cv * 4), *d_ref = nn_malloc(C * fv * 4), *d_tmp = nn_malloc(C * fv * 4), *d_g = nn_malloc(C * fv * 4), *d_gref = nn_malloc(C * cv * 4);
        void *d_sl = nn_malloc(C * fv * 2), *d_gs = nn_malloc(C * fv * 2); float *d_gc = nn_malloc(C * cv * 4);
        /* the whole-window chain (fp32 coarse logits: the reference) */
        nn_h2d(d_lc, lc, C * cv * 4);
        const shape5 qs = {1, C, d, h, w};
        nn_lowres_output(d_lc, 0, qs, 4, 1, d_tmp, d_ref); nn_d2h(ref, d_ref, C * fv * 4);
        nn_h2d(d_g, g, C * fv * 4);
        nn_lowres_grad(d_g, 0, qs, 4, 1, d_tmp, d_gref); nn_d2h(gref, d_gref, C * cv * 4);
        if (lh) nn_h2d(d_lc, lc16, C * cv * 2);
        for (int si = 0; si < 6; si++) {
            const int Z = slabs[si];
            nn_zero(d_gc, C * cv * 4);
            float eu = 0.f, ea = 0.f;
            for (int z0 = 0; z0 < D; z0 += Z) {
                const int nz = z0 + Z < D ? Z : D - z0; const size_t ns = (size_t)nz * H * W;
                nn_lat_up_slab(d_lc, lh, C, d, h, w, z0, nz, d_sl);
                nn_d2h(fine, d_sl, C * ns * 2);
                for (int c = 0; c < C; c++) for (size_t k = 0; k < ns; k++) {
                    const float a = h2f(fine[c * ns + k]), b = ref[c * fv + (size_t)z0 * H * W + k];
                    eu = fmaxf(eu, fabsf(a - b) / (1e-3f + fabsf(b)));
                }
                /* the slab's fp16 gradient, channel-major per slab as the head slabs lay it out */
                uint16_t *gs = malloc(C * ns * 2);
                for (int c = 0; c < C; c++) memcpy(gs + c * ns, g16 + c * fv + (size_t)z0 * H * W, ns * 2);
                nn_h2d(d_gs, gs, C * ns * 2); free(gs);
                nn_lat_up_adj(d_gs, C, d, h, w, z0, nz, d_gc, si & 1 ? nullptr : d_tmp);   /* (with / without the scratch of the separable path) */
            }
            nn_d2h(gc, d_gc, C * cv * 4);
            double dot1 = 0, dot2 = 0;
            for (size_t k = 0; k < C * cv; k++) { ea = fmaxf(ea, fabsf(gc[k] - gref[k]) / (1e-2f + fabsf(gref[k]))); dot2 += (double)lc[k] * gc[k]; }
            for (size_t k = 0; k < C * fv; k++) dot1 += (double)ref[k] * g[k];
            const double de = fabs(dot1 - dot2) / (1e-9 + fabs(dot1));
            cases++;
            if (!(eu < 2e-3f) || !(ea < 2e-4f) || !(de < 1e-5)) {   /* (ea: the per-voxel kernel reaches 1.03e-4 on the 20x9x7 grid, fp32 sums of ~1000 terms in another order) */
                fprintf(stderr, "FAIL: grid %dx%dx%d lh %d slab %d: up rel err %.3g, adj rel err %.3g, dot rel err %.3g\n", d, h, w, lh, Z, eu, ea, de);
                fails++;
            }
        }
        nn_free(d_lc); nn_free(d_ref); nn_free(d_tmp); nn_free(d_g); nn_free(d_gref); nn_free(d_sl); nn_free(d_gs); nn_free(d_gc);
        free(lc); free(ref); free(g); free(gref); free(gc); free(lc16); free(fine); free(g16);
    }
    {   /* the guided head (nn_lat_guide): dir 0 against a host reference; dir 1 is its adjoint in the coefficient planes: <fwd(b) - l, g> = <b, bwd(g)> */
        const int D = 12, H = 8, W = 8, base = 2, NC = base + 5; const size_t hw = (size_t)H * W, fv = (size_t)D * hw;
        uint16_t *x = malloc(fv * 2); for (size_t k = 0; k < fv; k++) x[k] = f2h(ru());
        void *d_x = nn_malloc(fv * 2); nn_h2d(d_x, x, fv * 2);
        static const int Zs[3] = {4, 12, 8};
        for (int si = 0; si < 3; si++) {
            const int Z = Zs[si]; double e0 = 0, dot1 = 0, dot2 = 0;
            for (int z0 = 0; z0 < D; z0 += Z) {
                const int nz = z0 + Z < D ? Z : D - z0; const size_t ns = (size_t)nz * hw;
                uint16_t *l = malloc(NC * ns * 2), *o = malloc(NC * ns * 2), *g = malloc(NC * ns * 2), *bg = malloc(NC * ns * 2);
                for (size_t k = 0; k < NC * ns; k++) { l[k] = f2h(ru()); g[k] = f2h(ru()); }
                void *d_l = nn_malloc(NC * ns * 2), *d_g = nn_malloc(NC * ns * 2);
                nn_h2d(d_l, l, NC * ns * 2); nn_lat_guide(d_x, D, H, W, z0, nz, d_l, base, 0, 0); nn_d2h(o, d_l, NC * ns * 2);
                nn_h2d(d_g, g, NC * ns * 2); nn_lat_guide(d_x, D, H, W, z0, nz, d_g, base, 0, 1); nn_d2h(bg, d_g, NC * ns * 2);
                for (size_t i = 0; i < ns; i++) {
                    const int z = z0 + (int)(i / hw), y = (int)(i / W % H), xx = (int)(i % W); const size_t v = (size_t)z * hw + (size_t)y * W + xx;
                    const float f[4] = {h2f(x[v]), 0.5f * (h2f(x[v + (z + 1 < D ? hw : 0)]) - h2f(x[v - (z > 0 ? hw : 0)])),
                                        0.5f * (h2f(x[v + (y + 1 < H ? W : 0)]) - h2f(x[v - (y > 0 ? W : 0)])), 0.5f * (h2f(x[v + (xx + 1 < W ? 1 : 0)]) - h2f(x[v - (xx > 0 ? 1 : 0)]))};
#define BP(k) h2f(l[(base + (k)) * ns + i])
                    const float t = BP(2) * f[1] + BP(3) * f[2] + BP(4) * f[3], a0 = BP(0) * f[0] + t, a1 = BP(1) * f[0] - t;
                    e0 = fmax(e0, fabs(h2f(o[i]) - (h2f(l[i]) + a0)) / (1e-2 + fabs(h2f(l[i]) + a0))); e0 = fmax(e0, fabs(h2f(o[ns + i]) - (h2f(l[ns + i]) + a1)) / (1e-2 + fabs(h2f(l[ns + i]) + a1)));
                    dot1 += (double)a0 * h2f(g[i]) + (double)a1 * h2f(g[ns + i]);
                    for (int k = 0; k < 5; k++) dot2 += (double)BP(k) * h2f(bg[(base + k) * ns + i]);
#undef BP
                }
                free(l); free(o); free(g); free(bg); nn_free(d_l); nn_free(d_g);
            }
            const double de = fabs(dot1 - dot2) / (1e-9 + fabs(dot1)); cases++;
            if (!(e0 < 3e-3) || !(de < 2e-3)) { fprintf(stderr, "FAIL: guided head, slab %d: forward rel err %.3g, adjoint dot rel err %.3g\n", Z, e0, de); fails++; }
        }
        free(x); nn_free(d_x);
    }
    const char *e = nn_check(); if (e) { fprintf(stderr, "FAIL: CUDA error %s\n", e); fails++; }
    if (fails) { printf("test_lat: %d of %d cases FAIL\n", fails, cases); return 1; }
    printf("test_lat: ok (%d cases: slab upsampling = the pool4 chain, adjoint = nn_lowres_grad, dot products)\n", cases);
    return 0;
}
