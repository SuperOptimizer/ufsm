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
    const char *e = nn_check(); if (e) { fprintf(stderr, "FAIL: CUDA error %s\n", e); fails++; }
    if (fails) { printf("test_lat: %d of %d cases FAIL\n", fails, cases); return 1; }
    printf("test_lat: ok (%d cases: slab upsampling = the pool4 chain, adjoint = nn_lowres_grad, dot products)\n", cases);
    return 0;
}
