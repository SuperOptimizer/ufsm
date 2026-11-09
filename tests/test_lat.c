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
    /* predict: the direct cell-grid input (nn_lat_pred_input) against nn_pred_input_d + nn_lowres_input, and the direct rv bytes
       (nn_lat_place_rv) against nn_lowres_output + nn_rv_fuse + nn_pred_place_d, shards at aligned and odd offsets, uint8 CT and mask bits */
    for (int f16 = 1; f16 >= 0; f16--) {
        nn_set_f16(f16);
        const int D = 32, W = 48, d = D / 4, h = W / 4, halo = 8, hz = 4, S = 24; const size_t w3 = (size_t)D * W * W, cv = (size_t)d * h * h;
        uint8_t *ct = malloc(w3); float *dy = malloc(2 * D * sizeof(float));
        for (size_t k = 0; k < w3; k++) { const float r = ru(); ct[k] = r < -0.8f ? 0 : (uint8_t)(128 + 100 * ru()); }
        for (int z = 0; z < 2 * D; z++) dy[z] = 300.f * ru();
        uint8_t *d_ct = nn_malloc(w3); float *d_dy = nn_malloc(2 * D * sizeof(float)); void *d_x = nn_malloc(4 * w3 * 2), *d_t = nn_malloc(4 * w3 / 8 * 2 + 64), *d_a = nn_malloc(4 * cv * 2), *d_b = nn_malloc(4 * cv * 2);
        nn_h2d(d_ct, ct, w3); nn_h2d(d_dy, dy, 2 * D * sizeof(float));
        uint16_t *ha = malloc(4 * cv * 2), *hb = malloc(4 * cv * 2);
        for (int axis = -1; axis <= 1; axis++) {
            nn_pred_input_d(d_ct, D, W, 117.3f, 1.f / 41.7f, d_dy, d_dy + D, axis, d_x, 1);
            nn_lowres_input(d_x, (shape5){1, 4, D, W, W}, 4, 1, d_t, d_a, 1);
            nn_lat_pred_input(d_ct, D, W, 117.3f, 1.f / 41.7f, d_dy, d_dy + D, axis, d_b);
            nn_d2h(ha, d_a, 4 * cv * 2); nn_d2h(hb, d_b, 4 * cv * 2);
            size_t nd = 0, n1 = 0; for (size_t k = 0; k < 4 * cv; k++) { nd += ha[k] != hb[k]; n1 += abs((int)ha[k] - (int)hb[k]) > 1; }
            cases++;
            if (n1 || nd > 4 * cv / 100) { fprintf(stderr, "FAIL: nn_lat_pred_input (f16 %d, axis %d): %zu of %zu values differ, %zu by more than one code\n", f16, axis, nd, 4 * cv, n1); fails++; }
        }
        /* the rv bytes: random side logits on the cell grid */
        float *lc = malloc(2 * cv * 4); uint16_t *lc16 = malloc(2 * cv * 2);
        for (size_t k = 0; k < 2 * cv; k++) { lc16[k] = f2h(3.f * ru()); lc[k] = h2f(lc16[k]); }
        void *d_lc = nn_malloc(2 * cv * 4), *d_l16 = nn_malloc(2 * cv * 2); float *d_lt = nn_malloc(2 * w3 / 8 * 4), *d_lf = nn_malloc(2 * w3 * 4);
        uint32_t *d_bits = nn_malloc((w3 + 31) / 32 * 4); uint8_t *d_s0 = nn_malloc((size_t)S * S * S), *d_s1 = nn_malloc((size_t)S * S * S), *s0 = malloc((size_t)S * S * S), *s1 = malloc((size_t)S * S * S);
        nn_h2d(d_lc, lc, 2 * cv * 4); nn_h2d(d_l16, lc16, 2 * cv * 2);
        nn_pred_ctmask(d_ct, w3, d_bits);
        nn_lowres_output(d_lc, 0, (shape5){1, 2, d, h, h}, 4, 1, d_lt, d_lf); nn_rv_fuse(d_lf, w3, 0); nn_set_logits_h16(0);
        static const int offs[5][3] = {{-4, -8, -8}, {-4, -8, -32}, {-27, -31, -9}, {3, -8, 5}, {-28, -32, -40}};
        for (int oi = 0; oi < 5; oi++) for (int m = 0; m < 4; m++) {   /* m & 1: the mask bits; m & 2: fp16 cell logits */
            const int *o = offs[oi], ez = oi == 2 ? 17 : S, ey = oi == 2 ? 19 : S, ex = S;
            nn_zero(d_s0, (size_t)S * S * S); nn_zero(d_s1, (size_t)S * S * S);
            if (m & 1) nn_pred_place_m_d(d_lf, d_bits, D, W, hz, halo, o[0], o[1], o[2], ez, ey, ex, S, d_s0); else nn_pred_place_d(d_lf, d_ct, D, W, hz, halo, o[0], o[1], o[2], ez, ey, ex, S, d_s0);
            nn_lat_place_rv(m & 2 ? d_l16 : d_lc, m >> 1, d, h, h, m & 1 ? nullptr : d_ct, m & 1 ? d_bits : nullptr, nn_get_pred_iso(), hz, halo, o[0], o[1], o[2], ez, ey, ex, S, d_s1);
            nn_d2h(s0, d_s0, (size_t)S * S * S); nn_d2h(s1, d_s1, (size_t)S * S * S);
            size_t nd = 0, n1 = 0, nz = 0; for (size_t k = 0; k < (size_t)S * S * S; k++) { nd += s0[k] != s1[k]; n1 += abs((int)s0[k] - (int)s1[k]) > 1 || (s0[k] == 0) != (s1[k] == 0); nz += s0[k] != 0; }
            cases++;
            if (n1 || nd > nz / 50 + 2 || (oi < 4 && !nz)) { fprintf(stderr, "FAIL: nn_lat_place_rv (f16 %d, offset %d, mode %d): %zu of %zu written bytes differ, %zu by more than 1 or in the mask\n", f16, oi, m, nd, nz, n1); fails++; }
        }
        nn_free(d_ct); nn_free(d_dy); nn_free(d_x); nn_free(d_t); nn_free(d_a); nn_free(d_b); nn_free(d_lc); nn_free(d_l16); nn_free(d_lt); nn_free(d_lf); nn_free(d_bits); nn_free(d_s0); nn_free(d_s1);
        free(ct); free(dy); free(ha); free(hb); free(lc); free(lc16); free(s0); free(s1);
    }
    const char *e = nn_check(); if (e) { fprintf(stderr, "FAIL: CUDA error %s\n", e); fails++; }
    if (fails) { printf("test_lat: %d of %d cases FAIL\n", fails, cases); return 1; }
    printf("test_lat: ok (%d cases: slab upsampling = the pool4 chain, adjoint = nn_lowres_grad, dot products; predict: direct input and rv bytes = the full-resolution chain)\n", cases);
    return 0;
}
