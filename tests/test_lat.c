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
    {   /* the general guided head (nn_lat_guide4): forward against a host reference, dir 1 = its adjoint in the coefficient planes */
        const int D = 12, H = 8, W = 8, base = 3, ng = 2, NC = base + 4 * ng; const size_t hw = (size_t)H * W, fv = (size_t)D * hw;
        uint16_t *x = malloc(fv * 2); for (size_t k = 0; k < fv; k++) x[k] = f2h(ru());
        void *d_x = nn_malloc(fv * 2); nn_h2d(d_x, x, fv * 2);
        static const int Zs[2] = {4, 12};
        for (int si = 0; si < 2; si++) {
            const int Z = Zs[si]; double e0 = 0, dot1 = 0, dot2 = 0;
            for (int z0 = 0; z0 < D; z0 += Z) {
                const int nz = z0 + Z < D ? Z : D - z0; const size_t ns = (size_t)nz * hw;
                uint16_t *l = malloc(NC * ns * 2), *o = malloc(NC * ns * 2), *g = malloc(NC * ns * 2), *bg = malloc(NC * ns * 2);
                for (size_t k = 0; k < NC * ns; k++) { l[k] = f2h(ru()); g[k] = f2h(ru()); }
                void *d_l = nn_malloc(NC * ns * 2), *d_g = nn_malloc(NC * ns * 2);
                nn_h2d(d_l, l, NC * ns * 2); nn_lat_guide4(d_x, D, H, W, z0, nz, d_l, base, 1, ng, 0); nn_d2h(o, d_l, NC * ns * 2);
                nn_h2d(d_g, g, NC * ns * 2); nn_lat_guide4(d_x, D, H, W, z0, nz, d_g, base, 1, ng, 1); nn_d2h(bg, d_g, NC * ns * 2);
                for (size_t i = 0; i < ns; i++) {
                    const int z = z0 + (int)(i / hw), y = (int)(i / W % H), xx = (int)(i % W); const size_t v = (size_t)z * hw + (size_t)y * W + xx;
                    const float f[4] = {h2f(x[v]), 0.5f * (h2f(x[v + (z + 1 < D ? hw : 0)]) - h2f(x[v - (z > 0 ? hw : 0)])),
                                        0.5f * (h2f(x[v + (y + 1 < H ? W : 0)]) - h2f(x[v - (y > 0 ? W : 0)])), 0.5f * (h2f(x[v + (xx + 1 < W ? 1 : 0)]) - h2f(x[v - (xx > 0 ? 1 : 0)]))};
                    for (int j = 0; j < ng; j++) {
                        float t = 0.f;
                        for (int k = 0; k < 4; k++) { t += h2f(l[(base + 4 * j + k) * ns + i]) * f[k]; dot2 += (double)h2f(l[(base + 4 * j + k) * ns + i]) * h2f(bg[(base + 4 * j + k) * ns + i]); }
                        const float s = h2f(l[(1 + j) * ns + i]) + t;
                        e0 = fmax(e0, fabs(h2f(o[(1 + j) * ns + i]) - s) / (1e-2 + fabs(s))); dot1 += (double)t * h2f(g[(1 + j) * ns + i]);
                    }
                }
                free(l); free(o); free(g); free(bg); nn_free(d_l); nn_free(d_g);
            }
            const double de = fabs(dot1 - dot2) / (1e-9 + fabs(dot1)); cases++;
            if (!(e0 < 3e-3) || !(de < 2e-3)) { fprintf(stderr, "FAIL: guided head (general), slab %d: forward rel err %.3g, adjoint dot rel err %.3g\n", Z, e0, de); fails++; }
        }
        free(x); nn_free(d_x);
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
        if (f16) for (int axis = 0; axis <= 1; axis++) for (int q = 0; q < 2; q++) {   /* a --lat-basis checkpoint's input: C = 5 random axes, with / without the quantiser */
            const int C = 5; float *bs = malloc(C * 64 * 4), gn[5] = {0.11f, 0.7f, 1.3f, 0.9f, 2.1f};
            for (int k = 0; k < C * 64; k++) bs[k] = 0.3f * ru();
            float *d_bs = nn_malloc(C * 64 * 4), *d_gn = nn_malloc(C * 4); void *d_ya = nn_malloc((C + 3) * cv * 2), *d_yb = nn_malloc((C + 3) * cv * 2);
            uint16_t *ya = malloc((C + 3) * cv * 2), *yb = malloc((C + 3) * cv * 2);
            nn_h2d(d_bs, bs, C * 64 * 4); nn_h2d(d_gn, gn, C * 4);
            nn_pred_input_d(d_ct, D, W, 117.3f, 1.f / 41.7f, d_dy, d_dy + D, axis, d_x, 1);
            nn_lat_input_x(d_x, d, h, h, d_bs, d_gn, C, q ? 0.05f : 0.f, d_ya);
            nn_lat_pred_input_c(d_ct, D, W, 117.3f, 1.f / 41.7f, d_dy, d_dy + D, axis, d_bs, d_gn, C, q ? 0.05f : 0.f, d_yb);
            nn_d2h(ya, d_ya, (C + 3) * cv * 2); nn_d2h(yb, d_yb, (C + 3) * cv * 2);
            size_t nd = 0, n1 = 0; for (size_t k = 0; k < (C + 3) * cv; k++) { nd += ya[k] != yb[k]; n1 += abs((int)ya[k] - (int)yb[k]) > 1; }
            cases++;
            if (n1 > (q ? (C + 3) * cv / 500 : 0) || nd > (C + 3) * cv / 50) { fprintf(stderr, "FAIL: nn_lat_pred_input_c (axis %d, step %d): %zu of %zu values differ, %zu by more than one code\n", axis, q, nd, (C + 3) * cv, n1); fails++; }   /* (quantised: a last-bit difference before rint can move a value by a step) */
            free(bs); free(ya); free(yb); nn_free(d_bs); nn_free(d_gn); nn_free(d_ya); nn_free(d_yb);
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
        if (f16) {   /* the guided head's direct bytes (nn_lat_place_rvg) against nn_lat_up_slab + nn_lat_guide + nn_rv_fuse + nn_pred_place: 9 cell channels, the coefficients in 4 .. 8 */
            const int NC = 9, nb = 4; const size_t hw = (size_t)W * W;
            float *gc = malloc(NC * cv * 4); uint16_t *gc16 = malloc(NC * cv * 2);
            for (size_t k = 0; k < NC * cv; k++) { gc16[k] = f2h((k < 2 * cv ? 3.f : 0.5f) * ru()); gc[k] = h2f(gc16[k]); }
            void *d_gc = nn_malloc(NC * cv * 4), *d_g16 = nn_malloc(NC * cv * 2), *d_sl = nn_malloc(7 * w3 * 2);
            nn_h2d(d_gc, gc, NC * cv * 4); nn_h2d(d_g16, gc16, NC * cv * 2);
            for (int m = 0; m < 2; m++) {   /* m: fp16 cell logits */
                const void *src = m ? d_g16 : d_gc; const size_t es = m ? 2 : 4;
                nn_lat_up_slab(src, m, 2, d, h, h, 0, D, d_sl); nn_lat_up_slab((const char *)src + nb * cv * es, m, 5, d, h, h, 0, D, (char *)d_sl + 2 * w3 * 2);
                nn_lat_guide(d_x, D, W, W, 0, D, d_sl, 2, 0, 0); nn_rv_fuse(d_sl, w3, 1); nn_set_logits_h16(1);
                for (int oi = 0; oi < 5; oi++) for (int mb = 0; mb < 2; mb++) {
                    const int *o = offs[oi], ez = oi == 2 ? 17 : S, ey = oi == 2 ? 19 : S, ex = S;
                    nn_zero(d_s0, (size_t)S * S * S); nn_zero(d_s1, (size_t)S * S * S);
                    if (mb) nn_pred_place_m_d(d_sl, d_bits, D, W, hz, halo, o[0], o[1], o[2], ez, ey, ex, S, d_s0); else nn_pred_place_d(d_sl, d_ct, D, W, hz, halo, o[0], o[1], o[2], ez, ey, ex, S, d_s0);
                    nn_lat_place_rvg(src, m, nb, d, h, h, d_ct, 117.3f, 1.f / 41.7f, mb ? nullptr : d_ct, mb ? d_bits : nullptr, nn_get_pred_iso(), hz, halo, o[0], o[1], o[2], ez, ey, ex, S, d_s1);
                    nn_d2h(s0, d_s0, (size_t)S * S * S); nn_d2h(s1, d_s1, (size_t)S * S * S);
                    size_t nd = 0, n1 = 0, nz = 0; for (size_t k = 0; k < (size_t)S * S * S; k++) { nd += s0[k] != s1[k]; n1 += abs((int)s0[k] - (int)s1[k]) > 1 || (s0[k] == 0) != (s1[k] == 0); nz += s0[k] != 0; }
                    cases++;
                    if (n1 > nz / 500 || nd > nz / 20 + 2 || (oi < 4 && !nz)) { fprintf(stderr, "FAIL: nn_lat_place_rvg (offset %d, fp16 logits %d, mask %d): %zu of %zu written bytes differ, %zu by more than 1 or in the mask\n", oi, m, mb, nd, nz, n1); fails++; }
                }
            }
            nn_set_logits_h16(0); (void)hw;
            free(gc); free(gc16); nn_free(d_gc); nn_free(d_g16); nn_free(d_sl);
        }
        nn_free(d_ct); nn_free(d_dy); nn_free(d_x); nn_free(d_t); nn_free(d_a); nn_free(d_b); nn_free(d_lc); nn_free(d_l16); nn_free(d_lt); nn_free(d_lf); nn_free(d_bits); nn_free(d_s0); nn_free(d_s1);
        free(ct); free(dy); free(ha); free(hb); free(lc); free(lc16); free(s0); free(s1);
    }
    {   /* the sub-cell head (nn_lat_shuffle / _adj): forward against a host reference (fp16 and fp32 coarse logits), the adjoint by dot products */
        const int d = 3, h = 2, w = 2, D = 12, H = 8, W = 8, base = 3, ng = 2, C = base + 64 * ng; const size_t nc = (size_t)d * h * w, hw = (size_t)H * W;
        float *lf = malloc(C * nc * 4); uint16_t *lh_ = malloc(C * nc * 2);
        for (size_t k = 0; k < C * nc; k++) { lh_[k] = f2h(ru()); lf[k] = h2f(lh_[k]); }
        void *d_lf = nn_malloc(C * nc * 4), *d_lh = nn_malloc(C * nc * 2), *d_gc = nn_malloc(C * nc * 4); nn_h2d(d_lf, lf, C * nc * 4); nn_h2d(d_lh, lh_, C * nc * 2);
        float *gc = malloc(C * nc * 4);
        for (int f = 0; f < 2; f++) {
            double e0 = 0, dot1 = 0, dot2 = 0; nn_zero(d_gc, C * nc * 4);
            for (int z0 = 0; z0 < D; z0 += 4) {
                const int nz = 4; const size_t ns = (size_t)nz * hw;
                uint16_t *y = malloc(ng * ns * 2), *o = malloc(ng * ns * 2), *g = malloc(ng * ns * 2);
                for (size_t k = 0; k < ng * ns; k++) { y[k] = f2h(ru()); g[k] = f2h(ru()); }
                void *d_y = nn_malloc(ng * ns * 2), *d_g = nn_malloc(ng * ns * 2);
                nn_h2d(d_y, y, ng * ns * 2); nn_lat_shuffle(f ? d_lh : d_lf, f, d, h, w, base, ng, z0, nz, d_y); nn_d2h(o, d_y, ng * ns * 2);
                nn_h2d(d_g, g, ng * ns * 2); nn_lat_shuffle_adj(d_g, d, h, w, base, ng, z0, nz, d_gc);
                for (size_t i = 0; i < ns; i++) {
                    const int z = z0 + (int)(i / hw), yy = (int)(i / W % H), x = (int)(i % W); const size_t cell = ((size_t)(z >> 2) * h + (yy >> 2)) * w + (x >> 2); const int sb = 16 * (z & 3) + 4 * (yy & 3) + (x & 3);
                    for (int j = 0; j < ng; j++) {
                        const float t = lf[(size_t)(base + 64 * j + sb) * nc + cell], sref = h2f(y[j * ns + i]) + t;
                        e0 = fmax(e0, fabs(h2f(o[j * ns + i]) - sref) / (1e-2 + fabs(sref))); dot1 += (double)t * h2f(g[j * ns + i]);
                    }
                }
                free(y); free(o); free(g); nn_free(d_y); nn_free(d_g);
            }
            nn_d2h(gc, d_gc, C * nc * 4);
            for (size_t k = (size_t)base * nc; k < C * nc; k++) dot2 += (double)lf[k] * gc[k];
            const double de = fabs(dot1 - dot2) / (1e-9 + fabs(dot1)); cases++;
            if (!(e0 < 3e-3) || !(de < 1e-4)) { fprintf(stderr, "FAIL: sub-cell head (%s logits): forward rel err %.3g, adjoint dot rel err %.3g\n", f ? "fp16" : "fp32", e0, de); fails++; }
        }
        free(lf); free(lh_); free(gc); nn_free(d_lf); nn_free(d_lh); nn_free(d_gc);
    }
    const char *e = nn_check(); if (e) { fprintf(stderr, "FAIL: CUDA error %s\n", e); fails++; }
    if (fails) { printf("test_lat: %d of %d cases FAIL\n", fails, cases); return 1; }
    printf("test_lat: ok (%d cases: slab upsampling = the pool4 chain, adjoint = nn_lowres_grad, dot products; predict: direct input and rv bytes = the full-resolution chain)\n", cases);
    return 0;
}
