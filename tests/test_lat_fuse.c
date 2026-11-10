/* the fused gradient pass of the latent head slabs (nn_lat_fuse_grad) against the slab chain it replaces: nn_lat_up_slab_c ->
   nn_lat_guide (dir 0) -> nn_loss_grad_only (+ nn_loss_grad_scale) -> nn_lat_guide (dir 1) -> nn_lat_up_adj_c, slab by slab, the
   same loss statistics. The per-voxel values are the chain's (the same fp16 roundings); the adjoint's sums associate differently:
   the coarse gradients agree to fp32 rounding. Twice the fused path: the same bits (no atomics). */
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint64_t rs = 0x9e3779b97f4a7c15ull;
static uint64_t rn(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static float ru(void) { return (float)((rn() >> 11) * (1.0 / 9007199254740992.0)) * 2.f - 1.f; }
static uint16_t f2h(float f) { _Float16 h = (_Float16)f; uint16_t u; memcpy(&u, &h, 2); return u; }
int main(void) {
    nn_init(0); nn_set_f16(1); nn_set_logits_h16(1); nn_set_loss_grad_h16(1); nn_set_loss_unit_ch7(6);
    int fails = 0, cases = 0;
    static const int dims[4][4] = {{6, 5, 7, 8}, {4, 4, 4, 4}, {8, 3, 6, 16}, {2, 2, 3, 8}};   /* d h w, slab planes */
    for (int tc = 0; tc < 8; tc++) {
        const int d = dims[tc % 4][0], h = dims[tc % 4][1], w = dims[tc % 4][2], Z = dims[tc % 4][3], D = 4 * d, H = 4 * h, W = 4 * w, C = 16, var = tc / 4;
        const size_t pl = (size_t)h * w, cv = (size_t)d * pl, hw = (size_t)H * W, fv = (size_t)D * hw, nsm = (size_t)Z * hw;
        nn_set_grad_scale(var ? 64.f : 1.f);
        const float lam_side = var ? 0.7f : 1.f, lam_n[7] = {var ? 0.5f : 1.f, var ? 0.5f : 1.f, var ? 0.5f : 1.f, var ? 0.5f : 1.f, var ? 0.5f : 1.f, var ? 0.5f : 1.f, var ? 2.f : 1.f};
        uint16_t *hl = malloc(C * cv * 2), *hx = malloc(fv * 2);
        for (size_t k = 0; k < C * cv; k++) hl[k] = f2h(2.f * ru());
        for (size_t k = 0; k < fv; k++) hx[k] = f2h(1.5f * ru());
        uint8_t *hts = malloc(2 * fv), *hms = malloc(fv), *htn = malloc(7 * fv), *hmn = malloc(fv);   /* [c][D][H][W] */
        for (size_t k = 0; k < 2 * fv; k++) hts[k] = rn() % 3 == 0 ? 255 : rn() % 5 == 0 ? (uint8_t)(rn() % 256) : 0;
        for (size_t k = 0; k < 7 * fv; k++) htn[k] = (uint8_t)(rn() % 256);
        for (size_t k = 0; k < fv; k++) { const unsigned r = (unsigned)(rn() % 10); hms[k] = r < 3 ? 0 : r < 8 ? 1 : 8; const unsigned q = (unsigned)(rn() % 10); hmn[k] = q < 2 ? 0 : q < 8 ? 1 : 8; }
        if (tc % 4 == 3) memset(hms, 0, 4 * hw);   /* (whole rows without supervision) */
        uint8_t hws[2] = {1, 1}, hwn[7] = {1, 1, 1, (uint8_t)(var ? 0 : 1), 1, 1, 1};
        void *dl = nn_malloc(C * cv * 2), *dx = nn_malloc(fv * 2), *slab = nn_malloc(C * nsm * 2);
        uint8_t *dts = nn_malloc(2 * nsm), *dms = nn_malloc(nsm), *dtn = nn_malloc(7 * nsm), *dmn = nn_malloc(nsm), *dws = nn_malloc(4), *dwn = nn_malloc(8);
        nn_h2d(dl, hl, C * cv * 2); nn_h2d(dx, hx, fv * 2); nn_h2d(dws, hws, 2); nn_h2d(dwn, hwn, 7);
        const shape5 w2 = {1, 2, D, H, W}, w7 = {1, 7, D, H, W};
        double *ds = nn_malloc(64 * sizeof(double)); nn_zero(ds, 64 * sizeof(double));
        double *ds2 = nn_malloc(64 * sizeof(double)); nn_zero(ds2, 64 * sizeof(double));
        float *ss = nn_malloc(nn_loss_scratch(w2) + 64), *sn = nn_malloc(nn_loss_scratch(w7) + 64);
        float *ga = nn_malloc(C * cv * 4), *gb = nn_malloc(C * cv * 4), *gb2 = nn_malloc(C * cv * 4), *scr = nn_malloc((size_t)C * Z * pl * 4);
        float *part = nn_malloc(nn_lat_fuse_part(7, h, w, Z));
        nn_zero(ga, C * cv * 4); nn_zero(gb, C * cv * 4); nn_zero(gb2, C * cv * 4);
        float pw, rw; int u7; nn_loss_cfg_get(&pw, &rw, &u7);
        for (int pass = 0; pass < 2; pass++) {
            for (int z0 = 0; z0 < D; z0 += Z) {
                const int nz = z0 + Z < D ? Z : D - z0; const size_t ns = (size_t)nz * hw, vb = (size_t)z0 * hw;
                const shape5 two = {1, 2, nz, H, W}, seven = {1, 7, nz, H, W};
                /* the slab's targets [c][nz][H][W] */
                for (int c = 0; c < 2; c++) nn_h2d(dts + c * ns, hts + c * fv + vb, ns);
                for (int c = 0; c < 7; c++) nn_h2d(dtn + c * ns, htn + c * fv + vb, ns);
                nn_h2d(dms, hms + vb, ns); nn_h2d(dmn, hmn + vb, ns); nn_sync();
                nn_lat_up_slab_c(dl, 1, C, d, h, w, z0, nz, slab, 0, d);
                nn_lat_guide(dx, D, H, W, z0, nz, slab, 11, 0, 0);
#define LG(c) ((char *)slab + (size_t)(c) * ns * 2)
                if (!pass) {
                    nn_loss_stats_acc(LG(0), dts, dms, dws, two, ds + 10); nn_loss_stats_acc(LG(2), dtn, dmn, dwn, seven, ds + 20);
                    nn_lat_fuse_stats(dl, 1, 0, d, d, h, w, z0, nz, 0, 2, 1, 11, dx, dts, dms, dws, pw, -1, ds2 + 10);
                    nn_lat_fuse_stats(dl, 1, 0, d, d, h, w, z0, nz, 2, 7, 0, -1, nullptr, dtn, dmn, dwn, pw, u7, ds2 + 20);
                    continue;
                }
                nn_loss_grad_only(LG(0), dts, dms, dws, two, 0.f, LG(0), ss);
                if (lam_side != 1.f) nn_loss_grad_scale(LG(0), 2 * ns, lam_side);
                nn_loss_grad_only(LG(2), dtn, dmn, dwn, seven, 0.f, LG(2), sn);
                if (lam_n[0] != 1.f) nn_loss_grad_scale(LG(2), 6 * ns, lam_n[0]);
                if (lam_n[6] != 1.f) nn_loss_grad_scale(LG(8), ns, lam_n[6]);
                nn_zero(LG(9), 2 * ns * 2);   /* (the winding mod 4 channels: not in this test) */
                nn_lat_guide(dx, D, H, W, z0, nz, slab, 11, 0, 1);
                nn_lat_up_adj_c(slab, C, d, h, w, z0, nz, ga, scr, 0, d);
                for (int rep = 0; rep < 2; rep++) {
                    float *g = rep ? gb2 : gb;
                    nn_lat_fuse_grad(dl, 1, 0, d, d, h, w, z0, nz, 0, 2, 1, 11, dx, dts, dms, dws, ss, pw, rw, -1, &lam_side, part, g, 0, d);
                    nn_lat_fuse_grad(dl, 1, 0, d, d, h, w, z0, nz, 2, 7, 0, -1, nullptr, dtn, dmn, dwn, sn, pw, rw, u7, lam_n, part, g, 0, d);
                }
#undef LG
            }
            if (!pass) {   /* the fused statistics against the chain's (before nn_loss_stats_fin reduces them) */
                double ha[64], hb[64]; nn_sync(); nn_d2h(ha, ds, sizeof ha); nn_d2h(hb, ds2, sizeof hb);
                double se = 0; int nzs = 0; for (int k = 10; k < 55; k++) { const double e = fabs(ha[k] - hb[k]) / (fabs(ha[k]) + 1e-12); if (e > se) se = e; nzs += ha[k] != 0; }
                const int sbad = !(se < 2e-5) || nzs < 30;
                printf("  %dx%dx%d cells, slabs of %d, variant %d: statistics: max relative difference %.2g over %d sums%s\n", d, h, w, Z, var, se, nzs, sbad ? "  FAIL" : ""); fails += sbad; cases++;
                nn_loss_stats_fin(ds + 10, dws, w2, ss); nn_loss_stats_fin(ds + 20, dwn, w7, sn);
            }
        }
        float *ra = malloc(C * cv * 4), *rb = malloc(C * cv * 4), *rb2 = malloc(C * cv * 4);
        nn_sync(); nn_d2h(ra, ga, C * cv * 4); nn_d2h(rb, gb, C * cv * 4); nn_d2h(rb2, gb2, C * cv * 4);
        double mx = 0, mg = 0, sq = 0, sr = 0; size_t nrep = 0, nzr = 0;
        for (size_t k = 0; k < C * cv; k++) {
            const double e = fabs((double)ra[k] - rb[k]); if (e > mx) mx = e; if (fabs(ra[k]) > mg) mg = fabs(ra[k]); sq += e * e; sr += (double)ra[k] * ra[k];
            nrep += memcmp(rb + k, rb2 + k, 4) != 0; nzr += ra[k] != 0;
        }
        const double rel = mx / (mg > 0 ? mg : 1), rms = sqrt(sq / (sr > 0 ? sr : 1));
        const int bad = !(rel < 2e-5) || !(rms < 5e-6) || nrep || nzr < C * cv / 3;
        printf("  %dx%dx%d cells, slabs of %d, variant %d: max |fused - chain| %.3g of max %.3g (%.2g), rms ratio %.2g, %zu of %zu differ between two fused runs%s\n",
               d, h, w, Z, var, mx, mg, rel, rms, nrep, C * cv, bad ? "  FAIL" : "");
        fails += bad; cases++;
        if (bad && getenv("TLF_DBG")) { int shown = 0; for (size_t k = 0; k < C * cv && shown < 16; k++) if (fabs((double)ra[k] - rb[k]) > 1e-4 * mg) { printf("    c %zu cell %zu: chain %g fused %g\n", k / cv, k % cv, ra[k], rb[k]); shown++; } }
        free(hl); free(hx); free(hts); free(hms); free(htn); free(hmn); free(ra); free(rb); free(rb2);
        nn_free(dl); nn_free(dx); nn_free(slab); nn_free(dts); nn_free(dms); nn_free(dtn); nn_free(dmn); nn_free(dws); nn_free(dwn); nn_free(ds); nn_free(ds2); nn_free(ss); nn_free(sn);
        nn_free(ga); nn_free(gb); nn_free(gb2); nn_free(scr); nn_free(part);
    }
    nn_set_grad_scale(1.f);
    if (fails) { printf("test_lat_fuse: %d of %d FAILED\n", fails, cases); return 1; }
    printf("test_lat_fuse: ok (%d cases)\n", cases);
    return 0;
}
