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
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
/* TLF_BENCH=P: one 32-plane slab of a P^2 window, ms per call (min and median of 80; beside another process on the card only the min means anything: its time slices add 2 - 6 ms steps) of the fused kernels and of the slab chain's pieces */
static int bench(int P) {
    const int d = 8, h = P / 4, w = P / 4, D = 32, H = P, W = P, C = 16, R = 80;
    const size_t pl = (size_t)h * w, cv = (size_t)d * pl, ns = (size_t)D * H * W;
    uint16_t *hl = malloc(C * cv * 2); for (size_t k = 0; k < C * cv; k++) hl[k] = f2h(2.f * ru());
    uint16_t *hx = malloc(ns * 2); for (size_t k = 0; k < ns; k++) hx[k] = f2h(1.5f * ru());
    uint8_t *ht = malloc(7 * ns), *hm = malloc(ns);
    for (size_t k = 0; k < 7 * ns; k++) ht[k] = (uint8_t)(rn() >> 20);
    for (size_t k = 0; k < ns; k++) { const unsigned r = (unsigned)(rn() % 10); hm[k] = r < 2 ? 0 : r < 8 ? 1 : 8; }
    uint8_t one7[7] = {1, 1, 1, 1, 1, 1, 1};
    void *dl = nn_malloc(C * cv * 2), *dx = nn_malloc(ns * 2), *slab = nn_malloc(C * ns * 2);
    uint8_t *dt = nn_malloc(7 * ns), *dm = nn_malloc(ns), *dw = nn_malloc(8);
    nn_h2d(dl, hl, C * cv * 2); nn_h2d(dx, hx, ns * 2); nn_h2d(dt, ht, 7 * ns); nn_h2d(dm, hm, ns); nn_h2d(dw, one7, 7);
    const shape5 two = {1, 2, D, H, W}, seven = {1, 7, D, H, W};
    double *ds = nn_malloc(64 * sizeof(double)); nn_zero(ds, 64 * sizeof(double));
    float *ss = nn_malloc(nn_loss_scratch(two) + 64), *sn = nn_malloc(nn_loss_scratch(seven) + 64), *g = nn_malloc(C * cv * 4), *scr = nn_malloc((size_t)C * D * pl * 4), *part = nn_malloc(nn_lat_fuse_part(7, h, w, D));
    nn_zero(g, C * cv * 4);
    float pw, rw; int u7; nn_loss_cfg_get(&pw, &rw, &u7);
    const float l1[7] = {1, 1, 1, 1, 1, 1, 1};
    nn_lat_up_slab_c(dl, 1, C, d, h, w, 0, D, slab, 0, d);
    nn_loss_stats_acc(slab, dt, dm, dw, two, ds + 10); nn_loss_stats_acc((char *)slab + 2 * ns * 2, dt, dm, dw, seven, ds + 20);
    nn_loss_stats_fin(ds + 10, dw, two, ss); nn_loss_stats_fin(ds + 20, dw, seven, sn); nn_sync();
    static const char *nm[] = {"fused grad normal (7 ch)", "fused grad normal + loss sums", "fused grad guided side", "fused grad guided side + loss sums", "fused stats normal", "fused stats guided side",
                               "fused stats normal, mask only", "chain: upsample 16 ch", "chain: guide fwd", "chain: stats normal", "chain: grad normal (in place)", "chain: adjoint 16 ch"};
    printf("bench P %d, one 32-plane slab (%.1f Mvox):\n", P, ns / 1e6);
    for (int k = 0; k < 12; k++) {
        double t[80];
        for (int r = 0; r < R; r++) {
            if (k == 10) { nn_lat_up_slab_c(dl, 1, C, d, h, w, 0, D, slab, 0, d); }
            nn_sync(); const double t0 = now();
            switch (k) {
            case 0: nn_lat_fuse_grad(dl, 1, 0, d, d, h, w, 0, D, 2, 7, 0, -1, nullptr, dt, dm, dw, sn, pw, rw, u7, l1, part, g, 0, d, nullptr); break;
            case 1: nn_lat_fuse_grad(dl, 1, 0, d, d, h, w, 0, D, 2, 7, 0, -1, nullptr, dt, dm, dw, sn, pw, rw, u7, l1, part, g, 0, d, ds + 20); break;
            case 2: nn_lat_fuse_grad(dl, 1, 0, d, d, h, w, 0, D, 0, 2, 1, 11, dx, dt, dm, dw, ss, pw, rw, -1, l1, part, g, 0, d, nullptr); break;
            case 3: nn_lat_fuse_grad(dl, 1, 0, d, d, h, w, 0, D, 0, 2, 1, 11, dx, dt, dm, dw, ss, pw, rw, -1, l1, part, g, 0, d, ds + 10); break;
            case 4: nn_lat_fuse_stats(dl, 1, 0, d, d, h, w, 0, D, 2, 7, 0, -1, nullptr, dt, dm, dw, pw, u7, ds + 20, 0); break;
            case 5: nn_lat_fuse_stats(dl, 1, 0, d, d, h, w, 0, D, 0, 2, 1, 11, dx, dt, dm, dw, pw, -1, ds + 10, 0); break;
            case 6: nn_lat_fuse_stats(dl, 1, 0, d, d, h, w, 0, D, 2, 7, 0, -1, nullptr, dt, dm, dw, pw, u7, ds + 20, 1); break;
            case 7: nn_lat_up_slab_c(dl, 1, C, d, h, w, 0, D, slab, 0, d); break;
            case 8: nn_lat_guide(dx, D, H, W, 0, D, slab, 11, 0, 0); break;
            case 9: nn_loss_stats_acc((char *)slab + 2 * ns * 2, dt, dm, dw, seven, ds + 20); break;
            case 10: nn_loss_grad_only((char *)slab + 2 * ns * 2, dt, dm, dw, seven, 0.f, (char *)slab + 2 * ns * 2, sn); break;
            case 11: nn_lat_up_adj_c(slab, C, d, h, w, 0, D, g, scr, 0, d); break;
            }
            nn_sync(); t[r] = (now() - t0) * 1e3;
        }
        for (int a = 0; a < R; a++) for (int c = a + 1; c < R; c++) if (t[c] < t[a]) { const double x = t[a]; t[a] = t[c]; t[c] = x; }
        printf("  %-36s min %7.2f ms  median %7.2f ms\n", nm[k], t[0], t[R / 2]);
    }
    return 0;
}
/* try6's head layout (audit 2026-11-14 #16): 160 cell channels (the sub-cell head: base 32 + 128), 23 upsampled (LUPN), 32 fp32 gradient
   rows (LGFC), the fused normal / density group at [2, 9) and the slab chain for [0, 2) + [9, 23) (train.c head_slab_pass fr: norm_ch 2,
   no guided side), gscale 1024, slabs of 16 planes with an 8-plane tail. The fused pass (nn_lat_fuse_grad + nn_lat_up_adj_c of the two
   ranges) against the chain (nn_lat_up_slab_c, nn_loss_grad_only + nn_loss_grad_scale, nn_lat_up_adj_c of all 23) at the per-voxel size
   of training: lambda = window voxels / the ~2.3e8 masked voxels of a P 768 window puts the BCE gradient at gscale x 1 / 2.3e8 (~4.5e-6,
   fp16 subnormal), the other channels' gradients the same size. Pass as the other cases (fp32 rounding of the adjoint's order).
   Soft report (printed, not a failure; hgscale-20261114 rescales the head gradient): the chain again with every gradient x 65536 (fp16
   normal: the reference), divided back; coarse gradient elements that are 0 while the reference is not, and the rel L2 error. */
static int try6_case(int *cases) {
    const int d = 6, h = 5, w = 7, Z = 16, D = 4 * d, H = 4 * h, W = 4 * w, C = 160, UPC = 23, NF = 32, fr[2][2] = {{0, 2}, {9, 23}};
    const size_t pl = (size_t)h * w, cv = (size_t)d * pl, hw = (size_t)H * W, fv = (size_t)D * hw, nsm = (size_t)Z * hw;
    const float gs = 1024.f, nprod = 768.f * 768.f * 768.f * 0.5f, lam = (float)fv / nprod, amp = gs / nprod, up = 65536.f;
    nn_set_grad_scale(gs);
    uint16_t *hl = malloc(C * cv * 2); for (size_t k = 0; k < C * cv; k++) hl[k] = f2h(2.f * ru());
    uint8_t *htn = malloc(7 * fv), *hmn = malloc(fv);
    for (size_t k = 0; k < 7 * fv; k++) htn[k] = (uint8_t)(rn() % 256);
    for (size_t k = 0; k < fv; k++) { const unsigned q = (unsigned)(rn() % 10); hmn[k] = q < 2 ? 0 : q < 8 ? 1 : 8; }
    float *hgo = calloc((size_t)UPC * fv, 4);   /* the other heads' slab gradients (fp32 before the fp16 rounding) on the chain channels */
    for (int r = 0; r < 2; r++) for (int c = fr[r][0]; c < fr[r][1]; c++) for (size_t k = 0; k < fv; k++) hgo[(size_t)c * fv + k] = amp * ru();
    uint16_t *ho = malloc(nsm * 2);
    const uint8_t hwn[7] = {1, 1, 1, 1, 1, 1, 1}; const float lam7[7] = {lam, lam, lam, lam, lam, lam, lam};
    void *dl = nn_malloc(C * cv * 2), *slab = nn_malloc(UPC * nsm * 2), *slab2 = nn_malloc(UPC * nsm * 2);
    uint8_t *dtn = nn_malloc(7 * nsm), *dmn = nn_malloc(nsm), *dwn = nn_malloc(8);
    nn_h2d(dl, hl, C * cv * 2); nn_h2d(dwn, hwn, 7);
    const shape5 w7 = {1, 7, D, H, W};
    double *ds = nn_malloc(64 * sizeof(double)); nn_zero(ds, 64 * sizeof(double));
    float *sn = nn_malloc(nn_loss_scratch(w7) + 64), *ga = nn_malloc(NF * cv * 4), *gb = nn_malloc(NF * cv * 4), *gr = nn_malloc(NF * cv * 4), *scr = nn_malloc((size_t)UPC * Z * pl * 4);
    float *part = nn_malloc(nn_lat_fuse_part(7, h, w, Z));
    nn_zero(ga, NF * cv * 4); nn_zero(gb, NF * cv * 4); nn_zero(gr, NF * cv * 4);
    float pw, rw; int u7; nn_loss_cfg_get(&pw, &rw, &u7);
#define LG(p, c) ((char *)(p) + (size_t)(c) * ns * 2)
    for (int pass = 0; pass < 2; pass++) {
        for (int z0 = 0; z0 < D; z0 += Z) {
            const int nz = z0 + Z < D ? Z : D - z0; const size_t ns = (size_t)nz * hw, vb = (size_t)z0 * hw;
            const shape5 seven = {1, 7, nz, H, W};
            for (int c = 0; c < 7; c++) nn_h2d(dtn + c * ns, htn + c * fv + vb, ns);
            nn_h2d(dmn, hmn + vb, ns); nn_sync();
            if (!pass) { nn_lat_up_slab_c(dl, 1, UPC, d, h, w, z0, nz, slab, 0, d); nn_loss_stats_acc(LG(slab, 2), dtn, dmn, dwn, seven, ds + 20); continue; }
            for (int ref = 0; ref < 2; ref++) {   /* 0: the chain at training size (ga) and the fused pass (gb); 1: the chain x 65536 (gr) */
                const float m = ref ? up : 1.f;
                nn_lat_up_slab_c(dl, 1, UPC, d, h, w, z0, nz, slab, 0, d);
                nn_loss_grad_only(LG(slab, 2), dtn, dmn, dwn, seven, 0.f, LG(slab, 2), sn); nn_loss_grad_scale(LG(slab, 2), 7 * ns, lam * m);
                for (int r = 0; r < 2; r++) for (int c = fr[r][0]; c < fr[r][1]; c++) {
                    for (size_t k = 0; k < ns; k++) ho[k] = f2h(m * hgo[(size_t)c * fv + vb + k]);
                    nn_h2d(LG(slab, c), ho, ns * 2); if (!ref) nn_h2d(LG(slab2, c), ho, ns * 2);
                }
                nn_lat_up_adj_c(slab, UPC, d, h, w, z0, nz, ref ? gr : ga, scr, 0, d);
                if (ref) continue;
                nn_lat_fuse_grad(dl, 1, 0, d, d, h, w, z0, nz, 2, 7, 0, -1, nullptr, dtn, dmn, dwn, sn, pw, rw, u7, lam7, part, gb, 0, d, nullptr);
                for (int r = 0; r < 2; r++) nn_lat_up_adj_c(LG(slab2, fr[r][0]), fr[r][1] - fr[r][0], d, h, w, z0, nz, gb + (size_t)fr[r][0] * cv, scr, 0, d);
            }
        }
        if (!pass) nn_loss_stats_fin(ds + 20, dwn, w7, sn);
    }
#undef LG
    float *ra = malloc(NF * cv * 4), *rb = malloc(NF * cv * 4), *rr = malloc(NF * cv * 4);
    nn_sync(); nn_d2h(ra, ga, NF * cv * 4); nn_d2h(rb, gb, NF * cv * 4); nn_d2h(rr, gr, NF * cv * 4);
    double mx = 0, mg = 0, sq = 0, sr = 0, e2 = 0, r2 = 0; size_t nzr = 0, zc = 0, zn = 0;
    for (size_t k = 0; k < NF * cv; k++) {
        const double e = fabs((double)ra[k] - rb[k]); if (e > mx) mx = e; if (fabs(ra[k]) > mg) mg = fabs(ra[k]); sq += e * e; sr += (double)ra[k] * ra[k]; nzr += ra[k] != 0;
        const double rf = rr[k] / (double)up; e2 += (rb[k] - rf) * (rb[k] - rf); r2 += rf * rf;
        if (rf != 0) { zn++; zc += rb[k] == 0.f; }
    }
    const double rel = mx / (mg > 0 ? mg : 1), rms = sqrt(sq / (sr > 0 ? sr : 1));
    const int bad = !(rel < 2e-5) || !(rms < 5e-6) || nzr < (size_t)UPC * cv / 2;
    printf("  try6 head (160 cells, 23 upsampled, 32 fp32 rows, fused [2, 9), chain [0, 2) + [9, 23), gscale 1024): max |fused - chain| %.3g of max %.3g (%.2g), rms ratio %.2g, %zu nonzero%s\n",
           mx, mg, rel, rms, nzr, bad ? "  FAIL" : "");
    printf("  try6 head, fp16 dead zone (soft report, not a failure: hgscale-20261114 rescales the head gradient): %zu of %zu coarse gradient elements 0 where the x 65536 "
           "reference is not, rel L2 vs it %.2e\n", zc, zn, sqrt(e2 / (r2 > 0 ? r2 : 1)));
    (*cases)++;
    free(hl); free(htn); free(hmn); free(hgo); free(ho); free(ra); free(rb); free(rr);
    nn_free(dl); nn_free(slab); nn_free(slab2); nn_free(dtn); nn_free(dmn); nn_free(dwn); nn_free(ds); nn_free(sn); nn_free(ga); nn_free(gb); nn_free(gr); nn_free(scr); nn_free(part);
    nn_set_grad_scale(1.f);
    return bad;
}
int main(void) {
    nn_init(0); nn_set_f16(1); nn_set_logits_h16(1); nn_set_loss_grad_h16(1); nn_set_loss_unit_ch7(6);
    if (getenv("TLF_BENCH")) return bench(atoi(getenv("TLF_BENCH")));
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
        double *ds3 = nn_malloc(64 * sizeof(double)); nn_zero(ds3, 64 * sizeof(double));   /* the mask sums of a mask_only pass + the gradient pass's loss sums */
        float *ss = nn_malloc(nn_loss_scratch(w2) + 64), *sn = nn_malloc(nn_loss_scratch(w7) + 64);
        float *ga = nn_malloc(C * cv * 4), *gb = nn_malloc(C * cv * 4), *gb2 = nn_malloc(C * cv * 4), *scr = nn_malloc((size_t)C * Z * pl * 4);
        float *part = nn_malloc(nn_lat_fuse_part(7, h, w, Z));
        nn_zero(ga, C * cv * 4); nn_zero(gb, C * cv * 4); nn_zero(gb2, C * cv * 4);
        float pw, rw; int u7; nn_loss_cfg_get(&pw, &rw, &u7);
        double hs0[64];
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
                    nn_lat_fuse_stats(dl, 1, 0, d, d, h, w, z0, nz, 0, 2, 1, 11, dx, dts, dms, dws, pw, -1, ds2 + 10, 0);
                    nn_lat_fuse_stats(dl, 1, 0, d, d, h, w, z0, nz, 2, 7, 0, -1, nullptr, dtn, dmn, dwn, pw, u7, ds2 + 20, 0);
                    nn_lat_fuse_stats(dl, 1, 0, d, d, h, w, z0, nz, 0, 2, 1, 11, dx, dts, dms, dws, pw, -1, ds3 + 10, 1);
                    nn_lat_fuse_stats(dl, 1, 0, d, d, h, w, z0, nz, 2, 7, 0, -1, nullptr, dtn, dmn, dwn, pw, u7, ds3 + 20, 1);
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
                    nn_lat_fuse_grad(dl, 1, 0, d, d, h, w, z0, nz, 0, 2, 1, 11, dx, dts, dms, dws, ss, pw, rw, -1, &lam_side, part, g, 0, d, rep ? nullptr : ds3 + 10);
                    nn_lat_fuse_grad(dl, 1, 0, d, d, h, w, z0, nz, 2, 7, 0, -1, nullptr, dtn, dmn, dwn, sn, pw, rw, u7, lam_n, part, g, 0, d, rep ? nullptr : ds3 + 20);
                }
#undef LG
            }
            if (!pass) {   /* the fused statistics against the chain's (before nn_loss_stats_fin reduces them) */
                double ha[64], hb[64]; nn_sync(); nn_d2h(ha, ds, sizeof ha); nn_d2h(hb, ds2, sizeof hb); memcpy(hs0, ha, sizeof ha);
                double se = 0; int nzs = 0; for (int k = 10; k < 55; k++) { const double e = fabs(ha[k] - hb[k]) / (fabs(ha[k]) + 1e-12); if (e > se) se = e; nzs += ha[k] != 0; }
                const int sbad = !(se < 2e-5) || nzs < 30;
                printf("  %dx%dx%d cells, slabs of %d, variant %d: statistics: max relative difference %.2g over %d sums%s\n", d, h, w, Z, var, se, nzs, sbad ? "  FAIL" : ""); fails += sbad; cases++;
                nn_loss_stats_fin(ds + 10, dws, w2, ss); nn_loss_stats_fin(ds + 20, dwn, w7, sn);
            }
        }
        {   /* mask-only statistics + the gradient pass's loss sums = the chain's statistics */
            double hc[64]; nn_sync(); nn_d2h(hc, ds3, sizeof hc);
            double se = 0; for (int k = 10; k < 55; k++) { const double e = fabs(hs0[k] - hc[k]) / (fabs(hs0[k]) + 1e-12); if (e > se) se = e; }
            const int sbad = !(se < 2e-5);
            printf("  %dx%dx%d cells, slabs of %d, variant %d: mask-only statistics + gradient-pass sums: max relative difference %.2g%s\n", d, h, w, Z, var, se, sbad ? "  FAIL" : ""); fails += sbad; cases++;
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
        nn_free(dl); nn_free(dx); nn_free(slab); nn_free(dts); nn_free(dms); nn_free(dtn); nn_free(dmn); nn_free(dws); nn_free(dwn); nn_free(ds); nn_free(ds2); nn_free(ds3); nn_free(ss); nn_free(sn);
        nn_free(ga); nn_free(gb); nn_free(gb2); nn_free(scr); nn_free(part);
    }
    nn_set_grad_scale(1.f);
    fails += try6_case(&cases);
    if (fails) { printf("test_lat_fuse: %d of %d FAILED\n", fails, cases); return 1; }
    printf("test_lat_fuse: ok (%d cases)\n", cases);
    return 0;
}
