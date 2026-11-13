/* nn_fit_accum_lat (predict --fit-zarr's fused fit kernel) against the plane path it replaces: nn_lat_up_slab + nn_lat_shuffle (the
   sub-cell pair 0-1) + nn_fstem_fwd on the z-scored CT (nn_lat_ct_x) into [11][W^3] fp16 planes, then nn_fit_accum_dm and nn_fit_accum_w4
   (task unwrap2: phase 0-1, normal 2-7, density 8, winding mod 4 9-10, d 16), field cells of 8. A 96^3 window (24^3 cells, cout 144 =
   16 + 128 sub-cell outputs), halo 16, the box at window offsets on and off the cell grid; with and without the stem and the sub-cell
   term; field cells of 8 (fit_accum_k's tree), 2 and 4 (fit_accum_cell_k's serial sums: the unroll's --fit-h 2). Pass: the 12 + 9
   accumulator planes bit for bit. Usage: test_fit_lat (UFSM_TEST_BENCH=1: timings at --fit-h 8 and 2) */
#include "nn.h"
#include "fit.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
enum { WW = 96, CW = 24, NB = 16, COUT = 144, HALO = 16, C = 16, NO = 6, DD = 16 };
static double rnd(uint64_t *s) { *s = *s * 6364136223846793005ull + 1442695040888963407ull; return (double)(*s >> 11) * 0x1p-53 - 0.5; }
static uint16_t f2h(float v) { _Float16 f = (_Float16)v; uint16_t h; memcpy(&h, &f, 2); return h; }
int main(void) {
    nn_init(0); nn_set_f16(1); nn_set_logits_h16(1); int fails = 0; uint64_t s = 7;
    const size_t w3 = (size_t)WW * WW * WW, cv = (size_t)CW * CW * CW, NP = C * 64 + C + NO * C;
    uint8_t *ct = malloc(w3); uint16_t *lg = malloc(COUT * cv * 2); float *pf = malloc(NP * 4);
    for (size_t i = 0; i < w3; i++) ct[i] = rnd(&s) < -0.3 ? 0 : (uint8_t)(128 + 250 * rnd(&s));
    for (size_t i = 0; i < COUT * cv; i++) lg[i] = f2h((float)(3 * rnd(&s)));
    for (size_t i = 0; i < NP; i++) pf[i] = (float)(0.3 * rnd(&s));
    const float mean = 100.f, isd = 1.f / 40.f, dscale = 32.f;
    uint8_t *ctd = nn_malloc(w3); void *lgd = nn_malloc(COUT * cv * 2), *xd = nn_malloc(w3 * 2), *pl = nn_malloc(11 * w3 * 2); float *pd = nn_malloc(NP * 4);
    nn_h2d(ctd, ct, w3); nn_h2d(lgd, lg, COUT * cv * 2); nn_h2d(pd, pf, NP * 4);
    nn_lat_ct_x(ctd, WW, 0, WW, mean, isd, xd);
    const int chs[NO] = {0, 1, 9, 10, -1, -1};   /* the stem's outputs: channels 0, 1, 9, 10, 11, 12 (predict FZ_STEM's map c < 11) */
    int so[16]; for (int c = 0; c < 16; c++) so[c] = -1;
    so[0] = 0; so[1] = 1; so[9] = 2; so[10] = 3; so[11] = 4; so[12] = 5;
    static const int OFF[3][3] = {{0, 0, 0}, {-8, 8, 0}, {16, -24, 8}};   /* the window's origin in box coordinates (tz ty tx), box multiples of 8 */
    static const int FHS[3] = {8, 2, 4};
    for (int fi = 0; fi < 3; fi++) for (int st = 0; st < 2; st++) for (int sb = 0; sb < 2; sb++) for (int o = 0; o < 3; o++) {
        const int FH = FHS[fi];
        const int tz = OFF[o][0], ty = OFF[o][1], tx = OFF[o][2], box[3] = {WW + 8, WW - 8, WW}, F[3] = {(box[0] + FH - 1) / FH, (box[1] + FH - 1) / FH, (box[2] + FH - 1) / FH};
        const size_t nf = (size_t)F[0] * F[1] * F[2];
        float *a0 = nn_malloc(12 * nf * 4), *a1 = nn_malloc(12 * nf * 4), *b0 = nn_malloc(9 * nf * 4), *b1 = nn_malloc(9 * nf * 4);
        nn_zero(a0, 12 * nf * 4); nn_zero(a1, 12 * nf * 4); nn_zero(b0, 9 * nf * 4); nn_zero(b1, 9 * nf * 4);
        nn_lat_up_slab(lgd, 1, 11, CW, CW, CW, 0, WW, pl);
        if (sb) nn_lat_shuffle(lgd, 1, CW, CW, CW, NB, 2, 0, WW, pl);
        if (st && nn_fstem_fwd(xd, WW, WW, WW, 0, WW, pd, C, NO, chs, pl, w3)) { fprintf(stderr, "fstem shape\n"); return 1; }
        nn_fit_accum_dm((const float *)pl, ctd, nullptr, WW, WW, HALO, HALO, tz, ty, tx, box, FH, F, -1, 0, 2, 8, dscale, 0, a0);
        nn_fit_accum_w4((const float *)pl, ctd, nullptr, WW, WW, HALO, HALO, tz, ty, tx, box, FH, F, 9, DD, b0);
        if (nn_fit_accum_lat(lgd, 1, CW, CW, CW, sb ? NB : -1, st ? pd : nullptr, C, so, ctd, ctd, nullptr, mean, isd, HALO, HALO, tz, ty, tx, box, FH, F,
                             -1, 0, 2, 8, dscale, 0, a1, 9, DD, b1)) { fprintf(stderr, "nn_fit_accum_lat refused the configuration\n"); return 1; }
        nn_sync();
        if (o == 0 && sb == 0 && fi == 0) {   /* (diagnosis: the per-voxel logits of the fused kernel against the planes) */
            float *dg = nn_malloc(11 * w3 * 4); nn_zero(dg, 11 * w3 * 4); nn_fit_accum_lat_debug(dg);
            float *sa = nn_malloc(12 * nf * 4), *sb4 = nn_malloc(9 * nf * 4); nn_zero(sa, 12 * nf * 4); nn_zero(sb4, 9 * nf * 4);   /* (scratch sums) */
            nn_fit_accum_lat(lgd, 1, CW, CW, CW, -1, st ? pd : nullptr, C, so, ctd, ctd, nullptr, mean, isd, HALO, HALO, tz, ty, tx, box, FH, F, -1, 0, 2, 8, dscale, 0, sa, 9, DD, sb4);
            nn_fit_accum_lat_debug(nullptr); nn_sync(); nn_free(sa); nn_free(sb4);
            float *hd = malloc(11 * w3 * 4); uint16_t *hp = malloc(11 * w3 * 2); nn_d2h(hd, dg, 11 * w3 * 4); nn_d2h(hp, pl, 11 * w3 * 2);
            static const int CH[11] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};   /* (staged order: phase 0-1, normal 2-7, density 8, winding 9-10) */
            for (int k = 0; k < 11; k++) { size_t n = 0, m = 0, first = 0;
                for (size_t i = 0; i < w3; i++) { const int z = (int)(i / (WW * WW)), y = (int)(i / WW % WW), x = (int)(i % WW);
                    if (z < HALO || z >= WW - HALO || y < HALO || y >= WW - HALO || x < HALO || x >= WW - HALO) continue;
                    _Float16 f; memcpy(&f, hp + (size_t)CH[k] * w3 + i, 2); m++;
                    if ((float)f != hd[(size_t)k * w3 + i]) { if (!n) first = i; n++; } }
                if (n) { _Float16 f; memcpy(&f, hp + (size_t)CH[k] * w3 + first, 2);
                         printf("     logits of channel %d (stem %d): %zu of %zu voxels differ, first at z %zu y %zu x %zu: planes %.6g fused %.6g\n", CH[k], st, n, m,
                                first / (WW * WW), first / WW % WW, first % WW, (double)(float)f, (double)hd[(size_t)k * w3 + first]); } }
            free(hd); free(hp); nn_free(dg);
        }
        uint32_t *h0 = malloc(12 * nf * 4), *h1 = malloc(12 * nf * 4), *g0 = malloc(9 * nf * 4), *g1 = malloc(9 * nf * 4);
        nn_d2h(h0, a0, 12 * nf * 4); nn_d2h(h1, a1, 12 * nf * 4); nn_d2h(g0, b0, 9 * nf * 4); nn_d2h(g1, b1, 9 * nf * 4);
        size_t nd[21] = {0}, nset = 0;
        for (int k = 0; k < 12; k++) for (size_t i = 0; i < nf; i++) nd[k] += h0[k * nf + i] != h1[k * nf + i];
        for (int k = 0; k < 9; k++) for (size_t i = 0; i < nf; i++) nd[12 + k] += g0[k * nf + i] != g1[k * nf + i];
        for (size_t i = 0; i < nf; i++) nset += h0[nf + i] != 0;
        size_t tot = 0; for (int k = 0; k < 21; k++) tot += nd[k];
        const int bad = tot != 0 || nset < nf / 4;
        printf("%s cells %d stem %d sub-cell %d window at %3d %3d %3d: %zu of %zu cells with material, %zu accumulator values differ", bad ? "FAIL" : "ok  ", FH, st, sb, tz, ty, tx, nset, nf, tot);
        if (tot) { printf(" (planes:"); for (int k = 0; k < 21; k++) if (nd[k]) printf(" %d:%zu", k, nd[k]); printf(")"); }
        printf("%s\n", bad ? "  <--" : "");
        fails += bad;
        free(h0); free(h1); free(g0); free(g1); nn_free(a0); nn_free(a1); nn_free(b0); nn_free(b1);
    }
    if (getenv("UFSM_TEST_BENCH")) {   /* timing: a 256^3 window (the stem and the sub-cell term on), the plane path against the fused kernel */
        enum { BW = 256, BC = 64 }; const size_t b3 = (size_t)BW * BW * BW, bcv = (size_t)BC * BC * BC;
        uint8_t *bct = malloc(b3); uint16_t *blg = malloc(COUT * bcv * 2);
        for (size_t i = 0; i < b3; i++) bct[i] = rnd(&s) < -0.3 ? 0 : (uint8_t)(128 + 250 * rnd(&s));
        for (size_t i = 0; i < COUT * bcv; i++) blg[i] = f2h((float)(3 * rnd(&s)));
        uint8_t *bctd = nn_malloc(b3); void *blgd = nn_malloc(COUT * bcv * 2), *bxd = nn_malloc(b3 * 2), *bpl = nn_malloc(11 * b3 * 2);
        nn_h2d(bctd, bct, b3); nn_h2d(blgd, blg, COUT * bcv * 2);
        for (int FH = 8; FH >= 2; FH -= 6) {
        const int box[3] = {BW, BW, BW}, F[3] = {BW / FH, BW / FH, BW / FH}; const size_t nf = (size_t)F[0] * F[1] * F[2];
        float *a = nn_malloc(12 * nf * 4), *b = nn_malloc(9 * nf * 4);
        double tp = 1e9, tf = 1e9, tu = 1e9, tst = 1e9, tdm = 1e9;
        for (int r = 0; r < 5; r++) {
            nn_sync(); double t0 = now_s();
            nn_lat_up_slab(blgd, 1, 11, BC, BC, BC, 0, BW, bpl); nn_lat_shuffle(blgd, 1, BC, BC, BC, NB, 2, 0, BW, bpl); nn_sync(); double t1 = now_s();
            nn_lat_ct_x(bctd, BW, 0, BW, mean, isd, bxd); nn_fstem_fwd(bxd, BW, BW, BW, 0, BW, pd, C, NO, chs, bpl, b3); nn_sync(); double t2 = now_s();
            nn_fit_accum_dm((const float *)bpl, bctd, nullptr, BW, BW, HALO, HALO, 0, 0, 0, box, FH, F, -1, 0, 2, 8, dscale, 0, a); nn_sync(); double t3 = now_s();
            nn_fit_accum_w4((const float *)bpl, bctd, nullptr, BW, BW, HALO, HALO, 0, 0, 0, box, FH, F, 9, DD, b); nn_sync(); double t4 = now_s();
            nn_fit_accum_lat(blgd, 1, BC, BC, BC, NB, pd, C, so, bctd, bctd, nullptr, mean, isd, HALO, HALO, 0, 0, 0, box, FH, F, -1, 0, 2, 8, dscale, 0, a, 9, DD, b); nn_sync(); double t5 = now_s();
            tu = fmin(tu, t1 - t0); tst = fmin(tst, t2 - t1); tdm = fmin(tdm, t3 - t2); tp = fmin(tp, t4 - t0); tf = fmin(tf, t5 - t4);
        }
        double tn = 1e9, tw = 1e9;   /* (the fused kernel without the stem; without the winding pass) */
        for (int r = 0; r < 5; r++) {
            nn_sync(); double t0 = now_s();
            nn_fit_accum_lat(blgd, 1, BC, BC, BC, NB, nullptr, C, so, bctd, bctd, nullptr, mean, isd, HALO, HALO, 0, 0, 0, box, FH, F, -1, 0, 2, 8, dscale, 0, a, 9, DD, b); nn_sync(); double t1 = now_s();
            nn_fit_accum_lat(blgd, 1, BC, BC, BC, NB, pd, C, so, bctd, bctd, nullptr, mean, isd, HALO, HALO, 0, 0, 0, box, FH, F, -1, 0, 2, 8, dscale, 0, a, -1, DD, b); nn_sync(); double t2 = now_s();
            tn = fmin(tn, t1 - t0); tw = fmin(tw, t2 - t1);
        }
        printf("bench cells %d: fused without the stem %.1f ms, without the winding pass %.1f ms\n", FH, 1e3 * tn, 1e3 * tw);
        printf("bench cells %d, %d^3 window: plane path %.1f ms (upsample + shuffle %.1f, stem %.1f, sums %.1f, winding %.1f), fused %.1f ms\n", FH, BW, 1e3 * tp, 1e3 * tu, 1e3 * tst, 1e3 * tdm,
               1e3 * (tp - tu - tst - tdm), 1e3 * tf);
        nn_free(a); nn_free(b); }
    }
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
