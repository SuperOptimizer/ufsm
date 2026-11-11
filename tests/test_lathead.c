/* nn_lat_head_place (predict --heads) against the slab composition it replaces: nn_lat_up_slab (+ nn_lat_shuffle) + nn_fstem_fwd on the
   z-scored CT planes (nn_lat_ct_x) + (nn_rv_fuse fp16) + nn_pred_place, on a 160^3 window (40^3 cells, cout 144 = 16 + 128 sub-cell
   outputs), halo 8, into a shard at an offset. Heads: the rv pair 11-12 (stem rows 4, 5), the phase channel 0 (sub-cell + stem row 0), a
   plain channel 3 (no stem); with and without the stem. Pass: at most 0.1 % of the bytes differ, none by more than 1 (the stem's fp32 sums
   in another order than the tensor-core kernel's). Usage: test_lathead */
#include "nn.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
enum { WW = 160, CW = 40, NB = 16, COUT = 144, HALO = 8, SH = 160, C = 16, NO = 6 };   /* (W / 2 = 80 stem blocks a row: past one 32-block warp pass) */
static double rnd(uint64_t *s) { *s = *s * 6364136223846793005ull + 1442695040888963407ull; return (double)(*s >> 11) * 0x1p-53 - 0.5; }
static uint16_t f2h(float v) { _Float16 f = (_Float16)v; uint16_t h; memcpy(&h, &f, 2); return h; }
int main(void) {
    nn_init(0); nn_set_f16(1); int fails = 0; uint64_t s = 11;
    const size_t w3 = (size_t)WW * WW * WW, cv = (size_t)CW * CW * CW, sh3 = (size_t)SH * SH * SH, NP = C * 64 + C + NO * C;
    uint8_t *ct = malloc(w3); uint16_t *lg = malloc(COUT * cv * 2); float *lgf = malloc(COUT * cv * 4); float *pf = malloc(NP * 4);
    for (size_t i = 0; i < w3; i++) ct[i] = rnd(&s) < -0.3 ? 0 : (uint8_t)(128 + 250 * rnd(&s));
    for (size_t i = 0; i < COUT * cv; i++) { lg[i] = f2h((float)(3 * rnd(&s))); _Float16 f; memcpy(&f, lg + i, 2); lgf[i] = (float)f; }
    for (size_t i = 0; i < NP; i++) pf[i] = (float)(0.3 * rnd(&s));
    const float mean = 100.f, isd = 1.f / 40.f; const int iso = 1;
    uint8_t *ctd = nn_malloc(w3), *d0 = nn_malloc(sh3), *d1 = nn_malloc(sh3); void *lgd = nn_malloc(COUT * cv * 2), *lgd32 = nn_malloc(COUT * cv * 4), *xd = nn_malloc(w3 * 2), *pl = nn_malloc(2 * w3 * 2); float *pd = nn_malloc(NP * 4);
    nn_h2d(ctd, ct, w3); nn_h2d(lgd, lg, COUT * cv * 2); nn_h2d(lgd32, lgf, COUT * cv * 4); nn_h2d(pd, pf, NP * 4);
    nn_lat_ct_x(ctd, WW, 0, WW, mean, isd, xd);
    uint8_t *h0 = malloc(sh3), *h1 = malloc(sh3);
    static const struct { const char *name; int c0, c1, sb0, sb1, s0, s1; } T[3] = {{"rv 11+12", 11, 12, -1, -1, 4, 5}, {"phase 0", 0, -1, 16, -1, 0, -1}, {"plain 3", 3, -1, -1, -1, -1, -1}};
    const int oz = 20, oy = -10, ox = 3;   /* window voxel v -> shard voxel o + v */
    for (int l32 = 0; l32 < 2; l32++) for (int st = 0; st < 2; st++) for (int t = 0; t < 3; t++) {   /* (l32: the cell logits in fp32, the same values) */
        const void *L = l32 ? lgd32 : lgd; const int lh = !l32, es = l32 ? 4 : 2;
        const int np = T[t].c1 >= 0 ? 2 : 1;
        /* the composition: both channels' planes [np][W^3] fp16, slab by slab */
        for (int z0 = 0; z0 < WW; z0 += 32) {
            const size_t ns = (size_t)32 * WW * WW; void *sl = nn_malloc(np * ns * 2);
            for (int c = 0; c < np; c++) nn_lat_up_slab((const char *)L + (size_t)(c ? T[t].c1 : T[t].c0) * cv * es, lh, 1, CW, CW, CW, z0, 32, (char *)sl + c * ns * 2);
            if (T[t].sb0 >= 0) nn_lat_shuffle(L, lh, CW, CW, CW, NB, 1, z0, 32, sl);   /* (channel 0 only: ng 1) */
            if (st && T[t].s0 >= 0) { int chs[NO]; for (int o = 0; o < NO; o++) chs[o] = o == T[t].s0 ? 0 : o == T[t].s1 ? 1 : -1;
                                       if (nn_fstem_fwd(xd, WW, WW, WW, z0, 32, pd, C, NO, chs, sl, ns)) { fprintf(stderr, "fstem shape\n"); return 1; } }
            for (int c = 0; c < np; c++) nn_d2d((char *)pl + ((size_t)c * w3 + (size_t)z0 * WW * WW) * 2, (char *)sl + c * ns * 2, ns * 2);
            nn_free(sl);
        }
        if (np == 2) nn_rv_fuse(pl, w3, 1);
        nn_zero(d0, sh3); nn_zero(d1, sh3);
        nn_set_logits_h16(1); nn_set_pred_iso(iso);
        nn_pred_place_d((const float *)pl, ctd, WW, WW, HALO, HALO, oz, oy, ox, SH, SH, SH, SH, d0);
        nn_lat_head_place(L, lh, CW, CW, CW, T[t].c0, T[t].c1, T[t].sb0, T[t].sb1, T[t].s0, T[t].s1, st ? pd : nullptr, C, NO, ctd, mean, isd, iso,
                          HALO, HALO, oz, oy, ox, SH, SH, SH, SH, d1);
        nn_sync(); nn_d2h(h0, d0, sh3); nn_d2h(h1, d1, sh3);
        size_t nd = 0, n2 = 0, nset = 0; for (size_t i = 0; i < sh3; i++) { const int e = abs((int)h0[i] - (int)h1[i]); nd += e != 0; n2 += e > 1; nset += h0[i] != 0; }
        const int bad = n2 > 0 || nd > sh3 / 1000 || nset < sh3 / 16;
        printf("%s %-9s logits %s stem %d: %zu of %zu bytes differ (%zu by more than 1), %zu set%s\n", bad ? "FAIL" : "ok  ", T[t].name, l32 ? "fp32" : "fp16", st, nd, sh3, n2, nset, bad ? "  <--" : "");
        fails += bad;
    }
    printf("%s\n", fails ? "FAIL" : "test_lathead: ok");
    return fails != 0;
}
