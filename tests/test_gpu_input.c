/* train --gpu-input: the network input built on the device from the CT channel and the radial records (nn_input_rad) must equal, byte
   for byte, the host path: sample.c write_x's four 16-bit channels converted by nn_h16_to_mx. Every cube symmetry (48), slabs
   (z-keeping symmetries), windows with and without an axis (incl. axis centres on voxel centres: a zero radial vector), large
   corners, fp16 / bf16 batches, MX-fp8 / MX-fp4 inputs, two samples per batch with different records. */
#include "nn.h"
#include "sample.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rs = 0x9e3779b97f4a7c15ull;
static uint64_t rn(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static float ru(void) { return (float)((rn() >> 11) * (1.0 / 9007199254740992.0)); }

int main(void) {
    nn_init(0); nn_set_tf32(1); nn_set_act_bf16(1);
    int fails = 0, cases = 0;
    static const int perms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
    for (int xfmt = 1; xfmt <= 2; xfmt++) {
        nn_set_f16(xfmt == 1);
        for (int bits = 8; bits >= 4; bits -= 4)
            for (int shape = 0; shape < 3; shape++) {
                const int D = shape == 1 ? 16 : 40, P = 40, N = shape == 2 ? 2 : 1;   /* 0: cube, 1: slab, 2: two cube samples */
                const size_t S = (size_t)D * P * P, rec = NN_RAD_REC(D);
                const shape5 xs = {N, 4, D, P, P};
                const size_t mxb = nn_mx_bytes(xs, bits);
                uint8_t *ct = malloc(S);
                uint16_t *h4 = malloc((size_t)N * 4 * S * 2), *h1 = malloc((size_t)N * S * 2), *t4 = malloc(4 * S * 2);
                uint8_t *recs = malloc((size_t)N * rec), *ya = malloc(mxb), *yb = malloc(mxb);
                void *d_h4 = nn_malloc((size_t)N * 4 * S * 2), *d_h1 = nn_malloc((size_t)N * S * 2), *d_rec = nn_malloc((size_t)N * rec), *d_ya = nn_malloc(mxb), *d_yb = nn_malloc(mxb);
                nn_set_storage(d_ya, mxb, bits); nn_set_storage(d_yb, mxb, bits);
                for (int sym = 0; sym < 48; sym++) {
                    const int pi = sym % 6, fl = sym / 6;
                    if (shape == 1 && perms[pi][0] != 0) continue;   /* slabs: z-keeping symmetries */
                    for (int ax = 0; ax < 2; ax++) {
                        for (int n = 0; n < N; n++) {
                            int perm[3], flip[3];
                            const int pj = n ? (pi + 3) % 6 : pi, fj = n ? (fl ^ 5) : fl;   /* the second sample: another symmetry */
                            for (int d = 0; d < 3; d++) { perm[d] = perms[shape == 1 ? pi : pj][d]; flip[d] = ((shape == 1 ? fl : fj) >> d) & 1; }
                            const int64_t o[3] = {(int64_t)(rn() % 4000), (int64_t)(rn() % 30000), (int64_t)(rn() % 30000)};
                            float cyz[64], cxz[64];
                            for (int z = 0; z < D; z++) {
                                if (z % 7 == 3) { cyz[z] = (float)(o[1] + (int64_t)(rn() % P)); cxz[z] = (float)(o[2] + (int64_t)(rn() % P)); }   /* on a voxel: radial 0 */
                                else { cyz[z] = (float)o[1] + 200.f * (ru() - 0.5f) + 0.37f; cxz[z] = (float)o[2] + 200.f * (ru() - 0.5f) - 0.11f; }
                            }
                            for (size_t k = 0; k < S; k++) ct[k] = (uint8_t)(rn() & 255);
                            sample_write_x_test(ct, D, P, o, 87.5f, 1.f / 31.f, ax, cyz, cxz, perm, flip, xfmt, 1234 + (uint64_t)sym * 17 + (uint64_t)n,
                                                t4, h1 + (size_t)n * S, recs + (size_t)n * rec);
                            for (int c = 0; c < 4; c++) memcpy(h4 + ((size_t)n * 4 + c) * S, t4 + (size_t)c * S, S * 2);
                            if (memcmp(h1 + (size_t)n * S, t4, S * 2)) { if (fails < 5) fprintf(stderr, "FAIL: the CT channel differs (xfmt %d sym %d ax %d)\n", xfmt, sym, ax); fails++; }
                        }
                        nn_h2d(d_h4, h4, (size_t)N * 4 * S * 2); nn_h2d(d_h1, h1, (size_t)N * S * 2); nn_h2d(d_rec, recs, (size_t)N * rec);
                        nn_h16_to_mx(d_h4, xs, d_ya);
                        nn_input_rad(d_h1, xs, d_rec, d_yb);
                        nn_d2h(ya, d_ya, mxb); nn_d2h(yb, d_yb, mxb);
                        cases++;
                        if (memcmp(ya, yb, mxb)) {
                            size_t k = 0; while (k < mxb && ya[k] == yb[k]) k++;
                            if (fails < 5) fprintf(stderr, "FAIL: device input differs (xfmt %d MX-fp%d shape %d sym %d axis %d): first byte %zu of %zu (%02x vs %02x)\n",
                                                   xfmt, bits, shape, sym, ax, k, mxb, ya[k], yb[k]);
                            fails++;
                        }
                        {   /* --half-res pack: the 8 input channels on the 2x2x2 blocks (nn_input_rad_pack) against the host chain
                               write_x -> nn_half_input(mode 2) -> nn_h16_to_mx */
                            const shape5 hs = {N, 8, D / 2, P / 2, P / 2};
                            const size_t hb = nn_mx_bytes(hs, bits), h8 = (size_t)N * 8 * shape_spatial(hs) * 2;
                            void *d_h8 = nn_malloc(h8), *d_pa = nn_malloc(hb), *d_pb = nn_malloc(hb);
                            nn_set_storage(d_pa, hb, bits); nn_set_storage(d_pb, hb, bits);
                            nn_half_input(d_h4, xs, 2, d_h8, 1);
                            nn_h16_to_mx(d_h8, hs, d_pa);
                            nn_input_rad_pack(d_h1, hs, d_rec, d_pb);
                            uint8_t *pa = malloc(hb), *pb = malloc(hb);
                            nn_d2h(pa, d_pa, hb); nn_d2h(pb, d_pb, hb);
                            cases++;
                            if (memcmp(pa, pb, hb)) {
                                size_t k = 0; while (k < hb && pa[k] == pb[k]) k++;
                                if (fails < 5) fprintf(stderr, "FAIL: packed device input differs (xfmt %d MX-fp%d shape %d sym %d axis %d): first byte %zu of %zu (%02x vs %02x)\n",
                                                       xfmt, bits, shape, sym, ax, k, hb, pa[k], pb[k]);
                                fails++;
                            }
                            free(pa); free(pb); nn_free(d_h8); nn_free(d_pa); nn_free(d_pb);
                        }
                    }
                }
                const char *e = nn_check(); if (e) { fprintf(stderr, "FAIL: CUDA error %s\n", e); fails++; }
                nn_free(d_h4); nn_free(d_h1); nn_free(d_rec); nn_free(d_ya); nn_free(d_yb);
                free(ct); free(h4); free(h1); free(t4); free(recs); free(ya); free(yb);
            }
    }
    /* warped windows (write_spatial: rotation + elastic, zoom in / out, affine): the transported radial vector built on the device from
       the warp in the record, and the CT-only writer's channel 0 / targets / mask equal to the 4-channel writer's; two samples per batch,
       the second unwarped (zoom 1, no rotation would still draw an elastic field: zoom 0 = the plain write_x record of a warped writer is
       not needed, the flag is per sample) */
    int wcases = 0;
    for (int xfmt = 1; xfmt <= 2; xfmt++) {
        nn_set_f16(xfmt == 1);
        for (int bits = 8; bits >= 4; bits -= 4) {
            const int P = 40, N = 2; const size_t S = (size_t)P * P * P, rec = NN_RAD_REC(P);
            const shape5 xs = {N, 4, P, P, P};
            const size_t mxb = nn_mx_bytes(xs, bits);
            uint8_t *ct = malloc(S), *tg = malloc(NCH * S), *mk = malloc(S), *T = malloc(NCH * S), *M = malloc(S);
            uint16_t *h4 = malloc((size_t)N * 4 * S * 2), *h1 = malloc((size_t)N * S * 2), *t4 = malloc(4 * S * 2);
            uint8_t *recs = malloc((size_t)N * rec), *ya = malloc(mxb), *yb = malloc(mxb);
            void *d_h4 = nn_malloc((size_t)N * 4 * S * 2), *d_h1 = nn_malloc((size_t)N * S * 2), *d_rec = nn_malloc((size_t)N * rec), *d_ya = nn_malloc(mxb), *d_yb = nn_malloc(mxb);
            nn_set_storage(d_ya, mxb, bits); nn_set_storage(d_yb, mxb, bits);
            for (int sym = 0; sym < 48; sym += 3) for (int ax = 0; ax < 2; ax++) for (int mode = 0; mode < 4; mode++) {
                const float zoom = mode == 1 ? .8f : mode == 2 ? .6f : 1.f; const int affine = mode == 3;
                for (int n = 0; n < N; n++) {
                    int perm[3], flip[3];
                    const int pi = (sym + 2 * n) % 6, fl = (sym / 6) ^ (n ? 3 : 0);
                    for (int d = 0; d < 3; d++) { perm[d] = perms[pi][d]; flip[d] = (fl >> d) & 1; }
                    const int64_t o[3] = {(int64_t)(rn() % 4000), (int64_t)(rn() % 30000), (int64_t)(rn() % 30000)};
                    float cyz[64], cxz[64];
                    for (int z = 0; z < P; z++) { cyz[z] = (float)o[1] + 150.f * (ru() - 0.5f) + 0.37f; cxz[z] = (float)o[2] + 150.f * (ru() - 0.5f) - 0.11f; }
                    for (size_t k = 0; k < S; k++) { ct[k] = (uint8_t)(rn() & 255); tg[k] = rn() & 1 ? 255 : 0; tg[S + k] = tg[k] ? 0 : 255; mk[k] = (rn() & 7) != 0; }
                    if (n == 1) {   /* an unwarped sample beside the warped one */
                        sample_write_x_test(ct, P, P, o, 87.5f, 1.f / 31.f, ax, cyz, cxz, perm, flip, xfmt, 99 + (uint64_t)sym, t4, h1 + S, recs + rec);
                    } else {
                        const int act = sample_write_spatial_test(ct, tg, mk, P, o, 87.5f, 1.f / 31.f, ax, cyz, cxz, perm, flip, xfmt, 4321 + (uint64_t)sym,
                                                                  77 + (uint64_t)(sym * 4 + mode), zoom, affine, 1, t4, h1, recs, T, M);
                        if (act != 1) { if (fails < 5) fprintf(stderr, "FAIL: warped writer: CT-only targets / mask differ or the warp is inactive (%d)\n", act); fails++; }
                    }
                    for (int c = 0; c < 4; c++) memcpy(h4 + ((size_t)n * 4 + c) * S, t4 + (size_t)c * S, S * 2);
                    if (memcmp(h1 + (size_t)n * S, t4, S * 2)) { if (fails < 5) fprintf(stderr, "FAIL: warped CT channel differs (sym %d mode %d)\n", sym, mode); fails++; }
                }
                nn_h2d(d_h4, h4, (size_t)N * 4 * S * 2); nn_h2d(d_h1, h1, (size_t)N * S * 2); nn_h2d(d_rec, recs, (size_t)N * rec);
                nn_h16_to_mx(d_h4, xs, d_ya);
                nn_input_rad(d_h1, xs, d_rec, d_yb);
                nn_d2h(ya, d_ya, mxb); nn_d2h(yb, d_yb, mxb);
                wcases++;
                if (memcmp(ya, yb, mxb)) {
                    size_t k = 0; while (k < mxb && ya[k] == yb[k]) k++;
                    if (fails < 5) fprintf(stderr, "FAIL: warped device input differs (xfmt %d MX-fp%d sym %d axis %d mode %d): first byte %zu of %zu (%02x vs %02x)\n",
                                           xfmt, bits, sym, ax, mode, k, mxb, ya[k], yb[k]);
                    fails++;
                }
            }
            const char *e = nn_check(); if (e) { fprintf(stderr, "FAIL: CUDA error %s\n", e); fails++; }
            nn_free(d_h4); nn_free(d_h1); nn_free(d_rec); nn_free(d_ya); nn_free(d_yb);
            free(ct); free(tg); free(mk); free(T); free(M); free(h4); free(h1); free(t4); free(recs); free(ya); free(yb);
        }
    }
    cases += wcases;
    if (fails) { printf("test_gpu_input: %d of %d cases FAIL\n", fails, cases); return 1; }
    printf("test_gpu_input: ok (%d cases bit-identical: symmetries x axis x fp16/bf16 x MX-fp8/fp4 x cube/slab/2 samples, plain and --half-res pack; %d of them warped: rotation + elastic, zoom in / out, affine)\n", cases, wcases);
    return 0;
}
