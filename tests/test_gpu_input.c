/* train --gpu-input: the network input built on the device from the CT channel and the radial records (nn_input_rad) must equal, byte
   for byte, the host path: sample.c write_x's four 16-bit channels converted by nn_h16_to_mx. Every cube symmetry (48), slabs
   (z-keeping symmetries), windows with and without an axis (incl. axis centres on voxel centres: a zero radial vector), large
   corners, fp16 / bf16 batches, MX-fp8 / MX-fp4 inputs, two samples per batch with different records.
   Also the scroll-axis record (sample_axis_record, the attention stage's unet_set_attn_axis input): the in-plane direction from its
   axis centre to every window voxel equals write_x's radial channels (fp16), for every symmetry, cubes and slabs. */
#include "nn.h"
#include "sample.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rs = 0x9e3779b97f4a7c15ull;
static uint64_t rn(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static float ru(void) { return (float)((rn() >> 11) * (1.0 / 9007199254740992.0)); }
#include <math.h>
static float h2f(uint16_t v) { const int e = (v >> 10) & 31, m = v & 1023; const float f = e == 0 ? ldexpf((float)m, -24) : ldexpf((float)(m | 1024), e - 25); return (v >> 15) ? -f : f; }
/* the record's radial direction against the window's radial channels t4[1..3] (fp16): max abs difference over voxels 2+ voxels off the axis */
static float axis_rec_err(const uint16_t *t4, int D, int P, const int64_t o[3], const float *cyz, const float *cxz, const int perm[3], const int flip[3]) {
    float r[4 + 2 * 64]; sample_axis_record(r, o, D, P, 1, perm, flip, cyz, cxz, 1);
    const int ad = (int)r[0], du = ad == 0 ? 1 : 0, dv = ad == 2 ? 1 : 2, ext[3] = {D, P, P}; const size_t S = (size_t)D * P * P; float e = 0;
    if (r[1] != 1.f || (int)r[2] != ext[ad]) return 1e9f;
    for (int z = 0; z < D; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
        const int w[3] = {z, y, x}; const float a = (float)w[du] - r[4 + 2 * w[ad]], b = (float)w[dv] - r[5 + 2 * w[ad]], n = sqrtf(a * a + b * b);
        if (n < 2.f) continue;
        const size_t k = ((size_t)z * P + y) * P + x; const float c[3] = {h2f(t4[S + k]), h2f(t4[2 * S + k]), h2f(t4[3 * S + k])};
        float d = fabsf(c[ad]); d = fmaxf(d, fabsf(c[du] - a / n)); d = fmaxf(d, fabsf(c[dv] - b / n)); e = fmaxf(e, d);
    }
    return e;
}

int main(void) {
    nn_init(0); nn_set_tf32(1); nn_set_act_bf16(1);
    int fails = 0, cases = 0, arcases = 0;
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
                            if (xfmt == 1 && bits == 8 && ax) {   /* the scroll-axis record */
                                const float e = axis_rec_err(t4, D, P, o, cyz, cxz, perm, flip); arcases++;
                                if (!(e < 2e-3f)) { if (fails < 5) fprintf(stderr, "FAIL: axis record vs radial channels %.3g (shape %d sym %d)\n", e, shape, sym); fails++; }
                            }
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
    {   /* warped windows (rotation + elastic, zoom, affine): a smooth axis (slope 0.3 voxel / z) 60-120 voxels off the window; the record's
           in-plane direction against the transported radial channels: angle (the elastic field and the axis tilt bend it a little) */
        const int P = 40; const size_t S = (size_t)P * P * P;
        uint8_t *ct = malloc(S), *tg = malloc(NCH * S), *mk = malloc(S), *T = malloc(NCH * S), *M = malloc(S), *rec = malloc(NN_RAD_REC(P));
        uint16_t *t4 = malloc(4 * S * 2), *h1 = malloc(S * 2); float worst = 0; nn_set_f16(1);
        for (int sym = 0; sym < 48; sym += 5) for (int mode = 0; mode < 4; mode++) {
            int perm[3], flip[3]; for (int d = 0; d < 3; d++) { perm[d] = perms[sym % 6][d]; flip[d] = (sym / 6 >> d) & 1; }
            const int64_t o[3] = {(int64_t)(rn() % 4000), (int64_t)(rn() % 30000), (int64_t)(rn() % 30000)};
            const float ang = 6.2831853f * ru(), rad = 60.f + 60.f * ru(); float cyz[64], cxz[64];
            for (int z = 0; z < P; z++) { cyz[z] = (float)o[1] + 20.f + rad * sinf(ang) + 0.3f * z; cxz[z] = (float)o[2] + 20.f + rad * cosf(ang) - 0.2f * z; }
            for (size_t k = 0; k < S; k++) { ct[k] = (uint8_t)(rn() & 255); tg[k] = 0; tg[S + k] = 255; mk[k] = 1; }
            const float zoom = mode == 1 ? .8f : mode == 2 ? .6f : 1.f; const int affine = mode == 3; const uint64_t as = 500 + (uint64_t)(sym * 4 + mode);
            sample_write_spatial_test(ct, tg, mk, P, o, 87.5f, 1.f / 31.f, 1, cyz, cxz, perm, flip, 1, 7, as, zoom, affine, 1, t4, h1, rec, T, M);
            float r[4 + 2 * 64]; sample_axis_record_spatial_test(r, o, P, 1, perm, flip, cyz, cxz, as, zoom, affine);
            const int ad = (int)r[0], du = ad == 0 ? 1 : 0, dv = ad == 2 ? 1 : 2; float e = 0; arcases++;
            for (int z = 0; z < P; z += 3) for (int y = 0; y < P; y += 3) for (int x = 0; x < P; x += 3) {
                const int w[3] = {z, y, x}; const float a = (float)w[du] - r[4 + 2 * w[ad]], b = (float)w[dv] - r[5 + 2 * w[ad]];
                const size_t k = ((size_t)z * P + y) * P + x; const float cu = h2f(t4[(1 + du) * S + k]), cv = h2f(t4[(1 + dv) * S + k]);
                if (cu * cu + cv * cv < 0.25f) continue;   /* (outside the source cube: zero) */
                const float d = fabsf(atan2f(a * cv - b * cu, a * cu + b * cv)); e = fmaxf(e, d);
            }
            worst = fmaxf(worst, e);
            if (!(e < 0.2f)) { if (fails < 5) fprintf(stderr, "FAIL: warped axis record vs radial channels: %.3f rad (sym %d mode %d)\n", e, sym, mode); fails++; }
        }
        printf("test_gpu_input: warped axis records: worst in-plane angle %.3f rad\n", worst);
        free(ct); free(tg); free(mk); free(T); free(M); free(rec); free(t4); free(h1);
    }
    printf("test_gpu_input: %d scroll-axis records match the radial channels\n", arcases);
    printf("test_gpu_input: ok (%d cases bit-identical: symmetries x axis x fp16/bf16 x MX-fp8/fp4 x cube/slab/2 samples, plain and --half-res pack; %d of them warped: rotation + elastic, zoom in / out, affine)\n", cases, wcases);
    return 0;
}
