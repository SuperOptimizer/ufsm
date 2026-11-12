/* latvol (src/latvol.h): the host codec and the device kernels (src/nn_latvol.cu).
   1. Host round trip: random and structured index volumes -> lv_encode_chunk -> lv_decode_chunk are equal (C 1, 13, 29, 64; flat and
      fitted tables; air cells, extreme values, an all-air chunk omitted), and a damaged stream is refused, never read out of bounds.
   2. The transform: lv_forward's indices are those of the float projection (rint(<b, v> / step_raw)) up to rounding ties, lv_inverse of
      them is within the quantiser's bound of the source, and forward(inverse(q)) = q where no byte clamps (the store is a fixed point
      of its own reader).
   3. Device = host, byte for byte: nn_latvol_forward (indices), nn_latvol_decode (a window over several chunks, chunks at negative /
      overhanging positions, absent chunks, strides C + 1 and 30 of 64), nn_latvol_synth (the CT bytes); a corrupt stream raises the flag. */
#include "latvol.h"
#include "nn_latvol.h"
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rs = 0x9e3779b97f4a7c15ull;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 32); }
static float ru(void) { return (float)rnd() * (1.f / 4294967296.f); }
enum { NC = LV_CELLS * LV_CELLS * LV_CELLS };
#define CK(x) do { if ((x) != cudaSuccess) { fprintf(stderr, "cuda error %s:%d\n", __FILE__, __LINE__); return 1; } } while (0)

/* an orthonormal basis: row 0 the mean, the rest Gram-Schmidt of random rows */
static void make_basis(float *b) {
    for (int k = 0; k < 64; k++) b[k] = 0.125f;
    for (int c = 1; c < 64; c++) {
        double v[64]; for (int k = 0; k < 64; k++) v[k] = ru() - 0.5;
        for (int p = 0; p < c; p++) { double d = 0; for (int k = 0; k < 64; k++) d += v[k] * b[p * 64 + k]; for (int k = 0; k < 64; k++) v[k] -= d * b[p * 64 + k]; }
        double n = 0; for (int k = 0; k < 64; k++) n += v[k] * v[k]; n = sqrt(n);
        for (int k = 0; k < 64; k++) b[c * 64 + k] = (float)(v[k] / n);
    }
}
/* indices like a CT's: a smooth mean, Laplacian axes narrowing with c; kind 1: extremes; air: share of air cells */
static void make_q(int16_t *q, int C, int kind, float air) {
    for (int i = 0; i < NC; i++) {
        int16_t *o = q + (size_t)i * C;
        if (ru() < air || kind == 3) { memset(o, 0, (size_t)C * sizeof *o); continue; }
        if (kind == 1) { o[0] = (int16_t)(1 + rnd() % 32767); for (int c = 1; c < C; c++) o[c] = (int16_t)((int)(rnd() % 255) - 127); continue; }
        o[0] = (int16_t)(60 + (i % 37) + (int)(rnd() % 9));
        for (int c = 1; c < C; c++) { const float s = 12.f / (1.f + 0.4f * c); const float u = ru(); const int m = (int)(-s * logf(u + 1e-9f)); const int v = m > 127 ? 127 : m; o[c] = (int16_t)(rnd() & 1 ? -v : v); }
    }
}
/* a CT chunk: sheets (a sine along a tilted direction) + noise, an air half-space when air */
static void make_ct(uint8_t *ct, int n, int air) {
    for (int z = 0; z < n; z++) for (int y = 0; y < n; y++) for (int x = 0; x < n; x++) {
        const float v = 90.f + 50.f * sinf(0.21f * x + 0.13f * y + 0.05f * z) + 24.f * (ru() - 0.5f);
        ct[((size_t)z * n + y) * n + x] = air && x + y / 2 < n / 3 ? 0 : (uint8_t)(v < 1 ? 1 : v > 255 ? 255 : v);
    }
}

int main(void) {
    int fails = 0;
    float *bas = malloc(64 * 64 * sizeof *bas); make_basis(bas);
    uint8_t *enc = malloc(LV_CHUNK_BOUND); int16_t *q = malloc((size_t)NC * 64 * 2), *q2 = malloc((size_t)NC * 64 * 2);
    /* ---- 1. host round trip ---- */
    static const int Cs[4] = {1, 13, 29, 64};
    for (int ci = 0; ci < 4; ci++) for (int kind = 0; kind < 4; kind++) for (int fit = 0; fit < 2; fit++) {
        const int C = Cs[ci]; make_q(q, C, kind, kind == 2 ? 0.6f : 0.02f);
        lv_codec lc; uint64_t (*hist)[LV_NSYM] = calloc(LV_MAXC, sizeof *hist);
        if (lv_codec_init(&lc, C, bas, 0.25f, 40.f, nullptr, 0, nullptr)) { printf("FAIL codec init\n"); return 1; }
        if (fit) { lv_hist(&lc, q, hist); lv_codec_free(&lc); if (lv_codec_init(&lc, C, bas, 0.25f, 40.f, nullptr, 0, hist)) { printf("FAIL fitted codec\n"); return 1; } }
        size_t len = 0;
        if (lv_encode_chunk(&lc, q, enc, &len)) { printf("FAIL encode\n"); fails++; }
        if (kind == 3) { if (len) { printf("FAIL: an all-air chunk has %zu bytes\n", len); fails++; } }
        else {
            memset(q2, 0x55, (size_t)NC * C * 2);
            if (lv_decode_chunk(&lc, enc, len, q2, nullptr) || memcmp(q, q2, (size_t)NC * C * 2)) { printf("FAIL: round trip C %d kind %d fit %d\n", C, kind, fit); fails++; }
            if (fit && kind == 0 && C == 29) {   /* the coded size against the ideal code length of the same tables' histograms */
                double bits = 0; for (int t = 0; t < lc.ntab; t++) { double tot = 0; for (int s = 0; s < LV_NSYM; s++) tot += (double)hist[t][s]; for (int s = 0; s < LV_NSYM; s++) if (hist[t][s]) bits -= (double)hist[t][s] * log2((double)hist[t][s] / tot); }
                double raw = 0; for (int i = 0; i < NC; i++) for (int c = 1; c < C; c++) { const int a = abs(q[(size_t)i * C + c]); raw += (a != 0) + (a >= 4 ? 31 - __builtin_clz((unsigned)a) : 0); }
                printf("latvol: C 29 chunk %zu bytes; tokens' entropy + raw bits %.0f bytes (+ the mean's raw bits)\n", len, (bits + raw) / 8);
                if ((double)len > 1.06 * (bits + raw) / 8 + 3 * NC / 8.0) { printf("FAIL: coded size far above the entropy\n"); fails++; }
            }
            /* damage: truncation, a flipped bit in a substream, a wrong C */
            if (!lv_decode_chunk(&lc, enc, len - 4, q2, nullptr)) { printf("FAIL: a truncated stream decoded\n"); fails++; }
            int caught = 0, tried = 0;
            for (int t = 0; t < 40; t++) { const size_t p = LV_HDR + rnd() % (len - LV_HDR); const uint8_t m = (uint8_t)(1u << (rnd() & 7)); enc[p] ^= m; tried++;
                                           caught += lv_decode_chunk(&lc, enc, len, q2, nullptr) || memcmp(q, q2, (size_t)NC * C * 2); enc[p] ^= m; }
            if (caught != tried) { printf("FAIL: %d of %d damaged streams decoded to the same indices\n", tried - caught, tried); fails++; }
        }
        lv_codec_free(&lc); free(hist);
    }
    /* the side file round trip */
    {
        lv_codec a, b; uint64_t (*hist)[LV_NSYM] = calloc(LV_MAXC, sizeof *hist); make_q(q, 29, 0, 0.1f);
        lv_codec_init(&a, 29, bas, 0.25f, 39.5f, nullptr, 0, nullptr); lv_hist(&a, q, hist); lv_codec_free(&a); lv_codec_init(&a, 29, bas, 0.25f, 39.5f, nullptr, 0, hist);
        char path[] = "/tmp/test_latvol_XXXXXX"; const int fd = mkstemp(path); if (fd < 0 || lv_codec_write(&a, path)) { printf("FAIL: side file write\n"); return 1; }
        FILE *f = fopen(path, "rb"); static uint8_t sb[1 << 16]; const size_t sn = fread(sb, 1, sizeof sb, f); fclose(f); remove(path);
        if (lv_codec_parse(&b, sb, sn) || b.C != 29 || b.step_raw != a.step_raw || memcmp(a.freq, b.freq, sizeof a.freq) || memcmp(a.bi, b.bi, sizeof a.bi) || memcmp(a.dt, b.dt, (size_t)a.ntab * LV_L * 4)) { printf("FAIL: side file round trip\n"); fails++; }
        if (!lv_codec_parse(&b, sb, 40)) { printf("FAIL: a short side file parsed\n"); fails++; }
        lv_codec_free(&a); lv_codec_free(&b); free(hist);
    }
    /* ---- 2. the transform ---- */
    uint8_t *ct = malloc(128 * 128 * 128), *ct2 = malloc(128 * 128 * 128);
    for (int ci = 1; ci < 4; ci++) {
        const int C = Cs[ci]; lv_codec lc; lv_codec_init(&lc, C, bas, 0.25f, 24.f, nullptr, 0, nullptr);
        make_ct(ct, 128, 1);
        const int nmat = lv_forward(&lc, ct, q);
        long diff = 0, tie = 0; double se = 0, sn = 0;
        for (int i = 0; i < NC; i++) {
            const int z = i / 1024, y = i / 32 % 32, x = i % 32; float v[64]; int any = 0;
            for (int k = 0; k < 64; k++) { v[k] = ct[((size_t)(4 * z + (k >> 4)) * 128 + 4 * y + (k >> 2 & 3)) * 128 + 4 * x + (k & 3)]; any |= v[k] != 0; }
            for (int c = 0; c < C; c++) {
                double s = 0; for (int k = 0; k < 64; k++) s += (double)bas[c * 64 + k] * v[k];
                const double f = s / lc.step_raw; int want = any ? (int)lrint(f) : 0; if (c && want > 127) want = 127; if (c && want < -127) want = -127; if (!c && any && want < 1) want = 1;
                if (want != q[(size_t)i * C + c]) { if (fabs(fabs(f - floor(f)) - 0.5) < 0.03) tie++; else diff++; }
            }
        }
        lv_inverse(&lc, q, ct2);
        long air_bad = 0; for (size_t i = 0; i < 128 * 128 * 128; i++) { const int cz = (int)(i / 16384) / 4, cy = (int)(i / 128 % 128) / 4, cx = (int)(i % 128) / 4; const int mat = q[(size_t)((cz * 32 + cy) * 32 + cx) * C] != 0;
            if (mat) { const double e = (double)ct2[i] - ct[i]; se += e * e; sn++; } else if (ct2[i] || ct[i]) air_bad++; }
        const double rms = sqrt(se / sn), bound = C == 64 ? lc.step_raw / sqrt(12.0) + 0.6 : 40.0;
        lv_forward(&lc, ct2, q2);
        long refix = 0, clampv = 0; for (size_t i = 0; i < 128 * 128 * 128; i++) clampv += ct2[i] == 255 || ct2[i] == 1;
        for (size_t i = 0; i < (size_t)NC * C; i++) refix += q[i] != q2[i];
        printf("latvol: C %2d: %d material cells, indices != float projection %ld (+ %ld ties), rebuilt rms error %.2f grey levels (step %.1f), re-encoded indices differing %ld of %zu (%ld clamped bytes)\n",
               C, nmat, diff, tie, rms, lc.step_raw, refix, (size_t)NC * C, clampv);
        if (diff || air_bad || rms > bound || refix > (long)((size_t)NC * C / 10)) { printf("FAIL: transform C %d (diff %ld air %ld rms %.2f refix %ld)\n", C, diff, air_bad, rms, refix); fails++; }
        lv_codec_free(&lc);
    }
    /* ---- 3. device = host ---- */
    {
        int ndev = 0; if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { printf("test_latvol: no GPU\n"); return 77; }
        for (int ci = 2; ci < 4; ci++) {
            const int C = Cs[ci], G = 3;   /* a 3^3 chunk grid = 384^3 voxels, one chunk absent, one all air */
            lv_codec lc; uint64_t (*hist)[LV_NSYM] = calloc(LV_MAXC, sizeof *hist);
            lv_codec_init(&lc, C, bas, 0.25f, 24.f, nullptr, 0, nullptr);
            const int V = 128 * G; uint8_t *vol = calloc((size_t)V * V * V, 1), *chunk = malloc(128 * 128 * 128);
            for (int g = 0; g < G * G * G; g++) {
                if (g == 5) continue;
                make_ct(chunk, 128, g % 3 == 1); const int gz = g / 9, gy = g / 3 % 3, gx = g % 3;
                for (int z = 0; z < 128; z++) for (int y = 0; y < 128; y++) memcpy(vol + ((size_t)(gz * 128 + z) * V + gy * 128 + y) * V + gx * 128, chunk + ((size_t)z * 128 + y) * 128, 128);
            }
            /* forward: device vs host */
            const int cd = V / 4; const size_t ncell = (size_t)cd * cd * cd;
            int16_t *qh = malloc(ncell * C * 2), *qg = malloc(ncell * C * 2);
            for (int g = 0; g < G * G * G; g++) {
                const int gz = g / 9, gy = g / 3 % 3, gx = g % 3;
                for (int z = 0; z < 128; z++) for (int y = 0; y < 128; y++) memcpy(chunk + ((size_t)z * 128 + y) * 128, vol + ((size_t)(gz * 128 + z) * V + gy * 128 + y) * V + gx * 128, 128);
                lv_forward(&lc, chunk, q); lv_hist(&lc, q, hist);
                for (int i = 0; i < NC; i++) { const int z = gz * 32 + i / 1024, y = gy * 32 + i / 32 % 32, x = gx * 32 + i % 32; memcpy(qh + (((size_t)z * cd + y) * cd + x) * C, q + (size_t)i * C, (size_t)C * 2); }
            }
            lv_codec_free(&lc); lv_codec_init(&lc, C, bas, 0.25f, 24.f, nullptr, 0, hist);
            nn_lv *L = nn_latvol_create(lc.dt, lc.ntab, lc.map, &lc.bi[0][0], C, lc.step_raw);
            uint8_t *d_vol; int16_t *d_q; CK(cudaMalloc((void **)&d_vol, (size_t)V * V * V)); CK(cudaMalloc((void **)&d_q, ncell * C * 2));
            CK(cudaMemcpy(d_vol, vol, (size_t)V * V * V, cudaMemcpyHostToDevice));
            nn_latvol_forward(L, d_vol, cd, cd, cd, d_q); CK(cudaMemcpy(qg, d_q, ncell * C * 2, cudaMemcpyDeviceToHost));
            if (memcmp(qh, qg, ncell * C * 2)) { size_t nd = 0; for (size_t i = 0; i < ncell * C; i++) nd += qh[i] != qg[i]; printf("FAIL: device forward C %d: %zu of %zu indices differ\n", C, nd, ncell * C); fails++; }
            /* encode every chunk on the host into one buffer */
            uint8_t *comp = malloc((size_t)G * G * G * LV_CHUNK_BOUND); size_t used = 0; lv_chunk_ref refs[27]; int nref = 0; size_t clen[27], coff[27];
            for (int g = 0; g < G * G * G; g++) {
                const int gz = g / 9, gy = g / 3 % 3, gx = g % 3;
                for (int i = 0; i < NC; i++) { const int z = gz * 32 + i / 1024, y = gy * 32 + i / 32 % 32, x = gx * 32 + i % 32; memcpy(q + (size_t)i * C, qh + (((size_t)z * cd + y) * cd + x) * C, (size_t)C * 2); }
                size_t len; lv_encode_chunk(&lc, q, enc, &len); clen[g] = len; coff[g] = used;
                if (!len) continue;
                memcpy(comp + used, enc, len); used += (len + 3) & ~(size_t)3;
            }
            if (clen[5]) { printf("FAIL: the absent chunk encoded to %zu bytes\n", clen[5]); fails++; }
            uint8_t *d_comp; CK(cudaMalloc((void **)&d_comp, used + 16)); CK(cudaMemcpy(d_comp, comp, used, cudaMemcpyHostToDevice));
            /* windows: the whole grid; an interior window off the chunk grid (cell origin 4-aligned: substream layers); one overhanging the volume */
            static const int wins[4][6] = {{0, 0, 0, 96, 96, 96}, {28, 20, 12, 40, 52, 60}, {-8, 60, 4, 64, 64, 64}, {33, 2, 31, 30, 61, 37}};
            for (int wi = 0; wi < 4; wi++) for (int st = 0; st < 2; st++) {
                const int *wb = wins[wi]; const int d = wb[3], h = wb[4], w = wb[5], stride = st ? 30 : C + 1; const size_t wc = (size_t)d * h * w;
                nref = 0; for (int g = 0; g < G * G * G; g++) if (clen[g]) refs[nref++] = (lv_chunk_ref){(uint32_t)coff[g], (uint32_t)clen[g], (g / 9) * 32 - wb[0], (g / 3 % 3) * 32 - wb[1], (g % 3) * 32 - wb[2]};
                int8_t *want = calloc(wc, (size_t)stride), *got = malloc(wc * stride); int8_t *d_cells; CK(cudaMalloc((void **)&d_cells, wc * stride));
                for (int z = 0; z < d; z++) for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
                    const int gz = z + wb[0], gy = y + wb[1], gx = x + wb[2]; if (gz < 0 || gy < 0 || gx < 0 || gz >= cd || gy >= cd || gx >= cd) continue;
                    lv_pack_cells(qh + (((size_t)gz * cd + gy) * cd + gx) * C, 1, C, want + (((size_t)z * h + y) * w + x) * stride, stride);
                }
                const int flag = nn_latvol_decode(L, d_comp, refs, nref, d, h, w, stride, d_cells); CK(cudaMemcpy(got, d_cells, wc * stride, cudaMemcpyDeviceToHost));
                if (flag || memcmp(want, got, wc * stride)) { size_t nd = 0; for (size_t i = 0; i < wc * stride; i++) nd += want[i] != got[i]; printf("FAIL: device decode C %d window %d stride %d: flag %d, %zu of %zu bytes differ\n", C, wi, stride, flag, nd, wc * stride); fails++; }
                if (!st && wi < 2) {   /* the CT bytes: device synth of the device cells vs the host inverse */
                    const size_t wv = wc * 64; uint8_t *cg = malloc(wv), *chh = malloc(wv), *d_ct; CK(cudaMalloc((void **)&d_ct, wv));
                    nn_latvol_synth(L, d_cells, stride, d, h, w, d_ct); CK(cudaMemcpy(cg, d_ct, wv, cudaMemcpyDeviceToHost));
                    uint8_t *full = calloc((size_t)V * V * V, 1);
                    for (int g = 0; g < G * G * G; g++) { if (!clen[g]) continue; const int gz = g / 9, gy = g / 3 % 3, gx = g % 3;
                        if (lv_decode_voxels(&lc, comp + coff[g], clen[g], chunk, nullptr)) { printf("FAIL: host voxel decode\n"); fails++; }
                        for (int z = 0; z < 128; z++) for (int y = 0; y < 128; y++) memcpy(full + ((size_t)(gz * 128 + z) * V + gy * 128 + y) * V + gx * 128, chunk + ((size_t)z * 128 + y) * 128, 128); }
                    for (int z = 0; z < 4 * d; z++) for (int y = 0; y < 4 * h; y++) memcpy(chh + ((size_t)z * 4 * h + y) * 4 * w, full + ((size_t)(z + 4 * wb[0]) * V + y + 4 * wb[1]) * V + 4 * wb[2], (size_t)4 * w);
                    if (memcmp(cg, chh, wv)) { size_t nd = 0; for (size_t i = 0; i < wv; i++) nd += cg[i] != chh[i]; printf("FAIL: device synth C %d window %d: %zu of %zu bytes differ\n", C, wi, nd, wv); fails++; }
                    free(cg); free(chh); free(full); cudaFree(d_ct);
                }
                free(want); free(got); cudaFree(d_cells);
            }
            /* a corrupt stream raises the flag */
            { uint8_t sv; const size_t p = coff[0] + LV_HDR + 5; CK(cudaMemcpy(&sv, d_comp + p, 1, cudaMemcpyDeviceToHost)); const uint8_t bad = sv ^ 0x10; CK(cudaMemcpy(d_comp + p, &bad, 1, cudaMemcpyHostToDevice));
              int8_t *d_cells; CK(cudaMalloc((void **)&d_cells, (size_t)32 * 32 * 32 * (C + 1))); nref = 1; refs[0] = (lv_chunk_ref){(uint32_t)coff[0], (uint32_t)clen[0], 0, 0, 0};
              memcpy(enc, comp + coff[0], clen[0]); enc[LV_HDR + 5] ^= 0x10; const int hostbad = lv_decode_chunk(&lc, enc, clen[0], q, nullptr) != 0;
              const int flag = nn_latvol_decode(L, d_comp, refs, 1, 32, 32, 32, C + 1, d_cells);
              if (hostbad && !flag) { printf("FAIL: the device decoded a stream the host refuses\n"); fails++; }
              cudaFree(d_cells); }
            printf("latvol: C %d device forward / decode (4 windows x 2 strides) / synth = host; %zu bytes for %d^3 voxels (%.3f B/vox)\n", C, used, V, (double)used / ((double)V * V * V));
            nn_latvol_free(L); cudaFree(d_vol); cudaFree(d_q); cudaFree(d_comp); free(comp); free(qh); free(qg); free(vol); free(chunk); free(hist); lv_codec_free(&lc);
        }
    }
    printf(fails ? "test_latvol: %d FAILED\n" : "test_latvol: all passed\n", fails);
    return fails != 0;
}
