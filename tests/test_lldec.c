/* test_lldec DIR: the volcomp lossless codes decode on the GPU (src/nn_band.cu k_ll_sub / k_ll_build, packed by src/band_onfly.c) against
   the CPU decode (z3_read), byte for byte: a 500 x 250 x 250 lossless array (shards 256, chunks 128; the edge chunks partial) whose z
   slabs exercise every block mode: sparse sheet labels (SPARSE + CONST 0), a smooth ramp with noise (DENSE), random bytes (RAW blocks
   and RAW chunks), a constant 7 slab (CONST chunks); boxes inside, across chunk and shard seams, and partly outside the array. Exit 0
   (PASS) or 1; 77 (SKIP) without a GPU. Then the pooled codes views (codes_pool2_open 1, 2, 3 deep): the device decode + k_codes_pool2
   (tile_gpu) against the CPU pooling (codes_pool2_box), the same boxes scaled down; and the count of voxels where the pooling's bin search
   differs from the atan2 rounding it replaced (2026-11-12; reported, near-ties only). */
#include "../src/zarr3.h"
#include "../src/z3w.h"
#include "../src/store.h"
#include "../src/band_onfly.h"
#include "../src/bandgpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../src/band.h"

static uint8_t val(int64_t z, int64_t y, int64_t x) {
    uint32_t h = (uint32_t)(z * 73856093u ^ y * 19349663u ^ x * 83492791u); h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    if (z < 128) { const int64_t r = (y - 125) * (y - 125) + (x - 125) * (x - 125); return r % 600 < 40 && (h & 7) ? (uint8_t)(1 + (z + y / 7) % 252) : (h % 997 == 0 ? 255 : 0); }
    if (z < 256) return (uint8_t)(x + 2 * y + (h & 3));
    if (z < 384) return (uint8_t)h;
    return 7;
}
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: test_lldec DIR\n"); return 2; }
    if (phase_onfly_config(0, 1, 512, 0)) { printf("SKIP: no GPU pair rule\n"); return 77; }
    char dir[1200]; snprintf(dir, sizeof dir, "%s/codes", argv[1]);
    const int64_t shape[3] = {500, 250, 250};
    z3w *w = z3w_create(dir, shape, 256, 0, 0, "{}"); if (!w) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    uint8_t *sh = malloc((size_t)256 * 256 * 256);
    for (int sz = 0; sz < 2; sz++) for (int sy = 0; sy < 1; sy++) for (int sx = 0; sx < 1; sx++) {
        for (int64_t z = 0; z < 256; z++) for (int64_t y = 0; y < 256; y++) for (int64_t x = 0; x < 256; x++) {
            const int64_t gz = sz * 256 + z, gy = sy * 256 + y, gx = sx * 256 + x;
            sh[((size_t)z * 256 + y) * 256 + x] = gz < shape[0] && gy < shape[1] && gx < shape[2] ? val(gz, gy, gx) : 0;
        }
        if (z3w_write_shard(w, sz, sy, sx, sh, 4)) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    }
    free(sh);
    if (z3w_close(w)) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    z3 *z = z3_open(store_open(argv[1]), "codes", nullptr); if (!z) { fprintf(stderr, "%s\n", z3_error()); return 1; }
    const int64_t box[][6] = {{0, 0, 0, 500, 250, 250}, {100, 60, 70, 200, 150, 120}, {-20, -30, 200, 90, 100, 90}, {380, 120, 0, 140, 140, 256}, {250, 0, 0, 20, 250, 250}};
    int fail = 0;
    for (int b = 0; b < (int)(sizeof box / sizeof box[0]); b++) {
        const int64_t o[3] = {box[b][0], box[b][1], box[b][2]}, n[3] = {box[b][3], box[b][4], box[b][5]};
        const size_t N = (size_t)n[0] * n[1] * n[2]; uint8_t *a = malloc(N), *g = band_pair_gpu_host_alloc ? band_pair_gpu_host_alloc(N) : malloc(N);
        memset(a, 0, N);
        /* the CPU reference: inside the array z3_read, outside 0 (the codes' fill) */
        int64_t ro[3], rn[3]; for (int d = 0; d < 3; d++) { ro[d] = o[d] < 0 ? 0 : o[d]; rn[d] = (o[d] + n[d] < shape[d] ? o[d] + n[d] : shape[d]) - ro[d]; }
        uint8_t *t = malloc((size_t)rn[0] * rn[1] * rn[2]);
        if (z3_read(z, ro, rn, t, 4)) { fprintf(stderr, "%s\n", z3_error()); return 1; }
        for (int64_t q = 0; q < rn[0]; q++) for (int64_t y = 0; y < rn[1]; y++) memcpy(a + ((size_t)(q + ro[0] - o[0]) * n[1] + (y + ro[1] - o[1])) * n[2] + (ro[2] - o[2]), t + ((size_t)q * rn[1] + y) * rn[2], (size_t)rn[2]);
        free(t);
        const int r = phase_onfly_decode_gpu(z, o, n, g);
        size_t d = 0; if (!r) for (size_t i = 0; i < N; i++) d += a[i] != g[i];
        printf("box %d (%lld,%lld,%lld + %lldx%lldx%lld): GPU rc %d, %zu of %zu bytes differ\n", b, (long long)o[0], (long long)o[1], (long long)o[2], (long long)n[0], (long long)n[1], (long long)n[2], r, d, N);
        if (r || d) fail = 1;
        free(a); if (band_pair_gpu_host_free) band_pair_gpu_host_free(g); else free(g);
    }
    z3_close(z);
    for (int k = 1; k <= 3; k++) {
        z3 *p = z3_open(store_open(argv[1]), "codes", nullptr); for (int q = 0; p && q < k; q++) p = codes_pool2_open(p);
        if (!p) { fprintf(stderr, "pool %d: %s\n", k, z3_error()); return 1; }
        const int64_t *ps = z3_meta_of(p)->shape;
        for (int b = 0; b < (int)(sizeof box / sizeof box[0]); b++) {
            const int64_t o[3] = {box[b][0] >> k, box[b][1] >> k, box[b][2] >> k}, n[3] = {(box[b][3] >> k) + 1, (box[b][4] >> k) + 1, (box[b][5] >> k) + 1};
            const size_t N = (size_t)n[0] * n[1] * n[2]; uint8_t *a = calloc(N, 1), *g = band_pair_gpu_host_alloc ? band_pair_gpu_host_alloc(N) : malloc(N);
            int64_t ro[3], rn[3]; for (int d = 0; d < 3; d++) { ro[d] = o[d] < 0 ? 0 : o[d]; rn[d] = (o[d] + n[d] < ps[d] ? o[d] + n[d] : ps[d]) - ro[d]; }
            uint8_t *t = malloc((size_t)rn[0] * rn[1] * rn[2]);
            if (z3_read(p, ro, rn, t, 4)) { fprintf(stderr, "%s\n", z3_error()); return 1; }
            for (int64_t q = 0; q < rn[0]; q++) for (int64_t y = 0; y < rn[1]; y++) memcpy(a + ((size_t)(q + ro[0] - o[0]) * n[1] + (y + ro[1] - o[1])) * n[2] + (ro[2] - o[2]), t + ((size_t)q * rn[1] + y) * rn[2], (size_t)rn[2]);
            free(t);
            const int r = phase_onfly_decode_gpu(p, o, n, g);
            size_t d = 0, nz = 0; if (!r) for (size_t i = 0; i < N; i++) { d += a[i] != g[i]; nz += a[i] != 0; }
            printf("pool %d box %d (%lld,%lld,%lld + %lldx%lldx%lld): GPU rc %d, %zu of %zu bytes differ (%zu non-empty)\n", k, b, (long long)o[0], (long long)o[1], (long long)o[2], (long long)n[0], (long long)n[1], (long long)n[2], r, d, N, nz);
            if (r || d) fail = 1;
            free(a); if (band_pair_gpu_host_free) band_pair_gpu_host_free(g); else free(g);
        }
        z3_close(p);
    }
    {   /* the bin search vs the atan2 rounding it replaced, over the whole 1x pooled array */
        z3 *c = z3_open(store_open(argv[1]), "codes", nullptr), *p = codes_pool2_open(z3_open(store_open(argv[1]), "codes", nullptr));
        const int64_t *ps = z3_meta_of(p)->shape, zo[3] = {0, 0, 0}; const size_t N = (size_t)ps[0] * ps[1] * ps[2], NF = (size_t)shape[0] * shape[1] * shape[2];
        uint8_t *a = malloc(N), *f = malloc(NF); if (z3_read(p, zo, ps, a, 4) || z3_read(c, zo, shape, f, 4)) { fprintf(stderr, "%s\n", z3_error()); return 1; }
        size_t diff = 0, known = 0, off2 = 0;
        for (int64_t za = 0; za < ps[0]; za++) for (int64_t yb = 0; yb < ps[1]; yb++) for (int64_t xc = 0; xc < ps[2]; xc++) {
            int first = -1, conflict = 0, kk = 0; float x = 0, y = 0;
            for (int q = 0; q < 8 && !conflict; q++) {
                const int64_t z = 2 * za + (q >> 2 & 1), yy = 2 * yb + (q >> 1 & 1), xx = 2 * xc + (q & 1);
                if (z >= shape[0] || yy >= shape[1] || xx >= shape[2]) continue;
                const int v = f[((size_t)z * shape[1] + yy) * shape[2] + xx];
                if (!v) continue;
                if (v > BAND_PERIOD || (first >= 0 && 2 * abs(band_diff(v - 1, first)) > BAND_STEPS / 2)) { conflict = 1; break; }
                if (first < 0) first = v - 1;
                x += (float)cos(2 * M_PI * (v - 1) / BAND_PERIOD); y += (float)sin(2 * M_PI * (v - 1) / BAND_PERIOD); kk++;
            }
            uint8_t o = 0;
            if (conflict) o = 255;
            else if (kk) { double th = atan2(y, x) / (2 * M_PI); if (th < 0) th += 1; o = (uint8_t)(1 + (int)lround(th * BAND_PERIOD) % BAND_PERIOD); }
            const uint8_t nw = a[((size_t)za * ps[1] + yb) * ps[2] + xc];
            known += o && o != 255; if (o != nw) { diff++; if (!o || !nw || o == 255 || nw == 255 || abs(band_diff(o - 1, nw - 1)) > 1) off2++; }
        }
        printf("pool 1, whole array: the bin search differs from the atan2 rounding in %zu of %zu coded voxels (%zu by more than one step or in empty / conflict)\n", diff, known, off2);
        if (off2) fail = 1;
        free(a); free(f); z3_close(c); z3_close(p);
    }
    printf(fail ? "FAIL\n" : "PASS: the GPU lossless decode and the device codes pooling are byte-identical to z3_read / codes_pool2_box\n");
    return fail;
}
