/* The codes rasters' grid convention (meshcodes.c mc_ctx note, ingest.c raster_shard; DESIGN 2026-11-14 "datafix"): level-l voxel j is
   centred on level-0 2^l j + (2^l - 1) / 2, the centre of its block, as the CT pyramid's pool2_mean has it. A plane x = X0 (integer
   level-0 voxel, its own 2^l block holding it) must land in level-l column X0 >> l at every level, for every X0 mod 2^l; the old rule
   (round(X0 / 2^l)) put half of them a column too far. Through mesh_codes_open (the CPU stamp; test_codes_onfly checks the GPU stamp
   against it byte for byte, and ufsm raster against both). Usage: test_codes_grid DIR */
#include "meshcodes.h"
#include "store.h"
#include "z3w.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
static int tif(const char *path, int w, int h, const float *v) {   /* a minimal uncompressed float32 single-strip TIFF (as test_codes_onfly.sh) */
    FILE *f = fopen(path, "wb"); if (!f) return -1;
    const uint32_t nb = (uint32_t)w * h * 4, ifd = 8 + nb;
    fwrite("II*\0", 1, 4, f); fwrite(&ifd, 4, 1, f); fwrite(v, 4, (size_t)w * h, f);
    const uint16_t n = 9; fwrite(&n, 2, 1, f);
    const uint16_t tags[9][2] = {{256, 4}, {257, 4}, {258, 3}, {259, 3}, {273, 4}, {277, 3}, {278, 4}, {279, 4}, {339, 3}};
    const uint32_t vals[9] = {(uint32_t)w, (uint32_t)h, 32, 1, 8, 1, (uint32_t)h, nb, 3};
    for (int k = 0; k < 9; k++) { const uint32_t one = 1; fwrite(tags[k], 2, 2, f); fwrite(&one, 4, 1, f);
        if (tags[k][1] == 3) { const uint16_t s2[2] = {(uint16_t)vals[k], 0}; fwrite(s2, 2, 2, f); } else fwrite(&vals[k], 4, 1, f); }
    const uint32_t z = 0; fwrite(&z, 4, 1, f); return fclose(f);
}
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: test_codes_grid DIR\n"); return 2; }
    const char *D = argv[1]; char p[4096]; mkdir(D, 0755);
    snprintf(p, sizeof p, "%s/mc", D); mkdir(p, 0755); setenv("UFSM_MESHCODES_CACHE", p, 1);
    const int W = 40, H = 40, N = 128; float *x = malloc(sizeof(float) * W * H * 4), *y = x + W * H, *z = y + W * H, *q = z + W * H;
    store *st = store_open(D); int fail = 0, checked = 0;
    for (int L = 0; L <= 3; L++) {   /* the CT level array the codes take their grid from (meta only) */
        const int64_t sh[3] = {N >> L, N >> L, N >> L}; snprintf(p, sizeof p, "%s/ct/%d", D, L);
        z3w *w = z3w_create(p, sh, 128, 0.f, 0, "{}"); if (!w) { fprintf(stderr, "z3w_create %s\n", p); return 1; }
        uint8_t *zero = calloc((size_t)128 * 128 * 128, 1); z3w_write_shard(w, 0, 0, 0, zero, 1); free(zero); z3w_close(w);
    }
    for (int X0 = 40; X0 < 48; X0++) {
        char md[4096]; snprintf(md, sizeof md, "%s/p%d", D, X0); mkdir(md, 0755); snprintf(p, sizeof p, "%s/plane%d.tifxyz", md, X0);   /* (distinct names: the pack key is the names + sizes + mtimes, not the dir) */ mkdir(p, 0755);
        for (int r = 0; r < H; r++) for (int c = 0; c < W; c++) { const int i = r * W + c; x[i] = (float)X0; y[i] = 30.f + c * 1.7f; z[i] = 30.f + r * 1.7f; q[i] = 3.0f; }
        const char *nm[4] = {"x", "y", "z", "q"}; const float *ar[4] = {x, y, z, q};
        for (int k = 0; k < 4; k++) { char t[4200]; snprintf(t, sizeof t, "%s/%s.tif", p, nm[k]); if (tif(t, W, H, ar[k])) return 1; }
        snprintf(p, sizeof p, "%s/plane%d.tifxyz/meta.json", md, X0); FILE *f = fopen(p, "w"); fputs("{}", f); fclose(f);
        for (int L = 0; L <= 3; L++) {
            char key[32]; snprintf(key, sizeof key, "ct/%d", L);
            z3 *ct = z3_open(st, key, nullptr); if (!ct) { fprintf(stderr, "z3_open %s\n", key); return 1; }
            z3 *mc = mesh_codes_open(md, "q.tif", nullptr, L, ct, -1); if (!mc) return 1;
            const int n = N >> L; const int64_t o[3] = {0, 0, 0}, nn[3] = {n, n, n}; uint8_t *b = malloc((size_t)n * n * n);
            if (z3_read(mc, o, nn, b, 1)) return 1;
            int lo = n, hi = -1; for (size_t i = 0; i < (size_t)n * n * n; i++) if (b[i]) { const int xi = (int)(i % n); if (xi < lo) lo = xi; if (xi > hi) hi = xi; }
            const int want = X0 >> L, ok = lo == want && hi == want; fail += !ok; checked++;
            if (!ok) fprintf(stderr, "plane x = %d at level %d: columns %d..%d, want %d  FAIL\n", X0, L, lo, hi, want);
            free(b); z3_close(mc);
        }
    }
    fprintf(stderr, "test_codes_grid: %d plane / level cases, %d wrong: %s\n", checked, fail, fail ? "FAIL" : "ok");
    return fail ? 1 : 0;
}
