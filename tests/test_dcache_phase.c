/* the derived cache's near-lossless phase codec (src/dcache.c UDC2, 2026-11-14). CPU only. A synthetic phase field (a 252-periodic ramp
   along a tilted axis with per-voxel noise, wrapping; unknown blobs; a few all-unknown and all-constant cells) cached at delta 0..3:
   every known voxel within delta of the computed phase (circular, period 252), every unknown voxel exactly 255 and no known one
   reported unknown; a cold reopen of the same disk cache (the cells read back, decoded) returns the very bytes the first read did
   (memory == disk); delta 0 is exact; the lossy cells are smaller than the exact ones; decoding a truncated cell is a miss. */
#include "dcache.h"
#include "store.h"
#include "z3w.h"
#include "zarr3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)
enum { S = 256, CELL = 128 };
static uint8_t ph(int64_t z, int64_t y, int64_t x) {
    if (z < 0 || y < 0 || x < 0 || z >= S || y >= S || x >= S) return 255;
    const double dz = z - 90.0, dy = y - 130.0, dx = x - 40.0; if (dz * dz + dy * dy + dx * dx < 900 || (z > 200 && x < 60)) return 255;   /* (unknown blobs) */
    if (z >= 128 && y < 128 && x >= 128) return 7;   /* (a constant cell) */
    const double t = 0.37 * z + 0.81 * y + 0.21 * x + 3.0 * (((uint32_t)(z * 73856093u ^ y * 19349663u ^ x * 83492791u) % 1000) / 1000.0);
    return (uint8_t)((int64_t)(t * 6.3) % 252);
}
static int derive(void *ctx, z3 *base, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads) {
    (void)ctx; (void)base; (void)nthreads;
    for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) for (int64_t x = 0; x < n[2]; x++) out[(z * n[1] + y) * n[2] + x] = ph(o[0] + z, o[1] + y, o[2] + x);
    return 0;
}
static char g_data[512];
static z3 *open_ph(const char *root, int delta) {
    z3 *b = z3_open(store_open(g_data), "base", nullptr), *d = b ? z3_open_derived(b, 255, derive, nullptr, nullptr) : nullptr;
    char key[128]; snprintf(key, sizeof key, "test-dcache-phase v1 delta %d", delta);
    z3 *c = d ? z3_open_cached(d, root, key, CELL, 1024) : nullptr; if (c && delta) CHECK(dcache_set_phase(c, delta) == 0, "dcache_set_phase");
    return c;
}
static long du(const char *dir) { char cmd[700]; snprintf(cmd, sizeof cmd, "du -sb %s | cut -f1", dir); FILE *f = popen(cmd, "r"); long n = -1; if (f && fscanf(f, "%ld", &n) != 1) n = -1; if (f) pclose(f); return n; }
int main(int argc, char **argv) {
    const char *T = argc > 1 ? argv[1] : "/tmp/test_dcache_phase"; char cmd[700]; snprintf(cmd, sizeof cmd, "rm -rf '%s' && mkdir -p '%s/data'", T, T); if (system(cmd)) return 1;
    snprintf(g_data, sizeof g_data, "%s/data", T);
    setenv("UFSM_DCACHE_GB", "0", 1); setenv("UFSM_DCACHE_FREE_GB", "0", 1);
    { char p[600]; snprintf(p, sizeof p, "%s/base", g_data); const int64_t sh[3] = {S, S, S}; z3w *w = z3w_create(p, sh, 128, 0, 0, nullptr); uint8_t *sb = calloc(128 * 128 * 128, 1);
      CHECK(w && !z3w_write_shard(w, 0, 0, 0, sb, 1) && !z3w_close(w), "base array"); free(sb); }
    const size_t N = (size_t)S * S * S; uint8_t *a = malloc(N), *b = malloc(N); const int64_t o[3] = {0, 0, 0}, n[3] = {S, S, S};
    long sz0 = 0;
    for (int d = 0; d <= 3; d++) {
        char root[600]; snprintf(root, sizeof root, "%s/c%d", T, d);
        z3 *c = open_ph(root, d); CHECK(c && !z3_read(c, o, n, a, 2), "delta %d: read", d); z3_close(c);
        size_t bad = 0, unk = 0, maxe = 0;
        for (int64_t z = 0; z < S; z++) for (int64_t y = 0; y < S; y++) for (int64_t x = 0; x < S; x++) {
            const uint8_t e = ph(z, y, x), g = a[((size_t)z * S + y) * S + x];
            if (e == 255 || g == 255) { unk += e != g; continue; }
            int er = abs((int)e - (int)g); if (er > 126) er = 252 - er; if ((size_t)er > maxe) maxe = (size_t)er; bad += er > d || g >= 252;
        }
        CHECK(!unk, "delta %d: %zu voxels with the unknown mask wrong", d, unk); CHECK(!bad, "delta %d: %zu voxels beyond delta (max %zu)", d, bad, maxe);
        c = open_ph(root, d); CHECK(c && !z3_read(c, o, n, b, 2), "delta %d: cold read", d); z3_close(c);   /* (a new process's view: every cell from disk) */
        CHECK(!memcmp(a, b, N), "delta %d: the cold read (cells from disk) differs from the first (memory == disk)", d);
        const long sz = du(root); if (d == 0) sz0 = sz; else CHECK(sz < sz0, "delta %d: %ld bytes on disk, not below exact %ld", d, sz, sz0);
        printf("delta %d: max error %zu, %ld bytes on disk\n", d, maxe, sz);
    }
    {   /* a truncated UDC2 cell: a miss (recomputed), the same bytes */
        char cmd2[800]; snprintf(cmd2, sizeof cmd2, "f=$(find %s/c2 -type f -size +8k -name '*.*' ! -name inputs.txt | head -1); truncate -s 5000 \"$f\"", T); CHECK(!system(cmd2), "truncate");
        char root[600]; snprintf(root, sizeof root, "%s/c2", T);
        z3 *c = open_ph(root, 2); dcache_stats s0; dcache_get_stats(&s0); CHECK(c && !z3_read(c, o, n, b, 2), "truncated: read"); dcache_stats s1; dcache_get_stats(&s1); z3_close(c);
        CHECK(s1.cells_computed - s0.cells_computed == 1, "truncated: one cell recomputed (%llu)", (unsigned long long)(s1.cells_computed - s0.cells_computed));
        c = open_ph(root, 2); uint8_t *a2 = malloc(N); CHECK(c && !z3_read(c, o, n, a2, 2), "truncated: reread"); z3_close(c);
        CHECK(!memcmp(a2, b, N), "truncated: the recomputed cell differs"); free(a2);
    }
    free(a); free(b);
    if (!fails) { snprintf(cmd, sizeof cmd, "rm -rf '%s'", T); if (system(cmd)) {} }
    printf(fails ? "test_dcache_phase: %d FAILED\n" : "test_dcache_phase: ok (delta 0..3 bounds, exact mask, memory == disk, smaller, truncated cell a miss)\n", fails);
    return fails ? 1 : 0;
}
