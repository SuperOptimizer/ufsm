/* zcache (src/zcache.h): reads through the shared decoded-chunk cache = z3_read, for random boxes (inside, straddling the edges,
   fully outside) from several threads at once, with a budget small enough to evict constantly; hits counted on repeats. */
#include "zcache.h"
#include "store.h"
#include "z3w.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static z3 *Z; static zcache *C; static int bad = 0;
static uint64_t lcg(uint64_t *s) { *s = *s * 6364136223846793005ull + 1442695040888963407ull; return *s >> 17; }
static void *run(void *arg) {
    uint64_t s = (uint64_t)(uintptr_t)arg * 7919 + 1; const int64_t *sh = z3_meta_of(Z)->shape;
    for (int t = 0; t < 120; t++) {
        int64_t o[3], n[3]; for (int d = 0; d < 3; d++) { n[d] = 1 + (int64_t)(lcg(&s) % 200); o[d] = (int64_t)(lcg(&s) % (uint64_t)(sh[d] + 200)) - 100; }
        const size_t nv = (size_t)n[0] * n[1] * n[2]; uint8_t *a = malloc(nv), *b = malloc(nv);
        const int ra = z3_read(Z, o, n, a, 1), rb = zcache_read(C, Z, o, n, b);
        if (ra || rb || memcmp(a, b, nv)) __atomic_add_fetch(&bad, 1, __ATOMIC_RELAXED);
        free(a); free(b);
    }
    return nullptr;
}
int main(void) {
    char dir[] = "/tmp/test_zcacheXXXXXX"; if (!mkdtemp(dir)) return 1;
    const int64_t shape[3] = {300, 260, 390}; char arr[512]; snprintf(arr, sizeof arr, "%s/a.zarr", dir);
    z3w *w = z3w_create(arr, shape, 128, 0, 0, nullptr); if (!w) { fprintf(stderr, "z3w: %s\n", z3w_error()); return 1; }
    uint8_t *sb = malloc(128 * 128 * 128);
    for (int64_t sz = 0; sz < 3; sz++) for (int64_t sy = 0; sy < 3; sy++) for (int64_t sx = 0; sx < 4; sx++) {
        for (int z = 0; z < 128; z++) for (int y = 0; y < 128; y++) for (int x = 0; x < 128; x++) {
            const int64_t gz = sz * 128 + z, gy = sy * 128 + y, gx = sx * 128 + x;
            sb[((size_t)z * 128 + y) * 128 + x] = (sz == 1 && sy == 1 && sx == 2) ? 0 : (uint8_t)((gz * 7 + gy * 13 + gx * 3) % 251 + 1);   /* (one empty chunk: fill) */
        }
        if (z3w_write_shard(w, sz, sy, sx, sb, 1)) { fprintf(stderr, "write: %s\n", z3w_error()); return 1; }
    }
    z3w_close(w); free(sb);
    store *st = store_open(dir); Z = z3_open(st, "a.zarr", nullptr); if (!Z) { fprintf(stderr, "open: %s\n", z3_error()); return 1; }
    C = zcache_open((size_t)6 * 128 * 128 * 128);   /* 6 chunks of 36: constant eviction */
    pthread_t th[8]; for (int i = 0; i < 8; i++) pthread_create(&th[i], nullptr, run, (void *)(uintptr_t)i);
    for (int i = 0; i < 8; i++) pthread_join(th[i], nullptr);
    uint64_t h, m; size_t by; zcache_stats(C, &h, &m, &by);
    printf("zcache: 960 random boxes x 8 threads, %llu hits %llu misses, %.1f MB held (budget 12.6)\n", (unsigned long long)h, (unsigned long long)m, by / 1e6);
    int fails = 0;
    printf("zcache: %-90s %s\n", "reads = z3_read (inside / edge / outside boxes, 8 threads, evicting)", bad ? "FAIL" : "ok"); fails += bad != 0;
    printf("zcache: %-90s %s\n", "budget respected (no referenced chunk left over) and hits happen", by <= (size_t)6 * 128 * 128 * 128 && h > 0 ? "ok" : "FAIL"); fails += !(by <= (size_t)6 * 128 * 128 * 128 && h > 0);
    zcache_close(C); z3_close(Z);
    char cmd[600]; snprintf(cmd, sizeof cmd, "rm -rf -- '%s'", dir); if (system(cmd)) {}
    return fails != 0;
}
