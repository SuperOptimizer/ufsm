/* the derived-array cache's batched reads (src/dcache.c dcache_set_batch, perf2 2026-11-13): a read's missed cells computed as one box
   per block of G^3 cells. CPU only. On a synthetic derived array whose values depend only on the voxel (as the onfly phase's), with a
   constant region (fill cells): batched == cell-by-cell == the direct values for boxes with partial cells and the array's ragged edge;
   4 threads reading overlapping boxes of one cold batched array (claims, waits on another reader's cells); a reopen of the same disk
   cache computes nothing; and the batched cold read asks the inner for fewer, larger boxes. */
#include "dcache.h"
#include "store.h"
#include "z3w.h"
#include "zarr3.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)
static _Atomic long g_calls, g_vox;
static uint8_t val(int64_t z, int64_t y, int64_t x) { return z < 40 ? 0 : (uint8_t)(((z * 7 + y * 13 + x * 3) >> 2) % 251 + 1); }
static int derive(void *ctx, z3 *base, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads) {
    (void)ctx; (void)base; (void)nthreads; atomic_fetch_add(&g_calls, 1); atomic_fetch_add(&g_vox, n[0] * n[1] * n[2]);
    for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) for (int64_t x = 0; x < n[2]; x++) out[(z * n[1] + y) * n[2] + x] = val(o[0] + z, o[1] + y, o[2] + x);
    return 0;
}
static const int64_t SH[3] = {120, 100, 128};
static char g_root[256];
static z3 *open_c(const char *cdir, int batch) {
    store *s = store_open(g_root); z3 *b = s ? z3_open(s, "base", nullptr) : nullptr; if (!b) return nullptr;
    z3 *d = z3_open_derived(b, 0, derive, nullptr, nullptr); if (!d) return nullptr;
    z3 *c = z3_open_cached(d, cdir, "test-dcache-batch v1", 32, 64);
    if (c && batch) CHECK(dcache_set_batch(c, batch) == 0, "dcache_set_batch");
    return c;
}
static int check_box(z3 *z, const int64_t o[3], const int64_t n[3], const char *what) {
    const size_t N = (size_t)n[0] * n[1] * n[2]; uint8_t *b = malloc(N); int bad = 0;
    if (z3_read(z, o, n, b, 1)) { free(b); fprintf(stderr, "FAIL: %s: read\n", what); return 1; }
    for (int64_t zz = 0; zz < n[0] && !bad; zz++) for (int64_t y = 0; y < n[1] && !bad; y++) for (int64_t x = 0; x < n[2] && !bad; x++) {
        const int64_t a = o[0] + zz, bb = o[1] + y, c = o[2] + x;
        const uint8_t e = a < 0 || bb < 0 || c < 0 || a >= SH[0] || bb >= SH[1] || c >= SH[2] ? 0 : val(a, bb, c);
        if (b[(zz * n[1] + y) * n[2] + x] != e) { fprintf(stderr, "FAIL: %s: (%lld %lld %lld) %d vs %d\n", what, (long long)a, (long long)bb, (long long)c, b[(zz * n[1] + y) * n[2] + x], e); bad = 1; }
    }
    free(b); return bad;
}
static const int64_t BOX[6][6] = {{0, 0, 0, 120, 100, 128}, {5, 7, 9, 70, 60, 90}, {33, 31, 1, 64, 33, 127}, {90, 70, 100, 40, 40, 40}, {-8, -8, -8, 50, 50, 50}, {64, 64, 64, 32, 32, 32}};
typedef struct { z3 *z; int t; } targ;
static void *thr(void *p) { targ *a = p; for (int r = 0; r < 6; r++) { const int k = (r + a->t) % 6; fails += check_box(a->z, BOX[k], BOX[k] + 3, "threaded"); } return nullptr; }
int main(void) {
    char tmp[] = "/tmp/test_dcache_batch_XXXXXX"; if (!mkdtemp(tmp)) { fprintf(stderr, "FAIL: mkdtemp\n"); return 1; }
    snprintf(g_root, sizeof g_root, "%s/z", tmp); mkdir(g_root, 0755);
    char p[512]; snprintf(p, sizeof p, "%s/base", g_root);
    z3w *w = z3w_create(p, SH, 128, 0, 0, nullptr); uint8_t *sb = calloc(128 * 128 * 128, 1); memset(sb, 1, 128 * 128 * 128);
    if (!w || z3w_write_shard(w, 0, 0, 0, sb, 1) || z3w_close(w)) { fprintf(stderr, "FAIL: base array\n"); return 1; }
    free(sb);
    char c1[300], c2[300], c3[300]; snprintf(c1, sizeof c1, "%s/c1", tmp); snprintf(c2, sizeof c2, "%s/c2", tmp); snprintf(c3, sizeof c3, "%s/c3", tmp);
    z3 *A = open_c(c1, 0), *B = open_c(c2, 96);
    CHECK(A && B, "open"); if (!A || !B) return 1;
    CHECK(dcache_set_batch(z3_base_of(A), 64) == -1, "dcache_set_batch on a non-cached array");
    g_calls = g_vox = 0; fails += check_box(A, BOX[0], BOX[0] + 3, "cell by cell, whole"); const long ca = g_calls, va = g_vox;
    g_calls = g_vox = 0; fails += check_box(B, BOX[0], BOX[0] + 3, "batched, whole"); const long cb = g_calls, vb = g_vox;
    fprintf(stderr, "cold whole read: cell by cell %ld inner reads (%ld voxels), batched %ld (%ld voxels)\n", ca, va, cb, vb);
    CHECK(ca == 64 && cb < ca && cb >= 8, "inner reads: %ld cell by cell (64 expected), %ld batched", ca, cb);
    for (int k = 1; k < 6; k++) { fails += check_box(A, BOX[k], BOX[k] + 3, "cell by cell"); fails += check_box(B, BOX[k], BOX[k] + 3, "batched"); }
    z3_close(A); z3_close(B);
    z3 *T = open_c(c3, 96); pthread_t th[4]; targ ta[4];
    for (int t = 0; t < 4; t++) { ta[t].z = T; ta[t].t = t; pthread_create(&th[t], nullptr, thr, &ta[t]); }
    for (int t = 0; t < 4; t++) pthread_join(th[t], nullptr);
    z3_close(T);
    dcache_stats s0; dcache_get_stats(&s0);
    z3 *R = open_c(c2, 96); g_calls = 0;
    fails += check_box(R, BOX[0], BOX[0] + 3, "reopened (disk)");
    dcache_stats s1; dcache_get_stats(&s1);
    CHECK(g_calls == 0 && s1.cells_computed == s0.cells_computed && s1.cells_disk - s0.cells_disk == 64, "reopen: %ld inner reads, %llu computed, %llu from disk",
          (long)g_calls, (unsigned long long)(s1.cells_computed - s0.cells_computed), (unsigned long long)(s1.cells_disk - s0.cells_disk));
    z3_close(R);
    char cmd[400]; snprintf(cmd, sizeof cmd, "rm -rf '%s'", tmp); if (system(cmd)) {}
    fprintf(stderr, "%s\n", fails ? "test_dcache_batch: FAILED" : "test_dcache_batch: ok");
    return fails != 0;
}
