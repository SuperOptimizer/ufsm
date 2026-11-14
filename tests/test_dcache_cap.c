/* the derived cache's disk cap (src/dcache.c, 2026-11-14: ~/.cache/ufsm/derived grew to 336 GB and filled the desk's root filesystem).
   CPU only. dcache_evict_pass on a root with fake key dirs of known sizes and last-use times: the least recently used go first until
   the root is under UFSM_DCACHE_GB, keys used within UFSM_DCACHE_PROTECT_MIN and keys open in this process stay; a key whose cell
   dirs were written recently counts as used even with an old inputs.txt. The write gate: below half of UFSM_DCACHE_FREE_GB no cell is
   written (memory only), and writes resume when the floor is met again. */
#include "dcache.h"
#include "store.h"
#include "z3w.h"
#include "zarr3.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)
static int derive(void *ctx, z3 *base, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads) {
    (void)ctx; (void)base; (void)nthreads;
    for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) for (int64_t x = 0; x < n[2]; x++) out[(z * n[1] + y) * n[2] + x] = (uint8_t)((o[0] + z) * 3 + (o[1] + y) * 5 + (o[2] + x));
    return 0;
}
static void mkfile(const char *p, size_t n) { FILE *f = fopen(p, "wb"); char b[4096] = {1}; for (size_t i = 0; i < n; i += sizeof b) fwrite(b, 1, n - i < sizeof b ? n - i : sizeof b, f); fclose(f); }
static void age(const char *p, double hours) { struct timespec t[2]; clock_gettime(CLOCK_REALTIME, &t[0]); t[0].tv_sec -= (time_t)(hours * 3600); t[1] = t[0]; utimensat(AT_FDCWD, p, t, 0); }
static void fake_key(const char *root, const char *name, size_t mb, double hours) {   /* ROOT/name/{inputs.txt, 0/0.0 of mb MB}, everything aged */
    char d[512], p[600]; snprintf(d, sizeof d, "%s/%s", root, name); mkdir(d, 0775); snprintf(p, sizeof p, "%s/0", d); mkdir(p, 0775);
    snprintf(p, sizeof p, "%s/0/0.0", d); mkfile(p, mb << 20); age(p, hours); snprintf(p, sizeof p, "%s/0", d); age(p, hours);
    snprintf(p, sizeof p, "%s/inputs.txt", d); mkfile(p, 10); age(p, hours); age(d, hours);
}
static int exists(const char *root, const char *name) { char p[600]; struct stat st; snprintf(p, sizeof p, "%s/%s", root, name); return !stat(p, &st); }
static long count_cells(const char *dir) { char cmd[700]; snprintf(cmd, sizeof cmd, "find %s -type f -regextype posix-extended -regex '.*/[0-9]+/[0-9]+\\.[0-9]+' | wc -l", dir); FILE *f = popen(cmd, "r"); long n = -1; if (f && fscanf(f, "%ld", &n) != 1) n = -1; if (f) pclose(f); return n; }
int main(int argc, char **argv) {
    const char *T = argc > 1 ? argv[1] : "/tmp/test_dcache_cap"; char cmd[600]; snprintf(cmd, sizeof cmd, "rm -rf '%s' && mkdir -p '%s'", T, T); if (system(cmd)) return 1;
    char root[512], data[512]; snprintf(root, sizeof root, "%s/cache", T); snprintf(data, sizeof data, "%s/data", T); mkdir(root, 0775);
    setenv("UFSM_DCACHE_CHECK_S", "100000", 1);   /* (the background pass only after 10 s: this test calls the pass itself) */
    mkdir(data, 0775);
    { char p[600]; snprintf(p, sizeof p, "%s/base", data); const int64_t sh[3] = {64, 64, 64}; z3w *w = z3w_create(p, sh, 128, 0, 0, nullptr); uint8_t *sb = calloc(128 * 128 * 128, 1);
      CHECK(w && !z3w_write_shard(w, 0, 0, 0, sb, 1) && !z3w_close(w), "base array"); free(sb); }
    /* 1) the write gate: an unreachable floor -> no cell on disk; floor off -> written */
    setenv("UFSM_DCACHE_FREE_GB", "1e9", 1); setenv("UFSM_DCACHE_GB", "0", 1);
    z3 *b = z3_open(store_open(data), "base", nullptr), *dz = b ? z3_open_derived(b, 0, derive, nullptr, nullptr) : nullptr;
    z3 *c = dz ? z3_open_cached(dz, root, "test-dcache-cap v1", 32, 64) : nullptr; CHECK(c, "open cached");
    uint8_t *buf = malloc(64 * 64 * 64); const int64_t o[3] = {0, 0, 0}, n[3] = {32, 64, 64}, n2[3] = {64, 64, 64};
    CHECK(c && !z3_read(c, o, n, buf, 1), "read 1");
    char kd[600]; snprintf(cmd, sizeof cmd, "%s", root); long cells = count_cells(root);
    CHECK(cells == 0, "below the free floor no cell may be written (%ld on disk)", cells);
    setenv("UFSM_DCACHE_FREE_GB", "0", 1);
    CHECK(c && !z3_read(c, o, n2, buf, 1), "read 2");
    cells = count_cells(root); CHECK(cells == 4, "with the floor off the 4 new cells are written (%ld)", cells);
    { int bad = 0; for (int z = 0; z < 64; z++) for (int y = 0; y < 64; y++) for (int x = 0; x < 64; x++) bad += buf[(z * 64 + y) * 64 + x] != (uint8_t)(z * 3 + y * 5 + x); CHECK(!bad, "values (%d wrong)", bad); }
    /* 2) eviction: 5 fake keys of 4 MB, ages 50 / 40 / 30 / 20 h and one used 10 min ago; one aged key with a fresh cell dir */
    fake_key(root, "aaaaaaaaaaaaaaaaaaaaaaaa", 4, 50); fake_key(root, "bbbbbbbbbbbbbbbbbbbbbbbb", 4, 40); fake_key(root, "cccccccccccccccccccccccc", 4, 30);
    fake_key(root, "dddddddddddddddddddddddd", 4, 20); fake_key(root, "eeeeeeeeeeeeeeeeeeeeeeee", 4, 10.0 / 60);
    fake_key(root, "ffffffffffffffffffffffff", 4, 45); { char p[600]; snprintf(p, sizeof p, "%s/ffffffffffffffffffffffff/0", root); age(p, 0.05); }   /* (written 3 min ago) */
    /* the open key (~4 cells of 32^3, ~0.1 MB) was opened just now; the cap 0.014 GB (~13.4 MB) of ~24.1 MB: a, b, c go (12 MB), d stays */
    setenv("UFSM_DCACHE_GB", "0.014", 1); setenv("UFSM_DCACHE_PROTECT_MIN", "60", 1);
    CHECK(dcache_evict_pass(root) == 0, "pass");
    CHECK(!exists(root, "aaaaaaaaaaaaaaaaaaaaaaaa") && !exists(root, "bbbbbbbbbbbbbbbbbbbbbbbb") && !exists(root, "cccccccccccccccccccccccc"), "the three oldest keys evicted");
    CHECK(exists(root, "dddddddddddddddddddddddd"), "the fourth oldest kept (under the cap after three)");
    CHECK(exists(root, "eeeeeeeeeeeeeeeeeeeeeeee") && exists(root, "ffffffffffffffffffffffff"), "keys used / written within the protect window kept");
    /* 3) the open key is never evicted, even over the cap with nothing protected */
    setenv("UFSM_DCACHE_GB", "0.000001", 1); setenv("UFSM_DCACHE_PROTECT_MIN", "0", 1);
    CHECK(dcache_evict_pass(root) == 0, "pass 2");
    cells = count_cells(root); CHECK(cells == 4, "the open array's cells kept (%ld)", cells);
    CHECK(!exists(root, "dddddddddddddddddddddddd") && !exists(root, "eeeeeeeeeeeeeeeeeeeeeeee") && !exists(root, "ffffffffffffffffffffffff"), "the rest evicted without a protect window");
    CHECK(c && !z3_read(c, o, n2, buf, 1), "read 3");
    z3_close(c); free(buf); (void)kd;
    if (!fails) { snprintf(cmd, sizeof cmd, "rm -rf '%s'", T); if (system(cmd)) {} }
    printf(fails ? "test_dcache_cap: %d FAILED\n" : "test_dcache_cap: ok (write gate, LRU eviction to the cap, protect window, open keys kept)\n", fails);
    return fails ? 1 : 0;
}
