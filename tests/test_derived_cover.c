/* z3_derived_cover (src/zarr3.c): a derived array takes up to Z3_MAX_COVER more arrays of its shape (one per extra set of an onfly phase
   "sets" fusion, which takes FUSE_MAX = 64 sets). Until w24 2026-11-26 the cap was 4, so a 6th set failed as "phase fuse: set 5 has another
   shape" (PHerc0814's 15 gauge-free per-segment sets). CPU only: 15 covers accepted, the union of their chunks answered by z3_box_any /
   z3_shard_present, a cover of another shape and one past the cap refused. */
#include "store.h"
#include "z3w.h"
#include "zarr3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)
static int derive(void *ctx, z3 *base, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads) {
    (void)ctx; (void)base; (void)o; (void)nthreads; memset(out, 7, (size_t)n[0] * n[1] * n[2]); return 0;
}
static char g_root[256];
/* an array "name" of shape sh (shard 128, the inner chunk) with only shard (sz, 0, 0) written */
static int mk(const char *name, const int64_t sh[3], int sz) {
    char p[512]; snprintf(p, sizeof p, "%s/%s", g_root, name);
    z3w *w = z3w_create(p, sh, 128, 0, 0, nullptr); uint8_t *b = calloc(128 * 128 * 128, 1); memset(b, 1, 128 * 128 * 128);
    const int r = !w || z3w_write_shard(w, sz, 0, 0, b, 1) || z3w_close(w); free(b); return r;
}
static z3 *op(const char *name) { store *s = store_open(g_root); return s ? z3_open(s, name, nullptr) : nullptr; }
int main(void) {
    char tmp[] = "/tmp/test_derived_cover_XXXXXX"; if (!mkdtemp(tmp)) { fprintf(stderr, "FAIL: mkdtemp\n"); return 1; }
    snprintf(g_root, sizeof g_root, "%s/z", tmp); mkdir(g_root, 0755);
    const int64_t SH[3] = {128 * 18, 128, 128}, SH2[3] = {128 * 18, 128, 256};
    char nm[32];
    for (int k = 0; k < 16; k++) { snprintf(nm, sizeof nm, "s%d", k); CHECK(!mk(nm, SH, k), "array %s", nm); }   /* set k: shard k only */
    CHECK(!mk("other", SH2, 0), "array other");
    z3 *b = op("s0"), *d = b ? z3_open_derived(b, 0, derive, nullptr, nullptr) : nullptr;
    CHECK(d != nullptr, "derived open"); if (!d) return 1;
    for (int k = 1; k < 16; k++) { snprintf(nm, sizeof nm, "s%d", k); z3 *c = op(nm); CHECK(c && z3_derived_cover(d, c) == 0, "cover %d of 15 refused", k); }
    z3 *o = op("other"); CHECK(o && z3_derived_cover(d, o) == -1, "a cover of another shape accepted"); z3_close(o);
    for (int k = 0; k < 16; k++) { const int64_t bo[3] = {128 * k, 0, 0}, bn[3] = {128, 128, 128};
        CHECK(z3_box_any(d, bo, bn) != 0, "box of set %d's shard: no chunk in the union", k);
        CHECK(z3_shard_present(d, k, 0, 0) != 0, "shard %d of set %d not present in the union", k, k); }
    { const int64_t bo[3] = {128 * 17, 0, 0}, bn[3] = {128, 128, 128}; CHECK(z3_box_any(d, bo, bn) == 0, "an unwritten shard reported in the union"); }
    int added = 15;   /* fill to the cap with more opens of s1, then one past it */
    while (added < Z3_MAX_COVER) { z3 *c = op("s1"); if (!c || z3_derived_cover(d, c)) { z3_close(c); break; } added++; }
    CHECK(added == Z3_MAX_COVER, "covers up to the cap: %d of %d", added, Z3_MAX_COVER);
    z3 *x = op("s1"); CHECK(x && z3_derived_cover(d, x) == -1, "a cover past Z3_MAX_COVER accepted"); z3_close(x);
    z3_close(d);
    char cmd[400]; snprintf(cmd, sizeof cmd, "rm -rf '%s'", tmp); if (system(cmd)) {}
    fprintf(stderr, "%s\n", fails ? "test_derived_cover: FAILED" : "test_derived_cover: ok");
    return fails != 0;
}
