/* inkfly_open_pooled (the coarser level of an on-the-fly ink target: 2x pooling of the level below, the known children's mean) read in z-slabs of the
   fine level == the pooling of one whole fine read, for boxes that start on odd planes, cross the slab cuts and leave the array.
   CPU only. usage: test_ink_pool DIR */
#include "inksplat.h"
#include "zarr3.h"
#include "z3w.h"
#include "store.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)
static z3 *make(const char *dir, const char *name, const int64_t sh[3], int fill, uint64_t seed) {
    char p[1600], pa[1700]; snprintf(p, sizeof p, "%s/%s", dir, name); snprintf(pa, sizeof pa, "%s/0", p);
    z3w *w = z3w_create(pa, sh, 128, 0, fill, nullptr); if (!w) return nullptr;
    uint8_t *b = malloc((size_t)128 * 128 * 128); uint64_t rs = seed;
    for (int64_t sz = 0; sz * 128 < sh[0]; sz++) for (int64_t sy = 0; sy * 128 < sh[1]; sy++) for (int64_t sx = 0; sx * 128 < sh[2]; sx++) {
        for (size_t k = 0; k < (size_t)128 * 128 * 128; k++) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;   /* mostly unknown, runs of known values */
            b[k] = (rs >> 20) % 5 < 3 ? 255 : (uint8_t)((rs >> 8) % 255); }
        if (z3w_write_shard(w, sz, sy, sx, b, 2)) { free(b); return nullptr; }
    }
    free(b); if (z3w_close(w)) return nullptr;
    const double um = 8; if (z3w_write_group(p, &um, 1, name, nullptr)) return nullptr;
    store *st = store_open(p); return st ? z3_open(st, "0", nullptr) : nullptr;
}
int main(int argc, char **argv) {
    if (argc == 6 && !strcmp(argv[1], "--bench")) {   /* --bench STORE FINE COARSE P: one P^3 pooled read from stored arrays (any u8 pyramid), its peak RSS */
        store *st = store_open(argv[2]); z3 *fine = st ? z3_open(st, argv[3], nullptr) : nullptr, *coarse = st ? z3_open(st, argv[4], nullptr) : nullptr;
        if (!fine || !coarse) { fprintf(stderr, "bench: cannot open\n"); return 1; }
        const int P = atoi(argv[5]); const int64_t *sh = z3_meta_of(coarse)->shape, o[3] = {(sh[0] - P) / 2, (sh[1] - P) / 2, (sh[2] - P) / 2}, n[3] = {P, P, P};
        z3 *pz = inkfly_open_pooled(coarse, fine, 0); uint8_t *out = malloc((size_t)P * P * P); struct rusage r0, r1; getrusage(RUSAGE_SELF, &r0);
        if (!pz || !out || z3_read(pz, o, n, out, 1)) { fprintf(stderr, "bench: read failed\n"); return 1; }
        getrusage(RUSAGE_SELF, &r1); uint64_t h = 0; for (size_t k = 0; k < (size_t)P * P * P; k++) h = h * 1099511628211ull + out[k];
        printf("bench: P %d pooled read, peak RSS %.2f GB (%.2f before the read), hash %016llx\n", P, r1.ru_maxrss / 1048576.0, r0.ru_maxrss / 1048576.0, (unsigned long long)h);
        return 0;
    }
    if (argc != 2) { fprintf(stderr, "usage: test_ink_pool DIR\n"); return 2; }
    char c[1500]; snprintf(c, sizeof c, "rm -rf '%s' && mkdir -p '%s'", argv[1], argv[1]); if (system(c)) return 1;
    setenv("UFSM_INKPOOL_SLAB", "64", 1);   /* (cuts every 32 coarse planes: several per box) */
    const int64_t fs[3] = {300, 200, 260}, cs[3] = {150, 100, 130};
    z3 *fine = make(argv[1], "fine", fs, 255, 0x9e3779b97f4a7c15ull), *coarse = make(argv[1], "coarse", cs, 255, 7);
    CHECK(fine && coarse, "the arrays"); if (!fine || !coarse) return 1;
    z3 *pz = inkfly_open_pooled(coarse, fine, 0); CHECK(pz != nullptr, "inkfly_open_pooled"); if (!pz) return 1;
    const int64_t bx[5][6] = {{0, 0, 0, 150, 100, 130}, {1, 3, 5, 97, 64, 100}, {31, 0, 0, 3, 100, 130}, {63, 10, 10, 66, 50, 50}, {100, 40, 60, 50, 60, 70}};
    size_t tot = 0, known = 0;
    for (int k = 0; k < 5; k++) {
        const int64_t o[3] = {bx[k][0], bx[k][1], bx[k][2]}, n[3] = {bx[k][3], bx[k][4], bx[k][5]}, fo[3] = {2 * o[0], 2 * o[1], 2 * o[2]}, fn[3] = {2 * n[0], 2 * n[1], 2 * n[2]};
        const size_t nv = (size_t)n[0] * n[1] * n[2], pl = (size_t)fn[1] * fn[2];
        uint8_t *got = malloc(nv), *f = malloc(8 * nv); size_t bad = 0;
        CHECK(!z3_read(pz, o, n, got, 2), "pooled read %d", k); CHECK(!z3_read(fine, fo, fn, f, 2), "fine read %d", k);
        for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) for (int64_t x = 0; x < n[2]; x++) {
            const uint8_t *q = f + ((size_t)(2 * z) * fn[1] + 2 * y) * fn[2] + 2 * x;
            const uint8_t v[8] = {q[0], q[1], q[fn[2]], q[fn[2] + 1], q[pl], q[pl + 1], q[pl + fn[2]], q[pl + fn[2] + 1]};
            unsigned sm = 0, kk = 0; for (int i = 0; i < 8; i++) if (v[i] != 255) { sm += v[i]; kk++; }   /* (the known mean, written out: ink_pool8) */
            const uint8_t want = kk ? (uint8_t)((sm + kk / 2) / kk) : 255; bad += got[((size_t)z * n[1] + y) * n[2] + x] != want; known += want != 255;
        }
        CHECK(bad == 0, "box %d: %zu of %zu pooled voxels differ from the whole-box pooling", k, bad, nv);
        tot += nv; free(got); free(f);
    }
    CHECK(known > tot / 4 && known < tot, "known %zu of %zu", known, tot);
    z3_close(pz); z3_close(fine);
    if (fails) { fprintf(stderr, "test_ink_pool: %d failure(s)\n", fails); return 1; }
    printf("test_ink_pool: OK (%zu voxels in 5 boxes, %zu known, slabs of 64 fine planes == one read)\n", tot, known);
    return 0;
}
