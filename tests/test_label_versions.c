/* Golden digests of the cached label code (2026-11-14, the dcache audit: the derived-cache keys carry label_versions.h constants, so a
   value change must come with a bump). A synthetic spiral's winding_mod14 codes (160^3, unknown outside a radius and in a few boxes),
   the CPU pair rule at rmax 8 (phase_onfly_open) and the pair rule over the 2x-pooled codes at rmax 4 (codes_pool2_open): FNV-1a 64 of
   each output, compared with the digest recorded for the current LBL_PHASE_PAIR (and LBL_CODES_POOL). A different digest under the same
   version fails: bump the constant in src/label_versions.h and record the new digest below (the old keys then miss). The device paths are
   byte-identical to these (test_phase_onfly), so the CPU digest covers them. Usage: test_label_versions DIR */
#include "band_onfly.h"
#include "label_versions.h"
#include "band.h"
#include "z3w.h"
#include "zarr3.h"
#include "store.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const struct { int v; unsigned long long h; } g_phase[] = {{2, 0x4223283d2a8421e2ull}};
static const struct { int v, p; unsigned long long h; } g_pool[] = {{2, 2, 0x59011c66dcb9788full}};

enum { N = 160 };
static unsigned long long fnv(const uint8_t *p, size_t n) { unsigned long long h = 1469598103934665603ull; for (size_t k = 0; k < n; k++) { h ^= p[k]; h *= 1099511628211ull; } return h; }
static uint8_t code_at(int z, int y, int x) {   /* a spiral about a tilted axis, pitch 11 voxels; unknown beyond r 70 and in boxes */
    const double cy = 80 + .05 * (z - 80), cx = 80 - .03 * (z - 80), r = hypot(y - cy, x - cx), th = atan2(y - cy, x - cx);
    const double w = r / 11. - th / (2 * M_PI) + .003 * z;
    const int blob = ((z / 40) * 7 + (y / 40) * 3 + x / 40) % 5 == 0;
    return r < 70 && r > 4 && !blob ? (uint8_t)(1 + ((int)floor(w) & 63)) : 0;
}
int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: test_label_versions DIR\n"); return 2; }
    char p[1200]; snprintf(p, sizeof p, "%s/codes", argv[1]);
    const int64_t sh[3] = {N, N, N};
    z3w *w = z3w_create(p, sh, 128, 0.f, 0, nullptr); if (!w) { fprintf(stderr, "FAIL: z3w_create %s\n", p); return 1; }
    uint8_t *b = malloc((size_t)128 * 128 * 128);
    for (int sz = 0; sz < 2; sz++) for (int sy = 0; sy < 2; sy++) for (int sx = 0; sx < 2; sx++) {
        for (int z = 0; z < 128; z++) for (int y = 0; y < 128; y++) for (int x = 0; x < 128; x++) {
            const int gz = sz * 128 + z, gy = sy * 128 + y, gx = sx * 128 + x;
            b[((size_t)z * 128 + y) * 128 + x] = gz < N && gy < N && gx < N ? code_at(gz, gy, gx) : 0;
        }
        if (z3w_write_shard(w, sz, sy, sx, b, 1)) { fprintf(stderr, "FAIL: write\n"); return 1; }
    }
    free(b); if (z3w_close(w)) { fprintf(stderr, "FAIL: close\n"); return 1; }
    store *st = store_open(argv[1]); int fails = 0;
    unsigned long long hp = 0, hq = 0;
    {   /* the pair rule */
        z3 *cz = st ? z3_open(st, "codes", nullptr) : nullptr, *ph = cz ? phase_onfly_open(cz, 8.f) : nullptr;
        const size_t n3 = (size_t)N * N * N; uint8_t *o = malloc(n3); const int64_t o0[3] = {0, 0, 0};
        if (!ph || z3_read(ph, o0, sh, o, 1)) { fprintf(stderr, "FAIL: the onfly phase\n"); return 1; }
        size_t kn = 0; for (size_t k = 0; k < n3; k++) kn += o[k] < PHASE_PERIOD;
        hp = fnv(o, n3); free(o); z3_close(ph);
        printf("pair rule (LBL_PHASE_PAIR %d): %zu known of %zu, digest 0x%016llx\n", LBL_PHASE_PAIR, kn, n3, hp);
        if (kn < n3 / 4) { fprintf(stderr, "FAIL: only %zu known phase voxels\n", kn); fails++; }
    }
    {   /* the pair rule over the 2x-pooled codes */
        z3 *cz = st ? z3_open(st, "codes", nullptr) : nullptr, *pz = cz ? codes_pool2_open(cz) : nullptr, *ph = pz ? phase_onfly_open(pz, 4.f) : nullptr;
        const int64_t s2[3] = {N / 2, N / 2, N / 2}, o0[3] = {0, 0, 0}; const size_t n3 = (size_t)s2[0] * s2[1] * s2[2]; uint8_t *o = malloc(n3);
        if (!ph || z3_read(ph, o0, s2, o, 1)) { fprintf(stderr, "FAIL: the pooled onfly phase\n"); return 1; }
        hq = fnv(o, n3); free(o); z3_close(ph);
        printf("pooled codes (LBL_CODES_POOL %d): digest 0x%016llx\n", LBL_CODES_POOL, hq);
    }
    int found = 0;
    for (size_t i = 0; i < sizeof g_phase / sizeof *g_phase; i++) if (g_phase[i].v == LBL_PHASE_PAIR) { found = 1;
        if (g_phase[i].h != hp) { fprintf(stderr, "FAIL: the pair rule's output changed under LBL_PHASE_PAIR %d (digest 0x%016llx, recorded 0x%016llx): bump LBL_PHASE_PAIR in src/label_versions.h "
                                                  "and record the new digest in tests/test_label_versions.c (cached phase cells of the old code would be served otherwise)\n", LBL_PHASE_PAIR, hp, g_phase[i].h); fails++; } }
    if (!found) { fprintf(stderr, "FAIL: no digest recorded for LBL_PHASE_PAIR %d: add {%d, 0x%016llxull} to g_phase\n", LBL_PHASE_PAIR, LBL_PHASE_PAIR, hp); fails++; }
    found = 0;
    for (size_t i = 0; i < sizeof g_pool / sizeof *g_pool; i++) if (g_pool[i].v == LBL_PHASE_PAIR && g_pool[i].p == LBL_CODES_POOL) { found = 1;
        if (g_pool[i].h != hq) { fprintf(stderr, "FAIL: the pooled codes' phase changed under LBL_PHASE_PAIR %d / LBL_CODES_POOL %d (digest 0x%016llx, recorded 0x%016llx): bump LBL_CODES_POOL "
                                                 "(or LBL_PHASE_PAIR) and record the new digest\n", LBL_PHASE_PAIR, LBL_CODES_POOL, hq, g_pool[i].h); fails++; } }
    if (!found) { fprintf(stderr, "FAIL: no digest recorded for LBL_PHASE_PAIR %d / LBL_CODES_POOL %d: add {%d, %d, 0x%016llxull} to g_pool\n", LBL_PHASE_PAIR, LBL_CODES_POOL, LBL_PHASE_PAIR, LBL_CODES_POOL, hq); fails++; }
    printf(fails ? "test_label_versions: %d FAILED\n" : "test_label_versions: ok (the cached label code matches its recorded versions)\n", fails);
    return fails != 0;
}
