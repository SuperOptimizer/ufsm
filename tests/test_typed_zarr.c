/* Typed zarr arrays (volcomp revision 5): z3w_create_typed writes uint16 (exact) and float32 (error-bounded) shards,
   z3_open_typed / z3_read read windows back (across shards, partial edge shards, a missing shard = the fill value);
   z3_open still rejects a typed array. */
#include "store.h"
#include "z3w.h"
#include "zarr3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

static double field(int64_t z, int64_t y, int64_t x) { return 256.0 * (hypot((double)y - 130, (double)x - 90) / 37.0 + 0.1 * sin(z * 0.05)); }

int main(void) {
    const char *root = "/tmp/ufsm_test_typed_zarr";
    if (system("rm -rf /tmp/ufsm_test_typed_zarr")) return 1;
    const int64_t shape[3] = {200, 300, 170};   /* 2 x 3 x 2 shards of 128, partial at the edges */
    const int S = 128; const size_t s3 = (size_t)S * S * S;
    for (int pass = 0; pass < 2; pass++) {   /* 0: uint16 exact, 1: float32 with q = 0.5 */
        char dir[256]; snprintf(dir, sizeof dir, "%s/%s", root, pass ? "f32" : "u16");
        z3w *w = z3w_create_typed(dir, shape, S, pass ? 5 : 1, pass ? 0.5f : 0.f, pass ? -1.0 : 7.0, nullptr);
        CHECK(w, "create %s: %s", dir, z3w_error());
        if (!w) continue;
        uint8_t *buf = malloc(s3 * 4);
        for (int64_t sz = 0; sz < 2; sz++) for (int64_t sy = 0; sy < 3; sy++) for (int64_t sx = 0; sx < 2; sx++) {
            if (sz == 1 && sy == 2 && sx == 1) continue;   /* missing: reads as the fill value */
            for (int z = 0; z < S; z++) for (int y = 0; y < S; y++) for (int x = 0; x < S; x++) {
                const double v = field(sz * S + z, sy * S + y, sx * S + x); const size_t i = ((size_t)z * S + y) * S + x;
                if (pass) { float f = (float)v; memcpy(buf + 4 * i, &f, 4); } else { uint16_t u = (uint16_t)lround(v); memcpy(buf + 2 * i, &u, 2); }
            }
            CHECK(!z3w_write_shard(w, sz, sy, sx, buf, 4), "write shard %lld %lld %lld: %s", (long long)sz, (long long)sy, (long long)sx, z3w_error());
        }
        z3w_close(w); free(buf);
        store *st = store_open(root);
        CHECK(!z3_open(st, pass ? "f32" : "u16", nullptr), "z3_open must reject a typed array");
        z3 *z = z3_open_typed(st, pass ? "f32" : "u16", nullptr);
        CHECK(z && z3_meta_of(z)->esize == (pass ? 4 : 2), "open typed: %s", z ? "" : z3_error());
        if (!z) continue;
        const int64_t o[3] = {100, 100, 60}, n[3] = {100, 200, 110};   /* spans 8 shards incl. the missing one and the array edges */
        const size_t nv = (size_t)n[0] * n[1] * n[2]; uint8_t *out = malloc(nv * 4);
        CHECK(!z3_read(z, o, n, out, 4), "read: %s", z3_error());
        double maxerr = 0; size_t bad_fill = 0;
        for (int64_t a = 0; a < n[0]; a++) for (int64_t b = 0; b < n[1]; b++) for (int64_t c = 0; c < n[2]; c++) {
            const int64_t gz = o[0] + a, gy = o[1] + b, gx = o[2] + c; const size_t i = ((size_t)a * n[1] + b) * n[2] + c;
            const int missing = gz >= 128 && gy >= 256 && gx >= 128;
            double got;
            if (pass) { float f; memcpy(&f, out + 4 * i, 4); got = f; } else { uint16_t u; memcpy(&u, out + 2 * i, 2); got = u; }
            if (missing) { bad_fill += got != (pass ? -1.0 : 7.0); continue; }
            const double ref = pass ? (double)(float)field(gz, gy, gx) : (double)(uint16_t)lround(field(gz, gy, gx));
            if (fabs(got - ref) > maxerr) maxerr = fabs(got - ref);
        }
        CHECK(bad_fill == 0, "%s: %zu missing-shard voxels differ from the fill value", pass ? "f32" : "u16", bad_fill);
        CHECK(pass ? maxerr <= 0.25 + 1e-4 : maxerr == 0, "%s: max error %g", pass ? "f32 q 0.5" : "u16 exact", maxerr);
        printf("%s: max error %g, fill ok\n", pass ? "float32 q 0.5" : "uint16 exact", maxerr);
        free(out); z3_close(z);
    }
    if (system("rm -rf /tmp/ufsm_test_typed_zarr")) fprintf(stderr, "cannot remove %s\n", root);
    if (fails) { fprintf(stderr, "test_typed_zarr: %d failure(s)\n", fails); return 1; }
    printf("test_typed_zarr: OK\n");
    return 0;
}
