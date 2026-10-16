/* src/fuse.h: per-voxel phase fusion (agreeing sets -> their circular mean, one set -> itself, disagreeing -> unknown) and the
   per-block recto line choice (one line per sheet: a duplicate trace of the same sheet off the CT ridge is dropped, another set's
   different sheet is kept). CPU only. */
#include "fuse.h"
#include "band.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

int main(void) {
    {   /* phase */
        const uint8_t A[6] = {10, 10, 255, 100, 0, 126}, B[6] = {12, 255, 255, 200, 250, 130}, C[6] = {11, 255, 255, 255, 255, 180};
        const uint8_t *in[3] = {A, B, C}; uint8_t out[6]; size_t st[4] = {0};
        fuse_phase(in, 3, 6, 0.1f, out, st, nullptr);
        CHECK(out[0] == 11, "agreeing 10 / 12 / 11 -> 11 (got %d)", out[0]);
        CHECK(out[1] == 10, "one set -> its value (got %d)", out[1]);
        CHECK(out[2] == PHASE_UNKNOWN, "no set -> unknown (got %d)", out[2]);
        CHECK(out[3] == PHASE_UNKNOWN, "100 vs 200 (0.4 turn) -> unknown (got %d)", out[3]);
        CHECK(out[4] == 251, "0 / 250 across the wrap -> 251 (got %d)", out[4]);
        CHECK(out[5] == PHASE_UNKNOWN, "126 / 130 / 180: one disagrees -> unknown (got %d)", out[5]);
        CHECK(st[0] == 1 && st[1] == 2 && st[2] == 2 && st[3] == 1, "stats %zu %zu %zu %zu", st[0], st[1], st[2], st[3]);
        printf("phase-fuse: agree -> mean, one -> itself, disagree -> unknown, wrap-around ok\n");
    }
    {   /* recto: CT ridge at z 30; set A traces it, set B traces the same sheet 2 voxels off the ridge (z 32) and another sheet (z 50) */
        const int D = 64; const size_t L = (size_t)D * D * D;
        uint8_t *ct = malloc(L), *A = calloc(L, 1), *Bs = calloc(L, 1), *out = malloc(L);
        for (size_t i = 0; i < L; i++) ct[i] = 50;
        for (int y = 0; y < D; y++) for (int x = 0; x < D; x++) { ct[((size_t)30 * D + y) * D + x] = 200; ct[((size_t)50 * D + y) * D + x] = 180;
            A[((size_t)30 * D + y) * D + x] = 255; Bs[((size_t)32 * D + y) * D + x] = 255; Bs[((size_t)50 * D + y) * D + x] = 255; }
        for (int order = 0; order < 2; order++) {
            const uint8_t *in[2] = {order ? Bs : A, order ? A : Bs}; size_t st[3] = {0};
            fuse_recto(in, 2, ct, D, D, D, 32, 3, out, st);
            size_t z30 = 0, z32 = 0, z50 = 0, other = 0;
            for (int z = 0; z < D; z++) for (int y = 0; y < D; y++) for (int x = 0; x < D; x++) { const uint8_t v = out[((size_t)z * D + y) * D + x]; if (!v) continue;
                if (z == 30) z30++; else if (z == 32) z32++; else if (z == 50) z50++; else other++; }
            CHECK(z30 == (size_t)D * D && z32 == 0 && z50 == (size_t)D * D && other == 0, "order %d: ridge line %zu (want %d), duplicate %zu (want 0), other sheet %zu (want %d), stray %zu",
                  order, z30, D * D, z32, z50, D * D, other);
        }
        printf("recto-fuse: the on-ridge trace kept, its off-ridge duplicate dropped, another set's separate sheet kept (both input orders)\n");
        free(ct); free(A); free(Bs); free(out);
    }
    printf("fuse tests: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
