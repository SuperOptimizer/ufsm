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


/* the dense reference of fuse_recto (every voxel, the full dup^3 scan): the sparse version must match it exactly */
static void recto_ref(const uint8_t *const *in, int n, const uint8_t *ct, int d, int block, int dup, uint8_t *out, size_t *st) {
    const int nb1 = (d + block - 1) / block; const size_t nb = (size_t)nb1 * nb1 * nb1, len = (size_t)d * d * d;
    double *sum = calloc(nb * n, sizeof *sum); size_t *cnt = calloc(nb * n, sizeof *cnt); memset(out, 0, len);
    for (size_t i = 0; i < len; i++) { const int z = (int)(i / ((size_t)d * d)), y = (int)(i / d % d), x = (int)(i % d); const size_t b = ((size_t)(z / block) * nb1 + y / block) * nb1 + x / block;
        for (int k = 0; k < n; k++) if (in[k][i]) { sum[b * n + k] += ct[i]; cnt[b * n + k]++; } }
    for (size_t i = 0; i < len; i++) { const int z = (int)(i / ((size_t)d * d)), y = (int)(i / d % d), x = (int)(i % d); const size_t b = ((size_t)(z / block) * nb1 + y / block) * nb1 + x / block;
        for (int k = 0; k < n; k++) { if (!in[k][i]) continue; const float sk = (float)(sum[b * n + k] / cnt[b * n + k]); int beaten = 0, other = 0;
            for (int zz = z - dup; zz <= z + dup; zz++) for (int yy = y - dup; yy <= y + dup; yy++) for (int xx = x - dup; xx <= x + dup; xx++) {
                if (zz < 0 || yy < 0 || xx < 0 || zz >= d || yy >= d || xx >= d) continue; const size_t j = ((size_t)zz * d + yy) * d + xx, bj = ((size_t)(zz / block) * nb1 + yy / block) * nb1 + xx / block;
                for (int o = 0; o < n; o++) { if (o == k || !in[o][j]) continue; other = 1; const float so = (float)(sum[bj * n + o] / cnt[bj * n + o]); if (so > sk || (so == sk && o < k)) beaten = 1; } }
            if (beaten) { st[2]++; continue; }
            if (!out[i]) { out[i] = 255; st[other ? 0 : 1]++; } } }
    free(sum); free(cnt);
}
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
    {   /* random sheets of 4 sets on 70^3 (blocks of 16, dup 3): the sparse fuse_recto equals the dense reference, voxel and stats */
        const int d = 70; const size_t len = (size_t)d * d * d; uint8_t *ins[4], *ctr = malloc(len), *o1 = malloc(len), *o2 = malloc(len); uint64_t r = 12345;
        for (size_t i = 0; i < len; i++) { r = r * 6364136223846793005ull + 1442695040888963407ull; ctr[i] = (uint8_t)(r >> 56); }
        for (int k = 0; k < 4; k++) { ins[k] = calloc(len, 1);
            for (int sheet = 0; sheet < 3; sheet++) { r = r * 6364136223846793005ull + 1; const int x0 = 8 + (int)(r >> 59) * 3 + sheet * 20 + k;   /* planes x ~ x0, some 1 voxel apart across sets */
                for (int z = 0; z < d; z++) for (int y = (k * 9) % 30; y < d - (k * 7) % 25; y++) { const int x = x0 + ((z + y + k) % 7 == 0); if (x < d) ins[k][((size_t)z * d + y) * d + x] = 1; } } }
        size_t s1[3] = {0, 0, 0}, s2[3] = {0, 0, 0};
        fuse_recto((const uint8_t *const *)ins, 4, ctr, d, d, d, 16, 3, o1, s1); recto_ref((const uint8_t *const *)ins, 4, ctr, d, 16, 3, o2, s2);
        CHECK(!memcmp(o1, o2, len) && !memcmp(s1, s2, sizeof s1), "recto-fuse sparse vs dense: voxels %s, stats %zu/%zu/%zu vs %zu/%zu/%zu", memcmp(o1, o2, len) ? "differ" : "same", s1[0], s1[1], s1[2], s2[0], s2[1], s2[2]);
        printf("recto-fuse: sparse == dense reference on 4 random sheet sets (kept near another %zu, unique %zu, dropped %zu)\n", s1[0], s1[1], s1[2]);
        for (int k = 0; k < 4; k++) free(ins[k]);
        free(ctr); free(o1); free(o2);
    }
    printf("fuse tests: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
