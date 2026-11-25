/* predict --tile-list with --seamless 2 (perf 11-25, inkperf casc/st-aux288): the passes run at --grid-origin 0 and S/2 with shards S/2;
   the list's tiles must stay on the user's grid (--tile-list-origin 0,0,0, what seamless.c gives both passes). Window 288 halo 16:
   S = 256, shift 128, tile 128 = shift. Pass b's box is seamless.c pass_box's: the user box moved down onto the shift grid. Before the
   fix the tiles sat at --grid-origin (one tile off at tile = shift: no tile met the box). Checks: every tile meets both passes' boxes, the
   kept shards cover each tile exactly, and the old origin (the shift) loses the tile, as the bug did. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

long pred_tile_keep(FILE *tf, int64_t tls, const int64_t tlo[3], const int64_t bo[3], int shard, const int64_t ns[3], uint8_t *keep, long *nl);
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)

static long run(const char *list, int64_t tls, const int64_t tlo[3], const int64_t bo[3], int shard, const int64_t ns[3], uint8_t *keep) {
    FILE *f = fmemopen((void *)list, strlen(list), "r"); long nl = 0;
    memset(keep, 0, (size_t)(ns[0] * ns[1] * ns[2]));
    const long n = pred_tile_keep(f, tls, tlo, bo, shard, ns, keep, &nl); fclose(f); return n;
}

int main(void) {
    const int S = 256, S2 = 128, shard = S2;
    for (int tls = 128; tls <= 256; tls += 128) {
        const char *list = tls == 128 ? "10 10 10\n" : "5 5 5\n";   /* the same voxels: [1280, 1280 + tls) per axis */
        const int64_t t0 = 1280, t1 = 1280 + tls;
        for (int pass = 0; pass < 2; pass++) {
            const int64_t shift = pass ? S2 : 0; int64_t bo[3], ns[3];
            for (int d = 0; d < 3; d++) {   /* seamless.c pass_box(S, shift) of the box [t0, t1) */
                long long lo = t0 - shift; lo = (lo >= 0 ? lo / S : -((-lo + S - 1) / S)) * S + shift; if (lo < 0) lo = shift;
                bo[d] = lo; ns[d] = (t1 - lo + shard - 1) / shard;
            }
            uint8_t keep[64]; const int64_t zero[3] = {0, 0, 0}, old[3] = {shift, shift, shift};
            const long n = run(list, tls, zero, bo, shard, ns, keep);
            CHECK(n == 1, "tile %d pass %c: %ld tiles meet the box (origin 0)", tls, 'a' + pass, n);
            int ok = 1;   /* kept shards = exactly those inside the tile (shards and tile both on the 128 grid) */
            for (int64_t z = 0; z < ns[0]; z++) for (int64_t y = 0; y < ns[1]; y++) for (int64_t x = 0; x < ns[2]; x++) {
                const int64_t q[3] = {z, y, x}; int in = 1;
                for (int d = 0; d < 3; d++) { const int64_t s0 = bo[d] + q[d] * shard; if (s0 < t0 || s0 + shard > t1) in = 0; }
                if (keep[(z * ns[1] + y) * ns[2] + x] != in) ok = 0;
            }
            CHECK(ok, "tile %d pass %c: the kept shards are not the tile's", tls, 'a' + pass);
            if (pass && tls == S2) {
                const long nb = run(list, tls, old, bo, shard, ns, keep);
                CHECK(nb == 0, "tile = shift, pass b at the old origin (the shift): %ld tiles, expected the bug's 0", nb);
            }
            printf("tile %d pass %c: box origin %lld, %lld^3 shards, tile met\n", tls, 'a' + pass, (long long)bo[0], (long long)ns[0]);
        }
    }
    printf("%s\n", fails ? "FAILED" : "ok"); return fails != 0;
}
