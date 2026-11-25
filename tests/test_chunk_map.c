/* z3_chunk_map (perf 11-25: the shard indices read by UFSM_CHUNKMAP_THREADS threads; R11d's teacher zarrs on rclone spent 309 s in 142 k
   serial index reads): a 5 x 3 x 4 shard grid of 256^3 shards (128^3 chunks, a ragged edge), some shards absent, some chunks all fill;
   the map with 1, 3 and 16 threads equals the chunks actually written, bit for bit, with the same count. */
#include "z3w.h"
#include "zarr3.h"
#include "store.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)

int main(void) {
    const char *t = getenv("TMPDIR"); char dir[600]; snprintf(dir, sizeof dir, "%s/chunkmap.%d", t && *t ? t : "/tmp", (int)getpid());
    const int64_t shape[3] = {5 * 256 - 100, 3 * 256, 4 * 256 - 30}, gs[3] = {5, 3, 4};
    const int64_t g[3] = {(shape[0] + 127) / 128, (shape[1] + 127) / 128, (shape[2] + 127) / 128};
    uint8_t *want = calloc((size_t)(g[0] * g[1] * g[2]), 1), *d = malloc((size_t)256 * 256 * 256);
    z3w *w = z3w_create(dir, shape, 256, 0.f, 0, nullptr); CHECK(w != nullptr, "create %s: %s", dir, z3w_error()); if (!w) return 1;
    int64_t nwant = 0;
    for (int64_t sz = 0; sz < gs[0]; sz++) for (int64_t sy = 0; sy < gs[1]; sy++) for (int64_t sx = 0; sx < gs[2]; sx++) {
        const int64_t s = (sz * gs[1] + sy) * gs[2] + sx; if (s % 5 == 2) continue;   /* (absent shards) */
        memset(d, 0, (size_t)256 * 256 * 256);
        for (int c = 0; c < 8; c++) {
            const int cz = c >> 2, cy = c >> 1 & 1, cx = c & 1; if ((s + c) % 3 == 0) continue;   /* (all-fill chunks) */
            const int64_t gz = sz * 2 + cz, gy = sy * 2 + cy, gx = sx * 2 + cx;
            if (gz * 128 >= shape[0] || gy * 128 >= shape[1] || gx * 128 >= shape[2]) continue;
            d[((size_t)(cz * 128 + 5) * 256 + cy * 128 + 7) * 256 + cx * 128 + 9] = (uint8_t)(1 + c);   /* one non-fill voxel inside the shape */
            want[(gz * g[1] + gy) * g[2] + gx] = 1; nwant++;
        }
        CHECK(!z3w_write_shard(w, sz, sy, sx, d, 2), "write shard %lld: %s", (long long)s, z3w_error());
    }
    z3w_close(w);
    const int nts[3] = {1, 3, 16};
    for (int k = 0; k < 3; k++) {
        char e[8]; snprintf(e, sizeof e, "%d", nts[k]); setenv("UFSM_CHUNKMAP_THREADS", e, 1);
        store *st = store_open(dir); z3 *z = st ? z3_open(st, ".", nullptr) : nullptr; CHECK(z != nullptr, "open"); if (!z) break;
        int64_t gr[3]; uint8_t *bits = nullptr; const int64_t n = z3_chunk_map(z, gr, &bits);
        CHECK(n == nwant && gr[0] == g[0] && gr[1] == g[1] && gr[2] == g[2] && bits && !memcmp(bits, want, (size_t)(g[0] * g[1] * g[2])),
              "threads %d: %lld chunks (want %lld), grid %lld %lld %lld", nts[k], (long long)n, (long long)nwant, (long long)gr[0], (long long)gr[1], (long long)gr[2]);
        printf("chunk map, %d thread(s): %lld of %lld chunks\n", nts[k], (long long)n, (long long)(g[0] * g[1] * g[2]));
        free(bits); z3_close(z); store_close(st);
    }
    char cmd[700]; snprintf(cmd, sizeof cmd, "rm -rf '%s'", dir); if (system(cmd)) {}
    free(want); free(d);
    printf("%s\n", fails ? "FAILED" : "ok"); return fails != 0;
}
