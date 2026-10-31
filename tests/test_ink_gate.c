/* ufsm ink-gate (src/inkgate.c): pseudo ink restricted to a phase band.
   - phase codes in [lo, hi] and ignore 0: out = round(254 x ink / 255); elsewhere (other codes, unknown phase, ignored) 255;
   - the ink on a 2x finer grid (--ink-um half the phase's) is read at the phase voxel centres;
   - a wrapping band (lo > hi) keeps codes lo..251 and 0..hi. */
#include "store.h"
#include "z3w.h"
#include "zarr3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int cmd_ink_gate(int argc, char **argv);   /* src/inkgate.c */

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

enum { D = 20, H = 140, W = 30 };   /* (H > 128: two input shards in y) */
static uint8_t P[D * H * W], G[D * H * W], I2[2 * D * 2 * H * 2 * W];

static void wr(const char *dir, const uint8_t *v, int d, int h, int w, int fill) {
    const int64_t sh[3] = {d, h, w}; uint8_t *sb = malloc(128 * 128 * 128);
    z3w *z = z3w_create(dir, sh, 128, 0, fill, nullptr); CHECK(z != nullptr, "z3w %s", dir);
    for (int sz = 0; sz * 128 < d; sz++) for (int sy = 0; sy * 128 < h; sy++) for (int sx = 0; sx * 128 < w; sx++) {
        memset(sb, fill, 128 * 128 * 128);
        for (int a = 0; a < 128 && sz * 128 + a < d; a++) for (int b = 0; b < 128 && sy * 128 + b < h; b++) for (int c = 0; c < 128 && sx * 128 + c < w; c++)
            sb[((size_t)a * 128 + b) * 128 + c] = v[((size_t)(sz * 128 + a) * h + sy * 128 + b) * w + sx * 128 + c];
        CHECK(z3w_write_shard(z, sz, sy, sx, sb, 1) == 0, "shard %s", dir);
    }
    z3w_close(z); free(sb);
}

static uint8_t *rd(const char *root, const char *key) {
    store *s = store_open(root); z3 *z = s ? z3_open(s, key, nullptr) : nullptr; CHECK(z != nullptr, "open %s/%s", root, key);
    if (!z) return nullptr;
    uint8_t *o = malloc(D * H * W); const int64_t a[3] = {0, 0, 0}, n[3] = {D, H, W};
    CHECK(z3_read(z, a, n, o, 1) == 0, "read %s", root); z3_close(z); store_close(s); return o;
}

int main(void) {
    char tmp[] = "/tmp/test_ink_gate_XXXXXX"; CHECK(mkdtemp(tmp) != nullptr, "mkdtemp"); char p[600], q[600], g[600], o[600], o2[600];
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const size_t i = ((size_t)z * H + y) * W + x;
        P[i] = (x + y) % 9 == 0 ? 255 : (uint8_t)((x * 7 + y * 3 + z) % 252);
        G[i] = (x == 5 && y % 2 == 0) ? 1 : 0;
    }
    for (int z = 0; z < 2 * D; z++) for (int y = 0; y < 2 * H; y++) for (int x = 0; x < 2 * W; x++)
        I2[((size_t)z * 2 * H + y) * 2 * W + x] = (uint8_t)((z / 2 * 13 + y / 2 * 5 + x / 2 * 11) % 256);
    snprintf(p, sizeof p, "%s/phase", tmp); mkdir(p, 0755); snprintf(q, sizeof q, "%s/phase/9.6", tmp); wr(q, P, D, H, W, 255);
    snprintf(g, sizeof g, "%s/ign", tmp); wr(g, G, D, H, W, 0);
    snprintf(q, sizeof q, "%s/ink", tmp); mkdir(q, 0755); snprintf(q, sizeof q, "%s/ink/4.8", tmp); wr(q, I2, 2 * D, 2 * H, 2 * W, 0);
    snprintf(q, sizeof q, "%s/ink", tmp); snprintf(o, sizeof o, "%s/out", tmp); snprintf(o2, sizeof o2, "%s/out2", tmp);
    char *a1[] = {"ufsm", "ink-gate", o, "--ink", q, "4.8", "--ink-um", "4.8", "--phase", p, "9.6", "--phase-um", "9.6", "--ignore", tmp, "ign",
                  "--lo", "0", "--hi", "125", "--threads", "3", nullptr};
    CHECK(cmd_ink_gate(22, a1) == 0, "ink-gate");
    uint8_t *r = rd(o, "9.6"); long bad = 0, in = 0;
    for (int z = 0; r && z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const size_t i = ((size_t)z * H + y) * W + x;
        const int c = P[i], ink = I2[((size_t)(2 * z + 1) * 2 * H + 2 * y + 1) * 2 * W + 2 * x + 1];
        const int want = c <= 125 && !G[i] ? (int)lround(254.0 * ink / 255) : 255;
        if (r[i] != want) bad++;
        if (want != 255) in++;
    }
    CHECK(r && bad == 0 && in > 1000, "band gate: %ld wrong of %d (%ld gated)", bad, D * H * W, in); free(r);
    char *a2[] = {"ufsm", "ink-gate", o2, "--ink", q, "4.8", "--ink-um", "4.8", "--phase", p, "9.6", "--phase-um", "9.6", "--lo", "200", "--hi", "20", nullptr};
    CHECK(cmd_ink_gate(17, a2) == 0, "ink-gate wrap");
    r = rd(o2, "9.6"); bad = 0;
    for (int z = 0; r && z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const size_t i = ((size_t)z * H + y) * W + x; const int c = P[i];
        const int keep = c <= 251 && (c >= 200 || c <= 20);
        if ((r[i] != 255) != keep) bad++;
    }
    CHECK(r && bad == 0, "wrapping band: %ld wrong", bad); free(r);
    char cmd[700]; snprintf(cmd, sizeof cmd, "rm -rf %s", tmp); if (system(cmd)) {}
    if (fails) { fprintf(stderr, "test_ink_gate: %d failures\n", fails); return 1; }
    printf("test_ink_gate: ok\n"); return 0;
}
