/* render_absent_count (unroll audit 11-14 #13): an ink volume of two 128^3 shards along y with only the first written, a CT with both;
   a flat surface over y 8 .. 228 (vertices every 4 voxels along y): the ink's absent share = the vertices at y >= 128, the CT's 0; vertices
   past the array are absent too; a crop counts only its vertices; invalid vertices are not counted; the render itself still runs. */
#include "imgops.h"
#include "store.h"
#include "z3w.h"
#include "zarr3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

int main(void) {
    char dir[] = "/tmp/test_render_absent_XXXXXX"; CHECK(mkdtemp(dir) != nullptr, "mkdtemp");
    const int S = 128; const int64_t shape[3] = {64, 256, 64};
    uint8_t *buf = malloc((size_t)S * S * S); z3 *zz[2]; char p[800], q[800];
    for (int v = 0; v < 2; v++) {   /* v 0: ink (shard (0,1,0) never written), v 1: CT */
        snprintf(p, sizeof p, "%s/v%d.zarr/0", dir, v);
        z3w *w = z3w_create(p, shape, S, 0, 0, "{}"); CHECK(w != nullptr, "z3w_create %s", z3w_error());
        for (int sy = 0; sy < 2; sy++) {
            if (v == 0 && sy == 1) continue;
            memset(buf, v ? 200 : 90, (size_t)S * S * S);
            CHECK(!z3w_write_shard(w, 0, sy, 0, buf, 2), "write shard %s", z3w_error());
        }
        CHECK(!z3w_close(w), "close");
        snprintf(q, sizeof q, "v%d.zarr/0", v);
        zz[v] = z3_open(store_open(dir), q, nullptr); CHECK(zz[v] != nullptr, "z3_open %s", z3_error());
    }
    const int w = 20, h = 56; float *X = malloc(w * h * 4), *Y = malloc(w * h * 4), *Z = malloc(w * h * 4);
    size_t want = 0, valid = 0;
    for (int r = 0; r < h; r++) for (int c = 0; c < w; c++) {
        const int i = r * w + c; X[i] = 10.f + 2 * c; Y[i] = 8.f + 4 * r; Z[i] = 30.f;
        if (r == 0 && c < 3) X[i] = Y[i] = Z[i] = -1.f;   /* invalid: not counted */
        else { valid++; want += Y[i] >= 128; }
    }
    const float *const Xs[1] = {X}, *const Ys[1] = {Y}, *const Zs[1] = {Z};
    size_t ab[2], ns[2], tot;
    CHECK(!render_absent_count(zz, 2, 1, Xs, Ys, Zs, w, h, 2.0, nullptr, ab, ns, &tot), "count");
    CHECK(tot == valid, "total %zu, want %zu", tot, valid);
    CHECK(ab[0] == want && ns[0] == 1, "ink: %zu absent in %zu shards, want %zu in 1", ab[0], ns[0], want);
    CHECK(ab[1] == 0 && ns[1] == 0, "CT: %zu absent in %zu shards", ab[1], ns[1]);
    const int crop[4] = {0, 0, 40, 40};   /* pixels (gs 2): rows 0..19 = y 8..84, all in the written shard */
    CHECK(!render_absent_count(zz, 2, 1, Xs, Ys, Zs, w, h, 2.0, crop, ab, ns, &tot), "count crop");
    CHECK(tot == 20 * 20 - 3 && ab[0] == 0, "crop: total %zu, ink absent %zu", tot, ab[0]);
    for (int c = 0; c < w; c++) Z[(h - 1) * w + c] = 70.f;   /* the last row past the array (z >= 64): absent in both volumes */
    CHECK(!render_absent_count(zz, 2, 1, Xs, Ys, Zs, w, h, 2.0, nullptr, ab, ns, &tot), "count outside");
    CHECK(ab[1] == (size_t)w && ab[0] == want, "outside: CT %zu ink %zu (want %d / %zu)", ab[1], ab[0], w, want);
    for (int c = 0; c < w; c++) Z[(h - 1) * w + c] = 30.f;
    const int ow = render_out_px(w, 2.0), oh = render_out_px(h, 2.0); const int dp[2] = {0, 0};
    uint8_t *im[2] = {calloc((size_t)ow * oh, 1), calloc((size_t)ow * oh, 1)};
    CHECK(!render_wraps_chunked(zz, 2, dp, 1, Xs, Ys, Zs, w, h, 2.0, im, ow, oh, nullptr, (size_t)1 << 30, 2, 2, 1), "render");
    CHECK(im[0][(size_t)4 * ow + 4] == 90 && im[0][(size_t)(oh - 4) * ow + 4] == 0 && im[1][(size_t)(oh - 4) * ow + 4] == 200,
          "render: ink %d / %d, CT %d", im[0][(size_t)4 * ow + 4], im[0][(size_t)(oh - 4) * ow + 4], im[1][(size_t)(oh - 4) * ow + 4]);
    char cmd[900]; snprintf(cmd, sizeof cmd, "rm -rf -- '%s'", dir); if (!strncmp(dir, "/tmp/test_render_absent_", 24)) (void)!system(cmd);
    printf("render absent: %s\n", fails ? "FAIL" : "ok"); return fails != 0;
}
