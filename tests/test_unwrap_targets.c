/* nn_unwrap_targets (task unwrap): on a planar phase field phase = frac(k . x) and on a radial one phase = frac(r / pitch) the
   targets are the sign-free normal n n^T and min(1, 32 rho) with rho = |grad phase| (turns per voxel); unknown voxels, voxels
   whose +-s stencil samples are unknown and the border are masked. */
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

static void run(const uint8_t *ph, int D, int H, int W, int s, uint8_t *t, uint8_t *m) {
    const size_t n = (size_t)D * H * W;
    uint8_t *dp = nn_malloc(n), *dt = nn_malloc(7 * n), *dm = nn_malloc(n);
    nn_h2d(dp, ph, n); nn_unwrap_targets(dp, D, H, W, s, dt, dm); nn_sync();
    nn_d2h(t, dt, 7 * n); nn_d2h(m, dm, n);
    nn_free(dp); nn_free(dt); nn_free(dm);
}
/* worst component error of the targets at masked voxels against n(z, y, x), rho(z, y, x) */
static double check(const uint8_t *t, const uint8_t *m, int D, int H, int W, void (*truth)(int, int, int, double *, double *), size_t *cnt) {
    const size_t n = (size_t)D * H * W; double err = 0; *cnt = 0;
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const size_t i = ((size_t)z * H + y) * W + x;
        if (!m[i]) continue;
        double nv[3], rho; truth(z, y, x, nv, &rho);
        const double ref[7] = {nv[0] * nv[0], nv[1] * nv[1], nv[2] * nv[2], 0.5 + nv[0] * nv[1], 0.5 + nv[0] * nv[2], 0.5 + nv[1] * nv[2], fmin(1, 32 * rho)};
        for (int c = 0; c < 7; c++) err = fmax(err, fabs(t[c * n + i] / 255. - ref[c]));
        (*cnt)++;
    }
    return err;
}
static double kv[3] = {0.004, -0.007, 0.009};   /* turns per voxel */
static void plane_truth(int z, int y, int x, double *nv, double *rho) {
    (void)z; (void)y; (void)x; const double r = sqrt(kv[0] * kv[0] + kv[1] * kv[1] + kv[2] * kv[2]);
    for (int d = 0; d < 3; d++) nv[d] = kv[d] / r;
    *rho = r;
}
static const double cy = 20.3, cx = -15.7, pitch = 60;
static void radial_truth(int z, int y, int x, double *nv, double *rho) {
    (void)z; const double dy = y - cy, dx = x - cx, r = hypot(dy, dx);
    nv[0] = 0; nv[1] = dy / r; nv[2] = dx / r; *rho = 1 / pitch;
}

int main(void) {
    nn_init(0);
    const int D = 40, H = 48, W = 56, s = 4;
    const size_t n = (size_t)D * H * W;
    uint8_t *ph = malloc(n), *t = malloc(7 * n), *m = malloc(n);
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        double q = kv[0] * z + kv[1] * y + kv[2] * x; q -= floor(q);
        ph[((size_t)z * H + y) * W + x] = (uint8_t)((int)lround(q * 252) % 252);
    }
    run(ph, D, H, W, s, t, m);
    size_t cnt; double e = check(t, m, D, H, W, plane_truth, &cnt);
    const size_t inner = (size_t)(D - 2 * s) * (H - 2 * s) * (W - 2 * s);
    CHECK(cnt == inner, "plane: %zu masked voxels, expected the %zu interior ones", cnt, inner);
    CHECK(e < 0.03, "plane: max target error %.4f", e);
    printf("plane: %zu voxels, max target error %.4f\n", cnt, e);
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {   /* radial, with an unknown slab */
        const size_t i = ((size_t)z * H + y) * W + x;
        double q = hypot(y - cy, x - cx) / pitch; q -= floor(q);
        ph[i] = z >= 20 && z < 23 ? 255 : (uint8_t)((int)lround(q * 252) % 252);
    }
    run(ph, D, H, W, s, t, m);
    e = check(t, m, D, H, W, radial_truth, &cnt);
    size_t bad = 0;
    for (int z = 16; z < 27; z++) {   /* unknown: z in [20, 23) and the voxels whose +-s samples land there */
        if (!((z >= 20 && z < 23) || (z + s >= 20 && z + s < 23) || (z - s >= 20 && z - s < 23))) continue;
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) bad += m[((size_t)z * H + y) * W + x];
    }
    CHECK(bad == 0, "radial: %zu voxels with an unknown stencil sample are in the mask", bad);
    CHECK(e < 0.04, "radial: max target error %.4f", e);
    printf("radial: %zu voxels, max target error %.4f\n", cnt, e);
    free(ph); free(t); free(m);
    if (fails) { fprintf(stderr, "test_unwrap_targets: %d failure(s)\n", fails); return 1; }
    printf("test_unwrap_targets: OK\n");
    return 0;
}
