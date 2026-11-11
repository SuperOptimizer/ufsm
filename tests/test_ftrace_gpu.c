/* test_ftrace_gpu: fiber-trace's per-voxel fibre axis on the GPU (src/nn_ftrace.cu) is byte-identical to the CPU reference
   (src/ftaxis.c ft_ridge_axis) on synthetic presence volumes: random tubes (H-like and V-like) in noise, odd sizes so every clamp at the
   box faces is exercised, a constant (zero-gradient) region, --st-r 1 / 2 / 3 and the three prior modes (none, H, V). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ftgpu.h"

static uint64_t rs = 0x9e3779b97f4a7c15ull;
static double rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (double)(rs >> 11) / 9007199254740992.0; }

int main(void) {
    const int64_t n[3] = {37, 45, 53}; const size_t nv = (size_t)n[0] * n[1] * n[2];
    uint8_t *pr = malloc(nv), *ax = malloc(nv), *ay = malloc(nv), *bx = malloc(nv), *by = malloc(nv);
    for (size_t i = 0; i < nv; i++) pr[i] = (uint8_t)(rnd() * 40);
    for (int t = 0; t < 30; t++) {   /* tubes of radius ~1.5 along random directions (half near-horizontal, half near-vertical) */
        double c[3] = {rnd() * n[0], rnd() * n[1], rnd() * n[2]}, d[3];
        if (t & 1) { d[0] = 1; d[1] = 0.3 * (rnd() - 0.5); d[2] = 0.3 * (rnd() - 0.5); } else { d[0] = 0.2 * (rnd() - 0.5); d[1] = rnd() - 0.5; d[2] = rnd() - 0.5; }
        const double m = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]); for (int k = 0; k < 3; k++) d[k] /= m;
        for (double s = -60; s <= 60; s += 0.5) for (int dz = -2; dz <= 2; dz++) for (int dy = -2; dy <= 2; dy++) for (int dx = -2; dx <= 2; dx++) {
            const int64_t z = llround(c[0] + s * d[0]) + dz, y = llround(c[1] + s * d[1]) + dy, x = llround(c[2] + s * d[2]) + dx;
            if (z < 0 || y < 0 || x < 0 || z >= n[0] || y >= n[1] || x >= n[2] || dz * dz + dy * dy + dx * dx > 3) continue;
            const size_t i = ((size_t)z * n[1] + y) * n[2] + x; const int v = 160 + (int)(rnd() * 95); if (v > pr[i]) pr[i] = (uint8_t)v;
        }
    }
    for (int64_t z = 0; z < 6; z++) for (int64_t y = 0; y < 8; y++) for (int64_t x = 0; x < 8; x++) pr[((size_t)z * n[1] + y) * n[2] + x] = 200;   /* flat: tr 0 */
    int fails = 0, runs = 0;
    for (int R = 1; R <= 3; R++) for (int p = 0; p < 3; p++) {
        const char prior = p == 0 ? 0 : p == 1 ? 'H' : 'V';
        ft_ridge_axis(pr, n, ax, ay, R, prior);
        if (ft_ridge_axis_gpu(0, pr, n, bx, by, R, prior)) { printf("test_ftrace_gpu: no GPU path (CUDA failure)\n"); return 1; }
        size_t dx = 0, dy = 0, nd = 0; for (size_t i = 0; i < nv; i++) { dx += ax[i] != bx[i]; dy += ay[i] != by[i]; nd += ax[i] != 128 || ay[i] != 128; }
        runs++; if (dx || dy) fails++;
        printf("R %d prior %c: %zu / %zu nx and %zu ny bytes differ (%zu voxels with an axis)\n", R, prior ? prior : '-', dx, nv, dy, nd);
    }
    printf("test_ftrace_gpu: %s (%d of %d configurations byte-identical)\n", fails ? "FAIL" : "PASS", runs - fails, runs);
    free(pr); free(ax); free(ay); free(bx); free(by);
    return fails != 0;
}
