/* train --fib-head: nn_fib_head_dir (the direction targets of a z-slab from the H / V prob254 tube planes, window-indexed with halo
   planes) against nn_fib_dir_targets (--fib-dir: whole window, binary targets + a shared mask) on random tubes: with every voxel known the
   bytes of each 24-plane slab equal the whole window's; with unknown voxels (255), a voxel unknown in either plane gets no target.
   Usage: test_fibhead */
#include "nn.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void) {
    nn_init(0); int fails = 0;
    const int D = 72, H = 40, W = 48, r = 4; const size_t n = (size_t)D * H * W, hw = (size_t)H * W;
    uint8_t *t = calloc(2 * n, 1), *m = malloc(n); memset(m, 1, n); srand(11);
    for (int k = 0; k < 60; k++) {   /* straight tubes of radius ~1.5 along random directions, H or V */
        const int fam = rand() & 1; const double c[3] = {rand() % D, rand() % H, rand() % W}; double a[3] = {rand() / (double)RAND_MAX - .5, rand() / (double)RAND_MAX - .5, rand() / (double)RAND_MAX - .5};
        const double l = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); for (int q = 0; q < 3; q++) a[q] /= l;
        for (double s = -40; s <= 40; s += .25) for (int dz = -2; dz <= 2; dz++) for (int dy = -2; dy <= 2; dy++) for (int dx = -2; dx <= 2; dx++) {
            if (dz * dz + dy * dy + dx * dx > 2) continue;
            const int z = (int)(c[0] + s * a[0]) + dz, y = (int)(c[1] + s * a[1]) + dy, x = (int)(c[2] + s * a[2]) + dx;
            if (z >= 0 && z < D && y >= 0 && y < H && x >= 0 && x < W) t[fam * n + ((size_t)z * H + y) * W + x] = 254;
        }
    }
    uint8_t *td = nn_malloc(2 * n), *md = nn_malloc(n), *o = nn_malloc(7 * n), *om = nn_malloc(n), *os = nn_malloc(7 * n), *oms = nn_malloc(n);
    uint8_t *ho = malloc(7 * n), *hom = malloc(n), *hs = malloc(7 * n), *hsm = malloc(n);
    nn_h2d(td, t, 2 * n); nn_h2d(md, m, n);
    nn_fib_dir_targets(td, md, D, H, W, r, 0.6f, o, om); nn_sync(); nn_d2h(ho, o, 7 * n); nn_d2h(hom, om, n);
    size_t nt = 0; for (size_t i = 0; i < n; i++) nt += hom[i];
    for (int z0 = 0; z0 < D; z0 += 24) {   /* every slab against the window's planes */
        const int nz = z0 + 24 < D ? 24 : D - z0; const size_t ns = (size_t)nz * hw;
        nn_fib_head_dir(td, td + n, D, H, W, z0, nz, r, 0.6f, os, oms); nn_sync(); nn_d2h(hs, os, 7 * ns); nn_d2h(hsm, oms, ns);
        size_t bad = 0; for (size_t i = 0; i < ns; i++) { bad += hsm[i] != hom[(size_t)z0 * hw + i]; for (int c = 0; c < 7; c++) bad += hs[c * ns + i] != ho[c * n + (size_t)z0 * hw + i]; }
        if (bad) { fprintf(stderr, "FAIL: slab %d: %zu bytes differ from the whole window's\n", z0, bad); fails++; }
    }
    printf("fib head: %zu direction targets in the window, every slab equal: %s\n", nt, fails ? "no" : "yes");
    if (nt < 1000) { fprintf(stderr, "FAIL: only %zu targets\n", nt); fails++; }
    /* unknown voxels: a third of the H plane unknown -> no target there, and no H neighbour counted from it */
    for (size_t i = 0; i < n; i += 3) t[i] = 255;
    nn_h2d(td, t, 2 * n); nn_fib_head_dir(td, td + n, D, H, W, 0, D, r, 0.6f, os, oms); nn_sync(); nn_d2h(hsm, oms, n);
    size_t bad = 0, nk = 0; for (size_t i = 0; i < n; i += 3) bad += hsm[i] != 0; for (size_t i = 0; i < n; i++) nk += hsm[i];
    if (bad) { fprintf(stderr, "FAIL: %zu unknown voxels with a target\n", bad); fails++; }
    printf("fib head: with a third of H unknown, %zu targets, none on unknown voxels: %s\n", nk, bad ? "no" : "yes");
    if (fails) return 1;
    printf("test_fibhead: ok\n");
    return 0;
}
