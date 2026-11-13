/* tests/test_ink_air.c: train --ink-air (nn_ink_air) against a host reference, byte-exact: after the smoothed ink targets of a slab,
   the unlabelled air voxels (papyrus plane known and < 127) on the S-voxel window lattice (z0 + z, y, x multiples of S) get the
   smoothed 0 target and mask 1; labelled voxels, papyrus, unknown papyrus and off-lattice voxels keep what nn_ink_targets_smooth wrote. */
#include "nn.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned rs = 12345;
static unsigned urand(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }
int main(void) {
    if (nn_init(0)) { fprintf(stderr, "test_ink_air: no GPU (skipped)\n"); return 0; }
    enum { NZ = 13, H = 37, W = 41 }; const size_t n = (size_t)NZ * H * W;
    uint8_t *ink = malloc(n), *pap = malloc(n), *t = malloc(n), *m = malloc(n), *rt = malloc(n), *rm = malloc(n);
    for (size_t i = 0; i < n; i++) {
        const unsigned r = urand() % 10; ink[i] = r < 7 ? 255 : (uint8_t)(urand() % 255);   /* 70 % unlabelled, else prob254 0..254 */
        const unsigned q = urand() % 6; static const uint8_t pv[6] = {0, 40, 126, 127, 200, 255}; pap[i] = pv[q];
    }
    uint8_t *dink = nn_malloc(n), *dpap = nn_malloc(n), *dt = nn_malloc(n), *dm = nn_malloc(n);
    nn_h2d(dink, ink, n); nn_h2d(dpap, pap, n);
    int fails = 0; const int Ss[3] = {1, 3, 4}, z0s[2] = {0, 22}; const float epss[2] = {0.f, 0.1f};
    for (int a = 0; a < 3; a++) for (int b = 0; b < 2; b++) for (int c = 0; c < 2; c++) {
        const int S = Ss[a], z0 = z0s[b]; const float eps = epss[c];
        nn_ink_targets_smooth(dink, n, dt, dm, eps); nn_d2h(rt, dt, n); nn_d2h(rm, dm, n);   /* (the device targets before: the reference's start) */
        nn_ink_air(dpap, n, z0, H, W, S, eps, dt, dm); nn_sync(); nn_d2h(t, dt, n); nn_d2h(m, dm, n);
        const uint8_t t0 = eps > 0 ? (uint8_t)lrintf(127.5f * eps) : 0; size_t nair = 0, bad = 0, nlab = 0;
        for (size_t i = 0; i < n; i++) {
            const int z = z0 + (int)(i / ((size_t)H * W)), y = (int)(i % ((size_t)H * W) / W), x = (int)(i % W);
            uint8_t et = rt[i], em = rm[i];
            if (!em && pap[i] != 255 && pap[i] < 127 && !(z % S) && !(y % S) && !(x % S)) { et = t0; em = 1; nair++; }
            nlab += rm[i] != 0;
            if (t[i] != et || m[i] != em) bad++;
            if (rm[i] && ink[i] == 255) bad++;   /* (the smoothed targets mask only the labelled voxels) */
        }
        fprintf(stderr, "ink air S %d z0 %d eps %.1f: %zu labelled, %zu air targets added, %zu bytes differ\n", S, z0, eps, nlab, nair, bad);
        if (bad || !nair) fails++;
    }
    nn_free(dink); nn_free(dpap); nn_free(dt); nn_free(dm); free(ink); free(pap); free(t); free(m); free(rt); free(rm);
    fprintf(stderr, "test_ink_air: %s\n", fails ? "FAIL" : "ok");
    return fails != 0;
}
