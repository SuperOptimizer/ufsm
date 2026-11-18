/* The voxel unwrapping's node fill (src/fit.c fit_node_fill, 2026-11-18): the frontier version against the per-pass sweep it replaces,
   byte for byte (corr and have) on random lattices: sparse and dense seeds, blobs far apart (the 64-pass cap reached, unreached nodes
   left 0), a pass cap of 1 and 3, lattice edges of 1. CPU only. */
#include "fit.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
static uint64_t rs = 0x9e3779b97f4a7c15ull;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

static void one(int lz, int ly, int lx, double dens, int blobs, int maxpass) {
    const size_t nl = (size_t)lz * ly * lx; uint8_t *h0 = calloc(nl, 1), *h1 = calloc(nl, 1); double *c0 = calloc(nl, sizeof(double)), *c1 = calloc(nl, sizeof(double));
    for (size_t k = 0; k < nl; k++) if ((double)(rnd() % 1000000) / 1e6 < dens) { h0[k] = 1; c0[k] = (double)(int64_t)(rnd() % 2000001 - 1000000) / 7919.0; }
    for (int b = 0; b < blobs; b++) {   /* a small solid block of filled nodes */
        const int z = (int)(rnd() % lz), y = (int)(rnd() % ly), x = (int)(rnd() % lx);
        for (int dz = 0; dz < 2 && z + dz < lz; dz++) for (int dy = 0; dy < 3 && y + dy < ly; dy++) for (int dx = 0; dx < 3 && x + dx < lx; dx++) {
            const size_t k = ((size_t)(z + dz) * ly + y + dy) * lx + x + dx; h0[k] = 1; c0[k] = (double)(rnd() % 100000) / 977.0 - 50;
        }
    }
    memcpy(h1, h0, nl); memcpy(c1, c0, nl * sizeof(double));
    const int p0 = fit_node_fill(lz, ly, lx, h0, c0, maxpass, 1), p1 = fit_node_fill(lz, ly, lx, h1, c1, maxpass, 0);
    size_t filled = 0; for (size_t k = 0; k < nl; k++) filled += h0[k] == 1;
    const int ok = !memcmp(h0, h1, nl) && !memcmp(c0, c1, nl * sizeof(double)) && p0 == p1;
    if (!ok) { fprintf(stderr, "FAIL: fill %dx%dx%d dens %g blobs %d maxpass %d: passes %d vs %d, have %s, corr %s\n", lz, ly, lx, dens, blobs, maxpass, p0, p1,
                       memcmp(h0, h1, nl) ? "differ" : "same", memcmp(c0, c1, nl * sizeof(double)) ? "differ" : "same"); fails++; }
    else printf("ok: fill %dx%dx%d dens %g blobs %d maxpass %d: %d passes, %zu of %zu filled\n", lz, ly, lx, dens, blobs, maxpass, p0, filled, nl);
    free(h0); free(h1); free(c0); free(c1);
}

int main(void) {
    one(10, 60, 50, 0.02, 0, 64);
    one(10, 60, 50, 0.3, 0, 64);
    one(11, 200, 180, 0, 2, 64);      /* far apart blobs: the cap reached, the corners stay 0 */
    one(11, 200, 180, 0.0005, 3, 3);
    one(5, 40, 40, 0.01, 1, 1);
    one(1, 1, 300, 0.05, 0, 64);
    one(1, 90, 1, 0, 1, 64);
    one(6, 30, 30, 0, 0, 64);         /* nothing to fill from */
    one(6, 30, 30, 1, 0, 64);         /* everything filled already */
    if (fails) { fprintf(stderr, "test_uwfill: %d failure(s)\n", fails); return 1; }
    printf("test_uwfill: ok\n"); return 0;
}
