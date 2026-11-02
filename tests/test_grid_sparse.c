/* fit_set_grid_sparse (the GT scorer's U at labelled voxels only, src/fit.c + nn_fit.cu fit_grid_sparse_k): on a fit set of a
   global model and 3 blended slabs, over a grid with partial last words and a random mask (empty words, full words, single bits),
   every packed value is bit-identical to the dense fit_set_grid's at its voxel, with_theta 0 and 1. */
#include "fit.h"
#include "nn.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)
static uint32_t hsh(uint32_t a) { a ^= a >> 16; a *= 0x7feb352du; a ^= a >> 15; a *= 0x846ca68bu; a ^= a >> 16; return a; }

static fit_model *mk(int seed, double z0, int D) {
    fit_model *m = calloc(1, sizeof *m);
    m->dir = 1; m->nk = 3; m->knot_dz = 64; m->pitch = malloc(12); m->offset = malloc(12);
    for (int k = 0; k < 3; k++) { m->pitch[k] = 20.f + 2 * k + seed; m->offset[k] = 0.1f * k - 0.05f * seed; }
    m->L = 16; m->D = D; m->H = 96; m->W = 112; m->h = 1; m->um = 9.6;
    m->lz = D / 16 + 2; m->ly = 96 / 16 + 2; m->lx = 112 / 16 + 2;
    m->origin[0] = (long long)z0;
    const int na = D + 1; m->cy = malloc(na * 4); m->cx = malloc(na * 4);
    for (int z = 0; z < na; z++) { m->cy[z] = 48.f + 0.05f * z; m->cx[z] = 50.f - 0.03f * z; }
    const size_t nl = (size_t)m->lz * m->ly * m->lx; m->delta = malloc(nl * 4);
    for (size_t i = 0; i < nl; i++) m->delta[i] = (float)((hsh((uint32_t)(i * 31 + seed)) & 1023) / 1023.0 - 0.5);
    return m;
}
int main(void) {
    if (nn_init(0)) { printf("SKIP: no GPU\n"); return 0; }
    fit_set s = {0}; s.global = mk(0, 0, 192); s.n = 3; s.blend = 6;
    s.slab = calloc(3, sizeof *s.slab); s.z0 = calloc(3, 8); s.z1 = calloc(3, 8);
    for (int k = 0; k < 3; k++) { s.slab[k] = mk(k + 1, 64.0 * k, 64); s.z0[k] = 64.0 * k; s.z1[k] = 64.0 * (k + 1); }
    fit_set_dev *g = fit_set_upload(&s); CHECK(g, "fit_set_upload");
    if (!g) return 1;
    const int64_t n[3] = {37, 51, 77};   /* 145299 points: the last word partial */
    const double o[3] = {3.75, -2.25, 1.75}; const double step = 2.5;
    const size_t nv = (size_t)n[0] * n[1] * n[2], nw = (nv + 63) / 64;
    float *dense = malloc(nv * 4), *pk = malloc(nv * 4); uint64_t *m = calloc(nw, 8); uint32_t *wo = calloc(nw, 4);
    for (int wt = 0; wt < 2; wt++) {
        CHECK(!fit_set_grid(g, &s, o, n, step, wt, dense), "dense grid");
        memset(m, 0, nw * 8); size_t nl = 0;
        for (size_t w = 0; w < nw; w++) {
            wo[w] = (uint32_t)nl; const int kind = (int)(hsh((uint32_t)w + 99) % 4);
            for (int j = 0; j < 64 && w * 64 + j < nv; j++) {
                const int on = kind == 0 ? 0 : kind == 1 ? 1 : kind == 2 ? (hsh((uint32_t)(w * 64 + j)) & 3) == 0 : j == 37;
                if (on) { m[w] |= 1ull << j; nl++; }
            }
        }
        memset(pk, 0xff, nv * 4);
        CHECK(!fit_set_grid_sparse(g, &s, o, n, step, wt, m, wo, nl, pk), "sparse grid");
        size_t k = 0, bad = 0;
        for (size_t i = 0; i < nv; i++) if (m[i / 64] >> (i % 64) & 1) { if (memcmp(&pk[k], &dense[i], 4)) bad++; k++; }
        CHECK(k == nl && !bad, "with_theta %d: %zu of %zu packed values differ from the dense grid's", wt, bad, nl);
        printf("grid_sparse with_theta %d: %zu labelled of %zu points, %zu differ\n", wt, nl, nv, bad);
    }
    fit_set_dev_free(g); free(dense); free(pk); free(m); free(wo);
    printf("test_grid_sparse: %s\n", fails ? "FAILED" : "ok");
    return fails != 0;
}
