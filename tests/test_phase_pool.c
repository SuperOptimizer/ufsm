/* The pooled phase of the coarse scale rungs (2026-11-11, band_onfly.h phase_pool2_box / phase_pool2_view): windows one level coarser
   than a stored phase read its 2^3 circular mean. The labels derived from it must agree with the same labels derived at the fine level
   and pooled, where the 2^3 block is coherent (all 8 children known, resultant >= 0.9):
   - the side / phase: the circular difference coarse - mean(children) under 1.5 codes;
   - the normal (the phase gradient's direction, central circular differences): cos >= 0.95 with the children's mean gradient;
   - the winding mod 4 (band_wind4_tiled, the unwrap2 w4 target): coarse - child in turns is an integer per pair of coarse / fine
     components (tiles of 8 / 16 voxels: the gauges), the same integer for >= 99% of the pair's coherent voxels.
   Also: phase_pool2_view read == phase_pool2_box, and the z slabs (UFSM_PHASE_POOL_SLAB, set by the run to ~2 coarse planes per slab)
   give the same bytes as plane-by-plane pooling, the box crossing the volume's end. Spiral phase W = r / 16 + theta / 2 pi
   about a z axis on 160 x 96 x 224 level-0 voxels (unknown at r < 8). Usage: test_phase_pool DIR */
#include "band.h"
#include "band_onfly.h"
#include "zarr3.h"
#include "z3w.h"
#include "store.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

enum { NZ = 160, NY = 96, NX = 224 };
static uint8_t spiral(int64_t z, int64_t y, int64_t x) {
    (void)z; const double dy = y - 48.3, dx = x - 112.6, r = sqrt(dy * dy + dx * dx);
    if (r < 8) return PHASE_UNKNOWN;
    const double W = r / 16 + atan2(dy, dx) / (2 * M_PI);
    return (uint8_t)((int)lround((W - floor(W)) * PHASE_PERIOD) % PHASE_PERIOD);
}
static double cdiff(int a, int b) { int d = a - b; while (d > PHASE_PERIOD / 2) d -= PHASE_PERIOD; while (d < -PHASE_PERIOD / 2) d += PHASE_PERIOD; return d; }
/* the phase gradient (codes / voxel) at (z, y, x) of a D x H x W plane; 0 when a neighbour is unknown or out of range */
static int grad(const uint8_t *p, int D, int H, int W, int z, int y, int x, double g[3]) {
    const int c[3] = {z, y, x}, n[3] = {D, H, W}; const size_t st[3] = {(size_t)H * W, (size_t)W, 1};
    const size_t k = ((size_t)z * H + y) * W + x;
    for (int d = 0; d < 3; d++) {
        if (c[d] < 1 || c[d] + 1 >= n[d]) return 0;
        const uint8_t a = p[k + st[d]], b = p[k - st[d]]; if (a >= PHASE_PERIOD || b >= PHASE_PERIOD) return 0;
        g[d] = cdiff(a, b) / 2;
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: test_phase_pool DIR\n"); return 2; }
    char p[1400]; snprintf(p, sizeof p, "rm -rf '%s' && mkdir -p '%s'", argv[1], argv[1]); if (system(p)) return 1;
    snprintf(p, sizeof p, "%s/phase", argv[1]);
    const int64_t sh[3] = {NZ, NY, NX};
    {   char pa[1500]; snprintf(pa, sizeof pa, "%s/8", p); z3w *w = z3w_create(pa, sh, 128, 0, PHASE_UNKNOWN, nullptr); if (!w) { fprintf(stderr, "FAIL: z3w_create\n"); return 1; }
        uint8_t *b = malloc((size_t)128 * 128 * 128);
        for (int64_t sz = 0; sz * 128 < NZ; sz++) for (int64_t sy = 0; sy * 128 < NY; sy++) for (int64_t sx = 0; sx * 128 < NX; sx++) {
            for (int z = 0; z < 128; z++) for (int y = 0; y < 128; y++) for (int x = 0; x < 128; x++) {
                const int64_t gz = sz * 128 + z, gy = sy * 128 + y, gx = sx * 128 + x;
                b[((size_t)z * 128 + y) * 128 + x] = gz < NZ && gy < NY && gx < NX ? spiral(gz, gy, gx) : PHASE_UNKNOWN;
            }
            if (z3w_write_shard(w, sz, sy, sx, b, 2)) { fprintf(stderr, "FAIL: shard\n"); return 1; }
        }
        free(b); if (z3w_close(w)) return 1;
        const double um = 8; if (z3w_write_group(p, &um, 1, "phase", nullptr)) return 1; }
    store *st = store_open(p); z3 *ph = st ? z3_open(st, "8", nullptr) : nullptr;
    if (!ph) { fprintf(stderr, "FAIL: open %s\n", p); return 1; }
    const int D = 64, H = NY, W = NX, d = D / 2, h = H / 2, w = W / 2;
    const size_t nf = (size_t)D * H * W, nc = (size_t)d * h * w;
    uint8_t *f = malloc(nf), *c = malloc(nc), *cv = malloc(nc);
    const int64_t fo[3] = {32, 0, 0}, fn[3] = {D, H, W}, co[3] = {16, 0, 0}, cn[3] = {d, h, w};
    CHECK(!z3_read(ph, fo, fn, f, 1), "read fine");
    CHECK(!phase_pool2_box(ph, co, cn, c, 2, 0), "phase_pool2_box");
    z3 *view = phase_pool2_view(ph, 0); CHECK(view != nullptr, "phase_pool2_view");
    if (view) { CHECK(!z3_read(view, co, cn, cv, 2), "read view"); CHECK(!memcmp(c, cv, nc), "the view's read differs from phase_pool2_box"); z3_close(view); }
    /* coherent blocks */
    uint8_t *coh = calloc(nc, 1); size_t ncoh = 0, nknown = 0, nany = 0, badp = 0;
    for (int a = 0; a < d; a++) for (int b = 0; b < h; b++) for (int e = 0; e < w; e++) {
        double sx = 0, sy = 0; int k = 0;
        for (int q = 0; q < 8; q++) { const uint8_t v = f[((size_t)(2 * a + (q >> 2 & 1)) * H + 2 * b + (q >> 1 & 1)) * W + 2 * e + (q & 1)];
            if (v < PHASE_PERIOD) { sx += cos(2 * M_PI * v / PHASE_PERIOD); sy += sin(2 * M_PI * v / PHASE_PERIOD); k++; } }
        const size_t ci = ((size_t)a * h + b) * w + e;
        nany += k > 0; nknown += c[ci] < PHASE_PERIOD;
        if (k == 8 && hypot(sx, sy) / 8 >= 0.9) {
            coh[ci] = 1; ncoh++;
            double m = atan2(sy, sx) / (2 * M_PI); if (m < 0) m += 1;
            badp += c[ci] >= PHASE_PERIOD || fabs(cdiff(c[ci], (int)lround(m * PHASE_PERIOD) % PHASE_PERIOD)) > 1.5;
        }
    }
    /* normals */
    size_t nn = 0, badn = 0;
    for (int a = 1; a + 1 < d; a++) for (int b = 1; b + 1 < h; b++) for (int e = 1; e + 1 < w; e++) {
        if (!coh[((size_t)a * h + b) * w + e]) continue;
        double gc[3], gf[3] = {0, 0, 0}; int ok = grad(c, d, h, w, a, b, e, gc);
        for (int q = 0; q < 8 && ok; q++) { double g[3]; ok = grad(f, D, H, W, 2 * a + (q >> 2 & 1), 2 * b + (q >> 1 & 1), 2 * e + (q & 1), g); for (int k = 0; k < 3; k++) gf[k] += g[k]; }
        if (!ok) continue;
        const double lc = sqrt(gc[0] * gc[0] + gc[1] * gc[1] + gc[2] * gc[2]), lf = sqrt(gf[0] * gf[0] + gf[1] * gf[1] + gf[2] * gf[2]);
        if (lc < 1e-9 || lf < 1e-9) continue;
        nn++; badn += (gc[0] * gf[0] + gc[1] * gf[1] + gc[2] * gf[2]) / (lc * lf) < 0.95;
    }
    /* winding mod 4 */
    uint8_t *w4f = malloc(nf), *cf = malloc(nf), *w4c = malloc(nc), *cc = malloc(nc);
    CHECK(band_wind4_tiled(f, D, H, W, 16, 80, 64, 15, w4f, cf, nullptr) >= 0 && band_wind4_tiled(c, d, h, w, 8, 80, 8, 15, w4c, cc, nullptr) >= 0, "band_wind4_tiled");
    size_t (*hist)[255][4] = calloc(255, sizeof *hist), nw = 0, badr = 0;
    for (int a = 0; a < d; a++) for (int b = 0; b < h; b++) for (int e = 0; e < w; e++) {
        const size_t ci = ((size_t)a * h + b) * w + e, fi = ((size_t)(2 * a) * H + 2 * b) * W + 2 * e;
        if (!coh[ci] || cc[ci] == 255 || cf[fi] == 255) continue;
        double t = (w4c[ci] - w4f[fi]) / 63.0; t = fmod(fmod(t, 4) + 4, 4);
        const long it = lround(t); badr += fabs(t - it) > 0.1;
        hist[cc[ci]][cf[fi]][it & 3]++; nw++;
    }
    size_t agree = 0; for (int k = 0; k < 255; k++) for (int k2 = 0; k2 < 255; k2++) { size_t mx = 0; for (int j = 0; j < 4; j++) if (hist[k][k2][j] > mx) mx = hist[k][k2][j]; agree += mx; }
    free(hist);
    printf("pooled phase: %zu of %zu coarse voxels with a known child known after pooling (%.3f lost); coherent %zu: phase off %zu; "
           "normals %zu, cos < 0.95 %zu; w4 %zu, off-integer %zu, one gauge per component %.4f\n",
           nknown, nany, nany ? 1 - (double)nknown / nany : 0, ncoh, badp, nn, badn, nw, badr, nw ? (double)agree / nw : 0);
    CHECK(ncoh > nc / 4, "only %zu coherent coarse voxels", ncoh);
    CHECK(badp == 0, "%zu coherent voxels whose pooled phase is off the children's circular mean", badp);
    CHECK(nn > 1000 && badn <= nn / 100, "normals: %zu of %zu disagree", badn, nn);
    CHECK(nw > 1000 && badr <= nw / 100 && agree >= nw * 99 / 100, "w4: %zu voxels, %zu off-integer, %zu agree with one gauge per component", nw, badr, agree);
    /* the slabs (the run's UFSM_PHASE_POOL_SLAB: a few coarse planes each) against plane-by-plane pooling */
    {   const int64_t o2[3] = {0, 0, 0}, n2[3] = {NZ / 2, NY / 2, NX / 2}; const size_t n2v = (size_t)n2[0] * n2[1] * n2[2];
        uint8_t *all = malloc(n2v), *one = malloc((size_t)n2[1] * n2[2]); size_t diff = 0;
        CHECK(!phase_pool2_box(ph, o2, n2, all, 1, 0), "whole pool");
        for (int64_t z = 0; z < n2[0]; z++) { const int64_t oz[3] = {z, 0, 0}, nz1[3] = {1, n2[1], n2[2]}; phase_pool2_box(ph, oz, nz1, one, 1, 0);
            diff += memcmp(one, all + (size_t)z * n2[1] * n2[2], (size_t)n2[1] * n2[2]) != 0; }
        CHECK(diff == 0, "%zu coarse planes differ between the whole box and plane by plane", diff);
        free(all); free(one); }
    z3_close(ph);
    free(f); free(c); free(cv); free(coh); free(w4f); free(cf); free(w4c); free(cc);
    if (fails) { fprintf(stderr, "test_phase_pool: %d failure(s)\n", fails); return 1; }
    printf("test_phase_pool: OK\n");
    return 0;
}
