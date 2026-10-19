/* A synthetic band + stored-phase source (the --task unwrap2 desk sources' layout: CT pyramid, band codes and a stored phase on the
   level-1 grid, an axis, no recto / ink targets) for sampler benchmarks and tests: make_wind_fixture DIR [N 1024] [q 2]
   DIR/ct (levels 0..4, N^3 at level 0, volcomp q lossy), DIR/band/1 and DIR/phase/1 ((N/2)^3), DIR/axis.json, DIR/sources.json (phase level 1); DIR/phase2/2 and DIR/sources2.json (phase level 2, a holdout box).
   A spiral scroll (radius 0.47 N, sheet pitch 24 voxels) around a slightly tilted axis; the phase is the winding's fraction where the
   labels reach (radius < 0.44 N, outside a few unlabelled blobs), unknown elsewhere; CT 0 outside the scroll. */
#include "z3w.h"
#include "band.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fx_N;
static void fx_axis_at(double z, double *cy, double *cx) { *cy = .5 * fx_N + .03 * (z - .5 * fx_N); *cx = .5 * fx_N - .02 * (z - .5 * fx_N); }
static double fx_wind(double z, double y, double x) {   /* winding number (turns) of a point; fx_N voxels at level 0 */
    double cy, cx; fx_axis_at(z, &cy, &cx);
    const double r = hypot(y - cy, x - cx), th = atan2(y - cy, x - cx);
    return r / 24.0 - th / (2 * M_PI) + .002 * z;
}
static uint8_t fx_ct_at(int z, int y, int x) {
    double cy, cx; fx_axis_at(z, &cy, &cx);
    const double r = hypot(y - cy, x - cx);
    if (r > .47 * fx_N) return 0;
    const double w = fx_wind(z, y, x), f = w - floor(w);
    const uint32_t h = (uint32_t)(((uint64_t)z * 73856093u) ^ ((uint64_t)y * 19349663u) ^ ((uint64_t)x * 83492791u)) * 2654435761u;
    const int noise = (int)(h >> 27) - 16;
    const int v = f < .3 ? 150 + noise : 45 + noise / 2;
    return (uint8_t)(v < 1 ? 1 : v > 255 ? 255 : v);
}
static int fx_write_arr(const char *dir, const int64_t shape[3], float q, int fill, uint8_t *(*gen)(int64_t, int64_t, int64_t, int, void *), void *ctx) {
    const int S = 512;
    z3w *w = z3w_create(dir, shape, S, q, fill, nullptr);
    if (!w) { fprintf(stderr, "%s: %s\n", dir, z3w_error()); return -1; }
    for (int64_t sz = 0; sz * S < shape[0]; sz++) for (int64_t sy = 0; sy * S < shape[1]; sy++) for (int64_t sx = 0; sx * S < shape[2]; sx++) {
        uint8_t *d = gen(sz * S, sy * S, sx * S, S, ctx);
        if (!d || z3w_write_shard(w, sz, sy, sx, d, 8)) { fprintf(stderr, "%s: %s\n", dir, z3w_error()); return -1; }
        free(d);
    }
    return z3w_close(w);
}
typedef struct { const uint8_t *vol; int64_t n; } fx_volctx;   /* a whole level held in memory (cut into shards) */
static uint8_t *fx_cut(int64_t z0, int64_t y0, int64_t x0, int S, void *p) {
    const fx_volctx *c = p; uint8_t *d = calloc((size_t)S * S * S, 1);
    for (int64_t z = 0; z < S && z0 + z < c->n; z++) for (int64_t y = 0; y < S && y0 + y < c->n; y++) {
        const int64_t nx = c->n - x0 < S ? c->n - x0 : S;
        if (nx > 0) memcpy(d + ((size_t)z * S + y) * S, c->vol + ((size_t)(z0 + z) * c->n + y0 + y) * c->n + x0, (size_t)nx);
    }
    return d;
}
static int make_wind_fixture(const char *D, int n_, float q) {
    fx_N = n_;
    if (fx_N < 256 || fx_N % 32) { fprintf(stderr, "make_wind_fixture: N a multiple of 32, >= 256\n"); return 2; }
    char p[1400];
    /* CT level 0 .. 4 (mean of 2^3 per level) */
    size_t n = (size_t)fx_N; uint8_t *lv = malloc(n * n * n);
    if (!lv) return 1;
    for (int z = 0; z < fx_N; z++) for (int y = 0; y < fx_N; y++) for (int x = 0; x < fx_N; x++) lv[((size_t)z * fx_N + y) * fx_N + x] = fx_ct_at(z, y, x);
    double um[5];
    for (int l = 0; l < 5; l++) {
        um[l] = 9.6 * (1 << l);
        char nm[64]; z3w_level_name(um[l], nm, sizeof nm); snprintf(p, sizeof p, "%s/ct/%s", D, nm);
        const int64_t sh[3] = {(int64_t)n, (int64_t)n, (int64_t)n}; fx_volctx c = {lv, (int64_t)n};
        if (fx_write_arr(p, sh, q, 0, fx_cut, &c)) return 1;
        if (l == 4) break;
        const size_t h = n / 2; uint8_t *nx = malloc(h * h * h);
        for (size_t z = 0; z < h; z++) for (size_t y = 0; y < h; y++) for (size_t x = 0; x < h; x++) {
            int s = 0;
            for (int k = 0; k < 8; k++) s += lv[((2 * z + (k >> 2)) * n + 2 * y + (k >> 1 & 1)) * n + 2 * x + (k & 1)];
            nx[(z * h + y) * h + x] = (uint8_t)((s + 4) / 8);
        }
        free(lv); lv = nx; n = h;
    }
    free(lv);
    snprintf(p, sizeof p, "%s/ct", D);
    if (z3w_write_group(p, um, 5, "ct", nullptr)) return 1;
    /* band codes and the stored phase on the level-1 grid (voxel j at level-0 2 j + 0.5) */
    const size_t h = (size_t)fx_N / 2; uint8_t *bd = malloc(h * h * h), *ph = malloc(h * h * h);
    for (size_t z = 0; z < h; z++) for (size_t y = 0; y < h; y++) for (size_t x = 0; x < h; x++) {
        const double Z = 2.0 * z + .5, Y = 2.0 * y + .5, X = 2.0 * x + .5; double cy, cx; fx_axis_at(Z, &cy, &cx);
        const double r = hypot(Y - cy, X - cx), w = fx_wind(Z, Y, X), f = w - floor(w);
        const int blob = (((int)(Z / 96) * 7 + (int)(Y / 96) * 3 + (int)(X / 96)) % 5) == 0;   /* unlabelled boxes */
        const size_t k = (z * h + y) * h + x;
        const int known = r < .44 * fx_N && !blob;
        bd[k] = known ? (uint8_t)(1 + ((int)floor(w) & 63)) : 0;
        int c = (int)(f * PHASE_PERIOD); if (c >= PHASE_PERIOD) c = PHASE_PERIOD - 1;
        ph[k] = known ? (uint8_t)c : PHASE_UNKNOWN;
    }
    const int64_t hs[3] = {(int64_t)h, (int64_t)h, (int64_t)h};
    fx_volctx cb = {bd, (int64_t)h}, cp = {ph, (int64_t)h};
    snprintf(p, sizeof p, "%s/band/1", D); if (fx_write_arr(p, hs, 0.f, 0, fx_cut, &cb)) return 1;
    snprintf(p, sizeof p, "%s/phase/1", D); if (fx_write_arr(p, hs, 0.f, PHASE_UNKNOWN, fx_cut, &cp)) return 1;
    free(bd); free(ph);
    {   /* the phase one level coarser (level 2: voxel j at level-0 4 j + 1.5), for sources2.json (phase level 2: interpolated into
           level-0 windows, read directly by level-2 windows) */
        const size_t g = h / 2; uint8_t *p2 = malloc(g * g * g);
        for (size_t z = 0; z < g; z++) for (size_t y = 0; y < g; y++) for (size_t x = 0; x < g; x++) {
            const double Z = 4.0 * z + 1.5, Y = 4.0 * y + 1.5, X = 4.0 * x + 1.5; double cy, cx; fx_axis_at(Z, &cy, &cx);
            const double r = hypot(Y - cy, X - cx), w = fx_wind(Z, Y, X), f = w - floor(w);
            const int blob = (((int)(Z / 96) * 7 + (int)(Y / 96) * 3 + (int)(X / 96)) % 5) == 0;
            int c = (int)(f * PHASE_PERIOD); if (c >= PHASE_PERIOD) c = PHASE_PERIOD - 1;
            p2[(z * g + y) * g + x] = r < .44 * fx_N && !blob ? (uint8_t)c : PHASE_UNKNOWN;
        }
        const int64_t gs[3] = {(int64_t)g, (int64_t)g, (int64_t)g}; fx_volctx c2 = {p2, (int64_t)g};
        snprintf(p, sizeof p, "%s/phase2/2", D); if (fx_write_arr(p, gs, 0.f, PHASE_UNKNOWN, fx_cut, &c2)) return 1;
        free(p2);
    }
    snprintf(p, sizeof p, "%s/axis.json", D); FILE *f = fopen(p, "w"); if (!f) return 1;
    fprintf(f, "{\"control_points\": [");
    for (int i = 0; i <= 8; i++) { const double z = (double)fx_N * i / 8; double cy, cx; fx_axis_at(z, &cy, &cx); fprintf(f, "%s{\"z\": %g, \"y\": %g, \"x\": %g}", i ? ", " : "", z, cy, cx); }
    fprintf(f, "]}\n"); fclose(f);
    snprintf(p, sizeof p, "%s/sources.json", D); f = fopen(p, "w"); if (!f) return 1;
    fprintf(f, "{\"sources\": [{\"name\": \"wind-fixture\", \"root\": \"%s\", \"ct\": \"ct\", \"um\": 9.6, \"axis\": \"%s/axis.json\", \"weight\": 1,\n"
               "  \"targets\": {\"band\": {\"root\": \"%s/band\", \"key\": \"1\", \"radius\": 80, \"span\": 75, \"phase\": {\"root\": \"%s/phase\", \"key\": \"1\", \"level\": 1}}}}]}\n", D, D, D, D);
    fclose(f);
    snprintf(p, sizeof p, "%s/sources2.json", D); f = fopen(p, "w"); if (!f) return 1;   /* + a holdout box, the phase at level 2 */
    fprintf(f, "{\"sources\": [{\"name\": \"wind-fixture2\", \"root\": \"%s\", \"ct\": \"ct\", \"um\": 9.6, \"axis\": \"%s/axis.json\", \"weight\": 1,\n"
               "  \"holdout\": [0, 0, 0, %d, %d, %d],\n"
               "  \"targets\": {\"band\": {\"root\": \"%s/band\", \"key\": \"1\", \"radius\": 80, \"span\": 75, \"phase\": {\"root\": \"%s/phase2\", \"key\": \"2\", \"level\": 2}}}}]}\n",
            D, D, fx_N / 4, fx_N / 4, fx_N / 4, D, D);
    fclose(f);
    return 0;
}
#ifndef WIND_FIXTURE_NO_MAIN
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: make_wind_fixture DIR [N 1024] [q 2]\n"); return 2; }
    return make_wind_fixture(argv[1], argc > 2 ? atoi(argv[2]) : 1024, argc > 3 ? (float)atof(argv[3]) : 2.f);
}
#endif
