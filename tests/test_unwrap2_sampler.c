/* train --task unwrap2 sampler on a synthetic NON-CUBIC band + phase-only source (no recto / ink target: the multi-scroll sources the
   desk's wind2 run trains on; Y != X like PHerc0172, the source of its crash windows). CT 160 x 96 x 224 voxels (level 0) and 80 x 48 x 112
   (level 1); band codes and a stored phase on level 1, the phase known only in a slab of x (so many windows hold no phase at all); an axis
   outside the box. Windows of P 64 at level 0 with the trainer's settings (side 2, winding mod 4 targets, --gpu-input records, symmetries,
   rotation / elastic warps, deterministic), with --min-phase 0 and 0.01. Checks per window: phase codes < 252 or 255; w4 / comp both
   known or both 255, comp < 255, w4 < 252; the input finite; the radial record's header (corner within the volume, D = P = 64, symmetry
   bits); with --min-phase 0.01 every window holds >= 1% known phase, with 0 some hold none (the degenerate windows exist and are what
   the option removes). Usage: test_unwrap2_sampler DIR (scratch directory for the synthetic source). */
#include "nn.h"
#include "sample.h"
#include "z3w.h"
#include "band.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

static int write_array(const char *dir, const char *name, const int64_t sh[3], uint8_t fill, uint8_t (*val)(int64_t, int64_t, int64_t)) {
    char p[1400]; snprintf(p, sizeof p, "%s/%s", dir, name);
    z3w *w = z3w_create(p, sh, 128, 0, fill, nullptr); if (!w) return -1;
    uint8_t *b = malloc((size_t)128 * 128 * 128);
    for (int64_t sz = 0; sz * 128 < sh[0]; sz++) for (int64_t sy = 0; sy * 128 < sh[1]; sy++) for (int64_t sx = 0; sx * 128 < sh[2]; sx++) {
        for (int z = 0; z < 128; z++) for (int y = 0; y < 128; y++) for (int x = 0; x < 128; x++) {
            const int64_t gz = sz * 128 + z, gy = sy * 128 + y, gx = sx * 128 + x;
            b[((size_t)z * 128 + y) * 128 + x] = gz < sh[0] && gy < sh[1] && gx < sh[2] ? val(gz, gy, gx) : fill;
        }
        if (z3w_write_shard(w, sz, sy, sx, b, 2)) { free(b); return -1; }
    }
    free(b);
    return z3w_close(w);
}
static uint8_t ct0(int64_t z, int64_t y, int64_t x) { return (uint8_t)(60 + (z * 7 + y * 13 + x * 5) % 150); }
static uint8_t ct1(int64_t z, int64_t y, int64_t x) { return ct0(2 * z, 2 * y, 2 * x); }
static uint8_t phase1(int64_t z, int64_t y, int64_t x) {   /* level 1: W = 0.09 x + 0.03 y (turns), known for x in [40, 80) */
    (void)z; if (x < 40 || x >= 80) return PHASE_UNKNOWN;
    const double W = 0.09 * x + 0.03 * y; return (uint8_t)((int)lround((W - floor(W)) * 252) % 252);
}
static uint8_t codes1(int64_t z, int64_t y, int64_t x) { return phase1(z, y, x) == 0 ? 37 : 0; }

static int run(sources *S, double min_phase, int nwin, size_t *empty, size_t *warped) {
    sample_cfg c = sample_cfg_default();
    c.P = 64; c.B = 1; c.nworkers = 3; c.nbuf = 3; c.seed = 7; c.deterministic = 1; c.xfmt = 1; c.gpu_input = 1;
    memset(c.level_p, 0, sizeof c.level_p); c.level_p[0] = 1;
    c.side = 2; c.wind4 = 1; c.wind4_tile = 16; c.min_phase = min_phase;
    c.augment = 3; c.symmetry_p = .5f; c.geometry_augment = 1; c.rotate_degrees = 5; c.rotate_p = .5f; c.elastic = 1; c.elastic_p = .5f;
    sampler *sp = sampler_start(S, &c);
    if (!sp) { fprintf(stderr, "FAIL: sampler_start\n"); return 1; }
    const size_t p3 = (size_t)64 * 64 * 64;
    *empty = *warped = 0;
    for (int i = 0; i < nwin; i++) {
        batch *b = sampler_next(sp);
        if (!b) { fprintf(stderr, "FAIL: sampler stopped\n"); fails++; break; }
        size_t kn = 0, badp = 0, badw = 0, badx = 0;
        for (size_t k = 0; k < p3; k++) {
            const uint8_t ph = b->side[k], w = b->w4[k], cp = b->wcomp[k];
            kn += ph < 252; badp += ph >= 252 && ph != 255;
            badw += (w == 255) != (cp == 255) || (cp != 255 && w >= 252);
            badx += (b->x16[k] & 0x7c00) == 0x7c00;
        }
        const int32_t *h = (const int32_t *)b->rad;
        const int hdr_ok = h[2] == 64 && h[3] == 64 && h[0] >= 0 && h[0] + 64 <= 96 && h[1] >= 0 && h[1] + 64 <= 224 && (h[6] & ~3) == 0;
        *warped += h[6] & 1;
        if (!kn) (*empty)++;
        CHECK(!badp && !badw && !badx && hdr_ok, "min-phase %g window %d (corner %lld,%lld,%lld): %zu bad phase codes, %zu bad w4 / comp, %zu non-finite inputs, record %d %d %d %d %d",
              min_phase, i, (long long)b->corner[0][0], (long long)b->corner[0][1], (long long)b->corner[0][2], badp, badw, badx, h[0], h[1], h[2], h[3], h[6]);
        CHECK(min_phase <= 0 || (double)kn >= min_phase * (double)p3, "min-phase %g: a window with %zu known phase voxels", min_phase, kn);
        sampler_release(sp, b);
    }
    sampler_stop(sp);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: test_unwrap2_sampler DIR\n"); return 2; }
    nn_init(0);
    const char *dir = argv[1]; char p[1400];
    snprintf(p, sizeof p, "rm -rf '%s'", dir); if (system(p)) { fprintf(stderr, "FAIL: %s\n", p); return 1; }   /* (a fresh fixture) */
    const int64_t s0[3] = {160, 96, 224}, s1[3] = {80, 48, 112};
    const double um[2] = {8, 16}, um1 = 16;
    snprintf(p, sizeof p, "%s/ct", dir);
    if (write_array(p, "8", s0, 0, ct0) || write_array(p, "16", s1, 0, ct1) || z3w_write_group(p, um, 2, "ct", nullptr)) { fprintf(stderr, "FAIL: CT\n"); return 1; }
    snprintf(p, sizeof p, "%s/codes", dir);
    if (write_array(p, "16", s1, 0, codes1) || z3w_write_group(p, &um1, 1, "codes", nullptr)) { fprintf(stderr, "FAIL: codes\n"); return 1; }
    snprintf(p, sizeof p, "%s/phase", dir);
    if (write_array(p, "16", s1, PHASE_UNKNOWN, phase1) || z3w_write_group(p, &um1, 1, "phase", nullptr)) { fprintf(stderr, "FAIL: phase\n"); return 1; }
    snprintf(p, sizeof p, "%s/axis.json", dir); FILE *f = fopen(p, "w");
    fprintf(f, "{\"control_points\": [{\"z\": 0, \"y\": -300.5, \"x\": 110.25}, {\"z\": 159, \"y\": -280.0, \"x\": 120.0}]}\n"); fclose(f);
    snprintf(p, sizeof p, "%s/sources.json", dir); f = fopen(p, "w");
    fprintf(f, "{\"sources\": [{\"name\": \"noncubic\", \"root\": \"%s\", \"ct\": \"ct\", \"um\": 8, \"max_level\": 0, \"weight\": 1,\n"
               " \"targets\": {\"band\": {\"root\": \"%s/codes\", \"key\": \"16\", \"radius\": 80, \"span\": 75, \"phase\": {\"root\": \"%s/phase\", \"key\": \"16\", \"level\": 1}}},\n"
               " \"axis\": \"%s/axis.json\"}]}\n", dir, dir, dir, dir);
    fclose(f);
    sources *S = sources_load(p);
    if (!S) { fprintf(stderr, "FAIL: sources_load %s\n", p); return 1; }
    size_t e0, w0, e1, w1;
    if (run(S, 0, 48, &e0, &w0) || run(S, 0.01, 48, &e1, &w1)) return 1;
    printf("min-phase 0: %zu of 48 windows without phase (%zu warped); min-phase 0.01: %zu (%zu warped)\n", e0, w0, e1, w1);
    CHECK(e0 > 0, "no phase-free windows drawn with --min-phase 0 (the test does not reach the degenerate case)");
    CHECK(e1 == 0, "phase-free windows with --min-phase 0.01");
    CHECK(w0 + w1 > 0, "no warped window");
    if (fails) { fprintf(stderr, "test_unwrap2_sampler: %d failure(s)\n", fails); return 1; }
    printf("test_unwrap2_sampler: OK\n");
    return 0;
}
