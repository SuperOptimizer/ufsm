/* train --task unwrap2 sampler on a synthetic NON-CUBIC band + phase-only source (no recto / ink target: the multi-scroll sources the
   desk's wind2 run trains on; Y != X like PHerc0172, the source of its crash windows). CT 160 x 96 x 224 voxels (level 0) and 80 x 48 x 112
   (level 1); band codes and a stored phase on level 1, the phase known only in a slab of x (so many windows hold no phase at all); an axis
   outside the box. Windows of P 64 at level 0 with the trainer's settings (side 2, winding mod 4 targets, --gpu-input records, symmetries,
   rotation / elastic warps, deterministic), with --min-phase 0 and 0.01. Checks per window: phase codes < 252 or 255; w4 / comp both
   known or both 255, comp < 255, w4 < 252; the input finite; the radial record's header (corner within the volume, D = P = 64, symmetry
   bits); with --min-phase 0.01 every window holds >= 1% known phase, with 0 some hold none (the degenerate windows exist and are what
   the option removes). "phase_sign": -1 on a negated copy of the phase gives byte-identical phase / w4 / comp targets. Usage: test_unwrap2_sampler DIR (scratch directory for the synthetic source). */
#include "nn.h"
#include "sample.h"
#include "z3w.h"
#include "band.h"
#include "embz.h"
#include <sys/stat.h>
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
static uint8_t phase1neg(int64_t z, int64_t y, int64_t x) { const uint8_t q = phase1(z, y, x); return q < 252 ? (uint8_t)((252 - q) % 252) : q; }   /* W -> -W */
static uint8_t codes1(int64_t z, int64_t y, int64_t x) { return phase1(z, y, x) == 0 ? 37 : 0; }

static uint8_t *g_rec;   /* non-null: run() records every window's phase / w4 / comp (3 p3 per window) */
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
        if (g_rec) { uint8_t *r = g_rec + (size_t)i * 3 * p3; memcpy(r, b->side, p3); memcpy(r + p3, b->w4, p3); memcpy(r + 2 * p3, b->wcomp, p3); }
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
    /* "phase_sign": -1 (2026-10-19, the multi-scroll labels rising inward): the same source with its stored phase written negated
       (p -> (252 - p) mod 252) and read with phase_sign -1 trains byte-identical phase / winding mod 4 / component targets; read
       without it, the targets differ (the check sees the sign) */
    snprintf(p, sizeof p, "%s/phaseneg", dir);
    if (write_array(p, "16", s1, PHASE_UNKNOWN, phase1neg) || z3w_write_group(p, &um1, 1, "phase", nullptr)) { fprintf(stderr, "FAIL: negated phase\n"); return 1; }
    char q[1400]; const int nw = 24; const size_t win = (size_t)3 * 64 * 64 * 64;
    uint8_t *rec[3]; for (int k = 0; k < 3; k++) rec[k] = malloc(nw * win);
    for (int k = 0; k < 3; k++) {   /* 0 the original, 1 negated + phase_sign -1, 2 negated as is */
        snprintf(q, sizeof q, "%s/sources-sign%d.json", dir, k); f = fopen(q, "w");
        fprintf(f, "{\"sources\": [{\"name\": \"noncubic\", \"root\": \"%s\", \"ct\": \"ct\", \"um\": 8, \"max_level\": 0, \"weight\": 1,\n"
                   " \"targets\": {\"band\": {\"root\": \"%s/codes\", \"key\": \"16\", \"radius\": 80, \"span\": 75, \"phase\": {\"root\": \"%s/%s\", \"key\": \"16\", \"level\": 1%s}}},\n"
                   " \"axis\": \"%s/axis.json\"}]}\n", dir, dir, dir, k ? "phaseneg" : "phase", k == 1 ? ", \"phase_sign\": -1" : "", dir);
        fclose(f);
        sources *Sk = sources_load(q); if (!Sk) { fprintf(stderr, "FAIL: sources_load %s\n", q); return 1; }
        g_rec = rec[k]; size_t e, w; if (run(Sk, 0.01, nw, &e, &w)) return 1; g_rec = nullptr;
        sources_free(Sk);
    }
    size_t d01 = 0, d02 = 0, kn = 0;
    for (size_t k = 0; k < nw * win; k++) { d01 += rec[0][k] != rec[1][k]; d02 += rec[0][k] != rec[2][k]; }
    for (int i = 0; i < nw; i++) for (size_t k = 0; k < win / 3; k++) kn += rec[0][i * win + k] < 252;
    printf("phase_sign -1: %zu of %zu target bytes differ from the original (negated without the option: %zu; %zu known phase voxels)\n", d01, (size_t)nw * win, d02, kn);
    CHECK(kn > 0, "no known phase in the sign windows");
    CHECK(d01 == 0, "phase_sign -1 on the negated phase: %zu target bytes differ from the original", d01);
    CHECK(d02 > kn / 2, "the negated phase read as is differs in only %zu target bytes (the comparison cannot see the sign)", d02);
    for (int k = 0; k < 3; k++) free(rec[k]);
    snprintf(q, sizeof q, "%s/sources-bad.json", dir); f = fopen(q, "w");
    fprintf(f, "{\"sources\": [{\"name\": \"bad\", \"root\": \"%s\", \"ct\": \"ct\", \"um\": 8, \"max_level\": 0, \"weight\": 1,\n"
               " \"targets\": {\"band\": {\"root\": \"%s/codes\", \"key\": \"16\", \"phase\": {\"root\": \"%s/phase\", \"key\": \"16\", \"level\": 1, \"phase_sign\": 2}}}}]}\n", dir, dir, dir);
    fclose(f);
    sources *Sb = sources_load(q); CHECK(!Sb, "\"phase_sign\": 2 accepted"); if (Sb) sources_free(Sb);
    {   /* train --distill through the sampler: an "emb" group at level 1 (step 4) whose cell vectors encode their own position; without
           geometry augmentation every head cell (8^3 voxels at level 0) must carry the vector of its centre's position, with it the
           targets must stay unit vectors */
        enum { EC = 8, ST = 4 };
        const int64_t en[3] = {s1[0] / ST, s1[1] / ST, s1[2] / ST}, em[3] = {(en[0] + 15) / 16 * 16, (en[1] + 15) / 16 * 16, (en[2] + 15) / 16 * 16};
        snprintf(p, sizeof p, "%s/emb", dir); mkdir(p, 0755);
        snprintf(q, sizeof q, "%s/zarr.json", p); f = fopen(q, "w"); fputs("{\"zarr_format\":3,\"node_type\":\"group\",\"attributes\":{}}", f); fclose(f);
        snprintf(q, sizeof q, "%s/emb/pca", dir);
        zw w; char attrs[128]; snprintf(attrs, sizeof attrs, "{\"step_zyx\":[%d,%d,%d]}", ST, ST, ST);
        int8_t *buf = calloc((size_t)em[0] * em[1] * em[2] * EC, 1);
        for (int64_t z = 0; z < en[0]; z++) for (int64_t y = 0; y < en[1]; y++) for (int64_t x = 0; x < en[2]; x++) {
            const double e[3] = {z * ST + 1.5, y * ST + 1.5, x * ST + 1.5}; double v[EC];
            for (int d = 0; d < 3; d++) { v[2 * d] = cos(e[d] / 40 + d) / sqrt(3.0); v[2 * d + 1] = sin(e[d] / 40 + d) / sqrt(3.0); }
            v[6] = v[7] = 0;
            for (int c = 0; c < EC; c++) buf[(((size_t)z * em[1] + y) * em[2] + x) * EC + c] = (int8_t)lround(127 * v[c]);
        }
        CHECK(!zw_create(&w, q, en, EC, "int8", 1, 32, 16, attrs) && !zw_put(&w, 0, 0, 0, (int)em[0], (int)em[1], (int)em[2], buf) && !zw_flush(&w), "emb group");
        zw_free(&w); free(buf);
        snprintf(q, sizeof q, "%s/sources-emb.json", dir); f = fopen(q, "w");
        fprintf(f, "{\"sources\": [{\"name\": \"noncubic\", \"root\": \"%s\", \"ct\": \"ct\", \"um\": 8, \"max_level\": 0, \"weight\": 1,\n"
                   " \"targets\": {\"band\": {\"root\": \"%s/codes\", \"key\": \"16\", \"radius\": 80, \"span\": 75, \"phase\": {\"root\": \"%s/phase\", \"key\": \"16\", \"level\": 1}}},\n"
                   " \"emb\": {\"groups\": [\"%s/emb\"], \"name\": \"pca\", \"level\": 1}, \"axis\": \"%s/axis.json\"}]}\n", dir, dir, dir, dir, dir);
        fclose(f);
        sources *Se = sources_load(q); if (!Se) { fprintf(stderr, "FAIL: sources_load %s\n", q); return 1; }
        snprintf(q, sizeof q, "%s/sources-embonly.json", dir); f = fopen(q, "w");   /* an emb-only source ("ct_only", no targets, a box), as train sources-d2 */
        fprintf(f, "{\"sources\": [{\"name\": \"embonly\", \"root\": \"%s\", \"ct\": \"ct\", \"um\": 8, \"max_level\": 0, \"weight\": 1, \"ct_only\": 1, \"boxes\": [[0, 0, 0, 160, 96, 224]],\n"
                   " \"emb\": {\"groups\": [\"%s/emb\"], \"name\": \"pca\", \"level\": 1}}]}\n", dir, dir);
        fclose(f);
        sources *Su = sources_load(q); if (!Su) { fprintf(stderr, "FAIL: sources_load %s\n", q); return 1; }
        size_t lab = 0;   /* (emb-only windows: phase / w4 targets set anywhere) */
        for (int aug = 0; aug < 3; aug++) {   /* 0 plain, 1 augmented, 2 the emb-only source (augmented, --min-phase on) */
            sample_cfg c = sample_cfg_default();
            c.P = 64; c.B = 1; c.nworkers = 2; c.nbuf = 3; c.seed = 11; c.deterministic = 1; c.xfmt = 1; c.gpu_input = 1;
            memset(c.level_p, 0, sizeof c.level_p); c.level_p[0] = 1;
            c.side = 2; c.wind4 = 1; c.wind4_tile = 16; c.emb_ch = EC; c.emb_div = 8;
            if (aug) { c.augment = 3; c.symmetry_p = .5f; c.geometry_augment = 1; c.rotate_degrees = 5; c.rotate_p = .5f; c.elastic = 1; c.elastic_p = .5f; }
            else { c.augment = 0; c.symmetry_p = 0; c.geometry_augment = 0; c.rotate_p = 0; c.elastic_p = 0; }
            if (aug == 2) c.min_phase = .001;
            sampler *sp = sampler_start(aug == 2 ? Su : Se, &c); if (!sp) { fprintf(stderr, "FAIL: sampler_start (emb)\n"); return 1; }
            size_t nset = 0, nnorm = 0; double worst = 1;
            for (int i = 0; i < 16; i++) {
                batch *b = sampler_next(sp); if (!b) { CHECK(0, "emb sampler stopped"); break; }
                const size_t ng = 512;
                if (aug == 2) for (size_t k = 0; k < (size_t)64 * 64 * 64; k++) lab += b->side[k] != 255 || b->w4[k] != 255;
                for (size_t k = 0; k < ng; k++) {
                    double l2 = 0, dot = 0; for (int ch = 0; ch < EC; ch++) l2 += (double)b->emb[ch * ng + k] * b->emb[ch * ng + k];
                    if (l2 == 0) continue;
                    nset++; nnorm += fabs(sqrt(l2) - 127) > 2;
                    if (aug) continue;
                    const int gz = (int)(k / 64), gy = (int)(k / 8 % 8), gx = (int)(k % 8);
                    const double u[3] = {8 * gz + 3.5, 8 * gy + 3.5, 8 * gx + 3.5}; double v[EC]; int in = 1;
                    for (int d = 0; d < 3; d++) {
                        const double e = (b->corner[0][d] + u[d] + .5) * .5 - .5, cc = (e - 1.5) / ST;   /* level 0 -> level 1, cell centres */
                        in &= cc >= 0 && cc <= en[d] - 1;
                        v[2 * d] = cos(e / 40 + d) / sqrt(3.0); v[2 * d + 1] = sin(e / 40 + d) / sqrt(3.0);
                    }
                    v[6] = v[7] = 0;
                    if (!in) continue;
                    for (int ch = 0; ch < EC; ch++) dot += v[ch] * b->emb[ch * ng + k];
                    if (dot / sqrt(l2) < worst) worst = dot / sqrt(l2);
                }
                sampler_release(sp, b);
            }
            sampler_stop(sp);
            printf("distill targets (%s): %zu head cells set, worst cos %.5f, %zu not unit\n", aug == 2 ? "emb-only" : aug ? "augmented" : "plain", nset, worst, nnorm);
            CHECK(nset > 1000 && nnorm == 0 && worst > 0.999, "distill targets (%s): %zu set, worst cos %g, %zu not unit", aug == 2 ? "emb-only" : aug ? "augmented" : "plain", nset, worst, nnorm);
        }
        sources_free(Se); sources_free(Su);
        printf("emb-only windows: %zu phase / w4 target voxels\n", lab);
        CHECK(lab == 0, "an emb-only source produced %zu phase / w4 target voxels", lab);
    }
    if (fails) { fprintf(stderr, "test_unwrap2_sampler: %d failure(s)\n", fails); return 1; }
    printf("test_unwrap2_sampler: OK\n");
    return 0;
}
