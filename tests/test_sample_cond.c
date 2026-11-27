/* train --cond-fields conditioning planes (src/sample_cond.h, 2026-11-27): a zarr fit-field set (h 8, 2.4 um) read into a level-0 window's
   8^3-voxel cells. Identity symmetry, cell-aligned corner: every cell is its field cell's decoded channels x 127 exactly; a corner half a
   field cell off: the mean of the 8 neighbours; cells beyond the set: 0. A constant set under a symmetry (y <-> x, z flipped): the winding
   mod 4 pairs follow the output axes (y reads the x plane) and the flipped axis's sine is negated. A second set covers what the first lacks.
   The decoder against hand values. */
#include "sample_cond.h"
#include "z3w.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)
static const char *pn[8] = {"pc", "ps", "w", "sp", "dq", "dz", "dy", "dx"};
enum { FD = 12, FH = 10, FW = 9 };   /* field cells */
static uint8_t F[8][FD * FH * FW];
static void mkset(const char *dir, const long long org[3], int w4, int constant) {
    mkdir(dir, 0755); char p[600];
    const int64_t sh[3] = {FD, FH, FW}; uint8_t *sb = calloc(128 * 128 * 128, 1);
    for (int k = 0; k < (w4 ? 8 : 4); k++) {
        for (int i = 0; i < FD * FH * FW; i++) F[k][i] = constant ? (uint8_t)(k == 0 ? 200 : k == 1 ? 60 : k == 4 ? 255 : k == 5 ? 64 : k == 6 ? 0 : k == 7 ? 32 : 90) : (uint8_t)((i * 37 + k * 101) % 251 + 1);
        snprintf(p, sizeof p, "%s/%s", dir, pn[k]);
        z3w *z = z3w_create(p, sh, 128, 0, k < 2 ? 128 : 0, nullptr); CHECK(z != nullptr, "z3w %s", pn[k]);
        memset(sb, k < 2 ? 128 : 0, 128 * 128 * 128);
        for (int a = 0; a < FD; a++) for (int b = 0; b < FH; b++) for (int c = 0; c < FW; c++) sb[((size_t)a * 128 + b) * 128 + c] = F[k][(a * FH + b) * FW + c];
        CHECK(z3w_write_shard(z, 0, 0, 0, sb, 1) == 0, "shard %s", pn[k]); z3w_close(z);
    }
    free(sb);
    snprintf(p, sizeof p, "%s/fields.json", dir); FILE *f = fopen(p, "w");
    fprintf(f, "{\"format\": \"zarr\", \"origin\": [%lld, %lld, %lld], \"shape\": [%d, %d, %d], \"h\": 8, \"um\": 2.4, \"w4_d\": %d, \"axis\": []}\n", org[0], org[1], org[2], FD, FH, FW, w4 ? 32 : 0);
    fclose(f);
}
static void cell(int z, int y, int x, uint8_t v[8]) { for (int k = 0; k < 8; k++) v[k] = F[k][(z * FH + y) * FW + x]; }
static int8_t q8(float v) { const long t = lroundf(v * 127.f); return (int8_t)(t > 127 ? 127 : t < -127 ? -127 : t); }

int main(void) {
    char tmp[] = "/tmp/test_sample_cond_XXXXXX"; if (!mkdtemp(tmp)) return 1;
    char d1[300], d2[300]; snprintf(d1, sizeof d1, "%s/a", tmp); snprintf(d2, sizeof d2, "%s/b", tmp);
    const int id[3] = {0, 1, 2}, nf[3] = {0, 0, 0};
    {   /* the decoder */
        const uint8_t v[8] = {255, 128, 51, 102, 255, 64, 128, 0}; float o[COND_K]; sample_cond_decode(v, 1, id, nf, o);
        CHECK(fabsf(o[0] - 1.f) < 1e-6f && fabsf(o[1]) < 1e-6f && fabsf(o[2] - 0.2f) < 1e-6f && fabsf(o[3] - 0.4f) < 1e-6f && fabsf(o[4] - 1.f) < 1e-6f, "decode scalars");
        CHECK(fabsf(o[5]) < 1e-6f && fabsf(o[6] - 1.f) < 1e-6f && fabsf(o[7] + 1.f) < 1e-6f && fabsf(o[8]) < 1e-6f && fabsf(o[9] - 1.f) < 1e-6f && fabsf(o[10]) < 1e-6f, "decode winding pairs");
        sample_cond_decode(v, 0, id, nf, o); CHECK(o[4] == 0.f && o[9] == 0.f, "no winding planes: 0");
    }
    const long long org[3] = {800, 1600, 2400};   /* native voxels (level 0, 2.4 um), field cell 8 voxels */
    mkset(d1, org, 1, 0);
    source s; memset(&s, 0, sizeof s); s.name = (char *)"t"; s.um = 2.4; s.ncond = 1; s.cond_dir[0] = d1;
    enum { Pz = 64, P = 96, G0 = Pz / 8, G1 = P / 8 };
    static int8_t out[COND_K * G0 * G1 * G1];
    const size_t ng = (size_t)G0 * G1 * G1;
    {   /* aligned: window corner = origin + 8 (2, 1, 3) - the cells (2, 1, 3) + g, the last ones beyond the set (FW 9: x cells 3 + g >= 9) */
        const int64_t o[3] = {org[0] + 16, org[1] + 8, org[2] + 24};
        CHECK(sample_cond_window(&s, 8, 0, o, Pz, P, id, nf, nullptr, out) == 0, "read");
        int bad = 0, outside = 0;
        for (int gz = 0; gz < G0; gz++) for (int gy = 0; gy < G1; gy++) for (int gx = 0; gx < G1; gx++) {
            const int cz = gz + 2, cy = gy + 1, cx = gx + 3; const size_t q = ((size_t)gz * G1 + gy) * G1 + gx;
            float e[COND_K] = {0};
            if (cz < FD && cy < FH && cx < FW) { uint8_t v[8]; cell(cz, cy, cx, v); sample_cond_decode(v, 1, id, nf, e); } else outside++;
            for (int c = 0; c < COND_K; c++) bad += out[c * ng + q] != q8(e[c]);
        }
        CHECK(!bad && outside > 0, "aligned cells: %d values differ (%d cells outside)", bad, outside);
    }
    {   /* half a cell off along every axis: the mean of the 8 neighbours (interior cells) */
        const int64_t o[3] = {org[0] + 4, org[1] + 4, org[2] + 4};
        CHECK(sample_cond_window(&s, 8, 0, o, Pz, P, id, nf, nullptr, out) == 0, "read");
        int bad = 0, n = 0;
        for (int gz = 0; gz + 1 < FD && gz < G0; gz++) for (int gy = 0; gy + 1 < FH && gy < G1; gy++) for (int gx = 0; gx + 1 < FW && gx < G1; gx++) {
            float e[COND_K] = {0};
            for (int a = 0; a < 8; a++) { uint8_t v[8]; float dv[COND_K]; cell(gz + (a >> 2), gy + ((a >> 1) & 1), gx + (a & 1), v); sample_cond_decode(v, 1, id, nf, dv); for (int c = 0; c < COND_K; c++) e[c] += 0.125f * dv[c]; }
            const size_t q = ((size_t)gz * G1 + gy) * G1 + gx; n++;
            for (int c = 0; c < COND_K; c++) bad += abs(out[c * ng + q] - q8(e[c])) > 1;
        }
        CHECK(!bad && n > 0, "half-cell offset: %d values off by more than 1 (%d cells)", bad, n);
    }
    {   /* a constant set under y <-> x with z flipped; a window beyond the first set, inside the second */
        mkset(d2, (long long[3]){0, 0, 0}, 1, 1);
        s.ncond = 2; s.cond_dir[1] = d2;
        const int pm[3] = {0, 2, 1}, fl[3] = {1, 0, 0};
        const int64_t o[3] = {8, 8, 0};
        CHECK(sample_cond_window(&s, 8, 0, o, Pz, P, pm, fl, nullptr, out) == 0, "read");
        const uint8_t v[8] = {200, 60, 90, 90, 255, 64, 0, 32}; float e[COND_K]; sample_cond_decode(v, 1, pm, fl, e);
        /* z: dz 64 (cos 0, sin 1 negated: -1); output y reads the x plane 32 (1/8 turn); output x reads the y plane 0 */
        CHECK(fabsf(e[5]) < 1e-6f && fabsf(e[6] + 1.f) < 1e-6f && fabsf(e[7] - (float)M_SQRT1_2) < 1e-6f && fabsf(e[9] - 1.f) < 1e-6f && fabsf(e[10]) < 1e-6f, "symmetry channel map");
        int bad = 0; const size_t q = ((size_t)2 * G1 + 3) * G1 + 4;   /* an interior cell */
        for (int c = 0; c < COND_K; c++) bad += out[c * ng + q] != q8(e[c]);
        CHECK(!bad, "constant set under the symmetry: %d channels differ", bad);
    }
    char cmd[400]; snprintf(cmd, sizeof cmd, "rm -rf %s", tmp); if (system(cmd)) {}
    if (fails) { fprintf(stderr, "test_sample_cond: %d FAILED\n", fails); return 1; }
    printf("test_sample_cond: ok\n");
    return 0;
}
