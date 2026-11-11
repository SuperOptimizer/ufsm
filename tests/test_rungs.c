/* train --um-rungs (sample.h nrung): levels picked by physical voxel size. Two synthetic scans of the same pattern, A at 2.4 um (levels
   2.4 / 4.8 / 9.6) and B at 7.9 um (levels 7.9 / 15.8), each with a binary recto target on every level; windows of P 16, deterministic.
   (1) rungs 9.6: A reads level 2, B level 0 (7.9 um is 0.28 octave from 9.6), every window; (2) rungs 4.8,9.6 at equal weights: A
   draws both its levels 1 and 2, B only level 0 (4.8 is 0.72 octave from 7.9, outside the 0.5 tolerance), and the windows of A split about
   evenly; (3) rungs 2.4,4.8 with --rung-shift-p 1: every draw tries a list neighbour (a missing one keeps the rung), so A trains both
   levels 0 and 1 and B, which reaches neither rung, is excluded; (4) rungs 38.4: no source reaches it, sampler_start fails. Usage: test_rungs DIR (scratch directory) */
#include "nn.h"
#include "sample.h"
#include "z3w.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

static int write_array(const char *dir, const char *name, int64_t n, int tgt, int sh) {
    char p[1400]; snprintf(p, sizeof p, "%s/%s", dir, name);
    const int64_t s[3] = {n, n, n}; z3w *w = z3w_create(p, s, 128, 0, 0, nullptr); if (!w) return -1;
    uint8_t *b = calloc((size_t)128 * 128 * 128, 1);
    for (int z = 0; z < 128; z++) for (int y = 0; y < 128; y++) for (int x = 0; x < 128; x++) {
        const int gz = z << sh, gy = y << sh, gx = x << sh;
        b[((size_t)z * 128 + y) * 128 + x] = z < n && y < n && x < n ? (tgt ? ((gz / 8) % 2 ? 255 : 0) : (uint8_t)(60 + (gz * 7 + gy * 13 + gx * 5) % 150)) : 0;
    }
    const int r = z3w_write_shard(w, 0, 0, 0, b, 2); free(b);
    return r || z3w_close(w);
}
static int scan(const char *dir, const char *name, const double *um, int nl) {
    char p[1400], k[32];
    for (int t = 0; t < 2; t++) {
        snprintf(p, sizeof p, "%s/%s%s", dir, name, t ? "_t" : "");
        for (int l = 0; l < nl; l++) { snprintf(k, sizeof k, "%g", um[l]); if (write_array(p, k, 128 >> l, t, l)) return -1; }
        if (z3w_write_group(p, um, nl, t ? "recto" : "ct", nullptr)) return -1;
    }
    return 0;
}
/* the windows' (source, level) counts: n[source][level] */
static int run(sources *S, const char *rungs, float shift, int nwin, int n[2][4]) {
    sample_cfg c = sample_cfg_default();
    c.P = 16; c.B = 1; c.nworkers = 2; c.nbuf = 3; c.seed = 11; c.deterministic = 1; c.augment = 0; c.min_fg = 0; c.empty_keep = 1;
    memset(c.level_p, 0, sizeof c.level_p); c.level_p[0] = 1;
    c.nrung = 0; char t[128]; snprintf(t, sizeof t, "%s", rungs);
    for (char *q = strtok(t, ","); q; q = strtok(nullptr, ",")) { c.rung_um[c.nrung] = (float)atof(q); c.rung_w[c.nrung++] = 1.f; }
    c.rung_shift_p = shift;
    sampler *sp = sampler_start(S, &c);
    if (!sp) return -1;
    memset(n, 0, sizeof(int) * 8);
    for (int i = 0; i < nwin; i++) { batch *b = sampler_next(sp); if (b->src[0] >= 0 && b->src[0] < 2 && b->level[0] >= 0 && b->level[0] < 4) n[b->src[0]][b->level[0]]++; sampler_release(sp, b); }
    sampler_stop(sp);
    return 0;
}
int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: test_rungs DIR\n"); return 2; }
    const char *dir = argv[1]; char p[1400];
    snprintf(p, sizeof p, "rm -rf '%s' && mkdir -p '%s'", dir, dir); if (system(p)) return 1;
    const double ua[3] = {2.4, 4.8, 9.6}, ub[2] = {7.9, 15.8};
    if (scan(dir, "a", ua, 3) || scan(dir, "b", ub, 2)) { fprintf(stderr, "FAIL: fixture\n"); return 1; }
    snprintf(p, sizeof p, "%s/sources.json", dir); FILE *f = fopen(p, "w");
    fprintf(f, "{\"sources\": [{\"name\": \"a\", \"root\": \"%s\", \"ct\": \"a\", \"um\": 2.4, \"weight\": 1, \"targets\": {\"recto\": {\"root\": \"%s\", \"group\": \"a_t\", \"encoding\": \"binary\"}}},\n"
               "             {\"name\": \"b\", \"root\": \"%s\", \"ct\": \"b\", \"um\": 7.9, \"weight\": 1, \"targets\": {\"recto\": {\"root\": \"%s\", \"group\": \"b_t\", \"encoding\": \"binary\"}}}]}\n", dir, dir, dir, dir);
    fclose(f);
    sources *S = sources_load(p); if (!S) { fprintf(stderr, "FAIL: sources_load\n"); return 1; }
    int n[2][4];
    CHECK(!run(S, "9.6", 0, 60, n), "rungs 9.6: sampler");
    CHECK(n[0][2] > 0 && n[0][0] + n[0][1] == 0 && n[1][0] > 0 && n[1][1] == 0, "rungs 9.6: A levels %d %d %d (want only 2), B levels %d %d (want only 0)", n[0][0], n[0][1], n[0][2], n[1][0], n[1][1]);
    CHECK(!run(S, "4.8,9.6", 0, 200, n), "rungs 4.8,9.6: sampler");
    CHECK(n[0][1] > 20 && n[0][2] > 20 && abs(n[0][1] - n[0][2]) < (n[0][1] + n[0][2]) / 3 && n[0][0] == 0 && n[1][0] > 0 && n[1][1] == 0,
          "rungs 4.8,9.6: A levels %d %d %d (want 1 and 2 about evenly), B levels %d %d (want only 0)", n[0][0], n[0][1], n[0][2], n[1][0], n[1][1]);
    CHECK(!run(S, "2.4,4.8", 1, 100, n), "rungs 2.4,4.8 shift 1: sampler");
    CHECK(n[0][0] > 10 && n[0][1] > 10 && n[0][2] == 0 && n[1][0] + n[1][1] == 0, "rungs 2.4,4.8 shift 1: A levels %d %d %d (want 0 and 1), B levels %d %d (want none)", n[0][0], n[0][1], n[0][2], n[1][0], n[1][1]);
    CHECK(run(S, "38.4", 0, 4, n) != 0, "rungs 38.4: a sampler started without any reachable level");
    sources_free(S);
    if (fails) { fprintf(stderr, "test_rungs: %d FAILED\n", fails); return 1; }
    printf("test_rungs: ok\n");
    return 0;
}
