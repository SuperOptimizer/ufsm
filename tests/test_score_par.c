/* The pipelined GT scorer (src/main.c sc_run, fit --codes / wind-score without a dump) on a synthetic raster: a spiral U with a
   slip, NaN holes and codes 0 / 253..255 mixed in, over a box of 5 x 2 blocks (z 560 = 4 x 128 + a 48-row block; y 1060). Block kinds
   per (z block, y half): L labelled, S stored chunks without labels (codes 253..255 only: read, skipped), Z all zero (no stored chunk:
   not read): z0 L L, z1 S S, z2 S Z, z3 L S, z4 L L. That makes three skips in a row while a tally holds a read slot, and one skip
   between two tallies (the tally buffers alternate over the tallied blocks).
   (1) the printed score (ids, |error|, gauge, pairs, the --score-detail integer errors) is byte-identical for 1, 5 and 16 worker
   threads; (2) the detail second pass (records over UFSM_SCORE_DETAIL_GB) prints the same; (3) the serial --score-dump path (the
   old one-voxel-at-a-time loop) gives the same score and detail lines.
   Usage: test_score_par <ufsm binary> */
#include "z3w.h"
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)
static int sh(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static int sh(const char *fmt, ...) {
    char cmd[8192]; va_list a; va_start(a, fmt); vsnprintf(cmd, sizeof cmd, fmt, a); va_end(a);
    const int r = system(cmd);
    if (r) fprintf(stderr, "command failed (%d): %s\n", r, cmd);
    return r;
}
static char *slurp(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return nullptr;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *s = malloc((size_t)n + 1); if (!s) { fclose(f); return nullptr; }
    s[fread(s, 1, (size_t)n, f)] = 0; fclose(f); return s;
}
/* the lines of s starting with one of the prefixes, concatenated */
static char *lines(const char *s, const char *const *pre, int np) {
    size_t cap = strlen(s) + 1; char *o = calloc(cap, 1); size_t n = 0;
    for (const char *p = s; *p;) {
        const char *e = strchr(p, '\n'); const size_t len = e ? (size_t)(e - p) + 1 : strlen(p);
        int keep = 0; for (int k = 0; k < np; k++) keep |= !strncmp(p, pre[k], strlen(pre[k]));
        if (keep) { memcpy(o + n, p, len); n += len; }
        p += len;
    }
    o[n] = 0; return o;
}
static uint32_t hsh(uint32_t a) { a ^= a >> 16; a *= 0x7feb352du; a ^= a >> 15; a *= 0x846ca68bu; a ^= a >> 16; return a; }

enum { Z = 560, Y = 1060, X = 60 };
static char kind(int z, int y) { static const char *k[5] = {"LL", "SS", "SZ", "LS", "LL"}; return k[z / 128][y >= 1024]; }
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: test_score_par <ufsm binary>\n"); return 2; }
    char D[] = "/tmp/ufsm_score_par_XXXXXX"; if (!mkdtemp(D)) return 1;
    char p[1400];
    const double cy0 = 530, cx0 = 30;
    snprintf(p, sizeof p, "%s/axis.json", D); FILE *f = fopen(p, "w");
    fprintf(f, "{\"control_points\":[{\"z\":0,\"y\":%g,\"x\":%g},{\"z\":%d,\"y\":%g,\"x\":%g}]}\n", cy0, cx0, Z, cy0 + 6, cx0 - 4); fclose(f);
    snprintf(p, sizeof p, "%s/sweep.json", D); f = fopen(p, "w");
    fprintf(f, "{\"shape\":[%d,%d,%d],\"origin_zyx\":[0,0,0],\"um\":9.6,\"scale\":1,\"axis\":\"%s/axis.json\"}\n", Z, Y, X, D); fclose(f);
    const size_t nv = (size_t)Z * Y * X; float *U = malloc(nv * 4);
    const int S = 512, nsz = (Z + S - 1) / S, nsy = (Y + S - 1) / S, nsx = (X + S - 1) / S;
    uint8_t *sh_ = calloc((size_t)S * S * S, 1);   /* one shard */
    snprintf(p, sizeof p, "%s/codes/9.6", D); sh("mkdir -p %s/codes", D);
    const int64_t shp[3] = {Z, Y, X}; z3w *w = z3w_create(p, shp, S, 0.f, 0, nullptr); CHECK(w, "z3w_create");
    for (int sz = 0; sz < nsz; sz++) for (int sy = 0; sy < nsy; sy++) for (int sx = 0; sx < nsx; sx++) {
        memset(sh_, 0, (size_t)S * S * S);
        for (int zz = 0; zz < S && sz * S + zz < Z; zz++) for (int yy = 0; yy < S && sy * S + yy < Y; yy++) for (int xx = 0; xx < S && sx * S + xx < X; xx++) {
            const int z = sz * S + zz, y = sy * S + yy, x = sx * S + xx; const size_t i = ((size_t)z * Y + y) * X + x;
            const double cy = cy0 + 6.0 * z / Z, cx = cx0 - 4.0 * z / Z, r = hypot(y - cy, x - cx), th = atan2(y - cy, x - cx);
            const double q = r / 12 - th / (2 * M_PI) + 0.003 * z;   /* the true winding */
            const uint32_t hh = hsh((uint32_t)i * 2654435761u + 17);
            double u = q + 0.15 * ((hh & 1023) / 1023.0 - 0.5) + 0.7;   /* noise and a gauge */
            if (x > 45 && y > 700) u += 1;   /* a slip */
            if (z > 100 && y < 200) u += 3.3;
            U[i] = (hh >> 10) % 97 == 0 ? NAN : (float)u;
            uint8_t c = 0;
            const char kd = kind(z, y);
            if (kd == 'L' && r < 520 && (hh >> 20) % 3) { const long k = lround(18 * q); c = (uint8_t)(1 + ((k % 252) + 252) % 252); }
            if (kd != 'Z' && (hh >> 24) % 211 == 0) c = (uint8_t)(253 + (hh >> 8) % 3);
            sh_[((size_t)zz * S + yy) * S + xx] = c;
        }
        CHECK(!z3w_write_shard(w, sz, sy, sx, sh_, 4), "write shard");
    }
    CHECK(!z3w_close(w), "z3w_close");
    snprintf(p, sizeof p, "%s/U.f32", D); f = fopen(p, "wb"); CHECK(fwrite(U, 4, nv, f) == nv, "write U"); fclose(f); free(U); free(sh_);

    const char *thr[3] = {"1", "5", "16"}; char *out[3] = {0};
    for (int k = 0; k < 3; k++) {
        CHECK(!sh("UFSM_FIT_THREADS=%s %s wind-score %s %s/codes 9.6 --score-detail 1 --score-pairs 1 > %s/t%d.out 2> %s/t%d.err", thr[k], argv[1], D, D, D, k, D, k), "wind-score threads %s", thr[k]);
        snprintf(p, sizeof p, "%s/t%d.out", D, k); out[k] = slurp(p);
        CHECK(out[k] && strstr(out[k], "fit score:") && strstr(out[k], "pair winding right") && strstr(out[k], "z octile"), "threads %s: score, pairs and detail printed", thr[k]);
    }
    for (int k = 1; k < 3; k++) CHECK(out[0] && out[k] && !strcmp(out[0], out[k]), "threads 1 vs %s: identical output\n--- 1:\n%s--- %s:\n%s", thr[k], out[0], thr[k], out[k]);
    CHECK(!sh("UFSM_FIT_THREADS=7 UFSM_SCORE_DETAIL_GB=0.000001 %s wind-score %s %s/codes 9.6 --score-detail 1 --score-pairs 1 > %s/p2.out 2> %s/p2.err", argv[1], D, D, D, D), "wind-score detail pass 2");
    snprintf(p, sizeof p, "%s/p2.out", D); char *p2 = slurp(p);
    CHECK(p2 && out[0] && !strcmp(p2, out[0]), "detail second pass: identical output\n--- records:\n%s--- pass 2:\n%s", out[0], p2);
    CHECK(!sh("%s wind-score %s %s/codes 9.6 --score-detail 1 --score-dump %s/dump.bin > %s/dump.out 2> %s/dump.err", argv[1], D, D, D, D, D), "wind-score serial dump path");
    snprintf(p, sizeof p, "%s/dump.out", D); char *du = slurp(p);
    const char *pre[4] = {"fit score:", "  ", "    z octile", "    radius"};
    char *a = out[0] ? lines(out[0], pre, 4) : nullptr, *b = du ? lines(du, pre, 4) : nullptr;
    CHECK(a && b && *a && !strcmp(a, b), "serial dump path: the same score and detail lines\n--- pipelined:\n%s--- serial:\n%s", a ? a : "", b ? b : "");
    /* empty GT: a missing level, a level without stored chunks, a level with stored chunks but no label code: nonzero exit (3 for an
       empty level / 0 labelled), an ERROR on stderr and no "fit score:" line; UFSM_SCORE_ALLOW_EMPTY=1 prints the empty score */
    {
        const int64_t s3[3] = {Z, Y, X};
        snprintf(p, sizeof p, "%s/codes/4.8", D); z3w *we = z3w_create(p, s3, S, 0.f, 0, nullptr); CHECK(we && !z3w_close(we), "empty level");
        snprintf(p, sizeof p, "%s/codes/2.4", D); z3w *wn = z3w_create(p, s3, S, 0.f, 0, nullptr); CHECK(wn, "no-label level");
        uint8_t *nl = calloc((size_t)S * S * S, 1); for (size_t i = 0; i < (size_t)S * S * S; i += 997) nl[i] = 254;
        if (wn) { CHECK(!z3w_write_shard(wn, 0, 0, 0, nl, 4), "no-label shard"); CHECK(!z3w_close(wn), "no-label close"); }
        free(nl);
        const char *key[3] = {"7.7", "4.8", "2.4"}; const int want[3] = {1, 3, 3};
        for (int k = 0; k < 3; k++) {   /* (the key is also the um: 4.8 / 2.4 scale the label voxel; the box is read all the same) */
            char cmd[2048]; snprintf(cmd, sizeof cmd, "%s wind-score %s %s/codes %s --score-pairs 1 > %s/e%d.out 2> %s/e%d.err", argv[1], D, D, key[k], D, k, D, k);
            const int st = system(cmd); const int ex = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
            snprintf(p, sizeof p, "%s/e%d.out", D, k); char *eo = slurp(p); snprintf(p, sizeof p, "%s/e%d.err", D, k); char *ee = slurp(p);
            CHECK(ex == want[k], "codes key %s: exit %d, want %d", key[k], ex, want[k]);
            CHECK(eo && !strstr(eo, "fit score:"), "codes key %s: no fit score line printed", key[k]);
            CHECK(ee && (k == 0 || strstr(ee, "ERROR")), "codes key %s: an ERROR on stderr", key[k]);
            if (k) CHECK(ee && strstr(ee, "score: codes level"), "codes key %s: the codes level printed up front", key[k]);
            free(eo); free(ee);
        }
        snprintf(p, sizeof p, "UFSM_SCORE_ALLOW_EMPTY=1 %s wind-score %s %s/codes 2.4 > %s/ea.out 2> %s/ea.err", argv[1], D, D, D, D);
        CHECK(system(p) == 0, "UFSM_SCORE_ALLOW_EMPTY=1: exit 0");
        snprintf(p, sizeof p, "%s/ea.out", D); char *ea = slurp(p);
        CHECK(ea && strstr(ea, "fit score: 0 labelled voxels"), "UFSM_SCORE_ALLOW_EMPTY=1: the empty score printed");
        free(ea);
    }
    if (out[0]) fputs(out[0], stdout);
    free(a); free(b); free(du); free(p2); for (int k = 0; k < 3; k++) free(out[k]);
    if (!fails && !getenv("UFSM_TEST_KEEP")) sh("rm -rf %s", D); else fprintf(stderr, "test_score_par: kept %s\n", D);
    printf("test_score_par: %s\n", fails ? "FAILED" : "ok");
    return fails != 0;
}
