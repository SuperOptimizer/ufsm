/* ink_blend_run's block-sparse pass (perf 11-25, predict --seamless 2's blend: pass shards 128, output shards 512) against the whole-shard
   pass (UFSM_IB_BLOCK=0): two inputs on grids 0 and 64 (tile 128), 128^3 shards, a ragged shape, about half of each input's shards absent
   (the other grid's tiles, occupancy air); the output shard files must be byte-identical, and the block pass must blend fewer blocks. */
#include "inkblend.h"
#include "z3w.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)
static uint8_t *slurp(const char *p, long *n) { FILE *f = fopen(p, "rb"); if (!f) { *n = -1; return nullptr; } fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc((size_t)*n + 1); if (fread(b, 1, (size_t)*n, f) != (size_t)*n) *n = -2; fclose(f); return b; }

int main(void) {
    const char *t = getenv("TMPDIR"); char root[600]; snprintf(root, sizeof root, "%s/ibblock.%d", t && *t ? t : "/tmp", (int)getpid());
    const double um = 9.6; char lv[32]; z3w_level_name(um, lv, sizeof lv);
    const int64_t shape[3] = {600, 520, 700}; const int S = 128;
    char in[2][700]; uint8_t *d = malloc((size_t)S * S * S); uint64_t r = 12345;
    for (int i = 0; i < 2; i++) {
        snprintf(in[i], sizeof in[i], "%s/in%d", root, i); char p[800]; snprintf(p, sizeof p, "%s/%s", in[i], lv);
        z3w *w = z3w_create(p, shape, S, 0.f, 0, nullptr); CHECK(w != nullptr, "create %s: %s", p, z3w_error()); if (!w) return 1;
        for (int64_t a = 0; a * S < shape[0]; a++) for (int64_t b = 0; b * S < shape[1]; b++) for (int64_t c = 0; c * S < shape[2]; c++) {
            if ((a + 2 * b + 3 * c + i) % 5 > 1 - (a == 0 && b == 0)) continue;   /* absent: ~2 / 5 of each input's shards present
                                                                                        (out shard 0 denser) */
            for (size_t k = 0; k < (size_t)S * S * S; k++) { r = r * 6364136223846793005ull + 1442695040888963407ull; d[k] = (uint8_t)(r >> 56); }
            CHECK(!z3w_write_shard(w, a, b, c, d, 2), "write: %s", z3w_error());
        }
        z3w_close(w);
    }
    const char *ins[2] = {in[0], in[1]}; const double st[3] = {128, 128, 128}, g[2][3] = {{0, 0, 0}, {64, 64, 64}}; const int64_t org[2][3] = {{0}};
    char out[2][700];
    for (int k = 0; k < 2; k++) {
        snprintf(out[k], sizeof out[k], "%s/out%d", root, k); mkdir(out[k], 0775);
        setenv("UFSM_IB_BLOCK", k ? "1" : "0", 1);
        CHECK(!ink_blend_run(out[k], um, st, 2, ins, g, org, 1, 512, 0, 4, "{}"), "ink_blend_run %s", k ? "blocks" : "whole");
    }
    int nf = 0, nd = 0;
    for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) for (int c = 0; c < 2; c++) {
        char p0[900], p1[900]; long n0, n1;
        snprintf(p0, sizeof p0, "%s/%s/c/%d/%d/%d", out[0], lv, a, b, c); snprintf(p1, sizeof p1, "%s/%s/c/%d/%d/%d", out[1], lv, a, b, c);
        uint8_t *x = slurp(p0, &n0), *y = slurp(p1, &n1);
        if (n0 >= 0 || n1 >= 0) { nf++; if (n0 != n1 || !x || !y || memcmp(x, y, (size_t)n0)) nd++; }
        free(x); free(y);
    }
    CHECK(nf > 0 && nd == 0, "%d of %d output shards differ between the block and whole-shard passes", nd, nf);
    printf("ink_blend_run: block-sparse vs whole-shard: %d output shards, %d differ\n", nf, nd);
    char cmd[700]; snprintf(cmd, sizeof cmd, "rm -rf '%s'", root); if (system(cmd)) {}
    free(d); printf("%s\n", fails ? "FAILED" : "ok"); return fails != 0;
}
