/* ufsm sheet-score on synthetic code rasters: GT = 10 flat sheets x = 8 + 8 k (k = 0..9, code 1 + 18 k), volume 64 x 128 x 96.
   Ours:
   - identical                         -> recall 1, precision 1, cell ids 1, global ids 1 (offset 0);
   - shifted by 1 voxel in x           -> all 1 at --tol 1; recall 0 at --tol 0;
   - sheet k = 4 missing               -> recall 0.9, precision 1;
   - every code + 1 turn               -> global ids 1 at offset 1;
   - + 1 turn only for y >= 64         -> global ids 0.5, cell ids 1 (cells of 32 never straddle y = 64): a gauge change between
                                          regions is not a local error, a skip inside a cell would be;
   - an extra sheet of ours between GT sheets 2 and 3 (x = 28) -> precision drops (the extra sheet is near GT but on none of them);
   - same numbers with 1 and 4 threads. */
#include "store.h"
#include "z3w.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int cmd_sheet_score(int argc, char **argv);
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)
enum { NZ = 64, NY = 128, NX = 96 };

static int put(const char *root, uint8_t (*f)(int, int, int)) {
    const int64_t sh[3] = {NZ, NY, NX}; char p[700]; snprintf(p, sizeof p, "%s/1", root);
    z3w *w = z3w_create(p, sh, 128, 0, 0, nullptr); if (!w) return -1;
    uint8_t *b = calloc((size_t)128 * 128 * 128, 1);
    for (int z = 0; z < NZ; z++) for (int y = 0; y < NY; y++) for (int x = 0; x < NX; x++) b[((size_t)z * 128 + y) * 128 + x] = f(z, y, x);
    const int rc = z3w_write_shard(w, 0, 0, 0, b, 2); free(b);
    const double um = 1; return rc || z3w_close(w) || z3w_write_group(root, &um, 1, "codes", nullptr) ? -1 : 0;
}
static uint8_t sheet_at(int x, int shift) { const int u = x - shift - 8; return u >= 0 && u % 8 == 0 && u / 8 < 10 ? (uint8_t)(1 + 18 * (u / 8)) : 0; }
static uint8_t gt(int z, int y, int x) { (void)z; (void)y; return sheet_at(x, 0); }
static uint8_t o_same(int z, int y, int x) { return gt(z, y, x); }
static uint8_t o_shift(int z, int y, int x) { (void)z; (void)y; return sheet_at(x, 1); }
static uint8_t o_miss(int z, int y, int x) { const uint8_t c = gt(z, y, x); return c == 1 + 18 * 4 ? 0 : c; }
static uint8_t o_plus1(int z, int y, int x) { const uint8_t c = gt(z, y, x); return c ? (uint8_t)(1 + (c - 1 + 18) % 252) : 0; }
static uint8_t o_half(int z, int y, int x) { const uint8_t c = gt(z, y, x); return c && y >= 64 ? (uint8_t)(1 + (c - 1 + 18) % 252) : c; }
static uint8_t m_half(int z, int y, int x) { (void)z; (void)x; return y >= 64; }   /* --mask: the half with the +1 turn gated out */
static const char *v_mask;
static uint8_t o_extra(int z, int y, int x) { const uint8_t c = gt(z, y, x); return c ? c : x == 28 ? (uint8_t)(1 + 18 * 2 + 9) : 0; }

static double v_recall, v_prec, v_cell, v_glob; static int v_off;
static int score(const char *root, const char *ours, const char *tol, const char *th) {
    char o[700], g[700], out[700]; snprintf(o, sizeof o, "%s/%s", root, ours); snprintf(g, sizeof g, "%s/gt", root); snprintf(out, sizeof out, "%s/out.txt", root);
    fflush(stdout); FILE *sv = fopen(out, "w"); if (!sv) return -1;
    const int fd = dup(fileno(stdout)); dup2(fileno(sv), fileno(stdout));
    char mk[700]; snprintf(mk, sizeof mk, "%s/%s", root, v_mask ? v_mask : "");
    char *av[] = {"ufsm", "sheet-score", o, "1", g, "1", "--tol", (char *)tol, "--prec-r", "6", "--cell", "32", "--block", "64", "--threads", (char *)th, "--mask", mk, "1"};
    const int rc = cmd_sheet_score((int)(sizeof av / sizeof *av) - (v_mask ? 0 : 3), av);
    fflush(stdout); dup2(fd, fileno(stdout)); close(fd); fclose(sv);
    FILE *f = fopen(out, "r"); char line[512]; v_recall = v_prec = v_cell = v_glob = -1; v_off = -1;
    while (f && fgets(line, sizeof line, f)) {
        fputs(line, stdout);
        const char *q;
        if ((q = strstr(line, "recall "))) v_recall = atof(q + 7);
        if ((q = strstr(line, "precision "))) v_prec = atof(q + 10);
        if ((q = strstr(line, "cell ids "))) v_cell = atof(q + 9);
        if ((q = strstr(line, "global ids "))) v_glob = atof(q + 11);
        if ((q = strstr(line, "(offset "))) v_off = atoi(q + 8);
    }
    if (f) fclose(f);
    return rc;
}
#define NEAR(a, b) (fabs((a) - (b)) < 0.005)

int main(void) {
    char root[256]; snprintf(root, sizeof root, "%s/test_sheetscore_%d", getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp", (int)getpid());
    char cmd[400]; snprintf(cmd, sizeof cmd, "rm -rf %s && mkdir -p %s", root, root); if (system(cmd)) return 1;
    struct { const char *n; uint8_t (*f)(int, int, int); } A[] = {{"gt", gt}, {"same", o_same}, {"shift", o_shift}, {"miss", o_miss}, {"plus1", o_plus1}, {"half", o_half}, {"extra", o_extra}, {"mhalf", m_half}};
    for (size_t i = 0; i < sizeof A / sizeof *A; i++) { char p[600]; snprintf(p, sizeof p, "%s/%s", root, A[i].n); if (put(p, A[i].f)) { fprintf(stderr, "FAIL: write %s\n", A[i].n); return 1; } }
    CHECK(!score(root, "same", "1", "1") && NEAR(v_recall, 1) && NEAR(v_prec, 1) && NEAR(v_cell, 1) && NEAR(v_glob, 1) && v_off == 0, "identical: %g %g %g %g %d", v_recall, v_prec, v_cell, v_glob, v_off);
    const double r1 = v_recall, p1 = v_prec, c1 = v_cell;
    CHECK(!score(root, "same", "1", "4") && v_recall == r1 && v_prec == p1 && v_cell == c1, "1 vs 4 threads differ");
    CHECK(!score(root, "shift", "1", "4") && NEAR(v_recall, 1) && NEAR(v_prec, 1) && NEAR(v_glob, 1), "shift 1 at tol 1: %g %g %g", v_recall, v_prec, v_glob);
    CHECK(!score(root, "shift", "0", "4") && NEAR(v_recall, 0), "shift 1 at tol 0: recall %g", v_recall);
    CHECK(!score(root, "miss", "1", "4") && NEAR(v_recall, 0.9) && NEAR(v_prec, 1), "missing sheet: %g %g", v_recall, v_prec);
    CHECK(!score(root, "plus1", "1", "4") && NEAR(v_glob, 1) && v_off == 1, "plus one turn: %g offset %d", v_glob, v_off);
    CHECK(!score(root, "half", "1", "4") && NEAR(v_glob, 0.5) && NEAR(v_cell, 1), "half +1 turn: global %g cell %g", v_glob, v_cell);
    v_mask = "mhalf"; CHECK(!score(root, "half", "1", "4") && NEAR(v_glob, 1) && NEAR(v_cell, 1) && fabs(v_recall - 0.5) < 0.02, "half +1 turn, masked: global %g cell %g recall %g", v_glob, v_cell, v_recall); v_mask = nullptr;
    CHECK(!score(root, "extra", "1", "4") && v_prec < 0.95 && NEAR(v_recall, 1), "extra sheet: precision %g recall %g", v_prec, v_recall);
    snprintf(cmd, sizeof cmd, "rm -rf %s", root); if (system(cmd)) {}
    if (fails) { fprintf(stderr, "test_sheetscore: %d failure(s)\n", fails); return 1; }
    printf("test_sheetscore: OK\n");
    return 0;
}
