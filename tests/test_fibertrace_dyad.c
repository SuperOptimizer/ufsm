/* ufsm fiber-trace --mask ... --dyad DIR: the axis from predicted dyad channels (train --fib-dir) instead of the mask's tube.
   Synthetic 128^3 volume (zarr v3, level "9.6"): the mask is a flat LAYER (z 60..67 at 254), the case where the mask's own structure
   tensor has no axis (any in-plane direction); the dyad channels c2..c7 encode the axis 10 degrees off x in the layer plane (not the 45 degrees the tube tracer's power iteration lands on in a flat layer)
   (zz, yy, xx, 0.5 + zy, 0.5 + zx, 0.5 + yx, x 255). Checks: with --dyad every traced line runs within 15 degrees of that axis and the
   lines are long (mean >= 60 voxels; the layer spans 128); without --dyad the same mask gives lines that do not (the test is not
   trivially met by the mask alone). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "afv.h"
#include "z3w.h"

int cmd_fiber_trace(int argc, char **argv);
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)
enum { N = 128 };
static int put(const char *dir, const uint8_t *v) {
    char p[600]; snprintf(p, sizeof p, "mkdir -p '%s'", dir); if (system(p)) return -1;
    const double um = 9.6; if (z3w_write_group(dir, &um, 1, "test", nullptr)) return -1;
    snprintf(p, sizeof p, "%s/9.6", dir); const int64_t sh[3] = {N, N, N};
    z3w *w = z3w_create(p, sh, 128, 0, 0, nullptr); if (!w) return -1;
    if (z3w_write_shard(w, 0, 0, 0, v, 1)) return -1;
    return z3w_close(w);
}
/* lines: mean length (voxels of the mask level, from level-0 points / 4) and the worst |cos| to the axis per line (minimum over lines) */
static void judge(const char *fib, double *mean, double *worst, int *nl) {
    fib_set F = {0}; *mean = 0; *worst = 1; *nl = 0; if (fib_load(fib, &F)) { *worst = -1; return; }
    const double ax[3] = {0, sin(10 * M_PI / 180), cos(10 * M_PI / 180)};
    for (int r = 0; r < F.n; r++) { const float *a = F.pts + 3 * (size_t)F.off[r], *b = F.pts + 3 * (size_t)(F.off[r + 1] - 1);
        double d[3] = {(b[0] - a[0]) / 4, (b[1] - a[1]) / 4, (b[2] - a[2]) / 4}; const double L = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        *mean += L; if (L > 0) { const double c = fabs(d[0] * ax[0] + d[1] * ax[1] + d[2] * ax[2]) / L; if (c < *worst) *worst = c; } }
    *nl = F.n; if (F.n) *mean /= F.n; fib_free(&F);
}
int main(void) {
    char root[] = "/tmp/ufsm-fibdyad-XXXXXX"; if (!mkdtemp(root)) { fprintf(stderr, "FAIL: tmp dir\n"); return 1; }
    const size_t nv = (size_t)N * N * N; uint8_t *m = calloc(nv, 1), *c[6]; for (int k = 0; k < 6; k++) c[k] = calloc(nv, 1);
    const double a[3] = {0, sin(10 * M_PI / 180), cos(10 * M_PI / 180)}, v[6] = {a[0] * a[0], a[1] * a[1], a[2] * a[2], 0.5 + a[0] * a[1], 0.5 + a[0] * a[2], 0.5 + a[1] * a[2]};
    for (int z = 60; z < 68; z++) for (int y = 0; y < N; y++) for (int x = 0; x < N; x++) { const size_t i = ((size_t)z * N + y) * N + x;
        m[i] = 254; for (int k = 0; k < 6; k++) c[k][i] = (uint8_t)lround(255 * v[k]); }
    char p[600], dy[600], f1[600], f2[600];
    snprintf(p, sizeof p, "%s/mask", root); CHECK(!put(p, m), "write the mask");
    snprintf(dy, sizeof dy, "%s/dyad", root);
    for (int k = 0; k < 6; k++) { snprintf(p, sizeof p, "%s/c%d", dy, 2 + k); CHECK(!put(p, c[k]), "write dyad channel %d", 2 + k); }
    snprintf(p, sizeof p, "%s/mask", root); snprintf(f1, sizeof f1, "%s/dyad.fib", root); snprintf(f2, sizeof f2, "%s/tube.fib", root);
    char *av1[] = {"ufsm", "fiber-trace", "-", "-", f1, "--mask", p, "9.6", "--box", "0,0,0,128,128,128", "--scaledown", "2", "--family", "H", "--half", "256", "--min-len", "20", "--dyad", dy, "--threads", "2"};
    CHECK(cmd_fiber_trace(22, av1) == 0, "fiber-trace --dyad");
    char *av2[] = {"ufsm", "fiber-trace", "-", "-", f2, "--mask", p, "9.6", "--box", "0,0,0,128,128,128", "--scaledown", "2", "--family", "H", "--half", "256", "--min-len", "20", "--threads", "2"};
    CHECK(cmd_fiber_trace(20, av2) == 0, "fiber-trace (tube)");
    double mean, worst; int nl; judge(f1, &mean, &worst, &nl);
    CHECK(nl > 0 && worst >= cos(15 * M_PI / 180) && mean >= 60, "--dyad: %d lines, mean %.1f voxels, worst |cos| to the axis %.3f", nl, mean, worst);
    double mean2, worst2; int nl2; judge(f2, &mean2, &worst2, &nl2);
    CHECK(nl2 == 0 || worst2 < cos(15 * M_PI / 180) || mean2 < 60, "the tube-only trace already meets the bar (%d lines, mean %.1f, worst |cos| %.3f): the test proves nothing", nl2, mean2, worst2);
    fprintf(stderr, "dyad: %d lines mean %.1f worst |cos| %.3f; tube only: %d lines mean %.1f worst |cos| %.3f\n", nl, mean, worst, nl2, mean2, worst2);
    snprintf(p, sizeof p, "rm -rf %s", root); if (!fails && system(p)) fails++;
    free(m); for (int k = 0; k < 6; k++) free(c[k]);
    printf(fails ? "test_fibertrace_dyad: %d FAILED\n" : "test_fibertrace_dyad: ok\n", fails); return fails ? 1 : 0;
}
