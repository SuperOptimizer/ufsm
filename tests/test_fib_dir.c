/* nn_fib_dir_targets (train --fib-dir): on synthetic H / V label tubes (radius 2) the target dyad is the tube axis: an H tube along x,
   an H tube along the diagonal (y = x) in a z plane, a V tube along z, and a tilted V tube (z with a 0.3 slope in y). Checks: at tube
   voxels away from the ends and crossings the decoded axis |a . axis| >= 0.98 and the family plane is right; voxels outside tubes,
   unknown voxels and voxels in both families' tubes are masked; a tube end still gets a target (>= 2r + 1 voxels) or none, never a
   wrong one. */
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)
enum { D = 64, H = 96, W = 96, NT = 4 };
static const char fam[NT] = {'H', 'H', 'V', 'V'};
/* the distance of (z, y, x) to tube k's axis and its unit direction (z, y, x) */
static double tube(int k, double z, double y, double x, double a[3]) {
    if (k == 0) { a[0] = 0; a[1] = 0; a[2] = 1; return hypot(z - 16, y - 20); }
    if (k == 1) { a[0] = 0; a[1] = a[2] = M_SQRT1_2; const double u = (y - x) * M_SQRT1_2; return hypot(z - 40, u - 20); }   /* y - x = 28.3 */
    if (k == 2) { a[0] = 1; a[1] = 0; a[2] = 0; return hypot(y - 70, x - 30); }
    const double s = 0.3, nn = sqrt(1 + s * s); a[0] = 1 / nn; a[1] = s / nn; a[2] = 0;   /* x = 70, y = 50 + 0.3 (z - 32) */
    const double dy = y - (50 + s * (z - 32)); return hypot(x - 70, dy / nn);
}
int main(void) {
    nn_init(0);
    const size_t n = (size_t)D * H * W; uint8_t *t = calloc(2 * n, 1), *m = malloc(n), *o = malloc(7 * n), *om = malloc(n);
    memset(m, 1, n);
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) { const size_t i = ((size_t)z * H + y) * W + x; double a[3];
        for (int k = 0; k < NT; k++) if (tube(k, z, y, x, a) <= 2.0) t[(fam[k] == 'V' ? n : 0) + i] = 254;
        if (z == 5 && y > 80) m[i] = 0; }   /* an unknown patch */
    uint8_t *dt = nn_malloc(2 * n), *dm = nn_malloc(n), *dout = nn_malloc(7 * n), *dom = nn_malloc(n);
    nn_h2d(dt, t, 2 * n); nn_h2d(dm, m, n); nn_fib_dir_targets(dt, dm, D, H, W, 4, 0.6f, dout, dom); nn_sync();
    nn_d2h(o, dout, 7 * n); nn_d2h(om, dom, n);
    size_t got[NT] = {0}, bad = 0, wrongmask = 0, nboth = 0;
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const size_t i = ((size_t)z * H + y) * W + x; const int h = t[i] >= 128, v = t[n + i] >= 128;
        if (!m[i] || h == v) { wrongmask += om[i] != 0; nboth += h && v; continue; }
        if (!om[i]) continue;
        /* decode: the largest eigenvector of the stored dyad */
        const double c[6] = {o[i] / 255., o[n + i] / 255., o[2 * n + i] / 255., o[3 * n + i] / 255. - 0.5, o[4 * n + i] / 255. - 0.5, o[5 * n + i] / 255. - 0.5};
        double e[3] = {1, 1, 1}; for (int it = 0; it < 60; it++) { const double w0 = c[0] * e[0] + c[3] * e[1] + c[4] * e[2], w1 = c[3] * e[0] + c[1] * e[1] + c[5] * e[2], w2 = c[4] * e[0] + c[5] * e[1] + c[2] * e[2];
            const double l = sqrt(w0 * w0 + w1 * w1 + w2 * w2); e[0] = w0 / l; e[1] = w1 / l; e[2] = w2 / l; }
        int k0 = -1; double best = 1e9, a[3], ab[3];
        for (int k = 0; k < NT; k++) { if ((fam[k] == 'V') != v) continue; const double d = tube(k, z, y, x, a); if (d < best) { best = d; k0 = k; memcpy(ab, a, sizeof a); } }
        const double dot = fabs(e[0] * ab[0] + e[1] * ab[1] + e[2] * ab[2]);
        /* away from the box faces (the tubes' ends) the axis must be right */
        const int inner = z >= 6 && z < D - 6 && y >= 6 && y < H - 6 && x >= 6 && x < W - 6;
        if (inner) { got[k0]++; if (dot < 0.98 || (o[6 * n + i] >= 128) != v) { if (bad < 5) fprintf(stderr, "voxel %d %d %d tube %d: |a . axis| %.3f family %d\n", z, y, x, k0, dot, o[6 * n + i]); bad++; } }
    }
    for (int k = 0; k < NT; k++) CHECK(got[k] > 200, "tube %d: %zu inner voxels with a target", k, got[k]);
    CHECK(!bad, "%zu inner tube voxels with a wrong axis or family", bad);
    CHECK(!wrongmask, "%zu voxels outside a single family's known tube carry a target (crossing voxels %zu)", wrongmask, nboth);
    nn_free(dt); nn_free(dm); nn_free(dout); nn_free(dom); free(t); free(m); free(o); free(om);
    printf(fails ? "test_fib_dir: %d FAILED\n" : "test_fib_dir: ok\n", fails); return fails ? 1 : 0;
}
