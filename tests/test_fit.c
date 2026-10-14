/* Winding fitter (src/fit.c) on synthetic fit fields: an Archimedean spiral around a tilted axis with z-varying pitch, an
   angular wobble, a packed sector and a hole, for dir +1 and -1. The fields are what predict --fit-fields will write
   (phase vector, n n^T, spacing, material weight) computed from the true winding coordinate W. Checks: the fitted direction,
   and W_fit - W_true = one integer gauge everywhere (winding ids >= 99.5% right, |error| < 0.08 turn at 99% of voxels). */
#include "fit.h"
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

static int g_dir; static double g_wob = 0.4, g_pack = 0.6, g_dpitch = 10; static int g_hole = 1; static double g_bias = 0;   /* FIT_BIAS: the phase under-counts the packed sector by this fraction (a weak network) */
static const double H8 = 8, R0 = 384;
static double g_split = 0; static int g_tile = 256;   /* the two-level unwrapping (fit_cfg uw_split / uw_tile) */
static double ax_y(double z) { return R0 + 0.02 * z; }
static double ax_x(double z) { return R0 - 0.01 * z; }
static double pitch_of(double z) { return 60 + g_dpitch * z / 384; }
static double U_true(double z, double y, double x) {
    const double dy = y - ax_y(z), dx = x - ax_x(z), r = hypot(dy, dx), th = atan2(dy, dx);
    double dth = th - 1.0; dth -= 2 * M_PI * floor((dth + M_PI) / (2 * M_PI));
    return r / pitch_of(z) + g_wob * sin(th) * r / 300 + g_pack / 60 * r * exp(-dth * dth / (2 * 0.3 * 0.3));
}
static double W_true(double z, double y, double x) { return U_true(z, y, x) - g_dir * atan2(y - ax_y(z), x - ax_x(z)) / (2 * M_PI); }
static double W_phase(double z, double y, double x) {   /* what the phase field encodes: the packed-sector turns scaled by 1 - g_bias */
    const double dy = y - ax_y(z), dx = x - ax_x(z), r = hypot(dy, dx), th = atan2(dy, dx);
    double dth = th - 1.0; dth -= 2 * M_PI * floor((dth + M_PI) / (2 * M_PI));
    return W_true(z, y, x) - g_bias * g_pack / 60 * r * exp(-dth * dth / (2 * 0.3 * 0.3));
}

static int run_case(int dir) {
    g_dir = dir;
    const int D = 48, H = 96, W = 96; const size_t n = (size_t)D * H * W;
    uint8_t *pc = malloc(n), *ps = malloc(n), *t[6], *sp = malloc(n), *w = malloc(n);
    for (int k = 0; k < 6; k++) t[k] = malloc(n);
    const int na = (int)(D * H8) + 1;
    float *cy = malloc(na * sizeof(float)), *cx = malloc(na * sizeof(float));
    for (int z = 0; z < na; z++) { cy[z] = (float)ax_y(z); cx[z] = (float)ax_x(z); }
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const size_t i = ((size_t)z * H + y) * W + x;
        const double pz = (z + 0.5) * H8, py = (y + 0.5) * H8, px = (x + 0.5) * H8;
        const double r = hypot(py - ax_y(pz), px - ax_x(pz));
        const int hole = g_hole && hypot(hypot(pz - 200, py - 600), px - 380) < 40;
        w[i] = r >= 3 * 70 && r <= 370 && !hole ? 255 : 0;
        const double Wv = W_phase(pz, py, px), ph = 2 * M_PI * (Wv - floor(Wv));
        const double amp = g_split > 0 && x >= 40 && x < 56 ? 0.4 : 1;   /* two-level: a band under the split, left to the serial pass */
        pc[i] = (uint8_t)lround(128 + 127 * amp * cos(ph)); ps[i] = (uint8_t)lround(128 + 127 * amp * sin(ph));
        double g[3]; const double e = 0.5;   /* grad W: central differences, the theta part wrapped */
        for (int d = 0; d < 3; d++) {
            double a[3] = {pz, py, px}, b[3] = {pz, py, px}; a[d] += e; b[d] -= e;
            double du = U_true(a[0], a[1], a[2]) - U_true(b[0], b[1], b[2]);
            double dth = atan2(a[1] - ax_y(a[0]), a[2] - ax_x(a[0])) - atan2(b[1] - ax_y(b[0]), b[2] - ax_x(b[0]));
            dth -= 2 * M_PI * floor((dth + M_PI) / (2 * M_PI));
            g[d] = (du - dir * dth / (2 * M_PI)) / (2 * e);
        }
        const double rho = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]), nz = g[0] / rho, ny = g[1] / rho, nx = g[2] / rho;
        const double tv[6] = {nz * nz, ny * ny, nx * nx, 0.5 + nz * ny, 0.5 + nz * nx, 0.5 + ny * nx};
        for (int k = 0; k < 6; k++) t[k][i] = (uint8_t)lround(255 * tv[k]);
        double s = 8 / (H8 * rho); sp[i] = (uint8_t)(s < 1 ? 1 : s > 255 ? 255 : lround(s));
    }
    fit_fields f = {D, H, W, H8, pc, ps, {t[0], t[1], t[2], t[3], t[4], t[5]}, sp, w, cy, cx};
    fit_cfg c = fit_cfg_default(); c.verbose = 1; c.uw_split = g_split; c.uw_tile = g_tile;
    fit_model *m = fit_run(&f, &c);
    int bad = 0;
    if (!m) { fprintf(stderr, "FAIL: fit_run\n"); return 1; }
    CHECK(m->dir == dir, "dir %+d: fitted dir %+d", dir, m->dir);
    /* the integer gauge: the most frequent round(W_fit - W_true) */
    int hist[401] = {0}; size_t nd = 0;
    for (size_t i = 0; i < n; i += 1) {
        if (!w[i]) continue;
        const int z = (int)(i / ((size_t)H * W)), y = (int)((i / W) % H), x = (int)(i % W);
        const double pz = (z + 0.5) * H8, py = (y + 0.5) * H8, px = (x + 0.5) * H8;
        const long k = lround(fit_W(m, pz, py, px) - W_true(pz, py, px));
        if (k >= -200 && k <= 200) hist[k + 200]++;
        nd++;
    }
    int k0 = 0; for (int k = 1; k < 401; k++) if (hist[k] > hist[k0]) k0 = k;
    size_t right = 0, close = 0;
    for (size_t i = 0; i < n; i++) {
        if (!w[i]) continue;
        const int z = (int)(i / ((size_t)H * W)), y = (int)((i / W) % H), x = (int)(i % W);
        const double pz = (z + 0.5) * H8, py = (y + 0.5) * H8, px = (x + 0.5) * H8;
        const double d = fit_W(m, pz, py, px) - W_true(pz, py, px) - (k0 - 200);
        right += fabs(d) < 0.5; close += fabs(d) < 0.08;
    }
    const double acc = (double)right / nd, prec = (double)close / nd;
    printf("dir %+d%s: gauge %d, winding ids right %.4f, |error| < 0.08 turn %.4f (%zu voxels), loss %.5f\n", dir, g_split > 0 ? " two-level" : "", k0 - 200, acc, prec, nd, m->loss[0]);
    if (g_bias == 0) {
        CHECK(acc >= 0.995, "dir %+d: winding ids right %.4f < 0.995", dir, acc);
        CHECK(prec >= 0.99, "dir %+d: |error| < 0.08 at %.4f < 0.99", dir, prec);
    }
    fit_free(m);
    free(pc); free(ps); free(sp); free(w); free(cy); free(cx); for (int k = 0; k < 6; k++) free(t[k]);
    return bad;
}

int main(void) {
    nn_init(0);
    if (getenv("FIT_BIAS")) g_bias = atof(getenv("FIT_BIAS"));
    if (getenv("FIT_SIMPLE")) { g_wob = atof(getenv("FIT_SIMPLE")); g_pack = 0; g_dpitch = 0; g_hole = 0; }
    run_case(+1);
    run_case(-1);
    g_split = 0.5; g_tile = 32; run_case(+1);   /* 18 tiles: pieces stitched across tile faces, the hole and the edge by the serial pass */
    if (fails) { fprintf(stderr, "test_fit: %d failure(s)\n", fails); return 1; }
    printf("test_fit: OK\n");
    return 0;
}
