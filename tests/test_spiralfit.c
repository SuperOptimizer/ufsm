/* ufsm spiral-fit (src/nn_spiral.cu, count-field --method spiral):
   1. the CPU reference of the model's hand-written adjoint against finite differences (double);
   2. (GPU) a crushed synthetic spiral (y squashed to 0.55, so the packed sheets are along y): true normals and density, the
      spiral fit anchored to count-field's LS U must reproduce the true winding (U' - theta / 2 pi) within 0.25 turn. */
#include "fit.h"
#include "spiralfit.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int cmd_count_field(int argc, char **argv);
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

enum { D = 8, H = 112, W = 112, h = 2 };
static const double p = 8.0, sq = 0.55, cy0 = 56 * h, cx0 = 56 * h;
static double trueW(double y, double x) { const double py = (y - cy0) / sq, px = x - cx0; return hypot(py, px) / p - atan2(py, px) / (2 * M_PI); }

int main(void) {
    int f = nn_spiral_gradcheck(getenv("SF_VERBOSE") != nullptr); setenv("SF_GC_FWD", "1", 1); f += nn_spiral_gradcheck(getenv("SF_VERBOSE") != nullptr); unsetenv("SF_GC_FWD");
    CHECK(f == 0, "gradient check: %d parameters off", f);
    if (!nn_spiral_has_gpu()) { printf("test_spiralfit: %s (no GPU: CPU part only)\n", fails ? "FAIL" : "ok"); return fails != 0; }
    char tmpl[] = "/tmp/test_spiralfit_XXXXXX"; char *dir = mkdtemp(tmpl); if (!dir) return 1;
    char in[512], mdir[512], mls[512]; snprintf(in, sizeof in, "%s/in", dir); snprintf(mdir, sizeof mdir, "%s/model", dir); snprintf(mls, sizeof mls, "%s/mls", dir);
    const size_t n = (size_t)D * H * W; uint8_t *a[10]; for (int k = 0; k < 10; k++) a[k] = calloc(n, 1);
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const size_t i = ((size_t)z * H + y) * W + x; const double py = (y + 0.5) * h, px = (x + 0.5) * h;
        if (hypot((py - cy0) / sq, px - cx0) < 3 * p || hypot((py - cy0) / sq, px - cx0) > 13 * p) continue;
        const double e = 0.25, gy = (trueW(py + e, px) - trueW(py - e, px)) / (2 * e), gx = (trueW(py, px + e) - trueW(py, px - e)) / (2 * e), gl = hypot(gy, gx);
        const double Wt = trueW(py, px), ny = gy / gl, nx = gx / gl;
        a[0][i] = (uint8_t)lrint(128 + 127 * 0.9 * cos(2 * M_PI * Wt)); a[1][i] = (uint8_t)lrint(128 + 127 * 0.9 * sin(2 * M_PI * Wt));
        a[3][i] = (uint8_t)lrint(255 * ny * ny); a[4][i] = (uint8_t)lrint(255 * nx * nx); a[5][i] = 128; a[6][i] = 128; a[7][i] = (uint8_t)lrint(255 * (0.5 + ny * nx));
        a[8][i] = (uint8_t)fmin(255, lrint(8 / (gl * h))); a[9][i] = 255;
    }
    const int na = D * h + 1; float *cy = malloc(na * sizeof(float)), *cx = malloc(na * sizeof(float)); for (int k = 0; k < na; k++) { cy[k] = (float)cy0; cx[k] = (float)cx0; }
    const long long org[3] = {0, 0, 0}; const int shp[3] = {D, H, W};
    if (fit_fields_save(in, org, shp, h, 9.6, a, cy, cx, na)) { fprintf(stderr, "cannot write the input\n"); return 1; }
    char *av[] = {"ufsm", "count-field", in, "/nonexistent", "--method", "spiral", "--R", "0", "--dir", "1", "--pool", "2", "--threads", "4", "--model", mdir, "--model-only", "1", "--rmin-um", "0",
                  "--sf-steps", "1500", "--sf-samp", "16384", "--sf-lf", "24", "--sf-sfac", "4", "--sf-la", "24", "--sf-lz", "64", "--sf-verbose", "500", nullptr};
    CHECK(cmd_count_field((int)(sizeof av / sizeof *av) - 1, av) == 0, "count-field --method spiral failed");
    char *av2[] = {"ufsm", "count-field", in, "/nonexistent", "--R", "0", "--dir", "1", "--pool", "2", "--threads", "4", "--model", mls, "--model-only", "1", "--rmin-um", "0", nullptr};
    CHECK(cmd_count_field((int)(sizeof av2 / sizeof *av2) - 1, av2) == 0, "count-field (LS) failed");
    fit_model *m = fit_load(mdir), *ml = fit_load(mls); CHECK(m && ml, "no model");
    if (m && ml) {
        double bias[2] = {0, 0}; int nn = 0;
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) { const double py = (y + 0.5) * h, px = (x + 0.5) * h, rr = hypot((py - cy0) / sq, px - cx0); if (rr < 4 * p || rr > 12 * p) continue;
            const double t = trueW(py, px); double d0 = fit_W(m, 8, py, px) - t, d1 = fit_W(ml, 8, py, px) - t; bias[0] += d0; bias[1] += d1; nn++; }
        bias[0] /= nn; bias[1] /= nn; int bad[2] = {0, 0};
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) { const double py = (y + 0.5) * h, px = (x + 0.5) * h, rr = hypot((py - cy0) / sq, px - cx0); if (rr < 4 * p || rr > 12 * p) continue;
            const double t = trueW(py, px); bad[0] += fabs(fit_W(m, 8, py, px) - t - round(bias[0])) > 0.25 + fabs(bias[0] - round(bias[0])); bad[1] += fabs(fit_W(ml, 8, py, px) - t - round(bias[1])) > 0.25 + fabs(bias[1] - round(bias[1])); }
        fprintf(stderr, "test_spiralfit: crushed spiral, cells off > 0.25 turn: spiral %d, LS %d of %d (bias %.3f / %.3f)\n", bad[0], bad[1], nn, bias[0], bias[1]);
        CHECK(bad[0] < nn / 10, "spiral model off on %d of %d cells", bad[0], nn);
    }
    char cmd[600]; snprintf(cmd, sizeof cmd, "rm -rf '%s'", dir); if (system(cmd)) {}
    printf("test_spiralfit: %s\n", fails ? "FAIL" : "ok");
    return fails != 0;
}
