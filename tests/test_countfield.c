/* ufsm count-field (src/countfield.c) on a synthetic spiral: W = r / p - theta / 2 pi, radial normals, the true density; the phase
   skips every other sheet in one quadrant (it rotates at half the rate there). W from the density must count every sheet there,
   the output phase (R 0) must follow frac(W) where the phase is right, and --model must reproduce W (fit_W). */
#include "fit.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int cmd_count_field(int argc, char **argv);
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

enum { D = 6, H = 96, W = 96, h = 2 };
static const double p = 8.0, cy0 = 48 * h, cx0 = 48 * h;   /* pitch (native voxels), axis at the box centre */
static int bad_quadrant(double th) { return th > 0.2 && th < M_PI / 2 - 0.2; }
static double trueW(double y, double x) { return hypot(y - cy0, x - cx0) / p - atan2(y - cy0, x - cx0) / (2 * M_PI); }

int main(void) {
    char tmpl[] = "/tmp/test_countfield_XXXXXX"; char *dir = mkdtemp(tmpl); if (!dir) return 1;
    char in[512], out[512], mdir[512]; snprintf(in, sizeof in, "%s/in", dir); snprintf(out, sizeof out, "%s/out", dir); snprintf(mdir, sizeof mdir, "%s/model", dir);
    const size_t n = (size_t)D * H * W; uint8_t *a[10]; for (int k = 0; k < 10; k++) a[k] = calloc(n, 1);
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const size_t i = ((size_t)z * H + y) * W + x; const double py = (y + 0.5) * h, px = (x + 0.5) * h, ry = py - cy0, rx = px - cx0, r = hypot(ry, rx);
        if (r < 3 * p) continue;
        const double th = atan2(ry, rx), Wt = trueW(py, px), ph = bad_quadrant(th) ? Wt / 2 : Wt;   /* the phase skips sheets in one quadrant */
        a[0][i] = (uint8_t)lrint(128 + 127 * 0.9 * cos(2 * M_PI * ph)); a[1][i] = (uint8_t)lrint(128 + 127 * 0.9 * sin(2 * M_PI * ph));
        const double nz = 0, ny = ry / r, nx = rx / r;   /* n n^T planes: n_z^2 n_y^2 n_x^2, 0.5 + n_z n_y, 0.5 + n_z n_x, 0.5 + n_y n_x */
        a[2][i] = (uint8_t)lrint(255 * nz * nz); a[3][i] = (uint8_t)lrint(255 * ny * ny); a[4][i] = (uint8_t)lrint(255 * nx * nx);
        a[5][i] = (uint8_t)lrint(255 * (0.5 + nz * ny)); a[6][i] = (uint8_t)lrint(255 * (0.5 + nz * nx)); a[7][i] = (uint8_t)lrint(255 * (0.5 + ny * nx));
        a[8][i] = (uint8_t)lrint(8 * p / h); a[9][i] = 255;   /* sp = 8 spacing / h; material */
    }
    const int na = D * h + 1; float *cy = malloc(na * sizeof(float)), *cx = malloc(na * sizeof(float)); for (int k = 0; k < na; k++) { cy[k] = (float)cy0; cx[k] = (float)cx0; }
    const long long org[3] = {0, 0, 0}; const int shp[3] = {D, H, W};
    if (fit_fields_save(in, org, shp, h, 9.6, a, cy, cx, na)) { fprintf(stderr, "cannot write the input\n"); return 1; }
    char *av[] = {"ufsm", "count-field", in, out, "--R", "0", "--dir", "1", "--pool", "1", "--threads", "4", "--model", mdir, "--model-R", "4", "--rmin-um", "0", nullptr};
    CHECK(cmd_count_field(18, av) == 0, "count-field failed");
    char pth[600]; snprintf(pth, sizeof pth, "%s/wc.f32", out); FILE *f = fopen(pth, "rb"); float *wc = malloc(n * sizeof(float));
    CHECK(f && fread(wc, sizeof(float), n, f) == n, "no wc.f32"); if (f) fclose(f);
    fit_fields_buf ob; CHECK(fit_fields_load(out, 0, 0, 1, &ob) == 0, "cannot load the output");
    /* counts along radial segments (r 12 .. 36 cells) in the bad quadrant: W_out vs truth, and the input phase's count */
    double emax = 0, ph_err = 0, gerr = 0; int ng = 0;
    for (int k = 0; k < 8; k++) {
        const double th = 0.3 + k * (M_PI / 2 - 0.6) / 7, r0 = 12 * h, r1 = 36 * h;
        const int y0 = (int)((cy0 + r0 * sin(th)) / h), x0 = (int)((cx0 + r0 * cos(th)) / h), y1 = (int)((cy0 + r1 * sin(th)) / h), x1 = (int)((cx0 + r1 * cos(th)) / h);
        const size_t i0 = ((size_t)2 * H + y0) * W + x0, i1 = ((size_t)2 * H + y1) * W + x1;
        const double tw = trueW((y1 + 0.5) * h, (x1 + 0.5) * h) - trueW((y0 + 0.5) * h, (x0 + 0.5) * h), cw = wc[i1] - wc[i0];
        if (fabs(cw - tw) > emax) emax = fabs(cw - tw);
        ph_err = fmax(ph_err, fabs(tw / 2 - tw));
    }
    CHECK(emax < 0.3, "bad-quadrant count: W off by %.3f turns over 24 cells (the phase alone by %.2f)", emax, ph_err);
    /* the good quadrants: the output phase = the input phase (= frac W) up to the gauge, little error */
    for (int y = 0; y < H; y += 3) for (int x = 0; x < W; x += 3) {
        const size_t i = ((size_t)3 * H + y) * W + x; const double py = (y + 0.5) * h, px = (x + 0.5) * h, r = hypot(py - cy0, px - cx0), th = atan2(py - cy0, px - cx0);
        if (r < 6 * p || r > 44 * h || th > -0.2) continue;   /* lower half: phase right */
        const double po = atan2((ob.f.ps[i] - 128) / 127.0, (ob.f.pc[i] - 128) / 127.0) / (2 * M_PI); double e = po - trueW(py, px); e -= round(e);
        gerr += fabs(e); ng++;
    }
    CHECK(ng > 100 && gerr / ng < 0.08, "output phase vs frac W where the phase is right: mean |error| %.3f turns (%d cells)", ng ? gerr / ng : 1, ng);
    /* the model: fit_W reproduces W_out's differences */
    fit_model *m = fit_load(mdir); CHECK(m != nullptr, "no model");
    if (m) {
        double me = 0; int nm = 0;
        for (int k = 0; k < 200; k++) {
            const double th = -2.9 + 2.6 * k / 200, r = (14 + (k % 20)) * h, py = cy0 + r * sin(th), px = cx0 + r * cos(th);
            const int y = (int)(py / h), x = (int)(px / h); const size_t i = ((size_t)3 * H + y) * W + x; if (!(wc[i] == wc[i])) continue;
            const double d = fit_W(m, 3.5 * h, (y + 0.5) * h, (x + 0.5) * h) - wc[i]; me += fabs(d - round(d)); nm++;
        }
        CHECK(nm > 100 && me / nm < 0.15, "model W vs wc: mean |frac error| %.3f (%d points)", nm ? me / nm : 1, nm);
        fit_free(m);
    }
    fit_fields_free(&ob);
    char cmd[700]; snprintf(cmd, sizeof cmd, "rm -rf '%s'", dir); if (system(cmd)) {}
    for (int k = 0; k < 10; k++) free(a[k]);
    free(cy); free(cx); free(wc);
    if (fails) { fprintf(stderr, "test_countfield: %d failures\n", fails); return 1; }
    printf("test_countfield: OK (bad-quadrant count error %.3f turns vs the phase's %.2f)\n", emax, ph_err);
    return 0;
}
