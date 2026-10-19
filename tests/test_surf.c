/* ufsm refine / surf-stats (src/surf.c) on synthetic fit fields: an Archimedean spiral W = r / p - theta / 2 pi around a vertical axis,
   phase 2 pi W (with a wavy offset of the sheets, so the fitted-looking smooth level set is off the sheet by up to 0.12 turn), and a
   tifxyz wrap at a smooth radius 0.2 turn off the sheet (the level set of a too-smooth fit). refine must bring the vertices onto
   W = k (|W - k| mean < 0.03 turn, from ~0.2) and, with a target offset, onto W = k + target (a corner of the box has no phase);
   surf-stats runs on both sets. */
#include "fit.h"
#include "tiff.h"
#include "tiffw.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int cmd_refine(int argc, char **argv);
int cmd_surf_stats(int argc, char **argv);
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

enum { D = 16, H = 96, Wd = 96, HH = 2 };   /* field cells (h = 2 native voxels of 9.6 um), the box at native (100, 200, 300) */
static const double P = 24, CY = 96, CX = 96, UM = 9.6, WUM = 4.8;   /* pitch (native), the axis (box-local native) */
static const long long ORG[3] = {100, 200, 300};
static double wav(double z, double th) { return 0.12 * sin(3 * th + z / 9); }   /* the sheets' wiggle (turns) */
static double W_of(double z, double y, double x) {   /* box-local native */
    const double r = hypot(y - CY, x - CX), th = atan2(y - CY, x - CX);
    return r / P - th / (2 * M_PI) + wav(z, th);
}
static float *tif(const char *p, int *w, int *h) {
    tiff *t = tiff_open_file(p); tiff_page pg; if (!t || tiff_page_info(t, 0, &pg)) { if (t) tiff_close(t); return nullptr; }
    float *d = malloc((size_t)pg.w * pg.h * 4); tiff_read_page(t, 0, d); *w = pg.w; *h = pg.h; tiff_close(t); return d;
}
/* the wrap k: radius where the smooth W (no wiggle) is k + off, rows every native z, columns every 0.02 rad (inside the box) */
static void make_wrap(const char *dir, int k, double off) {
    const int rows = D * HH, cols = 300; float *X = malloc(rows * cols * 4), *Y = malloc(rows * cols * 4), *Z = malloc(rows * cols * 4);
    for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) {
        const double th = -M_PI + 0.3 + c * 0.02, R = (k + off + th / (2 * M_PI)) * P, z = r + 0.5, y = CY + R * sin(th), x = CX + R * cos(th);
        const size_t i = (size_t)r * cols + c;
        const int in = y > 4 && y < H * HH - 4 && x > 4 && x < Wd * HH - 4;
        Z[i] = in ? (float)((z + ORG[0] + 0.5) * UM / WUM - 0.5) : -1; Y[i] = in ? (float)((y + ORG[1] + 0.5) * UM / WUM - 0.5) : -1; X[i] = in ? (float)((x + ORG[2] + 0.5) * UM / WUM - 0.5) : -1;
    }
    char p[512]; snprintf(p, sizeof p, "%s/w%03d", dir, k);
    tifxyz_write(p, cols, rows, X, Y, Z, 1.0, WUM, k, "test_surf");
    free(X); free(Y); free(Z);
}
/* mean |W - k - target| over the wrap's vertices (W with the wiggle: the true sheets) */
static double wrap_err(const char *dir, int k, double target, size_t *nv) {
    char p[512]; int w, h, w2, h2, w3, h3;
    snprintf(p, sizeof p, "%s/w%03d/x.tif", dir, k); float *X = tif(p, &w, &h);
    snprintf(p, sizeof p, "%s/w%03d/y.tif", dir, k); float *Y = tif(p, &w2, &h2);
    snprintf(p, sizeof p, "%s/w%03d/z.tif", dir, k); float *Z = tif(p, &w3, &h3);
    double s = 0; *nv = 0;
    if (X && Y && Z) for (size_t i = 0; i < (size_t)w * h; i++) {
        if (Z[i] < 0) continue;
        const double z = (Z[i] + 0.5) * WUM / UM - 0.5 - ORG[0], y = (Y[i] + 0.5) * WUM / UM - 0.5 - ORG[1], x = (X[i] + 0.5) * WUM / UM - 0.5 - ORG[2];
        if (x > 140 && y > 140) continue;   /* the corner without phase data (and its edge) */
        double e = W_of(z, y, x) - k - target; e -= round(e); s += fabs(e); (*nv)++;
    }
    free(X); free(Y); free(Z);
    return *nv ? s / *nv : 1;
}
int main(void) {
    char base[] = "/tmp/test_surf_XXXXXX"; if (!mkdtemp(base)) return 1;
    char fd[600], w0[600], w1[600], w2[600]; snprintf(fd, sizeof fd, "%s/fields", base); snprintf(w0, sizeof w0, "%s/wraps", base);
    snprintf(w1, sizeof w1, "%s/refined", base); snprintf(w2, sizeof w2, "%s/target", base);
    const size_t n = (size_t)D * H * Wd; uint8_t *a[10]; for (int k = 0; k < 10; k++) a[k] = calloc(n, 1);
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < Wd; x++) {
        const size_t i = ((size_t)z * H + y) * Wd + x; const double pz = (z + 0.5) * HH, py = (y + 0.5) * HH, px = (x + 0.5) * HH;
        const double Wv = W_of(pz, py, px), ph = 2 * M_PI * Wv, nodata = px > 150 && py > 150;   /* a corner without phase */
        a[0][i] = nodata ? 128 : (uint8_t)lround(128 + 127 * cos(ph)); a[1][i] = nodata ? 128 : (uint8_t)lround(128 + 127 * sin(ph));
        a[2][i] = 0; a[3][i] = 128; a[4][i] = 128; a[5][i] = 128; a[6][i] = 128; a[7][i] = 128;
        a[8][i] = (uint8_t)lround(P * 8 / HH); a[9][i] = 255;
    }
    const int na = D * HH + 1; float *cy = malloc(na * 4), *cx = malloc(na * 4); for (int z = 0; z < na; z++) { cy[z] = (float)CY; cx[z] = (float)CX; }
    const int shp[3] = {D, H, Wd};
    if (fit_fields_save(fd, ORG, shp, HH, UM, a, cy, cx, na)) { fprintf(stderr, "FAIL: fields\n"); return 1; }
    for (int k = 2; k <= 5; k++) make_wrap(w0, k, 0.2);
    size_t nv; const double e0 = wrap_err(w0, 3, 0, &nv);
    char *rv[] = {"ufsm", "refine", w0, w1, "--fields", fd, "--range", "0.4", "--smooth-um", "10", "--threads", "4", "--verbose", nullptr};
    CHECK(cmd_refine(13, rv) == 0, "refine rc");
    const double e1 = wrap_err(w1, 3, 0, &nv);
    printf("wrap 3: mean |W - 3| %.4f before, %.4f after refine (%zu vertices)\n", e0, e1, nv);
    CHECK(e0 > 0.1, "the test wrap should start off the sheet (%.4f)", e0);
    CHECK(e1 < 0.03, "refine: mean |W - k| %.4f >= 0.03", e1);
    char *tv[] = {"ufsm", "refine", w0, w2, "--fields", fd, "--range", "0.45", "--smooth-um", "10", "--target", "0.25", "--threads", "4", nullptr};
    CHECK(cmd_refine(14, tv) == 0, "refine --target rc");
    const double e2 = wrap_err(w2, 3, 0.25, &nv);
    printf("wrap 3: mean |W - 3.25| %.4f after refine --target 0.25\n", e2);
    CHECK(e2 < 0.03, "refine --target 0.25: mean |W - k - 0.25| %.4f >= 0.03", e2);
    char *sv[] = {"ufsm", "surf-stats", w1, "--fields", fd, "--threads", "4", "--per-wrap", nullptr};
    CHECK(cmd_surf_stats(8, sv) == 0, "surf-stats rc");
    char *sv0[] = {"ufsm", "surf-stats", w0, "--fields", fd, "--threads", "4", nullptr};
    CHECK(cmd_surf_stats(7, sv0) == 0, "surf-stats rc");
    char cmd[700]; snprintf(cmd, sizeof cmd, "rm -rf '%s'", base); if (system(cmd)) {}
    for (int k = 0; k < 10; k++) free(a[k]);
    free(cy); free(cx);
    if (fails) { fprintf(stderr, "test_surf: %d failure(s)\n", fails); return 1; }
    printf("test_surf: OK\n");
    return 0;
}
