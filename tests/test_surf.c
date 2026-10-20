/* ufsm refine / surf-stats (src/surf.c) on synthetic fit fields: an Archimedean spiral W = r / p - theta / 2 pi around a vertical axis,
   phase 2 pi W (with a wavy offset of the sheets, so the fitted-looking smooth level set is off the sheet by up to 0.12 turn), and a
   tifxyz wrap at a smooth radius 0.2 turn off the sheet (the level set of a too-smooth fit). refine must bring the vertices onto
   W = k (|W - k| mean < 0.03 turn, from ~0.2) and, with a target offset, onto W = k + target (a corner of the box has no phase);
   surf-stats runs on both sets. fit_set_gradW on a hand-made model (exact, across theta's cut; a 3-turn step is a fault), and
   surf-stats --model (faults, the largest clean sheet). */
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
int cmd_self_gate(int argc, char **argv);
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
/* the kept vertices of a self-gate output: count, how many lie in the no-phase corner (x, y > 156: clear of the cells next to it) or at native x < xmax, and whether
   every q.tif value equals its wrap's id */
static size_t gate_count(const char *dir, size_t *corner, size_t *lowx, double xmax, int *qok) {
    size_t n = 0; *corner = *lowx = 0; *qok = 1;
    for (int k = 0; k < 8; k++) {
        char p[700]; int w, h, w2, h2, w3, h3, w4, h4; snprintf(p, sizeof p, "%s/w%03d/x.tif", dir, k); float *X = tif(p, &w, &h); if (!X) continue;
        snprintf(p, sizeof p, "%s/w%03d/y.tif", dir, k); float *Y = tif(p, &w2, &h2); snprintf(p, sizeof p, "%s/w%03d/z.tif", dir, k); float *Z = tif(p, &w3, &h3);
        snprintf(p, sizeof p, "%s/w%03d/q.tif", dir, k); float *Q = tif(p, &w4, &h4);
        if (!Y || !Z || !Q || w4 != w || h4 != h) *qok = 0;
        else for (size_t i = 0; i < (size_t)w * h; i++) {
            if (Z[i] < 0) { if (!isnan(Q[i])) *qok = 0; continue; }
            n++; if (Q[i] != (float)k) *qok = 0;
            const double y = (Y[i] + 0.5) * WUM / UM - 0.5 - ORG[1], x = (X[i] + 0.5) * WUM / UM - 0.5 - ORG[2];
            *corner += x > 156 && y > 156; *lowx += x < xmax;
        }
        free(X); free(Y); free(Z); free(Q);
    }
    return n;
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
    /* the fit's W by hand (W = r / P - theta / 2 pi, no wiggle): fit_set_gradW = 1 / P everywhere, also across theta's cut (x < CX,
       y = CY); a model with a 3-turn step in delta over 2 native voxels has |grad W| > 3 x the density there (a fault); surf-stats --model */
    {
        fit_model m = {0}; m.dir = 1; m.nk = 2; m.knot_dz = 1024; float pit[2] = {(float)P, (float)P}, off[2] = {0, 0}; m.pitch = pit; m.offset = off;
        m.L = 8; m.lz = D * HH / 8 + 2; m.ly = H * HH / 8 + 2; m.lx = Wd * HH / 8 + 2; m.delta = calloc((size_t)m.lz * m.ly * m.lx, 4);
        m.D = D; m.H = H; m.W = Wd; m.h = HH; for (int d = 0; d < 3; d++) m.origin[d] = ORG[d]; m.um = UM; m.cy = cy; m.cx = cx;
        char md[600]; snprintf(md, sizeof md, "%s/model", base); CHECK(!fit_save(&m, md), "fit_save");
        fit_set *fs = fit_set_load(md); CHECK(fs != nullptr, "fit_set_load");
        if (fs) {
            double g[3], worst = 0; const double pts[4][3] = {{10, CY, CX - 40}, {10, CY + 0.3, CX - 50}, {20, CY - 30, CX + 20}, {5, CY + 50, CX - 50}};
            for (int k = 0; k < 4; k++) {   /* |grad W| = sqrt(1 / P^2 + 1 / (2 pi r)^2) */
                const double r = hypot(pts[k][1] - CY, pts[k][2] - CX), ex = sqrt(1 / (P * P) + 1 / (4 * M_PI * M_PI * r * r));
                worst = fmax(worst, fabs(fit_set_gradW(fs, pts[k][0], pts[k][1], pts[k][2], 1.0, g) / ex - 1));
            }
            printf("fit_set_gradW: relative error worst %.2g (the theta cut included)\n", worst);
            CHECK(worst < 1e-3, "fit_set_gradW: relative error %.3g", worst);
            fit_set_free(fs);
        }
        for (int a = 0; a < m.lz; a++) for (int b = 0; b < m.ly; b++) for (int c = 8; c < m.lx; c++) m.delta[((size_t)a * m.ly + b) * m.lx + c] = 3;   /* x >= 64: +3 turns */
        char mf[600]; snprintf(mf, sizeof mf, "%s/model-fault", base); CHECK(!fit_save(&m, mf), "fit_save");
        fs = fit_set_load(mf);
        if (fs) { double g[3]; const double gl = fit_set_gradW(fs, 10, 40, 60, 1.0, g); printf("fault model: |grad W| %.3f at the step (density %.3f)\n", gl, 1 / P);
                  CHECK(gl > 3 / P, "the 3-turn step is no fault (%.3f)", gl); fit_set_free(fs); }
        char *mv[] = {"ufsm", "surf-stats", w1, "--fields", fd, "--model", mf, "--threads", "4", nullptr};
        CHECK(cmd_surf_stats(9, mv) == 0, "surf-stats --model rc");
        /* self-gate: the refined wraps sit on the phase (conf / residual pass) except in the no-phase corner; the fault model gates the
           vertices around its 3-turn step; a neighbour equal to the model agrees everywhere, the fault model as neighbour disagrees on
           one side of its step; --exclude drops a slab of x */
        char g1[600], g2[600], g3[600], g4[600], g5[600], ex[128];
        snprintf(g1, sizeof g1, "%s/gate1", base); snprintf(g2, sizeof g2, "%s/gate2", base); snprintf(g3, sizeof g3, "%s/gate3", base);
        snprintf(g4, sizeof g4, "%s/gate4", base); snprintf(g5, sizeof g5, "%s/gate5", base);
        snprintf(ex, sizeof ex, "0,0,0,1000,1000,%lld", ORG[2] + 100);   /* global native x < ORG + 100 */
        char *gv1[] = {"ufsm", "self-gate", w1, g1, "--fields", fd, "--model", md, "--min-area", "0", "--erode", "0", "--threads", "4", "--per-wrap", nullptr};
        char *gv2[] = {"ufsm", "self-gate", w1, g2, "--fields", fd, "--model", mf, "--min-area", "0", "--erode", "0", "--threads", "4", nullptr};
        char *gv3[] = {"ufsm", "self-gate", w1, g3, "--fields", fd, "--model", md, "--nb", md, "--min-area", "0", "--erode", "0", "--threads", "4", nullptr};
        char *gv4[] = {"ufsm", "self-gate", w1, g4, "--fields", fd, "--model", md, "--nb", mf, "--nb-min", "0", "--min-area", "0", "--erode", "0", "--threads", "4", nullptr};
        char *gv5[] = {"ufsm", "self-gate", w1, g5, "--fields", fd, "--model", md, "--exclude", ex, "--min-area", "0", "--threads", "4", nullptr};
        CHECK(cmd_self_gate(15, gv1) == 0, "self-gate rc"); CHECK(cmd_self_gate(14, gv2) == 0, "self-gate fault rc");
        CHECK(cmd_self_gate(16, gv3) == 0, "self-gate --nb rc"); CHECK(cmd_self_gate(18, gv4) == 0, "self-gate --nb fault rc");
        CHECK(cmd_self_gate(14, gv5) == 0, "self-gate --exclude rc");
        size_t c1, l1, c2, l2, c3, l3, c4, l4, c5, l5, all = 0; int q1, q2, q3, q4, q5;
        for (int k = 2; k <= 5; k++) { size_t v; wrap_err(w1, k, 0, &v); all += v; }   /* (vertices outside the x, y > 140 corner) */
        const size_t n1 = gate_count(g1, &c1, &l1, 100, &q1), n2 = gate_count(g2, &c2, &l2, 100, &q2), n3 = gate_count(g3, &c3, &l3, 100, &q3);
        const size_t n4 = gate_count(g4, &c4, &l4, 100, &q4), n5 = gate_count(g5, &c5, &l5, 100, &q5);
        printf("self-gate: kept %zu (of %zu outside the corner), fault model %zu, nb = model %zu, nb = fault model %zu, excluded x < 100: %zu (%zu there)\n", n1, all, n2, n3, n4, n5, l5);
        CHECK(q1 && q2 && q3 && q4 && q5, "self-gate: q.tif must hold the wrap id at kept vertices, NaN elsewhere");
        CHECK(c1 == 0 && c3 == 0 && c5 == 0, "self-gate: %zu / %zu / %zu kept vertices in the no-phase corner", c1, c3, c5);
        CHECK(n1 > all * 0.9 && n1 < all * 1.2, "self-gate: kept %zu of %zu on-phase vertices", n1, all);
        CHECK(n2 < n1 * 0.95, "self-gate: the fault model keeps %zu (no fault gated, plain %zu)", n2, n1);
        CHECK(n3 == n1, "self-gate: a neighbour equal to the model changed the kept count (%zu vs %zu)", n3, n1);
        CHECK(n4 < n1 * 0.9 && n4 > 0, "self-gate: the fault neighbour keeps %zu of %zu", n4, n1);
        CHECK(l5 == 0 && n5 > 0 && l1 > 0, "self-gate --exclude: %zu kept vertices at x < 100 (%zu without it)", l5, l1);
        free(m.delta);
    }
    char cmd[700]; snprintf(cmd, sizeof cmd, "rm -rf '%s'", base); if (system(cmd)) {}
    for (int k = 0; k < 10; k++) free(a[k]);
    free(cy); free(cx);
    if (fails) { fprintf(stderr, "test_surf: %d failure(s)\n", fails); return 1; }
    printf("test_surf: OK\n");
    return 0;
}
