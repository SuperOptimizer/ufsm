/* Core line (src/coreline.h): polyline signed distance, z blending, detection on a synthetic flattened scroll (nested
   racetracks around a core segment, tilted), half-box masks through fit_fields_load / fit_field_array. */
#include "coreline.h"
#include "fit.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static double seg_dist(double py, double px, double ay, double ax, double by, double bx) {
    const double ey = by - ay, ex = bx - ax, l2 = ey * ey + ex * ex; double t = ((py - ay) * ey + (px - ax) * ex) / l2; t = t < 0 ? 0 : t > 1 ? 1 : t;
    return hypot(py - ay - t * ey, px - ax - t * ex);
}

static void test_sdist(void) {
    const double Y[3] = {0, 0, 10}, X[3] = {0, 10, 10};   /* along +x, then a left turn... (y grows: towards +y) */
    double q[2], n[2];
    double d = cl_poly_sdist(Y, X, 3, 3, 5, q, n);   /* first segment, direction +x: left normal (1, 0) = +y */
    CHECK(fabs(d - 3) < 1e-9 && fabs(q[0]) < 1e-9 && fabs(q[1] - 5) < 1e-9 && fabs(n[0] - 1) < 1e-9, "sd %g q %g %g n %g %g", d, q[0], q[1], n[0], n[1]);
    d = cl_poly_sdist(Y, X, 3, -2, 5, q, n); CHECK(fabs(d + 2) < 1e-9, "sd %g", d);
    d = cl_poly_sdist(Y, X, 3, 5, 13, q, n);   /* second segment, direction +y: left normal (0, -1) = -x; x 13 is on the - side */
    CHECK(fabs(d + 3) < 1e-9 && fabs(n[1] + 1) < 1e-9, "sd %g n %g %g", d, n[0], n[1]);
    d = cl_poly_sdist(Y, X, 3, -3, 13, q, n);   /* beyond the outer corner: nearest = the vertex, the mean tangent decides: - side */
    CHECK(fabs(fabs(d) - hypot(3, 3)) < 1e-9 && d < 0, "corner sd %g", d);
    d = cl_poly_sdist(Y, X, 3, 2, 8, q, n);   /* inside the turn: + side */
    CHECK(d > 0 && fabs(d - 2) < 1e-9, "inner sd %g", d);
    /* z blending between two slabs */
    coreline C = {0}; C.um = 1; C.ns = 2; C.s = calloc(2, sizeof(cl_slab));
    double y0[2] = {0, 0}, x0[2] = {0, 10}, y1[2] = {2, 2}, x1[2] = {0, 10};
    C.s[0] = (cl_slab){0, 10, 0, 2, y0, x0}; C.s[1] = (cl_slab){10, 20, 10, 2, y1, x1};
    d = cl_sdist(&C, 5, 4, 5, q, n); CHECK(fabs(d - 3) < 1e-9 && fabs(q[0] - 1) < 1e-9, "blend sd %g q %g", d, q[0]);
    d = cl_sdist(&C, -5, 4, 5, q, n); CHECK(fabs(d - 4) < 1e-9, "below sd %g", d);
    d = cl_sdist(&C, 50, 4, 5, q, n); CHECK(fabs(d - 2) < 1e-9, "above sd %g", d);
    free(C.s);
}

/* a flattened scroll: sheets at integer phase = distance to the core segment / pitch, material within R of it; rotated by ang */
static void synth_detect(double ang, int fold) {
    const int H = 640, W = 640, bin = 4; const double pitch = 6, R = 110, cy = 320, cx = 320, hl = 170;   /* core half length */
    const double ay = cy - hl * sin(ang), ax = cx - hl * cos(ang), by = cy + hl * sin(ang), bx = cx + hl * cos(ang);
    uint8_t *pc = malloc((size_t)H * W), *ps = malloc((size_t)H * W), *w = malloc((size_t)H * W);
    srand(7);
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const size_t i = (size_t)y * W + x; double d = seg_dist(y + 0.5, x + 0.5, ay, ax, by, bx);
        if (fold) d += 8 * sin((x + 0.5) / 40.0);   /* wavy sheets */
        const double ph = 2 * M_PI * d / pitch, c = 0.4 + 0.5 * (rand() / (double)RAND_MAX);   /* noisy confidence */
        pc[i] = (uint8_t)lround(128 + 127 * c * cos(ph)); ps[i] = (uint8_t)lround(128 + 127 * c * sin(ph)); w[i] = d < R ? 255 : 0;
    }
    const int Hb = H / bin, Wb = W / bin;
    float *gy = calloc((size_t)Hb * Wb, 4), *gx = calloc((size_t)Hb * Wb, 4), *wb = calloc((size_t)Hb * Wb, 4);
    cl_accum_row(pc, ps, w, H, W, bin, gy, gx, wb);
    for (int i = 0; i < Hb * Wb; i++) wb[i] /= bin * bin;
    cl_params P = cl_params_default(); cl_line L;
    const int rc = cl_detect(gy, gx, wb, Hb, Wb, &P, &L);
    CHECK(rc == 0 && L.n > 10, "detect rc %d n %d", rc, L.n);
    if (!rc) {
        double worst = 0, cover0 = 1e9, cover1 = -1e9;
        for (int i = 0; i < L.n; i++) {
            const double py = L.y[i] * bin, px = L.x[i] * bin;   /* cells */
            worst = fmax(worst, seg_dist(py, px, ay, ax, by, bx));
            const double t = (py - cy) * sin(ang) + (px - cx) * cos(ang); cover0 = fmin(cover0, t); cover1 = fmax(cover1, t);
        }
        fprintf(stderr, "  synth ang %.2f fold %d: %d points, worst distance to the core %.1f cells, along [%.0f, %.0f] of [%.0f, %.0f], contrast %.2f\n",
                ang, fold, L.n, worst, cover0, cover1, -hl, hl, L.contrast);
        CHECK(worst < (fold ? 22 : 16), "core line %.1f cells off the segment", worst);
        CHECK(cover0 < -0.6 * hl && cover1 > 0.6 * hl, "core line covers [%.0f, %.0f] of +-%.0f", cover0, cover1, hl);
        CHECK(cover0 > -hl - R * 0.2 && cover1 < hl + R * 0.2, "core line runs into the caps [%.0f, %.0f]", cover0, cover1);
        cl_line_free(&L);
    }
    free(pc); free(ps); free(w); free(gy); free(gx); free(wb);
}

static void test_mask(void) {
    char dir[] = "/tmp/test_coreline_XXXXXX"; if (!mkdtemp(dir)) { CHECK(0, "mkdtemp"); return; }
    /* the core line: y = 16 + z / 4 (native = core-line voxels: um 2.399 both), direction +x: the + side is larger y */
    coreline C = {0}; C.um = 2.399; C.ns = 2; C.s = calloc(2, sizeof(cl_slab));
    double y0[2] = {16, 16}, x0[2] = {-100, 100}, y1[2] = {18, 18}, x1[2] = {-100, 100};
    C.s[0] = (cl_slab){0, 4, 0, 2, y0, x0}; C.s[1] = (cl_slab){4, 8, 8, 2, y1, x1};
    char p[512]; snprintf(p, sizeof p, "%s/core.json", dir); CHECK(cl_save(&C, p, nullptr) == 0, "save");
    coreline L; CHECK(cl_load(&L, p) == 0 && L.ns == 2 && L.s[1].n == 2 && fabs(L.s[1].y[0] - 18) < 1e-6, "load");
    cl_free(&L); free(C.s);
    const int D = 8, H = 32, Wd = 32; static const char *nm[10] = {"pc", "ps", "t0", "t1", "t2", "t3", "t4", "t5", "sp", "w"};
    uint8_t *a = malloc((size_t)D * H * Wd);
    for (int k = 0; k < 10; k++) {
        memset(a, k == 9 ? 200 : k < 2 ? 250 : 100, (size_t)D * H * Wd);
        char q[600]; snprintf(q, sizeof q, "%s/%s.raw", dir, nm[k]); FILE *f = fopen(q, "wb"); fwrite(a, 1, (size_t)D * H * Wd, f); fclose(f);
    }
    snprintf(p, sizeof p, "%s/fields.json", dir); FILE *f = fopen(p, "w");
    fprintf(f, "{\"format\": \"raw\", \"origin\": [0, 0, 0], \"shape\": [%d, %d, %d], \"h\": 1, \"um\": 2.399, \"core_line\": \"%s/core.json\", \"core_side\": -1, \"axis\": [[0, 0]]}\n", D, H, Wd, dir);
    fclose(f);
    fit_fields_buf b;
    CHECK(fit_fields_load(dir, 0, 0, 1, &b) == 0, "load fields");
    int bad = 0;
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < Wd; x++) {
        const size_t i = ((size_t)z * H + y) * Wd + x; const double core = 16 + (z + 0.5) / 4;
        const int keep = y + 0.5 < core;   /* side -1: below the line */
        if (keep != (b.a[9][i] != 0) || (!keep && (b.a[0][i] != 128 || b.a[1][i] != 128)) || (keep && b.a[0][i] != 250) || b.a[2][i] != 100) bad++;
    }
    CHECK(bad == 0, "%d cells masked wrongly (fit_fields_load)", bad);
    fit_fields_free(&b);
    int sh[3]; uint8_t *w = fit_field_array(dir, "w", 1, sh); bad = 0;
    for (int z = 0; z < D && w; z++) for (int y = 0; y < H; y++) for (int x = 0; x < Wd; x++) { const int keep = y + 0.5 < 16 + (z + 0.5) / 4; if (keep != (w[((size_t)z * H + y) * Wd + x] != 0)) bad++; }
    CHECK(w && bad == 0, "%d cells masked wrongly (fit_field_array)", bad); free(w);
    uint8_t *t0 = fit_field_array(dir, "t0", 1, sh); size_t nz = 0; for (size_t i = 0; t0 && i < (size_t)D * H * Wd; i++) nz += t0[i] == 100;
    CHECK(t0 && nz == (size_t)D * H * Wd, "t0 must not be masked"); free(t0);
    /* pooled: the mask applies before pooling (material in every pooled cell that holds kept cells) */
    CHECK(fit_fields_load(dir, 0, 0, 2, &b) == 0, "pooled load");
    fit_fields_free(&b);
    for (int k = 0; k < 10; k++) { snprintf(p, sizeof p, "%s/%s.raw", dir, nm[k]); unlink(p); }
    snprintf(p, sizeof p, "%s/fields.json", dir); unlink(p); snprintf(p, sizeof p, "%s/core.json", dir); unlink(p); rmdir(dir); free(a);
}

int main(void) {
    test_sdist();
    synth_detect(0.0, 0);
    synth_detect(0.6, 0);
    synth_detect(-1.1, 1);
    test_mask();
    if (fails) { fprintf(stderr, "test_coreline: %d failure(s)\n", fails); return 1; }
    fprintf(stderr, "test_coreline: ok\n");
    return 0;
}
