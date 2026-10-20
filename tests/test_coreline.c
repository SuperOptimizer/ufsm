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

/* cl_poly_sdist before the group pruning and the deferred tangent (the reference: every segment, the tangent at each improvement) */
static double ref_poly_sdist(const double *Y, const double *X, int n, double py, double px, double q[2], double nrm[2]) {
    double best = 1e300, qy = Y[0], qx = X[0], ty = 0, tx = 1;
    for (int i = 0; i + 1 < n; i++) {
        const double ey = Y[i + 1] - Y[i], ex = X[i + 1] - X[i], l2 = ey * ey + ex * ex; if (l2 <= 0) continue;
        double t = ((py - Y[i]) * ey + (px - X[i]) * ex) / l2; t = t < 0 ? 0 : t > 1 ? 1 : t;
        const double cy = Y[i] + t * ey, cx = X[i] + t * ex, d2 = (py - cy) * (py - cy) + (px - cx) * (px - cx);
        if (d2 < best) {
            best = d2; qy = cy; qx = cx; const double l = sqrt(l2); ty = ey / l; tx = ex / l;
            const int v = t <= 0 ? i : t >= 1 ? i + 1 : -1;
            if (v > 0 && v < n - 1) {
                const double ay = Y[v] - Y[v - 1], ax = X[v] - X[v - 1], by = Y[v + 1] - Y[v], bx = X[v + 1] - X[v];
                const double la = hypot(ay, ax), lb = hypot(by, bx);
                if (la > 0 && lb > 0) { double my = ay / la + by / lb, mx = ax / la + bx / lb, lm = hypot(my, mx); if (lm > 1e-9) { ty = my / lm; tx = mx / lm; } }
            }
        }
    }
    const double ny = tx, nx = -ty; q[0] = qy; q[1] = qx; nrm[0] = ny; nrm[1] = nx;
    const double s = (py - qy) * ny + (px - qx) * nx;
    return (s < 0 ? -1 : 1) * sqrt(best);
}
static void test_sdist_fast(void) {   /* the pruned scan: the same distance, point and normal, bit for bit */
    srand(5); int bad = 0, tot = 0;
    for (int rep = 0; rep < 40; rep++) {
        const int n = 2 + rand() % 120; double Y[128], X[128], y = 3000, x = 2000;
        for (int i = 0; i < n; i++) {   /* a zigzag walk, repeated points and back-tracks included */
            Y[i] = y; X[i] = x; const int k = rand() % 10;
            if (k) { y += (rand() % 2001 - 1000) / 37.0; x += 20 + (rand() % 400) / 7.0; } }
        for (int t = 0; t < 2000; t++) {
            const double py = 3000 + (rand() % 40001 - 20000) / 10.0, px = 1500 + (rand() % 120001) / 10.0;
            const int v = rand() % n; const double qy0 = t % 3 ? py : Y[v] + (rand() % 3 - 1), qx0 = t % 3 ? px : X[v] + (rand() % 3 - 1);   /* also near vertices */
            double q[2], m[2], qr[2], mr[2];
            const double d = cl_poly_sdist(Y, X, n, qy0, qx0, q, m), dr = ref_poly_sdist(Y, X, n, qy0, qx0, qr, mr);
            bad += memcmp(&d, &dr, 8) || memcmp(q, qr, 16) || memcmp(m, mr, 16); tot++;
        }
    }
    CHECK(bad == 0, "the pruned cl_poly_sdist differs from the full scan at %d of %d points", bad, tot);
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

/* the box frame of a zigzag core line: the raw nearest-point normal swings by +-45 degrees, the fitted line's normal does not */
static void test_frame(void) {
    enum { NP = 41 }; coreline C = {0}; C.um = 1; C.ns = 2; C.s = calloc(2, sizeof(cl_slab));
    for (int k = 0; k < 2; k++) {
        cl_slab *s = &C.s[k]; s->z0 = k * 100; s->z1 = s->z0 + 100; s->zc = s->z0 + 50; s->n = NP; s->y = malloc(NP * sizeof(double)); s->x = malloc(NP * sizeof(double));
        for (int i = 0; i < NP; i++) { s->y[i] = 500 + (i % 2 ? 10 : -10); s->x[i] = 100 + 20 * i; }   /* along x at y 500, zigzag +-10 */
    }
    double q[2], n[2]; const long long org[3] = {0, 0, 0};
    for (int cx = 300; cx <= 700; cx += 37) {
        double rq[2], rn[2]; cl_sdist(&C, 50, 450, cx, rq, rn);
        cl_box_frame(&C, org, 200, 450, cx, 1.0, 200, q, n);
        CHECK(fabs(fabs(n[0]) - 1) < 1e-3 && n[0] * rn[0] + n[1] * rn[1] > 0, "frame normal (%.3f, %.3f) at x %d (raw (%.3f, %.3f))", n[0], n[1], cx, rn[0], rn[1]);
        CHECK(fabs(q[0] - 500) <= 10 && fabs(q[1] - cx) < 25, "frame point (%.1f, %.1f) at x %d", q[0], q[1], cx);
    }
    cl_free(&C);
}
/* the threaded, cached masks (fit_fields_load / fit_field_array: per-row cache of the blanked cells) and the merged pooled load
   (fit_fields_load_ex: pool-1 loads filling a pooled box) give the serial cl_mask_block arrays bit for bit: a zigzag core line, random
   data with empty 4 x 4 blocks, rows loaded repeatedly in overlapping ranges */
static void test_mask_cache(void) {
    char dir[] = "/tmp/test_coreline_XXXXXX"; if (!mkdtemp(dir)) { CHECK(0, "mkdtemp"); return; }
    coreline C = {0}; C.um = 2.399; C.ns = 3; C.s = calloc(3, sizeof(cl_slab));
    double y0[6] = {30, 42, 28, 45, 33, 40}, x0[6] = {-5, 12, 25, 41, 58, 75}, y1[5] = {50, 38, 52, 36, 49}, x1[5] = {-3, 20, 37, 55, 80};
    double y2[4] = {35, 47, 31, 44}, x2[4] = {0, 26, 50, 72};
    C.s[0] = (cl_slab){0, 40, 20, 6, y0, x0}; C.s[1] = (cl_slab){40, 80, 60, 5, y1, x1}; C.s[2] = (cl_slab){80, 120, 100, 4, y2, x2};
    char p[512]; snprintf(p, sizeof p, "%s/core.json", dir); CHECK(cl_save(&C, p, nullptr) == 0, "save");
    const int D = 44, H = 37, Wd = 41, h = 2; const size_t n = (size_t)D * H * Wd; static const char *nm[10] = {"pc", "ps", "t0", "t1", "t2", "t3", "t4", "t5", "sp", "w"};
    uint8_t *a[10]; srand(11);
    for (int k = 0; k < 10; k++) {
        a[k] = malloc(n);
        for (size_t i = 0; i < n; i++) a[k][i] = (uint8_t)(k == 8 ? 1 + rand() % 40 : rand() % 256);
    }
    for (size_t i = 0; i < n; i++) { const int y = (int)(i / Wd % H), x = (int)(i % Wd); if ((y / 4 + x / 4) % 3 == 0 || rand() % 4 == 0) a[9][i] = 0; }   /* empty blocks, holes */
    for (int k = 0; k < 10; k++) { char q[600]; snprintf(q, sizeof q, "%s/%s.raw", dir, nm[k]); FILE *f = fopen(q, "wb"); fwrite(a[k], 1, n, f); fclose(f); }
    snprintf(p, sizeof p, "%s/fields.json", dir); FILE *f = fopen(p, "w");
    fprintf(f, "{\"format\": \"raw\", \"origin\": [6, 4, 2], \"shape\": [%d, %d, %d], \"h\": %d, \"um\": 4.798, \"core_line\": \"%s/core.json\", \"core_side\": 1, \"axis\": [[0, 0]]}\n", D, H, Wd, h, dir);
    fclose(f);
    /* the reference: serial cl_mask_block over the whole box (mode w: pc, ps, w; mode 0: pc alone, every block) */
    const long long org[3] = {6, 4, 2};
    uint8_t *rpc = malloc(n), *rps = malloc(n), *rw = malloc(n), *rpc0 = malloc(n), *rps0 = malloc(n);
    memcpy(rpc, a[0], n); memcpy(rps, a[1], n); memcpy(rw, a[9], n); memcpy(rpc0, a[0], n); memcpy(rps0, a[1], n);
    const size_t nb = cl_mask_block(&C, 1, org, h, 4.798, 0, D, H, Wd, rpc, rps, rw);
    cl_mask_block(&C, 1, org, h, 4.798, 0, D, H, Wd, rpc0, nullptr, nullptr); cl_mask_block(&C, 1, org, h, 4.798, 0, D, H, Wd, rps0, nullptr, nullptr);
    CHECK(nb > n / 20, "the core must cut the box (%zu cells blanked)", nb);
    fit_set_threads(3);
    for (int pass = 0; pass < 2; pass++) {   /* pass 0 fills the cache, pass 1 reads it (other row ranges) */
        const int r[3][2] = {{0, 17}, {9, 31}, {25, D}};
        for (int q = 0; q < 3; q++) {
            fit_fields_buf b; CHECK(fit_fields_load(dir, r[q][0] + pass, r[q][1], 1, &b) == 0, "load");
            const size_t o = (size_t)(r[q][0] + pass) * H * Wd, m = (size_t)(r[q][1] - r[q][0] - pass) * H * Wd;
            CHECK(!memcmp(b.a[0], rpc + o, m) && !memcmp(b.a[1], rps + o, m) && !memcmp(b.a[9], rw + o, m) && !memcmp(b.a[2], a[2] + o, m), "pass %d rows %d..%d differ from the serial masks", pass, r[q][0] + pass, r[q][1]);
            fit_fields_free(&b);
        }
        int sh[3]; uint8_t *pc = fit_field_array(dir, "pc", 1, sh), *ps = fit_field_array(dir, "ps", 1, sh), *w = fit_field_array(dir, "w", 1, sh);
        int bad = 0; for (size_t i = 0; pc && ps && w && i < n; i++) bad += pc[i] != rpc0[i] || ps[i] != rps0[i] || w[i] != rw[i];
        CHECK(pc && ps && w && bad == 0, "pass %d: fit_field_array differs from the serial masks at %d cells", pass, bad);
        free(pc); free(ps); free(w);
    }
    /* the merged pooled load: pool-1 loads of 8-row slabs (and a last partial one) filling the pooled box */
    for (int pool = 2; pool <= 8; pool *= 4) {
        fit_fields_buf ref, mg; CHECK(fit_fields_load(dir, 0, 0, pool, &ref) == 0 && fit_fields_load_ex(dir, 0, 0, pool, &mg, 1, nullptr, 0, nullptr) == 0, "pooled loads");
        for (int z0 = 0; z0 < D; z0 += 8) { fit_fields_buf cb; CHECK(fit_fields_load_ex(dir, z0, z0 + 8 < D ? z0 + 8 : D, 1, &cb, 0, &mg, pool, nullptr) == 0, "slab load"); fit_fields_free(&cb); }
        const size_t np = (size_t)ref.shape[0] * ref.shape[1] * ref.shape[2]; int same = ref.h == mg.h && !memcmp(ref.shape, mg.shape, sizeof ref.shape);
        for (int k = 0; k < 10 && same; k++) same = !memcmp(ref.a[k], mg.a[k], np);
        CHECK(same, "pool %d: the merged pooled box differs from the pooled load", pool);
        fit_fields_free(&ref); fit_fields_free(&mg);
    }
    /* consecutive slabs: the overlap rows of the previous load moved, the rest read (and a non-overlapping one read whole) */
    { const int r[4][2] = {{0, 14}, {6, 30}, {22, D}, {3, 9}}; fit_fields_buf prev; memset(&prev, 0, sizeof prev);
      for (int q = 0; q < 4; q++) {
          fit_fields_buf b, ref; CHECK(fit_fields_load_ex(dir, r[q][0], r[q][1], 1, &b, 0, nullptr, 0, &prev) == 0 && fit_fields_load(dir, r[q][0], r[q][1], 1, &ref) == 0, "reload");
          const size_t m = (size_t)(r[q][1] - r[q][0]) * H * Wd; int same = b.shape[0] == ref.shape[0] && b.origin[0] == ref.origin[0];
          for (int k = 0; k < 10 && same; k++) same = !memcmp(b.a[k], ref.a[k], m);
          for (int z = 0; z <= b.shape[0] * h && same; z++) same = b.cy[z] == ref.cy[z] && b.cx[z] == ref.cx[z];
          CHECK(same, "rows %d..%d loaded after a previous slab differ from a fresh load", r[q][0], r[q][1]);
          fit_fields_free(&ref); prev = b;
      }
      fit_fields_free(&prev); }
    fit_set_threads(0);
    for (int k = 0; k < 10; k++) { snprintf(p, sizeof p, "%s/%s.raw", dir, nm[k]); unlink(p); free(a[k]); }
    snprintf(p, sizeof p, "%s/fields.json", dir); unlink(p); snprintf(p, sizeof p, "%s/core.json", dir); unlink(p); rmdir(dir);
    free(C.s); free(rpc); free(rps); free(rw); free(rpc0); free(rps0);
}

int main(void) {
    test_sdist();
    test_frame();
    test_sdist_fast();
    test_mask_cache();
    synth_detect(0.0, 0);
    synth_detect(0.6, 0);
    synth_detect(-1.1, 1);
    test_mask();
    if (fails) { fprintf(stderr, "test_coreline: %d failure(s)\n", fails); return 1; }
    fprintf(stderr, "test_coreline: ok\n");
    return 0;
}
