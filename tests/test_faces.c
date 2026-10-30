/* ufsm faces: papyrus bands of synthetic CT profiles, and recto (inner) / verso (outer) faces of a synthetic two-sheet cylinder found
   from a wrap sitting in the gap inside the first sheet, whichever way the grid normal points. */
#include "faces.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)

static double S(double u) { return 0.5 * (1 + tanh(u / 0.7)); }
/* sheets [a0, a1] and [b0, b1] (radial, voxels), CT 150 on papyrus over 20 void */
typedef struct { double cy, cx, a0, a1, b0, b1; } cyl;
static double cyl_ct(const cyl *C, double r) { return 20 + 130 * (S(r - C->a0) - S(r - C->a1)) + 130 * (S(r - C->b0) - S(r - C->b1)); }
static double cyl_sample(void *ctx, double z, double y, double x) { const cyl *C = ctx; (void)z; return cyl_ct(C, hypot(y - C->cy, x - C->cx)); }

int main(void) {
    /* 1. one profile: two sheets 5 and 4 voxels thick with a 3 voxel gap; edges at the half levels */
    {
        cyl C = {0, 0, 10, 15, 18, 22}; float f[201]; const double h = 0.1;
        for (int k = 0; k < 201; k++) f[k] = (float)cyl_ct(&C, k * h + 5);   /* t = r - 15 + ... : t0 = 5 */
        fc_band b[4]; const int n = faces_bands(f, 201, 5, h, 8, 0.1, b, 4);
        CHECK(n == 2, "two bands, got %d", n);
        if (n == 2) {
            CHECK(fabs(b[0].near - 10) < 0.15 && fabs(b[0].far - 15) < 0.15, "sheet 1 at 10..15, got %.3f..%.3f", b[0].near, b[0].far);
            CHECK(fabs(b[1].near - 18) < 0.15 && fabs(b[1].far - 22) < 0.15, "sheet 2 at 18..22, got %.3f..%.3f", b[1].near, b[1].far);
            CHECK(b[0].prom > 100, "contrast %.1f", b[0].prom);
        }
        /* touching sheets (no gap: a shallow valley): split at the valley */
        for (int k = 0; k < 201; k++) { const double r = k * h + 5; f[k] = (float)(20 + 130 * (S(r - 10) - S(r - 22)) - 30 * exp(-(r - 15) * (r - 15) / 0.5)); }
        const int m = faces_bands(f, 201, 5, h, 8, 0.1, b, 4);
        CHECK(m == 2, "touching: two bands, got %d", m);
        if (m == 2) CHECK(fabs(b[0].far - 15) < 0.3 && fabs(b[1].near - 15) < 0.3 && fabs(b[0].near - 10) < 0.2 && fabs(b[1].far - 22) < 0.2,
                          "touching: 10..15 | 15..22, got %.2f..%.2f | %.2f..%.2f", b[0].near, b[0].far, b[1].near, b[1].far);
        /* a shallow dip inside one sheet (two fibre layers) keeps one band */
        for (int k = 0; k < 201; k++) { const double r = k * h + 5; f[k] = (float)(20 + 130 * (S(r - 10) - S(r - 16)) - 6 * exp(-(r - 13) * (r - 13) / 0.5)); }
        const int q = faces_bands(f, 201, 5, h, 8, 0.1, b, 4);
        CHECK(q == 1 && fabs(b[0].near - 10) < 0.2 && fabs(b[0].far - 16) < 0.2, "fibre dip: one band 10..16, got %d %.2f..%.2f", q, q ? b[0].near : 0, q ? b[0].far : 0);
    }
    /* 2. a wrap (arc of a cylinder, rows = z) in the gap 2 voxels inside sheet 1 (r 100..105), sheet 2 at 109..113 */
    for (int flip = 0; flip < 2; flip++) {
        cyl C = {300, 300, 100, 105, 109, 113};
        const int w = 160, h = 24; const size_t n = (size_t)w * h;
        float *X = malloc(n * 4), *Y = malloc(n * 4), *Z = malloc(n * 4), *RX = malloc(n * 4), *RY = malloc(n * 4), *RZ = malloc(n * 4), *VX = malloc(n * 4), *VY = malloc(n * 4), *VZ = malloc(n * 4), *th = malloc(n * 4);
        uint8_t *cf = malloc(n);
        for (int r = 0; r < h; r++) for (int c = 0; c < w; c++) {
            const size_t i = (size_t)r * w + c; const double a = (flip ? w - 1 - c : c) * 0.02, rad = 98 + 0.3 * sin(0.1 * c + 0.2 * r);
            X[i] = (float)(C.cx + rad * cos(a)); Y[i] = (float)(C.cy + rad * sin(a)); Z[i] = (float)(50 + 2 * r);
        }
        Z[5 * w + 7] = -1; X[5 * w + 7] = Y[5 * w + 7] = -1;   /* a hole */
        const int sign = faces_orient(X, Y, Z, w, h, nullptr, 0);
        double N[3]; faces_normal(X, Y, Z, w, h, (size_t)10 * w + 80, N);
        const double ry = Y[10 * w + 80] - C.cy, rx = X[10 * w + 80] - C.cx;
        CHECK(sign * (N[1] * ry + N[2] * rx) > 0, "flip %d: orientation %d points inward", flip, sign);
        faces_opt o = {.range = 12, .tmin = 2, .tmax = 12, .min_prom = 8, .rel = 0.1, .tol = 2.5, .inset = 0, .smooth = 3, .nthreads = 4};
        faces_vol V = {&C, cyl_sample, nullptr, nullptr};
        faces_stats st = {0}; st.ct_thr = 90;
        CHECK(!faces_wrap(X, Y, Z, w, h, 1.0, 9.6, sign, &o, &V, RX, RY, RZ, VX, VY, VZ, cf, th, &st), "faces_wrap");
        double er = 0, ev = 0, et = 0; int k = 0;
        for (int r = 3; r < h - 3; r++) for (int c = 3; c < w - 3; c++) {
            const size_t i = (size_t)r * w + c; if (Z[i] < 0) continue;
            er = fmax(er, fabs(hypot(RY[i] - C.cy, RX[i] - C.cx) - 100)); ev = fmax(ev, fabs(hypot(VY[i] - C.cy, VX[i] - C.cx) - 105));
            et = fmax(et, fabs(th[i] - 5 * 9.6)); k++;
        }
        printf("flip %d: %d vertices, recto err %.3f verso err %.3f thick err %.2f um, picked %zu/%zu, sign %d\n", flip, k, er, ev, et, st.npick, st.nv, sign);
        CHECK(k > 0 && er < 0.3 && ev < 0.3, "flip %d: recto |r - 100| <= %.3f, verso |r - 105| <= %.3f", flip, er, ev);
        CHECK(et < 4, "flip %d: thickness error %.2f um", flip, et);
        CHECK(RZ[5 * w + 7] < 0 && VZ[5 * w + 7] < 0, "hole stays invalid");
        CHECK(st.npick > 0.9 * st.nv, "flip %d: picked %zu of %zu", flip, st.npick, st.nv);
        free(X); free(Y); free(Z); free(RX); free(RY); free(RZ); free(VX); free(VY); free(VZ); free(th); free(cf);
    }
    if (fails) { fprintf(stderr, "test_faces: %d failures\n", fails); return 1; }
    printf("test_faces: ok\n");
    return 0;
}
