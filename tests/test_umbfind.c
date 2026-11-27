/* ufsm umbilicus (src/umbfind.c, villa vc_gen_umbilicus port) on synthetic scrolls with a known tilted + curved axis. CPU only.
   - CT path: per slice a rendered Archimedean spiral (pitch 24 level-0 voxels, 12 turns, level-1 pixels, noise, CT 0 outside the
     roll) around c(z) = c0 + tilt z + bow sin(pi z / Z); umb_ct_points -> umb_estimate -> umb_zfit recovers c(z): median and max error
     median < 2, max < 4 level-0 voxels (measured 0.84 / 1.66; pitch 24, so < 1/10 turn spacing).
   - a crushed roll (x squeezed 0.6), uncrush 0 and 1.5: median < 3, max < 6 (measured 0.98 / 1.28 and 0.95 / 1.54).
   - zfit: a linear axis comes back exactly (local-linear); one slice 400 voxels off is rejected (bisquare) at its z.
   - determinism: the same seed gives the same estimate. */
#include "umbfind.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

enum { NZ = 24, S = 2, HP = 400 };   /* slices, level-0 voxels per pixel, pixels per side */
static const double ZSTEP = 200, PITCH = 24, R0 = 20, RMAX = 300;
static void axis_true(double z, double *cy, double *cx) {
    const double t = z / (NZ * ZSTEP);
    *cy = 380 + 60 * t + 25 * sin(M_PI * t);   /* tilt 60 voxels over the stack, bow 25 */
    *cx = 420 - 40 * t + 30 * sin(M_PI * t);
}
static uint64_t rs = 12345;
static double nrand(void) {   /* ~N(0, 1) from 12 uniforms */
    double s = 0; for (int i = 0; i < 12; i++) { rs = rs * 6364136223846793005ull + 1442695040888963407ull; s += (double)(rs >> 11) / 9007199254740992.0; }
    return s - 6;
}
/* level-1 image of a spiral r = R0 + PITCH (theta + 2 pi k) / 2 pi around (cy, cx); sx squeezes x (crush) */
static void render(float *img, double cy, double cx, double sx) {
    for (int i = 0; i < HP; i++) for (int j = 0; j < HP; j++) {
        const double y = i * S + 0.5 * (S - 1), x = j * S + 0.5 * (S - 1), dy = y - cy, dx = (x - cx) / sx, r = hypot(dy, dx);
        float v = 0;
        if (r <= RMAX) {
            const double th = atan2(dy, dx), ph = (r - R0) / PITCH - th / (2 * M_PI), c = cos(2 * M_PI * ph);
            v = (float)(60 + 150 * pow(fmax(0, c), 4) + 8 * nrand());
            if (v < 1) v = 1;
        }
        img[(size_t)i * HP + j] = v;
    }
}

static void run_stack(double sx, double uncrush, double *med, double *mx, double *medraw) {
    umb_cfg cfg; umb_cfg_default(&cfg, 7.91); cfg.uncrush = uncrush;
    float *img = malloc(sizeof(float) * HP * HP);
    double z[NZ], ey[NZ], ex[NZ], fy[NZ], fx[NZ], err[NZ], errr[NZ]; int n = 0;
    for (int k = 0; k < NZ; k++) {
        const double zz = (k + 0.5) * ZSTEP; double cy, cx; axis_true(zz, &cy, &cx);
        render(img, cy, cx, sx);
        umb_pt *p; double cen[2]; long long nc;
        const int np = umb_ct_points(img, HP, HP, S, 0, 1, 3, 2, 0.2, &p, cen, &nc);
        CHECK(np > 1000, "slice %d: only %d points", k, np);
        double est[2], sc; int in;
        CHECK(umb_estimate(p, np, &cfg, 7 + k, est, &sc, &in) == 0 && in, "slice %d: no estimate inside", k);
        z[n] = zz; ey[n] = est[0]; ex[n] = est[1]; errr[n] = hypot(est[0] - cy, est[1] - cx); n++;
        free(p);
    }
    umb_zfit(z, ey, ex, n, z, n, 1000, 3, fy, fx, nullptr);
    *mx = 0;
    for (int k = 0; k < n; k++) { double cy, cx; axis_true(z[k], &cy, &cx); err[k] = hypot(fy[k] - cy, fx[k] - cx); *mx = fmax(*mx, err[k]); }
    for (int a = 0; a < n; a++) for (int b = a + 1; b < n; b++) { if (err[b] < err[a]) { double t = err[a]; err[a] = err[b]; err[b] = t; } if (errr[b] < errr[a]) { double t = errr[a]; errr[a] = errr[b]; errr[b] = t; } }
    *med = 0.5 * (err[n / 2 - 1] + err[n / 2]); *medraw = 0.5 * (errr[n / 2 - 1] + errr[n / 2]);
    free(img);
}

int main(void) {
    double med, mx, medraw;
    run_stack(1.0, 0, &med, &mx, &medraw);
    printf("round roll: per-slice median %.2f, fitted median %.2f max %.2f level-0 voxels (pitch %.0f)\n", medraw, med, mx, PITCH);
    CHECK(med < 2 && mx < 4, "round roll: fitted median %.2f (< 2) max %.2f (< 4)", med, mx);
    double medc, mxc, medrc;
    run_stack(0.6, 0, &medc, &mxc, &medrc);
    double medu, mxu, medru;
    run_stack(0.6, 1.5, &medu, &mxu, &medru);
    printf("crushed x0.6: uncrush 0 median %.2f max %.2f; uncrush 1.5 median %.2f max %.2f\n", medc, mxc, medu, mxu);
    CHECK(medc < 3 && medu < 3 && mxc < 6 && mxu < 6, "crushed roll: medians %.2f / %.2f (< 3), max %.2f / %.2f (< 6)", medc, medu, mxc, mxu);

    /* zfit: exact on a line; an outlier rejected */
    double z[30], y[30], x[30], fy[30], fx[30];
    for (int i = 0; i < 30; i++) { z[i] = 100.0 * i; y[i] = 500 + 0.1 * z[i]; x[i] = 300 - 0.05 * z[i]; }
    umb_zfit(z, y, x, 30, z, 30, 450, 3, fy, fx, nullptr);
    double e = 0; for (int i = 0; i < 30; i++) e = fmax(e, hypot(fy[i] - y[i], fx[i] - x[i]));
    CHECK(e < 1e-6, "zfit on a line: max error %.3g", e);
    y[15] += 400;
    umb_zfit(z, y, x, 30, z, 30, 450, 3, fy, fx, nullptr);
    CHECK(fabs(fy[15] - (500 + 0.1 * z[15])) < 5, "zfit outlier at z %.0f: fitted y %.2f, want %.2f", z[15], fy[15], 500 + 0.1 * z[15]);

    /* determinism */
    float *img = malloc(sizeof(float) * HP * HP); render(img, 400, 400, 1);
    umb_pt *p; double cen[2]; long long nc; const int np = umb_ct_points(img, HP, HP, S, 0, 1, 3, 2, 0.2, &p, cen, &nc);
    umb_cfg cfg; umb_cfg_default(&cfg, 7.91);
    double a[2], b[2], sc; int in;
    umb_estimate(p, np, &cfg, 99, a, &sc, &in); umb_estimate(p, np, &cfg, 99, b, &sc, &in);
    CHECK(a[0] == b[0] && a[1] == b[1], "same seed, different estimates (%.3f %.3f) vs (%.3f %.3f)", a[0], a[1], b[0], b[1]);
    CHECK(hypot(cen[0] - 400, cen[1] - 400) < 3, "CT>0 centroid of a centred roll %.2f %.2f", cen[0], cen[1]);
    free(p); free(img);
    if (fails) { fprintf(stderr, "test_umbfind: %d FAILED\n", fails); return 1; }
    printf("test_umbfind: OK\n");
    return 0;
}
