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
static int g_cpu = 0; static float g_robust = 0, g_bend = 0, g_pgrad = 0;   /* the CPU backend, the Cauchy phase, curvature-adaptive smoothness */
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
    fit_fields f = {D, H, W, H8, pc, ps, {t[0], t[1], t[2], t[3], t[4], t[5]}, sp, w, cy, cx, {0}, 0, nullptr};
    fit_cfg c = fit_cfg_default(); c.verbose = 1; c.uw_split = g_split; c.uw_tile = g_tile; c.cpu = g_cpu; c.phase_robust = g_robust; c.bend = g_bend; c.w_pgrad = g_pgrad;
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
    printf("dir %+d%s%s%s%s: gauge %d, winding ids right %.4f, |error| < 0.08 turn %.4f (%zu voxels), loss %.5f\n", dir, g_split > 0 ? " two-level" : "", g_cpu ? " cpu" : "",
           g_robust > 0 ? " robust" : "", g_bend > 0 ? " bend" : "", k0 - 200, acc, prec, nd, m->loss[0]);
    if (g_pgrad > 0) printf("  (with phase-gradient matching)\n");
    if (g_bias == 0) {
        CHECK(acc >= 0.995, "dir %+d: winding ids right %.4f < 0.995", dir, acc);
        CHECK(prec >= 0.99, "dir %+d: |error| < 0.08 at %.4f < 0.99", dir, prec);
    }
    if (!g_cpu && g_split == 0 && dir > 0) {   /* the CPU and GPU data terms agree (every voxel, the fitted model, the robust phase, the
                                                 winding mod 4 term and the recto anchors on planes made from the true W) */
        uint8_t *wd = malloc(4 * n), *rc = malloc(n); const int w4d = 24;
        for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
            const size_t i = ((size_t)z * H + y) * W + x; const double pz = (z + 0.5) * H8, py = (y + 0.5) * H8, px = (x + 0.5) * H8, w0 = W_true(pz, py, px);
            for (int d = 0; d < 3; d++) {
                double b[3] = {pz, py, px}; b[d] += w4d;
                double dth = atan2(b[1] - ax_y(b[0]), b[2] - ax_x(b[0])) - atan2(py - ax_y(pz), px - ax_x(pz)); dth -= 2 * M_PI * floor((dth + M_PI) / (2 * M_PI));
                double q = fmod(U_true(b[0], b[1], b[2]) - U_true(pz, py, px) - dir * dth / (2 * M_PI) + 0.1 * sin(0.3 * i), 4.0); if (q < 0) q += 4;
                wd[d * n + i] = (uint8_t)(lround(q * 64) & 255);
            }
            wd[3 * n + i] = (uint8_t)(128 + (i * 37) % 128);
            const double fr = w0 - floor(w0 + 0.5); rc[i] = (uint8_t)lround(255 * exp(-fr * fr / (2 * 0.05 * 0.05)));
        }
        fit_dev fd = {D, H, W, (float)H8, pc, ps, {t[0], t[1], t[2], t[3], t[4], t[5]}, sp, w, cy, cx, na, m->dir, m->nk, m->knot_dz, m->pitch, m->offset,
                      m->L, m->lz, m->ly, m->lx, m->delta, 1e-5f, 1e-5f, 1e-5f, 1e-5f, 3.f, 1, 7u, 0.f, 0.f, 0.05f, 1.5f, 1e-5f, 0.25f, nullptr,
                      wd, 1e-5f, (float)w4d, rc, 1e-5f};
        const size_t nl = (size_t)m->lz * m->ly * m->lx; double ph[FIT_NPARTS] = {0}, pd[FIT_NPARTS] = {0}; float *gh = calloc(nl, 4), *gg = malloc(nl * 4);
        fit_loss2_host(&fd, ph, gh);
        uint8_t *dv = nn_malloc(15 * n); const uint8_t *hv[15] = {pc, ps, t[0], t[1], t[2], t[3], t[4], t[5], sp, w, wd, wd + n, wd + 2 * n, wd + 3 * n, rc};
        for (int k = 0; k < 15; k++) nn_h2d(dv + k * n, hv[k], n);
        float *dcy = nn_malloc(na * 4), *dcx = nn_malloc(na * 4), *dpi = nn_malloc(m->nk * 4), *dof = nn_malloc(m->nk * 4), *ddl = nn_malloc(nl * 4), *dg = nn_malloc(nl * 4);
        double *dp = nn_malloc(FIT_NPARTS * sizeof(double));
        nn_h2d(dcy, cy, na * 4); nn_h2d(dcx, cx, na * 4); nn_h2d(dpi, m->pitch, m->nk * 4); nn_h2d(dof, m->offset, m->nk * 4); nn_h2d(ddl, m->delta, nl * 4);
        nn_zero(dg, nl * 4); nn_zero(dp, FIT_NPARTS * sizeof(double));
        fit_dev fg = fd; fg.pc = dv; fg.ps = dv + n; for (int k = 0; k < 6; k++) fg.t[k] = dv + (2 + k) * n; fg.sp = dv + 8 * n; fg.w = dv + 9 * n; fg.wd = dv + 10 * n; fg.recto = dv + 14 * n;
        fg.cy = dcy; fg.cx = dcx; fg.pitch = dpi; fg.offset = dof; fg.delta = ddl;
        nn_fit_loss2(&fg, dp, dg); nn_sync(); nn_d2h(pd, dp, sizeof pd); nn_d2h(gg, dg, nl * 4);
        double gmax = 0, dmax = 0; for (size_t k = 0; k < nl; k++) { gmax = fmax(gmax, fabs(gh[k])); dmax = fmax(dmax, fabs(gh[k] - gg[k])); }
        printf("backends: loss parts cpu %.6g %.6g %.6g %.6g %.6g wind4 %.6g recto %.6g gpu %.6g %.6g %.6g %.6g %.6g wind4 %.6g recto %.6g; gradient max |cpu - gpu| %.3g of max %.3g\n",
               ph[1], ph[2], ph[3], ph[4], ph[5], ph[FIT_P_WIND4], ph[FIT_P_RECTO], pd[1], pd[2], pd[3], pd[4], pd[5], pd[FIT_P_WIND4], pd[FIT_P_RECTO], dmax, gmax);
        for (int k = 0; k < FIT_NPARTS; k++) if (k != FIT_P_SMOOTH) CHECK(fabs(ph[k] - pd[k]) <= 1e-3 * fabs(pd[k]) + 1e-9, "loss part %d: cpu %.8g gpu %.8g", k, ph[k], pd[k]);
        CHECK(pd[FIT_P_WIND4] > 0 && pd[FIT_P_RECTO] > 0, "the winding mod 4 / recto terms are inactive (%g %g)", pd[FIT_P_WIND4], pd[FIT_P_RECTO]);
        free(wd); free(rc);
        CHECK(dmax <= 1e-3 * gmax, "gradient: max |cpu - gpu| %.3g > 1e-3 x %.3g", dmax, gmax);
        nn_free(dv); nn_free(dcy); nn_free(dcx); nn_free(dpi); nn_free(dof); nn_free(ddl); nn_free(dg); nn_free(dp); free(gh); free(gg);
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
    g_split = 0; g_tile = 256;
    g_cpu = 1; run_case(+1); run_case(-1);       /* the CPU backend (src/fit_terms.h: the same terms) */
    g_robust = 0.1f; g_bend = 1.f; run_case(+1); /* the Cauchy phase residual and curvature-adaptive smoothness keep a clean fit */
    g_cpu = 0; run_case(-1);                     /* ... on the GPU too */
    g_robust = 0; g_bend = 0; g_pgrad = 1.f; run_case(+1); g_cpu = 1; run_case(-1);   /* phase-gradient matching, both backends */
    if (fails) { fprintf(stderr, "test_fit: %d failure(s)\n", fails); return 1; }
    printf("test_fit: OK\n");
    return 0;
}
