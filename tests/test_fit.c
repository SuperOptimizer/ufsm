/* Winding fitter (src/fit.c) on synthetic fit fields: an Archimedean spiral around a tilted axis with z-varying pitch, an
   angular wobble, a packed sector and a hole, for dir +1 and -1. The fields are what predict --fit-fields will write
   (phase vector, n n^T, spacing, material weight) computed from the true winding coordinate W. Checks: the fitted direction,
   and W_fit - W_true = one integer gauge everywhere (winding ids >= 99.5% right, |error| < 0.08 turn at 99% of voxels). */
#include "fit.h"
#include "fit_terms.h"
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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
    fit_fields f = {D, H, W, H8, pc, ps, {t[0], t[1], t[2], t[3], t[4], t[5]}, sp, w, cy, cx, {0}, 0, nullptr, nullptr, nullptr};
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
                      wd, 1e-5f, (float)w4d, rc, 1e-5f, nullptr, 1e-5f, 1.5f, 0.3f, 0.05f, nullptr, 1e-5f, 0.2f};
        uint8_t *ctm = malloc(n), *csp = malloc(n);   /* CT parity and spacing from the true W (the spacing off by a z-dependent factor: the
                                                         sp / cs ramp sees agreeing, partly and fully disagreeing cells) */
        for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
            const size_t i = ((size_t)z * H + y) * W + x; const double w0 = W_true((z + 0.5) * H8, (y + 0.5) * H8, (x + 0.5) * H8);
            ctm[i] = (uint8_t)lround(128 + 127 * 0.8 * cos(2 * M_PI * (w0 - 0.3)) * (x % 7 ? 1 : -0.5));
            const double cv = sp[i] * (1 + 0.8 * z / D); csp[i] = (uint8_t)(cv > 255 ? 255 : lround(cv));
        }
        fd.ctm = ctm; fd.cs = csp;
        const size_t nl = (size_t)m->lz * m->ly * m->lx; double ph[FIT_NPARTS] = {0}, pd[FIT_NPARTS] = {0}; float *gh = calloc(nl, 4), *gg = malloc(nl * 4);
        fit_loss2_host(&fd, ph, gh);
        uint8_t *dv = nn_malloc(17 * n); const uint8_t *hv[17] = {pc, ps, t[0], t[1], t[2], t[3], t[4], t[5], sp, w, wd, wd + n, wd + 2 * n, wd + 3 * n, rc, ctm, csp};
        for (int k = 0; k < 17; k++) nn_h2d(dv + k * n, hv[k], n);
        float *dcy = nn_malloc(na * 4), *dcx = nn_malloc(na * 4), *dpi = nn_malloc(m->nk * 4), *dof = nn_malloc(m->nk * 4), *ddl = nn_malloc(nl * 4), *dg = nn_malloc(nl * 4);
        double *dp = nn_malloc(FIT_NPARTS * sizeof(double));
        nn_h2d(dcy, cy, na * 4); nn_h2d(dcx, cx, na * 4); nn_h2d(dpi, m->pitch, m->nk * 4); nn_h2d(dof, m->offset, m->nk * 4); nn_h2d(ddl, m->delta, nl * 4);
        nn_zero(dg, nl * 4); nn_zero(dp, FIT_NPARTS * sizeof(double));
        fit_dev fg = fd; fg.pc = dv; fg.ps = dv + n; for (int k = 0; k < 6; k++) fg.t[k] = dv + (2 + k) * n; fg.sp = dv + 8 * n; fg.w = dv + 9 * n; fg.wd = dv + 10 * n; fg.recto = dv + 14 * n; fg.ctm = dv + 15 * n; fg.cs = dv + 16 * n;
        fg.cy = dcy; fg.cx = dcx; fg.pitch = dpi; fg.offset = dof; fg.delta = ddl;
        nn_fit_loss2(&fg, dp, dg); nn_sync(); nn_d2h(pd, dp, sizeof pd); nn_d2h(gg, dg, nl * 4);
        double gmax = 0, dmax = 0; for (size_t k = 0; k < nl; k++) { gmax = fmax(gmax, fabs(gh[k])); dmax = fmax(dmax, fabs(gh[k] - gg[k])); }
        printf("backends: loss parts cpu %.6g %.6g %.6g %.6g %.6g wind4 %.6g recto %.6g ct %.6g cspace %.6g gpu %.6g %.6g %.6g %.6g %.6g wind4 %.6g recto %.6g ct %.6g cspace %.6g; "
               "gradient max |cpu - gpu| %.3g of max %.3g\n", ph[1], ph[2], ph[3], ph[4], ph[5], ph[FIT_P_WIND4], ph[FIT_P_RECTO], ph[FIT_P_CT], ph[FIT_P_CSPACE],
               pd[1], pd[2], pd[3], pd[4], pd[5], pd[FIT_P_WIND4], pd[FIT_P_RECTO], pd[FIT_P_CT], pd[FIT_P_CSPACE], dmax, gmax);
        for (int k = 0; k < FIT_NPARTS; k++) if (k != FIT_P_SMOOTH) CHECK(fabs(ph[k] - pd[k]) <= 1e-3 * fabs(pd[k]) + 1e-9, "loss part %d: cpu %.8g gpu %.8g", k, ph[k], pd[k]);
        CHECK(pd[FIT_P_WIND4] > 0 && pd[FIT_P_RECTO] > 0, "the winding mod 4 / recto terms are inactive (%g %g)", pd[FIT_P_WIND4], pd[FIT_P_RECTO]);
        CHECK(pd[FIT_P_CT] != 0 && pd[FIT_P_CSPACE] > 0, "the CT parity / spacing terms are inactive (%g %g)", pd[FIT_P_CT], pd[FIT_P_CSPACE]);
        CHECK(dmax <= 1e-3 * gmax, "gradient: max |cpu - gpu| %.3g > 1e-3 x %.3g", dmax, gmax);
        /* minibatch steps (stride > 1: the GPU lists the step's voxels, then evaluates the list): the same voxels (hash(i ^ salt) %
           stride == 0, material) and terms x stride as a host loop over fit_vox2 */
        for (int st = 3; st <= 37; st += 34) {
            fit_dev fs = fd, gs = fg; fs.stride = gs.stride = st; fs.salt = gs.salt = 2654435761u * (unsigned)st + 12345u;
            double ps_h[FIT_NPARTS] = {0}, ps_d[FIT_NPARTS] = {0}; float *gr = calloc(nl, 4); size_t nsel = 0;
            for (size_t i = 0; i < n; i++) {
                if (fit_hash((unsigned)i ^ fs.salt) % (unsigned)st || !w[i]) continue;
                double lp[FIT_NPARTS] = {0}; size_t ic[FIT_VOX2_MAXC]; float gv[FIT_VOX2_MAXC];
                const int nc = fit_vox2(&fs, i, w[i] * (1.f / 255.f), (float)st, lp, ic, gv); nsel++;
                for (int q = 0; q < FIT_NPARTS; q++) ps_h[q] += lp[q];
                for (int q = 0; q < nc; q++) gr[ic[q]] += gv[q];
            }
            nn_zero(dg, nl * 4); nn_zero(dp, FIT_NPARTS * sizeof(double));
            nn_fit_loss2(&gs, dp, dg); nn_sync(); nn_d2h(ps_d, dp, sizeof ps_d); nn_d2h(gg, dg, nl * 4);
            double gm = 0, dm = 0; for (size_t k = 0; k < nl; k++) { gm = fmax(gm, fabs(gr[k])); dm = fmax(dm, fabs(gr[k] - gg[k])); }
            printf("minibatch stride %d: %zu voxels, loss host %.8g gpu %.8g, gradient max |host - gpu| %.3g of max %.3g\n", st, nsel, ps_h[0], ps_d[0], dm, gm);
            for (int k = 0; k < FIT_NPARTS; k++) if (k != FIT_P_SMOOTH) CHECK(fabs(ps_h[k] - ps_d[k]) <= 1e-3 * fabs(ps_h[k]) + 1e-9, "stride %d loss part %d: host %.12g gpu %.12g", st, k, ps_h[k], ps_d[k]);
            CHECK(nsel > n / st / 4 && dm <= 1e-3 * gm, "stride %d: gradient max |host - gpu| %.3g > 1e-3 x %.3g (%zu voxels)", st, dm, gm, nsel);
            free(gr);
        }
        free(wd); free(rc); free(ctm); free(csp);
        nn_free(dv); nn_free(dcy); nn_free(dcx); nn_free(dpi); nn_free(dof); nn_free(ddl); nn_free(dg); nn_free(dp); free(gh); free(gg);
    }
    fit_free(m);
    free(pc); free(ps); free(sp); free(w); free(cy); free(cx); for (int k = 0; k < 6; k++) free(t[k]);
    return bad;
}

/* CT in the fitter: the packed sector's network phase is weak (confidence 0.2, below the unwrapping's 0.3) and wrong (it and the network
   spacing under-count the sector's extra turns by g_ctbias), as on a crushed scroll; the ct plane is the true sheets' mean CT (papyrus at
   frac W_true = 0.2, noise), cs the true spacing. Without CT the fit follows the network into the slip; with --w-ct and --w-cspacing it
   must recover the ids and the sheet placement. Returns the fraction of ids right in the sector (and *all, *prec overall). */
static double g_ctbias = 0.5; static int g_ctflood = 0;   /* fit_cfg ct_flood (the CT phase in the voxel unwrapping) */
static double ct_sector(double z, double y, double x) {   /* the packed sector's weight */
    double dth = atan2(y - ax_y(z), x - ax_x(z)) - 1.0; dth -= 2 * M_PI * floor((dth + M_PI) / (2 * M_PI));
    return exp(-dth * dth / (2 * 0.3 * 0.3));
}
static double W_ctnet(double z, double y, double x) {   /* the weak network's W: the sector's extra turns scaled by 1 - g_ctbias */
    const double r = hypot(y - ax_y(z), x - ax_x(z));
    const double sw = ct_sector(z, y, x), bw = sw > 0.3 ? (sw - 0.3) / 0.7 : 0;   /* wrong only inside the weak sector */
    return W_true(z, y, x) - g_ctbias * g_pack / 60 * r * bw;
}
static double run_ct_case(int cpu, float w_ct, float w_cs, double *all, double *prec_out) {
    g_dir = 1;
    const int D = 48, H = 96, W = 96; const size_t n = (size_t)D * H * W;
    uint8_t *pc = malloc(n), *ps = malloc(n), *t[6], *sp = malloc(n), *w = malloc(n), *ct = malloc(n), *cs = malloc(n), *sec = malloc(n);
    for (int k = 0; k < 6; k++) t[k] = malloc(n);
    const int na = (int)(D * H8) + 1;
    float *cy = malloc(na * sizeof(float)), *cx = malloc(na * sizeof(float));
    for (int z = 0; z < na; z++) { cy[z] = (float)ax_y(z); cx[z] = (float)ax_x(z); }
    unsigned rs = 12345u;
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const size_t i = ((size_t)z * H + y) * W + x;
        const double pz = (z + 0.5) * H8, py = (y + 0.5) * H8, px = (x + 0.5) * H8, r = hypot(py - ax_y(pz), px - ax_x(pz));
        w[i] = r >= 3 * 70 && r <= 370 ? 255 : 0;
        const double sw = ct_sector(pz, py, px); sec[i] = sw > 0.3;
        const double Wn = W_ctnet(pz, py, px), ph = 2 * M_PI * (Wn - floor(Wn)), amp = sw > 0.3 ? 0.2 : 1;
        pc[i] = (uint8_t)lround(128 + 127 * amp * cos(ph)); ps[i] = (uint8_t)lround(128 + 127 * amp * sin(ph));
        double gn[3], gt[3]; const double e = 0.5;
        for (int d = 0; d < 3; d++) {
            double a[3] = {pz, py, px}, b[3] = {pz, py, px}; a[d] += e; b[d] -= e;
            double dn = W_ctnet(a[0], a[1], a[2]) - W_ctnet(b[0], b[1], b[2]), dt = W_true(a[0], a[1], a[2]) - W_true(b[0], b[1], b[2]);
            gn[d] = (dn - round(dn)) / (2 * e); gt[d] = (dt - round(dt)) / (2 * e);
        }
        const double rn = sqrt(gn[0] * gn[0] + gn[1] * gn[1] + gn[2] * gn[2]), rt = sqrt(gt[0] * gt[0] + gt[1] * gt[1] + gt[2] * gt[2]);
        const double nz = gt[0] / rt, ny = gt[1] / rt, nx = gt[2] / rt;   /* the normals are right (n n^T knows no turn count) */
        const double tv[6] = {nz * nz, ny * ny, nx * nx, 0.5 + nz * ny, 0.5 + nz * nx, 0.5 + ny * nx};
        for (int k = 0; k < 6; k++) t[k][i] = (uint8_t)lround(255 * tv[k]);
        double s1 = 8 / (H8 * rn), s2 = 8 / (H8 * rt);
        sp[i] = (uint8_t)(s1 < 1 ? 1 : s1 > 255 ? 255 : lround(s1)); cs[i] = (uint8_t)(s2 < 1 ? 1 : s2 > 255 ? 255 : lround(s2));
        rs = rs * 1664525u + 1013904223u; const double noise = ((rs >> 8) / 16777216.0 - 0.5) * 30;
        const double cv = 95 + 50 * cos(2 * M_PI * (W_true(pz, py, px) - 0.2)) + noise;
        ct[i] = w[i] ? (uint8_t)lround(cv < 1 ? 1 : cv > 255 ? 255 : cv) : 0;
    }
    fit_fields f = {D, H, W, H8, pc, ps, {t[0], t[1], t[2], t[3], t[4], t[5]}, sp, w, cy, cx, {0}, 0, nullptr, ct, cs};
    fit_cfg c = fit_cfg_default(); c.verbose = getenv("FIT_CTV") != nullptr; c.cpu = cpu; c.dir = 1;
    c.w_ct = w_ct; c.ct_phi = 0.2f; c.w_cspacing = w_cs; c.ct_flood = g_ctflood;
    fit_model *m = fit_run(&f, &c);
    if (!m) { fprintf(stderr, "FAIL: fit_run (ct)\n"); fails++; return 0; }
    int hist[401] = {0};
    for (size_t i = 0; i < n; i++) {
        if (!w[i] || sec[i]) continue;
        const int z = (int)(i / ((size_t)H * W)), y = (int)((i / W) % H), x = (int)(i % W);
        const double pz = (z + 0.5) * H8, py = (y + 0.5) * H8, px = (x + 0.5) * H8; const long k = lround(fit_W(m, pz, py, px) - W_true(pz, py, px));
        if (k >= -200 && k <= 200) hist[k + 200]++;
    }
    int k0 = 0; for (int k = 1; k < 401; k++) if (hist[k] > hist[k0]) k0 = k;   /* the gauge from outside the sector */
    size_t nd = 0, right = 0, close = 0, ns = 0, sright = 0, sclose = 0;
    for (size_t i = 0; i < n; i++) {
        if (!w[i]) continue;
        const int z = (int)(i / ((size_t)H * W)), y = (int)((i / W) % H), x = (int)(i % W);
        const double pz = (z + 0.5) * H8, py = (y + 0.5) * H8, px = (x + 0.5) * H8;
        const double d = fit_W(m, pz, py, px) - W_true(pz, py, px) - (k0 - 200);
        nd++; right += fabs(d) < 0.5; close += fabs(d) < 0.08;
        if (sec[i]) { ns++; sright += fabs(d) < 0.5; sclose += fabs(d) < 0.08; }
    }
    *all = (double)right / nd; *prec_out = (double)close / nd;
    printf("ct case%s%s, w_ct %g, w_cspacing %g: ids right %.4f (sector %.4f), |error| < 0.08 turn %.4f (sector %.4f)\n", cpu ? " cpu" : "", g_ctflood ? " flood" : "", w_ct, w_cs,
           *all, ns ? (double)sright / ns : 0, *prec_out, ns ? (double)sclose / ns : 0);
    const double r = ns ? (double)sright / ns : 0;
    fit_free(m);
    free(pc); free(ps); free(sp); free(w); free(ct); free(cs); free(sec); free(cy); free(cx); for (int k = 0; k < 6; k++) free(t[k]);
    return r;
}

/* (from the main checkout, 2026-10-20) a local stack with smooth sheet curvature, a confident 0.35-turn phase error, a missing recto
   patch and a small false recto patch: recto anchors (fit --w-recto, --phase-robust, --unwrap-voxels -1) must bring the geometry back,
   compared against the analytic surfaces rather than the corrupted prediction */
static double local_W(double z, double y, double x) {
    return (x + 3) / 12 + 0.12 * sin(2 * M_PI * z / 32) * sin(2 * M_PI * y / 32);
}
static int compare_error(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }
static void local_errors(const fit_model *m, double *mean, double *p90, double *min_dx) {
    const int D = 32, H = 32, W = 160; size_t n = 0; float *err = malloc((size_t)D * H * W * sizeof(float));
    double sm = 0; *min_dx = 1e9;
    const double gauge = round(fit_W(m, 16.5, 16.5, 16.5) - local_W(16.5, 16.5, 16.5));
    for (int z = 2; z < D - 2; z++) for (int y = 2; y < H - 2; y++) for (int x = 4; x < W - 4; x++) {
        const double w = fit_W(m, z + .5, y + .5, x + .5), e = fabs(w - local_W(z + .5, y + .5, x + .5) - gauge);
        err[n++] = (float)e; sm += e;
        const double dx = fit_W(m, z + .5, y + .5, x + 1.5) - w; if (dx < *min_dx) *min_dx = dx;
    }
    qsort(err, n, sizeof(float), compare_error);
    *mean = sm / n; *p90 = err[n * 9 / 10]; free(err);
}
static void run_local_surface_case(void) {
    const int D = 32, H = 32, W = 160; const size_t n = (size_t)D * H * W;
    uint8_t *a = malloc(11 * n); float cy[33], cx[33];
    for (int z = 0; z <= D; z++) { cy[z] = 16; cx[z] = -1200; }
    fit_fields f = {.D = D, .H = H, .W = W, .h = 1, .pc = a, .ps = a + n, .t = {a + 2 * n, a + 3 * n, a + 4 * n, a + 5 * n, a + 6 * n, a + 7 * n},
                    .sp = a + 8 * n, .w = a + 9 * n, .cy = cy, .cx = cx, .recto = a + 10 * n};
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const size_t i = ((size_t)z * H + y) * W + x; const double zz = z + .5, yy = y + .5, xx = x + .5, w = local_W(zz, yy, xx);
        const double bad = x >= 48 && x < 112 ? .35 : 0;
        a[i] = (uint8_t)lround(128 + 127 * cos(2 * M_PI * (w + bad))); a[n + i] = (uint8_t)lround(128 + 127 * sin(2 * M_PI * (w + bad)));
        double g[3] = {.12 * (2 * M_PI / 32) * cos(2 * M_PI * zz / 32) * sin(2 * M_PI * yy / 32), .12 * (2 * M_PI / 32) * sin(2 * M_PI * zz / 32) * cos(2 * M_PI * yy / 32), 1. / 12};
        const double rho = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]); for (int k = 0; k < 3; k++) g[k] /= rho;
        const double t[6] = {g[0] * g[0], g[1] * g[1], g[2] * g[2], .5 + g[0] * g[1], .5 + g[0] * g[2], .5 + g[1] * g[2]};
        for (int k = 0; k < 6; k++) a[(2 + k) * n + i] = (uint8_t)lround(255 * t[k]);
        a[8 * n + i] = (uint8_t)lround(8 / rho); a[9 * n + i] = 255;
        const double d = w - round(w); double r = exp(-.5 * d * d / (.07 * .07));
        if (x >= 64 && x < 88 && y >= 12 && y < 20) r = 0;                       /* missing recto: density / normals must bridge it */
        if (x >= 40 && x < 42 && y >= 12 && y < 16 && z >= 12 && z < 16) r = 1;   /* a small false positive */
        a[10 * n + i] = (uint8_t)lround(255 * r);
    }
    fit_cfg c = fit_cfg_default(); c.dir = 1; c.core = 0; c.verbose = 0; c.knot_dz = 128; c.nstage = 4; c.phase_stage = 2;
    const int L[4] = {32, 16, 8, 4}; for (int k = 0; k < 4; k++) { c.lattice[k] = L[k]; c.steps[k] = 250; }
    double old_mean = 0, old_p90 = 0, old_dx = 0;
    fit_model *baseline = fit_run(&f, &c); CHECK(baseline, "local baseline fit");
    if (baseline) { local_errors(baseline, &old_mean, &old_p90, &old_dx); fit_free(baseline); }
    for (int mode = 0; mode < 2; mode++) {
        c.w_recto = 8; c.phase_robust = .10f; c.w_phase = mode ? 0.f : .25f; c.unwrap_voxels = -1;
        fit_model *m = fit_run(&f, &c); CHECK(m, "recto local fit mode %d", mode); if (!m) continue;
        double mean, p90, dx; local_errors(m, &mean, &p90, &dx);
        printf("local recto mode %d: mean %.4f p90 %.4f turn min dW/dx %.4f; baseline %.4f / %.4f\n", mode, mean, p90, dx, old_mean, old_p90);
        CHECK(mean < .04 && p90 < .08, "local recto mode %d geometry error %.4f / %.4f", mode, mean, p90);
        CHECK(mean < .5 * old_mean, "local recto mode %d did not halve the corrupted-phase error", mode);
        CHECK(dx > .025, "local recto mode %d collapsed / reversed sheet spacing %.4f", mode, dx);
        fit_free(m);
    }
    free(a);
}
/* the optional r plane: saved, cropped in z on load, ignored when fields.json says "recto": false, an error when declared but missing */
static void check_recto_fields(void) {
    char dir[] = "/tmp/ufsm-fit-recto-XXXXXX"; if (!mkdtemp(dir)) { CHECK(0, "recto field fixture directory"); return; }
    const int shape[3] = {3, 2, 2}; const long long org[3] = {10, 20, 30}; uint8_t data[10][12], recto[12], *a[14] = {0}; float cy[7] = {0}, cx[7] = {0};
    for (int k = 0; k < 10; k++) { memset(data[k], k < 2 ? 128 : 255, 12); a[k] = data[k]; }
    for (int i = 0; i < 12; i++) recto[i] = (uint8_t)(20 * i);
    CHECK(!fit_fields_save_ex(dir, org, shape, 2, 9.6, a, 0, recto, cy, cx, 7), "save optional recto");
    fit_fields_buf b;
    int rc = fit_fields_load(dir, 1, 3, 1, &b); CHECK(!rc, "load recto z crop");
    if (!rc) { CHECK(b.f.recto && !memcmp(b.f.recto, recto + 4, 8), "recto crop coordinates"); CHECK(b.origin[0] == 12, "recto cropped origin"); fit_fields_free(&b); }
    CHECK(!fit_fields_save(dir, org, shape, 2, 9.6, a, cy, cx, 7), "save legacy fields over recto fields");
    rc = fit_fields_load(dir, 0, 0, 1, &b); CHECK(!rc, "load fields without declared recto");
    if (!rc) { CHECK(!b.f.recto, "stale recto must be ignored"); fit_fields_free(&b); }
    CHECK(!fit_fields_save_ex(dir, org, shape, 2, 9.6, a, 0, recto, cy, cx, 7), "save declared recto");
    char p[256]; snprintf(p, sizeof p, "%s/r.raw", dir); unlink(p);
    rc = fit_fields_load(dir, 0, 0, 1, &b); CHECK(rc != 0, "missing declared recto must fail"); if (!rc) fit_fields_free(&b);
    const char *names[11] = {"pc.raw", "ps.raw", "t0.raw", "t1.raw", "t2.raw", "t3.raw", "t4.raw", "t5.raw", "sp.raw", "w.raw", "fields.json"};
    for (int k = 0; k < 11; k++) { snprintf(p, sizeof p, "%s/%s", dir, names[k]); unlink(p); } rmdir(dir);
}

int main(void) {
    nn_init(0);
    if (getenv("FIT_BIAS")) g_bias = atof(getenv("FIT_BIAS"));
    if (getenv("FIT_SIMPLE")) { g_wob = atof(getenv("FIT_SIMPLE")); g_pack = 0; g_dpitch = 0; g_hole = 0; }
    if (!getenv("FIT_CTONLY")) {   /* (FIT_CTONLY: the CT cases alone) */
    run_case(+1);
    run_case(-1);
    g_split = 0.5; g_tile = 32; run_case(+1);   /* 18 tiles: pieces stitched across tile faces, the hole and the edge by the serial pass */
    g_split = 0; g_tile = 256;
    g_cpu = 1; run_case(+1); run_case(-1);       /* the CPU backend (src/fit_terms.h: the same terms) */
    g_robust = 0.1f; g_bend = 1.f; run_case(+1); /* the Cauchy phase residual and curvature-adaptive smoothness keep a clean fit */
    g_cpu = 0; run_case(-1);                     /* ... on the GPU too */
    g_robust = 0; g_bend = 0; g_pgrad = 1.f; run_case(+1); g_cpu = 1; run_case(-1);   /* phase-gradient matching, both backends */
    }
    g_pgrad = 0; g_cpu = 0;
    run_local_surface_case();   /* recto anchors on a corrupted phase (main checkout) */
    check_recto_fields();
    {   /* the CT sheet spacing of a profile: papyrus peaks of period 12.5 samples (0.5 native voxels apart) with noise; 2 peaks at least */
        float pr[160]; unsigned rs = 7u;
        for (int k = 0; k < 160; k++) { rs = rs * 1664525u + 1013904223u; pr[k] = (float)(90 + 40 * cos(2 * M_PI * (k - 3.3) / 12.5) + ((rs >> 8) / 16777216.0 - 0.5) * 20); }
        const double sp1 = fit_ct_profile_spacing(pr, 160, 0.5, 0.3), sp0 = fit_ct_profile_spacing(pr, 16, 0.5, 0.3);
        printf("CT profile spacing %.3f native voxels (true 6.25); one period: %.3f\n", sp1, sp0);
        CHECK(fabs(sp1 - 6.25) < 0.15, "CT profile spacing %.3f, not 6.25", sp1);
        CHECK(sp0 == 0, "a profile with fewer than 2 whole peaks gave spacing %.3f", sp0);
        /* the parity: local contrast, sign and the no-CT code */
        const int D = 8, H = 8, W = 64; uint8_t ct[8 * 8 * 64], pm[8 * 8 * 64];
        for (int i = 0; i < D * H * W; i++) { const int x = i % W; ct[i] = x < 4 ? 0 : (uint8_t)((x < 32 ? 60 : 140) + (x % 8 < 3 ? 40 : 0)); }
        fit_ct_parity(ct, D, H, W, 4, pm);
        int ok = 1; for (int i = 0; i < D * H * W; i++) { const int x = i % W; if (x < 4) ok &= pm[i] == 128; else if (x >= 8 && x < 28) ok &= (x % 8 < 3) == (pm[i] > 128); }
        CHECK(ok, "CT parity: papyrus above / void below the local mean, 128 without CT");
    }
    {   /* CT: a weak, wrong packed sector recovered by the CT parity and spacing, on both backends */
        double a0, p0, a1, p1, a2, p2;
        const double s0 = run_ct_case(1, 0.f, 0.f, &a0, &p0);
        const double s1 = run_ct_case(1, 2.f, 1.f, &a1, &p1);
        const double s2 = run_ct_case(0, 2.f, 1.f, &a2, &p2);
        CHECK(s0 < 0.95, "the weak sector should break the network-only fit (sector ids %.4f)", s0);
        if (getenv("FIT_CTONLY")) { run_ct_case(1, 2.f, 0.f, &a0, &p0); run_ct_case(1, 0.f, 1.f, &a0, &p0); run_ct_case(0, 0.5f, 0.f, &a0, &p0); }   /* the terms alone */
        CHECK(s1 >= 0.98 && a1 >= 0.99 && p1 >= 0.9, "cpu: CT terms: sector ids %.4f, ids %.4f, |e| < 0.08 %.4f", s1, a1, p1);
        CHECK(s2 >= 0.98 && a2 >= 0.99 && p2 >= 0.9, "gpu: CT terms: sector ids %.4f, ids %.4f, |e| < 0.08 %.4f", s2, a2, p2);
        g_ctflood = 1; const double s3 = run_ct_case(1, 2.f, 0.f, &a1, &p1); g_ctflood = 0;   /* with the CT phase in the voxel unwrapping */
        CHECK(s3 >= 0.98 && a1 >= 0.99 && p1 >= 0.9, "cpu, CT flood: sector ids %.4f, ids %.4f, |e| < 0.08 %.4f", s3, a1, p1);
    }
    if (fails) { fprintf(stderr, "test_fit: %d failure(s)\n", fails); return 1; }
    printf("test_fit: OK\n");
    return 0;
}
