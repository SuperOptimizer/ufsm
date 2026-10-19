/* Fit fields of task unwrap2 checkpoints (src/nn_fit.cu, src/fit.c).
   (1) predict's pooling with the unwrap2 channel layout (phase 0-1, normal 2-7, density 8; no recto) on logits that are the task
       unwrap layout's rows 2-10 gives byte-identical pc ps t0..t5 sp w planes (what --init-from maps from a wind96 checkpoint).
   (2) the winding mod 4 relative planes (nn_fit_accum_w4 / nn_fit_finalize_w4) against a host reference.
   (3) fit_fields_save_w4 / fit_fields_load round trip; 10-plane fields load without the planes.
   (4) the fitter's winding mod 4 term on synthetic spiral fields whose phase under-counts a packed sector (the networks' failure):
       planes from the true W restore the winding ids, and the term is ~0 at the truth. */
#include "fit.h"
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)
static uint64_t rs = 0x9E3779B97F4A7C15ull;
static double urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 11) * (1.0 / 9007199254740992.0); }

static void layout_and_planes(void) {
    const int Wn = 40, halo = 8, h = 4; const size_t w3 = (size_t)Wn * Wn * Wn;
    const int box[3] = {Wn, Wn, Wn}, F[3] = {Wn / h, Wn / h, Wn / h}; const size_t nf = (size_t)F[0] * F[1] * F[2];
    float *l1 = malloc(11 * w3 * 4), *l2 = malloc(11 * w3 * 4); uint8_t *ct = malloc(w3);
    for (size_t i = 0; i < 11 * w3; i++) l1[i] = (float)(4 * urand() - 2);
    for (size_t i = 0; i < w3; i++) ct[i] = urand() < 0.8 ? 100 : 0;
    /* the winding mod 4 channels: a smooth field W = 0.05 x + 0.02 y (turns), with logit noise */
    float *w4 = malloc(2 * w3 * 4);
    for (int z = 0; z < Wn; z++) for (int y = 0; y < Wn; y++) for (int x = 0; x < Wn; x++) {
        const size_t i = ((size_t)z * Wn + y) * Wn + x; const double a = 2 * M_PI * (0.05 * x + 0.02 * y + 0.01 * z) / 4;
        const double pc = fmin(0.98, fmax(0.02, 0.5 + 0.5 * cos(a) + 0.01 * (urand() - 0.5))), ps = fmin(0.98, fmax(0.02, 0.5 + 0.5 * sin(a) + 0.01 * (urand() - 0.5)));   /* a trained head: sigma = the soft target */
        w4[i] = (float)log(pc / (1 - pc)); w4[w3 + i] = (float)log(ps / (1 - ps));
    }
    memcpy(l2, l1 + 2 * w3, 9 * w3 * 4); memcpy(l2 + 9 * w3, w4, 2 * w3 * 4);
    float *dl1 = nn_malloc(11 * w3 * 4), *dl2 = nn_malloc(11 * w3 * 4), *acc1 = nn_malloc(12 * nf * 4), *acc2 = nn_malloc(12 * nf * 4), *acc4 = nn_malloc(9 * nf * 4);
    uint8_t *dct = nn_malloc(w3), *a1 = nn_malloc(10 * nf), *a2 = nn_malloc(10 * nf), *a4 = nn_malloc(4 * nf);
    nn_h2d(dl1, l1, 11 * w3 * 4); nn_h2d(dl2, l2, 11 * w3 * 4); nn_h2d(dct, ct, w3);
    nn_zero(acc1, 12 * nf * 4); nn_zero(acc2, 12 * nf * 4); nn_zero(acc4, 9 * nf * 4);
    for (int hh = 0; hh < 2; hh++) {   /* both pooling kernels: a thread per cell (h <= 4) and a block per cell (h > 4) */
        const int hc = hh ? 8 : h, Fc[3] = {Wn / hc, Wn / hc, Wn / hc}; const size_t nfc = (size_t)Fc[0] * Fc[1] * Fc[2];
        nn_zero(acc1, 12 * nfc * 4); nn_zero(acc2, 12 * nfc * 4);
        nn_fit_accum_dm(dl1, dct, nullptr, Wn, Wn, halo, halo, 0, 0, 0, box, hc, Fc, 0, 2, 4, 10, 4.f, 1, acc1);
        nn_fit_accum_dm(dl2, dct, nullptr, Wn, Wn, halo, halo, 0, 0, 0, box, hc, Fc, -1, 0, 2, 8, 4.f, 1, acc2);
        nn_fit_finalize(acc1, nfc, hc, a1); nn_fit_finalize(acc2, nfc, hc, a2); nn_sync();
        uint8_t *h1 = malloc(10 * nfc), *h2 = malloc(10 * nfc); nn_d2h(h1, a1, 10 * nfc); nn_d2h(h2, a2, 10 * nfc);
        size_t nm = 0, nd = 0; for (size_t i = 0; i < 10 * nfc; i++) { nd += h1[i] != h2[i]; nm += h1[i] != (i < 2 * nfc ? 128 : 0); }
        printf("layout (h %d): unwrap vs unwrap2 fit-field planes: %zu of %zu bytes differ (%zu informative)\n", hc, nd, 10 * nfc, nm);
        CHECK(nd == 0 && nm > 0, "h %d: the unwrap2 layout's planes differ from the unwrap layout's (%zu bytes)", hc, nd);
        free(h1); free(h2);
    }
    /* the relative planes vs a host reference */
    const int dd = 6;
    nn_fit_accum_w4(dl2, dct, nullptr, Wn, Wn, halo, halo, 0, 0, 0, box, h, F, 9, dd, acc4);
    nn_fit_finalize_w4(acc4, nf, a4); nn_sync();
    uint8_t *h4 = malloc(4 * nf); nn_d2h(h4, a4, 4 * nf);
    int worst = 0; size_t ninf = 0;
    for (int fz = 0; fz < F[0]; fz++) for (int fy = 0; fy < F[1]; fy++) for (int fx = 0; fx < F[2]; fx++) {
        const size_t ci = ((size_t)fz * F[1] + fy) * F[2] + fx;
        double re[3] = {0}, im[3] = {0}, cn[3] = {0};
        for (int a = 0; a < h; a++) for (int b = 0; b < h; b++) for (int c = 0; c < h; c++) {
            const int z = fz * h + a, y = fy * h + b, x = fx * h + c;
            if (z < halo || z >= Wn - halo || y < halo || y >= Wn - halo || x < halo || x >= Wn - halo) continue;
            const size_t i = ((size_t)z * Wn + y) * Wn + x; if (!ct[i]) continue;
            const double pr = 2 / (1 + exp(-w4[i])) - 1, pi = 2 / (1 + exp(-w4[w3 + i])) - 1;
            const size_t st[3] = {(size_t)Wn * Wn * dd, (size_t)Wn * dd, (size_t)dd};
            for (int ax = 0; ax < 3; ax++) {
                const size_t j = i + st[ax]; if (!ct[j]) continue;
                const double qr = 2 / (1 + exp(-w4[j])) - 1, qi = 2 / (1 + exp(-w4[w3 + j])) - 1;
                re[ax] += qr * pr + qi * pi; im[ax] += qi * pr - qr * pi; cn[ax] += 1;
            }
        }
        double q = 0; int na = 0;
        for (int ax = 0; ax < 3; ax++) {
            int b = 0; if (cn[ax] > 0) { b = (int)lrint(atan2(im[ax], re[ax]) * 256 / (2 * M_PI)) & 255; q += fmin(1, hypot(re[ax], im[ax]) / cn[ax]); na++; }
            int e = abs(b - h4[ax * nf + ci]); if (e > 128) e = 256 - e; if (e > worst) worst = e;
        }
        const int qb = na ? (int)lrint(255 * q / na) : 0; if (abs(qb - h4[3 * nf + ci]) > worst) worst = abs(qb - h4[3 * nf + ci]);
        ninf += h4[3 * nf + ci] > 0;
        if (na && fz == 4 && fy == 4 && fx == 4) {   /* the analytic difference: 0.05 d turns along x */
            const int ex = (int)lrint(0.05 * dd * 64) & 255;
            int e = abs(ex - h4[2 * nf + ci]); if (e > 128) e = 256 - e;
            CHECK(e <= 3, "dx plane %d, the field's %d", h4[2 * nf + ci], ex);
        }
    }
    printf("relative planes vs host reference: worst byte difference %d (%zu informative cells)\n", worst, ninf);
    CHECK(worst <= 1 && ninf > 0, "relative planes differ from the reference by %d", worst);
    free(h4); free(l1); free(l2); free(ct); free(w4);
    nn_free(dl1); nn_free(dl2); nn_free(acc1); nn_free(acc2); nn_free(acc4); nn_free(dct); nn_free(a1); nn_free(a2); nn_free(a4);
}

/* ---- the spiral of test_fit (dir +1, a packed sector the phase under-counts) ---- */
static const double H8 = 8, R0 = 384, PACK = 0.6;
static double ax_y(double z) { return R0 + 0.02 * z; }
static double ax_x(double z) { return R0 - 0.01 * z; }
static double pitch_of(double z) { return 60 + 10 * z / 384; }
static double packed(double y, double x, double z) {
    const double dy = y - ax_y(z), dx = x - ax_x(z), r = hypot(dy, dx), th = atan2(dy, dx);
    double dth = th - 1.0; dth -= 2 * M_PI * floor((dth + M_PI) / (2 * M_PI));
    return PACK / 60 * r * exp(-dth * dth / (2 * 0.3 * 0.3));
}
static double U_true(double z, double y, double x) { const double r = hypot(y - ax_y(z), x - ax_x(z)); return r / pitch_of(z) + 0.4 * sin(atan2(y - ax_y(z), x - ax_x(z))) * r / 300 + packed(y, x, z); }
static double W_true(double z, double y, double x) { return U_true(z, y, x) - atan2(y - ax_y(z), x - ax_x(z)) / (2 * M_PI); }
static double dW_true(const double a[3], const double b[3]) {   /* W(b) - W(a), theta wrapped */
    double dth = atan2(b[1] - ax_y(b[0]), b[2] - ax_x(b[0])) - atan2(a[1] - ax_y(a[0]), a[2] - ax_x(a[0]));
    dth -= 2 * M_PI * floor((dth + M_PI) / (2 * M_PI));
    return U_true(b[0], b[1], b[2]) - U_true(a[0], a[1], a[2]) - dth / (2 * M_PI);
}
static double run_spiral(int with_w4, double bias, double *w4loss) {
    const int D = 48, H = 96, W = 96, dd = getenv("FITW4_D") ? atoi(getenv("FITW4_D")) : 96; const size_t n = (size_t)D * H * W;   /* dd: ~1.5 turns of the 60-voxel pitch */
    uint8_t *pc = malloc(n), *ps = malloc(n), *t[6], *sp = malloc(n), *w = malloc(n), *wd[4];
    for (int k = 0; k < 6; k++) t[k] = malloc(n);
    for (int k = 0; k < 4; k++) wd[k] = malloc(n);
    const int na = (int)(D * H8) + 1;
    float *cy = malloc(na * sizeof(float)), *cx = malloc(na * sizeof(float));
    for (int z = 0; z < na; z++) { cy[z] = (float)ax_y(z); cx[z] = (float)ax_x(z); }
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const size_t i = ((size_t)z * H + y) * W + x;
        const double pz = (z + 0.5) * H8, py = (y + 0.5) * H8, px = (x + 0.5) * H8, r = hypot(py - ax_y(pz), px - ax_x(pz));
        w[i] = r >= 3 * 70 && r <= 370 ? 255 : 0;
        const double Wv = W_true(pz, py, px) - bias * packed(py, px, pz), ph = 2 * M_PI * (Wv - floor(Wv));   /* the phase under-counts the packed sector */
        pc[i] = (uint8_t)lround(128 + 127 * cos(ph)); ps[i] = (uint8_t)lround(128 + 127 * sin(ph));
        double g[3]; const double e = 0.5;
        for (int d = 0; d < 3; d++) { double a[3] = {pz, py, px}, b[3] = {pz, py, px}; a[d] -= e; b[d] += e; g[d] = dW_true(a, b) / (2 * e); }
        const double rho = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]), nz = g[0] / rho, ny = g[1] / rho, nx = g[2] / rho;
        const double tv[6] = {nz * nz, ny * ny, nx * nx, 0.5 + nz * ny, 0.5 + nz * nx, 0.5 + ny * nx};
        for (int k = 0; k < 6; k++) t[k][i] = (uint8_t)lround(255 * tv[k]);
        const double rb = rho * (1 - bias * 0.5 * (packed(py, px, pz) > 0.2 * PACK / 60 * r));   /* the density under-counts there too */
        double s = 8 / (H8 * rb); sp[i] = (uint8_t)(s < 1 ? 1 : s > 255 ? 255 : lround(s));
        for (int d = 0; d < 3; d++) {   /* the true W difference over dd voxels along each axis, mod 4 */
            double a[3] = {pz, py, px}, b[3] = {pz, py, px}; b[d] += dd;
            double q = fmod(dW_true(a, b), 4.0); if (q < 0) q += 4;
            wd[d][i] = (uint8_t)(lround(q * 64) & 255);
        }
        wd[3][i] = 255;
    }
    fit_fields f = {D, H, W, H8, pc, ps, {t[0], t[1], t[2], t[3], t[4], t[5]}, sp, w, cy, cx, {0}, 0};
    if (with_w4) { for (int k = 0; k < 4; k++) f.wd[k] = wd[k]; f.w4_d = dd; }
    fit_cfg c = fit_cfg_default(); c.verbose = getenv("FITW4_V") != nullptr; c.dir = 1; if (getenv("FITW4_W")) c.w_wind4 = (float)atof(getenv("FITW4_W"));
    fit_model *m = fit_run(&f, &c);
    if (!m) { fprintf(stderr, "FAIL: fit_run\n"); fails++; return 0; }
    int hist[401] = {0}; size_t nd = 0, right = 0;
    for (size_t i = 0; i < n; i++) {
        if (!w[i]) continue;
        const int z = (int)(i / ((size_t)H * W)), y = (int)((i / W) % H), x = (int)(i % W);
        const long k = lround(fit_W(m, (z + 0.5) * H8, (y + 0.5) * H8, (x + 0.5) * H8) - W_true((z + 0.5) * H8, (y + 0.5) * H8, (x + 0.5) * H8));
        if (k >= -200 && k <= 200) hist[k + 200]++;
        nd++;
    }
    int k0 = 0; for (int k = 1; k < 401; k++) if (hist[k] > hist[k0]) k0 = k;
    right = (size_t)hist[k0];
    if (w4loss) *w4loss = m->loss[0];
    fit_free(m);
    free(pc); free(ps); free(sp); free(w); free(cy); free(cx); for (int k = 0; k < 6; k++) free(t[k]); for (int k = 0; k < 4; k++) free(wd[k]);
    return (double)right / nd;
}

int main(void) {
    nn_init(0);
    layout_and_planes();
    /* (3) save / load round trip */
    {
        char dir[] = "/tmp/test_fit_w4_XXXXXX"; if (!mkdtemp(dir)) { fprintf(stderr, "FAIL: mkdtemp\n"); return 1; }
        const int shp[3] = {5, 6, 7}; const size_t nf = 5 * 6 * 7; const long long org[3] = {1, 2, 3};
        uint8_t *a[14]; for (int k = 0; k < 14; k++) { a[k] = malloc(nf); for (size_t i = 0; i < nf; i++) a[k][i] = (uint8_t)(k * 17 + i); }
        const int na = (int)(5 * 4) + 1; float *cy = calloc(na, 4), *cx = calloc(na, 4);
        CHECK(!fit_fields_save_w4(dir, org, shp, 4, 9.6, a, 12, cy, cx, na), "fit_fields_save_w4");
        fit_fields_buf b; CHECK(!fit_fields_load(dir, 0, 0, 1, &b), "fit_fields_load");
        int same = b.f.w4_d == 12; for (int k = 0; k < 4; k++) same &= b.f.wd[k] && !memcmp(b.f.wd[k], a[10 + k], nf);
        same &= !memcmp(b.f.pc, a[0], nf) && !memcmp(b.f.w, a[9], nf);
        CHECK(same, "the winding mod 4 planes did not round-trip");
        fit_fields_free(&b);
        CHECK(!fit_fields_load(dir, 0, 0, 2, &b) && !b.f.wd[0], "pooled loads carry no winding mod 4 planes"); fit_fields_free(&b);
        char cmd[256]; snprintf(cmd, sizeof cmd, "rm -rf %s", dir); if (system(cmd)) {}
        CHECK(!fit_fields_save(dir, org, shp, 4, 9.6, a, cy, cx, na), "fit_fields_save");
        CHECK(!fit_fields_load(dir, 0, 0, 1, &b) && !b.f.wd[0] && b.f.w4_d == 0, "10-plane fields loaded with winding mod 4 planes"); fit_fields_free(&b);
        if (system(cmd)) {}
        printf("save / load round trip: ok\n");
        for (int k = 0; k < 14; k++) { free(a[k]); } free(cy); free(cx);
    }
    /* (4) the fitter term */
    const double r0 = run_spiral(0, 0.0, nullptr), rb = run_spiral(0, 0.4, nullptr), rw = run_spiral(1, 0.4, nullptr), rt = run_spiral(1, 0.0, nullptr);
    printf("fit winding ids: true phase %.4f; phase under-counting the packed sector by 40%%: %.4f without, %.4f with the winding mod 4 term (true phase + term %.4f)\n", r0, rb, rw, rt);
    CHECK(r0 >= 0.995 && rt >= 0.995, "true fields: %.4f / %.4f", r0, rt);
    CHECK(rw >= rb && rw >= 0.98, "the winding mod 4 term did not restore the count: %.4f (without %.4f)", rw, rb);
    if (fails) { fprintf(stderr, "test_fit_w4: %d failure(s)\n", fails); return 1; }
    printf("test_fit_w4: OK\n");
    return 0;
}
