/* ufsm unroll-qa (src/unrollqa.c) on a synthetic Archimedean spiral r = 20 + 10 (W + theta / 2 pi) about a vertical axis at (128, 128)
   native (dir +1, h 2, 64 x 256 x 256 native): fit fields made from the true W (phase cos / sin 2 pi W, normal = grad W, spacing 1 / |grad W|,
   winding mod 4 over 32 voxels), H fibres along sheets (W = k + 0.3, half turns at z 24 / 40), and two hand-made models (verdicts read from --out/gates.tsv, values from --out/boxes.tsv):
   - right (pitch 10, offset -2): every fibre stays in its sheet (0 persistent changes), wind4 agreement 1 on all axes, tilt ~0, no
     density outliers, no low-confidence material, no phase-unwrap misses; the gates pass, exit 0 even under --strict;
   - wrong: the same plus a lattice term of 1.5 turns per 128 voxels along y (W slides across the sheets along each fibre: ~1.4 turns per
     half turn of fibre): the fibre gate fires (gates.tsv "warn"), exit 0 by default and 3 under --strict; the median tilt grows.
   Tiled mode (no --box) runs over the box's material tiles. CPU only. */
#include "afv.h"
#include "fit.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int cmd_unroll_qa(int argc, char **argv);
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

enum { NZ = 64, NYX = 256, H = 2, D = NZ / H, N = NYX / H, W4D = 32 };
static const double C0 = 128;
static double W_true(double y, double x) { return (hypot(y - C0, x - C0) - 20) / 10 - atan2(y - C0, x - C0) / (2 * M_PI); }
static double dW_true(double y0, double x0, double y1, double x1) {   /* continuous difference (theta continued) */
    double dt = atan2(y1 - C0, x1 - C0) - atan2(y0 - C0, x0 - C0); dt -= 2 * M_PI * floor(dt / (2 * M_PI) + 0.5);
    return (hypot(y1 - C0, x1 - C0) - hypot(y0 - C0, x0 - C0)) / 10 - dt / (2 * M_PI);
}
static uint8_t q255(double v) { v = round(v); return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v); }

static int save_model(const char *dir, double ygrad) {   /* ygrad: lattice delta in turns per 128 native voxels along y */
    fit_model m = {0}; m.dir = 1; m.nk = 2; m.knot_dz = 64; m.L = 128; m.lz = 2; m.ly = 3; m.lx = 3; m.D = D; m.H = N; m.W = N; m.h = H; m.um = 9.6;
    float pt[2] = {10, 10}, of[2] = {-2, -2}, de[18]; m.pitch = pt; m.offset = of; m.delta = de;
    for (int a = 0; a < 2; a++) for (int b = 0; b < 3; b++) for (int c = 0; c < 3; c++) de[(a * 3 + b) * 3 + c] = (float)(ygrad * b);
    float cy[NZ + 1], cx[NZ + 1]; for (int i = 0; i <= NZ; i++) cy[i] = cx[i] = (float)C0; m.cy = cy; m.cx = cx;
    if (!ygrad && fabs(fit_W(&m, 30, 150, 190) - W_true(150, 190)) > 1e-4) return -1;
    return fit_save(&m, dir);
}
/* boxes.tsv row 1 -> the value of column name */
static double col(const char *path, const char *name) {
    FILE *f = fopen(path, "r"); if (!f) return NAN; char hd[4096], rw[4096]; double v = NAN;
    if (fgets(hd, sizeof hd, f) && fgets(rw, sizeof rw, f)) {
        char *sh, *sr, *th = strtok_r(hd, "\t\n", &sh), *tr = strtok_r(rw, "\t\n", &sr);
        while (th && tr) { if (!strcmp(th, name)) { v = atof(tr); break; } th = strtok_r(nullptr, "\t\n", &sh); tr = strtok_r(nullptr, "\t\n", &sr); }
    }
    fclose(f); return v;
}
/* gates.tsv of an --out dir: the number of gates with verdict "warn" */
static int warns(const char *out) {
    char p[400]; snprintf(p, sizeof p, "%s/gates.tsv", out); FILE *f = fopen(p, "r"); char l[512]; int w = 0;
    while (f && fgets(l, sizeof l, f)) w += strstr(l, "\twarn") != nullptr; if (f) fclose(f); return w;
}

int main(void) {
    char root[] = "/tmp/ufsm-unrollqa-XXXXXX"; if (!mkdtemp(root)) { fprintf(stderr, "FAIL: tmp dir\n"); return 1; }
    /* ---- fields ---- */
    const size_t n = (size_t)D * N * N; uint8_t *a[14]; for (int k = 0; k < 14; k++) a[k] = calloc(n, 1);
    for (int z = 0; z < D; z++) for (int y = 0; y < N; y++) for (int x = 0; x < N; x++) {
        const size_t i = ((size_t)z * N + y) * N + x; const double py = (y + 0.5) * H, px = (x + 0.5) * H, r = hypot(py - C0, px - C0);
        if (r < 24) { a[0][i] = a[1][i] = 128; continue; }   /* the core: no material */
        const double w = W_true(py, px), gy = (py - C0) / (10 * r) - (px - C0) / (2 * M_PI * r * r), gx = (px - C0) / (10 * r) + (py - C0) / (2 * M_PI * r * r);
        const double gn = hypot(gy, gx), ny = gy / gn, nx = gx / gn;
        a[0][i] = q255(128 + 127 * cos(2 * M_PI * w)); a[1][i] = q255(128 + 127 * sin(2 * M_PI * w));
        a[2][i] = 0; a[3][i] = q255(255 * ny * ny); a[4][i] = q255(255 * nx * nx); a[5][i] = 128; a[6][i] = 128; a[7][i] = q255(255 * (0.5 + ny * nx));
        a[8][i] = q255(8 / (gn * H)); a[9][i] = 255;
        const double e[3][2] = {{py, px}, {py + W4D, px}, {py, px + W4D}};
        a[10][i] = 0;   /* along z: W does not change */
        for (int k = 1; k < 3; k++) { double d = dW_true(py, px, e[k][0], e[k][1]); d -= 4 * floor(d / 4); a[10 + k][i] = (uint8_t)((int)round(64 * d) & 255); }
        a[13][i] = 255;
    }
    float cy[NZ + 1], cx[NZ + 1]; for (int i = 0; i <= NZ; i++) cy[i] = cx[i] = (float)C0;
    char fd[300]; snprintf(fd, sizeof fd, "%s/fields", root);
    CHECK(!fit_fields_save_w4(fd, (const long long[3]){0, 0, 0}, (const int[3]){D, N, N}, H, 9.6, a, W4D, cy, cx, NZ + 1), "save the fields");
    for (int k = 0; k < 14; k++) free(a[k]);
    char good[300], bad[300]; snprintf(good, sizeof good, "%s/good", root); snprintf(bad, sizeof bad, "%s/bad", root);
    CHECK(!save_model(good, 0), "the right model"); CHECK(!save_model(bad, 1.5), "the wrong model");
    /* ---- fibres: H, along W = k + 0.3, theta from -pi / 2 to pi / 2, at z 24 and 40 (level-0 points: native x 4) ---- */
    {
        fib_set F = {0}; const int nr = 6, np = 181; F.n = nr; F.npts = nr * np; F.off = malloc((nr + 1) * sizeof(int)); F.fam = malloc(nr); F.pts = malloc((size_t)F.npts * 3 * sizeof(float));
        for (int r = 0; r < nr; r++) {
            const double W = 3 + r % 3 + 0.3, z = r < 3 ? 24 : 40; F.off[r] = r * np; F.fam[r] = 'H';
            for (int k = 0; k < np; k++) { const double th = -M_PI / 2 + M_PI * k / (np - 1), rr = 20 + 10 * (W + th / (2 * M_PI)); float *p = F.pts + 3 * (size_t)(r * np + k);
                p[0] = (float)(4 * z); p[1] = (float)(4 * (C0 + rr * sin(th))); p[2] = (float)(4 * (C0 + rr * cos(th))); }
        }
        F.off[nr] = nr * np; char fp[300]; snprintf(fp, sizeof fp, "%s/f.fib", root); CHECK(!fib_save(fp, &F), "save the fibres"); fib_free(&F);
    }
    char fib[300], ob[300], og[300], tsv[400], ot[300]; snprintf(fib, sizeof fib, "%s/f.fib", root);
    snprintf(og, sizeof og, "%s/og", root); snprintf(ob, sizeof ob, "%s/ob", root); snprintf(ot, sizeof ot, "%s/ot", root);
    mkdir(og, 0755); mkdir(ob, 0755); mkdir(ot, 0755);
    int warn = 0;
    /* ---- the right model ---- */
    {
        char *av[] = {"ufsm", "unroll-qa", "--fields", fd, "--model", good, "--fibres", fib, "--box", "0,0,0,64,256,256", "--name", "B", "--out", og, "--threads", "2", "--strict"};
        const int rc = cmd_unroll_qa(17, av); warn = warns(og);
        snprintf(tsv, sizeof tsv, "%s/boxes.tsv", og);
        CHECK(rc == 0 && warn == 0, "right model: exit %d, %d WARN lines (expected 0, 0)", rc, warn);
        const double fp = col(tsv, "fib_pers_h"), arc = col(tsv, "fib_arc_m");
        CHECK(fp == 0 && arc > 0.01, "right model: fibre H persistent %.1f /m over %.3f m (expected 0)", fp, arc);
        for (const char *const *c = (const char *const[]){"w4_y", "w4_x", "w4_z", nullptr}; *c; c++) CHECK(col(tsv, *c) > 0.99, "right model: %s %.4f (expected > 0.99)", *c, col(tsv, *c));
        CHECK(col(tsv, "w4_loop") < 0.01, "right model: wind4 loop closure misses %.4f", col(tsv, "w4_loop"));
        CHECK(col(tsv, "tilt_med") < 3, "right model: tilt median %.1f deg", col(tsv, "tilt_med"));
        CHECK(col(tsv, "dr_out") < 0.02, "right model: density outliers %.4f", col(tsv, "dr_out"));
        CHECK(col(tsv, "conf_lo") == 0 && col(tsv, "lcoh_lo") == 0, "right model: low confidence %.4f, low coherence blocks %.4f", col(tsv, "conf_lo"), col(tsv, "lcoh_lo"));
        CHECK(col(tsv, "pw16") < 0.01 && col(tsv, "pw48") < 0.01, "right model: phase-unwrap misses %.4f / %.4f", col(tsv, "pw16"), col(tsv, "pw48"));
    }
    /* ---- the wrong model: warn by default, exit 3 under --strict ---- */
    {
        char *av[] = {"ufsm", "unroll-qa", "--fields", fd, "--model", bad, "--fibres", fib, "--box", "0,0,0,64,256,256", "--name", "B", "--out", ob, "--threads", "2", "--strict"};
        int rc = cmd_unroll_qa(16, av); warn = warns(ob);
        snprintf(tsv, sizeof tsv, "%s/boxes.tsv", ob);
        CHECK(rc == 0 && warn >= 1, "wrong model, warn only: exit %d, %d WARN lines (expected 0, >= 1)", rc, warn);
        const double fp = col(tsv, "fib_pers_h");
        CHECK(fp > 300, "wrong model: fibre H persistent %.1f /m (expected > 300)", fp);
        char gt[400]; snprintf(gt, sizeof gt, "%s/boxes.tsv", og);
        CHECK(col(tsv, "tilt_med") > col(gt, "tilt_med") + 1, "wrong model: tilt median %.1f deg (right model %.1f)", col(tsv, "tilt_med"), col(gt, "tilt_med"));
        rc = cmd_unroll_qa(17, av); warn = warns(ob);
        CHECK(rc == 3 && warn >= 1, "wrong model, --strict: exit %d (expected 3)", rc);
    }
    /* ---- tiled mode ---- */
    {
        char *av[] = {"ufsm", "unroll-qa", "--fields", fd, "--model", good, "--tile", "32", "--tiles", "8", "--out", ot, "--threads", "2"};
        const int rc = cmd_unroll_qa(14, av);
        snprintf(tsv, sizeof tsv, "%s/boxes.tsv", ot); FILE *f = fopen(tsv, "r"); int rows = -1; char l[4096]; while (f && fgets(l, sizeof l, f)) rows++; if (f) fclose(f);
        CHECK(rc == 0 && rows == 8, "tiled: exit %d, %d tiles (expected 8)", rc, rows);
        CHECK(col(tsv, "w4_x") > 0.99, "tiled: first tile wind4 x %.4f", col(tsv, "w4_x"));
    }
    if (!fails) { char cmd[400]; snprintf(cmd, sizeof cmd, "rm -rf -- '%s'", root); if (!strncmp(root, "/tmp/ufsm-unrollqa-", 19) && system(cmd)) {} }
    else fprintf(stderr, "(kept %s)\n", root);
    printf("%s\n", fails ? "test_unroll_qa: FAILED" : "test_unroll_qa: ok");
    return fails != 0;
}
