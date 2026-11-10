/* ufsm wrap-dump, fit-phase-box and fiber-judge on a synthetic Archimedean spiral r = 20 + 10 W about a vertical axis at (128, 128),
   W = U - theta / 2 pi (dir +1), wraps k = 1..7 = the level sets W = k written as .sfc (rows z = 1, 3, .. (a decoded z below 0 marks an invalid vertex), columns from theta = +pi down,
   the layout of ufsm wraps), and the same spiral as a hand-made fit model (pitch 10, offset -2, no lattice).
   - wrap-dump: every vertex of the slab comes back (z, y, x, id, column / width) within the .sfc tolerance; --z / --yx filter;
   - fit-phase-box: the raster equals floor(252 frac(W)) of the analytic W at the documented positions (level 0 step 2; level 1 step 1);
   - fiber-judge, UFIB fibres: an H fibre along a sheet (W = 3.3), one along a sheet across the theta cut (W = 4.3), a V fibre along z:
     no excursion and every pair within half a turn; a radial line through 3.4 sheets (family U): exactly 3 excursions, all persistent,
     and pairs 16 voxels apart (1.6 turns) all beyond half a turn - on the wraps basis and on the fit-set basis alike. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "afv.h"
#include "fit.h"
#include "tiffw.h"

int cmd_wrap_dump(int argc, char **argv); int cmd_fit_phase_box(int argc, char **argv); int cmd_fiber_judge(int argc, char **argv);
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)
#define C0 128.0
enum { NZ = 64, NYX = 256, HR = 32, WC = 720 };
static double rk(double W) { return 20 + 10 * W; }
static double W_true(double y, double x) { return (hypot(y - C0, x - C0) - 20) / 10 - atan2(y - C0, x - C0) / (2 * M_PI); }

/* the judge's TSV row: family f (0 H, 1 V, 2 U) -> arc, exc, pers, pairs agreeing / pairs at the two distances */
static int row(const char *path, int f, double *arc, double *exc, double *pers, double pa[2], double pn[2]) {
    FILE *in = fopen(path, "r"); if (!in) return -1; char l[4096]; int ok = -1;
    if (fgets(l, sizeof l, in) && fgets(l, sizeof l, in)) {
        char *t = strtok(l, "\t"); for (int k = 0; k < 4 + 5 * f && t; k++) t = strtok(nullptr, "\t");
        if (t) { *arc = atof(t); t = strtok(nullptr, "\t"); } if (t) { *exc = atof(t); t = strtok(nullptr, "\t"); } if (t) { *pers = atof(t); t = strtok(nullptr, "\t"); }
        for (int d = 0; d < 2 && t; d++) { ok = sscanf(t, "%lf:%lf", &pa[d], &pn[d]) == 2 ? 0 : -1; t = strtok(nullptr, "\t\n"); }
    }
    fclose(in); return ok;
}

int main(void) {
    char root[] = "/tmp/ufsm-fiberjudge-XXXXXX"; if (!mkdtemp(root)) { fprintf(stderr, "FAIL: tmp dir\n"); return 1; }
    char wd[300], p[600]; snprintf(wd, sizeof wd, "%s/wraps", root); snprintf(p, sizeof p, "mkdir -p %s", wd); if (system(p)) return 1;
    tifxyz_set_sfc(0.01);
    for (int k = 1; k <= 7; k++) {
        float *x = malloc(WC * HR * 4), *y = malloc(WC * HR * 4), *z = malloc(WC * HR * 4);
        for (int r = 0; r < HR; r++) for (int c = 0; c < WC; c++) { const double th = M_PI - (c + 0.5) * 2 * M_PI / WC, rr = rk(k + th / (2 * M_PI));
            x[r * WC + c] = (float)(C0 + rr * cos(th)); y[r * WC + c] = (float)(C0 + rr * sin(th)); z[r * WC + c] = (float)(2 * r + 1); }
        snprintf(p, sizeof p, "%s/w%03d", wd, k); CHECK(!tifxyz_write(p, WC, HR, x, y, z, 2, 9.6, k, "test"), "write wrap %d", k);
        free(x); free(y); free(z);
    }
    tifxyz_set_sfc(0);
    /* ---- wrap-dump ---- */
    {
        char out[400]; snprintf(out, sizeof out, "%s/v.f32", root);
        char *av[] = {"ufsm", "wrap-dump", wd, out, "--z", "0,64", "--every", "1"}; CHECK(!cmd_wrap_dump(8, av), "wrap-dump");
        FILE *f = fopen(out, "rb"); CHECK(f != nullptr, "dump file"); if (!f) return 1;
        float *v = malloc(((size_t)7 * HR * WC + 1) * 20); const size_t n = fread(v, 20, (size_t)7 * HR * WC + 1, f); fclose(f);
        CHECK(n == (size_t)7 * HR * WC, "wrap-dump: %zu vertices, want %d", n, 7 * HR * WC);
        double worst = 0; int badid = 0;
        for (size_t i = 0; i < n; i++) { const float *g = v + 5 * i; const int k = (int)g[3], c = (int)lround(g[4] * WC); const double th = M_PI - (c + 0.5) * 2 * M_PI / WC, rr = rk(k + th / (2 * M_PI));
            const double e = hypot(g[1] - (C0 + rr * sin(th)), g[2] - (C0 + rr * cos(th))); if (e > worst) worst = e;
            badid += k < 1 || k > 7 || fabs((g[0] - 1) / 2 - round((g[0] - 1) / 2)) > 0.02 || g[0] < 0.98 || g[0] > 63.02; }
        CHECK(worst < 0.03 && !badid, "wrap-dump: vertex error %.3f voxels, %d bad ids / rows", worst, badid);
        char *aw[] = {"ufsm", "wrap-dump", wd, out, "--z", "10,20", "--yx", "128,400,0,400", "--every", "1"}; CHECK(!cmd_wrap_dump(10, aw), "wrap-dump window");
        f = fopen(out, "rb"); const size_t m = f ? fread(v, 20, (size_t)7 * HR * WC, f) : 0; if (f) fclose(f); int bad = 0;
        for (size_t i = 0; i < m; i++) bad += v[5 * i] < 10 || v[5 * i] >= 20 || v[5 * i + 1] < 128;
        CHECK(m > 1000 && m < n / 8 && !bad, "wrap-dump --z / --yx: %zu vertices, %d outside", m, bad);
        free(v);
    }
    /* ---- the same spiral as a fit model ---- */
    char md[400]; snprintf(md, sizeof md, "%s/model", root);
    {
        fit_model m = {0}; m.dir = 1; m.nk = 2; m.knot_dz = 64; float pit[2] = {10, 10}, off[2] = {-2, -2}; m.pitch = pit; m.offset = off;
        m.L = 1 << 20; m.lz = m.ly = m.lx = 2; float delta[8] = {0}; m.delta = delta; m.D = NZ / 2; m.H = m.W = NYX / 2; m.h = 2; m.um = 9.6;
        float cy[NZ + 1], cx[NZ + 1]; for (int i = 0; i <= NZ; i++) cy[i] = cx[i] = (float)C0; m.cy = cy; m.cx = cx;
        CHECK(fabs(fit_W(&m, 30, 150, 190) - W_true(150, 190)) < 1e-4, "the hand-made model is the spiral (W %.4f vs %.4f)", fit_W(&m, 30, 150, 190), W_true(150, 190));
        CHECK(!fit_save(&m, md), "save the model");
    }
    /* ---- fit-phase-box ---- */
    for (int lv = 0; lv < 2; lv++) {
        char out[400]; snprintf(out, sizeof out, "%s/ph%d.raw", root, lv); const int S = lv ? 1 : 2, n0 = NZ / 2, n1 = NYX / 2;
        char *av[] = {"ufsm", "fit-phase-box", md, "0", "0", "0", lv ? "32" : "64", lv ? "128" : "256", lv ? "128" : "256", out, "--level", lv ? "1" : "0", "--step", lv ? "1" : "2"};
        CHECK(!cmd_fit_phase_box(14, av), "fit-phase-box level %d", lv);
        FILE *f = fopen(out, "rb"); uint8_t *b = malloc((size_t)n0 * n1 * n1); const size_t n = f ? fread(b, 1, (size_t)n0 * n1 * n1, f) : 0; if (f) fclose(f);
        CHECK(n == (size_t)n0 * n1 * n1, "fit-phase-box level %d: %zu bytes", lv, n); long bad = 0, tested = 0;
        for (int j = 0; j < n1; j++) for (int k = 0; k < n1; k++) {
            /* level coordinate of cell (j, k): (j + 0.5) S - 0.5; native = (c + 0.5) 2^lv - 0.5 */
            const double cyv = ((j + 0.5) * S - 0.5 + 0.5) * (1 << lv) - 0.5, cxv = ((k + 0.5) * S - 0.5 + 0.5) * (1 << lv) - 0.5;
            if (hypot(cyv - C0, cxv - C0) < 12) continue;
            const double w = W_true(cyv, cxv); const int want = (int)floor((w - floor(w)) * 252), got = b[((size_t)(n0 / 2) * n1 + j) * n1 + k];
            int d = abs(got - want); if (d > 126) d = 252 - d; bad += got > 251 || d > 1; tested++;
        }
        CHECK(!bad && tested > 15000, "fit-phase-box level %d: %ld of %ld values off the analytic phase", lv, bad, tested); free(b);
    }
    /* ---- fibres (UFIB: level-0 z, y, x) ---- */
    char fb[400]; snprintf(fb, sizeof fb, "%s/f.fib", root);
    {
        enum { NP = 400 }; fib_set F = {0}; F.n = 4; F.npts = 4 * NP; int off[5]; char fam[5] = {'H', 'H', 'V', '?', 0}; float *pt = malloc(sizeof(float) * 3 * 4 * NP); F.off = off; F.fam = fam; F.pts = pt;
        for (int r = 0; r <= 4; r++) off[r] = r * NP;
        for (int i = 0; i < NP; i++) { const double t = i / (double)(NP - 1); double th, rr; float *q;
            th = -2.5 + 5 * t; rr = rk(3.3 + th / (2 * M_PI)); q = pt + 3 * i; q[0] = 30; q[1] = (float)(C0 + rr * sin(th)); q[2] = (float)(C0 + rr * cos(th));             /* H in a sheet */
            th = 2.6 + 1.2 * t; rr = rk(4.3 + th / (2 * M_PI)); q = pt + 3 * (NP + i); q[0] = 20; q[1] = (float)(C0 + rr * sin(th)); q[2] = (float)(C0 + rr * cos(th));   /* H across the cut */
            th = 1.0; rr = rk(2.4 + th / (2 * M_PI)); q = pt + 3 * (2 * NP + i); q[0] = (float)(4 + 54 * t); q[1] = (float)(C0 + rr * sin(th)); q[2] = (float)(C0 + rr * cos(th));   /* V */
            th = 0.5; rr = 43 + 34 * t; q = pt + 3 * (3 * NP + i); q[0] = 30; q[1] = (float)(C0 + rr * sin(th)); q[2] = (float)(C0 + rr * cos(th)); }                      /* radial: 3.4 sheets */
        CHECK(!fib_save(fb, &F), "write the fibres"); free(pt);
    }
    for (int basis = 0; basis < 2; basis++) {
        char out[400]; snprintf(out, sizeof out, "%s/j%d.tsv", root, basis);
        char *av[] = {"ufsm", "fiber-judge", fb, basis ? "--set" : "--wraps", basis ? md : wd, "--um", "9.6", "--slab", "0,64", "--region", "0,256,0,256", "--tile", "256", "--d-vox", "16,32", "--out", out};
        CHECK(!cmd_fiber_judge(17, av), "fiber-judge basis %d", basis);
        double arc, exc, pers, pa[2], pn[2];
        CHECK(!row(out, 0, &arc, &exc, &pers, pa, pn) && arc * 1e6 / 9.6 > 250 && exc == 0 && pers == 0 && pn[0] > 40 && pa[0] == pn[0] && pa[1] == pn[1], "basis %d, H fibres in a sheet: arc %.0f voxels, excursions %.0f, pairs %.0f / %.0f", basis, arc * 1e6 / 9.6, exc, pa[0], pn[0]);
        CHECK(!row(out, 1, &arc, &exc, &pers, pa, pn) && arc * 1e6 / 9.6 > 45 && exc == 0 && pa[0] == pn[0] && pn[0] > 2, "basis %d, V fibre: arc %.0f voxels, excursions %.0f, pairs %.0f / %.0f", basis, arc * 1e6 / 9.6, exc, pa[0], pn[0]);
        CHECK(!row(out, 2, &arc, &exc, &pers, pa, pn) && exc == 3 && pers == 3 && pn[0] > 2 && pa[0] == 0, "basis %d, radial line: excursions %.0f, persistent %.0f (want 3, 3), pairs at 16 voxels agreeing %.0f of %.0f", basis, exc, pers, pa[0], pn[0]);
    }
    snprintf(p, sizeof p, "rm -rf %s", root); if (!fails && system(p)) fails++;
    printf(fails ? "test_fiberjudge: %d FAILED\n" : "test_fiberjudge: ok\n", fails);
    return fails ? 1 : 0;
}
