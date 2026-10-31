/* ufsm count-field (src/countfield.c) on a synthetic spiral: W = r / p - theta / 2 pi, radial normals, the true density; the phase
   skips every other sheet in one quadrant (it rotates at half the rate there). W from the density must count every sheet there,
   the output phase (R 0) must follow frac(W) where the phase is right, and --model must reproduce W (fit_W). */
#include "fit.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int cmd_count_field(int argc, char **argv);
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

enum { D = 6, H = 96, W = 96, h = 2 };
static const double p = 8.0, cy0 = 48 * h, cx0 = 48 * h;   /* pitch (native voxels), axis at the box centre */
static int bad_quadrant(double th) { return th > 0.2 && th < M_PI / 2 - 0.2; }
static double trueW(double y, double x) { return hypot(y - cy0, x - cx0) / p - atan2(y - cy0, x - cx0) / (2 * M_PI); }

/* --cf-gpu 1 (src/nn_cf.cu) vs the CPU on the cf-ref recipe (--model-only, R 0, --phase-w 4, strided chunks): a 3D spiral with
   z-varying normals and phase noise. Pooled sums (--pool-cache) equal to ~1e-4 relative (the GPU sheet normal is fp32); the model's delta (U + gauge) agrees
   to << 1e-3 turn, like the CPU at another thread count (the CG sums in another order) */
int cf_gpu_ok(void) __attribute__((weak));
static float *read_f32(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb"); if (!f) return nullptr; fseek(f, 0, SEEK_END); const long sz = ftell(f); fseek(f, 0, SEEK_SET);
    float *v = malloc(sz); *n = (size_t)sz / 4; if (fread(v, 1, sz, f) != (size_t)sz) { free(v); v = nullptr; } fclose(f); return v;
}
static void gpu_vs_cpu(const char *dir) {
    if (!cf_gpu_ok || !cf_gpu_ok()) { printf("test_countfield: no GPU, CPU only\n"); return; }
    enum { D2 = 40, H2 = 88, W2 = 104 };
    const size_t n = (size_t)D2 * H2 * W2; uint8_t *a[10]; for (int k = 0; k < 10; k++) a[k] = calloc(n, 1);
    unsigned rs = 7;
    for (int z = 0; z < D2; z++) for (int y = 0; y < H2; y++) for (int x = 0; x < W2; x++) {
        const size_t i = ((size_t)z * H2 + y) * W2 + x; const double cy = 44 * h + 6 * sin(z * 0.1), cx = 50 * h + 4 * cos(z * 0.07);
        const double py = (y + 0.5) * h, px = (x + 0.5) * h, ry = py - cy, rx = px - cx, r = hypot(ry, rx); if (r < 2 * p) continue;
        rs = rs * 1664525u + 1013904223u; const double nse = ((rs >> 8) / 16777216.0 - 0.5) * 0.3;
        const double Wt = r / p - atan2(ry, rx) / (2 * M_PI) + 0.02 * z, ph = Wt + nse, cf = 0.4 + 0.55 * fabs(sin(0.3 * x + 0.2 * z));
        a[0][i] = (uint8_t)lrint(128 + 127 * cf * cos(2 * M_PI * ph)); a[1][i] = (uint8_t)lrint(128 + 127 * cf * sin(2 * M_PI * ph));
        const double t = 0.3 * sin(0.05 * z), nz = sin(t), ny = cos(t) * ry / r, nx = cos(t) * rx / r;
        a[2][i] = (uint8_t)lrint(255 * nz * nz); a[3][i] = (uint8_t)lrint(255 * ny * ny); a[4][i] = (uint8_t)lrint(255 * nx * nx);
        a[5][i] = (uint8_t)lrint(255 * (0.5 + nz * ny)); a[6][i] = (uint8_t)lrint(255 * (0.5 + nz * nx)); a[7][i] = (uint8_t)lrint(255 * (0.5 + ny * nx));
        a[8][i] = (uint8_t)lrint(8 * p / h * (1 + 0.2 * sin(0.1 * y))); a[9][i] = (uint8_t)(150 + (rs >> 26));
    }
    const int na = D2 * h + 1; float *cyv = malloc(na * sizeof(float)), *cxv = malloc(na * sizeof(float));
    for (int k = 0; k < na; k++) { cyv[k] = (float)(44 * h + 6 * sin(k / (double)h * 0.1)); cxv[k] = (float)(50 * h + 4 * cos(k / (double)h * 0.07)); }
    char in[600]; snprintf(in, sizeof in, "%s/in3", dir);
    const long long org[3] = {0, 0, 0}; const int shp[3] = {D2, H2, W2};
    if (fit_fields_save(in, org, shp, h, 9.6, a, cyv, cxv, na)) { fails++; fprintf(stderr, "FAIL: cannot write the 3D input\n"); return; }
    /* 8-row chunks (--mem-gb: 12 B per cell held), every 2nd chunk for the model gauge */
    char mg[32]; snprintf(mg, sizeof mg, "%.9f", 8.5 * 12.0 * H2 * W2 / 1e9);
    static const char *tag[4] = {"cpu4", "cpu1", "gpu", "gpu-lowmem"}; static const char *thr[4] = {"4", "1", "4", "2"}, *gp[4] = {"0", "0", "1", "1"};
    float *dl[4] = {0}, *pcache[4] = {0}; size_t nd[4] = {0}, npc[4] = {0};
    for (int r = 0; r < 4; r++) {
        char mdir[640], pcf[640], pth[700]; snprintf(mdir, sizeof mdir, "%s/m-%s", dir, tag[r]); snprintf(pcf, sizeof pcf, "%s/pool-%s.bin", dir, tag[r]);
        char *av[] = {"ufsm", "count-field", in, "-", "--R", "0", "--pool", "4", "--phase-w", "4", "--iters", "1200", "--model", mdir, "--model-only", "1",
                      "--model-stride", "2", "--threads", (char *)thr[r], "--mem-gb", mg, "--cf-gpu", (char *)gp[r], "--pool-cache", pcf, "--rmin-um", "50", nullptr};
        if (r == 3) { setenv("UFSM_CF_POOL_GZ", "1", 1); setenv("UFSM_CF_CG_NOAP", "1", 1); }   /* (one solve row per upload group: double buffers; the CG without a stored A p) */
        CHECK(cmd_count_field(28, av) == 0, "count-field %s failed", tag[r]);
        unsetenv("UFSM_CF_POOL_GZ"); unsetenv("UFSM_CF_CG_NOAP");
        snprintf(pth, sizeof pth, "%s/delta.f32", mdir); dl[r] = read_f32(pth, &nd[r]); pcache[r] = read_f32(pcf, &npc[r]);
        CHECK(dl[r] && pcache[r], "%s: no model / pool cache", tag[r]);
    }
    if (dl[0] && dl[1] && dl[2] && pcache[0] && pcache[2] && nd[0] == nd[2] && nd[1] == nd[0] && npc[0] == npc[2]) {
        size_t pdiff = 0; double prel = 0;
        for (size_t i = 8; i < npc[0]; i++) if (pcache[0][i] != pcache[2][i]) { pdiff++; prel = fmax(prel, fabs(pcache[0][i] - pcache[2][i]) / fmax(1e-6, fabs(pcache[0][i]))); }
        double dg = 0, dt = 0; size_t ig = 0;
        for (size_t i = 0; i < nd[0]; i++) { dg = fmax(dg, fabs(dl[2][i] - dl[0][i])); dt = fmax(dt, fabs(dl[1][i] - dl[0][i])); ig += lround(dl[2][i]) != lround(dl[0][i]); }
        printf("test_countfield: GPU vs CPU: pooled sums %zu of %zu differ (max rel %.1e); model delta max |GPU - CPU| %.2e turn (CPU 1 vs 4 threads %.2e), integers differ at %zu of %zu lattice points\n",
               pdiff, npc[0] - 8, prel, dg, dt, ig, nd[0]);
        CHECK(prel < 1e-3, "pooled sums: %zu differ, max rel %.2e", pdiff, prel);   /* (the GPU's sheet normal is fp32: last-bit sums, rel ~1e-4 near zero) */
        CHECK(dg < 1e-3, "model delta GPU vs CPU max %.3e turn", dg);
        CHECK(dl[3] && nd[3] == nd[2] && !memcmp(dl[2], dl[3], nd[2] * sizeof(float)) && pcache[3] && !memcmp(pcache[2] + 8, pcache[3] + 8, (npc[2] - 8) * sizeof(float)),
              "GPU low-memory paths (upload groups, CG without A p) not identical to the default GPU run");
    } else CHECK(0, "model / pool cache sizes differ");
    for (int r = 0; r < 4; r++) { free(dl[r]); free(pcache[r]); }
    for (int k = 0; k < 10; k++) free(a[k]);
    free(cyv); free(cxv);
}

int main(void) {
    char tmpl[] = "/tmp/test_countfield_XXXXXX"; char *dir = mkdtemp(tmpl); if (!dir) return 1;
    char in[512], out[512], mdir[512]; snprintf(in, sizeof in, "%s/in", dir); snprintf(out, sizeof out, "%s/out", dir); snprintf(mdir, sizeof mdir, "%s/model", dir);
    const size_t n = (size_t)D * H * W; uint8_t *a[10]; for (int k = 0; k < 10; k++) a[k] = calloc(n, 1);
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const size_t i = ((size_t)z * H + y) * W + x; const double py = (y + 0.5) * h, px = (x + 0.5) * h, ry = py - cy0, rx = px - cx0, r = hypot(ry, rx);
        if (r < 3 * p) continue;
        const double th = atan2(ry, rx), Wt = trueW(py, px), ph = bad_quadrant(th) ? Wt / 2 : Wt;   /* the phase skips sheets in one quadrant */
        a[0][i] = (uint8_t)lrint(128 + 127 * 0.9 * cos(2 * M_PI * ph)); a[1][i] = (uint8_t)lrint(128 + 127 * 0.9 * sin(2 * M_PI * ph));
        const double nz = 0, ny = ry / r, nx = rx / r;   /* n n^T planes: n_z^2 n_y^2 n_x^2, 0.5 + n_z n_y, 0.5 + n_z n_x, 0.5 + n_y n_x */
        a[2][i] = (uint8_t)lrint(255 * nz * nz); a[3][i] = (uint8_t)lrint(255 * ny * ny); a[4][i] = (uint8_t)lrint(255 * nx * nx);
        a[5][i] = (uint8_t)lrint(255 * (0.5 + nz * ny)); a[6][i] = (uint8_t)lrint(255 * (0.5 + nz * nx)); a[7][i] = (uint8_t)lrint(255 * (0.5 + ny * nx));
        a[8][i] = (uint8_t)lrint(8 * p / h); a[9][i] = 255;   /* sp = 8 spacing / h; material */
    }
    const int na = D * h + 1; float *cy = malloc(na * sizeof(float)), *cx = malloc(na * sizeof(float)); for (int k = 0; k < na; k++) { cy[k] = (float)cy0; cx[k] = (float)cx0; }
    const long long org[3] = {0, 0, 0}; const int shp[3] = {D, H, W};
    if (fit_fields_save(in, org, shp, h, 9.6, a, cy, cx, na)) { fprintf(stderr, "cannot write the input\n"); return 1; }
    char *av[] = {"ufsm", "count-field", in, out, "--R", "0", "--dir", "1", "--pool", "1", "--threads", "4", "--model", mdir, "--model-R", "4", "--rmin-um", "0", nullptr};
    CHECK(cmd_count_field(18, av) == 0, "count-field failed");
    char pth[600]; snprintf(pth, sizeof pth, "%s/wc.f32", out); FILE *f = fopen(pth, "rb"); float *wc = malloc(n * sizeof(float));
    CHECK(f && fread(wc, sizeof(float), n, f) == n, "no wc.f32"); if (f) fclose(f);
    fit_fields_buf ob; CHECK(fit_fields_load(out, 0, 0, 1, &ob) == 0, "cannot load the output");
    /* counts along radial segments (r 12 .. 36 cells) in the bad quadrant: W_out vs truth, and the input phase's count */
    double emax = 0, ph_err = 0, gerr = 0; int ng = 0;
    for (int k = 0; k < 8; k++) {
        const double th = 0.3 + k * (M_PI / 2 - 0.6) / 7, r0 = 12 * h, r1 = 36 * h;
        const int y0 = (int)((cy0 + r0 * sin(th)) / h), x0 = (int)((cx0 + r0 * cos(th)) / h), y1 = (int)((cy0 + r1 * sin(th)) / h), x1 = (int)((cx0 + r1 * cos(th)) / h);
        const size_t i0 = ((size_t)2 * H + y0) * W + x0, i1 = ((size_t)2 * H + y1) * W + x1;
        const double tw = trueW((y1 + 0.5) * h, (x1 + 0.5) * h) - trueW((y0 + 0.5) * h, (x0 + 0.5) * h), cw = wc[i1] - wc[i0];
        if (fabs(cw - tw) > emax) emax = fabs(cw - tw);
        ph_err = fmax(ph_err, fabs(tw / 2 - tw));
    }
    CHECK(emax < 0.3, "bad-quadrant count: W off by %.3f turns over 24 cells (the phase alone by %.2f)", emax, ph_err);
    /* the good quadrants: the output phase = the input phase (= frac W) up to the gauge, little error */
    for (int y = 0; y < H; y += 3) for (int x = 0; x < W; x += 3) {
        const size_t i = ((size_t)3 * H + y) * W + x; const double py = (y + 0.5) * h, px = (x + 0.5) * h, r = hypot(py - cy0, px - cx0), th = atan2(py - cy0, px - cx0);
        if (r < 6 * p || r > 44 * h || th > -0.2) continue;   /* lower half: phase right */
        const double po = atan2((ob.f.ps[i] - 128) / 127.0, (ob.f.pc[i] - 128) / 127.0) / (2 * M_PI); double e = po - trueW(py, px); e -= round(e);
        gerr += fabs(e); ng++;
    }
    CHECK(ng > 100 && gerr / ng < 0.08, "output phase vs frac W where the phase is right: mean |error| %.3f turns (%d cells)", ng ? gerr / ng : 1, ng);
    /* the model: fit_W reproduces W_out's differences */
    fit_model *m = fit_load(mdir); CHECK(m != nullptr, "no model");
    if (m) {
        double me = 0; int nm = 0;
        for (int k = 0; k < 200; k++) {
            const double th = -2.9 + 2.6 * k / 200, r = (14 + (k % 20)) * h, py = cy0 + r * sin(th), px = cx0 + r * cos(th);
            const int y = (int)(py / h), x = (int)(px / h); const size_t i = ((size_t)3 * H + y) * W + x; if (!(wc[i] == wc[i])) continue;
            const double d = fit_W(m, 3.5 * h, (y + 0.5) * h, (x + 0.5) * h) - wc[i]; me += fabs(d - round(d)); nm++;
        }
        CHECK(nm > 100 && me / nm < 0.15, "model W vs wc: mean |frac error| %.3f (%d points)", nm ? me / nm : 1, nm);
        fit_free(m);
    }
    fit_fields_free(&ob);
    gpu_vs_cpu(dir);
    char cmd[700]; snprintf(cmd, sizeof cmd, "rm -rf '%s'", dir); if (system(cmd)) {}
    for (int k = 0; k < 10; k++) free(a[k]);
    free(cy); free(cx); free(wc);
    if (fails) { fprintf(stderr, "test_countfield: %d failures\n", fails); return 1; }
    printf("test_countfield: OK (bad-quadrant count error %.3f turns vs the phase's %.2f)\n", emax, ph_err);
    return 0;
}
