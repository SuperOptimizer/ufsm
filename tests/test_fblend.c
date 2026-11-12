/* ufsm field-blend (src/fblend.h): the per-cell rules (phase as the confidence-weighted complex mean, n n^T / recto means, the
   known-spacing median, material max, winding mod 4 complex means) and whole field sets (a raw set and a zarr set cropped from a
   larger array by zarr_offset; output in the first input's layout, unknown planes passed through, loadable by fit_fields_load). */
#include "fblend.h"
#include "fit.h"
#include "z3w.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static void enc(double conf, double phi, uint8_t *pc, uint8_t *ps) { *pc = (uint8_t)lrint(128 + 127 * conf * cos(2 * M_PI * phi)); *ps = (uint8_t)lrint(128 + 127 * conf * sin(2 * M_PI * phi)); }
static void dec(uint8_t pc, uint8_t ps, double *conf, double *phi) {
    const double x = (pc - 128) / 127.0, y = (ps - 128) / 127.0; *conf = hypot(x, y); *phi = atan2(y, x) / (2 * M_PI); if (*phi < 0) *phi += 1;
}

static void test_cells(void) {
    const float w1[3] = {1, 1, 1}, w31[2] = {3, 1};
    uint8_t c[3], s[3], oc, os; double cf, ph;
    /* identical inputs: unchanged (up to the uint8 rounding) */
    enc(0.8, 0.3, &c[0], &s[0]); c[1] = c[0]; s[1] = s[0];
    { const uint8_t *pc[2] = {&c[0], &c[1]}, *ps[2] = {&s[0], &s[1]}; fb_phase(2, pc, ps, w1, 1, &oc, &os); }
    CHECK(oc == c[0] && os == s[0], "identical phase %d %d -> %d %d", c[0], s[0], oc, os);
    /* opposite phases of equal confidence cancel: confidence 0 */
    enc(0.9, 0.1, &c[0], &s[0]); enc(0.9, 0.6, &c[1], &s[1]);
    { const uint8_t *pc[2] = {&c[0], &c[1]}, *ps[2] = {&s[0], &s[1]}; fb_phase(2, pc, ps, w1, 1, &oc, &os); }
    dec(oc, os, &cf, &ph); CHECK(cf < 0.02, "opposite phases: confidence %g", cf);
    /* a quarter turn apart: the mean angle, length conf cos(45 deg) */
    enc(1.0, 0.0, &c[0], &s[0]); enc(1.0, 0.25, &c[1], &s[1]);
    { const uint8_t *pc[2] = {&c[0], &c[1]}, *ps[2] = {&s[0], &s[1]}; fb_phase(2, pc, ps, w1, 1, &oc, &os); }
    dec(oc, os, &cf, &ph); CHECK(fabs(ph - 0.125) < 0.005 && fabs(cf - M_SQRT1_2) < 0.01, "quarter turn: phase %g conf %g", ph, cf);
    /* the confident model dominates the direction */
    enc(1.0, 0.0, &c[0], &s[0]); enc(0.2, 0.25, &c[1], &s[1]);
    { const uint8_t *pc[2] = {&c[0], &c[1]}, *ps[2] = {&s[0], &s[1]}; fb_phase(2, pc, ps, w1, 1, &oc, &os); }
    dec(oc, os, &cf, &ph); CHECK(fabs(ph - atan2(0.2, 1.0) / (2 * M_PI)) < 0.005 && fabs(cf - hypot(1.0, 0.2) / 2) < 0.01, "conf weighting: phase %g conf %g", ph, cf);
    /* weights 3 : 1, and an input without the plane does not count */
    enc(1.0, 0.0, &c[0], &s[0]); enc(1.0, 0.5, &c[1], &s[1]);
    { const uint8_t *pc[2] = {&c[0], &c[1]}, *ps[2] = {&s[0], &s[1]}; fb_phase(2, pc, ps, w31, 1, &oc, &os); }
    dec(oc, os, &cf, &ph); CHECK(fabs(cf - 0.5) < 0.01 && (ph < 0.005 || ph > 0.995), "weights: phase %g conf %g", ph, cf);
    { const uint8_t *pc[2] = {&c[0], nullptr}, *ps[2] = {&s[0], nullptr}; fb_phase(2, pc, ps, w1, 1, &oc, &os); }
    CHECK(oc == c[0] && os == s[0], "missing input: %d %d -> %d %d", c[0], s[0], oc, os);
    /* sp: the median of the known codes */
    const uint8_t a0[5] = {0, 40, 10, 0, 10}, a1[5] = {50, 0, 20, 0, 20}, a2[5] = {70, 0, 30, 0, 30};
    const uint8_t *a[3] = {a0, a1, a2}; uint8_t o[5];
    fb_median_known(3, a, w1, 5, o);
    CHECK(o[0] == 60 && o[1] == 40 && o[2] == 20 && o[3] == 0, "median %d %d %d %d", o[0], o[1], o[2], o[3]);
    const float w113[3] = {1, 1, 3}; fb_median_known(3, a, w113, 5, o); CHECK(o[4] == 30, "weighted median %d", o[4]);
    fb_mean(3, a, w1, 5, o); CHECK(o[0] == 40 && o[2] == 20, "mean %d %d", o[0], o[2]);
    fb_max(3, a, 5, o); CHECK(o[0] == 70 && o[1] == 40 && o[3] == 0, "max %d %d %d", o[0], o[1], o[3]);
    /* winding mod 4: identical inputs unchanged, a half turn apart cancel */
    uint8_t d0[4][2] = {{10, 0}, {200, 64}, {90, 0}, {255, 200}}, d1[4][2] = {{10, 128}, {200, 192}, {90, 128}, {255, 200}}, out4[4][2];
    const uint8_t *const dd[2][4] = {{d0[0], d0[1], d0[2], d0[3]}, {d1[0], d1[1], d1[2], d1[3]}};
    uint8_t *const o4[4] = {out4[0], out4[1], out4[2], out4[3]};
    fb_wind4(2, dd, w1, 2, o4);
    CHECK(out4[0][0] == 10 && out4[1][0] == 200 && out4[2][0] == 90 && out4[3][0] == 255, "wind4 same: %d %d %d dq %d", out4[0][0], out4[1][0], out4[2][0], out4[3][0]);
    CHECK(out4[3][1] < 3, "wind4 opposite: dq %d", out4[3][1]);
}

/* overlapping predicts: tile weights (two grids half an interior apart sum to 1, peak mid-interior) and the per-cell weighted rules */
static void test_tiles(void) {
    double mx = 0, mn = 1, worst = 0;
    for (int c = 0; c < 600; c++) {
        const double a = fb_tile_w(100, 2, c, 256, 0), b = fb_tile_w(100, 2, c, 256, 128);
        worst = fmax(worst, fabs(a + b - 1)); mx = fmax(mx, a); mn = fmin(mn, a);
    }
    CHECK(worst < 1e-5 && mx > 0.999 && mn < 1e-3, "tile weights: |sum - 1| %g, max %g, min %g", worst, mx, mn);
    /* interior [0, 256) on grid 0: the cell centred at voxel 128 is mid-interior, the one at voxel 1 next to the face */
    CHECK(fb_tile_w(0, 2, 63, 256, 0) > 0.999 && fb_tile_w(0, 2, 0, 256, 0) < 2e-4 && fb_tile_w(-8, 2, 4, 256, 0) < 2e-4, "tile weight placement");
    const float w1[2] = {1, 1}; uint8_t c[2], s[2], oc, os; double cf, ph;
    /* per-cell weights: input 0 weighs ~0 -> input 1 */
    enc(0.9, 0.1, &c[0], &s[0]); enc(0.6, 0.4, &c[1], &s[1]);
    { const float q0 = 1e-6f, q1 = 1.f; const float *cw[2] = {&q0, &q1}; const uint8_t *pc[2] = {&c[0], &c[1]}, *ps[2] = {&s[0], &s[1]};
      fb_phase_cw(2, pc, ps, w1, cw, FB_CONF_RESULTANT, 1, &oc, &os); }
    CHECK(abs(oc - c[1]) <= 1 && abs(os - s[1]) <= 1, "cell weights: %d %d vs input 1 %d %d", oc, os, c[1], s[1]);
    /* --conf mean: a quarter turn apart keeps the mean confidence (resultant: cos 45 deg) along the mean direction */
    enc(1.0, 0.0, &c[0], &s[0]); enc(0.6, 0.25, &c[1], &s[1]);
    { const uint8_t *pc[2] = {&c[0], &c[1]}, *ps[2] = {&s[0], &s[1]}; fb_phase_cw(2, pc, ps, w1, nullptr, FB_CONF_MEAN, 1, &oc, &os); }
    dec(oc, os, &cf, &ph); CHECK(fabs(cf - 0.8) < 0.01 && fabs(ph - atan2(0.6, 1.0) / (2 * M_PI)) < 0.005, "conf mean: conf %g phase %g", cf, ph);
    /* opposite: no direction, the length stays the mean (0.9) only when a direction exists; exact cancel -> 0 */
    enc(0.9, 0.1, &c[0], &s[0]); enc(0.9, 0.6, &c[1], &s[1]);
    { const uint8_t *pc[2] = {&c[0], &c[1]}, *ps[2] = {&s[0], &s[1]}; fb_phase_cw(2, pc, ps, w1, nullptr, FB_CONF_MEAN, 1, &oc, &os); }
    dec(oc, os, &cf, &ph); CHECK(cf < 0.02 || fabs(cf - 0.9) < 0.02, "conf mean opposite: %g", cf);
    /* the _cw rules equal the plain ones without cell weights */
    const uint8_t a0[4] = {0, 40, 10, 200}, a1[4] = {50, 0, 20, 100}; const uint8_t *a[2] = {a0, a1}; uint8_t o1[4], o2[4];
    fb_mean(2, a, w1, 4, o1); fb_mean_cw(2, a, w1, nullptr, 4, o2); CHECK(!memcmp(o1, o2, 4), "mean_cw");
    fb_median_known(2, a, w1, 4, o1); fb_median_known_cw(2, a, w1, nullptr, 4, o2); CHECK(!memcmp(o1, o2, 4), "median_cw");
    { const float q0[4] = {1, 1, 3, 1}, q1[4] = {1, 1, 1, 3}; const float *cw[2] = {q0, q1}; fb_mean_cw(2, a, w1, cw, 4, o2);
      CHECK(o2[2] == 12 && o2[3] == 125, "mean_cw weighted %d %d", o2[2], o2[3]); }
}

/* field sets: A raw (with an extra plane ct and the winding mod 4 planes), B zarr arrays two cells larger per axis read through
   zarr_offset [1, 2, 1] */
enum { D = 6, H = 5, W = 7 };
static uint8_t A[15][D * H * W], B[15][D * H * W], ct[D * H * W];
static void mkfields(const char *ra, const char *zb) {
    srand(7);
    for (int k = 0; k < 15; k++) for (int i = 0; i < D * H * W; i++) { A[k][i] = (uint8_t)(rand() % 256); B[k][i] = (uint8_t)(rand() % 256); }
    for (int i = 0; i < D * H * W; i++) { ct[i] = (uint8_t)(i * 7); if (i % 5 == 0) A[8][i] = 0; if (i % 3 == 0) B[8][i] = 0; }
    const long long org[3] = {100, 200, 300}; const int sh[3] = {D, H, W}; const int na = D * 2 + 1;
    float cy[13], cx[13]; for (int z = 0; z < na; z++) { cy[z] = 3.5f + 0.01f * z; cx[z] = -2.25f; }
    uint8_t *a14[14]; for (int k = 0; k < 14; k++) a14[k] = A[k];
    CHECK(fit_fields_save_ex(ra, org, sh, 2, 9.6, a14, 16, A[14], cy, cx, na) == 0, "save A");
    char p[512]; snprintf(p, sizeof p, "%s/ct.raw", ra); FILE *f = fopen(p, "wb"); fwrite(ct, 1, sizeof ct, f); fclose(f);
    static const char *nm[15] = {"pc", "ps", "t0", "t1", "t2", "t3", "t4", "t5", "sp", "w", "dz", "dy", "dx", "dq", "r"};
    mkdir(zb, 0755);
    const int64_t bs[3] = {D + 2, H + 2, W + 2}; uint8_t *sb = calloc(128 * 128 * 128, 1);
    for (int k = 0; k < 15; k++) {
        snprintf(p, sizeof p, "%s/%s", zb, nm[k]);
        z3w *z = z3w_create(p, bs, 128, 0, k < 2 ? 128 : 0, nullptr); CHECK(z != nullptr, "z3w %s", nm[k]);
        memset(sb, 0, 128 * 128 * 128);
        for (int a = 0; a < D; a++) for (int b = 0; b < H; b++) for (int c = 0; c < W; c++) sb[((size_t)(a + 1) * 128 + b + 2) * 128 + c + 1] = B[k][(a * H + b) * W + c];
        CHECK(z3w_write_shard(z, 0, 0, 0, sb, 1) == 0, "shard %s", nm[k]); z3w_close(z);
    }
    free(sb);
    snprintf(p, sizeof p, "%s/fields.json", zb); f = fopen(p, "w");
    fprintf(f, "{\"format\": \"zarr\", \"zarr_offset\": [1, 2, 1], \"origin\": [100, 200, 300], \"shape\": [%d, %d, %d], \"h\": 2, \"um\": 9.6, \"w4_d\": 16, \"recto\": true, \"axis\": [", D, H, W);
    for (int z = 0; z < na; z++) fprintf(f, "%s[%.3f, %.3f]", z ? ", " : "", cy[z], cx[z]);
    fprintf(f, "]}\n"); fclose(f);
}

static void test_sets(void) {
    char tmp[] = "/tmp/test_fblend_XXXXXX"; if (!mkdtemp(tmp)) { CHECK(0, "mkdtemp"); return; }
    char ra[256], zb[256], o1[256], o2[256], o3[256], p[512];
    snprintf(ra, sizeof ra, "%s/a", tmp); snprintf(zb, sizeof zb, "%s/b", tmp); snprintf(o1, sizeof o1, "%s/ab", tmp); snprintf(o2, sizeof o2, "%s/ba", tmp); snprintf(o3, sizeof o3, "%s/bad", tmp);
    mkfields(ra, zb);
    const char *in1[2] = {ra, zb}, *in2[2] = {zb, ra};
    CHECK(fblend_run(o1, 2, in1, nullptr, 2) == 0, "blend a b");
    CHECK(fblend_run(o2, 2, in2, nullptr, 2) == 0, "blend b a");
    const size_t n = D * H * W; const float w1[2] = {1, 1};
    uint8_t ex[15][D * H * W];
    { const uint8_t *pc[2] = {A[0], B[0]}, *ps[2] = {A[1], B[1]}; fb_phase(2, pc, ps, w1, n, ex[0], ex[1]); }
    for (int k = 2; k < 8; k++) { const uint8_t *s[2] = {A[k], B[k]}; fb_mean(2, s, w1, n, ex[k]); }
    { const uint8_t *s[2] = {A[8], B[8]}; fb_median_known(2, s, w1, n, ex[8]); }
    { const uint8_t *s[2] = {A[9], B[9]}; fb_max(2, s, n, ex[9]); }
    { const uint8_t *const d[2][4] = {{A[10], A[11], A[12], A[13]}, {B[10], B[11], B[12], B[13]}}; uint8_t *const o[4] = {ex[10], ex[11], ex[12], ex[13]}; fb_wind4(2, d, w1, n, o); }
    { const uint8_t *s[2] = {A[14], B[14]}; fb_mean(2, s, w1, n, ex[14]); }
    fit_fields_buf g1, g2;
    CHECK(fit_fields_load(o1, 0, 0, 1, &g1) == 0, "load raw blend");
    CHECK(fit_fields_load(o2, 0, 0, 1, &g2) == 0, "load zarr blend");
    for (int k = 0; k < 15; k++) {
        int bad1 = 0, bad2 = 0;
        for (size_t i = 0; i < n; i++) { if (!g1.a[k] || g1.a[k][i] != ex[k][i]) bad1++; if (!g2.a[k] || g2.a[k][i] != ex[k][i]) bad2++; }
        CHECK(!bad1 && !bad2, "plane %d: %d / %d cells differ from the per-cell rules (raw / zarr output)", k, bad1, bad2);
    }
    CHECK(g1.origin[0] == 100 && g1.origin[2] == 300 && g1.shape[2] == W && g1.h == 2 && fabs(g1.um - 9.6) < 1e-9, "geometry");
    CHECK(g1.f.wd != nullptr && g1.f.w4_d == 16, "winding mod 4 planes in the blend (w4_d %g)", (double)g1.f.w4_d);
    CHECK(fabs(g1.cy[3] - (3.5f + 0.03f)) < 1e-3 && fabs(g2.cx[0] + 2.25f) < 1e-3, "axis kept");
    fit_fields_free(&g1); fit_fields_free(&g2);
    /* tile weights through the whole-set path (fblend_run2): the expected planes from the _cw rules with fb_tile_w weights */
    {
        static fb_opts op; memset(&op, 0, sizeof op); op.tile = 1; op.conf = FB_CONF_MEAN;
        op.stride[0] = 8; op.stride[1] = 6; op.stride[2] = 10; op.grid[1][0] = 4; op.grid[1][1] = 3; op.grid[1][2] = 5;
        char o4[256]; snprintf(o4, sizeof o4, "%s/tile", tmp);
        CHECK(fblend_run2(o4, 2, in1, nullptr, &op, 2) == 0, "blend --tile");
        static float q[2][D * H * W]; const long long org[3] = {100, 200, 300}; const int sh[3] = {D, H, W};
        for (int i = 0; i < 2; i++) for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
            const int c3[3] = {z, y, x}; double v = 1; for (int d = 0; d < 3; d++) v *= fb_tile_w(org[d], 2, c3[d], op.stride[d], op.grid[i][d]);
            q[i][(z * H + y) * W + x] = (float)v; (void)sh;
        }
        const float *cw[2] = {q[0], q[1]};
        { const uint8_t *pc[2] = {A[0], B[0]}, *ps[2] = {A[1], B[1]}; fb_phase_cw(2, pc, ps, w1, cw, FB_CONF_MEAN, n, ex[0], ex[1]); }
        for (int k = 2; k < 8; k++) { const uint8_t *s[2] = {A[k], B[k]}; fb_mean_cw(2, s, w1, cw, n, ex[k]); }
        { const uint8_t *s[2] = {A[8], B[8]}; fb_median_known_cw(2, s, w1, cw, n, ex[8]); }
        { const uint8_t *s[2] = {A[9], B[9]}; fb_max(2, s, n, ex[9]); }
        { const uint8_t *const d[2][4] = {{A[10], A[11], A[12], A[13]}, {B[10], B[11], B[12], B[13]}}; uint8_t *const o[4] = {ex[10], ex[11], ex[12], ex[13]}; fb_wind4_cw(2, d, w1, cw, n, o); }
        { const uint8_t *s[2] = {A[14], B[14]}; fb_mean_cw(2, s, w1, cw, n, ex[14]); }
        fit_fields_buf g4; CHECK(fit_fields_load(o4, 0, 0, 1, &g4) == 0, "load tile blend");
        for (int k = 0; k < 15; k++) { int bad = 0; for (size_t i = 0; i < n; i++) if (!g4.a[k] || g4.a[k][i] != ex[k][i]) bad++; CHECK(!bad, "tile plane %d: %d cells differ", k, bad); }
        /* and the tile weights matter: the plain blend differs somewhere in pc */
        fit_fields_buf g5; CHECK(fit_fields_load(o1, 0, 0, 1, &g5) == 0, "reload plain");
        int dif = 0; for (size_t i = 0; i < n; i++) dif += g4.a[0][i] != g5.a[0][i]; CHECK(dif > 0, "tile weights changed nothing");
        fit_fields_free(&g4); fit_fields_free(&g5);
    }
    /* ct: A's own plane passed through (first input A); with B first it is not B's, so it is left out */
    snprintf(p, sizeof p, "%s/ct.raw", o1); FILE *f = fopen(p, "rb"); uint8_t got[D * H * W] = {0};
    CHECK(f && fread(got, 1, n, f) == n && !memcmp(got, ct, n), "ct passed through"); if (f) fclose(f);
    struct stat st; snprintf(p, sizeof p, "%s/ct", o2); CHECK(stat(p, &st) != 0, "ct from a later input not passed");
    snprintf(p, sizeof p, "%s/pc/zarr.json", o2); CHECK(stat(p, &st) == 0, "zarr output for a zarr first input");
    /* a set on other cells is refused */
    snprintf(p, sizeof p, "%s/fields.json", zb); f = fopen(p, "r+"); fseek(f, 0, SEEK_END); long L = ftell(f); char *t = malloc(L + 1); fseek(f, 0, SEEK_SET); if (fread(t, 1, L, f) != (size_t)L) L = 0; t[L] = 0; fclose(f);
    char *q = strstr(t, "\"origin\": [100"); if (q) memcpy(q, "\"origin\": [102", 14);
    f = fopen(p, "w"); fputs(t, f); fclose(f); free(t);
    CHECK(fblend_run(o3, 2, in1, nullptr, 1) == 2, "different cells refused");
    char cmd[600]; snprintf(cmd, sizeof cmd, "rm -rf %s", tmp); if (system(cmd)) {}
}

/* --paste: a 4.8 um --fit-h 4 sub-box P (offset 1,1,2 cells, 3 x 3 x 4) over the 9.6 um --fit-h 2 set A, feather 2 cells: the
   expected planes from the _cw rules with P's sin^2 face ramps and A's share 1 - P's; the winding mod 4 planes A's own (w4_d 16 x
   4.8 um is not A's 16 x 9.6); --box writes A's geometry for the part, a box off the 128-cell shards is refused */
static void test_paste(void) {
    char tmp[] = "/tmp/test_fblend_paste_XXXXXX"; if (!mkdtemp(tmp)) { CHECK(0, "mkdtemp"); return; }
    char ra[256], zb[256], rp[256], o1[256], o2[256], o3[256];
    snprintf(ra, sizeof ra, "%s/a", tmp); snprintf(zb, sizeof zb, "%s/b", tmp); snprintf(rp, sizeof rp, "%s/p", tmp);
    snprintf(o1, sizeof o1, "%s/pasted", tmp); snprintf(o2, sizeof o2, "%s/boxed", tmp); snprintf(o3, sizeof o3, "%s/bad", tmp);
    mkfields(ra, zb);
    enum { PD = 3, PH = 3, PW = 4 }; const int so[3] = {1, 1, 2};
    static uint8_t P[15][PD * PH * PW];
    for (int k = 0; k < 15; k++) for (int i = 0; i < PD * PH * PW; i++) P[k][i] = (uint8_t)(rand() % 256);
    const long long porg[3] = {(100 + 2 * so[0]) * 2, (200 + 2 * so[1]) * 2, (300 + 2 * so[2]) * 2}; const int psh[3] = {PD, PH, PW};
    float cy[2 * PD * 2 + 1], cx[2 * PD * 2 + 1]; for (int z = 0; z < 2 * PD * 2 + 1; z++) { cy[z] = 7; cx[z] = 0; }
    uint8_t *p14[14]; for (int k = 0; k < 14; k++) p14[k] = P[k];
    CHECK(fit_fields_save_ex(rp, porg, psh, 4, 4.8, p14, 16, P[14], cy, cx, 2 * PD * 2 + 1) == 0, "save P");
    static fb_opts op; memset(&op, 0, sizeof op); op.paste = 1; op.feather = 2;
    const char *in[2] = {ra, rp};
    CHECK(fblend_run2(o1, 2, in, nullptr, &op, 2) == 0, "blend --paste");
    const size_t n = D * H * W; const float w1[2] = {1, 1};
    static uint8_t Pf[15][D * H * W]; static float q[2][D * H * W];   /* P on A's cells (neutral outside) and the weights */
    for (int k = 0; k < 15; k++) memset(Pf[k], k < 2 ? 128 : 0, n);
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const int c3[3] = {z, y, x}, s3[3] = {PD, PH, PW}; double v = 1; int in_ = 1;
        for (int d = 0; d < 3; d++) {
            const int a = c3[d] - so[d], e = s3[d] - 1 - a, dist = a < e ? a : e;
            if (dist < 0) { v = 0; in_ = 0; break; }
            const double sn = sin(M_PI / 2 * ((dist + 0.5) / 2)); v *= dist + 0.5 >= 2 ? 1 : sn * sn;
        }
        const size_t i = (size_t)(z * H + y) * W + x; q[1][i] = (float)v; q[0][i] = 1 - (float)v > 1e-6f ? 1 - (float)v : 1e-6f;
        if (in_) for (int k = 0; k < 15; k++) Pf[k][i] = P[k][((z - so[0]) * PH + (y - so[1])) * PW + (x - so[2])];
    }
    const float *cw[2] = {q[0], q[1]}; uint8_t ex[15][D * H * W];
    { const uint8_t *pc[2] = {A[0], Pf[0]}, *ps[2] = {A[1], Pf[1]}; fb_phase_cw(2, pc, ps, w1, cw, FB_CONF_RESULTANT, n, ex[0], ex[1]); }
    for (int k = 2; k < 8; k++) { const uint8_t *s2[2] = {A[k], Pf[k]}; fb_mean_cw(2, s2, w1, cw, n, ex[k]); }
    { const uint8_t *s2[2] = {A[8], Pf[8]}; fb_median_known_cw(2, s2, w1, cw, n, ex[8]); }
    { const uint8_t *s2[2] = {A[9], Pf[9]}; fb_mean_cw(2, s2, w1, cw, n, ex[9]); }
    { static float one[D * H * W]; for (size_t i = 0; i < n; i++) one[i] = 1; const float *c1[2] = {one, one};
      const uint8_t *const d[2][4] = {{A[10], A[11], A[12], A[13]}, {nullptr, nullptr, nullptr, nullptr}}; uint8_t *const o[4] = {ex[10], ex[11], ex[12], ex[13]};
      fb_wind4_cw(2, d, w1, c1, n, o); }
    { const uint8_t *s2[2] = {A[14], Pf[14]}; fb_mean_cw(2, s2, w1, cw, n, ex[14]); }
    fit_fields_buf g; CHECK(fit_fields_load(o1, 0, 0, 1, &g) == 0, "load pasted");
    int inner = 0, outer = 0;
    for (int k = 0; k < 15; k++) { int bad = 0; for (size_t i = 0; i < n; i++) if (!g.a[k] || g.a[k][i] != ex[k][i]) bad++; CHECK(!bad, "paste plane %d: %d cells differ", k, bad); }
    for (size_t i = 0; i < n; i++) { if (q[1][i] == 1) inner += g.a[2][i] == Pf[2][i]; if (q[1][i] == 0) outer += g.a[2][i] == A[2][i]; }
    CHECK(outer > 0, "paste: cells outside P"); (void)inner;
    { int nin = 0, nout = 0, okin = 0, okout = 0; for (size_t i = 0; i < n; i++) { if (q[1][i] == 0) { nout++; okout += g.a[5][i] == A[5][i]; } if (q[1][i] > 0.99f) { nin++; okin += g.a[5][i] == Pf[5][i]; } }
      CHECK(nout > 0 && okout == nout, "outside the box: A's cells (%d / %d)", okout, nout); CHECK(okin == nin, "deep in the box: P's cells (%d / %d)", okin, nin); }
    CHECK(g.f.w4_d == 16 && fabs(g.um - 9.6) < 1e-9 && g.h == 2 && g.origin[2] == 300, "geometry of A");
    fit_fields_free(&g);
    op.box = 1; for (int d = 0; d < 3; d++) { op.b0[d] = 0; op.bn[d] = d == 0 ? D : d == 1 ? H : W; }
    CHECK(fblend_run2(o2, 2, in, nullptr, &op, 2) == 0, "blend --paste --box (whole)");
    CHECK(fit_fields_load(o2, 0, 0, 1, &g) == 0 && g.shape[0] == D && g.origin[1] == 200, "boxed geometry"); fit_fields_free(&g);
    op.b0[0] = 1; op.bn[0] = D - 1; CHECK(fblend_run2(o3, 2, in, nullptr, &op, 2) == 2, "box off the shards refused");
    char cmd[600]; snprintf(cmd, sizeof cmd, "rm -rf %s", tmp); if (system(cmd)) {}
}

int main(void) {
    test_cells();
    test_tiles();
    test_sets();
    test_paste();
    if (fails) { fprintf(stderr, "test_fblend: %d FAILED\n", fails); return 1; }
    printf("test_fblend: ok\n");
    return 0;
}
