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

int main(void) {
    test_cells();
    test_sets();
    if (fails) { fprintf(stderr, "test_fblend: %d FAILED\n", fails); return 1; }
    printf("test_fblend: ok\n");
    return 0;
}
