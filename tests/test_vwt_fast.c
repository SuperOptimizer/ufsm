/* vwt fast conv kernels (src/nn_vwt_fast.cu, bf16 tensor cores) vs the reference (src/nn_vwt.cu, fp32): forward, backward-data, weight
   and bias gradients for every conv shape of the vwt net at small sizes (T 16, L 24 grids), pass: max |fast - ref| / rms(ref) < 2e-2.
   Usage: test_vwt_fast [bench]  (bench: ms per call of both at the full T 128 / L 384 shapes, and the slab's conv total) */
#include "vwt.h"
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

size_t vtf_conv_scratch(vt_t x, int co, int k, int sd, int sh, int sw);
int vtf_conv_fwd(vt_t x, const float *w, const float *b, int k, int sd, int sh, int sw, vt_t y, void *scratch);
int vtf_conv_bwd(vt_t x, vt_t gy, const float *w, int k, int sd, int sh, int sw, vt_t gx, float *gw, float *gb, void *scratch);

static uint64_t rs = 12345;
static float urnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((double)(rs >> 11) * 0x1.0p-53 * 2 - 1); }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static float *dev_rand(size_t n, float s) { float *h = malloc(n * 4); for (size_t i = 0; i < n; i++) h[i] = urnd() * s; float *d = nn_malloc(n * 4); nn_h2d(d, h, n * 4); free(h); return d; }
static float *dev_zero(size_t n) { float *d = nn_malloc(n * 4); nn_zero(d, n * 4); return d; }
static double cmp(const char *what, const float *a, const float *b, size_t n) {   /* max |a - b| / rms(b) */
    float *ha = malloc(n * 4), *hb = malloc(n * 4); nn_d2h(ha, a, n * 4); nn_d2h(hb, b, n * 4);
    double ss = 0, mx = 0; for (size_t i = 0; i < n; i++) { ss += (double)hb[i] * hb[i]; const double d = fabs((double)ha[i] - hb[i]); if (d > mx || d != d) mx = d != d ? 1e30 : d; }
    const double r = mx / (sqrt(ss / n) + 1e-30); free(ha); free(hb);
    printf("  %-6s rel %.3e\n", what, r); return r;
}
typedef struct { int ci, co, k, s, dd, hh; const char *nm; } shp;   /* input grid D = H = dd (transverse), W = L */
/* the vwt net's convs (T 128: stage grids 128 / 64 / 32 / 16) */
static const shp S[] = {
    {2, 32, 3, 2, 128, 128, "st0.c1"}, {32, 32, 3, 1, 64, 64, "st0.c2"}, {32, 64, 3, 2, 64, 64, "st1.c1"}, {64, 64, 3, 1, 32, 32, "st1.c2"},
    {64, 96, 3, 2, 32, 32, "st2.c1"}, {96, 96, 3, 1, 16, 16, "st2.c2"}, {96, 192, 1, 1, 16, 16, "trunk"}, {192, 192, 3, 1, 16, 16, "stem"},
    {256, 96, 3, 1, 32, 32, "dec0"}, {96, 96, 3, 1, 32, 32, "dec1"}, {128, 32, 1, 1, 64, 64, "head0"}, {32, 4, 1, 1, 64, 64, "head1"},
};
static const int NS = (int)(sizeof S / sizeof S[0]);
static int stems(const char *nm) { return !strcmp(nm, "stem") ? 4 : 1; }

int main(int argc, char **argv) {
    if (nn_init(0)) { printf("SKIP: no GPU\n"); return 0; }
    const int bench = argc > 1 && !strcmp(argv[1], "bench");
    const int L = bench ? 384 : 24;
    int fail = 0; double tot_ref = 0, tot_fast = 0;
    vt_t x0 = {nullptr, 1, 128, 128, 384, 2};
    const size_t sc = vtf_conv_scratch(x0, 0, 3, 1, 1, 1), sr = vt_conv_scratch(x0, 0, 3, 1, 1, 1);
    void *scf = nn_malloc(sc), *scr = nn_malloc(sr);
    printf("scratch: fast %.0f MB, reference %.0f MB\n", sc / 1e6, sr / 1e6);
    for (int i = 0; i < NS; i++) {
        const shp *q = &S[i];
        const int dd = bench ? q->dd : (q->dd >= 32 ? 16 : 8), s = q->s;
        vt_t x = {nullptr, 1, dd, dd, L, q->ci}, y = {nullptr, 1, (dd + s - 1) / s, (dd + s - 1) / s, L, q->co};
        const size_t nx = vt_numel(x), ny = vt_numel(y), nw = (size_t)q->co * q->k * q->k * q->k * q->ci;
        x.p = dev_rand(nx, 1.f);
        float *w = dev_rand(nw, 1.f / sqrtf((float)q->k * q->k * q->k * q->ci)), *b = dev_rand(q->co, 0.1f);
        vt_t yr = y, yf = y, gy = y, gxr = x, gxf = x; yr.p = dev_zero(ny); yf.p = dev_zero(ny); gy.p = dev_rand(ny, 1.f); gxr.p = dev_zero(nx); gxf.p = dev_zero(nx);
        float *gwr = dev_zero(nw), *gwf = dev_zero(nw), *gbr = dev_zero(q->co), *gbf = dev_zero(q->co);
        printf("%s: %d -> %d k%d s%d on %dx%dx%d\n", q->nm, q->ci, q->co, q->k, s, dd, dd, L);
        int rc = vt_conv_fwd(x, w, b, q->k, s, s, 1, yr, scr) | vtf_conv_fwd(x, w, b, q->k, s, s, 1, yf, scf);
        rc |= vt_conv_bwd(x, gy, w, q->k, s, s, 1, gxr, gwr, gbr, scr) | vtf_conv_bwd(x, gy, w, q->k, s, s, 1, gxf, gwf, gbf, scf);
        nn_sync();
        if (rc || nn_check()) { printf("  FAIL: call error %s\n", nn_check() ? nn_check() : ""); fail++; continue; }
        if (!bench) {
            const double e[4] = {cmp("y", yf.p, yr.p, ny), cmp("gx", gxf.p, gxr.p, nx), cmp("gw", gwf, gwr, nw), cmp("gb", gbf, gbr, q->co)};
            for (int j = 0; j < 4; j++) if (!(e[j] < 2e-2)) { printf("  FAIL\n"); fail++; break; }
        } else {
            double t[2][2];
            for (int f = 0; f < 2; f++) {
                double bf = 1e9, bb = 1e9;
                for (int it = 0; it < 4; it++) {
                    nn_sync(); double t0 = now();
                    if (f) vtf_conv_fwd(x, w, b, q->k, s, s, 1, yf, scf); else vt_conv_fwd(x, w, b, q->k, s, s, 1, yr, scr);
                    nn_sync(); double t1 = now();
                    if (f) vtf_conv_bwd(x, gy, w, q->k, s, s, 1, gxf, gwf, gbf, scf); else vt_conv_bwd(x, gy, w, q->k, s, s, 1, gxr, gwr, gbr, scr);
                    nn_sync(); double t2 = now();
                    if (t1 - t0 < bf) bf = t1 - t0; if (t2 - t1 < bb) bb = t2 - t1;
                }
                t[f][0] = bf * 1e3; t[f][1] = bb * 1e3;
            }
            const int m = stems(q->nm);
            printf("  ref fwd %.2f bwd %.2f ms | fast fwd %.2f bwd %.2f ms (x%d per slab)\n", t[0][0], t[0][1], t[1][0], t[1][1], m);
            tot_ref += m * (t[0][0] + t[0][1]); tot_fast += m * (t[1][0] + t[1][1]);
        }
        nn_free(x.p); nn_free(w); nn_free(b); nn_free(yr.p); nn_free(yf.p); nn_free(gy.p); nn_free(gxr.p); nn_free(gxf.p); nn_free(gwr); nn_free(gwf); nn_free(gbr); nn_free(gbf);
    }
    if (bench) printf("slab conv total fwd+bwd: reference %.1f ms, fast %.1f ms (%.1f Mvox/s of T 128 L 384 input)\n", tot_ref, tot_fast, 128.0 * 128 * 384 / (tot_fast * 1e3));
    printf(fail ? "FAIL (%d)\n" : "PASS\n", fail);
    return fail ? 1 : 0;
}
