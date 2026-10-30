/* vt_multiview (src/vwt_mv.c): a smooth world phase field seen by two rotated, shifted slabs: zero loss for the exact field plus any
   constant offset; positive loss after a perturbation; the analytic gradient matches central differences. */
#include "../src/vwt.h"
#include "../src/nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double field(const double p[3]) { return 0.05 * p[0] + 0.03 * p[1] + 0.02 * p[2] + 0.3 * sin(0.07 * p[0] + 0.05 * p[2]); }
static void frame(const double dir[3], const double c[3], int T, int L, double f[4][3]) {
    double d[3], n = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]); for (int e = 0; e < 3; e++) d[e] = dir[e] / n;
    double ref[3] = {0, 0, 0}; int m = 0; for (int e = 1; e < 3; e++) if (fabs(d[e]) < fabs(d[m])) m = e; ref[m] = 1;
    double a[3] = {ref[1] * d[2] - ref[2] * d[1], ref[2] * d[0] - ref[0] * d[2], ref[0] * d[1] - ref[1] * d[0]}; n = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); for (int e = 0; e < 3; e++) a[e] /= n;
    double b[3] = {d[1] * a[2] - d[2] * a[1], d[2] * a[0] - d[0] * a[2], d[0] * a[1] - d[1] * a[0]};
    for (int e = 0; e < 3; e++) { f[1][e] = a[e]; f[2][e] = b[e]; f[3][e] = d[e]; f[0][e] = c[e] - 0.5 * (T - 1) * (a[e] + b[e]) - 0.5 * (L - 1) * d[e]; }
}
static void render(const double f[4][3], int T, int L, double off, float *ph) {
    for (int i = 0; i < T; i++) for (int j = 0; j < T; j++) for (int k = 0; k < L; k++) {
        double p[3]; for (int e = 0; e < 3; e++) p[e] = f[0][e] + i * f[1][e] + j * f[2][e] + k * f[3][e];
        ph[((size_t)i * T + j) * L + k] = (float)(field(p) + off);
    }
}
int main(void) {
    const int T = 16, L = 48; const size_t n = (size_t)T * T * L; int fail = 0;
    double fa[4][3], fb[4][3]; const double c[3] = {100, 200, 300}, cb[3] = {103, 198, 301};
    frame((double[]){1, 0.2, 0.1}, c, T, L, fa); frame((double[]){0.9, 0.5, -0.1}, cb, T, L, fb);
    float *pa = malloc(n * 4), *pb = malloc(n * 4), *g = calloc(n, 4); uint8_t *va = malloc(n), *vb = malloc(n); memset(va, 1, n); memset(vb, 1, n);
    render(fa, T, L, 0, pa); render(fb, T, L, 7.25, pb);
    long ns; double l0 = vt_multiview(pa, pb, va, vb, fa, fb, T, L, 2, 0.5f, 1, 1, g, &ns);
    printf("exact field + offset: loss %.2e over %ld shared samples\n", l0, ns); if (!(l0 < 1e-6) || ns < 64) fail = 1;
    for (size_t q = 0; q < n; q++) pa[q] += 0.2f * (float)sin(0.3 * q);
    memset(g, 0, n * 4); const double l1 = vt_multiview(pa, pb, va, vb, fa, fb, T, L, 2, 0.5f, 1, 1, g, &ns);
    printf("perturbed: loss %.4f\n", l1); if (!(l1 > 1e-3)) fail = 1;
    double maxe = 0; int nchk = 0;
    for (size_t q = 0; q < n && nchk < 40; q += 97) {
        if (g[q] == 0) continue; const float s = pa[q]; const double h = 1e-3;
        pa[q] = s + (float)h; const double lp = vt_multiview(pa, pb, va, vb, fa, fb, T, L, 2, 0.5f, 1, 1, nullptr, &ns);
        pa[q] = s - (float)h; const double lm = vt_multiview(pa, pb, va, vb, fa, fb, T, L, 2, 0.5f, 1, 1, nullptr, &ns); pa[q] = s;
        const double fd = (lp - lm) / (2 * h), e = fabs(fd - g[q]) / (fabs(fd) + fabs(g[q]) + 1e-9); if (e > maxe) maxe = e; nchk++;
    }
    printf("gradient vs central differences: %d entries, max rel error %.3e\n", nchk, maxe); if (nchk < 10 || maxe > 2e-2) fail = 1;
    {   /* vt_multiview_dev vs the host on the perturbed field: loss, count, gradient (added to a nonzero buffer) */
        if (nn_init(0)) return 1;
        memset(g, 0, n * 4); const double lh = vt_multiview(pa, pb, va, vb, fa, fb, T, L, 2, 0.5f, 1, 0.5f, g, &ns); const long nh = ns;
        float *dpa = nn_malloc(n * 4), *dpb = nn_malloc(n * 4), *dg = nn_malloc(n * 4), *hg = malloc(n * 4); uint8_t *dva = nn_malloc(n), *dvb = nn_malloc(n);
        nn_h2d(dpa, pa, n * 4); nn_h2d(dpb, pb, n * 4); nn_h2d(dva, va, n); nn_h2d(dvb, vb, n);
        for (size_t q = 0; q < n; q++) hg[q] = 0.01f * (float)(q % 7); nn_h2d(dg, hg, n * 4);
        long nd = 0; const double ld = vt_multiview_dev(dpa, dpb, dva, dvb, fa, fb, T, L, 2, 0.5f, 1, 0.5f, dg, &nd);
        nn_d2h(hg, dg, n * 4); double ge = 0, gm = 0;
        for (size_t q = 0; q < n; q++) { const double d = fabs(hg[q] - 0.01f * (float)(q % 7) - g[q]); if (d > ge) ge = d; if (fabs(g[q]) > gm) gm = fabs(g[q]); }
        printf("device: loss %.6f (host %.6f), shared %ld (host %ld), max |dg| %.2e of max |g| %.2e\n", ld, lh, nd, nh, ge, gm);
        if (nd != nh || fabs(ld - lh) > 1e-5 * (1 + lh) || ge > 1e-3 * gm + 1e-9) fail = 1;
    }
    printf(fail ? "test_vwt_mv: FAIL\n" : "test_vwt_mv: ok\n"); return fail;
}
