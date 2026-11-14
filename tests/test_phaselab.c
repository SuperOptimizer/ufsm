/* test_phaselab: the packed-sheet label targets on a simulated fixture (CPU only; 2026-11-14, DESIGN "phaselab"). Flat sheets of spacing
   sp voxels at several tilts are stamped like src/meshcodes.c (samples every 0.5 voxel, voxel = floor(p + 0.5), codes 1 + 18 (k mod 14),
   two windings in one voxel = 255), run through band_field_phase_pair, and the density / normal targets are derived from the phase twice:
   BEFORE = the old nn_elem.cu stencil (one wrapped difference over +-s voxels), AFTER = src/phase_grad.h (sum of wrapped unit steps).
   Against the truth (rho = 1 / sp, n = the plane normal): the old stencil reads rho x sp = 1.0 / 0.5 / 0.2 / 0 at sp 4 / 3 / 2.5 / 2 on
   axis-aligned sheets and a wrong direction on tilted ones; the new one is exact down to ~2.5 voxels and masks below. Also the sheet-mask
   weight: the phase error of the labelled (phase 0) voxels against the exact sub-voxel position, weighted by pg_sheet_f. */
#include "../src/band.h"
#include "../src/phase_grad.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
enum { N = 44, M = 10 };   /* box edge, border excluded from the statistics */
static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { g_fail++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static int old_grad(const uint8_t *ph, size_t i, int D, int H, int W, int s, float g[3]) {   /* nn_elem.cu phase_grad before 2026-11-14, verbatim */
    const int z = (int)(i / ((size_t)H * W)), y = (int)((i / W) % H), x = (int)(i % W);
    if (ph[i] >= 252 || z < s || z >= D - s || y < s || y >= H - s || x < s || x >= W - s) return 0;
    const size_t sz = (size_t)s * H * W, sy = (size_t)s * W;
    const uint8_t a[6] = {ph[i + sz], ph[i - sz], ph[i + sy], ph[i - sy], ph[i + s], ph[i - s]};
    for (int k = 0; k < 6; k++) if (a[k] >= 252) return 0;
    const float sc = 1.f / (2.f * s * 252.f);
    g[0] = pg_wrap(a[0], a[1]) * sc; g[1] = pg_wrap(a[2], a[3]) * sc; g[2] = pg_wrap(a[4], a[5]) * sc;
    return 1;
}
typedef struct { double known, wrong, med_rs, med_cos; } gstat;   /* share of voxels with a gradient, share of those off by > 25 %, median rho x sp, median |n . n_true| */
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static gstat measure(const uint8_t *ph, int old, const double nt[3], double sp, int s) {
    size_t nv = 0, nk = 0, nw = 0; double *rs = malloc((size_t)N * N * N * 8), *cs = malloc((size_t)N * N * N * 8);
    for (int z = M; z < N - M; z++) for (int y = M; y < N - M; y++) for (int x = M; x < N - M; x++) {
        const size_t i = ((size_t)z * N + y) * N + x; float g[3]; nv++;
        if (!(old ? old_grad(ph, i, N, N, N, s, g) : pg_grad(ph, i, N, N, N, s, g) == 1)) continue;
        const double rho = sqrt((double)g[0] * g[0] + (double)g[1] * g[1] + (double)g[2] * g[2]); if (rho < 1e-3) continue;   /* (the target's mask) */
        nk++; rs[nk - 1] = rho * sp; cs[nk - 1] = fabs((g[0] * nt[0] + g[1] * nt[1] + g[2] * nt[2]) / rho); nw += fabs(rho * sp - 1) > 0.25;
    }
    gstat r = {(double)nk / nv, nk ? (double)nw / nk : 0, 0, 0};
    if (nk) { qsort(rs, nk, 8, cmpd); qsort(cs, nk, 8, cmpd); r.med_rs = rs[nk / 2]; r.med_cos = cs[nk / 2]; }
    free(rs); free(cs); return r;
}
/* the fixture: sheets k at n.(p - c) = off + k sp; returns codes; phase out; true fractional phase per voxel in tru (turns, [0, 1)) */
static void fixture(const double nrm[3], double sp, double off, uint8_t *codes, uint8_t *phase, double *tru) {
    const double c = N / 2.0; double a[3], b[3];
    const double t[3] = {0.3, 0.5, 0.81}; a[0] = nrm[1] * t[2] - nrm[2] * t[1]; a[1] = nrm[2] * t[0] - nrm[0] * t[2]; a[2] = nrm[0] * t[1] - nrm[1] * t[0];
    double l = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); if (l < 1e-6) { a[0] = 1; a[1] = 0; a[2] = 0; l = 1; }
    for (int d = 0; d < 3; d++) a[d] /= l;
    b[0] = nrm[1] * a[2] - nrm[2] * a[1]; b[1] = nrm[2] * a[0] - nrm[0] * a[2]; b[2] = nrm[0] * a[1] - nrm[1] * a[0];
    int *who = malloc((size_t)N * N * N * sizeof *who); memset(who, 0xff, (size_t)N * N * N * sizeof *who); memset(codes, 0, (size_t)N * N * N);
    const int kmax = (int)ceil(N * 1.8 / sp);
    for (int k = -kmax; k <= kmax; k++) {
        const double dd = off + k * sp;
        for (double u = -N; u < N; u += 0.5) for (double v = -N; v < N; v += 0.5) {
            long g[3]; int in = 1;
            for (int d = 0; d < 3; d++) { g[d] = (long)floor(c + dd * nrm[d] + u * a[d] + v * b[d] + 0.5); if (g[d] < 0 || g[d] >= N) in = 0; }
            if (!in) continue;
            const size_t i = ((size_t)g[0] * N + g[1]) * N + g[2];
            if (who[i] >= 0 && who[i] != k) codes[i] = 255;
            else if (codes[i] != 255) { who[i] = k; codes[i] = (uint8_t)(1 + 18 * ((k % 14 + 14) % 14)); }
        }
    }
    free(who);
    const int n3[3] = {N, N, N};
    if (band_field_phase_pair(codes, n3, 40.f, phase, nullptr)) { fprintf(stderr, "pair rule failed\n"); exit(1); }
    for (int z = 0; z < N; z++) for (int y = 0; y < N; y++) for (int x = 0; x < N; x++) {
        const double d = (z - c) * nrm[0] + (y - c) * nrm[1] + (x - c) * nrm[2], f = (d - off) / sp;
        tru[((size_t)z * N + y) * N + x] = f - floor(f);
    }
}
int main(void) {
    const int n3 = N * N * N; uint8_t *codes = malloc(n3), *ph = malloc(n3); double *tru = malloc((size_t)n3 * 8);
    const double tilts[4][3] = {{0, 0, 1}, {0, 1, 1}, {0.1, 0.2, 1}, {0.2, 1, 0.6}};
    for (int t = 0; t < 4; t++) {
        double nt[3], l = sqrt(tilts[t][0] * tilts[t][0] + tilts[t][1] * tilts[t][1] + tilts[t][2] * tilts[t][2]);
        for (int d = 0; d < 3; d++) nt[d] = tilts[t][d] / l;
        printf("sheet normal (%.2f %.2f %.2f)\n", nt[0], nt[1], nt[2]);
        for (int si = 0; si < 6; si++) {
            const double sps[6] = {2.0, 2.5, 3.0, 4.0, 6.0, 10.0}, sp = sps[si];
            fixture(nt, sp, 0.37 * sp, codes, ph, tru);
            const gstat o = measure(ph, 1, nt, sp, 1), a = measure(ph, 0, nt, sp, 1);
            printf("  sp %4.1f  BEFORE known %.2f wrong %.2f rho*sp %.2f |cos| %.2f   AFTER known %.2f wrong %.2f rho*sp %.2f |cos| %.2f\n", sp, o.known, o.wrong, o.med_rs, o.med_cos, a.known, a.wrong, a.med_rs, a.med_cos);
            if (sp >= 6) { CHECK(fabs(o.med_rs - a.med_rs) < 0.02 && fabs(o.known - a.known) < 0.05, "sp %.1f: the new stencil must agree with the old where the old was valid", sp); }
            if (sp >= 2.5 && sp <= 3.5 && t == 0) CHECK(fabs(a.med_rs - 1) < fabs(o.med_rs - 1) || a.known < 0.05, "axis sheets sp %.1f: median rho x sp %.2f (old %.2f)", sp, a.med_rs, o.med_rs);
            if (sp >= 3 && sp <= 4 && t == 0) CHECK(o.wrong > 0.3 * (sp < 3.5) || fabs(o.med_rs - 1) > 0.1 || sp > 3.5, "fixture: the old stencil should alias at sp %.1f", sp);
            if (sp >= 2.5) CHECK(a.wrong <= o.wrong + 0.05 && a.wrong < 0.5, "sp %.1f tilt %d: wrong-gradient share %.2f (old %.2f)", sp, t, a.wrong, o.wrong);
            CHECK(a.known > 0.02 || sp <= 2.0 || t > 0, "sp %.1f: nothing known", sp);
            if (sp <= 3.0 && t == 0) CHECK(fabs(a.med_rs - 1) < fabs(o.med_rs - 1) + 1e-6 || a.known < 0.05, "sp %.1f: new median %.2f not closer to 1 than old %.2f", sp, a.med_rs, o.med_rs);
            /* the sheet voxels: phase error of the labelled voxels, and what the mask leaves */
            double e2 = 0, w = 0, e2w = 0; size_t nl = 0;
            for (int z = M; z < N - M; z++) for (int y = M; y < N - M; y++) for (int x = M; x < N - M; x++) {
                const size_t i = ((size_t)z * N + y) * N + x; if (!(codes[i] > 0 && codes[i] < 255)) continue;
                double e = tru[i]; e -= floor(e + 0.5); const float rho = pg_rate(ph, i, N, N, N), f = rho > 0.f ? pg_sheet_f(rho, 3.f, 8.f) : 1.f;
                nl++; e2 += e * e; w += f; e2w += f * e * e;
            }
            if (nl) printf("      labelled voxels %.2f of the box: phase error rms %.3f, weighted by the sheet mask %.3f (mean weight %.2f)\n", (double)nl / (double)((N - 2 * M) * (N - 2 * M) * (N - 2 * M)), sqrt(e2 / nl), w > 0 ? sqrt(e2w / w) : 0.0, w / nl);
            if (nl && sp <= 3.0) CHECK(w / nl < 0.2, "sp %.1f: the sheet mask keeps %.2f of the labelled voxels", sp, w / nl);
            if (nl && sp >= 10.0) CHECK(w / nl > 0.7, "sp %.1f: the sheet mask drops wide-sheet voxels (%.2f)", sp, w / nl);
            if (nl && sp >= 6.0 && sp <= 6.0 && t == 0 && w > 0) CHECK(sqrt(e2w / w) <= sqrt(e2 / nl) + 1e-3, "weighted error must not exceed the plain one");
        }
    }
    /* a density-scale identity: the physical code at the reference voxel size is the legacy code */
    CHECK(pg_sheet_f(0.f, 3.f, 8.f) == 1.f && pg_sheet_f(0.4f, 3.f, 8.f) == 0.f && pg_sheet_f(0.05f, 3.f, 8.f) == 1.f, "pg_sheet_f");
    printf(g_fail ? "test_phaselab: %d FAILED\n" : "test_phaselab: ok\n", g_fail);
    free(codes); free(ph); free(tru);
    return g_fail != 0;
}
