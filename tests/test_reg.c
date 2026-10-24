/* ufsm reg core (src/reg.c): affine algebra, quadratic peak, NCC shift recovery, and an end-to-end synthetic pair:
   a fine scan (1 unit voxels, rotated) and a coarse scan (2 unit voxels, area-averaged, smoothly warped) of one random texture; a
   perturbed "published" affine; per-box NCC shifts + robust affine + regularised field must recover the true coarse -> fine map. */
#include "reg.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

/* texture: plane waves, wavelength 5..14 units */
#define NW 14
static double wk[NW][3], wp[NW];
static void tex_init(void) {
    srand(7);
    for (int k = 0; k < NW; k++) {
        double d[3], n = 0; for (int a = 0; a < 3; a++) { d[a] = rand() / (double)RAND_MAX * 2 - 1; n += d[a] * d[a]; }
        n = sqrt(n); const double lam = 5 + 9.0 * rand() / RAND_MAX;
        for (int a = 0; a < 3; a++) wk[k][a] = d[a] / n * 2 * M_PI / lam;
        wp[k] = rand() / (double)RAND_MAX * 2 * M_PI;
    }
}
static double tex(const double P[3]) {
    double s = 0; for (int k = 0; k < NW; k++) s += sin(wk[k][0] * P[0] + wk[k][1] * P[1] + wk[k][2] * P[2] + wp[k]);
    return 128 + 22 * s;
}
/* geometry: fine f -> P = Rf (f - cf) + cP; coarse c -> P = 2 (c - cc) + cP + w(c) */
static double Rf[3][3]; static const double cf[3] = {96, 96, 96}, cc[3] = {40, 40, 40}, cP[3] = {0, 0, 0};
static void warp(const double c[3], double w[3]) {
    w[0] = 1.6 * sin(2 * M_PI * c[2] / 150 + 0.3); w[1] = 1.2 * cos(2 * M_PI * c[0] / 170); w[2] = 1.0 * sin(2 * M_PI * c[1] / 140 + 1);   /* up to 0.8 coarse voxels */
}
static void coarse_to_P(const double c[3], double P[3]) { double w[3]; warp(c, w); for (int a = 0; a < 3; a++) P[a] = 2 * (c[a] - cc[a]) + cP[a] + w[a]; }
static void true_c2f(const double c[3], double f[3]) {
    double P[3]; coarse_to_P(c, P);
    for (int a = 0; a < 3; a++) { f[a] = cf[a]; for (int b = 0; b < 3; b++) f[a] += Rf[b][a] * (P[b] - cP[b]); }
}

/* reg init: a spiral scroll (axis along z at x = y = 80 units, radius 70), coarse 2-unit voxels (P = 2 c), fine 1-unit voxels rotated
   37 deg about the axis and covering z 60 .. 160 (P_xy = Rot (f_xy - 80) + 80, P_z = f_z + 60) */
static double scroll(const double P[3]) {
    const double dx = P[0] - 80, dy = P[1] - 80, r = sqrt(dx * dx + dy * dy);
    if (r > 70) return 0;
    const double ph = r / 6.0 - atan2(dy, dx) / (2 * M_PI), sheet = cos(2 * M_PI * ph) > 0.2 ? 190 : 70;
    return fmax(1, sheet * (0.75 + 0.25 * (tex(P) - 128) / 70));
}
static int test_init(void) {
    int f0 = fails;
    const double a = 37 * M_PI / 180;
    const int64_t cs[3] = {100, 80, 80}, fs[3] = {100, 160, 160};
    uint8_t *c = malloc(100 * 80 * 80), *f = malloc(100 * 160 * 160);
    for (int z = 0; z < 100; z++) for (int y = 0; y < 80; y++) for (int x = 0; x < 80; x++) {
        double s = 0; for (int k = 0; k < 8; k++) { const double P[3] = {2 * x + (k & 1 ? .5 : -.5), 2 * y + (k & 2 ? .5 : -.5), 2 * z + (k & 4 ? .5 : -.5)}; s += scroll(P); }
        c[((size_t)z * 80 + y) * 80 + x] = (uint8_t)(s / 8 + 0.5);
    }
    for (int z = 0; z < 100; z++) for (int y = 0; y < 160; y++) for (int x = 0; x < 160; x++) {
        const double P[3] = {cos(a) * (x - 80) - sin(a) * (y - 80) + 80, sin(a) * (x - 80) + cos(a) * (y - 80) + 80, z + 60};
        f[((size_t)z * 160 + y) * 160 + x] = (uint8_t)scroll(P);
    }
    rvol *cv = rvol_mem(2.0, 4, cs, c), *fv = rvol_mem(1.0, 4, fs, f);
    raff A; reg_init_info in;
    CHECK(reg_init_affine(cv, fv, 4.0, 10, 4, 1, &A, &in) == 0, "init failed");
    double emax = 0;
    for (int k = 0; k < 8; k++) {
        const double p[3] = {k & 1 ? 130 : 30, k & 2 ? 130 : 30, k & 4 ? 90 : 10}, P[3] = {cos(a) * (p[0] - 80) - sin(a) * (p[1] - 80) + 80, sin(a) * (p[0] - 80) + cos(a) * (p[1] - 80) + 80, p[2] + 60};
        double q[3]; raff_apply(&A, p, q); double e = 0; for (int d = 0; d < 3; d++) e += (q[d] - P[d] / 2) * (q[d] - P[d] / 2);
        emax = fmax(emax, sqrt(e));
    }
    printf("init: NCC %.3f, rotation %.2f deg (true 37), flips h %d phi %d, max corner error %.2f coarse voxels\n", in.ncc, in.rot_deg, in.hflip, in.pflip, emax);
    CHECK(!in.hflip && !in.pflip && emax < 2.0, "init: max corner error %.2f, flips %d %d", emax, in.hflip, in.pflip);
    rvol_close(cv); rvol_close(fv);
    return fails - f0;
}

int main(void) {
    /* affine algebra */
    raff A = {{{1.1, 0.2, -0.1, 5}, {0.05, 0.9, 0.3, -2}, {-0.2, 0.1, 1.3, 7}}};
    raff I = raff_mul(raff_inv(A), A); double e = 0;
    for (int i = 0; i < 3; i++) for (int j = 0; j < 4; j++) e = fmax(e, fabs(I.m[i][j] - (i == j)));
    CHECK(e < 1e-12, "raff_inv: %g", e);
    /* quadratic peak */
    { float f[27]; const double t[3] = {0.3, -0.2, 0.45};
      for (int z = -1; z <= 1; z++) for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
          const double dx = x - t[0], dy = y - t[1], dz = z - t[2]; f[(z + 1) * 9 + (y + 1) * 3 + x + 1] = (float)(1 - 0.1 * dx * dx - 0.2 * dy * dy - 0.15 * dz * dz - 0.05 * dx * dy); }
      double d[3], cv; const int r = reg_quad_peak(f, d, &cv);
      CHECK(r == 0 && fabs(d[0] - t[0]) < 1e-5 && fabs(d[1] - t[1]) < 1e-5 && fabs(d[2] - t[2]) < 1e-5, "quad peak %g %g %g", d[0], d[1], d[2]);
      CHECK(cv > 0.15 && cv < 0.25, "curvature %g", cv); }

    tex_init();
    const double ax = 8 * M_PI / 180, bx = 3 * M_PI / 180;   /* about z then x */
    const double Rz[3][3] = {{cos(ax), -sin(ax), 0}, {sin(ax), cos(ax), 0}, {0, 0, 1}}, Rx[3][3] = {{1, 0, 0}, {0, cos(bx), -sin(bx)}, {0, sin(bx), cos(bx)}};
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) { Rf[i][j] = 0; for (int k = 0; k < 3; k++) Rf[i][j] += Rx[i][k] * Rz[k][j]; }
    const int64_t fs[3] = {192, 192, 192}, cs[3] = {80, 80, 80};
    uint8_t *fv0 = malloc(192 * 192 * 192), *cv0 = malloc(80 * 80 * 80);
    for (int z = 0; z < 192; z++) for (int y = 0; y < 192; y++) for (int x = 0; x < 192; x++) {
        const double f[3] = {x, y, z}; double P[3];
        for (int a = 0; a < 3; a++) { P[a] = cP[a]; for (int b = 0; b < 3; b++) P[a] += Rf[a][b] * (f[b] - cf[b]); }
        fv0[((size_t)z * 192 + y) * 192 + x] = (uint8_t)fmin(255, fmax(1, tex(P)));
    }
    for (int z = 0; z < 80; z++) for (int y = 0; y < 80; y++) for (int x = 0; x < 80; x++) {
        double s = 0;
        for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) for (int c = 0; c < 3; c++) {
            const double cp[3] = {x + (c - 1) / 3.0, y + (b - 1) / 3.0, z + (a - 1) / 3.0}; double P[3]; coarse_to_P(cp, P); s += tex(P);
        }
        cv0[((size_t)z * 80 + y) * 80 + x] = (uint8_t)fmin(255, fmax(1, s / 27 + 0.5));
    }
    rvol *fv = rvol_mem(1.0, 3, fs, fv0), *cv = rvol_mem(2.0, 3, cs, cv0);
    /* the true affine part (no warp): c = cc + Rf^T... inverse of f = cf + Rf^T 2 (c - cc) -> c = cc + Rf (f - cf) / 2 ; perturbed */
    raff At = {0};
    for (int i = 0; i < 3; i++) { for (int j = 0; j < 3; j++) At.m[i][j] = Rf[i][j] / 2; At.m[i][3] = cc[i]; for (int j = 0; j < 3; j++) At.m[i][3] -= Rf[i][j] / 2 * cf[j]; }
    raff Ap = At; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Ap.m[i][j] *= 1.004;
    Ap.m[0][3] += 0.7; Ap.m[2][3] -= 0.5; Ap.m[1][3] += 0.2;
    /* field map round trip */
    { rfield F = {{3, 3, 3}, {0, 0, 0}, 40, calloc(81, sizeof(float))}; for (int i = 0; i < 81; i++) F.d[i] = 0.3f * sinf(i);
      rmap m = {Ap, raff_inv(Ap), &F}; const double c[3] = {33.3, 41.2, 50.7}; double f[3], c2[3]; rmap_c2f(&m, c, f); rmap_f2c(&m, f, c2);
      CHECK(fabs(c2[0] - c[0]) + fabs(c2[1] - c[1]) + fabs(c2[2] - c[2]) < 1e-5, "rmap round trip %g %g %g", c2[0], c2[1], c2[2]); free(F.d); }
    /* label warp core (reg warp): both directions against the exact per-voxel map, threads 1 == threads 4, a ragged block edge */
    { rfield F = {{3, 3, 3}, {0, 0, 0}, 40, calloc(81, sizeof(float))}; for (int i = 0; i < 81; i++) F.d[i] = 0.3f * sinf(i);
      rmap m = {Ap, raff_inv(Ap), &F};
      for (int c2f = 0; c2f <= 1; c2f++) {
          const int L = 0, G = 8, S = 64, n[3] = {61, 64, 57}; const int64_t o[3] = {c2f ? 8 : 20, c2f ? 6 : 24, c2f ? 10 : 30};
          const int64_t lo[3] = {0, 0, 0}, ln[3] = {200, 200, 200}; const double k = c2f ? 1.0 : 0.5;   /* label voxel in label-scan L0 voxels */
          uint8_t *lb = malloc(200 * 200 * 200);
          for (int z = 0; z < 200; z++) for (int y = 0; y < 200; y++) for (int x = 0; x < 200; x++) lb[((size_t)z * 200 + y) * 200 + x] = (uint8_t)((x / 3 + 7 * (y / 3) + 31 * (z / 3)) % 250);
          const int NN = reg_warp_nn(S, G); float *nd = malloc(sizeof(float) * 4 * NN * NN * NN);
          uint8_t *o1 = malloc((size_t)S * S * S), *o4 = malloc((size_t)S * S * S); double l0[3], h0[3];
          CHECK(reg_warp_lattice(&m, c2f, L, o, n, G, nullptr, nullptr, nd, l0, h0, 1), "warp lattice: no node ok");
          memset(o1, 255, (size_t)S * S * S); const long long c1 = reg_warp_fill(nd, G, n, S, k, lb, lo, ln, 255, o1, 1);
          CHECK(reg_warp_lattice(&m, c2f, L, o, n, G, nullptr, nullptr, nd, l0, h0, 4), "warp lattice (4 threads)");
          memset(o4, 255, (size_t)S * S * S); const long long c4 = reg_warp_fill(nd, G, n, S, k, lb, lo, ln, 255, o4, 4);
          CHECK(c1 == c4 && !memcmp(o1, o4, (size_t)S * S * S), "warp: threads 1 vs 4 differ (%lld %lld)", c1, c4);
          long long bad = 0, tot = 0, outside = 0; const double sc = 1 << L;
          for (int z = 0; z < S; z++) for (int y = 0; y < S; y++) for (int x = 0; x < S; x++) {
              const uint8_t v = o1[((size_t)z * S + y) * S + x];
              if (z >= n[0] || y >= n[1] || x >= n[2]) { outside += v != 255; continue; }
              const double p[3] = {(o[2] + x + 0.5) * sc - 0.5, (o[1] + y + 0.5) * sc - 0.5, (o[0] + z + 0.5) * sc - 0.5}; double q[3];
              if (c2f) rmap_c2f(&m, p, q); else rmap_f2c(&m, p, q);
              const long li[3] = {lrint((q[2] + 0.5) / k - 0.5), lrint((q[1] + 0.5) / k - 0.5), lrint((q[0] + 0.5) / k - 0.5)};
              if (li[0] < 0 || li[1] < 0 || li[2] < 0 || li[0] >= 200 || li[1] >= 200 || li[2] >= 200) continue;
              tot++; bad += v != lb[((size_t)li[0] * 200 + li[1]) * 200 + li[2]];
          }
          printf("label warp %s: %lld / %lld voxels differ from the exact map\n", c2f ? "c2f" : "f2c", bad, tot);
          CHECK(tot > 50000 && bad < tot / 50 && outside == 0, "label warp %s: %lld / %lld differ, %lld written outside the block", c2f ? "c2f" : "f2c", bad, tot, outside);
          free(lb); free(nd); free(o1); free(o4);
      }
      free(F.d); }
    rmap m = {Ap, raff_inv(Ap), nullptr};
    /* boxes */
    for (int lc = 0; lc <= 1; lc++) {
        rmeas q = {.cv = cv, .fv = fv, .m = &m, .lc = lc, .lf = rvol_level_for(fv, 2.0 * (1 << lc)), .B = lc ? 12 : 20, .R = 3, .ss = 1, .metric = REG_DOG, .iters = 5, .hp = 3, .lp = 0.7, .minfrac = 0.9};
        CHECK(q.lf == lc + 1, "fine level %d for coarse level %d", q.lf, lc);
        int n = 0; double cs3[512 * 3], v[512 * 3], w[512]; double emax = 0, esum = 0;
        for (double z = 20; z <= 60; z += 8) for (double y = 20; y <= 60; y += 8) for (double x = 20; x <= 60; x += 8) {
            rbox b = {.c = {x, y, z}};
            reg_measure(&q, &b);
            /* expected total: A_p(f_true(c)) - c */
            double ft[3], ce[3]; true_c2f(b.c, ft); raff_apply(&Ap, ft, ce);
            double err = 0; for (int a = 0; a < 3; a++) err += (b.s[a] - (ce[a] - b.c[a])) * (b.s[a] - (ce[a] - b.c[a]));
            err = sqrt(err);
            if (b.status == 0) { emax = fmax(emax, err); esum += err; for (int a = 0; a < 3; a++) { cs3[3 * n + a] = b.c[a]; v[3 * n + a] = b.s[a]; } w[n++] = 1; }
        }
        printf("level %d: %d boxes ok, shift error mean %.3f max %.3f coarse voxels\n", lc, n, esum / fmax(1, n), emax);
        CHECK(n > 150, "level %d: only %d boxes ok", lc, n);
        CHECK(esum / n < (lc ? 0.08 : 0.04) && emax < (lc ? 0.2 : 0.1), "level %d box shifts: mean %.3f max %.3f", lc, esum / n, emax);
        if (lc) continue;
        /* fits: affine + field on the residuals */
        double M[3][4]; CHECK(reg_fit_affine(n, cs3, v, w, 0.5, M) == 0, "affine fit");
        raff Ma; memcpy(Ma.m, M, sizeof M); double *r = malloc(n * 3 * sizeof *r);
        for (int i = 0; i < n; i++) { double p[3]; raff_apply(&Ma, cs3 + 3 * i, p); for (int a = 0; a < 3; a++) r[3 * i + a] = v[3 * i + a] - p[a]; }
        rfield F; CHECK(reg_fit_field(n, cs3, r, w, 8, 0.05, 0.5, 1, &F) == 0, "field fit");
        double fe = 0, fm = 0, ae = 0; int k = 0;
        for (double z = 23; z <= 57; z += 3.7) for (double y = 23; y <= 57; y += 3.7) for (double x = 23; x <= 57; x += 3.7) {
            const double c[3] = {x, y, z}; double ft[3], ce[3], p[3], u[3]; true_c2f(c, ft); raff_apply(&Ap, ft, ce); raff_apply(&Ma, c, p); rfield_eval(&F, c, u);
            double e1 = 0, e2 = 0; for (int a = 0; a < 3; a++) { const double t = ce[a] - c[a]; e1 += (t - p[a] - u[a]) * (t - p[a] - u[a]); e2 += (t - p[a]) * (t - p[a]); }
            fe += sqrt(e1); fm = fmax(fm, sqrt(e1)); ae += sqrt(e2); k++;
        }
        printf("map error over the interior: affine only mean %.3f, affine + field mean %.3f max %.3f coarse voxels\n", ae / k, fe / k, fm);
        CHECK(fe / k < 0.04 && fm < 0.12, "field map error mean %.3f max %.3f", fe / k, fm);
        CHECK(fe / k < 0.5 * ae / k, "the field should beat the affine (%.3f vs %.3f)", fe / k, ae / k);
        free(r); free(F.d);
    }
    {   /* the two-level search (R > 3): a prior 3.4 / -2.2 / 1.6 voxels off */
        raff Ab = Ap; Ab.m[0][3] += 3.4; Ab.m[1][3] -= 2.2; Ab.m[2][3] += 1.6;
        rmap mb = {Ab, raff_inv(Ab), nullptr};
        rmeas q = {.cv = cv, .fv = fv, .m = &mb, .lc = 0, .lf = 1, .B = 20, .R = 6, .ss = 1, .metric = REG_DOG, .iters = 5, .hp = 3, .lp = 0.7, .minfrac = 0.9};
        int n = 0; double emax = 0;
        for (double z = 28; z <= 52; z += 12) for (double y = 28; y <= 52; y += 12) for (double x = 28; x <= 52; x += 12) {
            rbox b = {.c = {x, y, z}}; reg_measure(&q, &b);
            double ft[3], ce[3]; true_c2f(b.c, ft); raff_apply(&Ab, ft, ce);
            double e = 0; for (int a = 0; a < 3; a++) e += (b.s[a] - (ce[a] - b.c[a])) * (b.s[a] - (ce[a] - b.c[a]));
            if (b.status == 0) { n++; emax = fmax(emax, sqrt(e)); }
        }
        printf("two-level search (R 6, prior ~4.4 voxels off): %d / 27 ok, max error %.3f\n", n, emax);
        CHECK(n == 27 && emax < 0.1, "two-level search: %d ok, max error %.3f", n, emax);
    }
    rvol_close(fv); rvol_close(cv);
    { const int k = test_init(); (void)k; }
    printf(fails ? "test_reg: %d FAILED\n" : "test_reg: all passed\n", fails);
    return fails != 0;
}
