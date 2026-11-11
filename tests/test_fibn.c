/* train --fib-normal-lambda (src/nn_loss.cu nn_fibn / nn_fibn_apply): the normal head orthogonal to fibres.
   A tube of radius 2 through the window along d, for d = every one of the 48 cube symmetries (axis permutations x flips) of an asymmetric
   direction: the device reads the tube plane after the window's symmetry, so the tangent it derives must follow the symmetry - checked as
   |t . d| on the tube axis (the derivation is the only "rotation" there is; no direction is transformed on the host). Against a host
   reference of the same structure-tensor tangent: the valid voxel count and the sum of t^T N t; the gradient (lambda / count x d(t^T N t) /
   dlogit, fp32 storage) against central differences of the host loss; slabs: two half-window slabs count only voxels whose stencil lies
   inside their own planes, and the planes outside the slab are never read (poisoned with tube values there). */
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

static float h2f(uint16_t h) {   /* IEEE half -> float */
    const uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023; uint32_t f;
    if (!e) { if (!m) f = s; else { int k = 0; uint32_t mm = m; while (!(mm & 1024)) { mm <<= 1; k++; } f = s | (uint32_t)(113 - k) << 23 | (mm & 1023) << 13; } }
    else if (e == 31) f = s | 0x7f800000u | m << 13; else f = s | (e + 112) << 23 | m << 13;
    float r; memcpy(&r, &f, 4); return r;
}
static uint16_t f2h(float v) {   /* float -> IEEE half, round to nearest (the test's values are normal halves) */
    uint32_t f; memcpy(&f, &v, 4); const uint32_t s = f >> 31 << 15; int e = (int)((f >> 23) & 255) - 112; uint32_t m = f & 0x7fffff;
    if (e <= 0) return (uint16_t)s; uint32_t r = (uint32_t)e << 10 | m >> 13; if ((m & 0x1fff) > 0x1000 || ((m & 0x1fff) == 0x1000 && (r & 1))) r++; return (uint16_t)(s | r);
}
static float fm(const uint8_t *f, size_t i) { return f[i] == 255 ? 0.f : f[i] * (1.f / 254.f); }
/* the host tangent (the kernel's algorithm in float) */
static int tangent(const uint8_t *fib, size_t i, int H, int W, int r, float t[3]) {
    const size_t hw = (size_t)H * W; float S[6] = {0};
    for (int dz = -r; dz <= r; dz++) for (int dy = -r; dy <= r; dy++) for (int dx = -r; dx <= r; dx++) {
        const size_t u = i + (ptrdiff_t)dz * (ptrdiff_t)hw + (ptrdiff_t)dy * W + dx;
        const float gz = 0.5f * (fm(fib, u + hw) - fm(fib, u - hw)), gy = 0.5f * (fm(fib, u + W) - fm(fib, u - W)), gx = 0.5f * (fm(fib, u + 1) - fm(fib, u - 1));
        S[0] += gz * gz; S[1] += gy * gy; S[2] += gx * gx; S[3] += gz * gy; S[4] += gz * gx; S[5] += gy * gx;
    }
    const float tr = S[0] + S[1] + S[2]; if (!(tr > 1e-3f)) return 0;
    const float k = 2.f / tr, T[3][3] = {{1.f - k * S[0], -k * S[3], -k * S[4]}, {-k * S[3], 1.f - k * S[1], -k * S[5]}, {-k * S[4], -k * S[5], 1.f - k * S[2]}};
    float q = 0; for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) q += T[a][b] * T[a][b];
    int c = 0; if (T[1][1] > T[c][c]) c = 1; if (T[2][2] > T[c][c]) c = 2;
    float v[3] = {T[0][c], T[1][c], T[2][c]};
    for (int it = 0; it < 3; it++) { float w[3]; for (int a = 0; a < 3; a++) w[a] = T[a][0] * v[0] + T[a][1] * v[1] + T[a][2] * v[2];
        const float nv = sqrtf(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]); if (!(nv > 1e-6f)) return 0; for (int a = 0; a < 3; a++) v[a] = w[a] / nv; }
    float Tv[3]; for (int a = 0; a < 3; a++) Tv[a] = T[a][0] * v[0] + T[a][1] * v[1] + T[a][2] * v[2];
    const float lam = v[0] * Tv[0] + v[1] * Tv[1] + v[2] * Tv[2];   /* the leading eigenvalue (1 for a tube); the other two hold q - lam^2 */
    if (lam < 0.75f || q - lam * lam > 0.5f) return 0;   /* (a blob: lam 1/3; a sheet: a second eigenvalue 1; an anisotropic tube raster: small) */
    t[0] = v[0]; t[1] = v[1]; t[2] = v[2]; return 1;
}
static float sig(float x) { return 1.f / (1.f + expf(-x)); }
/* host: valid voxels of slab [z0, z1) and the sum of t^T N t (logits lg: 6 planes of the slab's ns voxels) */
static double host_sum(const uint8_t *fib, const float *lg, int D, int H, int W, int z0, int z1, int r, int thr, double *cnt) {
    const size_t hw = (size_t)H * W, ns = (size_t)(z1 - z0) * hw; double s = 0; *cnt = 0; const int R = r + 1; (void)D;
    for (size_t li = 0; li < ns; li++) {
        const int z = z0 + (int)(li / hw), y = (int)((li / W) % H), x = (int)(li % W); const size_t i = (size_t)z * hw + (size_t)y * W + x;
        if (fib[i] == 255 || fib[i] < thr || z - R < z0 || z + R >= z1 || y < R || y + R >= H || x < R || x + R >= W) continue;
        float t[3]; if (!tangent(fib, i, H, W, r, t)) continue;
        const float c6[6] = {t[0] * t[0], t[1] * t[1], t[2] * t[2], 2 * t[0] * t[1], 2 * t[0] * t[2], 2 * t[1] * t[2]};
        float L = 0; for (int c = 0; c < 6; c++) { const float p = sig(lg[(size_t)c * ns + li]); L += c6[c] * (c < 3 ? p : p - 0.5f); }
        s += L; *cnt += 1;
    }
    return s;
}
static void tube(uint8_t *fib, int D, int H, int W, const double d[3], const double c[3]) {
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const double v[3] = {z - c[0], y - c[1], x - c[2]}, a = v[0] * d[0] + v[1] * d[1] + v[2] * d[2];
        const double q = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] - a * a, dist = sqrt(q > 0 ? q : 0);
        fib[((size_t)z * H + y) * W + x] = dist <= 2.0 ? 254 : dist <= 4.0 ? 0 : 255;   /* the tube, a known empty shell, unknown beyond */
    }
}

int main(void) {
    if (nn_init(0)) { fprintf(stderr, "no GPU\n"); return 1; }
    nn_set_logits_h16(1); nn_set_loss_grad_h16(0);
    const int D = 32, H = 32, W = 32, r = 2, thr = 128; const size_t n = (size_t)D * H * W, cap = nn_fibn_cap(n);
    uint8_t *fib = malloc(n); float *lgf = malloc(6 * n * sizeof(float)); uint16_t *lgh = malloc(6 * n * 2); float *gh = malloc(6 * n * sizeof(float));
    uint8_t *dfib = nn_malloc(n); void *dlg = nn_malloc(6 * n * 2), *dg = nn_malloc(6 * n * sizeof(float)), *dlist = nn_malloc(nn_fibn_list_bytes(cap));
    double *dacc = nn_malloc(64); unsigned *dnl = nn_malloc(64);
    srand(7);
    for (size_t k = 0; k < 6 * n; k++) { lgh[k] = f2h(-2.f + 4.f * rand() / (float)RAND_MAX); lgf[k] = h2f(lgh[k]); }
    nn_h2d(dlg, lgh, 6 * n * 2);
    const double d0[3] = {1.0 / sqrt(14.0), 2.0 / sqrt(14.0), 3.0 / sqrt(14.0)}, cen[3] = {15.3, 16.1, 15.7};
    static const int perms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
    double worst_dot = 1, sum_dot = 0; long ndot = 0; int sym = 0;
    for (int pi = 0; pi < 6; pi++) for (int fl = 0; fl < 8; fl++, sym++) {
        double d[3]; for (int a = 0; a < 3; a++) d[a] = d0[perms[pi][a]] * ((fl >> a & 1) ? -1 : 1);
        tube(fib, D, H, W, d, cen); nn_h2d(dfib, fib, n);
        /* tangent against d on the axis (host algorithm = kernel algorithm, compared below by the sums) */
        for (int z = 3; z < D - 3; z++) for (int y = 3; y < H - 3; y++) for (int x = 3; x < W - 3; x++) {
            const double v[3] = {z - cen[0], y - cen[1], x - cen[2]}, a = v[0] * d[0] + v[1] * d[1] + v[2] * d[2];
            if (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] - a * a > 0.5 * 0.5) continue;
            float t[3]; if (!tangent(fib, ((size_t)z * H + y) * W + x, H, W, r, t)) { CHECK(0, "symmetry %d: no tangent on the axis at %d %d %d", sym, z, y, x); continue; }
            const double dot = fabs(t[0] * d[0] + t[1] * d[1] + t[2] * d[2]); if (dot < worst_dot) worst_dot = dot; sum_dot += dot; ndot++;
        }
        /* whole window: count and sum */
        nn_zero(dacc, 64); nn_fibn(dlg, n, dfib, D, H, W, 0, D, r, thr, dacc, nullptr, nullptr, 0, 0);
        double acc[2]; nn_d2h(acc, dacc, sizeof acc); double hc; const double hs = host_sum(fib, lgf, D, H, W, 0, D, r, thr, &hc);
        CHECK(fabs(acc[0] - hc) <= 0.01 * hc + 2, "symmetry %d: valid voxels %g, host %g", sym, acc[0], hc);
        CHECK(fabs(acc[1] / acc[0] - hs / hc) < 2e-3, "symmetry %d: mean t^T N t %g, host %g", sym, acc[1] / acc[0], hs / hc);
        if (sym == 5) {   /* the gradient against central differences of the host loss lambda / count x sum */
            const float lambda = 0.7f; nn_zero(dg, 6 * n * sizeof(float)); nn_zero(dnl, 64);
            nn_fibn(dlg, n, dfib, D, H, W, 0, D, r, thr, dacc, dlist, dnl, cap, 1);
            nn_fibn_apply(dlist, dnl, cap, dacc, lambda, dg, n); nn_d2h(gh, dg, 6 * n * sizeof(float));
            unsigned nl; nn_d2h(&nl, dnl, sizeof nl);
            CHECK(nl == (unsigned)acc[0], "the gradient list holds %u voxels, the count %g", nl, acc[0]);
            double werr = 0, gmax = 0; int nchk = 0; size_t nz = 0;
            for (size_t k = 0; k < 6 * n; k++) nz += gh[k] != 0.f;
            for (size_t i = 0; i < n && nchk < 60; i += 37) for (int c = 0; c < 6; c++) {
                const size_t k = (size_t)c * n + i; if (gh[k] == 0.f && fib[i] != 254) continue;
                const float keep = lgf[k], e = 1e-2f; double c0, c1;
                lgf[k] = keep + e; const double fp = lambda * host_sum(fib, lgf, D, H, W, 0, D, r, thr, &c0) / acc[0];
                lgf[k] = keep - e; const double fn = lambda * host_sum(fib, lgf, D, H, W, 0, D, r, thr, &c1) / acc[0]; lgf[k] = keep;
                const double num = (fp - fn) / (2 * e), err = fabs(num - gh[k]); if (err > werr) werr = err; if (fabs(num) > gmax) gmax = fabs(num); nchk++;
            }
            printf("gradient: %u fibre voxels, %zu nonzero gradient elements (6 per voxel at most), %d checked, worst |analytic - central| %.2e (max |grad| %.2e)\n", nl, nz, nchk, werr, gmax);
            CHECK(nz <= 6 * (size_t)nl, "gradient outside the fibre voxels: %zu elements", nz);
            CHECK(werr < 1e-3 * (gmax > 1e-6 ? gmax : 1e-6) + 1e-7 && nchk > 20, "gradient vs central differences: worst %.3e of %.3e (%d checked)", werr, gmax, nchk);
        }
        if (sym == 9) {   /* two slabs: the sums add up to the host's per-slab sums; the planes outside a slab are never read */
            for (int s = 0; s < 2; s++) {
                const int z0 = s * 16, z1 = z0 + 16; const size_t ns = (size_t)16 * H * W;
                uint8_t *pz = malloc(n); memcpy(pz, fib, n);
                for (int z = 0; z < D; z++) if (z < z0 || z >= z1) memset(pz + (size_t)z * H * W, 254, (size_t)H * W);   /* poison: a read there breaks the tangent */
                nn_h2d(dfib, pz, n);
                float *ls = malloc(6 * ns * sizeof(float)); uint16_t *lh = malloc(6 * ns * 2);
                for (int c = 0; c < 6; c++) for (size_t k = 0; k < ns; k++) { lh[(size_t)c * ns + k] = lgh[(size_t)c * n + (size_t)z0 * H * W + k]; ls[(size_t)c * ns + k] = lgf[(size_t)c * n + (size_t)z0 * H * W + k]; }
                void *dls = nn_malloc(6 * ns * 2); nn_h2d(dls, lh, 6 * ns * 2);
                nn_zero(dacc, 64); nn_fibn(dls, ns, dfib, D, H, W, z0, z1, r, thr, dacc, nullptr, nullptr, 0, 0);
                double a2[2]; nn_d2h(a2, dacc, sizeof a2); double c2; const double s2 = host_sum(fib, ls, D, H, W, z0, z1, r, thr, &c2);
                printf("slab %d..%d: %g valid voxels (host %g), mean t^T N t %.5f (host %.5f)\n", z0, z1, a2[0], c2, a2[0] > 0 ? a2[1] / a2[0] : 0, c2 > 0 ? s2 / c2 : 0);
                CHECK(fabs(a2[0] - c2) <= 0.01 * c2 + 2 && c2 > 0, "slab %d: valid voxels %g, host %g", s, a2[0], c2);
                CHECK(c2 == 0 || fabs(a2[1] / a2[0] - s2 / c2) < 2e-3, "slab %d: mean %g, host %g", s, a2[1] / a2[0], s2 / c2);
                nn_free(dls); free(ls); free(lh); free(pz);
            }
        }
    }
    printf("tangent on the tube axis over the 48 symmetries: %ld voxels, mean |t . d| %.5f, worst %.5f\n", ndot, sum_dot / ndot, worst_dot);
    CHECK(ndot > 48 * 10 && sum_dot / ndot > 0.995 && worst_dot > 0.97, "tangent: mean %.4f worst %.4f over %ld voxels", sum_dot / ndot, worst_dot, ndot);
    printf(fails ? "test_fibn: %d FAILED\n" : "test_fibn: all passed\n", fails);
    return fails != 0;
}
