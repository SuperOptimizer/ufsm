/* nn_unwrap_targets (task unwrap): on a planar phase field phase = frac(k . x) and on a radial one phase = frac(r / pitch) the
   targets are the sign-free normal n n^T and min(1, scale rho) (32; also 8) with rho = |grad phase| (turns per voxel); unknown voxels, voxels
   whose +-s stencil samples are unknown and the border are masked. nn_unwrap_targets2: the unclipped density code x / (1 + x) and the
   packed-sheet loss weight in the mask. */
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
static float g_ds = 32.f;   /* the density scale: min(1, g_ds rho) */
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

static int g_code;   /* the density code: 0 min(1, x), 1 x / (1 + x) (train --density-code), x = g_ds rho */
static float g_um, g_pum, g_pw;   /* the packed-sheet weight: voxel size, --packed-um, --packed-weight (0: off) */
static void run(const uint8_t *ph, int D, int H, int W, int s, uint8_t *t, uint8_t *m) {
    const size_t n = (size_t)D * H * W;
    uint8_t *dp = nn_malloc(n), *dt = nn_malloc(7 * n), *dm = nn_malloc(n);
    nn_h2d(dp, ph, n); nn_unwrap_targets2(dp, D, H, W, s, g_ds, g_code, g_um, g_pum, g_pw, dt, dm); nn_sync();
    nn_d2h(t, dt, 7 * n); nn_d2h(m, dm, n);
    nn_free(dp); nn_free(dt); nn_free(dm);
}
/* worst component error of the targets at masked voxels against n(z, y, x), rho(z, y, x) */
static double check(const uint8_t *t, const uint8_t *m, int D, int H, int W, void (*truth)(int, int, int, double *, double *), size_t *cnt) {
    const size_t n = (size_t)D * H * W; double err = 0; *cnt = 0;
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const size_t i = ((size_t)z * H + y) * W + x;
        if (!m[i]) continue;
        double nv[3], rho; truth(z, y, x, nv, &rho);
        const double xd = g_ds * rho;
        const double ref[7] = {nv[0] * nv[0], nv[1] * nv[1], nv[2] * nv[2], 0.5 + nv[0] * nv[1], 0.5 + nv[0] * nv[2], 0.5 + nv[1] * nv[2], g_code ? xd / (1 + xd) : fmin(1, xd)};
        for (int c = 0; c < 7; c++) err = fmax(err, fabs(t[c * n + i] / 255. - ref[c]));
        (*cnt)++;
    }
    return err;
}
static double kv[3] = {0.004, -0.007, 0.009};   /* turns per voxel */
static void plane_truth(int z, int y, int x, double *nv, double *rho) {
    (void)z; (void)y; (void)x; const double r = sqrt(kv[0] * kv[0] + kv[1] * kv[1] + kv[2] * kv[2]);
    for (int d = 0; d < 3; d++) nv[d] = kv[d] / r;
    *rho = r;
}
static const double cy = 20.3, cx = -15.7, pitch = 60;
static void radial_truth(int z, int y, int x, double *nv, double *rho) {
    (void)z; const double dy = y - cy, dx = x - cx, r = hypot(dy, dx);
    nv[0] = 0; nv[1] = dy / r; nv[2] = dx / r; *rho = 1 / pitch;
}

int main(void) {
    nn_init(0);
    const int D = 40, H = 48, W = 56, s = 4;
    const size_t n = (size_t)D * H * W;
    uint8_t *ph = malloc(n), *t = malloc(7 * n), *m = malloc(n);
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        double q = kv[0] * z + kv[1] * y + kv[2] * x; q -= floor(q);
        ph[((size_t)z * H + y) * W + x] = (uint8_t)((int)lround(q * 252) % 252);
    }
    run(ph, D, H, W, s, t, m);
    size_t cnt; double e = check(t, m, D, H, W, plane_truth, &cnt);
    const size_t inner = (size_t)(D - 2 * s) * (H - 2 * s) * (W - 2 * s);
    CHECK(cnt == inner, "plane: %zu masked voxels, expected the %zu interior ones", cnt, inner);
    CHECK(e < 0.03, "plane: max target error %.4f", e);
    printf("plane: %zu voxels, max target error %.4f\n", cnt, e);
    g_ds = 8.f;   /* the 9.6 um density code (--density-scale 8) */
    run(ph, D, H, W, s, t, m); e = check(t, m, D, H, W, plane_truth, &cnt);
    CHECK(e < 0.03, "plane, density scale 8: max target error %.4f", e);
    g_ds = 32.f; g_code = 1;   /* the unclipped code: x = 32 rho = 0.39 here, and where min(1, x) clips (scale 160: x = 1.94) */
    run(ph, D, H, W, s, t, m); e = check(t, m, D, H, W, plane_truth, &cnt);
    CHECK(e < 0.03, "plane, density code 1: max target error %.4f", e);
    g_ds = 160.f; run(ph, D, H, W, s, t, m); e = check(t, m, D, H, W, plane_truth, &cnt);
    CHECK(e < 0.03, "plane, density code 1, scale 160: max target error %.4f", e);
    g_ds = 32.f; g_code = 0;
    {   /* the packed-sheet weight: rho = 0.0121 turns per voxel; at 0.5 um voxels the spacing is 41 um (< 120 / 2: weight 8), at 2 um
           165 um (>= 120: 1), at 1.2 um 99 um (between: 1 + 7 (120 - 99) / 60 = 3.4 -> 3) */
        const float ums[3] = {0.5f, 2.f, 1.2f}; const int want[3] = {8, 1, 3};
        for (int k = 0; k < 3; k++) {
            g_um = ums[k]; g_pum = 120.f; g_pw = 8.f; run(ph, D, H, W, s, t, m);
            size_t ok = 0, tot = 0; for (size_t i = 0; i < n; i++) if (m[i]) { tot++; ok += k < 2 ? m[i] == want[k] : abs(m[i] - want[k]) <= 1; }   /* (the middle
               case: the 1/252-turn phase quantisation moves the local spacing a few percent, 3.4 rounds to 3 or 4) */
            CHECK(tot && ok == tot, "packed weight at %.1f um voxels: %zu of %zu masked voxels weigh %d", ums[k], ok, tot, want[k]);
        }
        g_pw = 0.f;
    }
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {   /* radial, with an unknown slab */
        const size_t i = ((size_t)z * H + y) * W + x;
        double q = hypot(y - cy, x - cx) / pitch; q -= floor(q);
        ph[i] = z >= 20 && z < 23 ? 255 : (uint8_t)((int)lround(q * 252) % 252);
    }
    run(ph, D, H, W, s, t, m);
    e = check(t, m, D, H, W, radial_truth, &cnt);
    size_t bad = 0;
    for (int z = 16; z < 27; z++) {   /* unknown: z in [20, 23) and the voxels whose +-s samples land there */
        if (!((z >= 20 && z < 23) || (z + s >= 20 && z + s < 23) || (z - s >= 20 && z - s < 23))) continue;
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) bad += m[((size_t)z * H + y) * W + x];
    }
    CHECK(bad == 0, "radial: %zu voxels with an unknown stencil sample are in the mask", bad);
    CHECK(e < 0.04, "radial: max target error %.4f", e);
    printf("radial: %zu voxels, max target error %.4f\n", cnt, e);
    free(ph); free(t); free(m);
    {   /* nn_count_loss (train --count-lambda) on a planar phase of 0.0436 turns per voxel (64^3), segments of 32 voxels: density logits
           log(c rho dscale) predict c rho (the x / (1 + x) code): c = 1 gives a turns ratio ~1 and loss ~0; c = 0.5 a ratio ~0.5,
           loss ~ log 2 and a gradient that raises every logit it touches; the gradient matches finite differences of the loss */
        const int Q = 64; const size_t q3 = (size_t)Q * Q * Q; const double k3[3] = {.02, .03, .025}, rho = sqrt(k3[0] * k3[0] + k3[1] * k3[1] + k3[2] * k3[2]);
        const int K = 512, L = 32; const float ds = 8.f, lam = 1.f;
        uint8_t *qp = malloc(q3); float *lg = malloc(q3 * 4), *g = malloc(q3 * 4), out[3];
        for (int z = 0; z < Q; z++) for (int y = 0; y < Q; y++) for (int x = 0; x < Q; x++) { double q = k3[0] * z + k3[1] * y + k3[2] * x; q -= floor(q); qp[((size_t)z * Q + y) * Q + x] = (uint8_t)((int)lround(q * 252) % 252); }
        uint8_t *dp = nn_malloc(q3); float *dl = nn_malloc(q3 * 4), *dg = nn_malloc(q3 * 4), *acc = nn_malloc(q3 * 4), *seg = nn_malloc(3 * K * 4), *o2 = nn_malloc(16); unsigned *nv = nn_malloc(4);
        nn_h2d(dp, qp, q3); nn_zero(acc, q3 * 4);
        const double cs[2] = {1, .5};
        for (int ci = 0; ci < 2; ci++) {
            for (size_t i = 0; i < q3; i++) lg[i] = (float)log(cs[ci] * rho * ds);
            nn_h2d(dl, lg, q3 * 4); nn_zero(dg, q3 * 4);
            nn_count_loss(dl, dp, Q, Q, Q, 1, K, L, ds, 7, lam, dg, acc, seg, nv, o2); nn_sync(); nn_d2h(out, o2, 12); nn_d2h(g, dg, q3 * 4);
            double gs = 0; size_t pos = 0; for (size_t i = 0; i < q3; i++) { gs += g[i]; pos += g[i] > 0; }
            printf("count loss, density x %.1f: %d segments, loss %.4f, turns ratio %.3f, gradient sum %.3f\n", cs[ci], (int)out[1], out[0], out[2], gs);
            CHECK(out[1] > K / 2, "count loss: most segments valid (%d of %d)", (int)out[1], K);
            if (ci == 0) CHECK(out[0] < .03 && fabs(out[2] - 1) < .03, "count loss: exact density -> loss %.4f ratio %.3f", out[0], out[2]);
            else {
                CHECK(fabs(out[2] - .5) < .03 && fabs(out[0] - log(2.)) < .1, "count loss: half density -> ratio %.3f loss %.4f", out[2], out[0]);
                CHECK(pos == 0 && gs < -.8 * lam && gs > -1.0 * lam, "count loss: under-count raises every touched logit (sum %.3f, %zu positive)", gs, pos);
                size_t probe[4] = {0}; int np = 0; for (size_t i = 0; i < q3 && np < 4; i += 9973) if (g[i] != 0.f) probe[np++] = i;
                for (int pi = 0; pi < np; pi++) {   /* the loss as lambda x mean over segments: d / d logit by central differences */
                    const size_t i = probe[pi]; float lp, lm, o3[3];
                    lg[i] += 1e-2f; nn_h2d(dl, lg, q3 * 4); nn_count_loss(dl, dp, Q, Q, Q, 1, K, L, ds, 7, lam, nullptr, acc, seg, nv, o2); nn_sync(); nn_d2h(o3, o2, 12); lp = o3[0];
                    lg[i] -= 2e-2f; nn_h2d(dl, lg, q3 * 4); nn_count_loss(dl, dp, Q, Q, Q, 1, K, L, ds, 7, lam, nullptr, acc, seg, nv, o2); nn_sync(); nn_d2h(o3, o2, 12); lm = o3[0];
                    lg[i] += 1e-2f;
                    const double fd = lam * (lp - lm) / 2e-2;
                    CHECK(fabs(fd - g[i]) < .02 * fabs(g[i]) + 1e-6, "count loss gradient voxel %zu: analytic %.6g finite difference %.6g", i, g[i], fd);
                }
            }
        }
        nn_free(dp); nn_free(dl); nn_free(dg); nn_free(acc); nn_free(seg); nn_free(o2); nn_free(nv); free(qp); free(lg); free(g);
    }
    {   /* train --density-packed 0 (nn_set_loss_unit_ch7(6)): the 7-plane loss with a weighted mask equals the weighted loss on planes
           0..5 and the 0 / 1-mask loss on plane 6 (bit for bit: the same kernels, the weight read as 1) */
        const int Q = 24; const size_t q3 = (size_t)Q * Q * Q;
        const shape5 s7 = {1, 7, Q, Q, Q};
        float *lg = malloc(7 * q3 * 4), *ga = malloc(7 * q3 * 4), *gb = malloc(7 * q3 * 4);
        uint8_t *t = malloc(7 * q3), *m = malloc(q3), *m01 = malloc(q3), w7[7] = {1, 1, 1, 1, 1, 1, 1};
        unsigned rs = 7;
        for (size_t i = 0; i < 7 * q3; i++) { rs = rs * 1103515245u + 12345u; lg[i] = ((rs >> 8) % 2001) / 500.f - 2.f; t[i] = (uint8_t)(rs >> 20); }
        for (size_t i = 0; i < q3; i++) { rs = rs * 1103515245u + 12345u; m[i] = (uint8_t)((rs >> 16) % 9); m01[i] = m[i] ? 1 : 0; }   /* weights 0..8 */
        float *dl = nn_malloc(7 * q3 * 4), *dg = nn_malloc(7 * q3 * 4), *sc = nn_malloc(nn_loss_scratch(s7));
        uint8_t *dt = nn_malloc(7 * q3), *dm = nn_malloc(q3), *dw = nn_malloc(7);
        nn_h2d(dl, lg, 7 * q3 * 4); nn_h2d(dt, t, 7 * q3); nn_h2d(dw, w7, 7);
        float fa[16], fb[16], fw[16];
        nn_h2d(dm, m, q3); nn_set_loss_unit_ch7(6); nn_loss_async(dl, dt, dm, dw, s7, 0.f, dg, sc); nn_sync(); nn_d2h(ga, dg, 7 * q3 * 4); nn_loss_fetch(sc, s7, fa);
        nn_set_loss_unit_ch7(-1); nn_loss_async(dl, dt, dm, dw, s7, 0.f, dg, sc); nn_sync(); nn_d2h(gb, dg, 7 * q3 * 4); nn_loss_fetch(sc, s7, fw);
        size_t bad6 = 0, diff6 = 0;
        for (size_t i = 0; i < 6 * q3; i++) bad6 += ga[i] != gb[i];
        for (size_t i = 6 * q3; i < 7 * q3; i++) diff6 += ga[i] != gb[i];
        CHECK(bad6 == 0 && diff6 > 0, "unit density mask: planes 0..5 changed at %zu voxels, plane 6 changed at %zu", bad6, diff6);
        nn_h2d(dm, m01, q3); nn_loss_async(dl, dt, dm, dw, s7, 0.f, dg, sc); nn_sync(); nn_d2h(gb, dg, 7 * q3 * 4); nn_loss_fetch(sc, s7, fb);
        size_t bad = 0; for (size_t i = 6 * q3; i < 7 * q3; i++) bad += ga[i] != gb[i];
        CHECK(bad == 0 && fa[6] == fb[6], "unit density mask: plane 6 differs from the 0 / 1-mask loss at %zu voxels (bce %.6f vs %.6f)", bad, fa[6], fb[6]);
        printf("unit density mask: planes 0..5 unchanged, plane 6 = the 0 / 1-mask loss (bce %.4f weighted -> %.4f unit)\n", fw[6], fa[6]);
        nn_free(dl); nn_free(dg); nn_free(sc); nn_free(dt); nn_free(dm); nn_free(dw); free(lg); free(ga); free(gb); free(t); free(m); free(m01);
    }
    if (fails) { fprintf(stderr, "test_unwrap_targets: %d failure(s)\n", fails); return 1; }
    printf("test_unwrap_targets: OK\n");
    return 0;
}
