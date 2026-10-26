/* train --pgrad-lambda: the phase wavenumber loss of a head slab (src/nn_loss.cu nn_pgrad_slab / nn_pgrad_apply).
   (1) the weight-only call (lc = ls = nullptr, the training stats pass) sums exactly the pair weights of the full call and no loss;
   (2) the slab gradient against central differences of the loss (fp32 logits), at several voxels with pairs at each distance;
   (3) nn_pgrad_apply adds lambda / sum w times the gradient into 16-bit logit gradients and zeroes the accumulator. */
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)
static uint64_t rs = 0x9e3779b97f4a7c15ull;
static double urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 11) * (1.0 / 9007199254740992.0); }

int main(void) {
    nn_init(0);
    nn_set_logits_h16(0);
    const int D = 24, H = 40, W = 44, z0 = 5, z1 = 19, s = 1;
    const size_t n = (size_t)D * H * W, ns = (size_t)(z1 - z0) * H * W;
    const float dist[3] = {2.f, 4.f, 8.f};
    uint8_t *ph = malloc(n), *wm = malloc(ns);
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {   /* a curved sheet stack, ~10 voxels per turn */
        const double Wt = 0.02 * z + 0.05 * y + 0.08 * x + 0.0007 * (x - 20) * (x - 20) + 0.3;
        const double f = Wt - floor(Wt);
        ph[((size_t)z * H + y) * W + x] = (x == 30 && y < 8) ? 255 : (uint8_t)((int)lround(f * 252) % 252);   /* (a few unknown codes) */
    }
    for (size_t i = 0; i < ns; i++) { const double u = urand(); wm[i] = u < 0.97 ? 0 : u < 0.98 ? 8 : 1; }   /* (sparse weights: small fp32 block partials, so the central differences resolve) */
    float *lc = malloc(ns * 4), *ls = malloc(ns * 4);
    for (size_t i = 0; i < ns; i++) { lc[i] = (float)(2 * urand() - 1) * 2.f; ls[i] = (float)(2 * urand() - 1) * 2.f; }
    uint8_t *dph = nn_malloc(n), *dwm = nn_malloc(ns); float *dlc = nn_malloc(ns * 4), *dls = nn_malloc(ns * 4), *acc = nn_malloc(2 * ns * 4);
    double *dsum = nn_malloc(4 * sizeof(double));
    nn_h2d(dph, ph, n); nn_h2d(dwm, wm, ns); nn_h2d(dlc, lc, ns * 4); nn_h2d(dls, ls, ns * 4); nn_zero(acc, 2 * ns * 4);
    double full[2], wonly[2];
    /* (1) full call with the gradient, then the weight-only call */
    nn_zero(dsum, 4 * sizeof(double));
    nn_pgrad_slab(dlc, dls, dph, dwm, D, H, W, z0, z1, s, dist, 3, 0, 0, dsum, acc);
    nn_pgrad_slab(nullptr, nullptr, dph, dwm, D, H, W, z0, z1, s, dist, 3, 0, 0, dsum + 2, nullptr);
    nn_sync(); nn_d2h(full, dsum, 2 * sizeof(double)); nn_d2h(wonly, dsum + 2, 2 * sizeof(double));
    printf("pgrad: sum w loss %.6f, sum w %.1f (weight-only: %.6f, %.1f), loss %.5f\n", full[0], full[1], wonly[0], wonly[1], full[0] / full[1]);
    CHECK(full[1] > 500, "too few pairs (sum w %.1f)", full[1]);
    CHECK(wonly[0] == 0 && fabs(wonly[1] - full[1]) <= 1e-9 * full[1], "weight-only sums %.6f %.3f vs the full call's weights %.3f", wonly[0], wonly[1], full[1]);
    {   /* the same with --pgrad-mask 1 (reliable pairs only): the weight-only call keeps the mask's pair selection */
        double fm[2], wm1[2];
        nn_zero(dsum, 4 * sizeof(double));
        nn_pgrad_slab(dlc, dls, dph, dwm, D, H, W, z0, z1, s, dist, 3, 0, 1, dsum, nullptr);
        nn_pgrad_slab(nullptr, nullptr, dph, dwm, D, H, W, z0, z1, s, dist, 3, 0, 1, dsum + 2, nullptr);
        nn_sync(); nn_d2h(fm, dsum, 2 * sizeof(double)); nn_d2h(wm1, dsum + 2, 2 * sizeof(double));
        printf("pgrad mask 1: sum w %.1f (weight-only %.1f)\n", fm[1], wm1[1]);
        CHECK(fm[1] > 0 && wm1[0] == 0 && fabs(wm1[1] - fm[1]) <= 1e-9 * fm[1], "mask 1 weight-only sum w %.3f vs %.3f", wm1[1], fm[1]);
    }
    float *g = malloc(2 * ns * 4); nn_d2h(g, acc, 2 * ns * 4);
    /* (2) central differences of sum w loss at voxels with a gradient */
    int checked = 0; double worst = 0;
    for (size_t k = 0; k < ns && checked < 24; k += 7) {
        for (int ch = 0; ch < 2; ch++) {
            float *L = ch ? ls : lc; float *dL = ch ? dls : dlc;
            const float gk = g[(size_t)ch * ns + k];
            if (fabsf(gk) < 1e-3f) continue;
            const float h = 2e-2f, v0 = L[k];
            double fp[2], fm[2];
            L[k] = v0 + h; nn_h2d(dL, L, ns * 4); nn_zero(dsum, 2 * sizeof(double)); nn_pgrad_slab(dlc, dls, dph, dwm, D, H, W, z0, z1, s, dist, 3, 0, 0, dsum, nullptr); nn_sync(); nn_d2h(fp, dsum, 2 * sizeof(double));
            L[k] = v0 - h; nn_h2d(dL, L, ns * 4); nn_zero(dsum, 2 * sizeof(double)); nn_pgrad_slab(dlc, dls, dph, dwm, D, H, W, z0, z1, s, dist, 3, 0, 0, dsum, nullptr); nn_sync(); nn_d2h(fm, dsum, 2 * sizeof(double));
            L[k] = v0; nn_h2d(dL, L, ns * 4);
            const double fd = (fp[0] - fm[0]) / (2 * h), e = fabs(fd - gk) / fmax(fabs(gk), 1e-2);
            if (e > worst) worst = e;
            CHECK(e < 3e-2, "gradient at voxel %zu channel %d: %.6f vs central difference %.6f", k, ch, gk, fd);
            checked++;
        }
    }
    printf("pgrad: %d gradients against central differences, worst relative error %.4f\n", checked, worst);
    CHECK(checked >= 8, "only %d gradients checked", checked);
    /* (3) apply into fp16 gradients (16-bit loss gradients: nn_set_loss_grad_h16) */
    nn_set_f16(1); nn_set_loss_grad_h16(1);
    uint16_t *gz = calloc(2 * ns, 2); void *dgc = nn_malloc(ns * 2), *dgs = nn_malloc(ns * 2);
    nn_h2d(dgc, gz, ns * 2); nn_h2d(dgs, gz, ns * 2);
    nn_h2d(dsum, full, 2 * sizeof(double));
    const float lambda = 0.5f;
    nn_pgrad_apply(dgc, dgs, acc, ns, dsum, lambda);
    nn_sync();
    _Float16 *hc = malloc(ns * 2); nn_d2h(hc, dgc, ns * 2);
    float *ac = malloc(2 * ns * 4); nn_d2h(ac, acc, 2 * ns * 4);
    double we = 0; size_t nz = 0;
    for (size_t k = 0; k < ns; k++) {
        const double want = (double)lambda * nn_get_grad_scale() / full[1] * g[k], got = (double)hc[k];
        we = fmax(we, fabs(got - want) / (fabs(want) + 1e-4));   /* (fp16: relative above its normal range, ~6e-8 absolute below) */
        nz += ac[k] != 0.f || ac[ns + k] != 0.f;
    }
    printf("pgrad apply: worst relative error %.5f (fp16), %zu accumulator entries left\n", we, nz);
    CHECK(we < 2e-3, "applied gradient off by %.5f", we);
    CHECK(nz == 0, "the accumulator was not zeroed (%zu entries)", nz);
    if (fails) { fprintf(stderr, "test_pgrad: %d FAILED\n", fails); return 1; }
    printf("test_pgrad: OK\n");
    return 0;
}
