/* --attn-muon-lr F / --attn-muon-lr-out F2 (unet_set_attn_muon_lr; audit 2026-11-14 train #3): one optimizer step of a small attention
   U-Net with the multipliers against one without, the same weights and gradients. Every parameter's change is either identical (the
   convs and the AdamW parameters, bit for bit) or F (q/k/v, w1) or F x F2 (the out-projections wo, w2) times the reference's, to float
   rounding; both classes occur, and the F x F2 class equals the out-projections' size. Plain batched Muon and the ext path (--normuon 1). */
#include "unet.h"
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { printf("  FAIL: " __VA_ARGS__); printf("\n"); fails++; } } while (0)

static void run(int ext) {
    unet_cfg cfg = {}; cfg.nlev = 3; cfg.widths[0] = 16; cfg.widths[1] = 32; cfg.widths[2] = 48; cfg.cin = 4; cfg.cout = 2; cfg.G = 8; cfg.attn_blocks = 1; cfg.attn_pe = 0;
    const float F = 0.5f, F2 = 0.2f;
    unet *a = unet_create(&cfg), *b = unet_create(&cfg); unet_init(a, 3); unet_init(b, 3);
    unet_set_attn_muon_lr(a, F, F2);
    if (ext) { unet_set_muon_variant(a, 0, 1, 0.95f); unet_set_muon_variant(b, 0, 1, 0.95f); }
    const size_t n = unet_nparams(a); float *g = malloc(n * 4), *p0 = malloc(n * 4), *pa = malloc(n * 4), *pb = malloc(n * 4);
    unsigned s = 11; for (size_t i = 0; i < n; i++) { s = s * 1664525u + 1013904223u; g[i] = ((s >> 8) / 16777216.f - 0.5f) * 1e-2f; }
    unet_params_d2h(b, p0);
    unet_grad_h2d(a, g); unet_grad_h2d(b, g);
    unet_muon(a, 0.02f, 0.95f, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.1f, 1); unet_muon(b, 0.02f, 0.95f, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.1f, 1);
    unet_params_d2h(a, pa); unet_params_d2h(b, pb);
    size_t same = 0, cf = 0, cff = 0, other = 0, tiny = 0; double wf = 0, wff = 0;
    for (size_t i = 0; i < n; i++) {
        const double da = (double)pa[i] - p0[i], db = (double)pb[i] - p0[i];
        if (pa[i] == pb[i]) { same++; continue; }
        if (fabs(db) < 1e-9) { tiny++; continue; }
        const double r = da / db;
        if (fabs(r - F) < 1e-3) { cf++; wf = fmax(wf, fabs(r - F) / F); }
        else if (fabs(r - F * F2) < 1e-3) { cff++; wff = fmax(wff, fabs(r - F * F2) / (F * F2)); }
        else other++;
    }
    const size_t C = 48;   /* wo C x C + w2 C x F_mlp: at least C x C elements in the F x F2 class */
    printf("%s: %zu identical, %zu at F (worst rel %.1e), %zu at F x F2 (worst rel %.1e), %zu other, %zu tiny, of %zu\n", ext ? "ext (normuon)" : "plain",
           same, cf, wf, cff, wff, other, tiny, n);
    CHECK(cf > 0 && cff >= C * C, "both multiplier classes present");
    CHECK(other == 0, "%zu parameters changed by another ratio", other);
    CHECK(tiny < n / 1000, "too many tiny reference steps");
    free(g); free(p0); free(pa); free(pb); unet_free(a); unet_free(b);
}

int main(void) {
    if (nn_init(0)) return 1;
    run(0);
    run(1);
    if (fails) { printf("test_attnlr: %d FAILED\n", fails); return 1; }
    printf("test_attnlr: OK\n");
    return 0;
}
