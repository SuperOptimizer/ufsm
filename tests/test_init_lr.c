/* train --init-lr-mult (unet_set_init_lr): only the parameters unet_init_from* copied from a donor take the multiplier.
   A 2-level net from a 3-level donor with --init-shift 1 (its stem reads another input count, so the stem stays fresh) and a fresh head:
   - the inherited ranges are exactly the copied tensors (count = the donor's level-1.. encoder / down / decoder tensors);
   - with multiplier 0, one Muon step and one AdamW step leave every inherited parameter bit-identical and change the fresh stem and head;
   - with multiplier 1 the inherited parameters do change (the split is not a freeze by accident);
   - a checkpoint round trip keeps the ranges (unet_load), so a resumed run still holds the inherited parameters at the multiplier. */
#include "unet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static uint64_t rs = 777;
static float frand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 11) * 0x1.0p-53 * 2 - 1); }
static int in_ranges(const size_t *r, int n, size_t i) { for (int k = 0; k < n; k++) if (i >= r[2 * k] && i < r[2 * k] + r[2 * k + 1]) return 1; return 0; }
/* one forward / backward with a random logit gradient, then the optimiser (muon: unet_muon, else unet_adamw) */
static void step(unet *u, const float *x, shape5 xs, size_t nl, int muon, int it) {
    unet_forward(u, x, xs, 1);
    float *gh = malloc(nl * 4); for (size_t i = 0; i < nl; i++) gh[i] = frand() * 1e-2f;
    float *gd = nn_malloc(nl * 4); nn_h2d(gd, gh, nl * 4);
    unet_zero_grad(u); unet_backward(u, gd);
    if (muon) unet_muon(u, 0.02f, 0.95f, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, it); else unet_adamw(u, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, it);
    nn_free(gd); free(gh);
}
/* parameters equal / different inside and outside the inherited ranges after one step */
static int check(unet *u, const float *x, shape5 xs, size_t nl, int muon, float mult, const char *what) {
    const size_t np = unet_nparams(u); int nr; const size_t *r = unet_inherited_ranges(u, &nr);
    float *a = malloc(np * 4), *b = malloc(np * 4);
    unet_set_init_lr(u, mult); unet_params_d2h(u, a); step(u, x, xs, nl, muon, 1); unet_params_d2h(u, b);
    size_t inh = 0, inh_moved = 0, fresh = 0, fresh_moved = 0;
    for (size_t i = 0; i < np; i++) { const int m = memcmp(&a[i], &b[i], 4) != 0; if (in_ranges(r, nr, i)) { inh++; inh_moved += m; } else { fresh++; fresh_moved += m; } }
    const int ok = mult == 0.f ? inh_moved == 0 && fresh_moved > fresh / 2 : inh_moved > inh / 2;
    printf("%-28s mult %g: inherited %zu (%zu moved), fresh %zu (%zu moved)%s\n", what, mult, inh, inh_moved, fresh, fresh_moved, ok ? "" : "  FAIL");
    free(a); free(b); return !ok;
}
int main(void) {
    if (nn_init(0)) { printf("SKIP: no GPU\n"); return 0; }
    nn_set_tf32(1);
    unet_cfg cd = {0}; cd.nlev = 3; cd.widths[0] = 8; cd.widths[1] = 16; cd.widths[2] = 32; cd.cin = 1; cd.cout = 4; cd.G = 4; cd.down_norm = 1;
    unet *d = unet_create(&cd); unet_init(d, 3);
    char path[] = "/tmp/ufsm_test_initlr_XXXXXX"; { int fd = mkstemp(path); if (fd >= 0) close(fd); }
    if (unet_save(d, path, 0, nullptr)) { printf("cannot save\n"); return 1; }
    unet_cfg cu = {0}; cu.nlev = 2; cu.widths[0] = 16; cu.widths[1] = 32; cu.cin = 2; cu.cout = 4; cu.G = 4; cu.down_norm = 1;
    const int P = 16; const size_t p3 = (size_t)P * P * P, nl = p3 * 4; const shape5 xs = {1, 2, P, P, P};
    float *xh = malloc(2 * p3 * 4); for (size_t i = 0; i < 2 * p3; i++) xh[i] = frand();
    float *x = nn_malloc(2 * p3 * 4); nn_h2d(x, xh, 2 * p3 * 4);
    int bad = 0;
    for (int muon = 1; muon >= 0; muon--) {
        unet *u = unet_create(&cu); unet_init(u, 11);
        unet_set_init_shift(1); const int nt = unet_init_from(u, path); unet_set_init_shift(0);
        if (nt <= 0) { printf("init_from failed\n"); return 1; }
        int nr; unet_inherited_ranges(u, &nr);
        printf("%s: %d tensors copied, %zu of %zu parameters inherited in %d ranges\n", muon ? "Muon" : "AdamW", nt, unet_inherited(u), unet_nparams(u), nr);
        bad += !(unet_inherited(u) > 0 && unet_inherited(u) < unet_nparams(u));
        bad += check(u, x, xs, nl, muon, 0.f, muon ? "Muon + AdamW step" : "AdamW step");
        /* checkpoint round trip: the ranges come back, the multiplier still holds the inherited parameters */
        char p2[] = "/tmp/ufsm_test_initlr2_XXXXXX"; { int fd = mkstemp(p2); if (fd >= 0) close(fd); }
        if (unet_save(u, p2, 1, nullptr)) { printf("cannot save\n"); return 1; }
        unet *v = unet_create(&cu);
        if (unet_load(v, p2) < 0) { printf("cannot load\n"); return 1; }
        int nv; const size_t *rv = unet_inherited_ranges(v, &nv), *ru = unet_inherited_ranges(u, &nr);
        const int same = nv == nr && !memcmp(rv, ru, (size_t)nr * 2 * sizeof *ru);
        printf("checkpoint round trip: %d ranges back%s\n", nv, same ? "" : "  FAIL"); bad += !same;
        bad += check(v, x, xs, nl, muon, 0.f, "resumed step");
        bad += check(v, x, xs, nl, muon, 1.f, "resumed step, no multiplier");
        unet_free(v); unet_free(u); unlink(p2);
    }
    unlink(path);
    printf(bad ? "FAILED\n" : "ok\n");
    return bad != 0;
}
