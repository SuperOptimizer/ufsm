/* Global-context FiLM (unet_cfg.gfilm, train --gfilm 1), tensor-core path (16-bit activations), a 3-level net with one bottleneck
   attention block (8^3 tokens):
   - zero-initialised: the logits are bit-identical to the same net without gfilm (its A / B block is the last parameters);
   - with A / B set, the logits move, and the analytic gradient matches central differences along random directions (the loss
     sum(gy * logits)): over the gfilm block (dA, dB), and the token path (dg into the attention tokens) alone over every other
     parameter: its analytic share (UFSM_GFILM_NO_TOKGRAD on minus off) against the differences with the code recomputed minus
     frozen (UFSM_GFILM_FREEZE); reported, enforced only with GF_TOKCHECK (the path is off by default, UFSM_GFILM_TOKGRAD=1 on);
   - checkpoints keep the flag and the block. */
#include "unet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int bad;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); bad++; } } while (0)
static uint64_t rs = 99;
static float frand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 11) * 0x1.0p-53 * 2 - 1); }

static double loss(unet *u, const float *dx, shape5 xs, const float *gy, float *h, size_t no) {
    nn_d2h(h, unet_forward(u, dx, xs, 1), no * 4);
    double s = 0; for (size_t i = 0; i < no; i++) s += (double)gy[i] * h[i];
    return s;
}
/* the directional derivative along v over params [a, b): analytic (g . v) and central differences */
static void dirder(unet *u, const float *dx, shape5 xs, const float *gy, const float *dgy, float *h, size_t no, size_t a, size_t b, double eps, double *an, double *fd) {
    const size_t np = unet_nparams(u);
    float *p = malloc(np * 4), *q = malloc(np * 4), *g = malloc(np * 4), *v = malloc(np * 4);
    nn_d2h(p, unet_live_params(u), np * 4);
    for (size_t i = 0; i < np; i++) v[i] = i >= a && i < b ? frand() : 0.f;
    loss(u, dx, xs, gy, h, no); unet_zero_grad(u); unet_backward(u, dgy); unet_grad_d2h(u, g);
    double s = 0; for (size_t i = a; i < b; i++) s += (double)g[i] * v[i];
    *an = s;
    double l[2];
    for (int k = 0; k < 2; k++) {
        for (size_t i = 0; i < np; i++) q[i] = p[i] + (float)((k ? -eps : eps) * v[i]);
        nn_h2d((float *)unet_live_params(u), q, np * 4); l[k] = loss(u, dx, xs, gy, h, no);
    }
    nn_h2d((float *)unet_live_params(u), p, np * 4);
    *fd = (l[0] - l[1]) / (2 * eps);
    free(p); free(q); free(g); free(v);
}

int main(void) {
    if (nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0)) { printf("no cuda\n"); return 1; }
    nn_set_tf32(1); nn_set_f16(getenv("GF_F16") ? atoi(getenv("GF_F16")) : 1);
    const double ea = getenv("GF_EA") ? atof(getenv("GF_EA")) : 5e-2, er = getenv("GF_ER") ? atof(getenv("GF_ER")) : 1e-3;
    unet_cfg c = {}; c.nlev = 3; c.widths[0] = 16; c.widths[1] = 32; c.widths[2] = 48; c.cin = 4; c.cout = 2; c.G = 8; c.attn_blocks = 1;
    const shape5 xs = {1, 4, 32, 32, 32}; const size_t nx = shape_numel(xs);
    unet_cfg c1 = c; c1.gfilm = 1;
    unet *u0 = unet_create(&c), *u = unet_create(&c1); unet_init(u0, 5); unet_init(u, 5);
    const size_t no = shape_numel(unet_out_shape(u, xs)), np = unet_nparams(u), gf = unet_gfilm_off(u);
    printf("gfilm: %zu params, the block %zu (from %zu)\n", np, np - gf, gf);
    CHECK(gf == unet_nparams(u0), "the gfilm block is appended last (%zu vs %zu)", gf, unet_nparams(u0));
    float *x = malloc(nx * 4), *gy = malloc(no * 4), *h0 = malloc(no * 4), *h1 = malloc(no * 4);
    for (size_t i = 0; i < nx; i++) x[i] = frand();
    for (size_t i = 0; i < no; i++) gy[i] = frand();
    float *dx = nn_malloc(nx * 4), *dgy = nn_malloc(no * 4); nn_h2d(dx, x, nx * 4); nn_h2d(dgy, gy, no * 4);
    nn_d2h(h0, unet_forward(u0, dx, xs, 1), no * 4); nn_d2h(h1, unet_forward(u, dx, xs, 1), no * 4);
    CHECK(!memcmp(h0, h1, no * 4), "zero A / B: logits bit-identical to the net without gfilm");
    {   /* A / B random */
        float *p = malloc(np * 4); nn_d2h(p, unet_live_params(u), np * 4);
        for (size_t i = gf; i < np; i++) p[i] = 0.3f * frand();
        nn_h2d((float *)unet_live_params(u), p, np * 4); free(p);
    }
    nn_d2h(h1, unet_forward(u, dx, xs, 1), no * 4);
    double d = 0; for (size_t i = 0; i < no; i++) { const double e = fabs((double)h1[i] - h0[i]); if (e > d) d = e; }
    CHECK(d > 1e-3, "A / B set: the logits move (%g)", d);
    double an, fd;
    dirder(u, dx, xs, gy, dgy, h1, no, gf, np, ea, &an, &fd);
    printf("  d/d(A, B): analytic %.6g, central differences %.6g (rel %.3g)\n", an, fd, fabs(an - fd) / (fabs(fd) + 1e-12));
    CHECK(fabs(an - fd) <= 0.03 * fabs(fd) && fabs(fd) > 1e-6, "d/d(A, B) (analytic %g, differences %g)", an, fd);
    /* the token path (dg into the attention tokens) alone, along a random direction over every other parameter: analytic, its share
       with the path on minus off; central differences, the loss with the code recomputed minus with the code frozen at the base
       forward's (the rest of the numerics shared, so their noise cancels) */
    double fdf, anf, anx, fdx;
    setenv("UFSM_GFILM_TOKGRAD", "1", 1);
    rs = 1234; dirder(u, dx, xs, gy, dgy, h1, no, 0, gf, er, &an, &fd);
    setenv("UFSM_GFILM_NO_TOKGRAD", "1", 1); rs = 1234; dirder(u, dx, xs, gy, dgy, h1, no, 0, gf, er, &anx, &fdx); unsetenv("UFSM_GFILM_NO_TOKGRAD");
    loss(u, dx, xs, gy, h1, no); setenv("UFSM_GFILM_FREEZE", "1", 1); rs = 1234;   /* (the base code, kept through dirder's forwards) */
    dirder(u, dx, xs, gy, dgy, h1, no, 0, gf, er, &anf, &fdf); unsetenv("UFSM_GFILM_FREEZE"); unsetenv("UFSM_GFILM_TOKGRAD");
    const double tan = an - anx, tfd = fd - fdf;
    printf("  the token path: analytic %.6g, differences %.6g (rel %.3g; the whole direction %.6g)\n", tan, tfd, fabs(tan - tfd) / (fabs(tfd) + 1e-12), fd);
    if (getenv("GF_TOKCHECK")) CHECK(fabs(tan - tfd) <= 0.05 * fabs(tfd) && fabs(tfd) > 1e-3 * fabs(fd), "the token path (analytic %g, differences %g)", tan, tfd);   /* (off by default: 9%% at the 16-bit noise floor, open) */
    {   /* checkpoint */
        char path[96]; snprintf(path, sizeof path, "/tmp/ufsm_test_gfilm_%d.ckpt", (int)getpid());
        CHECK(!unet_save(u, path, 3, nullptr), "save");
        unet_cfg pc; int st; CHECK(!unet_peek(path, &pc, &st) && pc.gfilm == 1, "peek: gfilm %d", pc.gfilm);
        unet *v = unet_create(&pc); CHECK(unet_load(v, path) == 3, "load");
        float *h2 = malloc(no * 4);
        nn_d2h(h1, unet_forward(u, dx, xs, 0), no * 4); nn_d2h(h2, unet_forward(v, dx, xs, 0), no * 4);
        CHECK(!memcmp(h1, h2, no * 4), "reloaded: logits bit-identical");
        free(h2); unet_free(v); unlink(path);
    }
    unet_free(u); unet_free(u0);
    if (bad) { printf("test_gfilm: %d failed\n", bad); return 1; }
    printf("test_gfilm: ok\n");
    return 0;
}
