/* the bottleneck attention stage (unet_cfg.attn_blocks) in the U-Net, production storage (MX-fp4 activations / gradients, SR):
   - a fresh stage is the identity: a hybrid (2 blocks, PEG, 2 registers) that takes a conv net's trunk (unet_init_from_map) gives
     its logits bit for bit and its trunk gradients up to summation order (relative L2 < 1e-5; gy records off, below);
   - with random out projections (UFSM_ATTN_NONZERO) the stage changes the logits, every stage tensor gets a finite nonzero gradient,
     and a checkpoint round trip (unet_peek / unet_load) restores the configuration and the logits bit for bit. */
#include "unet.h"
#include "nn_lp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static uint64_t rs = 4242;
static float frand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 11) * 0x1.0p-53 * 2 - 1); }
static double rel_l2(const float *a, const float *b, size_t n) { double e = 0, r = 0; for (size_t i = 0; i < n; i++) { double d = (double)a[i] - b[i]; e += d * d; r += (double)b[i] * b[i]; } return sqrt(e / (r > 0 ? r : 1)); }
static float *logits_f32(unet *u, const float *lg, size_t n) {
    float *h = malloc(n * 4);
    if (unet_logits_h16(u)) { uint16_t *t = malloc(n * 2); nn_d2h(t, lg, n * 2); for (size_t i = 0; i < n; i++) { uint16_t v = t[i]; int e = (v >> 10) & 31, m = v & 1023; float f = e == 0 ? ldexpf((float)m, -24) : e == 31 ? INFINITY : ldexpf((float)(m | 1024), e - 25); h[i] = (v >> 15) ? -f : f; } free(t); }
    else nn_d2h(h, lg, n * 4);
    return h;
}
int main(void) {
    setenv("UFSM_ACT_MX4", "1", 0); setenv("UFSM_XIN_PREC", "8", 0); setenv("UFSM_GRAD_MX8", "1", 0); setenv("UFSM_GRAD_MX4", "1", 0);
    setenv("UFSM_F4_WGRAD", "1", 0); setenv("UFSM_DOWN_NORM", "1", 0);
    if (nn_init(0)) { printf("SKIP: no GPU\n"); return 0; }
    nn_set_tf32(1); nn_set_f16(1); nn_set_prec_policy("all=fp4:fp4:fp4,enc0.c1=fp16"); nn_set_gn_stored(1); nn_set_sr(1);
    unet_cfg c0 = {0}; c0.nlev = 3; c0.widths[0] = 16; c0.widths[1] = 32; c0.widths[2] = 96; c0.cin = 1; c0.cout = 16; c0.G = 8; c0.down_norm = 1;
    unet_cfg c1 = c0; c1.attn_blocks = 2; c1.attn_pe = 0; c1.attn_reg = 2;
    unet *u0 = unet_create(&c0); unet_init(u0, 5);
    char path[] = "/tmp/ufsm_test_hybrid_XXXXXX"; int fd = mkstemp(path); if (fd >= 0) close(fd);
    if (unet_save(u0, path, 0, nullptr)) { printf("cannot save\n"); return 1; }
    unet *u1 = unet_create(&c1); unet_init(u1, 9);
    int rows[16]; for (int i = 0; i < 16; i++) rows[i] = i;
    if (unet_init_from_map(u1, path, rows) < 0) { printf("init_from failed\n"); return 1; }
    const size_t n0 = unet_nparams(u0), n1 = unet_nparams(u1);
    printf("conv net %zu params, hybrid %zu (+%zu in the attention stage)\n", n0, n1, n1 - n0);
    const int P = 32; const size_t p3 = (size_t)P * P * P, nl = p3 * 16; const shape5 xs = {1, 1, P, P, P};
    float *xh = malloc(p3 * 4); for (size_t i = 0; i < p3; i++) xh[i] = frand();
    float *x = nn_malloc(p3 * 4); nn_h2d(x, xh, p3 * 4);
    float *gh = malloc(nl * 4); for (size_t i = 0; i < nl; i++) gh[i] = frand() * 1e-3f;
    float *gd = nn_malloc(nl * 4);
    int bad = 0;
    /* identity. The GN+SiLU backward's gy records (the fp4 weight gradient's pre-pass, matched by gy pointer) are off for this part: the
       stage hands the bottleneck block a gradient in another buffer, so its conv2 weight gradient runs the pre-pass (as after a residual
       block's identity add) with its own SR draws; without records both nets run the pre-pass with the same draws */
    lp_set_gyrec_off(1);
    nn_set_sr_step(100); const float *l = unet_forward(u0, x, xs, 1); float *a = logits_f32(u0, l, nl);
    nn_h2d(gd, gh, nl * 4); unet_zero_grad(u0); nn_set_sr_step(200); unet_backward(u0, gd);
    float *ga = malloc(n0 * 4); unet_grad_d2h(u0, ga);
    nn_set_sr_step(100); l = unet_forward(u1, x, xs, 1); float *b = logits_f32(u1, l, nl);
    nn_h2d(gd, gh, nl * 4); unet_zero_grad(u1); nn_set_sr_step(200); unet_backward(u1, gd);
    float *gb = malloc(n1 * 4); unet_grad_d2h(u1, gb);
    lp_set_gyrec_off(-1);
    double el = rel_l2(b, a, nl), eg = rel_l2(gb, ga, n0);
    printf("identity stage: logits rel L2 %.2e, trunk gradient rel L2 %.2e%s\n", el, eg, el < 1e-5 && eg < 1e-5 ? "" : "  FAIL"); bad += !(el < 1e-5 && eg < 1e-5);
    double sg = 0; int fin = 1; for (size_t i = n0; i < n1; i++) { sg += fabs(gb[i]); fin &= isfinite(gb[i]); }
    printf("identity stage: |stage gradient| sum %.3e, finite %d%s\n", sg, fin, sg > 0 && fin ? "" : "  FAIL"); bad += !(sg > 0 && fin);
    /* non-trivial stage */
    setenv("UFSM_ATTN_NONZERO", "1", 1);
    unet *u2 = unet_create(&c1); unet_init(u2, 11); unsetenv("UFSM_ATTN_NONZERO");
    if (unet_init_from_map(u2, path, rows) < 0) { printf("init_from failed\n"); return 1; }
    nn_set_sr_step(100); l = unet_forward(u2, x, xs, 1); float *c = logits_f32(u2, l, nl);
    double ec = rel_l2(c, a, nl);
    printf("random stage: logits differ from the conv net by rel L2 %.2e%s\n", ec, ec > 1e-2 ? "" : "  FAIL"); bad += !(ec > 1e-2);
    nn_h2d(gd, gh, nl * 4); unet_zero_grad(u2); nn_set_sr_step(200); unet_backward(u2, gd);
    float *gc = malloc(n1 * 4); unet_grad_d2h(u2, gc);
    /* the stage's tensors (offsets from the layout: PEG w, b, registers, then per block 14 tensors) all nonzero and finite */
    {
        const int C = 96, F = (8 * C / 3 + 31) / 32 * 32; size_t off = n0; int zero = 0, nonfin = 0, nt = 0;
        size_t sz[3 + 28] = {(size_t)C * 27, (size_t)C, (size_t)2 * C};
        for (int k = 0; k < 2; k++) { size_t t[14] = {C, C, (size_t)3 * C * C, (size_t)3 * C, C, C, (size_t)C * C, C, C, C, (size_t)2 * F * C, (size_t)2 * F, (size_t)C * F, C}; for (int j = 0; j < 14; j++) sz[3 + 14 * k + j] = t[j]; }
        for (int j = 0; j < 31; j++) { double s = 0; for (size_t i = 0; i < sz[j]; i++) { s += fabs(gc[off + i]); nonfin += !isfinite(gc[off + i]); } zero += s == 0; off += sz[j]; nt++; }
        printf("random stage: %d of %d stage tensors with zero gradient, %d non-finite, layout %s%s\n", zero, nt, nonfin, off == n1 ? "ok" : "MISMATCH", zero || nonfin || off != n1 ? "  FAIL" : "");
        bad += zero || nonfin || off != n1;
    }
    /* checkpoint round trip */
    if (unet_save(u2, path, 7, nullptr)) { printf("cannot save\n"); return 1; }
    unet_cfg pc; int st; if (unet_peek(path, &pc, &st)) { printf("peek failed\n"); return 1; }
    unet *u3 = unet_create(&pc); int ls = unet_load(u3, path);
    nn_set_sr_step(100); l = unet_forward(u3, x, xs, 1); float *d = logits_f32(u3, l, nl);
    nn_set_sr_step(100); l = unet_forward(u2, x, xs, 1); float *e = logits_f32(u2, l, nl);
    size_t neq = 0; for (size_t i = 0; i < nl; i++) neq += memcmp(d + i, e + i, 4) != 0;
    printf("checkpoint: step %d, attn_blocks %d pe %d reg %d mlp %d, logits differ in %zu%s\n", ls, pc.attn_blocks, pc.attn_pe, pc.attn_reg, pc.attn_mlp, neq,
           ls == 7 && pc.attn_blocks == 2 && pc.attn_reg == 2 && !neq ? "" : "  FAIL");
    bad += !(ls == 7 && pc.attn_blocks == 2 && pc.attn_reg == 2 && !neq);
    unlink(path);
    printf(bad ? "test_hybrid_identity: %d FAILED\n" : "test_hybrid_identity: all passed\n", bad);
    return bad != 0;
}
