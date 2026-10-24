/* the bottleneck attention stage (unet_cfg.attn_blocks) in the U-Net, production storage (MX-fp4 activations / gradients, SR):
   - a fresh stage is the identity: a hybrid (2 blocks, PEG, 2 registers) that takes a conv net's trunk (unet_init_from_map) gives
     its logits bit for bit and its trunk gradients up to summation order (relative L2 < 1e-5; gy records off, below);
   - with random out projections (UFSM_ATTN_NONZERO) the stage changes the logits, every stage tensor gets a finite nonzero gradient,
     and a checkpoint round trip (unet_peek / unet_load) restores the configuration and the logits bit for bit;
   - unet_set_attn_keep (no recompute in the backward) gives the recomputing backward's gradients, with each attention core. */
#include "unet.h"
#include "nn_lp.h"
#include "attn.h"
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
    char path0[] = "/tmp/ufsm_test_hybrid0_XXXXXX"; { int f0 = mkstemp(path0); if (f0 >= 0) close(f0); }   /* (the conv net, kept: path is reused) */
    if (unet_save(u0, path0, 0, nullptr)) { printf("cannot save\n"); return 1; }
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
    {   /* a warm start (unet_init_from_map) of a hybrid from a hybrid checkpoint copies the attention stage: the same logits */
        unet *u4 = unet_create(&c1); unet_init(u4, 99);
        const int n = unet_init_from_map(u4, path, nullptr);
        nn_set_sr_step(100); l = unet_forward(u4, x, xs, 1); float *f = logits_f32(u4, l, nl);
        size_t nd = 0; for (size_t i = 0; i < nl; i++) nd += memcmp(f + i, e + i, 4) != 0;
        printf("warm start: %d layers copied, logits differ in %zu%s\n", n, nd, nd ? "  FAIL" : ""); bad += nd != 0;
    }
    {   /* unet_set_attn_keep: the blocks' training forwards kept for the backward (no recompute): the same gradients as the recomputing
           backward of the random stage (up to atomic summation order), with each attention core */
        for (int core = 0; core < 3; core++) {
            at_set_core(core); double ek = 0; float *gk[2];
            for (int keep = 0; keep < 2; keep++) {
                unet_set_attn_keep(keep); unet *u5 = unet_create(&pc); unet_load(u5, path);
                nn_set_sr_step(100); unet_forward(u5, x, xs, 1);
                nn_h2d(gd, gh, nl * 4); unet_zero_grad(u5); nn_set_sr_step(200); unet_backward(u5, gd);
                gk[keep] = malloc(n1 * 4); unet_grad_d2h(u5, gk[keep]); unet_free(u5);
            }
            unet_set_attn_keep(0); ek = rel_l2(gk[1], gk[0], n1);
            printf("keep forwards (core %d): gradient rel L2 vs recompute %.2e%s\n", core, ek, ek < 1e-5 ? "" : "  FAIL"); bad += !(ek < 1e-5);
            free(gk[0]); free(gk[1]);
        }
        at_set_core(0);
    }
    /* the scroll-coordinate stages (attn_axis radial, attn_pe scroll, attn_repeat): an axis record per window (outside the window, drifting
       along z), then (a) fresh stages are the identity (logits bit for bit, trunk gradients as above), (b) a random tied stage (1 block run
       twice) equals the untied 2-block stage holding the same block twice: the same logits, and its block gradient is the sum of the two
       blocks' gradients, (c) the checkpoint keeps the configuration */
    {
        float axr[UNET_AXR(32)]; axr[0] = 0; axr[1] = 1; axr[2] = 32; axr[3] = 1;
        for (int z = 0; z < 32; z++) { axr[4 + 2 * z] = 41.5f - 0.2f * z; axr[5 + 2 * z] = -6.f + 0.15f * z; }
        struct { int axis, pe, rep; const char *nm; } V[] = {{1, 3, 1, "radial + scroll RoPE"}, {1, 2, 2, "radial, no PE, repeat 2"}, {0, 3, 2, "full + scroll RoPE, repeat 2"}};
        for (int vi = 0; vi < 3; vi++) {
            unet_cfg cv = c0; cv.attn_blocks = 2; cv.attn_pe = V[vi].pe; cv.attn_reg = 0; cv.attn_axis = V[vi].axis; cv.attn_repeat = V[vi].rep;
            unet *uv = unet_create(&cv); unet_init(uv, 21 + vi);
            if (unet_init_from_map(uv, path0, rows) < 0) { printf("init_from failed\n"); return 1; }
            unet_set_attn_axis(uv, axr, 1, UNET_AXR(32));
            lp_set_gyrec_off(1);
            nn_set_sr_step(100); l = unet_forward(uv, x, xs, 1); float *bv = logits_f32(uv, l, nl);
            nn_h2d(gd, gh, nl * 4); unet_zero_grad(uv); nn_set_sr_step(200); unet_backward(uv, gd);
            const size_t nv = unet_nparams(uv); float *gv = malloc(nv * 4); unet_grad_d2h(uv, gv);
            lp_set_gyrec_off(-1);
            int Q = 0, Nr = 0; double cov = 0; const int hr = unet_attn_rays(uv, &Q, &Nr, &cov);
            const double e1 = rel_l2(bv, a, nl), e2 = rel_l2(gv, ga, n0);
            printf("%s: identity logits rel L2 %.2e, trunk gradient %.2e", V[vi].nm, e1, e2);
            if (!hr) printf(", %d rays x %d slots, coverage %.3f", Q, Nr, cov);
            printf("%s\n", e1 < 1e-5 && e2 < 1e-5 ? "" : "  FAIL"); bad += !(e1 < 1e-5 && e2 < 1e-5);
            free(bv); free(gv); unet_free(uv);
        }
        /* (b) tied vs untied, random stage */
        unet_cfg ct = c0; ct.attn_blocks = 1; ct.attn_pe = 3; ct.attn_axis = 1; ct.attn_repeat = 2;
        unet_cfg cu = c0; cu.attn_blocks = 2; cu.attn_pe = 3; cu.attn_axis = 1;
        setenv("UFSM_ATTN_NONZERO", "1", 1);
        unet *ut = unet_create(&ct); unet_init(ut, 31); unsetenv("UFSM_ATTN_NONZERO");
        unet *uu = unet_create(&cu); unet_init(uu, 32);
        const size_t nt = unet_nparams(ut), nu = unet_nparams(uu), nb = nt - n0;
        float *pt = malloc(nu * 4); nn_d2h(pt, unet_live_params(ut), nt * 4); memcpy(pt + nt, pt + n0, nb * 4);
        nn_h2d((float *)unet_live_params(uu), pt, nu * 4);
        unet_set_attn_axis(ut, axr, 1, UNET_AXR(32)); unet_set_attn_axis(uu, axr, 1, UNET_AXR(32));
        nn_set_sr_step(100); l = unet_forward(ut, x, xs, 1); float *lt = logits_f32(ut, l, nl);
        nn_h2d(gd, gh, nl * 4); unet_zero_grad(ut); nn_set_sr_step(200); unet_backward(ut, gd); float *gt = malloc(nt * 4); unet_grad_d2h(ut, gt);
        nn_set_sr_step(100); l = unet_forward(uu, x, xs, 1); float *lu = logits_f32(uu, l, nl);
        nn_h2d(gd, gh, nl * 4); unet_zero_grad(uu); nn_set_sr_step(200); unet_backward(uu, gd); float *gu = malloc(nu * 4); unet_grad_d2h(uu, gu);
        size_t nd = 0; for (size_t i = 0; i < nl; i++) nd += memcmp(lt + i, lu + i, 4) != 0;
        for (size_t i = 0; i < nb; i++) gu[n0 + i] += gu[nt + i];
        const double eb = rel_l2(gt + n0, gu + n0, nb), etr = rel_l2(gt, gu, n0), dl = rel_l2(lt, a, nl);
        double sg = 0; int fin = 1; for (size_t i = n0; i < nt; i++) { sg += fabs(gt[i]); fin &= isfinite(gt[i]); }
        printf("tied (1 block x 2) vs untied (2 equal blocks), radial + scroll: logits differ in %zu (vs the conv net rel L2 %.2e), block gradient vs the sum rel L2 %.2e, trunk %.2e, |block grad| %.2e finite %d%s\n",
               nd, dl, eb, etr, sg, fin, !nd && dl > 1e-2 && eb < 1e-4 && etr < 1e-4 && sg > 0 && fin ? "" : "  FAIL");
        bad += !(!nd && dl > 1e-2 && eb < 1e-4 && etr < 1e-4 && sg > 0 && fin);
        /* (c) checkpoint */
        if (unet_save(ut, path, 3, nullptr)) { printf("cannot save\n"); return 1; }
        unet_cfg pc2; int st2; if (unet_peek(path, &pc2, &st2)) { printf("peek failed\n"); return 1; }
        unet *ul = unet_create(&pc2); unet_load(ul, path); unet_set_attn_axis(ul, axr, 1, UNET_AXR(32));
        nn_set_sr_step(100); l = unet_forward(ul, x, xs, 1); float *ll = logits_f32(ul, l, nl);
        size_t nd2 = 0; for (size_t i = 0; i < nl; i++) nd2 += memcmp(ll + i, lt + i, 4) != 0;
        printf("checkpoint: attn_axis %d attn_pe %d attn_repeat %d, logits differ in %zu%s\n", pc2.attn_axis, pc2.attn_pe, pc2.attn_repeat, nd2,
               pc2.attn_axis == 1 && pc2.attn_pe == 3 && pc2.attn_repeat == 2 && !nd2 ? "" : "  FAIL");
        bad += !(pc2.attn_axis == 1 && pc2.attn_pe == 3 && pc2.attn_repeat == 2 && !nd2);
        free(pt); free(lt); free(gt); free(lu); free(gu); free(ll); unet_free(ut); unet_free(uu); unet_free(ul);
    }
    unlink(path); unlink(path0);
    printf(bad ? "test_hybrid_identity: %d FAILED\n" : "test_hybrid_identity: all passed\n", bad);
    return bad != 0;
}
