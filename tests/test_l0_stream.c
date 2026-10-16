/* The level-0 slab stream (stage S3, UFSM_L0_STREAM / unet_set_l0_stream): the level-0 ops per slab in a skewed schedule on the whole
   tensors must give the unstreamed slab kernel path's bits. The production chain (MX-fp4 activations and gradients, fp8 stem, MX-fp4
   input, recompute 1) in the plain layout, 7 levels 16,16,32,64,128,192,256, slab-local GroupNorm (UFSM_L0_SLAB_KERN=1); slabs of 16
   and 32 planes, down_norm 0 / 1; two training steps with a validation forward between them: the logits, gout[1] after dec[0]'s
   backward, d/d downo[0] and the skip gradient bit for bit, the parameter gradients per part up to the weight gradients' atomic
   summation order, the same seed count. Run with UFSM_L0_STREAM_POISON=1 the transient gradients' planes are overwritten (NaN scales)
   as soon as the schedule declares them dead: the same bits prove the liveness the slab rings (S4) rely on.
   env UFSM_TEST_P: the window (default 128) */
#include "nn.h"
#include "nn_lp.h"
#include "unet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static uint64_t rs = 4242;
static float frand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 11) * 0x1.0p-53 * 2 - 1); }
static float *dev_rand(size_t n, float scale) { float *h = malloc(n * 4); for (size_t i = 0; i < n; i++) h[i] = frand() * scale; float *d = nn_malloc(n * 4); nn_h2d(d, h, n * 4); free(h); return d; }
static double rel(const float *a, const float *r, size_t lo, size_t hi) {
    double d2 = 0, r2 = 0;
    for (size_t i = lo; i < hi; i++) { const double d = (double)a[i] - r[i]; d2 += d * d; r2 += (double)r[i] * r[i]; }
    return r2 > 0 ? sqrt(d2 / r2) : d2 > 0 ? 1.0 : 0.0;
}
static size_t blk(int ci, int co) { return (size_t)co * ci * 27 + co + 2 * co + (size_t)co * co * 27 + co + 2 * co; }   /* a two-conv block's parameters */

int main(void) {
    const int P = getenv("UFSM_TEST_P") ? atoi(getenv("UFSM_TEST_P")) : 128;
    nn_init(0);
    nn_set_tf32(1); nn_set_act_bf16(1); nn_set_grad_bf16(1); nn_set_f16(1); nn_set_grad_scale(1024); nn_set_sr(1); nn_set_gn_stored(1);
    nn_set_prec(3); if (nn_set_prec_policy("all=fp4:fp4:fp4,enc0.c1=fp16")) return 2;
    setenv("UFSM_F4_WGRAD", "1", 1);
    unet_set_act_mx4(1); unet_set_grad_mx8(1); unet_set_grad_mx4(1); unet_set_input_prec(4); unet_set_recompute(1); unet_set_chunk_up(1); unet_set_lean(0);
    unet_set_share_enc_a1(0); unet_set_a0_share(0); unet_set_l0_3buf(0); unet_set_xin_offload(0); unet_set_l0_slab_kern(1);
    unet_l0_stream_dump(1);
    int bad = 0;
    for (int S = 16; S <= 32; S *= 2) for (int dn = 0; dn < 2; dn++) {
        unet_cfg cfg = {7, {16, 16, 32, 64, 128, 192, 256}, 4, 2, 8, dn}; cfg.l0_norm = 4; cfg.l0_slab = S;
        const shape5 xs = {1, 4, P, P, P};
        const size_t S3 = (size_t)P * P * P, nx = 4 * S3, no = 2 * S3;
        void *x[3]; void *gl[2];
        for (int k = 0; k < 3; k++) { float *f = dev_rand(nx, 1.f); x[k] = nn_malloc(nx * 2); nn_f32_to_h16(f, nx, x[k], 1.f); nn_free(f); }
        for (int k = 0; k < 2; k++) { float *f = dev_rand(no, 1.f); gl[k] = nn_malloc(no * 2); nn_f32_to_h16(f, no, gl[k], nn_get_grad_scale()); nn_free(f); }
        unet *v = unet_create(&cfg); unet_init(v, 77); const size_t np = unet_nparams(v);
        uint8_t *lg[4][3]; size_t lgb = 0; uint8_t *dmp[4][2][4]; size_t dmb[4] = {0, 0, 0, 0}; float *gg[4][2]; unsigned ctr[4][2]; double ms[4] = {0, 0, 0, 0};
        /* stage S4: the recomputing stream with the level-0 tensors in VMM slab rings, where the slabs lie on the 2 MB mapping granularity */
        const int nst = (size_t)S * P * P % ((size_t)2 << 20) == 0 && nn_ring_gran() ? 4 : 3;
        for (int st = 0; st < nst; st++) {   /* unstreamed, streamed, streamed and recomputing, recomputing in rings */
            unet_set_l0_stream(st < 3 ? st : 2); unet_set_l0_ring(st == 3);
            unet_build(v, xs, 1);
            for (int k = 0; k < 2; k++) {
                nn_sync(); const double t0 = now_ms();   /* (the second step's forward + backward: the timing) */
                nn_set_sr_step(5000 + 100 * k); const float *l = unet_forward_x(v, x[k], xs, 1, 1); unet_zero_grad(v);
                lgb = no * (unet_logits_h16(v) ? 2 : 4);
                lg[st][k] = malloc(lgb); nn_sync(); nn_d2h(lg[st][k], l, lgb);
                nn_set_sr_step(5050 + 100 * k); unet_backward_x(v, gl[k], 1); ctr[st][k] = nn_sr_ctr();
                nn_sync(); if (k) ms[st] = now_ms() - t0 - 0.0;
                gg[st][k] = malloc(np * 4); unet_grad_d2h(v, gg[st][k]);
                for (int d = 0; d < 4; d++) { const void *p = unet_l0_stream_dumped(d, &dmb[d]); dmp[st][k][d] = malloc(dmb[d]); nn_sync(); nn_d2h(dmp[st][k][d], p, dmb[d]); }
                if (!k) {   /* a validation forward between the steps */
                    nn_set_sr_step(5025); const float *lv = unet_forward_x(v, x[2], xs, 0, 1);
                    lg[st][2] = malloc(lgb); nn_sync(); nn_d2h(lg[st][2], lv, lgb);
                }
            }
        }
        const char *e = nn_check();
        for (int m = 1; m < nst; m++) {   /* each stream mode against the unstreamed run */
            size_t lneq = 0; for (int k = 0; k < 3; k++) for (size_t i = 0; i < lgb; i++) lneq += lg[0][k][i] != lg[m][k][i];
            size_t dneq[4] = {0, 0, 0, 0}; for (int k = 0; k < 2; k++) for (int d = 0; d < 4; d++) for (size_t i = 0; i < dmb[d]; i++) dneq[d] += dmp[0][k][d][i] != dmp[m][k][d][i];
            /* parameter gradients per part: enc[0], down0, dec[0], the head, all (unet_create's layout: encoders, downs, decoders, head) */
            const int *w = cfg.widths, L = cfg.nlev;
            size_t encs = 0; for (int i = 0; i < L; i++) encs += blk(i ? w[i - 1] : cfg.cin, w[i]);
            const size_t he = np - (dn ? (size_t)2 * (w[0] + w[1] + w[2] + w[3] + w[4] + w[5]) : 0), hs = he - ((size_t)cfg.cout * w[0] + cfg.cout), d0 = hs - blk(w[0] + w[1], w[0]);
            double worst[5] = {0, 0, 0, 0, 0};
            for (int k = 0; k < 2; k++) {
                const double r[5] = {rel(gg[m][k], gg[0][k], 0, blk(cfg.cin, w[0])), rel(gg[m][k], gg[0][k], encs, encs + (size_t)w[0] * w[0] * 27 + w[0]),
                                     rel(gg[m][k], gg[0][k], d0, hs), rel(gg[m][k], gg[0][k], hs, he), rel(gg[m][k], gg[0][k], 0, np)};
                for (int j = 0; j < 5; j++) if (r[j] > worst[j]) worst[j] = r[j];
            }
            int fin = 1; for (int k = 0; k < 2; k++) for (size_t i = 0; i < np; i++) if (!isfinite(gg[m][k][i])) { fin = 0; break; }
            const int ok = !e && !lneq && !dneq[0] && !dneq[1] && !dneq[2] && !dneq[3] && fin && worst[4] < 1e-5 && ctr[0][0] == ctr[m][0] && ctr[0][1] == ctr[m][1];
            printf("  P %d slab %d down_norm %d, stream %d%s%s: logits %zu of %zu bytes differ, gout[1] %zu, dec[0]'s skip gradient %zu, d/d downo[0] %zu, skip gradient %zu; parameter "
                   "gradients rel enc0 %.2g down0 %.2g dec0 %.2g head %.2g all %.2g%s; seeds %u/%u vs %u/%u; step %.0f vs %.0f ms (with the logits' d2h)%s%s\n",
                   P, S, dn, m < 3 ? m : 2, m == 2 ? " (recomputing)" : m == 3 ? " (recomputing, slab rings)" : "", getenv("UFSM_L0_STREAM_POISON") ? " (poison)" : "",
                   lneq, 3 * lgb, dneq[0], dneq[3], dneq[1], dneq[2], worst[0], worst[1], worst[2], worst[3], worst[4], fin ? "" : " (non-finite)",
                   ctr[m][0], ctr[m][1], ctr[0][0], ctr[0][1], ms[m], ms[0], e ? e : "", ok ? "" : "  FAIL");
            if (!ok) bad++;
        }
        for (int st = 0; st < nst; st++) for (int k = 0; k < 3; k++) free(lg[st][k]);
        for (int st = 0; st < nst; st++) for (int k = 0; k < 2; k++) { free(gg[st][k]); for (int d = 0; d < 4; d++) free(dmp[st][k][d]); }
        unet_set_l0_ring(0);
        for (int k = 0; k < 3; k++) nn_free(x[k]);
        for (int k = 0; k < 2; k++) nn_free(gl[k]);
        unet_free(v);
    }
    unet_set_l0_stream(-1); unet_set_l0_ring(-1);
    printf("level-0 slab stream: %s\n", bad ? "FAIL" : "PASS");
    return bad != 0;
}
