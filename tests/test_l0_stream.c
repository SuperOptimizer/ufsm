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
/* the stream head's callbacks (stage S4): the slabs' fp16 logits into a whole buffer, the logit gradient's slab over the slab's logits */
typedef struct { uint8_t *lgw; const uint8_t *gl; size_t hw, dhw; int cout; } headctx;
static int head_fwd(void *c, int z0, int z1, const void *lg) {
    const headctx *h = c; const size_t nz = (size_t)(z1 - z0) * h->hw * 2;
    for (int k = 0; k < h->cout; k++) nn_d2d(h->lgw + (size_t)k * h->dhw * 2 + (size_t)z0 * h->hw * 2, (const uint8_t *)lg + (size_t)k * nz, nz);
    return 0;
}
static int head_bwd(void *c, int z0, int z1, void *lg) {
    const headctx *h = c; const size_t nz = (size_t)(z1 - z0) * h->hw * 2;
    for (int k = 0; k < h->cout; k++) nn_d2d((uint8_t *)lg + (size_t)k * nz, h->gl + (size_t)k * h->dhw * 2 + (size_t)z0 * h->hw * 2, nz);
    return 0;
}
static float h2f(uint16_t h) {
    const uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 31, m = h & 1023;
    uint32_t b = !e ? (m ? 0 : s) : e == 31 ? s | 0x7f800000 | (m << 13) : s | ((e + 112) << 23) | (m << 13);
    if (!e && m) { float f = (float)m * 0x1p-24f; return h & 0x8000 ? -f : f; }
    float f; memcpy(&f, &b, 4); return f;
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
        uint8_t *lg[8][3]; size_t lgbs[8] = {0}; uint8_t *dmp[8][2][4]; size_t dmb[4] = {0, 0, 0, 0}; float *gg[8][2]; unsigned ctr[8][2]; double ms[8] = {0};
        /* stage S4: the recomputing stream with the level-0 tensors in VMM slab rings (3), where the slabs lie on the 2 MB mapping
           granularity; the stream head per slab (4: whole tensors, 5: in rings with dec[0].a2 and gout[0]) */
        const int ring = (size_t)S * P * P % ((size_t)2 << 20) == 0 && nn_ring_gran(), nst = 8;
        headctx hc = {nullptr, nullptr, (size_t)P * P, S3, cfg.cout};
        hc.lgw = nn_malloc(no * 2);
        /* arms 6 / 7: --gpu-input (the CT channel and radial records, nn_input_rad): the stream head with the whole input, and in rings with
           the input ring (rebuilt per slab) -- 7 against 6 bit for bit */
        void *ct[3]; for (int k = 0; k < 3; k++) { float *f = dev_rand(S3, 1.f); ct[k] = nn_malloc(S3 * 2); nn_f32_to_h16(f, S3, ct[k], 1.f); nn_free(f); }
        void *rec;
        {
            const size_t rb = NN_RAD_REC(P); uint8_t *h = calloc(1, rb); int32_t *hi = (int32_t *)h; float *hf = (float *)(h + 32);
            hi[0] = 3; hi[1] = 5; hi[2] = P; hi[3] = P; hi[4] = 1; hi[5] = 0 | 1 << 2 | 2 << 4 | 1 << 9;
            for (int z = 0; z < P; z++) { hf[z] = 3 + P / 2 + 0.37f * z / P; hf[P + z] = 5 + P / 3 - 0.21f * z / P; }
            rec = nn_malloc(rb); nn_h2d(rec, h, rb); free(h);
        }
        for (int st = 0; st < nst; st++) {   /* unstreamed, streamed, streamed and recomputing, recomputing in rings, the stream head, both */
            if ((st == 3 || st == 5 || st >= 6) && !ring) continue;
            unet_set_l0_stream(st < 3 ? st : 2); unet_set_l0_ring(st == 3 || st == 5 || st == 7);
            unet_set_l0_head(v, st >= 4 ? head_fwd : nullptr, st >= 4 ? head_bwd : nullptr, &hc);
            unet_set_input_rad(v, st >= 6 ? rec : nullptr);
            unet_build(v, xs, 1);
            for (int k = 0; k < 2; k++) {
                nn_sync(); const double t0 = now_ms();   /* (the second step's forward + backward: the timing) */
                nn_set_sr_step(5000 + 100 * k); const float *l = unet_forward_x(v, st >= 6 ? ct[k] : x[k], xs, 1, 1); unet_zero_grad(v);
                const size_t lgb = st >= 4 ? no * 2 : no * (unet_logits_h16(v) ? 2 : 4); lgbs[st] = lgb;
                lg[st][k] = malloc(lgb); nn_sync(); nn_d2h(lg[st][k], st >= 4 ? (const void *)hc.lgw : (const void *)l, lgb);
                hc.gl = gl[k];
                nn_set_sr_step(5050 + 100 * k); unet_backward_x(v, gl[k], 1); ctr[st][k] = nn_sr_ctr();
                nn_sync(); if (k) ms[st] = now_ms() - t0 - 0.0;
                gg[st][k] = malloc(np * 4); unet_grad_d2h(v, gg[st][k]);
                for (int d = 0; d < 4; d++) { const void *p = unet_l0_stream_dumped(d, &dmb[d]); dmp[st][k][d] = malloc(dmb[d]); nn_sync(); nn_d2h(dmp[st][k][d], p, dmb[d]); }
                if (!k) {   /* a validation forward between the steps */
                    nn_set_sr_step(5025); const float *lv = unet_forward_x(v, st >= 6 ? ct[2] : x[2], xs, 0, 1);
                    lg[st][2] = malloc(lgbs[st]); nn_sync(); nn_d2h(lg[st][2], st >= 4 ? (const void *)hc.lgw : (const void *)lv, lgbs[st]);
                }
            }
        }
        const char *e = nn_check();
        for (int m = 1; m < nst; m++) {   /* each stream mode against the unstreamed run (the stream head in rings: against the stream head) */
            if ((m == 3 || m == 5 || m >= 6) && !ring) continue;
            if (m == 6) continue;   /* (another input: the reference of arm 7) */
            const int r0 = m == 7 ? 6 : m == 5 ? 4 : 0;
            size_t lneq = 0; double lgd = 0, lgm = 0;
            if (lgbs[m] == lgbs[r0]) { for (int k = 0; k < 3; k++) for (size_t i = 0; i < lgbs[m]; i++) lneq += lg[r0][k][i] != lg[m][k][i]; }
            else for (int k = 0; k < 3; k++) for (size_t i = 0; i < no; i++) {   /* (the stream head's fp16 logits against the whole head's fp32) */
                const float a = h2f(((const uint16_t *)lg[m][k])[i]), b = ((const float *)lg[r0][k])[i];
                if (fabs(a - b) > lgd) lgd = fabs(a - b);
                if (fabs(b) > lgm) lgm = fabs(b);
            }
            size_t dneq[4] = {0, 0, 0, 0}; for (int k = 0; k < 2; k++) for (int d = 0; d < 4; d++) for (size_t i = 0; i < dmb[d]; i++) dneq[d] += dmp[r0][k][d][i] != dmp[m][k][d][i];
            /* parameter gradients per part: enc[0], down0, dec[0], the head, all (unet_create's layout: encoders, downs, decoders, head) */
            const int *w = cfg.widths, L = cfg.nlev;
            size_t encs = 0; for (int i = 0; i < L; i++) encs += blk(i ? w[i - 1] : cfg.cin, w[i]);
            const size_t he = np - (dn ? (size_t)2 * (w[0] + w[1] + w[2] + w[3] + w[4] + w[5]) : 0), hs = he - ((size_t)cfg.cout * w[0] + cfg.cout), d0 = hs - blk(w[0] + w[1], w[0]);
            double worst[5] = {0, 0, 0, 0, 0};
            for (int k = 0; k < 2; k++) {
                const double r[5] = {rel(gg[m][k], gg[r0][k], 0, blk(cfg.cin, w[0])), rel(gg[m][k], gg[r0][k], encs, encs + (size_t)w[0] * w[0] * 27 + w[0]),
                                     rel(gg[m][k], gg[r0][k], d0, hs), rel(gg[m][k], gg[r0][k], hs, he), rel(gg[m][k], gg[r0][k], 0, np)};
                for (int j = 0; j < 5; j++) if (r[j] > worst[j]) worst[j] = r[j];
            }
            int fin = 1; for (int k = 0; k < 2; k++) for (size_t i = 0; i < np; i++) if (!isfinite(gg[m][k][i])) { fin = 0; break; }
            const int ok = !e && !lneq && lgd <= 0.01 * lgm && !dneq[0] && !dneq[1] && !dneq[2] && !dneq[3] && fin && worst[0] < 1e-5 && worst[1] < 1e-5 && worst[2] < 1e-5 && worst[3] < 1e-5 && worst[4] < 1e-5 && ctr[r0][0] == ctr[m][0] && ctr[r0][1] == ctr[m][1];
            static const char *nm[8] = {"", "", " (recomputing)", " (recomputing, slab rings)", " (recomputing, the stream head)", " (recomputing, the stream head, slab rings: against the stream head)",
                                        "", " (recomputing, the stream head, --gpu-input, slab rings with the input ring: against the whole input)"};
            printf("  P %d slab %d down_norm %d, stream %d%s%s: logits %zu of %zu bytes differ (max |diff| %.3g of %.3g), gout[1] %zu, dec[0]'s skip gradient %zu, d/d downo[0] %zu, skip gradient %zu; parameter "
                   "gradients rel enc0 %.2g down0 %.2g dec0 %.2g head %.2g all %.2g%s; seeds %u/%u vs %u/%u; step %.0f vs %.0f ms (with the logits' d2h)%s%s\n",
                   P, S, dn, m < 3 ? m : 2, nm[m], getenv("UFSM_L0_STREAM_POISON") ? " (poison)" : "",
                   lneq, 3 * lgbs[m], lgd, lgm, dneq[0], dneq[3], dneq[1], dneq[2], worst[0], worst[1], worst[2], worst[3], worst[4], fin ? "" : " (non-finite)",
                   ctr[m][0], ctr[m][1], ctr[r0][0], ctr[r0][1], ms[m], ms[r0], e ? e : "", ok ? "" : "  FAIL");
            if (!ok) bad++;
        }
        for (int st = 0; st < nst; st++) if (!((st == 3 || st == 5 || st >= 6) && !ring)) for (int k = 0; k < 3; k++) free(lg[st][k]);
        for (int st = 0; st < nst; st++) if (!((st == 3 || st == 5 || st >= 6) && !ring)) for (int k = 0; k < 2; k++) { free(gg[st][k]); for (int d = 0; d < 4; d++) free(dmp[st][k][d]); }
        unet_set_l0_ring(0); unet_set_l0_head(v, nullptr, nullptr, nullptr); unet_set_input_rad(v, nullptr); nn_free(hc.lgw);
        for (int k = 0; k < 3; k++) nn_free(ct[k]);
        nn_free(rec);
        for (int k = 0; k < 3; k++) nn_free(x[k]);
        for (int k = 0; k < 2; k++) nn_free(gl[k]);
        unet_free(v);
    }
    unet_set_l0_stream(-1); unet_set_l0_ring(-1);
    printf("level-0 slab stream: %s\n", bad ? "FAIL" : "PASS");
    return bad != 0;
}
