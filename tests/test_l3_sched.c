/* The three level-0 buffers' copy schedule (unet l3_sched, UFSM_L3_SCHED): the host copies in the order they are due and the coarse
   encoders' conv1 re-run instead of their a1 copies must not change a bit. The trainer's production chain (MX-fp4 activations and
   gradients, lean 2, the GPU-built input from the batch head's CT, the logit gradient over the fp16 logits), 7 levels
   16,16,32,64,128,192,256: two training steps with a validation forward between them, schedule off and on. The logits bit for bit,
   the parameter gradients up to the weight gradients' atomic summation order, the same seed count, the batch tail untouched.
   env UFSM_TEST_P: the window (default 128; the trainer's P512 exercises the copies at their real sizes and timing) */
#include "nn.h"
#include "nn_lp.h"
#include "unet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rs = 777;
static float frand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 11) * 0x1.0p-53 * 2 - 1); }
static float *dev_rand(size_t n, float scale) { float *h = malloc(n * 4); for (size_t i = 0; i < n; i++) h[i] = frand() * scale; float *d = nn_malloc(n * 4); nn_h2d(d, h, n * 4); free(h); return d; }
static double rel(const float *a, const float *r, size_t lo, size_t hi) {
    double d2 = 0, r2 = 0;
    for (size_t i = lo; i < hi; i++) { const double d = (double)a[i] - r[i]; d2 += d * d; r2 += (double)r[i] * r[i]; }
    return r2 > 0 ? sqrt(d2 / r2) : d2 > 0 ? 1.0 : 0.0;
}
/* unet_create's parameter layout: conv (w, b) and GroupNorm (gamma, beta) sizes of a two-conv block */
static size_t blk(int ci, int co) { return (size_t)co * ci * 27 + co + 2 * co + (size_t)co * co * 27 + co + 2 * co; }

int main(void) {
    const int P = getenv("UFSM_TEST_P") ? atoi(getenv("UFSM_TEST_P")) : 128;
    nn_init(0);
    nn_set_tf32(1); nn_set_act_bf16(1); nn_set_grad_bf16(1); nn_set_f16(1); nn_set_grad_scale(1024); nn_set_sr(1); nn_set_gn_stored(1);
    nn_set_prec(3); if (nn_set_prec_policy("all=fp4:fp4:fp4,enc0.c1=fp16")) return 2;
    setenv("UFSM_F4_WGRAD", "1", 1);
    unet_set_act_mx4(1); unet_set_grad_mx8(1); unet_set_grad_mx4(1); unet_set_input_prec(4); unet_set_recompute(1); unet_set_chunk_up(2); unet_set_lean(2);
    unet_set_share_enc_a1(1); unet_set_a0_share(1);
    int bad = 0;
    for (int dn = 0; dn < 2; dn++) {
        const unet_cfg cfg = {7, {16, 16, 32, 64, 128, 192, 256}, 4, 2, 8, dn};
        const shape5 xs = {1, 4, P, P, P};
        const size_t S = (size_t)P * P * P, no = S * 2, tailb = 3 * S;
        /* the radial records of one window (as the sampler writes them) and two fp16 CT channels */
        const size_t rb = NN_RAD_REC(P);
        uint8_t *rec = calloc(1, rb); int32_t hd[8] = {1000, 2000, P, P, 1, 0 | 1 << 2 | 2 << 4, 0, 0}; memcpy(rec, hd, sizeof hd);
        for (int z = 0; z < P; z++) { ((float *)(rec + sizeof hd))[z] = 1000.f + 13.37f + 0.1f * z; ((float *)(rec + sizeof hd))[P + z] = 2000.f + 19.11f - 0.05f * z; }
        void *drec = nn_malloc(rb); nn_h2d(drec, rec, rb);
        void *ct[2]; float *gy[2];
        for (int k = 0; k < 2; k++) { float *f = dev_rand(S, 1.f); ct[k] = nn_malloc(S * 2); nn_f32_to_h16(f, S, ct[k], 1.f); nn_free(f); gy[k] = dev_rand(no, 1.f); }
        uint8_t *pat = malloc(tailb), *chk = malloc(tailb);
        for (size_t i = 0; i < tailb; i++) pat[i] = (uint8_t)(i * 2654435761u >> 13);
        unet *v = unet_create(&cfg); unet_init(v, 41); const size_t np = unet_nparams(v);
        uint16_t *lg[2][3]; float *gg[2][2]; unsigned ctr[2][2]; int l3[2] = {0, 0}, tail_ok = 1;
        for (int on = 0; on < 2; on++) {
            unet_set_l3_sched(on); unet_set_l0_3buf(1);
            unet_set_batch_bytes(v, 2 * S, tailb);
            unet_build(v, xs, 1); unet_set_input_rad(v, drec);
            l3[on] = unet_l0_3buf(v);
            void *tp = nullptr; void *hp = unet_batch_scratch2(v, 2 * S, tailb, &tp);
            if (!hp || !tp) { printf("  P %d down_norm %d schedule %d: no three-buffer batch layout  FAIL\n", P, dn, on); bad++; break; }
            for (int st = 0; st < 2; st++) {
                nn_d2d(hp, ct[st], S * 2); nn_h2d(tp, pat, tailb);
                nn_set_sr_step(9000 + 200 * st); const float *l = unet_forward_x(v, hp, xs, 1, 1); unet_zero_grad(v);
                lg[on][st] = malloc(no * 2); nn_sync(); nn_d2h(lg[on][st], l, no * 2);
                nn_d2h(chk, tp, tailb); if (memcmp(chk, pat, tailb)) tail_ok = 0;
                nn_f32_to_h16(gy[st], no, (void *)l, nn_get_grad_scale());
                nn_set_sr_step(9100 + 200 * st); unet_backward_x(v, (void *)l, 1); ctr[on][st] = nn_sr_ctr();
                gg[on][st] = malloc(np * 4); unet_grad_d2h(v, gg[on][st]);
                if (!st) {   /* the trainer's validation forward between steps */
                    nn_d2d(hp, ct[1], S * 2); nn_h2d(tp, pat, tailb);
                    nn_set_sr_step(9050); const float *lv = unet_forward_x(v, hp, xs, 0, 1);
                    lg[on][2] = malloc(no * 2); nn_sync(); nn_d2h(lg[on][2], lv, no * 2);
                    nn_d2h(chk, tp, tailb); if (memcmp(chk, pat, tailb)) tail_ok = 0;
                }
            }
            unet_set_input_rad(v, nullptr);
        }
        const char *e = nn_check();
        if (!bad) {
            size_t lneq = 0; for (int k = 0; k < 3; k++) for (size_t i = 0; i < no; i++) lneq += lg[0][k][i] != lg[1][k][i];
            /* per part, so a level-0 tensor's damage cannot hide in the coarse levels' norm: enc[0], down0, dec[0], the head, all */
            const int *w = cfg.widths, L = cfg.nlev;
            size_t encs = 0; for (int i = 0; i < L; i++) encs += blk(i ? w[i - 1] : cfg.cin, w[i]);
            const size_t he = np - (dn ? (size_t)2 * (w[0] + w[1] + w[2] + w[3] + w[4] + w[5]) : 0), hs = he - ((size_t)cfg.cout * w[0] + cfg.cout), d0 = hs - blk(w[0] + w[1], w[0]);
            const size_t seg[5][2] = {{0, blk(cfg.cin, w[0])}, {encs, encs + (size_t)w[0] * w[0] * 27 + w[0]}, {d0, hs}, {hs, he}, {0, np}};
            static const char *sn[5] = {"enc0", "down0", "dec0", "head", "all"};
            double worst = 0; char msg[256]; int m = 0;
            for (int s = 0; s < 5; s++) {
                double r = 0; for (int st = 0; st < 2; st++) { const double x = rel(gg[1][st], gg[0][st], seg[s][0], seg[s][1]); if (x > r) r = x; }
                if (r > worst) worst = r;
                m += snprintf(msg + m, sizeof msg - m, "%s%s %.2g", s ? ", " : "", sn[s], r);
            }
            const int ok = l3[0] && l3[1] && !lneq && worst < 1e-5 && ctr[0][0] == ctr[1][0] && ctr[0][1] == ctr[1][1] && tail_ok && !e;
            printf("  P %d down_norm %d: copy schedule on vs off: layouts %d %d, logits differ %zu, grads rel diff (both steps) %s, seeds %u %u / %u %u, batch tail %s%s%s\n",
                   P, dn, l3[0], l3[1], lneq, msg, ctr[0][0], ctr[0][1], ctr[1][0], ctr[1][1], tail_ok ? "kept" : "CLOBBERED", e ? e : "", ok ? "" : "  FAIL");
            if (!ok) bad++;
            for (int on = 0; on < 2; on++) { for (int k = 0; k < 3; k++) free(lg[on][k]); for (int st = 0; st < 2; st++) free(gg[on][st]); }
        }
        unet_set_l3_sched(-1); unet_set_l0_3buf(-1); unet_free(v);
        for (int k = 0; k < 2; k++) { nn_free(ct[k]); nn_free(gy[k]); }
        nn_free(drec); free(rec); free(pat); free(chk);
        if (bad) break;
    }
    printf("l3 copy schedule: %s\n", bad ? "FAIL" : "PASS");
    return bad != 0;
}
