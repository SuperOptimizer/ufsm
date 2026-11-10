/* The latent nets' head on the cell grid through unet: 32 / 96 inputs -> 144 outputs (--lat-guide 3: 16 interpolated outputs + 128
   sub-cell values), every conv fp4, MX-fp4 input and gradients, fp16 logits and logit gradient (the trainer's setting, train.c g_lat).
   - forward: the logits against a host W a + b on dec[0]'s output a as the head reads it (unet_head_ext_fwd with identity rows);
   - weight / bias gradient: against the host sum over voxels of g a^T / g on the same a and the 16-bit logit gradient;
   - data gradient (passes of 8 logit-gradient channels into MX-fp4 rows with exact SR, conv1_to_mx_t_k): (a) a logit gradient
     supported on 8 channels gives the trunk gradients of an 8-output net holding those head rows (the <= 16-output kernels);
     (b) the trunk's parameter gradients against the 16-bit-gradient backward of the same net: SR is unbiased, so the mean of K
     differently rounded backwards approaches it ~1 / sqrt(K). */
#include "nn.h"
#include "nn_lp.h"
#include "unet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rs = 777;
static float frand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 11) * 0x1.0p-53 * 2 - 1); }
static float h2f(uint16_t h) {
    const int e = h >> 10 & 31, m = h & 1023; const float s = h >> 15 ? -1.f : 1.f;
    return s * (e == 0 ? ldexpf((float)m, -24) : e == 31 ? INFINITY : ldexpf((float)(m + 1024), e - 25));
}
/* the streamed head's callback (train.c g_lat_hs): the head's backward per range of Z cell planes, the gradient copied from the whole layout */
typedef struct { unet *v; void *gl, *sb; size_t S, hw; int D, Z, CO, bad; } hs_ctx;
static void hs_cb(void *p) {
    hs_ctx *h = p;
    for (int z0 = 0; z0 < h->D; z0 += h->Z) {
        const int z1 = z0 + h->Z < h->D ? z0 + h->Z : h->D; const size_t ns = (size_t)(z1 - z0) * h->hw;
        for (int c = 0; c < h->CO; c++) nn_d2d((char *)h->sb + c * ns * 2, (const char *)h->gl + (c * h->S + z0 * h->hw) * 2, ns * 2);
        if (unet_head_bwd_z(h->v, z0, z1, h->sb, z0 == 0)) h->bad++;
    }
}
static double rel2(const float *a, const float *r, size_t n) { double d2 = 0, r2 = 0; for (size_t i = 0; i < n; i++) { const double d = (double)a[i] - r[i]; d2 += d * d; r2 += (double)r[i] * r[i]; } return sqrt(d2 / (r2 > 0 ? r2 : 1)); }
static void detail(const unet_cfg *cfg, const float *g4, const float *ga) {   /* per tensor (unet_create's layout) */
    const int *w = cfg->widths, L = cfg->nlev; size_t o = 0;
#define T(nm, k, n) do { const size_t n_ = (n); const double r = rel2(g4 + o, ga + o, n_); if (r > 1e-6) printf("      %s%d %.3g\n", nm, k, r); o += n_; } while (0)
    for (int i = 0; i < L; i++) { int ci = i ? w[i - 1] : cfg->cin, co = w[i];
        T("enc.c1.w ", i, (size_t)co * ci * 27); T("enc.c1.b ", i, co); T("enc.n1 ", i, 2 * co); T("enc.c2.w ", i, (size_t)co * co * 27); T("enc.c2.b ", i, co); T("enc.n2 ", i, 2 * co); }
    for (int i = 0; i < L - 1; i++) { T("down.w ", i, (size_t)w[i] * w[i] * 27); T("down.b ", i, w[i]); }
    for (int i = L - 2; i >= 0; i--) { int ci = w[i] + w[i + 1], co = w[i];
        T("dec.c1.w ", i, (size_t)co * ci * 27); T("dec.c1.b ", i, co); T("dec.n1 ", i, 2 * co); T("dec.c2.w ", i, (size_t)co * co * 27); T("dec.c2.b ", i, co); T("dec.n2 ", i, 2 * co); }
#undef T
}
int main(void) {
    int bad = 0; const int K = getenv("TLH_K") ? atoi(getenv("TLH_K")) : 24, CO = 144;
    if (nn_init(0)) { printf("test_lat_head: no GPU\n"); return 77; }
    nn_set_tf32(1); nn_set_act_bf16(1); nn_set_grad_bf16(1); nn_set_f16(1); nn_set_grad_scale(1024); nn_set_sr(1); nn_set_gn_stored(1);
    nn_set_prec(3); if (nn_set_prec_policy("all=fp4:fp4:fp4")) bad++;
    setenv("UFSM_F4_WGRAD", "1", 1);
    setenv("UFSM_HEAD_FUSED", "0", 1);   /* the 8-output comparison net with 32 inputs: the separate kernels too (the fused <= 16-output kernel keys its SR otherwise) */
    unet_set_act_mx4(1); unet_set_grad_mx8(1); unet_set_grad_mx4(1); unet_set_input_prec(4); unet_set_recompute(1); unet_set_chunk_up(2); unet_set_lean(2);
    unet_set_share_enc_a1(1); unet_set_a0_share(1);   /* (the trainer's lean-2 modes; TLH_SHARE: others) */
    if (getenv("TLH_SHARE")) { const int m = atoi(getenv("TLH_SHARE")); if (m & 1) unet_set_share_enc_a1(1); if (m & 2) unet_set_a0_share(1); if (m & 4) unet_set_a0_share(0); if (m & 8) unet_set_share_enc_a1(0); if (m & 16) unet_set_lean(0); if (m & 32) unet_set_chunk_up(0); }
    for (int wc = 0; wc < 2; wc++) {
        const int C0 = wc ? 96 : 32;
        unet_cfg cfg = {3, {C0, 64, 96}, 32, CO, 8, 1, 0};   /* (three levels as the trainer's smallest nets) */
        const shape5 xs = {1, 32, 16, 16, 16};
        const size_t nx = shape_numel(xs), S = nx / 32, no = S * CO;
        float *hx = malloc(nx * 4), *hg = malloc(no * 4);
        for (size_t i = 0; i < nx; i++) hx[i] = frand();
        for (size_t i = 0; i < no; i++) hg[i] = frand();
        float *x = nn_malloc(nx * 4), *gy = nn_malloc(no * 4); nn_h2d(x, hx, nx * 4); nn_h2d(gy, hg, no * 4);
        unet *u = unet_create(&cfg); unet_init(u, 41 + wc); unet_set_fp16_logits(u, 1);
        const size_t np = unet_nparams(u); size_t ow, ob; unet_head_offsets(u, &ow, &ob);
        /* enc[0]'s parameters (the far end of the backward) are compared apart: with a 96-wide level 0 under the lean-2 sharing modes their
           gradients are not reproduced between two equal backwards (another SR key on the way; found 2026-11-09, not the head's: every other
           tensor is equal to the last bits); they are unbiased, which the last check holds them to */
        const size_t ne0 = (size_t)C0 * cfg.cin * 27 + C0 + 2 * C0 + (size_t)C0 * C0 * 27 + C0 + 2 * C0;
        float *par = malloc(np * 4); nn_d2h(par, unet_live_params(u), np * 4);
        for (size_t i = 0; i < (size_t)CO; i++) par[ob + i] = 0.5f * frand();   /* (unet_init: all biases -2) */
        nn_h2d((float *)unet_live_params(u), par, np * 4);
        const float *W = par + ow, *Bv = par + ob;
        void *gl = nn_malloc(no * 2); nn_f32_to_h16(gy, no, gl, nn_get_grad_scale());
        uint16_t *lw = malloc(no * 2), *g16 = malloc(no * 2), *a16 = malloc(S * C0 * 2);
        nn_d2h(g16, gl, no * 2);
        /* the identity head for dec[0]'s output */
        float *hid = calloc((size_t)C0 * (C0 + 1), 4); for (int c = 0; c < C0; c++) hid[(size_t)c * C0 + c] = 1.f;
        float *did = nn_malloc((size_t)C0 * (C0 + 1) * 4); nn_h2d(did, hid, (size_t)C0 * (C0 + 1) * 4);
        void *da = nn_malloc(S * C0 * 2);
        float *ga = malloc(np * 4), *gr = malloc(np * 4), *g4 = malloc(np * 4), *dm = calloc(np, 4);
        nn_set_sr_step(3000); const float *lg = unet_forward(u, x, xs, 1); unet_zero_grad(u);
        nn_sync(); nn_d2h(lw, lg, no * 2);
        if (unet_head_ext_fwd(u, did, did + (size_t)C0 * C0, C0, da)) { printf("  dec[0]'s output through identity rows: not covered  FAIL\n"); bad++; }
        nn_sync(); nn_d2h(a16, da, S * C0 * 2);
        nn_set_sr_step(3100); unet_backward_x(u, gl, 1); const unsigned ctr0 = nn_sr_ctr(); unet_grad_d2h(u, ga);
        { const char *e = nn_check(); if (e) { printf("  CUDA error %s  FAIL\n", e); bad++; } }
        /* host references */
        float *a = malloc(S * C0 * 4), *lref = malloc(no * 4), *lgot = malloc(no * 4), *gwr = calloc((size_t)CO * C0 + CO, 4);
        for (size_t i = 0; i < S * C0; i++) a[i] = h2f(a16[i]);
        const float gs = nn_get_grad_scale(); double amax = 0;
        for (size_t i = 0; i < S * C0; i++) if (fabs(a[i]) > amax) amax = fabs(a[i]);
        for (int co = 0; co < CO; co++) {
            double *acc = calloc(C0 + 1, sizeof(double));
            for (size_t v = 0; v < S; v++) {
                double s = Bv[co]; for (int ci = 0; ci < C0; ci++) s += (double)W[(size_t)co * C0 + ci] * a[ci * S + v];
                lref[co * S + v] = (float)s; lgot[co * S + v] = h2f(lw[co * S + v]);
                const double g = h2f(g16[co * S + v]) / gs;
                for (int ci = 0; ci < C0; ci++) acc[ci] += g * a[ci * S + v];
                acc[C0] += g;
            }
            for (int ci = 0; ci < C0; ci++) gwr[(size_t)co * C0 + ci] = (float)acc[ci];
            gwr[(size_t)CO * C0 + co] = (float)acc[C0]; free(acc);
        }
        const double ef = rel2(lgot, lref, no), ew = rel2(ga + ow, gwr, (size_t)CO * C0), eb = rel2(ga + ob, gwr + (size_t)CO * C0, CO);
        double emax = 0; for (size_t i = 0; i < no; i++) { const double d = fabs((double)lgot[i] - lref[i]); if (d > emax) emax = d; }
        const int ok1 = ef < 2e-3 && ew < 2e-3 && eb < 1e-4 && amax > 0.1;
        printf("  %d -> %d head through unet vs host: logits rel %.3g (max abs %.3g, |a| max %.3g), weight grad rel %.3g, bias grad rel %.3g%s\n", C0, CO, ef, emax, amax, ew, eb, ok1 ? "" : "  FAIL");
        if (!ok1) bad++;
        if (getenv("TLH_REPRO")) {   /* the same backward again on this net, and on a second net of the same configuration */
            float *g2 = malloc(np * 4), *g3 = malloc(np * 4);
            nn_set_sr_step(3000); unet_forward(u, x, xs, 1); unet_zero_grad(u); nn_set_sr_step(3100); unet_backward_x(u, gl, 1); unet_grad_d2h(u, g2);
            unet *e = unet_create(&cfg); unet_init(e, 3); unet_set_fp16_logits(e, 1); nn_h2d((float *)unet_live_params(e), par, np * 4);
            nn_set_sr_step(3000); unet_forward(e, x, xs, 1); unet_zero_grad(e); nn_set_sr_step(3100); unet_backward_x(e, gl, 1); unet_grad_d2h(e, g3);
            printf("    repro: same net again enc[0] %.3g rest %.3g; second equal net enc[0] %.3g rest %.3g\n", rel2(g2, ga, ne0), rel2(g2 + ne0, ga + ne0, np - ne0), rel2(g3, ga, ne0), rel2(g3 + ne0, ga + ne0, np - ne0));
            unet_free(e); free(g2); free(g3);
        }
        /* (a) 8 logit-gradient channels: the same trunk gradients as the 8-output net with those head rows */
        for (int c0 = 0; c0 < CO; c0 += 72) {   /* output groups 0 and 9 (channels 0..7 interpolated, 72..79 sub-cell) */
            unet_cfg c8 = cfg; c8.cout = 8;
            unet *v = unet_create(&c8); unet_init(v, 5); unet_set_fp16_logits(v, 1);
            const size_t np8 = unet_nparams(v); size_t ow8, ob8; unet_head_offsets(v, &ow8, &ob8);
            int lay = ow8 == ow && ob8 == ow8 + (size_t)8 * C0 && ob == ow + (size_t)CO * C0 && np - (ob + CO) == np8 - (ob8 + 8);
            float *p8 = malloc(np8 * 4), *g8 = malloc(np8 * 4), *gz = calloc(no, 4);
            if (lay) {
                memcpy(p8, par, ow * 4); memcpy(p8 + ow8, W + (size_t)c0 * C0, (size_t)8 * C0 * 4); memcpy(p8 + ob8, Bv + c0, 8 * 4);
                memcpy(p8 + ob8 + 8, par + ob + CO, (np8 - ob8 - 8) * 4);
                nn_h2d((float *)unet_live_params(v), p8, np8 * 4);
            }
            memcpy(gz + c0 * S, hg + c0 * S, 8 * S * 4);
            float *dgz = nn_malloc(no * 4); nn_h2d(dgz, gz, no * 4);
            void *glz = nn_malloc(no * 2), *gl8 = nn_malloc(8 * S * 2); nn_f32_to_h16(dgz, no, glz, gs); nn_f32_to_h16(dgz + c0 * S, 8 * S, gl8, gs);
            uint16_t *l8 = malloc(8 * S * 2);
            nn_set_sr_step(3000); unet_forward(u, x, xs, 1); unet_zero_grad(u); nn_set_sr_step(3100); unet_backward_x(u, glz, 1); unet_grad_d2h(u, g4);
            nn_set_sr_step(3000); const float *lv = unet_forward(v, x, xs, 1); unet_zero_grad(v); nn_sync(); nn_d2h(l8, lv, 8 * S * 2);
            nn_set_sr_step(3100); unet_backward_x(v, gl8, 1); unet_grad_d2h(v, g8);
            size_t lneq = 0; for (size_t i = 0; i < 8 * S; i++) lneq += l8[i] != lw[c0 * S + i];
            const double et = rel2(g4 + ne0, g8 + ne0, ow - ne0), ee = rel2(g4, g8, ne0), eh = rel2(g4 + ow + (size_t)c0 * C0, g8 + ow8, (size_t)8 * C0), ed = rel2(g4 + ob + CO, g8 + ob8 + 8, np8 - ob8 - 8);
            const int ok = lay && !lneq && et < 1e-5 && eh < 1e-5 && ed < 1e-5;
            printf("  %d -> %d head, gradient on outputs %d..%d vs the 8-output net: layout %d, logits differ %zu, trunk grads rel %.3g (enc[0] %.3g), head rows %.3g, down norms %.3g%s\n",
                   C0, CO, c0, c0 + 7, lay, lneq, et, ee, eh, ed, ok ? "" : "  FAIL");
            if (!ok) bad++;
            if (getenv("TLH_DETAIL") && (et > 1e-5 || ee > 1e-5)) detail(&cfg, g4, g8);
            unet_free(v); free(p8); free(g8); free(gz); free(l8); nn_free(dgz); nn_free(glz); nn_free(gl8);
        }
        /* (b) dense logit gradient: MX-fp4 gradients (SR) against the 16-bit-gradient backward, mean of K */
        double one = 0; float *rm = calloc(np, 4);
        for (int k = 0; k < K; k++) {
            unet_set_grad_mx8(0); unet_set_grad_mx4(0); nn_set_sr_step(6000 + k); unet_forward(u, x, xs, 1); unet_zero_grad(u); unet_backward_x(u, gl, 1); unet_grad_d2h(u, gr);
            unet_set_grad_mx8(1); unet_set_grad_mx4(1); nn_set_sr_step(6000 + k); unet_forward(u, x, xs, 1); unet_zero_grad(u); unet_backward_x(u, gl, 1); unet_grad_d2h(u, g4);
            for (size_t i = 0; i < ow; i++) { dm[i] += (g4[i] - gr[i]) / K; rm[i] += gr[i] / K; }
            if (!k) one = rel2(g4, gr, ow);
        }
        double a2 = 0, r2 = 0; for (size_t i = 0; i < ow; i++) { a2 += (double)dm[i] * dm[i]; r2 += (double)rm[i] * rm[i]; }
        const double avg = sqrt(a2 / r2); const int ok2 = isfinite(one) && one > 0 && avg < one * 2.0 / sqrt((double)K) + 0.01;
        double e2 = 0, f2 = 0; for (size_t i = 0; i < ne0; i++) { e2 += (double)dm[i] * dm[i]; f2 += (double)rm[i] * rm[i]; }
        const double avg0 = sqrt(e2 / f2), one0 = rel2(g4, gr, ne0); const int ok3 = avg0 < 2.5 / sqrt((double)K) + 0.02;
        printf("  %d -> %d head, enc[0] grads alone: last draw %.3g, mean of %d %.3g%s\n", C0, CO, one0, K, avg0, ok3 ? "" : "  FAIL");
        if (!ok3) bad++;
        printf("  %d -> %d head, trunk param. grads, MX-fp4 vs 16-bit gradients: one %.3g, mean of %d %.3g%s\n", C0, CO, one, K, avg, ok2 ? "" : "  FAIL");
        if (!ok2) bad++;
        if (getenv("TLH_REPRO")) {   /* K more backwards on the built net (no rebuild between them, as training steps): still unbiased? */
            float *m2 = calloc(np, 4);
            for (int k = 0; k < K; k++) { nn_set_sr_step(6000 + k); unet_forward(u, x, xs, 1); unet_zero_grad(u); unet_backward_x(u, gl, 1); unet_grad_d2h(u, g4); for (size_t i = 0; i < ow; i++) m2[i] += g4[i] / K; }
            double e0 = 0, r0 = 0, e1 = 0, r1 = 0;
            for (size_t i = 0; i < ow; i++) { const double dd = (double)m2[i] - rm[i]; if (i < ne0) { e0 += dd * dd; r0 += (double)rm[i] * rm[i]; } else { e1 += dd * dd; r1 += (double)rm[i] * rm[i]; } }
            printf("    repeated backwards without rebuild, mean of %d vs the 16-bit reference: enc[0] %.3g, rest %.3g; last draw enc[0] %.3g rest %.3g\n", K, sqrt(e0 / r0), sqrt(e1 / r1), rel2(g4, gr, ne0), rel2(g4 + ne0, gr + ne0, ow - ne0));
            free(m2);
        }
        {   /* the head per range of 3 cell planes (unet_set_head_ext; the trainer's streamed latent head): the logits bit for bit, the gradients
               up to the weight gradient's summation order, the same SR draws, no whole logits in the build */
            const size_t hw = (size_t)xs.h * xs.w; const size_t m0 = unet_train_bytes(u, xs);
            unet *e = unet_create(&cfg); unet_init(e, 3); unet_set_fp16_logits(e, 1); nn_h2d((float *)unet_live_params(e), par, np * 4);   /* a second net with the same parameters, head slabs from its first build */
            const int HZ = getenv("TLH_Z") ? atoi(getenv("TLH_Z")) : 3;
            hs_ctx hc = {e, gl, nn_malloc((size_t)CO * (getenv("TLH_BIG") ? xs.d : HZ) * hw * 2), S, hw, xs.d, HZ, CO, 0};
            unet_set_head_ext(e, 1); unet_set_head_cb(e, hs_cb, &hc);
            if (getenv("TLH_DETAIL")) printf("    sb %p storage %d / %d, gl storage %d\n", hc.sb, nn_storage(hc.sb), nn_storage((char *)hc.sb + (size_t)CO * HZ * hw * 2 - 2), nn_storage(gl));
            const size_t m1 = unet_train_bytes(e, xs);
            uint16_t *ls = malloc(no * 2);
            nn_set_sr_step(3000); const float *l2 = unet_forward(e, x, xs, 1); unet_zero_grad(e);
            for (int z0 = 0; z0 < xs.d; z0 += HZ) {
                const int z1 = z0 + HZ < xs.d ? z0 + HZ : xs.d; const size_t ns = (size_t)(z1 - z0) * hw;
                if (unet_head_fwd_z(e, z0, z1, hc.sb)) { hc.bad++; break; }
                for (int c = 0; c < CO; c++) nn_d2h(ls + c * S + z0 * hw, (const char *)hc.sb + c * ns * 2, ns * 2);
            }
            if (getenv("TLH_DUMP")) nn_storage_dump("ext fwd");
            nn_set_sr_step(3100); unet_backward_x(e, nullptr, 1); const unsigned c1 = nn_sr_ctr(); unet_grad_d2h(e, g4);
            if (getenv("TLH_DUMP")) nn_storage_dump("ext bwd");
            size_t lneq = 0; for (size_t i = 0; i < no; i++) lneq += ls[i] != lw[i];
            const double eg = rel2(g4 + ne0, ga + ne0, np - ne0), ee = rel2(g4, ga, ne0); const int ok = !l2 && !hc.bad && !lneq && eg < 1e-6 && c1 == ctr0 && m1 + no * 2 <= m0;
            printf("  %d -> %d head per %d cell planes vs whole: logits differ %zu, grads rel %.3g (enc[0] %.3g), seeds %u / %u, train bytes %zu -> %zu%s\n", C0, CO, HZ, lneq, eg, ee, c1, ctr0, m0, m1, ok ? "" : "  FAIL");
            if (!ok) bad++;
            if (getenv("TLH_DETAIL")) printf("    trunk %.3g head.w %.3g head.b %.3g tail %.3g\n", rel2(g4, ga, ow), rel2(g4 + ow, ga + ow, (size_t)CO * C0), rel2(g4 + ob, ga + ob, CO), rel2(g4 + ob + CO, ga + ob + CO, np - ob - CO));
            if (getenv("TLH_DETAIL") && (eg > 1e-6 || ee > 1e-6)) detail(&cfg, g4, ga);
            unet_free(e); nn_free(hc.sb); free(ls);
        }
        { const char *e = nn_check(); if (e) { printf("  CUDA error %s  FAIL\n", e); bad++; } }
        unet_free(u); free(hx); free(hg); free(par); free(lw); free(g16); free(a16); free(hid); free(ga); free(gr); free(g4); free(dm); free(rm); free(a); free(lref); free(lgot); free(gwr);
        nn_free(x); nn_free(gy); nn_free(gl); nn_free(did); nn_free(da);
    }
    printf(bad ? "test_lat_head: %d FAILED\n" : "test_lat_head: ok\n", bad);
    return bad != 0;
}
