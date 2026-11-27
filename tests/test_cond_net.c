/* train --cond-fields (unet_cfg.cond_ch, 2026-11-27): a latent-grid net with a second 32-channel input block warm-started from a net without it
   (unet_init_from_map: copy_conv_in zeroes the stem's new input columns) computes its donor's function whatever the conditioning planes hold:
   the logits bit-identical for random planes, in the MX-fp8 round-to-nearest setup and the latent recipe's fp4 preset (lean 2, stored GN
   statistics). Then the planes are wired: the stem's new columns get a nonzero weight gradient under a logit gradient, and a nonzero new
   column changes the logits (a dropped window, no planes, does not). The round trip of the header key "cond_ch" (save / peek). */
#include "unet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "test_gpu.h"

enum { P = 64, PC = P / 4, LC = 29, K = 11, GC = PC / 2 };
static uint64_t rs = 0x9e3779b97f4a7c15ull;
static double ru(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (double)(rs >> 11) * (1.0 / 9007199254740992.0); }
static float hbas[LC * 64], hgain[LC];

static void logits(unet *u, const void *ct, int cout, float *out) {
    const shape5 xs = {1, unet_cfg_of(u)->cin, PC, PC, PC};
    const float *lg = unet_forward_x(u, ct, xs, 1, 1); nn_sync();
    const size_t n = (size_t)cout * PC * PC * PC;
    if (unet_logits_h16(u)) { _Float16 *t = malloc(n * 2); nn_d2h(t, lg, n * 2); for (size_t k = 0; k < n; k++) out[k] = (float)t[k]; free(t); }
    else nn_d2h(out, lg, n * 4);
}
static double rel(const float *a, const float *b, size_t n) { double d = 0, r = 0; for (size_t k = 0; k < n; k++) { d += (double)(a[k] - b[k]) * (a[k] - b[k]); r += (double)b[k] * b[k]; } return sqrt(d / (r + 1e-300)); }

int main(void) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (nn_init(0)) return test_no_gpu("test_cond_net", "nn_init(0) failed");
    nn_set_f16(1); nn_set_grad_scale(1024.f); nn_set_loss_grad_h16(1); setenv("UFSM_F4_WGRAD", "1", 0);
    const size_t p3 = (size_t)P * P * P, c3 = (size_t)PC * PC * PC, gc = (size_t)GC * GC * GC;
    _Float16 *hct = malloc(p3 * 2); const size_t RB = NN_RAD_REC(P); uint8_t *hrec = calloc(1, RB);
    {   double k[3][3], ph[3]; for (int j = 0; j < 3; j++) { for (int a = 0; a < 3; a++) k[j][a] = ru() * 0.25; ph[j] = ru() * 6.28; }
        for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
            double v = 0; for (int j = 0; j < 3; j++) v += sin(k[j][0] * z + k[j][1] * y + k[j][2] * x + ph[j]);
            hct[((size_t)z * P + y) * P + x] = (_Float16)(0.6 * v + 0.4 * (2 * ru() - 1));
        }
        int32_t h[8] = {P / 3, -P / 5, P, P, 1, 0 | 1 << 2 | 2 << 4, 0, 0}; memcpy(hrec, h, sizeof h);
        float *cyz = (float *)(hrec + 32), *cxz = cyz + P; for (int z = 0; z < P; z++) { cyz[z] = (float)(P * 0.4 + 0.3 * z); cxz[z] = (float)(-P * 0.1 - 0.2 * z); }
    }
    for (int c = 0; c < LC; c++) { hgain[c] = 1.f; double n = 0; for (int k = 0; k < 64; k++) { hbas[c * 64 + k] = c ? (float)(2 * ru() - 1) : 0.125f; n += (double)hbas[c * 64 + k] * hbas[c * 64 + k]; }
                                   for (int k = 0; k < 64; k++) hbas[c * 64 + k] = (float)(hbas[c * 64 + k] / sqrt(n)); }
    int8_t *hcd = malloc((size_t)K * gc); for (size_t q = 0; q < (size_t)K * gc; q++) hcd[q] = (int8_t)(int)(254 * ru() - 127);
    void *ct = nn_malloc(p3 * 2), *rec = nn_malloc(RB), *cd = nn_malloc((size_t)K * gc);
    nn_h2d(ct, hct, p3 * 2); nn_h2d(rec, hrec, RB); nn_h2d(cd, hcd, (size_t)K * gc);
    char path[] = "/tmp/test_cond_net_XXXXXX"; int fd = mkstemp(path); if (fd >= 0) close(fd);
    int fails = 0;
    for (int mode = 0; mode < 2; mode++) {
        if (mode == 0) { unet_set_act_mx4(0); unet_set_act_mx8(1); nn_set_sr(0); nn_set_prec_policy(""); unet_set_chunk_up(1); unet_set_grad_mx8(0); unet_set_lean(0);
                         unet_set_grad_mx4(0); unet_set_share_enc_a1(0); unet_set_recompute(1); }
        else { unet_set_act_mx4(1); nn_set_sr(0); if (nn_set_prec_policy("all=fp4:fp4:fp4")) return 2; unet_set_chunk_up(2); unet_set_grad_mx8(1); unet_set_lean(2);
               unet_set_grad_mx4(0); unet_set_share_enc_a1(1); unet_set_recompute(1); }
        unet_set_input_mx(1); unet_set_input_prec(mode ? 4 : 0); nn_set_gn_stored(mode);
        unet_cfg ca = {4, {32, 32, 64, 64}, LC + 3, 8, 8, 1}; ca.half_res = 3;
        unet_cfg cb = ca; cb.cond_ch = K; cb.cin = LC + 3 + unet_cond_cin(&cb);
        const size_t nl = (size_t)ca.cout * c3; float *la = malloc(nl * 4), *lb = malloc(nl * 4), *lz = malloc(nl * 4), *lw = malloc(nl * 4);
        unet *a = unet_create(&ca); unet_init(a, 7); unet_set_lat_basis(a, hbas, hgain, LC, 0.f); unet_set_input_rad(a, rec);
        if (unet_save(a, path, 1, nullptr)) { printf("FAIL: save\n"); return 1; }
        logits(a, ct, ca.cout, la);
        unet *b = unet_create(&cb); unet_init(b, 9); unet_set_lat_basis(b, hbas, hgain, LC, 0.f); unet_set_input_rad(b, rec);
        if (unet_init_from_map(b, path, nullptr) <= 0) { printf("FAIL: init-from\n"); return 1; }
        unet_set_cond_input(b, cd, GC, GC, GC); logits(b, ct, cb.cout, lb);
        unet_set_cond_input(b, nullptr, 0, 0, 0); logits(b, ct, cb.cout, lz);
        const int same = !memcmp(la, lb, nl * 4) && !memcmp(la, lz, nl * 4);
        printf("  mode %s: warm-started cond net vs donor, random planes: %s (rel %.2e), no planes: %s\n", mode ? "fp4 preset lean 2" : "MX-fp8 rn        ",
               !memcmp(la, lb, nl * 4) ? "bit-identical" : "DIFFER", rel(lb, la, nl), !memcmp(la, lz, nl * 4) ? "bit-identical" : "DIFFER");
        fails += !same;
        /* wiring: the new columns' weight gradient under a logit gradient, then a nonzero column */
        unet_set_cond_input(b, cd, GC, GC, GC);
        { const shape5 xs = {1, cb.cin, PC, PC, PC}; unet_zero_grad(b); unet_forward_x(b, ct, xs, 1, 1);
          _Float16 *g = malloc(nl * 2); for (size_t k = 0; k < nl; k++) g[k] = (_Float16)(2 * ru() - 1); void *dg = nn_malloc(nl * 2); nn_h2d(dg, g, nl * 2);
          unet_backward_x(b, dg, 1); nn_sync();
          const size_t nw = (size_t)cb.widths[0] * cb.cin * 27; float *gw = malloc(nw * 4); nn_d2h(gw, unet_grad_ptr(b), nw * 4);
          double gn_new = 0, gn_old = 0;
          for (int co = 0; co < cb.widths[0]; co++) for (int ci = 0; ci < cb.cin; ci++) for (int t = 0; t < 27; t++) {
              const double v = gw[((size_t)co * cb.cin + ci) * 27 + t]; if (ci >= LC + 3 && ci < LC + 3 + K) gn_new += v * v; else if (ci < LC + 3) gn_old += v * v; }
          const int okg = gn_new > 0 && isfinite(gn_new) && gn_old > 0;
          printf("    stem weight gradient: new columns |g| %.3e, old %.3e  %s\n", sqrt(gn_new), sqrt(gn_old), okg ? "ok" : "FAIL"); fails += !okg;
          free(g); nn_free(dg); free(gw); }
        { const size_t nw = (size_t)cb.widths[0] * cb.cin * 27; float *w = malloc(nw * 4); nn_d2h(w, unet_param_ptr(b), nw * 4);
          for (int co = 0; co < cb.widths[0]; co++) for (int t = 0; t < 27; t++) w[((size_t)co * cb.cin + LC + 3 + co % K) * 27 + t] = 0.05f * (float)(2 * ru() - 1);
          nn_h2d(unet_param_ptr(b), w, nw * 4); nn_h2d(unet_ema_ptr(b), w, nw * 4); free(w);
          logits(b, ct, cb.cout, lw); unet_set_cond_input(b, nullptr, 0, 0, 0); logits(b, ct, cb.cout, lz);
          const double dw = rel(lw, la, nl); const int okw = dw > 1e-3 && !memcmp(lz, la, nl * 4);
          printf("    a nonzero new column: logits rel %.2e with planes, no planes %s  %s\n", dw, !memcmp(lz, la, nl * 4) ? "bit-identical" : "DIFFER", okw ? "ok" : "FAIL"); fails += !okw; }
        if (mode == 0) {   /* the header key round trip */
            if (unet_save(b, path, 2, nullptr)) { printf("FAIL: save cond\n"); return 1; }
            unet_cfg pc; int st; const int okp = !unet_peek(path, &pc, &st) && pc.cond_ch == K && pc.cin == cb.cin && unet_cond_cin(&pc) == 32;
            printf("    header: cond_ch %d cin %d  %s\n", pc.cond_ch, pc.cin, okp ? "ok" : "FAIL"); fails += !okp;
        }
        unet_free(a); unet_free(b); free(la); free(lb); free(lz); free(lw);
    }
    unlink(path);
    nn_free(ct); nn_free(rec); nn_free(cd); free(hct); free(hrec); free(hcd);
    printf("test_cond_net: %s\n", fails ? "FAIL" : "ok");
    return fails != 0;
}
