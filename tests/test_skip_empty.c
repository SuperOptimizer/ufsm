/* --skip-empty (src/nn_skip.cu, DESIGN 2026-11-13 "empty-space skipping"): masked 3^3 convs on the MX-fp4 inference path.
   (1) every tile active: the logits bit-identical to the dense forward, and the masked kernel did run (masked launches > 0);
   (2) a window with material in its lower z third only, masks from the CT (threshold 0, margins 1-3): the deviation from the dense
       logits by z band. With fixed GroupNorm statistics (gn_ema) it stays within the convs' reach of the bias-filled tiles: material
       logits bit-identical at margin 3, within 2% at margin 2 (this net: 3 levels, no attention; scratch2's attention / gfilm couple
       further, measured separately);
   (3) nn_skip_off: dense again, bit-identical;
   (4) training (DESIGN 2026-11-14 "skip-air training"): forward + backward with MX-fp4 gradients. Every tile active: the gradient
       the dense one (within the dense run-to-run noise) and the masked backward-data / wgrad kernels did run; masks from the CT: the gradient finite, the
       masked backward counted, its cosine to the dense gradient reported (bit identity does not hold: the weight gradients' atomics); the checkpoint header round-trips skip_empty.
   (5) the masked backward kernels exactly, at the try6 level-0 width 256 (audit 2026-11-14 kernels.md #11) and at 64 (the ring kernels' width, which a
       masked conv bypasses): one masked conv (16 x 32 x 32,
       2 of 16 tiles active, a tile row out of every active tile's reach): backward-data on gy == the dense backward-data on gy zeroed outside
       the active tiles, bytes (the same SR seed; inactive tiles: the bias-0 epilogue); the weight gradient (the records of inactive tiles
       zeroed) == the dense one on the zeroed gy within the atomics' order (rel 1e-6); the bias gradient (summed before the zeroing) == the
       dense one on the whole gy. */
#include "unet.h"
#include "nn_lp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int bad;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); bad++; } } while (0)
static uint64_t rs = 777;
static float frand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 11) * 0x1.0p-53); }

static void all_active(int P) {   /* level masks of 8 x 8 x 16-cell tiles, every tile set */
    for (int l = 0; l < 3; l++) { const int c = P >> l; const size_t m = (size_t)((c + 7) / 8) * ((c + 7) / 8) * ((c + 15) / 16);
        uint8_t *h = malloc(m); memset(h, 1, m); nn_skip_set(l, h, c, c, c); free(h); }
}
static double rel(const float *a, const float *b, size_t n) {   /* |a - b| / |a| */
    double d = 0, r = 0; for (size_t i = 0; i < n; i++) { const double e = (double)a[i] - b[i]; d += e * e; r += (double)a[i] * a[i]; } return sqrt(d / (r + 1e-300));
}
static double cosine(const float *a, const float *b, size_t n, int *fin) {
    double ab = 0, aa = 0, bb = 0; *fin = 1; for (size_t i = 0; i < n; i++) { if (!isfinite(a[i])) *fin = 0; ab += (double)a[i] * b[i]; aa += (double)a[i] * a[i]; bb += (double)b[i] * b[i]; }
    return ab / sqrt(aa * bb + 1e-300);
}

int main(void) {
    setvbuf(stdout, nullptr, _IOLBF, 0);   /* (line-buffered: a killed run under compute-sanitizer kept nothing past section (2)) */
    setenv("UFSM_F4_WGRAD", "1", 0);   /* (train --fp4 2; read once) */
    if (nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0)) { printf("no cuda\n"); return 1; }
    nn_set_tf32(1); nn_set_f16(1); nn_set_sr(0); nn_set_prec(3);
    if (nn_set_prec_policy("all=fp4:fp4:fp4")) return 1;
    unet_set_act_mx4(1);
    /* UFSM_SANITIZE=racecheck (tests/sanitize.sh), the QUICK size: a 32^3 window with material in z < 6 (level 0: 2 of 4 tile planes active
       at margin 1, as 3-4 of 8 at P 64), no margin profile (the plain run checks it), and section (4) runs only the three passes that reach
       the masked backward kernels (dense, all active, CT masks) without its numeric comparisons. At P 64 racecheck took ~110 s a forward on
       the pod's RTX PRO 4000 and 3289 s in all (11-15), past the 30-min budget. (The full section (4) died under racecheck after section (2)
       with the pod cgroup's oom_kill counter rising, 11-14/15: 1900 s, oom_kill 3 -> 4; the cause is inferred, not confirmed.) */
    const char *san = getenv("UFSM_SANITIZE"); const int race = san && !strcmp(san, "racecheck");
    const int P = race ? 32 : 64, ZM = race ? 6 : 20;   /* material planes [0, ZM) */
    unet_cfg cfg = {}; cfg.nlev = 3; cfg.widths[0] = 96; cfg.widths[1] = 128; cfg.widths[2] = 160; cfg.cin = 4; cfg.cout = 2; cfg.G = 8; cfg.gn_ema = 1;
    unet *u = unet_create(&cfg); unet_init(u, 11);
    const shape5 xs = {1, 4, P, P, P}; const size_t p3 = (size_t)P * P * P, nx = 4 * p3;
    const shape5 os = unet_out_shape(u, xs); const size_t no = shape_numel(os);
    /* the window: channel 0 the CT (papyrus-like noise in z < ZM, 0 above), channels 1-3 smooth ramps (the radial channels' role: the
       input is nowhere constant) */
    uint8_t *ct = malloc(p3); float *x = malloc(nx * 4);
    for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int xx = 0; xx < P; xx++) {
        const size_t v = ((size_t)z * P + y) * P + xx; ct[v] = z < ZM ? (uint8_t)(40 + 200 * frand()) : 0;
        x[v] = ct[v] / 64.f - 1.f; x[p3 + v] = (float)y / P - 0.5f; x[2 * p3 + v] = (float)xx / P - 0.5f; x[3 * p3 + v] = (float)z / P - 0.5f;
    }
    float *dx = nn_malloc(nx * 4); nn_h2d(dx, x, nx * 4);
    uint8_t *dct = nn_malloc(p3); nn_h2d(dct, ct, p3);
    float *ld = malloc(no * 4), *lm = malloc(no * 4);
    unet_set_gn_ema(u, 1.f, 1); nn_set_sr_step(1); unet_forward(u, dx, xs, 1);   /* the running statistics: this window's (momentum 1) */
    nn_d2h(ld, unet_forward(u, dx, xs, 0), no * 4);
    long nm, nc;
    nn_skip_counts(&nm, &nc);
    {   /* (1) every tile active */
        all_active(P);
        nn_d2h(lm, unet_forward(u, dx, xs, 0), no * 4); nn_skip_counts(&nm, &nc);
        CHECK(!memcmp(ld, lm, no * 4), "all tiles active: logits bit-identical to the dense forward");
        CHECK(nm > 0, "the masked kernel ran (%ld of %ld conv launches masked)", nm, nc);
        printf("(1) all active: %s, %ld of %ld conv launches masked\n", memcmp(ld, lm, no * 4) ? "DIFFERENT" : "bit-identical", nm, nc);
    }
    {   /* (2) masks from the CT */
        nn_skip_off();
        CHECK(!nn_skip_build(dct, P, P, P, 1, 3, 0, 1), "nn_skip_build");
        const double s0 = nn_skip_share(0), s1 = nn_skip_share(1), s2 = nn_skip_share(2);
        nn_d2h(lm, unet_forward(u, dx, xs, 0), no * 4); nn_skip_counts(&nm, &nc);
        /* deviation by distance below / above the material top: material z < ZM - 8, the band [ZM - 8, ZM + 8), far air z >= ZM + 24 */
        double em = 0, eb = 0, ea = 0, sm = 0; size_t nmv = 0;
        for (int c = 0; c < os.c; c++) for (int z = 0; z < P; z++) for (int i = 0; i < P * P; i++) {
            const size_t k = ((size_t)c * P + z) * P * P + i; const double e = fabs((double)ld[k] - lm[k]);
            if (z < ZM - 8) { if (e > em) em = e; sm += fabs(ld[k]); nmv++; } else if (z < ZM + 8) { if (e > eb) eb = e; } else if (z >= ZM + 24 && e > ea) ea = e;
        }
        const double mag = nmv ? sm / nmv : 1;
        printf("(2) material z < %d: tiles active L0 %.3f L1 %.3f L2 %.3f, %ld of %ld launches masked; max |dlogit| material %.4g (mean |logit| %.4g), "
               "boundary band %.4g, far air %.4g\n", ZM, s0, s1, s2, nm, nc, em, mag, eb, ea);
        CHECK(s0 > 0.3 && s0 < 0.7, "level-0 active share %.3f (material + 1 tile of margin: 3 of 8 tile planes)", s0);
        CHECK(nm > 0, "masked launches");
    }
    {   /* the deviation profile by margin: mean |dlogit| per 4-plane band (z from the material top), over the mean |logit| */
        for (int mg = 1; mg <= (race ? 0 : 3); mg++) {
            nn_skip_off(); nn_skip_build(dct, P, P, P, 1, 3, 0, mg);
            const double a0 = nn_skip_share(0);
            nn_d2h(lm, unet_forward(u, dx, xs, 0), no * 4);
            {   /* the material voxels (z < ZM): margin 3 bit-identical (fixed GroupNorm statistics: the error stays within the convs' reach of
                   the fill), margin 2 within 2% (mean |dlogit| over mean |logit|; measured 0.2% on the pod, 2026-11-13) */
                double e = 0, m = 0; int same = 1;
                for (int c = 0; c < os.c; c++) for (size_t k = (size_t)c * p3; k < (size_t)c * p3 + (size_t)ZM * P * P; k++) { e += fabs((double)ld[k] - lm[k]); m += fabs(ld[k]); same &= ld[k] == lm[k]; }
                printf("  margin %d: material mean |dlogit| / mean |logit| %.5f%s\n", mg, e / m, same ? " (bit-identical)" : "");
                if (mg == 2) CHECK(e / m < 0.02, "margin 2: material logits within 2%% (%.4f)", e / m);
                if (mg == 3) CHECK(same, "margin 3: material logits bit-identical to the dense forward");
            }
            printf("  margin %d (L0 active %.3f): z band: mean |dlogit| / mean |logit|:", mg, a0);
            for (int z0 = 0; z0 < P; z0 += 4) { double e = 0, m = 0; for (int c = 0; c < os.c; c++) for (int z = z0; z < z0 + 4; z++) for (int i = 0; i < P * P; i++) {
                    const size_t k = ((size_t)c * P + z) * P * P + i; e += fabs((double)ld[k] - lm[k]); m += fabs(ld[k]); }
                printf(" %d:%.3f", z0, m > 0 ? e / m : 0); }
            printf("\n");
        }
    }
    {   /* (3) off */
        nn_skip_off();
        nn_d2h(lm, unet_forward(u, dx, xs, 0), no * 4);
        CHECK(!memcmp(ld, lm, no * 4), "nn_skip_off: bit-identical to the dense forward again");
    }
    {   /* (4) training */
        unet_set_grad_mx8(1); unet_set_grad_mx4(1); unet_set_input_prec(8);   /* train --fp 4 2's setting: fp4 gradients and wgrad, the input conv fp16 */
        if (nn_set_prec_policy("all=fp4:fp4:fp4,enc0.c1=fp16")) return 1;
        unet_set_gn_ema(u, 1.f, 1 << 20);   /* training forwards on their own (live) statistics: each pass independent of the last (with the running
                                              statistics a pass normalises with the previous pass's, so a dense pass after a masked one differs) */
        nn_set_sr(1);   /* (the fp4 gradients' pre-pass rounds stochastically: each pass the same seed, nn_set_sr_step) */
        const size_t np = unet_nparams(u); float *gd = malloc(np * 4), *gm = malloc(np * 4), *gy = malloc(no * 4);
        for (size_t i = 0; i < no; i++) gy[i] = (frand() - 0.5f) * 1e-3f;
        float *dgy = nn_malloc(no * 4); nn_h2d(dgy, gy, no * 4);
        long nb, nw;
        nn_skip_off(); nn_skip_counts_bwd(&nb, &nw);
        nn_set_sr_step(4242); unet_forward(u, dx, xs, 1); unet_zero_grad(u); unet_backward(u, dgy); unet_grad_d2h(u, gd);
        all_active(P);
        nn_set_sr_step(4242); unet_forward(u, dx, xs, 1); unet_zero_grad(u); unet_backward(u, dgy); unet_grad_d2h(u, gm); nn_skip_counts_bwd(&nb, &nw);
        int fin; double cs = cosine(gd, gm, np, &fin); const double ra = rel(gd, gm, np);
        if (race) {
            printf("(4) train, all active (racecheck: the masked passes only, no numeric checks): cosine %.6f, masked backward-data %ld, wgrad %ld\n", cs, nb, nw);
            CHECK(nb > 0 && nw > 0, "masked backward-data (%ld) and wgrad (%ld) ran", nb, nw);
            nn_skip_off(); CHECK(!nn_skip_build(dct, P, P, P, 1, 3, 0, 1), "nn_skip_build");
            nn_set_sr_step(4242); unet_forward(u, dx, xs, 1); unet_zero_grad(u); unet_backward(u, dgy); unet_grad_d2h(u, gm); nn_skip_counts_bwd(&nb, &nw);
            cs = cosine(gd, gm, np, &fin);
            printf("    train, CT masks (margin 1): gradient finite %d, cosine to dense %.4f, masked backward-data %ld, wgrad %ld\n", fin, cs, nb, nw);
            CHECK(nb > 0 && nw > 0, "train, CT masks: masked backward ran");
            nn_skip_off(); nn_free(dgy); free(gd); free(gm); free(gy);
            goto sec5;
        }
        /* (the weight gradients sum with atomics: two dense passes differ in the last bits; the run-to-run noise is the yardstick) */
        float *g2 = malloc(np * 4); nn_skip_off(); nn_set_sr_step(4242); unet_forward(u, dx, xs, 1); unet_zero_grad(u); unet_backward(u, dgy); unet_grad_d2h(u, g2);
        const double r0 = rel(gd, g2, np); free(g2);
        printf("(4) train, all active: gradient |d| / |g| %.3g (dense run to run %.3g, cosine %.6f), masked backward-data %ld, wgrad %ld\n", ra, r0, cs, nb, nw);
        CHECK(ra < 4 * r0 + 1e-5, "train, all tiles active: gradient = the dense one up to the atomics' noise (%.3g vs %.3g)", ra, r0);
        CHECK(nb > 0 && nw > 0, "masked backward-data (%ld) and wgrad (%ld) ran", nb, nw);
        nn_skip_off(); CHECK(!nn_skip_build(dct, P, P, P, 1, 3, 0, 1), "nn_skip_build");
        nn_set_sr_step(4242); unet_forward(u, dx, xs, 1); unet_zero_grad(u); unet_backward(u, dgy); unet_grad_d2h(u, gm); nn_skip_counts_bwd(&nb, &nw);
        cs = cosine(gd, gm, np, &fin);
        printf("    train, CT masks (margin 1): gradient finite %d, cosine to dense %.4f, masked backward-data %ld, wgrad %ld\n", fin, cs, nb, nw);
        CHECK(fin, "train, CT masks: gradient finite");
        CHECK(nb > 0 && nw > 0, "train, CT masks: masked backward ran");
        {   /* the weight-gradient kernels skip the items whose gy records are all zeroed: the same gradient as visiting them */
            float *gv = malloc(np * 4); setenv("UFSM_SKIP_WG_ITEMS", "0", 1);
            nn_skip_off(); nn_skip_build(dct, P, P, P, 1, 3, 0, 1);
            nn_set_sr_step(4242); unet_forward(u, dx, xs, 1); unet_zero_grad(u); unet_backward(u, dgy); unet_grad_d2h(u, gv); unsetenv("UFSM_SKIP_WG_ITEMS");
            printf("    train, CT masks: |d| / |g| item skipping vs visiting the zeroed records %.3g\n", rel(gv, gm, np));
            CHECK(rel(gv, gm, np) < 4 * r0 + 1e-5, "train, CT masks: the wgrad item skip changes the gradient (%.3g)", rel(gv, gm, np)); free(gv);
        }
        CHECK(cs > 0.3, "train, CT masks: gradient cosine to dense %.4f (0.55 on the pod, 11-14: a different net, live statistics shift too)", cs);
        nn_skip_off();
        nn_set_sr_step(4242); unet_forward(u, dx, xs, 1); unet_zero_grad(u); unet_backward(u, dgy); unet_grad_d2h(u, gm);
        printf("    train, nn_skip_off: |d| / |g| to the first dense pass %.3g\n", rel(gd, gm, np));
        CHECK(rel(gd, gm, np) < 4 * r0 + 1e-5, "train, nn_skip_off: the dense gradient again (%.3g)", rel(gd, gm, np));
        /* the checkpoint header */
        char path[256]; snprintf(path, sizeof path, "%s/test_skip_empty.%d.ckpt", getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp", (int)getpid());
        unet_cfg *uc = (unet_cfg *)unet_cfg_of(u); uc->skip_thr = 1; uc->skip_margin = 2; uc->skip_nl = 3;   /* the header writer reads the live cfg */
        unet_cfg pc = {}; int st = 0;
        CHECK(!unet_save(u, path, 5, nullptr) && !unet_peek(path, &pc, &st), "checkpoint save / peek");
        CHECK(pc.skip_thr == 1 && pc.skip_margin == 2 && pc.skip_nl == 3, "checkpoint skip_empty round trip (%d %d %d)", pc.skip_thr, pc.skip_margin, pc.skip_nl);
        uc->skip_thr = 0; CHECK(!unet_save(u, path, 5, nullptr) && !unet_peek(path, &pc, &st) && pc.skip_thr == 0, "no skip_empty key when off");
        remove(path); nn_free(dgy); free(gd); free(gm); free(gy);
    }
sec5:
    for (int wi = race; wi < 2; wi++) {   /* (5) masked backward-data / weight gradient == dense on the zeroed gy, widths 256 and 64 */
        /* width 64 (Cip 64): fwd_f4_t's ring kernels (launch_f4r) do not take the mask; a masked conv must bypass them (nn_skip_masks). Before
           (11-15) the 64-wide forward ran dense silently: masked counts 0, the racecheck run's only width, FAILed after 3289 s on the pod.
           UFSM_SANITIZE=racecheck (tests/sanitize.sh): width 64 only. At 256 the run died without output ~30 min in (the wide kernels'
           shared-memory tracking, 0 hazards, 11-15) */
        const int C = wi ? 64 : 256, D = 16, H = 32, W = 32; const shape5 s5 = {1, C, D, H, W}; const size_t S = (size_t)D * H * W, n = (size_t)C * S, nw = (size_t)C * C * 27;
        const int my = 4, mx = 2; uint8_t hm[2 * 4 * 2] = {};   /* 2 x 4 x 2 tiles */
        hm[(0 * my + 0) * mx + 0] = 1; hm[(1 * my + 1) * mx + 1] = 1;   /* dilated by one tile: y tile 3 out of reach (gx there 0) */
        float *hx = malloc(n * 4), *hg = malloc(n * 4), *hz = malloc(n * 4), *hw = malloc(nw * 4);
        for (size_t i = 0; i < n; i++) { hx[i] = 2 * frand() - 1; hg[i] = (2 * frand() - 1) * 1e-3f; }
        for (size_t i = 0; i < nw; i++) hw[i] = (2 * frand() - 1) * 0.03f;
        size_t nact = 0;
        for (int c = 0; c < C; c++) for (size_t v = 0; v < S; v++) {
            const int z = (int)(v / (H * W)), y = (int)(v / W % H), x = (int)(v % W), a = hm[((z >> 3) * my + (y >> 3)) * mx + (x >> 4)];
            hz[(size_t)c * S + v] = a ? hg[(size_t)c * S + v] : 0.f; nact += a && !c;
        }
        const size_t b4 = nn_mx4_bytes(s5);
        void *x4 = nn_malloc(b4), *g4 = nn_malloc(b4), *z4 = nn_malloc(b4), *y4 = nn_malloc(b4), *gxm = nn_malloc(b4), *gxd = nn_malloc(b4);
        void *mx4s[6] = {x4, g4, z4, y4, gxm, gxd}; for (int i = 0; i < 6; i++) { nn_zero(mx4s[i], b4); nn_set_storage(mx4s[i], b4, 4); }
        float *t = nn_malloc(n * 4);
        nn_h2d(t, hx, n * 4); lp_f32_to_mx4(t, 1, C, S, x4); nn_h2d(t, hg, n * 4); lp_f32_to_mx4(t, 1, C, S, g4); nn_h2d(t, hz, n * 4); lp_f32_to_mx4(t, 1, C, S, z4);
        nn_free(t);
        float *w = nn_malloc(nw * 4), *scr = nn_malloc(nn_conv3d_scratch(s5, C, 3) + 4096);
        float *gwm = nn_malloc(nw * 4), *gbm = nn_malloc(C * 4), *gwd = nn_malloc(nw * 4), *gbd = nn_malloc(C * 4), *gbf = nn_malloc(C * 4);
        nn_h2d(w, hw, nw * 4); nn_wmemo_clear();
        long fm, fc, nb, nwg; nn_skip_counts(&fm, &fc); nn_skip_counts_bwd(&nb, &nwg);
        nn_set_layer(7); nn_set_conv(1); nn_set_block(0);
        CHECK(!nn_skip_set(0, hm, D, H, W), "(5) nn_skip_set");
        const split_t s0 = {}; split_t sw = {}; sw.sr = 0x2545f491u;
        lp_conv_fwd_f4(x4, 4, s5, w, nullptr, C, y4, 4, (gnp_t){}, nullptr, 0, s0);   /* the masked forward: marks the conv */
        nn_set_sr_step(5151); nn_conv3d_bwd_data(g4, s5, w, s5, 3, 1, gxm, scr);
        nn_skip_counts(&fm, &fc); nn_skip_counts_bwd(&nb, &nwg);
        nn_skip_pause(1);
        nn_set_sr_step(5151); nn_conv3d_bwd_data(z4, s5, w, s5, 3, 1, gxd, scr);
        nn_zero(gwd, nw * 4); nn_zero(gbd, C * 4); nn_zero(gbf, C * 4); nn_zero(gwm, nw * 4); nn_zero(gbm, C * 4);
        lp_bwd_w_f4(x4, 4, s5, z4, 4, s5, gwd, gbd, (gnp_t){}, sw, 0);
        float *gwf = nn_malloc(nw * 4); nn_zero(gwf, nw * 4);
        lp_bwd_w_f4(x4, 4, s5, g4, 4, s5, gwf, gbf, (gnp_t){}, sw, 0);   /* (the bias gradient of the whole gy) */
        nn_skip_pause(0);   /* (clears the marks: forward again) */
        lp_conv_fwd_f4(x4, 4, s5, w, nullptr, C, y4, 4, (gnp_t){}, nullptr, 0, s0);
        nn_skip_wg_begin(); lp_bwd_w_f4(x4, 4, s5, g4, 4, s5, gwm, gbm, (gnp_t){}, sw, 0); nn_skip_wg_end();
        long nb2, nw2; nn_skip_counts_bwd(&nb2, &nw2);
        nn_sync(); const char *e = nn_check();
        CHECK(!e, "(5) CUDA error %s", e ? e : "");
        uint8_t *bm = malloc(b4), *bd = malloc(b4); nn_d2h(bm, gxm, b4); nn_d2h(bd, gxd, b4);
        float *hwm = malloc(nw * 4), *hwd = malloc(nw * 4), *hbm = malloc(C * 4), *hbd = malloc(C * 4), *hbf = malloc(C * 4);
        nn_d2h(hwm, gwm, nw * 4); nn_d2h(hwd, gwd, nw * 4); nn_d2h(hbm, gbm, C * 4); nn_d2h(hbd, gbd, C * 4); nn_d2h(hbf, gbf, C * 4);
        size_t nzd = 0; for (size_t i = 0; i < nw; i++) nzd += hwd[i] != 0.f;
        const double rw = rel(hwd, hwm, nw), rb = rel(hbf, hbm, C), rbz = rel(hbd, hbm, C);
        printf("(5) masked backward, width %d, %dx%dx%d, %zu of %zu voxels active: masked forward %ld, backward-data %ld, wgrad %ld; gx bytes %s; wgrad rel %.3g; bias rel %.3g (vs the zeroed gy's %.3g)\n",
               C, D, H, W, nact, S, fm, nb, nw2, memcmp(bm, bd, b4) ? "DIFFER" : "equal", rw, rb, rbz);
        CHECK(fm > 0 && nb > 0 && nw2 > 0, "(5) the masked kernels ran (forward %ld, backward-data %ld, wgrad %ld)", fm, nb, nw2);
        CHECK(!memcmp(bm, bd, b4), "(5) masked backward-data == dense backward-data on the zeroed gy, bytes");
        CHECK(nzd > nw / 2 && rw < 1e-6, "(5) masked wgrad == dense wgrad on the zeroed gy (rel %.3g, %zu nonzero)", rw, nzd);
        CHECK(rb < 1e-6 && rbz > 1e-3, "(5) masked bias gradient == dense on the whole gy (rel %.3g; the zeroed gy's differs: %.3g)", rb, rbz);
        nn_skip_off(); nn_set_layer(-1); nn_set_conv(-1); nn_set_block(0); nn_wmemo_clear();
        free(hx); free(hg); free(hz); free(hw); free(bm); free(bd); free(hwm); free(hwd); free(hbm); free(hbd); free(hbf);
        for (int i = 0; i < 6; i++) nn_free(mx4s[i]);
        nn_free(w); nn_free(scr); nn_free(gwm); nn_free(gbm); nn_free(gwd); nn_free(gbd); nn_free(gbf); nn_free(gwf);
    }
    if (bad) { printf("test_skip_empty: %d failed\n", bad); return 1; }
    printf("test_skip_empty: ok\n");
    return 0;
}
