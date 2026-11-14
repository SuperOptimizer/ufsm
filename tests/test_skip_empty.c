/* --skip-empty (src/nn_skip.cu, DESIGN 2026-11-13 "empty-space skipping"): masked 3^3 convs on the MX-fp4 inference path.
   (1) every tile active: the logits bit-identical to the dense forward, and the masked kernel did run (masked launches > 0);
   (2) a window with material in its lower z third only, masks from the CT (threshold 0, margins 1-3): the deviation from the dense
       logits by z band. With fixed GroupNorm statistics (gn_ema) it stays within the convs' reach of the bias-filled tiles: material
       logits bit-identical at margin 3, within 2% at margin 2 (this net: 3 levels, no attention; scratch2's attention / gfilm couple
       further, measured separately);
   (3) nn_skip_off: dense again, bit-identical;
   (4) training (DESIGN 2026-11-14 "skip-air training"): forward + backward with MX-fp4 gradients. Every tile active: the gradient
       bit-identical to the dense one and the masked backward-data / wgrad kernels did run; masks from the CT: the gradient finite, the
       masked backward counted, its cosine to the dense gradient reported; the checkpoint header round-trips skip_empty. */
#include "unet.h"
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
static double cosine(const float *a, const float *b, size_t n, int *fin) {
    double ab = 0, aa = 0, bb = 0; *fin = 1; for (size_t i = 0; i < n; i++) { if (!isfinite(a[i])) *fin = 0; ab += (double)a[i] * b[i]; aa += (double)a[i] * a[i]; bb += (double)b[i] * b[i]; }
    return ab / sqrt(aa * bb + 1e-300);
}

int main(void) {
    if (nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0)) { printf("no cuda\n"); return 1; }
    nn_set_tf32(1); nn_set_f16(1); nn_set_sr(0); nn_set_prec(3);
    if (nn_set_prec_policy("all=fp4:fp4:fp4")) return 1;
    unet_set_act_mx4(1);
    const int P = 64, ZM = 20;   /* material planes [0, ZM) */
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
        for (int mg = 1; mg <= 3; mg++) {
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
        unet_set_grad_mx8(1); unet_set_grad_mx4(1);
        const size_t np = unet_nparams(u); float *gd = malloc(np * 4), *gm = malloc(np * 4), *gy = malloc(no * 4);
        for (size_t i = 0; i < no; i++) gy[i] = (frand() - 0.5f) * 1e-3f;
        float *dgy = nn_malloc(no * 4); nn_h2d(dgy, gy, no * 4);
        long nb, nw;
        nn_skip_off(); nn_skip_counts_bwd(&nb, &nw);
        unet_forward(u, dx, xs, 1); unet_zero_grad(u); unet_backward(u, dgy); unet_grad_d2h(u, gd);
        all_active(P);
        unet_forward(u, dx, xs, 1); unet_zero_grad(u); unet_backward(u, dgy); unet_grad_d2h(u, gm); nn_skip_counts_bwd(&nb, &nw);
        int fin; double cs = cosine(gd, gm, np, &fin);
        printf("(4) train, all active: gradient %s (cosine %.6f), masked backward-data %ld, wgrad %ld\n", memcmp(gd, gm, np * 4) ? "DIFFERENT" : "bit-identical", cs, nb, nw);
        CHECK(!memcmp(gd, gm, np * 4), "train, all tiles active: gradient bit-identical to the dense one");
        CHECK(nb > 0 && nw > 0, "masked backward-data (%ld) and wgrad (%ld) ran", nb, nw);
        nn_skip_off(); CHECK(!nn_skip_build(dct, P, P, P, 1, 3, 0, 1), "nn_skip_build");
        unet_forward(u, dx, xs, 1); unet_zero_grad(u); unet_backward(u, dgy); unet_grad_d2h(u, gm); nn_skip_counts_bwd(&nb, &nw);
        cs = cosine(gd, gm, np, &fin);
        printf("    train, CT masks (margin 1): gradient finite %d, cosine to dense %.4f, masked backward-data %ld, wgrad %ld\n", fin, cs, nb, nw);
        CHECK(fin, "train, CT masks: gradient finite");
        CHECK(nb > 0 && nw > 0, "train, CT masks: masked backward ran");
        CHECK(cs > 0.5, "train, CT masks: gradient cosine to dense %.4f", cs);
        nn_skip_off();
        unet_forward(u, dx, xs, 1); unet_zero_grad(u); unet_backward(u, dgy); unet_grad_d2h(u, gm);
        CHECK(!memcmp(gd, gm, np * 4), "train, nn_skip_off: gradient bit-identical to dense again");
        /* the checkpoint header */
        char path[256]; snprintf(path, sizeof path, "%s/test_skip_empty.%d.ckpt", getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp", (int)getpid());
        unet_cfg *uc = (unet_cfg *)unet_cfg_of(u); uc->skip_thr = 1; uc->skip_margin = 2;   /* the header writer reads the live cfg */
        unet_cfg pc = {}; int st = 0;
        CHECK(!unet_save(u, path, 5, nullptr) && !unet_peek(path, &pc, &st), "checkpoint save / peek");
        CHECK(pc.skip_thr == 1 && pc.skip_margin == 2, "checkpoint skip_empty round trip (%d %d)", pc.skip_thr, pc.skip_margin);
        uc->skip_thr = 0; CHECK(!unet_save(u, path, 5, nullptr) && !unet_peek(path, &pc, &st) && pc.skip_thr == 0, "no skip_empty key when off");
        remove(path); nn_free(dgy); free(gd); free(gm); free(gy);
    }
    if (bad) { printf("test_skip_empty: %d failed\n", bad); return 1; }
    printf("test_skip_empty: ok\n");
    return 0;
}
