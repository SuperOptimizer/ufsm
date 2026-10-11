/* Inference memory placement (INFERENCE_MEMORY_PLAN.md): with the inference aliasing on (UFSM_INFER_ALIAS, unet_set_infer_alias)
   the production network (6 levels 32..192, MX-fp4 activations, MX-fp8 stem input, recompute, 16-bit input in the a1 scratch as
   predict feeds it) gives byte-identical logits to the separate buffers, uses fewer bytes per window voxel (network input in
   enc[0].a2's buffer, fp32 logits in T1), and the predict CT scratch (unet_input_scratch2) may hold junk at the window start
   without changing the result; with one head channel and all channels, scale conditioning on and off. */
#include "unet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

static void run_s(const unet_cfg *cfg, int P, int nc, int alias, int slab, float *out, size_t *bytes);
/* the 16-bit network input as predict builds it (4 channels: z-scored CT and three direction components) */
static void *input_h16(int P, unsigned seed) {
    const size_t n = (size_t)4 * P * P * P; float *h = malloc(n * 4);
    for (size_t i = 0; i < n; i++) h[i] = sinf((float)((i * 2654435761u + seed) % 9973) * 0.01f) + (float)((i * 2246822519u + seed) % 1000) / 1000.f - 0.5f;
    float *d = nn_malloc(n * 4); nn_h2d(d, h, n * 4); free(h);
    void *x = nn_malloc(n * 2); nn_f32_to_h16(d, n, x, 1.f); nn_free(d);
    return x;
}

static void run_s(const unet_cfg *cfg, int P, int nc, int alias, int slab, float *out, size_t *bytes) {
    unet_set_infer_alias(alias); unet_set_infer_slab(slab);
    unet *u = unet_create(cfg); unet_init(u, 7);
    if (cfg->scale_cond) unet_set_scale(u, 2.4);
    unet_set_head_channels(u, 0, nc);
    const shape5 xs = {1, 4, P, P, P};
    *bytes = unet_infer_bytes(u, xs);
    unet_build(u, xs, 0);
    const size_t n = (size_t)4 * P * P * P;
    void *src = input_h16(P, 3);
    void *xd = unet_input_scratch(u, n * 2);   /* as predict: the 16-bit input in the shared a1 scratch */
    if (xd) nn_d2d(xd, src, n * 2); else xd = src;
    uint8_t *ct = unet_input_scratch2(u, (size_t)P * P * P);
    CHECK(!alias || ct, "P %d: no CT scratch with the aliasing on", P);
    if (ct) nn_zero(ct, (size_t)P * P * P);   /* the predict CT window: dead before the forward */
    if (ct) { uint8_t *junk = malloc((size_t)P * P * P); for (size_t i = 0; i < (size_t)P * P * P; i++) junk[i] = (uint8_t)(i * 131u); nn_h2d(ct, junk, (size_t)P * P * P); free(junk); }
    const int co = nc ? nc : cfg->cout;
    const float *lg = unet_forward_x(u, xd, xs, 0, 1);
    nn_d2h(out, lg, (size_t)co * P * P * P * 4);
    const char *e = nn_check(); CHECK(!e, "cuda: %s", e ? e : "");
    /* a second window through the same build (the registry dance repeats; the input scratch was overwritten by the forward) */
    if (xd != src) nn_d2d(xd, src, n * 2);
    lg = unet_forward_x(u, xd, xs, 0, 1);
    float *again = malloc((size_t)co * P * P * P * 4); nn_d2h(again, lg, (size_t)co * P * P * P * 4);
    CHECK(!memcmp(again, out, (size_t)co * P * P * P * 4), "P %d nc %d alias %d slab %d: a second forward differs", P, nc, alias, slab);
    free(again);
    nn_free(src); unet_free(u);
    unet_set_infer_slab(0);
}
static void run(const unet_cfg *cfg, int P, int nc, int alias, float *out, size_t *bytes) { run_s(cfg, P, nc, alias, 0, out, bytes); }

int main(void) {
    nn_init(0); nn_set_tf32(1); nn_set_f16(1); nn_set_prec(3); nn_set_gn_stored(1);
    unet_set_act_mx4(1); unet_set_act_mx8(0); unet_set_grad_mx8(0); unet_set_input_mx(1);
    if (unet_set_input_prec(8)) return 2;
    for (int sc = 0; sc < 2; sc++) {
        unet_cfg cfg = {6, {32, 64, 96, 128, 160, 192}, 4, 7, 8, 1, sc};
        for (int P = 64; P <= 128; P += 64) for (int k = 0; k < 2; k++) {
            const int nc = k ? 0 : 1, co = nc ? nc : cfg.cout;
            const size_t no = (size_t)co * P * P * P;
            float *a = malloc(no * 4), *b = malloc(no * 4); size_t ba, bb;
            run(&cfg, P, nc, 0, a, &ba);
            run(&cfg, P, nc, 1, b, &bb);
            const double v = (double)P * P * P;
            CHECK(!memcmp(a, b, no * 4), "scale_cond %d P %d head channels %d: aliased logits differ", sc, P, co);
            const double saved = (ba - (double)bb) / v, expect = nc ? 9.0 + 4.0 : 9.0;   /* input (MX-fp8, 4 -> 8 ch) + fp32 logits (one channel; 7 do not fit T1) */
            CHECK(fabs(saved - expect) < 0.5, "scale_cond %d P %d head channels %d: %.2f B per voxel saved, expected %.1f", sc, P, co, saved, expect);
            printf("scale_cond %d P %3d head %d ch: %.2f -> %.2f B per window voxel (%.2f saved), logits identical\n", sc, P, co, ba / v, bb / v, saved);
            free(a); free(b);
        }
    }
    /* phase 2: in-place z-slab convs at level 0 (UFSM_INFER_SLAB): the same logits up to the GroupNorm summation order */
    for (int sc = 0; sc < 2; sc++) {
        unet_cfg cfg = {6, {32, 64, 96, 128, 160, 192}, 4, 7, 8, 1, sc};
        for (int P = 64; P <= 128; P += 64) for (int si = 0; si < 3; si++) {
            static const int slabs[3] = {8, 12, 32};
            const int sl = slabs[si];
            const size_t no = (size_t)P * P * P;
            float *a = malloc(no * 4), *b = malloc(no * 4); size_t ba, bb;
            run_s(&cfg, P, 1, 1, 0, a, &ba);
            run_s(&cfg, P, 1, 1, sl, b, &bb);
            double md = 0, ms = 0; size_t nbig = 0;
            for (size_t i = 0; i < no; i++) { const double d = fabs((double)a[i] - b[i]); md = fmax(md, d); ms += fabs((double)a[i]); nbig += d > 1e-3; }
            const double v = (double)P * P * P;
            CHECK(md < 2e-2 && nbig < no / 10000 + 1, "scale_cond %d P %d slab %d: logits differ by up to %.3g (%zu voxels > 1e-3; mean |logit| %.3g)", sc, P, sl, md, nbig, ms / no);
            printf("scale_cond %d P %3d slab %2d: %.2f -> %.2f B per window voxel, max |logit difference| %.2e (%zu > 1e-3)\n", sc, P, sl, ba / v, bb / v, md, nbig);
            free(a); free(b);
        }
    }
    /* phase 2b: the stem input built per slab from the uint8 CT (unet_forward_ct, the CT at the end of X) == predict's 16-bit input
       + the full-window conversion */
    for (int sc = 0; sc < 2; sc++) for (int P = 64; P <= 128; P += 64) for (int si = 0; si < 2; si++) {
        static const int slabs[2] = {8, 32};
        unet_cfg cfg = {6, {32, 64, 96, 128, 160, 192}, 4, 7, 8, 1, sc};
        const size_t p3 = (size_t)P * P * P;
        uint8_t *hct = malloc(p3); float hrow[2 * 128];
        for (size_t i = 0; i < p3; i++) { const int z = (int)(i / ((size_t)P * P)), y = (int)((i / P) % P), x = (int)(i % P); hct[i] = (uint8_t)(x < 5 ? 0 : 60 + 50 * sinf(0.21f * x + 0.13f * y) + 30 * cosf(0.17f * z) + (i * 2654435761u) % 23); }
        for (int z = 0; z < P; z++) { hrow[z] = -300.f + 0.5f * z; hrow[P + z] = 40.f - 0.25f * z; }   /* window origin relative to the axis per plane */
        float *drow = nn_malloc(2 * P * 4); nn_h2d(drow, hrow, 2 * P * 4);
        const float mean = 92.f, isd = 1.f / 31.f;
        float *a = malloc(p3 * 4), *b = malloc(p3 * 4); size_t ba, bb;
        {   /* reference: predict's 16-bit input in the a1 scratch, phase 1 layout */
            unet_set_infer_alias(1); unet_set_infer_slab(0);
            unet *u = unet_create(&cfg); unet_init(u, 7); if (sc) unet_set_scale(u, 2.4); unet_set_head_channels(u, 0, 1);
            const shape5 xs = {1, 4, P, P, P}; ba = unet_infer_bytes(u, xs); unet_build(u, xs, 0);
            uint8_t *ct = unet_input_scratch2(u, p3); void *xd = unet_input_scratch(u, 8 * p3);
            CHECK(ct && xd, "reference scratch");
            nn_h2d(ct, hct, p3); nn_pred_input(ct, P, mean, isd, drow, drow + P, 1, xd, 1);
            nn_d2h(a, unet_forward_x(u, xd, xs, 0, 1), p3 * 4); unet_free(u);
        }
        {
            unet_set_infer_alias(1); unet_set_infer_slab(slabs[si]);
            unet *u = unet_create(&cfg); unet_init(u, 7); if (sc) unet_set_scale(u, 2.4); unet_set_head_channels(u, 0, 1); unet_set_ct_input(u, 1);
            const shape5 xs = {1, 4, P, P, P}; bb = unet_infer_bytes(u, xs); unet_build(u, xs, 0);
            CHECK(unet_ct_stem(u), "P %d: no CT-stem build", P);
            uint8_t *ct = unet_input_scratch2(u, p3); CHECK(ct, "P %d: no CT scratch in X", P);
            nn_h2d(ct, hct, p3);
            nn_d2h(b, unet_forward_ct(u, ct, mean, isd, drow, drow + P, 1, xs, 1), p3 * 4);
            const char *e = nn_check(); CHECK(!e, "cuda: %s", e ? e : "");
            nn_h2d(ct, hct, p3);   /* a second window (the CT was consumed) */
            float *c = malloc(p3 * 4); nn_d2h(c, unet_forward_ct(u, ct, mean, isd, drow, drow + P, 1, xs, 1), p3 * 4);
            CHECK(!memcmp(b, c, p3 * 4), "P %d slab %d: a second CT forward differs", P, slabs[si]); free(c);
            unet_free(u); unet_set_infer_slab(0);
        }
        size_t nd = 0; double md = 0; for (size_t i = 0; i < p3; i++) { const double d = fabs((double)a[i] - b[i]); nd += d != 0; md = fmax(md, d); }
        CHECK(nd == 0, "scale_cond %d P %d slab %d: CT-stem logits differ at %zu voxels (max %.3g)", sc, P, slabs[si], nd, md);
        printf("scale_cond %d P %3d slab %2d CT stem: %.2f -> %.2f B per window voxel, logits %s\n", sc, P, slabs[si], ba / (double)p3, bb / (double)p3, nd ? "DIFFER" : "identical");
        free(a); free(b); free(hct); nn_free(drow);
    }
    if (fails) return 1;
    printf("test_infer_mem: ok\n");
    return 0;
}
