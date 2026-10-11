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

/* the 16-bit network input as predict builds it (4 channels: z-scored CT and three direction components) */
static void *input_h16(int P, unsigned seed) {
    const size_t n = (size_t)4 * P * P * P; float *h = malloc(n * 4);
    for (size_t i = 0; i < n; i++) h[i] = sinf((float)((i * 2654435761u + seed) % 9973) * 0.01f) + (float)((i * 2246822519u + seed) % 1000) / 1000.f - 0.5f;
    float *d = nn_malloc(n * 4); nn_h2d(d, h, n * 4); free(h);
    void *x = nn_malloc(n * 2); nn_f32_to_h16(d, n, x, 1.f); nn_free(d);
    return x;
}

static void run(const unet_cfg *cfg, int P, int nc, int alias, float *out, size_t *bytes) {
    unet_set_infer_alias(alias);
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
    CHECK(!memcmp(again, out, (size_t)co * P * P * P * 4), "P %d nc %d alias %d: a second forward differs", P, nc, alias);
    free(again);
    nn_free(src); unet_free(u);
}

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
    if (fails) return 1;
    printf("test_infer_mem: ok\n");
    return 0;
}
