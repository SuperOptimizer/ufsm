/* Inference memory placement (INFERENCE_MEMORY_PLAN.md): with the inference aliasing on (UFSM_INFER_ALIAS, unet_set_infer_alias)
   the production network (6 levels 32..192, MX-fp4 activations, MX-fp8 stem input, recompute, 16-bit input in the a1 scratch as
   predict feeds it) gives byte-identical logits to the separate buffers, uses fewer bytes per window voxel (network input in
   enc[0].a2's buffer, fp32 logits in T1), and the predict CT scratch (unet_input_scratch2) may hold junk at the window start
   without changing the result; with one head channel and all channels, scale conditioning on and off. Then the in-place slabs
   (phase 2), the stem built from the CT (2b), the coarse offload (phase 3, UFSM_INFER_OFFLOAD 1 / 2 = dec[0]'s conv1 on its own
   temp / 0) and the spatial split of a CT window (two GPUs, or both halves on one) against the phase-1 logits. */
#include "split.h"
#include "unet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
static int g_quick;   /* UFSM_TEST_QUICK (make test): fewer CT-stem cases (P 128, every path once) */
static int g_lh16;   /* the CT helpers below: fp16 logits (predict's unet_set_fp16_logits), 2 B per output voxel */
static float h2f(uint16_t h) {   /* IEEE half -> float (the test's comparisons) */
    const uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023;
    uint32_t f;
    if (e == 0) { if (!m) f = s; else { int k = -1; uint32_t mm = m; do { k++; mm <<= 1; } while (!(mm & 1024)); f = s | ((uint32_t)(112 - k) << 23) | ((mm & 1023) << 13); } }
    else if (e == 31) f = s | 0x7f800000u | (m << 13);
    else f = s | ((e + 112) << 23) | (m << 13);
    float r; memcpy(&r, &f, 4); return r;
}
/* logits a vs b (fp32, or fp16 with g_lh16) over n voxels: the count that differ, the largest difference */
static size_t out_diff(const void *a, const void *b, size_t n, double *md) {
    size_t nd = 0; *md = 0;
    for (size_t i = 0; i < n; i++) {
        const double x = g_lh16 ? h2f(((const uint16_t *)a)[i]) : ((const float *)a)[i], y = g_lh16 ? h2f(((const uint16_t *)b)[i]) : ((const float *)b)[i];
        const double d = fabs(x - y); nd += d != 0; if (d > *md) *md = d;
    }
    return nd;
}
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

/* the one-GPU phase-1 reference of a CT window: predict's 16-bit input in the a1 scratch, no slabs */
static void ct_ref(const unet_cfg *cfg, int sc, int P, const uint8_t *hct, const float *hrow, float *out) {
    const size_t p3 = (size_t)P * P * P;
    unet_set_infer_alias(1); unet_set_infer_slab(0);
    unet *u = unet_create(cfg); unet_init(u, 7); if (sc) unet_set_scale(u, 2.4); unet_set_head_channels(u, 0, 1);
    if (g_lh16) unet_set_fp16_logits(u, 1);
    const shape5 xs = {1, 4, P, P, P}; unet_build(u, xs, 0);
    uint8_t *ct = unet_input_scratch2(u, p3); void *xd = unet_input_scratch(u, 8 * p3);
    float *drow = nn_malloc(2 * (size_t)P * 4); nn_h2d(drow, hrow, 2 * (size_t)P * 4);
    nn_h2d(ct, hct, p3); nn_pred_input(ct, P, 92.f, 1.f / 31.f, drow, drow + P, 1, xd, 1);
    nn_d2h(out, unet_forward_x(u, xd, xs, 0, 1), p3 * (g_lh16 ? 2 : 4)); unet_free(u); nn_free(drow);
}
/* a CT-stem window of depth D (planes z0 .. of the whole one) on the current GPU: the network, the CT in its scratch, the axis rows */
typedef struct { unet *u; uint8_t *ct; float *drow, *out; int z0, D, P; size_t bytes; } half_t;
static void half_setup(half_t *h, const unet_cfg *cfg, int sc, int P, int z0, int D, const uint8_t *hct, const float *hrow) {
    h->u = unet_create(cfg); unet_init(h->u, 7); if (sc) unet_set_scale(h->u, 2.4); unet_set_head_channels(h->u, 0, 1); unet_set_ct_input(h->u, 1);
    if (g_lh16) unet_set_fp16_logits(h->u, 1);
    const shape5 xs = {1, 4, D, P, P}; h->bytes = unet_infer_bytes(h->u, xs); unet_build(h->u, xs, 0);
    const size_t pl = (size_t)P * P;
    h->ct = unet_input_scratch2(h->u, D * pl);
    float *r = malloc(2 * (size_t)D * 4); for (int j = 0; j < D; j++) { r[j] = hrow[z0 + j]; r[D + j] = hrow[P + z0 + j]; }
    h->drow = nn_malloc(2 * (size_t)D * 4); nn_h2d(h->drow, r, 2 * (size_t)D * 4); free(r);
    nn_h2d(h->ct, hct + z0 * pl, D * pl);
    h->out = malloc((size_t)D * pl * 4); h->z0 = z0; h->D = D; h->P = P;
}
static void half_fwd(half_t *h) {
    const shape5 xs = {1, 4, h->D, h->P, h->P};
    nn_d2h(h->out, unet_forward_ct(h->u, h->ct, 92.f, 1.f / 31.f, h->drow, h->drow + h->D, 1, xs, 1), (size_t)h->D * h->P * h->P * (g_lh16 ? 2 : 4));
}
static void half_free(half_t *h) { unet_free(h->u); nn_free(h->drow); free(h->out); }
static size_t ct_run(const unet_cfg *cfg, int sc, int P, int slab, int off, const uint8_t *hct, const float *hrow, float *out) {
    unet_set_infer_alias(1); unet_set_infer_slab(slab); unet_set_infer_offload(off);
    half_t h; half_setup(&h, cfg, sc, P, 0, P, hct, hrow);
    CHECK(unet_ct_stem(h.u) && h.ct, "P %d slab %d offload %d: no CT-stem build", P, slab, off);
    half_fwd(&h); memcpy(out, h.out, (size_t)P * P * P * (g_lh16 ? 2 : 4));
    const size_t b = h.bytes; half_free(&h); return b;
}
static half_t HS[2];
static void half_job(int side, void *arg) { (void)arg; half_fwd(&HS[side]); }
static size_t split_run_ct(split_ctx *ctx, const unet_cfg *cfg, int sc, int P, int slab, int off, const uint8_t *hct, const float *hrow, float *out) {
    const int H0 = 1 << (cfg->nlev - 1), Dl = P / 2 + H0, dev[2] = {0, nn_init(1) ? 0 : 1};
    unet_set_infer_alias(1); unet_set_infer_slab(slab); unet_set_infer_offload(off);
    for (int s = 0; s < 2; s++) {
        nn_init(dev[s]);
        half_setup(&HS[s], cfg, sc, P, s ? P / 2 - H0 : 0, Dl, hct, hrow);
        unet_set_split(HS[s].u, s, H0, split_halo); unet_set_split_async(HS[s].u, split_halo_begin, split_halo_end);
        CHECK(unet_ct_stem(HS[s].u), "split P %d: no CT-stem build", P);
    }
    split_run(ctx, half_job, nullptr);
    const size_t pl = (size_t)P * P, b = HS[0].bytes;
    const size_t ob = g_lh16 ? 2 : 4;
    memcpy(out, HS[0].out, (size_t)(P / 2) * pl * ob);                       /* side 0 owns its planes [0, P/2) */
    memcpy((char *)out + (size_t)(P / 2) * pl * ob, (char *)HS[1].out + (size_t)H0 * pl * ob, (size_t)(P / 2) * pl * ob);   /* side 1 its local [H0, Dl) */
    for (int s = 0; s < 2; s++) { nn_init(dev[s]); half_free(&HS[s]); }
    nn_init(0);
    return b;
}

int main(void) {
    g_quick = getenv("UFSM_TEST_QUICK") && atoi(getenv("UFSM_TEST_QUICK"));
    nn_init(0); nn_set_tf32(1); nn_set_f16(1); nn_set_prec(3); nn_set_gn_stored(1);
    if (!nn_mx_ok()) { printf("test_infer_mem: skipped (the MX paths are off: no fp4 / fp8 tensor cores and UFSM_MX_EMU not 1)\n"); return 0; }   /* (MX-fp4 placement only) */
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
    /* the coarse offload (UFSM_INFER_OFFLOAD: 1, 2 = dec[0]'s conv1 on its own temp, 0 off) and the spatial split (the window's two z
       halves, on two GPUs or both on one): CT-stem logits identical to the one-GPU phase-1 reference (split: on the planes each half
       owns) */
    {
        int dev1 = 1; if (nn_init(1)) { dev1 = 0; nn_check(); } nn_init(0);
        split_ctx *ctx = split_create(0, dev1);
        for (int sc = 0; sc < 2; sc++) for (int P = 128; P <= 192; P += 64) {
            if (g_quick && (P > 128 || sc)) continue;   /* (make test: P 128, scale_cond 0 (1 above); the rest in make test-slow) */
            unet_cfg cfg = {6, {32, 64, 96, 128, 160, 192}, 4, 7, 8, 1, sc};
            const size_t p3 = (size_t)P * P * P;
            uint8_t *hct = malloc(p3); float *hrow = malloc(2 * (size_t)P * 4);
            for (size_t i = 0; i < p3; i++) { const int z = (int)(i / ((size_t)P * P)), y = (int)((i / P) % P), x = (int)(i % P); hct[i] = (uint8_t)(x < 5 ? 0 : 60 + 50 * sinf(0.21f * x + 0.13f * y) + 30 * cosf(0.17f * z) + (i * 2654435761u) % 23); }
            for (int z = 0; z < P; z++) { hrow[z] = -300.f + 0.5f * z; hrow[P + z] = 40.f - 0.25f * z; }
            float *ref = malloc(p3 * 4), *b = malloc(p3 * 4);
            ct_ref(&cfg, sc, P, hct, hrow, ref);
            for (int off = 0; off < 3; off++) for (int sl = 0; sl < 2; sl++) {
                const int slab = sl ? 16 : 64;
                size_t by = ct_run(&cfg, sc, P, slab, off, hct, hrow, b);
                size_t nd = 0; for (size_t i = 0; i < p3; i++) nd += ref[i] != b[i];
                CHECK(nd == 0, "scale_cond %d P %d offload %d slab %d: CT-stem logits differ at %zu voxels", sc, P, off, slab, nd);
                printf("scale_cond %d P %3d offload %d slab %2d: %.2f B per window voxel, logits %s\n", sc, P, off, slab, by / (double)p3, nd ? "DIFFER" : "identical");
                double md = 0; size_t sb = split_run_ct(ctx, &cfg, sc, P, slab, off, hct, hrow, b);
                nd = 0; for (size_t i = 0; i < p3; i++) { const double d = fabs((double)ref[i] - b[i]); nd += d != 0; md = fmax(md, d); }
                CHECK(md < 1e-3, "scale_cond %d P %d offload %d slab %d split: logits differ by up to %.3g (%zu voxels)", sc, P, off, slab, md, nd);
                printf("scale_cond %d P %3d offload %d slab %2d split (local depth %d): %.2f B per half voxel, %zu voxels differ (max %.2e)\n", sc, P, off, slab, P / 2 + 32, sb / ((double)P * P * (P / 2 + 32)), nd, md);
            }
            free(ref); free(b); free(hct); free(hrow);
        }
        split_free(ctx); nn_init(0);
        unet_set_infer_offload(1); unet_set_infer_slab(0);
    }
    /* the 16-wide production net (7 levels 16, 16, 32, 64, 128, 192, 256; side head: 2 outputs, one written): in-place slabs at levels 0
       and 1 with 16-channel MX blocks, the coarse offload, the split, the CT stem with an MX-fp8 or an MX-fp4 input (train
       --gpu-input's UFSM_XIN_PREC=4): logits identical to the phase-1 reference of the same input precision */
    {
        int dev1 = 1; if (nn_init(1)) { dev1 = 0; nn_check(); } nn_init(0);
        split_ctx *ctx = split_create(0, dev1);
        const unet_cfg cfg = {7, {16, 16, 32, 64, 128, 192, 256}, 4, 2, 8, 0, 0};
        for (int lh = 0; lh < 2; lh++) for (int ip = 8; ip >= 4; ip -= 4) for (int P = 128; P <= 256; P += 128) {
            if (g_quick && (P > 128 || (!lh && ip == 4))) continue;   /* (make test: P 128; fp4 input with fp16 logits only) */
            if (lh && ip == 8) continue;   /* (fp16 logits over X: the production input precision) */
            g_lh16 = lh;
            if (unet_set_input_prec(ip)) return 2;
            const size_t p3 = (size_t)P * P * P;
            uint8_t *hct = malloc(p3); float *hrow = malloc(2 * (size_t)P * 4);
            for (size_t i = 0; i < p3; i++) { const int z = (int)(i / ((size_t)P * P)), y = (int)((i / P) % P), x = (int)(i % P); hct[i] = (uint8_t)(x < 5 ? 0 : 60 + 50 * sinf(0.21f * x + 0.13f * y) + 30 * cosf(0.17f * z) + (i * 2654435761u) % 23); }
            for (int z = 0; z < P; z++) { hrow[z] = -300.f + 0.5f * z; hrow[P + z] = 40.f - 0.25f * z; }
            float *ref = malloc(p3 * 4), *b = malloc(p3 * 4);
            ct_ref(&cfg, 0, P, hct, hrow, ref);
            size_t rb; { unet_set_infer_alias(1); unet_set_infer_slab(0); unet *u = unet_create(&cfg); unet_init(u, 7); unet_set_head_channels(u, 0, 1); if (lh) unet_set_fp16_logits(u, 1); rb = unet_infer_bytes(u, (shape5){1, 4, P, P, P}); unet_free(u); }
            for (int off = 0; off < 3; off++) for (int sl = 0; sl < 2; sl++) {
                const int slab = sl ? 16 : 64;
                size_t by = ct_run(&cfg, 0, P, slab, off, hct, hrow, b);
                double md; size_t nd = out_diff(ref, b, p3, &md);
                CHECK(nd == 0, "16-wide input fp%d %s logits P %d offload %d slab %d: CT-stem logits differ at %zu voxels (max %.3g)", ip, lh ? "fp16" : "fp32", P, off, slab, nd, md);
                printf("16-wide input fp%d %s logits P %3d offload %d slab %2d: %.2f -> %.2f B per window voxel, logits %s\n", ip, lh ? "fp16" : "fp32", P, off, slab, rb / (double)p3, by / (double)p3, nd ? "DIFFER" : "identical");
                size_t sb = split_run_ct(ctx, &cfg, 0, P, slab, off, hct, hrow, b);
                nd = out_diff(ref, b, p3, &md);
                CHECK(md < 1e-3, "16-wide input fp%d %s logits P %d offload %d slab %d split: logits differ by up to %.3g (%zu voxels)", ip, lh ? "fp16" : "fp32", P, off, slab, md, nd);
                printf("16-wide input fp%d %s logits P %3d offload %d slab %2d split (local depth %d): %.2f B per half voxel, %zu voxels differ (max %.2e)\n", ip, lh ? "fp16" : "fp32", P, off, slab, P / 2 + 64, sb / ((double)P * P * (P / 2 + 64)), nd, md);
            }
            free(ref); free(b); free(hct); free(hrow);
        }
        split_free(ctx); nn_init(0);
        unet_set_infer_offload(1); unet_set_infer_slab(0); g_lh16 = 0;
        if (unet_set_input_prec(8)) return 2;
    }
    /* the inference speed paths, each off vs on, bitwise: overlapped slab copies (unet_set_slab_ovl / UFSM_SLAB_OVL: two slab temps,
       the plane copies on the aux stream), the constant-bank fp16 head (nn_set_conv1_const / UFSM_CONV1_CONST), the frozen GroupNorm
       statistics replayed from their device copy (predict --gn-frozen). CT-stem windows with fp16 logits as predict runs them: the
       16-wide ink-like net (one head channel) and a 32-wide 6-level net with 11 head channels (wind96's head), slabs 16 / 64,
       offload 0 / 1, a second window through the same build */
    {
        static const unet_cfg cfgs[2] = {{7, {16, 16, 32, 64, 128, 192, 256}, 4, 2, 8, 0, 0}, {6, {32, 64, 96, 128, 160, 192}, 4, 11, 8, 1, 0}};
        if (unet_set_input_prec(8)) return 2;
        nn_init(0); for (int k = 0; k < 4 && nn_check(); k++) {}   /* (split_run_ct's probe of a second GPU leaves an invalid-device error on a one-GPU host: g_err, then CUDA's last error) */
        for (int ci = 0; ci < 2; ci++) for (int P = 128; P <= 256; P += 128) for (int frz = 0; frz < 2; frz++) {
            const unet_cfg *cfg = &cfgs[ci]; const int nc = ci ? 0 : 1, co = nc ? nc : cfg->cout;
            const size_t p3 = (size_t)P * P * P, ob = (size_t)co * p3 * 2;
            uint8_t *hct = malloc(p3); float *hrow = malloc(2 * (size_t)P * 4);
            for (size_t i = 0; i < p3; i++) { const int z = (int)(i / ((size_t)P * P)), y = (int)((i / P) % P), x = (int)(i % P); hct[i] = (uint8_t)(x < 5 ? 0 : 60 + 50 * sinf(0.21f * x + 0.13f * y) + 30 * cosf(0.17f * z) + (i * 2654435761u) % 23); }
            for (int z = 0; z < P; z++) { hrow[z] = -300.f + 0.5f * z; hrow[P + z] = 40.f - 0.25f * z; }
            float *drow = nn_malloc(2 * (size_t)P * 4); nn_h2d(drow, hrow, 2 * (size_t)P * 4);
            uint8_t *ref = malloc(ob), *b = malloc(ob), *b2 = malloc(ob);
            float *gst = nullptr;
            for (int mode = 0; mode < 5; mode++) {   /* 0: both paths off (the reference); 1-4: slab 16 / 64 x offload 0 / 1, both on */
                const int on = mode > 0, slab = mode == 0 ? 64 : mode <= 2 ? 16 : 64, off = mode == 0 ? 1 : (mode & 1);
                unet_set_slab_ovl(on); nn_set_conv1_const(on);
                unet_set_infer_alias(1); unet_set_infer_slab(slab); unet_set_infer_offload(off);
                unet *u = unet_create(cfg); unet_init(u, 7); unet_set_head_channels(u, 0, nc); unet_set_ct_input(u, 1); unet_set_fp16_logits(u, 1);
                const shape5 xs = {1, 4, P, P, P}; unet_build(u, xs, 0);
                CHECK(unet_ct_stem(u), "speed paths cfg %d P %d: no CT-stem build", ci, P);
                uint8_t *ct = unet_input_scratch2(u, p3);
                if (frz) {   /* statistics of the first window (the reference build), frozen for every build */
                    if (!gst) { gst = malloc(unet_gn_nstats(u) * 4); nn_h2d(ct, hct, p3); (void)unet_forward_ct(u, ct, 92.f, 1.f / 31.f, drow, drow + P, 1, xs, 1); unet_gn_stats_get(u, gst); }
                    unet_gn_freeze(u, gst);
                }
                uint8_t *dst = mode ? b : ref;
                for (int rep = 0; rep < 2; rep++) {
                    nn_h2d(ct, hct, p3);   /* (the forward consumes the CT window) */
                    nn_d2h(rep ? b2 : dst, unet_forward_ct(u, ct, 92.f, 1.f / 31.f, drow, drow + P, 1, xs, 1), ob);
                }
                const char *e = nn_check(); CHECK(!e, "cuda: %s", e ? e : "");
                CHECK(!memcmp(dst, b2, ob), "speed paths cfg %d P %d frozen %d mode %d: a second window differs", ci, P, frz, mode);
                if (mode) {
                    size_t nd = 0; for (size_t i = 0; i < ob / 2; i++) nd += ((uint16_t *)ref)[i] != ((uint16_t *)b)[i];
                    CHECK(nd == 0, "speed paths cfg %d P %d frozen %d slab %d offload %d: fp16 logits differ at %zu of %zu", ci, P, frz, slab, off, nd, ob / 2);
                    printf("speed paths: %s P %3d frozen GN %d slab %2d offload %d: %d fp16 logit channels %s\n", ci ? "32-wide" : "16-wide", P, frz, slab, off, co, nd ? "DIFFER" : "identical");
                }
                unet_free(u);
            }
            free(gst); free(ref); free(b); free(b2); free(hct); free(hrow); nn_free(drow);
        }
        unet_set_slab_ovl(-1); nn_set_conv1_const(1); unet_set_infer_offload(1); unet_set_infer_slab(0);   /* (-1: the default, by width) */
    }
    if (fails) return 1;
    printf("test_infer_mem: ok\n");
    return 0;
}
