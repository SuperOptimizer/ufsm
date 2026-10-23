/* the 32- and 48-output MX heads (dino-train embeddings; + the recipe's dense heads): a CO-output net against a 16-output net with the same trunk and the first 16 head rows
   (unet_init_from_map), production storage (MX-fp4 activations / gradients, SR): logits 0..15 bit for bit, and with a logit gradient that is
   zero on outputs 16..CO-1 the trunk and head-row gradients agree (summation order only). */
#include "unet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static uint64_t rs = 777;
static float h2f(uint16_t v) { int e = (v >> 10) & 31, m = v & 1023; float f = e == 0 ? ldexpf((float)m, -24) : e == 31 ? INFINITY : ldexpf((float)(m | 1024), e - 25); return (v >> 15) ? -f : f; }
static uint16_t f2h(float f) { uint32_t u; memcpy(&u, &f, 4); const uint32_t s = (u >> 16) & 0x8000; const float a = fabsf(f);
    if (a < 6.1035e-5f) { const int m = (int)lrintf(a * 16777216.f); return (uint16_t)(s | m); }   /* subnormal (m 1024: the smallest normal) */
    int e; const float fr = frexpf(a, &e); int m = (int)lrintf((fr * 2.f - 1.f) * 1024.f); e += 14; if (m == 1024) { m = 0; e++; } if (e >= 31) return (uint16_t)(s | 0x7c00);
    return (uint16_t)(s | (uint32_t)e << 10 | (uint32_t)m); }
typedef struct { unet *u; const float *gh; void *sd; uint16_t *sl; size_t ns, p3, hw; int P, Z, bad, co; } ctx_t;
static void slab_cb(void *v) {   /* the backward's head step: the logit gradient per slab, fp16 and scaled by the gradient scale as the whole path */
    ctx_t *c = v; const float gs = nn_get_grad_scale();
    for (int z0 = 0; z0 < c->P; z0 += c->Z) {
        for (int ch = 0; ch < c->co; ch++) for (size_t i = 0; i < c->ns; i++) c->sl[ch * c->ns + i] = f2h(c->gh[ch * c->p3 + (size_t)z0 * c->hw + i] * gs);
        nn_h2d(c->sd, c->sl, (size_t)c->co * c->ns * 2);
        if (unet_head_bwd_z(c->u, z0, z0 + c->Z, c->sd, z0 == 0)) c->bad = 1;
    }
}
static float frand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 11) * 0x1.0p-53 * 2 - 1); }
static int check(int CO) {
    unet_cfg c32 = {0}; c32.nlev = 3; c32.widths[0] = 16; c32.widths[1] = 32; c32.widths[2] = 64; c32.cin = 1; c32.cout = CO; c32.G = 8; c32.down_norm = 1;
    unet_cfg c16 = c32; c16.cout = 16;
    unet *u32 = unet_create(&c32); unet_init(u32, 5);
    char path[] = "/tmp/ufsm_test_head32_XXXXXX"; int fd = mkstemp(path); if (fd >= 0) close(fd);
    if (unet_save(u32, path, 0, nullptr)) { printf("cannot save\n"); return 1; }
    unet *u16 = unet_create(&c16); unet_init(u16, 9);
    int rows[16]; for (int i = 0; i < 16; i++) rows[i] = i;
    if (unet_init_from_map(u16, path, rows) < 0) { printf("init_from failed\n"); return 1; }
    unlink(path);
    const int P = 32; const size_t p3 = (size_t)P * P * P; const shape5 xs = {1, 1, P, P, P};
    float *xh = malloc(p3 * 4); for (size_t i = 0; i < p3; i++) xh[i] = frand();
    float *x = nn_malloc(p3 * 4); nn_h2d(x, xh, p3 * 4);
    float *gh = calloc(p3 * CO, 4); for (size_t i = 0; i < p3 * 16; i++) gh[i] = frand() * 1e-3f;
    float *g32 = nn_malloc(p3 * CO * 4), *g16 = nn_malloc(p3 * 16 * 4); nn_h2d(g32, gh, p3 * CO * 4); nn_h2d(g16, gh, p3 * 16 * 4);
    int bad = 0;
    nn_set_sr_step(100); const float *l32 = unet_forward(u32, x, xs, 1); const int h32 = unet_logits_h16(u32);
    const size_t eb = h32 ? 2 : 4; uint8_t *a = malloc(p3 * 16 * eb); nn_d2h(a, l32, p3 * 16 * eb);
    unet_zero_grad(u32); nn_set_sr_step(200); unet_backward(u32, g32);
    nn_set_sr_step(100); const float *l16 = unet_forward(u16, x, xs, 1);
    uint8_t *b = malloc(p3 * 16 * eb); nn_d2h(b, l16, p3 * 16 * eb);
    unet_zero_grad(u16); nn_set_sr_step(200); unet_backward(u16, g16);
    size_t neq = 0; for (size_t i = 0; i < p3 * 16 * eb; i++) neq += a[i] != b[i];
    printf("%d outputs: logits 0..15: %zu bytes differ (of %zu)%s\n", CO, neq, p3 * 16 * eb, neq ? "  FAIL" : ""); bad += neq != 0;
    const size_t n32 = unet_nparams(u32), n16 = unet_nparams(u16); float *ga = malloc(n32 * 4), *gb = malloc(n16 * 4);
    unet_grad_d2h(u32, ga); unet_grad_d2h(u16, gb);
    size_t hw32, hb32, hw16, hb16; unet_head_offsets(u32, &hw32, &hb32); unet_head_offsets(u16, &hw16, &hb16);
    double d2 = 0, r2 = 0;
    for (size_t i = 0; i < hw32; i++) { const double d = (double)ga[i] - gb[i]; d2 += d * d; r2 += (double)gb[i] * gb[i]; }   /* trunk before the head */
    for (size_t i = 0; i < 16 * 16; i++) { const double d = (double)ga[hw32 + i] - gb[hw16 + i]; d2 += d * d; r2 += (double)gb[hw16 + i] * gb[hw16 + i]; }   /* head rows 0..15 */
    const size_t t32 = hb32 + CO, t16 = hb16 + 16;   /* params after the head (down norms) */
    for (size_t i = 0; t32 + i < n32 && t16 + i < n16; i++) { const double d = (double)ga[t32 + i] - gb[t16 + i]; d2 += d * d; r2 += (double)gb[t16 + i] * gb[t16 + i]; }
    double z2 = 0; for (size_t i = 16 * 16; i < (size_t)CO * 16; i++) z2 += (double)ga[hw32 + i] * ga[hw32 + i];
    const double rel = sqrt(d2 / (r2 > 0 ? r2 : 1));
    printf("%d outputs: gradients: rel diff %.3g, rows 16..%d |g| %.3g%s\n", CO, rel, CO - 1, sqrt(z2), rel < 1e-3 && z2 == 0 ? "" : "  FAIL"); bad += !(rel < 1e-3 && z2 == 0);
    {   /* head slabs (unet_set_head_ext, dino-train --head-slab): the 32-output head per z-slab of 8 planes against the whole head */
        unet *us = unet_create(&c32); unet_init(us, 5);
        nn_set_logits_h16(1); unet_set_head_ext(us, 1);
        nn_set_sr_step(100); unet_forward(us, x, xs, 1);
        const int Z = 8; const size_t hw = (size_t)P * P, ns = (size_t)Z * hw;
        uint16_t *sl = malloc((size_t)CO * ns * 2); void *sd = nn_malloc((size_t)CO * ns * 2);
        double e2 = 0, n2 = 0;
        float *whole = malloc(p3 * CO * 4);
        nn_set_sr_step(100); l32 = unet_forward(u32, x, xs, 1);   /* (the whole head's logits again: u32's buffer saw a backward since) */
        if (h32) { uint16_t *t = malloc(p3 * CO * 2); nn_d2h(t, l32, p3 * CO * 2); for (size_t i = 0; i < p3 * CO; i++) whole[i] = h2f(t[i]); free(t); }
        else nn_d2h(whole, l32, p3 * CO * 4);
        int nc = 0;
        for (int z0 = 0; z0 < P; z0 += Z) {
            if (unet_head_fwd_z(us, z0, z0 + Z, sd)) { nc = 1; break; }
            nn_d2h(sl, sd, (size_t)CO * ns * 2);
            for (int c = 0; c < CO; c++) for (size_t v = 0; v < ns; v++) { const double a = h2f(sl[c * ns + v]), b = whole[c * p3 + (size_t)z0 * hw + v]; e2 += (a - b) * (a - b); n2 += b * b; }
        }
        const double el = sqrt(e2 / (n2 > 0 ? n2 : 1));
        printf("%d outputs: head slabs: logits rel diff %.3g%s\n", CO, el, !nc && el < 2e-3 ? "" : "  FAIL"); bad += !(!nc && el < 2e-3);
        /* backward: the logit gradient gh (rows 0..15, scaled by the gradient scale as the whole path does) per slab */
        ctx_t cx = {us, gh, sd, sl, ns, p3, hw, P, Z, 0, CO};
        unet_set_head_cb(us, slab_cb, &cx);
        unet_zero_grad(us); nn_set_sr_step(200); unet_backward(us, g32);
        float *gs_ = malloc(n32 * 4); unet_grad_d2h(us, gs_);
        double d = 0, r = 0; for (size_t i = 0; i < n32; i++) { const double q = (double)gs_[i] - ga[i]; d += q * q; r += (double)ga[i] * ga[i]; }
        const double eg = sqrt(d / (r > 0 ? r : 1));
        printf("%d outputs: head slabs: gradients rel diff %.3g%s\n", CO, eg, !cx.bad && eg < 1e-3 ? "" : "  FAIL"); bad += !(!cx.bad && eg < 1e-3);
        nn_set_logits_h16(0);
    }
    return bad;
}
int main(void) {
    setenv("UFSM_ACT_MX4", "1", 0); setenv("UFSM_XIN_PREC", "8", 0); setenv("UFSM_GRAD_MX8", "1", 0); setenv("UFSM_GRAD_MX4", "1", 0);
    setenv("UFSM_F4_WGRAD", "1", 0); setenv("UFSM_DOWN_NORM", "1", 0);
    if (nn_init(0)) { printf("SKIP: no GPU\n"); return 0; }
    nn_set_tf32(1); nn_set_f16(1); nn_set_prec_policy("all=fp4:fp4:fp4,enc0.c1=fp16"); nn_set_gn_stored(1); nn_set_sr(1);
    const int bad = check(32) + check(48);
    printf("%s\n", bad ? "test_head32: FAIL" : "test_head32: OK");
    return bad != 0;
}
