/* the 32-output MX head (dino-train embeddings): a 32-output net against a 16-output net with the same trunk and the first 16 head rows
   (unet_init_from_map), production storage (MX-fp4 activations / gradients, SR): logits 0..15 bit for bit, and with a logit gradient that is
   zero on outputs 16..31 the trunk and head-row gradients agree (summation order only). */
#include "unet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static uint64_t rs = 777;
static float frand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 11) * 0x1.0p-53 * 2 - 1); }
int main(void) {
    setenv("UFSM_ACT_MX4", "1", 0); setenv("UFSM_XIN_PREC", "8", 0); setenv("UFSM_GRAD_MX8", "1", 0); setenv("UFSM_GRAD_MX4", "1", 0);
    setenv("UFSM_F4_WGRAD", "1", 0); setenv("UFSM_DOWN_NORM", "1", 0);
    if (nn_init(0)) { printf("SKIP: no GPU\n"); return 0; }
    nn_set_tf32(1); nn_set_f16(1); nn_set_prec_policy("all=fp4:fp4:fp4,enc0.c1=fp16"); nn_set_gn_stored(1); nn_set_sr(1);
    unet_cfg c32 = {0}; c32.nlev = 3; c32.widths[0] = 16; c32.widths[1] = 32; c32.widths[2] = 64; c32.cin = 1; c32.cout = 32; c32.G = 8; c32.down_norm = 1;
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
    float *gh = calloc(p3 * 32, 4); for (size_t i = 0; i < p3 * 16; i++) gh[i] = frand() * 1e-3f;
    float *g32 = nn_malloc(p3 * 32 * 4), *g16 = nn_malloc(p3 * 16 * 4); nn_h2d(g32, gh, p3 * 32 * 4); nn_h2d(g16, gh, p3 * 16 * 4);
    int bad = 0;
    nn_set_sr_step(100); const float *l32 = unet_forward(u32, x, xs, 1); const int h32 = unet_logits_h16(u32);
    const size_t eb = h32 ? 2 : 4; uint8_t *a = malloc(p3 * 16 * eb); nn_d2h(a, l32, p3 * 16 * eb);
    unet_zero_grad(u32); nn_set_sr_step(200); unet_backward(u32, g32);
    nn_set_sr_step(100); const float *l16 = unet_forward(u16, x, xs, 1);
    uint8_t *b = malloc(p3 * 16 * eb); nn_d2h(b, l16, p3 * 16 * eb);
    unet_zero_grad(u16); nn_set_sr_step(200); unet_backward(u16, g16);
    size_t neq = 0; for (size_t i = 0; i < p3 * 16 * eb; i++) neq += a[i] != b[i];
    printf("logits 0..15: %zu bytes differ (of %zu)%s\n", neq, p3 * 16 * eb, neq ? "  FAIL" : ""); bad += neq != 0;
    const size_t n32 = unet_nparams(u32), n16 = unet_nparams(u16); float *ga = malloc(n32 * 4), *gb = malloc(n16 * 4);
    unet_grad_d2h(u32, ga); unet_grad_d2h(u16, gb);
    size_t hw32, hb32, hw16, hb16; unet_head_offsets(u32, &hw32, &hb32); unet_head_offsets(u16, &hw16, &hb16);
    double d2 = 0, r2 = 0;
    for (size_t i = 0; i < hw32; i++) { const double d = (double)ga[i] - gb[i]; d2 += d * d; r2 += (double)gb[i] * gb[i]; }   /* trunk before the head */
    for (size_t i = 0; i < 16 * 16; i++) { const double d = (double)ga[hw32 + i] - gb[hw16 + i]; d2 += d * d; r2 += (double)gb[hw16 + i] * gb[hw16 + i]; }   /* head rows 0..15 */
    const size_t t32 = hb32 + 32, t16 = hb16 + 16;   /* params after the head (down norms) */
    for (size_t i = 0; t32 + i < n32 && t16 + i < n16; i++) { const double d = (double)ga[t32 + i] - gb[t16 + i]; d2 += d * d; r2 += (double)gb[t16 + i] * gb[t16 + i]; }
    double z2 = 0; for (size_t i = 16 * 16; i < 32 * 16; i++) z2 += (double)ga[hw32 + i] * ga[hw32 + i];
    const double rel = sqrt(d2 / (r2 > 0 ? r2 : 1));
    printf("gradients: rel diff %.3g, rows 16..31 |g| %.3g%s\n", rel, sqrt(z2), rel < 1e-3 && z2 == 0 ? "" : "  FAIL"); bad += !(rel < 1e-3 && z2 == 0);
    printf("%s\n", bad ? "test_head32: FAIL" : "test_head32: OK");
    return bad != 0;
}
