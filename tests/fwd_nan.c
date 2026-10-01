/* forward of a checkpoint on a batch dumped by the trainer's non-finite-forward diagnostic (runs/<out>/nan_step<N>.bin):
   reports non-finite logits per sample and the per-block GroupNorm statistics. Env UFSM_ACTF32=1 / UFSM_F16 as usual. */
#include "nn.h"
#include "unet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: fwd_nan <ckpt> <nan_step.bin> [gpu]\n"); return 2; }
    nn_init(argc > 3 ? atoi(argv[3]) : 0);
    if (getenv("UFSM_F16")) { nn_set_f16(1); }
    if (getenv("UFSM_FP32")) nn_set_tf32(0);
    if (getenv("UFSM_PREC_POLICY") && nn_set_prec_policy(getenv("UFSM_PREC_POLICY"))) return 2;
    FILE *f = fopen(argv[2], "rb"); if (!f) return 1;
    int hdr[4]; if (fread(hdr, 4, 4, f) != 4) return 1;
    int B = hdr[0], P = hdr[1], xf = hdr[2], NCH = hdr[3]; size_t p3 = (size_t)P * P * P, nx = (size_t)B * 4 * p3;
    void *hx = malloc(nx * (xf ? 2 : 4)); if (fread(hx, xf ? 2 : 4, nx, f) != nx) return 1;
    fclose(f);
    float *x32 = malloc(nx * 4);
    for (size_t k = 0; k < nx; k++) { if (!xf) x32[k] = ((float *)hx)[k]; else if (xf == 1) { _Float16 h; memcpy(&h, (uint16_t *)hx + k, 2); x32[k] = (float)h; } else { uint32_t u = (uint32_t)((uint16_t *)hx)[k] << 16; memcpy(&x32[k], &u, 4); } }
    unet_cfg cfg = {4, {16, 32, 64, 80}, 4, NCH, 8};
    { int st; if (unet_peek(argv[1], &cfg, &st)) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; } }   /* widths / down_norm from the checkpoint */
    unet *u = unet_create(&cfg);
    if (unet_load(u, argv[1]) < 0) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    float *xd = nn_malloc(nx * 4); nn_h2d(xd, x32, nx * 4);
    shape5 xs = {B, 4, P, P, P};
    const float *lg = unet_forward(u, xd, xs, 1);
    nn_sync();
    size_t nl = (size_t)B * NCH * p3; float *hl = malloc(nl * 4); nn_d2h(hl, lg, nl * 4);
    for (int b = 0; b < B; b++) for (int c = 0; c < NCH; c++) { size_t nf = 0; double mx = 0; for (size_t k = 0; k < p3; k++) { float v = hl[((size_t)b * NCH + c) * p3 + k]; if (!isfinite(v)) nf++; else if (fabs(v) > mx) mx = fabs(v); } printf("sample %d ch %d: %zu non-finite, max |logit| %.3g\n", b, c, nf, mx); }
    unet_debug_stats(u);
    unet_debug_acts(u);
    const char *e = nn_check(); if (e) printf("cuda: %s\n", e);
    return 0;
}
