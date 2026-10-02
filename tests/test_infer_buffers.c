/* Consumed encoder skips can hold decoder outputs; repeated forwards must match separate buffers. */
#include "unet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    if (nn_init(0)) return 1;
    int fails = 0; nn_set_sr(0); nn_set_f16(1);
    for (int levels = 2; levels <= 4; levels += 2) for (int rc = 0; rc <= 2; rc++) for (int mode = 0; mode < 3; mode++) {
        unet_cfg cfg = {levels, {16, 32, 64, 80}, 4, 2, 8, 1};
        shape5 xs = {1, 4, 32, 32, 32}; size_t n = shape_numel(xs), no = n / 2;
        nn_set_prec(mode == 0 ? 4 : mode == 1 ? 2 : 3); nn_set_prec_policy("");
        unet_set_recompute(rc); unet_set_act_mx8(mode == 1); unet_set_act_mx4(mode == 2);
        float *h = malloc(n * 4), *out[3] = {malloc(no * 4), malloc(no * 4), malloc(no * 4)}, *x = nn_malloc(n * 4);
        unet *u[2]; size_t bytes[2];
        for (int old = 0; old < 2; old++) { setenv("UFSM_INFER_SCRATCH_OLD", old ? "1" : "0", 1); u[old] = unet_create(&cfg); unet_init(u[old], 19); }
        for (int repeat = 0; repeat < 3; repeat++) {
            for (size_t i = 0; i < n; i++) h[i] = (float)((i * 2654435761u + repeat * 97u) % 1009) / 504.f - 1;
            nn_h2d(x, h, n * 4);
            for (int old = 0; old < 2; old++) {
                setenv("UFSM_INFER_SCRATCH_OLD", old ? "1" : "0", 1);
                const float *p = unet_forward(u[old], x, xs, 0); nn_d2h(out[old], p, no * 4); bytes[old] = unet_activation_bytes(u[old]);
            }
            const float *p = unet_forward(u[1], x, xs, 0); nn_d2h(out[2], p, no * 4);
            double delta = 0, noise = 0, norm = 0;
            for (size_t i = 0; i < no; i++) { double a = out[0][i] - out[1][i], b = out[2][i] - out[1][i]; delta += a*a; noise += b*b; norm += (double)out[1][i]*out[1][i]; }
            delta = sqrt(delta / (norm + 1e-30)); noise = sqrt(noise / (norm + 1e-30));
            printf("  compare repeat%d rel %.3g, old repeated %.3g\n", repeat, delta, noise);
            /* FP16 group accumulators amplify last-bit GN atomic-order differences into fp16
               rounding changes: the unchanged path itself varies by up to 9e-4 here. */
            if (delta > 3 * noise + (mode == 0 ? 1e-3 : 1e-5) || !isfinite(delta)) { printf("FAIL logits L%d rc%d mode%d repeat%d\n", levels, rc, mode, repeat); fails++; }
        }
        if (rc && bytes[0] >= bytes[1]) { printf("FAIL memory L%d rc%d mode%d\n", levels, rc, mode); fails++; }
        printf("infer buffers L%d rc%d mode%d: %zu -> %zu bytes\n", levels, rc, mode, bytes[1], bytes[0]);
        const char *e = nn_check(); if (e) { printf("FAIL CUDA: %s\n", e); fails++; }
        for (int old = 0; old < 2; old++) { unet_free(u[old]); free(out[old]); }
        nn_free(x); free(h); free(out[2]);
    }
    unsetenv("UFSM_INFER_SCRATCH_OLD"); printf("inference buffer reuse: %s\n", fails ? "FAIL" : "ok"); return fails != 0;
}
