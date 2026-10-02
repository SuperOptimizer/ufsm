/* Reusing the last shared activation must preserve fresh and repeated backward gradients. */
#include "unet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static double relative(const float *a, const float *b, size_t n) {
    double delta = 0, norm = 0;
    for (size_t i = 0; i < n; i++) { double d = a[i] - b[i]; delta += d*d; norm += (double)b[i]*b[i]; }
    return sqrt(delta / (norm + 1e-30));
}

int main(void) {
    if (nn_init(0)) return 1;
    nn_set_f16(1); nn_set_grad_scale(1024); nn_set_sr(0); nn_set_gn_stored(1);
    setenv("UFSM_F4_WGRAD", "1", 1);
    int fails = 0;
    for (int levels = 2; levels <= 4; levels += 2) for (int mode = 0; mode < 3; mode++) for (int lean = 0; lean <= 2; lean += 2) for (int keep = 0; keep < 2; keep++) {
        unet_cfg cfg = {levels, {16, 32, 64, 80}, 4, 1, 8, 1};
        shape5 xs = {1, 4, 32, 32, 32}; size_t nx = shape_numel(xs), no = nx / 4;
        nn_set_prec(mode == 0 ? 4 : mode == 1 ? 2 : 3);
        if (nn_set_prec_policy(mode == 2 ? "all=fp4:fp4:fp4,enc0.c1=fp16" : "")) return 1;
        unet_set_act_mx8(mode == 1); unet_set_act_mx4(mode == 2); unet_set_grad_mx8(mode != 0);
        unet_set_input_prec(mode ? 8 : 0); unet_set_recompute(2); unet_set_chunk_up(2); unet_set_lean(lean);
        float *hx = malloc(nx * 4), *hg = malloc(no * 4), *x = nn_malloc(nx * 4), *gy = nn_malloc(no * 4);
        for (size_t i = 0; i < nx; i++) hx[i] = (float)((i * 2654435761u) % 1009) / 504.f - 1;
        for (size_t i = 0; i < no; i++) hg[i] = (float)((i * 97u) % 1009) / 504.f - 1;
        nn_h2d(x, hx, nx * 4); nn_h2d(gy, hg, no * 4);
        unet *u = unet_create(&cfg); unet_init(u, 19); size_t np = unet_nparams(u);
        float *g[4]; for (int i = 0; i < 4; i++) g[i] = malloc(np * 4);
        size_t bytes[2];
        for (int old = 0; old < 2; old++) {
            setenv("UFSM_RC_REDO_LAST", old ? "1" : "0", 1);
            setenv("UFSM_RC_KEEP_COARSE", old || !keep ? "0" : "1", 1);
            unet_forward(u, x, xs, 1);
            unet_zero_grad(u); nn_set_sr_step(123); unet_backward(u, gy); unet_grad_d2h(u, g[2 * old]);
            bytes[old] = unet_activation_bytes(u);   /* include lazily allocated input/logit-gradient buffers */
            /* All shared a1 values have changed; a second backward must reconstruct them. */
            unet_zero_grad(u); nn_set_sr_step(123); unet_backward(u, gy); unet_grad_d2h(u, g[2 * old + 1]);
        }
        double first = relative(g[0], g[2], np), repeat = relative(g[1], g[3], np), noise = relative(g[2], g[3], np);
        int ok = isfinite(first) && isfinite(repeat) && first <= 3 * noise + (mode == 0 ? 2e-3 : 1e-4)
                 && repeat <= 3 * noise + (mode == 0 ? 2e-3 : 1e-4)
                 && (keep ? bytes[0] >= bytes[1] : bytes[0] == bytes[1]);
        printf("live a1 L%d mode%d lean%d keep%d: fresh %.3g repeated %.3g reference %.3g memory %zu/%zu %s\n",
               levels, mode, lean, keep, first, repeat, noise, bytes[0], bytes[1], ok ? "ok" : "FAIL");
        if (!ok) fails++;
        const char *e = nn_check(); if (e) { printf("FAIL CUDA: %s\n", e); fails++; }
        for (int i = 0; i < 4; i++) free(g[i]);
        unet_free(u); nn_free(x); nn_free(gy); free(hx); free(hg);
    }
    unsetenv("UFSM_RC_REDO_LAST");
    unsetenv("UFSM_RC_KEEP_COARSE");
    printf("live final activation: %s\n", fails ? "FAIL" : "ok"); return fails != 0;
}
