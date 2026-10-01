/* Memory and time of one storage / precision mode on the real model (16,32,64,80), B x P^3 (env UFSM_B / UFSM_P, default
   2 x 96). Mode envs as in test_unet: UFSM_F16, UFSM_ACT_MX8, UFSM_GRAD_MX8, UFSM_RECOMPUTE, UFSM_PREC_POLICY.
   Prints the tracked activation + gradient bytes (unet_activation_bytes / unet_grad_bytes) and the device-level growth
   (cudaMemGetInfo before the first forward vs after a full train step: also counts kernel workspaces) for a training
   build, then the same for an inference-only build in a fresh model. */
#include "unet.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(void) {
    nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0);
    if (getenv("UFSM_PREC_POLICY") && nn_set_prec_policy(getenv("UFSM_PREC_POLICY"))) return 1;
    nn_set_tf32(1);
    const int P = getenv("UFSM_P") ? atoi(getenv("UFSM_P")) : 96, B = getenv("UFSM_B") ? atoi(getenv("UFSM_B")) : 2, it = 20;
    unet_cfg cfg = {4, {16, 32, 64, 80}, 4, 1, 8};
    shape5 xs = {B, 4, P, P, P};
    size_t nl = (size_t)B * P * P * P;
    float *x = nn_malloc(shape_numel(xs) * 4), *g = nn_malloc(nl * 4);
    nn_zero(x, shape_numel(xs) * 4); nn_zero(g, nl * 4);
    for (int mode = 0; mode < 2; mode++) {   /* 0 train, 1 inference */
        unet *u = unet_create(&cfg); unet_init(u, 1);
        nn_sync();
        size_t f0 = nn_mem_free();
        if (!mode) { unet_forward(u, x, xs, 1); unet_zero_grad(u); unet_backward(u, g); unet_adamw(u, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.f, 1); }
        else unet_forward(u, x, xs, 0);
        nn_sync();
        size_t f1 = nn_mem_free();
        double t0 = now();
        for (int i = 0; i < it; i++) {
            if (!mode) { unet_forward(u, x, xs, 1); unet_zero_grad(u); unet_backward(u, g); unet_adamw(u, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.f, i + 2); }
            else unet_forward(u, x, xs, 0);
        }
        nn_sync();
        double dt = (now() - t0) / it * 1e3;
        printf("%s: tracked %.3f GB (gradients %.3f GB), device growth %.3f GB, %.2f ms\n", mode ? "inference" : "train    ",
               unet_activation_bytes(u) / 1e9, unet_grad_bytes(u) / 1e9, (double)(f0 - f1) / 1e9, dt);
        unet_free(u);
        nn_sync();
    }
    const char *e = nn_check();
    if (e) { printf("cuda error: %s\n", e); return 1; }
    return 0;
}
