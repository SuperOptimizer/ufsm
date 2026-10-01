/* GroupNorm+SiLU backward throughput vs a device copy of the same bytes: bench_gn [P] [B] [C] [iters] (fp16 storage) */
#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
    int P = argc > 1 ? atoi(argv[1]) : 96, B = argc > 2 ? atoi(argv[2]) : 2, C = argc > 3 ? atoi(argv[3]) : 16, it = argc > 4 ? atoi(argv[4]) : 50, G = C < 8 ? C : 8;
    const char *gs = getenv("UFSM_GPU"); if (nn_init(gs ? atoi(gs) : 0)) return 1;
    nn_set_tf32(1); nn_set_act_bf16(1); nn_set_grad_bf16(1); nn_set_f16(1);
    shape5 s = {B, C, P, P, P};
    size_t n = (size_t)B * C * P * P * P, by = n * 2;
    void *x = nn_malloc(by), *gy = nn_malloc(by), *gx = nn_malloc(by);
    float *gamma = nn_malloc(C * 4), *beta = nn_malloc(C * 4), *mean = nn_malloc(B * G * 4), *rstd = nn_malloc(B * G * 4), *gg = nn_malloc(C * 4), *gb = nn_malloc(C * 4);
    float *scr = nn_malloc(nn_gn_scratch(s));
    unsigned short *h = malloc(by); for (size_t i = 0; i < n; i++) h[i] = 0x3c00 ^ (unsigned short)((i * 2654435761u) & 0x83ff);   /* fp16 values around +-1 */
    nn_h2d(x, h, by); nn_h2d(gy, h, by);
    const int xdt = getenv("UFSM_GN_X") ? atoi(getenv("UFSM_GN_X")) : 0;   /* 8 / 4: x stored MX-fp8 / MX-fp4 (gradients stay fp16) */
    if (xdt) {
        size_t bx = nn_mx_bytes(s, xdt); void *xm = nn_malloc(bx); nn_set_storage(xm, bx, xdt);
        nn_h16_to_mx(x, s, xm); x = xm;
    }
    float *hf = malloc(C * 4 > B * G * 4 ? C * 4 : B * G * 4);
    for (int i = 0; i < C; i++) hf[i] = 1.f; nn_h2d(gamma, hf, C * 4); for (int i = 0; i < C; i++) hf[i] = 0.f; nn_h2d(beta, hf, C * 4);
    for (int i = 0; i < B * G; i++) hf[i] = 0.f; nn_h2d(mean, hf, B * G * 4); for (int i = 0; i < B * G; i++) hf[i] = 1.f; nn_h2d(rstd, hf, B * G * 4);
    for (int i = 0; i < 3; i++) nn_gn_silu_bwd(x, s, G, gamma, beta, mean, rstd, gy, gx, gg, gb, scr);
    double tg = 1e9;   /* min over iterations (robust on a shared GPU) */
    for (int i = 0; i < it; i++) { nn_sync(); double t0 = now(); nn_gn_silu_bwd(x, s, G, gamma, beta, mean, rstd, gy, gx, gg, gb, scr); nn_sync(); double t = now() - t0; if (t < tg) tg = t; }
    double t0;
    t0 = now(); for (int i = 0; i < it; i++) nn_d2d(gx, gy, by); nn_sync(); double tc = (now() - t0) / it;
    double bytes = 5.0 * by;   /* stats reads x, gy; apply reads x, gy, writes gx */
    printf("x %s P %d B %d C %d: gn_silu_bwd %.3f ms (%.0f GB/s over %.0f MB), copy of %.0f MB %.3f ms (%.0f GB/s), ratio to the copy rate %.2fx\n",
           xdt == 8 ? "mx8" : xdt == 4 ? "mx4" : "fp16", P, B, C, tg * 1e3, bytes / tg / 1e9, bytes / 1e6, 2.0 * by / 1e6, tc * 1e3, 2.0 * by / tc / 1e9, (tg / bytes) / (tc / (2.0 * by)));
    const char *e = nn_check(); if (e) { printf("cuda: %s\n", e); return 1; }
    return 0;
}
