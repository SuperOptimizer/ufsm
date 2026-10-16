/* Level-0 normalisation alternatives (unet cfg.l0_norm): GroupNorm statistics held constant -- none (mean 0, rstd 1) or
   BatchNorm-style running statistics (ema) -- and the GroupNorm backwards without the statistics' gradient terms. */
#include "nn_common.cuh"

int g_gn_const;   /* nn_gn_bwd / nn_gn_silu_bwd / nn_gn_silu_bwd_head: the group sums (mean and variance terms) scaled by 0 */
extern "C" void nn_set_gn_const(int on) { g_gn_const = on; }
extern "C" int nn_get_gn_const(void) { return g_gn_const; }

__global__ void gn_const_fill_k(float *mean, float *rstd, const float *run, int N, int G, float eps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * G) return;
    const int g = i % G;
    mean[i] = run ? run[g] : 0.f;
    rstd[i] = run ? 1.f / sqrtf(run[G + g] + eps) : 1.f;
}
extern "C" void nn_gn_const_fill(float *mean, float *rstd, const float *run, int N, int G, float eps) {
    gn_const_fill_k<<<nblk((size_t)N * G, 128), 128>>>(mean, rstd, run, N, G, eps);
    KCHECK();
}
/* run [mean G | var G] <- (1 - mom) run + mom (the batch's mean and variance per group: the samples' statistics pooled) */
__global__ void gn_ema_k(float *run, const float *mean, const float *rstd, int N, int G, float mom, float eps) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= G) return;
    double sm = 0, smm = 0, sv = 0;
    for (int n = 0; n < N; n++) { const double m = mean[n * G + g], r = rstd[n * G + g]; sm += m; smm += m * m; sv += 1.0 / (r * r) - eps; }
    const double m = sm / N, v = fmax(sv / N + smm / N - m * m, 0.0);
    run[g] = mom >= 1.f ? (float)m : (float)((1.0 - mom) * run[g] + mom * m);
    run[G + g] = mom >= 1.f ? (float)v : (float)((1.0 - mom) * run[G + g] + mom * v);
}
extern "C" void nn_gn_ema_update(float *run, const float *mean, const float *rstd, int N, int G, float mom, float eps) {
    gn_ema_k<<<nblk((size_t)G, 32), 32>>>(run, mean, rstd, N, G, mom, eps);
    KCHECK();
}
