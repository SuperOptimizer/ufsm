/* CUDA ops: per-level scale FiLM on GroupNorm affines (src/unet.c cfg.film). A FiLM'd norm of C channels with K codes c_k (the window's
   log2 voxel sizes) has gamma_eff = gamma (1 + sum_k c_k A_kc), beta_eff = beta + sum_k c_k D_kc, A / D at fo (K x C each, A first).
   nn_film_fwd writes the effective pair into the forward weights fw; nn_film_bwd turns the gradient of the effective pair (in g at the
   norm's offsets) into the gradients of gamma, A and D (beta's is the effective one's). */
#include "nn_common.cuh"

struct film_codes { float c[4]; };
__global__ void film_fwd_k(const float *p, float *fw, size_t gam, size_t bet, size_t fo, int C, int K, film_codes cc) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= C) return;
    const float *A = p + fo, *D = p + fo + (size_t)K * C;
    float s = 0.f, t = 0.f;
    for (int k = 0; k < K; k++) { s += cc.c[k] * A[(size_t)k * C + c]; t += cc.c[k] * D[(size_t)k * C + c]; }
    fw[gam + c] = p[gam + c] * (1.f + s); fw[bet + c] = p[bet + c] + t;
}
__global__ void film_bwd_k(const float *p, float *g, size_t gam, size_t bet, size_t fo, int C, int K, film_codes cc) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= C) return;
    const float *A = p + fo; float *gA = g + fo, *gD = g + fo + (size_t)K * C;
    const float ge = g[gam + c], be = g[bet + c], ga = p[gam + c];
    float s = 0.f;
    for (int k = 0; k < K; k++) { s += cc.c[k] * A[(size_t)k * C + c]; gA[(size_t)k * C + c] += cc.c[k] * ge * ga; gD[(size_t)k * C + c] += cc.c[k] * be; }
    g[gam + c] = ge * (1.f + s);
}
static film_codes codes_of(const float *c, int K) { film_codes cc = {{0.f, 0.f, 0.f, 0.f}}; for (int k = 0; k < K && k < 4; k++) cc.c[k] = c[k]; return cc; }
extern "C" void nn_film_fwd(const float *p, float *fw, size_t gam, size_t bet, size_t fo, int C, int K, const float *codes) {
    film_fwd_k<<<(C + 127) / 128, 128>>>(p, fw, gam, bet, fo, C, K, codes_of(codes, K)); KCHECK();
}
extern "C" void nn_film_bwd(const float *p, float *g, size_t gam, size_t bet, size_t fo, int C, int K, const float *codes) {
    film_bwd_k<<<(C + 127) / 128, 128>>>(p, g, gam, bet, fo, C, K, codes_of(codes, K)); KCHECK();
}
