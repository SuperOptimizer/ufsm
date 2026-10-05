/* CUDA ops: fwd section of the former nn.cu */
#include "nn_common.cuh"

gnp_t to_gnp(const nn_gn_t *g) { gnp_t p = {}; if (g && g->G) { p.gamma = g->gamma; p.beta = g->beta; p.mean = g->mean; p.rstd = g->rstd; p.G = g->G; } return p; }
__global__ void prep_w16_k(const float *w, __half *wp, float *wsc, int Co, int Ci, int Cop, int Cip) {
    const int co = blockIdx.x;
    float am = 0.f;
    if (co < Co) for (int i = threadIdx.x; i < Ci * 27; i += blockDim.x) am = fmaxf(am, fabsf(w[(size_t)co * Ci * 27 + i]));
    __shared__ float red[32];
    for (int o = 16; o; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = am;
    __syncthreads();
    if (threadIdx.x < 32) { am = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f; for (int o = 16; o; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, o)); if (!threadIdx.x) red[0] = am; }
    __syncthreads();
    am = red[0];
    int e = 0;
    if (am > 0.f) frexpf(am / 16.f, &e);
    float inv = ldexpf(1.f, -e);
    if (threadIdx.x == 0) wsc[co] = ldexpf(1.f, e);
    for (int i = threadIdx.x; i < 27 * Cip; i += blockDim.x) {
        int t = i / Cip, ci = i % Cip;
        wp[((size_t)t * Cop + co) * Cip + ci] = __float2half_rn((co < Co && ci < Ci) ? w[((size_t)co * Ci + ci) * 27 + t] * inv : 0.f);
    }
}
