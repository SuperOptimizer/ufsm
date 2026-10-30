/* wtrunk kernels (src/wtrunk.h): the in-slab network input of the frozen wind96 U-Net and its outputs to channels-last bf16 */
#include "wtrunk.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>

#define WTK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "wtrunk: %s (%s:%d)\n", cudaGetErrorString(e_), __FILE__, __LINE__); abort(); } } while (0)

/* per-slab integer sums of the CT and its square (exact, as predict's nn_pred_stats) */
__global__ void wt_stats_k(const uint8_t *im, size_t V, unsigned long long *st) {
    const int s = blockIdx.y; const uint8_t *p = im + (size_t)s * V;
    unsigned long long a = 0, q = 0;
    for (size_t i = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) * 16; i < V; i += (size_t)gridDim.x * blockDim.x * 16) {
        const uint4 v = *(const uint4 *)(p + i);   /* (V is a multiple of 16) */
        const uint32_t w[4] = {v.x, v.y, v.z, v.w};
        for (int k = 0; k < 4; k++) for (int b = 0; b < 4; b++) { const unsigned c = w[k] >> (8 * b) & 255u; a += c; q += c * c; }
    }
    for (int o = 16; o > 0; o >>= 1) { a += __shfl_down_sync(0xffffffffu, a, o); q += __shfl_down_sync(0xffffffffu, q, o); }
    if ((threadIdx.x & 31) == 0) { atomicAdd(&st[2 * s], a); atomicAdd(&st[2 * s + 1], q); }
}
template <typename T> __device__ __forceinline__ T wt_cv(float v);
template <> __device__ __forceinline__ __half wt_cv<__half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ __nv_bfloat16 wt_cv<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }
/* one thread per network voxel (z, y, x) = (j, i, k): CT z-score (predict: mean / sd over the whole window incl. zeros, sd + 1e-3),
   0, then the horizontal radial unit vector's a and d components (predict's dy / dx channels, radial sign 1) */
template <typename HT> __global__ void wt_input_k(const uint8_t *im, const float *prm, int T, int L, const unsigned long long *st, HT *x) {
    const size_t V = (size_t)T * T * L, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const int s = blockIdx.y;
    if (i >= V) return;
    const int k = (int)(i % L), yi = (int)(i / L % T), zj = (int)(i / ((size_t)L * T));
    const double n = (double)V, m = (double)st[2 * s] / n, var = (double)st[2 * s + 1] / n - m * m;
    const float mean = (float)m, isd = (float)(1.0 / (sqrt(var > 0 ? var : 0) + 1e-3));
    const float *P = prm + 16 * s;
    float r[3];
    for (int e = 0; e < 3; e++) r[e] = P[e] + yi * P[3 + e] + zj * P[6 + e] + k * P[9 + e];   /* p - c = (o - c) + i a + j b + k d */
    const float ru = r[0] * P[12] + r[1] * P[13] + r[2] * P[14];
    for (int e = 0; e < 3; e++) r[e] -= ru * P[12 + e];
    const float rn = sqrtf(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]), inv = 1.f / (rn + 1e-6f);
    const float ra = (r[0] * P[3] + r[1] * P[4] + r[2] * P[5]) * inv, rd = (r[0] * P[9] + r[1] * P[10] + r[2] * P[11]) * inv;
    const float ct = (float)im[(size_t)s * V + ((size_t)yi * T + zj) * L + k];
    HT *xs = x + (size_t)s * 4 * V + i;
    xs[0] = wt_cv<HT>((ct - mean) * isd); xs[V] = wt_cv<HT>(0.f); xs[2 * V] = wt_cv<HT>(ra); xs[3 * V] = wt_cv<HT>(rd);
}
extern "C" void wt_input(const uint8_t *im, const float *prm, int n, int T, int L, unsigned long long *st, void *x, int h16) {
    const size_t V = (size_t)T * T * L;
    if (V % 16) { fprintf(stderr, "wt_input: T T L must be a multiple of 16\n"); abort(); }
    WTK(cudaMemsetAsync(st, 0, (size_t)n * 2 * sizeof *st));
    wt_stats_k<<<dim3(256, n), 256>>>(im, V, st);
    const dim3 g((unsigned)((V + 255) / 256), n);
    if (h16 == 2) wt_input_k<__nv_bfloat16><<<g, 256>>>(im, prm, T, L, st, (__nv_bfloat16 *)x);
    else wt_input_k<__half><<<g, 256>>>(im, prm, T, L, st, (__half *)x);
    WTK(cudaGetLastError());
}
/* logits / features [n][C][j][i][k] -> bf16 [n][i][j][k][C]; channels < cout through the sigmoid. One thread per voxel */
template <typename LT> __global__ void wt_output_k(const LT *lg, int C, int cout, int T, int L, __nv_bfloat16 *out) {
    const size_t V = (size_t)T * T * L, v = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const int s = blockIdx.y;
    if (v >= V) return;
    const int k = (int)(v % L), j = (int)(v / L % T), i = (int)(v / ((size_t)L * T));   /* v: output order (i, j, k) */
    const size_t src = ((size_t)j * T + i) * L + k;
    const LT *p = lg + (size_t)s * C * V + src;
    __nv_bfloat16 *o = out + ((size_t)s * V + v) * C;
    for (int c = 0; c < C; c++) {
        const float a = (float)p[(size_t)c * V];
        o[c] = __float2bfloat16_rn(c < cout ? 1.f / (1.f + __expf(-a)) : a);
    }
}
extern "C" void wt_output(const void *lg, int lh16, int n, int C, int cout, int T, int L, void *out) {
    const size_t V = (size_t)T * T * L;
    const dim3 g((unsigned)((V + 127) / 128), n);
    if (lh16) wt_output_k<__half><<<g, 128>>>((const __half *)lg, C, cout, T, L, (__nv_bfloat16 *)out);
    else wt_output_k<float><<<g, 128>>>((const float *)lg, C, cout, T, L, (__nv_bfloat16 *)out);
    WTK(cudaGetLastError());
}
