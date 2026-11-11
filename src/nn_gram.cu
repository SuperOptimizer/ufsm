/* CUDA ops: the Gram matrix of K planes of 16-bit values (train --grad-cos: the head-input gradient's task dot products are
   sum_ij (W W^T)_ij S_ij over the logit-gradient Gram S, so one reduction per slab replaces a backward per task). */
#include "nn_common.cuh"

/* S[i K + j] += sum_n x_i[n] x_j[n] for j >= i (the lower triangle is not written); fp32 per thread, double per block */
template <typename T> __device__ inline float g_ld(const T *p, size_t i);
template <> __device__ inline float g_ld<__half>(const __half *p, size_t i) { return __half2float(p[i]); }
template <> __device__ inline float g_ld<__nv_bfloat16>(const __nv_bfloat16 *p, size_t i) { return __bfloat162float(p[i]); }
template <typename T> __global__ void gram_k(const T *x, int K, size_t N, size_t ld, double *S) {
    const int pi = blockIdx.x; int i = 0, r = pi;   /* pair index -> (i, j >= i) */
    while (r >= K - i) { r -= K - i; i++; }
    const int j = i + r;
    float a = 0.f;
    for (size_t n = (size_t)blockIdx.y * blockDim.x + threadIdx.x; n < N; n += (size_t)gridDim.y * blockDim.x) a += g_ld(x, (size_t)i * ld + n) * g_ld(x, (size_t)j * ld + n);
    __shared__ float sh[256]; sh[threadIdx.x] = a; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s]; __syncthreads(); }
    if (!threadIdx.x) atomicAdd(&S[(size_t)i * K + j], (double)sh[0]);
}
extern "C" void nn_gram16(const void *x, int bf16, int K, size_t N, size_t ld, double *S) {
    if (K <= 0 || !N) return;
    const unsigned np = (unsigned)(K * (K + 1) / 2), ny = (unsigned)(N / (256 * 64) + 1 < 64 ? N / (256 * 64) + 1 : 64);
    if (bf16) gram_k<<<dim3(np, ny), 256>>>((const __nv_bfloat16 *)x, K, N, ld, S);
    else gram_k<<<dim3(np, ny), 256>>>((const __half *)x, K, N, ld, S);
    KCHECK();
}
