/* train --ssl (src/ssl.h): the student's block mask on the CT channel, the teacher's int8 targets, the variance-hinge point indices.
   Launched on the default stream of the current device (the nn layer's compute stream). */
#include "ssl.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

static inline unsigned nb_ssl(long long n, int t) { long long b = (n + t - 1) / t; return (unsigned)(b > 1048576LL ? 1048576LL : b < 1 ? 1 : b); }
#define SSL_CK() do { cudaError_t e_ = cudaGetLastError(); if (e_ != cudaSuccess) { fprintf(stderr, "nn_ssl: %s\n", cudaGetErrorString(e_)); abort(); } } while (0)

/* one thread per 4 voxels along x (P % 4 == 0 and bv % 4 == 0: the 4 share a block) */
__global__ void k_ssl_mask(void *x, int es, int Pz, int P, int bv, float ratio, uint64_t seed) {
    const long long n4 = (long long)Pz * P * (P / 4);
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n4; i += (long long)gridDim.x * blockDim.x) {
        const long long v = 4 * i; const int xx = (int)(v % P), yy = (int)((v / P) % P), zz = (int)(v / ((long long)P * P));
        if (!ssl_blk_masked(seed, zz / bv, yy / bv, xx / bv, ratio)) continue;
        if (es == 2) *(uint2 *)((uint16_t *)x + v) = make_uint2(0u, 0u);
        else *(uint4 *)((float *)x + v) = make_uint4(0u, 0u, 0u, 0u);
    }
}
extern "C" void nn_ssl_mask(void *x, int es, int Pz, int P, int bv, float ratio, uint64_t seed) {
    if ((es != 2 && es != 4) || P % 4 || bv % 4) { fprintf(stderr, "nn_ssl_mask: 2- or 4-byte voxels, P and the block a multiple of 4\n"); abort(); }
    const long long n4 = (long long)Pz * P * (P / 4);
    k_ssl_mask<<<nb_ssl(n4, 256), 256>>>(x, es, Pz, P, bv, ratio, seed);
    SSL_CK();
}

__global__ void k_ssl_mean(const float *y, int C, size_t S, double *mu) {   /* blockIdx.y = channel; per-block partial sums, double atomics */
    __shared__ double sh[256];
    const int c = blockIdx.y; double s = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < S; i += (size_t)gridDim.x * blockDim.x) s += y[(size_t)c * S + i];
    sh[threadIdx.x] = s; __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) { if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o]; __syncthreads(); }
    if (!threadIdx.x) atomicAdd(&mu[c], sh[0]);
}
__global__ void k_ssl_targets(const float *y, int C, int cd, int ch, int cw, int cv, int bv, float ratio, uint64_t seed, const double *mu, int8_t *t) {
    const size_t S = (size_t)cd * ch * cw;
    for (size_t s = blockIdx.x * (size_t)blockDim.x + threadIdx.x; s < S; s += (size_t)gridDim.x * blockDim.x) {
        const int x = (int)(s % cw), yy = (int)((s / cw) % ch), z = (int)(s / ((size_t)cw * ch));
        const int on = ssl_blk_masked(seed, z * cv / bv, yy * cv / bv, x * cv / bv, ratio);
        double n2 = 0;
        if (on) for (int c = 0; c < C; c++) { const double v = y[(size_t)c * S + s] - mu[c] / (double)S; n2 += v * v; }
        const double n = sqrt(n2);
        if (!on || !(n > 1e-6)) { for (int c = 0; c < C; c++) t[(size_t)c * S + s] = 0; continue; }
        for (int c = 0; c < C; c++) {
            const double v = (y[(size_t)c * S + s] - mu[c] / (double)S) / n;
            t[(size_t)c * S + s] = (int8_t)lrint(fmin(127.0, fmax(-127.0, 127.0 * v)));
        }
    }
}
extern "C" void nn_ssl_targets(const float *y, int C, int cd, int ch, int cw, int cv, int bv, float ratio, uint64_t seed, double *mu, int8_t *t) {
    if (bv % cv) { fprintf(stderr, "nn_ssl_targets: the mask block (%d voxels) is not a multiple of the cell (%d)\n", bv, cv); abort(); }
    const size_t S = (size_t)cd * ch * cw;
    cudaMemsetAsync(mu, 0, (size_t)C * sizeof(double));
    k_ssl_mean<<<dim3(nb_ssl((long long)S, 256) > 256 ? 256 : nb_ssl((long long)S, 256), C), 256>>>(y, C, S, mu);
    SSL_CK();
    k_ssl_targets<<<nb_ssl((long long)S, 256), 256>>>(y, C, cd, ch, cw, cv, bv, ratio, seed, mu, t);
    SSL_CK();
}
__global__ void k_ssl_idx(int *idx, int N, size_t S, size_t off, size_t stride) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N) idx[k] = (int)((off + (unsigned __int128)k * stride) % S);
}
extern "C" void nn_ssl_idx(int *idx, int N, size_t S, size_t off, size_t stride) {
    if (N > 0) k_ssl_idx<<<nb_ssl(N, 256), 256>>>(idx, N, S, off % S, stride % S);
    SSL_CK();
}

__device__ __forceinline__ float ssl_sig(float z) { return 1.f / (1.f + __expf(-z)); }
__global__ void k_ssl_ink_t(const __half *lg, int ink_ch, int rv_ch, size_t n, size_t off, int cd, int ch, int cw, int cv, int bv, float ratio, uint64_t seed,
                            float conf, uint8_t *t, unsigned long long *cnt) {
    __shared__ unsigned long long sh[2];
    if (threadIdx.x < 2) sh[threadIdx.x] = 0; __syncthreads();
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        const size_t s = off + i; const int x = (int)(s % cw), y = (int)((s / cw) % ch), z = (int)(s / ((size_t)cw * ch));
        const float p = ssl_sig(__half2float(lg[(size_t)ink_ch * n + i]));
        int on = !ssl_blk_masked(seed, z * cv / bv, y * cv / bv, x * cv / bv, ratio) && fabsf(p - 0.5f) >= conf;
        if (on && rv_ch >= 0) on = fmaxf(ssl_sig(__half2float(lg[(size_t)rv_ch * n + i])), ssl_sig(__half2float(lg[(size_t)(rv_ch + 1) * n + i]))) >= 0.5f;
        t[s] = on ? (uint8_t)lrintf(254.f * p) : 255;
        if (on) { atomicAdd(&sh[0], 1ull); if (p >= 0.5f) atomicAdd(&sh[1], 1ull); }
    }
    __syncthreads();
    if (threadIdx.x < 2 && sh[threadIdx.x]) atomicAdd(&cnt[threadIdx.x], sh[threadIdx.x]);
}
extern "C" void nn_ssl_ink_targets(const void *lg, int ink_ch, int rv_ch, size_t n, size_t off, int cd, int ch, int cw, int cv, int bv, float ratio, uint64_t seed,
                                   float conf, uint8_t *t, unsigned long long *cnt) {
    if (bv % cv) { fprintf(stderr, "nn_ssl_ink_targets: the mask block (%d voxels) is not a multiple of the cell (%d)\n", bv, cv); abort(); }
    if (n) k_ssl_ink_t<<<nb_ssl((long long)n, 256), 256>>>((const __half *)lg, ink_ch, rv_ch, n, off, cd, ch, cw, cv, bv, ratio, seed, conf, t, cnt);
    SSL_CK();
}
__global__ void k_ssl_ink_g(const __half *lg, int ink_ch, size_t n, size_t off, const uint8_t *t, float scale, __half *g, double *sums) {
    __shared__ float sh[2];
    if (threadIdx.x < 2) sh[threadIdx.x] = 0.f; __syncthreads();
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        const uint8_t q = t[off + i]; float gv = 0.f;
        if (q != 255) {
            const float z = __half2float(lg[(size_t)ink_ch * n + i]), y = q / 254.f, p = ssl_sig(z);
            gv = scale * (p - y);
            atomicAdd(&sh[0], fmaxf(z, 0.f) - z * y + log1pf(__expf(-fabsf(z)))); atomicAdd(&sh[1], 1.f);
        }
        g[(size_t)ink_ch * n + i] = __float2half(gv);
    }
    __syncthreads();
    if (threadIdx.x < 2 && sh[threadIdx.x] != 0.f) atomicAdd(&sums[threadIdx.x], (double)sh[threadIdx.x]);
}
extern "C" void nn_ssl_ink_grad(const void *lg, int ink_ch, size_t n, size_t off, const uint8_t *t, float scale, void *g, double *sums) {
    if (n) k_ssl_ink_g<<<nb_ssl((long long)n, 256), 256>>>((const __half *)lg, ink_ch, n, off, t, scale, (__half *)g, sums);
    SSL_CK();
}
