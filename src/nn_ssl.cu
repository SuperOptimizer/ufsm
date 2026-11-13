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
constexpr int SSLH_SLOTS = 32;
/* --ssl-heads targets: one thread per cell, every head (ssl.h) */
__global__ void k_sslh_t(const __half *lg, ssl_heads hs, int rv_ch, size_t n, size_t off, int cd, int ch, int cw, int cv, int bv, float ratio, uint64_t seed,
                         size_t S, uint8_t *t, unsigned long long *cnt) {
    __shared__ unsigned long long sh[SSLH_SLOTS];
    if (threadIdx.x < SSLH_SLOTS) sh[threadIdx.x] = 0; __syncthreads();
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        const size_t s = off + i; const int x = (int)(s % cw), y = (int)((s / cw) % ch), z = (int)(s / ((size_t)cw * ch));
        const int seen = !ssl_blk_masked(seed, z * cv / bv, y * cv / bv, x * cv / bv, ratio);
        const int pap = rv_ch >= 0 && fmaxf(ssl_sig(__half2float(lg[(size_t)rv_ch * n + i])), ssl_sig(__half2float(lg[(size_t)(rv_ch + 1) * n + i]))) >= 0.5f;
        for (int h = 0; h < hs.nh; h++) {
            const ssl_head e = hs.h[h]; float p[6]; int on[6];
            for (int c = 0; c < e.nch; c++) p[c] = ssl_sig(__half2float(lg[(size_t)(e.ch0 + c) * n + i]));
            const int ok = seen && (!e.pap || pap);
            if (e.mode == SSLH_NORM) {
                float f = 0.f; for (int c = 0; c < 3; c++) f += p[c] * p[c]; for (int c = 3; c < 6; c++) f += 2.f * (p[c] - 0.5f) * (p[c] - 0.5f);
                for (int c = 0; c < 6; c++) on[c] = ok && sqrtf(f) >= e.conf;
            } else if (e.mode == SSLH_PHASE) {
                const float u = 2.f * p[0] - 1.f, v = 2.f * p[1] - 1.f; on[0] = on[1] = ok && sqrtf(u * u + v * v) >= e.conf;
            } else for (int c = 0; c < e.nch; c++) on[c] = ok && fabsf(p[c] - 0.5f) >= e.conf;
            for (int c = 0; c < e.nch; c++) { t[(size_t)(e.slot0 + c) * S + s] = on[c] ? (uint8_t)lrintf(254.f * p[c]) : 255; if (on[c]) atomicAdd(&sh[e.slot0 + c], 1ull); }
        }
    }
    __syncthreads();
    if (threadIdx.x < hs.nslot && sh[threadIdx.x]) atomicAdd(&cnt[threadIdx.x], sh[threadIdx.x]);
}
static void sslh_check(const ssl_heads *hs) {
    int ns = 0; for (int h = 0; h < hs->nh; h++) { const ssl_head *e = &hs->h[h];
        if (e->nch < 1 || e->nch > 6 || (e->mode == SSLH_NORM && e->nch != 6) || (e->mode == SSLH_PHASE && e->nch != 2) || e->slot0 != ns) { fprintf(stderr, "nn_sslh: head %d: a bad spec\n", h); abort(); }
        ns += e->nch; }
    if (hs->nh > SSLH_MAX || ns != hs->nslot || ns > SSLH_SLOTS) { fprintf(stderr, "nn_sslh: %d heads / %d slots (at most %d / %d)\n", hs->nh, hs->nslot, SSLH_MAX, SSLH_SLOTS); abort(); }
}
extern "C" void nn_sslh_targets(const void *lg, const ssl_heads *hs, int rv_ch, size_t n, size_t off, int cd, int ch, int cw, int cv, int bv, float ratio,
                                uint64_t seed, uint8_t *t, unsigned long long *cnt) {
    if (bv % cv) { fprintf(stderr, "nn_sslh_targets: the mask block (%d voxels) is not a multiple of the cell (%d)\n", bv, cv); abort(); }
    sslh_check(hs);
    for (int h = 0; h < hs->nh; h++) if (hs->h[h].pap && rv_ch < 0) { fprintf(stderr, "nn_sslh_targets: head %d is gated by papyrus, no rv pair\n", h); abort(); }
    if (n) k_sslh_t<<<nb_ssl((long long)n, 256), 256>>>((const __half *)lg, *hs, rv_ch, n, off, cd, ch, cw, cv, bv, ratio, seed, (size_t)cd * ch * cw, t, cnt);
    SSL_CK();
}
__global__ void k_sslh_g(const __half *lg, ssl_heads hs, size_t n, size_t off, size_t S, const uint8_t *t, __half *g, double *sums) {
    __shared__ float sh[2 * SSLH_MAX];
    if (threadIdx.x < 2 * SSLH_MAX) sh[threadIdx.x] = 0.f; __syncthreads();
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) for (int h = 0; h < hs.nh; h++) {
        const ssl_head e = hs.h[h]; float l = 0.f, k = 0.f;
        for (int c = 0; c < e.nch; c++) {
            const uint8_t q = t[(size_t)(e.slot0 + c) * S + off + i]; float gv = 0.f;
            if (q != 255) {
                const float z = __half2float(lg[(size_t)(e.ch0 + c) * n + i]), y = q / 254.f;
                gv = hs.scale[h] * (ssl_sig(z) - y); l += fmaxf(z, 0.f) - z * y + log1pf(__expf(-fabsf(z))); k += 1.f;
            }
            g[(size_t)(e.ch0 + c) * n + i] = __float2half(gv);
        }
        if (k > 0.f) { atomicAdd(&sh[2 * h], l); atomicAdd(&sh[2 * h + 1], k); }
    }
    __syncthreads();
    if (threadIdx.x < 2 * hs.nh && sh[threadIdx.x] != 0.f) atomicAdd(&sums[threadIdx.x], (double)sh[threadIdx.x]);
}
extern "C" void nn_sslh_grad(const void *lg, const ssl_heads *hs, size_t n, size_t off, size_t S, const uint8_t *t, void *g, double *sums) {
    sslh_check(hs);
    if (n) k_sslh_g<<<nb_ssl((long long)n, 256), 256>>>((const __half *)lg, *hs, n, off, S, t, (__half *)g, sums);
    SSL_CK();
}
