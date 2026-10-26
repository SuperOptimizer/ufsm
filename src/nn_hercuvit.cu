/* hercuvit (src/hercuvit.h): the dense parts of the HercuViT mask readout (pooled attention-mask logits, per-voxel owner /
   membership) and the Gaussian window blend of ufsm hercuvit. On xnet_stream(); U1 fp16 channels-last with 32 channels. */
#include "hercuvit.h"
#include "xnet.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define ST ((cudaStream_t)xnet_stream())
enum { HV_C = 32, HV_QMAX = 64 };
static inline unsigned nb_of(long long n, int t) { long long b = (n + t - 1) / t; return (unsigned)(b > 2147483647LL ? 2147483647LL : b); }

__device__ __forceinline__ void load_u(const __half *U, size_t v, float *u) {
    const uint4 *p = (const uint4 *)(U + v * HV_C);
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const uint4 a = p[k]; const __half2 *h = (const __half2 *)&a;
#pragma unroll
        for (int j = 0; j < 4; j++) { const float2 f = __half22float2(h[j]); u[k * 8 + 2 * j] = f.x; u[k * 8 + 2 * j + 1] = f.y; }
    }
}
__device__ __forceinline__ void atomic_maxf(float *a, float v) {
    if (v >= 0.f) atomicMax((int *)a, __float_as_int(v)); else atomicMin((unsigned *)a, __float_as_uint(v));
}

/* one block of 256 voxels inside one cell: per query the block max, then one atomic per query */
__global__ void k_pool_max(const __half *__restrict__ U, int S, const float *__restrict__ me, int Q, int side, float *out) {
    __shared__ float sme[HV_QMAX * HV_C];
    __shared__ float smax[8][HV_QMAX];
    for (int i = threadIdx.x; i < Q * HV_C; i += blockDim.x) sme[i] = me[i];
    __syncthreads();
    const int k = S / side; const long long kk = (long long)k * k * k, per = kk / 256;
    const long long cell = blockIdx.x / per, chunk = blockIdx.x % per;
    const int cz = (int)(cell / (side * side)), cy = (int)(cell / side % side), cx = (int)(cell % side);
    const long long l = chunk * 256 + threadIdx.x;
    const int lz = (int)(l / (k * k)), ly = (int)(l / k % k), lx = (int)(l % k);
    const size_t v = ((size_t)(cz * k + lz) * S + cy * k + ly) * S + cx * k + lx;
    float u[HV_C]; load_u(U, v, u);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int q = 0; q < Q; q++) {
        const float *m = sme + q * HV_C; float s = 0.f;
#pragma unroll
        for (int c = 0; c < HV_C; c++) s = fmaf(m[c], u[c], s);
        for (int o = 16; o; o >>= 1) s = fmaxf(s, __shfl_xor_sync(0xffffffffu, s, o));
        if (!lane) smax[warp][q] = s;
    }
    __syncthreads();
    if (threadIdx.x < Q) {
        float m = smax[0][threadIdx.x];
        for (int w = 1; w < 8; w++) m = fmaxf(m, smax[w][threadIdx.x]);
        atomic_maxf(&out[(size_t)threadIdx.x * side * side * side + cell], m);
    }
}
__global__ void k_fill(float *p, size_t n, float v) { for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = v; }
extern "C" void hv_fill(float *p, size_t n, float v) { k_fill<<<nb_of((long long)n, 256) > 65535 * 4 ? 65535 * 4 : nb_of((long long)n, 256), 256, 0, ST>>>(p, n, v); }
extern "C" void hv_pool_max(const void *U, int S, const float *me_dev, int Q, int side, float *out_dev) {
    const long long k = S / side, kk = k * k * k;
    if (Q > HV_QMAX || S % side || kk % 256) { fprintf(stderr, "hv_pool_max: unsupported shape (S %d, side %d, Q %d)\n", S, side, Q); return; }
    hv_fill(out_dev, (size_t)Q * side * side * side, -INFINITY);
    const long long nb = (long long)side * side * side * (kk / 256);
    k_pool_max<<<(unsigned)nb, 256, 0, ST>>>((const __half *)U, S, me_dev, Q, side, out_dev);
}

__global__ void k_final(const __half *__restrict__ U, int S, const float *__restrict__ me, int Q, const float *__restrict__ margin, float lthr,
                        int z0, int y0, int x0, int nz, int ny, int nx, int oz, int oy, int ox, int dh, int dw,
                        uint32_t *ids, uint32_t base, uint8_t *score, uint8_t *owner, float *memb_f) {
    __shared__ float sme[HV_QMAX * HV_C];
    for (int i = threadIdx.x; i < Q * HV_C; i += blockDim.x) sme[i] = me[i];
    __syncthreads();
    const long long n = (long long)nz * ny * nx;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const int z = z0 + (int)(i / ((long long)ny * nx)), y = y0 + (int)(i / nx % ny), x = x0 + (int)(i % nx);
        const size_t v = ((size_t)z * S + y) * S + x;
        float u[HV_C]; load_u(U, v, u);
        float best = -INFINITY, sum = 0.f; int bq = 0;
        for (int q = 0; q < Q; q++) {   /* online logsumexp */
            const float *m = sme + q * HV_C; float s = 0.f;
#pragma unroll
            for (int c = 0; c < HV_C; c++) s = fmaf(m[c], u[c], s);
            if (s > best) { sum = sum * expf(best - s) + 1.f; best = s; bq = q; } else sum += expf(s - best);
        }
        const float memb = 1.f / sum;
        const size_t d = ((size_t)(z - oz) * dh + (y - oy)) * dw + (x - ox);
        if (owner) owner[d] = (uint8_t)bq;
        if (memb_f) memb_f[d] = memb;
        if (ids && margin[v] > lthr) { ids[d] = base + (uint32_t)bq + 1u; score[d] = (uint8_t)rintf(255.f * memb); }
    }
}
extern "C" void hv_final(const void *U, int S, const float *me_dev, int Q, const float *margin_dev, float lthr, const int r0[3], const int r1[3],
                         const int off[3], const int dd[3], uint32_t *ids, uint32_t base, uint8_t *score, uint8_t *owner, float *memb_f) {
    const int nz = r1[0] - r0[0], ny = r1[1] - r0[1], nx = r1[2] - r0[2];
    if (nz <= 0 || ny <= 0 || nx <= 0) return;
    if (Q > HV_QMAX) { fprintf(stderr, "hv_final: %d queries\n", Q); return; }
    const long long n = (long long)nz * ny * nx; unsigned nb = nb_of(n, 256); if (nb > 65535 * 8) nb = 65535 * 8;
    k_final<<<nb, 256, 0, ST>>>((const __half *)U, S, me_dev, Q, margin_dev, lthr, r0[0], r0[1], r0[2], nz, ny, nx, off[0], off[1], off[2], dd[1], dd[2],
                                ids, base, score, owner, memb_f);
}

__global__ void k_blend(float *acc, float *wsum, int dz, int dy, int dx, const float *__restrict__ m, int P, int oz, int oy, int ox,
                        int z0, int y0, int x0, int nz, int ny, int nx, float inv2s2) {
    const long long n = (long long)nz * ny * nx; const int c = P / 2;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const int z = z0 + (int)(i / ((long long)ny * nx)), y = y0 + (int)(i / nx % ny), x = x0 + (int)(i % nx);   /* window coords */
        const float r2 = (float)((z - c) * (z - c) + (y - c) * (y - c) + (x - c) * (x - c));
        const float g = expf(-r2 * inv2s2);
        const size_t t = ((size_t)(z + oz) * dy + (y + oy)) * dx + (x + ox), v = ((size_t)z * P + y) * P + x;
        acc[t] += m[v] * g; wsum[t] += g;
    }
}
extern "C" void hv_blend(float *acc, float *wsum, const int dd[3], const float *margin_dev, int P, const int org[3]) {
    int lo[3], hi[3];
    for (int a = 0; a < 3; a++) { lo[a] = org[a] < 0 ? -org[a] : 0; hi[a] = dd[a] - org[a] < P ? dd[a] - org[a] : P; if (hi[a] <= lo[a]) return; }
    const float s = P / 8.f;
    const long long n = (long long)(hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]); unsigned nb = nb_of(n, 256); if (nb > 65535 * 8) nb = 65535 * 8;
    k_blend<<<nb, 256, 0, ST>>>(acc, wsum, dd[0], dd[1], dd[2], margin_dev, P, org[0], org[1], org[2], lo[0], lo[1], lo[2], hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1.f / (2.f * s * s));
}
__global__ void k_finish(const float *acc, const float *wsum, size_t n, uint8_t *u8) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float w = wsum[i];
        if (!(w > 0.f)) { u8[i] = 0; continue; }
        const float l = fminf(fmaxf(acc[i] / fmaxf(w, 1e-12f), -30.f), 30.f);
        u8[i] = (uint8_t)fminf(fmaxf(rintf(255.f / (1.f + expf(-l))), 0.f), 255.f);
    }
}
extern "C" void hv_finish(const float *acc, const float *wsum, size_t n, uint8_t *u8) {
    unsigned nb = nb_of((long long)n, 256); if (nb > 65535 * 8) nb = 65535 * 8;
    k_finish<<<nb, 256, 0, ST>>>(acc, wsum, n, u8);
}
