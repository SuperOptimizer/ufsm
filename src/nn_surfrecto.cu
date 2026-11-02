/* surfrecto kernels (src/surfrecto.h): scSE + residual + LeakyReLU, mirror-TTA flips. */
#include "surfrecto.h"
#include "xnet.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

#define ST ((cudaStream_t)xnet_stream())
static unsigned nblocks(size_t n, int t) { size_t b = (n + t - 1) / t; return b > 65535u * 16 ? 65535u * 16 : (unsigned)b; }

/* one warp per voxel: s = sigmoid(h . sw + sb); y = lrelu(h (cg + s) + res); C a multiple of 32 (<= 512), half2 per lane */
__global__ void k_scse(__half *h, const __half *res, size_t nvox, int C, const float *sw, float sb, const float *cg) {
    const int lane = threadIdx.x & 31;
    const size_t wpb = blockDim.x >> 5;
    for (size_t v = blockIdx.x * wpb + (threadIdx.x >> 5); v < nvox; v += (size_t)gridDim.x * wpb) {
        __half2 *hp = (__half2 *)(h + v * C); const __half2 *rp = (const __half2 *)(res + v * C);
        const int C2 = C >> 1;
        float2 hv[8]; float dot = 0.f; int m = 0;
        for (int j = lane; j < C2; j += 32, m++) { hv[m] = __half22float2(hp[j]); dot += hv[m].x * sw[2 * j] + hv[m].y * sw[2 * j + 1]; }
        for (int o = 16; o; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
        const float s = 1.f / (1.f + __expf(-(dot + sb)));
        m = 0;
        for (int j = lane; j < C2; j += 32, m++) {
            const float2 r = __half22float2(rp[j]);
            float a = hv[m].x * (cg[2 * j] + s) + r.x, b = hv[m].y * (cg[2 * j + 1] + s) + r.y;
            a = a > 0.f ? a : 0.01f * a; b = b > 0.f ? b : 0.01f * b;
            hp[j] = __floats2half2_rn(a, b);
        }
    }
}
extern "C" void sr_scse(void *h, const void *res, size_t nvox, int C, const float *sw_dev, float sb, const float *cg_dev) {
    const int wpb = 8; size_t nb = (nvox + wpb - 1) / wpb; if (nb > 65535u * 32) nb = 65535u * 32;
    k_scse<<<(unsigned)nb, wpb * 32, 0, ST>>>((__half *)h, (const __half *)res, nvox, C, sw_dev, sb, cg_dev);
}

__global__ void k_flip_u8(const uint8_t *in, uint8_t *out, int P, int mask) {
    const size_t n = (size_t)P * P * P;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        int x = (int)(i % P); size_t t = i / P; int y = (int)(t % P); int z = (int)(t / P);
        if (mask & 1) x = P - 1 - x; if (mask & 2) y = P - 1 - y; if (mask & 4) z = P - 1 - z;
        out[i] = in[((size_t)z * P + y) * P + x];
    }
}
extern "C" void sr_flip_u8(const uint8_t *in, uint8_t *out, int P, int mask) {
    const size_t n = (size_t)P * P * P; k_flip_u8<<<nblocks(n, 256), 256, 0, ST>>>(in, out, P, mask);
}
__global__ void k_flip_acc(float *acc, const float *m, int P, int mask, int first) {
    const size_t n = (size_t)P * P * P;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        int x = (int)(i % P); size_t t = i / P; int y = (int)(t % P); int z = (int)(t / P);
        if (mask & 1) x = P - 1 - x; if (mask & 2) y = P - 1 - y; if (mask & 4) z = P - 1 - z;
        const float v = m[((size_t)z * P + y) * P + x];
        acc[i] = first ? v : acc[i] + v;
    }
}
extern "C" void sr_flip_acc(float *acc, const float *m, int P, int mask, int first) {
    const size_t n = (size_t)P * P * P; k_flip_acc<<<nblocks(n, 256), 256, 0, ST>>>(acc, m, P, mask, first);
}
__global__ void k_scale(float *p, size_t n, float s) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] *= s;
}
extern "C" void sr_scale(float *p, size_t n, float s) { k_scale<<<nblocks(n, 256), 256, 0, ST>>>(p, n, s); }
extern "C" void sr_copy_dev(void *dst, const void *src, size_t bytes) { cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, ST); }

/* ---- InstanceNorm (one sample, fp16 [S][C], C % 8 == 0): one-pass shifted statistics. Each thread owns 8 channels of a voxel
   column; sums of (x - K) and (x - K)^2 with the per-channel shift K = the first voxel's value (no cancellation for channels whose
   mean is far from 0), float per thread, float block reduction, double across blocks. ---- */
__global__ void k_in_stats(const __half *x, int C, long long S, long long per_block, double *acc /* [2][C] */) {
    extern __shared__ float sh[];   /* [2][C] */
    const int tpv = C >> 3, rows = blockDim.x / tpv, r = threadIdx.x / tpv, cg = threadIdx.x % tpv;
    for (int c = threadIdx.x; c < 2 * C; c += blockDim.x) sh[c] = 0.f;
    __syncthreads();
    if (r < rows) {
        const long long s0 = blockIdx.x * per_block, s1 = s0 + per_block < S ? s0 + per_block : S;
        float K[8], s[8], q[8];
        { const uint4 kv = *(const uint4 *)(x + cg * 8); const __half *kh = (const __half *)&kv; for (int j = 0; j < 8; j++) { K[j] = __half2float(kh[j]); s[j] = q[j] = 0.f; } }
        for (long long v = s0 + r; v < s1; v += rows) {
            const uint4 u = *(const uint4 *)(x + (size_t)v * C + cg * 8); const __half *h = (const __half *)&u;
            for (int j = 0; j < 8; j++) { const float d = __half2float(h[j]) - K[j]; s[j] += d; q[j] += d * d; }
        }
        for (int j = 0; j < 8; j++) { atomicAdd(&sh[cg * 8 + j], s[j]); atomicAdd(&sh[C + cg * 8 + j], q[j]); }
    }
    __syncthreads();
    for (int c = threadIdx.x; c < 2 * C; c += blockDim.x) atomicAdd(&acc[c], (double)sh[c]);
}
__global__ void k_in_final(const __half *x, const double *acc, int C, long long S, float eps, float *mr /* [2][C]: mean, rstd */) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= C) return;
    const double K = __half2float(x[c]), m = acc[c] / (double)S, var = acc[C + c] / (double)S - m * m;
    mr[c] = (float)(K + m); mr[C + c] = (float)(1.0 / sqrt((var > 0 ? var : 0) + eps));
}
extern "C" void sr_in_stats(const void *x, int C, long long S, float eps, float *mr_dev, double *acc_dev) {
    cudaMemsetAsync(acc_dev, 0, sizeof(double) * 2 * C, ST);
    const int tpv = C / 8; int th = (256 / tpv) * tpv; if (th < tpv) th = tpv;
    const int rows = th / tpv;
    long long per = (S + 4095) / 4096; if (per < rows * 16LL) per = rows * 16LL;
    const long long nb = (S + per - 1) / per;
    k_in_stats<<<(unsigned)nb, th, 2 * C * sizeof(float), ST>>>((const __half *)x, C, S, per, acc_dev);
    k_in_final<<<(C + 127) / 128, 128, 0, ST>>>((const __half *)x, acc_dev, C, S, eps, mr_dev);
}
/* y = act((x - mean) rstd gamma + beta), in place, 8 channels per thread */
__global__ void k_in_apply(__half *x, int C, size_t n8, const float *mr, const float *g, const float *b, int lrelu) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n8; i += (size_t)gridDim.x * blockDim.x) {
        const int c0 = (int)((i * 8) % C);
        uint4 u = *(uint4 *)(x + i * 8); __half *h = (__half *)&u;
        for (int j = 0; j < 8; j++) {
            const int c = c0 + j; float v = (__half2float(h[j]) - mr[c]) * mr[C + c] * g[c] + b[c];
            if (lrelu) v = v < 0.f ? 0.01f * v : v;
            h[j] = __float2half_rn(v);
        }
        *(uint4 *)(x + i * 8) = u;
    }
}
extern "C" void sr_in_apply(void *x, int C, size_t nvox, const float *mr_dev, const float *g_dev, const float *b_dev, int lrelu) {
    const size_t n8 = nvox * C / 8; k_in_apply<<<nblocks(n8, 256), 256, 0, ST>>>((__half *)x, C, n8, mr_dev, g_dev, b_dev, lrelu);
}
/* conv2 IN apply fused into scSE: hn = IN(h); y = lrelu(hn (cg + sigmoid(hn . sw + sb)) + res); one warp per voxel */
__global__ void k_in_scse(__half *h, const __half *res, size_t nvox, int C, const float *mr, const float *g, const float *b, const float *sw, float sb, const float *cg) {
    const int lane = threadIdx.x & 31;
    const size_t wpb = blockDim.x >> 5;
    for (size_t v = blockIdx.x * wpb + (threadIdx.x >> 5); v < nvox; v += (size_t)gridDim.x * wpb) {
        __half2 *hp = (__half2 *)(h + v * C); const __half2 *rp = (const __half2 *)(res + v * C);
        const int C2 = C >> 1;
        float2 hv[8]; float dot = 0.f; int m = 0;
        for (int j = lane; j < C2; j += 32, m++) {
            const float2 f = __half22float2(hp[j]); const int c = 2 * j;
            /* round the normalised value to fp16 as the separate IN pass would */
            hv[m].x = __half2float(__float2half_rn((f.x - mr[c]) * mr[C + c] * g[c] + b[c]));
            hv[m].y = __half2float(__float2half_rn((f.y - mr[c + 1]) * mr[C + c + 1] * g[c + 1] + b[c + 1]));
            dot += hv[m].x * sw[c] + hv[m].y * sw[c + 1];
        }
        for (int o = 16; o; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
        const float s = 1.f / (1.f + __expf(-(dot + sb)));
        m = 0;
        for (int j = lane; j < C2; j += 32, m++) {
            const float2 r = __half22float2(rp[j]);
            float a = hv[m].x * (cg[2 * j] + s) + r.x, bb = hv[m].y * (cg[2 * j + 1] + s) + r.y;
            a = a > 0.f ? a : 0.01f * a; bb = bb > 0.f ? bb : 0.01f * bb;
            hp[j] = __floats2half2_rn(a, bb);
        }
    }
}
extern "C" void sr_in_scse(void *h, const void *res, size_t nvox, int C, const float *mr_dev, const float *g_dev, const float *b_dev,
                           const float *sw_dev, float sb, const float *cg_dev) {
    const int wpb = 8; size_t nb = (nvox + wpb - 1) / wpb; if (nb > 65535u * 32) nb = 65535u * 32;
    k_in_scse<<<(unsigned)nb, wpb * 32, 0, ST>>>((__half *)h, (const __half *)res, nvox, C, mr_dev, g_dev, b_dev, sw_dev, sb, cg_dev);
}
