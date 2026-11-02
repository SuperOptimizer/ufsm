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
