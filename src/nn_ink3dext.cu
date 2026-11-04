/* ink3d-ext kernels (src/ink3dext.h): input preparation, Gaussian blend into z-ring accumulators, emission of prob254 rows. */
#include "ink3dext.h"
#include "xnet.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

#define ST ((cudaStream_t)xnet_stream())
static unsigned nblocks(size_t n, int t) { size_t b = (n + t - 1) / t; return b > 65535u * 16 ? 65535u * 16 : (unsigned)b; }

/* crop C^3 u8 -> net input (C / pool)^3 fp16: mean of the lut values over each pool^3 block (pool 1: the lut value) */
__global__ void k_i3_prep(const uint8_t *u8, const float *lut, int B, int C, int pool, __half *out) {
    const int N = C / pool;
    const size_t total = (size_t)B * N * N * N;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const int x = (int)(i % N); size_t t = i / N; const int y = (int)(t % N); t /= N; const int z = (int)(t % N); const size_t b = t / N;
        const uint8_t *s = u8 + b * (size_t)C * C * C + ((size_t)(pool * z) * C + pool * y) * C + pool * x;
        const float *L = lut + b * 256;
        float a = 0.f;   /* torch avg_pool3d sums the values in z, y, x order */
        for (int dz = 0; dz < pool; dz++) for (int dy = 0; dy < pool; dy++) for (int dx = 0; dx < pool; dx++)
            a += L[s[((size_t)dz * C + dy) * C + dx]];
        out[i] = __float2half_rn(a / (float)(pool * pool * pool));
    }
}
extern "C" void i3_prep(const uint8_t *u8, const float *lut, int B, int C, int pool, void *out_half) {
    const int N = C / pool; const size_t n = (size_t)B * N * N * N;
    k_i3_prep<<<nblocks(n, 256), 256, 0, ST>>>(u8, lut, B, C, pool, (__half *)out_half);
}

/* logits [B][N^3] -> [B][(N / 2)^3]: mean of each 2^3 block */
__global__ void k_i3_pool2(const float *in, int B, int N, float *out) {
    const int M = N / 2; const size_t total = (size_t)B * M * M * M;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const int x = (int)(i % M); size_t t = i / M; const int y = (int)(t % M); t /= M; const int z = (int)(t % M); const size_t b = t / M;
        const float *s = in + b * (size_t)N * N * N + ((size_t)(2 * z) * N + 2 * y) * N + 2 * x;
        float a = 0.f;
        for (int dz = 0; dz < 2; dz++) for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) a += s[((size_t)dz * N + dy) * N + dx];
        out[i] = a / 8.f;
    }
}
extern "C" void i3_pool2(const float *in, int B, int N, float *out) {
    const int M = N / 2; const size_t n = (size_t)B * M * M * M;
    k_i3_pool2<<<nblocks(n, 256), 256, 0, ST>>>(in, B, N, out);
}

/* villa volume.gaussian(): a delta through scipy gaussian_filter (sigma O / 8, truncate 4), divided by its maximum =
   prod over axes exp(-(i - O / 2)^2 / (2 sigma^2)); O = 64 (gambiarra): sigma 8 */
enum { I3_OMAX = 256 };
__constant__ float c_g1[I3_OMAX];
static int g_g1_O = 0;
static void g1_init(int O) {
    if (g_g1_O == O) return;
    float g[I3_OMAX]; const double sg = O / 8.0;
    for (int i = 0; i < O; i++) g[i] = (float)exp(-0.5 * (double)(i - O / 2) * (i - O / 2) / (sg * sg));
    cudaMemcpyToSymbol(c_g1, g, sizeof(float) * O); g_g1_O = O;
}
__global__ void k_i3_blend(float *acc, float *wsum, int R, int TY, int TX, const float *logits, const int *org, int B, int z0, int z1, int O) {
    const size_t total = (size_t)B * O * O * O;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const int x = (int)(i % O); size_t t = i / O; const int y = (int)(t % O); t /= O; const int z = (int)(t % O); const int b = (int)(t / O);
        const int gz = org[3 * b] + z, ty = org[3 * b + 1] + y, tx = org[3 * b + 2] + x;
        if (gz < z0 || gz >= z1 || ty < 0 || ty >= TY || tx < 0 || tx >= TX) continue;
        const float g = c_g1[z] * c_g1[y] * c_g1[x];
        const size_t o = ((size_t)(gz % R) * TY + ty) * TX + tx;
        atomicAdd(acc + o, g * logits[i]); atomicAdd(wsum + o, g);
    }
}
extern "C" void i3_blend(float *acc, float *wsum, int R, int TY, int TX, const float *logits, const int *org_dev, int B, int z0, int z1, int O) {
    g1_init(O);
    const size_t n = (size_t)B * O * O * O;
    k_i3_blend<<<nblocks(n, 256), 256, 0, ST>>>(acc, wsum, R, TY, TX, logits, org_dev, B, z0, z1, O);
}

__device__ __forceinline__ int hbin(float v) { const float b = floorf((v + 16.f) * 2.f); return b < 0.f ? 0 : b >= 64.f ? 65 : 1 + (int)b; }
/* one thread per 9.6 voxel (2 x 2 x 2 block of 4.8 voxels) */
__global__ void k_i3_emit(const float *acc, const float *wsum, int R, int TY, int TX, int z0, int z1, uint8_t *o48, uint8_t *o96, unsigned long long *hist) {
    __shared__ unsigned int sh[I3_HBINS];
    for (int k = threadIdx.x; k < I3_HBINS; k += blockDim.x) sh[k] = 0;
    __syncthreads();
    const int NZ = (z1 - z0 + 1) / 2, NY = (TY + 1) / 2, NX = (TX + 1) / 2;
    const size_t total = (size_t)NZ * NY * NX;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const int X = (int)(i % NX); size_t t = i / NX; const int Y = (int)(t % NY); const int Z = (int)(t / NY);
        float sp = 0.f; int nk = 0;
        for (int dz = 0; dz < 2; dz++) {
            const int z = z0 + 2 * Z + dz; if (z >= z1) continue;
            for (int dy = 0; dy < 2; dy++) {
                const int y = 2 * Y + dy; if (y >= TY) continue;
                for (int dx = 0; dx < 2; dx++) {
                    const int x = 2 * X + dx; if (x >= TX) continue;
                    const size_t o = ((size_t)(z % R) * TY + y) * TX + x;
                    const float w = wsum[o]; uint8_t v = 255;
                    if (w > 0.f) {
                        const float l = acc[o] / w, p = 1.f / (1.f + expf(-l));
                        sp += p; nk++; v = (uint8_t)__float2int_rn(p * 254.f);
                        if (hist) atomicAdd(&sh[hbin(l)], 1u);
                    }
                    if (o48) o48[((size_t)(z - z0) * TY + y) * TX + x] = v;
                }
            }
        }
        if (o96) o96[i] = nk ? (uint8_t)__float2int_rn(sp / nk * 254.f) : 255;
    }
    if (hist) { __syncthreads(); for (int k = threadIdx.x; k < I3_HBINS; k += blockDim.x) if (sh[k]) atomicAdd(hist + k, (unsigned long long)sh[k]); }
}
extern "C" void i3_emit(float *acc, float *wsum, int R, int TY, int TX, int z0, int z1, uint8_t *out48, uint8_t *out96, unsigned long long *hist) {
    if (z1 <= z0) return;
    const size_t n = (size_t)((z1 - z0 + 1) / 2) * ((TY + 1) / 2) * ((TX + 1) / 2);
    k_i3_emit<<<nblocks(n, 256), 256, 0, ST>>>(acc, wsum, R, TY, TX, z0, z1, out48, out96, hist);
    /* zero the emitted ring rows (z0 .. z1 - 1, contiguous modulo R) */
    const size_t row = (size_t)TY * TX;
    for (int z = z0; z < z1;) {
        const int r = z % R, n2 = (z1 - z) < (R - r) ? (z1 - z) : (R - r);
        cudaMemsetAsync(acc + (size_t)r * row, 0, (size_t)n2 * row * 4, ST);
        cudaMemsetAsync(wsum + (size_t)r * row, 0, (size_t)n2 * row * 4, ST);
        z += n2;
    }
}
