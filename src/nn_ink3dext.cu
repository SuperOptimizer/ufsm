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
static int g_g1_O[32];   /* mgpu: per-device (c_g1 lives in each context: the O uploaded to device d; one host thread per device) */
static void g1_init(int O) {
    int dv = 0; cudaGetDevice(&dv); dv &= 31;
    if (g_g1_O[dv] == O) return;
    float g[I3_OMAX]; const double sg = O / 8.0;
    for (int i = 0; i < O; i++) g[i] = (float)exp(-0.5 * (double)(i - O / 2) * (i - O / 2) / (sg * sg));
    cudaMemcpyToSymbol(c_g1, g, sizeof(float) * O); g_g1_O[dv] = O;
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

/* ---- the fast path (ink3d-ext --fast, gambiarra): level-1 input on a crop grid aligned to 16 level-1 voxels; a crop of T^3 (64: the
   published geometry; 128: larger tiles) has the parent [a - 32, a + T + 32)^3, exactly ((T + 64) / 16)^3 blocks of 16^3, so its p1 / p99
   histogram is a sum of block histograms computed once per slab; percentiles, the lut, the crop gather and the fp16 input on the GPU ---- */
__global__ void k_i3f_bhist(const uint8_t *slab, int Y, int X, uint32_t *hist) {   /* grid (X / 16, Y / 16, Z / 16), 256 threads */
    __shared__ uint32_t h[256];
    h[threadIdx.x] = 0; __syncthreads();
    const int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    for (int i = threadIdx.x; i < 16 * 16 * 4; i += 256) {   /* 4-byte words */
        const int xw = i & 3, y = (i >> 2) & 15, z = i >> 6;
        const uint32_t w = *(const uint32_t *)(slab + ((size_t)(bz * 16 + z) * Y + by * 16 + y) * X + bx * 16 + xw * 4);
        atomicAdd(&h[w & 255], 1u); atomicAdd(&h[(w >> 8) & 255], 1u); atomicAdd(&h[(w >> 16) & 255], 1u); atomicAdd(&h[w >> 24], 1u);
    }
    __syncthreads();
    hist[(((size_t)bz * gridDim.y + by) * gridDim.x + bx) * 256 + threadIdx.x] = h[threadIdx.x];
}
extern "C" void i3f_bhist(const uint8_t *slab, int Z, int Y, int X, uint32_t *hist) {
    k_i3f_bhist<<<dim3(X / 16, Y / 16, Z / 16), 256, 0, ST>>>(slab, Y, X, hist);
}
__global__ void k_i3f_cmax(const uint8_t *slab, int Y, int X, int T, const int *pyx, int *cmax) {   /* grid crops: max of the crop [32, 32 + T)^3 of its parent */
    __shared__ int m; if (!threadIdx.x) m = 0; __syncthreads();
    const int py = pyx[2 * blockIdx.x], px = pyx[2 * blockIdx.x + 1], TW = T / 4; uint32_t mx = 0;
    for (int i = threadIdx.x; i < T * T * TW; i += 256) {
        const int xw = i % TW, y = (i / TW) % T, z = i / (TW * T);
        const uint32_t w = *(const uint32_t *)(slab + ((size_t)(32 + z) * Y + py + 32 + y) * X + px + 32 + xw * 4);
        mx = max(mx, max(max(w & 255, (w >> 8) & 255), max((w >> 16) & 255, w >> 24)));
    }
    atomicMax(&m, (int)mx); __syncthreads();
    if (!threadIdx.x) cmax[blockIdx.x] = m;
}
extern "C" void i3f_cmax(const uint8_t *slab, int Y, int X, int T, const int *pyx_dev, int n, int *cmax_dev) {
    if (n > 0) k_i3f_cmax<<<n, 256, 0, ST>>>(slab, Y, X, T, pyx_dev, cmax_dev);
}
__global__ void k_i3f_crops(const uint8_t *slab, int Y, int X, int T, const uint32_t *hist, int NBY, int NBX, const int *pyx, __half *out, float *lohi) {
    __shared__ uint32_t cum[256]; __shared__ float lut[256]; __shared__ double plo, phi;
    const int py = pyx[2 * blockIdx.x], px = pyx[2 * blockIdx.x + 1], by0 = py / 16, bx0 = px / 16, t = threadIdx.x, NB = (T + 64) / 16;
    uint32_t s = 0;
    for (int bz = 0; bz < NB; bz++) for (int by = 0; by < NB; by++) for (int bx = 0; bx < NB; bx++) s += hist[(((size_t)bz * NBY + by0 + by) * NBX + bx0 + bx) * 256 + t];
    cum[t] = s; __syncthreads();
    for (int o = 1; o < 256; o <<= 1) { const uint32_t v = t >= o ? cum[t - o] : 0; __syncthreads(); cum[t] += v; __syncthreads(); }
    if (t == 0) {   /* i3_percentiles: np.percentile (linear) of the n = (T + 64)^3 values */
        const uint64_t n = (uint64_t)(T + 64) * (T + 64) * (T + 64); const double q[2] = {1.0 / 100.0, 99.0 / 100.0}; double r[2];
        for (int k = 0; k < 2; k++) {
            const double virt = (double)(n - 1) * q[k], below = floor(virt), tt = virt - below;
            uint64_t ia = (uint64_t)below, ib = ia + 1; if (ib > n - 1) ib = n - 1;
            int a = -1, b = -1;
            for (int v = 0; v < 256 && b < 0; v++) { if (a < 0 && cum[v] > ia) a = v; if (cum[v] > ib) b = v; }
            if (a < 0) a = 255; if (b < 0) b = 255;
            const double fa = a, fb = b, diff = fb - fa;
            r[k] = tt >= 0.5 ? fb - diff * (1.0 - tt) : fa + diff * tt;
        }
        plo = r[0]; phi = r[1];
        if (lohi) { lohi[2 * blockIdx.x] = (float)r[0]; lohi[2 * blockIdx.x + 1] = (float)r[1]; }
    }
    __syncthreads();
    {   /* i3_lut */
        const double lo = plo, hi = phi; float x;
        if (hi - lo < 1e-6) x = 0.f;
        else { const double c = t < lo ? lo : t > hi ? hi : (double)t; x = (float)c; x = (float)((double)x - lo); x = (float)((double)x / (hi - lo)); }
        lut[t] = x;
    }
    __syncthreads();
    __half *o = out + (size_t)blockIdx.x * T * T * T; const int TW = T / 4;
    for (int i = t; i < T * T * TW; i += 256) {
        const int xw = i % TW, y = (i / TW) % T, z = i / (TW * T);
        const uint32_t w = *(const uint32_t *)(slab + ((size_t)(32 + z) * Y + py + 32 + y) * X + px + 32 + xw * 4);
        __align__(8) __half h4[4] = {__float2half_rn(lut[w & 255]), __float2half_rn(lut[(w >> 8) & 255]), __float2half_rn(lut[(w >> 16) & 255]), __float2half_rn(lut[w >> 24])};
        *(uint2 *)(o + ((size_t)z * T + y) * T + xw * 4) = *(const uint2 *)h4;
    }
}
extern "C" void i3f_crops(const uint8_t *slab, int Y, int X, int T, const uint32_t *hist, const int *pyx_dev, int n, void *out_half, float *lohi_dev) {
    if (n > 0) k_i3f_crops<<<n, 256, 0, ST>>>(slab, Y, X, T, hist, Y / 16, X / 16, pyx_dev, (__half *)out_half, lohi_dev);
}
