/* ink-proto (2026-11-27, inkproto; src/inkproto.h): the prototype's cosine into the output logits, and the seed sums of confident voxels */
#include "nn_common.cuh"
extern "C" {
#include "inkproto.h"
}

/* (1 + cos(f, pr)) / 2 as a logit in the output channel (the placement kernels write round(255 sigmoid)); pr is unit length */
template <typename LT> __global__ void proto_cos_k(const f16 *ft, int C, size_t nv, const float *pr, LT *lg) {
    const size_t v = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (v >= nv) return;
    float d = 0.f, s = 0.f;
    for (int c = 0; c < C; c++) { const float f = __half2float(ft[(size_t)c * nv + v]); d = fmaf(f, pr[c], d); s = fmaf(f, f, s); }
    float p = s > 0.f ? 0.5f * (1.f + d / sqrtf(s)) : 0.5f;
    p = p < 1e-6f ? 1e-6f : p > 1.f - 1e-6f ? 1.f - 1e-6f : p;
    lg[v] = (LT)logf(p / (1.f - p));
}
extern "C" void nn_proto_cos(const void *ft, int C, size_t nv, const float *pr_dev, void *lg, int h16) {
    if (!nv) return;
    if (h16) proto_cos_k<f16><<<nblk(nv, 256), 256>>>((const f16 *)ft, C, nv, pr_dev, (f16 *)lg);
    else proto_cos_k<float><<<nblk(nv, 256), 256>>>((const f16 *)ft, C, nv, pr_dev, (float *)lg);
    KCHECK();
}

/* the seed sums: per block, each warp's sums of the passing voxels (shuffles), then the block's into acc by double atomics (one per entry
   and block: a confident region passes most voxels, so per-voxel atomics would serialise on the 2 C + 1 addresses) */
#define PA_T 256
template <typename LT> __global__ void proto_acc_k(const f16 *ft, int C, int z0, int nz, int D, int H, int W, int hz, int hy, int hx, const LT *lg, float thr, double *acc) {
    __shared__ float sh[PA_T / 32][2 * INK_PROTO_MAXC + 1];
    const size_t hw = (size_t)H * W, nv = (size_t)nz * hw, v = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    int ok = 0;
    if (v < nv) {
        const int z = z0 + (int)(v / hw), y = (int)(v / W % H), x = (int)(v % W);
        ok = z >= hz && z < D - hz && y >= hy && y < H - hy && x >= hx && x < W - hx && (float)lg[v] >= thr;
    }
    if (!__any_sync(~0u, ok)) { for (int e = lane; e < 2 * C + 1; e += 32) sh[wp][e] = 0.f; }
    else {
        float s2 = 0.f;
        if (ok) for (int c = 0; c < C; c++) { const float f = __half2float(ft[(size_t)c * nv + v]); s2 = fmaf(f, f, s2); }
        const float inv = ok && s2 > 0.f ? 1.f / sqrtf(s2) : 0.f;
        for (int c = 0; c < C; c++) {
            float a = ok ? __half2float(ft[(size_t)c * nv + v]) : 0.f, b = a * inv;
            for (int o = 16; o; o >>= 1) { a += __shfl_down_sync(~0u, a, o); b += __shfl_down_sync(~0u, b, o); }
            if (!lane) { sh[wp][c] = a; sh[wp][C + c] = b; }
        }
        float k = (float)ok; for (int o = 16; o; o >>= 1) k += __shfl_down_sync(~0u, k, o);
        if (!lane) sh[wp][2 * C] = k;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < 2 * C + 1; e += blockDim.x) {
        double t = 0; for (int w = 0; w < PA_T / 32; w++) t += sh[w][e];
        if (t != 0) atomicAdd(&acc[e], t);
    }
}
extern "C" void nn_proto_acc(const void *ft, int C, int z0, int nz, int D, int H, int W, int hz, int hy, int hx, const void *lg, int h16, float thr_logit, double *acc_dev) {
    const size_t nv = (size_t)nz * H * W;
    if (!nv || C > INK_PROTO_MAXC) return;
    if (h16) proto_acc_k<f16><<<nblk(nv, PA_T), PA_T>>>((const f16 *)ft, C, z0, nz, D, H, W, hz, hy, hx, (const f16 *)lg, thr_logit, acc_dev);
    else proto_acc_k<float><<<nblk(nv, PA_T), PA_T>>>((const f16 *)ft, C, z0, nz, D, H, W, hz, hy, hx, (const float *)lg, thr_logit, acc_dev);
    KCHECK();
}
