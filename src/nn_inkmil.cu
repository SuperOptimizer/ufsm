/* the normal-pooled MIL ink loss on the GPU (src/inkmil.h): one thread per ray, its samples in registers, the gradient by atomics. */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cub/device/device_radix_sort.cuh>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "inkmil.h"

enum { KMAX = 256 };
__device__ static inline float tri_f(const float *v, int D, int H, int W, float z, float y, float x, int c[3], float f[3]) {   /* (D, H, W >= 2) */
    if (!(z >= 0 && y >= 0 && x >= 0 && z <= D - 1 && y <= H - 1 && x <= W - 1)) return NAN;
    int z0 = (int)z, y0 = (int)y, x0 = (int)x; if (z0 > D - 2) z0 = D - 2; if (y0 > H - 2) y0 = H - 2; if (x0 > W - 2) x0 = W - 2;
    const float fz = z - z0, fy = y - y0, fx = x - x0; c[0] = z0; c[1] = y0; c[2] = x0; f[0] = fz; f[1] = fy; f[2] = fx;
    const size_t b = ((size_t)z0 * H + y0) * W + x0, sy = W, sz = (size_t)H * W;
    const float a00 = v[b] * (1 - fx) + v[b + 1] * fx, a01 = v[b + sy] * (1 - fx) + v[b + sy + 1] * fx;
    const float a10 = v[b + sz] * (1 - fx) + v[b + sz + 1] * fx, a11 = v[b + sz + sy] * (1 - fx) + v[b + sz + sy + 1] * fx;
    return (a00 * (1 - fy) + a01 * fy) * (1 - fz) + (a10 * (1 - fy) + a11 * fy) * fz;
}
__global__ void k_mil(const float *logit, float *grad, int D, int H, int W, const inkray *rays, int nr, float R, float step, int K, float tau, float eps, float scale, double *loss) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= nr) return;
    const inkray r = rays[i]; float z[KMAX]; int nv = 0, am = -1; float zm = -INFINITY;
    for (int k = 0; k < K; k++) {
        const float t = -R + k * step; int c[3]; float f[3];
        const float v = tri_f(logit, D, H, W, r.p[0] + t * r.n[0], r.p[1] + t * r.n[1], r.p[2] + t * r.n[2], c, f);
        z[k] = v; if (v == v) { nv++; if (v > zm) { zm = v; am = k; } }
    }
    if (!nv) return;
    float zr, se = 0;
    if (tau > 0) { for (int k = 0; k < K; k++) if (z[k] == z[k]) se += expf((z[k] - zm) / tau); zr = zm + tau * logf(se / nv); } else zr = zm;
    const float y = r.y ? 1 - eps / 2 : eps / 2;
    const float l = fmaxf(zr, 0) - zr * y + log1pf(expf(-fabsf(zr)));
    atomicAdd(loss, (double)(scale * r.w * l));
    const float g = scale * r.w * (1 / (1 + expf(-zr)) - y);
    for (int k = 0; k < K; k++) {
        if (!(z[k] == z[k])) continue;
        const float dk = tau > 0 ? expf((z[k] - zm) / tau) / se : (k == am ? 1.f : 0.f); if (dk == 0) continue;
        const float t = -R + k * step; int c[3]; float f[3];
        tri_f(logit, D, H, W, r.p[0] + t * r.n[0], r.p[1] + t * r.n[1], r.p[2] + t * r.n[2], c, f);
        for (int q = 0; q < 8; q++) {
            const int a = c[0] + (q >> 2 & 1), b = c[1] + (q >> 1 & 1), e = c[2] + (q & 1);
            const float wq = ((q >> 2 & 1) ? f[0] : 1 - f[0]) * ((q >> 1 & 1) ? f[1] : 1 - f[1]) * ((q & 1) ? f[2] : 1 - f[2]);
            if (wq != 0) atomicAdd(&grad[((size_t)a * H + b) * W + e], g * dk * wq);
        }
    }
}
extern "C" int ink_mil_loss_dev(const float *logit, float *grad, int D, int H, int W, const inkray *rays, int nr, float R, float step, float tau, float eps, float scale, double *loss, void *stream) {
    if (nr <= 0) return 0;
    const int K = (int)floorf(2 * R / step + 1e-4f) + 1; if (K > KMAX || K < 1) return -2;
    k_mil<<<(nr + 127) / 128, 128, 0, (cudaStream_t)stream>>>(logit, grad, D, H, W, rays, nr, R, step, K, tau, eps, scale, loss);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
/* tests: the same on host memory (copies in and out; grad accumulated into the host array) */
extern "C" int ink_mil_loss_hostmem(const float *logit, float *grad, int D, int H, int W, const inkray *rays, int nr, float R, float step, float tau, float eps, float scale, double *loss) {
    const size_t nv = (size_t)D * H * W; float *dl = nullptr, *dg = nullptr; inkray *dr = nullptr; double *dL = nullptr; int rc = -1;
    if (cudaMalloc(&dl, nv * 4) || cudaMalloc(&dg, nv * 4) || cudaMalloc(&dr, (size_t)(nr ? nr : 1) * sizeof *dr) || cudaMalloc(&dL, 8)) goto out;
    if (cudaMemcpy(dl, logit, nv * 4, cudaMemcpyHostToDevice) || cudaMemcpy(dg, grad, nv * 4, cudaMemcpyHostToDevice) || cudaMemcpy(dr, rays, (size_t)nr * sizeof *dr, cudaMemcpyHostToDevice) ||
        cudaMemcpy(dL, loss, 8, cudaMemcpyHostToDevice)) goto out;
    if (ink_mil_loss_dev(dl, dg, D, H, W, dr, nr, R, step, tau, eps, scale, dL, nullptr)) goto out;
    if (cudaMemcpy(grad, dg, nv * 4, cudaMemcpyDeviceToHost) || cudaMemcpy(loss, dL, 8, cudaMemcpyDeviceToHost)) goto out;
    rc = 0;
out:
    cudaFree(dl); cudaFree(dg); cudaFree(dr); cudaFree(dL); return rc;
}

/* ---- the slab-incremental form (train's head_slab_pass: pass 0 the statistics slab by slab, pass 1 the gradients slab by slab).
   A sample's trilinear value is the sum of its 8 corner terms; fwd_slab adds the terms of the corners inside the slab to the sample's
   accumulator (one writer per sample: no atomics), so after the last slab acc holds every sample exactly once whatever the slab cuts.
   fin turns acc into dL/dz per sample (pooling, BCE, smoothing; the per-ray losses reduced in a fixed order). bwd_slab lists the
   (voxel, slot) pairs of the slab's corners, sorts them by voxel then slot (cub radix sort) and sums each voxel's run in slot order,
   so the gradient bytes do not depend on thread timing (--det). */

__device__ static inline float ld16(const void *p, size_t i, int bf) { return bf ? __bfloat162float(((const __nv_bfloat16 *)p)[i]) : __half2float(((const __half *)p)[i]); }
__device__ static inline void st16(void *p, size_t i, int bf, float v) { if (bf) ((__nv_bfloat16 *)p)[i] = __float2bfloat16(v); else ((__half *)p)[i] = __float2half(v); }
/* the sample's corners and weights (the full kernel's tri_f cell rule); 0 outside the volume */
__device__ static inline int samp(const inkray &r, int k, float R, float step, int D, int H, int W, int c[3], float f[3]) {
    const float t = -R + k * step, z = r.p[0] + t * r.n[0], y = r.p[1] + t * r.n[1], x = r.p[2] + t * r.n[2];
    if (!(z >= 0 && y >= 0 && x >= 0 && z <= D - 1 && y <= H - 1 && x <= W - 1)) return 0;
    int z0 = (int)z, y0 = (int)y, x0 = (int)x; if (z0 > D - 2) z0 = D - 2; if (y0 > H - 2) y0 = H - 2; if (x0 > W - 2) x0 = W - 2;
    c[0] = z0; c[1] = y0; c[2] = x0; f[0] = z - z0; f[1] = y - y0; f[2] = x - x0; return 1;
}
__device__ static inline float cw(const float f[3], int q) { return ((q >> 2 & 1) ? f[0] : 1 - f[0]) * ((q >> 1 & 1) ? f[1] : 1 - f[1]) * ((q & 1) ? f[2] : 1 - f[2]); }
__global__ void k_fwd_slab(const void *lg, int bf, int z0, int nz, int D, int H, int W, const inkray *rays, int nr, int K, float R, float step, float *acc) {
    const long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (id >= (long long)nr * K) return;
    const inkray r = rays[id / K]; int c[3]; float f[3];
    if (!samp(r, (int)(id % K), R, step, D, H, W, c, f)) return;
    if (c[0] + 1 < z0 || c[0] >= z0 + nz) return;
    float s = 0; int any = 0;
    for (int q = 0; q < 8; q++) { const int a = c[0] + (q >> 2 & 1); if (a < z0 || a >= z0 + nz) continue;
        s += cw(f, q) * ld16(lg, ((size_t)(a - z0) * H + c[1] + (q >> 1 & 1)) * W + c[2] + (q & 1), bf); any = 1; }
    if (any) acc[id] += s;
}
__global__ void k_fin(float *acc, const inkray *rays, int nr, int K, int D, int H, int W, float R, float step, float tau, float eps, float scale, double *lray) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= nr) return;
    const inkray r = rays[i]; float *z = acc + (size_t)i * K; int nv = 0, am = -1; float zm = -INFINITY; unsigned long long vm[4] = {0, 0, 0, 0};   /* (valid mask, K <= 256) */
    for (int k = 0; k < K; k++) { int c[3]; float f[3]; if (!samp(r, k, R, step, D, H, W, c, f)) continue; vm[k >> 6] |= 1ull << (k & 63); nv++; if (z[k] > zm) { zm = z[k]; am = k; } }
    lray[i] = 0;
    if (!nv) { for (int k = 0; k < K; k++) z[k] = 0; return; }
    float zr, se = 0;
    if (tau > 0) { for (int k = 0; k < K; k++) if (vm[k >> 6] >> (k & 63) & 1) se += expf((z[k] - zm) / tau); zr = zm + tau * logf(se / nv); } else zr = zm;
    const float y = r.y ? 1 - eps / 2 : eps / 2;
    lray[i] = (double)(scale * r.w * (fmaxf(zr, 0) - zr * y + log1pf(expf(-fabsf(zr)))));
    const float g = scale * r.w * (1 / (1 + expf(-zr)) - y);
    for (int k = 0; k < K; k++) {
        const int v = vm[k >> 6] >> (k & 63) & 1;
        z[k] = !v ? 0.f : tau > 0 ? g * expf((z[k] - zm) / tau) / se : (k == am ? g : 0.f);
    }
}
__global__ void k_sum_fixed(const double *v, int n, double *out) {   /* one block: a fixed-order sum (deterministic) */
    __shared__ double sh[256]; double s = 0;
    for (int i = threadIdx.x; i < n; i += 256) s += v[i];
    sh[threadIdx.x] = s; __syncthreads();
    for (int w = 128; w; w >>= 1) { if ((int)threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w]; __syncthreads(); }
    if (!threadIdx.x) *out += sh[0];
}
__global__ void k_pairs(const float *dz, int z0, int nz, int D, int H, int W, const inkray *rays, int nr, int K, float R, float step, unsigned long long *key, float *val, unsigned *cnt) {
    const long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (id >= (long long)nr * K) return;
    const float g = dz[id]; if (g == 0) return;
    const inkray r = rays[id / K]; int c[3]; float f[3];
    if (!samp(r, (int)(id % K), R, step, D, H, W, c, f) || c[0] + 1 < z0 || c[0] >= z0 + nz) return;
    for (int q = 0; q < 8; q++) { const int a = c[0] + (q >> 2 & 1); if (a < z0 || a >= z0 + nz) continue;
        const float w = cw(f, q); if (w == 0) continue;
        const unsigned long long vox = ((unsigned long long)(a - z0) * H + c[1] + (q >> 1 & 1)) * W + c[2] + (q & 1);
        const unsigned j = atomicAdd(cnt, 1u); key[j] = vox << 32 | (unsigned long long)(id * 8 + q); val[j] = g * w; }
}
__global__ void k_plane_hist(const float *dz, int D, int H, int W, const inkray *rays, int nr, int K, float R, float step, unsigned *hist) {   /* pairs per corner plane (no w == 0 test: an upper bound) */
    const long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (id >= (long long)nr * K) return;
    if (dz[id] == 0) return;
    const inkray r = rays[id / K]; int c[3]; float f[3];
    if (!samp(r, (int)(id % K), R, step, D, H, W, c, f)) return;
    atomicAdd(&hist[c[0]], 4u); atomicAdd(&hist[c[0] + 1], 4u);
}
__global__ void k_fill_tail(unsigned long long *key, float *val, const unsigned *cnt, unsigned bound, unsigned long long sent, unsigned *ovf) {   /* [count, bound): keys that sort last */
    const unsigned c = *cnt, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!i && c > bound) atomicMax(ovf, c - bound);
    if (i >= c && i < bound) { key[i] = sent; val[i] = 0; }
}
__global__ void k_runs(const unsigned long long *key, const float *val, unsigned n, void *gl, int bf, const unsigned *cnt) {   /* one thread per run start: the run summed in slot order */
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (cnt && *cnt < n) n = *cnt; if (i >= n) return;
    const unsigned long long v = key[i] >> 32; if (i && key[i - 1] >> 32 == v) return;
    float s = 0; for (unsigned j = i; j < n && key[j] >> 32 == v; j++) s += val[j];
    st16(gl, (size_t)v, bf, ld16(gl, (size_t)v, bf) + s);
}
static __thread struct { void *p[8]; size_t cap[8]; } t_ms;
/* the per-plane pair counts ink_mil_fin leaves for the backward slabs (2026-11-13): ink_mil_bwd_slab used to read its slab's pair count back
   with a stream sync per slab (34 per ink window), which under --split drains the GPU while its thread holds the turn and serialises the two
   cards. fin now queues one histogram + its async copy; the first slab waits for it once, every slab sorts an upper bound (its planes'
   counts, one plane of margin each side) with the unused tail filled by keys that sort last, and k_runs reads the true count on the device:
   the same sorted prefix, the same sums. An exceeded bound (never expected) is reported by the next step's first slab. Layout: device
   d[0, cap) the plane counts, d[cap] the overflow word; pinned h[0, cap) the counts, h[cap + 1] the previous step's overflow word. */
static int g_mil_async = -1;
extern "C" void ink_mil_set_async(int on) { g_mil_async = on; }
static __thread struct { unsigned *h, *d; cudaEvent_t ev; const float *dz; const inkray *rays; int D, nr, K, pend, cap; float R, step; } t_ph;
static int mgrow(int i, size_t n) {
    if (t_ms.cap[i] >= n) return 0;
    if (t_ms.p[i]) cudaFree(t_ms.p[i]);
    t_ms.p[i] = nullptr; t_ms.cap[i] = 0; const size_t c = n + n / 4 + 256;
    if (cudaMalloc(&t_ms.p[i], c) != cudaSuccess) return -1;
    t_ms.cap[i] = c; return 0;
}
extern "C" int ink_mil_samples(float R, float step) { return (int)floorf(2 * R / step + 1e-4f) + 1; }
extern "C" int ink_mil_fwd_slab(const void *lg, int bf16, int z0, int nz, int D, int H, int W, const inkray *rays, int nr, float R, float step, float *acc, void *stream) {
    const int K = ink_mil_samples(R, step); if (K > KMAX || K < 1) return -2; if (nr <= 0) return 0;
    const long long n = (long long)nr * K;
    k_fwd_slab<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(lg, bf16, z0, nz, D, H, W, rays, nr, K, R, step, acc);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
extern "C" int ink_mil_fin(float *acc, const inkray *rays, int nr, int D, int H, int W, float R, float step, float tau, float eps, float scale, double *loss, void *stream) {
    const int K = ink_mil_samples(R, step); if (K > KMAX || K < 1) return -2; if (nr <= 0) return 0;
    if (mgrow(0, (size_t)nr * sizeof(double))) return -1;
    k_fin<<<(nr + 127) / 128, 128, 0, (cudaStream_t)stream>>>(acc, rays, nr, K, D, H, W, R, step, tau, eps, scale, (double *)t_ms.p[0]);
    k_sum_fixed<<<1, 256, 0, (cudaStream_t)stream>>>((const double *)t_ms.p[0], nr, loss);
    if (g_mil_async < 0) g_mil_async = !(getenv("UFSM_MIL_ASYNC") && !atoi(getenv("UFSM_MIL_ASYNC")));   /* UFSM_MIL_ASYNC=0: the per-slab readback (the plane bound is the default since test_ink_mil passed byte for byte on the pod) */
    t_ph.dz = nullptr;
    if (g_mil_async) {
        if (t_ph.cap < D + 1) {   /* hist[D + 1] (the last corner plane) + the overflow word */
            if (t_ph.h) cudaFreeHost(t_ph.h); if (t_ph.d) cudaFree(t_ph.d); t_ph.h = nullptr; t_ph.d = nullptr; t_ph.cap = 0;
            if (cudaMallocHost((void **)&t_ph.h, (size_t)(D + 3) * 4) || cudaMalloc((void **)&t_ph.d, (size_t)(D + 3) * 4) || cudaMemset(t_ph.d, 0, (size_t)(D + 3) * 4)) return -1;
            if (!t_ph.ev && cudaEventCreateWithFlags(&t_ph.ev, cudaEventDisableTiming)) return -1;
            t_ph.cap = D + 1;
        }
        cudaStream_t st = (cudaStream_t)stream; const long long n = (long long)nr * K;
        if (cudaMemcpyAsync(t_ph.h + t_ph.cap + 1, t_ph.d + t_ph.cap, 4, cudaMemcpyDeviceToHost, st) || cudaMemsetAsync(t_ph.d, 0, (size_t)(t_ph.cap + 1) * 4, st)) return -1;
        k_plane_hist<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(acc, D, H, W, rays, nr, K, R, step, t_ph.d);
        if (cudaMemcpyAsync(t_ph.h, t_ph.d, (size_t)t_ph.cap * 4, cudaMemcpyDeviceToHost, st) || cudaEventRecord(t_ph.ev, st)) return -1;
        t_ph.pend = 1; t_ph.dz = acc; t_ph.rays = rays; t_ph.D = D; t_ph.nr = nr; t_ph.K = K; t_ph.R = R; t_ph.step = step;
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
extern "C" int ink_mil_bwd_slab(const float *dz, int z0, int nz, int D, int H, int W, const inkray *rays, int nr, float R, float step, void *gl, int bf16, void *stream) {
    const int K = ink_mil_samples(R, step); if (K > KMAX || K < 1) return -2; if (nr <= 0) return 0;
    cudaStream_t st = (cudaStream_t)stream; const long long n = (long long)nr * K; const size_t cap = (size_t)n * 8;
    if (cap > 0xffffffffull || mgrow(1, cap * 8) || mgrow(2, cap * 4) || mgrow(3, cap * 8) || mgrow(4, cap * 4) || mgrow(5, 64)) return -1;
    unsigned *dc = (unsigned *)t_ms.p[5]; unsigned hc = 0;
    const int ph = t_ph.dz == dz && t_ph.rays == rays && t_ph.D == D && t_ph.nr == nr && t_ph.K == K && t_ph.R == R && t_ph.step == step;
    if (ph) {   /* the bound from fin's plane counts (waits for them once per step) */
        if (t_ph.pend) { if (cudaEventSynchronize(t_ph.ev)) return -1; t_ph.pend = 0;
            if (t_ph.h[t_ph.cap + 1]) { fprintf(stderr, "ink mil: a slab's pairs exceeded its plane bound by %u (gradient incomplete)\n", t_ph.h[t_ph.cap + 1]); return -1; } }
        unsigned long long b = 0; for (int a = z0 - 1; a <= z0 + nz; a++) if (a >= 0 && a <= t_ph.cap - 1) b += t_ph.h[a];
        if (!b) return 0;
        hc = (unsigned)(b < cap ? b : cap);
    }
    if (cudaMemsetAsync(dc, 0, 4, st)) return -1;
    k_pairs<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(dz, z0, nz, D, H, W, rays, nr, K, R, step, (unsigned long long *)t_ms.p[1], (float *)t_ms.p[2], dc);
    int bits = 32; while (bits < 64 && ((unsigned long long)nz * H * W) >> (bits - 32)) bits++;   /* key = voxel << 32 | slot */
    if (ph) {   /* slots [count, bound): the largest key in the sorted bits (real keys are below it: slot < cap < 2^32 - 1) */
        const unsigned long long sent = bits >= 64 ? ~0ull : (1ull << bits) - 1;
        k_fill_tail<<<(hc + 255) / 256, 256, 0, st>>>((unsigned long long *)t_ms.p[1], (float *)t_ms.p[2], dc, hc, sent, t_ph.d + t_ph.cap);
    } else {
        if (cudaMemcpyAsync(&hc, dc, 4, cudaMemcpyDeviceToHost, st) || cudaStreamSynchronize(st)) return -1;
        if (!hc) return 0;
    }
    size_t tb = 0; cub::DeviceRadixSort::SortPairs(nullptr, tb, (unsigned long long *)nullptr, (unsigned long long *)nullptr, (float *)nullptr, (float *)nullptr, (int)hc, 0, bits, st);
    if (mgrow(6, tb + 16)) return -1; void *tmp = t_ms.p[6];
    cub::DeviceRadixSort::SortPairs(tmp, tb, (unsigned long long *)t_ms.p[1], (unsigned long long *)t_ms.p[3], (float *)t_ms.p[2], (float *)t_ms.p[4], (int)hc, 0, bits, st);
    k_runs<<<(hc + 255) / 256, 256, 0, st>>>((const unsigned long long *)t_ms.p[3], (const float *)t_ms.p[4], hc, gl, bf16, ph ? dc : nullptr);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
/* tests: the slab pipeline over host arrays (the logit rounded to fp16 / bf16, nslab equal z slabs; grad: the 16-bit gradient as float,
   starting from 0) */
extern "C" int ink_mil_slab_hostmem(const float *logit, int bf16, int D, int H, int W, const inkray *rays, int nr, float R, float step, float tau, float eps, float scale,
                                    int nslab, float *grad, double *loss) {
    const size_t nv = (size_t)D * H * W; const int K = ink_mil_samples(R, step); int rc = -1;
    uint16_t *h = (uint16_t *)malloc(nv * 2); void *dl = nullptr, *dg = nullptr; inkray *dr = nullptr; float *acc = nullptr; double *dL = nullptr;
    for (size_t k = 0; k < nv; k++) { if (bf16) { __nv_bfloat16 b = __float2bfloat16(logit[k]); memcpy(&h[k], &b, 2); } else { __half b = __float2half(logit[k]); memcpy(&h[k], &b, 2); } }
    if (cudaMalloc(&dl, nv * 2) || cudaMalloc(&dg, nv * 2) || cudaMalloc(&dr, (size_t)nr * sizeof *dr) || cudaMalloc(&acc, (size_t)nr * K * 4) || cudaMalloc(&dL, 8)) goto out;
    if (cudaMemcpy(dl, h, nv * 2, cudaMemcpyHostToDevice) || cudaMemcpy(dr, rays, (size_t)nr * sizeof *dr, cudaMemcpyHostToDevice) || cudaMemset(dg, 0, nv * 2) ||
        cudaMemset(acc, 0, (size_t)nr * K * 4) || cudaMemset(dL, 0, 8)) goto out;
    for (int s = 0; s < nslab; s++) { const int z0 = (int)((long long)D * s / nslab), z1 = (int)((long long)D * (s + 1) / nslab);
        if (ink_mil_fwd_slab((const char *)dl + (size_t)z0 * H * W * 2, bf16, z0, z1 - z0, D, H, W, dr, nr, R, step, acc, nullptr)) goto out; }
    if (ink_mil_fin(acc, dr, nr, D, H, W, R, step, tau, eps, scale, dL, nullptr)) goto out;
    for (int s = 0; s < nslab; s++) { const int z0 = (int)((long long)D * s / nslab), z1 = (int)((long long)D * (s + 1) / nslab);
        if (ink_mil_bwd_slab(acc, z0, z1 - z0, D, H, W, dr, nr, R, step, (char *)dg + (size_t)z0 * H * W * 2, bf16, nullptr)) goto out; }
    if (cudaMemcpy(h, dg, nv * 2, cudaMemcpyDeviceToHost) || cudaMemcpy(loss, dL, 8, cudaMemcpyDeviceToHost)) goto out;
    for (size_t k = 0; k < nv; k++) { if (bf16) { __nv_bfloat16 b; memcpy(&b, &h[k], 2); grad[k] = __bfloat162float(b); } else { __half b; memcpy(&b, &h[k], 2); grad[k] = __half2float(b); } }
    rc = 0;
out:
    free(h); cudaFree(dl); cudaFree(dg); cudaFree(dr); cudaFree(acc); cudaFree(dL); return rc;
}
