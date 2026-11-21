/* The surface-aligned ink head (train --inksurf, src/inksurf.c; DESIGN 2026-11-20 "inksurf"): the trunk's level-0 features resampled onto
   a segment's mesh as a flattened patch (K layers along the normal, trilinear), its adjoint as a deterministic per-voxel gather over
   host-built inverse lists, and the small fp32 2-D ops of the head (3x3 / 1x1 convs, SiLU, 2x2 mean pool, nearest 2x up, BCE, AdamW).
   Every reduction runs in a fixed order (no atomics): two runs give the same bits. */
#include "nn_common.cuh"
#include "inksurf.h"

/* ---- resample: out[(c K + j) n + p] = trilinear feat[c] at P_p + (j - (K - 1) / 2) step N_p; corners outside the box contribute 0 ---- */
template <typename FT> __device__ __forceinline__ float sf_ld(const FT *f, size_t i) { return (float)f[i]; }
template <> __device__ __forceinline__ float sf_ld<bf16>(const bf16 *f, size_t i) { return __bfloat162float(f[i]); }
template <> __device__ __forceinline__ float sf_ld<f16>(const f16 *f, size_t i) { return __half2float(f[i]); }
template <typename FT> __global__ void surf_resample_k(const FT *__restrict__ feat, int C, int D, int H, int W, const float *__restrict__ pn, int n, int K,
                                                       float step, float *__restrict__ out) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)n * K) return;
    const int p = (int)(i % n), j = (int)(i / n);
    const float *q = pn + 6 * (size_t)p; const float t = (j - 0.5f * (K - 1)) * step;
    float z, y, x; isf_pos(q, t, &z, &y, &x);
    int iz, iy, ix; float wz, wy, wx; isf_cell(z, &iz, &wz); isf_cell(y, &iy, &wy); isf_cell(x, &ix, &wx);
    const size_t S = (size_t)D * H * W;
    for (int c = 0; c < C; c++) {
        float s = 0.f;
        for (int k = 0; k < 8; k++) {
            const int cz = iz + (k >> 2 & 1), cy = iy + (k >> 1 & 1), cx = ix + (k & 1);
            if (cz < 0 || cy < 0 || cx < 0 || cz >= D || cy >= H || cx >= W) continue;
            const float w = (k >> 2 & 1 ? wz : 1.f - wz) * (k >> 1 & 1 ? wy : 1.f - wy) * (k & 1 ? wx : 1.f - wx);
            s += w * sf_ld(feat, (size_t)c * S + ((size_t)cz * H + cy) * W + cx);
        }
        out[((size_t)c * K + j) * n + p] = s;
    }
}
extern "C" void nn_isf_resample(const void *feat, int ftype, int C, int D, int H, int W, const float *pn, int n, int K, float step, float *out) {
    const size_t m = (size_t)n * K;
    if (ftype == ISF_F32) surf_resample_k<float><<<nblk(m, 128), 128>>>((const float *)feat, C, D, H, W, pn, n, K, step, out);
    else if (ftype == ISF_F16) surf_resample_k<f16><<<nblk(m, 128), 128>>>((const f16 *)feat, C, D, H, W, pn, n, K, step, out);
    else surf_resample_k<bf16><<<nblk(m, 128), 128>>>((const bf16 *)feat, C, D, H, W, pn, n, K, step, out);
    KCHECK();
}
/* ---- the adjoint: one thread per (touched voxel, channel), its entries in list order (the same order every run) ---- */
template <typename GT> __device__ __forceinline__ void sf_st(GT *g, size_t i, float v) { g[i] = (GT)v; }
template <> __device__ __forceinline__ void sf_st<bf16>(bf16 *g, size_t i, float v) { g[i] = __float2bfloat16_rn(v); }
template <> __device__ __forceinline__ void sf_st<f16>(f16 *g, size_t i, float v) { g[i] = __float2half_rn(v); }
template <typename GT> __global__ void surf_gather_k(const float *__restrict__ gout, int C, int K, int n, const int *__restrict__ vox, const int *__restrict__ off,
                                                     const int *__restrict__ ent, const float *__restrict__ ew, int nv, GT *g, size_t S, float scale, const float *dscale, int acc) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)nv * C) return;
    const int u = (int)(i % nv), c = (int)(i / nv);
    float s = 0.f;
    for (int e = off[u]; e < off[u + 1]; e++) { const int smp = ent[e], j = smp / n, p = smp % n; s += ew[e] * gout[((size_t)c * K + j) * n + p]; }
    if (dscale) s *= dscale[0];
    const size_t o = (size_t)c * S + (size_t)(vox ? vox[u] : u);
    sf_st(g, o, acc ? sf_ld(g, o) + scale * s : scale * s);
}
extern "C" void nn_isf_gather(const float *gout, int C, int K, int n, const int *vox, const int *off, const int *ent, const float *ew, int nv,
                               void *g, int gtype, size_t S, float scale, const float *dscale, int acc) {
    const size_t m = (size_t)nv * C;
    if (!m) return;
    if (gtype == ISF_F32) surf_gather_k<float><<<nblk(m, 128), 128>>>(gout, C, K, n, vox, off, ent, ew, nv, (float *)g, S, scale, dscale, acc);
    else if (gtype == ISF_F16) surf_gather_k<f16><<<nblk(m, 128), 128>>>(gout, C, K, n, vox, off, ent, ew, nv, (f16 *)g, S, scale, dscale, acc);
    else surf_gather_k<bf16><<<nblk(m, 128), 128>>>(gout, C, K, n, vox, off, ent, ew, nv, (bf16 *)g, S, scale, dscale, acc);
    KCHECK();
}

/* ---- the compact feature path (train --ink-surf): only the voxels the patch's samples touch ---- */
/* fc[c nv + u] = feat16[c ns + vox[u] - vb] for u in [ua, ub) (the slab's touched voxels) */
__global__ void isf_take_k(const __half *__restrict__ f, int C, size_t ns, size_t vb, const int *__restrict__ vox, int ua, int ub, float *__restrict__ fc, int nv) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, m = (size_t)(ub - ua);
    if (i >= m * C) return;
    const int u = ua + (int)(i % m), c = (int)(i / m);
    fc[(size_t)c * nv + u] = __half2float(f[(size_t)c * ns + ((size_t)vox[u] - vb)]);
}
extern "C" void nn_isf_take(const void *feat16, int C, size_t ns, size_t vb, const int *vox, int ua, int ub, float *fc, int nv) {
    if (ub <= ua) return;
    isf_take_k<<<nblk((size_t)(ub - ua) * C, 256), 256>>>((const __half *)feat16, C, ns, vb, vox, ua, ub, fc, nv); KCHECK();
}
/* rows16[c ns + vox[u] - vb] = gc[c nv + u] (the caller zeroes the rows: untouched voxels have no feature gradient) */
__global__ void isf_put_k(const float *__restrict__ gc, int C, int nv, const int *__restrict__ vox, int ua, int ub, size_t vb, size_t ns, __half *__restrict__ rows) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, m = (size_t)(ub - ua);
    if (i >= m * C) return;
    const int u = ua + (int)(i % m), c = (int)(i / m);
    rows[(size_t)c * ns + ((size_t)vox[u] - vb)] = __float2half_rn(gc[(size_t)c * nv + u]);
}
extern "C" void nn_isf_put(const float *gc, int C, int nv, const int *vox, int ua, int ub, size_t vb, size_t ns, void *rows16) {
    if (ub <= ua) return;
    isf_put_k<<<nblk((size_t)(ub - ua) * C, 256), 256>>>(gc, C, nv, vox, ua, ub, vb, ns, (__half *)rows16); KCHECK();
}
/* out[(c K + j) n + p] = sum_k fw[s 8 + k] fc[c nv + fidx[s 8 + k]], s = j n + p (fidx < 0: none) */
__global__ void isf_resample_c_k(const float *__restrict__ fc, int C, int nv, const int *__restrict__ fidx, const float *__restrict__ fw, int n, int K, float *__restrict__ out) {
    const size_t s = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (s >= (size_t)n * K) return;
    const int j = (int)(s / n), p = (int)(s % n); const int *fi = fidx + 8 * s; const float *w = fw + 8 * s;
    for (int c = 0; c < C; c++) {
        float a = 0.f; const float *fcc = fc + (size_t)c * nv;
        for (int k = 0; k < 8; k++) if (fi[k] >= 0) a += w[k] * fcc[fi[k]];
        out[((size_t)c * K + j) * n + p] = a;
    }
}
extern "C" void nn_isf_resample_c(const float *fc, int C, int nv, const int *fidx, const float *fw, int n, int K, float *out) {
    isf_resample_c_k<<<nblk((size_t)n * K, 128), 128>>>(fc, C, nv, fidx, fw, n, K, out); KCHECK();
}

/* ---- 2-D ops, fp32 [C][H][W], batch 1 ---- */
__global__ void isf2_conv_k(const float *__restrict__ x, int Ci, int H, int W, const float *__restrict__ w, const float *__restrict__ b, int Co, int k, float *__restrict__ y) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, HW = (size_t)H * W;
    if (i >= (size_t)Co * HW) return;
    const int co = (int)(i / HW), py = (int)(i % HW / W), px = (int)(i % W), r = k / 2;
    float s = b ? b[co] : 0.f;
    for (int ci = 0; ci < Ci; ci++) {
        const float *xc = x + (size_t)ci * HW, *wc = w + ((size_t)co * Ci + ci) * k * k;
        for (int dy = 0; dy < k; dy++) { const int yy = py + dy - r; if (yy < 0 || yy >= H) continue;
            for (int dx = 0; dx < k; dx++) { const int xx = px + dx - r; if (xx < 0 || xx >= W) continue; s += wc[dy * k + dx] * xc[(size_t)yy * W + xx]; } }
    }
    y[i] = s;
}
extern "C" void nn_isf2_conv(const float *x, int Ci, int H, int W, const float *w, const float *b, int Co, int k, float *y) {
    isf2_conv_k<<<nblk((size_t)Co * H * W, 128), 128>>>(x, Ci, H, W, w, b, Co, k, y); KCHECK();
}
__global__ void isf2_conv_bd_k(const float *__restrict__ gy, int Co, int H, int W, const float *__restrict__ w, int Ci, int k, float *__restrict__ gx, int acc) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, HW = (size_t)H * W;
    if (i >= (size_t)Ci * HW) return;
    const int ci = (int)(i / HW), py = (int)(i % HW / W), px = (int)(i % W), r = k / 2;
    float s = 0.f;
    for (int co = 0; co < Co; co++) {
        const float *gc = gy + (size_t)co * HW, *wc = w + ((size_t)co * Ci + ci) * k * k;
        for (int dy = 0; dy < k; dy++) { const int yy = py - dy + r; if (yy < 0 || yy >= H) continue;
            for (int dx = 0; dx < k; dx++) { const int xx = px - dx + r; if (xx < 0 || xx >= W) continue; s += wc[dy * k + dx] * gc[(size_t)yy * W + xx]; } }
    }
    gx[i] = acc ? gx[i] + s : s;
}
extern "C" void nn_isf2_conv_bd(const float *gy, int Co, int H, int W, const float *w, int Ci, int k, float *gx, int acc) {
    isf2_conv_bd_k<<<nblk((size_t)Ci * H * W, 128), 128>>>(gy, Co, H, W, w, Ci, k, gx, acc); KCHECK();
}
/* weight gradient: one block per (co, ci, tap), a fixed-order tree over the pixels (256 threads, strided partials) */
__global__ void __launch_bounds__(256) isf2_conv_bw_k(const float *__restrict__ x, int Ci, int H, int W, const float *__restrict__ gy, int Co, int k, float *__restrict__ gw, int acc) {
    __shared__ float sh[256];
    const int kk = k * k, tap = blockIdx.x % kk, ci = blockIdx.x / kk % Ci, co = blockIdx.x / kk / Ci, r = k / 2, dy = tap / k - r, dx = tap % k - r;
    const size_t HW = (size_t)H * W; const float *xc = x + (size_t)ci * HW, *gc = gy + (size_t)co * HW;
    float s = 0.f;
    for (size_t q = threadIdx.x; q < HW; q += 256) { const int py = (int)(q / W), px = (int)(q % W), yy = py + dy, xx = px + dx;
        if (yy >= 0 && yy < H && xx >= 0 && xx < W) s += gc[q] * xc[(size_t)yy * W + xx]; }
    sh[threadIdx.x] = s; __syncthreads();
    for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o]; __syncthreads(); }
    if (!threadIdx.x) gw[((size_t)co * Ci + ci) * kk + tap] = sh[0] + (acc ? gw[((size_t)co * Ci + ci) * kk + tap] : 0.f);
}
__global__ void __launch_bounds__(256) isf2_bias_bw_k(const float *__restrict__ gy, size_t HW, float *__restrict__ gb, int acc) {
    __shared__ float sh[256]; const float *gc = gy + (size_t)blockIdx.x * HW; float s = 0.f;
    for (size_t q = threadIdx.x; q < HW; q += 256) s += gc[q];
    sh[threadIdx.x] = s; __syncthreads();
    for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o]; __syncthreads(); }
    if (!threadIdx.x) gb[blockIdx.x] = sh[0] + (acc ? gb[blockIdx.x] : 0.f);
}
extern "C" void nn_isf2_conv_bw(const float *x, int Ci, int H, int W, const float *gy, int Co, int k, float *gw, float *gb, int acc) {
    isf2_conv_bw_k<<<(unsigned)(Co * Ci * k * k), 256>>>(x, Ci, H, W, gy, Co, k, gw, acc);
    if (gb) isf2_bias_bw_k<<<(unsigned)Co, 256>>>(gy, (size_t)H * W, gb, acc);
    KCHECK();
}
__global__ void isf2_add_k(float *y, const float *x, size_t n) { const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) y[i] += x[i]; }
extern "C" void nn_isf2_add(float *y, const float *x, size_t n) { isf2_add_k<<<nblk(n, 256), 256>>>(y, x, n); KCHECK(); }
__global__ void isf2_silu_k(const float *x, size_t n, float *y) { const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) { const float v = x[i]; y[i] = v / (1.f + expf(-v)); } }
__global__ void isf2_silu_bwd_k(const float *x, const float *gy, size_t n, float *gx) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= n) return;
    const float v = x[i], s = 1.f / (1.f + expf(-v)); gx[i] = gy[i] * s * (1.f + v * (1.f - s));
}
extern "C" void nn_isf2_silu(const float *x, size_t n, float *y) { isf2_silu_k<<<nblk(n, 256), 256>>>(x, n, y); KCHECK(); }
extern "C" void nn_isf2_silu_bwd(const float *x, const float *gy, size_t n, float *gx) { isf2_silu_bwd_k<<<nblk(n, 256), 256>>>(x, gy, n, gx); KCHECK(); }
__global__ void isf2_pool_k(const float *x, int C, int H, int W, float *y) {   /* H, W even: [C][H/2][W/2] means */
    const int h = H / 2, w = W / 2; const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= (size_t)C * h * w) return;
    const int c = (int)(i / ((size_t)h * w)), py = (int)(i / w % h), px = (int)(i % w); const float *a = x + (size_t)c * H * W + (size_t)(2 * py) * W + 2 * px;
    y[i] = 0.25f * (a[0] + a[1] + a[W] + a[W + 1]);
}
__global__ void isf2_pool_bwd_k(const float *gy, int C, int H, int W, float *gx, int acc) {   /* gx [C][H][W] */
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= (size_t)C * H * W) return;
    const int c = (int)(i / ((size_t)H * W)), py = (int)(i / W % H), px = (int)(i % W);
    const float v = 0.25f * gy[(size_t)c * (H / 2) * (W / 2) + (size_t)(py / 2) * (W / 2) + px / 2]; gx[i] = acc ? gx[i] + v : v;
}
extern "C" void nn_isf2_pool(const float *x, int C, int H, int W, float *y) { isf2_pool_k<<<nblk((size_t)C * H * W / 4, 256), 256>>>(x, C, H, W, y); KCHECK(); }
extern "C" void nn_isf2_pool_bwd(const float *gy, int C, int H, int W, float *gx, int acc) { isf2_pool_bwd_k<<<nblk((size_t)C * H * W, 256), 256>>>(gy, C, H, W, gx, acc); KCHECK(); }
__global__ void isf2_up_k(const float *x, int C, int h, int w, float *y) {   /* [C][h][w] -> [C][2h][2w] nearest */
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= (size_t)C * 4 * h * w) return;
    const int c = (int)(i / ((size_t)4 * h * w)), py = (int)(i / (2 * w) % (2 * h)), px = (int)(i % (2 * w));
    y[i] = x[(size_t)c * h * w + (size_t)(py / 2) * w + px / 2];
}
__global__ void isf2_up_bwd_k(const float *gy, int C, int h, int w, float *gx) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= (size_t)C * h * w) return;
    const int c = (int)(i / ((size_t)h * w)), py = (int)(i / w % h), px = (int)(i % w); const float *a = gy + (size_t)c * 4 * h * w + (size_t)(2 * py) * 2 * w + 2 * px;
    gx[i] = (a[0] + a[1]) + (a[2 * w] + a[2 * w + 1]);
}
extern "C" void nn_isf2_up(const float *x, int C, int h, int w, float *y) { isf2_up_k<<<nblk((size_t)C * 4 * h * w, 256), 256>>>(x, C, h, w, y); KCHECK(); }
extern "C" void nn_isf2_up_bwd(const float *gy, int C, int h, int w, float *gx) { isf2_up_bwd_k<<<nblk((size_t)C * h * w, 256), 256>>>(gy, C, h, w, gx); KCHECK(); }
/* weighted BCE with logits on soft targets: g = w (sigmoid(z) - t) / wsum (the loss itself: on the host from the logits) */
__global__ void isf2_bce_grad_k(const float *z, const float *t, const float *wt, size_t n, float iws, float *g) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= n) return;
    g[i] = wt[i] > 0 ? wt[i] * (1.f / (1.f + expf(-z[i])) - t[i]) * iws : 0.f;
}
extern "C" void nn_isf2_bce_grad(const float *z, const float *t, const float *w, size_t n, float inv_wsum, float *g) { isf2_bce_grad_k<<<nblk(n, 256), 256>>>(z, t, w, n, inv_wsum, g); KCHECK(); }
__global__ void isf2_adamw_k(float *p, const float *g, float *m, float *v, size_t n, float lr, float b1, float b2, float eps, float wd, float c1, float c2) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= n) return;
    const float gi = g[i]; m[i] = b1 * m[i] + (1.f - b1) * gi; v[i] = b2 * v[i] + (1.f - b2) * gi * gi;
    p[i] -= lr * ((m[i] * c1) / (sqrtf(v[i] * c2) + eps) + wd * p[i]);
}
extern "C" void nn_isf2_adamw(float *p, const float *g, float *m, float *v, size_t n, float lr, float b1, float b2, float eps, float wd, int step) {
    const float c1 = 1.f / (1.f - powf(b1, (float)step)), c2 = 1.f / (1.f - powf(b2, (float)step));
    isf2_adamw_k<<<nblk(n, 256), 256>>>(p, g, m, v, n, lr, b1, b2, eps, wd, c1, c2); KCHECK();
}
