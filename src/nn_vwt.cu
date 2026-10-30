/* vwt layer kernels (src/vwt.h): REFERENCE fp32 implementations (the parity oracle for the fast paths). Default stream.
   Convs = im2col (channels-last, K order kd kh kw ci) + at_gemm (fp32 cuBLASLt; TF32 / MX per at_set_mx / UFSM_ATTN_TF32), chunked
   over output rows; backward-data = GEMM into column space + atomic col2im. */
#include "vwt.h"
#include "attn.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" size_t vtf_conv_scratch(vt_t x, int co, int k, int sd, int sh, int sw);
extern "C" int vtf_conv_fwd(vt_t x, const float *w, const float *b, int k, int sd, int sh, int sw, vt_t y, void *scratch);
extern "C" int vtf_conv_bwd(vt_t x, vt_t gy, const float *w, int k, int sd, int sh, int sw, vt_t gx, float *gw, float *gb, void *scratch);
static int use_ref(void) { static int v = -1; if (v < 0) { const char *e = getenv("UFSM_VWT_REF"); v = e && atoi(e); } return v; }

static unsigned nb_of(long long n, int t) { long long b = (n + t - 1) / t; return (unsigned)(b > 2147483647LL ? 2147483647LL : b); }
#define GRID(n) nb_of((n), 256), 256

static const size_t CONV_SCRATCH = 256ull << 20;   /* bytes of im2col chunk (x2 for the backward) */
static inline int odim(int n, int s) { return (n + s - 1) / s; }

static __global__ void k_im2col(const float *__restrict__ x, int D, int H, int W, int C, int k, int sd, int sh, int sw, int Do, int Ho, int Wo,
                         long long r0, long long R, float *__restrict__ col) {
    const int K = k * k * k * C; const long long tot = R * K;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < tot; i += (long long)gridDim.x * blockDim.x) {
        const long long r = r0 + i / K; const int q = (int)(i % K), c = q % C, t = q / C, kw = t % k, kh = (t / k) % k, kd = t / (k * k);
        long long rr = r; const int ow = (int)(rr % Wo); rr /= Wo; const int oh = (int)(rr % Ho); rr /= Ho; const int od = (int)(rr % Do); const long long n = rr / Do;
        const int p = k / 2, z = od * sd - p + kd, y = oh * sh - p + kh, xx = ow * sw - p + kw;
        col[i] = (z >= 0 && z < D && y >= 0 && y < H && xx >= 0 && xx < W) ? x[((((n * D + z) * H + y) * W + xx) * (long long)C) + c] : 0.f;
    }
}
static __global__ void k_col2im(const float *__restrict__ col, int D, int H, int W, int C, int k, int sd, int sh, int sw, int Do, int Ho, int Wo,
                         long long r0, long long R, float *__restrict__ gx) {
    const int K = k * k * k * C; const long long tot = R * K;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < tot; i += (long long)gridDim.x * blockDim.x) {
        const long long r = r0 + i / K; const int q = (int)(i % K), c = q % C, t = q / C, kw = t % k, kh = (t / k) % k, kd = t / (k * k);
        long long rr = r; const int ow = (int)(rr % Wo); rr /= Wo; const int oh = (int)(rr % Ho); rr /= Ho; const int od = (int)(rr % Do); const long long n = rr / Do;
        const int p = k / 2, z = od * sd - p + kd, y = oh * sh - p + kh, xx = ow * sw - p + kw;
        if (z >= 0 && z < D && y >= 0 && y < H && xx >= 0 && xx < W) atomicAdd(&gx[((((n * D + z) * H + y) * W + xx) * (long long)C) + c], col[i]);
    }
}
static __global__ void k_addbias(float *__restrict__ y, const float *__restrict__ b, long long rows, int C) {
    const long long n = rows * C;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) y[i] += b[i % C];
}
/* gb[c] += sum over rows of g[r][c]: blocks of 256 threads stride rows, shared partials per channel (C <= 1024) */
static __global__ void k_colsum(const float *__restrict__ g, long long rows, int C, float *__restrict__ gb) {
    extern __shared__ float sh[];
    for (int c = threadIdx.x; c < C; c += blockDim.x) sh[c] = 0.f;
    __syncthreads();
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < rows * C; i += (long long)gridDim.x * blockDim.x) atomicAdd(&sh[i % C], g[i]);
    __syncthreads();
    for (int c = threadIdx.x; c < C; c += blockDim.x) atomicAdd(&gb[c], sh[c]);
}
static void colsum(const float *g, long long rows, int C, float *gb) {
    unsigned nb = nb_of(rows * C, 256); if (nb > 1024) nb = 1024;
    k_colsum<<<nb, 256, C * sizeof(float)>>>(g, rows, C, gb);
}
static long long chunk_rows(int K) { long long r = (long long)(CONV_SCRATCH / (4ull * K)); return r < 1 ? 1 : r; }

extern "C" size_t vt_conv_scratch(vt_t x, int co, int k, int sd, int sh, int sw) { const size_t f = vtf_conv_scratch(x, co, k, sd, sh, sw); return f > CONV_SCRATCH ? f : CONV_SCRATCH; }

extern "C" int vtr_conv_fwd(vt_t x, const float *w, const float *b, int k, int sd, int sh, int sw, vt_t y, void *scratch);
extern "C" int vtr_conv_bwd(vt_t x, vt_t gy, const float *w, int k, int sd, int sh, int sw, vt_t gx, float *gw, float *gb, void *scratch);
/* dispatch: the fast bf16 path (src/nn_vwt_fast.cu, PF) unless UFSM_VWT_REF=1 */
extern "C" int vt_conv_fwd(vt_t x, const float *w, const float *b, int k, int sd, int sh, int sw, vt_t y, void *scratch) {
    return use_ref() ? vtr_conv_fwd(x, w, b, k, sd, sh, sw, y, scratch) : vtf_conv_fwd(x, w, b, k, sd, sh, sw, y, scratch);
}
extern "C" int vt_conv_bwd(vt_t x, vt_t gy, const float *w, int k, int sd, int sh, int sw, vt_t gx, float *gw, float *gb, void *scratch) {
    return use_ref() ? vtr_conv_bwd(x, gy, w, k, sd, sh, sw, gx, gw, gb, scratch) : vtf_conv_bwd(x, gy, w, k, sd, sh, sw, gx, gw, gb, scratch);
}
/* the fp32 reference (the parity oracle) */
extern "C" int vtr_conv_fwd(vt_t x, const float *w, const float *b, int k, int sd, int sh, int sw, vt_t y, void *scratch) {
    const int Do = odim(x.d, sd), Ho = odim(x.h, sh), Wo = odim(x.w, sw), K = k * k * k * x.c, co = y.c;
    if (y.d != Do || y.h != Ho || y.w != Wo || y.n != x.n) { fprintf(stderr, "vt_conv_fwd: output shape mismatch\n"); return -1; }
    const long long rows = (long long)x.n * Do * Ho * Wo;
    if (k == 1 && sd == 1 && sh == 1 && sw == 1) { if (at_gemm(0, 1, (int)rows, co, K, x.p, w, y.p, 0.f)) return -1; }
    else {
        const long long cr = chunk_rows(K); float *col = (float *)scratch;
        for (long long r0 = 0; r0 < rows; r0 += cr) {
            const long long R = rows - r0 < cr ? rows - r0 : cr;
            k_im2col<<<GRID(R * K)>>>(x.p, x.d, x.h, x.w, x.c, k, sd, sh, sw, Do, Ho, Wo, r0, R, col);
            if (at_gemm(0, 1, (int)R, co, K, col, w, y.p + r0 * co, 0.f)) return -1;
        }
    }
    if (b) k_addbias<<<GRID(rows * co)>>>(y.p, b, rows, co);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int vtr_conv_bwd(vt_t x, vt_t gy, const float *w, int k, int sd, int sh, int sw, vt_t gx, float *gw, float *gb, void *scratch) {
    const int Do = gy.d, Ho = gy.h, Wo = gy.w, K = k * k * k * x.c, co = gy.c;
    const long long rows = (long long)x.n * Do * Ho * Wo;
    if (gb) colsum(gy.p, rows, co, gb);
    const int direct = k == 1 && sd == 1 && sh == 1 && sw == 1;
    if (direct) {
        if (gw && at_gemm(1, 0, co, K, (int)rows, gy.p, x.p, gw, 1.f)) return -1;
        if (gx.p && at_gemm(0, 0, (int)rows, K, co, gy.p, w, gx.p, 0.f)) return -1;
        return cudaGetLastError() == cudaSuccess ? 0 : -1;
    }
    if (gx.p) cudaMemsetAsync(gx.p, 0, vt_numel(gx) * 4);
    const long long cr = chunk_rows(K); float *col = (float *)scratch;
    for (long long r0 = 0; r0 < rows; r0 += cr) {
        const long long R = rows - r0 < cr ? rows - r0 : cr;
        if (gw) {
            k_im2col<<<GRID(R * K)>>>(x.p, x.d, x.h, x.w, x.c, k, sd, sh, sw, Do, Ho, Wo, r0, R, col);
            if (at_gemm(1, 0, co, K, (int)R, gy.p + r0 * co, col, gw, 1.f)) return -1;
        }
        if (gx.p) {
            if (at_gemm(0, 0, (int)R, K, co, gy.p + r0 * co, w, col, 0.f)) return -1;
            k_col2im<<<GRID(R * K)>>>(col, x.d, x.h, x.w, x.c, k, sd, sh, sw, Do, Ho, Wo, r0, R, gx.p);
        }
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/* ---- LayerNorm (+ GELU), one warp per row, C <= 256 ---- */
__device__ __forceinline__ float gelu_f(float z) { return 0.5f * z * (1.f + erff(z * 0.70710678118654752f)); }
__device__ __forceinline__ float gelu_d(float z) { return 0.5f * (1.f + erff(z * 0.70710678118654752f)) + z * 0.3989422804014327f * expf(-0.5f * z * z); }
#define LNV 8
static __global__ void k_ln_fwd(const float *__restrict__ x, const float *__restrict__ g, const float *__restrict__ b, int gelu, float *__restrict__ y,
                         float *__restrict__ mean, float *__restrict__ rstd, long long M, int C) {
    const long long r = blockIdx.x * (long long)(blockDim.x / 32) + threadIdx.x / 32; if (r >= M) return;
    const int l = threadIdx.x & 31; float v[LNV]; float s = 0;
#pragma unroll
    for (int q = 0; q < LNV; q++) { const int c = l + 32 * q; v[q] = c < C ? x[r * C + c] : 0.f; s += v[q]; }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float mu = s / C; float qq = 0;
#pragma unroll
    for (int q = 0; q < LNV; q++) { const int c = l + 32 * q; if (c < C) { const float d = v[q] - mu; qq += d * d; } }
    for (int o = 16; o; o >>= 1) qq += __shfl_xor_sync(0xffffffffu, qq, o);
    const float rs = rsqrtf(qq / C + 1e-5f);
    if (l == 0) { mean[r] = mu; rstd[r] = rs; }
#pragma unroll
    for (int q = 0; q < LNV; q++) { const int c = l + 32 * q; if (c >= C) continue; float z = (v[q] - mu) * rs * g[c] + b[c]; y[r * C + c] = gelu ? gelu_f(z) : z; }
}
/* even C <= 256: lanes over channel pairs 2 lane + 64 q (float2), NQ = ceil(C / 64) */
template <int NQ>
static __global__ void k_ln_fwd2(const float *__restrict__ x, const float *__restrict__ g, const float *__restrict__ b, int gelu, float *__restrict__ y,
                                 float *__restrict__ mean, float *__restrict__ rstd, long long M, int C) {
    const long long r = blockIdx.x * (long long)(blockDim.x / 32) + threadIdx.x / 32; if (r >= M) return;
    const int l = threadIdx.x & 31; float2 v[NQ]; float s = 0;
#pragma unroll
    for (int q = 0; q < NQ; q++) { const int c = 2 * l + 64 * q; v[q] = c < C ? *(const float2 *)(x + r * C + c) : make_float2(0.f, 0.f); s += v[q].x + v[q].y; }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float mu = s / C; float qq = 0;
#pragma unroll
    for (int q = 0; q < NQ; q++) { const int c = 2 * l + 64 * q; if (c < C) { const float d0 = v[q].x - mu, d1 = v[q].y - mu; qq += d0 * d0 + d1 * d1; } }
    for (int o = 16; o; o >>= 1) qq += __shfl_xor_sync(0xffffffffu, qq, o);
    const float rs = rsqrtf(qq / C + 1e-5f);
    if (l == 0) { mean[r] = mu; rstd[r] = rs; }
#pragma unroll
    for (int q = 0; q < NQ; q++) {
        const int c = 2 * l + 64 * q; if (c >= C) continue;
        const float2 gg = *(const float2 *)(g + c), bb = *(const float2 *)(b + c);
        float z0 = (v[q].x - mu) * rs * gg.x + bb.x, z1 = (v[q].y - mu) * rs * gg.y + bb.y;
        if (gelu) { z0 = gelu_f(z0); z1 = gelu_f(z1); }
        *(float2 *)(y + r * C + c) = make_float2(z0, z1);
    }
}
template <int NQ>
static __global__ void k_ln_bwd2(const float *__restrict__ x, const float *__restrict__ g, const float *__restrict__ b, int gelu, const float *__restrict__ mean,
                                 const float *__restrict__ rstd, const float *__restrict__ gy, float *__restrict__ gx, float *__restrict__ gg, float *__restrict__ gbb, long long M, int C) {
    __shared__ float sg[256], sb[256];
    for (int c = threadIdx.x; c < C; c += blockDim.x) { sg[c] = 0.f; sb[c] = 0.f; }
    __syncthreads();
    const int l = threadIdx.x & 31, wpb = blockDim.x / 32; float2 ag[NQ], ab[NQ], gq[NQ], bq[NQ];
#pragma unroll
    for (int q = 0; q < NQ; q++) {
        const int c = 2 * l + 64 * q; ag[q] = ab[q] = make_float2(0.f, 0.f);
        gq[q] = c < C ? *(const float2 *)(g + c) : make_float2(0.f, 0.f); bq[q] = c < C ? *(const float2 *)(b + c) : make_float2(0.f, 0.f);
    }
    for (long long r = blockIdx.x * (long long)wpb + threadIdx.x / 32; r < M; r += (long long)gridDim.x * wpb) {
        const float mu = mean[r], rs = rstd[r]; float2 xh[NQ], dxh[NQ]; float s1 = 0, s2 = 0;
#pragma unroll
        for (int q = 0; q < NQ; q++) {
            const int c = 2 * l + 64 * q; xh[q] = dxh[q] = make_float2(0.f, 0.f); if (c >= C) continue;
            const float2 xv = *(const float2 *)(x + r * C + c); float2 dz = *(const float2 *)(gy + r * C + c);
            xh[q] = make_float2((xv.x - mu) * rs, (xv.y - mu) * rs);
            if (gelu) { dz.x *= gelu_d(xh[q].x * gq[q].x + bq[q].x); dz.y *= gelu_d(xh[q].y * gq[q].y + bq[q].y); }
            ag[q].x += dz.x * xh[q].x; ag[q].y += dz.y * xh[q].y; ab[q].x += dz.x; ab[q].y += dz.y;
            dxh[q] = make_float2(dz.x * gq[q].x, dz.y * gq[q].y); s1 += dxh[q].x + dxh[q].y; s2 += dxh[q].x * xh[q].x + dxh[q].y * xh[q].y;
        }
        for (int o = 16; o; o >>= 1) { s1 += __shfl_xor_sync(0xffffffffu, s1, o); s2 += __shfl_xor_sync(0xffffffffu, s2, o); }
        s1 /= C; s2 /= C;
#pragma unroll
        for (int q = 0; q < NQ; q++) { const int c = 2 * l + 64 * q; if (c < C) *(float2 *)(gx + r * C + c) = make_float2(rs * (dxh[q].x - s1 - xh[q].x * s2), rs * (dxh[q].y - s1 - xh[q].y * s2)); }
    }
#pragma unroll
    for (int q = 0; q < NQ; q++) { const int c = 2 * l + 64 * q; if (c < C) { atomicAdd(&sg[c], ag[q].x); atomicAdd(&sg[c + 1], ag[q].y); atomicAdd(&sb[c], ab[q].x); atomicAdd(&sb[c + 1], ab[q].y); } }
    __syncthreads();
    for (int c = threadIdx.x; c < C; c += blockDim.x) { if (gg) atomicAdd(&gg[c], sg[c]); if (gbb) atomicAdd(&gbb[c], sb[c]); }
}
extern "C" void vt_ln_fwd(vt_t x, const float *g, const float *b, int gelu, vt_t y, float *mean, float *rstd) {
    const long long M = (long long)vt_rows(x); if (x.c > 32 * LNV) { fprintf(stderr, "vt_ln_fwd: C %d > %d\n", x.c, 32 * LNV); return; }
    if (x.c % 2 == 0) {
        const int nq = (x.c + 63) / 64;
        if (nq == 1) k_ln_fwd2<1><<<nb_of(M, 8), 256>>>(x.p, g, b, gelu, y.p, mean, rstd, M, x.c);
        else if (nq == 2) k_ln_fwd2<2><<<nb_of(M, 8), 256>>>(x.p, g, b, gelu, y.p, mean, rstd, M, x.c);
        else if (nq == 3) k_ln_fwd2<3><<<nb_of(M, 8), 256>>>(x.p, g, b, gelu, y.p, mean, rstd, M, x.c);
        else k_ln_fwd2<4><<<nb_of(M, 8), 256>>>(x.p, g, b, gelu, y.p, mean, rstd, M, x.c);
        return;
    }
    k_ln_fwd<<<nb_of(M, 8), 256>>>(x.p, g, b, gelu, y.p, mean, rstd, M, x.c);
}
/* backward: per row dz = dy gelu'(z), dxh = dz g, dx = rstd (dxh - mean(dxh) - xh mean(dxh xh)); gg += dz xh, gb += dz via per-block
   shared partials (each block of 8 warps loops over rows) */
static __global__ void k_ln_bwd(const float *__restrict__ x, const float *__restrict__ g, const float *__restrict__ b, int gelu, const float *__restrict__ mean,
                         const float *__restrict__ rstd, const float *__restrict__ gy, float *__restrict__ gx, float *__restrict__ gg, float *__restrict__ gbb, long long M, int C) {
    __shared__ float sg[256], sb[256];
    for (int c = threadIdx.x; c < C; c += blockDim.x) { sg[c] = 0.f; sb[c] = 0.f; }
    __syncthreads();
    const int l = threadIdx.x & 31, wpb = blockDim.x / 32; float ag[LNV], ab[LNV];
#pragma unroll
    for (int q = 0; q < LNV; q++) { ag[q] = 0.f; ab[q] = 0.f; }
    for (long long r = blockIdx.x * (long long)wpb + threadIdx.x / 32; r < M; r += (long long)gridDim.x * wpb) {
        const float mu = mean[r], rs = rstd[r]; float xh[LNV], dxh[LNV], s1 = 0, s2 = 0;
#pragma unroll
        for (int q = 0; q < LNV; q++) {
            const int c = l + 32 * q; xh[q] = dxh[q] = 0.f; if (c >= C) continue;
            xh[q] = (x[r * C + c] - mu) * rs; float dz = gy[r * C + c];
            if (gelu) dz *= gelu_d(xh[q] * g[c] + b[c]);
            ag[q] += dz * xh[q]; ab[q] += dz; dxh[q] = dz * g[c]; s1 += dxh[q]; s2 += dxh[q] * xh[q];
        }
        for (int o = 16; o; o >>= 1) { s1 += __shfl_xor_sync(0xffffffffu, s1, o); s2 += __shfl_xor_sync(0xffffffffu, s2, o); }
        s1 /= C; s2 /= C;
#pragma unroll
        for (int q = 0; q < LNV; q++) { const int c = l + 32 * q; if (c < C) gx[r * C + c] = rs * (dxh[q] - s1 - xh[q] * s2); }
    }
#pragma unroll
    for (int q = 0; q < LNV; q++) { const int c = l + 32 * q; if (c < C) { atomicAdd(&sg[c], ag[q]); atomicAdd(&sb[c], ab[q]); } }
    __syncthreads();
    for (int c = threadIdx.x; c < C; c += blockDim.x) { if (gg) atomicAdd(&gg[c], sg[c]); if (gbb) atomicAdd(&gbb[c], sb[c]); }
}
extern "C" void vt_ln_bwd(vt_t x, const float *g, const float *b, int gelu, const float *mean, const float *rstd, vt_t gy, vt_t gx, float *gg, float *gb) {
    const long long M = (long long)vt_rows(x); unsigned nb = nb_of(M, 8); if (nb > 2048) nb = 2048;
    if (x.c % 2 == 0 && x.c <= 256) {
        const int nq = (x.c + 63) / 64;
        if (nq == 1) k_ln_bwd2<1><<<nb, 256>>>(x.p, g, b, gelu, mean, rstd, gy.p, gx.p, gg, gb, M, x.c);
        else if (nq == 2) k_ln_bwd2<2><<<nb, 256>>>(x.p, g, b, gelu, mean, rstd, gy.p, gx.p, gg, gb, M, x.c);
        else if (nq == 3) k_ln_bwd2<3><<<nb, 256>>>(x.p, g, b, gelu, mean, rstd, gy.p, gx.p, gg, gb, M, x.c);
        else k_ln_bwd2<4><<<nb, 256>>>(x.p, g, b, gelu, mean, rstd, gy.p, gx.p, gg, gb, M, x.c);
        return;
    }
    k_ln_bwd<<<nb, 256>>>(x.p, g, b, gelu, mean, rstd, gy.p, gx.p, gg, gb, M, x.c);
}

/* ---- x2 transverse upsample (align_corners False) ---- */
__device__ __forceinline__ void up_src(int o, int n, int *i0, int *i1, float *f) {
    float s = (o + 0.5f) * 0.5f - 0.5f; if (s < 0.f) s = 0.f;
    int a = (int)s; if (a > n - 1) a = n - 1; *i0 = a; *i1 = a + 1 < n ? a + 1 : n - 1; *f = s - a;
}
static __global__ void k_up2(const float *__restrict__ x, float *__restrict__ y, int N, int D, int H, int W, int C, int bwd) {
    const long long tot = (long long)N * 2 * D * 2 * H * W * C;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < tot; i += (long long)gridDim.x * blockDim.x) {
        long long t = i; const int c = (int)(t % C); t /= C; const int w = (int)(t % W); t /= W; const int oh = (int)(t % (2 * H)); t /= 2 * H; const int od = (int)(t % (2 * D)); const long long n = t / (2 * D);
        int d0, d1, h0, h1; float fd, fh; up_src(od, D, &d0, &d1, &fd); up_src(oh, H, &h0, &h1, &fh);
        const long long base = n * D;
#define XI(dd, hh) ((((base + (dd)) * H + (hh)) * W + w) * (long long)C + c)
        if (!bwd) y[i] = (1 - fd) * ((1 - fh) * x[XI(d0, h0)] + fh * x[XI(d0, h1)]) + fd * ((1 - fh) * x[XI(d1, h0)] + fh * x[XI(d1, h1)]);
        else { const float gv = y[i];   /* y = gy, x = gx (accumulated) */
            atomicAdd((float *)&x[XI(d0, h0)], (1 - fd) * (1 - fh) * gv); atomicAdd((float *)&x[XI(d0, h1)], (1 - fd) * fh * gv);
            atomicAdd((float *)&x[XI(d1, h0)], fd * (1 - fh) * gv); atomicAdd((float *)&x[XI(d1, h1)], fd * fh * gv); }
#undef XI
    }
}
/* C % 4 == 0: 4 channels per thread; the backward as a gather over the (<= 3 x 3 contributing) outputs of each input row, no atomics */
static __global__ void k_up2_f4(const float4 *__restrict__ x, float4 *__restrict__ y, int N, int D, int H, int W, int C4) {
    const long long tot = (long long)N * 2 * D * 2 * H * W * C4;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < tot; i += (long long)gridDim.x * blockDim.x) {
        long long t = i; const int c = (int)(t % C4); t /= C4; const int w = (int)(t % W); t /= W; const int oh = (int)(t % (2 * H)); t /= 2 * H; const int od = (int)(t % (2 * D)); const long long n = t / (2 * D);
        int d0, d1, h0, h1; float fd, fh; up_src(od, D, &d0, &d1, &fd); up_src(oh, H, &h0, &h1, &fh);
#define XI(dd, hh) ((((n * D + (dd)) * H + (hh)) * W + w) * (long long)C4 + c)
        const float4 a = x[XI(d0, h0)], b = x[XI(d0, h1)], e = x[XI(d1, h0)], f = x[XI(d1, h1)];
        const float w00 = (1 - fd) * (1 - fh), w01 = (1 - fd) * fh, w10 = fd * (1 - fh), w11 = fd * fh;
        y[i] = make_float4(w00 * a.x + w01 * b.x + w10 * e.x + w11 * f.x, w00 * a.y + w01 * b.y + w10 * e.y + w11 * f.y,
                           w00 * a.z + w01 * b.z + w10 * e.z + w11 * f.z, w00 * a.w + w01 * b.w + w10 * e.w + w11 * f.w);
#undef XI
    }
}
/* the weight of input i in output o (one axis, n inputs) */
__device__ __forceinline__ float up_w(int o, int n, int i) { int a, b; float f; up_src(o, n, &a, &b, &f); return (a == i ? 1.f - f : 0.f) + (b == i ? f : 0.f); }
static __global__ void k_up2_bwd_f4(float4 *__restrict__ gx, const float4 *__restrict__ gy, int N, int D, int H, int W, int C4) {
    const long long tot = (long long)N * D * H * W * C4;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < tot; i += (long long)gridDim.x * blockDim.x) {
        long long t = i; const int c = (int)(t % C4); t /= C4; const int w = (int)(t % W); t /= W; const int h = (int)(t % H); t /= H; const int d = (int)(t % D); const long long n = t / D;
        float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
        const int oz0 = 2 * d - 2 > 0 ? 2 * d - 2 : 0, oz1 = 2 * d + 3 < 2 * D - 1 ? 2 * d + 3 : 2 * D - 1;
        const int oy0 = 2 * h - 2 > 0 ? 2 * h - 2 : 0, oy1 = 2 * h + 3 < 2 * H - 1 ? 2 * h + 3 : 2 * H - 1;
        for (int oz = oz0; oz <= oz1; oz++) {
            const float wz = up_w(oz, D, d); if (wz == 0.f) continue;
            for (int oy = oy0; oy <= oy1; oy++) {
                const float wy = up_w(oy, H, h); if (wy == 0.f) continue;
                const float ww = wz * wy; const float4 g = gy[(((n * 2 * D + oz) * 2 * H + oy) * W + w) * (long long)C4 + c];
                s.x += ww * g.x; s.y += ww * g.y; s.z += ww * g.z; s.w += ww * g.w;
            }
        }
        gx[i] = s;
    }
}
extern "C" void vt_up2_fwd(vt_t x, vt_t y) {
    if (x.c % 4 == 0) { k_up2_f4<<<GRID((long long)vt_numel(y) / 4)>>>((const float4 *)x.p, (float4 *)y.p, x.n, x.d, x.h, x.w, x.c / 4); return; }
    k_up2<<<GRID((long long)vt_numel(y))>>>(x.p, y.p, x.n, x.d, x.h, x.w, x.c, 0);
}
extern "C" void vt_up2_bwd(vt_t gy, vt_t gx) {
    if (gx.c % 4 == 0) { k_up2_bwd_f4<<<GRID((long long)vt_numel(gx) / 4)>>>((float4 *)gx.p, (const float4 *)gy.p, gx.n, gx.d, gx.h, gx.w, gx.c / 4); return; }
    cudaMemsetAsync(gx.p, 0, vt_numel(gx) * 4); k_up2<<<GRID((long long)vt_numel(gy))>>>(gx.p, gy.p, gx.n, gx.d, gx.h, gx.w, gx.c, 1);
}

extern "C" void vt_cat_fwd(vt_t a, vt_t b, vt_t y) {
    const size_t rows = vt_rows(a);
    cudaMemcpy2DAsync(y.p, (size_t)y.c * 4, a.p, (size_t)a.c * 4, (size_t)a.c * 4, rows, cudaMemcpyDeviceToDevice);
    cudaMemcpy2DAsync(y.p + a.c, (size_t)y.c * 4, b.p, (size_t)b.c * 4, (size_t)b.c * 4, rows, cudaMemcpyDeviceToDevice);
}
extern "C" void vt_cat_bwd(vt_t gy, vt_t ga, vt_t gb) {
    const size_t rows = vt_rows(gy);
    if (ga.p) cudaMemcpy2DAsync(ga.p, (size_t)ga.c * 4, gy.p, (size_t)gy.c * 4, (size_t)ga.c * 4, rows, cudaMemcpyDeviceToDevice);
    if (gb.p) cudaMemcpy2DAsync(gb.p, (size_t)gb.c * 4, gy.p + ga.c, (size_t)gy.c * 4, (size_t)gb.c * 4, rows, cudaMemcpyDeviceToDevice);
}
static __global__ void k_add(float *__restrict__ y, const float *__restrict__ x, long long n) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) y[i] += x[i];
}
extern "C" void vt_add(float *y, const float *x, size_t n) { k_add<<<GRID((long long)n)>>>(y, x, (long long)n); }

/* ---- phase head ---- */
__device__ __forceinline__ float softplus_f(float x) { return x > 20.f ? x : log1pf(expf(x)); }
static __global__ void k_phase(const float *__restrict__ lg, int N, int D, int H, int L, float *__restrict__ inc, float *__restrict__ ph) {
    const long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (t >= (long long)N * 4 * D * H) return;   /* output column */
    const int T2 = 2 * H; const int oj = (int)(t % T2); long long r = t / T2; const int oi = (int)(r % (2 * D)); const long long n = r / (2 * D);
    const int d = oi >> 1, si = oi & 1, h = oj >> 1, sj = oj & 1, ch = si * 2 + sj;
    const float *s = lg + (((n * D + d) * H + h) * (long long)L) * 4 + ch; float a = 0;
    for (int k = 0; k < L; k++) { const float v = softplus_f(s[(long long)k * 4]); a += v; inc[t * L + k] = v; ph[t * L + k] = a; }
}
extern "C" void vt_phase_fwd(vt_t lg, float *inc, float *ph) { const long long nc = (long long)lg.n * 4 * lg.d * lg.h; k_phase<<<nb_of(nc, 128), 128>>>(lg.p, lg.n, lg.d, lg.h, lg.w, inc, ph); }
static __global__ void k_phase_bwd(const float *__restrict__ lg, int N, int D, int H, int L, const float *__restrict__ gph, const float *__restrict__ ginc, float *__restrict__ glg) {
    const long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (t >= (long long)N * 4 * D * H) return;
    const int T2 = 2 * H; const int oj = (int)(t % T2); long long r = t / T2; const int oi = (int)(r % (2 * D)); const long long n = r / (2 * D);
    const int d = oi >> 1, si = oi & 1, h = oj >> 1, sj = oj & 1, ch = si * 2 + sj;
    const long long o = (((n * D + d) * H + h) * (long long)L) * 4 + ch; float acc = 0;
    for (int k = L - 1; k >= 0; k--) {
        if (gph) acc += gph[t * L + k];
        const float gi = acc + (ginc ? ginc[t * L + k] : 0.f), x = lg[o + (long long)k * 4];
        glg[o + (long long)k * 4] = gi / (1.f + expf(-x));
    }
}
extern "C" void vt_phase_bwd(vt_t lg, const float *gph, const float *ginc, vt_t glg) {
    const long long nc = (long long)lg.n * 4 * lg.d * lg.h; k_phase_bwd<<<nb_of(nc, 128), 128>>>(lg.p, lg.n, lg.d, lg.h, lg.w, gph, ginc, glg.p);
}

/* ---- losses ---- */
__device__ __forceinline__ float huber(float r, float d) { const float a = fabsf(r); return a <= d ? 0.5f * r * r : d * (a - 0.5f * d); }
__device__ __forceinline__ float huber_d(float r, float d) { return r > d ? d : r < -d ? -d : r; }
__device__ __forceinline__ float wgt(const float *pw, const uint16_t *comp, long long i) { return comp[i] ? (pw ? pw[i] : 1.f) : 0.f; }
/* pass 1: per component sums w, w ph, w tg; density sums */
static __global__ void k_loss1(const float *__restrict__ ph, const float *__restrict__ tg, const uint16_t *__restrict__ comp, const float *__restrict__ pw, long long n,
                        double *__restrict__ cs /* [65536][3] */) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const float w = wgt(pw, comp, i); if (w <= 0.f) continue; const int c = comp[i];
        atomicAdd(&cs[c * 3], (double)w); atomicAdd(&cs[c * 3 + 1], (double)w * ph[i]); atomicAdd(&cs[c * 3 + 2], (double)w * tg[i]);
    }
}
/* pass 2: residuals, Huber, sum w psi per component; density loss and gradient (ginc) */
static __global__ void k_loss2(const float *__restrict__ ph, const float *__restrict__ inc, const float *__restrict__ tg, const uint16_t *__restrict__ comp,
                        const float *__restrict__ pw, long long n, int L, const double *__restrict__ cs, float dp, float dd, double *__restrict__ acc /* [4] */,
                        double *__restrict__ cpsi /* [65536] */, float *__restrict__ psi, float *__restrict__ ginc, float dscale) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const float w = wgt(pw, comp, i); psi[i] = 0.f; if (ginc) ginc[i] = 0.f;
        if (w > 0.f) {
            const int c = comp[i]; const double W = cs[c * 3];
            const float r = (float)((ph[i] - cs[c * 3 + 1] / W) - (tg[i] - cs[c * 3 + 2] / W));
            atomicAdd(&acc[0], (double)w * huber(r, dp)); const float p = w * huber_d(r, dp); psi[i] = p; atomicAdd(&cpsi[c], (double)p);
        }
        const int k = (int)(i % L);
        if (k > 0 && comp[i] && comp[i] == comp[i - 1]) {
            const float wd = fminf(w, wgt(pw, comp, i - 1)); if (wd <= 0.f) continue;
            const float r = inc[i] - (tg[i] - tg[i - 1]);
            atomicAdd(&acc[1], (double)wd * huber(r, dd)); atomicAdd(&acc[2], (double)wd);
            if (ginc) ginc[i] = wd * huber_d(r, dd) * dscale;
        }
    }
}
/* pass 3: gph = (psi - w / W_c sum_c psi) pscale */
static __global__ void k_loss3(const uint16_t *__restrict__ comp, const float *__restrict__ pw, long long n, const double *__restrict__ cs, const double *__restrict__ cpsi,
                        const float *__restrict__ psi, float *__restrict__ gph, float pscale) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const float w = wgt(pw, comp, i);
        gph[i] = w > 0.f ? (float)((psi[i] - (double)w / cs[comp[i] * 3] * cpsi[comp[i]]) * pscale) : 0.f;
    }
}
static __global__ void k_scale(float *x, long long n, float s) { for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) x[i] *= s; }

extern "C" int vt_losses(const float *ph, const float *inc, const float *tg, const uint16_t *comp, const float *pw, int T, int L, float delta_p, float delta_d,
                         float lam_p, float lam_d, float scale, float *gph, float *ginc, vt_loss *out) {
    static double *cs = nullptr, *cpsi = nullptr, *acc = nullptr; static float *psi = nullptr; static long long psin = 0;
    const long long n = (long long)T * T * L;
    if (!cs) { cudaMalloc(&cs, 65536 * 3 * 8); cudaMalloc(&cpsi, 65536 * 8); cudaMalloc(&acc, 4 * 8); }
    if (psin < n) { cudaFree(psi); cudaMalloc(&psi, n * 4); psin = n; }
    cudaMemsetAsync(cs, 0, 65536 * 3 * 8); cudaMemsetAsync(cpsi, 0, 65536 * 8); cudaMemsetAsync(acc, 0, 4 * 8);
    k_loss1<<<GRID(n)>>>(ph, tg, comp, pw, n, cs);
    /* the weight totals first (pass 2 needs them only for the scale; computed from cs on the host) */
    double *hcs = (double *)malloc(65536 * 3 * 8); cudaMemcpy(hcs, cs, 65536 * 3 * 8, cudaMemcpyDeviceToHost);
    double wtot = 0; for (int c = 1; c < 65536; c++) wtot += hcs[c * 3]; free(hcs);
    k_loss2<<<GRID(n)>>>(ph, inc, tg, comp, pw, n, L, cs, delta_p, delta_d, acc, cpsi, psi, ginc, 1.f);
    double h[4]; cudaMemcpy(h, acc, 4 * 8, cudaMemcpyDeviceToHost);
    out->wsum = wtot; out->dsum = h[2];
    out->phase = wtot > 0 ? h[0] / wtot : 0; out->density = h[2] > 0 ? h[1] / h[2] : 0;
    if (gph) k_loss3<<<GRID(n)>>>(comp, pw, n, cs, cpsi, psi, gph, wtot > 0 ? (float)(lam_p * scale / wtot) : 0.f);
    if (ginc) k_scale<<<GRID(n)>>>(ginc, n, h[2] > 0 ? (float)(lam_d * scale / h[2]) : 0.f);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/* ---- input: u8 image / valid -> [n rows][2] (image standardised over the valid voxels, population std + 1e-6; valid) ---- */
static __global__ void k_in_sums(const uint8_t *__restrict__ im, const uint8_t *__restrict__ va, long long n, double *__restrict__ s) {
    double a = 0, b = 0, c = 0;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) if (va[i]) { const double v = im[i]; a += v; b += v * v; c += 1; }
    for (int o = 16; o; o >>= 1) { a += __shfl_xor_sync(0xffffffffu, a, o); b += __shfl_xor_sync(0xffffffffu, b, o); c += __shfl_xor_sync(0xffffffffu, c, o); }
    if ((threadIdx.x & 31) == 0) { atomicAdd(&s[0], a); atomicAdd(&s[1], b); atomicAdd(&s[2], c); }
}
static __global__ void k_in(const uint8_t *__restrict__ im, const uint8_t *__restrict__ va, long long n, const double *__restrict__ s, float *__restrict__ x) {
    const double c = s[2] > 0 ? s[2] : 1, m = s[0] / c, var = s[1] / c - m * m; const float inv = (float)(1.0 / (sqrt(var > 0 ? var : 0) + 1e-6));
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const float v = va[i] ? 1.f : 0.f; x[2 * i] = (float)(im[i] - m) * v * inv; x[2 * i + 1] = v;
    }
}
extern "C" void vt_input(const uint8_t *im, const uint8_t *va, long long n, float *x) {
    static double *s = nullptr; if (!s) cudaMalloc(&s, 3 * 8);
    cudaMemsetAsync(s, 0, 3 * 8);
    unsigned nb = nb_of(n, 256); if (nb > 4096) nb = 4096;
    k_in_sums<<<nb, 256>>>(im, va, n, s); k_in<<<GRID(n)>>>(im, va, n, s, x);
}
/* rows [A][B][C] -> [B][A][C] */
static __global__ void k_tr(const float *__restrict__ x, float *__restrict__ y, long long A, long long B, int C) {
    const long long tot = A * B * C;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < tot; i += (long long)gridDim.x * blockDim.x) {
        const int c = (int)(i % C); const long long t = i / C, b = t % B, a = t / B; y[(b * A + a) * C + c] = x[i];
    }
}
static __global__ void k_tr4(const float4 *__restrict__ x, float4 *__restrict__ y, long long A, long long B, int C4) {
    const long long tot = A * B * C4;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < tot; i += (long long)gridDim.x * blockDim.x) {
        const int c = (int)(i % C4); const long long t = i / C4, b = t % B, a = t / B; y[(b * A + a) * C4 + c] = x[i];
    }
}
extern "C" void vt_transpose(const float *x, float *y, long long A, long long B, int C) {
    if (C % 4 == 0) k_tr4<<<GRID(A * B * C / 4)>>>((const float4 *)x, (float4 *)y, A, B, C / 4);
    else k_tr<<<GRID(A * B * C)>>>(x, y, A, B, C);
}

static __global__ void k_axpy(float *__restrict__ y, const float *__restrict__ x, long long n, float a) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) y[i] += a * x[i];
}
/* ---- position-only shell loss (villa position_only_phase_loss): per column (no phase labels in it) the centres of the shell runs
   along k (runs closer than 3 voxels merged); consecutive centres: hinge relu(1 - delta) + snap 0.5 (delta - round delta)^2 when
   delta >= 1 and |residual| <= gate; per-column means, averaged over the active columns. Gradients into g (written, unscaled per
   column; k_shell_scale applies 1 / active and the weights). ---- */
static __device__ __forceinline__ float ph_at(const float *p, int L, float t, int *k0, float *f) {
    if (t < 0) t = 0; if (t > L - 1) t = L - 1; int a = (int)t; if (a > L - 2) a = L - 2; *k0 = a; *f = t - a; return p[a] + (p[a + 1] - p[a]) * *f;
}
static __global__ void k_shell(const float *__restrict__ ph, const uint8_t *__restrict__ sh, const uint16_t *__restrict__ comp, long long ncol, int L,
                               float gate, float *__restrict__ g, double *__restrict__ acc /* hinge, snap, active */) {
    const long long c = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (c >= ncol) return;
    const float *p = ph + c * L; const uint8_t *s = sh + c * L; float *gc = g + c * L;
    for (int k = 0; k < L; k++) gc[k] = 0.f;
    if (comp) { for (int k = 0; k < L; k++) if (comp[c * L + k]) return; }
    float cen[64]; int nc = 0; int k = 0;
    while (k < L && nc < 64) {
        if (!s[k]) { k++; continue; }
        int a = k; while (k < L && s[k]) k++; int b = k - 1;
        while (k < L && k - b <= 3) { if (s[k]) { while (k < L && s[k]) k++; b = k - 1; } else k++; }   /* merge runs closer than 3 voxels */
        cen[nc++] = 0.5f * (a + b);
    }
    if (nc < 2) return;
    double hs = 0, ss = 0; const int np = nc - 1;
    for (int i = 0; i < np; i++) {
        int k0, k1; float f0, f1; const float a0 = ph_at(p, L, cen[i], &k0, &f0), a1 = ph_at(p, L, cen[i + 1], &k1, &f1), d = a1 - a0;
        float gd = 0.f;
        if (d < 1.f) { hs += 1.f - d; gd -= 1.f; }
        const float r = d - rintf(d);
        if (d >= 1.f && fabsf(r) <= gate) { ss += 0.5 * r * r; gd += r; }
        gd /= np;   /* d/d delta of the column mean */
        gc[k1] += gd * (1 - f1); gc[k1 + 1] += gd * f1; gc[k0] -= gd * (1 - f0); gc[k0 + 1] -= gd * f0;
    }
    atomicAdd(&acc[0], hs / np); atomicAdd(&acc[1], ss / np); atomicAdd(&acc[2], 1.0);
}
extern "C" int vt_shell_loss(const float *ph, const uint8_t *shell, const uint16_t *comp, int T, int L, float gate, float lam_h, float lam_s, float scale,
                             float *gph, double *hinge, double *snap, double *ncol) {
    static double *acc = nullptr; static float *gs = nullptr; static long long gn = 0;
    const long long nc = (long long)T * T, n = nc * L;
    if (!acc) cudaMalloc(&acc, 3 * 8);
    if (gn < n) { cudaFree(gs); cudaMalloc(&gs, n * 4); gn = n; }
    cudaMemsetAsync(acc, 0, 3 * 8);
    k_shell<<<nb_of(nc, 128), 128>>>(ph, shell, comp, nc, L, gate, gs, acc);
    double h[3]; cudaMemcpy(h, acc, 3 * 8, cudaMemcpyDeviceToHost);
    *hinge = h[2] > 0 ? h[0] / h[2] : 0; *snap = h[2] > 0 ? h[1] / h[2] : 0; *ncol = h[2];
    /* gs holds d(hinge + snap column means)/d ph with both terms weighted 1: re-weight is approximated by lam_h for the whole (hinge and
       snap gradients share the column); the trainer passes lam_h == lam_s */
    if (gph && h[2] > 0) { const float sc = (float)(lam_h * scale / h[2]); k_axpy<<<GRID(n)>>>(gph, gs, n, sc); }
    (void)lam_s;
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
