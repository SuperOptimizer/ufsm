/* bf16 transformer blocks (src/attn.h, at_set_mx(16) / UFSM_LINEAR_MX=16): the activations between the block's linear layers in bf16, the
   linear layers on bf16 tensor cores (cuBLASLt, fp32 accumulation) with the bias / residual in the GEMM epilogue, the bias gradients as
   column sums fused into the kernels that produce the bf16 gradient rows, and the backward's RoPE inverse + QK-norm + gain gradients in one
   pass. The residual stream (x, x1, y), LayerNorm statistics, the attention core's inputs (fp16 packed q / k / v) and the gradients that
   feed LayerNorm backward stay fp32.

   Forward (x fp32 [T][C]): LN1 -> xn bf16; qkv = xn Wqkv^T + b (bf16 [T][3C]); pack (QK-norm, RoPE) -> fp16 heads; flash attention ->
   o bf16; x1 = o Wo^T + bo + x (fp32); LN2 -> xn2 bf16; h = xn2 W1^T + b1 (bf16 [T][2F]); gg = silu(h_a) h_b (bf16); y = gg W2^T + b2 + x1.
   Backward: the forward recomputed (without its last GEMM), then the chain in reverse (at_block_bwd_h16 below).

   Qualifies (at_block_h16_ok): no deformable attention, no rays, no value residual, the fp16 tensor-core core, C % 64 == 0 (C <= 1024),
   F % 8 == 0. Workspace: at_ws as carved for the fp32 block (the bf16 tensors use the first half of their fp32 buffers; qkn, unused by the
   tensor-core core, holds bf16 gradient rows). */
#include "attn.h"
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HD AT_HD
typedef __nv_bfloat16 bf;
static unsigned nbk(long long n, int t) { return (unsigned)((n + t - 1) / t); }

/* ---- GEMM: row-major D[M][N] = op(A) op(B) (+ beta Cin) (+ bias[N]); A, B bf16, D fp32 (Cin fp32) or bf16; cuBLASLt column-major as
   in nn_attn.cu (D^T = op(B)^T op(A)^T) ---- */
typedef struct { int M, N, K, ta, tb, dbf, bias, al; cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a, b, c; cublasLtMatmulAlgo_t algo; int ok, tuned, nh;
                 cublasLtMatmulHeuristicResult_t h[8]; } hplan;
static hplan g_hp[128]; static int g_nhp = 0;
static cublasLtHandle_t g_h = nullptr; static void *g_hws = nullptr; static const size_t HWS = 32u << 20;
static hplan *hplan_get(int M, int N, int K, int ta, int tb, int dbf, int bias, int al) {
    for (int i = 0; i < g_nhp; i++) { hplan *p = &g_hp[i]; if (p->M == M && p->N == N && p->K == K && p->ta == ta && p->tb == tb && p->dbf == dbf && p->bias == bias && p->al == al) return p; }
    if (!g_h) { if (cublasLtCreate(&g_h) != CUBLAS_STATUS_SUCCESS) return nullptr; if (cudaMalloc(&g_hws, HWS) != cudaSuccess) return nullptr; }
    static int next = 0; hplan *p;
    if (g_nhp < 128) p = &g_hp[g_nhp++];
    else { p = &g_hp[next]; next = (next + 1) % 128; cublasLtMatmulDescDestroy(p->op); cublasLtMatrixLayoutDestroy(p->a); cublasLtMatrixLayoutDestroy(p->b); cublasLtMatrixLayoutDestroy(p->c); }
    memset(p, 0, sizeof *p);
    p->M = M; p->N = N; p->K = K; p->ta = ta; p->tb = tb; p->dbf = dbf; p->bias = bias; p->al = al;
    cublasLtMatmulDescCreate(&p->op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    const cublasOperation_t oX = tb ? CUBLAS_OP_T : CUBLAS_OP_N, oY = ta ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &oX, sizeof oX);
    cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSB, &oY, sizeof oY);
    if (bias) {
        const cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_BIAS; const cudaDataType_t bt = dbf ? CUDA_R_16BF : CUDA_R_32F;   /* (bf16 D: bf16 bias only) */
        cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof ep);
        cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bt, sizeof bt);
    }
    if (tb) cublasLtMatrixLayoutCreate(&p->a, CUDA_R_16BF, K, N, K); else cublasLtMatrixLayoutCreate(&p->a, CUDA_R_16BF, N, K, N);
    if (ta) cublasLtMatrixLayoutCreate(&p->b, CUDA_R_16BF, M, K, M); else cublasLtMatrixLayoutCreate(&p->b, CUDA_R_16BF, K, M, K);
    cublasLtMatrixLayoutCreate(&p->c, dbf ? CUDA_R_16BF : CUDA_R_32F, N, M, N);
    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &HWS, sizeof HWS);
    if (al < 16) { const uint32_t a4 = (uint32_t)al;
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &a4, sizeof a4);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &a4, sizeof a4); }
    int n = 0;
    if (cublasLtMatmulAlgoGetHeuristic(g_h, p->op, p->a, p->b, p->c, p->c, pref, 8, p->h, &n) == CUBLAS_STATUS_SUCCESS && n > 0) { p->algo = p->h[0].algo; p->ok = 1; p->nh = n; }
    cublasLtMatmulPreferenceDestroy(pref);
    if (!p->ok) fprintf(stderr, "at h16 gemm: no cuBLASLt algorithm for M %d N %d K %d ta %d tb %d dbf %d bias %d al %d\n", M, N, K, ta, tb, dbf, bias, al);
    return p;
}
static int hg(int ta, int tb, int M, int N, int K, const void *A, const void *B, void *D, int dbf, float beta, const float *Cin, const void *bias) {
    if (M <= 0 || N <= 0 || K <= 0) return -1;
    const uintptr_t u = (uintptr_t)D | (uintptr_t)(Cin ? Cin : D);
    hplan *p = hplan_get(M, N, K, ta, tb, dbf, bias != nullptr, (u & 15) ? 4 : 16);
    if (!p || !p->ok) return -1;
    if (bias) cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof bias);
    const float one = 1.f;
    if (!p->tuned) {   /* first call: time the heuristic's candidates on these operands (an in-place accumulation writes a scratch copy) */
        p->tuned = 1;
        const size_t db = (size_t)M * N * (dbf ? 2 : 4); void *Dt = D; static void *tmp = nullptr; static size_t tcap = 0;
        if (Cin == (const float *)D && beta != 0.f) {
            if (db > ((size_t)64 << 20)) p->nh = 1;
            else if (db > tcap) { if (tmp) cudaFree(tmp); tmp = nullptr; tcap = 0; if (cudaMalloc(&tmp, db) == cudaSuccess) tcap = db; else p->nh = 1; }
            Dt = tmp;
        }
        if (p->nh > 1 && getenv("UFSM_H16_NOTUNE") == nullptr) {
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); float best = 1e30f; int bi = 0;
            for (int i = 0; i < p->nh; i++) {
                float tt = 0.f; int bad = 0;
                for (int r = 0; r < 3 && !bad; r++) {
                    cudaEventRecord(e0, 0);
                    bad = cublasLtMatmul(g_h, p->op, &one, B, p->a, A, p->b, &beta, Cin ? (const void *)Cin : D, p->c, Dt, p->c, &p->h[i].algo, g_hws, HWS, 0) != CUBLAS_STATUS_SUCCESS;
                    cudaEventRecord(e1, 0); cudaEventSynchronize(e1); float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1); if (r) tt += ms;
                }
                if (!bad && tt < best) { best = tt; bi = i; }
            }
            p->algo = p->h[bi].algo; cudaEventDestroy(e0); cudaEventDestroy(e1);
        }
    }
    const cublasStatus_t s = cublasLtMatmul(g_h, p->op, &one, B, p->a, A, p->b, &beta, Cin ? (const void *)Cin : D, p->c, D, p->c, &p->algo, g_hws, HWS, 0);
    if (s != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "at h16 gemm: cublasLtMatmul failed (%d)\n", (int)s); return -1; }
    return 0;
}

/* ---- the block's weights in bf16 and biases at aligned addresses (one launch per block call) ---- */
typedef struct { const float *s[8]; void *d[8]; long long n[8]; int bf[8]; } prep_t;
__global__ void k_prep(prep_t q) {
    const int k = blockIdx.y; const long long n = q.n[k];
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        if (q.bf[k]) ((bf *)q.d[k])[i] = __float2bfloat16(q.s[k][i]); else ((float *)q.d[k])[i] = q.s[k][i];
    }
}
typedef struct { bf *wqkv, *wo, *w1, *w2, *bqkv, *b1; float *bo, *b2; } hw_t;   /* (the biases of bf16 outputs in bf16) */
static void *g_wbuf = nullptr; static size_t g_wcap = 0;
static size_t al256(size_t b) { return (b + 255) & ~(size_t)255; }
static int prep_weights(const at_dims *d, const float *P, const at_bp *bp, hw_t *hw) {
    const size_t C = d->C, F = d->F;
    const size_t sz[8] = {3 * C * C * 2, C * C * 2, 2 * F * C * 2, C * F * 2, 3 * C * 2, C * 4, 2 * F * 2, C * 4};
    size_t tot = 0; for (int k = 0; k < 8; k++) tot += al256(sz[k]);
    if (tot > g_wcap) { if (g_wbuf) cudaFree(g_wbuf); if (cudaMalloc(&g_wbuf, tot) != cudaSuccess) { g_wbuf = nullptr; g_wcap = 0; return -1; } g_wcap = tot; }
    char *b = (char *)g_wbuf; void *dp[8];
    for (int k = 0; k < 8; k++) { dp[k] = b; b += al256(sz[k]); }
    prep_t q; const size_t src[8] = {bp->wqkv, bp->wo, bp->w1, bp->w2, bp->bqkv, bp->bo, bp->b1, bp->b2};
    for (int k = 0; k < 8; k++) { q.s[k] = P + src[k]; q.d[k] = dp[k]; q.bf[k] = k < 4 || k == 4 || k == 6; q.n[k] = (long long)(sz[k] / (q.bf[k] ? 2 : 4)); }
    k_prep<<<dim3(64, 8), 256>>>(q);
    hw->wqkv = (bf *)dp[0]; hw->wo = (bf *)dp[1]; hw->w1 = (bf *)dp[2]; hw->w2 = (bf *)dp[3];
    hw->bqkv = (bf *)dp[4]; hw->bo = (float *)dp[5]; hw->b1 = (bf *)dp[6]; hw->b2 = (float *)dp[7];
    return 0;
}

/* ---- LayerNorm forward -> bf16: a warp per row, lane: column pairs 2 lane + 64 k ---- */
template <int NC2>
__global__ void __launch_bounds__(256) k_ln_fwd_h(const float *__restrict__ x, const float *__restrict__ g, const float *__restrict__ b, bf *__restrict__ y,
                                                  float *mean, float *rstd, long long T, int C) {
    const long long row = (blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5; const int lane = threadIdx.x & 31;
    if (row >= T) return;
    const float *xr = x + row * C; float2 v[NC2]; float s = 0.f;
#pragma unroll
    for (int k = 0; k < NC2; k++) { v[k] = *(const float2 *)(xr + 2 * lane + 64 * k); s += v[k].x + v[k].y; }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float m = s / C; float q = 0.f;
#pragma unroll
    for (int k = 0; k < NC2; k++) { const float a = v[k].x - m, c = v[k].y - m; q += a * a + c * c; }
    for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    const float r = 1.f / sqrtf(q / C + 1e-5f);
#pragma unroll
    for (int k = 0; k < NC2; k++) {
        const int c = 2 * lane + 64 * k; const float2 gg = *(const float2 *)(g + c), bb = *(const float2 *)(b + c);
        *(__nv_bfloat162 *)(y + row * C + c) = __floats2bfloat162_rn((v[k].x - m) * r * gg.x + bb.x, (v[k].y - m) * r * gg.y + bb.y);
    }
    if (!lane) { mean[row] = m; rstd[row] = r; }
}
/* LayerNorm backward: out = (res +) rstd (dy g - mean(dy g) - xhat mean(dy g xhat)); optional bf16 copy outb and column sums dcol of out;
   dg += sum dy xhat, db += sum dy. 32 rows per block (8 warps), per-lane partials -> shared -> one global atomic per (block, column) */
#define LBR 32
template <int NC2>
__global__ void __launch_bounds__(256) k_ln_bwd_h(const float *__restrict__ x, const float *__restrict__ g, const float *__restrict__ mean, const float *__restrict__ rstd,
                                                  const float *__restrict__ dy, const float *res, float *out, bf *outb, float *dg, float *db, float *dcol, long long T, int C) {
    __shared__ float sh[3][NC2 * 64];
    for (int c = threadIdx.x; c < 3 * NC2 * 64; c += blockDim.x) (&sh[0][0])[c] = 0.f;
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const long long r0 = (long long)blockIdx.x * LBR;
    float2 ag[NC2], ab[NC2], ac[NC2], gg[NC2];
#pragma unroll
    for (int k = 0; k < NC2; k++) { ag[k] = ab[k] = ac[k] = make_float2(0.f, 0.f); gg[k] = *(const float2 *)(g + 2 * lane + 64 * k); }
    for (long long row = r0 + warp; row < r0 + LBR && row < T; row += 8) {
        const float m = mean[row], r = rstd[row]; float2 xh[NC2], gy[NC2]; float a = 0.f, bb = 0.f;
#pragma unroll
        for (int k = 0; k < NC2; k++) {
            const int c = 2 * lane + 64 * k; const float2 xv = *(const float2 *)(x + row * C + c); gy[k] = *(const float2 *)(dy + row * C + c);
            xh[k] = make_float2((xv.x - m) * r, (xv.y - m) * r);
            const float g0 = gy[k].x * gg[k].x, g1 = gy[k].y * gg[k].y; a += g0 + g1; bb += g0 * xh[k].x + g1 * xh[k].y;
            ag[k].x += gy[k].x * xh[k].x; ag[k].y += gy[k].y * xh[k].y; ab[k].x += gy[k].x; ab[k].y += gy[k].y;
        }
        for (int o = 16; o; o >>= 1) { a += __shfl_xor_sync(0xffffffffu, a, o); bb += __shfl_xor_sync(0xffffffffu, bb, o); }
        a /= C; bb /= C;
#pragma unroll
        for (int k = 0; k < NC2; k++) {
            const int c = 2 * lane + 64 * k;
            float2 v = make_float2(r * (gy[k].x * gg[k].x - a - xh[k].x * bb), r * (gy[k].y * gg[k].y - a - xh[k].y * bb));
            if (res) { const float2 q = *(const float2 *)(res + row * C + c); v.x += q.x; v.y += q.y; }
            *(float2 *)(out + row * C + c) = v;
            if (outb) *(__nv_bfloat162 *)(outb + row * C + c) = __floats2bfloat162_rn(v.x, v.y);
            ac[k].x += v.x; ac[k].y += v.y;
        }
    }
#pragma unroll
    for (int k = 0; k < NC2; k++) {
        const int c = 2 * lane + 64 * k;
        atomicAdd(&sh[0][c], ag[k].x); atomicAdd(&sh[0][c + 1], ag[k].y); atomicAdd(&sh[1][c], ab[k].x); atomicAdd(&sh[1][c + 1], ab[k].y);
        if (dcol) { atomicAdd(&sh[2][c], ac[k].x); atomicAdd(&sh[2][c + 1], ac[k].y); }
    }
    __syncthreads();
    for (int c = threadIdx.x; c < C; c += blockDim.x) { atomicAdd(dg + c, sh[0][c]); atomicAdd(db + c, sh[1][c]); if (dcol) atomicAdd(dcol + c, sh[2][c]); }
}
#define NC2_SWITCH(C, F, ...) do { switch ((C) / 64) { \
    case 1: { constexpr int F = 1; __VA_ARGS__; } break; case 2: { constexpr int F = 2; __VA_ARGS__; } break; \
    case 3: { constexpr int F = 3; __VA_ARGS__; } break; case 4: { constexpr int F = 4; __VA_ARGS__; } break; \
    case 6: { constexpr int F = 6; __VA_ARGS__; } break; case 8: { constexpr int F = 8; __VA_ARGS__; } break; \
    case 12: { constexpr int F = 12; __VA_ARGS__; } break; case 16: { constexpr int F = 16; __VA_ARGS__; } break; \
    default: return -1; } } while (0)
static int nc2_ok(int C) { const int k = C / 64; return C % 64 == 0 && (k == 1 || k == 2 || k == 3 || k == 4 || k == 6 || k == 8 || k == 12 || k == 16); }
static int ln_fwd_h(const float *x, const float *g, const float *b, bf *y, float *mean, float *rstd, long long T, int C) {
    NC2_SWITCH(C, NC, (k_ln_fwd_h<NC><<<nbk(T * 32, 256), 256>>>(x, g, b, y, mean, rstd, T, C)));
    return 0;
}
static int ln_bwd_h(const float *x, const float *g, const float *mean, const float *rstd, const float *dy, const float *res, float *out, bf *outb, float *dg, float *db, float *dcol, long long T, int C) {
    NC2_SWITCH(C, NC, (k_ln_bwd_h<NC><<<nbk(T, LBR), 256>>>(x, g, mean, rstd, dy, res, out, outb, dg, db, dcol, T, C)));
    return 0;
}

/* ---- SwiGLU (h = [a | b] per row, bf16) ---- */
__global__ void k_glu_fwd_h(const bf *__restrict__ h, bf *__restrict__ gg, long long T, int F) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x, F2 = F / 2;
    if (i >= T * F2) return;
    const long long t = i / F2; const int f = 2 * (int)(i % F2);
    const float2 a = __bfloat1622float2(*(const __nv_bfloat162 *)(h + t * 2 * F + f)), b = __bfloat1622float2(*(const __nv_bfloat162 *)(h + t * 2 * F + F + f));
    *(__nv_bfloat162 *)(gg + t * F + f) = __floats2bfloat162_rn(a.x / (1.f + __expf(-a.x)) * b.x, a.y / (1.f + __expf(-a.y)) * b.y);
}
/* column-pair loops: x threads over pairs, y over rows, 64 rows per block; partial column sums -> shared -> one atomic per (block, column) */
#define CSR 64
__global__ void k_glu_bwd_h(const bf *__restrict__ h, const bf *__restrict__ dgg, bf *__restrict__ dh, float *db, long long T, int F) {
    extern __shared__ float shc[];   /* [2F] */
    const int F2 = F / 2, p = threadIdx.x, f = 2 * p;
    for (int c = threadIdx.y * blockDim.x + threadIdx.x; c < 2 * F; c += blockDim.x * blockDim.y) shc[c] = 0.f;
    __syncthreads();
    float sa0 = 0.f, sa1 = 0.f, sb0 = 0.f, sb1 = 0.f;
    if (p < F2) {
        const long long r1 = (long long)blockIdx.x * CSR + CSR < T ? (long long)blockIdx.x * CSR + CSR : T;
        for (long long t = (long long)blockIdx.x * CSR + threadIdx.y; t < r1; t += blockDim.y) {
            const float2 a = __bfloat1622float2(*(const __nv_bfloat162 *)(h + t * 2 * F + f)), b = __bfloat1622float2(*(const __nv_bfloat162 *)(h + t * 2 * F + F + f));
            const float2 g = __bfloat1622float2(*(const __nv_bfloat162 *)(dgg + t * F + f));
            const float s0 = 1.f / (1.f + __expf(-a.x)), s1 = 1.f / (1.f + __expf(-a.y));
            const __nv_bfloat162 da = __floats2bfloat162_rn(g.x * b.x * s0 * (1.f + a.x * (1.f - s0)), g.y * b.y * s1 * (1.f + a.y * (1.f - s1)));
            const __nv_bfloat162 dbv = __floats2bfloat162_rn(g.x * a.x * s0, g.y * a.y * s1);
            *(__nv_bfloat162 *)(dh + t * 2 * F + f) = da; *(__nv_bfloat162 *)(dh + t * 2 * F + F + f) = dbv;
            const float2 fa = __bfloat1622float2(da), fb = __bfloat1622float2(dbv);   /* (the sums of what the weight gradient sees) */
            sa0 += fa.x; sa1 += fa.y; sb0 += fb.x; sb1 += fb.y;
        }
        atomicAdd(shc + f, sa0); atomicAdd(shc + f + 1, sa1); atomicAdd(shc + F + f, sb0); atomicAdd(shc + F + f + 1, sb1);
    }
    __syncthreads();
    for (int c = threadIdx.y * blockDim.x + threadIdx.x; c < 2 * F; c += blockDim.x * blockDim.y) atomicAdd(db + c, shc[c]);
}
/* fp32 rows -> bf16, db += column sums (of the bf16 values) */
__global__ void k_cvt_cs(const float *__restrict__ x, bf *__restrict__ y, float *db, long long T, int N) {
    extern __shared__ float shc[];   /* [N] */
    const int N2 = N / 2, p = threadIdx.x, c = 2 * p;
    for (int k = threadIdx.y * blockDim.x + threadIdx.x; k < N; k += blockDim.x * blockDim.y) shc[k] = 0.f;
    __syncthreads();
    if (p < N2) {
        float s0 = 0.f, s1 = 0.f;
        const long long r1 = (long long)blockIdx.x * CSR + CSR < T ? (long long)blockIdx.x * CSR + CSR : T;
        for (long long t = (long long)blockIdx.x * CSR + threadIdx.y; t < r1; t += blockDim.y) {
            const float2 v = *(const float2 *)(x + t * N + c); const __nv_bfloat162 b = __floats2bfloat162_rn(v.x, v.y);
            *(__nv_bfloat162 *)(y + t * N + c) = b; const float2 q = __bfloat1622float2(b); s0 += q.x; s1 += q.y;
        }
        atomicAdd(shc + c, s0); atomicAdd(shc + c + 1, s1);
    }
    __syncthreads();
    for (int k = threadIdx.y * blockDim.x + threadIdx.x; k < N; k += blockDim.x * blockDim.y) atomicAdd(db + k, shc[k]);
}
static dim3 pair_block(int pairs) { const int x = (pairs + 31) / 32 * 32; int y = 256 / x; if (y < 1) y = 1; return dim3(x, y); }

/* ---- backward of pack: RoPE inverse on d q / d k (fp32 dqn [T][3C] from the attention backward), QK-norm backward against the bf16 qkv,
   the gains' gradients, d qkv -> bf16 and its column sums (the qkv bias gradient). Block: H warps (warp = head), 32 tokens; lane l < 24:
   head dims 2 l, 2 l + 1 (= RoPE pair l: axis l / 8, frequency l % 8) ---- */
#define QTB 32
__global__ void k_qkn_bwd_h(const float *__restrict__ dqn, const bf *__restrict__ qkv, const float *__restrict__ qg, const float *__restrict__ kg,
                            const float *__restrict__ rq, const float *__restrict__ rk, bf *__restrict__ dqkv, float *dqg, float *dkg, float *dbqkv,
                            long long T, int C, int S, int Hh, int W, int nreg, int rope, float lbase, const float *gpos) {
    const int h = threadIdx.x >> 5, l = threadIdx.x & 31, H = C / HD; const bool on = l < 24;
    const int ax = l / 8, fi = l % 8, c = h * HD + 2 * l;
    const float frq = exp2f(-lbase * (float)fi / 8.f);
    float2 gq = make_float2(0.f, 0.f), gk = gq, dgq = gq, dgk = gq, cs[3] = {gq, gq, gq};
    if (on) { gq = *(const float2 *)(qg + c); gk = *(const float2 *)(kg + c); }
    const long long t1 = (long long)blockIdx.x * QTB + QTB < T ? (long long)blockIdx.x * QTB + QTB : T;
    for (long long t = (long long)blockIdx.x * QTB; t < t1; t++) {
        float sn = 0.f, cn = 1.f;
        const int ts = (int)(t % S) - nreg;
        if (rope && ts >= 0 && on) {
            const float pos = gpos ? gpos[3 * t + ax] : (float)(ax == 0 ? ts / (Hh * W) : ax == 1 ? (ts / W) % Hh : ts % W);
            sincosf(pos * frq, &sn, &cn);
        }
        const long long o = t * 3 * C;
#pragma unroll
        for (int p = 0; p < 2; p++) {
            float2 d = make_float2(0.f, 0.f), x = d; const float2 g = p ? gk : gq;
            if (on) {
                const float2 e = *(const float2 *)(dqn + o + p * C + c);
                d = make_float2(e.x * cn + e.y * sn, -e.x * sn + e.y * cn);   /* (the rotation's transpose) */
                x = __bfloat1622float2(*(const __nv_bfloat162 *)(qkv + o + p * C + c));
            }
            float s = g.x * d.x * x.x + g.y * d.y * x.y;
            for (int k = 16; k; k >>= 1) s += __shfl_xor_sync(0xffffffffu, s, k);
            if (!on) continue;
            const float r = (p ? rk : rq)[t * H + h], k3 = r * r * r * s / HD;
            const __nv_bfloat162 b = __floats2bfloat162_rn(r * g.x * d.x - x.x * k3, r * g.y * d.y - x.y * k3);
            *(__nv_bfloat162 *)(dqkv + o + p * C + c) = b;
            const float2 q = __bfloat1622float2(b); cs[p].x += q.x; cs[p].y += q.y;
            float2 &dg = p ? dgk : dgq; dg.x += d.x * x.x * r; dg.y += d.y * x.y * r;
        }
        if (on) {
            const float2 e = *(const float2 *)(dqn + o + 2 * C + c); const __nv_bfloat162 b = __floats2bfloat162_rn(e.x, e.y);
            *(__nv_bfloat162 *)(dqkv + o + 2 * C + c) = b; const float2 q = __bfloat1622float2(b); cs[2].x += q.x; cs[2].y += q.y;
        }
    }
    if (!on) return;
    atomicAdd(dqg + c, dgq.x); atomicAdd(dqg + c + 1, dgq.y); atomicAdd(dkg + c, dgk.x); atomicAdd(dkg + c + 1, dgk.y);
#pragma unroll
    for (int p = 0; p < 3; p++) { atomicAdd(dbqkv + p * C + c, cs[p].x); atomicAdd(dbqkv + p * C + c + 1, cs[p].y); }
}

/* ---- the block ---- */
extern "C" int at_block_h16_ok(const at_dims *d, const at_geom *g, const at_ws *w) {
    return d->dK <= 0 && !g->rays && !w->vr && at_get_core() == 1 && nc2_ok(d->C) && d->C % HD == 0 && d->F % 8 == 0 && d->C / HD <= 32;
}
#define CK(x) do { if (x) return -1; } while (0)
static hw_t g_hw;
/* recompute: the backward's forward (weights converted, everything but the last GEMM; y unused) */
extern "C" int at_block_fwd_h16(const at_dims *d, const float *P, const at_bp *bp, const float *x, float *y, const at_geom *g, at_ws *w, int recompute) {
    const long long T = (long long)g->nseq * g->S; const int C = d->C, F = d->F, H = d->H;
    CK(prep_weights(d, P, bp, &g_hw));
    bf *xn = (bf *)w->xn, *qkv = (bf *)w->qkv, *o = (bf *)w->o, *xn2 = (bf *)w->xn2, *h = (bf *)w->h, *gg = (bf *)w->gg;
    CK(ln_fwd_h(x, P + bp->ln1g, P + bp->ln1b, xn, w->m1, w->r1, T, C));
    CK(hg(0, 1, (int)T, 3 * C, C, xn, g_hw.wqkv, qkv, 1, 0.f, nullptr, g_hw.bqkv));
    at_fa_pack_bf(qkv, P + bp->qg, P + bp->kg, w->qkvh, w->rq, w->rk, w->amx, g, d->nreg, C, d->rope, d->rope_base);
    at_fa_fwd_bf(w->qkvh, o, w->lse, g, H, C);
    CK(hg(0, 1, (int)T, C, C, o, g_hw.wo, w->x1, 0, 1.f, x, g_hw.bo));
    CK(ln_fwd_h(w->x1, P + bp->ln2g, P + bp->ln2b, xn2, w->m2, w->r2, T, C));
    CK(hg(0, 1, (int)T, 2 * F, C, xn2, g_hw.w1, h, 1, 0.f, nullptr, g_hw.b1));
    k_glu_fwd_h<<<nbk(T * F / 2, 256), 256>>>(h, gg, T, F);
    if (!recompute) CK(hg(0, 1, (int)T, C, F, gg, g_hw.w2, y, 0, 1.f, w->x1, g_hw.b2));
    return 0;
}
extern "C" int at_block_bwd_h16(const at_dims *d, const float *P, float *G, const at_bp *bp, const float *xin, float *gy, const at_geom *g, at_ws *w) {
    const long long T = (long long)g->nseq * g->S; const int C = d->C, F = d->F, H = d->H;
    CK(at_block_fwd_h16(d, P, bp, xin, nullptr, g, w, 1));
    w->kept = 0;
    const bf *xn = (const bf *)w->xn, *qkv = (const bf *)w->qkv, *o = (const bf *)w->o, *xn2 = (const bf *)w->xn2, *h = (const bf *)w->h, *gg = (const bf *)w->gg;
    bf *rows = (bf *)w->qkn, *dgg = (bf *)w->bB, *dh = (bf *)w->bA, *dqkv = (bf *)w->bB; float *dqn = w->bA;
    /* MLP */
    { const dim3 bk = pair_block(C / 2); k_cvt_cs<<<nbk(T, CSR), bk, C * 4>>>(gy, rows, G + bp->b2, T, C); }   /* gy -> bf16, d b2 */
    CK(hg(1, 0, C, F, (int)T, rows, gg, G + bp->w2, 0, 1.f, G + bp->w2, nullptr));        /* d W2 += gy^T gg */
    CK(hg(0, 0, (int)T, F, C, rows, g_hw.w2, dgg, 1, 0.f, nullptr, nullptr));            /* d gg = gy W2 */
    { const dim3 bk = pair_block(F / 2); k_glu_bwd_h<<<nbk(T, CSR), bk, 2 * F * 4>>>(h, dgg, dh, G + bp->b1, T, F); }   /* d h, d b1 */
    CK(hg(1, 0, 2 * F, C, (int)T, dh, xn2, G + bp->w1, 0, 1.f, G + bp->w1, nullptr));    /* d W1 += dh^T xn2 */
    CK(hg(0, 0, (int)T, C, 2 * F, dh, g_hw.w1, w->bC, 0, 0.f, nullptr, nullptr));        /* d xn2 */
    CK(ln_bwd_h(w->x1, P + bp->ln2g, w->m2, w->r2, w->bC, gy, w->dx1, rows, G + bp->ln2g, G + bp->ln2b, G + bp->bo, T, C));   /* d x1 (+ bf16, d bo) */
    /* attention */
    CK(hg(1, 0, C, C, (int)T, rows, o, G + bp->wo, 0, 1.f, G + bp->wo, nullptr));        /* d Wo += dx1^T o */
    CK(hg(0, 0, (int)T, C, C, rows, g_hw.wo, w->bC, 0, 0.f, nullptr, nullptr));          /* d o */
    at_fa_bwd_bf(w->qkvh, w->doh, o, w->lse, w->bC, w->dv, w->amx, dqn, g, H, C);
    k_qkn_bwd_h<<<nbk(T, QTB), 32 * H>>>(dqn, qkv, P + bp->qg, P + bp->kg, w->rq, w->rk, dqkv, G + bp->qg, G + bp->kg, G + bp->bqkv,
                                          T, C, g->S, g->Hh, g->W, d->nreg, d->rope, log2f(d->rope_base), g->pos);
    CK(hg(1, 0, 3 * C, C, (int)T, dqkv, xn, G + bp->wqkv, 0, 1.f, G + bp->wqkv, nullptr));   /* d Wqkv += dqkv^T xn */
    CK(hg(0, 0, (int)T, C, 3 * C, dqkv, g_hw.wqkv, w->bC, 0, 0.f, nullptr, nullptr));       /* d xn */
    CK(ln_bwd_h(xin, P + bp->ln1g, w->m1, w->r1, w->bC, w->dx1, gy, nullptr, G + bp->ln1g, G + bp->ln1b, nullptr, T, C));
    return 0;
}
