/* villa winding model (ufsm villawind, src/villawind_net.c) kernels: slab input, channel LayerNorm (fp16 / fp32 rows, optional GELU
   and residual), fp32 residual add, the relative-bias softmax of the ray / transverse attentions, the strided batched attention GEMMs
   (cuBLASLt, fp16 x fp16 -> fp32 scores, fp16 probabilities x fp16 V -> fp16), and the phase head (pixel shuffle, softplus, cumsum).
   Everything runs on the xnet stream. */
#include "xnet.h"
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define ST ((cudaStream_t)xnet_stream())
static unsigned nb_of(long long n, int t) { return (unsigned)((n + t - 1) / t); }

/* image u8 [D][H][W], valid u8 -> fp16 [D][H][W][8]: (x - mean) valid inv, valid, 0 x 6 */
__global__ void k_vw_input(const uint8_t *__restrict__ im, const uint8_t *__restrict__ va, float mean, float inv, __half *__restrict__ y, long long n) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i >= n) return;
    const float v = va[i] ? 1.f : 0.f; __half2 *o = (__half2 *)(y + i * 8);
    o[0] = __floats2half2_rn((im[i] - mean) * v * inv, v); o[1] = o[2] = o[3] = __floats2half2_rn(0.f, 0.f);
}
extern "C" void vw_input(const uint8_t *im, const uint8_t *va, float mean, float inv, void *y, long long n) {
    k_vw_input<<<nb_of(n, 256), 256, 0, ST>>>(im, va, mean, inv, (__half *)y, n);
}

__device__ __forceinline__ float gelu_f(float x) { return 0.5f * x * (1.f + erff(x * 0.70710678118654752f)); }

/* one warp per row: y = act(LN(x) w + b) (+ res); x fp16 (x32 = 0) or fp32 rows of C <= 1024; y fp16 (y32 = 0) or fp32 */
template <int NV> __global__ void k_vw_ln(const void *__restrict__ xv, int x32, const float *__restrict__ w, const float *__restrict__ b, float eps, int gelu,
                                          const __half *__restrict__ res, void *__restrict__ yv, int y32, long long M, int C) {
    const long long r = blockIdx.x * (long long)(blockDim.x / 32) + threadIdx.x / 32; if (r >= M) return;
    const int l = threadIdx.x & 31; float v[NV]; float s = 0;
#pragma unroll
    for (int k = 0; k < NV; k++) { const int c = l + 32 * k; v[k] = c < C ? (x32 ? ((const float *)xv)[r * C + c] : __half2float(((const __half *)xv)[r * C + c])) : 0.f; s += v[k]; }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float mu = s / C; float q = 0;
#pragma unroll
    for (int k = 0; k < NV; k++) { const int c = l + 32 * k; if (c < C) { const float d = v[k] - mu; q += d * d; } }
    for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    const float is = rsqrtf(q / C + eps);
#pragma unroll
    for (int k = 0; k < NV; k++) { const int c = l + 32 * k; if (c >= C) continue;
        float y = (v[k] - mu) * is * w[c] + b[c]; if (gelu) y = gelu_f(y); if (res) y += __half2float(res[r * C + c]);
        if (y32) ((float *)yv)[r * C + c] = y; else ((__half *)yv)[r * C + c] = __float2half_rn(y); }
}
extern "C" void vw_ln(const void *x, int x32, const float *w, const float *b, float eps, int gelu, const void *res, void *y, int y32, long long M, int C) {
    const unsigned g = nb_of(M, 8);
    if (C <= 64) k_vw_ln<2><<<g, 256, 0, ST>>>(x, x32, w, b, eps, gelu, (const __half *)res, y, y32, M, C);
    else if (C <= 128) k_vw_ln<4><<<g, 256, 0, ST>>>(x, x32, w, b, eps, gelu, (const __half *)res, y, y32, M, C);
    else if (C <= 256) k_vw_ln<8><<<g, 256, 0, ST>>>(x, x32, w, b, eps, gelu, (const __half *)res, y, y32, M, C);
    else k_vw_ln<32><<<g, 256, 0, ST>>>(x, x32, w, b, eps, gelu, (const __half *)res, y, y32, M, C);
}

__global__ void k_vw_addh(float *__restrict__ x, const __half *__restrict__ u, long long n) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i < n) x[i] += __half2float(u[i]);
}
extern "C" void vw_add_half(float *x, const void *u, long long n) { k_vw_addh<<<nb_of(n, 256), 256, 0, ST>>>(x, (const __half *)u, n); }
__global__ void k_vw_h2f(float *__restrict__ x, const __half *__restrict__ u, long long n) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i < n) x[i] = __half2float(u[i]);
}
extern "C" void vw_h2f(float *x, const void *u, long long n) { k_vw_h2f<<<nb_of(n, 256), 256, 0, ST>>>(x, (const __half *)u, n); }
__global__ void k_vw_f2h(__half *__restrict__ x, const float *__restrict__ u, long long n) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i < n) x[i] = __float2half_rn(u[i]);
}
extern "C" void vw_f2h(void *x, const float *u, long long n) { k_vw_f2h<<<nb_of(n, 256), 256, 0, ST>>>((__half *)x, u, n); }

/* scores S [nseq][L][L] fp32 (one head) -> P fp16 = softmax_j(S scale + bias(i, j)).
   mode 0 (ray, 1-D): bias[clamp(j - i, -md, md) + md]; mode 1 (plane side x side, row-major tokens): bias[(ri - rj + md) (2md + 1) + ci - cj + md]. */
template <int NV> __global__ void k_vw_softmax(const float *__restrict__ S, __half *__restrict__ P, long long rows, int L, int mode, int md, int side,
                                               const float *__restrict__ bias, float scale) {
    const long long r = blockIdx.x * (long long)(blockDim.x / 32) + threadIdx.x / 32; if (r >= rows) return;
    const int i = (int)(r % L), l = threadIdx.x & 31; const float *s = S + r * L; float v[NV], m = -1e30f;
#pragma unroll
    for (int k = 0; k < NV; k++) {
        const int j = l + 32 * k; v[k] = -1e30f; if (j >= L) continue;
        float bb;
        if (mode == 0) { int o = j - i; o = o < -md ? -md : o > md ? md : o; bb = bias[o + md]; }
        else { const int ri = i / side, ci = i % side, rj = j / side, cj = j % side; bb = bias[(ri - rj + md) * (2 * md + 1) + ci - cj + md]; }
        v[k] = s[j] * scale + bb; m = fmaxf(m, v[k]);
    }
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float z = 0;
#pragma unroll
    for (int k = 0; k < NV; k++) { const int j = l + 32 * k; v[k] = j < L ? __expf(v[k] - m) : 0.f; z += v[k]; }
    for (int o = 16; o; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
    const float iz = 1.f / z;
#pragma unroll
    for (int k = 0; k < NV; k++) { const int j = l + 32 * k; if (j < L) P[r * L + j] = __float2half_rn(v[k] * iz); }
}
extern "C" void vw_softmax(const float *S, void *P, long long rows, int L, int mode, int md, int side, const float *bias, float scale) {
    const unsigned g = nb_of(rows, 8);
    if (L <= 256) k_vw_softmax<8><<<g, 256, 0, ST>>>(S, (__half *)P, rows, L, mode, md, side, bias, scale);
    else if (L <= 512) k_vw_softmax<16><<<g, 256, 0, ST>>>(S, (__half *)P, rows, L, mode, md, side, bias, scale);
    else k_vw_softmax<32><<<g, 256, 0, ST>>>(S, (__half *)P, rows, L, mode, md, side, bias, scale);
}

/* strided batched column-major GEMM D (m x n) = op(A) op(B), A / B fp16, D fp16 or fp32, fp32 compute. Plans cached by shape. */
static cublasLtHandle_t g_lt = nullptr; static void *g_ws = nullptr; static const size_t WS = 32u << 20;
typedef struct { int ta, tb, m, n, k, lda, ldb, ldd, d32, batch; long long sa, sb, sd; cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t A, B, D; cublasLtMatmulAlgo_t algo; int ok; } vwplan;
static vwplan g_plans[16]; static int g_np = 0;
static vwplan *plan_of(int ta, int tb, int m, int n, int k, int lda, long long sa, int ldb, long long sb, int ldd, long long sd, int d32, int batch) {
    for (int i = 0; i < g_np; i++) { vwplan *p = &g_plans[i];
        if (p->ta == ta && p->tb == tb && p->m == m && p->n == n && p->k == k && p->lda == lda && p->sa == sa && p->ldb == ldb && p->sb == sb && p->ldd == ldd && p->sd == sd && p->d32 == d32 && p->batch == batch) return p; }
    if (!g_lt) { if (cublasLtCreate(&g_lt) != CUBLAS_STATUS_SUCCESS) return nullptr; cudaMalloc(&g_ws, WS); }
    if (g_np == 16) return nullptr;
    vwplan *p = &g_plans[g_np++]; *p = (vwplan){ta, tb, m, n, k, lda, ldb, ldd, d32, batch, sa, sb, sd};
    cublasLtMatmulDescCreate(&p->op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    const cublasOperation_t oa = ta ? CUBLAS_OP_T : CUBLAS_OP_N, ob = tb ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &oa, sizeof oa);
    cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSB, &ob, sizeof ob);
    cublasLtMatrixLayoutCreate(&p->A, CUDA_R_16F, ta ? k : m, ta ? m : k, lda);
    cublasLtMatrixLayoutCreate(&p->B, CUDA_R_16F, tb ? n : k, tb ? k : n, ldb);
    cublasLtMatrixLayoutCreate(&p->D, d32 ? CUDA_R_32F : CUDA_R_16F, m, n, ldd);
    cublasLtMatrixLayout_t L[3] = {p->A, p->B, p->D}; const long long s[3] = {sa, sb, sd};
    for (int i = 0; i < 3; i++) {
        cublasLtMatrixLayoutSetAttribute(L[i], CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof batch);
        cublasLtMatrixLayoutSetAttribute(L[i], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &s[i], sizeof s[i]);
    }
    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WS, sizeof WS);
    cublasLtMatmulHeuristicResult_t h; int nr = 0;
    if (cublasLtMatmulAlgoGetHeuristic(g_lt, p->op, p->A, p->B, p->D, p->D, pref, 1, &h, &nr) == CUBLAS_STATUS_SUCCESS && nr > 0) { p->algo = h.algo; p->ok = 1; }
    cublasLtMatmulPreferenceDestroy(pref);
    if (!p->ok) fprintf(stderr, "vw_gemm: no cuBLASLt algorithm for %dx%dx%d batch %d\n", m, n, k, batch);
    return p;
}
extern "C" int vw_gemm(int ta, int tb, int m, int n, int k, const void *A, int lda, long long sa, const void *B, int ldb, long long sb,
                       void *D, int ldd, long long sd, int d32, int batch) {
    vwplan *p = plan_of(ta, tb, m, n, k, lda, sa, ldb, sb, ldd, sd, d32, batch); if (!p || !p->ok) return -1;
    const float one = 1.f, zero = 0.f;
    const cublasStatus_t s = cublasLtMatmul(g_lt, p->op, &one, A, p->A, B, p->B, &zero, D, p->D, D, p->D, &p->algo, g_ws, WS, ST);
    if (s != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "vw_gemm: cublasLtMatmul failed (%d)\n", (int)s); return -1; }
    return 0;
}

/* phase head: logits fp32 [4][D][H][W] (channel s_i 2 + s_j, on the half-resolution transverse grid) ->
   phase [2D][2H][W] = cumsum_w softplus(logit) (and the increments when inc != nullptr) */
__global__ void k_vw_phase(const float *__restrict__ lg, int D, int H, int W, float *__restrict__ ph, float *__restrict__ inc) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x; if (t >= 4 * D * H) return;
    const int oi = t / (2 * H), oj = t % (2 * H), d = oi >> 1, si = oi & 1, h = oj >> 1, sj = oj & 1;
    const float *s = lg + ((size_t)(si * 2 + sj) * D * H + (size_t)d * H + h) * W; float *o = ph + (size_t)t * W, a = 0;
    for (int w = 0; w < W; w++) { const float x = s[w], sp = x > 20.f ? x : log1pf(expf(x)); a += sp; o[w] = a; if (inc) inc[(size_t)t * W + w] = sp; }
}
extern "C" void vw_phase(const float *lg, int D, int H, int W, float *ph, float *inc) { k_vw_phase<<<nb_of(4LL * D * H, 128), 128, 0, ST>>>(lg, D, H, W, ph, inc); }
