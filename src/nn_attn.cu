/* Transformer blocks for the U-Net bottleneck (src/attn.h): fp32 kernels, cuBLASLt fp32 GEMMs, flash-style attention with
   recomputing backward. Default stream throughout (the U-Net's compute stream). */
#include "attn.h"
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HD AT_HD
static unsigned nbk(long long n, int t) { return (unsigned)((n + t - 1) / t); }
static const float LOG2E = 1.4426950408889634f;

/* ---- profiling (env UFSM_ATTN_PROF=1): cudaEvent time per phase of the blocks (synchronising: for measurement only), summed per
   label with the phase's FLOP / byte counts; at_prof_report prints the table (also at exit), at_prof_reset clears it ---- */
typedef struct { const char *nm; double ms, fl, by; long n; } atpr;
static atpr g_pr[64]; static int g_npr = 0; static cudaEvent_t g_pe[2];
extern "C" void at_prof_report(FILE *f) {
    double tot = 0; for (int i = 0; i < g_npr; i++) tot += g_pr[i].ms;
    if (!g_npr || tot <= 0) return;
    fprintf(f, "attention-stage profile (%.1f ms total)\n  %-14s %9s %6s %6s %8s %8s\n", tot, "phase", "ms", "%", "calls", "TFLOP/s", "GB/s");
    for (int i = 0; i < g_npr; i++) { const atpr *p = &g_pr[i];
        fprintf(f, "  %-14s %9.2f %6.1f %6ld %8.1f %8.0f\n", p->nm, p->ms, 100 * p->ms / tot, p->n, p->fl / (p->ms * 1e9), p->by / (p->ms * 1e6)); }
}
extern "C" void at_prof_reset(void) { g_npr = 0; }
static void prof_exit(void) { at_prof_report(stderr); }
static int prof_on(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("UFSM_ATTN_PROF"); v = e && atoi(e); if (v) { cudaEventCreate(&g_pe[0]); cudaEventCreate(&g_pe[1]); atexit(prof_exit); } }
    return v;
}
static void pr_add(const char *nm, double ms, double fl, double by) {
    int i = 0; while (i < g_npr && strcmp(g_pr[i].nm, nm)) i++;
    if (i == g_npr) { if (g_npr == 64) return; g_pr[g_npr++] = (atpr){nm, 0, 0, 0, 0}; }
    g_pr[i].ms += ms; g_pr[i].fl += fl; g_pr[i].by += by; g_pr[i].n++;
}
#define PR(nm, fl, by, ...) do { if (!prof_on()) { __VA_ARGS__; } else { cudaEventRecord(g_pe[0], 0); __VA_ARGS__; cudaEventRecord(g_pe[1], 0); cudaEventSynchronize(g_pe[1]); \
    float ms_ = 0; cudaEventElapsedTime(&ms_, g_pe[0], g_pe[1]); pr_add(nm, ms_, (double)(fl), (double)(by)); } } while (0)

/* ---- GEMM: row-major C[M][N] = op(A) op(B) + beta C; A [M][K] (ta 0) or [K][M] (ta 1); B [K][N] (tb 0) or [N][K] (tb 1).
   cuBLASLt is column-major: C^T (N x M) = op(B)^T op(A)^T, so B is its first operand and A its second ---- */
typedef struct { int M, N, K, ta, tb; cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a, b, c; cublasLtMatmulAlgo_t algo; int ok; } atplan;
static atplan g_ap[256]; static int g_nap = 0;
static cublasLtHandle_t g_alt = nullptr; static void *g_altws = nullptr; static const size_t ALTWS = 32u << 20;
static int tf32_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("UFSM_ATTN_TF32"); v = e && atoi(e); } return v; }
static atplan *at_plan(int M, int N, int K, int ta, int tb) {
    for (int i = 0; i < g_nap; i++) { atplan *p = &g_ap[i]; if (p->M == M && p->N == N && p->K == K && p->ta == ta && p->tb == tb) return p; }
    if (!g_alt) { if (cublasLtCreate(&g_alt) != CUBLAS_STATUS_SUCCESS) return nullptr; if (cudaMalloc(&g_altws, ALTWS) != cudaSuccess) return nullptr; }
    static int next = 0; atplan *p;
    if (g_nap < 256) p = &g_ap[g_nap++];
    else {   /* full (row counts vary per step): recycle the slots round-robin */
        p = &g_ap[next]; next = (next + 1) % 256;
        if (p->op) cublasLtMatmulDescDestroy(p->op);
        if (p->a) cublasLtMatrixLayoutDestroy(p->a);
        if (p->b) cublasLtMatrixLayoutDestroy(p->b);
        if (p->c) cublasLtMatrixLayoutDestroy(p->c);
    }
    memset(p, 0, sizeof *p);
    p->M = M; p->N = N; p->K = K; p->ta = ta; p->tb = tb;
    cublasLtMatmulDescCreate(&p->op, tf32_on() ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F, CUDA_R_32F);
    const cublasOperation_t oX = tb ? CUBLAS_OP_T : CUBLAS_OP_N, oY = ta ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &oX, sizeof oX);
    cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSB, &oY, sizeof oY);
    if (tb) cublasLtMatrixLayoutCreate(&p->a, CUDA_R_32F, K, N, K); else cublasLtMatrixLayoutCreate(&p->a, CUDA_R_32F, N, K, N);
    if (ta) cublasLtMatrixLayoutCreate(&p->b, CUDA_R_32F, M, K, M); else cublasLtMatrixLayoutCreate(&p->b, CUDA_R_32F, K, M, K);
    cublasLtMatrixLayoutCreate(&p->c, CUDA_R_32F, N, M, N);
    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ALTWS, sizeof ALTWS);
    cublasLtMatmulHeuristicResult_t h; int n = 0;
    if (cublasLtMatmulAlgoGetHeuristic(g_alt, p->op, p->a, p->b, p->c, p->c, pref, 1, &h, &n) == CUBLAS_STATUS_SUCCESS && n > 0) { p->algo = h.algo; p->ok = 1; }
    cublasLtMatmulPreferenceDestroy(pref);
    if (!p->ok) fprintf(stderr, "at_gemm: no cuBLASLt algorithm for M %d N %d K %d ta %d tb %d\n", M, N, K, ta, tb);
    return p;
}
static int g_at_mx = -1;   /* the linear layers' GEMM precision: 0 fp32 (cuBLASLt), 4 / 8 MX-fp4 / MX-fp8 (lp_mx_gemm); env UFSM_LINEAR_MX */
static int g_at_core = -1;   /* the attention core: 0 fp32 CUDA cores, 1 fp16 tensor cores (src/nn_attn_tc.cu; the default), 2 MX-fp8 (src/lp_attn8.cu);
                                env UFSM_ATTN_CORE */
extern "C" void at_set_core(int c) { g_at_core = c; }
extern "C" int at_get_core(void) { if (g_at_core < 0) { const char *e = getenv("UFSM_ATTN_CORE"); g_at_core = e ? atoi(e) : 1; } return g_at_core; }
extern "C" void at_set_mx(int bits) { g_at_mx = bits; }
extern "C" int at_get_mx(void) { if (g_at_mx < 0) { const char *e = getenv("UFSM_LINEAR_MX"); g_at_mx = e ? atoi(e) : 0; } return g_at_mx; }
extern "C" int at_gemm_f32(int ta, int tb, int M, int N, int K, const float *A, const float *B, float *Cm, float beta);
extern "C" int at_gemm(int ta, int tb, int M, int N, int K, const float *A, const float *B, float *Cm, float beta) {
    const int mx = at_get_mx();
    if (mx == 4 || mx == 8) { if (M <= 0 || N <= 0) return 0; if (K <= 0) { if (beta == 0.f) cudaMemsetAsync(Cm, 0, (size_t)M * N * 4, 0); return 0; } return lp_mx_gemm(mx, ta, tb, M, N, K, A, B, Cm, beta); }
    return at_gemm_f32(ta, tb, M, N, K, A, B, Cm, beta);
}
extern "C" int at_gemm_f32(int ta, int tb, int M, int N, int K, const float *A, const float *B, float *Cm, float beta) {
    if (M <= 0 || N <= 0) return 0;
    if (K <= 0) { if (beta == 0.f) cudaMemsetAsync(Cm, 0, (size_t)M * N * 4, 0); return 0; }
    atplan *p = at_plan(M, N, K, ta, tb);
    if (!p || !p->ok) return -1;
    const float one = 1.f;
    const cublasStatus_t s = cublasLtMatmul(g_alt, p->op, &one, B, p->a, A, p->b, &beta, Cm, p->c, Cm, p->c, &p->algo, g_altws, ALTWS, 0);
    if (s != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "at_gemm: cublasLtMatmul failed (%d)\n", (int)s); return -1; }
    return 0;
}

/* ---- small elementwise / reduction kernels ---- */
__global__ void k_add_bias(float *y, const float *b, long long T, int N) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < T * N) y[i] += b[i % N];
}
static void add_bias(float *y, const float *b, long long T, int N) { k_add_bias<<<nbk(T * N, 256), 256>>>(y, b, T, N); }
extern "C" int at_gemm_ex(int ta, int tb, int M, int N, int K, const float *A, const float *B, float *Cm, float beta, const float *bias, const float *res) {
    const int mx = at_get_mx();
    if ((mx == 4 || mx == 8) && M > 0 && N > 0 && K > 0) return lp_mx_gemm_ex(mx, ta, tb, M, N, K, A, B, Cm, beta, bias, res);
    if (res) { if (beta != 0.f) return -1; if (res != Cm) cudaMemcpyAsync(Cm, res, (size_t)M * N * 4, cudaMemcpyDeviceToDevice, 0); beta = 1.f; }
    if (at_gemm(ta, tb, M, N, K, A, B, Cm, beta)) return -1;
    if (bias) add_bias(Cm, bias, M, N);
    return 0;
}
/* db[c] += sum over rows of dy[r][c]: blocks of 256 columns x 32 rows */
__global__ void k_colsum(const float *dy, long long T, int N, float *db) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x; const long long r0 = (long long)blockIdx.y * 32;
    if (c >= N) return;
    float s = 0.f; const long long r1 = r0 + 32 < T ? r0 + 32 : T;
    for (long long r = r0; r < r1; r++) s += dy[r * N + c];
    atomicAdd(db + c, s);
}
static void colsum(const float *dy, long long T, int N, float *db) { dim3 g(nbk(N, 256), nbk(T, 32)); k_colsum<<<g, 256>>>(dy, T, N, db); }

/* LayerNorm over C per row: one warp per row */
__global__ void k_ln_fwd(const float *x, const float *g, const float *b, float *y, float *mean, float *rstd, long long T, int C) {
    const long long row = (blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5; const int lane = threadIdx.x & 31;
    if (row >= T) return;
    const float *xr = x + row * C;
    float s = 0.f;
    for (int c = lane; c < C; c += 32) s += xr[c];
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float m = s / C; float q = 0.f;
    for (int c = lane; c < C; c += 32) { const float d = xr[c] - m; q += d * d; }
    for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    const float r = 1.f / sqrtf(q / C + 1e-5f);
    for (int c = lane; c < C; c += 32) y[row * C + c] = (xr[c] - m) * r * g[c] + b[c];
    if (!lane) { mean[row] = m; rstd[row] = r; }
}
extern "C" void at_ln_fwd(const float *x, const float *g, const float *b, float *y, float *mean, float *rstd, long T, int C) {
    k_ln_fwd<<<nbk((long long)T * 32, 256), 256>>>(x, g, b, y, mean, rstd, T, C);
}
/* dx (= or +=) rstd (dy g - mean(dy g) - xhat mean(dy g xhat)); dg += sum dy xhat, db += sum dy. A warp per row (lane: columns
   lane + 32 k, the row in registers), LNB_ROWS rows per block; the gain / bias partials in registers per lane, then shared memory, one
   global atomic per (block, column). NC = ceil(C / 32) */
#define LNB_ROWS 32
template <int NC>
__global__ void __launch_bounds__(256) k_ln_bwd(const float *__restrict__ x, const float *__restrict__ g, const float *__restrict__ mean, const float *__restrict__ rstd,
                                                const float *__restrict__ dy, float *dx, float *dg, float *db, long long T, int C, int acc) {
    __shared__ float sh[2][NC * 32];
    for (int c = threadIdx.x; c < 2 * NC * 32; c += blockDim.x) (&sh[0][0])[c] = 0.f;
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    const long long r0 = (long long)blockIdx.x * LNB_ROWS;
    float ag[NC], ab[NC], gg[NC];
#pragma unroll
    for (int k = 0; k < NC; k++) { ag[k] = 0.f; ab[k] = 0.f; gg[k] = lane + 32 * k < C ? g[lane + 32 * k] : 0.f; }
    for (long long row = r0 + warp; row < r0 + LNB_ROWS && row < T; row += nw) {
        const float m = mean[row], r = rstd[row]; const float *xr = x + row * C, *gr = dy + row * C;
        float xh[NC], gy[NC], a = 0.f, bb = 0.f;
#pragma unroll
        for (int k = 0; k < NC; k++) {
            const int c = lane + 32 * k; const bool in = c < C;
            xh[k] = in ? (xr[c] - m) * r : 0.f; gy[k] = in ? gr[c] : 0.f;
            const float gyg = gy[k] * gg[k]; a += gyg; bb += gyg * xh[k]; ag[k] += gy[k] * xh[k]; ab[k] += gy[k];
        }
        for (int o = 16; o; o >>= 1) { a += __shfl_xor_sync(0xffffffffu, a, o); bb += __shfl_xor_sync(0xffffffffu, bb, o); }
        a /= C; bb /= C;
#pragma unroll
        for (int k = 0; k < NC; k++) {
            const int c = lane + 32 * k; if (c >= C) continue;
            const float v = r * (gy[k] * gg[k] - a - xh[k] * bb);
            if (acc) dx[row * C + c] += v; else dx[row * C + c] = v;
        }
    }
#pragma unroll
    for (int k = 0; k < NC; k++) if (lane + 32 * k < C) { atomicAdd(&sh[0][lane + 32 * k], ag[k]); atomicAdd(&sh[1][lane + 32 * k], ab[k]); }
    __syncthreads();
    for (int c = threadIdx.x; c < C; c += blockDim.x) { atomicAdd(dg + c, sh[0][c]); atomicAdd(db + c, sh[1][c]); }
}
extern "C" void at_ln_bwd(const float *x, const float *g, const float *mean, const float *rstd, const float *dy, float *dx, float *dg, float *db, long T, int C, int acc) {
    const unsigned nb = nbk(T, LNB_ROWS);
    if (C <= 128) k_ln_bwd<4><<<nb, 256>>>(x, g, mean, rstd, dy, dx, dg, db, T, C, acc);
    else if (C <= 256) k_ln_bwd<8><<<nb, 256>>>(x, g, mean, rstd, dy, dx, dg, db, T, C, acc);
    else if (C <= 512) k_ln_bwd<16><<<nb, 256>>>(x, g, mean, rstd, dy, dx, dg, db, T, C, acc);
    else if (C <= 1024) k_ln_bwd<32><<<nb, 256>>>(x, g, mean, rstd, dy, dx, dg, db, T, C, acc);
    else { fprintf(stderr, "at_ln_bwd: width %d > 1024\n", C); abort(); }
}

/* QK-norm: q and k of each head scaled to unit RMS over its 48 dims, times a gain per channel; v copied. One thread per (token, head) */
__global__ void k_qkn_fwd(const float *qkv, const float *qg, const float *kg, float *qkn, float *rq, float *rk, long long T, int C) {
    const int H = C / HD; const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= T * H) return;
    const long long t = i / H; const int h = (int)(i % H);
    const float *row = qkv + t * 3 * C; float *orow = qkn + t * 3 * C;
    for (int p = 0; p < 2; p++) {
        const float *xs = row + p * C + h * HD; const float *gs = (p ? kg : qg) + h * HD;
        float s = 0.f;
#pragma unroll
        for (int d = 0; d < HD; d++) s += xs[d] * xs[d];
        const float r = 1.f / sqrtf(s / HD + 1e-6f);
#pragma unroll
        for (int d = 0; d < HD; d++) orow[p * C + h * HD + d] = xs[d] * r * gs[d];
        (p ? rk : rq)[i] = r;
    }
#pragma unroll
    for (int d = 0; d < HD; d++) orow[2 * C + h * HD + d] = row[2 * C + h * HD + d];
}
extern "C" void at_qknorm_fwd(const float *qkv, const float *qg, const float *kg, float *qkn, float *rq, float *rk, long T, int C) {
    k_qkn_fwd<<<nbk((long long)T * (C / HD), 128), 128>>>(qkv, qg, kg, qkn, rq, rk, T, C);
}
/* dx_d = r g_d dy_d - x_d r^3 / 48 sum_e g_e dy_e x_e (q and k); v passes through. One warp per (token, head, q / k / v): lanes over
   dims d = lane, lane + 32 (< 48) */
__global__ void k_qkn_bwd(const float *qkv, const float *qg, const float *kg, const float *rq, const float *rk, const float *dqkn, float *dqkv, long long T, int C) {
    const int H = C / HD; const long long w = (blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5; const int lane = threadIdx.x & 31;
    if (w >= T * H * 3) return;
    const int p = (int)(w % 3); const long long th = w / 3; const int h = (int)(th % H); const long long t = th / H;
    const long long o = t * 3 * C + p * C + h * HD; const bool two = lane < HD - 32;
    const float d0 = dqkn[o + lane], d1 = two ? dqkn[o + lane + 32] : 0.f;
    if (p == 2) { dqkv[o + lane] = d0; if (two) dqkv[o + lane + 32] = d1; return; }
    const float *gs = (p ? kg : qg) + h * HD; const float r = (p ? rk : rq)[th];
    const float x0 = qkv[o + lane], x1 = two ? qkv[o + lane + 32] : 0.f, g0 = gs[lane], g1 = two ? gs[lane + 32] : 0.f;
    float s = g0 * d0 * x0 + g1 * d1 * x1;
    for (int k = 16; k; k >>= 1) s += __shfl_xor_sync(0xffffffffu, s, k);
    const float k3 = r * r * r * s / HD;
    dqkv[o + lane] = r * g0 * d0 - x0 * k3; if (two) dqkv[o + lane + 32] = r * g1 * d1 - x1 * k3;
}
/* gain gradients: dg[c] += sum_t dy[t][c] x[t][c] r[t][head(c)], c over the q (p 0) and k (p 1) channels: blocks of 64 tokens, threads
   over the 2C channels (coalesced rows), one atomic per (block, channel) */
#define QDG_ROWS 16
__global__ void k_qkn_dg(const float *qkv, const float *rq, const float *rk, const float *dqkn, float *dqg, float *dkg, long long T, int C) {
    const int H = C / HD; const long long t0 = (long long)blockIdx.x * QDG_ROWS, t1 = t0 + QDG_ROWS < T ? t0 + QDG_ROWS : T;
    for (int c2 = threadIdx.x; c2 < 2 * C; c2 += blockDim.x) {
        const int p = c2 / C, c = c2 % C, h = c / HD; const float *r = p ? rk : rq;
        float s = 0.f;
        for (long long t = t0; t < t1; t++) { const long long o = t * 3 * C + c2; s += dqkn[o] * qkv[o] * r[t * H + h]; }
        atomicAdd((p ? dkg : dqg) + c, s);
    }
}
extern "C" void at_qknorm_bwd(const float *qkv, const float *qg, const float *kg, const float *rq, const float *rk, const float *dqkn, float *dqkv, float *dqg, float *dkg, long T, int C) {
    k_qkn_bwd<<<nbk((long long)T * (C / HD) * 3 * 32, 256), 256>>>(qkv, qg, kg, rq, rk, dqkn, dqkv, T, C);
    k_qkn_dg<<<nbk(T, QDG_ROWS), 256>>>(qkv, rq, rk, dqkn, dqg, dkg, T, C);
}

/* axial 3D RoPE on the grid tokens: head dims [16 a, 16 a + 16) rotate with the token's coordinate along axis a (z, y, x) in 8
   pairs (2i, 2i + 1) at frequencies base^(-i / 8); register tokens unrotated. inverse: the transposed rotation (backward) */
__global__ void k_rope(float *qkn, int nseq, int S, int D, int Hh, int W, int nreg, int C, float lbase, int inverse) {
    const int H = C / HD; const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    const long long n = (long long)nseq * S * H * 24;
    if (i >= n) return;
    const int pr = (int)(i % 24); const long long th = i / 24; const int h = (int)(th % H); const long long t = th / H; const int ts = (int)(t % S) - nreg;
    if (ts < 0) return;
    const int ax = pr / 8, fi = pr % 8;
    const int pos = ax == 0 ? ts / (Hh * W) : ax == 1 ? (ts / W) % Hh : ts % W;
    const float ang = (float)pos * exp2f(-lbase * (float)fi / 8.f);
    float sn, cs; sincosf(ang, &sn, &cs); if (inverse) sn = -sn;
    for (int p = 0; p < 2; p++) {
        float *v = qkn + t * 3 * C + p * C + h * HD + ax * 16 + 2 * fi;
        const float a = v[0], b = v[1];
        v[0] = a * cs - b * sn; v[1] = a * sn + b * cs;
    }
}
extern "C" void at_rope(float *qkn, const at_geom *g, int nreg, int C, float base, int inverse) {
    const long long n = (long long)g->nseq * g->S * (C / HD) * 24;
    k_rope<<<nbk(n, 256), 256>>>(qkn, g->nseq, g->S, g->D, g->Hh, g->W, nreg, C, log2f(base), inverse);
}

/* ---- attention, head dim 48, fp32. Logits in log2 units: s = q.k scale log2(e); lse2 = max + log2(sum exp2) per row ---- */
#define AQ 64
#define AK 32
__global__ void __launch_bounds__(AQ) k_attn_fwd(const float *__restrict__ qkn, float *__restrict__ o, float *__restrict__ lse, int S, int H, int C, float sl2) {
    const int seq = blockIdx.z, h = blockIdx.y, i = blockIdx.x * AQ + threadIdx.x; const int ld = 3 * C;
    const size_t base = (size_t)seq * S;
    __shared__ float ks[AK][HD], vs[AK][HD];
    const bool act = i < S;
    float q[HD], acc[HD];
#pragma unroll
    for (int d = 0; d < HD; d++) { q[d] = act ? qkn[(base + i) * ld + h * HD + d] * sl2 : 0.f; acc[d] = 0.f; }
    float m = -INFINITY, l = 0.f;
    for (int j0 = 0; j0 < S; j0 += AK) {
        __syncthreads();
        for (int e = threadIdx.x; e < AK * HD; e += AQ) {
            const int jj = e / HD, d = e % HD, j = j0 + jj;
            ks[jj][d] = j < S ? qkn[(base + j) * ld + C + h * HD + d] : 0.f;
            vs[jj][d] = j < S ? qkn[(base + j) * ld + 2 * C + h * HD + d] : 0.f;
        }
        __syncthreads();
        const int nj = S - j0 < AK ? S - j0 : AK;
        float s[AK]; float mt = m;
#pragma unroll
        for (int jj = 0; jj < AK; jj++) {
            float a = 0.f;
#pragma unroll
            for (int d = 0; d < HD; d++) a += q[d] * ks[jj][d];
            s[jj] = jj < nj ? a : -INFINITY;
            mt = fmaxf(mt, s[jj]);
        }
        const float corr = exp2f(m - mt);
        l *= corr;
#pragma unroll
        for (int d = 0; d < HD; d++) acc[d] *= corr;
#pragma unroll
        for (int jj = 0; jj < AK; jj++) {
            const float p = exp2f(s[jj] - mt); l += p;
#pragma unroll
            for (int d = 0; d < HD; d++) acc[d] += p * vs[jj][d];
        }
        m = mt;
    }
    if (!act) return;
    const float il = 1.f / l;
#pragma unroll
    for (int d = 0; d < HD; d++) o[(base + i) * C + h * HD + d] = acc[d] * il;
    lse[((size_t)seq * H + h) * S + i] = m + log2f(l);
}
extern "C" void at_attn_fwd(const float *qkn, float *o, float *lse, const at_geom *g, int H, int C) {
    dim3 gr(nbk(g->S, AQ), H, g->nseq);
    k_attn_fwd<<<gr, AQ>>>(qkn, o, lse, g->S, H, C, LOG2E / sqrtf((float)HD));
}
/* D[row, head] = dO . O over the head's dims */
__global__ void k_attn_dvec(const float *o, const float *dO, float *dv, int nseq, int S, int H, int C) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= (long long)nseq * S * H) return;
    const int h = (int)(i % H); const long long t = i / H; const int seq = (int)(t / S), s = (int)(t % S);
    float a = 0.f;
#pragma unroll
    for (int d = 0; d < HD; d++) a += o[t * C + h * HD + d] * dO[t * C + h * HD + d];
    dv[((size_t)seq * H + h) * S + s] = a;
}
/* dK, dV: one thread per key (its k, v rows in padded shared memory), query tiles of AK broadcast */
#define KB 64
__global__ void __launch_bounds__(KB) k_attn_dkv(const float *__restrict__ qkn, const float *__restrict__ lse, const float *__restrict__ dvv, const float *__restrict__ dO,
                                                float *__restrict__ dqkn, int S, int H, int C, float sl2, float scale) {
    const int seq = blockIdx.z, h = blockIdx.y, j = blockIdx.x * KB + threadIdx.x; const int ld = 3 * C;
    const size_t base = (size_t)seq * S;
    __shared__ float kS[KB][HD + 1], vS[KB][HD + 1], qT[AK][HD], gT[AK][HD], lT[AK], dT[AK];
    const bool act = j < S;
    for (int d = 0; d < HD; d++) { kS[threadIdx.x][d] = act ? qkn[(base + j) * ld + C + h * HD + d] : 0.f; vS[threadIdx.x][d] = act ? qkn[(base + j) * ld + 2 * C + h * HD + d] : 0.f; }
    float dk[HD], dv[HD];
#pragma unroll
    for (int d = 0; d < HD; d++) { dk[d] = 0.f; dv[d] = 0.f; }
    const float *L = lse + ((size_t)seq * H + h) * S, *Dv = dvv + ((size_t)seq * H + h) * S;
    for (int i0 = 0; i0 < S; i0 += AK) {
        __syncthreads();
        for (int e = threadIdx.x; e < AK * HD; e += KB) {
            const int ii = e / HD, d = e % HD, i = i0 + ii;
            qT[ii][d] = i < S ? qkn[(base + i) * ld + h * HD + d] : 0.f;
            gT[ii][d] = i < S ? dO[(base + i) * C + h * HD + d] : 0.f;
        }
        if (threadIdx.x < AK) { const int i = i0 + threadIdx.x; lT[threadIdx.x] = i < S ? L[i] : INFINITY; dT[threadIdx.x] = i < S ? Dv[i] : 0.f; }
        __syncthreads();
        const int ni = S - i0 < AK ? S - i0 : AK;
        for (int ii = 0; ii < ni; ii++) {
            float s = 0.f, dp = 0.f;
#pragma unroll
            for (int d = 0; d < HD; d++) { s += qT[ii][d] * kS[threadIdx.x][d]; dp += gT[ii][d] * vS[threadIdx.x][d]; }
            const float p = exp2f(s * sl2 - lT[ii]), ds = p * (dp - dT[ii]);
#pragma unroll
            for (int d = 0; d < HD; d++) { dv[d] += p * gT[ii][d]; dk[d] += ds * qT[ii][d]; }
        }
    }
    if (!act) return;
#pragma unroll
    for (int d = 0; d < HD; d++) { dqkn[(base + j) * ld + C + h * HD + d] = dk[d] * scale; dqkn[(base + j) * ld + 2 * C + h * HD + d] = dv[d]; }
}
/* dQ: one thread per query (its q, dO rows in padded shared memory), key tiles broadcast */
__global__ void __launch_bounds__(KB) k_attn_dq(const float *__restrict__ qkn, const float *__restrict__ lse, const float *__restrict__ dvv, const float *__restrict__ dO,
                                               float *__restrict__ dqkn, int S, int H, int C, float sl2, float scale) {
    const int seq = blockIdx.z, h = blockIdx.y, i = blockIdx.x * KB + threadIdx.x; const int ld = 3 * C;
    const size_t base = (size_t)seq * S;
    __shared__ float qS[KB][HD + 1], gS[KB][HD + 1], kT[AK][HD], vT[AK][HD];
    const bool act = i < S;
    for (int d = 0; d < HD; d++) { qS[threadIdx.x][d] = act ? qkn[(base + i) * ld + h * HD + d] : 0.f; gS[threadIdx.x][d] = act ? dO[(base + i) * C + h * HD + d] : 0.f; }
    const float li = act ? lse[((size_t)seq * H + h) * S + i] : INFINITY, Di = act ? dvv[((size_t)seq * H + h) * S + i] : 0.f;
    float dq[HD];
#pragma unroll
    for (int d = 0; d < HD; d++) dq[d] = 0.f;
    for (int j0 = 0; j0 < S; j0 += AK) {
        __syncthreads();
        for (int e = threadIdx.x; e < AK * HD; e += KB) {
            const int jj = e / HD, d = e % HD, j = j0 + jj;
            kT[jj][d] = j < S ? qkn[(base + j) * ld + C + h * HD + d] : 0.f;
            vT[jj][d] = j < S ? qkn[(base + j) * ld + 2 * C + h * HD + d] : 0.f;
        }
        __syncthreads();
        const int nj = S - j0 < AK ? S - j0 : AK;
        for (int jj = 0; jj < nj; jj++) {
            float s = 0.f, dp = 0.f;
#pragma unroll
            for (int d = 0; d < HD; d++) { s += qS[threadIdx.x][d] * kT[jj][d]; dp += gS[threadIdx.x][d] * vT[jj][d]; }
            const float p = exp2f(s * sl2 - li), ds = p * (dp - Di);
#pragma unroll
            for (int d = 0; d < HD; d++) dq[d] += ds * kT[jj][d];
        }
    }
    if (!act) return;
#pragma unroll
    for (int d = 0; d < HD; d++) dqkn[(base + i) * ld + h * HD + d] = dq[d] * scale;
}
extern "C" void at_attn_bwd(const float *qkn, const float *o, const float *lse, const float *dO, float *dv, float *dqkn, const at_geom *g, int H, int C) {
    const float scale = 1.f / sqrtf((float)HD), sl2 = scale * LOG2E;
    k_attn_dvec<<<nbk((long long)g->nseq * g->S * H, 256), 256>>>(o, dO, dv, g->nseq, g->S, H, C);
    dim3 gr(nbk(g->S, KB), H, g->nseq);
    k_attn_dkv<<<gr, KB>>>(qkn, lse, dv, dO, dqkn, g->S, H, C, sl2, scale);
    k_attn_dq<<<gr, KB>>>(qkn, lse, dv, dO, dqkn, g->S, H, C, sl2, scale);
}

/* SwiGLU: h [T][2F] = [a | b], gg = silu(a) b */
__global__ void k_glu_fwd(const float *h, float *gg, long long T, int F) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= T * F) return;
    const long long t = i / F; const int f = (int)(i % F);
    const float a = h[t * 2 * F + f], b = h[t * 2 * F + F + f];
    gg[i] = a / (1.f + expf(-a)) * b;
}
__global__ void k_glu_bwd(const float *h, const float *dgg, float *dh, long long T, int F) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= T * F) return;
    const long long t = i / F; const int f = (int)(i % F);
    const float a = h[t * 2 * F + f], b = h[t * 2 * F + F + f], sg = 1.f / (1.f + expf(-a)), g = dgg[i];
    dh[t * 2 * F + f] = g * b * sg * (1.f + a * (1.f - sg));
    dh[t * 2 * F + F + f] = g * a * sg;
}
extern "C" void at_glu_fwd(const float *h, float *gg, long T, int F) { k_glu_fwd<<<nbk((long long)T * F, 256), 256>>>(h, gg, T, F); }
extern "C" void at_glu_bwd(const float *h, const float *dgg, float *dh, long T, int F) { k_glu_bwd<<<nbk((long long)T * F, 256), 256>>>(h, dgg, dh, T, F); }

/* ---- parameters and scratch ---- */
extern "C" size_t at_block_nparams(const at_dims *d) {
    const size_t C = d->C, F = d->F;
    return 2 * C + 3 * C * C + 3 * C + 2 * C + C * C + C + 2 * C + 2 * F * C + 2 * F + C * F + C;
}
extern "C" size_t at_block_layout(const at_dims *d, at_bp *bp, size_t off) {
    const size_t C = d->C, F = d->F;
    bp->ln1g = off; off += C; bp->ln1b = off; off += C;
    bp->wqkv = off; off += 3 * C * C; bp->bqkv = off; off += 3 * C;
    bp->qg = off; off += C; bp->kg = off; off += C;
    bp->wo = off; off += C * C; bp->bo = off; off += C;
    bp->ln2g = off; off += C; bp->ln2b = off; off += C;
    bp->w1 = off; off += 2 * F * C; bp->b1 = off; off += 2 * F;
    bp->w2 = off; off += C * F; bp->b2 = off; off += C;
    return off;
}
static size_t al(size_t n) { return (n + 63) & ~(size_t)63; }   /* floats, 256-byte aligned pieces */
/* the forward part (what the backward reads; kept per block through at_ws_set_fwd) from fb, the backward scratch from bb; either may
   be nullptr (sizing only); returns the floats of part (0 forward, 1 backward) */
static size_t ws_walk(at_ws *w, const at_dims *d, long T, float *fb, float *bb, int part) {
    const size_t C = d->C, F = d->F, H = d->H, t = (size_t)T, wa = 2 * F > 3 * C ? 2 * F : 3 * C, wb = F > 3 * C ? F : 3 * C;
    at_ws tmp; if (!w) w = &tmp;
    size_t nf = 0, nb = 0;
#define TF(f, n) do { w->f = fb ? fb + nf : nullptr; nf += al(n); } while (0)
#define TB(f, n) do { w->f = bb ? bb + nb : nullptr; nb += al(n); } while (0)
    TF(xn, t * C); TF(o, t * C); TF(x1, t * C); TF(xn2, t * C); TF(qkv, t * 3 * C); TF(h, t * 2 * F);
    TF(m1, t); TF(r1, t); TF(m2, t); TF(r2, t); TF(rq, t * H); TF(rk, t * H); TF(lse, t * H);
    TF(qkvh, 3 * C * (t + 64) / 2); TF(amx, 4);   /* (fp16: halves) */
    TB(bC, t * C); TB(qkn, t * 3 * C); TB(gg, t * F);   /* (qkn: core 0 only, which always recomputes; gg: recomputed from h when kept) */
    TB(dx1, t * C); TB(dv, t * H); TB(bA, t * wa); TB(bB, t * wb); TB(doh, C * (t + 64) / 2);
#undef TF
#undef TB
    return part ? nb : nf;
}
extern "C" size_t at_ws_bytes(const at_dims *d, long T) { return (ws_walk(nullptr, d, T, nullptr, nullptr, 0) + ws_walk(nullptr, d, T, nullptr, nullptr, 1)) * 4; }
extern "C" void at_ws_carve(at_ws *w, const at_dims *d, long T, void *base) {
    const size_t nf = ws_walk(nullptr, d, T, nullptr, nullptr, 0);
    ws_walk(w, d, T, (float *)base, (float *)base + nf, 0); w->kept = 0; w->vr = 0;
}
extern "C" size_t at_ws_fwd_bytes(const at_dims *d, long T) { return ws_walk(nullptr, d, T, nullptr, nullptr, 0) * 4; }
extern "C" void at_ws_set_fwd(at_ws *w, const at_dims *d, long T, void *base) {
    at_ws t; ws_walk(&t, d, T, (float *)base, nullptr, 0);
    w->xn = t.xn; w->o = t.o; w->x1 = t.x1; w->xn2 = t.xn2; w->qkv = t.qkv; w->h = t.h;
    w->m1 = t.m1; w->r1 = t.r1; w->m2 = t.m2; w->r2 = t.r2; w->rq = t.rq; w->rk = t.rk; w->lse = t.lse; w->qkvh = t.qkvh; w->amx = t.amx; w->kept = 0;
}

/* ---- value residual (at_ws.vr) ---- */
__global__ void k_vr_first(const float *qkv, float *v1, long long T, int C) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i >= T * C) return;
    v1[i] = qkv[(i / C) * 3 * C + 2 * C + i % C];
}
__global__ void k_vr_mix(float *qkv, const float *v1, float *own, const float *lam, long long T, int C) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i >= T * C) return;
    const int c = (int)(i % C); float *v = qkv + (i / C) * 3 * C + 2 * C + c; const float o = *v; own[i] = o; *v = o + lam[c / HD] * (v1[i] - o);
}
__global__ void k_vr_add(float *dqkv, const float *dv1, long long T, int C) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i >= T * C) return;
    dqkv[(i / C) * 3 * C + 2 * C + i % C] += dv1[i];
}
/* one block per 64 tokens, threads over channels; one atomic per (block, head) via shared memory */
__global__ void k_vr_bwd(float *dqkv, const float *v1, const float *own, float *dv1, const float *lam, float *glam, long long T, int C) {
    __shared__ float sh[64]; const int H = C / HD;
    for (int h = threadIdx.x; h < H; h += blockDim.x) sh[h] = 0.f;
    __syncthreads();
    const long long t0 = (long long)blockIdx.x * 64, t1 = t0 + 64 < T ? t0 + 64 : T;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        const float l = lam[c / HD]; float s = 0.f;
        for (long long t = t0; t < t1; t++) { const long long i = t * C + c; float *g = dqkv + t * 3 * C + 2 * C + c; const float d = *g;
            s += d * (v1[i] - own[i]); dv1[i] += l * d; *g = (1.f - l) * d; }
        atomicAdd(sh + c / HD, s);
    }
    __syncthreads();
    for (int h = threadIdx.x; h < H; h += blockDim.x) atomicAdd(glam + h, sh[h]);
}

/* ---- one block ---- */
#define CK(x) do { if (x) return -1; } while (0)
extern "C" int at_block_fwd(const at_dims *d, const float *P, const at_bp *bp, const float *x, float *y, const at_geom *g, at_ws *w) {
    const long T = (long)g->nseq * g->S; const int C = d->C, F = d->F; const double t = (double)T, A = 4.0 * g->nseq * (double)g->S * g->S * HD * d->H;
    int e = 0;
    PR("ln1", 0, t * C * 8, at_ln_fwd(x, P + bp->ln1g, P + bp->ln1b, w->xn, w->m1, w->r1, T, C));
    PR("qkv gemm", 2 * t * 3 * C * C, t * C * 16, e |= at_gemm_ex(0, 1, (int)T, 3 * C, C, w->xn, P + bp->wqkv, w->qkv, 0.f, P + bp->bqkv, nullptr)); CK(e);
    if (w->vr == 1) k_vr_first<<<nbk((long long)T * C, 256), 256>>>(w->qkv, w->v1, T, C);
    else if (w->vr == 2) k_vr_mix<<<nbk((long long)T * C, 256), 256>>>(w->qkv, w->v1, w->own, w->lam, T, C);
    if (at_get_core()) {
        PR("qknorm+pack", 0, t * C * 18, at_fa_pack(w->qkv, P + bp->qg, P + bp->kg, w->qkvh, w->rq, w->rk, w->amx, g, d->nreg, C, d->rope, d->rope_base));
        if (at_get_core() == 2) PR("attn fwd", A, t * C * 6, e |= at_fa8_fwd(w->qkvh, w->o, w->lse, g, d->H, C));
        else PR("attn fwd", A, t * C * 10, at_fa_fwd(w->qkvh, w->o, w->lse, g, d->H, C));
        CK(e);
    } else {
        PR("qknorm+rope", 0, t * C * 24, at_qknorm_fwd(w->qkv, P + bp->qg, P + bp->kg, w->qkn, w->rq, w->rk, T, C); if (d->rope) at_rope(w->qkn, g, d->nreg, C, d->rope_base, 0));
        PR("attn fwd", A, t * C * 16, at_attn_fwd(w->qkn, w->o, w->lse, g, d->H, C));
    }
    PR("o gemm+res", 2 * t * C * C, t * C * 16, 
       e |= at_gemm_ex(0, 1, (int)T, C, C, w->o, P + bp->wo, w->x1, 0.f, P + bp->bo, x)); CK(e);
    PR("ln2", 0, t * C * 8, at_ln_fwd(w->x1, P + bp->ln2g, P + bp->ln2b, w->xn2, w->m2, w->r2, T, C));
    PR("w1 gemm", 2 * t * 2 * F * C, t * (C + 4 * F) * 4, e |= at_gemm_ex(0, 1, (int)T, 2 * F, C, w->xn2, P + bp->w1, w->h, 0.f, P + bp->b1, nullptr)); CK(e);
    PR("swiglu", 0, t * F * 12, at_glu_fwd(w->h, w->gg, T, F));
    PR("w2 gemm+res", 2 * t * C * F, t * (F + 3 * C) * 4, 
       e |= at_gemm_ex(0, 1, (int)T, C, F, w->gg, P + bp->w2, y, 0.f, P + bp->b2, w->x1)); CK(e);
    return 0;
}
extern "C" int at_block_bwd(const at_dims *d, const float *P, float *G, const at_bp *bp, const float *xin, float *gy, const at_geom *g, at_ws *w) {
    const long T = (long)g->nseq * g->S; const int C = d->C, F = d->F; const double t = (double)T, A = 4.0 * g->nseq * (double)g->S * g->S * HD * d->H;
    int e = 0;
    /* recompute the block's internals (its output into scratch), unless kept (core 0 always recomputes: its q / k / v rows are scratch);
       kept: the SwiGLU product from h (scratch) */
    if (!w->kept || !at_get_core()) CK(at_block_fwd(d, P, bp, xin, w->bA, g, w));
    else at_glu_fwd(w->h, w->gg, T, F);
    w->kept = 0;
    /* MLP */
    PR("b w2 gemms", 4 * t * C * F, t * (2 * F + C) * 8, e |= at_gemm(1, 0, C, F, (int)T, gy, w->gg, G + bp->w2, 1.f); colsum(gy, T, C, G + bp->b2);
       e |= at_gemm(0, 0, (int)T, F, C, gy, P + bp->w2, w->bB, 0.f)); CK(e);                 /* d gg */
    PR("b swiglu", 0, t * F * 20, at_glu_bwd(w->h, w->bB, w->bA, T, F));                    /* d h */
    PR("b w1 gemms", 8 * t * C * F, t * (2 * F + C) * 12, e |= at_gemm(1, 0, 2 * F, C, (int)T, w->bA, w->xn2, G + bp->w1, 1.f); colsum(w->bA, T, 2 * F, G + bp->b1);
       e |= at_gemm(0, 0, (int)T, C, 2 * F, w->bA, P + bp->w1, w->bC, 0.f)); CK(e);          /* d xn2 */
    PR("b ln2", 0, t * C * 20, cudaMemcpyAsync(w->dx1, gy, (size_t)T * C * 4, cudaMemcpyDeviceToDevice, 0);
       at_ln_bwd(w->x1, P + bp->ln2g, w->m2, w->r2, w->bC, w->dx1, G + bp->ln2g, G + bp->ln2b, T, C, 1));
    /* attention */
    PR("b o gemms", 4 * t * C * C, t * C * 20, e |= at_gemm(1, 0, C, C, (int)T, w->dx1, w->o, G + bp->wo, 1.f); colsum(w->dx1, T, C, G + bp->bo);
       e |= at_gemm(0, 0, (int)T, C, C, w->dx1, P + bp->wo, w->bC, 0.f)); CK(e);             /* d o */
    if (at_get_core() == 2) { PR("attn bwd", 3.5 * A, t * C * 20, e |= at_fa8_bwd(w->qkvh, w->doh, w->o, w->lse, w->bC, w->dv, w->amx, w->bA, g, d->H, C)); CK(e); }
    else if (at_get_core()) PR("attn bwd", 3.5 * A, t * C * 30, at_fa_bwd(w->qkvh, w->doh, w->o, w->lse, w->bC, w->dv, w->amx, w->bA, g, d->H, C));
    else PR("attn bwd", 3.5 * A, t * C * 40, at_attn_bwd(w->qkn, w->o, w->lse, w->bC, w->dv, w->bA, g, d->H, C));   /* d qkn */
    PR("b rope+qknorm", 0, t * C * 36, if (d->rope) at_rope(w->bA, g, d->nreg, C, d->rope_base, 1);
       at_qknorm_bwd(w->qkv, P + bp->qg, P + bp->kg, w->rq, w->rk, w->bA, w->bB, G + bp->qg, G + bp->kg, T, C));
    if (w->vr == 2) k_vr_bwd<<<(unsigned)((T + 63) / 64), 128>>>(w->bB, w->v1, w->own, w->dv1, w->lam, w->glam, T, C);
    else if (w->vr == 1) k_vr_add<<<nbk((long long)T * C, 256), 256>>>(w->bB, w->dv1, T, C);
    PR("b qkv gemms", 12 * t * C * C, t * C * 40, e |= at_gemm(1, 0, 3 * C, C, (int)T, w->bB, w->xn, G + bp->wqkv, 1.f); colsum(w->bB, T, 3 * C, G + bp->bqkv);
       e |= at_gemm(0, 0, (int)T, C, 3 * C, w->bB, P + bp->wqkv, w->bC, 0.f)); CK(e);        /* d xn */
    PR("b ln1", 0, t * C * 20, cudaMemcpyAsync(gy, w->dx1, (size_t)T * C * 4, cudaMemcpyDeviceToDevice, 0);
       at_ln_bwd(xin, P + bp->ln1g, w->m1, w->r1, w->bC, gy, G + bp->ln1g, G + bp->ln1b, T, C, 1));
    return 0;
}

/* ---- PEG: depthwise 3^3 conv on the grid tokens ---- */
__global__ void k_peg_fwd(const float *x, float *y, const float *w, const float *b, int nseq, int S, int D, int Hh, int W, int nreg, int C) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= (long long)nseq * S * C) return;
    const int c = (int)(i % C); const long long t = i / C; const int seq = (int)(t / S), ts = (int)(t % S) - nreg;
    if (ts < 0) { y[i] = x[i]; return; }
    const int z = ts / (Hh * W), yy = (ts / W) % Hh, xx = ts % W;
    float a = x[i] + b[c];
    for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
        const int zz = z + dz, y2 = yy + dy, x2 = xx + dx;
        if (zz < 0 || zz >= D || y2 < 0 || y2 >= Hh || x2 < 0 || x2 >= W) continue;
        const long long tn = (long long)seq * S + nreg + ((long long)zz * Hh + y2) * W + x2;
        a += w[c * 27 + (dz + 1) * 9 + (dy + 1) * 3 + dx + 1] * x[tn * C + c];
    }
    y[i] = a;
}
__global__ void k_peg_bwd_x(const float *gy, float *gx, const float *w, int nseq, int S, int D, int Hh, int W, int nreg, int C) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= (long long)nseq * S * C) return;
    const int c = (int)(i % C); const long long t = i / C; const int seq = (int)(t / S), ts = (int)(t % S) - nreg;
    if (ts < 0) { gx[i] = gy[i]; return; }
    const int z = ts / (Hh * W), yy = (ts / W) % Hh, xx = ts % W;
    float a = gy[i];
    for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {   /* y[t - o] read x[t] through tap o */
        const int zz = z - dz, y2 = yy - dy, x2 = xx - dx;
        if (zz < 0 || zz >= D || y2 < 0 || y2 >= Hh || x2 < 0 || x2 >= W) continue;
        const long long tn = (long long)seq * S + nreg + ((long long)zz * Hh + y2) * W + x2;
        a += w[c * 27 + (dz + 1) * 9 + (dy + 1) * 3 + dx + 1] * gy[tn * C + c];
    }
    gx[i] = a;
}
/* gw[c][tap] += sum_t gy[t][c] x[t + o_tap][c]; tap 27: gb[c] += sum_t gy[t][c]. grid (28, nseq), threads over channels */
__global__ void k_peg_bwd_w(const float *x, const float *gy, float *gw, float *gb, int S, int D, int Hh, int W, int nreg, int C) {
    const int tap = blockIdx.x, seq = blockIdx.y;
    const int dz = tap / 9 - 1, dy = (tap / 3) % 3 - 1, dx = tap % 3 - 1;
    const long long Sg = (long long)D * Hh * W;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float a = 0.f;
        for (long long ts = 0; ts < Sg; ts++) {
            const long long t = (long long)seq * S + nreg + ts;
            if (tap == 27) { a += gy[t * C + c]; continue; }
            const int z = (int)(ts / (Hh * W)), yy = (int)((ts / W) % Hh), xx = (int)(ts % W);
            const int zz = z + dz, y2 = yy + dy, x2 = xx + dx;
            if (zz < 0 || zz >= D || y2 < 0 || y2 >= Hh || x2 < 0 || x2 >= W) continue;
            const long long tn = (long long)seq * S + nreg + ((long long)zz * Hh + y2) * W + x2;
            a += gy[t * C + c] * x[tn * C + c];
        }
        if (tap == 27) atomicAdd(gb + c, a); else atomicAdd(gw + c * 27 + tap, a);
    }
}
extern "C" void at_peg_fwd(const float *x, float *y, const float *w, const float *b, const at_geom *g, int nreg, int C) {
    k_peg_fwd<<<nbk((long long)g->nseq * g->S * C, 256), 256>>>(x, y, w, b, g->nseq, g->S, g->D, g->Hh, g->W, nreg, C);
}
extern "C" void at_peg_bwd(const float *x, const float *gy, float *gx, const float *w, float *gw, float *gb, const at_geom *g, int nreg, int C) {
    k_peg_bwd_x<<<nbk((long long)g->nseq * g->S * C, 256), 256>>>(gy, gx, w, g->nseq, g->S, g->D, g->Hh, g->W, nreg, C);
    dim3 gr(28, g->nseq);
    k_peg_bwd_w<<<gr, C < 256 ? C : 256>>>(x, gy, gw, gb, g->S, g->D, g->Hh, g->W, nreg, C);
}

/* ---- layouts ---- */
__global__ void k_cs_to_tok(const float *x, float *t, const float *reg, int n, int C, long long Sg, int nreg) {
    const long long S = Sg + nreg, i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= (long long)n * S * C) return;
    const int c = (int)(i % C); const long long r = i / C; const int s = (int)(r / S); const long long ts = r % S - nreg;
    t[i] = ts < 0 ? reg[(r % S) * C + c] : x[((long long)s * C + c) * Sg + ts];
}
__global__ void k_tok_to_cs(const float *t, float *x, int n, int C, long long Sg, int nreg) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= (long long)n * C * Sg) return;
    const long long ts = i % Sg; const int c = (int)((i / Sg) % C), s = (int)(i / (Sg * C));
    x[i] = t[((long long)s * (Sg + nreg) + nreg + ts) * C + c];
}
__global__ void k_reg_grad(const float *gt, float *greg, int n, int C, long long Sg, int nreg) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nreg * C) return;
    float a = 0.f;
    for (int s = 0; s < n; s++) a += gt[(long long)s * (Sg + nreg) * C + i];
    greg[i] += a;
}
extern "C" void at_cs_to_tok(const float *x, float *t, const float *reg, int n, int C, long Sg, int nreg) { k_cs_to_tok<<<nbk((long long)n * (Sg + nreg) * C, 256), 256>>>(x, t, reg, n, C, Sg, nreg); }
extern "C" void at_tok_to_cs(const float *t, float *x, int n, int C, long Sg, int nreg) { k_tok_to_cs<<<nbk((long long)n * C * Sg, 256), 256>>>(t, x, n, C, Sg, nreg); }
extern "C" void at_reg_grad(const float *gt, float *greg, int n, int C, long Sg, int nreg) { if (nreg) k_reg_grad<<<nbk(nreg * C, 256), 256>>>(gt, greg, n, C, Sg, nreg); }

/* 16-bit storage (fp16 when f16, else bf16) -> fp32 */
#include <cuda_fp16.h>
#include <cuda_bf16.h>
__global__ void k_h16_to_f32(const void *x, long long n, int f16, float *y) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = f16 ? __half2float(((const __half *)x)[i]) : __bfloat162float(((const __nv_bfloat16 *)x)[i]);
}
extern "C" void at_h16_to_f32(const void *x, long long n, int f16, float *y) { k_h16_to_f32<<<nbk(n, 256), 256>>>(x, n, f16, y); }
