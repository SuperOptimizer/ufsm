/* CUDA ops: opt section of the former nn.cu */
#include "nn_common.cuh"

__global__ void adamw_k(float *p, const float *g, float *m, float *v, size_t n, float lr, float b1, float b2, float eps, float wd, float c1, float c2) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = g[i];
    float mi = m[i] = b1 * m[i] + (1.f - b1) * gi;
    float vi = v[i] = b2 * v[i] + (1.f - b2) * gi * gi;
    float mh = mi / c1, vh = vi / c2;
    p[i] -= lr * (mh / (sqrtf(vh) + eps) + wd * p[i]);
}
__global__ void mm_xxt_k(const float *X, int Co, int K, float *A) {   /* A = X X^T (Co x Co) */
    int i = blockIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= Co) return;
    const float *a = X + (size_t)i * K, *b = X + (size_t)j * K; float s = 0.f;
    for (int k = 0; k < K; k++) s += a[k] * b[k];
    A[(size_t)i * Co + j] = s;
}
__global__ void mm_sq_k(const float *A, int Co, float b, float c, float *B) {   /* B = b A + c A A (Co x Co) */
    int i = blockIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= Co) return;
    float s = 0.f;
    for (int k = 0; k < Co; k++) s += A[(size_t)i * Co + k] * A[(size_t)k * Co + j];
    B[(size_t)i * Co + j] = b * A[(size_t)i * Co + j] + c * s;
}
__global__ void mm_bx_k(const float *B, const float *X, int Co, int K, float a, float *Y) {   /* Y = a X + B X (Co x K) */
    int i = blockIdx.y; size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k >= (size_t)K) return;
    float s = 0.f;
    for (int j = 0; j < Co; j++) s += B[(size_t)i * Co + j] * X[(size_t)j * K + k];
    Y[(size_t)i * K + k] = a * X[(size_t)i * K + k] + s;
}
__global__ void muon_mom_k(const float *g, float *mom, float *x, size_t n, float beta) {   /* nesterov momentum: mom = beta mom + g; x = g + beta mom */
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= n) return;
    float m = beta * mom[i] + g[i]; mom[i] = m; x[i] = g[i] + beta * m;
}
__global__ void muon_sumsq_k(const float *x, size_t n, double *ss) {   /* block-reduced sum of squares into one double */
    __shared__ float r[256]; float a = 0.f; for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) a += x[i] * x[i];
    r[threadIdx.x] = a; __syncthreads(); for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) r[threadIdx.x] += r[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) atomicAdd(ss, (double)r[0]);
}
__global__ void muon_scale_k(float *x, size_t n, const double *ss, float eps) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) x[i] *= (float)(1.0 / (sqrt(*ss) + eps)); }
__global__ void muon_apply_k(float *p, const float *o, size_t n, float lr, float scale, float wd) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) p[i] = p[i] * (1.f - lr * wd) - lr * scale * o[i]; }
extern "C" void nn_muon(float *p, const float *g, float *mom, int Co, int K, float lr, float beta, float wd, float *work) {
    size_t n = (size_t)Co * K;
    float *X = work, *Y = work + n, *A = work + 2 * n, *B = A + (size_t)Co * Co;
    muon_mom_k<<<nblk(n, 256), 256>>>(g, mom, X, n, beta);
    double *ss = gn_dsums(1); cudaMemsetAsync(ss, 0, sizeof(double));
    muon_sumsq_k<<<nblk(n, 256) > 64 ? 64 : nblk(n, 256), 256>>>(X, n, ss);
    muon_scale_k<<<nblk(n, 256), 256>>>(X, n, ss, 1e-7f);
    const float a = 3.4445f, b = -4.7750f, c = 2.0315f;
    dim3 gco(nblk(Co, 128), Co), gk(nblk(K, 256), Co);
    for (int it = 0; it < 5; it++) {
        mm_xxt_k<<<gco, 128>>>(X, Co, K, A);
        mm_sq_k<<<gco, 128>>>(A, Co, b, c, B);
        mm_bx_k<<<gk, 256>>>(B, X, Co, K, a, Y);
        float *t = X; X = Y; Y = t;
    }
    float scale = sqrtf(fmaxf(1.f, (float)Co / (float)K));
    muon_apply_k<<<nblk(n, 256), 256>>>(p, X, n, lr, scale, wd);
    KCHECK();
}
__global__ void bmuon_mom_k(const muon_desc_t *d, float beta, double *ss) {
    const muon_desc_t D = d[blockIdx.z]; size_t n = (size_t)D.Co * D.K;
    __shared__ float r[256]; float a = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { float m = beta * D.mom[i] + D.g[i]; D.mom[i] = m; float x = D.g[i] + beta * m; D.X[i] = x; a += x * x; }
    r[threadIdx.x] = a; __syncthreads(); for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) r[threadIdx.x] += r[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) atomicAdd(&ss[blockIdx.z], (double)r[0]);
}
__global__ void bmuon_scale_k(const muon_desc_t *d, const double *ss) {
    const muon_desc_t D = d[blockIdx.z]; size_t n = (size_t)D.Co * D.K; float inv = (float)(1.0 / (sqrt(ss[blockIdx.z]) + 1e-7));
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) D.X[i] *= inv;
}
/* the update scale per matrix: 0 Keller Jordan's sqrt(max(1, Co / K)); 1 (train --muon-scale spectral, Scion / muP): sqrt(Co / K), the
   spectral norm's dual scaling sqrt(fan_out / fan_in) (fan_in = Ci k^3) under which the learning rate transfers across widths */
__constant__ int c_mscale;
extern "C" void nn_muon_scale_mode(int m) { cudaMemcpyToSymbol(c_mscale, &m, sizeof m); }
__device__ __forceinline__ float muon_scale(int Co, int K) { return c_mscale ? sqrtf((float)Co / (float)K) : sqrtf(fmaxf(1.f, (float)Co / (float)K)); }
__global__ void bmuon_apply_k(const muon_desc_t *d, int swap, float lr, float wd) {
    const muon_desc_t D = d[blockIdx.z]; const float *O = swap ? D.Y : D.X; size_t n = (size_t)D.Co * D.K;
    float scale = muon_scale(D.Co, D.K);
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) D.p[i] = D.p[i] * (1.f - lr * wd) - lr * scale * O[i];
}
/* 5 Newton-Schulz iterations (modded-nanogpt coefficients) on every table entry's X (Co x K); returns the buffer flag of the
   result (1: Y) */
/* the quintic maps x -> a x + b x^3 + c x^5 of the 5 iterations. 0: Keller Jordan's fixed (3.4445, -4.7750, 2.0315) (modded-nanogpt).
   1 (train --muon-ns polar): Polar Express (Amsel, Persson, Musco & Gower 2025, arXiv 2505.16932), the greedy minimax schedule:
   step t minimises max |1 - p_t(x)| over the current singular value interval [l, u], then [l, u] <- [min p_t, max p_t]; from
   [1e-3, 1] (derived here by linear programming on a dense grid, DESIGN.md 2026-11-13). After 5 steps every singular value in
   [1e-3, 1] lands in [0.887, 1.113] (the fixed map: [0.471, 1.202], [0.682, ..] from 1e-2). */
static const float g_ns_maps[2][5][3] = {
    {{3.4445f, -4.7750f, 2.0315f}, {3.4445f, -4.7750f, 2.0315f}, {3.4445f, -4.7750f, 2.0315f}, {3.4445f, -4.7750f, 2.0315f}, {3.4445f, -4.7750f, 2.0315f}},
    {{8.470329f, -25.108079f, 18.629279f}, {4.182835f, -3.108702f, 0.580607f}, {3.961857f, -2.954063f, 0.562976f}, {3.286586f, -2.464720f, 0.507358f},
     {2.273750f, -1.644660f, 0.416191f}}};
/* Tensor-core Newton-Schulz products (train --muon-tc 1; DESIGN "Muon variants"): gemm_ns_k's three MODEs with fp16 operands (the
   fp32 buffers rounded on the shared-memory load) and fp32 accumulation on wmma 16x16x16. 64 x 64 output tile per block, 4 warps
   of 32 x 32 (2 x 2 fragments), the contraction staged 32 at a time; the epilogue (b A + c acc, b X + acc) in fp32 as gemm_ns_k's.
   The operands are bounded (X unit Frobenius norm and its iterates, A = X X^T entries <= 1, B = b A + c A^2 within tens), so fp16's
   range holds and its 11-bit mantissa beats bf16's 8. */
static int g_ns_tc;
extern "C" void nn_muon_tc(int on) { g_ns_tc = on; }
template <int MODE> __global__ void __launch_bounds__(128) gemm_ns_tc_k(const muon_desc_t *d, int swap, float b, float c) {
    using namespace nvcuda;
    const muon_desc_t D = d[blockIdx.z];
    const int Co = D.Co, K = D.K, M = Co, N = MODE == 2 ? K : Co, KC = MODE == 0 ? K : Co;
    const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    if (m0 >= M || n0 >= N) return;
    const float *X = swap ? D.Y : D.X;
    const float *Lm = MODE == 0 ? X : MODE == 1 ? D.A : D.B;
    const float *Rm = MODE == 1 ? D.A : X;
    __shared__ __align__(32) __half sa[64][40], sb[64][40];   /* sa [m][k]; sb [n][k] (MODE 0: col-major B) or [k][n] (rows 0..31) */
    __shared__ __align__(32) float so[64][68];
    const int w = threadIdx.x >> 5, wm = (w >> 1) * 32, wn = (w & 1) * 32;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) wmma::fill_fragment(acc[i][j], 0.f);
    for (int k0 = 0; k0 < KC; k0 += 32) {
        for (int t = threadIdx.x; t < 64 * 32; t += 128) {
            const int r = t >> 5, kk = t & 31, m = m0 + r, k = k0 + kk;
            sa[r][kk] = __float2half(m < M && k < KC ? Lm[(size_t)m * KC + k] : 0.f);
            if (MODE == 0) { const int n = n0 + r; sb[r][kk] = __float2half(n < N && k < KC ? Rm[(size_t)n * K + k] : 0.f); }
            else { const int k2 = k0 + (t >> 6), n = n0 + (t & 63); (&sb[0][0])[t] = __float2half(n < N && k2 < KC ? Rm[(size_t)k2 * N + n] : 0.f); }
        }
        __syncthreads();
        for (int kk = 0; kk < 32; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[2];
            for (int i = 0; i < 2; i++) wmma::load_matrix_sync(fa[i], &sa[wm + 16 * i][kk], 40);
            if (MODE == 0) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb[2];
                for (int j = 0; j < 2; j++) wmma::load_matrix_sync(fb[j], &sb[wn + 16 * j][kk], 40);
                for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
            } else {   /* sb holds [k][n] in rows 0..31 with row stride 64 halves: view it as a 32 x 64 row-major tile */
                const __half *bt = &sb[0][0];
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb[2];
                for (int j = 0; j < 2; j++) wmma::load_matrix_sync(fb[j], bt + kk * 64 + wn + 16 * j, 64);
                for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) wmma::store_matrix_sync(&so[wm + 16 * i][wn + 16 * j], acc[i][j], 68, wmma::mem_row_major);
    __syncthreads();
    for (int t = threadIdx.x; t < 64 * 64; t += 128) {
        const int m = m0 + (t >> 6), n = n0 + (t & 63); if (m >= M || n >= N) continue;
        const float a = so[t >> 6][t & 63];
        if (MODE == 0) D.A[(size_t)m * Co + n] = a;
        else if (MODE == 1) D.B[(size_t)m * Co + n] = b * D.A[(size_t)m * Co + n] + c * a;
        else (swap ? D.X : D.Y)[(size_t)m * K + n] = b * X[(size_t)m * K + n] + a;
    }
}
/* cuBLASLt Newton-Schulz (train --muon-tc 2: TF32 tensor cores; fp32 buffers and accumulation. CUBLAS_COMPUTE_32F_FAST_16BF failed on the 92 M net's shapes). One GEMM
   per product and matrix (3 x 5 per matrix), column-major view of the row-major X (Co x K) as Xc (K x Co):
   A = Xc^T Xc, B = c A A + b A, Yc = Xc B + a Xc (B symmetric). The same products as gemm_ns_k, on the library's tuned kernels: the
   hand-tiled fp32 kernels reach ~10% of the card's fp32 rate on the 92 M net (121 ms per step) and the wmma fp16 ones are slower still */
#include <cublasLt.h>
typedef struct { int dev, m, n, k, ta, tc, ok; cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a, b, c; cublasLtMatmulAlgo_t algo; } nsplan;
static nsplan g_nsp[512]; static int g_nnsp;
static cublasLtHandle_t g_nslt[16]; static void *g_nsws[16]; static const size_t NSWS = 32u << 20;
static nsplan *nsplan_get(int dev, int ta, int m, int n, int k, int tc) {
    for (int i = 0; i < g_nnsp; i++) { nsplan *p = &g_nsp[i]; if (p->dev == dev && p->m == m && p->n == n && p->k == k && p->ta == ta && p->tc == tc) return p; }
    if (g_nnsp == 512 || dev < 0 || dev >= 16) return nullptr;
    if (!g_nslt[dev]) { if (cublasLtCreate(&g_nslt[dev]) != CUBLAS_STATUS_SUCCESS || cudaMalloc(&g_nsws[dev], NSWS) != cudaSuccess) return nullptr; }
    nsplan *p = &g_nsp[g_nnsp++]; memset(p, 0, sizeof *p); p->dev = dev; p->m = m; p->n = n; p->k = k; p->ta = ta; p->tc = tc;
    cublasLtMatmulDescCreate(&p->op, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F);
    const cublasOperation_t oa = ta ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &oa, sizeof oa);
    if (ta) cublasLtMatrixLayoutCreate(&p->a, CUDA_R_32F, k, m, k); else cublasLtMatrixLayoutCreate(&p->a, CUDA_R_32F, m, k, m);
    cublasLtMatrixLayoutCreate(&p->b, CUDA_R_32F, k, n, k); cublasLtMatrixLayoutCreate(&p->c, CUDA_R_32F, m, n, m);
    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &NSWS, sizeof NSWS);
    cublasLtMatmulHeuristicResult_t h; int nh = 0;
    if (cublasLtMatmulAlgoGetHeuristic(g_nslt[dev], p->op, p->a, p->b, p->c, p->c, pref, 1, &h, &nh) == CUBLAS_STATUS_SUCCESS && nh > 0) { p->algo = h.algo; p->ok = 1; }
    cublasLtMatmulPreferenceDestroy(pref);
    if (!p->ok) fprintf(stderr, "muon-tc %d: no cuBLASLt algorithm for m %d n %d k %d ta %d\n", tc, m, n, k, ta);
    return p;
}
/* column-major D (m x n) = alpha op(A) B + beta C (C may be D's input only when distinct; here C is never D) */
static int ns_gemm(int dev, int tc, int ta, int m, int n, int k, const float *A, const float *B, float alpha, float beta, const float *C, float *D) {
    nsplan *p = nsplan_get(dev, ta, m, n, k, tc); if (!p || !p->ok) return -1;
    return cublasLtMatmul(g_nslt[dev], p->op, &alpha, A, p->a, B, p->b, &beta, C, p->c, D, p->c, &p->algo, g_nsws[dev], NSWS, 0) == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}
static int ns5_lt(const muon_desc_t *d, int nconv, int mode) {
    static const void *hk[16]; static int hn[16]; static muon_desc_t *hd[16];   /* the table's host copy per device (built once, pointers stable) */
    int dev = 0; cudaGetDevice(&dev); if (dev < 0 || dev >= 16) return -1;
    if (hk[dev] != d || hn[dev] != nconv) { free(hd[dev]); hd[dev] = (muon_desc_t *)malloc(sizeof(muon_desc_t) * (size_t)nconv);
        cudaMemcpy(hd[dev], d, sizeof(muon_desc_t) * (size_t)nconv, cudaMemcpyDeviceToHost); hk[dev] = d; hn[dev] = nconv; }
    int e = 0;
    for (int i = 0; i < nconv; i++) {
        const muon_desc_t D = hd[dev][i]; float *X = D.X, *Y = D.Y;
        for (int it = 0; it < 5; it++) {
            const float a = g_ns_maps[mode][it][0], b = g_ns_maps[mode][it][1], c = g_ns_maps[mode][it][2];
            e |= ns_gemm(dev, g_ns_tc, 1, D.Co, D.Co, D.K, X, X, 1.f, 0.f, D.A, D.A);
            e |= ns_gemm(dev, g_ns_tc, 0, D.Co, D.Co, D.Co, D.A, D.A, c, b, D.A, D.B);
            e |= ns_gemm(dev, g_ns_tc, 0, D.K, D.Co, D.Co, X, D.B, 1.f, a, X, Y);
            float *t = X; X = Y; Y = t;
        }
    }
    if (e) { fprintf(stderr, "muon-tc %d: cuBLASLt Newton-Schulz failed\n", g_ns_tc); abort(); }
    return 1;   /* 5 iterations: the result in Y */
}
static int ns5_batch(const muon_desc_t *d, int nconv, int maxco, int maxk, int mode = 0) {
    if (g_ns_tc >= 2) return ns5_lt(d, nconv, mode);
    if (g_ns_tc) {
        dim3 gco(nblk(maxco, 64), nblk(maxco, 64), nconv), gk(nblk(maxk, 64), nblk(maxco, 64), nconv); int swap = 0;
        for (int it = 0; it < 5; it++) {
            const float a = g_ns_maps[mode][it][0], b = g_ns_maps[mode][it][1], c = g_ns_maps[mode][it][2];
            gemm_ns_tc_k<0><<<gco, 128>>>(d, swap, 0.f, 0.f); gemm_ns_tc_k<1><<<gco, 128>>>(d, swap, b, c); gemm_ns_tc_k<2><<<gk, 128>>>(d, swap, a, 0.f);
            swap ^= 1;
        }
        return swap;
    }
    dim3 gco(nblk(maxco, 64), nblk(maxco, 64), nconv), gk(nblk(maxk, 64), nblk(maxco, 64), nconv);
    int swap = 0;
    for (int it = 0; it < 5; it++) {
        const float a = g_ns_maps[mode][it][0], b = g_ns_maps[mode][it][1], c = g_ns_maps[mode][it][2];
        gemm_ns_k<0, muon_desc_t><<<gco, 256>>>(d, swap, 0.f, 0.f);
        gemm_ns_k<1, muon_desc_t><<<gco, 256>>>(d, swap, b, c);
        gemm_ns_k<2, muon_desc_t><<<gk, 256>>>(d, swap, a, 0.f);
        swap ^= 1;
    }
    return swap;
}
extern "C" void nn_muon_batch(const void *descs, int nconv, int maxco, int maxk, float lr, float beta, float wd) {
    const muon_desc_t *d = (const muon_desc_t *)descs;
    double *ss = gn_dsums((size_t)nconv); cudaMemsetAsync(ss, 0, (size_t)nconv * sizeof(double));
    dim3 g1(32, 1, nconv);
    bmuon_mom_k<<<g1, 256>>>(d, beta, ss);
    bmuon_scale_k<<<g1, 256>>>(d, ss);
    const int swap = ns5_batch(d, nconv, maxco, maxk);
    bmuon_apply_k<<<g1, 256>>>(d, swap, lr, wd);
    KCHECK();
}
/* ---- Muon-C (Qing & Li 2026, arXiv 2609.09676) and Hyperball (Wen, Dang, Lyu, Ma & Liang 2026, arXiv 2606.16899) ----
   Muon-C: the Nesterov momentum of a k^3 kernel [Co][Ci][k^3] goes through the orthonormal DFT over the three kernel axes
   (the critical Fourier grid: one Co x Ci channel-transfer block per kernel frequency), every block is polarised on its
   own by complex Newton-Schulz (the paper: 5 iterations with Muon's a, b, c, X0 = B / (|B|_F + 1e-8), a block with more
   rows than columns polarised as its conjugate transpose), and the inverse DFT returns a real update on the original
   support. A real kernel has conjugate-symmetric blocks and polar(conj B) = conj polar(B), so a 3^3 kernel polarises 14
   of its 27 frequencies (0 and one of each conjugate pair). Blocks are stored as interleaved complex (float2) and iterated
   by complex versions of Muon's tiled products (gemm_nsc_k). The applied update is rescaled to the RMS exact-polar unfolded Muon has for the
   conv (sqrt(max(1, Co/K)) sqrt(min(Co, K) / (Co K)), K = Ci k^3): the paper's matched applied-update RMS, so --muon-lr
   keeps its meaning. All frequencies of all convs go through one launch per stage (the Newton-Schulz table has one entry
   per conv and frequency, conv-major).
   Hyperball: W <- R Normalize(W - eta R Normalize(u)) with R the conv's |W|_F when the wrapper first ran and eta the
   Hyperball learning rate; it replaces the decay. u is the Muon or Muon-C direction (its scale cancels).
   Per-conv sums use 32 fixed partials summed in order (deterministic). */
typedef struct { const float *U; float *O; float *EX, *EY; int Co, Ci, k, m, n, tr; } muonc_desc_t;
__constant__ float c_mc_cs[2][14][27];   /* cos / sin of 2 pi (f . t) / 3: the 14 polarised frequencies f of a 3^3 grid, taps t */
__constant__ float c_mc_w[14];           /* 1: the zero frequency, 2: a conjugate pair */
static void mc_tables(void) {
    static int done[16];
    if (done[cur_dev()]) return;
    float cs[2][14][27], w[14]; int nf = 0;
    for (int f = 0; f < 27; f++) {
        const int p = f / 9, q = f / 3 % 3, r = f % 3, neg = (3 - p) % 3 * 9 + (3 - q) % 3 * 3 + (3 - r) % 3;
        if (f > neg) continue;
        for (int t = 0; t < 27; t++) {
            const double ph = 2.0 * M_PI * ((p * (t / 9) + q * (t / 3 % 3) + r * (t % 3)) % 3) / 3.0;
            cs[0][nf][t] = (float)cos(ph); cs[1][nf][t] = (float)sin(ph);
        }
        w[nf++] = f == neg ? 1.f : 2.f;
    }
    CK(cudaMemcpyToSymbol(c_mc_cs, cs, sizeof cs)); CK(cudaMemcpyToSymbol(c_mc_w, w, sizeof w));
    done[cur_dev()] = 1;
}
__device__ __forceinline__ void mc_part(double a, double *part) {   /* block sum into part[conv * 32 + block] */
    __shared__ double r[256];
    r[threadIdx.x] = a; __syncthreads();
    for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) r[threadIdx.x] += r[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) part[blockIdx.y * 32 + blockIdx.x] = r[0];
}
__device__ __forceinline__ double mc_total(const double *part) { double s = 0; for (int j = 0; j < 32; j++) s += part[blockIdx.y * 32 + j]; return s; }
/* complex products of the Newton-Schulz cascade (blockIdx.z = entry; X, Y: m x n, A, B: m x m, interleaved float2; Co = m,
   K = n): MODE 0 A = X X^*, MODE 1 B = b A + c A A, MODE 2 Y = b X + B X. Tiles as gemm_ns_k. */
template <int MODE> __global__ void __launch_bounds__(256) gemm_nsc_k(const muon_desc_t *d, int swap, float b, float c) {
    const muon_desc_t D = d[blockIdx.z];
    const int M = D.Co, NX = D.K, N = MODE == 2 ? NX : M, KC = MODE == 0 ? NX : M;
    const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    if (m0 >= M || n0 >= N) return;
    const float2 *X = (const float2 *)(swap ? D.Y : D.X), *A = (const float2 *)D.A, *Bm = (const float2 *)D.B;
    const float2 *Lm = MODE == 0 ? X : MODE == 1 ? A : Bm, *Rm = MODE == 1 ? A : X;
    __shared__ float2 sl[16][64], sr[16][64];   /* [k][m], [k][n] */
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    float2 acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = make_float2(0.f, 0.f);
    for (int k0 = 0; k0 < KC; k0 += 16) {
        for (int t = threadIdx.x; t < 1024; t += 256) {
            const int r = t >> 4, kk = t & 15, m = m0 + r, k = k0 + kk;
            sl[kk][r] = m < M && k < KC ? Lm[(size_t)m * KC + k] : make_float2(0.f, 0.f);
            if (MODE == 0) { const int n = n0 + r; const float2 v = n < N && k < KC ? Rm[(size_t)n * NX + k] : make_float2(0.f, 0.f); sr[kk][r] = make_float2(v.x, -v.y); }
            else { const int k2 = k0 + (t >> 6), n = n0 + (t & 63); sr[t >> 6][t & 63] = n < N && k2 < KC ? Rm[(size_t)k2 * N + n] : make_float2(0.f, 0.f); }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 16; kk++) {
            float2 lv[4], rv[4];
#pragma unroll
            for (int i = 0; i < 4; i++) { lv[i] = sl[kk][ty * 4 + i]; rv[i] = sr[kk][tx * 4 + i]; }
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) { acc[i][j].x += lv[i].x * rv[j].x - lv[i].y * rv[j].y; acc[i][j].y += lv[i].x * rv[j].y + lv[i].y * rv[j].x; }
        }
        __syncthreads();
    }
    float2 *Yo = (float2 *)(swap ? D.X : D.Y), *Ao = (float2 *)D.A, *Bo = (float2 *)D.B;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int m = m0 + ty * 4 + i;
        if (m >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int n = n0 + tx * 4 + j;
            if (n >= N) continue;
            const float2 v = acc[i][j];
            if (MODE == 0) Ao[(size_t)m * M + n] = v;
            else if (MODE == 1) { const float2 a = A[(size_t)m * M + n]; Bo[(size_t)m * M + n] = make_float2(b * a.x + c * v.x, b * a.y + c * v.y); }
            else { const float2 x = X[(size_t)m * NX + n]; Yo[(size_t)m * NX + n] = make_float2(b * x.x + v.x, b * x.y + v.y); }
        }
    }
}
static int ns5c_batch(const muon_desc_t *d, int n, int maxm, int maxn, int mode = 0) {   /* ns5_batch on complex entries */
    dim3 gm(nblk(maxm, 64), nblk(maxm, 64), n), gn(nblk(maxn, 64), nblk(maxm, 64), n);
    int swap = 0;
    for (int it = 0; it < 5; it++) {
        const float a = g_ns_maps[mode][it][0], b = g_ns_maps[mode][it][1], c = g_ns_maps[mode][it][2];
        gemm_nsc_k<0><<<gm, 256>>>(d, swap, 0.f, 0.f);
        gemm_nsc_k<1><<<gm, 256>>>(d, swap, b, c);
        gemm_nsc_k<2><<<gn, 256>>>(d, swap, a, 0.f);
        swap ^= 1;
    }
    return swap;
}
/* every frequency block of a conv in one pass over its taps: block f (m x n complex) at EX + f 2mn floats; sums per entry
   conv nf + f */
__global__ void muonc_dft_k(const muonc_desc_t *d, int nf, double *part) {
    const muonc_desc_t D = d[blockIdx.y];
    const int T = D.k * D.k * D.k; const size_t np = (size_t)D.Co * D.Ci, fs = (size_t)D.m * D.n; const float nrm = rsqrtf((float)T);
    double a[14] = {};
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < np; i += (size_t)gridDim.x * blockDim.x) {
        const float *u = D.U + i * T; float uv[27];
        for (int t = 0; t < T; t++) uv[t] = u[t];
        int r = (int)(i / D.Ci), c = (int)(i % D.Ci);
        if (D.tr) { const int t = r; r = c; c = t; }   /* the conjugate transpose: B~[ci][co] = conj B[co][ci] */
        for (int f = 0; f < nf; f++) {
            float re = 0.f, im = 0.f;
            if (T == 1) re = uv[0];
            else for (int t = 0; t < 27; t++) { re += uv[t] * c_mc_cs[0][f][t]; im -= uv[t] * c_mc_cs[1][f][t]; }
            re *= nrm; im *= D.tr ? -nrm : nrm;
            ((float2 *)D.EX)[f * fs + (size_t)r * D.n + c] = make_float2(re, im);
            a[f] += (double)re * re + (double)im * im;
        }
    }
    __shared__ double rs[256];
    for (int f = 0; f < nf; f++) {
        rs[threadIdx.x] = a[f]; __syncthreads();
        for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) rs[threadIdx.x] += rs[threadIdx.x + o]; __syncthreads(); }
        if (threadIdx.x == 0) part[((size_t)blockIdx.y * nf + f) * 32 + blockIdx.x] = rs[0];
        __syncthreads();
    }
}
__global__ void muonc_norm_k(const muonc_desc_t *d, int nf, const double *part) {   /* X0 = B / (|B|_F + 1e-8), the complex norm; blockIdx.y = conv nf + f */
    const muonc_desc_t D = d[blockIdx.y / nf]; const size_t fs = (size_t)2 * D.m * D.n; float *E = D.EX + (blockIdx.y % nf) * fs;
    const float inv = (float)(1.0 / (sqrt(mc_total(part)) + 1e-8));
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < fs; i += (size_t)gridDim.x * blockDim.x) E[i] *= inv;
}
__global__ void muonc_idft_k(const muonc_desc_t *d, int nf, int swap) {   /* O = the inverse DFT of the polarised blocks */
    const muonc_desc_t D = d[blockIdx.y]; const float2 *E0 = (const float2 *)(swap ? D.EY : D.EX);
    const int T = D.k * D.k * D.k; const size_t np = (size_t)D.Co * D.Ci, fs = (size_t)D.m * D.n;
    const float nrm = rsqrtf((float)T);
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < np; i += (size_t)gridDim.x * blockDim.x) {
        int r = (int)(i / D.Ci), c = (int)(i % D.Ci);
        if (D.tr) { const int t = r; r = c; c = t; }
        float pr[14], pi[14];
        for (int f = 0; f < nf; f++) {
            const float2 v = E0[f * fs + (size_t)r * D.n + c]; const float sc = (T == 1 ? 1.f : c_mc_w[f]) * nrm;
            pr[f] = sc * v.x; pi[f] = sc * (D.tr ? -v.y : v.y);   /* P = conj(X5)^T for a transposed block */
        }
        float *o = D.O + i * T;
        if (T == 1) o[0] = pr[0];
        else for (int t = 0; t < 27; t++) { float s = 0.f; for (int f = 0; f < nf; f++) s += pr[f] * c_mc_cs[0][f][t] - pi[f] * c_mc_cs[1][f][t]; o[t] = s; }
    }
}
__global__ void mopt_sumsq_k(const muon_desc_t *d, int which, double *part) {   /* which: 0 X, 1 Y, -1 the weights */
    const muon_desc_t D = d[blockIdx.y]; const float *x = which < 0 ? D.p : which ? D.Y : D.X; const size_t n = (size_t)D.Co * D.K;
    double a = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) a += (double)x[i] * x[i];
    mc_part(a, part);
}
__global__ void hb_rinit_k(const int *hb, float *R, const double *part) {   /* R = |W|_F where not yet set; one block per conv */
    const int c = blockIdx.x; if (threadIdx.x || !hb[c] || R[c] > 0.f) return;
    double s = 0; for (int j = 0; j < 32; j++) s += part[c * 32 + j];
    R[c] = (float)sqrt(s);
}
__global__ void mopt_apply_k(const muon_desc_t *d, int swap, int mc, const int *hb, const float *R, const double *po, double *pw,
                             float lr, float wd, float hb_lr) {
    const muon_desc_t D = d[blockIdx.y]; const float *O = swap ? D.Y : D.X; const size_t n = (size_t)D.Co * D.K;
    const double so = mc_total(po);
    double a = 0;
    if (hb && hb[blockIdx.y]) {   /* trial point W - eta R O / |O|; its norm for the projection */
        const float st = (float)((double)hb_lr * R[blockIdx.y] / fmax(sqrt(so), 1e-30));
        for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { const float w = D.p[i] - st * O[i]; D.p[i] = w; a += (double)w * w; }
    } else {
        float scale = muon_scale(D.Co, D.K);   /* Muon (as bmuon_apply_k) */
        if (mc) {
            const double nk = (double)D.Co * D.K, tgt = sqrt(fmax(1.0, (double)D.Co / D.K)) * sqrt(fmin((double)D.Co, (double)D.K) / nk);
            scale = (float)(tgt / (sqrt(so / nk) + 1e-8));
        }
        for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) D.p[i] = D.p[i] * (1.f - lr * wd) - lr * scale * O[i];
    }
    mc_part(a, pw);
}
__global__ void hb_proj_k(const muon_desc_t *d, const int *hb, const float *R, const double *pw) {   /* back onto the sphere |W|_F = R */
    if (!hb[blockIdx.y]) return;
    const muon_desc_t D = d[blockIdx.y]; const size_t n = (size_t)D.Co * D.K;
    const float s = (float)(R[blockIdx.y] / fmax(sqrt(mc_total(pw)), 1e-30));
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) D.p[i] *= s;
}
/* NorMuon (Li, Lin, Gonzalez, Wang, Zhang & Hao 2025, arXiv 2510.05491; train --normuon 1): after the orthogonalisation, each output row i
   of O is divided by sqrt(v_i), v_i = an EMA (beta2) of the row's mean square, and the whole O is rescaled to its own Frobenius norm
   (as ANVIL's lane equalisation): the update keeps Muon's RMS (and the sqrt(max(1, Co / K)) multiplier), only redistributed over rows.
   (The paper scales by 0.2 sqrt(mn) / |O'|_F to match AdamW's RMS instead; keeping Muon's RMS keeps --muon-lr comparable.) The EMA
   starts at 0 without bias correction: the common factor cancels in the rescale. rows: per conv off[c] .. off[c] + Co floats of EMA
   at rows[..] and the same of scratch at rows[tot + ..], tot = the EMA's length. */
__global__ void nm_rowpow_k(const muon_desc_t *d, int swap, const int *off, float *scr) {   /* grid (maxco, 1, nconv): one block per row */
    const muon_desc_t D = d[blockIdx.z]; const int i = blockIdx.x; if (i >= D.Co) return;
    const float *O = swap ? D.Y : D.X; __shared__ float r[256]; float a = 0.f;
    for (int k = threadIdx.x; k < D.K; k += blockDim.x) { const float v = O[(size_t)i * D.K + k]; a += v * v; }
    r[threadIdx.x] = a; __syncthreads(); for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) r[threadIdx.x] += r[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) scr[off[blockIdx.z] + i] = r[0] / (float)D.K;
}
__global__ void nm_eq_k(const muon_desc_t *d, const int *off, float *ema, float *scr, float b2) {   /* one thread per conv: the row factors */
    const int c = blockIdx.x; if (threadIdx.x) return;
    const int Co = d[c].Co; float *E = ema + off[c], *R = scr + off[c]; double pre = 0, post = 0;
    for (int i = 0; i < Co; i++) { const float pw = R[i]; pre += pw; E[i] += (1.f - b2) * (pw - E[i]); const float iv = 1.f / fmaxf(E[i], 1e-20f); post += pw * iv; }
    const float s = (float)sqrt(pre / fmax(post, 1e-30));
    for (int i = 0; i < Co; i++) R[i] = rsqrtf(fmaxf(E[i], 1e-20f)) * s;
}
__global__ void nm_scale_k(const muon_desc_t *d, int swap, const int *off, const float *scr) {   /* O row i *= its factor */
    const muon_desc_t D = d[blockIdx.z]; float *O = swap ? D.Y : D.X; const size_t n = (size_t)D.Co * D.K; const float *f = scr + off[blockIdx.z];
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) O[i] *= f[i / (size_t)D.K];
}
/* MARS variance reduction (Yuan, Liu, Wu, Gu & Gu 2024, arXiv 2411.10438; MARS-Muon; train --muon-vr G): the momentum's input is
   c = g + G beta / (1 - beta) (g - g_prev), scaled to unit Frobenius norm when larger (per matrix), then Muon's Nesterov momentum and
   Newton-Schulz on it. g_prev: one buffer per matrix (zero before the first step: c = (1 + G beta / (1 - beta)) g then). c goes to Y
   (Y is Newton-Schulz scratch, overwritten later). */
__global__ void vr_c_k(const muon_desc_t *d, float *const *prev, float k, double *ss) {
    const muon_desc_t D = d[blockIdx.z]; float *gp = prev[blockIdx.z]; const size_t n = (size_t)D.Co * D.K;
    __shared__ float r[256]; float a = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { const float g = D.g[i], c = g + k * (g - gp[i]); gp[i] = g; D.Y[i] = c; a += c * c; }
    r[threadIdx.x] = a; __syncthreads(); for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) r[threadIdx.x] += r[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) atomicAdd(&ss[blockIdx.z], (double)r[0]);
}
__global__ void vr_mom_k(const muon_desc_t *d, float beta, const double *cs, double *ss) {   /* bmuon_mom_k on the clipped c */
    const muon_desc_t D = d[blockIdx.z]; const size_t n = (size_t)D.Co * D.K; const double nc = sqrt(cs[blockIdx.z]); const float f = nc > 1.0 ? (float)(1.0 / nc) : 1.f;
    __shared__ float r[256]; float a = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { const float c = D.Y[i] * f; float m = beta * D.mom[i] + c; D.mom[i] = m; float x = c + beta * m; D.X[i] = x; a += x * x; }
    r[threadIdx.x] = a; __syncthreads(); for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) r[threadIdx.x] += r[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) atomicAdd(&ss[blockIdx.z], (double)r[0]);
}
/* Cautious masking (Liang, Chen, Liu & Liu 2024, arXiv 2411.16085; train --muon-cautious 1): the orthogonalised direction's elements whose
   sign disagrees with the current gradient are zeroed and the rest rescaled to keep the direction's Frobenius norm (per matrix). */
static int g_caut; static double *g_caut_s; static int g_caut_n;
/* AdEMAMix on Muon (Pagliardini, Ablin & Grangier 2024, arXiv 2409.03137; train --muon-ademamix A): a slow EMA m2 = b3 m2 + (1 - b3) g
   beside the Nesterov momentum, X = nesterov + alpha m2 / (1 - beta) (Muon's m is a sum, ~ g / (1 - beta): the factor puts m2 on
   AdEMAMix's EMA scale), alpha annealed by the caller (0 -> A over the warmup horizon). m2: one buffer per matrix. */
static float *const *g_mix_m2; static float g_mix_b3, g_mix_a;
extern "C" void nn_muon_ext_opts(int cautious) { g_caut = cautious; }
extern "C" void nn_muon_ext_mix(float *const *m2, float b3, float alpha) { g_mix_m2 = m2; g_mix_b3 = b3; g_mix_a = alpha; }
__global__ void mix_k(const muon_desc_t *d, float *const *m2, float b3, float k, double *ss) {   /* X += k m2 (m2 updated from g first); ss: |X|^2 */
    const muon_desc_t D = d[blockIdx.z]; float *m = m2[blockIdx.z]; const size_t n = (size_t)D.Co * D.K;
    __shared__ float r[256]; float a = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float v = b3 * m[i] + (1.f - b3) * D.g[i]; m[i] = v; const float x = D.X[i] + k * v; D.X[i] = x; a += x * x; }
    r[threadIdx.x] = a; __syncthreads(); for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) r[threadIdx.x] += r[threadIdx.x + o]; __syncthreads(); }
    if (!threadIdx.x) atomicAdd(&ss[blockIdx.z], (double)r[0]);
}
__global__ void caut_sum_k(const muon_desc_t *d, int swap, double *s) {   /* s[2 c] = |O|^2, s[2 c + 1] = |O masked|^2 */
    const muon_desc_t D = d[blockIdx.z]; const float *O = swap ? D.Y : D.X; const size_t n = (size_t)D.Co * D.K;
    __shared__ float r[2][256]; float a = 0.f, b = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { const float o = O[i]; a += o * o; if (o * D.g[i] > 0.f) b += o * o; }
    r[0][threadIdx.x] = a; r[1][threadIdx.x] = b; __syncthreads();
    for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) { r[0][threadIdx.x] += r[0][threadIdx.x + o]; r[1][threadIdx.x] += r[1][threadIdx.x + o]; } __syncthreads(); }
    if (!threadIdx.x) { atomicAdd(&s[2 * blockIdx.z], (double)r[0][0]); atomicAdd(&s[2 * blockIdx.z + 1], (double)r[1][0]); }
}
__global__ void caut_apply_k(const muon_desc_t *d, int swap, const double *s) {
    const muon_desc_t D = d[blockIdx.z]; float *O = swap ? D.Y : D.X; const size_t n = (size_t)D.Co * D.K;
    const float f = s[2 * blockIdx.z + 1] > 0.0 ? (float)sqrt(s[2 * blockIdx.z] / s[2 * blockIdx.z + 1]) : 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) O[i] = O[i] * D.g[i] > 0.f ? O[i] * f : 0.f;
}
extern "C" void nn_muon_ext_batch(const void *descs, int nconv, int maxco, int maxk, const void *mcd, const void *nsd, int k, int ns_maxm, int ns_maxn,
                                  const int *hb, float *R, int r_init, double *part, float lr, float beta, float wd, float hb_lr,
                                  int ns_mode, float *nm_rows, const int *nm_off, int nm_tot, float nm_b2, float *const *vr_prev, float vr_g) {
    const muon_desc_t *d = (const muon_desc_t *)descs;
    double *pa = part, *pb = part + (size_t)32 * nconv, *pf = part + (size_t)64 * nconv;
    dim3 g1(32, 1, nconv), g2(32, nconv);
    int swap;
    cudaMemsetAsync(pa, 0, (size_t)nconv * sizeof(double));   /* bmuon_mom_k's Frobenius sums (one per conv) */
    if (vr_prev && vr_g != 0.f) {   /* MARS: the momentum's input is the variance-reduced, clipped c (pb: its Frobenius sums) */
        cudaMemsetAsync(pb, 0, (size_t)nconv * sizeof(double));
        vr_c_k<<<g1, 256>>>(d, vr_prev, vr_g * beta / (1.f - beta), pb);
        vr_mom_k<<<g1, 256>>>(d, beta, pb, pa);
    } else bmuon_mom_k<<<g1, 256>>>(d, beta, pa);   /* X = the Nesterov momentum */
    if (g_mix_m2) { cudaMemsetAsync(pa, 0, (size_t)nconv * sizeof(double)); mix_k<<<g1, 256>>>(d, g_mix_m2, g_mix_b3, g_mix_a / (1.f - beta), pa); }
    if (!mcd) { bmuon_scale_k<<<g1, 256>>>(d, pa); swap = ns5_batch(d, nconv, maxco, maxk, ns_mode); }
    else {   /* Muon-C: every frequency of every conv in one launch per stage; the direction goes to Y */
        mc_tables();
        const muonc_desc_t *m = (const muonc_desc_t *)mcd; const muon_desc_t *s = (const muon_desc_t *)nsd;
        const int nf = k == 3 ? 14 : 1;   /* one kernel size for the whole table: 3 or 1 */
        muonc_dft_k<<<g2, 256>>>(m, nf, pf);
        muonc_norm_k<<<dim3(32, nconv * nf), 256>>>(m, nf, pf);
        muonc_idft_k<<<g2, 256>>>(m, nf, ns5c_batch(s, nconv * nf, ns_maxm, ns_maxn, ns_mode));
        swap = 1;
    }
    if (nm_rows) {   /* NorMuon: the per-row normalisation of the orthogonalised direction */
        nm_rowpow_k<<<dim3(maxco, 1, nconv), 256>>>(d, swap, nm_off, nm_rows + nm_tot);
        nm_eq_k<<<nconv, 32>>>(d, nm_off, nm_rows, nm_rows + nm_tot, nm_b2);
        nm_scale_k<<<g1, 256>>>(d, swap, nm_off, nm_rows + nm_tot);
    }
    if (g_caut) {
        if (g_caut_n < nconv) { cudaFree(g_caut_s); cudaMalloc(&g_caut_s, (size_t)2 * nconv * sizeof(double)); g_caut_n = nconv; }
        cudaMemsetAsync(g_caut_s, 0, (size_t)2 * nconv * sizeof(double));
        caut_sum_k<<<g1, 256>>>(d, swap, g_caut_s); caut_apply_k<<<g1, 256>>>(d, swap, g_caut_s);
    }
    if (r_init) { mopt_sumsq_k<<<g2, 256>>>(d, -1, pb); hb_rinit_k<<<nconv, 32>>>(hb, R, pb); }
    mopt_sumsq_k<<<g2, 256>>>(d, swap, pa);
    mopt_apply_k<<<g2, 256>>>(d, swap, mcd != nullptr, hb, R, pa, pb, lr, wd, hb_lr);
    if (hb) hb_proj_k<<<g2, 256>>>(d, hb, R, pb);
    KCHECK();
}
__constant__ float c_anvil_maps[6][3] = {{3.923798038567f, -6.095026865488f, 3.905234618423f}, {3.278126713798f, -3.328923386476f, 0.989127286973f},
    {3.505298394150f, -5.137358782410f, 1.968325560615f}, {2.815058591845f, -3.685181239622f, 1.417196497642f},
    {2.245503932403f, -2.443826979899f, 0.963091710461f}, {2.256537145403f, -2.166840097229f, 0.929501253245f}};
__global__ void anvil_mom_k(const anvil_desc_t *d, float bf, float bs, float w, float mu, double *ss) {
    const anvil_desc_t D = d[blockIdx.z]; size_t n = (size_t)D.Co * D.K;
    __shared__ float r[256]; float a = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float g = D.g[i], v0 = D.v0[i] + (1.f - bf) * (g - D.v0[i]), v1 = D.v1[i] + (1.f - bs) * (g - D.v1[i]);
        D.v0[i] = v0; D.v1[i] = v1;
        float m = w * v0 + (1.f - w) * v1, x = g + mu * (m - g);
        D.X[i] = x; a += x * x;
    }
    r[threadIdx.x] = a; __syncthreads(); for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) r[threadIdx.x] += r[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) atomicAdd(&ss[blockIdx.z], (double)r[0]);
}
__global__ void anvil_scale_k(const anvil_desc_t *d, const double *ss) {
    const anvil_desc_t D = d[blockIdx.z]; size_t n = (size_t)D.Co * D.K; float inv = (float)(1.0 / (sqrt(ss[blockIdx.z]) * 1.05 + 1e-6));
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) D.X[i] *= inv;
}
__global__ void anvil_sq_k(const anvil_desc_t *d, int it) {
    const anvil_desc_t D = d[blockIdx.z]; int Co = D.Co; float b = c_anvil_maps[it][1], c = c_anvil_maps[it][2];
    int i = blockIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x; if (i >= Co || j >= Co) return;
    float s = 0.f; for (int k = 0; k < Co; k++) s += D.A[(size_t)i * Co + k] * D.A[(size_t)k * Co + j];
    D.B[(size_t)i * Co + j] = b * D.A[(size_t)i * Co + j] + c * s;
}
__global__ void anvil_rowpow_k(const anvil_desc_t *d, int swap) {
    const anvil_desc_t D = d[blockIdx.z]; const float *O = swap ? D.Y : D.X; int i = blockIdx.x; if (i >= D.Co) return;
    __shared__ float r[256]; float a = 0.f; for (int k = threadIdx.x; k < D.K; k += blockDim.x) { float v = O[(size_t)i * D.K + k]; a += v * v; }
    r[threadIdx.x] = a; __syncthreads(); for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) r[threadIdx.x] += r[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) D.R[i] = r[0] / (float)D.K;
}
__global__ void anvil_eq_k(const anvil_desc_t *d, float b2) {
    const anvil_desc_t D = d[blockIdx.x]; if (threadIdx.x) return;
    float pre = 0.f, post = 0.f;
    for (int i = 0; i < D.Co; i++) { float pw = D.R[i]; pre += pw; D.E[i] += (1.f - b2) * (pw - D.E[i]); float gn = rsqrtf(fmaxf(D.E[i], 1e-10f)); post += pw * gn * gn; }
    float s = sqrtf(pre) / fmaxf(sqrtf(post), 1e-10f);
    for (int i = 0; i < D.Co; i++) D.R[i] = rsqrtf(fmaxf(D.E[i], 1e-10f)) * s;
}
__global__ void anvil_apply_k(const anvil_desc_t *d, int swap, float lr, float wd) {   /* sign-aligned decay + update */
    const anvil_desc_t D = d[blockIdx.z]; const float *O = swap ? D.Y : D.X; size_t n = (size_t)D.Co * D.K;
    float scale = sqrtf(fmaxf(1.f, (float)D.Co / (float)D.K));
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float u = O[i] * D.R[i / (size_t)D.K] * scale, p = D.p[i];
        float dec = (u * p >= 0.f) ? lr * wd * p : 0.f;
        D.p[i] = p - dec - lr * u;
    }
}
extern "C" void nn_anvil_batch(const void *descs, int nconv, int maxco, int maxk, float lr, float beta_fast, float beta_slow, float w_fast, float mu, float beta2, float wd) {
    const anvil_desc_t *d = (const anvil_desc_t *)descs;
    double *ss = gn_dsums((size_t)nconv); cudaMemsetAsync(ss, 0, (size_t)nconv * sizeof(double));
    dim3 g1(32, 1, nconv), gco(nblk(maxco, 128), maxco, nconv), g64(nblk(maxco, 64), nblk(maxco, 64), nconv), gk(nblk(maxk, 64), nblk(maxco, 64), nconv), grow(maxco, 1, nconv);
    anvil_mom_k<<<g1, 256>>>(d, beta_fast, beta_slow, w_fast, mu, ss);
    anvil_scale_k<<<g1, 256>>>(d, ss);
    int swap = 0;
    static const float maps_a[6] = {3.923798038567f, 3.278126713798f, 3.505298394150f, 2.815058591845f, 2.245503932403f, 2.256537145403f};
    for (int it = 0; it < 6; it++) {
        gemm_ns_k<0, anvil_desc_t><<<g64, 256>>>(d, swap, 0.f, 0.f);
        anvil_sq_k<<<gco, 128>>>(d, it);
        gemm_ns_k<2, anvil_desc_t><<<gk, 256>>>(d, swap, maps_a[it], 0.f);
        swap ^= 1;
    }
    anvil_rowpow_k<<<grow, 256>>>(d, swap);
    anvil_eq_k<<<nconv, 32>>>(d, beta2);
    anvil_apply_k<<<g1, 256>>>(d, swap, lr, wd);
    KCHECK();
}
extern "C" void nn_adamw(float *p, const float *g, float *m, float *v, size_t n, float lr, float b1, float b2, float eps, float wd, int step) {
    float c1 = 1.f - powf(b1, (float)step), c2 = 1.f - powf(b2, (float)step);
    adamw_k<<<nblk(n, 256), 256>>>(p, g, m, v, n, lr, b1, b2, eps, wd, c1, c2);
    KCHECK();
}
__global__ void ema_k(float *e, const float *p, size_t n, float d) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) e[i] = d * e[i] + (1.f - d) * p[i]; }
extern "C" void nn_ema(float *ema, const float *p, size_t n, float decay) { ema_k<<<nblk(n, 256), 256>>>(ema, p, n, decay); KCHECK(); }
__global__ void sum_k(const float *x, size_t n, float *out, int sq) {
    double s = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { double v = x[i]; s += sq ? v * v : v; }
    __shared__ double r[256];
    r[threadIdx.x] = s;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) r[threadIdx.x] += r[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) out[blockIdx.x] = (float)r[0];
}
double reduce(const float *x, size_t n, float *scratch, int sq) {
    int nb = 1024;
    sum_k<<<nb, 256>>>(x, n, scratch, sq);
    float h[1024];
    CK(cudaMemcpy(h, scratch, sizeof h, cudaMemcpyDeviceToHost));
    double s = 0;
    for (int i = 0; i < nb; i++) s += h[i];
    return s;
}
extern "C" double nn_sum(const float *x, size_t n, float *scratch) { return reduce(x, n, scratch, 0); }
extern "C" double nn_sumsq(const float *x, size_t n, float *scratch) { return reduce(x, n, scratch, 1); }
extern "C" void nn_fake_quant_affine(void *x, shape5 s, int fmt, const float *mean, const float *rstd, int G) {
    if (fmt <= 0 || ISMX(x) || fmt == 1) return;   /* MX formats only (the tensor scale of NVFP4 is not meaningful after the affine) */
    size_t S = shape_spatial(s); int C = s.c, N = s.n; const int gs = 32, ng = (C + gs - 1) / gs;
    size_t tot = (size_t)N * ng * S;
    if (ABF && g_h16) fq_aff_k<f16><<<nblk(tot, 256), 256>>>((f16 *)x, N, C, S, fmt, mean, rstd, G);
    else if (ABF) fq_aff_k<bf16><<<nblk(tot, 256), 256>>>((bf16 *)x, N, C, S, fmt, mean, rstd, G);
    else fq_aff_k<float><<<nblk(tot, 256), 256>>>((float *)x, N, C, S, fmt, mean, rstd, G);
    KCHECK();
}
extern "C" void nn_fake_quant(void *x, shape5 s, int fmt) {
    if (fmt <= 0 || ISMX(x)) return;
    size_t S = shape_spatial(s), n = shape_numel(s);
    static unsigned *am[8];
    if (!am[cur_dev()]) cudaMalloc(&am[cur_dev()], 4);
    cudaMemsetAsync(am[cur_dev()], 0, 4);
    const int gs = fmt == 1 ? 16 : 32;
    size_t nt = (size_t)s.n * ((s.c + gs - 1) / gs) * S;
    if (ABF && g_h16) { fq_amax_k<f16><<<512, 256>>>((const f16 *)x, n, am[cur_dev()]); fq_k<f16><<<nblk(nt, 256), 256>>>((f16 *)x, s.n, s.c, S, fmt, am[cur_dev()]); }
    else if (ABF) { fq_amax_k<bf16><<<512, 256>>>((const bf16 *)x, n, am[cur_dev()]); fq_k<bf16><<<nblk(nt, 256), 256>>>((bf16 *)x, s.n, s.c, S, fmt, am[cur_dev()]); }
    else { fq_amax_k<float><<<512, 256>>>((const float *)x, n, am[cur_dev()]); fq_k<float><<<nblk(nt, 256), 256>>>((float *)x, s.n, s.c, S, fmt, am[cur_dev()]); }
    KCHECK();
}

/* Scion's ell-infinity LMO for the parameters off Muon (biases, norm gains, the head; train --opt-rest sign): m = b m + (1 - b) g,
   p = p (1 - lr wd) - lr sign(m) (Pethick, Xie, Erdogan, Antonakopoulos, Silveti-Falls & Cevher 2025, arXiv 2502.07529). m: Adam's m. */
__global__ void signm_k(float *p, const float *g, float *m, size_t n, float lr, float b, float wd) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float v = b * m[i] + (1.f - b) * g[i]; m[i] = v; p[i] = p[i] * (1.f - lr * wd) - lr * (float)((v > 0.f) - (v < 0.f)); }
}
extern "C" void nn_signm(float *p, const float *g, float *m, size_t n, float lr, float b, float wd) {
    if (!n) return; const unsigned nb = (unsigned)(n < (size_t)256 * 1024 ? (n + 255) / 256 : 1024); signm_k<<<nb, 256>>>(p, g, m, n, lr, b, wd); KCHECK();
}
