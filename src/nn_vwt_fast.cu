/* vwt fast layer kernels (PF): bf16 tensor-core paths with the reference's semantics (src/vwt.h, src/nn_vwt.cu = the parity oracle).
   vtf_conv_fwd / vtf_conv_bwd have vt_conv_*'s signatures (channels-last fp32 tensors, weights [co][kd][kh][kw][ci], zero padding k / 2,
   output dims ceil(n / s), input voxel = o s - k / 2 + tap).

   Forward and backward-data: one implicit-GEMM kernel (mma m16n8k16 bf16 x bf16 -> fp32), fp32 operands converted to bf16 while staged
   (no bf16 copies of the activations): rows = output voxels (fwd: A[m][tap ci] = x at the tap's input voxel) or input voxels (bwd-data:
   A[m][tap c] = gy at the output voxel that reads m through that tap, when the stride divides it), columns = output / input channels.
   Weight gradient: bf16 im2col chunks (scratch) and gy chunks, cuBLASLt bf16 GEMM with fp32 accumulation straight into gw (beta 1). */
#include "vwt.h"
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned vb_of(long long n, int t) { long long b = (n + t - 1) / t; return (unsigned)(b > 2147483647LL ? 2147483647LL : b); }
static inline int odim(int n, int s) { return (n + s - 1) / s; }
#define VBK 32   /* K tile */

struct VtP {
    const float *x;          /* A source: fwd x [n][D][H][W][C]; bwd-data gy */
    const __nv_bfloat16 *w;  /* B [Np][Kp] bf16 */
    const float *bias;       /* fwd: [N] or nullptr */
    float *y;                /* [M][N] fp32 */
    int D, H, W, C;          /* A source dims */
    int Do, Ho, Wo;          /* row grid (fwd: output voxels; bwd: input voxels) */
    int k, sd, sh, sw, p;    /* kernel, strides, padding */
    int K, Kp, N;            /* K = k^3 C, padded; output channels */
    long long M;
};
__device__ __forceinline__ unsigned su32(const void *p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm4(unsigned *r, const void *p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(su32(p)));
}
__device__ __forceinline__ void mma_bf16(float *c, const unsigned *a, const unsigned *b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ unsigned pkbf(float a, float b) { __nv_bfloat162 h = __floats2bfloat162_rn(a, b); return *(unsigned *)&h; }
/* [rows][32 halfs]: 16-byte chunk q (0..3) of a row at q ^ ((row >> 1) & 3) (conflict-free ldmatrix) */
__device__ __forceinline__ int swz(int row, int q) { return row * VBK + ((q ^ ((row >> 1) & 3)) << 3); }

/* the A element source of row voxel (z, y, x) of sample nb (precomputed per thread row) for tap t and channel c: global index or -1 */
template <int MODE>
__device__ __forceinline__ long long a_src(const VtP &p, long long nb, int z, int y, int x, int tz, int ty, int tx) {
    if (MODE == 0) {
        const int zi = z * p.sd - p.p + tz, yi = y * p.sh - p.p + ty, xi = x * p.sw - p.p + tx;
        if (zi < 0 || zi >= p.D || yi < 0 || yi >= p.H || xi < 0 || xi >= p.W) return -1;
        return ((nb + zi) * p.H + yi) * (long long)p.W + xi;
    } else {   /* bwd-data: row = input voxel (z, y, x); output voxel o with o s - p + t = z */
        const int nz = z + p.p - tz, ny = y + p.p - ty, nx = x + p.p - tx;
        if (nz < 0 || ny < 0 || nx < 0) return -1;
        if (nz % p.sd || ny % p.sh || nx % p.sw) return -1;
        const int zo = nz / p.sd, yo = ny / p.sh, xo = nx / p.sw;
        if (zo >= p.D || yo >= p.H || xo >= p.W) return -1;
        return ((nb + zo) * p.H + yo) * (long long)p.W + xo;
    }
}

/* BM x BN tile, WARPS_M x WARPS_N warps; A staged from fp32 through registers (next K tile loaded during this one's MMAs), B (bf16) the same */
template <int MODE, int BM, int BN, int WARPS_M, int WARPS_N, bool VEC>
__global__ void __launch_bounds__(WARPS_M * WARPS_N * 32) vtf_igemm(VtP p) {
    constexpr int NTH = WARPS_M * WARPS_N * 32, WM = BM / WARPS_M, WN = BN / WARPS_N, MI = WM / 16, NI = WN / 8;
    constexpr int RS = NTH / 4, AP = BM / RS, BP = (BN + RS - 1) / RS;   /* 4 chunks of 8 per row; rows per pass; passes */
    __shared__ __align__(16) __nv_bfloat16 sA[2][BM * VBK], sB[2][BN * VBK];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm0 = (warp / WARPS_N) * WM, wn0 = (warp % WARPS_N) * WN;
    const long long m_base = (long long)blockIdx.x * BM; const int n_base = blockIdx.y * BN;
    const int q = tid & 3, r0 = tid >> 2;
    int zr[AP], yr[AP], xr[AP]; long long nr[AP];
#pragma unroll
    for (int i = 0; i < AP; i++) {
        const long long m = m_base + r0 + i * RS;
        if (m < p.M) { long long t = m; xr[i] = (int)(t % p.Wo); t /= p.Wo; yr[i] = (int)(t % p.Ho); t /= p.Ho; zr[i] = (int)(t % p.Do); t /= p.Do; nr[i] = t * p.D; }
        else { zr[i] = -(1 << 28); yr[i] = xr[i] = 0; nr[i] = 0; }
    }
    const int nk = p.Kp / VBK;
    uint4 ra[AP], rb[BP];
    auto gload = [&](int ks) {
        const int k0 = ks * VBK + q * 8;
#pragma unroll
        for (int i = 0; i < AP; i++) {
            float v[8];
            if (VEC) {   /* C % 8 == 0: the 8 channels of one tap */
                const int t = k0 / p.C, c = k0 - t * p.C;
                long long s = -1;
                if (k0 < p.K && zr[i] > -(1 << 27)) s = a_src<MODE>(p, nr[i], zr[i], yr[i], xr[i], t / (p.k * p.k), (t / p.k) % p.k, t % p.k);
                if (s >= 0) { const float4 a = __ldg((const float4 *)(p.x + s * p.C + c)), b = __ldg((const float4 *)(p.x + s * p.C + c) + 1);
                              v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w; }
                else for (int j = 0; j < 8; j++) v[j] = 0.f;
            } else {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const int kk = k0 + j; v[j] = 0.f;
                    if (kk < p.K && zr[i] > -(1 << 27)) {
                        const int t = kk / p.C, c = kk - t * p.C;
                        const long long s = a_src<MODE>(p, nr[i], zr[i], yr[i], xr[i], t / (p.k * p.k), (t / p.k) % p.k, t % p.k);
                        if (s >= 0) v[j] = __ldg(p.x + s * p.C + c);
                    }
                }
            }
            ra[i] = make_uint4(pkbf(v[0], v[1]), pkbf(v[2], v[3]), pkbf(v[4], v[5]), pkbf(v[6], v[7]));
        }
#pragma unroll
        for (int i = 0; i < BP; i++) {
            const int row = r0 + i * RS;
            rb[i] = row < BN ? __ldg((const uint4 *)(p.w + (size_t)(n_base + row) * p.Kp + ks * VBK + q * 8)) : make_uint4(0u, 0u, 0u, 0u);
        }
    };
    auto sstore = [&](int slot) {
#pragma unroll
        for (int i = 0; i < AP; i++) *(uint4 *)(sA[slot] + swz(r0 + i * RS, q)) = ra[i];
#pragma unroll
        for (int i = 0; i < BP; i++) { const int row = r0 + i * RS; if (row < BN) *(uint4 *)(sB[slot] + swz(row, q)) = rb[i]; }
    };
    float acc[MI][NI][4];
#pragma unroll
    for (int a = 0; a < MI; a++)
#pragma unroll
        for (int b = 0; b < NI; b++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[a][b][c] = 0.f;
    gload(0); sstore(0); __syncthreads();
    for (int ks = 0; ks < nk; ks++) {
        const int slot = ks & 1;
        if (ks + 1 < nk) gload(ks + 1);
        const __nv_bfloat16 *A = sA[slot], *B = sB[slot];
#pragma unroll
        for (int kk = 0; kk < VBK / 16; kk++) {
            unsigned af[MI][4], bf[NI][2];
#pragma unroll
            for (int mi = 0; mi < MI; mi++) { const int row = wm0 + mi * 16 + (lane & 15); ldsm4(af[mi], A + swz(row, kk * 2 + (lane >> 4))); }
#pragma unroll
            for (int nj = 0; nj < NI / 2; nj++) {
                const int row = wn0 + nj * 16 + (lane & 7) + ((lane >> 4) << 3); unsigned r[4];
                ldsm4(r, B + swz(row, kk * 2 + ((lane >> 3) & 1)));
                bf[2 * nj][0] = r[0]; bf[2 * nj][1] = r[1]; bf[2 * nj + 1][0] = r[2]; bf[2 * nj + 1][1] = r[3];
            }
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int ni = 0; ni < NI; ni++) mma_bf16(acc[mi][ni], af[mi], bf[ni]);
        }
        if (ks + 1 < nk) sstore(slot ^ 1);
        __syncthreads();
    }
    const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
    for (int ni = 0; ni < NI; ni++) {
        const int co = n_base + wn0 + ni * 8 + t4 * 2;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const long long m = m_base + wm0 + mi * 16 + g + h * 8; if (m >= p.M) continue;
            float *yr_ = p.y + m * p.N;
            if (co < p.N) yr_[co] = acc[mi][ni][2 * h] + (p.bias ? p.bias[co] : 0.f);
            if (co + 1 < p.N) yr_[co + 1] = acc[mi][ni][2 * h + 1] + (p.bias ? p.bias[co + 1] : 0.f);
        }
    }
}

/* weights fp32 [co][T][ci] -> bf16 B matrices: fwd [Np][Kp] (row co, K = t ci + c); bwd-data [Np = ci][Kp] (row ci, K = t co + c) */
__global__ void k_wprep(const float *__restrict__ w, int co, int T, int ci, int Kp, int Np, int bwd, __nv_bfloat16 *__restrict__ B) {
    const long long n = (long long)Np * Kp;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const int r = (int)(i / Kp), kk = (int)(i % Kp); float v = 0.f;
        if (!bwd) { if (r < co && kk < T * ci) { const int t = kk / ci, c = kk % ci; v = w[((size_t)r * T + t) * ci + c]; } }
        else { if (r < ci && kk < T * co) { const int t = kk / co, c = kk % co; v = w[((size_t)c * T + t) * ci + r]; } }
        B[i] = __float2bfloat16_rn(v);
    }
}

template <int MODE, int BM, int BN, int WM, int WN> static void igemm_go(const VtP &p) {
    const dim3 g((unsigned)((p.M + BM - 1) / BM), (unsigned)((p.N + BN - 1) / BN));
    if (p.C % 8 == 0) vtf_igemm<MODE, BM, BN, WM, WN, true><<<g, WM * WN * 32>>>(p);
    else vtf_igemm<MODE, BM, BN, WM, WN, false><<<g, WM * WN * 32>>>(p);
}
template <int MODE> static void igemm_launch(const VtP &p) {
    if (p.N <= 8) igemm_go<MODE, 128, 16, 4, 1>(p);          /* (the 4-output head: one 16-column tile) */
    else if (p.N <= 32) igemm_go<MODE, 128, 32, 4, 1>(p);
    else if (p.N <= 64) igemm_go<MODE, 128, 64, 4, 2>(p);
    else igemm_go<MODE, 128, 128, 2, 4>(p);
}
static int ntile(int N) { return N <= 8 ? 16 : N <= 32 ? 32 : N <= 64 ? 64 : 128; }

/* ---- scratch: B matrices (fwd + bwd) and the weight gradient's bf16 im2col / gy chunks ---- */
static const size_t WCHUNK = 192ull << 20, GCHUNK = 64ull << 20;   /* bf16 im2col / gy chunk bytes */
static size_t wbytes_max(int cimax, int comax) { const int K = 27 * cimax, Kp = (K + VBK - 1) / VBK * VBK; return ((size_t)128 * 2 * Kp * 2 + 255) & ~(size_t)255; }
extern "C" size_t vtf_conv_scratch(vt_t x, int co, int k, int sd, int sh, int sw) {
    (void)x; (void)co; (void)k; (void)sd; (void)sh; (void)sw;
    /* every layer of the vwt net (ci, co <= 256): two weight matrices, the im2col chunk and its gy chunk (<= 256 channels per row) */
    return 2 * wbytes_max(256, 256) + WCHUNK + GCHUNK + 4096;
}

static cublasLtHandle_t g_lt; static void *g_lws; static const size_t LWS = 32u << 20;
/* gw (row-major [co][K], = column-major K x co) += col^T gy over R rows: col bf16 [R][ldc], gy bf16 [R][co] */
static int wgrad_gemm(int K, int co, int R, const __nv_bfloat16 *col, int ldc, const __nv_bfloat16 *gyb, float *gw) {
    if (!g_lt) { if (cublasLtCreate(&g_lt) != CUBLAS_STATUS_SUCCESS || cudaMalloc(&g_lws, LWS) != cudaSuccess) return -1; }
    cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t A, B, D;
    cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    const cublasOperation_t on = CUBLAS_OP_N, ot = CUBLAS_OP_T;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &on, sizeof on);
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &ot, sizeof ot);
    cublasLtMatrixLayoutCreate(&A, CUDA_R_16BF, K, R, ldc);     /* col: column-major K x R */
    cublasLtMatrixLayoutCreate(&B, CUDA_R_16BF, co, R, co);     /* gy: column-major co x R, op T -> R x co */
    cublasLtMatrixLayoutCreate(&D, CUDA_R_32F, K, co, K);
    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &LWS, sizeof LWS);
    cublasLtMatmulHeuristicResult_t h; int nr = 0, rc = -1;
    if (cublasLtMatmulAlgoGetHeuristic(g_lt, op, A, B, D, D, pref, 1, &h, &nr) == CUBLAS_STATUS_SUCCESS && nr > 0) {
        const float one = 1.f;
        rc = cublasLtMatmul(g_lt, op, &one, col, A, gyb, B, &one, gw, D, gw, D, &h.algo, g_lws, LWS, 0) == CUBLAS_STATUS_SUCCESS ? 0 : -1;
    }
    cublasLtMatmulPreferenceDestroy(pref); cublasLtMatrixLayoutDestroy(A); cublasLtMatrixLayoutDestroy(B); cublasLtMatrixLayoutDestroy(D); cublasLtMatmulDescDestroy(op);
    if (rc) fprintf(stderr, "vtf_conv_bwd: weight-gradient GEMM %d x %d x %d failed\n", K, co, R);
    return rc;
}
/* bf16 im2col rows [r0, r0 + R) (ldc >= K, zero tail) */
__global__ void k_im2col_bf(const float *__restrict__ x, int D, int H, int W, int C, int k, int sd, int sh, int sw, int Do, int Ho, int Wo,
                            long long r0, long long R, int ldc, __nv_bfloat16 *__restrict__ col) {
    const int K = k * k * k * C, p = k / 2; const long long tot = R * ldc;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < tot; i += (long long)gridDim.x * blockDim.x) {
        const long long r = r0 + i / ldc; const int qk = (int)(i % ldc); float v = 0.f;
        if (qk < K) {
            const int c = qk % C, t = qk / C, kx = t % k, ky = (t / k) % k, kz = t / (k * k);
            long long rr = r; const int ow = (int)(rr % Wo); rr /= Wo; const int oh = (int)(rr % Ho); rr /= Ho; const int od = (int)(rr % Do); const long long n = rr / Do;
            const int z = od * sd - p + kz, y = oh * sh - p + ky, xx = ow * sw - p + kx;
            if (z >= 0 && z < D && y >= 0 && y < H && xx >= 0 && xx < W) v = x[(((n * D + z) * H + y) * W + xx) * (long long)C + c];
        }
        col[i] = __float2bfloat16_rn(v);
    }
}
/* the same for C % 8 == 0: a thread per 8 channels of one tap (two float4 loads, one 16-byte store); ldc % 8 == 0 */
__global__ void k_im2col_bf8(const float *__restrict__ x, int D, int H, int W, int C, int k, int sd, int sh, int sw, int Do, int Ho, int Wo,
                             long long r0, long long R, int ldc, __nv_bfloat16 *__restrict__ col) {
    const int K = k * k * k * C, p = k / 2, c8 = ldc / 8; const long long tot = R * c8;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < tot; i += (long long)gridDim.x * blockDim.x) {
        const long long r = r0 + i / c8; const int qk = (int)(i % c8) * 8;
        float4 a = make_float4(0.f, 0.f, 0.f, 0.f), b = a;
        if (qk < K) {
            const int c = qk % C, t = qk / C, kx = t % k, ky = (t / k) % k, kz = t / (k * k);
            long long rr = r; const int ow = (int)(rr % Wo); rr /= Wo; const int oh = (int)(rr % Ho); rr /= Ho; const int od = (int)(rr % Do); const long long n = rr / Do;
            const int z = od * sd - p + kz, y = oh * sh - p + ky, xx = ow * sw - p + kx;
            if (z >= 0 && z < D && y >= 0 && y < H && xx >= 0 && xx < W) {
                const float4 *s = (const float4 *)(x + (((n * D + z) * H + y) * W + xx) * (long long)C + c); a = __ldg(s); b = __ldg(s + 1);
            }
        }
        *(uint4 *)(col + (r - r0) * ldc + qk) = make_uint4(pkbf(a.x, a.y), pkbf(a.z, a.w), pkbf(b.x, b.y), pkbf(b.z, b.w));
    }
}
__global__ void k_f2bf(const float *__restrict__ x, __nv_bfloat16 *__restrict__ y, long long n) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) y[i] = __float2bfloat16_rn(x[i]);
}
/* gb[c] += sum over rows of g[r][c]: block = per x C threads (thread: channel tid % C, rows tid / C + per j) */
__global__ void k_colsum_f(const float *__restrict__ g, long long rows, int C, int per, float *__restrict__ gb) {
    extern __shared__ float sh[];
    for (int c = threadIdx.x; c < C; c += blockDim.x) sh[c] = 0.f;
    __syncthreads();
    const int c = threadIdx.x % C; float a = 0.f;
    for (long long r = blockIdx.x * (long long)per + threadIdx.x / C; r < rows; r += (long long)gridDim.x * per) a += g[r * C + c];
    atomicAdd(&sh[c], a);
    __syncthreads();
    for (int k = threadIdx.x; k < C; k += blockDim.x) atomicAdd(&gb[k], sh[k]);
}
static unsigned cap_blocks(long long n) { unsigned b = vb_of(n, 256); return b > 4096 ? 4096 : b; }

static VtP make_p(int mode, vt_t a, int k, int sd, int sh, int sw, int Do, int Ho, int Wo, int N, const __nv_bfloat16 *B, const float *bias, float *y) {
    VtP p; p.x = a.p; p.w = B; p.bias = bias; p.y = y; p.D = a.d; p.H = a.h; p.W = a.w; p.C = a.c; p.Do = Do; p.Ho = Ho; p.Wo = Wo;
    p.k = k; p.sd = sd; p.sh = sh; p.sw = sw; p.p = k / 2; p.K = k * k * k * a.c; p.Kp = (p.K + VBK - 1) / VBK * VBK; p.N = N;
    p.M = (long long)a.n * Do * Ho * Wo; (void)mode; return p;
}

extern "C" int vtf_conv_fwd(vt_t x, const float *w, const float *b, int k, int sd, int sh, int sw, vt_t y, void *scratch) {
    const int Do = odim(x.d, sd), Ho = odim(x.h, sh), Wo = odim(x.w, sw), co = y.c;
    if (y.d != Do || y.h != Ho || y.w != Wo || y.n != x.n) { fprintf(stderr, "vtf_conv_fwd: output shape mismatch\n"); return -1; }
    if (co > 256 || x.c > 256) { fprintf(stderr, "vtf_conv_fwd: %d -> %d unsupported\n", x.c, co); return -1; }
    const int K = k * k * k * x.c, Kp = (K + VBK - 1) / VBK * VBK, Np = (co + ntile(co) - 1) / ntile(co) * ntile(co);
    __nv_bfloat16 *B = (__nv_bfloat16 *)scratch;
    k_wprep<<<cap_blocks((long long)Np * Kp), 256>>>(w, co, k * k * k, x.c, Kp, Np, 0, B);
    VtP p = make_p(0, x, k, sd, sh, sw, Do, Ho, Wo, co, B, b, y.p);
    if (p.M > 0) igemm_launch<0>(p);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int vtf_conv_bwd(vt_t x, vt_t gy, const float *w, int k, int sd, int sh, int sw, vt_t gx, float *gw, float *gb, void *scratch) {
    const int co = gy.c, ci = x.c, T = k * k * k;
    const long long rows = (long long)gy.n * gy.d * gy.h * gy.w;
    if (gb) { const int per = 256 / co > 0 ? 256 / co : 1; k_colsum_f<<<1024, per * co, co * sizeof(float)>>>(gy.p, rows, co, per, gb); }
    char *s = (char *)scratch;
    if (gx.p) {   /* backward-data: rows = input voxels, columns = ci, K = taps x co */
        const int K = T * co, Kp = (K + VBK - 1) / VBK * VBK, Np = (ci + ntile(ci) - 1) / ntile(ci) * ntile(ci);
        __nv_bfloat16 *B = (__nv_bfloat16 *)s;
        k_wprep<<<cap_blocks((long long)Np * Kp), 256>>>(w, co, T, ci, Kp, Np, 1, B);
        VtP p = make_p(1, gy, k, sd, sh, sw, x.d, x.h, x.w, ci, B, nullptr, gx.p);
        p.M = (long long)x.n * x.d * x.h * x.w;
        if (p.M > 0) igemm_launch<1>(p);
    }
    if (gw) {   /* weight gradient: chunks of output rows, bf16 im2col (ld = K rounded to 8) and gy, cuBLASLt into gw */
        const int K = T * ci, ldc = (K + 7) / 8 * 8;
        __nv_bfloat16 *col = (__nv_bfloat16 *)(s + 2 * wbytes_max(256, 256));
        long long cr = (long long)(WCHUNK / ((size_t)ldc * 2)); cr = cr / 256 * 256; if (cr < 256) cr = 256;
        __nv_bfloat16 *gyb = (__nv_bfloat16 *)((char *)col + WCHUNK);
        const long long gmax = (long long)(GCHUNK / ((size_t)co * 2)) / 256 * 256; if (cr > gmax) cr = gmax;
        for (long long r0 = 0; r0 < rows; r0 += cr) {
            const long long R = rows - r0 < cr ? rows - r0 : cr;
            if (k == 1 && sd == 1 && sh == 1 && sw == 1 && ldc == K) k_f2bf<<<cap_blocks(R * K), 256>>>(x.p + r0 * K, col, R * K);
            else if (ci % 8 == 0) k_im2col_bf8<<<cap_blocks(R * ldc / 8), 256>>>(x.p, x.d, x.h, x.w, ci, k, sd, sh, sw, gy.d, gy.h, gy.w, r0, R, ldc, col);
            else k_im2col_bf<<<cap_blocks(R * ldc), 256>>>(x.p, x.d, x.h, x.w, ci, k, sd, sh, sw, gy.d, gy.h, gy.w, r0, R, ldc, col);
            k_f2bf<<<cap_blocks(R * co), 256>>>(gy.p + r0 * co, gyb, R * co);
            if (wgrad_gemm(K, co, (int)R, col, ldc, gyb, gw)) return -1;
        }
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
