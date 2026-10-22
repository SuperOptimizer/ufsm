/* Generic MX-fp4 / MX-fp8 GEMM for row-major fp32 operands (the attention stage's linears, the DINO / iBOT head MLPs, the prototype
   layer): C[M][N] = alpha op(A) op(B) + beta C with A [M][K] (ta 0) or [K][M] (ta 1), B [K][N] (tb 0) or [N][K] (tb 1), as at_gemm.
   Each 64 x 64 x 64 step stages both operands into shared memory quantised along K: every 32 K-values of an A row / B column get one
   ue8m0 scale (2^e, e = ceil(log2(amax / qmax))) and round to nearest e2m1 (fp4, qmax 6) or e4m3 (fp8, qmax 448); the tensor cores
   multiply with block scaling (m16n8k64 kind::mxf4 / m16n8k32 kind::mxf8f6f4) and accumulate in fp32. Out-of-range rows / K are zero. */
#include "lp_common.cuh"

#define GBM 64
#define GBN 64
#define GBK 64
template <int BITS> struct gq;
template <> struct gq<4> { static constexpr int RB = GBK / 2; static constexpr float IQ = 1.f / 6.f; };    /* bytes per staged row of 64 K */
template <> struct gq<8> { static constexpr int RB = GBK; static constexpr float IQ = 1.f / 448.f; };

/* stage rows [r0, r0 + 64) x K [k0, k0 + 64) of an operand: row-major [rows][K] (tr 0, ld = row stride) or [K][rows] (tr 1);
   one thread per (row, 32-K block) */
template <int BITS, int TR>
__device__ __forceinline__ void stage(const float *__restrict__ X, int ld, int rows, int K, int r0, int k0, uint8_t *sq, uint8_t *ss, int tid) {
    constexpr int RB = gq<BITS>::RB;
    const int row = tid & 63, kb = tid >> 6, r = r0 + row, kbase = k0 + kb * 32;   /* (128 threads: rows x 2 blocks) */
    float v[32];
#pragma unroll
    for (int j = 0; j < 32; j++) {
        const int k = kbase + j;
        v[j] = r < rows && k < K ? (TR ? X[(size_t)k * ld + r] : X[(size_t)r * ld + k]) : 0.f;
    }
    float am = 0.f;
#pragma unroll
    for (int j = 0; j < 32; j++) am = fmaxf(am, fabsf(v[j]));
    const int e = mx_exp(am, gq<BITS>::IQ); const float m = exp2i(-e);
    uint8_t *q = sq + row * (RB + 16) + kb * (RB / 2);   /* rows padded by 16 bytes (bank spread) */
    if constexpr (BITS == 4) {
#pragma unroll
        for (int w = 0; w < 4; w++) *(unsigned *)(q + 4 * w) = cvt_e2m1x8(v + 8 * w, m);
    } else {
#pragma unroll
        for (int w = 0; w < 8; w++) *(unsigned *)(q + 4 * w) = cvt_e4m3x4(v[4 * w] * m, v[4 * w + 1] * m, v[4 * w + 2] * m, v[4 * w + 3] * m);
    }
    ss[row * 2 + kb] = (uint8_t)(e + 127);
}

template <int BITS, int TA, int TB>
__global__ void __launch_bounds__(128) mx_gemm_k(const float *__restrict__ A, int lda, const float *__restrict__ B, int ldb, float *__restrict__ C, int ldc,
                                                  int M, int N, int K, float alpha, float beta) {
    constexpr int RB = gq<BITS>::RB, RS = RB + 16;
    __shared__ __align__(16) uint8_t sA[GBM * RS], sB[GBN * RS], sAs[GBM * 2], sBs[GBN * 2];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int m0 = blockIdx.y * GBM, n0 = blockIdx.x * GBN, wm = (warp >> 1) * 32, wn = (warp & 1) * 32;
    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int q = 0; q < 4; q++) acc[i][j][q] = 0.f;
    for (int k0 = 0; k0 < K; k0 += GBK) {
        __syncthreads();
        stage<BITS, TA>(A, lda, M, K, m0, k0, sA, sAs, tid);
        stage<BITS, TB ? 0 : 1>(B, ldb, N, K, n0, k0, sB, sBs, tid);   /* B rows = output columns: [N][K] (tb 1) is row-major, [K][N] (tb 0) transposed */
        __syncthreads();
        if constexpr (BITS == 4) {
            unsigned af[2][4], sa[2], bf[4][2], sb[4];
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const int r = wm + i * 16 + g;
                af[i][0] = *(const unsigned *)(sA + r * RS + 4 * t); af[i][1] = *(const unsigned *)(sA + (r + 8) * RS + 4 * t);
                af[i][2] = *(const unsigned *)(sA + r * RS + 16 + 4 * t); af[i][3] = *(const unsigned *)(sA + (r + 8) * RS + 16 + 4 * t);
                const int rs = wm + i * 16 + g + 8 * (t & 1);
                sa[i] = (unsigned)sAs[rs * 2] | ((unsigned)sAs[rs * 2 + 1] << 8);
            }
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int c = wn + j * 8 + g;
                bf[j][0] = *(const unsigned *)(sB + c * RS + 4 * t); bf[j][1] = *(const unsigned *)(sB + c * RS + 16 + 4 * t);
                sb[j] = (unsigned)sBs[c * 2] | ((unsigned)sBs[c * 2 + 1] << 8);
            }
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) mma_f4(acc[i][j], af[i], bf[j], sa[i], sb[j]);
        } else {
#pragma unroll
            for (int kh = 0; kh < 2; kh++) {   /* two k32 steps, one scale block each */
                unsigned af[2][4], sa[2], bf[4][2], sb[4];
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    const int r = wm + i * 16 + g;
                    af[i][0] = *(const unsigned *)(sA + r * RS + kh * 32 + 4 * t); af[i][1] = *(const unsigned *)(sA + (r + 8) * RS + kh * 32 + 4 * t);
                    af[i][2] = *(const unsigned *)(sA + r * RS + kh * 32 + 16 + 4 * t); af[i][3] = *(const unsigned *)(sA + (r + 8) * RS + kh * 32 + 16 + 4 * t);
                    const int rs = wm + i * 16 + g + 8 * (t & 1);
                    sa[i] = sAs[rs * 2 + kh];
                }
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int c = wn + j * 8 + g;
                    bf[j][0] = *(const unsigned *)(sB + c * RS + kh * 32 + 4 * t); bf[j][1] = *(const unsigned *)(sB + c * RS + kh * 32 + 16 + 4 * t);
                    sb[j] = sBs[c * 2 + kh];
                }
#pragma unroll
                for (int i = 0; i < 2; i++)
#pragma unroll
                    for (int j = 0; j < 4; j++) mma_f8(acc[i][j], af[i], bf[j], sa[i], sb[j]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const int r = m0 + wm + i * 16 + g + 8 * (q >> 1), c = n0 + wn + j * 8 + 2 * t + (q & 1);
                if (r < M && c < N) { float *p = C + (size_t)r * ldc + c; *p = alpha * acc[i][j][q] + (beta != 0.f ? beta * *p : 0.f); }
            }
}

/* ---- the packed path: each operand quantised once (a pre-pass into [rows][Kp] MX rows, Kp = K rounded up to 64), then a GEMM over the
   packed bytes (the 131k-row prototype layer: no per-tile re-quantisation of B) ---- */
template <int BITS, int TR>
__global__ void mx_pack_k(const float *__restrict__ X, int ld, int rows, int K, int Kp, uint8_t *__restrict__ q, uint8_t *__restrict__ sc) {
    constexpr float IQ = gq<BITS>::IQ; constexpr int BPB = BITS == 4 ? 16 : 32;   /* bytes per 32-value block */
    const long long nblk = (long long)rows * (Kp / 32);
    for (long long b = blockIdx.x * (long long)blockDim.x + threadIdx.x; b < nblk; b += (long long)gridDim.x * blockDim.x) {
        const int r = (int)(b / (Kp / 32)), kb = (int)(b % (Kp / 32)), k0 = kb * 32;
        float v[32];
#pragma unroll
        for (int j = 0; j < 32; j++) { const int k = k0 + j; v[j] = k < K ? (TR ? X[(size_t)k * ld + r] : X[(size_t)r * ld + k]) : 0.f; }
        float am = 0.f;
#pragma unroll
        for (int j = 0; j < 32; j++) am = fmaxf(am, fabsf(v[j]));
        const int e = mx_exp(am, IQ); const float m = exp2i(-e);
        uint8_t *o = q + (size_t)r * (Kp / 32) * BPB + (size_t)kb * BPB;
        if constexpr (BITS == 4) {
#pragma unroll
            for (int w = 0; w < 4; w++) *(unsigned *)(o + 4 * w) = cvt_e2m1x8(v + 8 * w, m);
        } else {
#pragma unroll
            for (int w = 0; w < 8; w++) *(unsigned *)(o + 4 * w) = cvt_e4m3x4(v[4 * w] * m, v[4 * w + 1] * m, v[4 * w + 2] * m, v[4 * w + 3] * m);
        }
        sc[(size_t)r * (Kp / 32) + kb] = (uint8_t)(e + 127);
    }
}
template <int BITS>
__global__ void __launch_bounds__(128) mx_gemm_p_k(const uint8_t *__restrict__ Aq, const uint8_t *__restrict__ As, const uint8_t *__restrict__ Bq, const uint8_t *__restrict__ Bs,
                                                   float *__restrict__ C, int ldc, int M, int N, int Kp, float alpha, float beta) {
    constexpr int RB = gq<BITS>::RB, RS = RB + 16, BPB = BITS == 4 ? 16 : 32;
    __shared__ __align__(16) uint8_t sA[GBM * RS], sB[GBN * RS], sAs[GBM * 2], sBs[GBN * 2];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int m0 = blockIdx.y * GBM, n0 = blockIdx.x * GBN, wm = (warp >> 1) * 32, wn = (warp & 1) * 32, nkb = Kp / 32;
    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int q = 0; q < 4; q++) acc[i][j][q] = 0.f;
    for (int k0 = 0; k0 < Kp; k0 += GBK) {
        __syncthreads();
        {   /* copy: 64 rows x 2 blocks per operand, one (row, block) per thread, 16-byte words */
            const int row = tid & 63, kb = tid >> 6, blk = k0 / 32 + kb;
            const int ra = m0 + row, rb = n0 + row;
            uint4 *da = (uint4 *)(sA + row * RS + kb * BPB), *db = (uint4 *)(sB + row * RS + kb * BPB);
            const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
            for (int w = 0; w < BPB / 16; w++) {
                da[w] = ra < M ? ((const uint4 *)(Aq + ((size_t)ra * nkb + blk) * BPB))[w] : z;
                db[w] = rb < N ? ((const uint4 *)(Bq + ((size_t)rb * nkb + blk) * BPB))[w] : z;
            }
            sAs[row * 2 + kb] = ra < M ? As[(size_t)ra * nkb + blk] : 127;
            sBs[row * 2 + kb] = rb < N ? Bs[(size_t)rb * nkb + blk] : 127;
        }
        __syncthreads();
        if constexpr (BITS == 4) {
            unsigned af[2][4], sa[2], bf[4][2], sb[4];
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const int r = wm + i * 16 + g;
                af[i][0] = *(const unsigned *)(sA + r * RS + 4 * t); af[i][1] = *(const unsigned *)(sA + (r + 8) * RS + 4 * t);
                af[i][2] = *(const unsigned *)(sA + r * RS + 16 + 4 * t); af[i][3] = *(const unsigned *)(sA + (r + 8) * RS + 16 + 4 * t);
                const int rs = wm + i * 16 + g + 8 * (t & 1);
                sa[i] = (unsigned)sAs[rs * 2] | ((unsigned)sAs[rs * 2 + 1] << 8);
            }
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int c = wn + j * 8 + g;
                bf[j][0] = *(const unsigned *)(sB + c * RS + 4 * t); bf[j][1] = *(const unsigned *)(sB + c * RS + 16 + 4 * t);
                sb[j] = (unsigned)sBs[c * 2] | ((unsigned)sBs[c * 2 + 1] << 8);
            }
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) mma_f4(acc[i][j], af[i], bf[j], sa[i], sb[j]);
        } else {
#pragma unroll
            for (int kh = 0; kh < 2; kh++) {
                unsigned af[2][4], sa[2], bf[4][2], sb[4];
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    const int r = wm + i * 16 + g;
                    af[i][0] = *(const unsigned *)(sA + r * RS + kh * 32 + 4 * t); af[i][1] = *(const unsigned *)(sA + (r + 8) * RS + kh * 32 + 4 * t);
                    af[i][2] = *(const unsigned *)(sA + r * RS + kh * 32 + 16 + 4 * t); af[i][3] = *(const unsigned *)(sA + (r + 8) * RS + kh * 32 + 16 + 4 * t);
                    sa[i] = sAs[(wm + i * 16 + g + 8 * (t & 1)) * 2 + kh];
                }
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int c = wn + j * 8 + g;
                    bf[j][0] = *(const unsigned *)(sB + c * RS + kh * 32 + 4 * t); bf[j][1] = *(const unsigned *)(sB + c * RS + kh * 32 + 16 + 4 * t);
                    sb[j] = sBs[c * 2 + kh];
                }
#pragma unroll
                for (int i = 0; i < 2; i++)
#pragma unroll
                    for (int j = 0; j < 4; j++) mma_f8(acc[i][j], af[i], bf[j], sa[i], sb[j]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const int r = m0 + wm + i * 16 + g + 8 * (q >> 1), c = n0 + wn + j * 8 + 2 * t + (q & 1);
                if (r < M && c < N) { float *p = C + (size_t)r * ldc + c; *p = alpha * acc[i][j][q] + (beta != 0.f ? beta * *p : 0.f); }
            }
}
static uint8_t *g_pk = nullptr; static size_t g_pkn = 0;
/* the packed path for large operands: pack both, then the packed GEMM (bits 4 / 8); 0, or -1 */
static int mx_gemm_packed(int bits, int ta, int tb, int M, int N, int K, const float *A, const float *B, float *C, float beta) {
    const int Kp = (K + 63) / 64 * 64, BPB = bits == 4 ? 16 : 32;
    const size_t qa = (size_t)M * (Kp / 32) * BPB, qb = (size_t)N * (Kp / 32) * BPB, sa = (size_t)M * (Kp / 32), sb = (size_t)N * (Kp / 32);
    const size_t need = ((qa + 255) & ~(size_t)255) + ((qb + 255) & ~(size_t)255) + ((sa + 255) & ~(size_t)255) + sb + 256;
    if (need > g_pkn) { if (g_pk) cudaFree(g_pk); if (cudaMalloc(&g_pk, need) != cudaSuccess) { g_pk = nullptr; g_pkn = 0; return -1; } g_pkn = need; }
    uint8_t *Aq = g_pk, *Bq = Aq + ((qa + 255) & ~(size_t)255), *As = Bq + ((qb + 255) & ~(size_t)255), *Bs = As + ((sa + 255) & ~(size_t)255);
    const int lda = ta ? M : K, ldb = tb ? K : N;
    const long long na = (long long)M * (Kp / 32), nb = (long long)N * (Kp / 32);
    const unsigned ga = (unsigned)((na + 255) / 256 < 65535 ? (na + 255) / 256 : 65535), gb = (unsigned)((nb + 255) / 256 < 65535 ? (nb + 255) / 256 : 65535);
#define PK(BI) do { if (ta) mx_pack_k<BI, 1><<<ga, 256>>>(A, lda, M, K, Kp, Aq, As); else mx_pack_k<BI, 0><<<ga, 256>>>(A, lda, M, K, Kp, Aq, As); \
                    if (tb) mx_pack_k<BI, 0><<<gb, 256>>>(B, ldb, N, K, Kp, Bq, Bs); else mx_pack_k<BI, 1><<<gb, 256>>>(B, ldb, N, K, Kp, Bq, Bs); } while (0)
    const dim3 grid((unsigned)((N + GBN - 1) / GBN), (unsigned)((M + GBM - 1) / GBM));
    if (bits == 4) { PK(4); mx_gemm_p_k<4><<<grid, 128>>>(Aq, As, Bq, Bs, C, N, M, N, Kp, 1.f, beta); }
    else { PK(8); mx_gemm_p_k<8><<<grid, 128>>>(Aq, As, Bq, Bs, C, N, M, N, Kp, 1.f, beta); }
#undef PK
    LPCK();
    return 0;
}

/* bits 4 / 8; returns 0, or -1 (unsupported) */
extern "C" int lp_mx_gemm(int bits, int ta, int tb, int M, int N, int K, const float *A, const float *B, float *C, float beta) {
    if (M <= 0 || N <= 0) return 0;
    static int mode = -1; if (mode < 0) { const char *e = getenv("UFSM_MX_GEMM_PACKED"); mode = e ? atoi(e) : 2; }   /* 0 tiles, 1 packed, 2 packed when either grid side > 4 tiles */
    if ((bits == 4 || bits == 8) && (mode == 1 || (mode == 2 && (M > 4 * GBM || N > 4 * GBN))) && !mx_gemm_packed(bits, ta, tb, M, N, K, A, B, C, beta)) return 0;
    const dim3 grid((unsigned)((N + GBN - 1) / GBN), (unsigned)((M + GBM - 1) / GBM));
    const int lda = ta ? M : K, ldb = tb ? K : N;
#define LMG(BI, TA_, TB_) mx_gemm_k<BI, TA_, TB_><<<grid, 128>>>(A, lda, B, ldb, C, N, M, N, K, 1.f, beta)
    if (bits == 4) { if (!ta && !tb) LMG(4, 0, 0); else if (!ta && tb) LMG(4, 0, 1); else if (ta && !tb) LMG(4, 1, 0); else LMG(4, 1, 1); }
    else if (bits == 8) { if (!ta && !tb) LMG(8, 0, 0); else if (!ta && tb) LMG(8, 0, 1); else if (ta && !tb) LMG(8, 1, 0); else LMG(8, 1, 1); }
    else return -1;
#undef LMG
    LPCK();
    return 0;
}
