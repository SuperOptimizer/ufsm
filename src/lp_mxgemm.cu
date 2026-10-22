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

/* bits 4 / 8; returns 0, or -1 (unsupported) */
extern "C" int lp_mx_gemm(int bits, int ta, int tb, int M, int N, int K, const float *A, const float *B, float *C, float beta) {
    if (M <= 0 || N <= 0) return 0;
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
