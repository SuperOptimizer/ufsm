/* Self-test of the software MX paths (lp_common.cuh LP_SW, mx_sw.cuh): every fp32 input through the e4m3 / e2m1 encoders,
   every e4m3 pair / e2m1 byte through the f16x2 decoders, and random fragments through the emulated block-scaled MMAs.
   References: an independent table search (encoders), the per-element decode (decoders) and a scalar dot product over the
   documented fragment layout (MMAs) in every build; on sm_120a additionally the hardware instructions, bit-exact for the
   conversions and to fp32 summation order for the MMAs. tests/test_mx_sw.c runs it (both builds). */
#include "lp_common.cuh"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
#define MXSW_HW 1
#endif

/* counters: [0] e4m3 sw != ref, [1] e4m3 sw != hw, [2] e2m1 sw != ref, [3] e2m1 sw != hw, [4] dec e4m3, [5] dec e2m1, [6] mma vs
   ref, [7] mma vs hw; first offending inputs in ex[counter][0..3] */
struct mxsw_res { unsigned long long n[8]; unsigned ex[8][4]; float worst[2]; };
__device__ void mxsw_note(mxsw_res *r, int k, unsigned v) {
    const unsigned long long i = atomicAdd(&r->n[k], 1ull);
    if (i < 4) r->ex[k][i] = v;
}
/* table-search references (no arithmetic on x: comparisons against the exact grid values and midpoints) */
__device__ unsigned ref_e4m3(float x) {
    const unsigned u = __float_as_uint(x), s = (u >> 24) & 0x80u;
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fu;
    const float a = fabsf(x);
    if (a >= 448.f) return s | 0x7eu;
    int lo = 0, hi = 126;                 /* largest code c with val(c) <= a */
    while (lo < hi) { const int m = (lo + hi + 1) >> 1; if (mxsw_dec_val_e4m3(m) <= a) lo = m; else hi = m - 1; }
    if (lo == 126) return s | 126u;
    const float v0 = mxsw_dec_val_e4m3(lo), v1 = mxsw_dec_val_e4m3(lo + 1);
    const double mid = 0.5 * ((double)v0 + (double)v1);
    const int c = (double)a < mid ? lo : (double)a > mid ? lo + 1 : ((lo & 1) ? lo + 1 : lo);
    return s | (unsigned)c;
}
__device__ unsigned ref_e2m1(float x) {
    const unsigned u = __float_as_uint(x), s = (u >> 28) & 8u;
    if ((u & 0x7fffffffu) > 0x7f800000u) return 7u;
    const float g[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float a = fabsf(x);
    if (a >= 6.f) return s | 7u;
    int lo = 0;
    while (lo < 7 && g[lo + 1] <= a) lo++;
    const float mid = 0.5f * (g[lo] + g[lo + 1]);   /* exact */
    const int c = a < mid ? lo : a > mid ? lo + 1 : ((lo & 1) ? lo + 1 : lo);
    return s | (unsigned)c;
}
__global__ void mxsw_enc_k(mxsw_res *r) {
    const unsigned per = 256, base = (blockIdx.x * blockDim.x + threadIdx.x) * per;
    for (unsigned i = 0; i < per; i++) {
        const unsigned u = base + i;
        const float x = __uint_as_float(u);
        const unsigned s8 = mxsw_enc_e4m3(x), s4 = mxsw_enc_e2m1(x);
        if (s8 != ref_e4m3(x)) mxsw_note(r, 0, u);
        if (s4 != ref_e2m1(x)) mxsw_note(r, 2, u);
#ifdef MXSW_HW
        unsigned short h8, h4;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(h8) : "f"(0.f), "f"(x));
        asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; cvt.u16.u8 %0, t; }" : "=h"(h4) : "f"(0.f), "f"(x));
        if (s8 != (h8 & 0xffu)) mxsw_note(r, 1, u);
        if (s4 != (h4 & 15u)) mxsw_note(r, 3, u);
#endif
    }
}
__global__ void mxsw_dec_k(mxsw_res *r) {
    const unsigned v = blockIdx.x * blockDim.x + threadIdx.x;   /* 65536 */
    unsigned want = mxsw_dec_e4m3x2((unsigned short)v);
#ifdef MXSW_HW
    unsigned hw; asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(hw) : "h"((unsigned short)v));
    if (hw != want) mxsw_note(r, 4, v | 0x80000000u);
    want = hw;
#endif
    if (h2_e4m3x2_sw(v) != want) mxsw_note(r, 4, v);
    if (v < 256) {
        unsigned w2 = mxsw_dec_e2m1x2(v);
#ifdef MXSW_HW
        unsigned h2; asm("{ .reg .b8 t; cvt.u8.u32 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }" : "=r"(h2) : "r"(v));
        if (h2 != w2) mxsw_note(r, 5, v | 0x80000000u);
        w2 = h2;
#endif
        if (h2_e2m1x2_sw(v) != w2) mxsw_note(r, 5, v);
    }
}
__device__ unsigned mxsw_rng(unsigned &s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
/* one warp per test: kind 0 = mma_f8 (k32, 1 scale), 1 = mma_f4 (k64, 2 scales), 2 = mma_f4x4 (k64, 4 scales) */
__global__ void mxsw_mma_k(mxsw_res *r, int kind, unsigned seed) {
    __shared__ unsigned fa[32][4], fb[32][2], fsa[32], fsb[32];
    const int l = threadIdx.x & 31, g = l >> 2, t = l & 3;
    unsigned s = seed ^ (blockIdx.x * 0x9e3779b9u) ^ (l * 0x85ebca6bu); s = s ? s : 1u;
    for (int i = 0; i < 8; i++) mxsw_rng(s);
    unsigned a[4], b[2];
    for (int i = 0; i < 4; i++) { a[i] = mxsw_rng(s); if (kind == 0) a[i] = (a[i] & 0xf7f7f7f7u) | 0x0u; }   /* e4m3: no NaN (x111 1111) */
    for (int i = 0; i < 2; i++) { b[i] = mxsw_rng(s); if (kind == 0) b[i] &= 0xf7f7f7f7u; }
    unsigned sa = 0, sb = 0;
    for (int j = 0; j < 4; j++) { sa |= (112u + mxsw_rng(s) % 32u) << (8 * j); sb |= (112u + mxsw_rng(s) % 32u) << (8 * j); }
    float c[4], c0[4];
    for (int e = 0; e < 4; e++) c[e] = c0[e] = (float)((int)(mxsw_rng(s) % 2001u) - 1000) * 0.01f;
    for (int i = 0; i < 4; i++) fa[l][i] = a[i];
    fb[l][0] = b[0]; fb[l][1] = b[1]; fsa[l] = sa; fsb[l] = sb;
    __syncwarp();
    float cs[4] = {c[0], c[1], c[2], c[3]};
    if (kind == 0) lpsw_mma_f8(cs, a, b, sa, sb); else if (kind == 1) lpsw_mma_f4(cs, a, b, sa, sb); else lpsw_mma_f4x4(cs, a, b, sa, sb);
    /* scalar reference over the layout */
    const int K = kind ? 64 : 32, KB = kind == 2 ? 16 : 32;
    double sums[4], mags[4]; (void)sums;
    for (int e = 0; e < 4; e++) {
        const int row = g + 8 * (e >> 1), col = 2 * t + (e & 1);
        double sum = c0[e], mag = fabs(c0[e]);
        for (int k = 0; k < K; k++) {
            float av, bv;
            if (kind) {
                const int la = 4 * (row & 7) + (k & 31) / 8, ra = (row >= 8) + 2 * (k / 32), lb = 4 * col + (k & 31) / 8, rb = k / 32;
                av = mxsw_dec_val_e2m1((fa[la][ra] >> (4 * (k & 7))) & 15u); bv = mxsw_dec_val_e2m1((fb[lb][rb] >> (4 * (k & 7))) & 15u);
            } else {
                const int la = 4 * (row & 7) + (k & 15) / 4, ra = (row >= 8) + 2 * (k / 16), lb = 4 * col + (k & 15) / 4, rb = k / 16;
                av = mxsw_dec_val_e4m3((fa[la][ra] >> (8 * (k & 3))) & 255u); bv = mxsw_dec_val_e4m3((fb[lb][rb] >> (8 * (k & 3))) & 255u);
            }
            const int blk = k / KB;
            const double sc = ldexp(1.0, (int)((fsa[4 * (row & 7) + (row >= 8)] >> (8 * blk)) & 255u) - 127) * ldexp(1.0, (int)((fsb[4 * col] >> (8 * blk)) & 255u) - 127);
            sum += (double)av * bv * sc; mag += fabs((double)av * bv * sc);
        }
        sums[e] = sum; mags[e] = mag > 0 ? mag : 1.0;
        const double err = fabs((double)cs[e] - sum) / mags[e];
        if (err > 1e-6) mxsw_note(r, 6, (unsigned)(blockIdx.x * 32 + l));
        atomicMax((unsigned *)&r->worst[0], __float_as_uint((float)err));
    }
#ifdef MXSW_HW
    if (kind == 0) mma_f8(c, a, b, sa, sb); else if (kind == 1) mma_f4(c, a, b, sa, sb); else mma_f4x4(c, a, b, sa, sb);
    for (int e = 0; e < 4; e++) {
        const float err = (float)(fabs((double)c[e] - sums[e]) / mags[e]);   /* the instruction vs the exact sum, relative to the sum of magnitudes (its accumulation order is its own) */
        if (err > 1e-5f) mxsw_note(r, 7, (unsigned)(blockIdx.x * 32 + l));
        atomicMax((unsigned *)&r->worst[1], __float_as_uint(err));
    }
#endif
}
/* returns the number of failed checks; hw: whether the instructions were compared (sm_120a build) */
extern "C" int lp_mxsw_hw(void);
extern "C" int lp_mxsw_selftest(int *hw) {
    *hw = lp_mxsw_hw();
    mxsw_res *d, h;
    cudaMalloc(&d, sizeof(mxsw_res)); cudaMemset(d, 0, sizeof(mxsw_res));
    mxsw_enc_k<<<(1u << 24) / 256, 256>>>(d);   /* 2^24 threads x 256 = every fp32 bit pattern */
    mxsw_dec_k<<<256, 256>>>(d);
    static const char *kn[3] = {"mma_f8 (e4m3 k32)", "mma_f4 (e2m1 k64, 2X)", "mma_f4x4 (e2m1 k64, 4X)"};
    static const char *cn[8] = {"e4m3 encode vs table", "e4m3 encode vs cvt", "e2m1 encode vs table", "e2m1 encode vs cvt",
                                "e4m3x2 decode", "e2m1x2 decode", "mma vs layout reference", "mma vs block-scaled instruction"};
    int fails = 0;
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("  launch failed: %s\n", cudaGetErrorString(e)); return 99; }
    cudaMemcpy(&h, d, sizeof h, cudaMemcpyDeviceToHost);
    for (int k = 0; k < 6; k++) {
        printf("  %-40s %s", cn[k], h.n[k] ? "FAIL" : "ok");
        if (h.n[k]) { printf(" (%llu, e.g.", h.n[k]); for (int i = 0; i < 4 && i < (int)h.n[k]; i++) printf(" %08x", h.ex[k][i]); printf(")"); fails++; }
        printf("\n");
    }
    for (int kind = 0; kind < 3; kind++) {
        cudaMemset(d, 0, sizeof(mxsw_res));
        mxsw_mma_k<<<4096, 32>>>(d, kind, 0x1234567u + kind);
        if ((e = cudaDeviceSynchronize()) != cudaSuccess) { printf("  launch failed: %s\n", cudaGetErrorString(e)); return 99; }
        cudaMemcpy(&h, d, sizeof h, cudaMemcpyDeviceToHost);
        for (int k = 6; k < 8; k++) {
            if (k == 7 && !*hw) continue;
            printf("  %-24s %-32s %s (worst rel %.2g)", kn[kind], cn[k], h.n[k] ? "FAIL" : "ok", h.worst[k - 6]);
            if (h.n[k]) { printf(" (%llu lanes)", h.n[k]); fails++; }
            printf("\n");
        }
    }
    cudaFree(d);
    return fails;
}
/* whether this build's lp kernels have the hardware instructions (sm_120a) or the software paths */
__global__ void mxsw_hw_k(int *o) {
#ifdef MXSW_HW
    *o = 1;
#else
    *o = 0;
#endif
}
extern "C" int lp_mxsw_hw(void) {
    int *d, h = -1;
    cudaMalloc(&d, 4); mxsw_hw_k<<<1, 1>>>(d); cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost); cudaFree(d);
    return h;
}
