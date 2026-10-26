#pragma once
/* Software MX element conversions for GPUs without the fp8 / fp4 conversion instructions (Ampere sm_86: no
   cvt.*.e4m3x2 below sm_89, no cvt.*.e2m1x2 below sm_100a). Each is bit-exact against the instruction it replaces on
   every fp32 input (tests/test_mx_sw.c runs all 2^32 against the sm_120a instructions and an independent table search):
     mxsw_enc_e4m3  = cvt.rn.satfinite.e4m3x2.f32 (one lane): round to nearest even, |x| >= 448 (incl. inf) -> +-448,
                      NaN -> 0x7f (sign kept); fp32 subnormals are below half the e4m3 step 2^-9 and give +-0.
     mxsw_enc_e2m1  = cvt.rn.satfinite.e2m1x2.f32 (one lane, the nibble): grid {0, .5, 1, 1.5, 2, 3, 4, 6}, ties to the even
                      code, |x| > 6 (incl. inf) -> +-6, NaN -> +6 (0x7: the instruction's NaN result).
     mxsw_dec_e4m3x2 / mxsw_dec_e2m1x2 = cvt.rn.f16x2.e4m3x2 / cvt.rn.f16x2.e2m1x2 (both exact in f16).
   Integer / bit operations only (plus one exact float scale per e4m3 lane), so they also compile under -use_fast_math with
   the same results. */
#include <cuda_fp16.h>

__host__ __device__ __forceinline__ unsigned mxsw_enc_e4m3(float x) {
    unsigned u;
#ifdef __CUDA_ARCH__
    u = __float_as_uint(x);
#else
    __builtin_memcpy(&u, &x, 4);
#endif
    const unsigned s = (u >> 24) & 0x80u, a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return s | 0x7fu;                      /* NaN */
    if (a >= 0x43e00000u) return s | 0x7eu;                     /* >= 448 (ties at 464 round to 448, larger saturate) */
    if (a >= 0x3c800000u) {                                     /* normal e4m3 (>= 2^-6): RNE on the 20 dropped mantissa bits */
        const unsigned r = (a + 0x7ffffu + ((a >> 20) & 1u)) >> 20;
        return s | (r - (120u << 3));                           /* exponent carry is the next code; < 464 never reaches 0x7f */
    }
    /* subnormal e4m3: code = round-to-even(|x| / 2^-9) in 0..8 (8 = 2^-6, the first normal code), from the integer bits:
       |x| = (1.m) 2^(e-127) with e <= 120, so |x| 2^9 = 1.m 2^(e-118) and the code is the mantissa shifted by 141 - e */
    const int e = (int)(a >> 23);
    if (e < 116) return s;                                      /* < 2^-11 < half the step (incl. fp32 subnormals and 0) */
    const unsigned m = (a & 0x7fffffu) | 0x800000u;             /* 24-bit significand */
    const int sh = 141 - e;                                     /* 21..25 */
    const unsigned q = m >> sh, rem = m & ((1u << sh) - 1u), half = 1u << (sh - 1);
    return s | (q + (rem > half || (rem == half && (q & 1u))));
}
__host__ __device__ __forceinline__ unsigned mxsw_enc_e2m1(float x) {
    unsigned u;
#ifdef __CUDA_ARCH__
    u = __float_as_uint(x);
#else
    __builtin_memcpy(&u, &x, 4);
#endif
    const unsigned s = (u >> 28) & 8u, a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return 7u;                             /* NaN -> +6 */
    if (a > 0x40a00000u) return s | 7u;                         /* > 5 (5 = tie of 4 and 6 -> 4): 6, saturating */
    if (a >= 0x3f800000u) {                                     /* [1, 5]: one mantissa bit, RNE on the 22 dropped bits */
        const unsigned r = (a + 0x1fffffu + ((a >> 22) & 1u)) >> 22;
        return s | (r - 252u);                                  /* 1 -> 2, 1.5 -> 3, 2 -> 4, 3 -> 5, 4 -> 6 */
    }
    /* [0, 1): codes 0 (0), 1 (0.5), 2 (1): ties 0.25 -> 0, 0.75 -> 2 */
    if (a <= 0x3e800000u) return s;
    if (a < 0x3f400000u) return s | 1u;
    return s | 2u;
}
/* e4m3 byte -> f16 bits: the e4m3 field placed at f16 bit 7 is the value times 2^-8 in f16 (exponent biases 7 vs 15, and the
   subnormals line up), so one exact multiply by 256 finishes it; 0x7f / 0xff (NaN) -> f16 NaN */
__host__ __device__ __forceinline__ unsigned short mxsw_e4m3_h16(unsigned b) {
    b &= 0xffu;
    if ((b & 0x7fu) == 0x7fu) return (unsigned short)(((b & 0x80u) << 8) | 0x7fffu);
    const unsigned em = b & 0x7fu, e = em >> 3, m = em & 7u;
    unsigned short h;
    if (e) h = (unsigned short)(((e + 8u) << 10) | (m << 7));    /* normal: f16 exponent e - 7 + 15 */
    else if (!m) h = 0;
    else {                                                       /* subnormal e4m3 m 2^-9 = normal f16 (m 2^-9 >= 2^-9) */
        const int lz = m >= 4 ? 0 : m >= 2 ? 1 : 2;             /* leading bit at 2 - lz */
        const unsigned ee = (unsigned)(15 - 9 + 2 - lz);        /* exponent of m 2^-9 */
        h = (unsigned short)((ee << 10) | (((m << (lz + 1)) & 7u) << 7));
    }
    return (unsigned short)(((b & 0x80u) << 8) | h);
}
__host__ __device__ __forceinline__ unsigned short mxsw_e2m1_h16(unsigned n) {
    /* {0, .5, 1, 1.5, 2, 3, 4, 6} in f16 bits */
    const unsigned short t[8] = {0x0000, 0x3800, 0x3c00, 0x3e00, 0x4000, 0x4200, 0x4400, 0x4600};
    return (unsigned short)(t[n & 7u] | ((n & 8u) << 12));
}
/* element values (fp32, exact): e4m3 code (NaN codes -> NaN), e2m1 nibble */
__host__ __device__ __forceinline__ float mxsw_dec_val_e4m3(unsigned b) {
    const unsigned e = (b >> 3) & 15u, m = b & 7u;
    if ((b & 0x7fu) == 0x7fu) { const unsigned q = 0x7fc00000u | ((b & 0x80u) << 24); float f; __builtin_memcpy(&f, &q, 4); return f; }
    const float v = e ? (float)(8u + m) * (float)(1u << e) * (1.f / 1024.f) : (float)m * 0.001953125f;
    return (b & 0x80u) ? -v : v;
}
__host__ __device__ __forceinline__ float mxsw_dec_val_e2m1(unsigned n) {
    const float v = (float)((n & 7u) < 4u ? (n & 7u) : (n & 7u) == 4u ? 4u : (n & 7u) == 5u ? 6u : (n & 7u) == 6u ? 8u : 12u) * 0.5f;
    return (n & 8u) ? -v : v;
}
#ifdef __CUDACC__
__device__ __forceinline__ unsigned mxsw_dec_e4m3x2(unsigned short v) {   /* = cvt.rn.f16x2.e4m3x2: low byte -> low half */
    return (unsigned)mxsw_e4m3_h16(v & 0xffu) | ((unsigned)mxsw_e4m3_h16(v >> 8) << 16);
}
__device__ __forceinline__ unsigned mxsw_dec_e2m1x2(unsigned b) {          /* = cvt.rn.f16x2.e2m1x2: low nibble -> low half */
    return (unsigned)mxsw_e2m1_h16(b & 15u) | ((unsigned)mxsw_e2m1_h16((b >> 4) & 15u) << 16);
}
#endif
