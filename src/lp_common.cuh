#pragma once
/* Shared declarations and device helpers of the low-precision (FP8 / FP4 MX) kernel families, split out of the
   former nn_fp8.cu so the families compile as separate translation units (in parallel). */
#include <type_traits>
/* FP8 (e4m3) / FP4 (e2m1) tensor-core 3^3 stride-1 convolutions with MX block scaling, fp32 accumulate.
   Built for sm_120a: `mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X` runs at
   4x the BF16 rate on GeForce Blackwell with fp32 accumulation (plain `.e4m3.e4m3.f32` without block scale runs
   at only 2x), and `m16n8k64 .kind::mxf4` at 8x. Every 32 K-elements of an A row / B column carry their own
   power-of-two (ue8m0) scale, so no global amax pass or delayed scaling is needed: scales are computed locally
   while the tiles are staged.

   Scale-register layout (determined empirically on the RTX 5060 Ti, byte-id 0 / thread-id 0):
     A row g  : lane 4g,   byte 0        A row g+8: lane 4g+1, byte 0
     B col g  : lane 4g,   byte 0        (for 2X / 4X the bytes 0..1 / 0..3 are the consecutive K blocks)

   Forward (also backward-data via flipped weights): Y[co][v] = sum_tap sum_ci W[tap][co][ci] X[ci][v+off(tap)].
     A = weights, one scale per (tap, co, 32-ci chunk) from a prep kernel.
     B = staged input tile sx[pos][32 ci] (fp8), one scale per (position, 32-ci chunk): the scale belongs to the
         voxel, so it moves with the shifted view of every tap.
   Weight gradient: GW[co][ci][tap] = sum_v GY[co][v] X[ci][v+off(tap)], K = voxels.
     A = GY, one scale per (co, 32 voxels = two output rows) (warp-local amax).
     B = X, one scale per (ci, staged z-tile): a K block crosses rows, so a per-row scale would not survive the
         y/z tap shifts; the per-channel tile scale is invariant under every shift. */
#include "nn.h"
#include "nn_lp.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

extern cudaError_t g_lp_err;
#define LPCK() do { cudaError_t e_ = cudaGetLastError(); if (e_ != cudaSuccess && g_lp_err == cudaSuccess) g_lp_err = e_; } while (0)
static inline unsigned nblk_(size_t n, unsigned b) { return (unsigned)((n + b - 1) / b); }
static inline int cur_dev_(void) { int d = 0; cudaGetDevice(&d); return d & 7; }

typedef __nv_bfloat16 bf16;
/* activation element type T = float or bf16 (inputs and outputs of the forward; gradients are float) */
struct mx8_t { unsigned char v; };   /* storage marker: MX-fp8 channel-blocked tensor (see below) */
struct mx4_t { unsigned char v; };   /* storage marker: MX-fp4 (packed e2m1 nibbles, ue8m0 per block row) tensor */
template <typename T> __device__ __forceinline__ float ldx(const T *p, size_t i);
template <> __device__ __forceinline__ float ldx<float>(const float *p, size_t i) { return __ldg(p + i); }
template <> __device__ __forceinline__ float ldx<bf16>(const bf16 *p, size_t i) { return __bfloat162float(__ldg(p + i)); }
template <> __device__ __forceinline__ float ldx<__half>(const __half *p, size_t i) { return __half2float(__ldg(p + i)); }
template <typename T> __device__ __forceinline__ void stx2(T *p, size_t i, float a, float b);
template <> __device__ __forceinline__ void stx2<float>(float *p, size_t i, float a, float b) { *(float2 *)(p + i) = make_float2(a, b); }
template <> __device__ __forceinline__ void stx2<bf16>(bf16 *p, size_t i, float a, float b) { *(__nv_bfloat162 *)(p + i) = __floats2bfloat162_rn(a, b); }
template <> __device__ __forceinline__ void stx2<__half>(__half *p, size_t i, float a, float b) { *(__half2 *)(p + i) = __floats2half2_rn(a, b); }
template <typename T> __device__ __forceinline__ void stx(T *p, size_t i, float v);
template <> __device__ __forceinline__ void stx<float>(float *p, size_t i, float v) { p[i] = v; }
template <> __device__ __forceinline__ void stx<bf16>(bf16 *p, size_t i, float v) { p[i] = __float2bfloat16(v); }
template <> __device__ __forceinline__ void stx<__half>(__half *p, size_t i, float v) { p[i] = __float2half(v); }
/* four consecutive elements starting at a 4-aligned index */
template <typename T> __device__ __forceinline__ void ld8x(const T *p, float *o);   /* 8 elements, 16 B (16-bit) / 32 B (float) aligned */
template <> __device__ __forceinline__ void ld8x<float>(const float *p, float *o) { float4 a = __ldg((const float4 *)p), b = __ldg((const float4 *)p + 1); o[0] = a.x; o[1] = a.y; o[2] = a.z; o[3] = a.w; o[4] = b.x; o[5] = b.y; o[6] = b.z; o[7] = b.w; }
template <> __device__ __forceinline__ void ld8x<bf16>(const bf16 *p, float *o) {
    uint4 u = __ldg((const uint4 *)p);
    o[0] = __uint_as_float(u.x << 16); o[1] = __uint_as_float(u.x & 0xffff0000u); o[2] = __uint_as_float(u.y << 16); o[3] = __uint_as_float(u.y & 0xffff0000u);
    o[4] = __uint_as_float(u.z << 16); o[5] = __uint_as_float(u.z & 0xffff0000u); o[6] = __uint_as_float(u.w << 16); o[7] = __uint_as_float(u.w & 0xffff0000u);
}
template <> __device__ __forceinline__ void ld8x<__half>(const __half *p, float *o) {
    uint4 u = __ldg((const uint4 *)p);
    const __half2 *h = (const __half2 *)&u;
#pragma unroll
    for (int j = 0; j < 4; j++) { float2 f = __half22float2(h[j]); o[2 * j] = f.x; o[2 * j + 1] = f.y; }
}
template <typename T> __device__ __forceinline__ float4 ldx4(const T *p);
template <> __device__ __forceinline__ float4 ldx4<float>(const float *p) { return __ldg((const float4 *)p); }
template <> __device__ __forceinline__ float4 ldx4<__half>(const __half *p) {
    uint2 u = __ldg((const uint2 *)p);
    float2 a = __half22float2(*(const __half2 *)&u.x), b = __half22float2(*(const __half2 *)&u.y);
    return make_float4(a.x, a.y, b.x, b.y);
}
template <> __device__ __forceinline__ float4 ldx4<bf16>(const bf16 *p) {
    uint2 u = __ldg((const uint2 *)p);
    return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u), __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xffff0000u));
}
/* per-channel staging descriptor: source plane pointer (nullptr = zero channel) and the GN+SiLU affine a*x + b */
/* ---- MX-fp8 activation storage (lp dtype 3): a tensor of C channels is stored channel-blocked, data[n][blk][voxel][bw] e4m3
   bytes followed by the scale plane sc[n][blk][voxel] (ue8m0), bw = 16 for C <= 16 else 32 (padding channels hold zeros).
   One 32-channel staging chunk of the fp8 kernels is then one stored block: a straight copy of the bytes and the scale. */
/* MX-fp4 activation storage (lp dtype 4): the same channel blocking with two e2m1 nibbles per byte, data[n][blk][voxel][bw/2]
   (channel k of a row = nibble k & 1 of byte k / 2, low nibble first) + the same ue8m0 scale plane; row = 16 B (bw 32) or 8 B
   (bw 16) = one staged row of the fp4 kernel. A NaN block amax is stored as scale byte 0xFF (decoded as inf), so a non-finite
   value never disappears into a finite nibble (0 * inf = NaN, else +-inf). */
/* plane-major accessors are never used on MX tensors (the kernels branch on IS_MX8 / IS_MX4); stubs keep the instantiations complete */
template <> __device__ __forceinline__ float ldx<mx8_t>(const mx8_t *, size_t) { return 0.f; }
template <> __device__ __forceinline__ void stx<mx8_t>(mx8_t *, size_t, float) {}
template <> __device__ __forceinline__ void stx2<mx8_t>(mx8_t *, size_t, float, float) {}
template <> __device__ __forceinline__ float4 ldx4<mx8_t>(const mx8_t *) { return make_float4(0.f, 0.f, 0.f, 0.f); }
template <> __device__ __forceinline__ void ld8x<mx8_t>(const mx8_t *, float *o) { for (int i = 0; i < 8; i++) o[i] = 0.f; }
template <> __device__ __forceinline__ float ldx<mx4_t>(const mx4_t *, size_t) { return 0.f; }
template <> __device__ __forceinline__ void stx<mx4_t>(mx4_t *, size_t, float) {}
template <> __device__ __forceinline__ void stx2<mx4_t>(mx4_t *, size_t, float, float) {}
template <> __device__ __forceinline__ float4 ldx4<mx4_t>(const mx4_t *) { return make_float4(0.f, 0.f, 0.f, 0.f); }
template <> __device__ __forceinline__ void ld8x<mx4_t>(const mx4_t *, float *o) { for (int i = 0; i < 8; i++) o[i] = 0.f; }
template <typename T> struct is_mx8_s { static constexpr bool v = false; };
template <> struct is_mx8_s<mx8_t> { static constexpr bool v = true; };
template <typename T> struct is_mx4_s { static constexpr bool v = false; };
template <> struct is_mx4_s<mx4_t> { static constexpr bool v = true; };
#define IS_MX8(T) (is_mx8_s<T>::v)
#define IS_MX4(T) (is_mx4_s<T>::v)
#define IS_MX(T) (is_mx8_s<T>::v || is_mx4_s<T>::v)
template <typename T> struct mx_bits_s { static constexpr int v = 0; };   /* element bits of an MX storage type (0: plane-major) */
template <> struct mx_bits_s<mx8_t> { static constexpr int v = 8; };
template <> struct mx_bits_s<mx4_t> { static constexpr int v = 4; };
#define MX_BITS(T) (mx_bits_s<T>::v)
template <int B> struct mx_type_s;
template <> struct mx_type_s<8> { typedef mx8_t t; };
template <> struct mx_type_s<4> { typedef mx4_t t; };
__host__ __device__ __forceinline__ int mx_bw(int C) { return C <= 8 ? 8 : C <= 16 ? 16 : 32; }   /* 8: the network input (4 channels; rows of 4 / 8 B stay word-aligned) */
__host__ __device__ __forceinline__ int mx_nb(int C) { int bw = mx_bw(C); return (C + bw - 1) / bw; }
__host__ __device__ __forceinline__ int mx_rb(int bw, int bits) { return bw * bits / 8; }   /* row bytes */
__device__ __forceinline__ float2 dec_e4m3x2(unsigned short v) {   /* low byte -> .x */
    unsigned r; asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(v));
    return __half22float2(*(__half2 *)&r);
}
__device__ __forceinline__ float dec_e4m3(unsigned b) { return dec_e4m3x2((unsigned short)(b & 0xff)).x; }
__device__ __forceinline__ float2 dec_e2m1x2(unsigned b) {   /* low nibble -> .x */
    unsigned r; asm("{ .reg .b8 t; cvt.u8.u32 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }" : "=r"(r) : "r"(b & 0xff));
    return __half22float2(*(__half2 *)&r);
}
__device__ __forceinline__ float dec_e2m1n(unsigned n) { return dec_e2m1x2(n & 15).x; }   /* one nibble */
__device__ __forceinline__ float mx_scale(unsigned sbyte) { return __uint_as_float(sbyte << 23); }   /* 0xFF -> inf */
/* ---- format trait: row decode / encode of one block row of bw channels (bw 16: v[16..31] = 0 on decode, ignored on encode) ---- */
template <int B> struct mxf;
template <> struct mxf<8> {
    static constexpr float inv_qmax = 1.f / 448.f;
    static __device__ __forceinline__ void dec_row(const uint8_t *p, int bw, float s, float *v) {
        const uint4 *src = (const uint4 *)p;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            uint4 u = make_uint4(0u, 0u, 0u, 0u);
            if (bw == 8) { if (!h) { const uint2 q = __ldg((const uint2 *)p); u.x = q.x; u.y = q.y; } }
            else if (h == 0 || bw == 32) u = __ldg(src + h);
            const unsigned *w = (const unsigned *)&u;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float2 a = dec_e4m3x2((unsigned short)(w[j] & 0xffff)), b = dec_e4m3x2((unsigned short)(w[j] >> 16));
                v[16 * h + 4 * j] = a.x * s; v[16 * h + 4 * j + 1] = a.y * s; v[16 * h + 4 * j + 2] = b.x * s; v[16 * h + 4 * j + 3] = b.y * s;
            }
        }
    }
    static __device__ __forceinline__ unsigned enc8x(const float *v, float m);   /* defined after cvt_e4m3x4 */
    static __device__ __forceinline__ void enc_row(uint8_t *p, int bw, const float *v, float m) {
        uint4 *dst = (uint4 *)p;
        if (bw == 8) { *(uint2 *)p = make_uint2(enc8x(v, m), enc8x(v + 4, m)); return; }
        dst[0] = make_uint4(enc8x(v, m), enc8x(v + 4, m), enc8x(v + 8, m), enc8x(v + 12, m));
        if (bw == 32) dst[1] = make_uint4(enc8x(v + 16, m), enc8x(v + 20, m), enc8x(v + 24, m), enc8x(v + 28, m));
    }
    static __device__ __forceinline__ float dec1(const uint8_t *p, int c) { return dec_e4m3(p[c]); }
};
template <> struct mxf<4> {
    static constexpr float inv_qmax = 1.f / 6.f;
    static __device__ __forceinline__ void dec_row(const uint8_t *p, int bw, float s, float *v) {
        uint2 lo = bw == 8 ? make_uint2(__ldg((const unsigned *)p), 0u) : __ldg((const uint2 *)p), hi = bw == 32 ? __ldg((const uint2 *)p + 1) : make_uint2(0u, 0u);
        const unsigned w[4] = {lo.x, lo.y, hi.x, hi.y};
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int q = 0; q < 4; q++) { float2 a = dec_e2m1x2(w[j] >> (8 * q)); v[8 * j + 2 * q] = a.x * s; v[8 * j + 2 * q + 1] = a.y * s; }
    }
    static __device__ __forceinline__ unsigned enc8x(const float *v, float m);   /* defined after cvt_e2m1x8 */
    static __device__ __forceinline__ void enc_row(uint8_t *p, int bw, const float *v, float m) {
        if (bw == 8) { *(unsigned *)p = enc8x(v, m); return; }
        *(uint2 *)p = make_uint2(enc8x(v, m), enc8x(v + 8, m));
        if (bw == 32) *((uint2 *)p + 1) = make_uint2(enc8x(v + 16, m), enc8x(v + 24, m));
    }
    static __device__ __forceinline__ float dec1(const uint8_t *p, int c) { return dec_e2m1n((unsigned)p[c >> 1] >> (4 * (c & 1))); }
};
/* padded input-channel index -> real channel (-1 = zero padding). With MX-stored inputs the x and x2 segments are padded
   to 32 channels each (CxP = pad32(Cx)) so that a 32-channel chunk never straddles two tensors or two stored blocks. */
__host__ __device__ __forceinline__ int seg_ci(int cp, int Ci, int Cx, int CxP) { if (cp < CxP) return cp < Cx ? cp : -1; int c = Cx + cp - CxP; return c < Ci ? c : -1; }
/* per-channel staging descriptor: source plane pointer (nullptr = zero channel) and the GN+SiLU affine a*x + b;
   for MX tensors p points at this channel's byte in block row 0 (element at voxel v: p[v * bw]) and sp at the block's scale row */
/* MX: p = the channel's byte in block row 0 (fp8) or the row's first byte (fp4, nibble index in nib), bw = block width,
   rb = row bytes (element at voxel v: row p + v * rb), sp = the block's scale row */
typedef struct { const void *p; const uint8_t *sp; float a, b; short bw, rb, g, nib; } chan_t;   /* g: gn+silu applies; 32 B */
template <typename T>
__device__ __forceinline__ chan_t make_chan(int ci, int Ci, int Cx, int n, size_t plane, const T *x, const split_t &sp, const gnp_t &gp, int N = 0) {
    chan_t c; c.p = nullptr; c.sp = nullptr; c.a = 1.f; c.b = 0.f; c.bw = 1; c.g = 0; c.rb = 1; c.nib = 0;
    if (ci >= 0 && ci < Ci) {
        if constexpr (IS_MX(T)) {
            const bool sec = ci >= Cx;
            const uint8_t *base = sec ? (const uint8_t *)sp.x2 : (const uint8_t *)x;
            const int C = sec ? Ci - Cx : Cx, cc = sec ? ci - Cx : ci, bw = mx_bw(C), nb = mx_nb(C), rb = mx_rb(bw, MX_BITS(T));
            c.p = base + (((size_t)n * nb + cc / bw) * plane) * rb + (IS_MX4(T) ? 0 : cc % bw);
            c.sp = base + (size_t)N * nb * plane * rb + ((size_t)n * nb + cc / bw) * plane;
            c.bw = (short)bw; c.rb = (short)rb; c.nib = (short)(cc % bw);
        } else c.p = ci >= Cx ? (const T *)sp.x2 + ((size_t)n * (Ci - Cx) + ci - Cx) * plane : x + ((size_t)n * Cx + ci) * plane;
        const bool s2g = sp.x2 && sp.gp2.G;   /* per-segment GroupNorms */
        const gnp_t &q = s2g && ci >= Cx ? sp.gp2 : gp;
        const int cc = s2g && ci >= Cx ? ci - Cx : ci, Cg = s2g ? (ci >= Cx ? Ci - Cx : Cx) : Ci;
        if (q.G) { int ng = n * q.G + cc / (Cg / q.G); float a = q.rstd[ng] * q.gamma[cc]; c.a = a; c.b = q.beta[cc] - q.mean[ng] * a; c.g = 1; }
    }
    return c;
}
/* voxel offset of (z, y, x) */
__device__ __forceinline__ size_t cof(const chan_t &, int z, int y, int x, int H, int W) { return ((size_t)z * H + y) * W + x; }
/* element (n, c, v) of an MX tensor (B bits) with N samples, C channels, S voxels */
template <int B> __device__ __forceinline__ float ldmx_e(const void *qv, int N, int C, size_t S, int n, int c, size_t v) {
    const uint8_t *q = (const uint8_t *)qv;
    const int bw = mx_bw(C), nb = mx_nb(C), rb = mx_rb(bw, B);
    const size_t ri = ((size_t)n * nb + c / bw) * S + v;
    return mxf<B>::dec1(q + ri * rb, c % bw) * mx_scale(q[(size_t)N * nb * S * rb + ri]);
}
__device__ __forceinline__ float ldmx4_e(const void *q, int N, int C, size_t S, int n, int c, size_t v) { return ldmx_e<4>(q, N, C, S, n, c, v); }
/* element of a channel at voxel offset off: plane-major types, or MX (decode * 2^(scale - 127)) */
template <typename T> __device__ __forceinline__ float ldc(const chan_t &c, size_t off) {
    if constexpr (IS_MX(T)) return mxf<MX_BITS(T)>::dec1((const uint8_t *)c.p + off * c.rb, IS_MX4(T) ? c.nib : 0) * mx_scale(c.sp[off]);
    else return ldx((const T *)c.p, off);
}
/* 4 consecutive voxels (off .. off + nv - 1, nv <= 4) of one channel of an MX tensor: the index math once, the 4 scale bytes in one
   32-bit load when al4 (W % 4 == 0 and off % 4 == 0), element pairs decoded with one cvt */
template <typename T> __device__ __forceinline__ float4 ldc4_mx(const chan_t &c, size_t off, int nv, bool al4) {
    const uint8_t *p = (const uint8_t *)c.p + off * c.rb;
    unsigned sw;
    if (al4 && nv == 4) sw = __ldg((const unsigned *)(c.sp + off));
    else { sw = 0u; for (int k = 0; k < nv; k++) sw |= (unsigned)c.sp[off + k] << (8 * k); }
    unsigned b[4];
#pragma unroll
    for (int k = 0; k < 4; k++) {
        unsigned v = k < nv ? (unsigned)__ldg(p + (size_t)k * c.rb + (IS_MX4(T) ? (c.nib >> 1) : 0)) : 0u;
        b[k] = IS_MX4(T) ? (v >> (4 * (c.nib & 1))) & 15u : v;
    }
    float2 lo, hi;
    if constexpr (IS_MX4(T)) { lo = dec_e2m1x2(b[0] | b[1] << 4); hi = dec_e2m1x2(b[2] | b[3] << 4); }
    else { lo = dec_e4m3x2((unsigned short)(b[0] | b[1] << 8)); hi = dec_e4m3x2((unsigned short)(b[2] | b[3] << 8)); }
    return make_float4(nv > 0 ? lo.x * mx_scale(sw & 255u) : 0.f, nv > 1 ? lo.y * mx_scale(sw >> 8 & 255u) : 0.f,
                       nv > 2 ? hi.x * mx_scale(sw >> 16 & 255u) : 0.f, nv > 3 ? hi.y * mx_scale(sw >> 24) : 0.f);
}
/* the 32 channels of one stored MX block row at voxel off (bw = 16: upper half zero), dequantized; ok == false: zeros */
template <int B> __device__ __forceinline__ void mx_row32(const chan_t &c0, size_t off, bool ok, float *v) {
    if (ok && c0.p) mxf<B>::dec_row((const uint8_t *)c0.p + off * c0.rb, c0.bw, mx_scale(c0.sp[off]), v);
    else {
#pragma unroll
        for (int k = 0; k < 32; k++) v[k] = 0.f;
    }
}
template <> __device__ __forceinline__ void mx_row32<8>(const chan_t &c0, size_t off, bool ok, float *v) {   /* branch-free (the fp8 staging is issue-bound) */
    uint4 h[2] = {make_uint4(0u, 0u, 0u, 0u), make_uint4(0u, 0u, 0u, 0u)};
    float s = 0.f;
    if (ok && c0.p) {
        const uint4 *src = (const uint4 *)((const uint8_t *)c0.p + off * c0.rb);
        if (c0.bw == 8) { const uint2 q = __ldg((const uint2 *)src); h[0].x = q.x; h[0].y = q.y; } else { h[0] = __ldg(src); if (c0.bw == 32) h[1] = __ldg(src + 1); }
        s = mx_scale(c0.sp[off]);
    }
#pragma unroll
    for (int q = 0; q < 2; q++) {
        const unsigned *w = (const unsigned *)&h[q];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 a = dec_e4m3x2((unsigned short)(w[j] & 0xffff)), b = dec_e4m3x2((unsigned short)(w[j] >> 16));
            v[16 * q + 4 * j] = a.x * s; v[16 * q + 4 * j + 1] = a.y * s; v[16 * q + 4 * j + 2] = b.x * s; v[16 * q + 4 * j + 3] = b.y * s;
        }
    }
}
/* 8 consecutive voxels vo .. vo + 7 (those < nv) of channel co of an MX-fp8 tensor (N, C channels, S voxels): index math once,
   the scale bytes in one 8 / 4-byte load when al (vo and S multiples of 8 for nv >= 8, of 4 for nv >= 4) */
__device__ __forceinline__ void ldmx8_8(const void *gyv, int N, int C, size_t S, int n, int co, size_t vo, int nv, bool al8, float *q) {
    const int bw = mx_bw(C), nbk = mx_nb(C);
    const size_t ri = ((size_t)n * nbk + co / bw) * S + vo;
    const uint8_t *qd = (const uint8_t *)gyv + ri * bw + co % bw, *qs = (const uint8_t *)gyv + (size_t)N * nbk * S * bw + ri;
    uint2 sw = make_uint2(0u, 0u);
    if (al8 && nv >= 8) sw = __ldg((const uint2 *)qs);
    else if (al8 && nv >= 4) { sw.x = __ldg((const unsigned *)qs); for (int j = 4; j < 8; j++) if (j < nv) sw.y |= (unsigned)qs[j] << (8 * (j & 3)); }
    else { for (int j = 0; j < 8; j++) if (j < nv) (j < 4 ? sw.x : sw.y) |= (unsigned)qs[j] << (8 * (j & 3)); }
#pragma unroll
    for (int j = 0; j < 8; j += 2) {
        const unsigned b0 = j < nv ? __ldg(qd + (size_t)j * bw) : 0u, b1 = j + 1 < nv ? __ldg(qd + (size_t)(j + 1) * bw) : 0u;
        const float2 d = dec_e4m3x2((unsigned short)(b0 | b1 << 8));
        const unsigned s4 = j < 4 ? sw.x : sw.y;
        q[j] = j < nv ? d.x * mx_scale(s4 >> (8 * (j & 3)) & 255u) : 0.f; q[j + 1] = j + 1 < nv ? d.y * mx_scale(s4 >> (8 * ((j + 1) & 3)) & 255u) : 0.f;
    }
}
/* v[q] += w * (entry nib + q of the stored block row of voxel off), q < CH: one row-word load per 8 (mx4) / 4 (mx8) channels */
template <typename T, int CH> __device__ __forceinline__ void mx_rowch_add(const chan_t &c0, size_t off, float w, float *v) {
    const float sc = mx_scale(c0.sp[off]);
    if constexpr (IS_MX4(T)) {
        const unsigned *rw = (const unsigned *)((const uint8_t *)c0.p + off * c0.rb + (c0.nib >> 1));
#pragma unroll
        for (int i = 0; i < CH / 8; i++) {
            const unsigned u = __ldg(rw + i);
#pragma unroll
            for (int q = 0; q < 4; q++) { const float2 d = dec_e2m1x2(u >> (8 * q)); v[8 * i + 2 * q] = fmaf(w, d.x * sc, v[8 * i + 2 * q]); v[8 * i + 2 * q + 1] = fmaf(w, d.y * sc, v[8 * i + 2 * q + 1]); }
        }
    } else {
        const unsigned *rw = (const unsigned *)((const uint8_t *)c0.p + off * c0.rb);   /* mx8: c.p already points at entry nib */
#pragma unroll
        for (int i = 0; i < CH / 4; i++) {
            const unsigned u = __ldg(rw + i);
            const float2 d0 = dec_e4m3x2((unsigned short)(u & 0xffffu)), d1 = dec_e4m3x2((unsigned short)(u >> 16));
            v[4 * i] = fmaf(w, d0.x * sc, v[4 * i]); v[4 * i + 1] = fmaf(w, d0.y * sc, v[4 * i + 1]);
            v[4 * i + 2] = fmaf(w, d1.x * sc, v[4 * i + 2]); v[4 * i + 3] = fmaf(w, d1.y * sc, v[4 * i + 3]);
        }
    }
}
/* the CH channels at fine voxel (gz, gy, gx) (dims D, H, W): stored directly, or (up) the exact 2x trilinear upsample of the
   half-resolution tensor c0 describes (align_corners = false, edge clamp; the weights and order of stage_up32) */
template <typename T, int CH> __device__ __forceinline__ void mx_rowch(const chan_t &c0, bool up, int gz, int gy, int gx, int D, int H, int W, float *v) {
#pragma unroll
    for (int q = 0; q < CH; q++) v[q] = 0.f;
    if (!up) { mx_rowch_add<T, CH>(c0, ((size_t)gz * H + gy) * W + gx, 1.f, v); return; }
    const int Dc = D >> 1, Hc = H >> 1, Wc = W >> 1;
    const int mz[2] = {gz >> 1, min(max((gz >> 1) + ((gz & 1) ? 1 : -1), 0), Dc - 1)}, my[2] = {gy >> 1, min(max((gy >> 1) + ((gy & 1) ? 1 : -1), 0), Hc - 1)},
              mx[2] = {gx >> 1, min(max((gx >> 1) + ((gx & 1) ? 1 : -1), 0), Wc - 1)};
#pragma unroll
    for (int a = 0; a < 2; a++)
#pragma unroll
        for (int b = 0; b < 2; b++)
#pragma unroll
            for (int c = 0; c < 2; c++)
                mx_rowch_add<T, CH>(c0, ((size_t)mz[a] * Hc + my[b]) * Wc + mx[c], (a ? 0.25f : 0.75f) * (b ? 0.25f : 0.75f) * (c ? 0.25f : 0.75f), v);
}
__device__ __forceinline__ void mx4_row32(const chan_t &c0, size_t off, bool ok, float *v) { mx_row32<4>(c0, off, ok, v); }
__device__ __forceinline__ float act_ab(float v, float a, float b, bool G);
/* the 32 staged values of one position (voxel offset off, inb = inside the tensor) for the 32-channel chunk described by
   ctab: dequantised (MX: one block row read), then the per-channel GN+SiLU affine where it applies (G); zero padding
   channels / outside positions. Shared by every stride-1 forward kernel (fp8 / fp4 staging). */
template <typename T>
__device__ __forceinline__ void stage_row32(const chan_t *ctab, size_t off, bool inb, bool G, float *v, const float2 *cab = nullptr, unsigned gmask = 0u) {
    if constexpr (IS_MX(T)) {
        mx_row32<MX_BITS(T)>(ctab[0], off, inb, v);   /* zeros outside, in padding channels and for a missing block */
        if (G && cab) {   /* compact coefficients: channels without a GroupNorm (mask bit 0) keep their (zero or raw) value */
#pragma unroll
            for (int k = 0; k < 32; k++) if ((gmask >> k) & 1u) { const float2 ab = cab[k]; v[k] = inb ? act_ab(v[k], ab.x, ab.y, true) : 0.f; }
        } else if (G) {
#pragma unroll
            for (int k = 0; k < 32; k++) { const chan_t &c = ctab[k]; v[k] = inb && c.p ? act_ab(v[k], c.a, c.b, c.g) : 0.f; }
        }
    } else {
#pragma unroll
        for (int k = 0; k < 32; k++) {
            const chan_t c = ctab[k];
            v[k] = inb && c.p ? act_ab(ldc<T>(c, off), c.a, c.b, G && c.g) : 0.f;
        }
    }
}
/* the 32 staged values of fine position (gz, gy, gx) of an x segment stored at half resolution (sp.up): the exact-2x trilinear
   upsample (align_corners = false, edge clamp; the values of nn_up2_fwd_into) of 8 coarse MX block rows, each first through
   the coarse tensor's GN+SiLU where gmask has the channel (the stored pre-GN a2: no kept s2). c0: the segment's chunk
   descriptor made with the coarse plane. D, H, W: fine dims (even). */
template <typename T>
__device__ __forceinline__ void stage_up32(const chan_t &c0, int gz, int gy, int gx, int D, int H, int W, bool inb, float *v, const float2 *cab = nullptr, unsigned gmask = 0u) {
#pragma unroll
    for (int k = 0; k < 32; k++) v[k] = 0.f;
    if constexpr (IS_MX(T)) {
        if (!inb || !c0.p) return;
        const int Dc = D >> 1, Hc = H >> 1, Wc = W >> 1;
        const int mz[2] = {gz >> 1, min(max((gz >> 1) + ((gz & 1) ? 1 : -1), 0), Dc - 1)}, my[2] = {gy >> 1, min(max((gy >> 1) + ((gy & 1) ? 1 : -1), 0), Hc - 1)},
                  mx[2] = {gx >> 1, min(max((gx >> 1) + ((gx & 1) ? 1 : -1), 0), Wc - 1)};
#pragma unroll
        for (int a = 0; a < 2; a++)
#pragma unroll
            for (int b = 0; b < 2; b++)
#pragma unroll
                for (int c = 0; c < 2; c++) {
                    float r[32];
                    const float w3 = (a ? 0.25f : 0.75f) * (b ? 0.25f : 0.75f) * (c ? 0.25f : 0.75f);
                    mx_row32<MX_BITS(T)>(c0, ((size_t)mz[a] * Hc + my[b]) * Wc + mx[c], true, r);
                    if (gmask) {
#pragma unroll
                        for (int k = 0; k < 32; k++) if ((gmask >> k) & 1u) { const float2 ab = cab[k]; r[k] = act_ab(r[k], ab.x, ab.y, true); }
                    }
#pragma unroll
                    for (int k = 0; k < 32; k++) v[k] = fmaf(w3, r[k], v[k]);
                }
    }
}
/* clamp to the fp16 range for a store that must not become inf; NaN stays NaN (fminf / fmaxf would turn it into
   +-65504 and hide a non-finite forward from the trainer's detection) */
__device__ __forceinline__ float sat_h16(float v) { return fabsf(v) > 65504.f ? copysignf(65504.f, v) : v; }
__device__ __forceinline__ float act_ab(float v, float a, float b, bool G) {
    if (!G) return v;
    v = fmaf(v, a, b);
    return __fdividef(v, 1.f + __expf(-v));
}
/* exponent e (scale 2^e) such that amax * 2^-e <= qmax; returned as the ue8m0 byte e + 127 */
/* NaN-propagating block amax: unsigned max over the magnitude bits (finite magnitudes order as their bits; NaN bits exceed inf),
   where fmaxf would drop a NaN element and the e2m1 conversion would then store it as a finite 6 */
__device__ __forceinline__ unsigned amax_u(unsigned a, float v) { return max(a, __float_as_uint(v) & 0x7fffffffu); }
__device__ __forceinline__ int mx_exp(float amax, float inv_qmax) {
    unsigned u = __float_as_uint(amax * inv_qmax);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 128;   /* NaN amax: scale byte 0xFF (decoded as inf) so the non-finite value survives storage */
    int e = (int)((u >> 23) & 0xff) - 127 + ((u & 0x7fffff) != 0);
    return max(-126, min(126, e));
}
__device__ __forceinline__ float exp2i(int e) { return __uint_as_float((unsigned)(127 + e) << 23); }
__device__ __forceinline__ unsigned cvt_e4m3x4(float a, float b, float c, float d) {   /* bytes a,b,c,d (a lowest) */
    unsigned short lo, hi;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(b), "f"(a));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(d), "f"(c));
    return (unsigned)lo | ((unsigned)hi << 16);
}
/* stochastic rounding to e4m3 by dithering: v (already scaled into the e4m3 range) plus a uniform offset in
   [-1/2, 1/2) ulp, then round to nearest. The ulp is that of v's binade (3 mantissa bits; subnormal below 2^-6). h: a
   per-element hash. */
__device__ __forceinline__ uint32_t sr_hash(uint32_t seed, uint64_t id) {
    uint32_t h = seed ^ (uint32_t)id * 0x9e3779b1u ^ (uint32_t)(id >> 32) * 0x85ebca77u;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    return h;
}
__device__ __forceinline__ float sr_e4m3(float v, uint32_t h) {
    int ex = (int)((__float_as_uint(v) >> 23) & 0xff) - 127;
    if (ex < -6) ex = -6;
    const float ulp = __uint_as_float((unsigned)(127 + ex - 3) << 23);
    return fmaf((float)(h >> 8) * (1.f / 16777216.f) - 0.5f, ulp, v);
}
__device__ __forceinline__ unsigned char cvt_e4m3(float a) {
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(0.f), "f"(a));
    return (unsigned char)(r & 0xff);
}
__device__ __forceinline__ unsigned char cvt_e2m1x2(float lo, float hi) {   /* low nibble = lo */
    unsigned short r;
    asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; cvt.u16.u8 %0, t; }" : "=h"(r) : "f"(hi), "f"(lo));
    return (unsigned char)r;
}
__device__ __forceinline__ unsigned cvt_e2m1x8(const float *v, float m) {   /* 8 values -> 8 nibbles, v[0] lowest */
    return (unsigned)cvt_e2m1x2(v[0] * m, v[1] * m) | ((unsigned)cvt_e2m1x2(v[2] * m, v[3] * m) << 8) |
           ((unsigned)cvt_e2m1x2(v[4] * m, v[5] * m) << 16) | ((unsigned)cvt_e2m1x2(v[6] * m, v[7] * m) << 24);
}
__device__ __forceinline__ unsigned mxf<8>::enc8x(const float *v, float m) { return cvt_e4m3x4(v[0] * m, v[1] * m, v[2] * m, v[3] * m); }   /* 4 values -> 4 bytes */
__device__ __forceinline__ unsigned mxf<4>::enc8x(const float *v, float m) { return cvt_e2m1x8(v, m); }                                   /* 8 values -> 8 nibbles */
/* exact stochastic rounding onto the e2m1 grid: v already scaled into the grid (|v| <= 6 after the block scale; larger
   magnitudes are clamped), the two bracketing grid points are chosen with probability proportional to the distance to the
   other one, so E[q] = v on the non-uniform grid (a dither-then-RN would be biased where the spacing changes). h: per-element
   hash. Returns a value exactly on the grid, so the round-to-nearest conversion that follows is exact. */
/* u: uniform in [0, 1). Inside a binade the e2m1 grid is uniform (spacing 0.5 below 2, 1 in [2, 4), 2 in [4, 6]) and every
   bracket's lower point is a multiple of that spacing, so floor(a / ulp + u) * ulp rounds up with probability frac: exact SR in
   a handful of instructions (no bracket search). */
__device__ __forceinline__ float sr_e2m1_u(float v, float u) {
    const float a = fminf(fabsf(v), 6.f);
    const float inv = a < 2.f ? 2.f : a < 4.f ? 1.f : 0.5f;
    return copysignf(fminf(floorf(fmaf(a, inv, u)) / inv, 6.f), v);
}
__device__ __forceinline__ float sr_e2m1(float v, uint32_t h) { return sr_e2m1_u(v, (float)(h >> 8) * (1.f / 16777216.f)); }
/* two 16-bit uniforms from one hash (the SR loops of the fp4 staging: half the hashing) */
__device__ __forceinline__ float sr_u16(uint32_t h, int hi) { return (float)(hi ? h >> 16 : h & 0xffffu) * (1.f / 65536.f); }
/* the same rounding returning the e2m1 nibble directly: the grid magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} are the codes 0..7 and
   t(a) = 2a / a + 2 / a/2 + 4 on [0, 2) / [2, 4) / [4, 6] is linear between consecutive grid points, so floor(t + u) is exact
   stochastic rounding (P(up) = frac to 2^-16) and needs no conversion instruction */
__device__ __forceinline__ unsigned sr_e2m1_nib(float v, unsigned u16) {   /* |v| <= 6 (block-scaled); t concave -> min of its 3 lines */
    const float a = fabsf(v);
    const float tt = fminf(fminf(a + a, a + 2.f), fmaf(a, 0.5f, 4.f));
    const float u = __uint_as_float(0x3f800000u | (u16 << 7)) - 1.f;   /* u16 / 65536 exactly, no int -> float conversion */
    return min((unsigned)(tt + u), 7u) | (__float_as_uint(v) >> 28 & 8u);
}
/* four 32-bit words (eight 16-bit uniforms) from one hash of key and three remixes: the per-8-value SR of the fp4 staging */
__device__ __forceinline__ void sr_hash4(uint32_t seed, uint64_t key, uint32_t *hh) {
    hh[0] = sr_hash(seed, key);
#pragma unroll
    for (int i = 1; i < 4; i++) { uint32_t h1 = (hh[i - 1] ^ (hh[i - 1] >> 15)) * 0x2c1b3c6du; h1 ^= h1 >> 12; h1 *= 0x297a2d39u; hh[i] = h1 ^ (h1 >> 15); }
}
/* eight values (already scaled by m into [-6, 6]) -> one e2m1 word (v[0] lowest), exact SR with the uniforms of hh */
__device__ __forceinline__ unsigned sr_e2m1_word(const float *v, float m, const uint32_t *hh) {
    unsigned w = 0u;
#pragma unroll
    for (int j = 0; j < 8; j++) w |= sr_e2m1_nib(v[j] * m, (hh[j >> 1] >> (16 * (j & 1))) & 0xffffu) << (4 * j);
    return w;
}
/* decoder up segment, shared-memory version: the coarse rows a block's fine tile needs ((TZ/2 + 4) x 8 x 12 for a TZ x 8 x 16 tile)
   are decoded, put through the coarse tensor's GN+SiLU (gmask) and stored once as e4m3 rows + one scale per row; each fine
   position then interpolates 8 of them from shared memory (the per-position global version decoded and normalised 8 rows
   per fine position: 2.5x slower with the GN). */
template <typename T>
__device__ __forceinline__ void stage_up_coarse(const chan_t &c0, uint8_t *cu, uint8_t *cus, int cz0, int cy0, int cx0, int CZ, int D, int H, int W,
                                                const float2 *cab, unsigned gmask) {
    if constexpr (!IS_MX(T)) return;
    else {
    const int Dc = D >> 1, Hc = H >> 1, Wc = W >> 1, n = CZ * 8 * 12;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int lx = i % 12, ly = (i / 12) % 8, lz = i / 96, cz = cz0 + lz, cy = cy0 + ly, cx = cx0 + lx;
        float r[32];
        const bool ok = c0.p && cz >= 0 && cz < Dc && cy >= 0 && cy < Hc && cx >= 0 && cx < Wc;
        mx_row32<MX_BITS(T)>(c0, ok ? ((size_t)cz * Hc + cy) * Wc + cx : 0, ok, r);
        if (gmask && ok) {
#pragma unroll
            for (int k = 0; k < 32; k++) if ((gmask >> k) & 1u) { const float2 ab = cab[k]; r[k] = act_ab(r[k], ab.x, ab.y, true); }
        }
        unsigned am = 0u;
#pragma unroll
        for (int k = 0; k < 32; k++) { const unsigned u = __float_as_uint(fabsf(r[k])); am = u <= 0x7f800000u ? max(am, u) : am; }
        const int e = mx_exp(__uint_as_float(am), 1.f / 448.f);
        const float m = exp2i(-e);
        uint4 *dst = (uint4 *)(cu + (size_t)i * 32);
        dst[0] = make_uint4(cvt_e4m3x4(r[0] * m, r[1] * m, r[2] * m, r[3] * m), cvt_e4m3x4(r[4] * m, r[5] * m, r[6] * m, r[7] * m),
                            cvt_e4m3x4(r[8] * m, r[9] * m, r[10] * m, r[11] * m), cvt_e4m3x4(r[12] * m, r[13] * m, r[14] * m, r[15] * m));
        dst[1] = make_uint4(cvt_e4m3x4(r[16] * m, r[17] * m, r[18] * m, r[19] * m), cvt_e4m3x4(r[20] * m, r[21] * m, r[22] * m, r[23] * m),
                            cvt_e4m3x4(r[24] * m, r[25] * m, r[26] * m, r[27] * m), cvt_e4m3x4(r[28] * m, r[29] * m, r[30] * m, r[31] * m));
        cus[i] = (uint8_t)(e + 127);
    }
}
}
__device__ __forceinline__ void stage_up_smem(const uint8_t *cu, const uint8_t *cus, int cz0, int cy0, int cx0, int gz, int gy, int gx, int D, int H, int W, bool inb, float *v) {
#pragma unroll
    for (int k = 0; k < 32; k++) v[k] = 0.f;
    if (!inb) return;
    const int Dc = D >> 1, Hc = H >> 1, Wc = W >> 1;
    const int mz[2] = {gz >> 1, min(max((gz >> 1) + ((gz & 1) ? 1 : -1), 0), Dc - 1)}, my[2] = {gy >> 1, min(max((gy >> 1) + ((gy & 1) ? 1 : -1), 0), Hc - 1)},
              mx[2] = {gx >> 1, min(max((gx >> 1) + ((gx & 1) ? 1 : -1), 0), Wc - 1)};
#pragma unroll
    for (int a = 0; a < 2; a++)
#pragma unroll
        for (int b = 0; b < 2; b++)
#pragma unroll
            for (int c = 0; c < 2; c++) {
                const int li = ((mz[a] - cz0) * 8 + (my[b] - cy0)) * 12 + (mx[c] - cx0);
                const float w3 = (a ? 0.25f : 0.75f) * (b ? 0.25f : 0.75f) * (c ? 0.25f : 0.75f) * mx_scale(cus[li]);
                const uint4 h0 = *(const uint4 *)(cu + (size_t)li * 32), h1 = *(const uint4 *)(cu + (size_t)li * 32 + 16);
                const unsigned w[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float2 p = dec_e4m3x2((unsigned short)(w[j] & 0xffff)), q = dec_e4m3x2((unsigned short)(w[j] >> 16));
                    v[4 * j] = fmaf(w3, p.x, v[4 * j]); v[4 * j + 1] = fmaf(w3, p.y, v[4 * j + 1]); v[4 * j + 2] = fmaf(w3, q.x, v[4 * j + 2]); v[4 * j + 3] = fmaf(w3, q.y, v[4 * j + 3]);
                }
            }
}
__device__ __forceinline__ unsigned smem_u32_(const void *p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm_x4(unsigned *r, const void *row_ptr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32_(row_ptr)));
}
__device__ __forceinline__ void mma_f8(float *c, const unsigned *a, const unsigned *b, unsigned sa, unsigned sb) {
#ifdef F8_NOMMA
    c[0] += __uint_as_float(a[0] ^ b[0] ^ sa ^ sb) ; return;
#endif
    asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb));
}
/* m16n8k64 e2m1 x e2m1, scales per 32 K (bytes 0, 1 = K blocks 0..31, 32..63) */
__device__ __forceinline__ void mma_f4(float *c, const unsigned *a, const unsigned *b, unsigned sa, unsigned sb) {
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb));
}

/* ======================= weights: wq[tap][Cop][Cip] e4m3, ws[tap][Cop][Cip/32] ue8m0 ======================= */
__global__ void prep_w8_k(const float *__restrict__ w, uint8_t *__restrict__ wq, uint8_t *__restrict__ ws, int Co, int Ci, int Cop, int Cip, int Cx = -1, int CxP = 0, int Ox = -1, int OxP = 0);   /* defined in lp_f8fwd.cu */

/* Forward epilogue straight from the accumulators: bias, optional accumulate into y (sp.accum), channel-split
   output (sp.y2), optional GroupNorm statistics of the output (osum, per (n, group) sums reduced in smem). Warp row r
   of NR maps to z = oz0 + wz * NR/2 + r/2, y = oy0 + wr + r%2. */
template <int MT, int NR, typename T, bool FST = false>   /* FST: try the shared-memory MX store path (checks its own requirements) */
__device__ __forceinline__ void fwd_epilogue(float (&acc)[MT][NR][2][4], unsigned char *smem_raw, T *y, const float *b, int n, int co0, int Co,
                                             int D, int H, int W, int oz0, int oy0, int ox0, int wz, int wr, double *osum, int Go, const split_t &sp, int N = 0) {
    constexpr int BM = MT * 16, RZ = NR / 2;
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int zs0 = sp.zlo, zs1 = D - sp.zhi;   /* statistics over the z planes this GPU owns (spatial split) */
    float *cs = (float *)smem_raw;
    if (osum) { __syncthreads(); for (int i = threadIdx.x; i < 2 * BM; i += blockDim.x) cs[i] = 0.f; __syncthreads(); }
    bool mx_done = false;
    if constexpr (IS_MX(T) && FST) {
        /* MX output through shared memory: per output block (a 16- or 32-row channel block of the output tensor, or of the
           second tensor of a split output), the quantised rows and scales of the warp's NR x 16 voxels are assembled in a
           per-warp buffer and written as 16-byte chunks (a row of 16 voxels is contiguous: 16 x 8 .. 32 bytes) instead of
           single-byte stores per lane; the bias is loaded once. Values, scales and stored_stats as in the per-lane path below.
           Needs W % 16 == 0 (16-byte alignment) and the kernel's dynamic smem to hold 512 + 8 warps x NR x 16 x 33 bytes. */
        constexpr int B = MX_BITS(T), RBM = 32 * B / 8, WBUF = NR * 16 * (RBM + 1);
        unsigned dsm; asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dsm));
        if ((W & 15) == 0 && blockDim.x == 256 && dsm >= 512u + 8u * WBUF) {
            const int warp = threadIdx.x >> 5;
            const size_t S = (size_t)D * H * W;
            const int OxP = sp.y2 ? (sp.o_split + 31) / 32 * 32 : Co;
            __syncthreads();   /* every warp is past its MMA loop: the staging tiles are free */
            uint8_t *wb = smem_raw + 512 + warp * WBUF;
            float bias[MT][2];
#pragma unroll
            for (int m = 0; m < MT; m++)
#pragma unroll
                for (int h = 0; h < 2; h++) { const int co = co0 + m * 16 + g + 8 * h; bias[m][h] = b && !sp.y2 && co < Co ? b[co] : 0.f; }
#pragma unroll
            for (int j = 0; j < MT; j++) {
                const int cp = co0 + j * 16, sec = sp.y2 && cp >= OxP, Ct = !sp.y2 ? Co : sec ? Co - sp.o_split : sp.o_split;
                const int cl = sec ? cp - OxP : cp, bwt = mx_bw(Ct), MBt = bwt / 16, rbt = mx_rb(bwt, B);
                if (cl % bwt || cl >= Ct) continue;   /* not the first tile of a block, or pure padding (block-uniform) */
                uint8_t *wsb = wb + NR * 16 * rbt;
#pragma unroll
                for (int r = 0; r < NR; r++)
#pragma unroll
                    for (int q2 = 0; q2 < 2; q2++)
#pragma unroll
                        for (int vv = 0; vv < 2; vv++) {
                            const int lv = r * 16 + q2 * 8 + 2 * t + vv;
                            float val[2][2], am = 0.f;
#pragma unroll
                            for (int mm = 0; mm < 2; mm++)
#pragma unroll
                                for (int h = 0; h < 2; h++) {
                                    const int m = j + mm;
                                    const float v = mm < MBt && m < MT ? acc[m < MT ? m : 0][r][q2][2 * h + vv] + bias[m < MT ? m : 0][h] : 0.f;
                                    val[mm][h] = v; am = fmaxf(am, fabsf(v));
                                }
                            am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, 4)); am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, 8)); am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, 16));
                            const int e = mx_exp(am, mxf<B>::inv_qmax);
                            const float mult = exp2i(-e);
                            if constexpr (B == 4) {
                                unsigned nib = 0u;
#pragma unroll
                                for (int mm = 0; mm < 2; mm++)
#pragma unroll
                                    for (int h = 0; h < 2; h++) nib |= (unsigned)(cvt_e2m1x2(val[mm][h] * mult, 0.f) & 15) << (8 * (2 * mm + h));
                                nib |= __shfl_down_sync(0xffffffff, nib, 4) << 4;   /* channel g + 1 into the high nibbles */
#pragma unroll
                                for (int mm = 0; mm < 2; mm++)
#pragma unroll
                                    for (int h = 0; h < 2; h++) if (mm < MBt) {
                                        if (!(g & 1)) wb[lv * rbt + ((mm * 16 + g + 8 * h) >> 1)] = (uint8_t)(nib >> (8 * (2 * mm + h)));
                                        if (osum && sp.stored_stats && j + mm < MT) acc[j + mm < MT ? j + mm : 0][r][q2][2 * h + vv] = dec_e2m1n(nib >> (8 * (2 * mm + h))) * exp2i(e);
                                    }
                            } else {
#pragma unroll
                                for (int mm = 0; mm < 2; mm++)
#pragma unroll
                                    for (int h = 0; h < 2; h++) if (mm < MBt) {
                                        const uint8_t code = cvt_e4m3(val[mm][h] * mult);
                                        wb[lv * rbt + mm * 16 + g + 8 * h] = code;
                                        if (osum && sp.stored_stats && j + mm < MT) acc[j + mm < MT ? j + mm : 0][r][q2][2 * h + vv] = dec_e4m3(code) * exp2i(e);
                                    }
                            }
                            if (g == 0) wsb[lv] = (uint8_t)(e + 127);
                        }
                __syncwarp();
                const int blk = cl / bwt, nbt = mx_nb(Ct);
                uint8_t *qt = sec ? (uint8_t *)sp.y2 : (uint8_t *)y, *st = qt + (size_t)N * nbt * S * rbt;
                const int nch = rbt;   /* 16-byte chunks per row of 16 voxels */
                for (int i = lane; i < NR * nch; i += 32) {
                    const int r = i / nch, c = i % nch, oz = oz0 + wz * RZ + (r >> 1), oy = oy0 + wr + (r & 1);
                    if (oz < D && oy < H) *(uint4 *)(qt + (((size_t)n * nbt + blk) * S + ((size_t)oz * H + oy) * W + ox0) * rbt + c * 16) = *(const uint4 *)(wb + r * 16 * rbt + c * 16);
                }
                if (lane < NR) {
                    const int r = lane, oz = oz0 + wz * RZ + (r >> 1), oy = oy0 + wr + (r & 1);
                    if (oz < D && oy < H) *(uint4 *)(st + ((size_t)n * nbt + blk) * S + ((size_t)oz * H + oy) * W + ox0) = *(const uint4 *)(wsb + r * 16);
                }
                __syncwarp();   /* the buffer is reused by the next block */
            }
            mx_done = true;
        }
    }
    if constexpr (IS_MX(T)) if (!mx_done) {   /* MX output: per voxel and 32- (16-) channel block, amax over the block's rows (lanes g, h, m);
                                   fp4: the nibbles of channels (2k, 2k+1) sit in lanes g, g + 1 -> paired with a shuffle, one byte per even g */
        constexpr int B = MX_BITS(T);
        const size_t S = (size_t)D * H * W;
        const int OxP = sp.y2 ? (sp.o_split + 31) / 32 * 32 : Co;
        uint8_t *q = (uint8_t *)y;
#pragma unroll
        for (int r = 0; r < NR; r++) {
            const int oz = oz0 + wz * RZ + (r >> 1), oy = oy0 + wr + (r & 1);
#pragma unroll
            for (int q2 = 0; q2 < 2; q2++)
#pragma unroll
                for (int vv = 0; vv < 2; vv++) {
                    const int ox = ox0 + q2 * 8 + 2 * t + vv;
                    const bool ok = oz < D && oy < H && ox < W;
#pragma unroll
                    for (int j = 0; j < MT; j++) {
                        /* the tensor and block of this 16-row tile: split outputs are stored as two MX tensors whose padded
                           channel segments (OxP = pad32(o_split)) start at row 0 and OxP */
                        const int cp = co0 + j * 16, sec = sp.y2 && cp >= OxP, Ct = !sp.y2 ? Co : sec ? Co - sp.o_split : sp.o_split;
                        const int cl = sec ? cp - OxP : cp, bwt = mx_bw(Ct), MBt = bwt / 16;
                        if (cl % bwt || cl >= Ct) continue;    /* not the first tile of a block, or pure padding */
                        float val[2][2], am = 0.f;
#pragma unroll
                        for (int mm = 0; mm < 2; mm++)
#pragma unroll
                            for (int h = 0; h < 2; h++) {
                                const int m = j + mm, co = co0 + m * 16 + g + 8 * h;
                                float v = 0.f;
                                if (mm < MBt && m < MT) v = acc[m < MT ? m : 0][r][q2][2 * h + vv] + (b && !sp.y2 && co < Co ? b[co] : 0.f);
                                val[mm][h] = v; am = fmaxf(am, fabsf(v));
                            }
                        am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, 4)); am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, 8)); am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, 16));
                        const int e = mx_exp(am, mxf<B>::inv_qmax);
                        const float mult = exp2i(-e);
                        unsigned nib = 0u;   /* fp4: this lane's 4 nibbles (mm, h) in bits 8 (2 mm + h) */
                        if constexpr (B == 4) {
#pragma unroll
                            for (int mm = 0; mm < 2; mm++)
#pragma unroll
                                for (int h = 0; h < 2; h++) nib |= (unsigned)(cvt_e2m1x2(val[mm][h] * mult, 0.f) & 15) << (8 * (2 * mm + h));
                            nib |= __shfl_down_sync(0xffffffff, nib, 4) << 4;   /* channel g + 1 (next lane group) into the high nibbles */
                        }
                        if (ok) {
                            const int blk = cl / bwt, nbt = mx_nb(Ct), rbt = mx_rb(bwt, B);
                            uint8_t *qt = sec ? (uint8_t *)sp.y2 : q, *st = qt + (size_t)N * nbt * S * rbt;
                            const size_t v = ((size_t)oz * H + oy) * W + ox;
                            uint8_t *dst = qt + (((size_t)n * nbt + blk) * S + v) * rbt;
#pragma unroll
                            for (int mm = 0; mm < 2; mm++)
#pragma unroll
                                for (int h = 0; h < 2; h++) if (mm < MBt) {
                                    if constexpr (B == 8) {
                                        const uint8_t code = cvt_e4m3(val[mm][h] * mult);
                                        dst[mm * 16 + g + 8 * h] = code;
                                        if (osum && sp.stored_stats && j + mm < MT)
                                            acc[j + mm][r][q2][2 * h + vv] = dec_e4m3(code) * exp2i(e);
                                    } else {
                                        if (!(g & 1)) dst[(mm * 16 + g + 8 * h) >> 1] = (uint8_t)(nib >> (8 * (2 * mm + h)));
                                        if (osum && sp.stored_stats && j + mm < MT)
                                            acc[j + mm][r][q2][2 * h + vv] = dec_e2m1n(nib >> (8 * (2 * mm + h))) * exp2i(e);
                                    }
                                }
                            if (g == 0) st[((size_t)n * nbt + blk) * S + v] = (uint8_t)(e + 127);
                        }
                    }
                }
        }
    }
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            int co = co0 + m * 16 + g + 8 * h;
            float ps = 0.f, pss = 0.f;
#pragma unroll
            for (int r = 0; r < NR; r++) {
                int oz = oz0 + wz * RZ + (r >> 1), oy = oy0 + wr + (r & 1);
                if (oz >= D || oy >= H || co >= Co) continue;
                float bias = IS_MX(T) && sp.stored_stats ? 0.f : b ? b[co] : 0.f;
                const bool zst = oz >= zs0 && oz < zs1;
                T *yp = IS_MX(T) ? y : (sp.y2 && co >= sp.o_split) ? (T *)sp.y2 + (((size_t)n * (Co - sp.o_split) + co - sp.o_split) * D + oz) * H * W + (size_t)oy * W
                                                    : y + (((size_t)n * (sp.y2 ? sp.o_split : Co) + co) * D + oz) * H * W + (size_t)oy * W;
#pragma unroll
                for (int q = 0; q < 2; q++) {
                    int ox = ox0 + q * 8 + 2 * t;
                    float v0 = acc[m][r][q][2 * h] + bias, v1 = acc[m][r][q][2 * h + 1] + bias;
                    if constexpr (IS_MX(T)) {   /* stored_stats rewrites acc with the exact decoded output above */
                        if (ox < W && zst) { ps += v0; pss += v0 * v0; }
                        if (ox + 1 < W && zst) { ps += v1; pss += v1 * v1; }
                    } else {
                        if (Go > 0 && std::is_same<T, __half>::value) {   /* include normalization-input replay without statistics */
                            v0 = sat_h16(v0); v1 = sat_h16(v1);
                        }
                        if (!(W & 1) && ox + 1 < W) {   /* paired store */
                            if (sp.accum) { v0 += ldx(yp, (size_t)ox); v1 += ldx(yp, (size_t)ox + 1); }
                            stx2(yp, (size_t)ox, v0, v1); if (zst) { ps += v0 + v1; pss += v0 * v0 + v1 * v1; } continue;
                        }
                        if (ox < W) { if (sp.accum) v0 += ldx(yp, (size_t)ox); stx(yp, (size_t)ox, v0); if (zst) { ps += v0; pss += v0 * v0; } }
                        if (ox + 1 < W) { if (sp.accum) v1 += ldx(yp, (size_t)ox + 1); stx(yp, (size_t)ox + 1, v1); if (zst) { ps += v1; pss += v1 * v1; } }
                    }
                }
            }
            if (osum) {
                ps += __shfl_xor_sync(0xffffffff, ps, 1); ps += __shfl_xor_sync(0xffffffff, ps, 2);
                pss += __shfl_xor_sync(0xffffffff, pss, 1); pss += __shfl_xor_sync(0xffffffff, pss, 2);
                if (t == 0) { atomicAdd(&cs[m * 16 + g + 8 * h], ps); atomicAdd(&cs[BM + m * 16 + g + 8 * h], pss); }
            }
        }
    if (osum) {
        __syncthreads();
        if (threadIdx.x < BM && co0 + (int)threadIdx.x < Co) {
            int cpg = Co / Go, ng = n * Go + (co0 + threadIdx.x) / cpg;
            atomicAdd(&osum[2 * ng], (double)cs[threadIdx.x]);
            atomicAdd(&osum[2 * ng + 1], (double)cs[BM + threadIdx.x]);
        }
    }
}


/* ---- shared by several kernel families ---- */
#define F8_CI 32
#define F8_T 720
__device__ __forceinline__ int sw16(int row, int half) { return row * 32 + ((half ^ ((row >> 2) & 1)) << 4); }
template <typename T> T *lp_buf(int slot, size_t n) {
    static void *buf[8][6]; static size_t cap[8][6];
    int d = cur_dev_();
    if (n * sizeof(T) > cap[d][slot]) { if (buf[d][slot]) cudaFree(buf[d][slot]); cudaMalloc(&buf[d][slot], n * sizeof(T)); cap[d][slot] = n * sizeof(T); }
    return (T *)buf[d][slot];
}
#define P16_KS 18
__global__ void prep_w8p_k(const float *__restrict__ w, uint8_t *__restrict__ wq, uint8_t *__restrict__ ws, int Co, int Ci, int Cop, int Ox, int OxP);   /* defined in lp_f8fwd.cu */
template <typename T>
__device__ __forceinline__ void stage_row16(const chan_t &c0, int Ci, size_t plane, size_t off, bool inb, bool G, const float2 *gab, float *v) {
    if (!inb || !c0.p) {
#pragma unroll
        for (int k = 0; k < 16; k++) v[k] = 0.f;
        return;
    }
    if constexpr (IS_MX(T)) {
        const uint8_t *p = (const uint8_t *)c0.p + off * c0.rb;
        const float s = mx_scale(c0.sp[off]);
        if constexpr (IS_MX8(T)) {
            uint4 u;
            if (c0.bw == 8) { const uint2 q = __ldg((const uint2 *)p); u = make_uint4(q.x, q.y, 0u, 0u); }
            else u = __ldg((const uint4 *)p);
            const unsigned w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float2 a = dec_e4m3x2((unsigned short)(w[j] & 0xffff)), b = dec_e4m3x2((unsigned short)(w[j] >> 16));
                v[4 * j] = a.x * s; v[4 * j + 1] = a.y * s; v[4 * j + 2] = b.x * s; v[4 * j + 3] = b.y * s;
            }
        } else {
            const uint2 u = c0.bw == 8 ? make_uint2(__ldg((const unsigned *)p), 0u) : __ldg((const uint2 *)p);
            const unsigned w[2] = {u.x, u.y};
#pragma unroll
            for (int j = 0; j < 2; j++)
#pragma unroll
                for (int q = 0; q < 4; q++) { float2 a = dec_e2m1x2(w[j] >> (8 * q)); v[8 * j + 2 * q] = a.x * s; v[8 * j + 2 * q + 1] = a.y * s; }
        }
    } else {
        const T *xp = (const T *)c0.p + off;
#pragma unroll
        for (int k = 0; k < 16; k++) v[k] = k < Ci ? ldx(xp, (size_t)k * plane) : 0.f;
    }
    if (G) {
#pragma unroll
        for (int k = 0; k < 16; k++) v[k] = k < Ci ? act_ab(v[k], gab[k].x, gab[k].y, true) : 0.f;
    }
}
static inline int lp_dtype_check(const char *fn, int xbf, int ybf) {
    if (xbf != ybf) { fprintf(stderr, "%s: mixed activation types (x bf16 %d, y bf16 %d) are not instantiated\n", fn, xbf, ybf); abort(); }
    return xbf;
}
#define X8_CS 1040   /* per-channel ring stride: 260 words == 4 mod 32 */
#define X8_PS 240    /* plane: 10 rows x 24 B */
#define G8_CS 272
#ifndef BW8_MINB
#define BW8_MINB 2
#endif
__device__ __forceinline__ float upc(int o, int m, int n_in) {
    if (o < 0 || o >= 2 * n_in) return 0.f;
    int m0 = o >> 1, m1 = (o & 1) ? min(m0 + 1, n_in - 1) : max(m0 - 1, 0);
    return (m == m0 ? 0.75f : 0.f) + (m == m1 ? 0.25f : 0.f);
}
