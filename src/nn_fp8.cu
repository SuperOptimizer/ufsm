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

static cudaError_t g_lp_err = cudaSuccess;
#define LPCK() do { cudaError_t e_ = cudaGetLastError(); if (e_ != cudaSuccess && g_lp_err == cudaSuccess) g_lp_err = e_; } while (0)
static inline unsigned nblk_(size_t n, unsigned b) { return (unsigned)((n + b - 1) / b); }
static int cur_dev_(void) { int d = 0; cudaGetDevice(&d); return d & 7; }

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
__host__ __device__ __forceinline__ int mx_bw(int C) { return C <= 16 ? 16 : 32; }
__host__ __device__ __forceinline__ int mx_nb(int C) { int bw = mx_bw(C); return (C + bw - 1) / bw; }
__host__ __device__ __forceinline__ int mx_rb(int bw, int bits) { return bw * bits / 8; }   /* row bytes */
extern "C" size_t lp_mx8_bytes(int N, int C, size_t S) { return (size_t)N * mx_nb(C) * S * (mx_bw(C) + 1); }
extern "C" size_t lp_mx4_bytes(int N, int C, size_t S) { return (size_t)N * mx_nb(C) * S * (mx_bw(C) / 2 + 1); }
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
            uint4 u = h == 0 || bw == 32 ? __ldg(src + h) : make_uint4(0u, 0u, 0u, 0u);
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
        dst[0] = make_uint4(enc8x(v, m), enc8x(v + 4, m), enc8x(v + 8, m), enc8x(v + 12, m));
        if (bw == 32) dst[1] = make_uint4(enc8x(v + 16, m), enc8x(v + 20, m), enc8x(v + 24, m), enc8x(v + 28, m));
    }
    static __device__ __forceinline__ float dec1(const uint8_t *p, int c) { return dec_e4m3(p[c]); }
};
template <> struct mxf<4> {
    static constexpr float inv_qmax = 1.f / 6.f;
    static __device__ __forceinline__ void dec_row(const uint8_t *p, int bw, float s, float *v) {
        uint2 lo = __ldg((const uint2 *)p), hi = bw == 32 ? __ldg((const uint2 *)p + 1) : make_uint2(0u, 0u);
        const unsigned w[4] = {lo.x, lo.y, hi.x, hi.y};
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int q = 0; q < 4; q++) { float2 a = dec_e2m1x2(w[j] >> (8 * q)); v[8 * j + 2 * q] = a.x * s; v[8 * j + 2 * q + 1] = a.y * s; }
    }
    static __device__ __forceinline__ unsigned enc8x(const float *v, float m);   /* defined after cvt_e2m1x8 */
    static __device__ __forceinline__ void enc_row(uint8_t *p, int bw, const float *v, float m) {
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
        h[0] = __ldg(src); if (c0.bw == 32) h[1] = __ldg(src + 1);
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
   upsample (align_corners = false, edge clamp; the values of nn_up2_fwd_into) of 8 coarse MX block rows, no transform.
   c0: the segment's chunk descriptor made with the coarse plane. D, H, W: fine dims (even). */
template <typename T>
__device__ __forceinline__ void stage_up32(const chan_t &c0, int gz, int gy, int gx, int D, int H, int W, bool inb, float *v) {
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
__global__ void prep_w8_k(const float *__restrict__ w, uint8_t *__restrict__ wq, uint8_t *__restrict__ ws, int Co, int Ci, int Cop, int Cip, int Cx = -1, int CxP = 0, int Ox = -1, int OxP = 0) {
    const int nch = Cip / 32;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)27 * Cop * nch) return;
    int ch = (int)(i % nch), cop = (int)((i / nch) % Cop), t = (int)(i / ((size_t)nch * Cop));
    int co = Ox < 0 ? cop : seg_ci(cop, Co, Ox, OxP);   /* padded output row -> real output channel (-1 = padding) */
    float v[32], amax = 0.f;
#pragma unroll
    for (int k = 0; k < 32; k++) { int ci = Cx < 0 ? ch * 32 + k : seg_ci(ch * 32 + k, Ci, Cx, CxP); v[k] = (co >= 0 && co < Co && ci >= 0 && ci < Ci) ? w[((size_t)co * Ci + ci) * 27 + t] : 0.f; amax = fmaxf(amax, fabsf(v[k])); }
    int e = mx_exp(amax, 1.f / 448.f);
    float m = exp2i(-e);
    uint4 *dst = (uint4 *)(wq + ((size_t)t * Cop + cop) * Cip + ch * 32);   /* padded row (co is the real channel or -1) */
    dst[0] = make_uint4(cvt_e4m3x4(v[0] * m, v[1] * m, v[2] * m, v[3] * m), cvt_e4m3x4(v[4] * m, v[5] * m, v[6] * m, v[7] * m),
                        cvt_e4m3x4(v[8] * m, v[9] * m, v[10] * m, v[11] * m), cvt_e4m3x4(v[12] * m, v[13] * m, v[14] * m, v[15] * m));
    dst[1] = make_uint4(cvt_e4m3x4(v[16] * m, v[17] * m, v[18] * m, v[19] * m), cvt_e4m3x4(v[20] * m, v[21] * m, v[22] * m, v[23] * m),
                        cvt_e4m3x4(v[24] * m, v[25] * m, v[26] * m, v[27] * m), cvt_e4m3x4(v[28] * m, v[29] * m, v[30] * m, v[31] * m));
    ws[i] = (uint8_t)(e + 127);
}

/* Forward epilogue straight from the accumulators: bias, optional accumulate into y (sp.accum), channel-split
   output (sp.y2), optional GroupNorm statistics of the output (osum, per (n, group) sums reduced in smem). Warp row r
   of NR maps to z = oz0 + wz * NR/2 + r/2, y = oy0 + wr + r%2. */
template <int MT, int NR, typename T>
__device__ __forceinline__ void fwd_epilogue(float (&acc)[MT][NR][2][4], unsigned char *smem_raw, T *y, const float *b, int n, int co0, int Co,
                                             int D, int H, int W, int oz0, int oy0, int ox0, int wz, int wr, double *osum, int Go, const split_t &sp, int N = 0) {
    constexpr int BM = MT * 16, RZ = NR / 2;
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    float *cs = (float *)smem_raw;
    if (osum) { __syncthreads(); for (int i = threadIdx.x; i < 2 * BM; i += blockDim.x) cs[i] = 0.f; __syncthreads(); }
    if constexpr (IS_MX(T)) {   /* MX output: per voxel and 32- (16-) channel block, amax over the block's rows (lanes g, h, m);
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
                                    if constexpr (B == 8) dst[mm * 16 + g + 8 * h] = cvt_e4m3(val[mm][h] * mult);
                                    else if (!(g & 1)) dst[(mm * 16 + g + 8 * h) >> 1] = (uint8_t)(nib >> (8 * (2 * mm + h)));
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
                float bias = b ? b[co] : 0.f;
                T *yp = IS_MX(T) ? y : (sp.y2 && co >= sp.o_split) ? (T *)sp.y2 + (((size_t)n * (Co - sp.o_split) + co - sp.o_split) * D + oz) * H * W + (size_t)oy * W
                                                    : y + (((size_t)n * (sp.y2 ? sp.o_split : Co) + co) * D + oz) * H * W + (size_t)oy * W;
#pragma unroll
                for (int q = 0; q < 2; q++) {
                    int ox = ox0 + q * 8 + 2 * t;
                    float v0 = acc[m][r][q][2 * h] + bias, v1 = acc[m][r][q][2 * h + 1] + bias;
                    if constexpr (IS_MX(T)) {   /* stores done above; statistics from the unquantized values */
                        if (ox < W) { ps += v0; pss += v0 * v0; }
                        if (ox + 1 < W) { ps += v1; pss += v1 * v1; }
                    } else {
                        if (osum && std::is_same<T, __half>::value) {   /* fp16 activation feeding a GroupNorm: saturate instead of inf (as conv_fwd_tc_k) */
                            v0 = sat_h16(v0); v1 = sat_h16(v1);
                        }
                        if (!(W & 1) && ox + 1 < W) {   /* paired store */
                            if (sp.accum) { v0 += ldx(yp, (size_t)ox); v1 += ldx(yp, (size_t)ox + 1); }
                            stx2(yp, (size_t)ox, v0, v1); ps += v0 + v1; pss += v0 * v0 + v1 * v1; continue;
                        }
                        if (ox < W) { if (sp.accum) v0 += ldx(yp, (size_t)ox); stx(yp, (size_t)ox, v0); ps += v0; pss += v0 * v0; }
                        if (ox + 1 < W) { if (sp.accum) v1 += ldx(yp, (size_t)ox + 1); stx(yp, (size_t)ox + 1, v1); ps += v1; pss += v1 * v1; }
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

/* ======================= FP8 forward, k=3, stride 1, pad 1 =======================
   Block = 8 warps over a 2 z x 8 y x 16 x output tile (as the BF16 kernel); warp w owns z = w/4, rows 2(w%4)+{0,1}
   (N = 32 voxels as 2 rows x 2 n-tiles), M = MT x 16 output channels. Input tile 4 x 10 x 18 positions x 32 ci,
   one 32-byte row per position; the two 16-byte halves are XOR-swizzled by bit 2 of the position so that
   ldmatrix over 8 consecutive positions is bank-conflict free (the swizzle is a function of the absolute
   position, so every shifted tap view sees it). */
#define F8_CI 32
#define F8_T 720
__device__ __forceinline__ int sw16(int row, int half) { return row * 32 + ((half ^ ((row >> 2) & 1)) << 4); }

template <int MT, int TZ, typename T, typename TO = T>   /* TZ = output z planes per block (2 or 4); warp owns TZ/2 planes x 2 rows */
__global__ void __launch_bounds__(256, 2) conv_fwd_f8_k(const T *__restrict__ x, const uint8_t *__restrict__ wq, const uint8_t *__restrict__ wsc,
                                                     const float *__restrict__ b, TO *__restrict__ y,
                                                     int N, int Ci, int D, int H, int W, int Co, int Cop, int Cip, gnp_t gp, double *__restrict__ osum, int Go, split_t sp) {
    constexpr int BM = MT * 16, TG = 9, RZ = TZ / 2, NR = 2 * RZ, TT = (TZ + 2) * 180, NRX = NR;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    uint8_t *sx = smem_raw;                         /* [TT pos][32 ci] swizzled */
    uint8_t *sxs = sx + TT * 32;                    /* [TT] position scales */
    uint8_t *wa = sxs + ((TT + 127) & ~127);                       /* [TG tap][BM co][32 ci] swizzled */
    uint8_t *was = wa + TG * BM * 32;               /* [TG][BM] */
    chan_t *ctab = (chan_t *)(was + TG * BM + 64 - (TG * BM) % 64);   /* [32] (16-aligned) */
    float2 *cab = (float2 *)(ctab + 32); unsigned *sgm = (unsigned *)(cab + 32);   /* [32] (a, b), GN mask */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wz = warp >> 2, wr = (warp & 3) * 2;
    const int ox0 = blockIdx.x * 16, oy0 = blockIdx.y * 8;
    int bz = blockIdx.z;
    const int nzt = (D + TZ - 1) / TZ;
    const int oz0 = (bz % nzt) * TZ; bz /= nzt;
    const int nmt = Cop / BM;
    const int co0 = (bz % nmt) * BM; const int n = bz / nmt;
    const int nch = Cip / 32;
    float acc[MT][NR][2][4];
#pragma unroll
    for (int m = 0; m < MT; m++) for (int r = 0; r < NR; r++) for (int q = 0; q < 2; q++) for (int k = 0; k < 4; k++) acc[m][r][q][k] = 0.f;
    const size_t plane = (size_t)D * H * W;
    const int Cx = sp.x2 ? sp.c_split : Ci;
    const bool G = gp.G != 0 || sp.gp2.G != 0;
    for (int ci0 = 0; ci0 < Cip; ci0 += F8_CI) {
        __syncthreads();
        const bool upc = IS_MX(T) && sp.up && ci0 < (Cx + 31) / 32 * 32;   /* this chunk is the half-resolution x segment (decoder up part) */
        if (threadIdx.x < 32) {
            const chan_t c = make_chan(IS_MX(T) ? seg_ci(ci0 + threadIdx.x, Ci, Cx, (Cx + 31) / 32 * 32) : ci0 + threadIdx.x, Ci, Cx, n, upc ? (size_t)(D >> 1) * (H >> 1) * (W >> 1) : plane, x, sp, gp, N);
            ctab[threadIdx.x] = c;
            cab[threadIdx.x] = make_float2(c.a, c.b);   /* compact GN coefficients for the MX staging (one broadcast LDS.64 per element) */
            const unsigned gm = __ballot_sync(0xffffffffu, c.g != 0);
            if (threadIdx.x == 0) *sgm = gm;
        }
        __syncthreads();
        const unsigned gmask = *sgm;
        if constexpr (IS_MX8(T)) if (!G && !upc) {   /* MX-fp8 input without transform: the chunk is one stored block -> copy bytes and scales */
            const chan_t c0 = ctab[0];
            for (int pos = threadIdx.x; pos < TT; pos += 256) {
                int ix = pos % 18, iy = (pos / 18) % 10, iz = pos / 180;
                int gz = oz0 - 1 + iz, gy = oy0 - 1 + iy, gx = ox0 - 1 + ix;
                bool inb = c0.p && gz >= 0 && gz < D && gy >= 0 && gy < H && gx >= 0 && gx < W;
                uint4 h0 = make_uint4(0u, 0u, 0u, 0u), h1 = h0;
                unsigned sc = 1u;
                if (inb) {
                    size_t off = ((size_t)gz * H + gy) * W + gx;
                    const uint4 *src = (const uint4 *)((const uint8_t *)c0.p + off * c0.rb);
                    h0 = __ldg(src); if (c0.bw == 32) h1 = __ldg(src + 1);
                    sc = c0.sp[off];
                }
                *(uint4 *)(sx + sw16(pos, 0)) = h0;
                *(uint4 *)(sx + sw16(pos, 1)) = h1;
                sxs[pos] = (uint8_t)sc;
            }
        }
        /* staging: one thread per position, all 32 channels (plane-major reads, or one MX row dequantised: fp8 with gn+silu,
           fp4 always) -> per-position amax -> e4m3 + scale */
        if (!IS_MX8(T) || G || upc) for (int pos = threadIdx.x; pos < TT; pos += 256) {
            int ix = pos % 18, iy = (pos / 18) % 10, iz = pos / 180;
            int gz = oz0 - 1 + iz, gy = oy0 - 1 + iy, gx = ox0 - 1 + ix;
            bool inb = gz >= 0 && gz < D && gy >= 0 && gy < H && gx >= 0 && gx < W;
            size_t off = inb ? ((size_t)gz * H + gy) * W + gx : 0;
            float v[32], amax = 0.f;
            if (upc) stage_up32<T>(ctab[0], gz, gy, gx, D, H, W, inb, v); else stage_row32<T>(ctab, off, inb, G, v, cab, gmask);
#pragma unroll
            for (int k = 0; k < 32; k++) amax = fmaxf(amax, fabsf(v[k]));   /* e4m3 keeps a NaN element itself (0x7f): plain max */
            int e = mx_exp(amax, 1.f / 448.f);
            float m = exp2i(-e);
            if (!IS_MX(T) && sp.sr) {   /* gradient operand: stochastic rounding, dither keyed by the element (deterministic per call); MX gradients are staged as stored */
                const uint64_t vid = (((uint64_t)n * Cip + ci0) * D + gz) * (uint64_t)H * W + (uint64_t)gy * W + gx;
#pragma unroll
                for (int k = 0; k < 32; k++) v[k] = sr_e4m3(v[k] * m, sr_hash(sp.sr, vid * 32 + k)) / m;
            }
            uint4 h0 = make_uint4(cvt_e4m3x4(v[0] * m, v[1] * m, v[2] * m, v[3] * m), cvt_e4m3x4(v[4] * m, v[5] * m, v[6] * m, v[7] * m),
                                  cvt_e4m3x4(v[8] * m, v[9] * m, v[10] * m, v[11] * m), cvt_e4m3x4(v[12] * m, v[13] * m, v[14] * m, v[15] * m));
            uint4 h1 = make_uint4(cvt_e4m3x4(v[16] * m, v[17] * m, v[18] * m, v[19] * m), cvt_e4m3x4(v[20] * m, v[21] * m, v[22] * m, v[23] * m),
                                  cvt_e4m3x4(v[24] * m, v[25] * m, v[26] * m, v[27] * m), cvt_e4m3x4(v[28] * m, v[29] * m, v[30] * m, v[31] * m));
            *(uint4 *)(sx + sw16(pos, 0)) = h0;
            *(uint4 *)(sx + sw16(pos, 1)) = h1;
            sxs[pos] = (uint8_t)(e + 127);
        }
        for (int t0 = 0; t0 < 27; t0 += TG) {
            if (t0) __syncthreads();
            for (int i = threadIdx.x; i < TG * BM * 2; i += 256) {
                int tt = i / (BM * 2), r = i % (BM * 2), c = r >> 1, h = r & 1;
                *(uint4 *)(wa + tt * BM * 32 + sw16(c, h)) = __ldg((const uint4 *)(wq + ((size_t)(t0 + tt) * Cop + co0 + c) * Cip + ci0 + h * 16));
            }
            for (int i = threadIdx.x; i < TG * BM; i += 256) {
                int tt = i / BM, c = i % BM;
                was[i] = wsc[((size_t)(t0 + tt) * Cop + co0 + c) * nch + ci0 / 32];
            }
            __syncthreads();
#pragma unroll
            for (int tt = 0; tt < TG; tt++) {
                int tap = t0 + tt, kz = tap / 9, ky = (tap / 3) % 3, kx = tap % 3;
                unsigned bfr[NR][4], sb[NR][2];
#pragma unroll
                for (int r = 0; r < NR; r++) {
                    int mat = lane >> 3, q = mat >> 1, kh = mat & 1;
                    int rowp = ((wz * RZ + (r >> 1) + kz) * 10 + wr + (r & 1) + ky) * 18 + kx;
                    int pos = rowp + q * 8 + (lane & 7);
                    ldsm_x4(bfr[r], sx + sw16(pos, kh));
                    sb[r][0] = sxs[rowp + g];
                    sb[r][1] = sxs[rowp + 8 + g];
                }
#pragma unroll
                for (int m = 0; m < MT; m++) {
                    unsigned af[4];
                    int mat = lane >> 3, row = m * 16 + (mat & 1) * 8 + (lane & 7);
                    ldsm_x4(af, wa + tt * BM * 32 + sw16(row, mat >> 1));
                    unsigned sa = was[tt * BM + m * 16 + g + 8 * (t & 1)];
#pragma unroll
                    for (int r = 0; r < NR; r++) { mma_f8(acc[m][r][0], af, bfr[r], sa, sb[r][0]); mma_f8(acc[m][r][1], af, bfr[r] + 2, sa, sb[r][1]); }
                }
            }
        }
    }
    fwd_epilogue<MT, NRX, TO>(acc, smem_raw, y, b, n, co0, Co, D, H, W, oz0, oy0, ox0, wz, wr, osum, Go, sp, N);
}

/* ---- small input-channel variant (Ci <= 16): channels padded to CP in {4, 8, 16}; one K block of 32 = TPK = 32/CP
   taps x CP channels, so 27 taps take ceil(27/TPK) mma k-steps instead of 27 half-empty ones. A K block mixes
   positions, so the B scale is one per staged tile (block amax). Weights: wq[kb][Cop][32], ws[kb][Cop]. */
template <int CP>
__global__ void prep_w8s_k(const float *__restrict__ w, uint8_t *__restrict__ wq, uint8_t *__restrict__ ws, int Co, int Ci, int Cop) {
    constexpr int TPK = 32 / CP, NKB = (27 + TPK - 1) / TPK;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)NKB * Cop) return;
    int co = (int)(i % Cop), kb = (int)(i / Cop);
    float v[32], amax = 0.f;
#pragma unroll
    for (int k = 0; k < 32; k++) {
        int tap = kb * TPK + k / CP, ci = k % CP;
        v[k] = (co < Co && ci < Ci && tap < 27) ? w[((size_t)co * Ci + ci) * 27 + tap] : 0.f;
        amax = fmaxf(amax, fabsf(v[k]));
    }
    int e = mx_exp(amax, 1.f / 448.f);
    float m = exp2i(-e);
    uint4 *dst = (uint4 *)(wq + i * 32);
    dst[0] = make_uint4(cvt_e4m3x4(v[0] * m, v[1] * m, v[2] * m, v[3] * m), cvt_e4m3x4(v[4] * m, v[5] * m, v[6] * m, v[7] * m),
                        cvt_e4m3x4(v[8] * m, v[9] * m, v[10] * m, v[11] * m), cvt_e4m3x4(v[12] * m, v[13] * m, v[14] * m, v[15] * m));
    dst[1] = make_uint4(cvt_e4m3x4(v[16] * m, v[17] * m, v[18] * m, v[19] * m), cvt_e4m3x4(v[20] * m, v[21] * m, v[22] * m, v[23] * m),
                        cvt_e4m3x4(v[24] * m, v[25] * m, v[26] * m, v[27] * m), cvt_e4m3x4(v[28] * m, v[29] * m, v[30] * m, v[31] * m));
    ws[i] = (uint8_t)(e + 127);
}
template <int MT, int CP, typename T, typename TO = T>   /* TO: output type (an MX output from a 16-bit network input) */
__global__ void __launch_bounds__(256, 2) conv_fwd_f8s_k(const T *__restrict__ x, const uint8_t *__restrict__ wq, const uint8_t *__restrict__ wsc,
                                                      const float *__restrict__ b, TO *__restrict__ y,
                                                      int N, int Ci, int D, int H, int W, int Co, int Cop, gnp_t gp, double *__restrict__ osum, int Go, split_t sp) {
    constexpr int BM = MT * 16, TPK = 32 / CP, NKB = (27 + TPK - 1) / TPK, NRX = 2;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    uint8_t *sx = smem_raw;                         /* [F8_T pos][CP] */
    uint8_t *wa = sx + F8_T * CP;                   /* [NKB][BM][32] swizzled */
    uint8_t *was = wa + NKB * BM * 32;              /* [NKB][BM] */
    unsigned *samax = (unsigned *)(was + ((NKB * BM + 15) & ~15));
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wz = warp >> 2, wr = (warp & 3) * 2;
    const int ox0 = blockIdx.x * 16, oy0 = blockIdx.y * 8;
    int bz = blockIdx.z;
    const int nzt = (D + 1) / 2;
    const int oz0 = (bz % nzt) * 2; bz /= nzt;
    const int nmt = Cop / BM;
    const int co0 = (bz % nmt) * BM; const int n = bz / nmt;
    float acc[MT][2][2][4];
#pragma unroll
    for (int m = 0; m < MT; m++) for (int r = 0; r < 2; r++) for (int q = 0; q < 2; q++) for (int k = 0; k < 4; k++) acc[m][r][q][k] = 0.f;
    const size_t plane = (size_t)D * H * W;
    const int Cx = sp.x2 ? sp.c_split : Ci;
    chan_t *ctab = (chan_t *)(samax + 4);              /* [16] */
    const bool G = gp.G != 0 || sp.gp2.G != 0;
    if (threadIdx.x == 0) *samax = 0u;
    if (threadIdx.x < CP) ctab[threadIdx.x] = make_chan(threadIdx.x, Ci, Cx, n, plane, x, sp, gp, N);
    for (int i = threadIdx.x; i < NKB * BM * 2; i += 256) {
        int kb = i / (BM * 2), r = i % (BM * 2), c = r >> 1, h = r & 1;
        *(uint4 *)(wa + kb * BM * 32 + sw16(c, h)) = __ldg((const uint4 *)(wq + ((size_t)kb * Cop + co0 + c) * 32 + h * 16));
    }
    for (int i = threadIdx.x; i < NKB * BM; i += 256) { int kb = i / BM, c = i % BM; was[i] = wsc[(size_t)kb * Cop + co0 + c]; }
    __syncthreads();
    float v[3][CP];
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 3; j++) {
        int pos = threadIdx.x + 256 * j;
        int ix = pos % 18, iy = (pos / 18) % 10, iz = pos / 180;
        int gz = oz0 - 1 + iz, gy = oy0 - 1 + iy, gx = ox0 - 1 + ix;
        bool inb = pos < F8_T && gz >= 0 && gz < D && gy >= 0 && gy < H && gx >= 0 && gx < W;
        size_t off = inb ? ((size_t)gz * H + gy) * W + gx : 0;
#pragma unroll
        for (int k = 0; k < CP; k++) {
            float val = 0.f;
            chan_t c = ctab[k];
            if (inb && c.p) val = act_ab(ldc<T>(c, off), c.a, c.b, G && c.g);
            v[j][k] = val; amax = fmaxf(amax, fabsf(val));
        }
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    if (lane == 0) atomicMax(samax, __float_as_uint(amax));
    __syncthreads();
    const int e = mx_exp(__uint_as_float(*samax), 1.f / 448.f);
    const unsigned sb = (unsigned)(e + 127);
    {
        float m = exp2i(-e);
#pragma unroll
        for (int j = 0; j < 3; j++) {
            int pos = threadIdx.x + 256 * j;
            if (pos >= F8_T) break;
            unsigned wv[CP / 4];
#pragma unroll
            for (int k = 0; k < CP / 4; k++) wv[k] = cvt_e4m3x4(v[j][4 * k] * m, v[j][4 * k + 1] * m, v[j][4 * k + 2] * m, v[j][4 * k + 3] * m);
            if constexpr (CP == 16) *(uint4 *)(sx + pos * 16) = make_uint4(wv[0], wv[1], wv[2], wv[3]);
            else if constexpr (CP == 8) *(uint2 *)(sx + pos * 8) = make_uint2(wv[0], wv[1]);
            else *(unsigned *)(sx + pos * 4) = wv[0];
        }
    }
    __syncthreads();
    /* lane (g, t): b0 holds k = 4t..4t+3 -> local tap 4t / CP, channel 4t % CP; b1 holds k = 16 + 4t.. */
    const int tl0 = (4 * t) / CP, ch0 = (4 * t) % CP, tl1 = (16 + 4 * t) / CP, ch1 = (16 + 4 * t) % CP;
#pragma unroll 2
    for (int kb = 0; kb < NKB; kb++) {
        int tap0 = min(kb * TPK + tl0, 26), tap1 = min(kb * TPK + tl1, 26);
        int base0 = ((wz + tap0 / 9) * 10 + wr + (tap0 / 3) % 3) * 18 + tap0 % 3 + g;
        int base1 = ((wz + tap1 / 9) * 10 + wr + (tap1 / 3) % 3) * 18 + tap1 % 3 + g;
        unsigned bfr[2][2][2];
#pragma unroll
        for (int r = 0; r < 2; r++)
#pragma unroll
            for (int q = 0; q < 2; q++) {
                bfr[r][q][0] = *(const unsigned *)(sx + (base0 + r * 18 + q * 8) * CP + ch0);
                bfr[r][q][1] = *(const unsigned *)(sx + (base1 + r * 18 + q * 8) * CP + ch1);
            }
#pragma unroll
        for (int m = 0; m < MT; m++) {
            unsigned af[4];
            int mat = lane >> 3, row = m * 16 + (mat & 1) * 8 + (lane & 7);
            ldsm_x4(af, wa + kb * BM * 32 + sw16(row, mat >> 1));
            unsigned sa = was[kb * BM + m * 16 + g + 8 * (t & 1)];
#pragma unroll
            for (int r = 0; r < 2; r++) { mma_f8(acc[m][r][0], af, bfr[r][0], sa, sb); mma_f8(acc[m][r][1], af, bfr[r][1], sa, sb); }
        }
    }
    fwd_epilogue<MT, NRX, TO>(acc, smem_raw, y, b, n, co0, Co, D, H, W, oz0, oy0, ox0, wz, wr, osum, Go, sp, N);
}

template <typename T> static T *lp_buf(int slot, size_t n) {
    static void *buf[8][6]; static size_t cap[8][6];
    int d = cur_dev_();
    if (n * sizeof(T) > cap[d][slot]) { if (buf[d][slot]) cudaFree(buf[d][slot]); cudaMalloc(&buf[d][slot], n * sizeof(T)); cap[d][slot] = n * sizeof(T); }
    return (T *)buf[d][slot];
}

template <int MT, int CP, typename T, typename TO> static void launch_f8s(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp) {
    constexpr int TPK = 32 / CP, NKB = (27 + TPK - 1) / TPK;
    size_t smem = (size_t)F8_T * CP + NKB * MT * 16 * 32 + ((NKB * MT * 16 + 15) & ~15) + 16 + 16 * sizeof(chan_t);
    if (smem < 8 * 256 * sizeof(float)) smem = 8 * 256 * sizeof(float);
    static int attr[8];
    if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)conv_fwd_f8s_k<MT, CP, T, TO>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
    conv_fwd_f8s_k<MT, CP, T, TO><<<grid, 256, smem>>>((const T *)x, wq, ws, b, (TO *)y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, gp, osum, Go, sp);
}
template <int CP, typename T, typename TO = T> static void small_f8(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp) {
    constexpr int TPK = 32 / CP, NKB = (27 + TPK - 1) / TPK;
    int Cop = (cout + 15) / 16 * 16;
    if (IS_MX(TO) && cout > 16) Cop = (cout + 31) / 32 * 32;
    uint8_t *wq = lp_buf<uint8_t>(0, (size_t)NKB * Cop * 32), *ws = lp_buf<uint8_t>(1, (size_t)NKB * Cop);
    prep_w8s_k<CP><<<nblk_((size_t)NKB * Cop, 128), 128>>>(w, wq, ws, cout, xs.c, Cop);
    int MT = Cop % 64 == 0 ? 4 : Cop % 32 == 0 ? 2 : 1;   /* BM = 16 MT must divide Cop (Cop = 48 -> MT = 1) */
    int nmt = Cop / (MT * 16);
    dim3 grid(nblk_(xs.w, 16), nblk_(xs.h, 8), (unsigned)(nblk_(xs.d, 2) * nmt * xs.n));
    switch (MT) {
    case 1: launch_f8s<1, CP, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    case 2: launch_f8s<2, CP, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    default: launch_f8s<4, CP, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    }
}
/* ---- 16-channel inputs, two taps per K block (fp8): k-step 2 r = [kx0 | kx1] and 2 r + 1 = [kx2 | 0] of tap row r = (kz, ky),
   16 channels per slot, so a conv takes 18 m16n8k32 k-steps instead of 27 half-empty ones and the staged tile is half as
   wide ([pos][16 B]). The K block [kx | kx + 1] of output voxel x is the 32 contiguous bytes of staged positions x + kx and
   x + kx + 1, so the B fragments are plain ldmatrix rows (8 consecutive 16-byte rows: conflict-free without a swizzle); the
   zero half of [kx2 | 0] reads a zero row. A K block spans two positions, so the B scale is one ue8m0 per staged row
   (z, y) of 18 positions. Weights wq[ks][Cop][32] (ks = 2 r + h), ws[ks][Cop]. Shared staging: stage_row32 (16 live
   channels), so plane-major, mx8 and mx4 inputs (bw 16 rows) and the GN+SiLU input transform all work; output via
   fwd_epilogue (any TO). */
#define P16_KS 18
__global__ void prep_w8p_k(const float *__restrict__ w, uint8_t *__restrict__ wq, uint8_t *__restrict__ ws, int Co, int Ci, int Cop, int Ox, int OxP) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)P16_KS * Cop) return;
    const int cop = (int)(i % Cop), ks = (int)(i / Cop), r = ks >> 1, h = ks & 1;
    const int co = Ox < 0 ? cop : seg_ci(cop, Co, Ox, OxP);
    float v[32], amax = 0.f;
#pragma unroll
    for (int k = 0; k < 32; k++) {
        const int kx = 2 * h + (k >> 4), ci = k & 15;
        v[k] = (kx < 3 && co >= 0 && co < Co && ci < Ci) ? w[((size_t)co * Ci + ci) * 27 + r * 3 + kx] : 0.f;
        amax = fmaxf(amax, fabsf(v[k]));
    }
    const int e = mx_exp(amax, 1.f / 448.f);
    const float m = exp2i(-e);
    uint4 *dst = (uint4 *)(wq + i * 32);
    dst[0] = make_uint4(cvt_e4m3x4(v[0] * m, v[1] * m, v[2] * m, v[3] * m), cvt_e4m3x4(v[4] * m, v[5] * m, v[6] * m, v[7] * m),
                        cvt_e4m3x4(v[8] * m, v[9] * m, v[10] * m, v[11] * m), cvt_e4m3x4(v[12] * m, v[13] * m, v[14] * m, v[15] * m));
    dst[1] = make_uint4(cvt_e4m3x4(v[16] * m, v[17] * m, v[18] * m, v[19] * m), cvt_e4m3x4(v[20] * m, v[21] * m, v[22] * m, v[23] * m),
                        cvt_e4m3x4(v[24] * m, v[25] * m, v[26] * m, v[27] * m), cvt_e4m3x4(v[28] * m, v[29] * m, v[30] * m, v[31] * m));
    ws[i] = (uint8_t)(e + 127);
}
/* the 16 staged values of one position for a 16-channel input: dequantised (MX: one bw-16 block row of c0) or read plane-major
   (c0.p = channel 0, channel k at + k plane), then the GN+SiLU affine gab[k].x x + gab[k].y (G; registers or shared memory);
   zeros outside and in channels >= Ci. Shared by the 16-channel kernels. */
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
            const uint4 u = __ldg((const uint4 *)p);
            const unsigned w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float2 a = dec_e4m3x2((unsigned short)(w[j] & 0xffff)), b = dec_e4m3x2((unsigned short)(w[j] >> 16));
                v[4 * j] = a.x * s; v[4 * j + 1] = a.y * s; v[4 * j + 2] = b.x * s; v[4 * j + 3] = b.y * s;
            }
        } else {
            const uint2 u = __ldg((const uint2 *)p);
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
template <int MT, int TZ, typename T, typename TO>
__global__ void __launch_bounds__(256, 2) conv_fwd_f8p_k(const T *__restrict__ x, const uint8_t *__restrict__ wq, const uint8_t *__restrict__ wsc,
                                                      const float *__restrict__ b, TO *__restrict__ y,
                                                      int N, int Ci, int D, int H, int W, int Co, int Cop, gnp_t gp, double *__restrict__ osum, int Go, split_t sp) {
    constexpr int BM = MT * 16, RZ = TZ / 2, NR = 2 * RZ, NROW = (TZ + 2) * 10, TT = NROW * 18;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    uint8_t *sx = smem_raw;                    /* [TT pos][16 B] */
    uint8_t *zrow = sx + TT * 16;              /* [16] zeros */
    uint8_t *sxs = zrow + 16;                  /* [NROW] row scales (64 reserved) */
    uint8_t *wa = sxs + 64;                    /* [18 ks][BM co][32] swizzled */
    uint8_t *was = wa + P16_KS * BM * 32;      /* [18][BM] */
    chan_t *ctab = (chan_t *)(was + ((P16_KS * BM + 15) & ~15));   /* [32] */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wz = warp >> 2, wr = (warp & 3) * 2;
    const int ox0 = blockIdx.x * 16, oy0 = blockIdx.y * 8;
    int bz = blockIdx.z;
    const int nzt = (D + TZ - 1) / TZ;
    const int oz0 = (bz % nzt) * TZ; bz /= nzt;
    const int nmt = Cop / BM;
    const int co0 = (bz % nmt) * BM; const int n = bz / nmt;
    const size_t plane = (size_t)D * H * W;
    const bool G = gp.G != 0;
    for (int i = threadIdx.x; i < P16_KS * BM * 2; i += 256) {
        const int ks = i / (BM * 2), r = i % (BM * 2), c = r >> 1, h = r & 1;
        *(uint4 *)(wa + ks * BM * 32 + sw16(c, h)) = __ldg((const uint4 *)(wq + ((size_t)ks * Cop + co0 + c) * 32 + h * 16));
    }
    for (int i = threadIdx.x; i < P16_KS * BM; i += 256) was[i] = wsc[(size_t)(i / BM) * Cop + co0 + i % BM];
    if (threadIdx.x < 32) ctab[threadIdx.x] = make_chan(threadIdx.x < 16 ? (int)threadIdx.x : -1, Ci, Ci, n, plane, x, sp, gp, N);
    if (threadIdx.x < 4) ((unsigned *)zrow)[threadIdx.x] = 0u;
    __syncthreads();
    /* staging: thread per position, values kept in registers until the row amax (smem atomicMax over the row's 18 positions)
       is known, then quantised with the row scale */
    {
        constexpr int NIT = (TT + 255) / 256;
        const chan_t c0 = ctab[0];
        float2 gab[16];
#pragma unroll
        for (int k = 0; k < 16; k++) gab[k] = make_float2(ctab[k].a, ctab[k].b);
        unsigned *ramax = (unsigned *)ctab + 32 * sizeof(chan_t) / 4;   /* [NROW] */
        for (int i = threadIdx.x; i < NROW; i += 256) ramax[i] = 0u;
        __syncthreads();
        float v[NIT][16];
#pragma unroll
        for (int it = 0; it < NIT; it++) {
            const int pos = threadIdx.x + 256 * it, row = pos / 18, ix = pos - 18 * row;
            const int gz = oz0 - 1 + row / 10, gy = oy0 - 1 + row % 10, gx = ox0 - 1 + ix;
            const bool inb = pos < TT && gz >= 0 && gz < D && gy >= 0 && gy < H && gx >= 0 && gx < W;
            stage_row16<T>(c0, Ci, plane, inb ? ((size_t)gz * H + gy) * W + gx : 0, inb, G, gab, v[it]);
            unsigned am = 0u;
#pragma unroll
            for (int k = 0; k < 16; k++) { const unsigned u = __float_as_uint(fabsf(v[it][k])); am = u <= 0x7f800000u ? max(am, u) : am; }   /* finite magnitudes (as fmaxf: e4m3 keeps a NaN element itself) */
            if (pos < TT && am) atomicMax(&ramax[row], am);
        }
        __syncthreads();
#pragma unroll
        for (int it = 0; it < NIT; it++) {
            const int pos = threadIdx.x + 256 * it, row = pos / 18, ix = pos - 18 * row;
            if (pos >= TT) break;
            const int e = mx_exp(__uint_as_float(ramax[row]), 1.f / 448.f);
            const float m = exp2i(-e);
            float *w = v[it];
            if (sp.sr) {   /* gradient operand (backward-data input): stochastic rounding keyed by the element */
                const int gz = oz0 - 1 + row / 10, gy = oy0 - 1 + row % 10, gx = ox0 - 1 + ix;
                const uint64_t vid = (((uint64_t)n * D + gz) * H + gy) * (uint64_t)W + gx;
#pragma unroll
                for (int k = 0; k < 16; k++) w[k] = sr_e4m3(w[k] * m, sr_hash(sp.sr, vid * 16 + k)) / m;
            }
            *(uint4 *)(sx + pos * 16) = make_uint4(cvt_e4m3x4(w[0] * m, w[1] * m, w[2] * m, w[3] * m), cvt_e4m3x4(w[4] * m, w[5] * m, w[6] * m, w[7] * m),
                                                   cvt_e4m3x4(w[8] * m, w[9] * m, w[10] * m, w[11] * m), cvt_e4m3x4(w[12] * m, w[13] * m, w[14] * m, w[15] * m));
            if (ix == 0) sxs[row] = (uint8_t)(e + 127);
        }
    }
    __syncthreads();
    float acc[MT][NR][2][4];
#pragma unroll
    for (int m = 0; m < MT; m++) for (int r = 0; r < NR; r++) for (int q = 0; q < 2; q++) for (int k = 0; k < 4; k++) acc[m][r][q][k] = 0.f;
    const int mat = lane >> 3, kh = mat & 1, vx = (mat >> 1) * 8 + (lane & 7);
#pragma unroll 1
    for (int tr = 0; tr < 9; tr++) {
        const int kz = tr / 3, ky = tr % 3;
        unsigned bA[NR][4], bB[NR][4], sb[NR];
#pragma unroll
        for (int r = 0; r < NR; r++) {
            const int rowi = (wz * RZ + (r >> 1) + kz) * 10 + wr + (r & 1) + ky, base = rowi * 18 + vx;
            ldsm_x4(bA[r], sx + (base + kh) * 16);              /* [kx0 | kx1] */
            ldsm_x4(bB[r], kh ? zrow : sx + (base + 2) * 16);   /* [kx2 | 0] */
            sb[r] = sxs[rowi];
        }
#pragma unroll
        for (int m = 0; m < MT; m++) {
            const int arow = m * 16 + (mat & 1) * 8 + (lane & 7);
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int ks = 2 * tr + h;
                unsigned af[4];
                ldsm_x4(af, wa + ks * BM * 32 + sw16(arow, mat >> 1));
                const unsigned sa = was[ks * BM + m * 16 + g + 8 * (t & 1)];
#pragma unroll
                for (int r = 0; r < NR; r++) {
                    const unsigned *bf = h ? bB[r] : bA[r];
                    mma_f8(acc[m][r][0], af, bf, sa, sb[r]); mma_f8(acc[m][r][1], af, bf + 2, sa, sb[r]);
                }
            }
        }
    }
    fwd_epilogue<MT, NR, TO>(acc, smem_raw, y, b, n, co0, Co, D, H, W, oz0, oy0, ox0, wz, wr, osum, Go, sp, N);
}
template <int MT, int TZ, typename T, typename TO> static void launch_f8p(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp) {
    constexpr int TT = (TZ + 2) * 180, BM = MT * 16;
    size_t smem = (size_t)TT * 16 + 16 + 64 + P16_KS * BM * 32 + ((P16_KS * BM + 15) & ~15) + 32 * sizeof(chan_t) + 64 * 4;
    if (smem < 8 * 256 * sizeof(float)) smem = 8 * 256 * sizeof(float);
    static int attr[8];
    if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)conv_fwd_f8p_k<MT, TZ, T, TO>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
    conv_fwd_f8p_k<MT, TZ, T, TO><<<grid, 256, smem>>>((const T *)x, wq, ws, b, (TO *)y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, gp, osum, Go, sp);
}
template <typename T, typename TO> static void p16_f8(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, int Cop, int Ox, int OxP, gnp_t gp, double *osum, int Go, split_t sp) {
    uint8_t *wq = lp_buf<uint8_t>(0, (size_t)P16_KS * Cop * 32), *ws = lp_buf<uint8_t>(1, (size_t)P16_KS * Cop);
    prep_w8p_k<<<nblk_((size_t)P16_KS * Cop, 128), 128>>>(w, wq, ws, cout, xs.c, Cop, IS_MX(TO) && sp.y2 ? Ox : -1, OxP);
    int MT = Cop % 64 == 0 ? 4 : Cop % 32 == 0 ? 2 : !IS_MX(TO) && Cop == 48 ? 3 : 1;   /* 48 plane-major outputs (dec0.c1 backward-data): one tile, the input staged once */
    const int nmt = Cop / (MT * 16);
    static int tz_env = -1;
    if (tz_env < 0) tz_env = getenv("UFSM_F8_TZ") ? atoi(getenv("UFSM_F8_TZ")) : 0;
    int TZ = tz_env ? tz_env : (MT <= 2 && xs.d >= 16 ? 4 : 2);
    if (MT == 4) TZ = 2;
    dim3 grid(nblk_(xs.w, 16), nblk_(xs.h, 8), (unsigned)(nblk_(xs.d, TZ) * nmt * xs.n));
    switch (MT * 10 + TZ) {
    case 12: launch_f8p<1, 2, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    case 14: launch_f8p<1, 4, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    case 22: launch_f8p<2, 2, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    case 24: launch_f8p<2, 4, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    case 32: launch_f8p<3, 2, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    case 34: launch_f8p<3, 4, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    default: launch_f8p<4, 2, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    }
}
template <int MT, int TZ, typename T, typename TO> static void launch_f8g(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, int Cip, gnp_t gp, double *osum, int Go, split_t sp) {
    constexpr int TT = (TZ + 2) * 180;
    size_t smem = (size_t)TT * 32 + ((TT + 127) & ~127) + 9 * MT * 16 * 33 + 64 + 32 * sizeof(chan_t) + 32 * 8 + 16;
    if (smem < 8 * 256 * sizeof(float)) smem = 8 * 256 * sizeof(float);
    static int attr[8];
    if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)conv_fwd_f8_k<MT, TZ, T, TO>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
    conv_fwd_f8_k<MT, TZ, T, TO><<<grid, 256, smem>>>((const T *)x, wq, ws, b, (TO *)y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, Cip, gp, osum, Go, sp);
}
template <typename T, typename TO = T> static void fwd_f8_t(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp) {
    static int small = -1;
    if (small < 0) small = ufsm_env_on("UFSM_F8_NOSMALL") ? 0 : 1;
    /* tap-packed small-channel kernel: fp32 input up to 16 channels; with bf16 input the general kernel is as fast at 16 */
    if (small && (std::is_same<T, TO>::value || (IS_MX(TO) && !IS_MX(T))) && xs.c <= (sizeof(T) == 2 || IS_MX(T) ? 8 : 16) && !sp.x2) {
        if (xs.c <= 4) small_f8<4, T, TO>(x, xs, w, b, cout, y, gp, osum, Go, sp);
        else if (xs.c <= 8) small_f8<8, T, TO>(x, xs, w, b, cout, y, gp, osum, Go, sp);
        else small_f8<16, T, TO>(x, xs, w, b, cout, y, gp, osum, Go, sp);
        return;
    }
    int Cop = (cout + 15) / 16 * 16, Cip = (xs.c + 31) / 32 * 32;
    int Cx = sp.x2 ? sp.c_split : xs.c, CxP = (Cx + 31) / 32 * 32;
    int Ox = sp.y2 ? sp.o_split : cout, OxP = (Ox + 31) / 32 * 32;
    if (IS_MX(TO)) {   /* MX output blocks of 32 channels need 32-row-aligned m-tiles (split outputs: per-tensor segments) */
        if (sp.y2) Cop = OxP + (cout - Ox + 31) / 32 * 32;
        else if (cout > 16) Cop = (cout + 31) / 32 * 32;
    }
    if (IS_MX(T)) Cip = CxP + (xs.c - Cx + 31) / 32 * 32;   /* MX inputs: per-tensor 32-channel segments */
    static int p16 = -1;
    if (p16 < 0) p16 = getenv("UFSM_F8_PACK16") ? atoi(getenv("UFSM_F8_PACK16")) : 1;
    if (p16 && xs.c <= 16 && !sp.x2 && !sp.up && sp.gp2.G == 0) { p16_f8<T, TO>(x, xs, w, b, cout, y, Cop, Ox, OxP, gp, osum, Go, sp); return; }   /* two taps per K block */
    int nch = Cip / 32;
    uint8_t *wq = lp_buf<uint8_t>(0, (size_t)27 * Cop * Cip), *ws = lp_buf<uint8_t>(1, (size_t)27 * Cop * nch);
    size_t nt = (size_t)27 * Cop * nch;
    if (IS_MX(T) || IS_MX(TO)) prep_w8_k<<<nblk_(nt, 128), 128>>>(w, wq, ws, cout, xs.c, Cop, Cip, IS_MX(T) ? Cx : -1, CxP, IS_MX(TO) && sp.y2 ? Ox : -1, OxP);
    else prep_w8_k<<<nblk_(nt, 128), 128>>>(w, wq, ws, cout, xs.c, Cop, Cip);
    int MT = Cop % 64 == 0 ? 4 : Cop % 32 == 0 ? 2 : 1;   /* BM = 16 MT must divide Cop (Cop = 48 -> MT = 1) */
    int nmt = Cop / (MT * 16);
    static int tz_env = -1;
    if (tz_env < 0) tz_env = getenv("UFSM_F8_TZ") ? atoi(getenv("UFSM_F8_TZ")) : 0;
    int TZ = tz_env ? tz_env : (MT <= 2 && xs.d >= 16 ? 4 : 2);
    if (MT == 4) TZ = 2;
    dim3 grid(nblk_(xs.w, 16), nblk_(xs.h, 8), (unsigned)(nblk_(xs.d, TZ) * nmt * xs.n));
    switch (MT * 10 + TZ) {
    case 12: launch_f8g<1, 2, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    case 14: launch_f8g<1, 4, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    case 22: launch_f8g<2, 2, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    case 24: launch_f8g<2, 4, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    default: launch_f8g<4, 2, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    }
}
static int lp_dtype_check(const char *fn, int xbf, int ybf) {
    if (xbf != ybf) { fprintf(stderr, "%s: mixed activation types (x bf16 %d, y bf16 %d) are not instantiated\n", fn, xbf, ybf); abort(); }
    return xbf;
}
extern "C" int lp_conv_fwd_f8(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, gnp_t gp, double *osum, int Go, split_t sp) {
    if ((xbf == 1 || xbf == 2) && ybf >= 3) {   /* 16-bit network input -> MX activation (tap-packed small kernel) */
        if (xs.c > 8 || sp.x2 || sp.accum) { fprintf(stderr, "lp_conv_fwd_f8: 16-bit in / MX out only for the network input (Ci <= 8)\n"); abort(); }
        if (xbf == 2 && ybf == 4) fwd_f8_t<__half, mx4_t>(x, xs, w, b, cout, y, gp, osum, Go, sp);
        else if (xbf == 2) fwd_f8_t<__half, mx8_t>(x, xs, w, b, cout, y, gp, osum, Go, sp);
        else if (ybf == 4) fwd_f8_t<bf16, mx4_t>(x, xs, w, b, cout, y, gp, osum, Go, sp);
        else fwd_f8_t<bf16, mx8_t>(x, xs, w, b, cout, y, gp, osum, Go, sp);
        LPCK();
        return 0;
    }
    int dt = lp_dtype_check("lp_conv_fwd_f8", xbf, ybf);
    if (dt >= 3 && sp.accum) { fprintf(stderr, "lp_conv_fwd_f8: MX storage with accumulate is not supported\n"); abort(); }
    if (dt == 4) fwd_f8_t<mx4_t>(x, xs, w, b, cout, y, gp, osum, Go, sp);
    else if (dt == 3) fwd_f8_t<mx8_t>(x, xs, w, b, cout, y, gp, osum, Go, sp);
    else if (dt == 2) fwd_f8_t<__half>(x, xs, w, b, cout, y, gp, osum, Go, sp);
    else if (dt) fwd_f8_t<bf16>(x, xs, w, b, cout, y, gp, osum, Go, sp);
    else fwd_f8_t<float>(x, xs, w, b, cout, y, gp, osum, Go, sp);
    LPCK();
    return 0;
}

/* ======================= FP8 weight gradient, k=3, stride 1 =======================
   Block = 9 warps; warp w owns the tap row (kz, ky) = (w/3, w%3) and all three kx, so one pair of 32-bit loads of
   an input row feeds the B fragments of kx = 0, 1, 2 through byte funnel shifts. M = 16*MT output channels (GY),
   N = 8*NT input channels, K = 32 voxels (two 16-voxel output rows of the same z plane). A block walks ZC
   z-steps of 2 output planes (8 y x 16 x each), accumulating in registers, then atomically adds into gw.
   The input tile is a ring of 4 z planes (10 x 18 positions) per channel: each z-step stages only the 2 new planes.
   Both rows of a K block lie in the same input plane for every tap, so the B scale is per (channel, plane).
   Smem: X ring fp8 [ci][4 planes][10 rows][24 B], position p of a row at byte 3 + p so that x = ox0.. is 4-aligned
         (ci stride 1040 B = 260 words == 4 mod 32 -> conflict-free fragment loads), GY tile fp8 [co][16 rows][16 B] (co stride 272 B, ldmatrix conflict-free) + scales. */
#define X8_CS 1040   /* per-channel ring stride: 260 words == 4 mod 32 */
#define X8_PS 240    /* plane: 10 rows x 24 B */
#define G8_CS 272
#ifndef BW8_MINB
#define BW8_MINB 2
#endif
template <int MT, int NT, typename T, typename TG>
__global__ void __launch_bounds__(288, BW8_MINB) conv_bwd_w_f8_k(const T *__restrict__ x, const TG *__restrict__ gy, float *__restrict__ gw, float *__restrict__ gb,
                                                       int N, int Ci, int D, int H, int W, int Co, gnp_t gp, split_t sp, int ZC, int coop) {
    constexpr int CH = 8 * NT, BMo = 16 * MT;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    uint8_t *sxq = smem_raw;                        /* [CH][X8_CS] */
    uint8_t *sg = sxq + CH * X8_CS;                 /* [BMo][G8_CS] */
    uint8_t *sgs = sg + BMo * G8_CS;                /* [BMo][8 ksteps] */
    uint8_t *sxs = sgs + BMo * 8;                   /* [CH][4 slots] */
    float *sbias = (float *)(sxs + CH * 4);         /* [BMo] */
    chan_t *ctab8 = (chan_t *)(((uintptr_t)(sbias + BMo) + 31) & ~(uintptr_t)31);   /* [CH] MX x: per-channel descriptors */
    unsigned *amx = (unsigned *)(ctab8 + CH);      /* [2][CH] MX x: per-channel plane amax (bits), double-buffered by plane */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int ci0 = blockIdx.x * CH, co0 = blockIdx.y * BMo;
    int bz = blockIdx.z;
    const int nxt = (W + 15) / 16, nyt = (H + 7) / 8, nzt = (D + 1) / 2;
    const int ox0 = (bz % nxt) * 16; bz /= nxt;
    const int oy0 = (bz % nyt) * 8; bz /= nyt;
    const int nzc = (nzt + ZC - 1) / ZC;
    const int zc = bz % nzc; const int n = bz / nzc;
    const size_t plane = (size_t)D * H * W;
    const int Cx = sp.x2 ? sp.c_split : Ci;
    const int kz = warp / 3, ky = warp % 3;
    const bool do_bias = gb && blockIdx.x == 0;
    float acc[3][MT][NT][4];
#pragma unroll
    for (int a = 0; a < 3; a++) for (int m = 0; m < MT; m++) for (int q = 0; q < NT; q++) for (int k = 0; k < 4; k++) acc[a][m][q][k] = 0.f;
    if (threadIdx.x < BMo) sbias[threadIdx.x] = 0.f;
    const bool vec = (W & 3) == 0;
    const int zt_begin = zc * ZC;
    /* MX x whose CH channels are consecutive entries of one stored block row (the common case): the voxel's row words are
       loaded and decoded once for all CH channels (thread per voxel), instead of one strided byte per (channel, voxel) */
    bool uni = false;
    if constexpr (IS_MX(T)) {
        if (threadIdx.x < CH) { const int ci = ci0 + threadIdx.x; ctab8[threadIdx.x] = make_chan(ci < Ci ? ci : Ci, Ci, Cx, n, plane, x, sp, gp, N); }
        if (threadIdx.x < 2 * CH) amx[threadIdx.x] = 0u;
        __syncthreads();
        uni = (coop & 1) && ctab8[0].p != nullptr && (ctab8[0].nib & 7) == 0;
#pragma unroll
        for (int q = 1; q < CH; q++) if (ctab8[q].p && (ctab8[q].sp != ctab8[0].sp || ctab8[q].nib != ctab8[0].nib + q)) uni = false;
    }
    for (int zt = zt_begin; zt < nzt && zt < (zc + 1) * ZC; zt++) {
        const int oz0 = zt * 2;
        const int np = zt == zt_begin ? 4 : 2, gz_first = zt == zt_begin ? oz0 - 1 : oz0 + 1;
        __syncthreads();
        if (IS_MX(T) && uni) {   /* thread per voxel (row tid / 18, p = tid % 18) of each new plane, all CH channels */
            const chan_t c0 = ctab8[0];
            const int tid = threadIdx.x, row = tid / 18, p = tid % 18;
            const bool Gany = gp.G != 0 || sp.gp2.G != 0;
            for (int pi = 0; pi < np; pi++) {
                const int gz = gz_first + pi, slot = (gz + 1) & 3;
                unsigned *am = amx + (pi & 1) * CH;
                const int gyy = oy0 - 1 + row, gx = ox0 - 1 + p;
                const bool inb = tid < 180 && gz >= 0 && gz < D && gyy >= 0 && gyy < H && gx >= 0 && gx < W;
                float v[CH];
#pragma unroll
                for (int q = 0; q < CH; q++) v[q] = 0.f;
                if (inb) {
                    const size_t off = cof(c0, gz, gyy, gx, H, W);
                    const float sc = mx_scale(c0.sp[off]);
                    if constexpr (IS_MX4(T)) {
                        const unsigned *rw = (const unsigned *)((const uint8_t *)c0.p + off * c0.rb + (c0.nib >> 1));
#pragma unroll
                        for (int w = 0; w < CH / 8; w++) {
                            const unsigned u = __ldg(rw + w);
#pragma unroll
                            for (int q = 0; q < 4; q++) { const float2 d = dec_e2m1x2(u >> (8 * q)); v[8 * w + 2 * q] = d.x * sc; v[8 * w + 2 * q + 1] = d.y * sc; }
                        }
                    } else {
                        const unsigned *rw = (const unsigned *)((const uint8_t *)c0.p + off * c0.rb);   /* mx8: c.p already points at channel nib */
#pragma unroll
                        for (int w = 0; w < CH / 4; w++) {
                            const unsigned u = __ldg(rw + w);
                            const float2 d0 = dec_e4m3x2((unsigned short)(u & 0xffffu)), d1 = dec_e4m3x2((unsigned short)(u >> 16));
                            v[4 * w] = d0.x * sc; v[4 * w + 1] = d0.y * sc; v[4 * w + 2] = d1.x * sc; v[4 * w + 3] = d1.y * sc;
                        }
                    }
#pragma unroll
                    for (int q = 0; q < CH; q++) { const chan_t &cq = ctab8[q]; v[q] = cq.p ? act_ab(v[q], cq.a, cq.b, Gany && cq.g) : 0.f; }
                }
#pragma unroll
                for (int q = 0; q < CH; q++) { const unsigned a = __reduce_max_sync(0xffffffffu, __float_as_uint(v[q]) & 0x7fffffffu); if (lane == 0 && a) atomicMax(&am[q], a); }
                __syncthreads();
                if (tid < 180) {
                    uint8_t *dst = sxq + slot * X8_PS + row * 24 + 3 + p;
#pragma unroll
                    for (int q = 0; q < CH; q++) dst[q * X8_CS] = cvt_e4m3(v[q] * exp2i(-mx_exp(__uint_as_float(am[q]), 1.f / 448.f)));
                }
                if (tid < CH) { sxs[tid * 4 + slot] = (uint8_t)(mx_exp(__uint_as_float(am[tid]), 1.f / 448.f) + 127); amx[((pi + 1) & 1) * CH + tid] = 0u; }
            }
        } else
        /* X: warp per (channel, plane): 10 rows x (4 aligned float4 for x = ox0..ox0+15, plus the halo voxels
           ox0-1 and ox0+16); position p of a row is stored at byte 3 + p of its 24-byte slot */
        for (int task = warp; task < CH * np; task += 9) {
            int k = task / np, gz = gz_first + task % np, slot = (gz + 1) & 3, ci = ci0 + k;
            const bool ok = ci < Ci && gz >= 0 && gz < D;
            chan_t c = make_chan(ok ? ci : Ci, Ci, Cx, n, plane, x, sp, gp, N);
            const T *xc = ok && !IS_MX(T) ? (const T *)c.p + (size_t)gz * H * W : x;
            const bool G = (gp.G != 0 || sp.gp2.G != 0) && c.g, el = IS_MX(T);   /* el: per-element access (MX rows) */
            float4 v4[2]; float vs = 0.f, amax = 0.f;
#pragma unroll
            for (int i = 0; i < 2; i++) {
                int f = lane + 32 * i, row = f >> 2, gyy = oy0 - 1 + row, gx = ox0 + 4 * (f & 3);
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (ok && f < 40 && gyy >= 0 && gyy < H) {
                    const T *src = xc + (size_t)gyy * W + gx;
                    if constexpr (IS_MX(T)) { if (gx < W) v = ldc4_mx<T>(c, cof(c, gz, gyy, gx, H, W), min(4, W - gx), vec); }   /* one call per 4 voxels */
                    else if (vec) { if (gx < W) v = ldx4(src); }
                    else { if (gx < W) v.x = ldx(src, 0); if (gx + 1 < W) v.y = ldx(src, 1); if (gx + 2 < W) v.z = ldx(src, 2); if (gx + 3 < W) v.w = ldx(src, 3); }
                    v.x = gx < W ? act_ab(v.x, c.a, c.b, G) : 0.f; v.y = gx + 1 < W ? act_ab(v.y, c.a, c.b, G) : 0.f;
                    v.z = gx + 2 < W ? act_ab(v.z, c.a, c.b, G) : 0.f; v.w = gx + 3 < W ? act_ab(v.w, c.a, c.b, G) : 0.f;
                }
                v4[i] = v;
                amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
            }
            {   /* halo: lane < 20 -> row lane >> 1, side lane & 1 (x = ox0 - 1 or ox0 + 16) */
                int row = lane >> 1, gyy = oy0 - 1 + row, gx = (lane & 1) ? ox0 + 16 : ox0 - 1;
                if (ok && lane < 20 && gyy >= 0 && gyy < H && gx >= 0 && gx < W) vs = act_ab(el ? ldc<T>(c, cof(c, gz, gyy, gx, H, W)) : ldx(xc, (size_t)gyy * W + gx), c.a, c.b, G);
                amax = fmaxf(amax, fabsf(vs));
            }
#pragma unroll
            for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
            int e = mx_exp(amax, 1.f / 448.f);
            float m = exp2i(-e);
            uint8_t *dst = sxq + k * X8_CS + slot * X8_PS;
#pragma unroll
            for (int i = 0; i < 2; i++) {
                int f = lane + 32 * i;
                if (f < 40) *(unsigned *)(dst + (f >> 2) * 24 + 4 + 4 * (f & 3)) = cvt_e4m3x4(v4[i].x * m, v4[i].y * m, v4[i].z * m, v4[i].w * m);
            }
            if (lane < 20) dst[(lane >> 1) * 24 + ((lane & 1) ? 20 : 3)] = cvt_e4m3(vs * m);
            if (lane == 0) sxs[k * 4 + slot] = (uint8_t)(e + 127);
        }
        /* GY stored MX-fp8 with the BMo output channels inside one block row: thread per output voxel of the z-step (256), the
           voxel's row words decoded once for all BMo channels; a K block (2 rows x 16 x) is exactly one warp, so the block
           amax is one redux per channel */
        bool gdone = false;
        if constexpr (IS_MX8(TG)) {
            const int gbw = mx_bw(Co);
            if ((coop & 2) && (co0 % gbw) + BMo <= gbw) {
                gdone = true;
                if (threadIdx.x < 256) {
                    const int ks = threadIdx.x >> 5, row = ks * 2 + (lane >> 4), xx = lane & 15, vz = row >> 3, vy = row & 7;
                    const int oz = oz0 + vz, oy = oy0 + vy, ox = ox0 + xx;
                    const size_t Sg = (size_t)D * H * W, gri = ((size_t)n * mx_nb(Co) + co0 / gbw) * Sg;
                    const uint8_t *gq = (const uint8_t *)gy;
                    float v[BMo];
#pragma unroll
                    for (int c = 0; c < BMo; c++) v[c] = 0.f;
                    if (oz < D && oy < H && ox < W) {
                        const size_t vo = ((size_t)oz * H + oy) * W + ox;
                        const unsigned *rw = (const unsigned *)(gq + (gri + vo) * gbw + co0 % gbw);
                        const float sc = mx_scale(gq[(size_t)N * mx_nb(Co) * Sg * gbw + gri + vo]);
#pragma unroll
                        for (int w = 0; w < BMo / 4; w++) {
                            const unsigned u = __ldg(rw + w);
                            const float2 d0 = dec_e4m3x2((unsigned short)(u & 0xffffu)), d1 = dec_e4m3x2((unsigned short)(u >> 16));
                            v[4 * w] = d0.x * sc; v[4 * w + 1] = d0.y * sc; v[4 * w + 2] = d1.x * sc; v[4 * w + 3] = d1.y * sc;
                        }
#pragma unroll
                        for (int c = 0; c < BMo; c++) if (co0 + c >= Co) v[c] = 0.f;
                    }
                    const uint64_t vid0 = (((uint64_t)n * Co * D + oz) * H + oy) * (uint64_t)W + ox;   /* element id of channel 0; + c * D H W */
#pragma unroll
                    for (int c = 0; c < BMo; c++) {
                        const int e = mx_exp(__uint_as_float(__reduce_max_sync(0xffffffffu, __float_as_uint(v[c]) & 0x7fffffffu)), 1.f / 448.f);
                        float q = v[c] * exp2i(-e);
                        if (sp.sr) q = sr_e4m3(q, sr_hash(sp.sr, vid0 + (uint64_t)(co0 + c) * Sg));   /* same key as the per-block path */
                        sg[c * G8_CS + row * 16 + xx] = cvt_e4m3(q);
                        if (lane == 0) sgs[c * 8 + ks] = (uint8_t)(e + 127);
                        if (do_bias) {
                            float sm = v[c];
#pragma unroll
                            for (int o = 16; o; o >>= 1) sm += __shfl_xor_sync(0xffffffffu, sm, o);
                            if (lane == 0) atomicAdd(&sbias[c], sm);
                        }
                    }
                }
            }
        }
        /* GY: K block (co, ks) = 2 rows x 16 voxels = 8 float4; warp covers 4 blocks, 8 lanes each */
        if (!gdone) for (int task = warp; task < BMo * 2; task += 9) {
            int blk = task * 4 + (lane >> 3), c = blk >> 3, ks = blk & 7, f = lane & 7;
            int row = ks * 2 + (f >> 2), vx = 4 * (f & 3), vz = row >> 3, vy = row & 7;
            int co = co0 + c, oz = oz0 + vz, oy = oy0 + vy, ox = ox0 + vx;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (co < Co && oz < D && oy < H) {
                if constexpr (IS_MX8(TG)) {
                    const size_t vo = ((size_t)oz * H + oy) * W + ox, Sg = (size_t)D * H * W;
                    float q8[8];
                    if (ox < W) { ldmx8_8(gy, N, Co, Sg, n, co, vo, min(4, W - ox), vec, q8); v = make_float4(q8[0], q8[1], q8[2], q8[3]); }
                } else {
                const TG *src = gy + (((size_t)n * Co + co) * D + oz) * H * W + (size_t)oy * W + ox;
                if (vec) { if (ox < W) v = ldx4<TG>(src); }
                else { if (ox < W) v.x = ldx(src, 0); if (ox + 1 < W) v.y = ldx(src, 1); if (ox + 2 < W) v.z = ldx(src, 2); if (ox + 3 < W) v.w = ldx(src, 3); }
                }
            }
            float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))), sm = (v.x + v.y) + (v.z + v.w);
#pragma unroll
            for (int o = 4; o; o >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o)); sm += __shfl_xor_sync(0xffffffff, sm, o); }
            int e = mx_exp(amax, 1.f / 448.f);
            float m = exp2i(-e);
            float4 q = make_float4(v.x * m, v.y * m, v.z * m, v.w * m);
            if (sp.sr) {   /* stochastic rounding of the gradient operand, keyed by the element */
                const uint64_t vid = ((((uint64_t)n * Co + co) * D + oz) * H + oy) * (uint64_t)W + ox;
                q.x = sr_e4m3(q.x, sr_hash(sp.sr, vid)); q.y = sr_e4m3(q.y, sr_hash(sp.sr, vid + 1));
                q.z = sr_e4m3(q.z, sr_hash(sp.sr, vid + 2)); q.w = sr_e4m3(q.w, sr_hash(sp.sr, vid + 3));
            }
            *(unsigned *)(sg + c * G8_CS + row * 16 + vx) = cvt_e4m3x4(q.x, q.y, q.z, q.w);
            if (f == 0) { sgs[c * 8 + ks] = (uint8_t)(e + 127); if (do_bias) atomicAdd(&sbias[c], sm); }
        }
        __syncthreads();
#pragma unroll
        for (int vz = 0; vz < 2; vz++) {
            const int slot = (oz0 + vz + kz) & 3;
            unsigned sb[NT];
#pragma unroll
            for (int q = 0; q < NT; q++) sb[q] = sxs[(q * 8 + g) * 4 + slot];
#pragma unroll 2
            for (int kk = 0; kk < 4; kk++) {
                const int ks = vz * 4 + kk, vy0 = 2 * kk;
                unsigned af[MT][4], sa[MT];
#pragma unroll
                for (int m = 0; m < MT; m++) {
                    int mat = lane >> 3, co = m * 16 + (mat & 1) * 8 + (lane & 7), row = 2 * ks + (mat >> 1);
                    ldsm_x4(af[m], sg + co * G8_CS + row * 16);
                    sa[m] = sgs[(m * 16 + g + 8 * (t & 1)) * 8 + ks];
                }
                const int o0 = slot * X8_PS + (vy0 + ky) * 24;
#pragma unroll
                for (int q = 0; q < NT; q++) {
                    const unsigned *p0 = (const unsigned *)(sxq + (q * 8 + g) * X8_CS + o0) + t;
                    unsigned a0 = p0[0], a1 = p0[1], a2 = p0[2], c0 = p0[6], c1 = p0[7], c2 = p0[8];   /* next row = +24 B = +6 words */
                    unsigned b[3][2] = {{__funnelshift_r(a0, a1, 24), __funnelshift_r(c0, c1, 24)}, {a1, c1}, {__funnelshift_r(a1, a2, 8), __funnelshift_r(c1, c2, 8)}};
#pragma unroll
                    for (int kx = 0; kx < 3; kx++)
#pragma unroll
                        for (int m = 0; m < MT; m++) mma_f8(acc[kx][m][q], af[m], b[kx], sa[m], sb[q]);
                }
            }
        }
    }
    if (do_bias) { __syncthreads(); if (threadIdx.x < BMo && co0 + (int)threadIdx.x < Co) atomicAdd(&gb[co0 + threadIdx.x], sbias[threadIdx.x]); }
#pragma unroll
    for (int kx = 0; kx < 3; kx++) {
        int tap = (kz * 3 + ky) * 3 + kx;
#pragma unroll
        for (int m = 0; m < MT; m++)
#pragma unroll
            for (int q = 0; q < NT; q++) {
                int ci = ci0 + q * 8 + 2 * t;
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    int co = co0 + m * 16 + g + 8 * h;
                    if (co >= Co) continue;
                    if (ci < Ci) atomicAdd(&gw[((size_t)co * Ci + ci) * 27 + tap], acc[kx][m][q][2 * h]);
                    if (ci + 1 < Ci) atomicAdd(&gw[((size_t)co * Ci + ci + 1) * 27 + tap], acc[kx][m][q][2 * h + 1]);
                }
            }
    }
}

/* cooperative MX staging of the fp8 weight gradient (bit 0: x, bit 1: gy); 0 = the per-element paths (test hook) */
static int g_f8w_coop = 1;   /* gy (bit 1) measured mixed: -7% at 96->32, +5..14% elsewhere (per-channel work per thread) */
extern "C" void lp_set_f8w_coop(int c) { g_f8w_coop = c; }
template <int MT, int NT, typename T, typename TG> static void launch_bw8(dim3 grid, size_t smem, const void *x, shape5 xs, const TG *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp, int ZC) {
    static int attr[8];
    if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)conv_bwd_w_f8_k<MT, NT, T, TG>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
    conv_bwd_w_f8_k<MT, NT, T, TG><<<grid, 288, smem>>>((const T *)x, gy, gw, gb, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, gp, sp, ZC, g_f8w_coop);
}
template <typename T, typename TG> static void bwd_w_f8_t(const void *x, shape5 xs, const TG *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp) {
    int MT = ys.c >= 32 ? 2 : 1;
    int NT = xs.c <= 8 ? 1 : MT == 1 && xs.c % 24 == 0 ? 3 : 2;   /* 24 input channels per block when cout = 16: gy restaged less (dec0.c1: -6%) */
    if (getenv("UFSM_F8_NT")) NT = atoi(getenv("UFSM_F8_NT"));
    if (getenv("UFSM_F8_MT")) MT = atoi(getenv("UFSM_F8_MT"));
    size_t smem = (size_t)8 * NT * X8_CS + 16 * MT * (G8_CS + 8) + 8 * NT * 4 + 16 * MT * 4 + 16 + 32 + (size_t)8 * NT * (sizeof(chan_t) + 8);   /* + MX x channel table, amax */
    /* z-steps per block: as many as possible (less halo restaging, fewer atomics) while keeping one full wave of blocks */
    int nzt = nblk_(ys.d, 2), base = (int)(((xs.c + 8 * NT - 1) / (8 * NT)) * ((ys.c + 16 * MT - 1) / (16 * MT)) * nblk_(ys.w, 16) * nblk_(ys.h, 8) * ys.n);
    int ZC = nzt < 12 ? nzt : 12;
    while (ZC > 1 && (size_t)base * nblk_(nzt, ZC) < 72) ZC--;
    if (getenv("UFSM_F8_ZC")) ZC = atoi(getenv("UFSM_F8_ZC"));
    int nzc = (nzt + ZC - 1) / ZC;
    dim3 grid((xs.c + 8 * NT - 1) / (8 * NT), (ys.c + 16 * MT - 1) / (16 * MT), (unsigned)(nblk_(ys.w, 16) * nblk_(ys.h, 8) * nzc * ys.n));
    switch (MT * 10 + NT) {
    case 11: launch_bw8<1, 1, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, gp, sp, ZC); break;
    case 12: launch_bw8<1, 2, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, gp, sp, ZC); break;
    case 21: launch_bw8<2, 1, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, gp, sp, ZC); break;
    case 22: launch_bw8<2, 2, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, gp, sp, ZC); break;
    case 13: launch_bw8<1, 3, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, gp, sp, ZC); break;
    case 14: launch_bw8<1, 4, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, gp, sp, ZC); break;
    case 16: launch_bw8<1, 6, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, gp, sp, ZC); break;
    default: fprintf(stderr, "lp_bwd_w_f8: bad MT/NT %d/%d\n", MT, NT); abort();
    }
}
extern "C" int lp_bwd_w_f8(const void *x, int xbf, shape5 xs, const void *gy, int gybf, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp) {
    if (gybf == 4) { fprintf(stderr, "lp_bwd_w_f8: fp4 gradients are not supported\n"); abort(); }
    if (xbf == 4 && gybf == 3) bwd_w_f8_t<mx4_t, mx8_t>(x, xs, (const mx8_t *)gy, ys, gw, gb, gp, sp);
    else if (xbf == 4 && gybf == 2) bwd_w_f8_t<mx4_t, __half>(x, xs, (const __half *)gy, ys, gw, gb, gp, sp);
    else if (xbf == 4 && gybf == 1) bwd_w_f8_t<mx4_t, bf16>(x, xs, (const bf16 *)gy, ys, gw, gb, gp, sp);
    else if (xbf == 4) bwd_w_f8_t<mx4_t, float>(x, xs, (const float *)gy, ys, gw, gb, gp, sp);
    else if (xbf == 3 && gybf == 3) bwd_w_f8_t<mx8_t, mx8_t>(x, xs, (const mx8_t *)gy, ys, gw, gb, gp, sp);
    else if (xbf == 3 && gybf == 2) bwd_w_f8_t<mx8_t, __half>(x, xs, (const __half *)gy, ys, gw, gb, gp, sp);
    else if (xbf == 3 && gybf == 1) bwd_w_f8_t<mx8_t, bf16>(x, xs, (const bf16 *)gy, ys, gw, gb, gp, sp);
    else if (xbf == 3) bwd_w_f8_t<mx8_t, float>(x, xs, (const float *)gy, ys, gw, gb, gp, sp);
    else if (xbf == 2 && gybf) bwd_w_f8_t<__half, __half>(x, xs, (const __half *)gy, ys, gw, gb, gp, sp);
    else if (xbf == 2) bwd_w_f8_t<__half, float>(x, xs, (const float *)gy, ys, gw, gb, gp, sp);
    else if (xbf && gybf) bwd_w_f8_t<bf16, bf16>(x, xs, (const bf16 *)gy, ys, gw, gb, gp, sp);
    else if (xbf) bwd_w_f8_t<bf16, float>(x, xs, (const float *)gy, ys, gw, gb, gp, sp);
    else if (gybf) { fprintf(stderr, "lp_bwd_w_f8: bf16 gradient with fp32 activations is not supported\n"); abort(); }
    else bwd_w_f8_t<float, float>(x, xs, (const float *)gy, ys, gw, gb, gp, sp);
    LPCK();
    return 0;
}

extern "C" const char *lp_check(void) { cudaError_t e = g_lp_err; g_lp_err = cudaSuccess; return e == cudaSuccess ? nullptr : cudaGetErrorString(e); }
/* fp32 -> bf16 copy (test helper for the bf16-activation instantiations) */
__global__ void lp_f2bf_k(const float *x, bf16 *y, size_t n) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) y[i] = __float2bfloat16(x[i]); }
__global__ void lp_bf2f_k(const bf16 *x, float *y, size_t n) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) y[i] = __bfloat162float(x[i]); }
extern "C" void lp_f32_to_bf16(const float *x, size_t n, void *y) { lp_f2bf_k<<<nblk_(n, 256), 256>>>(x, (bf16 *)y, n); LPCK(); }
extern "C" void lp_bf16_to_f32(const void *x, size_t n, float *y) { lp_bf2f_k<<<nblk_(n, 256), 256>>>((const bf16 *)x, y, n); LPCK(); }

/* ---- public direct entry points (fp32 tensors) ---- */
extern "C" void nn_conv3d_fwd_fp8(const float *x, shape5 xs, const float *w, const float *b, int cout, float *y) {
    gnp_t none = {nullptr, nullptr, nullptr, nullptr, 0}; split_t ns = {nullptr, 0, nullptr, 0, 0};
    lp_conv_fwd_f8(x, 0, xs, w, b, cout, y, 0, none, nullptr, 0, ns);
}
extern "C" void nn_conv3d_fwd_fp4(const float *x, shape5 xs, const float *w, const float *b, int cout, float *y) {
    gnp_t none = {nullptr, nullptr, nullptr, nullptr, 0}; split_t ns = {nullptr, 0, nullptr, 0, 0};
    lp_conv_fwd_f4(x, 0, xs, w, b, cout, y, 0, none, nullptr, 0, ns);
}
extern "C" void nn_conv3d_bwd_weight_fp8(const float *x, shape5 xs, const float *gy, shape5 ys, float *gw, float *gb) {
    gnp_t none = {nullptr, nullptr, nullptr, nullptr, 0}; split_t ns = {nullptr, 0, nullptr, 0, 0};
    lp_bwd_w_f8(x, 0, xs, gy, 0, ys, gw, gb, none, ns);
}
/* ======================= FP4 (e2m1, MX ue8m0 scales per 32 K) forward, k=3, stride 1 =======================
   m16n8k64: one mma = two taps x 32 input channels; K block 0 = tap 2p, K block 1 = tap 2p+1 (pair p, 14 pairs,
   the last one half zero). Same tile / warp layout as the FP8 kernel (TZ output planes per block); the input tile is
   [pos][16 B] (32 channels x 4 bit), one scale per (position, 32 channels) -> B scale register bytes 0 / 1 = the two
   taps' positions. Weights wq4[tap][Cop][Cip/2] with one scale per (tap, co, 32-channel chunk).
   Input types: mx4 (stored rows copied straight into the tile when no transform applies, else dequantised + GN+SiLU +
   requantised), mx8 (dequantised, requantised to e2m1), fp32 / 16-bit (staged as the fp8 kernel). Output type TO
   separate from the input (the backward feeds mx8 gradients into 16-bit / mx8 outputs). Inputs with 9..16 channels take the
   packed 16-channel kernel conv_fwd_f4p_k, Ci <= 8 (the network input) the fp8 tap-packed kernel. */
__global__ void prep_w4_k(const float *__restrict__ w, uint8_t *__restrict__ wq, uint8_t *__restrict__ ws, int Co, int Ci, int Cop, int Cip, int Cx = -1, int CxP = 0, int Ox = -1, int OxP = 0) {
    const int nch = Cip / 32;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)28 * Cop * nch) return;              /* tap 27 = the zero half of the last pair */
    int ch = (int)(i % nch), cop = (int)((i / nch) % Cop), t = (int)(i / ((size_t)nch * Cop));
    int co = Ox < 0 ? cop : seg_ci(cop, Co, Ox, OxP);   /* padded output row -> real output channel (-1 = padding) */
    float v[32], amax = 0.f;
#pragma unroll
    for (int k = 0; k < 32; k++) { int ci = Cx < 0 ? ch * 32 + k : seg_ci(ch * 32 + k, Ci, Cx, CxP); v[k] = (t < 27 && co >= 0 && co < Co && ci >= 0 && ci < Ci) ? w[((size_t)co * Ci + ci) * 27 + t] : 0.f; amax = fmaxf(amax, fabsf(v[k])); }
    int e = mx_exp(amax, 1.f / 6.f);
    float m = exp2i(-e);
    *(uint4 *)(wq + ((size_t)t * Cop + cop) * (Cip / 2) + ch * 16) = make_uint4(cvt_e2m1x8(v, m), cvt_e2m1x8(v + 8, m), cvt_e2m1x8(v + 16, m), cvt_e2m1x8(v + 24, m));
    ws[i] = (uint8_t)(e + 127);
}
/* 2D weight scales (UFSM_W4_2D=1): one ue8m0 per (tap, 32 padded output rows, 32-channel input chunk) tile instead of per
   (tap, row, chunk). The flipped weights of the backward-data conv tile the same 32 x 32 blocks with rows and chunks swapped,
   so forward and backward-data multiply the same quantised weights (exactly when both sides use the same 32-channel
   alignment: plane-major / 16-bit on both, or MX segments on both). Warp per tile, lane = row; same layout as prep_w4_k. */
static int g_w2d = -1;
extern "C" void lp_set_w4_2d(int on) { g_w2d = on; lp_wmemo_clear(); }   /* tests; default from UFSM_W4_2D */
__global__ void prep_w4_2d_k(const float *__restrict__ w, uint8_t *__restrict__ wq, uint8_t *__restrict__ ws, int Co, int Ci, int Cop, int Cip, int Cx = -1, int CxP = 0, int Ox = -1, int OxP = 0) {
    const int nch = Cip / 32, nrt = (Cop + 31) / 32, lane = threadIdx.x & 31;
    const size_t tile = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5;
    if (tile >= (size_t)28 * nrt * nch) return;
    const int ch = (int)(tile % nch), rt = (int)((tile / nch) % nrt), t = (int)(tile / ((size_t)nch * nrt));
    const int cop = rt * 32 + lane;
    const int co = cop >= Cop ? -1 : Ox < 0 ? cop : seg_ci(cop, Co, Ox, OxP);
    float v[32], amax = 0.f;
#pragma unroll
    for (int k = 0; k < 32; k++) { int ci = Cx < 0 ? ch * 32 + k : seg_ci(ch * 32 + k, Ci, Cx, CxP); v[k] = (t < 27 && co >= 0 && co < Co && ci >= 0 && ci < Ci) ? w[((size_t)co * Ci + ci) * 27 + t] : 0.f; amax = fmaxf(amax, fabsf(v[k])); }
#pragma unroll
    for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    const int e = mx_exp(amax, 1.f / 6.f);
    const float m = exp2i(-e);
    if (cop >= Cop) return;
    *(uint4 *)(wq + ((size_t)t * Cop + cop) * (Cip / 2) + ch * 16) = make_uint4(cvt_e2m1x8(v, m), cvt_e2m1x8(v + 8, m), cvt_e2m1x8(v + 16, m), cvt_e2m1x8(v + 24, m));
    ws[((size_t)t * Cop + cop) * nch + ch] = (uint8_t)(e + 127);
}
template <int MT, int TZ, typename T, typename TO>
__global__ void __launch_bounds__(256, 2) conv_fwd_f4_k(const T *__restrict__ x, const uint8_t *__restrict__ wq, const uint8_t *__restrict__ wsc,
                                                     const float *__restrict__ b, TO *__restrict__ y,
                                                     int N, int Ci, int D, int H, int W, int Co, int Cop, int Cip, gnp_t gp, double *__restrict__ osum, int Go, split_t sp) {
    constexpr int BM = MT * 16, PG = 7, RZ = TZ / 2, NR = 2 * RZ, TT = (TZ + 2) * 180, NRX = NR;   /* PG: tap pairs per weight stage */
    extern __shared__ __align__(128) unsigned char smem_raw[];
    uint8_t *sx = smem_raw;                         /* [TT pos][16 B] */
    uint8_t *sxs = sx + TT * 16;                    /* [TT] */
    uint8_t *wa = sxs + ((TT + 127) & ~127);        /* [PG pair][BM co][32 B] (tapA 16 B | tapB 16 B), swizzled */
    unsigned short *was = (unsigned short *)(wa + PG * BM * 32);   /* [PG][BM] (byte 0 tapA, byte 1 tapB) */
    chan_t *ctab = (chan_t *)(wa + PG * BM * 32 + ((PG * BM * 2 + 15) & ~15));
    float2 *cab = (float2 *)(ctab + 32); unsigned *sgm = (unsigned *)(cab + 32);   /* [32] (a, b), GN mask */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wz = warp >> 2, wr = (warp & 3) * 2;
    const int ox0 = blockIdx.x * 16, oy0 = blockIdx.y * 8;
    int bz = blockIdx.z;
    const int nzt = (D + TZ - 1) / TZ;
    const int oz0 = (bz % nzt) * TZ; bz /= nzt;
    const int nmt = Cop / BM;
    const int co0 = (bz % nmt) * BM; const int n = bz / nmt;
    const int nch = Cip / 32;
    float acc[MT][NR][2][4];
#pragma unroll
    for (int m = 0; m < MT; m++) for (int r = 0; r < NR; r++) for (int q = 0; q < 2; q++) for (int k = 0; k < 4; k++) acc[m][r][q][k] = 0.f;
    const size_t plane = (size_t)D * H * W;
    const int Cx = sp.x2 ? sp.c_split : Ci;
    const bool G = gp.G != 0 || sp.gp2.G != 0;
    for (int ci0 = 0; ci0 < Cip; ci0 += 32) {
        __syncthreads();
        const bool upc = IS_MX(T) && sp.up && ci0 < (Cx + 31) / 32 * 32;   /* this chunk is the half-resolution x segment (decoder up part) */
        if (threadIdx.x < 32) {
            const chan_t c = make_chan(IS_MX(T) ? seg_ci(ci0 + threadIdx.x, Ci, Cx, (Cx + 31) / 32 * 32) : ci0 + threadIdx.x, Ci, Cx, n, upc ? (size_t)(D >> 1) * (H >> 1) * (W >> 1) : plane, x, sp, gp, N);
            ctab[threadIdx.x] = c;
            cab[threadIdx.x] = make_float2(c.a, c.b);   /* compact GN coefficients for the MX staging (one broadcast LDS.64 per element) */
            const unsigned gm = __ballot_sync(0xffffffffu, c.g != 0);
            if (threadIdx.x == 0) *sgm = gm;
        }
        __syncthreads();
        const unsigned gmask = *sgm;
        if constexpr (IS_MX4(T)) if (!G && !sp.sr && !upc) {   /* mx4 input without transform: the chunk is one stored block -> copy the row and its scale */
            const chan_t c0 = ctab[0];
            for (int pos = threadIdx.x; pos < TT; pos += 256) {
                int ix = pos % 18, iy = (pos / 18) % 10, iz = pos / 180;
                int gz = oz0 - 1 + iz, gy = oy0 - 1 + iy, gx = ox0 - 1 + ix;
                bool inb = c0.p && gz >= 0 && gz < D && gy >= 0 && gy < H && gx >= 0 && gx < W;
                uint4 h0 = make_uint4(0u, 0u, 0u, 0u);
                unsigned sc = 1u;
                if (inb) {
                    size_t off = ((size_t)gz * H + gy) * W + gx;
                    const uint8_t *src = (const uint8_t *)c0.p + off * c0.rb;
                    if (c0.bw == 32) h0 = __ldg((const uint4 *)src); else { uint2 u = __ldg((const uint2 *)src); h0.x = u.x; h0.y = u.y; }
                    sc = c0.sp[off];
                }
                *(uint4 *)(sx + pos * 16) = h0;
                sxs[pos] = (uint8_t)sc;
            }
        }
        if (!IS_MX4(T) || G || sp.sr || upc) for (int pos = threadIdx.x; pos < TT; pos += 256) {   /* stage: dequantise / read, transform, requantise to e2m1 */
            int ix = pos % 18, iy = (pos / 18) % 10, iz = pos / 180;
            int gz = oz0 - 1 + iz, gy = oy0 - 1 + iy, gx = ox0 - 1 + ix;
            bool inb = gz >= 0 && gz < D && gy >= 0 && gy < H && gx >= 0 && gx < W;
            size_t off = inb ? ((size_t)gz * H + gy) * W + gx : 0;
            float v[32];
            if (upc) stage_up32<T>(ctab[0], gz, gy, gx, D, H, W, inb, v); else stage_row32<T>(ctab, off, inb, G, v, cab, gmask);
            unsigned amu = 0u;
#pragma unroll
            for (int k = 0; k < 32; k++) amu = amax_u(amu, v[k]);
            int e = mx_exp(__uint_as_float(amu), 1.f / 6.f);
            float m = exp2i(-e);
            if (sp.sr) {   /* gradient operand: exact stochastic rounding keyed by the element (deterministic per call) */
                const uint64_t vid = (((uint64_t)n * Cip + ci0) * D + gz) * (uint64_t)H * W + (uint64_t)gy * W + gx;
#pragma unroll
                for (int k = 0; k < 32; k += 2) { const uint32_t h = sr_hash(sp.sr, vid * 16 + k / 2); v[k] = sr_e2m1_u(v[k] * m, sr_u16(h, 0)) / m; v[k + 1] = sr_e2m1_u(v[k + 1] * m, sr_u16(h, 1)) / m; }
            }
            *(uint4 *)(sx + pos * 16) = make_uint4(cvt_e2m1x8(v, m), cvt_e2m1x8(v + 8, m), cvt_e2m1x8(v + 16, m), cvt_e2m1x8(v + 24, m));
            sxs[pos] = (uint8_t)(e + 127);
        }
        for (int p0 = 0; p0 < 14; p0 += PG) {
            __syncthreads();
            for (int i = threadIdx.x; i < PG * BM * 2; i += 256) {
                int pp = i / (BM * 2), r = i % (BM * 2), c = r >> 1, h = r & 1, tap = 2 * (p0 + pp) + h;
                *(uint4 *)(wa + pp * BM * 32 + sw16(c, h)) = __ldg((const uint4 *)(wq + ((size_t)tap * Cop + co0 + c) * (Cip / 2) + ci0 / 2));
            }
            for (int i = threadIdx.x; i < PG * BM; i += 256) {
                int pp = i / BM, c = i % BM, tap = 2 * (p0 + pp);
                was[i] = (unsigned short)(wsc[((size_t)tap * Cop + co0 + c) * nch + ci0 / 32] | (wsc[((size_t)(tap + 1) * Cop + co0 + c) * nch + ci0 / 32] << 8));
            }
            __syncthreads();
#pragma unroll
            for (int pp = 0; pp < PG; pp++) {
                const int tA = 2 * (p0 + pp), tB = min(tA + 1, 26);
                const int rA = ((wz * RZ + tA / 9) * 10 + wr + (tA / 3) % 3) * 18 + tA % 3, rB = ((wz * RZ + tB / 9) * 10 + wr + (tB / 3) % 3) * 18 + tB % 3;
                unsigned bfr[NR][4], sb[NR][2];
#pragma unroll
                for (int r = 0; r < NR; r++) {
                    const int rowo = (r >> 1) * 180 + (r & 1) * 18;   /* z plane r/2, row r%2 of the warp's tile */
                    int mat = lane >> 3, q = mat >> 1, isB = mat & 1;
                    int pos = (isB ? rB : rA) + rowo + q * 8 + (lane & 7);
                    ldsm_x4(bfr[r], sx + pos * 16);
#pragma unroll
                    for (int q2 = 0; q2 < 2; q2++) sb[r][q2] = (unsigned)sxs[rA + rowo + q2 * 8 + g] | ((unsigned)sxs[rB + rowo + q2 * 8 + g] << 8);
                }
#pragma unroll
                for (int m = 0; m < MT; m++) {
                    unsigned af[4];
                    int mat = lane >> 3, row = m * 16 + (mat & 1) * 8 + (lane & 7);
                    ldsm_x4(af, wa + pp * BM * 32 + sw16(row, mat >> 1));
                    unsigned sa = was[pp * BM + m * 16 + g + 8 * (t & 1)];
#pragma unroll
                    for (int r = 0; r < NR; r++) { mma_f4(acc[m][r][0], af, bfr[r], sa, sb[r][0]); mma_f4(acc[m][r][1], af, bfr[r] + 2, sa, sb[r][1]); }
                }
            }
        }
    }
    fwd_epilogue<MT, NRX, TO>(acc, smem_raw, y, b, n, co0, Co, D, H, W, oz0, oy0, ox0, wz, wr, osum, Go, sp, N);
}
template <int MT, int TZ, typename T, typename TO> static void launch_f4(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, int Cip, gnp_t gp, double *osum, int Go, split_t sp) {
    constexpr int TT = (TZ + 2) * 180;
    size_t smem = (size_t)TT * 16 + ((TT + 127) & ~127) + 7 * MT * 16 * 32 + ((7 * MT * 16 * 2 + 15) & ~15) + 32 * sizeof(chan_t) + 32 * 8 + 16;
    if (smem < 8 * 256 * sizeof(float)) smem = 8 * 256 * sizeof(float);
    static int attr[8];
    if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)conv_fwd_f4_k<MT, TZ, T, TO>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
    conv_fwd_f4_k<MT, TZ, T, TO><<<grid, 256, smem>>>((const T *)x, wq, ws, b, (TO *)y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, Cip, gp, osum, Go, sp);
}
/* prepared-weight memo: one (wq, ws) pair per conv id (split_t::wkey, set by nn.cu from layer / conv / pass) and device, valid
   while the weight pointer, the shapes and the step counter (lp_wmemo_step, from nn_set_sr_step) are unchanged; wkey 0 = no
   memo (prep every call, as before). lp_wmemo_clear() after any weight change that is not a training step (checkpoint load,
   EMA swap). UFSM_W4_MEMO=0 disables it. */
static unsigned g_wmemo_step = 0, g_wmemo_gen = 0;
extern "C" void lp_wmemo_step(unsigned step) { g_wmemo_step = step; }
extern "C" void lp_wmemo_clear(void) { g_wmemo_gen++; }
#define WMEMO_N 64
static struct wmemo_s { unsigned key, step, gen; const float *w; int Cop, Cip, Cx, CxP, Ox, OxP; uint8_t *wq, *ws; size_t nq, ns; } g_wmemo[8][WMEMO_N];
static int wmemo_on(void) { static int on = -1; if (on < 0) on = getenv("UFSM_W4_MEMO") ? atoi(getenv("UFSM_W4_MEMO")) : 1; return on; }
/* returns 1 if the buffers already hold this weight (no prep needed), 0 if they must be filled; sets *wq / *ws */
static int wmemo_get(unsigned key, const float *w, int Cop, int Cip, int Cx, int CxP, int Ox, int OxP, size_t nq, size_t ns, uint8_t **wq, uint8_t **ws, int slot) {
    if (!key || !wmemo_on()) { *wq = lp_buf<uint8_t>(slot, nq); *ws = lp_buf<uint8_t>(slot + 1, ns); return 0; }
    const int d = cur_dev_();
    wmemo_s *e = &g_wmemo[d][key % WMEMO_N];
    const bool hit = e->key == key && e->w == w && e->Cop == Cop && e->Cip == Cip && e->Cx == Cx && e->CxP == CxP && e->Ox == Ox && e->OxP == OxP && e->step == g_wmemo_step && e->gen == g_wmemo_gen && e->nq == nq && e->ns == ns;
    if (e->nq < nq) { if (e->wq) cudaFree(e->wq); cudaMalloc(&e->wq, nq); e->nq = nq; }
    if (e->ns < ns) { if (e->ws) cudaFree(e->ws); cudaMalloc(&e->ws, ns); e->ns = ns; }
    *wq = e->wq; *ws = e->ws;
    if (hit) return 1;
    e->key = key; e->w = w; e->Cop = Cop; e->Cip = Cip; e->Cx = Cx; e->CxP = CxP; e->Ox = Ox; e->OxP = OxP; e->step = g_wmemo_step; e->gen = g_wmemo_gen;
    return 0;
}
/* ---- 16-channel inputs on the fp4 MMA (m16n8k64 kind::mxf4): one k-step per tap row r = (kz, ky), K = 64 = [kx0 | kx1 | kx2 | 0]
   x 16 channels, two 32-K scale blocks [kx0 | kx1] and [kx2 | 0] (9 k-steps per n-tile). The staged tile holds, per position p,
   the 16 bytes [p | p + 1] (8 B = 16 e2m1 channels each; the last position of a row pairs with zeros), so both K blocks of output
   voxel x are 16-byte-aligned ldmatrix rows: [x | x + 1] and [x + 2 | x + 3] (the kx = 3 half meets zero weights; e2m1 has no
   NaN encoding and both halves share the pair scale, so it adds nothing). B scale: one ue8m0 per pair (the mma's 32-element
   block: max over the two positions, so each position is quantised once per pair it belongs to), A scales one per (row, co,
   32-K block). Weights wq4p[r][Cop][32 B], ws[r][Cop][2]. Staging shared with the fp8 version (stage_row16). */
__global__ void prep_w4p_k(const float *__restrict__ w, uint8_t *__restrict__ wq, uint8_t *__restrict__ ws, int Co, int Ci, int Cop, int Ox, int OxP, int Cs = -1, int cofs = 0) {   /* Cs: w channel stride (-1: Ci), cofs: first channel */
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;   /* (r, co, block) */
    if (i >= (size_t)9 * Cop * 2) return;
    const int blk = (int)(i & 1), cop = (int)((i >> 1) % Cop), r = (int)((i >> 1) / Cop);
    const int co = Ox < 0 ? cop : seg_ci(cop, Co, Ox, OxP);
    float v[32], amax = 0.f;
#pragma unroll
    for (int k = 0; k < 32; k++) {
        const int kx = 2 * blk + (k >> 4), ci = k & 15;
        v[k] = (kx < 3 && co >= 0 && co < Co && ci < Ci) ? w[((size_t)co * (Cs < 0 ? Ci : Cs) + cofs + ci) * 27 + r * 3 + kx] : 0.f;
        amax = fmaxf(amax, fabsf(v[k]));
    }
    const int e = mx_exp(amax, 1.f / 6.f);
    const float m = exp2i(-e);
    *(uint4 *)(wq + ((size_t)r * Cop + cop) * 32 + blk * 16) = make_uint4(cvt_e2m1x8(v, m), cvt_e2m1x8(v + 8, m), cvt_e2m1x8(v + 16, m), cvt_e2m1x8(v + 24, m));
    ws[i] = (uint8_t)(e + 127);
}
template <int MT, int TZ, typename T, typename TO>
__global__ void __launch_bounds__(256, 2) conv_fwd_f4p_k(const T *__restrict__ x, const uint8_t *__restrict__ wq, const uint8_t *__restrict__ wsc,
                                                      const float *__restrict__ b, TO *__restrict__ y,
                                                      int N, int Ci, int D, int H, int W, int Co, int Cop, gnp_t gp, double *__restrict__ osum, int Go, split_t sp) {
    constexpr int BM = MT * 16, RZ = TZ / 2, NR = 2 * RZ, NROW = (TZ + 2) * 10, TT = NROW * 18;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    uint8_t *sx = smem_raw;                    /* [TT pos][16 B] = [pos | pos + 1] */
    uint8_t *sxs = sx + TT * 16;               /* [TT] pair scales */
    uint8_t *wa = sxs + ((TT + 127) & ~127);   /* [9 r][BM co][32 B] swizzled */
    unsigned short *was = (unsigned short *)(wa + 9 * BM * 32);   /* [9][BM] (byte 0 block 0, byte 1 block 1) */
    chan_t *ctab = (chan_t *)(wa + 9 * BM * 32 + ((9 * BM * 2 + 15) & ~15));   /* [32] */
    unsigned *pam = (unsigned *)(ctab + 32);    /* [TT] per-position amax (bits) */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wz = warp >> 2, wr = (warp & 3) * 2;
    const int ox0 = blockIdx.x * 16, oy0 = blockIdx.y * 8;
    int bz = blockIdx.z;
    const int nzt = (D + TZ - 1) / TZ;
    const int oz0 = (bz % nzt) * TZ; bz /= nzt;
    const int nmt = Cop / BM;
    const int co0 = (bz % nmt) * BM; const int n = bz / nmt;
    const size_t plane = (size_t)D * H * W;
    const bool G = gp.G != 0;
    for (int i = threadIdx.x; i < 9 * BM * 2; i += 256) {
        const int r = i / (BM * 2), q = i % (BM * 2), c = q >> 1, h = q & 1;
        *(uint4 *)(wa + r * BM * 32 + sw16(c, h)) = __ldg((const uint4 *)(wq + ((size_t)r * Cop + co0 + c) * 32 + h * 16));
    }
    for (int i = threadIdx.x; i < 9 * BM; i += 256) { const size_t o = ((size_t)(i / BM) * Cop + co0 + i % BM) * 2; was[i] = (unsigned short)(wsc[o] | (wsc[o + 1] << 8)); }
    if (threadIdx.x < 32) ctab[threadIdx.x] = make_chan(threadIdx.x < 16 ? (int)threadIdx.x : -1, Ci, Ci, n, plane, x, sp, gp, N);
    __syncthreads();
    {   /* staging: thread per position, values kept in registers; pair p = [p | p + 1] is quantised with its own scale
           s_p = amax over both positions (the MX 32-element block of the mma), so every position is quantised twice: low half
           of pair p with s_p, high half of pair p - 1 with s_(p-1) */
        constexpr int NIT = (TT + 255) / 256;
        const chan_t c0 = ctab[0];
        float2 gab[16];
#pragma unroll
        for (int k = 0; k < 16; k++) gab[k] = make_float2(ctab[k].a, ctab[k].b);
        float v[NIT][16];
#pragma unroll
        for (int it = 0; it < NIT; it++) {
            const int pos = threadIdx.x + 256 * it, row = pos / 18, ix = pos - 18 * row;
            const int gz = oz0 - 1 + row / 10, gy = oy0 - 1 + row % 10, gx = ox0 - 1 + ix;
            const bool inb = pos < TT && gz >= 0 && gz < D && gy >= 0 && gy < H && gx >= 0 && gx < W;
            stage_row16<T>(c0, Ci, plane, inb ? ((size_t)gz * H + gy) * W + gx : 0, inb, G, gab, v[it]);
            unsigned am = 0u;
#pragma unroll
            for (int k = 0; k < 16; k++) am = amax_u(am, v[it][k]);
            if (pos < TT) pam[pos] = am;
        }
        __syncthreads();
#pragma unroll
        for (int it = 0; it < NIT; it++) {
            const int pos = threadIdx.x + 256 * it, row = pos / 18, ix = pos - 18 * row;
            if (pos >= TT) break;
            const unsigned a0 = pam[pos], an = ix < 17 ? pam[pos + 1] : 0u, ap = ix ? pam[pos - 1] : 0u;
            const int e = mx_exp(__uint_as_float(max(a0, an)), 1.f / 6.f), ep = mx_exp(__uint_as_float(max(a0, ap)), 1.f / 6.f);
            const float m = exp2i(-e), mp = exp2i(-ep);
            float *w = v[it];
            float wl[16], wh[16];
            if (sp.sr) {   /* gradient operand: exact stochastic rounding keyed by the element (each copy with its own scale) */
                const int gz = oz0 - 1 + row / 10, gy = oy0 - 1 + row % 10, gx = ox0 - 1 + ix;
                const uint64_t vid = (((uint64_t)n * D + gz) * H + gy) * (uint64_t)W + gx;
                uint32_t h0 = 0u;
#pragma unroll
                for (int k = 0; k < 16; k++) { const uint32_t h = (k & 1) ? h0 : (h0 = sr_hash(sp.sr, vid * 8 + k / 2)); const float u = sr_u16(h, k & 1); wl[k] = sr_e2m1_u(w[k] * m, u) / m; wh[k] = sr_e2m1_u(w[k] * mp, u) / mp; }
            } else {
#pragma unroll
                for (int k = 0; k < 16; k++) { wl[k] = w[k]; wh[k] = w[k]; }
            }
            *(uint2 *)(sx + pos * 16) = make_uint2(cvt_e2m1x8(wl, m), cvt_e2m1x8(wl + 8, m));                       /* low half of pair[pos] */
            if (ix) *(uint2 *)(sx + (pos - 1) * 16 + 8) = make_uint2(cvt_e2m1x8(wh, mp), cvt_e2m1x8(wh + 8, mp));   /* high half of pair[pos - 1] */
            if (ix == 17) *(uint2 *)(sx + pos * 16 + 8) = make_uint2(0u, 0u);                                      /* the row's last pair: [17 | 0] */
            sxs[pos] = (uint8_t)(e + 127);
        }
    }
    __syncthreads();
    float acc[MT][NR][2][4];
#pragma unroll
    for (int m = 0; m < MT; m++) for (int r = 0; r < NR; r++) for (int q = 0; q < 2; q++) for (int k = 0; k < 4; k++) acc[m][r][q][k] = 0.f;
    const int mat = lane >> 3, kb = mat & 1, vx = (mat >> 1) * 8 + (lane & 7);
#pragma unroll 3
    for (int tr = 0; tr < 9; tr++) {
        const int kz = tr / 3, ky = tr % 3;
        unsigned bfr[NR][4], sb[NR][2];
#pragma unroll
        for (int r = 0; r < NR; r++) {
            const int rowi = (wz * RZ + (r >> 1) + kz) * 10 + wr + (r & 1) + ky;
            ldsm_x4(bfr[r], sx + (rowi * 18 + vx + 2 * kb) * 16);   /* block 0: [x | x+1], block 1: [x+2 | x+3] */
#pragma unroll
            for (int q2 = 0; q2 < 2; q2++) { const int pc = rowi * 18 + q2 * 8 + g; sb[r][q2] = (unsigned)sxs[pc] | ((unsigned)sxs[pc + 2] << 8); }   /* column g: pair scales */
        }
#pragma unroll
        for (int m = 0; m < MT; m++) {
            unsigned af[4];
            const int arow = m * 16 + (mat & 1) * 8 + (lane & 7);
            ldsm_x4(af, wa + tr * BM * 32 + sw16(arow, mat >> 1));
            const unsigned sa = was[tr * BM + m * 16 + g + 8 * (t & 1)];
#pragma unroll
            for (int r = 0; r < NR; r++) { mma_f4(acc[m][r][0], af, bfr[r], sa, sb[r][0]); mma_f4(acc[m][r][1], af, bfr[r] + 2, sa, sb[r][1]); }
        }
    }
    fwd_epilogue<MT, NR, TO>(acc, smem_raw, y, b, n, co0, Co, D, H, W, oz0, oy0, ox0, wz, wr, osum, Go, sp, N);
}
template <int MT, int TZ, typename T, typename TO> static void launch_f4p(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp) {
    constexpr int TT = (TZ + 2) * 180, BM = MT * 16, NROW = (TZ + 2) * 10;
    size_t smem = (size_t)TT * 16 + ((TT + 127) & ~127) + 9 * BM * 32 + ((9 * BM * 2 + 15) & ~15) + 32 * sizeof(chan_t) + TT * 4;
    if (smem < 8 * 256 * sizeof(float)) smem = 8 * 256 * sizeof(float);
    static int attr[8];
    if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)conv_fwd_f4p_k<MT, TZ, T, TO>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
    conv_fwd_f4p_k<MT, TZ, T, TO><<<grid, 256, smem>>>((const T *)x, wq, ws, b, (TO *)y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, gp, osum, Go, sp);
}
static int wmemo_get(unsigned key, const float *w, int Cop, int Cip, int Cx, int CxP, int Ox, int OxP, size_t nq, size_t ns, uint8_t **wq, uint8_t **ws, int slot = 2);
template <typename T, typename TO> static void p16_f4(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, int Cop, int Ox, int OxP, gnp_t gp, double *osum, int Go, split_t sp) {
    const size_t nq = (size_t)9 * Cop * 32, ns = (size_t)9 * Cop * 2;
    const int pOx = IS_MX(TO) && sp.y2 ? Ox : -1;
    uint8_t *wq, *ws;
    if (!wmemo_get(sp.wkey, w, Cop, -16, -1, 0, pOx, OxP, nq, ns, &wq, &ws)) prep_w4p_k<<<nblk_(ns, 128), 128>>>(w, wq, ws, cout, xs.c, Cop, pOx, OxP);   /* Cip -16: packed layout */
    int MT = Cop % 64 == 0 ? 4 : Cop % 32 == 0 ? 2 : !IS_MX(TO) && Cop == 48 ? 3 : 1;   /* 48 plane-major outputs (dec0.c1 backward-data): one tile, the input staged once */
    const int nmt = Cop / (MT * 16);
    static int tz_env = -1;
    if (tz_env < 0) tz_env = getenv("UFSM_F4_TZ") ? atoi(getenv("UFSM_F4_TZ")) : 0;
    int TZ = tz_env ? tz_env : (MT <= 2 && xs.d >= 16 ? 4 : 2);
    if (MT == 4) TZ = 2;
    dim3 grid(nblk_(xs.w, 16), nblk_(xs.h, 8), (unsigned)(nblk_(xs.d, TZ) * nmt * xs.n));
    switch (MT * 10 + TZ) {
    case 12: launch_f4p<1, 2, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    case 14: launch_f4p<1, 4, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    case 22: launch_f4p<2, 2, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    case 24: launch_f4p<2, 4, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    case 32: launch_f4p<3, 2, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    case 34: launch_f4p<3, 4, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    default: launch_f4p<4, 2, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    }
}
template <typename T, typename TO> static void fwd_f4_t(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp) {
    int Cop = (cout + 15) / 16 * 16, Cip = (xs.c + 31) / 32 * 32;
    int Cx = sp.x2 ? sp.c_split : xs.c, CxP = (Cx + 31) / 32 * 32;
    int Ox = sp.y2 ? sp.o_split : cout, OxP = (Ox + 31) / 32 * 32;
    if (IS_MX(TO)) { if (sp.y2) Cop = OxP + (cout - Ox + 31) / 32 * 32; else if (cout > 16) Cop = (cout + 31) / 32 * 32; }   /* 32-row-aligned output blocks */
    if (IS_MX(T)) Cip = CxP + (xs.c - Cx + 31) / 32 * 32;                                                                /* per-tensor 32-channel segments */
    if (xs.c <= 16 && !sp.x2 && !sp.up && sp.gp2.G == 0) { p16_f4<T, TO>(x, xs, w, b, cout, y, Cop, Ox, OxP, gp, osum, Go, sp); return; }   /* K = [kx0 | kx1 | kx2 | 0] */
    const int nch = Cip / 32;
    const size_t nq = (size_t)28 * Cop * Cip / 2, ns = (size_t)28 * Cop * nch;
    const int pCx = IS_MX(T) ? Cx : -1, pOx = IS_MX(TO) && sp.y2 ? Ox : -1;
    uint8_t *wq, *ws;
    if (g_w2d < 0) g_w2d = getenv("UFSM_W4_2D") ? atoi(getenv("UFSM_W4_2D")) : 0;
    const int w2d = g_w2d;
    if (!wmemo_get(sp.wkey, w, Cop, Cip, pCx, CxP, pOx, OxP, nq, ns, &wq, &ws)) {
        if (w2d) prep_w4_2d_k<<<nblk_((size_t)28 * ((Cop + 31) / 32) * nch * 32, 128), 128>>>(w, wq, ws, cout, xs.c, Cop, Cip, pCx, CxP, pOx, OxP);
        else prep_w4_k<<<nblk_(ns, 128), 128>>>(w, wq, ws, cout, xs.c, Cop, Cip, pCx, CxP, pOx, OxP);
    }
    int MT = Cop % 64 == 0 ? 4 : Cop % 32 == 0 ? 2 : 1;   /* BM = 16 MT must divide Cop (Cop = 48 -> MT = 1) */
    int nmt = Cop / (MT * 16);
    static int tz_env = -1;
    if (tz_env < 0) tz_env = getenv("UFSM_F4_TZ") ? atoi(getenv("UFSM_F4_TZ")) : 0;
    int TZ = tz_env ? tz_env : (MT <= 2 && xs.d >= 16 ? 4 : 2);
    if (MT == 4) TZ = 2;
    dim3 grid(nblk_(xs.w, 16), nblk_(xs.h, 8), (unsigned)(nblk_(xs.d, TZ) * nmt * xs.n));
    switch (MT * 10 + TZ) {
    case 12: launch_f4<1, 2, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    case 14: launch_f4<1, 4, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    case 22: launch_f4<2, 2, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    case 24: launch_f4<2, 4, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    default: launch_f4<4, 2, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    }
}
/* xbf / ybf: 0 fp32, 1 bf16, 2 fp16, 3 MX-fp8, 4 MX-fp4; instantiated: equal types, and mx8 in -> fp16 / bf16 out (backward-data) */
extern "C" int lp_conv_fwd_f4(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, gnp_t gp, double *osum, int Go, split_t sp) {
    if (xs.c <= 8) return lp_conv_fwd_f8(x, xbf, xs, w, b, cout, y, ybf, gp, osum, Go, sp);   /* the network input (enc0.c1): fp8 tap-packed kernel */
    if (sp.accum && (xbf >= 3 || ybf >= 3)) { fprintf(stderr, "lp_conv_fwd_f4: MX storage with accumulate is not supported\n"); abort(); }
    if (xbf == ybf) {
        if (xbf == 4) fwd_f4_t<mx4_t, mx4_t>(x, xs, w, b, cout, y, gp, osum, Go, sp);
        else if (xbf == 3) fwd_f4_t<mx8_t, mx8_t>(x, xs, w, b, cout, y, gp, osum, Go, sp);
        else if (xbf == 2) fwd_f4_t<__half, __half>(x, xs, w, b, cout, y, gp, osum, Go, sp);
        else if (xbf) fwd_f4_t<bf16, bf16>(x, xs, w, b, cout, y, gp, osum, Go, sp);
        else fwd_f4_t<float, float>(x, xs, w, b, cout, y, gp, osum, Go, sp);
    } else if (xbf == 4 && ybf == 0) fwd_f4_t<mx4_t, float>(x, xs, w, b, cout, y, gp, osum, Go, sp);   /* tests: exact staging check with an unquantised output */
    else if (xbf == 3 && ybf == 2) fwd_f4_t<mx8_t, __half>(x, xs, w, b, cout, y, gp, osum, Go, sp);
    else if (xbf == 3 && ybf == 1) fwd_f4_t<mx8_t, bf16>(x, xs, w, b, cout, y, gp, osum, Go, sp);
    else { fprintf(stderr, "lp_conv_fwd_f4: activation types (x %d, y %d) are not instantiated\n", xbf, ybf); abort(); }
    LPCK();
    return 0;
}

/* ======================= FP8 stride-2 forward (k=3, pad 1) =======================
   Output tile 2 z x 4 y x 8 x per block; warp w owns z = w/4, row w%4 and the single 8-voxel n-tile. The input tile
   (5 x 9 x 17 positions x 32 channels) is stored with each row parity-split (even x first: 9 entries, then odd x: 8),
   so the stride-2 voxel columns of every tap are consecutive positions and the 16-byte XOR swizzle keeps ldmatrix
   conflict-free. Scales as in the stride-1 kernel: per (position, 32 channels) for the input, per (tap, co, chunk)
   for the weights (same prep). */
#define S2F8_T 765   /* 5 * 9 * 17 */
/* 8 e2m1 nibbles (channel k = nibble k) -> 8 e4m3 bytes (exact: the e2m1 grid {0, .5, 1, 1.5, 2, 3, 4, 6} is a subset of e4m3) */
__device__ __forceinline__ unsigned e2m1_e4m3(unsigned nb) { return (__byte_perm(0x3C383000u, 0x4C484440u, nb & 7u) & 0xffu) | ((nb & 8u) << 4); }
__device__ __forceinline__ uint2 nib8_e4m3(unsigned w) {
    unsigned lo = 0u, hi = 0u;
#pragma unroll
    for (int k = 0; k < 4; k++) { lo |= e2m1_e4m3(w >> (4 * k)) << (8 * k); hi |= e2m1_e4m3(w >> (16 + 4 * k)) << (8 * k); }
    return make_uint2(lo, hi);
}
__device__ __forceinline__ int s2pos(int row, int x) { return row * 17 + ((x & 1) ? 9 + (x >> 1) : (x >> 1)); }
template <int MT, typename T, bool P16 = false>   /* P16: <= 16 input channels, two taps per K block ([kx0 | kx1], [kx2 | 0]) */
__global__ void __launch_bounds__(256, 2) conv_fwd_s2_f8_k(const T *__restrict__ x, const uint8_t *__restrict__ wq, const uint8_t *__restrict__ wsc,
                                                        const float *__restrict__ b, T *__restrict__ y,
                                                        int N, int Ci, int D, int H, int W, int Co, int Cop, int Cip, int Do, int Ho, int Wo, gnp_t gp) {
    constexpr int BM = MT * 16, TG = 9;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    uint8_t *sx = smem_raw;                         /* [S2F8_T pos][32 ci] swizzled */
    uint8_t *sxs = sx + S2F8_T * 32;                /* [S2F8_T] (768) */
    uint8_t *wa = sxs + 768;                        /* [TG][BM][32] swizzled */
    uint8_t *was = wa + TG * BM * 32;               /* [TG][BM] */
    chan_t *ctab = (chan_t *)(was + ((TG * BM + 15) & ~15));   /* [32] (input transform path) */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wz = warp >> 2, wr = warp & 3;
    const int ox0 = blockIdx.x * 8, oy0 = blockIdx.y * 4;
    const bool G = gp.G != 0;
    const split_t nsp = {};
    int bz = blockIdx.z;
    const int nzt = (Do + 1) / 2;
    const int oz0 = (bz % nzt) * 2; bz /= nzt;
    const int nmt = Cop / BM;
    const int co0 = (bz % nmt) * BM; const int n = bz / nmt;
    const int nch = Cip / 32;
    float acc[MT][4];
#pragma unroll
    for (int m = 0; m < MT; m++) for (int k = 0; k < 4; k++) acc[m][k] = 0.f;
    const size_t plane = (size_t)D * H * W;
    if constexpr (P16) {   /* staged [5 x 9 rows][17 parity-split positions][16 B], one e4m3 scale per row; weights as prep_w8p_k */
        uint8_t *px = smem_raw;                        /* [S2F8_T][16] */
        uint8_t *zrow = px + S2F8_T * 16;              /* [16] zeros */
        uint8_t *rs = zrow + 16;                       /* [45] row scales (64) */
        unsigned *ram = (unsigned *)(rs + 64);         /* [45] row amax (64) */
        uint8_t *pa = (uint8_t *)(ram + 64);           /* [18][BM][32] swizzled */
        uint8_t *pas = pa + 18 * BM * 32;              /* [18][BM] */
        chan_t *pc = (chan_t *)(pas + ((18 * BM + 15) & ~15));   /* [16] */
        for (int i = threadIdx.x; i < 18 * BM * 2; i += 256) {
            const int ks = i / (BM * 2), r = i % (BM * 2), c = r >> 1, h = r & 1;
            *(uint4 *)(pa + ks * BM * 32 + sw16(c, h)) = __ldg((const uint4 *)(wq + ((size_t)ks * Cop + co0 + c) * 32 + h * 16));
        }
        for (int i = threadIdx.x; i < 18 * BM; i += 256) pas[i] = wsc[(size_t)(i / BM) * Cop + co0 + i % BM];
        if (threadIdx.x < 16) pc[threadIdx.x] = make_chan((int)threadIdx.x, Ci, Ci, n, plane, x, nsp, gp, N);
        if (threadIdx.x < 4) ((unsigned *)zrow)[threadIdx.x] = 0u;
        if (threadIdx.x < 64) ram[threadIdx.x] = 0u;
        __syncthreads();
        const chan_t c0 = pc[0];
        float2 gab[16];
#pragma unroll
        for (int k = 0; k < 16; k++) gab[k] = make_float2(pc[k].a, pc[k].b);
        float v[3][16];
#pragma unroll
        for (int it = 0; it < 3; it++) {
            const int p = threadIdx.x + 256 * it, row = p / 17, c = p - row * 17, ix = c < 9 ? 2 * c : 2 * (c - 9) + 1;
            const int iz = row / 9, iy = row - iz * 9, gz = 2 * oz0 - 1 + iz, gyy = 2 * oy0 - 1 + iy, gx = 2 * ox0 - 1 + ix;
            const bool inb = p < S2F8_T && gz >= 0 && gz < D && gyy >= 0 && gyy < H && gx >= 0 && gx < W;
            stage_row16<T>(c0, Ci, plane, inb ? ((size_t)gz * H + gyy) * W + gx : 0, inb, G, gab, v[it]);
            unsigned am = 0u;
#pragma unroll
            for (int k = 0; k < 16; k++) { const unsigned u = __float_as_uint(fabsf(v[it][k])); am = u <= 0x7f800000u ? max(am, u) : am; }
            if (p < S2F8_T && am) atomicMax(&ram[row], am);
        }
        __syncthreads();
#pragma unroll
        for (int it = 0; it < 3; it++) {
            const int p = threadIdx.x + 256 * it, row = p / 17, c = p - row * 17;
            if (p >= S2F8_T) break;
            const int e = mx_exp(__uint_as_float(ram[row]), 1.f / 448.f);
            const float m = exp2i(-e);
            const float *w = v[it];
            *(uint4 *)(px + p * 16) = make_uint4(cvt_e4m3x4(w[0] * m, w[1] * m, w[2] * m, w[3] * m), cvt_e4m3x4(w[4] * m, w[5] * m, w[6] * m, w[7] * m),
                                                 cvt_e4m3x4(w[8] * m, w[9] * m, w[10] * m, w[11] * m), cvt_e4m3x4(w[12] * m, w[13] * m, w[14] * m, w[15] * m));
            if (c == 0) rs[row] = (uint8_t)(e + 127);
        }
        __syncthreads();
        const int mat = lane >> 3, kh = mat & 1, vx = lane & 7;
#pragma unroll 3
        for (int tr = 0; tr < 9; tr++) {
            const int kz = tr / 3, ky = tr % 3, row = (wz * 2 + kz) * 9 + wr * 2 + ky;
            unsigned bA[4], bB[4];
            ldsm_x4(bA, px + (row * 17 + (kh ? 9 : 0) + vx) * 16);          /* [x 2j | x 2j+1] = [E j | O j] */
            ldsm_x4(bB, kh ? zrow : px + (row * 17 + vx + 1) * 16);         /* [x 2j+2 | 0] = [E j+1 | 0] */
            const unsigned sb = rs[row];
#pragma unroll
            for (int m = 0; m < MT; m++) {
                const int ar = m * 16 + (mat & 1) * 8 + (lane & 7);
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    unsigned af[4];
                    ldsm_x4(af, pa + (2 * tr + h) * BM * 32 + sw16(ar, mat >> 1));
                    mma_f8(acc[m], af, h ? bB : bA, pas[(2 * tr + h) * BM + m * 16 + g + 8 * (t & 1)], sb);
                }
            }
        }
    } else
    for (int ci0 = 0; ci0 < Cip; ci0 += 32) {
        __syncthreads();
        if (G) { if (threadIdx.x < 32) ctab[threadIdx.x] = make_chan(ci0 + threadIdx.x, Ci, Ci, n, plane, x, nsp, gp, N); __syncthreads(); }
        for (int p = threadIdx.x; p < S2F8_T; p += 256) {   /* p = row * 17 + parity-split column */
            int row = p / 17, c = p - row * 17, ix = c < 9 ? 2 * c : 2 * (c - 9) + 1;
            int iz = row / 9, iy = row - iz * 9;
            int gz = 2 * oz0 - 1 + iz, gyy = 2 * oy0 - 1 + iy, gx = 2 * ox0 - 1 + ix;
            bool inb = gz >= 0 && gz < D && gyy >= 0 && gyy < H && gx >= 0 && gx < W;
            size_t off = ((size_t)gz * H + gyy) * W + gx;
            if (IS_MX(T) && !G) {   /* MX input: the 32-channel chunk is one stored block -> copy (mx4: nibbles -> e4m3 bytes, exact) */
                const int bw = mx_bw(Ci), nb = mx_nb(Ci), blk = ci0 / 32, rb = mx_rb(bw, MX_BITS(T));
                uint4 h0 = make_uint4(0u, 0u, 0u, 0u), h1 = h0;
                unsigned sc = 1u;
                if (inb) {
                    const uint8_t *q = (const uint8_t *)x;
                    const size_t ri = ((size_t)n * nb + blk) * plane + off;
                    if constexpr (IS_MX8(T)) { const uint4 *src = (const uint4 *)(q + ri * bw); h0 = __ldg(src); if (bw == 32) h1 = __ldg(src + 1); }
                    else {
                        const uint2 *src = (const uint2 *)(q + ri * rb);
                        const uint2 lo = __ldg(src), hi = bw == 32 ? __ldg(src + 1) : make_uint2(0u, 0u);
                        const uint2 a = nib8_e4m3(lo.x), b2 = nib8_e4m3(lo.y), c = nib8_e4m3(hi.x), d = nib8_e4m3(hi.y);
                        h0 = make_uint4(a.x, a.y, b2.x, b2.y); h1 = make_uint4(c.x, c.y, d.x, d.y);
                    }
                    sc = q[(size_t)N * nb * plane * rb + ri];
                }
                *(uint4 *)(sx + sw16(p, 0)) = h0;
                *(uint4 *)(sx + sw16(p, 1)) = h1;
                sxs[p] = (uint8_t)sc;
                continue;
            }
            float v[32], amax = 0.f;
            if (IS_MX(T)) mx_row32<IS_MX(T) ? MX_BITS(T) : 8>(ctab[0], inb ? off : 0, inb, v);   /* (G here: the copy path took !G) */
#pragma unroll
            for (int k = 0; k < 32; k++) {
                int ci = ci0 + k;
                if (IS_MX(T)) { const chan_t &c = ctab[k]; v[k] = inb && c.p ? act_ab(v[k], c.a, c.b, true) : 0.f; }
                else if (G) { const chan_t c = ctab[k]; v[k] = inb && c.p ? act_ab(ldc<T>(c, off), c.a, c.b, true) : 0.f; }
                else v[k] = inb && ci < Ci ? ldx(x + ((size_t)n * Ci + ci) * plane, off) : 0.f;
                amax = fmaxf(amax, fabsf(v[k]));
            }
            int e = mx_exp(amax, 1.f / 448.f);
            float m = exp2i(-e);
            *(uint4 *)(sx + sw16(p, 0)) = make_uint4(cvt_e4m3x4(v[0] * m, v[1] * m, v[2] * m, v[3] * m), cvt_e4m3x4(v[4] * m, v[5] * m, v[6] * m, v[7] * m),
                                                     cvt_e4m3x4(v[8] * m, v[9] * m, v[10] * m, v[11] * m), cvt_e4m3x4(v[12] * m, v[13] * m, v[14] * m, v[15] * m));
            *(uint4 *)(sx + sw16(p, 1)) = make_uint4(cvt_e4m3x4(v[16] * m, v[17] * m, v[18] * m, v[19] * m), cvt_e4m3x4(v[20] * m, v[21] * m, v[22] * m, v[23] * m),
                                                     cvt_e4m3x4(v[24] * m, v[25] * m, v[26] * m, v[27] * m), cvt_e4m3x4(v[28] * m, v[29] * m, v[30] * m, v[31] * m));
            sxs[p] = (uint8_t)(e + 127);
        }
        for (int t0 = 0; t0 < 27; t0 += TG) {
            if (t0) __syncthreads();
            for (int i = threadIdx.x; i < TG * BM * 2; i += 256) {
                int tt = i / (BM * 2), r = i % (BM * 2), c = r >> 1, h = r & 1;
                *(uint4 *)(wa + tt * BM * 32 + sw16(c, h)) = __ldg((const uint4 *)(wq + ((size_t)(t0 + tt) * Cop + co0 + c) * Cip + ci0 + h * 16));
            }
            for (int i = threadIdx.x; i < TG * BM; i += 256) { int tt = i / BM, c = i % BM; was[i] = wsc[((size_t)(t0 + tt) * Cop + co0 + c) * nch + ci0 / 32]; }
            __syncthreads();
#pragma unroll
            for (int tt = 0; tt < TG; tt++) {
                int tap = t0 + tt, kz = tap / 9, ky = (tap / 3) % 3, kx = tap % 3;
                int row = (wz * 2 + kz) * 9 + wr * 2 + ky;
                unsigned bfr[4];
                {   /* x4: (k half 0, voxels 0..7), (k half 1, voxels 0..7), duplicated (only b0, b1 used) */
                    int mat = lane >> 3, kh = mat & 1, vx = lane & 7;
                    ldsm_x4(bfr, sx + sw16(s2pos(row, 2 * vx + kx), kh));
                }
                unsigned sb = sxs[s2pos(row, 2 * g + kx)];
#pragma unroll
                for (int m = 0; m < MT; m++) {
                    unsigned af[4];
                    int mat = lane >> 3, r = m * 16 + (mat & 1) * 8 + (lane & 7);
                    ldsm_x4(af, wa + tt * BM * 32 + sw16(r, mat >> 1));
                    unsigned sa = was[tt * BM + m * 16 + g + 8 * (t & 1)];
                    mma_f8(acc[m], af, bfr, sa, sb);
                }
            }
        }
    }
    if constexpr (IS_MX(T)) {   /* MX output (blocks of 32 / 16 channels per voxel); fp4: nibble pairs across lanes g, g + 1 as fwd_epilogue */
        constexpr int B = MX_BITS(T);
        const int bw = mx_bw(Co), nb = mx_nb(Co), rb = mx_rb(bw, B);
        const size_t So = (size_t)Do * Ho * Wo;
        const int oz = oz0 + wz, oy = oy0 + wr;
        uint8_t *q = (uint8_t *)y, *scp = q + (size_t)N * nb * So * rb;
#pragma unroll
        for (int vv = 0; vv < 2; vv++) {
            const int ox = ox0 + 2 * t + vv;
            const bool ok = oz < Do && oy < Ho && ox < Wo;
#pragma unroll
            for (int j = 0; j < MT; j++) {
                if (j * 16 % bw) continue;
                float val[2][2], am = 0.f;
#pragma unroll
                for (int mm = 0; mm < 2; mm++)
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const int m = j + mm, co = co0 + m * 16 + g + 8 * h;
                        float v = 0.f;
                        if (mm < bw / 16 && m < MT) v = acc[m < MT ? m : 0][2 * h + vv] + (b && co < Co ? b[co] : 0.f);
                        val[mm][h] = v; am = fmaxf(am, fabsf(v));
                    }
                am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, 4)); am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, 8)); am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, 16));
                const int e = mx_exp(am, mxf<B>::inv_qmax);
                const float mult = exp2i(-e);
                unsigned nib = 0u;
                if constexpr (B == 4) {
#pragma unroll
                    for (int mm = 0; mm < 2; mm++)
#pragma unroll
                        for (int h = 0; h < 2; h++) nib |= (unsigned)(cvt_e2m1x2(val[mm][h] * mult, 0.f) & 15) << (8 * (2 * mm + h));
                    nib |= __shfl_down_sync(0xffffffff, nib, 4) << 4;   /* channel g + 1 into the high nibbles */
                }
                if (ok) {
                    const int blk = (co0 + j * 16) / bw;
                    const size_t v = ((size_t)oz * Ho + oy) * Wo + ox;
                    uint8_t *dst = q + (((size_t)n * nb + blk) * So + v) * rb;
#pragma unroll
                    for (int mm = 0; mm < 2; mm++)
#pragma unroll
                        for (int h = 0; h < 2; h++) if (mm < bw / 16) {
                            if constexpr (B == 8) dst[mm * 16 + g + 8 * h] = cvt_e4m3(val[mm][h] * mult);
                            else if (!(g & 1)) dst[(mm * 16 + g + 8 * h) >> 1] = (uint8_t)(nib >> (8 * (2 * mm + h)));
                        }
                    if (g == 0) scp[((size_t)n * nb + blk) * So + v] = (uint8_t)(e + 127);
                }
            }
        }
        return;
    }
#pragma unroll
    for (int m = 0; m < MT; m++) {
        int oz = oz0 + wz, oy = oy0 + wr;
        if (oz >= Do || oy >= Ho) continue;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            int co = co0 + m * 16 + g + 8 * h;
            if (co >= Co) continue;
            float bias = b ? b[co] : 0.f;
            T *yp = y + (((size_t)n * Co + co) * Do + oz) * Ho * Wo + (size_t)oy * Wo;
            int ox = ox0 + 2 * t;
            float v0 = acc[m][2 * h] + bias, v1 = acc[m][2 * h + 1] + bias;
            if (!(Wo & 1) && ox + 1 < Wo) stx2(yp, (size_t)ox, v0, v1);
            else { if (ox < Wo) stx(yp, (size_t)ox, v0); if (ox + 1 < Wo) stx(yp, (size_t)ox + 1, v1); }
        }
    }
}
template <int MT, typename T, bool P16 = false> static void launch_s2f8(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, int Cip, shape5 ys, gnp_t gp) {
    size_t smem = (size_t)S2F8_T * 32 + 768 + 9 * MT * 16 * 32 + ((9 * MT * 16 + 15) & ~15) + 32 * sizeof(chan_t);
    static int attr[8];
    if (P16) { const size_t sp16 = (size_t)S2F8_T * 16 + 16 + 64 + 256 + 18 * MT * 16 * 33 + 16 + 16 * sizeof(chan_t); if (sp16 > smem) smem = sp16; }
    if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)conv_fwd_s2_f8_k<MT, T, P16>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
    conv_fwd_s2_f8_k<MT, T, P16><<<grid, 256, smem>>>((const T *)x, wq, ws, b, (T *)y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, Cip, ys.d, ys.h, ys.w, gp);
}
extern "C" int lp_conv_fwd_s2_f8(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, shape5 ys, gnp_t gp) {
    lp_dtype_check("lp_conv_fwd_s2_f8", xbf, ybf);
    int Cop = (cout + 15) / 16 * 16, Cip = (xs.c + 31) / 32 * 32, nch = Cip / 32;
    if (xbf >= 3 && cout > 16) Cop = (cout + 31) / 32 * 32;   /* MX output: 32-row-aligned m-tiles */
    static int p16 = -1;
    if (p16 < 0) p16 = getenv("UFSM_S2_PACK16") ? atoi(getenv("UFSM_S2_PACK16")) : 1;
    if (p16 && xs.c <= 16) {   /* down0: two taps per K block */
        uint8_t *wq = lp_buf<uint8_t>(0, (size_t)P16_KS * Cop * 32), *ws = lp_buf<uint8_t>(1, (size_t)P16_KS * Cop);
        prep_w8p_k<<<nblk_((size_t)P16_KS * Cop, 128), 128>>>(w, wq, ws, cout, xs.c, Cop, -1, 0);
        const int MT = Cop % 64 == 0 ? 4 : Cop % 32 == 0 ? 2 : 1, nmt = Cop / (MT * 16);
        dim3 grid(nblk_(ys.w, 8), nblk_(ys.h, 4), (unsigned)(nblk_(ys.d, 2) * nmt * xs.n));
#define S2P(MT_) do { if (xbf == 4) launch_s2f8<MT_, mx4_t, true>(grid, x, xs, wq, ws, b, cout, y, Cop, 32, ys, gp); else if (xbf == 3) launch_s2f8<MT_, mx8_t, true>(grid, x, xs, wq, ws, b, cout, y, Cop, 32, ys, gp); else if (xbf == 2) launch_s2f8<MT_, __half, true>(grid, x, xs, wq, ws, b, cout, y, Cop, 32, ys, gp); else if (xbf) launch_s2f8<MT_, bf16, true>(grid, x, xs, wq, ws, b, cout, y, Cop, 32, ys, gp); else launch_s2f8<MT_, float, true>(grid, x, xs, wq, ws, b, cout, y, Cop, 32, ys, gp); } while (0)
        switch (MT) { case 1: S2P(1); break; case 2: S2P(2); break; default: S2P(4); break; }
#undef S2P
        LPCK();
        return 0;
    }
    uint8_t *wq = lp_buf<uint8_t>(0, (size_t)27 * Cop * Cip), *ws = lp_buf<uint8_t>(1, (size_t)27 * Cop * nch);
    size_t nt = (size_t)27 * Cop * nch;
    prep_w8_k<<<nblk_(nt, 128), 128>>>(w, wq, ws, cout, xs.c, Cop, Cip);
    int MT = Cop % 64 == 0 ? 4 : Cop % 32 == 0 ? 2 : 1;
    int nmt = Cop / (MT * 16);
    dim3 grid(nblk_(ys.w, 8), nblk_(ys.h, 4), (unsigned)(nblk_(ys.d, 2) * nmt * xs.n));
#define S2L(MT_) do { if (xbf == 4) launch_s2f8<MT_, mx4_t>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, ys, gp); else if (xbf == 3) launch_s2f8<MT_, mx8_t>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, ys, gp); else if (xbf == 2) launch_s2f8<MT_, __half>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, ys, gp); else if (xbf) launch_s2f8<MT_, bf16>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, ys, gp); else launch_s2f8<MT_, float>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, ys, gp); } while (0)
    switch (MT) { case 1: S2L(1); break; case 2: S2L(2); break; default: S2L(4); break; }
#undef S2L
    LPCK();
    return 0;
}

/* ======================= FP8 stride-2 weight gradient (k=3, pad 1) =======================
   GW[co][ci][tap] = sum_o GY[co][o] X[ci][2 o + k - 1]. Same block / warp structure as conv_bwd_w_f8_k (9 warps, warp
   = (kz, ky) with all three kx; M = 16 MT output channels, N = 8 NT input channels, K = 32 output voxels = two output
   rows of one output plane), z-steps of 2 output planes x 8 rows x 16 columns. The input slab for a step is 5 planes x
   17 rows x 33 positions per channel; each row is stored parity-split (even positions E at bytes 0..16, odd O at
   20..35), so for output voxels 4t..4t+3 the tap kx = 0 / 1 / 2 operand is E[4t..] / O[4t..] / E[4t+1..] (one funnel
   shift). Scales: X per (channel, input plane) (both rows of a K block read one input plane for every tap), GY per
   (cout, 32 voxels). No GroupNorm input transform (the down convs read stored activations). */
#define XS2_CS 3088   /* per channel: 5 planes x 17 rows x 36 B = 3060, padded to 772 words == 4 mod 32 */
#define XS2_PS 612
template <int MT, int NT, typename T, typename TG>
__global__ void __launch_bounds__(288, 2) conv_bwd_w_s2_f8_k(const T *__restrict__ x, const TG *__restrict__ gy, float *__restrict__ gw, float *__restrict__ gb,
                                                          int N, int Ci, int D, int H, int W, int Co, int Do, int Ho, int Wo, int ZC, gnp_t gp) {
    constexpr int CH = 8 * NT, BMo = 16 * MT;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    uint8_t *sxq = smem_raw;                        /* [CH][XS2_CS] */
    uint8_t *sg = sxq + CH * XS2_CS;                /* [BMo][G8_CS] */
    uint8_t *sgs = sg + BMo * G8_CS;                /* [BMo][8 ksteps] */
    uint8_t *sxs = sgs + BMo * 8;                   /* [CH][5 planes] (8 reserved) */
    float *sbias = (float *)(sxs + CH * 8);         /* [BMo] */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int ci0 = blockIdx.x * CH, co0 = blockIdx.y * BMo;
    int bz = blockIdx.z;
    const int nxt = (Wo + 15) / 16, nyt = (Ho + 7) / 8, nzt = (Do + 1) / 2;
    const int ox0 = (bz % nxt) * 16; bz /= nxt;
    const int oy0 = (bz % nyt) * 8; bz /= nyt;
    const int nzc = (nzt + ZC - 1) / ZC;
    const int zc = bz % nzc; const int n = bz / nzc;
    const size_t plane = (size_t)D * H * W;
    const int kz = warp / 3, ky = warp % 3;
    const bool do_bias = gb && blockIdx.x == 0;
    const bool vec = (Wo & 3) == 0, vec8 = (W & 7) == 0;
    float acc[3][MT][NT][4];
#pragma unroll
    for (int a = 0; a < 3; a++) for (int m = 0; m < MT; m++) for (int q = 0; q < NT; q++) for (int k = 0; k < 4; k++) acc[a][m][q][k] = 0.f;
    if (threadIdx.x < BMo) sbias[threadIdx.x] = 0.f;
    for (int zt = zc * ZC; zt < nzt && zt < (zc + 1) * ZC; zt++) {
        const int oz0 = zt * 2;
        __syncthreads();
        /* X: warp per (channel, input plane). Row = halo position u = 0 (x = 2 ox0 - 1) + 4 vectors of 8 positions
           u = 8j + 1 .. 8j + 8 (x = 2 ox0 + 8j ..): the 4 odd u of a vector are one aligned word O[4j..4j+3], the 4 even
           u are bytes E[4j+1..4j+4]. Lane item f = lane + 32 i < 68 -> row f / 4, vector f % 4; lanes < 17 do the halo. */
        for (int task = warp; task < CH * 5; task += 9) {
            int k = task / 5, iz = task - 5 * k, ci = ci0 + k, gz = 2 * oz0 - 1 + iz;
            const bool ok = ci < Ci && gz >= 0 && gz < D;
            const T *xc = IS_MX(T) ? x : x + ((size_t)n * Ci + (ok ? ci : 0)) * plane + (size_t)(ok ? gz : 0) * H * W;
            const split_t nsp = {};
            const chan_t cm = make_chan(ok ? ci : Ci, Ci, Ci, n, plane, x, nsp, gp, N);   /* MX element access, gn+silu coefficients */
            float v[3][8], vh = 0.f, amax = 0.f;
#pragma unroll
            for (int i = 0; i < 3; i++) {
                int f = lane + 32 * i, iy = f >> 2, j = f & 3, gyy = 2 * oy0 - 1 + iy, gx = 2 * ox0 + 8 * j;
#pragma unroll
                for (int e = 0; e < 8; e++) v[i][e] = 0.f;
                if (ok && f < 68 && gyy >= 0 && gyy < H) {
                    const T *src = xc + (size_t)gyy * W + gx;
                    if constexpr (IS_MX(T)) { const size_t vo = ((size_t)gz * H + gyy) * W + gx; for (int e = 0; e < 8; e++) if (gx + e < W) v[i][e] = ldc<T>(cm, vo + e); }
                    else if (vec8 && gx + 8 <= W) { float tmp[8]; ld8x<T>(src, tmp); for (int e = 0; e < 8; e++) v[i][e] = tmp[e]; }
                    else { for (int e = 0; e < 8; e++) if (gx + e < W) v[i][e] = ldx(src, e); }
                    if (cm.g) for (int e = 0; e < 8; e++) if (gx + e < W) v[i][e] = act_ab(v[i][e], cm.a, cm.b, true);
                }
#pragma unroll
                for (int e = 0; e < 8; e++) amax = fmaxf(amax, fabsf(v[i][e]));
            }
            {
                int gyy = 2 * oy0 - 1 + lane, gx = 2 * ox0 - 1;
                if (ok && lane < 17 && gyy >= 0 && gyy < H && gx >= 0) vh = act_ab(IS_MX(T) ? ldc<T>(cm, ((size_t)gz * H + gyy) * W + gx) : ldx(xc, (size_t)gyy * W + gx), cm.a, cm.b, cm.g);
                amax = fmaxf(amax, fabsf(vh));
            }
#pragma unroll
            for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
            int e = mx_exp(amax, 1.f / 448.f);
            float m = exp2i(-e);
            uint8_t *dst = sxq + k * XS2_CS + iz * XS2_PS;
#pragma unroll
            for (int i = 0; i < 3; i++) {
                int f = lane + 32 * i, iy = f >> 2, j = f & 3;
                if (f < 68) {
                    uint8_t *r = dst + iy * 36;
                    *(unsigned *)(r + 20 + 4 * j) = cvt_e4m3x4(v[i][0] * m, v[i][2] * m, v[i][4] * m, v[i][6] * m);   /* O[4j..4j+3] */
                    unsigned ev = cvt_e4m3x4(v[i][1] * m, v[i][3] * m, v[i][5] * m, v[i][7] * m);                     /* E[4j+1..4j+4] */
                    r[4 * j + 1] = (uint8_t)ev; r[4 * j + 2] = (uint8_t)(ev >> 8); r[4 * j + 3] = (uint8_t)(ev >> 16); r[4 * j + 4] = (uint8_t)(ev >> 24);
                }
            }
            if (lane < 17) dst[lane * 36] = cvt_e4m3(vh * m);                                                             /* E[0] */
            if (lane == 0) sxs[k * 8 + iz] = (uint8_t)(e + 127);
        }
        /* GY: K block (co, ks) = 2 rows x 16 voxels = 8 x 4 voxels; warp covers 4 blocks, 8 lanes each */
        for (int task = warp; task < BMo * 2; task += 9) {
            int blk = task * 4 + (lane >> 3), c = blk >> 3, ks = blk & 7, f = lane & 7;
            int row = ks * 2 + (f >> 2), vx = 4 * (f & 3), vz = row >> 3, vy = row & 7;
            int co = co0 + c, oz = oz0 + vz, oy = oy0 + vy, ox = ox0 + vx;
            float4 vv = make_float4(0.f, 0.f, 0.f, 0.f);
            if (co < Co && oz < Do && oy < Ho) {
                if constexpr (IS_MX8(TG)) {
                    const size_t vo = ((size_t)oz * Ho + oy) * Wo + ox, Sg = (size_t)Do * Ho * Wo;
                    if (ox < Wo) vv.x = ldmx_e<8>(gy, N, Co, Sg, n, co, vo); if (ox + 1 < Wo) vv.y = ldmx_e<8>(gy, N, Co, Sg, n, co, vo + 1);
                    if (ox + 2 < Wo) vv.z = ldmx_e<8>(gy, N, Co, Sg, n, co, vo + 2); if (ox + 3 < Wo) vv.w = ldmx_e<8>(gy, N, Co, Sg, n, co, vo + 3);
                } else {
                const TG *src = gy + (((size_t)n * Co + co) * Do + oz) * Ho * Wo + (size_t)oy * Wo + ox;
                if (vec) { if (ox < Wo) vv = ldx4<TG>(src); }
                else { if (ox < Wo) vv.x = ldx(src, 0); if (ox + 1 < Wo) vv.y = ldx(src, 1); if (ox + 2 < Wo) vv.z = ldx(src, 2); if (ox + 3 < Wo) vv.w = ldx(src, 3); }
                }
            }
            float amax = fmaxf(fmaxf(fabsf(vv.x), fabsf(vv.y)), fmaxf(fabsf(vv.z), fabsf(vv.w))), sm = (vv.x + vv.y) + (vv.z + vv.w);
#pragma unroll
            for (int o = 4; o; o >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o)); sm += __shfl_xor_sync(0xffffffff, sm, o); }
            int e = mx_exp(amax, 1.f / 448.f);
            float m = exp2i(-e);
            *(unsigned *)(sg + c * G8_CS + row * 16 + vx) = cvt_e4m3x4(vv.x * m, vv.y * m, vv.z * m, vv.w * m);
            if (f == 0) { sgs[c * 8 + ks] = (uint8_t)(e + 127); if (do_bias) atomicAdd(&sbias[c], sm); }
        }
        __syncthreads();
#pragma unroll
        for (int vz = 0; vz < 2; vz++) {
            const int iz = 2 * vz + kz;
            unsigned sb[NT];
#pragma unroll
            for (int q = 0; q < NT; q++) sb[q] = sxs[(q * 8 + g) * 8 + iz];
#pragma unroll 2
            for (int kk = 0; kk < 4; kk++) {
                const int ks = vz * 4 + kk, vy0 = 2 * kk;
                unsigned af[MT][4], sa[MT];
#pragma unroll
                for (int m = 0; m < MT; m++) {
                    int mat = lane >> 3, co = m * 16 + (mat & 1) * 8 + (lane & 7), row = 2 * ks + (mat >> 1);
                    ldsm_x4(af[m], sg + co * G8_CS + row * 16);
                    sa[m] = sgs[(m * 16 + g + 8 * (t & 1)) * 8 + ks];
                }
                const int iy0 = 2 * vy0 + ky;
#pragma unroll
                for (int q = 0; q < NT; q++) {
                    const unsigned *p0 = (const unsigned *)(sxq + (q * 8 + g) * XS2_CS + iz * XS2_PS + iy0 * 36) + t;
                    const unsigned *p1 = p0 + 18;                 /* input row iy0 + 2 (= next output row): +72 B */
                    unsigned e0 = p0[0], e1 = p0[1], o0 = p0[5], f0 = p1[0], f1 = p1[1], o1 = p1[5];
                    unsigned b[3][2] = {{e0, f0}, {o0, o1}, {__funnelshift_r(e0, e1, 8), __funnelshift_r(f0, f1, 8)}};
#pragma unroll
                    for (int kx = 0; kx < 3; kx++)
#pragma unroll
                        for (int m = 0; m < MT; m++) mma_f8(acc[kx][m][q], af[m], b[kx], sa[m], sb[q]);
                }
            }
        }
    }
    if (do_bias) { __syncthreads(); if (threadIdx.x < BMo && co0 + (int)threadIdx.x < Co) atomicAdd(&gb[co0 + threadIdx.x], sbias[threadIdx.x]); }
#pragma unroll
    for (int kx = 0; kx < 3; kx++) {
        int tap = (kz * 3 + ky) * 3 + kx;
#pragma unroll
        for (int m = 0; m < MT; m++)
#pragma unroll
            for (int q = 0; q < NT; q++) {
                int ci = ci0 + q * 8 + 2 * t;
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    int co = co0 + m * 16 + g + 8 * h;
                    if (co >= Co) continue;
                    if (ci < Ci) atomicAdd(&gw[((size_t)co * Ci + ci) * 27 + tap], acc[kx][m][q][2 * h]);
                    if (ci + 1 < Ci) atomicAdd(&gw[((size_t)co * Ci + ci + 1) * 27 + tap], acc[kx][m][q][2 * h + 1]);
                }
            }
    }
}
template <int MT, int NT, typename T, typename TG> static void launch_bws2(dim3 grid, size_t smem, const void *x, shape5 xs, const TG *gy, shape5 ys, float *gw, float *gb, int ZC, gnp_t gp) {
    static int attr[8];
    if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)conv_bwd_w_s2_f8_k<MT, NT, T, TG>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
    conv_bwd_w_s2_f8_k<MT, NT, T, TG><<<grid, 288, smem>>>((const T *)x, gy, gw, gb, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, ys.d, ys.h, ys.w, ZC, gp);
}
template <typename T, typename TG> static void bwd_w_s2_f8_t(const void *x, shape5 xs, const TG *gy, shape5 ys, float *gw, float *gb, gnp_t gp) {
    int MT = ys.c >= 32 ? 2 : 1, NT = xs.c <= 8 ? 1 : 2;
    size_t smem = (size_t)8 * NT * XS2_CS + 16 * MT * (G8_CS + 8) + 8 * NT * 8 + 16 * MT * 4 + 16;
    int nzt = nblk_(ys.d, 2), base = (int)(((xs.c + 8 * NT - 1) / (8 * NT)) * ((ys.c + 16 * MT - 1) / (16 * MT)) * nblk_(ys.w, 16) * nblk_(ys.h, 8) * ys.n);
    int ZC = nzt < 8 ? nzt : 8;
    while (ZC > 1 && (size_t)base * nblk_(nzt, ZC) < 72) ZC--;
    dim3 grid((xs.c + 8 * NT - 1) / (8 * NT), (ys.c + 16 * MT - 1) / (16 * MT), (unsigned)(nblk_(ys.w, 16) * nblk_(ys.h, 8) * nblk_(nzt, ZC) * ys.n));
    switch (MT * 10 + NT) {
    case 11: launch_bws2<1, 1, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, ZC, gp); break;
    case 12: launch_bws2<1, 2, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, ZC, gp); break;
    case 21: launch_bws2<2, 1, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, ZC, gp); break;
    default: launch_bws2<2, 2, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, ZC, gp); break;
    }
}
extern "C" int lp_bwd_w_s2_f8(const void *x, int xbf, shape5 xs, const void *gy, int gybf, shape5 ys, float *gw, float *gb, gnp_t gp) {
    if (gybf == 4) { fprintf(stderr, "lp_bwd_w_s2_f8: fp4 gradients are not supported\n"); abort(); }
    if (xbf == 4 && gybf == 3) bwd_w_s2_f8_t<mx4_t, mx8_t>(x, xs, (const mx8_t *)gy, ys, gw, gb, gp);
    else if (xbf == 4 && gybf == 2) bwd_w_s2_f8_t<mx4_t, __half>(x, xs, (const __half *)gy, ys, gw, gb, gp);
    else if (xbf == 4 && gybf == 1) bwd_w_s2_f8_t<mx4_t, bf16>(x, xs, (const bf16 *)gy, ys, gw, gb, gp);
    else if (xbf == 4) bwd_w_s2_f8_t<mx4_t, float>(x, xs, (const float *)gy, ys, gw, gb, gp);
    else if (xbf == 3 && gybf == 3) bwd_w_s2_f8_t<mx8_t, mx8_t>(x, xs, (const mx8_t *)gy, ys, gw, gb, gp);
    else if (xbf == 3 && gybf == 2) bwd_w_s2_f8_t<mx8_t, __half>(x, xs, (const __half *)gy, ys, gw, gb, gp);
    else if (xbf == 3 && gybf == 1) bwd_w_s2_f8_t<mx8_t, bf16>(x, xs, (const bf16 *)gy, ys, gw, gb, gp);
    else if (xbf == 3) bwd_w_s2_f8_t<mx8_t, float>(x, xs, (const float *)gy, ys, gw, gb, gp);
    else if (xbf == 2 && gybf) bwd_w_s2_f8_t<__half, __half>(x, xs, (const __half *)gy, ys, gw, gb, gp);
    else if (xbf == 2) bwd_w_s2_f8_t<__half, float>(x, xs, (const float *)gy, ys, gw, gb, gp);
    else if (xbf && gybf) bwd_w_s2_f8_t<bf16, bf16>(x, xs, (const bf16 *)gy, ys, gw, gb, gp);
    else if (xbf) bwd_w_s2_f8_t<bf16, float>(x, xs, (const float *)gy, ys, gw, gb, gp);
    else bwd_w_s2_f8_t<float, float>(x, xs, (const float *)gy, ys, gw, gb, gp);
    LPCK();
    return 0;
}

/* ======================= elementwise ops on MX-fp8 activation tensors =======================
   Voxel-major over channel blocks: a thread owns one (n, block, voxel) row of bw channels, so the per-voxel amax that
   defines the output scale is local. */
/* row ri = (n*nb + blk)*S + voxel of a B-bit MX tensor; q = data, sc = scale plane */
template <int B> __device__ __forceinline__ void mx_load_row_b(const uint8_t *q, const uint8_t *sc, size_t ri, int bw, float *v) {
    mxf<B>::dec_row(q + ri * mx_rb(bw, B), bw, mx_scale(sc[ri]), v);
}
/* sr != 0: exact stochastic rounding keyed by (sr, ri, channel) (fp4 only; fp8 keeps round-to-nearest here) */
template <int B> __device__ __forceinline__ void mx_store_row_b(uint8_t *q, uint8_t *sc, size_t ri, int bw, float *v, uint32_t sr = 0) {
    unsigned am = 0u;
#pragma unroll
    for (int k = 0; k < 32; k++) if (k < bw) am = amax_u(am, v[k]);
    const int e = mx_exp(__uint_as_float(am), mxf<B>::inv_qmax);
    const float m = exp2i(-e);
    if (B == 4 && sr) {
#pragma unroll
        for (int k = 0; k < 32; k++) if (k < bw) v[k] = sr_e2m1(v[k] * m, sr_hash(sr, ri * 32 + k)) / m;
    }
    mxf<B>::enc_row(q + ri * mx_rb(bw, B), bw, v, m);
    sc[ri] = (uint8_t)(e + 127);
}
__device__ __forceinline__ void mx_load_row(const uint8_t *q, const uint8_t *sc, size_t ri, int bw, float *v) { mx_load_row_b<8>(q, sc, ri, bw, v); }
__device__ __forceinline__ void mx_store_row(uint8_t *q, uint8_t *sc, size_t ri, int bw, float *v) { mx_store_row_b<8>(q, sc, ri, bw, v); }
/* scale plane of a B-bit MX tensor */
template <int B> __device__ __forceinline__ uint8_t *mx_sc(uint8_t *q, int N, int C, size_t S) { return q + (size_t)N * mx_nb(C) * S * mx_rb(mx_bw(C), B); }
template <int B> __device__ __forceinline__ const uint8_t *mx_sc(const uint8_t *q, int N, int C, size_t S) { return q + (size_t)N * mx_nb(C) * S * mx_rb(mx_bw(C), B); }
/* fp32 / 16-bit [n][C][S] -> MX (B bits) */
template <int B, typename TI>
__global__ void f32_to_mx_k(const TI *x, uint8_t *y, int N, int C, size_t S) {
    const int bw = mx_bw(C), nb = mx_nb(C);
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * nb * S) return;
    size_t v = i % S; int blk = (int)((i / S) % nb), n = (int)(i / (S * nb));
    float r[32];
#pragma unroll
    for (int k = 0; k < 32; k++) { int c = blk * bw + k; r[k] = k < bw && c < C ? ldx(x, ((size_t)n * C + c) * S + v) : 0.f; }
    mx_store_row_b<B>(y, mx_sc<B>(y, N, C, S), i, bw, r);
}
extern "C" void lp_f32_to_mx8(const float *x, int N, int C, size_t S, void *y) {
    size_t n = (size_t)N * mx_nb(C) * S;
    f32_to_mx_k<8, float><<<nblk_(n, 256), 256>>>(x, (uint8_t *)y, N, C, S); LPCK();
}
extern "C" void lp_f32_to_mx4(const float *x, int N, int C, size_t S, void *y) {
    size_t n = (size_t)N * mx_nb(C) * S;
    f32_to_mx_k<4, float><<<nblk_(n, 256), 256>>>(x, (uint8_t *)y, N, C, S); LPCK();
}
/* 16-bit (dt 1 bf16, 2 fp16) [n][C][S] -> MX */
extern "C" void lp_h16_to_mx8(const void *x, int dt, int N, int C, size_t S, void *y) {
    size_t n = (size_t)N * mx_nb(C) * S;
    if (dt == 2) f32_to_mx_k<8, __half><<<nblk_(n, 256), 256>>>((const __half *)x, (uint8_t *)y, N, C, S);
    else f32_to_mx_k<8, bf16><<<nblk_(n, 256), 256>>>((const bf16 *)x, (uint8_t *)y, N, C, S);
    LPCK();
}
extern "C" void lp_h16_to_mx4(const void *x, int dt, int N, int C, size_t S, void *y) {
    size_t n = (size_t)N * mx_nb(C) * S;
    if (dt == 2) f32_to_mx_k<4, __half><<<nblk_(n, 256), 256>>>((const __half *)x, (uint8_t *)y, N, C, S);
    else f32_to_mx_k<4, bf16><<<nblk_(n, 256), 256>>>((const bf16 *)x, (uint8_t *)y, N, C, S);
    LPCK();
}
/* MX -> fp32 [n][C][S] (tests) */
template <int B>
__global__ void mx_to_f32_k(const uint8_t *x, float *y, int N, int C, size_t S) {
    const int bw = mx_bw(C), nb = mx_nb(C);
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * nb * S) return;
    size_t v = i % S; int blk = (int)((i / S) % nb), n = (int)(i / (S * nb));
    float r[32];
    mx_load_row_b<B>(x, mx_sc<B>(x, N, C, S), i, bw, r);
#pragma unroll
    for (int k = 0; k < 32; k++) { int c = blk * bw + k; if (k < bw && c < C) y[((size_t)n * C + c) * S + v] = r[k]; }
}
extern "C" void lp_mx8_to_f32(const void *x, int N, int C, size_t S, float *y) {
    size_t n = (size_t)N * mx_nb(C) * S;
    mx_to_f32_k<8><<<nblk_(n, 256), 256>>>((const uint8_t *)x, y, N, C, S); LPCK();
}
extern "C" void lp_mx4_to_f32(const void *x, int N, int C, size_t S, float *y) {
    size_t n = (size_t)N * mx_nb(C) * S;
    mx_to_f32_k<4><<<nblk_(n, 256), 256>>>((const uint8_t *)x, y, N, C, S); LPCK();
}
/* test probes: mean of n stochastic e2m1 roundings of v (seeded per element), and the raw cvt.rn.satfinite.e2m1x2 nibble of v */
__global__ void sr_e2m1_probe_k(float v, size_t n, double *acc) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float q = sr_e2m1(v, sr_hash(0x1234567u, i));
    atomicAdd(acc, (double)q);
}
extern "C" double lp_sr_e2m1_mean(float v, size_t n) {
    double *d; cudaMalloc(&d, sizeof(double)); cudaMemset(d, 0, sizeof(double));
    sr_e2m1_probe_k<<<nblk_(n, 256), 256>>>(v, n, d);
    double h = 0; cudaMemcpy(&h, d, sizeof(double), cudaMemcpyDeviceToHost); cudaFree(d); LPCK();
    return h / (double)n;
}
__global__ void cvt_e2m1_probe_k(const float *v, unsigned char *o, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = cvt_e2m1x2(v[i], 0.f) & 15; }
extern "C" void lp_cvt_e2m1_probe(const float *hv, unsigned char *ho, int n) {
    float *d; unsigned char *o; cudaMalloc(&d, n * 4); cudaMalloc(&o, n); cudaMemcpy(d, hv, n * 4, cudaMemcpyHostToDevice);
    cvt_e2m1_probe_k<<<nblk_(n, 128), 128>>>(d, o, n);
    cudaMemcpy(ho, o, n, cudaMemcpyDeviceToHost); cudaFree(d); cudaFree(o); LPCK();
}
/* elementwise ops on MX activations of either format: B / BX / BY = element bits of the MX operand (8: mx8, 4: mx4), BX = 0 a
   plane-major input of type TI. lp dtype codes at the entry points: 0 fp32, 1 bf16, 2 fp16, 3 mx8, 4 mx4. */
#define MXB(dt) ((dt) == 4 ? 4 : 8)
/* row ri of block blk of a plane-major [n][C][S] tensor as a 32-wide row (bw channels, rest zero) */
template <typename TI> __device__ __forceinline__ void pm_load_row(const TI *x, int N, int C, size_t S, size_t ri, int bw, float *r) {
    const int nb = mx_nb(C);
    const size_t v = ri % S; const int blk = (int)((ri / S) % nb), n = (int)(ri / (S * nb));
#pragma unroll
    for (int k = 0; k < 32; k++) { int c = blk * bw + k; r[k] = k < bw && c < C ? ldx(x, ((size_t)n * C + c) * S + v) : 0.f; }
}
/* y = silu(gn(x)): x MX (BX bits) or plane-major (BX = 0, type TI), y MX (BY bits) */
template <int BX, int BY, typename TI>
__global__ void gn_silu_apply_mx_k(const void *xv, uint8_t *y, int N, int C, int G, size_t S, const float *gamma, const float *beta, const float *mean, const float *rstd) {
    const int bw = mx_bw(C), nb = mx_nb(C), cpg = C / G;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * nb * S) return;
    int blk = (int)((i / S) % nb), n = (int)(i / (S * nb));
    float r[32];
    if constexpr (BX != 0) { const uint8_t *x = (const uint8_t *)xv; mx_load_row_b<BX>(x, mx_sc<BX>(x, N, C, S), i, bw, r); }
    else pm_load_row<TI>((const TI *)xv, N, C, S, i, bw, r);
#pragma unroll
    for (int k = 0; k < 32; k++) {
        int c = blk * bw + k;
        if (k < bw && c < C) { int ng = n * G + c / cpg; float v = (r[k] - mean[ng]) * rstd[ng] * gamma[c] + beta[c]; r[k] = v / (1.f + __expf(-v)); }
        else r[k] = 0.f;
    }
    mx_store_row_b<BY>(y, mx_sc<BY>(y, N, C, S), i, bw, r);
}
extern "C" void lp_gn_silu_apply_mx(const void *x, int xdt, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd, void *y, int ydt) {
    size_t S = shape_spatial(s), n = (size_t)s.n * mx_nb(s.c) * S;
    if (ydt != 3 && ydt != 4) { fprintf(stderr, "lp_gn_silu_apply_mx: output dtype %d is not MX\n", ydt); abort(); }
#define GSA(BX, BY, TI) gn_silu_apply_mx_k<BX, BY, TI><<<nblk_(n, 256), 256>>>(x, (uint8_t *)y, s.n, s.c, G, S, gamma, beta, mean, rstd)
#define GSA_Y(BX, TI) do { if (ydt == 4) GSA(BX, 4, TI); else GSA(BX, 8, TI); } while (0)
    if (xdt == 4) GSA_Y(4, float); else if (xdt == 3) GSA_Y(8, float); else if (xdt == 2) GSA_Y(0, __half); else if (xdt == 1) GSA_Y(0, bf16); else GSA_Y(0, float);
#undef GSA_Y
#undef GSA
    LPCK();
}
/* exact-2x trilinear upsample (align_corners = false, edge-clamped): out[o] = 0.75 in[o/2] + 0.25 in[o/2 -+ 1] per axis;
   x MX (BX bits), y MX (BY bits), gp: silu(gn(x)) is upsampled */
__device__ __forceinline__ void row_gn(float *r, const gnp_t &gp, int n, int blk, int bw, int Ci);
template <int BX, int BY>
__global__ void up2_mx_k(const uint8_t *x, uint8_t *y, int N, int C, int D, int H, int W, gnp_t gp) {
    const int bw = mx_bw(C), nb = mx_nb(C), Do = 2 * D, Ho = 2 * H, Wo = 2 * W;
    const size_t S = (size_t)D * H * W, So = (size_t)Do * Ho * Wo;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * nb * So) return;
    size_t vo = i % So; size_t nbk = i / So;
    int ox = (int)(vo % Wo), oy = (int)((vo / Wo) % Ho), oz = (int)(vo / ((size_t)Wo * Ho));
    int mz[2] = {oz >> 1, min(max((oz >> 1) + ((oz & 1) ? 1 : -1), 0), D - 1)}, my[2] = {oy >> 1, min(max((oy >> 1) + ((oy & 1) ? 1 : -1), 0), H - 1)}, mx[2] = {ox >> 1, min(max((ox >> 1) + ((ox & 1) ? 1 : -1), 0), W - 1)};
    const float wt[2] = {0.75f, 0.25f};
    const uint8_t *sc = mx_sc<BX>(x, N, C, S);
    float acc[32] = {};
#pragma unroll
    for (int a = 0; a < 2; a++)
#pragma unroll
        for (int bb = 0; bb < 2; bb++)
#pragma unroll
            for (int c = 0; c < 2; c++) {
                float r[32], w3 = wt[a] * wt[bb] * wt[c];
                mx_load_row_b<BX>(x, sc, nbk * S + ((size_t)mz[a] * H + my[bb]) * W + mx[c], bw, r);
                row_gn(r, gp, (int)(nbk / nb), (int)(nbk % nb), bw, C);
#pragma unroll
                for (int k = 0; k < 32; k++) acc[k] += w3 * r[k];
            }
    mx_store_row_b<BY>(y, mx_sc<BY>(y, N, C, So), i, bw, acc);
}
extern "C" void lp_up2_fwd_mx(const void *x, int xdt, shape5 xs, void *y, int ydt, gnp_t gp) {
    size_t n = (size_t)xs.n * mx_nb(xs.c) * 8 * shape_spatial(xs);
    const uint8_t *xq = (const uint8_t *)x; uint8_t *yq = (uint8_t *)y;
#define U2(BX, BY) up2_mx_k<BX, BY><<<nblk_(n, 256), 256>>>(xq, yq, xs.n, xs.c, xs.d, xs.h, xs.w, gp)
    if (xdt == 4) { if (ydt == 4) U2(4, 4); else U2(4, 8); }
    else { if (ydt == 4) U2(8, 4); else U2(8, 8); }
#undef U2
    LPCK();
}
/* in-place silu(gn(.)) of the bw channels of block blk of an MX row (gp.G == 0: none) */
__device__ __forceinline__ void row_gn(float *r, const gnp_t &gp, int n, int blk, int bw, int Ci) {
    if (!gp.G) return;
    const int cpg = Ci / gp.G;
    for (int k = 0; k < bw; k++) {
        int ci = blk * bw + k; if (ci >= Ci) break;
        int ng = n * gp.G + ci / cpg; float a = gp.rstd[ng] * gp.gamma[ci], b = gp.beta[ci] - gp.mean[ng] * a;
        r[k] = act_ab(r[k], a, b, true);
    }
}
/* 1^3 conv (head) reading an MX tensor (B bits): y[n][co][v] = b[co] + sum_ci w[co][ci] x[ci][v] (fp32 output) */
template <int B>
__global__ void conv1_mx_k(const uint8_t *x, const float *w, const float *b, float *y, int N, int Ci, int Co, size_t S, gnp_t gp) {
    const int bw = mx_bw(Ci), nb = mx_nb(Ci);
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * S) return;
    int n = (int)(i / S); size_t v = i % S;
    float acc[8];
    for (int co = 0; co < Co && co < 8; co++) acc[co] = b ? b[co] : 0.f;
    const uint8_t *sc = mx_sc<B>(x, N, Ci, S);
    for (int blk = 0; blk < nb; blk++) {
        float r[32];
        mx_load_row_b<B>(x, sc, ((size_t)n * nb + blk) * S + v, bw, r);
        row_gn(r, gp, n, blk, bw, Ci);
        for (int k = 0; k < bw; k++) { int ci = blk * bw + k; if (ci < Ci) for (int co = 0; co < Co && co < 8; co++) acc[co] += w[co * Ci + ci] * r[k]; }
    }
    for (int co = 0; co < Co && co < 8; co++) y[((size_t)n * Co + co) * S + v] = acc[co];
}
extern "C" void lp_conv1_fwd_mx(const void *x, int xdt, shape5 xs, const float *w, const float *b, int cout, float *y, gnp_t gp) {
    if (cout > 8) { fprintf(stderr, "lp_conv1_fwd_mx: cout %d > 8\n", cout); abort(); }
    size_t S = shape_spatial(xs);
    if (xdt == 4) conv1_mx_k<4><<<nblk_((size_t)xs.n * S, 256), 256>>>((const uint8_t *)x, w, b, y, xs.n, xs.c, cout, S, gp);
    else conv1_mx_k<8><<<nblk_((size_t)xs.n * S, 256), 256>>>((const uint8_t *)x, w, b, y, xs.n, xs.c, cout, S, gp);
    LPCK();
}
/* 1^3 weight gradient with an MX input (B bits): gw[co][ci] += sum_v gy[co][v] x[ci][v] (Ci x Co <= 64); block partials via atomics */
template <int B, typename TG>
__global__ void __launch_bounds__(256) conv_bwd_w1_mx_k(const uint8_t *x, const TG *gy, float *gw, int N, int Ci, int Co, size_t S, gnp_t gp) {
    const int bw = mx_bw(Ci), nb = mx_nb(Ci);
    const uint8_t *sc = mx_sc<B>(x, N, Ci, S);
    float acc[64] = {};
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < (size_t)N * S; i += (size_t)gridDim.x * blockDim.x) {
        int n = (int)(i / S); size_t v = i % S;
        float gv[4];
        for (int co = 0; co < Co && co < 4; co++) gv[co] = ldx(gy, ((size_t)n * Co + co) * S + v);
        for (int blk = 0; blk < nb; blk++) {
            float r[32];
            mx_load_row_b<B>(x, sc, ((size_t)n * nb + blk) * S + v, bw, r);
            row_gn(r, gp, n, blk, bw, Ci);
            for (int k = 0; k < bw; k++) { int ci = blk * bw + k; if (ci < Ci) for (int co = 0; co < Co && co < 4; co++) acc[ci * Co + co] += r[k] * gv[co]; }
        }
    }
    __shared__ float red[64];
    if (threadIdx.x < 64) red[threadIdx.x] = 0.f;
    __syncthreads();
    for (int k = 0; k < Ci * Co && k < 64; k++) {
        float a = acc[k];
        for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o);
        if ((threadIdx.x & 31) == 0) atomicAdd(&red[k], a);
    }
    __syncthreads();
    if (threadIdx.x < Ci * Co && threadIdx.x < 64) { int ci = threadIdx.x / Co, co = threadIdx.x % Co; atomicAdd(&gw[(size_t)co * Ci + ci], red[threadIdx.x]); }
}
/* fast path: one MX block row (Ci <= 32) and CO <= 2 outputs (the head): static accumulator indices (the generic kernel's
   acc[ci * Co + co] lived in local memory), per-sample GN coefficients in shared memory; grid (voxel chunks, sample) */
template <int B, typename TG, int CO>
__global__ void __launch_bounds__(256) conv_bwd_w1_mx1_k(const uint8_t *x, const TG *gy, float *gw, int N, int Ci, size_t S, gnp_t gp, int nper) {
    const int bw = mx_bw(Ci), n = blockIdx.y;
    __shared__ float ca[32], cb[32], red[32 * CO];
    if (threadIdx.x < 32) {
        const int k = threadIdx.x;
        float a = 1.f, bb = 0.f;
        if (gp.G && k < Ci) { const int ng = n * gp.G + k / (Ci / gp.G); a = gp.rstd[ng] * gp.gamma[k]; bb = gp.beta[k] - gp.mean[ng] * a; }
        ca[k] = a; cb[k] = bb;
    }
    if (threadIdx.x < 32 * CO) red[threadIdx.x] = 0.f;
    __syncthreads();
    const uint8_t *sc = mx_sc<B>(x, N, Ci, S);
    float acc[32][CO];
#pragma unroll
    for (int k = 0; k < 32; k++)
#pragma unroll
        for (int co = 0; co < CO; co++) acc[k][co] = 0.f;
    const size_t v0 = (size_t)blockIdx.x * nper, v1 = min(S, v0 + nper);
    for (size_t v = v0 + threadIdx.x; v < v1; v += 256) {
        float r[32], g[CO];
        mx_load_row_b<B>(x, sc, (size_t)n * S + v, bw, r);
#pragma unroll
        for (int co = 0; co < CO; co++) g[co] = ldx(gy, ((size_t)n * CO + co) * S + v);
#pragma unroll
        for (int k = 0; k < 32; k++) {
            if (k < Ci) {
                const float xv = gp.G ? act_ab(r[k], ca[k], cb[k], true) : r[k];
#pragma unroll
                for (int co = 0; co < CO; co++) acc[k][co] += xv * g[co];
            }
        }
    }
#pragma unroll
    for (int k = 0; k < 32; k++) {
        if (k < Ci) {
#pragma unroll
            for (int co = 0; co < CO; co++) {
                float a = acc[k][co];
                for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o);
                if ((threadIdx.x & 31) == 0) atomicAdd(&red[k * CO + co], a);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < Ci * CO) { const int k = threadIdx.x / CO, co = threadIdx.x % CO; atomicAdd(&gw[(size_t)co * Ci + k], red[threadIdx.x]); }
}
extern "C" void lp_bwd_w1_mx(const void *x, int xdt, shape5 xs, const void *gy, int gydt, shape5 ys, float *gw, gnp_t gp) {
    if (xs.c <= 32 && ys.c <= 2) {
        const size_t S = shape_spatial(xs);
        const int nper = 8192, nbx = (int)((S + nper - 1) / nper);
        const dim3 grid(nbx, xs.n);
        const uint8_t *xq = (const uint8_t *)x;
#define BW1F(B, CO) do { if (gydt == 2) conv_bwd_w1_mx1_k<B, __half, CO><<<grid, 256>>>(xq, (const __half *)gy, gw, xs.n, xs.c, S, gp, nper); \
                         else if (gydt == 1) conv_bwd_w1_mx1_k<B, bf16, CO><<<grid, 256>>>(xq, (const bf16 *)gy, gw, xs.n, xs.c, S, gp, nper); \
                         else conv_bwd_w1_mx1_k<B, float, CO><<<grid, 256>>>(xq, (const float *)gy, gw, xs.n, xs.c, S, gp, nper); } while (0)
        if (xdt == 4) { if (ys.c == 2) BW1F(4, 2); else BW1F(4, 1); }
        else { if (ys.c == 2) BW1F(8, 2); else BW1F(8, 1); }
#undef BW1F
        LPCK();
        return;
    }
    if (xs.c * ys.c > 64 || ys.c > 4) { fprintf(stderr, "lp_bwd_w1_mx: %d x %d channels unsupported\n", xs.c, ys.c); abort(); }
    size_t S = shape_spatial(xs);
    int nbk = (int)(((size_t)xs.n * S + 4095) / 4096); if (nbk > 1024) nbk = 1024;
    const uint8_t *xq = (const uint8_t *)x;
#define BW1(B) do { if (gydt == 2) conv_bwd_w1_mx_k<B, __half><<<nbk, 256>>>(xq, (const __half *)gy, gw, xs.n, xs.c, ys.c, S, gp); \
                    else if (gydt == 1) conv_bwd_w1_mx_k<B, bf16><<<nbk, 256>>>(xq, (const bf16 *)gy, gw, xs.n, xs.c, ys.c, S, gp); \
                    else conv_bwd_w1_mx_k<B, float><<<nbk, 256>>>(xq, (const float *)gy, gw, xs.n, xs.c, ys.c, S, gp); } while (0)
    if (xdt == 4) BW1(4); else BW1(8);
#undef BW1
    LPCK();
}
/* GroupNorm sums of an MX tensor (B bits): per (n, group) sum and sum of squares (double, accumulated); a block = 256 voxels of
   one sample, all channel blocks; per-group partials in shared memory */
template <int B>
__global__ void __launch_bounds__(256) gn_sums_mx_k(const uint8_t *x, int N, int C, int G, size_t S, double *sums) {
    const int bw = mx_bw(C), nb = mx_nb(C), cpg = C / G;
    const int nblk_per = (int)((S + 255) / 256);
    const int n = blockIdx.x / nblk_per; const size_t v = (size_t)(blockIdx.x % nblk_per) * 256 + threadIdx.x;
    __shared__ float g1[64], g2[64];
    if (threadIdx.x < 64) { g1[threadIdx.x] = 0.f; g2[threadIdx.x] = 0.f; }
    __syncthreads();
    const uint8_t *sc = mx_sc<B>(x, N, C, S);
    for (int blk = 0; blk < nb; blk++) {
        float r[32];
        if (v < S) mx_load_row_b<B>(x, sc, ((size_t)n * nb + blk) * S + v, bw, r);
        else for (int k = 0; k < 32; k++) r[k] = 0.f;
        /* the block row's channels in order, flushed to their group whenever the group changes (any C / G: a group may
           straddle two block rows; the group index is warp-uniform, so the reductions are) */
        int gcur = -1; float a = 0.f, q = 0.f;
        for (int k = 0; k < bw; k++) {
            const int c = blk * bw + k; if (c >= C) break;
            const int gi = c / cpg;
            if (gi != gcur) {
                if (gcur >= 0) {
                    for (int o = 16; o; o >>= 1) { a += __shfl_xor_sync(0xffffffff, a, o); q += __shfl_xor_sync(0xffffffff, q, o); }
                    if ((threadIdx.x & 31) == 0) { atomicAdd(&g1[gcur], a); atomicAdd(&g2[gcur], q); }
                }
                gcur = gi; a = 0.f; q = 0.f;
            }
            a += r[k]; q += r[k] * r[k];
        }
        if (gcur >= 0) {
            for (int o = 16; o; o >>= 1) { a += __shfl_xor_sync(0xffffffff, a, o); q += __shfl_xor_sync(0xffffffff, q, o); }
            if ((threadIdx.x & 31) == 0) { atomicAdd(&g1[gcur], a); atomicAdd(&g2[gcur], q); }
        }
    }
    __syncthreads();
    for (int g = threadIdx.x; g < G; g += 256) { atomicAdd(&sums[2 * ((size_t)n * G + g)], (double)g1[g]); atomicAdd(&sums[2 * ((size_t)n * G + g) + 1], (double)g2[g]); }
}
extern "C" int lp_gn_sums_mx(const void *x, int xdt, int N, int C, int G, size_t S, double *sums) {
    const int cpg = C / G;
    if (G > 64 || cpg < 1 || C % G) return -1;
    const int nblk_per = (int)((S + 255) / 256);
    if (xdt == 4) gn_sums_mx_k<4><<<N * nblk_per, 256>>>((const uint8_t *)x, N, C, G, S, sums);
    else gn_sums_mx_k<8><<<N * nblk_per, 256>>>((const uint8_t *)x, N, C, G, S, sums);
    LPCK();
    return 0;
}
/* backward through silu(gn(x)) with an MX x (B bits; gy / gx plane-major of type TG / TO, or both MX-fp8): a = gy * silu'(gn(x)).
   Pass 1 (stats): per (n, c) sums of a and a * xhat. Block = (voxel slab, channel block, sample) with >= 8k elements: each
   thread accumulates its voxels in registers (one channel block, bw accumulators), one warp reduction per block at the end
   (the former 256-voxel blocks reduced every channel across the warp per 32 voxels and launched only S / 256 blocks: 2.5x
   the 16-bit kernel, launch-bound at levels 2-3). Pass 2 (apply): thread per (voxel, channel block), gx = rstd (a gamma -
   A/len - xhat B/len) with the group sums A, B. */
/* channels h0 .. h0 + 15 of MX row ri (B bits, row width bw), dequantised (static indexing: no local-memory row) */
template <int B> __device__ __forceinline__ void mx_load16(const uint8_t *q, const uint8_t *sc, size_t ri, int bw, int h0, float *v) {
    const float s = mx_scale(sc[ri]);
    const uint8_t *p = q + ri * mx_rb(bw, B) + h0 * B / 8;
    if constexpr (B == 8) {
        const uint4 u = __ldg((const uint4 *)p);
        const unsigned w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 a = dec_e4m3x2((unsigned short)(w[j] & 0xffff)), b = dec_e4m3x2((unsigned short)(w[j] >> 16));
            v[4 * j] = a.x * s; v[4 * j + 1] = a.y * s; v[4 * j + 2] = b.x * s; v[4 * j + 3] = b.y * s;
        }
    } else {
        const uint2 u = __ldg((const uint2 *)p);
        const unsigned w[2] = {u.x, u.y};
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int q2 = 0; q2 < 4; q2++) { float2 a = dec_e2m1x2(w[j] >> (8 * q2)); v[8 * j + 2 * q2] = a.x * s; v[8 * j + 2 * q2 + 1] = a.y * s; }
    }
}
template <int B, typename TG, int V>   /* V consecutive voxels per thread step (vector gy loads; 1: any S) */
__global__ void __launch_bounds__(256) gn_silu_bwd_stats_mx_k(const uint8_t *x, const TG *gy, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                                           int N, int C, int G, size_t S, float *part) {
    const int bw = mx_bw(C), nb = mx_nb(C), cpg = C / G;
    const int hpb = bw / 16, blk = blockIdx.y / hpb, h0 = (blockIdx.y % hpb) * 16, n = blockIdx.z, nsl = gridDim.x;   /* 16 channels per block (registers: occupancy) */
    const size_t nq = S / V, q0 = nq * blockIdx.x / nsl, q1 = nq * (blockIdx.x + 1) / nsl;   /* slab in units of V voxels */
    __shared__ float cm[16], cr[16], cg[16], cb[16], s1[16], s2[16];
    if (threadIdx.x < 16) {
        const int k = threadIdx.x, c = blk * bw + h0 + k;
        const bool ok = h0 + k < bw && c < C;
        const int ng = n * G + (ok ? c / cpg : 0);
        cm[k] = ok ? mean[ng] : 0.f; cr[k] = ok ? rstd[ng] : 0.f; cg[k] = ok ? gamma[c] : 0.f; cb[k] = ok ? beta[c] : 0.f; s1[k] = 0.f; s2[k] = 0.f;
    }
    __syncthreads();
    const uint8_t *sc = mx_sc<B>(x, N, C, S);
    float a1[16], a2[16];
#pragma unroll
    for (int k = 0; k < 16; k++) { a1[k] = 0.f; a2[k] = 0.f; }
    const int kmax = max(0, min(min(bw, C - blk * bw) - h0, 16));
    const size_t rbase = ((size_t)n * nb + blk) * S;
    const TG *gyb = IS_MX8(TG) ? gy : gy + ((size_t)n * C + blk * bw + h0) * S;
    for (size_t q = q0 + threadIdx.x; q < q1; q += 256) {
        const size_t v = q * V;
        float g[V][16];
        if constexpr (!IS_MX8(TG)) {
#pragma unroll
            for (int k = 0; k < 16; k++) {
                if (k < kmax) {
                    if constexpr (V == 4) { const float4 f = ldx4<TG>(gyb + (size_t)k * S + v); g[0][k] = f.x; g[1][k] = f.y; g[2][k] = f.z; g[3][k] = f.w; }
                    else if constexpr (V == 2) { const TG *pp = gyb + (size_t)k * S + v; g[0][k] = ldx(pp, 0); g[1][k] = ldx(pp, 1); }
                    else g[0][k] = ldx(gyb, (size_t)k * S + v);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < V; j++) {
            float r[16], gr[16];
            mx_load16<B>(x, sc, rbase + v + j, bw, h0, r);
            if constexpr (IS_MX8(TG)) mx_load16<8>((const uint8_t *)gy, mx_sc<8>((const uint8_t *)gy, N, C, S), rbase + v + j, bw, h0, gr);
#pragma unroll
            for (int k = 0; k < 16; k++) {
                if (k < kmax) {
                    const float xhat = (r[k] - cm[k]) * cr[k], u = xhat * cg[k] + cb[k], sg = __fdividef(1.f, 1.f + __expf(-u));
                    const float a = (IS_MX8(TG) ? gr[k] : g[j][k]) * (sg * (1.f + u * (1.f - sg)));
                    a1[k] += a; a2[k] += a * xhat;
                }
            }
        }
    }
#pragma unroll
    for (int k = 0; k < 16; k++) {
        if (k < kmax) {
            float a = a1[k], q = a2[k];
            for (int o = 16; o; o >>= 1) { a += __shfl_xor_sync(0xffffffff, a, o); q += __shfl_xor_sync(0xffffffff, q, o); }
            if ((threadIdx.x & 31) == 0) { atomicAdd(&s1[k], a); atomicAdd(&s2[k], q); }
        }
    }
    __syncthreads();
    /* per-slab partials (same-address double atomics across slabs serialised at the small levels); summed by gn_part_sum_k */
    if ((int)threadIdx.x < kmax) { const size_t c = (size_t)n * C + blk * bw + h0 + threadIdx.x, o = ((size_t)blockIdx.x * N * C + c) * 2; part[o] = s1[threadIdx.x]; part[o + 1] = s2[threadIdx.x]; }
}
__global__ void gn_part_sum_k(const float *part, int nsl, int NC, double *ds) {   /* ds[2 c + j] = sum over slabs (double) */
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * NC) return;
    double a = 0.0;
    for (int sl = 0; sl < nsl; sl++) a += (double)part[(size_t)sl * 2 * NC + i];
    ds[i] = a;
}
template <int B, typename TG, typename TO>
__global__ void __launch_bounds__(256) gn_silu_bwd_apply_mx_k(const uint8_t *x, const TG *gy, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                                           const float *AB, TO *gx, int N, int C, int G, size_t S) {
    /* grid (voxel chunks of 256, channel block, sample): the block's per-channel coefficients come from shared memory */
    const int bw = mx_bw(C), nb = mx_nb(C), cpg = C / G, blk = blockIdx.y, n = blockIdx.z;
    __shared__ float cm[32], cr[32], cg[32], cb[32], ca[32], cc[32];
    if (threadIdx.x < 32) {
        const int k = threadIdx.x, c = blk * bw + k;
        const bool ok = k < bw && c < C;
        const int ng = n * G + (ok ? c / cpg : 0);
        const float len = (float)cpg * (float)S;
        cm[k] = ok ? mean[ng] : 0.f; cr[k] = ok ? rstd[ng] : 0.f; cg[k] = ok ? gamma[c] : 0.f; cb[k] = ok ? beta[c] : 0.f;
        ca[k] = ok ? AB[2 * ng] / len : 0.f; cc[k] = ok ? AB[2 * ng + 1] / len : 0.f;
    }
    __syncthreads();
    const size_t v = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (v >= S) return;
    const size_t i = ((size_t)n * nb + blk) * S + v;
    const uint8_t *sc = mx_sc<B>(x, N, C, S);
    float r[32], gr[32], out[32];
    mx_load_row_b<B>(x, sc, i, bw, r);
    if constexpr (IS_MX8(TG)) mx_load_row((const uint8_t *)gy, mx_sc<8>((const uint8_t *)gy, N, C, S), i, bw, gr);
    const int kmax = min(bw, C - blk * bw);
    TO *gxb = IS_MX8(TO) ? gx : gx + ((size_t)n * C + blk * bw) * S + v;
    const TG *gyb = IS_MX8(TG) ? gy : gy + ((size_t)n * C + blk * bw) * S + v;
#pragma unroll
    for (int k = 0; k < 32; k++) {
        out[k] = 0.f;
        if (k < kmax) {
            const float rs = cr[k], ga = cg[k];
            const float xhat = (r[k] - cm[k]) * rs, u = xhat * ga + cb[k], sg = __fdividef(1.f, 1.f + __expf(-u));
            const float a = (IS_MX8(TG) ? gr[k] : ldx(gyb, (size_t)k * S)) * (sg * (1.f + u * (1.f - sg)));
            const float gv = rs * (a * ga - ca[k] - xhat * cc[k]);
            if constexpr (IS_MX8(TO)) out[k] = gv; else stx(gxb, (size_t)k * S, gv);
        }
    }
    if constexpr (IS_MX8(TO)) mx_store_row((uint8_t *)gx, mx_sc<8>((uint8_t *)gx, N, C, S), i, bw, out);
}
extern "C" void lp_gn_silu_bwd_mx(const void *x, int xdt, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                  const void *gy, void *gx, int gdt, double *ds, float *st, float *AB) {
    /* ds: 2 N C doubles (zeroed here), st: 2 N C floats, AB: 2 N G floats; the caller finishes with the group sums / param grads */
    size_t S = shape_spatial(s);
    const int NC = s.n * s.c, nb = mx_nb(s.c), bw = mx_bw(s.c);
    /* slabs: >= 8 voxel steps per thread (256 threads x V 2) to amortise the end-of-block reduction, but >= ~160 blocks in all
       (small levels were latency-bound with a handful of blocks), and >= one step per thread */
    const size_t ny = (size_t)nb * (bw / 16) * s.n;
    size_t slabs = S / 4096, want = (160 + ny - 1) / ny, maxs = S / 512 > 0 ? S / 512 : 1;
    if (slabs < want) slabs = want;
    if (slabs > maxs) slabs = maxs;
    if (slabs > 1024) slabs = 1024;
    const dim3 grid((unsigned)slabs, (unsigned)(nb * (bw / 16)), (unsigned)s.n);
    float *part = lp_buf<float>(5, slabs * 2 * NC);
    if (s.c > 160) { fprintf(stderr, "lp_gn_silu_bwd_mx: C %d > 160\n", s.c); abort(); }
    if (gdt == 4) { fprintf(stderr, "lp_gn_silu_bwd_mx: fp4 gradients are not supported\n"); abort(); }
    cudaMemsetAsync(ds, 0, (size_t)2 * NC * sizeof(double));
    const uint8_t *xq = (const uint8_t *)x;
#define GBS2(B, V) do { if (gdt == 3) gn_silu_bwd_stats_mx_k<B, mx8_t, 1><<<grid, 256>>>(xq, (const mx8_t *)gy, gamma, beta, mean, rstd, s.n, s.c, G, S, part); \
                    else if (gdt == 2) gn_silu_bwd_stats_mx_k<B, __half, V><<<grid, 256>>>(xq, (const __half *)gy, gamma, beta, mean, rstd, s.n, s.c, G, S, part); \
                    else if (gdt == 1) gn_silu_bwd_stats_mx_k<B, bf16, V><<<grid, 256>>>(xq, (const bf16 *)gy, gamma, beta, mean, rstd, s.n, s.c, G, S, part); \
                    else gn_silu_bwd_stats_mx_k<B, float, V><<<grid, 256>>>(xq, (const float *)gy, gamma, beta, mean, rstd, s.n, s.c, G, S, part); } while (0)
#define GBS(B) do { if (S % 4 == 0) GBS2(B, 2); else GBS2(B, 1); } while (0)
    if (xdt == 4) GBS(4); else GBS(8);
#undef GBS
#undef GBS2
    gn_part_sum_k<<<nblk_((size_t)2 * NC, 128), 128>>>(part, (int)slabs, NC, ds);
    (void)st; (void)AB;
    LPCK();
}
extern "C" void lp_gn_silu_bwd_apply_mx(const void *x, int xdt, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                        const void *gy, void *gx, int gdt, const float *AB) {
    size_t S = shape_spatial(s);
    const dim3 ga(nblk_(S, 256), (unsigned)mx_nb(s.c), (unsigned)s.n);
    const uint8_t *xq = (const uint8_t *)x;
#define GBA(B) do { if (gdt == 3) gn_silu_bwd_apply_mx_k<B, mx8_t, mx8_t><<<ga, 256>>>(xq, (const mx8_t *)gy, gamma, beta, mean, rstd, AB, (mx8_t *)gx, s.n, s.c, G, S); \
                    else if (gdt == 2) gn_silu_bwd_apply_mx_k<B, __half, __half><<<ga, 256>>>(xq, (const __half *)gy, gamma, beta, mean, rstd, AB, (__half *)gx, s.n, s.c, G, S); \
                    else if (gdt == 1) gn_silu_bwd_apply_mx_k<B, bf16, bf16><<<ga, 256>>>(xq, (const bf16 *)gy, gamma, beta, mean, rstd, AB, (bf16 *)gx, s.n, s.c, G, S); \
                    else gn_silu_bwd_apply_mx_k<B, float, float><<<ga, 256>>>(xq, (const float *)gy, gamma, beta, mean, rstd, AB, (float *)gx, s.n, s.c, G, S); } while (0)
    if (xdt == 4) GBA(4); else GBA(8);
#undef GBA
    LPCK();
}

/* 1^3 conv with a plane-major input (dtype gdt) and an MX output (head backward-data: logit gradient -> gout) */
template <typename TG>
__global__ void conv1_to_mx_k(const TG *x, const float *w, uint8_t *y, int N, int Ci, int Co, size_t S) {
    const int bw = mx_bw(Co), nb = mx_nb(Co);
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * nb * S) return;
    size_t v = i % S; int blk = (int)((i / S) % nb), n = (int)(i / (S * nb));
    float xi[8], r[32];
    for (int ci = 0; ci < Ci && ci < 8; ci++) xi[ci] = ldx(x, ((size_t)n * Ci + ci) * S + v);
#pragma unroll
    for (int k = 0; k < 32; k++) {
        int co = blk * bw + k; float a = 0.f;
        if (k < bw && co < Co) for (int ci = 0; ci < Ci && ci < 8; ci++) a += w[co * Ci + ci] * xi[ci];
        r[k] = a;
    }
    mx_store_row(y, y + (size_t)N * nb * S * bw, i, bw, r);
}
extern "C" void lp_conv1_to_mx(const void *x, int gdt, int N, int Ci, size_t S, const float *w, int Co, void *y) {
    if (Ci > 8) { fprintf(stderr, "lp_conv1_to_mx: Ci %d > 8\n", Ci); abort(); }
    size_t n = (size_t)N * mx_nb(Co) * S;
    if (gdt == 2) conv1_to_mx_k<__half><<<nblk_(n, 256), 256>>>((const __half *)x, w, (uint8_t *)y, N, Ci, Co, S);
    else if (gdt == 1) conv1_to_mx_k<bf16><<<nblk_(n, 256), 256>>>((const bf16 *)x, w, (uint8_t *)y, N, Ci, Co, S);
    else conv1_to_mx_k<float><<<nblk_(n, 256), 256>>>((const float *)x, w, (uint8_t *)y, N, Ci, Co, S);
    LPCK();
}

/* ---- MX backward of the exact-2x trilinear upsample: gx[m] = sum over fine outputs o in [2m - 1, 2m + 2] (per axis) of
   coef(o, m) gy[o], coef as in the forward (0.75 for in[o/2], 0.25 for the clamped neighbour) ---- */
__device__ __forceinline__ float upc(int o, int m, int n_in) {
    if (o < 0 || o >= 2 * n_in) return 0.f;
    int m0 = o >> 1, m1 = (o & 1) ? min(m0 + 1, n_in - 1) : max(m0 - 1, 0);
    return (m == m0 ? 0.75f : 0.f) + (m == m1 ? 0.25f : 0.f);
}
__global__ void up2_bwd_mx_k(const uint8_t *gy, uint8_t *gx, int N, int C, int D, int H, int W) {   /* D, H, W: coarse (gx) grid */
    const int bw = mx_bw(C), nb = mx_nb(C), Do = 2 * D, Ho = 2 * H, Wo = 2 * W;
    const size_t S = (size_t)D * H * W, So = (size_t)Do * Ho * Wo;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * nb * S) return;
    size_t v = i % S, nbk = i / S;
    int x = (int)(v % W), y = (int)((v / W) % H), z = (int)(v / ((size_t)W * H));
    const uint8_t *sc = gy + (size_t)N * nb * So * bw;
    float acc[32] = {};
#pragma unroll 1
    for (int a = 0; a < 4; a++) {
        int oz = 2 * z - 1 + a; float wz = upc(oz, z, D); if (wz == 0.f) continue;
#pragma unroll 1
        for (int bb = 0; bb < 4; bb++) {
            int oy = 2 * y - 1 + bb; float wy = upc(oy, y, H); if (wy == 0.f) continue;
#pragma unroll 1
            for (int c = 0; c < 4; c++) {
                int ox = 2 * x - 1 + c; float wx = upc(ox, x, W); if (wx == 0.f) continue;
                float r[32], w3 = wz * wy * wx;
                mx_load_row(gy, sc, nbk * So + ((size_t)oz * Ho + oy) * Wo + ox, bw, r);
#pragma unroll
                for (int k = 0; k < 32; k++) acc[k] += w3 * r[k];
            }
        }
    }
    mx_store_row(gx, gx + (size_t)N * nb * S * bw, i, bw, acc);
}
extern "C" void lp_up2_bwd_mx(const void *gy, shape5 xs, void *gx) {
    size_t n = (size_t)xs.n * mx_nb(xs.c) * shape_spatial(xs);
    up2_bwd_mx_k<<<nblk_(n, 256), 256>>>((const uint8_t *)gy, (uint8_t *)gx, xs.n, xs.c, xs.d, xs.h, xs.w); LPCK();
}
/* channel slice [c0, c0 + nc) of an MX-fp8 gx with ctot channels from an MX-fp8 gy of nc channels (the chunked up-part gradient
   of the decoder): thread = (n, gx block touched by the slice, coarse voxel). A slice that starts at its block's first channel
   writes the block fresh (other channels zero); a later slice of the same block (16-channel chunks of a 32-channel block)
   decodes the stored row, replaces its channels and requantises (one extra e4m3 rounding of the earlier chunk's channels when
   the block scale grows). Chunks must be processed in increasing c0; a slice lies in one gy block (nc <= 16, or 32-aligned). */
__global__ void up2_bwd_mx_slice_k(const uint8_t *gy, uint8_t *gx, int N, int nc, int ctot, int c0, int ob0, int nob, int D, int H, int W) {
    const int bwy = mx_bw(nc), nby = mx_nb(nc), bwx = mx_bw(ctot), nbx = mx_nb(ctot), Do = 2 * D, Ho = 2 * H, Wo = 2 * W;
    const size_t S = (size_t)D * H * W, So = (size_t)Do * Ho * Wo;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * nob * S) return;
    const size_t v = i % S; const int ob = ob0 + (int)((i / S) % nob), n = (int)(i / (S * nob));
    const int lo = max(c0, ob * bwx), hi = min(c0 + nc, min(ob * bwx + bwx, ctot)), yb = (lo - c0) / bwy, ko = (lo - c0) - yb * bwy;
    const int x = (int)(v % W), y = (int)((v / W) % H), z = (int)(v / ((size_t)W * H));
    const uint8_t *scy = mx_sc<8>(gy, N, nc, So);
    float acc[32] = {};
#pragma unroll 1
    for (int a = 0; a < 4; a++) {
        int oz = 2 * z - 1 + a; float wz = upc(oz, z, D); if (wz == 0.f) continue;
#pragma unroll 1
        for (int bb = 0; bb < 4; bb++) {
            int oy = 2 * y - 1 + bb; float wy = upc(oy, y, H); if (wy == 0.f) continue;
#pragma unroll 1
            for (int c = 0; c < 4; c++) {
                int ox = 2 * x - 1 + c; float wx = upc(ox, x, W); if (wx == 0.f) continue;
                float r[32], w3 = wz * wy * wx;
                mx_load_row(gy, scy, ((size_t)n * nby + yb) * So + ((size_t)oz * Ho + oy) * Wo + ox, bwy, r);
#pragma unroll
                for (int k = 0; k < 32; k++) acc[k] += w3 * r[k];
            }
        }
    }
    const size_t ri = ((size_t)n * nbx + ob) * S + v;
    uint8_t *scx = mx_sc<8>(gx, N, ctot, S);
    float out[32];
    if (lo == ob * bwx) {
#pragma unroll
        for (int k = 0; k < 32; k++) out[k] = 0.f;
    } else mx_load_row(gx, scx, ri, bwx, out);
#pragma unroll
    for (int k = 0; k < 32; k++) { const int cch = ob * bwx + k; if (cch >= lo && cch < hi) { const int kk = cch - lo + ko; float val = 0.f;
#pragma unroll
        for (int q = 0; q < 32; q++) if (q == kk) val = acc[q];
        out[k] = val; } }
    mx_store_row(gx, scx, ri, bwx, out);
}
extern "C" void lp_up2_bwd_mx_slice(const void *gy, shape5 xs, void *gx, int ctot, int c0) {   /* xs: coarse shape with nc = xs.c channels */
    const int nc = xs.c, bwx = mx_bw(ctot), bwy = mx_bw(nc);
    if (c0 % 16 || (nc > 16 && (c0 % 32 || (nc % 32 && c0 + nc != ctot))) || (c0 % bwx + nc > bwx && nc <= 16)) { fprintf(stderr, "lp_up2_bwd_mx_slice: unaligned slice c0 %d nc %d of %d\n", c0, nc, ctot); abort(); }
    (void)bwy;
    const int ob0 = c0 / bwx, ob1 = (c0 + nc - 1) / bwx, nob = ob1 - ob0 + 1;
    if (nob > 1 && nc <= 16) { fprintf(stderr, "lp_up2_bwd_mx_slice: slice spans blocks\n"); abort(); }
    if (nob > 1) {   /* 32-aligned multi-block slice: one gy block per gx block, launch per block */
        for (int ob = ob0; ob <= ob1; ob++) {
            size_t n = (size_t)xs.n * shape_spatial(xs);
            up2_bwd_mx_slice_k<<<nblk_(n, 256), 256>>>((const uint8_t *)gy, (uint8_t *)gx, xs.n, nc, ctot, c0, ob, 1, xs.d, xs.h, xs.w);
        }
    } else {
        size_t n = (size_t)xs.n * shape_spatial(xs);
        up2_bwd_mx_slice_k<<<nblk_(n, 256), 256>>>((const uint8_t *)gy, (uint8_t *)gx, xs.n, nc, ctot, c0, ob0, 1, xs.d, xs.h, xs.w);
    }
    LPCK();
}
/* ---- MX stride-2 backward-data (k = 3, pad 1): gx[ci][u] (+)= sum over the taps k with (u + 1 - k) even, o = (u + 1 - k) / 2
   in range, of sum_co w[co][ci][k] gy[co][o]. Thread = (n, ci block, gx voxel); weights from shared memory when they fit. */
template <int WS>
__global__ void __launch_bounds__(128) bwd_data_s2_mx_k(const uint8_t *gy, const float *w, uint8_t *gx, int N, int Ci, int Co,
                                                       int D, int H, int W, int Do, int Ho, int Wo, int accum) {
    extern __shared__ float sw[];
    const int bwx = mx_bw(Ci), nbx = mx_nb(Ci), bwy = mx_bw(Co), nby = mx_nb(Co);
    const size_t S = (size_t)D * H * W, So = (size_t)Do * Ho * Wo;
    if (WS) { for (int i = threadIdx.x; i < Co * Ci * 27; i += blockDim.x) sw[i] = w[i]; __syncthreads(); }
    const float *wt = WS ? sw : w;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * nbx * S) return;
    size_t v = i % S; int blk = (int)((i / S) % nbx), n = (int)(i / (S * nbx));
    int ux = (int)(v % W), uy = (int)((v / W) % H), uz = (int)(v / ((size_t)W * H));
    int kz[2], oz[2], nz = 0, ky[2], oy[2], ny = 0, kx[2], ox[2], nx = 0;
    for (int k = 0; k < 3; k++) {
        int t;
        t = uz + 1 - k; if (!(t & 1) && t >= 0 && (t >> 1) < Do) { kz[nz] = k; oz[nz++] = t >> 1; }
        t = uy + 1 - k; if (!(t & 1) && t >= 0 && (t >> 1) < Ho) { ky[ny] = k; oy[ny++] = t >> 1; }
        t = ux + 1 - k; if (!(t & 1) && t >= 0 && (t >> 1) < Wo) { kx[nx] = k; ox[nx++] = t >> 1; }
    }
    float acc[32] = {};
    const uint8_t *scy = gy + (size_t)N * nby * So * bwy;
#pragma unroll 1
    for (int a = 0; a < nz; a++) for (int bb = 0; bb < ny; bb++) for (int c = 0; c < nx; c++) {
        const int tap = (kz[a] * 3 + ky[bb]) * 3 + kx[c];
        const size_t vo = ((size_t)oz[a] * Ho + oy[bb]) * Wo + ox[c];
        for (int yb = 0; yb < nby; yb++) {
            float r[32];
            mx_load_row(gy, scy, ((size_t)n * nby + yb) * So + vo, bwy, r);
            for (int k = 0; k < bwy; k++) {
                const int co = yb * bwy + k;
                if (co >= Co) break;
                const float g = r[k];
                const float *wr = wt + ((size_t)co * Ci + blk * bwx) * 27 + tap;
#pragma unroll
                for (int cc = 0; cc < 32; cc++) if (cc < bwx && blk * bwx + cc < Ci) acc[cc] += wr[cc * 27] * g;
            }
        }
    }
    const uint8_t *scx = gx + (size_t)N * nbx * S * bwx;
    if (accum) {
        float r[32];
        mx_load_row(gx, scx, i, bwx, r);
#pragma unroll
        for (int k = 0; k < 32; k++) acc[k] += r[k];
    }
    mx_store_row(gx, gx + (size_t)N * nbx * S * bwx, i, bwx, acc);
}
/* MX-fp8 stride-2 backward-data, parity-major: a block covers one parity class p = (pz, py, px) of the gx grid (u = 2 m + p
   per axis), so every thread of the block uses the same 1..8 taps (k = 1 for an even coordinate: o = m; k = 0 / 2 for an odd
   one: o = m + 1 / m) and reads the weights of those taps as broadcast float4 from shared memory: [tap][co][32 ci of this
   block]. Thread = one gx voxel of the class and one ci block (all its bwx channels), gy rows of consecutive threads are
   consecutive voxels (coalesced). The former thread-per-voxel kernel mixed parities inside a warp (divergent tap sets, bank
   conflicted weights): down0 3.66 ms vs 0.59 for the 16-bit tensor-core path at 96^3 B2. */
__global__ void __launch_bounds__(128) bwd_data_s2_mx2_k(const uint8_t *gy, const float *w, uint8_t *gx, int N, int Ci, int Co,
                                                         int D, int H, int W, int Do, int Ho, int Wo, int accum) {
    extern __shared__ float4 swq[];   /* [ntap][Co][bwx / 4] */
    const int bwx = mx_bw(Ci), nbx = mx_nb(Ci), bwy = mx_bw(Co), nby = mx_nb(Co);
    const int p = blockIdx.y, pz = p >> 2, py = (p >> 1) & 1, px = p & 1;
    const int blk = blockIdx.z % nbx, n = blockIdx.z / nbx;
    const int Dp = (D - pz + 1) / 2, Hp = (H - py + 1) / 2, Wp = (W - px + 1) / 2;
    /* tap sets per axis: even -> {k 1, o = m}; odd -> {k 0, o = m + 1}, {k 2, o = m} */
    const int nz = pz ? 2 : 1, ny = py ? 2 : 1, nx = px ? 2 : 1, ntap = nz * ny * nx;
    const int q4 = bwx / 4;
    float *sw = (float *)swq;
    for (int i = threadIdx.x; i < ntap * Co * bwx; i += blockDim.x) {
        const int cc = i % bwx, co = (i / bwx) % Co, tl = i / (bwx * Co);
        const int a = tl / (ny * nx), bb = (tl / nx) % ny, c = tl % nx;
        const int kz = pz ? (a ? 2 : 0) : 1, ky = py ? (bb ? 2 : 0) : 1, kx = px ? (c ? 2 : 0) : 1;
        const int ci = blk * bwx + cc;
        sw[i] = ci < Ci ? w[((size_t)co * Ci + ci) * 27 + (kz * 3 + ky) * 3 + kx] : 0.f;
    }
    __syncthreads();
    const size_t mi = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (mi >= (size_t)Dp * Hp * Wp) return;
    const int mx = (int)(mi % Wp), my = (int)((mi / Wp) % Hp), mz = (int)(mi / ((size_t)Wp * Hp));
    const int uz = 2 * mz + pz, uy = 2 * my + py, ux = 2 * mx + px;
    const size_t So = (size_t)Do * Ho * Wo, S = (size_t)D * H * W;
    const uint8_t *scy = mx_sc<8>(gy, N, Co, So);
    float acc[32];
#pragma unroll
    for (int k = 0; k < 32; k++) acc[k] = 0.f;
    for (int tl = 0; tl < ntap; tl++) {
        const int a = tl / (ny * nx), bb = (tl / nx) % ny, c = tl % nx;
        const int oz = mz + (pz && !a ? 1 : 0), oy = my + (py && !bb ? 1 : 0), ox = mx + (px && !c ? 1 : 0);
        if (oz >= Do || oy >= Ho || ox >= Wo) continue;
        const size_t vo = ((size_t)oz * Ho + oy) * Wo + ox;
        for (int yb = 0; yb < nby; yb++) {
            float r[32];
            mx_load_row_b<8>(gy, scy, ((size_t)n * nby + yb) * So + vo, bwy, r);
            const float4 *wt = swq + ((size_t)tl * Co + yb * bwy) * q4;
#pragma unroll
            for (int k = 0; k < 32; k++) {
                if (k < bwy && yb * bwy + k < Co) {
                    const float g = r[k];
                    const float4 *wr = wt + (size_t)k * q4;
#pragma unroll
                    for (int j = 0; j < 8; j++) if (j < q4) { const float4 f = wr[j]; acc[4 * j] += f.x * g; acc[4 * j + 1] += f.y * g; acc[4 * j + 2] += f.z * g; acc[4 * j + 3] += f.w * g; }
                }
            }
        }
    }
    const size_t ri = ((size_t)n * nbx + blk) * S + ((size_t)uz * H + uy) * W + ux;
    uint8_t *scx = mx_sc<8>(gx, N, Ci, S);
    if (accum) {
        float r[32];
        mx_load_row_b<8>(gx, scx, ri, bwx, r);
#pragma unroll
        for (int k = 0; k < 32; k++) acc[k] += r[k];
    }
    mx_store_row_b<8>(gx, scx, ri, bwx, acc);
}
extern "C" void lp_bwd_data_s2_mx(const void *gy, shape5 ys, const float *w, shape5 xs, void *gx, int accum) {
    static int pm = -1;
    if (pm < 0) pm = ufsm_env_on("UFSM_S2B_MX_OLD") ? 0 : 1;
    if (pm) {
        const int bwx = mx_bw(xs.c), nbx = mx_nb(xs.c);
        const size_t smem = (size_t)8 * ys.c * bwx * sizeof(float);
        static int attr[8];
        if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)bwd_data_s2_mx2_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
        const size_t mmax = (size_t)((xs.d + 1) / 2) * ((xs.h + 1) / 2) * ((xs.w + 1) / 2);
        const dim3 grid(nblk_(mmax, 128), 8, (unsigned)(nbx * xs.n));
        bwd_data_s2_mx2_k<<<grid, 128, smem>>>((const uint8_t *)gy, w, (uint8_t *)gx, xs.n, xs.c, ys.c, xs.d, xs.h, xs.w, ys.d, ys.h, ys.w, accum);
        LPCK();
        return;
    }
    size_t n = (size_t)xs.n * mx_nb(xs.c) * shape_spatial(xs), wb = (size_t)ys.c * xs.c * 27 * sizeof(float);
    if (wb <= 48 * 1024) bwd_data_s2_mx_k<1><<<nblk_(n, 128), 128, wb>>>((const uint8_t *)gy, w, (uint8_t *)gx, xs.n, xs.c, ys.c, xs.d, xs.h, xs.w, ys.d, ys.h, ys.w, accum);
    else bwd_data_s2_mx_k<0><<<nblk_(n, 128), 128>>>((const uint8_t *)gy, w, (uint8_t *)gx, xs.n, xs.c, ys.c, xs.d, xs.h, xs.w, ys.d, ys.h, ys.w, accum);
    LPCK();
}

/* ======================= FP4 weight gradient, k=3, stride 1 =======================
   GW[co][ci][tap] = sum_v GY[co][v] X[ci][v + off(tap)] with m16n8k64 kind::mxf4 (e2m1 x e2m1, ue8m0 per 32 K, fp32
   accumulate). Same block / warp structure as conv_bwd_w_f8_k: 9 warps, warp w owns the tap row (kz, ky) = (w/3, w%3)
   and all three kx, M = 16 MT output channels (GY), N = 8 NT input channels, a block walks ZC z-steps of 2 output planes
   (8 y x 16 x each) and atomically adds into gw. K = 64 voxels = 4 output rows of 16 x of one z plane; each 32-voxel
   scale block is a PAIR of output rows (vy, vy + 1).
   A = GY: one ue8m0 per (co, row pair), exact stochastic rounding onto the e2m1 grid (sr_e2m1_nib: the nibble straight
         from floor(t(a) + u), one hash per 8 elements) keyed by the element; staged 8 values (one e2m1 word) per lane.
   B = X, round to nearest, in one of three layouts (template LY; host: UFSM_F4W_LAYOUT, the Hadamard / x-SR modes always
         use LY 0). A K block of X is the row pair (vy + ky, vy + ky + 1) seen through the kx shift.
     LY 2 (default): rows of 18 nibbles at 24 B (p at nibble 7 + p), one scale per (ci, plane) as the fp8 kernel; the tap
         shifts are funnel shifts and row offsets. One rounding per value: the only layout that is not slower than fp8.
     LY 1: the same rows stored twice, as 5 even-aligned (0-1 .. 8-9) and 4 odd-aligned (1-2 .. 7-8, used by ky = 1) row
         pairs, one scale per pair (36 positions incl. the halo): scale per (ci, 32 positions) as specified, ~10% slower.
     LY 0: every staged plane quantised into its 27 shifted blocks (each pair in the 3 kx windows, 16 B + one scale each,
         432 B / plane): exact per-32-position blocks, needed when the block is transformed (Hadamard) or SR'd.
     LY 3 (UFSM_F4_HAD_W=2, the fast Hadamard): transform I2 x H16 diag(s16) (H16 along x inside each row, rows untouched)
         on both operands, so each row's 3 kx windows are transformed once (30 windows per plane, not 27 x 32-blocks) and
         stored pre-shifted [kx][row][8 B] with one scale per (ci, plane); epilogue 1/16.
   MX inputs (mx4 / mx8 x, mx8 gy, the --fp4 training storage) are read 4 / 8 voxels at a time with the index math hoisted
   and the scale bytes in one vector load. x SR (UFSM_F4_SRX) uses the same direct-nibble rounding, one hash + xorshift.
   Optional fixed-sign Hadamard (had): every 32-element block of both operands is multiplied by H32 diag(s) (s: the fixed
         sign vector F4W_SGN) before its scale / rounding; the element order inside a block (k = 16 * row + x) is the same
         for both, so H^T H = 32 I gives the exact product * 32, undone in the epilogue.
   Smem: X ring e2m1 [ci][4 planes][27 blocks][16 B] (ci stride 1744 B = 436 words == 20 mod 32: the 8 channels x 4 words
   of a fragment load hit 32 banks), scale pairs [ci][4][32] u16 (byte 0 = block P, byte 1 = block P + 3 = the next pair of
   the same alignment; ci stride 260 B), GY e2m1 [co][8 pair blocks][16 B] (co stride 144 B, ldmatrix conflict-free) +
   scales [co][8], and per warp a [10 rows][18] fp32 staging plane (LY 0). LY 1 uses the same strides with 9 pairs x 2 rows x
   24 B per plane (pair P at 48 P, scale-pair byte 1 = pair P + 1); LY 2 10 rows x 24 B per plane, ci stride 976 B (also
   20 mod 32 words), which leaves room for NT 3 at 2 blocks / SM. */
#define X4_PS 432
#define X4_CS 1744
#define X4_SCS 130   /* u16 per channel of the scale-pair table (4 planes x 32 + 2 pad) */
#define G4_CS 144
#define F4W_SGN 0x9c6d2a73u
#define F4W_SGN16 0x2a73u   /* sign vector of the H16 variant (UFSM_F4_HAD_W=2), over the 16 x positions of a row window */
/* stochastic rounding onto the e2m1 grid with a 16-bit uniform (two per hash): P(up) = ceil(frac * 65536) / 65536, so the
   bias is below 2^-16 of a grid step; branch-light (the grid step is 0.5 / 1 / 2 on [0, 2) / [2, 4) / [4, 6]). */
/* the same rounding returning the e2m1 nibble directly: the grid magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} are the codes 0..7 and
   t(a) = 2a / a + 2 / a/2 + 4 on [0, 2) / [2, 4) / [4, 6] is linear between consecutive grid points, so floor(t + u) is exact
   stochastic rounding (P(up) = frac to 2^-16) and needs no conversion instruction */
__device__ __forceinline__ unsigned sr_e2m1_nib(float v, unsigned u16) {   /* |v| <= 6 (block-scaled); t concave -> min of its 3 lines */
    const float a = fabsf(v);
    const float tt = fminf(fminf(a + a, a + 2.f), fmaf(a, 0.5f, 4.f));
    const float u = __uint_as_float(0x3f800000u | (u16 << 7)) - 1.f;   /* u16 / 65536 exactly, no int -> float conversion */
    return min((unsigned)(tt + u), 7u) | (__float_as_uint(v) >> 28 & 8u);
}
__device__ __forceinline__ float sr_e2m1_u16(float v, unsigned u16) {
    const float a = fminf(fabsf(v), 6.f);
    const float inv = a < 2.f ? 2.f : a < 4.f ? 1.f : 0.5f, fl = floorf(a * inv);
    const float up = (float)u16 < (a * inv - fl) * 65536.f ? 1.f : 0.f;
    return copysignf((fl + up) * __frcp_rn(inv), v);
}   /* fixed random sign vector of the weight-gradient Hadamard (bit k set: element k negated) */
__device__ __forceinline__ void had_lane(float &v, float p, bool hi) { v = hi ? p - v : v + p; }
template <int MT, int NT, int LY, typename T, typename TG>   /* LY 0: 27 shifted blocks, 1: row pairs (2 pairings), 2: rows, one scale per (ci, plane) */
__global__ void __launch_bounds__(288, 2) conv_bwd_w_f4_k(const T *__restrict__ x, const TG *__restrict__ gy, float *__restrict__ gw, float *__restrict__ gb,
                                                       int N, int Ci, int D, int H, int W, int Co, gnp_t gp, split_t sp, int ZC, int had) {
    constexpr int CH = 8 * NT, BMo = 16 * MT;
    constexpr int XPS = LY >= 2 ? 240 : X4_PS, XCS = LY >= 2 ? 976 : X4_CS;   /* x bytes per plane / per channel (LY 2: 10 rows x 24 B, LY 3: 3 kx x 10 rows x 8 B) */
    extern __shared__ __align__(128) unsigned char smem_raw[];
    uint8_t *sxq = smem_raw;                                          /* [CH][XCS] */
    unsigned short *sxp = (unsigned short *)(sxq + CH * XCS);       /* [CH][X4_SCS] */
    uint8_t *sg = (uint8_t *)(sxp + CH * X4_SCS);                     /* [BMo][G4_CS] */
    uint8_t *sgs = sg + BMo * G4_CS;                                  /* [BMo][8] */
    float *sbias = (float *)(sgs + BMo * 8);                          /* [BMo] */
    uint8_t *wsc = (uint8_t *)(sbias + BMo);                          /* [9 warps][32] */
    float *scr = (float *)(wsc + 9 * 32);                             /* [9 warps][180], LY 0 only */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int ci0 = blockIdx.x * CH, co0 = blockIdx.y * BMo;
    int bz = blockIdx.z;
    const int nxt = (W + 15) / 16, nyt = (H + 7) / 8, nzt = (D + 1) / 2;
    const int ox0 = (bz % nxt) * 16; bz /= nxt;
    const int oy0 = (bz % nyt) * 8; bz /= nyt;
    const int nzc = (nzt + ZC - 1) / ZC;
    const int zc = bz % nzc; const int n = bz / nzc;
    const size_t plane = (size_t)D * H * W;
    const int Cx = sp.x2 ? sp.c_split : Ci;
    const int kz = warp / 3, ky = warp % 3;
    const bool do_bias = gb && blockIdx.x == 0;
    float *ws = scr + warp * 180;
    uint8_t *wscw = wsc + warp * 32;
    float acc[3][MT][NT][4];
#pragma unroll
    for (int a = 0; a < 3; a++) for (int m = 0; m < MT; m++) for (int q = 0; q < NT; q++) for (int k = 0; k < 4; k++) acc[a][m][q][k] = 0.f;
    if (threadIdx.x < BMo) sbias[threadIdx.x] = 0.f;
    const bool vec = (W & 3) == 0;
    const int zt_begin = zc * ZC;
    for (int zt = zt_begin; zt < nzt && zt < (zc + 1) * ZC; zt++) {
        const int oz0 = zt * 2;
        const int np = zt == zt_begin ? 4 : 2, gz_first = zt == zt_begin ? oz0 - 1 : oz0 + 1;
        __syncthreads();
        /* X: warp per (channel, plane). Phase 1: the 10 x 18 positions (GN+SiLU applied) into the warp's fp32 plane,
           position p of a row = x ox0 - 1 + p. Phase 2: the 27 shifted 32-position blocks, 4 lanes x 8 values each. */
        constexpr int U = LY == 1 || LY == 2 ? 2 : 1;   /* x tasks in flight per warp: the loads of both are issued before either is quantised */
        const int NXT = CH * np;
        for (int task0 = warp; task0 < NXT; task0 += 9 * U) {
            float4 V4[U][2]; float VS[U];
#pragma unroll
            for (int u = 0; u < U; u++) {
            const int task = task0 + 9 * u;
            int k = task / np, gz = gz_first + task % np, ci = ci0 + k;
            const bool ok = task < NXT && ci < Ci && gz >= 0 && gz < D;
            chan_t c = make_chan(ok ? ci : Ci, Ci, Cx, n, plane, x, sp, gp, N);
            const T *xc = ok && !IS_MX(T) ? (const T *)c.p + (size_t)gz * H * W : x;
            const bool G = (gp.G != 0 || sp.gp2.G != 0) && c.g, el = IS_MX(T);
            __syncwarp();
            float4 *v4 = V4[u]; float &vs = VS[u]; vs = 0.f;
#pragma unroll
            for (int i = 0; i < 2; i++) {
                int f = lane + 32 * i, row = f >> 2, gyy = oy0 - 1 + row, gx = ox0 + 4 * (f & 3);
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (ok && f < 40 && gyy >= 0 && gyy < H) {
                    const T *src = xc + (size_t)gyy * W + gx;
                    if constexpr (IS_MX(T)) { if (gx < W) v = ldc4_mx<T>(c, cof(c, gz, gyy, gx, H, W), min(4, W - gx), vec); }
                    else if (vec) { if (gx < W) v = ldx4(src); }
                    else { if (gx < W) v.x = ldx(src, 0); if (gx + 1 < W) v.y = ldx(src, 1); if (gx + 2 < W) v.z = ldx(src, 2); if (gx + 3 < W) v.w = ldx(src, 3); }
                    v.x = gx < W ? act_ab(v.x, c.a, c.b, G) : 0.f; v.y = gx + 1 < W ? act_ab(v.y, c.a, c.b, G) : 0.f;
                    v.z = gx + 2 < W ? act_ab(v.z, c.a, c.b, G) : 0.f; v.w = gx + 3 < W ? act_ab(v.w, c.a, c.b, G) : 0.f;
                }
                v4[i] = v;
                if ((LY == 0 || LY == 3) && f < 40) { float *d = ws + row * 18 + 1 + 4 * (f & 3); d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w; }
            }
            {   /* halo: lane < 20 -> row lane >> 1, side lane & 1 (x = ox0 - 1 or ox0 + 16) */
                int row = lane >> 1, gyy = oy0 - 1 + row, gx = (lane & 1) ? ox0 + 16 : ox0 - 1;
                if (ok && lane < 20 && gyy >= 0 && gyy < H && gx >= 0 && gx < W) vs = act_ab(el ? ldc<T>(c, cof(c, gz, gyy, gx, H, W)) : ldx(xc, (size_t)gyy * W + gx), c.a, c.b, G);
                if ((LY == 0 || LY == 3) && lane < 20) ws[row * 18 + ((lane & 1) ? 17 : 0)] = vs;
            }
            }
#pragma unroll
            for (int u = 0; u < U; u++) {
            const int task = task0 + 9 * u;
            if (task >= NXT) break;
            const int k = task / np, gz = gz_first + task % np, slot = (gz + 1) & 3, ci = ci0 + k;
            const float4 *v4 = V4[u]; const float vs = VS[u];
            __syncwarp();
            uint8_t *dst = sxq + k * XCS + slot * XPS;
            if constexpr (LY == 3) {   /* H16 variant: lane (kx = lane / 10, row = lane % 10) < 30 transforms its 16-position window
                                      (x ox0 - 1 + kx ..) with H16 diag(s16), one scale per (ci, plane), stored pre-shifted [kx][row][8 B] */
                const unsigned FM = 0xffffffffu;
                const bool lok = lane < 30;
                const int kx = lane / 10, r = lane % 10;
                float v[16];
                const float *src = ws + (lok ? r : 0) * 18 + (lok ? kx : 0);
#pragma unroll
                for (int j = 0; j < 16; j++) v[j] = lok ? ((F4W_SGN16 >> j) & 1u ? -src[j] : src[j]) : 0.f;
#pragma unroll
                for (int st = 1; st < 16; st <<= 1)
#pragma unroll
                    for (int j = 0; j < 16; j++) if (!(j & st)) { float a0 = v[j], a1 = v[j | st]; v[j] = a0 + a1; v[j | st] = a0 - a1; }
                unsigned a = 0u;
#pragma unroll
                for (int j = 0; j < 16; j++) a = amax_u(a, v[j]);
#pragma unroll
                for (int o = 16; o; o >>= 1) a = max(a, __shfl_xor_sync(FM, a, o));
                const int e = mx_exp(__uint_as_float(a), 1.f / 6.f);
                const float m = exp2i(-e);
                uint2 u;
                if ((had & 2) && sp.sr) {   /* x SR: one hash + 7 xorshift steps per 16 values */
                    const uint64_t vid = ((((uint64_t)n * Ci + ci) * D + gz) * gridDim.z + blockIdx.z) * 32 + lane;
                    uint32_t h = sr_hash(sp.sr ^ 0x6a09e667u, vid);
                    unsigned w2[2] = {0u, 0u};
#pragma unroll
                    for (int j = 0; j < 16; j += 2) {
                        if (j) { h ^= h << 13; h ^= h >> 17; h ^= h << 5; }   /* xorshift32 after the hash (h != 0 w.p. 1 - 2^-32) */
                        w2[j >> 3] |= (sr_e2m1_nib(v[j] * m, h & 0xffffu) | sr_e2m1_nib(v[j + 1] * m, h >> 16) << 4) << (4 * (j & 7));
                    }
                    u = make_uint2(w2[0], w2[1]);
                } else u = make_uint2(cvt_e2m1x8(v, m), cvt_e2m1x8(v + 8, m));
                if (lok) *(uint2 *)(dst + (kx * 10 + r) * 8) = u;
                if (lane == 0) sxp[k * X4_SCS + slot * 32] = (unsigned short)((e + 127) * 0x101);
            } else
            if constexpr (LY == 2) {   /* rows at 24 B (p at nibble 7 + p), one scale per (ci, plane): one rounding per value, as fp8 */
                const unsigned FM = 0xffffffffu;
                unsigned a = amax_u(0u, vs);
#pragma unroll
                for (int i = 0; i < 2; i++) a = amax_u(amax_u(amax_u(amax_u(a, v4[i].x), v4[i].y), v4[i].z), v4[i].w);
#pragma unroll
                for (int o = 16; o; o >>= 1) a = max(a, __shfl_xor_sync(FM, a, o));
                const int e = mx_exp(__uint_as_float(a), 1.f / 6.f);
                const float m = exp2i(-e);
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    const int f = lane + 32 * i, r = f >> 2, cq = f & 3;
                    const unsigned h16 = (unsigned)cvt_e2m1x2(v4[i].x * m, v4[i].y * m) | ((unsigned)cvt_e2m1x2(v4[i].z * m, v4[i].w * m) << 8);
                    const unsigned hn = __shfl_down_sync(FM, h16, 1);
                    if (f < 40 && !(cq & 1)) *(unsigned *)(dst + r * 24 + 4 * (1 + (cq >> 1))) = h16 | (hn << 16);
                }
                if (lane < 20) { const unsigned nib = (unsigned)cvt_e2m1x2(vs * m, 0.f) & 15u; *(unsigned *)(dst + (lane >> 1) * 24 + ((lane & 1) ? 12 : 0)) = (lane & 1) ? nib : nib << 28; }
                if (lane == 0) sxp[k * X4_SCS + slot * 32] = (unsigned short)((e + 127) * 0x101);
            } else
            if constexpr (LY == 1) {   /* paired layout, from registers: lane (i, f = lane + 32 i < 40) holds row f >> 2, positions
                                     p = 1 + 4 (f & 3) .. 4 + 4 (f & 3) (= nibbles 8 + 4 c .., word 1 + c / 2); lane < 20 the halo
                                     of row lane >> 1 (p = 0: nibble 7 of word 0, p = 17: nibble 0 of word 3). Row amax -> the
                                     even-pair (rows 2e, 2e + 1) and odd-pair (2o + 1, 2o + 2) amax by shuffles; each value is
                                     rounded once per pairing. */
                const unsigned FM = 0xffffffffu;
                unsigned ra[2];
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    unsigned a = amax_u(amax_u(amax_u(amax_u(0u, v4[i].x), v4[i].y), v4[i].z), v4[i].w);
                    a = max(a, __shfl_xor_sync(FM, a, 1)); ra[i] = max(a, __shfl_xor_sync(FM, a, 2));
                }
                unsigned hm = amax_u(0u, vs); hm = max(hm, __shfl_xor_sync(FM, hm, 1));   /* halo amax of row lane >> 1 */
                ra[0] = max(ra[0], __shfl_sync(FM, hm, (lane >> 2) * 2));
                ra[1] = max(ra[1], __shfl_sync(FM, hm, min(16 + (lane >> 2) * 2, 31)));
                unsigned amE[2], amO[2];
#pragma unroll
                for (int i = 0; i < 2; i++) amE[i] = max(ra[i], __shfl_xor_sync(FM, ra[i], 4));
                {
                    const int r = lane >> 2;   /* i = 0 row */
                    const unsigned n0 = __shfl_sync(FM, ra[0], (r & 1) ? min(lane + 4, 31) : max(lane - 4, 0));
                    const unsigned x01 = __shfl_sync(FM, ra[1], lane & 3), x10 = __shfl_sync(FM, ra[0], 28 + (lane & 3));
                    amO[0] = max(ra[0], lane >= 28 ? x01 : n0);   /* row 7 pairs with row 8 (i = 1) */
                    amO[1] = max(ra[1], x10);                     /* row 8 (lanes 0..3) pairs with row 7; row 9 unused */
                }
                /* halo lanes: the pair amaxes of their row (row hr from lane 4 hr of i = 0, or 4 (hr - 8) of i = 1) */
                const int hr = min(lane >> 1, 9);
                const unsigned hE0 = __shfl_sync(FM, amE[0], (hr & 7) * 4), hE1 = __shfl_sync(FM, amE[1], (hr & 7) * 4);
                const unsigned hO0 = __shfl_sync(FM, amO[0], (hr & 7) * 4), hO1 = __shfl_sync(FM, amO[1], (hr & 7) * 4);
                const unsigned hamE = hr < 8 ? hE0 : hE1, hamO = hr < 8 ? hO0 : hO1;
#pragma unroll
                for (int pz = 0; pz < 2; pz++) {
#pragma unroll
                    for (int i = 0; i < 2; i++) {
                        const int f = lane + 32 * i, r = f >> 2, cq = f & 3;
                        const int e = mx_exp(__uint_as_float(pz ? amO[i] : amE[i]), 1.f / 6.f);
                        const float m = exp2i(-e);
                        const unsigned h16 = (unsigned)cvt_e2m1x2(v4[i].x * m, v4[i].y * m) | ((unsigned)cvt_e2m1x2(v4[i].z * m, v4[i].w * m) << 8);
                        const unsigned hn = __shfl_down_sync(FM, h16, 1);
                        const bool wr = f < 40 && (!pz || (r >= 1 && r <= 8));
                        const int pair = pz ? 5 + ((r - 1) >> 1) : r >> 1, rin = pz ? (r - 1) & 1 : r & 1;
                        if (wr && !(cq & 1)) *(unsigned *)(dst + pair * 48 + rin * 24 + 4 * (1 + (cq >> 1))) = h16 | (hn << 16);
                        if (wr && cq == 0 && rin == 0) wscw[pair] = (uint8_t)(e + 127);
                    }
                    {   /* halo */
                        const int r = hr, sd = lane & 1;
                        const int e = mx_exp(__uint_as_float(pz ? hamO : hamE), 1.f / 6.f);
                        const unsigned nib = (unsigned)cvt_e2m1x2(vs * exp2i(-e), 0.f) & 15u;
                        const bool wr = lane < 20 && (!pz || (r >= 1 && r <= 8));
                        const int pair = pz ? 5 + ((r - 1) >> 1) : r >> 1, rin = pz ? (r - 1) & 1 : r & 1;
                        if (wr) *(unsigned *)(dst + pair * 48 + rin * 24 + (sd ? 12 : 0)) = sd ? nib : nib << 28;
                    }
                }
                __syncwarp();
                if (lane < 9) sxp[k * X4_SCS + slot * 32 + lane] = (unsigned short)(wscw[lane] | (lane != 4 && lane < 8 ? (unsigned)wscw[lane + 1] << 8 : 0u));
            } else {
#pragma unroll
            for (int r4 = 0; r4 < 4; r4++) {
                const int task4 = r4 * 32 + lane, b = task4 >> 2, w = task4 & 3;
                const bool bok = b < 27;
                const int ev = b < 15, pr = ev ? b / 3 : (b - 15) / 3, kx = ev ? b % 3 : (b - 15) % 3, r0 = ev ? 2 * pr : 2 * pr + 1;
                float v[8];
                const float *src = ws + (r0 + (w >> 1)) * 18 + kx + 8 * (w & 1);
#pragma unroll
                for (int j = 0; j < 8; j++) v[j] = bok ? src[j] : 0.f;
                if (had & 1) {
#pragma unroll
                    for (int j = 0; j < 8; j++) if ((F4W_SGN >> (8 * w + j)) & 1u) v[j] = -v[j];
#pragma unroll
                    for (int s = 1; s < 8; s <<= 1)
#pragma unroll
                        for (int j = 0; j < 8; j++) if (!(j & s)) { float a0 = v[j], a1 = v[j | s]; v[j] = a0 + a1; v[j | s] = a0 - a1; }
#pragma unroll
                    for (int s = 1; s < 4; s <<= 1)
#pragma unroll
                        for (int j = 0; j < 8; j++) had_lane(v[j], __shfl_xor_sync(0xffffffffu, v[j], s), lane & s);
                }
                unsigned amu = 0u;
#pragma unroll
                for (int j = 0; j < 8; j++) amu = amax_u(amu, v[j]);
                amu = max(amu, __shfl_xor_sync(0xffffffffu, amu, 1)); amu = max(amu, __shfl_xor_sync(0xffffffffu, amu, 2));
                const int e = mx_exp(__uint_as_float(amu), 1.f / 6.f);
                unsigned word;
                if ((had & 2) && sp.sr) {   /* UFSM_F4_SRX: stochastic rounding of x too, keyed by (ci, plane, tile, block, k); one hash + 3 remixes */
                    const float mm = exp2i(-e);
                    const uint64_t vid = (((((uint64_t)n * Ci + ci) * D + gz) * gridDim.z + blockIdx.z) * 32 + b) * 32 + 8 * w;
                    uint32_t hh[4];
                    hh[0] = sr_hash(sp.sr ^ 0x6a09e667u, vid);
#pragma unroll
                    for (int i = 1; i < 4; i++) { uint32_t h1 = (hh[i - 1] ^ (hh[i - 1] >> 15)) * 0x2c1b3c6du; h1 ^= h1 >> 12; h1 *= 0x297a2d39u; hh[i] = h1 ^ (h1 >> 15); }
                    word = 0u;
#pragma unroll
                    for (int j = 0; j < 8; j++) word |= sr_e2m1_nib(v[j] * mm, (hh[j >> 1] >> (16 * (j & 1))) & 0xffffu) << (4 * j);
                } else word = cvt_e2m1x8(v, exp2i(-e));
                if (bok) { *(unsigned *)(dst + b * 16 + 4 * w) = word; if (w == 0) wscw[b] = (uint8_t)(e + 127); }
            }
            __syncwarp();
            if (lane < 27) sxp[k * X4_SCS + slot * 32 + lane] = (unsigned short)(wscw[lane] | (lane + 3 < 27 ? (unsigned)wscw[lane + 3] << 8 : 0u));
            }
            }
        }
        /* GY: block (co, pb) = row pair pb of the z-step (rows 2 pb, 2 pb + 1; vz = pb >> 2), 4 lanes x 8 values: lane f holds
           block elements 8 f .. 8 f + 7 (row f >> 1, x 8 (f & 1) ..) = one e2m1 word; a warp task covers 8 blocks */
        for (int task = warp; task < BMo; task += 9) {
            int blk = task * 8 + (lane >> 2), c = blk >> 3, pb = blk & 7, f = lane & 3;
            int row = pb * 2 + (f >> 1), vx = 8 * (f & 1), vz = row >> 3, vy = row & 7;
            int co = co0 + c, oz = oz0 + vz, oy = oy0 + vy, ox = ox0 + vx;
            float q[8];
#pragma unroll
            for (int j = 0; j < 8; j++) q[j] = 0.f;
            if (co < Co && oz < D && oy < H) {
                if constexpr (IS_MX8(TG)) {
                    const size_t vo = ((size_t)oz * H + oy) * W + ox, Sg = (size_t)D * H * W;
                    ldmx8_8(gy, N, Co, Sg, n, co, vo, min(8, W - ox), (W & 7) == 0, q);
                } else {
                    const TG *src = gy + (((size_t)n * Co + co) * D + oz) * H * W + (size_t)oy * W + ox;
                    if (vec) {
#pragma unroll
                        for (int h = 0; h < 2; h++) if (ox + 4 * h < W) { float4 v = ldx4<TG>(src + 4 * h); q[4 * h] = v.x; q[4 * h + 1] = v.y; q[4 * h + 2] = v.z; q[4 * h + 3] = v.w; }
                    } else {
#pragma unroll
                        for (int j = 0; j < 8; j++) if (ox + j < W) q[j] = ldx(src, j);
                    }
                }
            }
            float sm = ((q[0] + q[1]) + (q[2] + q[3])) + ((q[4] + q[5]) + (q[6] + q[7]));
            sm += __shfl_xor_sync(0xffffffffu, sm, 1); sm += __shfl_xor_sync(0xffffffffu, sm, 2);
            if (had & 4) {   /* H16 variant: H16 diag(s16) along x within each row (x = 8 (f & 1) + j) */
#pragma unroll
                for (int j = 0; j < 8; j++) if ((F4W_SGN16 >> (8 * (f & 1) + j)) & 1u) q[j] = -q[j];
#pragma unroll
                for (int s = 1; s < 8; s <<= 1)
#pragma unroll
                    for (int j = 0; j < 8; j++) if (!(j & s)) { float a0 = q[j], a1 = q[j | s]; q[j] = a0 + a1; q[j | s] = a0 - a1; }
#pragma unroll
                for (int j = 0; j < 8; j++) had_lane(q[j], __shfl_xor_sync(0xffffffffu, q[j], 1), lane & 1);
            } else if (had & 1) {
#pragma unroll
                for (int j = 0; j < 8; j++) if ((F4W_SGN >> (8 * f + j)) & 1u) q[j] = -q[j];
#pragma unroll
                for (int s = 1; s < 8; s <<= 1)
#pragma unroll
                    for (int j = 0; j < 8; j++) if (!(j & s)) { float a0 = q[j], a1 = q[j | s]; q[j] = a0 + a1; q[j | s] = a0 - a1; }
#pragma unroll
                for (int s = 1; s < 4; s <<= 1)
#pragma unroll
                    for (int j = 0; j < 8; j++) had_lane(q[j], __shfl_xor_sync(0xffffffffu, q[j], s), lane & s);
            }
            unsigned amu = 0u;
#pragma unroll
            for (int j = 0; j < 8; j++) amu = amax_u(amu, q[j]);
            amu = max(amu, __shfl_xor_sync(0xffffffffu, amu, 1)); amu = max(amu, __shfl_xor_sync(0xffffffffu, amu, 2));
            const int e = mx_exp(__uint_as_float(amu), 1.f / 6.f);
            const float m = exp2i(-e);
            unsigned word;
            if (sp.sr) {   /* exact stochastic rounding of the gradient operand, keyed by the element (block slot): one hash + 3 remixes */
                const uint64_t vid = ((((uint64_t)n * Co + co) * D + oz) * H + oy) * (uint64_t)W + ox;
                uint32_t hh[4];
                hh[0] = sr_hash(sp.sr, vid);
#pragma unroll
                for (int i = 1; i < 4; i++) { uint32_t h1 = (hh[i - 1] ^ (hh[i - 1] >> 15)) * 0x2c1b3c6du; h1 ^= h1 >> 12; h1 *= 0x297a2d39u; hh[i] = h1 ^ (h1 >> 15); }
                word = 0u;
#pragma unroll
                for (int j = 0; j < 8; j++) word |= sr_e2m1_nib(q[j] * m, (hh[j >> 1] >> (16 * (j & 1))) & 0xffffu) << (4 * j);
            } else word = cvt_e2m1x8(q, m);
            *(unsigned *)(sg + c * G4_CS + pb * 16 + 4 * f) = word;
            if (f == 0) { sgs[c * 8 + pb] = (uint8_t)(e + 127); if (do_bias) atomicAdd(&sbias[c], sm); }
        }
        __syncthreads();
#pragma unroll
        for (int vz = 0; vz < 2; vz++) {
            const int slot = (oz0 + vz + kz) & 3;
#pragma unroll
            for (int kk = 0; kk < 2; kk++) {
                const int ks = vz * 2 + kk, r0 = 4 * kk + ky;
                const int P0 = (ky & 1) ? 15 + ((r0 - 1) >> 1) * 3 : (r0 >> 1) * 3;   /* first block (kx = 0) of the K step's row pairs */
                unsigned af[MT][4], sa[MT];
#pragma unroll
                for (int m = 0; m < MT; m++) {
                    int mat = lane >> 3, co = m * 16 + (mat & 1) * 8 + (lane & 7);
                    ldsm_x4(af[m], sg + co * G4_CS + (2 * ks + (mat >> 1)) * 16);
                    sa[m] = *(const unsigned short *)(sgs + (m * 16 + g + 8 * (t & 1)) * 8 + 2 * ks);
                }
#pragma unroll
                if constexpr (LY == 3) {
#pragma unroll
                    for (int q = 0; q < NT; q++) {
                        const uint8_t *xb = sxq + (q * 8 + g) * XCS + slot * XPS + (r0 + (t >> 1)) * 8 + 4 * (t & 1);
                        const unsigned sb = sxp[(q * 8 + g) * X4_SCS + slot * 32];
#pragma unroll
                        for (int kx = 0; kx < 3; kx++) {
                            unsigned b[2] = {*(const unsigned *)(xb + kx * 80), *(const unsigned *)(xb + kx * 80 + 16)};   /* rows + 2: +16 B */
#pragma unroll
                            for (int m = 0; m < MT; m++) mma_f4(acc[kx][m][q], af[m], b, sa[m], sb);
                        }
                    }
                } else
                if constexpr (LY >= 1) {
                    const int P = LY == 2 ? 0 : (ky & 1) ? 5 + ((r0 - 1) >> 1) : r0 >> 1;   /* LY 1: the K step's first row pair; the second is P + 1 */
                    const int ro = LY == 2 ? r0 * 24 : P * 48;
#pragma unroll
                    for (int q = 0; q < NT; q++) {
                        const unsigned *xr = (const unsigned *)(sxq + (q * 8 + g) * XCS + slot * XPS + ro + (t >> 1) * 24) + (t & 1);
                        const unsigned w0 = xr[0], w1 = xr[1], w2 = xr[2], c0 = xr[12], c1 = xr[13], c2 = xr[14];   /* next pair: +48 B */
                        const unsigned sb = sxp[(q * 8 + g) * X4_SCS + slot * 32 + P];
                        unsigned b[3][2] = {{__funnelshift_r(w0, w1, 28), __funnelshift_r(c0, c1, 28)}, {w1, c1}, {__funnelshift_r(w1, w2, 4), __funnelshift_r(c1, c2, 4)}};
#pragma unroll
                        for (int kx = 0; kx < 3; kx++)
#pragma unroll
                            for (int m = 0; m < MT; m++) mma_f4(acc[kx][m][q], af[m], b[kx], sa[m], sb);
                    }
                } else
#pragma unroll
                for (int q = 0; q < NT; q++) {
                    const uint8_t *xb = sxq + (q * 8 + g) * XCS + slot * XPS + 4 * t;
                    const unsigned short *xs2 = sxp + (q * 8 + g) * X4_SCS + slot * 32;
#pragma unroll
                    for (int kx = 0; kx < 3; kx++) {
                        const int P = P0 + kx;
                        unsigned b[2] = {*(const unsigned *)(xb + P * 16), *(const unsigned *)(xb + (P + 3) * 16)};
                        const unsigned sb = xs2[P];
#pragma unroll
                        for (int m = 0; m < MT; m++) mma_f4(acc[kx][m][q], af[m], b, sa[m], sb);
                    }
                }
            }
        }
    }
    if (do_bias) { __syncthreads(); if (threadIdx.x < BMo && co0 + (int)threadIdx.x < Co) atomicAdd(&gb[co0 + threadIdx.x], sbias[threadIdx.x]); }
    const float osc = (had & 4) ? 1.f / 16.f : (had & 1) ? 1.f / 32.f : 1.f;
#pragma unroll
    for (int kx = 0; kx < 3; kx++) {
        int tap = (kz * 3 + ky) * 3 + kx;
#pragma unroll
        for (int m = 0; m < MT; m++)
#pragma unroll
            for (int q = 0; q < NT; q++) {
                int ci = ci0 + q * 8 + 2 * t;
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    int co = co0 + m * 16 + g + 8 * h;
                    if (co >= Co) continue;
                    if (ci < Ci) atomicAdd(&gw[((size_t)co * Ci + ci) * 27 + tap], acc[kx][m][q][2 * h] * osc);
                    if (ci + 1 < Ci) atomicAdd(&gw[((size_t)co * Ci + ci + 1) * 27 + tap], acc[kx][m][q][2 * h + 1] * osc);
                }
            }
    }
}
template <int MT, int NT, typename T, typename TG> static void launch_bw4(dim3 grid, size_t smem, const void *x, shape5 xs, const TG *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp, int ZC, int had, int lay) {
    static int attr[8];
    if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)conv_bwd_w_f4_k<MT, NT, 0, T, TG>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); cudaFuncSetAttribute((const void *)conv_bwd_w_f4_k<MT, NT, 1, T, TG>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); cudaFuncSetAttribute((const void *)conv_bwd_w_f4_k<MT, NT, 2, T, TG>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); cudaFuncSetAttribute((const void *)conv_bwd_w_f4_k<MT, NT, 3, T, TG>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
    if (lay == 3) conv_bwd_w_f4_k<MT, NT, 3, T, TG><<<grid, 288, smem>>>((const T *)x, gy, gw, gb, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, gp, sp, ZC, had);
    else if (lay == 2) conv_bwd_w_f4_k<MT, NT, 2, T, TG><<<grid, 288, smem>>>((const T *)x, gy, gw, gb, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, gp, sp, ZC, had);
    else if (lay == 1) conv_bwd_w_f4_k<MT, NT, 1, T, TG><<<grid, 288, smem>>>((const T *)x, gy, gw, gb, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, gp, sp, ZC, had);
    else conv_bwd_w_f4_k<MT, NT, 0, T, TG><<<grid, 288, smem>>>((const T *)x, gy, gw, gb, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, gp, sp, ZC, had);
}
template <typename T, typename TG> static void bwd_w_f4_t(const void *x, shape5 xs, const TG *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp, int had) {
    static int nt_env = -2, mt_env = -2, zc_env = -2;
    if (nt_env == -2) { nt_env = getenv("UFSM_F4W_NT") ? atoi(getenv("UFSM_F4W_NT")) : -1; mt_env = getenv("UFSM_F4W_MT") ? atoi(getenv("UFSM_F4W_MT")) : -1; zc_env = getenv("UFSM_F4W_ZC") ? atoi(getenv("UFSM_F4W_ZC")) : -1; }
    static int lay_env = -1;
    if (lay_env < 0) lay_env = getenv("UFSM_F4W_LAYOUT") ? atoi(getenv("UFSM_F4W_LAYOUT")) : 1;   /* 1: per-32 x scales; 2 (one scale per (ci, plane), faster) failed the seed-1 stair: 0.201 vs 0.310 */
    /* plain mode (no Hadamard / x SR): 1 = row pairs, scale per (ci, 32 positions); 2 = rows, scale per (ci, plane) (fp8's
       granularity, one rounding per value); 0 = the 27 shifted blocks (always used with the Hadamard or x SR) */
    const int lay = (had & 4) ? 3 : (had & 3) ? 0 : lay_env;   /* had bit 2: the H16 variant (UFSM_F4_HAD_W=2) -> LY 3 */
    int MT = ys.c >= 32 ? 2 : 1;
    int NT = xs.c <= 8 ? 1 : lay >= 2 && MT == 1 && xs.c % 24 == 0 ? 3 : 2;   /* NT 3 (as fp8 on dec0.c1) only fits 2 blocks / SM with the LY 2 tile */
    if (nt_env > 0) NT = nt_env;
    if (mt_env > 0) MT = mt_env;
    size_t smem = (size_t)8 * NT * ((lay >= 2 ? 976 : X4_CS) + 2 * X4_SCS) + 16 * MT * (G4_CS + 8 + 4) + 9 * 32 + (lay == 0 || lay == 3 ? 9 * 180 * 4 : 0);
    int nzt = nblk_(ys.d, 2), base = (int)(((xs.c + 8 * NT - 1) / (8 * NT)) * ((ys.c + 16 * MT - 1) / (16 * MT)) * nblk_(ys.w, 16) * nblk_(ys.h, 8) * ys.n);
    int ZC = nzt < 12 ? nzt : 12;
    while (ZC > 1 && (size_t)base * nblk_(nzt, ZC) < 72) ZC--;
    if (zc_env > 0) ZC = zc_env;
    int nzc = (nzt + ZC - 1) / ZC;
    dim3 grid((xs.c + 8 * NT - 1) / (8 * NT), (ys.c + 16 * MT - 1) / (16 * MT), (unsigned)(nblk_(ys.w, 16) * nblk_(ys.h, 8) * nzc * ys.n));
    switch (MT * 10 + NT) {
    case 11: launch_bw4<1, 1, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, gp, sp, ZC, had, lay); break;
    case 12: launch_bw4<1, 2, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, gp, sp, ZC, had, lay); break;
    case 13: launch_bw4<1, 3, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, gp, sp, ZC, had, lay); break;
    case 21: launch_bw4<2, 1, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, gp, sp, ZC, had, lay); break;
    case 22: launch_bw4<2, 2, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, gp, sp, ZC, had, lay); break;
    default: fprintf(stderr, "lp_bwd_w_f4: bad MT/NT %d/%d\n", MT, NT); abort();
    }
}
extern "C" int lp_bwd_w_f4(const void *x, int xbf, shape5 xs, const void *gy, int gybf, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp, int had) {
    if (gybf == 4) { fprintf(stderr, "lp_bwd_w_f4: fp4 gradients are not supported\n"); abort(); }
    if (xbf == 4 && gybf == 3) bwd_w_f4_t<mx4_t, mx8_t>(x, xs, (const mx8_t *)gy, ys, gw, gb, gp, sp, had);
    else if (xbf == 4 && gybf == 2) bwd_w_f4_t<mx4_t, __half>(x, xs, (const __half *)gy, ys, gw, gb, gp, sp, had);
    else if (xbf == 4 && gybf == 1) bwd_w_f4_t<mx4_t, bf16>(x, xs, (const bf16 *)gy, ys, gw, gb, gp, sp, had);
    else if (xbf == 4) bwd_w_f4_t<mx4_t, float>(x, xs, (const float *)gy, ys, gw, gb, gp, sp, had);
    else if (xbf == 3 && gybf == 3) bwd_w_f4_t<mx8_t, mx8_t>(x, xs, (const mx8_t *)gy, ys, gw, gb, gp, sp, had);
    else if (xbf == 3 && gybf == 2) bwd_w_f4_t<mx8_t, __half>(x, xs, (const __half *)gy, ys, gw, gb, gp, sp, had);
    else if (xbf == 3 && gybf == 1) bwd_w_f4_t<mx8_t, bf16>(x, xs, (const bf16 *)gy, ys, gw, gb, gp, sp, had);
    else if (xbf == 3) bwd_w_f4_t<mx8_t, float>(x, xs, (const float *)gy, ys, gw, gb, gp, sp, had);
    else if (xbf == 2 && gybf) bwd_w_f4_t<__half, __half>(x, xs, (const __half *)gy, ys, gw, gb, gp, sp, had);
    else if (xbf == 2) bwd_w_f4_t<__half, float>(x, xs, (const float *)gy, ys, gw, gb, gp, sp, had);
    else if (xbf && gybf) bwd_w_f4_t<bf16, bf16>(x, xs, (const bf16 *)gy, ys, gw, gb, gp, sp, had);
    else if (xbf) bwd_w_f4_t<bf16, float>(x, xs, (const float *)gy, ys, gw, gb, gp, sp, had);
    else if (gybf) { fprintf(stderr, "lp_bwd_w_f4: bf16 gradient with fp32 activations is not supported\n"); abort(); }
    else bwd_w_f4_t<float, float>(x, xs, (const float *)gy, ys, gw, gb, gp, sp, had);
    LPCK();
    return 0;
}
/* test probe: mean of n direct-nibble stochastic roundings of v (grid units, |v| <= 6) */
__global__ void sr_nib_probe_k(float v, size_t n, double *acc) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    atomicAdd(acc, (double)dec_e2m1n(sr_e2m1_nib(v, sr_hash(0x7654321u, i) & 0xffffu)));
}
extern "C" double lp_sr_e2m1_nib_mean(float v, size_t n) {
    double *d; cudaMalloc(&d, sizeof(double)); cudaMemset(d, 0, sizeof(double));
    sr_nib_probe_k<<<nblk_(n, 256), 256>>>(v, n, d);
    double h = 0; cudaMemcpy(&h, d, sizeof(double), cudaMemcpyDeviceToHost); cudaFree(d); LPCK();
    return h / (double)n;
}
