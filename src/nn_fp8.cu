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
/* plane-major accessors are never used on MX tensors (the kernels branch on IS_MX8); stubs keep the instantiations complete */
template <> __device__ __forceinline__ float ldx<mx8_t>(const mx8_t *, size_t) { return 0.f; }
template <> __device__ __forceinline__ void stx<mx8_t>(mx8_t *, size_t, float) {}
template <> __device__ __forceinline__ void stx2<mx8_t>(mx8_t *, size_t, float, float) {}
template <> __device__ __forceinline__ float4 ldx4<mx8_t>(const mx8_t *) { return make_float4(0.f, 0.f, 0.f, 0.f); }
template <> __device__ __forceinline__ void ld8x<mx8_t>(const mx8_t *, float *o) { for (int i = 0; i < 8; i++) o[i] = 0.f; }
template <typename T> struct is_mx8_s { static constexpr bool v = false; };
template <> struct is_mx8_s<mx8_t> { static constexpr bool v = true; };
#define IS_MX8(T) (is_mx8_s<T>::v)
__host__ __device__ __forceinline__ int mx_bw(int C) { return C <= 16 ? 16 : 32; }
__host__ __device__ __forceinline__ int mx_nb(int C) { int bw = mx_bw(C); return (C + bw - 1) / bw; }
extern "C" size_t lp_mx8_bytes(int N, int C, size_t S) { return (size_t)N * mx_nb(C) * S * (mx_bw(C) + 1); }
__device__ __forceinline__ float2 dec_e4m3x2(unsigned short v) {   /* low byte -> .x */
    unsigned r; asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(v));
    return __half22float2(*(__half2 *)&r);
}
__device__ __forceinline__ float dec_e4m3(unsigned b) { return dec_e4m3x2((unsigned short)(b & 0xff)).x; }
/* padded input-channel index -> real channel (-1 = zero padding). With MX-stored inputs the x and x2 segments are padded
   to 32 channels each (CxP = pad32(Cx)) so that a 32-channel chunk never straddles two tensors or two stored blocks. */
__host__ __device__ __forceinline__ int seg_ci(int cp, int Ci, int Cx, int CxP) { if (cp < CxP) return cp < Cx ? cp : -1; int c = Cx + cp - CxP; return c < Ci ? c : -1; }
/* per-channel staging descriptor: source plane pointer (nullptr = zero channel) and the GN+SiLU affine a*x + b;
   for MX tensors p points at this channel's byte in block row 0 (element at voxel v: p[v * bw]) and sp at the block's scale row */
typedef struct { const void *p; const uint8_t *sp; float a, b; int bw; } chan_t;
template <typename T>
__device__ __forceinline__ chan_t make_chan(int ci, int Ci, int Cx, int n, size_t plane, const T *x, const split_t &sp, const gnp_t &gp, int N = 0) {
    chan_t c; c.p = nullptr; c.sp = nullptr; c.a = 1.f; c.b = 0.f; c.bw = 1;
    if (ci >= 0 && ci < Ci) {
        if constexpr (IS_MX8(T)) {
            const bool sec = ci >= Cx;
            const uint8_t *base = sec ? (const uint8_t *)sp.x2 : (const uint8_t *)x;
            const int C = sec ? Ci - Cx : Cx, cc = sec ? ci - Cx : ci, bw = mx_bw(C), nb = mx_nb(C);
            c.p = base + (((size_t)n * nb + cc / bw) * plane) * bw + cc % bw;
            c.sp = base + (size_t)N * nb * plane * bw + ((size_t)n * nb + cc / bw) * plane;
            c.bw = bw;
        } else c.p = ci >= Cx ? (const T *)sp.x2 + ((size_t)n * (Ci - Cx) + ci - Cx) * plane : x + ((size_t)n * Cx + ci) * plane;
        if (gp.G) { int ng = n * gp.G + ci / (Ci / gp.G); float a = gp.rstd[ng] * gp.gamma[ci]; c.a = a; c.b = gp.beta[ci] - gp.mean[ng] * a; }
    }
    return c;
}
/* element (n, c, v) of an MX tensor with N samples, C channels, S voxels */
__device__ __forceinline__ float ldmx_e(const void *qv, int N, int C, size_t S, int n, int c, size_t v) {
    const uint8_t *q = (const uint8_t *)qv;
    const int bw = mx_bw(C), nb = mx_nb(C);
    const size_t ri = ((size_t)n * nb + c / bw) * S + v;
    return dec_e4m3(q[ri * bw + c % bw]) * __uint_as_float((unsigned)q[(size_t)N * nb * S * bw + ri] << 23);
}
/* element of a channel at voxel offset off: plane-major types, or MX (decode * 2^(scale - 127)) */
template <typename T> __device__ __forceinline__ float ldc(const chan_t &c, size_t off) {
    if constexpr (IS_MX8(T)) return dec_e4m3(((const uint8_t *)c.p)[off * c.bw]) * __uint_as_float((unsigned)c.sp[off] << 23);
    else return ldx((const T *)c.p, off);
}
__device__ __forceinline__ float act_ab(float v, float a, float b, bool G) {
    if (!G) return v;
    v = fmaf(v, a, b);
    return __fdividef(v, 1.f + __expf(-v));
}
/* exponent e (scale 2^e) such that amax * 2^-e <= qmax; returned as the ue8m0 byte e + 127 */
__device__ __forceinline__ int mx_exp(float amax, float inv_qmax) {
    unsigned u = __float_as_uint(amax * inv_qmax);
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
    uint4 *dst = (uint4 *)(wq + ((size_t)t * Cop + co) * Cip + ch * 32);
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
    if constexpr (IS_MX8(T)) {   /* MX-fp8 output: per voxel and 32- (16-) channel block, amax over the block's rows (lanes g, h, m) */
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
                        const int e = mx_exp(am, 1.f / 448.f);
                        const float mult = exp2i(-e);
                        if (ok) {
                            const int blk = cl / bwt, nbt = mx_nb(Ct);
                            uint8_t *qt = sec ? (uint8_t *)sp.y2 : q, *st = qt + (size_t)N * nbt * S * bwt;
                            const size_t v = ((size_t)oz * H + oy) * W + ox;
                            uint8_t *dst = qt + (((size_t)n * nbt + blk) * S + v) * bwt;
#pragma unroll
                            for (int mm = 0; mm < 2; mm++)
#pragma unroll
                                for (int h = 0; h < 2; h++) if (mm < MBt) dst[mm * 16 + g + 8 * h] = cvt_e4m3(val[mm][h] * mult);
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
                T *yp = IS_MX8(T) ? y : (sp.y2 && co >= sp.o_split) ? (T *)sp.y2 + (((size_t)n * (Co - sp.o_split) + co - sp.o_split) * D + oz) * H * W + (size_t)oy * W
                                                    : y + (((size_t)n * (sp.y2 ? sp.o_split : Co) + co) * D + oz) * H * W + (size_t)oy * W;
#pragma unroll
                for (int q = 0; q < 2; q++) {
                    int ox = ox0 + q * 8 + 2 * t;
                    float v0 = acc[m][r][q][2 * h] + bias, v1 = acc[m][r][q][2 * h + 1] + bias;
                    if constexpr (IS_MX8(T)) {   /* stores done above; statistics from the unquantized values */
                        if (ox < W) { ps += v0; pss += v0 * v0; }
                        if (ox + 1 < W) { ps += v1; pss += v1 * v1; }
                    } else {
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

template <int MT, int TZ, typename T>   /* TZ = output z planes per block (2 or 4); warp owns TZ/2 planes x 2 rows */
__global__ void __launch_bounds__(256, 2) conv_fwd_f8_k(const T *__restrict__ x, const uint8_t *__restrict__ wq, const uint8_t *__restrict__ wsc,
                                                     const float *__restrict__ b, T *__restrict__ y,
                                                     int N, int Ci, int D, int H, int W, int Co, int Cop, int Cip, gnp_t gp, double *__restrict__ osum, int Go, split_t sp) {
    constexpr int BM = MT * 16, TG = 9, RZ = TZ / 2, NR = 2 * RZ, TT = (TZ + 2) * 180, NRX = NR;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    uint8_t *sx = smem_raw;                         /* [TT pos][32 ci] swizzled */
    uint8_t *sxs = sx + TT * 32;                    /* [TT] position scales */
    uint8_t *wa = sxs + ((TT + 127) & ~127);                       /* [TG tap][BM co][32 ci] swizzled */
    uint8_t *was = wa + TG * BM * 32;               /* [TG][BM] */
    chan_t *ctab = (chan_t *)(was + TG * BM + 64 - (TG * BM) % 64);   /* [32] (16-aligned) */
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
    const bool G = gp.G != 0;
    for (int ci0 = 0; ci0 < Cip; ci0 += F8_CI) {
        __syncthreads();
        if (threadIdx.x < 32) ctab[threadIdx.x] = make_chan(IS_MX8(T) ? seg_ci(ci0 + threadIdx.x, Ci, Cx, (Cx + 31) / 32 * 32) : ci0 + threadIdx.x, Ci, Cx, n, plane, x, sp, gp, N);
        __syncthreads();
        if constexpr (IS_MX8(T)) if (!G) {   /* MX input without transform: the chunk is one stored block -> copy bytes and scales */
            const chan_t c0 = ctab[0];
            for (int pos = threadIdx.x; pos < TT; pos += 256) {
                int ix = pos % 18, iy = (pos / 18) % 10, iz = pos / 180;
                int gz = oz0 - 1 + iz, gy = oy0 - 1 + iy, gx = ox0 - 1 + ix;
                bool inb = c0.p && gz >= 0 && gz < D && gy >= 0 && gy < H && gx >= 0 && gx < W;
                uint4 h0 = make_uint4(0u, 0u, 0u, 0u), h1 = h0;
                unsigned sc = 1u;
                if (inb) {
                    size_t off = ((size_t)gz * H + gy) * W + gx;
                    const uint4 *src = (const uint4 *)((const uint8_t *)c0.p + off * c0.bw);
                    h0 = __ldg(src); if (c0.bw == 32) h1 = __ldg(src + 1);
                    sc = c0.sp[off];
                }
                *(uint4 *)(sx + sw16(pos, 0)) = h0;
                *(uint4 *)(sx + sw16(pos, 1)) = h1;
                sxs[pos] = (uint8_t)sc;
            }
        }
        /* staging: one thread per position, all 32 channels -> per-position amax -> e4m3 + scale */
        if (!IS_MX8(T) || G) for (int pos = threadIdx.x; pos < TT; pos += 256) {
            int ix = pos % 18, iy = (pos / 18) % 10, iz = pos / 180;
            int gz = oz0 - 1 + iz, gy = oy0 - 1 + iy, gx = ox0 - 1 + ix;
            bool inb = gz >= 0 && gz < D && gy >= 0 && gy < H && gx >= 0 && gx < W;
            size_t off = inb ? ((size_t)gz * H + gy) * W + gx : 0;
            float v[32], amax = 0.f;
#pragma unroll
            for (int k = 0; k < 32; k++) {
                chan_t c = ctab[k];
                float val = 0.f;
                if (inb && c.p) val = act_ab(ldc<T>(c, off), c.a, c.b, G);
                v[k] = val; amax = fmaxf(amax, fabsf(val));
            }
            int e = mx_exp(amax, 1.f / 448.f);
            float m = exp2i(-e);
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
    fwd_epilogue<MT, NRX, T>(acc, smem_raw, y, b, n, co0, Co, D, H, W, oz0, oy0, ox0, wz, wr, osum, Go, sp, N);
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
template <int MT, int CP, typename T>
__global__ void __launch_bounds__(256, 2) conv_fwd_f8s_k(const T *__restrict__ x, const uint8_t *__restrict__ wq, const uint8_t *__restrict__ wsc,
                                                      const float *__restrict__ b, T *__restrict__ y,
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
    const bool G = gp.G != 0;
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
            if (inb && c.p) val = act_ab(ldc<T>(c, off), c.a, c.b, G);
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
    fwd_epilogue<MT, NRX, T>(acc, smem_raw, y, b, n, co0, Co, D, H, W, oz0, oy0, ox0, wz, wr, osum, Go, sp, N);
}

template <typename T> static T *lp_buf(int slot, size_t n) {
    static void *buf[8][4]; static size_t cap[8][4];
    int d = cur_dev_();
    if (n * sizeof(T) > cap[d][slot]) { if (buf[d][slot]) cudaFree(buf[d][slot]); cudaMalloc(&buf[d][slot], n * sizeof(T)); cap[d][slot] = n * sizeof(T); }
    return (T *)buf[d][slot];
}

template <int MT, int CP, typename T> static void launch_f8s(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp) {
    constexpr int TPK = 32 / CP, NKB = (27 + TPK - 1) / TPK;
    size_t smem = (size_t)F8_T * CP + NKB * MT * 16 * 32 + ((NKB * MT * 16 + 15) & ~15) + 16 + 16 * sizeof(chan_t);
    if (smem < 8 * 256 * sizeof(float)) smem = 8 * 256 * sizeof(float);
    static int attr[8];
    if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)conv_fwd_f8s_k<MT, CP, T>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
    conv_fwd_f8s_k<MT, CP, T><<<grid, 256, smem>>>((const T *)x, wq, ws, b, (T *)y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, gp, osum, Go, sp);
}
template <int CP, typename T> static void small_f8(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp) {
    constexpr int TPK = 32 / CP, NKB = (27 + TPK - 1) / TPK;
    int Cop = (cout + 15) / 16 * 16;
    if (IS_MX8(T) && cout > 16) Cop = (cout + 31) / 32 * 32;
    uint8_t *wq = lp_buf<uint8_t>(0, (size_t)NKB * Cop * 32), *ws = lp_buf<uint8_t>(1, (size_t)NKB * Cop);
    prep_w8s_k<CP><<<nblk_((size_t)NKB * Cop, 128), 128>>>(w, wq, ws, cout, xs.c, Cop);
    int MT = Cop % 64 == 0 ? 4 : Cop % 32 == 0 ? 2 : 1;   /* BM = 16 MT must divide Cop (Cop = 48 -> MT = 1) */
    int nmt = Cop / (MT * 16);
    dim3 grid(nblk_(xs.w, 16), nblk_(xs.h, 8), (unsigned)(nblk_(xs.d, 2) * nmt * xs.n));
    switch (MT) {
    case 1: launch_f8s<1, CP, T>(grid, x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    case 2: launch_f8s<2, CP, T>(grid, x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    default: launch_f8s<4, CP, T>(grid, x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    }
}
template <int MT, int TZ, typename T> static void launch_f8g(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, int Cip, gnp_t gp, double *osum, int Go, split_t sp) {
    constexpr int TT = (TZ + 2) * 180;
    size_t smem = (size_t)TT * 32 + ((TT + 127) & ~127) + 9 * MT * 16 * 33 + 64 + 32 * sizeof(chan_t);
    if (smem < 8 * 256 * sizeof(float)) smem = 8 * 256 * sizeof(float);
    static int attr[8];
    if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)conv_fwd_f8_k<MT, TZ, T>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
    conv_fwd_f8_k<MT, TZ, T><<<grid, 256, smem>>>((const T *)x, wq, ws, b, (T *)y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, Cip, gp, osum, Go, sp);
}
template <typename T> static void fwd_f8_t(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp) {
    static int small = -1;
    if (small < 0) small = getenv("UFSM_F8_NOSMALL") ? 0 : 1;
    /* tap-packed small-channel kernel: fp32 input up to 16 channels; with bf16 input the general kernel is as fast at 16 */
    if (small && xs.c <= (sizeof(T) == 2 || IS_MX8(T) ? 8 : 16) && !sp.x2) {
        if (xs.c <= 4) small_f8<4, T>(x, xs, w, b, cout, y, gp, osum, Go, sp);
        else if (xs.c <= 8) small_f8<8, T>(x, xs, w, b, cout, y, gp, osum, Go, sp);
        else small_f8<16, T>(x, xs, w, b, cout, y, gp, osum, Go, sp);
        return;
    }
    int Cop = (cout + 15) / 16 * 16, Cip = (xs.c + 31) / 32 * 32;
    int Cx = sp.x2 ? sp.c_split : xs.c, CxP = (Cx + 31) / 32 * 32;
    int Ox = sp.y2 ? sp.o_split : cout, OxP = (Ox + 31) / 32 * 32;
    if (IS_MX8(T)) {   /* MX output blocks of 32 channels need 32-row-aligned m-tiles (split outputs: per-tensor segments);
                          MX inputs: per-tensor 32-channel segments */
        if (sp.y2) Cop = OxP + (cout - Ox + 31) / 32 * 32;
        else if (cout > 16) Cop = (cout + 31) / 32 * 32;
        Cip = CxP + (xs.c - Cx + 31) / 32 * 32;
    }
    int nch = Cip / 32;
    uint8_t *wq = lp_buf<uint8_t>(0, (size_t)27 * Cop * Cip), *ws = lp_buf<uint8_t>(1, (size_t)27 * Cop * nch);
    size_t nt = (size_t)27 * Cop * nch;
    if (IS_MX8(T)) prep_w8_k<<<nblk_(nt, 128), 128>>>(w, wq, ws, cout, xs.c, Cop, Cip, Cx, CxP, sp.y2 ? Ox : -1, OxP);
    else prep_w8_k<<<nblk_(nt, 128), 128>>>(w, wq, ws, cout, xs.c, Cop, Cip);
    int MT = Cop % 64 == 0 ? 4 : Cop % 32 == 0 ? 2 : 1;   /* BM = 16 MT must divide Cop (Cop = 48 -> MT = 1) */
    int nmt = Cop / (MT * 16);
    static int tz_env = -1;
    if (tz_env < 0) tz_env = getenv("UFSM_F8_TZ") ? atoi(getenv("UFSM_F8_TZ")) : 0;
    int TZ = tz_env ? tz_env : (MT <= 2 && xs.d >= 16 ? 4 : 2);
    if (MT == 4) TZ = 2;
    dim3 grid(nblk_(xs.w, 16), nblk_(xs.h, 8), (unsigned)(nblk_(xs.d, TZ) * nmt * xs.n));
    switch (MT * 10 + TZ) {
    case 12: launch_f8g<1, 2, T>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    case 14: launch_f8g<1, 4, T>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    case 22: launch_f8g<2, 2, T>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    case 24: launch_f8g<2, 4, T>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    default: launch_f8g<4, 2, T>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    }
}
static int lp_dtype_check(const char *fn, int xbf, int ybf) {
    if (xbf != ybf) { fprintf(stderr, "%s: mixed activation types (x bf16 %d, y bf16 %d) are not instantiated\n", fn, xbf, ybf); abort(); }
    return xbf;
}
extern "C" int lp_conv_fwd_f8(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, gnp_t gp, double *osum, int Go, split_t sp) {
    int dt = lp_dtype_check("lp_conv_fwd_f8", xbf, ybf);
    if (dt == 3) { if (sp.accum) { fprintf(stderr, "lp_conv_fwd_f8: MX-fp8 storage with accumulate is not supported\n"); abort(); } fwd_f8_t<mx8_t>(x, xs, w, b, cout, y, gp, osum, Go, sp); }
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
                                                       int N, int Ci, int D, int H, int W, int Co, gnp_t gp, split_t sp, int ZC) {
    constexpr int CH = 8 * NT, BMo = 16 * MT;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    uint8_t *sxq = smem_raw;                        /* [CH][X8_CS] */
    uint8_t *sg = sxq + CH * X8_CS;                 /* [BMo][G8_CS] */
    uint8_t *sgs = sg + BMo * G8_CS;                /* [BMo][8 ksteps] */
    uint8_t *sxs = sgs + BMo * 8;                   /* [CH][4 slots] */
    float *sbias = (float *)(sxs + CH * 4);         /* [BMo] */
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
    for (int zt = zt_begin; zt < nzt && zt < (zc + 1) * ZC; zt++) {
        const int oz0 = zt * 2;
        const int np = zt == zt_begin ? 4 : 2, gz_first = zt == zt_begin ? oz0 - 1 : oz0 + 1;
        __syncthreads();
        /* X: warp per (channel, plane): 10 rows x (4 aligned float4 for x = ox0..ox0+15, plus the halo voxels
           ox0-1 and ox0+16); position p of a row is stored at byte 3 + p of its 24-byte slot */
        for (int task = warp; task < CH * np; task += 9) {
            int k = task / np, gz = gz_first + task % np, slot = (gz + 1) & 3, ci = ci0 + k;
            const bool ok = ci < Ci && gz >= 0 && gz < D;
            chan_t c = make_chan(ok ? ci : Ci, Ci, Cx, n, plane, x, sp, gp, N);
            const T *xc = ok && !IS_MX8(T) ? (const T *)c.p + (size_t)gz * H * W : x;
            const bool G = gp.G != 0;
            float4 v4[2]; float vs = 0.f, amax = 0.f;
#pragma unroll
            for (int i = 0; i < 2; i++) {
                int f = lane + 32 * i, row = f >> 2, gyy = oy0 - 1 + row, gx = ox0 + 4 * (f & 3);
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (ok && f < 40 && gyy >= 0 && gyy < H) {
                    const T *src = xc + (size_t)gyy * W + gx;
                    if constexpr (IS_MX8(T)) {
                        const size_t vo = ((size_t)gz * H + gyy) * W + gx;
                        if (gx < W) v.x = ldc<T>(c, vo); if (gx + 1 < W) v.y = ldc<T>(c, vo + 1); if (gx + 2 < W) v.z = ldc<T>(c, vo + 2); if (gx + 3 < W) v.w = ldc<T>(c, vo + 3);
                    } else if (vec) { if (gx < W) v = ldx4(src); }
                    else { if (gx < W) v.x = ldx(src, 0); if (gx + 1 < W) v.y = ldx(src, 1); if (gx + 2 < W) v.z = ldx(src, 2); if (gx + 3 < W) v.w = ldx(src, 3); }
                    v.x = gx < W ? act_ab(v.x, c.a, c.b, G) : 0.f; v.y = gx + 1 < W ? act_ab(v.y, c.a, c.b, G) : 0.f;
                    v.z = gx + 2 < W ? act_ab(v.z, c.a, c.b, G) : 0.f; v.w = gx + 3 < W ? act_ab(v.w, c.a, c.b, G) : 0.f;
                }
                v4[i] = v;
                amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
            }
            {   /* halo: lane < 20 -> row lane >> 1, side lane & 1 (x = ox0 - 1 or ox0 + 16) */
                int row = lane >> 1, gyy = oy0 - 1 + row, gx = (lane & 1) ? ox0 + 16 : ox0 - 1;
                if (ok && lane < 20 && gyy >= 0 && gyy < H && gx >= 0 && gx < W) vs = act_ab(IS_MX8(T) ? ldc<T>(c, ((size_t)gz * H + gyy) * W + gx) : ldx(xc, (size_t)gyy * W + gx), c.a, c.b, G);
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
        /* GY: K block (co, ks) = 2 rows x 16 voxels = 8 float4; warp covers 4 blocks, 8 lanes each */
        for (int task = warp; task < BMo * 2; task += 9) {
            int blk = task * 4 + (lane >> 3), c = blk >> 3, ks = blk & 7, f = lane & 7;
            int row = ks * 2 + (f >> 2), vx = 4 * (f & 3), vz = row >> 3, vy = row & 7;
            int co = co0 + c, oz = oz0 + vz, oy = oy0 + vy, ox = ox0 + vx;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (co < Co && oz < D && oy < H) {
                if constexpr (IS_MX8(TG)) {
                    const size_t vo = ((size_t)oz * H + oy) * W + ox, Sg = (size_t)D * H * W;
                    if (ox < W) v.x = ldmx_e(gy, N, Co, Sg, n, co, vo); if (ox + 1 < W) v.y = ldmx_e(gy, N, Co, Sg, n, co, vo + 1);
                    if (ox + 2 < W) v.z = ldmx_e(gy, N, Co, Sg, n, co, vo + 2); if (ox + 3 < W) v.w = ldmx_e(gy, N, Co, Sg, n, co, vo + 3);
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
            *(unsigned *)(sg + c * G8_CS + row * 16 + vx) = cvt_e4m3x4(v.x * m, v.y * m, v.z * m, v.w * m);
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

template <int MT, int NT, typename T, typename TG> static void launch_bw8(dim3 grid, size_t smem, const void *x, shape5 xs, const TG *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp, int ZC) {
    static int attr[8];
    if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)conv_bwd_w_f8_k<MT, NT, T, TG>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
    conv_bwd_w_f8_k<MT, NT, T, TG><<<grid, 288, smem>>>((const T *)x, gy, gw, gb, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, gp, sp, ZC);
}
template <typename T, typename TG> static void bwd_w_f8_t(const void *x, shape5 xs, const TG *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp) {
    int MT = ys.c >= 32 ? 2 : 1;
    int NT = xs.c <= 8 ? 1 : MT == 1 && xs.c % 24 == 0 ? 3 : 2;   /* 24 input channels per block when cout = 16: gy restaged less (dec0.c1: -6%) */
    if (getenv("UFSM_F8_NT")) NT = atoi(getenv("UFSM_F8_NT"));
    if (getenv("UFSM_F8_MT")) MT = atoi(getenv("UFSM_F8_MT"));
    size_t smem = (size_t)8 * NT * X8_CS + 16 * MT * (G8_CS + 8) + 8 * NT * 4 + 16 * MT * 4 + 16;
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
    if (xbf == 3 && gybf == 3) bwd_w_f8_t<mx8_t, mx8_t>(x, xs, (const mx8_t *)gy, ys, gw, gb, gp, sp);
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
   the last one half zero). Same tile / warp layout as the FP8 kernel; the input tile is [pos][16 B] (32 channels
   x 4 bit), one scale per (position, 32 channels) -> B scale register bytes 0 / 1 = the two taps' positions.
   Weights wq4[tap][Cop][Cip/2] with one scale per (tap, co, 32-channel chunk). Inputs with Ci <= 16 use the FP8
   small-channel kernel. */
__device__ __forceinline__ unsigned cvt_e2m1x8(const float *v, float m) {
    return (unsigned)cvt_e2m1x2(v[0] * m, v[1] * m) | ((unsigned)cvt_e2m1x2(v[2] * m, v[3] * m) << 8) |
           ((unsigned)cvt_e2m1x2(v[4] * m, v[5] * m) << 16) | ((unsigned)cvt_e2m1x2(v[6] * m, v[7] * m) << 24);
}
__global__ void prep_w4_k(const float *__restrict__ w, uint8_t *__restrict__ wq, uint8_t *__restrict__ ws, int Co, int Ci, int Cop, int Cip) {
    const int nch = Cip / 32;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)28 * Cop * nch) return;              /* tap 27 = the zero half of the last pair */
    int ch = (int)(i % nch), co = (int)((i / nch) % Cop), t = (int)(i / ((size_t)nch * Cop));
    float v[32], amax = 0.f;
#pragma unroll
    for (int k = 0; k < 32; k++) { int ci = ch * 32 + k; v[k] = (t < 27 && co < Co && ci < Ci) ? w[((size_t)co * Ci + ci) * 27 + t] : 0.f; amax = fmaxf(amax, fabsf(v[k])); }
    int e = mx_exp(amax, 1.f / 6.f);
    float m = exp2i(-e);
    *(uint4 *)(wq + ((size_t)t * Cop + co) * (Cip / 2) + ch * 16) = make_uint4(cvt_e2m1x8(v, m), cvt_e2m1x8(v + 8, m), cvt_e2m1x8(v + 16, m), cvt_e2m1x8(v + 24, m));
    ws[i] = (uint8_t)(e + 127);
}
template <int MT, typename T>
__global__ void __launch_bounds__(256, 2) conv_fwd_f4_k(const T *__restrict__ x, const uint8_t *__restrict__ wq, const uint8_t *__restrict__ wsc,
                                                     const float *__restrict__ b, T *__restrict__ y,
                                                     int N, int Ci, int D, int H, int W, int Co, int Cop, int Cip, gnp_t gp, double *__restrict__ osum, int Go, split_t sp) {
    constexpr int BM = MT * 16, PG = 7, NRX = 2;      /* PG: tap pairs per weight stage */
    extern __shared__ __align__(128) unsigned char smem_raw[];
    uint8_t *sx = smem_raw;                         /* [F8_T pos][16 B] */
    uint8_t *sxs = sx + F8_T * 16;                  /* [F8_T] (768 reserved) */
    uint8_t *wa = sxs + 768;                        /* [PG pair][BM co][32 B] (tapA 16 B | tapB 16 B), swizzled */
    unsigned short *was = (unsigned short *)(wa + PG * BM * 32);   /* [PG][BM] (byte 0 tapA, byte 1 tapB) */
    chan_t *ctab = (chan_t *)(wa + PG * BM * 32 + ((PG * BM * 2 + 15) & ~15));
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wz = warp >> 2, wr = (warp & 3) * 2;
    const int ox0 = blockIdx.x * 16, oy0 = blockIdx.y * 8;
    int bz = blockIdx.z;
    const int nzt = (D + 1) / 2;
    const int oz0 = (bz % nzt) * 2; bz /= nzt;
    const int nmt = Cop / BM;
    const int co0 = (bz % nmt) * BM; const int n = bz / nmt;
    const int nch = Cip / 32;
    float acc[MT][2][2][4];
#pragma unroll
    for (int m = 0; m < MT; m++) for (int r = 0; r < 2; r++) for (int q = 0; q < 2; q++) for (int k = 0; k < 4; k++) acc[m][r][q][k] = 0.f;
    const size_t plane = (size_t)D * H * W;
    const int Cx = sp.x2 ? sp.c_split : Ci;
    const bool G = gp.G != 0;
    for (int ci0 = 0; ci0 < Cip; ci0 += 32) {
        __syncthreads();
        if (threadIdx.x < 32) ctab[threadIdx.x] = make_chan(ci0 + threadIdx.x, Ci, Cx, n, plane, x, sp, gp, N);
        __syncthreads();
        for (int pos = threadIdx.x; pos < F8_T; pos += 256) {
            int ix = pos % 18, iy = (pos / 18) % 10, iz = pos / 180;
            int gz = oz0 - 1 + iz, gy = oy0 - 1 + iy, gx = ox0 - 1 + ix;
            bool inb = gz >= 0 && gz < D && gy >= 0 && gy < H && gx >= 0 && gx < W;
            size_t off = inb ? ((size_t)gz * H + gy) * W + gx : 0;
            float v[32], amax = 0.f;
#pragma unroll
            for (int k = 0; k < 32; k++) {
                chan_t c = ctab[k];
                float val = 0.f;
                if (inb && c.p) val = act_ab(ldc<T>(c, off), c.a, c.b, G);
                v[k] = val; amax = fmaxf(amax, fabsf(val));
            }
            int e = mx_exp(amax, 1.f / 6.f);
            float m = exp2i(-e);
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
                const int rA = ((wz + tA / 9) * 10 + wr + (tA / 3) % 3) * 18 + tA % 3, rB = ((wz + tB / 9) * 10 + wr + (tB / 3) % 3) * 18 + tB % 3;
                unsigned bfr[2][4], sb[2][2];
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    int mat = lane >> 3, q = mat >> 1, isB = mat & 1;
                    int pos = (isB ? rB : rA) + r * 18 + q * 8 + (lane & 7);
                    ldsm_x4(bfr[r], sx + pos * 16);
#pragma unroll
                    for (int q2 = 0; q2 < 2; q2++) sb[r][q2] = (unsigned)sxs[rA + r * 18 + q2 * 8 + g] | ((unsigned)sxs[rB + r * 18 + q2 * 8 + g] << 8);
                }
#pragma unroll
                for (int m = 0; m < MT; m++) {
                    unsigned af[4];
                    int mat = lane >> 3, row = m * 16 + (mat & 1) * 8 + (lane & 7);
                    ldsm_x4(af, wa + pp * BM * 32 + sw16(row, mat >> 1));
                    unsigned sa = was[pp * BM + m * 16 + g + 8 * (t & 1)];
#pragma unroll
                    for (int r = 0; r < 2; r++) { mma_f4(acc[m][r][0], af, bfr[r], sa, sb[r][0]); mma_f4(acc[m][r][1], af, bfr[r] + 2, sa, sb[r][1]); }
                }
            }
        }
    }
    fwd_epilogue<MT, NRX, T>(acc, smem_raw, y, b, n, co0, Co, D, H, W, oz0, oy0, ox0, wz, wr, osum, Go, sp, N);
}
template <int MT, typename T> static void launch_f4(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, int Cip, gnp_t gp, double *osum, int Go, split_t sp) {
    size_t smem = (size_t)F8_T * 16 + 768 + 7 * MT * 16 * 32 + ((7 * MT * 16 * 2 + 15) & ~15) + 32 * sizeof(chan_t);
    if (smem < 8 * 256 * sizeof(float)) smem = 8 * 256 * sizeof(float);
    static int attr[8];
    if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)conv_fwd_f4_k<MT, T>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
    conv_fwd_f4_k<MT, T><<<grid, 256, smem>>>((const T *)x, wq, ws, b, (T *)y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, Cip, gp, osum, Go, sp);
}
template <typename T> static void fwd_f4_t(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp) {
    int Cop = (cout + 15) / 16 * 16, Cip = (xs.c + 31) / 32 * 32, nch = Cip / 32;
    uint8_t *wq = lp_buf<uint8_t>(2, (size_t)28 * Cop * Cip / 2), *ws = lp_buf<uint8_t>(3, (size_t)28 * Cop * nch);
    size_t nt = (size_t)28 * Cop * nch;
    prep_w4_k<<<nblk_(nt, 128), 128>>>(w, wq, ws, cout, xs.c, Cop, Cip);
    int MT = Cop % 64 == 0 ? 4 : Cop % 32 == 0 ? 2 : 1;   /* BM = 16 MT must divide Cop (Cop = 48 -> MT = 1) */
    int nmt = Cop / (MT * 16);
    dim3 grid(nblk_(xs.w, 16), nblk_(xs.h, 8), (unsigned)(nblk_(xs.d, 2) * nmt * xs.n));
    switch (MT) {
    case 1: launch_f4<1, T>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    case 2: launch_f4<2, T>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    default: launch_f4<4, T>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    }
}
extern "C" int lp_conv_fwd_f4(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, gnp_t gp, double *osum, int Go, split_t sp) {
    static int small = -1;
    if (small < 0) small = getenv("UFSM_F4_SMALL") ? atoi(getenv("UFSM_F4_SMALL")) : 16;
    if (xs.c <= small) return lp_conv_fwd_f8(x, xbf, xs, w, b, cout, y, ybf, gp, osum, Go, sp);
    int dt = lp_dtype_check("lp_conv_fwd_f4", xbf, ybf);
    if (dt == 2) fwd_f4_t<__half>(x, xs, w, b, cout, y, gp, osum, Go, sp);
    else if (dt) fwd_f4_t<bf16>(x, xs, w, b, cout, y, gp, osum, Go, sp);
    else fwd_f4_t<float>(x, xs, w, b, cout, y, gp, osum, Go, sp);
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
__device__ __forceinline__ int s2pos(int row, int x) { return row * 17 + ((x & 1) ? 9 + (x >> 1) : (x >> 1)); }
template <int MT, typename T>
__global__ void __launch_bounds__(256, 2) conv_fwd_s2_f8_k(const T *__restrict__ x, const uint8_t *__restrict__ wq, const uint8_t *__restrict__ wsc,
                                                        const float *__restrict__ b, T *__restrict__ y,
                                                        int N, int Ci, int D, int H, int W, int Co, int Cop, int Cip, int Do, int Ho, int Wo) {
    constexpr int BM = MT * 16, TG = 9;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    uint8_t *sx = smem_raw;                         /* [S2F8_T pos][32 ci] swizzled */
    uint8_t *sxs = sx + S2F8_T * 32;                /* [S2F8_T] (768) */
    uint8_t *wa = sxs + 768;                        /* [TG][BM][32] swizzled */
    uint8_t *was = wa + TG * BM * 32;               /* [TG][BM] */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wz = warp >> 2, wr = warp & 3;
    const int ox0 = blockIdx.x * 8, oy0 = blockIdx.y * 4;
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
    for (int ci0 = 0; ci0 < Cip; ci0 += 32) {
        __syncthreads();
        for (int p = threadIdx.x; p < S2F8_T; p += 256) {   /* p = row * 17 + parity-split column */
            int row = p / 17, c = p - row * 17, ix = c < 9 ? 2 * c : 2 * (c - 9) + 1;
            int iz = row / 9, iy = row - iz * 9;
            int gz = 2 * oz0 - 1 + iz, gyy = 2 * oy0 - 1 + iy, gx = 2 * ox0 - 1 + ix;
            bool inb = gz >= 0 && gz < D && gyy >= 0 && gyy < H && gx >= 0 && gx < W;
            size_t off = ((size_t)gz * H + gyy) * W + gx;
            if constexpr (IS_MX8(T)) {   /* MX input: the 32-channel chunk is one stored block -> copy */
                const int bw = mx_bw(Ci), nb = mx_nb(Ci), blk = ci0 / 32;
                uint4 h0 = make_uint4(0u, 0u, 0u, 0u), h1 = h0;
                unsigned sc = 1u;
                if (inb) {
                    const uint8_t *q = (const uint8_t *)x;
                    const uint4 *src = (const uint4 *)(q + (((size_t)n * nb + blk) * plane + off) * bw);
                    h0 = __ldg(src); if (bw == 32) h1 = __ldg(src + 1);
                    sc = q[(size_t)N * nb * plane * bw + ((size_t)n * nb + blk) * plane + off];
                }
                *(uint4 *)(sx + sw16(p, 0)) = h0;
                *(uint4 *)(sx + sw16(p, 1)) = h1;
                sxs[p] = (uint8_t)sc;
                continue;
            }
            float v[32], amax = 0.f;
#pragma unroll
            for (int k = 0; k < 32; k++) {
                int ci = ci0 + k;
                v[k] = inb && ci < Ci ? ldx(x + ((size_t)n * Ci + ci) * plane, off) : 0.f;
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
    if constexpr (IS_MX8(T)) {   /* MX-fp8 output (blocks of 32 / 16 channels per voxel) */
        const int bw = mx_bw(Co), nb = mx_nb(Co);
        const size_t So = (size_t)Do * Ho * Wo;
        const int oz = oz0 + wz, oy = oy0 + wr;
        uint8_t *q = (uint8_t *)y, *scp = q + (size_t)N * nb * So * bw;
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
                const int e = mx_exp(am, 1.f / 448.f);
                const float mult = exp2i(-e);
                if (ok) {
                    const int blk = (co0 + j * 16) / bw;
                    const size_t v = ((size_t)oz * Ho + oy) * Wo + ox;
                    uint8_t *dst = q + (((size_t)n * nb + blk) * So + v) * bw;
#pragma unroll
                    for (int mm = 0; mm < 2; mm++)
#pragma unroll
                        for (int h = 0; h < 2; h++) if (mm < bw / 16) dst[mm * 16 + g + 8 * h] = cvt_e4m3(val[mm][h] * mult);
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
template <int MT, typename T> static void launch_s2f8(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, int Cip, shape5 ys) {
    size_t smem = (size_t)S2F8_T * 32 + 768 + 9 * MT * 16 * 33;
    static int attr[8];
    if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)conv_fwd_s2_f8_k<MT, T>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
    conv_fwd_s2_f8_k<MT, T><<<grid, 256, smem>>>((const T *)x, wq, ws, b, (T *)y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, Cip, ys.d, ys.h, ys.w);
}
extern "C" int lp_conv_fwd_s2_f8(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, shape5 ys) {
    lp_dtype_check("lp_conv_fwd_s2_f8", xbf, ybf);
    int Cop = (cout + 15) / 16 * 16, Cip = (xs.c + 31) / 32 * 32, nch = Cip / 32;
    if (xbf == 3 && cout > 16) Cop = (cout + 31) / 32 * 32;   /* MX output: 32-row-aligned m-tiles */
    uint8_t *wq = lp_buf<uint8_t>(0, (size_t)27 * Cop * Cip), *ws = lp_buf<uint8_t>(1, (size_t)27 * Cop * nch);
    size_t nt = (size_t)27 * Cop * nch;
    prep_w8_k<<<nblk_(nt, 128), 128>>>(w, wq, ws, cout, xs.c, Cop, Cip);
    int MT = Cop % 64 == 0 ? 4 : Cop % 32 == 0 ? 2 : 1;
    int nmt = Cop / (MT * 16);
    dim3 grid(nblk_(ys.w, 8), nblk_(ys.h, 4), (unsigned)(nblk_(ys.d, 2) * nmt * xs.n));
#define S2L(MT_) do { if (xbf == 3) launch_s2f8<MT_, mx8_t>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, ys); else if (xbf == 2) launch_s2f8<MT_, __half>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, ys); else if (xbf) launch_s2f8<MT_, bf16>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, ys); else launch_s2f8<MT_, float>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, ys); } while (0)
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
                                                          int N, int Ci, int D, int H, int W, int Co, int Do, int Ho, int Wo, int ZC) {
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
            const T *xc = IS_MX8(T) ? x : x + ((size_t)n * Ci + (ok ? ci : 0)) * plane + (size_t)(ok ? gz : 0) * H * W;
            const split_t nsp = {nullptr, 0, nullptr, 0, 0}; const gnp_t ngp = {nullptr, nullptr, nullptr, nullptr, 0};
            const chan_t cm = make_chan(ok ? ci : Ci, Ci, Ci, n, plane, x, nsp, ngp, N);   /* MX element access */
            float v[3][8], vh = 0.f, amax = 0.f;
#pragma unroll
            for (int i = 0; i < 3; i++) {
                int f = lane + 32 * i, iy = f >> 2, j = f & 3, gyy = 2 * oy0 - 1 + iy, gx = 2 * ox0 + 8 * j;
#pragma unroll
                for (int e = 0; e < 8; e++) v[i][e] = 0.f;
                if (ok && f < 68 && gyy >= 0 && gyy < H) {
                    const T *src = xc + (size_t)gyy * W + gx;
                    if constexpr (IS_MX8(T)) { const size_t vo = ((size_t)gz * H + gyy) * W + gx; for (int e = 0; e < 8; e++) if (gx + e < W) v[i][e] = ldc<T>(cm, vo + e); }
                    else if (vec8 && gx + 8 <= W) { float tmp[8]; ld8x<T>(src, tmp); for (int e = 0; e < 8; e++) v[i][e] = tmp[e]; }
                    else { for (int e = 0; e < 8; e++) if (gx + e < W) v[i][e] = ldx(src, e); }
                }
#pragma unroll
                for (int e = 0; e < 8; e++) amax = fmaxf(amax, fabsf(v[i][e]));
            }
            {
                int gyy = 2 * oy0 - 1 + lane, gx = 2 * ox0 - 1;
                if (ok && lane < 17 && gyy >= 0 && gyy < H && gx >= 0) vh = IS_MX8(T) ? ldc<T>(cm, ((size_t)gz * H + gyy) * W + gx) : ldx(xc, (size_t)gyy * W + gx);
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
                    if (ox < Wo) vv.x = ldmx_e(gy, N, Co, Sg, n, co, vo); if (ox + 1 < Wo) vv.y = ldmx_e(gy, N, Co, Sg, n, co, vo + 1);
                    if (ox + 2 < Wo) vv.z = ldmx_e(gy, N, Co, Sg, n, co, vo + 2); if (ox + 3 < Wo) vv.w = ldmx_e(gy, N, Co, Sg, n, co, vo + 3);
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
template <int MT, int NT, typename T, typename TG> static void launch_bws2(dim3 grid, size_t smem, const void *x, shape5 xs, const TG *gy, shape5 ys, float *gw, float *gb, int ZC) {
    static int attr[8];
    if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)conv_bwd_w_s2_f8_k<MT, NT, T, TG>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
    conv_bwd_w_s2_f8_k<MT, NT, T, TG><<<grid, 288, smem>>>((const T *)x, gy, gw, gb, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, ys.d, ys.h, ys.w, ZC);
}
template <typename T, typename TG> static void bwd_w_s2_f8_t(const void *x, shape5 xs, const TG *gy, shape5 ys, float *gw, float *gb) {
    int MT = ys.c >= 32 ? 2 : 1, NT = xs.c <= 8 ? 1 : 2;
    size_t smem = (size_t)8 * NT * XS2_CS + 16 * MT * (G8_CS + 8) + 8 * NT * 8 + 16 * MT * 4 + 16;
    int nzt = nblk_(ys.d, 2), base = (int)(((xs.c + 8 * NT - 1) / (8 * NT)) * ((ys.c + 16 * MT - 1) / (16 * MT)) * nblk_(ys.w, 16) * nblk_(ys.h, 8) * ys.n);
    int ZC = nzt < 8 ? nzt : 8;
    while (ZC > 1 && (size_t)base * nblk_(nzt, ZC) < 72) ZC--;
    dim3 grid((xs.c + 8 * NT - 1) / (8 * NT), (ys.c + 16 * MT - 1) / (16 * MT), (unsigned)(nblk_(ys.w, 16) * nblk_(ys.h, 8) * nblk_(nzt, ZC) * ys.n));
    switch (MT * 10 + NT) {
    case 11: launch_bws2<1, 1, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, ZC); break;
    case 12: launch_bws2<1, 2, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, ZC); break;
    case 21: launch_bws2<2, 1, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, ZC); break;
    default: launch_bws2<2, 2, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, ZC); break;
    }
}
extern "C" int lp_bwd_w_s2_f8(const void *x, int xbf, shape5 xs, const void *gy, int gybf, shape5 ys, float *gw, float *gb) {
    if (xbf == 3 && gybf == 3) bwd_w_s2_f8_t<mx8_t, mx8_t>(x, xs, (const mx8_t *)gy, ys, gw, gb);
    else if (xbf == 3 && gybf == 2) bwd_w_s2_f8_t<mx8_t, __half>(x, xs, (const __half *)gy, ys, gw, gb);
    else if (xbf == 3 && gybf == 1) bwd_w_s2_f8_t<mx8_t, bf16>(x, xs, (const bf16 *)gy, ys, gw, gb);
    else if (xbf == 3) bwd_w_s2_f8_t<mx8_t, float>(x, xs, (const float *)gy, ys, gw, gb);
    else if (xbf == 2 && gybf) bwd_w_s2_f8_t<__half, __half>(x, xs, (const __half *)gy, ys, gw, gb);
    else if (xbf == 2) bwd_w_s2_f8_t<__half, float>(x, xs, (const float *)gy, ys, gw, gb);
    else if (xbf && gybf) bwd_w_s2_f8_t<bf16, bf16>(x, xs, (const bf16 *)gy, ys, gw, gb);
    else if (xbf) bwd_w_s2_f8_t<bf16, float>(x, xs, (const float *)gy, ys, gw, gb);
    else bwd_w_s2_f8_t<float, float>(x, xs, (const float *)gy, ys, gw, gb);
    LPCK();
    return 0;
}

/* ======================= elementwise ops on MX-fp8 activation tensors =======================
   Voxel-major over channel blocks: a thread owns one (n, block, voxel) row of bw channels, so the per-voxel amax that
   defines the output scale is local. */
__device__ __forceinline__ void mx_load_row(const uint8_t *q, const uint8_t *sc, size_t ri, int bw, float *v) {   /* ri = (n*nb + blk)*S + voxel */
    const float s = __uint_as_float((unsigned)sc[ri] << 23);
    const uint4 *src = (const uint4 *)(q + ri * bw);
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
__device__ __forceinline__ void mx_store_row(uint8_t *q, uint8_t *sc, size_t ri, int bw, const float *v) {
    float am = 0.f;
#pragma unroll
    for (int k = 0; k < 32; k++) if (k < bw) am = fmaxf(am, fabsf(v[k]));
    const int e = mx_exp(am, 1.f / 448.f);
    const float m = exp2i(-e);
    uint4 *dst = (uint4 *)(q + ri * bw);
    dst[0] = make_uint4(cvt_e4m3x4(v[0] * m, v[1] * m, v[2] * m, v[3] * m), cvt_e4m3x4(v[4] * m, v[5] * m, v[6] * m, v[7] * m),
                        cvt_e4m3x4(v[8] * m, v[9] * m, v[10] * m, v[11] * m), cvt_e4m3x4(v[12] * m, v[13] * m, v[14] * m, v[15] * m));
    if (bw == 32) dst[1] = make_uint4(cvt_e4m3x4(v[16] * m, v[17] * m, v[18] * m, v[19] * m), cvt_e4m3x4(v[20] * m, v[21] * m, v[22] * m, v[23] * m),
                                      cvt_e4m3x4(v[24] * m, v[25] * m, v[26] * m, v[27] * m), cvt_e4m3x4(v[28] * m, v[29] * m, v[30] * m, v[31] * m));
    sc[ri] = (uint8_t)(e + 127);
}
/* fp32 [n][C][S] -> MX */
__global__ void f32_to_mx8_k(const float *x, uint8_t *y, int N, int C, size_t S) {
    const int bw = mx_bw(C), nb = mx_nb(C);
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * nb * S) return;
    size_t v = i % S; int blk = (int)((i / S) % nb), n = (int)(i / (S * nb));
    float r[32];
#pragma unroll
    for (int k = 0; k < 32; k++) { int c = blk * bw + k; r[k] = k < bw && c < C ? x[((size_t)n * C + c) * S + v] : 0.f; }
    mx_store_row(y, y + (size_t)N * nb * S * bw, i, bw, r);
}
extern "C" void lp_f32_to_mx8(const float *x, int N, int C, size_t S, void *y) {
    size_t n = (size_t)N * mx_nb(C) * S;
    f32_to_mx8_k<<<nblk_(n, 256), 256>>>(x, (uint8_t *)y, N, C, S); LPCK();
}
/* MX -> fp32 [n][C][S] (tests) */
__global__ void mx8_to_f32_k(const uint8_t *x, float *y, int N, int C, size_t S) {
    const int bw = mx_bw(C), nb = mx_nb(C);
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * nb * S) return;
    size_t v = i % S; int blk = (int)((i / S) % nb), n = (int)(i / (S * nb));
    float r[32];
    mx_load_row(x, x + (size_t)N * nb * S * bw, i, bw, r);
#pragma unroll
    for (int k = 0; k < 32; k++) { int c = blk * bw + k; if (k < bw && c < C) y[((size_t)n * C + c) * S + v] = r[k]; }
}
extern "C" void lp_mx8_to_f32(const void *x, int N, int C, size_t S, float *y) {
    size_t n = (size_t)N * mx_nb(C) * S;
    mx8_to_f32_k<<<nblk_(n, 256), 256>>>((const uint8_t *)x, y, N, C, S); LPCK();
}
/* y = silu(gn(x)), both MX */
__global__ void gn_silu_apply_mx_k(const uint8_t *x, uint8_t *y, int N, int C, int G, size_t S, const float *gamma, const float *beta, const float *mean, const float *rstd) {
    const int bw = mx_bw(C), nb = mx_nb(C), cpg = C / G;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * nb * S) return;
    int blk = (int)((i / S) % nb), n = (int)(i / (S * nb));
    float r[32];
    mx_load_row(x, x + (size_t)N * nb * S * bw, i, bw, r);
#pragma unroll
    for (int k = 0; k < 32; k++) {
        int c = blk * bw + k;
        if (k < bw && c < C) { int ng = n * G + c / cpg; float v = (r[k] - mean[ng]) * rstd[ng] * gamma[c] + beta[c]; r[k] = v / (1.f + __expf(-v)); }
        else r[k] = 0.f;
    }
    mx_store_row(y, y + (size_t)N * nb * S * bw, i, bw, r);
}
extern "C" void lp_gn_silu_apply_mx(const void *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd, void *y) {
    size_t S = shape_spatial(s), n = (size_t)s.n * mx_nb(s.c) * S;
    gn_silu_apply_mx_k<<<nblk_(n, 256), 256>>>((const uint8_t *)x, (uint8_t *)y, s.n, s.c, G, S, gamma, beta, mean, rstd); LPCK();
}
/* exact-2x trilinear upsample (align_corners = false, edge-clamped): out[o] = 0.75 in[o/2] + 0.25 in[o/2 -+ 1] per axis */
__global__ void up2_mx_k(const uint8_t *x, uint8_t *y, int N, int C, int D, int H, int W) {
    const int bw = mx_bw(C), nb = mx_nb(C), Do = 2 * D, Ho = 2 * H, Wo = 2 * W;
    const size_t S = (size_t)D * H * W, So = (size_t)Do * Ho * Wo;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * nb * So) return;
    size_t vo = i % So; size_t nbk = i / So;
    int ox = (int)(vo % Wo), oy = (int)((vo / Wo) % Ho), oz = (int)(vo / ((size_t)Wo * Ho));
    int mz[2] = {oz >> 1, min(max((oz >> 1) + ((oz & 1) ? 1 : -1), 0), D - 1)}, my[2] = {oy >> 1, min(max((oy >> 1) + ((oy & 1) ? 1 : -1), 0), H - 1)}, mx[2] = {ox >> 1, min(max((ox >> 1) + ((ox & 1) ? 1 : -1), 0), W - 1)};
    const float wt[2] = {0.75f, 0.25f};
    const uint8_t *sc = x + (size_t)N * nb * S * bw;
    float acc[32] = {};
#pragma unroll
    for (int a = 0; a < 2; a++)
#pragma unroll
        for (int bb = 0; bb < 2; bb++)
#pragma unroll
            for (int c = 0; c < 2; c++) {
                float r[32], w3 = wt[a] * wt[bb] * wt[c];
                mx_load_row(x, sc, nbk * S + ((size_t)mz[a] * H + my[bb]) * W + mx[c], bw, r);
#pragma unroll
                for (int k = 0; k < 32; k++) acc[k] += w3 * r[k];
            }
    mx_store_row(y, y + (size_t)N * nb * So * bw, i, bw, acc);
}
extern "C" void lp_up2_fwd_mx(const void *x, shape5 xs, void *y) {
    size_t n = (size_t)xs.n * mx_nb(xs.c) * 8 * shape_spatial(xs);
    up2_mx_k<<<nblk_(n, 256), 256>>>((const uint8_t *)x, (uint8_t *)y, xs.n, xs.c, xs.d, xs.h, xs.w); LPCK();
}
/* 1^3 conv (head) reading an MX tensor: y[n][co][v] = b[co] + sum_ci w[co][ci] x[ci][v] (fp32 output) */
__global__ void conv1_mx_k(const uint8_t *x, const float *w, const float *b, float *y, int N, int Ci, int Co, size_t S) {
    const int bw = mx_bw(Ci), nb = mx_nb(Ci);
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * S) return;
    int n = (int)(i / S); size_t v = i % S;
    float acc[8];
    for (int co = 0; co < Co && co < 8; co++) acc[co] = b ? b[co] : 0.f;
    for (int blk = 0; blk < nb; blk++) {
        float r[32];
        mx_load_row(x, x + (size_t)N * nb * S * bw, ((size_t)n * nb + blk) * S + v, bw, r);
        for (int k = 0; k < bw; k++) { int ci = blk * bw + k; if (ci < Ci) for (int co = 0; co < Co && co < 8; co++) acc[co] += w[co * Ci + ci] * r[k]; }
    }
    for (int co = 0; co < Co && co < 8; co++) y[((size_t)n * Co + co) * S + v] = acc[co];
}
extern "C" void lp_conv1_fwd_mx(const void *x, shape5 xs, const float *w, const float *b, int cout, float *y) {
    if (cout > 8) { fprintf(stderr, "lp_conv1_fwd_mx: cout %d > 8\n", cout); abort(); }
    size_t S = shape_spatial(xs);
    conv1_mx_k<<<nblk_((size_t)xs.n * S, 256), 256>>>((const uint8_t *)x, w, b, y, xs.n, xs.c, cout, S); LPCK();
}
/* 1^3 weight gradient with an MX input: gw[co][ci] += sum_v gy[co][v] x[ci][v] (Ci x Co <= 64); block partials via atomics */
template <typename TG>
__global__ void __launch_bounds__(256) conv_bwd_w1_mx_k(const uint8_t *x, const TG *gy, float *gw, int N, int Ci, int Co, size_t S) {
    const int bw = mx_bw(Ci), nb = mx_nb(Ci);
    float acc[64] = {};
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < (size_t)N * S; i += (size_t)gridDim.x * blockDim.x) {
        int n = (int)(i / S); size_t v = i % S;
        float gv[4];
        for (int co = 0; co < Co && co < 4; co++) gv[co] = ldx(gy, ((size_t)n * Co + co) * S + v);
        for (int blk = 0; blk < nb; blk++) {
            float r[32];
            mx_load_row(x, x + (size_t)N * nb * S * bw, ((size_t)n * nb + blk) * S + v, bw, r);
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
extern "C" void lp_bwd_w1_mx(const void *x, shape5 xs, const void *gy, int gydt, shape5 ys, float *gw) {
    if (xs.c * ys.c > 64 || ys.c > 4) { fprintf(stderr, "lp_bwd_w1_mx: %d x %d channels unsupported\n", xs.c, ys.c); abort(); }
    size_t S = shape_spatial(xs);
    int nbk = (int)(((size_t)xs.n * S + 4095) / 4096); if (nbk > 1024) nbk = 1024;
    if (gydt == 2) conv_bwd_w1_mx_k<__half><<<nbk, 256>>>((const uint8_t *)x, (const __half *)gy, gw, xs.n, xs.c, ys.c, S);
    else if (gydt == 1) conv_bwd_w1_mx_k<bf16><<<nbk, 256>>>((const uint8_t *)x, (const bf16 *)gy, gw, xs.n, xs.c, ys.c, S);
    else conv_bwd_w1_mx_k<float><<<nbk, 256>>>((const uint8_t *)x, (const float *)gy, gw, xs.n, xs.c, ys.c, S);
    LPCK();
}
/* backward through silu(gn(x)) with an MX x (plane-major gy / gx of type TG / TO): a = gy * silu'(gn(x)).
   Pass 1 (stats): per (n, c) sums of a and a * xhat; block = 256 voxels of one sample, all channels (smem partials).
   Pass 2 (apply): gx = rstd (a gamma - A/len - xhat B/len) with the group sums A, B. */
template <typename TG>
__global__ void __launch_bounds__(256) gn_silu_bwd_stats_mx_k(const uint8_t *x, const TG *gy, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                                           int N, int C, int G, size_t S, double *ds) {
    const int bw = mx_bw(C), nb = mx_nb(C), cpg = C / G;
    const int nblk_per = (int)((S + 255) / 256);
    const int n = blockIdx.x / nblk_per; const size_t v = (size_t)(blockIdx.x % nblk_per) * 256 + threadIdx.x;
    __shared__ float s1[160], s2[160];
    for (int c = threadIdx.x; c < 2 * 80 && c < C; c += 256) { s1[c] = 0.f; s2[c] = 0.f; }
    __syncthreads();
    const uint8_t *sc = x + (size_t)N * nb * S * bw;
    for (int blk = 0; blk < nb; blk++) {
        float r[32], gr[32];
        if (v < S) {
            mx_load_row(x, sc, ((size_t)n * nb + blk) * S + v, bw, r);
            if constexpr (IS_MX8(TG)) mx_load_row((const uint8_t *)gy, (const uint8_t *)gy + (size_t)N * nb * S * bw, ((size_t)n * nb + blk) * S + v, bw, gr);
        }
        for (int k = 0; k < bw; k++) {
            int c = blk * bw + k;
            if (c >= C) break;
            float a = 0.f, axh = 0.f;
            if (v < S) {
                int ng = n * G + c / cpg;
                float xhat = (r[k] - mean[ng]) * rstd[ng], u = xhat * gamma[c] + beta[c], sg = 1.f / (1.f + __expf(-u));
                a = (IS_MX8(TG) ? gr[k] : ldx(gy, ((size_t)n * C + c) * S + v)) * (sg * (1.f + u * (1.f - sg)));
                axh = a * xhat;
            }
            for (int o = 16; o; o >>= 1) { a += __shfl_xor_sync(0xffffffff, a, o); axh += __shfl_xor_sync(0xffffffff, axh, o); }
            if ((threadIdx.x & 31) == 0) { atomicAdd(&s1[c], a); atomicAdd(&s2[c], axh); }
        }
    }
    __syncthreads();
    for (int c = threadIdx.x; c < C; c += 256) { atomicAdd(&ds[2 * ((size_t)n * C + c)], (double)s1[c]); atomicAdd(&ds[2 * ((size_t)n * C + c) + 1], (double)s2[c]); }
}
template <typename TG, typename TO>
__global__ void __launch_bounds__(256) gn_silu_bwd_apply_mx_k(const uint8_t *x, const TG *gy, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                                           const float *AB, TO *gx, int N, int C, int G, size_t S) {
    const int bw = mx_bw(C), nb = mx_nb(C), cpg = C / G;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * S) return;
    const int n = (int)(i / S); const size_t v = i % S;
    const uint8_t *sc = x + (size_t)N * nb * S * bw;
    const float len = (float)cpg * (float)S;
    for (int blk = 0; blk < nb; blk++) {
        float r[32], gr[32], out[32];
        const size_t ri = ((size_t)n * nb + blk) * S + v;
        mx_load_row(x, sc, ri, bw, r);
        if constexpr (IS_MX8(TG)) mx_load_row((const uint8_t *)gy, (const uint8_t *)gy + (size_t)N * nb * S * bw, ri, bw, gr);
#pragma unroll
        for (int k = 0; k < 32; k++) out[k] = 0.f;
        for (int k = 0; k < bw; k++) {
            int c = blk * bw + k;
            if (c >= C) break;
            int ng = n * G + c / cpg;
            float m = mean[ng], rs = rstd[ng], ga = gamma[c];
            float xhat = (r[k] - m) * rs, u = xhat * ga + beta[c], sg = 1.f / (1.f + __expf(-u));
            size_t o = ((size_t)n * C + c) * S + v;
            float a = (IS_MX8(TG) ? gr[k] : ldx(gy, o)) * (sg * (1.f + u * (1.f - sg)));
            float gv = rs * (a * ga - AB[2 * ng] / len - xhat * AB[2 * ng + 1] / len);
            if constexpr (IS_MX8(TO)) out[k] = gv; else stx(gx, o, gv);
        }
        if constexpr (IS_MX8(TO)) mx_store_row((uint8_t *)gx, (uint8_t *)gx + (size_t)N * nb * S * bw, ri, bw, out);
    }
}
extern "C" void lp_gn_silu_bwd_mx(const void *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                  const void *gy, void *gx, int gdt, double *ds, float *st, float *AB) {
    /* ds: 2 N C doubles (zeroed here), st: 2 N C floats, AB: 2 N G floats; the caller finishes with the group sums / param grads */
    size_t S = shape_spatial(s);
    int NC = s.n * s.c, nbp = (int)((S + 255) / 256);
    if (s.c > 160) { fprintf(stderr, "lp_gn_silu_bwd_mx: C %d > 160\n", s.c); abort(); }
    cudaMemsetAsync(ds, 0, (size_t)2 * NC * sizeof(double));
    if (gdt == 3) gn_silu_bwd_stats_mx_k<mx8_t><<<s.n * nbp, 256>>>((const uint8_t *)x, (const mx8_t *)gy, gamma, beta, mean, rstd, s.n, s.c, G, S, ds);
    else if (gdt == 2) gn_silu_bwd_stats_mx_k<__half><<<s.n * nbp, 256>>>((const uint8_t *)x, (const __half *)gy, gamma, beta, mean, rstd, s.n, s.c, G, S, ds);
    else if (gdt == 1) gn_silu_bwd_stats_mx_k<bf16><<<s.n * nbp, 256>>>((const uint8_t *)x, (const bf16 *)gy, gamma, beta, mean, rstd, s.n, s.c, G, S, ds);
    else gn_silu_bwd_stats_mx_k<float><<<s.n * nbp, 256>>>((const uint8_t *)x, (const float *)gy, gamma, beta, mean, rstd, s.n, s.c, G, S, ds);
    (void)st; (void)AB;
    LPCK();
}
extern "C" void lp_gn_silu_bwd_apply_mx(const void *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                        const void *gy, void *gx, int gdt, const float *AB) {
    size_t S = shape_spatial(s), n = (size_t)s.n * S;
    if (gdt == 3) gn_silu_bwd_apply_mx_k<mx8_t, mx8_t><<<nblk_(n, 256), 256>>>((const uint8_t *)x, (const mx8_t *)gy, gamma, beta, mean, rstd, AB, (mx8_t *)gx, s.n, s.c, G, S);
    else if (gdt == 2) gn_silu_bwd_apply_mx_k<__half, __half><<<nblk_(n, 256), 256>>>((const uint8_t *)x, (const __half *)gy, gamma, beta, mean, rstd, AB, (__half *)gx, s.n, s.c, G, S);
    else if (gdt == 1) gn_silu_bwd_apply_mx_k<bf16, bf16><<<nblk_(n, 256), 256>>>((const uint8_t *)x, (const bf16 *)gy, gamma, beta, mean, rstd, AB, (bf16 *)gx, s.n, s.c, G, S);
    else gn_silu_bwd_apply_mx_k<float, float><<<nblk_(n, 256), 256>>>((const uint8_t *)x, (const float *)gy, gamma, beta, mean, rstd, AB, (float *)gx, s.n, s.c, G, S);
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
extern "C" void lp_bwd_data_s2_mx(const void *gy, shape5 ys, const float *w, shape5 xs, void *gx, int accum) {
    size_t n = (size_t)xs.n * mx_nb(xs.c) * shape_spatial(xs), wb = (size_t)ys.c * xs.c * 27 * sizeof(float);
    if (wb <= 48 * 1024) bwd_data_s2_mx_k<1><<<nblk_(n, 128), 128, wb>>>((const uint8_t *)gy, w, (uint8_t *)gx, xs.n, xs.c, ys.c, xs.d, xs.h, xs.w, ys.d, ys.h, ys.w, accum);
    else bwd_data_s2_mx_k<0><<<nblk_(n, 128), 128>>>((const uint8_t *)gy, w, (uint8_t *)gx, xs.n, xs.c, ys.c, xs.d, xs.h, xs.w, ys.d, ys.h, ys.w, accum);
    LPCK();
}
