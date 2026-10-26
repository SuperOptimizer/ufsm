#pragma once
/* device helpers shared by the MX elementwise / 1x1 / GroupNorm / upsample units (lp_mx_*.cu) */
#include "lp_common.cuh"
/* ======================= elementwise ops on MX-fp8 activation tensors =======================
   Voxel-major over channel blocks: a thread owns one (n, block, voxel) row of bw channels, so the per-voxel amax that
   defines the output scale is local. */
/* row ri = (n*nb + blk)*S + voxel of a B-bit MX tensor; q = data, sc = scale plane */
template <int B> __device__ __forceinline__ void mx_load_row_b(const uint8_t *q, const uint8_t *sc, size_t ri, int bw, float *v) {
    mxf<B>::dec_row(q + ri * mx_rb(bw, B), bw, mx_scale(sc[ri]), v);
}
/* mxf<B>::dec_row of a whole 32-channel row already in registers (B / 4 16-byte words; dec_row loads through __ldg, which must not be
   given a local array): the same decode operations in the same order (prefetching kernels: the row loaded a step ahead) */
template <int B> __device__ __forceinline__ void dec_row32_regs(const uint4 *raw, float s, float *v) {
    if constexpr (B == 4) {
        const unsigned w[4] = {raw[0].x, raw[0].y, raw[0].z, raw[0].w};
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int q = 0; q < 4; q++) { float2 a = dec_e2m1x2(w[j] >> (8 * q)); v[8 * j + 2 * q] = a.x * s; v[8 * j + 2 * q + 1] = a.y * s; }
    } else {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const unsigned w[4] = {raw[h].x, raw[h].y, raw[h].z, raw[h].w};
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float2 a = dec_e4m3x2((unsigned short)(w[j] & 0xffff)), b = dec_e4m3x2((unsigned short)(w[j] >> 16));
                v[16 * h + 4 * j] = a.x * s; v[16 * h + 4 * j + 1] = a.y * s; v[16 * h + 4 * j + 2] = b.x * s; v[16 * h + 4 * j + 3] = b.y * s;
            }
        }
    }
}
/* the MX-fp4 row of bw <= BW values with exact SR (sr != 0) as mx_store_row_b<4> below stores it (the same operations, kept apart so
   that function's code stays as it was): words w[0 .. BW / 8) (0 past bw) and the block exponent (scale byte e + 127) */
template <int BW> __device__ __forceinline__ int mx4_sr_row(const float *v, int bw, size_t ri, uint32_t sr, unsigned *w) {
    unsigned am = 0u;
#pragma unroll
    for (int k = 0; k < BW; k++) if (k < bw) am = amax_u(am, v[k]);
    const int e = mx_exp(__uint_as_float(am), mxf<4>::inv_qmax);
    const float m = exp2i(-e);
#pragma unroll
    for (int j = 0; j < BW / 8; j++) { w[j] = 0u; if (8 * j < bw) { uint32_t hh[4]; sr_hash4(sr, ri * 4 + j, hh); w[j] = sr_e2m1_word(v + 8 * j, m, hh); } }
    return e;
}
/* sr != 0: exact stochastic rounding keyed by (sr, ri, channel) (fp4 only; fp8 keeps round-to-nearest here) */
template <int B> __device__ __forceinline__ void mx_store_row_b(uint8_t *q, uint8_t *sc, size_t ri, int bw, float *v, uint32_t sr = 0) {
    unsigned am = 0u;
#pragma unroll
    for (int k = 0; k < 32; k++) if (k < bw) am = amax_u(am, v[k]);
    const int e = mx_exp(__uint_as_float(am), mxf<B>::inv_qmax);
    const float m = exp2i(-e);
    if (B == 4 && sr) {   /* exact SR with 16-bit uniforms straight to nibbles: one hash + 3 remixes per 8 values, keyed by (sr, row,
                             8-value group) (a full hash, SR to a float and the e2m1 conversion per value made the GN backward apply
                             compute-bound) */
        unsigned w[4];
#pragma unroll
        for (int j = 0; j < 4; j++) { w[j] = 0u; if (8 * j < bw) { uint32_t hh[4]; sr_hash4(sr, ri * 4 + j, hh); w[j] = sr_e2m1_word(v + 8 * j, m, hh); } }
        uint8_t *p = q + ri * mx_rb(bw, B);
        if (bw == 8) *(unsigned *)p = w[0];
        else { *(uint2 *)p = make_uint2(w[0], w[1]); if (bw == 32) *((uint2 *)p + 1) = make_uint2(w[2], w[3]); }
        sc[ri] = (uint8_t)(e + 127);
        return;
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
#define MXB(dt) ((dt) == 4 ? 4 : 8)
template <typename TI> __device__ __forceinline__ void pm_load_row(const TI *x, int N, int C, size_t S, size_t ri, int bw, float *r) {
    const int nb = mx_nb(C);
    const size_t v = ri % S; const int blk = (int)((ri / S) % nb), n = (int)(ri / (S * nb));
#pragma unroll
    for (int k = 0; k < 32; k++) { int c = blk * bw + k; r[k] = k < bw && c < C ? ldx(x, ((size_t)n * C + c) * S + v) : 0.f; }
}
__device__ __forceinline__ void row_gn(float *r, const gnp_t &gp, int n, int blk, int bw, int Ci) {
    if (!gp.G) return;
    const int cpg = Ci / gp.G;
    for (int k = 0; k < bw; k++) {
        int ci = blk * bw + k; if (ci >= Ci) break;
        int ng = n * gp.G + ci / cpg; float a = gp.rstd[ng] * gp.gamma[ci], b = gp.beta[ci] - gp.mean[ng] * a;
        r[k] = act_ab(r[k], a, b, true);
    }
}
template <int B> __device__ __forceinline__ void mx_load16(const uint8_t *q, const uint8_t *sc, size_t ri, int bw, int h0, float *v) {
    const float s = mx_scale(sc[ri]);
    const uint8_t *p = q + ri * mx_rb(bw, B) + h0 * B / 8;
    if constexpr (B == 8) {   /* (an 8-channel row: 8 bytes, the rest zero) */
        const uint4 u = bw == 8 ? make_uint4(__ldg((const unsigned *)p), __ldg((const unsigned *)p + 1), 0u, 0u) : __ldg((const uint4 *)p);
        const unsigned w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 a = dec_e4m3x2((unsigned short)(w[j] & 0xffff)), b = dec_e4m3x2((unsigned short)(w[j] >> 16));
            v[4 * j] = a.x * s; v[4 * j + 1] = a.y * s; v[4 * j + 2] = b.x * s; v[4 * j + 3] = b.y * s;
        }
    } else {   /* (an 8-channel row: 4 bytes) */
        const uint2 u = bw == 8 ? make_uint2(__ldg((const unsigned *)p), 0u) : __ldg((const uint2 *)p);
        const unsigned w[2] = {u.x, u.y};
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int q2 = 0; q2 < 4; q2++) { float2 a = dec_e2m1x2(w[j] >> (8 * q2)); v[8 * j + 2 * q2] = a.x * s; v[8 * j + 2 * q2 + 1] = a.y * s; }
    }
}
