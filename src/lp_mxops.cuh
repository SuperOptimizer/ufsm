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
