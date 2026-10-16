/* MX tensor conversions and rounding probes */
#include "lp_mxops.cuh"
/* Elementwise, 1x1, GroupNorm and upsample ops on MX activation tensors. */
#include "lp_common.cuh"

extern "C" size_t lp_mx8_bytes(int N, int C, size_t S) { return (size_t)N * mx_nb(C) * S * (mx_bw(C) + 1); }
extern "C" size_t lp_mx4_bytes(int N, int C, size_t S) { return (size_t)N * mx_nb(C) * S * (mx_bw(C) / 2 + 1); }

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
/* the network input of predict (nn_pred_input: z-scored CT, 0, and the unit direction away from the axis) for window planes
   [z0, z0 + nz), each value rounded to the 16-bit input type (HD 2 fp16, 1 bf16, 0 none) and stored as MX rows (B 8: fp8, 4: fp4)
   of one 8-channel block (channels 4..7 zero): bit for bit nn_pred_input followed by the 16-bit -> MX input conversion, without the
   16-bit window. y: an MX tensor of 4 channels and depth nz. */
template <int HD> __device__ __forceinline__ float in_round(float v) {
    if constexpr (HD == 2) return __half2float(__float2half(v));
    else if constexpr (HD == 1) return __bfloat162float(__float2bfloat16(v));
    else return v;
}
template <int B, int HD>
__global__ void pred_in_mx_k(const uint8_t *ct, int W, float mean, float isd, const float *dyo, const float *dxo, int axis, int z0, int nz, uint8_t *y) {
    const size_t S = (size_t)nz * W * W, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= S) return;
    const int xx = (int)(i % W), yy = (int)((i / W) % W), z = z0 + (int)(i / ((size_t)W * W));
    const float dy = dyo[z] + (float)yy, dx = dxo[z] + (float)xx, inv = axis ? (float)axis / (sqrtf(dy * dy + dx * dx) + 1e-6f) : 0.f;
    float r[32];
#pragma unroll
    for (int k = 0; k < 32; k++) r[k] = 0.f;
    r[0] = in_round<HD>(((float)ct[(size_t)z * W * W + (size_t)yy * W + xx] - mean) * isd); r[1] = in_round<HD>(0.f);
    r[2] = in_round<HD>(dy * inv); r[3] = in_round<HD>(dx * inv);
    mx_store_row_b<B>(y, mx_sc<B>(y, 1, 4, S), i, 8, r);
}
extern "C" void lp_pred_input_mx(const uint8_t *ct, int W, float mean, float isd, const float *dyo, const float *dxo, int axis, int z0, int nz, void *y, int hd, int bits) {
    const size_t S = (size_t)nz * W * W;
    uint8_t *yy = (uint8_t *)y;
    if (bits == 4) {
        if (hd == 2) pred_in_mx_k<4, 2><<<nblk_(S, 256), 256>>>(ct, W, mean, isd, dyo, dxo, axis, z0, nz, yy);
        else if (hd == 1) pred_in_mx_k<4, 1><<<nblk_(S, 256), 256>>>(ct, W, mean, isd, dyo, dxo, axis, z0, nz, yy);
        else pred_in_mx_k<4, 0><<<nblk_(S, 256), 256>>>(ct, W, mean, isd, dyo, dxo, axis, z0, nz, yy);
    } else {
        if (hd == 2) pred_in_mx_k<8, 2><<<nblk_(S, 256), 256>>>(ct, W, mean, isd, dyo, dxo, axis, z0, nz, yy);
        else if (hd == 1) pred_in_mx_k<8, 1><<<nblk_(S, 256), 256>>>(ct, W, mean, isd, dyo, dxo, axis, z0, nz, yy);
        else pred_in_mx_k<8, 0><<<nblk_(S, 256), 256>>>(ct, W, mean, isd, dyo, dxo, axis, z0, nz, yy);
    }
    LPCK();
}
extern "C" void lp_pred_input_mx8(const uint8_t *ct, int W, float mean, float isd, const float *dyo, const float *dxo, int axis, int z0, int nz, void *y, int hd) {
    lp_pred_input_mx(ct, W, mean, isd, dyo, dxo, axis, z0, nz, y, hd, 8);
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
/* train --gpu-input (nn.h nn_input_rad): the 4 input channels of sample.c write_x from the 16-bit CT channel and the sample's radial
   record, each rounded to the 16-bit type, then one MX row of 8 channels (4..7 zero) as f32_to_mx_k. The radial vector with IEEE
   intrinsics (the build uses -use_fast_math, which would contract dy dy + dx dx and approximate sqrt / division): the host's ISO C
   float expressions (no contraction), bit for bit. */
template <typename TI> __device__ __forceinline__ float rnd16(float v);
template <> __device__ __forceinline__ float rnd16<__half>(float v) { return __half2float(__float2half_rn(v)); }
template <> __device__ __forceinline__ float rnd16<bf16>(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
template <int B, typename TI>
__global__ void input_rad_k(const TI *ct, const uint8_t *recs, size_t rec_bytes, int N, int D, int P, uint8_t *y) {   /* grid (x blocks, y, n D) */
    const int ix = blockIdx.x * blockDim.x + threadIdx.x, iy = blockIdx.y, n = blockIdx.z / D, iz = blockIdx.z - n * D;   /* (the voxel without 64-bit
                                                                                                                     divisions: they were most of the time) */
    if (ix >= P) return;
    const size_t S = (size_t)D * P * P, i = (((size_t)n * D + iz) * P + iy) * P + ix;
    const int *h = (const int *)(recs + (size_t)n * rec_bytes);
    const float *cyz = (const float *)(h + 8), *cxz = cyz + h[2];
    const int o1 = h[0], o2 = h[1], hasax = h[4], pf = h[5];
    const int io[3] = {iz, iy, ix}, nd[3] = {D, P, P};
    int s[3] = {0, 0, 0};
#pragma unroll
    for (int d = 0; d < 3; d++) { const int pd = (pf >> (2 * d)) & 3; s[pd] = (pf >> (8 + d)) & 1 ? nd[d] - 1 - io[d] : io[d]; }   /* output axis d reads source axis perm[d] */
    float r[32];
#pragma unroll
    for (int k = 0; k < 32; k++) r[k] = 0.f;
    r[0] = ldx(ct, i);
    if (h[6] & 1) {   /* a warped window: sample.c write_spatial's transported radial vector, the host's float expressions in order */
        const float *tab = cxz + D, *R = tab + 6 * P, *A = R + 9, *Ai = A + 9; const float sc = R[27];   /* R[9], A[9], Ainv[9], scale */
        const float c = __fmul_rn(.5f, (float)(P - 1)), fP1 = (float)(P - 1);
        float v[3], jac[3], u[3], src[3];
#pragma unroll
        for (int d = 0; d < 3; d++) {
            v[d] = __fsub_rn(__fsub_rn((float)io[d], c), tab[d * P + io[d]]);
            jac[d] = __fdiv_rn(1.f, __fsub_rn(1.f, tab[(3 + d) * P + io[d]]));
        }
        if (h[6] & 2) {
            float w[3];
#pragma unroll
            for (int d = 0; d < 3; d++) w[d] = __fadd_rn(__fadd_rn(__fmul_rn(R[d], v[0]), __fmul_rn(R[3 + d], v[1])), __fmul_rn(R[6 + d], v[2]));
#pragma unroll
            for (int d = 0; d < 3; d++) src[d] = __fadd_rn(c, __fmul_rn(sc, __fadd_rn(__fadd_rn(__fmul_rn(A[3 * d], w[0]), __fmul_rn(A[3 * d + 1], w[1])), __fmul_rn(A[3 * d + 2], w[2]))));
        } else {
#pragma unroll
            for (int d = 0; d < 3; d++) src[d] = __fadd_rn(c, __fmul_rn(sc, __fadd_rn(__fadd_rn(__fmul_rn(R[d], v[0]), __fmul_rn(R[3 + d], v[1])), __fmul_rn(R[6 + d], v[2]))));
        }
        int perm[3], flip[3];
#pragma unroll
        for (int d = 0; d < 3; d++) { perm[d] = (pf >> (2 * d)) & 3; flip[d] = (pf >> (8 + d)) & 1; }
        for (int d = 0; d < 3; d++) u[perm[d]] = flip[d] ? __fsub_rn(fP1, src[d]) : src[d];   /* (the source frame: write_spatial's v) */
        int inside = 1;
#pragma unroll
        for (int d = 0; d < 3; d++) if (u[d] < 0.f || u[d] > fP1) inside = 0;
        if (inside) {
            int b0 = (int)u[0]; if (b0 >= P - 1) b0 = P - 2;
            const float frac = __fsub_rn(u[0], (float)b0);
            const float ayy = __fadd_rn(__fmul_rn(cyz[b0], __fsub_rn(1.f, frac)), __fmul_rn(cyz[b0 + 1], frac));
            const float axx = __fadd_rn(__fmul_rn(cxz[b0], __fsub_rn(1.f, frac)), __fmul_rn(cxz[b0 + 1], frac));
            const float dy = __fsub_rn(__fadd_rn((float)o1, u[1]), ayy), dx = __fsub_rn(__fadd_rn((float)o2, u[2]), axx);
            const float radius = __fadd_rn(__fsqrt_rn(__fadd_rn(__fmul_rn(dy, dy), __fmul_rn(dx, dx))), 1e-6f);
            const float radial[3] = {0.f, hasax ? __fdiv_rn(dy, radius) : 0.f, hasax ? __fdiv_rn(dx, radius) : 0.f};
            float disc[3], tr[3];
#pragma unroll
            for (int d = 0; d < 3; d++) disc[d] = __fmul_rn(flip[d] ? -1.f : 1.f, radial[perm[d]]);
            if (h[6] & 2) {
                float q[3];
#pragma unroll
                for (int d = 0; d < 3; d++) q[d] = __fadd_rn(__fadd_rn(__fmul_rn(Ai[3 * d], disc[0]), __fmul_rn(Ai[3 * d + 1], disc[1])), __fmul_rn(Ai[3 * d + 2], disc[2]));
#pragma unroll
                for (int d = 0; d < 3; d++) disc[d] = q[d];
            }
            float norm = 1e-12f;
#pragma unroll
            for (int d = 0; d < 3; d++) {
                tr[d] = __fmul_rn(jac[d], __fadd_rn(__fadd_rn(__fmul_rn(R[3 * d], disc[0]), __fmul_rn(R[3 * d + 1], disc[1])), __fmul_rn(R[3 * d + 2], disc[2])));
                norm = __fadd_rn(norm, __fmul_rn(tr[d], tr[d]));
            }
            norm = __fsqrt_rn(norm);
#pragma unroll
            for (int d = 0; d < 3; d++) r[1 + d] = rnd16<TI>(__fdiv_rn(tr[d], norm));
        }
        mx_store_row_b<B>(y, mx_sc<B>(y, N, 4, S), i, 8, r);
        return;
    }
    float comp[3] = {0.f, 0.f, 0.f};
    if (hasax) {
        const float dy = __fsub_rn((float)(o1 + s[1]), cyz[s[0]]), dx = __fsub_rn((float)(o2 + s[2]), cxz[s[0]]);
        const float inv = __fdiv_rn(1.f, __fadd_rn(__fsqrt_rn(__fadd_rn(__fmul_rn(dy, dy), __fmul_rn(dx, dx))), 1e-6f));
        comp[1] = __fmul_rn(dy, inv); comp[2] = __fmul_rn(dx, inv);
    }
#pragma unroll
    for (int d = 0; d < 3; d++) r[1 + d] = rnd16<TI>(__fmul_rn((pf >> (8 + d)) & 1 ? -1.f : 1.f, comp[(pf >> (2 * d)) & 3]));   /* sign x component (signed zeros as the host) */
    mx_store_row_b<B>(y, mx_sc<B>(y, N, 4, S), i, 8, r);
}
/* --half-res pack with --gpu-input: the network's 8 input channels on the 2x2x2 blocks of the window, straight from the full-resolution
   16-bit CT channel and the radial records: each sub-voxel's 4 channels as input_rad_k (rounded to the 16-bit type), then nn_half_input
   mode 2 (the block means of the 4 channels, the CT's Haar details along z, y, x and its mean absolute deviation, each rounded to the
   16-bit type), then one 8-channel MX row as f32_to_mx_k: bit for bit the host-built chain (sample.c write_x -> nn_half_input ->
   nn_h16_to_mx) without its full-resolution 4-channel window. One thread per block; d, h (= w): the network grid */
template <typename TI> __device__ __forceinline__ void rad_voxel(const TI *ct, const int *h, size_t i, int fz, int fy, int fx, const int *nd, float *r4) {
    const float *cyz = (const float *)(h + 8), *cxz = cyz + h[2];
    const int o1 = h[0], o2 = h[1], hasax = h[4], pf = h[5];
    const int io[3] = {fz, fy, fx};
    int s[3] = {0, 0, 0};
#pragma unroll
    for (int d = 0; d < 3; d++) { const int pd = (pf >> (2 * d)) & 3; s[pd] = (pf >> (8 + d)) & 1 ? nd[d] - 1 - io[d] : io[d]; }
    float comp[3] = {0.f, 0.f, 0.f};
    if (hasax) {
        const float dy = __fsub_rn((float)(o1 + s[1]), cyz[s[0]]), dx = __fsub_rn((float)(o2 + s[2]), cxz[s[0]]);
        const float inv = __fdiv_rn(1.f, __fadd_rn(__fsqrt_rn(__fadd_rn(__fmul_rn(dy, dy), __fmul_rn(dx, dx))), 1e-6f));
        comp[1] = __fmul_rn(dy, inv); comp[2] = __fmul_rn(dx, inv);
    }
    r4[0] = ldx(ct, i);
#pragma unroll
    for (int d = 0; d < 3; d++) r4[1 + d] = rnd16<TI>(__fmul_rn((pf >> (8 + d)) & 1 ? -1.f : 1.f, comp[(pf >> (2 * d)) & 3]));
}
template <int B, typename TI>
__global__ void input_rad_pack_k(const TI *ct, const uint8_t *recs, size_t rec_bytes, int N, int d, int p, uint8_t *y) {
    const size_t hv = (size_t)d * p * p, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * hv) return;
    const int n = (int)(i / hv); const size_t v = i % hv;
    const int z = (int)(v / ((size_t)p * p)), yy = (int)((v / p) % p), xx = (int)(v % p);
    const int D = 2 * d, P = 2 * p, nd[3] = {D, P, P};
    const size_t S = (size_t)D * P * P;
    const int *h = (const int *)(recs + (size_t)n * rec_bytes);
    float q[4][8];
#pragma unroll
    for (int k = 0; k < 8; k++) {
        const int fz = 2 * z + (k >> 2), fy = 2 * yy + (k >> 1 & 1), fx = 2 * xx + (k & 1);
        float r4[4];
        rad_voxel<TI>(ct, h, (size_t)n * S + ((size_t)fz * P + fy) * P + fx, fz, fy, fx, nd, r4);
#pragma unroll
        for (int c = 0; c < 4; c++) q[c][k] = r4[c];
    }
    float r[32];
#pragma unroll
    for (int k = 0; k < 32; k++) r[k] = 0.f;
    float m0 = 0.f;
#pragma unroll
    for (int c = 0; c < 4; c++) {   /* half_in_k: the block mean (fp32 sum in sub-voxel order) */
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < 8; k++) s += q[c][k];
        const float m = 0.125f * s;
        if (!c) m0 = m;
        r[c] = rnd16<TI>(m);
    }
    float g[3] = {0.f, 0.f, 0.f}, a = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) { g[0] += k & 4 ? q[0][k] : -q[0][k]; g[1] += k & 2 ? q[0][k] : -q[0][k]; g[2] += k & 1 ? q[0][k] : -q[0][k]; a += fabsf(q[0][k] - m0); }
#pragma unroll
    for (int e = 0; e < 3; e++) r[4 + e] = rnd16<TI>(0.125f * g[e]);
    r[7] = rnd16<TI>(0.125f * a);
    mx_store_row_b<B>(y, mx_sc<B>(y, N, 8, hv), i, 8, r);
}
extern "C" void lp_input_rad_pack(const void *ct16, int dt, const void *rec, size_t rec_bytes, int N, int d, int p, void *y, int bits) {
    const size_t n = (size_t)N * d * p * p;
    const uint8_t *rr = (const uint8_t *)rec; uint8_t *yy = (uint8_t *)y;
#define IRP(B, T) input_rad_pack_k<B, T><<<nblk_(n, 256), 256>>>((const T *)ct16, rr, rec_bytes, N, d, p, yy)
    if (bits == 4) { if (dt == 2) IRP(4, __half); else IRP(4, bf16); }
    else { if (dt == 2) IRP(8, __half); else IRP(8, bf16); }
#undef IRP
    LPCK();
}
extern "C" void lp_input_rad(const void *ct16, int dt, const void *rec, size_t rec_bytes, int N, int D, int P, void *y, int bits) {
    const uint8_t *rr = (const uint8_t *)rec; uint8_t *yy = (uint8_t *)y;
    const int bx = P < 128 ? (P + 31) / 32 * 32 : 128;
    const dim3 grid((unsigned)((P + bx - 1) / bx), (unsigned)P, (unsigned)(N * D));
    if (bits == 4) { if (dt == 2) input_rad_k<4, __half><<<grid, bx>>>((const __half *)ct16, rr, rec_bytes, N, D, P, yy); else input_rad_k<4, bf16><<<grid, bx>>>((const bf16 *)ct16, rr, rec_bytes, N, D, P, yy); }
    else { if (dt == 2) input_rad_k<8, __half><<<grid, bx>>>((const __half *)ct16, rr, rec_bytes, N, D, P, yy); else input_rad_k<8, bf16><<<grid, bx>>>((const bf16 *)ct16, rr, rec_bytes, N, D, P, yy); }
    LPCK();
}
/* consistent fp4 weights (unet_cfg.w4_2d): w[co][ci][t] rounded in place to MX-fp4 with one ue8m0 scale per B x B (co x ci) block
   and tap (mx_exp of the block's absolute maximum, round to nearest as the weight preps): every per-input-channel block of the forward
   preps and per-output-channel block of the backward-data preps inside one such block re-quantises these values exactly (a smaller
   scale 2^-j only doubles grid values <= 6 / 2^j, which stay on the e2m1 grid). One warp per (tap, block) */
__global__ void fq_w4_2d_k(float *w, int Co, int Ci, int T, int Bo, int Bi) {   /* (Bo x Bi blocks: 1 x 16 / 32 = the forward preps' own
                                                                                     blocks, 16 / 32 x 1 backward-data's, for tests) */
    const int nbo = (Co + Bo - 1) / Bo, nbi = (Ci + Bi - 1) / Bi, lane = threadIdx.x & 31;
    const size_t wid = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5;
    if (wid >= (size_t)T * nbo * nbi) return;
    const int t = (int)(wid % T), bi = (int)((wid / T) % nbi), bo = (int)(wid / ((size_t)T * nbi));
    const int o0 = bo * Bo, i0 = bi * Bi, no = min(Bo, Co - o0), ni = min(Bi, Ci - i0);
    float amax = 0.f;
    for (int e = lane; e < no * ni; e += 32) amax = fmaxf(amax, fabsf(w[((size_t)(o0 + e / ni) * Ci + i0 + e % ni) * T + t]));
#pragma unroll
    for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    const int ex = mx_exp(amax, 1.f / 6.f);
    const float m = exp2i(-ex), s = exp2i(ex);
    for (int e = lane; e < no * ni; e += 32) {
        float *p = w + ((size_t)(o0 + e / ni) * Ci + i0 + e % ni) * T + t;
        *p = dec_e2m1n(cvt_e2m1x2(*p * m, 0.f) & 15u) * s;
    }
}
extern "C" void lp_fakeq_w4(float *w, int Co, int Ci, int T, int Bo, int Bi) {
    const size_t nw = (size_t)T * ((Co + Bo - 1) / Bo) * ((Ci + Bi - 1) / Bi);
    fq_w4_2d_k<<<nblk_(nw * 32, 128), 128>>>(w, Co, Ci, T, Bo, Bi);
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
