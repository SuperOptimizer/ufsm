/* GroupNorm+SiLU backward on MX tensors and 1x1 conv to MX */
#include "lp_mxops.cuh"
/* d silu(u) / du = sg (1 + u (1 - sg)), sg = sigmoid(u) = 1/2 + tanh(u / 2) / 2: one MUFU op (exp + reciprocal took two;
   tanh.approx ~2^-11 relative, far below the e4m3 rounding of the stored gradient) */
__device__ __forceinline__ float dsilu(float u) {
    float th; asm("tanh.approx.f32 %0, %1;" : "=f"(th) : "f"(0.5f * u));
    const float sg = fmaf(0.5f, th, 0.5f);
    return sg * (1.f + u * (1.f - sg));
}
template <int B, typename TG, int V, int HPB>   /* V consecutive voxels per thread step (vector gy loads; 1: any S); HPB 16-channel halves per row */
__global__ void __launch_bounds__(256) gn_silu_bwd_stats_mx_k(const uint8_t *x, const TG *gy, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                                           int N, int C, int G, size_t S, float *part) {
    /* block = (voxel slab, stored channel block, sample); a thread holds 16 channels (registers: occupancy). 32-wide blocks: the
       lane pair (2 k, 2 k + 1) takes the two halves of one voxel's row, so a warp reads 16 whole contiguous rows (a block per
       half read every row twice from DRAM: its partner block ran a whole grid row later) */
    const int bw = mx_bw(C), nb = mx_nb(C), cpg = C / G;
    const int blk = blockIdx.y, n = blockIdx.z, nsl = gridDim.x;
    const int h0 = HPB == 2 ? 16 * (int)(threadIdx.x & 1) : 0, tv = (int)threadIdx.x / HPB, nt = 256 / HPB;
    const size_t nq = S / V, q0 = nq * blockIdx.x / nsl, q1 = nq * (blockIdx.x + 1) / nsl;   /* slab in units of V voxels */
    /* per-warp partials summed in warp order below (shared-memory float atomics made the sums, hence the stored gradients'
       MX rounding, depend on the warps' timing: identical runs differed) */
    __shared__ float cm[32], cr[32], cg[32], cb[32], w1[8][32], w2[8][32];
    if (threadIdx.x < 32) {
        const int k = threadIdx.x, c = blk * bw + k;
        const bool ok = k < bw && c < C;
        const int ng = n * G + (ok ? c / cpg : 0);
        cm[k] = ok ? mean[ng] : 0.f; cr[k] = ok ? rstd[ng] : 0.f; cg[k] = ok ? gamma[c] : 0.f; cb[k] = ok ? beta[c] : 0.f;
    }
    w1[threadIdx.x >> 5][threadIdx.x & 31] = 0.f; w2[threadIdx.x >> 5][threadIdx.x & 31] = 0.f;
    __syncthreads();
    const uint8_t *sc = mx_sc<B>(x, N, C, S);
    float a1[16], a2[16];
#pragma unroll
    for (int k = 0; k < 16; k++) { a1[k] = 0.f; a2[k] = 0.f; }
    const int kmax = max(0, min(min(bw, C - blk * bw) - h0, 16));
    const size_t rbase = ((size_t)n * nb + blk) * S;
    const TG *gyb = IS_MX(TG) ? gy : gy + ((size_t)n * C + blk * bw + h0) * S;
    for (size_t q = q0 + tv; q < q1; q += nt) {
        const size_t v = q * V;
        float g[V][16];
        if constexpr (!IS_MX(TG)) {
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
            if constexpr (IS_MX(TG)) mx_load16<MX_BITS(TG)>((const uint8_t *)gy, mx_sc<MX_BITS(TG)>((const uint8_t *)gy, N, C, S), rbase + v + j, bw, h0, gr);
#pragma unroll
            for (int k = 0; k < 16; k++) {
                if (k < kmax) {
                    const float xhat = (r[k] - cm[h0 + k]) * cr[h0 + k], u = xhat * cg[h0 + k] + cb[h0 + k];
                    const float a = (IS_MX(TG) ? gr[k] : g[j][k]) * dsilu(u);
                    a1[k] += a; a2[k] += a * xhat;
                }
            }
        }
    }
#pragma unroll
    for (int k = 0; k < 16; k++) {   /* kmax differs between the halves only in its padding tail: reduce all 16, add the real ones */
        float a = a1[k], q = a2[k];
        for (int o = 16; o >= HPB; o >>= 1) { a += __shfl_xor_sync(0xffffffff, a, o); q += __shfl_xor_sync(0xffffffff, q, o); }
        if ((int)(threadIdx.x & 31) < HPB && k < kmax) { w1[threadIdx.x >> 5][h0 + k] = a; w2[threadIdx.x >> 5][h0 + k] = q; }
    }
    __syncthreads();
    /* per-slab partials (same-address double atomics across slabs serialised at the small levels); summed by gn_part_sum_k */
    const int kb = min(bw, C - blk * bw);
    if ((int)threadIdx.x < kb) {
        float a = 0.f, q = 0.f;
#pragma unroll
        for (int w = 0; w < 8; w++) { a += w1[w][threadIdx.x]; q += w2[w][threadIdx.x]; }
        const size_t c = (size_t)n * C + blk * bw + threadIdx.x, o = ((size_t)blockIdx.x * N * C + c) * 2; part[o] = a; part[o + 1] = q;
    }
}
/* the same for MX-fp4 gy with folded coefficients (Ah, Bh, mean, rstd in one float4: p = x Ah + Bh = u / 2) and a one-step
   raw-word prefetch: accumulates sum t and sum t (x - mean) with t = gy (2 dsilu(u)) and finishes with the factors 1/2 and
   rstd / 2, so part[] holds the same sums (sum a, sum a xhat) as gn_silu_bwd_stats_mx_k. The generic kernel loaded six
   coefficients and decoded through per-element guards (issue-bound, ~2x its traffic floor). */
template <int B, int HPB, bool FULL>
__global__ void __launch_bounds__(256) gn_silu_bwd_stats_mx2_k(const uint8_t *x, const uint8_t *gy, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                                            int N, int C, int G, size_t S, float *part) {
    const int bw = mx_bw(C), nb = mx_nb(C), cpg = C / G;
    const int blk = blockIdx.y, n = blockIdx.z, nsl = gridDim.x;
    const int h = HPB == 2 ? (int)(threadIdx.x & 1) : 0, h0 = 16 * h, tv = (int)threadIdx.x / HPB, nt = 256 / HPB;
    const size_t q0 = S * blockIdx.x / nsl, q1 = S * (blockIdx.x + 1) / nsl;
    __shared__ float4 cf[32];
    __shared__ float w1[8][32], w2[8][32];
    if (threadIdx.x < 32) {
        const int k = threadIdx.x, c = blk * bw + k;
        const bool ok = k < bw && c < C;
        const int ng = n * G + (ok ? c / cpg : 0);
        const float rs = ok ? rstd[ng] : 0.f, mu = ok ? mean[ng] : 0.f, A = ok ? rs * gamma[c] : 0.f, be = ok ? beta[c] : 0.f;
        cf[k] = make_float4(0.5f * A, 0.5f * (be - mu * A), mu, rs);
    }
    w1[threadIdx.x >> 5][threadIdx.x & 31] = 0.f; w2[threadIdx.x >> 5][threadIdx.x & 31] = 0.f;
    __syncthreads();
    const int kmax = FULL ? 16 : max(0, min(min(bw, C - blk * bw) - h0, 16));
    const size_t rbase = ((size_t)n * nb + blk) * S;
    const uint8_t *sx = mx_sc<B>(x, N, C, S), *sg = mx_sc<4>(gy, N, C, S);
    const int rbx = mx_rb(bw, B), rbg = mx_rb(bw, 4);
    float s1[16], s2[16];
#pragma unroll
    for (int k = 0; k < 16; k++) { s1[k] = 0.f; s2[k] = 0.f; }
    uint4 xw = make_uint4(0u, 0u, 0u, 0u); uint2 gw = make_uint2(0u, 0u); unsigned xs = 0u, gs = 0u;
    auto load = [&](size_t v) {
        const size_t ri = rbase + v;
        if constexpr (B == 4) { const uint2 t = bw == 8 ? make_uint2(__ldg((const unsigned *)(x + ri * rbx)), 0u) : __ldg((const uint2 *)(x + ri * rbx + 8 * h)); xw.x = t.x; xw.y = t.y; }
        else xw = __ldg((const uint4 *)(x + ri * rbx + 16 * h));
        gw = bw == 8 ? make_uint2(__ldg((const unsigned *)(gy + ri * rbg)), 0u) : __ldg((const uint2 *)(gy + ri * rbg + 8 * h));
        xs = __ldg(sx + ri); gs = __ldg(sg + ri);
    };
    size_t q = q0 + tv;
    if (q < q1) load(q);
    for (; q < q1; q += nt) {
        const uint4 cx = xw; const uint2 cg = gw; const float fx = mx_scale(xs), fg = mx_scale(gs);
        if (q + nt < q1) load(q + nt);   /* next row in flight during this one's arithmetic */
        float r[16], g[16];
        if constexpr (B == 4) {
            const unsigned w4[2] = {cx.x, cx.y};
#pragma unroll
            for (int j = 0; j < 2; j++)
#pragma unroll
                for (int i = 0; i < 4; i++) { const float2 a = dec_e2m1x2(w4[j] >> (8 * i)); r[8 * j + 2 * i] = a.x * fx; r[8 * j + 2 * i + 1] = a.y * fx; }
        } else {
            const unsigned w8[4] = {cx.x, cx.y, cx.z, cx.w};
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const float2 a = dec_e4m3x2((unsigned short)(w8[j] & 0xffff)), b2 = dec_e4m3x2((unsigned short)(w8[j] >> 16));
                r[4 * j] = a.x * fx; r[4 * j + 1] = a.y * fx; r[4 * j + 2] = b2.x * fx; r[4 * j + 3] = b2.y * fx;
            }
        }
        {
            const unsigned w4[2] = {cg.x, cg.y};
#pragma unroll
            for (int j = 0; j < 2; j++)
#pragma unroll
                for (int i = 0; i < 4; i++) { const float2 a = dec_e2m1x2(w4[j] >> (8 * i)); g[8 * j + 2 * i] = a.x; g[8 * j + 2 * i + 1] = a.y; }
        }
#pragma unroll
        for (int k = 0; k < 16; k++) {
            if (FULL || k < kmax) {
                const float4 c4 = cf[h0 + k];
                const float p = fmaf(r[k], c4.x, c4.y);
                float th; asm("tanh.approx.f32 %0, %1;" : "=f"(th) : "f"(p));
                const float t = g[k] * ((1.f + th) * (fmaf(-p, th, p) + 1.f)) * fg;   /* gy 2 dsilu(u) */
                s1[k] += t; s2[k] = fmaf(t, r[k] - c4.z, s2[k]);
            }
        }
    }
#pragma unroll
    for (int k = 0; k < 16; k++) {
        float a = s1[k], b2 = s2[k];
        for (int o = 16; o >= HPB; o >>= 1) { a += __shfl_xor_sync(0xffffffff, a, o); b2 += __shfl_xor_sync(0xffffffff, b2, o); }
        if ((int)(threadIdx.x & 31) < HPB && k < kmax) { w1[threadIdx.x >> 5][h0 + k] = a; w2[threadIdx.x >> 5][h0 + k] = b2; }
    }
    __syncthreads();
    const int kb = min(bw, C - blk * bw);
    if ((int)threadIdx.x < kb) {
        float a = 0.f, b2 = 0.f;
#pragma unroll
        for (int w = 0; w < 8; w++) { a += w1[w][threadIdx.x]; b2 += w2[w][threadIdx.x]; }
        const size_t c = (size_t)n * C + blk * bw + threadIdx.x, o = ((size_t)blockIdx.x * N * C + c) * 2;
        part[o] = 0.5f * a; part[o + 1] = 0.5f * cf[threadIdx.x].w * b2;
    }
}
__global__ void gn_part_sum_k(const float *part, int nsl, int NC, double *ds) {   /* ds[2 c + j] = sum over slabs (double) */
    /* a warp per output: lanes stride over the slabs (up to 1024: one thread walking them serially in fp64 took 0.1 ms per call) */
    const int i = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5), lane = threadIdx.x & 31;
    if (i >= 2 * NC) return;
    double a = 0.0;
    for (int sl = lane; sl < nsl; sl += 32) a += (double)part[(size_t)sl * 2 * NC + i];
#pragma unroll
    for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    if (lane == 0) ds[i] = a;
}
template <int B, typename TG, typename TO>
__global__ void __launch_bounds__(256) gn_silu_bwd_apply_mx_k(const uint8_t *x, const TG *gy, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                                           const float *AB, TO *gx, int N, int C, int G, size_t S, unsigned gsr) {   /* gsr: SR seed of an MX-fp4 gx */
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
    if constexpr (IS_MX(TG)) mx_load_row_b<MX_BITS(TG)>((const uint8_t *)gy, mx_sc<MX_BITS(TG)>((const uint8_t *)gy, N, C, S), i, bw, gr);
    const int kmax = min(bw, C - blk * bw);
    TO *gxb = IS_MX(TO) ? gx : gx + ((size_t)n * C + blk * bw) * S + v;
    const TG *gyb = IS_MX(TG) ? gy : gy + ((size_t)n * C + blk * bw) * S + v;
#pragma unroll
    for (int k = 0; k < 32; k++) {
        out[k] = 0.f;
        if (k < kmax) {
            const float rs = cr[k], ga = cg[k];
            const float xhat = (r[k] - cm[k]) * rs, u = xhat * ga + cb[k];
            const float a = (IS_MX(TG) ? gr[k] : ldx(gyb, (size_t)k * S)) * dsilu(u);
            const float gv = rs * (a * ga - ca[k] - xhat * cc[k]);
            if constexpr (IS_MX(TO)) out[k] = gv; else stx(gxb, (size_t)k * S, gv);
        }
    }
    if constexpr (IS_MX(TO)) mx_store_row_b<MX_BITS(TO)>((uint8_t *)gx, mx_sc<MX_BITS(TO)>((uint8_t *)gx, N, C, S), i, bw, out, gsr);   /* fp4: exact SR keyed by (gsr, row, channel) */
}
/* the same for MX-fp4 gy and gx, issue-bound in the generic kernel (~65 SASS instructions per element: six shared-memory
   coefficient loads, per-element channel guards, a full hash per element). Here a lane pair per voxel row (HPB 2, 32-wide
   blocks; 16 channels per lane), the GN+SiLU backward folded into four per-channel coefficients in one float4,
       p = x Ah + Bh = u / 2,  th = tanh p,  2 dsilu(u) = (1 + th) (1 + p - p th),
       gx = gy (2 dsilu) Ah + x K1 + K0      (Ah = rstd gamma / 2, Bh = (beta - mean rstd gamma) / 2,
                                              K1 = -rstd^2 cc, K0 = rstd (mean rstd cc - ca); ca, cc: the group sums / len),
   and the exact SR of sr_e2m1_word with the floor taken by a round-toward-zero magic add (no F2I). FULL: the row has all 16
   channels of the lane (no guards). Same expected values as gn_silu_bwd_apply_mx_k (different rounding order and SR keys). */
__device__ __forceinline__ unsigned sr_e2m1_nib_fast(float vm, unsigned u16) {   /* |vm| <= 6 (block-scaled): sr_e2m1_nib without F2I / clamp */
    const float a = fabsf(vm);
    const float tt = fminf(fminf(fmaf(a, 2.f, -1.f), a + 1.f), fmaf(a, 0.5f, 3.f));   /* t(a) - 1 */
    const float u1 = __uint_as_float(0x3f800000u | (u16 << 7));                       /* 1 + u16 / 65536 */
    const unsigned fl = __float_as_uint(__fadd_rz(tt + u1, 8388608.f));               /* 2^23 + floor(t + u): exact, t + u < 8 */
    return (fl & 7u) | (__float_as_uint(vm) >> 28 & 8u);
}
template <int B, int HPB, bool FULL>
__global__ void __launch_bounds__(256) gn_silu_bwd_apply_mx2_k(const uint8_t *x, const uint8_t *gy, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                                            const float *AB, uint8_t *gx, int N, int C, int G, size_t S, unsigned gsr) {
    const int bw = mx_bw(C), nb = mx_nb(C), cpg = C / G, blk = blockIdx.y, n = blockIdx.z;
    __shared__ float4 cf[32];
    if (threadIdx.x < 32) {
        const int k = threadIdx.x, c = blk * bw + k;
        const bool ok = k < bw && c < C;
        const int ng = n * G + (ok ? c / cpg : 0);
        const float len = (float)cpg * (float)S;
        const float rs = ok ? rstd[ng] : 0.f, mu = ok ? mean[ng] : 0.f, ga = ok ? gamma[c] : 0.f, be = ok ? beta[c] : 0.f;
        const float ca = ok ? AB[2 * ng] / len : 0.f, cc = ok ? AB[2 * ng + 1] / len : 0.f, A = rs * ga;
        cf[k] = make_float4(0.5f * A, 0.5f * (be - mu * A), -rs * rs * cc, rs * (mu * rs * cc - ca));
    }
    __syncthreads();
    const size_t v = ((size_t)blockIdx.x * 256 + threadIdx.x) / HPB;
    if (v >= S) return;
    const int h = HPB == 2 ? (int)(threadIdx.x & 1) : 0, h0 = 16 * h, kmax = FULL ? 16 : max(0, min(min(bw, C - blk * bw) - h0, 16));
    const size_t ri = ((size_t)n * nb + blk) * S + v;
    const int rbx = mx_rb(bw, B), rbg = mx_rb(bw, 4);
    uint4 xw = make_uint4(0u, 0u, 0u, 0u);
    if constexpr (B == 4) { const uint2 t = bw == 8 ? make_uint2(__ldg((const unsigned *)(x + ri * rbx)), 0u) : __ldg((const uint2 *)(x + ri * rbx + 8 * h)); xw.x = t.x; xw.y = t.y; }
    else xw = __ldg((const uint4 *)(x + ri * rbx + 16 * h));
    const uint2 gw = bw == 8 ? make_uint2(__ldg((const unsigned *)(gy + ri * rbg)), 0u) : __ldg((const uint2 *)(gy + ri * rbg + 8 * h));
    const float fx = mx_scale(__ldg(mx_sc<B>(x, N, C, S) + ri)), fg = mx_scale(__ldg(mx_sc<4>(gy, N, C, S) + ri));
    float r[16], g[16], out[16];
    if constexpr (B == 4) {
        const unsigned w4[2] = {xw.x, xw.y};
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int q = 0; q < 4; q++) { const float2 a = dec_e2m1x2(w4[j] >> (8 * q)); r[8 * j + 2 * q] = a.x * fx; r[8 * j + 2 * q + 1] = a.y * fx; }
    } else {
        const unsigned w8[4] = {xw.x, xw.y, xw.z, xw.w};
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const float2 a = dec_e4m3x2((unsigned short)(w8[j] & 0xffff)), b2 = dec_e4m3x2((unsigned short)(w8[j] >> 16));
            r[4 * j] = a.x * fx; r[4 * j + 1] = a.y * fx; r[4 * j + 2] = b2.x * fx; r[4 * j + 3] = b2.y * fx;
        }
    }
    {
        const unsigned w4[2] = {gw.x, gw.y};
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int q = 0; q < 4; q++) { const float2 a = dec_e2m1x2(w4[j] >> (8 * q)); g[8 * j + 2 * q] = a.x; g[8 * j + 2 * q + 1] = a.y; }   /* raw; fg below */
    }
    unsigned am = 0u;
#pragma unroll
    for (int k = 0; k < 16; k++) {
        out[k] = 0.f;
        if (FULL || k < kmax) {
            const float4 c4 = cf[h0 + k];
            const float p = fmaf(r[k], c4.x, c4.y);
            float th; asm("tanh.approx.f32 %0, %1;" : "=f"(th) : "f"(p));
            const float d2 = (1.f + th) * (fmaf(-p, th, p) + 1.f);   /* 2 dsilu(2p) */
            out[k] = fmaf(g[k] * d2 * c4.x, fg, fmaf(r[k], c4.z, c4.w));
            am = amax_u(am, out[k]);
        }
    }
    if (HPB == 2) am = max(am, __shfl_xor_sync(0xffffffffu, am, 1));
    const int e = mx_exp(__uint_as_float(am), mxf<4>::inv_qmax);
    const float m = exp2i(-e);
    unsigned w[2];
#pragma unroll
    for (int j = 0; j < 2; j++) {
        if (gsr) {
            uint32_t hh[4]; sr_hash4(gsr, ri * 4 + 2 * h + j, hh);
            unsigned wj = 0u;
#pragma unroll
            for (int i = 0; i < 8; i++) wj |= sr_e2m1_nib_fast(out[8 * j + i] * m, (hh[i >> 1] >> (16 * (i & 1))) & 0xffffu) << (4 * i);
            w[j] = wj;
        } else w[j] = mxf<4>::enc8x(out + 8 * j, m);
    }
    uint8_t *p = gx + ri * rbg + 8 * h;
    if (bw == 8) *(unsigned *)p = w[0]; else *(uint2 *)p = make_uint2(w[0], w[1]);
    if (!h) mx_sc<4>(gx, N, C, S)[ri] = (uint8_t)(e + 127);
}
extern "C" void lp_gn_silu_bwd_mx(const void *x, int xdt, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                  const void *gy, void *gx, int gdt, double *ds, float *st, float *AB) {
    /* ds: 2 N C doubles (zeroed here), st: 2 N C floats, AB: 2 N G floats; the caller finishes with the group sums / param grads */
    size_t S = shape_spatial(s);
    const int NC = s.n * s.c, nb = mx_nb(s.c), bw = mx_bw(s.c);
    /* slabs: >= 8 voxel steps per thread (256 threads x V 2) to amortise the end-of-block reduction, but >= ~160 blocks in all
       (small levels were latency-bound with a handful of blocks), and >= one step per thread */
    const size_t ny = (size_t)nb * s.n;
    size_t slabs = S / 4096, want = (160 + ny - 1) / ny, maxs = S / 512 > 0 ? S / 512 : 1;
    if (slabs < want) slabs = want;
    if (slabs > maxs) slabs = maxs;
    if (slabs > 1024) slabs = 1024;
    const dim3 grid((unsigned)slabs, (unsigned)nb, (unsigned)s.n);
    float *part = lp_buf<float>(5, slabs * 2 * NC);
    if (s.c > 1024) { fprintf(stderr, "lp_gn_silu_bwd_mx: C %d > 1024\n", s.c); abort(); }
    cudaMemsetAsync(ds, 0, (size_t)2 * NC * sizeof(double));
    const uint8_t *xq = (const uint8_t *)x;
    static int s2on = -1; if (s2on < 0) s2on = getenv("UFSM_GN_APPLY2") ? atoi(getenv("UFSM_GN_APPLY2")) : 1;
    if (s2on && gdt == 4) {   /* MX-fp4 gradients: lane-pair kernel */
        const int full = bw >= 16 && s.c % bw == 0;
        const uint8_t *gyq = (const uint8_t *)gy;
#define GBS3(B) do { if (bw == 32 && full) gn_silu_bwd_stats_mx2_k<B, 2, true><<<grid, 256>>>(xq, gyq, gamma, beta, mean, rstd, s.n, s.c, G, S, part); \
                     else if (bw == 32) gn_silu_bwd_stats_mx2_k<B, 2, false><<<grid, 256>>>(xq, gyq, gamma, beta, mean, rstd, s.n, s.c, G, S, part); \
                     else if (full) gn_silu_bwd_stats_mx2_k<B, 1, true><<<grid, 256>>>(xq, gyq, gamma, beta, mean, rstd, s.n, s.c, G, S, part); \
                     else gn_silu_bwd_stats_mx2_k<B, 1, false><<<grid, 256>>>(xq, gyq, gamma, beta, mean, rstd, s.n, s.c, G, S, part); } while (0)
        if (xdt == 4) GBS3(4); else GBS3(8);
#undef GBS3
        gn_part_sum_k<<<nblk_((size_t)2 * NC * 32, 256), 256>>>(part, (int)slabs, NC, ds);
        LPCK();
        return;
    }
#define GBS2(B, V, H) do { if (gdt == 3) gn_silu_bwd_stats_mx_k<B, mx8_t, 1, H><<<grid, 256>>>(xq, (const mx8_t *)gy, gamma, beta, mean, rstd, s.n, s.c, G, S, part); \
                    else if (gdt == 4) gn_silu_bwd_stats_mx_k<B, mx4_t, 1, H><<<grid, 256>>>(xq, (const mx4_t *)gy, gamma, beta, mean, rstd, s.n, s.c, G, S, part); \
                    else if (gdt == 2) gn_silu_bwd_stats_mx_k<B, __half, V, H><<<grid, 256>>>(xq, (const __half *)gy, gamma, beta, mean, rstd, s.n, s.c, G, S, part); \
                    else if (gdt == 1) gn_silu_bwd_stats_mx_k<B, bf16, V, H><<<grid, 256>>>(xq, (const bf16 *)gy, gamma, beta, mean, rstd, s.n, s.c, G, S, part); \
                    else gn_silu_bwd_stats_mx_k<B, float, V, H><<<grid, 256>>>(xq, (const float *)gy, gamma, beta, mean, rstd, s.n, s.c, G, S, part); } while (0)
#define GBS1(B, H) do { if (S % 4 == 0) GBS2(B, 2, H); else GBS2(B, 1, H); } while (0)
#define GBS(B) do { if (bw == 32) GBS1(B, 2); else GBS1(B, 1); } while (0)
    if (xdt == 4) GBS(4); else GBS(8);
#undef GBS
#undef GBS1
#undef GBS2
    gn_part_sum_k<<<nblk_((size_t)2 * NC * 32, 256), 256>>>(part, (int)slabs, NC, ds);
    (void)st; (void)AB;
    LPCK();
}
extern "C" void lp_gn_silu_bwd_apply_mx(const void *x, int xdt, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                        const void *gy, void *gx, int gdt, const float *AB, unsigned gsr) {
    size_t S = shape_spatial(s);
    const dim3 ga(nblk_(S, 256), (unsigned)mx_nb(s.c), (unsigned)s.n);
    const uint8_t *xq = (const uint8_t *)x;
    static int a2 = -1; if (a2 < 0) a2 = getenv("UFSM_GN_APPLY2") ? atoi(getenv("UFSM_GN_APPLY2")) : 1;
    if (a2 && gdt == 4) {   /* MX-fp4 gradients (the trainer default): lane-pair kernel */
        const int bw = mx_bw(s.c), hpb = bw == 32 ? 2 : 1, full = bw >= 16 && s.c % bw == 0;
        const dim3 g2((unsigned)(((size_t)S * hpb + 255) / 256), (unsigned)mx_nb(s.c), (unsigned)s.n);
        uint8_t *gq = (uint8_t *)gx; const uint8_t *gyq = (const uint8_t *)gy;
#define GBA2(B) do { if (hpb == 2 && full) gn_silu_bwd_apply_mx2_k<B, 2, true><<<g2, 256>>>(xq, gyq, gamma, beta, mean, rstd, AB, gq, s.n, s.c, G, S, gsr); \
                     else if (hpb == 2) gn_silu_bwd_apply_mx2_k<B, 2, false><<<g2, 256>>>(xq, gyq, gamma, beta, mean, rstd, AB, gq, s.n, s.c, G, S, gsr); \
                     else if (full) gn_silu_bwd_apply_mx2_k<B, 1, true><<<g2, 256>>>(xq, gyq, gamma, beta, mean, rstd, AB, gq, s.n, s.c, G, S, gsr); \
                     else gn_silu_bwd_apply_mx2_k<B, 1, false><<<g2, 256>>>(xq, gyq, gamma, beta, mean, rstd, AB, gq, s.n, s.c, G, S, gsr); } while (0)
        if (xdt == 4) GBA2(4); else GBA2(8);
#undef GBA2
        LPCK();
        return;
    }
#define GBA(B) do { if (gdt == 3) gn_silu_bwd_apply_mx_k<B, mx8_t, mx8_t><<<ga, 256>>>(xq, (const mx8_t *)gy, gamma, beta, mean, rstd, AB, (mx8_t *)gx, s.n, s.c, G, S, 0u); \
                    else if (gdt == 4) gn_silu_bwd_apply_mx_k<B, mx4_t, mx4_t><<<ga, 256>>>(xq, (const mx4_t *)gy, gamma, beta, mean, rstd, AB, (mx4_t *)gx, s.n, s.c, G, S, gsr); \
                    else if (gdt == 2) gn_silu_bwd_apply_mx_k<B, __half, __half><<<ga, 256>>>(xq, (const __half *)gy, gamma, beta, mean, rstd, AB, (__half *)gx, s.n, s.c, G, S, 0u); \
                    else if (gdt == 1) gn_silu_bwd_apply_mx_k<B, bf16, bf16><<<ga, 256>>>(xq, (const bf16 *)gy, gamma, beta, mean, rstd, AB, (bf16 *)gx, s.n, s.c, G, S, 0u); \
                    else gn_silu_bwd_apply_mx_k<B, float, float><<<ga, 256>>>(xq, (const float *)gy, gamma, beta, mean, rstd, AB, (float *)gx, s.n, s.c, G, S, 0u); } while (0)
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
/* the same with the input count a template parameter: grid (voxels, n x MX block), so no 64-bit index divisions, the block's
   weights staged in smem, the CI inputs in registers; same arithmetic order as conv1_to_mx_k */
template <typename TG, int CI, int YB = 8>   /* YB: MX output bits (4: an MX-fp4 gradient, exact SR keyed by sr); Ci <= CI inputs at run time */
__global__ void __launch_bounds__(256) conv1_to_mx_t_k(const TG *x, const float *w, uint8_t *y, int N, int Co, size_t S, unsigned sr, int Ci) {
    __shared__ float sw[32 * CI];
    const int bw = mx_bw(Co), nb = mx_nb(Co), n = blockIdx.y / nb, blk = blockIdx.y % nb;
    for (int t = threadIdx.x; t < 32 * CI; t += blockDim.x) { const int co = blk * bw + t / CI, ci = t % CI; sw[t] = t / CI < bw && co < Co && ci < Ci ? w[co * Ci + ci] : 0.f; }
    __syncthreads();
    const size_t v = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (v >= S) return;
    float xi[CI], r[32];
#pragma unroll
    for (int ci = 0; ci < CI; ci++) xi[ci] = ci < Ci ? ldx(x, ((size_t)n * Ci + ci) * S + v) : 0.f;
#pragma unroll
    for (int k = 0; k < 32; k++) {
        float a = 0.f;
#pragma unroll
        for (int ci = 0; ci < CI; ci++) a += sw[k * CI + ci] * xi[ci];
        r[k] = a;
    }
    mx_store_row_b<YB>(y, mx_sc<YB>(y, N, Co, S), ((size_t)n * nb + blk) * S + v, bw, r, sr);
}
template <typename TG, int YB> static void conv1_to_mx_t(const TG *x, int N, int Ci, size_t S, const float *w, int Co, void *y, unsigned sr) {
    const dim3 grid(nblk_(S, 256), N * mx_nb(Co));
    uint8_t *yq = (uint8_t *)y;
    switch (Ci) {
        case 1: conv1_to_mx_t_k<TG, 1, YB><<<grid, 256>>>(x, w, yq, N, Co, S, sr, Ci); break;
        case 2: conv1_to_mx_t_k<TG, 2, YB><<<grid, 256>>>(x, w, yq, N, Co, S, sr, Ci); break;
        case 3: conv1_to_mx_t_k<TG, 3, YB><<<grid, 256>>>(x, w, yq, N, Co, S, sr, Ci); break;
        case 4: conv1_to_mx_t_k<TG, 4, YB><<<grid, 256>>>(x, w, yq, N, Co, S, sr, Ci); break;
        case 5: conv1_to_mx_t_k<TG, 5, YB><<<grid, 256>>>(x, w, yq, N, Co, S, sr, Ci); break;
        case 6: conv1_to_mx_t_k<TG, 6, YB><<<grid, 256>>>(x, w, yq, N, Co, S, sr, Ci); break;
        case 7: conv1_to_mx_t_k<TG, 7, YB><<<grid, 256>>>(x, w, yq, N, Co, S, sr, Ci); break;
        case 8: conv1_to_mx_t_k<TG, 8, YB><<<grid, 256>>>(x, w, yq, N, Co, S, sr, Ci); break;
        default: conv1_to_mx_t_k<TG, 16, YB><<<grid, 256>>>(x, w, yq, N, Co, S, sr, Ci);   /* 9..16 logits (phase heads) */
    }
}
extern "C" void lp_conv1_to_mx(const void *x, int gdt, int N, int Ci, size_t S, const float *w, int Co, void *y, int ydt, unsigned sr) {
    if (Ci > 16) { fprintf(stderr, "lp_conv1_to_mx: Ci %d > 16\n", Ci); abort(); }
    static int old = -1; if (old < 0) old = getenv("UFSM_CONV1_OLD") ? atoi(getenv("UFSM_CONV1_OLD")) : 0;
    if (!old || ydt == 4 || Ci > 8) {
#define C1M(YB) do { if (gdt == 2) conv1_to_mx_t<__half, YB>((const __half *)x, N, Ci, S, w, Co, y, sr); \
                     else if (gdt == 1) conv1_to_mx_t<bf16, YB>((const bf16 *)x, N, Ci, S, w, Co, y, sr); \
                     else conv1_to_mx_t<float, YB>((const float *)x, N, Ci, S, w, Co, y, sr); } while (0)
        if (ydt == 4) C1M(4); else C1M(8);
#undef C1M
        LPCK();
        return;
    }
    size_t n = (size_t)N * mx_nb(Co) * S;
    if (gdt == 2) conv1_to_mx_k<__half><<<nblk_(n, 256), 256>>>((const __half *)x, w, (uint8_t *)y, N, Ci, Co, S);
    else if (gdt == 1) conv1_to_mx_k<bf16><<<nblk_(n, 256), 256>>>((const bf16 *)x, w, (uint8_t *)y, N, Ci, Co, S);
    else conv1_to_mx_k<float><<<nblk_(n, 256), 256>>>((const float *)x, w, (uint8_t *)y, N, Ci, Co, S);
    LPCK();
}
