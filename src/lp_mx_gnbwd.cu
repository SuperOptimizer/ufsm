/* GroupNorm+SiLU backward on MX tensors and 1x1 conv to MX */
#include "lp_mxops.cuh"
/* d silu(u) / du = sg (1 + u (1 - sg)), sg = sigmoid(u) = 1/2 + tanh(u / 2) / 2: one MUFU op (exp + reciprocal took two;
   tanh.approx ~2^-11 relative, far below the e4m3 rounding of the stored gradient) */
__device__ __forceinline__ float dsilu(float u) {
    float th; asm("tanh.approx.f32 %0, %1;" : "=f"(th) : "f"(0.5f * u));
    const float sg = fmaf(0.5f, th, 0.5f);
    return sg * (1.f + u * (1.f - sg));
}
/* the head's input gradient of one channel from the CI logit-gradient values of its voxel as conv1_to_mx_t_k computes it (its
   a += w x compiles to this fused multiply-add chain from +0; test_grad_mx4 checks the recomputing GN backward against it bit for bit) */
template <int CI> __device__ __forceinline__ float head_dot(const float *w, const float *x) {
    float a = 0.f;
#pragma unroll
    for (int ci = 0; ci < CI; ci++) a = fmaf(w[ci], x[ci], a);
    return a;
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

/* the z-window (nn_set_zwin) as a range of (pseudo-)samples [n0, n0 + cnt): slab-local GroupNorm's backward runs on the [slabs] shape
   of one real sample (slab = sample), so a window of whole slabs is a sample range; the whole tensor without a window */
static void gn_nwin(shape5 s, int *n0, int *cnt) {
    int a, b; *n0 = 0; *cnt = s.n;
    if (!zw_win(s.n * s.d, &a, &b)) return;
    if (a % s.d || (b % s.d && b != s.n * s.d)) { fprintf(stderr, "gn backward: z-window [%d, %d) is not whole slabs of %d planes\n", a, b, s.d); abort(); }
    *n0 = a / s.d; *cnt = (b + s.d - 1) / s.d - *n0;
}
template <int B, int HPB, bool FULL, int NK = 16>   /* NK: the channels of a FULL lane (16, or 8: a whole 8-channel tensor) */
__global__ void __launch_bounds__(256) gn_silu_bwd_stats_mx2_k(const uint8_t *x, const uint8_t *gy, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                                            int N, int C, int G, size_t S, float *part, int n0 = 0) {   /* n0: first (pseudo-)sample (z-window) */
    const int bw = mx_bw(C), nb = mx_nb(C), cpg = C / G;
    const int blk = blockIdx.y, n = n0 + (int)blockIdx.z, nsl = gridDim.x;
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
    const int kmax = FULL ? NK : max(0, min(min(bw, C - blk * bw) - h0, 16));
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
        else if (bw == 8) { const uint2 t = __ldg((const uint2 *)(x + ri * rbx)); xw.x = t.x; xw.y = t.y; }   /* MX-fp8 8-channel row: 8 bytes */
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
            for (int j = 0; j < (FULL ? NK / 8 : 2); j++)
#pragma unroll
                for (int i = 0; i < 4; i++) { const float2 a = dec_e2m1x2(w4[j] >> (8 * i)); r[8 * j + 2 * i] = a.x * fx; r[8 * j + 2 * i + 1] = a.y * fx; }
        } else {
            const unsigned w8[4] = {cx.x, cx.y, cx.z, cx.w};
#pragma unroll
            for (int j = 0; j < (FULL ? NK / 4 : 4); j++) {
                const float2 a = dec_e4m3x2((unsigned short)(w8[j] & 0xffff)), b2 = dec_e4m3x2((unsigned short)(w8[j] >> 16));
                r[4 * j] = a.x * fx; r[4 * j + 1] = a.y * fx; r[4 * j + 2] = b2.x * fx; r[4 * j + 3] = b2.y * fx;
            }
        }
        {
            const unsigned w4[2] = {cg.x, cg.y};
#pragma unroll
            for (int j = 0; j < (FULL ? NK / 8 : 2); j++)
#pragma unroll
                for (int i = 0; i < 4; i++) { const float2 a = dec_e2m1x2(w4[j] >> (8 * i)); g[8 * j + 2 * i] = a.x; g[8 * j + 2 * i + 1] = a.y; }
        }
#pragma unroll
        for (int k = 0; k < 16; k++) {
            if (FULL ? k < NK : k < kmax) {
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
template <int B, int HPB, bool FULL, int NK = 16>   /* NK: the channels of a FULL lane (16, or 8: a whole 8-channel tensor) */
__global__ void __launch_bounds__(256, 5) gn_silu_bwd_apply_mx2_k(const uint8_t *x, const uint8_t *gy, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                                            const float *AB, uint8_t *gx, int N, int C, int G, size_t S, unsigned gsr, int n0 = 0) {
    const int bw = mx_bw(C), nb = mx_nb(C), cpg = C / G, blk = blockIdx.y, n = n0 + (int)blockIdx.z;
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
    /* persistent grid-stride rows (lp_gn_silu_bwd_apply_mx sizes the grid to the resident blocks) with the next row's loads in flight
       during this one's arithmetic (a row per thread left DRAM ~80% busy with ~85% of the issue slots taken); warp-uniform trips for
       the lane-pair shuffle, rows past S computed but not stored. Each row's arithmetic and SR keys as before: the same bits */
    const int h = HPB == 2 ? (int)(threadIdx.x & 1) : 0, h0 = 16 * h, kmax = FULL ? NK : max(0, min(min(bw, C - blk * bw) - h0, 16));
    const size_t rb0 = ((size_t)n * nb + blk) * S, nv = (size_t)gridDim.x * (256 / HPB), lo = (size_t)((threadIdx.x & 31) / HPB);
    const int rbx = mx_rb(bw, B), rbg = mx_rb(bw, 4);
    const uint8_t *sx = mx_sc<B>(x, N, C, S), *sg = mx_sc<4>(gy, N, C, S);
    uint8_t *sgx = mx_sc<4>(gx, N, C, S);
    uint4 xw = make_uint4(0u, 0u, 0u, 0u); uint2 gw = make_uint2(0u, 0u); unsigned xs = 0u, gs = 0u;
    auto load = [&](size_t ri) {
        if constexpr (B == 4) { const uint2 t = bw == 8 ? make_uint2(__ldg((const unsigned *)(x + ri * rbx)), 0u) : __ldg((const uint2 *)(x + ri * rbx + 8 * h)); xw.x = t.x; xw.y = t.y; }
        else if (bw == 8) { const uint2 t = __ldg((const uint2 *)(x + ri * rbx)); xw.x = t.x; xw.y = t.y; }   /* MX-fp8 8-channel row: 8 bytes */
        else xw = __ldg((const uint4 *)(x + ri * rbx + 16 * h));
        gw = bw == 8 ? make_uint2(__ldg((const unsigned *)(gy + ri * rbg)), 0u) : __ldg((const uint2 *)(gy + ri * rbg + 8 * h));
        xs = __ldg(sx + ri); gs = __ldg(sg + ri);
    };
    size_t v = ((size_t)blockIdx.x * 256 + threadIdx.x) / HPB;
    if (v < S) load(rb0 + v);
    for (; v - lo < S; v += nv) {
        const size_t ri = rb0 + v;
        const uint4 cx = xw; const uint2 cg = gw;
        const float fx = mx_scale(xs), fg = mx_scale(gs);
        if (v + nv < S) load(ri + nv);
        float r[16], g[16], out[16];
        if constexpr (B == 4) {
            const unsigned w4[2] = {cx.x, cx.y};
#pragma unroll
            for (int j = 0; j < (FULL ? NK / 8 : 2); j++)
#pragma unroll
                for (int q = 0; q < 4; q++) { const float2 a = dec_e2m1x2(w4[j] >> (8 * q)); r[8 * j + 2 * q] = a.x * fx; r[8 * j + 2 * q + 1] = a.y * fx; }
        } else {
            const unsigned w8[4] = {cx.x, cx.y, cx.z, cx.w};
#pragma unroll
            for (int j = 0; j < (FULL ? NK / 4 : 4); j++) {
                const float2 a = dec_e4m3x2((unsigned short)(w8[j] & 0xffff)), b2 = dec_e4m3x2((unsigned short)(w8[j] >> 16));
                r[4 * j] = a.x * fx; r[4 * j + 1] = a.y * fx; r[4 * j + 2] = b2.x * fx; r[4 * j + 3] = b2.y * fx;
            }
        }
        {
            const unsigned w4[2] = {cg.x, cg.y};
#pragma unroll
            for (int j = 0; j < (FULL ? NK / 8 : 2); j++)
#pragma unroll
                for (int q = 0; q < 4; q++) { const float2 a = dec_e2m1x2(w4[j] >> (8 * q)); g[8 * j + 2 * q] = a.x; g[8 * j + 2 * q + 1] = a.y; }   /* raw; fg below */
        }
        unsigned am = 0u;
#pragma unroll
        for (int k = 0; k < 16; k++) {
            out[k] = 0.f;
            if (FULL ? k < NK : k < kmax) {
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
        unsigned w[2] = {0u, 0u};
#pragma unroll
        for (int j = 0; j < (FULL ? NK / 8 : 2); j++) {
            if (gsr) {
                uint32_t hh[4]; sr_hash4(gsr, ri * 4 + 2 * h + j, hh);
                unsigned wj = 0u;
#pragma unroll
                for (int i = 0; i < 8; i++) wj |= sr_e2m1_nib_fast(out[8 * j + i] * m, (hh[i >> 1] >> (16 * (i & 1))) & 0xffffu) << (4 * i);
                w[j] = wj;
            } else w[j] = mxf<4>::enc8x(out + 8 * j, m);
        }
        if (v < S) {
            uint8_t *p = gx + ri * rbg + 8 * h;
            if (bw == 8) *(unsigned *)p = w[0]; else *(uint2 *)p = make_uint2(w[0], w[1]);
            if (!h) sgx[ri] = (uint8_t)(e + 127);
        }
    }
}
/* dec[0]'s GroupNorm+SiLU backward with the head's input gradient recomputed instead of read (unet's three level-0 buffers, where it
   is never stored): gy of a voxel row is the 1^3 conv of its CI fp16 logit-gradient values with the head weights (wh [CI][C]),
   stored MX-fp4 with exact SR keyed by (hs, row) as the head's backward-data writes it (conv1_to_mx_t_k: head_dot, mx4_sr_row).
   The stats and apply passes (the lane-per-row gn_silu_bwd_*_mx2_k<4, 1, FULL> for C 8 / 16, one channel block) see the bits that
   backward-data would have stored, so gx and the GroupNorm sums equal the two-op path's. The apply writes gx in place over x
   (plain loads: no read-only cache on x).
   EX (the default; lp_set_gn_head_exact(0) / UFSM_GN_HEAD_EXACT=0 restores the two-op path's bits): gy is the fp32 head_dot itself,
   not put through the MX-fp4 SR that only its storage needed. The GN backward is linear in gy and that SR unbiased, so gx keeps its
   expectation with less variance (only its own SR, the same keys; the seed sequence unchanged). That SR was most of the two kernels'
   issue slots (P512 desk, 16 channels: stats 10.5 -> 5.2 ms, apply 15.4 -> 10.6 ms per step). */
template <int CI, bool EX> __device__ __forceinline__ void head_gy_row_x(const float *sw, const float *xi, int bw, size_t ri, unsigned hs, float *g, float &fg) {
    float r[16];   /* xi: the CI logit-gradient values of the voxel */
#pragma unroll
    for (int k = 0; k < 16; k++) r[k] = head_dot<CI>(sw + k * CI, xi);
    if constexpr (EX) {
#pragma unroll
        for (int k = 0; k < 16; k++) g[k] = r[k];
        fg = 1.f;
        return;
    }
    unsigned w[2];
    const int e = mx4_sr_row<16>(r, bw, ri, hs, w);
#pragma unroll
    for (int j = 0; j < 2; j++)
#pragma unroll
        for (int i = 0; i < 4; i++) { const float2 a = dec_e2m1x2(w[j] >> (8 * i)); g[8 * j + 2 * i] = a.x; g[8 * j + 2 * i + 1] = a.y; }   /* raw; fg below */
    fg = mx_scale((unsigned)(uint8_t)(e + 127));
}
/* ns > 1: n is a pseudo sample (sample n / ns, z-slab n % ns of S voxels: slab-local GroupNorm over the pseudo-sample shape); gl stays
   [sample][CI][ns S] */
template <int CI, bool EX> __device__ __forceinline__ void head_gy_row(const float *sw, const __half *gl, int n, size_t S, size_t v, int bw, size_t ri, unsigned hs,
                                                                       float *g, float &fg, int ns) {
    float xi[CI];
    const size_t gb = ns > 1 ? (size_t)(n / ns) * CI * S * ns + (size_t)(n % ns) * S : (size_t)n * CI * S, gs = ns > 1 ? S * ns : S;
#pragma unroll
    for (int ci = 0; ci < CI; ci++) xi[ci] = __half2float(gl[gb + (size_t)ci * gs + v]);
    head_gy_row_x<CI, EX>(sw, xi, bw, ri, hs, g, fg);
}
/* the head weights as conv1_to_mx_t_k stages them for channel block 0 (its w is the flipped [C][CI] copy: the same values) */
template <int CI> __device__ __forceinline__ void head_sw(float *sw, const float *wh, int C, int bw) {
    for (int t = threadIdx.x; t < 16 * CI; t += blockDim.x) { const int k = t / CI, ci = t % CI; sw[t] = k < bw && k < C ? wh[(size_t)ci * C + k] : 0.f; }
}
template <int CI, bool FULL, bool EX>
__global__ void __launch_bounds__(256) gn_silu_bwd_stats_mx2_hg_k(const uint8_t *x, const __half *gl, const float *wh, unsigned hs, const float *gamma, const float *beta,
                                                               const float *mean, const float *rstd, int N, int C, int G, size_t S, float *part, int ns, int n0 = 0) {
    const int bw = mx_bw(C), nb = mx_nb(C), cpg = C / G;
    const int blk = blockIdx.y, n = n0 + (int)blockIdx.z, nsl = gridDim.x;
    const int h0 = 0, tv = (int)threadIdx.x, nt = 256;
    const size_t q0 = S * blockIdx.x / nsl, q1 = S * (blockIdx.x + 1) / nsl;
    __shared__ float4 cf[32];
    __shared__ float w1[8][32], w2[8][32], sw[16 * CI];
    if (threadIdx.x < 32) {
        const int k = threadIdx.x, c = blk * bw + k;
        const bool ok = k < bw && c < C;
        const int ng = n * G + (ok ? c / cpg : 0);
        const float rs = ok ? rstd[ng] : 0.f, mu = ok ? mean[ng] : 0.f, A = ok ? rs * gamma[c] : 0.f, be = ok ? beta[c] : 0.f;
        cf[k] = make_float4(0.5f * A, 0.5f * (be - mu * A), mu, rs);
    }
    head_sw<CI>(sw, wh, C, bw);
    w1[threadIdx.x >> 5][threadIdx.x & 31] = 0.f; w2[threadIdx.x >> 5][threadIdx.x & 31] = 0.f;
    __syncthreads();
    const int kmax = FULL ? 16 : max(0, min(min(bw, C - blk * bw) - h0, 16));
    const size_t rbase = ((size_t)n * nb + blk) * S;
    const uint8_t *sx = mx_sc<4>(x, N, C, S);
    const int rbx = mx_rb(bw, 4);
    float s1[16], s2[16];
#pragma unroll
    for (int k = 0; k < 16; k++) { s1[k] = 0.f; s2[k] = 0.f; }
    uint2 xw = make_uint2(0u, 0u); unsigned xs = 0u;
    auto load = [&](size_t v) {
        const size_t ri = rbase + v;
        xw = bw == 8 ? make_uint2(__ldg((const unsigned *)(x + ri * rbx)), 0u) : __ldg((const uint2 *)(x + ri * rbx));
        xs = __ldg(sx + ri);
    };
    size_t q = q0 + tv;
    if (q < q1) load(q);
    for (; q < q1; q += nt) {
        const uint2 cx = xw; const float fx = mx_scale(xs);
        const size_t v = q;
        if (q + nt < q1) load(q + nt);   /* next row in flight during this one's arithmetic */
        float r[16], g[16], fg;
        {
            const unsigned w4[2] = {cx.x, cx.y};
#pragma unroll
            for (int j = 0; j < 2; j++)
#pragma unroll
                for (int i = 0; i < 4; i++) { const float2 a = dec_e2m1x2(w4[j] >> (8 * i)); r[8 * j + 2 * i] = a.x * fx; r[8 * j + 2 * i + 1] = a.y * fx; }
        }
        head_gy_row<CI, EX>(sw, gl, n, S, v, bw, rbase + v, hs, g, fg, ns);
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
        for (int o = 16; o >= 1; o >>= 1) { a += __shfl_xor_sync(0xffffffff, a, o); b2 += __shfl_xor_sync(0xffffffff, b2, o); }
        if ((int)(threadIdx.x & 31) < 1 && k < kmax) { w1[threadIdx.x >> 5][h0 + k] = a; w2[threadIdx.x >> 5][h0 + k] = b2; }
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
template <int CI, bool FULL, bool EX>
__global__ void __launch_bounds__(256) gn_silu_bwd_apply_mx2_hg_k(uint8_t *x, const __half *gl, const float *wh, unsigned hs, const float *gamma, const float *beta,
                                                               const float *mean, const float *rstd, const float *AB, int N, int C, int G, size_t S, unsigned gsr, int ns, int n0 = 0) {
    const int bw = mx_bw(C), nb = mx_nb(C), cpg = C / G, blk = blockIdx.y, n = n0 + (int)blockIdx.z;
    __shared__ float4 cf[32];
    __shared__ float sw[16 * CI];
    if (threadIdx.x < 32) {
        const int k = threadIdx.x, c = blk * bw + k;
        const bool ok = k < bw && c < C;
        const int ng = n * G + (ok ? c / cpg : 0);
        const float len = (float)cpg * (float)S;
        const float rs = ok ? rstd[ng] : 0.f, mu = ok ? mean[ng] : 0.f, ga = ok ? gamma[c] : 0.f, be = ok ? beta[c] : 0.f;
        const float ca = ok ? AB[2 * ng] / len : 0.f, cc = ok ? AB[2 * ng + 1] / len : 0.f, A = rs * ga;
        cf[k] = make_float4(0.5f * A, 0.5f * (be - mu * A), -rs * rs * cc, rs * (mu * rs * cc - ca));
    }
    head_sw<CI>(sw, wh, C, bw);
    __syncthreads();
    const size_t v = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (v >= S) return;
    const int h0 = 0, kmax = FULL ? 16 : max(0, min(min(bw, C - blk * bw) - h0, 16));
    const size_t ri = ((size_t)n * nb + blk) * S + v;
    const int rbx = mx_rb(bw, 4);
    const uint2 xw = bw == 8 ? make_uint2(*(const unsigned *)(x + ri * rbx), 0u) : *(const uint2 *)(x + ri * rbx);
    const float fx = mx_scale(mx_sc<4>(x, N, C, S)[ri]);
    float r[16], g[16], out[16], fg;
    {
        const unsigned w4[2] = {xw.x, xw.y};
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int q = 0; q < 4; q++) { const float2 a = dec_e2m1x2(w4[j] >> (8 * q)); r[8 * j + 2 * q] = a.x * fx; r[8 * j + 2 * q + 1] = a.y * fx; }
    }
    head_gy_row<CI, EX>(sw, gl, n, S, v, bw, ri, hs, g, fg, ns);
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
    const int e = mx_exp(__uint_as_float(am), mxf<4>::inv_qmax);
    const float m = exp2i(-e);
    unsigned w[2];
#pragma unroll
    for (int j = 0; j < 2; j++) {
        if (gsr) {
            uint32_t hh[4]; sr_hash4(gsr, ri * 4 + j, hh);
            unsigned wj = 0u;
#pragma unroll
            for (int i = 0; i < 8; i++) wj |= sr_e2m1_nib_fast(out[8 * j + i] * m, (hh[i >> 1] >> (16 * (i & 1))) & 0xffffu) << (4 * i);
            w[j] = wj;
        } else w[j] = mxf<4>::enc8x(out + 8 * j, m);
    }
    uint8_t *p = x + ri * rbx;
    if (bw == 8) *(unsigned *)p = w[0]; else *(uint2 *)p = make_uint2(w[0], w[1]);
    mx_sc<4>(x, N, C, S)[ri] = (uint8_t)(e + 127);
}
/* GN+SiLU backward apply for MX-fp4 gy / gx (32-channel blocks) that also writes the fp4 weight gradient's gy records of the whole
   layer (the gy_pre4_k pass over the stored gx folded in: it re-read gx, ~10 ms per level-0 layer). The block takes gy_pre4_k's tile
   (z-step x 8 rows x 16 x: warp = block pb = 4 (z & 1) + (yp & 3), lane = 16 (y & 1) + x): each lane computes its voxel's 32 gx
   values (the folded coefficients of gn_silu_bwd_apply_mx2_k), stores the SR-rounded row and puts the unrounded values into the
   transpose buffer (one rounding: gy_pre4_k re-rounded the stored row), and the record words and scales are formed as gy_pre4_k does
   (SR keyed by the element with rsr). gbsum += the per-channel sums of gy (the bias gradient gy_pre4_k accumulated). */
#define GREC 136
template <int B, int BW = 32>   /* BW: the MX block width of gy / gx (32, or a whole 8 / 16-channel level: opt-in UFSM_GYREC=2; P512, desk:
                                   8 channels -2% step for 570 MB, 16 channels ~0 for 1.14 GB: this kernel then costs what the pre-pass did) */
__global__ void __launch_bounds__(256, 3) gn_silu_bwd_apply_rec_k(const uint8_t *__restrict__ x, const uint8_t *__restrict__ gy, const float *gamma, const float *beta,
                                                            const float *mean, const float *rstd, const float *AB, uint8_t *__restrict__ gx, uint8_t *__restrict__ out,
                                                            float *__restrict__ gbsum, int N, int C, int G, int D, int H, int W, unsigned gsr, unsigned rsr) {
    constexpr int NTASK = BW / 8, GRB = BW / 2;   /* record tasks per lane; MX-fp4 gy / gx row bytes */
    __shared__ __align__(16) __nv_bfloat16 v[8][BW][40];   /* bf16 (its 2^-9 rounding far below the fp4 SR that follows): 3 blocks / SM */
    __shared__ float4 cf[BW];
    __shared__ float bred[BW];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nb = C / BW, cb = blockIdx.y % nb, n = blockIdx.y / nb, cpg = C / G;
    const size_t S = (size_t)D * H * W;
    const int Wb = (W + 15) / 16, ZT = (D + 1) / 2, TY = (H + 7) / 8;
    const unsigned ngrp = (unsigned)ZT * TY * Wb;
    if (threadIdx.x < BW) {
        const int k = threadIdx.x, c = cb * BW + k, ng = n * G + c / cpg;
        const float len = (float)cpg * (float)S;
        const float rs = rstd[ng], mu = mean[ng], ga = gamma[c], be = beta[c];
        const float ca = AB[2 * ng] / len, cc = AB[2 * ng + 1] / len, A = rs * ga;
        cf[k] = make_float4(0.5f * A, 0.5f * (be - mu * A), -rs * rs * cc, rs * (mu * rs * cc - ca));
        bred[k] = 0.f;
    }
    __syncthreads();
    const uint8_t *sx = mx_sc<B>(x, N, C, S), *sg = mx_sc<4>(gy, N, C, S);
    uint8_t *sgx = mx_sc<4>(gx, N, C, S);
    const int rbx = mx_rb(BW, B);
    float bacc[NTASK];
#pragma unroll
    for (int k = 0; k < NTASK; k++) bacc[k] = 0.f;
    for (unsigned grp = blockIdx.x; grp < ngrp; grp += gridDim.x) {
        const int xb = (int)(grp % Wb), ty = (int)((grp / Wb) % TY), zt = (int)(grp / ((unsigned)Wb * TY));
        const int z = 2 * zt + (warp >> 2), yp = 4 * ty + (warp & 3), y = 2 * yp + (lane >> 4), xx = 16 * xb + (lane & 15);
        float q[BW];
#pragma unroll
        for (int k = 0; k < BW; k++) q[k] = 0.f;
        if (z < D && y < H && xx < W) {
            const size_t ri = ((size_t)n * nb + cb) * S + ((size_t)z * H + y) * W + xx;
            float r[32], g[32];
            mxf<B>::dec_row(x + ri * rbx, BW, mx_scale(sx[ri]), r);
            mxf<4>::dec_row(gy + ri * GRB, BW, 1.f, g);   /* raw; the scale below */
            const float fg = mx_scale(sg[ri]);
            float o[BW]; unsigned am = 0u;
#pragma unroll
            for (int k = 0; k < BW; k++) {
                const float4 c4 = cf[k];
                const float p = fmaf(r[k], c4.x, c4.y);
                float th; asm("tanh.approx.f32 %0, %1;" : "=f"(th) : "f"(p));
                const float d2 = (1.f + th) * (fmaf(-p, th, p) + 1.f);
                o[k] = fmaf(g[k] * d2 * c4.x, fg, fmaf(r[k], c4.z, c4.w));
                am = amax_u(am, o[k]);
            }
            const int e = mx_exp(__uint_as_float(am), mxf<4>::inv_qmax);
            const float m = exp2i(-e), sc = exp2i(e);
            unsigned w4[4] = {0u, 0u, 0u, 0u};
#pragma unroll
            for (int j = 0; j < NTASK; j++) {
                if (gsr) {
                    uint32_t hh[4]; sr_hash4(gsr, ri * 4 + j, hh);
                    unsigned wj = 0u;
#pragma unroll
                    for (int i = 0; i < 8; i++) wj |= sr_e2m1_nib_fast(o[8 * j + i] * m, (hh[i >> 1] >> (16 * (i & 1))) & 0xffffu) << (4 * i);
                    w4[j] = wj;
                } else w4[j] = mxf<4>::enc8x(o + 8 * j, m);
            }
            if constexpr (BW == 32) *(uint4 *)(gx + ri * 16) = make_uint4(w4[0], w4[1], w4[2], w4[3]);
            else if constexpr (BW == 16) *(uint2 *)(gx + ri * 8) = make_uint2(w4[0], w4[1]);
            else *(unsigned *)(gx + ri * 4) = w4[0];
            sgx[ri] = (uint8_t)(e + 127);
#pragma unroll
            for (int k = 0; k < BW; k++) q[k] = o[k];   /* the records round the unrounded gy once (gy_pre4_k re-rounded the stored values) */
            (void)sc;
        }
#pragma unroll
        for (int k = 0; k < BW; k++) v[warp][k][lane] = __float2bfloat16_rn(q[k]);
        __syncwarp();
#pragma unroll
        for (int k = 0; k < NTASK; k++) {   /* gy_pre4_k's record words (BW / 8 tasks per lane) */
            const int task = k * 32 + lane, c = task >> 2, f = task & 3, co = cb * BW + c;
            float qq[8];
            {
                const uint4 w8 = *(const uint4 *)&v[warp][c][8 * f];
                const unsigned ww[4] = {w8.x, w8.y, w8.z, w8.w};
#pragma unroll
                for (int j = 0; j < 4; j++) { qq[2 * j] = __uint_as_float(ww[j] << 16); qq[2 * j + 1] = __uint_as_float(ww[j] & 0xffff0000u); }
            }
            bacc[k] += ((qq[0] + qq[1]) + (qq[2] + qq[3])) + ((qq[4] + qq[5]) + (qq[6] + qq[7]));
            unsigned amu = 0u;
#pragma unroll
            for (int j = 0; j < 8; j++) amu = amax_u(amu, qq[j]);
            amu = max(amu, __shfl_xor_sync(0xffffffffu, amu, 1)); amu = max(amu, __shfl_xor_sync(0xffffffffu, amu, 2));
            const int e = mx_exp(__uint_as_float(amu), 1.f / 6.f);
            const float m = exp2i(-e);
            const uint64_t vid = ((((uint64_t)n * C + co) * D + z) * H + (2 * yp + (f >> 1))) * (uint64_t)W + 16 * xb + 8 * (f & 1);
            uint32_t hh[4];
            sr_hash4(rsr, vid, hh);
            /* straight to the record (the warp's 16 bytes of channel c, its scale byte): no block gather, no barriers per group
               (L2 merges the partial sectors of the 8 warps' slices) */
            uint8_t *rec = out + ((((size_t)n * C + co) * ZT + zt) * TY + ty) * Wb * GREC + (size_t)xb * GREC;
            *(unsigned *)(rec + warp * 16 + 4 * f) = sr_e2m1_word(qq, m, hh);
            if (f == 0) rec[128 + warp] = (uint8_t)(e + 127);
        }
        __syncwarp();   /* v is rewritten by the next group */
    }
#pragma unroll
    for (int k = 0; k < NTASK; k++) {
        float a = bacc[k]; a += __shfl_xor_sync(0xffffffffu, a, 1); a += __shfl_xor_sync(0xffffffffu, a, 2);
        if ((lane & 3) == 0) atomicAdd(&bred[(k * 32 + lane) >> 2], a);
    }
    __syncthreads();
    if (threadIdx.x < BW) atomicAdd(&gbsum[(size_t)n * C + cb * BW + threadIdx.x], bred[threadIdx.x]);
}
thread_local lp_gyrec_t g_gyrec;
static int g_gyrec_narrow = -1;   /* records also for whole 8- / 16-channel levels (-1: env UFSM_GYREC=2) */
extern "C" void lp_set_gyrec_narrow(int on) { g_gyrec_narrow = on; }
static int g_gyrec_off = -1;   /* test hook: 1 no records at all (the plain apply), -1 env UFSM_GYREC */
extern "C" void lp_set_gyrec_off(int off) { g_gyrec_off = off; }
extern "C" int lp_gyrec_narrow(void) { return g_gyrec_narrow >= 0 ? g_gyrec_narrow : getenv("UFSM_GYREC") && atoi(getenv("UFSM_GYREC")) >= 2; }
extern "C" int lp_gn_silu_bwd_apply_rec(const void *x, int xdt, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                        const void *gy, void *gx, const float *AB, unsigned gsr, unsigned rsr) {
    static int on = -1; if (on < 0) on = getenv("UFSM_GYREC") ? atoi(getenv("UFSM_GYREC")) : 1;
    lp_gyrec_t *R = &g_gyrec;
    const int d = cur_dev_();
    R->valid = 0;
    const int bw = mx_bw(s.c);   /* 32, or a whole 8- / 16-channel level */
    if (!on || g_gyrec_off == 1 || s.c % bw || s.n > 1 || (xdt != 4 && xdt != 8)) return 0;
    if (bw < 32 && !lp_gyrec_narrow()) return 0;   /* narrow levels: opt-in (records ~8.5 B per level-0 voxel at 16 channels) */
    if (R->dev != d) { R->rec = nullptr; R->cap = 0; R->gb = nullptr; R->gbc = 0; R->dev = d; }   /* (a thread that changed GPU) */
    const int Wb = (s.w + 15) / 16, ZT = (s.d + 1) / 2, TY = (s.h + 7) / 8, nb = s.c / bw;
    const size_t nrec = (size_t)s.n * s.c * ZT * TY * Wb * GREC;
    if (R->cap < nrec) {   /* this thread's own buffer (the pre-pass's lp_buf slot could be rewritten before the records are read) */
        if (R->rec) cudaFree(R->rec);
        if (cudaMalloc((void **)&R->rec, nrec) != cudaSuccess) { R->rec = nullptr; R->cap = 0; cudaGetLastError(); return 0; }
        R->cap = nrec;
        if (getenv("UFSM_MEM_REPORT") && atoi(getenv("UFSM_MEM_REPORT")) > 0) fprintf(stderr, "gy records: %.1f MB (GPU %d)\n", nrec / 1e6, d);
    }
    if (R->gbc < s.n * s.c) { if (R->gb) cudaFree(R->gb); cudaMalloc((void **)&R->gb, (size_t)s.n * s.c * 4); R->gbc = s.n * s.c; }
    cudaMemsetAsync(R->gb, 0, (size_t)s.n * s.c * 4);
    const unsigned ngrp = (unsigned)ZT * TY * Wb;
    const dim3 grid(ngrp < 4096 ? ngrp : 4096, (unsigned)(nb * s.n));
#define GREC_K(BWT) do { if (xdt == 4) gn_silu_bwd_apply_rec_k<4, BWT><<<grid, 256>>>((const uint8_t *)x, (const uint8_t *)gy, gamma, beta, mean, rstd, AB, (uint8_t *)gx, R->rec, R->gb, s.n, s.c, G, s.d, s.h, s.w, gsr, rsr); \
                          else gn_silu_bwd_apply_rec_k<8, BWT><<<grid, 256>>>((const uint8_t *)x, (const uint8_t *)gy, gamma, beta, mean, rstd, AB, (uint8_t *)gx, R->rec, R->gb, s.n, s.c, G, s.d, s.h, s.w, gsr, rsr); } while (0)
    if (bw == 32) GREC_K(32); else if (bw == 16) GREC_K(16); else GREC_K(8);
#undef GREC_K
    LPCK();
    R->gy = gx; R->n = s.n; R->c = s.c; R->d = s.d; R->h = s.h; R->w = s.w; R->bw = bw; R->valid = 1;
    return 1;
}
/* 8-channel tensors (bw 8) take the FULL lane-per-row kernels with NK 8: the general ones guarded every channel and computed the
   second, unstored SR word (P512 desk, 8-wide net level 0: apply 9.4 -> 6.0 ms, stats 4.5 -> 3.5 ms per call). Test hook: 0 the general ones */
static int g_gn_full8 = 1;
extern "C" void lp_set_gn_full8(int on) { g_gn_full8 = on; }
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
    int n0, cnt; gn_nwin(s, &n0, &cnt);   /* z-window: its slabs (the other rows' partial sums zero) */
    const dim3 grid((unsigned)slabs, (unsigned)nb, (unsigned)cnt);
    float *part = lp_buf<float>(5, slabs * 2 * NC);
    if (s.c > 1024) { fprintf(stderr, "lp_gn_silu_bwd_mx: C %d > 1024\n", s.c); abort(); }
    if (cnt < s.n) {
        if (gdt != 4) { fprintf(stderr, "lp_gn_silu_bwd_mx: the z-window needs MX-fp4 gradients\n"); abort(); }
        cudaMemsetAsync(part, 0, slabs * 2 * NC * sizeof(float));
    }
    cudaMemsetAsync(ds, 0, (size_t)2 * NC * sizeof(double));
    const uint8_t *xq = (const uint8_t *)x;
    static int s2on = -1; if (s2on < 0) s2on = getenv("UFSM_GN_APPLY2") ? atoi(getenv("UFSM_GN_APPLY2")) : 1;
    if (s2on && gdt == 4) {   /* MX-fp4 gradients: lane-pair kernel */
        const int full = bw >= 16 && s.c % bw == 0, full8 = bw == 8 && s.c == 8 && g_gn_full8 != 0;
        const uint8_t *gyq = (const uint8_t *)gy;
#define GBS3(B) do { if (bw == 32 && full) gn_silu_bwd_stats_mx2_k<B, 2, true><<<grid, 256>>>(xq, gyq, gamma, beta, mean, rstd, s.n, s.c, G, S, part, n0); \
                     else if (bw == 32) gn_silu_bwd_stats_mx2_k<B, 2, false><<<grid, 256>>>(xq, gyq, gamma, beta, mean, rstd, s.n, s.c, G, S, part, n0); \
                     else if (full) gn_silu_bwd_stats_mx2_k<B, 1, true><<<grid, 256>>>(xq, gyq, gamma, beta, mean, rstd, s.n, s.c, G, S, part, n0); \
                     else if (full8) gn_silu_bwd_stats_mx2_k<B, 1, true, 8><<<grid, 256>>>(xq, gyq, gamma, beta, mean, rstd, s.n, s.c, G, S, part, n0); \
                     else gn_silu_bwd_stats_mx2_k<B, 1, false><<<grid, 256>>>(xq, gyq, gamma, beta, mean, rstd, s.n, s.c, G, S, part, n0); } while (0)
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
/* x blocks of a persistent grid-stride kernel (256 threads): those resident on the whole GPU over the ny = channel blocks x batch rows
   of the grid (cached per kernel and device), at most need */
static int g_gn_persist = -1;   /* test hook: 0 a row per thread (the grid of need blocks: one trip, the pre-persistent kernels' schedule), -1 env UFSM_GN_PERSIST */
extern "C" void lp_set_gn_persist(int on) { g_gn_persist = on; }
static unsigned gn_pgrid(const void *k, size_t need, size_t ny) {
    static int env = -1; if (env < 0) env = getenv("UFSM_GN_PERSIST") ? atoi(getenv("UFSM_GN_PERSIST")) != 0 : 1;
    if (!(g_gn_persist >= 0 ? g_gn_persist : env)) return (unsigned)need;
    thread_local struct { const void *k; int dev, b; } c[16];
    thread_local int nc = 0;
    const int d = cur_dev_();
    int b = 0;
    for (int i = 0; i < nc; i++) if (c[i].k == k && c[i].dev == d) b = c[i].b;
    if (!b) {
        int per = 0, nsm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k, 256, 0); cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, d);
        b = (per > 0 ? per : 1) * (nsm > 0 ? nsm : 1);
        if (nc < 16) { c[nc].k = k; c[nc].dev = d; c[nc].b = b; nc++; }
    }
    size_t r = (size_t)b / ny; if (r < 1) r = 1;
    return (unsigned)(need < r ? need : r);
}
template <int B, int HPB, bool FULL, int NK = 16>
static void gba2(const uint8_t *xq, const uint8_t *gyq, const float *gamma, const float *beta, const float *mean, const float *rstd, const float *AB, uint8_t *gq,
                 shape5 s, int G, unsigned gsr) {
    const size_t S = shape_spatial(s);
    int n0, cnt; gn_nwin(s, &n0, &cnt);
    const dim3 g2(gn_pgrid((const void *)gn_silu_bwd_apply_mx2_k<B, HPB, FULL, NK>, (S * HPB + 255) / 256, (size_t)mx_nb(s.c) * cnt), (unsigned)mx_nb(s.c), (unsigned)cnt);
    gn_silu_bwd_apply_mx2_k<B, HPB, FULL, NK><<<g2, 256>>>(xq, gyq, gamma, beta, mean, rstd, AB, gq, s.n, s.c, G, S, gsr, n0);
}
extern "C" void lp_gn_silu_bwd_apply_mx(const void *x, int xdt, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                        const void *gy, void *gx, int gdt, const float *AB, unsigned gsr) {
    size_t S = shape_spatial(s);
    const dim3 ga(nblk_(S, 256), (unsigned)mx_nb(s.c), (unsigned)s.n);
    const uint8_t *xq = (const uint8_t *)x;
    static int a2 = -1; if (a2 < 0) a2 = getenv("UFSM_GN_APPLY2") ? atoi(getenv("UFSM_GN_APPLY2")) : 1;
    if (a2 && gdt == 4) {   /* MX-fp4 gradients (the trainer default): lane-pair kernel */
        const int bw = mx_bw(s.c), hpb = bw == 32 ? 2 : 1, full = bw >= 16 && s.c % bw == 0, full8 = bw == 8 && s.c == 8 && g_gn_full8 != 0;
        uint8_t *gq = (uint8_t *)gx; const uint8_t *gyq = (const uint8_t *)gy;
#define GBA2(B) do { if (hpb == 2 && full) gba2<B, 2, true>(xq, gyq, gamma, beta, mean, rstd, AB, gq, s, G, gsr); \
                     else if (hpb == 2) gba2<B, 2, false>(xq, gyq, gamma, beta, mean, rstd, AB, gq, s, G, gsr); \
                     else if (full) gba2<B, 1, true>(xq, gyq, gamma, beta, mean, rstd, AB, gq, s, G, gsr); \
                     else if (full8) gba2<B, 1, true, 8>(xq, gyq, gamma, beta, mean, rstd, AB, gq, s, G, gsr); \
                     else gba2<B, 1, false>(xq, gyq, gamma, beta, mean, rstd, AB, gq, s, G, gsr); } while (0)
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

/* dec[0]'s GN+SiLU backward from the recomputed head input gradient (gn_silu_bwd_*_hg_k): the stats into ds (as lp_gn_silu_bwd_mx) and
   the apply in place over x (as lp_gn_silu_bwd_apply_mx; no gy records below 32 channels: invalidated as lp_gn_silu_bwd_apply_rec
   does). C 8 or 16 MX-fp4, 1..4 fp16 logit channels, the lane-pair kernels on (UFSM_GN_APPLY2) */
extern "C" int lp_gn_silu_bwd_head_ok(int C, int ci) {
    static int a2 = -1; if (a2 < 0) a2 = getenv("UFSM_GN_APPLY2") ? atoi(getenv("UFSM_GN_APPLY2")) : 1;
    return a2 && (C == 8 || C == 16) && ci >= 1 && ci <= 4;
}
static int g_hg_exact = -1;   /* the recomputed gy: 1 exact fp32 (EX), 0 the two-op path's MX-fp4 bits, -1 env UFSM_GN_HEAD_EXACT (default 1) */
extern "C" void lp_set_gn_head_exact(int on) { g_hg_exact = on; }
static int hg_exact(void) {
    static int env = -1; if (env < 0) env = getenv("UFSM_GN_HEAD_EXACT") ? atoi(getenv("UFSM_GN_HEAD_EXACT")) != 0 : 1;
    return g_hg_exact >= 0 ? g_hg_exact : env;
}
#define HGK1(K, F, EX, ...) do { switch (ci) { case 1: K<1, F, EX><<<__VA_ARGS__; break; case 2: K<2, F, EX><<<__VA_ARGS__; break; \
                                              case 3: K<3, F, EX><<<__VA_ARGS__; break; default: K<4, F, EX><<<__VA_ARGS__; } } while (0)
#define HGK(K, F, ...) do { if (hg_exact()) HGK1(K, F, true, __VA_ARGS__); else HGK1(K, F, false, __VA_ARGS__); } while (0)
extern "C" void lp_gn_silu_bwd_head_stats(const void *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                          const void *gl, int ci, const float *wh, unsigned hs, double *ds, int ns) {
    size_t S = shape_spatial(s);
    const int NC = s.n * s.c, nb = mx_nb(s.c), bw = mx_bw(s.c);
    const size_t ny = (size_t)nb * s.n;   /* the slabs of lp_gn_silu_bwd_mx (the same partial sums) */
    size_t slabs = S / 4096, want = (160 + ny - 1) / ny, maxs = S / 512 > 0 ? S / 512 : 1;
    if (slabs < want) slabs = want;
    if (slabs > maxs) slabs = maxs;
    if (slabs > 1024) slabs = 1024;
    int n0, cnt; gn_nwin(s, &n0, &cnt);   /* z-window: its slabs (the other rows' partial sums zero) */
    const dim3 grid((unsigned)slabs, (unsigned)nb, (unsigned)cnt);
    float *part = lp_buf<float>(5, slabs * 2 * NC);
    if (cnt < s.n) cudaMemsetAsync(part, 0, slabs * 2 * NC * sizeof(float));
    cudaMemsetAsync(ds, 0, (size_t)2 * NC * sizeof(double));
    const uint8_t *xq = (const uint8_t *)x; const __half *g = (const __half *)gl;
    if (bw >= 16 && s.c % bw == 0) HGK(gn_silu_bwd_stats_mx2_hg_k, true, grid, 256>>>(xq, g, wh, hs, gamma, beta, mean, rstd, s.n, s.c, G, S, part, ns, n0));
    else HGK(gn_silu_bwd_stats_mx2_hg_k, false, grid, 256>>>(xq, g, wh, hs, gamma, beta, mean, rstd, s.n, s.c, G, S, part, ns, n0));
    gn_part_sum_k<<<nblk_((size_t)2 * NC * 32, 256), 256>>>(part, (int)slabs, NC, ds);
    LPCK();
}
template <int CI, bool F, bool EX>
static void hga(uint8_t *xq, const __half *g, const float *wh, unsigned hs, const float *gamma, const float *beta, const float *mean, const float *rstd,
                const float *AB, shape5 s, int G, unsigned gsr, int ns) {
    const size_t S = shape_spatial(s);
    int n0, cnt; gn_nwin(s, &n0, &cnt);
    const dim3 g2((unsigned)((S + 255) / 256), (unsigned)mx_nb(s.c), (unsigned)cnt);   /* a row per thread (a persistent loop hoisted the 48 coefficients: 171 registers, 1.4x slower) */
    gn_silu_bwd_apply_mx2_hg_k<CI, F, EX><<<g2, 256>>>(xq, g, wh, hs, gamma, beta, mean, rstd, AB, s.n, s.c, G, S, gsr, ns, n0);
}
extern "C" void lp_gn_silu_bwd_head_apply(void *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                          const void *gl, int ci, const float *wh, unsigned hs, const float *AB, unsigned gsr, int ns) {
    g_gyrec.valid = 0;
    const int bw = mx_bw(s.c), full = bw >= 16 && s.c % bw == 0, ex = hg_exact();
    uint8_t *xq = (uint8_t *)x; const __half *g = (const __half *)gl;
#define HGA1(F, EX) do { switch (ci) { case 1: hga<1, F, EX>(xq, g, wh, hs, gamma, beta, mean, rstd, AB, s, G, gsr, ns); break; \
                                       case 2: hga<2, F, EX>(xq, g, wh, hs, gamma, beta, mean, rstd, AB, s, G, gsr, ns); break; \
                                       case 3: hga<3, F, EX>(xq, g, wh, hs, gamma, beta, mean, rstd, AB, s, G, gsr, ns); break; \
                                       default: hga<4, F, EX>(xq, g, wh, hs, gamma, beta, mean, rstd, AB, s, G, gsr, ns); } } while (0)
    if (full && ex) HGA1(true, true); else if (full) HGA1(true, false); else if (ex) HGA1(false, true); else HGA1(false, false);
#undef HGA1
    LPCK();
}
#undef HGK
#undef HGK1

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
__global__ void __launch_bounds__(256) conv1_to_mx_t_k(const TG *x, const float *w, uint8_t *y, int N, int Co, size_t S, unsigned sr, int Ci,
                                                        size_t vbase = 0, size_t vn = 0) {   /* (vn: rows [vbase, vbase + vn), x holding vn per channel; 0: all) */
    __shared__ float sw[32 * CI];
    const int bw = mx_bw(Co), nb = mx_nb(Co), n = blockIdx.y / nb, blk = blockIdx.y % nb;
    for (int t = threadIdx.x; t < 32 * CI; t += blockDim.x) { const int co = blk * bw + t / CI, ci = t % CI; sw[t] = t / CI < bw && co < Co && ci < Ci ? w[co * Ci + ci] : 0.f; }
    __syncthreads();
    if (!vn) vn = S;
    const size_t v = vbase + blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (v >= vbase + vn) return;
    float xi[CI], r[32];
#pragma unroll
    for (int ci = 0; ci < CI; ci++) xi[ci] = ci < Ci ? ldx(x, ((size_t)n * Ci + ci) * vn + (v - vbase)) : 0.f;
#pragma unroll
    for (int k = 0; k < 32; k++) {
        float a = 0.f;
#pragma unroll
        for (int ci = 0; ci < CI; ci++) a += sw[k * CI + ci] * xi[ci];
        r[k] = a;
    }
    mx_store_row_b<YB>(y, mx_sc<YB>(y, N, Co, S), ((size_t)n * nb + blk) * S + v, bw, r, sr);
}
template <typename TG, int YB> static void conv1_to_mx_t(const TG *x, int N, int Ci, size_t S, const float *w, int Co, void *y, unsigned sr, size_t vb = 0, size_t vn = 0) {
    const dim3 grid(nblk_(vn ? vn : S, 256), N * mx_nb(Co));
    uint8_t *yq = (uint8_t *)y;
    switch (Ci) {
        case 1: conv1_to_mx_t_k<TG, 1, YB><<<grid, 256>>>(x, w, yq, N, Co, S, sr, Ci, vb, vn); break;
        case 2: conv1_to_mx_t_k<TG, 2, YB><<<grid, 256>>>(x, w, yq, N, Co, S, sr, Ci, vb, vn); break;
        case 3: conv1_to_mx_t_k<TG, 3, YB><<<grid, 256>>>(x, w, yq, N, Co, S, sr, Ci, vb, vn); break;
        case 4: conv1_to_mx_t_k<TG, 4, YB><<<grid, 256>>>(x, w, yq, N, Co, S, sr, Ci, vb, vn); break;
        case 5: conv1_to_mx_t_k<TG, 5, YB><<<grid, 256>>>(x, w, yq, N, Co, S, sr, Ci, vb, vn); break;
        case 6: conv1_to_mx_t_k<TG, 6, YB><<<grid, 256>>>(x, w, yq, N, Co, S, sr, Ci, vb, vn); break;
        case 7: conv1_to_mx_t_k<TG, 7, YB><<<grid, 256>>>(x, w, yq, N, Co, S, sr, Ci, vb, vn); break;
        case 8: conv1_to_mx_t_k<TG, 8, YB><<<grid, 256>>>(x, w, yq, N, Co, S, sr, Ci, vb, vn); break;
        default: conv1_to_mx_t_k<TG, 16, YB><<<grid, 256>>>(x, w, yq, N, Co, S, sr, Ci, vb, vn);   /* 9..16 logits (phase heads) */
    }
}
/* the same over rows [vb, vb + vn) (x holding vn voxels per channel; the level-0 slab stream's head, stage S4): fp16 / bf16 / fp32 x */
extern "C" void lp_conv1_to_mx_r(const void *x, int gdt, int N, int Ci, size_t S, const float *w, int Co, void *y, int ydt, unsigned sr, size_t vb, size_t vn) {
    if (Ci > 16) { fprintf(stderr, "lp_conv1_to_mx_r: Ci %d > 16\n", Ci); abort(); }
#define C1M(YB) do { if (gdt == 2) conv1_to_mx_t<__half, YB>((const __half *)x, N, Ci, S, w, Co, y, sr, vb, vn); \
                     else if (gdt == 1) conv1_to_mx_t<bf16, YB>((const bf16 *)x, N, Ci, S, w, Co, y, sr, vb, vn); \
                     else conv1_to_mx_t<float, YB>((const float *)x, N, Ci, S, w, Co, y, sr, vb, vn); } while (0)
    if (ydt == 4) C1M(4); else C1M(8);
#undef C1M
    LPCK();
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
