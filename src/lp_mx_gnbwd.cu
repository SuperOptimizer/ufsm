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
    __shared__ float cm[32], cr[32], cg[32], cb[32], s1[32], s2[32];
    if (threadIdx.x < 32) {
        const int k = threadIdx.x, c = blk * bw + k;
        const bool ok = k < bw && c < C;
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
    for (size_t q = q0 + tv; q < q1; q += nt) {
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
                    const float xhat = (r[k] - cm[h0 + k]) * cr[h0 + k], u = xhat * cg[h0 + k] + cb[h0 + k];
                    const float a = (IS_MX8(TG) ? gr[k] : g[j][k]) * dsilu(u);
                    a1[k] += a; a2[k] += a * xhat;
                }
            }
        }
    }
#pragma unroll
    for (int k = 0; k < 16; k++) {   /* kmax differs between the halves only in its padding tail: reduce all 16, add the real ones */
        float a = a1[k], q = a2[k];
        for (int o = 16; o >= HPB; o >>= 1) { a += __shfl_xor_sync(0xffffffff, a, o); q += __shfl_xor_sync(0xffffffff, q, o); }
        if ((int)(threadIdx.x & 31) < HPB && k < kmax) { atomicAdd(&s1[h0 + k], a); atomicAdd(&s2[h0 + k], q); }
    }
    __syncthreads();
    /* per-slab partials (same-address double atomics across slabs serialised at the small levels); summed by gn_part_sum_k */
    const int kb = min(bw, C - blk * bw);
    if ((int)threadIdx.x < kb) { const size_t c = (size_t)n * C + blk * bw + threadIdx.x, o = ((size_t)blockIdx.x * N * C + c) * 2; part[o] = s1[threadIdx.x]; part[o + 1] = s2[threadIdx.x]; }
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
            const float xhat = (r[k] - cm[k]) * rs, u = xhat * ga + cb[k];
            const float a = (IS_MX8(TG) ? gr[k] : ldx(gyb, (size_t)k * S)) * dsilu(u);
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
    const size_t ny = (size_t)nb * s.n;
    size_t slabs = S / 4096, want = (160 + ny - 1) / ny, maxs = S / 512 > 0 ? S / 512 : 1;
    if (slabs < want) slabs = want;
    if (slabs > maxs) slabs = maxs;
    if (slabs > 1024) slabs = 1024;
    const dim3 grid((unsigned)slabs, (unsigned)nb, (unsigned)s.n);
    float *part = lp_buf<float>(5, slabs * 2 * NC);
    if (s.c > 1024) { fprintf(stderr, "lp_gn_silu_bwd_mx: C %d > 1024\n", s.c); abort(); }
    if (gdt == 4) { fprintf(stderr, "lp_gn_silu_bwd_mx: fp4 gradients are not supported\n"); abort(); }
    cudaMemsetAsync(ds, 0, (size_t)2 * NC * sizeof(double));
    const uint8_t *xq = (const uint8_t *)x;
#define GBS2(B, V, H) do { if (gdt == 3) gn_silu_bwd_stats_mx_k<B, mx8_t, 1, H><<<grid, 256>>>(xq, (const mx8_t *)gy, gamma, beta, mean, rstd, s.n, s.c, G, S, part); \
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
/* the same with the input count a template parameter: grid (voxels, n x MX block), so no 64-bit index divisions, the block's
   weights staged in smem, the CI inputs in registers; same arithmetic order as conv1_to_mx_k */
template <typename TG, int CI>
__global__ void __launch_bounds__(256) conv1_to_mx_t_k(const TG *x, const float *w, uint8_t *y, int N, int Co, size_t S) {
    __shared__ float sw[32 * CI];
    const int bw = mx_bw(Co), nb = mx_nb(Co), n = blockIdx.y / nb, blk = blockIdx.y % nb;
    for (int t = threadIdx.x; t < 32 * CI; t += blockDim.x) { const int co = blk * bw + t / CI; sw[t] = t / CI < bw && co < Co ? w[co * CI + t % CI] : 0.f; }
    __syncthreads();
    const size_t v = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (v >= S) return;
    float xi[CI], r[32];
#pragma unroll
    for (int ci = 0; ci < CI; ci++) xi[ci] = ldx(x, ((size_t)n * CI + ci) * S + v);
#pragma unroll
    for (int k = 0; k < 32; k++) {
        float a = 0.f;
#pragma unroll
        for (int ci = 0; ci < CI; ci++) a += sw[k * CI + ci] * xi[ci];
        r[k] = a;
    }
    mx_store_row(y, y + (size_t)N * nb * S * bw, ((size_t)n * nb + blk) * S + v, bw, r);
}
template <typename TG> static void conv1_to_mx_t(const TG *x, int N, int Ci, size_t S, const float *w, int Co, void *y) {
    const dim3 grid(nblk_(S, 256), N * mx_nb(Co));
    uint8_t *yq = (uint8_t *)y;
    switch (Ci) {
        case 1: conv1_to_mx_t_k<TG, 1><<<grid, 256>>>(x, w, yq, N, Co, S); break;
        case 2: conv1_to_mx_t_k<TG, 2><<<grid, 256>>>(x, w, yq, N, Co, S); break;
        case 3: conv1_to_mx_t_k<TG, 3><<<grid, 256>>>(x, w, yq, N, Co, S); break;
        case 4: conv1_to_mx_t_k<TG, 4><<<grid, 256>>>(x, w, yq, N, Co, S); break;
        case 5: conv1_to_mx_t_k<TG, 5><<<grid, 256>>>(x, w, yq, N, Co, S); break;
        case 6: conv1_to_mx_t_k<TG, 6><<<grid, 256>>>(x, w, yq, N, Co, S); break;
        case 7: conv1_to_mx_t_k<TG, 7><<<grid, 256>>>(x, w, yq, N, Co, S); break;
        default: conv1_to_mx_t_k<TG, 8><<<grid, 256>>>(x, w, yq, N, Co, S);
    }
}
extern "C" void lp_conv1_to_mx(const void *x, int gdt, int N, int Ci, size_t S, const float *w, int Co, void *y) {
    if (Ci > 8) { fprintf(stderr, "lp_conv1_to_mx: Ci %d > 8\n", Ci); abort(); }
    static int old = -1; if (old < 0) old = getenv("UFSM_CONV1_OLD") ? atoi(getenv("UFSM_CONV1_OLD")) : 0;
    if (!old) {
        if (gdt == 2) conv1_to_mx_t((const __half *)x, N, Ci, S, w, Co, y);
        else if (gdt == 1) conv1_to_mx_t((const bf16 *)x, N, Ci, S, w, Co, y);
        else conv1_to_mx_t((const float *)x, N, Ci, S, w, Co, y);
        LPCK();
        return;
    }
    size_t n = (size_t)N * mx_nb(Co) * S;
    if (gdt == 2) conv1_to_mx_k<__half><<<nblk_(n, 256), 256>>>((const __half *)x, w, (uint8_t *)y, N, Ci, Co, S);
    else if (gdt == 1) conv1_to_mx_k<bf16><<<nblk_(n, 256), 256>>>((const bf16 *)x, w, (uint8_t *)y, N, Ci, Co, S);
    else conv1_to_mx_k<float><<<nblk_(n, 256), 256>>>((const float *)x, w, (uint8_t *)y, N, Ci, Co, S);
    LPCK();
}
