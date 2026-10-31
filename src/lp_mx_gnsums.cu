/* GroupNorm statistics of MX tensors */
#include "lp_mxops.cuh"
/* The group sums (sum x, sum x^2) of an MX tensor (B bits) over the voxels [v0, v1) (a spatial split's own part). Blocks = (voxel slab,
   channel block, sample) as gn_silu_bwd_stats_mx2_k: each thread accumulates its rows of the slab per channel in registers, the next
   row's loads in flight (HPB 2: two lanes per 32-channel row, 16 channels each), one reduction per block and a double atomic per
   channel into its group's sums. (The kernel this replaced, a block per 256 voxels and double atomics per block and group on the same
   2 N G addresses, took ~14 ms per 8-channel level-0 call at P512 on the desk: 55 ms of an 8-wide net's step.) */
template <int B, int HPB>
__global__ void __launch_bounds__(256) gn_sums_mx2_k(const uint8_t *x, int N, int C, int G, size_t S, double *sums, size_t v0, size_t v1, int n0 = 0) {   /* n0: first sample */
    const int bw = mx_bw(C), nb = mx_nb(C), cpg = C / G, blk = blockIdx.y, n = n0 + (int)blockIdx.z, nsl = gridDim.x;
    const int h = HPB == 2 ? (int)(threadIdx.x & 1) : 0, h0 = 16 * h, tv = (int)threadIdx.x / HPB, nt = 256 / HPB;
    size_t q0 = S * blockIdx.x / nsl, q1 = S * (blockIdx.x + 1) / nsl;
    if (q0 < v0) q0 = v0;
    if (q1 > v1) q1 = v1;
    __shared__ float w1[8][32], w2[8][32];
    w1[threadIdx.x >> 5][threadIdx.x & 31] = 0.f; w2[threadIdx.x >> 5][threadIdx.x & 31] = 0.f;
    __syncthreads();
    const int kmax = max(0, min(min(bw, C - blk * bw) - h0, 16));
    const size_t rbase = ((size_t)n * nb + blk) * S;
    const uint8_t *sx = mx_sc<B>(x, N, C, S);
    const int rbx = mx_rb(bw, B);
    float s1[16], s2[16];
#pragma unroll
    for (int k = 0; k < 16; k++) { s1[k] = 0.f; s2[k] = 0.f; }
    uint4 xw = make_uint4(0u, 0u, 0u, 0u); unsigned xs = 0u;
    auto load = [&](size_t v) {
        const size_t ri = rbase + v;
        if constexpr (B == 4) { const uint2 t = bw == 8 ? make_uint2(__ldg((const unsigned *)(x + ri * rbx)), 0u) : __ldg((const uint2 *)(x + ri * rbx + 8 * h)); xw.x = t.x; xw.y = t.y; }
        else if (bw == 8) { const uint2 t = __ldg((const uint2 *)(x + ri * rbx)); xw.x = t.x; xw.y = t.y; }   /* MX-fp8 8-channel row: 8 bytes */
        else xw = __ldg((const uint4 *)(x + ri * rbx + 16 * h));
        xs = __ldg(sx + ri);
    };
    size_t q = q0 + tv;
    if (q < q1) load(q);
    for (; q < q1; q += nt) {
        const uint4 cx = xw; const float fx = mx_scale(xs);
        if (q + nt < q1) load(q + nt);   /* next row in flight during this one's arithmetic */
        float r[16];
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
#pragma unroll
        for (int k = 0; k < 16; k++) if (k < kmax) { s1[k] += r[k]; s2[k] = fmaf(r[k], r[k], s2[k]); }
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
        const size_t o = 2 * ((size_t)n * G + (blk * bw + (int)threadIdx.x) / cpg);
        gn_acc_add(sums, o, (double)a); gn_acc_add(sums, o + 1, (double)b2);   /* exact fixed point (src/gnacc.cuh) */
    }
}
extern "C" int lp_gn_sums_mx(const void *x, int xdt, int N, int C, int G, size_t S, double *sums, size_t v0, size_t v1) { return lp_gn_sums_mx_n(x, xdt, N, C, G, S, sums, v0, v1, 0, N); }
/* samples [n0, n0 + cnt) of an N-sample tensor only, with the slab partition of the whole launch (the same per-block float sums: the
   level-0 slab stream's per-slab statistics equal the whole tensor's pass on the pseudo-sample shape) */
extern "C" int lp_gn_sums_mx_n(const void *x, int xdt, int N, int C, int G, size_t S, double *sums, size_t v0, size_t v1, int n0, int cnt) {
    const int cpg = C / G;
    if (G > 64 || cpg < 1 || C % G) return -1;
    const int nb = mx_nb(C), hpb = mx_bw(C) == 32 ? 2 : 1;
    /* slabs as lp_gn_silu_bwd_mx: >= ~16 rows per thread, >= ~160 blocks in all, <= 1024 per (channel block, sample) */
    const size_t ny = (size_t)nb * N;
    size_t slabs = S / 4096, want = (160 + ny - 1) / ny, maxs = S / 512 > 0 ? S / 512 : 1;
    if (slabs < want) slabs = want;
    if (slabs > maxs) slabs = maxs;
    if (slabs > 1024) slabs = 1024;
    if (cnt <= 0) return 0;
    const dim3 grid((unsigned)slabs, (unsigned)nb, (unsigned)cnt);
    const uint8_t *xq = (const uint8_t *)x;
    if (xdt == 4) { if (hpb == 2) gn_sums_mx2_k<4, 2><<<grid, 256>>>(xq, N, C, G, S, sums, v0, v1, n0); else gn_sums_mx2_k<4, 1><<<grid, 256>>>(xq, N, C, G, S, sums, v0, v1, n0); }
    else { if (hpb == 2) gn_sums_mx2_k<8, 2><<<grid, 256>>>(xq, N, C, G, S, sums, v0, v1, n0); else gn_sums_mx2_k<8, 1><<<grid, 256>>>(xq, N, C, G, S, sums, v0, v1, n0); }
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
