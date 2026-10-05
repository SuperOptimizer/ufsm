/* GroupNorm statistics of MX tensors */
#include "lp_mxops.cuh"
template <int B>
__global__ void __launch_bounds__(256) gn_sums_mx_k(const uint8_t *x, int N, int C, int G, size_t S, double *sums, size_t v0, size_t v1) {
    const int bw = mx_bw(C), nb = mx_nb(C), cpg = C / G;
    const int nblk_per = (int)((S + 255) / 256);
    const int n = blockIdx.x / nblk_per; const size_t v = (size_t)(blockIdx.x % nblk_per) * 256 + threadIdx.x;
    __shared__ float g1[64], g2[64];
    if (threadIdx.x < 64) { g1[threadIdx.x] = 0.f; g2[threadIdx.x] = 0.f; }
    __syncthreads();
    const uint8_t *sc = mx_sc<B>(x, N, C, S);
    for (int blk = 0; blk < nb; blk++) {
        float r[32];
        if (v < S && v >= v0 && v < v1) mx_load_row_b<B>(x, sc, ((size_t)n * nb + blk) * S + v, bw, r);   /* [v0, v1): voxels this GPU owns (spatial split) */
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
extern "C" int lp_gn_sums_mx(const void *x, int xdt, int N, int C, int G, size_t S, double *sums, size_t v0, size_t v1) {
    const int cpg = C / G;
    if (G > 64 || cpg < 1 || C % G) return -1;
    const int nblk_per = (int)((S + 255) / 256);
    if (xdt == 4) gn_sums_mx_k<4><<<N * nblk_per, 256>>>((const uint8_t *)x, N, C, G, S, sums, v0, v1);
    else gn_sums_mx_k<8><<<N * nblk_per, 256>>>((const uint8_t *)x, N, C, G, S, sums, v0, v1);
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
