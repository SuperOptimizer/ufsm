/* --l0-norm slab: an MX tensor's GroupNorm affine in place, x <- (x - mean) rstd per (sample, group) (mean nullptr: 0), stored back
   with exact SR keyed by sr (fp4; fp8 rounds to nearest). The per-slab statistics come from a pseudo-sample shape (unet.c l0s_shape:
   one channel block, so (n, z-slab) is a sample of the same layout) */
#include "lp_mxops.cuh"
template <int B>
__global__ void __launch_bounds__(256) mx_affine_k(uint8_t *x, int N, int C, size_t S, int G, const float *mean, const float *rstd, unsigned sr) {
    const int bw = mx_bw(C), nb = mx_nb(C), cpg = C / G, blk = blockIdx.y, n = blockIdx.z;
    __shared__ float cm[32], cr[32];
    if (threadIdx.x < 32) {
        const int k = threadIdx.x, c = blk * bw + k;
        const bool ok = k < bw && c < C;
        const int ng = n * G + (ok ? c / cpg : 0);
        cm[k] = ok && mean ? mean[ng] : 0.f; cr[k] = ok ? rstd[ng] : 0.f;
    }
    __syncthreads();
    const size_t v = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (v >= S) return;
    const size_t i = ((size_t)n * nb + blk) * S + v;
    uint8_t *sc = mx_sc<B>(x, N, C, S);
    float r[32];
    mx_load_row_b<B>(x, sc, i, bw, r);
#pragma unroll
    for (int k = 0; k < 32; k++) r[k] = k < bw ? (r[k] - cm[k]) * cr[k] : 0.f;
    mx_store_row_b<B>(x, sc, i, bw, r, sr);
}
extern "C" int lp_mx_affine(void *x, int bits, shape5 s, int G, const float *mean, const float *rstd, unsigned sr) {
    const size_t S = shape_spatial(s);
    const dim3 g(nblk_(S, 256), (unsigned)mx_nb(s.c), (unsigned)s.n);
    if (bits == 4) mx_affine_k<4><<<g, 256>>>((uint8_t *)x, s.n, s.c, S, G, mean, rstd, sr);
    else if (bits == 8) mx_affine_k<8><<<g, 256>>>((uint8_t *)x, s.n, s.c, S, G, mean, rstd, sr);
    else return -1;
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
