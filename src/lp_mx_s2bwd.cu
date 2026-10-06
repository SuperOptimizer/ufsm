/* MX stride-2 backward-data */
#include "lp_mxops.cuh"
/* ---- MX stride-2 backward-data (k = 3, pad 1): gx[ci][u] (+)= sum over the taps k with (u + 1 - k) even, o = (u + 1 - k) / 2
   in range, of sum_co w[co][ci][k] gy[co][o]. */
/* MX-fp8 stride-2 backward-data, parity-major: a block covers one parity class p = (pz, py, px) of the gx grid (u = 2 m + p
   per axis), so every thread of the block uses the same 1..8 taps (k = 1 for an even coordinate: o = m; k = 0 / 2 for an odd
   one: o = m + 1 / m) and reads the weights of those taps as broadcast float4 from shared memory: [tap][co][32 ci of this
   block]. Thread = one gx voxel of the class and one ci block (all its bwx channels), gy rows of consecutive threads are
   consecutive voxels (coalesced). The former thread-per-voxel kernel mixed parities inside a warp (divergent tap sets, bank
   conflicted weights): down0 3.66 ms vs 0.59 for the 16-bit tensor-core path at 96^3 B2. */
__global__ void __launch_bounds__(128) bwd_data_s2_mx2_k(const uint8_t *gy, const float *w, uint8_t *gx, int N, int Ci, int Co,
                                                         int D, int H, int W, int Do, int Ho, int Wo, int accum) {
    extern __shared__ float4 swq[];   /* [ntap][Co][bwx / 4] */
    const int bwx = mx_bw(Ci), nbx = mx_nb(Ci), bwy = mx_bw(Co), nby = mx_nb(Co);
    const int p = blockIdx.y, pz = p >> 2, py = (p >> 1) & 1, px = p & 1;
    const int blk = blockIdx.z % nbx, n = blockIdx.z / nbx;
    const int Dp = (D - pz + 1) / 2, Hp = (H - py + 1) / 2, Wp = (W - px + 1) / 2;
    /* tap sets per axis: even -> {k 1, o = m}; odd -> {k 0, o = m + 1}, {k 2, o = m} */
    const int nz = pz ? 2 : 1, ny = py ? 2 : 1, nx = px ? 2 : 1, ntap = nz * ny * nx;
    const int q4 = bwx / 4;
    float *sw = (float *)swq;
    for (int i = threadIdx.x; i < ntap * Co * bwx; i += blockDim.x) {
        const int cc = i % bwx, co = (i / bwx) % Co, tl = i / (bwx * Co);
        const int a = tl / (ny * nx), bb = (tl / nx) % ny, c = tl % nx;
        const int kz = pz ? (a ? 2 : 0) : 1, ky = py ? (bb ? 2 : 0) : 1, kx = px ? (c ? 2 : 0) : 1;
        const int ci = blk * bwx + cc;
        sw[i] = ci < Ci ? w[((size_t)co * Ci + ci) * 27 + (kz * 3 + ky) * 3 + kx] : 0.f;
    }
    __syncthreads();
    const size_t mi = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (mi >= (size_t)Dp * Hp * Wp) return;
    const int mx = (int)(mi % Wp), my = (int)((mi / Wp) % Hp), mz = (int)(mi / ((size_t)Wp * Hp));
    const int uz = 2 * mz + pz, uy = 2 * my + py, ux = 2 * mx + px;
    const size_t So = (size_t)Do * Ho * Wo, S = (size_t)D * H * W;
    const uint8_t *scy = mx_sc<8>(gy, N, Co, So);
    float acc[32];
#pragma unroll
    for (int k = 0; k < 32; k++) acc[k] = 0.f;
    for (int tl = 0; tl < ntap; tl++) {
        const int a = tl / (ny * nx), bb = (tl / nx) % ny, c = tl % nx;
        const int oz = mz + (pz && !a ? 1 : 0), oy = my + (py && !bb ? 1 : 0), ox = mx + (px && !c ? 1 : 0);
        if (oz >= Do || oy >= Ho || ox >= Wo) continue;
        const size_t vo = ((size_t)oz * Ho + oy) * Wo + ox;
        for (int yb = 0; yb < nby; yb++) {
            float r[32];
            mx_load_row_b<8>(gy, scy, ((size_t)n * nby + yb) * So + vo, bwy, r);
            const float4 *wt = swq + ((size_t)tl * Co + yb * bwy) * q4;
#pragma unroll
            for (int k = 0; k < 32; k++) {
                if (k < bwy && yb * bwy + k < Co) {
                    const float g = r[k];
                    const float4 *wr = wt + (size_t)k * q4;
#pragma unroll
                    for (int j = 0; j < 8; j++) if (j < q4) { const float4 f = wr[j]; acc[4 * j] += f.x * g; acc[4 * j + 1] += f.y * g; acc[4 * j + 2] += f.z * g; acc[4 * j + 3] += f.w * g; }
                }
            }
        }
    }
    const size_t ri = ((size_t)n * nbx + blk) * S + ((size_t)uz * H + uy) * W + ux;
    uint8_t *scx = mx_sc<8>(gx, N, Ci, S);
    if (accum) {
        float r[32];
        mx_load_row_b<8>(gx, scx, ri, bwx, r);
#pragma unroll
        for (int k = 0; k < 32; k++) acc[k] += r[k];
    }
    mx_store_row_b<8>(gx, scx, ri, bwx, acc);
}
extern "C" void lp_bwd_data_s2_mx_wide(const void *gy, shape5 ys, const float *w, shape5 xs, void *gx, int accum);   /* lp_mx_s2bwd_wide.cu */
extern "C" void lp_bwd_data_s2_mx(const void *gy, shape5 ys, const float *w, shape5 xs, void *gx, int accum) {
    {
        const int bwx = mx_bw(xs.c), nbx = mx_nb(xs.c);
        const size_t smem = (size_t)8 * ys.c * bwx * sizeof(float);
        if (smem > 96 * 1024) { lp_bwd_data_s2_mx_wide(gy, ys, w, xs, gx, accum); return; }   /* wide layers: thread-per-voxel kernel */
        static int attr[8];
        if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)bwd_data_s2_mx2_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
        const size_t mmax = (size_t)((xs.d + 1) / 2) * ((xs.h + 1) / 2) * ((xs.w + 1) / 2);
        const dim3 grid(nblk_(mmax, 128), 8, (unsigned)(nbx * xs.n));
        bwd_data_s2_mx2_k<<<grid, 128, smem>>>((const uint8_t *)gy, w, (uint8_t *)gx, xs.n, xs.c, ys.c, xs.d, xs.h, xs.w, ys.d, ys.h, ys.w, accum);
        LPCK();
        return;
    }
}
