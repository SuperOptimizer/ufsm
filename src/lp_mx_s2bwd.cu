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
/* ---- wide layers on the tensor cores: gx = conv_s1(dilate2(gy), flip(w)^T) with the MX-fp8 stride-1 kernel. dilate2 puts gy
   row o at fine position u = 2 o (every axis) and zero rows elsewhere, so gx[u] = sum_k w[k] gd[u + 1 - k] is a pad-1 stride-1
   convolution with the spatially flipped, channel-transposed weights. 8x the MACs of the parity form, but fp8 MMA instead of
   fp32 FMA (and the weights are quantised to fp8 like every other tensor-core pass); gy rows are copied verbatim. */
__global__ void dilate2_mx8_k(const uint8_t *gy, uint8_t *gd, int NB, int D, int H, int W, int Do, int Ho, int Wo) {
    const unsigned S = (unsigned)D * H * W, So = (unsigned)Do * Ho * Wo;
    const unsigned u = blockIdx.x * blockDim.x + threadIdx.x; if (u >= S) return;
    const size_t r = (size_t)blockIdx.y * S + u;   /* row (n * nb + blk, u); 32 e4m3 bytes, scales after the NB * S rows */
    const unsigned ux = u % W, uy = (u / W) % H, uz = u / (W * H);
    uint4 *d = (uint4 *)(gd + r * 32);
    if (!(ux & 1) && !(uy & 1) && !(uz & 1) && (uz >> 1) < (unsigned)Do && (uy >> 1) < (unsigned)Ho && (ux >> 1) < (unsigned)Wo) {
        const size_t ro = (size_t)blockIdx.y * So + ((size_t)(uz >> 1) * Ho + (uy >> 1)) * Wo + (ux >> 1);
        const uint4 *s = (const uint4 *)(gy + ro * 32);
        d[0] = s[0]; d[1] = s[1]; gd[(size_t)NB * S * 32 + r] = gy[(size_t)NB * So * 32 + ro];
    } else { d[0] = d[1] = make_uint4(0, 0, 0, 0); gd[(size_t)NB * S * 32 + r] = 127; }
}
__global__ void flipT_w_k(const float *w, float *wt, int Co, int Ci) {   /* wt[ci][co][t] = w[co][ci][26 - t] */
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= (size_t)Co * Ci * 27) return;
    const int t = (int)(i % 27), co = (int)((i / 27) % Co), ci = (int)(i / ((size_t)27 * Co));
    wt[i] = w[((size_t)co * Ci + ci) * 27 + 26 - t];
}
__global__ void mx8_add_k(uint8_t *y, const uint8_t *t, size_t rows) {   /* y += t, MX-fp8 rows of 32 (re-encoded) */
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= rows) return;
    float a[32], b[32];
    mx_load_row_b<8>(y, y + rows * 32, i, 32, a); mx_load_row_b<8>(t, t + rows * 32, i, 32, b);
#pragma unroll
    for (int k = 0; k < 32; k++) a[k] += b[k];
    mx_store_row_b<8>(y, y + rows * 32, i, 32, a);
}
static int s2b_dilate(const void *gy, shape5 ys, const float *w, shape5 xs, void *gx, int accum) {
    static int on = -1; if (on < 0) on = getenv("UFSM_S2B_DIL") ? atoi(getenv("UFSM_S2B_DIL")) : 1;
    if (!on || xs.c < 64 || ys.c < 64 || mx_bw(xs.c) != 32 || mx_bw(ys.c) != 32) return 0;
    const size_t S = shape_spatial(xs), rd = (size_t)xs.n * mx_nb(ys.c) * S, rx = (size_t)xs.n * mx_nb(xs.c) * S;
    uint8_t *gd = lp_buf<uint8_t>(2, rd * 33), *tmp = accum ? lp_buf<uint8_t>(3, rx * 33) : (uint8_t *)gx;
    float *wt = lp_buf<float>(4, (size_t)ys.c * xs.c * 27);
    dilate2_mx8_k<<<dim3(nblk_(S, 256), (unsigned)(xs.n * mx_nb(ys.c))), 256>>>((const uint8_t *)gy, gd, xs.n * mx_nb(ys.c), xs.d, xs.h, xs.w, ys.d, ys.h, ys.w);
    flipT_w_k<<<nblk_((size_t)ys.c * xs.c * 27, 256), 256>>>(w, wt, ys.c, xs.c);
    shape5 ds = xs; ds.c = ys.c;
    lp_conv_fwd_f8(gd, 3, ds, wt, nullptr, xs.c, tmp, 3, gnp_t{}, nullptr, 0, split_t{});
    if (accum) mx8_add_k<<<nblk_(rx, 256), 256>>>((uint8_t *)gx, tmp, rx);
    LPCK();
    return 1;
}
/* MX row format change (BI -> BO bits, same channels): fp4 -> fp8 is exact; fp8 -> fp4 with exact SR keyed by sr */
template <int BI, int BO> __global__ void mx_recode_k(const uint8_t *x, uint8_t *y, int N, int C, size_t S, unsigned sr) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, rows = (size_t)N * mx_nb(C) * S;
    if (i >= rows) return;
    const int bw = mx_bw(C);
    float r[32];
    mx_load_row_b<BI>(x, mx_sc<BI>(x, N, C, S), i, bw, r);
    mx_store_row_b<BO>(y, mx_sc<BO>(y, N, C, S), i, bw, r, sr);
}
extern "C" int lp_bwd_data_s2_mx_tc(const void *gy, shape5 ys, const float *w, shape5 xs, void *gx, int accum, int gdt, int xdt, unsigned osr);   /* lp_mx_s2bwd_tc.cu */
/* returns the compute precision used: 2 (fp8 tensor cores, wide layers), 1 (bf16 tensor cores, 16 / 32 channels) or 0 (fp32) */
extern "C" int lp_bwd_data_s2_mx(const void *gy, shape5 ys, const float *w, shape5 xs, void *gx, int accum, int gdt, int xdt, unsigned osr) {
    /* the direct tensor-core kernel first (16 / 32 / 64 channels: no fine-level scratch), then the dilated fp8 conv (wider) */
    if (lp_bwd_data_s2_mx_tc(gy, ys, w, xs, gx, accum, gdt, xdt, osr)) return 1;
    if (gdt == 4 || xdt == 4) {   /* MX-fp4 gradients on the wide (small, coarse) levels: through MX-fp8 copies (gy and the old gx exact,
                                     the result back to fp4 with SR) */
        if (gdt != 4 || xdt != 4) { fprintf(stderr, "lp_bwd_data_s2_mx: gy and gx must share the MX format\n"); abort(); }
        const size_t Sy = shape_spatial(ys), Sx = shape_spatial(xs), ry = (size_t)ys.n * mx_nb(ys.c) * Sy, rx = (size_t)xs.n * mx_nb(xs.c) * Sx;
        uint8_t *gy8 = lp_buf<uint8_t>(5, lp_mx8_bytes(ys.n, ys.c, Sy)), *gx8 = lp_buf<uint8_t>(4, lp_mx8_bytes(xs.n, xs.c, Sx));   /* slots 4 (the wgrad gy pre-pass, free here) and 5 */
        mx_recode_k<4, 8><<<nblk_(ry, 256), 256>>>((const uint8_t *)gy, gy8, ys.n, ys.c, Sy, 0u);
        if (accum) mx_recode_k<4, 8><<<nblk_(rx, 256), 256>>>((const uint8_t *)gx, gx8, xs.n, xs.c, Sx, 0u);
        const int pr = lp_bwd_data_s2_mx(gy8, ys, w, xs, gx8, accum, 3, 3, 0u);
        mx_recode_k<8, 4><<<nblk_(rx, 256), 256>>>(gx8, (uint8_t *)gx, xs.n, xs.c, Sx, osr);
        LPCK();
        return pr;
    }
    if (s2b_dilate(gy, ys, w, xs, gx, accum)) return 2;
    {
        const int bwx = mx_bw(xs.c), nbx = mx_nb(xs.c);
        const size_t smem = (size_t)8 * ys.c * bwx * sizeof(float);
        if (smem > 96 * 1024) { lp_bwd_data_s2_mx_wide(gy, ys, w, xs, gx, accum); return 0; }   /* wide layers: thread-per-voxel kernel */
        static int attr[8];
        if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)bwd_data_s2_mx2_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
        const size_t mmax = (size_t)((xs.d + 1) / 2) * ((xs.h + 1) / 2) * ((xs.w + 1) / 2);
        const dim3 grid(nblk_(mmax, 128), 8, (unsigned)(nbx * xs.n));
        bwd_data_s2_mx2_k<<<grid, 128, smem>>>((const uint8_t *)gy, w, (uint8_t *)gx, xs.n, xs.c, ys.c, xs.d, xs.h, xs.w, ys.d, ys.h, ys.w, accum);
        LPCK();
        return 0;
    }
}
