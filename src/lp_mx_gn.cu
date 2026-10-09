/* GroupNorm+SiLU apply and 2x upsample forward on MX tensors */
#include "lp_mxops.cuh"
/* row ri of block blk of a plane-major [n][C][S] tensor as a 32-wide row (bw channels, rest zero) */
/* y = silu(gn(x)): x MX (BX bits) or plane-major (BX = 0, type TI), y MX (BY bits) */
template <int BX, int BY, typename TI>
__global__ void gn_silu_apply_mx_k(const void *xv, uint8_t *y, int N, int C, int G, size_t S, const float *gamma, const float *beta, const float *mean, const float *rstd) {
    const int bw = mx_bw(C), nb = mx_nb(C), cpg = C / G;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * nb * S) return;
    int blk = (int)((i / S) % nb), n = (int)(i / (S * nb));
    float r[32];
    if constexpr (BX != 0) { const uint8_t *x = (const uint8_t *)xv; mx_load_row_b<BX>(x, mx_sc<BX>(x, N, C, S), i, bw, r); }
    else pm_load_row<TI>((const TI *)xv, N, C, S, i, bw, r);
#pragma unroll
    for (int k = 0; k < 32; k++) {
        int c = blk * bw + k;
        if (k < bw && c < C) { int ng = n * G + c / cpg; float v = (r[k] - mean[ng]) * rstd[ng] * gamma[c] + beta[c]; r[k] = v / (1.f + __expf(-v)); }
        else r[k] = 0.f;
    }
    mx_store_row_b<BY>(y, mx_sc<BY>(y, N, C, S), i, bw, r);
}
extern "C" void lp_gn_silu_apply_mx(const void *x, int xdt, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd, void *y, int ydt) {
    size_t S = shape_spatial(s), n = (size_t)s.n * mx_nb(s.c) * S;
    if (ydt != 3 && ydt != 4) { fprintf(stderr, "lp_gn_silu_apply_mx: output dtype %d is not MX\n", ydt); abort(); }
#define GSA(BX, BY, TI) gn_silu_apply_mx_k<BX, BY, TI><<<nblk_(n, 256), 256>>>(x, (uint8_t *)y, s.n, s.c, G, S, gamma, beta, mean, rstd)
#define GSA_Y(BX, TI) do { if (ydt == 4) GSA(BX, 4, TI); else GSA(BX, 8, TI); } while (0)
    if (xdt == 4) GSA_Y(4, float); else if (xdt == 3) GSA_Y(8, float); else if (xdt == 2) GSA_Y(0, __half); else if (xdt == 1) GSA_Y(0, bf16); else GSA_Y(0, float);
#undef GSA_Y
#undef GSA
    LPCK();
}
/* exact-2x trilinear upsample (align_corners = false, edge-clamped): out[o] = 0.75 in[o/2] + 0.25 in[o/2 -+ 1] per axis;
   x MX (BX bits), y MX (BY bits), gp: silu(gn(x)) is upsampled */
template <int BX, int BY>
__global__ void up2_mx_k(const uint8_t *x, uint8_t *y, int N, int C, int D, int H, int W, gnp_t gp) {
    /* grid (fine voxels, n x channel block): the block's GN+SiLU coefficients once in smem (row_gn reloaded gamma / beta /
       mean / rstd from global memory for every value of the 8 coarse rows); same coefficients and arithmetic order */
    const int bw = mx_bw(C), nb = mx_nb(C), Do = 2 * D, Ho = 2 * H, Wo = 2 * W;
    const size_t S = (size_t)D * H * W, So = (size_t)Do * Ho * Wo;
    const int nbk = blockIdx.y, n = nbk / nb, blk = nbk % nb;
    __shared__ float ca[32], cb[32];
    if (threadIdx.x < 32) {
        const int k = threadIdx.x, ci = blk * bw + k;
        float a = 1.f, b = 0.f;
        if (gp.G && k < bw && ci < C) { const int ng = n * gp.G + ci / (C / gp.G); a = gp.rstd[ng] * gp.gamma[ci]; b = gp.beta[ci] - gp.mean[ng] * a; }
        ca[k] = a; cb[k] = b;
    }
    __syncthreads();
    const size_t vo = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (vo >= So) return;
    const int ox = (int)(vo % Wo), oy = (int)((vo / Wo) % Ho), oz = (int)(vo / ((size_t)Wo * Ho));
    int mz[2] = {oz >> 1, min(max((oz >> 1) + ((oz & 1) ? 1 : -1), 0), D - 1)}, my[2] = {oy >> 1, min(max((oy >> 1) + ((oy & 1) ? 1 : -1), 0), H - 1)}, mx[2] = {ox >> 1, min(max((ox >> 1) + ((ox & 1) ? 1 : -1), 0), W - 1)};
    const float wt[2] = {0.75f, 0.25f};
    const uint8_t *sc = mx_sc<BX>(x, N, C, S);
    const int kmax = min(bw, C - blk * bw);
    float acc[32] = {};
#pragma unroll
    for (int a = 0; a < 2; a++)
#pragma unroll
        for (int bb = 0; bb < 2; bb++)
#pragma unroll
            for (int c = 0; c < 2; c++) {
                float r[32], w3 = wt[a] * wt[bb] * wt[c];
                mx_load_row_b<BX>(x, sc, (size_t)nbk * S + ((size_t)mz[a] * H + my[bb]) * W + mx[c], bw, r);
                if (gp.G) {
#pragma unroll
                    for (int k = 0; k < 32; k++) if (k < kmax) r[k] = act_ab(r[k], ca[k], cb[k], true);
                }
#pragma unroll
                for (int k = 0; k < 32; k++) acc[k] += w3 * r[k];
            }
    mx_store_row_b<BY>(y, mx_sc<BY>(y, N, C, So), (size_t)nbk * So + vo, bw, acc);
}
/* the plain upsample (no GN), tiled: a block decodes the coarse region of a 4 x 8 x 32 fine tile (4 x 6 x 18 coarse voxels,
   edge-clamped coordinates) once into smem as bf16 (decoded MX values are exact in bf16) and every fine voxel interpolates from
   it with the weights and order of up2_mx_k (identical output); up2_mx_k decoded each coarse row 8 times from global memory */
template <int BX, int BY>
__global__ void __launch_bounds__(256) up2_mx_tile_k(const uint8_t *x, uint8_t *y, int N, int C, int D, int H, int W, int Cx, int xb0) {   /* C: y channels;
                                                                                    x: Cx channels, its blocks xb0.. (32-wide when Cx != C) */
    /* block = 4 x 4 x 32 fine voxels (thread: x lane, y (tid >> 5) & 3, two z); the coarse rows they read decoded once into an fp32
       tile (exact: the stored values), so the 8 x 32 trilinear terms of a voxel are plain loads + FMAs (no bf16 unpacking) */
    constexpr int CZ = 4, CY = 4, CX = 18, RS = 36;   /* RS: floats per coarse row (+4: neighbouring columns' float4 reads hit different banks) */
    extern __shared__ __align__(16) unsigned char up_smem[];
    float *tile = (float *)up_smem;   /* [CZ][CY][CX][RS] */
    const int bw = mx_bw(C), Do = 2 * D, Ho = 2 * H, Wo = 2 * W;
    const size_t S = (size_t)D * H * W, So = (size_t)Do * Ho * Wo;
    const int nbk = blockIdx.y, ntx = (Wo + 31) / 32, nty = (Ho + 3) / 4, nby = mx_nb(C);
    const size_t xrb = (size_t)((nbk / nby) * mx_nb(Cx) + xb0 + nbk % nby) * S;   /* the source block's first row */
    int bt = blockIdx.x;
    const int ox0 = (bt % ntx) * 32; bt /= ntx;
    const int oy0 = (bt % nty) * 4; bt /= nty;
    const int oz0 = bt * 4;
    const int cz0 = (oz0 >> 1) - 1, cy0 = (oy0 >> 1) - 1, cx0 = (ox0 >> 1) - 1;
    const uint8_t *sc = mx_sc<BX>(x, N, Cx, S);
    for (int i = threadIdx.x; i < CZ * CY * CX; i += 256) {
        const int tx = i % CX, ty = (i / CX) % CY, tz = i / (CX * CY);
        const int z = min(max(cz0 + tz, 0), D - 1), yy = min(max(cy0 + ty, 0), H - 1), xx = min(max(cx0 + tx, 0), W - 1);
        float r[32];
        mx_load_row_b<BX>(x, sc, xrb + ((size_t)z * H + yy) * W + xx, bw, r);
        float4 *d = (float4 *)(tile + (size_t)i * RS);
#pragma unroll
        for (int k = 0; k < 32; k += 4) d[k / 4] = make_float4(k < bw ? r[k] : 0.f, k + 1 < bw ? r[k + 1] : 0.f, k + 2 < bw ? r[k + 2] : 0.f, k + 3 < bw ? r[k + 3] : 0.f);
    }
    __syncthreads();
    const float wt[2] = {0.75f, 0.25f};
    const int ox = ox0 + (threadIdx.x & 31), oy = oy0 + ((threadIdx.x >> 5) & 3), dz0 = (threadIdx.x >> 7) * 2;
    if (ox >= Wo || oy >= Ho) return;
    const int my[2] = {oy >> 1, min(max((oy >> 1) + ((oy & 1) ? 1 : -1), 0), H - 1)}, mx[2] = {ox >> 1, min(max((ox >> 1) + ((ox & 1) ? 1 : -1), 0), W - 1)};
    for (int dz = dz0; dz < dz0 + 2; dz++) {
        const int oz = oz0 + dz;
        if (oz >= Do) break;
        const int mz[2] = {oz >> 1, min(max((oz >> 1) + ((oz & 1) ? 1 : -1), 0), D - 1)};
        float acc[32] = {};
#pragma unroll
        for (int a = 0; a < 2; a++)
#pragma unroll
            for (int bb = 0; bb < 2; bb++)
#pragma unroll
                for (int c = 0; c < 2; c++) {
                    /* tile index of the clamped coarse coordinate: the tile holds clamp(c0 + t) at t, and c0 + t == m is in range */
                    const int tz = mz[a] - cz0, ty = my[bb] - cy0, tx = mx[c] - cx0;
                    const float w3 = wt[a] * wt[bb] * wt[c];
                    const float4 *src = (const float4 *)(tile + ((size_t)(tz * CY + ty) * CX + tx) * RS);
#pragma unroll
                    for (int q = 0; q < 8; q++) {
                        const float4 f = src[q];
                        acc[4 * q] += w3 * f.x; acc[4 * q + 1] += w3 * f.y; acc[4 * q + 2] += w3 * f.z; acc[4 * q + 3] += w3 * f.w;
                    }
                }
        mx_store_row_b<BY>(y, mx_sc<BY>(y, N, C, So), (size_t)nbk * So + ((size_t)oz * Ho + oy) * Wo + ox, bw, acc);
    }
}
/* y += x for two MX tensors of the same format (rows re-encoded; sr != 0: exact stochastic rounding keyed by sr) */
template <int B> __global__ void add_mx_k(uint8_t *y, const uint8_t *x, int N, int C, size_t S, unsigned sr) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * mx_nb(C) * S) return;
    const int bw = mx_bw(C);
    float a[32], b[32];
    mx_load_row_b<B>(y, mx_sc<B>(y, N, C, S), i, bw, a); mx_load_row_b<B>(x, mx_sc<B>(x, N, C, S), i, bw, b);
#pragma unroll
    for (int k = 0; k < 32; k++) a[k] += b[k];
    mx_store_row_b<B>(y, mx_sc<B>(y, N, C, S), i, bw, a, sr);
}
extern "C" void lp_add_mx(void *y, const void *x, shape5 s, int dt, unsigned sr) {
    const size_t S = shape_spatial(s), rows = (size_t)s.n * mx_nb(s.c) * S;
    if (dt == 4) add_mx_k<4><<<nblk_(rows, 256), 256>>>((uint8_t *)y, (const uint8_t *)x, s.n, s.c, S, sr);
    else add_mx_k<8><<<nblk_(rows, 256), 256>>>((uint8_t *)y, (const uint8_t *)x, s.n, s.c, S, sr);
    LPCK();
}
extern "C" void lp_up2_fwd_mx(const void *x, int xdt, shape5 xs, void *y, int ydt, gnp_t gp) {
    const size_t So = 8 * shape_spatial(xs);
    const dim3 grid(nblk_(So, 256), (unsigned)(xs.n * mx_nb(xs.c)));
    const uint8_t *xq = (const uint8_t *)x; uint8_t *yq = (uint8_t *)y;
    static int tiled = -1; if (tiled < 0) tiled = getenv("UFSM_UP2_TILE") ? atoi(getenv("UFSM_UP2_TILE")) : 1;
    if (tiled && !gp.G) {
        const size_t sm = (size_t)4 * 4 * 18 * 36 * 4;
        const dim3 gt((unsigned)(((2 * xs.w + 31) / 32) * ((2 * xs.h + 3) / 4) * ((2 * xs.d + 3) / 4)), (unsigned)(xs.n * mx_nb(xs.c)));
#define U2T(BX, BY) up2_mx_tile_k<BX, BY><<<gt, 256, sm>>>(xq, yq, xs.n, xs.c, xs.d, xs.h, xs.w, xs.c, 0)
        if (xdt == 4) { if (ydt == 4) U2T(4, 4); else U2T(4, 8); }
        else { if (ydt == 4) U2T(8, 4); else U2T(8, 8); }
#undef U2T
        LPCK();
        return;
    }
#define U2(BX, BY) up2_mx_k<BX, BY><<<grid, 256>>>(xq, yq, xs.n, xs.c, xs.d, xs.h, xs.w, gp)
    if (xdt == 4) { if (ydt == 4) U2(4, 4); else U2(4, 8); }
    else { if (ydt == 4) U2(8, 4); else U2(8, 8); }
#undef U2
    LPCK();
}
/* the plain upsample of 32-channel blocks [b0, b0 + nbk) of x (xs.c channels) into a 32 nbk-channel y */
extern "C" void lp_up2_fwd_mx_blocks(const void *x, int xdt, shape5 xs, int b0, int nbk, void *y, int ydt) {
    if (mx_bw(xs.c) != 32 || b0 < 0 || b0 + nbk > mx_nb(xs.c)) { fprintf(stderr, "lp_up2_fwd_mx_blocks: blocks %d + %d of %d channels\n", b0, nbk, xs.c); abort(); }
    const size_t sm = (size_t)4 * 4 * 18 * 36 * 4;
    const dim3 gt((unsigned)(((2 * xs.w + 31) / 32) * ((2 * xs.h + 3) / 4) * ((2 * xs.d + 3) / 4)), (unsigned)(xs.n * nbk));
    const uint8_t *xq = (const uint8_t *)x; uint8_t *yq = (uint8_t *)y;
#define U2T(BX, BY) up2_mx_tile_k<BX, BY><<<gt, 256, sm>>>(xq, yq, xs.n, 32 * nbk, xs.d, xs.h, xs.w, xs.c, b0)
    if (xdt == 4) { if (ydt == 4) U2T(4, 4); else U2T(4, 8); }
    else { if (ydt == 4) U2T(8, 4); else U2T(8, 8); }
#undef U2T
    LPCK();
}
/* in-place silu(gn(.)) of the bw channels of block blk of an MX row (gp.G == 0: none) */
