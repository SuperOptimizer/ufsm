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
    const int bw = mx_bw(C), nb = mx_nb(C), Do = 2 * D, Ho = 2 * H, Wo = 2 * W;
    const size_t S = (size_t)D * H * W, So = (size_t)Do * Ho * Wo;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * nb * So) return;
    size_t vo = i % So; size_t nbk = i / So;
    int ox = (int)(vo % Wo), oy = (int)((vo / Wo) % Ho), oz = (int)(vo / ((size_t)Wo * Ho));
    int mz[2] = {oz >> 1, min(max((oz >> 1) + ((oz & 1) ? 1 : -1), 0), D - 1)}, my[2] = {oy >> 1, min(max((oy >> 1) + ((oy & 1) ? 1 : -1), 0), H - 1)}, mx[2] = {ox >> 1, min(max((ox >> 1) + ((ox & 1) ? 1 : -1), 0), W - 1)};
    const float wt[2] = {0.75f, 0.25f};
    const uint8_t *sc = mx_sc<BX>(x, N, C, S);
    float acc[32] = {};
#pragma unroll
    for (int a = 0; a < 2; a++)
#pragma unroll
        for (int bb = 0; bb < 2; bb++)
#pragma unroll
            for (int c = 0; c < 2; c++) {
                float r[32], w3 = wt[a] * wt[bb] * wt[c];
                mx_load_row_b<BX>(x, sc, nbk * S + ((size_t)mz[a] * H + my[bb]) * W + mx[c], bw, r);
                row_gn(r, gp, (int)(nbk / nb), (int)(nbk % nb), bw, C);
#pragma unroll
                for (int k = 0; k < 32; k++) acc[k] += w3 * r[k];
            }
    mx_store_row_b<BY>(y, mx_sc<BY>(y, N, C, So), i, bw, acc);
}
extern "C" void lp_up2_fwd_mx(const void *x, int xdt, shape5 xs, void *y, int ydt, gnp_t gp) {
    size_t n = (size_t)xs.n * mx_nb(xs.c) * 8 * shape_spatial(xs);
    const uint8_t *xq = (const uint8_t *)x; uint8_t *yq = (uint8_t *)y;
#define U2(BX, BY) up2_mx_k<BX, BY><<<nblk_(n, 256), 256>>>(xq, yq, xs.n, xs.c, xs.d, xs.h, xs.w, gp)
    if (xdt == 4) { if (ydt == 4) U2(4, 4); else U2(4, 8); }
    else { if (ydt == 4) U2(8, 4); else U2(8, 8); }
#undef U2
    LPCK();
}
/* in-place silu(gn(.)) of the bw channels of block blk of an MX row (gp.G == 0: none) */
