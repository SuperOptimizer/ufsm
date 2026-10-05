/* MX backward of the 2x upsample */
#include "lp_mxops.cuh"
/* ---- MX backward of the exact-2x trilinear upsample: gx[m] = sum over fine outputs o in [2m - 1, 2m + 2] (per axis) of
   coef(o, m) gy[o], coef as in the forward (0.75 for in[o/2], 0.25 for the clamped neighbour) ---- */
__global__ void up2_bwd_mx_k(const uint8_t *gy, uint8_t *gx, int N, int C, int D, int H, int W) {   /* D, H, W: coarse (gx) grid */
    const int bw = mx_bw(C), nb = mx_nb(C), Do = 2 * D, Ho = 2 * H, Wo = 2 * W;
    const size_t S = (size_t)D * H * W, So = (size_t)Do * Ho * Wo;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * nb * S) return;
    size_t v = i % S, nbk = i / S;
    int x = (int)(v % W), y = (int)((v / W) % H), z = (int)(v / ((size_t)W * H));
    const uint8_t *sc = gy + (size_t)N * nb * So * bw;
    float acc[32] = {};
#pragma unroll 1
    for (int a = 0; a < 4; a++) {
        int oz = 2 * z - 1 + a; float wz = upc(oz, z, D); if (wz == 0.f) continue;
#pragma unroll 1
        for (int bb = 0; bb < 4; bb++) {
            int oy = 2 * y - 1 + bb; float wy = upc(oy, y, H); if (wy == 0.f) continue;
#pragma unroll 1
            for (int c = 0; c < 4; c++) {
                int ox = 2 * x - 1 + c; float wx = upc(ox, x, W); if (wx == 0.f) continue;
                float r[32], w3 = wz * wy * wx;
                mx_load_row(gy, sc, nbk * So + ((size_t)oz * Ho + oy) * Wo + ox, bw, r);
#pragma unroll
                for (int k = 0; k < 32; k++) acc[k] += w3 * r[k];
            }
        }
    }
    mx_store_row(gx, gx + (size_t)N * nb * S * bw, i, bw, acc);
}
extern "C" void lp_up2_bwd_mx(const void *gy, shape5 xs, void *gx) {
    size_t n = (size_t)xs.n * mx_nb(xs.c) * shape_spatial(xs);
    up2_bwd_mx_k<<<nblk_(n, 256), 256>>>((const uint8_t *)gy, (uint8_t *)gx, xs.n, xs.c, xs.d, xs.h, xs.w); LPCK();
}
/* channel slice [c0, c0 + nc) of an MX-fp8 gx with ctot channels from an MX-fp8 gy of nc channels (the chunked up-part gradient
   of the decoder): thread = (n, gx block touched by the slice, coarse voxel). A slice that starts at its block's first channel
   writes the block fresh (other channels zero); a later slice of the same block (16-channel chunks of a 32-channel block)
   decodes the stored row, replaces its channels and requantises (one extra e4m3 rounding of the earlier chunk's channels when
   the block scale grows). Chunks must be processed in increasing c0; a slice lies in one gy block (nc <= 16, or 32-aligned). */
__global__ void up2_bwd_mx_slice_k(const uint8_t *gy, uint8_t *gx, int N, int nc, int ctot, int c0, int ob0, int nob, int D, int H, int W) {
    const int bwy = mx_bw(nc), nby = mx_nb(nc), bwx = mx_bw(ctot), nbx = mx_nb(ctot), Do = 2 * D, Ho = 2 * H, Wo = 2 * W;
    const size_t S = (size_t)D * H * W, So = (size_t)Do * Ho * Wo;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * nob * S) return;
    const size_t v = i % S; const int ob = ob0 + (int)((i / S) % nob), n = (int)(i / (S * nob));
    const int lo = max(c0, ob * bwx), hi = min(c0 + nc, min(ob * bwx + bwx, ctot)), yb = (lo - c0) / bwy, ko = (lo - c0) - yb * bwy;
    const int x = (int)(v % W), y = (int)((v / W) % H), z = (int)(v / ((size_t)W * H));
    const uint8_t *scy = mx_sc<8>(gy, N, nc, So);
    float acc[32] = {};
#pragma unroll 1
    for (int a = 0; a < 4; a++) {
        int oz = 2 * z - 1 + a; float wz = upc(oz, z, D); if (wz == 0.f) continue;
#pragma unroll 1
        for (int bb = 0; bb < 4; bb++) {
            int oy = 2 * y - 1 + bb; float wy = upc(oy, y, H); if (wy == 0.f) continue;
#pragma unroll 1
            for (int c = 0; c < 4; c++) {
                int ox = 2 * x - 1 + c; float wx = upc(ox, x, W); if (wx == 0.f) continue;
                float r[32], w3 = wz * wy * wx;
                mx_load_row(gy, scy, ((size_t)n * nby + yb) * So + ((size_t)oz * Ho + oy) * Wo + ox, bwy, r);
#pragma unroll
                for (int k = 0; k < 32; k++) acc[k] += w3 * r[k];
            }
        }
    }
    const size_t ri = ((size_t)n * nbx + ob) * S + v;
    uint8_t *scx = mx_sc<8>(gx, N, ctot, S);
    float out[32];
    if (lo == ob * bwx) {
#pragma unroll
        for (int k = 0; k < 32; k++) out[k] = 0.f;
    } else mx_load_row(gx, scx, ri, bwx, out);
#pragma unroll
    for (int k = 0; k < 32; k++) { const int cch = ob * bwx + k; if (cch >= lo && cch < hi) { const int kk = cch - lo + ko; float val = 0.f;
#pragma unroll
        for (int q = 0; q < 32; q++) if (q == kk) val = acc[q];
        out[k] = val; } }
    mx_store_row(gx, scx, ri, bwx, out);
}
extern "C" void lp_up2_bwd_mx_slice(const void *gy, shape5 xs, void *gx, int ctot, int c0) {   /* xs: coarse shape with nc = xs.c channels */
    const int nc = xs.c, bwx = mx_bw(ctot), bwy = mx_bw(nc);
    if (c0 % 16 || (nc > 16 && (c0 % 32 || (nc % 32 && c0 + nc != ctot))) || (c0 % bwx + nc > bwx && nc <= 16)) { fprintf(stderr, "lp_up2_bwd_mx_slice: unaligned slice c0 %d nc %d of %d\n", c0, nc, ctot); abort(); }
    (void)bwy;
    const int ob0 = c0 / bwx, ob1 = (c0 + nc - 1) / bwx, nob = ob1 - ob0 + 1;
    if (nob > 1 && nc <= 16) { fprintf(stderr, "lp_up2_bwd_mx_slice: slice spans blocks\n"); abort(); }
    if (nob > 1) {   /* 32-aligned multi-block slice: one gy block per gx block, launch per block */
        for (int ob = ob0; ob <= ob1; ob++) {
            size_t n = (size_t)xs.n * shape_spatial(xs);
            up2_bwd_mx_slice_k<<<nblk_(n, 256), 256>>>((const uint8_t *)gy, (uint8_t *)gx, xs.n, nc, ctot, c0, ob, 1, xs.d, xs.h, xs.w);
        }
    } else {
        size_t n = (size_t)xs.n * shape_spatial(xs);
        up2_bwd_mx_slice_k<<<nblk_(n, 256), 256>>>((const uint8_t *)gy, (uint8_t *)gx, xs.n, nc, ctot, c0, ob0, 1, xs.d, xs.h, xs.w);
    }
    LPCK();
}
