/* CUDA ops: every head of a latent-grid net (unet_cfg.half_res 3, train --lat 4) placed straight from the cell logits into a device shard
   buffer (predict's multi-head path, 2026-11-11). One thread per cell of the window interior inside the shard: the cell's 3^3 neighbourhood
   of the head's channel(s) upsampled to its 4^3 voxels (lat_up4_k's separable LW weights, each value rounded to fp16 as nn_lat_up_slab
   stores it), the sub-cell term of the phase pair (unet_cfg.lat_guide 64: lat_shuf_k), the fine voxel stem (unet_cfg.fstem: nn_fstem_fwd,
   4^3 CT taps at stride 2 per 2^3 block, C silu features, the projection row of the channel) each added in fp16 as those kernels do, then
   the byte: a probability (nn_pred_place) or the recto / verso fusion of a pair (nn_rv_fuse in fp16 + nn_pred_place), 0 where CT < iso.
   = nn_lat_up_slab + nn_lat_shuffle + nn_fstem_fwd + (nn_rv_fuse) + nn_pred_place without the full-resolution input window (8 B a voxel)
   or logit planes; the stem reads the window's CT bytes (z-scored and rounded to fp16 as nn_pred_input_d / nn_lat_ct_x store them, 0
   outside the window). The stem's 64 -> C product runs in fp32 over the fp16 taps and weights (the tensor-core kernel's fp32 sums in
   another order: its logits may differ in the last fp16 bit, rarely a byte). */
#include "nn_common.cuh"

__device__ static const float LH_LW[4][3] = {{0.375f, 0.625f, 0.f}, {0.1875f, 0.75f, 0.0625f}, {0.0625f, 0.75f, 0.1875f}, {0.f, 0.625f, 0.375f}};
enum { LH_MAXC = 16, LH_MAXO = 16, LH_BT = 128 };

struct lh_args {
    int d, h, w, c0, c1, sb0, sb1, s0, s1, C, nout, iso;   /* c1 < 0: a probability of c0; sbN / sN < 0: no sub-cell / stem term */
    float mean, isd;
    int z0, y0, x0, z1, y1, x1, oz, oy, ox, shard;
};
__device__ inline float lh_h(float v) { return __half2float(__float2half_rn(v)); }

template <typename T>
__global__ void __launch_bounds__(LH_BT) lat_head_k(const T *lg, const uint8_t *ct, const float *sp_g, lh_args A, uint8_t *dsh) {
    __shared__ float W1[LH_MAXC * 64], b1[LH_MAXC], W2[2][LH_MAXC];
    const int C = A.C;
    if (C) {   /* the stem's parameters: W1 as the tensor-core kernel takes it (fp16), b1, the projection rows of the head's channels */
        for (int k = threadIdx.x; k < C * 64; k += blockDim.x) W1[k] = __half2float(__float2half_rn(sp_g[k]));
        for (int k = threadIdx.x; k < C; k += blockDim.x) {
            b1[k] = sp_g[C * 64 + k];
            W2[0][k] = A.s0 >= 0 ? sp_g[C * 64 + C + A.s0 * C + k] : 0.f;
            W2[1][k] = A.s1 >= 0 ? sp_g[C * 64 + C + A.s1 * C + k] : 0.f;
        }
        __syncthreads();
    }
    const int d = A.d, h = A.h, w = A.w, D = 4 * d, H = 4 * h, W = 4 * w;
    const int c0z = A.z0 >> 2, c0y = A.y0 >> 2, c0x = A.x0 >> 2, ncz = ((A.z1 + 3) >> 2) - c0z, ncy = ((A.y1 + 3) >> 2) - c0y, ncx = ((A.x1 + 3) >> 2) - c0x;
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)ncz * ncy * ncx) return;
    const int cz = c0z + (int)(i / ((size_t)ncy * ncx)), cy = c0y + (int)(i / ncx % ncy), cx = c0x + (int)(i % ncx);
    const int iz[3] = {max(cz - 1, 0), cz, min(cz + 1, d - 1)}, iy[3] = {max(cy - 1, 0), cy, min(cy + 1, h - 1)}, ix[3] = {max(cx - 1, 0), cx, min(cx + 1, w - 1)};
    const size_t cv = (size_t)d * h * w, cell = ((size_t)cz * h + cy) * w + cx;
    const int np = A.c1 >= 0 ? 2 : 1, ch[2] = {A.c0, A.c1};
    float v[2][3][3][3];
#pragma unroll
    for (int c = 0; c < 2; c++) {
        if (c >= np) break;
#pragma unroll
        for (int a = 0; a < 3; a++)
#pragma unroll
            for (int b = 0; b < 3; b++) {
                const size_t row = (size_t)ch[c] * cv + ((size_t)iz[a] * h + iy[b]) * w;
#pragma unroll
                for (int e = 0; e < 3; e++) v[c][a][b][e] = ldv(lg, row + ix[e]);
            }
    }
    float st[2][8];   /* the stem's term of each channel on the cell's 2^3 blocks (bz, by, bx) = (2 cz + a, 2 cy + b, 2 cx + e) */
#pragma unroll
    for (int c = 0; c < 2; c++)
#pragma unroll
        for (int k = 0; k < 8; k++) st[c][k] = 0.f;
    if (C) {
        const size_t hw = (size_t)H * W;
        for (int k = 0; k < 8; k++) {
            const int bz = 2 * cz + (k >> 2), by = 2 * cy + (k >> 1 & 1), bx = 2 * cx + (k & 1);
            float t[64];
#pragma unroll
            for (int r = 0; r < 16; r++) {
                const int z = 2 * bz - 1 + r / 4, y = 2 * by - 1 + r % 4; const bool rin = z >= 0 && z < D && y >= 0 && y < H;
                const uint8_t *row = ct + (size_t)z * hw + (size_t)y * W;
#pragma unroll
                for (int q = 0; q < 4; q++) { const int x = 2 * bx - 1 + q; t[4 * r + q] = rin && x >= 0 && x < W ? lh_h(((float)row[x] - A.mean) * A.isd) : 0.f; }
            }
            float s0 = 0.f, s1 = 0.f;
            for (int c = 0; c < C; c++) {
                const float *wr = W1 + c * 64; float p = 0.f;
#pragma unroll
                for (int q = 0; q < 64; q++) p += wr[q] * t[q];
                p += b1[c]; const float hc = p / (1.f + __expf(-p));
                s0 += W2[0][c] * hc; s1 += W2[1][c] * hc;
            }
            st[0][k] = s0; st[1][k] = s1;
        }
    }
    const T *sub[2] = {A.sb0 >= 0 ? lg + (size_t)A.sb0 * cv + cell : nullptr, A.sb1 >= 0 ? lg + (size_t)A.sb1 * cv + cell : nullptr};
    const int st_on[2] = {C && A.s0 >= 0, C && A.s1 >= 0};
#pragma unroll
    for (int rz = 0; rz < 4; rz++) {
        const int z = 4 * cz + rz;
        if (z < A.z0 || z >= A.z1) continue;
        float p[2][3][3];
#pragma unroll
        for (int c = 0; c < 2; c++)
#pragma unroll
            for (int b = 0; b < 3; b++)
#pragma unroll
                for (int e = 0; e < 3; e++) p[c][b][e] = LH_LW[rz][0] * v[c][0][b][e] + LH_LW[rz][1] * v[c][1][b][e] + LH_LW[rz][2] * v[c][2][b][e];
#pragma unroll
        for (int ry = 0; ry < 4; ry++) {
            const int y = 4 * cy + ry;
            if (y < A.y0 || y >= A.y1) continue;
            float q[2][3];
#pragma unroll
            for (int c = 0; c < 2; c++)
#pragma unroll
                for (int e = 0; e < 3; e++) q[c][e] = LH_LW[ry][0] * p[c][0][e] + LH_LW[ry][1] * p[c][1][e] + LH_LW[ry][2] * p[c][2][e];
            const size_t kr = ((size_t)z * H + y) * W, gr = ((size_t)(A.oz + z) * A.shard + (A.oy + y)) * A.shard + A.ox;
#pragma unroll
            for (int rx = 0; rx < 4; rx++) {
                const int x = 4 * cx + rx;
                if (x < A.x0 || x >= A.x1) continue;
                uint8_t o = 0;
                if (ct[kr + x] >= A.iso) {
                    const int s = 16 * rz + 4 * ry + rx, blk = 4 * (rz >> 1) + 2 * (ry >> 1) + (rx >> 1);
                    float l[2];
#pragma unroll
                    for (int c = 0; c < 2; c++) {
                        if (c >= np) break;
                        float a = lh_h(LH_LW[rx][0] * q[c][0] + LH_LW[rx][1] * q[c][1] + LH_LW[rx][2] * q[c][2]);
                        if (sub[c]) a = lh_h(a + ldv(sub[c], (size_t)s * cv));
                        if (st_on[c]) a = lh_h(a + st[c][blk]);
                        l[c] = a;
                    }
                    float lo = l[0];
                    if (np == 2) {   /* nn_rv_fuse (fp16 logits), then nn_pred_place */
                        const float pr = 1.f / (1.f + __expf(-l[0])), pv = 1.f / (1.f + __expf(-l[1]));
                        float f = (1.f + 254.f * (0.5f + 0.5f * (pr - pv))) / 255.f; f = fminf(fmaxf(f, 1.5f / 255.f), 1.f - 1e-4f);
                        lo = lh_h(__logf(f / (1.f - f)));
                    }
                    o = (uint8_t)(255.f / (1.f + __expf(-lo)) + 0.5f);
                }
                dsh[gr + x] = o;
            }
        }
    }
}
/* the head (c0, c1, sb0, sb1, s0, s1 as lh_args) over the interior of a D x W x W window ([hz, D - hz) x [halo, W - halo)^2) that falls in
   the shard [0, e) at window offset o (window voxel v -> shard voxel o + v); stem: the stem's parameters (W1 [C][64], b1 [C], W2 [nout][C])
   or null; ct: the window's CT bytes (mask and stem taps) */
extern "C" void nn_lat_head_place(const void *lg, int lh16, int d, int h, int w, int c0, int c1, int sb0, int sb1, int s0, int s1,
                                  const float *stem, int C, int nout, const uint8_t *ct, float mean, float isd, int iso,
                                  int hz, int halo, int oz, int oy, int ox, int ez, int ey, int ex, int shard, uint8_t *dsh) {
    const int D = 4 * d, H = 4 * h, W = 4 * w; (void)H;
    lh_args A = {d, h, w, c0, c1, sb0, sb1, s0, s1, stem ? C : 0, nout, iso < 1 ? 1 : iso, mean, isd,
                 max(hz, -oz), max(halo, -oy), max(halo, -ox), min(D - hz, ez - oz), min(4 * h - halo, ey - oy), min(W - halo, ex - ox), oz, oy, ox, shard};
    if (A.z1 <= A.z0 || A.y1 <= A.y0 || A.x1 <= A.x0) return;
    if (A.C > LH_MAXC || (A.C && (s0 >= nout || s1 >= nout))) { fprintf(stderr, "nn_lat_head_place: stem %d / %d outside the kernel's range\n", C, nout); abort(); }
    const size_t n = (size_t)(((A.z1 + 3) >> 2) - (A.z0 >> 2)) * (((A.y1 + 3) >> 2) - (A.y0 >> 2)) * (((A.x1 + 3) >> 2) - (A.x0 >> 2));
    if (lh16) lat_head_k<f16><<<nblk(n, LH_BT), LH_BT>>>((const f16 *)lg, ct, stem, A, dsh);
    else lat_head_k<float><<<nblk(n, LH_BT), LH_BT>>>((const float *)lg, ct, stem, A, dsh);
    KCHECK();
}
