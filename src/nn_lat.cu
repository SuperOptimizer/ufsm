/* Latent-grid networks (train --lat 4, unet_cfg.half_res 3 = pool4): the network runs on 4^3 cells of the full-resolution window and its
   logits reach the full-resolution losses through the pool4 chain of nn_lowres_output (two trilinear 2x steps, fine voxel f of each step
   sampling (f + 0.5) / 2 - 0.5 edge-clamped: weight 0.75 on f >> 1, 0.25 on its neighbour toward f), one z-slab of fine planes at a time,
   so the window's fine logits never exist whole. Per axis the chain is a 4-tap filter of the coarse grid (taps may repeat at the edges);
   nn_lat_up_slab applies it, nn_lat_up_adj adds its exact adjoint of a slab's fine gradient into the coarse gradient (a gather per coarse
   voxel: no atomics, one slab after another). */
#include "nn_common.cuh"

/* the 4 coarse taps of fine index f (fine axis of 4 n voxels, coarse axis of n): mid m of the first 2x step, then coarse of the second */
__host__ __device__ static inline void lat_taps(int f, int n, int *idx, float *w) {
    const int nm = 2 * n;
    const int m0 = f >> 1, m1 = f & 1 ? (m0 + 1 < nm ? m0 + 1 : nm - 1) : (m0 - 1 > 0 ? m0 - 1 : 0);
    const int mm[2] = {m0, m1}; const float wm[2] = {0.75f, 0.25f};
    for (int a = 0; a < 2; a++) {
        const int m = mm[a], c0 = m >> 1, c1 = m & 1 ? (c0 + 1 < n ? c0 + 1 : n - 1) : (c0 - 1 > 0 ? c0 - 1 : 0);
        idx[2 * a] = c0; w[2 * a] = wm[a] * 0.75f;
        idx[2 * a + 1] = c1; w[2 * a + 1] = wm[a] * 0.25f;
    }
}

template <typename T>
__global__ void lat_up_k(const T *lg, int C, int d, int h, int w, int z0, int nz, f16 *y) {   /* one thread per fine voxel of the slab, all channels */
    const int H = 4 * h, W = 4 * w;
    const size_t ns = (size_t)nz * H * W, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= ns) return;
    const int fz = z0 + (int)(i / ((size_t)H * W)), fy = (int)(i / W % H), fx = (int)(i % W);
    int iz[4], iy[4], ix[4]; float wz[4], wy[4], wx[4];
    lat_taps(fz, d, iz, wz); lat_taps(fy, h, iy, wy); lat_taps(fx, w, ix, wx);
    const size_t cv = (size_t)d * h * w;
    for (int c = 0; c < C; c++) {
        const T *l = lg + (size_t)c * cv;
        float s = 0.f;
#pragma unroll
        for (int a = 0; a < 4; a++)
#pragma unroll
            for (int b = 0; b < 4; b++) {
                const size_t row = ((size_t)iz[a] * h + iy[b]) * w;
                const float wab = wz[a] * wy[b];
#pragma unroll
                for (int e = 0; e < 4; e++) s += wab * wx[e] * ldv(l, row + ix[e]);
            }
        y[(size_t)c * ns + i] = __float2half_rn(s);
    }
}

/* the weight of coarse c in fine f's taps (0 when c is not among them) */
__device__ static inline float lat_wt(int f, int c, int n) {
    int idx[4]; float w[4]; lat_taps(f, n, idx, w);
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < 4; k++) if (idx[k] == c) s += w[k];
    return s;
}

/* coarse voxels of planes [cz0, cz1) gather the slab's fine gradient (fine planes [z0, z0 + nz)) of their support: fine indices
   4 c - 6 .. 4 c + 9 along each axis (taps reach at most 1.5 coarse cells, plus the edge clamps, which only fold onto the edge cells) */
__global__ void lat_adj_k(const f16 *g, int C, int d, int h, int w, int z0, int nz, int cz0, int cz1, float *gc) {
    const int H = 4 * h, W = 4 * w;
    const size_t pl = (size_t)h * w, nc = (size_t)(cz1 - cz0) * pl, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nc * C) return;
    const int c = (int)(i / nc); const size_t v = i % nc;
    const int cz = cz0 + (int)(v / pl), cy = (int)(v / w % h), cx = (int)(v % w);
    const int fz0 = max(z0, cz == 0 ? 0 : 4 * cz - 6), fz1 = min(z0 + nz, cz == d - 1 ? 4 * d : 4 * cz + 10);
    const int fy0 = cy == 0 ? 0 : max(4 * cy - 6, 0), fy1 = cy == h - 1 ? H : min(4 * cy + 10, H);
    const int fx0 = cx == 0 ? 0 : max(4 * cx - 6, 0), fx1 = cx == w - 1 ? W : min(4 * cx + 10, W);
    if (fz0 >= fz1) return;
    const size_t ns = (size_t)nz * H * W;
    const f16 *gs = g + (size_t)c * ns;
    float wxs[16]; int nx = 0;
    for (int fx = fx0; fx < fx1 && nx < 16; fx++) wxs[nx++] = lat_wt(fx, cx, w);
    float s = 0.f;
    for (int fz = fz0; fz < fz1; fz++) {
        const float wzv = lat_wt(fz, cz, d); if (wzv == 0.f) continue;
        for (int fy = fy0; fy < fy1; fy++) {
            const float wyz = wzv * lat_wt(fy, cy, h); if (wyz == 0.f) continue;
            const f16 *row = gs + ((size_t)(fz - z0) * H + fy) * W;
            for (int k = 0; k < nx; k++) if (wxs[k] != 0.f) s += wyz * wxs[k] * __half2float(row[fx0 + k]);
        }
    }
    gc[(size_t)c * d * pl + (size_t)cz * pl + (size_t)cy * w + cx] += s;
}

extern "C" void nn_lat_up_slab(const void *lg, int lh16, int C, int d, int h, int w, int z0, int nz, void *y) {
    const size_t ns = (size_t)nz * 16 * h * w;
    if (lh16) lat_up_k<f16><<<nblk(ns, 256), 256>>>((const f16 *)lg, C, d, h, w, z0, nz, (f16 *)y);
    else lat_up_k<float><<<nblk(ns, 256), 256>>>((const float *)lg, C, d, h, w, z0, nz, (f16 *)y);
    KCHECK();
}
extern "C" void nn_lat_up_adj(const void *g, int C, int d, int h, int w, int z0, int nz, float *gc) {
    /* coarse planes whose support meets fine planes [z0, z0 + nz) */
    int cz0 = (z0 - 9) / 4; if (z0 - 9 < 0) cz0 = 0;
    int cz1 = (z0 + nz + 6 + 3) / 4 + 1; if (cz1 > d) cz1 = d;
    const size_t n = (size_t)(cz1 - cz0) * h * w * C;
    if (n) lat_adj_k<<<nblk(n, 256), 256>>>((const f16 *)g, C, d, h, w, z0, nz, cz0, cz1, gc);
    KCHECK();
}
