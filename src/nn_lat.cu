/* Latent-grid networks (train --lat 4, unet_cfg.half_res 3 = pool4): the network runs on 4^3 cells of the full-resolution window and its
   logits reach the full-resolution losses through the pool4 chain of nn_lowres_output (two trilinear 2x steps, fine voxel f of each step
   sampling (f + 0.5) / 2 - 0.5 edge-clamped: weight 0.75 on f >> 1, 0.25 on its neighbour toward f), one z-slab of fine planes at a time,
   so the window's fine logits never exist whole. Per axis the chain is a 4-tap filter of the coarse grid (taps may repeat at the edges);
   nn_lat_up_slab applies it, nn_lat_up_adj adds its exact adjoint of a slab's fine gradient into the coarse gradient (a gather per coarse
   voxel: no atomics, one slab after another). Slabs of whole coarse planes take the separable fast paths below (lat_up4_k, lat_adj_xy_k +
   lat_adj_z_k); any other split the per-voxel kernels (UFSM_LAT_SLOW=1 forces them). */
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

/* --- the fast paths: slabs of whole coarse planes (z0 and nz multiples of 4; the head slabs are 32 planes) ---
   Per axis the chain collapses to 3 taps: fine 4 c + r reads coarse c - 1, c, c + 1 (edge-clamped; the clamps of the mid grid only occur
   where they fold onto the same edge cell) with the weights LW[r]. One thread per coarse cell writes its 4^3 fine voxels of every channel
   from the cell's 3^3 neighbourhood, separably (z, then y, then x): 27 loads and ~7 multiply-adds per fine voxel against 64 + 64 of the
   per-voxel kernel, and each fine row leaves as one 8-byte store, contiguous over the warp. P768, 11 channels: 302 -> 21 ms per pass. The
   sums associate differently from lat_up_k's 64-term chain (fp32 rounding before the fp16 store). */
__device__ static const float LW[4][3] = {{0.375f, 0.625f, 0.f}, {0.1875f, 0.75f, 0.0625f}, {0.0625f, 0.75f, 0.1875f}, {0.f, 0.625f, 0.375f}};
template <typename T>
__global__ void lat_up4_k(const T *lg, int C, int d, int h, int w, int z0, int nz, f16 *y) {
    const int H = 4 * h, W = 4 * w;
    const size_t pl = (size_t)h * w, nc = (size_t)(nz / 4) * pl, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nc) return;
    const int sz = (int)(i / pl), cz = z0 / 4 + sz, cy = (int)(i / w % h), cx = (int)(i % w);
    const int iz[3] = {max(cz - 1, 0), cz, min(cz + 1, d - 1)}, iy[3] = {max(cy - 1, 0), cy, min(cy + 1, h - 1)}, ix[3] = {max(cx - 1, 0), cx, min(cx + 1, w - 1)};
    const size_t cv = (size_t)d * pl, ns = (size_t)nz * H * W;
    for (int c = 0; c < C; c++) {
        const T *l = lg + (size_t)c * cv;
        float v[3][3][3];
#pragma unroll
        for (int a = 0; a < 3; a++)
#pragma unroll
            for (int b = 0; b < 3; b++) {
                const size_t row = ((size_t)iz[a] * h + iy[b]) * w;
#pragma unroll
                for (int e = 0; e < 3; e++) v[a][b][e] = ldv(l, row + ix[e]);
            }
        unsigned long long *yc = (unsigned long long *)(y + (size_t)c * ns);
#pragma unroll
        for (int rz = 0; rz < 4; rz++) {
            float p[3][3];
#pragma unroll
            for (int b = 0; b < 3; b++)
#pragma unroll
                for (int e = 0; e < 3; e++) p[b][e] = LW[rz][0] * v[0][b][e] + LW[rz][1] * v[1][b][e] + LW[rz][2] * v[2][b][e];
#pragma unroll
            for (int ry = 0; ry < 4; ry++) {
                float q[3];
#pragma unroll
                for (int e = 0; e < 3; e++) q[e] = LW[ry][0] * p[0][e] + LW[ry][1] * p[1][e] + LW[ry][2] * p[2][e];
                unsigned long long o = 0;
#pragma unroll
                for (int rx = 0; rx < 4; rx++)
                    o |= (unsigned long long)__half_as_ushort(__float2half_rn(LW[rx][0] * q[0] + LW[rx][1] * q[1] + LW[rx][2] * q[2])) << (16 * rx);
                yc[(((size_t)(4 * sz + rz) * H + 4 * cy + ry) * W) / 4 + cx] = o;
            }
        }
    }
}

/* the adjoint's weights of coarse c along one axis of n cells: fine 4 c - 3 + k, k in [k0, k1) (the fine voxels of cells c - 1, c, c + 1
   whose taps reach c; at the edges the clamped taps fold onto the edge cell) */
__device__ static inline void lat_adj_w(int c, int n, float *wt, int *k0, int *k1) {
    static const float A[10] = {0.0625f, 0.1875f, 0.375f, 0.625f, 0.75f, 0.75f, 0.625f, 0.375f, 0.1875f, 0.0625f};
#pragma unroll
    for (int k = 0; k < 10; k++) wt[k] = A[k];
    if (c == 0) { wt[3] += 0.375f; wt[4] += 0.1875f; wt[5] += 0.0625f; }
    if (c == n - 1) { wt[4] += 0.0625f; wt[5] += 0.1875f; wt[6] += 0.375f; }
    *k0 = c == 0 ? 3 : 0; *k1 = c == n - 1 ? 7 : 10;
}
/* adjoint, stage 1 (y and x): t[c][fz][cy][cx] = the slab's fine gradient plane fz gathered onto the coarse (cy, cx) grid: 100 loads per
   (fine plane, coarse cell) = 6.25 per fine voxel, against the ~16 (and a lat_taps per tap) of lat_adj_k's 16^3 support loop */
__global__ void lat_adj_xy_k(const f16 *g, int C, int h, int w, int nz, float *t) {
    const int H = 4 * h, W = 4 * w;
    const size_t pl = (size_t)h * w, n1 = (size_t)nz * pl, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n1 * C) return;
    const int c = (int)(i / n1); const size_t v = i % n1;
    const int sz = (int)(v / pl), cy = (int)(v / w % h), cx = (int)(v % w);
    float wy[10], wx[10]; int y0, y1, x0, x1;
    lat_adj_w(cy, h, wy, &y0, &y1); lat_adj_w(cx, w, wx, &x0, &x1);
    const f16 *gs = g + ((size_t)c * nz + sz) * H * W;
    float s = 0.f;
    for (int a = y0; a < y1; a++) {
        const f16 *row = gs + (size_t)(4 * cy - 3 + a) * W + (4 * cx - 3);
        float r = 0.f;
        for (int b = x0; b < x1; b++) r += wx[b] * __half2float(row[b]);
        s += wy[a] * r;
    }
    t[i] = s;
}
/* stage 2 (z): coarse planes [cz0, cz1) add their fine planes of the slab */
__global__ void lat_adj_z_k(const float *t, int C, int d, int h, int w, int z0, int nz, int cz0, int cz1, float *gc) {
    const size_t pl = (size_t)h * w, nc = (size_t)(cz1 - cz0) * pl, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nc * C) return;
    const int c = (int)(i / nc); const size_t v = i % nc;
    const int cz = cz0 + (int)(v / pl); const size_t yx = v % pl;
    float wz[10]; int k0, k1; lat_adj_w(cz, d, wz, &k0, &k1);
    const float *tc = t + (size_t)c * nz * pl + yx;
    float s = 0.f;
    for (int k = k0; k < k1; k++) { const int fz = 4 * cz - 3 + k - z0; if (fz >= 0 && fz < nz) s += wz[k] * tc[(size_t)fz * pl]; }
    gc[((size_t)c * d + cz) * pl + yx] += s;
}

/* predict --channel rv on a latent-grid checkpoint: the two side logits of the cell grid (lg [2][d][h][w]) straight into a device shard
   buffer: upsampled per coarse cell as lat_up4_k (fp32, not rounded to fp16), fused as nn_rv_fuse and quantised as nn_pred_place, masked
   by the window's CT (ct >= iso, or the mask bits). Writes window voxels [z0, z1) x [y0, y1) x [x0, x1) (the interior inside the shard)
   at dsh[(o + voxel)]; the window's fp32 logit volume (8 bytes a voxel for the two channels), its fuse and its place pass never exist. */
template <typename T>
__global__ void lat_place_rv_k(const T *lg, int d, int h, int w, const uint8_t *ct, const uint32_t *bits, int iso, int z0, int y0, int x0, int z1, int y1, int x1,
                               int oz, int oy, int ox, int shard, uint8_t *dsh) {
    const int H = 4 * h, W = 4 * w, c0z = z0 >> 2, c0y = y0 >> 2, c0x = x0 >> 2, ncy = ((y1 + 3) >> 2) - c0y, ncx = ((x1 + 3) >> 2) - c0x, ncz = ((z1 + 3) >> 2) - c0z;
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)ncz * ncy * ncx) return;
    const int cz = c0z + (int)(i / ((size_t)ncy * ncx)), cy = c0y + (int)(i / ncx % ncy), cx = c0x + (int)(i % ncx);
    const int iz[3] = {max(cz - 1, 0), cz, min(cz + 1, d - 1)}, iy[3] = {max(cy - 1, 0), cy, min(cy + 1, h - 1)}, ix[3] = {max(cx - 1, 0), cx, min(cx + 1, w - 1)};
    const size_t cv = (size_t)d * h * w;
    float v[2][3][3][3];
#pragma unroll
    for (int c = 0; c < 2; c++)
#pragma unroll
        for (int a = 0; a < 3; a++)
#pragma unroll
            for (int b = 0; b < 3; b++) {
                const size_t row = (size_t)c * cv + ((size_t)iz[a] * h + iy[b]) * w;
#pragma unroll
                for (int e = 0; e < 3; e++) v[c][a][b][e] = ldv(lg, row + ix[e]);
            }
#pragma unroll
    for (int rz = 0; rz < 4; rz++) {
        const int z = 4 * cz + rz;
        if (z < z0 || z >= z1) continue;
        float p[2][3][3];
#pragma unroll
        for (int c = 0; c < 2; c++)
#pragma unroll
            for (int b = 0; b < 3; b++)
#pragma unroll
                for (int e = 0; e < 3; e++) p[c][b][e] = LW[rz][0] * v[c][0][b][e] + LW[rz][1] * v[c][1][b][e] + LW[rz][2] * v[c][2][b][e];
#pragma unroll
        for (int ry = 0; ry < 4; ry++) {
            const int y = 4 * cy + ry;
            if (y < y0 || y >= y1) continue;
            float q[2][3];
#pragma unroll
            for (int c = 0; c < 2; c++)
#pragma unroll
                for (int e = 0; e < 3; e++) q[c][e] = LW[ry][0] * p[c][0][e] + LW[ry][1] * p[c][1][e] + LW[ry][2] * p[c][2][e];
            const size_t kr = ((size_t)z * H + y) * W, gr = ((size_t)(oz + z) * shard + (oy + y)) * shard + ox;
#pragma unroll
            for (int rx = 0; rx < 4; rx++) {
                const int x = 4 * cx + rx;
                if (x < x0 || x >= x1) continue;
                const size_t k = kr + x;
                uint8_t o = 0;
                if (ct ? ct[k] >= iso : (bits[k >> 5] >> (k & 31)) & 1u) {
                    const float a = LW[rx][0] * q[0][0] + LW[rx][1] * q[0][1] + LW[rx][2] * q[0][2], b = LW[rx][0] * q[1][0] + LW[rx][1] * q[1][1] + LW[rx][2] * q[1][2];
                    const float pr = 1.f / (1.f + __expf(-a)), pv = 1.f / (1.f + __expf(-b));
                    float f = (1.f + 254.f * (0.5f + 0.5f * (pr - pv))) / 255.f; f = fminf(fmaxf(f, 1.5f / 255.f), 1.f - 1e-4f);   /* (nn_rv_fuse) */
                    const float l = __logf(f / (1.f - f));
                    o = (uint8_t)(255.f / (1.f + __expf(-l)) + 0.5f);   /* (nn_pred_place) */
                }
                dsh[gr + x] = o;
            }
        }
    }
}
extern "C" void nn_lat_place_rv(const void *lg, int lh16, int d, int h, int w, const uint8_t *ct, const uint32_t *bits, int iso, int hz, int halo,
                                int oz, int oy, int ox, int ez, int ey, int ex, int shard, uint8_t *dsh) {
    const int D = 4 * d, H = 4 * h, W = 4 * w;
    const int z0 = max(hz, -oz), y0 = max(halo, -oy), x0 = max(halo, -ox), z1 = min(D - hz, ez - oz), y1 = min(H - halo, ey - oy), x1 = min(W - halo, ex - ox);
    if (z1 <= z0 || y1 <= y0 || x1 <= x0) return;
    const size_t n = (size_t)(((z1 + 3) >> 2) - (z0 >> 2)) * (((y1 + 3) >> 2) - (y0 >> 2)) * (((x1 + 3) >> 2) - (x0 >> 2));
    if (lh16) lat_place_rv_k<f16><<<nblk(n, 128), 128>>>((const f16 *)lg, d, h, w, ct, bits, iso, z0, y0, x0, z1, y1, x1, oz, oy, ox, shard, dsh);
    else lat_place_rv_k<float><<<nblk(n, 128), 128>>>((const float *)lg, d, h, w, ct, bits, iso, z0, y0, x0, z1, y1, x1, oz, oy, ox, shard, dsh);
    KCHECK();
}

/* predict on a latent-grid net: the network's input (4 channels on the cell grid, 16-bit storage) straight from the uint8 CT window:
   per cell the 4^3 voxel values of pred_in_k (rounded to the storage type as it stores them), averaged over 2^3 twice as
   nn_lowres_input does (each mean rounded to the storage type, the same summation order): the same values without the window's
   4-channel 16-bit input (8 bytes a voxel) and its half-resolution intermediate (1 byte a voxel) */
template <typename HT>
__global__ void lat_pred_in_k(const uint8_t *ct, int D, int W, float mean, float isd, const float *dyo, const float *dxo, int axis, HT *y) {
    const int d = D / 4, h = W / 4;
    const size_t cv = (size_t)d * h * h, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= cv) return;
    const int cz = (int)(i / ((size_t)h * h)), cy = (int)(i / h % h), cx = (int)(i % h);
    float o[3] = {0.f, 0.f, 0.f};   /* channels 0, 2, 3 (1 is zero) */
#pragma unroll
    for (int k2 = 0; k2 < 8; k2++) {   /* the half-resolution voxels of the cell, (z, y, x) bits as half_in_k */
        float s[3] = {0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const int z = 4 * cz + 2 * (k2 >> 2) + (k >> 2), yy = 4 * cy + 2 * (k2 >> 1 & 1) + (k >> 1 & 1), xx = 4 * cx + 2 * (k2 & 1) + (k & 1);
            const float dy = dyo[z] + (float)yy, dx = dxo[z] + (float)xx, inv = axis ? (float)axis / (sqrtf(dy * dy + dx * dx) + 1e-6f) : 0.f;
            s[0] += h2f<HT>(f2h<HT>(((float)ct[((size_t)z * W + yy) * W + xx] - mean) * isd));
            s[1] += h2f<HT>(f2h<HT>(dy * inv)); s[2] += h2f<HT>(f2h<HT>(dx * inv));
        }
#pragma unroll
        for (int c = 0; c < 3; c++) o[c] += h2f<HT>(f2h<HT>(0.125f * s[c]));
    }
    y[i] = f2h<HT>(0.125f * o[0]); y[cv + i] = f2h<HT>(0.f); y[2 * cv + i] = f2h<HT>(0.125f * o[1]); y[3 * cv + i] = f2h<HT>(0.125f * o[2]);
}
extern "C" void nn_lat_pred_input(const uint8_t *ct, int D, int W, float mean, float isd, const float *dyo, const float *dxo, int axis, void *y) {
    const size_t cv = (size_t)(D / 4) * (W / 4) * (W / 4);
    if (g_h16) lat_pred_in_k<f16><<<nblk(cv, 128), 128>>>(ct, D, W, mean, isd, dyo, dxo, axis, (f16 *)y);
    else lat_pred_in_k<bf16><<<nblk(cv, 128), 128>>>(ct, D, W, mean, isd, dyo, dxo, axis, (bf16 *)y);
    KCHECK();
}

static int g_lat_slow = -1;   /* UFSM_LAT_SLOW=1 / nn_lat_set_slow: the per-voxel kernels for every slab (A/B, train's UFSM_LAT_CHECK) */
static int lat_slow(void) { if (g_lat_slow < 0) { const char *e = getenv("UFSM_LAT_SLOW"); g_lat_slow = e && *e == '1'; } return g_lat_slow; }
extern "C" int nn_lat_set_slow(int on) { const int was = lat_slow(); g_lat_slow = on; return was; }
extern "C" void nn_lat_up_slab(const void *lg, int lh16, int C, int d, int h, int w, int z0, int nz, void *y) {
    const size_t ns = (size_t)nz * 16 * h * w;
    const int slow = lat_slow();
    if (!slow && z0 % 4 == 0 && nz % 4 == 0) {
        const size_t nc = (size_t)(nz / 4) * h * w;
        if (lh16) lat_up4_k<f16><<<nblk(nc, 128), 128>>>((const f16 *)lg, C, d, h, w, z0, nz, (f16 *)y);
        else lat_up4_k<float><<<nblk(nc, 128), 128>>>((const float *)lg, C, d, h, w, z0, nz, (f16 *)y);
        KCHECK(); return;
    }
    if (lh16) lat_up_k<f16><<<nblk(ns, 256), 256>>>((const f16 *)lg, C, d, h, w, z0, nz, (f16 *)y);
    else lat_up_k<float><<<nblk(ns, 256), 256>>>((const float *)lg, C, d, h, w, z0, nz, (f16 *)y);
    KCHECK();
}
extern "C" void nn_lat_up_adj(const void *g, int C, int d, int h, int w, int z0, int nz, float *gc, void *scratch) {
    const int slow = lat_slow();
    if (!slow && scratch && z0 % 4 == 0 && nz % 4 == 0) {   /* two separable gathers through scratch (C nz h w fp32) */
        const size_t n1 = (size_t)C * nz * h * w;
        lat_adj_xy_k<<<nblk(n1, 256), 256>>>((const f16 *)g, C, h, w, nz, (float *)scratch);
        const int a0 = max(z0 / 4 - 1, 0), a1 = min((z0 + nz) / 4 + 1, d);
        lat_adj_z_k<<<nblk((size_t)(a1 - a0) * h * w * C, 256), 256>>>((const float *)scratch, C, d, h, w, z0, nz, a0, a1, gc);
        KCHECK(); return;
    }
    /* coarse planes whose support meets fine planes [z0, z0 + nz) */
    int cz0 = (z0 - 9) / 4; if (z0 - 9 < 0) cz0 = 0;
    int cz1 = (z0 + nz + 6 + 3) / 4 + 1; if (cz1 > d) cz1 = d;
    const size_t n = (size_t)(cz1 - cz0) * h * w * C;
    if (n) lat_adj_k<<<nblk(n, 256), 256>>>((const f16 *)g, C, d, h, w, z0, nz, cz0, cz1, gc);
    KCHECK();
}
