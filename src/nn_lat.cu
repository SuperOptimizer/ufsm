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
