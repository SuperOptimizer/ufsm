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
__global__ void lat_up_k(const T *lg, int C, int d, int h, int w, int z0, int nz, f16 *y, int q0, int qd) {   /* one thread per fine voxel of the slab, all channels; lg holds cell planes [q0, q0 + qd) */
    const int H = 4 * h, W = 4 * w;
    const size_t ns = (size_t)nz * H * W, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= ns) return;
    const int fz = z0 + (int)(i / ((size_t)H * W)), fy = (int)(i / W % H), fx = (int)(i % W);
    int iz[4], iy[4], ix[4]; float wz[4], wy[4], wx[4];
    lat_taps(fz, d, iz, wz); lat_taps(fy, h, iy, wy); lat_taps(fx, w, ix, wx);
    const size_t cv = (size_t)qd * h * w;
    for (int c = 0; c < C; c++) {
        const T *l = lg + (size_t)c * cv;
        float s = 0.f;
#pragma unroll
        for (int a = 0; a < 4; a++)
#pragma unroll
            for (int b = 0; b < 4; b++) {
                const size_t row = ((size_t)(iz[a] - q0) * h + iy[b]) * w;
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
__global__ void lat_adj_k(const f16 *g, int C, int d, int h, int w, int z0, int nz, int cz0, int cz1, float *gc, int q0, int qd) {
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
    gc[(size_t)c * qd * pl + (size_t)(cz - q0) * pl + (size_t)cy * w + cx] += s;
}

/* predict on a --lat-basis checkpoint: input_latc_k's channels from the host-path 4-channel fp16 window */
__global__ void lat_input_x_k(const f16 *x, int d, int h, int w, const float *bas, const float *gain, int C, float step, f16 *y) {
    const size_t cv = (size_t)d * h * w, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= cv) return;
    const int z = (int)(i / ((size_t)h * w)), yy = (int)(i / w % h), xx = (int)(i % w), H = 4 * h, W = 4 * w; const size_t fv = (size_t)64 * cv;
    float acc[3] = {0.f, 0.f, 0.f}, v[64];
    for (int k = 0; k < 64; k++) {
        const size_t j = ((size_t)(4 * z + (k >> 4)) * H + 4 * yy + (k >> 2 & 3)) * W + 4 * xx + (k & 3);
        v[k] = __half2float(x[j]); for (int c = 0; c < 3; c++) acc[c] += __half2float(x[(size_t)(1 + c) * fv + j]);
    }
    for (int c = 0; c < C; c++) {
        float s = 0.f; for (int k = 0; k < 64; k++) s += bas[c * 64 + k] * v[k];
        if (step > 0.f) { const float lim = c ? 127.f : 32767.f; s = fminf(lim, fmaxf(-lim, rintf(s / step))) * step; }
        y[(size_t)c * cv + i] = __float2half_rn(s * gain[c]);
    }
    for (int c = 0; c < 3; c++) y[(size_t)(C + c) * cv + i] = __float2half_rn(acc[c] * (1.f / 64.f));
}
__global__ void lat_input_q_k(const int8_t *q, int qs, const f16 *x, int d, int h, int w, const float *gain, int C, float sraw, float mean, float isd, f16 *y) {
    const size_t cv = (size_t)d * h * w, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= cv) return;
    const int z = (int)(i / ((size_t)h * w)), yy = (int)(i / w % h), xx = (int)(i % w), H = 4 * h, W = 4 * w; const size_t fv = (size_t)64 * cv;
    const int8_t *qc = q + i * (size_t)qs;
    const float dc = (float)((int)(uint8_t)qc[0] + 256 * (int)qc[1]) * sraw;
    y[i] = __float2half_rn((dc - 8.f * mean) * isd * gain[0]);
    for (int c = 1; c < C; c++) y[(size_t)c * cv + i] = __float2half_rn((float)qc[c + 1] * sraw * isd * gain[c]);
    float acc[3] = {0.f, 0.f, 0.f};
    for (int k = 0; k < 64; k++) {
        const size_t j = ((size_t)(4 * z + (k >> 4)) * H + 4 * yy + (k >> 2 & 3)) * W + 4 * xx + (k & 3);
        for (int c = 0; c < 3; c++) acc[c] += __half2float(x[(size_t)(1 + c) * fv + j]);
    }
    for (int c = 0; c < 3; c++) y[(size_t)(C + c) * cv + i] = __float2half_rn(acc[c] * (1.f / 64.f));
}
extern "C" void nn_lat_input_qs(const void *q, int stride, const void *x, int d, int h, int w, const float *gain, int C, float step_raw, float mean, float isd, void *y) {
    const size_t cv = (size_t)d * h * w;
    lat_input_q_k<<<nblk(cv, 128), 128>>>((const int8_t *)q, stride, (const f16 *)x, d, h, w, gain, C, step_raw, mean, isd, (f16 *)y); KCHECK();
}
extern "C" void nn_lat_input_q(const void *q, const void *x, int d, int h, int w, const float *gain, int C, float step_raw, float mean, float isd, void *y) {
    nn_lat_input_qs(q, C + 1, x, d, h, w, gain, C, step_raw, mean, isd, y);
}
/* The guided head of latent-grid nets (unet_cfg.lat_guide 5: the MX head's weight gradient takes <= 16 outputs, 11 + 5): the two side
   channels' fine logits are their upsampled coarse logits plus terms in the voxel's z-scored CT value x and its central differences
   grad x (edge-clamped), with 5 more coarse outputs as coefficients: l_0 += b_0 x + B . grad x, l_1 += b_1 x - B . grad x (the two faces
   of a sheet lie on opposite sides of its CT ridge: one direction B, opposite signs). The cell grid cannot place a sheet's faces
   inside a cell; the CT can. One slab [.][nz][H][W] of fp16 planes lg: dir 0 adds the terms to planes c0, c0 + 1 from the
   coefficient planes base .. base + 4 (b_0, b_1, B_z, B_y, B_x); dir 1 (planes c0, c0 + 1 hold the loss gradient) writes the
   coefficients' gradients over those planes. x: the window's CT channel [D][H][W] (fp16). */
__global__ void lat_guide_k(const f16 *x, int D, int H, int W, int z0, int nz, f16 *lg, int base, int c0, int dir) {
    const size_t hw = (size_t)H * W, ns = (size_t)nz * hw, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= ns) return;
    const int z = z0 + (int)(i / hw), y = (int)(i / W % H), xx = (int)(i % W);
    const size_t v = (size_t)z * hw + (size_t)y * W + xx;
    const float f[4] = {__half2float(x[v]),
                        0.5f * (__half2float(x[v + (z + 1 < D ? hw : 0)]) - __half2float(x[v - (z > 0 ? hw : 0)])),
                        0.5f * (__half2float(x[v + (y + 1 < H ? W : 0)]) - __half2float(x[v - (y > 0 ? W : 0)])),
                        0.5f * (__half2float(x[v + (xx + 1 < W ? 1 : 0)]) - __half2float(x[v - (xx > 0 ? 1 : 0)]))};
    f16 *b = lg + (size_t)base * ns + i, *l0 = lg + (size_t)c0 * ns + i, *l1 = l0 + ns;
    if (!dir) {
        const float t = __half2float(b[2 * ns]) * f[1] + __half2float(b[3 * ns]) * f[2] + __half2float(b[4 * ns]) * f[3];
        *l0 = __float2half_rn(__half2float(*l0) + __half2float(b[0]) * f[0] + t);
        *l1 = __float2half_rn(__half2float(*l1) + __half2float(b[ns]) * f[0] - t);
    } else {
        const float g0 = __half2float(*l0), g1 = __half2float(*l1), gd = g0 - g1;
        b[0] = __float2half_rn(g0 * f[0]); b[ns] = __float2half_rn(g1 * f[0]);
        b[2 * ns] = __float2half_rn(gd * f[1]); b[3 * ns] = __float2half_rn(gd * f[2]); b[4 * ns] = __float2half_rn(gd * f[3]);
    }
}
/* the general form (unet_cfg.lat_guide 8; the head has 32 outputs, the coefficient planes are its last 4 ng): each guided channel j
   has its own 4 coefficients, l_j += b_j0 x + b_j(1..3) . grad x (the phase channels of the winding nets have no face antisymmetry).
   dir 1 also zeroes nothing: the caller zeroes the unused planes. */
__global__ void lat_guide4_k(const f16 *x, int D, int H, int W, int z0, int nz, f16 *lg, int base, int c0, int ng, int dir) {
    const size_t hw = (size_t)H * W, ns = (size_t)nz * hw, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= ns) return;
    const int z = z0 + (int)(i / hw), y = (int)(i / W % H), xx = (int)(i % W);
    const size_t v = (size_t)z * hw + (size_t)y * W + xx;
    const float f[4] = {__half2float(x[v]),
                        0.5f * (__half2float(x[v + (z + 1 < D ? hw : 0)]) - __half2float(x[v - (z > 0 ? hw : 0)])),
                        0.5f * (__half2float(x[v + (y + 1 < H ? W : 0)]) - __half2float(x[v - (y > 0 ? W : 0)])),
                        0.5f * (__half2float(x[v + (xx + 1 < W ? 1 : 0)]) - __half2float(x[v - (xx > 0 ? 1 : 0)]))};
    for (int j = 0; j < ng; j++) {
        f16 *b = lg + (size_t)(base + 4 * j) * ns + i, *l = lg + (size_t)(c0 + j) * ns + i;
        if (!dir) { float s = __half2float(*l); for (int k = 0; k < 4; k++) s += __half2float(b[(size_t)k * ns]) * f[k]; *l = __float2half_rn(s); }
        else { const float g = __half2float(*l); for (int k = 0; k < 4; k++) b[(size_t)k * ns] = __float2half_rn(g * f[k]); }
    }
}
/* The sub-cell ("cell-shuffle") head of latent-grid nets (unet_cfg.lat_guide 64; the head has 144 outputs): sheets 4-10 voxels apart
   repeat every 1-2.5 cells, which no interpolation of one value per cell can follow (the plain head found under half the turns of
   PHerc1667, the phase loss of a winding net stayed at ln 2). Each guided channel j (the phase / side pair) therefore has 64 more
   coarse outputs, one per voxel of the cell: l_j(v) = up(coarse_j)(v) + S[base + 64 j + sub(v)](cell(v)), sub = 16 (z & 3) + 4 (y & 3) + (x & 3).
   dir 0 (shuf_k): adds the terms to the slab planes y [ng][nz][H][W] (fp16) from the coarse logits lg [C][d][h][w]; the adjoint
   (shuf_adj_k): the planes hold the loss gradient, written (each coarse element has exactly one voxel: no sum, so straight into the 16-bit
   gradient the backward reads) into gc [C][d][h][w] fp16. */
template <typename T> __global__ void lat_shuf_k(const T *lg, int d, int h, int w, int base, int ng, int z0, int nz, f16 *y, int q0, int qd) {
    const int H = 4 * h, W = 4 * w; const size_t hw = (size_t)H * W, ns = (size_t)nz * hw, nc = (size_t)qd * h * w, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= ns) return;
    const int z = z0 + (int)(i / hw), yy = (int)(i / W % H), x = (int)(i % W);
    const size_t cell = ((size_t)((z >> 2) - q0) * h + (yy >> 2)) * w + (x >> 2); const int sub = 16 * (z & 3) + 4 * (yy & 3) + (x & 3);
    for (int j = 0; j < ng; j++) y[(size_t)j * ns + i] = __float2half_rn(__half2float(y[(size_t)j * ns + i]) + (float)lg[(size_t)(base + 64 * j + sub) * nc + cell]);
}
__global__ void lat_shuf_adj_k(const f16 *g, int d, int h, int w, int base, int ng, int z0, int nz, f16 *gc, int q0, int qd) {
    const int H = 4 * h, W = 4 * w; const size_t hw = (size_t)H * W, ns = (size_t)nz * hw, nc = (size_t)qd * h * w, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= ns) return;
    const int z = z0 + (int)(i / hw), yy = (int)(i / W % H), x = (int)(i % W);
    const size_t cell = ((size_t)((z >> 2) - q0) * h + (yy >> 2)) * w + (x >> 2); const int sub = 16 * (z & 3) + 4 * (yy & 3) + (x & 3);
    for (int j = 0; j < ng; j++) gc[(size_t)(base + 64 * j + sub) * nc + cell] = g[(size_t)j * ns + i];
}
/* _c: lg / gc hold cell planes [q0, q0 + qd) only ([C][qd][h][w]; the planes the slab touches must lie inside) */
extern "C" void nn_lat_shuffle_c(const void *lg, int lh16, int d, int h, int w, int base, int ng, int z0, int nz, void *y, int q0, int qd) {
    const size_t ns = (size_t)nz * 16 * h * w;
    if (lh16) lat_shuf_k<f16><<<nblk(ns, 256), 256>>>((const f16 *)lg, d, h, w, base, ng, z0, nz, (f16 *)y, q0, qd);
    else lat_shuf_k<float><<<nblk(ns, 256), 256>>>((const float *)lg, d, h, w, base, ng, z0, nz, (f16 *)y, q0, qd);
    KCHECK();
}
extern "C" void nn_lat_shuffle(const void *lg, int lh16, int d, int h, int w, int base, int ng, int z0, int nz, void *y) { nn_lat_shuffle_c(lg, lh16, d, h, w, base, ng, z0, nz, y, 0, d); }
extern "C" void nn_lat_shuffle_adj_c(const void *g, int d, int h, int w, int base, int ng, int z0, int nz, void *gc, int q0, int qd) {
    const size_t ns = (size_t)nz * 16 * h * w;
    lat_shuf_adj_k<<<nblk(ns, 256), 256>>>((const f16 *)g, d, h, w, base, ng, z0, nz, (f16 *)gc, q0, qd); KCHECK();
}
extern "C" void nn_lat_shuffle_adj(const void *g, int d, int h, int w, int base, int ng, int z0, int nz, void *gc) { nn_lat_shuffle_adj_c(g, d, h, w, base, ng, z0, nz, gc, 0, d); }
/* the rolling coarse gradient of the streamed latent head (train.c head_slab_pass): a [nf][cap][pl] fp32 sums and sb [C][cap][pl] fp16
   (channels >= nf: the sub-cell outputs' gradient) hold cap cell planes. fin: planes [i0, i0 + np) of every channel as fp16
   gc [C][np][pl] (channels < nf rounded as nn_f32_to_h16 does), what unet_head_bwd_z reads. carry: planes [from, from + n) move to
   [0, n), a's other planes are zeroed (sb's are rewritten whole by the next slab) */
__global__ void lat_gfin_k(const float *a, int nf, const f16 *sb, int C, size_t pl, int cap, int i0, int np, f16 *gc) {
    const size_t n = (size_t)np * pl, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n * C) return;
    const int c = (int)(i / n); const size_t v = i % n, src = ((size_t)c * cap + i0) * pl + v;
    gc[i] = c < nf ? f2h<f16>(a[src]) : sb[src];
}
__global__ void lat_gcarry_k(float *a, int nf, f16 *sb, int C, size_t pl, int cap, int from, int n) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= pl * C) return;
    const int c = (int)(i / pl); const size_t v = i % pl;
    if (c < nf) { float *r = a + (size_t)c * cap * pl + v; for (int k = 0; k < cap; k++) r[(size_t)k * pl] = k < n ? r[(size_t)(from + k) * pl] : 0.f; }
    else { f16 *r = sb + (size_t)c * cap * pl + v; for (int k = 0; k < n; k++) r[(size_t)k * pl] = r[(size_t)(from + k) * pl]; }
}
extern "C" void nn_lat_grad_fin(const float *a, int nf, const void *sb, int C, size_t pl, int cap, int i0, int np, void *gc) {
    lat_gfin_k<<<nblk((size_t)np * pl * C, 256), 256>>>(a, nf, (const f16 *)sb, C, pl, cap, i0, np, (f16 *)gc); KCHECK();
}
extern "C" void nn_lat_grad_carry(float *a, int nf, void *sb, int C, size_t pl, int cap, int from, int n) {
    lat_gcarry_k<<<nblk(pl * C, 256), 256>>>(a, nf, (f16 *)sb, C, pl, cap, from, n); KCHECK();
}
extern "C" void nn_lat_guide4(const void *x, int D, int H, int W, int z0, int nz, void *lg, int base, int c0, int ng, int dir) {
    const size_t ns = (size_t)nz * H * W;
    lat_guide4_k<<<nblk(ns, 256), 256>>>((const f16 *)x, D, H, W, z0, nz, (f16 *)lg, base, c0, ng, dir); KCHECK();
}
extern "C" void nn_lat_guide(const void *x, int D, int H, int W, int z0, int nz, void *lg, int base, int c0, int dir) {
    const size_t ns = (size_t)nz * H * W;
    lat_guide_k<<<nblk(ns, 256), 256>>>((const f16 *)x, D, H, W, z0, nz, (f16 *)lg, base, c0, dir); KCHECK();
}
extern "C" void nn_lat_input_x(const void *x, int d, int h, int w, const float *bas, const float *gain, int C, float step, void *y) {
    const size_t cv = (size_t)d * h * w;
    lat_input_x_k<<<nblk(cv, 128), 128>>>((const f16 *)x, d, h, w, bas, gain, C, step, (f16 *)y); KCHECK();
}
/* --- the fast paths: slabs of whole coarse planes (z0 and nz multiples of 4; the head slabs are 32 planes) ---
   Per axis the chain collapses to 3 taps: fine 4 c + r reads coarse c - 1, c, c + 1 (edge-clamped; the clamps of the mid grid only occur
   where they fold onto the same edge cell) with the weights LW[r]. One thread per coarse cell writes its 4^3 fine voxels of every channel
   from the cell's 3^3 neighbourhood, separably (z, then y, then x): 27 loads and ~7 multiply-adds per fine voxel against 64 + 64 of the
   per-voxel kernel, and each fine row leaves as one 8-byte store, contiguous over the warp. P768, 11 channels: 302 -> 21 ms per pass. The
   sums associate differently from lat_up_k's 64-term chain (fp32 rounding before the fp16 store). */
__device__ static const float LW[4][3] = {{0.375f, 0.625f, 0.f}, {0.1875f, 0.75f, 0.0625f}, {0.0625f, 0.75f, 0.1875f}, {0.f, 0.625f, 0.375f}};
template <typename T>
__global__ void lat_up4_k(const T *lg, int C, int d, int h, int w, int z0, int nz, f16 *y, int q0, int qd) {
    const int H = 4 * h, W = 4 * w;
    const size_t pl = (size_t)h * w, nc = (size_t)(nz / 4) * pl, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nc) return;
    const int sz = (int)(i / pl), cz = z0 / 4 + sz, cy = (int)(i / w % h), cx = (int)(i % w);
    const int iz[3] = {max(cz - 1, 0), cz, min(cz + 1, d - 1)}, iy[3] = {max(cy - 1, 0), cy, min(cy + 1, h - 1)}, ix[3] = {max(cx - 1, 0), cx, min(cx + 1, w - 1)};
    const size_t cv = (size_t)qd * pl, ns = (size_t)nz * H * W;
    for (int c = 0; c < C; c++) {
        const T *l = lg + (size_t)c * cv;
        float v[3][3][3];
#pragma unroll
        for (int a = 0; a < 3; a++)
#pragma unroll
            for (int b = 0; b < 3; b++) {
                const size_t row = ((size_t)(iz[a] - q0) * h + iy[b]) * w;
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
__global__ void lat_adj_z_k(const float *t, int C, int d, int h, int w, int z0, int nz, int cz0, int cz1, float *gc, int q0, int qd) {
    const size_t pl = (size_t)h * w, nc = (size_t)(cz1 - cz0) * pl, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nc * C) return;
    const int c = (int)(i / nc); const size_t v = i % nc;
    const int cz = cz0 + (int)(v / pl); const size_t yx = v % pl;
    float wz[10]; int k0, k1; lat_adj_w(cz, d, wz, &k0, &k1);
    const float *tc = t + (size_t)c * nz * pl + yx;
    float s = 0.f;
    for (int k = k0; k < k1; k++) { const int fz = 4 * cz - 3 + k - z0; if (fz >= 0 && fz < nz) s += wz[k] * tc[(size_t)fz * pl]; }
    gc[((size_t)c * qd + (cz - q0)) * pl + yx] += s;
}

/* predict --channel rv on a latent-grid checkpoint: the two side logits of the cell grid (lg [2][d][h][w]) straight into a device shard
   buffer: upsampled per coarse cell as lat_up4_k (fp32, not rounded to fp16), fused as nn_rv_fuse and quantised as nn_pred_place, masked
   by the window's CT (ct >= iso, or the mask bits). Writes window voxels [z0, z1) x [y0, y1) x [x0, x1) (the interior inside the shard)
   at dsh[(o + voxel)]; the window's fp32 logit volume (8 bytes a voxel for the two channels), its fuse and its place pass never exist. */
template <typename T>
__global__ void lat_place_rv_k(const T *lg, int d, int h, int w, const uint8_t *ct, const uint32_t *bits, int iso, int z0, int y0, int x0, int z1, int y1, int x1,
                               int oz, int oy, int ox, int shard, uint8_t *dsh, int sb) {
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
                    float a = LW[rx][0] * q[0][0] + LW[rx][1] * q[0][1] + LW[rx][2] * q[0][2], b = LW[rx][0] * q[1][0] + LW[rx][1] * q[1][1] + LW[rx][2] * q[1][2];
                    if (sb >= 0) {   /* the sub-cell head (lat_shuf_k): the pair's per-voxel outputs, with the slab path's fp16 roundings */
                        const size_t cell = (size_t)(sb + 16 * rz + 4 * ry + rx) * cv + ((size_t)cz * h + cy) * w + cx;
                        a = __half2float(__float2half_rn(__half2float(__float2half_rn(a)) + ldv(lg, cell)));
                        b = __half2float(__float2half_rn(__half2float(__float2half_rn(b)) + ldv(lg, cell + 64 * cv)));
                    }
                    const float pr = 1.f / (1.f + __expf(-a)), pv = 1.f / (1.f + __expf(-b));
                    float f = (1.f + 254.f * (0.5f + 0.5f * (pr - pv))) / 255.f; f = fminf(fmaxf(f, 1.5f / 255.f), 1.f - 1e-4f);   /* (nn_rv_fuse) */
                    float l = __logf(f / (1.f - f));
                    if (sb >= 0) l = __half2float(__float2half(l));   /* (fused in fp16) */
                    o = (uint8_t)(255.f / (1.f + __expf(-l)) + 0.5f);   /* (nn_pred_place) */
                }
                dsh[gr + x] = o;
            }
        }
    }
}
extern "C" void nn_lat_place_rvs(const void *lg, int lh16, int sb, int d, int h, int w, const uint8_t *ct, const uint32_t *bits, int iso, int hz, int halo,
                                 int oz, int oy, int ox, int ez, int ey, int ex, int shard, uint8_t *dsh);
extern "C" void nn_lat_place_rv(const void *lg, int lh16, int d, int h, int w, const uint8_t *ct, const uint32_t *bits, int iso, int hz, int halo,
                                int oz, int oy, int ox, int ez, int ey, int ex, int shard, uint8_t *dsh) {
    nn_lat_place_rvs(lg, lh16, -1, d, h, w, ct, bits, iso, hz, halo, oz, oy, ox, ez, ey, ex, shard, dsh);
}
/* sb >= 0: the sub-cell head (unet_cfg.lat_guide 64), its side pair's per-voxel outputs at cell channels sb + 64 j + sub
   = nn_lat_up_slab + nn_lat_shuffle + nn_rv_fuse (fp16) + nn_pred_place */
extern "C" void nn_lat_place_rvs(const void *lg, int lh16, int sb, int d, int h, int w, const uint8_t *ct, const uint32_t *bits, int iso, int hz, int halo,
                                 int oz, int oy, int ox, int ez, int ey, int ex, int shard, uint8_t *dsh) {
    const int D = 4 * d, H = 4 * h, W = 4 * w;
    const int z0 = max(hz, -oz), y0 = max(halo, -oy), x0 = max(halo, -ox), z1 = min(D - hz, ez - oz), y1 = min(H - halo, ey - oy), x1 = min(W - halo, ex - ox);
    if (z1 <= z0 || y1 <= y0 || x1 <= x0) return;
    const size_t n = (size_t)(((z1 + 3) >> 2) - (z0 >> 2)) * (((y1 + 3) >> 2) - (y0 >> 2)) * (((x1 + 3) >> 2) - (x0 >> 2));
    if (lh16) lat_place_rv_k<f16><<<nblk(n, 128), 128>>>((const f16 *)lg, d, h, w, ct, bits, iso, z0, y0, x0, z1, y1, x1, oz, oy, ox, shard, dsh, sb);
    else lat_place_rv_k<float><<<nblk(n, 128), 128>>>((const float *)lg, d, h, w, ct, bits, iso, z0, y0, x0, z1, y1, x1, oz, oy, ox, shard, dsh, sb);
    KCHECK();
}

/* the same for the guided head (lat_guide_k, dir 0): the two side logits and the 5 coefficient planes (cell channels nb .. nb + 4)
   upsampled per voxel, each rounded to fp16 as nn_lat_up_slab stores them, the CT terms from the window's bytes ctg (the z-scored value
   rounded to fp16 as the input stores it, its central differences edge-clamped in the window), the sum and the fused logit rounded to
   fp16 as lat_guide_k / nn_rv_fuse store them: nn_lat_up_slab + nn_lat_guide + nn_rv_fuse + nn_pred_place without the [7][32][W][W]
   slabs, the [2][W][W][W] fp16 logits and the 4-channel full-resolution input */
template <typename T>
__global__ void lat_place_rvg_k(const T *lg, int nb, int d, int h, int w, const uint8_t *ctg, float mean, float isd, const uint8_t *ct, const uint32_t *bits, int iso,
                                int z0, int y0, int x0, int z1, int y1, int x1, int oz, int oy, int ox, int shard, uint8_t *dsh) {
    const int D = 4 * d, H = 4 * h, W = 4 * w, c0z = z0 >> 2, c0y = y0 >> 2, c0x = x0 >> 2, ncy = ((y1 + 3) >> 2) - c0y, ncx = ((x1 + 3) >> 2) - c0x, ncz = ((z1 + 3) >> 2) - c0z;
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)ncz * ncy * ncx) return;
    const int cz = c0z + (int)(i / ((size_t)ncy * ncx)), cy = c0y + (int)(i / ncx % ncy), cx = c0x + (int)(i % ncx);
    const int iz[3] = {max(cz - 1, 0), cz, min(cz + 1, d - 1)}, iy[3] = {max(cy - 1, 0), cy, min(cy + 1, h - 1)}, ix[3] = {max(cx - 1, 0), cx, min(cx + 1, w - 1)};
    const size_t cv = (size_t)d * h * w, hw = (size_t)H * W;
    float v[7][3][3][3];
    for (int c = 0; c < 7; c++)
#pragma unroll
        for (int a = 0; a < 3; a++)
#pragma unroll
            for (int b = 0; b < 3; b++) {
                const size_t row = (size_t)(c < 2 ? c : nb + c - 2) * cv + ((size_t)iz[a] * h + iy[b]) * w;
#pragma unroll
                for (int e = 0; e < 3; e++) v[c][a][b][e] = ldv(lg, row + ix[e]);
            }
#define XH(k_) __half2float(__float2half(((float)ctg[k_] - mean) * isd))
    for (int rz = 0; rz < 4; rz++) {
        const int z = 4 * cz + rz;
        if (z < z0 || z >= z1) continue;
        for (int ry = 0; ry < 4; ry++) {
            const int y = 4 * cy + ry;
            if (y < y0 || y >= y1) continue;
            float q[7][3];
            for (int c = 0; c < 7; c++)
#pragma unroll
                for (int e = 0; e < 3; e++) {
                    float p[3];
#pragma unroll
                    for (int b = 0; b < 3; b++) p[b] = LW[rz][0] * v[c][0][b][e] + LW[rz][1] * v[c][1][b][e] + LW[rz][2] * v[c][2][b][e];
                    q[c][e] = LW[ry][0] * p[0] + LW[ry][1] * p[1] + LW[ry][2] * p[2];
                }
            const size_t kr = ((size_t)z * H + y) * W, gr = ((size_t)(oz + z) * shard + (oy + y)) * shard + ox;
            for (int rx = 0; rx < 4; rx++) {
                const int x = 4 * cx + rx;
                if (x < x0 || x >= x1) continue;
                const size_t k = kr + x;
                uint8_t o = 0;
                if (ct ? ct[k] >= iso : (bits[k >> 5] >> (k & 31)) & 1u) {
                    float u[7];
#pragma unroll
                    for (int c = 0; c < 7; c++) u[c] = __half2float(__float2half_rn(LW[rx][0] * q[c][0] + LW[rx][1] * q[c][1] + LW[rx][2] * q[c][2]));
                    const float f0 = XH(k), f1 = 0.5f * (XH(k + (z + 1 < D ? hw : 0)) - XH(k - (z > 0 ? hw : 0))),
                                f2 = 0.5f * (XH(k + (y + 1 < H ? W : 0)) - XH(k - (y > 0 ? W : 0))), f3 = 0.5f * (XH(k + (x + 1 < W ? 1 : 0)) - XH(k - (x > 0 ? 1 : 0)));
                    const float t = u[4] * f1 + u[5] * f2 + u[6] * f3;
                    const float a = __half2float(__float2half_rn(u[0] + u[2] * f0 + t)), b = __half2float(__float2half_rn(u[1] + u[3] * f0 - t));
                    const float pr = 1.f / (1.f + __expf(-a)), pv = 1.f / (1.f + __expf(-b));
                    float f = (1.f + 254.f * (0.5f + 0.5f * (pr - pv))) / 255.f; f = fminf(fmaxf(f, 1.5f / 255.f), 1.f - 1e-4f);   /* (nn_rv_fuse, fp16) */
                    const float l = __half2float(__float2half(__logf(f / (1.f - f))));
                    o = (uint8_t)(255.f / (1.f + __expf(-l)) + 0.5f);   /* (nn_pred_place) */
                }
                dsh[gr + x] = o;
            }
        }
    }
#undef XH
}
/* planes z0 .. z0 + nz of a window's z-scored CT channel as the 16-bit input stores it (pred_in_k), from its bytes: the guided head's
   x for one slab (predict --fit-zarr in z chunks) */
__global__ void lat_ct_x_k(const uint8_t *ct, size_t n, float mean, float isd, f16 *x) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) x[i] = __float2half(((float)ct[i] - mean) * isd);
}
extern "C" void nn_lat_ct_x(const uint8_t *ct, int W, int z0, int nz, float mean, float isd, void *x) {
    const size_t hw = (size_t)W * W, n = (size_t)nz * hw;
    lat_ct_x_k<<<nblk(n, 256), 256>>>(ct + (size_t)z0 * hw, n, mean, isd, (f16 *)x); KCHECK();
}
extern "C" void nn_lat_place_rvg(const void *lg, int lh16, int nb, int d, int h, int w, const uint8_t *ctg, float mean, float isd, const uint8_t *ct, const uint32_t *bits,
                                 int iso, int hz, int halo, int oz, int oy, int ox, int ez, int ey, int ex, int shard, uint8_t *dsh) {
    const int D = 4 * d, H = 4 * h, W = 4 * w;
    const int z0 = max(hz, -oz), y0 = max(halo, -oy), x0 = max(halo, -ox), z1 = min(D - hz, ez - oz), y1 = min(H - halo, ey - oy), x1 = min(W - halo, ex - ox);
    if (z1 <= z0 || y1 <= y0 || x1 <= x0) return;
    const size_t n = (size_t)(((z1 + 3) >> 2) - (z0 >> 2)) * (((y1 + 3) >> 2) - (y0 >> 2)) * (((x1 + 3) >> 2) - (x0 >> 2));
    if (lh16) lat_place_rvg_k<f16><<<nblk(n, 128), 128>>>((const f16 *)lg, nb, d, h, w, ctg, mean, isd, ct, bits, iso, z0, y0, x0, z1, y1, x1, oz, oy, ox, shard, dsh);
    else lat_place_rvg_k<float><<<nblk(n, 128), 128>>>((const float *)lg, nb, d, h, w, ctg, mean, isd, ct, bits, iso, z0, y0, x0, z1, y1, x1, oz, oy, ox, shard, dsh);
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
/* the same for a --lat-basis checkpoint: lat_input_x_k's channels (C projections of the cell's 64 z-scored CT values onto the basis,
   quantised by step, times the gains; then the cell means of input channels 1-3) from the CT bytes, each voxel value rounded to fp16
   as pred_in_k stores it, the same summation order */
__global__ void lat_pred_in_bas_k(const uint8_t *ct, int D, int W, float mean, float isd, const float *dyo, const float *dxo, int axis,
                                  const float *bas, const float *gain, int C, float step, f16 *y) {
    const int d = D / 4, h = W / 4;
    const size_t cv = (size_t)d * h * h, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= cv) return;
    const int z = (int)(i / ((size_t)h * h)), yy = (int)(i / h % h), xx = (int)(i % h);
    float acc[3] = {0.f, 0.f, 0.f}, v[64];
    for (int k = 0; k < 64; k++) {
        const int fz = 4 * z + (k >> 4), fy = 4 * yy + (k >> 2 & 3), fx = 4 * xx + (k & 3);
        const float dy = dyo[fz] + (float)fy, dx = dxo[fz] + (float)fx, inv = axis ? (float)axis / (sqrtf(dy * dy + dx * dx) + 1e-6f) : 0.f;
        v[k] = __half2float(__float2half(((float)ct[((size_t)fz * W + fy) * W + fx] - mean) * isd));
        acc[1] += __half2float(__float2half(dy * inv)); acc[2] += __half2float(__float2half(dx * inv));
    }
    for (int c = 0; c < C; c++) {
        float s = 0.f; for (int k = 0; k < 64; k++) s += bas[c * 64 + k] * v[k];
        if (step > 0.f) { const float lim = c ? 127.f : 32767.f; s = fminf(lim, fmaxf(-lim, rintf(s / step))) * step; }
        y[(size_t)c * cv + i] = __float2half_rn(s * gain[c]);
    }
    for (int c = 0; c < 3; c++) y[(size_t)(C + c) * cv + i] = __float2half_rn(acc[c] * (1.f / 64.f));
}
extern "C" void nn_lat_pred_input_c(const uint8_t *ct, int D, int W, float mean, float isd, const float *dyo, const float *dxo, int axis,
                                    const float *bas, const float *gain, int C, float step, void *y) {
    const size_t cv = (size_t)(D / 4) * (W / 4) * (W / 4);
    lat_pred_in_bas_k<<<nblk(cv, 128), 128>>>(ct, D, W, mean, isd, dyo, dxo, axis, bas, gain, C, step, (f16 *)y); KCHECK();
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
extern "C" void nn_lat_up_slab_c(const void *lg, int lh16, int C, int d, int h, int w, int z0, int nz, void *y, int q0, int qd) {
    const size_t ns = (size_t)nz * 16 * h * w;
    const int slow = lat_slow();
    if (!slow && z0 % 4 == 0 && nz % 4 == 0) {
        const size_t nc = (size_t)(nz / 4) * h * w;
        if (lh16) lat_up4_k<f16><<<nblk(nc, 128), 128>>>((const f16 *)lg, C, d, h, w, z0, nz, (f16 *)y, q0, qd);
        else lat_up4_k<float><<<nblk(nc, 128), 128>>>((const float *)lg, C, d, h, w, z0, nz, (f16 *)y, q0, qd);
        KCHECK(); return;
    }
    if (lh16) lat_up_k<f16><<<nblk(ns, 256), 256>>>((const f16 *)lg, C, d, h, w, z0, nz, (f16 *)y, q0, qd);
    else lat_up_k<float><<<nblk(ns, 256), 256>>>((const float *)lg, C, d, h, w, z0, nz, (f16 *)y, q0, qd);
    KCHECK();
}
extern "C" void nn_lat_up_slab(const void *lg, int lh16, int C, int d, int h, int w, int z0, int nz, void *y) { nn_lat_up_slab_c(lg, lh16, C, d, h, w, z0, nz, y, 0, d); }
extern "C" void nn_lat_up_adj_c(const void *g, int C, int d, int h, int w, int z0, int nz, float *gc, void *scratch, int q0, int qd) {
    const int slow = lat_slow();
    if (!slow && scratch && z0 % 4 == 0 && nz % 4 == 0) {   /* two separable gathers through scratch (c nz h w fp32: 7 bytes per slab voxel hold 28 channels; more go in groups of 28) */
        const int a0 = max(z0 / 4 - 1, 0), a1 = min((z0 + nz) / 4 + 1, d);
        for (int c0 = 0; c0 < C; c0 += 28) {
            const int c = min(28, C - c0); const size_t n1 = (size_t)c * nz * h * w;
            lat_adj_xy_k<<<nblk(n1, 256), 256>>>((const f16 *)g + (size_t)c0 * nz * 16 * h * w, c, h, w, nz, (float *)scratch);
            lat_adj_z_k<<<nblk((size_t)(a1 - a0) * h * w * c, 256), 256>>>((const float *)scratch, c, d, h, w, z0, nz, a0, a1, gc + (size_t)c0 * qd * h * w, q0, qd);
        }
        KCHECK(); return;
    }
    /* coarse planes whose support meets fine planes [z0, z0 + nz) */
    int cz0 = (z0 - 9) / 4; if (z0 - 9 < 0) cz0 = 0;
    int cz1 = (z0 + nz + 6 + 3) / 4 + 1; if (cz1 > d) cz1 = d;
    if (cz0 < q0) cz0 = q0;   /* (a chunk holds every plane with a non-zero weight: z0 / 4 - 1 .. (z0 + nz) / 4) */
    if (cz1 > q0 + qd) cz1 = q0 + qd;
    const size_t n = (size_t)(cz1 - cz0) * h * w * C;
    if (n) lat_adj_k<<<nblk(n, 256), 256>>>((const f16 *)g, C, d, h, w, z0, nz, cz0, cz1, gc, q0, qd);
    KCHECK();
}
extern "C" void nn_lat_up_adj(const void *g, int C, int d, int h, int w, int z0, int nz, float *gc, void *scratch) { nn_lat_up_adj_c(g, C, d, h, w, z0, nz, gc, scratch, 0, d); }
/* predict --latents (a stored ufsm lc-encode volume instead of the CT): q is the window's cells [D/4][W/4][W/4] x (C + 1) int8 (channels
   0-1 the cell-mean coefficient's index, low byte uint8 / high byte int8; 2.. the principal-axis indices; all bytes 0: air), step_raw the
   store's raw-domain quantiser step. nn_lat_synth_ct: the window's CT bytes rebuilt from them, v = sum_c q_c step_raw b_c (orthonormal
   rows: the inverse transform), rounded and clamped to 1..255 in material cells and 0 in air cells, so every reader of the CT window
   (statistics, occupancy, the "0 where CT 0" output masks at cell granularity, the guided heads' CT terms) runs unchanged on it. */
__global__ void lat_synth_k(const int8_t *q, int C, const float *bas, float sraw, int d, int h, int w, uint8_t *ct) {
    const size_t cv = (size_t)d * h * w, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= cv) return;
    const int z = (int)(i / ((size_t)h * w)), yy = (int)(i / w % h), xx = (int)(i % w), H = 4 * h, W = 4 * w;
    const int8_t *qc = q + i * (C + 1);
    int any = 0; for (int c = 0; c <= C; c++) any |= qc[c];
    float a[32]; a[0] = (float)((int)(uint8_t)qc[0] + 256 * (int)qc[1]) * sraw;
    for (int c = 1; c < C; c++) a[c] = (float)qc[c + 1] * sraw;
    for (int k = 0; k < 64; k++) {
        float v = 0.f; for (int c = 0; c < C; c++) v += a[c] * bas[c * 64 + k];
        const size_t j = ((size_t)(4 * z + (k >> 4)) * H + 4 * yy + (k >> 2 & 3)) * W + 4 * xx + (k & 3);
        ct[j] = any ? (uint8_t)fminf(255.f, fmaxf(1.f, rintf(v))) : (uint8_t)0;
    }
}
extern "C" void nn_lat_synth_ct(const void *q, int C, const float *bas, float step_raw, int D, int W, uint8_t *ct) {
    const size_t cv = (size_t)(D / 4) * (W / 4) * (W / 4);
    lat_synth_k<<<nblk(cv, 128), 128>>>((const int8_t *)q, C, bas, step_raw, D / 4, W / 4, W / 4, ct); KCHECK();
}
/* the network input of a --lat-basis checkpoint from the stored coefficients (nn_lat_pred_input_c's channels without the CT): channel c
   (q_c step_raw - [c = 0] 8 mean) isd gain_c (the window's z-score is linear; no second quantiser), then the cell means of the radial
   channels from the window's axis offsets alone, each voxel value rounded to fp16 as lat_pred_in_bas_k does */
__global__ void lat_pred_in_q_k(const int8_t *q, int qs, int D, int W, float mean, float isd, const float *dyo, const float *dxo, int axis,
                                const float *gain, int C, float sraw, f16 *y) {
    const int d = D / 4, h = W / 4;
    const size_t cv = (size_t)d * h * h, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= cv) return;
    const int z = (int)(i / ((size_t)h * h)), yy = (int)(i / h % h), xx = (int)(i % h);
    const int8_t *qc = q + i * (size_t)qs;
    const float dc = (float)((int)(uint8_t)qc[0] + 256 * (int)qc[1]) * sraw;
    y[i] = __float2half_rn((dc - 8.f * mean) * isd * gain[0]);
    for (int c = 1; c < C; c++) y[(size_t)c * cv + i] = __float2half_rn((float)qc[c + 1] * sraw * isd * gain[c]);
    float acc[2] = {0.f, 0.f};
    for (int k = 0; k < 64; k++) {
        const int fz = 4 * z + (k >> 4), fy = 4 * yy + (k >> 2 & 3), fx = 4 * xx + (k & 3);
        const float dy = dyo[fz] + (float)fy, dx = dxo[fz] + (float)fx, inv = axis ? (float)axis / (sqrtf(dy * dy + dx * dx) + 1e-6f) : 0.f;
        acc[0] += __half2float(__float2half(dy * inv)); acc[1] += __half2float(__float2half(dx * inv));
    }
    y[(size_t)C * cv + i] = __float2half_rn(0.f);
    y[(size_t)(C + 1) * cv + i] = __float2half_rn(acc[0] * (1.f / 64.f)); y[(size_t)(C + 2) * cv + i] = __float2half_rn(acc[1] * (1.f / 64.f));
}
extern "C" void nn_lat_pred_input_qs(const void *q, int stride, int D, int W, float mean, float isd, const float *dyo, const float *dxo, int axis,
                                     const float *gain, int C, float step_raw, void *y) {
    const size_t cv = (size_t)(D / 4) * (W / 4) * (W / 4);
    lat_pred_in_q_k<<<nblk(cv, 128), 128>>>((const int8_t *)q, stride, D, W, mean, isd, dyo, dxo, axis, gain, C, step_raw, (f16 *)y); KCHECK();
}
extern "C" void nn_lat_pred_input_q(const void *q, int D, int W, float mean, float isd, const float *dyo, const float *dxo, int axis,
                                    const float *gain, int C, float step_raw, void *y) {
    nn_lat_pred_input_qs(q, C + 1, D, W, mean, isd, dyo, dxo, axis, gain, C, step_raw, y);
}

/* --- the head slabs' gradient pass fused (train --lat, streamed head): per coarse cell its 4^3 fine logits (lat_up4_k's arithmetic, each
   rounded to fp16 as the slab stored it), the guided head's terms (lat_guide_k), the BCE logit gradient (loss_grad_v, rounded to fp16
   and scaled as nn_loss_grad_only / nn_loss_grad_scale left it) and the transposed upsampling of that gradient onto the cell's 3^3
   neighbourhood, without the fp16 slab in between: the slab path moved ~8 bytes per fine voxel and channel through memory (upsample
   store, gradient load and store, adjoint load), this one the targets and 27 fp32 partials per cell and channel (1.7 B per voxel).
   Stage 1 (lat_fuse_k): one thread per coarse cell of the slab writes part[k][slot][cell], slot = (a 3 + b) 3 + e the neighbour
   (iz[a], iy[b], ix[e]) of lat_up4_k. Stage 2 (lat_fuse_fin_k): every coarse cell gathers the partials addressed to it in a fixed
   order (no atomics: the same bits every run) and adds them to the rolling gradient planes. The values are those of the slab path;
   the sums associate differently (fp32 rounding). Groups: n plain BCE channels c0 .. c0 + n - 1 (targets t [n][ns], one mask), or
   guide 1: the two side channels c0, c0 + 1 with the 5 coefficient channels gb .. gb + 4 (part k = 2 .. 6). */
struct fuse_lam { float v[7]; };
/* a block's sums of n per-thread floats (blocks of 128 threads, every thread calls it): warp shuffles in fp64, the 4 warp sums through
   shared memory, thread 0 adds sum j to ds[idx[j]] (one barrier; the per-value shared-memory tree took 9 barriers a value: 2.1 of
   the mask-only statistics' 2.4 ms per P 768 slab) */
__device__ __forceinline__ void fuse_block_add(const float *acc, int n, const int *idx, double *ds, double (*sh)[36]) {
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    for (int j = 0; j < n; j++) {
        double v = (double)acc[j];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) sh[wp][j] = v;
    }
    __syncthreads();
    if (threadIdx.x == 0) for (int j = 0; j < n; j++) { const double v = sh[0][j] + sh[1][j] + sh[2][j] + sh[3][j]; if (v != 0.) atomicAdd(&ds[idx[j]], v); }
}
/* loss_grad_v for the head slabs' case (no dice, no tolerance band; --recall goes through loss_grad_v itself): the same operations in the
   same order with the channel's statistics in registers (inm = max(nm, 1), fsc = fin[2 C + 1]); s: the sigmoid, for the loss sums */
__device__ __forceinline__ float fuse_grad_v(float x, unsigned tb, unsigned mb, float inm, float fsc, float pw, float gscale, float *s_) {
    const float p = tb * (1.f / 255.f), s = 1.f / (1.f + expf(-x)), mw = (float)mb;
    *s_ = s;
    const float g = mw * ((1.f - p) * s - pw * p * (1.f - s)) / inm;
    return __half2float(f2h<f16>(g * fsc * gscale));
}
template <typename T>
__global__ void lat_fuse_k(const T *lg, int qd, int q0, int d, int h, int w, int z0, int nz, int c0, int n, int guide, int gb, const f16 *x, const uint8_t *t,
                           const uint8_t *m, const uint8_t *wch, const float *st, float pw, float gscale, float rec_w, int unit_c, fuse_lam lam, float *part, double *ds,
                           int szlo, int szhi) {   /* szlo / szhi: the fine planes whose voxels add to the loss sums (the split's owned planes) */
    const int H = 4 * h, W = 4 * w, D = 4 * d, C = guide ? 2 : n, nk = guide ? 7 : n;
    const size_t pl = (size_t)h * w, ncs = (size_t)(nz / 4) * pl, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    /* ds: also the loss sums 1 .. 4 of nn_loss_stats_acc (the statistics pass then only counts the mask weights: nn_lat_fuse_stats mask_only) */
    float acc[7][4];
    if (ds) for (int k = 0; k < 7; k++) for (int j = 0; j < 4; j++) acc[k][j] = 0.f;
#define FUSE_LSUM(k, xv, tbv, mbv, sg_) do { if (ds && fzs_ >= szlo && fzs_ < szhi) { const float mw_ = (float)(mbv), x_ = (xv), p_ = (tbv) * (1.f / 255.f); \
        const float spp_ = fmaxf(x_, 0.f) + log1pf(__expf(-fabsf(x_))); const float bce_ = pw * p_ * (spp_ - x_) + (1.f - p_) * spp_; \
        acc[k][0] += mw_ * bce_; acc[k][1] += (sg_) * p_; acc[k][2] += (sg_); acc[k][3] += p_; } } while (0)
    const bool slow = rec_w > 0.f;   /* --recall: loss_grad_v as it stands */
    if (i < ncs) {
    const int sz = (int)(i / pl), cz = z0 / 4 + sz, cy = (int)(i / w % h), cx = (int)(i % w);
    const int iz[3] = {max(cz - 1, 0), cz, min(cz + 1, d - 1)}, iy[3] = {max(cy - 1, 0), cy, min(cy + 1, h - 1)}, ix[3] = {max(cx - 1, 0), cx, min(cx + 1, w - 1)};
    const size_t cv = (size_t)qd * pl, hw = (size_t)H * W, ns = (size_t)nz * hw;
    const float *fin = st + (size_t)5 * C;
    if (!guide) {
        for (int k = 0; k < n; k++) {
            const T *l = lg + (size_t)(c0 + k) * cv;
            const bool on = wch[k] != 0, unit = k == unit_c;
            const float inm = fmaxf(st[k * 5], 1.f), fsc = fin[2 * C + 1];
            float v[3][3][3], ga[3][3][3];
#pragma unroll
            for (int a = 0; a < 3; a++)
#pragma unroll
                for (int b = 0; b < 3; b++) {
                    const size_t row = ((size_t)(iz[a] - q0) * h + iy[b]) * w;
#pragma unroll
                    for (int e = 0; e < 3; e++) { v[a][b][e] = ldv(l, row + ix[e]); ga[a][b][e] = 0.f; }
                }
            if (on) {
#pragma unroll
                for (int rz = 0; rz < 4; rz++) {
                    float p[3][3], gp[3][3];
#pragma unroll
                    for (int b = 0; b < 3; b++)
#pragma unroll
                        for (int e = 0; e < 3; e++) { p[b][e] = LW[rz][0] * v[0][b][e] + LW[rz][1] * v[1][b][e] + LW[rz][2] * v[2][b][e]; gp[b][e] = 0.f; }
                    const int fzs_ = z0 + 4 * sz + rz;
#pragma unroll
                    for (int ry = 0; ry < 4; ry++) {
                        float q[3], gq[3] = {0.f, 0.f, 0.f};
#pragma unroll
                        for (int e = 0; e < 3; e++) q[e] = LW[ry][0] * p[0][e] + LW[ry][1] * p[1][e] + LW[ry][2] * p[2][e];
                        const size_t vr = ((size_t)(4 * sz + rz) * H + 4 * cy + ry) * W + 4 * cx;
                        const unsigned m4 = *(const unsigned *)(m + vr);
                        if (m4) {
                            const unsigned t4 = *(const unsigned *)(t + (size_t)k * ns + vr);
#pragma unroll
                            for (int rx = 0; rx < 4; rx++) {
                                unsigned mb = m4 >> (8 * rx) & 255u; if (unit && mb) mb = 1u;
                                if (!mb) continue;
                                const float u = __half2float(__float2half_rn(LW[rx][0] * q[0] + LW[rx][1] * q[1] + LW[rx][2] * q[2]));
                                float sg = 0.f;
                                float g = slow ? __half2float(loss_grad_v<f16, f16>(nullptr, C, ns, st, 0.f, fin, pw, gscale, 0, nullptr, D, H, W, rec_w, k, 0, vr + rx, u, t4 >> (8 * rx) & 255u, mb))
                                               : fuse_grad_v(u, t4 >> (8 * rx) & 255u, mb, inm, fsc, pw, gscale, &sg);
                                if (slow) sg = 1.f / (1.f + __expf(-u));
                                FUSE_LSUM(k, u, t4 >> (8 * rx) & 255u, mb, sg);
                                if (lam.v[k] != 1.f) g = __half2float(f2h<f16>(lam.v[k] * g));
#pragma unroll
                                for (int e = 0; e < 3; e++) gq[e] += LW[rx][e] * g;
                            }
                        }
#pragma unroll
                        for (int b = 0; b < 3; b++)
#pragma unroll
                            for (int e = 0; e < 3; e++) gp[b][e] += LW[ry][b] * gq[e];
                    }
#pragma unroll
                    for (int a = 0; a < 3; a++)
#pragma unroll
                        for (int b = 0; b < 3; b++)
#pragma unroll
                            for (int e = 0; e < 3; e++) ga[a][b][e] += LW[rz][a] * gp[b][e];
                }
            }
#pragma unroll
            for (int a = 0; a < 3; a++)
#pragma unroll
                for (int b = 0; b < 3; b++)
#pragma unroll
                    for (int e = 0; e < 3; e++) part[((size_t)k * 27 + (a * 3 + b) * 3 + e) * ncs + i] = ga[a][b][e];
        }
    } else {
    /* the guided side pair: 7 cell channels at once */
    float v[7][27], ga[7][27];
    for (int k = 0; k < 7; k++) {
        const T *l = lg + (size_t)(k < 2 ? c0 + k : gb + k - 2) * cv;
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++) {
                const size_t row = ((size_t)(iz[a] - q0) * h + iy[b]) * w;
                for (int e = 0; e < 3; e++) { v[k][(a * 3 + b) * 3 + e] = ldv(l, row + ix[e]); ga[k][(a * 3 + b) * 3 + e] = 0.f; }
            }
    }
    const bool on0 = wch[0] != 0, on1 = wch[1] != 0;
    const float inm0 = fmaxf(st[0], 1.f), inm1 = fmaxf(st[5], 1.f), fsc = fin[5];
    for (int rz = 0; rz < 4; rz++) {
        float p[7][9], gp[7][9];
        for (int k = 0; k < 7; k++)
            for (int j = 0; j < 9; j++) { p[k][j] = LW[rz][0] * v[k][j] + LW[rz][1] * v[k][9 + j] + LW[rz][2] * v[k][18 + j]; gp[k][j] = 0.f; }
        const int z = z0 + 4 * sz + rz, fzs_ = z;
        for (int ry = 0; ry < 4; ry++) {
            float q[7][3], gq[7][3];
            for (int k = 0; k < 7; k++)
                for (int e = 0; e < 3; e++) { q[k][e] = LW[ry][0] * p[k][e] + LW[ry][1] * p[k][3 + e] + LW[ry][2] * p[k][6 + e]; gq[k][e] = 0.f; }
            const int y = 4 * cy + ry;
            const size_t vr = ((size_t)(4 * sz + rz) * H + y) * W + 4 * cx;
            const unsigned m4 = *(const unsigned *)(m + vr);
            if (m4) {
                const unsigned ta = *(const unsigned *)(t + vr), tb = *(const unsigned *)(t + ns + vr);
                for (int rx = 0; rx < 4; rx++) {
                    const unsigned mb = m4 >> (8 * rx) & 255u;
                    if (!mb) continue;
                    const int xx = 4 * cx + rx;
                    const size_t gv = (size_t)z * hw + (size_t)y * W + xx;
                    const float f0 = __half2float(x[gv]),
                                f1 = 0.5f * (__half2float(x[gv + (z + 1 < D ? hw : 0)]) - __half2float(x[gv - (z > 0 ? hw : 0)])),
                                f2 = 0.5f * (__half2float(x[gv + (y + 1 < H ? W : 0)]) - __half2float(x[gv - (y > 0 ? W : 0)])),
                                f3 = 0.5f * (__half2float(x[gv + (xx + 1 < W ? 1 : 0)]) - __half2float(x[gv - (xx > 0 ? 1 : 0)]));
                    float u[7];
                    for (int k = 0; k < 7; k++) u[k] = __half2float(__float2half_rn(LW[rx][0] * q[k][0] + LW[rx][1] * q[k][1] + LW[rx][2] * q[k][2]));
                    const float tt = u[4] * f1 + u[5] * f2 + u[6] * f3;
                    const float l0 = __half2float(__float2half_rn(u[0] + u[2] * f0 + tt)), l1 = __half2float(__float2half_rn(u[1] + u[3] * f0 - tt));
                    float g0 = 0.f, g1 = 0.f, s0 = 0.f, s1 = 0.f;
                    if (slow) {
                        if (on0) { g0 = __half2float(loss_grad_v<f16, f16>(nullptr, 2, ns, st, 0.f, fin, pw, gscale, 0, nullptr, D, H, W, rec_w, 0, 0, vr + rx, l0, ta >> (8 * rx) & 255u, mb)); s0 = 1.f / (1.f + __expf(-l0)); }
                        if (on1) { g1 = __half2float(loss_grad_v<f16, f16>(nullptr, 2, ns, st, 0.f, fin, pw, gscale, 0, nullptr, D, H, W, rec_w, 1, 0, vr + rx, l1, tb >> (8 * rx) & 255u, mb)); s1 = 1.f / (1.f + __expf(-l1)); }
                    } else {
                        if (on0) g0 = fuse_grad_v(l0, ta >> (8 * rx) & 255u, mb, inm0, fsc, pw, gscale, &s0);
                        if (on1) g1 = fuse_grad_v(l1, tb >> (8 * rx) & 255u, mb, inm1, fsc, pw, gscale, &s1);
                    }
                    if (on0) FUSE_LSUM(0, l0, ta >> (8 * rx) & 255u, mb, s0);
                    if (on1) FUSE_LSUM(1, l1, tb >> (8 * rx) & 255u, mb, s1);
                    if (lam.v[0] != 1.f) { g0 = __half2float(f2h<f16>(lam.v[0] * g0)); g1 = __half2float(f2h<f16>(lam.v[0] * g1)); }
                    const float gd = g0 - g1;
                    const float gk[7] = {g0, g1, __half2float(__float2half_rn(g0 * f0)), __half2float(__float2half_rn(g1 * f0)), __half2float(__float2half_rn(gd * f1)),
                                         __half2float(__float2half_rn(gd * f2)), __half2float(__float2half_rn(gd * f3))};
                    for (int k = 0; k < 7; k++)
                        for (int e = 0; e < 3; e++) gq[k][e] += LW[rx][e] * gk[k];
                }
            }
            for (int k = 0; k < 7; k++)
                for (int b = 0; b < 3; b++)
                    for (int e = 0; e < 3; e++) gp[k][b * 3 + e] += LW[ry][b] * gq[k][e];
        }
        for (int k = 0; k < 7; k++)
            for (int a = 0; a < 3; a++)
                for (int j = 0; j < 9; j++) ga[k][a * 9 + j] += LW[rz][a] * gp[k][j];
    }
    for (int k = 0; k < nk; k++)
        for (int s = 0; s < 27; s++) part[((size_t)k * 27 + s) * ncs + i] = ga[k][s];
    }
    }
#undef FUSE_LSUM
    if (!ds) return;
    __shared__ double sh[4][36];
    float fl[28]; int ix[28];
    for (int k = 0; k < C; k++) for (int j = 0; j < 4; j++) { fl[k * 4 + j] = acc[k][j]; ix[k * 4 + j] = k * 5 + 1 + j; }
    fuse_block_add(fl, C * 4, ix, ds, sh);
}
/* the statistics pass fused the same way (nn_loss_stats_acc's sums of a group from the cell logits: loss_stats_k's per-voxel arithmetic on
   lat_up4_k's / lat_guide_k's fp16-rounded logits; fp32 partials per cell, fp64 per block, added to ds [5 per channel]) */
template <typename T>
__global__ void lat_fuse_stats_k(const T *lg, int qd, int q0, int d, int h, int w, int z0, int nz, int c0, int n, int guide, int gb, const f16 *x, const uint8_t *t,
                                 const uint8_t *m, const uint8_t *wch, float pw, int unit_c, double *ds, int mask_only) {
    const int H = 4 * h, W = 4 * w, D = 4 * d, nk = guide ? 2 : n;
    const size_t pl = (size_t)h * w, ncs = (size_t)(nz / 4) * pl, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    float acc[7][5];
    for (int k = 0; k < 7; k++) for (int j = 0; j < 5; j++) acc[k][j] = 0.f;
    if (i < ncs) {
        const int sz = (int)(i / pl), cz = z0 / 4 + sz, cy = (int)(i / w % h), cx = (int)(i % w);
        const int iz[3] = {max(cz - 1, 0), cz, min(cz + 1, d - 1)}, iy[3] = {max(cy - 1, 0), cy, min(cy + 1, h - 1)}, ix[3] = {max(cx - 1, 0), cx, min(cx + 1, w - 1)};
        const size_t cv = (size_t)qd * pl, hw = (size_t)H * W, ns = (size_t)nz * hw;
#define FUSE_STAT(k, xv, tbv, mbv) do { const float mw_ = (float)(mbv), x_ = (xv), p_ = (tbv) * (1.f / 255.f); const float sg_ = 1.f / (1.f + __expf(-x_)); \
            const float spp_ = fmaxf(x_, 0.f) + log1pf(__expf(-fabsf(x_))); const float bce_ = pw * p_ * (spp_ - x_) + (1.f - p_) * spp_; \
            acc[k][0] += mw_; acc[k][1] += mw_ * bce_; acc[k][2] += sg_ * p_; acc[k][3] += sg_; acc[k][4] += p_; } while (0)
        if (mask_only) {   /* the mask weight sums alone (no logit, no target: the gradient pass adds the other four sums) */
            float sw = 0.f, su = 0.f;
            for (int r = 0; r < 16; r++) {
                const unsigned m4 = *(const unsigned *)(m + ((size_t)(4 * sz + (r >> 2)) * H + 4 * cy + (r & 3)) * W + 4 * cx);
                if (!m4) continue;
#pragma unroll
                for (int rx = 0; rx < 4; rx++) { const unsigned mb = m4 >> (8 * rx) & 255u; sw += (float)mb; su += mb ? 1.f : 0.f; }
            }
            for (int k = 0; k < nk; k++) if (wch[k]) acc[k][0] = !guide && k == unit_c ? su : sw;
        } else if (!guide) {
            for (int k = 0; k < n; k++) {
                if (!wch[k]) continue;
                const T *l = lg + (size_t)(c0 + k) * cv; const bool unit = k == unit_c;
                float v[3][3][3];
#pragma unroll
                for (int a = 0; a < 3; a++)
#pragma unroll
                    for (int b = 0; b < 3; b++) {
                        const size_t row = ((size_t)(iz[a] - q0) * h + iy[b]) * w;
#pragma unroll
                        for (int e = 0; e < 3; e++) v[a][b][e] = ldv(l, row + ix[e]);
                    }
#pragma unroll
                for (int rz = 0; rz < 4; rz++) {
                    float p[3][3];
#pragma unroll
                    for (int b = 0; b < 3; b++)
#pragma unroll
                        for (int e = 0; e < 3; e++) p[b][e] = LW[rz][0] * v[0][b][e] + LW[rz][1] * v[1][b][e] + LW[rz][2] * v[2][b][e];
#pragma unroll
                    for (int ry = 0; ry < 4; ry++) {
                        const size_t vr = ((size_t)(4 * sz + rz) * H + 4 * cy + ry) * W + 4 * cx;
                        const unsigned m4 = *(const unsigned *)(m + vr);
                        if (!m4) continue;
                        float q[3];
#pragma unroll
                        for (int e = 0; e < 3; e++) q[e] = LW[ry][0] * p[0][e] + LW[ry][1] * p[1][e] + LW[ry][2] * p[2][e];
                        const unsigned t4 = *(const unsigned *)(t + (size_t)k * ns + vr);
#pragma unroll
                        for (int rx = 0; rx < 4; rx++) {
                            unsigned mb = m4 >> (8 * rx) & 255u; if (unit && mb) mb = 1u;
                            if (!mb) continue;
                            const float u = __half2float(__float2half_rn(LW[rx][0] * q[0] + LW[rx][1] * q[1] + LW[rx][2] * q[2]));
                            FUSE_STAT(k, u, t4 >> (8 * rx) & 255u, mb);
                        }
                    }
                }
            }
        } else {
            float v[7][27];
            for (int k = 0; k < 7; k++) {
                const T *l = lg + (size_t)(k < 2 ? c0 + k : gb + k - 2) * cv;
                for (int a = 0; a < 3; a++)
                    for (int b = 0; b < 3; b++) {
                        const size_t row = ((size_t)(iz[a] - q0) * h + iy[b]) * w;
                        for (int e = 0; e < 3; e++) v[k][(a * 3 + b) * 3 + e] = ldv(l, row + ix[e]);
                    }
            }
            const bool on0 = wch[0] != 0, on1 = wch[1] != 0;
            for (int rz = 0; rz < 4; rz++) {
                float p[7][9];
                for (int k = 0; k < 7; k++)
                    for (int j = 0; j < 9; j++) p[k][j] = LW[rz][0] * v[k][j] + LW[rz][1] * v[k][9 + j] + LW[rz][2] * v[k][18 + j];
                const int z = z0 + 4 * sz + rz;
                for (int ry = 0; ry < 4; ry++) {
                    const int y = 4 * cy + ry;
                    const size_t vr = ((size_t)(4 * sz + rz) * H + y) * W + 4 * cx;
                    const unsigned m4 = *(const unsigned *)(m + vr);
                    if (!m4) continue;
                    float q[7][3];
                    for (int k = 0; k < 7; k++)
                        for (int e = 0; e < 3; e++) q[k][e] = LW[ry][0] * p[k][e] + LW[ry][1] * p[k][3 + e] + LW[ry][2] * p[k][6 + e];
                    const unsigned ta = *(const unsigned *)(t + vr), tb = *(const unsigned *)(t + ns + vr);
                    for (int rx = 0; rx < 4; rx++) {
                        const unsigned mb = m4 >> (8 * rx) & 255u;
                        if (!mb) continue;
                        const int xx = 4 * cx + rx;
                        const size_t gv = (size_t)z * hw + (size_t)y * W + xx;
                        const float f0 = __half2float(x[gv]),
                                    f1 = 0.5f * (__half2float(x[gv + (z + 1 < D ? hw : 0)]) - __half2float(x[gv - (z > 0 ? hw : 0)])),
                                    f2 = 0.5f * (__half2float(x[gv + (y + 1 < H ? W : 0)]) - __half2float(x[gv - (y > 0 ? W : 0)])),
                                    f3 = 0.5f * (__half2float(x[gv + (xx + 1 < W ? 1 : 0)]) - __half2float(x[gv - (xx > 0 ? 1 : 0)]));
                        float u[7];
                        for (int k = 0; k < 7; k++) u[k] = __half2float(__float2half_rn(LW[rx][0] * q[k][0] + LW[rx][1] * q[k][1] + LW[rx][2] * q[k][2]));
                        const float tt = u[4] * f1 + u[5] * f2 + u[6] * f3;
                        const float l0 = __half2float(__float2half_rn(u[0] + u[2] * f0 + tt)), l1 = __half2float(__float2half_rn(u[1] + u[3] * f0 - tt));
                        if (on0) FUSE_STAT(0, l0, ta >> (8 * rx) & 255u, mb);
                        if (on1) FUSE_STAT(1, l1, tb >> (8 * rx) & 255u, mb);
                    }
                }
            }
        }
#undef FUSE_STAT
    }
    __shared__ double sh[4][36];
    float fl[35]; int ix[35]; int nv = 0;
    for (int k = 0; k < nk; k++) for (int j = 0; j < (mask_only ? 1 : 5); j++) { fl[nv] = acc[k][j]; ix[nv++] = k * 5 + j; }
    fuse_block_add(fl, nv, ix, ds, sh);
}
extern "C" void nn_lat_fuse_stats(const void *lg, int lh16, int lq0, int lqd, int d, int h, int w, int z0, int nz, int c0, int n, int guide, int gb, const void *x,
                                  const uint8_t *t, const uint8_t *m, const uint8_t *wch, float pw, int unit_c, double *ds, int mask_only) {
    if (z0 % 4 || nz % 4 || n > 7 || (guide && n != 2)) { fprintf(stderr, "nn_lat_fuse_stats: whole coarse planes, at most 7 channels (a guided pair: 2)\n"); abort(); }
    const size_t ncs = (size_t)(nz / 4) * h * w;
    if (lh16) lat_fuse_stats_k<f16><<<nblk(ncs, 128), 128>>>((const f16 *)lg, lqd, lq0, d, h, w, z0, nz, c0, n, guide, gb, (const f16 *)x, t, m, wch, pw, unit_c, ds, mask_only);
    else lat_fuse_stats_k<float><<<nblk(ncs, 128), 128>>>((const float *)lg, lqd, lq0, d, h, w, z0, nz, c0, n, guide, gb, (const f16 *)x, t, m, wch, pw, unit_c, ds, mask_only);
    KCHECK();
}
/* the (source cell, slot) pairs along one axis whose neighbour is cell c of n: c - 1 (slot 2), c (1), c + 1 (0), and the clamped folds */
__device__ __forceinline__ int fuse_src(int c, int n, int *src, int *slot) {
    int k = 0;
    if (c > 0) { src[k] = c - 1; slot[k++] = 2; }
    src[k] = c; slot[k++] = 1;
    if (c + 1 < n) { src[k] = c + 1; slot[k++] = 0; }
    if (c == 0) { src[k] = 0; slot[k++] = 0; }
    if (c == n - 1) { src[k] = n - 1; slot[k++] = 2; }
    return k;
}
__global__ void lat_fuse_fin_k(const float *part, int nk, int d, int h, int w, int z0, int nz, int cz0, int cz1, float *gc, int q0, int qd, int c0, int gb) {
    const size_t pl = (size_t)h * w, nc = (size_t)(cz1 - cz0) * pl, ncs = (size_t)(nz / 4) * pl, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nc * nk) return;
    const int k = (int)(i / nc); const size_t v = i % nc;
    const int cz = cz0 + (int)(v / pl), cy = (int)(v % pl / w), cx = (int)(v % w), s0 = z0 / 4, sn = nz / 4;
    int zs[4], za[4], ys[4], ya[4], xs[4], xa[4];
    const int nzs = fuse_src(cz, d, zs, za), nys = fuse_src(cy, h, ys, ya), nxs = fuse_src(cx, w, xs, xa);
    const float *pk = part + (size_t)k * 27 * ncs;
    float s = 0.f;
    for (int a = 0; a < nzs; a++) {
        if (zs[a] < s0 || zs[a] >= s0 + sn) continue;
        float sy = 0.f;
        for (int b = 0; b < nys; b++) {
            float sx = 0.f;
            for (int e = 0; e < nxs; e++) sx += pk[(size_t)((za[a] * 3 + ya[b]) * 3 + xa[e]) * ncs + ((size_t)(zs[a] - s0) * h + ys[b]) * w + xs[e]];
            sy += sx;
        }
        s += sy;
    }
    const int ch = gb >= 0 && k >= 2 ? gb + k - 2 : c0 + k;
    gc[((size_t)ch * qd + (cz - q0)) * pl + (size_t)cy * w + cx] += s;
}
/* lg: the cell logits [.][lqd][h][w] from plane lq0 (fp16 if lh16); st: the group's loss scratch (nn_loss_stats_fin); part: 27 n
   floats per coarse cell of the slab (n = 7 for a guided group); gc: the rolling gradient planes [.][gqd][h][w] from plane gq0 */
extern "C" size_t nn_lat_fuse_part(int n, int h, int w, int nz) { return (size_t)n * 27 * (nz / 4) * h * w * sizeof(float); }
/* the fine planes [lo, hi) whose voxels add to nn_lat_fuse_grad's loss sums (default all): under --split z the gradient pass also covers
   a 4-plane halo band the statistics pass leaves out, so the "light" statistics pass (loss sums from the gradient pass) clips them to
   the statistics pass's planes (perf2 2026-11-13) */
static thread_local int g_fsum_lo = -(1 << 30), g_fsum_hi = 1 << 30;
extern "C" void nn_lat_fuse_sum_zwin(int lo, int hi) { g_fsum_lo = lo; g_fsum_hi = hi; }
extern "C" void nn_lat_fuse_grad(const void *lg, int lh16, int lq0, int lqd, int d, int h, int w, int z0, int nz, int c0, int n, int guide, int gb, const void *x,
                                 const uint8_t *t, const uint8_t *m, const uint8_t *wch, const float *st, float pw, float rec_w, int unit_c, const float *lam,
                                 float *part, float *gc, int gq0, int gqd, double *ds) {
    if (z0 % 4 || nz % 4 || n > 7 || (guide && n != 2)) { fprintf(stderr, "nn_lat_fuse_grad: whole coarse planes, at most 7 channels (a guided pair: 2)\n"); abort(); }
    fuse_lam L; for (int k = 0; k < 7; k++) L.v[k] = k < n ? lam[k] : 1.f;
    const int nk = guide ? 7 : n; const size_t ncs = (size_t)(nz / 4) * h * w;
    if (lh16) lat_fuse_k<f16><<<nblk(ncs, 128), 128>>>((const f16 *)lg, lqd, lq0, d, h, w, z0, nz, c0, n, guide, gb, (const f16 *)x, t, m, wch, st, pw, g_gscale, rec_w, unit_c, L, part, ds, g_fsum_lo, g_fsum_hi);
    else lat_fuse_k<float><<<nblk(ncs, 128), 128>>>((const float *)lg, lqd, lq0, d, h, w, z0, nz, c0, n, guide, gb, (const f16 *)x, t, m, wch, st, pw, g_gscale, rec_w, unit_c, L, part, ds, g_fsum_lo, g_fsum_hi);
    const int a0 = max(z0 / 4 - 1, 0), a1 = min((z0 + nz) / 4 + 1, d);
    lat_fuse_fin_k<<<nblk((size_t)(a1 - a0) * h * w * nk, 256), 256>>>(part, nk, d, h, w, z0, nz, a0, a1, gc, gq0, gqd, c0, guide ? gb : -1);
    KCHECK();
}
