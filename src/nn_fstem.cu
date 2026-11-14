/* CUDA ops: the fine voxel stem of the latent-grid net (train --fstem C). The trunk runs on 4^3-voxel cells (38.4 um at 9.6 um voxels);
   the stem reads the CT channel itself at stride 2 (19.2 um): per 2^3 block i a 4^3 conv (taps 2 i - 1 .. 2 i + 2 each axis, zero
   outside the window), h = silu(W1 x + b1) (C channels), and out_o = sum_c W2[o][c] h_c added to output channel chs[o] on the block's 8
   voxels (the head's logits, before the losses; chs[o] < 0: output o skipped). Parameters, contiguous fp32: W1 [C][64], b1 [C], W2 [nout][C]. nn_fstem_bwd: the
   parameter gradients from the logits' gradient planes (as the losses left them: fp16, times the backward's gradient scale, which
   unet removes from the whole gradient), accumulated into gp. Slabs: z0, nz, D, H, W even. */
#include "nn_common.cuh"

enum { FS_MAXC = 16, FS_MAXO = 16, FS_T = 64 };
struct fs_chs { int c[FS_MAXO]; };

/* the kernels work on row tiles: up to FS_BT consecutive stem blocks (bx0 .. bx0 + nj - 1) of one (bz, by) row, one per thread, whose 4^3 taps
   come from the 16 input rows (z 2 bz - 1 .. + 2, y 2 by - 1 .. + 2, x 2 bx0 - 1 .. 2 (bx0 + nj) + 0) staged once in shared memory (each
   voxel read once instead of 8 times; P 256 forward 2.0 -> see DESIGN ms) */
enum { FS_BT = 128, FS_RW = 2 * FS_BT + 2, FS_FS = FS_MAXC + 1, FS_OS = FS_MAXO + 1 };
struct fs_tile { int bz, by, bx0, nj; };
__device__ inline fs_tile fs_tile_of(size_t tl, int z0, int h2, int w2) {
    const int nch = (w2 + FS_BT - 1) / FS_BT; const size_t row = tl / nch; fs_tile t;
    t.bz = z0 / 2 + (int)(row / h2); t.by = (int)(row % h2); t.bx0 = (int)(tl % nch) * FS_BT; t.nj = w2 - t.bx0 < FS_BT ? w2 - t.bx0 : FS_BT;
    return t;
}
__device__ inline void fs_stage(const f16 *x, int D, int H, int W, fs_tile t, __half (*sr)[FS_RW]) {
    const int rl = 2 * t.nj + 2; const size_t hw = (size_t)H * W;
    for (int k = threadIdx.x; k < 16 * rl; k += blockDim.x) {
        const int r = k / rl, q = k % rl, z = 2 * t.bz - 1 + r / 4, y = 2 * t.by - 1 + r % 4, xx = 2 * t.bx0 - 1 + q;
        sr[r][q] = z >= 0 && z < D && y >= 0 && y < H && xx >= 0 && xx < W ? x[(size_t)z * hw + (size_t)y * W + xx] : __float2half(0.f);
    }
}
__device__ inline float fs_silu(float v) { return v / (1.f + __expf(-v)); }
__device__ inline void fs_load(const __half (*sr)[FS_RW], int j, float *t) {   /* block j's 64 taps from the staged rows, in fp32 */
    for (int r = 0; r < 16; r++) {
        const __half2 p0 = *(const __half2 *)&sr[r][2 * j], p1 = *(const __half2 *)&sr[r][2 * j + 2];
        t[4 * r] = __low2float(p0); t[4 * r + 1] = __high2float(p0); t[4 * r + 2] = __low2float(p1); t[4 * r + 3] = __high2float(p1);
    }
}
__device__ inline float fs_pre(const float *W1, const float *b1, int c, const float *t) {   /* b1 + W1 row c . taps */
    const float4 *w4 = (const float4 *)(W1 + c * FS_T); float a = b1[c];
    for (int k = 0; k < FS_T / 4; k++) { const float4 w = w4[k]; a += w.x * t[4 * k] + w.y * t[4 * k + 1] + w.z * t[4 * k + 2] + w.w * t[4 * k + 3]; }
    return a;
}

/* the forward on the tensor cores (mma m16n8k16, fp16 x fp16 -> fp32): pre [16 c][8 blocks] = W1 (fp16) x taps per n-tile of a warp, B fragments
   straight from the staged rows (taps 16 s + 2 t, + 1 of block g: row 4 s + t / 2, a 4-byte pair at 2 (j0 + g) + 2 (t % 2)); h of all the
   tile's blocks into shared memory, then thread j projects block j (W2 . h, the same sums in the same order) and does its 4 logit pairs'
   read-modify-write per output: a warp's rows are 128 contiguous bytes (perf2 2026-11-13: lanes l % 8 x outputs l / 8 wrote 32-byte pieces
   from 4 planes at a time, 100 GB/s, 3.7 ms per P 512 32-plane slab with 11 outputs; the loads of an output's 4 rows now go out together) */
/* UP (nn_fstem_up_fwd): the slab's logits are written, not added to: per output the cell grid's upsampled value (lat_up4_k's separable
   arithmetic, constant weights folded as there), rounded to fp16, plus the sub-cell term on the side pair (lat_shuf_k, rounded), plus v
   (rounded): the bytes of nn_lat_up_slab_c + nn_lat_shuffle_c + nn_fstem_fwd, one write per voxel instead of a write and two
   read-modify-writes. The tile's 4 fine rows (2 z x 2 y) share their z / y-collapsed values per coarse column: qs [o][dz dy][column] */
__device__ static const float FLW[4][3] = {{0.375f, 0.625f, 0.f}, {0.1875f, 0.75f, 0.0625f}, {0.0625f, 0.75f, 0.1875f}, {0.f, 0.625f, 0.375f}};   /* (nn_lat.cu LW) */
enum { FSU_NC = FS_BT / 2 + 2 };
struct fs_up { const f16 *lc; int d, h, w, q0, qd, side, base; };
template <int RZ0, int RY0> __device__ __forceinline__ void fsu_q(const float (*v)[3], float *qo) {
#pragma unroll
    for (int dz = 0; dz < 2; dz++) {
        float pp[3];
#pragma unroll
        for (int b = 0; b < 3; b++) pp[b] = FLW[RZ0 + dz][0] * v[0][b] + FLW[RZ0 + dz][1] * v[1][b] + FLW[RZ0 + dz][2] * v[2][b];
#pragma unroll
        for (int dy = 0; dy < 2; dy++) qo[(2 * dz + dy) * FSU_NC] = FLW[RY0 + dy][0] * pp[0] + FLW[RY0 + dy][1] * pp[1] + FLW[RY0 + dy][2] * pp[2];
    }
}
template <int RX0> __device__ __forceinline__ void fsu_x(const float *q, f16 *u) {   /* q: the 3 columns' values of one (dz, dy) row */
#pragma unroll
    for (int dx = 0; dx < 2; dx++) u[dx] = __float2half_rn(FLW[RX0 + dx][0] * q[0] + FLW[RX0 + dx][1] * q[1] + FLW[RX0 + dx][2] * q[2]);
}
__device__ __forceinline__ void fsu_row(const float *qo, int bx, int cx0, f16 (*u)[2]) {   /* block bx's 2 x 2 x 2 upsampled values (u[dz dy][dx]) */
    const int k0 = (bx >> 1) - cx0;
#pragma unroll
    for (int r = 0; r < 4; r++) {
        const float q[3] = {qo[r * FSU_NC + k0], qo[r * FSU_NC + k0 + 1], qo[r * FSU_NC + k0 + 2]};
        if (bx & 1) fsu_x<2>(q, u[r]); else fsu_x<0>(q, u[r]);
    }
}
__device__ __forceinline__ void fsu_side(const fs_up &up, int ch, fs_tile t, int bx, f16 (*u)[2]) {   /* the side pair's sub-cell terms (lat_shuf_k) */
    const int js = ch - up.side; if (up.side < 0 || js < 0 || js > 1) return;
    const size_t nc = (size_t)up.qd * up.h * up.w, cell = ((size_t)((t.bz >> 1) - up.q0) * up.h + (t.by >> 1)) * up.w + (bx >> 1);
    const f16 *s = up.lc + (size_t)(up.base + 64 * js) * nc + cell;
#pragma unroll
    for (int r = 0; r < 4; r++)
#pragma unroll
        for (int dx = 0; dx < 2; dx++) {
            const int sub = 16 * ((2 * t.bz + (r >> 1)) & 3) + 4 * ((2 * t.by + (r & 1)) & 3) + ((2 * bx + dx) & 3);
            u[r][dx] = __float2half_rn(__half2float(u[r][dx]) + (float)s[(size_t)sub * nc]);
        }
}
template <bool UP>
__global__ void __launch_bounds__(FS_BT) fstem_fwd_k(const f16 *__restrict__ x, int D, int H, int W, int z0, int nz, const float *__restrict__ p, int C, int nout, fs_chs chs, f16 *__restrict__ lg, size_t ns, fs_up up) {
    __shared__ __align__(16) float sp[FS_MAXC + FS_MAXO * FS_MAXC];   /* b1, W2 */
    __shared__ __align__(16) __half sr[16][FS_RW];
    __shared__ float hs[FS_BT][FS_FS];
    __shared__ float qs[UP ? FS_MAXO * 4 * FSU_NC : 1];
    const int tid = threadIdx.x, lane = tid & 31, wp = tid >> 5, g = lane >> 2, tq = lane & 3;
    for (int k = tid; k < C + nout * C; k += blockDim.x) sp[k] = p[C * FS_T + k];
    const fs_tile t = fs_tile_of(blockIdx.x, z0, H / 2, W / 2);
    if (!UP || C) fs_stage(x, D, H, W, t, sr);
    if constexpr (UP) {
        const int cz = t.bz >> 1, cy = t.by >> 1, cx0 = t.bx0 >> 1, ncol = t.nj / 2 + 2, rz0 = (2 * t.bz) & 3, ry0 = (2 * t.by) & 3;
        const int iz[3] = {max(cz - 1, 0) - up.q0, cz - up.q0, min(cz + 1, up.d - 1) - up.q0}, iy[3] = {max(cy - 1, 0), cy, min(cy + 1, up.h - 1)};
        const size_t cv = (size_t)up.qd * up.h * up.w;
        for (int k = tid; k < nout * ncol; k += blockDim.x) {
            const int o = k / ncol, col = k % ncol, ch = chs.c[o]; if (ch < 0) continue;
            const int cxs = min(max(cx0 - 1 + col, 0), up.w - 1);
            const f16 *l = up.lc + (size_t)ch * cv;
            float v[3][3];
#pragma unroll
            for (int a = 0; a < 3; a++)
#pragma unroll
                for (int b = 0; b < 3; b++) v[a][b] = ldv(l, ((size_t)iz[a] * up.h + iy[b]) * up.w + cxs);
            float *qo = qs + o * 4 * FSU_NC + col;
            if (rz0) { if (ry0) fsu_q<2, 2>(v, qo); else fsu_q<2, 0>(v, qo); } else { if (ry0) fsu_q<0, 2>(v, qo); else fsu_q<0, 0>(v, qo); }
        }
        if (!C) {   /* (no stem on this window: the upsampled values and the sub-cell terms only) */
            __syncthreads();
            const int j = tid; if (j >= t.nj) return;
            const size_t hw = (size_t)H * W; const int bx = t.bx0 + j;
            const size_t r0 = (size_t)(2 * t.bz - z0) * hw + (size_t)(2 * t.by) * W + 2 * bx;
            for (int o = 0; o < nout; o++) {
                const int ch = chs.c[o]; if (ch < 0) continue;
                f16 u[4][2]; fsu_row(qs + o * 4 * FSU_NC, bx, cx0, u); fsu_side(up, ch, t, bx, u);
                __half2 *q0 = (__half2 *)(lg + (size_t)ch * ns + r0), *q1 = (__half2 *)((f16 *)q0 + W), *q2 = (__half2 *)((f16 *)q0 + hw), *q3 = (__half2 *)((f16 *)q2 + W);
                *q0 = __halves2half2(u[0][0], u[0][1]); *q1 = __halves2half2(u[1][0], u[1][1]); *q2 = __halves2half2(u[2][0], u[2][1]); *q3 = __halves2half2(u[3][0], u[3][1]);
            }
            return;
        }
    }
    unsigned af[4][4];   /* W1's A fragments (rows c >= C zero) */
    for (int s_ = 0; s_ < 4; s_++) for (int r = 0; r < 4; r++) {
        const int row = g + (r & 1) * 8, col = 16 * s_ + 2 * tq + (r >> 1) * 8;
        const float w0 = row < C ? p[row * FS_T + col] : 0.f, w1 = row < C ? p[row * FS_T + col + 1] : 0.f;
        const __half2 h2 = __floats2half2_rn(w0, w1); af[s_][r] = *(const unsigned *)&h2;
    }
    __syncthreads();
    const float *b1 = sp, *W2 = sp + C;
    for (int j0 = wp * 8; j0 < t.nj; j0 += FS_BT / 32 * 8) {   /* (FS_BT / 32 warps x 8 blocks: one n-tile per warp and pass) */
        float c4[4] = {0.f, 0.f, 0.f, 0.f};
        const int jb = j0 + g < t.nj ? j0 + g : t.nj - 1;   /* (a tail tile repeats its last block: computed, not written) */
        for (int s_ = 0; s_ < 4; s_++) {
            unsigned bf[2];
            bf[0] = *(const unsigned *)&sr[4 * s_ + tq / 2][2 * jb + 2 * (tq & 1)];
            bf[1] = *(const unsigned *)&sr[4 * s_ + 2 + tq / 2][2 * jb + 2 * (tq & 1)];
            mma16816<f16>(c4, af[s_], bf);
        }
        /* c4: pre[c g, g + 8][blocks 2 tq, 2 tq + 1] */
        for (int r = 0; r < 4; r++) { const int c = g + (r >> 1) * 8, b = 2 * tq + (r & 1); if (c < C && j0 + b < FS_BT) hs[j0 + b][c] = fs_silu(c4[r] + b1[c]); }
    }
    __syncthreads();
    const int j = tid;
    if (j < t.nj) {
        const size_t hw = (size_t)H * W; const int bx = t.bx0 + j;
        float h[FS_MAXC];
#pragma unroll
        for (int c = 0; c < FS_MAXC; c++) h[c] = c < C ? hs[j][c] : 0.f;
        const size_t r0 = (size_t)(2 * t.bz - z0) * hw + (size_t)(2 * t.by) * W + 2 * bx;
        for (int o = 0; o < nout; o++) {
            if (chs.c[o] < 0) continue;   /* (an output the caller's planes do not hold) */
            float v = 0.f; for (int c = 0; c < C; c++) v += W2[o * C + c] * h[c];
            __half2 *q0 = (__half2 *)(lg + (size_t)chs.c[o] * ns + r0), *q1 = (__half2 *)((f16 *)q0 + W), *q2 = (__half2 *)((f16 *)q0 + hw), *q3 = (__half2 *)((f16 *)q2 + W);
            __half2 o0, o1, o2, o3;
            if constexpr (UP) {
                f16 u[4][2]; fsu_row(qs + o * 4 * FSU_NC, bx, t.bx0 >> 1, u); fsu_side(up, chs.c[o], t, bx, u);
                o0 = __halves2half2(u[0][0], u[0][1]); o1 = __halves2half2(u[1][0], u[1][1]); o2 = __halves2half2(u[2][0], u[2][1]); o3 = __halves2half2(u[3][0], u[3][1]);
            } else { o0 = *q0; o1 = *q1; o2 = *q2; o3 = *q3; }   /* (x pairs: 4-byte aligned, W even) */
            *q0 = __floats2half2_rn(__low2float(o0) + v, __high2float(o0) + v); *q1 = __floats2half2_rn(__low2float(o1) + v, __high2float(o1) + v);
            *q2 = __floats2half2_rn(__low2float(o2) + v, __high2float(o2) + v); *q3 = __floats2half2_rn(__low2float(o3) + v, __high2float(o3) + v);
        }
    }
}
/* the backward: per row tile the forward recompute (h, the pre-activation gradient; rows padded against bank conflicts), then the
   parameter-gradient outer products over the tile with register blocking: thread t owns W1 rows 2 (t / 16) .. +1 x taps 4 (t % 16) .. +3
   (one staged row: 4 consecutive halves), and up to 3 of the b1 / W2 entries */
__device__ __forceinline__ unsigned fs_tf(float v) { unsigned r; asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(r) : "f"(v)); return r; }
/* mma m16n8k8 tf32 x tf32 -> f32: A (16x8 row) a0 = A[g][t], a1 = A[g+8][t], a2 = A[g][t+4], a3 = A[g+8][t+4]; B (8x8 col) b0 = B[t][g], b1 = B[t+4][g]; C as m16n8k16 */
__device__ __forceinline__ void fs_mma_tf32(float *c, const unsigned *a, const unsigned *b) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
enum { FS_BH = FS_BT / 2 };   /* the backward's half tiles: 22 KB of shared memory, 4 blocks an SM (one tile at a time: 2, 16 % occupancy, 4.0 ms) */
__global__ void __launch_bounds__(FS_BT) fstem_bwd_k(const f16 *__restrict__ x, int D, int H, int W, int z0, int nz, const float *__restrict__ p, int C, int nout, fs_chs chs, const f16 *__restrict__ g, size_t ns, float *__restrict__ gp) {   /* gp: this block's partial gradient row (part + blockIdx.x * npar) */
    __shared__ float sp[FS_MAXC + FS_MAXO * FS_MAXC];   /* b1, W2 (W1 lives in the A fragments) */
    __shared__ __align__(16) __half sr[16][FS_RW];
    __shared__ float sh[FS_BH][FS_FS], sgp[FS_BH][FS_FS], sgo[FS_BH][FS_OS];
    const int tid = threadIdx.x, n2 = C + nout * C;
    for (int k = tid; k < n2; k += FS_BT) sp[k] = p[C * FS_T + k];
    gp += (size_t)blockIdx.x * (C * FS_T + n2);
    const float *b1 = sp, *W2 = sp + C;
    const int c0 = 2 * (tid / 16), t0 = 4 * (tid % 16), lane = tid & 31, wp = tid >> 5, gq = lane >> 2, tq = lane & 3;
    unsigned af[4][4];   /* W1's A fragments for the recompute (as the forward: the same fp16 weights) */
    for (int s_ = 0; s_ < 4; s_++) for (int r = 0; r < 4; r++) {
        const int row = gq + (r & 1) * 8, col = 16 * s_ + 2 * tq + (r >> 1) * 8;
        const __half2 h2 = __floats2half2_rn(row < C ? p[row * FS_T + col] : 0.f, row < C ? p[row * FS_T + col + 1] : 0.f); af[s_][r] = *(const unsigned *)&h2;
    }
    float d1[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}}, d2[4] = {0.f, 0.f, 0.f, 0.f}, db = 0.f;   /* dW1 n-tiles 2 wp, 2 wp + 1 (taps); dW2 n-tile wp (warps 0-1); b1 (tid < C) */
    const int h2 = H / 2, w2 = W / 2; const size_t ntile = (size_t)(nz / 2) * h2 * ((w2 + FS_BT - 1) / FS_BT), hw = (size_t)H * W;
    for (size_t tl = blockIdx.x; tl < ntile; tl += gridDim.x) {
        __syncthreads();   /* (the previous tile's readers are done) */
        const fs_tile tt = fs_tile_of(tl, z0, h2, w2);
        fs_stage(x, D, H, W, tt, sr);
        for (int jlo = 0; jlo < tt.nj; jlo += FS_BH) {
            const int nh = tt.nj - jlo < FS_BH ? tt.nj - jlo : FS_BH;
            __syncthreads();   /* (the staged rows; the previous half's readers are done) */
            if (wp * 8 < nh) {   /* the pre-activations on the tensor cores into sh (n-tiles of 8 blocks: one per warp and half) */
                const int j0 = jlo + wp * 8; float c4[4] = {0.f, 0.f, 0.f, 0.f}; const int jb = j0 + gq < jlo + nh ? j0 + gq : jlo + nh - 1;
                for (int s_ = 0; s_ < 4; s_++) {
                    unsigned bf[2];
                    bf[0] = *(const unsigned *)&sr[4 * s_ + tq / 2][2 * jb + 2 * (tq & 1)];
                    bf[1] = *(const unsigned *)&sr[4 * s_ + 2 + tq / 2][2 * jb + 2 * (tq & 1)];
                    mma16816<f16>(c4, af[s_], bf);
                }
                for (int r = 0; r < 4; r++) { const int c = gq + (r >> 1) * 8, b = wp * 8 + 2 * tq + (r & 1); if (c < C) sh[b][c] = c4[r] + b1[c]; }
            }
            if (FS_BH > 32) for (int w = wp + FS_BT / 32; w * 8 < nh && w * 8 < FS_BH; w += FS_BT / 32) {   /* (64-block halves: warps take two n-tiles) */
                const int j0 = jlo + w * 8; float c4[4] = {0.f, 0.f, 0.f, 0.f}; const int jb = j0 + gq < jlo + nh ? j0 + gq : jlo + nh - 1;
                for (int s_ = 0; s_ < 4; s_++) {
                    unsigned bf[2];
                    bf[0] = *(const unsigned *)&sr[4 * s_ + tq / 2][2 * jb + 2 * (tq & 1)];
                    bf[1] = *(const unsigned *)&sr[4 * s_ + 2 + tq / 2][2 * jb + 2 * (tq & 1)];
                    mma16816<f16>(c4, af[s_], bf);
                }
                for (int r = 0; r < 4; r++) { const int c = gq + (r >> 1) * 8, b = w * 8 + 2 * tq + (r & 1); if (c < C) sh[b][c] = c4[r] + b1[c]; }
            }
            for (int k = tid; k < nout * FS_BH; k += FS_BT) {   /* the output gradients summed over each block's 8 voxels, all threads (coalesced along x) */
                const int o = k / FS_BH, j = k % FS_BH; float v = 0.f;
                if (j < nh && chs.c[o] >= 0) {
                    const f16 *pl = g + (size_t)chs.c[o] * ns + (size_t)(2 * tt.bz - z0) * hw + (size_t)(2 * tt.by) * W + 2 * (tt.bx0 + jlo + j);
                    const __half2 q0 = *(const __half2 *)pl, q1 = *(const __half2 *)(pl + W), q2 = *(const __half2 *)(pl + hw), q3 = *(const __half2 *)(pl + hw + W);
                    v = __low2float(q0) + __high2float(q0) + __low2float(q1) + __high2float(q1) + __low2float(q2) + __high2float(q2) + __low2float(q3) + __high2float(q3);
                }
                sgo[j][o] = v;
            }
            __syncthreads();
            if (tid < nh) {
                float go[FS_MAXO];
                for (int o = 0; o < nout; o++) go[o] = sgo[tid][o];
                for (int c = 0; c < C; c++) {
                    const float a = sh[tid][c], s = 1.f / (1.f + __expf(-a));
                    float gh = 0.f; for (int o = 0; o < nout; o++) gh += W2[o * C + c] * go[o];
                    sh[tid][c] = a * s; sgp[tid][c] = gh * s * (1.f + a * (1.f - s));
                }
            } else if (tid < FS_BH) {   /* (blocks past the tile: zero rows, the products' K sums read whole k-steps) */
                for (int o = nout; o < FS_MAXO; o++) sgo[tid][o] = 0.f;
                for (int c = 0; c < FS_MAXC; c++) sh[tid][c] = sgp[tid][c] = 0.f;
            }
            __syncthreads();
            const int nk = (nh + 7) / 8;   /* the parameter-gradient products over the half's blocks on the tensor cores (tf32: fp32 range) */
            for (int ks = 0; ks < nk; ks++) {
                const int ja = 8 * ks + tq, jb2 = ja + 4;
                const unsigned a[4] = {fs_tf(sgp[ja][gq]), fs_tf(sgp[ja][gq + 8]), fs_tf(sgp[jb2][gq]), fs_tf(sgp[jb2][gq + 8])};   /* A = dpre [c][j] */
                for (int n = 0; n < 2; n++) {
                    const int k = 8 * (2 * wp + n) + gq;   /* tap */
                    const unsigned b[2] = {fs_tf(ja < nh ? __half2float(sr[k / 4][2 * (jlo + ja) + k % 4]) : 0.f), fs_tf(jb2 < nh ? __half2float(sr[k / 4][2 * (jlo + jb2) + k % 4]) : 0.f)};
                    fs_mma_tf32(d1[n], a, b);
                }
                if (wp < 2) {   /* dW2 [o][c]: A = go [o][j], B = h [j][c] (n-tile wp: c 8 wp ..) */
                    const unsigned a2_[4] = {fs_tf(sgo[ja][gq]), fs_tf(sgo[ja][gq + 8]), fs_tf(sgo[jb2][gq]), fs_tf(sgo[jb2][gq + 8])};
                    const unsigned b2_[2] = {fs_tf(sh[ja][8 * wp + gq]), fs_tf(sh[jb2][8 * wp + gq])};
                    fs_mma_tf32(d2, a2_, b2_);
                }
            }
            if (tid < C) for (int j = 0; j < nh; j++) db += sgp[j][tid];
        }
    }
    for (int n = 0; n < 2; n++) for (int r = 0; r < 4; r++) {   /* C fragments: rows gq, gq + 8 (c), columns 2 tq, + 1 */
        const int c = gq + (r >> 1) * 8, k = 8 * (2 * wp + n) + 2 * tq + (r & 1);
        if (c < C) gp[c * FS_T + k] = d1[n][r];
    }
    if (wp < 2) for (int r = 0; r < 4; r++) { const int o = gq + (r >> 1) * 8, c = 8 * wp + 2 * tq + (r & 1); if (o < nout && c < C) gp[C * FS_T + C + o * C + c] = d2[r]; }
    if (tid < C) gp[C * FS_T + tid] = db;
}
/* the blocks' partial rows added in block order (deterministic: float atomics made --det runs differ in the last bits, which a discrete
   choice later amplified -- the 2-GPU threads gate caught it) */
__global__ void fstem_red_k(const float *__restrict__ part, int nb, int npar, float *__restrict__ gp) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npar) return;
    float s = 0.f; for (int b = 0; b < nb; b++) s += part[(size_t)b * npar + i];
    gp[i] += s;
}
static fs_chs fs_list(const int *chs, int nout) { fs_chs l; for (int o = 0; o < FS_MAXO; o++) l.c[o] = o < nout ? chs[o] : 0; return l; }
static int fs_ok(int D, int H, int W, int z0, int nz, int C, int nout) {
    return C >= 1 && C <= FS_MAXC && nout >= 1 && nout <= FS_MAXO && !(D & 1) && !(H & 1) && !(W & 1) && !(z0 & 1) && !(nz & 1);
}
extern "C" int nn_fstem_fwd(const void *x, int D, int H, int W, int z0, int nz, const float *p, int C, int nout, const int *chs, void *lg, size_t ns) {
    if (!fs_ok(D, H, W, z0, nz, C, nout)) return 1;
    const size_t nt = (size_t)(nz / 2) * (H / 2) * ((W / 2 + FS_BT - 1) / FS_BT); if (!nt) return 0;
    fstem_fwd_k<false><<<(unsigned)nt, FS_BT>>>((const f16 *)x, D, H, W, z0, nz, p, C, nout, fs_list(chs, nout), (f16 *)lg, ns, fs_up{}); KCHECK();
    return 0;
}
extern "C" int nn_fstem_up_fwd(const void *x, int D, int H, int W, int z0, int nz, const float *p, int C, int nout, const int *chs, void *lg, size_t ns,
                               const void *lc, int q0, int qd, int side, int base) {
    if (!(C == 0 ? nout >= 1 && nout <= FS_MAXO && !(H & 3) && !(W & 3) : fs_ok(D, H, W, z0, nz, C, nout)) || (D & 3) || (H & 3) || (W & 3) || (z0 & 3) || (nz & 3)) return 1;
    const int d = D / 4, h = H / 4, w = W / 4;
    const int lo = max(z0 / 4 - 1, 0), hi = min((z0 + nz) / 4, d - 1); if (lo < q0 || hi >= q0 + qd) return 1;   /* (the cell planes the taps read lie in [q0, q0 + qd)) */
    const size_t nt = (size_t)(nz / 2) * (H / 2) * ((W / 2 + FS_BT - 1) / FS_BT); if (!nt) return 0;
    fstem_fwd_k<true><<<(unsigned)nt, FS_BT>>>((const f16 *)x, D, H, W, z0, nz, p, C, nout, fs_list(chs, nout), (f16 *)lg, ns, fs_up{(const f16 *)lc, d, h, w, q0, qd, side, base}); KCHECK();
    return 0;
}
extern "C" int nn_fstem_bwd(const void *x, int D, int H, int W, int z0, int nz, const float *p, int C, int nout, const int *chs, const void *g, size_t ns, float *gp) {
    if (!fs_ok(D, H, W, z0, nz, C, nout)) return 1;
    const size_t ntile = (size_t)(nz / 2) * (H / 2) * ((W / 2 + FS_BT - 1) / FS_BT); if (!ntile) return 0;
    const unsigned nbk = (unsigned)(ntile < 1024 ? ntile : 1024);
    const int npar = C * FS_T + C + nout * C;
    static float *part[8]; static size_t cap[8]; int dv = 0; cudaGetDevice(&dv); dv &= 7;   /* (per device: one host thread per GPU) */
    const size_t need = (size_t)nbk * npar;
    if (need > cap[dv]) { if (part[dv]) cudaFree(part[dv]); if (cudaMalloc(&part[dv], need * 4) != cudaSuccess) { fprintf(stderr, "nn_fstem_bwd: no device memory\n"); abort(); } cap[dv] = need; }
    fstem_bwd_k<<<nbk, FS_BT>>>((const f16 *)x, D, H, W, z0, nz, p, C, nout, fs_list(chs, nout), (const f16 *)g, ns, part[dv]); KCHECK();
    fstem_red_k<<<(npar + 127) / 128, 128>>>(part[dv], (int)nbk, npar, gp); KCHECK();
    return 0;
}
