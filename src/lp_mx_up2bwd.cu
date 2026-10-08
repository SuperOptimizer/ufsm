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
/* tiled: block = (16 x 4 coarse columns, z chunk of ZC coarse planes, n, gx block); thread = (coarse x, coarse y, 8 channels).
   The block walks the fine planes 2 z0 - 1 .. 2 z1, each staged raw (e4m3 rows + scale bytes, 34 x 10 fine positions) into
   shared memory once with cp.async (double-buffered), and adds w_y w_x gy over its 4 x 4 fine taps into the two coarse planes
   the fine plane feeds (fine 2k + 1 and 2k + 2 feed coarse k and k + 1; coarse k is complete after fine 2k + 2). The
   per-voxel kernel read every fine row 8 times (L2-bound). Whole 32-channel blocks of gy and gx; gx written fresh; the
   sums in another order (then the same e4m3 quantisation). Needs Wo % 4 == 0 (4-byte scale copies). */
#define U2B_RW 34   /* fine positions per staged row: 2 x 16 + 2 */
#define U2B_RH 10   /* fine rows: 2 x 4 + 2 */
#define U2B_BUF (U2B_RH * U2B_RW * 32 + U2B_RH * 40)
__global__ void __launch_bounds__(256) up2_bwd_mx_tile_k(const uint8_t *__restrict__ gy, uint8_t *__restrict__ gx, int N, int nby, int nbx, int yb0, int ob0, int nob,
                                                         int D, int H, int W, int ZC) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    const int tid = threadIdx.x, cg = tid & 3, cx = (tid >> 2) & 15, cy = tid >> 6;
    const int x0 = blockIdx.x * 16, y0 = blockIdx.y * 4, nzc = (D + ZC - 1) / ZC;
    int bz = blockIdx.z;
    const int zc = bz % nzc; bz /= nzc;
    const int ob = ob0 + bz % nob, n = bz / nob, yb = yb0 + (ob - ob0);
    const int z0 = zc * ZC, z1 = min(D, z0 + ZC), Do = 2 * D, Ho = 2 * H, Wo = 2 * W;
    const size_t S = (size_t)D * H * W, So = (size_t)Do * Ho * Wo;
    const uint8_t *gyb = gy + ((size_t)n * nby + yb) * So * 32, *gys = gy + (size_t)N * nby * So * 32 + ((size_t)n * nby + yb) * So;
    const int fx0 = 2 * x0 - 1, fy0 = 2 * y0 - 1;
    auto stage = [&](int fz, unsigned char *buf) {   /* raw fine plane fz: rows fy0 .. fy0 + 9, positions fx0 .. fx0 + 33 */
        const bool zok = fz >= 0 && fz < Do;
        for (int k = tid; k < U2B_RH * U2B_RW * 2; k += 256) {
            const int h = k & 1, p = (k >> 1) % U2B_RW, r = (k >> 1) / U2B_RW, oy = fy0 + r, ox = fx0 + p;
            const bool ok = zok && oy >= 0 && oy < Ho && ox >= 0 && ox < Wo;
            cp_async<16>(buf + (r * U2B_RW + p) * 32 + 16 * h, ok ? gyb + (((size_t)fz * Ho + oy) * Wo + ox) * 32 + 16 * h : gyb, ok);
        }
        if (tid < U2B_RH * 10) {   /* scale words: fine x 2 x0 - 4 + 4 j .. + 3 (aligned); byte of position p at p + 3 */
            const int r = tid / 10, j = tid % 10, oy = fy0 + r, ox = 2 * x0 - 4 + 4 * j;
            const bool ok = zok && oy >= 0 && oy < Ho && ox >= 0 && ox + 3 < Wo;
            cp_async<4>(buf + U2B_RH * U2B_RW * 32 + r * 40 + 4 * j, ok ? gys + ((size_t)fz * Ho + oy) * Wo + ox : gys, ok);
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    };
    const int x = x0 + cx, y = y0 + cy;
    float wy[4], wx[4];
#pragma unroll
    for (int a = 0; a < 4; a++) { wy[a] = upc(2 * y - 1 + a, y, H); wx[a] = upc(2 * x - 1 + a, x, W); }
    float acc0[8], acc1[8];
#pragma unroll
    for (int c = 0; c < 8; c++) { acc0[c] = 0.f; acc1[c] = 0.f; }
    uint8_t *gxs = gx + (size_t)N * nbx * S * 32;
    const int f0 = 2 * z0 - 1, f1 = 2 * z1;   /* fine planes f0 .. f1 */
    stage(f0, smem_raw);
    for (int fz = f0; fz <= f1; fz++) {
        const int i = fz - f0;
        if (fz < f1) { stage(fz + 1, smem_raw + ((i + 1) & 1) * U2B_BUF); asm volatile("cp.async.wait_group 1;\n" ::: "memory"); }
        else asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();
        const unsigned char *buf = smem_raw + (i & 1) * U2B_BUF, *bs = buf + U2B_RH * U2B_RW * 32;
        const int k = (fz - 1) >> 1;   /* the fine plane feeds coarse k and k + 1 */
        const float wz0 = upc(fz, k, D), wz1 = upc(fz, k + 1, D);
        float t[8];
#pragma unroll
        for (int c = 0; c < 8; c++) t[c] = 0.f;
#pragma unroll
        for (int a = 0; a < 4; a++) {
            const int r = 2 * cy + a;   /* local fine row of 2 y - 1 + a */
#pragma unroll
            for (int b = 0; b < 4; b++) {
                const int p = 2 * cx + b;
                const uint2 u = *(const uint2 *)(buf + (r * U2B_RW + p) * 32 + 8 * cg);
                const float sc = mx_scale(bs[r * 40 + p + 3]) * (wy[a] * wx[b]);
                const float2 d0 = dec_e4m3x2((unsigned short)(u.x & 0xffffu)), d1 = dec_e4m3x2((unsigned short)(u.x >> 16));
                const float2 d2 = dec_e4m3x2((unsigned short)(u.y & 0xffffu)), d3 = dec_e4m3x2((unsigned short)(u.y >> 16));
                t[0] = fmaf(d0.x, sc, t[0]); t[1] = fmaf(d0.y, sc, t[1]); t[2] = fmaf(d1.x, sc, t[2]); t[3] = fmaf(d1.y, sc, t[3]);
                t[4] = fmaf(d2.x, sc, t[4]); t[5] = fmaf(d2.y, sc, t[5]); t[6] = fmaf(d3.x, sc, t[6]); t[7] = fmaf(d3.y, sc, t[7]);
            }
        }
#pragma unroll
        for (int c = 0; c < 8; c++) { acc0[c] = fmaf(wz0, t[c], acc0[c]); acc1[c] = fmaf(wz1, t[c], acc1[c]); }
        if ((fz & 1) == 0) {   /* fine 2 k + 2 done: coarse k complete (k < z0: the previous chunk's) */
            if (k >= z0 && x < W && y < H) {
                unsigned am = 0u;
#pragma unroll
                for (int c = 0; c < 8; c++) am = amax_u(am, acc0[c]);
                am = max(am, __shfl_xor_sync(0xffffffffu, am, 1)); am = max(am, __shfl_xor_sync(0xffffffffu, am, 2));
                const int e = mx_exp(__uint_as_float(am), 1.f / 448.f);
                const float m = exp2i(-e);
                const size_t ri = ((size_t)n * nbx + ob) * S + ((size_t)k * H + y) * W + x;
                *(uint2 *)(gx + ri * 32 + 8 * cg) = make_uint2(cvt_e4m3x4(acc0[0] * m, acc0[1] * m, acc0[2] * m, acc0[3] * m), cvt_e4m3x4(acc0[4] * m, acc0[5] * m, acc0[6] * m, acc0[7] * m));
                if (cg == 0) gxs[ri] = (uint8_t)(e + 127);
            } else if (k >= z0) {   /* keep the shuffles convergent (outside lanes) */
                unsigned am = 0u;
                am = max(am, __shfl_xor_sync(0xffffffffu, am, 1)); am = max(am, __shfl_xor_sync(0xffffffffu, am, 2));
            }
#pragma unroll
            for (int c = 0; c < 8; c++) { acc0[c] = acc1[c]; acc1[c] = 0.f; }
        }
        __syncthreads();   /* the buffer is restaged two planes on */
    }
}
static int up2b_tile_on(void) { static int on = -1; if (on < 0) on = getenv("UFSM_UP2B_TILE") ? atoi(getenv("UFSM_UP2B_TILE")) : 1; return on; }
static void up2_bwd_tile(const void *gy, int nby, int yb0, shape5 xs, void *gx, int nbx, int ob0, int nob) {   /* xs: coarse */
    int ZC = 8;
    const dim3 grid((unsigned)nblk_(xs.w, 16), (unsigned)nblk_(xs.h, 4), (unsigned)(nblk_(xs.d, ZC) * xs.n * nob));
    static int attr[8];
    if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)up2_bwd_mx_tile_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * U2B_BUF); }
    up2_bwd_mx_tile_k<<<grid, 256, 2 * U2B_BUF>>>((const uint8_t *)gy, (uint8_t *)gx, xs.n, nby, nbx, yb0, ob0, nob, xs.d, xs.h, xs.w, ZC);
}
extern "C" void lp_up2_bwd_mx(const void *gy, shape5 xs, void *gx) {
    if (up2b_tile_on() && xs.c % 32 == 0 && xs.w % 2 == 0) { up2_bwd_tile(gy, mx_nb(xs.c), 0, xs, gx, mx_nb(xs.c), 0, mx_nb(xs.c)); LPCK(); return; }
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
    if (up2b_tile_on() && bwx == 32 && bwy == 32 && c0 % 32 == 0 && nc % 32 == 0 && xs.w % 2 == 0) {   /* whole 32-channel blocks, written fresh */
        up2_bwd_tile(gy, mx_nb(nc), 0, xs, gx, mx_nb(ctot), ob0, nob); LPCK(); return;
    }
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
