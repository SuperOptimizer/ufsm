/* FP8 stride-2 forward and weight-gradient kernels. */
#include "lp_common.cuh"

/* ======================= FP8 stride-2 forward (k=3, pad 1) =======================
   Output tile 2 z x 4 y x 8 x per block; warp w owns z = w/4, row w%4 and the single 8-voxel n-tile. The input tile
   (5 x 9 x 17 positions x 32 channels) is stored with each row parity-split (even x first: 9 entries, then odd x: 8),
   so the stride-2 voxel columns of every tap are consecutive positions and the 16-byte XOR swizzle keeps ldmatrix
   conflict-free. Scales as in the stride-1 kernel: per (position, 32 channels) for the input, per (tap, co, chunk)
   for the weights (same prep). */
#define S2F8_T 765   /* 5 * 9 * 17 */
/* 8 e2m1 nibbles (channel k = nibble k) -> 8 e4m3 bytes (exact: the e2m1 grid {0, .5, 1, 1.5, 2, 3, 4, 6} is a subset of e4m3) */
__device__ __forceinline__ unsigned e2m1_e4m3(unsigned nb) { return (__byte_perm(0x3C383000u, 0x4C484440u, nb & 7u) & 0xffu) | ((nb & 8u) << 4); }
__device__ __forceinline__ uint2 nib8_e4m3(unsigned w) {
    unsigned lo = 0u, hi = 0u;
#pragma unroll
    for (int k = 0; k < 4; k++) { lo |= e2m1_e4m3(w >> (4 * k)) << (8 * k); hi |= e2m1_e4m3(w >> (16 + 4 * k)) << (8 * k); }
    return make_uint2(lo, hi);
}
__device__ __forceinline__ int s2pos(int row, int x) { return row * 17 + ((x & 1) ? 9 + (x >> 1) : (x >> 1)); }
template <int MT, typename T, bool P16 = false>   /* P16: <= 16 input channels, two taps per K block ([kx0 | kx1], [kx2 | 0]) */
__global__ void __launch_bounds__(256, 2) conv_fwd_s2_f8_k(const T *__restrict__ x, const uint8_t *__restrict__ wq, const uint8_t *__restrict__ wsc,
                                                        const float *__restrict__ b, T *__restrict__ y,
                                                        int N, int Ci, int D, int H, int W, int Co, int Cop, int Cip, int Do, int Ho, int Wo, gnp_t gp) {
    constexpr int BM = MT * 16, TG = 9;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    uint8_t *sx = smem_raw;                         /* [S2F8_T pos][32 ci] swizzled */
    uint8_t *sxs = sx + S2F8_T * 32;                /* [S2F8_T] (768) */
    uint8_t *wa = sxs + 768;                        /* [TG][BM][32] swizzled */
    uint8_t *was = wa + TG * BM * 32;               /* [TG][BM] */
    chan_t *ctab = (chan_t *)(was + ((TG * BM + 15) & ~15));   /* [32] (input transform path) */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wz = warp >> 2, wr = warp & 3;
    const int ox0 = blockIdx.x * 8, oy0 = blockIdx.y * 4;
    const bool G = gp.G != 0;
    const split_t nsp = {};
    int bz = blockIdx.z;
    const int nzt = (Do + 1) / 2;
    const int oz0 = (bz % nzt) * 2; bz /= nzt;
    const int nmt = Cop / BM;
    const int co0 = (bz % nmt) * BM; const int n = bz / nmt;
    const int nch = Cip / 32;
    float acc[MT][4];
#pragma unroll
    for (int m = 0; m < MT; m++) for (int k = 0; k < 4; k++) acc[m][k] = 0.f;
    const size_t plane = (size_t)D * H * W;
    if constexpr (P16) {   /* staged [5 x 9 rows][17 parity-split positions][16 B], one e4m3 scale per row; weights as prep_w8p_k */
        uint8_t *px = smem_raw;                        /* [S2F8_T][16] */
        uint8_t *zrow = px + S2F8_T * 16;              /* [16] zeros */
        uint8_t *rs = zrow + 16;                       /* [45] row scales (64) */
        unsigned *ram = (unsigned *)(rs + 64);         /* [45] row amax (64) */
        uint8_t *pa = (uint8_t *)(ram + 64);           /* [18][BM][32] swizzled */
        uint8_t *pas = pa + 18 * BM * 32;              /* [18][BM] */
        chan_t *pc = (chan_t *)(pas + ((18 * BM + 15) & ~15));   /* [16] */
        for (int i = threadIdx.x; i < 18 * BM * 2; i += 256) {
            const int ks = i / (BM * 2), r = i % (BM * 2), c = r >> 1, h = r & 1;
            *(uint4 *)(pa + ks * BM * 32 + sw16(c, h)) = __ldg((const uint4 *)(wq + ((size_t)ks * Cop + co0 + c) * 32 + h * 16));
        }
        for (int i = threadIdx.x; i < 18 * BM; i += 256) pas[i] = wsc[(size_t)(i / BM) * Cop + co0 + i % BM];
        if (threadIdx.x < 16) pc[threadIdx.x] = make_chan((int)threadIdx.x, Ci, Ci, n, plane, x, nsp, gp, N);
        if (threadIdx.x < 4) ((unsigned *)zrow)[threadIdx.x] = 0u;
        if (threadIdx.x < 64) ram[threadIdx.x] = 0u;
        __syncthreads();
        const chan_t c0 = pc[0];
        float2 gab[16];
#pragma unroll
        for (int k = 0; k < 16; k++) gab[k] = make_float2(pc[k].a, pc[k].b);
        float v[3][16];
#pragma unroll
        for (int it = 0; it < 3; it++) {
            const int p = threadIdx.x + 256 * it, row = p / 17, c = p - row * 17, ix = c < 9 ? 2 * c : 2 * (c - 9) + 1;
            const int iz = row / 9, iy = row - iz * 9, gz = 2 * oz0 - 1 + iz, gyy = 2 * oy0 - 1 + iy, gx = 2 * ox0 - 1 + ix;
            const bool inb = p < S2F8_T && gz >= 0 && gz < D && gyy >= 0 && gyy < H && gx >= 0 && gx < W;
            stage_row16<T>(c0, Ci, plane, inb ? ((size_t)gz * H + gyy) * W + gx : 0, inb, G, gab, v[it]);
            unsigned am = 0u;
#pragma unroll
            for (int k = 0; k < 16; k++) { const unsigned u = __float_as_uint(fabsf(v[it][k])); am = u <= 0x7f800000u ? max(am, u) : am; }
            if (p < S2F8_T && am) atomicMax(&ram[row], am);
        }
        __syncthreads();
#pragma unroll
        for (int it = 0; it < 3; it++) {
            const int p = threadIdx.x + 256 * it, row = p / 17, c = p - row * 17;
            if (p >= S2F8_T) break;
            const int e = mx_exp(__uint_as_float(ram[row]), 1.f / 448.f);
            const float m = exp2i(-e);
            const float *w = v[it];
            *(uint4 *)(px + p * 16) = make_uint4(cvt_e4m3x4(w[0] * m, w[1] * m, w[2] * m, w[3] * m), cvt_e4m3x4(w[4] * m, w[5] * m, w[6] * m, w[7] * m),
                                                 cvt_e4m3x4(w[8] * m, w[9] * m, w[10] * m, w[11] * m), cvt_e4m3x4(w[12] * m, w[13] * m, w[14] * m, w[15] * m));
            if (c == 0) rs[row] = (uint8_t)(e + 127);
        }
        __syncthreads();
        const int mat = lane >> 3, kh = mat & 1, vx = lane & 7;
#pragma unroll 3
        for (int tr = 0; tr < 9; tr++) {
            const int kz = tr / 3, ky = tr % 3, row = (wz * 2 + kz) * 9 + wr * 2 + ky;
            unsigned bA[4], bB[4];
            ldsm_x4(bA, px + (row * 17 + (kh ? 9 : 0) + vx) * 16);          /* [x 2j | x 2j+1] = [E j | O j] */
            ldsm_x4(bB, kh ? zrow : px + (row * 17 + vx + 1) * 16);         /* [x 2j+2 | 0] = [E j+1 | 0] */
            const unsigned sb = rs[row];
#pragma unroll
            for (int m = 0; m < MT; m++) {
                const int ar = m * 16 + (mat & 1) * 8 + (lane & 7);
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    unsigned af[4];
                    ldsm_x4(af, pa + (2 * tr + h) * BM * 32 + sw16(ar, mat >> 1));
                    mma_f8(acc[m], af, h ? bB : bA, pas[(2 * tr + h) * BM + m * 16 + g + 8 * (t & 1)], sb);
                }
            }
        }
    } else
    for (int ci0 = 0; ci0 < Cip; ci0 += 32) {
        __syncthreads();
        if (G) { if (threadIdx.x < 32) ctab[threadIdx.x] = make_chan(ci0 + threadIdx.x, Ci, Ci, n, plane, x, nsp, gp, N); __syncthreads(); }
        for (int p = threadIdx.x; p < S2F8_T; p += 256) {   /* p = row * 17 + parity-split column */
            int row = p / 17, c = p - row * 17, ix = c < 9 ? 2 * c : 2 * (c - 9) + 1;
            int iz = row / 9, iy = row - iz * 9;
            int gz = 2 * oz0 - 1 + iz, gyy = 2 * oy0 - 1 + iy, gx = 2 * ox0 - 1 + ix;
            bool inb = gz >= 0 && gz < D && gyy >= 0 && gyy < H && gx >= 0 && gx < W;
            size_t off = ((size_t)gz * H + gyy) * W + gx;
            if (IS_MX(T) && !G) {   /* MX input: the 32-channel chunk is one stored block -> copy (mx4: nibbles -> e4m3 bytes, exact) */
                const int bw = mx_bw(Ci), nb = mx_nb(Ci), blk = ci0 / 32, rb = mx_rb(bw, MX_BITS(T));
                uint4 h0 = make_uint4(0u, 0u, 0u, 0u), h1 = h0;
                unsigned sc = 1u;
                if (inb) {
                    const uint8_t *q = (const uint8_t *)x;
                    const size_t ri = ((size_t)n * nb + blk) * plane + off;
                    if constexpr (IS_MX8(T)) { const uint4 *src = (const uint4 *)(q + ri * bw); h0 = __ldg(src); if (bw == 32) h1 = __ldg(src + 1); }
                    else {
                        const uint2 *src = (const uint2 *)(q + ri * rb);
                        const uint2 lo = __ldg(src), hi = bw == 32 ? __ldg(src + 1) : make_uint2(0u, 0u);
                        const uint2 a = nib8_e4m3(lo.x), b2 = nib8_e4m3(lo.y), c = nib8_e4m3(hi.x), d = nib8_e4m3(hi.y);
                        h0 = make_uint4(a.x, a.y, b2.x, b2.y); h1 = make_uint4(c.x, c.y, d.x, d.y);
                    }
                    sc = q[(size_t)N * nb * plane * rb + ri];
                }
                *(uint4 *)(sx + sw16(p, 0)) = h0;
                *(uint4 *)(sx + sw16(p, 1)) = h1;
                sxs[p] = (uint8_t)sc;
                continue;
            }
            float v[32], amax = 0.f;
            if (IS_MX(T)) mx_row32<IS_MX(T) ? MX_BITS(T) : 8>(ctab[0], inb ? off : 0, inb, v);   /* (G here: the copy path took !G) */
#pragma unroll
            for (int k = 0; k < 32; k++) {
                int ci = ci0 + k;
                if (IS_MX(T)) { const chan_t &c = ctab[k]; v[k] = inb && c.p ? act_ab(v[k], c.a, c.b, true) : 0.f; }
                else if (G) { const chan_t c = ctab[k]; v[k] = inb && c.p ? act_ab(ldc<T>(c, off), c.a, c.b, true) : 0.f; }
                else v[k] = inb && ci < Ci ? ldx(x + ((size_t)n * Ci + ci) * plane, off) : 0.f;
                amax = fmaxf(amax, fabsf(v[k]));
            }
            int e = mx_exp(amax, 1.f / 448.f);
            float m = exp2i(-e);
            *(uint4 *)(sx + sw16(p, 0)) = make_uint4(cvt_e4m3x4(v[0] * m, v[1] * m, v[2] * m, v[3] * m), cvt_e4m3x4(v[4] * m, v[5] * m, v[6] * m, v[7] * m),
                                                     cvt_e4m3x4(v[8] * m, v[9] * m, v[10] * m, v[11] * m), cvt_e4m3x4(v[12] * m, v[13] * m, v[14] * m, v[15] * m));
            *(uint4 *)(sx + sw16(p, 1)) = make_uint4(cvt_e4m3x4(v[16] * m, v[17] * m, v[18] * m, v[19] * m), cvt_e4m3x4(v[20] * m, v[21] * m, v[22] * m, v[23] * m),
                                                     cvt_e4m3x4(v[24] * m, v[25] * m, v[26] * m, v[27] * m), cvt_e4m3x4(v[28] * m, v[29] * m, v[30] * m, v[31] * m));
            sxs[p] = (uint8_t)(e + 127);
        }
        for (int t0 = 0; t0 < 27; t0 += TG) {
            if (t0) __syncthreads();
            for (int i = threadIdx.x; i < TG * BM * 2; i += 256) {
                int tt = i / (BM * 2), r = i % (BM * 2), c = r >> 1, h = r & 1;
                *(uint4 *)(wa + tt * BM * 32 + sw16(c, h)) = __ldg((const uint4 *)(wq + ((size_t)(t0 + tt) * Cop + co0 + c) * Cip + ci0 + h * 16));
            }
            for (int i = threadIdx.x; i < TG * BM; i += 256) { int tt = i / BM, c = i % BM; was[i] = wsc[((size_t)(t0 + tt) * Cop + co0 + c) * nch + ci0 / 32]; }
            __syncthreads();
#pragma unroll
            for (int tt = 0; tt < TG; tt++) {
                int tap = t0 + tt, kz = tap / 9, ky = (tap / 3) % 3, kx = tap % 3;
                int row = (wz * 2 + kz) * 9 + wr * 2 + ky;
                unsigned bfr[4];
                {   /* x4: (k half 0, voxels 0..7), (k half 1, voxels 0..7), duplicated (only b0, b1 used) */
                    int mat = lane >> 3, kh = mat & 1, vx = lane & 7;
                    ldsm_x4(bfr, sx + sw16(s2pos(row, 2 * vx + kx), kh));
                }
                unsigned sb = sxs[s2pos(row, 2 * g + kx)];
#pragma unroll
                for (int m = 0; m < MT; m++) {
                    unsigned af[4];
                    int mat = lane >> 3, r = m * 16 + (mat & 1) * 8 + (lane & 7);
                    ldsm_x4(af, wa + tt * BM * 32 + sw16(r, mat >> 1));
                    unsigned sa = was[tt * BM + m * 16 + g + 8 * (t & 1)];
                    mma_f8(acc[m], af, bfr, sa, sb);
                }
            }
        }
    }
    if constexpr (IS_MX(T)) {   /* MX output (blocks of 32 / 16 channels per voxel); fp4: nibble pairs across lanes g, g + 1 as fwd_epilogue */
        constexpr int B = MX_BITS(T);
        const int bw = mx_bw(Co), nb = mx_nb(Co), rb = mx_rb(bw, B);
        const size_t So = (size_t)Do * Ho * Wo;
        const int oz = oz0 + wz, oy = oy0 + wr;
        uint8_t *q = (uint8_t *)y, *scp = q + (size_t)N * nb * So * rb;
#pragma unroll
        for (int vv = 0; vv < 2; vv++) {
            const int ox = ox0 + 2 * t + vv;
            const bool ok = oz < Do && oy < Ho && ox < Wo;
#pragma unroll
            for (int j = 0; j < MT; j++) {
                if (j * 16 % bw) continue;
                float val[2][2], am = 0.f;
#pragma unroll
                for (int mm = 0; mm < 2; mm++)
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const int m = j + mm, co = co0 + m * 16 + g + 8 * h;
                        float v = 0.f;
                        if (mm < bw / 16 && m < MT) v = acc[m < MT ? m : 0][2 * h + vv] + (b && co < Co ? b[co] : 0.f);
                        val[mm][h] = v; am = fmaxf(am, fabsf(v));
                    }
                am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, 4)); am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, 8)); am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, 16));
                const int e = mx_exp(am, mxf<B>::inv_qmax);
                const float mult = exp2i(-e);
                unsigned nib = 0u;
                if constexpr (B == 4) {
#pragma unroll
                    for (int mm = 0; mm < 2; mm++)
#pragma unroll
                        for (int h = 0; h < 2; h++) nib |= (unsigned)(cvt_e2m1x2(val[mm][h] * mult, 0.f) & 15) << (8 * (2 * mm + h));
                    nib |= __shfl_down_sync(0xffffffff, nib, 4) << 4;   /* channel g + 1 into the high nibbles */
                }
                if (ok) {
                    const int blk = (co0 + j * 16) / bw;
                    const size_t v = ((size_t)oz * Ho + oy) * Wo + ox;
                    uint8_t *dst = q + (((size_t)n * nb + blk) * So + v) * rb;
#pragma unroll
                    for (int mm = 0; mm < 2; mm++)
#pragma unroll
                        for (int h = 0; h < 2; h++) if (mm < bw / 16) {
                            if constexpr (B == 8) dst[mm * 16 + g + 8 * h] = cvt_e4m3(val[mm][h] * mult);
                            else if (!(g & 1)) dst[(mm * 16 + g + 8 * h) >> 1] = (uint8_t)(nib >> (8 * (2 * mm + h)));
                        }
                    if (g == 0) scp[((size_t)n * nb + blk) * So + v] = (uint8_t)(e + 127);
                }
            }
        }
        return;
    }
#pragma unroll
    for (int m = 0; m < MT; m++) {
        int oz = oz0 + wz, oy = oy0 + wr;
        if (oz >= Do || oy >= Ho) continue;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            int co = co0 + m * 16 + g + 8 * h;
            if (co >= Co) continue;
            float bias = b ? b[co] : 0.f;
            T *yp = y + (((size_t)n * Co + co) * Do + oz) * Ho * Wo + (size_t)oy * Wo;
            int ox = ox0 + 2 * t;
            float v0 = acc[m][2 * h] + bias, v1 = acc[m][2 * h + 1] + bias;
            if (!(Wo & 1) && ox + 1 < Wo) stx2(yp, (size_t)ox, v0, v1);
            else { if (ox < Wo) stx(yp, (size_t)ox, v0); if (ox + 1 < Wo) stx(yp, (size_t)ox + 1, v1); }
        }
    }
}
template <int MT, typename T, bool P16 = false> static void launch_s2f8(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, int Cip, shape5 ys, gnp_t gp) {
    size_t smem = (size_t)S2F8_T * 32 + 768 + 9 * MT * 16 * 32 + ((9 * MT * 16 + 15) & ~15) + 32 * sizeof(chan_t);
    static int attr[8];
    if (P16) { const size_t sp16 = (size_t)S2F8_T * 16 + 16 + 64 + 256 + 18 * MT * 16 * 33 + 16 + 16 * sizeof(chan_t); if (sp16 > smem) smem = sp16; }
    if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)conv_fwd_s2_f8_k<MT, T, P16>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
    conv_fwd_s2_f8_k<MT, T, P16><<<grid, 256, smem>>>((const T *)x, wq, ws, b, (T *)y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, Cip, ys.d, ys.h, ys.w, gp);
}
extern "C" int lp_conv_fwd_s2_f8(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, shape5 ys, gnp_t gp) {
    lp_dtype_check("lp_conv_fwd_s2_f8", xbf, ybf);
    int Cop = (cout + 15) / 16 * 16, Cip = (xs.c + 31) / 32 * 32, nch = Cip / 32;
    if (xbf >= 3 && cout > 16) Cop = (cout + 31) / 32 * 32;   /* MX output: 32-row-aligned m-tiles */
    static int p16 = -1;
    if (p16 < 0) p16 = getenv("UFSM_S2_PACK16") ? atoi(getenv("UFSM_S2_PACK16")) : 1;
    if (p16 && xs.c <= 16) {   /* down0: two taps per K block */
        uint8_t *wq = lp_buf<uint8_t>(0, (size_t)P16_KS * Cop * 32), *ws = lp_buf<uint8_t>(1, (size_t)P16_KS * Cop);
        prep_w8p_k<<<nblk_((size_t)P16_KS * Cop, 128), 128>>>(w, wq, ws, cout, xs.c, Cop, -1, 0);
        const int MT = Cop % 64 == 0 ? 4 : Cop % 32 == 0 ? 2 : 1, nmt = Cop / (MT * 16);
        dim3 grid(nblk_(ys.w, 8), nblk_(ys.h, 4), (unsigned)(nblk_(ys.d, 2) * nmt * xs.n));
#define S2P(MT_) do { if (xbf == 4) launch_s2f8<MT_, mx4_t, true>(grid, x, xs, wq, ws, b, cout, y, Cop, 32, ys, gp); else if (xbf == 3) launch_s2f8<MT_, mx8_t, true>(grid, x, xs, wq, ws, b, cout, y, Cop, 32, ys, gp); else if (xbf == 2) launch_s2f8<MT_, __half, true>(grid, x, xs, wq, ws, b, cout, y, Cop, 32, ys, gp); else if (xbf) launch_s2f8<MT_, bf16, true>(grid, x, xs, wq, ws, b, cout, y, Cop, 32, ys, gp); else launch_s2f8<MT_, float, true>(grid, x, xs, wq, ws, b, cout, y, Cop, 32, ys, gp); } while (0)
        switch (MT) { case 1: S2P(1); break; case 2: S2P(2); break; default: S2P(4); break; }
#undef S2P
        LPCK();
        return 0;
    }
    uint8_t *wq = lp_buf<uint8_t>(0, (size_t)27 * Cop * Cip), *ws = lp_buf<uint8_t>(1, (size_t)27 * Cop * nch);
    size_t nt = (size_t)27 * Cop * nch;
    prep_w8_k<<<nblk_(nt, 128), 128>>>(w, wq, ws, cout, xs.c, Cop, Cip);
    int MT = Cop % 64 == 0 ? 4 : Cop % 32 == 0 ? 2 : 1;
    int nmt = Cop / (MT * 16);
    dim3 grid(nblk_(ys.w, 8), nblk_(ys.h, 4), (unsigned)(nblk_(ys.d, 2) * nmt * xs.n));
#define S2L(MT_) do { if (xbf == 4) launch_s2f8<MT_, mx4_t>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, ys, gp); else if (xbf == 3) launch_s2f8<MT_, mx8_t>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, ys, gp); else if (xbf == 2) launch_s2f8<MT_, __half>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, ys, gp); else if (xbf) launch_s2f8<MT_, bf16>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, ys, gp); else launch_s2f8<MT_, float>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, ys, gp); } while (0)
    switch (MT) { case 1: S2L(1); break; case 2: S2L(2); break; default: S2L(4); break; }
#undef S2L
    LPCK();
    return 0;
}

/* ======================= FP8 stride-2 weight gradient (k=3, pad 1) =======================
   GW[co][ci][tap] = sum_o GY[co][o] X[ci][2 o + k - 1]. Same block / warp structure as conv_bwd_w_f8_k (9 warps, warp
   = (kz, ky) with all three kx; M = 16 MT output channels, N = 8 NT input channels, K = 32 output voxels = two output
   rows of one output plane), z-steps of 2 output planes x 8 rows x 16 columns. The input slab for a step is 5 planes x
   17 rows x 33 positions per channel; each row is stored parity-split (even positions E at bytes 0..16, odd O at
   20..35), so for output voxels 4t..4t+3 the tap kx = 0 / 1 / 2 operand is E[4t..] / O[4t..] / E[4t+1..] (one funnel
   shift). Scales: X per (channel, input plane) (both rows of a K block read one input plane for every tap), GY per
   (cout, 32 voxels). No GroupNorm input transform (the down convs read stored activations). */
#ifndef S2W_PROF
#define S2W_PROF 0   /* 1: per-phase clock64 sums of conv_bwd_w_s2_f8_k (x staging / gy staging / mma), printed per call when env S2W_PROF is set */
#endif
#if S2W_PROF
__device__ unsigned long long g_s2wprof[4];
#endif
#define XS2_CS 3088   /* per channel: 5 planes x 17 rows x 36 B = 3060, padded to 772 words == 4 mod 32 */
#define XS2_PS 612
template <int MT, int NT, typename T, typename TG>
__global__ void __launch_bounds__(288, 2) conv_bwd_w_s2_f8_k(const T *__restrict__ x, const TG *__restrict__ gy, float *__restrict__ gw, float *__restrict__ gb,
                                                          int N, int Ci, int D, int H, int W, int Co, int Do, int Ho, int Wo, int ZC, gnp_t gp) {
    constexpr int CH = 8 * NT, BMo = 16 * MT;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    uint8_t *sxq = smem_raw;                        /* [CH][XS2_CS] */
    uint8_t *sg = sxq + CH * XS2_CS;                /* [BMo][G8_CS] */
    uint8_t *sgs = sg + BMo * G8_CS;                /* [BMo][8 ksteps] */
    uint8_t *sxs = sgs + BMo * 8;                   /* [CH][5 ring slots] (8 reserved) */
    float *sbias = (float *)(sxs + CH * 8);         /* [BMo] */
    chan_t *ctab = (chan_t *)(smem_raw + (((unsigned char *)(sbias + BMo) - smem_raw + 31) & ~31));   /* [CH] MX x */
    unsigned *amx = (unsigned *)(ctab + CH);        /* [3][CH] MX x: per-channel plane amax (bits), three buffers */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int ci0 = blockIdx.x * CH, co0 = blockIdx.y * BMo;
    int bz = blockIdx.z;
    const int nxt = (Wo + 15) / 16, nyt = (Ho + 7) / 8, nzt = (Do + 1) / 2;
    const int ox0 = (bz % nxt) * 16; bz /= nxt;
    const int oy0 = (bz % nyt) * 8; bz /= nyt;
    const int nzc = (nzt + ZC - 1) / ZC;
    const int zc = bz % nzc; const int n = bz / nzc;
    const size_t plane = (size_t)D * H * W;
    const int kz = warp / 3, ky = warp % 3;
    const bool do_bias = gb && blockIdx.x == 0;
    const bool vec = (Wo & 3) == 0, vec8 = (W & 7) == 0;
    float acc[3][MT][NT][4];
#pragma unroll
    for (int a = 0; a < 3; a++) for (int m = 0; m < MT; m++) for (int q = 0; q < NT; q++) for (int k = 0; k < 4; k++) acc[a][m][q][k] = 0.f;
    if (threadIdx.x < BMo) sbias[threadIdx.x] = 0.f;
    /* MX x whose CH channels are consecutive entries of one stored block row: thread per voxel, the row decoded once for all
       CH channels (as conv_bwd_w_f8_k), instead of one strided byte per (channel, voxel) */
    bool uni = false;
    if constexpr (IS_MX(T)) {
        const split_t nsp = {};
        if (threadIdx.x < CH) { const int ci = ci0 + threadIdx.x; ctab[threadIdx.x] = make_chan(ci < Ci ? ci : Ci, Ci, Ci, n, plane, x, nsp, gp, N); }
        if (threadIdx.x < 3 * CH) amx[threadIdx.x] = 0u;
        __syncthreads();
        uni = ctab[0].p != nullptr && (ctab[0].nib & 7) == 0;
#pragma unroll
        for (int q = 1; q < CH; q++) if (ctab[q].p && (ctab[q].sp != ctab[0].sp || ctab[q].nib != ctab[0].nib + q)) uni = false;
    }
    int npass = 0;   /* uni staging passes so far (selects the amax buffer) */
    const int zt_begin = zc * ZC;
#if S2W_PROF
    unsigned long long pa = 0, pb_ = 0, pc = 0, t0 = 0, t1 = 0, t2 = 0;
#endif
    for (int zt = zc * ZC; zt < nzt && zt < (zc + 1) * ZC; zt++) {
        const int oz0 = zt * 2;
        __syncthreads();
#if S2W_PROF
        if (threadIdx.x == 0) t0 = clock64();
#endif
        /* X: warp per (channel, input plane). Row = halo position u = 0 (x = 2 ox0 - 1) + 4 vectors of 8 positions
           u = 8j + 1 .. 8j + 8 (x = 2 ox0 + 8j ..): the 4 odd u of a vector are one aligned word O[4j..4j+3], the 4 even
           u are bytes E[4j+1..4j+4]. Lane item f = lane + 32 i < 68 -> row f / 4, vector f % 4; lanes < 17 do the halo. */
        /* input plane gz lives in ring slot (gz + 1) % 5: consecutive z-steps share one plane (the first stages all 5) */
        if (IS_MX(T) && uni) {
            const chan_t c0 = ctab[0];
            const bool Gany = gp.G != 0;
            for (int iz = zt == zt_begin ? 0 : 1; iz < 5; iz++, npass++) {
                const int gz = 2 * oz0 - 1 + iz, slot = (gz + 1) % 5;
                unsigned *am = amx + (npass % 3) * CH;
                float v[2][CH];
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    const int idx = threadIdx.x + 288 * r, row = idx / 33, u = idx - 33 * row, gyy = 2 * oy0 - 1 + row, gx = 2 * ox0 - 1 + u;
#pragma unroll
                    for (int q = 0; q < CH; q++) v[r][q] = 0.f;
                    if (idx < 561 && gz >= 0 && gz < D && gyy >= 0 && gyy < H && gx >= 0 && gx < W) {
                        mx_rowch<T, CH>(c0, false, gz, gyy, gx, D, H, W, v[r]);
#pragma unroll
                        for (int q = 0; q < CH; q++) { const chan_t &cq = ctab[q]; v[r][q] = cq.p ? act_ab(v[r][q], cq.a, cq.b, Gany && cq.g) : 0.f; }
                    }
                }
#pragma unroll
                for (int q = 0; q < CH; q++) {
                    const unsigned a = __reduce_max_sync(0xffffffffu, max(__float_as_uint(v[0][q]) & 0x7fffffffu, __float_as_uint(v[1][q]) & 0x7fffffffu));
                    if (lane == 0 && a) atomicMax(&am[q], a);
                }
                __syncthreads();
                float mq[CH];   /* the channels' scale multipliers, read once (the stores below could alias am for the compiler) */
#pragma unroll
                for (int q = 0; q < CH; q++) mq[q] = exp2i(-mx_exp(__uint_as_float(am[q]), 1.f / 448.f));
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    const int idx = threadIdx.x + 288 * r, row = idx / 33, u = idx - 33 * row;
                    if (idx < 561) {   /* position u: even -> E[u / 2] (byte u / 2), odd -> O[(u - 1) / 2] (byte 20 + (u - 1) / 2) */
                        uint8_t *dst = sxq + slot * XS2_PS + row * 36 + ((u & 1) ? 20 + (u >> 1) : u >> 1);
                        unsigned char b[CH];
#pragma unroll
                        for (int q = 0; q < CH; q++) b[q] = cvt_e4m3(v[r][q] * mq[q]);
#pragma unroll
                        for (int q = 0; q < CH; q++) dst[q * XS2_CS] = b[q];
                    }
                }
                if (threadIdx.x < CH) { sxs[threadIdx.x * 8 + slot] = (uint8_t)(mx_exp(__uint_as_float(am[threadIdx.x]), 1.f / 448.f) + 127); amx[((npass + 2) % 3) * CH + threadIdx.x] = 0u; }
            }
        } else
        for (int task = warp; task < CH * 5; task += 9) {
            int k = task / 5, iz = task - 5 * k, ci = ci0 + k, gz = 2 * oz0 - 1 + iz;
            const bool ok = ci < Ci && gz >= 0 && gz < D;
            const T *xc = IS_MX(T) ? x : x + ((size_t)n * Ci + (ok ? ci : 0)) * plane + (size_t)(ok ? gz : 0) * H * W;
            const split_t nsp = {};
            const chan_t cm = make_chan(ok ? ci : Ci, Ci, Ci, n, plane, x, nsp, gp, N);   /* MX element access, gn+silu coefficients */
            float v[3][8], vh = 0.f, amax = 0.f;
#pragma unroll
            for (int i = 0; i < 3; i++) {
                int f = lane + 32 * i, iy = f >> 2, j = f & 3, gyy = 2 * oy0 - 1 + iy, gx = 2 * ox0 + 8 * j;
#pragma unroll
                for (int e = 0; e < 8; e++) v[i][e] = 0.f;
                if (ok && f < 68 && gyy >= 0 && gyy < H) {
                    const T *src = xc + (size_t)gyy * W + gx;
                    if constexpr (IS_MX(T)) {   /* two 4-voxel reads (index math once, the scale bytes as one word) */
                        const size_t vo = ((size_t)gz * H + gyy) * W + gx;
                        const float4 a = ldc4_mx<T>(cm, vo, max(0, min(4, W - gx)), vec8), b = ldc4_mx<T>(cm, vo + 4, max(0, min(4, W - gx - 4)), vec8);
                        v[i][0] = a.x; v[i][1] = a.y; v[i][2] = a.z; v[i][3] = a.w; v[i][4] = b.x; v[i][5] = b.y; v[i][6] = b.z; v[i][7] = b.w;
                    }
                    else if (vec8 && gx + 8 <= W) { float tmp[8]; ld8x<T>(src, tmp); for (int e = 0; e < 8; e++) v[i][e] = tmp[e]; }
                    else { for (int e = 0; e < 8; e++) if (gx + e < W) v[i][e] = ldx(src, e); }
                    if (cm.g) {
#pragma unroll
                        for (int e = 0; e < 8; e++) { const float y = act_ab(v[i][e], cm.a, cm.b, true); v[i][e] = gx + e < W ? y : 0.f; }
                    }
                }
#pragma unroll
                for (int e = 0; e < 8; e++) amax = fmaxf(amax, fabsf(v[i][e]));
            }
            {
                int gyy = 2 * oy0 - 1 + lane, gx = 2 * ox0 - 1;
                if (ok && lane < 17 && gyy >= 0 && gyy < H && gx >= 0) vh = act_ab(IS_MX(T) ? ldc<T>(cm, ((size_t)gz * H + gyy) * W + gx) : ldx(xc, (size_t)gyy * W + gx), cm.a, cm.b, cm.g);
                amax = fmaxf(amax, fabsf(vh));
            }
#pragma unroll
            for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
            int e = mx_exp(amax, 1.f / 448.f);
            float m = exp2i(-e);
            uint8_t *dst = sxq + k * XS2_CS + ((gz + 1) % 5) * XS2_PS;
#pragma unroll
            for (int i = 0; i < 3; i++) {
                int f = lane + 32 * i, iy = f >> 2, j = f & 3;
                if (f < 68) {
                    uint8_t *r = dst + iy * 36;
                    *(unsigned *)(r + 20 + 4 * j) = cvt_e4m3x4(v[i][0] * m, v[i][2] * m, v[i][4] * m, v[i][6] * m);   /* O[4j..4j+3] */
                    unsigned ev = cvt_e4m3x4(v[i][1] * m, v[i][3] * m, v[i][5] * m, v[i][7] * m);                     /* E[4j+1..4j+4] */
                    r[4 * j + 1] = (uint8_t)ev; r[4 * j + 2] = (uint8_t)(ev >> 8); r[4 * j + 3] = (uint8_t)(ev >> 16); r[4 * j + 4] = (uint8_t)(ev >> 24);
                }
            }
            if (lane < 17) dst[lane * 36] = cvt_e4m3(vh * m);                                                             /* E[0] */
            if (lane == 0) sxs[k * 8 + (gz + 1) % 5] = (uint8_t)(e + 127);
        }
#if S2W_PROF
        __syncthreads();
        if (threadIdx.x == 0) t1 = clock64();
#endif
        /* GY: K block (co, ks) = 2 rows x 16 voxels = 8 x 4 voxels; warp covers 4 blocks, 8 lanes each */
        for (int task = warp; task < BMo * 2; task += 9) {
            int blk = task * 4 + (lane >> 3), c = blk >> 3, ks = blk & 7, f = lane & 7;
            int row = ks * 2 + (f >> 2), vx = 4 * (f & 3), vz = row >> 3, vy = row & 7;
            int co = co0 + c, oz = oz0 + vz, oy = oy0 + vy, ox = ox0 + vx;
            float4 vv = make_float4(0.f, 0.f, 0.f, 0.f);
            if (co < Co && oz < Do && oy < Ho) {
                if constexpr (IS_MX8(TG)) {   /* 4 voxels of channel co: index math once, the scale bytes as one word when aligned */
                    const size_t vo = ((size_t)oz * Ho + oy) * Wo + ox, Sg = (size_t)Do * Ho * Wo;
                    const int bw = mx_bw(Co), nbk = mx_nb(Co), nv = max(0, min(4, Wo - ox));
                    const size_t ri = ((size_t)n * nbk + co / bw) * Sg + vo;
                    const uint8_t *qd = (const uint8_t *)gy + ri * bw + co % bw, *qs = (const uint8_t *)gy + (size_t)N * nbk * Sg * bw + ri;
                    unsigned sw = 0u;
                    if (vec && nv == 4) sw = __ldg((const unsigned *)qs);
                    else for (int j = 0; j < nv; j++) sw |= (unsigned)qs[j] << (8 * j);
                    const unsigned b0 = nv > 0 ? __ldg(qd) : 0u, b1 = nv > 1 ? __ldg(qd + bw) : 0u, b2 = nv > 2 ? __ldg(qd + 2 * bw) : 0u, b3 = nv > 3 ? __ldg(qd + 3 * bw) : 0u;
                    const float2 d0 = dec_e4m3x2((unsigned short)(b0 | b1 << 8)), d1 = dec_e4m3x2((unsigned short)(b2 | b3 << 8));
                    vv = make_float4(d0.x * mx_scale(sw & 255u), d0.y * mx_scale(sw >> 8 & 255u), d1.x * mx_scale(sw >> 16 & 255u), d1.y * mx_scale(sw >> 24));
                } else {
                const TG *src = gy + (((size_t)n * Co + co) * Do + oz) * Ho * Wo + (size_t)oy * Wo + ox;
                if (vec) { if (ox < Wo) vv = ldx4<TG>(src); }
                else { if (ox < Wo) vv.x = ldx(src, 0); if (ox + 1 < Wo) vv.y = ldx(src, 1); if (ox + 2 < Wo) vv.z = ldx(src, 2); if (ox + 3 < Wo) vv.w = ldx(src, 3); }
                }
            }
            float amax = fmaxf(fmaxf(fabsf(vv.x), fabsf(vv.y)), fmaxf(fabsf(vv.z), fabsf(vv.w))), sm = (vv.x + vv.y) + (vv.z + vv.w);
#pragma unroll
            for (int o = 4; o; o >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o)); sm += __shfl_xor_sync(0xffffffff, sm, o); }
            int e = mx_exp(amax, 1.f / 448.f);
            float m = exp2i(-e);
            *(unsigned *)(sg + c * G8_CS + row * 16 + vx) = cvt_e4m3x4(vv.x * m, vv.y * m, vv.z * m, vv.w * m);
            if (f == 0) { sgs[c * 8 + ks] = (uint8_t)(e + 127); if (do_bias) atomicAdd(&sbias[c], sm); }
        }
        __syncthreads();
#if S2W_PROF
        if (threadIdx.x == 0) t2 = clock64();
#endif
#pragma unroll
        for (int vz = 0; vz < 2; vz++) {
            const int iz = (2 * oz0 + 2 * vz + kz) % 5;   /* ring slot of input plane 2 oz0 - 1 + 2 vz + kz */
            unsigned sb[NT];
#pragma unroll
            for (int q = 0; q < NT; q++) sb[q] = sxs[(q * 8 + g) * 8 + iz];
#pragma unroll 2
            for (int kk = 0; kk < 4; kk++) {
                const int ks = vz * 4 + kk, vy0 = 2 * kk;
                unsigned af[MT][4], sa[MT];
#pragma unroll
                for (int m = 0; m < MT; m++) {
                    int mat = lane >> 3, co = m * 16 + (mat & 1) * 8 + (lane & 7), row = 2 * ks + (mat >> 1);
                    ldsm_x4(af[m], sg + co * G8_CS + row * 16);
                    sa[m] = sgs[(m * 16 + g + 8 * (t & 1)) * 8 + ks];
                }
                const int iy0 = 2 * vy0 + ky;
#pragma unroll
                for (int q = 0; q < NT; q++) {
                    const unsigned *p0 = (const unsigned *)(sxq + (q * 8 + g) * XS2_CS + iz * XS2_PS + iy0 * 36) + t;
                    const unsigned *p1 = p0 + 18;                 /* input row iy0 + 2 (= next output row): +72 B */
                    unsigned e0 = p0[0], e1 = p0[1], o0 = p0[5], f0 = p1[0], f1 = p1[1], o1 = p1[5];
                    unsigned b[3][2] = {{e0, f0}, {o0, o1}, {__funnelshift_r(e0, e1, 8), __funnelshift_r(f0, f1, 8)}};
#pragma unroll
                    for (int kx = 0; kx < 3; kx++)
#pragma unroll
                        for (int m = 0; m < MT; m++) mma_f8(acc[kx][m][q], af[m], b[kx], sa[m], sb[q]);
                }
            }
        }
#if S2W_PROF
        __syncthreads();
        if (threadIdx.x == 0) { const unsigned long long t3 = clock64(); pa += t1 - t0; pb_ += t2 - t1; pc += t3 - t2; }
#endif
    }
#if S2W_PROF
    if (threadIdx.x == 0) { atomicAdd(&g_s2wprof[0], pa); atomicAdd(&g_s2wprof[1], pb_); atomicAdd(&g_s2wprof[2], pc); atomicAdd(&g_s2wprof[3], 1ull); }
#endif
    if (do_bias) { __syncthreads(); if (threadIdx.x < BMo && co0 + (int)threadIdx.x < Co) atomicAdd(&gb[co0 + threadIdx.x], sbias[threadIdx.x]); }
#pragma unroll
    for (int kx = 0; kx < 3; kx++) {
        int tap = (kz * 3 + ky) * 3 + kx;
#pragma unroll
        for (int m = 0; m < MT; m++)
#pragma unroll
            for (int q = 0; q < NT; q++) {
                int ci = ci0 + q * 8 + 2 * t;
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    int co = co0 + m * 16 + g + 8 * h;
                    if (co >= Co) continue;
                    if (ci < Ci) atomicAdd(&gw[((size_t)co * Ci + ci) * 27 + tap], acc[kx][m][q][2 * h]);
                    if (ci + 1 < Ci) atomicAdd(&gw[((size_t)co * Ci + ci + 1) * 27 + tap], acc[kx][m][q][2 * h + 1]);
                }
            }
    }
}
template <int MT, int NT, typename T, typename TG> static void launch_bws2(dim3 grid, size_t smem, const void *x, shape5 xs, const TG *gy, shape5 ys, float *gw, float *gb, int ZC, gnp_t gp) {
    static int attr[8];
    if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)conv_bwd_w_s2_f8_k<MT, NT, T, TG>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
    conv_bwd_w_s2_f8_k<MT, NT, T, TG><<<grid, 288, smem>>>((const T *)x, gy, gw, gb, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, ys.d, ys.h, ys.w, ZC, gp);
}
template <typename T, typename TG> static void bwd_w_s2_f8_t(const void *x, shape5 xs, const TG *gy, shape5 ys, float *gw, float *gb, gnp_t gp) {
    int MT = ys.c >= 32 ? 2 : 1, NT = xs.c <= 8 ? 1 : 2;
    size_t smem = (size_t)8 * NT * XS2_CS + 16 * MT * (G8_CS + 8) + 8 * NT * 8 + 16 * MT * 4 + 32 + (size_t)8 * NT * (sizeof(chan_t) + 3 * 4);
    int nzt = nblk_(ys.d, 2), base = (int)(((xs.c + 8 * NT - 1) / (8 * NT)) * ((ys.c + 16 * MT - 1) / (16 * MT)) * nblk_(ys.w, 16) * nblk_(ys.h, 8) * ys.n);
    int ZC = nzt < 8 ? nzt : 8;
    while (ZC > 1 && (size_t)base * nblk_(nzt, ZC) < 72) ZC--;
    dim3 grid((xs.c + 8 * NT - 1) / (8 * NT), (ys.c + 16 * MT - 1) / (16 * MT), (unsigned)(nblk_(ys.w, 16) * nblk_(ys.h, 8) * nblk_(nzt, ZC) * ys.n));
    switch (MT * 10 + NT) {
    case 11: launch_bws2<1, 1, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, ZC, gp); break;
    case 12: launch_bws2<1, 2, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, ZC, gp); break;
    case 21: launch_bws2<2, 1, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, ZC, gp); break;
    default: launch_bws2<2, 2, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, ZC, gp); break;
    }
#if S2W_PROF
    if (getenv("S2W_PROF")) {
        unsigned long long h[4]; cudaDeviceSynchronize(); cudaMemcpyFromSymbol(h, g_s2wprof, sizeof h);
        const double t = (double)(h[0] + h[1] + h[2]);
        fprintf(stderr, "s2wprof %d->%d @%d MT %d NT %d blocks %llu: x %.1f%% gy %.1f%% mma %.1f%% (Mcyc/block %.3f)\n", xs.c, ys.c, xs.d, MT, NT, h[3], 100 * h[0] / t, 100 * h[1] / t, 100 * h[2] / t, t / h[3] / 1e6);
        memset(h, 0, sizeof h); cudaMemcpyToSymbol(g_s2wprof, h, sizeof h);
    }
#endif
}
extern "C" int lp_bwd_w_s2_f8(const void *x, int xbf, shape5 xs, const void *gy, int gybf, shape5 ys, float *gw, float *gb, gnp_t gp) {
    if (gybf == 4) { fprintf(stderr, "lp_bwd_w_s2_f8: fp4 gradients are not supported\n"); abort(); }
    if (xbf == 4 && gybf == 3) bwd_w_s2_f8_t<mx4_t, mx8_t>(x, xs, (const mx8_t *)gy, ys, gw, gb, gp);
    else if (xbf == 4 && gybf == 2) bwd_w_s2_f8_t<mx4_t, __half>(x, xs, (const __half *)gy, ys, gw, gb, gp);
    else if (xbf == 4 && gybf == 1) bwd_w_s2_f8_t<mx4_t, bf16>(x, xs, (const bf16 *)gy, ys, gw, gb, gp);
    else if (xbf == 4) bwd_w_s2_f8_t<mx4_t, float>(x, xs, (const float *)gy, ys, gw, gb, gp);
    else if (xbf == 3 && gybf == 3) bwd_w_s2_f8_t<mx8_t, mx8_t>(x, xs, (const mx8_t *)gy, ys, gw, gb, gp);
    else if (xbf == 3 && gybf == 2) bwd_w_s2_f8_t<mx8_t, __half>(x, xs, (const __half *)gy, ys, gw, gb, gp);
    else if (xbf == 3 && gybf == 1) bwd_w_s2_f8_t<mx8_t, bf16>(x, xs, (const bf16 *)gy, ys, gw, gb, gp);
    else if (xbf == 3) bwd_w_s2_f8_t<mx8_t, float>(x, xs, (const float *)gy, ys, gw, gb, gp);
    else if (xbf == 2 && gybf) bwd_w_s2_f8_t<__half, __half>(x, xs, (const __half *)gy, ys, gw, gb, gp);
    else if (xbf == 2) bwd_w_s2_f8_t<__half, float>(x, xs, (const float *)gy, ys, gw, gb, gp);
    else if (xbf && gybf) bwd_w_s2_f8_t<bf16, bf16>(x, xs, (const bf16 *)gy, ys, gw, gb, gp);
    else if (xbf) bwd_w_s2_f8_t<bf16, float>(x, xs, (const float *)gy, ys, gw, gb, gp);
    else bwd_w_s2_f8_t<float, float>(x, xs, (const float *)gy, ys, gw, gb, gp);
    LPCK();
    return 0;
}
