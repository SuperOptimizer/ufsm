#pragma once
/* kernels and launchers of lp_f8fwd (instantiated per type in lp_f8fwd_i*.cu) */
#include "lp_common.cuh"


/* ======================= FP8 forward, k=3, stride 1, pad 1 =======================
   Block = 8 warps over a 2 z x 8 y x 16 x output tile (as the BF16 kernel); warp w owns z = w/4, rows 2(w%4)+{0,1}
   (N = 32 voxels as 2 rows x 2 n-tiles), M = MT x 16 output channels. Input tile 4 x 10 x 18 positions x 32 ci,
   one 32-byte row per position; the two 16-byte halves are XOR-swizzled by bit 2 of the position so that
   ldmatrix over 8 consecutive positions is bank-conflict free (the swizzle is a function of the absolute
   position, so every shifted tap view sees it). */

template <int MT, int TZ, typename T, typename TO = T>   /* TZ = output z planes per block (2 or 4); warp owns TZ/2 planes x 2 rows */
__global__ void __launch_bounds__(256, 2) conv_fwd_f8_k(const T *__restrict__ x, const uint8_t *__restrict__ wq, const uint8_t *__restrict__ wsc,
                                                     const float *__restrict__ b, TO *__restrict__ y,
                                                     int N, int Ci, int D, int H, int W, int Co, int Cop, int Cip, gnp_t gp, double *__restrict__ osum, int Go, split_t sp) {
    constexpr int BM = MT * 16, TG = 9, RZ = TZ / 2, NR = 2 * RZ, TT = (TZ + 2) * 180, NRX = NR;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    uint8_t *sx = smem_raw;                         /* [TT pos][32 ci] swizzled */
    uint8_t *sxs = sx + TT * 32;                    /* [TT] position scales */
    uint8_t *wa = sxs + ((TT + 127) & ~127);                       /* [TG tap][BM co][32 ci] swizzled */
    uint8_t *was = wa + TG * BM * 32;               /* [TG][BM] */
    chan_t *ctab = (chan_t *)(was + TG * BM + 64 - (TG * BM) % 64);   /* [32] (16-aligned) */
    float2 *cab = (float2 *)(ctab + 32); unsigned *sgm = (unsigned *)(cab + 32);   /* [32] (a, b), GN mask */
    uint8_t *cu = (uint8_t *)(sgm + 4), *cus = cu + (TZ / 2 + 4) * 96 * 32;   /* sp.up: coarse rows of the up segment (stage_up_coarse) */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wz = warp >> 2, wr = (warp & 3) * 2;
    const int ox0 = blockIdx.x * 16, oy0 = blockIdx.y * 8;
    int bz = blockIdx.z;
    const int nzt = (D + TZ - 1) / TZ;
    const int oz0 = (bz % nzt) * TZ; bz /= nzt;
    const int nmt = Cop / BM;
    const int co0 = (bz % nmt) * BM; const int n = bz / nmt;
    const int nch = Cip / 32;
    float acc[MT][NR][2][4];
#pragma unroll
    for (int m = 0; m < MT; m++) for (int r = 0; r < NR; r++) for (int q = 0; q < 2; q++) for (int k = 0; k < 4; k++) acc[m][r][q][k] = 0.f;
    const size_t plane = (size_t)D * H * W;
    const int Cx = sp.x2 ? sp.c_split : Ci;
    const bool G = gp.G != 0 || sp.gp2.G != 0;
    for (int ci0 = 0; ci0 < Cip; ci0 += F8_CI) {
        __syncthreads();
        const bool upc = IS_MX(T) && sp.up && ci0 < (Cx + 31) / 32 * 32;   /* this chunk is the half-resolution x segment (decoder up part) */
        if (threadIdx.x < 32) {
            const chan_t c = make_chan(IS_MX(T) ? seg_ci(ci0 + threadIdx.x, Ci, Cx, (Cx + 31) / 32 * 32) : ci0 + threadIdx.x, Ci, Cx, n, upc ? (size_t)(D >> 1) * (H >> 1) * (W >> 1) : plane, x, sp, gp, N);
            ctab[threadIdx.x] = c;
            cab[threadIdx.x] = make_float2(c.a, c.b);   /* compact GN coefficients for the MX staging (one broadcast LDS.64 per element) */
            const unsigned gm = __ballot_sync(0xffffffffu, c.g != 0);
            if (threadIdx.x == 0) *sgm = gm;
        }
        __syncthreads();
        const unsigned gmask = *sgm;
        if (upc) { stage_up_coarse<T>(ctab[0], cu, cus, (oz0 >> 1) - 2, (oy0 >> 1) - 2, (ox0 >> 1) - 2, TZ / 2 + 4, D, H, W, cab, gmask); __syncthreads(); }
        if constexpr (IS_MX8(T)) if (!G && !upc) {   /* MX-fp8 input without transform: the chunk is one stored block -> copy bytes and scales */
            const chan_t c0 = ctab[0];
            for (int pos = threadIdx.x; pos < TT; pos += 256) {
                int ix = pos % 18, iy = (pos / 18) % 10, iz = pos / 180;
                int gz = oz0 - 1 + iz, gy = oy0 - 1 + iy, gx = ox0 - 1 + ix;
                bool inb = c0.p && gz >= 0 && gz < D && gy >= 0 && gy < H && gx >= 0 && gx < W;
                uint4 h0 = make_uint4(0u, 0u, 0u, 0u), h1 = h0;
                unsigned sc = 1u;
                if (inb) {
                    size_t off = ((size_t)gz * H + gy) * W + gx;
                    const uint4 *src = (const uint4 *)((const uint8_t *)c0.p + off * c0.rb);
                    if (c0.bw == 8) { const uint2 u = __ldg((const uint2 *)src); h0.x = u.x; h0.y = u.y; }
                    else { h0 = __ldg(src); if (c0.bw == 32) h1 = __ldg(src + 1); }
                    sc = c0.sp[off];
                }
                *(uint4 *)(sx + sw16(pos, 0)) = h0;
                *(uint4 *)(sx + sw16(pos, 1)) = h1;
                sxs[pos] = (uint8_t)sc;
            }
        }
        /* staging: one thread per position, all 32 channels (plane-major reads, or one MX row dequantised: fp8 with gn+silu,
           fp4 always) -> per-position amax -> e4m3 + scale */
        if (!IS_MX8(T) || G || upc) for (int pos = threadIdx.x; pos < TT; pos += 256) {
            int ix = pos % 18, iy = (pos / 18) % 10, iz = pos / 180;
            int gz = oz0 - 1 + iz, gy = oy0 - 1 + iy, gx = ox0 - 1 + ix;
            bool inb = gz >= 0 && gz < D && gy >= 0 && gy < H && gx >= 0 && gx < W;
            size_t off = inb ? ((size_t)gz * H + gy) * W + gx : 0;
            float v[32], amax = 0.f;
            if (upc) stage_up_smem(cu, cus, (oz0 >> 1) - 2, (oy0 >> 1) - 2, (ox0 >> 1) - 2, gz, gy, gx, D, H, W, inb, v); else stage_row32<T>(ctab, off, inb, G, v, cab, gmask);
#pragma unroll
            for (int k = 0; k < 32; k++) amax = fmaxf(amax, fabsf(v[k]));   /* e4m3 keeps a NaN element itself (0x7f): plain max */
            int e = mx_exp(amax, 1.f / 448.f);
            float m = exp2i(-e);
            if (!IS_MX(T) && sp.sr) {   /* gradient operand: stochastic rounding, dither keyed by the element (deterministic per call); MX gradients are staged as stored */
                const uint64_t vid = (((uint64_t)n * Cip + ci0) * D + gz) * (uint64_t)H * W + (uint64_t)gy * W + gx;
#pragma unroll
                for (int k = 0; k < 32; k++) v[k] = sr_e4m3(v[k] * m, sr_hash(sp.sr, vid * 32 + k)) / m;
            }
            uint4 h0 = make_uint4(cvt_e4m3x4(v[0] * m, v[1] * m, v[2] * m, v[3] * m), cvt_e4m3x4(v[4] * m, v[5] * m, v[6] * m, v[7] * m),
                                  cvt_e4m3x4(v[8] * m, v[9] * m, v[10] * m, v[11] * m), cvt_e4m3x4(v[12] * m, v[13] * m, v[14] * m, v[15] * m));
            uint4 h1 = make_uint4(cvt_e4m3x4(v[16] * m, v[17] * m, v[18] * m, v[19] * m), cvt_e4m3x4(v[20] * m, v[21] * m, v[22] * m, v[23] * m),
                                  cvt_e4m3x4(v[24] * m, v[25] * m, v[26] * m, v[27] * m), cvt_e4m3x4(v[28] * m, v[29] * m, v[30] * m, v[31] * m));
            *(uint4 *)(sx + sw16(pos, 0)) = h0;
            *(uint4 *)(sx + sw16(pos, 1)) = h1;
            sxs[pos] = (uint8_t)(e + 127);
        }
        for (int t0 = 0; t0 < 27; t0 += TG) {
            if (t0) __syncthreads();
            for (int i = threadIdx.x; i < TG * BM * 2; i += 256) {
                int tt = i / (BM * 2), r = i % (BM * 2), c = r >> 1, h = r & 1;
                *(uint4 *)(wa + tt * BM * 32 + sw16(c, h)) = __ldg((const uint4 *)(wq + ((size_t)(t0 + tt) * Cop + co0 + c) * Cip + ci0 + h * 16));
            }
            for (int i = threadIdx.x; i < TG * BM; i += 256) {
                int tt = i / BM, c = i % BM;
                was[i] = wsc[((size_t)(t0 + tt) * Cop + co0 + c) * nch + ci0 / 32];
            }
            __syncthreads();
#pragma unroll
            for (int tt = 0; tt < TG; tt++) {
                int tap = t0 + tt, kz = tap / 9, ky = (tap / 3) % 3, kx = tap % 3;
                unsigned bfr[NR][4], sb[NR][2];
#pragma unroll
                for (int r = 0; r < NR; r++) {
                    int mat = lane >> 3, q = mat >> 1, kh = mat & 1;
                    int rowp = ((wz * RZ + (r >> 1) + kz) * 10 + wr + (r & 1) + ky) * 18 + kx;
                    int pos = rowp + q * 8 + (lane & 7);
                    ldsm_x4(bfr[r], sx + sw16(pos, kh));
                    sb[r][0] = sxs[rowp + g];
                    sb[r][1] = sxs[rowp + 8 + g];
                }
#pragma unroll
                for (int m = 0; m < MT; m++) {
                    unsigned af[4];
                    int mat = lane >> 3, row = m * 16 + (mat & 1) * 8 + (lane & 7);
                    ldsm_x4(af, wa + tt * BM * 32 + sw16(row, mat >> 1));
                    unsigned sa = was[tt * BM + m * 16 + g + 8 * (t & 1)];
#pragma unroll
                    for (int r = 0; r < NR; r++) { mma_f8(acc[m][r][0], af, bfr[r], sa, sb[r][0]); mma_f8(acc[m][r][1], af, bfr[r] + 2, sa, sb[r][1]); }
                }
            }
        }
    }
    fwd_epilogue<MT, NRX, TO>(acc, smem_raw, y, b, n, co0, Co, D, H, W, oz0, oy0, ox0, wz, wr, osum, Go, sp, N);
}

/* ---- small input-channel variant (Ci <= 16): channels padded to CP in {4, 8, 16}; one K block of 32 = TPK = 32/CP
   taps x CP channels, so 27 taps take ceil(27/TPK) mma k-steps instead of 27 half-empty ones. A K block mixes
   positions, so the B scale is one per staged tile (block amax). Weights: wq[kb][Cop][32], ws[kb][Cop]. */
template <int CP>
__global__ void prep_w8s_k(const float *__restrict__ w, uint8_t *__restrict__ wq, uint8_t *__restrict__ ws, int Co, int Ci, int Cop) {
    constexpr int TPK = 32 / CP, NKB = (27 + TPK - 1) / TPK;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)NKB * Cop) return;
    int co = (int)(i % Cop), kb = (int)(i / Cop);
    float v[32], amax = 0.f;
#pragma unroll
    for (int k = 0; k < 32; k++) {
        int tap = kb * TPK + k / CP, ci = k % CP;
        v[k] = (co < Co && ci < Ci && tap < 27) ? w[((size_t)co * Ci + ci) * 27 + tap] : 0.f;
        amax = fmaxf(amax, fabsf(v[k]));
    }
    int e = mx_exp(amax, 1.f / 448.f);
    float m = exp2i(-e);
    uint4 *dst = (uint4 *)(wq + i * 32);
    dst[0] = make_uint4(cvt_e4m3x4(v[0] * m, v[1] * m, v[2] * m, v[3] * m), cvt_e4m3x4(v[4] * m, v[5] * m, v[6] * m, v[7] * m),
                        cvt_e4m3x4(v[8] * m, v[9] * m, v[10] * m, v[11] * m), cvt_e4m3x4(v[12] * m, v[13] * m, v[14] * m, v[15] * m));
    dst[1] = make_uint4(cvt_e4m3x4(v[16] * m, v[17] * m, v[18] * m, v[19] * m), cvt_e4m3x4(v[20] * m, v[21] * m, v[22] * m, v[23] * m),
                        cvt_e4m3x4(v[24] * m, v[25] * m, v[26] * m, v[27] * m), cvt_e4m3x4(v[28] * m, v[29] * m, v[30] * m, v[31] * m));
    ws[i] = (uint8_t)(e + 127);
}
template <int MT, int CP, typename T, typename TO = T>   /* TO: output type (an MX output from a 16-bit network input) */
__global__ void __launch_bounds__(256, 2) conv_fwd_f8s_k(const T *__restrict__ x, const uint8_t *__restrict__ wq, const uint8_t *__restrict__ wsc,
                                                      const float *__restrict__ b, TO *__restrict__ y,
                                                      int N, int Ci, int D, int H, int W, int Co, int Cop, gnp_t gp, double *__restrict__ osum, int Go, split_t sp) {
    constexpr int BM = MT * 16, TPK = 32 / CP, NKB = (27 + TPK - 1) / TPK, NRX = 2;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    uint8_t *sx = smem_raw;                         /* [F8_T pos][CP] */
    uint8_t *wa = sx + F8_T * CP;                   /* [NKB][BM][32] swizzled */
    uint8_t *was = wa + NKB * BM * 32;              /* [NKB][BM] */
    unsigned *samax = (unsigned *)(was + ((NKB * BM + 15) & ~15));
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wz = warp >> 2, wr = (warp & 3) * 2;
    const int ox0 = blockIdx.x * 16, oy0 = blockIdx.y * 8;
    int bz = blockIdx.z;
    const int nzt = (D + 1) / 2;
    const int oz0 = (bz % nzt) * 2; bz /= nzt;
    const int nmt = Cop / BM;
    const int co0 = (bz % nmt) * BM; const int n = bz / nmt;
    float acc[MT][2][2][4];
#pragma unroll
    for (int m = 0; m < MT; m++) for (int r = 0; r < 2; r++) for (int q = 0; q < 2; q++) for (int k = 0; k < 4; k++) acc[m][r][q][k] = 0.f;
    const size_t plane = (size_t)D * H * W;
    const int Cx = sp.x2 ? sp.c_split : Ci;
    chan_t *ctab = (chan_t *)(samax + 4);              /* [16] */
    const bool G = gp.G != 0 || sp.gp2.G != 0;
    if (threadIdx.x == 0) *samax = 0u;
    if (threadIdx.x < CP) ctab[threadIdx.x] = make_chan(threadIdx.x, Ci, Cx, n, plane, x, sp, gp, N);
    for (int i = threadIdx.x; i < NKB * BM * 2; i += 256) {
        int kb = i / (BM * 2), r = i % (BM * 2), c = r >> 1, h = r & 1;
        *(uint4 *)(wa + kb * BM * 32 + sw16(c, h)) = __ldg((const uint4 *)(wq + ((size_t)kb * Cop + co0 + c) * 32 + h * 16));
    }
    for (int i = threadIdx.x; i < NKB * BM; i += 256) { int kb = i / BM, c = i % BM; was[i] = wsc[(size_t)kb * Cop + co0 + c]; }
    __syncthreads();
    float v[3][CP];
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 3; j++) {
        int pos = threadIdx.x + 256 * j;
        int ix = pos % 18, iy = (pos / 18) % 10, iz = pos / 180;
        int gz = oz0 - 1 + iz, gy = oy0 - 1 + iy, gx = ox0 - 1 + ix;
        bool inb = pos < F8_T && gz >= 0 && gz < D && gy >= 0 && gy < H && gx >= 0 && gx < W;
        size_t off = inb ? ((size_t)gz * H + gy) * W + gx : 0;
#pragma unroll
        for (int k = 0; k < CP; k++) {
            float val = 0.f;
            chan_t c = ctab[k];
            if (inb && c.p) val = act_ab(ldc<T>(c, off), c.a, c.b, G && c.g);
            v[j][k] = val; amax = fmaxf(amax, fabsf(val));
        }
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    if (lane == 0) atomicMax(samax, __float_as_uint(amax));
    __syncthreads();
    const int e = mx_exp(__uint_as_float(*samax), 1.f / 448.f);
    const unsigned sb = (unsigned)(e + 127);
    {
        float m = exp2i(-e);
#pragma unroll
        for (int j = 0; j < 3; j++) {
            int pos = threadIdx.x + 256 * j;
            if (pos >= F8_T) break;
            unsigned wv[CP / 4];
#pragma unroll
            for (int k = 0; k < CP / 4; k++) wv[k] = cvt_e4m3x4(v[j][4 * k] * m, v[j][4 * k + 1] * m, v[j][4 * k + 2] * m, v[j][4 * k + 3] * m);
            if constexpr (CP == 16) *(uint4 *)(sx + pos * 16) = make_uint4(wv[0], wv[1], wv[2], wv[3]);
            else if constexpr (CP == 8) *(uint2 *)(sx + pos * 8) = make_uint2(wv[0], wv[1]);
            else *(unsigned *)(sx + pos * 4) = wv[0];
        }
    }
    __syncthreads();
    /* lane (g, t): b0 holds k = 4t..4t+3 -> local tap 4t / CP, channel 4t % CP; b1 holds k = 16 + 4t.. */
    const int tl0 = (4 * t) / CP, ch0 = (4 * t) % CP, tl1 = (16 + 4 * t) / CP, ch1 = (16 + 4 * t) % CP;
#pragma unroll 2
    for (int kb = 0; kb < NKB; kb++) {
        int tap0 = min(kb * TPK + tl0, 26), tap1 = min(kb * TPK + tl1, 26);
        int base0 = ((wz + tap0 / 9) * 10 + wr + (tap0 / 3) % 3) * 18 + tap0 % 3 + g;
        int base1 = ((wz + tap1 / 9) * 10 + wr + (tap1 / 3) % 3) * 18 + tap1 % 3 + g;
        unsigned bfr[2][2][2];
#pragma unroll
        for (int r = 0; r < 2; r++)
#pragma unroll
            for (int q = 0; q < 2; q++) {
                bfr[r][q][0] = *(const unsigned *)(sx + (base0 + r * 18 + q * 8) * CP + ch0);
                bfr[r][q][1] = *(const unsigned *)(sx + (base1 + r * 18 + q * 8) * CP + ch1);
            }
#pragma unroll
        for (int m = 0; m < MT; m++) {
            unsigned af[4];
            int mat = lane >> 3, row = m * 16 + (mat & 1) * 8 + (lane & 7);
            ldsm_x4(af, wa + kb * BM * 32 + sw16(row, mat >> 1));
            unsigned sa = was[kb * BM + m * 16 + g + 8 * (t & 1)];
#pragma unroll
            for (int r = 0; r < 2; r++) { mma_f8(acc[m][r][0], af, bfr[r][0], sa, sb); mma_f8(acc[m][r][1], af, bfr[r][1], sa, sb); }
        }
    }
    fwd_epilogue<MT, NRX, TO>(acc, smem_raw, y, b, n, co0, Co, D, H, W, oz0, oy0, ox0, wz, wr, osum, Go, sp, N);
}


template <int MT, int CP, typename T, typename TO> void launch_f8s(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp) {
    constexpr int TPK = 32 / CP, NKB = (27 + TPK - 1) / TPK;
    size_t smem = (size_t)F8_T * CP + NKB * MT * 16 * 32 + ((NKB * MT * 16 + 15) & ~15) + 16 + 16 * sizeof(chan_t);
    if (smem < 8 * 256 * sizeof(float)) smem = 8 * 256 * sizeof(float);
    static int attr[8];
    if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)conv_fwd_f8s_k<MT, CP, T, TO>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
    conv_fwd_f8s_k<MT, CP, T, TO><<<grid, 256, smem>>>((const T *)x, wq, ws, b, (TO *)y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, gp, osum, Go, sp);
}
template <int CP, typename T, typename TO = T> static void small_f8(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp) {
    constexpr int TPK = 32 / CP, NKB = (27 + TPK - 1) / TPK;
    int Cop = (cout + 15) / 16 * 16;
    if (IS_MX(TO) && cout > 16) Cop = (cout + 31) / 32 * 32;
    uint8_t *wq = lp_buf<uint8_t>(0, (size_t)NKB * Cop * 32), *ws = lp_buf<uint8_t>(1, (size_t)NKB * Cop);
    prep_w8s_k<CP><<<nblk_((size_t)NKB * Cop, 128), 128>>>(w, wq, ws, cout, xs.c, Cop);
    int MT = Cop % 64 == 0 ? 4 : Cop % 32 == 0 ? 2 : 1;   /* BM = 16 MT must divide Cop (Cop = 48 -> MT = 1) */
    int nmt = Cop / (MT * 16);
    dim3 grid(nblk_(xs.w, 16), nblk_(xs.h, 8), (unsigned)(nblk_(xs.d, 2) * nmt * xs.n));
    switch (MT) {
    case 1: launch_f8s<1, CP, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    case 2: launch_f8s<2, CP, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    default: launch_f8s<4, CP, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    }
}
/* ---- 16-channel inputs, two taps per K block (fp8): k-step 2 r = [kx0 | kx1] and 2 r + 1 = [kx2 | 0] of tap row r = (kz, ky),
   16 channels per slot, so a conv takes 18 m16n8k32 k-steps instead of 27 half-empty ones and the staged tile is half as
   wide ([pos][16 B]). The K block [kx | kx + 1] of output voxel x is the 32 contiguous bytes of staged positions x + kx and
   x + kx + 1, so the B fragments are plain ldmatrix rows (8 consecutive 16-byte rows: conflict-free without a swizzle); the
   zero half of [kx2 | 0] reads a zero row. A K block spans two positions, so the B scale is one ue8m0 per staged row
   (z, y) of 18 positions. Weights wq[ks][Cop][32] (ks = 2 r + h), ws[ks][Cop]. Shared staging: stage_row32 (16 live
   channels), so plane-major, mx8 and mx4 inputs (bw 16 rows) and the GN+SiLU input transform all work; output via
   fwd_epilogue (any TO). */
/* the 16 staged values of one position for a 16-channel input: dequantised (MX: one bw-16 block row of c0) or read plane-major
   (c0.p = channel 0, channel k at + k plane), then the GN+SiLU affine gab[k].x x + gab[k].y (G; registers or shared memory);
   zeros outside and in channels >= Ci. Shared by the 16-channel kernels. */
template <int MT, int TZ, typename T, typename TO>
__global__ void __launch_bounds__(256, 2) conv_fwd_f8p_k(const T *__restrict__ x, const uint8_t *__restrict__ wq, const uint8_t *__restrict__ wsc,
                                                      const float *__restrict__ b, TO *__restrict__ y,
                                                      int N, int Ci, int D, int H, int W, int Co, int Cop, gnp_t gp, double *__restrict__ osum, int Go, split_t sp) {
    constexpr int BM = MT * 16, RZ = TZ / 2, NR = 2 * RZ, NROW = (TZ + 2) * 10, TT = NROW * 18;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    uint8_t *sx = smem_raw;                    /* [TT pos][16 B] */
    uint8_t *zrow = sx + TT * 16;              /* [16] zeros */
    uint8_t *sxs = zrow + 16;                  /* [NROW] row scales (64 reserved) */
    uint8_t *wa = sxs + 64;                    /* [18 ks][BM co][32] swizzled */
    uint8_t *was = wa + P16_KS * BM * 32;      /* [18][BM] */
    chan_t *ctab = (chan_t *)(was + ((P16_KS * BM + 15) & ~15));   /* [32] */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wz = warp >> 2, wr = (warp & 3) * 2;
    const int ox0 = blockIdx.x * 16, oy0 = blockIdx.y * 8;
    int bz = blockIdx.z;
    const int nzt = (D + TZ - 1) / TZ;
    const int oz0 = (bz % nzt) * TZ; bz /= nzt;
    const int nmt = Cop / BM;
    const int co0 = (bz % nmt) * BM; const int n = bz / nmt;
    const size_t plane = (size_t)D * H * W;
    const bool G = gp.G != 0;
    for (int i = threadIdx.x; i < P16_KS * BM * 2; i += 256) {
        const int ks = i / (BM * 2), r = i % (BM * 2), c = r >> 1, h = r & 1;
        *(uint4 *)(wa + ks * BM * 32 + sw16(c, h)) = __ldg((const uint4 *)(wq + ((size_t)ks * Cop + co0 + c) * 32 + h * 16));
    }
    for (int i = threadIdx.x; i < P16_KS * BM; i += 256) was[i] = wsc[(size_t)(i / BM) * Cop + co0 + i % BM];
    if (threadIdx.x < 32) ctab[threadIdx.x] = make_chan(threadIdx.x < 16 ? (int)threadIdx.x : -1, Ci, Ci, n, plane, x, sp, gp, N);
    if (threadIdx.x < 4) ((unsigned *)zrow)[threadIdx.x] = 0u;
    __syncthreads();
    /* staging: thread per position, values kept in registers until the row amax (smem atomicMax over the row's 18 positions)
       is known, then quantised with the row scale */
    {
        constexpr int NIT = (TT + 255) / 256;
        const chan_t c0 = ctab[0];
        float2 gab[16];
#pragma unroll
        for (int k = 0; k < 16; k++) gab[k] = make_float2(ctab[k].a, ctab[k].b);
        unsigned *ramax = (unsigned *)ctab + 32 * sizeof(chan_t) / 4;   /* [NROW] */
        for (int i = threadIdx.x; i < NROW; i += 256) ramax[i] = 0u;
        __syncthreads();
        float v[NIT][16];
#pragma unroll
        for (int it = 0; it < NIT; it++) {
            const int pos = threadIdx.x + 256 * it, row = pos / 18, ix = pos - 18 * row;
            const int gz = oz0 - 1 + row / 10, gy = oy0 - 1 + row % 10, gx = ox0 - 1 + ix;
            const bool inb = pos < TT && gz >= 0 && gz < D && gy >= 0 && gy < H && gx >= 0 && gx < W;
            stage_row16<T>(c0, Ci, plane, inb ? ((size_t)gz * H + gy) * W + gx : 0, inb, G, gab, v[it]);
            unsigned am = 0u;
#pragma unroll
            for (int k = 0; k < 16; k++) { const unsigned u = __float_as_uint(fabsf(v[it][k])); am = u <= 0x7f800000u ? max(am, u) : am; }   /* finite magnitudes (as fmaxf: e4m3 keeps a NaN element itself) */
            if (pos < TT && am) atomicMax(&ramax[row], am);
        }
        __syncthreads();
#pragma unroll
        for (int it = 0; it < NIT; it++) {
            const int pos = threadIdx.x + 256 * it, row = pos / 18, ix = pos - 18 * row;
            if (pos >= TT) break;
            const int e = mx_exp(__uint_as_float(ramax[row]), 1.f / 448.f);
            const float m = exp2i(-e);
            float *w = v[it];
            if (sp.sr) {   /* gradient operand (backward-data input): stochastic rounding keyed by the element */
                const int gz = oz0 - 1 + row / 10, gy = oy0 - 1 + row % 10, gx = ox0 - 1 + ix;
                const uint64_t vid = (((uint64_t)n * D + gz) * H + gy) * (uint64_t)W + gx;
#pragma unroll
                for (int k = 0; k < 16; k++) w[k] = sr_e4m3(w[k] * m, sr_hash(sp.sr, vid * 16 + k)) / m;
            }
            *(uint4 *)(sx + pos * 16) = make_uint4(cvt_e4m3x4(w[0] * m, w[1] * m, w[2] * m, w[3] * m), cvt_e4m3x4(w[4] * m, w[5] * m, w[6] * m, w[7] * m),
                                                   cvt_e4m3x4(w[8] * m, w[9] * m, w[10] * m, w[11] * m), cvt_e4m3x4(w[12] * m, w[13] * m, w[14] * m, w[15] * m));
            if (ix == 0) sxs[row] = (uint8_t)(e + 127);
        }
    }
    __syncthreads();
    float acc[MT][NR][2][4];
#pragma unroll
    for (int m = 0; m < MT; m++) for (int r = 0; r < NR; r++) for (int q = 0; q < 2; q++) for (int k = 0; k < 4; k++) acc[m][r][q][k] = 0.f;
    const int mat = lane >> 3, kh = mat & 1, vx = (mat >> 1) * 8 + (lane & 7);
#pragma unroll 1
    for (int tr = 0; tr < 9; tr++) {
        const int kz = tr / 3, ky = tr % 3;
        unsigned bA[NR][4], bB[NR][4], sb[NR];
#pragma unroll
        for (int r = 0; r < NR; r++) {
            const int rowi = (wz * RZ + (r >> 1) + kz) * 10 + wr + (r & 1) + ky, base = rowi * 18 + vx;
            ldsm_x4(bA[r], sx + (base + kh) * 16);              /* [kx0 | kx1] */
            ldsm_x4(bB[r], kh ? zrow : sx + (base + 2) * 16);   /* [kx2 | 0] */
            sb[r] = sxs[rowi];
        }
#pragma unroll
        for (int m = 0; m < MT; m++) {
            const int arow = m * 16 + (mat & 1) * 8 + (lane & 7);
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int ks = 2 * tr + h;
                unsigned af[4];
                ldsm_x4(af, wa + ks * BM * 32 + sw16(arow, mat >> 1));
                const unsigned sa = was[ks * BM + m * 16 + g + 8 * (t & 1)];
#pragma unroll
                for (int r = 0; r < NR; r++) {
                    const unsigned *bf = h ? bB[r] : bA[r];
                    mma_f8(acc[m][r][0], af, bf, sa, sb[r]); mma_f8(acc[m][r][1], af, bf + 2, sa, sb[r]);
                }
            }
        }
    }
    fwd_epilogue<MT, NR, TO>(acc, smem_raw, y, b, n, co0, Co, D, H, W, oz0, oy0, ox0, wz, wr, osum, Go, sp, N);
}
template <int MT, int TZ, typename T, typename TO> void launch_f8p(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp) {
    constexpr int TT = (TZ + 2) * 180, BM = MT * 16;
    size_t smem = (size_t)TT * 16 + 16 + 64 + P16_KS * BM * 32 + ((P16_KS * BM + 15) & ~15) + 32 * sizeof(chan_t) + 64 * 4;
    if (smem < 8 * 256 * sizeof(float)) smem = 8 * 256 * sizeof(float);
    static int attr[8];
    if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)conv_fwd_f8p_k<MT, TZ, T, TO>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
    conv_fwd_f8p_k<MT, TZ, T, TO><<<grid, 256, smem>>>((const T *)x, wq, ws, b, (TO *)y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, gp, osum, Go, sp);
}
template <typename T, typename TO> static void p16_f8(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, int Cop, int Ox, int OxP, gnp_t gp, double *osum, int Go, split_t sp) {
    uint8_t *wq = lp_buf<uint8_t>(0, (size_t)P16_KS * Cop * 32), *ws = lp_buf<uint8_t>(1, (size_t)P16_KS * Cop);
    prep_w8p_k<<<nblk_((size_t)P16_KS * Cop, 128), 128>>>(w, wq, ws, cout, xs.c, Cop, IS_MX(TO) && sp.y2 ? Ox : -1, OxP);
    int MT = Cop % 64 == 0 ? 4 : Cop % 32 == 0 ? 2 : !IS_MX(TO) && Cop == 48 ? 3 : 1;   /* 48 plane-major outputs (dec0.c1 backward-data): one tile, the input staged once */
    const int nmt = Cop / (MT * 16);
    static int tz_env = -1;
    if (tz_env < 0) tz_env = getenv("UFSM_F8_TZ") ? atoi(getenv("UFSM_F8_TZ")) : 0;
    int TZ = tz_env ? tz_env : (MT <= 2 && xs.d >= 16 ? 4 : 2);
    if (MT == 4) TZ = 2;
    dim3 grid(nblk_(xs.w, 16), nblk_(xs.h, 8), (unsigned)(nblk_(xs.d, TZ) * nmt * xs.n));
    switch (MT * 10 + TZ) {
    case 12: launch_f8p<1, 2, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    case 14: launch_f8p<1, 4, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    case 22: launch_f8p<2, 2, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    case 24: launch_f8p<2, 4, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    case 32: launch_f8p<3, 2, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    case 34: launch_f8p<3, 4, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    default: launch_f8p<4, 2, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    }
}
template <int MT, int TZ, typename T, typename TO> void launch_f8g(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, int Cip, gnp_t gp, double *osum, int Go, split_t sp) {
    constexpr int TT = (TZ + 2) * 180;
    size_t smem = (size_t)TT * 32 + ((TT + 127) & ~127) + 9 * MT * 16 * 33 + 64 + 32 * sizeof(chan_t) + 32 * 8 + 16 + (sp.up ? (TZ / 2 + 4) * 96 * 33 + 16 : 0);
    if (smem < 8 * 256 * sizeof(float)) smem = 8 * 256 * sizeof(float);
    static int attr[8];
    if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)conv_fwd_f8_k<MT, TZ, T, TO>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
    conv_fwd_f8_k<MT, TZ, T, TO><<<grid, 256, smem>>>((const T *)x, wq, ws, b, (TO *)y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, Cip, gp, osum, Go, sp);
}
template <typename T, typename TO = T> void fwd_f8_t(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp) {
    static int small = -1;
    if (small < 0) small = ufsm_env_on("UFSM_F8_NOSMALL") ? 0 : 1;
    /* tap-packed small-channel kernel: fp32 input up to 16 channels; with bf16 input the general kernel is as fast at 16 */
    if (small && (std::is_same<T, TO>::value || IS_MX(TO)) && xs.c <= (sizeof(T) == 2 || IS_MX(T) ? 8 : 16) && !sp.x2) {
        if (xs.c <= 4) small_f8<4, T, TO>(x, xs, w, b, cout, y, gp, osum, Go, sp);
        else if (xs.c <= 8) small_f8<8, T, TO>(x, xs, w, b, cout, y, gp, osum, Go, sp);
        else small_f8<16, T, TO>(x, xs, w, b, cout, y, gp, osum, Go, sp);
        return;
    }
    if constexpr (!std::is_same<T, TO>::value) {   /* mixed in / out types exist only for the network stem (Ci <= 8, small kernel):
                                                       the general kernels are not instantiated for them (compile time) */
        fprintf(stderr, "lp_conv_fwd_f8: mixed input / output storage needs the small-channel kernel (Ci %d, UFSM_F8_NOSMALL unset)\n", xs.c); abort();
    } else {
    int Cop = (cout + 15) / 16 * 16, Cip = (xs.c + 31) / 32 * 32;
    int Cx = sp.x2 ? sp.c_split : xs.c, CxP = (Cx + 31) / 32 * 32;
    int Ox = sp.y2 ? sp.o_split : cout, OxP = (Ox + 31) / 32 * 32;
    if (IS_MX(TO)) {   /* MX output blocks of 32 channels need 32-row-aligned m-tiles (split outputs: per-tensor segments) */
        if (sp.y2) Cop = OxP + (cout - Ox + 31) / 32 * 32;
        else if (cout > 16) Cop = (cout + 31) / 32 * 32;
    }
    if (IS_MX(T)) Cip = CxP + (xs.c - Cx + 31) / 32 * 32;   /* MX inputs: per-tensor 32-channel segments */
    static int p16 = -1;
    if (p16 < 0) p16 = getenv("UFSM_F8_PACK16") ? atoi(getenv("UFSM_F8_PACK16")) : 1;
    if (p16 && xs.c <= 16 && !sp.x2 && !sp.up && sp.gp2.G == 0) { p16_f8<T, TO>(x, xs, w, b, cout, y, Cop, Ox, OxP, gp, osum, Go, sp); return; }   /* two taps per K block */
    int nch = Cip / 32;
    uint8_t *wq = lp_buf<uint8_t>(0, (size_t)27 * Cop * Cip), *ws = lp_buf<uint8_t>(1, (size_t)27 * Cop * nch);
    size_t nt = (size_t)27 * Cop * nch;
    if (IS_MX(T) || IS_MX(TO)) prep_w8_k<<<nblk_(nt, 128), 128>>>(w, wq, ws, cout, xs.c, Cop, Cip, IS_MX(T) ? Cx : -1, CxP, IS_MX(TO) && sp.y2 ? Ox : -1, OxP);
    else prep_w8_k<<<nblk_(nt, 128), 128>>>(w, wq, ws, cout, xs.c, Cop, Cip);
    int MT = Cop % 64 == 0 ? 4 : Cop % 32 == 0 ? 2 : 1;   /* BM = 16 MT must divide Cop (Cop = 48 -> MT = 1) */
    int nmt = Cop / (MT * 16);
    static int tz_env = -1;
    if (tz_env < 0) tz_env = getenv("UFSM_F8_TZ") ? atoi(getenv("UFSM_F8_TZ")) : 0;
    int TZ = tz_env ? tz_env : (MT <= 2 && xs.d >= 16 ? 4 : 2);
    if (MT == 4) TZ = 2;
    dim3 grid(nblk_(xs.w, 16), nblk_(xs.h, 8), (unsigned)(nblk_(xs.d, TZ) * nmt * xs.n));
    switch (MT * 10 + TZ) {
    case 12: launch_f8g<1, 2, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    case 14: launch_f8g<1, 4, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    case 22: launch_f8g<2, 2, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    case 24: launch_f8g<2, 4, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    default: launch_f8g<4, 2, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    }
    }
}
