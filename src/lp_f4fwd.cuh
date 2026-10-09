#pragma once
/* kernels and launchers of lp_f4fwd (instantiated per type in lp_f4fwd_i*.cu) */
#include "lp_common.cuh"

/* ======================= FP4 (e2m1, MX ue8m0 scales per 32 K) forward, k=3, stride 1 =======================
   m16n8k64: one mma = two taps x 32 input channels; K block 0 = tap 2p, K block 1 = tap 2p+1 (pair p, 14 pairs,
   the last one half zero). Same tile / warp layout as the FP8 kernel (TZ output planes per block); the input tile is
   [pos][16 B] (32 channels x 4 bit), one scale per (position, 32 channels) -> B scale register bytes 0 / 1 = the two
   taps' positions. Weights wq4[tap][Cop][Cip/2] with one scale per (tap, co, 32-channel chunk).
   Input types: mx4 (stored rows copied straight into the tile when no transform applies, else dequantised + GN+SiLU +
   requantised), mx8 (dequantised, requantised to e2m1), fp32 / 16-bit (staged as the fp8 kernel). Output type TO
   separate from the input (the backward feeds mx8 gradients into 16-bit / mx8 outputs). Inputs with 9..16 channels take the
   packed 16-channel kernel conv_fwd_f4p_k, Ci <= 8 (the network input) the fp8 tap-packed kernel. */
__global__ void prep_w4_k(const float *__restrict__ w, uint8_t *__restrict__ wq, uint8_t *__restrict__ ws, int Co, int Ci, int Cop, int Cip, int Cx = -1, int CxP = 0, int Ox = -1, int OxP = 0);   /* defined in lp_f4fwd.cu */
/* 2D weight scales (UFSM_W4_2D=1): one ue8m0 per (tap, 32 padded output rows, 32-channel input chunk) tile instead of per
   (tap, row, chunk). The flipped weights of the backward-data conv tile the same 32 x 32 blocks with rows and chunks swapped,
   so forward and backward-data multiply the same quantised weights (exactly when both sides use the same 32-channel
   alignment: plane-major / 16-bit on both, or MX segments on both). Warp per tile, lane = row; same layout as prep_w4_k. */
extern int g_w2d;
__global__ void prep_w4_2d_k(const float *__restrict__ w, uint8_t *__restrict__ wq, uint8_t *__restrict__ ws, int Co, int Ci, int Cop, int Cip, int Cx = -1, int CxP = 0, int Ox = -1, int OxP = 0);   /* defined in lp_f4fwd.cu */
template <int MT, int TZ, typename T, typename TO>
__global__ void __launch_bounds__(256, 2) conv_fwd_f4_k(const T *__restrict__ x, const uint8_t *__restrict__ wq, const uint8_t *__restrict__ wsc,
                                                     const float *__restrict__ b, TO *__restrict__ y,
                                                     int N, int Ci, int D, int H, int W, int Co, int Cop, int Cip, gnp_t gp, double *__restrict__ osum, int Go, split_t sp) {
    constexpr int BM = MT * 16, PG = 7, RZ = TZ / 2, NR = 2 * RZ, TT = (TZ + 2) * 180, NRX = NR;   /* PG: tap pairs per weight stage */
    extern __shared__ __align__(128) unsigned char smem_raw[];
    uint8_t *sx = smem_raw;                         /* [TT pos][16 B] */
    uint8_t *sxs = sx + TT * 16;                    /* [TT] */
    uint8_t *wa = sxs + ((TT + 127) & ~127);        /* [PG pair][BM co][32 B] (tapA 16 B | tapB 16 B), swizzled */
    unsigned short *was = (unsigned short *)(wa + PG * BM * 32);   /* [PG][BM] (byte 0 tapA, byte 1 tapB) */
    chan_t *ctab = (chan_t *)(wa + PG * BM * 32 + ((PG * BM * 2 + 15) & ~15));
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
    for (int ci0 = 0; ci0 < Cip; ci0 += 32) {
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
        if constexpr (IS_MX4(T)) if (!G && !sp.sr && !upc) {   /* mx4 input without transform: the chunk is one stored block -> copy the row and its scale */
            const chan_t c0 = ctab[0];
            for (int pos = threadIdx.x; pos < TT; pos += 256) {
                int ix = pos % 18, iy = (pos / 18) % 10, iz = pos / 180;
                int gz = oz0 - 1 + iz, gy = oy0 - 1 + iy, gx = ox0 - 1 + ix;
                bool inb = c0.p && gz >= 0 && gz < D && gy >= 0 && gy < H && gx >= 0 && gx < W;
                uint4 h0 = make_uint4(0u, 0u, 0u, 0u);
                unsigned sc = 1u;
                if (inb) {
                    size_t off = ((size_t)gz * H + gy) * W + gx;
                    const uint8_t *src = (const uint8_t *)c0.p + off * c0.rb;
                    if (c0.bw == 32) h0 = __ldg((const uint4 *)src); else { uint2 u = __ldg((const uint2 *)src); h0.x = u.x; h0.y = u.y; }
                    sc = c0.sp[off];
                }
                *(uint4 *)(sx + pos * 16) = h0;
                sxs[pos] = (uint8_t)sc;
            }
        }
        if (!IS_MX4(T) || G || sp.sr || upc) for (int pos = threadIdx.x; pos < TT; pos += 256) {   /* stage: dequantise / read, transform, requantise to e2m1 */
            int ix = pos % 18, iy = (pos / 18) % 10, iz = pos / 180;
            int gz = oz0 - 1 + iz, gy = oy0 - 1 + iy, gx = ox0 - 1 + ix;
            bool inb = gz >= 0 && gz < D && gy >= 0 && gy < H && gx >= 0 && gx < W;
            size_t off = inb ? ((size_t)gz * H + gy) * W + gx : 0;
            float v[32];
            if (upc) stage_up_smem(cu, cus, (oz0 >> 1) - 2, (oy0 >> 1) - 2, (ox0 >> 1) - 2, gz, gy, gx, D, H, W, inb, v); else stage_row32<T>(ctab, off, inb, G, v, cab, gmask);
            unsigned amu = 0u;
#pragma unroll
            for (int k = 0; k < 32; k++) amu = amax_u(amu, v[k]);
            int e = mx_exp(__uint_as_float(amu), 1.f / 6.f);
            float m = exp2i(-e);
            if (sp.sr) {   /* gradient operand: exact stochastic rounding keyed by the element (deterministic per call): one hash and three
                              remixes per 8 values, the nibbles straight from the rounding (sr_e2m1_word) */
                const uint64_t vid = (((uint64_t)n * Cip + ci0) * D + gz) * (uint64_t)H * W + (uint64_t)gy * W + gx;
                unsigned wd[4];
#pragma unroll
                for (int q = 0; q < 4; q++) { uint32_t hh[4]; sr_hash4(sp.sr, vid * 4 + q, hh); wd[q] = sr_e2m1_word(v + 8 * q, m, hh); }
                *(uint4 *)(sx + pos * 16) = make_uint4(wd[0], wd[1], wd[2], wd[3]);
            } else
            *(uint4 *)(sx + pos * 16) = make_uint4(cvt_e2m1x8(v, m), cvt_e2m1x8(v + 8, m), cvt_e2m1x8(v + 16, m), cvt_e2m1x8(v + 24, m));
            sxs[pos] = (uint8_t)(e + 127);
        }
        for (int p0 = 0; p0 < 14; p0 += PG) {
            __syncthreads();
            for (int i = threadIdx.x; i < PG * BM * 2; i += 256) {
                int pp = i / (BM * 2), r = i % (BM * 2), c = r >> 1, h = r & 1, tap = 2 * (p0 + pp) + h;
                *(uint4 *)(wa + pp * BM * 32 + sw16(c, h)) = __ldg((const uint4 *)(wq + ((size_t)tap * Cop + co0 + c) * (Cip / 2) + ci0 / 2));
            }
            for (int i = threadIdx.x; i < PG * BM; i += 256) {
                int pp = i / BM, c = i % BM, tap = 2 * (p0 + pp);
                was[i] = (unsigned short)(wsc[((size_t)tap * Cop + co0 + c) * nch + ci0 / 32] | (wsc[((size_t)(tap + 1) * Cop + co0 + c) * nch + ci0 / 32] << 8));
            }
            __syncthreads();
#pragma unroll
            for (int pp = 0; pp < PG; pp++) {
                const int tA = 2 * (p0 + pp), tB = min(tA + 1, 26);
                const int rA = ((wz * RZ + tA / 9) * 10 + wr + (tA / 3) % 3) * 18 + tA % 3, rB = ((wz * RZ + tB / 9) * 10 + wr + (tB / 3) % 3) * 18 + tB % 3;
                unsigned bfr[NR][4], sb[NR][2];
#pragma unroll
                for (int r = 0; r < NR; r++) {
                    const int rowo = (r >> 1) * 180 + (r & 1) * 18;   /* z plane r/2, row r%2 of the warp's tile */
                    int mat = lane >> 3, q = mat >> 1, isB = mat & 1;
                    int pos = (isB ? rB : rA) + rowo + q * 8 + (lane & 7);
                    ldsm_x4(bfr[r], sx + pos * 16);
#pragma unroll
                    for (int q2 = 0; q2 < 2; q2++) sb[r][q2] = (unsigned)sxs[rA + rowo + q2 * 8 + g] | ((unsigned)sxs[rB + rowo + q2 * 8 + g] << 8);
                }
#pragma unroll
                for (int m = 0; m < MT; m++) {
                    unsigned af[4];
                    int mat = lane >> 3, row = m * 16 + (mat & 1) * 8 + (lane & 7);
                    ldsm_x4(af, wa + pp * BM * 32 + sw16(row, mat >> 1));
                    unsigned sa = was[pp * BM + m * 16 + g + 8 * (t & 1)];
#pragma unroll
                    for (int r = 0; r < NR; r++) { mma_f4(acc[m][r][0], af, bfr[r], sa, sb[r][0]); mma_f4(acc[m][r][1], af, bfr[r] + 2, sa, sb[r][1]); }
                }
            }
        }
    }
    fwd_epilogue<MT, NRX, TO, true>(acc, smem_raw, y, b, n, co0, Co, D, H, W, oz0, oy0, ox0, wz, wr, osum, Go, sp, N);
}
template <int MT, int TZ, typename T, typename TO> void launch_f4(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, int Cip, gnp_t gp, double *osum, int Go, split_t sp) {
    constexpr int TT = (TZ + 2) * 180;
    size_t smem = (size_t)TT * 16 + ((TT + 127) & ~127) + 7 * MT * 16 * 32 + ((7 * MT * 16 * 2 + 15) & ~15) + 32 * sizeof(chan_t) + 32 * 8 + 16 + (sp.up ? (TZ / 2 + 4) * 96 * 33 + 16 : 0);
    if (smem < 8 * 256 * sizeof(float)) smem = 8 * 256 * sizeof(float);
    static int attr[8];
    if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)conv_fwd_f4_k<MT, TZ, T, TO>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
    conv_fwd_f4_k<MT, TZ, T, TO><<<grid, 256, smem>>>((const T *)x, wq, ws, b, (TO *)y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, Cip, gp, osum, Go, sp);
}
/* ---- one or two 32-channel input chunks (Cip == 32 NCH, no fused upsample): conv_fwd_f4_k with the input staged once per z
   tile for all output tiles, and z tiles walked by the block with each chunk's input planes in a ring (TZ + 2 slots, plane gz in
   slot (gz + 1) % NP), so after the first tile only TZ planes are staged. Staging, MMA order (chunks, then tap pairs) and
   epilogue as conv_fwd_f4_k (identical results); the epilogue scratch shares the weight stage (the epilogue starts after every
   warp's MMA; the next weight stage after a barrier). ---- */
template <int MT, int TZ, int NCH> __host__ __device__ constexpr unsigned f4r_wa_off() {   /* the rings, then the channel tables */
    constexpr int TT = (TZ + 2) * 180;
    return (unsigned)((NCH * (TT * 16 + ((TT + 127) & ~127)) + NCH * 32 * 32 + NCH * 32 * 8 + 16 + 127) & ~127);
}
template <int MT, int TZ, int NCH, typename T, typename TO>
__global__ void __launch_bounds__(256, 2) conv_fwd_f4r_k(const T *__restrict__ x, const uint8_t *__restrict__ wq, const uint8_t *__restrict__ wsc,
                                                      const float *__restrict__ b, TO *__restrict__ y,
                                                      int N, int Ci, int D, int H, int W, int Co, int Cop, gnp_t gp, double *__restrict__ osum, int Go, split_t sp, int ZC) {
    constexpr int BM = MT * 16, PG = 7, RZ = TZ / 2, NR = 2 * RZ, NP = TZ + 2, TT = NP * 180, Cip = 32 * NCH, RB = TT * 16 + ((TT + 127) & ~127);
    extern __shared__ __align__(128) unsigned char smem_raw[];
    /* chunk c: sx = smem_raw + c RB [NP slots x 180 pos][16 B], sxs = sx + TT * 16 [TT] */
    chan_t *ctab = (chan_t *)(smem_raw + NCH * RB);                          /* [NCH][32] */
    float2 *cab = (float2 *)(ctab + NCH * 32); unsigned *sgm = (unsigned *)(cab + NCH * 32);   /* [NCH][32], [NCH] */
    uint8_t *wa = smem_raw + f4r_wa_off<MT, TZ, NCH>();                     /* [PG pair][BM co][32 B] (tapA 16 B | tapB 16 B), swizzled */
    unsigned short *was = (unsigned short *)(wa + PG * BM * 32);           /* [PG][BM] */
    unsigned char *scr = wa;                                                 /* epilogue scratch (after the MMA) */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wz = warp >> 2, wr = (warp & 3) * 2;
    const int ox0 = blockIdx.x * 16, oy0 = blockIdx.y * 8;
    const int nzt = (D + TZ - 1) / TZ, nzc = (nzt + ZC - 1) / ZC;
    const int zc = blockIdx.z % nzc, n = blockIdx.z / nzc;
    const int nmt = Cop / BM, nch = NCH;
    const size_t plane = (size_t)D * H * W;
    const int Cx = sp.x2 ? sp.c_split : Ci;
    const bool G = gp.G != 0 || sp.gp2.G != 0;
    if (threadIdx.x < 32 * NCH) {
        const int cp = threadIdx.x;   /* padded input channel */
        const chan_t c = make_chan(IS_MX(T) ? seg_ci(cp, Ci, Cx, (Cx + 31) / 32 * 32) : cp, Ci, Cx, n, plane, x, sp, gp, N);
        ctab[cp] = c;
        cab[cp] = make_float2(c.a, c.b);
        const unsigned gm = __ballot_sync(0xffffffffu, c.g != 0);
        if ((cp & 31) == 0) sgm[cp >> 5] = gm;
    }
    __syncthreads();
    for (int tz = zc * ZC; tz < nzt && tz < (zc + 1) * ZC; tz++) {
        const int oz0 = tz * TZ, p0n = tz == zc * ZC ? 0 : 2, npos = (NP - p0n) * 180;   /* tile planes p0n .. NP - 1 are new */
        __syncthreads();   /* the previous tile's MMA (and epilogue scratch use) is done */
#pragma unroll
        for (int ch = 0; ch < NCH; ch++) {
            uint8_t *sx = smem_raw + ch * RB, *sxs = sx + TT * 16;
            const chan_t *ct = ctab + 32 * ch;
            if constexpr (IS_MX4(T)) if (!G && !sp.sr) {   /* mx4 input without transform: the chunk is one stored block -> copy the row and its scale */
                const chan_t c0 = ct[0];
                for (int li = threadIdx.x; li < npos; li += 256) {
                    const int pl = p0n + li / 180, rem = li % 180, ix = rem % 18, iy = rem / 18;
                    const int gz = oz0 - 1 + pl, gy = oy0 - 1 + iy, gx = ox0 - 1 + ix, pos = ((gz + 1) % NP) * 180 + rem;
                    const bool inb = c0.p && gz >= 0 && gz < D && gy >= 0 && gy < H && gx >= 0 && gx < W;
                    uint4 h0 = make_uint4(0u, 0u, 0u, 0u);
                    unsigned sc = 1u;
                    if (inb) {
                        size_t off = ((size_t)gz * H + gy) * W + gx;
                        const uint8_t *src = (const uint8_t *)c0.p + off * c0.rb;
                        if (c0.bw == 32) h0 = __ldg((const uint4 *)src); else { uint2 u = __ldg((const uint2 *)src); h0.x = u.x; h0.y = u.y; }
                        sc = c0.sp[off];
                    }
                    *(uint4 *)(sx + pos * 16) = h0;
                    sxs[pos] = (uint8_t)sc;
                }
            }
            if (!IS_MX4(T) || G || sp.sr) for (int li = threadIdx.x; li < npos; li += 256) {   /* stage: dequantise / read, transform, requantise */
                const int pl = p0n + li / 180, rem = li % 180, ix = rem % 18, iy = rem / 18;
                const int gz = oz0 - 1 + pl, gy = oy0 - 1 + iy, gx = ox0 - 1 + ix, pos = ((gz + 1) % NP) * 180 + rem;
                const bool inb = gz >= 0 && gz < D && gy >= 0 && gy < H && gx >= 0 && gx < W;
                const size_t off = inb ? ((size_t)gz * H + gy) * W + gx : 0;
                float v[32];
                stage_row32<T>(ct, off, inb, G, v, cab + 32 * ch, sgm[ch]);
                unsigned amu = 0u;
#pragma unroll
                for (int k = 0; k < 32; k++) amu = amax_u(amu, v[k]);
                const int e = mx_exp(__uint_as_float(amu), 1.f / 6.f);
                const float m = exp2i(-e);
                if (sp.sr) {
                    const uint64_t vid = (((uint64_t)n * Cip + 32 * ch) * D + gz) * (uint64_t)H * W + (uint64_t)gy * W + gx;
                    unsigned wd[4];
#pragma unroll
                    for (int q = 0; q < 4; q++) { uint32_t hh[4]; sr_hash4(sp.sr, vid * 4 + q, hh); wd[q] = sr_e2m1_word(v + 8 * q, m, hh); }
                    *(uint4 *)(sx + pos * 16) = make_uint4(wd[0], wd[1], wd[2], wd[3]);
                } else
                *(uint4 *)(sx + pos * 16) = make_uint4(cvt_e2m1x8(v, m), cvt_e2m1x8(v + 8, m), cvt_e2m1x8(v + 16, m), cvt_e2m1x8(v + 24, m));
                sxs[pos] = (uint8_t)(e + 127);
            }
        }
        for (int ct = 0; ct < nmt; ct++) {   /* every output tile from the same staged input */
            const int co0 = ct * BM;
            float acc[MT][NR][2][4];
#pragma unroll
            for (int m = 0; m < MT; m++) for (int r = 0; r < NR; r++) for (int q = 0; q < 2; q++) for (int k = 0; k < 4; k++) acc[m][r][q][k] = 0.f;
            for (int ch = 0; ch < NCH; ch++) {
                const uint8_t *sx = smem_raw + ch * RB, *sxs = sx + TT * 16;
                for (int p0 = 0; p0 < 14; p0 += PG) {
                    __syncthreads();
                    for (int i = threadIdx.x; i < PG * BM * 2; i += 256) {
                        int pp = i / (BM * 2), r = i % (BM * 2), c = r >> 1, h = r & 1, tap = 2 * (p0 + pp) + h;
                        *(uint4 *)(wa + pp * BM * 32 + sw16(c, h)) = __ldg((const uint4 *)(wq + ((size_t)tap * Cop + co0 + c) * (Cip / 2) + 16 * ch));
                    }
                    for (int i = threadIdx.x; i < PG * BM; i += 256) {
                        int pp = i / BM, c = i % BM, tap = 2 * (p0 + pp);
                        was[i] = (unsigned short)(wsc[((size_t)tap * Cop + co0 + c) * nch + ch] | (wsc[((size_t)(tap + 1) * Cop + co0 + c) * nch + ch] << 8));
                    }
                    __syncthreads();
#pragma unroll
                    for (int pp = 0; pp < PG; pp++) {
                        const int tA = 2 * (p0 + pp), tB = min(tA + 1, 26);
                        unsigned bfr[NR][4], sb[NR][2];
#pragma unroll
                        for (int r = 0; r < NR; r++) {
                            /* rows of the warp's output row r for taps A / B: tile plane wz RZ + r / 2 + tap / 9 -> its ring slot */
                            const int sA = (oz0 + wz * RZ + (r >> 1) + tA / 9) % NP, sB = (oz0 + wz * RZ + (r >> 1) + tB / 9) % NP;
                            const int rA = sA * 180 + (wr + (r & 1) + (tA / 3) % 3) * 18 + tA % 3, rB = sB * 180 + (wr + (r & 1) + (tB / 3) % 3) * 18 + tB % 3;
                            int mat = lane >> 3, q = mat >> 1, isB = mat & 1;
                            ldsm_x4(bfr[r], sx + ((isB ? rB : rA) + q * 8 + (lane & 7)) * 16);
#pragma unroll
                            for (int q2 = 0; q2 < 2; q2++) sb[r][q2] = (unsigned)sxs[rA + q2 * 8 + g] | ((unsigned)sxs[rB + q2 * 8 + g] << 8);
                        }
#pragma unroll
                        for (int m = 0; m < MT; m++) {
                            unsigned af[4];
                            int mat = lane >> 3, row = m * 16 + (mat & 1) * 8 + (lane & 7);
                            ldsm_x4(af, wa + pp * BM * 32 + sw16(row, mat >> 1));
                            unsigned sa = was[pp * BM + m * 16 + g + 8 * (t & 1)];
#pragma unroll
                            for (int r = 0; r < NR; r++) { mma_f4(acc[m][r][0], af, bfr[r], sa, sb[r][0]); mma_f4(acc[m][r][1], af, bfr[r] + 2, sa, sb[r][1]); }
                        }
                    }
                }
            }
            fwd_epilogue<MT, NR, TO, true>(acc, scr, y, b, n, co0, Co, D, H, W, oz0, oy0, ox0, wz, wr, osum, Go, sp, N, f4r_wa_off<MT, TZ, NCH>());
        }
    }
}
template <int MT, int TZ, int NCH, typename T, typename TO> static void launch_f4r(const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp) {
    constexpr int NR = TZ, RBM = IS_MX(TO) ? 32 * MX_BITS(TO) / 8 : 0, BM = MT * 16;
    size_t scr = (size_t)8 * 256 * sizeof(float);
    if (IS_MX(TO) && scr < (size_t)512 + 8 * NR * 16 * (RBM + 1)) scr = (size_t)512 + 8 * NR * 16 * (RBM + 1);
    const size_t wst = (size_t)7 * BM * 32 + ((7 * BM * 2 + 15) & ~15);
    if (IS_MX(TO) && MT % 2 == 0) {   /* the transposed MX epilogue (fwd_epi_mxt): whole rows when that adds <= 2 KB, else half rows */
        const size_t s0 = scr > wst ? scr : wst;
        if (s0 < FEPI_SCR(8)) scr = FEPI_SCR(8);
        if (s0 < FEPI_SCR(16) && FEPI_SCR(16) <= s0 + 2048) scr = FEPI_SCR(16);
    }
    const size_t smem = f4r_wa_off<MT, TZ, NCH>() + (scr > wst ? scr : wst);
    const int nzt = nblk_(xs.d, TZ);
    const size_t base = (size_t)nblk_(xs.w, 16) * nblk_(xs.h, 8) * xs.n;
    static int zc_env = -2;
    if (zc_env == -2) zc_env = getenv("UFSM_F4R_ZC") ? atoi(getenv("UFSM_F4R_ZC")) : -1;
    int ZC = zc_env > 0 ? zc_env : nzt < 8 ? nzt : 8;
    while (ZC > 1 && base * nblk_(nzt, ZC) < 480) ZC--;
    const dim3 grid(nblk_(xs.w, 16), nblk_(xs.h, 8), (unsigned)(nblk_(nzt, ZC) * xs.n));
    static int attr[8];
    if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)conv_fwd_f4r_k<MT, TZ, NCH, T, TO>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
    conv_fwd_f4r_k<MT, TZ, NCH, T, TO><<<grid, 256, smem>>>((const T *)x, wq, ws, b, (TO *)y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, gp, osum, Go, sp, ZC);
}
/* prepared-weight memo: one (wq, ws) pair per conv id (split_t::wkey, set by nn.cu from layer / conv / pass) and device, valid
   while the weight pointer, the shapes and the step counter (lp_wmemo_step, from nn_set_sr_step) are unchanged; wkey 0 = no
   memo (prep every call, as before). lp_wmemo_clear() after any weight change that is not a training step (checkpoint load,
   EMA swap). UFSM_W4_MEMO=0 disables it. */
extern unsigned g_wmemo_step, g_wmemo_gen;
#define WMEMO_N 4096   /* > every conv_wkey (NN_MAXLAYER 32 layers x 16 blocks): no two convs share an entry */
struct wmemo_s { unsigned key, step, gen; const float *w; int Cop, Cip, Cx, CxP, Ox, OxP; uint8_t *wq, *ws; size_t nq, ns; }; extern wmemo_s g_wmemo[8][WMEMO_N];
static int wmemo_on(void) { static int on = -1; if (on < 0) on = getenv("UFSM_W4_MEMO") ? atoi(getenv("UFSM_W4_MEMO")) : 1; return on; }
/* returns 1 if the buffers already hold this weight (no prep needed), 0 if they must be filled; sets *wq / *ws */
static int wmemo_get(unsigned key, const float *w, int Cop, int Cip, int Cx, int CxP, int Ox, int OxP, size_t nq, size_t ns, uint8_t **wq, uint8_t **ws, int slot) {
    if (!key || !wmemo_on()) { *wq = lp_buf<uint8_t>(slot, nq); *ws = lp_buf<uint8_t>(slot + 1, ns); return 0; }
    const int d = cur_dev_();
    wmemo_s *e = &g_wmemo[d][key % WMEMO_N];
    const bool hit = e->key == key && e->w == w && e->Cop == Cop && e->Cip == Cip && e->Cx == Cx && e->CxP == CxP && e->Ox == Ox && e->OxP == OxP && e->step == g_wmemo_step && e->gen == g_wmemo_gen && e->nq == nq && e->ns == ns;
    if (e->nq < nq) { if (e->wq) cudaFree(e->wq); cudaMalloc(&e->wq, nq); e->nq = nq; }
    if (e->ns < ns) { if (e->ws) cudaFree(e->ws); cudaMalloc(&e->ws, ns); e->ns = ns; }
    *wq = e->wq; *ws = e->ws;
    if (hit) return 1;
    e->key = key; e->w = w; e->Cop = Cop; e->Cip = Cip; e->Cx = Cx; e->CxP = CxP; e->Ox = Ox; e->OxP = OxP; e->step = g_wmemo_step; e->gen = g_wmemo_gen;
    return 0;
}
/* ---- 16-channel inputs on the fp4 MMA (m16n8k64 kind::mxf4): one k-step per tap row r = (kz, ky), K = 64 = [kx0 | kx1 | kx2 | 0]
   x 16 channels, two 32-K scale blocks [kx0 | kx1] and [kx2 | 0] (9 k-steps per n-tile). The staged tile holds, per position p,
   the 16 bytes [p | p + 1] (8 B = 16 e2m1 channels each; the last position of a row pairs with zeros), so both K blocks of output
   voxel x are 16-byte-aligned ldmatrix rows: [x | x + 1] and [x + 2 | x + 3] (the kx = 3 half meets zero weights; e2m1 has no
   NaN encoding and both halves share the pair scale, so it adds nothing). B scale: one ue8m0 per pair (the mma's 32-element
   block: max over the two positions, so each position is quantised once per pair it belongs to), A scales one per (row, co,
   32-K block). Weights wq4p[r][Cop][32 B], ws[r][Cop][2]. Staging shared with the fp8 version (stage_row16). */
__global__ void prep_w4p_k(const float *__restrict__ w, uint8_t *__restrict__ wq, uint8_t *__restrict__ ws, int Co, int Ci, int Cop, int Ox, int OxP, int Cs = -1, int cofs = 0);   /* defined in lp_f4fwd.cu */
/* the epilogue scratch of conv_fwd_f4p_k: after its ring and tables, so the halo planes survive into the next z tile */
template <int MT, int TZ> __host__ __device__ constexpr unsigned f4p_scr_off() {
    constexpr int BM = MT * 16, TT = (TZ + 2) * 180;
    return (unsigned)((TT * 16 + ((TT + 127) & ~127) + 9 * BM * 32 + ((9 * BM * 2 + 15) & ~15) + 32 * 32 + TT * 4 + 127) & ~127);
}
template <int MT, int TZ, typename T, typename TO>
__global__ void __launch_bounds__(256, 2) conv_fwd_f4p_k(const T *__restrict__ x, const uint8_t *__restrict__ wq, const uint8_t *__restrict__ wsc,
                                                      const float *__restrict__ b, TO *__restrict__ y,
                                                      int N, int Ci, int D, int H, int W, int Co, int Cop, gnp_t gp, double *__restrict__ osum, int Go, split_t sp, int ZC) {
    constexpr int BM = MT * 16, RZ = TZ / 2, NR = 2 * RZ, NP = TZ + 2, TT = NP * 180;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    uint8_t *sx = smem_raw;                    /* [NP slots x 180 pos][16 B] = [pos | pos + 1]; input plane gz in slot (gz + 1) % NP */
    uint8_t *sxs = sx + TT * 16;               /* [TT] pair scales */
    uint8_t *wa = sxs + ((TT + 127) & ~127);   /* [9 r][BM co][32 B] swizzled */
    unsigned short *was = (unsigned short *)(wa + 9 * BM * 32);   /* [9][BM] (byte 0 block 0, byte 1 block 1) */
    chan_t *ctab = (chan_t *)(wa + 9 * BM * 32 + ((9 * BM * 2 + 15) & ~15));   /* [32] */
    unsigned *pam = (unsigned *)(ctab + 32);    /* [TT] per-position amax (bits) */
    unsigned char *scr = smem_raw + f4p_scr_off<MT, TZ>();   /* epilogue scratch */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wz = warp >> 2, wr = (warp & 3) * 2;
    const int ox0 = blockIdx.x * 16, oy0 = blockIdx.y * 8;
    int bz = blockIdx.z;
    const int nzt = (D + TZ - 1) / TZ, nzc = (nzt + ZC - 1) / ZC;
    const int zc = bz % nzc; bz /= nzc;
    const int nmt = Cop / BM;
    const int co0 = (bz % nmt) * BM; const int n = bz / nmt;
    const size_t plane = (size_t)D * H * W;
    const bool G = gp.G != 0;
    for (int i = threadIdx.x; i < 9 * BM * 2; i += 256) {
        const int r = i / (BM * 2), q = i % (BM * 2), c = q >> 1, h = q & 1;
        *(uint4 *)(wa + r * BM * 32 + sw16(c, h)) = __ldg((const uint4 *)(wq + ((size_t)r * Cop + co0 + c) * 32 + h * 16));
    }
    for (int i = threadIdx.x; i < 9 * BM; i += 256) { const size_t o = ((size_t)(i / BM) * Cop + co0 + i % BM) * 2; was[i] = (unsigned short)(wsc[o] | (wsc[o + 1] << 8)); }
    if (threadIdx.x < 32) ctab[threadIdx.x] = make_chan(threadIdx.x < 16 ? (int)threadIdx.x : -1, Ci, Ci, n, plane, x, sp, gp, N);
    __syncthreads();
    const chan_t c0 = ctab[0];
    float2 gab[16];
#pragma unroll
    for (int k = 0; k < 16; k++) gab[k] = make_float2(ctab[k].a, ctab[k].b);
    /* z tiles of TZ output planes: the block walks ZC of them; tile oz0 reads input planes oz0 - 1 .. oz0 + TZ, of which the
       first two are the previous tile's last two (kept in the ring), so after the first tile only TZ planes are staged */
    for (int tz = zc * ZC; tz < nzt && tz < (zc + 1) * ZC; tz++) {
    const int oz0 = tz * TZ, p0 = tz == zc * ZC ? 0 : 2, npos = (NP - p0) * 180;   /* tile planes p0 .. NP - 1 are new */
    __syncthreads();   /* the previous tile's MMA has read the slots about to be replaced */
    {   /* staging: thread per position, values kept in registers; pair p = [p | p + 1] is quantised with its own scale
           s_p = amax over both positions (the MX 32-element block of the mma), so every position is quantised twice: low half
           of pair p with s_p, high half of pair p - 1 with s_(p-1) */
        constexpr int NIT = (TT + 255) / 256;
        float v[NIT][16];
#pragma unroll
        for (int it = 0; it < NIT; it++) {
            const int li = threadIdx.x + 256 * it, pl = p0 + li / 180, rem = li % 180, row = rem / 18, ix = rem - 18 * row;
            const int gz = oz0 - 1 + pl, gy = oy0 - 1 + row, gx = ox0 - 1 + ix, spos = ((gz + 1) % NP) * 180 + rem;
            const bool inb = li < npos && gz >= 0 && gz < D && gy >= 0 && gy < H && gx >= 0 && gx < W;
            stage_row16<T>(c0, Ci, plane, inb ? ((size_t)gz * H + gy) * W + gx : 0, inb, G, gab, v[it]);
            unsigned am = 0u;
#pragma unroll
            for (int k = 0; k < 16; k++) am = amax_u(am, v[it][k]);
            if (li < npos) pam[spos] = am;
        }
        __syncthreads();
#pragma unroll
        for (int it = 0; it < NIT; it++) {
            const int li = threadIdx.x + 256 * it, pl = p0 + li / 180, rem = li % 180, row = rem / 18, ix = rem - 18 * row;
            if (li >= npos) break;
            const int gz = oz0 - 1 + pl, pos = ((gz + 1) % NP) * 180 + rem;
            const unsigned a0 = pam[pos], an = ix < 17 ? pam[pos + 1] : 0u, ap = ix ? pam[pos - 1] : 0u;
            const int e = mx_exp(__uint_as_float(max(a0, an)), 1.f / 6.f), ep = mx_exp(__uint_as_float(max(a0, ap)), 1.f / 6.f);
            const float m = exp2i(-e), mp = exp2i(-ep);
            float *w = v[it];
            uint2 lo, hi;
            if (sp.sr) {   /* gradient operand: exact stochastic rounding keyed by the element (each copy with its own scale, the same
                              uniform): one hash and three remixes per 8 values, the nibbles straight from the rounding */
                const int gy = oy0 - 1 + row, gx = ox0 - 1 + ix;
                const uint64_t vid = (((uint64_t)n * D + gz) * H + gy) * (uint64_t)W + gx;
                uint32_t ha[4], hb[4];
                sr_hash4(sp.sr, vid * 2, ha); sr_hash4(sp.sr, vid * 2 + 1, hb);
                lo = make_uint2(sr_e2m1_word(w, m, ha), sr_e2m1_word(w + 8, m, hb));
                hi = make_uint2(sr_e2m1_word(w, mp, ha), sr_e2m1_word(w + 8, mp, hb));
            } else {
                lo = make_uint2(cvt_e2m1x8(w, m), cvt_e2m1x8(w + 8, m));
                hi = make_uint2(cvt_e2m1x8(w, mp), cvt_e2m1x8(w + 8, mp));
            }
            *(uint2 *)(sx + pos * 16) = lo;                       /* low half of pair[pos] */
            if (ix) *(uint2 *)(sx + (pos - 1) * 16 + 8) = hi;     /* high half of pair[pos - 1] */
            if (ix == 17) *(uint2 *)(sx + pos * 16 + 8) = make_uint2(0u, 0u);                                      /* the row's last pair: [17 | 0] */
            sxs[pos] = (uint8_t)(e + 127);
        }
    }
    __syncthreads();
    float acc[MT][NR][2][4];
#pragma unroll
    for (int m = 0; m < MT; m++) for (int r = 0; r < NR; r++) for (int q = 0; q < 2; q++) for (int k = 0; k < 4; k++) acc[m][r][q][k] = 0.f;
    const int mat = lane >> 3, kb = mat & 1, vx = (mat >> 1) * 8 + (lane & 7);
#pragma unroll 3
    for (int tr = 0; tr < 9; tr++) {
        const int kz = tr / 3, ky = tr % 3;
        unsigned bfr[NR][4], sb[NR][2];
#pragma unroll
        for (int r = 0; r < NR; r++) {
            const int rowi = ((oz0 + wz * RZ + (r >> 1) + kz) % NP) * 10 + wr + (r & 1) + ky;   /* the tile plane's ring slot */
            ldsm_x4(bfr[r], sx + (rowi * 18 + vx + 2 * kb) * 16);   /* block 0: [x | x+1], block 1: [x+2 | x+3] */
#pragma unroll
            for (int q2 = 0; q2 < 2; q2++) { const int pc = rowi * 18 + q2 * 8 + g; sb[r][q2] = (unsigned)sxs[pc] | ((unsigned)sxs[pc + 2] << 8); }   /* column g: pair scales */
        }
#pragma unroll
        for (int m = 0; m < MT; m++) {
            unsigned af[4];
            const int arow = m * 16 + (mat & 1) * 8 + (lane & 7);
            ldsm_x4(af, wa + tr * BM * 32 + sw16(arow, mat >> 1));
            const unsigned sa = was[tr * BM + m * 16 + g + 8 * (t & 1)];
#pragma unroll
            for (int r = 0; r < NR; r++) { mma_f4(acc[m][r][0], af, bfr[r], sa, sb[r][0]); mma_f4(acc[m][r][1], af, bfr[r] + 2, sa, sb[r][1]); }
        }
    }
    fwd_epilogue<MT, NR, TO, true>(acc, scr, y, b, n, co0, Co, D, H, W, oz0, oy0, ox0, wz, wr, osum, Go, sp, N, f4p_scr_off<MT, TZ>());
    }
}
template <int MT, int TZ, typename T, typename TO> void launch_f4p(dim3, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp) {
    constexpr int BM = MT * 16, RZ = TZ / 2, NR = 2 * RZ;
    constexpr int RBM = IS_MX(TO) ? 32 * MX_BITS(TO) / 8 : 0;   /* the epilogue's MX smem store path: 512 + 8 warps x NR x 16 x (RBM + 1) */
    size_t scr = (size_t)8 * 256 * sizeof(float);
    if (IS_MX(TO) && scr < (size_t)512 + 8 * NR * 16 * (RBM + 1)) scr = (size_t)512 + 8 * NR * 16 * (RBM + 1);
    const size_t smem = f4p_scr_off<MT, TZ>() + scr;
    (void)BM;
    /* z tiles per block: the ring saves 2 of TZ + 2 staged planes per tile after the first; keep >= 4 blocks per SM-pair of waves */
    const int nzt = nblk_(xs.d, TZ), nmt = Cop / BM;
    const size_t base = (size_t)nblk_(xs.w, 16) * nblk_(xs.h, 8) * nmt * xs.n;
    static int zc_env = -2;
    if (zc_env == -2) zc_env = getenv("UFSM_F4P_ZC") ? atoi(getenv("UFSM_F4P_ZC")) : -1;
    int ZC = zc_env > 0 ? zc_env : nzt < 8 ? nzt : 8;
    while (ZC > 1 && base * nblk_(nzt, ZC) < 480) ZC--;
    const dim3 grid(nblk_(xs.w, 16), nblk_(xs.h, 8), (unsigned)(nblk_(nzt, ZC) * nmt * xs.n));   /* (the caller's grid argument is unused) */
    static int attr[8];
    if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)conv_fwd_f4p_k<MT, TZ, T, TO>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
    conv_fwd_f4p_k<MT, TZ, T, TO><<<grid, 256, smem>>>((const T *)x, wq, ws, b, (TO *)y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, gp, osum, Go, sp, ZC);
}
static int wmemo_get(unsigned key, const float *w, int Cop, int Cip, int Cx, int CxP, int Ox, int OxP, size_t nq, size_t ns, uint8_t **wq, uint8_t **ws, int slot = 2);
template <typename T, typename TO> static void p16_f4(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, int Cop, int Ox, int OxP, gnp_t gp, double *osum, int Go, split_t sp) {
    const size_t nq = (size_t)9 * Cop * 32, ns = (size_t)9 * Cop * 2;
    const int pOx = IS_MX(TO) && sp.y2 ? Ox : -1;
    uint8_t *wq, *ws;
    if (!wmemo_get(sp.wkey, w, Cop, -16, -1, 0, pOx, OxP, nq, ns, &wq, &ws)) prep_w4p_k<<<nblk_(ns, 128), 128>>>(w, wq, ws, cout, xs.c, Cop, pOx, OxP);   /* Cip -16: packed layout */
    int MT = Cop % 64 == 0 ? 4 : Cop % 32 == 0 ? 2 : !IS_MX(TO) && Cop == 48 ? 3 : 1;   /* 48 plane-major outputs (dec0.c1 backward-data): one tile, the input staged once */
    const int nmt = Cop / (MT * 16);
    static int tz_env = -1;
    if (tz_env < 0) tz_env = getenv("UFSM_F4_TZ") ? atoi(getenv("UFSM_F4_TZ")) : 0;
    int TZ = tz_env ? tz_env : (MT <= 2 && xs.d >= 16 ? 4 : 2);
    if (MT == 4) TZ = 2;
    (void)nmt;
    switch (MT * 10 + TZ) {
    case 12: launch_f4p<1, 2, T, TO>(dim3(), x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    case 14: launch_f4p<1, 4, T, TO>(dim3(), x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    case 22: launch_f4p<2, 2, T, TO>(dim3(), x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    case 24: launch_f4p<2, 4, T, TO>(dim3(), x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    case 32: launch_f4p<3, 2, T, TO>(dim3(), x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    case 34: launch_f4p<3, 4, T, TO>(dim3(), x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    default: launch_f4p<4, 2, T, TO>(dim3(), x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); break;
    }
}
template <typename T, typename TO> void fwd_f4_t(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp) {
    int Cop = (cout + 15) / 16 * 16, Cip = (xs.c + 31) / 32 * 32;
    int Cx = sp.x2 ? sp.c_split : xs.c, CxP = (Cx + 31) / 32 * 32;
    int Ox = sp.y2 ? sp.o_split : cout, OxP = (Ox + 31) / 32 * 32;
    if (IS_MX(TO)) { if (sp.y2) Cop = OxP + (cout - Ox + 31) / 32 * 32; else if (cout > 16) Cop = (cout + 31) / 32 * 32; }   /* 32-row-aligned output blocks */
    if (IS_MX(T)) Cip = CxP + (xs.c - Cx + 31) / 32 * 32;                                                                /* per-tensor 32-channel segments */
    if (xs.c <= 16 && !sp.x2 && !sp.up && sp.gp2.G == 0) { p16_f4<T, TO>(x, xs, w, b, cout, y, Cop, Ox, OxP, gp, osum, Go, sp); return; }   /* K = [kx0 | kx1 | kx2 | 0] */
    const int nch = Cip / 32;
    const size_t nq = (size_t)28 * Cop * Cip / 2, ns = (size_t)28 * Cop * nch;
    const int pCx = IS_MX(T) ? Cx : -1, pOx = IS_MX(TO) && sp.y2 ? Ox : -1;
    uint8_t *wq, *ws;
    if (g_w2d < 0) g_w2d = getenv("UFSM_W4_2D") ? atoi(getenv("UFSM_W4_2D")) : 0;
    const int w2d = g_w2d;
    if (!wmemo_get(sp.wkey, w, Cop, Cip, pCx, CxP, pOx, OxP, nq, ns, &wq, &ws)) {
        if (w2d) prep_w4_2d_k<<<nblk_((size_t)28 * ((Cop + 31) / 32) * nch * 32, 128), 128>>>(w, wq, ws, cout, xs.c, Cop, Cip, pCx, CxP, pOx, OxP);
        else prep_w4_k<<<nblk_(ns, 128), 128>>>(w, wq, ws, cout, xs.c, Cop, Cip, pCx, CxP, pOx, OxP);
    }
    int MT = Cop % 64 == 0 ? 4 : Cop % 32 == 0 ? 2 : 1;   /* BM = 16 MT must divide Cop (Cop = 48 -> MT = 1) */
    int nmt = Cop / (MT * 16);
    static int tz_env = -1;
    if (tz_env < 0) tz_env = getenv("UFSM_F4_TZ") ? atoi(getenv("UFSM_F4_TZ")) : 0;
    int TZ = tz_env ? tz_env : (MT <= 2 && xs.d >= 16 ? 4 : 2);
    if (MT == 4) TZ = 2;
    static int ring_env = -1;
    if (ring_env < 0) ring_env = getenv("UFSM_F4R") ? atoi(getenv("UFSM_F4R")) : 1;
    if (ring_env && (Cip == 32 || (Cip == 64 && ring_env >= 1)) && !sp.up && !tz_env) {   /* one or two input chunks: staged once per z tile for
                                                                                         every output tile, planes in a ring */
        if (Cip == 32) switch (MT * 10 + TZ) {
        case 14: launch_f4r<1, 4, 1, T, TO>(x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); return;
        case 24: launch_f4r<2, 4, 1, T, TO>(x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); return;
        case 42: launch_f4r<4, 2, 1, T, TO>(x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); return;
        default: break;
        } else switch (MT * 10 + TZ) {
        case 14: launch_f4r<1, 4, 2, T, TO>(x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); return;
        case 24: launch_f4r<2, 4, 2, T, TO>(x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); return;
        case 42: launch_f4r<4, 2, 2, T, TO>(x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp); return;
        default: break;
        }
    }
    dim3 grid(nblk_(xs.w, 16), nblk_(xs.h, 8), (unsigned)(nblk_(xs.d, TZ) * nmt * xs.n));
    switch (MT * 10 + TZ) {
    case 12: launch_f4<1, 2, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    case 14: launch_f4<1, 4, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    case 22: launch_f4<2, 2, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    case 24: launch_f4<2, 4, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    default: launch_f4<4, 2, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    }
}
/* xbf / ybf: 0 fp32, 1 bf16, 2 fp16, 3 MX-fp8, 4 MX-fp4; instantiated: equal types, and mx8 in -> fp16 / bf16 out (backward-data) */
