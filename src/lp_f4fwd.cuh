#pragma once
/* kernels and launchers of lp_f4fwd (instantiated per type in lp_f4fwd_i*.cu) */
#include "lp_common.cuh"
#ifndef LP_SW
/* mma_f4 with the scale selectors: A row scales from lanes 4g + TA (+1), B column g's from lane 4g + TB, bytes BA / BB, + 1 (0 or 2):
   one scale register per lane then serves up to 4 x 2 MMAs (the lanes of a quad hold different fragments' scales) */
template <int BA, int TA, int BB, int TB>
__device__ __forceinline__ void mma_f4_sel(float *c, const unsigned *a, const unsigned *b, unsigned sa, unsigned sb) {
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12, %13}, %11, {%14, %15};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb),
                   "n"(BA), "n"(TA), "n"(BB), "n"(TB));
}
#endif

#ifndef LP_SW
/* sm_120a pieces of f4r_mma_grp's 4-row path, every pair and row a template constant: ldmatrix at a register + immediate address, and a
   register add the compiler cannot hoist (asm volatile: the per-lane tap-B offsets stay one instruction, not 12 live registers) */
template <int OFF> __device__ __forceinline__ void f4r_ldsm(unsigned *r, unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4+%5];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a), "n"(OFF));
}
__device__ __forceinline__ unsigned f4r_vadd(unsigned a, unsigned b) { unsigned r; asm volatile("add.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
/* the B fragment of output row R for pair PP: S[j] = shared address of (tile plane wz RZ + j, the warp's row, this lane's ldmatrix row),
   d1 / d16 = (tap B lane) x 1 / 16 positions (x 16 B) */
template <int PP, int R, int NS>
__device__ __forceinline__ void f4r_bfrag(unsigned *b, const unsigned (&S)[NS], unsigned d1, unsigned d16, int isB) {
    constexpr int tA = 2 * PP, tB = tA + 1 < 26 ? tA + 1 : 26, kzA = tA / 9, kzB = tB / 9;
    constexpr int oA = ((R & 1) + (tA / 3) % 3) * 18 + tA % 3, oB = ((R & 1) + (tB / 3) % 3) * 18 + tB % 3;
    if constexpr (kzA != kzB) {   /* (pair 4: taps in two planes) */
        const unsigned a = isB ? S[(R >> 1) + kzB] + oB * 16 : S[(R >> 1) + kzA] + oA * 16;
        f4r_ldsm<0>(b, a);
    } else if constexpr (oB == oA) f4r_ldsm<oA * 16>(b, S[(R >> 1) + kzA]);
    else {
        static_assert(oB - oA == 1 || oB - oA == 16, "tap pair offset");
        f4r_ldsm<oA * 16>(b, f4r_vadd(S[(R >> 1) + kzA], oB - oA == 1 ? d1 : d16));
    }
}
/* pairs PB + I .. PB + PN - 1 of the 4-row path (see f4r_mma_grp) */
template <int MT, int PB, int PN, int I = 0, int NS>
__device__ __forceinline__ void f4r_pairs4(float (&acc)[MT][4][2][4], const unsigned (&S)[NS], const int (&P)[3], unsigned d1, unsigned d16, int isB,
                                           const uint8_t *__restrict__ sxs, const uint8_t *__restrict__ wag, const unsigned short *__restrict__ wasg, int lane) {
    if constexpr (I < PN) {
        constexpr int BM = MT * 16, pp = PB + I, tA = 2 * pp, tB = tA + 1 < 26 ? tA + 1 : 26;
        constexpr int cA = ((tA / 3) % 3) * 18 + tA % 3, cB = ((tB / 3) % 3) * 18 + tB % 3;
        const int g = lane >> 2, t = lane & 3, mat = lane >> 3;
        (void)g; (void)t;
        const unsigned sc = (unsigned)sxs[P[tA / 9] + cA] | ((unsigned)sxs[P[tB / 9] + cB] << 8) | ((unsigned)sxs[P[tA / 9] + cA + 8] << 16) | ((unsigned)sxs[P[tB / 9] + cB + 8] << 24);
        unsigned b0[4], b1[4], b2[4], b3[4];
        f4r_bfrag<pp, 0>(b0, S, d1, d16, isB); f4r_bfrag<pp, 1>(b1, S, d1, d16, isB); f4r_bfrag<pp, 2>(b2, S, d1, d16, isB); f4r_bfrag<pp, 3>(b3, S, d1, d16, isB);
#pragma unroll
        for (int m = 0; m < MT; m++) {
            const int row = m * 16 + (mat & 1) * 8 + (lane & 7);
            unsigned af[4];
            ldsm_x4(af, wag + I * BM * 32 + sw16(row, mat >> 1));
            const unsigned sa = wasg[I * BM + m * 16 + LPA0];
            mma_f4_sel<0, 0, 0, 0>(acc[m][0][0], af, b0, sa, sc); mma_f4_sel<0, 0, 2, 0>(acc[m][0][1], af, b0 + 2, sa, sc);
            mma_f4_sel<0, 0, 0, 1>(acc[m][1][0], af, b1, sa, sc); mma_f4_sel<0, 0, 2, 1>(acc[m][1][1], af, b1 + 2, sa, sc);
            mma_f4_sel<0, 0, 0, 2>(acc[m][2][0], af, b2, sa, sc); mma_f4_sel<0, 0, 2, 2>(acc[m][2][1], af, b2 + 2, sa, sc);
            mma_f4_sel<0, 0, 0, 3>(acc[m][3][0], af, b3, sa, sc); mma_f4_sel<0, 0, 2, 3>(acc[m][3][1], af, b3 + 2, sa, sc);
        }
        f4r_pairs4<MT, PB, PN, I + 1>(acc, S, P, d1, d16, isB, sxs, wag, wasg, lane);
    }
}
#endif
/* conv_fwd_f4r_k's MMA over tap pairs PB .. PB + PN - 1 (weights of pair PB + i at index i of wag / wasg): the fragments, MMAs and
   their order of the original pair-group loop, but every tap offset a compile-time constant and the ring slots of the warp's planes
   computed once (the runtime pair index made each fragment address a chain of divisions: ~260 IMAD per 112 MMA). sm_120a with 4
   output rows per warp: the B scales through the MMA's selectors (mma_f4_sel): lane t of a quad holds row r = t's two column-g scale
   pairs (n-tile q2 0 / 1 in bytes 0-1 / 2-3), so a pair costs 4 byte loads per lane (was 16, plus 8 merges) and one scale register;
   the scale bytes the hardware reads are those of before. */
template <int MT, int RZ, int NP, int PB, int PN, int NR = 2 * RZ>
__device__ __forceinline__ void f4r_mma_grp(float (&acc)[MT][NR][2][4], const uint8_t *__restrict__ sx, const uint8_t *__restrict__ sxs,
                                            const uint8_t *__restrict__ wag, const unsigned short *__restrict__ wasg, int oz0, int wz, int wr, int lane) {
    constexpr int BM = MT * 16;
    const int g = lane >> 2, t = lane & 3;
    (void)g; (void)t;
    int sl[RZ + 2];   /* ring slot (x 180, + the warp's first row) of tile plane wz RZ + j */
#pragma unroll
    for (int j = 0; j < RZ + 2; j++) sl[j] = (NP ? (oz0 + wz * RZ + j) % (NP ? NP : 1) : wz * RZ + j) * 180 + wr * 18;   /* (NP 0: conv_fwd_f4_k's plain tile) */
    const int mat = lane >> 3, q = mat >> 1, isB = mat & 1;
#ifndef LP_SW
    if constexpr (NR == 4) {
        int P[3];   /* the scale index of lane row t's (plane t / 2 + kz, row t % 2, column g) */
#pragma unroll
        for (int kz = 0; kz < 3; kz++) P[kz] = (NP ? (oz0 + wz * RZ + (t >> 1) + kz) % (NP ? NP : 1) : wz * RZ + (t >> 1) + kz) * 180 + (wr + (t & 1)) * 18 + g;
        unsigned S[RZ + 2];
        const unsigned lb = smem_u32_(sx) + (q * 8 + (lane & 7)) * 16;
#pragma unroll
        for (int j = 0; j < RZ + 2; j++) S[j] = lb + sl[j] * 16;
        f4r_pairs4<MT, PB, PN>(acc, S, P, isB * 16u, isB * 256u, isB, sxs, wag, wasg, lane);
    } else
#endif
    {
#pragma unroll
        for (int i = 0; i < PN; i++) {
            const int pp = PB + i, tA = 2 * pp, tB = tA + 1 < 26 ? tA + 1 : 26;
            mx4b bf[NR][2];
#pragma unroll
            for (int r = 0; r < NR; r++) {
                const int rA = sl[(r >> 1) + tA / 9] + ((r & 1) + (tA / 3) % 3) * 18 + tA % 3, rB = sl[(r >> 1) + tB / 9] + ((r & 1) + (tB / 3) % 3) * 18 + tB % 3;
#ifdef LP_SW
                unsigned qa[4], qb[4];   /* int8 rows of taps A / B (see conv_fwd_f4_k) */
                lp_ldsm_s8(qa, sx, rA + q * 8 + (lane & 7), isB); lp_ldsm_s8(qb, sx, rB + q * 8 + (lane & 7), isB);
#pragma unroll
                for (int q2 = 0; q2 < 2; q2++)
                    bf[r][q2] = mx4b_make_s8(qa[2 * q2], qa[2 * q2 + 1], qb[2 * q2], qb[2 * q2 + 1],
                                             (unsigned)sxs[rA + q2 * 8 + LPB0] | ((unsigned)sxs[rB + q2 * 8 + LPB0] << 8),
                                             (unsigned)sxs[rA + q2 * 8 + LPB1] | ((unsigned)sxs[rB + q2 * 8 + LPB1] << 8));
#else
                unsigned bfr[4];
                ldsm_x4(bfr, sx + ((isB ? rB : rA) + q * 8 + (lane & 7)) * 16);
#pragma unroll
                for (int q2 = 0; q2 < 2; q2++)
                    bf[r][q2] = mx4b_make(bfr + 2 * q2, (unsigned)sxs[rA + q2 * 8 + LPB0] | ((unsigned)sxs[rB + q2 * 8 + LPB0] << 8),
                                          (unsigned)sxs[rA + q2 * 8 + LPB1] | ((unsigned)sxs[rB + q2 * 8 + LPB1] << 8));
#endif
            }
#pragma unroll
            for (int m = 0; m < MT; m++) {
                const int row = m * 16 + (mat & 1) * 8 + (lane & 7);
#ifdef LP_SW
                unsigned qa[4], qb[4];
                lp_ldsm_s8(qa, wag + (i * 2) * BM * 32, row, mat >> 1); lp_ldsm_s8(qb, wag + (i * 2 + 1) * BM * 32, row, mat >> 1);
                const mx4a A = mx4a_make_s8(qa, qb, wasg[i * BM + m * 16 + LPA0], wasg[i * BM + m * 16 + LPA1]);
#else
                unsigned af[4];
                ldsm_x4(af, wag + i * BM * 32 + sw16(row, mat >> 1));
                const mx4a A = mx4a_make(af, wasg[i * BM + m * 16 + LPA0], wasg[i * BM + m * 16 + LPA1]);
#endif
#pragma unroll
                for (int r = 0; r < NR; r++) { mma4p(acc[m][r][0], A, bf[r][0]); mma4p(acc[m][r][1], A, bf[r][1]); }
            }
        }
    }
}
/* ---- the fused 2x upsample chunk of conv_fwd_f4r_k (UPK: dec0.c1 of a 32-wide level 0, its up part one 32-channel chunk). Per z tile
   only its new fine planes are staged into the ring (conv_fwd_f4_k restaged all TZ + 2 per tile), in batches of <= 4 fine planes, each
   reading <= 4 coarse planes x 6 rows x 10 columns (the exact coarse footprint of a 16 x 8 tile's 18 x 10 halo; conv_fwd_f4_k staged
   TZ / 2 + 4 planes x 8 x 12). The coarse rows (stage_up_coarse) and each fine position's blend (stage_up_smem_h2) and requantisation
   are those of conv_fwd_f4_k, only the coarse buffer's layout differs: the same bits. The coarse buffer overlays the epilogue scratch
   (the weight stage), unused during staging. */
constexpr int F4RU_NY = 6, F4RU_NX = 10, F4RU_NR = 4 * F4RU_NY * F4RU_NX;   /* coarse rows per batch (<= 4 planes) */
template <typename T>
__device__ __forceinline__ void f4ru_coarse(const chan_t &c0, const float2 *cab, unsigned gmask, uint8_t *cu, uint8_t *cus, int cz0, int cy0, int cx0, int nz, int D, int H, int W) {
    if constexpr (!IS_MX(T)) return; else {
    const int Dc = D >> 1, Hc = H >> 1, Wc = W >> 1, n = nz * F4RU_NY * F4RU_NX;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int lx = i % F4RU_NX, ly = (i / F4RU_NX) % F4RU_NY, lz = i / (F4RU_NY * F4RU_NX), cz = cz0 + lz, cy = cy0 + ly, cx = cx0 + lx;
        float r[32];
        const bool ok = c0.p && cz >= 0 && cz < Dc && cy >= 0 && cy < Hc && cx >= 0 && cx < Wc;
        mx_row32<MX_BITS(T)>(c0, ok ? ((size_t)cz * Hc + cy) * Wc + cx : 0, ok, r);
        if (gmask && ok) {
#pragma unroll
            for (int k = 0; k < 32; k++) if ((gmask >> k) & 1u) { const float2 ab = cab[k]; r[k] = act_ab(r[k], ab.x, ab.y, true); }
        }
        unsigned am = 0u;
#pragma unroll
        for (int k = 0; k < 32; k++) { const unsigned u = __float_as_uint(fabsf(r[k])); am = u <= 0x7f800000u ? max(am, u) : am; }
        const int e = mx_exp(__uint_as_float(am), 1.f / 448.f);
        const float m = exp2i(-e);
        uint4 *dst = (uint4 *)(cu + (size_t)i * 32);
        dst[0] = make_uint4(cvt_e4m3x4(r[0] * m, r[1] * m, r[2] * m, r[3] * m), cvt_e4m3x4(r[4] * m, r[5] * m, r[6] * m, r[7] * m),
                            cvt_e4m3x4(r[8] * m, r[9] * m, r[10] * m, r[11] * m), cvt_e4m3x4(r[12] * m, r[13] * m, r[14] * m, r[15] * m));
        dst[1] = make_uint4(cvt_e4m3x4(r[16] * m, r[17] * m, r[18] * m, r[19] * m), cvt_e4m3x4(r[20] * m, r[21] * m, r[22] * m, r[23] * m),
                            cvt_e4m3x4(r[24] * m, r[25] * m, r[26] * m, r[27] * m), cvt_e4m3x4(r[28] * m, r[29] * m, r[30] * m, r[31] * m));
        cus[i] = (uint8_t)(e + 127);
    }
}
}
/* stage_up_smem_h2 over the F4RU_NY x F4RU_NX coarse layout (the same arithmetic) */
__device__ __forceinline__ void f4ru_blend(const uint8_t *cu, const uint8_t *cus, int cz0, int cy0, int cx0, int gz, int gy, int gx, int D, int H, int W, bool inb, float *v) {
#pragma unroll
    for (int k = 0; k < 32; k++) v[k] = 0.f;
    if (!inb) return;
    const int Dc = D >> 1, Hc = H >> 1, Wc = W >> 1;
    const int mz[2] = {gz >> 1, min(max((gz >> 1) + ((gz & 1) ? 1 : -1), 0), Dc - 1)}, my[2] = {gy >> 1, min(max((gy >> 1) + ((gy & 1) ? 1 : -1), 0), Hc - 1)},
              mx[2] = {gx >> 1, min(max((gx >> 1) + ((gx & 1) ? 1 : -1), 0), Wc - 1)};
    __half2 acc[16];
#pragma unroll
    for (int j = 0; j < 16; j++) acc[j] = __float2half2_rn(0.f);
#pragma unroll
    for (int a = 0; a < 2; a++)
#pragma unroll
        for (int b = 0; b < 2; b++)
#pragma unroll
            for (int c = 0; c < 2; c++) {
                const int li = ((mz[a] - cz0) * F4RU_NY + (my[b] - cy0)) * F4RU_NX + (mx[c] - cx0);
                const __half2 w3 = __float2half2_rn((a ? 0.25f : 0.75f) * (b ? 0.25f : 0.75f) * (c ? 0.25f : 0.75f) * mx_scale(cus[li]));
                const uint4 h0 = *(const uint4 *)(cu + (size_t)li * 32), h1 = *(const uint4 *)(cu + (size_t)li * 32 + 16);
                const unsigned w[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    unsigned lo, hi;
                    lo = h2_e4m3x2(w[j] & 0xffff);
                    hi = h2_e4m3x2(w[j] >> 16);
                    acc[2 * j] = __hfma2(w3, *(const __half2 *)&lo, acc[2 * j]); acc[2 * j + 1] = __hfma2(w3, *(const __half2 *)&hi, acc[2 * j + 1]);
                }
            }
#pragma unroll
    for (int j = 0; j < 16; j++) { const float2 f = __half22float2(acc[j]); v[2 * j] = f.x; v[2 * j + 1] = f.y; }
}
/* the up chunk's new planes p0n .. NP - 1 of the z tile at oz0 into its ring (plane gz in slot (gz + 1) % NP); cu: >= F4RU_NR * 33 B */
template <typename T, int NP>
__device__ __forceinline__ void f4ru_stage(const chan_t &c0, const float2 *cab, unsigned gmask, uint8_t *cu, uint8_t *sx, uint8_t *sxs, int p0n, int oz0, int oy0, int ox0,
                                           int zi0, int Di, int H, int W) {
    uint8_t *cus = cu + F4RU_NR * 32;
    const int cy0 = (oy0 >> 1) - 1, cx0 = (ox0 >> 1) - 1;
    for (int b0 = p0n; b0 < NP;) {
        const int nb = NP - b0 > 4 ? NP - b0 - 4 : NP - b0;   /* (first tile, 6 new planes: 2 + 4) */
        const int gzA = oz0 - 1 + b0 + zi0, gzB = gzA + nb - 1;   /* the batch's input planes (fine) */
        const int czlo = (gzA & 1) ? (gzA >> 1) : (gzA >> 1) - 1, czhi = (gzB & 1) ? (gzB >> 1) + 1 : (gzB >> 1);   /* <= 4 coarse planes */
        __syncthreads();   /* the coarse buffer's previous readers (the last batch's blend; the epilogue / weight stage) are done */
        f4ru_coarse<T>(c0, cab, gmask, cu, cus, czlo, cy0, cx0, czhi - czlo + 1, Di, H, W);
        __syncthreads();
        for (int li = threadIdx.x; li < nb * 180; li += 256) {
            const int pl = b0 + li / 180, rem = li % 180, ix = rem % 18, iy = rem / 18;
            const int gz = oz0 - 1 + pl, gy = oy0 - 1 + iy, gx = ox0 - 1 + ix, pos = ((gz + 1) % NP) * 180 + rem;
            const bool inb = gz + zi0 >= 0 && gz + zi0 < Di && gy >= 0 && gy < H && gx >= 0 && gx < W;
            float v[32];
            f4ru_blend(cu, cus, czlo, cy0, cx0, gz + zi0, gy, gx, Di, H, W, inb, v);
            unsigned amu = 0u;
#pragma unroll
            for (int k = 0; k < 32; k++) amu = amax_u(amu, v[k]);
            const int e = mx_exp(__uint_as_float(amu), 1.f / 6.f);
            const float m = exp2i(-e);
            lp_st_row4(sx, pos, make_uint4(cvt_e2m1x8(v, m), cvt_e2m1x8(v + 8, m), cvt_e2m1x8(v + 16, m), cvt_e2m1x8(v + 24, m)));
            sxs[pos] = (uint8_t)(e + 127);
        }
        b0 += nb;
    }
}
/* conv_fwd_f4r_k's staging of an mx4 chunk with GN+SiLU (stage_row32 + the e2m1 requantisation, the same bits): a thread's (<= 3)
   positions per round with every row load issued first (the load-to-use latency was ~1 / 4 of the staging), and, when every channel
   has a GroupNorm and one coefficient bank serves the tile (RG), the halved affine (a / 2, b / 2) in registers: SiLU(u) = h + h tanh(h)
   with h = fmaf(v, a / 2, b / 2), the bits of act_ab's 0.5 fmaf(v, a, b) (halving commutes with the rounding; dec_row16_gnh), one
   shared load and one multiply less per value */
template <int NP, bool RG>
__device__ __forceinline__ void f4r_stage_g_t(const chan_t &c0, const float2 *cab, const float2 *cab2, int zb, unsigned gmask, uint8_t *sx, uint8_t *sxs,
                                              int p0n, int npos, int oz0, int oy0, int ox0, int zi0, int Di, int H, int W) {
    constexpr int NI = 3;
    float2 hab[RG ? 32 : 1];
    if constexpr (RG) {
#pragma unroll
        for (int j = 0; j < 32; j++) { const float2 ab = cab[j]; hab[j] = make_float2(0.5f * ab.x, 0.5f * ab.y); }
    }
    for (int l0 = 0; l0 < npos; l0 += NI * 256) {
        uint2 lo[NI], hi[NI];
        unsigned rs[NI];
        int pos[NI], gzz[NI];
        bool in[NI];
#pragma unroll
        for (int k = 0; k < NI; k++) {
            const int li = l0 + threadIdx.x + 256 * k;
            const int pl = p0n + li / 180, rem = li % 180, ix = rem % 18, iy = rem / 18;
            const int gz = oz0 - 1 + pl, gy = oy0 - 1 + iy, gx = ox0 - 1 + ix;
            pos[k] = li < npos ? (NP ? ((gz + 1) % (NP ? NP : 1)) * 180 + rem : li) : -1;   /* (NP 0: conv_fwd_f4_k's tile, p0n 0) */
            gzz[k] = gz + zi0;
            in[k] = li < npos && gz + zi0 >= 0 && gz + zi0 < Di && gy >= 0 && gy < H && gx >= 0 && gx < W;
            lo[k] = make_uint2(0u, 0u); hi[k] = make_uint2(0u, 0u); rs[k] = 0u;
            if (in[k] && c0.p) {   /* mx_row32 / mxf<4>::dec_row's loads */
                const size_t off = ((size_t)(gz + zi0) * H + gy) * W + gx;
                const uint8_t *p = (const uint8_t *)c0.p + off * c0.rb;
                lo[k] = c0.bw == 8 ? make_uint2(__ldg((const unsigned *)p), 0u) : __ldg((const uint2 *)p);
                if (c0.bw == 32) hi[k] = __ldg((const uint2 *)p + 1);
                rs[k] = c0.sp[off];
            }
        }
#pragma unroll
        for (int k = 0; k < NI; k++) {
            if (pos[k] < 0) break;
            float v[32];
            if (in[k] && c0.p) {
                const float s = mx_scale(rs[k]);
                const unsigned w[4] = {lo[k].x, lo[k].y, hi[k].x, hi[k].y};
#pragma unroll
                for (int j = 0; j < 4; j++)
#pragma unroll
                    for (int q = 0; q < 4; q++) { const float2 a = dec_e2m1x2(w[j] >> (8 * q)); v[8 * j + 2 * q] = a.x * s; v[8 * j + 2 * q + 1] = a.y * s; }
            } else {
#pragma unroll
                for (int j = 0; j < 32; j++) v[j] = 0.f;
            }
            if constexpr (RG) {
#pragma unroll
                for (int j = 0; j < 32; j++) {
                    const float h = fmaf(v[j], hab[j].x, hab[j].y);
                    float th; asm("tanh.approx.f32 %0, %1;" : "=f"(th) : "f"(h));
                    v[j] = in[k] ? fmaf(h, th, h) : 0.f;
                }
            } else {
            const float2 *cb = gzz[k] >= zb ? cab2 : cab;
            if (gmask == 0xffffffffu) {
#pragma unroll
                for (int j = 0; j < 32; j++) { const float2 ab = cb[j]; v[j] = in[k] ? act_ab(v[j], ab.x, ab.y, true) : 0.f; }
            } else {
#pragma unroll
                for (int j = 0; j < 32; j++) if ((gmask >> j) & 1u) { const float2 ab = cb[j]; v[j] = in[k] ? act_ab(v[j], ab.x, ab.y, true) : 0.f; }
            }
            }
            unsigned amu = 0u;
#pragma unroll
            for (int j = 0; j < 32; j++) amu = amax_u(amu, v[j]);
            const int e = mx_exp(__uint_as_float(amu), 1.f / 6.f);
            const float m = exp2i(-e);
            lp_st_row4(sx, pos[k], make_uint4(cvt_e2m1x8(v, m), cvt_e2m1x8(v + 8, m), cvt_e2m1x8(v + 16, m), cvt_e2m1x8(v + 24, m)));
            sxs[pos[k]] = (uint8_t)(e + 127);
        }
    }
}
template <int NP>
__device__ __forceinline__ void f4r_stage_g(const chan_t &c0, const float2 *cab, const float2 *cab2, int zb, unsigned gmask, uint8_t *sx, uint8_t *sxs,
                                            int p0n, int npos, int oz0, int oy0, int ox0, int zi0, int Di, int H, int W) {
    if (gmask == 0xffffffffu && zb == INT_MAX) f4r_stage_g_t<NP, true>(c0, cab, cab2, zb, gmask, sx, sxs, p0n, npos, oz0, oy0, ox0, zi0, Di, H, W);
    else f4r_stage_g_t<NP, false>(c0, cab, cab2, zb, gmask, sx, sxs, p0n, npos, oz0, oy0, ox0, zi0, Di, H, W);
}
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
__global__ void __launch_bounds__(256, LP_MINB(2)) conv_fwd_f4_k(const T *__restrict__ x, const uint8_t *__restrict__ wq, const uint8_t *__restrict__ wsc,
                                                     const float *__restrict__ b, TO *__restrict__ y,
                                                     int N, int Ci, int D, int H, int W, int Co, int Cop, int Cip, gnp_t gp, double *__restrict__ osum, int Go, split_t sp) {
    constexpr int BM = MT * 16, PG = 7, RZ = TZ / 2, NR = 2 * RZ, TT = (TZ + 2) * 180, NRX = NR;   /* PG: tap pairs per weight stage */
    extern __shared__ __align__(128) unsigned char smem_raw[];
    uint8_t *sx = smem_raw;                         /* [TT pos][16 B] (LP_SW: [TT][32 B] int8, lp_st_row4) */
    uint8_t *sxs = sx + TT * LP_XB;                 /* [TT] */
    uint8_t *wa = sxs + ((TT + 127) & ~127);        /* [PG pair][BM co][32 B] (tapA 16 B | tapB 16 B), swizzled (LP_SW: [PG][tap][BM][32 B] int8) */
    unsigned short *was = (unsigned short *)(wa + PG * BM * 2 * LP_XB);   /* [PG][BM] (byte 0 tapA, byte 1 tapB) */
    chan_t *ctab = (chan_t *)(wa + PG * BM * 2 * LP_XB + ((PG * BM * 2 + 15) & ~15));
    float2 *cab = (float2 *)(ctab + 32); unsigned *sgm = (unsigned *)(cab + 32);   /* [32] (a, b), GN mask */
    uint8_t *cu = (uint8_t *)(sgm + 4), *cus = cu + (TZ / 2 + 4) * 96 * 32;   /* sp.up: coarse rows of the up segment (stage_up_coarse) */
    float2 *cab2 = (float2 *)(((uintptr_t)(sp.up ? cus + (TZ / 2 + 4) * 96 : cu) + 15) & ~(uintptr_t)15);   /* [32] slab GN: bank 1 (launch_f4) */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wz = warp >> 2, wr = (warp & 3) * 2;
    const int ox0 = blockIdx.x * 16, oy0 = blockIdx.y * 8;
    int bz = blockIdx.z;
    const int tz0 = sp.zw1 ? sp.zw0 / TZ : 0, nzt = sp.zw1 ? (sp.zw1 + TZ - 1) / TZ - tz0 : (D + TZ - 1) / TZ;   /* (z-window: its tiles) */
    const int oz0 = (tz0 + bz % nzt) * TZ; bz /= nzt;
    /* slab-local GroupNorm: the staged planes oz0 - 1 .. oz0 + TZ in at most two slabs, bank 1 (cab2) from plane zb */
    const int zb = gn_zb(gp, sp.gp2, oz0 - 1 + sp.zi0, oz0 + TZ + sp.zi0), zg0 = max(oz0 - 1 + sp.zi0, 0);
    if (gn_zb3(gp, sp.gp2, oz0 - 1 + sp.zi0, oz0 + TZ + sp.zi0)) __trap();
    const int nmt = Cop / BM;
    const int co0 = (bz % nmt) * BM; const int n = bz / nmt;
    const int nch = Cip / 32;
    float acc[MT][NR][2][4];
#pragma unroll
    for (int m = 0; m < MT; m++) for (int r = 0; r < NR; r++) for (int q = 0; q < 2; q++) for (int k = 0; k < 4; k++) acc[m][r][q][k] = 0.f;
    const int zi0 = sp.zi0, Di = sp.Di ? sp.Di : D;   /* z-range (nn_set_zrange): output plane z reads input plane z + zi0 of a depth-Di input */
    const size_t plane = (size_t)Di * H * W;
    const int Cx = sp.x2 ? sp.c_split : Ci;
    const bool G = gp.G != 0 || sp.gp2.G != 0;
    for (int ci0 = 0; ci0 < Cip; ci0 += 32) {
        __syncthreads();
        const bool upc = IS_MX(T) && sp.up && ci0 < (Cx + 31) / 32 * 32;   /* this chunk is the half-resolution x segment (decoder up part) */
        if (threadIdx.x < 32) {
            const int cpi = IS_MX(T) ? seg_ci(ci0 + threadIdx.x, Ci, Cx, (Cx + 31) / 32 * 32) : ci0 + threadIdx.x;
            const size_t cpl = upc ? (size_t)(Di >> 1) * (H >> 1) * (W >> 1) : plane;
            const chan_t c = make_chan(cpi, Ci, Cx, n, cpl, x, sp, gp, N, zg0);
            ctab[threadIdx.x] = c;
            cab[threadIdx.x] = make_float2(c.a, c.b);   /* compact GN coefficients for the MX staging (one broadcast LDS.64 per element) */
            if (zb != INT_MAX) { const chan_t c2 = make_chan(cpi, Ci, Cx, n, cpl, x, sp, gp, N, zb); cab2[threadIdx.x] = make_float2(c2.a, c2.b); }
            const unsigned gm = __ballot_sync(0xffffffffu, c.g != 0);
            if (threadIdx.x == 0) *sgm = gm;
        }
        __syncthreads();
        const unsigned gmask = *sgm;
        int czl = 0;   /* (sp.up == 2: the first coarse plane of f4ru_coarse's exact footprint of the tile's TZ + 2 fine planes, <= 5) */
        if (upc) {
            if (sp.up == 2) {
                const int gzA = oz0 - 1 + zi0, gzB = gzA + TZ + 1, czh = (gzB & 1) ? (gzB >> 1) + 1 : (gzB >> 1);
                czl = (gzA & 1) ? (gzA >> 1) : (gzA >> 1) - 1;
                f4ru_coarse<T>(ctab[0], cab, gmask, cu, cus, czl, (oy0 >> 1) - 1, (ox0 >> 1) - 1, czh - czl + 1, Di, H, W);
            } else stage_up_coarse<T>(ctab[0], cu, cus, ((oz0 + zi0) >> 1) - 2, (oy0 >> 1) - 2, (ox0 >> 1) - 2, TZ / 2 + 4, Di, H, W, cab, gmask);
            __syncthreads();
        }
        if constexpr (IS_MX4(T)) if (!G && !sp.sr && !upc) {   /* mx4 input without transform: the chunk is one stored block -> copy the row and its scale */
            const chan_t c0 = ctab[0];
            for (int pos = threadIdx.x; pos < TT; pos += 256) {
                int ix = pos % 18, iy = (pos / 18) % 10, iz = pos / 180;
                int gz = oz0 - 1 + iz, gy = oy0 - 1 + iy, gx = ox0 - 1 + ix;
                bool inb = c0.p && gz + zi0 >= 0 && gz + zi0 < Di && gy >= 0 && gy < H && gx >= 0 && gx < W;
                uint4 h0 = make_uint4(0u, 0u, 0u, 0u);
                unsigned sc = 1u;
                if (inb) {
                    size_t off = ((size_t)(gz + zi0) * H + gy) * W + gx;
                    const uint8_t *src = (const uint8_t *)c0.p + off * c0.rb;
                    if (c0.bw == 32) h0 = __ldg((const uint4 *)src); else if (c0.bw == 16) { uint2 u = __ldg((const uint2 *)src); h0.x = u.x; h0.y = u.y; } else h0.x = __ldg((const unsigned *)src);   /* 32 / 16 / 8 channels: 16 / 8 / 4 bytes */
                    sc = c0.sp[off];
                }
                lp_st_row4(sx, pos, h0);
                sxs[pos] = (uint8_t)sc;
            }
        }
        const bool fastg = IS_MX4(T) && G && !sp.sr && !upc;   /* GN+SiLU on a stored mx4 chunk: f4r_stage_g (loads first, the same bits) */
        if constexpr (IS_MX4(T)) if (fastg) f4r_stage_g<0>(ctab[0], cab, cab2, zb, gmask, sx, sxs, 0, TT, oz0, oy0, ox0, zi0, Di, H, W);
        if ((!IS_MX4(T) || G || sp.sr || upc) && !fastg) for (int pos = threadIdx.x; pos < TT; pos += 256) {   /* stage: dequantise / read, transform, requantise to e2m1 */
            int ix = pos % 18, iy = (pos / 18) % 10, iz = pos / 180;
            int gz = oz0 - 1 + iz, gy = oy0 - 1 + iy, gx = ox0 - 1 + ix;
            bool inb = gz + zi0 >= 0 && gz + zi0 < Di && gy >= 0 && gy < H && gx >= 0 && gx < W;
            size_t off = inb ? ((size_t)(gz + zi0) * H + gy) * W + gx : 0;
            float v[32];
            if (upc) {
                if (sp.up == 2) f4ru_blend(cu, cus, czl, (oy0 >> 1) - 1, (ox0 >> 1) - 1, gz + zi0, gy, gx, Di, H, W, inb, v);
                else stage_up_smem(cu, cus, ((oz0 + zi0) >> 1) - 2, (oy0 >> 1) - 2, (ox0 >> 1) - 2, gz + zi0, gy, gx, Di, H, W, inb, v);
            } else stage_row32<T>(ctab, off, inb, G, v, gz + zi0 >= zb ? cab2 : cab, gmask);
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
                lp_st_row4(sx, pos, make_uint4(wd[0], wd[1], wd[2], wd[3]));
            } else
            lp_st_row4(sx, pos, make_uint4(cvt_e2m1x8(v, m), cvt_e2m1x8(v + 8, m), cvt_e2m1x8(v + 16, m), cvt_e2m1x8(v + 24, m)));
            sxs[pos] = (uint8_t)(e + 127);
        }
        for (int p0 = 0; p0 < 14; p0 += PG) {
            __syncthreads();
            for (int i = threadIdx.x; i < PG * BM * 2; i += 256) {
                int pp = i / (BM * 2), r = i % (BM * 2), c = r >> 1, h = r & 1, tap = 2 * (p0 + pp) + h;
                const uint4 wv = __ldg((const uint4 *)(wq + ((size_t)tap * Cop + co0 + c) * (Cip / 2) + ci0 / 2));
#ifdef LP_SW
                lp_st_row4(wa + (pp * 2 + h) * BM * 32, c, wv);
#else
                *(uint4 *)(wa + pp * BM * 32 + sw16(c, h)) = wv;
#endif
            }
            for (int i = threadIdx.x; i < PG * BM; i += 256) {
                int pp = i / BM, c = i % BM, tap = 2 * (p0 + pp);
                was[i] = (unsigned short)(wsc[((size_t)tap * Cop + co0 + c) * nch + ci0 / 32] | (wsc[((size_t)(tap + 1) * Cop + co0 + c) * nch + ci0 / 32] << 8));
            }
            __syncthreads();
            if (p0 == 0) f4r_mma_grp<MT, RZ, 0, 0, PG>(acc, sx, sxs, wa, was, oz0, wz, wr, lane);   /* (compile-time tap offsets, f4r_mma_grp) */
            else f4r_mma_grp<MT, RZ, 0, PG, 14 - PG>(acc, sx, sxs, wa, was, oz0, wz, wr, lane);
        }
    }
    fwd_epilogue<MT, NRX, TO, true>(acc, smem_raw, y, b, n, co0, Co, D, H, W, oz0, oy0, ox0, wz, wr, osum, Go, sp, N);
}
template <int MT, int TZ, typename T, typename TO> void launch_f4(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, int Cip, gnp_t gp, double *osum, int Go, split_t sp) {
    constexpr int TT = (TZ + 2) * 180;
    const int xb = lp_xb();   /* staged row bytes: 16 (nibbles), 32 (sm_86 int8 tiles) */
    size_t smem = (size_t)TT * xb + ((TT + 127) & ~127) + 7 * MT * 16 * 2 * xb + ((7 * MT * 16 * 2 + 15) & ~15) + 32 * sizeof(chan_t) + 32 * 8 + 16 + (sp.up ? (TZ / 2 + 4) * 96 * 33 + 16 : 0)
                  + (gp.ns || sp.gp2.ns ? 32 * 8 + 16 : 0);   /* (slab GroupNorm: the second coefficient bank) */
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
#ifndef F4R_PROF
#define F4R_PROF 0   /* 1: clock64 phase sums of conv_fwd_f4r_k (thread 0: staging, weight stages, mma, epilogue), printed per call */
#endif
#if F4R_PROF
__device__ unsigned long long g_f4r_prof[8];
#define F4RP(i) do { if (threadIdx.x == 0) { const long long t_ = clock64(); atomicAdd(&g_f4r_prof[i], (unsigned long long)(t_ - t_prev)); t_prev = t_; } } while (0)
#else
#define F4RP(i) do { } while (0)
#endif
template <int MT, int TZ, int NCH> __host__ __device__ constexpr unsigned f4r_wa_off(int xb) {   /* the rings, then the channel tables (xb: staged row bytes, LP_XB / lp_xb()) */
    constexpr int TT = (TZ + 2) * 180;
    return (unsigned)((NCH * (TT * xb + ((TT + 127) & ~127)) + NCH * 32 * 32 + NCH * 32 * 8 + 16 + 127) & ~127);
}
template <int MT, int TZ, int NCH, typename T, typename TO, int UPK = 0>   /* UPK: chunk 0 is the fused 2x upsample of a 32-channel half-resolution x (dec0.c1) */
__global__ void __launch_bounds__(256, LP_MINB(2)) conv_fwd_f4r_k(const T *__restrict__ x, const uint8_t *__restrict__ wq, const uint8_t *__restrict__ wsc,
                                                      const float *__restrict__ b, TO *__restrict__ y,
                                                      int N, int Ci, int D, int H, int W, int Co, int Cop, gnp_t gp, double *__restrict__ osum, int Go, split_t sp, int ZC,
                                                      int wres) {   /* wres: all 14 tap pairs resident (one output tile, one chunk), staged once */
    constexpr int BM = MT * 16, PG = 7, RZ = TZ / 2, NR = 2 * RZ, NP = TZ + 2, TT = NP * 180, Cip = 32 * NCH, RB = TT * LP_XB + ((TT + 127) & ~127);
    extern __shared__ __align__(128) unsigned char smem_raw[];
    /* chunk c: sx = smem_raw + c RB [NP slots x 180 pos][LP_XB B], sxs = sx + TT * LP_XB [TT] */
    chan_t *ctab = (chan_t *)(smem_raw + NCH * RB);                          /* [NCH][32] */
    float2 *cab = (float2 *)(ctab + NCH * 32); unsigned *sgm = (unsigned *)(cab + NCH * 32);   /* [NCH][32], [NCH] */
    uint8_t *wa = smem_raw + f4r_wa_off<MT, TZ, NCH>(LP_XB);                     /* [PG pair][BM co][32 B] (tapA 16 B | tapB 16 B), swizzled */
    unsigned short *was = (unsigned short *)(wa + (wres ? 14 : PG) * BM * 2 * LP_XB);   /* [PG][BM] (wres: [14][BM]) */
    /* epilogue scratch (after the MMA): over the weight stage, or after the resident weights */
    const unsigned scr_off = f4r_wa_off<MT, TZ, NCH>(LP_XB) + (wres ? (unsigned)((14 * BM * 2 * LP_XB + 14 * BM * 2 + 127) & ~127) : 0u);
    unsigned char *scr = smem_raw + scr_off;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wz = warp >> 2, wr = (warp & 3) * 2;
    const int ox0 = blockIdx.x * 16, oy0 = blockIdx.y * 8;
    /* tiles [tz0, nzt): all, or the z-window's (chunks of ZC from the window's first tile; ring and staging as for the whole launch) */
    const int tz0 = sp.zw1 ? sp.zw0 / TZ : 0, nzt = sp.zw1 ? (sp.zw1 + TZ - 1) / TZ : (D + TZ - 1) / TZ, nzc = (nzt - tz0 + ZC - 1) / ZC;
    const int zc = blockIdx.z % nzc, n = blockIdx.z / nzc;
    const int nmt = Cop / BM, nch = NCH;
    const int zi0 = sp.zi0, Di = sp.Di ? sp.Di : D;   /* z-range (nn_set_zrange): output plane z reads input plane z + zi0 of a depth-Di input */
    const size_t plane = (size_t)Di * H * W;
    const int Cx = sp.x2 ? sp.c_split : Ci;
    const bool G = gp.G != 0 || sp.gp2.G != 0;
    /* slab-local GroupNorm: per z tile, its new planes in at most two slabs; bank 0 (cab) holds the first plane's slab, bank 1 (cab2,
       at the end of the dynamic shared memory) the next one's from plane zb; rebuilt when the pair changes */
    const bool slb = gp.ns || sp.gp2.ns;
    float2 *cab2 = gn_bank2(smem_raw, NCH * 32);
    int skey = -1;
    if (threadIdx.x < 32 * NCH) {
        const int cp = threadIdx.x;   /* padded input channel */
        const chan_t c = make_chan(IS_MX(T) ? seg_ci(cp, Ci, Cx, (Cx + 31) / 32 * 32) : cp, Ci, Cx, n, UPK && cp < 32 ? (size_t)(Di >> 1) * (H >> 1) * (W >> 1) : plane, x, sp, gp, N);
        ctab[cp] = c;
        cab[cp] = make_float2(c.a, c.b);
        const unsigned gm = __ballot_sync(0xffffffffu, c.g != 0);
        if ((cp & 31) == 0) sgm[cp >> 5] = gm;
    }
    if (wres) {   /* the whole weight set once (nmt == 1, NCH == 1): no weight stage or its barriers per tile */
        for (int i = threadIdx.x; i < 14 * BM * 2; i += 256) {
            const int pp = i / (BM * 2), r = i % (BM * 2), c = r >> 1, h = r & 1, tap = 2 * pp + h;
            lp_st_w4(wa, pp, BM, c, h, __ldg((const uint4 *)(wq + ((size_t)tap * Cop + c) * (Cip / 2))));
        }
        for (int i = threadIdx.x; i < 14 * BM; i += 256) {
            const int pp = i / BM, c = i % BM, tap = 2 * pp;
            was[i] = (unsigned short)(wsc[(size_t)tap * Cop + c] | (wsc[(size_t)(tap + 1) * Cop + c] << 8));
        }
    }
    __syncthreads();
#if F4R_PROF
    long long t_prev = clock64();
#endif
    for (int tz = tz0 + zc * ZC; tz < nzt && tz < tz0 + (zc + 1) * ZC; tz++) {
        const int oz0 = tz * TZ, p0n = tz == tz0 + zc * ZC ? 0 : 2, npos = (NP - p0n) * 180;   /* tile planes p0n .. NP - 1 are new */
        __syncthreads();   /* the previous tile's MMA (and epilogue scratch use) is done */
        int zb = INT_MAX;
        if (slb) {   /* (block-uniform) */
            const int z0 = oz0 - 1 + p0n + zi0, z1 = oz0 + TZ + zi0;
            if (gn_zb3(gp, sp.gp2, z0, z1)) __trap();
            zb = gn_zb(gp, sp.gp2, z0, z1);
            const int key = 2 * gn_row(gp.ns ? gp : sp.gp2, 0, max(z0, 0)) + (zb != INT_MAX);
            if (key != skey) {
                skey = key;
                if (threadIdx.x < 32 * NCH) {
                    const int cp = threadIdx.x, ci = IS_MX(T) ? seg_ci(cp, Ci, Cx, (Cx + 31) / 32 * 32) : cp;
                    const chan_t c = make_chan(ci, Ci, Cx, n, plane, x, sp, gp, N, max(z0, 0));
                    cab[cp] = make_float2(c.a, c.b); ctab[cp].a = c.a; ctab[cp].b = c.b;
                    if (zb != INT_MAX) { const chan_t c2 = make_chan(ci, Ci, Cx, n, plane, x, sp, gp, N, zb); cab2[cp] = make_float2(c2.a, c2.b); }
                }
                __syncthreads();
            }
        }
        F4RP(0);
#pragma unroll
        for (int ch = 0; ch < NCH; ch++) {
            uint8_t *sx = smem_raw + ch * RB, *sxs = sx + TT * LP_XB;
            const chan_t *ct = ctab + 32 * ch;
            if constexpr (UPK != 0 && IS_MX(T)) if (ch == 0) {   /* the up chunk: the tile's new planes in batches of <= 4 */
                f4ru_stage<T, NP>(ct[0], cab, sgm[0], smem_raw + scr_off, sx, sxs, p0n, oz0, oy0, ox0, zi0, Di, H, W);
                continue;
            }
            if constexpr (IS_MX4(T)) if (!G && !sp.sr) {   /* mx4 input without transform: the chunk is one stored block -> copy the row and its scale */
                const chan_t c0 = ct[0];
                for (int li = threadIdx.x; li < npos; li += 256) {
                    const int pl = p0n + li / 180, rem = li % 180, ix = rem % 18, iy = rem / 18;
                    const int gz = oz0 - 1 + pl, gy = oy0 - 1 + iy, gx = ox0 - 1 + ix, pos = ((gz + 1) % NP) * 180 + rem;
                    const bool inb = c0.p && gz + zi0 >= 0 && gz + zi0 < Di && gy >= 0 && gy < H && gx >= 0 && gx < W;
                    uint4 h0 = make_uint4(0u, 0u, 0u, 0u);
                    unsigned sc = 1u;
                    if (inb) {
                        size_t off = ((size_t)(gz + zi0) * H + gy) * W + gx;
                        const uint8_t *src = (const uint8_t *)c0.p + off * c0.rb;
                        if (c0.bw == 32) h0 = __ldg((const uint4 *)src); else if (c0.bw == 16) { uint2 u = __ldg((const uint2 *)src); h0.x = u.x; h0.y = u.y; } else h0.x = __ldg((const unsigned *)src);   /* 32 / 16 / 8 channels: 16 / 8 / 4 bytes */
                        sc = c0.sp[off];
                    }
                    lp_st_row4(sx, pos, h0);
                    sxs[pos] = (uint8_t)sc;
                }
            }
            if constexpr (IS_MX4(T)) if (G && !sp.sr) {   /* the usual case: GN+SiLU on a stored mx4 input (f4r_stage_g) */
                f4r_stage_g<NP>(ct[0], cab + 32 * ch, cab2 + 32 * ch, zb, sgm[ch], sx, sxs, p0n, npos, oz0, oy0, ox0, zi0, Di, H, W);
                continue;
            }
            if (!IS_MX4(T) || G || sp.sr) for (int li = threadIdx.x; li < npos; li += 256) {   /* stage: dequantise / read, transform, requantise */
                const int pl = p0n + li / 180, rem = li % 180, ix = rem % 18, iy = rem / 18;
                const int gz = oz0 - 1 + pl, gy = oy0 - 1 + iy, gx = ox0 - 1 + ix, pos = ((gz + 1) % NP) * 180 + rem;
                const bool inb = gz + zi0 >= 0 && gz + zi0 < Di && gy >= 0 && gy < H && gx >= 0 && gx < W;
                const size_t off = inb ? ((size_t)(gz + zi0) * H + gy) * W + gx : 0;
                float v[32];
                stage_row32<T>(ct, off, inb, G, v, (gz + zi0 >= zb ? cab2 : cab) + 32 * ch, sgm[ch]);
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
                    lp_st_row4(sx, pos, make_uint4(wd[0], wd[1], wd[2], wd[3]));
                } else
                lp_st_row4(sx, pos, make_uint4(cvt_e2m1x8(v, m), cvt_e2m1x8(v + 8, m), cvt_e2m1x8(v + 16, m), cvt_e2m1x8(v + 24, m)));
                sxs[pos] = (uint8_t)(e + 127);
            }
        }
        for (int ct = 0; ct < nmt; ct++) {   /* every output tile from the same staged input */
            const int co0 = ct * BM;
            float acc[MT][NR][2][4];
#pragma unroll
            for (int m = 0; m < MT; m++) for (int r = 0; r < NR; r++) for (int q = 0; q < 2; q++) for (int k = 0; k < 4; k++) acc[m][r][q][k] = 0.f;
            for (int ch = 0; ch < NCH; ch++) {
                const uint8_t *sx = smem_raw + ch * RB, *sxs = sx + TT * LP_XB;
                /* two groups of PG = 7 pairs: resident weights (wres: only the staging barrier), else each group's weights staged */
#pragma unroll
                for (int gi = 0; gi < 2; gi++) {
                    const int p0 = gi * PG;
                    if (!wres || !p0) __syncthreads();
                    F4RP(p0 ? 3 : 1);   /* 1: staging (its barrier), 3: the first pair group's mma */
                    const uint8_t *wag = wres ? wa + p0 * BM * 2 * LP_XB : wa;
                    const unsigned short *wasg = wres ? was + p0 * BM : was;
                    if (!wres) {
                        for (int i = threadIdx.x; i < PG * BM * 2; i += 256) {
                            int pp = i / (BM * 2), r = i % (BM * 2), c = r >> 1, h = r & 1, tap = 2 * (p0 + pp) + h;
                            lp_st_w4(wa, pp, BM, c, h, __ldg((const uint4 *)(wq + ((size_t)tap * Cop + co0 + c) * (Cip / 2) + 16 * ch)));
                        }
                        for (int i = threadIdx.x; i < PG * BM; i += 256) {
                            int pp = i / BM, c = i % BM, tap = 2 * (p0 + pp);
                            was[i] = (unsigned short)(wsc[((size_t)tap * Cop + co0 + c) * nch + ch] | (wsc[((size_t)(tap + 1) * Cop + co0 + c) * nch + ch] << 8));
                        }
                        __syncthreads();
                    }
                    F4RP(2);
                    if (gi == 0) f4r_mma_grp<MT, RZ, NP, 0, PG>(acc, sx, sxs, wag, wasg, oz0, wz, wr, lane);
                    else f4r_mma_grp<MT, RZ, NP, PG, 14 - PG>(acc, sx, sxs, wag, wasg, oz0, wz, wr, lane);
                }
            }
#if F4R_PROF
            __syncthreads();
#endif
            F4RP(4);   /* the second pair group's mma (to the barrier: the slowest warp) */
            fwd_epilogue<MT, NR, TO, true>(acc, scr, y, b, n, co0, Co, D, H, W, oz0, oy0, ox0, wz, wr, osum, Go, sp, N, scr_off);
            F4RP(5);
        }
    }
#if F4R_PROF
    if (threadIdx.x == 0) atomicAdd(&g_f4r_prof[7], 1ull);
#endif
}
template <int MT, int TZ, int NCH, typename T, typename TO, int UPK = 0> static void launch_f4r(const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp) {
    constexpr int NR = TZ, RBM = IS_MX(TO) ? 32 * MX_BITS(TO) / 8 : 0, BM = MT * 16;
    size_t scr = (size_t)8 * 256 * sizeof(float);
    if (IS_MX(TO) && scr < (size_t)512 + 8 * NR * 16 * (RBM + 1)) scr = (size_t)512 + 8 * NR * 16 * (RBM + 1);
    const int xb = lp_xb();   /* staged row bytes (sm_86: int8 tiles) */
    const size_t wst = (size_t)7 * BM * 2 * xb + ((7 * BM * 2 + 15) & ~15);
    if (IS_MX(TO) && MT % 2 == 0) {   /* the transposed MX epilogue (fwd_epi_mxt): whole rows when that adds <= 2 KB, else half rows */
        const size_t s0 = scr > wst ? scr : wst;
        if (s0 < FEPI_SCR(8)) scr = FEPI_SCR(8);
        if (s0 < FEPI_SCR(16) && FEPI_SCR(16) <= s0 + 2048) scr = FEPI_SCR(16);
    }
    static int wres_env = -1;
    if (wres_env < 0) wres_env = getenv("UFSM_F4R_WRES") ? atoi(getenv("UFSM_F4R_WRES")) : 1;
    const int wres = wres_env && NCH == 1 && Cop == BM && MT <= 2;   /* one output tile, one chunk: the 14 tap pairs (<= 15 KB) resident */
    const size_t wfix = f4r_wa_off<MT, TZ, NCH>(xb) + (wres ? ((size_t)(14 * BM * 2 * xb + 14 * BM * 2 + 127) & ~(size_t)127) : 0);
    if (!wres && scr < wst) scr = wst;   /* (the scratch overlays the weight stage) */
    if (UPK && scr < (size_t)F4RU_NR * 33) scr = (size_t)F4RU_NR * 33;   /* (and the up chunk's coarse rows, f4ru_stage) */
    if (IS_MX(TO) && MT == 1) scr = fepi16_scr(wfix, scr, cout, sp);   /* the 16-channel-block MX epilogue (fwd_epi_mx16) */
    const size_t smem = wfix + scr + (gp.ns || sp.gp2.ns ? (size_t)NCH * 32 * 8 + 16 : 0);   /* (slab GroupNorm: the second coefficient bank) */
    const int nzt = zw_tiles(xs.d, TZ, sp);
    const size_t base = (size_t)nblk_(xs.w, 16) * nblk_(xs.h, 8) * xs.n;
    static int zc_env = -2;
    if (zc_env == -2) zc_env = getenv("UFSM_F4R_ZC") ? atoi(getenv("UFSM_F4R_ZC")) : -1;
    int ZC = zc_env > 0 ? zc_env : nzt < 8 ? nzt : 8;
    while (ZC > 1 && base * nblk_(nzt, ZC) < 480) ZC--;
    const dim3 grid(nblk_(xs.w, 16), nblk_(xs.h, 8), (unsigned)(nblk_(nzt, ZC) * xs.n));
    static int attr[8];
    if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)conv_fwd_f4r_k<MT, TZ, NCH, T, TO, UPK>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
    conv_fwd_f4r_k<MT, TZ, NCH, T, TO, UPK><<<grid, 256, smem>>>((const T *)x, wq, ws, b, (TO *)y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, gp, osum, Go, sp, ZC, wres);
#if F4R_PROF
    if (getenv("F4R_PROF")) {
        unsigned long long h[8]; cudaDeviceSynchronize(); cudaMemcpyFromSymbol(h, g_f4r_prof, sizeof h);
        const double nb = (double)h[7], tot = (double)(h[0] + h[1] + h[2] + h[3] + h[4] + h[5]);
        fprintf(stderr, "f4r %d->%d @%dx%dx%d MT %d TZ %d NCH %d ZC %d: per block %.0f cycles: tile start %.0f%% staging %.0f%% wstage %.0f%% mma1 %.0f%% mma2 %.0f%% epilogue %.0f%%\n",
                xs.c, cout, xs.d, xs.h, xs.w, MT, TZ, NCH, ZC, tot / nb, 100 * h[0] / tot, 100 * h[1] / tot, 100 * h[2] / tot, 100 * h[3] / tot, 100 * h[4] / tot, 100 * h[5] / tot);
        cudaMemset(g_f4r_prof, 0, sizeof h); unsigned long long z[8] = {}; cudaMemcpyToSymbol(g_f4r_prof, z, sizeof z);
    }
#endif
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
/* ---- 16-channel inputs on the fp4 MMA (m16n8k64 kind::mxf4nvf4, scale_vec::4X: one ue8m0 scale per 16 K, mma_f4x4): one k-step
   per tap row r = (kz, ky), K = 64 = [kx0 | kx1 | kx2 | kx3] x 16 channels, each 16-K group with its own scale (9 k-steps per
   n-tile; the kx3 group meets zero weights). Every staged position is quantised once, with its own scale over its 16 channels:
   the stored MX layout of a <= 16-channel tensor, so an MX-fp4 input without a transform (the backward-data of an fp4 gradient)
   is copied bit-exact. The staged tile holds, per position p, the 16 bytes [p | p + 1] (8 B = 16 e2m1 channels each; the last
   position of a row pairs with zeros), so both 32-K halves of output voxel x are 16-byte-aligned ldmatrix rows: [x | x + 1] and
   [x + 2 | x + 3]. B scale bytes: s(x), s(x + 1), s(x + 2) and 1 (kx3: zero weights; e2m1 has no NaN encoding). A scales one
   per (row, co, kx). Weights wq4p[r][Cop][32 B] (8 B per kx), ws[r][Cop][4]. Staging shared with the fp8 version (stage_row16). */
__global__ void prep_w4p_k(const float *__restrict__ w, uint8_t *__restrict__ wq, uint8_t *__restrict__ ws, int Co, int Ci, int Cop, int Ox, int OxP, int Cs = -1, int cofs = 0);   /* defined in lp_f4fwd.cu */
/* the epilogue scratch of conv_fwd_f4p_k: after its ring and tables, so the halo planes survive into the next z tile */
template <int MT, int TZ> __host__ __device__ constexpr unsigned f4p_scr_off() {
    constexpr int BM = MT * 16, TT = (TZ + 2) * 180;
    return (unsigned)((TT * 16 + ((TT * 4 + 127) & ~127) + 9 * BM * 32 + 9 * BM * 4 + 32 * 32 + 16 * 8 + 127) & ~127);   /* (+ the GN pairs, MT > 1) */
}
/* one staged position of conv_fwd_f4p_k: its 8 bytes into pair[pos] (low half) and pair[pos - 1] (high half), the row's last pair
   padded with zeros, and its scale into the B scale words of the voxels whose K groups it feeds: byte 0 of word[pos], 1 of
   word[pos - 1], 2 of word[pos - 2] (byte 3, the kx3 group: 1). The MMA then loads one word per column (was three bytes). */
/* (the words of a row's columns 0..15 interleaved, column c at word 2 (c & 7) + (c >> 3), so the MMA's two columns g and 8 + g are
   one 8-byte load; columns 16, 17 in place) */
__device__ __forceinline__ int p16_wi(int ix) { return ix < 16 ? ((ix & 7) << 1 | ix >> 3) : ix; }
template <bool IL> __device__ __forceinline__ void p16_put(uint8_t *sx, unsigned *sxw, int pos, int ix, uint2 q, unsigned sc) {   /* IL: interleaved words */
    *(uint2 *)(sx + pos * 16) = q;
    if (ix) *(uint2 *)(sx + (pos - 1) * 16 + 8) = q;
    if (ix == 17) *(uint2 *)(sx + pos * 16 + 8) = make_uint2(0u, 0u);
    uint8_t *sb = (uint8_t *)(sxw + (pos - ix));
    const int w0 = IL ? p16_wi(ix) : ix, w1 = IL ? p16_wi(ix - 1) : ix - 1, w2 = IL ? p16_wi(ix - 2) : ix - 2;
    sb[4 * w0] = (uint8_t)sc; sb[4 * w0 + 3] = 0x7f;
    if (ix >= 1) sb[4 * w1 + 1] = (uint8_t)sc;
    if (ix >= 2) sb[4 * w2 + 2] = (uint8_t)sc;
}
#ifndef F4P_UNR9
#define F4P_UNR9 2   /* conv_fwd_f4p_k: the tap-row loop fully unrolled for MT <= F4P_UNR9 */
#endif
template <int N> struct f4p_nit { static constexpr int value = N; };   /* conv_fwd_f4p_k's row-load count (a compile-time tag) */
/* dec_row16_gn with the GN affine pre-halved (gh = (a / 2, b / 2)): SiLU(u) = h + h tanh(h) with h = fmaf(v, a / 2, b / 2), the
   same bits as act_ab's 0.5 fmaf(v, a, b) (halving commutes with the rounding), one multiply less per value */
template <typename GAB>
__device__ __forceinline__ void dec_row16_gnh(uint2 raw, unsigned rsc, bool in, int Ci, bool G, const GAB &gh, float *v) {
    const float s = mx_scale(rsc);
    const unsigned w[2] = {raw.x, raw.y};
#pragma unroll
    for (int j = 0; j < 2; j++)
#pragma unroll
        for (int k = 0; k < 4; k++) { const float2 a = dec_e2m1x2(w[j] >> (8 * k)); v[8 * j + 2 * k] = a.x * s; v[8 * j + 2 * k + 1] = a.y * s; }   /* (!in: ld_row16's
                                                                                                                              zero row and scale byte decode to +0) */
    if (G && in) {
        if (Ci >= 16) {   /* (block-uniform: no channel masks, 32 instructions per position less) */
#pragma unroll
            for (int k = 0; k < 16; k++) {
                const float h = fmaf(v[k], gh[k].x, gh[k].y);
                float th; asm("tanh.approx.f32 %0, %1;" : "=f"(th) : "f"(h));
                v[k] = fmaf(h, th, h);
            }
        } else {
#pragma unroll
            for (int k = 0; k < 16; k++) {
                const float h = fmaf(v[k], gh[k].x, gh[k].y);
                float th; asm("tanh.approx.f32 %0, %1;" : "=f"(th) : "f"(h));
                v[k] = k < Ci ? fmaf(h, th, h) : 0.f;
            }
        }
    }
}
template <int MT, int TZ, typename T, typename TO, bool SL = false>   /* SL: slab-local GroupNorm statistics (MT 1; its reload kept out of the usual
                                                                          instance, where its registers spilled 108 B) */
__global__ void __launch_bounds__(256, LP_MINB(2)) conv_fwd_f4p_k(const T *__restrict__ x, const uint8_t *__restrict__ wq, const uint8_t *__restrict__ wsc,
                                                      const float *__restrict__ b, TO *__restrict__ y,
                                                      int N, int Ci, int D, int H, int W, int Co, int Cop, gnp_t gp, double *__restrict__ osum, int Go, split_t sp, int ZC) {
    constexpr int BM = MT * 16, RZ = TZ / 2, NR = 2 * RZ, NP = TZ + 2, TT = NP * 180;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    uint8_t *sx = smem_raw;                    /* [NP slots x 180 pos][16 B] = [pos | pos + 1]; input plane gz in slot (gz + 1) % NP */
    unsigned *sxw = (unsigned *)(sx + TT * 16);   /* [TT] B scale words: s(p), s(p + 1), s(p + 2), 1 (p16_put) */
    uint8_t *wa = sx + TT * 16 + ((TT * 4 + 127) & ~127);   /* [9 r][BM co][32 B] swizzled */
    unsigned *was = (unsigned *)(wa + 9 * BM * 32);   /* [9][BM] (byte kx = the kx group's scale) */
    chan_t *ctab = (chan_t *)(was + 9 * BM);   /* [32] */
    float2 *gabs = (float2 *)(ctab + 32);      /* [16] the GN pairs in shared memory (MT > 1) */
    unsigned char *scr = smem_raw + f4p_scr_off<MT, TZ>();   /* epilogue scratch */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wz = warp >> 2, wr = (warp & 3) * 2;
    const int ox0 = blockIdx.x * 16, oy0 = blockIdx.y * 8;
    int bz = blockIdx.z;
    const int tz0 = sp.zw1 ? sp.zw0 / TZ : 0, nzt = sp.zw1 ? (sp.zw1 + TZ - 1) / TZ : (D + TZ - 1) / TZ, nzc = (nzt - tz0 + ZC - 1) / ZC;   /* tiles [tz0, nzt) (z-window) */
    const int zc = bz % nzc; bz /= nzc;
    const int nmt = Cop / BM;
    const int co0 = (bz % nmt) * BM; const int n = bz / nmt;
    const int zi0 = sp.zi0, Di = sp.Di ? sp.Di : D;   /* z-range (nn_set_zrange): output plane z reads input plane z + zi0 of a depth-Di input */
    const size_t plane = (size_t)Di * H * W;
    const bool G = gp.G != 0;
    for (int i = threadIdx.x; i < 9 * BM * 2; i += 256) {
        const int r = i / (BM * 2), q = i % (BM * 2), c = q >> 1, h = q & 1;
        *(uint4 *)(wa + r * BM * 32 + sw16(c, h)) = __ldg((const uint4 *)(wq + ((size_t)r * Cop + co0 + c) * 32 + h * 16));
    }
    for (int i = threadIdx.x; i < 9 * BM; i += 256) was[i] = __ldg((const unsigned *)wsc + (size_t)(i / BM) * Cop + co0 + i % BM);
    if (threadIdx.x < 32) ctab[threadIdx.x] = make_chan(threadIdx.x < 16 ? (int)threadIdx.x : -1, Ci, Ci, n, plane, x, sp, gp, N);
    if (osum) fwd_stats_zero<BM>(scr);   /* the block's statistics over all its z tiles (OWN epilogue) */
    __syncthreads();
#if F4R_PROF
    long long t_prev = clock64();
#endif
    const chan_t c0 = ctab[0];
    /* the GN pairs (MX-fp4 input: the affine pre-halved, dec_row16_gnh): in registers for MT 1; MT > 1 reads them from shared memory
       (their 32 registers spilled the doubled accumulators; its calls are mostly the copy-staged backward-data, which reads none) */
    float2 gabr[MT == 1 ? 16 : 1];
    if constexpr (MT == 1) {
#pragma unroll
        for (int k = 0; k < 16; k++) gabr[k] = IS_MX4(T) ? make_float2(0.5f * ctab[k].a, 0.5f * ctab[k].b) : make_float2(ctab[k].a, ctab[k].b);
    } else {
        if (threadIdx.x < 16) gabs[threadIdx.x] = IS_MX4(T) ? make_float2(0.5f * ctab[threadIdx.x].a, 0.5f * ctab[threadIdx.x].b) : make_float2(ctab[threadIdx.x].a, ctab[threadIdx.x].b);
        __syncthreads();
    }
    const float2 *gab = MT == 1 ? gabr : gabs;
    int srow = gp.ns ? -1 : n;   /* slab-local GroupNorm (MT 1; fwd_f4_t routes MT > 1 around): the statistics row in gabr, reloaded when
                                    a thread's next position lies in another slab (its planes ascend within a tile) */
    auto slab_ab = [&](int gz) {
        if constexpr (MT == 1 && SL) if (G && gp.ns) {
            const int rz = gn_row(gp, n, max(gz + zi0, 0));
            if (rz != srow) {
                srow = rz;
#pragma unroll
                for (int k = 0; k < 16; k++) if (k < Ci) { const float2 ab = gn_ab(gp, rz, k, Ci); gabr[k] = IS_MX4(T) ? make_float2(0.5f * ab.x, 0.5f * ab.y) : ab; }
            }
        }
    };
    bool cpy = false;   /* mx4 input without transform: the stored row (<= 16 channels, one scale) is the staged row */
    if constexpr (IS_MX4(T)) cpy = !G && !sp.sr;
    /* MX-fp4 input: the rows of a tile are loaded before it is staged -- the first tile's (NIT per thread) here, every later tile's
       (NIT2) right after the previous tile's staging, so they arrive during its MMA and epilogue (the staging waited on them) */
    constexpr int NIT = (TT + 255) / 256, NIT2 = (TZ * 180 + 255) / 256;
    uint2 raw[NIT];
    unsigned rsc[NIT], okm = 0u;
    auto ldrows = [&](int oz0_, int p0_, int npos_, auto nit_c) {   /* the rows of tile oz0_'s new planes p0_ .. -> raw / rsc / okm */
        constexpr int NI = decltype(nit_c)::value;
        okm = 0u;
#pragma unroll
        for (int it = 0; it < NI; it++) {
            const int li = threadIdx.x + 256 * it, pl = p0_ + li / 180, rem = li % 180, row = rem / 18, ix = rem - 18 * row;
            const int gz = oz0_ - 1 + pl, gy = oy0 - 1 + row, gx = ox0 - 1 + ix;
            const bool ok = li < npos_ && c0.p && gz + zi0 >= 0 && gz + zi0 < Di && gy >= 0 && gy < H && gx >= 0 && gx < W;
            ld_row16(c0, ok ? ((size_t)(gz + zi0) * H + gy) * W + gx : 0, ok, raw[it], rsc[it]);
            okm |= (unsigned)ok << it;
        }
    };
    constexpr bool PF = MT == 1;   /* (MT 2: the rows live across the MMA spill its doubled accumulators: +12%, so loaded at the tile) */
    constexpr bool IL = MT == 1;   /* interleaved scale words (p16_put); MT 2: the extra staging index math outweighed the saved loads (+1%) */
    if constexpr (IS_MX4(T) && PF) if ((tz0 + zc * ZC) < nzt) ldrows((tz0 + zc * ZC) * TZ, 0, TT, f4p_nit<NIT>{});
    /* z tiles of TZ output planes: the block walks ZC of them; tile oz0 reads input planes oz0 - 1 .. oz0 + TZ, of which the
       first two are the previous tile's last two (kept in the ring), so after the first tile only TZ planes are staged */
    for (int tz = (tz0 + zc * ZC); tz < nzt && tz < tz0 + (zc + 1) * ZC; tz++) {
    const int oz0 = tz * TZ, p0 = tz == (tz0 + zc * ZC) ? 0 : 2, npos = (NP - p0) * 180;   /* tile planes p0 .. NP - 1 are new */
    __syncthreads();   /* the previous tile's MMA has read the slots about to be replaced */
    F4RP(0);
    if constexpr (IS_MX4(T)) {   /* staging, MX-fp4 input: every position's row and scale were loaded ahead (ldrows), each position is
                                    quantised once with its own scale and written to both pairs it belongs to (low half of pair p, high
                                    half of pair p - 1) */
        if constexpr (!PF) ldrows(oz0, p0, npos, f4p_nit<NIT>{});
#pragma unroll
        for (int it = 0; it < NIT; it++) {
            const int li = threadIdx.x + 256 * it, pl = p0 + li / 180, rem = li % 180, row = rem / 18, ix = rem - 18 * row;
            if (li >= npos) break;
            const int gz = oz0 - 1 + pl, gy = oy0 - 1 + row, gx = ox0 - 1 + ix, pos = ((gz + 1) % NP) * 180 + rem;
            const bool in = (okm >> it) & 1u;
            uint2 q = raw[it];
            unsigned sc = in ? rsc[it] : 1u;
            if (!cpy) {   /* dequantise, GN+SiLU (stage_row16's arithmetic), requantise */
                slab_ab(gz);
                float v[16];
                dec_row16_gnh(q, rsc[it], in, Ci, G, gab, v);
                float am = 0.f;
#pragma unroll
                for (int k = 0; k < 16; k++) am = amax_xs(am, v[k]);   /* (sign bit meaningless: mx_exp ignores it) */
                const int e = mx_exp(am, 1.f / 6.f);
                const float m = exp2i(-e);
                if (sp.sr) {   /* gradient operand: exact stochastic rounding keyed by the element (deterministic per call) */
                    const uint64_t vid = (((uint64_t)n * D + gz) * H + gy) * (uint64_t)W + gx;
                    uint32_t ha[4], hb[4];
                    sr_hash4(sp.sr, vid * 2, ha); sr_hash4(sp.sr, vid * 2 + 1, hb);
                    q = make_uint2(sr_e2m1_word(v, m, ha), sr_e2m1_word(v + 8, m, hb));
                } else q = make_uint2(cvt_e2m1x8(v, m), cvt_e2m1x8(v + 8, m));
                sc = (unsigned)(e + 127);
            }
            p16_put<IL>(sx, sxw, pos, ix, q, sc);
        }
    } else
    for (int li = threadIdx.x; li < npos; li += 256) {   /* staging: thread per position, quantised once with its own scale, written
                                                            to both pairs it belongs to (low half of pair p, high half of pair p - 1) */
        const int pl = p0 + li / 180, rem = li % 180, row = rem / 18, ix = rem - 18 * row;
        const int gz = oz0 - 1 + pl, gy = oy0 - 1 + row, gx = ox0 - 1 + ix, pos = ((gz + 1) % NP) * 180 + rem;
        const bool inb = gz + zi0 >= 0 && gz + zi0 < Di && gy >= 0 && gy < H && gx >= 0 && gx < W;
        const size_t off = inb ? ((size_t)(gz + zi0) * H + gy) * W + gx : 0;
        uint2 q;
        float v[16];
        slab_ab(gz);
        stage_row16<T>(c0, Ci, plane, off, inb, G, gab, v);
        unsigned am = 0u;
#pragma unroll
        for (int k = 0; k < 16; k++) am = amax_u(am, v[k]);
        const int e = mx_exp(__uint_as_float(am), 1.f / 6.f);
        const float m = exp2i(-e);
        if (sp.sr) {   /* gradient operand: exact stochastic rounding keyed by the element (deterministic per call): one hash and
                          three remixes per 8 values, the nibbles straight from the rounding */
            const uint64_t vid = (((uint64_t)n * D + gz) * H + gy) * (uint64_t)W + gx;
            uint32_t ha[4], hb[4];
            sr_hash4(sp.sr, vid * 2, ha); sr_hash4(sp.sr, vid * 2 + 1, hb);
            q = make_uint2(sr_e2m1_word(v, m, ha), sr_e2m1_word(v + 8, m, hb));
        } else q = make_uint2(cvt_e2m1x8(v, m), cvt_e2m1x8(v + 8, m));
        p16_put<IL>(sx, sxw, pos, ix, q, (unsigned)(e + 127));
    }
    __syncthreads();
    if constexpr (IS_MX4(T) && PF) if (tz + 1 < nzt && tz + 1 < tz0 + (zc + 1) * ZC) ldrows(oz0 + TZ, 2, TZ * 180, f4p_nit<NIT2>{});   /* the next tile's rows */
    F4RP(1);
    float acc[MT][NR][2][4];
#pragma unroll
    for (int m = 0; m < MT; m++) for (int r = 0; r < NR; r++) for (int q = 0; q < 2; q++) for (int k = 0; k < 4; k++) acc[m][r][q][k] = 0.f;
    const int mat = lane >> 3, kb = mat & 1, vx = (mat >> 1) * 8 + (lane & 7);
    int sl[RZ];   /* ring slots of the warp's output planes (kz = 0): one % NP per tile, the taps' slots by a wrap-around add */
#pragma unroll
    for (int rz = 0; rz < RZ; rz++) sl[rz] = (oz0 + wz * RZ + rz) % NP;
    constexpr int UNR = F4P_UNR9 >= MT ? 9 : 3;   /* the tap-row loop fully unrolled: kz constant */
#pragma unroll (UNR)
    for (int tr = 0; tr < 9; tr++) {
        const int kz = tr / 3, ky = tr % 3;
        mx4b4 bf[NR][2];
#pragma unroll
        for (int r = 0; r < NR; r++) {
            const int s0 = sl[r >> 1] + kz, slot = s0 >= NP ? s0 - NP : s0;
            const int rowi = slot * 10 + wr + (r & 1) + ky;   /* the tile plane's ring slot */
            unsigned bfr[4];
            ldsm_x4(bfr, sx + (rowi * 18 + vx + 2 * kb) * 16);   /* K 0..31: [x | x+1], K 32..63: [x+2 | x+3] */
            if constexpr (IL) {   /* columns c, 8 + c: their three positions' scales, kx3 = 1 (interleaved: [2c] = column c, [2c + 1] = 8 + c) */
                const uint2 s0 = *(const uint2 *)(sxw + rowi * 18 + 2 * LPB0), s1 = *(const uint2 *)(sxw + rowi * 18 + 2 * LPB1);
                bf[r][0] = mx4b4_make(bfr, s0.x, s1.x); bf[r][1] = mx4b4_make(bfr + 2, s0.y, s1.y);
            } else {
#pragma unroll
                for (int q2 = 0; q2 < 2; q2++) bf[r][q2] = mx4b4_make(bfr + 2 * q2, sxw[rowi * 18 + q2 * 8 + LPB0], sxw[rowi * 18 + q2 * 8 + LPB1]);
            }
        }
#pragma unroll
        for (int m = 0; m < MT; m++) {
            unsigned af[4];
            const int arow = m * 16 + (mat & 1) * 8 + (lane & 7);
            ldsm_x4(af, wa + tr * BM * 32 + sw16(arow, mat >> 1));
            const mx4a4 A = mx4a4_make(af, was[tr * BM + m * 16 + LPA0], was[tr * BM + m * 16 + LPA1]);
#pragma unroll
            for (int r = 0; r < NR; r++) { mma4p4(acc[m][r][0], A, bf[r][0]); mma4p4(acc[m][r][1], A, bf[r][1]); }
        }
    }
#if F4R_PROF
    __syncthreads();
#endif
    F4RP(3);
    fwd_epilogue<MT, NR, TO, true, MT <= 2, true>(acc, scr, y, b, n, co0, Co, D, H, W, oz0, oy0, ox0, wz, wr, osum, Go, sp, N, f4p_scr_off<MT, TZ>());
    F4RP(5);
    }
    if (osum) fwd_stats_flush<BM>(scr, osum, n, co0, Co, Go);
#if F4R_PROF
    if (threadIdx.x == 0) atomicAdd(&g_f4r_prof[7], 1ull);
#endif
}
template <int MT, int TZ, typename T, typename TO> void launch_f4p(dim3, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp) {
    constexpr int BM = MT * 16, RZ = TZ / 2, NR = 2 * RZ;
    constexpr int RBM = IS_MX(TO) ? 32 * MX_BITS(TO) / 8 : 0;   /* the epilogue's MX smem store path: 512 + 8 warps x NR x 16 x (RBM + 1) */
    size_t scr = (size_t)8 * 256 * sizeof(float);
    if (IS_MX(TO) && scr < (size_t)512 + 8 * NR * 16 * (RBM + 1)) scr = (size_t)512 + 8 * NR * 16 * (RBM + 1);
    if (IS_MX(TO) && MT <= 2) scr = fepi16_scr(f4p_scr_off<MT, TZ>(), scr, cout, sp);   /* the 16-channel-block MX epilogue (fwd_epi_mx16) */
    const size_t smem = f4p_scr_off<MT, TZ>() + scr;
    (void)BM;
    /* z tiles per block: the ring saves 2 of TZ + 2 staged planes per tile after the first; keep >= 4 blocks per SM-pair of waves */
    const int nzt = zw_tiles(xs.d, TZ, sp), nmt = Cop / BM;
    const size_t base = (size_t)nblk_(xs.w, 16) * nblk_(xs.h, 8) * nmt * xs.n;
    static int zc_env = -2;
    if (zc_env == -2) zc_env = getenv("UFSM_F4P_ZC") ? atoi(getenv("UFSM_F4P_ZC")) : -1;
    /* the longest z chunks (64, 32, 16 tiles) that keep >= 2048 blocks: a block's first tile stages 2 extra planes and loads the
       weights (desk P512: ZC 8 -> 64 fwd 19.1 -> 18.0 ms, bdata 17.0 -> 16.4; P256 is best at 16); else the old rule. (The output
       GN statistics are summed per block in fp32: their grouping, not the outputs, changes with ZC) */
    int ZC = 0;
    if (zc_env > 0) ZC = zc_env;
    else for (int z = 64; z > 8; z >>= 1) if (z <= nzt && base * nblk_(nzt, z) >= 2048) { ZC = z; break; }
    if (!ZC) ZC = nzt < 8 ? nzt : 8;
    while (ZC > 1 && base * nblk_(nzt, ZC) < 480) ZC--;
    const dim3 grid(nblk_(xs.w, 16), nblk_(xs.h, 8), (unsigned)(nblk_(nzt, ZC) * nmt * xs.n));   /* (the caller's grid argument is unused) */
    static int attr[8];
    if (MT == 1 && gp.ns && gp.G) {   /* slab-local GroupNorm statistics: the instance with the per-slab reload */
        static int attr1[8];
        if (!attr1[cur_dev_()]) { attr1[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)conv_fwd_f4p_k<MT, TZ, T, TO, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
        conv_fwd_f4p_k<MT, TZ, T, TO, true><<<grid, 256, smem>>>((const T *)x, wq, ws, b, (TO *)y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, gp, osum, Go, sp, ZC);
    } else {
        if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)conv_fwd_f4p_k<MT, TZ, T, TO>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
        conv_fwd_f4p_k<MT, TZ, T, TO><<<grid, 256, smem>>>((const T *)x, wq, ws, b, (TO *)y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, gp, osum, Go, sp, ZC);
    }
#if F4R_PROF
    if (getenv("F4R_PROF")) {
        unsigned long long h[8]; cudaDeviceSynchronize(); cudaMemcpyFromSymbol(h, g_f4r_prof, sizeof h);
        const double nb = (double)h[7], tot = (double)(h[0] + h[1] + h[2] + h[3] + h[4] + h[5]);
        fprintf(stderr, "f4p %d->%d @%dx%dx%d MT %d TZ %d ZC %d: per block %.0f cycles: tile start %.0f%% staging %.0f%% mma %.0f%% epilogue %.0f%%\n",
                xs.c, cout, xs.d, xs.h, xs.w, MT, TZ, ZC, tot / nb, 100 * h[0] / tot, 100 * h[1] / tot, 100 * h[3] / tot, 100 * h[5] / tot);
        unsigned long long z[8] = {}; cudaMemcpyToSymbol(g_f4r_prof, z, sizeof z);
    }
#endif
}
static int wmemo_get(unsigned key, const float *w, int Cop, int Cip, int Cx, int CxP, int Ox, int OxP, size_t nq, size_t ns, uint8_t **wq, uint8_t **ws, int slot = 2);
template <typename T, typename TO> static void p16_f4(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, int Cop, int Ox, int OxP, gnp_t gp, double *osum, int Go, split_t sp) {
    const size_t nq = (size_t)9 * Cop * 32, ns = (size_t)9 * Cop * 4;
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
/* the 16-wide decoder conv1 with the fused upsample, MX-fp4 in / out (lp_f4up.cu): 0 = not this shape (the caller's path) */
int lp_f4u_fwd(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, const split_t &sp);
template <typename T, typename TO> void fwd_f4_t(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp) {
    static int uph2 = -1; if (uph2 < 0) uph2 = getenv("UFSM_UP_H2") ? atoi(getenv("UFSM_UP_H2")) : 1;
    if (sp.up && uph2) sp.up = 2;   /* the fused upsample blends in half2 (stage_up_smem_h2); UFSM_UP_H2=0: fp32 */
    int Cop = (cout + 15) / 16 * 16, Cip = (xs.c + 31) / 32 * 32;
    int Cx = sp.x2 ? sp.c_split : xs.c, CxP = (Cx + 31) / 32 * 32;
    int Ox = sp.y2 ? sp.o_split : cout, OxP = (Ox + 31) / 32 * 32;
    if (IS_MX(TO)) { if (sp.y2) Cop = OxP + (cout - Ox + 31) / 32 * 32; else if (cout > 16) Cop = (cout + 31) / 32 * 32; }   /* 32-row-aligned output blocks */
    if (IS_MX(T)) Cip = CxP + (xs.c - Cx + 31) / 32 * 32;                                                                /* per-tensor 32-channel segments */
    /* MX-fp4 gradient backward-data (fp4 gy, output SR keyed by osr, no staging SR): <= 16-channel gy rows go to the packed kernel,
       which copies each row with its own scale (one scale per 16 K: exact, no re-rounding) */
    const bool g4_exact = IS_MX4(T) && sp.osr && !sp.sr;
    /* a split MX-fp4 gradient output of two <= 16-channel tensors (dec0.c1's backward-data of a 16-wide level 0: up part | skip):
       the second tensor at row 16, not 32 -- Cop 32 instead of 64 (tile MT 2 / TZ 4 with resident weights, not MT 4 / TZ 2 with half
       the rows padding); the output SR keys keep the 32-padded row numbers, so the bits are unchanged. UFSM_F4_OPAD16=0: off */
    static int opad_env = -1; if (opad_env < 0) opad_env = getenv("UFSM_F4_OPAD16") ? atoi(getenv("UFSM_F4_OPAD16")) : 1;
    if (opad_env && g4_exact && IS_MX(TO) && sp.y2 && Ox <= 16 && cout - Ox <= 16) { OxP = 16; Cop = 32; sp.o_pad = 16; }
    /* (slab-local GroupNorm input, gnp_t ns: the chunk kernels below, which apply each staged plane's slab; conv_fwd_f4p_k does not yet) */
    if (xs.c <= 16 && !sp.x2 && !sp.up && sp.gp2.G == 0 && !(gp.ns && gp.G && Cop > 16)) { p16_f4<T, TO>(x, xs, w, b, cout, y, Cop, Ox, OxP, gp, osum, Go, sp); return; }   /* K = [kx0 | kx1 | kx2 | kx3] x 16, a scale per kx */
    if constexpr (IS_MX4(T) && IS_MX4(TO)) {   /* the 16-wide decoder conv1 with the fused upsample (lp_f4up.cu); UFSM_F4U=0: the chunk kernel */
        static int f4u = -1; if (f4u < 0) f4u = getenv("UFSM_F4U") ? atoi(getenv("UFSM_F4U")) : 1;
        if (f4u && sp.up == 2 && !sp.zw1 && lp_f4u_fwd(x, xs, w, b, cout, y, gp, osum, Go, sp)) return;
    }
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
    if constexpr (IS_MX(T)) {   /* dec0.c1 of a 32-wide level 0 (32-channel up part | skip): the ring kernel with the up chunk staged per
                                   new plane (f4ru_stage; same bits as conv_fwd_f4_k). UFSM_F4RU=0: conv_fwd_f4_k */
        static int ru_env = -1; if (ru_env < 0) ru_env = getenv("UFSM_F4RU") ? atoi(getenv("UFSM_F4RU")) : 1;
        if (ru_env && ring_env && sp.up == 2 && sp.x2 && !sp.sr && Cip == 64 && CxP == 32 && MT == 2 && TZ == 4 && !tz_env && !(gp.ns || sp.gp2.ns) && xs.c > Cx) {
            launch_f4r<2, 4, 2, T, TO, 1>(x, xs, wq, ws, b, cout, y, Cop, gp, osum, Go, sp);
            return;
        }
    }
    dim3 grid(nblk_(xs.w, 16), nblk_(xs.h, 8), (unsigned)(zw_tiles(xs.d, TZ, sp) * nmt * xs.n));
    switch (MT * 10 + TZ) {
    case 12: launch_f4<1, 2, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    case 14: launch_f4<1, 4, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    case 22: launch_f4<2, 2, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    case 24: launch_f4<2, 4, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    default: launch_f4<4, 2, T, TO>(grid, x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp); break;
    }
}
/* xbf / ybf: 0 fp32, 1 bf16, 2 fp16, 3 MX-fp8, 4 MX-fp4; instantiated: equal types, and mx8 in -> fp16 / bf16 out (backward-data) */
