/* FP4 decoder conv1 with the fused 2x upsample for a 16-wide level 0 (dec0.c1: up part <= 16 channels at half resolution,
   skip <= 16 channels, <= 16 outputs; MX-fp4 in / out). */
#include "lp_f4fwd.cuh"

/* Both segments share one 32-K block per staged position, [up 16 | skip 16] with one ue8m0 scale per 16-K group (mma_f4x4), so a
   tap pair is one m16n8k64 MMA: K = [tapA up | tapA skip | tapB up | tapB skip] (14 k-steps per n-tile; conv_fwd_f4_k ran 28
   over two 32-channel chunks, each half zero padding, and restaged the z halo and the weights per tile). The block walks ZC z
   tiles with the staged planes in a ring (TZ + 2 slots, plane gz in slot (gz + 1) % NP: after the first tile only TZ new
   planes), the weights resident. Per tile, the coarse rows its new fine planes interpolate from are decoded, put through the
   coarse tensor's GN+SiLU and stored as 16 e4m3 + one scale (stage_up_coarse16); each fine position blends 8 of them in half2
   (stage_up_smem16_h2). The staged values, their scales and the weights' nibbles and scales are those of conv_fwd_f4_k (its
   padding channels are zero: the same amax); only the fp32 accumulation order inside the tensor core differs. */
template <int NCZ>   /* CZ <= NCZ coarse planes; blockDim 256 */
__device__ __forceinline__ void stage_up_coarse16(const chan_t &c0, int Cu, const float2 *gab, bool G, uint8_t *cu, uint8_t *cus, int cz0, int cy0, int cx0, int CZ,
                                                  int D, int H, int W) {
    constexpr int NIT = (NCZ * 96 + 255) / 256;
    const int Dc = D >> 1, Hc = H >> 1, Wc = W >> 1, n = CZ * 96;
    uint2 raw[NIT];
    unsigned rsc[NIT], okm = 0u;
#pragma unroll
    for (int it = 0; it < NIT; it++) {   /* every row's load first (NIT in flight per thread), then the decode */
        const int i = threadIdx.x + 256 * it, lx = i % 12, ly = (i / 12) % 8, lz = i / 96, cz = cz0 + lz, cy = cy0 + ly, cx = cx0 + lx;
        const bool ok = i < n && c0.p && cz >= 0 && cz < Dc && cy >= 0 && cy < Hc && cx >= 0 && cx < Wc;
        ld_row16(c0, ok ? ((size_t)cz * Hc + cy) * Wc + cx : 0, ok, raw[it], rsc[it]);
        okm |= (unsigned)ok << it;
    }
#pragma unroll
    for (int it = 0; it < NIT; it++) {
        const int i = threadIdx.x + 256 * it;
        if (i >= n) break;
        float r[16];
        dec_row16_gn(raw[it], rsc[it], (okm >> it) & 1u, Cu, G, gab, r);
        unsigned am = 0u;
#pragma unroll
        for (int k = 0; k < 16; k++) { const unsigned u = __float_as_uint(fabsf(r[k])); am = u <= 0x7f800000u ? max(am, u) : am; }
        const int e = mx_exp(__uint_as_float(am), 1.f / 448.f);
        const float m = exp2i(-e);
        *(uint4 *)(cu + (size_t)i * 16) = make_uint4(cvt_e4m3x4(r[0] * m, r[1] * m, r[2] * m, r[3] * m), cvt_e4m3x4(r[4] * m, r[5] * m, r[6] * m, r[7] * m),
                                                     cvt_e4m3x4(r[8] * m, r[9] * m, r[10] * m, r[11] * m), cvt_e4m3x4(r[12] * m, r[13] * m, r[14] * m, r[15] * m));
        cus[i] = (uint8_t)(e + 127);
    }
}
/* stage_up_smem_h2 on 16-channel coarse rows (the same blend: e4m3 pairs straight to half2, 8 taps with HFMA2) */
__device__ __forceinline__ void stage_up_smem16_h2(const uint8_t *cu, const uint8_t *cus, int cz0, int cy0, int cx0, int gz, int gy, int gx, int D, int H, int W, bool inb, float *v) {
#pragma unroll
    for (int k = 0; k < 16; k++) v[k] = 0.f;
    if (!inb) return;
    const int Dc = D >> 1, Hc = H >> 1, Wc = W >> 1;
    const int mz[2] = {gz >> 1, min(max((gz >> 1) + ((gz & 1) ? 1 : -1), 0), Dc - 1)}, my[2] = {gy >> 1, min(max((gy >> 1) + ((gy & 1) ? 1 : -1), 0), Hc - 1)},
              mx[2] = {gx >> 1, min(max((gx >> 1) + ((gx & 1) ? 1 : -1), 0), Wc - 1)};
    __half2 acc[8];
#pragma unroll
    for (int j = 0; j < 8; j++) acc[j] = __float2half2_rn(0.f);
#pragma unroll
    for (int a = 0; a < 2; a++)
#pragma unroll
        for (int b = 0; b < 2; b++)
#pragma unroll
            for (int c = 0; c < 2; c++) {
                const int li = ((mz[a] - cz0) * 8 + (my[b] - cy0)) * 12 + (mx[c] - cx0);
                const __half2 w3 = __float2half2_rn((a ? 0.25f : 0.75f) * (b ? 0.25f : 0.75f) * (c ? 0.25f : 0.75f) * mx_scale(cus[li]));
                const uint4 h0 = *(const uint4 *)(cu + (size_t)li * 16);
                const unsigned w[4] = {h0.x, h0.y, h0.z, h0.w};
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    unsigned lo, hi;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(lo) : "h"((unsigned short)(w[j] & 0xffff)));
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(hi) : "h"((unsigned short)(w[j] >> 16)));
                    acc[2 * j] = __hfma2(w3, *(const __half2 *)&lo, acc[2 * j]); acc[2 * j + 1] = __hfma2(w3, *(const __half2 *)&hi, acc[2 * j + 1]);
                }
            }
#pragma unroll
    for (int j = 0; j < 8; j++) { const float2 f = __half22float2(acc[j]); v[2 * j] = f.x; v[2 * j + 1] = f.y; }
}
/* weights: wq[tap 28][16 co][16 B] (channels [up 0..15 | skip 0..15]), ws[tap][co][2] (one scale per 16-channel group) */
__global__ void prep_w4u_k(const float *__restrict__ w, uint8_t *__restrict__ wq, uint8_t *__restrict__ ws, int Co, int Ci, int Cx) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;   /* (tap, co, group) */
    if (i >= (size_t)28 * 16 * 2) return;
    const int grp = (int)(i & 1), co = (int)((i >> 1) % 16), t = (int)((i >> 1) / 16);
    float v[16], amax = 0.f;
#pragma unroll
    for (int k = 0; k < 16; k++) {
        const int ci = grp ? (k < Ci - Cx ? Cx + k : -1) : (k < Cx ? k : -1);
        v[k] = t < 27 && co < Co && ci >= 0 ? w[((size_t)co * Ci + ci) * 27 + t] : 0.f;
        amax = fmaxf(amax, fabsf(v[k]));
    }
    const int e = mx_exp(amax, 1.f / 6.f);
    const float m = exp2i(-e);
    *(uint2 *)(wq + ((size_t)t * 16 + co) * 16 + grp * 8) = make_uint2(cvt_e2m1x8(v, m), cvt_e2m1x8(v + 8, m));
    ws[i] = (uint8_t)(e + 127);
}
template <int TZ>
__host__ __device__ constexpr unsigned f4u_off(int part) {   /* smem: 0 ring, 1 scales, 2 weights, 3 weight scales, 4 tables, 5 coarse rows / epilogue scratch */
    constexpr unsigned TT = (TZ + 2) * 180, o1 = TT * 16, o2 = (o1 + TT * 4 + 127) & ~127u, o3 = o2 + 14 * 16 * 32, o4 = o3 + 14 * 16 * 4,
                       o5 = (o4 + 32 * 32 + 32 * 8 + 127) & ~127u;
    return part == 0 ? 0u : part == 1 ? o1 : part == 2 ? o2 : part == 3 ? o3 : part == 4 ? o4 : o5;
}
template <int TZ>
__host__ __device__ constexpr unsigned f4u_coarse_bytes() { return (unsigned)(TZ / 2 + 4) * 96 * 17; }   /* rows + scales */
template <int TZ, typename T, typename TO>
__global__ void __launch_bounds__(256, 2) conv_fwd_f4u_k(const T *__restrict__ x, const uint8_t *__restrict__ wq, const uint8_t *__restrict__ wsc,
                                                      const float *__restrict__ b, TO *__restrict__ y, int N, int Ci, int Cx, int D, int H, int W, int Co,
                                                      gnp_t gp, double *__restrict__ osum, int Go, split_t sp, int ZC) {
    constexpr int RZ = TZ / 2, NR = 2 * RZ, NP = TZ + 2, CZ = TZ / 2 + 4;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    uint8_t *sx = smem_raw;                                                   /* [NP slots x 180 pos][16 B] = [up 8 B | skip 8 B] */
    unsigned *sxw = (unsigned *)(smem_raw + f4u_off<TZ>(1));                  /* [pos] scale bytes (up, skip) of pos | of pos + 1 << 16 */
    uint8_t *wa = smem_raw + f4u_off<TZ>(2);                                  /* [14 pairs][16 co][32 B] swizzled (tapA 16 B | tapB 16 B) */
    unsigned *was = (unsigned *)(smem_raw + f4u_off<TZ>(3));                  /* [14][16]: bytes (tapA up, tapA skip, tapB up, tapB skip) */
    chan_t *ctab = (chan_t *)(smem_raw + f4u_off<TZ>(4));                     /* [32]: up channels 0..15, skip 16..31 */
    float2 *cab = (float2 *)(ctab + 32);                                      /* [32] GN coefficients */
    unsigned char *scr = smem_raw + f4u_off<TZ>(5);                           /* coarse rows during staging, the epilogue scratch after the MMA */
    uint8_t *cu = scr, *cus = scr + CZ * 96 * 16;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wz = warp >> 2, wr = (warp & 3) * 2;
    const int ox0 = blockIdx.x * 16, oy0 = blockIdx.y * 8;
    const int nzt = (D + TZ - 1) / TZ, nzc = (nzt + ZC - 1) / ZC;
    const int zc = blockIdx.z % nzc, n = blockIdx.z / nzc;
    const int zi0 = sp.zi0, Di = sp.Di ? sp.Di : D;   /* z-range (nn_set_zrange): output plane z reads input plane z + zi0 of a depth-Di input */
    const size_t plane = (size_t)Di * H * W, cplane = (size_t)(Di >> 1) * (H >> 1) * (W >> 1);
    const int Cs = Ci - Cx;
    for (int i = threadIdx.x; i < 14 * 16 * 2; i += 256) {
        const int pp = i >> 5, r = i & 31, c = r >> 1, h = r & 1, tap = 2 * pp + h;
        *(uint4 *)(wa + pp * 16 * 32 + sw16(c, h)) = __ldg((const uint4 *)(wq + ((size_t)tap * 16 + c) * 16));
    }
    for (int i = threadIdx.x; i < 14 * 16; i += 256) {
        const int pp = i >> 4, c = i & 15;
        was[i] = (unsigned)__ldg((const unsigned short *)wsc + (size_t)2 * pp * 16 + c) | ((unsigned)__ldg((const unsigned short *)wsc + (size_t)(2 * pp + 1) * 16 + c) << 16);
    }
    if (threadIdx.x < 32) {   /* padded channel cp: up cp (coarse planes), skip cp - 16 */
        const int cp = threadIdx.x, ci = cp < 16 ? (cp < Cx ? cp : -1) : (cp - 16 < Cs ? Cx + cp - 16 : -1);
        const chan_t c = make_chan(ci, Ci, Cx, n, cp < 16 ? cplane : plane, x, sp, gp, N);
        ctab[cp] = c; cab[cp] = make_float2(c.a, c.b);
    }
    __syncthreads();
    const bool Gu = ctab[0].g != 0, Gs = ctab[16].g != 0;   /* (per-segment GroupNorms, or one over both: make_chan's choice) */
    const chan_t cs0 = ctab[16];
#if F4R_PROF
    long long t_prev = clock64();
#endif
    for (int tz = zc * ZC; tz < nzt && tz < (zc + 1) * ZC; tz++) {
        const int oz0 = tz * TZ, p0n = tz == zc * ZC ? 0 : 2, npos = (NP - p0n) * 180;   /* tile planes p0n .. NP - 1 are new */
        const int glo = oz0 - 1 + p0n + zi0, ghi = oz0 + TZ + zi0, cz0 = (glo >> 1) - 1, cy0 = (oy0 >> 1) - 2, cx0 = (ox0 >> 1) - 2;
        __syncthreads();   /* the previous tile's MMA and epilogue are done with the ring slots and the scratch */
        F4RP(4);
        stage_up_coarse16<CZ>(ctab[0], Cx, cab, Gu, cu, cus, cz0, cy0, cx0, min(CZ, ((ghi >> 1) + 1) - cz0 + 1), Di, H, W);
        __syncthreads();
        F4RP(0);
        constexpr int NIT = (NP * 180 + 255) / 256;
        float2 gs[16];   /* the skip's GN coefficients in registers for the staging (were two shared loads per element) */
#pragma unroll
        for (int k = 0; k < 16; k++) gs[k] = cab[16 + k];
        uint2 raw[NIT];
        unsigned rsc[NIT], okm = 0u;   /* the skip rows, all loads first (NIT in flight per thread) */
#pragma unroll
        for (int it = 0; it < NIT; it++) {
            const int li = threadIdx.x + 256 * it, pl = p0n + li / 180, rem = li % 180, ix = rem % 18, iy = rem / 18;
            const int gz = oz0 - 1 + pl, gy = oy0 - 1 + iy, gx = ox0 - 1 + ix;
            const bool ok = li < npos && cs0.p && gz + zi0 >= 0 && gz + zi0 < Di && gy >= 0 && gy < H && gx >= 0 && gx < W;
            ld_row16(cs0, ok ? ((size_t)(gz + zi0) * H + gy) * W + gx : 0, ok, raw[it], rsc[it]);
            okm |= (unsigned)ok << it;
        }
#pragma unroll
        for (int it = 0; it < NIT; it++) {   /* stage: up part interpolated, skip dequantised + GN+SiLU, each requantised */
            const int li = threadIdx.x + 256 * it, pl = p0n + li / 180, rem = li % 180, ix = rem % 18, iy = rem / 18;
            if (li >= npos) break;
            const int gz = oz0 - 1 + pl, gy = oy0 - 1 + iy, gx = ox0 - 1 + ix, pos = ((gz + 1) % NP) * 180 + rem;
            const bool inb = gz + zi0 >= 0 && gz + zi0 < Di && gy >= 0 && gy < H && gx >= 0 && gx < W;
            float v[16];
            unsigned am, q[4], sc;
            stage_up_smem16_h2(cu, cus, cz0, cy0, cx0, gz + zi0, gy, gx, Di, H, W, inb, v);
            am = 0u;
#pragma unroll
            for (int k = 0; k < 16; k++) am = amax_u(am, v[k]);
            int e = mx_exp(__uint_as_float(am), 1.f / 6.f);
            float m = exp2i(-e);
            q[0] = cvt_e2m1x8(v, m); q[1] = cvt_e2m1x8(v + 8, m); sc = (unsigned)(e + 127);
            dec_row16_gn(raw[it], rsc[it], (okm >> it) & 1u, Cs, Gs, gs, v);
            am = 0u;
#pragma unroll
            for (int k = 0; k < 16; k++) am = amax_u(am, v[k]);
            e = mx_exp(__uint_as_float(am), 1.f / 6.f);
            m = exp2i(-e);
            q[2] = cvt_e2m1x8(v, m); q[3] = cvt_e2m1x8(v + 8, m); sc |= (unsigned)(e + 127) << 8;
            *(uint4 *)(sx + pos * 16) = make_uint4(q[0], q[1], q[2], q[3]);
            ((unsigned short *)sxw)[2 * pos] = (unsigned short)sc;                /* low half of word[pos] */
            if (ix) ((unsigned short *)sxw)[2 * pos - 1] = (unsigned short)sc;   /* high half of word[pos - 1] */
        }
        __syncthreads();
        F4RP(1);
        float acc[1][NR][2][4];
#pragma unroll
        for (int r = 0; r < NR; r++) for (int q = 0; q < 2; q++) for (int k = 0; k < 4; k++) acc[0][r][q][k] = 0.f;
        int slot[RZ + 2];   /* the warp's first row in the ring slot of tile plane wz RZ + j (j = r / 2 + kz) */
#pragma unroll
        for (int j = 0; j < RZ + 2; j++) slot[j] = ((oz0 + wz * RZ + j) % NP) * 180 + wr * 18;
#pragma unroll
        for (int pp = 0; pp < 14; pp++) {
            const int tA = 2 * pp, tB = min(tA + 1, 26);
            unsigned bfr[NR][4], sb[NR][2];
#pragma unroll
            for (int r = 0; r < NR; r++) {
                const int rA = slot[(r >> 1) + tA / 9] + ((r & 1) + (tA / 3) % 3) * 18 + tA % 3, rB = slot[(r >> 1) + tB / 9] + ((r & 1) + (tB / 3) % 3) * 18 + tB % 3;
                const int mat = lane >> 3, q = mat >> 1, isB = mat & 1;
                ldsm_x4(bfr[r], sx + ((isB ? rB : rA) + q * 8 + (lane & 7)) * 16);
#pragma unroll
                for (int q2 = 0; q2 < 2; q2++) {   /* (up, skip) of tap A | of tap B << 16: one word when B is A's x neighbour */
                    if (tB == tA + 1 && tA % 3 != 2) sb[r][q2] = sxw[rA + q2 * 8 + g];
                    else sb[r][q2] = (sxw[rA + q2 * 8 + g] & 0xffffu) | (sxw[rB + q2 * 8 + g] << 16);
                }
            }
            unsigned af[4];
            const int mat = lane >> 3, row = (mat & 1) * 8 + (lane & 7);
            ldsm_x4(af, wa + pp * 16 * 32 + sw16(row, mat >> 1));
            const unsigned sa = was[pp * 16 + g + 8 * (t & 1)];
#pragma unroll
            for (int r = 0; r < NR; r++) { mma_f4x4(acc[0][r][0], af, bfr[r], sa, sb[r][0]); mma_f4x4(acc[0][r][1], af, bfr[r] + 2, sa, sb[r][1]); }
        }
#if F4R_PROF
        __syncthreads();
#endif
        F4RP(3);
        fwd_epilogue<1, NR, TO, true>(acc, scr, y, b, n, 0, Co, D, H, W, oz0, oy0, ox0, wz, wr, osum, Go, sp, N, f4u_off<TZ>(5));
        F4RP(5);
    }
#if F4R_PROF
    if (threadIdx.x == 0) atomicAdd(&g_f4r_prof[7], 1ull);
#endif
}
int lp_f4u_fwd(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, const split_t &sp) {
    constexpr int TZ = 4;
    const int Cx = sp.c_split;
    if (!sp.x2 || Cx > 16 || xs.c - Cx > 16 || cout > 16 || sp.y2 || sp.sr || sp.osr || sp.accum || (xs.d | xs.h | xs.w) & 1) return 0;
    uint8_t *wq, *ws;
    if (!wmemo_get(sp.wkey, w, 16, -32, Cx, 16, -1, 0, (size_t)28 * 16 * 16, (size_t)28 * 16 * 2, &wq, &ws, 2)) prep_w4u_k<<<nblk_(28 * 16 * 2, 128), 128>>>(w, wq, ws, cout, xs.c, Cx);
    size_t scr = f4u_coarse_bytes<TZ>();
    if (scr < (size_t)8 * 256 * sizeof(float)) scr = (size_t)8 * 256 * sizeof(float);
    if (scr < FEPI16_SCR(16)) scr = FEPI16_SCR(16);
    const size_t smem = f4u_off<TZ>(5) + scr;
    const int nzt = nblk_(xs.d, TZ);
    const size_t base = (size_t)nblk_(xs.w, 16) * nblk_(xs.h, 8) * xs.n;
    static int zc_env = -2;
    if (zc_env == -2) zc_env = getenv("UFSM_F4U_ZC") ? atoi(getenv("UFSM_F4U_ZC")) : -1;
    int ZC = zc_env > 0 ? zc_env : nzt < 8 ? nzt : 8;
    while (ZC > 1 && base * nblk_(nzt, ZC) < 480) ZC--;
    const dim3 grid(nblk_(xs.w, 16), nblk_(xs.h, 8), (unsigned)(nblk_(nzt, ZC) * xs.n));
    static int attr[8];
    if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)conv_fwd_f4u_k<TZ, mx4_t, mx4_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
    conv_fwd_f4u_k<TZ, mx4_t, mx4_t><<<grid, 256, smem>>>((const mx4_t *)x, wq, ws, b, (mx4_t *)y, xs.n, xs.c, Cx, xs.d, xs.h, xs.w, cout, gp, osum, Go, sp, ZC);
#if F4R_PROF
    if (getenv("F4R_PROF")) {
        unsigned long long h[8]; cudaDeviceSynchronize(); cudaMemcpyFromSymbol(h, g_f4r_prof, sizeof h);
        const double nb = (double)h[7], tot = (double)(h[0] + h[1] + h[3] + h[4] + h[5]);
        fprintf(stderr, "f4u %d + %d -> %d @%dx%dx%d ZC %d: per block %.0f cycles: tile start %.0f%% coarse %.0f%% fine staging %.0f%% mma %.0f%% epilogue %.0f%%\n",
                Cx, xs.c - Cx, cout, xs.d, xs.h, xs.w, ZC, tot / nb, 100 * h[4] / tot, 100 * h[0] / tot, 100 * h[1] / tot, 100 * h[3] / tot, 100 * h[5] / tot);
        unsigned long long z[8] = {}; cudaMemcpyToSymbol(g_f4r_prof, z, sizeof z);
    }
#endif
    return 1;
}
