/* 2:4 sparse MX-fp4 3^3 stride-1 convolution (forward, and backward-data through flipped weights): conv_fwd_f4_k (lp_f4fwd.cuh)
   with the dense tap-pair MMA (m16n8k64, two taps x 32 input channels) replaced by the sparse tap-quad MMA
   (mma.sp m16n8k128 kind::mxf4nvf4 scale_vec::4X: four taps x 32 input channels, A stored compressed). The staging and the
   epilogue are conv_fwd_f4_k's code unchanged (the same bits); per 32-channel chunk 7 quads (taps 24..26 + a zero tap) replace
   14 pairs, so the MMAs and the weight ldmatrix halve while the B ldmatrix and the staging stay.
   Sparsity pattern (the hardware's, tests/test_sp24_mma.c): an e2m1 byte (two adjacent input channels) is the sparse element, so per
   (output channel, tap, 8 input channels) at most 2 of the 4 channel pairs are non-zero ("pair-wise 4:8"). The weights arrive
   already masked (nn_mask24 mode 2 on the forward copy); the prep keeps the 2 largest pairs of each group and counts groups with
   more non-zero pairs (lp_sp24_err: such weights would be silently truncated).
   DESIGN.md 2026-11-13 "2:4 sparse fp4 convs". */
#include "lp_f4fwd.cuh"
#include "lp_sp24.h"

#define SP_NQ 7   /* tap quads: 0..3, ..., 24..26 + zero tap 27 */
__device__ unsigned g_sp24_err;   /* groups with > 2 non-zero pairs seen by the prep (truncated) */

#ifndef LP_SW
__device__ __forceinline__ void sp_mma4(float *c, const unsigned *a, const unsigned *b, unsigned e, unsigned sa, unsigned sb) {
    asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0, %13, {0, 0}, %14, {0, 0};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
                   "r"(e), "r"(sa), "r"(sb));
}
#endif
/* metadata lane layout (tests/test_sp24_mma.c: candidate 1 matches on sm_120a): lane (g, t) holds row g + 8 (t & 1), groups 8 (t >> 1) .. + 7 (nibble q = group) */
#define SP_MROW(t) (((t) & 1) * 8)
#define SP_MHALF(t) ((t) >> 1)

/* prep: thread per (quad, padded output row, 32-channel chunk). Per tap: the dense prep_w4_k scale and nibbles (the same bits), then
   per 4-byte group the 2 bytes of largest |w0| + |w1| (ties: lower index), in index order */
__global__ void prep_sp24_k(const float *__restrict__ w, uint8_t *__restrict__ wq, unsigned *__restrict__ ws, uint2 *__restrict__ wm,
                            int Co, int Ci, int Cop, int Cip, int Cx, int CxP, int Ox, int OxP) {
    const int nch = Cip / 32;
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)SP_NQ * Cop * nch) return;
    const int ch = (int)(i % nch), cop = (int)((i / nch) % Cop), qd = (int)(i / ((size_t)nch * Cop));
    const int co = Ox < 0 ? cop : seg_ci(cop, Co, Ox, OxP);
    unsigned comp[8], meta[2] = {0u, 0u}, sc = 0u, bad = 0u;
    for (int j = 0; j < 4; j++) {
        const int t = 4 * qd + j;
        float v[32], amax = 0.f;
#pragma unroll
        for (int k = 0; k < 32; k++) {
            const int ci = Cx < 0 ? ch * 32 + k : seg_ci(ch * 32 + k, Ci, Cx, CxP);
            v[k] = (t < 27 && co >= 0 && co < Co && ci >= 0 && ci < Ci) ? w[((size_t)co * Ci + ci) * 27 + t] : 0.f;
            amax = fmaxf(amax, fabsf(v[k]));
        }
        const int e = mx_exp(amax, 1.f / 6.f);
        const float m = exp2i(-e);
        const unsigned q[4] = {cvt_e2m1x8(v, m), cvt_e2m1x8(v + 8, m), cvt_e2m1x8(v + 16, m), cvt_e2m1x8(v + 24, m)};
        sc |= (unsigned)(e + 127) << (8 * j);
        unsigned cw = 0u, cw2 = 0u;
#pragma unroll
        for (int G = 0; G < 4; G++) {   /* group G = bytes 4G..4G+3 = channels 8G..8G+7 = word q[G] */
            float mg[4]; int nz = 0;
#pragma unroll
            for (int b = 0; b < 4; b++) { mg[b] = fabsf(v[8 * G + 2 * b]) + fabsf(v[8 * G + 2 * b + 1]); nz += ((q[G] >> (8 * b)) & 0x77u) != 0u; }
            if (nz > 2) bad++;
            int i0 = 0, i1 = 1;   /* the two largest, i0 < i1 after the sort */
            if (mg[1] > mg[0]) { i0 = 1; i1 = 0; }
            for (int b = 2; b < 4; b++) { if (mg[b] > mg[i0]) { i1 = i0; i0 = b; } else if (mg[b] > mg[i1]) i1 = b; }
            if (i0 > i1) { const int s = i0; i0 = i1; i1 = s; }
            const unsigned pair = ((q[G] >> (8 * i0)) & 0xffu) | (((q[G] >> (8 * i1)) & 0xffu) << 8);
            if (G < 2) cw |= pair << (16 * G); else cw2 |= pair << (16 * (G - 2));
            meta[j >> 1] |= (unsigned)(i0 | i1 << 2) << (4 * (4 * (j & 1) + G));
        }
        comp[2 * j] = cw; comp[2 * j + 1] = cw2;
    }
    uint4 *dst = (uint4 *)(wq + i * 32);
    dst[0] = make_uint4(comp[0], comp[1], comp[2], comp[3]); dst[1] = make_uint4(comp[4], comp[5], comp[6], comp[7]);
    ws[i] = sc; wm[i] = make_uint2(meta[0], meta[1]);
    if (bad) atomicAdd(&g_sp24_err, bad);
}

#ifndef LP_SW
/* the MMAs of one 32-channel chunk: 7 tap quads; B of a quad = the dense B fragments of its two tap pairs (b0 / b1 taps A / B, b2 / b3
   taps C / D), the B scales of the four taps' positions in bytes 0..3 */
template <int MT, int RZ, int NR = 2 * RZ>
__device__ __forceinline__ void sp_mma_grp(float (&acc)[MT][NR][2][4], const uint8_t *__restrict__ sx, const uint8_t *__restrict__ sxs,
                                           const uint8_t *__restrict__ wa, const unsigned *__restrict__ was, const uint2 *__restrict__ wme, int wz, int wr, int lane) {
    constexpr int BM = MT * 16;
    const int g = lane >> 2, t = lane & 3, mat = lane >> 3, q = mat >> 1, isB = mat & 1;
    int sl[RZ + 2];
#pragma unroll
    for (int j = 0; j < RZ + 2; j++) sl[j] = (wz * RZ + j) * 180 + wr * 18;
#pragma unroll
    for (int qd = 0; qd < SP_NQ; qd++) {
        int tp[4];
#pragma unroll
        for (int j = 0; j < 4; j++) tp[j] = 4 * qd + j < 26 ? 4 * qd + j : 26;   /* (tap 27: zero weights, any finite B) */
        unsigned af[MT][4], sa[MT], e[MT];   /* the quad's A fragments first (6 MT registers), then B row by row (10 registers) */
#pragma unroll
        for (int m = 0; m < MT; m++) {
            ldsm_x4(af[m], wa + qd * BM * 32 + sw16(m * 16 + (mat & 1) * 8 + (lane & 7), mat >> 1));
            sa[m] = was[qd * BM + m * 16 + g + 8 * (t & 1)];
            const uint2 md = wme[qd * BM + m * 16 + g + SP_MROW(t)];
            e[m] = SP_MHALF(t) ? md.y : md.x;
        }
#pragma unroll
        for (int r = 0; r < NR; r++) {
            int ro[4];
#pragma unroll
            for (int j = 0; j < 4; j++) ro[j] = sl[(r >> 1) + tp[j] / 9] + ((r & 1) + (tp[j] / 3) % 3) * 18 + tp[j] % 3;
            unsigned b01[4], b23[4];
            ldsm_x4(b01, sx + ((isB ? ro[1] : ro[0]) + q * 8 + (lane & 7)) * 16);
            ldsm_x4(b23, sx + ((isB ? ro[3] : ro[2]) + q * 8 + (lane & 7)) * 16);
#pragma unroll
            for (int q2 = 0; q2 < 2; q2++) {
                const unsigned bq[4] = {b01[2 * q2], b01[2 * q2 + 1], b23[2 * q2], b23[2 * q2 + 1]};
                const unsigned sb = (unsigned)sxs[ro[0] + q2 * 8 + g] | ((unsigned)sxs[ro[1] + q2 * 8 + g] << 8) | ((unsigned)sxs[ro[2] + q2 * 8 + g] << 16) |
                                    ((unsigned)sxs[ro[3] + q2 * 8 + g] << 24);
#pragma unroll
                for (int m = 0; m < MT; m++) sp_mma4(acc[m][r][q2], af[m], bq, e[m], sa[m], sb);
            }
        }
    }
}
template <int MT, int TZ, typename T, typename TO>
__global__ void __launch_bounds__(256, LP_MINB(2)) conv_fwd_sp24_k(const T *__restrict__ x, const uint8_t *__restrict__ wq, const unsigned *__restrict__ wsc, const uint2 *__restrict__ wmd,
                                                     const float *__restrict__ b, TO *__restrict__ y,
                                                     int N, int Ci, int D, int H, int W, int Co, int Cop, int Cip, gnp_t gp, double *__restrict__ osum, int Go, split_t sp) {
    constexpr int BM = MT * 16, RZ = TZ / 2, NR = 2 * RZ, TT = (TZ + 2) * 180, NRX = NR;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    uint8_t *sx = smem_raw;                         /* [TT pos][16 B] */
    uint8_t *sxs = sx + TT * 16;                    /* [TT] */
    uint8_t *wa = sxs + ((TT + 127) & ~127);        /* [7 quads][BM co][32 B compressed: 4 taps x 8 B], sw16-swizzled halves */
    unsigned *was = (unsigned *)(wa + SP_NQ * BM * 32);   /* [7][BM]: the 4 taps' ue8m0 scales (bytes 0..3) */
    uint2 *wme = (uint2 *)(was + SP_NQ * BM);             /* [7][BM]: metadata (x taps 0 / 1, y taps 2 / 3) */
    chan_t *ctab = (chan_t *)(wme + SP_NQ * BM);
    float2 *cab = (float2 *)(ctab + 32); unsigned *sgm = (unsigned *)(cab + 32);   /* [32] (a, b), GN mask */
    uint8_t *cu = (uint8_t *)(sgm + 4), *cus = cu + (TZ / 2 + 4) * 96 * 32;   /* sp.up: coarse rows of the up segment (stage_up_coarse) */
    float2 *cab2 = (float2 *)(((uintptr_t)(sp.up ? cus + (TZ / 2 + 4) * 96 : cu) + 15) & ~(uintptr_t)15);   /* [32] slab GN: bank 1 (launch_f4) */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
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
        __syncthreads();   /* (the previous chunk's MMAs read wa) */
        for (int i = threadIdx.x; i < SP_NQ * BM * 2; i += 256) {   /* compressed rows: two 16 B halves */
            const int qd = i / (BM * 2), r = i % (BM * 2), c = r >> 1, h = r & 1;
            const size_t ix = ((size_t)qd * Cop + co0 + c) * nch + ci0 / 32;
            *(uint4 *)(wa + qd * BM * 32 + sw16(c, h)) = __ldg((const uint4 *)(wq + ix * 32 + h * 16));
            if (!h) { was[qd * BM + c] = __ldg(wsc + ix); wme[qd * BM + c] = __ldg(wmd + ix); }
        }
        __syncthreads();
        sp_mma_grp<MT, RZ>(acc, sx, sxs, wa, was, wme, wz, wr, lane);
    }
    fwd_epilogue<MT, NRX, TO, true>(acc, smem_raw, y, b, n, co0, Co, D, H, W, oz0, oy0, ox0, wz, wr, osum, Go, sp, N);
}
#endif

/* prepared sparse weights per conv id (split_t::wkey) and device, valid while pointer, shapes and the step counter are unchanged
   (the dense memo's rule, lp_wmemo_step / lp_wmemo_clear) */
struct sp24memo_s { unsigned key, step, gen; const float *w; int Cop, Cip, Cx, Ox; uint8_t *wq; unsigned *ws; uint2 *wm; size_t n; };
static sp24memo_s g_sp24memo[8][WMEMO_N];
static int sp24_prep(unsigned key, const float *w, int cout, int ci, int Cop, int Cip, int Cx, int CxP, int Ox, int OxP, uint8_t **wq, unsigned **ws, uint2 **wm) {
    const size_t n = (size_t)SP_NQ * Cop * (Cip / 32);
    const int d = cur_dev_();
    static sp24memo_s nomemo[8];
    sp24memo_s *e = key && wmemo_on() ? &g_sp24memo[d][key % WMEMO_N] : &nomemo[d];
    const bool hit = key && wmemo_on() && e->key == key && e->w == w && e->Cop == Cop && e->Cip == Cip && e->Cx == Cx && e->Ox == Ox && e->step == g_wmemo_step &&
                     e->gen == g_wmemo_gen && e->n >= n;
    if (e->n < n) { cudaFree(e->wq); cudaFree(e->ws); cudaFree(e->wm); cudaMalloc(&e->wq, n * 32); cudaMalloc(&e->ws, n * 4); cudaMalloc(&e->wm, n * 8); e->n = n; }
    *wq = e->wq; *ws = e->ws; *wm = e->wm;
    if (hit) return 1;
    e->key = key; e->w = w; e->Cop = Cop; e->Cip = Cip; e->Cx = Cx; e->Ox = Ox; e->step = g_wmemo_step; e->gen = g_wmemo_gen;
    prep_sp24_k<<<nblk_(n, 128), 128>>>(w, *wq, *ws, *wm, cout, ci, Cop, Cip, Cx, CxP, Ox, OxP);
    return 0;
}
#ifndef LP_SW
template <int MT, int TZ> static void launch_sp24(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const unsigned *ws, const uint2 *wm, const float *b, int cout, void *y,
                                                  int Cop, int Cip, gnp_t gp, double *osum, int Go, split_t sp) {
    constexpr int TT = (TZ + 2) * 180, BM = MT * 16;
    size_t smem = (size_t)TT * 16 + ((TT + 127) & ~127) + SP_NQ * BM * (32 + 4 + 8) + 32 * sizeof(chan_t) + 32 * 8 + 16 + (sp.up ? (TZ / 2 + 4) * 96 * 33 + 16 : 0)
                  + (gp.ns || sp.gp2.ns ? 32 * 8 + 16 : 0);
    if (smem < 8 * 256 * sizeof(float)) smem = 8 * 256 * sizeof(float);
    static int attr[8];
    if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)conv_fwd_sp24_k<MT, TZ, mx4_t, mx4_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
    conv_fwd_sp24_k<MT, TZ, mx4_t, mx4_t><<<grid, 256, smem>>>((const mx4_t *)x, wq, ws, wm, b, (mx4_t *)y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, Cip, gp, osum, Go, sp);
}
#endif
static int g_sp24_force = -1;   /* test hook: 1 the sparse kernel for every eligible call, 0 never, -1 by the registered weights */
extern "C" void lp_sp24_force(int on) { g_sp24_force = on; }
/* the convs whose forward weights carry the pattern (unet_apply_sparse24: per host thread, i.e. per GPU trainer), and the flipped copies a
   backward-data made from such weights when the mask transposes (mode 3, lp_sp24_note_flip) */
#define SP24_NREG 512
struct sp24reg_s { const float *w; size_t n; int mode; };
static thread_local sp24reg_s g_sp24_reg[SP24_NREG];
static thread_local int g_sp24_nreg;
static thread_local const float *g_sp24_flo, *g_sp24_fhi;
extern "C" void lp_sp24_reg_clear(void) { g_sp24_nreg = 0; g_sp24_flo = g_sp24_fhi = nullptr; }
extern "C" void lp_sp24_reg(const float *w, size_t n, int mode) { if (g_sp24_nreg < SP24_NREG) g_sp24_reg[g_sp24_nreg++] = sp24reg_s{w, n, mode}; }
static int sp24_mode_of(const float *w) {   /* the registered conv whose weights contain w (a backward-data range call reads an offset) */
    for (int i = 0; i < g_sp24_nreg; i++) if (w >= g_sp24_reg[i].w && w < g_sp24_reg[i].w + g_sp24_reg[i].n) return g_sp24_reg[i].mode;
    return 0;
}
extern "C" void lp_sp24_note_flip(const float *w, const float *flipped, size_t n) {
    if (sp24_mode_of(w) == 3) { g_sp24_flo = flipped; g_sp24_fhi = flipped + n; } else g_sp24_flo = g_sp24_fhi = nullptr;
}
static unsigned long g_sp24_calls;   /* sparse convs run (all threads; tests) */
extern "C" unsigned long lp_sp24_calls(void) { return g_sp24_calls; }
extern "C" unsigned lp_sp24_err(void) { unsigned h = 0; cudaMemcpyFromSymbol(&h, g_sp24_err, 4); return h; }
extern "C" void lp_sp24_err_clear(void) { const unsigned z = 0; cudaMemcpyToSymbol(g_sp24_err, &z, 4); }
static int sp24_eligible(const float *w) {   /* (the flip note is one-shot: the conv_fwd_tc call right after the flip consumes it) */
    if (g_sp24_force >= 0) return g_sp24_force;
    const bool fl = g_sp24_flo && w >= g_sp24_flo && w < g_sp24_fhi;
    g_sp24_flo = g_sp24_fhi = nullptr;
    if (fl) return 1;
    const int m = sp24_mode_of(w);
    return m == 2 || m == 3;
}
extern "C" int lp_sp24_conv_fwd(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, gnp_t gp, double *osum, int Go, split_t sp) {
#ifdef LP_SW
    return 0;
#else
    if (!sp24_eligible(w) || lp_sw_build() || xbf != 4 || ybf != 4 || xs.c <= 64 || sp.accum) return 0;
    if (g_w2d < 0) g_w2d = getenv("UFSM_W4_2D") ? atoi(getenv("UFSM_W4_2D")) : 0;
    if (g_w2d) return 0;   /* (2D weight scales: not in the sparse prep) */
    if (sp.up) sp.up = 2;  /* (fwd_f4_t: UFSM_UP_H2 default) */
    int Cop = (cout + 15) / 16 * 16;
    const int Cx = sp.x2 ? sp.c_split : xs.c, CxP = (Cx + 31) / 32 * 32;
    int Ox = sp.y2 ? sp.o_split : cout, OxP = (Ox + 31) / 32 * 32;
    if (sp.y2) Cop = OxP + (cout - Ox + 31) / 32 * 32; else if (cout > 16) Cop = (cout + 31) / 32 * 32;
    const int Cip = CxP + (xs.c - Cx + 31) / 32 * 32;
    const int MT = Cop % 64 == 0 ? 4 : Cop % 32 == 0 ? 2 : 1;
    const int TZ = MT == 4 ? 2 : xs.d >= 16 ? 4 : 2;
    if (MT == 1) return 0;
    uint8_t *wq; unsigned *ws; uint2 *wm;
    sp24_prep(sp.wkey, w, cout, xs.c, Cop, Cip, Cx, CxP, sp.y2 ? Ox : -1, OxP, &wq, &ws, &wm);
    const int nmt = Cop / (MT * 16);
    {   /* several output tiles over a GroupNorm input: requantise it once (fwd_f4_t's prenorm, the same bits) */
        static int pn = -1; if (pn < 0) pn = getenv("UFSM_F4_PRENORM") ? atoi(getenv("UFSM_F4_PRENORM")) : 1;
        if (pn && nmt >= 2 && gp.G && !gp.ns && !sp.x2 && !sp.up && !sp.sr && !sp.Di && !sp.zw1 && xs.c % 32 == 0 && Cip == xs.c) {
            const size_t S = (size_t)xs.d * xs.h * xs.w, nbytes = (size_t)xs.n * (xs.c / 32) * S * 17;
            uint8_t *q = f4_prenorm_buf(nbytes);
            if (q) {
                gn_prenorm_f4_k<mx4_t><<<dim3(nblk_(S, 256), (unsigned)(xs.c / 32), (unsigned)xs.n), 256>>>((const mx4_t *)x, xs.n, xs.c, S, gp, q);
                x = q; gp = gnp_t{};
            }
        }
    }
    dim3 grid(nblk_(xs.w, 16), nblk_(xs.h, 8), (unsigned)(zw_tiles(xs.d, TZ, sp) * nmt * xs.n));
    if (MT == 4) launch_sp24<4, 2>(grid, x, xs, wq, ws, wm, b, cout, y, Cop, Cip, gp, osum, Go, sp);
    else if (TZ == 4) launch_sp24<2, 4>(grid, x, xs, wq, ws, wm, b, cout, y, Cop, Cip, gp, osum, Go, sp);
    else launch_sp24<2, 2>(grid, x, xs, wq, ws, wm, b, cout, y, Cop, Cip, gp, osum, Go, sp);
    LPCK();
    __atomic_add_fetch(&g_sp24_calls, 1, __ATOMIC_RELAXED);
    return 1;
#endif
}
