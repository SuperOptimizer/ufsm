/* conv_fwd_f4v_k: the MX-fp4 3^3 stride-1 conv for stored MX-fp4 inputs copied without a transform (the trunk's forward after
   gn_prenorm_f4_k / of a raw block output, the lean recompute, the backward-data of an MX-fp4 gradient), MT 4 (64 output rows a block).
   conv_fwd_f4_k<4, 2> (ncu, L0 32 x 256^2 x 256, 2026-11-17): tensor pipe 46 %, issue slots 23 % busy, 128 registers (two blocks an SM)
   with 116 M local loads + 116 M local stores a call (55 % of the L1 sectors), the staging behind barriers with nothing in flight,
   the weight rows gathered at 16 B per 32 B sector. Here, with the MMA core and its order unchanged (f4r_mma_grp, the 4-row path):
   - one block an SM (255 registers: no spills) with a 16 x 8 x 4 output tile: 64 rows x 4 rows of 16 voxels a warp (MT 4, NR 4), so
     a staged input byte feeds 2x the MMAs and the halo is 1080 positions for 512 voxels (720 for 256);
   - the input tile double-buffered by cp.async (16 B rows, zero-filled outside / masked) and its scale bytes through registers: chunk
     c + 1 is issued while chunk c's two weight stages run their MMAs;
   - the weights re-laid out once a call (prep_w4v_k: [co block][chunk][pair][64 rows][32 B], the sw16 swizzle and the scale pairs
     pre-built), so each 7-pair stage is one contiguous 15 KB cp.async copy, double-buffered: stage s + 1 in flight during stage s.
   Same MMAs on the same operands in the same order per accumulator: the outputs are those of conv_fwd_f4_k<4, 2> bit for bit; the GN
   statistics too (f4v_epi_st keeps each 2-plane half's sums apart and adds them in conv_fwd_f4_k's warp order). UFSM_F4V=0: off. */
#include "lp_f4fwd.cuh"

int g_f4v = -1;   /* mgpu: env (UFSM_F4V; lp_set_f4v: tests) */
extern "C" void lp_set_f4v(int on) { g_f4v = on; }
int g_f4_pn = -1;   /* mgpu: env */
extern "C" void lp_set_f4_prenorm(int on) { g_f4_pn = on; }
static int g_f4v_preslab = -1;   /* mgpu: env (UFSM_F4V_PRESLAB=Z: force the slabbed prenorm with Z-plane slabs; tests) */
extern "C" void lp_set_f4v_preslab(int z) { g_f4v_preslab = z; }
static unsigned long g_f4v_calls;   /* mgpu: guarded (atomics; tests: lp_f4v_calls) */
extern "C" unsigned long lp_f4v_calls(void) { return g_f4v_calls; }
extern "C" int lp_get_f4v(void) { if (g_f4v < 0) g_f4v = getenv("UFSM_F4V") ? atoi(getenv("UFSM_F4V")) : 0; return g_f4v; }
int lp_f4v_slab_forced(void) { if (g_f4v_preslab < 0) g_f4v_preslab = getenv("UFSM_F4V_PRESLAB") ? atoi(getenv("UFSM_F4V_PRESLAB")) : 0; return lp_get_f4v() && g_f4v_preslab > 0; }

#ifndef LP_SW
constexpr int F4V_PG = 7;   /* tap pairs a weight stage (two a chunk, as conv_fwd_f4_k) */
template <int MT, int TZ> struct f4v_geo {
    static constexpr int BM = MT * 16, TT = (TZ + 2) * 180, XB = TT * 16, XS = (TT + 127) & ~127, XBUF = XB + XS;
    static constexpr int WST = F4V_PG * BM * 32, WSS = F4V_PG * BM * 2, WBUF = WST + WSS, GT = 2 * 32 * 8 + 16, SMEM = 2 * XBUF + 2 * WBUF + GT, NI = (TT + 255) / 256;
};
/* wq [tap 28][Cop][Cip / 2] / ws [tap][Cop][nch] (prep_w4_k) -> wv [Cop / BM][nch][pair 14][BM][32 B] (tap 2 pp | 2 pp + 1 at sw16) and
   wvs [Cop / BM][nch][14][BM] (byte 0 tap 2 pp's scale, byte 1 tap 2 pp + 1's): conv_fwd_f4_k's weight stage contents, in its order */
__global__ void prep_w4v_k(const uint8_t *__restrict__ wq, const uint8_t *__restrict__ ws, uint8_t *__restrict__ wv, unsigned short *__restrict__ wvs, int Cop, int Cip, int BM) {
    const int nch = Cip / 32;
    const size_t nr = (size_t)Cop * nch * 14, i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;   /* (cb, c, pp, r) rows */
    if (i >= nr * 2) return;
    const int h = (int)(i & 1);
    size_t q = i >> 1;
    const int r = (int)(q % BM); q /= BM;
    const int pp = (int)(q % 14); q /= 14;
    const int c = (int)(q % nch), cb = (int)(q / nch), co = cb * BM + r, tap = 2 * pp + h;
    *(uint4 *)(wv + (((size_t)cb * nch + c) * 14 + pp) * BM * 32 + sw16(r, h)) = *(const uint4 *)(wq + ((size_t)tap * Cop + co) * (Cip / 2) + c * 16);
    if (!h) wvs[(((size_t)cb * nch + c) * 14 + pp) * BM + r] = (unsigned short)(ws[((size_t)tap * Cop + co) * nch + c] | (ws[((size_t)(tap + 1) * Cop + co) * nch + c] << 8));
}
/* fwd_epilogue + fwd_epi_mxt<16> (the MX output when every 32-channel tile is one stored block, the GN statistics in osum) for the
   16 x 8 x 4 tile, the statistics of planes oz0, oz0 + 1 and oz0 + 2, oz0 + 3 apart: each half is one conv_fwd_f4_k<MT, 2> block,
   its warp w (wz' = w / 4 plane, wr = 2 (w % 4) rows) = this kernel's warp 4 half + w % 4, rows 2 (w / 4) .. + 1, and its sums are
   formed as there (rows in order, 16 voxels a row, the warps in order from 0, one fixed-point add per channel and half): the same
   statistics bits. Values, scales and SR keys per voxel as fwd_epi_mxt. */
template <int MT, int NR, typename T>
__device__ __forceinline__ void f4v_epi_st(float (&acc)[MT][NR][2][4], unsigned char *smem_raw, T *y, const float *b, int n, int co0, int Co, int D, int H, int W,
                                           int oz0, int oy0, int ox0, int wz, int wr, double *osum, int Go, const split_t &sp, int N) {
    static_assert(NR == 4 && MT % 2 == 0, "f4v_epi_st: two 2-plane halves");
    constexpr int HV = 16, B = MX_BITS(T), RBM = 32 * B / 8, LPV = 32 / HV, CPL = 32 / LPV, NW = CPL * B / 32, RZ = NR / 2, BM = MT * 16;
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, vx = lane / LPV, qd = lane % LPV;
    int zs0 = sp.zlo, zs1 = D - sp.zhi;
    if (sp.yD) { oz0 += sp.yzo; zs0 += sp.yzo; zs1 += sp.yzo; D = sp.yD; }
    const size_t S = (size_t)D * H * W;
    const bool sst = sp.stored_stats;
    __syncthreads();   /* every warp is past its MMA loop: the stages are free */
    float *tb = (float *)(smem_raw + 1024) + (threadIdx.x >> 5) * HV * 36;
    float bias[MT][2];
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
        for (int h = 0; h < 2; h++) { const int co = co0 + m * 16 + g + 8 * h; bias[m][h] = b && co < Co ? b[co] : 0.f; }
    float ps[2][MT / 2], pss[2][MT / 2];
#pragma unroll
    for (int hf = 0; hf < 2; hf++)
#pragma unroll
        for (int jb = 0; jb < MT / 2; jb++) { ps[hf][jb] = 0.f; pss[hf][jb] = 0.f; }
#pragma unroll
    for (int r = 0; r < NR; r++) {
        const int oz = oz0 + wz * RZ + (r >> 1), oy = oy0 + wr + (r & 1), hf = r >> 1;
        if (oz >= D || oy >= H) continue;   /* warp-uniform */
        const bool zst = oz >= zs0 && oz < zs1;
#pragma unroll
        for (int jb = 0; jb < MT / 2; jb++) {
            const int cp = co0 + 32 * jb;
            if (cp >= Co) continue;   /* pure padding */
            const int blk = cp / 32, nbt = mx_nb(Co);
            uint8_t *qt = (uint8_t *)y;
#pragma unroll
            for (int q2 = 0; q2 < 2; q2++)
#pragma unroll
                for (int mm = 0; mm < 2; mm++)
#pragma unroll
                    for (int k = 0; k < 4; k++) tb[(8 * q2 + 2 * t + (k & 1)) * 36 + mm * 16 + g + 8 * (k >> 1)] = acc[2 * jb + mm][r][q2][k] + bias[2 * jb + mm][k >> 1];
            __syncwarp();
            float v[CPL], am = 0.f;
#pragma unroll
            for (int k = 0; k < CPL / 4; k++) { const float4 f = *(const float4 *)(tb + vx * 36 + CPL * qd + 4 * k); v[4 * k] = f.x; v[4 * k + 1] = f.y; v[4 * k + 2] = f.z; v[4 * k + 3] = f.w; }
#pragma unroll
            for (int k = 0; k < CPL; k++) am = fmaxf(am, fabsf(v[k]));
#pragma unroll
            for (int o = 1; o < LPV; o <<= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
            const int e = mx_exp(am, mxf<B>::inv_qmax);
            const float mult = exp2i(-e), sc = exp2i(e);
            const size_t ri = ((size_t)n * nbt + blk) * S + ((size_t)oz * H + oy) * W + ox0 + vx;
            unsigned wq[NW];
#pragma unroll
            for (int k = 0; k < NW; k++) wq[k] = mxf<B>::enc8x(v + (32 / B) * k, mult);
            if constexpr (B == 4) if (sp.osr) {
#pragma unroll
                for (int k = 0; k < NW; k++) { uint32_t hh[4]; sr_hash4(sp.osr, (uint64_t)ri * 4 + (CPL * qd) / 8 + k, hh); wq[k] = sr_e2m1_word(v + 8 * k, mult, hh); }
            }
            uint8_t *dst = qt + ri * RBM + (CPL * B / 8) * qd;
            if constexpr (NW == 4) *(uint4 *)dst = make_uint4(wq[0], wq[1], wq[2], wq[3]);
            else if constexpr (NW == 2) *(uint2 *)dst = make_uint2(wq[0], wq[1]);
            else *(unsigned *)dst = wq[0];
            if (!qd) (qt + (size_t)N * nbt * S * RBM)[ri] = (uint8_t)(e + 127);
            if (zst) {
                if (sst) {
#pragma unroll
                    for (int k = 0; k < NW; k++) {
                        if constexpr (B == 4) {
#pragma unroll
                            for (int c = 0; c < 4; c++) { const float2 d = dec_e2m1x2((wq[k] >> (8 * c)) & 255u); v[8 * k + 2 * c] = d.x * sc; v[8 * k + 2 * c + 1] = d.y * sc; }
                        } else {
                            const float2 d0 = dec_e4m3x2((unsigned short)(wq[k] & 0xffffu)), d1 = dec_e4m3x2((unsigned short)(wq[k] >> 16));
                            v[4 * k] = d0.x * sc; v[4 * k + 1] = d0.y * sc; v[4 * k + 2] = d1.x * sc; v[4 * k + 3] = d1.y * sc;
                        }
                    }
                    __syncwarp();
#pragma unroll
                    for (int k = 0; k < CPL / 4; k++) *(float4 *)(tb + vx * 36 + CPL * qd + 4 * k) = make_float4(v[4 * k], v[4 * k + 1], v[4 * k + 2], v[4 * k + 3]);
                }
                __syncwarp();
#pragma unroll
                for (int k = 0; k < HV; k++) { const float xv = tb[k * 36 + lane]; ps[hf][jb] += xv; pss[hf][jb] += xv * xv; }
            }
            __syncwarp();
        }
    }
#pragma unroll
    for (int hf = 0; hf < 2; hf++)
#pragma unroll
        for (int jb = 0; jb < MT / 2; jb++) { tb[2 * BM * hf + 32 * jb + lane] = ps[hf][jb]; tb[2 * BM * hf + BM + 32 * jb + lane] = pss[hf][jb]; }
    __syncthreads();
    const float *t0 = (const float *)(smem_raw + 1024);
    float *cs = (float *)smem_raw;   /* [2 halves][2 BM] */
    for (int i = threadIdx.x; i < 4 * BM; i += blockDim.x) {
        const int hb = i / (2 * BM), c = i % (2 * BM);
        float s = 0.f;
        for (int w = 0; w < 8; w++) s += t0[(hb * 4 + (w & 3)) * HV * 36 + 2 * BM * (w >> 2) + c];   /* conv_fwd_f4_k's warp w of the half's block */
        float cv = 0.f; cv += s;   /* (its zeroed cs[c] += s) */
        cs[i] = cv;
    }
    __syncthreads();
    if (threadIdx.x < 2 * BM) {
        const int hb = threadIdx.x / BM, c = threadIdx.x % BM;
        if (co0 + c < Co) {
            const int cpg = Co / Go, ng = n * Go + (co0 + c) / cpg;
            gn_acc_add(osum, 2 * ng, (double)cs[2 * BM * hb + c]);
            gn_acc_add(osum, 2 * ng + 1, (double)cs[2 * BM * hb + BM + c]);
        }
    }
}
template <int MT, int TZ, typename TO, bool ST, bool G>   /* G: GN+SiLU staging (gp / sp.gp2) */
__global__ void __launch_bounds__(256, 1) conv_fwd_f4v_k(const mx4_t *__restrict__ x, const uint8_t *__restrict__ wv, const unsigned short *__restrict__ wvs,
                                                         const float *__restrict__ b, TO *__restrict__ y, int N, int Ci, int D, int H, int W, int Co, int Cop, int Cip,
                                                         gnp_t gp, double *__restrict__ osum, int Go, split_t sp, skipm_t sk) {
    using GEO = f4v_geo<MT, TZ>;
    constexpr int BM = GEO::BM, RZ = TZ / 2, NR = 2 * RZ, TT = GEO::TT, NI = GEO::NI, XB = GEO::XB, XBUF = GEO::XBUF, WST = GEO::WST, WBUF = GEO::WBUF;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    uint8_t *const xb0 = smem_raw, *const wb0 = smem_raw + 2 * XBUF;   /* input buffers [rows TT x 16 B | scales], weight buffers [rows | scale pairs] */
    float2 *const gt = (float2 *)(wb0 + 2 * WBUF);                      /* GN+SiLU: [2][32] (a, b) of a chunk's channels, then [2] GN masks */
    unsigned *const gmw = (unsigned *)(gt + 64);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wz = warp >> 2, wr = (warp & 3) * 2;
    const int ox0 = blockIdx.x * 16, oy0 = blockIdx.y * 8;
    int bz = blockIdx.z;
    const int tz0 = sp.zw1 ? sp.zw0 / TZ : 0, nzt = sp.zw1 ? (sp.zw1 + TZ - 1) / TZ - tz0 : (D + TZ - 1) / TZ;
    const int oz0 = (tz0 + bz % nzt) * TZ; bz /= nzt;
    const int nmt = Cop / BM, cb = bz % nmt, co0 = cb * BM, n = bz / nmt, nch = Cip / 32;
    float acc[MT][NR][2][4];
#pragma unroll
    for (int m = 0; m < MT; m++) for (int r = 0; r < NR; r++) for (int q = 0; q < 2; q++) for (int k = 0; k < 4; k++) acc[m][r][q][k] = 0.f;
    const int zi0 = sp.zi0, Di = sp.Di ? sp.Di : D;
    const size_t plane = (size_t)Di * H * W;
    const int Cx = sp.x2 ? sp.c_split : Ci, CxP = (Cx + 31) / 32 * 32;
    if (sk_active(sk, oz0, oy0, ox0)) {
        int off[NI];   /* this thread's staged positions: voxel index in the chunk's plane, -1 zero row (outside, masked), -2 none */
#pragma unroll
        for (int k = 0; k < NI; k++) {
            const int li = threadIdx.x + 256 * k, ix = li % 18, iy = (li / 18) % 10, iz = li / 180;
            const int gz = oz0 - 1 + iz, gy = oy0 - 1 + iy, gx = ox0 - 1 + ix;
            bool inb = gz + zi0 >= 0 && gz + zi0 < Di && gy >= 0 && gy < H && gx >= 0 && gx < W;
            if (inb && sk.mi) inb = sk_in(sk, gz, gy, gx);
            off[k] = li >= TT ? -2 : inb ? (int)(((size_t)(gz + zi0) * H + gy) * W + gx) : -1;
        }
        unsigned sreg[NI];
        /* chunk c's rows into input buffer bi by cp.async (zero-filled: nothing read), its scale bytes into sreg (1 for a zero row, as
           conv_fwd_f4_k's copy staging) */
        auto issue_x = [&](int c, int bi) {
            const chan_t ch = make_chan(seg_ci(c * 32, Ci, Cx, CxP), Ci, Cx, n, plane, x, sp, gp, N, 0);
            uint8_t *sx = xb0 + bi * XBUF;
#pragma unroll
            for (int k = 0; k < NI; k++) {
                if (off[k] == -2) continue;
                const int li = threadIdx.x + 256 * k;
                const bool ok = ch.p && off[k] >= 0;
                const uint8_t *src = ok ? (const uint8_t *)ch.p + (size_t)off[k] * ch.rb : (const uint8_t *)wv;
                if (ch.bw == 32) cp_async<16>(sx + li * 16, src, ok);
                else if (ch.bw == 16) { cp_async<8>(sx + li * 16, src, ok); cp_async<8>(sx + li * 16 + 8, src, false); }
                else { cp_async<4>(sx + li * 16, src, ok); cp_async<4>(sx + li * 16 + 4, src, false); cp_async<8>(sx + li * 16 + 8, src, false); }
                sreg[k] = ok ? (unsigned)__ldg(ch.sp + off[k]) : 1u;
            }
        };
        auto store_s = [&](int bi) {
            uint8_t *sxs = xb0 + bi * XBUF + XB;
#pragma unroll
            for (int k = 0; k < NI; k++) if (off[k] != -2) sxs[threadIdx.x + 256 * k] = (uint8_t)sreg[k];
        };
        /* weight stage s (chunk s / 2, pairs 7 (s & 1) ..) into weight buffer s & 1: one contiguous copy */
        auto issue_w = [&](int s) {
            uint8_t *wb = wb0 + (s & 1) * WBUF;
            const size_t st = ((size_t)cb * nch + (s >> 1)) * 14 + F4V_PG * (s & 1);
            const uint8_t *src = wv + st * BM * 32, *ss = (const uint8_t *)(wvs + st * BM);
            for (int i = threadIdx.x; i < WST / 16; i += 256) cp_async<16>(wb + i * 16, src + i * 16, true);
            for (int i = threadIdx.x; i < GEO::WSS / 16; i += 256) cp_async<16>(wb + WST + i * 16, ss + i * 16, true);
        };
        /* GN+SiLU inputs (conv_fwd_f4_k's fastg staging, f4r_stage_g): the stored rows land by cp.async as above, then each thread
           transforms its own positions in place one stage later (its own copies are visible to it after its wait_group): decode,
           affine + SiLU (the halved affine when every channel has a GroupNorm, f4r_stage_g_t RG), requantise; the same bits */
        auto gtab = [&](int c, int tb) {   /* chunk c's coefficients -> table tb (threads 0 .. 31) */
            if (threadIdx.x < 32) {
                const chan_t ch = make_chan(seg_ci(c * 32 + (int)threadIdx.x, Ci, Cx, CxP), Ci, Cx, n, plane, x, sp, gp, N, 0);
                gt[tb * 32 + threadIdx.x] = make_float2(ch.a, ch.b);
                const unsigned gm = __ballot_sync(0xffffffffu, ch.g != 0);
                if (threadIdx.x == 0) gmw[tb] = gm;
            }
        };
        auto xform = [&](int bi, int tb) {
            uint8_t *sx = xb0 + bi * XBUF, *sxs = sx + XB;
            const float2 *ab = gt + tb * 32;
            const unsigned gm = gmw[tb];
#pragma unroll 1
            for (int k = 0; k < NI; k++) {
                if (off[k] == -2) continue;
                const int li = threadIdx.x + 256 * k;
                const bool in = off[k] >= 0;
                const uint4 rw = *(const uint4 *)(sx + li * 16);
                const float sc = mx_scale(sreg[k]);
                const unsigned w4[4] = {rw.x, rw.y, rw.z, rw.w};
                float v[32];
#pragma unroll
                for (int j = 0; j < 4; j++)
#pragma unroll
                    for (int q = 0; q < 4; q++) { const float2 a = dec_e2m1x2(w4[j] >> (8 * q)); v[8 * j + 2 * q] = a.x * sc; v[8 * j + 2 * q + 1] = a.y * sc; }
                if (gm == 0xffffffffu) {
#pragma unroll
                    for (int j = 0; j < 32; j++) {
                        const float2 c2 = ab[j];
                        const float hh = fmaf(v[j], 0.5f * c2.x, 0.5f * c2.y);
                        const float th = tanh_silu(hh);
                        v[j] = in ? fmaf(hh, th, hh) : 0.f;
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < 32; j++) if ((gm >> j) & 1u) { const float2 c2 = ab[j]; v[j] = in ? act_ab(v[j], c2.x, c2.y, true) : 0.f; }
                }
                unsigned amu = 0u;
#pragma unroll
                for (int j = 0; j < 32; j++) amu = amax_u(amu, v[j]);
                const int e = mx_exp(__uint_as_float(amu), 1.f / 6.f);
                const float m = exp2i(-e);
                *(uint4 *)(sx + li * 16) = make_uint4(cvt_e2m1x8(v, m), cvt_e2m1x8(v + 8, m), cvt_e2m1x8(v + 16, m), cvt_e2m1x8(v + 24, m));
                sxs[li] = (uint8_t)(e + 127);
            }
        };
        issue_x(0, 0); issue_w(0);
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        if constexpr (G) {
            gtab(0, 0);
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            __syncthreads();
            xform(0, 0);
        } else store_s(0);
        const int ns = 2 * nch;
        for (int s = 0; s < ns; s++) {
            const int c = s >> 1, h = s & 1;
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");   /* stage s and (h 0) chunk c landed (this thread's copies) */
            __syncthreads();                                         /* everyone's; and every warp is past stage s - 1's MMA */
            if (s + 1 < ns) {
                if (!h && c + 1 < nch) { issue_x(c + 1, (c + 1) & 1); if constexpr (G) gtab(c + 1, (c + 1) & 1); }   /* (buffers (c + 1) & 1: chunk c - 1's, done) */
                issue_w(s + 1);
                asm volatile("cp.async.commit_group;\n" ::: "memory");
                if (h) { if constexpr (G) xform((c + 1) & 1, (c + 1) & 1); else store_s((c + 1) & 1); }   /* (chunk c + 1's rows: this thread's copies, waited for above) */
            }
            const uint8_t *sx = xb0 + (c & 1) * XBUF, *sxs = sx + XB, *wa = wb0 + (s & 1) * WBUF;
            const unsigned short *was = (const unsigned short *)(wa + WST);
            if (!h) f4r_mma_grp<MT, RZ, 0, 0, F4V_PG>(acc, sx, sxs, wa, was, oz0, wz, wr, lane);
            else f4r_mma_grp<MT, RZ, 0, F4V_PG, 14 - F4V_PG>(acc, sx, sxs, wa, was, oz0, wz, wr, lane);
        }
    }
    if constexpr (ST) f4v_epi_st<MT, NR, TO>(acc, smem_raw, y, b, n, co0, Co, D, H, W, oz0, oy0, ox0, wz, wr, osum, Go, sp, N);
    else fwd_epilogue<MT, NR, TO, true>(acc, smem_raw, y, b, n, co0, Co, D, H, W, oz0, oy0, ox0, wz, wr, osum, Go, sp, N);
}
template <int MT, int TZ, typename TO>
static void launch_f4v(const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, int Cip, gnp_t gp, double *osum, int Go, const split_t &sp,
                       const skipm_t &sk, bool prep = true) {
    using GEO = f4v_geo<MT, TZ>;
    const int nch = Cip / 32;
    uint8_t *wv = lp_buf<uint8_t>(0, (size_t)Cop * nch * 14 * 32);
    unsigned short *wvs = (unsigned short *)lp_buf<uint8_t>(1, (size_t)Cop * nch * 14 * 2);
    const size_t nr = (size_t)Cop * nch * 14 * 2;
    if (prep) prep_w4v_k<<<nblk_(nr, 256), 256>>>(wq, ws, wv, wvs, Cop, Cip, GEO::BM);
    size_t smem = GEO::SMEM;
    if (smem < 1024 + FEPI_SCR(16)) smem = 1024 + FEPI_SCR(16);
    dim3 grid(nblk_(xs.w, 16), nblk_(xs.h, 8), (unsigned)(zw_tiles(xs.d, TZ, sp) * (Cop / GEO::BM) * xs.n));
    const int d = cur_dev_(), st = osum != nullptr, g = gp.G != 0 || sp.gp2.G != 0;
#define F4V_L(STV, GV) do { static int attr[8]; if (!attr[d]) { attr[d] = 1; cudaFuncSetAttribute((const void *)conv_fwd_f4v_k<MT, TZ, TO, STV, GV>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); } \
                            conv_fwd_f4v_k<MT, TZ, TO, STV, GV><<<grid, 256, smem>>>((const mx4_t *)x, wv, wvs, b, (TO *)y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, Cip, gp, osum, Go, sp, sk); } while (0)
    if (st) { if (g) F4V_L(true, true); else F4V_L(true, false); }
    else { if (g) F4V_L(false, true); else F4V_L(false, false); }
#undef F4V_L
}
#endif
/* fwd_f4_t<mx4_t, mx4_t>'s hook, after its weight prep and gn_prenorm_f4_k: 1 if conv_fwd_f4v_k ran the conv */
int lp_f4v_try(const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, int Cip, gnp_t gp, double *osum, int Go, const split_t &sp) {
#ifdef LP_SW
    (void)x; (void)xs; (void)wq; (void)ws; (void)b; (void)cout; (void)y; (void)Cop; (void)Cip; (void)gp; (void)osum; (void)Go; (void)sp;
    return 0;
#else
    if (!lp_get_f4v() || lp_sw_build()) return 0;
    const size_t plane = (size_t)(sp.Di ? sp.Di : xs.d) * xs.h * xs.w;
    const char *why = Cop % 64 ? "Cop % 64" : gp.ns || sp.gp2.ns ? "slab-local GN" : sp.sr ? "staging SR" : sp.up ? "fused upsample" : xs.d < 8 ? "depth < 8"   /* (MT 4; copy staging only;
                                                                                                                            small depths: conv_fwd_f4_k's 2-plane tiles) */
                    : sp.zw1 && (sp.zw0 % 4 || (sp.zw1 % 4 && sp.zw1 != xs.d)) ? "z-window off 4-plane tiles" : plane >= ((size_t)1 << 31) ? "plane >= 2^31"
                    : osum && (sp.y2 || (xs.w & 15) || cout <= 16) ? "statistics epilogue (split output / W % 16 / <= 16 outputs)" : nullptr;   /* (f4v_epi_st: fwd_epi_mxt's conditions) */
    static int dbg = -1; /* mgpu: debug */ if (dbg < 0) dbg = getenv("UFSM_F4V_DEBUG") ? atoi(getenv("UFSM_F4V_DEBUG")) : 0;   /* the first N calls: kernel and reason */
    if (dbg > 0) { dbg--; fprintf(stderr, "f4v: %d->%d %dx%dx%d Cop %d Cip %d stats %d osr %d x2 %d (c_split %d) up %d gn %d/%d: %s%s\n", xs.c, cout, xs.d, xs.h, xs.w, Cop, Cip, osum != nullptr, sp.osr != 0, sp.x2 != nullptr, sp.c_split, sp.up, gp.G, sp.gp2.G,
                                   why ? "conv_fwd_f4_k, " : "conv_fwd_f4v_k", why ? why : ""); }
    if (why) return 0;
    launch_f4v<4, 4, mx4_t>(x, xs, wq, ws, b, cout, y, Cop, Cip, gp, osum, Go, sp, lp_skip_get(xs, sp, 1));
    __atomic_fetch_add(&g_f4v_calls, 1ul, __ATOMIC_RELAXED);
    return 1;
#endif
}

#ifndef LP_SW
/* gn_prenorm_f4_k over the input planes [a, a + L) only (into a depth-L tensor): the same staging and requantisation per voxel */
/* (a split input's segment: padded channels cpo .. cpo + 32 gridDim.y of the conv's input, chan as the conv stages them; nb blocks out) */
__global__ void __launch_bounds__(256) gn_prenorm_f4z_k(const mx4_t *__restrict__ x, int N, int Ci, int Cx, int CxP, int cpo, size_t S, int a, int L, int HW, gnp_t gp, split_t sp,
                                                        uint8_t *__restrict__ y) {
    __shared__ chan_t ctab[32]; __shared__ float2 cab[32]; __shared__ unsigned sgm;
    const int blk = blockIdx.y, n = blockIdx.z, nb = gridDim.y;
    if (threadIdx.x < 32) {
        const chan_t c = make_chan(seg_ci(cpo + blk * 32 + (int)threadIdx.x, Ci, Cx, CxP), Ci, Cx, n, S, x, sp, gp, N);
        ctab[threadIdx.x] = c; cab[threadIdx.x] = make_float2(c.a, c.b);
        const unsigned gm = __ballot_sync(0xffffffffu, c.g != 0);
        if (threadIdx.x == 0) sgm = gm;
    }
    __syncthreads();
    const size_t Ss = (size_t)L * HW, v = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (v >= Ss) return;
    float val[32];
    stage_row32<mx4_t>(ctab, (size_t)a * HW + v, true, true, val, cab, sgm);
    unsigned amu = 0u;
#pragma unroll
    for (int k = 0; k < 32; k++) amu = amax_u(amu, val[k]);
    const int e = mx_exp(__uint_as_float(amu), 1.f / 6.f);
    const float m = exp2i(-e);
    const size_t ri = ((size_t)n * nb + blk) * Ss + v;
    *(uint4 *)(y + ri * 16) = make_uint4(cvt_e2m1x8(val, m), cvt_e2m1x8(val + 8, m), cvt_e2m1x8(val + 16, m), cvt_e2m1x8(val + 24, m));
    y[(size_t)N * nb * Ss * 16 + ri] = (uint8_t)(e + 127);
}
#endif
/* fwd_f4_t's GroupNorm input when the layer's MX-fp4 copy (gn_prenorm_f4_k) is over its workspace budget (unet_ws_plan's spare: try9's
   level 0 needs 963 MB, the plan has 150): the copy by z slabs of Z output planes (a multiple of 4: conv_fwd_f4v_k's tiles), each slab's
   input planes [z0 - 1, z1 + 1) prenormed into the budget, then conv_fwd_f4v_k on the output z-window [z0, z1) with that slab as a z-range
   input (zi0 = -(z0 - 1), Di = its planes). The skip mask is taken once for the whole conv. Each output tile and its statistics are
   those of the unslabbed conv (the tiles do not change; the statistics add in fixed point): the same bits as the in-kernel GroupNorm
   staging (test_f4v) at a copy pass of the input instead of the staging's per-tile tanh (conv_fwd_f4v_k's G staging is not overlapped
   with its MMAs: ncu L0 50 ms vs 29 ms copy-staged) */
int lp_f4v_preslab(const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, int Cip, gnp_t gp, double *osum, int Go, const split_t &sp) {
#ifdef LP_SW
    (void)x; (void)xs; (void)wq; (void)ws; (void)b; (void)cout; (void)y; (void)Cop; (void)Cip; (void)gp; (void)osum; (void)Go; (void)sp;
    return 0;
#else
    if (!lp_get_f4v() || lp_sw_build() || Cop % 64 || xs.d < 8 || sp.zw1 || sp.Di || sp.y2 || (osum && ((xs.w & 15) || cout <= 16))) return 0;
    if (g_f4v_preslab < 0) g_f4v_preslab = getenv("UFSM_F4V_PRESLAB") ? atoi(getenv("UFSM_F4V_PRESLAB")) : 0;
    const int HW = xs.h * xs.w, nb = xs.c / 32;
    const size_t pp = (size_t)xs.n * nb * HW * 17;   /* prenorm bytes per input plane */
    if ((size_t)(xs.d + 2) * HW >= ((size_t)1 << 31)) return 0;
    int Z = 0; uint8_t *q = nullptr;
    if (g_f4v_preslab > 0) { Z = g_f4v_preslab & ~3; if (Z < 4) Z = 4; q = (uint8_t *)nn_ws(NN_WS_F4PRE, (size_t)(Z + 2) * pp); }
    else for (Z = (xs.d + 3) & ~3; Z >= 8 && !(q = (uint8_t *)nn_ws(NN_WS_F4PRE, (size_t)(Z + 2) * pp)); Z = Z > 64 ? 64 : Z - 4) {}   /* the largest slab the budget holds */
    if (!q) return 0;
    const skipm_t sk = lp_skip_get(xs, sp, 1);
    for (int z0 = 0; z0 < xs.d; z0 += Z) {
        const int z1 = z0 + Z < xs.d ? z0 + Z : xs.d, a = z0 > 0 ? z0 - 1 : 0, e = z1 < xs.d ? z1 + 1 : xs.d, L = e - a;
        gn_prenorm_f4z_k<<<dim3(nblk_((size_t)L * HW, 256), (unsigned)nb, (unsigned)xs.n), 256>>>((const mx4_t *)x, xs.n, xs.c, xs.c, xs.c, 0, (size_t)xs.d * HW, a, L, HW, gp, split_t{}, q);
        split_t s2 = sp; s2.zi0 = (int16_t)-a; s2.Di = (int16_t)L; s2.zw0 = (int16_t)z0; s2.zw1 = (int16_t)z1;
        launch_f4v<4, 4, mx4_t>(q, xs, wq, ws, b, cout, y, Cop, Cip, gnp_t{}, osum, Go, s2, sk, z0 == 0);
    }
    __atomic_fetch_add(&g_f4v_calls, 1ul, __ATOMIC_RELAXED);
    return 1;
#endif
}

#ifndef LP_SW
/* the fused upsample's coarse rows (f4ru_coarse per coarse voxel: decode, the coarse GroupNorm + SiLU, e4m3 requantisation) of coarse planes
   [cza, cza + Lc): [N][nb1][Lc HWc] x 32 B, then the scale bytes */
__global__ void __launch_bounds__(256) up_coarse_f4v_k(const mx4_t *__restrict__ x, int N, int Ci, int Cx, size_t cpl, int cza, int Lc, int HWc, gnp_t gp, split_t sp, uint8_t *__restrict__ cu) {
    __shared__ chan_t ctab[32]; __shared__ float2 cab[32]; __shared__ unsigned sgm;
    const int blk = blockIdx.y, n = blockIdx.z, nb1 = gridDim.y, CxP = (Cx + 31) / 32 * 32;
    if (threadIdx.x < 32) {
        const chan_t c = make_chan(seg_ci(blk * 32 + (int)threadIdx.x, Ci, Cx, CxP), Ci, Cx, n, cpl, x, sp, gp, N);
        ctab[threadIdx.x] = c; cab[threadIdx.x] = make_float2(c.a, c.b);
        const unsigned gm = __ballot_sync(0xffffffffu, c.g != 0);
        if (threadIdx.x == 0) sgm = gm;
    }
    __syncthreads();
    const size_t Ls = (size_t)Lc * HWc, i = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (i >= Ls) return;
    const chan_t c0 = ctab[0];
    const unsigned gmask = sgm;
    float r[32];
    const bool ok = c0.p != nullptr;
    mx_row32<4>(c0, ok ? (size_t)cza * HWc + i : 0, ok, r);
    if (gmask && ok) {
#pragma unroll
        for (int k = 0; k < 32; k++) if ((gmask >> k) & 1u) { const float2 ab = cab[k]; r[k] = act_ab(r[k], ab.x, ab.y, true); }
    }
    unsigned am = 0u;
#pragma unroll
    for (int k = 0; k < 32; k++) { const unsigned u = __float_as_uint(fabsf(r[k])); am = u <= 0x7f800000u ? max(am, u) : am; }
    const int e = mx_exp(__uint_as_float(am), 1.f / 448.f);
    const float m = exp2i(-e);
    const size_t ri = ((size_t)n * nb1 + blk) * Ls + i;
    uint4 *dst = (uint4 *)(cu + ri * 32);
    dst[0] = make_uint4(cvt_e4m3x4(r[0] * m, r[1] * m, r[2] * m, r[3] * m), cvt_e4m3x4(r[4] * m, r[5] * m, r[6] * m, r[7] * m),
                        cvt_e4m3x4(r[8] * m, r[9] * m, r[10] * m, r[11] * m), cvt_e4m3x4(r[12] * m, r[13] * m, r[14] * m, r[15] * m));
    dst[1] = make_uint4(cvt_e4m3x4(r[16] * m, r[17] * m, r[18] * m, r[19] * m), cvt_e4m3x4(r[20] * m, r[21] * m, r[22] * m, r[23] * m),
                        cvt_e4m3x4(r[24] * m, r[25] * m, r[26] * m, r[27] * m), cvt_e4m3x4(r[28] * m, r[29] * m, r[30] * m, r[31] * m));
    cu[(size_t)N * nb1 * Ls * 32 + ri] = (uint8_t)(e + 127);
}
/* the up segment at fine planes [a, a + L): f4ru_blend's trilinear blend of the coarse rows (the same taps, weights, half2 sums and order),
   requantised to e2m1 per (32-channel block, voxel) as conv_fwd_f4_k's staging: a depth-L MX-fp4 tensor of Cx channels */
__global__ void __launch_bounds__(256) up_fine_f4v_k(const uint8_t *__restrict__ cu, int N, int cza, int Lc, int D, int H, int W, int a, int L, uint8_t *__restrict__ y) {
    const int blk = blockIdx.y, n = blockIdx.z, nb1 = gridDim.y, HW = H * W, Dc = D >> 1, Hc = H >> 1, Wc = W >> 1;
    const size_t Ls = (size_t)L * HW, v0 = (size_t)blockIdx.x * 256 + threadIdx.x, Lcs = (size_t)Lc * Hc * Wc;
    if (v0 >= Ls) return;
    const int gz = a + (int)(v0 / HW), gy = (int)(v0 % HW) / W, gx = (int)(v0 % W);
    const uint8_t *cb = cu + ((size_t)n * nb1 + blk) * Lcs * 32, *cbs = cu + (size_t)N * nb1 * Lcs * 32 + ((size_t)n * nb1 + blk) * Lcs;
    const int mz[2] = {gz >> 1, min(max((gz >> 1) + ((gz & 1) ? 1 : -1), 0), Dc - 1)}, my[2] = {gy >> 1, min(max((gy >> 1) + ((gy & 1) ? 1 : -1), 0), Hc - 1)},
              mx[2] = {gx >> 1, min(max((gx >> 1) + ((gx & 1) ? 1 : -1), 0), Wc - 1)};
    int lis[8]; unsigned em = 0u;
#pragma unroll
    for (int t = 0; t < 8; t++) { const int a2 = t >> 2, b2 = (t >> 1) & 1, c2 = t & 1; lis[t] = ((mz[a2] - cza) * Hc + my[b2]) * Wc + mx[c2]; em = max(em, (unsigned)cbs[lis[t]]); }
    __half2 acc[16];
#pragma unroll
    for (int j = 0; j < 16; j++) acc[j] = __float2half2_rn(0.f);
#pragma unroll
    for (int a2 = 0; a2 < 2; a2++)
#pragma unroll
        for (int b2 = 0; b2 < 2; b2++)
#pragma unroll
            for (int c2 = 0; c2 < 2; c2++) {
                const int li = lis[(a2 * 2 + b2) * 2 + c2];
                const int de = (int)cbs[li] - (int)em;
                const __half2 w3 = __float2half2_rn((a2 ? 0.25f : 0.75f) * (b2 ? 0.25f : 0.75f) * (c2 ? 0.25f : 0.75f) * (de >= -126 ? __uint_as_float((unsigned)(127 + de) << 23) : 0.f));
                const uint4 h0 = *(const uint4 *)(cb + (size_t)li * 32), h1 = *(const uint4 *)(cb + (size_t)li * 32 + 16);
                const unsigned w[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    unsigned lo, hi;
                    lo = h2_e4m3x2(w[j] & 0xffff);
                    hi = h2_e4m3x2(w[j] >> 16);
                    acc[2 * j] = __hfma2(w3, *(const __half2 *)&lo, acc[2 * j]); acc[2 * j + 1] = __hfma2(w3, *(const __half2 *)&hi, acc[2 * j + 1]);
                }
            }
    const float sm = mx_scale(em);
    float val[32];
#pragma unroll
    for (int j = 0; j < 16; j++) { const float2 f = __half22float2(acc[j]); val[2 * j] = f.x * sm; val[2 * j + 1] = f.y * sm; }
    unsigned amu = 0u;
#pragma unroll
    for (int k = 0; k < 32; k++) amu = amax_u(amu, val[k]);
    const int e = mx_exp(__uint_as_float(amu), 1.f / 6.f);
    const float m = exp2i(-e);
    const size_t ri = ((size_t)n * nb1 + blk) * Ls + v0;
    *(uint4 *)(y + ri * 16) = make_uint4(cvt_e2m1x8(val, m), cvt_e2m1x8(val + 8, m), cvt_e2m1x8(val + 16, m), cvt_e2m1x8(val + 24, m));
    y[(size_t)N * nb1 * Ls * 16 + ri] = (uint8_t)(e + 127);
}
#endif
/* fwd_f4_t's decoder conv1 with the fused 2x upsample (sp.up 2: [up2(x) | x2], per-segment GroupNorms): conv_fwd_f4_k staged the up
   chunks (f4ru_coarse + f4ru_blend: 8 coarse rows a position, per output-channel tile and halo, 4 x 2.8x over) and GN-staged the skip
   chunks. Here, by z slabs (as lp_f4v_preslab): the slab's coarse rows once (up_coarse_f4v_k), its up part at full resolution
   (up_fine_f4v_k), its skip part prenormed (gn_prenorm_f4z_k on the second segment), then conv_fwd_f4v_k copy-staged on the two slab
   tensors as a plain split input. Per voxel the staged bytes are conv_fwd_f4_k's (the same functions' arithmetic): the same outputs
   and statistics bits (test_f4v) */
int lp_f4v_preup(const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, int Cip, gnp_t gp, double *osum, int Go, const split_t &sp) {
#ifdef LP_SW
    (void)x; (void)xs; (void)wq; (void)ws; (void)b; (void)cout; (void)y; (void)Cop; (void)Cip; (void)gp; (void)osum; (void)Go; (void)sp;
    return 0;
#else
    const int Ci = xs.c, Cx = sp.c_split, CxP = (Cx + 31) / 32 * 32;
    if (!lp_get_f4v() || lp_sw_build() || sp.up != 2 || !sp.x2 || Cx <= 16 || Ci - Cx <= 16 || sp.sr || gp.ns || sp.gp2.ns || sp.Di || sp.zw1 || sp.y2 || Cop % 64 || xs.d < 8
        || (osum && ((xs.w & 15) || cout <= 16)) || ((size_t)(xs.d + 2) * xs.h * xs.w >= ((size_t)1 << 31))) return 0;
    if (g_f4v_preslab < 0) g_f4v_preslab = getenv("UFSM_F4V_PRESLAB") ? atoi(getenv("UFSM_F4V_PRESLAB")) : 0;
    const int HW = xs.h * xs.w, Hc = xs.h >> 1, Wc = xs.w >> 1, HWc = Hc * Wc, Dc = xs.d >> 1, nb1 = CxP / 32, nb2 = (Cip - CxP) / 32;
    auto bytes = [&](int Z, size_t *o1, size_t *o2) {   /* (fine up | fine skip | coarse) for a Z-plane slab */
        const size_t L = (size_t)Z + 2, Lc = L / 2 + 3;
        *o1 = ((size_t)xs.n * nb1 * L * HW * 17 + 255) & ~(size_t)255;
        *o2 = *o1 + (((size_t)xs.n * nb2 * L * HW * 17 + 255) & ~(size_t)255);
        return *o2 + (size_t)xs.n * nb1 * Lc * HWc * 33;
    };
    size_t o1 = 0, o2 = 0;
    int Z = 0; uint8_t *q = nullptr;
    if (g_f4v_preslab > 0) { Z = g_f4v_preslab & ~3; if (Z < 4) Z = 4; q = (uint8_t *)nn_ws(NN_WS_F4PRE, bytes(Z, &o1, &o2)); }
    else for (Z = (xs.d + 3) & ~3; Z >= 4 && !(q = (uint8_t *)nn_ws(NN_WS_F4PRE, bytes(Z, &o1, &o2))); Z = Z > 64 ? 64 : Z - 4) {}
    if (!q) return 0;
    bytes(Z, &o1, &o2);
    const skipm_t sk = lp_skip_get(xs, sp, 1);
    const size_t cpl = (size_t)Dc * HWc;
    for (int z0 = 0; z0 < xs.d; z0 += Z) {
        const int z1 = z0 + Z < xs.d ? z0 + Z : xs.d, a = z0 > 0 ? z0 - 1 : 0, e = z1 < xs.d ? z1 + 1 : xs.d, L = e - a;
        const int cza = max((a >> 1) - 1, 0), czb = min(((e - 1) >> 1) + 2, Dc), Lc = czb - cza;   /* the coarse planes the blend reads */
        uint8_t *qu = q, *qs = q + o1, *qc = q + o2;
        up_coarse_f4v_k<<<dim3(nblk_((size_t)Lc * HWc, 256), (unsigned)nb1, (unsigned)xs.n), 256>>>((const mx4_t *)x, xs.n, Ci, Cx, cpl, cza, Lc, HWc, gp, sp, qc);
        up_fine_f4v_k<<<dim3(nblk_((size_t)L * HW, 256), (unsigned)nb1, (unsigned)xs.n), 256>>>(qc, xs.n, cza, Lc, xs.d, xs.h, xs.w, a, L, qu);
        gn_prenorm_f4z_k<<<dim3(nblk_((size_t)L * HW, 256), (unsigned)nb2, (unsigned)xs.n), 256>>>((const mx4_t *)x, xs.n, Ci, Cx, CxP, CxP, (size_t)xs.d * HW, a, L, HW, gp, sp, qs);
        split_t s2 = sp; s2.up = 0; s2.x2 = qs; s2.gp2 = gnp_t{}; s2.zi0 = (int16_t)-a; s2.Di = (int16_t)L; s2.zw0 = (int16_t)z0; s2.zw1 = (int16_t)z1;
        launch_f4v<4, 4, mx4_t>(qu, xs, wq, ws, b, cout, y, Cop, Cip, gnp_t{}, osum, Go, s2, sk, z0 == 0);
    }
    __atomic_fetch_add(&g_f4v_calls, 1ul, __ATOMIC_RELAXED);
    return 1;
#endif
}
