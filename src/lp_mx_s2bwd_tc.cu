/* MX stride-2 backward-data of the narrow down convs (16 or 32 channels, Ci == Co) on the tensor cores */
#include "lp_mxops.cuh"
/* gx[ci][u] (+)= sum over the taps k with (u + 1 - k) even, o = (u + 1 - k) / 2 in range, of sum_co w[co][ci][k] gy[co][o]
   (k = 3, pad 1). The fine voxels split into 8 parity classes p (u = 2 m + p per axis); within a class every voxel uses
   the same 1..8 taps (even coordinate: k = 1 at o = m; odd: k = 0 at o = m + 1 and k = 2 at o = m), so a class is a GEMM
   M = ci, N = voxels, K = taps x co: bf16 m16n8k16 with fp32 accumulation (A = w^T of the tap, staged per class; B = gy).
   A block takes TMZ x TMY x TMX class positions m (all 8 classes), decodes gy at o in [m, m + 1] once into smem as bf16
   (MX-fp8 values are exact in bf16; zero outside the coarse grid) and writes each fine voxel's MX-fp8 row once (decoded,
   added to the existing row when accumulating, re-encoded). Differs from the fp32 parity kernel by the bf16 weights
   (2^-9 relative, below the e4m3 output rounding) and the summation order. */
/* the weights once per call as bf16 [27 taps][C ci][C + 8 co] (the kernels' per-class A tiles are then contiguous 16-byte copies) */
template <int C> __global__ void s2tc_prep_w_k(const float *__restrict__ w, __nv_bfloat16 *__restrict__ wtg) {
    constexpr int GROW = C + 8;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 27 * C * GROW) return;
    const int co = i % GROW, ci = (i / GROW) % C, tap = i / (GROW * C);
    wtg[i] = __float2bfloat16_rn(co < C ? w[((size_t)co * C + ci) * 27 + tap] : 0.f);
}
/* C = 64 (two 32-channel MX blocks): a block computes one 32-channel block of gx (blockIdx.y; M = 32, K = all 64 gy channels),
   so its A tiles stay 32 x 72; the dilated fp8 path it replaces needed two fine-level MX-fp8 scratch tensors (~1 GB at P 512)
   and 8x the MACs. GB / XB: MX bits of gy / gx (4: MX-fp4 gradients, gx re-encoded with exact SR keyed by osr) */
template <int C, int GB = 8, int XB = 8>
__global__ void __launch_bounds__(256) bwd_data_s2_tc_k(const uint8_t *__restrict__ gy, const __nv_bfloat16 *__restrict__ wtg, uint8_t *__restrict__ gx,
                                                     int N, int D, int H, int W, int Do, int Ho, int Wo, int accum, unsigned osr) {
    constexpr int CB = C > 32 ? 32 : C, BW = C > 32 ? 32 : C, NB = C / BW;   /* gx channels per block; MX block width and count */
    constexpr int RBY = BW * GB / 8, RBX = BW * XB / 8;                      /* row bytes of gy / gx */
    constexpr int TMZ = 2, TMY = C > 32 ? 4 : 8, TMX = C == 16 ? 16 : 8;   /* class positions per block */
    constexpr int GZ = TMZ + 1, GY = TMY + 1, GX = TMX + 1, GROW = C + 8;   /* gy tile, rows padded by 8 bf16 (bank spread) */
    constexpr int MT = CB / 16, KS = C / 16, NTW = TMZ * TMY * (TMX / 8) / 8;   /* M tiles, k-steps, n-tiles per warp */
    static_assert(XB == 8 || CB / 4 == 8 || CB / 4 == 4, "MX-fp4 gx: 8 (32 channels) or 4 (16) channels per lane");
    extern __shared__ __align__(16) unsigned char s2t_smem[];
    __nv_bfloat16 *gyt = (__nv_bfloat16 *)s2t_smem;                       /* [GZ][GY][GX][GROW] */
    __nv_bfloat16 *wt = gyt + GZ * GY * GX * GROW;                          /* [8 taps][CB ci][GROW co] */
    float *ob = (float *)(wt + 8 * CB * GROW);                              /* [8 warps][8 voxels][CB] */
    const int cb = blockIdx.y, ci0 = cb * CB;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int Mz = (D + 1) / 2, My = (H + 1) / 2, Mx = (W + 1) / 2;
    const int ntx = (Mx + TMX - 1) / TMX, nty = (My + TMY - 1) / TMY, ntz = (Mz + TMZ - 1) / TMZ;
    int bt = blockIdx.x;
    const int mx0 = (bt % ntx) * TMX; bt /= ntx;
    const int my0 = (bt % nty) * TMY; bt /= nty;
    const int mz0 = (bt % ntz) * TMZ; const int n = bt / ntz;
    const size_t So = (size_t)Do * Ho * Wo, S = (size_t)D * H * W;
    const uint8_t *gsc = gy + (size_t)N * NB * So * RBY;
    uint8_t *xsc = gx + (size_t)N * NB * S * RBX;
    for (int i = threadIdx.x; i < GZ * GY * GX; i += blockDim.x) {
        const int tx = i % GX, ty = (i / GX) % GY, tz = i / (GX * GY);
        const int oz = mz0 + tz, oy = my0 + ty, ox = mx0 + tx;
        float r[32];
#pragma unroll
        for (int k = 0; k < 32; k++) r[k] = 0.f;
        __nv_bfloat16 *d = gyt + (size_t)i * GROW;
#pragma unroll
        for (int b = 0; b < NB; b++) {
            if (b) {
#pragma unroll
                for (int k = 0; k < 32; k++) r[k] = 0.f;
            }
            if (oz < Do && oy < Ho && ox < Wo) {
                const size_t ri = ((size_t)n * NB + b) * So + ((size_t)oz * Ho + oy) * Wo + ox;
                mxf<GB>::dec_row(gy + ri * RBY, BW, mx_scale(gsc[ri]), r);
            }
#pragma unroll
            for (int k = 0; k < BW; k += 2) *(__nv_bfloat162 *)(d + b * BW + k) = __floats2bfloat162_rn(r[k], r[k + 1]);
        }
    }
    float *wob = ob + warp * 8 * CB;
#pragma unroll 1
    for (int p = 0; p < 8; p++) {
        const int pz = p >> 2, py = (p >> 1) & 1, px = p & 1;
        const int nz = pz ? 2 : 1, ny = py ? 2 : 1, nx = px ? 2 : 1, ntap = nz * ny * nx;
        __syncthreads();   /* the previous class's A tiles are consumed (and, for p = 0, the gy tile is written) */
        constexpr int TQ = CB * GROW / 8;   /* uint4 per tap tile */
        for (int i = threadIdx.x; i < ntap * TQ; i += blockDim.x) {
            const int tl = i / TQ, j = i % TQ;
            const int a = tl / (ny * nx), bb = (tl / nx) % ny, c = tl % nx;
            const int kz = pz ? (a ? 2 : 0) : 1, ky = py ? (bb ? 2 : 0) : 1, kx = px ? (c ? 2 : 0) : 1;
            ((uint4 *)(wt + (size_t)tl * CB * GROW))[j] = __ldg((const uint4 *)(wtg + ((size_t)((kz * 3 + ky) * 3 + kx) * C + ci0) * GROW) + j);
        }
        __syncthreads();
#pragma unroll 1
        for (int nt = 0; nt < NTW; nt++) {
            const int tile = warp * NTW + nt, row = tile / (TMX / 8), xh = tile % (TMX / 8), rz = row / TMY, ry = row % TMY;
            /* this lane's slice of the old gx row (accumulating), loaded before the MMA so its latency is hidden */
            constexpr int CQ = CB / 4;
            const int vl = lane >> 2, part = lane & 3;
            const int uz = 2 * (mz0 + rz) + pz, uy = 2 * (my0 + ry) + py, ux = 2 * (mx0 + 8 * xh + vl) + px;
            const bool ok = uz < D && uy < H && ux < W;
            const size_t ri = ok ? ((size_t)n * NB + cb) * S + ((size_t)uz * H + uy) * W + ux : 0;
            unsigned wd[CQ / 4] = {}, osc = 0u;
            if (accum && ok) {
                if constexpr (XB == 4 && CQ == 4) wd[0] = *(const unsigned short *)(gx + ri * RBX + part * CQ / 2);   /* 4 nibbles (16 channels) */
                else if constexpr (XB == 4) wd[0] = *(const unsigned *)(gx + ri * RBX + part * CQ / 2);   /* 8 nibbles */
                else if constexpr (CQ == 8) { const uint2 u = *(const uint2 *)(gx + ri * RBX + part * CQ); wd[0] = u.x; wd[1] = u.y; }
                else wd[0] = *(const unsigned *)(gx + ri * RBX + part * CQ);
                osc = xsc[ri];   /* plain loads: this thread rewrites the row below */
            }
            float acc[MT][4];
#pragma unroll
            for (int m = 0; m < MT; m++) acc[m][0] = acc[m][1] = acc[m][2] = acc[m][3] = 0.f;
            for (int tl = 0; tl < ntap; tl++) {
                const int a = tl / (ny * nx), bb = (tl / nx) % ny, c = tl % nx;
                const int dz = pz && !a ? 1 : 0, dy = py && !bb ? 1 : 0, dx = px && !c ? 1 : 0;
                /* B: k = co, n = voxel g of the tile; the gy row of class position (rz, ry, 8 xh + g) shifted by d */
                const __nv_bfloat16 *brow = gyt + ((size_t)((rz + dz) * GY + ry + dy) * GX + 8 * xh + g + dx) * GROW;
#pragma unroll
                for (int ks = 0; ks < KS; ks++) {
                    unsigned b[2] = {*(const unsigned *)(brow + 16 * ks + 2 * t), *(const unsigned *)(brow + 16 * ks + 2 * t + 8)};
#pragma unroll
                    for (int m = 0; m < MT; m++) {
                        const __nv_bfloat16 *arow = wt + ((size_t)tl * CB + 16 * m + g) * GROW + 16 * ks + 2 * t;
                        unsigned af[4] = {*(const unsigned *)arow, *(const unsigned *)(arow + 8 * GROW), *(const unsigned *)(arow + 8), *(const unsigned *)(arow + 8 * GROW + 8)};
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                                     : "+f"(acc[m][0]), "+f"(acc[m][1]), "+f"(acc[m][2]), "+f"(acc[m][3])
                                     : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(b[0]), "r"(b[1]));
                    }
                }
            }
            /* C[ci][voxel]: c0, c1 = (ci 16 m + g, voxels 2 t, 2 t + 1), c2, c3 = (ci 16 m + g + 8, same voxels) */
#pragma unroll
            for (int m = 0; m < MT; m++) {
                wob[(2 * t) * CB + 16 * m + g] = acc[m][0]; wob[(2 * t + 1) * CB + 16 * m + g] = acc[m][1];
                wob[(2 * t) * CB + 16 * m + g + 8] = acc[m][2]; wob[(2 * t + 1) * CB + 16 * m + g + 8] = acc[m][3];
            }
            __syncwarp();
            {   /* 4 lanes per voxel (lane >> 2), C / 4 channels each: add the old row (accumulating), block amax by shuffles,
                   encode (the same per-value conversions as mxf<8>::enc_row) */
                float v[CQ];
#pragma unroll
                for (int k = 0; k < CQ; k++) v[k] = wob[vl * CB + part * CQ + k];
                if (accum && ok) {
                    const float sc = mx_scale(osc);
                    if constexpr (XB == 4) {
#pragma unroll
                        for (int q = 0; q < CQ / 2; q++) { const float2 d = dec_e2m1x2(wd[0] >> (8 * q)); v[2 * q] += d.x * sc; v[2 * q + 1] += d.y * sc; }
                    } else {
#pragma unroll
                    for (int q = 0; q < CQ / 4; q++) {
                        const float2 d0 = dec_e4m3x2((unsigned short)(wd[q] & 0xffffu)), d1 = dec_e4m3x2((unsigned short)(wd[q] >> 16));
                        v[4 * q] += d0.x * sc; v[4 * q + 1] += d0.y * sc; v[4 * q + 2] += d1.x * sc; v[4 * q + 3] += d1.y * sc;
                    }
                    }
                }
                unsigned am = 0u;
#pragma unroll
                for (int k = 0; k < CQ; k++) am = amax_u(am, v[k]);
                am = max(am, __shfl_xor_sync(0xffffffffu, am, 1)); am = max(am, __shfl_xor_sync(0xffffffffu, am, 2));
                const int e = mx_exp(__uint_as_float(am), mxf<XB>::inv_qmax);
                const float m = exp2i(-e);
                if (ok) {
                    if constexpr (XB == 4 && CQ == 4) {   /* 16 channels: 4 nibbles per lane, SR keyed as the 32-channel words */
                        unsigned q4 = 0u;
                        if (osr) {
                            uint32_t hh[4]; sr_hash4(osr, (uint64_t)ri * 4 + part, hh);
#pragma unroll
                            for (int j = 0; j < 4; j++) q4 |= sr_e2m1_nib(v[j] * m, (hh[j >> 1] >> (16 * (j & 1))) & 0xffffu) << (4 * j);
                        } else { const float v8[8] = {v[0], v[1], v[2], v[3], 0.f, 0.f, 0.f, 0.f}; q4 = cvt_e2m1x8(v8, m) & 0xffffu; }
                        *(unsigned short *)(gx + ri * RBX + part * CQ / 2) = (unsigned short)q4;
                    } else if constexpr (XB == 4) {
                        unsigned q4;
                        if (osr) { uint32_t hh[4]; sr_hash4(osr, (uint64_t)ri * 4 + part, hh); q4 = sr_e2m1_word(v, m, hh); }
                        else q4 = cvt_e2m1x8(v, m);
                        *(unsigned *)(gx + ri * RBX + part * CQ / 2) = q4;
                    }
                    else if constexpr (CQ == 8) *(uint2 *)(gx + ri * RBX + part * CQ) = make_uint2(mxf<8>::enc8x(v, m), mxf<8>::enc8x(v + 4, m));
                    else *(unsigned *)(gx + ri * RBX + part * CQ) = mxf<8>::enc8x(v, m);
                    if (part == 0) xsc[ri] = (uint8_t)(e + 127);
                }
            }
            __syncwarp();   /* the warp's output buffer is reused by the next tile */
        }
    }
}
/* ======================= FP4 stride-2 backward-data for 32 channels (MX-fp4 gy and gx: down0) =======================
   The parity classes of bwd_data_s2_tc_k on e2m1 m16n8k64 MMAs: class p (u = 2 m + p per axis) is a stride-1 conv on the coarse grid
   with taps k = 1 at offset 0 (even axis) or k = 0 at +1 and k = 2 at 0 (odd axis); its 1 .. 8 taps go in pairs (K = 2 taps x 32
   channels; 14 pairs over the 8 classes, an odd class's last pair padded with prep's zero tap 27). The coarse gy rows are staged as
   stored (MX-fp4: exact, no requantisation), the 14 pair tiles of the transposed weights (M = ci, K = co) are resident. Block: class
   positions 2 z x 4 y x 16 x (all 8 classes), walked over ZC z tiles; warp (mz, my) = 16 positions along x (two n-tiles per A
   fragment). Each class's tile goes through a per-warp buffer to 4 lanes per fine voxel (8 channels each), added to the old row
   (accum) and stored with exact SR keyed by (osr, row, word) as bwd_data_s2_tc_k. */
struct s2f4_pairs { signed char ta[27], tb[27], da[27][3], db[27][3], first[9]; };   /* K units: tap pairs (32 channels, 14) or taps (64, 27) */
__global__ void s2f4_wt_k(const float *__restrict__ w, float *__restrict__ wt, int C) {   /* wt[ci][co][27] = w[co][ci][27] */
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C * C * 27) return;
    const int tap = i % 27, co = (i / 27) % C, ci = i / (27 * C);
    wt[i] = w[((size_t)co * C + ci) * 27 + tap];
}
__global__ void prep_w4_k(const float *__restrict__ w, uint8_t *__restrict__ wq, uint8_t *__restrict__ ws, int Co, int Ci, int Cop, int Cip, int Cx, int CxP, int Ox, int OxP);   /* lp_f4fwd.cu */
/* NCH 2 (64 channels: down1): a K unit is one tap's 64 gy channels (the two 32-channel blocks: halves of the A rows / B
   fragments), 27 units; a block computes one 32-channel half hb of gx (blockIdx.z = (n nzc + zc) 2 + hb), i.e. one MX block */
template <int XB, int NCH = 1>
__global__ void __launch_bounds__(256, 2) bwd_data_s2_f4_k(const uint8_t *__restrict__ gy, const uint8_t *__restrict__ wq, const uint8_t *__restrict__ wsc, uint8_t *__restrict__ gx,
                                                        int N, int D, int H, int W, int Do, int Ho, int Wo, int accum, unsigned osr, s2f4_pairs PT, int ZC) {
    constexpr int TMZ = 2, TMY = 4, TMX = 16, GZ = TMZ + 1, GY = TMY + 1, GX = TMX + 1, NG = GZ * GY * GX, BM = 32, NU = NCH == 1 ? 14 : 27, C = 32 * NCH;
    constexpr int RBX = 32 * XB / 8;
    extern __shared__ __align__(128) unsigned char s2f4_smem[];
    uint8_t *gyt = s2f4_smem;                                   /* [NG][NCH][16 B] */
    uint8_t *gys = gyt + NG * 16 * NCH;                         /* [NG][NCH] */
    uint8_t *wp = gys + ((NG * NCH + 127) & ~127);              /* [NU][32 rows][32 B] swizzled */
    unsigned short *wps = (unsigned short *)(wp + NU * BM * 32);   /* [NU][32] */
    float *ob = (float *)(wps + NU * BM);                       /* [8 warps][8 voxels][36] */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int mzw = warp >> 2, myw = warp & 3;
    const int mx0 = blockIdx.x * TMX, my0 = blockIdx.y * TMY;
    const int hb = blockIdx.z % NCH, bz = blockIdx.z / NCH;
    const int nzt = (Do + TMZ - 1) / TMZ, nzc = (nzt + ZC - 1) / ZC, zc = bz % nzc, n = bz / nzc;
    const size_t So = (size_t)Do * Ho * Wo, S = (size_t)D * H * W;
    const uint8_t *gsc = gy + (size_t)N * NCH * So * 16;
    uint8_t *xsc = gx + (size_t)N * NCH * S * RBX;
    for (int i = threadIdx.x; i < NU * BM * 2; i += 256) {
        const int q = i / (BM * 2), r = i % (BM * 2), c = r >> 1, h = r & 1;
        const uint8_t *src = NCH == 1 ? wq + ((size_t)(h ? PT.tb[q] : PT.ta[q]) * BM + c) * 16 : wq + ((size_t)PT.ta[q] * C + 32 * hb + c) * 32 + 16 * h;
        *(uint4 *)(wp + q * BM * 32 + sw16(c, h)) = __ldg((const uint4 *)src);
    }
    for (int i = threadIdx.x; i < NU * BM; i += 256) {
        const int q = i / BM, c = i % BM;
        if (NCH == 1) wps[i] = (unsigned short)(wsc[(size_t)PT.ta[q] * BM + c] | (wsc[(size_t)PT.tb[q] * BM + c] << 8));
        else { const size_t k = ((size_t)PT.ta[q] * C + 32 * hb + c) * 2; wps[i] = (unsigned short)(wsc[k] | (wsc[k + 1] << 8)); }
    }
    float *wob = ob + warp * 8 * 36;
    for (int tz = zc * ZC; tz < nzt && tz < (zc + 1) * ZC; tz++) {
        const int mz0 = tz * TMZ;
        __syncthreads();   /* the weights (first tile) / the previous tile's reads of the gy tile */
        for (int i = threadIdx.x; i < NG; i += 256) {
            const int lx = i % GX, ly = (i / GX) % GY, lz = i / (GX * GY), oz = mz0 + lz, oy = my0 + ly, ox = mx0 + lx;
            const bool in = oz < Do && oy < Ho && ox < Wo;
#pragma unroll
            for (int c = 0; c < NCH; c++) {   /* the gy rows as stored (MX-fp4: exact) */
                uint4 h0 = make_uint4(0u, 0u, 0u, 0u); unsigned sc = 1u;
                if (in) { const size_t ri = ((size_t)n * NCH + c) * So + ((size_t)oz * Ho + oy) * Wo + ox; h0 = __ldg((const uint4 *)(gy + ri * 16)); sc = gsc[ri]; }
                *(uint4 *)(gyt + (i * NCH + c) * 16) = h0; gys[i * NCH + c] = (uint8_t)sc;
            }
        }
        __syncthreads();
#pragma unroll 1
        for (int p = 0; p < 8; p++) {
            const int pz = p >> 2, py = (p >> 1) & 1, px = p & 1;
            float acc[2][2][4];
#pragma unroll
            for (int m = 0; m < 2; m++) for (int q = 0; q < 2; q++) for (int k = 0; k < 4; k++) acc[m][q][k] = 0.f;
            const int mat = lane >> 3, isB = mat & 1, qn = mat >> 1, v8 = lane & 7;
            for (int q = PT.first[p]; q < PT.first[p + 1]; q++) {
                auto gpos = [&](int b_, int o) { const signed char *d = b_ ? PT.db[q] : PT.da[q]; return ((mzw + d[0]) * GY + myw + d[1]) * GX + d[2] + o; };
                unsigned bfr[4], sb[2];
                if (NCH == 1) {   /* halves: taps A / B */
                    ldsm_x4(bfr, gyt + gpos(isB, 8 * qn + v8) * 16);
#pragma unroll
                    for (int q2 = 0; q2 < 2; q2++) sb[q2] = (unsigned)gys[gpos(0, 8 * q2 + g)] | ((unsigned)gys[gpos(1, 8 * q2 + g)] << 8);
                } else {          /* halves: the tap's channel blocks 0 / 1 */
                    ldsm_x4(bfr, gyt + (gpos(0, 8 * qn + v8) * NCH + isB) * 16);
#pragma unroll
                    for (int q2 = 0; q2 < 2; q2++) { const int pp = gpos(0, 8 * q2 + g) * NCH; sb[q2] = (unsigned)gys[pp] | ((unsigned)gys[pp + 1] << 8); }
                }
#pragma unroll
                for (int m = 0; m < 2; m++) {
                    unsigned af[4];
                    const int row = m * 16 + (mat & 1) * 8 + (lane & 7);
                    ldsm_x4(af, wp + q * BM * 32 + sw16(row, mat >> 1));
                    const unsigned sa = wps[q * BM + m * 16 + g + 8 * (t & 1)];
                    mma_f4(acc[m][0], af, bfr, sa, sb[0]); mma_f4(acc[m][1], af, bfr + 2, sa, sb[1]);
                }
            }
#pragma unroll
            for (int q2 = 0; q2 < 2; q2++) {
                /* C[ci][voxel]: acc[m][q2][0, 1] = (ci 16 m + g, voxels 2 t, 2 t + 1), [2, 3] = ci + 8 */
#pragma unroll
                for (int m = 0; m < 2; m++) {
                    wob[(2 * t) * 36 + 16 * m + g] = acc[m][q2][0]; wob[(2 * t + 1) * 36 + 16 * m + g] = acc[m][q2][1];
                    wob[(2 * t) * 36 + 16 * m + g + 8] = acc[m][q2][2]; wob[(2 * t + 1) * 36 + 16 * m + g + 8] = acc[m][q2][3];
                }
                __syncwarp();
                {
                    const int vl = lane >> 2, part = lane & 3;
                    const int uz = 2 * (mz0 + mzw) + pz, uy = 2 * (my0 + myw) + py, ux = 2 * (mx0 + 8 * q2 + vl) + px;
                    const bool ok = uz < D && uy < H && ux < W;
                    const size_t ri = ok ? ((size_t)n * NCH + hb) * S + ((size_t)uz * H + uy) * W + ux : 0;
                    float v[8];
#pragma unroll
                    for (int k = 0; k < 8; k++) v[k] = wob[vl * 36 + part * 8 + k];
                    if (accum && ok) {
                        const float sc = mx_scale(xsc[ri]);
                        if constexpr (XB == 4) {
                            const unsigned wd = *(const unsigned *)(gx + ri * RBX + part * 4);
#pragma unroll
                            for (int q4 = 0; q4 < 4; q4++) { const float2 d = dec_e2m1x2(wd >> (8 * q4)); v[2 * q4] += d.x * sc; v[2 * q4 + 1] += d.y * sc; }
                        } else {
                            const uint2 u = *(const uint2 *)(gx + ri * RBX + part * 8); const unsigned wd[2] = {u.x, u.y};
#pragma unroll
                            for (int q4 = 0; q4 < 2; q4++) {
                                const float2 d0 = dec_e4m3x2((unsigned short)(wd[q4] & 0xffffu)), d1 = dec_e4m3x2((unsigned short)(wd[q4] >> 16));
                                v[4 * q4] += d0.x * sc; v[4 * q4 + 1] += d0.y * sc; v[4 * q4 + 2] += d1.x * sc; v[4 * q4 + 3] += d1.y * sc;
                            }
                        }
                    }
                    unsigned am = 0u;
#pragma unroll
                    for (int k = 0; k < 8; k++) am = amax_u(am, v[k]);
                    am = max(am, __shfl_xor_sync(0xffffffffu, am, 1)); am = max(am, __shfl_xor_sync(0xffffffffu, am, 2));
                    const int e = mx_exp(__uint_as_float(am), mxf<XB>::inv_qmax);
                    const float m = exp2i(-e);
                    if (ok) {
                        if constexpr (XB == 4) {
                            unsigned q4;
                            if (osr) { uint32_t hh[4]; sr_hash4(osr, (uint64_t)ri * 4 + part, hh); q4 = sr_e2m1_word(v, m, hh); }
                            else q4 = cvt_e2m1x8(v, m);
                            *(unsigned *)(gx + ri * RBX + part * 4) = q4;
                        } else *(uint2 *)(gx + ri * RBX + part * 8) = make_uint2(mxf<8>::enc8x(v, m), mxf<8>::enc8x(v + 4, m));
                        if (part == 0) xsc[ri] = (uint8_t)(e + 127);
                    }
                }
                __syncwarp();
            }
        }
    }
}
static int g_s2b_f4 = -1;
extern "C" void lp_set_s2b_f4(int on) { g_s2b_f4 = on; }
static int s2b_f4(const void *gy, shape5 ys, const float *w, shape5 xs, void *gx, int accum, int gdt, int xdt, unsigned osr) {
    if (g_s2b_f4 < 0) g_s2b_f4 = getenv("UFSM_S2B_F4") ? atoi(getenv("UFSM_S2B_F4")) : 3;
    const int C = xs.c, NCH = C / 32;
    if (!g_s2b_f4 || (C != 32 && C != 64) || ys.c != C || gdt != 4 || (xdt != 4 && xdt != 8) || xs.d != 2 * ys.d || xs.h != 2 * ys.h || xs.w != 2 * ys.w || ys.w % 16 || ys.h % 4) return 0;
    if (C == 64 && (g_s2b_f4 & 2) == 0) return 0;   /* UFSM_S2B_F4=1: 32 channels only (3, the default: also 64) */
    static s2f4_pairs PT, PT2; static int built = 0, built2 = 0;
    if (!built2) {   /* 64 channels: one K unit per tap, class by class */
        int q = 0;
        for (int p = 0; p < 8; p++) {
            PT2.first[p] = (signed char)q;
            const int pa[3] = {p >> 2, (p >> 1) & 1, p & 1};
            for (int a = 0; a < (pa[0] ? 2 : 1); a++) for (int bb = 0; bb < (pa[1] ? 2 : 1); bb++) for (int c = 0; c < (pa[2] ? 2 : 1); c++, q++) {
                const int sel[3] = {a, bb, c}; int k3[3];
                for (int ax = 0; ax < 3; ax++) { k3[ax] = !pa[ax] ? 1 : !sel[ax] ? 0 : 2; PT2.da[q][ax] = PT2.db[q][ax] = (signed char)(pa[ax] && !sel[ax]); }
                PT2.ta[q] = PT2.tb[q] = (signed char)((k3[0] * 3 + k3[1]) * 3 + k3[2]);
            }
        }
        PT2.first[8] = (signed char)q;
        built2 = q == 27;
        if (!built2) { fprintf(stderr, "s2b_f4: %d taps\n", q); abort(); }
    }
    if (!built) {   /* class p's taps: per axis even -> (k 1, +0), odd -> (k 0, +1), (k 2, +0); paired in order, an odd count padded with tap 27 */
        int q = 0;
        for (int p = 0; p < 8; p++) {
            PT.first[p] = (signed char)q;
            int tk[8], td[8][3], nt = 0;
            const int pa[3] = {p >> 2, (p >> 1) & 1, p & 1};
            const int nz = pa[0] ? 2 : 1, ny = pa[1] ? 2 : 1, nx = pa[2] ? 2 : 1;
            for (int a = 0; a < nz; a++) for (int bb = 0; bb < ny; bb++) for (int c = 0; c < nx; c++) {
                const int sel[3] = {a, bb, c}; int k3[3], d3[3];
                for (int ax = 0; ax < 3; ax++) { if (!pa[ax]) { k3[ax] = 1; d3[ax] = 0; } else if (!sel[ax]) { k3[ax] = 0; d3[ax] = 1; } else { k3[ax] = 2; d3[ax] = 0; } }
                tk[nt] = (k3[0] * 3 + k3[1]) * 3 + k3[2]; td[nt][0] = d3[0]; td[nt][1] = d3[1]; td[nt][2] = d3[2]; nt++;
            }
            for (int i = 0; i < nt; i += 2, q++) {
                PT.ta[q] = (signed char)tk[i]; for (int ax = 0; ax < 3; ax++) PT.da[q][ax] = (signed char)td[i][ax];
                if (i + 1 < nt) { PT.tb[q] = (signed char)tk[i + 1]; for (int ax = 0; ax < 3; ax++) PT.db[q][ax] = (signed char)td[i + 1][ax]; }
                else { PT.tb[q] = 27; for (int ax = 0; ax < 3; ax++) PT.db[q][ax] = PT.da[q][ax]; }
            }
        }
        PT.first[8] = (signed char)q;
        built = q == 14;
        if (!built) { fprintf(stderr, "s2b_f4: %d tap pairs\n", q); abort(); }
    }
    float *wt = lp_buf<float>(2, (size_t)C * C * 27);
    uint8_t *wq = lp_buf<uint8_t>(0, (size_t)28 * C * C / 2), *ws = lp_buf<uint8_t>(1, (size_t)28 * C * NCH);
    s2f4_wt_k<<<nblk_((size_t)C * C * 27, 256), 256>>>(w, wt, C);
    prep_w4_k<<<nblk_((size_t)28 * C * NCH, 128), 128>>>(wt, wq, ws, C, C, C, C, -1, 0, -1, 0);
    const int Mz = ys.d, nzt = (Mz + 1) / 2, base = (ys.w / 16) * (ys.h / 4) * NCH;
    int ZC = nzt < 8 ? nzt : 8;
    while (ZC > 1 && (size_t)base * ((nzt + ZC - 1) / ZC) < 480) ZC--;
    const dim3 grid((unsigned)(ys.w / 16), (unsigned)(ys.h / 4), (unsigned)(((nzt + ZC - 1) / ZC) * xs.n * NCH));
    const int NU = NCH == 1 ? 14 : 27;
    const size_t sm = (size_t)3 * 5 * 17 * 16 * NCH + ((3 * 5 * 17 * NCH + 127) & ~127) + (size_t)NU * 32 * 32 + (size_t)NU * 32 * 2 + 8 * 8 * 36 * 4;
    static unsigned at = 0; int dev = 0; cudaGetDevice(&dev);
    const unsigned bit = 1u << (4 * dev + 2 * (NCH - 1) + (xdt == 4));
#define S2BF(XB_, NCH_) do { if (!(at & bit)) { at |= bit; cudaFuncSetAttribute((const void *)bwd_data_s2_f4_k<XB_, NCH_>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); } \
                             bwd_data_s2_f4_k<XB_, NCH_><<<grid, 256, sm>>>((const uint8_t *)gy, wq, ws, (uint8_t *)gx, xs.n, xs.d, xs.h, xs.w, ys.d, ys.h, ys.w, accum, osr, NCH_ == 1 ? PT : PT2, ZC); } while (0)
    if (NCH == 1) { if (xdt == 4) S2BF(4, 1); else S2BF(8, 1); }
    else { if (xdt == 4) S2BF(4, 2); else S2BF(8, 2); }
#undef S2BF
    LPCK();
    return 1;
}
/* the 96 KB dynamic shared memory attribute is per device: set it once on every GPU that launches (split z uses two) */
extern "C" int lp_bwd_data_s2_mx_tc(const void *gy, shape5 ys, const float *w, shape5 xs, void *gx, int accum, int gdt, int xdt, unsigned osr) {
    if (s2b_f4(gy, ys, w, xs, gx, accum, gdt, xdt, osr)) return 3;   /* 32 channels, MX-fp4 gy: the fp4 kernel (UFSM_S2B_F4=0: bf16); 3 = fp4 compute */
    static int on = -1; if (on < 0) on = getenv("UFSM_S2B_TC") ? atoi(getenv("UFSM_S2B_TC")) : 1;
    if (!on || xs.c != ys.c || (xs.c != 16 && xs.c != 32 && xs.c != 64)) return 0;
    const int GB = gdt == 4 ? 4 : 8, XB = xdt == 4 ? 4 : 8;
    const int C = xs.c, CB = C > 32 ? 32 : C, TMX = C == 16 ? 16 : 8, TMZ = 2, TMY = C > 32 ? 4 : 8;
    const int Mz = (xs.d + 1) / 2, My = (xs.h + 1) / 2, Mx = (xs.w + 1) / 2;
    const unsigned nb = (unsigned)(((Mx + TMX - 1) / TMX) * ((My + TMY - 1) / TMY) * ((Mz + TMZ - 1) / TMZ) * xs.n);
    const size_t sm = (size_t)(TMZ + 1) * (TMY + 1) * (TMX + 1) * (C + 8) * 2 + (size_t)8 * CB * (C + 8) * 2 + (size_t)8 * 8 * CB * 4;
    const dim3 grid(nb, (unsigned)(C / CB));
    __nv_bfloat16 *wtg = lp_buf<__nv_bfloat16>(0, (size_t)27 * C * (C + 8));
    const uint8_t *gq = (const uint8_t *)gy; uint8_t *xq = (uint8_t *)gx;
#define S2TC(CC, G, X) do { static unsigned at_ = 0; int dev_ = 0; cudaGetDevice(&dev_); if (!(at_ >> dev_ & 1u)) { at_ |= 1u << dev_; cudaFuncSetAttribute((const void *)bwd_data_s2_tc_k<CC, G, X>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); } \
                            s2tc_prep_w_k<CC><<<nblk_(27 * CC * (CC + 8), 256), 256>>>(w, wtg); \
                            bwd_data_s2_tc_k<CC, G, X><<<grid, 256, sm>>>(gq, wtg, xq, xs.n, xs.d, xs.h, xs.w, ys.d, ys.h, ys.w, accum, osr); } while (0)
    if (C == 16) { if (GB == 8 && XB == 8) S2TC(16, 8, 8); else if (GB == 4 && XB == 4) S2TC(16, 4, 4); else return 0; }
    else if (C == 32) { if (GB == 8 && XB == 8) S2TC(32, 8, 8); else if (GB == 4 && XB == 4) S2TC(32, 4, 4); else return 0; }
    else { if (GB == 8 && XB == 8) S2TC(64, 8, 8); else if (GB == 4 && XB == 4) S2TC(64, 4, 4); else return 0; }
#undef S2TC
    LPCK();
    return 1;
}
