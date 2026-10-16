#pragma once
/* kernels and launchers of lp_f8wgrad (instantiated per type in lp_f8wgrad_i*.cu) */
#include "lp_common.cuh"
#include "lp_sr.cuh"   /* sr_e4m3x: the gy operand's exact stochastic rounding (MX-sourced gy: off the grid only below 2^-6 after the block scale, rare) */
#ifndef BW8F_MINB
#define BW8F_MINB BW8_MINB   /* blocks per SM of the network input conv's FAST path */
#endif

/* ======================= FP8 weight gradient, k=3, stride 1 =======================
   Block = 9 warps; warp w owns the tap row (kz, ky) = (w/3, w%3) and all three kx, so one pair of 32-bit loads of
   an input row feeds the B fragments of kx = 0, 1, 2 through byte funnel shifts. M = 16*MT output channels (GY),
   N = 8*NT input channels, K = 32 voxels (two 16-voxel output rows of the same z plane). A block walks ZC
   z-steps of 2 output planes (8 y x 16 x each), accumulating in registers, then atomically adds into gw.
   The input tile is a ring of 4 z planes (10 x 18 positions) per channel: each z-step stages only the 2 new planes.
   Both rows of a K block lie in the same input plane for every tap, so the B scale is per (channel, plane).
   Smem: X ring fp8 [ci][4 planes][10 rows][24 B], position p of a row at byte 3 + p so that x = ox0.. is 4-aligned
         (ci stride 1040 B = 260 words == 4 mod 32 -> conflict-free fragment loads), GY tile fp8 [co][16 rows][16 B] (co stride 272 B, ldmatrix conflict-free) + scales. */
template <int MT, int NT, typename T, typename TG, int FAST = 0>   /* FAST: the network input conv's path, 4 or 8 = the real x channels */
__global__ void __launch_bounds__(288, FAST ? BW8F_MINB : BW8_MINB) conv_bwd_w_f8_k(const T *__restrict__ x, const TG *__restrict__ gy, float *__restrict__ gw, float *__restrict__ gb,
                                                       int N, int Ci, int D, int H, int W, int Co, gnp_t gp, split_t sp, int ZC, int coop) {
    constexpr int CH = 8 * NT, BMo = 16 * MT;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    uint8_t *sxq = smem_raw;                        /* [CH][X8_CS] */
    uint8_t *sg = sxq + CH * X8_CS;                 /* [BMo][G8_CS] */
    uint8_t *sgs = sg + BMo * G8_CS;                /* [BMo][8 ksteps] */
    uint8_t *sxs = sgs + BMo * 8;                   /* [CH][4 slots] */
    float *sbias = (float *)(sxs + CH * 4);         /* [BMo] */
    chan_t *ctab8 = (chan_t *)(smem_raw + (((unsigned char *)(sbias + BMo) - smem_raw + 31) & ~31));   /* offset from smem_raw: an integer round trip would make every access below a generic load */   /* [CH] MX x: per-channel descriptors */
    unsigned *amx = (unsigned *)(ctab8 + CH);      /* [2][CH] MX x: per-channel plane amax (bits), double-buffered by plane */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int ci0 = blockIdx.y * CH, co0 = blockIdx.z * BMo;
    int bz = blockIdx.x;   /* spatial tiles use grid.x: grid.z is limited to 65535 */
    const int nxt = (W + 15) / 16, nyt = (H + 7) / 8, nzt = (D + 1) / 2;
    const int ox0 = (bz % nxt) * 16; bz /= nxt;
    const int oy0 = (bz % nyt) * 8; bz /= nyt;
    const int nzc = (nzt + ZC - 1) / ZC;
    const int zc = bz % nzc; const int n = bz / nzc;
    const size_t plane = (size_t)D * H * W;
    const int Cx = sp.x2 ? sp.c_split : Ci;
    const int kz = warp / 3, ky = warp % 3;
    const bool do_bias = gb && blockIdx.y == 0;
    float acc[3][MT][NT][4];
#pragma unroll
    for (int a = 0; a < 3; a++) for (int m = 0; m < MT; m++) for (int q = 0; q < NT; q++) for (int k = 0; k < 4; k++) acc[a][m][q][k] = 0.f;
    if (threadIdx.x < BMo) sbias[threadIdx.x] = 0.f;
    const bool vec = (W & 3) == 0;
    const int zt_begin = zc * ZC;
    float bpart[BMo];   /* coop gy path: this thread's gy sums per output channel over the block's z-steps (reduced once at the end) */
#pragma unroll
    for (int c = 0; c < BMo; c++) bpart[c] = 0.f;
    /* MX x whose CH channels are consecutive entries of one stored block row (the common case): the voxel's row words are
       loaded and decoded once for all CH channels (thread per voxel), instead of one strided byte per (channel, voxel) */
    bool uni = false;
    const bool upt = sp.up && ci0 < Cx;   /* fused decoder upsample: this tile reads the half-resolution x (host: Cx % CH == 0) */
    if constexpr (IS_MX(T)) {
        if (threadIdx.x < CH) { const int ci = ci0 + threadIdx.x; ctab8[threadIdx.x] = make_chan(ci < Ci ? ci : Ci, Ci, Cx, n, upt ? plane / 8 : plane, x, sp, gp, N); }
        if (threadIdx.x < 2 * CH) amx[threadIdx.x] = 0u;
        __syncthreads();
        uni = (coop & 1) && ctab8[0].p != nullptr && (ctab8[0].nib & 7) == 0;
#pragma unroll
        for (int q = 1; q < CH; q++) if (ctab8[q].p && (ctab8[q].sp != ctab8[0].sp || ctab8[q].nib != ctab8[0].nib + q)) uni = false;
    }
    if (upt && !uni) __trap();   /* the per-channel paths do not upsample: the host only fuses aligned 16-channel tiles */
    auto mma_zstep = [&](int oz0) {   /* the z-step's MMAs from the staged x ring and gy tile */
        unsigned long long sa8[MT];   /* this lane's 8 K-block scale bytes of the A rows (one 8-byte load, not 8 byte loads: LSU-bound) */
#pragma unroll
        for (int m = 0; m < MT; m++) sa8[m] = *(const unsigned long long *)(sgs + (m * 16 + g + 8 * (t & 1)) * 8);
#pragma unroll
        for (int vz = 0; vz < 2; vz++) {
            const int slot = (oz0 + vz + kz) & 3;
            unsigned sb[NT];
#pragma unroll
            for (int q = 0; q < NT; q++) sb[q] = sxs[(q * 8 + g) * 4 + slot];
#pragma unroll 2
            for (int kk = 0; kk < 4; kk++) {
                const int ks = vz * 4 + kk, vy0 = 2 * kk;
                unsigned af[MT][4], sa[MT];
#pragma unroll
                for (int m = 0; m < MT; m++) {
                    int mat = lane >> 3, co = m * 16 + (mat & 1) * 8 + (lane & 7), row = 2 * ks + (mat >> 1);
                    ldsm_x4(af[m], sg + co * G8_CS + row * 16);
                    sa[m] = (unsigned)(sa8[m] >> (8 * ks)) & 0xffu;
                }
                const int o0 = slot * X8_PS + (vy0 + ky) * 24;
#pragma unroll
                for (int q = 0; q < NT; q++) {
                    const unsigned *p0 = (const unsigned *)(sxq + (q * 8 + g) * X8_CS + o0) + t;
                    unsigned a0 = p0[0], a1 = p0[1], a2 = p0[2], c0 = p0[6], c1 = p0[7], c2 = p0[8];   /* next row = +24 B = +6 words */
                    unsigned b[3][2] = {{__funnelshift_r(a0, a1, 24), __funnelshift_r(c0, c1, 24)}, {a1, c1}, {__funnelshift_r(a1, a2, 8), __funnelshift_r(c1, c2, 8)}};
#pragma unroll
                    for (int kx = 0; kx < 3; kx++)
#pragma unroll
                        for (int m = 0; m < MT; m++) mma_f8(acc[kx][m][q], af[m], b[kx], sa[m], sb[q]);
                }
            }
        }
    };
    /* FAST (the network input conv: NT 1, one 8-channel MX x row per voxel, the BMo gy channels inside one stored block):
       - x: thread per voxel; the next z-step's two raw planes are loaded into registers before the MMA (latency overlapped);
         the per-channel scales are computed once (threads < 2 CH) instead of in every thread;
       - gy: the next z-step's raw rows and scale bytes are copied into smem (cp.async) during the MMA, and rounded with lane =
         channel, a warp per K block (32 voxels): the K block amax, its scale and the bias sum are in-lane loops (lane = voxel
         needed a warp reduction and the scale arithmetic per channel in every lane), the e4m3 bytes four to a word.
       Same values, scales and SR keys as the per-element paths (bias: summation order differs). */
    float bacc = 0.f;
    if constexpr (FAST) {
        const bool fast = uni && !upt && (coop & 3) == 3 && ((co0 % mx_bw(Co)) + BMo <= mx_bw(Co) || (co0 == 0 && mx_bw(Co) < BMo && Co <= mx_bw(Co)));   /* (or one
                                                                                                     block narrower than the tile: 8 channels) */
        if (!fast) __trap();   /* the host only picks FAST for these */
        /* amx: [2 buffers][2 planes][CH] plane amaxes (the z-steps alternate buffers: a buffer is reset after the barrier that ends its
           readers and reused two z-steps later, past the next z-step's first barrier) */
        uint8_t *graw = smem_raw + (((unsigned char *)(amx + 4 * CH) - smem_raw + 15) & ~15);   /* [256 voxels][BMo] raw gy (e4m3) */
        uint8_t *gscl = graw + 256 * BMo;                                                           /* [256] gy scale bytes */
        constexpr int XW = IS_MX4(T) ? 1 : 2;   /* 32-bit words of the 8-channel x row */
        const chan_t c0 = ctab8[0];
        const int tid = threadIdx.x, xrow = tid / 18, xp = tid % 18, xgy = oy0 - 1 + xrow, xgx = ox0 - 1 + xp;
        const bool xin = tid < 180 && xgy >= 0 && xgy < H && xgx >= 0 && xgx < W;
        const bool Gany = gp.G != 0 || sp.gp2.G != 0;
        unsigned xr[2][XW], xsb[2]; bool xok[2];
        int xgz0 = 0;   /* the planes in xr (slab-local GroupNorm: xdecode applies their slabs) */
        auto xload = [&](int gz0) {   /* raw rows of planes gz0, gz0 + 1 */
            xgz0 = gz0;
#pragma unroll
            for (int pl = 0; pl < 2; pl++) {
                const int gz = gz0 + pl;
                xok[pl] = xin && gz >= 0 && gz < D;
                xsb[pl] = 0u;
#pragma unroll
                for (int w = 0; w < XW; w++) xr[pl][w] = 0u;
                if (xok[pl]) {
                    const size_t off = ((size_t)gz * H + xgy) * W + xgx;
                    const unsigned *rw = (const unsigned *)((const uint8_t *)c0.p + off * c0.rb + (IS_MX4(T) ? (c0.nib >> 1) : 0));
                    xsb[pl] = c0.sp[off];
#pragma unroll
                    for (int w = 0; w < XW; w++) xr[pl][w] = __ldg(rw + w);
                }
            }
        };
        /* XC: the real x channels (the 4-channel network input fills half of the 8-channel tile): only those are decoded, reduced and
           staged; the padding channels' ring rows stay zero and their scale bytes 1 (mx_exp of a zero amax: -126), as staging zeros
           wrote them before */
        constexpr int XC = FAST < CH ? FAST : CH;
        if constexpr (XC < CH) {
            for (int k = threadIdx.x; k < (CH - XC) * X8_CS; k += blockDim.x) sxq[XC * X8_CS + k] = 0;
            if (threadIdx.x < (CH - XC) * 4) sxs[XC * 4 + threadIdx.x] = (uint8_t)(-126 + 127);
        }
        /* the new planes gz0, gz0 + 1 from xr, in two halves around one barrier: xreduce decodes and transforms them and red.shared's
           the per-(plane, channel) amaxes into amx buffer b; after the barrier xstore takes the scales (lanes < 2 XC, shuffled to the
           warp) and writes the e4m3 bytes into the ring */
        auto xdecode = [&](float (&v)[2][XC]) {
#pragma unroll
            for (int pl = 0; pl < 2; pl++) {
#pragma unroll
                for (int q = 0; q < XC; q++) v[pl][q] = 0.f;
                if (xok[pl]) {
                    const float sc = mx_scale(xsb[pl]);
                    if constexpr (IS_MX4(T)) {
#pragma unroll
                        for (int q = 0; q < XC / 2; q++) { const float2 d = dec_e2m1x2(xr[pl][0] >> (8 * q)); v[pl][2 * q] = fmaf(1.f, d.x * sc, 0.f); v[pl][2 * q + 1] = fmaf(1.f, d.y * sc, 0.f); }
                    } else {
#pragma unroll
                        for (int w = 0; w < XC / 4; w++) {
                            const float2 d0 = dec_e4m3x2((unsigned short)(xr[pl][w] & 0xffffu)), d1 = dec_e4m3x2((unsigned short)(xr[pl][w] >> 16));
                            v[pl][4 * w] = fmaf(1.f, d0.x * sc, 0.f); v[pl][4 * w + 1] = fmaf(1.f, d0.y * sc, 0.f);
                            v[pl][4 * w + 2] = fmaf(1.f, d1.x * sc, 0.f); v[pl][4 * w + 3] = fmaf(1.f, d1.y * sc, 0.f);
                        }
                    }
                    if (gp.ns || sp.gp2.ns) {   /* slab-local GroupNorm: plane xgz0 + pl's slab */
#pragma unroll
                        for (int q = 0; q < XC; q++) {
                            const chan_t &cq = ctab8[q]; const float2 ab = chan_ab(ci0 + q < Ci ? ci0 + q : Ci, Ci, Cx, n, sp, gp, max(xgz0 + pl, 0));
                            v[pl][q] = cq.p ? act_ab(v[pl][q], ab.x, ab.y, Gany && cq.g) : 0.f;
                        }
                    } else {
#pragma unroll
                    for (int q = 0; q < XC; q++) { const chan_t &cq = ctab8[q]; v[pl][q] = cq.p ? act_ab(v[pl][q], cq.a, cq.b, Gany && cq.g) : 0.f; }
                    }
                }
            }
        };
        auto xreduce = [&](const float (&v)[2][XC], int b) {
            unsigned ra[2][XC];
#pragma unroll
            for (int pl = 0; pl < 2; pl++)
#pragma unroll
                for (int q = 0; q < XC; q++) ra[pl][q] = __reduce_max_sync(0xffffffffu, __float_as_uint(v[pl][q]) & 0x7fffffffu);
            if (lane == 0) {   /* red.shared directly: atomicMax by one lane still got the compiler's warp-aggregation code */
#pragma unroll
                for (int pl = 0; pl < 2; pl++)
#pragma unroll
                    for (int q = 0; q < XC; q++) asm volatile("red.shared.max.u32 [%0], %1;" :: "r"((unsigned)__cvta_generic_to_shared(&amx[(b * 2 + pl) * CH + q])), "r"(ra[pl][q]) : "memory");
            }
        };
        auto xstore = [&](const float (&v)[2][XC], int gz0, int b) {
            const int lp = lane < 2 * XC ? lane : 0, lpl = lp / XC, lq = lp % XC;
            const int e = mx_exp(__uint_as_float(amx[(b * 2 + lpl) * CH + lq]), 1.f / 448.f);
            const float ml = exp2i(-e);
            if (warp == 0 && lane < 2 * XC) sxs[lq * 4 + ((gz0 + lpl + 1) & 3)] = (uint8_t)(e + 127);
            float mq[2][XC];
#pragma unroll
            for (int pl = 0; pl < 2; pl++)
#pragma unroll
                for (int q = 0; q < XC; q++) mq[pl][q] = __shfl_sync(0xffffffffu, ml, pl * XC + q);
            if (tid < 180) {
#pragma unroll
                for (int pl = 0; pl < 2; pl++) {
                    uint8_t *dst = sxq + ((gz0 + pl + 1) & 3) * X8_PS + xrow * 24 + 3 + xp;
#pragma unroll
                    for (int q = 0; q < XC; q++) dst[q * X8_CS] = cvt_e4m3(v[pl][q] * mq[pl][q]);
                }
            }
        };
        const int gbw = mx_bw(Co);
        const size_t Sg = (size_t)D * H * W, gri = ((size_t)n * mx_nb(Co) + co0 / gbw) * Sg;
        constexpr int GB = MX_BITS(TG), GRB = BMo * GB / 8, GCB = GRB >= 16 ? 16 : 8;   /* gy bits (MX-fp8 or MX-fp4 gradients), raw row bytes, copy chunk */
        const uint8_t *gq = (const uint8_t *)gy, *gqs = gq + (size_t)N * mx_nb(Co) * Sg * mx_rb(gbw, GB);
        const int rrb = mx_rb(gbw, GB) < GRB ? mx_rb(gbw, GB) : GRB;   /* the stored row's bytes in the slot (one 8-channel block: 4 / 8, the rest zero) */
        auto gload = [&](int oz0) {   /* cp.async: voxel i = row 16 + xx of the z-step (row = vz 8 + vy) */
            if (rrb < GRB) for (int k = tid; k < 256 * (GRB / 4); k += 288) {   /* narrow rows: 4-byte copies, zero past the row */
                const int i = k / (GRB / 4), h = k % (GRB / 4), row = i >> 4, oz = oz0 + (row >> 3), oy = oy0 + (row & 7), ox = ox0 + (i & 15);
                const bool ok = oz < D && oy < H && ox < W && 4 * h < rrb;
                cp_async<4>(graw + i * GRB + 4 * h, ok ? gq + (gri + ((size_t)oz * H + oy) * W + ox) * mx_rb(gbw, GB) + 4 * h : gq, ok);
            }
            else for (int k = tid; k < 256 * (GRB / GCB); k += 288) {
                const int i = k / (GRB / GCB), h = k % (GRB / GCB), row = i >> 4, oz = oz0 + (row >> 3), oy = oy0 + (row & 7), ox = ox0 + (i & 15);
                const bool ok = oz < D && oy < H && ox < W;
                cp_async<GCB>(graw + i * GRB + GCB * h, ok ? gq + (gri + ((size_t)oz * H + oy) * W + ox) * mx_rb(gbw, GB) + (co0 % gbw) * GB / 8 + GCB * h : gq, ok);
            }
            if (tid < 64) {   /* scale bytes, 4 voxels per copy (host: W % 4 == 0) */
                const int row = tid >> 2, oz = oz0 + (row >> 3), oy = oy0 + (row & 7), ox = ox0 + 4 * (tid & 3);
                const bool ok = oz < D && oy < H && ox < W;
                cp_async<4>(gscl + 4 * tid, ok ? gqs + gri + ((size_t)oz * H + oy) * W + ox : gqs, ok);
            }
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        };
        auto gstage = [&](int oz0) {   /* warp = K block ks (2 rows x 16 voxels); lane = channel c, and with 16 outputs its row of the block */
            constexpr int NP = 32 / BMo, NV = 32 / NP;   /* lanes per channel, voxels per lane */
            const int c = lane % BMo, part = lane / BMo, ks = warp, i0 = part * NV;
            if (ks >= 8) return;
            const bool cok = co0 + c < Co;
            const uint8_t *rb = graw + (ks * 32 + i0) * GRB + (GB == 8 ? c : c >> 1), *sb = gscl + ks * 32 + i0;
            unsigned sw[NV / 4];   /* the NV scale bytes (16-byte aligned): NV / 16 vector loads, not NV byte loads (the kernel is LSU-bound) */
#pragma unroll
            for (int j = 0; j < NV / 16; j++) { const uint4 w4 = *(const uint4 *)(sb + 16 * j); sw[4 * j] = w4.x; sw[4 * j + 1] = w4.y; sw[4 * j + 2] = w4.z; sw[4 * j + 3] = w4.w; }
            float v[NV];
            unsigned am = 0u;
#pragma unroll
            for (int i = 0; i < NV; i += 2) {
                float2 d;
                if constexpr (GB == 8) d = dec_e4m3x2((unsigned short)(rb[i * GRB] | (rb[(i + 1) * GRB] << 8)));
                else { const float2 a = dec_e2m1x2(rb[i * GRB]), b = dec_e2m1x2(rb[(i + 1) * GRB]); d = (c & 1) ? make_float2(a.y, b.y) : make_float2(a.x, b.x); }   /* this lane's nibble of each row */
                const unsigned s0 = (sw[i >> 2] >> (8 * (i & 3))) & 0xffu, s1 = (sw[(i + 1) >> 2] >> (8 * ((i + 1) & 3))) & 0xffu;
                v[i] = cok ? d.x * mx_scale(s0) : 0.f; v[i + 1] = cok ? d.y * mx_scale(s1) : 0.f;
                am = max(am, max(__float_as_uint(v[i]) & 0x7fffffffu, __float_as_uint(v[i + 1]) & 0x7fffffffu));
            }
            if constexpr (NP == 2) am = max(am, __shfl_xor_sync(0xffffffffu, am, 16));   /* the block's other row */
            const unsigned u = __float_as_uint(__uint_as_float(am) * (1.f / 448.f));   /* mx_exp without its early return */
            const int e = (u & 0x7fffffffu) > 0x7f800000u ? 128 : max(-126, min(126, (int)((u >> 23) & 0xff) - 127 + ((u & 0x7fffff) != 0)));
            const float m = exp2i(-e);
            if (part == 0) sgs[c * 8 + ks] = (uint8_t)(e + 127);
            unsigned *dst = (unsigned *)(sg + c * G8_CS + ks * 32 + i0);
            float q[NV];
            unsigned mn = 0xffffffffu;   /* the smallest nonzero |q| (bits) - 1 */
#pragma unroll
            for (int i = 0; i < NV; i++) { q[i] = v[i] * m; mn = min(mn, (__float_as_uint(q[i]) & 0x7fffffffu) - 1u); }
            if (sp.sr && mn < 0x3c7fffffu) {   /* a value below 2^-6 (MX gy: the only ones that can be off the e4m3 grid): sr_e4m3x */
                const uint64_t id0 = (((uint64_t)n * Co + co0 + c) * D + oz0 + ((2 * ks) >> 3)) * H;   /* element id: ((n Co + co) D + oz) H W + oy W + ox */
#pragma unroll
                for (int i = 0; i < NV; i++) {
                    const int ii = i0 + i, row = 2 * ks + (ii >> 4);
                    q[i] = sr_e4m3x(q[i], sp.sr, (id0 + (uint64_t)(oy0 + (row & 7))) * (uint64_t)W + (uint64_t)(ox0 + (ii & 15)));
                }
            }
#pragma unroll
            for (int w = 0; w < NV / 4; w++) dst[w] = cvt_e4m3x4(q[4 * w], q[4 * w + 1], q[4 * w + 2], q[4 * w + 3]);   /* (on the grid: exact) */
            if (do_bias) {
                float bs = 0.f;
#pragma unroll
                for (int i = 0; i < NV; i++) bs += v[i];
                bacc += bs;
            }
        };
        const int zt_end = min(nzt, (zc + 1) * ZC);
        /* two barriers per z-step: B2 after the x amax reductions (graw landed, every warp past the previous MMA: the ring slots and
           the gy tile may be rewritten), B4 after the staging (the MMA's operands complete) */
        if (tid < 2 * CH) amx[2 * CH + tid] = 0u;   /* buffer 1 (buffer 0 was zeroed with the channel table) */
        xload(2 * zt_begin - 1);   /* the first z-step's two leading planes */
        gload(2 * zt_begin);
        {
            float v[2][XC];
            xdecode(v); xreduce(v, 0);
            __syncthreads();
            xstore(v, 2 * zt_begin - 1, 0);
        }
        xload(2 * zt_begin + 1);
        for (int zt = zt_begin; zt < zt_end; zt++) {
            const int oz0 = zt * 2, b = (zt - zt_begin + 1) & 1;
            float v[2][XC];
            xdecode(v); xreduce(v, b);
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            __syncthreads();   /* B2: amx[b] complete; graw has landed; every warp is past the previous MMA */
            if (zt == zt_begin && tid < 2 * CH) amx[tid] = 0u;   /* the prologue's buffer 0 (its readers are past B2) */
            xstore(v, oz0 + 1, b);
            gstage(oz0);
            __syncthreads();   /* B4 */
            if (tid < 2 * CH) amx[b * 2 * CH + tid] = 0u;   /* reused two z-steps on, after the next B2 */
            if (zt + 1 < zt_end) { xload(oz0 + 3); gload(oz0 + 2); }
            mma_zstep(oz0);
        }
    } else
    for (int zt = zt_begin; zt < nzt && zt < (zc + 1) * ZC; zt++) {
        const int oz0 = zt * 2;
        const int np = zt == zt_begin ? 4 : 2, gz_first = zt == zt_begin ? oz0 - 1 : oz0 + 1;
        __syncthreads();
        if (IS_MX(T) && uni) {   /* thread per voxel (row tid / 18, p = tid % 18) of each new plane, all CH channels */
            const chan_t c0 = ctab8[0];
            const int tid = threadIdx.x, row = tid / 18, p = tid % 18;
            const bool Gany = gp.G != 0 || sp.gp2.G != 0;
            for (int pi = 0; pi < np; pi++) {
                const int gz = gz_first + pi, slot = (gz + 1) & 3;
                unsigned *am = amx + (pi & 1) * CH;
                const int gyy = oy0 - 1 + row, gx = ox0 - 1 + p;
                const bool inb = tid < 180 && gz >= 0 && gz < D && gyy >= 0 && gyy < H && gx >= 0 && gx < W;
                float v[CH];
#pragma unroll
                for (int q = 0; q < CH; q++) v[q] = 0.f;
                if (inb) {
                    mx_rowch<T, CH>(c0, upt, gz, gyy, gx, D, H, W, v);   /* upt: the tile is the half-resolution x segment of a decoder conv */
#pragma unroll
                    for (int q = 0; q < CH; q++) {   /* (slab-local GroupNorm: plane gz's slab) */
                        const chan_t &cq = ctab8[q];
                        const float2 ab = gp.ns || sp.gp2.ns ? chan_ab(ci0 + q < Ci ? ci0 + q : Ci, Ci, Cx, n, sp, gp, gz) : make_float2(cq.a, cq.b);
                        v[q] = cq.p ? act_ab(v[q], ab.x, ab.y, Gany && cq.g) : 0.f;
                    }
                }
                unsigned ra[CH];   /* the channels' warp amaxes first (independent reductions), then lane 0's shared maxima */
#pragma unroll
                for (int q = 0; q < CH; q++) ra[q] = __reduce_max_sync(0xffffffffu, __float_as_uint(v[q]) & 0x7fffffffu);
                if (lane == 0) {
#pragma unroll
                    for (int q = 0; q < CH; q++) if (ra[q]) atomicMax(&am[q], ra[q]);
                }
                __syncthreads();
                if (tid < 180) {
                    uint8_t *dst = sxq + slot * X8_PS + row * 24 + 3 + p;
                    float mq[CH];   /* the multipliers read once (the stores below could alias am for the compiler) */
#pragma unroll
                    for (int q = 0; q < CH; q++) mq[q] = exp2i(-mx_exp(__uint_as_float(am[q]), 1.f / 448.f));
#pragma unroll
                    for (int q = 0; q < CH; q++) dst[q * X8_CS] = cvt_e4m3(v[q] * mq[q]);
                }
                if (tid < CH) { sxs[tid * 4 + slot] = (uint8_t)(mx_exp(__uint_as_float(am[tid]), 1.f / 448.f) + 127); amx[((pi + 1) & 1) * CH + tid] = 0u; }
            }
        } else
        /* X: warp per (channel, plane): 10 rows x (4 aligned float4 for x = ox0..ox0+15, plus the halo voxels
           ox0-1 and ox0+16); position p of a row is stored at byte 3 + p of its 24-byte slot */
        for (int task = warp; task < CH * np; task += 9) {
            int k = task / np, gz = gz_first + task % np, slot = (gz + 1) & 3, ci = ci0 + k;
            const bool ok = ci < Ci && gz >= 0 && gz < D;
            chan_t c = make_chan(ok ? ci : Ci, Ci, Cx, n, plane, x, sp, gp, N, ok ? gz : 0);   /* (slab GroupNorm: plane gz's slab) */
            const T *xc = ok && !IS_MX(T) ? (const T *)c.p + (size_t)gz * H * W : x;
            const bool G = (gp.G != 0 || sp.gp2.G != 0) && c.g, el = IS_MX(T);   /* el: per-element access (MX rows) */
            float4 v4[2]; float vs = 0.f, amax = 0.f;
#pragma unroll
            for (int i = 0; i < 2; i++) {
                int f = lane + 32 * i, row = f >> 2, gyy = oy0 - 1 + row, gx = ox0 + 4 * (f & 3);
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (ok && f < 40 && gyy >= 0 && gyy < H) {
                    const T *src = xc + (size_t)gyy * W + gx;
                    if constexpr (IS_MX(T)) { if (gx < W) v = ldc4_mx<T>(c, cof(c, gz, gyy, gx, H, W), min(4, W - gx), vec); }   /* one call per 4 voxels */
                    else if (vec) { if (gx < W) v = ldx4(src); }
                    else { if (gx < W) v.x = ldx(src, 0); if (gx + 1 < W) v.y = ldx(src, 1); if (gx + 2 < W) v.z = ldx(src, 2); if (gx + 3 < W) v.w = ldx(src, 3); }
                    v.x = gx < W ? act_ab(v.x, c.a, c.b, G) : 0.f; v.y = gx + 1 < W ? act_ab(v.y, c.a, c.b, G) : 0.f;
                    v.z = gx + 2 < W ? act_ab(v.z, c.a, c.b, G) : 0.f; v.w = gx + 3 < W ? act_ab(v.w, c.a, c.b, G) : 0.f;
                }
                v4[i] = v;
                amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
            }
            {   /* halo: lane < 20 -> row lane >> 1, side lane & 1 (x = ox0 - 1 or ox0 + 16) */
                int row = lane >> 1, gyy = oy0 - 1 + row, gx = (lane & 1) ? ox0 + 16 : ox0 - 1;
                if (ok && lane < 20 && gyy >= 0 && gyy < H && gx >= 0 && gx < W) vs = act_ab(el ? ldc<T>(c, cof(c, gz, gyy, gx, H, W)) : ldx(xc, (size_t)gyy * W + gx), c.a, c.b, G);
                amax = fmaxf(amax, fabsf(vs));
            }
#pragma unroll
            for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
            int e = mx_exp(amax, 1.f / 448.f);
            float m = exp2i(-e);
            uint8_t *dst = sxq + k * X8_CS + slot * X8_PS;
#pragma unroll
            for (int i = 0; i < 2; i++) {
                int f = lane + 32 * i;
                if (f < 40) *(unsigned *)(dst + (f >> 2) * 24 + 4 + 4 * (f & 3)) = cvt_e4m3x4(v4[i].x * m, v4[i].y * m, v4[i].z * m, v4[i].w * m);
            }
            if (lane < 20) dst[(lane >> 1) * 24 + ((lane & 1) ? 20 : 3)] = cvt_e4m3(vs * m);
            if (lane == 0) sxs[k * 4 + slot] = (uint8_t)(e + 127);
        }
        /* GY stored MX-fp8 with the BMo output channels inside one block row: thread per output voxel of the z-step (256), the
           voxel's row words decoded once for all BMo channels; a K block (2 rows x 16 x) is exactly one warp, so the block
           amax is one redux per channel */
        bool gdone = false;
        if constexpr (IS_MX8(TG)) {
            const int gbw = mx_bw(Co);
            if ((coop & 2) && (co0 % gbw) + BMo <= gbw) {
                gdone = true;
                if (threadIdx.x < 256) {
                    const int ks = threadIdx.x >> 5, row = ks * 2 + (lane >> 4), xx = lane & 15, vz = row >> 3, vy = row & 7;
                    const int oz = oz0 + vz, oy = oy0 + vy, ox = ox0 + xx;
                    const size_t Sg = (size_t)D * H * W, gri = ((size_t)n * mx_nb(Co) + co0 / gbw) * Sg;
                    const uint8_t *gq = (const uint8_t *)gy;
                    float v[BMo];
#pragma unroll
                    for (int c = 0; c < BMo; c++) v[c] = 0.f;
                    if (oz < D && oy < H && ox < W) {
                        const size_t vo = ((size_t)oz * H + oy) * W + ox;
                        const unsigned *rw = (const unsigned *)(gq + (gri + vo) * gbw + co0 % gbw);
                        const float sc = mx_scale(gq[(size_t)N * mx_nb(Co) * Sg * gbw + gri + vo]);
#pragma unroll
                        for (int w = 0; w < BMo / 4; w++) {
                            const unsigned u = __ldg(rw + w);
                            const float2 d0 = dec_e4m3x2((unsigned short)(u & 0xffffu)), d1 = dec_e4m3x2((unsigned short)(u >> 16));
                            v[4 * w] = d0.x * sc; v[4 * w + 1] = d0.y * sc; v[4 * w + 2] = d1.x * sc; v[4 * w + 3] = d1.y * sc;
                        }
#pragma unroll
                        for (int c = 0; c < BMo; c++) if (co0 + c >= Co) v[c] = 0.f;
                    }
                    const uint64_t vid0 = (((uint64_t)n * Co * D + oz) * H + oy) * (uint64_t)W + ox;   /* element id of channel 0; + c * D H W */
                    /* in phases over the BMo channels (the block amaxes, the scales, then the rounding), so the channels' latency
                       chains overlap; the SR test is outside the loops and the bias sums stay in registers */
                    unsigned amc[BMo];
#pragma unroll
                    for (int c = 0; c < BMo; c++) amc[c] = __reduce_max_sync(0xffffffffu, __float_as_uint(v[c]) & 0x7fffffffu);
                    int ec[BMo];
#pragma unroll
                    for (int c = 0; c < BMo; c++) {
                        const unsigned u = __float_as_uint(__uint_as_float(amc[c]) * (1.f / 448.f));   /* mx_exp without its early return */
                        const int e = max(-126, min(126, (int)((u >> 23) & 0xff) - 127 + ((u & 0x7fffff) != 0)));
                        ec[c] = (u & 0x7fffffffu) > 0x7f800000u ? 128 : e;
                    }
                    if (sp.sr) {
#pragma unroll
                        for (int c = 0; c < BMo; c++)   /* same key as the per-block path */
                            sg[c * G8_CS + row * 16 + xx] = cvt_e4m3(sr_e4m3x(v[c] * exp2i(-ec[c]), sp.sr, vid0 + (uint64_t)(co0 + c) * Sg));
                    } else {
#pragma unroll
                        for (int c = 0; c < BMo; c++) sg[c * G8_CS + row * 16 + xx] = cvt_e4m3(v[c] * exp2i(-ec[c]));
                    }
                    if (lane == 0) {
#pragma unroll
                        for (int c = 0; c < BMo; c++) sgs[c * 8 + ks] = (uint8_t)(ec[c] + 127);
                    }
#pragma unroll
                    for (int c = 0; c < BMo; c++) bpart[c] += v[c];
                }
            }
        }
        /* GY: K block (co, ks) = 2 rows x 16 voxels = 8 float4; warp covers 4 blocks, 8 lanes each */
        if (!gdone) for (int task = warp; task < BMo * 2; task += 9) {
            int blk = task * 4 + (lane >> 3), c = blk >> 3, ks = blk & 7, f = lane & 7;
            int row = ks * 2 + (f >> 2), vx = 4 * (f & 3), vz = row >> 3, vy = row & 7;
            int co = co0 + c, oz = oz0 + vz, oy = oy0 + vy, ox = ox0 + vx;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (co < Co && oz < D && oy < H) {
                if constexpr (IS_MX8(TG)) {
                    const size_t vo = ((size_t)oz * H + oy) * W + ox, Sg = (size_t)D * H * W;
                    float q8[8];
                    if (ox < W) { ldmx8_8(gy, N, Co, Sg, n, co, vo, min(4, W - ox), vec, q8); v = make_float4(q8[0], q8[1], q8[2], q8[3]); }
                } else {
                const TG *src = gy + (((size_t)n * Co + co) * D + oz) * H * W + (size_t)oy * W + ox;
                if (vec) { if (ox < W) v = ldx4<TG>(src); }
                else { if (ox < W) v.x = ldx(src, 0); if (ox + 1 < W) v.y = ldx(src, 1); if (ox + 2 < W) v.z = ldx(src, 2); if (ox + 3 < W) v.w = ldx(src, 3); }
                }
            }
            float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))), sm = (v.x + v.y) + (v.z + v.w);
#pragma unroll
            for (int o = 4; o; o >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o)); sm += __shfl_xor_sync(0xffffffff, sm, o); }
            int e = mx_exp(amax, 1.f / 448.f);
            float m = exp2i(-e);
            float4 q = make_float4(v.x * m, v.y * m, v.z * m, v.w * m);
            if (sp.sr) {   /* stochastic rounding of the gradient operand, keyed by the element */
                const uint64_t vid = ((((uint64_t)n * Co + co) * D + oz) * H + oy) * (uint64_t)W + ox;
                q.x = sr_e4m3x(q.x, sp.sr, vid); q.y = sr_e4m3x(q.y, sp.sr, vid + 1);
                q.z = sr_e4m3x(q.z, sp.sr, vid + 2); q.w = sr_e4m3x(q.w, sp.sr, vid + 3);
            }
            *(unsigned *)(sg + c * G8_CS + row * 16 + vx) = cvt_e4m3x4(q.x, q.y, q.z, q.w);
            if (f == 0) { sgs[c * 8 + ks] = (uint8_t)(e + 127); if (do_bias) atomicAdd(&sbias[c], sm); }
        }
        __syncthreads();
        mma_zstep(oz0);
    }
    if (do_bias) {   /* the coop path's register partials (zero otherwise): one warp reduction per channel for the whole block */
        if constexpr (FAST) { if (warp < 8 && bacc != 0.f) atomicAdd(&sbias[lane % BMo], bacc); }
        else
#pragma unroll
        for (int c = 0; c < BMo; c++) {
            float sm = bpart[c];
#pragma unroll
            for (int o = 16; o; o >>= 1) sm += __shfl_xor_sync(0xffffffffu, sm, o);
            if (lane == 0 && sm != 0.f) atomicAdd(&sbias[c], sm);
        }
        __syncthreads();
        if (threadIdx.x < BMo && co0 + (int)threadIdx.x < Co) atomicAdd(&gb[co0 + threadIdx.x], sbias[threadIdx.x]);
    }
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

/* cooperative MX staging of the fp8 weight gradient (bit 0: x, bit 1: gy); 0 = the per-element paths (test hook) */
extern int g_f8w_coop;
template <int MT, int NT, typename T, typename TG> static void launch_bw8(dim3 grid, size_t smem, const void *x, shape5 xs, const TG *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp, int ZC) {
    static int attr[8][2];
    const int coop = g_f8w_coop | ((g_f8w_coop & 4) && xs.c <= 8 && (ys.c <= 32 || ys.c % 32 == 0) ? 2 : 0);   /* bit 2: the cooperative gy path for the
                                                                    network input conv (> 32 outputs: half-resolution nets, 32-output tiles) */
    if constexpr (NT == 1 && IS_MX(T) && IS_MX(TG)) {   /* the network input conv: conv_bwd_w_f8_k's FAST path (MX-fp8 or MX-fp4 gy) */
        if (xs.c <= 8 && !sp.up && !sp.x2 && x && (coop & 3) == 3 && (16 * MT <= mx_bw(ys.c) || (ys.c <= mx_bw(ys.c) && mx_bw(ys.c) < 16 * MT)) && ys.w % 4 == 0) {
            smem += 2 * 8 * NT * 4 + 16 + 256 * 16 * MT + 256;   /* plane multipliers, raw gy rows and scales */
            static int x4env = -1;
            if (x4env < 0) x4env = getenv("UFSM_F8W_X4") ? atoi(getenv("UFSM_F8W_X4")) : 1;   /* 0: stage all 8 tile channels (A/B) */
            if (xs.c <= 4 && x4env) {   /* the 4-channel network input: half the x staging */
                if (!(attr[cur_dev_()][1] & 1)) { attr[cur_dev_()][1] |= 1; cudaFuncSetAttribute((const void *)conv_bwd_w_f8_k<MT, NT, T, TG, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
                conv_bwd_w_f8_k<MT, NT, T, TG, 4><<<grid, 288, smem>>>((const T *)x, gy, gw, gb, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, gp, sp, ZC, coop);
                return;
            }
            if (!(attr[cur_dev_()][1] & 2)) { attr[cur_dev_()][1] |= 2; cudaFuncSetAttribute((const void *)conv_bwd_w_f8_k<MT, NT, T, TG, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
            conv_bwd_w_f8_k<MT, NT, T, TG, 8><<<grid, 288, smem>>>((const T *)x, gy, gw, gb, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, gp, sp, ZC, coop);
            return;
        }
    }
    if constexpr (IS_MX4(TG)) { fprintf(stderr, "lp_bwd_w_f8: MX-fp4 gradients only for the network input conv's FAST path\n"); abort(); }
    else {
    if (!attr[cur_dev_()][0]) { attr[cur_dev_()][0] = 1; cudaFuncSetAttribute((const void *)conv_bwd_w_f8_k<MT, NT, T, TG>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
    conv_bwd_w_f8_k<MT, NT, T, TG><<<grid, 288, smem>>>((const T *)x, gy, gw, gb, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, gp, sp, ZC, coop);
    }
}
template <typename T, typename TG> void bwd_w_f8_t(const void *x, shape5 xs, const TG *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp) {
    int MT = ys.c >= 32 ? 2 : 1;
    int NT = xs.c <= 8 ? 1 : MT == 1 && xs.c % 24 == 0 && !sp.up ? 3 : 2;   /* 24 input channels per block when cout = 16: gy restaged less (dec0.c1: -6%); fused up: 16-channel tiles */
    if (getenv("UFSM_F8_NT")) NT = atoi(getenv("UFSM_F8_NT"));
    if (getenv("UFSM_F8_MT")) MT = atoi(getenv("UFSM_F8_MT"));
    size_t smem = (size_t)8 * NT * X8_CS + 16 * MT * (G8_CS + 8) + 8 * NT * 4 + 16 * MT * 4 + 16 + 32 + (size_t)8 * NT * (sizeof(chan_t) + 8);   /* + MX x channel table, amax */
    /* z-steps per block: as many as possible (less halo restaging, fewer atomics) while keeping one full wave of blocks */
    int nzt = nblk_(ys.d, 2), base = (int)(((xs.c + 8 * NT - 1) / (8 * NT)) * ((ys.c + 16 * MT - 1) / (16 * MT)) * nblk_(ys.w, 16) * nblk_(ys.h, 8) * ys.n);
    const int zcap = xs.c <= 8 ? 48 : 12;   /* the network input conv: 48 z-steps (fewer halo restagings and gw atomics: P512 19.5 -> 18.2 ms) */
    int ZC = nzt < zcap ? nzt : zcap;
    while (ZC > 1 && (size_t)base * nblk_(nzt, ZC) < 72) ZC--;
    if (getenv("UFSM_F8_ZC")) ZC = atoi(getenv("UFSM_F8_ZC"));
    int nzc = (nzt + ZC - 1) / ZC;
    dim3 grid((unsigned)(nblk_(ys.w, 16) * nblk_(ys.h, 8) * nzc * ys.n), (xs.c + 8 * NT - 1) / (8 * NT), (ys.c + 16 * MT - 1) / (16 * MT));
    switch (MT * 10 + NT) {
    case 11: launch_bw8<1, 1, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, gp, sp, ZC); break;
    case 12: launch_bw8<1, 2, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, gp, sp, ZC); break;
    case 21: launch_bw8<2, 1, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, gp, sp, ZC); break;
    case 22: launch_bw8<2, 2, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, gp, sp, ZC); break;
    case 13: launch_bw8<1, 3, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, gp, sp, ZC); break;
    case 14: launch_bw8<1, 4, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, gp, sp, ZC); break;
    case 16: launch_bw8<1, 6, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, gp, sp, ZC); break;
    default: fprintf(stderr, "lp_bwd_w_f8: bad MT/NT %d/%d\n", MT, NT); abort();
    }
}
