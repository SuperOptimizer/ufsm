#pragma once
/* kernels and launchers of lp_f8wgrad (instantiated per type in lp_f8wgrad_i*.cu) */
#include "lp_common.cuh"

/* ======================= FP8 weight gradient, k=3, stride 1 =======================
   Block = 9 warps; warp w owns the tap row (kz, ky) = (w/3, w%3) and all three kx, so one pair of 32-bit loads of
   an input row feeds the B fragments of kx = 0, 1, 2 through byte funnel shifts. M = 16*MT output channels (GY),
   N = 8*NT input channels, K = 32 voxels (two 16-voxel output rows of the same z plane). A block walks ZC
   z-steps of 2 output planes (8 y x 16 x each), accumulating in registers, then atomically adds into gw.
   The input tile is a ring of 4 z planes (10 x 18 positions) per channel: each z-step stages only the 2 new planes.
   Both rows of a K block lie in the same input plane for every tap, so the B scale is per (channel, plane).
   Smem: X ring fp8 [ci][4 planes][10 rows][24 B], position p of a row at byte 3 + p so that x = ox0.. is 4-aligned
         (ci stride 1040 B = 260 words == 4 mod 32 -> conflict-free fragment loads), GY tile fp8 [co][16 rows][16 B] (co stride 272 B, ldmatrix conflict-free) + scales. */
template <int MT, int NT, typename T, typename TG>
__global__ void __launch_bounds__(288, BW8_MINB) conv_bwd_w_f8_k(const T *__restrict__ x, const TG *__restrict__ gy, float *__restrict__ gw, float *__restrict__ gb,
                                                       int N, int Ci, int D, int H, int W, int Co, gnp_t gp, split_t sp, int ZC, int coop) {
    constexpr int CH = 8 * NT, BMo = 16 * MT;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    uint8_t *sxq = smem_raw;                        /* [CH][X8_CS] */
    uint8_t *sg = sxq + CH * X8_CS;                 /* [BMo][G8_CS] */
    uint8_t *sgs = sg + BMo * G8_CS;                /* [BMo][8 ksteps] */
    uint8_t *sxs = sgs + BMo * 8;                   /* [CH][4 slots] */
    float *sbias = (float *)(sxs + CH * 4);         /* [BMo] */
    chan_t *ctab8 = (chan_t *)(((uintptr_t)(sbias + BMo) + 31) & ~(uintptr_t)31);   /* [CH] MX x: per-channel descriptors */
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
                    for (int q = 0; q < CH; q++) { const chan_t &cq = ctab8[q]; v[q] = cq.p ? act_ab(v[q], cq.a, cq.b, Gany && cq.g) : 0.f; }
                }
#pragma unroll
                for (int q = 0; q < CH; q++) { const unsigned a = __reduce_max_sync(0xffffffffu, __float_as_uint(v[q]) & 0x7fffffffu); if (lane == 0 && a) atomicMax(&am[q], a); }
                __syncthreads();
                if (tid < 180) {
                    uint8_t *dst = sxq + slot * X8_PS + row * 24 + 3 + p;
#pragma unroll
                    for (int q = 0; q < CH; q++) dst[q * X8_CS] = cvt_e4m3(v[q] * exp2i(-mx_exp(__uint_as_float(am[q]), 1.f / 448.f)));
                }
                if (tid < CH) { sxs[tid * 4 + slot] = (uint8_t)(mx_exp(__uint_as_float(am[tid]), 1.f / 448.f) + 127); amx[((pi + 1) & 1) * CH + tid] = 0u; }
            }
        } else
        /* X: warp per (channel, plane): 10 rows x (4 aligned float4 for x = ox0..ox0+15, plus the halo voxels
           ox0-1 and ox0+16); position p of a row is stored at byte 3 + p of its 24-byte slot */
        for (int task = warp; task < CH * np; task += 9) {
            int k = task / np, gz = gz_first + task % np, slot = (gz + 1) & 3, ci = ci0 + k;
            const bool ok = ci < Ci && gz >= 0 && gz < D;
            chan_t c = make_chan(ok ? ci : Ci, Ci, Cx, n, plane, x, sp, gp, N);
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
#pragma unroll
                    for (int c = 0; c < BMo; c++) {
                        const int e = mx_exp(__uint_as_float(__reduce_max_sync(0xffffffffu, __float_as_uint(v[c]) & 0x7fffffffu)), 1.f / 448.f);
                        float q = v[c] * exp2i(-e);
                        if (sp.sr) q = sr_e4m3(q, sr_hash(sp.sr, vid0 + (uint64_t)(co0 + c) * Sg));   /* same key as the per-block path */
                        sg[c * G8_CS + row * 16 + xx] = cvt_e4m3(q);
                        if (lane == 0) sgs[c * 8 + ks] = (uint8_t)(e + 127);
                        if (do_bias) {
                            float sm = v[c];
#pragma unroll
                            for (int o = 16; o; o >>= 1) sm += __shfl_xor_sync(0xffffffffu, sm, o);
                            if (lane == 0) atomicAdd(&sbias[c], sm);
                        }
                    }
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
                q.x = sr_e4m3(q.x, sr_hash(sp.sr, vid)); q.y = sr_e4m3(q.y, sr_hash(sp.sr, vid + 1));
                q.z = sr_e4m3(q.z, sr_hash(sp.sr, vid + 2)); q.w = sr_e4m3(q.w, sr_hash(sp.sr, vid + 3));
            }
            *(unsigned *)(sg + c * G8_CS + row * 16 + vx) = cvt_e4m3x4(q.x, q.y, q.z, q.w);
            if (f == 0) { sgs[c * 8 + ks] = (uint8_t)(e + 127); if (do_bias) atomicAdd(&sbias[c], sm); }
        }
        __syncthreads();
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
                    sa[m] = sgs[(m * 16 + g + 8 * (t & 1)) * 8 + ks];
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
    }
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

/* cooperative MX staging of the fp8 weight gradient (bit 0: x, bit 1: gy); 0 = the per-element paths (test hook) */
extern int g_f8w_coop;
template <int MT, int NT, typename T, typename TG> static void launch_bw8(dim3 grid, size_t smem, const void *x, shape5 xs, const TG *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp, int ZC) {
    static int attr[8];
    if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)conv_bwd_w_f8_k<MT, NT, T, TG>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
    conv_bwd_w_f8_k<MT, NT, T, TG><<<grid, 288, smem>>>((const T *)x, gy, gw, gb, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, gp, sp, ZC, g_f8w_coop);
}
template <typename T, typename TG> void bwd_w_f8_t(const void *x, shape5 xs, const TG *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp) {
    int MT = ys.c >= 32 ? 2 : 1;
    int NT = xs.c <= 8 ? 1 : MT == 1 && xs.c % 24 == 0 && !sp.up ? 3 : 2;   /* 24 input channels per block when cout = 16: gy restaged less (dec0.c1: -6%); fused up: 16-channel tiles */
    if (getenv("UFSM_F8_NT")) NT = atoi(getenv("UFSM_F8_NT"));
    if (getenv("UFSM_F8_MT")) MT = atoi(getenv("UFSM_F8_MT"));
    size_t smem = (size_t)8 * NT * X8_CS + 16 * MT * (G8_CS + 8) + 8 * NT * 4 + 16 * MT * 4 + 16 + 32 + (size_t)8 * NT * (sizeof(chan_t) + 8);   /* + MX x channel table, amax */
    /* z-steps per block: as many as possible (less halo restaging, fewer atomics) while keeping one full wave of blocks */
    int nzt = nblk_(ys.d, 2), base = (int)(((xs.c + 8 * NT - 1) / (8 * NT)) * ((ys.c + 16 * MT - 1) / (16 * MT)) * nblk_(ys.w, 16) * nblk_(ys.h, 8) * ys.n);
    int ZC = nzt < 12 ? nzt : 12;
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
