#pragma once
/* kernels and launchers of lp_f4wgrad (instantiated per type in lp_f4wgrad_i*.cu) */
#include "lp_common.cuh"

/* ======================= FP4 weight gradient, k=3, stride 1 =======================
   GW[co][ci][tap] = sum_v GY[co][v] X[ci][v + off(tap)] with m16n8k64 kind::mxf4 (e2m1 x e2m1, ue8m0 per 32 K, fp32
   accumulate). Same block / warp structure as conv_bwd_w_f8_k: 9 warps, warp w owns the tap row (kz, ky) = (w/3, w%3)
   and all three kx, M = 16 MT output channels (GY), N = 8 NT input channels, a block walks ZC z-steps of 2 output planes
   (8 y x 16 x each) and atomically adds into gw. K = 64 voxels = 4 output rows of 16 x of one z plane; each 32-voxel
   scale block is a PAIR of output rows (vy, vy + 1).
   A = GY: one ue8m0 per (co, row pair), exact stochastic rounding onto the e2m1 grid (sr_e2m1_nib: the nibble straight
         from floor(t(a) + u), one hash per 8 elements) keyed by the element; staged 8 values (one e2m1 word) per lane.
   B = X, round to nearest, in one of three layouts (template LY; host: UFSM_F4W_LAYOUT, the Hadamard / x-SR modes always
         use LY 0). A K block of X is the row pair (vy + ky, vy + ky + 1) seen through the kx shift.
     LY 2 (default): rows of 18 nibbles at 24 B (p at nibble 7 + p), one scale per (ci, plane) as the fp8 kernel; the tap
         shifts are funnel shifts and row offsets. One rounding per value: the only layout that is not slower than fp8.
     LY 1: the same rows stored twice, as 5 even-aligned (0-1 .. 8-9) and 4 odd-aligned (1-2 .. 7-8, used by ky = 1) row
         pairs, one scale per pair (36 positions incl. the halo): scale per (ci, 32 positions) as specified, ~10% slower.
     LY 0: every staged plane quantised into its 27 shifted blocks (each pair in the 3 kx windows, 16 B + one scale each,
         432 B / plane): exact per-32-position blocks, needed when the block is transformed (Hadamard) or SR'd.
     LY 3 (UFSM_F4_HAD_W=2, the fast Hadamard): transform I2 x H16 diag(s16) (H16 along x inside each row, rows untouched)
         on both operands, so each row's 3 kx windows are transformed once (30 windows per plane, not 27 x 32-blocks) and
         stored pre-shifted [kx][row][8 B] with one scale per (ci, plane); epilogue 1/16.
   MX inputs (mx4 / mx8 x, mx8 gy, the --fp4 training storage) are read 4 / 8 voxels at a time with the index math hoisted
   and the scale bytes in one vector load. x SR (UFSM_F4_SRX) uses the same direct-nibble rounding, one hash + xorshift.
   Optional fixed-sign Hadamard (had): every 32-element block of both operands is multiplied by H32 diag(s) (s: the fixed
         sign vector F4W_SGN) before its scale / rounding; the element order inside a block (k = 16 * row + x) is the same
         for both, so H^T H = 32 I gives the exact product * 32, undone in the epilogue.
   Smem: X ring e2m1 [ci][4 planes][27 blocks][16 B] (ci stride 1744 B = 436 words == 20 mod 32: the 8 channels x 4 words
   of a fragment load hit 32 banks), scale pairs [ci][4][32] u16 (byte 0 = block P, byte 1 = block P + 3 = the next pair of
   the same alignment; ci stride 260 B), GY e2m1 [co][8 pair blocks][16 B] (co stride 144 B, ldmatrix conflict-free) +
   scales [co][8], and per warp a [10 rows][18] fp32 staging plane (LY 0). LY 1 uses the same strides with 9 pairs x 2 rows x
   24 B per plane (pair P at 48 P, scale-pair byte 1 = pair P + 1); LY 2 10 rows x 24 B per plane, ci stride 976 B (also
   20 mod 32 words), which leaves room for NT 3 at 2 blocks / SM. */
#define X4_PS 432
#define X4_CS 1744
#define X4_SCS 130   /* u16 per channel of the scale-pair table (4 planes x 32 + 2 pad) */
#define G4_CS 144
#define GP4_REC 136   /* gy pre-pass record: 8 blocks x 16 B + 8 scale bytes per (n, co, z-step, 8-row tile, 16-x tile) */
#define F4W_SGN 0x9c6d2a73u
#define F4W_SGN16 0x2a73u   /* sign vector of the H16 variant (UFSM_F4_HAD_W=2), over the 16 x positions of a row window */
#ifndef F4W_PROF
#define F4W_PROF 0   /* 1: per-phase clock64 sums of conv_bwd_w_f4_k (x staging / gy staging / mma), printed per call when env F4W_PROF is set */
#endif
#if F4W_PROF
__device__ unsigned long long g_f4wprof[6];
#define FP_T(v) unsigned long long v = 0; if (threadIdx.x == 0) v = clock64();
#else
#define FP_T(v)
#endif
extern int g_f4w_coop;
extern int g_f4w_gypre_kb;
static int f4w_coop(void) {   /* read-only initialization for concurrent training on multiple GPUs */
    return g_f4w_coop >= 0 ? g_f4w_coop : getenv("UFSM_F4W_COOP_GY") ? ufsm_env_on("UFSM_F4W_COOP_GY") : 1;
}
/* stochastic rounding onto the e2m1 grid with a 16-bit uniform (two per hash): P(up) = ceil(frac * 65536) / 65536, so the
   bias is below 2^-16 of a grid step; branch-light (the grid step is 0.5 / 1 / 2 on [0, 2) / [2, 4) / [4, 6]). */
__device__ __forceinline__ float sr_e2m1_u16(float v, unsigned u16) {
    const float a = fminf(fabsf(v), 6.f);
    const float inv = a < 2.f ? 2.f : a < 4.f ? 1.f : 0.5f, fl = floorf(a * inv);
    const float up = (float)u16 < (a * inv - fl) * 65536.f ? 1.f : 0.f;
    return copysignf((fl + up) * __frcp_rn(inv), v);
}   /* fixed random sign vector of the weight-gradient Hadamard (bit k set: element k negated) */
__device__ __forceinline__ void had_lane(float &v, float p, bool hi) { v = hi ? p - v : v + p; }
template <int MT, int NT, int LY, typename T, typename TG>   /* LY 0: 27 shifted blocks, 1: row pairs (2 pairings), 2: rows, one scale per (ci, plane) */
__global__ void __launch_bounds__(288, 2) conv_bwd_w_f4_k(const T *__restrict__ x, const TG *__restrict__ gy, float *__restrict__ gw, float *__restrict__ gb,
                                                       int N, int Ci, int D, int H, int W, int Co, gnp_t gp, split_t sp, int ZC, int had, int coop,
                                                       const uint8_t *__restrict__ gpre, int zs0, int zs1) {
    constexpr int CH = 8 * NT, BMo = 16 * MT;
    constexpr int XPS = LY >= 2 ? 240 : X4_PS, XCS = LY >= 2 ? 976 : X4_CS;   /* x bytes per plane / per channel (LY 2: 10 rows x 24 B, LY 3: 3 kx x 10 rows x 8 B) */
    extern __shared__ __align__(128) unsigned char smem_raw[];
    uint8_t *sxq = smem_raw;                                          /* [CH][XCS] */
    unsigned short *sxp = (unsigned short *)(sxq + CH * XCS);       /* [CH][X4_SCS] */
    uint8_t *sg = (uint8_t *)(sxp + CH * X4_SCS);                     /* [BMo][G4_CS] */
    uint8_t *sgs = sg + BMo * G4_CS;                                  /* [BMo][8] */
    float *sbias = (float *)(sgs + BMo * 8);                          /* [BMo] */
    chan_t *ctab4 = (chan_t *)(smem_raw + (((unsigned char *)(sbias + BMo) - smem_raw + 31) & ~31));   /* offset from smem_raw: an integer round trip would make every access below a generic load */   /* [CH] MX x only */
    __nv_bfloat16 *xt = (__nv_bfloat16 *)(ctab4 + (IS_MX(T) ? CH : 0));             /* [CH][10 rows][24] MX x, LY 1-3: decoded plane, p at 3 + p */
    const int xt_len = IS_MX(T) && LY ? CH * 240 : 0;
    unsigned *ram = (unsigned *)(xt + (coop ? max(xt_len, BMo * 256) : xt_len));   /* decoded X / GY share one tile */
    uint8_t *tsc = (uint8_t *)(ram + (IS_MX(T) && LY == 1 ? CH * 10 : 0));   /* [CH][9] MX x, LY 1: pair scale bytes */
    uint8_t *wsc = tsc + (IS_MX(T) && LY == 1 ? (CH * 9 + 15) / 16 * 16 : 0);   /* [9 warps][32] */
    float *scr = (float *)(wsc + 9 * 32);                             /* [9 warps][180], LY 0 only */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int ci0 = blockIdx.y * CH, co0 = blockIdx.z * BMo;
    int bz = blockIdx.x;   /* spatial tiles use grid.x: grid.z is limited to 65535 */
    /* z-steps [zs0, nzt) of the volume (zs1 > 0: a slab [zs0, zs1), with gpre holding its pre-rounded gy blocks) */
    const int nxt = (W + 15) / 16, nyt = (H + 7) / 8, nzt = zs1 > 0 ? zs1 : (D + 1) / 2;
    const int ox0 = (bz % nxt) * 16; bz /= nxt;
    const int oy0 = (bz % nyt) * 8; bz /= nyt;
    const int nzc = (nzt - zs0 + ZC - 1) / ZC;
    const int zc = bz % nzc; const int n = bz / nzc;
    const size_t plane = (size_t)D * H * W;
    const int Cx = sp.x2 ? sp.c_split : Ci;
    const int kz = warp / 3, ky = warp % 3;
    const bool do_bias = gb && blockIdx.y == 0 && !gpre;   /* the gy pre-pass accumulates the bias itself */
    float *ws = scr + warp * 180;
    uint8_t *wscw = wsc + warp * 32;
    float acc[3][MT][NT][4];
#pragma unroll
    for (int a = 0; a < 3; a++) for (int m = 0; m < MT; m++) for (int q = 0; q < NT; q++) for (int k = 0; k < 4; k++) acc[a][m][q][k] = 0.f;
    if (threadIdx.x < BMo) sbias[threadIdx.x] = 0.f;
    const bool vec = (W & 3) == 0;
    const int zt_begin = zs0 + zc * ZC;
    bool mxu = false;
    const bool upt = sp.up && ci0 < Cx;   /* fused decoder upsample (see conv_bwd_w_f8_k) */
    if constexpr (IS_MX(T)) {
        if (threadIdx.x < CH) { const int ci = ci0 + threadIdx.x; ctab4[threadIdx.x] = make_chan(ci < Ci ? ci : Ci, Ci, Cx, n, upt ? plane / 8 : plane, x, sp, gp, N); }
        __syncthreads();
        mxu = LY != 0 && (had & 8) == 0 && ctab4[0].p != nullptr && (ctab4[0].nib & 7) == 0;   /* had bit 3: per-element path (test hook); LY 0
                                                                                                 keeps the per-channel reads (its tile would cost an SM block) */
#pragma unroll
        for (int q = 1; q < CH; q++) if (ctab4[q].p && (ctab4[q].sp != ctab4[0].sp || ctab4[q].nib != ctab4[0].nib + q)) mxu = false;
    }
    if (upt && !mxu) __trap();   /* only the tile path upsamples (host: LY 1-3, aligned 16-channel tiles) */
#if F4W_PROF
    unsigned long long pa = 0, pb_ = 0, pc = 0, sA = 0, sB = 0, tq = 0;   /* x / gy / mma phases; x: tile decode, LY 1 act + quantise */
#endif
    for (int zt = zt_begin; zt < nzt && zt < zt_begin + ZC; zt++) {
        const int oz0 = zt * 2;
        const int np = zt == zt_begin ? 4 : 2, gz_first = zt == zt_begin ? oz0 - 1 : oz0 + 1;
        __syncthreads();
        FP_T(t0)
        if (gpre) {   /* gy pre-pass: this z-step's 136-byte record per output channel (8 blocks + 8 scale bytes, gy_pre4_k) copied
                         asynchronously into sg / sgs (warp per channel, lane = word, lane 0 also the scales); waited for after the x staging */
            const int ZT = (min(2 * nzt, D) - 2 * zs0 + 1) / 2, TY = (H + 7) / 8, TX = (W + 15) / 16;
            for (int task = warp; task < BMo; task += 9) {
                const int co = co0 + task;
                const uint8_t *rec = gpre + (((((size_t)n * Co + (co < Co ? co : 0)) * ZT + (zt - zs0)) * TY + (oy0 >> 3)) * TX + (ox0 >> 4)) * GP4_REC;
                cp_async<4>(sg + task * G4_CS + 4 * lane, rec + 4 * lane, co < Co);
                if (lane == 0) cp_async<8>(sgs + task * 8, rec + 128, co < Co);
            }
        }
        /* X: warp per (channel, plane). Phase 1: the 10 x 18 positions (GN+SiLU applied) into the warp's fp32 plane,
           position p of a row = x ox0 - 1 + p. Phase 2: the 27 shifted 32-position blocks, 4 lanes x 8 values each. */
        constexpr int U = LY == 1 || LY == 2 ? 2 : 1;   /* x tasks in flight per warp: the loads of both are issued before either is quantised */
        /* MX x with the CH channels in one stored block row (mxu): plane by plane, the plane is first decoded thread-per-voxel into
           the bf16 tile xt (one row load per voxel for all CH channels; e2m1 / e4m3 x 2^e is exact in bf16), then the per-channel
           tasks read it and apply gn+silu as before (bit-identical to the per-element path) */
        for (int pass = 0; pass < (mxu ? np : 1); pass++) {
        if (mxu) {
            __syncthreads();
#if F4W_PROF
            if (threadIdx.x == 0) tq = clock64();
#endif
            const int tid = threadIdx.x, row = tid / 18, pp = tid % 18, gz = gz_first + pass, gyy = oy0 - 1 + row, gx = ox0 - 1 + pp;
            if (tid < 180) {
                float v[CH];
#pragma unroll
                for (int q = 0; q < CH; q++) v[q] = 0.f;
                if (gz >= 0 && gz < D && gyy >= 0 && gyy < H && gx >= 0 && gx < W) mx_rowch<T, CH>(ctab4[0], upt, gz, gyy, gx, D, H, W, v);   /* up: the
                    interpolated value is rounded to bf16 here (raw stored values are exact) */
#pragma unroll
                for (int q = 0; q < CH; q++) xt[(q * 10 + row) * 24 + 3 + pp] = __float2bfloat16(v[q]);   /* raw decoded value: exact in bf16 */
            }
            __syncthreads();
#if F4W_PROF
            if (threadIdx.x == 0) { unsigned long long t = clock64(); sA += t - tq; tq = t; }
#endif
        }
        if (LY == 1 && mxu) {   /* LY 1 from the tile: lane (channel q = 3 warp + lane / 10, row r = lane % 10) < 30: 18 values, row amax
                                   locally, the pair amaxes and the scale-pair table from the neighbouring rows' lanes (shuffles, no barrier);
                                   straight-line loads / activation / selects so the 18 values overlap (bit-identical to the per-value form) */
            const unsigned FM = 0xffffffffu;
            const int grp = lane / 10, r = lane - 10 * grp, q = 3 * warp + grp, gz = gz_first + pass, slot = (gz + 1) & 3, gyy = oy0 - 1 + r;
            const bool act = grp < 3 && q < CH;
            const chan_t &cq = ctab4[act ? q : 0];
            const bool rok = act && cq.p && gz >= 0 && gz < D && gyy >= 0 && gyy < H, G = (gp.G != 0 || sp.gp2.G != 0) && cq.g;
            const float ca = cq.a, cb = cq.b;
            const int jlo = ox0 == 0 ? 1 : 0, jhi = W - ox0 + 1;   /* x = ox0 - 1 + j inside [0, W) */
            const unsigned *src = (const unsigned *)(xt + ((act ? q : 0) * 10 + r) * 24 + 2);   /* words: bf16 2 .. 21 = positions -1 .. 18 */
            unsigned wv[10];
#pragma unroll
            for (int i = 0; i < 10; i++) wv[i] = src[i];
            float v[18];
            unsigned am = 0u;
#pragma unroll
            for (int j = 0; j < 18; j++) {
                const float xv = __uint_as_float((j & 1) ? wv[(j + 1) >> 1] << 16 : wv[j >> 1] & 0xffff0000u);   /* position j = bf16 3 + j */
                const float u = fmaf(xv, ca, cb), h = 0.5f * u;
                float th; asm("tanh.approx.f32 %0, %1;" : "=f"(th) : "f"(h));
                v[j] = rok && j >= jlo && j < jhi ? (G ? fmaf(h, th, h) : xv) : 0.f;
                am = amax_u(am, v[j]);
            }
            const unsigned amE = max(am, __shfl_xor_sync(FM, am, 1));                                   /* rows (2 P, 2 P + 1) */
            const unsigned amO = max(am, __shfl_sync(FM, am, (r & 1) ? min(lane + 1, 31) : max(lane - 1, 0)));   /* rows (2 P + 1, 2 P + 2), r = 1 .. 8 */
            int eP[2];
            uint8_t *dst = sxq + (act ? q : 0) * XCS + slot * XPS;
#pragma unroll
            for (int pz = 0; pz < 2; pz++) {
                const int e = mx_exp(__uint_as_float(pz ? amO : amE), 1.f / 6.f);
                eP[pz] = e;
                if (!act || (pz && (r < 1 || r > 8))) continue;
                const float m = exp2i(-e);
                const int pair = pz ? 5 + ((r - 1) >> 1) : r >> 1, rin = pz ? (r - 1) & 1 : r & 1;
                uint2 *o = (uint2 *)(dst + pair * 48 + rin * 24);
                o[0] = make_uint2(((unsigned)cvt_e2m1x2(v[0] * m, 0.f) & 15u) << 28, cvt_e2m1x8(v + 1, m));
                o[1] = make_uint2(cvt_e2m1x8(v + 9, m), (unsigned)cvt_e2m1x2(v[17] * m, 0.f) & 15u);
            }
            /* scale pairs (byte 0 = pair P, byte 1 = pair P + 1 of the same alignment, none after P = 4 / 8): even pair P = r / 2 from
               lane r = 2 P with pair P + 1 two lanes up; odd pair P = 5 + (r - 1) / 2 from lane r odd, likewise */
            const unsigned eb = (unsigned)(((r & 1) ? eP[1] : eP[0]) + 127) & 255u;
            const unsigned up = __shfl_down_sync(FM, eb, 2);
            if (act && r < 9) sxp[q * X4_SCS + slot * 32 + ((r & 1) ? 5 + ((r - 1) >> 1) : r >> 1)] = (unsigned short)(eb | (r < 7 ? up << 8 : 0u));
#if F4W_PROF
            __syncthreads();
            if (threadIdx.x == 0) { unsigned long long t = clock64(); sB += t - tq; tq = t; }
#endif
        } else {
        const int NXT = mxu ? CH : CH * np;
        for (int task0 = warp; task0 < NXT; task0 += 9 * U) {
            float4 V4[U][2]; float VS[U];
#pragma unroll
            for (int u = 0; u < U; u++) {
            const int task = task0 + 9 * u;
            int k = mxu ? task : task / np, gz = gz_first + (mxu ? pass : task % np), ci = ci0 + k;
            const bool ok = task < NXT && ci < Ci && gz >= 0 && gz < D;
            chan_t c = make_chan(ok ? ci : Ci, Ci, Cx, n, plane, x, sp, gp, N);
            const T *xc = ok && !IS_MX(T) ? (const T *)c.p + (size_t)gz * H * W : x;
            const bool G = (gp.G != 0 || sp.gp2.G != 0) && c.g, el = IS_MX(T);
            __syncwarp();
            float4 *v4 = V4[u]; float &vs = VS[u]; vs = 0.f;
#pragma unroll
            for (int i = 0; i < 2; i++) {
                int f = lane + 32 * i, row = f >> 2, gyy = oy0 - 1 + row, gx = ox0 + 4 * (f & 3);
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (mxu) {   /* from the decoded tile (zeros outside, gn+silu applied): 4 bf16 at 4 + 4 (f & 3) */
                    if (ok && f < 40 && gyy >= 0 && gyy < H) {
                        const uint2 u2 = *(const uint2 *)(xt + (k * 10 + row) * 24 + 4 + 4 * (f & 3));
                        const float2 a = __bfloat1622float2(*(const __nv_bfloat162 *)&u2.x), b = __bfloat1622float2(*(const __nv_bfloat162 *)&u2.y);
                        v.x = gx < W ? act_ab(a.x, c.a, c.b, G) : 0.f; v.y = gx + 1 < W ? act_ab(a.y, c.a, c.b, G) : 0.f;
                        v.z = gx + 2 < W ? act_ab(b.x, c.a, c.b, G) : 0.f; v.w = gx + 3 < W ? act_ab(b.y, c.a, c.b, G) : 0.f;
                    }
                } else if (ok && f < 40 && gyy >= 0 && gyy < H) {
                    const T *src = xc + (size_t)gyy * W + gx;
                    if constexpr (IS_MX(T)) { if (gx < W) v = ldc4_mx<T>(c, cof(c, gz, gyy, gx, H, W), min(4, W - gx), vec); }
                    else if (vec) { if (gx < W) v = ldx4(src); }
                    else { if (gx < W) v.x = ldx(src, 0); if (gx + 1 < W) v.y = ldx(src, 1); if (gx + 2 < W) v.z = ldx(src, 2); if (gx + 3 < W) v.w = ldx(src, 3); }
                    v.x = gx < W ? act_ab(v.x, c.a, c.b, G) : 0.f; v.y = gx + 1 < W ? act_ab(v.y, c.a, c.b, G) : 0.f;
                    v.z = gx + 2 < W ? act_ab(v.z, c.a, c.b, G) : 0.f; v.w = gx + 3 < W ? act_ab(v.w, c.a, c.b, G) : 0.f;
                }
                v4[i] = v;
                if ((LY == 0 || LY == 3) && f < 40) { float *d = ws + row * 18 + 1 + 4 * (f & 3); d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w; }
            }
            {   /* halo: lane < 20 -> row lane >> 1, side lane & 1 (x = ox0 - 1 or ox0 + 16) */
                int row = lane >> 1, gyy = oy0 - 1 + row, gx = (lane & 1) ? ox0 + 16 : ox0 - 1;
                if (mxu) { if (ok && lane < 20 && gyy >= 0 && gyy < H && gx >= 0 && gx < W) vs = act_ab(__bfloat162float(xt[(k * 10 + row) * 24 + ((lane & 1) ? 20 : 3)]), c.a, c.b, G); }
                else if (ok && lane < 20 && gyy >= 0 && gyy < H && gx >= 0 && gx < W) vs = act_ab(el ? ldc<T>(c, cof(c, gz, gyy, gx, H, W)) : ldx(xc, (size_t)gyy * W + gx), c.a, c.b, G);
                if ((LY == 0 || LY == 3) && lane < 20) ws[row * 18 + ((lane & 1) ? 17 : 0)] = vs;
            }
            }
#pragma unroll
            for (int u = 0; u < U; u++) {
            const int task = task0 + 9 * u;
            if (task >= NXT) break;
            const int k = mxu ? task : task / np, gz = gz_first + (mxu ? pass : task % np), slot = (gz + 1) & 3, ci = ci0 + k;
            const float4 *v4 = V4[u]; const float vs = VS[u];
            __syncwarp();
            uint8_t *dst = sxq + k * XCS + slot * XPS;
            if constexpr (LY == 3) {   /* H16 variant: lane (kx = lane / 10, row = lane % 10) < 30 transforms its 16-position window
                                      (x ox0 - 1 + kx ..) with H16 diag(s16), one scale per (ci, plane), stored pre-shifted [kx][row][8 B] */
                const unsigned FM = 0xffffffffu;
                const bool lok = lane < 30;
                const int kx = lane / 10, r = lane % 10;
                float v[16];
                const float *src = ws + (lok ? r : 0) * 18 + (lok ? kx : 0);
#pragma unroll
                for (int j = 0; j < 16; j++) v[j] = lok ? ((F4W_SGN16 >> j) & 1u ? -src[j] : src[j]) : 0.f;
#pragma unroll
                for (int st = 1; st < 16; st <<= 1)
#pragma unroll
                    for (int j = 0; j < 16; j++) if (!(j & st)) { float a0 = v[j], a1 = v[j | st]; v[j] = a0 + a1; v[j | st] = a0 - a1; }
                unsigned a = 0u;
#pragma unroll
                for (int j = 0; j < 16; j++) a = amax_u(a, v[j]);
#pragma unroll
                for (int o = 16; o; o >>= 1) a = max(a, __shfl_xor_sync(FM, a, o));
                const int e = mx_exp(__uint_as_float(a), 1.f / 6.f);
                const float m = exp2i(-e);
                uint2 u;
                if ((had & 2) && sp.sr) {   /* x SR: one hash + 7 xorshift steps per 16 values */
                    const uint64_t vid = ((((uint64_t)n * Ci + ci) * D + gz) * gridDim.x + blockIdx.x) * 32 + lane;
                    uint32_t h = sr_hash(sp.sr ^ 0x6a09e667u, vid);
                    unsigned w2[2] = {0u, 0u};
#pragma unroll
                    for (int j = 0; j < 16; j += 2) {
                        if (j) { h ^= h << 13; h ^= h >> 17; h ^= h << 5; }   /* xorshift32 after the hash (h != 0 w.p. 1 - 2^-32) */
                        w2[j >> 3] |= (sr_e2m1_nib(v[j] * m, h & 0xffffu) | sr_e2m1_nib(v[j + 1] * m, h >> 16) << 4) << (4 * (j & 7));
                    }
                    u = make_uint2(w2[0], w2[1]);
                } else u = make_uint2(cvt_e2m1x8(v, m), cvt_e2m1x8(v + 8, m));
                if (lok) *(uint2 *)(dst + (kx * 10 + r) * 8) = u;
                if (lane == 0) sxp[k * X4_SCS + slot * 32] = (unsigned short)((e + 127) * 0x101);
            } else
            if constexpr (LY == 2) {   /* rows at 24 B (p at nibble 7 + p), one scale per (ci, plane): one rounding per value, as fp8 */
                const unsigned FM = 0xffffffffu;
                unsigned a = amax_u(0u, vs);
#pragma unroll
                for (int i = 0; i < 2; i++) a = amax_u(amax_u(amax_u(amax_u(a, v4[i].x), v4[i].y), v4[i].z), v4[i].w);
#pragma unroll
                for (int o = 16; o; o >>= 1) a = max(a, __shfl_xor_sync(FM, a, o));
                const int e = mx_exp(__uint_as_float(a), 1.f / 6.f);
                const float m = exp2i(-e);
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    const int f = lane + 32 * i, r = f >> 2, cq = f & 3;
                    const unsigned h16 = (unsigned)cvt_e2m1x2(v4[i].x * m, v4[i].y * m) | ((unsigned)cvt_e2m1x2(v4[i].z * m, v4[i].w * m) << 8);
                    const unsigned hn = __shfl_down_sync(FM, h16, 1);
                    if (f < 40 && !(cq & 1)) *(unsigned *)(dst + r * 24 + 4 * (1 + (cq >> 1))) = h16 | (hn << 16);
                }
                if (lane < 20) { const unsigned nib = (unsigned)cvt_e2m1x2(vs * m, 0.f) & 15u; *(unsigned *)(dst + (lane >> 1) * 24 + ((lane & 1) ? 12 : 0)) = (lane & 1) ? nib : nib << 28; }
                if (lane == 0) sxp[k * X4_SCS + slot * 32] = (unsigned short)((e + 127) * 0x101);
            } else
            if constexpr (LY == 1) {   /* paired layout, from registers: lane (i, f = lane + 32 i < 40) holds row f >> 2, positions
                                     p = 1 + 4 (f & 3) .. 4 + 4 (f & 3) (= nibbles 8 + 4 c .., word 1 + c / 2); lane < 20 the halo
                                     of row lane >> 1 (p = 0: nibble 7 of word 0, p = 17: nibble 0 of word 3). Row amax -> the
                                     even-pair (rows 2e, 2e + 1) and odd-pair (2o + 1, 2o + 2) amax by shuffles; each value is
                                     rounded once per pairing. */
                const unsigned FM = 0xffffffffu;
                unsigned ra[2];
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    unsigned a = amax_u(amax_u(amax_u(amax_u(0u, v4[i].x), v4[i].y), v4[i].z), v4[i].w);
                    a = max(a, __shfl_xor_sync(FM, a, 1)); ra[i] = max(a, __shfl_xor_sync(FM, a, 2));
                }
                unsigned hm = amax_u(0u, vs); hm = max(hm, __shfl_xor_sync(FM, hm, 1));   /* halo amax of row lane >> 1 */
                ra[0] = max(ra[0], __shfl_sync(FM, hm, (lane >> 2) * 2));
                ra[1] = max(ra[1], __shfl_sync(FM, hm, min(16 + (lane >> 2) * 2, 31)));
                unsigned amE[2], amO[2];
#pragma unroll
                for (int i = 0; i < 2; i++) amE[i] = max(ra[i], __shfl_xor_sync(FM, ra[i], 4));
                {
                    const int r = lane >> 2;   /* i = 0 row */
                    const unsigned n0 = __shfl_sync(FM, ra[0], (r & 1) ? min(lane + 4, 31) : max(lane - 4, 0));
                    const unsigned x01 = __shfl_sync(FM, ra[1], lane & 3), x10 = __shfl_sync(FM, ra[0], 28 + (lane & 3));
                    amO[0] = max(ra[0], lane >= 28 ? x01 : n0);   /* row 7 pairs with row 8 (i = 1) */
                    amO[1] = max(ra[1], x10);                     /* row 8 (lanes 0..3) pairs with row 7; row 9 unused */
                }
                /* halo lanes: the pair amaxes of their row (row hr from lane 4 hr of i = 0, or 4 (hr - 8) of i = 1) */
                const int hr = min(lane >> 1, 9);
                const unsigned hE0 = __shfl_sync(FM, amE[0], (hr & 7) * 4), hE1 = __shfl_sync(FM, amE[1], (hr & 7) * 4);
                const unsigned hO0 = __shfl_sync(FM, amO[0], (hr & 7) * 4), hO1 = __shfl_sync(FM, amO[1], (hr & 7) * 4);
                const unsigned hamE = hr < 8 ? hE0 : hE1, hamO = hr < 8 ? hO0 : hO1;
#pragma unroll
                for (int pz = 0; pz < 2; pz++) {
#pragma unroll
                    for (int i = 0; i < 2; i++) {
                        const int f = lane + 32 * i, r = f >> 2, cq = f & 3;
                        const int e = mx_exp(__uint_as_float(pz ? amO[i] : amE[i]), 1.f / 6.f);
                        const float m = exp2i(-e);
                        const unsigned h16 = (unsigned)cvt_e2m1x2(v4[i].x * m, v4[i].y * m) | ((unsigned)cvt_e2m1x2(v4[i].z * m, v4[i].w * m) << 8);
                        const unsigned hn = __shfl_down_sync(FM, h16, 1);
                        const bool wr = f < 40 && (!pz || (r >= 1 && r <= 8));
                        const int pair = pz ? 5 + ((r - 1) >> 1) : r >> 1, rin = pz ? (r - 1) & 1 : r & 1;
                        if (wr && !(cq & 1)) *(unsigned *)(dst + pair * 48 + rin * 24 + 4 * (1 + (cq >> 1))) = h16 | (hn << 16);
                        if (wr && cq == 0 && rin == 0) wscw[pair] = (uint8_t)(e + 127);
                    }
                    {   /* halo */
                        const int r = hr, sd = lane & 1;
                        const int e = mx_exp(__uint_as_float(pz ? hamO : hamE), 1.f / 6.f);
                        const unsigned nib = (unsigned)cvt_e2m1x2(vs * exp2i(-e), 0.f) & 15u;
                        const bool wr = lane < 20 && (!pz || (r >= 1 && r <= 8));
                        const int pair = pz ? 5 + ((r - 1) >> 1) : r >> 1, rin = pz ? (r - 1) & 1 : r & 1;
                        if (wr) *(unsigned *)(dst + pair * 48 + rin * 24 + (sd ? 12 : 0)) = sd ? nib : nib << 28;
                    }
                }
                __syncwarp();
                if (lane < 9) sxp[k * X4_SCS + slot * 32 + lane] = (unsigned short)(wscw[lane] | (lane != 4 && lane < 8 ? (unsigned)wscw[lane + 1] << 8 : 0u));
            } else {
#pragma unroll
            for (int r4 = 0; r4 < 4; r4++) {
                const int task4 = r4 * 32 + lane, b = task4 >> 2, w = task4 & 3;
                const bool bok = b < 27;
                const int ev = b < 15, pr = ev ? b / 3 : (b - 15) / 3, kx = ev ? b % 3 : (b - 15) % 3, r0 = ev ? 2 * pr : 2 * pr + 1;
                float v[8];
                const float *src = ws + (r0 + (w >> 1)) * 18 + kx + 8 * (w & 1);
#pragma unroll
                for (int j = 0; j < 8; j++) v[j] = bok ? src[j] : 0.f;
                if (had & 1) {
#pragma unroll
                    for (int j = 0; j < 8; j++) if ((F4W_SGN >> (8 * w + j)) & 1u) v[j] = -v[j];
#pragma unroll
                    for (int s = 1; s < 8; s <<= 1)
#pragma unroll
                        for (int j = 0; j < 8; j++) if (!(j & s)) { float a0 = v[j], a1 = v[j | s]; v[j] = a0 + a1; v[j | s] = a0 - a1; }
#pragma unroll
                    for (int s = 1; s < 4; s <<= 1)
#pragma unroll
                        for (int j = 0; j < 8; j++) had_lane(v[j], __shfl_xor_sync(0xffffffffu, v[j], s), lane & s);
                }
                unsigned amu = 0u;
#pragma unroll
                for (int j = 0; j < 8; j++) amu = amax_u(amu, v[j]);
                amu = max(amu, __shfl_xor_sync(0xffffffffu, amu, 1)); amu = max(amu, __shfl_xor_sync(0xffffffffu, amu, 2));
                const int e = mx_exp(__uint_as_float(amu), 1.f / 6.f);
                unsigned word;
                if ((had & 2) && sp.sr) {   /* UFSM_F4_SRX: stochastic rounding of x too, keyed by (ci, plane, tile, block, k); one hash + 3 remixes */
                    const float mm = exp2i(-e);
                    const uint64_t vid = (((((uint64_t)n * Ci + ci) * D + gz) * gridDim.x + blockIdx.x) * 32 + b) * 32 + 8 * w;
                    uint32_t hh[4];
                    hh[0] = sr_hash(sp.sr ^ 0x6a09e667u, vid);
#pragma unroll
                    for (int i = 1; i < 4; i++) { uint32_t h1 = (hh[i - 1] ^ (hh[i - 1] >> 15)) * 0x2c1b3c6du; h1 ^= h1 >> 12; h1 *= 0x297a2d39u; hh[i] = h1 ^ (h1 >> 15); }
                    word = 0u;
#pragma unroll
                    for (int j = 0; j < 8; j++) word |= sr_e2m1_nib(v[j] * mm, (hh[j >> 1] >> (16 * (j & 1))) & 0xffffu) << (4 * j);
                } else word = cvt_e2m1x8(v, exp2i(-e));
                if (bok) { *(unsigned *)(dst + b * 16 + 4 * w) = word; if (w == 0) wscw[b] = (uint8_t)(e + 127); }
            }
            __syncwarp();
            if (lane < 27) sxp[k * X4_SCS + slot * 32 + lane] = (unsigned short)(wscw[lane] | (lane + 3 < 27 ? (unsigned)wscw[lane + 3] << 8 : 0u));
            }
            }
        }
        }   /* LY 1 from the tile / tasks */
        }   /* pass */
#if F4W_PROF
        __syncthreads();
#endif
        FP_T(t1)
        bool gy_coop = false;
        if constexpr (IS_MX8(TG)) {
            const int gbw = mx_bw(Co);
            if (coop && !gpre && (co0 % gbw) + BMo <= gbw) {
                gy_coop = true;
                __shared__ int gy_unsafe;
                if (threadIdx.x == 0) gy_unsafe = 0;
                __syncthreads();   /* X packing has consumed its decoded tile */
                if (threadIdx.x < 256) {
                    const int row = threadIdx.x / 16, oz = oz0 + row / 8, oy = oy0 + row % 8, ox = ox0 + threadIdx.x % 16;
                    const size_t Sg = (size_t)D * H * W;
                    const size_t vo = ((size_t)oz * H + oy) * W + ox;
                    const size_t ri = ((size_t)n * mx_nb(Co) + co0 / gbw) * Sg + vo;
                    const uint8_t *gq = (const uint8_t *)gy;
                    const bool ok = oz < D && oy < H && ox < W;
                    const unsigned scale = ok ? gq[(size_t)N * mx_nb(Co) * Sg * gbw + ri] : 127u;
                    /* Small scales can create FP32 subnormals before the BF16 store. The old
                       fused load/scale arithmetic must be retained for these blocks. */
                    if (ok && scale < 10u) atomicExch(&gy_unsafe, 1);
                    const float sc = ok ? mx_scale(scale) : 0.f;
#pragma unroll
                    for (int w = 0; w < BMo / 4; w++) {
                        const unsigned u = ok ? __ldg((const unsigned *)(gq + ri * gbw + co0 % gbw) + w) : 0u;
                        const float2 a = dec_e4m3x2((unsigned short)u), b = dec_e4m3x2((unsigned short)(u >> 16));
                        /* e4m3 times a power of two is exact in bf16 for finite training gradients. */
                        xt[(4 * w + 0) * 256 + threadIdx.x] = __float2bfloat16(co0 + 4 * w + 0 < Co ? a.x * sc : 0.f);
                        xt[(4 * w + 1) * 256 + threadIdx.x] = __float2bfloat16(co0 + 4 * w + 1 < Co ? a.y * sc : 0.f);
                        xt[(4 * w + 2) * 256 + threadIdx.x] = __float2bfloat16(co0 + 4 * w + 2 < Co ? b.x * sc : 0.f);
                        xt[(4 * w + 3) * 256 + threadIdx.x] = __float2bfloat16(co0 + 4 * w + 3 < Co ? b.y * sc : 0.f);
                    }
                }
                __syncthreads();
                gy_coop = gy_unsafe == 0;
            }
        }
        /* GY: block (co, pb) = row pair pb of the z-step (rows 2 pb, 2 pb + 1; vz = pb >> 2), 4 lanes x 8 values: lane f holds
           block elements 8 f .. 8 f + 7 (row f >> 1, x 8 (f & 1) ..) = one e2m1 word; a warp task covers 8 blocks */
        if (gpre) {
            cp_async_wait();   /* the barrier below publishes the copies */
        } else
        for (int task = warp; task < BMo; task += 9) {
            int blk = task * 8 + (lane >> 2), c = blk >> 3, pb = blk & 7, f = lane & 3;
            int row = pb * 2 + (f >> 1), vx = 8 * (f & 1), vz = row >> 3, vy = row & 7;
            int co = co0 + c, oz = oz0 + vz, oy = oy0 + vy, ox = ox0 + vx;
            float q[8];
#pragma unroll
            for (int j = 0; j < 8; j++) q[j] = 0.f;
            if (co < Co && oz < D && oy < H) {
                if constexpr (IS_MX8(TG)) {
                    if (gy_coop) {
                        const __nv_bfloat16 *src = xt + c * 256 + row * 16 + vx;
                        const uint4 u = *(const uint4 *)src;
                        q[0] = __uint_as_float(u.x << 16); q[1] = __uint_as_float(u.x & 0xffff0000u);
                        q[2] = __uint_as_float(u.y << 16); q[3] = __uint_as_float(u.y & 0xffff0000u);
                        q[4] = __uint_as_float(u.z << 16); q[5] = __uint_as_float(u.z & 0xffff0000u);
                        q[6] = __uint_as_float(u.w << 16); q[7] = __uint_as_float(u.w & 0xffff0000u);
                    } else {
                        const size_t vo = ((size_t)oz * H + oy) * W + ox, Sg = (size_t)D * H * W;
                        ldmx8_8(gy, N, Co, Sg, n, co, vo, min(8, W - ox), (W & 7) == 0, q);
                    }
                } else {
                    const TG *src = gy + (((size_t)n * Co + co) * D + oz) * H * W + (size_t)oy * W + ox;
                    if (vec) {
#pragma unroll
                        for (int h = 0; h < 2; h++) if (ox + 4 * h < W) { float4 v = ldx4<TG>(src + 4 * h); q[4 * h] = v.x; q[4 * h + 1] = v.y; q[4 * h + 2] = v.z; q[4 * h + 3] = v.w; }
                    } else {
#pragma unroll
                        for (int j = 0; j < 8; j++) if (ox + j < W) q[j] = ldx(src, j);
                    }
                }
            }
            float sm = ((q[0] + q[1]) + (q[2] + q[3])) + ((q[4] + q[5]) + (q[6] + q[7]));
            sm += __shfl_xor_sync(0xffffffffu, sm, 1); sm += __shfl_xor_sync(0xffffffffu, sm, 2);
            if (had & 4) {   /* H16 variant: H16 diag(s16) along x within each row (x = 8 (f & 1) + j) */
#pragma unroll
                for (int j = 0; j < 8; j++) if ((F4W_SGN16 >> (8 * (f & 1) + j)) & 1u) q[j] = -q[j];
#pragma unroll
                for (int s = 1; s < 8; s <<= 1)
#pragma unroll
                    for (int j = 0; j < 8; j++) if (!(j & s)) { float a0 = q[j], a1 = q[j | s]; q[j] = a0 + a1; q[j | s] = a0 - a1; }
#pragma unroll
                for (int j = 0; j < 8; j++) had_lane(q[j], __shfl_xor_sync(0xffffffffu, q[j], 1), lane & 1);
            } else if (had & 1) {
#pragma unroll
                for (int j = 0; j < 8; j++) if ((F4W_SGN >> (8 * f + j)) & 1u) q[j] = -q[j];
#pragma unroll
                for (int s = 1; s < 8; s <<= 1)
#pragma unroll
                    for (int j = 0; j < 8; j++) if (!(j & s)) { float a0 = q[j], a1 = q[j | s]; q[j] = a0 + a1; q[j | s] = a0 - a1; }
#pragma unroll
                for (int s = 1; s < 4; s <<= 1)
#pragma unroll
                    for (int j = 0; j < 8; j++) had_lane(q[j], __shfl_xor_sync(0xffffffffu, q[j], s), lane & s);
            }
            unsigned amu = 0u;
#pragma unroll
            for (int j = 0; j < 8; j++) amu = amax_u(amu, q[j]);
            amu = max(amu, __shfl_xor_sync(0xffffffffu, amu, 1)); amu = max(amu, __shfl_xor_sync(0xffffffffu, amu, 2));
            const int e = mx_exp(__uint_as_float(amu), 1.f / 6.f);
            const float m = exp2i(-e);
            unsigned word;
            if (sp.sr) {   /* exact stochastic rounding of the gradient operand, keyed by the element (block slot): one hash + 3 remixes */
                const uint64_t vid = ((((uint64_t)n * Co + co) * D + oz) * H + oy) * (uint64_t)W + ox;
                uint32_t hh[4];
                sr_hash4(sp.sr, vid, hh);
                word = 0u;
#pragma unroll
                for (int j = 0; j < 8; j++) word |= sr_e2m1_nib(q[j] * m, (hh[j >> 1] >> (16 * (j & 1))) & 0xffffu) << (4 * j);
            } else word = cvt_e2m1x8(q, m);
            *(unsigned *)(sg + c * G4_CS + pb * 16 + 4 * f) = word;
            if (f == 0) { sgs[c * 8 + pb] = (uint8_t)(e + 127); if (do_bias) atomicAdd(&sbias[c], sm); }
        }
        __syncthreads();
        FP_T(t2)
#pragma unroll
        for (int vz = 0; vz < 2; vz++) {
            const int slot = (oz0 + vz + kz) & 3;
#pragma unroll
            for (int kk = 0; kk < 2; kk++) {
                const int ks = vz * 2 + kk, r0 = 4 * kk + ky;
                const int P0 = (ky & 1) ? 15 + ((r0 - 1) >> 1) * 3 : (r0 >> 1) * 3;   /* first block (kx = 0) of the K step's row pairs */
                unsigned af[MT][4], sa[MT];
#pragma unroll
                for (int m = 0; m < MT; m++) {
                    int mat = lane >> 3, co = m * 16 + (mat & 1) * 8 + (lane & 7);
                    ldsm_x4(af[m], sg + co * G4_CS + (2 * ks + (mat >> 1)) * 16);
                    sa[m] = *(const unsigned short *)(sgs + (m * 16 + g + 8 * (t & 1)) * 8 + 2 * ks);
                }
                if constexpr (LY == 3) {
#pragma unroll
                    for (int q = 0; q < NT; q++) {
                        const uint8_t *xb = sxq + (q * 8 + g) * XCS + slot * XPS + (r0 + (t >> 1)) * 8 + 4 * (t & 1);
                        const unsigned sb = sxp[(q * 8 + g) * X4_SCS + slot * 32];
#pragma unroll
                        for (int kx = 0; kx < 3; kx++) {
                            unsigned b[2] = {*(const unsigned *)(xb + kx * 80), *(const unsigned *)(xb + kx * 80 + 16)};   /* rows + 2: +16 B */
#pragma unroll
                            for (int m = 0; m < MT; m++) mma_f4(acc[kx][m][q], af[m], b, sa[m], sb);
                        }
                    }
                } else
                if constexpr (LY >= 1) {
                    const int P = LY == 2 ? 0 : (ky & 1) ? 5 + ((r0 - 1) >> 1) : r0 >> 1;   /* LY 1: the K step's first row pair; the second is P + 1 */
                    const int ro = LY == 2 ? r0 * 24 : P * 48;
#pragma unroll
                    for (int q = 0; q < NT; q++) {
                        const unsigned *xr = (const unsigned *)(sxq + (q * 8 + g) * XCS + slot * XPS + ro + (t >> 1) * 24) + (t & 1);
                        const unsigned w0 = xr[0], w1 = xr[1], w2 = xr[2], c0 = xr[12], c1 = xr[13], c2 = xr[14];   /* next pair: +48 B */
                        const unsigned sb = sxp[(q * 8 + g) * X4_SCS + slot * 32 + P];
                        unsigned b[3][2] = {{__funnelshift_r(w0, w1, 28), __funnelshift_r(c0, c1, 28)}, {w1, c1}, {__funnelshift_r(w1, w2, 4), __funnelshift_r(c1, c2, 4)}};
#pragma unroll
                        for (int kx = 0; kx < 3; kx++)
#pragma unroll
                            for (int m = 0; m < MT; m++) mma_f4(acc[kx][m][q], af[m], b[kx], sa[m], sb);
                    }
                } else
#pragma unroll
                for (int q = 0; q < NT; q++) {
                    const uint8_t *xb = sxq + (q * 8 + g) * XCS + slot * XPS + 4 * t;
                    const unsigned short *xs2 = sxp + (q * 8 + g) * X4_SCS + slot * 32;
#pragma unroll
                    for (int kx = 0; kx < 3; kx++) {
                        const int P = P0 + kx;
                        unsigned b[2] = {*(const unsigned *)(xb + P * 16), *(const unsigned *)(xb + (P + 3) * 16)};
                        const unsigned sb = xs2[P];
#pragma unroll
                        for (int m = 0; m < MT; m++) mma_f4(acc[kx][m][q], af[m], b, sa[m], sb);
                    }
                }
            }
        }
#if F4W_PROF
        __syncthreads();
        if (threadIdx.x == 0) { unsigned long long t3 = clock64(); pa += t1 - t0; pb_ += t2 - t1; pc += t3 - t2; }
#endif
    }
#if F4W_PROF
    if (threadIdx.x == 0) { atomicAdd(&g_f4wprof[0], pa); atomicAdd(&g_f4wprof[1], pb_); atomicAdd(&g_f4wprof[2], pc); atomicAdd(&g_f4wprof[3], 1ull);
                            atomicAdd(&g_f4wprof[4], sA); atomicAdd(&g_f4wprof[5], sB); }
#endif
    if (do_bias) { __syncthreads(); if (threadIdx.x < BMo && co0 + (int)threadIdx.x < Co) atomicAdd(&gb[co0 + threadIdx.x], sbias[threadIdx.x]); }
    const float osc = (had & 4) ? 1.f / 16.f : (had & 1) ? 1.f / 32.f : 1.f;
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
                    if (ci < Ci) atomicAdd(&gw[((size_t)co * Ci + ci) * 27 + tap], acc[kx][m][q][2 * h] * osc);
                    if (ci + 1 < Ci) atomicAdd(&gw[((size_t)co * Ci + ci + 1) * 27 + tap], acc[kx][m][q][2 * h + 1] * osc);
                }
            }
    }
}
/* ---- warp-specialised fp4 weight gradient (the training configuration: MX x whose CH channels sit in one stored block row,
   LY 1, no transform modes, gy from the pre-pass records, no fused upsample): the staging of conv_bwd_w_f4_k ran between
   block barriers with 5-6 of its 9 warps busy, ~4x longer than its work. Here 8 producer warps stage z-step i + 1 (both new
   x planes: decode tile, then the LY 1 lanes; the gy records by cp.async) while the 9 MMA warps (as conv_bwd_w_f4_k: warp =
   tap row) consume z-step i. Persistent: one block per SM walks (16 x 8 tile, z chunk) items of its channel tile, i counting
   z-steps across items, and keeps accumulating in registers, so the per-block fill, drain and gw atomics (~20% of a
   20-z-step block) happen once per SM. The x planes live in a ring of 8 slots, numbered by a running plane count (an item
   with s z-steps holds 2 s + 2 planes), so the slots a z-step stages were last read three z-steps before, also across an
   item boundary (whose first z-step stages 4 planes); gy in two buffers; named barriers FULL[i & 1] (producers arrive,
   consumers wait) and EMPTY[i & 1] (consumers arrive after their MMA, producers wait before staging i + 2). One block per
   SM (~89 KB smem); 672 threads get 80 registers each. Same values as conv_bwd_w_f4_k with gpre; the gw sums group
   differently (per block over several tiles). ---- */
#define WS_NC 9            /* consumer (MMA) warps */
#define WS_NP 12           /* producer warps */
#define WS_NT ((WS_NC + WS_NP) * 32)
#define WS_PS 432          /* x bytes per (channel, plane slot): LY 1, 9 row pairs x 2 rows x 24 B */
#define WS_NS 8            /* x plane slots */
#define WS_CS 3536         /* per channel: 8 slots x 432 B + 80 (884 words == 20 mod 32, as X4_CS) */
#define WS_SCS 258         /* u16 per channel of the scale-pair table: 8 slots x 32 + 2 */
__device__ __forceinline__ void ws_bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void ws_bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(n) : "memory"); }
static int ws_nsm(void) {   /* SMs of the current device */
    static int n[8];
    const int d = cur_dev_();
    if (!n[d]) { int dev = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&n[d], cudaDevAttrMultiProcessorCount, dev); if (n[d] < 1) n[d] = 1; }
    return n[d];
}
template <int MT, int NT> __host__ __device__ constexpr size_t ws_smem() {
    return (size_t)8 * NT * WS_CS + (size_t)8 * NT * WS_SCS * 2 + 2 * (size_t)16 * MT * (G4_CS + 8) + 32 * 32 + (size_t)2 * 8 * NT * 240 * 2 + 64;
}
template <int MT, int NT, typename T>
__global__ void __launch_bounds__(WS_NT, 1) conv_bwd_w_f4ws_k(const T *__restrict__ x, float *__restrict__ gw, int N, int Ci, int D, int H, int W, int Co,
                                                            gnp_t gp, split_t sp, int ZC, const uint8_t *__restrict__ gpre, int zs0, int zs1) {
    constexpr int CH = 8 * NT, BMo = 16 * MT;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    uint8_t *sxq = smem_raw;                                              /* [CH][WS_CS] */
    unsigned short *sxp = (unsigned short *)(sxq + CH * WS_CS);           /* [CH][WS_SCS] */
    uint8_t *sgb = (uint8_t *)(sxp + CH * WS_SCS);                        /* [2][BMo][G4_CS] then [2][BMo][8] */
    uint8_t *sgsb = sgb + 2 * BMo * G4_CS;
    chan_t *ctab4 = (chan_t *)(smem_raw + (((sgsb + 2 * BMo * 8) - smem_raw + 31) & ~31));   /* [CH] */
    __nv_bfloat16 *xt = (__nv_bfloat16 *)(ctab4 + 32);                    /* [2 planes][CH][10 rows][24] */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int ci0 = blockIdx.y * CH, co0 = blockIdx.z * BMo;
    const int nxt = (W + 15) / 16, nyt = (H + 7) / 8, nzt = zs1;
    const int nzc = (nzt - zs0 + ZC - 1) / ZC, nitem = nxt * nyt * nzc, nper = gridDim.x / N;
    const int n = blockIdx.x / nper, w0 = blockIdx.x % nper;   /* items w0, w0 + nper, ... of batch entry n */
    const size_t plane = (size_t)D * H * W;
    const int Cx = sp.x2 ? sp.c_split : Ci;
    int ntot = 0;   /* z-steps of this block */
    for (int w = w0; w < nitem; w += nper) { const int zb = zs0 + (w / (nxt * nyt)) * ZC; ntot += min(nzt, zb + ZC) - zb; }
    if (threadIdx.x < CH) { const int ci = ci0 + threadIdx.x; ctab4[threadIdx.x] = make_chan(ci < Ci ? ci : Ci, Ci, Cx, n, plane, x, sp, gp, N); }
    __syncthreads();
    {
        bool mxu = ctab4[0].p != nullptr && (ctab4[0].nib & 7) == 0;
#pragma unroll
        for (int q = 1; q < CH; q++) if (ctab4[q].p && (ctab4[q].sp != ctab4[0].sp || ctab4[q].nib != ctab4[0].nib + q)) mxu = false;
        if (!mxu) __trap();   /* the host only sends aligned MX channel tiles */
    }
    if (warp >= WS_NC) {   /* ================= producers ================= */
        const int pt = threadIdx.x - WS_NC * 32, pw = warp - WS_NC;   /* 0 .. 255, 0 .. 7 */
        const int ZT = (min(2 * nzt, D) - 2 * zs0 + 1) / 2, TY = (H + 7) / 8, TX = (W + 15) / 16;
        const bool Gany = gp.G != 0 || sp.gp2.G != 0;
        int it = 0, pbase = 0;   /* z-steps and planes of the earlier items */
        for (int w = w0; w < nitem; w += nper) {
        const int ox0 = (w % nxt) * 16, oy0 = ((w / nxt) % nyt) * 8, zt_begin = zs0 + (w / (nxt * nyt)) * ZC, nzs = min(nzt, zt_begin + ZC) - zt_begin;
        const int pz0 = 2 * zt_begin - 1 - pbase;   /* plane gz lives in slot (gz - pz0) % WS_NS */
        for (int i = 0; i < nzs; i++, it++) {
            const int zt = zt_begin + i, oz0 = 2 * zt, b = it & 1;
            if (it >= 2) ws_bar_sync(3 + b, WS_NT);   /* EMPTY[b]: z-step it - 2 is consumed (its gy buffer; the slots staged now were read by it - 3) */
            /* gy records of z-step i -> buffer b (waited for at the end) */
            for (int task = pw; task < BMo; task += WS_NP) {
                const int co = co0 + task;
                const uint8_t *rec = gpre + (((((size_t)n * Co + (co < Co ? co : 0)) * ZT + (zt - zs0)) * TY + (oy0 >> 3)) * TX + (ox0 >> 4)) * GP4_REC;
                cp_async<4>(sgb + (b * BMo + task) * G4_CS + 4 * lane, rec + 4 * lane, co < Co);
                if (lane == 0) cp_async<8>(sgsb + (b * BMo + task) * 8, rec + 128, co < Co);
            }
            const int np = i == 0 ? 4 : 2, gz_first = i == 0 ? oz0 - 1 : oz0 + 1;
            for (int pp0 = 0; pp0 < np; pp0 += 2) {   /* two planes per pass (the decode tile holds two) */
                ws_bar_sync(5, WS_NP * 32);   /* the previous pass's LY 1 lanes have read the tile */
                for (int idx = pt; idx < 2 * 180; idx += WS_NP * 32) {   /* decode: thread per voxel, all CH channels */
                    const int pl = idx / 180, rem = idx % 180, row = rem / 18, pp = rem % 18;
                    const int gz = gz_first + pp0 + pl, gyy = oy0 - 1 + row, gx = ox0 - 1 + pp;
                    float v[CH];
#pragma unroll
                    for (int q = 0; q < CH; q++) v[q] = 0.f;
                    if (gz >= 0 && gz < D && gyy >= 0 && gyy < H && gx >= 0 && gx < W) mx_rowch<T, CH>(ctab4[0], false, gz, gyy, gx, D, H, W, v);
                    __nv_bfloat16 *d = xt + (size_t)pl * CH * 240;
#pragma unroll
                    for (int q = 0; q < CH; q++) d[(q * 10 + row) * 24 + 3 + pp] = __float2bfloat16(v[q]);
                }
                ws_bar_sync(5, WS_NP * 32);
                /* LY 1 lanes: (plane, channel) groups of 10 rows, 3 per warp (as conv_bwd_w_f4_k) */
                for (int grp0 = pw * 3; grp0 < 2 * CH; grp0 += WS_NP * 3) {
                    const unsigned FM = 0xffffffffu;
                    const int gi = lane / 10, r = lane - 10 * gi, grp = grp0 + gi, pl = grp / CH, q = grp % CH;
                    const int gz = gz_first + pp0 + pl, slot = (gz - pz0) % WS_NS, gyy = oy0 - 1 + r;
                    const bool act = gi < 3 && grp < 2 * CH;
                    const chan_t &cq = ctab4[act ? q : 0];
                    const bool rok = act && cq.p && gz >= 0 && gz < D && gyy >= 0 && gyy < H, G = Gany && cq.g;
                    const float ca = cq.a, cb = cq.b;
                    const int jlo = ox0 == 0 ? 1 : 0, jhi = W - ox0 + 1;
                    const unsigned *src = (const unsigned *)(xt + (size_t)(act ? pl : 0) * CH * 240 + ((act ? q : 0) * 10 + r) * 24 + 2);
                    unsigned wv[10];
#pragma unroll
                    for (int k = 0; k < 10; k++) wv[k] = src[k];
                    float v[18];
                    unsigned am = 0u;
#pragma unroll
                    for (int j = 0; j < 18; j++) {
                        const float xv = __uint_as_float((j & 1) ? wv[(j + 1) >> 1] << 16 : wv[j >> 1] & 0xffff0000u);
                        const float u = fmaf(xv, ca, cb), h = 0.5f * u;
                        float th; asm("tanh.approx.f32 %0, %1;" : "=f"(th) : "f"(h));
                        v[j] = rok && j >= jlo && j < jhi ? (G ? fmaf(h, th, h) : xv) : 0.f;
                        am = amax_u(am, v[j]);
                    }
                    const unsigned amE = max(am, __shfl_xor_sync(FM, am, 1));
                    const unsigned amO = max(am, __shfl_sync(FM, am, (r & 1) ? min(lane + 1, 31) : max(lane - 1, 0)));
                    int eP[2];
                    uint8_t *dst = sxq + (act ? q : 0) * WS_CS + slot * WS_PS;
#pragma unroll
                    for (int pz = 0; pz < 2; pz++) {
                        const int e = mx_exp(__uint_as_float(pz ? amO : amE), 1.f / 6.f);
                        eP[pz] = e;
                        if (!act || (pz && (r < 1 || r > 8))) continue;
                        const float m = exp2i(-e);
                        const int pair = pz ? 5 + ((r - 1) >> 1) : r >> 1, rin = pz ? (r - 1) & 1 : r & 1;
                        uint2 *o = (uint2 *)(dst + pair * 48 + rin * 24);
                        o[0] = make_uint2(((unsigned)cvt_e2m1x2(v[0] * m, 0.f) & 15u) << 28, cvt_e2m1x8(v + 1, m));
                        o[1] = make_uint2(cvt_e2m1x8(v + 9, m), (unsigned)cvt_e2m1x2(v[17] * m, 0.f) & 15u);
                    }
                    const unsigned eb = (unsigned)(((r & 1) ? eP[1] : eP[0]) + 127) & 255u;
                    const unsigned up = __shfl_down_sync(FM, eb, 2);
                    if (act && r < 9) sxp[q * WS_SCS + slot * 32 + ((r & 1) ? 5 + ((r - 1) >> 1) : r >> 1)] = (unsigned short)(eb | (r < 7 ? up << 8 : 0u));
                }
            }
            cp_async_wait();
            ws_bar_arrive(1 + b, WS_NT);   /* FULL[b]: z-step it is staged */
        }
        pbase += 2 * nzs + 2;
        }
        return;
    }
    /* ================= consumers: as conv_bwd_w_f4_k (LY 1) ================= */
    const int kz = warp / 3, ky = warp % 3;
    float acc[3][MT][NT][4];
#pragma unroll
    for (int a = 0; a < 3; a++) for (int m = 0; m < MT; m++) for (int q = 0; q < NT; q++) for (int k = 0; k < 4; k++) acc[a][m][q][k] = 0.f;
    int it = 0, pbase = 0;
    for (int w = w0; w < nitem; w += nper) {
    const int zt_begin = zs0 + (w / (nxt * nyt)) * ZC, nzs = min(nzt, zt_begin + ZC) - zt_begin;
    for (int i = 0; i < nzs; i++, it++) {
        const int b = it & 1, s0 = pbase + 2 * i;   /* the slot index of plane oz0 - 1 (before the modulo) */
        const uint8_t *sg = sgb + b * BMo * G4_CS, *sgs = sgsb + b * BMo * 8;
        ws_bar_sync(1 + b, WS_NT);   /* FULL[b] */
#pragma unroll
        for (int vz = 0; vz < 2; vz++) {
            const int slot = (s0 + vz + kz) % WS_NS;   /* plane oz0 + vz + kz - 1 */
#pragma unroll
            for (int kk = 0; kk < 2; kk++) {
                const int ks = vz * 2 + kk, r0 = 4 * kk + ky;
                unsigned af[MT][4], sa[MT];
#pragma unroll
                for (int m = 0; m < MT; m++) {
                    int mat = lane >> 3, co = m * 16 + (mat & 1) * 8 + (lane & 7);
                    ldsm_x4(af[m], sg + co * G4_CS + (2 * ks + (mat >> 1)) * 16);
                    sa[m] = *(const unsigned short *)(sgs + (m * 16 + g + 8 * (t & 1)) * 8 + 2 * ks);
                }
                const int P = (ky & 1) ? 5 + ((r0 - 1) >> 1) : r0 >> 1;
#pragma unroll
                for (int q = 0; q < NT; q++) {
                    const unsigned *xr = (const unsigned *)(sxq + (q * 8 + g) * WS_CS + slot * WS_PS + P * 48 + (t >> 1) * 24) + (t & 1);
                    const unsigned w0 = xr[0], w1 = xr[1], w2 = xr[2], c0 = xr[12], c1 = xr[13], c2 = xr[14];
                    const unsigned sb = sxp[(q * 8 + g) * WS_SCS + slot * 32 + P];
                    unsigned bb[3][2] = {{__funnelshift_r(w0, w1, 28), __funnelshift_r(c0, c1, 28)}, {w1, c1}, {__funnelshift_r(w1, w2, 4), __funnelshift_r(c1, c2, 4)}};
#pragma unroll
                    for (int kx = 0; kx < 3; kx++)
#pragma unroll
                        for (int m = 0; m < MT; m++) mma_f4(acc[kx][m][q], af[m], bb[kx], sa[m], sb);
                }
            }
        }
        if (it + 2 < ntot) ws_bar_arrive(3 + b, WS_NT);   /* EMPTY[b] (producers wait for it before staging z-step it + 2) */
    }
    pbase += 2 * nzs + 2;
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
/* gy pre-pass of the fp4 weight gradient (MX-fp8 gy, stochastic rounding, plain mode): the A operand exactly as the kernel's
   staging forms it, once per layer instead of in every block's z-step. Block (n, co, z, row pair yp, 16-x block xb) = rows
   2 yp, 2 yp + 1 x 16 xb ..; lane word f = row f >> 1, x 8 (f & 1) .. + 7 (zero outside the volume); ue8m0 of the block amax
   / 6 (byte e + 127) and exact SR keyed by the word's first element, as in the kernel. Covers the planes [z0, z0 + Zs) of
   one slab, laid out as the kernel copies it: one 136-byte record per (n, co, z-step zt, 8-row tile ty, 16-x tile xb) =
   its 8 blocks pb = 4 (z & 1) + (yp & 3) (16 B each) then their 8 scale bytes, blocks past the volume zero (scale byte 1).
   gb: += the sum of gy over those planes.
   A warp per tile (32 voxels x one MX channel block: lane k decodes voxel k's row), tiles strided over all warps of the grid,
   only warp-level synchronisation; the BW channels x 4 words are BW / 8 tasks per lane (the 4 words of a block in 4
   consecutive lanes for the amax shuffles). The bias sums stay in registers until the end (block-reduced, one atomic per
   channel and CTA). */
extern int g_f4w_gy4;   /* lp_bwd_w_f4: the (mx8_t-typed) gy holds MX-fp4 rows; only the gy pre-pass reads it */
template <int BW, int GB = 8> __global__ void __launch_bounds__(256) gy_pre4_k(const uint8_t *__restrict__ gy, uint8_t *__restrict__ out, float *__restrict__ gb,
                                                                  int N, int Co, int D, int H, int W, int z0, int Zs, unsigned sr) {   /* GB: gy MX bits */
    constexpr int NTASK = BW / 8;   /* (channel, word) tasks per lane */
    __shared__ float v[8][BW][33];  /* per warp: [channel][block element k = 16 row + x] */
    __shared__ __align__(16) uint8_t recb[BW][GP4_REC];   /* the CTA's records (one per channel), written out whole */
    __shared__ float bred[BW];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nb = mx_nb(Co), cb = blockIdx.y % nb, n = blockIdx.y / nb;
    const int Wb = (W + 15) / 16, ZT = (Zs + 1) / 2, TY = (H + 7) / 8, Yb = 4 * TY;   /* padded block grid: 2 ZT planes x 4 TY row pairs */
    const size_t Sg = (size_t)D * H * W;
    const unsigned ngrp = (unsigned)ZT * TY * Wb;   /* record groups (z-step, 8-row tile, 16-x tile): a CTA each, warp = block pb */
    (void)Yb;
    const uint8_t *gsc = gy + (size_t)N * nb * Sg * mx_rb(BW, GB);
    if (threadIdx.x < BW) bred[threadIdx.x] = 0.f;
    float bacc[NTASK];
#pragma unroll
    for (int k = 0; k < NTASK; k++) bacc[k] = 0.f;
    for (unsigned grp = blockIdx.x; grp < ngrp; grp += gridDim.x) {
        const int xb = (int)(grp % Wb), ty = (int)((grp / Wb) % TY), zt = (int)(grp / ((unsigned)Wb * TY));
        const int zz = 2 * zt + (warp >> 2), yp = 4 * ty + (warp & 3), z = z0 + zz;   /* this warp's block: pb = warp */
        {
            const int y = 2 * yp + (lane >> 4), x = 16 * xb + (lane & 15);
            float r[32];
#pragma unroll
            for (int j = 0; j < 32; j++) r[j] = 0.f;
            if (zz < Zs && y < H && x < W) {
                const size_t ri = ((size_t)n * nb + cb) * Sg + ((size_t)z * H + y) * W + x;
                mxf<GB>::dec_row(gy + ri * mx_rb(BW, GB), BW, mx_scale(gsc[ri]), r);
            }
#pragma unroll
            for (int j = 0; j < BW; j++) v[warp][j][lane] = r[j];
        }
        __syncwarp();
#pragma unroll
        for (int k = 0; k < NTASK; k++) {
            const int task = k * 32 + lane, c = task >> 2, f = task & 3, co = cb * BW + c;
            float q[8];
#pragma unroll
            for (int j = 0; j < 8; j++) q[j] = v[warp][c][8 * f + j];
            bacc[k] += ((q[0] + q[1]) + (q[2] + q[3])) + ((q[4] + q[5]) + (q[6] + q[7]));
            unsigned amu = 0u;
#pragma unroll
            for (int j = 0; j < 8; j++) amu = amax_u(amu, q[j]);
            amu = max(amu, __shfl_xor_sync(0xffffffffu, amu, 1)); amu = max(amu, __shfl_xor_sync(0xffffffffu, amu, 2));
            const int e = mx_exp(__uint_as_float(amu), 1.f / 6.f);
            const float m = exp2i(-e);
            const uint64_t vid = ((((uint64_t)n * Co + co) * D + z) * H + (2 * yp + (f >> 1))) * (uint64_t)W + 16 * xb + 8 * (f & 1);
            uint32_t hh[4];
            sr_hash4(sr, vid, hh);
            const unsigned word = sr_e2m1_word(q, m, hh);
            *(unsigned *)(recb[c] + warp * 16 + 4 * f) = word;
            if (f == 0) recb[c][128 + warp] = (uint8_t)(e + 127);
        }
        __syncthreads();
        for (int i = threadIdx.x; i < BW * (GP4_REC / 8); i += blockDim.x) {   /* whole records: 8-byte stores, consecutive in a record */
            const int c = i / (GP4_REC / 8), w8 = i % (GP4_REC / 8), co = cb * BW + c;
            if (co < Co) *(uint2 *)(out + ((((size_t)n * Co + co) * ZT + zt) * TY + ty) * Wb * GP4_REC + (size_t)xb * GP4_REC + 8 * w8) = *(const uint2 *)(recb[c] + 8 * w8);
        }
        __syncthreads();   /* the tile and record buffers are reused */
    }
    if (gb) {
        __syncthreads();
#pragma unroll
        for (int k = 0; k < NTASK; k++) {
            float a = bacc[k]; a += __shfl_xor_sync(0xffffffffu, a, 1); a += __shfl_xor_sync(0xffffffffu, a, 2);
            if ((lane & 3) == 0) atomicAdd(&bred[(k * 32 + lane) >> 2], a);
        }
        __syncthreads();
        if (threadIdx.x < BW && cb * BW + (int)threadIdx.x < Co) atomicAdd(&gb[cb * BW + threadIdx.x], bred[threadIdx.x]);
    }
}
#if F4W_PROF
static void f4w_prof_dump(shape5 xs, shape5 ys, int MT, int NT, int lay) {
    if (!getenv("F4W_PROF")) return;
    unsigned long long h[6]; cudaDeviceSynchronize(); cudaMemcpyFromSymbol(h, g_f4wprof, sizeof h);
    const double s = (double)(h[0] + h[1] + h[2]);
    fprintf(stderr, "f4wprof %d->%d @%d MT %d NT %d LY %d blocks %llu: x %.1f%% (decode %.1f, act + quantise %.1f) gy %.1f%% mma %.1f%%  (Mcyc/block %.3f)\n", xs.c, ys.c, xs.d, MT, NT, lay, h[3],
            100 * h[0] / s, 100 * h[4] / s, 100 * h[5] / s, 100 * h[1] / s, 100 * h[2] / s, s / h[3] / 1e6);
    memset(h, 0, sizeof h); cudaMemcpyToSymbol(g_f4wprof, h, sizeof h);
}
#endif
template <int MT, int NT, typename T, typename TG> static void launch_bw4(dim3 grid, size_t smem, const void *x, shape5 xs, const TG *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp, int ZC, int had, int lay,
                                                                          const uint8_t *gpre = nullptr, int zs0 = 0, int zs1 = 0) {
    static int attr[8];
    if (!attr[cur_dev_()]) { attr[cur_dev_()] = 1; cudaFuncSetAttribute((const void *)conv_bwd_w_f4_k<MT, NT, 0, T, TG>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); cudaFuncSetAttribute((const void *)conv_bwd_w_f4_k<MT, NT, 1, T, TG>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); cudaFuncSetAttribute((const void *)conv_bwd_w_f4_k<MT, NT, 2, T, TG>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); cudaFuncSetAttribute((const void *)conv_bwd_w_f4_k<MT, NT, 3, T, TG>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
    const int coop = f4w_coop() && ys.c <= 16 && IS_MX(T) && IS_MX8(TG) && (lay == 1 || lay == 2);
    if (lay == 3) conv_bwd_w_f4_k<MT, NT, 3, T, TG><<<grid, 288, smem>>>((const T *)x, gy, gw, gb, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, gp, sp, ZC, had, coop, gpre, zs0, zs1);
    else if (lay == 2) conv_bwd_w_f4_k<MT, NT, 2, T, TG><<<grid, 288, smem>>>((const T *)x, gy, gw, gb, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, gp, sp, ZC, had, coop, gpre, zs0, zs1);
    else if (lay == 1) conv_bwd_w_f4_k<MT, NT, 1, T, TG><<<grid, 288, smem>>>((const T *)x, gy, gw, gb, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, gp, sp, ZC, had, coop, gpre, zs0, zs1);
    else conv_bwd_w_f4_k<MT, NT, 0, T, TG><<<grid, 288, smem>>>((const T *)x, gy, gw, gb, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, gp, sp, ZC, had, coop, gpre, zs0, zs1);
}
template <typename T, typename TG> void bwd_w_f4_t(const void *x, shape5 xs, const TG *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp, int had) {
    static int nt_env = -2, mt_env = -2, zc_env = -2;
    if (nt_env == -2) { nt_env = getenv("UFSM_F4W_NT") ? atoi(getenv("UFSM_F4W_NT")) : -1; mt_env = getenv("UFSM_F4W_MT") ? atoi(getenv("UFSM_F4W_MT")) : -1; zc_env = getenv("UFSM_F4W_ZC") ? atoi(getenv("UFSM_F4W_ZC")) : -1; }
    static int lay_env = -1;
    if (lay_env < 0) lay_env = getenv("UFSM_F4W_LAYOUT") ? atoi(getenv("UFSM_F4W_LAYOUT")) : 1;   /* 1: per-32 x scales; 2 (one scale per (ci, plane), faster) failed the seed-1 stair: 0.201 vs 0.310 */
    /* plain mode (no Hadamard / x SR): 1 = row pairs, scale per (ci, 32 positions); 2 = rows, scale per (ci, plane) (fp8's
       granularity, one rounding per value); 0 = the 27 shifted blocks (always used with the Hadamard or x SR) */
    const int lay = (had & 4) ? 3 : (had & 3) ? 0 : lay_env;   /* had bit 2: the H16 variant (UFSM_F4_HAD_W=2) -> LY 3 */
    int MT = ys.c >= 32 ? 2 : 1;
    int NT = xs.c <= 8 ? 1 : (lay == 2 || (lay == 3 && !IS_MX(T))) && !sp.up && MT == 1 && xs.c % 24 == 0 ? 3 : 2;   /* NT 3 (as fp8 on dec0.c1) only fits 2 blocks / SM with the LY 2 tile */
    if (nt_env > 0) NT = nt_env;
    if (mt_env > 0) MT = mt_env;
    size_t smem = (size_t)8 * NT * ((lay >= 2 ? 976 : X4_CS) + 2 * X4_SCS) + 16 * MT * (G4_CS + 8 + 4) + 9 * 32 + (lay == 0 || lay == 3 ? 9 * 180 * 4 : 0);
    if (IS_MX(T)) smem += 32 + (size_t)8 * NT * (sizeof(chan_t) + (lay ? 240 * 2 : 0)) + (lay == 1 ? (size_t)8 * NT * (40 + 9) + 16 : 0);   /* channel table + bf16 decoded plane (+ LY 1 row amax, pair scales) */
    if (f4w_coop() && ys.c <= 16 && IS_MX(T) && IS_MX8(TG) && (lay == 1 || lay == 2)) smem += (size_t)max(0, 16 * MT * 256 - 8 * NT * 240) * 2;
    int nzt = nblk_(ys.d, 2), base = (int)(((xs.c + 8 * NT - 1) / (8 * NT)) * ((ys.c + 16 * MT - 1) / (16 * MT)) * nblk_(ys.w, 16) * nblk_(ys.h, 8) * ys.n);
    /* Row-pair MX4/MX8 operands use global-element rounding keys. A deeper
       persistent tile amortizes halo setup and atomics on large volumes without
       enlarging per-block storage. Other layouts retain their existing depth. */
    const int max_zc = !had && lay == 1 && IS_MX4(T) && IS_MX8(TG) && nzt >= 64 ? 96 : 12;
    int ZC = nzt < max_zc ? nzt : max_zc;
    while (ZC > 1 && (size_t)base * nblk_(nzt, ZC) < 72) ZC--;
    if (zc_env > 0) ZC = zc_env;
    /* gy pre-pass (MX-fp8 gy with SR, plain mode): z slabs of at most ~96 MB of pre-rounded blocks, each followed by the
       kernel over that slab's z-steps; UFSM_F4W_GYPRE=0 keeps the in-kernel rounding */
    static int pre_env = -1;
    if (pre_env < 0) pre_env = getenv("UFSM_F4W_GYPRE") ? atoi(getenv("UFSM_F4W_GYPRE")) : 96;
    const size_t cap_kb = g_f4w_gypre_kb >= 0 ? (size_t)g_f4w_gypre_kb : (size_t)pre_env * 1024;
    if (g_f4w_gy4 && !(IS_MX8(TG) && cap_kb && sp.sr && !had && (lay == 1 || lay == 2) && zc_env <= 0)) { fprintf(stderr, "lp_bwd_w_f4: MX-fp4 gradients need the gy pre-pass (SR on, plain layout)\n"); abort(); }
    if constexpr (IS_MX8(TG)) if (cap_kb && sp.sr && !had && (lay == 1 || lay == 2) && zc_env <= 0) {
        const int Wb = (ys.w + 15) / 16, gbw = mx_bw(ys.c);
        const size_t per_zt = (size_t)ys.n * ys.c * ((ys.h + 7) / 8) * Wb * GP4_REC;   /* one record per (n, co, z-step, 8-row tile, 16-x tile) */
        const size_t cap_zt = (cap_kb << 10) / per_zt;
        const int szt = cap_zt < 1 ? 1 : cap_zt < (size_t)nzt ? (int)cap_zt : nzt;
        uint8_t *buf = lp_buf<uint8_t>(4, per_zt * szt);
        for (int zs0 = 0; zs0 < nzt; zs0 += szt) {
            const int zs1 = zs0 + szt < nzt ? zs0 + szt : nzt, Zs = (2 * zs1 < ys.d ? 2 * zs1 : ys.d) - 2 * zs0;
            const size_t ntile = (size_t)2 * ((Zs + 1) / 2) * 4 * ((ys.h + 7) / 8) * Wb;   /* the padded block grid of the records */
            const size_t nwb = (ntile + 7) / 8;   /* CTAs of 8 warps, a few tiles per warp */
            const dim3 pg((unsigned)(nwb < 4096 ? nwb : 4096), (unsigned)(ys.n * mx_nb(ys.c)));
#define GP4(GB) do { if (gbw == 8) gy_pre4_k<8, GB><<<pg, 256>>>((const uint8_t *)gy, buf, gb, ys.n, ys.c, ys.d, ys.h, ys.w, 2 * zs0, Zs, sp.sr); \
                        else if (gbw == 16) gy_pre4_k<16, GB><<<pg, 256>>>((const uint8_t *)gy, buf, gb, ys.n, ys.c, ys.d, ys.h, ys.w, 2 * zs0, Zs, sp.sr); \
                        else gy_pre4_k<32, GB><<<pg, 256>>>((const uint8_t *)gy, buf, gb, ys.n, ys.c, ys.d, ys.h, ys.w, 2 * zs0, Zs, sp.sr); } while (0)
            if (g_f4w_gy4) GP4(4); else GP4(8);
#undef GP4
            int zc_s = zs1 - zs0 < max_zc ? zs1 - zs0 : max_zc;
            while (zc_s > 1 && (size_t)base * nblk_(zs1 - zs0, zc_s) < 72) zc_s--;
            const dim3 g((unsigned)(nblk_(ys.w, 16) * nblk_(ys.h, 8) * nblk_(zs1 - zs0, zc_s) * ys.n), (xs.c + 8 * NT - 1) / (8 * NT), (ys.c + 16 * MT - 1) / (16 * MT));
            static int ws_env = -1;
            if (ws_env < 0) ws_env = getenv("UFSM_F4W_WS") ? atoi(getenv("UFSM_F4W_WS")) : 1;
            if constexpr (IS_MX(T)) if (ws_env && lay == 1 && !had && !sp.up && NT == 2 && MT == 2 && (sp.x2 == nullptr || sp.c_split % 16 == 0)) {   /* MT 1 (16 outputs): no gain */
                /* warp-specialised persistent kernel: the blocks of a channel tile split its (tile, z chunk) items; z chunks
                   short enough for >= 4 items per SM, then the block count with the shortest estimated makespan (waves x
                   items per block, in z-steps: + 1 per item for its 4-plane first step, + 6 per block for fill and drain) */
                const int nsm = ws_nsm(), combos = ((xs.c + 15) / 16) * ((ys.c + 16 * MT - 1) / (16 * MT)) * ys.n, ntile = nblk_(ys.w, 16) * nblk_(ys.h, 8);
                int zc_w = zs1 - zs0;
                while (zc_w > 4 && (size_t)combos * ntile * nblk_(zs1 - zs0, zc_w) < (size_t)4 * nsm) zc_w = (zc_w + 1) / 2;
                const int nitem = ntile * nblk_(zs1 - zs0, zc_w);
                int nper = 1; double best = 1e30;
                for (int k = 1; k <= nitem; k++) {
                    const double waves = (double)nblk_(combos * k, nsm), cost = waves * ((double)nblk_(nitem, k) * (zc_w + 1) + 6);
                    if (cost < best - 1e-9) { best = cost; nper = k; }
                }
                const dim3 gw3((unsigned)(nper * ys.n), (xs.c + 15) / 16, (ys.c + 16 * MT - 1) / (16 * MT));
                static int wattr[8][2];
                const int mi = MT - 1;
                if (MT == 1) {
                    if (!wattr[cur_dev_()][mi]) { wattr[cur_dev_()][mi] = 1; cudaFuncSetAttribute((const void *)conv_bwd_w_f4ws_k<1, 2, T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ws_smem<1, 2>()); }
                    conv_bwd_w_f4ws_k<1, 2, T><<<gw3, WS_NT, ws_smem<1, 2>()>>>((const T *)x, gw, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, gp, sp, zc_w, buf, zs0, zs1);
                } else {
                    if (!wattr[cur_dev_()][mi]) { wattr[cur_dev_()][mi] = 1; cudaFuncSetAttribute((const void *)conv_bwd_w_f4ws_k<2, 2, T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ws_smem<2, 2>()); }
                    conv_bwd_w_f4ws_k<2, 2, T><<<gw3, WS_NT, ws_smem<2, 2>()>>>((const T *)x, gw, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, gp, sp, zc_w, buf, zs0, zs1);
                }
                continue;
            }
            switch (MT * 10 + NT) {
            case 11: launch_bw4<1, 1, T, TG>(g, smem, x, xs, gy, ys, gw, gb, gp, sp, zc_s, had, lay, buf, zs0, zs1); break;
            case 12: launch_bw4<1, 2, T, TG>(g, smem, x, xs, gy, ys, gw, gb, gp, sp, zc_s, had, lay, buf, zs0, zs1); break;
            case 13: launch_bw4<1, 3, T, TG>(g, smem, x, xs, gy, ys, gw, gb, gp, sp, zc_s, had, lay, buf, zs0, zs1); break;
            case 21: launch_bw4<2, 1, T, TG>(g, smem, x, xs, gy, ys, gw, gb, gp, sp, zc_s, had, lay, buf, zs0, zs1); break;
            case 22: launch_bw4<2, 2, T, TG>(g, smem, x, xs, gy, ys, gw, gb, gp, sp, zc_s, had, lay, buf, zs0, zs1); break;
            default: fprintf(stderr, "lp_bwd_w_f4: bad MT/NT %d/%d\n", MT, NT); abort();
            }
        }
#if F4W_PROF
        f4w_prof_dump(xs, ys, MT, NT, lay);
#endif
        return;
    }
    int nzc = (nzt + ZC - 1) / ZC;
    dim3 grid((unsigned)(nblk_(ys.w, 16) * nblk_(ys.h, 8) * nzc * ys.n), (xs.c + 8 * NT - 1) / (8 * NT), (ys.c + 16 * MT - 1) / (16 * MT));
    switch (MT * 10 + NT) {
    case 11: launch_bw4<1, 1, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, gp, sp, ZC, had, lay); break;
    case 12: launch_bw4<1, 2, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, gp, sp, ZC, had, lay); break;
    case 13: launch_bw4<1, 3, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, gp, sp, ZC, had, lay); break;
    case 21: launch_bw4<2, 1, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, gp, sp, ZC, had, lay); break;
    case 22: launch_bw4<2, 2, T, TG>(grid, smem, x, xs, gy, ys, gw, gb, gp, sp, ZC, had, lay); break;
    default: fprintf(stderr, "lp_bwd_w_f4: bad MT/NT %d/%d\n", MT, NT); abort();
    }
#if F4W_PROF
    f4w_prof_dump(xs, ys, MT, NT, lay);
#endif
}
