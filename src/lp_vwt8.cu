/* vwt convs on MX-fp8 tensor cores (PF; src/nn_vwt_fast.cu dispatches here when vtf_set_mx(8) / UFSM_VTF_MX=8): forward and
   backward-data as implicit GEMMs with mma m16n8k32 kind::mxf8f6f4 (e4m3 x e4m3, ue8m0 scale per 32 K, fp32 accumulation; 4x the
   bf16 rate on GeForce Blackwell).

   The A source (fwd: x, bwd-data: gy; channels-last fp32 [vox][C], C % 32 == 0) is quantized once per call to MX-fp8: e4m3 bytes
   xq[vox][C] and one scale byte per (voxel, 32-channel block) xs[vox][SB] (SB = C / 32 rounded up to 4). A K step of 32 = one tap's
   32-channel block, so an A row's block scale is its source voxel's (it moves with every tap shift). B = the weights in e4m3 with a
   scale per (output column, 32 K). Tiles: BM rows (voxels) x BN columns, K steps of 32 through a 4-stage cp.async pipeline (16-byte
   data chunks, 4-byte scale words); rows of 32 bytes, the 16-byte halves swizzled by bit 2 of the row (conflict-free ldmatrix).
   Strided backward-data skips the taps no row of a tile reads (as the bf16 kernel). */
#include "lp_common.cuh"
#include "vwt.h"

#define V8K 32
#define V8S 3   /* pipeline stages */
#define V8B 2   /* 32-K blocks per stage (each its own tap / channel block: a stage may span taps) */

struct V8P {
    const uint8_t *xq, *xs, *wq, *ws; const float *bias; float *y;
    __nv_bfloat16 *yb; int yld, yc0, yc1;   /* yb: bf16 output of columns [yc0, yc1) at yb[m yld + co - yc0] instead of y (vtf8_conv_bwd_bfout) */
    int D, H, W, C, SB;      /* A source dims, its scale row stride (bytes) */
    int Do, Ho, Wo;          /* row grid */
    int k, sd, sh, sw, p;
    int K, N, WSB;           /* K = k^3 C; columns; weight scale row stride (bytes) */
    long long M;
};
template <int MODE>
__device__ __forceinline__ long long v8_src(const V8P &p, long long nb, int z, int y, int x, int tz, int ty, int tx) {
    if (MODE == 1) {   /* strides 1 / 2 (vtf8_ok): shifts and masks, no integer division */
        const int nz = z + p.p - tz, ny = y + p.p - ty, nx = x + p.p - tx, lz = p.sd - 1, ly = p.sh - 1, lx = p.sw - 1;
        if ((nz | ny | nx) < 0 || (nz & lz) | (ny & ly) | (nx & lx)) return -1;
        const int zo = nz >> lz, yo = ny >> ly, xo = nx >> lx;
        if (zo >= p.D || yo >= p.H || xo >= p.W) return -1;
        return ((nb + zo) * p.H + yo) * (long long)p.W + xo;
    }
    if (MODE == 0) {
        const int zi = z * p.sd - p.p + tz, yi = y * p.sh - p.p + ty, xi = x * p.sw - p.p + tx;
        if (zi < 0 || zi >= p.D || yi < 0 || yi >= p.H || xi < 0 || xi >= p.W) return -1;
        return ((nb + zi) * p.H + yi) * (long long)p.W + xi;
    } else {
        const int nz = z + p.p - tz, ny = y + p.p - ty, nx = x + p.p - tx;
        if (nz < 0 || ny < 0 || nx < 0) return -1;
        if (nz % p.sd || ny % p.sh || nx % p.sw) return -1;
        const int zo = nz / p.sd, yo = ny / p.sh, xo = nx / p.sw;
        if (zo >= p.D || yo >= p.H || xo >= p.W) return -1;
        return ((nb + zo) * p.H + yo) * (long long)p.W + xo;
    }
}
__device__ __forceinline__ int sw32(int row, int q) { return row * 32 + ((q ^ ((row >> 2) & 1)) << 4); }
__device__ __forceinline__ void ldsm4b(unsigned *r, const void *p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((unsigned)__cvta_generic_to_shared(p)));
}
/* bf16 rows of 64 bytes: the 16-byte chunk q (0..3) at q ^ ((row >> 1) & 3) */
__device__ __forceinline__ int sw64b(int row, int q) { return row * 64 + ((q ^ ((row >> 1) & 3)) << 4); }
template <bool F8> __device__ __forceinline__ int swr(int row, int q) { return F8 ? sw32(row, q) : sw64b(row, q); }
__device__ __forceinline__ void mma_bf16v(float *c, const unsigned *a, const unsigned *b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void cpa_commit8() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void cpa_wait8() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

/* 256 threads; BM == 128 (thread pair per A row), BN <= 128. F8: MX-fp8 (rows of 32 bytes per 32-K block, scales, V8B blocks per stage);
   else bf16 (rows of 64 bytes, one block per stage: 2 blocks per SM fit) */
template <bool F8> __host__ __device__ constexpr int v8_vb() { return F8 ? V8B : 1; }
template <bool F8, int BN> __host__ __device__ constexpr int v8_smem() { return V8S * v8_vb<F8>() * ((F8 ? 32 : 64) * (128 + BN) + (F8 ? 4 * (128 + BN) : 0)); }
template <int MODE, int BN, int WARPS_M, int WARPS_N, bool TS, bool F8>
__global__ void __launch_bounds__(256, 2) vtf8_igemm(V8P p) {
    constexpr int BM = 128, WM = BM / WARPS_M, WN = BN / WARPS_N, MI = WM / 16, NI = WN / 8;
    constexpr int VB = v8_vb<F8>(), RB = F8 ? 32 : 64, EB = F8 ? 1 : 2, CHT = RB / 32;   /* blocks per stage, row bytes, element bytes, chunks per thread */
    static_assert(WARPS_M * WARPS_N == 8 && NI % 2 == 0, "tile");
    extern __shared__ __align__(128) uint8_t smem8[];
    /* stage s, block b: A tile at sA(s, b), B tile at sB(s, b), their scale words */
    uint8_t *const sA0 = smem8, *const sB0 = smem8 + V8S * VB * BM * RB;
    uint32_t *const sAs0 = (uint32_t *)(sB0 + V8S * VB * BN * RB), *const sBs0 = sAs0 + V8S * VB * BM;
#define SA(s, b) (sA0 + ((s) * VB + (b)) * BM * RB)
#define SB(s, b) (sB0 + ((s) * VB + (b)) * BN * RB)
#define SAS(s, b) (sAs0 + ((s) * VB + (b)) * BM)
#define SBS(s, b) (sBs0 + ((s) * VB + (b)) * BN)
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm0 = (warp / WARPS_N) * WM, wn0 = (warp % WARPS_N) * WN;
    const long long m_base = (long long)blockIdx.x * BM; const int n_base = blockIdx.y * BN;
    /* loader: A row ra = tid / 2, half qa = tid & 1 (+ its scale word when qa == 0); B row = tid / 2 (< BN) the same */
    const int ra = tid >> 1, qa = tid & 1;
    int zr, yr, xr; long long nr;
    { const long long m = m_base + ra;
      if (m < p.M) { long long t = m; xr = (int)(t % p.Wo); t /= p.Wo; yr = (int)(t % p.Ho); t /= p.Ho; zr = (int)(t % p.Do); t /= p.Do; nr = t * p.D; }
      else { zr = -(1 << 28); yr = xr = 0; nr = 0; } }
    const int ntap = p.k * p.k * p.k, spt = p.C / V8K, nk = p.K / V8K;
    unsigned tmask = 0xffffffffu;
    if (TS && ntap <= 32) {
        tmask = 0;
        for (int t = 0; t < ntap; t++) {
            const int any = zr > -(1 << 27) && v8_src<MODE>(p, nr, zr, yr, xr, t / (p.k * p.k), (t / p.k) % p.k, t % p.k) >= 0;
            if (__syncthreads_or(any)) tmask |= 1u << t;
        }
    }
    /* the loader's position: K step kl = tap lt (lz, ly, lx) x spt + block lcb; the source voxel of the current tap cached in ls */
    int kl = 0, lt = 0, lcb = 0, lz = 0, ly = 0, lx = 0, lst = -1; long long ls = -1;
    auto tap_step = [&]() { lt++; if (++lx == p.k) { lx = 0; if (++ly == p.k) { ly = 0; lz++; } } };
    auto advance = [&]() {   /* to the next K step some row reads */
        if (++lcb == spt) { lcb = 0; tap_step(); if (TS) while (lt < ntap && !((tmask >> lt) & 1u)) tap_step(); }
        kl = lt < ntap ? lt * spt + lcb : nk;
    };
    if (TS) { while (lt < ntap && !((tmask >> lt) & 1u)) tap_step(); kl = lt < ntap ? lt * spt : nk; }
    auto load = [&](int ks, int st, int b) {
        const int cb = lcb;
        if (lst != lt) { lst = lt; ls = zr > -(1 << 27) ? v8_src<MODE>(p, nr, zr, yr, xr, lz, ly, lx) : -1; }
        const long long s = ls;
        const bool ok = s >= 0;
#pragma unroll
        for (int j = 0; j < CHT; j++) { const int q = qa + 2 * j; cp_async<16>(SA(st, b) + swr<F8>(ra, q), p.xq + (ok ? (s * p.C + cb * 32) * EB + q * 16 : 0), ok); }
        if (F8 && !qa) cp_async<4>(SAS(st, b) + ra, p.xs + (ok ? s * p.SB + (cb & ~3) : 0), ok);
        if (ra < BN) {
            const size_t row = (size_t)(n_base + ra);
#pragma unroll
            for (int j = 0; j < CHT; j++) { const int q = qa + 2 * j; cp_async<16>(SB(st, b) + swr<F8>(ra, q), p.wq + (row * p.K + ks * 32) * EB + q * 16, true); }
            if (F8 && !qa) cp_async<4>(SBS(st, b) + ra, p.ws + row * p.WSB + (ks & ~3), true);
        }
    };
    float acc[MI][NI][4];
#pragma unroll
    for (int a = 0; a < MI; a++)
#pragma unroll
        for (int b = 0; b < NI; b++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[a][b][c] = 0.f;
    __shared__ int kc[V8S][VB], kcb[V8S][VB];   /* the K step each stage block holds (nk: none) and its channel block (block-uniform) */
    auto fill = [&](int st) {
#pragma unroll
        for (int b = 0; b < VB; b++) { kc[st][b] = kl; kcb[st][b] = lcb; if (kl < nk) { load(kl, st, b); advance(); } }
        cpa_commit8();
    };
#pragma unroll
    for (int s = 0; s < V8S - 1; s++) fill(s);
    const int g = lane >> 2, t4 = lane & 3;
    for (int st = 0; ; ) {
        cpa_wait8<V8S - 2>(); __syncthreads();
        if (kc[st][0] >= nk) break;
        int kb[VB], cbb[VB];
#pragma unroll
        for (int b = 0; b < VB; b++) { kb[b] = kc[st][b]; cbb[b] = kcb[st][b]; }
        fill((st + V8S - 1) % V8S);
#pragma unroll
        for (int b = 0; b < VB; b++) {
            const int ks = kb[b];
            if (ks >= nk) break;
            const uint8_t *A = SA(st, b), *B = SB(st, b);
            if constexpr (!F8) {
#pragma unroll
                for (int kk = 0; kk < 2; kk++) {
                    unsigned af[MI][4], bfr[NI][2];
#pragma unroll
                    for (int mi = 0; mi < MI; mi++) { const int row = wm0 + mi * 16 + (lane & 15); ldsm4b(af[mi], A + sw64b(row, kk * 2 + (lane >> 4))); }
#pragma unroll
                    for (int nj = 0; nj < NI / 2; nj++) {
                        const int row = wn0 + nj * 16 + (lane & 7) + ((lane >> 4) << 3); unsigned r[4];
                        ldsm4b(r, B + sw64b(row, kk * 2 + ((lane >> 3) & 1)));
                        bfr[2 * nj][0] = r[0]; bfr[2 * nj][1] = r[1]; bfr[2 * nj + 1][0] = r[2]; bfr[2 * nj + 1][1] = r[3];
                    }
#pragma unroll
                    for (int mi = 0; mi < MI; mi++)
#pragma unroll
                        for (int ni = 0; ni < NI; ni++) mma_bf16v(acc[mi][ni], af[mi], bfr[ni]);
                }
                continue;
            }
            const int bsh = (cbb[b] & 3) * 8, wsh = (ks & 3) * 8;
            unsigned af[MI][4], bfr[NI][2], sa[MI], sb[NI];
#pragma unroll
            for (int mi = 0; mi < MI; mi++) {
                const int row = wm0 + mi * 16 + (lane & 15); ldsm4b(af[mi], A + sw32(row, lane >> 4));
                const int sr = wm0 + mi * 16 + g + (t4 & 1) * 8; sa[mi] = (SAS(st, b)[sr] >> bsh) & 0xffu;
            }
#pragma unroll
            for (int nj = 0; nj < NI / 2; nj++) {
                const int row = wn0 + nj * 16 + (lane & 7) + ((lane >> 4) << 3); unsigned r[4];
                ldsm4b(r, B + sw32(row, (lane >> 3) & 1));
                bfr[2 * nj][0] = r[0]; bfr[2 * nj][1] = r[1]; bfr[2 * nj + 1][0] = r[2]; bfr[2 * nj + 1][1] = r[3];
            }
#pragma unroll
            for (int ni = 0; ni < NI; ni++) sb[ni] = (SBS(st, b)[wn0 + ni * 8 + g] >> wsh) & 0xffu;
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int ni = 0; ni < NI; ni++) mma_f8(acc[mi][ni], af[mi], bfr[ni], sa[mi], sb[ni]);
        }
        st = (st + 1) % V8S;
    }
#undef SA
#undef SB
#undef SAS
#undef SBS
    cpa_wait8<0>();
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
    for (int ni = 0; ni < NI; ni++) {
        const int co = n_base + wn0 + ni * 8 + t4 * 2;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const long long m = m_base + wm0 + mi * 16 + g + h * 8; if (m >= p.M) continue;
            if (p.yb) {
                __nv_bfloat16 *yr_ = p.yb + m * p.yld - p.yc0;
                if (co >= p.yc0 && co < p.yc1) yr_[co] = __float2bfloat16_rn(acc[mi][ni][2 * h]);
                if (co + 1 >= p.yc0 && co + 1 < p.yc1) yr_[co + 1] = __float2bfloat16_rn(acc[mi][ni][2 * h + 1]);
                continue;
            }
            float *yr_ = p.y + m * p.N;
            if (co < p.N) yr_[co] = acc[mi][ni][2 * h] + (p.bias ? p.bias[co] : 0.f);
            if (co + 1 < p.N) yr_[co + 1] = acc[mi][ni][2 * h + 1] + (p.bias ? p.bias[co + 1] : 0.f);
        }
    }
}

/* fp32 rows [M][C] (C % 32 == 0) -> e4m3 bytes + a scale byte per 32-channel block; 4 lanes per block, 8 channels each */
__global__ void k_v8_quant(const float *__restrict__ x, long long M, int C, int SB, uint8_t *__restrict__ q, uint8_t *__restrict__ s) {
    const long long n8 = M * (C / 8);
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < ((n8 + 31) & ~31LL); i += (long long)gridDim.x * blockDim.x) {
        const bool in = i < n8; float v[8];
        if (in) ld8x<float>(x + i * 8, v); else for (int e = 0; e < 8; e++) v[e] = 0.f;
        float a = 0.f;
#pragma unroll
        for (int e = 0; e < 8; e++) a = amax_xs(a, v[e]);
        a = amax_xs(a, __shfl_xor_sync(0xffffffffu, a, 1)); a = amax_xs(a, __shfl_xor_sync(0xffffffffu, a, 2));
        const int e = mx_exp(a, 1.f / 448.f); const float m = exp2i(-e);
        if (!in) continue;
        *(uint2 *)(q + i * 8) = make_uint2(cvt_e4m3x4(v[0] * m, v[1] * m, v[2] * m, v[3] * m), cvt_e4m3x4(v[4] * m, v[5] * m, v[6] * m, v[7] * m));
        if (!(i & 3)) { const long long row = i / (C / 8); const int blk = (int)((i % (C / 8)) / 4); s[row * SB + blk] = (uint8_t)(e + 127); }
    }
}
/* weights fp32 [co][T][ci] -> B [Np][K] e4m3 + scales [Np][WSB]: fwd row = co, K = t ci + c; bwd-data row = ci, K = t co + c. A thread per
   (row, 32-K block) */
__global__ void k_v8_wprep(const float *__restrict__ w, int co, int T, int ci, int Np, int K, int WSB, int bwd, uint8_t *__restrict__ q, uint8_t *__restrict__ s) {
    const int nkb = K / 32; const long long n = (long long)Np * nkb;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const int r = (int)(i / nkb), kb = (int)(i % nkb);
        float v[32]; float a = 0.f;
#pragma unroll
        for (int j = 0; j < 32; j++) {
            const int kk = kb * 32 + j; float x = 0.f;
            if (!bwd) { if (r < co) { const int t = kk / ci, c = kk % ci; x = w[((size_t)r * T + t) * ci + c]; } }
            else { if (r < ci) { const int t = kk / co, c = kk % co; x = w[((size_t)c * T + t) * ci + r]; } }
            v[j] = x; a = amax_xs(a, x);
        }
        const int e = mx_exp(a, 1.f / 448.f); const float m = exp2i(-e);
        uint32_t *d = (uint32_t *)(q + (size_t)r * K + kb * 32);
#pragma unroll
        for (int j = 0; j < 8; j++) d[j] = cvt_e4m3x4(v[4 * j] * m, v[4 * j + 1] * m, v[4 * j + 2] * m, v[4 * j + 3] * m);
        s[(size_t)r * WSB + kb] = (uint8_t)(e + 127);
    }
}

template <int MODE, int BN, int WM, int WN, bool F8> static void v8_go(const V8P &p, bool ts) {
    const dim3 g((unsigned)((p.M + 127) / 128), (unsigned)((p.N + BN - 1) / BN));
    constexpr int SM = v8_smem<F8, BN>();
    static bool set = false;
    if (!set) { cudaFuncSetAttribute(vtf8_igemm<MODE, BN, WM, WN, true, F8>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM);
                cudaFuncSetAttribute(vtf8_igemm<MODE, BN, WM, WN, false, F8>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM); set = true; }
    if (ts) vtf8_igemm<MODE, BN, WM, WN, true, F8><<<g, 256, SM>>>(p); else vtf8_igemm<MODE, BN, WM, WN, false, F8><<<g, 256, SM>>>(p);
}
template <int MODE, bool F8> static void v8_launch(const V8P &p, int bt, bool ts) {
    if (bt == 32) v8_go<MODE, 32, 4, 2, F8>(p, ts); else if (bt == 64) v8_go<MODE, 64, 4, 2, F8>(p, ts);
    else if (bt == 96) v8_go<MODE, 96, 4, 2, F8>(p, ts); else v8_go<MODE, 128, 2, 4, F8>(p, ts);
}
/* fp32 -> bf16 rows; weights -> bf16 B [Np][K] (fwd row = co, K = t ci + c; bwd-data row = ci, K = t co + c) */
__global__ void k_v16_cvt(const float *__restrict__ x, long long n8, uint4 *__restrict__ y) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n8; i += (long long)gridDim.x * blockDim.x) {
        float v[8]; ld8x<float>(x + i * 8, v);
        __nv_bfloat162 a = __floats2bfloat162_rn(v[0], v[1]), b = __floats2bfloat162_rn(v[2], v[3]), c = __floats2bfloat162_rn(v[4], v[5]), d = __floats2bfloat162_rn(v[6], v[7]);
        y[i] = make_uint4(*(unsigned *)&a, *(unsigned *)&b, *(unsigned *)&c, *(unsigned *)&d);
    }
}
__global__ void k_v16_wprep(const float *__restrict__ w, int co, int T, int ci, int Np, int K, int bwd, __nv_bfloat16 *__restrict__ q) {
    const long long n = (long long)Np * K;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const int r = (int)(i / K), kk = (int)(i % K); float x = 0.f;
        if (!bwd) { if (r < co) { const int t = kk / ci, c = kk % ci; x = w[((size_t)r * T + t) * ci + c]; } }
        else { if (r < ci) { const int t = kk / co, c = kk % co; x = w[((size_t)c * T + t) * ci + r]; } }
        q[i] = __float2bfloat16_rn(x);
    }
}
static int v8_tile(int N) { return N <= 32 ? 32 : N <= 64 ? 64 : N <= 96 ? 96 : N <= 128 ? 128 : N <= 192 ? 96 : 128; }
static unsigned v8_blocks(long long n) { long long b = (n + 255) / 256; return (unsigned)(b > 4096 ? 4096 : b < 1 ? 1 : b); }
static size_t r256(size_t b) { return (b + 255) & ~(size_t)255; }

/* the A source src [n][D][H][W][C] (fwd: x; bwd-data: gy), rows = the M voxels of the (Do, Ho, Wo) grid, N columns; w [co][T][ci]
   (fwd: N = co, C = ci; bwd-data: N = ci, C = co). qbuf: >= vtf8_qbytes(), wbuf: >= vtf8_wbytes(). Returns -1 when the shapes do not
   qualify (C % 32, N < 32) */
extern "C" size_t vtf8_qbytes(long long rows, int C, int bits) { const int SB = (C / 32 + 3) & ~3; return bits == 8 ? r256((size_t)rows * C) + r256((size_t)rows * SB) : r256((size_t)rows * C * 2); }
extern "C" size_t vtf8_wbytes(int N, int K, int bits) { const int Np = (N + 127) / 128 * 128 + 128, WSB = (K / 32 + 3) & ~3; return bits == 8 ? r256((size_t)Np * K) + r256((size_t)Np * WSB) : r256((size_t)Np * K * 2); }
extern "C" int vtf8_ok(int C, int N) { return C % 32 == 0 && N >= 32 && N <= 256; }   /* (and strides 1 / 2: vtf8_conv) */
extern "C" int vtf8_conv(int mode, int bits, const float *src, int n, int D, int H, int W, int C, const float *w, int co, int ci, int k, int sd, int sh, int sw,
                         int Do, int Ho, int Wo, int N, const float *bias, float *y, void *qbuf, void *wbuf) {
    if (!vtf8_ok(C, N) || sd < 1 || sd > 2 || sh < 1 || sh > 2 || sw < 1 || sw > 2) return -1;
    const long long rows = (long long)n * D * H * W, M = (long long)n * Do * Ho * Wo; const int T = k * k * k, K = T * C;
    const int SB = (C / 32 + 3) & ~3, WSB = (K / 32 + 3) & ~3, bt = v8_tile(N), Np = (N + bt - 1) / bt * bt;
    const bool f8 = bits == 8;
    uint8_t *xq = (uint8_t *)qbuf, *xs = xq + r256((size_t)rows * C), *wq = (uint8_t *)wbuf, *ws = wq + r256((size_t)Np * K);
    if (f8) {
        k_v8_quant<<<v8_blocks(rows * (C / 8)), 256>>>(src, rows, C, SB, xq, xs);
        k_v8_wprep<<<v8_blocks((long long)Np * (K / 32)), 256>>>(w, co, T, ci, Np, K, WSB, mode, wq, ws);
    } else {
        k_v16_cvt<<<v8_blocks(rows * (C / 8)), 256>>>(src, rows * (C / 8), (uint4 *)xq);
        k_v16_wprep<<<v8_blocks((long long)Np * K), 256>>>(w, co, T, ci, Np, K, mode, (__nv_bfloat16 *)wq);
    }
    V8P p; p.xq = xq; p.xs = xs; p.wq = wq; p.ws = ws; p.bias = bias; p.y = y; p.yb = nullptr; p.yld = p.yc0 = p.yc1 = 0;
    p.D = D; p.H = H; p.W = W; p.C = C; p.SB = SB; p.Do = Do; p.Ho = Ho; p.Wo = Wo; p.k = k; p.sd = sd; p.sh = sh; p.sw = sw; p.p = k / 2;
    p.K = K; p.N = N; p.WSB = WSB; p.M = M;
    if (M <= 0) return 0;
    const bool ts = mode == 1 && (sd > 1 || sh > 1 || sw > 1);
    if (mode == 0) { if (f8) v8_launch<0, true>(p, bt, ts); else v8_launch<0, false>(p, bt, ts); }
    else { if (f8) v8_launch<1, true>(p, bt, ts); else v8_launch<1, false>(p, bt, ts); }
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/* forward from a bf16 source already in the kernel's layout ([rows][C] channels-last, C % 32 == 0; channels ci .. C - 1 zero): no input
   conversion; the weights w [co][T][ci] padded to C channels per tap (the vwt input conv: CT, valid and trunk features, src/vwt_net.c) */
__global__ void k_v16_wprep_pad(const float *__restrict__ w, int co, int T, int ci, int C, int Np, int K, __nv_bfloat16 *__restrict__ q) {
    const long long n = (long long)Np * K;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const int r = (int)(i / K), kk = (int)(i % K), t = kk / C, c = kk % C;
        q[i] = __float2bfloat16_rn(r < co && c < ci ? w[((size_t)r * T + t) * ci + c] : 0.f);
    }
}
extern "C" int vtf8_conv_bfin(const void *xb, int n, int D, int H, int W, int C, int ci, const float *w, int co, int k, int sd, int sh, int sw,
                              int Do, int Ho, int Wo, const float *bias, float *y, void *wbuf) {
    if (!vtf8_ok(C, co) || ci > C || sd < 1 || sd > 2 || sh < 1 || sh > 2 || sw < 1 || sw > 2) return -1;
    const long long M = (long long)n * Do * Ho * Wo; const int T = k * k * k, K = T * C, bt = v8_tile(co), Np = (co + bt - 1) / bt * bt;
    k_v16_wprep_pad<<<v8_blocks((long long)Np * K), 256>>>(w, co, T, ci, C, Np, K, (__nv_bfloat16 *)wbuf);
    V8P p; p.xq = (const uint8_t *)xb; p.xs = nullptr; p.wq = (const uint8_t *)wbuf; p.ws = nullptr; p.bias = bias; p.y = y; p.yb = nullptr; p.yld = p.yc0 = p.yc1 = 0;
    p.D = D; p.H = H; p.W = W; p.C = C; p.SB = 0; p.Do = Do; p.Ho = Ho; p.Wo = Wo; p.k = k; p.sd = sd; p.sh = sh; p.sw = sw; p.p = k / 2;
    p.K = K; p.N = co; p.WSB = 0; p.M = M;
    if (M <= 0) return 0;
    v8_launch<0, false>(p, bt, false);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/* backward-data of a conv into input channels [c0, c1) as bf16 rows of ld (the vwt input conv's gradient at the trunk features): gy fp32
   [n][Do][Ho][Wo][co] (co % 32 == 0), w [co][T][ci]; columns N = c1 rounded up to 16 (>= 32) */
extern "C" int vtf8_conv_bwd_bfout(const float *gy, int n, int Do, int Ho, int Wo, int co, const float *w, int ci, int k, int sd, int sh, int sw,
                                   int D, int H, int W, int c0, int c1, void *out, int ld, void *qbuf, void *wbuf) {
    int N = (c1 + 15) / 16 * 16; if (N < 32) N = 32;
    if (!vtf8_ok(co, N) || c1 > ci || sd < 1 || sd > 2 || sh < 1 || sh > 2 || sw < 1 || sw > 2) return -1;
    const long long rows = (long long)n * Do * Ho * Wo, M = (long long)n * D * H * W; const int T = k * k * k, K = T * co, bt = v8_tile(N), Np = (N + bt - 1) / bt * bt;
    uint8_t *xq = (uint8_t *)qbuf, *wq = (uint8_t *)wbuf;
    k_v16_cvt<<<v8_blocks(rows * (co / 8)), 256>>>(gy, rows * (co / 8), (uint4 *)xq);
    k_v16_wprep<<<v8_blocks((long long)Np * K), 256>>>(w, co, T, ci, Np, K, 1, (__nv_bfloat16 *)wq);
    V8P p; p.xq = xq; p.xs = nullptr; p.wq = wq; p.ws = nullptr; p.bias = nullptr; p.y = nullptr; p.yb = (__nv_bfloat16 *)out; p.yld = ld; p.yc0 = c0; p.yc1 = c1;
    p.D = Do; p.H = Ho; p.W = Wo; p.C = co; p.SB = 0; p.Do = D; p.Ho = H; p.Wo = W; p.k = k; p.sd = sd; p.sh = sh; p.sw = sw; p.p = k / 2;
    p.K = K; p.N = N; p.WSB = 0; p.M = M;
    if (M <= 0) return 0;
    v8_launch<1, false>(p, bt, sd > 1 || sh > 1 || sw > 1);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
