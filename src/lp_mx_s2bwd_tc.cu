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
template <int C>
__global__ void __launch_bounds__(256) bwd_data_s2_tc_k(const uint8_t *__restrict__ gy, const __nv_bfloat16 *__restrict__ wtg, uint8_t *__restrict__ gx,
                                                     int N, int D, int H, int W, int Do, int Ho, int Wo, int accum) {
    constexpr int TMZ = 2, TMY = 8, TMX = C == 16 ? 16 : 8;   /* class positions per block */
    constexpr int GZ = TMZ + 1, GY = TMY + 1, GX = TMX + 1, GROW = C + 8;   /* gy tile, rows padded by 8 bf16 (bank spread) */
    constexpr int MT = C / 16, KS = C / 16, NTW = TMZ * TMY * (TMX / 8) / 8;   /* M tiles, k-steps, n-tiles per warp */
    extern __shared__ __align__(16) unsigned char s2t_smem[];
    __nv_bfloat16 *gyt = (__nv_bfloat16 *)s2t_smem;                       /* [GZ][GY][GX][GROW] */
    __nv_bfloat16 *wt = gyt + GZ * GY * GX * GROW;                          /* [8 taps][C ci][GROW co] */
    float *ob = (float *)(wt + 8 * C * GROW);                               /* [8 warps][8 voxels][C] */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int Mz = (D + 1) / 2, My = (H + 1) / 2, Mx = (W + 1) / 2;
    const int ntx = (Mx + TMX - 1) / TMX, nty = (My + TMY - 1) / TMY, ntz = (Mz + TMZ - 1) / TMZ;
    int bt = blockIdx.x;
    const int mx0 = (bt % ntx) * TMX; bt /= ntx;
    const int my0 = (bt % nty) * TMY; bt /= nty;
    const int mz0 = (bt % ntz) * TMZ; const int n = bt / ntz;
    const size_t So = (size_t)Do * Ho * Wo, S = (size_t)D * H * W;
    const uint8_t *gsc = gy + (size_t)N * So * C, *xsc = gx + (size_t)N * S * C;
    for (int i = threadIdx.x; i < GZ * GY * GX; i += blockDim.x) {
        const int tx = i % GX, ty = (i / GX) % GY, tz = i / (GX * GY);
        const int oz = mz0 + tz, oy = my0 + ty, ox = mx0 + tx;
        float r[32];
#pragma unroll
        for (int k = 0; k < 32; k++) r[k] = 0.f;
        if (oz < Do && oy < Ho && ox < Wo) {
            const size_t ri = (size_t)n * So + ((size_t)oz * Ho + oy) * Wo + ox;
            mxf<8>::dec_row(gy + ri * C, C, mx_scale(gsc[ri]), r);
        }
        __nv_bfloat16 *d = gyt + (size_t)i * GROW;
#pragma unroll
        for (int k = 0; k < C; k += 2) *(__nv_bfloat162 *)(d + k) = __floats2bfloat162_rn(r[k], r[k + 1]);
    }
    float *wob = ob + warp * 8 * C;
#pragma unroll 1
    for (int p = 0; p < 8; p++) {
        const int pz = p >> 2, py = (p >> 1) & 1, px = p & 1;
        const int nz = pz ? 2 : 1, ny = py ? 2 : 1, nx = px ? 2 : 1, ntap = nz * ny * nx;
        __syncthreads();   /* the previous class's A tiles are consumed (and, for p = 0, the gy tile is written) */
        constexpr int TQ = C * GROW / 8;   /* uint4 per tap tile */
        for (int i = threadIdx.x; i < ntap * TQ; i += blockDim.x) {
            const int tl = i / TQ, j = i % TQ;
            const int a = tl / (ny * nx), bb = (tl / nx) % ny, c = tl % nx;
            const int kz = pz ? (a ? 2 : 0) : 1, ky = py ? (bb ? 2 : 0) : 1, kx = px ? (c ? 2 : 0) : 1;
            ((uint4 *)(wt + (size_t)tl * C * GROW))[j] = __ldg((const uint4 *)(wtg + (size_t)((kz * 3 + ky) * 3 + kx) * C * GROW) + j);
        }
        __syncthreads();
#pragma unroll 1
        for (int nt = 0; nt < NTW; nt++) {
            const int tile = warp * NTW + nt, row = tile / (TMX / 8), xh = tile % (TMX / 8), rz = row / TMY, ry = row % TMY;
            /* this lane's slice of the old gx row (accumulating), loaded before the MMA so its latency is hidden */
            constexpr int CQ = C / 4;
            const int vl = lane >> 2, part = lane & 3;
            const int uz = 2 * (mz0 + rz) + pz, uy = 2 * (my0 + ry) + py, ux = 2 * (mx0 + 8 * xh + vl) + px;
            const bool ok = uz < D && uy < H && ux < W;
            const size_t ri = ok ? (size_t)n * S + ((size_t)uz * H + uy) * W + ux : 0;
            unsigned wd[CQ / 4] = {}, osc = 0u;
            if (accum && ok) {
                if constexpr (CQ == 8) { const uint2 u = *(const uint2 *)(gx + ri * C + part * CQ); wd[0] = u.x; wd[1] = u.y; }
                else wd[0] = *(const unsigned *)(gx + ri * C + part * CQ);
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
                        const __nv_bfloat16 *arow = wt + ((size_t)tl * C + 16 * m + g) * GROW + 16 * ks + 2 * t;
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
                wob[(2 * t) * C + 16 * m + g] = acc[m][0]; wob[(2 * t + 1) * C + 16 * m + g] = acc[m][1];
                wob[(2 * t) * C + 16 * m + g + 8] = acc[m][2]; wob[(2 * t + 1) * C + 16 * m + g + 8] = acc[m][3];
            }
            __syncwarp();
            {   /* 4 lanes per voxel (lane >> 2), C / 4 channels each: add the old row (accumulating), block amax by shuffles,
                   encode (the same per-value conversions as mxf<8>::enc_row) */
                float v[CQ];
#pragma unroll
                for (int k = 0; k < CQ; k++) v[k] = wob[vl * C + part * CQ + k];
                if (accum && ok) {
                    const float sc = mx_scale(osc);
#pragma unroll
                    for (int q = 0; q < CQ / 4; q++) {
                        const float2 d0 = dec_e4m3x2((unsigned short)(wd[q] & 0xffffu)), d1 = dec_e4m3x2((unsigned short)(wd[q] >> 16));
                        v[4 * q] += d0.x * sc; v[4 * q + 1] += d0.y * sc; v[4 * q + 2] += d1.x * sc; v[4 * q + 3] += d1.y * sc;
                    }
                }
                unsigned am = 0u;
#pragma unroll
                for (int k = 0; k < CQ; k++) am = amax_u(am, v[k]);
                am = max(am, __shfl_xor_sync(0xffffffffu, am, 1)); am = max(am, __shfl_xor_sync(0xffffffffu, am, 2));
                const int e = mx_exp(__uint_as_float(am), mxf<8>::inv_qmax);
                const float m = exp2i(-e);
                if (ok) {
                    if constexpr (CQ == 8) *(uint2 *)(gx + ri * C + part * CQ) = make_uint2(mxf<8>::enc8x(v, m), mxf<8>::enc8x(v + 4, m));
                    else *(unsigned *)(gx + ri * C + part * CQ) = mxf<8>::enc8x(v, m);
                    if (part == 0) gx[(size_t)N * S * C + ri] = (uint8_t)(e + 127);
                }
            }
            __syncwarp();   /* the warp's output buffer is reused by the next tile */
        }
    }
}
extern "C" int lp_bwd_data_s2_mx_tc(const void *gy, shape5 ys, const float *w, shape5 xs, void *gx, int accum) {
    static int on = -1; if (on < 0) on = getenv("UFSM_S2B_TC") ? atoi(getenv("UFSM_S2B_TC")) : 1;
    if (!on || xs.c != ys.c || (xs.c != 16 && xs.c != 32)) return 0;
    const int C = xs.c, TMX = C == 16 ? 16 : 8, TMZ = 2, TMY = 8;
    const int Mz = (xs.d + 1) / 2, My = (xs.h + 1) / 2, Mx = (xs.w + 1) / 2;
    const unsigned nb = (unsigned)(((Mx + TMX - 1) / TMX) * ((My + TMY - 1) / TMY) * ((Mz + TMZ - 1) / TMZ) * xs.n);
    const size_t sm = (size_t)(TMZ + 1) * (TMY + 1) * (TMX + 1) * (C + 8) * 2 + (size_t)8 * C * (C + 8) * 2 + (size_t)8 * 8 * C * 4;
    static int attr[8][2];
    const int ci = C == 16 ? 0 : 1;
    if (!attr[cur_dev_()][ci]) {
        attr[cur_dev_()][ci] = 1;
        if (C == 16) cudaFuncSetAttribute((const void *)bwd_data_s2_tc_k<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
        else cudaFuncSetAttribute((const void *)bwd_data_s2_tc_k<32>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
    }
    __nv_bfloat16 *wtg = lp_buf<__nv_bfloat16>(0, (size_t)27 * C * (C + 8));
    if (C == 16) {
        s2tc_prep_w_k<16><<<nblk_(27 * 16 * 24, 256), 256>>>(w, wtg);
        bwd_data_s2_tc_k<16><<<nb, 256, sm>>>((const uint8_t *)gy, wtg, (uint8_t *)gx, xs.n, xs.d, xs.h, xs.w, ys.d, ys.h, ys.w, accum);
    } else {
        s2tc_prep_w_k<32><<<nblk_(27 * 32 * 40, 256), 256>>>(w, wtg);
        bwd_data_s2_tc_k<32><<<nb, 256, sm>>>((const uint8_t *)gy, wtg, (uint8_t *)gx, xs.n, xs.d, xs.h, xs.w, ys.d, ys.h, ys.w, accum);
    }
    LPCK();
    return 1;
}
