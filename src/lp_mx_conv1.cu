/* 1x1 convolution forward and weight gradient on MX tensors */
#include "lp_mxops.cuh"
/* 1^3 conv (head) reading an MX tensor (B bits): y[n][co][v] = b[co] + sum_ci w[co][ci] x[ci][v] (fp32 output) */
template <int B>
__global__ void conv1_mx_k(const uint8_t *x, const float *w, const float *b, float *y, int N, int Ci, int Co, size_t S, gnp_t gp) {
    const int bw = mx_bw(Ci), nb = mx_nb(Ci);
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * S) return;
    int n = (int)(i / S); size_t v = i % S;
    float acc[8];
    for (int co = 0; co < Co && co < 8; co++) acc[co] = b ? b[co] : 0.f;
    const uint8_t *sc = mx_sc<B>(x, N, Ci, S);
    for (int blk = 0; blk < nb; blk++) {
        float r[32];
        mx_load_row_b<B>(x, sc, ((size_t)n * nb + blk) * S + v, bw, r);
        row_gn(r, gp, n, blk, bw, Ci);
        for (int k = 0; k < bw; k++) { int ci = blk * bw + k; if (ci < Ci) for (int co = 0; co < Co && co < 8; co++) acc[co] += w[co * Ci + ci] * r[k]; }
    }
    for (int co = 0; co < Co && co < 8; co++) y[((size_t)n * Co + co) * S + v] = acc[co];
}
/* the same with the output count a template parameter (accumulators in registers) and the gn+silu coefficients (per n, ci),
   the weights and the bias staged once per block in smem instead of reloaded per voxel; same arithmetic order as conv1_mx_k.
   Weights [ci][8] (two broadcast float4 loads per input channel) and coefficients float2 [n][ci]: one scalar smem load per
   (co, ci) made the kernel smem-instruction-bound */
template <int B, int CO>
__global__ void __launch_bounds__(256) conv1_mx_t_k(const uint8_t *x, const float *w, const float *b, float *y, int N, int Ci, size_t S, gnp_t gp) {
    extern __shared__ __align__(16) float c1s[];
    float4 *sw = (float4 *)c1s;                       /* [Ci][2] (co 0..3, 4..7) */
    float2 *sab = (float2 *)(sw + 2 * Ci);            /* [N][Ci] (a, b) */
    for (int i = threadIdx.x; i < N * Ci; i += blockDim.x) {
        const int n = i / Ci, ci = i % Ci;
        if (gp.G) { const int ng = n * gp.G + ci / (Ci / gp.G); const float a = gp.rstd[ng] * gp.gamma[ci]; sab[i] = make_float2(a, gp.beta[ci] - gp.mean[ng] * a); }
    }
    for (int i = threadIdx.x; i < 8 * Ci; i += blockDim.x) { const int ci = i >> 3, co = i & 7; c1s[i] = co < CO ? w[co * Ci + ci] : 0.f; }
    __syncthreads();
    const int bw = mx_bw(Ci), nb = mx_nb(Ci);
    const int n = blockIdx.y; const size_t v = blockIdx.x * (size_t)blockDim.x + threadIdx.x;   /* grid (voxels, sample) */
    if (v >= S) return;
    float acc[CO];
#pragma unroll
    for (int co = 0; co < CO; co++) acc[co] = b ? b[co] : 0.f;
    const uint8_t *sc = mx_sc<B>(x, N, Ci, S);
    for (int blk = 0; blk < nb; blk++) {
        float r[32];
        mx_load_row_b<B>(x, sc, ((size_t)n * nb + blk) * S + v, bw, r);
#pragma unroll
        for (int k = 0; k < 32; k++) {
            const int ci = blk * bw + k;
            if (k >= bw || ci >= Ci) break;
            float xv = r[k];
            if (gp.G) { const float2 ab = sab[n * Ci + ci]; xv = act_ab(xv, ab.x, ab.y, true); }
            const float4 w0 = sw[2 * ci], w1 = sw[2 * ci + 1];
            const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
            for (int co = 0; co < CO; co++) acc[co] += wv[co] * xv;
        }
    }
#pragma unroll
    for (int co = 0; co < CO; co++) y[((size_t)n * CO + co) * S + v] = acc[co];
}
template <int B, int CO> static void conv1_mx_t(const void *x, shape5 xs, const float *w, const float *b, float *y, gnp_t gp) {
    const size_t S = shape_spatial(xs), sm = (size_t)(2 * xs.n * xs.c + 8 * xs.c) * sizeof(float);
    conv1_mx_t_k<B, CO><<<dim3(nblk_(S, 256), xs.n), 256, sm>>>((const uint8_t *)x, w, b, y, xs.n, xs.c, S, gp);
}
extern "C" void lp_conv1_fwd_mx(const void *x, int xdt, shape5 xs, const float *w, const float *b, int cout, float *y, gnp_t gp) {
    if (cout > 8) { fprintf(stderr, "lp_conv1_fwd_mx: cout %d > 8\n", cout); abort(); }
    size_t S = shape_spatial(xs);
    static int old = -1; if (old < 0) old = getenv("UFSM_CONV1_OLD") ? atoi(getenv("UFSM_CONV1_OLD")) : 0;
    if (!old && (size_t)(2 * xs.n * xs.c + 8 * xs.c) * sizeof(float) <= 32 * 1024) {
#define C1T(B) do { switch (cout) { case 1: conv1_mx_t<B, 1>(x, xs, w, b, y, gp); break; case 2: conv1_mx_t<B, 2>(x, xs, w, b, y, gp); break; \
                               case 3: conv1_mx_t<B, 3>(x, xs, w, b, y, gp); break; case 4: conv1_mx_t<B, 4>(x, xs, w, b, y, gp); break; \
                               case 5: conv1_mx_t<B, 5>(x, xs, w, b, y, gp); break; case 6: conv1_mx_t<B, 6>(x, xs, w, b, y, gp); break; \
                               case 7: conv1_mx_t<B, 7>(x, xs, w, b, y, gp); break; default: conv1_mx_t<B, 8>(x, xs, w, b, y, gp); } } while (0)
        if (xdt == 4) C1T(4); else C1T(8);
#undef C1T
    } else if (xdt == 4) conv1_mx_k<4><<<nblk_((size_t)xs.n * S, 256), 256>>>((const uint8_t *)x, w, b, y, xs.n, xs.c, cout, S, gp);
    else conv1_mx_k<8><<<nblk_((size_t)xs.n * S, 256), 256>>>((const uint8_t *)x, w, b, y, xs.n, xs.c, cout, S, gp);
    LPCK();
}
/* 1^3 weight gradient with an MX input (B bits): gw[co][ci] += sum_v gy[co][v] x[ci][v] (Ci x Co <= 64); block partials via atomics */
template <int B, typename TG>
__global__ void __launch_bounds__(256) conv_bwd_w1_mx_k(const uint8_t *x, const TG *gy, float *gw, int N, int Ci, int Co, size_t S, gnp_t gp) {
    const int bw = mx_bw(Ci), nb = mx_nb(Ci);
    const uint8_t *sc = mx_sc<B>(x, N, Ci, S);
    float acc[64] = {};
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < (size_t)N * S; i += (size_t)gridDim.x * blockDim.x) {
        int n = (int)(i / S); size_t v = i % S;
        float gv[4];
        for (int co = 0; co < Co && co < 4; co++) gv[co] = ldx(gy, ((size_t)n * Co + co) * S + v);
        for (int blk = 0; blk < nb; blk++) {
            float r[32];
            mx_load_row_b<B>(x, sc, ((size_t)n * nb + blk) * S + v, bw, r);
            row_gn(r, gp, n, blk, bw, Ci);
            for (int k = 0; k < bw; k++) { int ci = blk * bw + k; if (ci < Ci) for (int co = 0; co < Co && co < 4; co++) acc[ci * Co + co] += r[k] * gv[co]; }
        }
    }
    __shared__ float red[64];
    if (threadIdx.x < 64) red[threadIdx.x] = 0.f;
    __syncthreads();
    for (int k = 0; k < Ci * Co && k < 64; k++) {
        float a = acc[k];
        for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o);
        if ((threadIdx.x & 31) == 0) atomicAdd(&red[k], a);
    }
    __syncthreads();
    if (threadIdx.x < Ci * Co && threadIdx.x < 64) { int ci = threadIdx.x / Co, co = threadIdx.x % Co; atomicAdd(&gw[(size_t)co * Ci + ci], red[threadIdx.x]); }
}
/* fast path: one MX block row (Ci <= KM <= 32) and up to CO outputs (the head; Co <= CO at run time): static accumulator
   indices (the generic kernel's acc[ci * Co + co] lived in local memory), per-sample GN coefficients in shared memory;
   grid (voxel chunks, sample). KM = 16 keeps the 8-output accumulators (affinity heads) in registers. */
template <int B, typename TG, int CO, int KM = 32>
__global__ void __launch_bounds__(256) conv_bwd_w1_mx1_k(const uint8_t *x, const TG *gy, float *gw, int N, int Ci, size_t S, gnp_t gp, int nper, int Co) {
    const int bw = mx_bw(Ci), n = blockIdx.y;
    __shared__ float ca[32], cb[32], red[32 * CO];
    if (threadIdx.x < 32) {
        const int k = threadIdx.x;
        float a = 1.f, bb = 0.f;
        if (gp.G && k < Ci) { const int ng = n * gp.G + k / (Ci / gp.G); a = gp.rstd[ng] * gp.gamma[k]; bb = gp.beta[k] - gp.mean[ng] * a; }
        ca[k] = a; cb[k] = bb;
    }
    if (threadIdx.x < 32 * CO) red[threadIdx.x] = 0.f;
    __syncthreads();
    const uint8_t *sc = mx_sc<B>(x, N, Ci, S);
    float acc[KM][CO];
#pragma unroll
    for (int k = 0; k < KM; k++)
#pragma unroll
        for (int co = 0; co < CO; co++) acc[k][co] = 0.f;
    const size_t v0 = (size_t)blockIdx.x * nper, v1 = min(S, v0 + nper);
    for (size_t v = v0 + threadIdx.x; v < v1; v += 256) {
        float r[32], g[CO];
        mx_load_row_b<B>(x, sc, (size_t)n * S + v, bw, r);
#pragma unroll
        for (int co = 0; co < CO; co++) g[co] = co < Co ? ldx(gy, ((size_t)n * Co + co) * S + v) : 0.f;
#pragma unroll
        for (int k = 0; k < KM; k++) {
            if (k < Ci) {
                const float xv = gp.G ? act_ab(r[k], ca[k], cb[k], true) : r[k];
#pragma unroll
                for (int co = 0; co < CO; co++) acc[k][co] += xv * g[co];
            }
        }
    }
#pragma unroll
    for (int k = 0; k < KM; k++) {
        if (k < Ci) {
#pragma unroll
            for (int co = 0; co < CO; co++) {
                if (co >= Co) continue;
                float a = acc[k][co];
                for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o);
                if ((threadIdx.x & 31) == 0) atomicAdd(&red[k * CO + co], a);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < Ci * CO) { const int k = threadIdx.x / CO, co = threadIdx.x % CO; if (co < Co) atomicAdd(&gw[(size_t)co * Ci + k], red[threadIdx.x]); }
}
/* tensor-core head weight gradient for one 16-channel MX block row and up to 8 outputs with fp16 gy (the training
   storage): gw[co][ci] += sum_v act(x)[ci][v] gy[co][v], and (gb) gb[co] += sum_v gy[co][v] in the same pass over gy.
   Per warp and 16 voxels: lanes 0-15 dequantise one voxel row each and apply the GN+SiLU affine into fp16 A[ci][voxel],
   lanes 16-31 copy that voxel's gy into B[co][voxel] (and sum it for the bias); one m16n8k16 MMA with fp32 accumulation.
   The fp16 rounding of act(x) is the only numerical difference from conv_bwd_w1_mx1_k. */
template <int B, int CI>   /* CI = 16 or 32 input channels (one MX block row): CI / 16 MMA row tiles per 16 voxels */
__global__ void __launch_bounds__(256) conv_bwd_w1_tc_k(const uint8_t *x, const __half *gy, float *gw, float *gb, int N, size_t S, gnp_t gp, int nper, int Co) {
    constexpr int MTI = CI / 16;
    __shared__ float ca[CI], cb[CI], red[8][CI * 8 + 8];
    __shared__ __align__(16) __half sA[8][CI][40], sB[8][8][40];   /* [warp][ci][voxel], [warp][co][voxel] (32 voxels); rows padded */
    const int n = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (threadIdx.x < CI) {
        const int k = threadIdx.x;
        float a = 1.f, bb = 0.f;
        if (gp.G) { const int ng = n * gp.G + k / (CI / gp.G); a = gp.rstd[ng] * gp.gamma[k]; bb = gp.beta[k] - gp.mean[ng] * a; }
        ca[k] = a; cb[k] = bb;
    }
    __syncthreads();
    const uint8_t *sc = mx_sc<B>(x, N, CI, S);
    float c[MTI][4], bs[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int m = 0; m < MTI; m++) c[m][0] = c[m][1] = c[m][2] = c[m][3] = 0.f;
    const int g = lane >> 2, t4 = lane & 3;
    const size_t v0 = (size_t)blockIdx.x * nper, v1 = min(S, v0 + nper);
    for (size_t vb = v0 + warp * 32; vb < v1; vb += 8 * 32) {   /* 32 voxels per warp step: lane = voxel (its x row and its gy) */
        const size_t v = vb + lane; const bool in = v < v1;
        {
            float r[32];
            if (in) mx_load_row_b<B>(x, sc, (size_t)n * S + v, CI, r);
#pragma unroll
            for (int k = 0; k < CI; k++) sA[warp][k][lane] = __float2half_rn(in ? (gp.G ? act_ab(r[k], ca[k], cb[k], true) : r[k]) : 0.f);
#pragma unroll
            for (int co = 0; co < 8; co++) {
                const __half h = in && co < Co ? gy[((size_t)n * Co + co) * S + v] : __float2half_rn(0.f);
                sB[warp][co][lane] = h; bs[co] += __half2float(h);
            }
        }
        __syncwarp();
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {
            unsigned b[2];
            b[0] = *(const unsigned *)&sB[warp][g][16 * ks + 2 * t4];     b[1] = *(const unsigned *)&sB[warp][g][16 * ks + 2 * t4 + 8];
#pragma unroll
            for (int m = 0; m < MTI; m++) {
                unsigned a[4];
                a[0] = *(const unsigned *)&sA[warp][16 * m + g][16 * ks + 2 * t4];     a[1] = *(const unsigned *)&sA[warp][16 * m + g + 8][16 * ks + 2 * t4];
                a[2] = *(const unsigned *)&sA[warp][16 * m + g][16 * ks + 2 * t4 + 8]; a[3] = *(const unsigned *)&sA[warp][16 * m + g + 8][16 * ks + 2 * t4 + 8];
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+f"(c[m][0]), "+f"(c[m][1]), "+f"(c[m][2]), "+f"(c[m][3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            }
        }
        __syncwarp();
    }
    /* C[ci][co]: c0, c1 = C[16 m + g][2 t4 .. + 1], c2, c3 = C[16 m + g + 8][2 t4 .. + 1] */
#pragma unroll
    for (int m = 0; m < MTI; m++) {
        red[warp][(16 * m + g) * 8 + 2 * t4] = c[m][0]; red[warp][(16 * m + g) * 8 + 2 * t4 + 1] = c[m][1];
        red[warp][(16 * m + g + 8) * 8 + 2 * t4] = c[m][2]; red[warp][(16 * m + g + 8) * 8 + 2 * t4 + 1] = c[m][3];
    }
#pragma unroll
    for (int co = 0; co < 8; co++) { float s = bs[co]; for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o); bs[co] = s; }
    if (lane == 0) for (int co = 0; co < 8; co++) red[warp][CI * 8 + co] = bs[co];
    __syncthreads();
    for (int i = threadIdx.x; i < CI * 8 + 8; i += blockDim.x) {
        float s = 0.f;
        for (int w = 0; w < 8; w++) s += red[w][i];
        if (i < CI * 8) { const int ci = i / 8, co = i % 8; if (co < Co) atomicAdd(&gw[(size_t)co * CI + ci], s); }
        else if (gb && i - CI * 8 < Co) atomicAdd(&gb[i - CI * 8], s);
    }
}
/* returns 1 when the bias gradient (gb) was accumulated as well (tensor-core head path), else 0 (the caller adds it) */
extern "C" int lp_bwd_w1_mx_b(const void *x, int xdt, shape5 xs, const void *gy, int gydt, shape5 ys, float *gw, float *gb, gnp_t gp) {
    static int old = -1; if (old < 0) old = getenv("UFSM_CONV1_OLD") ? atoi(getenv("UFSM_CONV1_OLD")) : 0;
    if (!old && (xs.c == 16 || xs.c == 32) && ys.c <= 8 && gydt == 2 && (!gp.G || xs.c % gp.G == 0)) {
        const size_t S = shape_spatial(xs);
        const int nper = 8192;
        const dim3 grid((unsigned)((S + nper - 1) / nper), xs.n);
        if (xs.c == 32) {   /* 32 inputs: the CUDA-core kernel kept 32 x 8 accumulators per thread (spilled; 14.4 vs ~2 ms at 384^3) */
            if (xdt == 4) conv_bwd_w1_tc_k<4, 32><<<grid, 256>>>((const uint8_t *)x, (const __half *)gy, gw, gb, xs.n, S, gp, nper, ys.c);
            else conv_bwd_w1_tc_k<8, 32><<<grid, 256>>>((const uint8_t *)x, (const __half *)gy, gw, gb, xs.n, S, gp, nper, ys.c);
        } else if (xdt == 4) conv_bwd_w1_tc_k<4, 16><<<grid, 256>>>((const uint8_t *)x, (const __half *)gy, gw, gb, xs.n, S, gp, nper, ys.c);
        else conv_bwd_w1_tc_k<8, 16><<<grid, 256>>>((const uint8_t *)x, (const __half *)gy, gw, gb, xs.n, S, gp, nper, ys.c);
        LPCK();
        return gb != nullptr;
    }
    lp_bwd_w1_mx(x, xdt, xs, gy, gydt, ys, gw, gp);
    return 0;
}
extern "C" void lp_bwd_w1_mx(const void *x, int xdt, shape5 xs, const void *gy, int gydt, shape5 ys, float *gw, gnp_t gp) {
    if (xs.c <= 32 && ys.c <= 8) {
        const size_t S = shape_spatial(xs);
        const int nper = 8192, nbx = (int)((S + nper - 1) / nper);
        const dim3 grid(nbx, xs.n);
        const uint8_t *xq = (const uint8_t *)x;
#define BW1F(B, CO, KM) do { if (gydt == 2) conv_bwd_w1_mx1_k<B, __half, CO, KM><<<grid, 256>>>(xq, (const __half *)gy, gw, xs.n, xs.c, S, gp, nper, ys.c); \
                         else if (gydt == 1) conv_bwd_w1_mx1_k<B, bf16, CO, KM><<<grid, 256>>>(xq, (const bf16 *)gy, gw, xs.n, xs.c, S, gp, nper, ys.c); \
                         else conv_bwd_w1_mx1_k<B, float, CO, KM><<<grid, 256>>>(xq, (const float *)gy, gw, xs.n, xs.c, S, gp, nper, ys.c); } while (0)
        if (ys.c > 2 && xs.c <= 16) { if (xdt == 4) BW1F(4, 8, 16); else BW1F(8, 8, 16); }   /* affinity heads: up to 8 outputs */
        else if (ys.c > 2) { if (xdt == 4) BW1F(4, 8, 32); else BW1F(8, 8, 32); }
        else if (xdt == 4) { if (ys.c == 2) BW1F(4, 2, 32); else BW1F(4, 1, 32); }
        else { if (ys.c == 2) BW1F(8, 2, 32); else BW1F(8, 1, 32); }
#undef BW1F
        LPCK();
        return;
    }
    if (xs.c * ys.c > 64 || ys.c > 4) { fprintf(stderr, "lp_bwd_w1_mx: %d x %d channels unsupported\n", xs.c, ys.c); abort(); }
    size_t S = shape_spatial(xs);
    int nbk = (int)(((size_t)xs.n * S + 4095) / 4096); if (nbk > 1024) nbk = 1024;
    const uint8_t *xq = (const uint8_t *)x;
#define BW1(B) do { if (gydt == 2) conv_bwd_w1_mx_k<B, __half><<<nbk, 256>>>(xq, (const __half *)gy, gw, xs.n, xs.c, ys.c, S, gp); \
                    else if (gydt == 1) conv_bwd_w1_mx_k<B, bf16><<<nbk, 256>>>(xq, (const bf16 *)gy, gw, xs.n, xs.c, ys.c, S, gp); \
                    else conv_bwd_w1_mx_k<B, float><<<nbk, 256>>>(xq, (const float *)gy, gw, xs.n, xs.c, ys.c, S, gp); } while (0)
    if (xdt == 4) BW1(4); else BW1(8);
#undef BW1
    LPCK();
}
/* GroupNorm sums of an MX tensor (B bits): per (n, group) sum and sum of squares (double, accumulated); a block = 256 voxels of
   one sample, all channel blocks; per-group partials in shared memory */
