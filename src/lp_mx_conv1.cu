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
   the weights and the bias staged once per block in smem instead of reloaded per voxel; same arithmetic order as conv1_mx_k */
template <int B, int CO>
__global__ void __launch_bounds__(256) conv1_mx_t_k(const uint8_t *x, const float *w, const float *b, float *y, int N, int Ci, size_t S, gnp_t gp) {
    extern __shared__ float c1s[];
    float *sa = c1s, *sb = sa + N * Ci, *sw = sb + N * Ci;   /* [N][Ci] a, b; [CO][Ci] w */
    for (int i = threadIdx.x; i < N * Ci; i += blockDim.x) {
        const int n = i / Ci, ci = i % Ci;
        if (gp.G) { const int ng = n * gp.G + ci / (Ci / gp.G); const float a = gp.rstd[ng] * gp.gamma[ci]; sa[i] = a; sb[i] = gp.beta[ci] - gp.mean[ng] * a; }
    }
    for (int i = threadIdx.x; i < CO * Ci; i += blockDim.x) sw[i] = w[i];
    __syncthreads();
    const int bw = mx_bw(Ci), nb = mx_nb(Ci);
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * S) return;
    const int n = (int)(i / S); const size_t v = i % S;
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
            const float xv = gp.G ? act_ab(r[k], sa[n * Ci + ci], sb[n * Ci + ci], true) : r[k];
#pragma unroll
            for (int co = 0; co < CO; co++) acc[co] += sw[co * Ci + ci] * xv;
        }
    }
#pragma unroll
    for (int co = 0; co < CO; co++) y[((size_t)n * CO + co) * S + v] = acc[co];
}
template <int B, int CO> static void conv1_mx_t(const void *x, shape5 xs, const float *w, const float *b, float *y, gnp_t gp) {
    const size_t S = shape_spatial(xs), sm = (size_t)(2 * xs.n * xs.c + CO * xs.c) * sizeof(float);
    conv1_mx_t_k<B, CO><<<nblk_((size_t)xs.n * S, 256), 256, sm>>>((const uint8_t *)x, w, b, y, xs.n, xs.c, S, gp);
}
extern "C" void lp_conv1_fwd_mx(const void *x, int xdt, shape5 xs, const float *w, const float *b, int cout, float *y, gnp_t gp) {
    if (cout > 8) { fprintf(stderr, "lp_conv1_fwd_mx: cout %d > 8\n", cout); abort(); }
    size_t S = shape_spatial(xs);
    static int old = -1; if (old < 0) old = getenv("UFSM_CONV1_OLD") ? atoi(getenv("UFSM_CONV1_OLD")) : 0;
    if (!old && (cout == 1 || cout == 2) && (size_t)(2 * xs.n * xs.c + 2 * xs.c) * sizeof(float) <= 32 * 1024) {
        if (xdt == 4) { if (cout == 1) conv1_mx_t<4, 1>(x, xs, w, b, y, gp); else conv1_mx_t<4, 2>(x, xs, w, b, y, gp); }
        else { if (cout == 1) conv1_mx_t<8, 1>(x, xs, w, b, y, gp); else conv1_mx_t<8, 2>(x, xs, w, b, y, gp); }
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
