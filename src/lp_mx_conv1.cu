/* 1x1 convolution forward and weight gradient on MX tensors */
#include "lp_mxops.cuh"
#include "lp_slab.cuh"
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
        row_gn(r, gp, gn_row_v(gp, n, v), blk, bw, Ci);
        for (int k = 0; k < bw; k++) { int ci = blk * bw + k; if (ci < Ci) for (int co = 0; co < Co && co < 8; co++) acc[co] += w[co * Ci + ci] * r[k]; }
    }
    for (int co = 0; co < Co && co < 8; co++) y[((size_t)n * Co + co) * S + v] = acc[co];
}
/* the same with the output count a template parameter (accumulators in registers) and the gn+silu coefficients (per n, ci),
   the weights and the bias staged once per block in smem instead of reloaded per voxel; same arithmetic order as conv1_mx_k.
   Weights [ci][CP] (CP / 4 broadcast float4 loads per input channel) and coefficients float2 [n][ci]: one scalar smem load per
   (co, ci) made the kernel smem-instruction-bound. CO <= 8 (CP 8) or 16 (CP 16: 9..16 outputs, Co at run time). */
template <int B, int CO, typename YT = float>   /* YT: fp32 or fp16 logits (lean training: the logit gradient fits beside fp16 logits) */
__global__ void __launch_bounds__(256) conv1_mx_t_k(const uint8_t *x, const float *w, const float *b, YT *y, int N, int Ci, size_t S, gnp_t gp, int Co,
                                                    size_t vb, size_t Sy) {   /* voxels [vb, vb + Sy) of x into y planes of Sy voxels (head slabs) */
    constexpr int CP = CO <= 8 ? 8 : 16, C4 = CP / 4;
    extern __shared__ __align__(16) float c1s[];
    float4 *sw = (float4 *)c1s;                       /* [Ci][C4] (co 0..3, 4..7, ..) */
    float2 *sab = (float2 *)(sw + C4 * Ci);           /* [N][Ci] (a, b); slab GroupNorm: [2][Ci], the block's first slab and the next */
    const int nb0 = blockIdx.y;
    const int r0 = gp.ns ? gn_row_v(gp, nb0, min(vb + (size_t)blockIdx.x * blockDim.x, S - 1)) : 0;
    for (int i = threadIdx.x; i < (gp.ns ? 2 : N) * Ci; i += blockDim.x) {
        const int n = i / Ci, ci = i % Ci;
        if (gp.G) sab[i] = gn_ab(gp, gp.ns ? min(r0 + n, nb0 * gp.ns + gp.ns - 1) : n, ci, Ci);
    }
    for (int i = threadIdx.x; i < CP * Ci; i += blockDim.x) { const int ci = i / CP, co = i % CP; c1s[i] = co < Co ? w[co * Ci + ci] : 0.f; }
    __syncthreads();
    const int bw = mx_bw(Ci), nb = mx_nb(Ci);
    const int n = blockIdx.y; const size_t vl = blockIdx.x * (size_t)blockDim.x + threadIdx.x, v = vb + vl;   /* grid (voxels, sample) */
    if (vl >= Sy || v >= S) return;
    float acc[CO];
#pragma unroll
    for (int co = 0; co < CO; co++) acc[co] = b && co < Co ? b[co] : 0.f;
    const uint8_t *sc = mx_sc<B>(x, N, Ci, S);
    const float2 *sabv = sab + (gp.ns ? (gn_row_v(gp, n, v) - r0) * Ci : n * Ci);   /* (slab GroupNorm: this voxel's bank) */
    for (int blk = 0; blk < nb; blk++) {
        float r[32];
        mx_load_row_b<B>(x, sc, ((size_t)n * nb + blk) * S + v, bw, r);
#pragma unroll
        for (int k = 0; k < 32; k++) {
            const int ci = blk * bw + k;
            if (k >= bw || ci >= Ci) break;
            float xv = r[k];
            if (gp.G) { const float2 ab = sabv[ci]; xv = act_ab(xv, ab.x, ab.y, true); }
            float wv[CP];
#pragma unroll
            for (int j = 0; j < C4; j++) { const float4 q = sw[C4 * ci + j]; wv[4 * j] = q.x; wv[4 * j + 1] = q.y; wv[4 * j + 2] = q.z; wv[4 * j + 3] = q.w; }
#pragma unroll
            for (int co = 0; co < CO; co++) acc[co] += wv[co] * xv;
        }
    }
#pragma unroll
    for (int co = 0; co < CO; co++) {
        if (co >= Co) break;
        if constexpr (sizeof(YT) == 2) y[((size_t)n * Co + co) * Sy + vl] = __float2half(acc[co]); else y[((size_t)n * Co + co) * Sy + vl] = acc[co];
    }
}
/* voxels per block of the head's weight-gradient kernels: at most one slab (slab GroupNorm: a block's voxels span <= 2 slabs, its
   two coefficient banks) */
static int slab_nper(const gnp_t &gp, int nper) {
    if (!gp.ns || !gp.G) return nper;
    const long long sv = (long long)gp.zs * gp.zhw;
    return sv < nper ? (int)sv : nper;
}
static bool slab_small(const gnp_t &gp) { return gp.ns && gp.G && (long long)gp.zs * gp.zhw < 256; }   /* staged forward: >= 256 voxels per slab */
static size_t conv1_smem(shape5 xs, int cout) { return (size_t)(2 * (xs.n > 2 ? xs.n : 2) * xs.c + (cout <= 8 ? 8 : 16) * xs.c) * sizeof(float); }   /* (>= 2 coefficient rows: slab banks) */
template <int B, int CO, typename YT = float> static void conv1_mx_t(const void *x, shape5 xs, const float *w, const float *b, YT *y, gnp_t gp, int cout, size_t vb = 0, size_t nv = 0) {
    const size_t S = shape_spatial(xs), sm = conv1_smem(xs, cout), Sy = nv ? nv : S;
    conv1_mx_t_k<B, CO, YT><<<dim3(nblk_(Sy, 256), xs.n), 256, sm>>>((const uint8_t *)x, w, b, y, xs.n, xs.c, S, gp, cout, vb, Sy);
}
/* the same with fp16 logits (unet lean training; the staged path only) */
extern "C" void lp_conv1_fwd_mx_h16(const void *x, int xdt, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp) {
    lp_conv1_fwd_mx_h16_r(x, xdt, xs, w, b, cout, y, gp, 0, 0);
}
/* voxels [vb, vb + nv) of x (nv 0: all) into fp16 planes of nv voxels: the trainer's head slabs (train.c head_slab) */
extern "C" void lp_conv1_fwd_mx_h16_r(const void *x, int xdt, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, size_t vb, size_t nv) {
    if (cout > 16 || conv1_smem(xs, cout) > 32 * 1024) { fprintf(stderr, "lp_conv1_fwd_mx_h16: cout %d / %d inputs unsupported\n", cout, xs.c); abort(); }
    if (slab_small(gp)) { fprintf(stderr, "lp_conv1_fwd_mx_h16: slab GroupNorm with %d voxels per slab (< 256)\n", gp.zs * gp.zhw); abort(); }
    __half *yh = (__half *)y;
#define C1H(B) do { switch (cout) { case 1: conv1_mx_t<B, 1>(x, xs, w, b, yh, gp, cout, vb, nv); break; case 2: conv1_mx_t<B, 2>(x, xs, w, b, yh, gp, cout, vb, nv); break; \
                               case 3: conv1_mx_t<B, 3>(x, xs, w, b, yh, gp, cout, vb, nv); break; case 4: conv1_mx_t<B, 4>(x, xs, w, b, yh, gp, cout, vb, nv); break; \
                               case 5: conv1_mx_t<B, 5>(x, xs, w, b, yh, gp, cout, vb, nv); break; case 6: conv1_mx_t<B, 6>(x, xs, w, b, yh, gp, cout, vb, nv); break; \
                               case 7: conv1_mx_t<B, 7>(x, xs, w, b, yh, gp, cout, vb, nv); break; case 8: conv1_mx_t<B, 8>(x, xs, w, b, yh, gp, cout, vb, nv); break; \
                               default: conv1_mx_t<B, 16>(x, xs, w, b, yh, gp, cout, vb, nv); } } while (0)
    if (xdt == 4) C1H(4); else C1H(8);
#undef C1H
    LPCK();
}
extern "C" void lp_conv1_fwd_mx(const void *x, int xdt, shape5 xs, const float *w, const float *b, int cout, float *y, gnp_t gp) {
    if (cout > 16) { fprintf(stderr, "lp_conv1_fwd_mx: cout %d > 16\n", cout); abort(); }
    size_t S = shape_spatial(xs);
    static int old = -1; if (old < 0) old = getenv("UFSM_CONV1_OLD") ? atoi(getenv("UFSM_CONV1_OLD")) : 0;
    if (cout > 8 && (old || conv1_smem(xs, cout) > 32 * 1024)) { fprintf(stderr, "lp_conv1_fwd_mx: %d outputs need the staged kernel\n", cout); abort(); }
    if (!old && conv1_smem(xs, cout) <= 32 * 1024 && !slab_small(gp)) {
#define C1T(B) do { switch (cout) { case 1: conv1_mx_t<B, 1>(x, xs, w, b, y, gp, cout); break; case 2: conv1_mx_t<B, 2>(x, xs, w, b, y, gp, cout); break; \
                               case 3: conv1_mx_t<B, 3>(x, xs, w, b, y, gp, cout); break; case 4: conv1_mx_t<B, 4>(x, xs, w, b, y, gp, cout); break; \
                               case 5: conv1_mx_t<B, 5>(x, xs, w, b, y, gp, cout); break; case 6: conv1_mx_t<B, 6>(x, xs, w, b, y, gp, cout); break; \
                               case 7: conv1_mx_t<B, 7>(x, xs, w, b, y, gp, cout); break; case 8: conv1_mx_t<B, 8>(x, xs, w, b, y, gp, cout); break; \
                               default: conv1_mx_t<B, 16>(x, xs, w, b, y, gp, cout); } } while (0)
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
            row_gn(r, gp, gn_row_v(gp, n, v), blk, bw, Ci);
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
template <int B, typename TG, int CO, int KM = 32>   /* outputs co0 .. co0 + CO - 1 of Co (heads wider than CO: one launch per CO outputs) */
__global__ void __launch_bounds__(256) conv_bwd_w1_mx1_k(const uint8_t *x, const TG *gy, float *gw, int N, int Ci, size_t S, gnp_t gp, int nper, int Co, int co0 = 0, split_t sp = {}) {
    const int nbx = (int)((S + nper - 1) / nper), bid = blockIdx.x + sp.wpart_x0;
    const int bw = mx_bw(Ci), n = sp.wpart ? bid / nbx : blockIdx.y, bx = sp.wpart ? bid % nbx : blockIdx.x;
    if (sp.wpart) { const size_t nw = (size_t)Co * Ci; gw = sp.wpart + (size_t)blockIdx.x * (nw + Co); }
    __shared__ float ca[2][32], cb[2][32], red[8][32 * CO];   /* [bank]: slab GroupNorm, the block's first slab and the next (nper <= a slab) */
    const int r0 = gp.ns ? gn_row_v(gp, n, min((size_t)bx * nper, S - 1)) : 0;
    if (threadIdx.x < 64) {
        const int k = threadIdx.x & 31, bk = threadIdx.x >> 5;
        float a = 1.f, bb = 0.f;
        if (gp.G && k < Ci) { const float2 ab = gn_ab(gp, gp.ns ? min(r0 + bk, n * gp.ns + gp.ns - 1) : n, k, Ci); a = ab.x; bb = ab.y; }
        ca[bk][k] = a; cb[bk][k] = bb;
    }
    for (int i = threadIdx.x; i < 8 * 32 * CO; i += blockDim.x) ((float *)red)[i] = 0.f;
    __syncthreads();
    const uint8_t *sc = mx_sc<B>(x, N, Ci, S);
    float acc[KM][CO];
#pragma unroll
    for (int k = 0; k < KM; k++)
#pragma unroll
        for (int co = 0; co < CO; co++) acc[k][co] = 0.f;
    const size_t v0 = (size_t)bx * nper, v1 = min(S, v0 + nper);
    for (size_t v = v0 + threadIdx.x; v < v1; v += 256) {
        float r[32], g[CO];
        mx_load_row_b<B>(x, sc, (size_t)n * S + v, bw, r);
        const int bk = gp.ns ? gn_row_v(gp, n, v) - r0 : 0;
        const float *cav = ca[bk], *cbv = cb[bk];
#pragma unroll
        for (int co = 0; co < CO; co++) g[co] = co0 + co < Co ? ldx(gy, ((size_t)n * Co + co0 + co) * S + v) : 0.f;
#pragma unroll
        for (int k = 0; k < KM; k++) {
            if (k < Ci) {
                const float xv = gp.G ? act_ab(r[k], cav[k], cbv[k], true) : r[k];
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
                if (co0 + co >= Co) continue;
                float a = acc[k][co];
                for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o);
                if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5][k * CO + co] = a;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < Ci * CO) { const int k = threadIdx.x / CO, co = threadIdx.x % CO; if (co0 + co < Co) { float sum = 0.f; for (int w = 0; w < 8; w++) sum += red[w][threadIdx.x]; atomicAdd(&gw[(size_t)(co0 + co) * Ci + k], sum); } }
}
/* tensor-core head weight gradient for one 16-channel MX block row and up to 8 outputs with fp16 gy (the training
   storage): gw[co][ci] += sum_v act(x)[ci][v] gy[co][v], and (gb) gb[co] += sum_v gy[co][v] in the same pass over gy.
   Per warp and 16 voxels: lanes 0-15 dequantise one voxel row each and apply the GN+SiLU affine into fp16 A[ci][voxel],
   lanes 16-31 copy that voxel's gy into B[co][voxel] (and sum it for the bias); one m16n8k16 MMA with fp32 accumulation.
   The fp16 rounding of act(x) is the only numerical difference from conv_bwd_w1_mx1_k. */
template <int B, int CI>   /* CI = 16 or 32 input channels (one MX block row, block blk of Ct): CI / 16 MMA row tiles per 16 voxels; outputs co0 .. co0 + 7 */
__global__ void __launch_bounds__(256) conv_bwd_w1_tc_k(const uint8_t *x, const __half *gy, float *gw, float *gb, int N, size_t S, gnp_t gp, int nper, int Co, int co0, int Ct, int blk, split_t sp = {}) {
    constexpr int MTI = CI / 16;
    __shared__ float ca[2][CI], cb[2][CI], red[8][CI * 8 + 8];   /* [bank]: slab GroupNorm, the block's first slab and the next */
    __shared__ __align__(16) __half sA[8][CI][40], sB[8][8][40];   /* [warp][ci][voxel], [warp][co][voxel] (32 voxels); rows padded */
    const int nbx = (int)((S + nper - 1) / nper), bid = blockIdx.x + sp.wpart_x0;
    const int n = sp.wpart ? bid / nbx : (int)blockIdx.y, bx = sp.wpart ? bid % nbx : (int)blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (sp.wpart) { const size_t nw = (size_t)Ct * Co; gw = sp.wpart + (size_t)blockIdx.x * (nw + Co); if (gb) gb = gw + nw; }
    const int r0 = gp.ns ? gn_row_v(gp, n, min((size_t)bx * nper, S - 1)) : 0;
    if (threadIdx.x < 2 * CI) {
        const int k = threadIdx.x % CI, bk = threadIdx.x / CI, kc = blk * CI + k;
        float a = 1.f, bb = 0.f;
        if (gp.G) { const float2 ab = gn_ab(gp, gp.ns ? min(r0 + bk, n * gp.ns + gp.ns - 1) : n, kc, Ct); a = ab.x; bb = ab.y; }
        ca[bk][k] = a; cb[bk][k] = bb;
    }
    __syncthreads();
    const uint8_t *sc = mx_sc<B>(x, N, Ct, S);
    float c[MTI][4], bs[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int m = 0; m < MTI; m++) c[m][0] = c[m][1] = c[m][2] = c[m][3] = 0.f;
    const int g = lane >> 2, t4 = lane & 3;
    const size_t v0 = (size_t)bx * nper, v1 = min(S, v0 + nper);
    for (size_t vb = v0 + warp * 32; vb < v1; vb += 8 * 32) {   /* 32 voxels per warp step: lane = voxel (its x row and its gy) */
        const size_t v = vb + lane; const bool in = v < v1;
        {
            float r[32];
            if (in) mx_load_row_b<B>(x, sc, ((size_t)n * (Ct / CI) + blk) * S + v, CI, r);
            const int bk = gp.ns && in ? gn_row_v(gp, n, v) - r0 : 0;
            const float *cav = ca[bk], *cbv = cb[bk];
#pragma unroll
            for (int k = 0; k < CI; k++) sA[warp][k][lane] = __float2half_rn(in ? (gp.G ? act_ab(r[k], cav[k], cbv[k], true) : r[k]) : 0.f);
#pragma unroll
            for (int co = 0; co < 8; co++) {
                const __half h = in && co0 + co < Co ? gy[((size_t)n * Co + co0 + co) * S + v] : __float2half_rn(0.f);
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
        if (i < CI * 8) { const int ci = i / 8, co = i % 8; if (co0 + co < Co) atomicAdd(&gw[(size_t)(co0 + co) * Ct + blk * CI + ci], s); }
        else if (gb && !blk && co0 + i - CI * 8 < Co) atomicAdd(&gb[co0 + i - CI * 8], s);
    }
}
/* the whole head backward in one pass over the 32-channel MX input x and the fp16 logit gradient gy (up to 16 logits):
   weight and bias gradients as conv_bwd_w1_tc_k (two m16n8k16 column tiles: all outputs at once), and the input gradient
   gx[ci][v] = sum_co w[co][ci] gy[co][v] (the gradient of act(x), the head's input) on the CUDA cores into an MX row (YB bits,
   exact SR keyed by sr for fp4), as conv1_to_mx_t_k. The separate kernels read x twice (8 outputs per pass) and gy twice. */
template <int B, int YB>
__global__ void __launch_bounds__(256, 2) conv1_bwd_fused_k(const uint8_t *x, const __half *gy, const float *w, float *gw, float *gb, uint8_t *gx, int N, size_t S, gnp_t gp,
                                                      int nper, int Co, unsigned sr, size_t v00, size_t Sy, split_t sp = {}) {   /* voxels [v00, v00 + Sy); gy planes of Sy voxels */
    constexpr int CI = 32;
    __shared__ float ca[2][CI], cb[2][CI];   /* [bank]: slab GroupNorm, the block's first slab and the next */
    __shared__ __align__(16) float sw[16][CI];                   /* w[co][ci] (rows >= Co zero) */
    __shared__ __align__(16) __half sA[8][CI][40], sB[8][16][40];   /* [warp][ci][voxel], [warp][co][voxel] */
    const int nbx = (int)((Sy + nper - 1) / nper), bid = blockIdx.x + sp.wpart_x0;
    const int n = sp.wpart ? bid / nbx : (int)blockIdx.y, bx = sp.wpart ? bid % nbx : (int)blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (sp.wpart) { const size_t nw = (size_t)CI * Co; gw = sp.wpart + (size_t)blockIdx.x * (nw + Co); if (gb) gb = gw + nw; }
    const int r0 = gp.ns ? gn_row_v(gp, n, min(v00 + (size_t)bx * nper, S - 1)) : 0;
    if (threadIdx.x < 2 * CI) {
        const int k = threadIdx.x % CI, bk = threadIdx.x / CI;
        float a = 1.f, bb = 0.f;
        if (gp.G) { const float2 ab = gn_ab(gp, gp.ns ? min(r0 + bk, n * gp.ns + gp.ns - 1) : n, k, CI); a = ab.x; bb = ab.y; }
        ca[bk][k] = a; cb[bk][k] = bb;
    }
    for (int i = threadIdx.x; i < 16 * CI; i += 256) sw[i / CI][i % CI] = i / CI < Co ? w[i] : 0.f;
    __syncthreads();
    const uint8_t *sc = mx_sc<B>(x, N, CI, S);
    uint8_t *gsc = mx_sc<YB>(gx, N, CI, S);
    float c[3][2][4];   /* m 2: the bias sums, from an A tile whose row 0 is ones (lanes g == 0 hold them) */
#pragma unroll
    for (int m = 0; m < 3; m++)
#pragma unroll
        for (int q = 0; q < 2; q++) c[m][q][0] = c[m][q][1] = c[m][q][2] = c[m][q][3] = 0.f;
    const int g = lane >> 2, t4 = lane & 3;
    const size_t v0 = v00 + (size_t)bx * nper, v1 = min(min(S, v00 + Sy), v0 + nper);
    for (size_t vb = v0 + warp * 32; vb < v1; vb += 8 * 32) {   /* 32 voxels per warp step: lane = voxel */
        const size_t v = vb + lane; const bool in = v < v1;
        {
            float r[32], gv[16], o[32];
            if (in) mx_load_row_b<B>(x, sc, (size_t)n * S + v, CI, r);
            const int bk = gp.ns && in ? gn_row_v(gp, n, v) - r0 : 0;
            const float *cav = ca[bk], *cbv = cb[bk];
#pragma unroll
            for (int k = 0; k < CI; k++) sA[warp][k][lane] = __float2half_rn(in ? (gp.G ? act_ab(r[k], cav[k], cbv[k], true) : r[k]) : 0.f);
#pragma unroll
            for (int co = 0; co < 16; co++) {
                const __half h = in && co < Co ? gy[((size_t)n * Co + co) * Sy + (v - v00)] : __float2half_rn(0.f);
                sB[warp][co][lane] = h; gv[co] = __half2float(h);
            }
#pragma unroll
            for (int k = 0; k < CI; k++) o[k] = 0.f;
#pragma unroll
            for (int co = 0; co < 16; co++) {
                if (co >= Co) break;
#pragma unroll
                for (int k4 = 0; k4 < CI / 4; k4++) {
                    const float4 q = *(const float4 *)&sw[co][4 * k4];
                    o[4 * k4] = fmaf(q.x, gv[co], o[4 * k4]); o[4 * k4 + 1] = fmaf(q.y, gv[co], o[4 * k4 + 1]);
                    o[4 * k4 + 2] = fmaf(q.z, gv[co], o[4 * k4 + 2]); o[4 * k4 + 3] = fmaf(q.w, gv[co], o[4 * k4 + 3]);
                }
            }
            if (in) mx_store_row_b<YB>(gx, gsc, (size_t)n * S + v, CI, o, sr);
        }
        __syncwarp();
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {
#pragma unroll
            for (int q = 0; q < 2; q++) {
                unsigned b[2];
                b[0] = *(const unsigned *)&sB[warp][8 * q + g][16 * ks + 2 * t4]; b[1] = *(const unsigned *)&sB[warp][8 * q + g][16 * ks + 2 * t4 + 8];
#pragma unroll
                for (int m = 0; m < 2; m++) {
                    unsigned a[4];
                    a[0] = *(const unsigned *)&sA[warp][16 * m + g][16 * ks + 2 * t4];     a[1] = *(const unsigned *)&sA[warp][16 * m + g + 8][16 * ks + 2 * t4];
                    a[2] = *(const unsigned *)&sA[warp][16 * m + g][16 * ks + 2 * t4 + 8]; a[3] = *(const unsigned *)&sA[warp][16 * m + g + 8][16 * ks + 2 * t4 + 8];
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                                 : "+f"(c[m][q][0]), "+f"(c[m][q][1]), "+f"(c[m][q][2]), "+f"(c[m][q][3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
                }
                {
                    const unsigned one = g == 0 ? 0x3c003c00u : 0u;   /* A row 0 = ones (k 2 t4 .. + 1 and + 8 .. + 9) */
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                                 : "+f"(c[2][q][0]), "+f"(c[2][q][1]), "+f"(c[2][q][2]), "+f"(c[2][q][3]) : "r"(one), "r"(0u), "r"(one), "r"(0u), "r"(b[0]), "r"(b[1]));
                }
            }
        }
        __syncwarp();
    }
    __syncthreads();   /* the reduction reuses sA: [warp][CI * 16 + 16] floats (8 x 528 x 4 B <= 20 KB) */
    float (*red)[CI * 16 + 16] = (float (*)[CI * 16 + 16])&sA[0][0][0];
    /* C[ci][co]: c[m][q][0..1] = C[16 m + g][8 q + 2 t4 .. + 1], c[m][q][2..3] = C[16 m + g + 8][..] */
#pragma unroll
    for (int m = 0; m < 2; m++)
#pragma unroll
        for (int q = 0; q < 2; q++) {
            red[warp][(16 * m + g) * 16 + 8 * q + 2 * t4] = c[m][q][0]; red[warp][(16 * m + g) * 16 + 8 * q + 2 * t4 + 1] = c[m][q][1];
            red[warp][(16 * m + g + 8) * 16 + 8 * q + 2 * t4] = c[m][q][2]; red[warp][(16 * m + g + 8) * 16 + 8 * q + 2 * t4 + 1] = c[m][q][3];
        }
    if (g == 0) { red[warp][CI * 16 + 2 * t4] = c[2][0][0]; red[warp][CI * 16 + 2 * t4 + 1] = c[2][0][1]; red[warp][CI * 16 + 8 + 2 * t4] = c[2][1][0]; red[warp][CI * 16 + 8 + 2 * t4 + 1] = c[2][1][1]; }
    __syncthreads();
    for (int i = threadIdx.x; i < CI * 16 + 16; i += blockDim.x) {
        float s2 = 0.f;
        for (int w2 = 0; w2 < 8; w2++) s2 += red[w2][i];
        if (i < CI * 16) { const int ci = i / 16, co = i % 16; if (co < Co) atomicAdd(&gw[(size_t)co * CI + ci], s2); }
        else if (gb && i - CI * 16 < Co) atomicAdd(&gb[i - CI * 16], s2);
    }
}
/* the fused head backward (weight, bias and input gradient in one pass); returns 0 when the shapes / types are not covered (the
   caller runs the separate kernels) */
extern "C" int lp_head_bwd_mx(const void *x, int xdt, shape5 xs, const void *gy, int gydt, shape5 ys, const float *w, float *gw, float *gb, void *gx, int gxdt, gnp_t gp, unsigned sr) {
    return lp_head_bwd_mx_r(x, xdt, xs, gy, gydt, ys, w, gw, gb, gx, gxdt, gp, sr, 0, 0);
}
/* voxels [vb, vb + nv) (nv 0: all), gy planes of nv voxels; gx rows and their SR keys stay global (head slabs: the same bits) */
extern "C" int lp_head_bwd_mx_r(const void *x, int xdt, shape5 xs, const void *gy, int gydt, shape5 ys, const float *w, float *gw, float *gb, void *gx, int gxdt, gnp_t gp,
                                unsigned sr, size_t vb, size_t nv) {
    static int on = -1; if (on < 0) on = getenv("UFSM_HEAD_FUSED") ? atoi(getenv("UFSM_HEAD_FUSED")) : 1;
    if (!on || xs.c != 32 || ys.c > 16 || gydt != 2 || (gxdt != 4 && gxdt != 8) || (xdt != 4 && xdt != 8) || (gp.G && 32 % gp.G)) return 0;
    const size_t S = shape_spatial(xs), Sy = nv ? nv : S;
    const int nper = slab_nper(gp, 8192);
    const dim3 grid((unsigned)((Sy + nper - 1) / nper), xs.n);
#define HBF(B, YB) conv1_bwd_fused_k<B, YB><<<piece, 256>>>((const uint8_t *)x, (const __half *)gy, w, gw, gb, (uint8_t *)gx, xs.n, S, gp, nper, ys.c, sr, vb, Sy, ss)
    const dim3 launch_grid = nn_get_deterministic() ? dim3(grid.x * grid.y, 1) : grid;
    det_wgrad_launch(launch_grid, gw, gb, xs.c, ys.c, 1, split_t{}, [&](dim3 piece, split_t ss) {
        if (xdt == 4) { if (gxdt == 4) HBF(4, 4); else HBF(4, 8); }
        else { if (gxdt == 4) HBF(8, 4); else HBF(8, 8); }
    });
#undef HBF
    LPCK();
    return 1;
}
/* returns 1 when the bias gradient (gb) was accumulated as well (tensor-core head path), else 0 (the caller adds it) */
extern "C" int lp_bwd_w1_mx_b(const void *x, int xdt, shape5 xs, const void *gy, int gydt, shape5 ys, float *gw, float *gb, gnp_t gp) {
    static int old = -1; if (old < 0) old = getenv("UFSM_CONV1_OLD") ? atoi(getenv("UFSM_CONV1_OLD")) : 0;
    if (!old && (xs.c == 16 || xs.c % 32 == 0) && ys.c <= 16 && gydt == 2 && (!gp.G || xs.c % gp.G == 0)) {   /* > 32 inputs: one pass per 32-channel block */
        const size_t S = shape_spatial(xs);
        const int nper = slab_nper(gp, 8192);
        const dim3 grid((unsigned)((S + nper - 1) / nper), xs.n);
        for (int co0 = 0; co0 < ys.c; co0 += 8)   /* 8 outputs per pass (one m16n8k16 column tile) */
            for (int blk = 0; blk < (xs.c == 16 ? 1 : xs.c / 32); blk++) {
                const dim3 launch_grid = nn_get_deterministic() ? dim3(grid.x * grid.y, 1) : grid;
                det_wgrad_launch(launch_grid, gw, gb, xs.c, ys.c, 1, split_t{}, [&](dim3 piece, split_t ss) {
                    if (xs.c >= 32) {
                        if (xdt == 4) conv_bwd_w1_tc_k<4, 32><<<piece, 256>>>((const uint8_t *)x, (const __half *)gy, gw, gb, xs.n, S, gp, nper, ys.c, co0, xs.c, blk, ss);
                        else conv_bwd_w1_tc_k<8, 32><<<piece, 256>>>((const uint8_t *)x, (const __half *)gy, gw, gb, xs.n, S, gp, nper, ys.c, co0, xs.c, blk, ss);
                    } else if (xdt == 4) conv_bwd_w1_tc_k<4, 16><<<piece, 256>>>((const uint8_t *)x, (const __half *)gy, gw, gb, xs.n, S, gp, nper, ys.c, co0, 16, 0, ss);
                    else conv_bwd_w1_tc_k<8, 16><<<piece, 256>>>((const uint8_t *)x, (const __half *)gy, gw, gb, xs.n, S, gp, nper, ys.c, co0, 16, 0, ss);
                });
            }
        LPCK();
        return gb != nullptr;
    }
    lp_bwd_w1_mx(x, xdt, xs, gy, gydt, ys, gw, gp);
    return 0;
}
extern "C" void lp_bwd_w1_mx(const void *x, int xdt, shape5 xs, const void *gy, int gydt, shape5 ys, float *gw, gnp_t gp) {
    if (xs.c <= 32 && ys.c <= 16) {
        const size_t S = shape_spatial(xs);
        const int nper = slab_nper(gp, 8192), nbx = (int)((S + nper - 1) / nper);
        const dim3 grid = nn_get_deterministic() ? dim3(nbx * xs.n, 1) : dim3(nbx, xs.n);
        const uint8_t *xq = (const uint8_t *)x;
        int co0 = 0;
#define BW1F(B, CO, KM) do { det_wgrad_launch(grid, gw, nullptr, xs.c, ys.c, 1, split_t{}, [&](dim3 piece, split_t ss) { if (gydt == 2) conv_bwd_w1_mx1_k<B, __half, CO, KM><<<piece, 256>>>(xq, (const __half *)gy, gw, xs.n, xs.c, S, gp, nper, ys.c, co0, ss); \
                         else if (gydt == 1) conv_bwd_w1_mx1_k<B, bf16, CO, KM><<<piece, 256>>>(xq, (const bf16 *)gy, gw, xs.n, xs.c, S, gp, nper, ys.c, co0, ss); \
                         else conv_bwd_w1_mx1_k<B, float, CO, KM><<<piece, 256>>>(xq, (const float *)gy, gw, xs.n, xs.c, S, gp, nper, ys.c, co0, ss); }); } while (0)
        if (ys.c > 2 && xs.c <= 16) { for (; co0 < ys.c; co0 += 8) { if (xdt == 4) BW1F(4, 8, 16); else BW1F(8, 8, 16); } }   /* affinity heads: 8 outputs per pass */
        else if (ys.c > 2) { for (; co0 < ys.c; co0 += 8) { if (xdt == 4) BW1F(4, 8, 32); else BW1F(8, 8, 32); } }
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
