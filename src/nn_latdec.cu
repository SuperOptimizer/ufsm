/* CUDA ops: the learned sub-cell decoder of a latent-grid net (unet_cfg.lat_dec, train --lat-dec H; DESIGN 2026-11-19 "learned sub-cell decoder").
   The sub-cell head (lat_guide 64) predicts per cell 64 values per guided channel with a linear 1^3 row each (nn_lat_shuffle). The decoder adds
   a nonlinear term to those rows: per cell c
       pre_k = A_k . f_c + a_k  (H more rows of the 1^3 head: the trunk's features, computed by the head kernels themselves)
             + sum_{c', t} B[c'][t][k] L[c'](c + t)     (t over the 3^3 neighbour cells: the cell's latent input and its neighbours')
       h_k = silu(pre_k),   rows[o](c) += sum_k W2[o][k] h_k   (o < 64 ng: the guided channels' sub-cell rows)
   L is the latent input itself, the quantised KLT coefficients of each 4^3 cell exactly as input_latc_k makes them before the MX rounding (or, raw,
   the cell's 64 voxels), so the decoder reads what the latent store holds and adds full-resolution detail the trunk's cells do not carry.
   W2 starts at zero: the net computes its donor's function (rows untouched, byte for byte: a zero term is never added).
   Parameters (fp32, contiguous, unet's lat-dec block after A / a): B [Cl][27][H], W2 [64 ng][H].
   Backward: from the fp16 gradient of the sub-cell rows (at the head plane scale, as the losses left them) the gradient of the pre rows
   (written as fp16 rows for the head's backward), and B / W2's gradients reduced over the cells in a fixed order (no float atomics). */
#include "nn_common.cuh"

enum { LD_TC = 32, LD_NT = 128, LD_HS = 65, LD_GA = 32, LD_GN = 32, LD_CH = 4096 };

/* the cells' latent input: L [Cl][nq][h][w] fp32 for cell planes [q0, q0 + nq) of a D x H x W fp16 window (planes outside: 0) */
__global__ void latdec_cells_k(const f16 *__restrict__ x, int D, int H, int W, int q0, int nq, const float *__restrict__ bas, const float *__restrict__ gain,
                               int Cl, float step, float *__restrict__ L) {
    const int h = H / 4, w = W / 4; const size_t pl = (size_t)h * w, nc = (size_t)nq * pl, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nc) return;
    const int q = q0 + (int)(i / pl), y = (int)(i / w % h), xx = (int)(i % w);
    if (q < 0 || q >= D / 4) { for (int c = 0; c < Cl; c++) L[(size_t)c * nc + i] = 0.f; return; }
    float cv[64];
    const size_t hw = (size_t)H * W;
#pragma unroll
    for (int k = 0; k < 64; k++) cv[k] = __half2float(x[(size_t)(4 * q + (k >> 4)) * hw + (size_t)(4 * y + (k >> 2 & 3)) * W + 4 * xx + (k & 3)]);
    if (!bas) { for (int c = 0; c < Cl; c++) L[(size_t)c * nc + i] = cv[c]; return; }
    for (int c = 0; c < Cl; c++) {   /* input_latc_k's sums and quantiser */
        float s = 0.f; for (int k = 0; k < 64; k++) s += bas[c * 64 + k] * cv[k];
        if (step > 0.f) { const float lim = c ? 127.f : 32767.f; s = fminf(lim, fmaxf(-lim, rintf(s / step))) * step; }
        L[(size_t)c * nc + i] = s * gain[c];
    }
}

/* the tile: LD_TC cells of one (q, y) row, 4 thread groups of KG hidden units each. Ls [Cl][9][LD_TC + 2]: the 3 x 3 rows of neighbour cells */
__device__ __forceinline__ void ld_stage(const float *__restrict__ L, int nq, int h, int w, int q, int y, int x0, int Cl, float *Ls) {
    const size_t pl = (size_t)h * w, nl = (size_t)(nq + 2) * pl; const int n = Cl * 9 * (LD_TC + 2);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int j = i % (LD_TC + 2), r = i / (LD_TC + 2) % 9, c = i / (9 * (LD_TC + 2));
        const int yy = y + r % 3 - 1, xx = x0 + j - 1;
        Ls[i] = yy >= 0 && yy < h && xx >= 0 && xx < w ? L[(size_t)c * nl + (size_t)(q + r / 3) * pl + (size_t)yy * w + xx] : 0.f;
    }
}
template <int KG> __device__ __forceinline__ void ld_conv(const float *Ls, const float *__restrict__ B, int Hd, int Cl, int cell, int k0, float *acc) {
#pragma unroll
    for (int i = 0; i < KG; i++) acc[i] = 0.f;
    for (int c = 0; c < Cl; c++)
        for (int t = 0; t < 27; t++) {
            const float v = Ls[(c * 9 + t / 3) * (LD_TC + 2) + cell + t % 3];
            const float4 *b4 = (const float4 *)(B + (size_t)(c * 27 + t) * Hd + k0);
#pragma unroll
            for (int i = 0; i < KG / 4; i++) { const float4 b = __ldg(b4 + i); acc[4 * i] += b.x * v; acc[4 * i + 1] += b.y * v; acc[4 * i + 2] += b.z * v; acc[4 * i + 3] += b.w * v; }
        }
}
template <typename T> __device__ __forceinline__ float ld_ld(const T *p);
template <> __device__ __forceinline__ float ld_ld<f16>(const f16 *p) { return __half2float(*p); }
template <> __device__ __forceinline__ float ld_ld<float>(const float *p) { return *p; }
template <typename T> __device__ __forceinline__ void ld_st(T *p, float v);
template <> __device__ __forceinline__ void ld_st<f16>(f16 *p, float v) { *p = __float2half_rn(v); }
template <> __device__ __forceinline__ void ld_st<float>(float *p, float v) { *p = v; }

/* forward: rows[o * ld + n] += y_o(n) for the nq planes of h x w cells (n = (q h + y) w + x; rows / pre start at the first processed plane);
   pre [Hd][*] row stride ldp; L: nq + 2 planes from the plane before the first processed one */
template <int KG, typename TR, typename TP>
__global__ void __launch_bounds__(LD_NT) latdec_fwd_k(TR *rows, size_t ld, const TP *__restrict__ pre, size_t ldp, const float *__restrict__ L, int nq, int h, int w,
                                                    const float *__restrict__ B, const float *__restrict__ W2, int Cl, int no) {
    extern __shared__ float sm[];
    constexpr int Hd = 4 * KG;
    float *Ls = sm, *hs = sm + (size_t)Cl * 9 * (LD_TC + 2);
    const int q = blockIdx.z, y = blockIdx.y, x0 = blockIdx.x * LD_TC, cell = threadIdx.x & (LD_TC - 1), g = threadIdx.x / LD_TC, k0 = g * KG, x = x0 + cell;
    const bool ok = x < w; const size_t n = ((size_t)q * h + y) * w + x;
    ld_stage(L, nq, h, w, q, y, x0, Cl, Ls);
    __syncthreads();
    float acc[KG];
    ld_conv<KG>(Ls, B, Hd, Cl, cell, k0, acc);
#pragma unroll
    for (int i = 0; i < KG; i++) { const float p = acc[i] + (ok ? ld_ld<TP>(pre + (size_t)(k0 + i) * ldp + n) : 0.f); hs[cell * LD_HS + k0 + i] = p / (1.f + __expf(-p)); }
    __syncthreads();
    if (!ok) return;
    for (int o = g; o < no; o += 4) {
        const float4 *w4 = (const float4 *)(W2 + (size_t)o * Hd); float s = 0.f;
#pragma unroll
        for (int k = 0; k < Hd / 4; k++) { const float4 a = __ldg(w4 + k); const float *hh = hs + cell * LD_HS + 4 * k; s += a.x * hh[0] + a.y * hh[1] + a.z * hh[2] + a.w * hh[3]; }
        if (s != 0.f) { TR *r = rows + (size_t)o * ld + n; ld_st<TR>(r, ld_ld<TR>(r) + s); }
    }
}
/* backward per cell: h and dpre (fp32 scratch Hs / DP [Hd][N]) and dpre as fp16 rows gpre (stride ldg) from the sub-cell rows' gradient gy */
template <int KG>
__global__ void __launch_bounds__(LD_NT) latdec_bwd_k(const f16 *__restrict__ gy, size_t ldg, f16 *gpre, const f16 *__restrict__ pre, size_t ldp, const float *__restrict__ L,
                                                    int nq, int h, int w, const float *__restrict__ B, const float *__restrict__ W2, int Cl, int no,
                                                    float *__restrict__ Hs, float *__restrict__ DP, size_t N) {
    extern __shared__ float sm[];
    constexpr int Hd = 4 * KG;
    float *Ls = sm, *dys = sm + (size_t)Cl * 9 * (LD_TC + 2);
    const int q = blockIdx.z, y = blockIdx.y, x0 = blockIdx.x * LD_TC, cell = threadIdx.x & (LD_TC - 1), g = threadIdx.x / LD_TC, k0 = g * KG, x = x0 + cell;
    const bool ok = x < w; const size_t n = ((size_t)q * h + y) * w + x;
    ld_stage(L, nq, h, w, q, y, x0, Cl, Ls);
    for (int o = g; o < no; o += 4) dys[cell * (no + 1) + o] = ok ? __half2float(gy[(size_t)o * ldg + n]) : 0.f;
    __syncthreads();
    if (!ok) return;
    float acc[KG], dh[KG];
    ld_conv<KG>(Ls, B, Hd, Cl, cell, k0, acc);
#pragma unroll
    for (int i = 0; i < KG; i++) dh[i] = 0.f;
    for (int o = 0; o < no; o++) {
        const float d = dys[cell * (no + 1) + o]; const float4 *w4 = (const float4 *)(W2 + (size_t)o * Hd + k0);
#pragma unroll
        for (int i = 0; i < KG / 4; i++) { const float4 a = __ldg(w4 + i); dh[4 * i] += a.x * d; dh[4 * i + 1] += a.y * d; dh[4 * i + 2] += a.z * d; dh[4 * i + 3] += a.w * d; }
    }
#pragma unroll
    for (int i = 0; i < KG; i++) {
        const int k = k0 + i; const float p = acc[i] + __half2float(pre[(size_t)k * ldp + n]), sg = 1.f / (1.f + __expf(-p)), dp = dh[i] * sg * (1.f + p * (1.f - sg));
        Hs[(size_t)k * N + n] = p * sg; DP[(size_t)k * N + n] = dp; gpre[(size_t)k * ldg + n] = __float2half_rn(dp);
    }
}
/* the weight gradients as chunked outer-product sums: part[chunk][a][b] = sum over the chunk's cells n of P(a, n) Q[b][n] (Q fp32 [Hd][N]).
   MODE 0: P = gy rows (W2's gradient, a = o); MODE 1: P = the latent patch L[c](n + t) (B's gradient, a = 27 c + t) */
template <int MODE>
__global__ void __launch_bounds__(256) latdec_gw_k(const f16 *__restrict__ gy, size_t ldg, const float *__restrict__ L, int nq, int h, int w, int A, int Hd,
                                                 const float *__restrict__ Q, size_t N, float *__restrict__ part) {
    __shared__ float Ps[LD_GA][LD_GN + 1], Qs[64][LD_GN + 1];
    const int a0 = blockIdx.x * LD_GA, al = threadIdx.x >> 3, bg = threadIdx.x & 7, nb = Hd / 8;
    const size_t c0 = (size_t)blockIdx.y * LD_CH, c1 = c0 + LD_CH < N ? c0 + LD_CH : N, pl = (size_t)h * w, nl = (size_t)(nq + 2) * pl;
    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; j++) acc[j] = 0.f;
    for (size_t n0 = c0; n0 < c1; n0 += LD_GN) {
        for (int i = threadIdx.x; i < LD_GA * LD_GN; i += 256) {
            const int ai = i / LD_GN, ni = i % LD_GN, a = a0 + ai; const size_t n = n0 + ni; float v = 0.f;
            if (a < A && n < c1) {
                if (MODE == 0) v = __half2float(gy[(size_t)a * ldg + n]);
                else {
                    const int c = a / 27, t = a % 27, q = (int)(n / pl), yy = (int)(n / w % h) + t / 3 % 3 - 1, xx = (int)(n % w) + t % 3 - 1;
                    if (yy >= 0 && yy < h && xx >= 0 && xx < w) v = L[(size_t)c * nl + (size_t)(q + t / 9) * pl + (size_t)yy * w + xx];
                }
            }
            Ps[ai][ni] = v;
        }
        for (int i = threadIdx.x; i < Hd * LD_GN; i += 256) { const int b = i / LD_GN, ni = i % LD_GN; const size_t n = n0 + ni; Qs[b][ni] = n < c1 ? Q[(size_t)b * N + n] : 0.f; }
        __syncthreads();
        for (int ni = 0; ni < LD_GN; ni++) {
            const float pv = Ps[al][ni];
#pragma unroll
            for (int j = 0; j < 8; j++) if (j < nb) acc[j] += pv * Qs[bg + 8 * j][ni];
        }
        __syncthreads();
    }
    const int a = a0 + al;
    if (a < A) for (int j = 0; j < nb; j++) part[((size_t)blockIdx.y * A + a) * Hd + bg + 8 * j] = acc[j];
}
__global__ void latdec_red_k(const float *__restrict__ part, int nch, size_t npar, float *__restrict__ gp, const float *__restrict__ hp) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= npar) return;
    float s = 0.f; for (int c = 0; c < nch; c++) s += part[(size_t)c * npar + i];
    gp[i] += (hp ? hp[1] : 1.f) * s;   /* (hp: the head plane scale out, a power of two) */
}

static size_t ld_smem(int Cl, int no, int bwd) { return sizeof(float) * ((size_t)Cl * 9 * (LD_TC + 2) + (bwd ? (size_t)LD_TC * (no + 1) : (size_t)LD_TC * LD_HS)); }
static int ld_ok(int Hd, int Cl, int no) { return (Hd == 16 || Hd == 32 || Hd == 64) && Cl >= 1 && Cl <= 64 && no >= 1 && no <= 256 && ld_smem(Cl, no, 1) <= 99 * 1024 && ld_smem(Cl, no, 0) <= 99 * 1024; }
static void ld_attr(const void *k, size_t sm) { if (sm > 48 * 1024) cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024); }
extern "C" void nn_latdec_cells(const void *x16, int D, int H, int W, int q0, int nq, const float *bas, const float *gain, int Cl, float step, float *L) {
    const size_t nc = (size_t)nq * (H / 4) * (W / 4); if (!nc) return;
    latdec_cells_k<<<nblk(nc, 128), 128>>>((const f16 *)x16, D, H, W, q0, nq, bas, gain, Cl, step, L); KCHECK();
}
#define LD_FWD(KG, TR, TP) do { ld_attr((const void *)latdec_fwd_k<KG, TR, TP>, sm); latdec_fwd_k<KG, TR, TP><<<grid, LD_NT, sm>>>((TR *)rows, ld, (const TP *)pre, ldp, L, nq, h, w, p, p + (size_t)Cl * 27 * Hd, Cl, no); } while (0)
extern "C" int nn_latdec_fwd(void *rows, int r32, size_t ld, const void *pre, int p32, size_t ldp, const float *L, int nq, int h, int w, const float *p, int Hd, int Cl, int ng) {
    const int no = 64 * ng;
    if (!ld_ok(Hd, Cl, no)) return 1;
    if (!nq) return 0;
    const dim3 grid((unsigned)((w + LD_TC - 1) / LD_TC), (unsigned)h, (unsigned)nq); const size_t sm = ld_smem(Cl, no, 0);
#define LD_FWD_T(KG) do { if (r32 && p32) LD_FWD(KG, float, float); else if (r32) LD_FWD(KG, float, f16); else if (p32) LD_FWD(KG, f16, float); else LD_FWD(KG, f16, f16); } while (0)
    if (Hd == 16) LD_FWD_T(4); else if (Hd == 32) LD_FWD_T(8); else LD_FWD_T(16);
#undef LD_FWD_T
    KCHECK();
    return 0;
}
#undef LD_FWD
extern "C" size_t nn_latdec_scratch(size_t N, int Hd, int Cl, int ng) {
    const size_t nch = (N + LD_CH - 1) / LD_CH, A = (size_t)(Cl * 27 > 64 * ng ? Cl * 27 : 64 * ng);
    return (2 * (size_t)Hd * N + nch * A * Hd) * sizeof(float) + 256;
}
extern "C" int nn_latdec_bwd(const void *gy16, size_t ldg, void *gpre16, const void *pre16, size_t ldp, const float *L, int nq, int h, int w, const float *p,
                             int Hd, int Cl, int ng, float *gp, void *scratch) {
    const int no = 64 * ng;
    if (!ld_ok(Hd, Cl, no)) return 1;
    const size_t N = (size_t)nq * h * w; if (!N) return 0;
    float *Hs = (float *)scratch, *DP = Hs + (size_t)Hd * N, *part = DP + (size_t)Hd * N;
    const float *B = p, *W2 = p + (size_t)Cl * 27 * Hd; float *gB = gp, *gW2 = gp + (size_t)Cl * 27 * Hd;
    const dim3 grid((unsigned)((w + LD_TC - 1) / LD_TC), (unsigned)h, (unsigned)nq); const size_t sm = ld_smem(Cl, no, 1);
#define LD_BWD(KG) do { ld_attr((const void *)latdec_bwd_k<KG>, sm); latdec_bwd_k<KG><<<grid, LD_NT, sm>>>((const f16 *)gy16, ldg, (f16 *)gpre16, (const f16 *)pre16, ldp, L, nq, h, w, B, W2, Cl, no, Hs, DP, N); } while (0)
    if (Hd == 16) LD_BWD(4); else if (Hd == 32) LD_BWD(8); else LD_BWD(16);
#undef LD_BWD
    KCHECK();
    const int nch = (int)((N + LD_CH - 1) / LD_CH);
    latdec_gw_k<0><<<dim3((unsigned)((no + LD_GA - 1) / LD_GA), (unsigned)nch), 256>>>((const f16 *)gy16, ldg, L, nq, h, w, no, Hd, Hs, N, part); KCHECK();
    latdec_red_k<<<nblk((size_t)no * Hd, 256), 256>>>(part, nch, (size_t)no * Hd, gW2, GSHP); KCHECK();
    latdec_gw_k<1><<<dim3((unsigned)((Cl * 27 + LD_GA - 1) / LD_GA), (unsigned)nch), 256>>>((const f16 *)gy16, ldg, L, nq, h, w, Cl * 27, Hd, DP, N, part); KCHECK();
    latdec_red_k<<<nblk((size_t)Cl * 27 * Hd, 256), 256>>>(part, nch, (size_t)Cl * 27 * Hd, gB, GSHP); KCHECK();
    return 0;
}
