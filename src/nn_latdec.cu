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

enum { LD_TC = 32, LD_NT = 128, LD_HS = 65, LD_GA = 32, LD_GN = 32, LD_CH = 4096, LD_OC = 128, LD_T2 = 64 };   /* LD_T2: the tensor-core kernels' cells a block (a warp 16, all H) */

/* the cells' latent input: L [Cl][nq][h][w] fp32 for cell planes [q0, q0 + nq) of a D x H x W fp16 window (planes outside: 0) */
__global__ void latdec_cells_k(const f16 *__restrict__ x, int D, int H, int W, int q0, int nq, const float *__restrict__ bas, const float *__restrict__ gain,
                               int Cl, float step, __half *__restrict__ L) {
    const int h = H / 4, w = W / 4; const size_t pl = (size_t)h * w, nc = (size_t)nq * pl, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nc) return;
    const int q = q0 + (int)(i / pl), y = (int)(i / w % h), xx = (int)(i % w);
    if (q < 0 || q >= D / 4) { for (int c = 0; c < Cl; c++) L[(size_t)c * nc + i] = __float2half(0.f); return; }
    float cv[64];
    const size_t hw = (size_t)H * W;
#pragma unroll
    for (int k = 0; k < 64; k++) cv[k] = __half2float(x[(size_t)(4 * q + (k >> 4)) * hw + (size_t)(4 * y + (k >> 2 & 3)) * W + 4 * xx + (k & 3)]);
    if (!bas) { for (int c = 0; c < Cl; c++) L[(size_t)c * nc + i] = __float2half_rn(cv[c]); return; }
    for (int c = 0; c < Cl; c++) {   /* input_latc_k's sums and quantiser */
        float s = 0.f; for (int k = 0; k < 64; k++) s += bas[c * 64 + k] * cv[k];
        if (step > 0.f) { const float lim = c ? 127.f : 32767.f; s = fminf(lim, fmaxf(-lim, rintf(s / step))) * step; }
        L[(size_t)c * nc + i] = __float2half_rn(s * gain[c]);
    }
}

/* the tile: LD_TC cells of one (q, y) row, 4 thread groups of KG hidden units each. Ls [Cl][9][LD_TC + 2]: the 3 x 3 rows of neighbour cells */
__device__ __forceinline__ void ld_stage(const __half *__restrict__ L, int nq, int h, int w, int q, int y, int x0, int Cl, float *Ls) {
    const size_t pl = (size_t)h * w, nl = (size_t)(nq + 2) * pl; const int n = Cl * 9 * (LD_TC + 2);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int j = i % (LD_TC + 2), r = i / (LD_TC + 2) % 9, c = i / (9 * (LD_TC + 2));
        const int yy = y + r % 3 - 1, xx = x0 + j - 1;
        Ls[i] = yy >= 0 && yy < h && xx >= 0 && xx < w ? __half2float(L[(size_t)c * nl + (size_t)(q + r / 3) * pl + (size_t)yy * w + xx]) : 0.f;
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
__global__ void __launch_bounds__(LD_NT) latdec_fwd_k(TR *rows, size_t ld, const TP *__restrict__ pre, size_t ldp, const __half *__restrict__ L, int nq, int h, int w,
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
__global__ void __launch_bounds__(LD_NT) latdec_bwd_k(const f16 *__restrict__ gy, size_t ldg, f16 *gpre, const f16 *__restrict__ pre, size_t ldp, const __half *__restrict__ L,
                                                    int nq, int h, int w, const float *__restrict__ B, const float *__restrict__ W2, int Cl, int no,
                                                    float *__restrict__ Hs, float *__restrict__ DP, size_t N) {
    extern __shared__ float sm[];
    constexpr int Hd = 4 * KG;
    float *Ls = sm, *dys = sm + (size_t)Cl * 9 * (LD_TC + 2);
    const int q = blockIdx.z, y = blockIdx.y, x0 = blockIdx.x * LD_TC, cell = threadIdx.x & (LD_TC - 1), g = threadIdx.x / LD_TC, k0 = g * KG, x = x0 + cell;
    const bool ok = x < w; const size_t n = ((size_t)q * h + y) * w + x;
    ld_stage(L, nq, h, w, q, y, x0, Cl, Ls);
    __syncthreads();
    float acc[KG], dh[KG];
    ld_conv<KG>(Ls, B, Hd, Cl, cell, k0, acc);
#pragma unroll
    for (int i = 0; i < KG; i++) dh[i] = 0.f;
    for (int oc = 0; oc < no; oc += LD_OC) {   /* the rows' gradient in chunks of LD_OC outputs through shared memory (--lat-sub: up to 1024) */
        const int nc = no - oc < LD_OC ? no - oc : LD_OC;
        __syncthreads();
        for (int o = g; o < nc; o += 4) dys[cell * (LD_OC + 1) + o] = ok ? __half2float(gy[(size_t)(oc + o) * ldg + n]) : 0.f;
        __syncthreads();
        for (int o = 0; o < nc; o++) {
            const float d = dys[cell * (LD_OC + 1) + o]; const float4 *w4 = (const float4 *)(W2 + (size_t)(oc + o) * Hd + k0);
#pragma unroll
            for (int i = 0; i < KG / 4; i++) { const float4 a = __ldg(w4 + i); dh[4 * i] += a.x * d; dh[4 * i + 1] += a.y * d; dh[4 * i + 2] += a.z * d; dh[4 * i + 3] += a.w * d; }
        }
    }
    if (!ok) return;
#pragma unroll
    for (int i = 0; i < KG; i++) {
        const int k = k0 + i; const float p = acc[i] + __half2float(pre[(size_t)k * ldp + n]), sg = 1.f / (1.f + __expf(-p)), dp = dh[i] * sg * (1.f + p * (1.f - sg));
        Hs[(size_t)k * N + n] = p * sg; DP[(size_t)k * N + n] = dp; gpre[(size_t)k * ldg + n] = __float2half_rn(dp);
    }
}
/* the weight gradients as chunked outer-product sums: part[chunk][a][b] = sum over the chunk's cells n of P(a, n) Q[b][n] (Q fp32 [Hd][N]).
   MODE 0: P = gy rows (W2's gradient, a = o); MODE 1: P = the latent patch L[c](n + t) (B's gradient, a = 27 c + t) */
template <int MODE>
__global__ void __launch_bounds__(256) latdec_gw_k(const f16 *__restrict__ gy, size_t ldg, const __half *__restrict__ L, int nq, int h, int w, int A, int Hd,
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
                    if (yy >= 0 && yy < h && xx >= 0 && xx < w) v = __half2float(L[(size_t)c * nl + (size_t)(q + t / 9) * pl + (size_t)yy * w + xx]);
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


/* ---- tensor-core path (default; UFSM_LATDEC_FFMA=1: the FFMA kernels above): mma m16n8k16 fp16 x fp16 -> fp32. The weights go to fp16 copies
   per call (Bt [H][KP]: B transposed, k = K3 r + CP dx + c (ld_conv_tc), KP = 27 CP; W2h [no][H]; W2t [H][no]); L to fp16 as staged; h and the gradients'
   products in fp16, every sum in fp32. bench_latdec P 768 ng 7 H 64: see DESIGN. */
__device__ __forceinline__ void ld_mma(float *c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned ld_pk(__half a, __half b) { __half2 v = __halves2half2(a, b); return *(unsigned *)&v; }
__device__ __forceinline__ unsigned ld_pkf(float a, float b) { __half2 v = __floats2half2_rn(a, b); return *(unsigned *)&v; }
__global__ void latdec_pack_k(const float *__restrict__ p, int Cl, int Hd, int no, int KP, __half *Bt, __half *W2h, __half *W2t) {
    const size_t nb = (size_t)Hd * KP, nw = (size_t)no * Hd, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const float *B = p, *W2 = p + (size_t)Cl * 27 * Hd;
    if (i < nb) { const int n = (int)(i / KP), k = (int)(i % KP), K3 = KP / 9, CP = K3 / 3, r = k / K3, dx = k % K3 / CP, c = k % CP;   /* (k = K3 r + CP dx + c: B's 27 c + 3 r + dx) */
                  Bt[i] = __float2half_rn(c < Cl ? B[((size_t)c * 27 + 3 * r + dx) * Hd + n] : 0.f); }
    else if (i < nb + nw) { const size_t j = i - nb; const int o = (int)(j / Hd), k = (int)(j % Hd); const __half v = __float2half_rn(W2[j]); W2h[j] = v; W2t[(size_t)k * no + o] = v; }
}
/* conv(L) for the warp's 16 cells (rows r0 = 16 warp + g, + 8) x all HD hidden into acc[HD / 8][4], the block's LD_T2 cells of row (q, y) from x0.
   K is ordered by neighbour row, then dx, then channel: k = K3 r + CP dx + c (r = 3 dz + dy; CP = 32 or 64 >= Cl, padded with zeros; K3 = 3 CP;
   Bt packed so). The block stages one row of L at a time transposed, Lt[x][c] fp16 (stride CP + 8, double-buffered: one barrier a row),
   so an A fragment's pairs (c, c + 1) are aligned 4-byte loads at Lt[r + dx][c], each built once and used HD / 8 times. Every thread of the block calls it */
template <int HD> __device__ __forceinline__ void ld_conv_tc(const __half *__restrict__ L, int nq, int h, int w, int q, int y, int x0, int Cl, int CP, __half *Lr,
                                                         const __half *__restrict__ Bt, int r0, float (*acc)[4]) {
    constexpr int NT = HD / 8, RW = LD_T2 + 2; const int lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3, CS = CP + 8, K3 = 3 * CP, KP = 9 * K3;
    const size_t pl = (size_t)h * w, nl = (size_t)(nq + 2) * pl;
#pragma unroll
    for (int j = 0; j < NT; j++) for (int e = 0; e < 4; e++) acc[j][e] = 0.f;
    for (int r = 0; r < 9; r++) {
        __half *Lt = Lr + (size_t)(r & 1) * RW * CS; const int yy = y + r % 3 - 1, qq = q + r / 3;
        for (int i0 = threadIdx.x; i0 < CP * RW; i0 += 8 * LD_NT) {   /* (x fastest: coalesced reads of L's rows; 8 loads in flight a thread, then the stores) */
            __half v[8];
#pragma unroll
            for (int u = 0; u < 8; u++) { const int i = i0 + u * LD_NT, c = i / RW, xx = x0 + i % RW - 1;
                v[u] = i < CP * RW && c < Cl && yy >= 0 && yy < h && xx >= 0 && xx < w ? L[(size_t)c * nl + (size_t)qq * pl + (size_t)yy * w + xx] : __float2half(0.f); }
#pragma unroll
            for (int u = 0; u < 8; u++) { const int i = i0 + u * LD_NT; if (i < CP * RW) Lt[(i % RW) * CS + i / RW] = v[u]; }
        }
        __syncthreads();
        unsigned bf[NT][2];   /* (B fragments one k-step ahead: the loads' latency under the previous step's MMAs) */
        const __half *bb = Bt + (size_t)g * KP + r * K3 + 2 * t4;
#pragma unroll
        for (int j = 0; j < NT; j++) { bf[j][0] = __ldg((const unsigned *)(bb + (size_t)8 * j * KP)); bf[j][1] = __ldg((const unsigned *)(bb + (size_t)8 * j * KP + 8)); }
        for (int k0 = 0; k0 < K3; k0 += 16) {
            const int dx = k0 / CP, c = k0 % CP + 2 * t4;   /* (16 k never straddle a dx block: CP a multiple of 16) */
            const __half *l0 = Lt + (r0 + dx) * CS + c, *l1 = l0 + 8 * CS;
            const unsigned a0 = *(const unsigned *)l0, a1 = *(const unsigned *)l1, a2 = *(const unsigned *)(l0 + 8), a3 = *(const unsigned *)(l1 + 8);
            const int kn = k0 + 16 < K3 ? k0 + 16 : k0;
            unsigned bn[NT][2];
#pragma unroll
            for (int j = 0; j < NT; j++) { bn[j][0] = __ldg((const unsigned *)(bb + (size_t)8 * j * KP + kn)); bn[j][1] = __ldg((const unsigned *)(bb + (size_t)8 * j * KP + kn + 8)); }
#pragma unroll
            for (int j = 0; j < NT; j++) { ld_mma(acc[j], a0, a1, a2, a3, bf[j][0], bf[j][1]); bf[j][0] = bn[j][0]; bf[j][1] = bn[j][1]; }
        }
    }
}
template <int HD, typename TR, typename TP>
__global__ void __launch_bounds__(LD_NT) latdec_fwd_tc_k(TR *rows, size_t ld, const TP *__restrict__ pre, size_t ldp, const __half *__restrict__ L, int nq, int h, int w,
                                                       const __half *__restrict__ Bt, int KP, const __half *__restrict__ W2h, int Cl, int no) {
    extern __shared__ __align__(16) unsigned char smb[];
    constexpr int NT = HD / 8, HS = HD + 8, KS = HD / 16;
    __half *hs = (__half *)smb, *Lr = hs + LD_T2 * HS;
    const int q = blockIdx.z, y = blockIdx.y, x0 = blockIdx.x * LD_T2, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3, r0 = 16 * warp + g;
    const size_t nrow = ((size_t)q * h + y) * w;
    float acc[NT][4];
    ld_conv_tc<HD>(L, nq, h, w, q, y, x0, Cl, KP / 27, Lr, Bt, r0, acc);
    TP pv[NT][4];   /* (the pre rows' raw values: every load issued before the first conversion) */
    const bool ok0 = x0 + r0 < w, ok1 = x0 + r0 + 8 < w;
#pragma unroll
    for (int j = 0; j < NT; j++)
#pragma unroll
        for (int e = 0; e < 4; e++) { const int col = 8 * j + 2 * t4 + (e & 1); pv[j][e] = (e < 2 ? ok0 : ok1) ? pre[(size_t)col * ldp + nrow + x0 + r0 + 8 * (e >> 1)] : pre[(size_t)col * ldp + nrow + x0]; }
#pragma unroll
    for (int j = 0; j < NT; j++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int r = r0 + 8 * (e >> 1), col = 8 * j + 2 * t4 + (e & 1);
            const float p = acc[j][e] + ld_ld<TP>(&pv[j][e]);
            hs[r * HS + col] = __float2half_rn(p / (1.f + __expf(-p)));
        }
    __syncwarp();   /* (each warp reads back only its own 16 rows) */
    unsigned a[KS][4];
#pragma unroll
    for (int ks = 0; ks < KS; ks++) { const int ka = 16 * ks + 2 * t4;
        a[ks][0] = *(const unsigned *)&hs[r0 * HS + ka]; a[ks][1] = *(const unsigned *)&hs[(r0 + 8) * HS + ka]; a[ks][2] = *(const unsigned *)&hs[r0 * HS + ka + 8]; a[ks][3] = *(const unsigned *)&hs[(r0 + 8) * HS + ka + 8]; }
    unsigned wb[KS][2];
    { const __half *bp = W2h + (size_t)g * HD + 2 * t4;
#pragma unroll
      for (int ks = 0; ks < KS; ks++) { wb[ks][0] = __ldg((const unsigned *)(bp + 16 * ks)); wb[ks][1] = __ldg((const unsigned *)(bp + 16 * ks + 8)); } }
    const size_t rq[4] = {nrow + x0 + (ok0 ? r0 : 0), nrow + x0 + (ok0 ? r0 : 0), nrow + x0 + (ok1 ? r0 + 8 : 0), nrow + x0 + (ok1 ? r0 + 8 : 0)};   /* (clamped: always a valid address) */
    TR rv[4];   /* (the rows' old values one output tile ahead: their loads under the MMAs) */
#pragma unroll
    for (int e = 0; e < 4; e++) rv[e] = rows[(size_t)(2 * t4 + (e & 1)) * ld + rq[e]];
    for (int jt = 0; jt < no / 8; jt++) {
        float c[4] = {0.f, 0.f, 0.f, 0.f}; const int jn = jt + 1 < no / 8 ? jt + 1 : jt; const __half *bq = W2h + (size_t)(8 * jn + g) * HD + 2 * t4;
        TR rn[4];
#pragma unroll
        for (int e = 0; e < 4; e++) rn[e] = rows[(size_t)(8 * jn + 2 * t4 + (e & 1)) * ld + rq[e]];
        unsigned wn[KS][2];
#pragma unroll
        for (int ks = 0; ks < KS; ks++) { wn[ks][0] = __ldg((const unsigned *)(bq + 16 * ks)); wn[ks][1] = __ldg((const unsigned *)(bq + 16 * ks + 8)); }
#pragma unroll
        for (int ks = 0; ks < KS; ks++) { ld_mma(c, a[ks][0], a[ks][1], a[ks][2], a[ks][3], wb[ks][0], wb[ks][1]); wb[ks][0] = wn[ks][0]; wb[ks][1] = wn[ks][1]; }
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int r = r0 + 8 * (e >> 1), o = 8 * jt + 2 * t4 + (e & 1);
            if ((e < 2 ? ok0 : ok1) && c[e] != 0.f) ld_st<TR>(rows + (size_t)o * ld + nrow + x0 + r, ld_ld<TR>(&rv[e]) + c[e]);
            rv[e] = rn[e];
        }
    }
}
template <int HD>
__global__ void __launch_bounds__(LD_NT) latdec_bwd_tc_k(const f16 *__restrict__ gy, size_t ldg, f16 *gpre, const f16 *__restrict__ pre, size_t ldp, const __half *__restrict__ L,
                                                       int nq, int h, int w, const __half *__restrict__ Bt, int KP, const __half *__restrict__ W2t, int Cl, int no, __half *Hs, size_t N) {
    extern __shared__ __align__(16) unsigned char smb[];
    constexpr int NT = HD / 8, DS = LD_OC + 8;
    __half *dys = (__half *)smb, *Lr = dys + LD_T2 * DS;
    const int q = blockIdx.z, y = blockIdx.y, x0 = blockIdx.x * LD_T2, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3, r0 = 16 * warp + g;
    const size_t nrow = ((size_t)q * h + y) * w;
    float acc[NT][4], da[NT][4];
    ld_conv_tc<HD>(L, nq, h, w, q, y, x0, Cl, KP / 27, Lr, Bt, r0, acc);
#pragma unroll
    for (int j = 0; j < NT; j++) for (int e = 0; e < 4; e++) da[j][e] = 0.f;
    for (int oc = 0; oc < no; oc += LD_OC) {   /* dh = dy W2: the rows' gradient through shared memory, LD_OC outputs a chunk (each warp its own 16 rows) */
        const int nc = no - oc < LD_OC ? no - oc : LD_OC;
        __syncwarp();
        for (int i0 = lane; i0 < 16 * nc; i0 += 8 * 32) {   /* (8 loads in flight a lane) */
            __half v[8];
#pragma unroll
            for (int u = 0; u < 8; u++) { const int i = i0 + 32 * u, r = 16 * warp + i % 16, o = i / 16; v[u] = i < 16 * nc && x0 + r < w ? gy[(size_t)(oc + o) * ldg + nrow + x0 + r] : __float2half(0.f); }
#pragma unroll
            for (int u = 0; u < 8; u++) { const int i = i0 + 32 * u; if (i < 16 * nc) dys[(16 * warp + i % 16) * DS + i / 16] = v[u]; }
        }
        __syncwarp();
        for (int k0 = 0; k0 < nc; k0 += 16) {
            const int ka = k0 + 2 * t4;
            const unsigned a0 = *(const unsigned *)&dys[r0 * DS + ka], a1 = *(const unsigned *)&dys[(r0 + 8) * DS + ka], a2 = *(const unsigned *)&dys[r0 * DS + ka + 8], a3 = *(const unsigned *)&dys[(r0 + 8) * DS + ka + 8];
#pragma unroll
            for (int j = 0; j < NT; j++) {
                const __half *bp = W2t + (size_t)(8 * j + g) * no + oc;
                ld_mma(da[j], a0, a1, a2, a3, __ldg((const unsigned *)(bp + ka)), __ldg((const unsigned *)(bp + ka + 8)));
            }
        }
    }
    __half pv[NT][4];   /* (raw: every load before the first conversion) */
    const size_t pq[2] = {nrow + x0 + (x0 + r0 < w ? r0 : 0), nrow + x0 + (x0 + r0 + 8 < w ? r0 + 8 : 0)};
#pragma unroll
    for (int j = 0; j < NT; j++)
#pragma unroll
        for (int e = 0; e < 4; e++) pv[j][e] = pre[(size_t)(8 * j + 2 * t4 + (e & 1)) * ldp + pq[e >> 1]];
#pragma unroll
    for (int j = 0; j < NT; j++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int r = r0 + 8 * (e >> 1), col = 8 * j + 2 * t4 + (e & 1), x = x0 + r;
            if (x >= w) continue;
            const size_t n = nrow + x;
            const float p = acc[j][e] + __half2float(pv[j][e]), sg = 1.f / (1.f + __expf(-p));
            Hs[(size_t)col * N + n] = __float2half_rn(p * sg); gpre[(size_t)col * ldg + n] = __float2half_rn(da[j][e] * sg * (1.f + p * (1.f - sg)));
        }
}
/* part[chunk][a][b] = sum over the chunk's cells of P(a, n) Q[b][n] (Q fp16 [HD][*] row stride ldq), 64 a rows a block (16 a warp), all HD b.
   MODE 0: P = gy rows (stride ldg; W2's gradient); MODE 1: P = the latent patch L[c](n + t), a = 27 c + t (B's gradient) */
template <int HD, int MODE>
__global__ void __launch_bounds__(LD_NT) latdec_gw_tc_k(const f16 *__restrict__ gy, size_t ldg, const __half *__restrict__ L, int nq, int h, int w, int A,
                                                      const __half *__restrict__ Q, size_t ldq, size_t N, float *__restrict__ part) {
    constexpr int NT = HD / 8;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3, ar0 = blockIdx.x * 64 + 16 * warp + g, ar1 = ar0 + 8;
    const size_t c0 = (size_t)blockIdx.y * LD_CH, c1 = c0 + LD_CH < N ? c0 + LD_CH : N, pl = (size_t)h * w, nl = (size_t)(nq + 2) * pl;
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; j++) for (int e = 0; e < 4; e++) acc[j][e] = 0.f;
    int cA[2] = {0, 0}, dz[2] = {0, 0}, dy_[2] = {0, 0}, dx[2] = {0, 0}; bool va[2] = {ar0 < A, ar1 < A};
    if (MODE == 1) for (int i = 0; i < 2; i++) { const int a = i ? ar1 : ar0, t = a % 27; cA[i] = a / 27; dz[i] = t / 9; dy_[i] = t / 3 % 3 - 1; dx[i] = t % 3 - 1; }
    for (size_t n0 = c0; n0 < c1; n0 += 16) {
        const size_t na = n0 + 2 * t4, nb = na + 8;
        unsigned a[4];
        if (MODE == 0) {
            a[0] = va[0] && na < c1 ? *(const unsigned *)(gy + (size_t)ar0 * ldg + na) : 0u; a[1] = va[1] && na < c1 ? *(const unsigned *)(gy + (size_t)ar1 * ldg + na) : 0u;
            a[2] = va[0] && nb < c1 ? *(const unsigned *)(gy + (size_t)ar0 * ldg + nb) : 0u; a[3] = va[1] && nb < c1 ? *(const unsigned *)(gy + (size_t)ar1 * ldg + nb) : 0u;
        } else {
            float v[2][4];
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const size_t n = (u < 2 ? na : nb) + (u & 1);
                const int qq = (int)(n / pl), yy = (int)(n / w % h), xx = (int)(n % w);
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    const int y2 = yy + dy_[i], x2 = xx + dx[i];
                    v[i][u] = va[i] && n < c1 && y2 >= 0 && y2 < h && x2 >= 0 && x2 < w ? __half2float(L[(size_t)cA[i] * nl + (size_t)(qq + dz[i]) * pl + (size_t)y2 * w + x2]) : 0.f;
                }
            }
            a[0] = ld_pkf(v[0][0], v[0][1]); a[1] = ld_pkf(v[1][0], v[1][1]); a[2] = ld_pkf(v[0][2], v[0][3]); a[3] = ld_pkf(v[1][2], v[1][3]);
        }
#pragma unroll
        for (int j = 0; j < NT; j++) {
            const __half *bp = Q + (size_t)(8 * j + g) * ldq;
            const unsigned b0 = na < c1 ? *(const unsigned *)(bp + na) : 0u, b1 = nb < c1 ? *(const unsigned *)(bp + nb) : 0u;
            ld_mma(acc[j], a[0], a[1], a[2], a[3], b0, b1);
        }
    }
#pragma unroll
    for (int j = 0; j < NT; j++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int a = e < 2 ? ar0 : ar1, b = 8 * j + 2 * t4 + (e & 1);
            if (a < A) part[((size_t)blockIdx.y * A + a) * HD + b] = acc[j][e];
        }
}
static int ld_ffma(void) { static int v = -1; if (v < 0) v = getenv("UFSM_LATDEC_FFMA") && atoi(getenv("UFSM_LATDEC_FFMA")) != 0; return v; }   /* mgpu: env */
/* the MX-fp4 weight variant (nn_latdec_wq): every block of 32 consecutive K of a packed GEMM operand (Bt rows along k; W2h rows along H, 16
   for H 16) rounded onto e2m1 x 2^e, e the smallest power-of-two scale with amax 2^-e <= 6 (the trunk's ue8m0 rule, lp_common mx_exp), round to
   nearest even on the e2m1 grid {0, .5, 1, 1.5, 2, 3, 4, 6}; from the fp16 packed values (tests/test_latdec.c ld_wq_ref is the host copy).
   W2t is the transpose of the quantised W2h. The fp32 master weights and their gradients are untouched (straight through) */
__device__ __forceinline__ float ld_e2m1(float x) {
    const float a = fabsf(x);
    const float q = a <= 0.25f ? 0.f : a < 0.75f ? 0.5f : a <= 1.25f ? 1.f : a < 1.75f ? 1.5f : a <= 2.5f ? 2.f : a < 3.5f ? 3.f : a <= 5.f ? 4.f : 6.f;
    return copysignf(q, x);
}
__global__ void latdec_wq_k(__half *w, size_t nblk_, int bs) {
    const size_t b = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (b >= nblk_) return;
    __half *v = w + b * bs; float am = 0.f;
    for (int i = 0; i < bs; i++) am = fmaxf(am, fabsf(__half2float(v[i])));
    const unsigned u = __float_as_uint(am * (1.f / 6.f)); int e = (int)((u >> 23) & 0xff) - 127 + ((u & 0x7fffff) != 0); e = max(-24, min(15, e));
    const float sc = ldexpf(1.f, e), is = ldexpf(1.f, -e);
    for (int i = 0; i < bs; i++) v[i] = __float2half_rn(ld_e2m1(__half2float(v[i]) * is) * sc);
}
__global__ void latdec_wt_k(const __half *W2h, __half *W2t, int Hd, int no) {
    const size_t j = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (j >= (size_t)no * Hd) return;
    W2t[(j % Hd) * no + j / Hd] = W2h[j];
}
static thread_local int g_ld_wq;   /* mgpu: per-thread (each GPU thread sets it before its calls) */
extern "C" void nn_latdec_wq(int fp4) { g_ld_wq = fp4; }
static __half *ld_pack(int Hd, int Cl, int no, const float *p, int *KPo) {   /* per device, grown: the fp16 weight copies of this call */
    static thread_local __half *buf[8]; static thread_local size_t cap[8]; int dv = 0; cudaGetDevice(&dv); dv &= 7;
    const int KP = 27 * (Cl <= 32 ? 32 : 64); const size_t n = (size_t)Hd * KP + 2 * (size_t)no * Hd;
    if (cap[dv] < n) { if (buf[dv]) cudaFree(buf[dv]); CK(cudaMalloc(&buf[dv], n * 2)); cap[dv] = n; }
    latdec_pack_k<<<nblk(n - (size_t)no * Hd, 256), 256>>>(p, Cl, Hd, no, KP, buf[dv], buf[dv] + (size_t)Hd * KP, buf[dv] + (size_t)Hd * KP + (size_t)no * Hd); KCHECK();
    if (g_ld_wq) {
        __half *Bt = buf[dv], *W2h = Bt + (size_t)Hd * KP; const int bs = Hd < 32 ? Hd : 32;
        latdec_wq_k<<<nblk((size_t)Hd * KP / 32, 128), 128>>>(Bt, (size_t)Hd * KP / 32, 32);
        latdec_wq_k<<<nblk((size_t)no * Hd / bs, 128), 128>>>(W2h, (size_t)no * Hd / bs, bs);
        latdec_wt_k<<<nblk((size_t)no * Hd, 256), 256>>>(W2h, W2h + (size_t)no * Hd, Hd, no); KCHECK();
    }
    *KPo = KP; return buf[dv];
}
static size_t ld_smem_tc(int Hd, int Cl, int KP, int bwd) { (void)Cl; return (size_t)LD_T2 * (bwd ? LD_OC + 8 : Hd + 8) * 2 + (size_t)2 * (LD_T2 + 2) * (KP / 27 + 8) * 2 + 16; }
static size_t ld_smem(int Cl, int no, int bwd) { return sizeof(float) * ((size_t)Cl * 9 * (LD_TC + 2) + (bwd ? (size_t)LD_TC * (LD_OC + 1) : (size_t)LD_TC * LD_HS)); }
static int ld_ok(int Hd, int Cl, int no) { return (Hd == 16 || Hd == 32 || Hd == 64) && Cl >= 1 && Cl <= 64 && no >= 1 && no <= 1024 && ld_smem(Cl, no, 1) <= 99 * 1024 && ld_smem(Cl, no, 0) <= 99 * 1024; }
static void ld_attr(const void *k, size_t sm) { if (sm > 48 * 1024) cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024); }
extern "C" void nn_latdec_cells(const void *x16, int D, int H, int W, int q0, int nq, const float *bas, const float *gain, int Cl, float step, void *L) {
    const size_t nc = (size_t)nq * (H / 4) * (W / 4); if (!nc) return;
    latdec_cells_k<<<nblk(nc, 128), 128>>>((const f16 *)x16, D, H, W, q0, nq, bas, gain, Cl, step, (__half *)L); KCHECK();
}
#define LD_FWD(KG, TR, TP) do { ld_attr((const void *)latdec_fwd_k<KG, TR, TP>, sm); latdec_fwd_k<KG, TR, TP><<<grid, LD_NT, sm>>>((TR *)rows, ld, (const TP *)pre, ldp, L, nq, h, w, p, p + (size_t)Cl * 27 * Hd, Cl, no); } while (0)
extern "C" int nn_latdec_fwd(void *rows, int r32, size_t ld, const void *pre, int p32, size_t ldp, const void *Lv, int nq, int h, int w, const float *p, int Hd, int Cl, int ng) {
    const int no = 64 * ng; const __half *L = (const __half *)Lv;
    if (!ld_ok(Hd, Cl, no) || (g_ld_wq && ld_ffma())) return 1;
    if (!nq) return 0;
    const dim3 grid((unsigned)((w + LD_TC - 1) / LD_TC), (unsigned)h, (unsigned)nq), grid2((unsigned)((w + LD_T2 - 1) / LD_T2), (unsigned)h, (unsigned)nq);
    if (!ld_ffma()) {
        int KP; const __half *pk = ld_pack(Hd, Cl, no, p, &KP); const __half *Bt = pk, *W2h = pk + (size_t)Hd * KP; const size_t sm = ld_smem_tc(Hd, Cl, KP, 0);
#define LDT(HD, TR, TP) do { ld_attr((const void *)latdec_fwd_tc_k<HD, TR, TP>, sm); latdec_fwd_tc_k<HD, TR, TP><<<grid2, LD_NT, sm>>>((TR *)rows, ld, (const TP *)pre, ldp, L, nq, h, w, Bt, KP, W2h, Cl, no); } while (0)
#define LDT_T(HD) do { if (r32 && p32) LDT(HD, float, float); else if (r32) LDT(HD, float, f16); else if (p32) LDT(HD, f16, float); else LDT(HD, f16, f16); } while (0)
        if (Hd == 16) LDT_T(16); else if (Hd == 32) LDT_T(32); else LDT_T(64);
#undef LDT_T
#undef LDT
        KCHECK(); return 0;
    }
    const size_t sm = ld_smem(Cl, no, 0);
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
extern "C" int nn_latdec_bwd(const void *gy16, size_t ldg, void *gpre16, const void *pre16, size_t ldp, const void *Lv, int nq, int h, int w, const float *p,
                             int Hd, int Cl, int ng, float *gp, void *scratch) {
    const int no = 64 * ng; const __half *L = (const __half *)Lv;
    if (!ld_ok(Hd, Cl, no)) return 1;
    const size_t N = (size_t)nq * h * w; if (!N) return 0;
    float *Hs = (float *)scratch, *DP = Hs + (size_t)Hd * N, *part = DP + (size_t)Hd * N;
    const float *B = p, *W2 = p + (size_t)Cl * 27 * Hd; float *gB = gp, *gW2 = gp + (size_t)Cl * 27 * Hd;
    const dim3 grid((unsigned)((w + LD_TC - 1) / LD_TC), (unsigned)h, (unsigned)nq), grid2((unsigned)((w + LD_T2 - 1) / LD_T2), (unsigned)h, (unsigned)nq);
    if (!ld_ffma() && !(w & 1)) {   /* (w even: the gradient rows' fp16 pairs stay 4-byte aligned) */
        int KP; const __half *pk = ld_pack(Hd, Cl, no, p, &KP); const __half *Bt = pk, *W2t = pk + (size_t)Hd * KP + (size_t)no * Hd; const size_t sm = ld_smem_tc(Hd, Cl, KP, 1);
        __half *Hh = (__half *)scratch; const int nch = (int)((N + LD_CH - 1) / LD_CH);
#define LDB(HD) do { ld_attr((const void *)latdec_bwd_tc_k<HD>, sm); latdec_bwd_tc_k<HD><<<grid2, LD_NT, sm>>>((const f16 *)gy16, ldg, (f16 *)gpre16, (const f16 *)pre16, ldp, L, nq, h, w, Bt, KP, W2t, Cl, no, Hh, N); \
                     latdec_gw_tc_k<HD, 0><<<dim3((unsigned)((no + 63) / 64), (unsigned)nch), LD_NT>>>((const f16 *)gy16, ldg, L, nq, h, w, no, Hh, N, N, part); KCHECK(); \
                     latdec_red_k<<<nblk((size_t)no * Hd, 256), 256>>>(part, nch, (size_t)no * Hd, gW2, GSHP); KCHECK(); \
                     latdec_gw_tc_k<HD, 1><<<dim3((unsigned)((Cl * 27 + 63) / 64), (unsigned)nch), LD_NT>>>(nullptr, 0, L, nq, h, w, Cl * 27, (const __half *)gpre16, ldg, N, part); KCHECK(); \
                     latdec_red_k<<<nblk((size_t)Cl * 27 * Hd, 256), 256>>>(part, nch, (size_t)Cl * 27 * Hd, gB, GSHP); KCHECK(); } while (0)
        if (Hd == 16) LDB(16); else if (Hd == 32) LDB(32); else LDB(64);
#undef LDB
        return 0;
    }
    if (g_ld_wq) return 1;   /* (the fp4 weights: tensor-core path only) */
    const size_t sm = ld_smem(Cl, no, 1);
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
