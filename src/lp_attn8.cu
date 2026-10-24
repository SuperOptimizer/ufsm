/* MX-fp8 attention core (at_set_core 2): the products of src/nn_attn_tc.cu on block-scaled e4m3 tensor cores (mma m16n8k32
   kind::mxf8f6f4, one ue8m0 scale per 32 K, fp32 accumulation), head dim 48 (K = 48 padded to 64 for the products over dims).

   Sources: the fp16 head-major q / k / v of at_fa_pack and the scaled fp16 dO of at_fa_bprep, converted per call into
   - row-quantised tensors X8[h][R][64] (+ scales [h][R][2]: dims 0..31, 32..47) for the products over the head dims (Q K^T, dO V^T);
   - transposed tensors Xt8[h][48][R] (+ scales [h][48][R / 32]), quantised along the rows in 32-row blocks, for the products over the
     tokens (P V, P^T dO, dS^T Q, dS K). Inside a block, slot k holds row pi(k) = 16 (k / 16) + 2 ((k % 16) / 4) + (k & 1) + 8 ((k / 2) & 1):
     the order in which an m16n8 accumulator's columns land in the k32 A fragment, so P / dS go from accumulators to A fragments in
     registers (amax per row per 32 columns over the quad, e4m3 conversion) with no shuffle of the values.
   R = nseq Sp, Sp = S rounded up to 64: every sequence starts on a tile (rows past S zero). Masks, lse / D, the dO and dS gradient scaling
   and the outputs are those of the fp16 path. */
#include "lp_common.cuh"
#include "attn.h"

#define HD AT_HD
#define FT 64
#define RS8 80   /* shared row stride (bytes) of 64-byte rows: conflict-free fragment loads */
static const float IQ8 = 1.f / 448.f;

__device__ __forceinline__ void cpa16_(void *s, const void *g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned)__cvta_generic_to_shared(s)), "l"(g));
}
__device__ __forceinline__ void cpa_commit_() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cpa_wait_() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ int pi32(int k) { const int r = k & 15; return (k & 16) + 2 * (r >> 2) + (r & 1) + 8 * ((r >> 1) & 1); }

/* ---- conversions ---- */
/* row-quantised: thread per (h, row, 32-dim block); src fp16 [h][T + 64][48] (row seq S + i), dst [h][R][64] + [h][R][2] */
__global__ void k_a8_rq(const __half *__restrict__ src, uint8_t *__restrict__ dst, uint8_t *__restrict__ sc, int H, int nseq, int S, int Sp, long long Tp) {
    const long long R = (long long)nseq * Sp, i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= (long long)H * R * 2) return;
    const int kb = (int)(i & 1); const long long hr = i >> 1; const int h = (int)(hr / R); const long long row = hr % R;
    const int seq = (int)(row / Sp), ii = (int)(row % Sp), n = kb ? HD - 32 : 32;
    float v[32];
    const __half *s = src + ((size_t)h * Tp + (size_t)seq * S + ii) * HD + 32 * kb;
#pragma unroll
    for (int j = 0; j < 32; j++) v[j] = ii < S && j < n ? __half2float(s[j]) : 0.f;
    float am = 0.f;
#pragma unroll
    for (int j = 0; j < 32; j++) am = fmaxf(am, fabsf(v[j]));
    const int e = mx_exp(am, IQ8); const float m = exp2i(-e);
    unsigned *o = (unsigned *)(dst + ((size_t)h * R + row) * 64 + 32 * kb);
#pragma unroll
    for (int w = 0; w < 8; w++) o[w] = cvt_e4m3x4(v[4 * w] * m, v[4 * w + 1] * m, v[4 * w + 2] * m, v[4 * w + 3] * m);
    sc[((size_t)h * R + row) * 2 + kb] = (uint8_t)(e + 127);
}
/* transposed: block per (h, 32-row block), 64 threads; dst [h][48][R] in pi order + [h][48][R / 32] */
__global__ void k_a8_tq(const __half *__restrict__ src, uint8_t *__restrict__ dst, uint8_t *__restrict__ sc, int H, int nseq, int S, int Sp, long long Tp) {
    const long long R = (long long)nseq * Sp; const int nb = (int)(R / 32), h = blockIdx.x / nb, b = blockIdx.x % nb;
    const long long r0 = (long long)b * 32; const int seq = (int)(r0 / Sp), i0 = (int)(r0 % Sp);
    __shared__ float t[32][HD + 1];
    for (int e = threadIdx.x; e < 32 * HD; e += blockDim.x) {
        const int r = e / HD, d = e % HD, ii = i0 + r;
        t[r][d] = ii < S ? __half2float(src[((size_t)h * Tp + (size_t)seq * S + ii) * HD + d]) : 0.f;
    }
    __syncthreads();
    const int d = threadIdx.x;
    if (d >= HD) return;
    float am = 0.f;
    for (int r = 0; r < 32; r++) am = fmaxf(am, fabsf(t[r][d]));
    const int e = mx_exp(am, IQ8); const float m = exp2i(-e);
    unsigned *o = (unsigned *)(dst + ((size_t)h * HD + d) * R + r0);
#pragma unroll
    for (int w = 0; w < 8; w++) o[w] = cvt_e4m3x4(t[pi32(4 * w)][d] * m, t[pi32(4 * w + 1)][d] * m, t[pi32(4 * w + 2)][d] * m, t[pi32(4 * w + 3)][d] * m);
    sc[((size_t)h * HD + d) * (R / 32) + b] = (uint8_t)(e + 127);
}

/* ---- fragments ---- */
/* A fragments of rows r, r + 8 (r = the warp's g row) of a row-quantised tensor: [kh][4] + scales [kh] (row g + 8 (t & 1)) */
__device__ __forceinline__ void afrag8(unsigned (*a)[4], unsigned *sa, const uint8_t *X, const uint8_t *Xs, long long r, int t) {
#pragma unroll
    for (int kh = 0; kh < 2; kh++) {
        a[kh][0] = *(const unsigned *)(X + r * 64 + 32 * kh + 4 * t); a[kh][1] = *(const unsigned *)(X + (r + 8) * 64 + 32 * kh + 4 * t);
        a[kh][2] = *(const unsigned *)(X + r * 64 + 32 * kh + 16 + 4 * t); a[kh][3] = *(const unsigned *)(X + (r + 8) * 64 + 32 * kh + 16 + 4 * t);
        sa[kh] = Xs[(r + 8 * (t & 1)) * 2 + kh];
    }
}
/* stage a 64-row tile of a row-quantised tensor (rows r0.., [64][RS8] + scales [64][2]) */
__device__ __forceinline__ void stage_rq(uint8_t *s, uint8_t *ss, const uint8_t *X, const uint8_t *Xs, long long r0, int tid) {
    for (int c = tid; c < FT * 4; c += 128) { const int r = c >> 2, k = c & 3; cpa16_(s + r * RS8 + 16 * k, X + (r0 + r) * 64 + 16 * k); }
    if (tid < 8) cpa16_(ss + 16 * tid, Xs + r0 * 2 + 16 * tid);
}
/* stage the 64 columns c0.. of a transposed tensor ([48][RS8] + scales [48][2]) */
__device__ __forceinline__ void stage_tq(uint8_t *s, uint8_t *ss, const uint8_t *X, const uint8_t *Xs, long long R, long long c0, int tid) {
    for (int c = tid; c < HD * 4; c += 128) { const int d = c >> 2, k = c & 3; cpa16_(s + d * RS8 + 16 * k, X + (size_t)d * R + c0 + 16 * k); }
    if (tid < HD) *(unsigned short *)(ss + 2 * tid) = *(const unsigned short *)(Xs + (size_t)tid * (R / 32) + c0 / 32);
}
/* s[8][4] = A (16 x 48 dims) . tile^T (64 rows of a row-quantised tile) */
__device__ __forceinline__ void mm8_rt(float (*s)[4], unsigned (*a)[4], const unsigned *sa, const uint8_t *tile, const uint8_t *ts, int g, int t) {
#pragma unroll
    for (int j = 0; j < 8; j++)
#pragma unroll
        for (int q = 0; q < 4; q++) s[j][q] = 0.f;
#pragma unroll
    for (int kh = 0; kh < 2; kh++)
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int r = 8 * j + g; unsigned b[2] = {*(const unsigned *)(tile + r * RS8 + 32 * kh + 4 * t), *(const unsigned *)(tile + r * RS8 + 32 * kh + 16 + 4 * t)};
            mma_f8(s[j], a[kh], b, sa[kh], ts[r * 2 + kh]);
        }
}
/* acc[6][4] += P (16 x 64 tile rows, accumulator layout; quantised here per row per 32 columns) . tile (transposed [48][64]) */
__device__ __forceinline__ void mm8_pt(float (*acc)[4], float (*p)[4], const uint8_t *tile, const uint8_t *ts, int g, int t) {
#pragma unroll
    for (int b = 0; b < 2; b++) {
        float m0 = 0.f, m1 = 0.f;
#pragma unroll
        for (int jj = 0; jj < 4; jj++) { m0 = fmaxf(m0, fmaxf(fabsf(p[4 * b + jj][0]), fabsf(p[4 * b + jj][1]))); m1 = fmaxf(m1, fmaxf(fabsf(p[4 * b + jj][2]), fabsf(p[4 * b + jj][3]))); }
        m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1)); m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
        m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1)); m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
        const int e0 = mx_exp(m0, IQ8), e1 = mx_exp(m1, IQ8); const float s0 = exp2i(-e0), s1 = exp2i(-e1);
        const float (*q)[4] = p + 4 * b;
        unsigned a[4] = {cvt_e4m3x4(q[0][0] * s0, q[0][1] * s0, q[1][0] * s0, q[1][1] * s0), cvt_e4m3x4(q[0][2] * s1, q[0][3] * s1, q[1][2] * s1, q[1][3] * s1),
                         cvt_e4m3x4(q[2][0] * s0, q[2][1] * s0, q[3][0] * s0, q[3][1] * s0), cvt_e4m3x4(q[2][2] * s1, q[2][3] * s1, q[3][2] * s1, q[3][3] * s1)};
        const unsigned sa = (unsigned)((t & 1 ? e1 : e0) + 127);
#pragma unroll
        for (int jd = 0; jd < 6; jd++) {
            const int d = 8 * jd + g; unsigned bb[2] = {*(const unsigned *)(tile + d * RS8 + 32 * b + 4 * t), *(const unsigned *)(tile + d * RS8 + 32 * b + 16 + 4 * t)};
            mma_f8(acc[jd], a, bb, sa, ts[d * 2 + b]);
        }
    }
}

/* ---- forward: grid (Sp / 64, H, nseq) ---- */
__global__ void __launch_bounds__(128, 4) k_a8_fwd(const uint8_t *__restrict__ Q8, const uint8_t *__restrict__ Q8s, const uint8_t *__restrict__ K8, const uint8_t *__restrict__ K8s,
                                                const uint8_t *__restrict__ Vt, const uint8_t *__restrict__ Vts, float *__restrict__ o, float *__restrict__ lse,
                                                int S, int Sp, int H, int C, long long R, float sl2) {
    const int seq = blockIdx.z, h = blockIdx.y, q0 = blockIdx.x * FT, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    __shared__ __align__(16) uint8_t Ks[2][FT * RS8], Vs[2][HD * RS8], Kss[2][FT * 2], Vss[2][HD * 2];
    const long long rb = (long long)h * R + (long long)seq * Sp;   /* this sequence's first row in the row-quantised tensors */
    const uint8_t *K = K8 + rb * 64, *Ksc = K8s + rb * 2, *V = Vt + (size_t)h * HD * R, *Vsc = Vts + (size_t)h * HD * (R / 32);
    const long long c0 = (long long)seq * Sp;   /* its first column in the transposed ones */
    const int nt = (S + FT - 1) / FT;
    stage_rq(Ks[0], Kss[0], K, Ksc, 0, tid); stage_tq(Vs[0], Vss[0], V, Vsc, R, c0, tid); cpa_commit_();
    unsigned qa[2][4], qs[2]; afrag8(qa, qs, Q8 + rb * 64, Q8s + rb * 2, q0 + warp * 16 + g, t);
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f}, acc[6][4];
#pragma unroll
    for (int j = 0; j < 6; j++)
#pragma unroll
        for (int q = 0; q < 4; q++) acc[j][q] = 0.f;
    for (int it = 0; it < nt; it++) {
        const int cur = it & 1;
        if (it + 1 < nt) { stage_rq(Ks[cur ^ 1], Kss[cur ^ 1], K, Ksc, (it + 1) * FT, tid); stage_tq(Vs[cur ^ 1], Vss[cur ^ 1], V, Vsc, R, c0 + (it + 1) * FT, tid); }
        cpa_commit_(); cpa_wait_<1>(); __syncthreads();
        float s[8][4]; mm8_rt(s, qa, qs, Ks[cur], Kss[cur], g, t);
        const int kb = it * FT; const bool full = kb + FT <= S;   /* (only the last tile masks) */
        float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int q = 0; q < 4; q++) { const float v = full || kb + 8 * j + 2 * t + (q & 1) < S ? s[j][q] * sl2 : -INFINITY; s[j][q] = v; mx[q >> 1] = fmaxf(mx[q >> 1], v); }
        float corr[2];
#pragma unroll
        for (int r = 0; r < 2; r++) {
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1)); mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
            const float mn = fmaxf(m[r], mx[r]); corr[r] = exp2f(m[r] - mn); m[r] = mn; l[r] *= corr[r];
        }
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int q = 0; q < 4; q++) { const float pv = exp2f(s[j][q] - m[q >> 1]); s[j][q] = pv; l[q >> 1] += pv; }
#pragma unroll
        for (int j = 0; j < 6; j++)
#pragma unroll
            for (int q = 0; q < 4; q++) acc[j][q] *= corr[q >> 1];
        mm8_pt(acc, s, Vs[cur], Vss[cur], g, t);
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < 2; r++) { l[r] += __shfl_xor_sync(0xffffffffu, l[r], 1); l[r] += __shfl_xor_sync(0xffffffffu, l[r], 2); }
#pragma unroll
    for (int r = 0; r < 2; r++) {
        const int i = q0 + warp * 16 + g + 8 * r;
        if (i >= S) continue;
        const float il = 1.f / l[r]; float *orow = o + ((size_t)seq * S + i) * C + h * HD;
#pragma unroll
        for (int j = 0; j < 6; j++) *(float2 *)(orow + 8 * j + 2 * t) = make_float2(acc[j][2 * r] * il, acc[j][2 * r + 1] * il);
        if (!t) lse[((size_t)seq * H + h) * S + i] = m[r] + log2f(l[r]);
    }
}

__device__ __forceinline__ int a8_e(const float *amx) { const float a = amx[0]; return a > 0.f && a < INFINITY ? -ilogbf(a) - 1 : 0; }   /* as nn_attn_tc.cu */

/* ---- dK, dV: grid (Sp / 64 key tiles, H, nseq) ---- */
struct a8t { const uint8_t *Q8, *Q8s, *K8, *K8s, *V8, *V8s, *G8, *G8s, *Qt, *Qts, *Kt, *Kts, *Gt, *Gts; };
__global__ void __launch_bounds__(128, 3) k_a8_dkv(a8t A, const float *__restrict__ lse, const float *__restrict__ Dv, const float *__restrict__ amx, float *__restrict__ dqkn,
                                                int S, int Sp, int H, int C, long long R, float sl2, float scale) {
    const int seq = blockIdx.z, h = blockIdx.y, j0 = blockIdx.x * FT, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    __shared__ __align__(16) uint8_t Qs[2][FT * RS8], Gs[2][FT * RS8], Qts[2][HD * RS8], Gts[2][HD * RS8], Qss[2][FT * 2], Gss[2][FT * 2], Qtss[2][HD * 2], Gtss[2][HD * 2];
    __shared__ float Ls[2][FT], Ds[2][FT];
    const long long rb = (long long)h * R + (long long)seq * Sp, c0 = (long long)seq * Sp;
    const uint8_t *Q = A.Q8 + rb * 64, *Qsc = A.Q8s + rb * 2, *G = A.G8 + rb * 64, *Gsc = A.G8s + rb * 2;
    const uint8_t *QT = A.Qt + (size_t)h * HD * R, *QTs = A.Qts + (size_t)h * HD * (R / 32), *GT = A.Gt + (size_t)h * HD * R, *GTs = A.Gts + (size_t)h * HD * (R / 32);
    const float *L = lse + ((size_t)seq * H + h) * S, *Dr = Dv + ((size_t)seq * H + h) * S;
    const int e = a8_e(amx);
    const int nt = (S + FT - 1) / FT;
    auto stage = [&](int buf, int it) {
        stage_rq(Qs[buf], Qss[buf], Q, Qsc, it * FT, tid); stage_rq(Gs[buf], Gss[buf], G, Gsc, it * FT, tid);
        stage_tq(Qts[buf], Qtss[buf], QT, QTs, R, c0 + it * FT, tid); stage_tq(Gts[buf], Gtss[buf], GT, GTs, R, c0 + it * FT, tid);
        if (tid < FT) { const int i = it * FT + tid; Ls[buf][tid] = i < S ? L[i] : INFINITY; Ds[buf][tid] = i < S ? Dr[i] : 0.f; }
    };
    stage(0, 0); cpa_commit_();
    unsigned ka[2][4], ks[2], va[2][4], vs[2];
    afrag8(ka, ks, A.K8 + rb * 64, A.K8s + rb * 2, j0 + warp * 16 + g, t); afrag8(va, vs, A.V8 + rb * 64, A.V8s + rb * 2, j0 + warp * 16 + g, t);
    float dk[6][4], dv[6][4];
#pragma unroll
    for (int j = 0; j < 6; j++)
#pragma unroll
        for (int q = 0; q < 4; q++) { dk[j][q] = 0.f; dv[j][q] = 0.f; }
    for (int it = 0; it < nt; it++) {
        const int cur = it & 1;
        if (it + 1 < nt) stage(cur ^ 1, it + 1);
        cpa_commit_(); cpa_wait_<1>(); __syncthreads();
        float p[8][4], ds[8][4];
        mm8_rt(p, ka, ks, Qs[cur], Qss[cur], g, t);    /* S^T */
        mm8_rt(ds, va, vs, Gs[cur], Gss[cur], g, t);   /* dP^T */
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const int c = 8 * j + 2 * t + (q & 1);
                const float pv = exp2f(p[j][q] * sl2 - Ls[cur][c]);
                p[j][q] = pv; ds[j][q] = pv * (ds[j][q] - Ds[cur][c]);
            }
        mm8_pt(dv, p, Gts[cur], Gtss[cur], g, t);    /* dV += P^T dO */
        mm8_pt(dk, ds, Qts[cur], Qtss[cur], g, t);   /* dK += dS^T Q */
        __syncthreads();
    }
    const float fk = scale * exp2f((float)-e), fv = exp2f((float)-e);
#pragma unroll
    for (int r = 0; r < 2; r++) {
        const int j = j0 + warp * 16 + g + 8 * r;
        if (j >= S) continue;
        float *row = dqkn + ((size_t)seq * S + j) * 3 * C + h * HD;
#pragma unroll
        for (int jd = 0; jd < 6; jd++) {
            *(float2 *)(row + C + 8 * jd + 2 * t) = make_float2(dk[jd][2 * r] * fk, dk[jd][2 * r + 1] * fk);
            *(float2 *)(row + 2 * C + 8 * jd + 2 * t) = make_float2(dv[jd][2 * r] * fv, dv[jd][2 * r + 1] * fv);
        }
    }
}

/* ---- dQ: grid (Sp / 64 query tiles, H, nseq) ---- */
__global__ void __launch_bounds__(128, 4) k_a8_dq(a8t A, const float *__restrict__ lse, const float *__restrict__ Dv, const float *__restrict__ amx, float *__restrict__ dqkn,
                                               int S, int Sp, int H, int C, long long R, float sl2, float scale) {
    const int seq = blockIdx.z, h = blockIdx.y, q0 = blockIdx.x * FT, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    __shared__ __align__(16) uint8_t Ks[2][FT * RS8], Vs[2][FT * RS8], Kts[2][HD * RS8], Kss[2][FT * 2], Vss[2][FT * 2], Ktss[2][HD * 2];
    const long long rb = (long long)h * R + (long long)seq * Sp, c0 = (long long)seq * Sp;
    const uint8_t *K = A.K8 + rb * 64, *Ksc = A.K8s + rb * 2, *V = A.V8 + rb * 64, *Vsc = A.V8s + rb * 2;
    const uint8_t *KT = A.Kt + (size_t)h * HD * R, *KTs = A.Kts + (size_t)h * HD * (R / 32);
    const float *L = lse + ((size_t)seq * H + h) * S, *Dr = Dv + ((size_t)seq * H + h) * S;
    const int e = a8_e(amx);
    const int nt = (S + FT - 1) / FT;
    auto stage = [&](int buf, int it) {
        stage_rq(Ks[buf], Kss[buf], K, Ksc, it * FT, tid); stage_rq(Vs[buf], Vss[buf], V, Vsc, it * FT, tid);
        stage_tq(Kts[buf], Ktss[buf], KT, KTs, R, c0 + it * FT, tid);
    };
    stage(0, 0); cpa_commit_();
    const int i0 = q0 + warp * 16 + g;
    unsigned qa[2][4], qs[2], ga[2][4], gs[2];
    afrag8(qa, qs, A.Q8 + rb * 64, A.Q8s + rb * 2, i0, t); afrag8(ga, gs, A.G8 + rb * 64, A.G8s + rb * 2, i0, t);
    const float li[2] = {i0 < S ? L[i0] : INFINITY, i0 + 8 < S ? L[i0 + 8] : INFINITY}, di[2] = {i0 < S ? Dr[i0] : 0.f, i0 + 8 < S ? Dr[i0 + 8] : 0.f};
    float dq[6][4];
#pragma unroll
    for (int j = 0; j < 6; j++)
#pragma unroll
        for (int q = 0; q < 4; q++) dq[j][q] = 0.f;
    for (int it = 0; it < nt; it++) {
        const int cur = it & 1;
        if (it + 1 < nt) stage(cur ^ 1, it + 1);
        cpa_commit_(); cpa_wait_<1>(); __syncthreads();
        float p[8][4], ds[8][4];
        mm8_rt(p, qa, qs, Ks[cur], Kss[cur], g, t);    /* S */
        mm8_rt(ds, ga, gs, Vs[cur], Vss[cur], g, t);   /* dP */
        const int kb = it * FT; const bool full = kb + FT <= S;
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const float pv = full || kb + 8 * j + 2 * t + (q & 1) < S ? exp2f(p[j][q] * sl2 - li[q >> 1]) : 0.f;
                ds[j][q] = pv * (ds[j][q] - di[q >> 1]);
            }
        mm8_pt(dq, ds, Kts[cur], Ktss[cur], g, t);   /* dQ += dS K */
        __syncthreads();
    }
    const float fq = scale * exp2f((float)-e);
#pragma unroll
    for (int r = 0; r < 2; r++) {
        const int i = i0 + 8 * r;
        if (i >= S) continue;
        float *row = dqkn + ((size_t)seq * S + i) * 3 * C + h * HD;
#pragma unroll
        for (int jd = 0; jd < 6; jd++) *(float2 *)(row + 8 * jd + 2 * t) = make_float2(dq[jd][2 * r] * fq, dq[jd][2 * r + 1] * fq);
    }
}

/* ---- host: scratch (grown on demand), conversions, launches ---- */
static uint8_t *g_a8 = nullptr; static size_t g_a8n = 0;
typedef struct { uint8_t *q, *k, *v, *g, *qs, *ks, *vs, *gs, *qt, *kt, *vt, *gt, *qts, *kts, *vts, *gts; } a8buf;
static int a8_bufs(a8buf *b, int H, long long R) {
    const size_t rq = (size_t)H * R * 64, rqs = (size_t)H * R * 2, tq = (size_t)H * HD * R, tqs = (size_t)H * HD * (R / 32);
    const size_t al = 256, n = 4 * ((rq + al) + (rqs + al) + (tq + al) + (tqs + al));
    if (n > g_a8n) { if (g_a8) cudaFree(g_a8); if (cudaMalloc(&g_a8, n) != cudaSuccess) { g_a8 = nullptr; g_a8n = 0; return -1; } g_a8n = n; }
    uint8_t *p = g_a8;
    auto take = [&](size_t m) { uint8_t *r = p; p += (m + al - 1) / al * al; return r; };
    b->q = take(rq); b->k = take(rq); b->v = take(rq); b->g = take(rq); b->qs = take(rqs); b->ks = take(rqs); b->vs = take(rqs); b->gs = take(rqs);
    b->qt = take(tq); b->kt = take(tq); b->vt = take(tq); b->gt = take(tq); b->qts = take(tqs); b->kts = take(tqs); b->vts = take(tqs); b->gts = take(tqs);
    return 0;
}
static void a8_rq(const __half *src, uint8_t *d, uint8_t *s, int H, const at_geom *g, int Sp, long long Tp) {
    const long long n = (long long)H * g->nseq * Sp * 2; k_a8_rq<<<nblk_(n, 128), 128>>>(src, d, s, H, g->nseq, g->S, Sp, Tp);
}
static void a8_tq(const __half *src, uint8_t *d, uint8_t *s, int H, const at_geom *g, int Sp, long long Tp) {
    k_a8_tq<<<(unsigned)(H * (g->nseq * (long long)Sp / 32)), 64>>>(src, d, s, H, g->nseq, g->S, Sp, Tp);
}
extern "C" int at_fa8_fwd(const void *qkvh, float *o, float *lse, const at_geom *g, int H, int C) {
    const long long T = (long long)g->nseq * g->S, Tp = T + FT; const int Sp = (g->S + FT - 1) / FT * FT; const long long R = (long long)g->nseq * Sp;
    a8buf b; if (a8_bufs(&b, H, R)) return -1;
    const __half *Qh = (const __half *)qkvh, *Kh = Qh + (size_t)H * Tp * HD, *Vh = Kh + (size_t)H * Tp * HD;
    a8_rq(Qh, b.q, b.qs, H, g, Sp, Tp); a8_rq(Kh, b.k, b.ks, H, g, Sp, Tp); a8_tq(Vh, b.vt, b.vts, H, g, Sp, Tp);
    dim3 gr(Sp / FT, H, g->nseq);
    k_a8_fwd<<<gr, 128>>>(b.q, b.qs, b.k, b.ks, b.vt, b.vts, o, lse, g->S, Sp, H, C, R, 1.4426950408889634f / sqrtf((float)HD));
    LPCK();
    return 0;
}
extern "C" void at_fa_bprep(const float *o, const float *dO, float *amx, void *doh, float *Dv, const at_geom *g, int C);
extern "C" int at_fa8_bwd(const void *qkvh, void *doh, const float *o, const float *lse, const float *dO, float *Dv, float *amx, float *dqkn, const at_geom *g, int H, int C) {
    const long long T = (long long)g->nseq * g->S, Tp = T + FT; const int Sp = (g->S + FT - 1) / FT * FT; const long long R = (long long)g->nseq * Sp;
    a8buf b; if (a8_bufs(&b, H, R)) return -1;
    at_fa_bprep(o, dO, amx, doh, Dv, g, C);
    const __half *Qh = (const __half *)qkvh, *Kh = Qh + (size_t)H * Tp * HD, *Vh = Kh + (size_t)H * Tp * HD, *Gh = (const __half *)doh;
    a8_rq(Qh, b.q, b.qs, H, g, Sp, Tp); a8_rq(Kh, b.k, b.ks, H, g, Sp, Tp); a8_rq(Vh, b.v, b.vs, H, g, Sp, Tp); a8_rq(Gh, b.g, b.gs, H, g, Sp, Tp);
    a8_tq(Qh, b.qt, b.qts, H, g, Sp, Tp); a8_tq(Kh, b.kt, b.kts, H, g, Sp, Tp); a8_tq(Gh, b.gt, b.gts, H, g, Sp, Tp);
    const a8t A = {b.q, b.qs, b.k, b.ks, b.v, b.vs, b.g, b.gs, b.qt, b.qts, b.kt, b.kts, b.gt, b.gts};
    const float scale = 1.f / sqrtf((float)HD), sl2 = scale * 1.4426950408889634f;
    dim3 gr(Sp / FT, H, g->nseq);
    k_a8_dkv<<<gr, 128>>>(A, lse, Dv, amx, dqkn, g->S, Sp, H, C, R, sl2, scale);
    k_a8_dq<<<gr, 128>>>(A, lse, Dv, amx, dqkn, g->S, Sp, H, C, R, sl2, scale);
    LPCK();
    return 0;
}
