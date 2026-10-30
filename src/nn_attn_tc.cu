/* Tensor-core attention core of the transformer blocks (src/attn.h, at_set_core 1): fp16 mma.sync m16n8k16 with fp32 accumulation,
   head dim 48, flash-style (online softmax, per-row log2-sum-exp only; the backward recomputes the probabilities tile by tile).

   Operands: q, k, v of every head packed by at_fa_pack (QK-norm + gain + RoPE fused) into fp16 head-major rows X[h][T + 64][48]
   (row = the token's global index seq * S + i; the 64 rows past T are zero, so a 64-row tile never leaves the buffer; rows of the next
   sequence that a tile reaches are masked: keys by -inf logits, queries by +inf log-sum-exp in the backward / not stored).
   Gradient side: dO packed to fp16 scaled by 2^e (e from amax |dO|: |dO| 2^e <= 1, no fp16 overflow anywhere below), dS = P (dP - D)
   scaled by 2^kS with kS from the bound |dP| <= 48 max|dO 2^e| max|v| (|dS| 2^kS <= 2^14); the results are unscaled in fp32.
   CTA = 4 warps x 16 rows = 64 queries (forward, dQ) or 64 keys (dK / dV); 64-row tiles of the other side stream through shared memory
   (cp.async double buffer, rows padded to 56 halfs: conflict-free ldmatrix). */
#include "attn.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#define HD AT_HD
#define FT 64   /* tile rows */
#define FS 56   /* shared-memory row stride (halfs) */

__device__ __forceinline__ void cpa16(void *s, const void *g) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a), "l"(g));
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm4(unsigned *r, const __half *p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm4t(unsigned *r, const __half *p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma16(float *c, const unsigned *a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned pk(float a, float b) { __half2 h = __floats2half2_rn(a, b); return *(unsigned *)&h; }
__device__ __forceinline__ unsigned ld32(const __half *p) { return *(const unsigned *)p; }

/* a 64 x 48 tile (rows r0..r0+63 of a packed head) into shared memory */
__device__ __forceinline__ void stage(__half *sm, const __half *g, int tid) {
#pragma unroll
    for (int c = tid; c < FT * 6; c += 128) { const int r = c / 6, k = c % 6; cpa16(sm + r * FS + k * 8, g + (size_t)r * HD + k * 8); }
}
/* A fragments (16 rows x 48) of rows r (= the warp's g row) and r + 8 from a packed head in global memory */
__device__ __forceinline__ void afrag(unsigned (*a)[4], const __half *X, int r, int t) {
#pragma unroll
    for (int kk = 0; kk < 3; kk++) {
        a[kk][0] = ld32(X + (size_t)r * HD + 16 * kk + 2 * t); a[kk][1] = ld32(X + (size_t)(r + 8) * HD + 16 * kk + 2 * t);
        a[kk][2] = ld32(X + (size_t)r * HD + 16 * kk + 8 + 2 * t); a[kk][3] = ld32(X + (size_t)(r + 8) * HD + 16 * kk + 8 + 2 * t);
    }
}
/* s[8][4] (16 rows x 64 tile rows) = A (16 x 48) . tile^T: B[k = dim][n = tile row] from the row-major tile (ldmatrix, no transpose) */
__device__ __forceinline__ void mm_rt(float (*s)[4], unsigned (*a)[4], const __half *tile, int lane) {
    const int r = lane & 7, mi = lane >> 3;
#pragma unroll
    for (int j = 0; j < 8; j++)
#pragma unroll
        for (int q = 0; q < 4; q++) s[j][q] = 0.f;
#pragma unroll
    for (int kk = 0; kk < 3; kk++)
#pragma unroll
        for (int jp = 0; jp < 4; jp++) {
            unsigned b[4]; ldsm4(b, tile + (16 * jp + r + (mi >> 1) * 8) * FS + 16 * kk + (mi & 1) * 8);
            mma16(s[2 * jp], a[kk], b[0], b[1]); mma16(s[2 * jp + 1], a[kk], b[2], b[3]);
        }
}
/* acc[6][4] (16 rows x 48) += P (16 x 64, accumulator layout) . tile: B[k = tile row][n = dim] (ldmatrix transposed) */
__device__ __forceinline__ void mm_pt(float (*acc)[4], float (*p)[4], const __half *tile, int lane) {
    const int r = lane & 7, mi = lane >> 3;
#pragma unroll
    for (int k2 = 0; k2 < 4; k2++) {
        unsigned a[4] = {pk(p[2 * k2][0], p[2 * k2][1]), pk(p[2 * k2][2], p[2 * k2][3]), pk(p[2 * k2 + 1][0], p[2 * k2 + 1][1]), pk(p[2 * k2 + 1][2], p[2 * k2 + 1][3])};
#pragma unroll
        for (int jp = 0; jp < 3; jp++) {
            unsigned b[4]; ldsm4t(b, tile + (16 * k2 + r + (mi & 1) * 8) * FS + 16 * jp + (mi >> 1) * 8);
            mma16(acc[2 * jp], a, b[0], b[1]); mma16(acc[2 * jp + 1], a, b[2], b[3]);
        }
    }
}

/* ---- pack: qkv [T][3C] fp32 -> QK-norm (+ gain), RoPE, fp16 head-major q / k / v; rows T..T+63 zero; amx[1] = max |v| ---- */
/* 8 consecutive values (16- / 32-byte aligned) of an fp32 or bf16 row -> fp32 */
__device__ __forceinline__ void ld8f(const float *p, float *v) { const float4 a = *(const float4 *)p, b = *(const float4 *)(p + 4); v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w; }
__device__ __forceinline__ void ld8f(const __nv_bfloat16 *p, float *v) {
    const uint4 u = *(const uint4 *)p; const unsigned w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int k = 0; k < 4; k++) { v[2 * k] = __uint_as_float(w[k] << 16); v[2 * k + 1] = __uint_as_float(w[k] & 0xffff0000u); }
}
__device__ __forceinline__ void st2(float *p, float a, float b) { *(float2 *)p = make_float2(a, b); }
__device__ __forceinline__ void st2(__nv_bfloat16 *p, float a, float b) { *(__nv_bfloat162 *)p = __floats2bfloat162_rn(a, b); }
template <typename TI>
__global__ void k_fa_pack(const TI *__restrict__ qkv, const float *__restrict__ qg, const float *__restrict__ kg, __half *Qh, __half *Kh, __half *Vh,
                          float *rq, float *rk, float *amx, long long T, int C, int S, int Hh, int W, int nreg, int rope, float lbase, const float *gpos) {
    const int H = C / HD; const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= (T + FT) * H * 3) return;
    const int p = (int)(i % 3); const long long th = i / 3; const int h = (int)(th % H); const long long t = th / H;
    __half *dst = (p == 0 ? Qh : p == 1 ? Kh : Vh) + ((size_t)h * (T + FT) + t) * HD;
    if (t >= T) { for (int d = 0; d < HD; d += 8) *(uint4 *)(dst + d) = make_uint4(0, 0, 0, 0); return; }
    const TI *xs = qkv + t * 3 * C + p * C + h * HD;
    float v[HD];
#pragma unroll
    for (int d = 0; d < HD; d += 8) ld8f(xs + d, v + d);
    if (p < 2) {
        float s = 0.f;
#pragma unroll
        for (int d = 0; d < HD; d++) s += v[d] * v[d];
        const float r = 1.f / sqrtf(s / HD + 1e-6f); const float *gs = (p ? kg : qg) + h * HD;
#pragma unroll
        for (int d = 0; d < HD; d++) v[d] *= r * gs[d];
        (p ? rk : rq)[t * H + h] = r;
        const int ts = (int)(t % S) - nreg;
        if (rope && ts >= 0) {
            const float pos[3] = {gpos ? gpos[3 * t] : (float)(ts / (Hh * W)), gpos ? gpos[3 * t + 1] : (float)((ts / W) % Hh), gpos ? gpos[3 * t + 2] : (float)(ts % W)};
#pragma unroll
            for (int pr = 0; pr < 24; pr++) {
                const int ax = pr / 8, fi = pr % 8; float sn, cs; sincosf(pos[ax] * exp2f(-lbase * (float)fi / 8.f), &sn, &cs);
                const float a = v[ax * 16 + 2 * fi], b = v[ax * 16 + 2 * fi + 1];
                v[ax * 16 + 2 * fi] = a * cs - b * sn; v[ax * 16 + 2 * fi + 1] = a * sn + b * cs;
            }
        }
    } else {
        float m = 0.f;
#pragma unroll
        for (int d = 0; d < HD; d++) m = fmaxf(m, fabsf(v[d]));
        atomicMax((unsigned *)amx + 1, __float_as_uint(m));
    }
#pragma unroll
    for (int d = 0; d < HD; d += 8) *(uint4 *)(dst + d) = make_uint4(pk(v[d], v[d + 1]), pk(v[d + 2], v[d + 3]), pk(v[d + 4], v[d + 5]), pk(v[d + 6], v[d + 7]));
}

/* ---- forward: grid (ceil(S / 64), H, nseq) ---- */
template <typename TO>
__global__ void __launch_bounds__(128) k_fa_fwd(const __half *__restrict__ Qh, const __half *__restrict__ Kh, const __half *__restrict__ Vh, TO *__restrict__ o,
                                                float *__restrict__ lse, int S, int H, int C, long long Tp, float sl2, const int *lens) {
    const int seq = blockIdx.z, h = blockIdx.y, q0 = blockIdx.x * FT, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    __shared__ __align__(16) __half Ks[2][FT * FS], Vs[2][FT * FS];
    const size_t base = (size_t)h * Tp + (size_t)seq * S;
    const __half *Q = Qh + base * HD, *K = Kh + base * HD, *V = Vh + base * HD;
    const int SK = lens ? lens[seq] : S, nt = (SK + FT - 1) / FT;   /* (keys past the sequence's length masked) */
    stage(Ks[0], K, tid); stage(Vs[0], V, tid); cpa_commit();
    unsigned qa[3][4]; afrag(qa, Q, q0 + warp * 16 + g, t);
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f}, acc[6][4];
#pragma unroll
    for (int j = 0; j < 6; j++)
#pragma unroll
        for (int q = 0; q < 4; q++) acc[j][q] = 0.f;
    for (int it = 0; it < nt; it++) {
        const int cur = it & 1;
        if (it + 1 < nt) { stage(Ks[cur ^ 1], K + (size_t)(it + 1) * FT * HD, tid); stage(Vs[cur ^ 1], V + (size_t)(it + 1) * FT * HD, tid); }
        cpa_commit(); cpa_wait<1>(); __syncthreads();
        float s[8][4]; mm_rt(s, qa, Ks[cur], lane);
        const int kb = it * FT;
        float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int q = 0; q < 4; q++) { const float v = kb + 8 * j + 2 * t + (q & 1) < SK ? s[j][q] * sl2 : -INFINITY; s[j][q] = v; mx[q >> 1] = fmaxf(mx[q >> 1], v); }
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
        mm_pt(acc, s, Vs[cur], lane);
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < 2; r++) { l[r] += __shfl_xor_sync(0xffffffffu, l[r], 1); l[r] += __shfl_xor_sync(0xffffffffu, l[r], 2); }
#pragma unroll
    for (int r = 0; r < 2; r++) {
        const int i = q0 + warp * 16 + g + 8 * r;
        if (i >= S) continue;
        const float il = l[r] > 0.f ? 1.f / l[r] : 0.f; TO *orow = o + ((size_t)seq * S + i) * C + h * HD;
#pragma unroll
        for (int j = 0; j < 6; j++) st2(orow + 8 * j + 2 * t, acc[j][2 * r] * il, acc[j][2 * r + 1] * il);
        if (!t) lse[((size_t)seq * H + h) * S + i] = l[r] > 0.f ? m[r] + log2f(l[r]) : 0.f;   /* (a length-0 sequence: zero output) */
    }
}

/* ---- backward prologue: amx[0] = max |dO| ---- */
__global__ void k_fa_amax(const float *__restrict__ x, long long n, float *amx) {
    float m = 0.f;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) m = fmaxf(m, fabsf(x[i]));
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (!(threadIdx.x & 31)) atomicMax((unsigned *)amx, __float_as_uint(m));
}
__device__ __forceinline__ int fa_e(const float *amx) { const float a = amx[0]; return a > 0.f && a < INFINITY ? -ilogbf(a) - 1 : 0; }   /* |dO| 2^e < 1 */
__device__ __forceinline__ int fa_ks(const float *amx) {   /* 48 |dO 2^e| max|v| 2^kS <= 2^14 */
    const float b = 48.f * fmaxf(amx[1], 1e-30f); int k = 14 - ilogbf(b) - 1; return k < -30 ? -30 : k > 30 ? 30 : k;
}
/* D = dO . O per (row, head), scaled by 2^e; dO scaled -> fp16 head-major (rows T..T+63 zero) */
template <typename TO>
__global__ void k_fa_bprep(const TO *__restrict__ o, const float *__restrict__ dO, const float *amx, __half *dOh, float *Dv, long long T, int S, int C) {
    const int H = C / HD; const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= (T + FT) * H) return;
    const int h = (int)(i % H); const long long t = i / H;
    __half *dst = dOh + ((size_t)h * (T + FT) + t) * HD;
    if (t >= T) { for (int d = 0; d < HD; d += 8) *(uint4 *)(dst + d) = make_uint4(0, 0, 0, 0); return; }
    const float sc = exp2f((float)fa_e(amx)); const TO *orow = o + t * C + h * HD; const float *grow = dO + t * C + h * HD;
    float a = 0.f;
#pragma unroll
    for (int d = 0; d < HD; d += 8) {
        float v[8], ov[8]; ld8f(orow + d, ov);
#pragma unroll
        for (int k = 0; k < 8; k++) { v[k] = grow[d + k]; a += v[k] * ov[k]; v[k] *= sc; }
        *(uint4 *)(dst + d) = make_uint4(pk(v[0], v[1]), pk(v[2], v[3]), pk(v[4], v[5]), pk(v[6], v[7]));
    }
    Dv[((size_t)(t / S) * H + h) * S + t % S] = a * sc;
}

/* ---- dK, dV: grid (ceil(S / 64) key tiles, H, nseq); query tiles stream ---- */
__global__ void __launch_bounds__(128) k_fa_dkv(const __half *__restrict__ Qh, const __half *__restrict__ Kh, const __half *__restrict__ Vh, const __half *__restrict__ dOh,
                                                const float *__restrict__ lse, const float *__restrict__ Dv, const float *__restrict__ amx, float *__restrict__ dqkn,
                                                int S, int H, int C, long long Tp, float sl2, float scale, const int *lens) {
    const int seq = blockIdx.z, h = blockIdx.y, j0 = blockIdx.x * FT, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    __shared__ __align__(16) __half Qs[2][FT * FS], Gs[2][FT * FS];
    __shared__ float Ls[2][FT], Ds[2][FT];
    const size_t base = (size_t)h * Tp + (size_t)seq * S;
    const __half *Q = Qh + base * HD, *K = Kh + base * HD, *V = Vh + base * HD, *G = dOh + base * HD;
    const float *L = lse + ((size_t)seq * H + h) * S, *Dr = Dv + ((size_t)seq * H + h) * S;
    const int e = fa_e(amx), ks = fa_ks(amx); const float sks = exp2f((float)ks);
    const int nt = (S + FT - 1) / FT, SK = lens ? lens[seq] : S;
    /* (a masked key's row: P 0, not exp2(s - lse), which overflows once its score passes the valid keys' -> inf x 0 = NaN) */
    const float kv[2] = {j0 + warp * 16 + g < SK ? 1.f : 0.f, j0 + warp * 16 + g + 8 < SK ? 1.f : 0.f};
    stage(Qs[0], Q, tid); stage(Gs[0], G, tid); cpa_commit();
    if (tid < FT) { Ls[0][tid] = tid < S ? L[tid] : INFINITY; Ds[0][tid] = tid < S ? Dr[tid] : 0.f; }
    unsigned ka[3][4], va[3][4]; afrag(ka, K, j0 + warp * 16 + g, t); afrag(va, V, j0 + warp * 16 + g, t);
    float dk[6][4], dv[6][4];
#pragma unroll
    for (int j = 0; j < 6; j++)
#pragma unroll
        for (int q = 0; q < 4; q++) { dk[j][q] = 0.f; dv[j][q] = 0.f; }
    for (int it = 0; it < nt; it++) {
        const int cur = it & 1;
        if (it + 1 < nt) {
            stage(Qs[cur ^ 1], Q + (size_t)(it + 1) * FT * HD, tid); stage(Gs[cur ^ 1], G + (size_t)(it + 1) * FT * HD, tid);
            if (tid < FT) { const int i = (it + 1) * FT + tid; Ls[cur ^ 1][tid] = i < S ? L[i] : INFINITY; Ds[cur ^ 1][tid] = i < S ? Dr[i] : 0.f; }
        }
        cpa_commit(); cpa_wait<1>(); __syncthreads();
        float p[8][4], ds[8][4];
        mm_rt(p, ka, Qs[cur], lane);    /* S^T: keys x queries */
        mm_rt(ds, va, Gs[cur], lane);   /* dP^T */
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const int c = 8 * j + 2 * t + (q & 1);
                const float pv = kv[q >> 1] != 0.f ? exp2f(p[j][q] * sl2 - Ls[cur][c]) : 0.f;
                p[j][q] = pv; ds[j][q] = pv * (ds[j][q] - Ds[cur][c]) * sks;
            }
        mm_pt(dv, p, Gs[cur], lane);    /* dV += P^T dO */
        mm_pt(dk, ds, Qs[cur], lane);   /* dK += dS^T Q */
        __syncthreads();
    }
    const float fk = scale * exp2f((float)(-e - ks)), fv = exp2f((float)-e);
#pragma unroll
    for (int r = 0; r < 2; r++) {
        const int j = j0 + warp * 16 + g + 8 * r;
        if (j >= S) continue;
        float *row = dqkn + ((size_t)seq * S + j) * 3 * C + h * HD;
        const float mk = lens && j >= lens[seq] ? 0.f : 1.f;   /* (a masked key: zero gradients) */
#pragma unroll
        for (int jd = 0; jd < 6; jd++) {
            *(float2 *)(row + C + 8 * jd + 2 * t) = make_float2(mk * dk[jd][2 * r] * fk, mk * dk[jd][2 * r + 1] * fk);
            *(float2 *)(row + 2 * C + 8 * jd + 2 * t) = make_float2(mk * dv[jd][2 * r] * fv, mk * dv[jd][2 * r + 1] * fv);
        }
    }
}

/* ---- dQ: grid (ceil(S / 64) query tiles, H, nseq); key tiles stream ---- */
__global__ void __launch_bounds__(128) k_fa_dq(const __half *__restrict__ Qh, const __half *__restrict__ Kh, const __half *__restrict__ Vh, const __half *__restrict__ dOh,
                                               const float *__restrict__ lse, const float *__restrict__ Dv, const float *__restrict__ amx, float *__restrict__ dqkn,
                                               int S, int H, int C, long long Tp, float sl2, float scale, const int *lens) {
    const int seq = blockIdx.z, h = blockIdx.y, q0 = blockIdx.x * FT, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    __shared__ __align__(16) __half Ks[2][FT * FS], Vs[2][FT * FS];
    const size_t base = (size_t)h * Tp + (size_t)seq * S;
    const __half *Q = Qh + base * HD, *K = Kh + base * HD, *V = Vh + base * HD, *G = dOh + base * HD;
    const float *L = lse + ((size_t)seq * H + h) * S, *Dr = Dv + ((size_t)seq * H + h) * S;
    const int e = fa_e(amx), ks = fa_ks(amx); const float sks = exp2f((float)ks);
    const int SK = lens ? lens[seq] : S, nt = (SK + FT - 1) / FT;
    stage(Ks[0], K, tid); stage(Vs[0], V, tid); cpa_commit();
    unsigned qa[3][4], ga[3][4]; const int i0 = q0 + warp * 16 + g; afrag(qa, Q, i0, t); afrag(ga, G, i0, t);
    const float li[2] = {i0 < S ? L[i0] : INFINITY, i0 + 8 < S ? L[i0 + 8] : INFINITY}, di[2] = {i0 < S ? Dr[i0] : 0.f, i0 + 8 < S ? Dr[i0 + 8] : 0.f};
    float dq[6][4];
#pragma unroll
    for (int j = 0; j < 6; j++)
#pragma unroll
        for (int q = 0; q < 4; q++) dq[j][q] = 0.f;
    for (int it = 0; it < nt; it++) {
        const int cur = it & 1;
        if (it + 1 < nt) { stage(Ks[cur ^ 1], K + (size_t)(it + 1) * FT * HD, tid); stage(Vs[cur ^ 1], V + (size_t)(it + 1) * FT * HD, tid); }
        cpa_commit(); cpa_wait<1>(); __syncthreads();
        float p[8][4], ds[8][4];
        mm_rt(p, qa, Ks[cur], lane);    /* S */
        mm_rt(ds, ga, Vs[cur], lane);   /* dP */
        const int kb = it * FT;
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const float pv = kb + 8 * j + 2 * t + (q & 1) < SK ? exp2f(p[j][q] * sl2 - li[q >> 1]) : 0.f;
                ds[j][q] = pv * (ds[j][q] - di[q >> 1]) * sks;
            }
        mm_pt(dq, ds, Ks[cur], lane);   /* dQ += dS K */
        __syncthreads();
    }
    const float fq = scale * exp2f((float)(-e - ks));
#pragma unroll
    for (int r = 0; r < 2; r++) {
        const int i = i0 + 8 * r;
        if (i >= S) continue;
        float *row = dqkn + ((size_t)seq * S + i) * 3 * C + h * HD;
#pragma unroll
        for (int jd = 0; jd < 6; jd++) *(float2 *)(row + 8 * jd + 2 * t) = make_float2(dq[jd][2 * r] * fq, dq[jd][2 * r + 1] * fq);
    }
}

static unsigned nbk(long long n, int t) { return (unsigned)((n + t - 1) / t); }
/* the backward prologue: amx[0] = max |dO|, D = dO . O (x 2^e) per (row, head), dO x 2^e -> fp16 head-major doh */
template <typename TO> static void fa_bprep(const TO *o, const float *dO, float *amx, void *doh, float *Dv, const at_geom *g, int C) {
    const long long T = (long long)g->nseq * g->S, Tp = T + FT; const int H = C / HD;
    cudaMemsetAsync(amx, 0, sizeof(float), 0);
    k_fa_amax<<<120, 256>>>(dO, T * C, amx);
    k_fa_bprep<TO><<<nbk(Tp * H, 128), 128>>>(o, dO, amx, (__half *)doh, Dv, T, g->S, C);
}
extern "C" void at_fa_bprep(const float *o, const float *dO, float *amx, void *doh, float *Dv, const at_geom *g, int C) { fa_bprep(o, dO, amx, doh, Dv, g, C); }
template <typename TI> static void fa_pack(const TI *qkv, const float *qg, const float *kg, void *qkvh, float *rq, float *rk, float *amx, const at_geom *g, int nreg, int C, int rope, float base) {
    const long long T = (long long)g->nseq * g->S, Tp = T + FT; const int H = C / HD;
    __half *Qh = (__half *)qkvh, *Kh = Qh + (size_t)H * Tp * HD, *Vh = Kh + (size_t)H * Tp * HD;
    cudaMemsetAsync(amx, 0, 4 * sizeof(float), 0);
    k_fa_pack<TI><<<nbk(Tp * H * 3, 128), 128>>>(qkv, qg, kg, Qh, Kh, Vh, rq, rk, amx, T, C, g->S, g->Hh, g->W, nreg, rope, log2f(base), g->pos);
}
extern "C" void at_fa_pack(const float *qkv, const float *qg, const float *kg, void *qkvh, float *rq, float *rk, float *amx, const at_geom *g, int nreg, int C, int rope, float base) {
    fa_pack(qkv, qg, kg, qkvh, rq, rk, amx, g, nreg, C, rope, base);
}
extern "C" void at_fa_pack_bf(const void *qkv, const float *qg, const float *kg, void *qkvh, float *rq, float *rk, float *amx, const at_geom *g, int nreg, int C, int rope, float base) {
    fa_pack((const __nv_bfloat16 *)qkv, qg, kg, qkvh, rq, rk, amx, g, nreg, C, rope, base);
}
template <typename TO> static void fa_fwd(const void *qkvh, TO *o, float *lse, const at_geom *g, int H, int C) {
    const long long Tp = (long long)g->nseq * g->S + FT;
    const __half *Qh = (const __half *)qkvh, *Kh = Qh + (size_t)H * Tp * HD, *Vh = Kh + (size_t)H * Tp * HD;
    dim3 gr(nbk(g->S, FT), H, g->nseq);
    k_fa_fwd<TO><<<gr, 128>>>(Qh, Kh, Vh, o, lse, g->S, H, C, Tp, 1.4426950408889634f / sqrtf((float)HD), g->lens);
}
extern "C" void at_fa_fwd(const void *qkvh, float *o, float *lse, const at_geom *g, int H, int C) { fa_fwd(qkvh, o, lse, g, H, C); }
extern "C" void at_fa_fwd_bf(const void *qkvh, void *o, float *lse, const at_geom *g, int H, int C) { fa_fwd(qkvh, (__nv_bfloat16 *)o, lse, g, H, C); }
template <typename TO> static void fa_bwd(const void *qkvh, void *doh, const TO *o, const float *lse, const float *dO, float *Dv, float *amx, float *dqkn, const at_geom *g, int H, int C) {
    const long long T = (long long)g->nseq * g->S, Tp = T + FT;
    const __half *Qh = (const __half *)qkvh, *Kh = Qh + (size_t)H * Tp * HD, *Vh = Kh + (size_t)H * Tp * HD; __half *Gh = (__half *)doh;
    fa_bprep(o, dO, amx, Gh, Dv, g, C);
    const float scale = 1.f / sqrtf((float)HD), sl2 = scale * 1.4426950408889634f;
    dim3 gr(nbk(g->S, FT), H, g->nseq);
    k_fa_dkv<<<gr, 128>>>(Qh, Kh, Vh, Gh, lse, Dv, amx, dqkn, g->S, H, C, Tp, sl2, scale, g->lens);
    k_fa_dq<<<gr, 128>>>(Qh, Kh, Vh, Gh, lse, Dv, amx, dqkn, g->S, H, C, Tp, sl2, scale, g->lens);
}
extern "C" void at_fa_bwd(const void *qkvh, void *doh, const float *o, const float *lse, const float *dO, float *Dv, float *amx, float *dqkn, const at_geom *g, int H, int C) {
    fa_bwd(qkvh, doh, o, lse, dO, Dv, amx, dqkn, g, H, C);
}
extern "C" void at_fa_bwd_bf(const void *qkvh, void *doh, const void *o, const float *lse, const float *dO, float *Dv, float *amx, float *dqkn, const at_geom *g, int H, int C) {
    fa_bwd(qkvh, doh, (const __nv_bfloat16 *)o, lse, dO, Dv, amx, dqkn, g, H, C);
}
