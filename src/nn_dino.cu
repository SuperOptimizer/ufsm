/* dino (src/dino.h): input, point gather, SupCon / DINO losses and their gradients for ufsm dino-train. Launched on the default stream
   of the current device (the nn layer's compute stream is the legacy default stream). */
#include "dino.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

static inline unsigned nb_of(long long n, int t) { long long b = (n + t - 1) / t; return (unsigned)(b > 2147483647LL ? 2147483647LL : b); }
__device__ __forceinline__ uint32_t hash32(uint32_t x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }
__device__ __forceinline__ float u01(uint32_t h) { return (h >> 8) * (1.f / 16777216.f) + 0.5f / 16777216.f; }

__global__ void k_input(const uint8_t *__restrict__ ct, float *__restrict__ x, int P, dn_geom g, float center, float scale, float gain, float bias,
                        float sigma, uint32_t seed, int mcube, float mfrac, uint32_t mseed, float mval) {
    const long long n = (long long)P * P * P;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const int o[3] = {(int)(i / ((long long)P * P)), (int)((i / P) % P), (int)(i % P)};
        int s[3];
        for (int d = 0; d < 3; d++) s[g.perm[d]] = g.flip[d] ? P - 1 - o[d] : o[d];
        float v = ((float)ct[((size_t)s[0] * P + s[1]) * P + s[2]] - center) / scale * gain + bias;
        if (sigma > 0.f) {   /* Box-Muller from two hashes */
            const uint32_t h1 = hash32((uint32_t)i * 2u + seed * 0x9E3779B9u), h2 = hash32((uint32_t)i * 2u + 1u + seed * 0x85EBCA6Bu);
            v += sigma * sqrtf(-2.f * logf(u01(h1))) * cospif(2.f * u01(h2));
        }
        if (mcube > 0 && mfrac > 0.f) {
            const int nc = (P + mcube - 1) / mcube; const uint32_t c = (uint32_t)(((o[0] / mcube) * nc + o[1] / mcube) * nc + o[2] / mcube);
            if (u01(hash32(c ^ (mseed * 0x27D4EB2Fu))) < mfrac) v = mval;
        }
        x[i] = v;
    }
}
extern "C" void dn_input(const uint8_t *ct, float *x, int P, dn_geom g, float center, float scale, float gain, float bias, float sigma, uint32_t seed,
                         int mcube, float mfrac, uint32_t mseed, float mval) {
    k_input<<<nb_of((long long)P * P * P, 256), 256>>>(ct, x, P, g, center, scale, gain, bias, sigma, seed, mcube, mfrac, mseed, mval);
}

/* one warp per point */
__global__ void k_gather(const void *__restrict__ lg, int h16, size_t p3, int D, const int *__restrict__ idx, int N, float *__restrict__ e, float *__restrict__ nrm) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (i >= N) return;
    const size_t v = (size_t)idx[i];
    float y[4]; float s = 0.f; int n = 0;
    for (int c = lane; c < D; c += 32, n++) { y[n] = h16 ? __half2float(((const __half *)lg)[(size_t)c * p3 + v]) : ((const float *)lg)[(size_t)c * p3 + v]; s += y[n] * y[n]; }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float r = sqrtf(s), inv = 1.f / fmaxf(r, 1e-6f);
    n = 0;
    for (int c = lane; c < D; c += 32, n++) e[(size_t)i * D + c] = y[n] * inv;
    if (lane == 0) nrm[i] = fmaxf(r, 1e-6f);
}
extern "C" void dn_gather(const void *logits, int h16, size_t p3, int D, const int *idx, int N, float *e, float *nrm) {
    if (D > 128) { fprintf(stderr, "dn_gather: D %d > 128\n", D); return; }
    if (N > 0) k_gather<<<nb_of((long long)N * 32, 256), 256>>>(logits, h16, p3, D, idx, N, e, nrm);
}

/* ---- SupCon: keys = the N anchors (student) + N2 teacher keys. Three passes over the N x (N + N2) similarities (D = 32): per-anchor
   statistics (lse, positives), the anchors' gradient sum_j g_ij k_j, the student keys' gradient sum_i g_ij a_i, with
   g_ij = (w_ij exp(s_ij / tau - lse_i) - [pos_ij] / np_i) / tau ---- */
__device__ __forceinline__ int wrap_dk(int ci, int cj) {   /* winding difference in turns from codes 1 + round(18 q) mod 252; classes >= 256 (e.g. 256 = air): same class 0, else far */
    if (ci >= 256 || cj >= 256) return ci == cj ? 0 : 99;
    int d = ((cj - ci) % 252 + 252) % 252; if (d >= 126) d -= 252;
    return (int)lrintf(d / 18.f);
}
__device__ __forceinline__ float blk_sum(float v, float *sh) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.f; for (int w = 0; w < (int)(blockDim.x >> 5); w++) t += sh[w];
    __syncthreads();
    return t;
}
__device__ __forceinline__ float blk_max(float v, float *sh) {
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = -INFINITY; for (int w = 0; w < (int)(blockDim.x >> 5); w++) t = fmaxf(t, sh[w]);
    __syncthreads();
    return t;
}
/* the block-wide sum of a per-thread SD-vector into out[SD] (overwritten) */
template <int SD> __device__ __forceinline__ void blk_vec_sum(float (&v)[SD], float *out, float (*sh)[SD]) {
#pragma unroll
    for (int c = 0; c < SD; c++) for (int o = 16; o; o >>= 1) v[c] += __shfl_xor_sync(0xffffffffu, v[c], o);
    if ((threadIdx.x & 31) == 0) for (int c = 0; c < SD; c++) sh[threadIdx.x >> 5][c] = v[c];
    __syncthreads();
    if (threadIdx.x < SD) { float t = 0.f; for (int w = 0; w < (int)(blockDim.x >> 5); w++) t += sh[w][threadIdx.x]; out[threadIdx.x] = t; }
    __syncthreads();
}
struct sc_args { const float *a, *t; const int *code, *xyz, *tcode, *txyz; int N, N2; float tau, hw, hr2; };
template <int SD> __device__ __forceinline__ const float *sc_key(const sc_args &A, int j) { return j < A.N ? A.a + (size_t)j * SD : A.t + (size_t)(j - A.N) * SD; }
__device__ __forceinline__ float sc_w(const sc_args &A, int i, int j, int dk) {
    if (abs(dk) != 1 || A.hw == 1.f) return 1.f;
    const int *pi = A.xyz + 3 * i, *pj = j < A.N ? A.xyz + 3 * j : A.txyz + 3 * (j - A.N);
    const float dx = pj[0] - pi[0], dy = pj[1] - pi[1], dz = pj[2] - pi[2];
    return dx * dx + dy * dy + dz * dz <= A.hr2 ? A.hw : 1.f;
}
__device__ __forceinline__ int sc_code(const sc_args &A, int j) { return j < A.N ? A.code[j] : A.tcode[j - A.N]; }
template <int SD> __global__ void __launch_bounds__(256) k_sc_stats(sc_args A, float *lse, float *npos, float *acc) {
    __shared__ float sh[8]; __shared__ float ai[SD];
    const int i = blockIdx.x, M = A.N + A.N2, ci = A.code[i];
    if (threadIdx.x < SD) ai[threadIdx.x] = A.a[(size_t)i * SD + threadIdx.x];
    __syncthreads();
    float mx = -INFINITY, np = 0.f;
    for (int j = threadIdx.x; j < M; j += blockDim.x) {
        if (j == i) continue;
        const float *k = sc_key<SD>(A, j); float s = 0.f;
#pragma unroll
        for (int c = 0; c < SD; c++) s += ai[c] * k[c];
        mx = fmaxf(mx, s / A.tau); np += wrap_dk(ci, sc_code(A, j)) == 0;
    }
    mx = blk_max(mx, sh); np = blk_sum(np, sh);
    float den = 0.f, psum = 0.f, pcos = 0.f, ncos = 0.f, nneg = 0.f;
    for (int j = threadIdx.x; j < M; j += blockDim.x) {
        if (j == i) continue;
        const float *k = sc_key<SD>(A, j); float s = 0.f;
#pragma unroll
        for (int c = 0; c < SD; c++) s += ai[c] * k[c];
        const int dk = wrap_dk(ci, sc_code(A, j));
        den += sc_w(A, i, j, dk) * expf(s / A.tau - mx);
        if (dk == 0) { psum += s / A.tau; pcos += s; } else { ncos += s; nneg += 1.f; }
    }
    den = blk_sum(den, sh); psum = blk_sum(psum, sh); pcos = blk_sum(pcos, sh); ncos = blk_sum(ncos, sh); nneg = blk_sum(nneg, sh);
    if (threadIdx.x == 0) {
        lse[i] = mx + logf(den); npos[i] = np;
        if (np > 0.f) { atomicAdd(&acc[0], lse[i] - psum / np); atomicAdd(&acc[1], 1.f); atomicAdd(&acc[2], pcos); atomicAdd(&acc[3], np); }
        atomicAdd(&acc[4], ncos); atomicAdd(&acc[5], nneg);
    }
}
template <int SD> __global__ void __launch_bounds__(256) k_sc_ganchor(sc_args A, const float *lse, const float *npos, float *ga) {
    __shared__ float sh[8][SD]; __shared__ float ai[SD], res[SD];
    const int i = blockIdx.x, M = A.N + A.N2, ci = A.code[i];
    if (npos[i] <= 0.f) { if (threadIdx.x < SD) ga[(size_t)i * SD + threadIdx.x] = 0.f; return; }
    if (threadIdx.x < SD) ai[threadIdx.x] = A.a[(size_t)i * SD + threadIdx.x];
    __syncthreads();
    float g[SD];
#pragma unroll
    for (int c = 0; c < SD; c++) g[c] = 0.f;
    for (int j = threadIdx.x; j < M; j += blockDim.x) {
        if (j == i) continue;
        const float *k = sc_key<SD>(A, j); float s = 0.f;
#pragma unroll
        for (int c = 0; c < SD; c++) s += ai[c] * k[c];
        const int dk = wrap_dk(ci, sc_code(A, j));
        const float gij = (sc_w(A, i, j, dk) * expf(s / A.tau - lse[i]) - (dk == 0 ? 1.f / npos[i] : 0.f)) / A.tau;
#pragma unroll
        for (int c = 0; c < SD; c++) g[c] += gij * k[c];
    }
    blk_vec_sum(g, res, sh);
    if (threadIdx.x < SD) ga[(size_t)i * SD + threadIdx.x] = res[threadIdx.x];
}
template <int SD> __global__ void __launch_bounds__(256) k_sc_gkey(sc_args A, const float *lse, const float *npos, float *ga) {   /* keys j < N (student anchors) */
    __shared__ float sh[8][SD]; __shared__ float kj[SD], res[SD];
    const int j = blockIdx.x, cj = A.code[j];
    if (threadIdx.x < SD) kj[threadIdx.x] = A.a[(size_t)j * SD + threadIdx.x];
    __syncthreads();
    float g[SD];
#pragma unroll
    for (int c = 0; c < SD; c++) g[c] = 0.f;
    for (int i = threadIdx.x; i < A.N; i += blockDim.x) {
        if (i == j || npos[i] <= 0.f) continue;
        const float *ai = A.a + (size_t)i * SD; float s = 0.f;
#pragma unroll
        for (int c = 0; c < SD; c++) s += ai[c] * kj[c];
        const int dk = wrap_dk(A.code[i], cj);
        const float gij = (sc_w(A, i, j, dk) * expf(s / A.tau - lse[i]) - (dk == 0 ? 1.f / npos[i] : 0.f)) / A.tau;
#pragma unroll
        for (int c = 0; c < SD; c++) g[c] += gij * ai[c];
    }
    blk_vec_sum(g, res, sh);
    if (threadIdx.x < SD) ga[(size_t)j * SD + threadIdx.x] += res[threadIdx.x];
}
__global__ void k_sc_fin(float *ga, int n, const float *acc, float *out) {
    const float na = fmaxf(acc[1], 1.f);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) ga[i] /= na;
    if (blockIdx.x == 0 && threadIdx.x == 0) { out[0] = acc[0] / na; out[1] = acc[2] / fmaxf(acc[3], 1.f); out[2] = acc[4] / fmaxf(acc[5], 1.f); out[3] = acc[1]; }
}
template <int SD> static void supcon_t(const float *a, const int *code, const int *xyz, int N, const float *t, const int *tcode, const int *txyz, int N2,
                                       float tau, float hw, float hr, float *ga, float *out, float *scratch) {
    /* scratch: lse [N], npos [N], acc [8] */
    float *lse = scratch, *npos = lse + N, *acc = npos + N;
    cudaMemsetAsync(acc, 0, 8 * 4);
    if (N <= 0) { cudaMemsetAsync(out, 0, 16); return; }
    sc_args A = {a, t, code, xyz, tcode, txyz, N, N2, tau, hw, hr * hr};
    k_sc_stats<SD><<<N, 256>>>(A, lse, npos, acc);
    k_sc_ganchor<SD><<<N, 256>>>(A, lse, npos, ga);
    k_sc_gkey<SD><<<N, 256>>>(A, lse, npos, ga);
    k_sc_fin<<<nb_of((long long)N * SD, 256), 256>>>(ga, N * SD, acc, out);
}
extern "C" void dn_supcon(const float *a, const int *code, const int *xyz, int N, const float *t, const int *tcode, const int *txyz, int N2, int D,
                          float tau, float hw, float hr, float *ga, float *out, float *scratch) {
    if (D == 16) supcon_t<16>(a, code, xyz, N, t, tcode, txyz, N2, tau, hw, hr, ga, out, scratch);
    else if (D == 32) supcon_t<32>(a, code, xyz, N, t, tcode, txyz, N2, tau, hw, hr, ga, out, scratch);
    else fprintf(stderr, "dn_supcon: D %d (16 or 32)\n", D);
}

/* ---- DINO prototypes: logits matrices in scratch; the gradients as two small matrix products ---- */
__global__ void k_normrows(const float *W, float *C, float *Wn, int K, int D) {   /* C = W / |W| per row, Wn the norms */
    const int k = blockIdx.x * blockDim.x + threadIdx.x; if (k >= K) return;
    float s = 0.f; for (int c = 0; c < D; c++) s += W[(size_t)k * D + c] * W[(size_t)k * D + c];
    const float r = fmaxf(sqrtf(s), 1e-6f); Wn[k] = r;
    for (int c = 0; c < D; c++) C[(size_t)k * D + c] = W[(size_t)k * D + c] / r;
}
/* one block per point: G[i][k] = (softmax_s - softmax_t) / ts / N, teacher statistics */
template <int SD> __global__ void __launch_bounds__(256) k_dino_rows(const float *s, const float *t, int N, const float *C, const float *Ct, const float *center, int K,
                                                   float ts, float tt, float *G, float *tmean, float *pbar, float *acc) {
    __shared__ float sh[8]; __shared__ float si[SD], ti[SD];
    const int i = blockIdx.x; float *zs = G + (size_t)i * K;   /* zs, then G, in place */
    extern __shared__ float pt[];   /* [K] teacher logits */
    if (threadIdx.x < SD) { si[threadIdx.x] = s[(size_t)i * SD + threadIdx.x]; ti[threadIdx.x] = t[(size_t)i * SD + threadIdx.x]; }
    __syncthreads();
    float ms = -INFINITY, mt = -INFINITY;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float a = 0.f, b = 0.f;
#pragma unroll
        for (int c = 0; c < SD; c++) { a += si[c] * C[(size_t)k * SD + c]; b += ti[c] * Ct[(size_t)k * SD + c]; }
        atomicAdd(&tmean[k], b / N);
        zs[k] = a / ts; pt[k] = (b - center[k]) / tt;
        ms = fmaxf(ms, zs[k]); mt = fmaxf(mt, pt[k]);
    }
    ms = blk_max(ms, sh); mt = blk_max(mt, sh);
    float ss = 0.f, st = 0.f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) { zs[k] = expf(zs[k] - ms); pt[k] = expf(pt[k] - mt); ss += zs[k]; st += pt[k]; }
    ss = blk_sum(ss, sh); st = blk_sum(st, sh);
    float loss = 0.f, ent = 0.f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        const float ps = zs[k] / ss, p = pt[k] / st;
        loss -= p * logf(fmaxf(ps, 1e-30f)); ent -= p * logf(fmaxf(p, 1e-30f));
        atomicAdd(&pbar[k], p / N);
        zs[k] = (ps - p) / ts / N;
    }
    loss = blk_sum(loss, sh); ent = blk_sum(ent, sh);
    if (threadIdx.x == 0) { atomicAdd(&acc[0], loss / N); atomicAdd(&acc[2], ent / N); }
}
template <int SD> __global__ void __launch_bounds__(256) k_dino_gs(const float *G, const float *C, int N, int K, float *gs) {   /* gs[i] = sum_k G[i][k] C[k] */
    __shared__ float sh[8][SD]; __shared__ float res[SD];
    const int i = blockIdx.x; float g[SD];
#pragma unroll
    for (int c = 0; c < SD; c++) g[c] = 0.f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) { const float w = G[(size_t)i * K + k];
#pragma unroll
        for (int c = 0; c < SD; c++) g[c] += w * C[(size_t)k * SD + c]; }
    blk_vec_sum(g, res, sh);
    if (threadIdx.x < SD) gs[(size_t)i * SD + threadIdx.x] = res[threadIdx.x];
}
template <int SD> __global__ void __launch_bounds__(256) k_dino_gc(const float *G, const float *s, int N, int K, float *gC) {   /* gC[k] = sum_i G[i][k] s[i] */
    __shared__ float sh[8][SD]; __shared__ float res[SD];
    const int k = blockIdx.x; float g[SD];
#pragma unroll
    for (int c = 0; c < SD; c++) g[c] = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) { const float w = G[(size_t)i * K + k];
#pragma unroll
        for (int c = 0; c < SD; c++) g[c] += w * s[(size_t)i * SD + c]; }
    blk_vec_sum(g, res, sh);
    if (threadIdx.x < SD) gC[(size_t)k * SD + threadIdx.x] = res[threadIdx.x];
}
template <int SD> __global__ void k_dino_fin(const float *gC, const float *C, const float *Wn, float *gW, int K, const float *pbar, float *acc) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < K) {   /* back through c = W / |W| */
        float dot = 0.f; for (int c = 0; c < SD; c++) dot += gC[(size_t)k * SD + c] * C[(size_t)k * SD + c];
        for (int c = 0; c < SD; c++) gW[(size_t)k * SD + c] = (gC[(size_t)k * SD + c] - C[(size_t)k * SD + c] * dot) / Wn[k];
        if (pbar[k] > 0.f) atomicAdd(&acc[1], -pbar[k] * logf(pbar[k]));
    }
}
__global__ void k_copy3(const float *acc, float *out) { out[0] = acc[0]; out[1] = acc[1]; out[2] = acc[2]; }
extern "C" size_t dn_dino_scratch(int N, int K) { return ((size_t)N * K + (size_t)4 * K * 32 + 3 * (size_t)K + 8) * 4; }
template <int SD> static void dino_t(const float *s, const float *t, int N, const float *W, const float *Wt, const float *center, int K, float ts, float tt,
                                     float *gs, float *gW, float *tmean, float *out, float *scratch) {
    float *G = scratch, *C = G + (size_t)N * K, *Ct = C + (size_t)K * SD, *gC = Ct + (size_t)K * SD, *Wn = gC + (size_t)K * SD, *Wtn = Wn + K, *pbar = Wtn + K, *acc = pbar + K;
    k_normrows<<<nb_of(K, 128), 128>>>(W, C, Wn, K, SD); k_normrows<<<nb_of(K, 128), 128>>>(Wt, Ct, Wtn, K, SD);
    cudaMemsetAsync(pbar, 0, (size_t)K * 4); cudaMemsetAsync(acc, 0, 16); cudaMemsetAsync(tmean, 0, (size_t)K * 4);
    if (N > 0) {
        k_dino_rows<SD><<<N, 256, (size_t)K * 4>>>(s, t, N, C, Ct, center, K, ts, tt, G, tmean, pbar, acc);
        k_dino_gs<SD><<<N, 256>>>(G, C, N, K, gs);
        k_dino_gc<SD><<<K, 256>>>(G, s, N, K, gC);
    } else cudaMemsetAsync(gC, 0, (size_t)K * SD * 4);
    k_dino_fin<SD><<<nb_of(K, 128), 128>>>(gC, C, Wn, gW, K, pbar, acc);
    k_copy3<<<1, 1>>>(acc, out);
}
extern "C" void dn_dino(const float *s, const float *t, int N, const float *W, const float *Wt, const float *center, int K, int D, float ts, float tt,
                        float *gs, float *gW, float *tmean, float *out, float *scratch) {
    if (D == 16) dino_t<16>(s, t, N, W, Wt, center, K, ts, tt, gs, gW, tmean, out, scratch);
    else if (D == 32) dino_t<32>(s, t, N, W, Wt, center, K, ts, tt, gs, gW, tmean, out, scratch);
    else fprintf(stderr, "dn_dino: D %d (16 or 32)\n", D);
}

__global__ void k_scatter(const float *e, const float *ge, const float *nrm, const int *idx, int N, int D, size_t p3, float w, float *gl) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (i >= N) return;
    float d = 0.f; for (int c = lane; c < D; c += 32) d += e[(size_t)i * D + c] * ge[(size_t)i * D + c];
    for (int o = 16; o; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
    const float inv = w / nrm[i];
    for (int c = lane; c < D; c += 32) atomicAdd(&gl[(size_t)c * p3 + idx[i]], (ge[(size_t)i * D + c] - e[(size_t)i * D + c] * d) * inv);
}
extern "C" void dn_scatter_grad(const float *e, const float *ge, const float *nrm, const int *idx, int N, int D, size_t p3, float w, float *gl) {
    if (N > 0) k_scatter<<<nb_of((long long)N * 32, 256), 256>>>(e, ge, nrm, idx, N, D, p3, w, gl);
}
__global__ void k_adam(float *p, const float *g, float *m, float *v, size_t n, float lr, float b1, float b2, float eps, float wd, float c1, float c2) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        m[i] = b1 * m[i] + (1 - b1) * g[i]; v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
        p[i] -= lr * ((m[i] / c1) / (sqrtf(v[i] / c2) + eps) + wd * p[i]);
    }
}
extern "C" void dn_adam(float *p, const float *g, float *m, float *v, size_t n, float lr, float b1, float b2, float eps, float wd, int step) {
    k_adam<<<nb_of((long long)n, 256), 256>>>(p, g, m, v, n, lr, b1, b2, eps, wd, 1.f - powf(b1, (float)step), 1.f - powf(b2, (float)step));
}
__global__ void k_axpby(float *y, const float *x, size_t n, float a, float b) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) y[i] = a * y[i] + b * x[i];
}
extern "C" void dn_ema(float *pt, const float *p, size_t n, float decay) { k_axpby<<<nb_of((long long)n, 256), 256>>>(pt, p, n, decay, 1.f - decay); }
extern "C" void dn_axpby(float *y, const float *x, size_t n, float a, float b) { k_axpby<<<nb_of((long long)n, 256), 256>>>(y, x, n, a, b); }
__global__ void k_std(const float *e, int N, int D, float *out) {
    __shared__ float sh[8];
    float acc = 0.f;
    for (int c = 0; c < D; c++) {
        float s = 0.f, s2 = 0.f;
        for (int i = threadIdx.x; i < N; i += blockDim.x) { const float v = e[(size_t)i * D + c]; s += v; s2 += v * v; }
        s = blk_sum(s, sh); s2 = blk_sum(s2, sh);
        acc += sqrtf(fmaxf(s2 / N - (s / N) * (s / N), 0.f));
    }
    if (threadIdx.x == 0) out[0] = acc / D;
}
extern "C" void dn_emb_std(const float *e, int N, int D, float *out) { if (N > 0) k_std<<<1, 256>>>(e, N, D, out); }

__global__ void k_tile_out(const void *lg, int h16, int P, int D, const uint8_t *ct, int z0, int z1, int y0, int y1, int x0, int x1,
                           int8_t *out, int onz, int ony, int onx, int ooz, int ooy, int oox) {
    const long long n = (long long)(z1 - z0) * (y1 - y0) * (x1 - x0);
    const size_t p3 = (size_t)P * P * P;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const int x = x0 + (int)(i % (x1 - x0)), y = y0 + (int)((i / (x1 - x0)) % (y1 - y0)), z = z0 + (int)(i / ((long long)(x1 - x0) * (y1 - y0)));
        const int oz = ooz + z - z0, oy = ooy + y - y0, ox = oox + x - x0;
        if (oz < 0 || oy < 0 || ox < 0 || oz >= onz || oy >= ony || ox >= onx) continue;
        const size_t v = ((size_t)z * P + y) * P + x;
        int8_t *o = out + (((size_t)oz * ony + oy) * onx + ox) * D;
        if (!ct[v]) { for (int c = 0; c < D; c++) o[c] = 0; continue; }
        float s = 0.f;
        for (int c = 0; c < D; c++) { const float a = h16 ? __half2float(((const __half *)lg)[c * p3 + v]) : ((const float *)lg)[c * p3 + v]; s += a * a; }
        const float inv = 127.f / fmaxf(sqrtf(s), 1e-6f);
        for (int c = 0; c < D; c++) { const float a = h16 ? __half2float(((const __half *)lg)[c * p3 + v]) : ((const float *)lg)[c * p3 + v]; o[c] = (int8_t)fminf(fmaxf(rintf(a * inv), -127.f), 127.f); }
    }
}
extern "C" void dn_tile_out(const void *logits, int h16, int P, int D, const uint8_t *ct, int z0, int z1, int y0, int y1, int x0, int x1,
                            int8_t *out, int onz, int ony, int onx, int ooz, int ooy, int oox) {
    const long long n = (long long)(z1 - z0) * (y1 - y0) * (x1 - x0);
    if (n > 0) k_tile_out<<<nb_of(n, 256), 256>>>(logits, h16, P, D, ct, z0, z1, y0, y1, x0, x1, out, onz, ony, onx, ooz, ooy, oox);
}
/* variance hinge (VICReg): L = lambda / D sum_c relu(gamma - std_c) over N points of e [N][D]; adds d L / d e into ge; loss -> out[0] */
__global__ void k_varreg(const float *e, int N, int D, float gamma, float lambda, float *ge, float *out) {
    __shared__ float sh[8];
    float L = 0.f;
    for (int c = 0; c < D; c++) {
        float s = 0.f, s2 = 0.f;
        for (int i = threadIdx.x; i < N; i += blockDim.x) { const float v = e[(size_t)i * D + c]; s += v; s2 += v * v; }
        s = blk_sum(s, sh); s2 = blk_sum(s2, sh);
        const float mean = s / N, var = fmaxf(s2 / N - mean * mean, 0.f), sd = sqrtf(var + 1e-6f);
        if (sd < gamma) {
            L += (gamma - sd) * lambda / D;
            const float g = -lambda / D / (N * sd);
            for (int i = threadIdx.x; i < N; i += blockDim.x) ge[(size_t)i * D + c] += g * (e[(size_t)i * D + c] - mean);
        }
    }
    if (threadIdx.x == 0) out[0] = L;
}
extern "C" void dn_varreg(const float *e, int N, int D, float gamma, float lambda, float *ge, float *out) { if (N > 1) k_varreg<<<1, 256>>>(e, N, D, gamma, lambda, ge, out); }
/* the axis-distance input channel: x[o] = rs * |(sy, sx) - axis(sz)| with the window's axis (y, x) per source plane ayx [P][2] (window coords) */
__global__ void k_input_axis(float *x, int P, dn_geom g, const float *ayx, float rs) {
    const long long n = (long long)P * P * P;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const int o[3] = {(int)(i / ((long long)P * P)), (int)((i / P) % P), (int)(i % P)};
        int s[3];
        for (int d = 0; d < 3; d++) s[g.perm[d]] = g.flip[d] ? P - 1 - o[d] : o[d];
        const float dy = s[1] - ayx[2 * s[0]], dx = s[2] - ayx[2 * s[0] + 1];
        x[i] = rs * sqrtf(dy * dy + dx * dx);
    }
}
extern "C" void dn_input_axis(float *x, int P, dn_geom g, const float *ayx, float rs) {
    k_input_axis<<<nb_of((long long)P * P * P, 256), 256>>>(x, P, g, ayx, rs);
}
