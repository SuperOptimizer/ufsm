/* dinor kernels (src/dinor.h): multi-crop views, the DINO head, teacher targets (Sinkhorn-Knopp / centred softmax), the multi-crop
   cross-entropy, KoLeo, pooling, a small AdamW / EMA. fp32, default stream. */
#include "dinor.h"
#include "attn.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned nbk(long long n, int t) { return (unsigned)((n + t - 1) / t); }
__device__ __forceinline__ uint32_t dr_hash(uint32_t x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }
__device__ __forceinline__ float dr_u01(uint32_t h) { return ((h >> 8) + 0.5f) * (1.f / 16777216.f); }

/* ---- views ---- */
__global__ void k_crops(const uint8_t *__restrict__ src, int Ps, const dr_xf *__restrict__ xf, const dr_int *__restrict__ it, int ncrop, int n, float *__restrict__ x) {
    const long long n3 = (long long)n * n * n, tot = n3 * ncrop;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < tot; i += (long long)gridDim.x * blockDim.x) {
        const int c = (int)(i / n3); const long long j = i % n3;
        const dr_xf t = xf[c]; const dr_int a = it[c];
        const int v[3] = {(int)(j / ((long long)n * n)), (int)((j / n) % n), (int)(j % n)};
        float s[3];
        for (int d = 0; d < 3; d++) { const float o = t.flip[d] ? (float)(n - 1 - v[d]) : (float)v[d]; s[t.perm[d]] = (float)t.off[t.perm[d]] + (o + 0.5f) * (float)t.m / (float)n - 0.5f; }
        int i0[3]; float f[3];
        for (int d = 0; d < 3; d++) { float q = fminf(fmaxf(s[d], 0.f), (float)(Ps - 1)); i0[d] = min((int)floorf(q), Ps - 2 < 0 ? 0 : Ps - 2); f[d] = q - (float)i0[d]; }
        const uint8_t *S = src + (size_t)t.src * Ps * Ps * Ps;
        float val = 0.f;
        for (int dz = 0; dz < 2; dz++) for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
            const float w = (dz ? f[0] : 1.f - f[0]) * (dy ? f[1] : 1.f - f[1]) * (dx ? f[2] : 1.f - f[2]);
            if (w != 0.f) val += w * (float)S[((size_t)min(i0[0] + dz, Ps - 1) * Ps + min(i0[1] + dy, Ps - 1)) * Ps + min(i0[2] + dx, Ps - 1)];
        }
        if (a.gamma != 1.f && a.hi > a.lo) val = a.lo + (a.hi - a.lo) * powf(fminf(fmaxf((val - a.lo) / (a.hi - a.lo), 0.f), 1.f), a.gamma);
        float y = (val - a.center) / a.scale;
        if (a.contrast != 1.f) y = a.mean + a.contrast * (y - a.mean);
        y += a.bias;
        if (a.invert) y = -y;
        if (a.sigma > 0.f) {
            const uint32_t h1 = dr_hash((uint32_t)j * 2u + a.seed * 0x9E3779B9u), h2 = dr_hash((uint32_t)j * 2u + 1u + a.seed * 0x85EBCA6Bu);
            y += a.sigma * sqrtf(-2.f * logf(dr_u01(h1))) * cospif(2.f * dr_u01(h2));
        }
        x[i] = y;
    }
}
extern "C" void dr_crops(const uint8_t *src, int Ps, const dr_xf *xf, const dr_int *it, int ncrop, int n, float *x) {
    const long long tot = (long long)n * n * n * ncrop;
    k_crops<<<nbk(tot, 256), 256>>>(src, Ps, xf, it, ncrop, n, x);
}

/* ---- DINO head ---- */
extern "C" void dr_head_layout(dr_head *h, int Cin, int H, int B, int K) {
    size_t o = 0; h->Cin = Cin; h->H = H; h->B = B; h->K = K;
    h->w1 = o; o += (size_t)H * Cin; h->b1 = o; o += H;
    h->w2 = o; o += (size_t)H * H; h->b2 = o; o += H;
    h->w3 = o; o += (size_t)B * H; h->b3 = o; o += B;
    h->v = o; o += (size_t)K * B; h->g = o; o += K;
    h->n = o;
}
static size_t al(size_t n) { return (n + 63) & ~(size_t)63; }
extern "C" size_t dr_hws_bytes(const dr_head *h, int R) {
    const size_t r = R, H = h->H, B = h->B, K = h->K;
    return (al(r * H) * 6 + al(r * B) * 3 + al(r) + al(K * B) * 2 + al(K) * 2) * 4;
}
extern "C" void dr_hws_carve(dr_hws *w, const dr_head *h, int R, void *base) {
    const size_t r = R, H = h->H, B = h->B, K = h->K; float *p = (float *)base;
#define TK(f, n) do { w->f = p; p += al(n); } while (0)
    TK(a1, r * H); TK(h1, r * H); TK(a2, r * H); TK(h2, r * H); TK(dh, r * H); TK(dh2, r * H);
    TK(z, r * B); TK(zn, r * B); TK(dzn, r * B); TK(zr, r);
    TK(vn, K * B); TK(dvn, K * B); TK(vr, K); TK(dg, K);
#undef TK
}
__device__ __forceinline__ float gelu_e(float v) { return 0.5f * v * (1.f + erff(v * 0.70710678118654752f)); }
__device__ __forceinline__ float gelu_d(float v) { return 0.5f * (1.f + erff(v * 0.70710678118654752f)) + v * 0.3989422804014327f * expf(-0.5f * v * v); }
__global__ void k_bias_gelu(float *a, const float *b, float *h, long long R, int N) {   /* a += b; h = gelu(a) */
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= R * N) return;
    const float v = a[i] + b[i % N]; a[i] = v; h[i] = gelu_e(v);
}
__global__ void k_bias(float *a, const float *b, long long R, int N) { const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i < R * N) a[i] += b[i % N]; }
__global__ void k_gelu_bwd(float *d, const float *a, long long n) { const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i < n) d[i] *= gelu_d(a[i]); }
/* row L2 normalisation: y = x / max(|x|, eps), r = |x| (one warp per row) */
__global__ void k_rownorm(const float *x, float *y, float *r, int R, int N, float eps, const float *gs) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= R) return;
    float s = 0.f; for (int c = lane; c < N; c += 32) { const float v = x[(size_t)row * N + c]; s += v * v; }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float n = sqrtf(s), inv = 1.f / fmaxf(n, eps), g = gs ? gs[row] : 1.f;
    for (int c = lane; c < N; c += 32) y[(size_t)row * N + c] = x[(size_t)row * N + c] * inv * g;
    if (!lane) r[row] = n;
}
/* back through y = x / max(|x|, eps): dx = (dy - y (y . dy)) / |x| (|x| > eps), dy / eps otherwise; y the unit rows. acc: += */
__global__ void k_rownorm_bwd(const float *y, const float *r, const float *dy, float *dx, int R, int N, float eps, int acc) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= R) return;
    const float n = r[row];
    float d = 0.f; if (n > eps) for (int c = lane; c < N; c += 32) d += y[(size_t)row * N + c] * dy[(size_t)row * N + c];
    for (int o = 16; o; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
    const float inv = 1.f / fmaxf(n, eps);
    for (int c = lane; c < N; c += 32) { const size_t k = (size_t)row * N + c; const float v = (dy[k] - (n > eps ? y[k] * d : 0.f)) * inv; if (acc) dx[k] += v; else dx[k] = v; }
}
__global__ void k_colsum2(const float *dy, long long T, int N, float *db) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x; const long long r0 = (long long)blockIdx.y * 256;
    if (c >= N) return;
    float s = 0.f; const long long r1 = r0 + 256 < T ? r0 + 256 : T;
    for (long long r = r0; r < r1; r++) s += dy[r * N + c];
    atomicAdd(db + c, s);
}
static void colsum2(const float *dy, long long T, int N, float *db) { dim3 g(nbk(N, 256), nbk(T, 256)); k_colsum2<<<g, 256>>>(dy, T, N, db); }
/* weight-norm scales' gradient: dg_k = vn_k . dW_k; then dW_k -> d v_k = g_k / |v_k| (dW_k - vn_k (vn_k . dW_k)), written over dW */
__global__ void k_wn_bwd(const float *vn, const float *vr, const float *g, float *dW, float *dgk, int K, int B) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= K) return;
    float d = 0.f; for (int c = lane; c < B; c += 32) d += vn[(size_t)row * B + c] * dW[(size_t)row * B + c];
    for (int o = 16; o; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
    const float s = g[row] / fmaxf(vr[row], 1e-12f);
    for (int c = lane; c < B; c += 32) { const size_t k = (size_t)row * B + c; dW[k] = s * (dW[k] - vn[k] * d); }
    if (!lane) dgk[row] += d;
}
__global__ void k_add(float *y, const float *x, long long n) { const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i < n) y[i] += x[i]; }
#define CK(x) do { if (x) return -1; } while (0)
extern "C" int dr_head_fwd(const dr_head *h, const float *P, const float *x, int R, float *logits, dr_hws *w) {
    const int H = h->H, B = h->B, K = h->K;
    CK(at_gemm(0, 1, R, H, h->Cin, x, P + h->w1, w->a1, 0.f)); k_bias_gelu<<<nbk((long long)R * H, 256), 256>>>(w->a1, P + h->b1, w->h1, R, H);
    CK(at_gemm(0, 1, R, H, H, w->h1, P + h->w2, w->a2, 0.f)); k_bias_gelu<<<nbk((long long)R * H, 256), 256>>>(w->a2, P + h->b2, w->h2, R, H);
    CK(at_gemm(0, 1, R, B, H, w->h2, P + h->w3, w->z, 0.f)); k_bias<<<nbk((long long)R * B, 256), 256>>>(w->z, P + h->b3, R, B);
    k_rownorm<<<nbk((long long)R * 32, 256), 256>>>(w->z, w->zn, w->zr, R, B, 1e-12f, nullptr);
    k_rownorm<<<nbk((long long)K * 32, 256), 256>>>(P + h->v, w->vn, w->vr, K, B, 1e-12f, nullptr);   /* unit prototypes */
    /* logits = zn (g vn)^T: scale the unit rows by g into dvn (scratch) */
    k_rownorm<<<nbk((long long)K * 32, 256), 256>>>(P + h->v, w->dvn, w->dg, K, B, 1e-12f, P + h->g);
    CK(at_gemm(0, 1, R, K, B, w->zn, w->dvn, logits, 0.f));
    return 0;
}
extern "C" int dr_head_bwd(const dr_head *h, const float *P, float *G, const float *x, const float *dl, int R, float *dx, int last, dr_hws *w) {
    const int H = h->H, B = h->B, K = h->K, Cin = h->Cin;
    k_rownorm<<<nbk((long long)K * 32, 256), 256>>>(P + h->v, w->vn, w->vr, K, B, 1e-12f, nullptr);
    k_rownorm<<<nbk((long long)K * 32, 256), 256>>>(P + h->v, w->dvn, w->dg, K, B, 1e-12f, P + h->g);   /* W = g vn */
    CK(at_gemm(0, 0, R, B, K, dl, w->dvn, w->dzn, 0.f));                 /* d zn = dl W */
    if (last) {
        CK(at_gemm(1, 0, K, B, R, dl, w->zn, w->dvn, 0.f));              /* d W = dl^T zn (over W's scratch) */
        k_wn_bwd<<<nbk((long long)K * 32, 256), 256>>>(w->vn, w->vr, P + h->g, w->dvn, G + h->g, K, B);
        k_add<<<nbk((long long)K * B, 256), 256>>>(G + h->v, w->dvn, (long long)K * B);
    }
    k_rownorm_bwd<<<nbk((long long)R * 32, 256), 256>>>(w->zn, w->zr, w->dzn, w->dzn, R, B, 1e-12f, 0);   /* d z (in place) */
    CK(at_gemm(1, 0, B, H, R, w->dzn, w->h2, G + h->w3, 1.f)); colsum2(w->dzn, R, B, G + h->b3);
    CK(at_gemm(0, 0, R, H, B, w->dzn, P + h->w3, w->dh2, 0.f));
    k_gelu_bwd<<<nbk((long long)R * H, 256), 256>>>(w->dh2, w->a2, (long long)R * H);
    CK(at_gemm(1, 0, H, H, R, w->dh2, w->h1, G + h->w2, 1.f)); colsum2(w->dh2, R, H, G + h->b2);
    CK(at_gemm(0, 0, R, H, H, w->dh2, P + h->w2, w->dh, 0.f));
    k_gelu_bwd<<<nbk((long long)R * H, 256), 256>>>(w->dh, w->a1, (long long)R * H);
    CK(at_gemm(1, 0, H, Cin, R, w->dh, x, G + h->w1, 1.f)); colsum2(w->dh, R, H, G + h->b1);
    if (dx) CK(at_gemm(0, 0, R, Cin, H, w->dh, P + h->w1, dx, 0.f));
    return 0;
}

/* ---- teacher targets ---- */
extern "C" size_t dr_sk_scratch(int R, int K) { return (al((size_t)K) + al((size_t)R) + 64) * 4; }
__global__ void k_tq(const float *l, float *q, long long n, float tt) {   /* q = exp((l - 1) / tt): cosine-bounded logits (|l| <= g) */
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i < n) q[i] = expf((l[i] - 1.f) / tt);
}
__global__ void k_maxrow(const float *l, long long n, float *mx) {   /* one block: the global max into mx[0] */
    __shared__ float sh[256]; float m = -INFINITY;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, l[i]);
    sh[threadIdx.x] = m; __syncthreads();
    for (int k = blockDim.x / 2; k; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + k]); __syncthreads(); }
    if (!threadIdx.x) mx[0] = sh[0];
}
__global__ void k_tq_m(const float *l, float *q, long long n, float tt, const float *mx) { const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i < n) q[i] = expf((l[i] - mx[0]) / tt); }
__global__ void k_sum_all(const float *q, long long n, float *out) {   /* one block */
    __shared__ double sh[256]; double s = 0;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) s += q[i];
    sh[threadIdx.x] = s; __syncthreads();
    for (int k = blockDim.x / 2; k; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k]; __syncthreads(); }
    if (!threadIdx.x) out[0] = (float)sh[0];
}
__global__ void k_scale_all(float *q, long long n, const float *s) { const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i < n) q[i] /= s[0]; }
/* per prototype k: sum over the R rows -> q[r][k] /= (sum K) */
__global__ void k_sk_proto(float *q, int R, int K) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x; if (k >= K) return;
    float s = 0.f; for (int r = 0; r < R; r++) s += q[(size_t)r * K + k];
    const float d = s * (float)K; for (int r = 0; r < R; r++) q[(size_t)r * K + k] = s > 0.f ? q[(size_t)r * K + k] / d : 0.f;
}
/* per row r (one block): q[r][.] /= (sum R) * fin (fin: the last iteration multiplies by R back: rows sum to 1) */
__global__ void k_sk_row(float *q, int R, int K, int last) {
    __shared__ float sh[256]; const int r = blockIdx.x; float s = 0.f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) s += q[(size_t)r * K + k];
    sh[threadIdx.x] = s; __syncthreads();
    for (int k = blockDim.x / 2; k; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k]; __syncthreads(); }
    const float d = sh[0] * (float)R, f = last ? (float)R : 1.f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) q[(size_t)r * K + k] = q[(size_t)r * K + k] / d * f;
}
__global__ void k_softmax_c(const float *l, const float *c, float *q, int K, float tt) {   /* one block per row */
    __shared__ float sh[256]; const int r = blockIdx.x; float m = -INFINITY;
    for (int k = threadIdx.x; k < K; k += blockDim.x) m = fmaxf(m, (l[(size_t)r * K + k] - (c ? c[k] : 0.f)) / tt);
    sh[threadIdx.x] = m; __syncthreads();
    for (int k = blockDim.x / 2; k; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + k]); __syncthreads(); }
    m = sh[0]; __syncthreads();
    float s = 0.f; for (int k = threadIdx.x; k < K; k += blockDim.x) s += expf((l[(size_t)r * K + k] - (c ? c[k] : 0.f)) / tt - m);
    sh[threadIdx.x] = s; __syncthreads();
    for (int k = blockDim.x / 2; k; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k]; __syncthreads(); }
    const float inv = 1.f / sh[0];
    for (int k = threadIdx.x; k < K; k += blockDim.x) q[(size_t)r * K + k] = expf((l[(size_t)r * K + k] - (c ? c[k] : 0.f)) / tt - m) * inv;
}
extern "C" void dr_teacher_probs(const float *l, int R, int K, float tt, int iters, const float *center, float *q, void *scratch) {
    const long long n = (long long)R * K;
    if (iters <= 0) { k_softmax_c<<<R, 256>>>(l, center, q, K, tt); return; }
    float *s = (float *)scratch;
    k_maxrow<<<1, 256>>>(l, n, s + 1);   /* (the global max: exp stays finite for any logit scale; the normalisation removes it) */
    k_tq_m<<<nbk(n, 256), 256>>>(l, q, n, tt, s + 1);
    k_sum_all<<<1, 256>>>(q, n, s); k_scale_all<<<nbk(n, 256), 256>>>(q, n, s);
    for (int it = 0; it < iters; it++) {
        k_sk_proto<<<nbk(K, 256), 256>>>(q, R, K);
        k_sk_row<<<R, 256>>>(q, R, K, it == iters - 1);
    }
}
__global__ void k_center(float *c, const float *l, int R, int K, float m) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x; if (k >= K) return;
    float s = 0.f; for (int r = 0; r < R; r++) s += l[(size_t)r * K + k];
    c[k] = c[k] * m + s / R * (1.f - m);
}
extern "C" void dr_center_update(float *center, const float *logits, int R, int K, float m) { k_center<<<nbk(K, 256), 256>>>(center, logits, R, K, m); }

/* ---- multi-crop cross-entropy ---- */
extern "C" size_t dr_ce_scratch(int Rs) { return al((size_t)Rs) * 4; }
__global__ void k_lse(const float *ls, int K, float ts, float *lse) {   /* one block per student row: logsumexp(ls / ts) */
    __shared__ float sh[256]; const int r = blockIdx.x; float m = -INFINITY;
    for (int k = threadIdx.x; k < K; k += blockDim.x) m = fmaxf(m, ls[(size_t)r * K + k] / ts);
    sh[threadIdx.x] = m; __syncthreads();
    for (int k = blockDim.x / 2; k; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + k]); __syncthreads(); }
    m = sh[0]; __syncthreads();
    float s = 0.f; for (int k = threadIdx.x; k < K; k += blockDim.x) s += expf(ls[(size_t)r * K + k] / ts - m);
    sh[threadIdx.x] = s; __syncthreads();
    for (int k = blockDim.x / 2; k; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k]; __syncthreads(); }
    if (!threadIdx.x) lse[r] = m + logf(sh[0]);
}
/* grid (K chunks, npair): dls[s][k] += w (softmax[k] - q[t][k]) / ts; loss += w (lse_s - sum_k q[t][k] ls[s][k] / ts) */
__global__ void k_ce(const float *ls, int K, float ts, const float *q, const int *ps, const int *pt, const float *pw, const float *lse, float *dls, float *out) {
    __shared__ float sh[256];
    const int p = blockIdx.y, s = ps[p], t = pt[p]; const float w = pw[p], L = lse[s];
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    float c = 0.f;
    if (k < K) {
        const float l = ls[(size_t)s * K + k] / ts, qt = q[(size_t)t * K + k];
        atomicAdd(&dls[(size_t)s * K + k], w * (expf(l - L) - qt) / ts);
        c = qt * l;
    }
    sh[threadIdx.x] = c; __syncthreads();
    for (int j = blockDim.x / 2; j; j >>= 1) { if (threadIdx.x < j) sh[threadIdx.x] += sh[threadIdx.x + j]; __syncthreads(); }
    if (!threadIdx.x) atomicAdd(out, -w * sh[0] + (blockIdx.x == 0 ? w * L : 0.f));
}
extern "C" void dr_ce(const float *ls, int Rs, int K, float ts, const float *q, const int *ps, const int *pt, const float *pw, int npair, float *dls, float *out, void *scratch) {
    float *lse = (float *)scratch;
    k_lse<<<Rs, 256>>>(ls, K, ts, lse);
    cudaMemsetAsync(dls, 0, (size_t)Rs * K * 4, 0);
    cudaMemsetAsync(out, 0, 4, 0);
    if (npair > 0) { dim3 g(nbk(K, 256), npair); k_ce<<<g, 256>>>(ls, K, ts, q, ps, pt, pw, lse, dls, out); }
}

/* ---- KoLeo (one block; N <= 1024 rows, D <= 4096) ---- */
__global__ void k_koleo(const float *x, int N, int D, int gs, float w, float *gx, float *out, float *xh, float *nrm, int *nn, float *dxh) {
    for (int i = threadIdx.x >> 5; i < N; i += blockDim.x >> 5) {   /* unit rows */
        const int lane = threadIdx.x & 31; float s = 0.f;
        for (int c = lane; c < D; c += 32) { const float v = x[(size_t)i * D + c]; s += v * v; }
        for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        const float n = sqrtf(s), inv = 1.f / fmaxf(n, 1e-8f);
        for (int c = lane; c < D; c += 32) { xh[(size_t)i * D + c] = x[(size_t)i * D + c] * inv; dxh[(size_t)i * D + c] = 0.f; }
        if (!lane) nrm[i] = n;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < N; i += blockDim.x) {   /* nearest neighbour by the largest dot product within the group */
        const int g0 = gs > 0 ? i / gs * gs : 0, g1 = gs > 0 ? min(g0 + gs, N) : N;
        float best = -INFINITY; int bj = -1;
        for (int j = g0; j < g1; j++) { if (j == i) continue; float d = 0.f; for (int c = 0; c < D; c++) d += xh[(size_t)i * D + c] * xh[(size_t)j * D + c]; if (d > best) { best = d; bj = j; } }
        nn[i] = bj;
    }
    __syncthreads();
    float L = 0.f;
    if (threadIdx.x == 0) {   /* the loss and d L / d xh (serial: N small) */
        for (int i = 0; i < N; i++) {
            const int j = nn[i]; if (j < 0) continue;
            float s = 0.f; for (int c = 0; c < D; c++) { const float df = xh[(size_t)i * D + c] - xh[(size_t)j * D + c] + 1e-8f; s += df * df; }
            const float d = sqrtf(s); L += -logf(d + 1e-8f);
            const float k = -w / N / ((d + 1e-8f) * fmaxf(d, 1e-30f));
            for (int c = 0; c < D; c++) { const float df = xh[(size_t)i * D + c] - xh[(size_t)j * D + c] + 1e-8f; dxh[(size_t)i * D + c] += k * df; dxh[(size_t)j * D + c] -= k * df; }
        }
        out[0] = w * L / N;
    }
    __syncthreads();
    for (int i = threadIdx.x >> 5; i < N; i += blockDim.x >> 5) {   /* back through the normalisation */
        const int lane = threadIdx.x & 31; float d = 0.f;
        for (int c = lane; c < D; c += 32) d += xh[(size_t)i * D + c] * dxh[(size_t)i * D + c];
        for (int o = 16; o; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        const float n = nrm[i], inv = 1.f / fmaxf(n, 1e-8f);
        for (int c = lane; c < D; c += 32) gx[(size_t)i * D + c] += (dxh[(size_t)i * D + c] - (n > 1e-8f ? xh[(size_t)i * D + c] * d : 0.f)) * inv;
    }
}
extern "C" void dr_koleo(const float *x, int N, int D, int gs, float w, float *gx, float *out) {
    static float *buf = nullptr; static size_t cap = 0;
    const size_t need = ((size_t)2 * N * D + 2 * (size_t)N) * 4;
    if (need > cap) { if (buf) cudaFree(buf); cudaMalloc(&buf, need); cap = need; }
    float *xh = buf, *dxh = buf + (size_t)N * D, *nrm = dxh + (size_t)N * D; int *nn = (int *)(nrm + N);
    k_koleo<<<1, 256>>>(x, N, D, gs, w, gx, out, xh, nrm, nn, dxh);
}

/* ---- pooling, optimiser ---- */
__global__ void k_pool(const float *t, int n, int S, int C, int r0, float *cls) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n * C) return;
    const int s = i / C, c = i % C; float a = 0.f;
    for (int r = r0; r < S; r++) a += t[((size_t)s * S + r) * C + c];
    cls[i] = a / (float)(S - r0);
}
__global__ void k_pool_bwd(const float *g, int n, int S, int C, int r0, float *gt) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i >= (long long)n * S * C) return;
    const int c = (int)(i % C); const long long rr = i / C; const int r = (int)(rr % S), s = (int)(rr / S);
    if (r >= r0) gt[i] += g[(size_t)s * C + c] / (float)(S - r0);
}
extern "C" void dr_pool_fwd(const float *t, int n, int S, int C, int r0, float *cls) { k_pool<<<nbk((long long)n * C, 256), 256>>>(t, n, S, C, r0, cls); }
extern "C" void dr_pool_bwd(const float *g, int n, int S, int C, int r0, float *gt) { k_pool_bwd<<<nbk((long long)n * S * C, 256), 256>>>(g, n, S, C, r0, gt); }
__global__ void k_adamw(float *p, const float *g, float *m, float *v, size_t n, float lr, float b1, float b2, float eps, float wd, float c1, float c2) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        m[i] = b1 * m[i] + (1 - b1) * g[i]; v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
        p[i] -= lr * ((m[i] / c1) / (sqrtf(v[i] / c2) + eps) + wd * p[i]);
    }
}
extern "C" void dr_adamw(float *p, const float *g, float *m, float *v, size_t n, float lr, float b1, float b2, float eps, float wd, int step) {
    k_adamw<<<nbk((long long)n, 256) < 65535 ? nbk((long long)n, 256) : 65535, 256>>>(p, g, m, v, n, lr, b1, b2, eps, wd, 1.f - powf(b1, (float)step), 1.f - powf(b2, (float)step));
}
__global__ void k_ema(float *pt, const float *p, size_t n, float mom) { for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) pt[i] = mom * pt[i] + (1.f - mom) * p[i]; }
extern "C" void dr_ema(float *pt, const float *p, size_t n, float mom) { k_ema<<<nbk((long long)n, 256) < 65535 ? nbk((long long)n, 256) : 65535, 256>>>(pt, p, n, mom); }

/* KoLeo against a candidate set (queue of earlier rows, no gradient): one block; N <= 64 rows, M <= 4096 candidates */
__global__ void k_koleo_q(const float *x, int N, const float *cand, int M, int D, float w, float *gx, float *out, float *xh, float *ch, int *nn) {
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int i = wp; i < N + M; i += nw) {   /* unit rows: the current rows, then the candidates */
        const float *src = i < N ? x + (size_t)i * D : cand + (size_t)(i - N) * D; float *dst = i < N ? xh + (size_t)i * D : ch + (size_t)(i - N) * D;
        float s = 0.f; for (int c = lane; c < D; c += 32) s += src[c] * src[c];
        for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        const float inv = 1.f / fmaxf(sqrtf(s), 1e-8f);
        for (int c = lane; c < D; c += 32) dst[c] = src[c] * inv;
    }
    __syncthreads();
    __shared__ float bd[1024]; __shared__ int bj[1024];
    for (int i = 0; i < N; i++) {   /* nearest candidate of row i: each thread scans a share, then a block argmax */
        float best = -INFINITY; int b = -1;
        for (int j = threadIdx.x; j < M; j += blockDim.x) { float d = 0.f; for (int c = 0; c < D; c++) d += xh[(size_t)i * D + c] * ch[(size_t)j * D + c]; if (d > best) { best = d; b = j; } }
        bd[threadIdx.x] = best; bj[threadIdx.x] = b; __syncthreads();
        for (int k = blockDim.x / 2; k; k >>= 1) { if (threadIdx.x < k && bd[threadIdx.x + k] > bd[threadIdx.x]) { bd[threadIdx.x] = bd[threadIdx.x + k]; bj[threadIdx.x] = bj[threadIdx.x + k]; } __syncthreads(); }
        if (!threadIdx.x) nn[i] = bj[0];
        __syncthreads();
    }
    if (wp == 0) {   /* loss and gradient, one warp per row in turn */
        float L = 0.f;
        for (int i = 0; i < N; i++) {
            const int j = nn[i]; float s = 0.f;
            for (int c = lane; c < D; c += 32) { const float df = xh[(size_t)i * D + c] - ch[(size_t)j * D + c] + 1e-8f; s += df * df; }
            for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
            const float d = sqrtf(s), k = -w / N / ((d + 1e-8f) * fmaxf(d, 1e-30f));
            L += -logf(d + 1e-8f);
            /* d / d xh_i = k (xh_i - c_j + eps); back through xh = x / |x| */
            float dot = 0.f, nx = 0.f;
            for (int c = lane; c < D; c += 32) { const float g = k * (xh[(size_t)i * D + c] - ch[(size_t)j * D + c] + 1e-8f); dot += g * xh[(size_t)i * D + c]; nx += x[(size_t)i * D + c] * x[(size_t)i * D + c]; }
            for (int o = 16; o; o >>= 1) { dot += __shfl_xor_sync(0xffffffffu, dot, o); nx += __shfl_xor_sync(0xffffffffu, nx, o); }
            const float n = sqrtf(nx), inv = 1.f / fmaxf(n, 1e-8f);
            for (int c = lane; c < D; c += 32) { const float g = k * (xh[(size_t)i * D + c] - ch[(size_t)j * D + c] + 1e-8f); gx[(size_t)i * D + c] += (g - (n > 1e-8f ? xh[(size_t)i * D + c] * dot : 0.f)) * inv; }
        }
        if (!lane) out[0] = w * L / N;
    }
}
extern "C" void dr_koleo_q(const float *x, int N, const float *cand, int M, int D, float w, float *gx, float *out) {
    static float *buf = nullptr; static size_t cap = 0;
    const size_t need = ((size_t)(N + M) * D + N) * 4;
    if (need > cap) { if (buf) cudaFree(buf); cudaMalloc(&buf, need); cap = need; }
    if (M < 1) { cudaMemsetAsync(out, 0, 4, 0); return; }
    k_koleo_q<<<1, 256>>>(x, N, cand, M, D, w, gx, out, buf, buf + (size_t)N * D, (int *)(buf + (size_t)(N + M) * D));
}

/* ---- Muon for one [Co][K] matrix on cuBLASLt GEMMs (Newton-Schulz on the smaller side: X^T X when Co > K). Same update as nn_muon:
   nesterov momentum, X = M / |M|_F, 5 quintic iterations X <- a X + (b A + c A^2) X (A = X X^T; tall: X (b A + c A^2), A = X^T X),
   p <- p (1 - lr wd) - lr sqrt(max(1, Co / K)) X ---- */
__global__ void k_mu_mom(const float *g, float *mom, float *x, size_t n, float beta, double *ss) {
    __shared__ float r[256]; float a = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { const float m = beta * mom[i] + g[i]; mom[i] = m; const float v = g[i] + beta * m; x[i] = v; a += v * v; }
    r[threadIdx.x] = a; __syncthreads(); for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) r[threadIdx.x] += r[threadIdx.x + o]; __syncthreads(); }
    if (!threadIdx.x) atomicAdd(ss, (double)r[0]);
}
__global__ void k_mu_scale(float *x, size_t n, const double *ss) { const float inv = (float)(1.0 / (sqrt(ss[0]) + 1e-7)); for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) x[i] *= inv; }
__global__ void k_mu_lin(float *y, const float *x, size_t n, float a, float b) { for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) y[i] = a * y[i] + b * x[i]; }   /* y = a y + b x */
__global__ void k_mu_apply(float *p, const float *o, size_t n, float lr, float sc, float wd) { for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = p[i] * (1.f - lr * wd) - lr * sc * o[i]; }
extern "C" size_t dr_muon_work(int Co, int K) { const size_t m = Co < K ? Co : K; return ((size_t)2 * Co * K + 2 * m * m + 16) * 4; }
extern "C" void dr_muon(float *p, const float *g, float *mom, int Co, int K, float lr, float beta, float wd, float *work) {
    const size_t n = (size_t)Co * K, m = Co < K ? Co : K; const int tall = Co > K;
    float *X = work, *Y = X + n, *A = Y + n, *B = A + m * m; double *ss = (double *)(B + m * m);
    const unsigned nb = nbk((long long)n, 256) < 1024 ? nbk((long long)n, 256) : 1024, nm = nbk((long long)(m * m), 256) < 1024 ? nbk((long long)(m * m), 256) : 1024;
    cudaMemsetAsync(ss, 0, 8, 0);
    k_mu_mom<<<nb, 256>>>(g, mom, X, n, beta, ss);
    k_mu_scale<<<nb, 256>>>(X, n, ss);
    const float a = 3.4445f, b = -4.7750f, c = 2.0315f;
    for (int it = 0; it < 5; it++) {
        if (tall) at_gemm(1, 0, (int)m, (int)m, Co, X, X, A, 0.f);   /* A = X^T X */
        else at_gemm(0, 1, (int)m, (int)m, K, X, X, A, 0.f);         /* A = X X^T */
        at_gemm(0, 0, (int)m, (int)m, (int)m, A, A, B, 0.f);         /* B = A^2 */
        k_mu_lin<<<nm, 256>>>(B, A, m * m, c, b);                     /* B = c A^2 + b A */
        cudaMemcpyAsync(Y, X, n * 4, cudaMemcpyDeviceToDevice, 0);
        if (tall) at_gemm(0, 0, Co, K, K, X, B, Y, 1.f);   /* Y = X + X B ... */
        else at_gemm(0, 0, Co, K, Co, B, X, Y, 1.f);                        /* Y = X + B X ... */
        k_mu_lin<<<nb, 256>>>(Y, X, n, 1.f, a - 1.f);                 /* ... + (a - 1) X = a X + (B) X */
        float *t = X; X = Y; Y = t;
    }
    k_mu_apply<<<nb, 256>>>(p, X, n, lr, sqrtf(fmaxf(1.f, (float)Co / (float)K)), wd);
}
