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
    w->zc = nullptr; w->zc_m = 0.f;
}
__global__ void k_zcenter(float *z, float *c, int R, int B, float m) {   /* one thread per column: optional EMA update, then z -= c */
    const int b = blockIdx.x * blockDim.x + threadIdx.x; if (b >= B) return;
    if (m > 0.f) { float s = 0.f; for (int r = 0; r < R; r++) s += z[(size_t)r * B + b]; c[b] = c[b] * m + s / R * (1.f - m); }
    const float cb = c[b]; for (int r = 0; r < R; r++) z[(size_t)r * B + b] -= cb;
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
    if (w->zc) k_zcenter<<<nbk(B, 256), 256>>>(w->z, w->zc, R, B, w->zc_m);
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
extern "C" size_t dr_sk_scratch(int R, int K) { return (al((size_t)K) + al((size_t)R) + 64) * 4; }   /* (the first 64 bytes: the reductions) */
__global__ void k_tq(const float *l, float *q, long long n, float tt) {   /* q = exp((l - 1) / tt): cosine-bounded logits (|l| <= g) */
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i < n) q[i] = expf((l[i] - 1.f) / tt);
}
__device__ __forceinline__ void atomic_max_f(float *a, float v) {   /* (any sign) */
    if (v >= 0.f) atomicMax((int *)a, __float_as_int(v)); else atomicMin((unsigned *)a, __float_as_uint(v));
}
__global__ void k_maxrow(const float *l, long long n, float *mx) {   /* grid-stride blocks, atomic max into mx[0] (set to -inf first) */
    __shared__ float sh[256]; float m = -INFINITY;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) m = fmaxf(m, l[i]);
    sh[threadIdx.x] = m; __syncthreads();
    for (int k = blockDim.x / 2; k; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + k]); __syncthreads(); }
    if (!threadIdx.x) atomic_max_f(mx, sh[0]);
}
__global__ void k_setf(float *p, float v) { p[0] = v; }
__global__ void k_tq_m(const float *l, float *q, long long n, float tt, const float *mx) { const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i < n) q[i] = expf((l[i] - mx[0]) / tt); }
__global__ void k_sum_all(const float *q, long long n, double *out) {   /* grid-stride, atomic double sum into out[0] (zeroed first) */
    __shared__ double sh[256]; double v = 0;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) v += q[i];
    sh[threadIdx.x] = v; __syncthreads();
    for (int k = blockDim.x / 2; k; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k]; __syncthreads(); }
    if (!threadIdx.x) atomicAdd(out, sh[0]);
}
__global__ void k_scale_all(float *q, long long n, const double *s) { const float inv = (float)(1.0 / s[0]); const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i < n) q[i] *= inv; }
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
    double *ds = (double *)scratch; float *mx = (float *)scratch + 4;
    const unsigned gr = nbk(n, 256) < 1024 ? nbk(n, 256) : 1024;
    k_setf<<<1, 1>>>(mx, -INFINITY);
    k_maxrow<<<gr, 256>>>(l, n, mx);   /* (the global max: exp stays finite for any logit scale; the normalisation removes it) */
    k_tq_m<<<nbk(n, 256), 256>>>(l, q, n, tt, mx);
    cudaMemsetAsync(ds, 0, 8, 0);
    k_sum_all<<<gr, 256>>>(q, n, ds); k_scale_all<<<nbk(n, 256), 256>>>(q, n, ds);
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
__global__ void k_rownorm(const float *x, int C, float *y, float *rstd) {   /* one block per row */
    __shared__ float sh[256]; const int r = blockIdx.x; const float *xr = x + (size_t)r * C; float *yr = y + (size_t)r * C;
    float s = 0.f; for (int c = threadIdx.x; c < C; c += blockDim.x) s += xr[c];
    sh[threadIdx.x] = s; __syncthreads(); for (int k = blockDim.x / 2; k; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k]; __syncthreads(); }
    const float m = sh[0] / C; __syncthreads();
    float v = 0.f; for (int c = threadIdx.x; c < C; c += blockDim.x) { const float d = xr[c] - m; v += d * d; }
    sh[threadIdx.x] = v; __syncthreads(); for (int k = blockDim.x / 2; k; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k]; __syncthreads(); }
    const float rs = rsqrtf(sh[0] / C + 1e-6f); __syncthreads();
    for (int c = threadIdx.x; c < C; c += blockDim.x) yr[c] = (xr[c] - m) * rs;
    if (threadIdx.x == 0) rstd[r] = rs;
}
__global__ void k_rownorm_bwd(const float *dy, const float *y, const float *rstd, int C, float *dx) {
    __shared__ float sa[256], sb[256]; const int r = blockIdx.x; const float *g = dy + (size_t)r * C, *yr = y + (size_t)r * C;
    float a = 0.f, b = 0.f; for (int c = threadIdx.x; c < C; c += blockDim.x) { a += g[c]; b += g[c] * yr[c]; }
    sa[threadIdx.x] = a; sb[threadIdx.x] = b; __syncthreads();
    for (int k = blockDim.x / 2; k; k >>= 1) { if (threadIdx.x < k) { sa[threadIdx.x] += sa[threadIdx.x + k]; sb[threadIdx.x] += sb[threadIdx.x + k]; } __syncthreads(); }
    const float ma = sa[0] / C, mb = sb[0] / C, rs = rstd[r]; __syncthreads();
    for (int c = threadIdx.x; c < C; c += blockDim.x) dx[(size_t)r * C + c] = rs * (g[c] - ma - yr[c] * mb);
}
extern "C" void dr_rownorm_fwd(const float *x, int R, int C, float *y, float *rstd) { k_rownorm<<<R, 256>>>(x, C, y, rstd); }
extern "C" void dr_rownorm_bwd(const float *dy, const float *y, const float *rstd, int R, int C, float *dx) { k_rownorm_bwd<<<R, 256>>>(dy, y, rstd, C, dx); }
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
        if (tall) at_gemm_f32(1, 0, (int)m, (int)m, Co, X, X, A, 0.f);   /* A = X^T X */
        else at_gemm_f32(0, 1, (int)m, (int)m, K, X, X, A, 0.f);         /* A = X X^T */
        at_gemm_f32(0, 0, (int)m, (int)m, (int)m, A, A, B, 0.f);         /* B = A^2 */
        k_mu_lin<<<nm, 256>>>(B, A, m * m, c, b);                     /* B = c A^2 + b A */
        cudaMemcpyAsync(Y, X, n * 4, cudaMemcpyDeviceToDevice, 0);
        if (tall) at_gemm_f32(0, 0, Co, K, K, X, B, Y, 1.f);   /* Y = X + X B ... */
        else at_gemm_f32(0, 0, Co, K, Co, B, X, Y, 1.f);                        /* Y = X + B X ... */
        k_mu_lin<<<nb, 256>>>(Y, X, n, 1.f, a - 1.f);                 /* ... + (a - 1) X = a X + (B) X */
        float *t = X; X = Y; Y = t;
    }
    k_mu_apply<<<nb, 256>>>(p, X, n, lr, sqrtf(fmaxf(1.f, (float)Co / (float)K)), wd);
}

/* ---- iBOT input masking: the voxels of masked token cells (edge ps; mask [G^3], G = n / ps) set to val ---- */
__global__ void k_mask_cells(float *x, int n, int ps, const uint8_t *mask, float val) {
    const long long n3 = (long long)n * n * n;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n3; i += (long long)gridDim.x * blockDim.x) {
        const int z = (int)(i / ((long long)n * n)), y = (int)((i / n) % n), xx = (int)(i % n), G = n / ps;
        if (mask[((size_t)(z / ps) * G + y / ps) * G + xx / ps]) x[i] = val;
    }
}
extern "C" void dr_mask_cells(float *x, int n, int ps, const uint8_t *mask, float val) { k_mask_cells<<<nbk((long long)n * n * n, 256), 256>>>(x, n, ps, mask, val); }

/* ---- per-row argmax (prototype usage monitor) ---- */
__global__ void k_argmax_rows(const float *q, int K, int *out) {
    __shared__ float bv[256]; __shared__ int bi[256];
    const int r = blockIdx.x; float best = -INFINITY; int b = 0;
    for (int k = threadIdx.x; k < K; k += blockDim.x) { const float v = q[(size_t)r * K + k]; if (v > best) { best = v; b = k; } }
    bv[threadIdx.x] = best; bi[threadIdx.x] = b; __syncthreads();
    for (int s = blockDim.x / 2; s; s >>= 1) { if (threadIdx.x < s && bv[threadIdx.x + s] > bv[threadIdx.x]) { bv[threadIdx.x] = bv[threadIdx.x + s]; bi[threadIdx.x] = bi[threadIdx.x + s]; } __syncthreads(); }
    if (!threadIdx.x) out[r] = bi[0];
}
extern "C" void dr_argmax_rows(const float *q, int R, int K, int *out) { if (R > 0) k_argmax_rows<<<R, 256>>>(q, K, out); }
/* unit rows in place (the weight-normed prototype directions after a Muon step) */
extern "C" void dr_unit_rows(float *v, int R, int N) { static float *r = nullptr; static int rn = 0; if (rn < R) { if (r) cudaFree(r); cudaMalloc(&r, (size_t)R * 4); rn = R; } k_rownorm<<<nbk((long long)R * 32, 256), 256>>>(v, v, r, R, N, 1e-12f, nullptr); }

/* ---- object / region terms (upstream ObjectLoss): margin-capped SupCon L_out between student anchors S [N][D] and teacher keys T [N][D]
   (the same points in the other view; unit vectors), objects obj [N] (> 0 labelled), validity vs / vt [N] (u8). One block per anchor ---- */
__global__ void k_obj(const float *S, const float *T, const int *obj, const uint8_t *vs, const uint8_t *vt, int N, int D, float temp, float margin,
                      float *gs, float *lossv, float *wv) {
    extern __shared__ float sh[];   /* [N] similarities, then 256 reduction slots */
    float *sim = sh, *red = sh + N;
    const int i = blockIdx.x, oi = obj[i];
    if (!vs[i] || oi <= 0) { if (!threadIdx.x) { lossv[i] = 0.f; wv[i] = 0.f; } return; }
    int npos = 0, npns = 0, nneg = 0;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        float a = 0.f; for (int c = 0; c < D; c++) a += S[(size_t)i * D + c] * T[(size_t)j * D + c];
        sim[j] = a;
    }
    __syncthreads();
    /* counts (serial over N by thread 0: N small) */
    __shared__ int cnt[3]; __shared__ float nlse;
    if (!threadIdx.x) {
        int p = 0, q = 0, n = 0; float m = -INFINITY;
        for (int j = 0; j < N; j++) { if (!vt[j] || obj[j] <= 0) continue; if (obj[j] == oi) { p++; if (j != i) q++; } else { n++; m = fmaxf(m, sim[j] / temp); } }
        cnt[0] = p; cnt[1] = q; cnt[2] = n;
        float z = 0.f; for (int j = 0; j < N; j++) if (vt[j] && obj[j] > 0 && obj[j] != oi) z += expf(sim[j] / temp - m);
        nlse = n ? m + logf(z) : 0.f;
    }
    __syncthreads();
    npos = cnt[0]; npns = cnt[1]; nneg = cnt[2];
    if (!nneg || !npos) { if (!threadIdx.x) { lossv[i] = 0.f; wv[i] = 0.f; } return; }
    const int excl = npns > 0;   /* exclude the self point when another positive exists */
    const int np_ = excl ? npns : npos;
    /* loss = 1/np sum_pos softplus(nlse - min(s, m) / t); d/ds_pos = -sig / np / t (s < m); d nlse = sum_pos sig / np -> softmax over negatives / t */
    float L = 0.f, sg = 0.f;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        if (!vt[j] || obj[j] != oi || (excl && j == i)) continue;
        const float pl = fminf(sim[j], margin) / temp, x = nlse - pl;
        L += x > 20.f ? x : log1pf(expf(x)); sg += 1.f / (1.f + expf(-x));
    }
    red[threadIdx.x] = L; __syncthreads();
    for (int s = blockDim.x / 2; s; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s]; __syncthreads(); }
    const float Lt = red[0]; __syncthreads();
    red[threadIdx.x] = sg; __syncthreads();
    for (int s = blockDim.x / 2; s; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s]; __syncthreads(); }
    const float sgt = red[0] / np_; __syncthreads();
    /* ds_j into sim (reused as the coefficient row) */
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        float c = 0.f;
        if (vt[j] && obj[j] > 0) {
            if (obj[j] == oi) { if (!(excl && j == i) && sim[j] < margin) { const float x = nlse - sim[j] / temp; c = -1.f / (1.f + expf(-x)) / np_ / temp; } }
            else c = sgt * expf(sim[j] / temp - nlse) / temp;
        }
        sim[j] = c;
    }
    __syncthreads();
    for (int c = threadIdx.x; c < D; c += blockDim.x) { float a = 0.f; for (int j = 0; j < N; j++) a += sim[j] * T[(size_t)j * D + c]; gs[(size_t)i * D + c] = a; }
    if (!threadIdx.x) { lossv[i] = Lt / np_; wv[i] = 1.f; }
}
/* the anchors' mean: loss -> out[0] (+=), the per-anchor gradients scaled by w / count into gS (+=); returns via out[1] the anchor count */
__global__ void k_obj_fin(const float *lossv, const float *wv, const float *gtmp, int N, int D, float w, float cnt_extra, float *gS, float *out) {
    __shared__ float red[256], cr[256];
    float a = 0.f, c = 0.f; for (int i = threadIdx.x; i < N; i += blockDim.x) { a += lossv[i]; c += wv[i]; }
    red[threadIdx.x] = a; cr[threadIdx.x] = c; __syncthreads();
    for (int s = blockDim.x / 2; s; s >>= 1) { if (threadIdx.x < s) { red[threadIdx.x] += red[threadIdx.x + s]; cr[threadIdx.x] += cr[threadIdx.x + s]; } __syncthreads(); }
    const float n = cr[0] + cnt_extra, inv = n > 0.f ? w / n : 0.f;
    for (int k = threadIdx.x; k < N * D; k += blockDim.x) if (wv[k / D] > 0.f) gS[k] += gtmp[k] * inv;
    if (!threadIdx.x) { out[0] += red[0] * inv; out[1] += cr[0]; }
}
extern "C" void dr_obj(const float *S, const float *T, const int *obj, const uint8_t *vs, const uint8_t *vt, int N, int D, float temp, float margin, float w, float *gS, float *out) {
    static float *buf = nullptr; static size_t cap = 0;
    const size_t need = ((size_t)N * D + 2 * (size_t)N) * 4;
    if (need > cap) { if (buf) cudaFree(buf); cudaMalloc(&buf, need); cap = need; }
    if (N <= 0) return;
    float *gt = buf, *lv = gt + (size_t)N * D, *wv = lv + N;
    k_obj<<<N, 128, ((size_t)N + 256) * 4>>>(S, T, obj, vs, vt, N, D, temp, margin, gt, lv, wv);
    k_obj_fin<<<1, 256>>>(lv, wv, gt, N, D, w, 0.f, gS, out);
}

/* ---- per-object means of point rows: x [N][D], grp [N] (-1: none) -> m [M][D], cnt [M]; backward: gm [M][D] / cnt into gx (+=) ---- */
__global__ void k_gmean(const float *x, const int *grp, int N, int D, int M, float *m, float *cnt) {
    const int g = blockIdx.x; if (g >= M) return;
    for (int c = threadIdx.x; c < D; c += blockDim.x) { float a = 0.f; int n = 0; for (int i = 0; i < N; i++) if (grp[i] == g) { a += x[(size_t)i * D + c]; n++; } m[(size_t)g * D + c] = n ? a / n : 0.f; if (!c) cnt[g] = (float)n; }
}
__global__ void k_gmean_bwd(const float *gm, const float *cnt, const int *grp, int N, int D, float *gx) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= N * D) return;
    const int g = grp[i / D]; if (g < 0 || cnt[g] <= 0.f) return;
    gx[i] += gm[(size_t)g * D + i % D] / cnt[g];
}
extern "C" void dr_gmean(const float *x, const int *grp, int N, int D, int M, float *m, float *cnt) { if (M > 0) k_gmean<<<M, 64>>>(x, grp, N, D, M, m, cnt); }
extern "C" void dr_gmean_bwd(const float *gm, const float *cnt, const int *grp, int N, int D, float *gx) { if (N > 0) k_gmean_bwd<<<nbk((long long)N * D, 256), 256>>>(gm, cnt, grp, N, D, gx); }

/* ---- two-layer MLP (Linear -> GELU -> Linear) + L2 normalisation: the region head ---- */
extern "C" void dr_mlp2_layout(dr_mlp2 *m, int Cin, int H, int O) { m->Cin = Cin; m->H = H; m->O = O; size_t o = 0; m->w1 = o; o += (size_t)H * Cin; m->b1 = o; o += H; m->w2 = o; o += (size_t)O * H; m->b2 = o; o += O; m->n = o; }
extern "C" size_t dr_mlp2_ws(const dr_mlp2 *m, int R) { return (al((size_t)R * m->H) * 3 + al((size_t)R * m->O) * 3 + al((size_t)R)) * 4; }
extern "C" int dr_mlp2_fwd(const dr_mlp2 *m, const float *P, const float *x, int R, float *y, void *ws) {
    float *a = (float *)ws, *h = a + al((size_t)R * m->H), *z = h + al((size_t)R * m->H) * 2, *r = z + al((size_t)R * m->O) * 3;
    CK(at_gemm(0, 1, R, m->H, m->Cin, x, P + m->w1, a, 0.f)); k_bias_gelu<<<nbk((long long)R * m->H, 256), 256>>>(a, P + m->b1, h, R, m->H);
    CK(at_gemm(0, 1, R, m->O, m->H, h, P + m->w2, z, 0.f)); k_bias<<<nbk((long long)R * m->O, 256), 256>>>(z, P + m->b2, R, m->O);
    k_rownorm<<<nbk((long long)R * 32, 256), 256>>>(z, y, r, R, m->O, 1e-12f, nullptr);
    return 0;
}
extern "C" int dr_mlp2_bwd(const dr_mlp2 *m, const float *P, float *G, const float *x, const float *y, const float *dy, int R, float *dx, void *ws) {
    float *a = (float *)ws, *h = a + al((size_t)R * m->H), *dh = h + al((size_t)R * m->H), *z = dh + al((size_t)R * m->H), *dz = z + al((size_t)R * m->O), *r = dz + al((size_t)R * m->O) * 2;
    (void)z;
    k_rownorm_bwd<<<nbk((long long)R * 32, 256), 256>>>(y, r, dy, dz, R, m->O, 1e-12f, 0);
    CK(at_gemm(1, 0, m->O, m->H, R, dz, h, G + m->w2, 1.f)); colsum2(dz, R, m->O, G + m->b2);
    CK(at_gemm(0, 0, R, m->H, m->O, dz, P + m->w2, dh, 0.f));
    k_gelu_bwd<<<nbk((long long)R * m->H, 256), 256>>>(dh, a, (long long)R * m->H);
    CK(at_gemm(1, 0, m->H, m->Cin, R, dh, x, G + m->w1, 1.f)); colsum2(dh, R, m->H, G + m->b1);
    if (dx) CK(at_gemm(0, 0, R, m->Cin, m->H, dh, P + m->w1, dx, 1.f));   /* (accumulated) */
    return 0;
}

/* ---- ink term (upstream ObjectLoss._ink_loss, one anchor family): anchors A [Na][D] (unit) pulled toward keys Kp [np][D] and pushed from
   Kn [nn][D] (unit; teacher points of the other view + the FIFO queues). Pull keys whose point id equals the anchor's are left out while
   another pull key exists. Per anchor, with logits l = s / T: push mean m_n and lme_n = lse(l_n) - log nn; pull logits capped at
   max(margin, max(s_n) + T) / T (cap detached): mean m_p and lme_p over the kept keys; loss 0.5 (softplus(m_n - m_p) + softplus(lme_n -
   lme_p)). The mean over anchors times w -> out[0] (+=); d / d A (w / Na) added into gA. One block per anchor ---- */
__global__ void k_ink(const float *A, const int *aid, const float *Kp, const int *kid, int np, const float *Kn, int nn, int D, float T, float margin,
                      float *lv, float *gt) {
    extern __shared__ float sh[];
    float *sp_ = sh, *sn = sh + np, *red = sn + nn;
    const int i = blockIdx.x, me = aid ? aid[i] : -1;
    for (int j = threadIdx.x; j < np + nn; j += blockDim.x) {
        const float *k = j < np ? Kp + (size_t)j * D : Kn + (size_t)(j - np) * D; float a = 0.f;
        for (int c = 0; c < D; c++) a += A[(size_t)i * D + c] * k[c];
        sh[j] = a;
    }
    __syncthreads();
    __shared__ float st[8]; __shared__ int nkeep;
    if (!threadIdx.x) {
        float mx = -INFINITY, mn = 0.f; for (int j = 0; j < nn; j++) { mx = fmaxf(mx, sn[j]); mn += sn[j] / T; } mn /= nn;
        float z = 0.f; for (int j = 0; j < nn; j++) z += expf(sn[j] / T - mx / T);
        const float lmen = mx / T + logf(z) - logf((float)nn), cap = fmaxf(margin, mx + T);
        int other = 0; for (int j = 0; j < np; j++) other += !(kid && kid[j] == me && me >= 0);
        const int excl = other > 0; int cnt = 0; float pm = 0.f, pmx = -INFINITY;
        for (int j = 0; j < np; j++) { if (excl && kid && kid[j] == me && me >= 0) continue; const float pl = fminf(sp_[j], cap) / T; pm += pl; pmx = fmaxf(pmx, pl); cnt++; }
        pm /= cnt; float zp = 0.f;
        for (int j = 0; j < np; j++) { if (excl && kid && kid[j] == me && me >= 0) continue; zp += expf(fminf(sp_[j], cap) / T - pmx); }
        const float lmep = pmx + logf(zp) - logf((float)cnt);
        const float x1 = mn - pm, x2 = lmen - lmep;
        lv[i] = 0.5f * ((x1 > 20.f ? x1 : log1pf(expf(x1))) + (x2 > 20.f ? x2 : log1pf(expf(x2))));
        st[0] = 0.5f / (1.f + expf(-x1)); st[1] = 0.5f / (1.f + expf(-x2)); st[2] = mx / T + logf(z); st[3] = pmx + logf(zp); st[4] = cap; st[5] = (float)excl; nkeep = cnt;
    }
    __syncthreads();
    const float Aa = st[0], Bc = st[1], lsen = st[2], lsep = st[3], cap = st[4]; const int excl = st[5] > 0.f, cnt = nkeep;
    for (int j = threadIdx.x; j < np + nn; j += blockDim.x) {
        float c;
        if (j < np) {
            if ((excl && kid && kid[j] == me && me >= 0) || sp_[j] >= cap) c = 0.f;
            else c = -(Aa / (cnt * T) + Bc * expf(sp_[j] / T - lsep) / T);
        } else c = Aa / (nn * T) + Bc * expf(sn[j - np] / T - lsen) / T;
        red[j] = c;
    }
    __syncthreads();
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        float a = 0.f;
        for (int j = 0; j < np; j++) a += red[j] * Kp[(size_t)j * D + c];
        for (int j = 0; j < nn; j++) a += red[np + j] * Kn[(size_t)j * D + c];
        gt[(size_t)i * D + c] = a;
    }
}
__global__ void k_ink_fin(const float *lv, const float *gt, int Na, int D, float w, float *gA, float *out) {
    __shared__ float red[256]; float a = 0.f;
    for (int i = threadIdx.x; i < Na; i += blockDim.x) a += lv[i];
    red[threadIdx.x] = a; __syncthreads();
    for (int s = blockDim.x / 2; s; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s]; __syncthreads(); }
    for (int k = threadIdx.x; k < Na * D; k += blockDim.x) gA[k] += gt[k] * w / Na;
    if (!threadIdx.x) out[0] += red[0] * w / Na;
}
extern "C" void dr_ink(const float *A, const int *aid, int Na, const float *Kp, const int *kid, int np, const float *Kn, int nn, int D, float T, float margin, float w, float *gA, float *out) {
    if (Na <= 0 || np <= 0 || nn <= 0) return;
    static float *buf = nullptr; static size_t cap = 0;
    const size_t need = ((size_t)Na * D + Na) * 4;
    if (need > cap) { if (buf) cudaFree(buf); cudaMalloc(&buf, need); cap = need; }
    float *gt = buf, *lv = buf + (size_t)Na * D;
    k_ink<<<Na, 128, (size_t)2 * (np + nn) * 4>>>(A, aid, Kp, kid, np, Kn, nn, D, T, margin, lv, gt);
    k_ink_fin<<<1, 256>>>(lv, gt, Na, D, w, gA, out);
}

/* ---- InfoNCE over a key bank ---- */
extern "C" size_t dr_nce_scratch(int N, int M) { return al((size_t)N * M) * 4 + 256; }
__global__ void k_nce_rows(float *S, int M, const int *pos, float tau, float w, int N, float *out) {   /* one block per anchor: S row -> d L / d logit */
    __shared__ float sh[256]; __shared__ int si[256];
    const int r = blockIdx.x; float *row = S + (size_t)r * M; const int p = pos[r];
    float m = -INFINITY; int am = 0;
    for (int k = threadIdx.x; k < M; k += blockDim.x) { const float v = row[k] / tau; if (v > m) { m = v; am = k; } }
    sh[threadIdx.x] = m; si[threadIdx.x] = am; __syncthreads();
    for (int k = blockDim.x / 2; k; k >>= 1) { if (threadIdx.x < k && (sh[threadIdx.x + k] > sh[threadIdx.x] || (sh[threadIdx.x + k] == sh[threadIdx.x] && si[threadIdx.x + k] < si[threadIdx.x]))) { sh[threadIdx.x] = sh[threadIdx.x + k]; si[threadIdx.x] = si[threadIdx.x + k]; } __syncthreads(); }
    m = sh[0]; const int amax = si[0]; __syncthreads();
    float z = 0.f; for (int k = threadIdx.x; k < M; k += blockDim.x) z += expf(row[k] / tau - m);
    sh[threadIdx.x] = z; __syncthreads();
    for (int k = blockDim.x / 2; k; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k]; __syncthreads(); }
    z = sh[0]; const float sp = row[p] / tau; __syncthreads();
    const float g = w / ((float)N * tau);
    for (int k = threadIdx.x; k < M; k += blockDim.x) row[k] = g * (expf(row[k] / tau - m) / z - (k == p ? 1.f : 0.f));
    if (threadIdx.x == 0) { atomicAdd(out, w * (m + logf(z) - sp) / N); atomicAdd(out + 1, (amax == p ? 1.f : 0.f) / N); }
}
extern "C" void dr_infonce(const float *Q, int N, const float *K, int M, const int *pos, int D, float tau, float w, float *gQ, float *out, void *scratch) {
    if (N <= 0 || M <= 0) return;
    float *S = (float *)scratch;
    at_gemm_f32(0, 1, N, M, D, Q, K, S, 0.f);          /* S = Q K^T */
    k_nce_rows<<<N, 256>>>(S, M, pos, tau, w, N, out);  /* S -> d L / d S (times 1 / tau, w / N) */
    at_gemm_f32(0, 0, N, D, M, S, K, gQ, 1.f);          /* gQ += G K */
}

__global__ void k_rowdot(const float *A, const float *B, int N, int D, float *o) {   /* one warp per row */
    const int r = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, l = threadIdx.x & 31; if (r >= N) return;
    float s = 0.f; for (int d = l; d < D; d += 32) s += A[(size_t)r * D + d] * B[(size_t)r * D + d];
    for (int o2 = 16; o2; o2 >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o2);
    if (l == 0) o[r] = s;
}
__global__ void k_nce_rows_p(float *S, float *sp, int M, float tau, float w, int N, float *out) {   /* S [N][M] negatives, sp [N] positives -> d L / d s */
    __shared__ float sh[256]; const int r = blockIdx.x; float *row = S + (size_t)r * M; const float p = sp[r] / tau;
    float m = p, mn = -INFINITY; for (int k = threadIdx.x; k < M; k += blockDim.x) { const float v = row[k] / tau; m = fmaxf(m, v); mn = fmaxf(mn, v); }
    sh[threadIdx.x] = m; __syncthreads(); for (int k = blockDim.x / 2; k; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + k]); __syncthreads(); }
    m = sh[0]; __syncthreads();
    sh[threadIdx.x] = mn; __syncthreads(); for (int k = blockDim.x / 2; k; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + k]); __syncthreads(); }
    mn = sh[0]; __syncthreads();
    float z = 0.f; for (int k = threadIdx.x; k < M; k += blockDim.x) z += expf(row[k] / tau - m);
    sh[threadIdx.x] = z; __syncthreads(); for (int k = blockDim.x / 2; k; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k]; __syncthreads(); }
    z = sh[0] + expf(p - m); __syncthreads();
    const float g = w / ((float)N * tau);
    for (int k = threadIdx.x; k < M; k += blockDim.x) row[k] = g * expf(row[k] / tau - m) / z;
    if (threadIdx.x == 0) { sp[r] = g * (expf(p - m) / z - 1.f); atomicAdd(out, w * (m + logf(z) - p) / N); atomicAdd(out + 1, (p > mn ? 1.f : 0.f) / N); }
}
__global__ void k_axpy_rows(float *y, const float *a, const float *x, int N, int D) {   /* y[r] += a[r] x[r] */
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i >= (long long)N * D) return;
    y[i] += a[i / D] * x[i];
}
extern "C" void dr_infonce_p(const float *Q, const float *P, int N, const float *Kn, int M, int D, float tau, float w, float *gQ, float *out, void *scratch) {
    if (N <= 0) return;
    float *sp = (float *)scratch, *S = sp + al((size_t)N);
    k_rowdot<<<nbk((long long)N * 32, 256), 256>>>(Q, P, N, D, sp);
    if (M > 0) at_gemm_f32(0, 1, N, M, D, Q, Kn, S, 0.f);
    k_nce_rows_p<<<N, 256>>>(S, sp, M, tau, w, N, out);
    if (M > 0) at_gemm_f32(0, 0, N, D, M, S, Kn, gQ, 1.f);
    k_axpy_rows<<<nbk((long long)N * D, 256), 256>>>(gQ, sp, P, N, D);
}

__global__ void k_rows_gather(const float *src, int C, const int *idx, int n, float *dst) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i >= (long long)n * C) return;
    dst[i] = src[(size_t)idx[i / C] * C + i % C];
}
__global__ void k_rows_scatter_add(const float *src, const int *idx, int n, int C, float scale, float *dst) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i >= (long long)n * C) return;
    dst[(size_t)idx[i / C] * C + i % C] += scale * src[i];
}
extern "C" void dr_rows_gather(const float *src, int C, const int *idx, int n, float *dst) { if (n > 0) k_rows_gather<<<nbk((long long)n * C, 256), 256>>>(src, C, idx, n, dst); }
extern "C" void dr_rows_scatter_add(const float *src, const int *idx, int n, int C, float scale, float *dst) { if (n > 0) k_rows_scatter_add<<<nbk((long long)n * C, 256), 256>>>(src, idx, n, C, scale, dst); }
