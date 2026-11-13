/* CUDA ops: per-level scale FiLM on GroupNorm affines (src/unet.c cfg.film). A FiLM'd norm of C channels with K codes c_k (the window's
   log2 voxel sizes) has gamma_eff = gamma (1 + sum_k c_k A_kc), beta_eff = beta + sum_k c_k D_kc, A / D at fo (K x C each, A first).
   nn_film_fwd writes the effective pair into the forward weights fw; nn_film_bwd turns the gradient of the effective pair (in g at the
   norm's offsets) into the gradients of gamma, A and D (beta's is the effective one's). */
#include "nn_common.cuh"

struct film_codes { float c[4]; };
__global__ void film_fwd_k(const float *p, float *fw, size_t gam, size_t bet, size_t fo, int C, int K, film_codes cc) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= C) return;
    const float *A = p + fo, *D = p + fo + (size_t)K * C;
    float s = 0.f, t = 0.f;
    for (int k = 0; k < K; k++) { s += cc.c[k] * A[(size_t)k * C + c]; t += cc.c[k] * D[(size_t)k * C + c]; }
    fw[gam + c] = p[gam + c] * (1.f + s); fw[bet + c] = p[bet + c] + t;
}
__global__ void film_bwd_k(const float *p, float *g, size_t gam, size_t bet, size_t fo, int C, int K, film_codes cc) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= C) return;
    const float *A = p + fo; float *gA = g + fo, *gD = g + fo + (size_t)K * C;
    const float ge = g[gam + c], be = g[bet + c], ga = p[gam + c];
    float s = 0.f;
    for (int k = 0; k < K; k++) { s += cc.c[k] * A[(size_t)k * C + c]; gA[(size_t)k * C + c] += cc.c[k] * ge * ga; gD[(size_t)k * C + c] += cc.c[k] * be; }
    g[gam + c] = ge * (1.f + s);
}
static film_codes codes_of(const float *c, int K) { film_codes cc = {{0.f, 0.f, 0.f, 0.f}}; for (int k = 0; k < K && k < 4; k++) cc.c[k] = c[k]; return cc; }
extern "C" void nn_film_fwd(const float *p, float *fw, size_t gam, size_t bet, size_t fo, int C, int K, const float *codes) {
    film_fwd_k<<<(C + 127) / 128, 128>>>(p, fw, gam, bet, fo, C, K, codes_of(codes, K)); KCHECK();
}
extern "C" void nn_film_bwd(const float *p, float *g, size_t gam, size_t bet, size_t fo, int C, int K, const float *codes) {
    film_bwd_k<<<(C + 127) / 128, 128>>>(p, g, gam, bet, fo, C, K, codes_of(codes, K)); KCHECK();
}

/* ---- cfg.gfilm: the global-context FiLM (device codes: the attention stage's pooled output) ---- */
__global__ void gf_pool_k(const float *x, int C, long S, float *g) {
    const int c = blockIdx.x; double a = 0;
    for (long s = threadIdx.x; s < S; s += blockDim.x) a += x[(size_t)c * S + s];
    __shared__ double r[256]; r[threadIdx.x] = a; __syncthreads();
    for (int h = blockDim.x / 2; h > 0; h >>= 1) { if ((int)threadIdx.x < h) r[threadIdx.x] += r[threadIdx.x + h]; __syncthreads(); }
    if (!threadIdx.x) g[c] = (float)(r[0] / (double)S);
}
__global__ void gf_fwd_k(const float *p, float *fw, size_t gam, size_t bet, size_t fo, int C, int K, const float *gv, float *g1) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= C) return;
    const float *A = p + fo, *B = p + fo + (size_t)K * C;
    float s = 0.f, t = 0.f;
    for (int k = 0; k < K; k++) { s += gv[k] * A[(size_t)k * C + c]; t += gv[k] * B[(size_t)k * C + c]; }
    const float a = fw[gam + c]; g1[c] = a;
    fw[gam + c] = a * (1.f + s); fw[bet + c] += t;
}
__global__ void gf_ab_k(const float *g, size_t gam, size_t bet, float *gp, size_t fo, int C, int K, const float *gv, const float *g1) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= (size_t)K * C) return;
    const int c = (int)(i % C), k = (int)(i / C);
    gp[fo + i] += gv[k] * g[gam + c] * g1[c]; gp[fo + (size_t)K * C + i] += gv[k] * g[bet + c];
}
__global__ void gf_dg_k(const float *p, const float *g, size_t gam, size_t bet, size_t fo, int C, int K, const float *g1, float *dg) {
    const int k = blockIdx.x; const float *A = p + fo + (size_t)k * C, *B = p + fo + (size_t)K * C + (size_t)k * C;
    float a = 0.f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) a += A[c] * g[gam + c] * g1[c] + B[c] * g[bet + c];
    __shared__ float r[256]; r[threadIdx.x] = a; __syncthreads();
    for (int h = blockDim.x / 2; h > 0; h >>= 1) { if ((int)threadIdx.x < h) r[threadIdx.x] += r[threadIdx.x + h]; __syncthreads(); }
    if (!threadIdx.x) dg[k] += r[0];
}
__global__ void gf_conv_k(const float *p, float *g, size_t gam, size_t fo, int C, int K, const float *gv) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= C) return;
    const float *A = p + fo; float s = 0.f;
    for (int k = 0; k < K; k++) s += gv[k] * A[(size_t)k * C + c];
    g[gam + c] *= 1.f + s;
}
__global__ void gf_tok_k(float *x, int C, long S, const float *dg) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= (size_t)C * S) return;
    x[i] += dg[i / S] / (float)S;
}
extern "C" void nn_gfilm_pool(const float *x, int C, long S, float *g) { gf_pool_k<<<C, 256>>>(x, C, S, g); KCHECK(); }
extern "C" void nn_gfilm_fwd(const float *p, float *fw, size_t gam, size_t bet, size_t fo, int C, int K, const float *gv, float *g1) {
    gf_fwd_k<<<(C + 127) / 128, 128>>>(p, fw, gam, bet, fo, C, K, gv, g1); KCHECK();
}
extern "C" void nn_gfilm_bwd(const float *p, float *g, size_t gam, size_t bet, size_t fo, int C, int K, const float *gv, const float *g1, float *dg) {
    const size_t n = (size_t)K * C;
    gf_ab_k<<<(unsigned)((n + 255) / 256), 256>>>(g, gam, bet, g, fo, C, K, gv, g1);
    gf_dg_k<<<K, 256>>>(p, g, gam, bet, fo, C, K, g1, dg);
    gf_conv_k<<<(C + 127) / 128, 128>>>(p, g, gam, fo, C, K, gv);   /* (after both: they read the effective affine's gradient) */
    KCHECK();
}
extern "C" void nn_gfilm_tok(float *x, int C, long S, const float *dg) { const size_t n = (size_t)C * S; gf_tok_k<<<(unsigned)((n + 255) / 256), 256>>>(x, C, S, dg); KCHECK(); }
