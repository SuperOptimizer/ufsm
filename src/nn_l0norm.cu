/* Level-0 normalisation alternatives (unet cfg.l0_norm): GroupNorm statistics held constant -- none (mean 0, rstd 1) or
   BatchNorm-style running statistics (ema) -- and the GroupNorm backwards without the statistics' gradient terms. */
#include "nn_common.cuh"

int g_gn_const;   /* nn_gn_bwd / nn_gn_silu_bwd / nn_gn_silu_bwd_head: the group sums (mean and variance terms) scaled by 0 */
extern "C" void nn_set_gn_const(int on) { g_gn_const = on; }
extern "C" int nn_get_gn_const(void) { return g_gn_const; }

__global__ void gn_const_fill_k(float *mean, float *rstd, const float *run, int N, int G, float eps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * G) return;
    const int g = i % G;
    mean[i] = run ? run[g] : 0.f;
    rstd[i] = run ? 1.f / sqrtf(run[G + g] + eps) : 1.f;
}
extern "C" void nn_gn_const_fill(float *mean, float *rstd, const float *run, int N, int G, float eps) {
    gn_const_fill_k<<<nblk((size_t)N * G, 128), 128>>>(mean, rstd, run, N, G, eps);
    KCHECK();
}
/* run [mean G | var G] <- (1 - mom) run + mom (the batch's mean and variance per group: the samples' statistics pooled) */
__global__ void gn_ema_k(float *run, const float *mean, const float *rstd, int N, int G, float mom, float eps) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= G) return;
    double sm = 0, smm = 0, sv = 0;
    for (int n = 0; n < N; n++) { const double m = mean[n * G + g], r = rstd[n * G + g]; sm += m; smm += m * m; sv += 1.0 / (r * r) - eps; }
    const double m = sm / N, v = fmax(sv / N + smm / N - m * m, 0.0);
    run[g] = mom >= 1.f ? (float)m : (float)((1.0 - mom) * run[g] + mom * m);
    run[G + g] = mom >= 1.f ? (float)v : (float)((1.0 - mom) * run[G + g] + mom * v);
}
extern "C" void nn_gn_ema_update(float *run, const float *mean, const float *rstd, int N, int G, float mom, float eps) {
    gn_ema_k<<<nblk((size_t)G, 32), 32>>>(run, mean, rstd, N, G, mom, eps);
    KCHECK();
}
/* scaled weight standardisation (unet cfg.l0_norm 3) per output channel o, its fan-in n contiguous: wh = gain (w - mean) /
   (std sqrt(n) + eps) (std over the fan-in); the backward g <- g0 + J^T (g - g0), g0 the gradient before this backward (nullptr: 0) */
__device__ double ws_block_sum(double v, double *r) {
    r[threadIdx.x] = v;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) { if (threadIdx.x < o) r[threadIdx.x] += r[threadIdx.x + o]; __syncthreads(); }
    const double s = r[0];
    __syncthreads();
    return s;
}
__device__ void ws_moments(const float *w, int n, double *r, double *mu, double *sd) {
    double s = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += w[i];
    const double m = ws_block_sum(s, r) / n;
    double v = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { const double d = w[i] - m; v += d * d; }
    *mu = m; *sd = sqrt(ws_block_sum(v, r) / n);
}
__global__ void __launch_bounds__(256) ws_fwd_k(float *wh, const float *w, int n, float gain, float eps) {
    __shared__ double r[256];
    const float *wo = w + (size_t)blockIdx.x * n; float *ho = wh + (size_t)blockIdx.x * n;
    double mu, sd; ws_moments(wo, n, r, &mu, &sd);
    const double sc = gain / (sd * sqrt((double)n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) ho[i] = (float)((wo[i] - mu) * sc);
}
__global__ void __launch_bounds__(256) ws_bwd_k(float *g, const float *g0, const float *w, int n, float gain, float eps) {
    __shared__ double r[256];
    const size_t o = (size_t)blockIdx.x * n;
    double mu, sd; ws_moments(w + o, n, r, &mu, &sd);
    double a = 0, b = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { const double d = (double)g[o + i] - (g0 ? g0[o + i] : 0.f); a += d; b += d * (w[o + i] - mu); }
    a = ws_block_sum(a, r); b = ws_block_sum(b, r);
    const double s = sd * sqrt((double)n) + eps, c1 = gain / s, c2 = sd > 0 ? gain * b / (s * s * sqrt((double)n) * sd) : 0.0;   /* d/dw of std sqrt(n): (w - mu) / (sqrt(n) std) */
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float z = g0 ? g0[o + i] : 0.f;
        g[o + i] = (float)(z + c1 * ((double)g[o + i] - z - a / n) - c2 * (w[o + i] - mu));
    }
}
extern "C" void nn_ws_fwd(float *wh, const float *w, int cout, int n, float gain, float eps) { ws_fwd_k<<<cout, 256>>>(wh, w, n, gain, eps); KCHECK(); }
extern "C" void nn_ws_bwd(float *g, const float *g0, const float *w, int cout, int n, float gain, float eps) { ws_bwd_k<<<cout, 256>>>(g, g0, w, n, gain, eps); KCHECK(); }
