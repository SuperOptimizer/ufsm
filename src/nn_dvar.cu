/* train --density-nll (2026-11-27, densvar; upstream villa winding_targets.density_loss): a learned per-voxel log-variance s of the density
   head (one more output channel, its logit is s), trained with the heteroscedastic Gaussian NLL 0.5 (e^-s r^2 + s) of the density residual
   r = sigmoid(z_density) - t / 255 (the density code space), clamped to s in [DV_SMIN, DV_SMAX] (no gradient outside, as upstream's clamp),
   times the detached beta-NLL factor e^(beta s) (Seitzer et al.; upstream beta 0.5). The mask is the density's (m > 0, as 0 / 1: the packed
   weight is the other scheme), less the --density-min-gap-um voxels (target code >= thr; thr 0: none).
   The mean keeps its BCE (mode 0: the variance learns from the residual, the mean's gradient is unchanged), or (--density-nll-mean 1) the
   BCE gradient of each voxel is scaled by a = e^(-(1 - beta) s) / mean(a) (the beta-NLL's attenuation of the mean, renormalised over the
   window so the density head's total weight stays: only where it pulls changes).
   acc (per window, from the statistics pass): [0] supervised voxels, [1] sum of the NLL (unweighted by beta, the logged loss), [2] sum of a,
   [3] sum of s. All logits fp16 (the head slabs' planes); the gradients 16-bit in the loss producers' scale (GSH x the head plane scale) */
#include "nn_common.cuh"

#define DV_SMIN (-12.f)
#define DV_SMAX 6.f
__device__ __forceinline__ bool dv_on(unsigned m, unsigned t, int thr) { return m != 0u && !(thr > 0 && (int)t >= thr); }
__device__ __forceinline__ float dv_sig(float x) { return 1.f / (1.f + expf(-x)); }

__global__ void dvar_stats_k(const f16 *zd, const f16 *zs, const uint8_t *t, const uint8_t *m, size_t n, int thr, float beta, double *acc) {
    double a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        if (!dv_on(m[i], t[i], thr)) continue;
        const float s = fminf(DV_SMAX, fmaxf(DV_SMIN, __half2float(zs[i]))), r = dv_sig(__half2float(zd[i])) - t[i] * (1.f / 255.f);
        a0 += 1.0; a1 += 0.5 * ((double)expf(-s) * r * r + s); a2 += expf(-(1.f - beta) * s); a3 += s;
    }
    __shared__ double red[4][256];
    red[0][threadIdx.x] = a0; red[1][threadIdx.x] = a1; red[2][threadIdx.x] = a2; red[3][threadIdx.x] = a3; __syncthreads();
    for (int o = 128; o > 0; o >>= 1) { if ((int)threadIdx.x < o) for (int k = 0; k < 4; k++) red[k][threadIdx.x] += red[k][threadIdx.x + o]; __syncthreads(); }
    if (!threadIdx.x) for (int k = 0; k < 4; k++) atomicAdd(&acc[k], red[k][0]);
}
/* the log-variance gradient in place over zs (every voxel written: 0 off the mask), and (af != nullptr) each voxel's mean factor a / mean(a) */
template <typename GT> __global__ void dvar_grad_k(const f16 *zd, f16 *zs, const uint8_t *t, const uint8_t *m, size_t n, int thr, float beta, float lam,
                                                   const double *acc, float gscale, const float *hp, f16 *af) {
    gscale = hgs_mul(gscale, hp);
    const double cnt = acc[0]; const float inv = cnt >= 1.0 ? (float)(1.0 / cnt) : 0.f, abar = cnt >= 1.0 && acc[2] > 0 ? (float)(acc[2] / cnt) : 1.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float x = __half2float(zs[i]); float g = 0.f, a = 1.f;
        if (dv_on(m[i], t[i], thr)) {
            const float s = fminf(DV_SMAX, fmaxf(DV_SMIN, x)), r = dv_sig(__half2float(zd[i])) - t[i] * (1.f / 255.f);
            if (x > DV_SMIN && x < DV_SMAX) g = lam * inv * 0.5f * (1.f - expf(-s) * r * r) * expf(beta * s);
            a = expf(-(1.f - beta) * s) / abar;
        }
        ((GT *)zs)[i] = f2h<GT>(g * gscale);
        if (af) af[i] = __float2half(a);
    }
}
/* the density plane's 16-bit gradient times af (--density-nll-mean 1) */
template <typename GT> __global__ void dvar_scale_k(GT *g, const f16 *af, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) g[i] = f2h<GT>(h2f<GT>(g[i]) * __half2float(af[i]));
}
static unsigned dv_grid(size_t n) { const size_t b = (n + 255) / 256; return (unsigned)(b < 1024 ? (b ? b : 1) : 1024); }
extern "C" void nn_dvar_stats(const void *zd, const void *zs, const uint8_t *t, const uint8_t *m, size_t n, int thr, float beta, double *acc) {
    dvar_stats_k<<<dv_grid(n), 256>>>((const f16 *)zd, (const f16 *)zs, t, m, n, thr, beta, acc); KCHECK();
}
extern "C" void nn_dvar_grad(const void *zd, void *zs, const uint8_t *t, const uint8_t *m, size_t n, int thr, float beta, float lam, const double *acc, void *af) {
    if (!g_loss_g16) { fprintf(stderr, "nn_dvar_grad: needs 16-bit loss gradients (the head slabs' planes)\n"); abort(); }
    if (g_h16) dvar_grad_k<f16><<<dv_grid(n), 256>>>((const f16 *)zd, (f16 *)zs, t, m, n, thr, beta, lam, acc, GSH, GSHP, (f16 *)af);
    else dvar_grad_k<bf16><<<dv_grid(n), 256>>>((const f16 *)zd, (f16 *)zs, t, m, n, thr, beta, lam, acc, GSH, GSHP, (f16 *)af);
    KCHECK();
}
extern "C" void nn_dvar_scale(void *g, const void *af, size_t n) {
    if (g_h16) dvar_scale_k<f16><<<dv_grid(n), 256>>>((f16 *)g, (const f16 *)af, n); else dvar_scale_k<bf16><<<dv_grid(n), 256>>>((bf16 *)g, (const f16 *)af, n);
    KCHECK();
}
