/* CUDA ops: fwd section of the former nn.cu */
#include "nn_common.cuh"

gnp_t to_gnp(const nn_gn_t *g) { return g && g->G ? gnp_mk(g->gamma, g->beta, g->mean, g->rstd, g->G) : gnp_t{}; }
/* slab-local GroupNorm registry (nn_lp.h): the statistics arrays that are per (sample, z-slab) and their geometry */
static struct { const float *mean; int zs, zo, ns, zhw; } g_slab[8];
static int g_nslab;
extern "C" void lp_gn_slab_reg(const float *mean, int zs, int zo, int ns, int zhw) {   /* ns <= 0: remove mean's entry */
    if (!mean) { g_nslab = 0; return; }
    for (int i = 0; i < g_nslab; i++) if (g_slab[i].mean == mean) { if (ns > 0) g_slab[i] = {mean, zs, zo, ns, zhw}; else g_slab[i] = g_slab[--g_nslab]; return; }
    if (ns <= 0) return;
    if (g_nslab == 8) { fprintf(stderr, "lp_gn_slab_reg: more than 8 slab statistics arrays\n"); abort(); }
    g_slab[g_nslab++] = {mean, zs, zo, ns, zhw};
}
extern "C" int lp_gn_slab_any(void) { return g_nslab > 0; }
extern "C" gnp_t gnp_mk(const float *gamma, const float *beta, const float *mean, const float *rstd, int G) {
    gnp_t p = {gamma, beta, mean, rstd, G};
    if (G) for (int i = 0; i < g_nslab; i++) if (g_slab[i].mean == mean) { p.zs = g_slab[i].zs; p.zo = g_slab[i].zo; p.ns = g_slab[i].ns; p.zhw = g_slab[i].zhw; break; }
    return p;
}
__global__ void prep_w16_k(const float *w, __half *wp, float *wsc, int Co, int Ci, int Cop, int Cip) {
    const int co = blockIdx.x;
    float am = 0.f;
    if (co < Co) for (int i = threadIdx.x; i < Ci * 27; i += blockDim.x) am = fmaxf(am, fabsf(w[(size_t)co * Ci * 27 + i]));
    __shared__ float red[32];
    for (int o = 16; o; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = am;
    __syncthreads();
    if (threadIdx.x < 32) { am = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f; for (int o = 16; o; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, o)); if (!threadIdx.x) red[0] = am; }
    __syncthreads();
    am = red[0];
    int e = 0;
    if (am > 0.f) frexpf(am / 16.f, &e);
    float inv = ldexpf(1.f, -e);
    if (threadIdx.x == 0) wsc[co] = ldexpf(1.f, e);
    for (int i = threadIdx.x; i < 27 * Cip; i += blockDim.x) {
        int t = i / Cip, ci = i % Cip;
        wp[((size_t)t * Cop + co) * Cip + ci] = __float2half_rn((co < Co && ci < Ci) ? w[((size_t)co * Ci + ci) * 27 + t] * inv : 0.f);
    }
}
