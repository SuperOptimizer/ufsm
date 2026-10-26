/* xnet (xnet.h): fp16 channels-last layers for the external ink networks of ufsm ink-ext. */
#include "xnet.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static cudaStream_t g_st = nullptr;
static int g_dev_ok = 0, g_verbose = 0;
static size_t g_budget = (size_t)1 << 62;   /* device bytes in use + cached before the cache is dropped */
static double g_macs = 0;
extern "C" double xnet_macs(void) { return g_macs; }

extern "C" int xnet_init(int gpu) {
    if (g_dev_ok) return 0;
    if (cudaSetDevice(gpu) != cudaSuccess) { fprintf(stderr, "xnet: no GPU %d\n", gpu); return -1; }
    if (cudaStreamCreateWithFlags(&g_st, cudaStreamNonBlocking) != cudaSuccess) return -1;
    size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
    g_budget = fr > ((size_t)1 << 30) ? fr - ((size_t)1 << 30) : fr / 2;   /* free memory at start minus 1 GB headroom */
    g_dev_ok = 1; g_verbose = getenv("UFSM_XNET_VERBOSE") ? atoi(getenv("UFSM_XNET_VERBOSE")) : 0;
    return 0;
}
extern "C" void *xnet_stream(void) { return (void *)g_st; }
extern "C" void xnet_sync(void) { cudaStreamSynchronize(g_st); }
extern "C" int xnet_check(const char *where) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "xnet: %s: %s\n", where, cudaGetErrorString(e)); return -1; }
    return 0;
}

/* ---- size-keyed device memory cache ---- */
struct blk { void *p; size_t n; };
typedef struct { blk *b; size_t n, cap; } blist;
static blist g_free, g_used;
static size_t g_cached = 0, g_inuse = 0; static int g_warned = 0;
extern "C" void xnet_set_budget(size_t bytes) { g_budget = bytes; }
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static size_t round_up(size_t n) { const size_t q = n < (1 << 20) ? 4096 : (1 << 20); return (n + q - 1) / q * q; }
static void bl_push(blist *l, blk b) {
    if (l->n == l->cap) { l->cap = l->cap ? l->cap * 2 : 256; l->b = (blk *)realloc(l->b, l->cap * sizeof *l->b); }
    l->b[l->n++] = b;
}
static blk bl_take(blist *l, size_t i) { blk b = l->b[i]; l->b[i] = l->b[--l->n]; return b; }
extern "C" void *xmalloc_dev(size_t bytes) {
    const size_t n = round_up(bytes ? bytes : 1);
    pthread_mutex_lock(&g_mu);
    for (size_t i = 0; i < g_free.n; i++) if (g_free.b[i].n == n) {
        blk b = bl_take(&g_free, i); g_cached -= n; bl_push(&g_used, b); g_inuse += n;
        pthread_mutex_unlock(&g_mu); return b.p;
    }
    void *p = nullptr;
    if (g_inuse + g_cached + n > g_budget && g_cached) {   /* over budget: drop the cached blocks (WDDM pages instead of failing) */
        cudaStreamSynchronize(g_st);
        for (size_t i = 0; i < g_free.n; i++) cudaFree(g_free.b[i].p);
        g_free.n = 0; g_cached = 0;
    }
    if (g_inuse + n > g_budget && !g_warned) { g_warned = 1; fprintf(stderr, "xnet: %zu MB in use exceeds the %zu MB device budget (lower the batch)\n", (g_inuse + n) >> 20, g_budget >> 20); }
    if (cudaMalloc(&p, n) != cudaSuccess) {   /* free the cache and retry once */
        cudaGetLastError(); cudaStreamSynchronize(g_st);
        for (size_t i = 0; i < g_free.n; i++) cudaFree(g_free.b[i].p);
        g_free.n = 0; g_cached = 0;
        if (cudaMalloc(&p, n) != cudaSuccess) { cudaGetLastError(); pthread_mutex_unlock(&g_mu); fprintf(stderr, "xnet: cudaMalloc of %zu MB failed\n", n >> 20); return nullptr; }
    }
    bl_push(&g_used, blk{p, n}); g_inuse += n;
    pthread_mutex_unlock(&g_mu);
    return p;
}
extern "C" void xfree_dev(void *p) {
    if (!p) return;
    pthread_mutex_lock(&g_mu);
    for (size_t i = g_used.n; i-- > 0;) if (g_used.b[i].p == p) { blk b = bl_take(&g_used, i); g_inuse -= b.n; bl_push(&g_free, b); g_cached += b.n; break; }
    pthread_mutex_unlock(&g_mu);
}
extern "C" size_t xnet_mem_cached(void) { return g_cached; }
extern "C" size_t xnet_mem_used(void) { return g_inuse; }
extern "C" void xnet_trim(void) {
    cudaStreamSynchronize(g_st);
    pthread_mutex_lock(&g_mu);
    for (size_t i = 0; i < g_free.n; i++) cudaFree(g_free.b[i].p);
    g_free.n = 0; g_cached = 0;
    pthread_mutex_unlock(&g_mu);
}
extern "C" void xnet_mem_info(size_t *f, size_t *t) { *f = *t = 0; cudaMemGetInfo(f, t); }
extern "C" xt xt_new(int n, int d, int h, int w, int c) {
    xt t = {nullptr, n, d, h, w, c};
    t.p = xmalloc_dev(xt_numel(t) * 2);
    return t;
}
extern "C" void xt_free(xt *t) { if (t && t->p) { xfree_dev(t->p); t->p = nullptr; } }
extern "C" void xh2d(void *dst, const void *src, size_t bytes) { cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, g_st); cudaStreamSynchronize(g_st); }
extern "C" void xd2h(void *dst, const void *src, size_t bytes) { cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, g_st); cudaStreamSynchronize(g_st); }
extern "C" float *xdev_floats(const float *host, size_t n) {
    float *d = (float *)xmalloc_dev(n * sizeof(float));
    if (d) xh2d(d, host, n * sizeof(float));
    return d;
}

static inline unsigned nblk(size_t n, int t) { return (unsigned)((n + t - 1) / t); }
__device__ __forceinline__ float act_f(float v, int act) {
    act &= 15;
    return act == XA_RELU ? fmaxf(v, 0.f) : act == XA_LRELU ? (v < 0.f ? 0.01f * v : v) : act == XA_GELU ? 0.5f * v * (1.f + erff(v * 0.70710678118654752f))
         : act == XA_SILU ? v / (1.f + expf(-v)) : v;
}

/* ---- implicit-GEMM convolution ---- */
struct ConvP {
    const __half *x, *w, *res; const float *scale, *bias; __half *y;
    int Di, Hi, Wi, Ci, Do, Ho, Wo, Co;
    int kd, kh, kw, sd, sh, sw, pd, ph, pw, K, Kp;
    long long M; int act;
    const float *lnw, *lnb; float lneps;   /* XA_LN: channel LayerNorm of the whole output row (one block holds every channel) */
};
__device__ __forceinline__ unsigned smem_u32(const void *p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(void *dst, const void *src, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(smem_u32(dst)), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ void ldsm_x4(unsigned *r, const void *p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma16816(float *c, const unsigned *a, const unsigned *b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
/* shared tile [rows][BK halfs]: 16-byte chunk q of a row stored at q ^ f(row) (conflict-free ldmatrix of 8 rows) */
template <int BK> __device__ __forceinline__ int swz(int row, int q) { return row * BK + ((q ^ (BK == 32 ? (row >> 1) & 3 : row & 7)) << 3); }

enum { KPAD = 64 };
template <int BM, int BN, int WARPS_M, int WARPS_N, int BK, int NST>
__global__ void __launch_bounds__(WARPS_M * WARPS_N * 32) conv_igemm(ConvP p) {
    constexpr int NTH = WARPS_M * WARPS_N * 32, WM = BM / WARPS_M, WN = BN / WARPS_N, MI = WM / 16, NI = WN / 8;
    constexpr int QC = BK / 8, RS = NTH / QC, AP = BM / RS, BP = BN / RS;   /* chunks per row, rows per pass, passes */
    extern __shared__ __align__(128) unsigned char smem[];
    __half *sA = (__half *)smem, *sB = sA + NST * BM * BK;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm0 = (warp / WARPS_N) * WM, wn0 = (warp % WARPS_N) * WN;
    const long long m_base = (long long)blockIdx.x * BM; const int n_base = blockIdx.y * BN;
    const int q = tid % QC, r0 = tid / QC; int zb[AP], yb[AP], xb[AP]; long long nb[AP];
#pragma unroll
    for (int i = 0; i < AP; i++) {
        const long long m = m_base + r0 + i * RS;
        if (m < p.M) {
            long long t = m; const int xo = (int)(t % p.Wo); t /= p.Wo; const int yo = (int)(t % p.Ho); t /= p.Ho; const int zo = (int)(t % p.Do); t /= p.Do;
            zb[i] = zo * p.sd - p.pd; yb[i] = yo * p.sh - p.ph; xb[i] = xo * p.sw - p.pw; nb[i] = t * p.Di;
        } else { zb[i] = -(1 << 28); yb[i] = xb[i] = 0; nb[i] = 0; }
    }
    const int nk = p.Kp / BK;
    /* this thread's K position (tap tz, ty, tx and channel ci of k = ks BK + q 8), advanced by BK per loaded stage (loads go in ks order) */
    int ci = q * 8, tx = 0, ty = 0, tz = 0;
    while (ci >= p.Ci) { ci -= p.Ci; if (++tx == p.kw) { tx = 0; if (++ty == p.kh) { ty = 0; tz++; } } }
    auto load = [&](int slot, int ks) {
        const int k = ks * BK + q * 8; const bool kin = k < p.K;
        __half *A = sA + slot * BM * BK, *B = sB + slot * BN * BK;
#pragma unroll
        for (int i = 0; i < AP; i++) {
            const int row = r0 + i * RS;
            const int zi = zb[i] + tz, yi = yb[i] + ty, xi = xb[i] + tx;
            const bool ok = kin && zi >= 0 && zi < p.Di && yi >= 0 && yi < p.Hi && xi >= 0 && xi < p.Wi;
            const __half *src = ok ? p.x + ((((nb[i] + zi) * p.Hi + yi) * (long long)p.Wi + xi) * p.Ci + ci) : p.x;
            cp_async16(A + swz<BK>(row, q), src, ok ? 16 : 0);
        }
#pragma unroll
        for (int i = 0; i < BP; i++) {
            const int row = r0 + i * RS;
            cp_async16(B + swz<BK>(row, q), p.w + (size_t)(n_base + row) * p.Kp + ks * BK + q * 8, 16);
        }
        ci += BK;
        while (ci >= p.Ci) { ci -= p.Ci; if (++tx == p.kw) { tx = 0; if (++ty == p.kh) { ty = 0; tz++; } } }
    };
    float acc[MI][NI][4];
#pragma unroll
    for (int a = 0; a < MI; a++)
#pragma unroll
        for (int b = 0; b < NI; b++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[a][b][c] = 0.f;
    for (int s = 0; s < NST - 1; s++) { if (s < nk) load(s, s); cp_commit(); }
    for (int ks = 0; ks < nk; ks++) {
        cp_wait<NST - 2>(); __syncthreads();
        const int slot = ks % NST;
        if (ks + NST - 1 < nk) load((ks + NST - 1) % NST, ks + NST - 1);
        cp_commit();
        const __half *A = sA + slot * BM * BK, *B = sB + slot * BN * BK;
#pragma unroll
        for (int kk = 0; kk < BK / 16; kk++) {
            unsigned af[MI][4], bf[NI][2];
#pragma unroll
            for (int mi = 0; mi < MI; mi++) { const int row = wm0 + mi * 16 + (lane & 15); ldsm_x4(af[mi], A + swz<BK>(row, kk * 2 + (lane >> 4))); }
#pragma unroll
            for (int nj = 0; nj < NI / 2; nj++) {
                const int row = wn0 + nj * 16 + (lane & 7) + ((lane >> 4) << 3); unsigned r[4];
                ldsm_x4(r, B + swz<BK>(row, kk * 2 + ((lane >> 3) & 1)));
                bf[2 * nj][0] = r[0]; bf[2 * nj][1] = r[1]; bf[2 * nj + 1][0] = r[2]; bf[2 * nj + 1][1] = r[3];
            }
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int ni = 0; ni < NI; ni++) mma16816(acc[mi][ni], af[mi], bf[ni]);
        }
    }
    cp_wait<0>();
    const int g = lane >> 2, t4 = lane & 3;
    if constexpr (WARPS_N == 1) if (p.act & XA_LN) {   /* rows complete in this warp: v = conv + bias, LN over the Co channels, + res, act */
#pragma unroll
        for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const long long m = m_base + wm0 + mi * 16 + g + h * 8;
            float v[NI][2], s = 0.f;
#pragma unroll
            for (int ni = 0; ni < NI; ni++) { const int co = ni * 8 + t4 * 2; v[ni][0] = acc[mi][ni][2 * h] * p.scale[co] + p.bias[co]; v[ni][1] = acc[mi][ni][2 * h + 1] * p.scale[co + 1] + p.bias[co + 1]; s += v[ni][0] + v[ni][1]; }
            s += __shfl_xor_sync(0xffffffffu, s, 1); s += __shfl_xor_sync(0xffffffffu, s, 2);
            const float mean = s / (NI * 8); float q = 0.f;
#pragma unroll
            for (int ni = 0; ni < NI; ni++) { const float a = v[ni][0] - mean, b = v[ni][1] - mean; q += a * a + b * b; }
            q += __shfl_xor_sync(0xffffffffu, q, 1); q += __shfl_xor_sync(0xffffffffu, q, 2);
            const float rs = rsqrtf(q / (NI * 8) + p.lneps);
            if (m >= p.M) continue;
#pragma unroll
            for (int ni = 0; ni < NI; ni++) {
                const int co = ni * 8 + t4 * 2; const size_t o = (size_t)m * p.Co + co;
                float a = __half2float(__float2half_rn((v[ni][0] - mean) * rs * p.lnw[co] + p.lnb[co])), b = __half2float(__float2half_rn((v[ni][1] - mean) * rs * p.lnw[co + 1] + p.lnb[co + 1]));
                if (p.res) { const __half2 r = *(const __half2 *)(p.res + o); a += __low2float(r); b += __high2float(r); }
                *(__half2 *)(p.y + o) = __floats2half2_rn(act_f(a, p.act), act_f(b, p.act));
            }
        }
        return;
    }
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
    for (int ni = 0; ni < NI; ni++) {
        const int co = n_base + wn0 + ni * 8 + t4 * 2; if (co >= p.Co) continue;
        const float s0 = p.scale[co], s1 = p.scale[co + 1], b0 = p.bias[co], b1 = p.bias[co + 1];
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const long long m = m_base + wm0 + mi * 16 + g + h * 8; if (m >= p.M) continue;
            float v0 = acc[mi][ni][2 * h] * s0 + b0, v1 = acc[mi][ni][2 * h + 1] * s1 + b1;
            if (p.act & XA_GLU) {   /* channel pair (co, co + 1) = (gate x1_j, value x2_j): y[m][co / 2] = silu(x1) x2 */
                p.y[(size_t)m * (p.Co / 2) + co / 2] = __float2half_rn(v0 / (1.f + expf(-v0)) * v1);
                continue;
            }
            const size_t o = (size_t)m * p.Co + co;
            if (p.act & XA_POSTRES) { v0 = act_f(v0, p.act); v1 = act_f(v1, p.act); }
            if (p.res) { const __half2 r = *(const __half2 *)(p.res + o); v0 += __low2float(r); v1 += __high2float(r); }
            if (!(p.act & XA_POSTRES)) { v0 = act_f(v0, p.act); v1 = act_f(v1, p.act); }
            *(__half2 *)(p.y + o) = __floats2half2_rn(v0, v1);
        }
    }
}

extern "C" void xbn_fold(int co, const float *g, const float *b, const float *m, const float *v, float eps, const float *cb, float *scale, float *bias) {
    for (int c = 0; c < co; c++) {
        const double s = (double)g[c] / sqrt((double)v[c] + eps);
        scale[c] = (float)s; bias[c] = (float)((double)b[c] - (double)m[c] * s + (cb ? (double)cb[c] * s : 0.0));
    }
}

static int conv_upload(xconv *c, const float *wk /* [co][K] in GEMM order */, int co, const float *scale, const float *bias) {
    c->Kp = (c->K + KPAD - 1) / KPAD * KPAD; c->Cop = (co + 255) / 256 * 256;
    __half *hw = (__half *)calloc((size_t)c->Cop * c->Kp, sizeof(__half)); float *sc = (float *)malloc(co * sizeof(float)), *bi = (float *)malloc(co * sizeof(float));
    if (!hw || !sc || !bi) { free(hw); free(sc); free(bi); return -1; }
    float wmax = 0;
    for (int o = 0; o < co; o++) for (int k = 0; k < c->K; k++) { const float v = wk[(size_t)o * c->K + k]; hw[(size_t)o * c->Kp + k] = __float2half_rn(v); if (fabsf(v) > wmax) wmax = fabsf(v); }
    if (wmax > 60000.f) fprintf(stderr, "xnet: conv weight %g overflows fp16\n", wmax);
    for (int o = 0; o < co; o++) { sc[o] = scale ? scale[o] : 1.f; bi[o] = bias ? bias[o] : 0.f; }
    c->w = xmalloc_dev((size_t)c->Cop * c->Kp * 2); c->scale = (float *)xmalloc_dev(co * 4); c->bias = (float *)xmalloc_dev(co * 4);
    if (!c->w || !c->scale || !c->bias) { free(hw); free(sc); free(bi); return -1; }
    xh2d(c->w, hw, (size_t)c->Cop * c->Kp * 2); xh2d(c->scale, sc, co * 4); xh2d(c->bias, bi, co * 4);
    free(hw); free(sc); free(bi);
    return 0;
}

extern "C" int xconv_init(xconv *c, const float *w, int co, int ci, int kd, int kh, int kw, int sd, int sh, int sw, int pd, int ph, int pw,
                          const float *scale, const float *bias) {
    memset(c, 0, sizeof *c);
    if (co % 8) { fprintf(stderr, "xnet: conv with %d output channels (need a multiple of 8)\n", co); return -1; }
    c->co = co; c->kd = kd; c->kh = kh; c->sd = sd; c->sh = sh; c->sw = sw; c->pd = pd; c->ph = ph;
    float *wk;
    if (ci == 1) {   /* unfold along W: taps j < kw0 become 8 channels, the conv is kd x kh x 1 */
        if (kw > 8) { fprintf(stderr, "xnet: unfolded conv kw %d > 8\n", kw); return -1; }
        c->unfold = 1; c->kw0 = kw; c->pw0 = pw; c->ci = 8; c->kw = 1; c->pw = 0;
        c->K = kd * kh * 8; wk = (float *)calloc((size_t)co * c->K, sizeof(float)); if (!wk) return -1;
        for (int o = 0; o < co; o++) for (int a = 0; a < kd; a++) for (int b = 0; b < kh; b++) for (int j = 0; j < kw; j++)
            wk[(size_t)o * c->K + (a * kh + b) * 8 + j] = w[(((size_t)o * kd + a) * kh + b) * kw + j];
    } else {
        if (ci % 8) { fprintf(stderr, "xnet: conv with %d input channels (need 1 or a multiple of 8)\n", ci); return -1; }
        c->ci = ci; c->kw = kw; c->pw = pw; c->K = kd * kh * kw * ci;
        wk = (float *)malloc((size_t)co * c->K * sizeof(float)); if (!wk) return -1;
        for (int o = 0; o < co; o++) for (int i = 0; i < ci; i++) for (int a = 0; a < kd; a++) for (int b = 0; b < kh; b++) for (int e = 0; e < kw; e++)
            wk[(size_t)o * c->K + ((a * kh + b) * kw + e) * ci + i] = w[((((size_t)o * ci + i) * kd + a) * kh + b) * kw + e];
    }
    const int rc = conv_upload(c, wk, co, scale, bias); free(wk);
    return rc;
}
extern "C" int xtconv_init(xconv *c, const float *w, int ci, int co, const float *bias) {
    memset(c, 0, sizeof *c);
    if (ci % 8 || co % 8) { fprintf(stderr, "xnet: transposed conv %d -> %d channels\n", ci, co); return -1; }
    c->tconv = 1; c->co = 4 * co; c->ci = ci; c->kd = c->kh = c->kw = 1; c->sd = c->sh = c->sw = 1; c->K = ci;
    float *wk = (float *)malloc((size_t)4 * co * ci * sizeof(float)); if (!wk) return -1;
    for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) for (int o = 0; o < co; o++) for (int i = 0; i < ci; i++)
        wk[((size_t)(a * 2 + b) * co + o) * ci + i] = w[(((size_t)i * co + o) * 2 + a) * 2 + b];
    int rc = conv_upload(c, wk, 4 * co, nullptr, nullptr); free(wk);
    if (rc) return rc;
    float *tb = (float *)calloc(co, sizeof(float)); if (bias) memcpy(tb, bias, co * sizeof(float));
    c->tbias = xdev_floats(tb, co); free(tb);
    return c->tbias ? 0 : -1;
}
extern "C" int xtconv3_init(xconv *c, const float *w, int ci, int co, const float *bias) {
    memset(c, 0, sizeof *c);
    if (ci % 8 || co % 8) { fprintf(stderr, "xnet: 3D transposed conv %d -> %d channels\n", ci, co); return -1; }
    c->tconv = 2; c->co = 8 * co; c->ci = ci; c->kd = c->kh = c->kw = 1; c->sd = c->sh = c->sw = 1; c->K = ci;
    float *wk = (float *)malloc((size_t)8 * co * ci * sizeof(float)); if (!wk) return -1;
    for (int abc = 0; abc < 8; abc++) for (int o = 0; o < co; o++) for (int i = 0; i < ci; i++)
        wk[((size_t)abc * co + o) * ci + i] = w[((size_t)i * co + o) * 8 + abc];   /* torch [ci][co][2][2][2]: abc = (a * 2 + b) * 2 + e */
    int rc = conv_upload(c, wk, 8 * co, nullptr, nullptr); free(wk);
    if (rc) return rc;
    float *tb = (float *)calloc(co, sizeof(float)); if (bias) memcpy(tb, bias, co * sizeof(float));
    c->tbias = xdev_floats(tb, co); free(tb);
    return c->tbias ? 0 : -1;
}
extern "C" void xconv_free(xconv *c) { xfree_dev(c->w); xfree_dev(c->scale); xfree_dev(c->bias); xfree_dev(c->tbias); memset(c, 0, sizeof *c); }

__global__ void k_tshuffle(const __half *g, const float *bias, __half *y, int n, int h, int w, int co) {
    /* g [n][h][w][4][co] -> y [n][2h][2w][co] */
    const size_t total = (size_t)n * h * w * 4 * co;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const int o = (int)(i % co); size_t t = i / co; const int ab = (int)(t % 4); t /= 4; const int x = (int)(t % w); t /= w; const int yy = (int)(t % h); const size_t nn = t / h;
        const int a = ab >> 1, b = ab & 1;
        y[(((nn * 2 * h) + 2 * yy + a) * (size_t)(2 * w) + 2 * x + b) * co + o] = __float2half_rn(__half2float(g[i]) + bias[o]);
    }
}

__global__ void k_tshuffle3(const __half *g, const float *bias, __half *y, size_t nd, int h, int w, int co) {
    /* g [nd][h][w][8][co] -> y [2 nd][2h][2w][co] (nd = n d: plane 2 (n D + z) + a = n 2D + 2z + a) */
    const size_t total = nd * h * w * 8 * co;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const int o = (int)(i % co); size_t t = i / co; const int abc = (int)(t % 8); t /= 8; const int x = (int)(t % w); t /= w; const int yy = (int)(t % h); const size_t zz = t / h;
        const int a = abc >> 2, b = (abc >> 1) & 1, e = abc & 1;
        y[(((2 * zz + a) * (size_t)(2 * h) + 2 * yy + b) * (size_t)(2 * w) + 2 * x + e) * co + o] = __float2half_rn(__half2float(g[i]) + bias[o]);
    }
}

/* tile shapes: UFSM_XNET_CFG selects (default 0); per variant the tile follows the output channels (64 / 128 / >= 256) */
template <int BM, int BN, int WMn, int WNn, int BK, int NST> static void launch1(const ConvP &p, int co) {
    const size_t sm = (size_t)NST * (BM + BN) * BK * 2;
    static int attr = 0; if (!attr) { cudaFuncSetAttribute(conv_igemm<BM, BN, WMn, WNn, BK, NST>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); attr = 1; }
    dim3 g((unsigned)((p.M + BM - 1) / BM), (unsigned)((co + BN - 1) / BN));
    conv_igemm<BM, BN, WMn, WNn, BK, NST><<<g, WMn * WNn * 32, sm, g_st>>>(p);
}
static int g_cfg = -1;
static void conv_launch(const ConvP &p, int co) {
    if (g_cfg < 0) g_cfg = getenv("UFSM_XNET_CFG") ? atoi(getenv("UFSM_XNET_CFG")) : 0;
    switch (g_cfg) {
    case 1:   /* 4 warps of 64 x 64 / 64 x 32 */
        if (co <= 64) launch1<128, 64, 2, 2, 32, 3>(p, co); else launch1<128, 128, 2, 2, 32, 3>(p, co); break;
    case 2:   /* 128 x 256 tiles of 8 warps 64 x 64 for wide outputs */
        if (co <= 64) launch1<128, 64, 2, 2, 32, 3>(p, co); else if (co <= 128) launch1<128, 128, 2, 2, 32, 3>(p, co); else launch1<128, 256, 2, 4, 32, 3>(p, co); break;
    case 3:   /* 256 x 128 tiles of 8 warps 64 x 64 */
        if (co <= 64) launch1<256, 64, 4, 2, 32, 3>(p, co); else launch1<256, 128, 4, 2, 32, 3>(p, co); break;
    case 4:
        if (co <= 64) launch1<128, 64, 2, 2, 32, 4>(p, co); else if (co <= 128) launch1<128, 128, 2, 2, 32, 4>(p, co); else launch1<128, 256, 2, 4, 32, 4>(p, co); break;
    default:  /* 8 warps of 32 x 32 / 64 x 32; 4 warps of 32 x 32 for <= 32 outputs (no padded half tile) */
        if (p.act & XA_LN) {   /* one warp column holds all channels */
            if (co == 32) launch1<128, 32, 4, 1, 32, 3>(p, co); else if (co == 64) launch1<128, 64, 4, 1, 32, 3>(p, co);   /* (BK 64, 64-row and 4-stage variants measured slower) */
            else fprintf(stderr, "xnet: fused LayerNorm needs 32 or 64 outputs (%d)\n", co);
            break;
        }
        if (co <= 32) launch1<128, 32, 4, 1, 32, 3>(p, co);
        else if (co <= 64) launch1<128, 64, 4, 2, 32, 3>(p, co); else launch1<128, 128, 2, 4, 32, 3>(p, co); break;
    }
}
static xt conv_run(const xconv *c, xt x, const xt *res, int act, const float *lnw = nullptr, const float *lnb = nullptr, float lneps = 0.f) {
    xt in = x;
    if (c->unfold) in = xunfold(x, c->kw0, c->pw0);
    const int Do = (in.d + 2 * c->pd - c->kd) / c->sd + 1, Ho = (in.h + 2 * c->ph - c->kh) / c->sh + 1, Wo = (in.w + 2 * c->pw - c->kw) / c->sw + 1;
    xt y = xt_new(in.n, Do, Ho, Wo, act & XA_GLU ? c->co / 2 : c->co);
    if (in.c != c->ci) fprintf(stderr, "xnet: conv input has %d channels, expected %d\n", in.c, c->ci);
    ConvP p;
    p.x = (const __half *)in.p; p.w = (const __half *)c->w; p.res = res ? (const __half *)res->p : nullptr; p.scale = c->scale; p.bias = c->bias; p.y = (__half *)y.p;
    p.Di = in.d; p.Hi = in.h; p.Wi = in.w; p.Ci = in.c; p.Do = Do; p.Ho = Ho; p.Wo = Wo; p.Co = c->co;
    p.kd = c->kd; p.kh = c->kh; p.kw = c->kw; p.sd = c->sd; p.sh = c->sh; p.sw = c->sw; p.pd = c->pd; p.ph = c->ph; p.pw = c->pw;
    p.K = c->K; p.Kp = c->Kp; p.M = (long long)in.n * Do * Ho * Wo; p.act = act; p.lnw = lnw; p.lnb = lnb; p.lneps = lneps;
    g_macs += (double)p.M * c->co * c->K;
    if (g_verbose) fprintf(stderr, "xnet conv %dx%dx%d s%d%d%d ci %d co %d: in %dx%dx%dx%d out %dx%dx%d M %lld\n", c->kd, c->kh, c->kw, c->sd, c->sh, c->sw, c->ci, c->co, in.n, in.d, in.h, in.w, Do, Ho, Wo, p.M);
    if (p.M > 0) conv_launch(p, c->co);
    if (c->unfold) xt_free(&in);
    return y;
}
extern "C" xt xconv_fwd_ln(const xconv *c, xt x, const xt *res, int act, const float *lnw, const float *lnb, float eps) {
    return conv_run(c, x, res, act | XA_LN, lnw, lnb, eps);
}
extern "C" xt xconv_fwd(const xconv *c, xt x, const xt *res, int act) {
    if (!c->tconv) return conv_run(c, x, res, act);
    if (c->tconv == 2) {
        xt g = conv_run(c, x, nullptr, XA_NONE);   /* [n][d][h][w][8 co] */
        const int co = c->co / 8; xt y = xt_new(x.n, 2 * x.d, 2 * x.h, 2 * x.w, co);
        const size_t total = (size_t)x.n * x.d * x.h * x.w * 8 * co;
        k_tshuffle3<<<nblk(total, 256) > 65535 * 8 ? 65535 * 8 : nblk(total, 256), 256, 0, g_st>>>((const __half *)g.p, c->tbias, (__half *)y.p, (size_t)x.n * x.d, x.h, x.w, co);
        xt_free(&g);
        return y;
    }
    xt g = conv_run(c, x, nullptr, XA_NONE);   /* [n][1][h][w][4 co] */
    const int co = c->co / 4; xt y = xt_new(x.n, x.d, 2 * x.h, 2 * x.w, co);
    const size_t total = (size_t)x.n * x.d * x.h * x.w * 4 * co;
    k_tshuffle<<<nblk(total, 256) > 65535 * 8 ? 65535 * 8 : nblk(total, 256), 256, 0, g_st>>>((const __half *)g.p, c->tbias, (__half *)y.p, x.n * x.d, x.h, x.w, co);
    xt_free(&g);
    return y;
}

/* ---- GroupNorm / InstanceNorm ---- */
__global__ void k_chansum(const __half *x, int C, long long S, int nchunk, const float *gmean /* [n][C] per-channel group mean or nullptr */, double *out /* [n][C] */) {
    extern __shared__ float sh[];   /* C floats */
    const int n = blockIdx.y; const long long per = (S + nchunk - 1) / nchunk, s0 = blockIdx.x * per, s1 = s0 + per < S ? s0 + per : S;
    for (int c = threadIdx.x; c < C; c += blockDim.x) sh[c] = 0.f;
    __syncthreads();
    const __half *xb = x + (size_t)n * S * C; const float *gm = gmean ? gmean + (size_t)n * C : nullptr;
    /* each thread a fixed channel pair column when C is even: elements i = s * C + c */
    const long long total = (s1 - s0) * C;
    float accv[8] = {0}; int cslot = -1;
    if (C % 2 == 0 && blockDim.x % (C / 2) == 0 && C / 2 <= (int)blockDim.x) {
        const int pairs = C / 2, cp = threadIdx.x % pairs, rows = blockDim.x / pairs; const int c = cp * 2;
        float a0 = 0, a1 = 0; const float m0 = gm ? gm[c] : 0.f, m1 = gm ? gm[c + 1] : 0.f;
        for (long long s = s0 + threadIdx.x / pairs; s < s1; s += rows) {
            const __half2 v = *(const __half2 *)(xb + (size_t)s * C + c); const float f0 = __low2float(v) - m0, f1 = __high2float(v) - m1;
            if (gm) { a0 += f0 * f0; a1 += f1 * f1; } else { a0 += f0; a1 += f1; }
        }
        atomicAdd(&sh[c], a0); atomicAdd(&sh[c + 1], a1); cslot = 1;
    }
    if (cslot < 0) {
        for (long long i = threadIdx.x; i < total; i += blockDim.x) {
            const int c = (int)(i % C); const float f = __half2float(xb[(size_t)s0 * C + i]) - (gm ? gm[c] : 0.f);
            atomicAdd(&sh[c], gm ? f * f : f);
        }
    }
    (void)accv;
    __syncthreads();
    for (int c = threadIdx.x; c < C; c += blockDim.x) atomicAdd(&out[(size_t)n * C + c], (double)sh[c]);
}
__global__ void k_gn_mean(const double *sum, int n, int C, int G, long long S, float *gmean) {   /* per channel: its group's mean */
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n * C) return;
    const int nn = i / C, c = i % C, cpg = C / G, g = c / cpg; double s = 0;
    for (int k = 0; k < cpg; k++) s += sum[(size_t)nn * C + g * cpg + k];
    gmean[i] = (float)(s / ((double)S * cpg));
}
__global__ void k_gn_rstd(const double *sq, int n, int C, int G, long long S, float eps, float *rstd) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n * C) return;
    const int nn = i / C, c = i % C, cpg = C / G, g = c / cpg; double s = 0;
    for (int k = 0; k < cpg; k++) s += sq[(size_t)nn * C + g * cpg + k];
    rstd[i] = (float)(1.0 / sqrt(s / ((double)S * cpg) + eps));
}
__global__ void k_gn_apply(const __half *x, const float *gm, const float *rs, const float *gamma, const float *beta, const __half *res, int act, __half *y, int C, long long S, size_t total) {
    for (size_t i = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) * 2; i < total; i += (size_t)gridDim.x * blockDim.x * 2) {
        const int c = (int)(i % C); const size_t nn = i / ((size_t)S * C); const size_t q = nn * C + c;
        const __half2 v = *(const __half2 *)(x + i);
        float a = (__low2float(v) - gm[q]) * rs[q] * gamma[c] + beta[c], b = (__high2float(v) - gm[q + 1]) * rs[q + 1] * gamma[c + 1] + beta[c + 1];
        if (res) { const __half2 r = *(const __half2 *)(res + i); a += __low2float(r); b += __high2float(r); }
        *(__half2 *)(y + i) = __floats2half2_rn(act_f(a, act), act_f(b, act));
    }
}
extern "C" void xgn(xt x, int groups, const float *gamma, const float *beta, float eps, const xt *res, int act, xt y) {
    const int C = x.c, n = x.n; const long long S = (long long)x.d * x.h * x.w;
    double *acc = (double *)xmalloc_dev((size_t)n * C * sizeof(double) * 2); float *gm = (float *)xmalloc_dev((size_t)n * C * 4 * 2), *rs = gm + (size_t)n * C;
    cudaMemsetAsync(acc, 0, (size_t)n * C * sizeof(double) * 2, g_st);
    int nchunk = (int)((S + 4095) / 4096); if (nchunk > 512) nchunk = 512; if (nchunk < 1) nchunk = 1;
    const int th = (C / 2 <= 256 && 256 % (C / 2) == 0) ? 256 : (C / 2 <= 512 && 512 % (C / 2) == 0 ? 512 : 256);
    dim3 g(nchunk, n);
    k_chansum<<<g, th, C * sizeof(float), g_st>>>((const __half *)x.p, C, S, nchunk, nullptr, acc);
    k_gn_mean<<<nblk((size_t)n * C, 256), 256, 0, g_st>>>(acc, n, C, groups, S, gm);
    k_chansum<<<g, th, C * sizeof(float), g_st>>>((const __half *)x.p, C, S, nchunk, gm, acc + (size_t)n * C);
    k_gn_rstd<<<nblk((size_t)n * C, 256), 256, 0, g_st>>>(acc + (size_t)n * C, n, C, groups, S, eps, rs);
    const size_t total = xt_numel(x); unsigned nb = nblk(total / 2, 256); if (nb > 65535 * 4) nb = 65535 * 4;
    k_gn_apply<<<nb, 256, 0, g_st>>>((const __half *)x.p, gm, rs, gamma, beta, res ? (const __half *)res->p : nullptr, act, (__half *)y.p, C, S, total);
    xfree_dev(acc); xfree_dev(gm);
}

/* ---- data movement ---- */
__global__ void k_unfold(const __half *x, __half *y, long long rows, int W, int Wu, int kw, int pw) {
    const long long total = rows * Wu;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        const long long r = i / Wu; const int w = (int)(i % Wu);
        __align__(16) __half v[8];
        for (int j = 0; j < 8; j++) { const int xi = w - pw + j; v[j] = j < kw && xi >= 0 && xi < W ? x[r * W + xi] : __float2half(0.f); }
        *(uint4 *)(y + i * 8) = *(const uint4 *)v;
    }
}
extern "C" xt xunfold(xt x, int kw, int pw) {
    const int Wu = x.w + 2 * pw - kw + 1; xt y = xt_new(x.n, x.d, x.h, Wu, 8);
    const long long rows = (long long)x.n * x.d * x.h; unsigned nb = nblk((size_t)rows * Wu, 256); if (nb > 65535 * 8) nb = 65535 * 8;
    k_unfold<<<nb, 256, 0, g_st>>>((const __half *)x.p, (__half *)y.p, rows, x.w, Wu, kw, pw);
    return y;
}
__global__ void k_maxpool3s2(const __half *x, __half *y, long long planes, int H, int W, int Ho, int Wo, int C) {
    const long long total = planes * Ho * Wo * C;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        const int c = (int)(i % C); long long t = i / C; const int xo = (int)(t % Wo); t /= Wo; const int yo = (int)(t % Ho); const long long pl = t / Ho;
        float m = -INFINITY;
        for (int a = -1; a <= 1; a++) for (int b = -1; b <= 1; b++) {
            const int yi = yo * 2 + a, xi = xo * 2 + b; if (yi < 0 || yi >= H || xi < 0 || xi >= W) continue;
            m = fmaxf(m, __half2float(x[((pl * H + yi) * W + xi) * C + c]));
        }
        y[i] = __float2half(m);
    }
}
extern "C" xt xmaxpool2d_k3s2(xt x) {
    const int Ho = (x.h + 2 - 3) / 2 + 1, Wo = (x.w + 2 - 3) / 2 + 1; xt y = xt_new(x.n, x.d, Ho, Wo, x.c);
    const size_t total = xt_numel(y); unsigned nb = nblk(total, 256); if (nb > 65535 * 8) nb = 65535 * 8;
    k_maxpool3s2<<<nb, 256, 0, g_st>>>((const __half *)x.p, (__half *)y.p, (long long)x.n * x.d, x.h, x.w, Ho, Wo, x.c);
    return y;
}
__global__ void k_avgpool2(const __half *x, __half *y, long long planes, int H, int W, int Ho, int Wo, int C) {
    const long long total = planes * Ho * Wo * C;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        const int c = (int)(i % C); long long t = i / C; const int xo = (int)(t % Wo); t /= Wo; const int yo = (int)(t % Ho); const long long pl = t / Ho;
        const size_t b = ((pl * H + 2 * yo) * W + 2 * xo) * C + c;
        y[i] = __float2half((__half2float(x[b]) + __half2float(x[b + C]) + __half2float(x[b + (size_t)W * C]) + __half2float(x[b + (size_t)W * C + C])) * 0.25f);
    }
}
extern "C" xt xavgpool2d_k2(xt x) {
    xt y = xt_new(x.n, x.d, x.h / 2, x.w / 2, x.c);
    const size_t total = xt_numel(y); unsigned nb = nblk(total, 256); if (nb > 65535 * 8) nb = 65535 * 8;
    k_avgpool2<<<nb, 256, 0, g_st>>>((const __half *)x.p, (__half *)y.p, (long long)x.n * x.d, x.h, x.w, x.h / 2, x.w / 2, x.c);
    return y;
}
__global__ void k_avgpool3(const __half *x, __half *y, long long total, int D, int H, int W, int C) {
    const int Do = D / 2, Ho = H / 2, Wo = W / 2;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        const int c = (int)(i % C); long long t = i / C; const int xo = (int)(t % Wo); t /= Wo; const int yo = (int)(t % Ho); t /= Ho; const int zo = (int)(t % Do); const long long nn = t / Do;
        float a = 0.f;
        for (int dz = 0; dz < 2; dz++) for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++)
            a += __half2float(x[(((nn * D + 2 * zo + dz) * H + 2 * yo + dy) * (long long)W + 2 * xo + dx) * C + c]);
        y[i] = __float2half(a * 0.125f);
    }
}
extern "C" xt xavgpool3d_k2(xt x) {
    xt y = xt_new(x.n, x.d / 2, x.h / 2, x.w / 2, x.c);
    const size_t total = xt_numel(y); unsigned nb = nblk(total, 256); if (nb > 65535 * 8) nb = 65535 * 8;
    k_avgpool3<<<nb, 256, 0, g_st>>>((const __half *)x.p, (__half *)y.p, (long long)total, x.d, x.h, x.w, x.c);
    return y;
}
__global__ void k_depthmax(const __half *x, __half *y, int n, int D, long long HWC) {
    const long long total = (long long)n * HWC;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        const long long nn = i / HWC, r = i % HWC; __half m = x[(nn * D) * HWC + r];
        for (int d = 1; d < D; d++) m = __hmax(m, x[(nn * D + d) * HWC + r]);
        y[i] = m;
    }
}
extern "C" xt xdepthmax(xt x) {
    xt y = xt_new(x.n, 1, x.h, x.w, x.c); const long long HWC = (long long)x.h * x.w * x.c;
    unsigned nb = nblk((size_t)x.n * HWC, 256); if (nb > 65535 * 8) nb = 65535 * 8;
    k_depthmax<<<nb, 256, 0, g_st>>>((const __half *)x.p, (__half *)y.p, x.n, x.d, HWC);
    return y;
}
/* PyTorch area_pixel_compute_source_index (align_corners False) + the linear interpolation indices */
__device__ __forceinline__ void src_idx(int o, int in, int out, int &i0, int &i1, float &l) {
    if (in == out) { i0 = i1 = o; l = 0.f; return; }
    const float sc = (float)in / (float)out; float s = sc * (o + 0.5f) - 0.5f; if (s < 0.f) s = 0.f;
    i0 = (int)s; if (i0 > in - 1) i0 = in - 1; i1 = i0 + (i0 < in - 1 ? 1 : 0); l = s - (float)i0;
}
__global__ void k_resize(const __half *x, __half *y, int n, int Di, int Hi, int Wi, int Do, int Ho, int Wo, int C) {
    const int CV = C % 8 == 0 ? 8 : 1, nv = C / CV;
    const long long total = (long long)n * Do * Ho * Wo * nv;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        const int cv = (int)(i % nv); long long t = i / nv; const int xo = (int)(t % Wo); t /= Wo; const int yo = (int)(t % Ho); t /= Ho; const int zo = (int)(t % Do); const long long nn = t / Do;
        int z0, z1, y0, y1, x0, x1; float lz, ly, lx;
        src_idx(zo, Di, Do, z0, z1, lz); src_idx(yo, Hi, Ho, y0, y1, ly); src_idx(xo, Wi, Wo, x0, x1, lx);
        const float wz[2] = {1.f - lz, lz}, wy[2] = {1.f - ly, ly}, wx[2] = {1.f - lx, lx}; const int zz[2] = {z0, z1}, yy[2] = {y0, y1}, xx[2] = {x0, x1};
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) for (int e = 0; e < 2; e++) {
            const float wgt = wz[a] * wy[b] * wx[e];
            const size_t o = (((nn * Di + zz[a]) * Hi + yy[b]) * (size_t)Wi + xx[e]) * C + (size_t)cv * CV;
            if (CV == 8) { __align__(16) __half v[8]; *(uint4 *)v = *(const uint4 *)(x + o); for (int k = 0; k < 8; k++) acc[k] += wgt * __half2float(v[k]); }
            else acc[0] += wgt * __half2float(x[o]);
        }
        const size_t oo = (size_t)i * CV;
        if (CV == 8) { __align__(16) __half v[8]; for (int k = 0; k < 8; k++) v[k] = __float2half_rn(acc[k]); *(uint4 *)(y + oo) = *(const uint4 *)v; }
        else y[oo] = __float2half_rn(acc[0]);
    }
}
extern "C" xt xresize(xt x, int d, int h, int w) {
    xt y = xt_new(x.n, d, h, w, x.c);
    const size_t total = xt_numel(y) / (x.c % 8 == 0 ? 8 : 1); unsigned nb = nblk(total, 256); if (nb > 65535 * 8) nb = 65535 * 8;
    k_resize<<<nb, 256, 0, g_st>>>((const __half *)x.p, (__half *)y.p, x.n, x.d, x.h, x.w, d, h, w, x.c);
    return y;
}
extern "C" xt xconcat(xt a, xt b) {
    xt y = xt_new(a.n, a.d, a.h, a.w, a.c + b.c); const size_t rows = (size_t)a.n * a.d * a.h * a.w;
    cudaMemcpy2DAsync(y.p, (size_t)y.c * 2, a.p, (size_t)a.c * 2, (size_t)a.c * 2, rows, cudaMemcpyDeviceToDevice, g_st);
    cudaMemcpy2DAsync((char *)y.p + (size_t)a.c * 2, (size_t)y.c * 2, b.p, (size_t)b.c * 2, (size_t)b.c * 2, rows, cudaMemcpyDeviceToDevice, g_st);
    return y;
}
__global__ void k_input(const uint8_t *x, __half *y, long long per, long long total, const float *prm /* [n][4] lo hi a b */) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        const float *q = prm + (i / per) * 4; const float v = fminf(fmaxf((float)x[i], q[0]), q[1]);
        y[i] = __float2half_rn(v * q[2] + q[3]);
    }
}
extern "C" xt xinput_u8(const uint8_t *dev, int n, int d, int h, int w, const float *lo, const float *hi, const float *a, const float *b) {
    xt y = xt_new(n, d, h, w, 1); float *prm = (float *)malloc((size_t)n * 16);
    for (int i = 0; i < n; i++) { prm[4 * i] = lo[i]; prm[4 * i + 1] = hi[i]; prm[4 * i + 2] = a[i]; prm[4 * i + 3] = b[i]; }
    float *dp = xdev_floats(prm, (size_t)n * 4); free(prm);
    const long long per = (long long)d * h * w, total = per * n; unsigned nb = nblk((size_t)total, 256); if (nb > 65535 * 8) nb = 65535 * 8;
    k_input<<<nb, 256, 0, g_st>>>(dev, (__half *)y.p, per, total, dp);
    xfree_dev(dp);
    return y;
}
__global__ void k_head(const __half *x, const float *w, float bias, float *out, long long nvox, int C) {
    /* one warp per voxel */
    const long long v = (blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5; const int lane = threadIdx.x & 31;
    if (v >= nvox) return;
    float s = 0; for (int c = lane; c < C; c += 32) s += __half2float(x[v * C + c]) * w[c];
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (!lane) out[v] = s + bias;
}
extern "C" void xhead(xt x, const float *w_dev, float bias, float *out_dev) {
    const long long nvox = (long long)x.n * x.d * x.h * x.w;
    k_head<<<nblk((size_t)nvox * 32, 256), 256, 0, g_st>>>((const __half *)x.p, w_dev, bias, out_dev, nvox, x.c);
}
__global__ void k_attn_max(const __half *f, const float *a, __half *y, int n, int D, long long HW, int C) {
    const long long total = (long long)n * HW * C;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        const int c = (int)(i % C); const long long t = i / C, p = t % HW, nn = t / HW;
        float am = -INFINITY; for (int d = 0; d < D; d++) am = fmaxf(am, a[(nn * D + d) * HW + p]);
        float den = 0; for (int d = 0; d < D; d++) den += __expf(a[(nn * D + d) * HW + p] - am);
        float s = 0, mx = -INFINITY;
        for (int d = 0; d < D; d++) {
            const float fv = __half2float(f[((nn * D + d) * HW + p) * C + c]);
            s += fv * (__expf(a[(nn * D + d) * HW + p] - am) / den); mx = fmaxf(mx, fv);
        }
        y[(nn * HW + p) * 2 * C + c] = __float2half_rn(s); y[(nn * HW + p) * 2 * C + C + c] = __float2half_rn(mx);
    }
}
extern "C" xt xdepth_attn_max(xt f, const float *a_dev) {
    xt y = xt_new(f.n, 1, f.h, f.w, 2 * f.c); const long long HW = (long long)f.h * f.w;
    unsigned nb = nblk((size_t)f.n * HW * f.c, 256); if (nb > 65535 * 8) nb = 65535 * 8;
    k_attn_max<<<nb, 256, 0, g_st>>>((const __half *)f.p, a_dev, (__half *)y.p, f.n, f.d, HW, f.c);
    return y;
}
__global__ void k_hecate_collapse(const float *s, const float *l, const uint8_t *sup, int n, int D, long long HW, float zs, float *out) {
    const long long total = (long long)n * HW;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        const long long nn = i / HW, p = i % HW; const int half = D / 2; const bool ok = sup[i] != 0;
        float mx = -INFINITY;
        for (int d = 0; d < D; d++) { const float v = ok ? s[(nn * D + d) * HW + p] + zs * ((float)(d - half) / (float)half) : -1e4f; mx = fmaxf(mx, v); }
        float den = 0, num = 0;
        for (int d = 0; d < D; d++) {
            const float v = ok ? s[(nn * D + d) * HW + p] + zs * ((float)(d - half) / (float)half) : -1e4f;
            const float e = expf(v - mx); den += e; num += e * l[(nn * D + d) * HW + p];
        }
        out[i] = ok ? num / den : 0.f;   /* softmax * support renormalised: the plain softmax where supported, 0 elsewhere */
    }
}
extern "C" void xhecate_collapse(const float *s_dev, const float *l_dev, const uint8_t *sup_dev, int n, int d, int h, int w, float zs, float *out_dev) {
    const long long HW = (long long)h * w; unsigned nb = nblk((size_t)n * HW, 256); if (nb > 65535 * 8) nb = 65535 * 8;
    k_hecate_collapse<<<nb, 256, 0, g_st>>>(s_dev, l_dev, sup_dev, n, d, HW, zs, out_dev);
}
