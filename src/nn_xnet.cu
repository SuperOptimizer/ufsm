/* xnet (xnet.h): fp16 channels-last layers for the external ink networks of ufsm ink-ext. */
#include "xnet.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "lp_common.cuh"
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static cudaStream_t g_st = nullptr;
static int g_dev_ok = 0, g_verbose = 0, g_dev = 0;
static size_t g_budget = (size_t)1 << 62;   /* device bytes in use + cached before the cache is dropped */
static double g_macs = 0;
extern "C" double xnet_macs(void) { return g_macs; }

extern "C" int xnet_init(int gpu) {
    if (g_dev_ok) return 0;
    if (cudaSetDevice(gpu) != cudaSuccess) { fprintf(stderr, "xnet: no GPU %d\n", gpu); return -1; }
    g_dev = gpu;
    if (cudaStreamCreateWithFlags(&g_st, cudaStreamNonBlocking) != cudaSuccess) return -1;
    size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
    g_budget = fr > ((size_t)1 << 30) ? fr - ((size_t)1 << 30) : fr / 2;   /* free memory at start minus 1 GB headroom */
    g_dev_ok = 1; g_verbose = getenv("UFSM_XNET_VERBOSE") ? atoi(getenv("UFSM_XNET_VERBOSE")) : 0;
    return 0;
}
extern "C" void xnet_bind(void) { if (g_dev_ok) cudaSetDevice(g_dev); }
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
extern "C" void xt_free(xt *t) { if (t && t->p) { xfree_dev(t->p); t->p = nullptr; } if (t && t->q) { xfree_dev(t->q); t->q = nullptr; } }
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

/* ---- fake quantisation (UFSM_XNET_FQ=A,W: MX bits of the conv activations / weights, 0 off, 4 MX-fp4 e2m1, 8 MX-fp8 e4m3; one ue8m0
   scale per 32 channels of a voxel / 32 K of a weight row, round to nearest even): the accuracy of an MX conv path, run on the fp16 kernels.
   The products of such values are exact in fp32, so a real MX kernel differs only by the fp32 summation order. Convs whose input has a
   channel count not divisible by 32 (the unfolded 1-channel stems) stay fp16. ---- */
static int g_fq_a = -1, g_fq_w = 0, g_fq_seq = 0, g_fq_wseq = 0; static long long g_fq_mask = -1;   /* UFSM_XNET_FQ_MASK: hex bit i = conv i of a forward (xnet_seq_reset); mgpu: debug (UFSM_XNET_FQ accuracy probes; xnet is the single-threaded teacher runtime) */
extern "C" void xnet_seq_reset(void) { g_fq_seq = 0; }
static void fq_env(void) {
    if (g_fq_a >= 0) return;
    g_fq_a = 0; g_fq_w = 0; const char *e = getenv("UFSM_XNET_FQ");
    if (e && *e) { g_fq_a = atoi(e); const char *c = strchr(e, ','); g_fq_w = c ? atoi(c + 1) : g_fq_a; }
    const char *m = getenv("UFSM_XNET_FQ_MASK"); if (m && *m) g_fq_mask = strtoll(m, nullptr, 16);
}
static float e2m1_rn(float v) {   /* nearest of +-{0, .5, 1, 1.5, 2, 3, 4, 6}, ties to the even code, saturating */
    static const float g[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float a = fabsf(v); int b = 0; float bd = 1e30f;
    for (int i = 0; i < 8; i++) { const float d = fabsf(a - g[i]); if (d < bd || (d == bd && !(i & 1))) { bd = d; b = i; } }
    return copysignf(g[b], v);
}
/* e2m1_rn(v / d) exactly under fast math: |v| against the grid midpoints times d in double (products of <= 3 + 24 significant bits:
   exact), ties to the even grid index; saturates at 6 */
static float e2m1_div(float v, float d) {
    static const double M[7] = {.25, .75, 1.25, 1.75, 2.5, 3.5, 5.}; static const float G[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const double a = fabs((double)v); int b = 0;
    for (int i = 0; i < 7; i++) { const double t = M[i] * (double)d; if (a > t) { b = i + 1; continue; } if (a == t) b = (i & 1) ? i + 1 : i; break; }
    return copysignf(G[b], v);
}
static inline float div6(float a) { return (float)((double)a / 6.0); }   /* a / 6 correctly rounded to float (as __fdiv_rn) under fast math */
static float e4m3_rn(float v) {   /* e4m3 (bias 7, max 448, subnormals 2^-9 steps), ties to even */
    const float a = fabsf(v); if (a >= 448.f) return copysignf(448.f, v); if (a == 0.f) return v;
    int ex; frexpf(a, &ex); ex -= 1; if (ex < -6) ex = -6;   /* a in [2^ex, 2^(ex + 1)) */
    const float ulp = ldexpf(1.f, ex - 3); return copysignf(nearbyintf(a / ulp) * ulp, v);
}
/* bits 5: NVFP4-style, blocks of 16 with an e4m3 scale (amax / 6 rounded to e4m3, times gs: a per-row / per-tensor fp32 scale) */
static void fq_nv(float *v, int n, float gs) {
    for (int b0 = 0; b0 < n; b0 += 16) {
        float am = 0.f; for (int i = b0; i < b0 + 16 && i < n; i++) am = fmaxf(am, fabsf(v[i]));
        const float sc = e4m3_rn(gs == 1.f ? div6(am) : am / 6.f / gs) * gs; if (sc == 0.f) { for (int i = b0; i < b0 + 16 && i < n; i++) v[i] = 0.f; continue; }
        for (int i = b0; i < b0 + 16 && i < n; i++) v[i] = e2m1_div(v[i], sc) * sc;
    }
}
static void fq_host(float *v, int n, int bits) {   /* one block of n <= 32 values in place */
    float am = 0.f; for (int i = 0; i < n; i++) am = fmaxf(am, fabsf(v[i]));
    if (am == 0.f) return;
    if (bits == 5) { fq_nv(v, n, 1.f); return; }
    const float q = bits == 4 ? 6.f : 448.f; int e; frexpf(am / q, &e); if (ldexpf(1.f, e - 1) >= am / q) e -= 1;   /* 2^e >= am / q, smallest */
    for (int i = 0; i < n; i++) { const float x = ldexpf(v[i], -e); v[i] = ldexpf(bits == 4 ? e2m1_rn(x) : e4m3_rn(x), e); }
}
__global__ void k_fq(const __half *x, __half *y, long long nblk, int bits) {   /* blocks of 32 consecutive values (channels of a voxel) */
    const long long b = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (b >= nblk) return;
    float v[32]; float am = 0.f;
    for (int i = 0; i < 32; i++) { v[i] = __half2float(x[b * 32 + i]); am = fmaxf(am, fabsf(v[i])); }
    static constexpr float G[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    if (bits == 6) {   /* two blocks of 16, power-of-two scale (the MX kernel's ue8m0 format) */
        for (int h = 0; h < 2; h++) {
            float a = 0.f; for (int i = 16 * h; i < 16 * h + 16; i++) a = fmaxf(a, fabsf(v[i]));
            const int e = mx_exp(a, 1.f / 6.f); const float m = exp2i(-e), d = exp2i(e);
            for (int i = 16 * h; i < 16 * h + 16; i++) { const unsigned char c = cvt_e2m1x2(v[i] * m, 0.f) & 15; y[b * 32 + i] = __float2half_rn((c & 8 ? -1.f : 1.f) * G[c & 7] * d); }
        }
        return;
    }
    if (bits == 9) {   /* probe: each 16-block rotated by the normalised Hadamard H16 (H16 / 4, its own inverse), MX5-quantised, rotated back */
        for (int h = 0; h < 2; h++) {
            float *u = v + 16 * h;
            for (int len = 1; len < 16; len <<= 1) for (int i = 0; i < 16; i += 2 * len) for (int j = i; j < i + len; j++) { const float a = u[j], c = u[j + len]; u[j] = a + c; u[j + len] = a - c; }
            float a = 0.f; for (int i = 0; i < 16; i++) { u[i] *= .25f; a = fmaxf(a, fabsf(u[i])); }
            const unsigned char c8 = cvt_e4m3(__fdiv_rn(a, 6.f)); const int ex = (c8 >> 3) & 15, mt = c8 & 7;
            const float sc = ex ? ldexpf(1.f + mt / 8.f, ex - 7) : ldexpf(mt / 8.f, -6);
            for (int i = 0; i < 16; i++) { if (sc == 0.f) { u[i] = 0.f; continue; } const unsigned char c = cvt_e2m1x2(u[i] / sc, 0.f) & 15; u[i] = (c & 8 ? -1.f : 1.f) * G[c & 7] * sc; }
            for (int len = 1; len < 16; len <<= 1) for (int i = 0; i < 16; i += 2 * len) for (int j = i; j < i + len; j++) { const float a2 = u[j], c = u[j + len]; u[j] = a2 + c; u[j + len] = a2 - c; }
            for (int i = 0; i < 16; i++) y[b * 32 + 16 * h + i] = __float2half_rn(u[i] * .25f);
        }
        return;
    }
    if (bits == 5 || bits == 7) {   /* two blocks of 16, e4m3 scale amax / 6 (no tensor scale: IN-normalised activations, |x| < 2688); 7: the e4m3
                                       code of least squared error among amax / 6's -4 .. +2 (probe of a least-squares activation quantiser) */
        for (int h = 0; h < 2; h++) {
            float a = 0.f; for (int i = 16 * h; i < 16 * h + 16; i++) a = fmaxf(a, fabsf(v[i]));
            unsigned char c8 = cvt_e4m3(__fdiv_rn(a, 6.f));
            if (bits == 7 && a > 0.f) {
                float be = -1.f; int bc = c8;
                for (int cc = (int)c8 - 4; cc <= (int)c8 + 2; cc++) {
                    if (cc < 1 || cc > 126) continue;
                    const int ex = (cc >> 3) & 15, mt = cc & 7; const float sc = ex ? ldexpf(1.f + mt / 8.f, ex - 7) : ldexpf(mt / 8.f, -6); float e = 0.f;
                    for (int i = 16 * h; i < 16 * h + 16; i++) { const unsigned char c = cvt_e2m1x2(v[i] / sc, 0.f) & 15; const float q = (c & 8 ? -1.f : 1.f) * G[c & 7] * sc - v[i]; e += q * q; }
                    if (be < 0.f || e < be * (1.f - 1e-6f) || (cc == (int)c8 && e <= be)) { be = e; bc = cc; }
                }
                c8 = (unsigned char)bc;
            }
            const int ex = (c8 >> 3) & 15, mt = c8 & 7;
            const float sc = ex ? ldexpf(1.f + mt / 8.f, ex - 7) : ldexpf(mt / 8.f, -6);
            for (int i = 16 * h; i < 16 * h + 16; i++) {
                if (sc == 0.f) { y[b * 32 + i] = __float2half_rn(0.f); continue; }
                const unsigned char c = cvt_e2m1x2(v[i] / sc, 0.f) & 15; y[b * 32 + i] = __float2half_rn((c & 8 ? -1.f : 1.f) * G[c & 7] * sc);
            }
        }
        return;
    }
    const int e = mx_exp(am, bits == 4 ? 1.f / 6.f : 1.f / 448.f); const float m = exp2i(-e), d = exp2i(e);
    for (int i = 0; i < 32; i++) {
        float q;
        if (bits == 4) { const unsigned char c = cvt_e2m1x2(v[i] * m, 0.f) & 15; q = (c & 8 ? -1.f : 1.f) * G[c & 7]; }
        else { const unsigned char c = cvt_e4m3(v[i] * m); const int s = c >> 7, ex = (c >> 3) & 15, mt = c & 7; q = (s ? -1.f : 1.f) * (ex ? ldexpf(1.f + mt / 8.f, ex - 7) : ldexpf(mt / 8.f, -6)); }
        y[b * 32 + i] = __float2half_rn(q * d);
    }
}
/* ---- implicit-GEMM convolution ---- */
struct ConvP {
    const __half *x, *w, *res; const float *scale, *bias; __half *y;
    int Di, Hi, Wi, Ci, Do, Ho, Wo, Co;
    int kd, kh, kw, sd, sh, sw, pd, ph, pw, K, Kp;
    long long M; int act;
    const float *lnw, *lnb; float lneps;   /* XA_LN: channel LayerNorm of the whole output row (one block holds every channel) */
    double *st;   /* IN sums [n][Co][2] of the fp16 output (plain epilogue, no res / act) */
    const float *gm, *grs, *gg, *gb; int gact;   /* GNA: the input GroupNorm-applied on load as k_gn_apply ((x - gm) rs gamma + beta, act, fp16) */
};
__device__ __forceinline__ unsigned smem_u32(const void *p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(void *dst, const void *src, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(smem_u32(dst)), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ void xldsm_x4(unsigned *r, const void *p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma16816(float *c, const unsigned *a, const unsigned *b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
/* shared tile [rows][BK halfs]: 16-byte chunk q of a row stored at q ^ f(row) (conflict-free ldmatrix of 8 rows) */
template <int BK> __device__ __forceinline__ int swz(int row, int q) { return row * BK + ((q ^ (BK == 32 ? (row >> 1) & 3 : row & 7)) << 3); }

/* the IN sums of a thread's output values: for its two columns (co, co + 1), rows of one sample; lanes of equal t reduced, then atomics */
__device__ __forceinline__ void st_add(double *st, long long n, int Co, int co, float s0, float q0, float s1, float q1, int lane) {
#pragma unroll
    for (int o = 4; o < 32; o <<= 1) { s0 += __shfl_xor_sync(0xffffffffu, s0, o); q0 += __shfl_xor_sync(0xffffffffu, q0, o); s1 += __shfl_xor_sync(0xffffffffu, s1, o); q1 += __shfl_xor_sync(0xffffffffu, q1, o); }
    if (lane < 4 && co < Co) { double *d = st + ((size_t)n * Co + co) * 2; atomicAdd(d, (double)s0); atomicAdd(d + 1, (double)q0); atomicAdd(d + 2, (double)s1); atomicAdd(d + 3, (double)q1); }
}

enum { KPAD = 64 };
#ifndef XN_MINB   /* 8-warp tiles held to 2 blocks an SM (128 registers, was 136 at 1): reader3's 128 x 128 conv_igemm -22..-28 % (pinned-batch
                     ink-ext w035 A/B x3, byte-identical: register allocation only); 1 the old bound */
#define XN_MINB 2
#endif
template <int BM, int BN, int WARPS_M, int WARPS_N, int BK, int NST, bool GNA = false>
__global__ void __launch_bounds__(WARPS_M * WARPS_N * 32, WARPS_M * WARPS_N == 8 ? XN_MINB : 1) conv_igemm(ConvP p) {
    constexpr int NTH = WARPS_M * WARPS_N * 32, WM = BM / WARPS_M, WN = BN / WARPS_N, MI = WM / 16, NI = WN / 8;
    constexpr int QC = BK / 8, RS = NTH / QC, AP = BM / RS, BP = BN >= RS ? BN / RS : 1;   /* chunks per row, rows per pass, passes (BN < RS: one, part of the threads) */
    extern __shared__ __align__(128) unsigned char smem[];
    __half *sA = (__half *)smem, *sB = sA + NST * BM * BK;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm0 = (warp / WARPS_N) * WM, wn0 = (warp % WARPS_N) * WN;
    const long long m_base = (long long)blockIdx.x * BM; const int n_base = blockIdx.y * BN;
    const int q = tid % QC, r0 = tid / QC; int zb[AP], yb[AP], xb[AP]; long long nb[AP]; int ns[AP];
#pragma unroll
    for (int i = 0; i < AP; i++) {
        const long long m = m_base + r0 + i * RS;
        if (m < p.M) {
            long long t = m; const int xo = (int)(t % p.Wo); t /= p.Wo; const int yo = (int)(t % p.Ho); t /= p.Ho; const int zo = (int)(t % p.Do); t /= p.Do;
            zb[i] = zo * p.sd - p.pd; yb[i] = yo * p.sh - p.ph; xb[i] = xo * p.sw - p.pw; nb[i] = t * p.Di; ns[i] = (int)t;
        } else { zb[i] = -(1 << 28); yb[i] = xb[i] = 0; nb[i] = 0; ns[i] = 0; }
    }
    const int nk = p.Kp / BK;
    /* this thread's K position (tap tz, ty, tx and channel ci of k = ks BK + q 8), advanced by BK per loaded stage (loads go in ks order) */
    int ci = q * 8, tx = 0, ty = 0, tz = 0;
    while (ci >= p.Ci) { ci -= p.Ci; if (++tx == p.kw) { tx = 0; if (++ty == p.kh) { ty = 0; tz++; } } }
    unsigned gok = 0u; unsigned long long gci = 0ull;   /* GNA: per ring slot, this thread's in-bounds chunk bits (AP each) and its channel (16 bits each) */
    static_assert(!GNA || (NST * AP <= 32 && NST <= 4), "GNA ring metadata");
    auto load = [&](int slot, int ks) {
        const int k = ks * BK + q * 8; const bool kin = k < p.K;
        __half *A = sA + slot * BM * BK, *B = sB + slot * BN * BK;
        unsigned okb = 0u;
#pragma unroll
        for (int i = 0; i < AP; i++) {
            const int row = r0 + i * RS;
            const int zi = zb[i] + tz, yi = yb[i] + ty, xi = xb[i] + tx;
            const bool ok = kin && zi >= 0 && zi < p.Di && yi >= 0 && yi < p.Hi && xi >= 0 && xi < p.Wi;
            const __half *src = ok ? p.x + ((((nb[i] + zi) * p.Hi + yi) * (long long)p.Wi + xi) * p.Ci + ci) : p.x;
            cp_async16(A + swz<BK>(row, q), src, ok ? 16 : 0); okb |= (unsigned)ok << i;
        }
        if constexpr (GNA) {
            gok = (gok & ~(((1u << AP) - 1u) << (slot * AP))) | okb << (slot * AP);
            gci = (gci & ~(0xffffull << (slot * 16))) | (unsigned long long)ci << (slot * 16);
        }
#pragma unroll
        for (int i = 0; i < BP; i++) {
            const int row = r0 + i * RS;
            if (BN >= RS || row < BN) cp_async16(B + swz<BK>(row, q), p.w + (size_t)(n_base + row) * p.Kp + ks * BK + q * 8, 16);
        }
        ci += BK;
        while (ci >= p.Ci) { ci -= p.Ci; if (++tx == p.kw) { tx = 0; if (++ty == p.kh) { ty = 0; tz++; } } }
    };
    auto norm = [&](int slot) {   /* GNA: normalise this thread's own landed chunks of the slot in place (zero padding stays 0) */
        if constexpr (GNA) {
            __half *A = sA + slot * BM * BK; const int rci = (int)(gci >> (slot * 16) & 0xffffull); const unsigned okb = gok >> (slot * AP);
#pragma unroll
            for (int i = 0; i < AP; i++) {
                if (!(okb >> i & 1u)) continue;
                uint4 *ap = (uint4 *)(A + swz<BK>(r0 + i * RS, q)); uint4 val = *ap;
                __half *hv = (__half *)&val; const int q0 = ns[i] * p.Ci + rci;
#pragma unroll
                for (int k = 0; k < 8; k++) hv[k] = __float2half_rn(act_f((__half2float(hv[k]) - p.gm[q0 + k]) * p.grs[q0 + k] * p.gg[rci + k] + p.gb[rci + k], p.gact));
                *ap = val;
            }
        }
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
        cp_wait<NST - 2>(); norm(ks % NST); __syncthreads();   /* (a thread's own cp.async data is visible to it after the wait) */
        const int slot = ks % NST; const bool ld = ks + NST - 1 < nk;
        if (ld) load((ks + NST - 1) % NST, ks + NST - 1);
        cp_commit();
        const __half *A = sA + slot * BM * BK, *B = sB + slot * BN * BK;
#pragma unroll
        for (int kk = 0; kk < BK / 16; kk++) {
            unsigned af[MI][4], bf[NI][2];
#pragma unroll
            for (int mi = 0; mi < MI; mi++) { const int row = wm0 + mi * 16 + (lane & 15); xldsm_x4(af[mi], A + swz<BK>(row, kk * 2 + (lane >> 4))); }
#pragma unroll
            for (int nj = 0; nj < NI / 2; nj++) {
                const int row = wn0 + nj * 16 + (lane & 7) + ((lane >> 4) << 3); unsigned r[4];
                xldsm_x4(r, B + swz<BK>(row, kk * 2 + ((lane >> 3) & 1)));
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
    if (p.st) {   /* the IN path: y = acc scale + bias (fp16), with the per-(sample, channel) sums of the stored values */
        const long long S = (long long)p.Do * p.Ho * p.Wo, nw = (m_base + wm0) / S;
#pragma unroll
        for (int ni = 0; ni < NI; ni++) {
            const int co = n_base + wn0 + ni * 8 + t4 * 2; const bool cok = co < p.Co;
            const float s0 = cok ? p.scale[co] : 0.f, s1 = cok ? p.scale[co + 1] : 0.f, b0 = cok ? p.bias[co] : 0.f, b1 = cok ? p.bias[co + 1] : 0.f;
            float a0 = 0.f, a1 = 0.f, c0 = 0.f, c1 = 0.f;
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const long long m = m_base + wm0 + mi * 16 + g + h * 8; if (m >= p.M || !cok) continue;
                const __half2 o = __floats2half2_rn(acc[mi][ni][2 * h] * s0 + b0, acc[mi][ni][2 * h + 1] * s1 + b1);
                *(__half2 *)(p.y + (size_t)m * p.Co + co) = o;
                const float v0 = __low2float(o), v1 = __high2float(o); const long long nn = m / S;
                if (nn == nw) { a0 += v0; c0 += v0 * v0; a1 += v1; c1 += v1 * v1; }
                else { double *d = p.st + ((size_t)nn * p.Co + co) * 2; atomicAdd(d, (double)v0); atomicAdd(d + 1, (double)v0 * v0); atomicAdd(d + 2, (double)v1); atomicAdd(d + 3, (double)v1 * v1); }
            }
            if (nw * S < p.M) st_add(p.st, nw, p.Co, co, a0, c0, a1, c1, lane);
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

/* the MX-fp4 weight pack (see conv_mx): rows padded to Cop (multiple of 64), chunks to an even count; ue4m3 (5): per-row scale gs =
   amax / (6 448) so every block scale fits e4m3, values v / (e4m3(amax_b / 6 / gs) gs).
   UFSM_XNET_MXQ (xnet_set_mxq; default 0 = the amax / 6 scales above): 1 = per 16-K block the scale of least squared error among the
   e4m3 codes -4 .. +2 around amax_b / 6 (5) / the exponents e - 1, e (6); 2 = 1 plus, for 5, the row scale gs 2^(j / 8), j 0..7 (gs = amax_row / 2688 in double), of
   least row error (aligns the blocks' e4m3 grid). A candidate replaces an earlier one only when 1e-9 better (fast-math sums agree). No data: the weights alone. UFSM_XNET_MX_SKIP=i,j-k: convs (initialisation order of
   a net, xnet_wseq_reset) kept fp16. UFSM_XNET_MXQ_LOG=1: each packed conv's relative weight error (old / new). */
static int g_mx_env = -1, g_mxq = -1, g_mxq_log = 0;   /* mgpu: env */
static char g_mx_skip[4096]; static int g_mx_skip_set = 0;   /* mgpu: env */
extern "C" void xnet_set_mx(int sf) { g_mx_env = sf; }
extern "C" void xnet_set_mxq(int mode) { g_mxq = mode; }
extern "C" void xnet_set_mx_skip(const char *list) { snprintf(g_mx_skip, sizeof g_mx_skip, "%s", list ? list : ""); g_mx_skip_set = 1; }
extern "C" void xnet_wseq_reset(void) { g_fq_wseq = 0; }
static void mxq_env(void) {
    if (g_mxq < 0) { const char *e = getenv("UFSM_XNET_MXQ"); g_mxq = e && *e ? atoi(e) : 0; const char *l = getenv("UFSM_XNET_MXQ_LOG"); g_mxq_log = l && *l ? atoi(l) : 0; }
    if (!g_mx_skip_set) { const char *e = getenv("UFSM_XNET_MX_SKIP"); snprintf(g_mx_skip, sizeof g_mx_skip, "%s", e ? e : ""); g_mx_skip_set = 1; }
}
static int mx_skipped(int seq) {   /* seq in the list "i,j-k,..." */
    const char *p = g_mx_skip;
    while (*p) {
        char *e; const long a = strtol(p, &e, 10); if (e == p) break; long b = a; p = e;
        if (*p == '-') { b = strtol(p + 1, &e, 10); p = e; }
        if (seq >= a && seq <= b) return 1;
        while (*p == ',' || *p == ' ') p++;
    }
    return 0;
}
static float g_e4[127];   /* e4m3 code -> value (0 .. 126 = 448), filled once before any pack thread; mgpu: process */
static float e4val(int cc) { const int E = (cc >> 3) & 15, M = cc & 7; return E ? ldexpf(1.f + M / 8.f, E - 7) : ldexpf(M / 8.f, -6); }
static int e4code(float sv) {   /* sv an e4m3 value -> its code */
    if (g_e4[126] == 0.f) for (int cc = 0; cc < 127; cc++) g_e4[cc] = e4val(cc);
    int lo = 0, hi = 126; while (lo < hi) { const int m = (lo + hi) / 2; if (g_e4[m] < sv) lo = m + 1; else hi = m; }
    return lo;
}
static double blk_err(const float *v, float d, int sf) {   /* squared error of one 16-block quantised with scale d (exact rounding) */
    double e = 0.0;
    for (int i = 0; i < 16; i++) { const double q = (double)(d > 0.f ? (sf == 6 ? e2m1_rn(v[i] / d) : e2m1_div(v[i], d)) : 0.f) * d - v[i]; e += q * q; }
    return e;
}
/* the block scale: sf 6 the exponent e (d = 2^e), sf 5 the e4m3 code (d = e4[code] gs); mode 0: from amax, else least squared error */
static int blk_scale(const float *v, float a, float gs, int sf, int mode, double *err) {
    if (sf == 6) {
        int e = 0; if (a > 0.f) { frexpf(a / 6.f, &e); if (ldexpf(1.f, e - 1) >= a / 6.f) e -= 1; } else e = -127;
        if (e < -127) e = -127; if (e > 127) e = 127;
        if (mode && a > 0.f) {
            double b = blk_err(v, ldexpf(1.f, e), 6); int be = e;
            if (e > -127) { const double t = blk_err(v, ldexpf(1.f, e - 1), 6); if (t < b * (1.0 - 1e-9)) { b = t; be = e - 1; } }
            e = be; if (err) *err = b;
        } else if (err) *err = blk_err(v, ldexpf(1.f, e), 6);
        return e;
    }
    const float sv = e4m3_rn(mode ? (float)((double)a / 6.0 / gs) : a / 6.f / gs); const int c0 = e4code(sv);   /* (mode >= 1: in double, as the reference under fast math) */
    double b = blk_err(v, sv * gs, 5); int bc = c0;
    if (mode && a > 0.f)
        for (int cc = c0 - 4; cc <= c0 + 2; cc++) {
            if (cc < 1 || cc > 126 || cc == c0) continue;
            const double t = blk_err(v, g_e4[cc] * gs, 5); if (t < b * (1.0 - 1e-9)) { b = t; bc = cc; }
        }
    if (err) *err = b;
    return bc;
}
typedef struct { const float *wk; int K, co, nch, sf, mode; size_t rch; uint8_t *q, *sb; float *gs; double *err; int Cop; int next; pthread_mutex_t mu; } mxpk;
static void pack_row(const mxpk *P, int o) {   /* row o: its scales and nibbles; err[3 o ..]: old / new squared error, sum of squares */
    const int K = P->K, sf = P->sf, mode = P->mode, nb = P->nch * 2; const size_t rch = P->rch;
    const float *r = o < P->co ? P->wk + (size_t)o * K : nullptr; float am = 0.f; double wss = 0.0, e_old = 0.0, e_new = 0.0;
    if (r) for (int k = 0; k < K; k++) { am = fmaxf(am, fabsf(r[k])); wss += (double)r[k] * r[k]; }
    const float gs0 = sf == 5 && am > 0.f ? (mode ? (float)((double)am / 2688.0) : am / (6.f * 448.f)) : 1.f;
    float g = gs0;   /* the row scale: gs0, or (mode 2, sf 5) the best of gs0 2^(j / 8) */
    if (mode >= 2 && sf == 5 && r) {
        double best = -1.0; float vb[16];
        for (int j = 0; j < 8; j++) {
            static const float R8[8] = {1.f, 1.0905077f, 1.1892071f, 1.2968396f, 1.4142135f, 1.5422108f, 1.6817929f, 1.8340081f};   /* 2^(j / 8) */
            const float gj = gs0 * R8[j]; double tot = 0.0;
            for (int bb = 0; bb < nb; bb++) {
                float a = 0.f; for (int i = 0; i < 16; i++) { const int k = bb * 16 + i; vb[i] = k < K ? r[k] : 0.f; a = fmaxf(a, fabsf(vb[i])); }
                double e; blk_scale(vb, a, gj, 5, mode, &e); tot += e;
            }
            if (best < 0.0 || tot < best * (1.0 - 1e-9)) { best = tot; g = gj; }
        }
    }
    P->gs[o] = g;
    for (int b = 0; b < nb; b++) {   /* 16-K blocks */
        float vb[16], a = 0.f;
        for (int i = 0; i < 16; i++) { const int k = b * 16 + i; vb[i] = r && k < K ? r[k] : 0.f; a = fmaxf(a, fabsf(vb[i])); }
        float d; uint8_t code; double eb;
        if (g_mxq_log && r) { double e0; blk_scale(vb, a, gs0, sf, 0, &e0); e_old += e0; }
        if (sf == 6) { const int e = blk_scale(vb, a, 1.f, 6, mode, &eb); code = (uint8_t)(e + 127); d = ldexpf(1.f, e); }
        else { const int cc = blk_scale(vb, a, g, 5, mode, &eb); code = (uint8_t)cc; d = g_e4[cc] * g; }
        if (r) e_new += eb;
        uint8_t *qb = P->q + ((size_t)o * rch + b / 2) * 16 + (b & 1) * 8;
        for (int i = 0; i < 16; i++) {
            const float rq = d > 0.f ? (sf == 6 ? e2m1_rn(vb[i] / d) : e2m1_div(vb[i], d)) : 0.f;   /* (sf 6: d a power of two, exact) */
            static const float G[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f}; int nib = 0; for (int gg = 0; gg < 8; gg++) if (G[gg] == fabsf(rq)) nib = gg;
            if (rq < 0.f) nib |= 8;
            qb[i / 2] |= (uint8_t)(nib << ((i & 1) * 4));
        }
        P->sb[((size_t)o * rch + b / 2) * 4 + (b & 1)] = code;
    }
    P->err[3 * o] = e_old; P->err[3 * o + 1] = e_new; P->err[3 * o + 2] = wss;
}
static void *pack_worker(void *a) {
    mxpk *P = (mxpk *)a;
    for (;;) {
        pthread_mutex_lock(&P->mu); const int o0 = P->next; P->next += 8; pthread_mutex_unlock(&P->mu);
        if (o0 >= P->Cop) return nullptr;
        for (int o = o0; o < o0 + 8 && o < P->Cop; o++) pack_row(P, o);
    }
}
static int mx_pack1(xconv *c, const float *wk, int co, int sf, int wseq);
/* UFSM_XNET_MXROT=1: weights and activations of every MX conv H16-rotated per 16-block (exact in real arithmetic: block dot products are invariant) */
static int g_mxrot = -1;   /* mgpu: env */
static int g_mxa = -1;   /* UFSM_XNET_MXA=1: least-squares activation block scales (k_xq MODE bit 1); mgpu: env */
extern "C" void xnet_set_mxrot(int on) { g_mxrot = on; }
extern "C" void xnet_set_mxa(int on) { g_mxa = on; }
static int mx_pack(xconv *c, const float *wk, int co, int sf, int wseq) {
    if (g_mxrot < 0) { const char *e = getenv("UFSM_XNET_MXROT"); g_mxrot = e && *e ? atoi(e) : 0; }
    float *wr = nullptr;
    if (g_mxrot && c->K % 16 == 0) {
        wr = (float *)malloc((size_t)co * c->K * sizeof(float)); if (!wr) return -1;
        memcpy(wr, wk, (size_t)co * c->K * sizeof(float));
        for (size_t b = 0; b < (size_t)co * c->K; b += 16) {
            float *u = wr + b; for (int len = 1; len < 16; len <<= 1) for (int i = 0; i < 16; i += 2 * len) for (int j = i; j < i + len; j++) { const float a = u[j], cc = u[j + len]; u[j] = a + cc; u[j + len] = a - cc; }
            for (int i = 0; i < 16; i++) u[i] *= .25f;
        }
        wk = wr; c->mxrot |= 1;
    }
    if (g_mxa < 0) { const char *e = getenv("UFSM_XNET_MXA"); g_mxa = e && *e ? atoi(e) : 0; }
    if (g_mxa) c->mxrot |= 2;
    const int rc = mx_pack1(c, wk, co, sf, wseq); free(wr);
    return rc;
}
static int mx_pack1(xconv *c, const float *wk, int co, int sf, int wseq) {
    const int nch = (c->K + 31) / 32, nst = (nch + 1) / 2, Cop = (co + 63) / 64 * 64; const size_t rch = (size_t)nst * 2;
    uint8_t *q = (uint8_t *)calloc((size_t)Cop * rch * 16, 1), *sb = (uint8_t *)calloc((size_t)Cop * rch * 4, 1); float *gs = (float *)malloc(Cop * sizeof(float));
    double *err = (double *)calloc((size_t)Cop * 3, sizeof(double));
    if (!q || !sb || !gs || !err) { free(q); free(sb); free(gs); free(err); return -1; }
    e4code(1.f);
    const int mode = g_mxq; double e_old = 0.0, e_new = 0.0, wss = 0.0;
    mxpk P = {wk, c->K, co, nch, sf, mode, rch, q, sb, gs, err, Cop, 0, PTHREAD_MUTEX_INITIALIZER};
    /* rows in parallel when the search costs (mode >= 1: ~50x the amax pack); rows are independent, so the result does not depend on it */
    const long work = (long)Cop * c->K * (mode >= 2 ? 8 : mode ? 1 : 0); int nth = work > (1L << 20) ? 16 : 1;
    if (nth > 1) { pthread_t th[16]; int n = 0; for (; n < nth; n++) if (pthread_create(&th[n], nullptr, pack_worker, &P)) break; if (!n) pack_worker(&P); for (int t = 0; t < n; t++) pthread_join(th[t], nullptr); }
    else pack_worker(&P);
    for (int o = 0; o < Cop; o++) { e_old += err[3 * o]; e_new += err[3 * o + 1]; wss += err[3 * o + 2]; }
    free(err);
    if (g_mxq_log) fprintf(stderr, "xnet mxq conv %d: %dx%dx%d ci %d co %d K %d: rel weight err %.5f -> %.5f (mode %d, sf %d)\n", wseq, c->kd, c->kh, c->kw, c->ci, co, c->K,
                           sqrt(e_old / (wss + 1e-30)), sqrt(e_new / (wss + 1e-30)), mode, sf);
    c->mxq = xmalloc_dev((size_t)Cop * rch * 16); c->mxs = xmalloc_dev((size_t)Cop * rch * 4); c->mxgs = (float *)xmalloc_dev(Cop * sizeof(float));
    int rc = c->mxq && c->mxs && c->mxgs ? 0 : -1;
    if (!rc) { xh2d(c->mxq, q, (size_t)Cop * rch * 16); xh2d(c->mxs, sb, (size_t)Cop * rch * 4); xh2d(c->mxgs, gs, Cop * sizeof(float)); c->mxsf = sf; c->nch = nch; }
    if (!rc) {   /* the halo kernel's step scales: per (row, chunk pair) the four 16-K block scales in one word */
        uint8_t *s2 = (uint8_t *)malloc((size_t)Cop * rch * 2);
        for (size_t r = 0; r < (size_t)Cop * rch / 2; r++) { s2[4 * r] = sb[8 * r]; s2[4 * r + 1] = sb[8 * r + 1]; s2[4 * r + 2] = sb[8 * r + 4]; s2[4 * r + 3] = sb[8 * r + 5]; }
        c->mxs2 = xmalloc_dev((size_t)Cop * rch * 2); rc = c->mxs2 ? 0 : -1; if (!rc) xh2d(c->mxs2, s2, (size_t)Cop * rch * 2); free(s2);
    }
    free(q); free(sb); free(gs);
    return rc;
}
static int conv_upload(xconv *c, const float *wk /* [co][K] in GEMM order */, int co, const float *scale, const float *bias) {
    c->Kp = (c->K + KPAD - 1) / KPAD * KPAD; c->Cop = (co + 255) / 256 * 256;
    __half *hw = (__half *)calloc((size_t)c->Cop * c->Kp, sizeof(__half)); float *sc = (float *)malloc(co * sizeof(float)), *bi = (float *)malloc(co * sizeof(float));
    if (!hw || !sc || !bi) { free(hw); free(sc); free(bi); return -1; }
    float wmax = 0; fq_env();
    const int wseq = g_fq_wseq++;
    if (g_fq_w && c->ci % 32 == 0 && !c->unfold && (g_fq_mask >> (wseq & 63) & 1)) {
        float *wq = (float *)wk;
        for (int o = 0; o < co; o++) {
            float *r = wq + (size_t)o * c->K;
            if (g_fq_w == 9) {   /* probe: H16-rotated 16-K blocks (as k_fq 9) */
                auto wht = [](float *u) { for (int len = 1; len < 16; len <<= 1) for (int i = 0; i < 16; i += 2 * len) for (int j = i; j < i + len; j++) { const float a = u[j], c = u[j + len]; u[j] = a + c; u[j + len] = a - c; } for (int i = 0; i < 16; i++) u[i] *= .25f; };
                for (int k = 0; k + 16 <= c->K; k += 16) wht(r + k);
                float am = 0.f; for (int k = 0; k < c->K; k++) am = fmaxf(am, fabsf(r[k])); if (am > 0.f) fq_nv(r, c->K, am / (6.f * 448.f));
                for (int k = 0; k + 16 <= c->K; k += 16) wht(r + k);
            } else if (g_fq_w == 5) { float am = 0.f; for (int k = 0; k < c->K; k++) am = fmaxf(am, fabsf(r[k])); if (am > 0.f) fq_nv(r, c->K, am / (6.f * 448.f)); }
            else if (g_fq_w == 6) for (int k = 0; k < c->K; k += 16) fq_host(r + k, 16, 4);
            else for (int k = 0; k < c->K; k += 32) fq_host(r + k, 32, g_fq_w);
        }
    }
    for (int o = 0; o < co; o++) for (int k = 0; k < c->K; k++) { const float v = wk[(size_t)o * c->K + k]; hw[(size_t)o * c->Kp + k] = __float2half_rn(v); if (fabsf(v) > wmax) wmax = fabsf(v); }
    if (wmax > 60000.f) fprintf(stderr, "xnet: conv weight %g overflows fp16\n", wmax);
    for (int o = 0; o < co; o++) { sc[o] = scale ? scale[o] : 1.f; bi[o] = bias ? bias[o] : 0.f; }
    if (g_mx_env < 0) { const char *e = getenv("UFSM_XNET_MX"); g_mx_env = e && *e ? atoi(e) : 0; if (g_mx_env != 0 && g_mx_env != 5 && g_mx_env != 6) { fprintf(stderr, "xnet: UFSM_XNET_MX %d (5 or 6)\n", g_mx_env); g_mx_env = 0; } }
    mxq_env();
    if (g_mx_env && c->ci % 32 == 0 && !c->unfold && co % 8 == 0 && !mx_skipped(wseq) && mx_pack(c, wk, co, g_mx_env, wseq)) { free(hw); free(sc); free(bi); return -1; }
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
extern "C" void xconv_free(xconv *c) { xfree_dev(c->mxq); xfree_dev(c->mxs); xfree_dev(c->mxs2); xfree_dev(c->mxgs); xfree_dev(c->w); xfree_dev(c->scale); xfree_dev(c->bias); xfree_dev(c->tbias); memset(c, 0, sizeof *c); }

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
template <int BM, int BN, int WMn, int WNn, int BK, int NST, bool GNA = false> static void launch1(const ConvP &p, int co) {
    const size_t sm = (size_t)NST * (BM + BN) * BK * 2;
    static unsigned attr = 0u;   /* mgpu: per-device (a bit per device: the attribute is per context; set first, then the bit, atomically) */
    int dv = 0; cudaGetDevice(&dv);
    if (!(__atomic_load_n(&attr, __ATOMIC_ACQUIRE) >> (dv & 31) & 1u)) { cudaFuncSetAttribute(conv_igemm<BM, BN, WMn, WNn, BK, NST, GNA>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); __atomic_fetch_or(&attr, 1u << (dv & 31), __ATOMIC_RELEASE); }
    dim3 g((unsigned)((p.M + BM - 1) / BM), (unsigned)((co + BN - 1) / BN));
    conv_igemm<BM, BN, WMn, WNn, BK, NST, GNA><<<g, WMn * WNn * 32, sm, g_st>>>(p);
}
/* ---- MX-fp4 implicit-GEMM convolution (UFSM_XNET_MX = 6: blocks of 16 K with a power-of-two ue8m0 scale, 5: NVFP4-style ue4m3 block
   scales and a per-output-channel fp32 weight scale): m16n8k64 kind::mxf4nvf4 scale_vec::4X, fp32 accumulation, fp16 output.
   Activations: q [M][C / 2] (channel 2i in the low nibble of byte i), s [M][C / 32][4] (bytes 0, 1: the scales of channels 0..15, 16..31
   of the 32-channel chunk; 4-byte aligned so cp.async moves them). K order as the fp16 GEMM (tap-major, channels inner): 32-K chunk j =
   (tap j / (C / 32), channel block j % (C / 32)); a K step of 64 = chunks 2s, 2s + 1. Tiles 128 voxels x BN outputs, 4 warps of 32 rows,
   3-stage cp.async ring. Equal to the UFSM_XNET_FQ fake quantisation of the same format up to the fp32 summation order. ---- */
struct MxP {
    const uint8_t *xq, *xs, *wq, *ws; const float *gs, *scale, *bias; __half *y;
    int Di, Hi, Wi, Ci, Do, Ho, Wo, Co, kd, kh, kw, sd, sh, sw, pd, ph, pw, nch, nsteps; long long M;
    double *st;   /* IN sums of the fp16 output, [n][Co][2] (nullptr: none) */
    int tco; const float *tb; uint8_t *oq, *os; int oC, oc0;   /* tco > 0: a ConvTranspose3d(k 2, s 2) GEMM (column abc tco + o): output voxel
                                                                  (n, 2z + a, 2y + b, 2x + e), + tb[o]; oq / os: written as MX-fp4 channels
                                                                  [oc0, oc0 + tco) of an oC-channel tensor (y may then be nullptr) */
};
template <int SF> __device__ __forceinline__ void mma_x4(float *c, const unsigned *a, const unsigned *b, unsigned sa, unsigned sb) {
    if constexpr (SF == 6)
        asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue8m0 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb));
    else
        asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb));
}
template <int NB> __device__ __forceinline__ void cpa(void *dst, const void *src, bool ok) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" :: "r"(smem_u32(dst)), "l"(src), "n"(NB), "r"(ok ? NB : 0) : "memory");
}
/* tiles BM voxels x BN outputs, warps of 64 x 32 (BN 32: 4 x 1 warps, BM 256; BN 64: 2 x 2, BM 128; BN 128: 2 x 4, BM 128); a pipeline stage
   holds K 128 = 4 chunks (rows of 64 B + 16 B pad: conflict-free ldmatrix), 3 stages; the chunk -> (tap, channel block) table in shared */
enum { MX_KC = 4, MX_RS = 80, MX_NST = 3, MX_MAXCH = 1024 };
template <int BM, int BN> __host__ __device__ constexpr int mx_smem() { return MX_NST * ((BM + BN) * MX_RS + (BM + BN) * MX_KC * 4) + MX_MAXCH * 4; }
template <int BM, int BN, int SF>
__global__ void __launch_bounds__((BM / 64) * (BN / 32) * 32) conv_mx(MxP p) {
    constexpr int WMn = BM / 64, WNn = BN / 32, NTH = WMn * WNn * 32;
    extern __shared__ __align__(128) uint8_t smx[];
    uint8_t *sA = smx, *sB = sA + MX_NST * BM * MX_RS;
    unsigned *sAs = (unsigned *)(sB + MX_NST * BN * MX_RS), *sBs = sAs + MX_NST * MX_KC * BM; int *tab = (int *)(sBs + MX_NST * MX_KC * BN);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const long long m_base = (long long)blockIdx.x * BM; const int n_base = blockIdx.y * BN;
    const int CB = p.Ci / 32, khw = p.kh * p.kw, nchp = p.nsteps * 2;   /* chunks padded to even (the weight rows) */
    for (int j = tid; j < p.nch && j < MX_MAXCH; j += NTH) { const int tap = j / CB, cb = j - tap * CB, tz = tap / khw, r = tap - tz * khw, ty = r / p.kw, tx = r - ty * p.kw; tab[j] = tz << 24 | ty << 16 | tx << 8 | cb; }
    /* A rows of this thread: BM / NTH (2 for BN 32 / 64, 1 for BN 128) at row tid + i NTH */
    constexpr int AR = BM >= NTH ? BM / NTH : 1; int zb[AR], yb[AR], xb[AR]; long long nb[AR];
#pragma unroll
    for (int i = 0; i < AR; i++) {
        const long long m = m_base + tid + i * NTH;
        if (m < p.M) { long long q = m; const int xo = (int)(q % p.Wo); q /= p.Wo; const int yo = (int)(q % p.Ho); q /= p.Ho; const int zo = (int)(q % p.Do); q /= p.Do;
                       zb[i] = zo * p.sd - p.pd; yb[i] = yo * p.sh - p.ph; xb[i] = xo * p.sw - p.pw; nb[i] = q * p.Di; }
        else { zb[i] = -(1 << 28); yb[i] = xb[i] = 0; nb[i] = 0; }
    }
    __syncthreads();
    const int nks = (nchp + MX_KC - 1) / MX_KC;   /* pipeline stages of 4 chunks */
    auto load = [&](int slot, int s) {
        uint8_t *A = sA + slot * BM * MX_RS, *B = sB + slot * BN * MX_RS; unsigned *As = sAs + slot * MX_KC * BM, *Bs = sBs + slot * MX_KC * BN;
#pragma unroll
        for (int h = 0; h < MX_KC; h++) {
            const int j = s * MX_KC + h; int e = 0; const bool jin = j < p.nch; if (jin) e = tab[j];
            const int tz = e >> 24, ty = (e >> 16) & 255, tx = (e >> 8) & 255, cb = e & 255;
#pragma unroll
            for (int i = 0; i < AR; i++) {
                const int row = tid + i * NTH; if (BM < NTH && row >= BM) break;
                const int zi = zb[i] + tz, yi = yb[i] + ty, xi = xb[i] + tx;
                const bool ok = jin && zi >= 0 && zi < p.Di && yi >= 0 && yi < p.Hi && xi >= 0 && xi < p.Wi;
                const long long v = ok ? ((nb[i] + zi) * p.Hi + yi) * (long long)p.Wi + xi : 0;
                cpa<16>(A + row * MX_RS + h * 16, p.xq + (size_t)v * (p.Ci / 2) + cb * 16, ok); cpa<4>(As + h * BM + row, p.xs + ((size_t)v * CB + cb) * 4, ok);
            }
        }
        for (int c = tid; c < BN * MX_KC; c += NTH) {
            const int row = c / MX_KC, h = c % MX_KC, j = s * MX_KC + h; const bool ok = j < nchp;
            const size_t w = (size_t)(n_base + row) * nchp + (ok ? j : 0);
            cpa<16>(B + row * MX_RS + h * 16, p.wq + w * 16, ok); cpa<4>(Bs + h * BN + row, p.ws + w * 4, ok);
        }
    };
    float acc[4][4][4];
#pragma unroll
    for (int a = 0; a < 4; a++)
#pragma unroll
        for (int b = 0; b < 4; b++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[a][b][c] = 0.f;
    for (int s = 0; s < MX_NST - 1; s++) { if (s < nks) load(s, s); cp_commit(); }
    const int wm0 = (warp / WNn) * 64, wn0 = (warp % WNn) * 32;
    for (int ks = 0; ks < nks; ks++) {
        cp_wait<MX_NST - 2>(); __syncthreads();
        const int slot = ks % MX_NST;
        if (ks + MX_NST - 1 < nks) load((ks + MX_NST - 1) % MX_NST, ks + MX_NST - 1);
        cp_commit();
        const uint8_t *A = sA + slot * BM * MX_RS, *B = sB + slot * BN * MX_RS; const unsigned *As = sAs + slot * MX_KC * BM, *Bs = sBs + slot * MX_KC * BN;
#pragma unroll
        for (int kp = 0; kp < MX_KC / 2; kp++) {   /* k64 = chunks 2kp, 2kp + 1 */
            unsigned af[4][4], sa[4], bf[4][2], sb[4];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int r = wm0 + i * 16 + (lane & 15);
                xldsm_x4(af[i], A + r * MX_RS + kp * 32 + (lane >> 4) * 16);   /* matrices (rows 0-7, 8-15) x (K 0-31, 32-63) = a0 a1 a2 a3 */
                const int rs = wm0 + i * 16 + g + 8 * (t & 1);
                sa[i] = (As[(2 * kp) * BM + rs] & 0xffffu) | (As[(2 * kp + 1) * BM + rs] << 16);
            }
#pragma unroll
            for (int jj = 0; jj < 2; jj++) {   /* two n8 blocks per ldmatrix.x4: (cols 0-7 K lo, cols 0-7 K hi, cols 8-15 K lo, cols 8-15 K hi) */
                const int c = wn0 + jj * 16 + (lane & 7) + ((lane >> 4) << 3); unsigned r4[4];
                xldsm_x4(r4, B + c * MX_RS + kp * 32 + ((lane >> 3) & 1) * 16);
                bf[2 * jj][0] = r4[0]; bf[2 * jj][1] = r4[1]; bf[2 * jj + 1][0] = r4[2]; bf[2 * jj + 1][1] = r4[3];
            }
#pragma unroll
            for (int j = 0; j < 4; j++) { const int c = wn0 + j * 8 + g; sb[j] = (Bs[(2 * kp) * BN + c] & 0xffffu) | (Bs[(2 * kp + 1) * BN + c] << 16); }
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) mma_x4<SF>(acc[i][j], af[i], bf[j], sa[i], sb[j]);
        }
    }
    cp_wait<0>();
    if (p.tco) {   /* transposed conv: shuffle + bias (+ MX quantisation over 16 output channels = two n8 blocks across the quad) */
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const long long m = m_base + wm0 + i * 16 + g + h * 8; const bool rok = m < p.M;
            long long q = rok ? m : 0; const int x = (int)(q % p.Wo); q /= p.Wo; const int y = (int)(q % p.Ho); q /= p.Ho; const int z = (int)(q % p.Do); const long long n = q / p.Do;
#pragma unroll
            for (int jp = 0; jp < 2; jp++) {
                const int cb = n_base + wn0 + jp * 16, abc = cb / p.tco, o0 = cb - abc * p.tco; const bool cok = rok && cb < p.Co;
                const int a = abc >> 2, b = (abc >> 1) & 1, e = abc & 1;
                const long long ov = ((n * 2 * p.Do + 2 * z + a) * (2LL * p.Ho) + 2 * y + b) * (2LL * p.Wo) + 2 * x + e;
                float v[4];
#pragma unroll
                for (int u = 0; u < 2; u++) {
                    const int col = cb + u * 8 + 2 * t, o = o0 + u * 8 + 2 * t;
                    const float b0 = cok ? p.tb[o] : 0.f, b1 = cok ? p.tb[o + 1] : 0.f, s0 = cok ? p.scale[col] * p.gs[col] : 0.f, s1 = cok ? p.scale[col + 1] * p.gs[col + 1] : 0.f;
                    const __half2 hv = __floats2half2_rn(acc[i][2 * jp + u][2 * h] * s0 + b0, acc[i][2 * jp + u][2 * h + 1] * s1 + b1);
                    v[2 * u] = cok ? __low2float(hv) : 0.f; v[2 * u + 1] = cok ? __high2float(hv) : 0.f;
                    if (cok && p.y) *(__half2 *)(p.y + (size_t)ov * p.tco + o) = hv;
                }
                if (p.oq) {
                    float am = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
                    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1)); am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
                    uint8_t sc, lo, hi;
                    if constexpr (SF == 6) { const int ee = mx_exp(am, 1.f / 6.f); const float mm = exp2i(-ee); lo = cvt_e2m1x2(v[0] * mm, v[1] * mm); hi = cvt_e2m1x2(v[2] * mm, v[3] * mm); sc = (uint8_t)(ee + 127); }
                    else { sc = cvt_e4m3(__fdiv_rn(am, 6.f)); const int ex = (sc >> 3) & 15, mt = sc & 7; const float d = ex ? ldexpf(1.f + mt / 8.f, ex - 7) : ldexpf(mt / 8.f, -6);
                           const float r = d > 0.f ? 1.f / d : 0.f; (void)r;
                           lo = cvt_e2m1x2(d > 0.f ? __fdiv_rn(v[0], d) : 0.f, d > 0.f ? __fdiv_rn(v[1], d) : 0.f); hi = cvt_e2m1x2(d > 0.f ? __fdiv_rn(v[2], d) : 0.f, d > 0.f ? __fdiv_rn(v[3], d) : 0.f); }
                    if (cok) {
                        const int ch = p.oc0 + o0; uint8_t *rec = p.oq + (size_t)ov * (p.oC / 2) + ch / 2;
                        rec[t] = lo; rec[4 + t] = hi;
                        if (t == 0) p.os[((size_t)ov * (p.oC / 32) + ch / 32) * 4 + ((ch / 16) & 1)] = sc;
                    }
                }
            }
        }
        return;
    }
    cp_wait<0>();
    const long long S = (long long)p.Do * p.Ho * p.Wo, nw = (m_base + wm0) / S;   /* the warp's first row's sample */
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int co = n_base + wn0 + j * 8 + t * 2; const bool cok = co < p.Co;
        const float s0 = cok ? p.scale[co] * p.gs[co] : 0.f, s1 = cok ? p.scale[co + 1] * p.gs[co + 1] : 0.f, b0 = cok ? p.bias[co] : 0.f, b1 = cok ? p.bias[co + 1] : 0.f;
        float a0 = 0.f, a1 = 0.f, c0 = 0.f, c1 = 0.f;
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const long long m = m_base + wm0 + i * 16 + g + h * 8; if (m >= p.M || !cok) continue;
            const __half2 o = __floats2half2_rn(acc[i][j][2 * h] * s0 + b0, acc[i][j][2 * h + 1] * s1 + b1);
            *(__half2 *)(p.y + (size_t)m * p.Co + co) = o;
            if (p.st) {
                const float v0 = __low2float(o), v1 = __high2float(o); const long long nn = m / S;
                if (nn == nw) { a0 += v0; c0 += v0 * v0; a1 += v1; c1 += v1 * v1; }
                else { double *d = p.st + ((size_t)nn * p.Co + co) * 2; atomicAdd(d, (double)v0); atomicAdd(d + 1, (double)v0 * v0); atomicAdd(d + 2, (double)v1); atomicAdd(d + 3, (double)v1 * v1); }
            }
        }
        if (p.st && nw * S < p.M) st_add(p.st, nw, p.Co, co, a0, c0, a1, c1, lane);
    }
}
template <int BM, int BN, int SF> static void mx_launch(const MxP &m) {
    static unsigned attr = 0; constexpr int sm = mx_smem<BM, BN>(); const int dv = g_dev & 31;
    if (!(__atomic_load_n(&attr, __ATOMIC_ACQUIRE) >> dv & 1u)) { cudaFuncSetAttribute(conv_mx<BM, BN, SF>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); __atomic_fetch_or(&attr, 1u << dv, __ATOMIC_RELEASE); }
    dim3 gr((unsigned)((m.M + BM - 1) / BM), (unsigned)((m.Co + BN - 1) / BN));
    conv_mx<BM, BN, SF><<<gr, (BM / 64) * (BN / 32) * 32, sm, g_st>>>(m);
}
/* ---- the halo-tiled MX-fp4 3x3x3 stride-1 convolution: a block computes an output tile TZ x TY x TX (256 voxels) x 32 outputs; the input
   halo ((TZ + 2) x (TY + 2) x (TX + 2) positions) of one or two 32-channel chunks is staged in shared once and read by every tap through
   ldmatrix row addresses (the tap shift is a per-lane address offset), so the input is read ~2.5x per tile instead of 27x. K steps of 64:
   Ci 32 -> taps (2s, 2s + 1) of the one chunk (14 steps, the 28th tap has zero weights); Ci % 64 == 0 -> per chunk pair, per kz plane,
   9 steps (tap, chunks 2c | 2c + 1). Weights of a pass staged per step as two 16-byte planes [step][half][32 co] + scales. 4 warps of 64
   voxels x 32 outputs. ---- */
enum { MH_NPMAX = 816, MH_SMAX = 14 };
struct MhP { MxP m; const unsigned *ws2; int tz_n, ty_n, tx_n; };
template <int CFG> struct mh_cfg;   /* output tile (z, y, x) */
template <> struct mh_cfg<0> { static constexpr int TZ = 2, TY = 4, TX = 32; };
template <> struct mh_cfg<1> { static constexpr int TZ = 4, TY = 4, TX = 16; };
template <> struct mh_cfg<2> { static constexpr int TZ = 4, TY = 8, TX = 8; };
template <int SF, int CFG, bool C32>
__global__ void __launch_bounds__(128) conv_mxh(MhP P) {
    constexpr int TZ = mh_cfg<CFG>::TZ, TY = mh_cfg<CFG>::TY, TX = mh_cfg<CFG>::TX, HZ = TZ + 2, HY = TY + 2, HX = TX + 2, NP = HZ * HY * HX, HC = C32 ? 1 : 2;
    constexpr int NST = C32 ? 14 : 9;
    static_assert(NP <= MH_NPMAX, "halo");
    const MxP &p = P.m;
    __shared__ __align__(16) uint8_t hq[HC][NP * 16];
    __shared__ __align__(16) unsigned hs[HC][NP];
    __shared__ __align__(16) uint8_t wq[NST][2][32 * 16];
    __shared__ __align__(16) unsigned ws[NST][32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    long long tile = blockIdx.x; const int txi = (int)(tile % P.tx_n); tile /= P.tx_n; const int tyi = (int)(tile % P.ty_n); tile /= P.ty_n;
    const int tzi = (int)(tile % P.tz_n); const long long n = tile / P.tz_n;
    const int oz0 = tzi * TZ, oy0 = tyi * TY, ox0 = txi * TX, n_base = blockIdx.y * 32;
    const int CB = p.Ci / 32, nchp = p.nsteps * 2;
    const int wm0 = warp * 64; unsigned ab[4], sab[4];   /* shared addresses: this lane's ldmatrix row at tap 0 (plane by lane half), its scale row */
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int r = wm0 + i * 16 + (lane & 15); const int pl = C32 ? 0 : lane >> 4;
        ab[i] = smem_u32(hq[pl]) + (((r / (TY * TX)) * HY + (r / TX) % TY) * HX + r % TX) * 16;
        r = wm0 + i * 16 + g + 8 * (t & 1); sab[i] = smem_u32(hs[0]) + (((r / (TY * TX)) * HY + (r / TX) % TY) * HX + r % TX) * 4;
    }
    float acc[4][4][4];
#pragma unroll
    for (int a = 0; a < 4; a++)
#pragma unroll
        for (int b = 0; b < 4; b++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[a][b][c] = 0.f;
    const int npass = C32 ? 1 : CB / 2;
    /* halo staging: thread x along a row of HX positions (planes side by side: HC HX <= 128 threads) */
    const int spl = tid >= HX ? 1 : 0, shx = tid - spl * HX; const bool sact = tid < HX * HC;
    const int sxi = ox0 - 1 + shx; const bool sxok = sxi >= 0 && sxi < p.Wi;
    for (int cp = 0; cp < npass; cp++) {
        __syncthreads();
        if (sact) {
            const int cb = 2 * cp + spl;
#pragma unroll 2
            for (int rr = 0; rr < HZ * HY; rr++) {
                const int hz = rr / HY, hy = rr - hz * HY, zi = oz0 - 1 + hz, yi = oy0 - 1 + hy, pos = rr * HX + shx;
                const bool ok = sxok && zi >= 0 && zi < p.Di && yi >= 0 && yi < p.Hi;
                const long long v = ok ? ((n * p.Di + zi) * p.Hi + yi) * (long long)p.Wi + sxi : 0;
                cpa<16>(hq[spl] + pos * 16, p.xq + (size_t)v * (p.Ci / 2) + cb * 16, ok); cpa<4>(&hs[spl][pos], p.xs + ((size_t)v * CB + cb) * 4, ok);
            }
        }
        constexpr int NG = C32 ? 1 : 3;
#pragma unroll 1
        for (int kz = 0; kz < NG; kz++) {
            if (kz) __syncthreads();   /* the previous group's weights are read */
            for (int e = tid; e < NST * 64; e += 128) {   /* weights: (step, co row, half) */
                const int hf = e & 1, row = (e >> 1) & 31, st = e >> 6;
                const int j = C32 ? 2 * st : (kz * 9 + st) * CB + 2 * cp;   /* the step's first chunk (even) */
                const size_t w = (size_t)(n_base + row) * nchp + j;
                cpa<16>(wq[st][hf] + row * 16, p.wq + (w + hf) * 16, true);
                if (!hf) cpa<4>(&ws[st][row], P.ws2 + w / 2, true);
            }
            cp_commit(); cp_wait<0>(); __syncthreads();
#pragma unroll
            for (int st = 0; st < NST; st++) {
                int offA, offB;   /* halo position offsets of the two K halves (C32: taps 2 st, 2 st + 1 of plane 0; else tap kz 9 + st of planes 0 | 1) */
                if constexpr (C32) { constexpr int dummy = 0; (void)dummy;
                    const int ta = 2 * st, tb = ta + 1 < 27 ? ta + 1 : 26;
                    offA = ((ta / 9) * HY + (ta / 3) % 3) * HX + ta % 3; offB = ((tb / 9) * HY + (tb / 3) % 3) * HX + tb % 3; }
                else { const int ta = st; offA = offB = (kz * HY + (ta / 3) % 3) * HX + ta % 3; }
                const int mo = (lane < 16 ? offA : offB) * 16;
                unsigned af[4][4], sa[4], bf[4][2], sb[4];
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(af[i][0]), "=r"(af[i][1]), "=r"(af[i][2]), "=r"(af[i][3]) : "r"(ab[i] + mo));
                    unsigned s0, s1;
                    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(s0) : "r"(sab[i] + offA * 4));
                    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(s1) : "r"(sab[i] + offB * 4 + (C32 ? 0 : NP * 4)));
                    sa[i] = (s0 & 0xffffu) | (s1 << 16);
                }
#pragma unroll
                for (int jj = 0; jj < 2; jj++) {
                    unsigned r4[4]; const int c = jj * 16 + (lane & 7) + ((lane >> 4) << 3);
                    xldsm_x4(r4, wq[st][(lane >> 3) & 1] + c * 16);
                    bf[2 * jj][0] = r4[0]; bf[2 * jj][1] = r4[1]; bf[2 * jj + 1][0] = r4[2]; bf[2 * jj + 1][1] = r4[3];
                }
#pragma unroll
                for (int j = 0; j < 4; j++) sb[j] = ws[st][j * 8 + g];
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int j = 0; j < 4; j++) mma_x4<SF>(acc[i][j], af[i], bf[j], sa[i], sb[j]);
            }
        }
    }
    long long mo[4][2];   /* output voxel of rows wm0 + i 16 + g + 8 h (-1: outside) */
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int r = wm0 + i * 16 + g + h * 8, z = oz0 + r / (TY * TX), y = oy0 + (r / TX) % TY, x = ox0 + r % TX;
            mo[i][h] = z < p.Do && y < p.Ho && x < p.Wo ? ((n * p.Do + z) * p.Ho + y) * (long long)p.Wo + x : -1;
        }
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int co = n_base + j * 8 + t * 2; const bool cok = co < p.Co;
        const float s0 = cok ? p.scale[co] * p.gs[co] : 0.f, s1 = cok ? p.scale[co + 1] * p.gs[co + 1] : 0.f, b0 = cok ? p.bias[co] : 0.f, b1 = cok ? p.bias[co + 1] : 0.f;
        float a0 = 0.f, a1 = 0.f, c0 = 0.f, c1 = 0.f;
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int h = 0; h < 2; h++) if (mo[i][h] >= 0 && cok) {
                const __half2 o = __floats2half2_rn(acc[i][j][2 * h] * s0 + b0, acc[i][j][2 * h + 1] * s1 + b1);
                *(__half2 *)(p.y + (size_t)mo[i][h] * p.Co + co) = o;
                const float v0 = __low2float(o), v1 = __high2float(o); a0 += v0; c0 += v0 * v0; a1 += v1; c1 += v1 * v1;
            }
        if (p.st) st_add(p.st, n, p.Co, co, a0, c0, a1, c1, lane);
    }
}
template <int SF, int CFG> static void mxh_go(const MhP &P, dim3 gr, bool c32) {
    if (c32) conv_mxh<SF, CFG, true><<<gr, 128, 0, g_st>>>(P); else conv_mxh<SF, CFG, false><<<gr, 128, 0, g_st>>>(P);
}
static int mxh_launch(const MxP &m, int sf, const unsigned *ws2) {
    if (m.kd != 3 || m.kh != 3 || m.kw != 3 || m.sd != 1 || m.sh != 1 || m.sw != 1 || m.pd != 1 || m.ph != 1 || m.pw != 1) return 0;
    if (!(m.Ci == 32 || m.Ci % 64 == 0) || m.Wo < 8 || !ws2) return 0;
    static int env = -2; if (env == -2) env = getenv("UFSM_XNET_MXH") ? atoi(getenv("UFSM_XNET_MXH")) : 1;   /* mgpu: env */
    if (!env) return 0;
    MhP P; P.m = m; P.ws2 = ws2; const int cfg = m.Wo >= 32 ? 0 : m.Wo >= 16 ? 1 : 2;
    const int TZ = cfg == 0 ? 2 : 4, TY = cfg == 2 ? 8 : 4, TX = cfg == 0 ? 32 : cfg == 1 ? 16 : 8;
    P.tz_n = (m.Do + TZ - 1) / TZ; P.ty_n = (m.Ho + TY - 1) / TY; P.tx_n = (m.Wo + TX - 1) / TX;
    const long long nt = (long long)(m.M / ((long long)m.Do * m.Ho * m.Wo)) * P.tz_n * P.ty_n * P.tx_n;
    dim3 gr((unsigned)nt, (unsigned)((m.Co + 31) / 32)); const bool c32 = m.Ci == 32;
    if (sf == 6) { if (cfg == 0) mxh_go<6, 0>(P, gr, c32); else if (cfg == 1) mxh_go<6, 1>(P, gr, c32); else mxh_go<6, 2>(P, gr, c32); }
    else { if (cfg == 0) mxh_go<5, 0>(P, gr, c32); else if (cfg == 1) mxh_go<5, 1>(P, gr, c32); else mxh_go<5, 2>(P, gr, c32); }
    return 1;
}
/* fp16 NDHWC (C % 32 == 0) -> the MX-fp4 activation layout above, one thread per (voxel, 16-channel block) */
__device__ __forceinline__ void wht16(float *u) {   /* u <- H16 u / 4 (orthonormal, its own inverse) */
#pragma unroll
    for (int len = 1; len < 16; len <<= 1)
#pragma unroll
        for (int i = 0; i < 16; i += 2 * len)
#pragma unroll
            for (int j = i; j < i + len; j++) { const float a = u[j], c = u[j + len]; u[j] = a + c; u[j + len] = a - c; }
#pragma unroll
    for (int i = 0; i < 16; i++) u[i] *= .25f;
}
__device__ __forceinline__ float e4dec(int c) { const int ex = (c >> 3) & 15, mt = c & 7; return ex ? ldexpf(1.f + mt / 8.f, ex - 7) : ldexpf(mt / 8.f, -6); }
__device__ __forceinline__ float sse16(const float *v, float d, bool pow2) {   /* squared error of one 16-block on the e2m1 grid with scale d */
    constexpr float G[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f}; float e = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i++) { const unsigned c = cvt_e2m1x2(pow2 ? v[i] * (1.f / d) : __fdiv_rn(v[i], d), 0.f) & 15u; const float r = (c & 8 ? -G[c & 7] : G[c & 7]) * d - v[i]; e += r * r; }
    return e;
}
/* MODE bit 0 (UFSM_XNET_MXROT 1): each 16-block H16-rotated first; bit 1 (UFSM_XNET_MXA 1): the block scale of least squared error, sf 5
   among the e4m3 codes -4 .. +2 around amax / 6, sf 6 among 2^e, 2^(e - 1) (an earlier candidate is kept unless 1e-5 worse) */
template <int SF, int MODE = 0> __global__ void k_xq(const __half *x, uint8_t *q, uint8_t *s, long long nb16) {
    const long long b = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (b >= nb16) return;
    __align__(16) __half hv[16]; *(uint4 *)hv = *(const uint4 *)(x + b * 16); *(uint4 *)(hv + 8) = *(const uint4 *)(x + b * 16 + 8);
    float v[16], am = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i++) v[i] = __half2float(hv[i]);
    if constexpr (MODE & 1) wht16(v);
#pragma unroll
    for (int i = 0; i < 16; i++) am = fmaxf(am, fabsf(v[i]));
    uint2 o; uint8_t sc;
    if constexpr (SF == 6) {
        int e = mx_exp(am, 1.f / 6.f);
        if constexpr ((MODE & 2) != 0) if (am > 0.f && e > -127 && sse16(v, exp2i(e - 1), true) < sse16(v, exp2i(e), true) * (1.f - 1e-5f)) e -= 1;
        const float m = exp2i(-e); o.x = cvt_e2m1x8(v, m); o.y = cvt_e2m1x8(v + 8, m); sc = (uint8_t)(e + 127);
    } else {
        sc = cvt_e4m3(__fdiv_rn(am, 6.f));
        if constexpr ((MODE & 2) != 0) if (am > 0.f) {
            float be = sse16(v, e4dec(sc), false); int bc = sc;
            for (int cc = (int)sc - 4; cc <= (int)sc + 2; cc++) { if (cc < 1 || cc > 126 || cc == (int)sc) continue; const float t = sse16(v, e4dec(cc), false); if (t < be * (1.f - 1e-5f)) { be = t; bc = cc; } }
            sc = (uint8_t)bc;
        }
        const float d = e4dec(sc);
        float w[16];
#pragma unroll
        for (int i = 0; i < 16; i++) w[i] = d > 0.f ? __fdiv_rn(v[i], d) : 0.f;
        o.x = cvt_e2m1x8(w, 1.f); o.y = cvt_e2m1x8(w + 8, 1.f);
    }
    *(uint2 *)(q + b * 8) = o;
    s[(b >> 1) * 4 + (b & 1)] = sc;
}
template <int SF> static void xq_launch(int mode, const __half *x, uint8_t *q, uint8_t *s, long long nb16) {
    switch (mode & 3) {
    case 1: k_xq<SF, 1><<<nblk(nb16, 256), 256, 0, g_st>>>(x, q, s, nb16); break;
    case 2: k_xq<SF, 2><<<nblk(nb16, 256), 256, 0, g_st>>>(x, q, s, nb16); break;
    case 3: k_xq<SF, 3><<<nblk(nb16, 256), 256, 0, g_st>>>(x, q, s, nb16); break;
    default: k_xq<SF, 0><<<nblk(nb16, 256), 256, 0, g_st>>>(x, q, s, nb16); break;
    }
}
/* ---- the Reader depth fusion's 16 -> 16 channel 3x3x3 conv (stride 1, pad 1), fp16: halo-tiled direct conv. The implicit GEMM gathered
   27 x 2 16-byte rows per output voxel from global (A-load bound: 9.8 ms per 4 reader3 tiles); here a block stages its (TZ + 2) (TY + 2)
   (TX + 2) input halo and the 27 x 16 x 16 weights once in shared memory and runs 27 taps of m16n8k16 per 16 output voxels. Rows of 32 B
   with the two 16-byte halves swapped on bit 2 of the row: 8 consecutive rows hit 8 distinct bank groups (conflict-free ldmatrix). ---- */
template <int TZ, int TY, int TX, int NW> __global__ void __launch_bounds__(NW * 32) k_dfc16(const __half *x, const __half *w, int Kp, const float *scale, const float *bias,
                                                                               __half *y, int D, int H, int W, const float *gm, const float *rs, const float *gg, const float *gb, int act) {
    /* gm != nullptr: the input is GroupNorm-applied on load as k_gn_apply (x - gm) rs gamma + beta, act, fp16-rounded (padding stays 0) */
    extern __shared__ uint4 sm4[];
    constexpr int HY = TY + 2, HX = TX + 2, HV = (TZ + 2) * HY * HX, MT = TZ * TY * TX / 16, MW = MT / NW, NT = NW * 32;
    uint4 *sx = sm4, *sw = sm4 + HV * 2;
    const int ntx = (W + TX - 1) / TX, nty = (H + TY - 1) / TY, ntz = (D + TZ - 1) / TZ;
    long long bi = blockIdx.x; const int bx = (int)(bi % ntx); bi /= ntx; const int by = (int)(bi % nty); bi /= nty; const int bz = (int)(bi % ntz); const long long n = bi / ntz;
    const int z0 = bz * TZ, y0 = by * TY, x0 = bx * TX, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    for (int i = tid; i < 27 * 16 * 2; i += NT) {
        const int t = i >> 5, co = (i >> 1) & 15, h = i & 1;
        sw[(t * 16 + co) * 2 + (h ^ ((co >> 2) & 1))] = *(const uint4 *)(w + (size_t)co * Kp + t * 16 + h * 8);
    }
    for (int i = tid; i < HV * 2; i += NT) {
        const int v = i >> 1, h = i & 1, hz = v / (HY * HX), r = v - hz * HY * HX, hy = r / HX, hx = r - hy * HX;
        const int gz = z0 - 1 + hz, gy = y0 - 1 + hy, gx = x0 - 1 + hx;
        uint4 val = make_uint4(0u, 0u, 0u, 0u);
        if (gz >= 0 && gz < D && gy >= 0 && gy < H && gx >= 0 && gx < W) {
            val = *(const uint4 *)(x + ((((size_t)n * D + gz) * H + gy) * (size_t)W + gx) * 16 + h * 8);
            if (gm) {
                __half *hv = (__half *)&val; const int q0 = (int)n * 16 + h * 8;
#pragma unroll
                for (int k = 0; k < 8; k++) hv[k] = __float2half_rn(act_f((__half2float(hv[k]) - gm[q0 + k]) * rs[q0 + k] * gg[h * 8 + k] + gb[h * 8 + k], act));
            }
        }
        sx[v * 2 + (h ^ ((v >> 2) & 1))] = val;
    }
    __syncthreads();
    float acc[MW][2][4];
#pragma unroll
    for (int i = 0; i < MW; i++)
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int k = 0; k < 4; k++) acc[i][j][k] = 0.f;
    int vb[MW];   /* halo row of each m-tile's first voxel (tap 0, 0, 0) */
#pragma unroll
    for (int i = 0; i < MW; i++) {
        const int mt = warp * MW + i, oz = mt / (TY * TX / 16), rm = mt - oz * (TY * TX / 16), oy = rm / (TX / 16), ox = (rm - oy * (TX / 16)) * 16;
        vb[i] = (oz * HY + oy) * HX + ox + (lane & 15);
    }
    const int brow = (lane & 7) + ((lane >> 4) << 3), bh = (lane >> 3) & 1, ah = lane >> 4;
#pragma unroll 1
    for (int t = 0; t < 27; t++) {
        const int a = t / 9, b = (t / 3) % 3, c = t % 3, off = (a * HY + b) * HX + c;
        unsigned r[4]; xldsm_x4(r, sw + (t * 16 + brow) * 2 + (bh ^ ((brow >> 2) & 1)));
        const unsigned b0[2] = {r[0], r[1]}, b1[2] = {r[2], r[3]};
#pragma unroll
        for (int i = 0; i < MW; i++) {
            const int v = vb[i] + off; unsigned af[4]; xldsm_x4(af, sx + v * 2 + (ah ^ ((v >> 2) & 1)));
            mma16816(acc[i][0], af, b0); mma16816(acc[i][1], af, b1);
        }
    }
    const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int i = 0; i < MW; i++) {
        const int mt = warp * MW + i, oz = mt / (TY * TX / 16), rm = mt - oz * (TY * TX / 16), oy = rm / (TX / 16), ox = (rm - oy * (TX / 16)) * 16;
        const int gz = z0 + oz, gy = y0 + oy; if (gz >= D || gy >= H) continue;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int gx = x0 + ox + g + 8 * h; if (gx >= W) continue;
            __half *o = y + ((((size_t)n * D + gz) * H + gy) * (size_t)W + gx) * 16;
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int co = j * 8 + 2 * t4;
                *(__half2 *)(o + co) = __floats2half2_rn(acc[i][j][2 * h] * scale[co] + bias[co], acc[i][j][2 * h + 1] * scale[co + 1] + bias[co + 1]);
            }
        }
    }
}
struct GnIn { const float *gm, *rs, *g, *b; int act; };
template <int TZ, int TY, int TX, int NW> static bool dfc16_go(const xconv *c, xt in, xt y, const GnIn &gi) {
    constexpr size_t sm = ((size_t)(TZ + 2) * (TY + 2) * (TX + 2) * 2 + 27 * 16 * 2) * 16;
    static unsigned attr = 0u;   /* mgpu: per-device (as launch1) */
    int dv = 0; cudaGetDevice(&dv);
    if (!(__atomic_load_n(&attr, __ATOMIC_ACQUIRE) >> (dv & 31) & 1u)) { cudaFuncSetAttribute(k_dfc16<TZ, TY, TX, NW>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); __atomic_fetch_or(&attr, 1u << (dv & 31), __ATOMIC_RELEASE); }
    const long long nb = (long long)in.n * ((in.d + TZ - 1) / TZ) * ((in.h + TY - 1) / TY) * ((in.w + TX - 1) / TX);
    k_dfc16<TZ, TY, TX, NW><<<(unsigned)nb, NW * 32, sm, g_st>>>((const __half *)in.p, (const __half *)c->w, c->Kp, c->scale, c->bias, (__half *)y.p, in.d, in.h, in.w,
                                                                  gi.gm, gi.rs, gi.g, gi.b, gi.act);
    return true;
}
/* k_dfc16's depth-streaming form: a block owns a TY x TX column of one sample over the whole depth and keeps a ring of three (TY + 2)
   (TX + 2) input planes; plane z + 2 is loaded into registers before output plane z's 27 taps run and stored (GroupNorm-applied) over the
   ring slot of plane z - 1 after them, so global latency overlaps the MMAs. Input reads and GN applies fall from (TZ+2)(TY+2)(TX+2) /
   TZ TY TX (2.66x at 2 x 8 x 32) to (TY+2)(TX+2) / TY TX (1.33x at 8 x 32), and the smem (46 KB at 8 x 32) fits 2 blocks an SM where the
   4 x 10 x 34 halo (57 KB) fit one. Each output's taps run in the same order through the same mma16816 on the same rows as k_dfc16, and
   the GN apply and epilogue are the same expressions: byte-identical (test_xmx dfs). */
template <int TY, int TX, int NW> __global__ void __launch_bounds__(NW * 32) k_dfc16s(const __half *x, const __half *w, int Kp, const float *scale, const float *bias,
                                                                         __half *y, int D, int H, int W, const float *gm, const float *rs, const float *gg, const float *gb, int act) {
    extern __shared__ uint4 sm4[];
    constexpr int HY = TY + 2, HX = TX + 2, PV = HY * HX, PE = PV * 2, MT = TY * TX / 16, MW = MT / NW, NT = NW * 32, LP = (PE + NT - 1) / NT;
    static_assert(MT % NW == 0, "m-tiles per warp");
    uint4 *sw = sm4, *sx = sm4 + 27 * 16 * 2;   /* sx: 3 planes of PE */
    const int ntx = (W + TX - 1) / TX, nty = (H + TY - 1) / TY;
    long long bi = blockIdx.x; const int bx = (int)(bi % ntx); bi /= ntx; const int by = (int)(bi % nty); const long long n = bi / nty;
    const int y0 = by * TY, x0 = bx * TX, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    for (int i = tid; i < 27 * 16 * 2; i += NT) {
        const int t = i >> 5, co = (i >> 1) & 15, h = i & 1;
        sw[(t * 16 + co) * 2 + (h ^ ((co >> 2) & 1))] = *(const uint4 *)(w + (size_t)co * Kp + t * 16 + h * 8);
    }
    uint4 rg[LP];
    auto fetch = [&](int gz) {   /* raw plane gz into registers (zeros outside the volume) */
#pragma unroll
        for (int j = 0; j < LP; j++) {
            const int i = tid + j * NT; rg[j] = make_uint4(0u, 0u, 0u, 0u); if (i >= PE) continue;
            const int v = i >> 1, h = i & 1, hy = v / HX, hx = v - hy * HX, gy = y0 - 1 + hy, gx = x0 - 1 + hx;
            if (gz >= 0 && gz < D && gy >= 0 && gy < H && gx >= 0 && gx < W) rg[j] = *(const uint4 *)(x + ((((size_t)n * D + gz) * H + gy) * (size_t)W + gx) * 16 + h * 8);
        }
    };
    auto put = [&](int gz) {     /* the registers (GroupNorm-applied inside the volume) into gz's ring slot */
        uint4 *pl = sx + (size_t)((gz + 3) % 3) * PE;
#pragma unroll
        for (int j = 0; j < LP; j++) {
            const int i = tid + j * NT; if (i >= PE) continue;
            const int v = i >> 1, h = i & 1, hy = v / HX, hx = v - hy * HX, gy = y0 - 1 + hy, gx = x0 - 1 + hx;
            uint4 val = rg[j];
            if (gm && gz >= 0 && gz < D && gy >= 0 && gy < H && gx >= 0 && gx < W) {
                __half *hv = (__half *)&val; const int q0 = (int)n * 16 + h * 8;
#pragma unroll
                for (int k = 0; k < 8; k++) hv[k] = __float2half_rn(act_f((__half2float(hv[k]) - gm[q0 + k]) * rs[q0 + k] * gg[h * 8 + k] + gb[h * 8 + k], act));
            }
            pl[v * 2 + (h ^ ((v >> 2) & 1))] = val;
        }
    };
    fetch(-1); put(-1); fetch(0); put(0); fetch(1); put(1);
    __syncthreads();
    int vb[MW];   /* plane row of each m-tile's first voxel (tap 0, 0) */
#pragma unroll
    for (int i = 0; i < MW; i++) { const int mt = warp * MW + i, oy = mt / (TX / 16), ox = (mt - oy * (TX / 16)) * 16; vb[i] = oy * HX + ox + (lane & 15); }
    const int brow = (lane & 7) + ((lane >> 4) << 3), bh = (lane >> 3) & 1, ah = lane >> 4, g = lane >> 2, t4 = lane & 3;
#pragma unroll 1
    for (int z = 0; z < D; z++) {
        if (z + 2 <= D) fetch(z + 2);
        float acc[MW][2][4];
#pragma unroll
        for (int i = 0; i < MW; i++)
#pragma unroll
            for (int j = 0; j < 2; j++)
#pragma unroll
                for (int k = 0; k < 4; k++) acc[i][j][k] = 0.f;
#pragma unroll 1
        for (int a = 0; a < 3; a++) {
            const uint4 *pl = sx + (size_t)((z + a + 2) % 3) * PE;   /* plane z - 1 + a */
#pragma unroll
            for (int bc = 0; bc < 9; bc++) {
                const int t = a * 9 + bc, off = (bc / 3) * HX + bc % 3;
                unsigned r[4]; xldsm_x4(r, sw + (t * 16 + brow) * 2 + (bh ^ ((brow >> 2) & 1)));
                const unsigned b0[2] = {r[0], r[1]}, b1[2] = {r[2], r[3]};
#pragma unroll
                for (int i = 0; i < MW; i++) {
                    const int v = vb[i] + off; unsigned af[4]; xldsm_x4(af, pl + v * 2 + (ah ^ ((v >> 2) & 1)));
                    mma16816(acc[i][0], af, b0); mma16816(acc[i][1], af, b1);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < MW; i++) {
            const int mt = warp * MW + i, oy = mt / (TX / 16), ox = (mt - oy * (TX / 16)) * 16, gy = y0 + oy; if (gy >= H) continue;
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int gx = x0 + ox + g + 8 * h; if (gx >= W) continue;
                __half *o = y + ((((size_t)n * D + z) * H + gy) * (size_t)W + gx) * 16;
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    const int co = j * 8 + 2 * t4;
                    *(__half2 *)(o + co) = __floats2half2_rn(acc[i][j][2 * h] * scale[co] + bias[co], acc[i][j][2 * h + 1] * scale[co + 1] + bias[co + 1]);
                }
            }
        }
        if (z + 2 <= D) { __syncthreads(); put(z + 2); __syncthreads(); }   /* slot of plane z - 1 is free once every warp ran plane z */
    }
}
template <int TY, int TX, int NW> static bool dfc16s_go(const xconv *c, xt in, xt y, const GnIn &gi) {
    constexpr size_t sm = ((size_t)3 * (TY + 2) * (TX + 2) * 2 + 27 * 16 * 2) * 16;
    static unsigned attr = 0u;   /* mgpu: per-device (as launch1) */
    int dv = 0; cudaGetDevice(&dv);
    if (!(__atomic_load_n(&attr, __ATOMIC_ACQUIRE) >> (dv & 31) & 1u)) { cudaFuncSetAttribute(k_dfc16s<TY, TX, NW>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); __atomic_fetch_or(&attr, 1u << (dv & 31), __ATOMIC_RELEASE); }
    const long long nb = (long long)in.n * ((in.h + TY - 1) / TY) * ((in.w + TX - 1) / TX);
    k_dfc16s<TY, TX, NW><<<(unsigned)nb, NW * 32, sm, g_st>>>((const __half *)in.p, (const __half *)c->w, c->Kp, c->scale, c->bias, (__half *)y.p, in.d, in.h, in.w,
                                                              gi.gm, gi.rs, gi.g, gi.b, gi.act);
    return true;
}
static int g_df16 = -1;   /* UFSM_XNET_DF16=0: the depth-fusion conv on the implicit GEMM (A/B); mgpu: env */
extern "C" void xnet_set_df16(int v) { g_df16 = v; }
static bool dfc16_try(const xconv *c, xt in, xt y, const xt *res, int act, double *st, const GnIn &gi = GnIn{}) {
    if (g_df16 < 0) g_df16 = getenv("UFSM_XNET_DF16") ? atoi(getenv("UFSM_XNET_DF16")) : 10;
    if (!g_df16 || c->mxq || c->unfold || c->tconv || c->ci != 16 || c->co != 16 || in.c != 16 || c->kd != 3 || c->kh != 3 || c->kw != 3) return false;
    if (c->sd != 1 || c->sh != 1 || c->sw != 1 || c->pd != 1 || c->ph != 1 || c->pw != 1 || res || act != XA_NONE || st) return false;
    switch (g_df16) {
    case 2: return dfc16_go<2, 4, 32, 4>(c, in, y, gi);
    case 4: return dfc16_go<1, 8, 32, 4>(c, in, y, gi);
    case 5: return dfc16_go<2, 4, 32, 2>(c, in, y, gi);
    case 6: return dfc16_go<1, 4, 32, 4>(c, in, y, gi);
    case 7: return dfc16_go<1, 8, 32, 8>(c, in, y, gi);
    case 8: return dfc16_go<1, 16, 32, 8>(c, in, y, gi);
    case 9: return dfc16_go<2, 8, 32, 4>(c, in, y, gi);
    case 11: return dfc16s_go<4, 32, 4>(c, in, y, gi);   /* depth-streaming variants */
    case 12: return dfc16s_go<8, 32, 4>(c, in, y, gi);
    case 13: return dfc16s_go<16, 32, 8>(c, in, y, gi);
    case 14: return dfc16s_go<8, 32, 2>(c, in, y, gi);
    case 15: return dfc16s_go<16, 32, 4>(c, in, y, gi);
    case 16: return dfc16s_go<4, 32, 2>(c, in, y, gi);
    case 1: return dfc16_go<2, 8, 32, 8>(c, in, y, gi);   /* 8 warps of 4 m-tiles (consistently 1.5-1.7x the implicit GEMM in contended runs) */
    default: return dfc16s_go<8, 32, 8>(c, in, y, gi);   /* 10: depth-streaming, 2.1x k_dfc16 plain, 2.3x with the GN on load (XMX_DFBENCH) */
    }
}
static int g_cfg = -1, g_c16 = -1;   /* mgpu: env */
static void conv_launch(const ConvP &p, int co) {
    if (g_cfg < 0) g_cfg = getenv("UFSM_XNET_CFG") ? atoi(getenv("UFSM_XNET_CFG")) : 0;
    if (p.gm) {   /* GN on load (gna_ok): the default tiles */
        if (co <= 32) launch1<128, 32, 4, 1, 32, 3, true>(p, co); else if (co <= 64) launch1<128, 64, 4, 2, 32, 3, true>(p, co); else launch1<128, 128, 2, 4, 32, 3, true>(p, co);
        return;
    }
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
        if (co <= 16) { if (g_c16 < 0) g_c16 = getenv("UFSM_XNET_C16") ? atoi(getenv("UFSM_XNET_C16")) : 1;   /* 16 outputs (the Reader depth fusion): no half-empty tile */
                        if (g_c16 == 2) launch1<256, 16, 8, 1, 32, 3>(p, co); else if (g_c16) launch1<128, 16, 4, 1, 32, 3>(p, co); else launch1<128, 32, 4, 1, 32, 3>(p, co); }
        else if (co <= 32) launch1<128, 32, 4, 1, 32, 3>(p, co);
        else if (co <= 64) launch1<128, 64, 4, 2, 32, 3>(p, co); else launch1<128, 128, 2, 4, 32, 3>(p, co); break;
    }
}
__global__ void k_stats(const __half *x, int C, long long S, double *st) {   /* [n][C][2] sums of an fp16 NDHWC tensor; grid (chunks, n) */
    const int n = blockIdx.y; const long long per = (S + gridDim.x - 1) / gridDim.x, s0 = blockIdx.x * per, s1 = s0 + per < S ? s0 + per : S;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float a = 0.f, b = 0.f;
        for (long long v = s0; v < s1; v++) { const float f = __half2float(x[((size_t)n * S + v) * C + c]); a += f; b += f * f; }
        atomicAdd(st + ((size_t)n * C + c) * 2, (double)a); atomicAdd(st + ((size_t)n * C + c) * 2 + 1, (double)b);
    }
}
static void xstats(xt y, double *st) {
    const long long S = (long long)y.d * y.h * y.w; int ch = (int)((S + 1023) / 1024); if (ch > 1024) ch = 1024;
    k_stats<<<dim3(ch, y.n), y.c < 256 ? y.c : 256, 0, g_st>>>((const __half *)y.p, y.c, S, st);
}
static struct { int on; uint8_t *oq, *os; int oC; } g_tq;   /* xtconv_cat_q's target for the next MX tconv GEMM; mgpu: process (xnet runs on one host thread and one stream: set and consumed within one xtconv_cat_q call) */
static inline size_t xq_soff(size_t numel) { return (numel / 2 + 15) / 16 * 16; }   /* the scales after the nibbles in xt.q */
static inline size_t xq_size(size_t numel) { return xq_soff(numel) + numel / 8 + 16; }
/* a 1 -> 16 channel 3x3x3 conv (pad 1, stride 1) on a one-channel fp16 input: thread per output voxel, 27 inputs (L1) x 16 fp16 weights
   (the packed unfolded GEMM row, as the implicit GEMM used them), fp32 sums. Replaces xunfold (8 channels written) + the implicit GEMM
   (K 72 of mostly-zero taps): the Reader depth-fusion stem. */
__global__ void k_dfc1(const __half *x, const __half *w, int Kp, const float *scale, const float *bias, __half *y, long long nvox, int D, int H, int W) {
    __shared__ float sw[16 * 27];
    for (int i = threadIdx.x; i < 16 * 27; i += blockDim.x) { const int o = i / 27, t = i % 27; sw[i] = __half2float(w[(size_t)o * Kp + (t / 3) * 8 + t % 3]); }
    __syncthreads();
    const long long v = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (v >= nvox) return;
    const int xx = (int)(v % W); long long r = v / W; const int yy = (int)(r % H); r /= H; const int zz = (int)(r % D); const long long n = r / D;
    float in[27];
#pragma unroll
    for (int a = 0; a < 3; a++)
#pragma unroll
        for (int b = 0; b < 3; b++)
#pragma unroll
            for (int e = 0; e < 3; e++) {
                const int z = zz + a - 1, yv = yy + b - 1, xv = xx + e - 1;
                in[(a * 3 + b) * 3 + e] = z >= 0 && z < D && yv >= 0 && yv < H && xv >= 0 && xv < W ? __half2float(x[((n * D + z) * H + yv) * (long long)W + xv]) : 0.f;
            }
    __align__(16) __half o[16];
#pragma unroll
    for (int c = 0; c < 16; c++) { float s = 0.f;
#pragma unroll
        for (int t = 0; t < 27; t++) s += in[t] * sw[c * 27 + t];
        o[c] = __float2half_rn(s * scale[c] + bias[c]); }
    uint4 *dst = (uint4 *)(y + v * 16); dst[0] = ((const uint4 *)o)[0]; dst[1] = ((const uint4 *)o)[1];
}
/* k_dfc1 with XB consecutive x voxels a thread: each channel's 27 weights (7 float4 smem reads of [c][28]) feed XB FMA chains (k_dfc1:
   one broadcast LDS an FMA); every output's chain is k_dfc1's (27 taps in order, zero taps included): byte-identical. */
template <int XB> __global__ void __launch_bounds__(128) k_dfc1x(const __half *x, const __half *w, int Kp, const float *scale, const float *bias, __half *y, long long nthr, int D, int H, int W) {
    __shared__ __align__(16) float sw[16 * 28];
    for (int i = threadIdx.x; i < 16 * 28; i += blockDim.x) { const int o = i / 28, t = i % 28; sw[i] = t < 27 ? __half2float(w[(size_t)o * Kp + (t / 3) * 8 + t % 3]) : 0.f; }
    __syncthreads();
    const int WB = (W + XB - 1) / XB;
    const long long u = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (u >= nthr) return;
    const int xb = (int)(u % WB) * XB; long long r = u / WB; const int yy = (int)(r % H); r /= H; const int zz = (int)(r % D); const long long n = r / D;
    float in[27 * (XB + 2) / 3];   /* [a][b][e], e < XB + 2 */
#pragma unroll
    for (int a = 0; a < 3; a++)
#pragma unroll
        for (int b = 0; b < 3; b++)
#pragma unroll
            for (int e = 0; e < XB + 2; e++) {
                const int z = zz + a - 1, yv = yy + b - 1, xv = xb + e - 1;
                in[(a * 3 + b) * (XB + 2) + e] = z >= 0 && z < D && yv >= 0 && yv < H && xv >= 0 && xv < W ? __half2float(x[((n * D + z) * H + yv) * (long long)W + xv]) : 0.f;
            }
    unsigned pk[XB][8];
#pragma unroll
    for (int c = 0; c < 16; c++) {
        float wq[28];
#pragma unroll
        for (int t4 = 0; t4 < 7; t4++) { const float4 v = *(const float4 *)(sw + c * 28 + t4 * 4); wq[t4 * 4] = v.x; wq[t4 * 4 + 1] = v.y; wq[t4 * 4 + 2] = v.z; wq[t4 * 4 + 3] = v.w; }
        float s[XB];
#pragma unroll
        for (int k = 0; k < XB; k++) s[k] = 0.f;
#pragma unroll
        for (int t = 0; t < 27; t++)
#pragma unroll
            for (int k = 0; k < XB; k++) s[k] += in[(t / 3) * (XB + 2) + t % 3 + k] * wq[t];
#pragma unroll
        for (int k = 0; k < XB; k++) {
            const unsigned short h = __half_as_ushort(__float2half_rn(s[k] * scale[c] + bias[c]));
            if (c & 1) pk[k][c >> 1] |= (unsigned)h << 16; else pk[k][c >> 1] = h;
        }
    }
    const long long v0 = ((n * D + zz) * H + yy) * (long long)W + xb;
#pragma unroll
    for (int k = 0; k < XB; k++) if (xb + k < W) { uint4 *dst = (uint4 *)(y + (v0 + k) * 16); dst[0] = make_uint4(pk[k][0], pk[k][1], pk[k][2], pk[k][3]); dst[1] = make_uint4(pk[k][4], pk[k][5], pk[k][6], pk[k][7]); }
}
static int g_df1 = -1;   /* UFSM_XNET_DF1=0: the stem through xunfold + implicit GEMM (A/B); 1 k_dfc1; 2 (default) k_dfc1x<2> (XMX_DFBENCH 2.83 -> 1.75 ms), 4 <4> (2.38); mgpu: env */
extern "C" void xnet_set_df1(int v) { g_df1 = v; }
static bool dfc1_try(const xconv *c, xt x, const xt *res, int act, double *st, xt *out) {
    if (g_df1 < 0) g_df1 = getenv("UFSM_XNET_DF1") ? atoi(getenv("UFSM_XNET_DF1")) : 2;
    if (!g_df1 || !c->unfold || c->kw0 != 3 || c->pw0 != 1 || c->co != 16 || c->kd != 3 || c->kh != 3 || c->pd != 1 || c->ph != 1) return false;
    if (c->sd != 1 || c->sh != 1 || c->sw != 1 || x.c != 1 || res || act != XA_NONE || st) return false;
    *out = xt_new(x.n, x.d, x.h, x.w, 16); const long long nv = (long long)x.n * x.d * x.h * x.w;
    const __half *xp = (const __half *)x.p, *wp = (const __half *)c->w; __half *yp = (__half *)out->p; const long long rows = (long long)x.n * x.d * x.h;
    if (g_df1 == 2) { const long long nt = rows * ((x.w + 1) / 2); k_dfc1x<2><<<nblk((size_t)nt, 128), 128, 0, g_st>>>(xp, wp, c->Kp, c->scale, c->bias, yp, nt, x.d, x.h, x.w); }
    else if (g_df1 == 4) { const long long nt = rows * ((x.w + 3) / 4); k_dfc1x<4><<<nblk((size_t)nt, 128), 128, 0, g_st>>>(xp, wp, c->Kp, c->scale, c->bias, yp, nt, x.d, x.h, x.w); }
    else k_dfc1<<<nblk((size_t)nv, 128), 128, 0, g_st>>>(xp, wp, c->Kp, c->scale, c->bias, yp, nv, x.d, x.h, x.w);
    return true;
}
static xt conv_run(const xconv *c, xt x, const xt *res, int act, const float *lnw = nullptr, const float *lnb = nullptr, float lneps = 0.f, double *st = nullptr, const GnIn *gna = nullptr) {
    { xt o; if (!lnw && dfc1_try(c, x, res, act, st, &o)) { g_fq_seq++; g_macs += (double)xt_numel(o) * 27; return o; } }
    xt in = x;
    if (c->unfold) in = xunfold(x, c->kw0, c->pw0);
    fq_env(); xt fq = {0};
    const int seq = g_fq_seq++;
    if (g_fq_a && !c->unfold && in.c % 32 == 0 && (g_fq_mask >> (seq & 63) & 1)) {
        fq = xt_new(in.n, in.d, in.h, in.w, in.c); const long long nb = (long long)(xt_numel(in) / 32);
        k_fq<<<nblk(nb, 128), 128, 0, g_st>>>((const __half *)in.p, (__half *)fq.p, nb, g_fq_a); in = fq;
    }
    const int Do = (in.d + 2 * c->pd - c->kd) / c->sd + 1, Ho = (in.h + 2 * c->ph - c->kh) / c->sh + 1, Wo = (in.w + 2 * c->pw - c->kw) / c->sw + 1;
    xt y = g_tq.on && c->tconv == 2 && c->mxq ? xt{nullptr, in.n, Do, Ho, Wo, c->co, nullptr} : xt_new(in.n, Do, Ho, Wo, act & XA_GLU ? c->co / 2 : c->co);
    if (in.c != c->ci) fprintf(stderr, "xnet: conv input has %d channels, expected %d\n", in.c, c->ci);
    ConvP p;
    p.x = (const __half *)in.p; p.w = (const __half *)c->w; p.res = res ? (const __half *)res->p : nullptr; p.scale = c->scale; p.bias = c->bias; p.y = (__half *)y.p;
    p.Di = in.d; p.Hi = in.h; p.Wi = in.w; p.Ci = in.c; p.Do = Do; p.Ho = Ho; p.Wo = Wo; p.Co = c->co;
    p.kd = c->kd; p.kh = c->kh; p.kw = c->kw; p.sd = c->sd; p.sh = c->sh; p.sw = c->sw; p.pd = c->pd; p.ph = c->ph; p.pw = c->pw;
    p.K = c->K; p.Kp = c->Kp; p.M = (long long)in.n * Do * Ho * Wo; p.act = act; p.lnw = lnw; p.lnb = lnb; p.lneps = lneps;
    p.st = st && !res && act == XA_NONE ? st : nullptr;
    p.gm = p.grs = p.gg = p.gb = nullptr; p.gact = 0;
    if (gna) { p.gm = gna->gm; p.grs = gna->rs; p.gg = gna->g; p.gb = gna->b; p.gact = gna->act; }
    g_macs += (double)p.M * c->co * c->K;
    if (c->mxrot && !in.p) fprintf(stderr, "xnet: an MX conv with UFSM_XNET_MXROT / MXA needs its fp16 input (fused MX-only producers quantise the plain way)\n");
    if (c->mxq && !fq.p && in.c == c->ci && p.M > 0 && (in.p || (in.q && !c->mxrot)) && !(act & (XA_LN | XA_GLU)) && !res && (act & 15) == XA_NONE) {   /* the MX-fp4 path */
        const long long nb16 = (long long)(xt_numel(in) / 16); uint8_t *qb = c->mxrot ? nullptr : (uint8_t *)in.q, *qtmp = nullptr;   /* (a rotated conv re-quantises its input) */
        if (!qb) {
            qb = qtmp = (uint8_t *)xmalloc_dev(xq_size(xt_numel(in)));
            if (c->mxsf == 6) xq_launch<6>(c->mxrot, (const __half *)in.p, qb, qb + xq_soff(xt_numel(in)), nb16);
            else xq_launch<5>(c->mxrot, (const __half *)in.p, qb, qb + xq_soff(xt_numel(in)), nb16);
        }
        uint8_t *sb = qb + xq_soff(xt_numel(in));
        MxP m; m.xq = qb; m.xs = sb; m.wq = (const uint8_t *)c->mxq; m.ws = (const uint8_t *)c->mxs; m.gs = c->mxgs; m.scale = c->scale; m.bias = c->bias; m.y = (__half *)y.p;
        m.Di = in.d; m.Hi = in.h; m.Wi = in.w; m.Ci = in.c; m.Do = Do; m.Ho = Ho; m.Wo = Wo; m.Co = c->co;
        m.kd = c->kd; m.kh = c->kh; m.kw = c->kw; m.sd = c->sd; m.sh = c->sh; m.sw = c->sw; m.pd = c->pd; m.ph = c->ph; m.pw = c->pw;
        m.nch = c->nch; m.nsteps = (c->nch + 1) / 2; m.M = p.M; m.st = st; m.tco = 0; m.tb = nullptr; m.oq = m.os = nullptr; m.oC = m.oc0 = 0;
        if (g_tq.on && c->tconv == 2) { m.tco = c->co / 8; m.tb = c->tbias; m.oq = g_tq.oq; m.os = g_tq.os; m.oC = g_tq.oC; m.oc0 = 0; m.y = nullptr; }
        if (mxh_launch(m, c->mxsf, (const unsigned *)c->mxs2)) { xfree_dev(qtmp); if (c->unfold) xt_free(&in); return y; }
        if (c->nch > MX_MAXCH) fprintf(stderr, "xnet: MX conv with %d K chunks (> %d)\n", c->nch, MX_MAXCH);
        if (c->co <= 32) { if (c->mxsf == 6) mx_launch<256, 32, 6>(m); else mx_launch<256, 32, 5>(m); }
        else if (c->co <= 64 || p.M < 4096) { if (c->mxsf == 6) mx_launch<128, 64, 6>(m); else mx_launch<128, 64, 5>(m); }
        else { if (c->mxsf == 6) mx_launch<128, 128, 6>(m); else mx_launch<128, 128, 5>(m); }
        xfree_dev(qtmp);
        if (c->unfold) xt_free(&in);
        return y;
    }
    if (g_verbose) fprintf(stderr, "xnet conv %dx%dx%d s%d%d%d ci %d co %d: in %dx%dx%dx%d out %dx%dx%d M %lld\n", c->kd, c->kh, c->kw, c->sd, c->sh, c->sw, c->ci, c->co, in.n, in.d, in.h, in.w, Do, Ho, Wo, p.M);
    if (p.M > 0 && !dfc16_try(c, in, y, res, act, st)) conv_launch(p, c->co);
    if (st && !p.st && p.M > 0) xstats(y, st);
    if (c->unfold || fq.p) xt_free(&in);
    return y;
}
__global__ void k_gn_apply(const __half *x, const float *gm, const float *rs, const float *gamma, const float *beta, const __half *res, int act, __half *y, int C, long long S, size_t total);
/* y = conv(act(GroupNorm(x))) with the norm applied while the conv loads x (the 16 -> 16 depth-fusion conv; x unchanged), gm / rs from
   xgn_stats; other convs: the apply runs first (x is overwritten) */
extern "C" xt xconv_fwd_gnin(const xconv *c, xt x, const float *gm, const float *rs, const float *gamma, const float *beta, int act) {
    const int Do = (x.d + 2 * c->pd - c->kd) / c->sd + 1, Ho = (x.h + 2 * c->ph - c->kh) / c->sh + 1, Wo = (x.w + 2 * c->pw - c->kw) / c->sw + 1;
    xt y = xt_new(x.n, Do, Ho, Wo, c->co);
    if (dfc16_try(c, x, y, nullptr, XA_NONE, nullptr, GnIn{gm, rs, gamma, beta, act})) return y;
    xt_free(&y);
    static int gna = -1; if (gna < 0) gna = getenv("UFSM_XNET_GNA") ? atoi(getenv("UFSM_XNET_GNA")) : 1;   /* 0: apply, then the conv (A/B); mgpu: env */
    fq_env();
    if (gna && g_cfg <= 0 && !g_fq_a && !c->mxq && !c->unfold && !c->tconv && x.c == c->ci && x.c % 8 == 0 && c->co > 16 && (act & 15) == act) {
        const GnIn gi{gm, rs, gamma, beta, act};   /* conv_igemm normalises its A chunks as it stages them: k_gn_apply's values, one pass less */
        return conv_run(c, x, nullptr, XA_NONE, nullptr, nullptr, 0.f, nullptr, &gi);
    }
    const size_t total = xt_numel(x); unsigned nb = nblk(total / 2, 256); if (nb > 65535 * 4) nb = 65535 * 4;
    k_gn_apply<<<nb, 256, 0, g_st>>>((const __half *)x.p, gm, rs, gamma, beta, nullptr, act, (__half *)x.p, x.c, (long long)x.d * x.h * x.w, total);
    return xconv_fwd(c, x, nullptr, XA_NONE);
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

/* ---- the fused MX inference path (xnet.h) ---- */
extern "C" int xnet_mx(void) {
    if (g_mx_env < 0) { const char *e = getenv("UFSM_XNET_MX"); g_mx_env = e && *e ? atoi(e) : 0; if (g_mx_env != 0 && g_mx_env != 5 && g_mx_env != 6) g_mx_env = 0; }
    return g_mx_env;
}
extern "C" xt xconv_fwd_st(const xconv *c, xt x, double **st) {
    const int Do = c->unfold ? x.d : (x.d + 2 * c->pd - c->kd) / c->sd + 1;   /* (stats sized for the output) */
    const int Ho = (x.h + 2 * c->ph - c->kh) / c->sh + 1; (void)Ho;
    const size_t nb = (size_t)x.n * c->co * 2 * sizeof(double); (void)Do;
    *st = (double *)xmalloc_dev(nb); cudaMemsetAsync(*st, 0, nb, g_st);
    return conv_run(c, x, nullptr, XA_NONE, nullptr, nullptr, 0.f, *st);
}
__global__ void k_in_coef(const double *st, int nC, int C, double S, const float *gamma, const float *beta, float eps, float *coef) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= nC) return;
    const int c = i % C; const double m = st[2 * i] / S, v = fmax(st[2 * i + 1] / S - m * m, 0.0);
    const float rs = (float)(1.0 / sqrt(v + (double)eps)), a = rs * gamma[c];
    coef[2 * i] = a; coef[2 * i + 1] = beta[c] - (float)m * a;
}
extern "C" float *xin_coef(const double *st, int n, int C, long long S, const float *gamma, const float *beta, float eps) {
    float *coef = (float *)xmalloc_dev((size_t)n * C * 2 * sizeof(float));
    k_in_coef<<<nblk((size_t)n * C, 128), 128, 0, g_st>>>(st, n * C, C, (double)S, gamma, beta, eps, coef);
    return coef;
}
/* y = act(a x + b + r'), r' = a_r r + b_r (rcoef) or r; per (voxel, 16-channel block): fp16 y over x (want_half), MX q (want_q) */
template <int SF> __global__ void k_inq(__half *x, int C, long long S, long long nb16, const float *coef, const __half *res, const float *rcoef, int act, int wh, uint8_t *q, uint8_t *sq,
                                       const float *hw, float hb, float *hout, int res1) {
    long long b = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (hout) {   /* the head: logit = sum_c y_c w_c + hb over the C / 16 consecutive threads of a voxel (all lanes reach the shuffles) */
        const bool ok = b < nb16; const long long bb = ok ? b : 0; const long long e0 = bb * 16; const int c0 = (int)(e0 % C); const long long n = e0 / ((long long)C * S);
        const float *cf = coef + ((size_t)n * C + c0) * 2; float d = 0.f;
        if (ok) {
            __align__(16) __half hv[16]; *(uint4 *)hv = *(const uint4 *)(x + e0); *(uint4 *)(hv + 8) = *(const uint4 *)(x + e0 + 8);
#pragma unroll
            for (int i = 0; i < 16; i++) d += __half2float(__float2half_rn(act_f(__half2float(hv[i]) * cf[2 * i] + cf[2 * i + 1], act))) * hw[c0 + i];
        }
        for (int o = 1; o < C / 16; o <<= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        if (ok && c0 == 0) hout[e0 / C] = d + hb;
        return;
    }
    if (b >= nb16) return;
    const long long e0 = b * 16; const int c0 = (int)(e0 % C); const long long n = e0 / ((long long)C * S);
    const float *cf = coef + ((size_t)n * C + c0) * 2, *rc = rcoef ? rcoef + ((size_t)n * C + c0) * 2 : nullptr;
    __align__(16) __half hv[16], rv[16]; *(uint4 *)hv = *(const uint4 *)(x + e0); *(uint4 *)(hv + 8) = *(const uint4 *)(x + e0 + 8);
    if (res && res1) { const __half r1 = res[e0 / C]; for (int i = 0; i < 16; i++) rv[i] = r1; }
    else if (res) { *(uint4 *)rv = *(const uint4 *)(res + e0); *(uint4 *)(rv + 8) = *(const uint4 *)(res + e0 + 8); }
    float v[16], am = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        float f = __half2float(hv[i]) * cf[2 * i] + cf[2 * i + 1];
        if (res) { const float r = __half2float(rv[i]); f += rc ? r * rc[2 * i] + rc[2 * i + 1] : r; }
        f = act_f(f, act); v[i] = __half2float(__float2half_rn(f)); am = fmaxf(am, fabsf(v[i]));
    }
    if (wh) {
#pragma unroll
        for (int i = 0; i < 16; i++) hv[i] = __float2half_rn(v[i]);
        *(uint4 *)(x + e0) = *(const uint4 *)hv; *(uint4 *)(x + e0 + 8) = *(const uint4 *)(hv + 8);
    }
    if (!q) return;
    uint2 o; uint8_t sc;
    if constexpr (SF == 6) { const int e = mx_exp(am, 1.f / 6.f); const float m = exp2i(-e); o.x = cvt_e2m1x8(v, m); o.y = cvt_e2m1x8(v + 8, m); sc = (uint8_t)(e + 127); }
    else {
        sc = cvt_e4m3(__fdiv_rn(am, 6.f)); const int ex = (sc >> 3) & 15, mt = sc & 7; const float d = ex ? ldexpf(1.f + mt / 8.f, ex - 7) : ldexpf(mt / 8.f, -6);
        float w[16];
#pragma unroll
        for (int i = 0; i < 16; i++) w[i] = d > 0.f ? __fdiv_rn(v[i], d) : 0.f;
        o.x = cvt_e2m1x8(w, 1.f); o.y = cvt_e2m1x8(w + 8, 1.f);
    }
    *(uint2 *)(q + b * 8) = o; sq[(b >> 1) * 4 + (b & 1)] = sc;
}
extern "C" void xin_apply(xt *h, const float *coef, const xt *res, const float *rcoef, int act, int want_half, int want_q) {
    const size_t ne = xt_numel(*h); const long long nb16 = (long long)(ne / 16), S = (long long)h->d * h->h * h->w;
    uint8_t *q = nullptr;
    if (want_q) { if (!h->q) h->q = xmalloc_dev(xq_size(ne)); q = (uint8_t *)h->q; }
    const int sf = xnet_mx();
    if (sf == 5) k_inq<5><<<nblk(nb16, 128), 128, 0, g_st>>>((__half *)h->p, h->c, S, nb16, coef, res ? (const __half *)res->p : nullptr, rcoef, act, want_half, q, q ? q + xq_soff(ne) : nullptr, nullptr, 0.f, nullptr, res && res->c == 1 && h->c > 1);
    else k_inq<6><<<nblk(nb16, 128), 128, 0, g_st>>>((__half *)h->p, h->c, S, nb16, coef, res ? (const __half *)res->p : nullptr, rcoef, act, want_half, q, q ? q + xq_soff(ne) : nullptr, nullptr, 0.f, nullptr, res && res->c == 1 && h->c > 1);
    if (!want_half) { xfree_dev(h->p); h->p = nullptr; }
}
/* ---- a 1 -> 32 channel 3x3x3 conv (pad 1, bias) on a one-channel fp16 input, never stored: pass 0 accumulates the IN sums of its fp16
   outputs ([n][32][2]) and of the input ([n][2]); pass 1 recomputes it and writes act(a y + b (+ alpha x + beta)) as fp16 (yh) and / or
   MX-fp4 (q). Thread per output voxel (32 accumulators), stats reduced over the warp then atomics. ---- */
__constant__ float c_stem_w[32 * 27], c_stem_b[32];   /* xstem1's weights (FFMA constant operands; one stem net per process at a time); mgpu: process (one stem net per process; xnet is single-threaded, one device) */
__device__ __forceinline__ void stem_conv(const __half *x, long long n, int D, int H, int W, int zz, int yy, int xx, bool ok, float *v, float *xc) {
    float in[27];
#pragma unroll
    for (int a = 0; a < 3; a++)
#pragma unroll
        for (int b = 0; b < 3; b++)
#pragma unroll
            for (int c = 0; c < 3; c++) {
                const int zi = zz - 1 + a, yi = yy - 1 + b, xi = xx - 1 + c;
                in[(a * 3 + b) * 3 + c] = ok && zi >= 0 && zi < D && yi >= 0 && yi < H && xi >= 0 && xi < W ? __half2float(x[((n * D + zi) * H + yi) * (long long)W + xi]) : 0.f;
            }
#pragma unroll
    for (int o = 0; o < 32; o++) {
        float a = c_stem_b[o];
#pragma unroll
        for (int k = 0; k < 27; k++) a = fmaf(in[k], c_stem_w[o * 27 + k], a);
        v[o] = __half2float(__float2half_rn(a));   /* the fp16 conv output */
    }
    *xc = in[13];
}
enum { STEM_V = 8 };   /* voxels per thread in the sums pass */
__global__ void __launch_bounds__(128) k_stem_st(const __half *x, int N, int D, int H, int W, double *st, double *xst) {
    const long long S = (long long)D * H * W, NS = (long long)N * S, v0 = (long long)blockIdx.x * 128 * STEM_V;
    const long long n0 = v0 / S; const bool same = (v0 + 128 * STEM_V - 1 < NS ? v0 + 128 * STEM_V - 1 : NS - 1) / S == n0;
    float s[32], q[32], xs = 0.f, xq = 0.f;
#pragma unroll
    for (int o = 0; o < 32; o++) s[o] = q[o] = 0.f;
    for (int j = 0; j < STEM_V; j++) {
        const long long vox = v0 + j * 128 + threadIdx.x; const bool ok = vox < NS; const long long n = ok ? vox / S : 0; long long r = ok ? vox - n * S : 0;
        const int xx = (int)(r % W); r /= W; const int yy = (int)(r % H); const int zz = (int)(r / H);
        float v[32], xc; stem_conv(x, n, D, H, W, zz, yy, xx, ok, v, &xc);
        if (!ok) continue;
        if (same) {
#pragma unroll
            for (int o = 0; o < 32; o++) { s[o] += v[o]; q[o] += v[o] * v[o]; }
            xs += xc; xq += xc * xc;
        } else {
            for (int o = 0; o < 32; o++) { atomicAdd(st + (n * 32 + o) * 2, (double)v[o]); atomicAdd(st + (n * 32 + o) * 2 + 1, (double)v[o] * v[o]); }
            atomicAdd(xst + n * 2, (double)xc); atomicAdd(xst + n * 2 + 1, (double)xc * xc);
        }
    }
    if (!same) return;
    const int lane = threadIdx.x & 31; float my_s = 0.f, my_q = 0.f;
#pragma unroll
    for (int o = 0; o < 32; o++) {
        float a = s[o], b = q[o];
#pragma unroll
        for (int m = 16; m; m >>= 1) { a += __shfl_xor_sync(0xffffffffu, a, m); b += __shfl_xor_sync(0xffffffffu, b, m); }
        if (lane == o) { my_s = a; my_q = b; }
    }
    atomicAdd(st + (n0 * 32 + lane) * 2, (double)my_s); atomicAdd(st + (n0 * 32 + lane) * 2 + 1, (double)my_q);
#pragma unroll
    for (int m = 16; m; m >>= 1) { xs += __shfl_xor_sync(0xffffffffu, xs, m); xq += __shfl_xor_sync(0xffffffffu, xq, m); }
    if (lane == 0) { atomicAdd(xst + n0 * 2, (double)xs); atomicAdd(xst + n0 * 2 + 1, (double)xq); }
}
template <int SF>
__global__ void __launch_bounds__(128) k_stem(const __half *x, int N, int D, int H, int W, const float *coef, int act, __half *yh, uint8_t *q, uint8_t *sq) {
    const long long S = (long long)D * H * W, vox = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (vox >= (long long)N * S) return;
    const long long n = vox / S; long long r = vox - n * S;
    const int xx = (int)(r % W); r /= W; const int yy = (int)(r % H); const int zz = (int)(r / H);
    float v[32], xc; stem_conv(x, n, D, H, W, zz, yy, xx, true, v, &xc);
    const float *cf = coef + n * 64;
#pragma unroll
    for (int o = 0; o < 32; o++) v[o] = __half2float(__float2half_rn(act_f(v[o] * cf[2 * o] + cf[2 * o + 1], act)));
    if (yh) {
#pragma unroll
        for (int o = 0; o < 32; o += 8) { __align__(16) __half h8[8]; for (int i = 0; i < 8; i++) h8[i] = __float2half_rn(v[o + i]); *(uint4 *)(yh + vox * 32 + o) = *(const uint4 *)h8; }
    }
    if (q) {
        uint4 o4; uint8_t s2[2];
#pragma unroll
        for (int hb = 0; hb < 2; hb++) {
            float am = 0.f;
#pragma unroll
            for (int i = 0; i < 16; i++) am = fmaxf(am, fabsf(v[16 * hb + i]));
            unsigned lo, hi;
            if constexpr (SF == 6) { const int e = mx_exp(am, 1.f / 6.f); const float m = exp2i(-e); lo = cvt_e2m1x8(v + 16 * hb, m); hi = cvt_e2m1x8(v + 16 * hb + 8, m); s2[hb] = (uint8_t)(e + 127); }
            else { const uint8_t sc = cvt_e4m3(__fdiv_rn(am, 6.f)); const int ex = (sc >> 3) & 15, mt = sc & 7; const float d = ex ? ldexpf(1.f + mt / 8.f, ex - 7) : ldexpf(mt / 8.f, -6);
                   float ww[16];
#pragma unroll
                   for (int i = 0; i < 16; i++) ww[i] = d > 0.f ? __fdiv_rn(v[16 * hb + i], d) : 0.f;
                   lo = cvt_e2m1x8(ww, 1.f); hi = cvt_e2m1x8(ww + 8, 1.f); s2[hb] = sc; }
            if (hb == 0) { o4.x = lo; o4.y = hi; } else { o4.z = lo; o4.w = hi; }
        }
        *(uint4 *)(q + vox * 16) = o4; *(uint32_t *)(sq + vox * 4) = (uint32_t)s2[0] | (uint32_t)s2[1] << 8;
    }
}
/* skip = IN(1x1 conv w_sk of the one-channel x): r'_c = alpha_c x + beta_c from the input's sums (exact: the 1x1 conv is a scale) */
__global__ void k_skip1(const double *xst, int N, double S, const float *wsk, const float *gamma, const float *beta, float eps, float *rcoef) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= N * 32) return;
    const int n = i / 32, c = i % 32; const double m = xst[2 * n] / S, v = fmax(xst[2 * n + 1] / S - m * m, 0.0), w = wsk[c];
    const double a = gamma[c] / sqrt(w * w * v + (double)eps);
    rcoef[2 * i] = (float)(a * w); rcoef[2 * i + 1] = (float)(beta[c] - a * w * m);
}
extern "C" xt xstem1(xt x, const float *w_dev, const float *b_dev, const float *gamma, const float *beta, float eps, const float *wsk_dev, const float *gsk, const float *bsk,
                     float **rcoef, int act, int want_half, int want_q) {
    const long long S = (long long)x.d * x.h * x.w, nv = (long long)x.n * S;
    double *st = (double *)xmalloc_dev((size_t)x.n * 66 * sizeof(double)), *xst = st + (size_t)x.n * 64;
    cudaMemsetAsync(st, 0, (size_t)x.n * 66 * sizeof(double), g_st);
    const int sf = xnet_mx(); const unsigned nb = nblk((size_t)nv, 128);
    cudaMemcpyToSymbolAsync(c_stem_w, w_dev, sizeof(float) * 32 * 27, 0, cudaMemcpyDeviceToDevice, g_st);
    if (b_dev) cudaMemcpyToSymbolAsync(c_stem_b, b_dev, sizeof(float) * 32, 0, cudaMemcpyDeviceToDevice, g_st);
    else { static const float z[32] = {0}; cudaMemcpyToSymbolAsync(c_stem_b, z, sizeof z, 0, cudaMemcpyHostToDevice, g_st); }
    k_stem_st<<<nblk((size_t)nv, 128 * STEM_V), 128, 0, g_st>>>((const __half *)x.p, x.n, x.d, x.h, x.w, st, xst);
    float *coef = xin_coef(st, x.n, 32, S, gamma, beta, eps), *rc = nullptr;
    if (wsk_dev) { rc = (float *)xmalloc_dev((size_t)x.n * 64 * sizeof(float)); k_skip1<<<nblk((size_t)x.n * 32, 128), 128, 0, g_st>>>(xst, x.n, (double)S, wsk_dev, gsk, bsk, eps, rc); }
    xt y = {nullptr, x.n, x.d, x.h, x.w, 32, nullptr};
    if (want_half) y.p = xmalloc_dev((size_t)nv * 32 * 2);
    if (want_q) y.q = xmalloc_dev(xq_size((size_t)nv * 32));
    uint8_t *q = (uint8_t *)y.q, *sq = q ? q + xq_soff((size_t)nv * 32) : nullptr;
    if (sf == 5) k_stem<5><<<nb, 128, 0, g_st>>>((const __half *)x.p, x.n, x.d, x.h, x.w, coef, act, (__half *)y.p, q, sq);
    else k_stem<6><<<nb, 128, 0, g_st>>>((const __half *)x.p, x.n, x.d, x.h, x.w, coef, act, (__half *)y.p, q, sq);
    xfree_dev(st); xfree_dev(coef);
    if (rcoef) *rcoef = rc; else xfree_dev(rc);
    return y;
}
extern "C" int xin_head(xt *h, const float *coef, int act, const float *hw_dev, float hb, float *out_dev) {
    if (h->c % 16 || 32 % (h->c / 16) || h->c > 512) return -1;
    const size_t ne = xt_numel(*h); const long long nb16 = (long long)(ne / 16), S = (long long)h->d * h->h * h->w;
    k_inq<6><<<nblk(nb16, 128), 128, 0, g_st>>>((__half *)h->p, h->c, S, nb16, coef, nullptr, nullptr, act, 0, nullptr, nullptr, hw_dev, hb, out_dev, 0);
    return 0;
}
extern "C" int xq_ensure(xt *x) {
    if (x->q) return 0;
    const size_t ne = xt_numel(*x); const long long nb16 = (long long)(ne / 16); if (x->c % 32) return -1;
    x->q = xmalloc_dev(xq_size(ne)); if (!x->q) return -1;
    uint8_t *q = (uint8_t *)x->q;
    if (xnet_mx() == 5) k_xq<5><<<nblk(nb16, 256), 256, 0, g_st>>>((const __half *)x->p, q, q + xq_soff(ne), nb16);
    else k_xq<6><<<nblk(nb16, 256), 256, 0, g_st>>>((const __half *)x->p, q, q + xq_soff(ne), nb16);
    return 0;
}
/* ConvTranspose3d of lo (MX GEMM; the epilogue shuffles, adds the bias and quantises) written straight into channels [0, co) of an MX
   concat with skip's MX copy as channels [co, co + skip.c): the decoder input, with q only */
extern "C" xt xtconv_cat_q(const xconv *up, xt lo, xt skip) {
    const int co = up->co / 8; xt y = {nullptr, lo.n, 2 * lo.d, 2 * lo.h, 2 * lo.w, co + skip.c, nullptr};
    const size_t rows = (size_t)y.n * y.d * y.h * y.w, ne = rows * y.c;
    if (!up->mxq || co % 16 || (size_t)skip.n * skip.d * skip.h * skip.w != rows || !skip.q) { xt u = xconv_fwd(up, lo, nullptr, XA_NONE); xq_ensure(&u); xt c = xconcat_q(u, skip); xt_free(&u); return c; }
    y.q = xmalloc_dev(xq_size(ne)); uint8_t *yq = (uint8_t *)y.q, *ys = yq + xq_soff(ne);
    g_tq.on = 1; g_tq.oq = yq; g_tq.os = ys; g_tq.oC = y.c;
    xt g = conv_run(up, lo, nullptr, XA_NONE); g_tq.on = 0;
    xt_free(&g);   /* (allocated, not written) */
    const uint8_t *bq = (const uint8_t *)skip.q, *bs = bq + xq_soff(rows * skip.c);
    cudaMemcpy2DAsync(yq + co / 2, (size_t)y.c / 2, bq, (size_t)skip.c / 2, (size_t)skip.c / 2, rows, cudaMemcpyDeviceToDevice, g_st);
    cudaMemcpy2DAsync(ys + co / 8, (size_t)y.c / 8, bs, (size_t)skip.c / 8, (size_t)skip.c / 8, rows, cudaMemcpyDeviceToDevice, g_st);
    return y;
}
extern "C" xt xconcat_q(xt a, xt b) {
    xt y = {nullptr, a.n, a.d, a.h, a.w, a.c + b.c, nullptr}; const size_t rows = (size_t)a.n * a.d * a.h * a.w, ne = rows * y.c;
    y.q = xmalloc_dev(xq_size(ne));
    uint8_t *yq = (uint8_t *)y.q; const uint8_t *aq = (const uint8_t *)a.q, *bq = (const uint8_t *)b.q;
    cudaMemcpy2DAsync(yq, (size_t)y.c / 2, aq, (size_t)a.c / 2, (size_t)a.c / 2, rows, cudaMemcpyDeviceToDevice, g_st);
    cudaMemcpy2DAsync(yq + a.c / 2, (size_t)y.c / 2, bq, (size_t)b.c / 2, (size_t)b.c / 2, rows, cudaMemcpyDeviceToDevice, g_st);
    uint8_t *ys = yq + xq_soff(ne); const uint8_t *as = aq + xq_soff(rows * a.c), *bs = bq + xq_soff(rows * b.c);
    cudaMemcpy2DAsync(ys, (size_t)y.c / 8, as, (size_t)a.c / 8, (size_t)a.c / 8, rows, cudaMemcpyDeviceToDevice, g_st);
    cudaMemcpy2DAsync(ys + a.c / 8, (size_t)y.c / 8, bs, (size_t)b.c / 8, (size_t)b.c / 8, rows, cudaMemcpyDeviceToDevice, g_st);
    return y;
}

/* ---- GroupNorm / InstanceNorm ---- */
/* deterministic one-pass statistics: block (chunk, n) sums x - sh_c and its square per channel (sh_c = the sample's first voxel: no
   cancellation), each thread a fixed channel pair and a fixed row stride, rows combined in a fixed order -> part[n][chunk][C][2]
   (double). The old two-pass kernel combined with shared / global float atomics, so its sums depended on the schedule: harmless in
   fp16, but every flip of an MX-fp4 rounding downstream moved the 2-D teachers' AUC by up to 0.004 per scroll between identical runs. */
__global__ void k_gnpart(const __half *x, int C, long long S, int nchunk, double *part) {
    extern __shared__ float sh[];   /* [rows][C][2] */
    const int n = blockIdx.y, pairs = C / 2, t = threadIdx.x; const int rows = pairs <= (int)blockDim.x ? (int)blockDim.x / pairs : 1;
    const long long per = (S + nchunk - 1) / nchunk, s0 = blockIdx.x * per, s1 = s0 + per < S ? s0 + per : S;
    const __half *xb = x + (size_t)n * S * C;
    for (int cp0 = pairs <= (int)blockDim.x ? t % pairs : t; cp0 < pairs; cp0 += (int)blockDim.x) {
        const int row = pairs <= (int)blockDim.x ? t / pairs : 0; if (row >= rows) break;
        const int c = cp0 * 2; const __half2 h0 = *(const __half2 *)(xb + c); const float m0 = __low2float(h0), m1 = __high2float(h0);
        float a0 = 0.f, a1 = 0.f, q0 = 0.f, q1 = 0.f;
        for (long long v = s0 + row; v < s1; v += rows) {
            const __half2 h = *(const __half2 *)(xb + (size_t)v * C + c); const float f0 = __low2float(h) - m0, f1 = __high2float(h) - m1;
            a0 += f0; a1 += f1; q0 += f0 * f0; q1 += f1 * f1;
        }
        float *o = sh + ((size_t)row * C + c) * 2; o[0] = a0; o[1] = q0; o[2] = a1; o[3] = q1;
        if (pairs <= (int)blockDim.x) break;
    }
    __syncthreads();
    for (int c = t; c < C; c += blockDim.x) {
        double a = 0.0, q = 0.0;
        for (int r = 0; r < rows; r++) { a += sh[((size_t)r * C + c) * 2]; q += sh[((size_t)r * C + c) * 2 + 1]; }
        double *o = part + (((size_t)n * nchunk + blockIdx.x) * C + c) * 2; o[0] = a; o[1] = q;
    }
}
__global__ void k_gnfin(const __half *x, const double *part, int n, int C, int G, long long S, int nchunk, float eps, float *gm, float *rs) {   /* thread per (n, group) */
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n * G) return;
    const int nn = i / G, g = i % G, cpg = C / G; double sx = 0.0, sxx = 0.0;
    for (int c = g * cpg; c < (g + 1) * cpg; c++) {
        double a = 0.0, q = 0.0; const double m = __half2float(x[(size_t)nn * S * C + c]);
        for (int k = 0; k < nchunk; k++) { const double *p = part + (((size_t)nn * nchunk + k) * C + c) * 2; a += p[0]; q += p[1]; }
        sx += a + (double)S * m; sxx += q + 2.0 * m * a + (double)S * m * m;
    }
    const double N = (double)S * cpg, mean = sx / N, var = fmax(sxx / N - mean * mean, 0.0); const float r = (float)(1.0 / sqrt(var + eps));
    for (int c = g * cpg; c < (g + 1) * cpg; c++) { gm[(size_t)nn * C + c] = (float)mean; rs[(size_t)nn * C + c] = r; }
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
/* the group statistics of x: device gm / rs [n][C] (group mean, 1 / std per channel, one allocation: rs = gm + n C; free gm) */
extern "C" float *xgn_stats(xt x, int groups, float eps) {
    const int C = x.c, n = x.n; const long long S = (long long)x.d * x.h * x.w;
    int nchunk = (int)((512 + n - 1) / n); if (nchunk > S / 64) nchunk = (int)(S / 64); if (nchunk < 1) nchunk = 1;   /* ~512 blocks, >= 64 voxels each */
    double *part = (double *)xmalloc_dev((size_t)n * nchunk * C * 2 * sizeof(double)); float *gm = (float *)xmalloc_dev((size_t)n * C * 4 * 2), *rs = gm + (size_t)n * C;
    const int pairs = C / 2, rows = pairs <= 256 ? 256 / pairs : 1;
    k_gnpart<<<dim3(nchunk, n), 256, (size_t)rows * C * 2 * sizeof(float), g_st>>>((const __half *)x.p, C, S, nchunk, part);
    k_gnfin<<<nblk((size_t)n * groups, 64), 64, 0, g_st>>>((const __half *)x.p, part, n, C, groups, S, nchunk, eps, gm, rs);
    xfree_dev(part);
    return gm;
}
extern "C" void xgn(xt x, int groups, const float *gamma, const float *beta, float eps, const xt *res, int act, xt y) {
    const int C = x.c, n = x.n; const long long S = (long long)x.d * x.h * x.w;
    float *gm = xgn_stats(x, groups, eps), *rs = gm + (size_t)n * C;
    const size_t total = xt_numel(x); unsigned nb = nblk(total / 2, 256); if (nb > 65535 * 4) nb = 65535 * 4;
    k_gn_apply<<<nb, 256, 0, g_st>>>((const __half *)x.p, gm, rs, gamma, beta, res ? (const __half *)res->p : nullptr, act, (__half *)y.p, C, S, total);
    xfree_dev(gm);
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
/* the Reader depth collapse in one pass (replaces xgn apply + xhead + xdepth_attn_max on the 17-layer tensor): thread per pixel, the 16
   channels of each layer normalised and activated as k_gn_apply (rounded to fp16), the attention logit w . v + b, then softmax over the
   layers (as k_attn_max: max, sum of exp, weights exp / den) re-reading the layers for the weighted sum and the max. f is read twice
   (L2) and never written: the separate path wrote f and the logits and read f three times. */
template <int DM> __global__ void k_dattn(const __half *f, const float *gm, const float *rs, const float *gamma, const float *beta, int act, const float *w, float b,
                                          __half *y, int n, int D, long long HW) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i >= (long long)n * HW) return;
    const long long nn = i / HW, p = i - nn * HW; const float *g = gm + nn * 16, *r = rs + nn * 16;
    float a[DM], am = -INFINITY;
    auto load = [&](int d, float *v) {
        __align__(16) __half hv[16]; const __half *src = f + ((nn * D + d) * HW + p) * 16;
        *(uint4 *)hv = *(const uint4 *)src; *(uint4 *)(hv + 8) = *(const uint4 *)(src + 8);
#pragma unroll
        for (int c = 0; c < 16; c++) v[c] = __half2float(__float2half_rn(act_f((__half2float(hv[c]) - g[c]) * r[c] * gamma[c] + beta[c], act)));
    };
    for (int d = 0; d < D; d++) {
        float v[16]; load(d, v); float s = 0.f;
#pragma unroll
        for (int c = 0; c < 16; c++) s += v[c] * w[c];
        a[d] = s + b; am = fmaxf(am, a[d]);
    }
    float den = 0.f; for (int d = 0; d < D; d++) den += __expf(a[d] - am);
    float sm[16], mx[16];
#pragma unroll
    for (int c = 0; c < 16; c++) { sm[c] = 0.f; mx[c] = -INFINITY; }
    for (int d = 0; d < D; d++) {
        float v[16]; load(d, v); const float e = __expf(a[d] - am) / den;
#pragma unroll
        for (int c = 0; c < 16; c++) { sm[c] += v[c] * e; mx[c] = fmaxf(mx[c], v[c]); }
    }
    __align__(16) __half o[32];
#pragma unroll
    for (int c = 0; c < 16; c++) { o[c] = __float2half_rn(sm[c]); o[16 + c] = __float2half_rn(mx[c]); }
    uint4 *dst = (uint4 *)(y + i * 32);
#pragma unroll
    for (int k = 0; k < 4; k++) dst[k] = ((const uint4 *)o)[k];
}
extern "C" xt xdepth_collapse16(xt f, int groups, const float *gamma, const float *beta, float eps, int act, const float *w_dev, float bias) {
    if (f.c != 16 || f.d > 32) { fprintf(stderr, "xnet: xdepth_collapse16 needs 16 channels and <= 32 layers (%d, %d)\n", f.c, f.d); return xt{}; }
    float *gm = xgn_stats(f, groups, eps); const long long HW = (long long)f.h * f.w;
    xt y = xt_new(f.n, 1, f.h, f.w, 32);
    k_dattn<32><<<nblk((size_t)f.n * HW, 128), 128, 0, g_st>>>((const __half *)f.p, gm, gm + (size_t)f.n * 16, gamma, beta, act, w_dev, bias, (__half *)y.p, f.n, f.d, HW);
    xfree_dev(gm);
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
