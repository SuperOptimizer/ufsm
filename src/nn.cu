/* CUDA ops behind src/nn.h. fp32, NCDHW. Direct convolutions with shared-memory tiling; no cuBLAS/cuDNN. */
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
using namespace nvcuda;

static int g_tf32 = 1;   /* 1 = tensor-core implicit GEMM for 3^3 convs, 0 = exact fp32 CUDA-core kernels */
extern "C" void nn_set_f16(int on);
static int g_prec = 1, g_pref = 1;   /* precision of the tensor-core path: 1 bf16, 2 fp8 (e4m3, MX block scales), 3 fp4 forward/backward-data + fp8 weight gradient (src/nn_fp8.cu) */
extern "C" void nn_set_tf32(int on) { g_tf32 = on; g_prec = on ? g_pref : 0; }
extern "C" void nn_set_prec(int p) { g_prec = p; g_tf32 = p > 0; if (p > 0) g_pref = p; }   /* the fp8/fp4 kernels accept fp32, bf16 or fp16 storage */
extern "C" int nn_get_prec(void) { return g_prec; }
/* per-layer precision: unet.c tags each conv with a layer id; a layer may override the global tensor-core precision */
#define NN_MAXLAYER 32
static int g_layer = -1, g_lprec[NN_MAXLAYER];
static int g_lprec_init = 0;
static void lprec_init(void) { if (!g_lprec_init) { for (int i = 0; i < NN_MAXLAYER; i++) g_lprec[i] = -1; g_lprec_init = 1; } }
extern "C" void nn_set_layer(int id) { g_layer = id; }
extern "C" void nn_set_layer_prec(int id, int p) { lprec_init(); if (id >= 0 && id < NN_MAXLAYER) g_lprec[id] = p; }
/* finer policy: per conv of a layer (sub 0 = c1, 1 = c2; down / head use sub 0) and per pass (0 forward, 1 backward-data,
   2 weight gradient); 0 = not set (falls back to the layer precision, then the global one) */
static int g_sub = -1, g_pass = 0;
static signed char g_lprec3[NN_MAXLAYER][2][3];
extern "C" void nn_set_conv(int sub) { g_sub = sub; }
extern "C" int nn_get_layer(void) { return g_layer; }
extern "C" int nn_get_conv(void) { return g_sub; }
extern "C" void nn_set_conv_prec(int id, int sub, int p_fwd, int p_bwd_data, int p_wgrad) {
    if (id < 0 || id >= NN_MAXLAYER) return;
    for (int s2 = 0; s2 < 2; s2++) if (sub < 0 || sub == s2) { g_lprec3[id][s2][0] = (signed char)(p_fwd > 0 ? p_fwd : 0); g_lprec3[id][s2][1] = (signed char)(p_bwd_data > 0 ? p_bwd_data : 0); g_lprec3[id][s2][2] = (signed char)(p_wgrad > 0 ? p_wgrad : 0); }
}
extern "C" int nn_get_conv_prec(int id, int sub, int pass) { return id >= 0 && id < NN_MAXLAYER && sub >= 0 && sub < 2 && pass >= 0 && pass < 3 ? g_lprec3[id][sub][pass] : 0; }
/* quantization-aware training: the weight gradient may use its own (higher) precision while forward / backward-data run
   at the deployment precision; -1 = same as the layer precision */
static int g_prec_w = -1;
extern "C" void nn_set_prec_wgrad(int p) { g_prec_w = p; }
static int eff_prec_pass(int pass) {
    lprec_init();
    if (!g_tf32) return 0;
    if (pass == 2 && g_prec_w >= 1) return g_prec_w;
    if (g_layer >= 0 && g_layer < NN_MAXLAYER) {
        int p3 = g_lprec3[g_layer][g_sub > 0 ? 1 : 0][pass];
        if (p3 >= 1) return p3;
        if (g_lprec[g_layer] >= 1) return g_lprec[g_layer];
    }
    return g_prec;
}
static int eff_prec(void) { return eff_prec_pass(g_pass); }
static int eff_prec_w(void) { return eff_prec_pass(2); }
extern "C" int nn_cur_prec(void) { return eff_prec(); }
extern "C" int nn_prec_parse(const char *s) {
    static const char *nm[] = {"fp32", "bf16", "fp8", "fp4", "fp16"};
    for (int i = 0; i < 5; i++) if (!strcmp(s, nm[i])) return i;
    char *e; long v = strtol(s, &e, 10);
    return *s && !*e && v >= 0 && v <= 4 ? (int)v : -1;
}
extern "C" const char *nn_prec_name(int p) { static const char *nm[] = {"fp32", "bf16", "fp8", "fp4", "fp16"}; return p >= 0 && p <= 4 ? nm[p] : "?"; }
/* policy string: "enc0=1,enc1=2,down0=2,dec2=3,head=1" or positional "1,1,2,2,2,2,2,2,2,1,1" (unet order: enc0..3, down0..2,
   dec2, dec1, dec0, head); values 1 bf16, 2 fp8, 3 fp4, 4 fp16 (fp16 operands, fp16 group accumulation) or their names;
   a layer left out keeps the global precision. Finer entries: a single conv of a block (enc1.c2, dec0.c1), "all", and
   per-pass values fwd:bwd_data:wgrad (e.g. dec0=fp16:fp16:fp8). Later entries override earlier ones. */
extern "C" int nn_set_prec_policy(const char *pol) {
    static const char *names[] = {"enc0", "enc1", "enc2", "enc3", "down0", "down1", "down2", "dec2", "dec1", "dec0", "head"};
    lprec_init();
    for (int i = 0; i < NN_MAXLAYER; i++) { g_lprec[i] = -1; nn_set_conv_prec(i, -1, 0, 0, 0); }
    if (!pol || !*pol) return 0;
    char buf[2048]; snprintf(buf, sizeof buf, "%s", pol);
    int pos = 0;
    char *save = nullptr;
    for (char *tok = strtok_r(buf, ", ", &save); tok; tok = strtok_r(nullptr, ", ", &save)) {
        char *eq = strchr(tok, '=');
        if (!eq) { int v = nn_prec_parse(tok); if (v < 1 || pos >= 11) { fprintf(stderr, "nn_set_prec_policy: bad entry '%s'\n", tok); return -1; } g_lprec[pos++] = v; continue; }
        *eq = 0;
        char *val = eq + 1, *c1 = strchr(val, ':'), *c2 = c1 ? strchr(c1 + 1, ':') : nullptr;
        int pv[3];
        if (c1 && c2) { *c1 = *c2 = 0; pv[0] = nn_prec_parse(val); pv[1] = nn_prec_parse(c1 + 1); pv[2] = nn_prec_parse(c2 + 1); }
        else if (!c1) pv[0] = pv[1] = pv[2] = nn_prec_parse(val);
        else pv[0] = -1;
        if (pv[0] < 1 || pv[1] < 1 || pv[2] < 1) { fprintf(stderr, "nn_set_prec_policy: bad precision in '%s'\n", val); return -1; }
        int sub = -1;
        char *dot = strchr(tok, '.');
        if (dot) { if (!strcmp(dot, ".c1")) sub = 0; else if (!strcmp(dot, ".c2")) sub = 1; else { fprintf(stderr, "nn_set_prec_policy: bad conv '%s'\n", tok); return -1; } *dot = 0; }
        int lo = -1, hi = -1;
        if (!strcmp(tok, "all")) { lo = 0; hi = 10; }
        else for (int i = 0; i < 11; i++) if (!strcmp(tok, names[i])) lo = hi = i;
        if (lo < 0) { fprintf(stderr, "nn_set_prec_policy: unknown layer '%s'\n", tok); return -1; }
        for (int i = lo; i <= hi; i++) {
            if (sub < 0 && pv[0] == pv[1] && pv[1] == pv[2]) { g_lprec[i] = pv[0]; nn_set_conv_prec(i, -1, 0, 0, 0); }
            else nn_set_conv_prec(i, sub, pv[0], pv[1], pv[2]);
        }
    }
    return 0;
}
extern "C" int nn_get_tf32(void) { return g_tf32; }
static int g_actbf = 1;  /* 1 = activations stored as bf16 in tensor-core mode (gradients stay fp32) */
extern "C" void nn_set_act_bf16(int on) { g_actbf = on; }
extern "C" int nn_get_act_bf16(void) { return g_actbf; }
#define ABF (g_tf32 && g_actbf)
static int g_h16 = 0;     /* 16-bit type for activation/gradient storage and the MMA operands: 0 bf16, 1 fp16 (8x finer mantissa, same rate) */
extern "C" void nn_set_f16(int on) { g_h16 = on; }
extern "C" int nn_get_f16(void) { return g_h16; }
#define LPDT(flag) ((flag) ? (g_h16 ? 2 : 1) : 0)   /* storage code of the lp_* (nn_fp8.cu) entry points: 0 fp32, 1 bf16, 2 fp16 */
static float g_gscale = 1.f;   /* activation gradients are stored scaled by this (fp16 storage range); parameter grads are unscaled by the network */
extern "C" void nn_set_grad_scale(float s) { g_gscale = s; }
extern "C" float nn_get_grad_scale(void) { return g_gscale; }
static int g_gradbf = 1;  /* 1 = activation gradients stored as bf16 as well (requires act-bf16) */
extern "C" void nn_set_grad_bf16(int on) { g_gradbf = on; }
extern "C" int nn_get_grad_bf16(void) { return g_gradbf; }
#define GBF (ABF && g_gradbf)

static cudaError_t g_err = cudaSuccess;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess && g_err == cudaSuccess) g_err = e_; } while (0)
#define KCHECK() CK(cudaGetLastError())

extern "C" int nn_init(int device) { if (getenv("UFSM_ACTF32")) g_actbf = 0; if (getenv("UFSM_GRADF32")) g_gradbf = 0; if (getenv("UFSM_F16")) { g_h16 = 1; g_gscale = getenv("UFSM_GSCALE") ? (float)atof(getenv("UFSM_GSCALE")) : 1024.f; } return cudaSetDevice(device) == cudaSuccess ? 0 : -1; }
extern "C" const char *nn_check(void) {
    cudaError_t e = g_err;
    g_err = cudaSuccess;
    if (e == cudaSuccess) e = cudaGetLastError();
    return e == cudaSuccess ? nullptr : cudaGetErrorString(e);
}
extern "C" void *nn_malloc(size_t n) { void *p = nullptr; CK(cudaMalloc(&p, n)); return p; }
/* ---- per-tensor storage registry: a tensor registered as MX-fp8 (dt 8) takes the MX paths of the ops that read or
   write it (see nn_set_storage in nn.h); lookups match any address inside a registered range ---- */
#define NN_MAXREG 1024
static struct { const char *p; size_t n; int dt; } g_reg[NN_MAXREG];
static int g_nreg;
extern "C" void nn_storage_forget(const void *p) { for (int i = 0; i < g_nreg; i++) if (g_reg[i].p == (const char *)p) { g_reg[i] = g_reg[--g_nreg]; return; } }
extern "C" void nn_set_storage(const void *p, size_t bytes, int dt) {
    nn_storage_forget(p);
    if (!p || !dt) return;
    if (g_nreg >= NN_MAXREG) { fprintf(stderr, "nn_set_storage: registry full\n"); abort(); }
    g_reg[g_nreg].p = (const char *)p; g_reg[g_nreg].n = bytes; g_reg[g_nreg].dt = dt; g_nreg++;
}
extern "C" int nn_storage(const void *p) {
    const char *c = (const char *)p;
    for (int i = 0; i < g_nreg; i++) if (c >= g_reg[i].p && c < g_reg[i].p + g_reg[i].n) return g_reg[i].dt;
    return 0;
}
#define ISMX(p) (g_nreg && nn_storage(p) == 8)
extern "C" size_t nn_mx8_bytes(shape5 s) { int bw = s.c <= 16 ? 16 : 32, nb = (s.c + bw - 1) / bw; return (size_t)s.n * nb * shape_spatial(s) * (bw + 1); }   /* = lp_mx8_bytes */
extern "C" void nn_free(void *p) { if (p) { if (g_nreg) nn_storage_forget(p); CK(cudaFree(p)); } }
extern "C" void nn_zero(void *p, size_t n) { CK(cudaMemset(p, 0, n)); }
extern "C" void nn_h2d(void *d, const void *s, size_t n) { CK(cudaMemcpy(d, s, n, cudaMemcpyHostToDevice)); }
extern "C" void nn_d2h(void *d, const void *s, size_t n) { CK(cudaMemcpy(d, s, n, cudaMemcpyDeviceToHost)); }
extern "C" void nn_d2d(void *d, const void *s, size_t n) { CK(cudaMemcpy(d, s, n, cudaMemcpyDeviceToDevice)); }
extern "C" void nn_sync(void) { CK(cudaDeviceSynchronize()); }
extern "C" void *nn_host_alloc(size_t n) { void *p = nullptr; CK(cudaMallocHost(&p, n)); return p; }
extern "C" void nn_host_free(void *p) { if (p) CK(cudaFreeHost(p)); }
/* per-device copy stream (non-blocking, overlaps the legacy compute stream) and events for host<->device pipelining */
static cudaStream_t copy_stream(void) {
    static cudaStream_t st[8]; int d = 0; cudaGetDevice(&d); d &= 7;
    if (!st[d]) CK(cudaStreamCreateWithFlags(&st[d], cudaStreamNonBlocking));
    return st[d];
}
extern "C" void nn_h2d_copy_stream(void *d, const void *s, size_t n) { CK(cudaMemcpyAsync(d, s, n, cudaMemcpyHostToDevice, copy_stream())); }   /* s must be pinned */
extern "C" void *nn_event_create(void) { cudaEvent_t e; CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming)); return (void *)e; }
extern "C" void nn_event_record(void *e, int on_copy_stream) { CK(cudaEventRecord((cudaEvent_t)e, on_copy_stream ? copy_stream() : 0)); }
extern "C" void nn_stream_wait(int copy_stream_waits, void *e) { CK(cudaStreamWaitEvent(copy_stream_waits ? copy_stream() : 0, (cudaEvent_t)e, 0)); }
extern "C" void nn_event_sync(void *e) { CK(cudaEventSynchronize((cudaEvent_t)e)); }
/* ---- event profiler: GPU timestamps around ops, no host syncs ---- */
#define NPROF 8192
static cudaEvent_t g_ev[NPROF][2];
static int g_evk[NPROF], g_nev, g_ev_init;
extern "C" void nn_prof_begin(int k) {
    if (!g_ev_init) { for (int i = 0; i < NPROF; i++) { cudaEventCreate(&g_ev[i][0]); cudaEventCreate(&g_ev[i][1]); } g_ev_init = 1; }
    if (g_nev >= NPROF) return;
    g_evk[g_nev] = k; cudaEventRecord(g_ev[g_nev][0], 0);
}
extern "C" void nn_prof_end(void) { if (g_nev < NPROF) { cudaEventRecord(g_ev[g_nev][1], 0); g_nev++; } }
/* sums elapsed ms per category into out[nk], resets */
extern "C" void nn_prof_collect(double *out, int nk) {
    cudaDeviceSynchronize();
    for (int i = 0; i < nk; i++) out[i] = 0;
    for (int i = 0; i < g_nev; i++) { float ms = 0; cudaEventElapsedTime(&ms, g_ev[i][0], g_ev[i][1]); if (g_evk[i] < nk) out[g_evk[i]] += ms; }
    g_nev = 0;
}
extern "C" size_t nn_mem_free(void) { size_t f = 0, t = 0; cudaMemGetInfo(&f, &t); return f; }

#ifndef KSLAB
#define KSLAB 32
#endif
static int cur_dev(void);
static double *gn_dsums(size_t n);
__global__ void d2f_k(const double *d, float *f, int n);
__global__ void gn_finalize_k(const double *sums, int NG, size_t len, float eps, float *mean, float *rstd);
template <int SILU, typename TI, typename TO> __global__ void gn_apply_k(const TI *x, const float *gamma, const float *beta, const float *mean, const float *rstd, TO *y, int C, int G, size_t S);
static inline unsigned nblk(size_t n, unsigned b) { return (unsigned)((n + b - 1) / b); }

/* ================= conv3d forward =================
   Register-blocked direct convolution: a thread computes VX consecutive x-outputs for COT output channels
   (VX*COT accumulators); a block covers a TZ x TY x (TX*VX) output tile. Per input channel the input tile
   (with halo) and the COT x K^3 weights are staged in shared memory; each tap then costs VX + COT smem reads
   for VX*COT FMAs. */
#define COT 8
#define TZ 4
#define TY 8
#define TX 8

template <int K, int S, int VX>
__global__ void __launch_bounds__(256) conv_fwd_k(const float *__restrict__ x, const float *__restrict__ w, const float *__restrict__ b, float *__restrict__ y,
                           int N, int Ci, int D, int H, int W, int Co, int Do, int Ho, int Wo) {
    constexpr int OX = TX * VX;
    constexpr int IZ = TZ * S + K - 1, IY = TY * S + K - 1, IX = OX * S + K - 1;
    constexpr int PAD = K / 2;
    __shared__ float sx[IZ * IY * IX];
    __shared__ float sw[COT * K * K * K];
    const int tx = threadIdx.x % TX, ty = (threadIdx.x / TX) % TY, tz = threadIdx.x / (TX * TY);
    const int ox0 = blockIdx.x * OX, oy0 = blockIdx.y * TY;
    int bz = blockIdx.z;
    const int nzt = (Do + TZ - 1) / TZ;
    const int oz0 = (bz % nzt) * TZ; bz /= nzt;
    const int nct = (Co + COT - 1) / COT;
    const int co0 = (bz % nct) * COT; const int n = bz / nct;
    const int oy = oy0 + ty, oz = oz0 + tz;
    float acc[COT][VX];
#pragma unroll
    for (int c = 0; c < COT; c++)
#pragma unroll
        for (int v = 0; v < VX; v++) acc[c][v] = 0.f;
    const size_t plane = (size_t)D * H * W;
    for (int ci = 0; ci < Ci; ci++) {
        const float *xc = x + ((size_t)n * Ci + ci) * plane;
        for (int i = threadIdx.x; i < IZ * IY * IX; i += blockDim.x) {
            int ix = i % IX, iy = (i / IX) % IY, iz = i / (IX * IY);
            int gz = oz0 * S - PAD + iz, gy = oy0 * S - PAD + iy, gx = ox0 * S - PAD + ix;
            sx[i] = (gz >= 0 && gz < D && gy >= 0 && gy < H && gx >= 0 && gx < W) ? xc[((size_t)gz * H + gy) * W + gx] : 0.f;
        }
        for (int i = threadIdx.x; i < COT * K * K * K; i += blockDim.x) {
            int c = i / (K * K * K), t = i % (K * K * K);
            sw[i] = co0 + c < Co ? w[((size_t)(co0 + c) * Ci + ci) * (K * K * K) + t] : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int kz = 0; kz < K; kz++)
#pragma unroll
            for (int ky = 0; ky < K; ky++)
#pragma unroll
                for (int kx = 0; kx < K; kx++) {
                    const float *row = sx + ((tz * S + kz) * IY + (ty * S + ky)) * IX + (tx * VX) * S + kx;
                    float xv[VX];
#pragma unroll
                    for (int v = 0; v < VX; v++) xv[v] = row[v * S];
                    int t = (kz * K + ky) * K + kx;
#pragma unroll
                    for (int c = 0; c < COT; c++) {
                        float wv = sw[c * K * K * K + t];
#pragma unroll
                        for (int v = 0; v < VX; v++) acc[c][v] += xv[v] * wv;
                    }
                }
        __syncthreads();
    }
    if (oy < Ho && oz < Do)
#pragma unroll
        for (int c = 0; c < COT; c++) {
            int co = co0 + c;
            if (co >= Co) continue;
            float bias = b ? b[co] : 0.f;
            float *yp = y + (((size_t)n * Co + co) * Do + oz) * Ho * Wo + (size_t)oy * Wo;
#pragma unroll
            for (int v = 0; v < VX; v++) { int ox = ox0 + tx * VX + v; if (ox < Wo) yp[ox] = acc[c][v] + bias; }
        }
}

/* ================= tensor-core forward (k=3, stride 1, pad 1), BF16 inputs with fp32 accumulate =================
   Y[co, v] = sum_tap sum_ci W[tap][co][ci] * X[ci][v + off(tap)].  For a fixed tap and a 16-voxel row
   segment, the B operand (k = ci, n = voxels) is a strided view of the staged input tile (ldm = TC_T, the
   per-channel tile stride), so no im2col matrix is built. Block = 8 warps, output tile 2 z x 8 rows x 16 x
   (256 voxels); warp w owns z = w/4 and rows 2(w%4), 2(w%4)+1. Input slab = 16 channels (k = 16 per mma). */
#define TC_CI 16
#define TC_TZ 2
#define TC_TY 8
#define TC_T 720
#define NVT (TC_TZ * TC_TY * 16)
typedef __nv_bfloat16 bf16;
typedef __half f16;
/* 16-bit operand helpers, specialised for bf16 and fp16 */
template <typename HT> __device__ __forceinline__ HT f2h(float v);
template <> __device__ __forceinline__ bf16 f2h<bf16>(float v) { return __float2bfloat16(v); }
template <> __device__ __forceinline__ f16 f2h<f16>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ float f2h<float>(float v) { return v; }   /* identity: kernels templated on the output type */
template <typename HT> __device__ __forceinline__ float h2f(HT v);
template <> __device__ __forceinline__ float h2f<bf16>(bf16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float h2f<f16>(f16 v) { return __half2float(v); }
template <typename T> struct is_f16 { static const bool v = false; };
template <> struct is_f16<f16> { static const bool v = true; };
/* clamp to the fp16 range for a store that must not become inf; NaN stays NaN (fminf / fmaxf would turn it into
   +-65504 and hide a non-finite forward from the trainer's detection) */
__device__ __forceinline__ float sat_h16(float v) { return fabsf(v) > 65504.f ? copysignf(65504.f, v) : v; }

template <typename HT> __device__ __forceinline__ unsigned packh(float a, float b);
template <> __device__ __forceinline__ unsigned packh<bf16>(float a, float b) { __nv_bfloat162 r = __floats2bfloat162_rn(a, b); return *(unsigned *)&r; }
template <> __device__ __forceinline__ unsigned packh<f16>(float a, float b) { __half2 r = __floats2half2_rn(a, b); return *(unsigned *)&r; }
/* activation element type: float or bf16 (gradients are always float) */
template <typename T> __device__ __forceinline__ float ldv(const T *p, size_t i);
template <> __device__ __forceinline__ float ldv<float>(const float *p, size_t i) { return p[i]; }
template <> __device__ __forceinline__ float ldv<bf16>(const bf16 *p, size_t i) { return __bfloat162float(p[i]); }
template <> __device__ __forceinline__ float ldv<f16>(const f16 *p, size_t i) { return __half2float(p[i]); }
template <typename T> __device__ __forceinline__ void stv(T *p, size_t i, float v);
template <> __device__ __forceinline__ void stv<float>(float *p, size_t i, float v) { p[i] = v; }
template <> __device__ __forceinline__ void stv<bf16>(bf16 *p, size_t i, float v) { p[i] = __float2bfloat16(v); }
template <> __device__ __forceinline__ void stv<f16>(f16 *p, size_t i, float v) { p[i] = __float2half(v); }
#include "nn_lp.h"   /* gnp_t, split_t and the fp8/fp4 entry points */
/* channel split: input channels >= c_split come from x2 (channel ci - c_split); output channels >= o_split go to y2 */
/* tap set for the parity-decomposed stride-2 backward-data (S2B mode of the forward kernel): entries are
   (offset dz,dy,dx in {0,1} on the gy grid, weight tap index); outputs land at 2*o + parity in a gx of Dx x Hx x Wx */
typedef struct { int ntap; signed char dz[27], dy[27], dx[27], wt[27]; int pz, py, px, Dx, Hx, Wx; } tapset_t;   /* accum: y += conv instead of y = conv */
/* per-channel affine form of gn: v' = v * a + b (a = 0 / b = 0 markers unused; caller checks p.G) */
__device__ __forceinline__ void gn_coef(const gnp_t &p, int n, int ci, int C, float *a, float *b) {
    int cpg = C / p.G, ng = n * p.G + ci / cpg;
    float sc = p.rstd[ng] * p.gamma[ci];
    *a = sc; *b = p.beta[ci] - p.mean[ng] * sc;
}
__device__ __forceinline__ float silu_f(float v) { return v / (1.f + __expf(-v)); }
/* GroupNorm + SiLU coefficients of input channel ci: sp.gp2 for the x2 segment when set (then gp covers x alone), else gp over
   all channels; false when the channel is not transformed */
__device__ __forceinline__ bool in_gn(const gnp_t &gp, const split_t &sp, int n, int ci, int Ci, float *a, float *b) {
    if (sp.x2 && sp.gp2.G) {
        if (ci >= sp.c_split) { gn_coef(sp.gp2, n, ci - sp.c_split, Ci - sp.c_split, a, b); return true; }
        if (gp.G) { gn_coef(gp, n, ci, sp.c_split, a, b); return true; }
        return false;
    }
    if (gp.G) { gn_coef(gp, n, ci, Ci, a, b); return true; }
    return false;
}
/* 18-element staged row (fine columns ox0 - 1 .. ox0 + 16, ox0 even; D, H, W even) of the trilinear 2x upsample of
   a half-resolution channel xc: weights 3/4 and 1/4 per axis with the partner index clamped at the edges; zero outside
   [0, W) (conv padding). 4 coarse rows x 10 columns. */
template <typename T> __device__ __forceinline__ void ld4(const T *p, float *o);
template <typename TI> __device__ __forceinline__ void row18_up2(const TI *xc, int gz, int gyy, int ox0, int D, int H, int W, float *v) {
    const int Dh = D >> 1, Hh = H >> 1, Wh = W >> 1, c0 = (ox0 >> 1) - 1;
    const int mz = gz >> 1, my = gyy >> 1;
    const int mz1 = min(max(mz + ((gz & 1) ? 1 : -1), 0), Dh - 1), my1 = min(max(my + ((gyy & 1) ? 1 : -1), 0), Hh - 1);
    const TI *r00 = xc + ((size_t)mz * Hh + my) * Wh, *r01 = xc + ((size_t)mz * Hh + my1) * Wh;
    const TI *r10 = xc + ((size_t)mz1 * Hh + my) * Wh, *r11 = xc + ((size_t)mz1 * Hh + my1) * Wh;
    float t[10];
    const bool vfull = ox0 + 16 <= W && !(Wh & 3);   /* coarse columns c0 + 1 .. c0 + 8 in range, 8 / 16-byte aligned */
    const TI *rr[4] = {r00, r01, r10, r11};
    const float wr[4] = {0.5625f, 0.1875f, 0.1875f, 0.0625f};
#pragma unroll
    for (int j = 0; j < 10; j++) t[j] = 0.f;
#pragma unroll
    for (int q = 0; q < 4; q++) {
        float a[10];
        if (vfull) { ld4<TI>(rr[q] + c0 + 1, a + 1); ld4<TI>(rr[q] + c0 + 5, a + 5); }
        else {
#pragma unroll
            for (int j = 1; j < 9; j++) a[j] = ldv(rr[q], min(max(c0 + j, 0), Wh - 1));
        }
        a[0] = ldv(rr[q], max(c0, 0)); a[9] = ldv(rr[q], min(c0 + 9, Wh - 1));
#pragma unroll
        for (int j = 0; j < 10; j++) t[j] = fmaf(wr[q], a[j], t[j]);
    }
#pragma unroll
    for (int k = 0; k < 9; k++) {
        v[2 * k] = 0.75f * t[k] + 0.25f * t[k + 1];          /* fine column ox0 - 1 + 2k (odd) */
        v[2 * k + 1] = 0.75f * t[k + 1] + 0.25f * t[k];      /* fine column ox0 + 2k (even) */
    }
    if (ox0 == 0) v[0] = 0.f;
#pragma unroll
    for (int j = 1; j < 18; j++) if (ox0 - 1 + j >= W) v[j] = 0.f;
}
/* silu(gn(.)) of an 18-element staged row starting at input column x0, zero outside [0, W) */
__device__ __forceinline__ void row18_gn(float *v, float a, float b, int x0, int W) {
#pragma unroll
    for (int j = 0; j < 18; j++) v[j] = x0 + j >= 0 && x0 + j < W ? silu_f(v[j] * a + b) : 0.f;
}
static gnp_t to_gnp(const nn_gn_t *g) { gnp_t p = {}; if (g && g->G) { p.gamma = g->gamma; p.beta = g->beta; p.mean = g->mean; p.rstd = g->rstd; p.G = g->G; } return p; }
__device__ __forceinline__ float gn_silu_at(float v, const gnp_t &p, int n, int ci, int C) {
    if (!p.G) return v;
    int cpg = C / p.G, ng = n * p.G + ci / cpg;
    v = (v - p.mean[ng]) * p.rstd[ng] * p.gamma[ci] + p.beta[ci];
    return v / (1.f + expf(-v));
}
template <typename HT>
__global__ void prep_w_k(const float *w, HT *wp, int Co, int Ci, int Cop, int Cip) {   /* wp[tap][co][ci] */
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)27 * Cop * Cip) return;
    int ci = (int)(i % Cip), co = (int)((i / Cip) % Cop), t = (int)(i / ((size_t)Cip * Cop));
    wp[i] = f2h<HT>((co < Co && ci < Ci) ? w[((size_t)co * Ci + ci) * 27 + t] : 0.f);
}
/* Hand-rolled mma.sync m16n8k16 (bf16 x bf16 -> f32). Fragment layouts (PTX ISA): g = lane/4, t = lane%4;
   A (16x16 row): a0 = A[g][2t..2t+1], a1 = A[g+8][2t..], a2 = A[g][2t+8..], a3 = A[g+8][2t+8..];
   B (16x8 col):  b0 = B[2t..2t+1][g], b1 = B[2t+8..2t+9][g];
   C (16x8 f32):  c0 = C[g][2t], c1 = C[g][2t+1], c2 = C[g+8][2t], c3 = C[g+8][2t+1].
   The input tile is stored channel-contiguous (sx[pos][16 ci]) so every B register is one 32-bit load at an
   arbitrary voxel offset. M = output channels, N = voxels (8 per n-tile), K = 16 input channels. */
__device__ __forceinline__ unsigned smem_u32(const void *p) { return (unsigned)__cvta_generic_to_shared(p); }
/* four 8x8 b16 matrices; lane l supplies the row address of row l%8 of matrix l/8 */
__device__ __forceinline__ void ldmatrix_x4(unsigned *r, const void *row_ptr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(row_ptr)));
}
__device__ __forceinline__ void ldmatrix_x4_trans(unsigned *r, const void *row_ptr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(row_ptr)));
}
template <typename HT> __device__ __forceinline__ void mma16816(float *c, const unsigned *a, const unsigned *b);
template <> __device__ __forceinline__ void mma16816<bf16>(float *c, const unsigned *a, const unsigned *b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
template <> __device__ __forceinline__ void mma16816<f16>(float *c, const unsigned *a, const unsigned *b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
/* fp16 x fp16 -> fp16 accumulate (2x the 16-bit/fp32-accumulate rate on GeForce Blackwell); C/D: d0 = C[g][2t..2t+1], d1 = C[g+8][2t..2t+1] */
__device__ __forceinline__ void mma16816_h(unsigned *c, const unsigned *a, const unsigned *b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
/* prec 4 weights: fp16 wp[tap][co][ci] scaled per output channel, w * 2^-e with amax * 2^-e < 16; wsc[co] = 2^e. One block per padded co. */
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
/* in-place conversion of a staged 16-bit tile (8 elements per uint4) to fp16 scaled by inv: HT = bf16 or f16 */
template <typename HT> __device__ __forceinline__ void tile_to_f16(void *p, int n, float inv, int nthr) {
    for (int i = threadIdx.x * 8; i < n; i += nthr * 8) {
        uint4 u = *(uint4 *)((unsigned short *)p + i);
        unsigned *pu = (unsigned *)&u;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            float a, b;
            if constexpr (__is_same(HT, __half)) { float2 f = __half22float2(*(__half2 *)&pu[j]); a = f.x; b = f.y; }
            else { a = __uint_as_float(pu[j] << 16); b = __uint_as_float(pu[j] & 0xffff0000u); }
            __half2 hh = __floats2half2_rn(a * inv, b * inv);
            pu[j] = *(unsigned *)&hh;
        }
        *(uint4 *)((unsigned short *)p + i) = u;
    }
}
/* two consecutive elements at an even index of a row-aligned tensor (one 8- or 4-byte store) */
template <typename T> __device__ __forceinline__ void stv2(T *p, size_t i, float a, float b);
template <> __device__ __forceinline__ void stv2<float>(float *p, size_t i, float a, float b) { *(float2 *)(p + i) = make_float2(a, b); }
template <> __device__ __forceinline__ void stv2<bf16>(bf16 *p, size_t i, float a, float b) { *(__nv_bfloat162 *)(p + i) = __floats2bfloat162_rn(a, b); }
template <> __device__ __forceinline__ void stv2<f16>(f16 *p, size_t i, float a, float b) { *(__half2 *)(p + i) = __floats2half2_rn(a, b); }
template <typename T> __device__ __forceinline__ void ld4(const T *p, float *o);
template <> __device__ __forceinline__ void ld4<float>(const float *p, float *o) { float4 v = *(const float4 *)p; o[0] = v.x; o[1] = v.y; o[2] = v.z; o[3] = v.w; }
template <> __device__ __forceinline__ void ld4<f16>(const f16 *p, float *o) {
    uint2 v = *(const uint2 *)p; __half2 a = *(__half2 *)&v.x, b = *(__half2 *)&v.y; float2 fa = __half22float2(a), fb = __half22float2(b);
    o[0] = fa.x; o[1] = fa.y; o[2] = fb.x; o[3] = fb.y;
}
template <> __device__ __forceinline__ void ld4<bf16>(const bf16 *p, float *o) {
    uint2 v = *(const uint2 *)p;
    o[0] = __uint_as_float(v.x << 16); o[1] = __uint_as_float(v.x & 0xffff0000u);
    o[2] = __uint_as_float(v.y << 16); o[3] = __uint_as_float(v.y & 0xffff0000u);
}
__device__ __forceinline__ unsigned pack2(float a, float b) {
    __nv_bfloat162 r = __floats2bfloat162_rn(a, b);
    return *(unsigned *)&r;
}
template <typename T> __device__ __forceinline__ void ld8(const T *p, float *o);   /* 8 consecutive elements, 16-byte aligned for bf16 */
template <> __device__ __forceinline__ void ld8<float>(const float *p, float *o) { ld4<float>(p, o); ld4<float>(p + 4, o + 4); }
template <> __device__ __forceinline__ void ld8<f16>(const f16 *p, float *o) { ld4<f16>(p, o); ld4<f16>(p + 4, o + 4); }
template <> __device__ __forceinline__ void ld8<bf16>(const bf16 *p, float *o) {
    uint4 v = *(const uint4 *)p;
    o[0] = __uint_as_float(v.x << 16); o[1] = __uint_as_float(v.x & 0xffff0000u); o[2] = __uint_as_float(v.y << 16); o[3] = __uint_as_float(v.y & 0xffff0000u);
    o[4] = __uint_as_float(v.z << 16); o[5] = __uint_as_float(v.z & 0xffff0000u); o[6] = __uint_as_float(v.w << 16); o[7] = __uint_as_float(v.w & 0xffff0000u);
}
template <typename T> __device__ __forceinline__ void st4(T *p, const float *v);
template <> __device__ __forceinline__ void st4<float>(float *p, const float *v) { *(float4 *)p = make_float4(v[0], v[1], v[2], v[3]); }
template <> __device__ __forceinline__ void st4<bf16>(bf16 *p, const float *v) { *(uint2 *)p = make_uint2(pack2(v[0], v[1]), pack2(v[2], v[3])); }
template <> __device__ __forceinline__ void st4<f16>(f16 *p, const float *v) { *(uint2 *)p = make_uint2(packh<f16>(v[0], v[1]), packh<f16>(v[2], v[3])); }
/* slab bounds for per-(n,c) plane kernels: block y = slab of 4-aligned extent */
__device__ __forceinline__ void slab_range(size_t S, int slab, size_t *lo, size_t *hi) {
    size_t per = ((S + KSLAB - 1) / KSLAB + 3) & ~(size_t)3;
    *lo = (size_t)slab * per; *hi = *lo + per < S ? *lo + per : S;
}
/* forward tile: FW_TZ output planes x 8 rows x 16 columns per block, one warp per (plane, row pair) */
#define FW_TZ 2   /* 4 (512 threads) was tried: MT=4 spills, net slower */
#define FW_NT (FW_TZ * 128)
#define FW_T ((FW_TZ + 2) * 10 * 18)
#define FW_R 4   /* output rows per warp (MT = 4 keeps 2: accumulator registers) */
__host__ __device__ constexpr int fw_rows(int MT) { return MT == 4 ? 2 : FW_R; }
__host__ __device__ constexpr int fw_nth(int MT) { return 32 * FW_TZ * 8 / fw_rows(MT); }
__host__ __device__ constexpr int fw_blocks(int MT) { return fw_nth(MT) <= 128 ? 3 : (MT == 1 ? 3 : 2); }
/* OP = 1 (prec 4, HT = bf16 or f16 storage): fp16 operands with fp16 accumulation per tap group. The tile is staged in the storage
   type (exact) while tracking the block amax, then converted in place to fp16 scaled by 2^-ex (amax < 16); weights are
   fp16 scaled per output channel (wsc, amax < 16), so a 144-term group sum stays below 36864 < 65504. Each group's fp16
   sum is folded into the fp32 accumulators times 2^ex * wsc[co]. */
template <int MT, typename TI, typename TO, int S2B, typename HT, int OP = 0>
__global__ void __launch_bounds__(fw_nth(MT), fw_blocks(MT)) conv_fwd_tc_k(const TI *__restrict__ x, const HT *__restrict__ wp, const float *__restrict__ b, TO *__restrict__ y,
                              int N, int Ci, int D, int H, int W, int Co, int Cop, int Cip, gnp_t gp, double *__restrict__ osum, int Go, split_t sp, tapset_t ts,
                              const float *__restrict__ wsc = nullptr) {
    constexpr int BM = MT * 16, TG = 9, WDB = MT <= 2;  /* WDB: double-buffered weight groups */
    constexpr int R = fw_rows(MT), NTH = fw_nth(MT);    /* output rows per warp, threads per block */
    extern __shared__ __align__(32) unsigned char smem_raw[];
    HT *sx = (HT *)smem_raw;                       /* [FW_T pos][TC_CI ci] */
    HT *wa = sx + TC_CI * FW_T;                      /* [2 if WDB][TG tap][BM co][TC_CI ci] */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wz = warp / (8 / R), wr = (warp % (8 / R)) * R;
    const int ox0 = blockIdx.x * 16, oy0 = blockIdx.y * TC_TY;
    int bz = blockIdx.z;
    const int nzt = (D + FW_TZ - 1) / FW_TZ;
    const int oz0 = (bz % nzt) * FW_TZ; bz /= nzt;
    const int nmt = Cop / BM;
    const int co0 = (bz % nmt) * BM; const int n = bz / nmt;
    float acc[MT][R][2][4];     /* [m-tile][row][n-tile][c0..c3] */
#pragma unroll
    for (int m = 0; m < MT; m++) for (int r = 0; r < R; r++) for (int q = 0; q < 2; q++) for (int k = 0; k < 4; k++) acc[m][r][q][k] = 0.f;
    const size_t plane = (size_t)D * H * W;
    float wsr[MT][2];           /* OP: weight scale of this thread's rows g, g + 8 per m-tile */
    __shared__ unsigned s_amax;
    if (OP) {
#pragma unroll
        for (int m = 0; m < MT; m++) { wsr[m][0] = wsc[co0 + m * 16 + g]; wsr[m][1] = wsc[co0 + m * 16 + g + 8]; }
    }
    if (sp.accum) {   /* y += conv: start the accumulators from the existing output (no split output in this mode) */
#pragma unroll
        for (int m = 0; m < MT; m++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                int co = co0 + m * 16 + g + 8 * h;
#pragma unroll
                for (int r = 0; r < R; r++) {
                    int oz = oz0 + wz, oy = oy0 + wr + r;
                    if (oz >= D || oy >= H || co >= Co) continue;
                    const TO *yp = S2B ? y + (((size_t)n * Co + co) * ts.Dx + (2 * oz + ts.pz)) * ts.Hx * ts.Wx + (size_t)(2 * oy + ts.py) * ts.Wx + ts.px
                                       : y + (((size_t)n * Co + co) * D + oz) * H * W + (size_t)oy * W;
#pragma unroll
                    for (int q = 0; q < 2; q++) {
                        int ox = ox0 + q * 8 + 2 * t;
                        if (ox < W) acc[m][r][q][2 * h] = ldv(yp, (size_t)ox * (S2B ? 2 : 1));
                        if (ox + 1 < W) acc[m][r][q][2 * h + 1] = ldv(yp, (size_t)(ox + 1) * (S2B ? 2 : 1));
                    }
                }
            }
    }
    for (int ci0 = 0; ci0 < Cip; ci0 += TC_CI) {
        float amax = 0.f;
        if (OP && threadIdx.x == 0) s_amax = 0u;
        {   /* row-wise staging: lane k = ci, 16 lanes share a row; a thread converts a whole 18-element row (vector loads) */
            const int k = threadIdx.x & 15, ci = ci0 + k;
            const bool cok = ci < Ci;
            const bool xfull = ox0 + 16 <= W && !(W & 3);   /* vector path: interior in range and 8/16-byte row alignment */
            const bool upc = sp.up && !(sp.x2 && ci >= sp.c_split);   /* half-resolution x segment, upsampled while staging */
            const TI *xc = (sp.x2 && ci >= sp.c_split) ? (const TI *)sp.x2 + ((size_t)n * (Ci - sp.c_split) + (cok ? ci - sp.c_split : 0)) * plane
                                                       : x + ((size_t)n * (sp.x2 ? sp.c_split : Ci) + (cok ? ci : 0)) * (upc ? plane >> 3 : plane);
            float ga = 1.f, gb = 0.f;
            const bool gtr = cok && in_gn(gp, sp, n, ci, Ci, &ga, &gb);
            for (int rr = threadIdx.x >> 4; rr < (FW_TZ + 2) * 10; rr += NTH / 16) {   /* (FW_TZ + 2) z-planes x 10 rows */
                int iz = rr / 10, iy = rr - iz * 10;
                int gz = oz0 - 1 + iz, gyy = oy0 - 1 + iy;
                float v[18];
                HT *dst = sx + (size_t)rr * 18 * TC_CI + k;
                if (cok && upc && gz >= 0 && gz < D && gyy >= 0 && gyy < H) {
                    row18_up2(xc, gz, gyy, ox0, D, H, W, v);
#pragma unroll
                    for (int j = 0; j < 18; j++) { HT hv = f2h<HT>(v[j]); dst[j * TC_CI] = hv; if (OP) amax = fmaxf(amax, fabsf(h2f<HT>(hv))); }
                } else if (cok && gz >= 0 && gz < D && gyy >= 0 && gyy < H) {
                    const TI *xr = xc + ((size_t)gz * H + gyy) * W + ox0;
                    v[0] = ox0 > 0 ? ldv(xr - 1, 0) : 0.f;
                    if (xfull) {
#pragma unroll
                        for (int j = 0; j < 4; j++) ld4<TI>(xr + 4 * j, v + 1 + 4 * j);
                    } else {
#pragma unroll
                        for (int j = 0; j < 16; j++) v[1 + j] = ox0 + j < W ? ldv(xr, j) : 0.f;
                    }
                    v[17] = ox0 + 16 < W ? ldv(xr, 16) : 0.f;
                    if (gtr) {   /* transform, then restore the zero padding outside the volume */
#pragma unroll
                        for (int j = 0; j < 18; j++) v[j] = silu_f(v[j] * ga + gb);
                        if (ox0 == 0) v[0] = 0.f;
                        if (ox0 + 16 >= W) { v[17] = 0.f; if (!xfull) { for (int j = 0; j < 16; j++) if (ox0 + j >= W) v[1 + j] = 0.f; } }
                    }
#pragma unroll
                    for (int j = 0; j < 18; j++) { HT hv = f2h<HT>(v[j]); dst[j * TC_CI] = hv; if (OP) amax = fmaxf(amax, fabsf(h2f<HT>(hv))); }
                } else {
#pragma unroll
                    for (int j = 0; j < 18; j++) dst[j * TC_CI] = f2h<HT>(0.f);
                }
            }
        }
        if constexpr (OP != 0) {   /* block amax -> scale 2^-ex, in-place bf16 -> fp16 conversion of the tile (8 elements per step) */
            for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
            __syncthreads();
            if (lane == 0) atomicMax(&s_amax, __float_as_uint(amax));
            __syncthreads();
            float am = __uint_as_float(s_amax);
            int ex = 0;
            if (am > 0.f) frexpf(am / 16.f, &ex);
            tile_to_f16<HT>(sx, FW_T * TC_CI, ldexpf(1.f, -ex), NTH);
            amax = ldexpf(1.f, ex);                            /* reuse: tile scale 2^ex for the fold */
        }
        __syncthreads();
        const int NTAP = S2B ? ts.ntap : 27;
        /* weights of a tap group: [TG tap][BM co][16 ci] 16-bit, loaded 4 ci at a time; for MT <= 2 the two weight
           buffers alternate so group g+1 streams in while group g is being multiplied (one barrier per group) */
        auto load_group = [&](int t0, HT *dst) {
            for (int i = threadIdx.x * 4; i < TG * BM * TC_CI; i += NTH * 4) {
                int tt = i / (BM * TC_CI), r = i % (BM * TC_CI), c = r / TC_CI, k = r % TC_CI;
                if (S2B && t0 + tt >= NTAP) break;
                int wtap = S2B ? ts.wt[t0 + tt] : t0 + tt;
                *(uint2 *)(dst + i) = *(const uint2 *)(wp + ((size_t)wtap * Cop + co0 + c) * Cip + ci0 + k);
            }
        };
        if (WDB) load_group(0, wa);
        for (int t0 = 0; t0 < NTAP; t0 += TG) {
            HT *wg = WDB ? wa + ((t0 / TG) & 1) * (TG * BM * TC_CI) : wa;
            if (!WDB) load_group(t0, wa);
            __syncthreads();                                   /* this group's weights visible; previous group's buffer free */
            if (WDB && t0 + TG < NTAP) load_group(t0 + TG, wa + (((t0 / TG) + 1) & 1) * (TG * BM * TC_CI));
#ifndef TC_NOMMA
            unsigned hacc[OP ? MT : 1][OP ? R : 1][2][2];   /* fp16 group accumulators */
            if (OP) {
#pragma unroll
                for (int m = 0; m < MT; m++) for (int r = 0; r < R; r++) for (int q = 0; q < 2; q++) hacc[OP ? m : 0][OP ? r : 0][q][0] = hacc[OP ? m : 0][OP ? r : 0][q][1] = 0u;
            }
#pragma unroll
            for (int tt = 0; tt < TG; tt++) {
                int tap = t0 + tt;
                if (S2B && tap >= NTAP) break;
                int kz = S2B ? 1 + ts.dz[tap] : tap / 9, ky = S2B ? 1 + ts.dy[tap] : (tap / 3) % 3, kx = S2B ? 1 + ts.dx[tap] : tap % 3;
                /* B via ldmatrix.x4.trans: matrices = (n-tile q, k half); lane supplies row (voxel) l%8 of matrix l/8 */
                unsigned bf[R][4];   /* [row][q0:b0,b1, q1:b0,b1] */
#pragma unroll
                for (int r = 0; r < R; r++) {
                    int mat = lane >> 3, q = mat >> 1, kh = mat & 1;
                    int pos = ((wz + kz) * 10 + wr + r + ky) * 18 + kx + q * 8 + (lane & 7);
                    ldmatrix_x4(bf[r], sx + pos * TC_CI + kh * 8);
                }
#pragma unroll
                for (int m = 0; m < MT; m++) {
                    /* A via ldmatrix.x4: matrix (l/8): rows (m&1)*8 + l%8, cols (m>>1)*8 */
                    unsigned af[4];
                    int mat = lane >> 3;
                    ldmatrix_x4(af, wg + (tt * BM + m * 16 + (mat & 1) * 8 + (lane & 7)) * TC_CI + (mat >> 1) * 8);
#pragma unroll
                    for (int r = 0; r < R; r++) {
                        if (OP) { mma16816_h(hacc[OP ? m : 0][OP ? r : 0][0], af, bf[r]); mma16816_h(hacc[OP ? m : 0][OP ? r : 0][1], af, bf[r] + 2); }
                        else { mma16816<HT>(acc[m][r][0], af, bf[r]); mma16816<HT>(acc[m][r][1], af, bf[r] + 2); }
                    }
                }
            }
            if (OP) {   /* fold the group: acc += fp16 sum * 2^ex * wsc[row] */
#pragma unroll
                for (int m = 0; m < MT; m++) {
                    const float f0 = amax * wsr[m][0], f1 = amax * wsr[m][1];
#pragma unroll
                    for (int r = 0; r < R; r++)
#pragma unroll
                        for (int q = 0; q < 2; q++) {
                            float2 lo = __half22float2(*(__half2 *)&hacc[OP ? m : 0][OP ? r : 0][q][0]), hi = __half22float2(*(__half2 *)&hacc[OP ? m : 0][OP ? r : 0][q][1]);
                            acc[m][r][q][0] += lo.x * f0; acc[m][r][q][1] += lo.y * f0; acc[m][r][q][2] += hi.x * f1; acc[m][r][q][3] += hi.y * f1;
                        }
                }
            }
#endif
            if (!WDB) __syncthreads();
        }
        __syncthreads();                                       /* tile and weight buffers free for the next chunk */
    }
    /* epilogue straight from registers; optional GroupNorm statistics of the output (sum, sum of squares per
       (n, group)) reduced within the block in smem, one pair of double atomics per channel per block */
    float *cs = (float *)smem_raw;                     /* [2][BM] block partials */
    if (osum) { __syncthreads(); for (int i = threadIdx.x; i < 2 * BM; i += NTH) cs[i] = 0.f; __syncthreads(); }
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            int co = co0 + m * 16 + g + 8 * h;
            float ps = 0.f, pss = 0.f;
#pragma unroll
            for (int r = 0; r < R; r++) {
                int oz = oz0 + wz, oy = oy0 + wr + r;
                if (oz >= D || oy >= H || co >= Co) continue;
                float bias = b ? b[co] : 0.f;
                TO *yp = S2B ? y + (((size_t)n * Co + co) * ts.Dx + (2 * oz + ts.pz)) * ts.Hx * ts.Wx + (size_t)(2 * oy + ts.py) * ts.Wx + ts.px
                             : (sp.y2 && co >= sp.o_split) ? (TO *)sp.y2 + (((size_t)n * (Co - sp.o_split) + co - sp.o_split) * D + oz) * H * W + (size_t)oy * W
                                                           : y + (((size_t)n * (sp.y2 ? sp.o_split : Co) + co) * D + oz) * H * W + (size_t)oy * W;
#pragma unroll
                for (int q = 0; q < 2; q++) {
                    int ox = ox0 + q * 8 + 2 * t;
                    float v0 = acc[m][r][q][2 * h] + bias, v1 = acc[m][r][q][2 * h + 1] + bias;
                    if (osum && is_f16<TO>::v) {   /* activation feeding a GroupNorm, stored as fp16: saturate instead of inf (the statistics use the stored value) */
                        v0 = sat_h16(v0); v1 = sat_h16(v1);
                    }
                    if (!S2B && !(W & 1) && ox + 1 < W) { stv2(yp, (size_t)ox, v0, v1); ps += v0 + v1; pss += v0 * v0 + v1 * v1; continue; }   /* paired store */
                    if (ox < W) { stv(yp, (size_t)ox * (S2B ? 2 : 1), v0); ps += v0; pss += v0 * v0; }
                    if (ox + 1 < W) { stv(yp, (size_t)(ox + 1) * (S2B ? 2 : 1), v1); ps += v1; pss += v1 * v1; }
                }
            }
            if (osum) {
                ps += __shfl_xor_sync(0xffffffff, ps, 1); ps += __shfl_xor_sync(0xffffffff, ps, 2);
                pss += __shfl_xor_sync(0xffffffff, pss, 1); pss += __shfl_xor_sync(0xffffffff, pss, 2);
                if (t == 0) { atomicAdd(&cs[m * 16 + g + 8 * h], ps); atomicAdd(&cs[BM + m * 16 + g + 8 * h], pss); }
            }
        }
    if (osum) {
        __syncthreads();
        if (threadIdx.x < BM && co0 + (int)threadIdx.x < Co) {
            int cpg = Co / Go, ng = n * Go + (co0 + threadIdx.x) / cpg;
            atomicAdd(&osum[2 * ng], (double)cs[threadIdx.x]);
            atomicAdd(&osum[2 * ng + 1], (double)cs[BM + threadIdx.x]);
        }
    }
}

/* ---- fused parity-decomposed stride-2 backward-data for gradients of at most 16 channels: the gy tile (on the gy grid)
   is staged once and all 8 parity classes (1..8 taps each, 27 in total) run from it, each with its own weight group,
   accumulators and scattered epilogue (gx[2o + p] for its parity p). Replaces 8 launches that each restaged the tile.
   MT = 1 (16-row m-tiles: gx channels padded to 16 per m-tile, looped by blockIdx); OP = 1: fp16 operands as in
   conv_fwd_tc_k (block-amax tile scale, per-channel weight scale, one fp16 group per class). */
typedef struct { tapset_t c[8]; } s2cls_t;
template <typename T, int OP, typename HT>
__global__ void __launch_bounds__(128, 3) conv_bwd_s2_fused_k(const T *__restrict__ gy, const HT *__restrict__ wp, T *__restrict__ gx,
                                                          int N, int Ci, int D, int H, int W, int Co, int Cop, s2cls_t cls, int accum, const float *__restrict__ wsc) {
    constexpr int R = 4, NTH = 128;
    extern __shared__ __align__(32) unsigned char smem_raw[];
    HT *sx = (HT *)smem_raw;                       /* [FW_T pos][16 ci] */
    HT *wa = sx + TC_CI * FW_T;                      /* [8 tap][16 co][16 ci] */
    __shared__ unsigned s_amax;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wz = warp / 2, wr = (warp % 2) * R;
    const int ox0 = blockIdx.x * 16, oy0 = blockIdx.y * TC_TY;
    int bz = blockIdx.z;
    const int nzt = (D + FW_TZ - 1) / FW_TZ;
    const int oz0 = (bz % nzt) * FW_TZ; bz /= nzt;
    const int nmt = Cop / 16;
    const int co0 = (bz % nmt) * 16; const int n = bz / nmt;
    const size_t plane = (size_t)D * H * W;
    const int Dx = cls.c[0].Dx, Hx = cls.c[0].Hx, Wx = cls.c[0].Wx;
    float ws0 = 1.f, ws1 = 1.f;
    if (OP) { ws0 = wsc[co0 + g]; ws1 = wsc[co0 + g + 8]; if (threadIdx.x == 0) s_amax = 0u; }
    {   /* stage the gy tile (4 planes x 10 rows x 18 columns x 16 channels), zero outside the grid */
        const int k = threadIdx.x & 15;
        const bool cok = k < Ci, xfull = ox0 + 16 <= W && !(W & 3);
        const T *xc = gy + ((size_t)n * Ci + (cok ? k : 0)) * plane;
        float amax = 0.f;
        for (int rr = threadIdx.x >> 4; rr < (FW_TZ + 2) * 10; rr += NTH / 16) {
            int iz = rr / 10, iy = rr - iz * 10, gz = oz0 - 1 + iz, gyy = oy0 - 1 + iy;
            float v[18];
            HT *dst = sx + (size_t)rr * 18 * TC_CI + k;
            if (cok && gz >= 0 && gz < D && gyy >= 0 && gyy < H) {
                const T *xr = xc + ((size_t)gz * H + gyy) * W + ox0;
                v[0] = ox0 > 0 ? ldv(xr - 1, 0) : 0.f;
                if (xfull) {
#pragma unroll
                    for (int j = 0; j < 4; j++) ld4<T>(xr + 4 * j, v + 1 + 4 * j);
                } else {
#pragma unroll
                    for (int j = 0; j < 16; j++) v[1 + j] = ox0 + j < W ? ldv(xr, j) : 0.f;
                }
                v[17] = ox0 + 16 < W ? ldv(xr, 16) : 0.f;
#pragma unroll
                for (int j = 0; j < 18; j++) { HT hv = f2h<HT>(v[j]); dst[j * TC_CI] = hv; if (OP) amax = fmaxf(amax, fabsf(h2f<HT>(hv))); }
            } else {
#pragma unroll
                for (int j = 0; j < 18; j++) dst[j * TC_CI] = f2h<HT>(0.f);
            }
        }
        if (OP) {
            for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
            __syncthreads();
            if (lane == 0) atomicMax(&s_amax, __float_as_uint(amax));
        }
    }
    __syncthreads();
    float tsc = 1.f;   /* fp16 tile scale 2^ex */
    if (OP) {
        float am = __uint_as_float(s_amax);
        int ex = 0;
        if (am > 0.f) frexpf(am / 16.f, &ex);
        tsc = ldexpf(1.f, ex);
        tile_to_f16<HT>(sx, FW_T * TC_CI, ldexpf(1.f, -ex), NTH);
    }
#pragma unroll 1
    for (int c = 0; c < 8; c++) {
        const tapset_t &ts = cls.c[c];
        const int ntap = ts.ntap;
        __syncthreads();                                   /* previous class done with wa (and the conversion pass done) */
        for (int i = threadIdx.x * 4; i < ntap * 16 * TC_CI; i += NTH * 4) {
            int e = i / (16 * TC_CI), r = i % (16 * TC_CI), cc = r / TC_CI, kk = r % TC_CI;
            *(uint2 *)(wa + i) = *(const uint2 *)(wp + ((size_t)ts.wt[e] * Cop + co0 + cc) * TC_CI + kk);
        }
        __syncthreads();
        float acc[R][2][4];
#pragma unroll
        for (int r = 0; r < R; r++) for (int q = 0; q < 2; q++) for (int kk = 0; kk < 4; kk++) acc[r][q][kk] = 0.f;
        if (accum) {   /* prologue read of the existing gx (overlaps the MMAs; an epilogue read stalls) */
#pragma unroll
            for (int h = 0; h < 2; h++) {
                int co = co0 + g + 8 * h;
                if (co >= Co) continue;
#pragma unroll
                for (int r = 0; r < R; r++) {
                    int oz = oz0 + wz, oy = oy0 + wr + r;
                    if (oz >= D || oy >= H) continue;
                    const T *yp = gx + (((size_t)n * Co + co) * Dx + (2 * oz + ts.pz)) * Hx * Wx + (size_t)(2 * oy + ts.py) * Wx + ts.px;
#pragma unroll
                    for (int q = 0; q < 2; q++) {
                        int ox = ox0 + q * 8 + 2 * t;
                        if (ox < W) acc[r][q][2 * h] = ldv(yp, (size_t)ox * 2);
                        if (ox + 1 < W) acc[r][q][2 * h + 1] = ldv(yp, (size_t)(ox + 1) * 2);
                    }
                }
            }
        }
        unsigned hacc[OP ? R : 1][2][2];
        if (OP) {
#pragma unroll
            for (int r = 0; r < R; r++) for (int q = 0; q < 2; q++) hacc[OP ? r : 0][q][0] = hacc[OP ? r : 0][q][1] = 0u;
        }
        for (int e = 0; e < ntap; e++) {
            const int kz = 1 + ts.dz[e], ky = 1 + ts.dy[e], kx = 1 + ts.dx[e];
            unsigned bf[R][4];
#pragma unroll
            for (int r = 0; r < R; r++) {
                int mat = lane >> 3, q = mat >> 1, kh = mat & 1;
                int pos = ((wz + kz) * 10 + wr + r + ky) * 18 + kx + q * 8 + (lane & 7);
                ldmatrix_x4(bf[r], sx + pos * TC_CI + kh * 8);
            }
            unsigned af[4];
            {
                int mat = lane >> 3;
                ldmatrix_x4(af, wa + (e * 16 + (mat & 1) * 8 + (lane & 7)) * TC_CI + (mat >> 1) * 8);
            }
#pragma unroll
            for (int r = 0; r < R; r++) {
                if (OP) { mma16816_h(hacc[OP ? r : 0][0], af, bf[r]); mma16816_h(hacc[OP ? r : 0][1], af, bf[r] + 2); }
                else { mma16816<HT>(acc[r][0], af, bf[r]); mma16816<HT>(acc[r][1], af, bf[r] + 2); }
            }
        }
        if (OP) {
            const float f0 = tsc * ws0, f1 = tsc * ws1;
#pragma unroll
            for (int r = 0; r < R; r++)
#pragma unroll
                for (int q = 0; q < 2; q++) {
                    float2 lo = __half22float2(*(__half2 *)&hacc[OP ? r : 0][q][0]), hi = __half22float2(*(__half2 *)&hacc[OP ? r : 0][q][1]);
                    acc[r][q][0] += lo.x * f0; acc[r][q][1] += lo.y * f0; acc[r][q][2] += hi.x * f1; acc[r][q][3] += hi.y * f1;
                }
        }
        /* scattered epilogue: gx[n][co][2 oz + pz][2 oy + py][2 ox + px] (+= with accum) */
#pragma unroll
        for (int h = 0; h < 2; h++) {
            int co = co0 + g + 8 * h;
            if (co >= Co) continue;
#pragma unroll
            for (int r = 0; r < R; r++) {
                int oz = oz0 + wz, oy = oy0 + wr + r;
                if (oz >= D || oy >= H) continue;
                T *yp = gx + (((size_t)n * Co + co) * Dx + (2 * oz + ts.pz)) * Hx * Wx + (size_t)(2 * oy + ts.py) * Wx + ts.px;
#pragma unroll
                for (int q = 0; q < 2; q++) {
                    int ox = ox0 + q * 8 + 2 * t;
                    float v0 = acc[r][q][2 * h], v1 = acc[r][q][2 * h + 1];
                    if (ox < W) stv(yp, (size_t)ox * 2, v0);
                    if (ox + 1 < W) stv(yp, (size_t)(ox + 1) * 2, v1);
                }
            }
        }
    }
}

/* ---- stride-2 tensor-core forward: output tile 2 z x 4 rows x 8 x; input tile 6 x 10 x 18 per channel.
   Warp w owns z = w/4, row = w%4 and the single 8-voxel n-tile; B[k=ci][n] = x[ci][pos0 + 2n + kx]. */
#define S2_T 1080
template <int MT, typename TI, typename TO, typename HT>
__global__ void __launch_bounds__(256, 2) conv_fwd_tc_s2_k(const TI *__restrict__ x, const HT *__restrict__ wp, const float *__restrict__ b, TO *__restrict__ y,
                              int N, int Ci, int D, int H, int W, int Co, int Cop, int Cip, int Do, int Ho, int Wo, gnp_t gp = {}) {
    constexpr int BM = MT * 16, TG = 9;
    extern __shared__ __align__(32) unsigned char smem_raw[];
    HT *sx = (HT *)smem_raw;                       /* [S2_T pos][TC_CI ci] */
    HT *wa = sx + TC_CI * S2_T;                      /* [TG tap][BM co][TC_CI ci] */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wz = warp >> 2, wr = warp & 3;
    const int ox0 = blockIdx.x * 8, oy0 = blockIdx.y * 4;
    int bz = blockIdx.z;
    const int nzt = (Do + 1) / 2;
    const int oz0 = (bz % nzt) * 2; bz /= nzt;
    const int nmt = Cop / BM;
    const int co0 = (bz % nmt) * BM; const int n = bz / nmt;
    float acc[MT][4];
#pragma unroll
    for (int m = 0; m < MT; m++) for (int k = 0; k < 4; k++) acc[m][k] = 0.f;
    const size_t plane = (size_t)D * H * W;
    for (int ci0 = 0; ci0 < Cip; ci0 += TC_CI) {
        {   /* row-wise staging: lane k = ci, 16 lanes share a row of 18 input columns starting at 2*ox0 - 1 */
            const int k = threadIdx.x & 15, ci = ci0 + k;
            const bool cok = ci < Ci, xfull = 2 * ox0 + 16 <= W && !(W & 3);
            const TI *xc = x + ((size_t)n * Ci + (cok ? ci : 0)) * plane;
            float ga = 1.f, gb = 0.f;
            const bool gtr = cok && gp.G;
            if (gtr) gn_coef(gp, n, ci, Ci, &ga, &gb);
            for (int rr = threadIdx.x >> 4; rr < 60; rr += 16) {
                int iz = rr / 10, iy = rr - iz * 10;
                int gz = oz0 * 2 - 1 + iz, gyy = oy0 * 2 - 1 + iy;
                float v[18];
                HT *dst = sx + (size_t)rr * 18 * TC_CI + k;
                if (cok && gz >= 0 && gz < D && gyy >= 0 && gyy < H) {
                    const TI *xr = xc + ((size_t)gz * H + gyy) * W + 2 * ox0;
                    v[0] = ox0 > 0 ? ldv(xr - 1, 0) : 0.f;
                    if (xfull) {
#pragma unroll
                        for (int j = 0; j < 4; j++) ld4<TI>(xr + 4 * j, v + 1 + 4 * j);
                    } else {
#pragma unroll
                        for (int j = 0; j < 16; j++) v[1 + j] = 2 * ox0 + j < W ? ldv(xr, j) : 0.f;
                    }
                    v[17] = 2 * ox0 + 16 < W ? ldv(xr, 16) : 0.f;
                    if (gtr) row18_gn(v, ga, gb, 2 * ox0 - 1, W);
#pragma unroll
                    for (int j = 0; j < 18; j++) dst[j * TC_CI] = f2h<HT>(v[j]);
                } else {
#pragma unroll
                    for (int j = 0; j < 18; j++) dst[j * TC_CI] = f2h<HT>(0.f);
                }
            }
        }
        __syncthreads();
        for (int t0 = 0; t0 < 27; t0 += TG) {
            for (int i = threadIdx.x; i < TG * BM * TC_CI; i += 256) {
                int tt = i / (BM * TC_CI), r = i % (BM * TC_CI), c = r / TC_CI, k = r % TC_CI;
                wa[i] = wp[((size_t)(t0 + tt) * Cop + co0 + c) * Cip + ci0 + k];
            }
            __syncthreads();
#pragma unroll
            for (int tt = 0; tt < TG; tt++) {
                int tap = t0 + tt, kz = tap / 9, ky = (tap / 3) % 3, kx = tap % 3;
                int pos = ((wz * 2 + kz) * 10 + wr * 2 + ky) * 18 + kx + 2 * g;
                const unsigned *pb = (const unsigned *)(sx + pos * TC_CI + 2 * t);
                unsigned bf[2] = {pb[0], pb[4]};
#pragma unroll
                for (int m = 0; m < MT; m++) {
                    const unsigned *pa = (const unsigned *)(wa + (tt * BM + m * 16) * TC_CI);
                    unsigned af[4] = {pa[(g * TC_CI + 2 * t) / 2], pa[((g + 8) * TC_CI + 2 * t) / 2], pa[(g * TC_CI + 2 * t + 8) / 2], pa[((g + 8) * TC_CI + 2 * t + 8) / 2]};
                    mma16816<HT>(acc[m], af, bf);
                }
            }
            __syncthreads();
        }
    }
#pragma unroll
    for (int m = 0; m < MT; m++) {
        int oz = oz0 + wz, oy = oy0 + wr;
        if (oz >= Do || oy >= Ho) continue;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            int co = co0 + m * 16 + g + 8 * h;
            if (co >= Co) continue;
            float bias = b ? b[co] : 0.f;
            TO *yp = y + (((size_t)n * Co + co) * Do + oz) * Ho * Wo + (size_t)oy * Wo;
            int ox = ox0 + 2 * t;
            if (ox < Wo) stv(yp, (size_t)ox, acc[m][2 * h] + bias);
            if (ox + 1 < Wo) stv(yp, (size_t)ox + 1, acc[m][2 * h + 1] + bias);
        }
    }
}

static int cur_dev(void) { int d = 0; cudaGetDevice(&d); return d & 7; }
static void *tc_wbuf(size_t n) { static void *buf[8]; static size_t cap[8]; int d = cur_dev(); if (n > cap[d]) { if (buf[d]) cudaFree(buf[d]); cudaMalloc(&buf[d], n * 2); cap[d] = n; } return buf[d]; }

/* Tensor-core path for k=3, stride 1 (same spatial size). Returns 0 if handled. */
template <typename TI, typename TO, int S2B, typename HT, int OP = 0>
static void conv_fwd_tc_launch(int MT, dim3 grid, size_t smem, const TI *x, const HT *wp, const float *b, TO *y, shape5 xs, int cout, int Cop, int Cip, gnp_t gp, double *osum, int Go, split_t sp, tapset_t ts, const float *wsc = nullptr) {
    static int attr_set[8][5];
    if (!attr_set[cur_dev()][MT]) {
        cudaFuncSetAttribute(MT == 1 ? (const void *)conv_fwd_tc_k<1, TI, TO, S2B, HT, OP> : MT == 2 ? (const void *)conv_fwd_tc_k<2, TI, TO, S2B, HT, OP> : (const void *)conv_fwd_tc_k<4, TI, TO, S2B, HT, OP>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
        attr_set[cur_dev()][MT] = 1;
    }
    switch (MT) {
    case 1: conv_fwd_tc_k<1, TI, TO, S2B, HT, OP><<<grid, fw_nth(1), smem>>>(x, wp, b, y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, Cip, gp, osum, Go, sp, ts, wsc); break;
    case 2: conv_fwd_tc_k<2, TI, TO, S2B, HT, OP><<<grid, fw_nth(2), smem>>>(x, wp, b, y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, Cip, gp, osum, Go, sp, ts, wsc); break;
    default: conv_fwd_tc_k<4, TI, TO, S2B, HT, OP><<<grid, fw_nth(4), smem>>>(x, wp, b, y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, Cip, gp, osum, Go, sp, ts, wsc); break;
    }
}
static float *tc_wscbuf(size_t n) { static float *buf[8]; static size_t cap[8]; int d = cur_dev(); if (n > cap[d]) { if (buf[d]) cudaFree(buf[d]); cudaMalloc(&buf[d], n * sizeof(float)); cap[d] = n; } return buf[d]; }
/* prec 4 (16-bit storage HT): fp16 operands with fp16 group accumulation folded into fp32 (conv_fwd_tc_k OP = 1) */
template <typename HT>
static int conv_fwd_tc_f16acc(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, gnp_t gp, double *osum, int Go, split_t sp, const tapset_t *ts) {
    int Cop = (cout + 15) / 16 * 16, Cip = (xs.c + TC_CI - 1) / TC_CI * TC_CI;
    HT *wp = (HT *)tc_wbuf((size_t)27 * Cop * Cip);
    float *wsc = tc_wscbuf(Cop);
    prep_w16_k<<<Cop, 256>>>(w, (__half *)wp, wsc, cout, xs.c, Cop, Cip);
    int MT = Cop % 64 == 0 ? 4 : Cop % 32 == 0 ? 2 : 1;
    int nmt = Cop / (MT * 16);
    dim3 grid(nblk(xs.w, 16), nblk(xs.h, TC_TY), (unsigned)(nblk(xs.d, FW_TZ) * nmt * xs.n));
    size_t smem = (size_t)(TC_CI * FW_T + (MT <= 2 ? 2 : 1) * 9 * MT * 16 * TC_CI) * 2;
    tapset_t none = {};
    const tapset_t &tt = ts ? *ts : none;
    if (ts) {
        if (xbf && ybf) conv_fwd_tc_launch<HT, HT, 1, HT, 1>(MT, grid, smem, (const HT *)x, wp, b, (HT *)y, xs, cout, Cop, Cip, gp, osum, Go, sp, tt, wsc);
        else conv_fwd_tc_launch<float, float, 1, HT, 1>(MT, grid, smem, (const float *)x, wp, b, (float *)y, xs, cout, Cop, Cip, gp, osum, Go, sp, tt, wsc);
    }
    else if (xbf && ybf) conv_fwd_tc_launch<HT, HT, 0, HT, 1>(MT, grid, smem, (const HT *)x, wp, b, (HT *)y, xs, cout, Cop, Cip, gp, osum, Go, sp, tt, wsc);
    else if (xbf) conv_fwd_tc_launch<HT, float, 0, HT, 1>(MT, grid, smem, (const HT *)x, wp, b, (float *)y, xs, cout, Cop, Cip, gp, osum, Go, sp, tt, wsc);
    else if (ybf) conv_fwd_tc_launch<float, HT, 0, HT, 1>(MT, grid, smem, (const float *)x, wp, b, (HT *)y, xs, cout, Cop, Cip, gp, osum, Go, sp, tt, wsc);
    else conv_fwd_tc_launch<float, float, 0, HT, 1>(MT, grid, smem, (const float *)x, wp, b, (float *)y, xs, cout, Cop, Cip, gp, osum, Go, sp, tt, wsc);
    return 0;
}
template <typename HT>
static int conv_fwd_tc_h(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, gnp_t gp, double *osum, int Go, split_t sp, const tapset_t *ts) {
    int Cop = (cout + 15) / 16 * 16, Cip = (xs.c + TC_CI - 1) / TC_CI * TC_CI;
    size_t nw = (size_t)27 * Cop * Cip;
    HT *wp = (HT *)tc_wbuf(nw);
    prep_w_k<HT><<<nblk(nw, 256), 256>>>(w, wp, cout, xs.c, Cop, Cip);
    int MT = Cop % 64 == 0 ? 4 : Cop % 32 == 0 ? 2 : 1;   /* BM = 16 MT must divide Cop (e.g. Cop = 48 -> MT = 1) */
    int nmt = Cop / (MT * 16);
    dim3 grid(nblk(xs.w, 16), nblk(xs.h, TC_TY), (unsigned)(nblk(xs.d, FW_TZ) * nmt * xs.n));
    size_t smem = (size_t)(TC_CI * FW_T + (MT <= 2 ? 2 : 1) * 9 * MT * 16 * TC_CI) * 2;
    tapset_t none = {};
    if (ts) {
        if (xbf && ybf) conv_fwd_tc_launch<HT, HT, 1, HT>(MT, grid, smem, (const HT *)x, wp, b, (HT *)y, xs, cout, Cop, Cip, gp, osum, Go, sp, *ts);
        else conv_fwd_tc_launch<float, float, 1, HT>(MT, grid, smem, (const float *)x, wp, b, (float *)y, xs, cout, Cop, Cip, gp, osum, Go, sp, *ts);
    }
    else if (xbf && ybf) conv_fwd_tc_launch<HT, HT, 0, HT>(MT, grid, smem, (const HT *)x, wp, b, (HT *)y, xs, cout, Cop, Cip, gp, osum, Go, sp, none);
    else if (xbf) conv_fwd_tc_launch<HT, float, 0, HT>(MT, grid, smem, (const HT *)x, wp, b, (float *)y, xs, cout, Cop, Cip, gp, osum, Go, sp, none);
    else if (ybf) conv_fwd_tc_launch<float, HT, 0, HT>(MT, grid, smem, (const float *)x, wp, b, (HT *)y, xs, cout, Cop, Cip, gp, osum, Go, sp, none);
    else conv_fwd_tc_launch<float, float, 0, HT>(MT, grid, smem, (const float *)x, wp, b, (float *)y, xs, cout, Cop, Cip, gp, osum, Go, sp, none);
    return 0;
}
/* xbf / ybf: input / output tensors are 16-bit (bf16, or fp16 with nn_set_f16) instead of float.
   ts != nullptr: parity-decomposed stride-2 backward-data (x = gy on its own grid, output scattered into gx) */
static int conv_fwd_tc(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, gnp_t gp, double *osum, int Go, split_t sp, const tapset_t *ts = nullptr) {
    if (!ts && (ISMX(x) || ISMX(y))) {   /* MX-fp8 activation storage: fp8 compute, staged by copy */
        if (!ISMX(x) || !ISMX(y) || (sp.x2 && !ISMX(sp.x2)) || (sp.y2 && !ISMX(sp.y2))) { fprintf(stderr, "conv: MX-fp8 storage needs MX inputs and outputs\n"); abort(); }
        return lp_conv_fwd_f8(x, 3, xs, w, b, cout, y, 3, gp, osum, Go, sp);
    }
    const int pr = eff_prec();
    if (!ts && pr == 2 && !sp.up) return lp_conv_fwd_f8(x, LPDT(xbf), xs, w, b, cout, y, LPDT(ybf), gp, osum, Go, sp);   /* sp.up: 16-bit kernels only */
    if (!ts && pr == 3 && !sp.up) return lp_conv_fwd_f4(x, LPDT(xbf), xs, w, b, cout, y, LPDT(ybf), gp, osum, Go, sp);
    if (pr == 4) return g_h16 ? conv_fwd_tc_f16acc<f16>(x, xbf, xs, w, b, cout, y, ybf, gp, osum, Go, sp, ts) : conv_fwd_tc_f16acc<bf16>(x, xbf, xs, w, b, cout, y, ybf, gp, osum, Go, sp, ts);
    return g_h16 ? conv_fwd_tc_h<f16>(x, xbf, xs, w, b, cout, y, ybf, gp, osum, Go, sp, ts) : conv_fwd_tc_h<bf16>(x, xbf, xs, w, b, cout, y, ybf, gp, osum, Go, sp, ts);
}

template <typename TI, typename TO, typename HT>
static void conv_fwd_tc_s2_launch(int MT, dim3 grid, size_t smem, const TI *x, const HT *wp, const float *b, TO *y, shape5 xs, int cout, int Cop, int Cip, shape5 ys, gnp_t gp) {
    static int attr_set[8][5];
    if (!attr_set[cur_dev()][MT]) { cudaFuncSetAttribute(MT == 1 ? (const void *)conv_fwd_tc_s2_k<1, TI, TO, HT> : MT == 2 ? (const void *)conv_fwd_tc_s2_k<2, TI, TO, HT> : (const void *)conv_fwd_tc_s2_k<4, TI, TO, HT>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); attr_set[cur_dev()][MT] = 1; }
    switch (MT) {
    case 1: conv_fwd_tc_s2_k<1, TI, TO, HT><<<grid, 256, smem>>>(x, wp, b, y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, Cip, ys.d, ys.h, ys.w, gp); break;
    case 2: conv_fwd_tc_s2_k<2, TI, TO, HT><<<grid, 256, smem>>>(x, wp, b, y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, Cip, ys.d, ys.h, ys.w, gp); break;
    default: conv_fwd_tc_s2_k<4, TI, TO, HT><<<grid, 256, smem>>>(x, wp, b, y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, Cip, ys.d, ys.h, ys.w, gp); break;
    }
}
template <typename HT>
static int conv_fwd_tc_s2_h(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, shape5 ys, gnp_t gp) {
    int Cop = (cout + 15) / 16 * 16, Cip = (xs.c + TC_CI - 1) / TC_CI * TC_CI;
    size_t nw = (size_t)27 * Cop * Cip;
    HT *wp = (HT *)tc_wbuf(nw);
    prep_w_k<HT><<<nblk(nw, 256), 256>>>(w, wp, cout, xs.c, Cop, Cip);
    int MT = Cop % 64 == 0 ? 4 : Cop % 32 == 0 ? 2 : 1;
    int nmt = Cop / (MT * 16);
    dim3 grid(nblk(ys.w, 8), nblk(ys.h, 4), (unsigned)(nblk(ys.d, 2) * nmt * xs.n));
    size_t smem = (size_t)(TC_CI * S2_T + 9 * MT * 16 * TC_CI) * 2;
    if (xbf && ybf) conv_fwd_tc_s2_launch<HT, HT, HT>(MT, grid, smem, (const HT *)x, wp, b, (HT *)y, xs, cout, Cop, Cip, ys, gp);
    else if (xbf) conv_fwd_tc_s2_launch<HT, float, HT>(MT, grid, smem, (const HT *)x, wp, b, (float *)y, xs, cout, Cop, Cip, ys, gp);
    else if (ybf) conv_fwd_tc_s2_launch<float, HT, HT>(MT, grid, smem, (const float *)x, wp, b, (HT *)y, xs, cout, Cop, Cip, ys, gp);
    else conv_fwd_tc_s2_launch<float, float, HT>(MT, grid, smem, (const float *)x, wp, b, (float *)y, xs, cout, Cop, Cip, ys, gp);
    return 0;
}
static int conv_fwd_tc_s2(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, shape5 ys, gnp_t gp = {}) {
    if (ISMX(x) || ISMX(y)) { if (!ISMX(x) || !ISMX(y)) { fprintf(stderr, "conv s2: MX-fp8 storage needs MX input and output\n"); abort(); } return lp_conv_fwd_s2_f8(x, 3, xs, w, b, cout, y, 3, ys, gp); }
    if ((eff_prec() == 2 || eff_prec() == 3) && xbf == ybf) return lp_conv_fwd_s2_f8(x, LPDT(xbf), xs, w, b, cout, y, LPDT(ybf), ys, gp);
    return g_h16 ? conv_fwd_tc_s2_h<f16>(x, xbf, xs, w, b, cout, y, ybf, ys, gp) : conv_fwd_tc_s2_h<bf16>(x, xbf, xs, w, b, cout, y, ybf, ys, gp);
}

/* 1^3 conv reading activations of type TI (head): y[co] = b[co] + sum_ci w[co][ci] x[ci] */
template <typename TI, typename TO>
__global__ void conv1_f_k(const TI *__restrict__ x, const float *__restrict__ w, const float *__restrict__ b, TO *__restrict__ y, int N, int Ci, int Co, size_t S, gnp_t gp = {}) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * S) return;
    int n = (int)(i / S); size_t v = i % S;
    const TI *xp = x + (size_t)n * Ci * S + v;
    for (int co = 0; co < Co; co++) {
        float a = b ? b[co] : 0.f;
        for (int ci = 0; ci < Ci; ci++) a += w[co * Ci + ci] * gn_silu_at(ldv(xp, (size_t)ci * S), gp, n, ci, Ci);
        stv(y, ((size_t)n * Co + co) * S + v, a);
    }
}

extern "C" shape5 nn_conv3d_out_shape(shape5 xs, int cout, int k, int stride) {
    shape5 o = {xs.n, cout, (xs.d + 2 * (k / 2) - k) / stride + 1, (xs.h + 2 * (k / 2) - k) / stride + 1, (xs.w + 2 * (k / 2) - k) / stride + 1};
    return o;
}

extern "C" void nn_conv3d_fwd(const float *x, shape5 xs, const float *w, const float *b, int cout, int k, int stride, float *y) {
    if (k == 1 && g_tf32 && ISMX(x)) { lp_conv1_fwd_mx(x, xs, w, b, cout, y, gnp_t{}); KCHECK(); return; }   /* head reading an MX tensor (fp32 output) */
    shape5 ys = nn_conv3d_out_shape(xs, cout, k, stride);
    if (k == 3 && stride == 1 && g_tf32) { gnp_t none = {nullptr, nullptr, nullptr, nullptr, 0}; split_t ns = {nullptr, 0, nullptr, 0}; conv_fwd_tc(x, ABF, xs, w, b, cout, y, ABF, none, nullptr, 0, ns); KCHECK(); return; }
    if (k == 3 && stride == 2 && g_tf32) { conv_fwd_tc_s2(x, ABF, xs, w, b, cout, y, ABF, ys); KCHECK(); return; }
    if (k == 1 && stride == 1 && ABF) { size_t S = shape_spatial(xs); if (g_h16) conv1_f_k<f16, float><<<nblk((size_t)xs.n * S, 256), 256>>>((const f16 *)x, w, b, y, xs.n, xs.c, cout, S); else conv1_f_k<bf16, float><<<nblk((size_t)xs.n * S, 256), 256>>>((const bf16 *)x, w, b, y, xs.n, xs.c, cout, S); KCHECK(); return; }
    if (k == 3 && stride == 1) {
        dim3 grid(nblk(ys.w, TX * 4), nblk(ys.h, TY), nblk(ys.d, TZ) * nblk(cout, COT) * xs.n);
        conv_fwd_k<3, 1, 4><<<grid, 256>>>(x, w, b, y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, ys.d, ys.h, ys.w);
    } else if (k == 3 && stride == 2) {
        dim3 grid(nblk(ys.w, TX * 2), nblk(ys.h, TY), nblk(ys.d, TZ) * nblk(cout, COT) * xs.n);
        conv_fwd_k<3, 2, 2><<<grid, 256>>>(x, w, b, y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, ys.d, ys.h, ys.w);
    } else if (k == 1 && stride == 1) {
        dim3 grid(nblk(ys.w, TX * 4), nblk(ys.h, TY), nblk(ys.d, TZ) * nblk(cout, COT) * xs.n);
        conv_fwd_k<1, 1, 4><<<grid, 256>>>(x, w, b, y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, ys.d, ys.h, ys.w);
    }
    else { fprintf(stderr, "nn_conv3d_fwd: unsupported k=%d stride=%d\n", k, stride); abort(); }
    KCHECK();
}

/* ---- backward data ----
   stride 1: gx = conv(gy, w') with w'[ci][co][flipped tap]  (same fwd kernel).
   stride 2: gather kernel. */
__global__ void transpose_w_k(const float *w, float *wt, int Co, int Ci, int T) {   /* wt[ci][co][t] = w[co][ci][t] */
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)Co * Ci * T) return;
    int t = (int)(i % T), ci = (int)((i / T) % Ci), co = (int)(i / ((size_t)T * Ci));
    wt[((size_t)ci * Co + co) * T + t] = w[i];
}
__global__ void flip_w_k(const float *w, float *wt, int Co, int Ci, int T) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)Co * Ci * T) return;
    int t = (int)(i % T), ci = (int)((i / T) % Ci), co = (int)(i / ((size_t)T * Ci));
    wt[((size_t)ci * Co + co) * T + (T - 1 - t)] = w[i];
}

template <int K>
__global__ void conv_bwd_data_s2_k(const float *__restrict__ gy, const float *__restrict__ w, float *__restrict__ gx,
                                   int N, int Ci, int D, int H, int W, int Co, int Do, int Ho, int Wo) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t plane = (size_t)D * H * W;
    if (i >= (size_t)N * Ci * plane) return;
    int xx = (int)(i % W), yy = (int)((i / W) % H), zz = (int)((i / ((size_t)W * H)) % D);
    int ci = (int)((i / plane) % Ci), n = (int)(i / (plane * Ci));
    constexpr int PAD = K / 2;
    float acc = 0.f;
    for (int kz = 0; kz < K; kz++) {
        int oz2 = zz + PAD - kz; if (oz2 < 0 || oz2 & 1) continue; int oz = oz2 >> 1; if (oz >= Do) continue;
        for (int ky = 0; ky < K; ky++) {
            int oy2 = yy + PAD - ky; if (oy2 < 0 || oy2 & 1) continue; int oy = oy2 >> 1; if (oy >= Ho) continue;
            for (int kx = 0; kx < K; kx++) {
                int ox2 = xx + PAD - kx; if (ox2 < 0 || ox2 & 1) continue; int ox = ox2 >> 1; if (ox >= Wo) continue;
                int t = (kz * K + ky) * K + kx;
                for (int co = 0; co < Co; co++)
                    acc += gy[(((size_t)n * Co + co) * Do + oz) * Ho * Wo + (size_t)oy * Wo + ox] * w[((size_t)co * Ci + ci) * (K * K * K) + t];
            }
        }
    }
    gx[i] = acc;
}

__global__ void dilate2_k(const float *gy, float *gd, int N, int C, int D, int H, int W, int Do, int Ho, int Wo) {
    size_t Si = (size_t)D * H * W;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * C * Si) return;
    int x = (int)(i % W), y = (int)((i / W) % H), z = (int)((i / ((size_t)W * H)) % D), nc = (int)(i / Si);
    float v = 0.f;
    if (!(x & 1) && !(y & 1) && !(z & 1) && x / 2 < Wo && y / 2 < Ho && z / 2 < Do) v = gy[((size_t)nc * Do + z / 2) * Ho * Wo + (size_t)(y / 2) * Wo + x / 2];
    gd[i] = v;
}

/* scratch: flipped weights, plus (stride 2) the zero-inserted gradient at input resolution */
/* backward-data scratch: the flipped/transposed weights, plus (fp32 kernels only) the zero-inserted gradient for stride 2 */
extern "C" size_t nn_conv3d_scratch(shape5 xs, int cout, int k) {
    size_t w = (size_t)cout * xs.c * k * k * k * sizeof(float);
    return g_tf32 ? w + 256 : w + (size_t)xs.n * cout * shape_spatial(xs) * sizeof(float);
}

template <typename HT>
static void s2b_fused(const float *gy, shape5 ys, shape5 xs, float *gx, const float *wt, const s2cls_t &all, int accum, int op) {
    int Cop = (xs.c + 15) / 16 * 16, Cip = TC_CI;
    size_t nwp = (size_t)27 * Cop * Cip;
    HT *wp = (HT *)tc_wbuf(nwp);
    float *wsc = nullptr;
    if (op) { wsc = tc_wscbuf(Cop); prep_w16_k<<<Cop, 256>>>(wt, (__half *)wp, wsc, xs.c, ys.c, Cop, Cip); }
    else prep_w_k<HT><<<nblk(nwp, 256), 256>>>(wt, wp, xs.c, ys.c, Cop, Cip);
    dim3 grid(nblk(ys.w, 16), nblk(ys.h, TC_TY), (unsigned)(nblk(ys.d, FW_TZ) * (Cop / 16) * ys.n));
    size_t smem = (size_t)(TC_CI * FW_T + 8 * 16 * TC_CI) * 2;
    if (GBF) { if (op) conv_bwd_s2_fused_k<HT, 1, HT><<<grid, 128, smem>>>((const HT *)gy, wp, (HT *)gx, ys.n, ys.c, ys.d, ys.h, ys.w, xs.c, Cop, all, accum, wsc);
               else conv_bwd_s2_fused_k<HT, 0, HT><<<grid, 128, smem>>>((const HT *)gy, wp, (HT *)gx, ys.n, ys.c, ys.d, ys.h, ys.w, xs.c, Cop, all, accum, wsc); }
    else { if (op) conv_bwd_s2_fused_k<float, 1, HT><<<grid, 128, smem>>>(gy, wp, gx, ys.n, ys.c, ys.d, ys.h, ys.w, xs.c, Cop, all, accum, wsc);
           else conv_bwd_s2_fused_k<float, 0, HT><<<grid, 128, smem>>>(gy, wp, gx, ys.n, ys.c, ys.d, ys.h, ys.w, xs.c, Cop, all, accum, wsc); }
}
static void bwd_data_impl_(const float *gy, shape5 ys, const float *w, shape5 xs, int k, int stride, float *gx, float *scratch, int accum);
static void bwd_data_impl(const float *gy, shape5 ys, const float *w, shape5 xs, int k, int stride, float *gx, float *scratch, int accum) {
    int save = g_pass; g_pass = 1; bwd_data_impl_(gy, ys, w, xs, k, stride, gx, scratch, accum); g_pass = save;
}
static void bwd_data_impl_(const float *gy, shape5 ys, const float *w, shape5 xs, int k, int stride, float *gx, float *scratch, int accum) {
    int T = k * k * k;
    if (stride == 1) {
        size_t nw = (size_t)ys.c * xs.c * T;
        flip_w_k<<<nblk(nw, 256), 256>>>(w, scratch, ys.c, xs.c, T);
        /* gy has shape ys; treat it as input with ys.c channels, "cout" = xs.c. Spatial sizes equal for stride 1 / pad k/2. */
        if (k == 3 && g_tf32) { gnp_t none = {nullptr, nullptr, nullptr, nullptr, 0}; split_t ns = {nullptr, 0, nullptr, 0, accum}; conv_fwd_tc(gy, GBF, ys, scratch, nullptr, xs.c, gx, GBF, none, nullptr, 0, ns); }
        else if (k == 1 && g_tf32 && ISMX(gx)) lp_conv1_to_mx(gy, ISMX(gy) ? 3 : LPDT(GBF), xs.n, ys.c, shape_spatial(xs), scratch, xs.c, gx);   /* head: logit gradient -> MX gout */
        else if (k == 1 && GBF) { size_t S = shape_spatial(xs); if (g_h16) conv1_f_k<f16, f16><<<nblk((size_t)xs.n * S, 256), 256>>>((const f16 *)gy, scratch, nullptr, (f16 *)gx, xs.n, ys.c, xs.c, S); else conv1_f_k<bf16, bf16><<<nblk((size_t)xs.n * S, 256), 256>>>((const bf16 *)gy, scratch, nullptr, (bf16 *)gx, xs.n, ys.c, xs.c, S); }
        else { int save = g_actbf; g_actbf = 0; nn_conv3d_fwd(gy, ys, scratch, nullptr, xs.c, k, 1, gx); g_actbf = save; }
    } else if (stride == 2 && k == 3 && g_tf32 && (ISMX(gy) || ISMX(gx))) {   /* MX gradients: direct voxel-major kernel, accumulate in MX */
        if (!ISMX(gy) || !ISMX(gx)) { fprintf(stderr, "bwd_data s2: MX gradient storage needs MX gy and gx\n"); abort(); }
        lp_bwd_data_s2_mx(gy, ys, w, xs, gx, accum);
    } else if (stride == 2 && k == 3 && g_tf32 && !getenv("UFSM_S2DIL") && !(xs.d & 1) && !(xs.h & 1) && !(xs.w & 1)) {
        /* parity decomposition: gx[2m + p] = sum over the taps compatible with parity p of w . gy[m + d]; each of the
           8 parity classes is a stride-1 conv on the gy grid with 1..8 taps (27 total: no wasted MACs) */
        size_t nw = (size_t)ys.c * xs.c * T;
        transpose_w_k<<<nblk(nw, 256), 256>>>(w, scratch, ys.c, xs.c, T);
        gnp_t none = {nullptr, nullptr, nullptr, nullptr, 0};
        split_t ns = {nullptr, 0, nullptr, 0, accum};
        static int nofuse = -1;
        if (nofuse < 0) nofuse = getenv("UFSM_S2B_NOFUSE") != nullptr;
        const int fprec = eff_prec();
        const bool fused = !nofuse && ys.c <= TC_CI;   /* one launch for the 8 classes (gy of <= 16 channels) */
        s2cls_t all = {};
        for (int cls = 0; cls < 8; cls++) {
            tapset_t &ts = all.c[cls];
            ts.pz = cls >> 2; ts.py = (cls >> 1) & 1; ts.px = cls & 1; ts.Dx = xs.d; ts.Hx = xs.h; ts.Wx = xs.w;
            /* per axis: even output -> tap 1 at offset 0; odd -> tap 0 at offset +1 and tap 2 at offset 0 */
            int nz = ts.pz ? 2 : 1, ny = ts.py ? 2 : 1, nx = ts.px ? 2 : 1;
            for (int a = 0; a < nz; a++) for (int bb = 0; bb < ny; bb++) for (int c = 0; c < nx; c++) {
                int tz = ts.pz ? (a ? 2 : 0) : 1, ty = ts.py ? (bb ? 2 : 0) : 1, tx = ts.px ? (c ? 2 : 0) : 1;
                int e = ts.ntap++;
                ts.dz[e] = (signed char)(ts.pz ? (a ? 0 : 1) : 0); ts.dy[e] = (signed char)(ts.py ? (bb ? 0 : 1) : 0); ts.dx[e] = (signed char)(ts.px ? (c ? 0 : 1) : 0);
                ts.wt[e] = (signed char)(tz * 9 + ty * 3 + tx);
            }
            if (!fused) conv_fwd_tc(gy, GBF, ys, scratch, nullptr, xs.c, gx, GBF, none, nullptr, 0, ns, &ts);
        }
        if (fused) {   /* stage the gy tile once, run the 8 classes from it */
            if (g_h16) s2b_fused<f16>(gy, ys, xs, gx, scratch, all, accum, fprec == 4);
            else s2b_fused<bf16>(gy, ys, xs, gx, scratch, all, accum, fprec == 4);
        }
    } else if (stride == 2 && k == 3 && !(xs.d & 1) && !(xs.h & 1) && !(xs.w & 1)) {
        /* even input sizes: gx = conv_s1(dilate2(gy), flip(w)) */
        size_t nw = (size_t)ys.c * xs.c * T;
        float *gd = scratch + nw;
        flip_w_k<<<nblk(nw, 256), 256>>>(w, scratch, ys.c, xs.c, T);
        shape5 ds = xs; ds.c = ys.c;
        size_t n = shape_numel(ds);
        dilate2_k<<<nblk(n, 256), 256>>>(gy, gd, xs.n, ys.c, xs.d, xs.h, xs.w, ys.d, ys.h, ys.w);
        if (g_tf32) { gnp_t none = {nullptr, nullptr, nullptr, nullptr, 0}; split_t ns = {nullptr, 0, nullptr, 0, accum}; conv_fwd_tc(gd, 0, ds, scratch, nullptr, xs.c, gx, 0, none, nullptr, 0, ns); }
        else nn_conv3d_fwd(gd, ds, scratch, nullptr, xs.c, k, 1, gx);
    } else {
        size_t n = shape_numel(xs);
        if (k == 3) conv_bwd_data_s2_k<3><<<nblk(n, 256), 256>>>(gy, w, gx, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, ys.d, ys.h, ys.w);
        else { fprintf(stderr, "nn_conv3d_bwd_data: unsupported k=%d stride=%d\n", k, stride); abort(); }
    }
    KCHECK();
}
extern "C" void nn_conv3d_bwd_data(const float *gy, shape5 ys, const float *w, shape5 xs, int k, int stride, float *gx, float *scratch) { bwd_data_impl(gy, ys, w, xs, k, stride, gx, scratch, 0); }
/* gx += backward-data (tensor-core k=3 paths only; returns -1 otherwise, caller falls back to bwd_data + axpy) */
extern "C" int nn_conv3d_bwd_data_acc(const float *gy, shape5 ys, const float *w, shape5 xs, int k, int stride, float *gx, float *scratch) {
    if (!(g_tf32 && k == 3 && (stride == 1 || (stride == 2 && !(xs.d & 1) && !(xs.h & 1) && !(xs.w & 1))))) return -1;
    bwd_data_impl(gy, ys, w, xs, k, stride, gx, scratch, 1);
    return 0;
}

/* ---- backward weight ----
   Shared-memory tiled: a block owns one input channel ci, a group of 8 output channels and a 4x8x8 output
   tile. The x tile (with halo) and the gy tile are staged in smem; thread t handles the (4 cout x 3 tap)
   register tile (t % 18) over the voxel subset (t / 18); partials are reduced in smem and atomically added
   into gw. k = 1 convs use the simple kernel. */
#define WTZ 8
#define WTY 8
#define WTX 8
#define WCO 8
template <int S>
__global__ void conv_bwd_w3_k(const float *__restrict__ x, const float *__restrict__ gy, float *__restrict__ gw,
                              int N, int Ci, int D, int H, int W, int Co, int Do, int Ho, int Wo) {
    constexpr int K = 3, T = 27, PAD = 1;
    constexpr int IZ = WTZ * S + K - 1, IY = WTY * S + K - 1, IX = WTX * S + K - 1;
    constexpr int NV = WTZ * WTY * WTX;            /* 256 output voxels per tile */
    __shared__ float sx[IZ * IY * IX];
    __shared__ float sg[WCO * NV];
    __shared__ float red[WCO * T];
    const int ci = blockIdx.x;
    const int co0 = blockIdx.y * WCO;
    int bz = blockIdx.z;
    const int nxt = (Wo + WTX - 1) / WTX, nyt = (Ho + WTY - 1) / WTY, nzt = (Do + WTZ - 1) / WTZ;
    const int tx0 = (bz % nxt) * WTX; bz /= nxt;
    const int ty0 = (bz % nyt) * WTY; bz /= nyt;
    const int tz0 = (bz % nzt) * WTZ; const int n = bz / nzt;
    const size_t Si = (size_t)D * H * W, So = (size_t)Do * Ho * Wo;
    const float *xc = x + ((size_t)n * Ci + ci) * Si;
    for (int i = threadIdx.x; i < IZ * IY * IX; i += blockDim.x) {
        int ix = i % IX, iy = (i / IX) % IY, iz = i / (IX * IY);
        int gz = tz0 * S - PAD + iz, gyy = ty0 * S - PAD + iy, gx = tx0 * S - PAD + ix;
        sx[i] = (gz >= 0 && gz < D && gyy >= 0 && gyy < H && gx >= 0 && gx < W) ? xc[((size_t)gz * H + gyy) * W + gx] : 0.f;
    }
    for (int i = threadIdx.x; i < WCO * NV; i += blockDim.x) {
        int c = i / NV, v = i % NV, vx = v % WTX, vy = (v / WTX) % WTY, vz = v / (WTX * WTY);
        int oz = tz0 + vz, oy = ty0 + vy, ox = tx0 + vx, co = co0 + c;
        sg[i] = (co < Co && oz < Do && oy < Ho && ox < Wo) ? gy[((size_t)n * Co + co) * So + ((size_t)oz * Ho + oy) * Wo + ox] : 0.f;
    }
    for (int i = threadIdx.x; i < WCO * T; i += blockDim.x) red[i] = 0.f;
    __syncthreads();
    const int tile = threadIdx.x / 14, sub = threadIdx.x % 14;   /* 18 tiles x 14 voxel subsets; lanes of a warp share a tile and walk consecutive voxels */
    if (tile < 18) {
        const int cg = tile / 9, tg = tile % 9;                   /* cout group of 4, tap group of 3 */
        int toff[3];
#pragma unroll
        for (int b = 0; b < 3; b++) { int t = tg * 3 + b; toff[b] = ((t / 9) * IY + ((t / 3) % 3)) * IX + t % 3; }
        float acc[4][3] = {};
        const float *g0 = sg + (cg * 4) * NV;
#pragma unroll 4
        for (int v = sub; v < NV; v += 14) {
            int vx = v & (WTX - 1), vy = (v >> 3) & (WTY - 1), vz = v >> 6;
            const float *xb = sx + ((vz * S) * IY + (vy * S)) * IX + vx * S;
            float g[4];
#pragma unroll
            for (int a = 0; a < 4; a++) g[a] = g0[a * NV + v];
#pragma unroll
            for (int b = 0; b < 3; b++) {
                float xv = xb[toff[b]];
#pragma unroll
                for (int a = 0; a < 4; a++) acc[a][b] += xv * g[a];
            }
        }
#pragma unroll
        for (int a = 0; a < 4; a++)
#pragma unroll
            for (int b = 0; b < 3; b++) atomicAdd(&red[(cg * 4 + a) * T + tg * 3 + b], acc[a][b]);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < WCO * T; i += blockDim.x) {
        int c = i / T, t = i % T, co = co0 + c;
        if (co < Co) atomicAdd(&gw[((size_t)co * Ci + ci) * T + t], red[i]);
    }
}

/* ---- tensor-core weight gradient (k=3, stride 1): GW[co][ci][tap] = sum_v GY[co][v] X[ci][v + off(tap)] ----
   M = 16 output channels, N = 8 input channels (one slab), K = the 16 voxels of an output row.
   A = GY row tile [co][16 voxels] (bf16, voxel pairs contiguous); B = X shifted view: three copies of the input
   tile shifted by kx = 0,1,2 keep every voxel pair 32-bit aligned. Block = 8 warps over a 2 x 8 x 16 voxel
   tile; warp w owns taps {w, w+8, w+16, w+24}. Results are atomically added into gw. */
#define WG_SLAB 8
#define WG_ZC 6
/* ---- stride-1 weight gradient, v2: 9 warps, warp = (kz, ky) pair, its three kx taps share one row of loads.
   Tile layout [8 ci][4 z][10 y][20 x] (row stride padded to 20 so every fragment load is 8-byte aligned) with the
   k -> voxel map k = 2t+{0,1} -> x = 4t+{0,1}, k = 2t+8+{0,1} -> x = 4t+2+{0,1} (used consistently for A and B):
   a lane's whole A fragment is two 64-bit loads and its three B fragments one 64-bit + one 32-bit load.
   Staging is row-wise: one thread converts a full 18-element input row with 4-wide vector loads. */
#define W2_RS 20
#define W2_T 800
#define W2_NTHR 288
template <int NT, typename TI, typename TG, typename HT>
__global__ void __launch_bounds__(W2_NTHR, 3) conv_bwd_w_tc2_k(const TI *__restrict__ x, const TG *__restrict__ gy, float *__restrict__ gw, float *__restrict__ gb,
                                 int N, int Ci, int D, int H, int W, int Co, gnp_t gp, split_t sp) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    HT *sx = (HT *)smem_raw;                      /* [8 ci][W2_T] */
    HT *sg = sx + 8 * W2_T;                         /* [16 co][NVT vox]: vox = (z*8 + row)*16 + x */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int kz = warp / 3, ky = warp - 3 * kz;
    const int ci0 = blockIdx.x * 8 * NT, co0 = blockIdx.y * 16;
    int bz = blockIdx.z;
    const int nxt = (W + 15) / 16, nyt = (H + TC_TY - 1) / TC_TY, nzt = (D + TC_TZ - 1) / TC_TZ;
    const int ox0 = (bz % nxt) * 16; bz /= nxt;
    const int oy0 = (bz % nyt) * TC_TY; bz /= nyt;
    const int nzc = (nzt + WG_ZC - 1) / WG_ZC;
    const int zc = bz % nzc; const int n = bz / nzc;
    const size_t plane = (size_t)D * H * W;
    const bool xfull = ox0 + 16 <= W && !(W & 3);     /* the 16 interior columns are in range and rows are vector-aligned */
    float acc[3][NT][4] = {};
    float bsum = 0.f;
    for (int zt = zc * WG_ZC; zt < nzt && zt < (zc + 1) * WG_ZC; zt++) {
        const int oz0 = zt * TC_TZ;
        __syncthreads();
        /* gy tile: 16 co x 256 vox, 4 vox per thread-iteration */
        for (int i = threadIdx.x; i < 16 * NVT / 4; i += W2_NTHR) {
            int c = i / (NVT / 4), v4 = i - c * (NVT / 4), vx = (v4 & 3) * 4, vy = (v4 >> 2) & 7, vz = v4 >> 5;
            int co = co0 + c, oz = oz0 + vz, oy = oy0 + vy, ox = ox0 + vx;
            float f[4] = {0.f, 0.f, 0.f, 0.f};
            if (co < Co && oz < D && oy < H) {
                const TG *p = gy + (((size_t)n * Co + co) * D + oz) * H * W + (size_t)oy * W + ox;
                if (xfull) ld4<TG>(p, f);
                else { for (int j = 0; j < 4; j++) if (ox + j < W) f[j] = ldv(p, j); }
            }
            *(uint2 *)(sg + c * NVT + v4 * 4) = make_uint2(packh<HT>(f[0], f[1]), packh<HT>(f[2], f[3]));
        }
        __syncthreads();
        if (gb && blockIdx.x == 0 && threadIdx.x < 16) {
            const HT *row = sg + threadIdx.x * NVT;
            for (int v = 0; v < NVT; v++) bsum += h2f<HT>(row[v]);
        }
        for (int sl = 0; sl < NT; sl++) {
            __syncthreads();
            /* stage 8 channels x 40 rows (4 z-planes); one thread per row */
            for (int r = threadIdx.x; r < 8 * 40; r += W2_NTHR) {
                int k = r / 40, rr = r - k * 40, iz = rr / 10, iy = rr - iz * 10;
                int ci = ci0 + sl * 8 + k;
                int gz = oz0 - 1 + iz, gyy = oy0 - 1 + iy;
                float v[18];
                bool rok = ci < Ci && gz >= 0 && gz < D && gyy >= 0 && gyy < H;
                const bool upc = sp.up && !(sp.x2 && ci >= sp.c_split);
                if (rok && upc) row18_up2(x + ((size_t)n * sp.c_split + ci) * (plane >> 3), gz, gyy, ox0, D, H, W, v);
                else if (rok) {
                    const TI *xc = (sp.x2 && ci >= sp.c_split) ? (const TI *)sp.x2 + ((size_t)n * (Ci - sp.c_split) + ci - sp.c_split) * plane
                                                               : x + ((size_t)n * (sp.x2 ? sp.c_split : Ci) + ci) * plane;
                    xc += ((size_t)gz * H + gyy) * W + ox0;   /* column ox0 = element ix 1 */
                    v[0] = ox0 > 0 ? ldv(xc - 1, 0) : 0.f;
                    if (xfull) {
#pragma unroll
                        for (int j = 0; j < 4; j++) ld4<TI>(xc + 4 * j, v + 1 + 4 * j);
                    } else {
#pragma unroll
                        for (int j = 0; j < 16; j++) v[1 + j] = ox0 + j < W ? ldv(xc, j) : 0.f;
                    }
                    v[17] = ox0 + 16 < W ? ldv(xc, 16) : 0.f;
                    float ga, gb;
                    if (in_gn(gp, sp, n, ci, Ci, &ga, &gb)) {   /* transform, then restore the zero padding outside the volume */
#pragma unroll
                        for (int j = 0; j < 18; j++) v[j] = silu_f(v[j] * ga + gb);
                        if (ox0 == 0) v[0] = 0.f;
                        if (ox0 + 16 >= W) { v[17] = 0.f; if (!xfull) { for (int j = 0; j < 16; j++) if (ox0 + j >= W) v[1 + j] = 0.f; } }
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < 18; j++) v[j] = 0.f;
                }
                uint2 *dst = (uint2 *)(sx + k * W2_T + rr * W2_RS);
                dst[0] = make_uint2(packh<HT>(v[0], v[1]), packh<HT>(v[2], v[3]));
                dst[1] = make_uint2(packh<HT>(v[4], v[5]), packh<HT>(v[6], v[7]));
                dst[2] = make_uint2(packh<HT>(v[8], v[9]), packh<HT>(v[10], v[11]));
                dst[3] = make_uint2(packh<HT>(v[12], v[13]), packh<HT>(v[14], v[15]));
                *(unsigned *)(dst + 4) = packh<HT>(v[16], v[17]);
            }
            __syncthreads();
#pragma unroll 4
            for (int row = 0; row < TC_TZ * TC_TY; row++) {
                int vz = row >> 3, vy = row & 7;
                uint2 a01 = *(const uint2 *)(sg + g * NVT + row * 16 + 4 * t);
                uint2 a23 = *(const uint2 *)(sg + (g + 8) * NVT + row * 16 + 4 * t);
                unsigned af[4] = {a01.x, a23.x, a01.y, a23.y};
                const HT *pb = sx + g * W2_T + ((vz + kz) * 10 + vy + ky) * W2_RS + 4 * t;
                uint2 w01 = *(const uint2 *)pb;
                unsigned w2 = *(const unsigned *)(pb + 4);
                unsigned b0[2] = {w01.x, w01.y};
                unsigned b1[2] = {__funnelshift_r(w01.x, w01.y, 16), __funnelshift_r(w01.y, w2, 16)};
                unsigned b2[2] = {w01.y, w2};
                mma16816<HT>(acc[0][sl], af, b0);
                mma16816<HT>(acc[1][sl], af, b1);
                mma16816<HT>(acc[2][sl], af, b2);
            }
        }
    }
    if (gb && blockIdx.x == 0 && threadIdx.x < 16 && co0 + (int)threadIdx.x < Co) atomicAdd(&gb[co0 + threadIdx.x], bsum);
#pragma unroll
    for (int kx = 0; kx < 3; kx++) {
        int tap = (kz * 3 + ky) * 3 + kx;
#pragma unroll
        for (int q = 0; q < NT; q++) {
            int ci = ci0 + q * 8 + 2 * t;
#pragma unroll
            for (int h = 0; h < 2; h++) {
                int co = co0 + g + 8 * h;
                if (co >= Co) continue;
                if (ci < Ci) atomicAdd(&gw[((size_t)co * Ci + ci) * 27 + tap], acc[kx][q][2 * h]);
                if (ci + 1 < Ci) atomicAdd(&gw[((size_t)co * Ci + ci + 1) * 27 + tap], acc[kx][q][2 * h + 1]);
            }
        }
    }
}

template <typename TI, typename TG, typename HT>
static void launch_bwd_w_tc_t(const TI *x, shape5 xs, const TG *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp) {
    int nt8 = (xs.c + 7) / 8;
    int NT = nt8 >= 2 ? 2 : 1;
    int nzc = (nblk(ys.d, TC_TZ) + WG_ZC - 1) / WG_ZC;
    dim3 grid((xs.c + 8 * NT - 1) / (8 * NT), (ys.c + 15) / 16, (unsigned)(nblk(ys.w, 16) * nblk(ys.h, TC_TY) * nzc * ys.n));
    static int attr2[8];
    if (!attr2[cur_dev()]) { attr2[cur_dev()] = 1; cudaFuncSetAttribute((const void *)conv_bwd_w_tc2_k<1, TI, TG, HT>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); cudaFuncSetAttribute((const void *)conv_bwd_w_tc2_k<2, TI, TG, HT>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); }
    size_t smem2 = (size_t)(8 * W2_T + 16 * NVT) * 2;
    if (NT == 1) conv_bwd_w_tc2_k<1, TI, TG, HT><<<grid, W2_NTHR, smem2>>>(x, gy, gw, gb, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, gp, sp);
    else conv_bwd_w_tc2_k<2, TI, TG, HT><<<grid, W2_NTHR, smem2>>>(x, gy, gw, gb, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, gp, sp);
}
template <typename HT>
static void launch_bwd_w_tc_h(const void *x, int xbf, shape5 xs, const void *gy, int gybf, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp) {
    if (xbf && gybf) launch_bwd_w_tc_t<HT, HT, HT>((const HT *)x, xs, (const HT *)gy, ys, gw, gb, gp, sp);
    else if (xbf) launch_bwd_w_tc_t<HT, float, HT>((const HT *)x, xs, (const float *)gy, ys, gw, gb, gp, sp);
    else launch_bwd_w_tc_t<float, float, HT>((const float *)x, xs, (const float *)gy, ys, gw, gb, gp, sp);
}
static void launch_bwd_w_tc(const void *x, int xbf, shape5 xs, const void *gy, int gybf, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp) {
    if (ISMX(x)) { lp_bwd_w_f8(x, 3, xs, gy, ISMX(gy) ? 3 : LPDT(gybf), ys, gw, gb, gp, sp); return; }
    if ((eff_prec_w() == 2 || eff_prec_w() == 3) && !sp.up) { lp_bwd_w_f8(x, LPDT(xbf), xs, gy, LPDT(gybf), ys, gw, gb, gp, sp); return; }   /* prec 4 (fp16) keeps the 16-bit kernel for the weight gradient */
    if (g_h16) launch_bwd_w_tc_h<f16>(x, xbf, xs, gy, gybf, ys, gw, gb, gp, sp);
    else launch_bwd_w_tc_h<bf16>(x, xbf, xs, gy, gybf, ys, gw, gb, gp, sp);
}

/* ---- stride-2 tensor-core weight gradient: output tile 2 z x 4 rows x 8 x (per row k = 8 outputs, two rows per
   k-step of 16). The input tile (6 x 10 x 18 per channel) is stored de-interleaved by x parity so that the pair
   (x[2k + kx], x[2k + 2 + kx]) is one aligned 32-bit word: parity copy p = kx & 1 at half-position (2k + kx) / 2. */
#define S2W_CI 8
template <typename TI, typename TG, typename HT>
__global__ void __launch_bounds__(256, 3) conv_bwd_w_tc_s2_k(const TI *__restrict__ x, const TG *__restrict__ gy, float *__restrict__ gw, float *__restrict__ gb,
                                   int N, int Ci, int D, int H, int W, int Co, int Do, int Ho, int Wo, gnp_t gp = {}) {
    extern __shared__ __align__(32) unsigned char smem_raw[];
    HT *sx = (HT *)smem_raw;                      /* [2 parity][S2W_CI ci][60 rows][10 half-pos]: row = z*10 + y of the 6 x 10 tile */
    HT *sg = sx + 2 * S2W_CI * 600;                 /* [16 co][64 vox]: vox = (z*4 + row)*8 + x */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int ci0 = blockIdx.x * S2W_CI, co0 = blockIdx.y * 16;
    int bz = blockIdx.z;
    const int nxt = (Wo + 7) / 8, nyt = (Ho + 3) / 4, nzt = (Do + 1) / 2;
    const int ox0 = (bz % nxt) * 8; bz /= nxt;
    const int oy0 = (bz % nyt) * 4; bz /= nyt;
    const int nzc = (nzt + WG_ZC - 1) / WG_ZC;
    const int zc = bz % nzc; const int n = bz / nzc;
    const size_t plane = (size_t)D * H * W;
    float acc[4][4] = {};
    float bsum = 0.f;
    for (int zt = zc * WG_ZC; zt < nzt && zt < (zc + 1) * WG_ZC; zt++) {
        const int oz0 = zt * 2;
        __syncthreads();
        {   /* row-wise staging: one thread per (channel, row); the 18 columns are split by parity into two 9-element half rows */
            const bool xfull = 2 * ox0 + 16 <= W && !(W & 3);
            for (int r = threadIdx.x; r < S2W_CI * 60; r += 256) {
                int k = r / 60, rr = r - k * 60, iz = rr / 10, iy = rr - iz * 10;
                int ci = ci0 + k, gz = oz0 * 2 - 1 + iz, gyy = oy0 * 2 - 1 + iy;
                float v[18];
                if (ci < Ci && gz >= 0 && gz < D && gyy >= 0 && gyy < H) {
                    const TI *xr = x + ((size_t)n * Ci + ci) * plane + ((size_t)gz * H + gyy) * W + 2 * ox0;
                    v[0] = ox0 > 0 ? ldv(xr - 1, 0) : 0.f;
                    if (xfull) {
#pragma unroll
                        for (int j = 0; j < 4; j++) ld4<TI>(xr + 4 * j, v + 1 + 4 * j);
                    } else {
#pragma unroll
                        for (int j = 0; j < 16; j++) v[1 + j] = 2 * ox0 + j < W ? ldv(xr, j) : 0.f;
                    }
                    v[17] = 2 * ox0 + 16 < W ? ldv(xr, 16) : 0.f;
                    if (gp.G) { float ga, gb; gn_coef(gp, n, ci, Ci, &ga, &gb); row18_gn(v, ga, gb, 2 * ox0 - 1, W); }
                } else {
#pragma unroll
                    for (int j = 0; j < 18; j++) v[j] = 0.f;
                }
                HT *d0 = sx + (((size_t)0 * S2W_CI + k) * 60 + rr) * 10, *d1 = sx + (((size_t)1 * S2W_CI + k) * 60 + rr) * 10;   /* even / odd columns */
                unsigned *w0 = (unsigned *)d0, *w1 = (unsigned *)d1;   /* half rows are 20 bytes apart: 4-byte aligned only */
                w0[0] = packh<HT>(v[0], v[2]); w0[1] = packh<HT>(v[4], v[6]); w0[2] = packh<HT>(v[8], v[10]); w0[3] = packh<HT>(v[12], v[14]); d0[8] = f2h<HT>(v[16]);
                w1[0] = packh<HT>(v[1], v[3]); w1[1] = packh<HT>(v[5], v[7]); w1[2] = packh<HT>(v[9], v[11]); w1[3] = packh<HT>(v[13], v[15]); d1[8] = f2h<HT>(v[17]);
            }
        }
        for (int i = threadIdx.x; i < 16 * 64; i += 256) {
            int c = i / 64, v = i % 64, vx = v & 7, vy = (v >> 3) & 3, vz = v >> 5;
            int co = co0 + c, oz = oz0 + vz, oy = oy0 + vy, ox = ox0 + vx;
            sg[i] = f2h<HT>((co < Co && oz < Do && oy < Ho && ox < Wo) ? ldv(gy, (((size_t)n * Co + co) * Do + oz) * Ho * Wo + (size_t)oy * Wo + ox) : 0.f);
        }
        __syncthreads();
        if (gb && blockIdx.x == 0 && threadIdx.x < 16) { const HT *row = sg + threadIdx.x * 64; for (int v = 0; v < 64; v++) bsum += h2f<HT>(row[v]); }
        for (int rp = 0; rp < 4; rp++) {              /* k-step = two consecutive output rows (16 voxels): rows 2rp, 2rp+1 of the 8 (z*4 + y) */
            int r0 = 2 * rp, r1 = r0 + 1;
            const unsigned *pa = (const unsigned *)(sg + r0 * 8);   /* A[co][k]: k 0..7 = row r0, 8..15 = row r1 (contiguous) */
            unsigned af[4] = {pa[(g * 64 + 2 * t) / 2], pa[((g + 8) * 64 + 2 * t) / 2], pa[(g * 64 + 2 * t + 8) / 2], pa[((g + 8) * 64 + 2 * t + 8) / 2]};
#pragma unroll
            for (int j = 0; j < 4; j++) {
                int tap = warp + 8 * j;
                if (tap >= 27) break;
                int kz = tap / 9, ky = (tap / 3) % 3, kx = tap % 3, par = kx & 1, half = kx >> 1;
                /* B[k][n=ci g]: k = 2t,2t+1 (row r0, outputs 2t, 2t+1 -> x = 4t + kx, 4t + 2 + kx -> half-pos 2t + half, 2t + 1 + half) */
                int vz0 = r0 >> 2, vy0 = r0 & 3, vz1 = r1 >> 2, vy1 = r1 & 3;
                int row0 = (vz0 * 2 + kz) * 10 + vy0 * 2 + ky, row1 = (vz1 * 2 + kz) * 10 + vy1 * 2 + ky;
                const HT *b0 = sx + (((size_t)par * S2W_CI + g) * 60 + row0) * 10 + 2 * t + half;
                const HT *b1 = sx + (((size_t)par * S2W_CI + g) * 60 + row1) * 10 + 2 * t + half;
                unsigned bf[2];
                if (half) { const unsigned *q0 = (const unsigned *)(b0 - 1), *q1 = (const unsigned *)(b1 - 1); bf[0] = __funnelshift_r(q0[0], q0[1], 16); bf[1] = __funnelshift_r(q1[0], q1[1], 16); }
                else { bf[0] = *(const unsigned *)b0; bf[1] = *(const unsigned *)b1; }
                mma16816<HT>(acc[j], af, bf);
            }
        }
    }
    if (gb && blockIdx.x == 0 && threadIdx.x < 16 && co0 + (int)threadIdx.x < Co) atomicAdd(&gb[co0 + threadIdx.x], bsum);
#pragma unroll
    for (int j = 0; j < 4; j++) {
        int tap = warp + 8 * j;
        if (tap >= 27) break;
        int ci = ci0 + 2 * t;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            int co = co0 + g + 8 * h;
            if (co >= Co) continue;
            if (ci < Ci) atomicAdd(&gw[((size_t)co * Ci + ci) * 27 + tap], acc[j][2 * h]);
            if (ci + 1 < Ci) atomicAdd(&gw[((size_t)co * Ci + ci + 1) * 27 + tap], acc[j][2 * h + 1]);
        }
    }
}

/* k = 1: gw[co][ci] = sum gy[co] * x[ci]; block per (ci, co-group of 8), grid-stride over voxels */
/* 1^3 weight gradient: a block owns a voxel slab and accumulates all Ci x Co products (Ci x Co <= 64 handled here), so
   x and gy are read exactly once; block partials are added with atomics */
template <typename TI, typename TG, int CIO>
__global__ void __launch_bounds__(256) conv_bwd_w1_k(const TI *__restrict__ x, const TG *__restrict__ gy, float *__restrict__ gw, int N, int Ci, int Co, size_t S, gnp_t xg = {}) {
    float acc[CIO] = {};
    size_t per = ((S + gridDim.x - 1) / gridDim.x + 3) & ~(size_t)3, lo = (size_t)blockIdx.x * per, hi = lo + per < S ? lo + per : S;
    for (int n = 0; n < N; n++) {
        const TI *xp = x + (size_t)n * Ci * S; const TG *gp = gy + (size_t)n * Co * S;
        if (!(S & 3)) {
            for (size_t v = lo + 4 * threadIdx.x; v < hi; v += 4 * blockDim.x) {
                float gv[8][4];
                for (int co = 0; co < Co; co++) ld4<TG>(gp + (size_t)co * S + v, gv[co]);
                for (int ci = 0; ci < Ci; ci++) {
                    float xv[4]; ld4<TI>(xp + (size_t)ci * S + v, xv);
                    if (xg.G) { float a, b; gn_coef(xg, n, ci, Ci, &a, &b); for (int j = 0; j < 4; j++) xv[j] = silu_f(xv[j] * a + b); }
                    for (int co = 0; co < Co; co++) acc[ci * Co + co] += xv[0] * gv[co][0] + xv[1] * gv[co][1] + xv[2] * gv[co][2] + xv[3] * gv[co][3];
                }
            }
        } else {
            for (size_t v = lo + threadIdx.x; v < hi; v += blockDim.x) {
                float gv[8];
                for (int co = 0; co < Co; co++) gv[co] = ldv(gp, (size_t)co * S + v);
                for (int ci = 0; ci < Ci; ci++) {
                    float xv = gn_silu_at(ldv(xp, (size_t)ci * S + v), xg, n, ci, Ci);
                    for (int co = 0; co < Co; co++) acc[ci * Co + co] += xv * gv[co];
                }
            }
        }
    }
    __shared__ float red[CIO][33];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int k = 0; k < Ci * Co; k++) {
        float a = acc[k];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o);
        if (lane == 0) red[k][warp] = a;
    }
    __syncthreads();
    if (threadIdx.x < Ci * Co) { int ci = threadIdx.x / Co, co = threadIdx.x % Co; float a = 0.f; for (int w = 0; w < 8; w++) a += red[threadIdx.x][w]; atomicAdd(&gw[(size_t)co * Ci + ci], a); }   /* gw[co][ci] */
}
template <typename TG>
__global__ void bias_grad_k(const TG *gy, float *gb, int N, int Co, size_t So) {
    int co = blockIdx.x, slab = blockIdx.y;
    size_t per = (So + KSLAB - 1) / KSLAB, lo = slab * per, hi = lo + per < So ? lo + per : So;
    float s = 0.f;
    for (int n = 0; n < N; n++) {
        const TG *g = gy + ((size_t)n * Co + co) * So;
        for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x) s += ldv(g, i);
    }
    __shared__ float red[256];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) atomicAdd(&gb[co], red[0]);
}

template <typename HT>
static void bwd_w_s2_h(const float *x, shape5 xs, const float *gy, shape5 ys, float *gw, float *gb, gnp_t gp = {}) {
    static int attr[8];
    if (!attr[cur_dev()]) { cudaFuncSetAttribute((const void *)conv_bwd_w_tc_s2_k<float, float, HT>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); cudaFuncSetAttribute((const void *)conv_bwd_w_tc_s2_k<HT, float, HT>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); cudaFuncSetAttribute((const void *)conv_bwd_w_tc_s2_k<HT, HT, HT>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); attr[cur_dev()] = 1; }
    size_t smem = (size_t)(2 * S2W_CI * 600 + 16 * 64) * 2;
    int nzc = (nblk(ys.d, 2) + WG_ZC - 1) / WG_ZC;
    dim3 grid((xs.c + S2W_CI - 1) / S2W_CI, (ys.c + 15) / 16, (unsigned)(nblk(ys.w, 8) * nblk(ys.h, 4) * nzc * ys.n));
    if (GBF) conv_bwd_w_tc_s2_k<HT, HT, HT><<<grid, 256, smem>>>((const HT *)x, (const HT *)gy, gw, gb, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, ys.d, ys.h, ys.w, gp);
    else if (ABF) conv_bwd_w_tc_s2_k<HT, float, HT><<<grid, 256, smem>>>((const HT *)x, gy, gw, gb, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, ys.d, ys.h, ys.w, gp);
    else conv_bwd_w_tc_s2_k<float, float, HT><<<grid, 256, smem>>>(x, gy, gw, gb, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, ys.d, ys.h, ys.w, gp);
}
template <typename HT>
static void bwd_w1_h(const float *x, shape5 xs, const float *gy, shape5 ys, float *gw, size_t So, gnp_t xg = {}) {
    int nb = (int)((So + 2047) / 2048); if (nb > 4096) nb = 4096; if (nb < 1) nb = 1;
    if (GBF) conv_bwd_w1_k<HT, HT, 64><<<nb, 256>>>((const HT *)x, (const HT *)gy, gw, xs.n, xs.c, ys.c, So, xg);
    else if (ABF) conv_bwd_w1_k<HT, float, 64><<<nb, 256>>>((const HT *)x, gy, gw, xs.n, xs.c, ys.c, So, xg);
    else conv_bwd_w1_k<float, float, 64><<<nb, 256>>>(x, gy, gw, xs.n, xs.c, ys.c, So, xg);
}
extern "C" void nn_conv3d_bwd_weight(const float *x, shape5 xs, const float *gy, shape5 ys, int k, int stride, float *gw, float *gb) {
    size_t So = shape_spatial(ys);
    if (k == 3 && stride == 1 && g_tf32) {
        gnp_t none = {nullptr, nullptr, nullptr, nullptr, 0};
        split_t ns = {nullptr, 0, nullptr, 0};
        launch_bwd_w_tc(x, ABF, xs, gy, GBF, ys, gw, gb, none, ns);
        KCHECK();
        return;
    }
    if (k == 3 && stride == 2 && g_tf32 && ISMX(x)) { lp_bwd_w_s2_f8(x, 3, xs, gy, ISMX(gy) ? 3 : LPDT(GBF), ys, gw, gb, gnp_t{}); KCHECK(); return; }
    if (k == 1 && stride == 1 && g_tf32 && ISMX(x)) {
        lp_bwd_w1_mx(x, xs, gy, LPDT(GBF), ys, gw, gnp_t{});
        size_t So1 = shape_spatial(ys);
        if (gb) { if (GBF && g_h16) bias_grad_k<f16><<<dim3(ys.c, KSLAB), 256>>>((const f16 *)gy, gb, ys.n, ys.c, So1); else if (GBF) bias_grad_k<bf16><<<dim3(ys.c, KSLAB), 256>>>((const bf16 *)gy, gb, ys.n, ys.c, So1); else bias_grad_k<float><<<dim3(ys.c, KSLAB), 256>>>(gy, gb, ys.n, ys.c, So1); }
        KCHECK(); return;
    }
    if (k == 3 && stride == 2 && g_tf32 && (eff_prec_w() == 2 || eff_prec_w() == 3)) { lp_bwd_w_s2_f8(x, LPDT(ABF), xs, gy, LPDT(GBF), ys, gw, gb, gnp_t{}); KCHECK(); return; }
    if (k == 3 && stride == 2 && g_tf32) {
        if (g_h16) bwd_w_s2_h<f16>(x, xs, gy, ys, gw, gb); else bwd_w_s2_h<bf16>(x, xs, gy, ys, gw, gb);
        KCHECK();
        return;
    }
    if (k == 3) {
        int nxt = (ys.w + WTX - 1) / WTX, nyt = (ys.h + WTY - 1) / WTY, nzt = (ys.d + WTZ - 1) / WTZ;
        dim3 grid(xs.c, (ys.c + WCO - 1) / WCO, nxt * nyt * nzt * ys.n);
        if (stride == 1) conv_bwd_w3_k<1><<<grid, 256>>>(x, gy, gw, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, ys.d, ys.h, ys.w);
        else if (stride == 2) conv_bwd_w3_k<2><<<grid, 256>>>(x, gy, gw, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, ys.d, ys.h, ys.w);
        else { fprintf(stderr, "nn_conv3d_bwd_weight: unsupported stride %d\n", stride); abort(); }
    } else if (k == 1 && stride == 1 && xs.c * ys.c <= 64 && ys.c <= 8) {
        if (g_h16) bwd_w1_h<f16>(x, xs, gy, ys, gw, So); else bwd_w1_h<bf16>(x, xs, gy, ys, gw, So);
    } else { fprintf(stderr, "nn_conv3d_bwd_weight: unsupported k=%d stride=%d\n", k, stride); abort(); }
    if (gb) { if (GBF && g_h16) bias_grad_k<f16><<<dim3(ys.c, KSLAB), 256>>>((const f16 *)gy, gb, ys.n, ys.c, So); else if (GBF) bias_grad_k<bf16><<<dim3(ys.c, KSLAB), 256>>>((const bf16 *)gy, gb, ys.n, ys.c, So); else bias_grad_k<float><<<dim3(ys.c, KSLAB), 256>>>(gy, gb, ys.n, ys.c, So); }
    KCHECK();
}

__global__ void silu_f_k(const float *x, float *y, size_t n);

/* y = conv3d(silu(gn(x))) for k=3 stride 1 on the tensor-core path; returns -1 when that path is unavailable */
extern "C" int nn_conv3d_fwd_gn(const float *x, shape5 xs, int G, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                const float *w, const float *b, int cout, float *y) {
    if (G > xs.c) G = xs.c;   /* GroupNorm uses min(G, C) groups */
    if (!g_tf32) return -1;
    gnp_t gp = {gamma, beta, mean, rstd, G};
    split_t ns = {nullptr, 0, nullptr, 0};
    conv_fwd_tc(x, ABF, xs, w, b, cout, y, ABF, gp, nullptr, 0, ns);
    KCHECK();
    return 0;
}
/* Same conv (optionally with the gn+silu input transform when G_in > 0) that also produces the GroupNorm
   statistics of its OUTPUT for G_out groups: mean/rstd of y. Tensor-core path only; -1 when unavailable. */
extern "C" int nn_conv3d_fwd_gn_stats(const float *x, shape5 xs, int G_in, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                      const float *w, const float *b, int cout, float *y, int G_out, float eps, float *omean, float *orstd) {
    if (G_in > xs.c) G_in = xs.c; if (G_out > cout) G_out = cout;
    if (!g_tf32) return -1;
    gnp_t gp = {gamma, beta, mean, rstd, G_in};
    int NG = xs.n * G_out;
    double *sums = gn_dsums((size_t)2 * NG);
    cudaMemsetAsync(sums, 0, (size_t)2 * NG * sizeof(double));
    split_t ns = {nullptr, 0, nullptr, 0};
    conv_fwd_tc(x, ABF, xs, w, b, cout, y, ABF, gp, sums, G_out, ns);
    gn_finalize_k<<<nblk(NG, 128), 128>>>(sums, NG, (size_t)(cout / G_out) * shape_spatial(xs), eps, omean, orstd);
    KCHECK();
    return 0;
}
/* Forward with a channel-split input: channels [0, c_split) from x (xs.c = c_split + channels of x2), the rest
   from x2; otherwise like nn_conv3d_fwd_gn_stats (G_in applies gn+silu to BOTH inputs with the same params). */
extern "C" int nn_conv3d_fwd_split(const float *x, const float *x2, int c_split, shape5 xs, int G_in, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                   const float *w, const float *b, int cout, float *y, int G_out, float eps, float *omean, float *orstd) {
    if (G_in > xs.c) G_in = xs.c; if (G_out > cout) G_out = cout;
    if (!g_tf32) return -1;
    gnp_t gp = {gamma, beta, mean, rstd, G_in};
    split_t sp = {x2, c_split, nullptr, 0};
    double *sums = nullptr;
    if (G_out) { int NG = xs.n * G_out; sums = gn_dsums((size_t)2 * NG); cudaMemsetAsync(sums, 0, (size_t)2 * NG * sizeof(double)); }
    conv_fwd_tc(x, ABF, xs, w, b, cout, y, ABF, gp, sums, G_out, sp);
    if (G_out) gn_finalize_k<<<nblk(xs.n * G_out, 128), 128>>>(sums, xs.n * G_out, (size_t)(cout / G_out) * shape_spatial(xs), eps, omean, orstd);
    KCHECK();
    return 0;
}
/* Backward-data (k=3, stride 1) writing input-channel gradients [0, o_split) to gx and the rest to gx2. */
extern "C" int nn_conv3d_bwd_data_split(const float *gy, shape5 ys, const float *w, shape5 xs, float *gx, float *gx2, int o_split, float *scratch) {
    if (!g_tf32) return -1;
    int T = 27;
    size_t nw = (size_t)ys.c * xs.c * T;
    flip_w_k<<<nblk(nw, 256), 256>>>(w, scratch, ys.c, xs.c, T);
    gnp_t none = {nullptr, nullptr, nullptr, nullptr, 0};
    split_t sp = {nullptr, 0, gx2, o_split};
    int save = g_pass; g_pass = 1;
    conv_fwd_tc(gy, GBF, ys, scratch, nullptr, xs.c, gx, GBF, none, nullptr, 0, sp);
    g_pass = save;
    KCHECK();
    return 0;
}
/* Backward-data (k=3, stride 1) of input channels [c0, c0 + nc) only, into gx (nc channels). scratch as for
   nn_conv3d_bwd_data. Lets a caller produce a wide input gradient in channel chunks. */
extern "C" int nn_conv3d_bwd_data_range(const float *gy, shape5 ys, const float *w, shape5 xs, int c0, int nc, float *gx, float *scratch) {
    if (!g_tf32 || c0 < 0 || nc <= 0 || c0 + nc > xs.c) return -1;
    const int T = 27;
    size_t nw = (size_t)ys.c * xs.c * T;
    flip_w_k<<<nblk(nw, 256), 256>>>(w, scratch, ys.c, xs.c, T);   /* scratch[ci][co][t]: rows ci = output channels of this conv */
    gnp_t none = {}; split_t ns = {};
    shape5 gs = xs; gs.c = nc;
    int save = g_pass; g_pass = 1;
    conv_fwd_tc(gy, GBF, ys, scratch + (size_t)c0 * ys.c * T, nullptr, nc, gx, GBF, none, nullptr, 0, ns);
    g_pass = save;
    (void)gs;
    KCHECK();
    return 0;
}
/* y = silu(gn(x)) from precomputed statistics, one pass */
extern "C" void nn_gn_silu_apply(const float *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd, float *y) {
    if (G > s.c) G = s.c;
    if (ISMX(x)) { if (!ISMX(y)) { fprintf(stderr, "gn_silu_apply: MX input needs an MX output\n"); abort(); } lp_gn_silu_apply_mx(x, s, G, gamma, beta, mean, rstd, y); KCHECK(); return; }
    size_t n = shape_numel(s);
    if (ABF && g_h16) gn_apply_k<1, f16, f16><<<dim3(s.n * s.c, KSLAB), 256>>>((const f16 *)x, gamma, beta, mean, rstd, (f16 *)y, s.c, G, shape_spatial(s));
    else if (ABF) gn_apply_k<1, bf16, bf16><<<dim3(s.n * s.c, KSLAB), 256>>>((const bf16 *)x, gamma, beta, mean, rstd, (bf16 *)y, s.c, G, shape_spatial(s));
    else gn_apply_k<1, float, float><<<dim3(s.n * s.c, KSLAB), 256>>>(x, gamma, beta, mean, rstd, y, s.c, G, shape_spatial(s));
    KCHECK();
}
/* gw += dconv/dw with the conv input silu(gn(x)) recomputed at staging; -1 when unavailable */
extern "C" int nn_conv3d_bwd_weight_gn(const float *x, shape5 xs, int G, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                       const float *gy, shape5 ys, float *gw, float *gb) {
    if (G > xs.c) G = xs.c;
    if (!g_tf32) return -1;
    gnp_t gp = {gamma, beta, mean, rstd, G};
    split_t ns = {nullptr, 0, nullptr, 0};
    launch_bwd_w_tc(x, ABF, xs, gy, GBF, ys, gw, gb, gp, ns);
    KCHECK();
    return 0;
}
/* Weight gradient with a channel-split input (see nn_conv3d_fwd_split); G > 0 applies gn+silu to both inputs. */
extern "C" int nn_conv3d_bwd_weight_split(const float *x, const float *x2, int c_split, shape5 xs, int G, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                          const float *gy, shape5 ys, float *gw, float *gb) {
    if (G > xs.c) G = xs.c;
    if (!g_tf32) return -1;
    gnp_t gp = {gamma, beta, mean, rstd, G};
    split_t sp = {x2, c_split, nullptr, 0};
    launch_bwd_w_tc(x, ABF, xs, gy, GBF, ys, gw, gb, gp, sp);
    KCHECK();
    return 0;
}

/* ---- input-side recompute: the conv input silu(gn(.)) of a stored pre-norm tensor (and, for the decoder, the nearest
   upsample of the coarse block output) is formed while staging, so neither the normalized activation nor the
   upsampled tensor is stored ---- */
/* fused upsample: 16-bit tensor-core kernels only. An fp8 / fp4 policy on that conv is served by the 16-bit kernels
   (more precise; keeps the transient away); MX-stored inputs cannot be read by them -> -1 (caller uses the transient) */
static int up_kernel_ok(const float *x, const float *x2, int wgrad) { (void)wgrad; return !ISMX(x) && !ISMX(x2); }
static int xsplit(const float *x2, const nn_gn_t *gx, const nn_gn_t *gx2, int c_split, int up, shape5 xs, split_t *sp) {
    *sp = split_t{};
    if (up && (!x2 || (gx && gx->G) || ((xs.d | xs.h | xs.w) & 1))) return -1;   /* up: decoder split, x untransformed, even dims */
    if (!x2) return 0;
    sp->up = up;
    if ((gx && gx->G) && !(gx2 && gx2->G)) { fprintf(stderr, "conv: a split input with a GroupNorm on x needs one on x2\n"); return -1; }
    sp->x2 = x2; sp->c_split = c_split; sp->gp2 = to_gnp(gx2);
    return 0;
}
extern "C" int nn_conv3d_fwd_x(const float *x, const nn_gn_t *gx, const float *x2, const nn_gn_t *gx2, int c_split, int up, shape5 xs,
                               const float *w, const float *b, int cout, int k, int stride, float *y, int G_out, float eps, float *omean, float *orstd) {
    if (!g_tf32) return -1;
    gnp_t gp = to_gnp(gx);
    shape5 ys = nn_conv3d_out_shape(xs, cout, k, stride);
    if (k == 3 && stride == 1) {
        split_t sp; if (xsplit(x2, gx, gx2, c_split, up, xs, &sp)) return -1;
        if (up && !up_kernel_ok(x, x2, 0)) return -1;
        double *sums = nullptr;
        if (G_out) { int NG = xs.n * G_out; sums = gn_dsums((size_t)2 * NG); cudaMemsetAsync(sums, 0, (size_t)2 * NG * sizeof(double)); }
        conv_fwd_tc(x, ABF, xs, w, b, cout, y, ABF, gp, sums, G_out, sp);
        if (G_out) gn_finalize_k<<<nblk(xs.n * G_out, 128), 128>>>(sums, xs.n * G_out, (size_t)(cout / G_out) * shape_spatial(ys), eps, omean, orstd);
        KCHECK();
        return 0;
    }
    if (x2 || up || G_out) return -1;
    if (k == 3 && stride == 2) { conv_fwd_tc_s2(x, ABF, xs, w, b, cout, y, ABF, ys, gp); KCHECK(); return 0; }
    if (k == 1 && stride == 1) {
        if (ISMX(x)) lp_conv1_fwd_mx(x, xs, w, b, cout, y, gp);
        else { size_t S = shape_spatial(xs); dim3 gr(nblk((size_t)xs.n * S, 256));
            if (ABF && g_h16) conv1_f_k<f16, float><<<gr, 256>>>((const f16 *)x, w, b, y, xs.n, xs.c, cout, S, gp);
            else if (ABF) conv1_f_k<bf16, float><<<gr, 256>>>((const bf16 *)x, w, b, y, xs.n, xs.c, cout, S, gp);
            else conv1_f_k<float, float><<<gr, 256>>>(x, w, b, y, xs.n, xs.c, cout, S, gp); }
        KCHECK();
        return 0;
    }
    return -1;
}
extern "C" int nn_conv3d_bwd_weight_x(const float *x, const nn_gn_t *gx, const float *x2, const nn_gn_t *gx2, int c_split, int up, shape5 xs,
                                      const float *gy, shape5 ys, int k, int stride, float *gw, float *gb) {
    if (!g_tf32) return -1;
    gnp_t gp = to_gnp(gx);
    if (k == 3 && stride == 1) {
        split_t sp; if (xsplit(x2, gx, gx2, c_split, up, xs, &sp)) return -1;
        if (up && !up_kernel_ok(x, x2, 1)) return -1;
        launch_bwd_w_tc(x, ABF, xs, gy, GBF, ys, gw, gb, gp, sp);
        KCHECK();
        return 0;
    }
    if (x2 || up) return -1;
    if (k == 3 && stride == 2) {
        if (ISMX(x)) lp_bwd_w_s2_f8(x, 3, xs, gy, ISMX(gy) ? 3 : LPDT(GBF), ys, gw, gb, gp);
        else if (eff_prec_w() == 2 || eff_prec_w() == 3) lp_bwd_w_s2_f8(x, LPDT(ABF), xs, gy, LPDT(GBF), ys, gw, gb, gp);
        else if (g_h16) bwd_w_s2_h<f16>(x, xs, gy, ys, gw, gb, gp); else bwd_w_s2_h<bf16>(x, xs, gy, ys, gw, gb, gp);
        KCHECK();
        return 0;
    }
    if (k == 1 && stride == 1) {
        size_t So = shape_spatial(ys);
        if (ISMX(x)) lp_bwd_w1_mx(x, xs, gy, LPDT(GBF), ys, gw, gp);
        else if (xs.c * ys.c <= 64 && ys.c <= 8) { if (g_h16) bwd_w1_h<f16>(x, xs, gy, ys, gw, So, gp); else bwd_w1_h<bf16>(x, xs, gy, ys, gw, So, gp); }
        else return -1;
        if (gb) { if (GBF && g_h16) bias_grad_k<f16><<<dim3(ys.c, KSLAB), 256>>>((const f16 *)gy, gb, ys.n, ys.c, So); else if (GBF) bias_grad_k<bf16><<<dim3(ys.c, KSLAB), 256>>>((const bf16 *)gy, gb, ys.n, ys.c, So); else bias_grad_k<float><<<dim3(ys.c, KSLAB), 256>>>(gy, gb, ys.n, ys.c, So); }
        KCHECK();
        return 0;
    }
    return -1;
}

/* gx = gy * silu'(g) with g = gn(x) recomputed from the saved statistics (no materialized g) */
template <typename TI>
__global__ void silu_bwd_gn_k(const TI *x, const float *gamma, const float *beta, const float *mean, const float *rstd, const float *gy, float *gx, int N, int C, int G, size_t S) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * C * S) return;
    int c = (int)((i / S) % C), n = (int)(i / (S * C)), cpg = C / G, ng = n * G + c / cpg;
    float v = (ldv(x, i) - mean[ng]) * rstd[ng] * gamma[c] + beta[c];
    float s = 1.f / (1.f + expf(-v));
    gx[i] = gy[i] * (s * (1.f + v * (1.f - s)));
}
extern "C" void nn_silu_bwd_gn(const float *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd, const float *gy, float *gx) {
    if (G > s.c) G = s.c;
    size_t n = shape_numel(s);
    if (ABF && g_h16) silu_bwd_gn_k<f16><<<nblk(n, 256), 256>>>((const f16 *)x, gamma, beta, mean, rstd, gy, gx, s.n, s.c, G, shape_spatial(s));
    else if (ABF) silu_bwd_gn_k<bf16><<<nblk(n, 256), 256>>>((const bf16 *)x, gamma, beta, mean, rstd, gy, gx, s.n, s.c, G, shape_spatial(s));
    else silu_bwd_gn_k<float><<<nblk(n, 256), 256>>>(x, gamma, beta, mean, rstd, gy, gx, s.n, s.c, G, shape_spatial(s));
    KCHECK();
}


/* ---- fused backward through silu(gn(x)): given gy = dL/d silu, produce gx = dL/dx and the gamma/beta grads.
   a = gy * silu'(gn(x)) is recomputed in both passes instead of being stored. ---- */
template <typename TI, typename TG>
__global__ void gn_silu_bwd_stats_k(const TI *x, const TG *gy, const float *gamma, const float *beta, const float *mean, const float *rstd, int C, int G, size_t S, double *ds) {
    int nc = blockIdx.x, slab = blockIdx.y, n = nc / C, c = nc % C, cpg = C / G, ng = n * G + c / cpg;
    const TI *xp = x + (size_t)nc * S; const TG *gp = gy + (size_t)nc * S;
    float m = mean[ng], r = rstd[ng], ga = gamma[c], be = beta[c];
    size_t lo, hi; slab_range(S, slab, &lo, &hi);
    float s1 = 0.f, s2 = 0.f;   /* per-thread partials in fp32 (a few hundred terms), block reduction in fp64 */
    if (!(S & 3)) {
        for (size_t i = lo + 4 * threadIdx.x; i < hi; i += 4 * blockDim.x) {
            float xv[4], gv[4]; ld4<TI>(xp + i, xv); ld4<TG>(gp + i, gv);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float xhat = (xv[j] - m) * r, v = xhat * ga + be, sg = 1.f / (1.f + __expf(-v));
                float a = gv[j] * (sg * (1.f + v * (1.f - sg)));
                s1 += a; s2 += a * xhat;
            }
        }
    } else {
        for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
            float xhat = (ldv(xp, i) - m) * r, v = xhat * ga + be, sg = 1.f / (1.f + __expf(-v));
            float a = ldv(gp, i) * (sg * (1.f + v * (1.f - sg)));
            s1 += a; s2 += a * xhat;
        }
    }
    __shared__ double r1[256], r2[256];
    r1[threadIdx.x] = s1; r2[threadIdx.x] = s2;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) { r1[threadIdx.x] += r1[threadIdx.x + o]; r2[threadIdx.x] += r2[threadIdx.x + o]; } __syncthreads(); }
    if (threadIdx.x == 0) { atomicAdd(&ds[2 * nc], r1[0]); atomicAdd(&ds[2 * nc + 1], r2[0]); }
}
__global__ void gn_group_sums_k(const float *st, const float *gamma, int N, int C, int G, float *AB) {   /* AB[2*ng] = sum st*gamma, AB[2*ng+1] = sum st2*gamma */
    int ng = blockIdx.x * blockDim.x + threadIdx.x;
    if (ng >= N * G) return;
    int n = ng / G, g = ng % G, cpg = C / G;
    float A = 0.f, B = 0.f;
    for (int cc = g * cpg; cc < (g + 1) * cpg; cc++) { A += st[2 * (n * C + cc)] * gamma[cc]; B += st[2 * (n * C + cc) + 1] * gamma[cc]; }
    AB[2 * ng] = A; AB[2 * ng + 1] = B;
}
template <typename TI, typename TG, typename TO>
__global__ void gn_silu_bwd_apply_k(const TI *x, const TG *gy, const float *gamma, const float *beta, const float *mean, const float *rstd, const float *AB,
                                    TO *gx, int C, int G, size_t S) {
    int nc = blockIdx.x, slab = blockIdx.y, n = nc / C, c = nc % C, cpg = C / G, ng = n * G + c / cpg;
    float A = AB[2 * ng], B = AB[2 * ng + 1];
    float len = (float)cpg * (float)S, m = mean[ng], r = rstd[ng], ga = gamma[c], be = beta[c];
    float ka = A / len, kb = B / len;
    const TI *xp = x + (size_t)nc * S; const TG *gp = gy + (size_t)nc * S; TO *op = gx + (size_t)nc * S;
    size_t lo, hi; slab_range(S, slab, &lo, &hi);
    if (!(S & 3)) {
        for (size_t i = lo + 4 * threadIdx.x; i < hi; i += 4 * blockDim.x) {
            float xv[4], gv[4], o[4]; ld4<TI>(xp + i, xv); ld4<TG>(gp + i, gv);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float xhat = (xv[j] - m) * r, v = xhat * ga + be, sg = 1.f / (1.f + __expf(-v));
                float a = gv[j] * (sg * (1.f + v * (1.f - sg)));
                o[j] = r * (a * ga - ka - xhat * kb);
            }
            st4<TO>(op + i, o);
        }
    } else {
        for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
            float xhat = (ldv(xp, i) - m) * r, v = xhat * ga + be, sg = 1.f / (1.f + __expf(-v));
            float a = ldv(gp, i) * (sg * (1.f + v * (1.f - sg)));
            stv(op, i, r * (a * ga - ka - xhat * kb));
        }
    }
}
__global__ void gn_param_grad_k2(const float *st, float *ggamma, float *gbeta, int N, int C);
template <typename TI, typename TG, typename TO>
static void gn_silu_bwd_t(const TI *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd, const TG *gy,
                          TO *gx, float *ggamma, float *gbeta, float *scratch) {
    size_t S = shape_spatial(s);
    int NC = s.n * s.c;
    double *ds = gn_dsums((size_t)2 * NC);
    cudaMemsetAsync(ds, 0, (size_t)2 * NC * sizeof(double));
    gn_silu_bwd_stats_k<<<dim3(NC, KSLAB), 256>>>(x, gy, gamma, beta, mean, rstd, s.c, G, S, ds);
    d2f_k<<<nblk(2 * NC, 128), 128>>>(ds, scratch, 2 * NC);
    float *AB = scratch + 2 * NC;   /* scratch holds 2*NC stats followed by 2*N*G group sums (nn_gn_scratch sized accordingly) */
    gn_group_sums_k<<<nblk(s.n * G, 128), 128>>>(scratch, gamma, s.n, s.c, G, AB);
    gn_silu_bwd_apply_k<<<dim3(NC, KSLAB), 256>>>(x, gy, gamma, beta, mean, rstd, AB, gx, s.c, G, S);
    gn_param_grad_k2<<<nblk(s.c, 128), 128>>>(scratch, ggamma, gbeta, s.n, s.c);
    KCHECK();
}
extern "C" void nn_gn_silu_bwd(const float *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd, const float *gy,
                               float *gx, float *ggamma, float *gbeta, float *scratch) {
    if (G > s.c) G = s.c;
    if (ISMX(x)) {   /* MX x: voxel-major passes; gradients keep their storage */
        int NC = s.n * s.c;
        double *ds = gn_dsums((size_t)2 * NC);
        float *AB = scratch + 2 * NC;
        const int gdt = ISMX(gy) ? 3 : LPDT(GBF);
        if ((gdt == 3) != (ISMX(gx) != 0)) { fprintf(stderr, "gn_silu_bwd: gy and gx must share the MX storage\n"); abort(); }
        lp_gn_silu_bwd_mx(x, s, G, gamma, beta, mean, rstd, gy, gx, gdt, ds, scratch, AB);
        d2f_k<<<nblk(2 * NC, 128), 128>>>(ds, scratch, 2 * NC);
        gn_group_sums_k<<<nblk(s.n * G, 128), 128>>>(scratch, gamma, s.n, s.c, G, AB);
        lp_gn_silu_bwd_apply_mx(x, s, G, gamma, beta, mean, rstd, gy, gx, gdt, AB);
        gn_param_grad_k2<<<nblk(s.c, 128), 128>>>(scratch, ggamma, gbeta, s.n, s.c);
        KCHECK();
        return;
    }
    if (GBF && g_h16) gn_silu_bwd_t<f16, f16, f16>((const f16 *)x, s, G, gamma, beta, mean, rstd, (const f16 *)gy, (f16 *)gx, ggamma, gbeta, scratch);
    else if (GBF) gn_silu_bwd_t<bf16, bf16, bf16>((const bf16 *)x, s, G, gamma, beta, mean, rstd, (const bf16 *)gy, (bf16 *)gx, ggamma, gbeta, scratch);
    else if (ABF && g_h16) gn_silu_bwd_t<f16, float, float>((const f16 *)x, s, G, gamma, beta, mean, rstd, gy, gx, ggamma, gbeta, scratch);
    else if (ABF) gn_silu_bwd_t<bf16, float, float>((const bf16 *)x, s, G, gamma, beta, mean, rstd, gy, gx, ggamma, gbeta, scratch);
    else gn_silu_bwd_t<float, float, float>(x, s, G, gamma, beta, mean, rstd, gy, gx, ggamma, gbeta, scratch);
}

/* ================= GroupNorm =================
   Statistics are reduced by (n*G) x KSLAB blocks accumulating into double sums with atomics, then finalized. */
template <typename T = float>
__global__ void gn_sums_k(const T *x, int C, int G, size_t S, double *sums) {
    int ng = blockIdx.x, slab = blockIdx.y;
    int n = ng / G, g = ng % G, cpg = C / G;
    const T *p = x + ((size_t)n * C + (size_t)g * cpg) * S;
    size_t len = (size_t)cpg * S, per = (len + KSLAB - 1) / KSLAB, lo = (size_t)slab * per, hi = lo + per < len ? lo + per : len;
    double s1 = 0, s2 = 0;
    for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x) { double v = ldv(p, i); s1 += v; s2 += v * v; }
    __shared__ double r1[256], r2[256];
    r1[threadIdx.x] = s1; r2[threadIdx.x] = s2;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) { r1[threadIdx.x] += r1[threadIdx.x + o]; r2[threadIdx.x] += r2[threadIdx.x + o]; } __syncthreads(); }
    if (threadIdx.x == 0) { atomicAdd(&sums[2 * ng], r1[0]); atomicAdd(&sums[2 * ng + 1], r2[0]); }
}
__global__ void gn_finalize_k(const double *sums, int NG, size_t len, float eps, float *mean, float *rstd) {
    int ng = blockIdx.x * blockDim.x + threadIdx.x;
    if (ng >= NG) return;
    double m = sums[2 * ng] / (double)len, v = sums[2 * ng + 1] / (double)len - m * m;
    mean[ng] = (float)m;
    rstd[ng] = (float)(1.0 / sqrt((v > 0 ? v : 0) + (double)eps));
}
static double *gn_dsums(size_t n) {   /* small persistent device scratch for the double sums, per device */
    static double *buf[8]; static size_t cap[8];
    int d = cur_dev();
    if (n > cap[d]) { if (buf[d]) cudaFree(buf[d]); cudaMalloc(&buf[d], n * sizeof(double)); cap[d] = n; }
    return buf[d];
}

template <int SILU, typename TI, typename TO>
__global__ void gn_apply_k(const TI *x, const float *gamma, const float *beta, const float *mean, const float *rstd, TO *y, int C, int G, size_t S) {
    int nc = blockIdx.x, slab = blockIdx.y, n = nc / C, c = nc % C, cpg = C / G, ng = n * G + c / cpg;
    float m = mean[ng], r = rstd[ng], ga = gamma[c], be = beta[c];
    const TI *xp = x + (size_t)nc * S; TO *yp = y + (size_t)nc * S;
    size_t lo, hi; slab_range(S, slab, &lo, &hi);
    if (!(S & 3)) {
        for (size_t i = lo + 4 * threadIdx.x; i < hi; i += 4 * blockDim.x) {
            float xv[4], o[4]; ld4<TI>(xp + i, xv);
#pragma unroll
            for (int j = 0; j < 4; j++) { float v = (xv[j] - m) * r * ga + be; o[j] = SILU ? v / (1.f + __expf(-v)) : v; }
            st4<TO>(yp + i, o);
        }
    } else {
        for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x) { float v = (ldv(xp, i) - m) * r * ga + be; stv(yp, i, SILU ? v / (1.f + __expf(-v)) : v); }
    }
}

extern "C" void nn_gn_fwd(const float *x, shape5 s, int G, float eps, const float *gamma, const float *beta, float *y, float *mean, float *rstd) {
    if (G > s.c) G = s.c;
    size_t S = shape_spatial(s);
    int NG = s.n * G;
    double *sums = gn_dsums((size_t)2 * NG);
    cudaMemsetAsync(sums, 0, (size_t)2 * NG * sizeof(double));
    gn_sums_k<<<dim3(NG, KSLAB), 256>>>(x, s.c, G, S, sums);
    gn_finalize_k<<<nblk(NG, 128), 128>>>(sums, NG, (size_t)(s.c / G) * S, eps, mean, rstd);
    size_t n = shape_numel(s);
    if (y) gn_apply_k<0, float, float><<<dim3(s.n * s.c, KSLAB), 256>>>(x, gamma, beta, mean, rstd, y, s.c, G, S);   /* y == nullptr: statistics only */
    KCHECK();
}
/* GroupNorm statistics of a tensor in activation storage (16-bit on the tensor-core path); -1 for MX storage */
extern "C" int nn_gn_stats(const float *x, shape5 s, int G, float eps, float *mean, float *rstd) {
    if (ISMX(x)) return -1;
    if (G > s.c) G = s.c;
    size_t S = shape_spatial(s);
    int NG = s.n * G;
    double *sums = gn_dsums((size_t)2 * NG);
    cudaMemsetAsync(sums, 0, (size_t)2 * NG * sizeof(double));
    if (ABF && g_h16) gn_sums_k<f16><<<dim3(NG, KSLAB), 256>>>((const f16 *)x, s.c, G, S, sums);
    else if (ABF) gn_sums_k<bf16><<<dim3(NG, KSLAB), 256>>>((const bf16 *)x, s.c, G, S, sums);
    else gn_sums_k<float><<<dim3(NG, KSLAB), 256>>>(x, s.c, G, S, sums);
    gn_finalize_k<<<nblk(NG, 128), 128>>>(sums, NG, (size_t)(s.c / G) * S, eps, mean, rstd);
    KCHECK();
    return 0;
}
extern "C" void nn_gn_fwd_silu(const float *x, shape5 s, int G, float eps, const float *gamma, const float *beta, float *y, float *mean, float *rstd) {
    if (G > s.c) G = s.c;
    size_t S = shape_spatial(s);
    int NG = s.n * G;
    double *sums = gn_dsums((size_t)2 * NG);
    cudaMemsetAsync(sums, 0, (size_t)2 * NG * sizeof(double));
    gn_sums_k<<<dim3(NG, KSLAB), 256>>>(x, s.c, G, S, sums);
    gn_finalize_k<<<nblk(NG, 128), 128>>>(sums, NG, (size_t)(s.c / G) * S, eps, mean, rstd);
    size_t n = shape_numel(s);
    gn_apply_k<1, float, float><<<dim3(s.n * s.c, KSLAB), 256>>>(x, gamma, beta, mean, rstd, y, s.c, G, S);
    KCHECK();
}
extern "C" void nn_gn_apply_silu(const float *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd, float *g, float *sil) {
    if (G > s.c) G = s.c;
    size_t n = shape_numel(s), S = shape_spatial(s);
    gn_apply_k<0, float, float><<<dim3(s.n * s.c, KSLAB), 256>>>(x, gamma, beta, mean, rstd, g, s.c, G, S);
    silu_f_k<<<nblk(n, 256), 256>>>(g, sil, n);
    KCHECK();
}

/* per (n,c): sum gy, sum gy*xhat -> dsums[2*(n*C+c)] (double, atomics over slabs), then copied to st (float) */
__global__ void gn_bwd_stats_k(const float *x, const float *gy, const float *mean, const float *rstd, int C, int G, size_t S, double *ds) {
    int nc = blockIdx.x, slab = blockIdx.y, n = nc / C, c = nc % C, cpg = C / G, ng = n * G + c / cpg;
    const float *xp = x + (size_t)nc * S, *gp = gy + (size_t)nc * S;
    float m = mean[ng], r = rstd[ng];
    size_t per = (S + KSLAB - 1) / KSLAB, lo = (size_t)slab * per, hi = lo + per < S ? lo + per : S;
    double s1 = 0, s2 = 0;
    for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x) { double g = gp[i]; s1 += g; s2 += g * ((xp[i] - m) * r); }
    __shared__ double r1[256], r2[256];
    r1[threadIdx.x] = s1; r2[threadIdx.x] = s2;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) { r1[threadIdx.x] += r1[threadIdx.x + o]; r2[threadIdx.x] += r2[threadIdx.x + o]; } __syncthreads(); }
    if (threadIdx.x == 0) { atomicAdd(&ds[2 * nc], r1[0]); atomicAdd(&ds[2 * nc + 1], r2[0]); }
}
__global__ void d2f_k(const double *d, float *f, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) f[i] = (float)d[i]; }

__global__ void gn_bwd_apply_k(const float *x, const float *gy, const float *gamma, const float *mean, const float *rstd, const float *st,
                               float *gx, int N, int C, int G, size_t S) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * C * S) return;
    int c = (int)((i / S) % C), n = (int)(i / (S * C)), cpg = C / G, g = c / cpg, ng = n * G + g;
    /* group sums of gxhat = gy*gamma and gxhat*xhat */
    float a = 0.f, b = 0.f;
    for (int cc = g * cpg; cc < (g + 1) * cpg; cc++) { a += st[2 * (n * C + cc)] * gamma[cc]; b += st[2 * (n * C + cc) + 1] * gamma[cc]; }
    float len = (float)cpg * (float)S;
    float xhat = (x[i] - mean[ng]) * rstd[ng];
    float gxhat = gy[i] * gamma[c];
    gx[i] = rstd[ng] * (gxhat - a / len - xhat * b / len);
}

__global__ void gn_param_grad_k2(const float *st, float *ggamma, float *gbeta, int N, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float sg = 0.f, sb = 0.f;
    for (int n = 0; n < N; n++) { sb += st[2 * (n * C + c)]; sg += st[2 * (n * C + c) + 1]; }
    ggamma[c] += sg;
    gbeta[c] += sb;
}
__global__ void gn_param_grad_k(const float *st, float *ggamma, float *gbeta, int N, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float sg = 0.f, sb = 0.f;
    for (int n = 0; n < N; n++) { sb += st[2 * (n * C + c)]; sg += st[2 * (n * C + c) + 1]; }
    ggamma[c] += sg;
    gbeta[c] += sb;
}

extern "C" size_t nn_gn_scratch(shape5 s) { return (size_t)4 * s.n * s.c * sizeof(float); }
extern "C" void nn_gn_apply(const float *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd, float *y) {
    if (G > s.c) G = s.c;
    size_t n = shape_numel(s);
    gn_apply_k<0, float, float><<<dim3(s.n * s.c, KSLAB), 256>>>(x, gamma, beta, mean, rstd, y, s.c, G, shape_spatial(s));
    KCHECK();
}

extern "C" void nn_gn_bwd(const float *x, shape5 s, int G, const float *gamma, const float *mean, const float *rstd, const float *gy,
                          float *gx, float *ggamma, float *gbeta, float *scratch) {
    if (G > s.c) G = s.c;
    size_t S = shape_spatial(s);
    int NC = s.n * s.c;
    double *ds = gn_dsums((size_t)2 * NC);
    cudaMemsetAsync(ds, 0, (size_t)2 * NC * sizeof(double));
    gn_bwd_stats_k<<<dim3(NC, KSLAB), 256>>>(x, gy, mean, rstd, s.c, G, S, ds);
    d2f_k<<<nblk(2 * NC, 128), 128>>>(ds, scratch, 2 * NC);
    size_t n = shape_numel(s);
    gn_bwd_apply_k<<<nblk(n, 256), 256>>>(x, gy, gamma, mean, rstd, scratch, gx, s.n, s.c, G, S);
    gn_param_grad_k<<<nblk(s.c, 128), 128>>>(scratch, ggamma, gbeta, s.n, s.c);
    KCHECK();
}

/* ================= elementwise ================= */
__global__ void silu_f_k(const float *x, float *y, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; y[i] = v / (1.f + expf(-v)); }
}
__global__ void silu_b_k(const float *x, const float *gy, float *gx, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i], s = 1.f / (1.f + expf(-v)); gx[i] = gy[i] * (s * (1.f + v * (1.f - s))); }
}
__global__ void axpy_k(float *y, float a, const float *x, size_t n) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) y[i] += a * x[i]; }
__global__ void scale_k(float *y, float a, size_t n) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) y[i] *= a; }
__global__ void u8f_k(const uint8_t *x, float s, float *y, size_t n) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) y[i] = x[i] * s; }
__global__ void sigm_k(const float *x, float *y, size_t n) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) y[i] = 1.f / (1.f + expf(-x[i])); }

extern "C" void nn_silu_fwd(const float *x, size_t n, float *y) { silu_f_k<<<nblk(n, 256), 256>>>(x, y, n); KCHECK(); }
extern "C" void nn_silu_bwd(const float *x, const float *gy, size_t n, float *gx) { silu_b_k<<<nblk(n, 256), 256>>>(x, gy, gx, n); KCHECK(); }
extern "C" void nn_axpy(float *y, float a, const float *x, size_t n) { axpy_k<<<nblk(n, 256), 256>>>(y, a, x, n); KCHECK(); }
extern "C" void nn_scale(float *y, float a, size_t n) { scale_k<<<nblk(n, 256), 256>>>(y, a, n); KCHECK(); }
extern "C" void nn_u8_to_f32(const uint8_t *x, size_t n, float scale, float *y) { u8f_k<<<nblk(n, 256), 256>>>(x, scale, y, n); KCHECK(); }
template <typename HT> __global__ void f2h_k(const float *x, HT *y, size_t n, float scale) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) y[i] = f2h<HT>(x[i] * scale); }
/* fp32 -> the current 16-bit storage type (bf16, or fp16 with nn_set_f16), optionally scaled */
extern "C" void nn_f32_to_h16(const float *x, size_t n, void *y, float scale) { if (g_h16) f2h_k<f16><<<nblk(n, 256), 256>>>(x, (f16 *)y, n, scale); else f2h_k<bf16><<<nblk(n, 256), 256>>>(x, (bf16 *)y, n, scale); KCHECK(); }
extern "C" void nn_f32_to_bf16(const float *x, size_t n, void *y) { nn_f32_to_h16(x, n, y, 1.f); }
/* network input -> activation storage (16-bit, or MX-fp8 when y is registered MX) */
extern "C" void nn_f32_to_act(const float *x, shape5 s, void *y) {
    if (ISMX(y)) { lp_f32_to_mx8(x, s.n, s.c, shape_spatial(s), y); KCHECK(); return; }
    nn_f32_to_h16(x, shape_numel(s), y, 1.f);
}
/* 2:4 structured sparsity along the input channels of a [co][ci][taps] weight: in every group of 4 consecutive ci (same co, tap)
   only the two largest |w| survive. mask24: out = masked w (out may alias w). srste24: g = g (kept) + lambda * w (pruned):
   the sparse-refined straight-through estimator (pruned weights get a decaying pull so they can be revisited). */
__global__ void mask24_k(const float *w, float *out, int Co, int Ci, int T) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, ng = (size_t)Co * (Ci / 4) * T;
    if (i >= ng) return;
    int t = (int)(i % T), g4 = (int)((i / T) % (Ci / 4)), co = (int)(i / ((size_t)T * (Ci / 4)));
    size_t base = ((size_t)co * Ci + 4 * g4) * T + t;
    float v[4], a[4];
#pragma unroll
    for (int j = 0; j < 4; j++) { v[j] = w[base + (size_t)j * T]; a[j] = fabsf(v[j]); }
    int i0 = 0; for (int j = 1; j < 4; j++) if (a[j] > a[i0]) i0 = j;
    int i1 = -1; for (int j = 0; j < 4; j++) if (j != i0 && (i1 < 0 || a[j] > a[i1])) i1 = j;
#pragma unroll
    for (int j = 0; j < 4; j++) out[base + (size_t)j * T] = (j == i0 || j == i1) ? v[j] : 0.f;
}
__global__ void srste24_k(float *g, const float *w, int Co, int Ci, int T, float lambda) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, ng = (size_t)Co * (Ci / 4) * T;
    if (i >= ng) return;
    int t = (int)(i % T), g4 = (int)((i / T) % (Ci / 4)), co = (int)(i / ((size_t)T * (Ci / 4)));
    size_t base = ((size_t)co * Ci + 4 * g4) * T + t;
    float a[4];
#pragma unroll
    for (int j = 0; j < 4; j++) a[j] = fabsf(w[base + (size_t)j * T]);
    int i0 = 0; for (int j = 1; j < 4; j++) if (a[j] > a[i0]) i0 = j;
    int i1 = -1; for (int j = 0; j < 4; j++) if (j != i0 && (i1 < 0 || a[j] > a[i1])) i1 = j;
#pragma unroll
    for (int j = 0; j < 4; j++) if (j != i0 && j != i1) g[base + (size_t)j * T] += lambda * w[base + (size_t)j * T];
}
/* Weights kept on an fp8 (e4m3) or fp4 (e2m1) grid with MX block scaling (one power-of-two scale per 32 input channels
   of a (co, tap)): the master array holds the dequantized values, so the forward's quantization is exact and the model
   is a true fp8/fp4 model. Stochastic rounding keeps the expected update unbiased (round-to-nearest would discard
   optimizer steps far below the grid spacing). */
__device__ __forceinline__ float grid_spacing(float a, int bits) {   /* a = |x| / scale in [0, qmax] */
    if (bits == 8) { if (a < 0.015625f) return 0.001953125f; int e; frexpf(a, &e); return ldexpf(1.f, e - 1 - 3); }   /* e4m3: 3 mantissa bits, subnormal step 2^-9 */
    else { if (a < 1.f) return 0.5f; int e; frexpf(a, &e); return ldexpf(1.f, e - 1 - 1); }                           /* e2m1: 1 mantissa bit, subnormal step 0.5 */
}
__device__ __forceinline__ unsigned hash32(unsigned x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }
__global__ void wquant_k(float *w, int Co, int Ci, int T, int bits, unsigned seed) {
    size_t nb = (size_t)Co * ((Ci + 31) / 32) * T, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nb) return;
    int t = (int)(i % T), blk = (int)((i / T) % ((Ci + 31) / 32)), co = (int)(i / ((size_t)T * ((Ci + 31) / 32)));
    int c0 = blk * 32, c1 = c0 + 32 < Ci ? c0 + 32 : Ci;
    const float qmax = bits == 8 ? 448.f : 6.f;
    float amax = 0.f;
    for (int c = c0; c < c1; c++) amax = fmaxf(amax, fabsf(w[((size_t)co * Ci + c) * T + t]));
    if (amax == 0.f) return;
    int e = (int)ceilf(log2f(amax / qmax));
    float scale = ldexpf(1.f, e), inv = ldexpf(1.f, -e);
    for (int c = c0; c < c1; c++) {
        size_t k = ((size_t)co * Ci + c) * T + t;
        float x = w[k], a = fabsf(x) * inv;
        if (a > qmax) a = qmax;
        float sp = grid_spacing(a, bits), lo = floorf(a / sp) * sp, hi = lo + sp;
        if (hi > qmax) hi = qmax;
        float u = (float)(hash32((unsigned)k * 2654435761u ^ seed) & 0xffffff) * (1.f / 16777216.f);
        float q = (u < (a - lo) / sp) ? hi : lo;
        w[k] = copysignf(q * scale, x);
    }
}
/* ---- packed storage: one byte (e4m3) or one nibble (e2m1, low nibble first) per weight, plus one ue8m0 scale byte per block of
   32 input channels of a (co, tap). Element k = (co * Ci + ci) * T + t has block index ((co * nblk + ci / 32) * T + t). ---- */
__device__ __forceinline__ float sr_quant(float a, int bits, float qmax, unsigned rnd) {   /* a >= 0 on the unit grid, stochastic rounding */
    if (a > qmax) a = qmax;
    float sp = grid_spacing(a, bits), lo = floorf(a / sp) * sp, hi = lo + sp;
    if (hi > qmax) hi = qmax;
    float u = (float)(rnd & 0xffffff) * (1.f / 16777216.f);
    return (u < (a - lo) / sp) ? hi : lo;
}
__device__ __forceinline__ unsigned char enc_e4m3(float q) { unsigned short r; asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(0.f), "f"(q)); return (unsigned char)r; }
__device__ __forceinline__ float dec_e4m3(unsigned char b) {   /* e4m3: bias 7, 3 mantissa bits, no inf, 0x7f = nan (never produced) */
    int sg = b >> 7, e = (b >> 3) & 15, m = b & 7;
    float v = e ? ldexpf(1.f + m / 8.f, e - 7) : ldexpf(m / 8.f, -6);
    return sg ? -v : v;
}
__device__ __forceinline__ unsigned char enc_e2m1(float q) {   /* q in {0, .5, 1, 1.5, 2, 3, 4, 6} (unsigned) */
    static const float g[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    unsigned char best = 0; float bd = 1e9f;
    for (int i = 0; i < 8; i++) { float d = fabsf(g[i] - q); if (d < bd) { bd = d; best = (unsigned char)i; } }
    return best;
}
__device__ __forceinline__ float dec_e2m1(unsigned char n) { static const float g[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f}; return (n & 8) ? -g[n & 7] : g[n & 7]; }
/* storage index: fp8 = the element index k; fp4 = block-contiguous nibbles, (block index) * 32 + (ci - c0), so the 16 bytes
   of a block are written by the one thread that owns the block (no read-modify-write races between threads) */
__device__ __forceinline__ size_t wq_sidx(int bits, size_t k, size_t blk, int cc) { return bits == 8 ? k : blk * 32 + (size_t)cc; }
__device__ __forceinline__ float wq_get(const unsigned char *q, size_t si, int bits) {
    if (bits == 8) return dec_e4m3(q[si]);
    unsigned char b = q[si >> 1]; return dec_e2m1((si & 1) ? (b >> 4) : (b & 15));
}
__device__ __forceinline__ void wq_put(unsigned char *q, size_t si, int bits, float v) {   /* v already on the grid */
    if (bits == 8) { q[si] = enc_e4m3(v); return; }
    unsigned char n = enc_e2m1(fabsf(v)) | (v < 0 ? 8 : 0);
    unsigned char b = q[si >> 1];
    q[si >> 1] = (si & 1) ? (unsigned char)((b & 0x0f) | (n << 4)) : (unsigned char)((b & 0xf0) | n);
}
/* fp4 with error feedback: w = q4 * s + r8 * sr, r8 = e4m3 residual of the fp32 update that the fp4 grid cannot hold.
   The kernels read only q4 (true fp4 weights); the residual lives on the optimizer side. */
__device__ __forceinline__ void res_write(unsigned char *r, unsigned char *rsc, size_t blk, const float *res, int cn, const size_t *ks) {
    float amax = 0.f; for (int j = 0; j < cn; j++) amax = fmaxf(amax, fabsf(res[j]));
    int e = amax > 0.f ? (int)ceilf(log2f(amax / 448.f)) : -40; if (e < -40) e = -40; if (e > 60) e = 60;
    rsc[blk] = (unsigned char)(e + 127);
    float inv = ldexpf(1.f, -e);
    for (int j = 0; j < cn; j++) r[ks[j]] = enc_e4m3(res[j] * inv);
}
/* block-wise kernels: one thread per (co, block of 32 ci, tap) */
__global__ void wq_pack_k(const float *w, unsigned char *q, unsigned char *sc, int Co, int Ci, int T, int bits, unsigned seed) {
    int nblk = (Ci + 31) / 32; size_t nb = (size_t)Co * nblk * T, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nb) return;
    int t = (int)(i % T), blk = (int)((i / T) % nblk), co = (int)(i / ((size_t)T * nblk)), c0 = blk * 32, c1 = c0 + 32 < Ci ? c0 + 32 : Ci;
    const float qmax = bits == 8 ? 448.f : 6.f;
    float amax = 0.f;
    for (int c = c0; c < c1; c++) amax = fmaxf(amax, fabsf(w[((size_t)co * Ci + c) * T + t]));
    int e = amax > 0.f ? (int)ceilf(log2f(amax / qmax)) : -40; if (e < -40) e = -40; if (e > 60) e = 60;   /* clamp: keeps 1/scale finite; values below 2^-40 flush to 0 */
    sc[i] = (unsigned char)(e + 127);
    float inv = ldexpf(1.f, -e);
    for (int c = c0; c < c1; c++) {
        size_t k = ((size_t)co * Ci + c) * T + t; float x = w[k];
        float qv = seed ? sr_quant(fabsf(x) * inv, bits, qmax, hash32((unsigned)k * 2654435761u ^ seed)) : fminf(fabsf(x) * inv, qmax);
        if (!seed) { float sp = grid_spacing(qv, bits); qv = rintf(qv / sp) * sp; }
        wq_put(q, wq_sidx(bits, k, i, c - c0), bits, copysignf(qv, x));
    }
}
__global__ void wq_unpack_k(const unsigned char *q, const unsigned char *sc, float *w, int Co, int Ci, int T, int bits) {
    int nblk = (Ci + 31) / 32; size_t n = (size_t)Co * Ci * T, k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k >= n) return;
    int t = (int)(k % T), ci = (int)((k / T) % Ci), co = (int)(k / ((size_t)T * Ci));
    size_t b = ((size_t)co * nblk + ci / 32) * T + t;
    w[k] = wq_get(q, wq_sidx(bits, k, b, ci % 32), bits) * ldexpf(1.f, (int)sc[b] - 127);
}
/* AdamW directly on the packed weights: dequantize the block, update in fp32 (m, v stay fp32), rescale, requantize stochastically */
__global__ void wq_adamw_k(unsigned char *q, unsigned char *sc, unsigned char *r, unsigned char *rsc, const float *g, float *m, float *v, int Co, int Ci, int T, int bits,
                           float lr, float b1, float b2, float eps, float wd, float c1, float c2, unsigned seed) {
    int nblk = (Ci + 31) / 32; size_t nb = (size_t)Co * nblk * T, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nb) return;
    int t = (int)(i % T), blk = (int)((i / T) % nblk), co = (int)(i / ((size_t)T * nblk)), c0 = blk * 32, cn = c0 + 32 < Ci ? c0 + 32 : Ci;
    const float qmax = bits == 8 ? 448.f : 6.f;
    float scale = ldexpf(1.f, (int)sc[i] - 127), nw[32], amax = 0.f, rs = r ? ldexpf(1.f, (int)rsc[i] - 127) : 0.f;
    for (int c = c0; c < cn; c++) {
        size_t k = ((size_t)co * Ci + c) * T + t;
        float x = wq_get(q, wq_sidx(bits, k, i, c - c0), bits) * scale + (r ? dec_e4m3(r[k]) * rs : 0.f), gi = g[k];
        float mi = m[k] = b1 * m[k] + (1.f - b1) * gi, vi = v[k] = b2 * v[k] + (1.f - b2) * gi * gi;
        x -= lr * ((mi / c1) / (sqrtf(vi / c2) + eps) + wd * x);
        nw[c - c0] = x; amax = fmaxf(amax, fabsf(x));
    }
    int e = amax > 0.f ? (int)ceilf(log2f(amax / qmax)) : -40; if (e < -40) e = -40; if (e > 60) e = 60;   /* clamp: keeps 1/scale finite; values below 2^-40 flush to 0 */
    sc[i] = (unsigned char)(e + 127);
    float inv = ldexpf(1.f, -e);
    if (r) {   /* round to nearest on the fp4 grid, residual carries the rest */
        float res[32]; size_t ks[32];
        for (int c = c0; c < cn; c++) { size_t k = ((size_t)co * Ci + c) * T + t; float x = nw[c - c0], a = fminf(fabsf(x) * inv, qmax), sp = grid_spacing(a, bits), qv = rintf(a / sp) * sp; if (qv > qmax) qv = qmax; wq_put(q, wq_sidx(bits, k, i, c - c0), bits, copysignf(qv, x)); res[c - c0] = x - copysignf(qv, x) * ldexpf(1.f, e); ks[c - c0] = k; }
        res_write(r, rsc, i, res, cn - c0, ks);
        return;
    }
    for (int c = c0; c < cn; c++) { size_t k = ((size_t)co * Ci + c) * T + t; float x = nw[c - c0]; wq_put(q, wq_sidx(bits, k, i, c - c0), bits, copysignf(sr_quant(fabsf(x) * inv, bits, qmax, hash32((unsigned)k * 2654435761u ^ seed)), x)); }
}
/* EMA directly on packed weights: e = d e + (1 - d) p, both packed; result requantized stochastically */
__global__ void wq_ema_k(unsigned char *qe, unsigned char *sce, unsigned char *re, unsigned char *rsce, const unsigned char *qp, const unsigned char *scp, const unsigned char *rp, const unsigned char *rscp, int Co, int Ci, int T, int bits, float d, unsigned seed) {
    int nblk = (Ci + 31) / 32; size_t nb = (size_t)Co * nblk * T, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nb) return;
    int t = (int)(i % T), blk = (int)((i / T) % nblk), co = (int)(i / ((size_t)T * nblk)), c0 = blk * 32, cn = c0 + 32 < Ci ? c0 + 32 : Ci;
    const float qmax = bits == 8 ? 448.f : 6.f;
    float se = ldexpf(1.f, (int)sce[i] - 127), sp = ldexpf(1.f, (int)scp[i] - 127), nw[32], amax = 0.f;
    float rse = re ? ldexpf(1.f, (int)rsce[i] - 127) : 0.f, rsp = rp ? ldexpf(1.f, (int)rscp[i] - 127) : 0.f;
    for (int c = c0; c < cn; c++) { size_t k = ((size_t)co * Ci + c) * T + t, si = wq_sidx(bits, k, i, c - c0); float x = d * (wq_get(qe, si, bits) * se + (re ? dec_e4m3(re[k]) * rse : 0.f)) + (1.f - d) * (wq_get(qp, si, bits) * sp + (rp ? dec_e4m3(rp[k]) * rsp : 0.f)); nw[c - c0] = x; amax = fmaxf(amax, fabsf(x)); }
    int e = amax > 0.f ? (int)ceilf(log2f(amax / qmax)) : -40; if (e < -40) e = -40; if (e > 60) e = 60;   /* clamp: keeps 1/scale finite; values below 2^-40 flush to 0 */
    sce[i] = (unsigned char)(e + 127);
    float inv = ldexpf(1.f, -e);
    if (re) {
        float res[32]; size_t ks[32];
        for (int c = c0; c < cn; c++) { size_t k = ((size_t)co * Ci + c) * T + t; float x = nw[c - c0], a = fminf(fabsf(x) * inv, qmax), spc = grid_spacing(a, bits), qv = rintf(a / spc) * spc; if (qv > qmax) qv = qmax; wq_put(qe, wq_sidx(bits, k, i, c - c0), bits, copysignf(qv, x)); res[c - c0] = x - copysignf(qv, x) * ldexpf(1.f, e); ks[c - c0] = k; }
        res_write(re, rsce, i, res, cn - c0, ks);
        return;
    }
    for (int c = c0; c < cn; c++) { size_t k = ((size_t)co * Ci + c) * T + t; float x = nw[c - c0]; wq_put(qe, wq_sidx(bits, k, i, c - c0), bits, copysignf(sr_quant(fabsf(x) * inv, bits, qmax, hash32((unsigned)k * 2654435761u ^ seed)), x)); }
}
extern "C" size_t nn_wq_nblocks(int co, int ci, int taps) { return (size_t)co * ((ci + 31) / 32) * taps; }
extern "C" size_t nn_wq_bytes(int co, int ci, int taps, int bits) { return bits == 8 ? (size_t)co * ci * taps : nn_wq_nblocks(co, ci, taps) * 16; }
__global__ void wq_residual_k(const float *w, const unsigned char *q, const unsigned char *sc, unsigned char *r, unsigned char *rsc, int Co, int Ci, int T, int bits) {
    int nblk = (Ci + 31) / 32; size_t nb = (size_t)Co * nblk * T, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nb) return;
    int t = (int)(i % T), blk = (int)((i / T) % nblk), co = (int)(i / ((size_t)T * nblk)), c0 = blk * 32, cn = c0 + 32 < Ci ? c0 + 32 : Ci;
    float scale = ldexpf(1.f, (int)sc[i] - 127), res[32]; size_t ks[32];
    for (int c = c0; c < cn; c++) { size_t k = ((size_t)co * Ci + c) * T + t; res[c - c0] = w[k] - wq_get(q, wq_sidx(bits, k, i, c - c0), bits) * scale; ks[c - c0] = k; }
    res_write(r, rsc, i, res, cn - c0, ks);
}
extern "C" void nn_wq_pack(const float *w, void *q, void *sc, int co, int ci, int taps, int bits, unsigned seed) {
    size_t nb = nn_wq_nblocks(co, ci, taps); wq_pack_k<<<nblk(nb, 128), 128>>>(w, (unsigned char *)q, (unsigned char *)sc, co, ci, taps, bits, seed); KCHECK();
}
extern "C" void nn_wq_residual(const float *w, const void *q, const void *sc, void *r, void *rsc, int co, int ci, int taps, int bits) {
    size_t nb = nn_wq_nblocks(co, ci, taps); wq_residual_k<<<nblk(nb, 128), 128>>>(w, (const unsigned char *)q, (const unsigned char *)sc, (unsigned char *)r, (unsigned char *)rsc, co, ci, taps, bits); KCHECK();
}
extern "C" void nn_wq_unpack(const void *q, const void *sc, float *w, int co, int ci, int taps, int bits) {
    size_t n = (size_t)co * ci * taps; wq_unpack_k<<<nblk(n, 256), 256>>>((const unsigned char *)q, (const unsigned char *)sc, w, co, ci, taps, bits); KCHECK();
}
/* r / rsc: optional fp8 error-feedback residual (fp4 weights); nullptr = stochastic rounding without residual */
extern "C" void nn_wq_adamw(void *q, void *sc, void *r, void *rsc, const float *g, float *m, float *v, int co, int ci, int taps, int bits, float lr, float b1, float b2, float eps, float wd, int step, unsigned seed) {
    float c1 = 1.f - powf(b1, (float)step), c2 = 1.f - powf(b2, (float)step);
    size_t nb = nn_wq_nblocks(co, ci, taps);
    wq_adamw_k<<<nblk(nb, 128), 128>>>((unsigned char *)q, (unsigned char *)sc, (unsigned char *)r, (unsigned char *)rsc, g, m, v, co, ci, taps, bits, lr, b1, b2, eps, wd, c1, c2, seed); KCHECK();
}
extern "C" void nn_wq_ema(void *qe, void *sce, void *re, void *rsce, const void *qp, const void *scp, const void *rp, const void *rscp, int co, int ci, int taps, int bits, float decay, unsigned seed) {
    size_t nb = nn_wq_nblocks(co, ci, taps);
    wq_ema_k<<<nblk(nb, 128), 128>>>((unsigned char *)qe, (unsigned char *)sce, (unsigned char *)re, (unsigned char *)rsce, (const unsigned char *)qp, (const unsigned char *)scp, (const unsigned char *)rp, (const unsigned char *)rscp, co, ci, taps, bits, decay, seed); KCHECK();
}
extern "C" void nn_wquant(float *w, int co, int ci, int taps, int bits, unsigned seed) {
    size_t nb = (size_t)co * ((ci + 31) / 32) * taps; wquant_k<<<nblk(nb, 128), 128>>>(w, co, ci, taps, bits, seed); KCHECK();
}
extern "C" void nn_mask24(const float *w, float *out, int co, int ci, int taps) {
    if (ci % 4) { if (out != w) cudaMemcpyAsync(out, w, (size_t)co * ci * taps * 4, cudaMemcpyDeviceToDevice); return; }
    size_t ng = (size_t)co * (ci / 4) * taps; mask24_k<<<nblk(ng, 256), 256>>>(w, out, co, ci, taps); KCHECK();
}
extern "C" void nn_srste24(float *g, const float *w, int co, int ci, int taps, float lambda) {
    if (ci % 4) return;
    size_t ng = (size_t)co * (ci / 4) * taps; srste24_k<<<nblk(ng, 256), 256>>>(g, w, co, ci, taps, lambda); KCHECK();
}
extern "C" void nn_sigmoid(const float *x, size_t n, float *y) { sigm_k<<<nblk(n, 256), 256>>>(x, y, n); KCHECK(); }
/* inference input straight from the uint8 CT window: channel 0 = (ct - mean) * isd, 1 = 0, 2/3 = unit radial (y, x) vector
   from the scroll axis (dyo[z] = window y origin - axis y at slice z, likewise dxo); written as the 16-bit storage type
   (h16: fp16 with nn_set_f16, else bf16) or fp32. Output: recto probability * 255 where the CT is nonzero, else 0. */
template <typename HT> __global__ void pred_in_k(const uint8_t *ct, int W, float mean, float isd, const float *dyo, const float *dxo, int axis, HT *x) {
    size_t w3 = (size_t)W * W * W, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= w3) return;
    int xx = (int)(i % W), y = (int)((i / W) % W), z = (int)(i / ((size_t)W * W));
    float dy = dyo[z] + (float)y, dx = dxo[z] + (float)xx, inv = axis ? 1.f / (sqrtf(dy * dy + dx * dx) + 1e-6f) : 0.f;
    x[i] = f2h<HT>(((float)ct[i] - mean) * isd); x[w3 + i] = f2h<HT>(0.f); x[2 * w3 + i] = f2h<HT>(dy * inv); x[3 * w3 + i] = f2h<HT>(dx * inv);
}
extern "C" void nn_pred_input(const uint8_t *ct, int W, float mean, float isd, const float *dyo, const float *dxo, int axis, void *x, int h16) {
    size_t n = (size_t)W * W * W;
    if (!h16) pred_in_k<float><<<nblk(n, 256), 256>>>(ct, W, mean, isd, dyo, dxo, axis, (float *)x);
    else if (g_h16) pred_in_k<f16><<<nblk(n, 256), 256>>>(ct, W, mean, isd, dyo, dxo, axis, (f16 *)x);
    else pred_in_k<bf16><<<nblk(n, 256), 256>>>(ct, W, mean, isd, dyo, dxo, axis, (bf16 *)x);
    KCHECK();
}
__global__ void pred_out_k(const float *lg, const uint8_t *ct, size_t n, uint8_t *out) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = ct[i] ? (uint8_t)(255.f / (1.f + __expf(-lg[i])) + 0.5f) : 0;
}
extern "C" void nn_pred_output(const float *lg, const uint8_t *ct, size_t n, uint8_t *out) { pred_out_k<<<nblk(n, 256), 256>>>(lg, ct, n, out); KCHECK(); }

/* ================= trilinear 2x (align_corners = false) =================
   out[2m] = 0.75 in[m] + 0.25 in[m-1], out[2m+1] = 0.75 in[m] + 0.25 in[m+1], neighbours clamped. */
__device__ __forceinline__ void up_src(int o, int n_in, int *m0, int *m1, float *w1) {
    int m = o >> 1;
    if (o & 1) { *m0 = m; *m1 = min(m + 1, n_in - 1); } else { *m0 = m; *m1 = max(m - 1, 0); }
    *w1 = 0.25f;
}
/* forward exact-2x trilinear upsample: block = 4 x 8 x 16 outputs, the 4 x 6 x 10 input tile staged in smem */
template <typename T> __device__ __forceinline__ void st2(T *p, float a, float b);
template <> __device__ __forceinline__ void st2<float>(float *p, float a, float b) { *(float2 *)p = make_float2(a, b); }
template <> __device__ __forceinline__ void st2<bf16>(bf16 *p, float a, float b) { *(unsigned *)p = pack2(a, b); }
template <> __device__ __forceinline__ void st2<f16>(f16 *p, float a, float b) { *(unsigned *)p = packh<f16>(a, b); }
/* forward exact-2x trilinear upsample, separable: block = 4 x 8 x 32 outputs from a 4 x 6 x 18 input tile (staged with
   clamped source indices, which makes the edge clamping implicit); x-blend, y-blend, z-blend passes in shared memory */
template <typename TI, typename TO>
__global__ void __launch_bounds__(256) up2_f_k(const TI *x, TO *y, int N, int C, int D, int H, int W, int ctot, int c0, gnp_t gp = {}) {
    __shared__ float sx[4 * 6 * 18], tx[4 * 6 * 32], ty[4 * 8 * 32];
    const int ox0 = blockIdx.x * 32, oy0 = blockIdx.y * 8;
    int bz = blockIdx.z;
    const int nzt = (2 * D + 3) / 4;
    const int oz0 = (bz % nzt) * 4, nc = bz / nzt;
    const int n = nc / C, c = nc % C;
    const int Do = 2 * D, Ho = 2 * H, Wo = 2 * W;
    const TI *p = x + (size_t)nc * D * H * W;
    const int mz0 = oz0 / 2 - 1, my0 = oy0 / 2 - 1, mx0 = ox0 / 2 - 1;     /* input tile origin (may be -1) */
    for (int i = threadIdx.x; i < 4 * 6 * 18; i += 256) {
        int ix = i % 18, iy = (i / 18) % 6, iz = i / 108;
        int mz = min(max(mz0 + iz, 0), D - 1), my = min(max(my0 + iy, 0), H - 1), mx = min(max(mx0 + ix, 0), W - 1);
        sx[i] = ldv(p, ((size_t)mz * H + my) * W + mx);
        if (gp.G) sx[i] = gn_silu_at(sx[i], gp, n, c, C);   /* upsample of silu(gn(x)) */
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 4 * 6 * 32; i += 256) {   /* tx[iz][iy][oxl] */
        int oxl = i & 31, r = i >> 5;                        /* r = iz * 6 + iy */
        int ox = ox0 + oxl, m = (ox >> 1) - mx0, m1 = m + ((ox & 1) ? 1 : -1);
        tx[i] = 0.75f * sx[r * 18 + m] + 0.25f * sx[r * 18 + m1];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 4 * 8 * 32; i += 256) {   /* ty[iz][oyl][oxl] */
        int oxl = i & 31, oyl = (i >> 5) & 7, iz = i >> 8;
        int oy = oy0 + oyl, m = (oy >> 1) - my0, m1 = m + ((oy & 1) ? 1 : -1);
        ty[i] = 0.75f * tx[(iz * 6 + m) * 32 + oxl] + 0.25f * tx[(iz * 6 + m1) * 32 + oxl];
    }
    __syncthreads();
    const int txp = threadIdx.x & 15, oyl = (threadIdx.x >> 4) & 7, tz = threadIdx.x >> 7;   /* x pair, row, z half */
    const int ox = ox0 + 2 * txp, oy = oy0 + oyl;
    if (ox >= Wo || oy >= Ho) return;
#pragma unroll
    for (int dz = 0; dz < 2; dz++) {
        int oz = oz0 + 2 * tz + dz;
        if (oz >= Do) break;
        int m = (oz >> 1) - mz0, m1 = m + ((oz & 1) ? 1 : -1);
        const float *a = ty + (m * 8 + oyl) * 32 + 2 * txp, *b = ty + (m1 * 8 + oyl) * 32 + 2 * txp;
        float ve = 0.75f * a[0] + 0.25f * b[0], vo = 0.75f * a[1] + 0.25f * b[1];
        TO *yp = y + (((size_t)n * ctot + c0 + c) * Do + oz) * Ho * Wo + (size_t)oy * Wo + ox;
        if (ox + 1 < Wo) st2<TO>(yp, ve, vo); else stv(yp, 0, ve);
    }
}
/* coefficient of in[m] in out[o] along one axis */
__device__ __forceinline__ float up_coef(int o, int m, int n_in) {
    int m0, m1; float w1;
    up_src(o, n_in, &m0, &m1, &w1);
    return (m == m0 ? 0.75f : 0.f) + (m == m1 ? w1 : 0.f);
}
/* backward of the exact-2x trilinear upsample as a gather: gx[m] = sum over the 4^3 outputs o in [2m-1, 2m+2]
   of w(o,m) gy[o], with per-axis weights {0.25, 0.75, 0.75, 0.25} (edge-clamped). Block = 4 x 8 x 8 inputs;
   the 10 x 18 x 18 gradient tile is staged in shared memory once. */
__device__ __forceinline__ void up_wts(int m, int n_in, float *wt) {   /* weights of outputs 2m-1 .. 2m+2 */
    int n_out = 2 * n_in;
#pragma unroll
    for (int k = 0; k < 4; k++) { int o = 2 * m - 1 + k; wt[k] = (o >= 0 && o < n_out) ? up_coef(o, m, n_in) : 0.f; }
}
template <typename TG, typename TO>
__global__ void __launch_bounds__(256) up2_b_k(const TG *gy, TO *gx, int NC, int D, int H, int W, int C = 1, int ctot = 1, int c0 = 0) {   /* gx: channels c0.. of ctot */
    __shared__ float sg[10 * 18 * 18];                 /* gradient tile */
    __shared__ float hx[10 * 18 * 8], hy[10 * 8 * 8];  /* x-blended, then y-blended partials */
    const int tx = threadIdx.x & 7, ty = (threadIdx.x >> 3) & 7, tz = threadIdx.x >> 6;
    const int mx0 = blockIdx.x * 8, my0 = blockIdx.y * 8;
    int bz = blockIdx.z;
    const int nzt = (D + 3) / 4;
    const int mz0 = (bz % nzt) * 4, nc = bz / nzt;
    const int Do = 2 * D, Ho = 2 * H, Wo = 2 * W;
    const TG *g = gy + (size_t)nc * Do * Ho * Wo;
    /* tile row r = (iz, iy) covers columns 2*mx0 - 1 .. 2*mx0 + 16 (2*mx0 is a multiple of 16): four consecutive lanes load
       the four aligned 8-element chunks starting at 2*mx0 - 8, so a warp reads whole 32-byte sectors */
    for (int u = threadIdx.x; u < 10 * 18 * 4; u += 256) {
        int r = u >> 2, ch = u & 3, iz = r / 18, iy = r - iz * 18;
        int oz = 2 * mz0 - 1 + iz, oy = 2 * my0 - 1 + iy;
        float *row = sg + r * 18;
        float v[8];
        int cx = 2 * mx0 - 8 + 8 * ch;
        if (oz >= 0 && oz < Do && oy >= 0 && oy < Ho) {
            const TG *gr = g + ((size_t)oz * Ho + oy) * Wo;
            if (cx >= 0 && cx + 8 <= Wo && !(Wo & 7)) ld8<TG>(gr + cx, v);   /* 16-byte loads need 16-byte-aligned rows */
            else { for (int j = 0; j < 8; j++) v[j] = (cx + j >= 0 && cx + j < Wo) ? ldv(gr, cx + j) : 0.f; }
        } else {
#pragma unroll
            for (int j = 0; j < 8; j++) v[j] = 0.f;
        }
#pragma unroll
        for (int j = 0; j < 8; j++) { int e = 8 * ch - 7 + j; if (e >= 0 && e < 18) row[e] = v[j]; }
    }
    __syncthreads();
    /* separable backward: blend along x (4 taps), then y, then z; per-axis weights {0.25, 0.75, 0.75, 0.25} with edge clamping */
    float wq[4];
    for (int i = threadIdx.x; i < 10 * 18 * 8; i += 256) {   /* hx[iz][iy][mx] = sum_k wx[k] g[iz][iy][2 mx + k] */
        int mxl = i & 7, iy = (i >> 3) % 18, iz = i / 144;
        int mx = mx0 + mxl;
        float v = 0.f;
        if (mx < W) { up_wts(mx, W, wq); const float *row = sg + (iz * 18 + iy) * 18 + 2 * mxl; v = wq[0] * row[0] + wq[1] * row[1] + wq[2] * row[2] + wq[3] * row[3]; }
        hx[i] = v;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 10 * 8 * 8; i += 256) {    /* hy[iz][my][mx] = sum_k wy[k] hx[iz][2 my + k][mx] */
        int mxl = i & 7, myl = (i >> 3) & 7, iz = i >> 6;
        int my = my0 + myl;
        float v = 0.f;
        if (my < H) { up_wts(my, H, wq); const float *col = hx + (iz * 18 + 2 * myl) * 8 + mxl; v = wq[0] * col[0] + wq[1] * col[8] + wq[2] * col[16] + wq[3] * col[24]; }
        hy[i] = v;
    }
    __syncthreads();
    int mx = mx0 + tx, my = my0 + ty, mz = mz0 + tz;
    if (mx >= W || my >= H || mz >= D) return;
    up_wts(mz, D, wq);
    const float *col = hy + (2 * tz * 8 + ty) * 8 + tx;
    float acc = wq[0] * col[0] + wq[1] * col[64] + wq[2] * col[128] + wq[3] * col[192];
    const int nco = (nc / C) * ctot + c0 + nc % C;
    stv(gx, ((size_t)nco * D + mz) * H * W + (size_t)my * W + mx, acc);
}
extern "C" void nn_up2_fwd_gn_into(const float *x, shape5 xs, const nn_gn_t *g, float *y, int ctot, int c0) {
    const gnp_t gp = to_gnp(g);
    if (ISMX(x)) { if (!ISMX(y) || ctot != xs.c || c0) { fprintf(stderr, "up2: MX input needs a whole MX output tensor\n"); abort(); } lp_up2_fwd_mx(x, xs, y, gp); KCHECK(); return; }
    dim3 grid(nblk(2 * xs.w, 32), nblk(2 * xs.h, 8), (unsigned)(nblk(2 * xs.d, 4) * xs.n * xs.c));
    if (ABF && g_h16) up2_f_k<f16, f16><<<grid, 256>>>((const f16 *)x, (f16 *)y, xs.n, xs.c, xs.d, xs.h, xs.w, ctot, c0, gp);
    else if (ABF) up2_f_k<bf16, bf16><<<grid, 256>>>((const bf16 *)x, (bf16 *)y, xs.n, xs.c, xs.d, xs.h, xs.w, ctot, c0, gp);
    else up2_f_k<float, float><<<grid, 256>>>(x, y, xs.n, xs.c, xs.d, xs.h, xs.w, ctot, c0, gp);
    KCHECK();
}
extern "C" void nn_up2_fwd_into(const float *x, shape5 xs, float *y, int ctot, int c0) { nn_up2_fwd_gn_into(x, xs, nullptr, y, ctot, c0); }
extern "C" void nn_up2_fwd(const float *x, shape5 xs, float *y) { nn_up2_fwd_into(x, xs, y, xs.c, 0); }
extern "C" void nn_up2_bwd_into(const float *gy, shape5 xs, float *gx, int ctot, int c0) {
    if (ISMX(gy)) { if (!ISMX(gx) || ctot != xs.c || c0) { fprintf(stderr, "up2_bwd: MX gy needs a whole MX gx\n"); abort(); } lp_up2_bwd_mx(gy, xs, gx); KCHECK(); return; }
    dim3 grid(nblk(xs.w, 8), nblk(xs.h, 8), (unsigned)(nblk(xs.d, 4) * xs.n * xs.c));
    if (GBF && g_h16) up2_b_k<f16, f16><<<grid, 256>>>((const f16 *)gy, (f16 *)gx, xs.n * xs.c, xs.d, xs.h, xs.w, xs.c, ctot, c0);
    else if (GBF) up2_b_k<bf16, bf16><<<grid, 256>>>((const bf16 *)gy, (bf16 *)gx, xs.n * xs.c, xs.d, xs.h, xs.w, xs.c, ctot, c0);
    else up2_b_k<float, float><<<grid, 256>>>(gy, gx, xs.n * xs.c, xs.d, xs.h, xs.w, xs.c, ctot, c0);
    KCHECK();
}
extern "C" void nn_up2_bwd(const float *gy, shape5 xs, float *gx) { nn_up2_bwd_into(gy, xs, gx, xs.c, 0); }

/* ================= concat ================= */
__global__ void concat_k(const float *a, int ca, const float *b, int cb, float *y, int N, size_t S, int fwd) {
    int C = ca + cb;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * C * S) return;
    int c = (int)((i / S) % C), n = (int)(i / (S * C));
    size_t s = i % S;
    float *src = c < ca ? (float *)a + ((size_t)n * ca + c) * S + s : (float *)b + ((size_t)n * cb + c - ca) * S + s;
    if (fwd) y[i] = *src; else *src = y[i];
}
extern "C" void nn_concat_fwd(const float *a, int ca, const float *b, int cb, shape5 s, float *y) {
    size_t S = shape_spatial(s), n = (size_t)s.n * (ca + cb) * S;
    concat_k<<<nblk(n, 256), 256>>>(a, ca, b, cb, y, s.n, S, 1);
    KCHECK();
}
extern "C" void nn_concat_bwd(const float *gy, int ca, int cb, shape5 s, float *ga, float *gb) {
    size_t S = shape_spatial(s), n = (size_t)s.n * (ca + cb) * S;
    concat_k<<<nblk(n, 256), 256>>>(ga, ca, gb, cb, (float *)gy, s.n, S, 0);
    KCHECK();
}

/* ================= loss =================
   scratch layout per (n,c): [nmask, bce_sum, sum_sig_p, sum_sig, sum_p] (5 floats). */
static float g_posw = 1.f;   /* BCE weight of the positive class */
extern "C" void nn_set_pos_weight(float w) { g_posw = w; }
__global__ void loss_stats_k(const float *lg, const uint8_t *t, const uint8_t *m, const uint8_t *w, int C, size_t S, double *ds, float pw) {
    int nc = blockIdx.x, slab = blockIdx.y, n = nc / C;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0;   /* per-thread fp32 partials, fp64 block reduction */
    if (w[nc]) {
        const float *l = lg + (size_t)nc * S; const uint8_t *tp = t + (size_t)nc * S, *mp = m + (size_t)n * S;
        size_t per = (S + KSLAB - 1) / KSLAB, lo = (size_t)slab * per, hi = lo + per < S ? lo + per : S;
        for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
            if (!mp[i]) continue;
            float x = l[i], p = tp[i] * (1.f / 255.f);
            float sg = 1.f / (1.f + __expf(-x));
            float spp = fmaxf(x, 0.f) + log1pf(__expf(-fabsf(x)));   /* softplus(x) */
            float bce = pw * p * (spp - x) + (1.f - p) * spp;           /* softplus(-x) = softplus(x) - x */
            a0 += 1; a1 += bce; a2 += sg * p; a3 += sg; a4 += p;
        }
    }
    __shared__ double r[5][256];
    r[0][threadIdx.x] = a0; r[1][threadIdx.x] = a1; r[2][threadIdx.x] = a2; r[3][threadIdx.x] = a3; r[4][threadIdx.x] = a4;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) for (int k = 0; k < 5; k++) r[k][threadIdx.x] += r[k][threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) for (int k = 0; k < 5; k++) atomicAdd(&ds[nc * 5 + k], r[k][0]);
}
__global__ void loss_d2f_k(const double *d, float *f, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) f[i] = (float)d[i]; }
/* finalize on device: per-channel mean bce / dice, active count, and 1/active for the gradient kernel.
   layout of fin[]: [0..C) bce, [C..2C) dice, [2C] active, [2C+1] inv_active */
__global__ void loss_fin_k(const float *st, const uint8_t *w, int N, int C, float *fin) {
    if (threadIdx.x || blockIdx.x) return;
    int active = 0, cnt[16] = {0};
    for (int c = 0; c < 2 * C + 2; c++) fin[c] = 0.f;
    for (int nc = 0; nc < N * C; nc++) {
        if (!w[nc] || st[nc * 5] < 1.f) continue;
        active++;
        int c = nc % C;
        fin[c] += st[nc * 5 + 1] / st[nc * 5];
        fin[C + c] += 1.f - (2.f * st[nc * 5 + 2] + 1.f) / (st[nc * 5 + 3] + st[nc * 5 + 4] + 1.f);
        cnt[c]++;
    }
    for (int c = 0; c < C; c++) if (cnt[c]) { fin[c] /= cnt[c]; fin[C + c] /= cnt[c]; }
    fin[2 * C] = (float)active;
    fin[2 * C + 1] = active ? 1.f / active : 0.f;
}
template <typename GT> __global__ void loss_grad_k(const float *lg, const uint8_t *t, const uint8_t *m, const uint8_t *w, int N, int C, size_t S, const float *st,
                            float dice_w, const float *fin, GT *gl, float pw, float gscale) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * C * S) return;
    int nc = (int)(i / S), n = nc / C;
    if (!w[nc] || !m[(size_t)n * S + i % S]) { gl[i] = f2h<GT>(0.f); return; }
    float nm = st[nc * 5], Ssp = st[nc * 5 + 2], Ss = st[nc * 5 + 3], Sp = st[nc * 5 + 4];
    float x = lg[i], p = t[i] * (1.f / 255.f), s = 1.f / (1.f + expf(-x));
    float g = ((1.f - p) * s - pw * p * (1.f - s)) / fmaxf(nm, 1.f);
    float den = Ss + Sp + 1.f;
    float ddice_ds = -(2.f * p * den - (2.f * Ssp + 1.f)) / (den * den);
    g += dice_w * ddice_ds * s * (1.f - s);
    gl[i] = f2h<GT>(g * fin[2 * C + 1] * gscale);
}
static int g_loss_g16 = 0;
extern "C" void nn_set_loss_grad_h16(int on) { g_loss_g16 = on; }
/* scratch: 5 floats per (n,c) statistics followed by 2C+2 finalized values */
extern "C" size_t nn_loss_scratch(shape5 s) { return ((size_t)5 * s.n * s.c + 2 * s.c + 2) * sizeof(float); }
/* Asynchronous: launches the statistics, finalize and gradient kernels; nothing is copied to the host. */
extern "C" void nn_loss_async(const float *logits, const uint8_t *t, const uint8_t *m, const uint8_t *w, shape5 s, float dice_w, float *gl, float *scratch) {
    size_t S = shape_spatial(s);
    int NC = s.n * s.c;
    float *fin = scratch + (size_t)5 * NC;
    double *ds = gn_dsums((size_t)5 * NC);
    cudaMemsetAsync(ds, 0, (size_t)5 * NC * sizeof(double));
    loss_stats_k<<<dim3(NC, KSLAB), 256>>>(logits, t, m, w, s.c, S, ds, g_posw);
    loss_d2f_k<<<nblk(5 * NC, 128), 128>>>(ds, scratch, 5 * NC);
    loss_fin_k<<<1, 32>>>(scratch, w, s.n, s.c, fin);
    if (gl) {
        size_t n = shape_numel(s);
        if (!g_loss_g16) loss_grad_k<float><<<nblk(n, 256), 256>>>(logits, t, m, w, s.n, s.c, S, scratch, dice_w, fin, gl, g_posw, 1.f);
        else if (g_h16) loss_grad_k<f16><<<nblk(n, 256), 256>>>(logits, t, m, w, s.n, s.c, S, scratch, dice_w, fin, (f16 *)gl, g_posw, g_gscale);
        else loss_grad_k<bf16><<<nblk(n, 256), 256>>>(logits, t, m, w, s.n, s.c, S, scratch, dice_w, fin, (bf16 *)gl, g_posw, g_gscale);
    }
    KCHECK();
}
/* Copies the finalized values (2C+1 floats: bce per channel, dice per channel, active) to the host (synchronous). */
extern "C" void nn_loss_fetch(const float *scratch, shape5 s, float *out) {
    CK(cudaMemcpy(out, scratch + (size_t)5 * s.n * s.c, (size_t)(2 * s.c + 1) * sizeof(float), cudaMemcpyDeviceToHost));
}
extern "C" void nn_loss(const float *logits, const uint8_t *t, const uint8_t *m, const uint8_t *w, shape5 s, float dice_w,
                        float *gl, float *out, float *scratch) {
    nn_loss_async(logits, t, m, w, s, dice_w, gl, scratch);
    nn_loss_fetch(scratch, s, out);
}

/* ---- multi-GPU: copy between devices (peer access when available, else staged through the host) ---- */
extern "C" void nn_peer_copy(void *dst, int dst_dev, const void *src, int src_dev, size_t bytes) {
    static char enabled[8][8];
    if (!enabled[dst_dev & 7][src_dev & 7]) {
        int can = 0; cudaDeviceCanAccessPeer(&can, dst_dev, src_dev);
        if (can) { int cur; cudaGetDevice(&cur); cudaSetDevice(dst_dev); cudaDeviceEnablePeerAccess(src_dev, 0); cudaGetLastError(); cudaSetDevice(cur); }
        enabled[dst_dev & 7][src_dev & 7] = 1;
    }
    CK(cudaMemcpyPeer(dst, dst_dev, src, src_dev, bytes));
}

/* ================= optimizer / reductions ================= */
__global__ void adamw_k(float *p, const float *g, float *m, float *v, size_t n, float lr, float b1, float b2, float eps, float wd, float c1, float c2) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = g[i];
    float mi = m[i] = b1 * m[i] + (1.f - b1) * gi;
    float vi = v[i] = b2 * v[i] + (1.f - b2) * gi * gi;
    float mh = mi / c1, vh = vi / c2;
    p[i] -= lr * (mh / (sqrtf(vh) + eps) + wd * p[i]);
}
extern "C" void nn_adamw(float *p, const float *g, float *m, float *v, size_t n, float lr, float b1, float b2, float eps, float wd, int step) {
    float c1 = 1.f - powf(b1, (float)step), c2 = 1.f - powf(b2, (float)step);
    adamw_k<<<nblk(n, 256), 256>>>(p, g, m, v, n, lr, b1, b2, eps, wd, c1, c2);
    KCHECK();
}
__global__ void ema_k(float *e, const float *p, size_t n, float d) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) e[i] = d * e[i] + (1.f - d) * p[i]; }
extern "C" void nn_ema(float *ema, const float *p, size_t n, float decay) { ema_k<<<nblk(n, 256), 256>>>(ema, p, n, decay); KCHECK(); }

__global__ void sum_k(const float *x, size_t n, float *out, int sq) {
    double s = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { double v = x[i]; s += sq ? v * v : v; }
    __shared__ double r[256];
    r[threadIdx.x] = s;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) r[threadIdx.x] += r[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) out[blockIdx.x] = (float)r[0];
}
static double reduce(const float *x, size_t n, float *scratch, int sq) {
    int nb = 1024;
    sum_k<<<nb, 256>>>(x, n, scratch, sq);
    float h[1024];
    CK(cudaMemcpy(h, scratch, sizeof h, cudaMemcpyDeviceToHost));
    double s = 0;
    for (int i = 0; i < nb; i++) s += h[i];
    return s;
}
extern "C" double nn_sum(const float *x, size_t n, float *scratch) { return reduce(x, n, scratch, 0); }
extern "C" double nn_sumsq(const float *x, size_t n, float *scratch) { return reduce(x, n, scratch, 1); }

/* ---- fake quantization of stored activations (accuracy study of narrower storage formats) ----
   The tensor (activation storage type) is rounded in place to the values a block-scaled format can hold; groups are
   consecutive channels at one voxel. fmt: 1 NVFP4 (e2m1, e4m3 scale per 16, fp32 tensor scale), 2 MXFP4 (e2m1, ue8m0 per
   32), 3 MXFP6 e2m3, 4 MXFP6 e3m2, 5 MXFP8 e4m3 (ue8m0 per 32). */
__device__ __forceinline__ float fq_round(float x, int mbits, int emin, float maxv) {   /* round to nearest on the grid */
    float a = fabsf(x);
    if (a == 0.f) return 0.f;
    int e; frexpf(a, &e); e -= 1;                       /* a = 1.m * 2^e */
    if (e < emin) e = emin;
    float step = ldexpf(1.f, e - mbits);
    float r = rintf(a / step) * step;
    return copysignf(fminf(r, maxv), x);
}
__device__ __forceinline__ float e4m3_round(float x) { return fq_round(x, 3, -6, 448.f); }
template <typename T>
__global__ void fq_amax_k(const T *x, size_t n, unsigned *am) {
    float m = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) m = fmaxf(m, fabsf(ldv(x, i)));
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    if ((threadIdx.x & 31) == 0) atomicMax(am, __float_as_uint(m));
}
template <typename T>
__global__ void fq_k(T *x, int N, int C, size_t S, int fmt, const unsigned *am) {
    const int gs = fmt == 1 ? 16 : 32, ng = (C + gs - 1) / gs;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * ng * S) return;
    size_t v = i % S; int gi = (int)((i / S) % ng), n = (int)(i / (S * ng));
    T *p = x + ((size_t)n * C + gi * gs) * S + v;
    const int cn = min(gs, C - gi * gs);
    float amax = 0.f;
    for (int k = 0; k < cn; k++) amax = fmaxf(amax, fabsf(ldv(p, (size_t)k * S)));
    if (amax == 0.f) return;
    const int mb = fmt <= 2 ? 1 : fmt == 3 ? 3 : fmt == 4 ? 2 : 3, emin = fmt <= 3 ? 0 : fmt == 4 ? -2 : -6;
    const float qmax = fmt <= 2 ? 6.f : fmt == 3 ? 7.5f : fmt == 4 ? 28.f : 448.f;
    float sc;
    if (fmt == 1) {   /* NVFP4: e4m3 block scale relative to the fp32 tensor scale ts = amax_tensor / (6 * 448) */
        float ts = __uint_as_float(*am) / (6.f * 448.f);
        if (ts == 0.f) return;
        sc = e4m3_round(amax / 6.f / ts) * ts;
        if (sc == 0.f) sc = ldexpf(1.f, -9) * ts;
    } else {          /* MX: power-of-two scale with amax / scale <= qmax */
        int e; frexpf(amax / qmax, &e);
        sc = ldexpf(1.f, (amax / qmax) == ldexpf(1.f, e - 1) ? e - 1 : e);
    }
    for (int k = 0; k < cn; k++) { float val = ldv(p, (size_t)k * S); stv(p, (size_t)k * S, fq_round(val / sc, mb, emin, qmax) * sc); }
}
extern "C" void nn_fake_quant(void *x, shape5 s, int fmt) {
    if (fmt <= 0 || ISMX(x)) return;
    size_t S = shape_spatial(s), n = shape_numel(s);
    static unsigned *am[8];
    if (!am[cur_dev()]) cudaMalloc(&am[cur_dev()], 4);
    cudaMemsetAsync(am[cur_dev()], 0, 4);
    const int gs = fmt == 1 ? 16 : 32;
    size_t nt = (size_t)s.n * ((s.c + gs - 1) / gs) * S;
    if (ABF && g_h16) { fq_amax_k<f16><<<512, 256>>>((const f16 *)x, n, am[cur_dev()]); fq_k<f16><<<nblk(nt, 256), 256>>>((f16 *)x, s.n, s.c, S, fmt, am[cur_dev()]); }
    else if (ABF) { fq_amax_k<bf16><<<512, 256>>>((const bf16 *)x, n, am[cur_dev()]); fq_k<bf16><<<nblk(nt, 256), 256>>>((bf16 *)x, s.n, s.c, S, fmt, am[cur_dev()]); }
    else { fq_amax_k<float><<<512, 256>>>((const float *)x, n, am[cur_dev()]); fq_k<float><<<nblk(nt, 256), 256>>>((float *)x, s.n, s.c, S, fmt, am[cur_dev()]); }
    KCHECK();
}
