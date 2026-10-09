#pragma once
/* Shared declarations, device helpers and templates of the CUDA ops (src/nn_*.cu, split out of nn.cu so the
   sections compile as separate translation units in parallel). */
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

extern int g_tf32;
extern int g_gn_stored;
extern "C" void nn_set_gn_stored(int on);
extern "C" int nn_get_gn_stored(void);
extern "C" void nn_set_f16(int on);
extern int g_prec, g_pref;
extern "C" void nn_set_tf32(int on);
extern "C" void nn_set_prec(int p);
extern "C" int nn_get_prec(void);
/* per-layer precision: unet.c tags each conv with a layer id; a layer may override the global tensor-core precision */
#define NN_MAXLAYER 32
extern int g_layer, g_lprec[NN_MAXLAYER];
extern int g_lprec_init;
void lprec_init(void);
extern "C" void nn_set_layer(int id);
extern "C" void nn_set_nlev(int L);
extern "C" int nn_get_nlev(void);
extern "C" const char *nn_layer_name(int id, char *buf);
extern "C" void nn_set_layer_prec(int id, int p);
/* finer policy: per conv of a layer (sub 0 = c1, 1 = c2; down / head use sub 0) and per pass (0 forward, 1 backward-data,
   2 weight gradient); 0 = not set (falls back to the layer precision, then the global one) */
extern int g_sub, g_pass;
extern unsigned g_exec_prec[NN_MAXLAYER][2][3];
void exec_prec(int pass, int p);
/* stochastic rounding of fp8 gradient operands: one seed per conv call, derived from the step (deterministic per step) */
extern int g_sr;
extern unsigned g_sr_step, g_sr_ctr;
int sr_on(void);
unsigned sr_seed(void);
extern "C" void nn_set_sr(int on);

extern "C" void lp_wmemo_step(unsigned step); extern "C" void lp_wmemo_clear(void);
extern "C" void nn_set_sr_step(unsigned step);
extern "C" void nn_wmemo_clear(void);
/* id of the current conv call for the fp4 prepared-weight memo (0 outside a layer context) */
unsigned conv_wkey(void);
extern signed char g_lprec3[NN_MAXLAYER][2][3];
extern "C" void nn_set_conv(int sub);
extern "C" int nn_get_layer(void);
extern "C" int nn_get_conv(void);
extern "C" void nn_set_conv_prec(int id, int sub, int p_fwd, int p_bwd_data, int p_wgrad);
extern "C" int nn_get_conv_prec(int id, int sub, int pass);
/* quantization-aware training: the weight gradient may use its own (higher) precision while forward / backward-data run
   at the deployment precision; -1 = same as the layer precision */
extern int g_prec_w;
extern "C" void nn_set_prec_wgrad(int p);
int eff_prec_pass(int pass);
int eff_prec(void);
int eff_prec_w(void);
extern "C" int nn_cur_prec(void);
/* Requested precision policy; packed storage and unsupported kernel shapes can select a different compute path.
   nn_exec_manifest reports the paths observed during execution. Storage modes are appended by the caller. */
extern "C" int nn_prec_parse(const char *s);
extern "C" const char *nn_prec_name(int p);
/* policy string: "enc0=1,enc1=2,down0=2,dec2=3,head=1" or positional "1,1,2,2,2,2,2,2,2,1,1" (unet order: enc0..3, down0..2,
   dec2, dec1, dec0, head); values 1 bf16, 2 fp8, 3 fp4, 4 fp16 (fp16 operands, fp16 group accumulation) or their names;
   a layer left out keeps the global precision. Finer entries: a single conv of a block (enc1.c2, dec0.c1), "all", and
   per-pass values fwd:bwd_data:wgrad (e.g. dec0=fp16:fp16:fp8). Later entries override earlier ones. */
extern "C" int nn_set_prec_policy(const char *pol);
extern "C" int nn_get_tf32(void);
extern int g_actbf;
extern "C" void nn_set_act_bf16(int on);
extern "C" int nn_get_act_bf16(void);
#define ABF (g_tf32 && g_actbf)
extern int g_h16;
extern "C" void nn_set_f16(int on);
/* effective precision manifest (see nn.h) */
const char *pname16(int p);
extern "C" int nn_prec_manifest(char *buf, size_t n);
extern "C" int nn_exec_manifest(char *buf, size_t n);
extern "C" int nn_get_f16(void);
#define LPDT(flag) ((flag) ? (g_h16 ? 2 : 1) : 0)   /* storage code of the lp_* (nn_fp8.cu) entry points: 0 fp32, 1 bf16, 2 fp16 */
extern float g_gscale;
extern "C" void nn_set_grad_scale(float s);
extern "C" float nn_get_grad_scale(void);
extern int g_gradbf;
extern "C" void nn_set_grad_bf16(int on);
extern "C" int nn_get_grad_bf16(void);
#define GBF (ABF && g_gradbf)

extern cudaError_t g_err;
extern "C" const char *lp_check(void);
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess && g_err == cudaSuccess) { g_err = e_; if (ufsm_env_on("UFSM_CUDA_TRACE")) fprintf(stderr,"CUDA at %s:%d (%s): %s\n",__FILE__,__LINE__,#x,cudaGetErrorString(e_)); } } while (0)
#define KCHECK() CK(cudaGetLastError())

extern "C" int nn_init(int device);
extern "C" const char *nn_check(void);
extern "C" void *nn_malloc(size_t n);
/* ---- per-tensor storage registry: a tensor registered as MX-fp8 (dt 8) takes the MX paths of the ops that read or
   write it (see nn_set_storage in nn.h); lookups match any address inside a registered range ---- */
#define NN_MAXREG 1024
struct g_reg_t { const char *p; size_t n; int dt; };
extern g_reg_t g_reg[NN_MAXREG];
extern int g_nreg;
extern "C" void nn_storage_forget(const void *p);
extern "C" void nn_set_storage(const void *p, size_t bytes, int dt);
extern "C" int nn_storage(const void *p);
/* registry dt 8 = MX-fp8, 4 = MX-fp4; MXDT(p) = the lp dtype of a registered tensor (3 fp8, 4 fp4), 0 = plane-major */
static inline int mxdt_of(const void *p) { int d = g_nreg ? nn_storage(p) : 0; return d == 8 ? 3 : d == 4 ? 4 : 0; }
#define MXDT(p) mxdt_of(p)
#define ISMX(p) (mxdt_of(p) != 0)
#define ISMX4(p) (mxdt_of(p) == 4)
extern "C" size_t nn_mx8_bytes(shape5 s);
extern "C" size_t nn_mx4_bytes(shape5 s);
extern "C" size_t nn_mx_bytes(shape5 s, int dt);
extern "C" void nn_free(void *p);
extern "C" void nn_zero(void *p, size_t n);
extern "C" void nn_h2d(void *d, const void *s, size_t n);
extern "C" void nn_d2h(void *d, const void *s, size_t n);
extern "C" void nn_d2d(void *d, const void *s, size_t n);
extern "C" void nn_sync(void);
extern "C" void *nn_host_alloc(size_t n);
extern "C" void nn_host_free(void *p);
/* per-device copy stream (non-blocking, overlaps the legacy compute stream) and events for host<->device pipelining */
cudaStream_t copy_stream(void);
extern "C" void nn_h2d_copy_stream(void *d, const void *s, size_t n);
extern "C" void *nn_event_create(void);
extern "C" void nn_event_record(void *e, int on_copy_stream);
extern "C" void nn_stream_wait(int copy_stream_waits, void *e);
extern "C" void nn_event_sync(void *e);
/* ---- event profiler: GPU timestamps around ops, no host syncs ---- */
#define NPROF 8192
extern cudaEvent_t g_ev[NPROF][2];
extern int g_evk[NPROF], g_nev, g_ev_init;
extern "C" void nn_prof_begin(int k);
extern "C" void nn_prof_end(void);
/* sums elapsed ms per category into out[nk], resets */
extern "C" void nn_prof_collect(double *out, int nk);
extern "C" size_t nn_mem_free(void);

#ifndef KSLAB
#define KSLAB 32
#endif
int cur_dev(void);
double *gn_dsums(size_t n);
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
template <> __device__ __forceinline__ float h2f<float>(float v) { return v; }
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
gnp_t to_gnp(const nn_gn_t *g);
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
__global__ void prep_w16_k(const float *w, __half *wp, float *wsc, int Co, int Ci, int Cop, int Cip);
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
    size_t per = ((S + gridDim.y - 1) / gridDim.y + 3) & ~(size_t)3;   /* slabs = gridDim.y (KSLAB for large tensors) */
    *lo = (size_t)slab * per; *hi = *lo + per < S ? *lo + per : S;
}
/* forward tile: FW_TZ output planes x 8 rows x 16 columns per block, one warp per (plane, row pair) */
#define FW_TZ 2   /* 4 (512 threads) was tried: MT=4 spills, net slower */
#define FW_TZ_MT1 4   /* default z tile for MT = 1 (fw_tz): -14% on the level-0 16-channel convs against 2; 6 is slower (occupancy) */
#define FW_NT (FW_TZ * 128)
#define FW_T ((FW_TZ + 2) * 10 * 18)
#define FW_R 4   /* output rows per warp (MT = 4 keeps 2: accumulator registers) */
__host__ __device__ constexpr int fw_rows(int MT) { return MT == 4 ? 2 : FW_R; }
__host__ __device__ constexpr int fw_nth(int MT) { return 32 * FW_TZ * 8 / fw_rows(MT); }
__host__ __device__ constexpr int fw_blocks(int MT) { return fw_nth(MT) <= 128 ? 3 : (MT == 1 ? 3 : 2); }
/* deeper z tiles (TZ output planes per block): less halo restaging and fewer weight loads per output; used for MT = 1 */
__host__ __device__ constexpr int fw_nth_z(int MT, int FZ) { return 32 * FZ * 8 / fw_rows(MT); }
__host__ __device__ constexpr int fw_blocks_z(int MT, int FZ) { return FZ == FW_TZ ? fw_blocks(MT) : (fw_nth_z(MT, FZ) <= 256 ? 2 : 1); }
/* OP = 1 (prec 4, HT = bf16 or f16 storage): fp16 operands with fp16 accumulation per tap group. The tile is staged in the storage
   type (exact) while tracking the block amax, then converted in place to fp16 scaled by 2^-ex (amax < 16); weights are
   fp16 scaled per output channel (wsc, amax < 16), so a 144-term group sum stays below 36864 < 65504. Each group's fp16
   sum is folded into the fp32 accumulators times 2^ex * wsc[co]. */
/* XP = 1 (Ci <= 4, plain input): x-shift packing. Staged channel slot k holds input channel k % 4 shifted by k / 4 columns,
   so one K = 16 MMA covers the taps kx = 0..2 of 4 channels (weights wp[kz * 3 + ky][co][kx * 4 + c], kx = 3 zero):
   9 tap rows instead of 27 padded taps. */
template <int MT, typename TI, typename TO, int S2B, typename HT, int OP = 0, int FZ = FW_TZ, int XP = 0>
__global__ void __launch_bounds__(fw_nth_z(MT, FZ), fw_blocks_z(MT, FZ)) conv_fwd_tc_k(const TI *__restrict__ x, const HT *__restrict__ wp, const float *__restrict__ b, TO *__restrict__ y,
                              int N, int Ci, int D, int H, int W, int Co, int Cop, int Cip, gnp_t gp, double *__restrict__ osum, int Go, split_t sp, tapset_t ts,
                              const float *__restrict__ wsc = nullptr) {
    constexpr int BM = MT * 16, TG = 9, WDB = MT <= 2;  /* WDB: double-buffered weight groups */
    constexpr int R = fw_rows(MT), NTH = fw_nth_z(MT, FZ);    /* output rows per warp, threads per block */
    constexpr int TT_ = (FZ + 2) * 10 * 18;                     /* staged positions */
    extern __shared__ __align__(32) unsigned char smem_raw[];
    HT *sx = (HT *)smem_raw;                       /* [TT_ pos][TC_CI ci] */
    HT *wa = sx + TC_CI * TT_;                      /* [2 if WDB][TG tap][BM co][TC_CI ci] */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wz = warp / (8 / R), wr = (warp % (8 / R)) * R;
    const int ox0 = blockIdx.x * 16, oy0 = blockIdx.y * TC_TY;
    int bz = blockIdx.z;
    const int nzt = (D + FZ - 1) / FZ;
    const int oz0 = (bz % nzt) * FZ; bz /= nzt;
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
        if constexpr (XP) {   /* x-shift packing: slot k = (shift k / 4, channel k % 4) */
            const int k = threadIdx.x & 15, c = k & 3, sh = k >> 2;
            const bool cok = c < Ci && sh < 3;
            const TI *xc = x + ((size_t)n * Ci + (cok ? c : 0)) * plane;
            for (int rr = threadIdx.x >> 4; rr < (FZ + 2) * 10; rr += NTH / 16) {
                int iz = rr / 10, iy = rr - iz * 10;
                int gz = oz0 - 1 + iz, gyy = oy0 - 1 + iy;
                HT *dst = sx + (size_t)rr * 18 * TC_CI + k;
                const bool rok = cok && gz >= 0 && gz < D && gyy >= 0 && gyy < H;
                const TI *xr = xc + ((size_t)(rok ? gz : 0) * H + (rok ? gyy : 0)) * W;
#pragma unroll
                for (int j = 0; j < 18; j++) { const int col = ox0 - 1 + j + sh; dst[j * TC_CI] = f2h<HT>(rok && col >= 0 && col < W ? ldv(xr, col) : 0.f); }
            }
        } else
        {   /* row-wise staging: lane k = ci, 16 lanes share a row; a thread converts a whole 18-element row (vector loads) */
            const int k = threadIdx.x & 15, ci = ci0 + k;
            const bool cok = ci < Ci;
            const bool xfull = ox0 + 16 <= W && !(W & 3);   /* vector path: interior in range and 8/16-byte row alignment */
            const bool upc = sp.up && !(sp.x2 && ci >= sp.c_split);   /* half-resolution x segment, upsampled while staging */
            const TI *xc = (sp.x2 && ci >= sp.c_split) ? (const TI *)sp.x2 + ((size_t)n * (Ci - sp.c_split) + (cok ? ci - sp.c_split : 0)) * plane
                                                       : x + ((size_t)n * (sp.x2 ? sp.c_split : Ci) + (cok ? ci : 0)) * (upc ? plane >> 3 : plane);
            float ga = 1.f, gb = 0.f;
            const bool gtr = cok && in_gn(gp, sp, n, ci, Ci, &ga, &gb);
            for (int rr = threadIdx.x >> 4; rr < (FZ + 2) * 10; rr += NTH / 16) {   /* (FZ + 2) z-planes x 10 rows */
                int iz = rr / 10, iy = rr - iz * 10;
                int gz = oz0 - 1 + iz, gyy = oy0 - 1 + iy;
                float v[18];
                HT *dst = sx + (size_t)rr * 18 * TC_CI + k;
                if (cok && upc && gz >= 0 && gz < D && gyy >= 0 && gyy < H) {
                    row18_up2(xc, gz, gyy, ox0, D, H, W, v);
#pragma unroll
                    for (int j = 0; j < 18; j++) { HT hv = f2h<HT>(v[j]); dst[j * TC_CI] = hv; if (OP) amax = fmaxf(amax, fabsf(h2f<HT>(hv))); }
                }
#ifdef TC_NOSTAGE
                else if (cok) { for (int j = 0; j < 18; j++) dst[j * TC_CI] = f2h<HT>(0.25f); }   /* diagnostic: no global loads */
#endif
                else if (cok && gz >= 0 && gz < D && gyy >= 0 && gyy < H) {
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
            tile_to_f16<HT>(sx, TT_ * TC_CI, ldexpf(1.f, -ex), NTH);
            amax = ldexpf(1.f, ex);                            /* reuse: tile scale 2^ex for the fold */
        }
        __syncthreads();
        const int NTAP = XP ? 9 : S2B ? ts.ntap : 27;
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
            if constexpr (XP) {   /* 9 tap rows (kz, ky); the K slots carry kx = 0..2 */
#pragma unroll
                for (int kz = 0; kz < 3; kz++) {
                    unsigned bf[R + 2][4];
#pragma unroll
                    for (int rr = 0; rr < R + 2; rr++) {
                        int mat = lane >> 3, q = mat >> 1, kh = mat & 1;
                        int pos = ((wz + kz) * 10 + wr + rr) * 18 + q * 8 + (lane & 7);
                        ldmatrix_x4(bf[rr], sx + pos * TC_CI + kh * 8);
                    }
#pragma unroll
                    for (int ky = 0; ky < 3; ky++) {
                        const int tt = kz * 3 + ky;
                        unsigned af[4];
                        int mat = lane >> 3;
                        ldmatrix_x4(af, wg + (tt * BM + (mat & 1) * 8 + (lane & 7)) * TC_CI + (mat >> 1) * 8);
#pragma unroll
                        for (int r = 0; r < R; r++) { mma16816<HT>(acc[0][r][0], af, bf[r + ky]); mma16816<HT>(acc[0][r][1], af, bf[r + ky] + 2); }
                    }
                }
            } else
            if constexpr (!S2B) {   /* one kz plane per group: a warp's R rows x 3 ky span R + 2 input rows per kx, loaded once */
                const int kz = t0 / 9;
#pragma unroll
                for (int kx = 0; kx < 3; kx++) {
                    unsigned bf[R + 2][4];
#pragma unroll
                    for (int rr = 0; rr < R + 2; rr++) {
                        int mat = lane >> 3, q = mat >> 1, kh = mat & 1;
                        int pos = ((wz + kz) * 10 + wr + rr) * 18 + kx + q * 8 + (lane & 7);
                        ldmatrix_x4(bf[rr], sx + pos * TC_CI + kh * 8);
                    }
#pragma unroll
                    for (int ky = 0; ky < 3; ky++) {
                        const int tt = ky * 3 + kx;
#pragma unroll
                        for (int m = 0; m < MT; m++) {
                            unsigned af[4];
                            int mat = lane >> 3;
                            ldmatrix_x4(af, wg + (tt * BM + m * 16 + (mat & 1) * 8 + (lane & 7)) * TC_CI + (mat >> 1) * 8);
#pragma unroll
                            for (int r = 0; r < R; r++) {
                                if (OP) { mma16816_h(hacc[OP ? m : 0][OP ? r : 0][0], af, bf[r + ky]); mma16816_h(hacc[OP ? m : 0][OP ? r : 0][1], af, bf[r + ky] + 2); }
                                else { mma16816<HT>(acc[m][r][0], af, bf[r + ky]); mma16816<HT>(acc[m][r][1], af, bf[r + ky] + 2); }
                            }
                        }
                    }
                }
            } else
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
    const int zs0 = sp.zlo, zs1 = D - sp.zhi;          /* statistics over the z planes this GPU owns (spatial split) */
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
                const bool zst = oz >= zs0 && oz < zs1;
                TO *yp = S2B ? y + (((size_t)n * Co + co) * ts.Dx + (2 * oz + ts.pz)) * ts.Hx * ts.Wx + (size_t)(2 * oy + ts.py) * ts.Wx + ts.px
                             : (sp.y2 && co >= sp.o_split) ? (TO *)sp.y2 + (((size_t)n * (Co - sp.o_split) + co - sp.o_split) * D + oz) * H * W + (size_t)oy * W
                                                           : y + (((size_t)n * (sp.y2 ? sp.o_split : Co) + co) * D + oz) * H * W + (size_t)oy * W;
#pragma unroll
                for (int q = 0; q < 2; q++) {
                    int ox = ox0 + q * 8 + 2 * t;
                    float v0 = acc[m][r][q][2 * h] + bias, v1 = acc[m][r][q][2 * h + 1] + bias;
                    if (Go > 0 && is_f16<TO>::v) {   /* also clamp normalization inputs during replay without statistics */
                        v0 = sat_h16(v0); v1 = sat_h16(v1);
                    }
                    if (!S2B && !(W & 1) && ox + 1 < W) { stv2(yp, (size_t)ox, v0, v1); if (zst) { ps += v0 + v1; pss += v0 * v0 + v1 * v1; } continue; }   /* paired store */
                    if (ox < W) { stv(yp, (size_t)ox * (S2B ? 2 : 1), v0); if (zst) { ps += v0; pss += v0 * v0; } }
                    if (ox + 1 < W) { stv(yp, (size_t)(ox + 1) * (S2B ? 2 : 1), v1); if (zst) { ps += v1; pss += v1 * v1; } }
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
                if (iz == 5 || iy == 9) continue;   /* plane 5 / row 9 of the 6 x 10 layout are never read (taps reach 2o + 1) */
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
            for (int i = threadIdx.x * 4; i < TG * BM * TC_CI; i += 256 * 4) {   /* 4 ci per load (rows of 16 ci) */
                int tt = i / (BM * TC_CI), r = i % (BM * TC_CI), c = r / TC_CI, k = r % TC_CI;
                *(uint2 *)(wa + i) = *(const uint2 *)(wp + ((size_t)(t0 + tt) * Cop + co0 + c) * Cip + ci0 + k);
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

int cur_dev(void);

/* ---- spatial split of one window along z across two GPUs (nn_split_cfg, per device): every tensor of this GPU covers its own
   z planes plus halo planes owned by the other GPU (lo0 at the low end and hi0 at the high end of the level-0 depth D0; a
   level-l tensor of depth D has lo0 D / D0 and hi0 D / D0 of them). GroupNorm statistics skip the halo planes and are summed
   across the two GPUs (the reduce callback), so mean / rstd are those of the whole window (Dg0 planes at level 0). */
struct g_zs_t { int on, lo0, hi0, D0, Dg0; };
extern g_zs_t g_zs[10];
extern void (*g_split_reduce)(double *, int);
extern "C" void nn_split_cfg(int lo0, int hi0, int D0, int Dg0);
extern "C" void nn_split_set_reduce(void (*fn)(double *, int));
int zs_on(void);
int zs_slot(void);   /* index of this GPU's (or, both halves on one GPU, this half's) split state */
void zs_range(int D, int *lo, int *hi);
size_t zs_len(size_t len, int D);
void zs_reduce(double *b, int n);
split_t zs_split(split_t sp, int D);
int gn_fused_stored(const void *y, int cout);
/* x-shift packed weights for XP: wp[kz * 3 + ky][co][kx * 4 + c] (kx = 3 and c >= Ci zero), Cop rows */
template <typename HT>
__global__ void prep_wxp_k(const float *w, HT *wp, int Co, int Ci, int Cop) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 9 * Cop * 16) return;
    int k = i % 16, co = (i / 16) % Cop, t9 = i / (16 * Cop), kx = k >> 2, c = k & 3, kz = t9 / 3, ky = t9 % 3;
    wp[i] = f2h<HT>(kx < 3 && c < Ci && co < Co ? w[((size_t)co * Ci + c) * 27 + kz * 9 + ky * 3 + kx] : 0.f);
}
void *tc_wbuf(size_t n);

/* Tensor-core path for k=3, stride 1 (same spatial size). Returns 0 if handled. */
/* output z planes per block: deeper tiles for MT = 1 (Cop = 16: the level-0 convs) amortise the halo and the weight loads
   over more outputs (env UFSM_FW_TZ = 2 / 4 / 6 overrides) */
int fw_tz(int MT);
size_t fw_smem(int MT, int tz);
template <typename TI, typename TO, int S2B, typename HT, int OP = 0>
static void conv_fwd_tc_launch(int MT, dim3 grid, size_t smem, const TI *x, const HT *wp, const float *b, TO *y, shape5 xs, int cout, int Cop, int Cip, gnp_t gp, double *osum, int Go, split_t sp, tapset_t ts, const float *wsc = nullptr) {
    static int attr_set[8][8];
    const int tz = fw_tz(MT);
    const int ai = MT == 1 ? (tz == 4 ? 5 : tz == 6 ? 6 : 1) : MT;
    if (!attr_set[cur_dev()][ai]) {
        cudaFuncSetAttribute(MT == 1 ? (tz == 4 ? (const void *)conv_fwd_tc_k<1, TI, TO, S2B, HT, OP, 4> : tz == 6 ? (const void *)conv_fwd_tc_k<1, TI, TO, S2B, HT, OP, 6> : (const void *)conv_fwd_tc_k<1, TI, TO, S2B, HT, OP>)
                             : MT == 2 ? (const void *)conv_fwd_tc_k<2, TI, TO, S2B, HT, OP> : (const void *)conv_fwd_tc_k<4, TI, TO, S2B, HT, OP>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
        attr_set[cur_dev()][ai] = 1;
    }
    switch (MT) {
    case 1:
        if (tz == 4) conv_fwd_tc_k<1, TI, TO, S2B, HT, OP, 4><<<grid, fw_nth_z(1, 4), smem>>>(x, wp, b, y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, Cip, gp, osum, Go, sp, ts, wsc);
        else if (tz == 6) conv_fwd_tc_k<1, TI, TO, S2B, HT, OP, 6><<<grid, fw_nth_z(1, 6), smem>>>(x, wp, b, y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, Cip, gp, osum, Go, sp, ts, wsc);
        else conv_fwd_tc_k<1, TI, TO, S2B, HT, OP><<<grid, fw_nth(1), smem>>>(x, wp, b, y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, Cip, gp, osum, Go, sp, ts, wsc);
        break;
    case 2: conv_fwd_tc_k<2, TI, TO, S2B, HT, OP><<<grid, fw_nth(2), smem>>>(x, wp, b, y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, Cip, gp, osum, Go, sp, ts, wsc); break;
    default: conv_fwd_tc_k<4, TI, TO, S2B, HT, OP><<<grid, fw_nth(4), smem>>>(x, wp, b, y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, Cop, Cip, gp, osum, Go, sp, ts, wsc); break;
    }
}
float *tc_wscbuf(size_t n);
/* prec 4 (16-bit storage HT): fp16 operands with fp16 group accumulation folded into fp32 (conv_fwd_tc_k OP = 1) */
template <typename HT>
int conv_fwd_tc_f16acc(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, gnp_t gp, double *osum, int Go, split_t sp, const tapset_t *ts) {
    int Cop = (cout + 15) / 16 * 16, Cip = (xs.c + TC_CI - 1) / TC_CI * TC_CI;
    HT *wp = (HT *)tc_wbuf((size_t)27 * Cop * Cip);
    float *wsc = tc_wscbuf(Cop);
    prep_w16_k<<<Cop, 256>>>(w, (__half *)wp, wsc, cout, xs.c, Cop, Cip);
    int MT = Cop % 64 == 0 ? 4 : Cop % 32 == 0 ? 2 : 1;
    int nmt = Cop / (MT * 16);
    const int tz = fw_tz(MT);
    dim3 grid(nblk(xs.w, 16), nblk(xs.h, TC_TY), (unsigned)(nblk(xs.d, tz) * nmt * xs.n));
    size_t smem = fw_smem(MT, tz);
    tapset_t none = {};
    const tapset_t &tt = ts ? *ts : none;
    if (ts) {
        if (xbf && ybf) conv_fwd_tc_launch<HT, HT, 1, HT, 1>(MT, grid, smem, (const HT *)x, wp, b, (HT *)y, xs, cout, Cop, Cip, gp, osum, Go, sp, tt, wsc);
        else VERIFY_ONLY(conv_fwd_tc_launch<float, float, 1, HT, 1>(MT, grid, smem, (const float *)x, wp, b, (float *)y, xs, cout, Cop, Cip, gp, osum, Go, sp, tt, wsc));
    }
    else if (xbf && ybf) conv_fwd_tc_launch<HT, HT, 0, HT, 1>(MT, grid, smem, (const HT *)x, wp, b, (HT *)y, xs, cout, Cop, Cip, gp, osum, Go, sp, tt, wsc);
    else if (xbf) VERIFY_ONLY(conv_fwd_tc_launch<HT, float, 0, HT, 1>(MT, grid, smem, (const HT *)x, wp, b, (float *)y, xs, cout, Cop, Cip, gp, osum, Go, sp, tt, wsc));
    else if (ybf) VERIFY_ONLY(conv_fwd_tc_launch<float, HT, 0, HT, 1>(MT, grid, smem, (const float *)x, wp, b, (HT *)y, xs, cout, Cop, Cip, gp, osum, Go, sp, tt, wsc));
    else VERIFY_ONLY(conv_fwd_tc_launch<float, float, 0, HT, 1>(MT, grid, smem, (const float *)x, wp, b, (float *)y, xs, cout, Cop, Cip, gp, osum, Go, sp, tt, wsc));
    return 0;
}
template <typename HT>
int conv_fwd_tc_h(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, gnp_t gp, double *osum, int Go, split_t sp, const tapset_t *ts) {
    int Cop = (cout + 15) / 16 * 16, Cip = (xs.c + TC_CI - 1) / TC_CI * TC_CI;
    size_t nw = (size_t)27 * Cop * Cip;
    HT *wp = (HT *)tc_wbuf(nw);
    prep_w_k<HT><<<nblk(nw, 256), 256>>>(w, wp, cout, xs.c, Cop, Cip);
    static int xpe = -1; if (xpe < 0) { const char *e = getenv("UFSM_XPACK"); xpe = e ? atoi(e) : 1; }
    if (xpe && xs.c <= 4 && Cop == 16 && !ts && !sp.x2 && !sp.y2 && !sp.accum && !sp.up && !gp.G) {   /* tiny input (the network input): x-shift packing */
        prep_wxp_k<HT><<<nblk(9 * 16 * 16, 256), 256>>>(w, wp, cout, xs.c, 16);
        const int tz = 4;
        dim3 grid(nblk(xs.w, 16), nblk(xs.h, TC_TY), (unsigned)(nblk(xs.d, tz) * xs.n));
        size_t smem = fw_smem(1, tz);
        static int at[8]; if (!at[cur_dev()]) { cudaFuncSetAttribute((const void *)conv_fwd_tc_k<1, HT, HT, 0, HT, 0, 4, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); cudaFuncSetAttribute((const void *)conv_fwd_tc_k<1, float, HT, 0, HT, 0, 4, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); at[cur_dev()] = 1; }
        tapset_t nt = {};
        if (xbf && ybf) conv_fwd_tc_k<1, HT, HT, 0, HT, 0, 4, 1><<<grid, fw_nth_z(1, 4), smem>>>((const HT *)x, wp, b, (HT *)y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, 16, 16, gp, osum, Go, sp, nt);
        else if (!xbf && ybf) conv_fwd_tc_k<1, float, HT, 0, HT, 0, 4, 1><<<grid, fw_nth_z(1, 4), smem>>>((const float *)x, wp, b, (HT *)y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, 16, 16, gp, osum, Go, sp, nt);
        else goto plain;
        return 0;
    }
plain:
    int MT = Cop % 64 == 0 ? 4 : Cop % 32 == 0 ? 2 : 1;   /* BM = 16 MT must divide Cop (e.g. Cop = 48 -> MT = 1) */
    int nmt = Cop / (MT * 16);
    const int tz = fw_tz(MT);
    dim3 grid(nblk(xs.w, 16), nblk(xs.h, TC_TY), (unsigned)(nblk(xs.d, tz) * nmt * xs.n));
    size_t smem = fw_smem(MT, tz);
    tapset_t none = {};
    if (ts) {
        if (xbf && ybf) conv_fwd_tc_launch<HT, HT, 1, HT>(MT, grid, smem, (const HT *)x, wp, b, (HT *)y, xs, cout, Cop, Cip, gp, osum, Go, sp, *ts);
        else VERIFY_ONLY(conv_fwd_tc_launch<float, float, 1, HT>(MT, grid, smem, (const float *)x, wp, b, (float *)y, xs, cout, Cop, Cip, gp, osum, Go, sp, *ts));
    }
    else if (xbf && ybf) conv_fwd_tc_launch<HT, HT, 0, HT>(MT, grid, smem, (const HT *)x, wp, b, (HT *)y, xs, cout, Cop, Cip, gp, osum, Go, sp, none);
    else if (xbf) VERIFY_ONLY(conv_fwd_tc_launch<HT, float, 0, HT>(MT, grid, smem, (const HT *)x, wp, b, (float *)y, xs, cout, Cop, Cip, gp, osum, Go, sp, none));
    else if (ybf) VERIFY_ONLY(conv_fwd_tc_launch<float, HT, 0, HT>(MT, grid, smem, (const float *)x, wp, b, (HT *)y, xs, cout, Cop, Cip, gp, osum, Go, sp, none));
    else VERIFY_ONLY(conv_fwd_tc_launch<float, float, 0, HT>(MT, grid, smem, (const float *)x, wp, b, (float *)y, xs, cout, Cop, Cip, gp, osum, Go, sp, none));
    return 0;
}
/* xbf / ybf: input / output tensors are 16-bit (bf16, or fp16 with nn_set_f16) instead of float.
   ts != nullptr: parity-decomposed stride-2 backward-data (x = gy on its own grid, output scattered into gx) */
int conv_fwd_tc(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, gnp_t gp, double *osum, int Go, split_t sp, const tapset_t *ts = nullptr);

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
int conv_fwd_tc_s2_h(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, shape5 ys, gnp_t gp) {
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
int conv_fwd_tc_s2(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, shape5 ys, gnp_t gp = {});

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

extern "C" shape5 nn_conv3d_out_shape(shape5 xs, int cout, int k, int stride);

extern "C" void nn_conv3d_fwd(const float *x, shape5 xs, const float *w, const float *b, int cout, int k, int stride, float *y);

/* ---- backward data ----
   stride 1: gx = conv(gy, w') with w'[ci][co][flipped tap]  (same fwd kernel).
   stride 2: gather kernel. */
__global__ void transpose_w_k(const float *w, float *wt, int Co, int Ci, int T);
__global__ void flip_w_k(const float *w, float *wt, int Co, int Ci, int T);

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

__global__ void dilate2_k(const float *gy, float *gd, int N, int C, int D, int H, int W, int Do, int Ho, int Wo);

/* scratch: flipped weights, plus (stride 2) the zero-inserted gradient at input resolution */
/* backward-data scratch: the flipped/transposed weights, plus (fp32 kernels only) the zero-inserted gradient for stride 2 */
extern "C" size_t nn_conv3d_scratch(shape5 xs, int cout, int k);

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
void bwd_data_impl_(const float *gy, shape5 ys, const float *w, shape5 xs, int k, int stride, float *gx, float *scratch, int accum);
void bwd_data_impl(const float *gy, shape5 ys, const float *w, shape5 xs, int k, int stride, float *gx, float *scratch, int accum);
void bwd_data_impl_(const float *gy, shape5 ys, const float *w, shape5 xs, int k, int stride, float *gx, float *scratch, int accum);
extern "C" void nn_conv3d_bwd_data(const float *gy, shape5 ys, const float *w, shape5 xs, int k, int stride, float *gx, float *scratch);
/* gx += backward-data (tensor-core k=3 paths only; returns -1 otherwise, caller falls back to bwd_data + axpy) */
extern "C" int nn_conv3d_bwd_data_acc(const float *gy, shape5 ys, const float *w, shape5 xs, int k, int stride, float *gx, float *scratch);

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
    const int ci0 = blockIdx.z * 8 * NT, co0 = blockIdx.y * 16;
    int bz = blockIdx.x;
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
        if (gb && blockIdx.z == 0 && threadIdx.x < 16) {
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
    if (gb && blockIdx.z == 0 && threadIdx.x < 16 && co0 + (int)threadIdx.x < Co) atomicAdd(&gb[co0 + threadIdx.x], bsum);
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
    /* Spatial tiles can exceed CUDA's 65535 limit for grid.z at large windows.
       Put them on grid.x; the channel-block dimensions are small. */
    dim3 grid((unsigned)(nblk(ys.w, 16) * nblk(ys.h, TC_TY) * nzc * ys.n), (ys.c + 15) / 16, (xs.c + 8 * NT - 1) / (8 * NT));
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
/* fp4 weight gradient (lp_bwd_w_f4) for prec-3 weight-gradient passes: opt-in (UFSM_F4_WGRAD=1) until it passes the stairs;
   UFSM_F4_HAD_W=1 adds the fixed-sign H32 on both operands. Stride 2 stays fp8. */
int f4_wgrad(void);
int f4_had_w(void);
void launch_bwd_w_tc(const void *x, int xbf, shape5 xs, const void *gy, int gybf, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp);

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
    const int ci0 = blockIdx.z * S2W_CI, co0 = blockIdx.y * 16;
    int bz = blockIdx.x;
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
        if (gb && blockIdx.z == 0 && threadIdx.x < 16) { const HT *row = sg + threadIdx.x * 64; for (int v = 0; v < 64; v++) bsum += h2f<HT>(row[v]); }
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
    if (gb && blockIdx.z == 0 && threadIdx.x < 16 && co0 + (int)threadIdx.x < Co) atomicAdd(&gb[co0 + threadIdx.x], bsum);
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
    dim3 grid((unsigned)(nblk(ys.w, 8) * nblk(ys.h, 4) * nzc * ys.n), (ys.c + 15) / 16, (xs.c + S2W_CI - 1) / S2W_CI);
    if (GBF) conv_bwd_w_tc_s2_k<HT, HT, HT><<<grid, 256, smem>>>((const HT *)x, (const HT *)gy, gw, gb, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, ys.d, ys.h, ys.w, gp);
    else if (ABF) conv_bwd_w_tc_s2_k<HT, float, HT><<<grid, 256, smem>>>((const HT *)x, gy, gw, gb, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, ys.d, ys.h, ys.w, gp);
    else conv_bwd_w_tc_s2_k<float, float, HT><<<grid, 256, smem>>>(x, gy, gw, gb, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, ys.d, ys.h, ys.w, gp);
}
template <typename HT>
static void bwd_w1_h(const float *x, shape5 xs, const float *gy, shape5 ys, float *gw, size_t So, gnp_t xg = {}) {
    int nb = (int)((So + 2047) / 2048); if (nb > 4096) nb = 4096; if (nb < 1) nb = 1;
    if (xs.c * ys.c > 64) {   /* wide heads (affinity outputs): up to 128 channel pairs */
        if (GBF) conv_bwd_w1_k<HT, HT, 128><<<nb, 256>>>((const HT *)x, (const HT *)gy, gw, xs.n, xs.c, ys.c, So, xg);
        else if (ABF) conv_bwd_w1_k<HT, float, 128><<<nb, 256>>>((const HT *)x, gy, gw, xs.n, xs.c, ys.c, So, xg);
        else conv_bwd_w1_k<float, float, 128><<<nb, 256>>>(x, gy, gw, xs.n, xs.c, ys.c, So, xg);   /* exact fp32 (reference) */
        return;
    }
    if (GBF) conv_bwd_w1_k<HT, HT, 64><<<nb, 256>>>((const HT *)x, (const HT *)gy, gw, xs.n, xs.c, ys.c, So, xg);
    else if (ABF) conv_bwd_w1_k<HT, float, 64><<<nb, 256>>>((const HT *)x, gy, gw, xs.n, xs.c, ys.c, So, xg);
    else conv_bwd_w1_k<float, float, 64><<<nb, 256>>>(x, gy, gw, xs.n, xs.c, ys.c, So, xg);
}
extern "C" void nn_conv3d_bwd_weight(const float *x, shape5 xs, const float *gy, shape5 ys, int k, int stride, float *gw, float *gb);

__global__ void silu_f_k(const float *x, float *y, size_t n);

/* y = conv3d(silu(gn(x))) for k=3 stride 1 on the tensor-core path; returns -1 when that path is unavailable */
extern "C" int nn_conv3d_fwd_gn(const float *x, shape5 xs, int G, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                const float *w, const float *b, int cout, float *y);
/* Same conv (optionally with the gn+silu input transform when G_in > 0) that also produces the GroupNorm
   statistics of its OUTPUT for G_out groups: mean/rstd of y. Tensor-core path only; -1 when unavailable. */
extern "C" int nn_conv3d_fwd_gn_stats(const float *x, shape5 xs, int G_in, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                      const float *w, const float *b, int cout, float *y, int G_out, float eps, float *omean, float *orstd);
/* Forward with a channel-split input: channels [0, c_split) from x (xs.c = c_split + channels of x2), the rest
   from x2; otherwise like nn_conv3d_fwd_gn_stats (G_in applies gn+silu to BOTH inputs with the same params). */
extern "C" int nn_conv3d_fwd_split(const float *x, const float *x2, int c_split, shape5 xs, int G_in, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                   const float *w, const float *b, int cout, float *y, int G_out, float eps, float *omean, float *orstd);
/* Backward-data (k=3, stride 1) writing input-channel gradients [0, o_split) to gx and the rest to gx2. */
extern "C" int nn_conv3d_bwd_data_split(const float *gy, shape5 ys, const float *w, shape5 xs, float *gx, float *gx2, int o_split, float *scratch);
/* Backward-data (k=3, stride 1) of input channels [c0, c0 + nc) only, into gx (nc channels). scratch as for
   nn_conv3d_bwd_data. Lets a caller produce a wide input gradient in channel chunks. */
extern "C" int nn_conv3d_bwd_data_range(const float *gy, shape5 ys, const float *w, shape5 xs, int c0, int nc, float *gx, float *scratch);
/* y = silu(gn(x)) from precomputed statistics, one pass */
extern "C" void nn_gn_silu_apply(const float *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd, float *y);
/* gw += dconv/dw with the conv input silu(gn(x)) recomputed at staging; -1 when unavailable */
extern "C" int nn_conv3d_bwd_weight_gn(const float *x, shape5 xs, int G, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                       const float *gy, shape5 ys, float *gw, float *gb);
/* Weight gradient with a channel-split input (see nn_conv3d_fwd_split); G > 0 applies gn+silu to both inputs. */
extern "C" int nn_conv3d_bwd_weight_split(const float *x, const float *x2, int c_split, shape5 xs, int G, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                          const float *gy, shape5 ys, float *gw, float *gb);

/* ---- input-side recompute: the conv input silu(gn(.)) of a stored pre-norm tensor (and, for the decoder, the nearest
   upsample of the coarse block output) is formed while staging, so neither the normalized activation nor the
   upsampled tensor is stored ---- */
/* fused upsample: 16-bit tensor-core kernels only. An fp8 / fp4 policy on that conv is served by the 16-bit kernels
   (more precise; keeps the transient away); MX-stored inputs cannot be read by them -> -1 (caller uses the transient) */
/* MX inputs: the forward stages the coarse MX rows directly (fp8 / fp4 kernels, stage_up32); the MX weight gradient still
   takes the transient (UFSM_MX_UP=0: transient for the forward too) */
/* fused decoder upsample in the MX weight gradient (UFSM_MX_UP_W, default 1): the fp8 / fp4 kernels' cooperative decode
   interpolates the half-resolution x segment while staging (no full-resolution transient). Needs both segments MX of one
   type, 16-channel-aligned segments (the tiles never straddle) and, for the fp4 kernel, a layout with the decoded tile
   (not LY 0: the H32 / x-SR modes and UFSM_F4W_LAYOUT=0 keep the transient). */
int f4_had_w(void);
int f4_wgrad(void);
int eff_prec_w(void);
int mx_up_w_ok(const float *x, const float *x2, int c_split, shape5 xs);
int up_kernel_ok(const float *x, const float *x2, int wgrad);
int xsplit(const float *x2, const nn_gn_t *gx, const nn_gn_t *gx2, int c_split, int up, shape5 xs, split_t *sp);
extern "C" int nn_conv3d_fwd_x(const float *x, const nn_gn_t *gx, const float *x2, const nn_gn_t *gx2, int c_split, int up, shape5 xs,
                               const float *w, const float *b, int cout, int k, int stride, float *y, int G_out, float eps, float *omean, float *orstd);
extern "C" int nn_conv3d_bwd_weight_x(const float *x, const nn_gn_t *gx, const float *x2, const nn_gn_t *gx2, int c_split, int up, shape5 xs,
                                      const float *gy, shape5 ys, int k, int stride, float *gw, float *gb);

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
extern "C" void nn_silu_bwd_gn(const float *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd, const float *gy, float *gx);


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
__global__ void gn_group_sums_k(const float *st, const float *gamma, int N, int C, int G, float *AB, float f = 1.f);
/* spatial split: the per-(n, c) backward sums ds (2 NC doubles) of this GPU give its share of the GroupNorm parameter
   gradients (summed across GPUs with the weight gradients), then are summed across the GPUs for the group sums AB, which the
   apply kernels divide by their local element count: f rescales them to the whole window's count */
__global__ void gn_param_grad_k2(const float *st, float *ggamma, float *gbeta, int N, int C);
float gn_bwd_reduce(double *ds, float *st, int NC, int D, float *ggamma, float *gbeta, int N, int C);
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
    const int nsl = (int)(S / 8192 < 1 ? 1 : S / 8192 > KSLAB ? KSLAB : S / 8192);   /* >= 8k elements per block: tiny slabs were launch-bound */
    gn_silu_bwd_stats_k<<<dim3(NC, nsl), 256>>>(x, gy, gamma, beta, mean, rstd, s.c, G, S, ds);
    d2f_k<<<nblk(2 * NC, 128), 128>>>(ds, scratch, 2 * NC);
    float *AB = scratch + 2 * NC;   /* scratch holds 2*NC stats followed by 2*N*G group sums (nn_gn_scratch sized accordingly) */
    const int zsp = zs_on();
    const float f = gn_bwd_reduce(ds, scratch, NC, s.d, ggamma, gbeta, s.n, s.c);   /* spatial split: local param grads, global sums */
    gn_group_sums_k<<<nblk(s.n * G, 128), 128>>>(scratch, gamma, s.n, s.c, G, AB, f);
    gn_silu_bwd_apply_k<<<dim3(NC, nsl), 256>>>(x, gy, gamma, beta, mean, rstd, AB, gx, s.c, G, S);
    if (!zsp) gn_param_grad_k2<<<nblk(s.c, 128), 128>>>(scratch, ggamma, gbeta, s.n, s.c);
    KCHECK();
}
extern "C" void nn_gn_silu_bwd(const float *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd, const float *gy,
                               float *gx, float *ggamma, float *gbeta, float *scratch);

/* ================= GroupNorm =================
   Statistics are reduced by (n*G) x KSLAB blocks accumulating into double sums with atomics, then finalized. */
template <typename T = float>
__global__ void gn_sums_k(const T *x, int C, int G, size_t S, double *sums, size_t v0 = 0, size_t v1 = ~(size_t)0) {   /* voxels [v0, v1) only */
    int ng = blockIdx.x, slab = blockIdx.y;
    int n = ng / G, g = ng % G, cpg = C / G;
    const T *p = x + ((size_t)n * C + (size_t)g * cpg) * S;
    size_t len = (size_t)cpg * S, per = (len + KSLAB - 1) / KSLAB, lo = (size_t)slab * per, hi = lo + per < len ? lo + per : len;
    double s1 = 0, s2 = 0;
    const bool all = v0 == 0 && v1 >= S;
    for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x) { if (!all) { size_t v = i % S; if (v < v0 || v >= v1) continue; } double v = ldv(p, i); s1 += v; s2 += v * v; }
    __shared__ double r1[256], r2[256];
    r1[threadIdx.x] = s1; r2[threadIdx.x] = s2;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) { r1[threadIdx.x] += r1[threadIdx.x + o]; r2[threadIdx.x] += r2[threadIdx.x + o]; } __syncthreads(); }
    if (threadIdx.x == 0) { atomicAdd(&sums[2 * ng], r1[0]); atomicAdd(&sums[2 * ng + 1], r2[0]); }
}
__global__ void gn_finalize_k(const double *sums, int NG, size_t len, float eps, float *mean, float *rstd);
double *gn_dsums(size_t n);

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

extern "C" void nn_gn_fwd(const float *x, shape5 s, int G, float eps, const float *gamma, const float *beta, float *y, float *mean, float *rstd);
/* GroupNorm statistics of a tensor in activation storage (16-bit, fp32 or MX-fp8) */
extern "C" int nn_gn_stats(const float *x, shape5 s, int G, float eps, float *mean, float *rstd);
extern "C" void nn_gn_fwd_silu(const float *x, shape5 s, int G, float eps, const float *gamma, const float *beta, float *y, float *mean, float *rstd);
extern "C" void nn_gn_apply_silu(const float *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd, float *g, float *sil);

/* per (n,c): sum gy, sum gy*xhat -> dsums[2*(n*C+c)] (double, atomics over slabs), then copied to st (float) */
__global__ void gn_bwd_stats_k(const float *x, const float *gy, const float *mean, const float *rstd, int C, int G, size_t S, double *ds);
__global__ void d2f_k(const double *d, float *f, int n);

__global__ void gn_bwd_apply_k(const float *x, const float *gy, const float *gamma, const float *mean, const float *rstd, const float *st,
                               float *gx, int N, int C, int G, size_t S);

__global__ void gn_param_grad_k2(const float *st, float *ggamma, float *gbeta, int N, int C);
__global__ void gn_param_grad_k(const float *st, float *ggamma, float *gbeta, int N, int C);

extern "C" size_t nn_gn_scratch(shape5 s);
extern "C" void nn_gn_apply(const float *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd, float *y);

extern "C" void nn_gn_bwd(const float *x, shape5 s, int G, const float *gamma, const float *mean, const float *rstd, const float *gy,
                          float *gx, float *ggamma, float *gbeta, float *scratch);

/* ================= elementwise ================= */
__global__ void silu_f_k(const float *x, float *y, size_t n);
__global__ void silu_b_k(const float *x, const float *gy, float *gx, size_t n);
__global__ void axpy_k(float *y, float a, const float *x, size_t n);
__global__ void scale_k(float *y, float a, size_t n);
__global__ void u8f_k(const uint8_t *x, float s, float *y, size_t n);
__global__ void sigm_k(const float *x, float *y, size_t n);

extern "C" void nn_silu_fwd(const float *x, size_t n, float *y);
extern "C" void nn_silu_bwd(const float *x, const float *gy, size_t n, float *gx);
extern "C" void nn_axpy(float *y, float a, const float *x, size_t n);
extern "C" void nn_scale(float *y, float a, size_t n);
extern "C" void nn_u8_to_f32(const uint8_t *x, size_t n, float scale, float *y);
template <typename HT> __global__ void f2h_k(const float *x, HT *y, size_t n, float scale) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) y[i] = f2h<HT>(x[i] * scale); }
/* fp32 -> the current 16-bit storage type (bf16, or fp16 with nn_set_f16), optionally scaled */
extern "C" void nn_f32_to_h16(const float *x, size_t n, void *y, float scale);
extern "C" void nn_f32_to_bf16(const float *x, size_t n, void *y);
/* network input -> activation storage (16-bit, or MX-fp8 when y is registered MX) */
/* a 16-bit tensor (the storage type of nn_set_f16) into an MX-registered tensor */
extern "C" void nn_h16_to_mx(const void *x, shape5 s, void *y);
extern "C" void nn_f32_to_act(const float *x, shape5 s, void *y);
/* 2:4 structured sparsity along the input channels of a [co][ci][taps] weight: in every group of 4 consecutive ci (same co, tap)
   only the two largest |w| survive. mask24: out = masked w (out may alias w). srste24: g = g (kept) + lambda * w (pruned):
   the sparse-refined straight-through estimator (pruned weights get a decaying pull so they can be revisited). */
__global__ void mask24_k(const float *w, float *out, int Co, int Ci, int T);
__global__ void srste24_k(float *g, const float *w, int Co, int Ci, int T, float lambda);
/* Weights kept on an fp8 (e4m3) or fp4 (e2m1) grid with MX block scaling (one power-of-two scale per 32 input channels
   of a (co, tap)): the master array holds the dequantized values, so the forward's quantization is exact and the model
   is a true fp8/fp4 model. Stochastic rounding keeps the expected update unbiased (round-to-nearest would discard
   optimizer steps far below the grid spacing). */
__device__ __forceinline__ float grid_spacing(float a, int bits) {   /* a = |x| / scale in [0, qmax] */
    if (bits == 8) { if (a < 0.015625f) return 0.001953125f; int e; frexpf(a, &e); return ldexpf(1.f, e - 1 - 3); }   /* e4m3: 3 mantissa bits, subnormal step 2^-9 */
    else { if (a < 1.f) return 0.5f; int e; frexpf(a, &e); return ldexpf(1.f, e - 1 - 1); }                           /* e2m1: 1 mantissa bit, subnormal step 0.5 */
}
__device__ __forceinline__ unsigned hash32(unsigned x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }
__global__ void wquant_k(float *w, int Co, int Ci, int T, int bits, unsigned seed);
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
__global__ void wq_pack_k(const float *w, unsigned char *q, unsigned char *sc, int Co, int Ci, int T, int bits, unsigned seed);
__global__ void wq_unpack_k(const unsigned char *q, const unsigned char *sc, float *w, int Co, int Ci, int T, int bits);
/* AdamW directly on the packed weights: dequantize the block, update in fp32 (m, v stay fp32), rescale, requantize stochastically */
__global__ void wq_adamw_k(unsigned char *q, unsigned char *sc, unsigned char *r, unsigned char *rsc, const float *g, float *m, float *v, int Co, int Ci, int T, int bits,
                           float lr, float b1, float b2, float eps, float wd, float c1, float c2, unsigned seed);
/* EMA directly on packed weights: e = d e + (1 - d) p, both packed; result requantized stochastically */
__global__ void wq_ema_k(unsigned char *qe, unsigned char *sce, unsigned char *re, unsigned char *rsce, const unsigned char *qp, const unsigned char *scp, const unsigned char *rp, const unsigned char *rscp, int Co, int Ci, int T, int bits, float d, unsigned seed);
extern "C" size_t nn_wq_nblocks(int co, int ci, int taps);
extern "C" size_t nn_wq_bytes(int co, int ci, int taps, int bits);
__global__ void wq_residual_k(const float *w, const unsigned char *q, const unsigned char *sc, unsigned char *r, unsigned char *rsc, int Co, int Ci, int T, int bits);
extern "C" void nn_wq_pack(const float *w, void *q, void *sc, int co, int ci, int taps, int bits, unsigned seed);
extern "C" void nn_wq_residual(const float *w, const void *q, const void *sc, void *r, void *rsc, int co, int ci, int taps, int bits);
extern "C" void nn_wq_unpack(const void *q, const void *sc, float *w, int co, int ci, int taps, int bits);
/* r / rsc: optional fp8 error-feedback residual (fp4 weights); nullptr = stochastic rounding without residual */
extern "C" void nn_wq_adamw(void *q, void *sc, void *r, void *rsc, const float *g, float *m, float *v, int co, int ci, int taps, int bits, float lr, float b1, float b2, float eps, float wd, int step, unsigned seed);
extern "C" void nn_wq_ema(void *qe, void *sce, void *re, void *rsce, const void *qp, const void *scp, const void *rp, const void *rscp, int co, int ci, int taps, int bits, float decay, unsigned seed);
extern "C" void nn_wquant(float *w, int co, int ci, int taps, int bits, unsigned seed);
extern "C" void nn_mask24(const float *w, float *out, int co, int ci, int taps);
extern "C" void nn_srste24(float *g, const float *w, int co, int ci, int taps, float lambda);
extern "C" void nn_sigmoid(const float *x, size_t n, float *y);
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
extern "C" void nn_pred_input(const uint8_t *ct, int W, float mean, float isd, const float *dyo, const float *dxo, int axis, void *x, int h16);
__global__ void pred_out_k(const float *lg, const uint8_t *ct, size_t n, uint8_t *out);
extern "C" void nn_pred_output(const float *lg, const uint8_t *ct, size_t n, uint8_t *out);
/* the probability of the window's interior (halo .. W - halo) written straight into a device shard buffer (shard^3, x fastest):
   window voxel (z, y, x) lands at shard voxel (oz + z, oy + y, ox + x) when inside [0, e) (o = window origin - shard origin) */
__global__ void pred_place_k(const float *lg, const uint8_t *ct, int W, int halo, int oz, int oy, int ox, int ez, int ey, int ex, int shard, uint8_t *dsh);
extern "C" void nn_pred_place(const float *lg, const uint8_t *ct, int W, int halo, int oz, int oy, int ox, int ez, int ey, int ex, int shard, uint8_t *dsh);
/* exact window statistics (nonzero count, sum, sum of squares) of a uint8 window: integer sums, as the host loop computed */
__global__ void pred_stats_k(const uint8_t *ct, size_t n, unsigned long long *acc);
extern "C" void nn_pred_stats(const uint8_t *ct, size_t n, void *scratch, size_t *nz, double *sum, double *sq);

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
extern "C" void nn_up2_fwd_gn_into(const float *x, shape5 xs, const nn_gn_t *g, float *y, int ctot, int c0);
extern "C" void nn_up2_fwd_into(const float *x, shape5 xs, float *y, int ctot, int c0);
extern "C" void nn_up2_fwd(const float *x, shape5 xs, float *y);
extern "C" void nn_up2_bwd_into(const float *gy, shape5 xs, float *gx, int ctot, int c0);
extern "C" void nn_up2_bwd(const float *gy, shape5 xs, float *gx);

/* ================= concat ================= */
__global__ void concat_k(const float *a, int ca, const float *b, int cb, float *y, int N, size_t S, int fwd);
extern "C" void nn_concat_fwd(const float *a, int ca, const float *b, int cb, shape5 s, float *y);
extern "C" void nn_concat_bwd(const float *gy, int ca, int cb, shape5 s, float *ga, float *gb);

/* ================= loss =================
   scratch layout per (n,c): [nmask, bce_sum, sum_sig_p, sum_sig, sum_p] (5 floats). */
extern float g_posw;
extern "C" void nn_set_pos_weight(float w);
/* Offset-tolerant positives (nn_set_loss_tol r > 0, channel 0 only). For a supervised voxel with target p >= 0.5 the
   positive BCE term p * softplus(-x) uses the maximum logit over the supervised voxels within r steps along the local
   sheet normal, so a surface predicted up to r voxels off a misregistered label still earns full credit. The normal is
   one of 13 lattice directions: the one along which the target mass of the 5^3 neighbourhood is most compact (smallest
   second moment about its centroid; a sheet spreads in-plane). Tolerance along the normal only, so the prediction must
   still cover every in-plane position (a sparse lattice of dots earns nothing: a full-cube tolerance allowed that).
   The negative term (1 - p) * softplus(x) and both dice sums stay on each voxel's own logit (dice stays in [0, 1]).
   code[i] = 5 * dir + (k + r) for the voxel at i + k * dir that carries i's positive term, 255 = i itself; the gradient
   kernel gathers, for each voxel, the target mass of the voxels whose line maximum it is. */
extern int g_tol;
extern "C" void nn_set_loss_tol(int r);
extern "C" int nn_get_loss_tol(void);
__device__ __forceinline__ void tol_dir(int d, int &dz, int &dy, int &dx) {   /* the 13 lattice directions (one of each +- pair) */
    const int t[13][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}, {1, -1, 0}, {1, 0, 1}, {1, 0, -1}, {0, 1, 1}, {0, 1, -1},
                          {1, 1, 1}, {1, 1, -1}, {1, -1, 1}, {1, -1, -1}};
    dz = t[d][0]; dy = t[d][1]; dx = t[d][2];
}
/* lattice direction of the sheet normal at voxel (z, y, x) from the target t (uint8) of the 5^3 neighbourhood */
__device__ __forceinline__ int tol_normal(const uint8_t *tp, int z, int y, int x, int D, int H, int W) {
    float m0 = 0, mz = 0, my = 0, mx = 0, zz = 0, yy = 0, xx = 0, zy = 0, zx = 0, yx = 0;
    for (int a = -2; a <= 2; a++) { const int za = z + a; if (za < 0 || za >= D) continue;
        for (int b = -2; b <= 2; b++) { const int yb = y + b; if (yb < 0 || yb >= H) continue;
            for (int c = -2; c <= 2; c++) { const int xc = x + c; if (xc < 0 || xc >= W) continue;
                const float v = tp[((size_t)za * H + yb) * W + xc];
                if (v == 0.f) continue;
                m0 += v; mz += v * a; my += v * b; mx += v * c;
                zz += v * a * a; yy += v * b * b; xx += v * c * c; zy += v * a * b; zx += v * a * c; yx += v * b * c; } } }
    if (m0 <= 0.f) return 0;
    const float iz = mz / m0, iy = my / m0, ix = mx / m0;   /* covariance about the centroid */
    const float czz = zz / m0 - iz * iz, cyy = yy / m0 - iy * iy, cxx = xx / m0 - ix * ix, czy = zy / m0 - iz * iy, czx = zx / m0 - iz * ix, cyx = yx / m0 - iy * ix;
    int best = 0; float bq = 3.4e38f;
    for (int d = 0; d < 13; d++) {
        int dz, dy, dx; tol_dir(d, dz, dy, dx);
        const float q = (czz * dz * dz + cyy * dy * dy + cxx * dx * dx + 2.f * (czy * dz * dy + czx * dz * dx + cyx * dy * dx)) / (float)(dz * dz + dy * dy + dx * dx);
        if (q < bq) { bq = q; best = d; }
    }
    return best;
}
/* logits element as float: fp32 or fp16 storage (training logits in the gradient buffer, nn_set_logits_h16) */
template <typename LT> __device__ __forceinline__ float lgv(const LT *p, size_t i);
template <> __device__ __forceinline__ float lgv<float>(const float *p, size_t i) { return p[i]; }
template <> __device__ __forceinline__ float lgv<f16>(const f16 *p, size_t i) { return __half2float(p[i]); }
extern int g_logits_h16;
extern int g_head_h16;   /* the MX head (1^3 conv) writes fp16 logits (nn_set_head_out_h16) */
/* maximum logit over the supervised voxels i + k * dir, |k| <= r; returns the logit and sets the code */
template <typename LT>
__device__ __forceinline__ float tol_line_max(const LT *l, const uint8_t *mp, int z, int y, int x, int D, int H, int W, int d, int r, uint8_t *code) {
    int dz, dy, dx; tol_dir(d, dz, dy, dx);
    float best = lgv(l, ((size_t)z * H + y) * W + x); int bk = 0;
    for (int k = -r; k <= r; k++) {
        if (!k) continue;
        const int zz = z + k * dz, yy = y + k * dy, xx = x + k * dx;
        if (zz < 0 || zz >= D || yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
        const size_t j = ((size_t)zz * H + yy) * W + xx;
        if (mp[j] && lgv(l, j) > best) { best = lgv(l, j); bk = k; }
    }
    *code = bk ? (uint8_t)(5 * d + bk + r) : (uint8_t)255;
    return best;
}
template <typename LT> __global__ void loss_stats_k(const LT *lg, const uint8_t *t, const uint8_t *m, const uint8_t *w, int C, size_t S, double *ds, float pw,
                                                   int tol, uint8_t *code, int D, int H, int W);
__global__ void loss_d2f_k(const double *d, float *f, int n);
/* finalize on device: per-channel mean bce / dice, active count, and 1/active for the gradient kernel.
   layout of fin[]: [0..C) bce, [C..2C) dice, [2C] active, [2C+1] inv_active */
__global__ void loss_fin_k(const float *st, const uint8_t *w, int N, int C, float *fin);
template <typename GT, typename LT = float> __global__ void loss_grad_k(const LT *lg, const uint8_t *t, const uint8_t *m, const uint8_t *w, int N, int C, size_t S, const float *st,
                            float dice_w, const float *fin, GT *gl, float pw, float gscale, int tol, const uint8_t *code, int D, int H, int W) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * C * S) return;
    int nc = (int)(i / S), n = nc / C;
    if (!w[nc] || !m[(size_t)n * S + i % S]) { gl[i] = f2h<GT>(0.f); return; }
    float nm = st[nc * 5], Ssp = st[nc * 5 + 2], Ss = st[nc * 5 + 3], Sp = st[nc * 5 + 4];
    float x = lgv(lg, i), p = t[i] * (1.f / 255.f), s = 1.f / (1.f + expf(-x));
    float den = Ss + Sp + 1.f;
    float pq = p;   /* the target mass whose positive BCE terms this voxel's logit carries */
    if (code && nc % C == 0) {   /* offset-tolerant: gather p of the voxels whose normal-line maximum is this voxel */
        const size_t v = i % S; const uint8_t *cp = code + (size_t)n * S, *tp = t + (size_t)nc * S;
        const int z = (int)(v / ((size_t)H * W)), y = (int)((v / W) % H), x0 = (int)(v % W);
        if (cp[v] != 255) pq = 0.f;   /* this voxel's own positive term moved to another voxel of its line */
        for (int d = 0; d < 13; d++) {
            int dz, dy, dx; tol_dir(d, dz, dy, dx);
            for (int k = -tol; k <= tol; k++) {
                if (!k) continue;
                const int zz = z - k * dz, yy = y - k * dy, xx = x0 - k * dx;   /* voxel j with j + k * dir = this voxel */
                if (zz < 0 || zz >= D || yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
                const size_t j = ((size_t)zz * H + yy) * W + xx;
                if (cp[j] == (uint8_t)(5 * d + k + tol)) pq += tp[j] * (1.f / 255.f);
            }
        }
    }
    float g = ((1.f - p) * s - pw * pq * (1.f - s)) / fmaxf(nm, 1.f);
    float ddice_ds = -(2.f * p * den - (2.f * Ssp + 1.f)) / (den * den);   /* dice on the voxel's own value */
    g += dice_w * ddice_ds * s * (1.f - s);
    gl[i] = f2h<GT>(g * fin[2 * C + 1] * gscale);
}
extern int g_loss_g16;
extern "C" void nn_set_loss_grad_h16(int on);
/* scratch: 5 floats per (n,c) statistics followed by 2C+2 finalized values */
extern "C" size_t nn_loss_scratch(shape5 s);
/* Asynchronous: launches the statistics, finalize and gradient kernels; nothing is copied to the host. */
extern "C" void nn_loss_async(const float *logits, const uint8_t *t, const uint8_t *m, const uint8_t *w, shape5 s, float dice_w, float *gl, float *scratch);
extern "C" void nn_loss_async_tol(const float *logits, const uint8_t *t, const uint8_t *m, const uint8_t *w, shape5 s, float dice_w, float *gl, float *scratch, uint8_t *code);
/* Copies the finalized values (2C+1 floats: bce per channel, dice per channel, active) to the host (synchronous). */
extern "C" void nn_loss_fetch(const float *scratch, shape5 s, float *out);
extern "C" void nn_loss(const float *logits, const uint8_t *t, const uint8_t *m, const uint8_t *w, shape5 s, float dice_w,
                        float *gl, float *out, float *scratch);

__global__ void sheet_gather_k(const float *lg,shape5 s,const float *xyz,size_t np,int z0,int lo,int hi,double *out);
extern "C" void nn_sheet_gather(const float *lg,shape5 s,const float *coords,size_t np,int z0,int lo,int hi,double *values);
template<typename T> __global__ void sheet_scatter_k(T *gl,const uint64_t *idx,const float *v,size_t n,float scale) {
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
    if (i<n) gl[idx[i]]=f2h<T>(h2f(gl[idx[i]])+v[i]*scale);
}
extern "C" void nn_sheet_scatter(float *gl,const uint64_t *indices,const float *values,size_t n,int h16);

template<typename T> __global__ void sheet_input_k(T *input,int W,const float *rows,float center,float scale) {
    size_t k=blockIdx.x*(size_t)blockDim.x+threadIdx.x,S=(size_t)W*W*W; if (k>=S) return;
    int z=k/((size_t)W*W),y=(k/W)%W,x=k%W;
    const float *r=rows+4*z;
    input[S+k]=f2h<T>((hypotf(y+r[0],x+r[1])*r[2]+r[3]-center)/scale);
}
extern "C" void nn_sheet_input(void *input,int W,const float *rows,float center,float scale,int h16);
__global__ void sheet_gate_k(float *r,const float *surface,const uint8_t *ct,int W,const float *rows);
extern "C" void nn_sheet_gate(float *r,const float *surface,const uint8_t *ct,int W,const float *rows);

/* ---- multi-GPU: copy between devices (peer access when available, else staged through the host) ---- */
extern "C" void nn_peer_copy(void *dst, int dst_dev, const void *src, int src_dev, size_t bytes);

/* ---- spatial split: halo exchange and sums across the two GPUs (called by one host thread for both sides) ----
   A tensor is one or two plane-major segments (rows of D planes): esz > 0 = [rows = n c][D][H][W] elements of esz bytes; esz 0 =
   the MX tensor registered at p: data [n][blk][voxel][rb] (rows n nb of D planes of H W rb bytes) + the scale plane (rows of H W
   bytes). Side 0 owns the low z planes (its halo is at the high end), side 1 the high ones. */
typedef struct { size_t base, rows, pb, pitch; } zseg_t;
int zsegs(const void *p, shape5 s, int esz, zseg_t *sg);
size_t zsegs_plane_bytes(const zseg_t *sg, int ns);
/* planes [z0, z0 + nz) of every segment of t <-> contiguous buffer c (dir 0: gather into c, 1: scatter from c) */
void zsegs_copy(void *t, const zseg_t *sg, int ns, int z0, int nz, void *c, int dir);
void zsegs_zero(void *t, const zseg_t *sg, int ns, int z0, int nz);
/* zero the halo planes (lo at the low end, hi at the high end) of a tensor on the current device */
extern "C" void nn_split_zero(const void *t, shape5 s, int esz, int lo, int hi);
cudaEvent_t zs_ev(int dev, int k);
/* both default streams wait for each other (event k) */
void zs_xbar(const int *dev, int k);
extern "C" size_t nn_split_halo_bytes(shape5 s, int esz);
/* halo exchange of tensor t[i] on device dev[i] (same shape on both, h halo planes): the innermost halo plane receives the
   other side's boundary plane, the outer h - 1 halo planes are zeroed. sb / rb: per-side send / receive buffers of at least
   nn_split_halo_bytes. Asynchronous in two parts: begin (both sides at once) packs each side's boundary plane and zeroes its
   outer halo planes on the compute stream, then copies it to the other side's receive buffer on a per-device communication
   stream, so the compute stream runs on (a weight gradient that does not read the halo) while the plane crosses PCIe; end
   (each side, its own thread) makes the compute stream wait for the arrival and unpacks. One exchange in flight per slot
   (0 / 1: the caller's own buffers and events). */
cudaStream_t zs_comm(int dev);
cudaEvent_t zs_hev(int dev, int slot, int k);
extern "C" void nn_split_halo_begin(void *const *t, const int *dev, shape5 s, int esz, int h, void *const *sb, void *const *rb, int slot);
extern "C" void nn_split_halo_end(void *t, int side, shape5 s, int esz, int h, void *rb, int slot);
__global__ void zs_add_k(double *a, const double *b, int n);
/* b[i] (n doubles on dev[i]) = b[0] + b[1] on both devices; r[i]: n-double receive buffers */
extern "C" void nn_split_allreduce(double *const *b, const int *dev, int n, double *const *r);

/* ================= optimizer / reductions ================= */
__global__ void adamw_k(float *p, const float *g, float *m, float *v, size_t n, float lr, float b1, float b2, float eps, float wd, float c1, float c2);
/* ---- Muon (modded-nanogpt): momentum, then Newton-Schulz orthogonalisation of the Co x K gradient matrix ---- */
__global__ void mm_xxt_k(const float *X, int Co, int K, float *A);
__global__ void mm_sq_k(const float *A, int Co, float b, float c, float *B);
__global__ void mm_bx_k(const float *B, const float *X, int Co, int K, float a, float *Y);
__global__ void muon_mom_k(const float *g, float *mom, float *x, size_t n, float beta);
__global__ void muon_sumsq_k(const float *x, size_t n, double *ss);
__global__ void muon_scale_k(float *x, size_t n, const double *ss, float eps);
__global__ void muon_apply_k(float *p, const float *o, size_t n, float lr, float scale, float wd);
/* work: >= 2 Co K + 2 Co Co floats; the gradient of a [Co][K] weight (row-major) is orthogonalised with 5 Newton-Schulz
   iterations (coefficients from modded-nanogpt) and applied with lr * sqrt(max(1, Co / K)). mom is the momentum buffer. */
extern "C" void nn_muon(float *p, const float *g, float *mom, int Co, int K, float lr, float beta, float wd, float *work);
/* batched register-tiled fp32 products of the orthogonalisation cascades (Muon, ANVIL), blockIdx.z = conv, any Co:
   MODE 0 (xxt): A = X X^T          (M = N = Co, contraction K)
   MODE 1 (sq):  B = b A + c A A    (M = N = contraction = Co)
   MODE 2 (bx):  Y = b X + B X      (M = Co, N = K, contraction Co; Y is the other buffer)
   64 x 64 output tile per block (grid x over N, y over M), 256 threads with 4 x 4 outputs each, the contraction staged 16
   at a time. Every output sums its products in increasing contraction order in one accumulator, as the scalar kernels did. */
template <int MODE, typename DT> __global__ void __launch_bounds__(256) gemm_ns_k(const DT *d, int swap, float b, float c) {
    const DT D = d[blockIdx.z];
    const int Co = D.Co, K = D.K, M = Co, N = MODE == 2 ? K : Co, KC = MODE == 0 ? K : Co;
    const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    if (m0 >= M || n0 >= N) return;
    const float *X = swap ? D.Y : D.X;
    const float *Lm = MODE == 0 ? X : MODE == 1 ? D.A : D.B;   /* L[m][k], row stride KC */
    const float *Rm = MODE == 1 ? D.A : X;                     /* R[k][n]: X[n][k] (MODE 0), A[k][n], X[k][n] */
    __shared__ __align__(16) float sl[16][68], sr[16][68];     /* [k][m], [k][n] */
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < KC; k0 += 16) {
        for (int t = threadIdx.x; t < 1024; t += 256) {
            const int r = t >> 4, kk = t & 15, m = m0 + r, k = k0 + kk;
            sl[kk][r] = m < M && k < KC ? Lm[(size_t)m * KC + k] : 0.f;
            if (MODE == 0) { const int n = n0 + r; sr[kk][r] = n < N && k < KC ? Rm[(size_t)n * K + k] : 0.f; }
            else { const int k2 = k0 + (t >> 6), n = n0 + (t & 63); sr[t >> 6][t & 63] = n < N && k2 < KC ? Rm[(size_t)k2 * N + n] : 0.f; }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 16; kk++) {
            const float4 l = *(const float4 *)&sl[kk][ty * 4], r = *(const float4 *)&sr[kk][tx * 4];
            const float lv[4] = {l.x, l.y, l.z, l.w}, rv[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] += lv[i] * rv[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int m = m0 + ty * 4 + i;
        if (m >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int n = n0 + tx * 4 + j;
            if (n >= N) continue;
            if (MODE == 0) D.A[(size_t)m * Co + n] = acc[i][j];
            else if (MODE == 1) D.B[(size_t)m * Co + n] = b * D.A[(size_t)m * Co + n] + c * acc[i][j];
            else (swap ? D.X : D.Y)[(size_t)m * K + n] = b * X[(size_t)m * K + n] + acc[i][j];
        }
    }
}
/* batched Muon: one launch per stage for all convs (blockIdx.z = conv). descs live on the device. */
typedef struct { float *p; const float *g; float *mom, *X, *Y, *A, *B; int Co, K; } muon_desc_t;
__global__ void bmuon_mom_k(const muon_desc_t *d, float beta, double *ss);
__global__ void bmuon_scale_k(const muon_desc_t *d, const double *ss);
__global__ void bmuon_apply_k(const muon_desc_t *d, int swap, float lr, float wd);
/* descs: device array of nconv descriptors (p, g, mom, X, Y, A, B scratch of Co K, Co K, Co Co, Co Co floats, Co, K);
   maxco / maxk: the largest Co and K among them. 18 launches for all convs. */
extern "C" void nn_muon_batch(const void *descs, int nconv, int maxco, int maxk, float lr, float beta, float wd);
/* ---- ANVIL II (modded-nanogpt record #90, hyperstition.cc): twin-rail Nesterov velocity, 1.05 Frobenius normalisation,
   six quintic spectral maps, per-row energy equalisation at constant norm (NorMuon), sign-aligned weight decay.
   Descriptor layout as Muon plus: v1 (slow rail, uses the Muon mom field as v0), E (Co floats lane energy), R (Co floats scratch). */
typedef struct { float *p; const float *g; float *v0, *X, *Y, *A, *B, *v1, *E, *R; int Co, K; } anvil_desc_t;
__global__ void anvil_mom_k(const anvil_desc_t *d, float bf, float bs, float w, float mu, double *ss);
__global__ void anvil_scale_k(const anvil_desc_t *d, const double *ss);
__global__ void anvil_sq_k(const anvil_desc_t *d, int it);
/* lane (row) power of the cascade output, one block per row */
__global__ void anvil_rowpow_k(const anvil_desc_t *d, int swap);
/* per conv: lane energy EMA, gain = 1/sqrt(E), global rescale to the pre-equalisation Frobenius norm; R <- row scale */
__global__ void anvil_eq_k(const anvil_desc_t *d, float b2);
__global__ void anvil_apply_k(const anvil_desc_t *d, int swap, float lr, float wd);
extern "C" void nn_anvil_batch(const void *descs, int nconv, int maxco, int maxk, float lr, float beta_fast, float beta_slow, float w_fast, float mu, float beta2, float wd);
extern "C" void nn_adamw(float *p, const float *g, float *m, float *v, size_t n, float lr, float b1, float b2, float eps, float wd, int step);
__global__ void ema_k(float *e, const float *p, size_t n, float d);
extern "C" void nn_ema(float *ema, const float *p, size_t n, float decay);

__global__ void sum_k(const float *x, size_t n, float *out, int sq);
double reduce(const float *x, size_t n, float *scratch, int sq);
extern "C" double nn_sum(const float *x, size_t n, float *scratch);
extern "C" double nn_sumsq(const float *x, size_t n, float *scratch);

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
/* affine variant: quantise (x - mean) * rstd (per n, group of G channels), then restore. mean/rstd: [N][G]. */
template <typename T>
__global__ void fq_aff_k(T *x, int N, int C, size_t S, int fmt, const float *mean, const float *rstd, int G) {
    const int gs = fmt == 1 ? 16 : 32, ng = (C + gs - 1) / gs, cpg = C / G;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * ng * S) return;
    size_t v = i % S; int gi = (int)((i / S) % ng), n = (int)(i / (S * ng));
    T *p = x + ((size_t)n * C + gi * gs) * S + v;
    const int cn = min(gs, C - gi * gs);
    float z[32], amax = 0.f;
    for (int k = 0; k < cn; k++) { int g = (gi * gs + k) / cpg; z[k] = (ldv(p, (size_t)k * S) - mean[n * G + g]) * rstd[n * G + g]; amax = fmaxf(amax, fabsf(z[k])); }
    if (amax == 0.f) return;
    const int mb = fmt <= 2 ? 1 : fmt == 3 ? 3 : fmt == 4 ? 2 : 3, emin = fmt <= 3 ? 0 : fmt == 4 ? -2 : -6;
    const float qmax = fmt <= 2 ? 6.f : fmt == 3 ? 7.5f : fmt == 4 ? 28.f : 448.f;
    int e; frexpf(amax / qmax, &e);
    float sc = ldexpf(1.f, (amax / qmax) == ldexpf(1.f, e - 1) ? e - 1 : e);
    for (int k = 0; k < cn; k++) { int g = (gi * gs + k) / cpg; float q = fq_round(z[k] / sc, mb, emin, qmax) * sc; stv(p, (size_t)k * S, q / rstd[n * G + g] + mean[n * G + g]); }
}
template <typename T>
__global__ void fq_k(T *x, int N, int C, size_t S, int fmt, const unsigned *am);
extern "C" void nn_fake_quant_affine(void *x, shape5 s, int fmt, const float *mean, const float *rstd, int G);
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
extern "C" void nn_fake_quant(void *x, shape5 s, int fmt);

