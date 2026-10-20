/* xnet: a small fp16 / tensor-core layer set for running external PyTorch inference networks (ufsm ink-ext: v8in, Hecate,
   Reader V3) on the GPU. Tensors are fp16, channels-last (N, D, H, W, C); 2D maps are D = 1.

   Convolutions are one implicit-GEMM kernel (mma.sync m16n8k16 f16 x f16 -> f32, cp.async multistage) for any kernel size,
   stride and zero padding, with a per-output-channel fp32 epilogue  y = act(acc * scale[c] + bias[c] (+ res))  so BatchNorm is
   folded at load time without touching the fp16 weights. Inputs with one channel (the stems) are unfolded along W into 8
   channels first (xunfold), so every GEMM loads 16-byte vectors (Ci % 8 == 0, Co % 8 == 0).

   Norms (GroupNorm / InstanceNorm = GroupNorm with C groups) take two-pass statistics over D x H x W per sample and group
   in double precision, and apply  act(norm(x) * gamma + beta (+ res)) . Resizes follow PyTorch interpolate(align_corners=False)
   for trilinear / bilinear (an axis of equal size is the identity). Device memory comes from a size-keyed cache so repeated
   forwards do not call cudaMalloc. All calls run on one CUDA stream (xnet_stream). */
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { void *p; int n, d, h, w, c; } xt;            /* fp16 NDHWC; p == nullptr: empty */
static inline size_t xt_numel(xt t) { return (size_t)t.n * t.d * t.h * t.w * t.c; }

int xnet_init(int gpu);                 /* select the device, create the stream; 0 on success */
void *xnet_stream(void);                /* the cudaStream_t of every xnet call */
void xnet_sync(void);
int xnet_check(const char *where);      /* 0, or prints the pending CUDA error and returns -1 */
size_t xnet_mem_cached(void);           /* bytes held by the cache (free blocks) */
size_t xnet_mem_used(void);             /* bytes handed out */
void xnet_trim(void);                   /* free the cached blocks */
void xnet_mem_info(size_t *free_b, size_t *total_b);   /* device memory free / total (cudaMemGetInfo) */

void *xmalloc_dev(size_t bytes);        /* cached device memory (nullptr on failure) */
void xfree_dev(void *p);
xt xt_new(int n, int d, int h, int w, int c);
void xt_free(xt *t);
void xh2d(void *dst, const void *src, size_t bytes);   /* synchronous copies on the stream */
void xd2h(void *dst, const void *src, size_t bytes);

enum { XA_NONE = 0, XA_RELU = 1, XA_LRELU = 2 };       /* LeakyReLU slope 0.01 */

typedef struct {
    int ci, co, kd, kh, kw, sd, sh, sw, pd, ph, pw;      /* the GEMM conv (after unfolding) */
    int unfold, kw0, pw0;                                /* ci == 1 source unfolded along W: kw0 taps, pad pw0 -> 8 channels, kw = 1 */
    int K, Kp, Cop;                                      /* K = kd kh kw ci, padded to the K / N tiles */
    void *w; float *scale, *bias;                        /* device: fp16 [Cop][Kp]; fp32 [co] */
    int tconv;                                           /* ConvTranspose k2 s2: GEMM to 4 co channels, shuffled */
    float *tbias;                                        /* the transposed conv's bias [co] */
} xconv;

/* w: PyTorch Conv weight [co][ci][kd][kh][kw] (fp32 host); scale / bias: per output channel (nullptr: 1 / 0). 0 on success. */
int xconv_init(xconv *c, const float *w, int co, int ci, int kd, int kh, int kw, int sd, int sh, int sw, int pd, int ph, int pw,
               const float *scale, const float *bias);
/* ConvTranspose2d(ci, co, 2, stride 2) of a 2D map, w [ci][co][2][2], bias [co] or nullptr */
int xtconv_init(xconv *c, const float *w, int ci, int co, const float *bias);
void xconv_free(xconv *c);
/* y = act(conv(x) * scale + bias (+ res)); allocates y. res (optional) has y's shape. */
xt xconv_fwd(const xconv *c, xt x, const xt *res, int act);
/* fold BatchNorm (gamma, beta, mean, var, eps) into per-channel scale / bias, plus a conv bias cb (nullptr: none) */
void xbn_fold(int co, const float *g, const float *b, const float *m, const float *v, float eps, const float *cb, float *scale, float *bias);

/* GroupNorm(groups) over D H W per sample, y = act(norm(x) gamma + beta (+ res)); y may be x (in place) */
void xgn(xt x, int groups, const float *gamma_dev, const float *beta_dev, float eps, const xt *res, int act, xt y);
float *xdev_floats(const float *host, size_t n);       /* a cached device copy of host floats (free with xfree_dev) */

xt xunfold(xt x, int kw, int pw);                      /* c = 1 -> c = 8: out[.., w, j] = x[.., w - pw + j] (j < kw), width W + 2pw - kw + 1 */
xt xmaxpool2d_k3s2(xt x);                              /* per depth plane, kernel 3, stride 2, pad 1 (-inf) */
xt xavgpool2d_k2(xt x);                                /* per depth plane, kernel 2, stride 2 */
xt xdepthmax(xt x);                                    /* max over D -> D = 1 */
xt xresize(xt x, int d, int h, int w);                 /* trilinear / bilinear, align_corners False */
xt xconcat(xt a, xt b);                                /* channels: a then b */
/* inputs: uint8 [n][d][h][w] on the device -> fp16 c = 1 with per-sample  y = (clamp(x, lo, hi)) * a + b  (host arrays [n]) */
xt xinput_u8(const uint8_t *dev, int n, int d, int h, int w, const float *lo, const float *hi, const float *a, const float *b);
/* 1x1 conv to one channel in fp32: out[voxel] = sum_c x[voxel][c] w[c] + bias (w on the device) */
void xhead(xt x, const float *w_dev, float bias, float *out_dev);
/* Reader V3 depth fusion: f [n][d][h][w][c], a [n][d][h][w] fp32 attention logits ->
   [n][1][h][w][2c] = (sum_d softmax_d(a) f, max_d f) */
xt xdepth_attn_max(xt f, const float *a_dev);
/* Hecate depth collapse: s, l [n][d][h][w] fp32 (attention scores, logits), sup [n][h][w] uint8 (column has data),
   z-scale zs: out[n][h][w] = sum_d w_d l_d with w = masked softmax_d(s + zs (d - D/2) / (D/2)) renormalised */
void xhecate_collapse(const float *s_dev, const float *l_dev, const uint8_t *sup_dev, int n, int d, int h, int w, float zs, float *out_dev);

#ifdef __cplusplus
}
#endif
