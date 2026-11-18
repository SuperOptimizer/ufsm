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

typedef struct { void *p; int n, d, h, w, c; void *q; } xt;   /* fp16 NDHWC; p == nullptr: empty. q: its MX-fp4 copy (UFSM_XNET_MX; conv_mx layout) or nullptr */
static inline size_t xt_numel(xt t) { return (size_t)t.n * t.d * t.h * t.w * t.c; }

int xnet_init(int gpu);                 /* select the device, create the stream; 0 on success */
void xnet_bind(void);                   /* make the xnet_init device current on the calling thread (worker threads) */
void *xnet_stream(void);                /* the cudaStream_t of every xnet call */
void xnet_sync(void);
int xnet_check(const char *where);      /* 0, or prints the pending CUDA error and returns -1 */
void xnet_seq_reset(void);           /* a new forward: conv sequence numbers restart at 0 (UFSM_XNET_FQ_MASK) */
double xnet_macs(void);                 /* multiply-adds issued by the convolutions so far (UFSM_XNET_VERBOSE=1: each conv's shape) */
size_t xnet_mem_cached(void);           /* bytes held by the cache (free blocks) */
size_t xnet_mem_used(void);             /* bytes handed out */
void xnet_trim(void);                   /* free the cached blocks */
void xnet_set_budget(size_t bytes);     /* device bytes (in use + cached) before the cache is dropped; default: free at init - 1 GB */
void xnet_mem_info(size_t *free_b, size_t *total_b);   /* device memory free / total (cudaMemGetInfo) */

void *xmalloc_dev(size_t bytes);        /* cached device memory (nullptr on failure) */
void xfree_dev(void *p);
xt xt_new(int n, int d, int h, int w, int c);
void xt_free(xt *t);
void xh2d(void *dst, const void *src, size_t bytes);   /* synchronous copies on the stream */
void xd2h(void *dst, const void *src, size_t bytes);

enum { XA_NONE = 0, XA_RELU = 1, XA_LRELU = 2, XA_GELU = 3, XA_SILU = 4,   /* LeakyReLU slope 0.01; GELU exact (erf) */
       XA_POSTRES = 16,                                       /* flag: y = act(acc scale + bias) + res (default: act(.. + res)) */
       XA_GLU = 32,                                           /* flag: output co / 2 channels silu(v[2j]) v[2j + 1] (no res) */
       XA_LN = 64 };                                          /* flag (xconv_fwd_ln): channel LayerNorm before res / act, co 32 or 64 */

typedef struct {
    int ci, co, kd, kh, kw, sd, sh, sw, pd, ph, pw;      /* the GEMM conv (after unfolding) */
    int unfold, kw0, pw0;                                /* ci == 1 source unfolded along W: kw0 taps, pad pw0 -> 8 channels, kw = 1 */
    int K, Kp, Cop;                                      /* K = kd kh kw ci, padded to the K / N tiles */
    void *w; float *scale, *bias;                        /* device: fp16 [Cop][Kp]; fp32 [co] */
    int tconv;                                           /* ConvTranspose k2 s2: GEMM to 4 (2D; tconv 1) or 8 (3D; tconv 2) co channels, shuffled */
    float *tbias;                                        /* the transposed conv's bias [co] */
    int mxrot;                                           /* bit 0 UFSM_XNET_MXROT: the packed weights are H16-rotated per 16-K block; bit 1 UFSM_XNET_MXA: least-
                                                            squares activation scales; either: the input is re-quantised by k_xq */
    void *mxq, *mxs, *mxs2; float *mxgs; int mxsf, nch;         /* UFSM_XNET_MX (ci % 32 == 0): packed fp4 weights [Cop][Kp / 2], scales [Cop][Kp / 32][4] (two 16-K
                                                            blocks per 32-K chunk, bytes 2..3 unused), the per-row fp32 scale (ue4m3 format), the scale format (6 ue8m0,
                                                            5 ue4m3), nch = 32-K chunks of K (the GEMM runs on the MX-fp4 tensor cores, inputs quantised on the fly) */
} xconv;

/* w: PyTorch Conv weight [co][ci][kd][kh][kw] (fp32 host); scale / bias: per output channel (nullptr: 1 / 0). 0 on success. */
int xconv_init(xconv *c, const float *w, int co, int ci, int kd, int kh, int kw, int sd, int sh, int sw, int pd, int ph, int pw,
               const float *scale, const float *bias);
/* ConvTranspose2d(ci, co, 2, stride 2) of a 2D map, w [ci][co][2][2], bias [co] or nullptr */
int xtconv_init(xconv *c, const float *w, int ci, int co, const float *bias);
/* ConvTranspose3d(ci, co, 2, stride 2), w [ci][co][2][2][2], bias [co] or nullptr */
int xtconv3_init(xconv *c, const float *w, int ci, int co, const float *bias);
void xconv_free(xconv *c);
void xnet_set_mx(int sf);
int xnet_mx(void);                      /* the MX format of convs initialised now (0: fp16) */
void xnet_set_mxq(int mode);            /* the MX weight quantiser (UFSM_XNET_MXQ): 0 amax / 6 block scales, 1 least-squares block scales, 2 + row scale */
void xnet_set_mx_skip(const char *list);   /* convs kept fp16 by their initialisation index ("3,7-9"; UFSM_XNET_MX_SKIP) */
void xnet_set_mxrot(int on);            /* UFSM_XNET_MXROT: H16-rotate the 16-blocks of MX convs initialised from now (weights at pack, inputs in k_xq) */
void xnet_set_mxa(int on);              /* UFSM_XNET_MXA: least-squares activation block scales for MX convs initialised from now */
void xnet_set_df16(int v);              /* the 16 -> 16 3x3x3 fp16 conv: 0 implicit GEMM, 1 the halo kernel k_dfc16, 2..9 its other tilings, 10 (default) the depth-streaming k_dfc16s, 11..16 its tilings (UFSM_XNET_DF16) */
void xnet_wseq_reset(void);             /* a new net: initialisation indices restart at 0 (UFSM_XNET_MX_SKIP, UFSM_XNET_FQ weight mask) */
/* the fused MX inference path (ink3d-ext): a conv that also returns the InstanceNorm sums of its output (st: device doubles [n][co][2]:
   sum, sum of squares; free with xfree_dev), the per-(sample, channel) affine of IN (coef [n][C][2]: a = gamma rstd, b = beta - mean a),
   and the apply y = act(a x + b (+ a_r r + b_r | + r)) writing y over h's fp16 (want_half) and / or its MX copy (want_q). */
xt xconv_fwd_st(const xconv *c, xt x, double **st);
float *xin_coef(const double *st, int n, int C, long long S, const float *gamma, const float *beta, float eps);
void xin_apply(xt *h, const float *coef, const xt *res, const float *rcoef, int act, int want_half, int want_q);
int xq_ensure(xt *x);
/* the fused one-channel stem: act(IN(conv3x3x3_{1->32}(x) + b)) (+ IN(1x1 conv wsk of x) when wsk_dev: gambiarra's first block without its
   second conv's apply), never storing the conv output: two passes (sums, then apply). w_dev [32][27] fp32, x one channel. fp16 y and / or MX copy */
xt xstem1(xt x, const float *w_dev, const float *b_dev, const float *gamma, const float *beta, float eps, const float *wsk_dev, const float *gsk, const float *bsk,
          float **rcoef, int act, int want_half, int want_q);   /* rcoef (with wsk_dev): the residual IN(1x1 conv of x) as per-(n, c) affine of x, for xin_apply with the one-channel x */
int xin_head(xt *h, const float *coef, int act, const float *hw_dev, float hb, float *out_dev);   /* the last apply fused with a 1x1 one-channel head (fp32 logits); -1: unsupported width */                   /* x->q from x->p (MX format of xnet_mx()); 0 on success */
xt xconcat_q(xt a, xt b);
xt xtconv_cat_q(const xconv *up, xt lo, xt skip);   /* the decoder input: ConvTranspose3d(lo) (MX GEMM, shuffled + quantised in its epilogue) ++ skip, MX only */              /* channel concat of the MX copies (a.q, b.q) only: the result has q and no fp16 p */               /* convs initialised from now on: 0 fp16, 6 / 5 MX-fp4 (ue8m0 / ue4m3 scales; default: env UFSM_XNET_MX) */
/* y = act(conv(x) * scale + bias (+ res)); allocates y. res (optional) has y's shape. */
xt xconv_fwd(const xconv *c, xt x, const xt *res, int act);
/* y = act(LN_c(conv(x) scale + bias) lnw + lnb (+ res)) with the LayerNorm in the conv epilogue (co 32 or 64; the default tile config) */
xt xconv_fwd_ln(const xconv *c, xt x, const xt *res, int act, const float *lnw, const float *lnb, float eps);
/* fold BatchNorm (gamma, beta, mean, var, eps) into per-channel scale / bias, plus a conv bias cb (nullptr: none) */
void xbn_fold(int co, const float *g, const float *b, const float *m, const float *v, float eps, const float *cb, float *scale, float *bias);

/* GroupNorm(groups) over D H W per sample, y = act(norm(x) gamma + beta (+ res)); y may be x (in place) */
void xgn(xt x, int groups, const float *gamma_dev, const float *beta_dev, float eps, const xt *res, int act, xt y);
float *xdev_floats(const float *host, size_t n);       /* a cached device copy of host floats (free with xfree_dev) */

xt xunfold(xt x, int kw, int pw);                      /* c = 1 -> c = 8: out[.., w, j] = x[.., w - pw + j] (j < kw), width W + 2pw - kw + 1 */
xt xmaxpool2d_k3s2(xt x);                              /* per depth plane, kernel 3, stride 2, pad 1 (-inf) */
xt xavgpool2d_k2(xt x);                                /* per depth plane, kernel 2, stride 2 */
xt xavgpool3d_k2(xt x);                                /* 3D, kernel 2, stride 2 (even D H W) */
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
float *xgn_stats(xt x, int groups, float eps);
xt xconv_fwd_gnin(const xconv *c, xt x, const float *gm, const float *rs, const float *gamma_dev, const float *beta_dev, int act);   /* conv(act(GN(x))) */   /* device [n][C] group mean, then [n][C] 1 / std (one block: free the returned pointer) */
/* the Reader depth collapse fused: act(GroupNorm(f)) -> logits w . v + b per layer -> [n][1][h][w][32] = (softmax-weighted sum, max) over
   the layers; equals xgn + xhead + xdepth_attn_max up to the logit summation order. f: 16 channels, <= 32 layers; f is not modified */
xt xdepth_collapse16(xt f, int groups, const float *gamma_dev, const float *beta_dev, float eps, int act, const float *w_dev, float bias);
/* Hecate depth collapse: s, l [n][d][h][w] fp32 (attention scores, logits), sup [n][h][w] uint8 (column has data),
   z-scale zs: out[n][h][w] = sum_d w_d l_d with w = masked softmax_d(s + zs (d - D/2) / (D/2)) renormalised */
void xhecate_collapse(const float *s_dev, const float *l_dev, const uint8_t *sup_dev, int n, int d, int h, int w, float zs, float *out_dev);

#ifdef __cplusplus
}
#endif
