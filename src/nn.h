/* CUDA ops with a C ABI. Tensors are fp32, NCDHW, on the device. Every op is synchronous on the
   default stream unless noted; nn_check() reports the last CUDA error. */
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { int n, c, d, h, w; } shape5;

static inline size_t shape_numel(shape5 s) { return (size_t)s.n * s.c * s.d * s.h * s.w; }
static inline size_t shape_spatial(shape5 s) { return (size_t)s.d * s.h * s.w; }

/* ---- device / memory ---- */
int nn_init(int device);                 /* selects the device; returns 0 on success */
const char *nn_check(void);              /* nullptr if no error since the last call, else the message */
void *nn_malloc(size_t bytes);
void nn_free(void *p);
void nn_zero(void *p, size_t bytes);
void nn_h2d(void *dst, const void *src, size_t bytes);
void nn_d2h(void *dst, const void *src, size_t bytes);
void nn_d2d(void *dst, const void *src, size_t bytes);
void nn_sync(void);
void *nn_host_alloc(size_t bytes);       /* pinned host memory */
void nn_host_free(void *p);
/* pipelined uploads: an async copy on a per-device copy stream (pinned source), events to order it against the compute stream */
void nn_h2d_copy_stream(void *dst, const void *src, size_t bytes);
void *nn_event_create(void);
void nn_event_record(void *ev, int on_copy_stream);      /* 0: compute stream, 1: copy stream */
void nn_stream_wait(int copy_stream_waits, void *ev);    /* make the copy stream (1) or the compute stream (0) wait for ev */
void nn_event_sync(void *ev);
size_t nn_mem_free(void);
/* event profiler (no host syncs): bracket ops, then collect per-category GPU milliseconds */
void nn_prof_begin(int k);
void nn_prof_end(void);
void nn_prof_collect(double *out, int nk);

/* Tensor-core TF32 path for the 3^3 stride-1 convolutions (default on; fp32 CUDA-core kernels otherwise). */
void nn_set_tf32(int on);
/* precision of the tensor-core path: 1 bf16 (default), 2 fp8 (e4m3 with MX block scales, fp32 accumulate), 3 fp4 forward +
   backward-data with fp8 weight gradient. nn_set_tf32(0) selects exact fp32; nn_set_tf32(1) restores the last precision. */
void nn_set_prec(int p);
int nn_get_prec(void);
/* per-layer precision: the network tags each conv with a layer id (unet order: enc0..3 = 0..3, down0..2 = 4..6, dec2, dec1,
   dec0 = 7..9, head = 10); a layer precision >= 1 overrides the global one. Policy strings: "enc0=1,dec0=1" or positional. */
void nn_set_layer(int id);
void nn_set_prec_wgrad(int p);   /* QAT: weight-gradient precision override (-1 = same as the layer); e.g. prec 2 forward with wgrad 1 */
/* 2:4 structured weight sparsity (groups of 4 input channels keep their 2 largest weights) and its SR-STE gradient term */
void nn_mask24(const float *w, float *out, int co, int ci, int taps);
/* true fp8 (bits 8, e4m3) / fp4 (bits 4, e2m1) weights: snaps w onto the MX block-scaled grid with stochastic rounding */
void nn_wquant(float *w, int co, int ci, int taps, int bits, unsigned seed);
/* packed fp8 / fp4 weight storage: q = one byte (e4m3) or one nibble (e2m1) per weight, sc = one ue8m0 scale per block of 32 input
   channels of a (co, tap). The optimizer and the EMA update the packed weights directly (stochastic rounding); unpack dequantizes
   into an fp32 shadow for the conv kernels. seed 0 in pack = round to nearest. */
size_t nn_wq_nblocks(int co, int ci, int taps);
void nn_wq_pack(const float *w, void *q, void *sc, int co, int ci, int taps, int bits, unsigned seed);
void nn_wq_unpack(const void *q, const void *sc, float *w, int co, int ci, int taps, int bits);
void nn_wq_adamw(void *q, void *sc, const float *g, float *m, float *v, int co, int ci, int taps, int bits, float lr, float b1, float b2, float eps, float wd, int step, unsigned seed);
void nn_wq_ema(void *qe, void *sce, const void *qp, const void *scp, int co, int ci, int taps, int bits, float decay, unsigned seed);
void nn_srste24(float *g, const float *w, int co, int ci, int taps, float lambda);
void nn_set_layer_prec(int id, int p);
int nn_set_prec_policy(const char *policy);
void nn_conv3d_fwd_fp8(const float *x, shape5 xs, const float *w, const float *b, int cout, float *y);        /* k=3 stride 1, fp32 tensors */
void nn_conv3d_fwd_fp4(const float *x, shape5 xs, const float *w, const float *b, int cout, float *y);
void nn_conv3d_bwd_weight_fp8(const float *x, shape5 xs, const float *gy, shape5 ys, float *gw, float *gb);
int nn_get_tf32(void);
/* Activation storage. With act-bf16 on (default; env UFSM_ACTF32=1 or nn_set_act_bf16(0) turns it off) and tensor
   cores on, the activation operands of the ops below (conv/up inputs x, conv/up/gn-apply outputs y, the x of the
   gn backward ops) are bf16 arrays behind the float* type: half the bytes. Gradients, GroupNorm statistics,
   parameters and the 1^3 head output are always fp32. The fp32 kernel path ignores the setting. */
void nn_set_act_bf16(int on);
int nn_get_act_bf16(void);
/* Gradient storage: with grad-bf16 on too (default; env UFSM_GRADF32=1 or nn_set_grad_bf16(0) turns it off) the
   activation-gradient tensors (gy/gx of the conv backward-data, gn backward and upsample backward ops, the gy of
   the weight-gradient ops) are bf16 as well; parameter gradients stay fp32. */
void nn_set_grad_bf16(int on);
int nn_get_grad_bf16(void);
void nn_f32_to_bf16(const float *x, size_t n, void *y);   /* converts to the current 16-bit storage type */
void nn_f32_to_h16(const float *x, size_t n, void *y, float scale);
/* 16-bit type for activation/gradient storage and MMA operands: 0 bf16 (default), 1 fp16 (8x finer mantissa, same tensor-core rate;
   env UFSM_F16=1). With fp16 gradient storage the activation gradients are kept scaled by nn_get_grad_scale() (env UFSM_GSCALE,
   default 1024) and the network divides the parameter gradients by it. The fp8/fp4 kernels require the bf16 setting. */
void nn_set_f16(int on);
int nn_get_f16(void);
void nn_set_grad_scale(float s);
float nn_get_grad_scale(void);

/* ---- conv3d: weight [cout][cin][k][k][k], bias [cout] or nullptr, pad k/2, stride 1 or 2 ---- */
shape5 nn_conv3d_out_shape(shape5 xs, int cout, int k, int stride);
void nn_conv3d_fwd(const float *x, shape5 xs, const float *w, const float *b, int cout, int k, int stride, float *y);
/* gx (shape xs) is SET. */
void nn_conv3d_bwd_data(const float *gy, shape5 ys, const float *w, shape5 xs, int k, int stride, float *gx, float *scratch);
int nn_conv3d_bwd_data_acc(const float *gy, shape5 ys, const float *w, shape5 xs, int k, int stride, float *gx, float *scratch);   /* gx += ...; -1 if unsupported */
/* gw / gb are ACCUMULATED (caller zeros them). */
void nn_conv3d_bwd_weight(const float *x, shape5 xs, const float *gy, shape5 ys, int k, int stride, float *gw, float *gb);
size_t nn_conv3d_scratch(shape5 xs, int cout, int k);   /* bytes of scratch bwd_data needs */

/* ---- GroupNorm over (C/G, D, H, W) per (n, g); gamma/beta [C]; mean/rstd [n*G] saved for bwd ---- */
void nn_gn_fwd(const float *x, shape5 s, int G, float eps, const float *gamma, const float *beta, float *y, float *mean, float *rstd);
/* gx SET; ggamma/gbeta ACCUMULATED. scratch: nn_gn_scratch(s) bytes. */
void nn_gn_bwd(const float *x, shape5 s, int G, const float *gamma, const float *mean, const float *rstd, const float *gy,
               float *gx, float *ggamma, float *gbeta, float *scratch);
size_t nn_gn_scratch(shape5 s);
/* Fused variants on the tensor-core path (return -1 when unavailable): conv of silu(gn(x)) without
   materialising the activation, and its weight gradient; plus the SiLU backward through a recomputed GN. */
int nn_conv3d_fwd_gn(const float *x, shape5 xs, int G, const float *gamma, const float *beta, const float *mean, const float *rstd,
                     const float *w, const float *b, int cout, float *y);
int nn_conv3d_bwd_weight_gn(const float *x, shape5 xs, int G, const float *gamma, const float *beta, const float *mean, const float *rstd,
                            const float *gy, shape5 ys, float *gw, float *gb);
void nn_silu_bwd_gn(const float *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd, const float *gy, float *gx);
/* Conv whose epilogue also yields the GroupNorm statistics of its output (G_in > 0 applies gn+silu to the input). */
int nn_conv3d_fwd_gn_stats(const float *x, shape5 xs, int G_in, const float *gamma, const float *beta, const float *mean, const float *rstd,
                           const float *w, const float *b, int cout, float *y, int G_out, float eps, float *omean, float *orstd);
int nn_conv3d_fwd_split(const float *x, const float *x2, int c_split, shape5 xs, int G_in, const float *gamma, const float *beta, const float *mean, const float *rstd,
                        const float *w, const float *b, int cout, float *y, int G_out, float eps, float *omean, float *orstd);
int nn_conv3d_bwd_data_split(const float *gy, shape5 ys, const float *w, shape5 xs, float *gx, float *gx2, int o_split, float *scratch);
int nn_conv3d_bwd_weight_split(const float *x, const float *x2, int c_split, shape5 xs, int G, const float *gamma, const float *beta, const float *mean, const float *rstd,
                               const float *gy, shape5 ys, float *gw, float *gb);
void nn_gn_silu_apply(const float *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd, float *y);
/* Backward through silu(gn(x)) in one go: gx SET (may alias gy), ggamma/gbeta ACCUMULATED. */
void nn_gn_silu_bwd(const float *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd, const float *gy,
                    float *gx, float *ggamma, float *gbeta, float *scratch);
/* GroupNorm forward fused with SiLU (y = silu(gn(x))), and recompute of both g = gn(x) and silu(g). */
void nn_gn_fwd_silu(const float *x, shape5 s, int G, float eps, const float *gamma, const float *beta, float *y, float *mean, float *rstd);
void nn_gn_apply_silu(const float *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd, float *g, float *sil);
/* Recompute y = gn(x) from saved mean/rstd (no statistics pass). */
void nn_gn_apply(const float *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd, float *y);

/* ---- elementwise ---- */
void nn_silu_fwd(const float *x, size_t n, float *y);
void nn_silu_bwd(const float *x, const float *gy, size_t n, float *gx);   /* gx SET */
void nn_axpy(float *y, float a, const float *x, size_t n);               /* y += a x */
void nn_scale(float *y, float a, size_t n);
void nn_u8_to_f32(const uint8_t *x, size_t n, float scale, float *y);

/* ---- trilinear 2x upsample (align_corners = false); ys = 2x spatial of xs ---- */
void nn_up2_fwd(const float *x, shape5 xs, float *y);
/* Same, writing into channels [c0, c0 + xs.c) of a destination tensor with ctot channels (e.g. a concat buffer). */
void nn_up2_fwd_into(const float *x, shape5 xs, float *y, int ctot, int c0);
void nn_up2_bwd(const float *gy, shape5 xs, float *gx);                  /* gx SET */

/* ---- channel concat: y[:, :ca] = a, y[:, ca:] = b ---- */
void nn_concat_fwd(const float *a, int ca, const float *b, int cb, shape5 s, float *y);   /* s.c unused */
void nn_concat_bwd(const float *gy, int ca, int cb, shape5 s, float *ga, float *gb);      /* both SET */

/* ---- loss: logits [n][C][S], teacher t uint8 [n][C][S] (p*255), mask m uint8 [n][S], w uint8 [n][C].
   For every (n,c) with w = 1: bce = mean over mask of BCE(logit, p); dice = 1 - (2 sum(sig p)+1)/(sum sig + sum p + 1).
   loss = sum over active (n,c) of (bce + dice_w * dice) / max(1, active). gl SET to d loss / d logits.
   out[0..C-1] = mean bce per channel, out[C..2C-1] = mean dice per channel, out[2C] = active count. */
void nn_loss(const float *logits, const uint8_t *t, const uint8_t *m, const uint8_t *w, shape5 s, float dice_w,
             float *gl, float *out, float *scratch);
size_t nn_loss_scratch(shape5 s);
/* Asynchronous loss: kernels only; fetch the host values later (bce per channel, dice per channel, active). */
void nn_loss_async(const float *logits, const uint8_t *t, const uint8_t *m, const uint8_t *w, shape5 s, float dice_w, float *gl, float *scratch);
void nn_loss_fetch(const float *scratch, shape5 s, float *out);
/* Device-to-device copy across GPUs (peer access when possible). */
void nn_peer_copy(void *dst, int dst_dev, const void *src, int src_dev, size_t bytes);

/* ---- optimizer ---- */
void nn_adamw(float *p, const float *g, float *m, float *v, size_t n, float lr, float b1, float b2, float eps, float wd, int step);
void nn_ema(float *ema, const float *p, size_t n, float decay);
double nn_sum(const float *x, size_t n, float *scratch);     /* scratch: >= 4096 floats */
double nn_sumsq(const float *x, size_t n, float *scratch);
void nn_sigmoid(const float *x, size_t n, float *y);

#ifdef __cplusplus
}
#endif
