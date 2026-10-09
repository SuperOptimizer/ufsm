/* CUDA ops with a C ABI. Tensors are fp32, NCDHW, on the device. Every op is synchronous on the
   default stream unless noted; nn_check() reports the last CUDA error. */
#pragma once
#include <stdlib.h>
#include <string.h>
/* boolean environment switch: set and not "0" (UFSM_X=0 means off) */
static inline int ufsm_env_on(const char *n) { const char *e = getenv(n); return e && *e && strcmp(e, "0") != 0; }
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
void *nn_host_alloc_try(size_t bytes);   /* the same, nullptr when the host cannot pin that much */
void nn_host_free(void *p);
/* pipelined uploads: an async copy on a per-device copy stream (pinned source), events to order it against the compute stream */
void nn_h2d_copy_stream(void *dst, const void *src, size_t bytes);
void *nn_event_create(void);
void nn_event_record(void *ev, int on_copy_stream);      /* 0: compute stream, 1: copy stream */
void nn_stream_wait(int copy_stream_waits, void *ev);    /* make the copy stream (1) or the compute stream (0) wait for ev */
void nn_event_sync(void *ev);
/* activation offload: copy (pinned host <-> device) on an offload stream once the compute stream reaches this point (ev_start);
   ev_done completes with the copy (nn_stream_wait(0, ev_done) before the compute stream touches dst / src again) */
void nn_offload_copy(void *dst, const void *src, size_t n, int to_host, void *ev_start, void *ev_done);
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
void nn_set_sr(int on);              /* stochastic rounding of the fp8 gradient operands: backward-data input, weight-gradient gy (env UFSM_SR=1) */
void nn_set_sr_step(unsigned step);  /* reseed per training step (deterministic per step) */
void nn_wmemo_clear(void);   /* the fp4 kernels memoise prepared weights per conv and step: call after a weight change outside a training step (checkpoint load, EMA swap) */
int nn_get_prec(void);
/* per-layer precision: the network tags each conv with a layer id (unet order: enc0..3 = 0..3, down0..2 = 4..6, dec2, dec1,
   dec0 = 7..9, head = 10); a layer precision >= 1 overrides the global one. Policy strings: "enc0=1,dec0=1" or positional. */
void nn_set_layer(int id);
void nn_set_nlev(int L);   /* U-Net depth behind the layer ids / names (default 4; unet_create sets it) */
int nn_get_nlev(void);
const char *nn_layer_name(int id, char *buf);   /* "enc2", "down0", "dec1", "head" for the current depth; buf: 16 bytes */
void nn_set_prec_wgrad(int p);   /* QAT: weight-gradient precision override (-1 = same as the layer); e.g. prec 2 forward with wgrad 1 */
/* 2:4 structured weight sparsity (groups of 4 input channels keep their 2 largest weights) and its SR-STE gradient term */
void nn_mask24(const float *w, float *out, int co, int ci, int taps);
/* true fp8 (bits 8, e4m3) / fp4 (bits 4, e2m1) weights: snaps w onto the MX block-scaled grid with stochastic rounding */
void nn_wquant(float *w, int co, int ci, int taps, int bits, unsigned seed);
/* packed fp8 / fp4 weight storage: q = one byte (e4m3) or one nibble (e2m1) per weight, sc = one ue8m0 scale per block of 32 input
   channels of a (co, tap). The optimizer and the EMA update the packed weights directly (stochastic rounding); unpack dequantizes
   into an fp32 shadow for the conv kernels. seed 0 in pack = round to nearest. */
size_t nn_wq_nblocks(int co, int ci, int taps);
size_t nn_wq_bytes(int co, int ci, int taps, int bits);   /* packed bytes: fp8 = elements, fp4 = 16 per block of 32 */
void nn_wq_pack(const float *w, void *q, void *sc, int co, int ci, int taps, int bits, unsigned seed);
void nn_wq_unpack(const void *q, const void *sc, float *w, int co, int ci, int taps, int bits);
void nn_wq_adamw(void *q, void *sc, void *r, void *rsc, const float *g, float *m, float *v, int co, int ci, int taps, int bits, float lr, float b1, float b2, float eps, float wd, int step, unsigned seed);
void nn_wq_ema(void *qe, void *sce, void *re, void *rsce, const void *qp, const void *scp, const void *rp, const void *rscp, int co, int ci, int taps, int bits, float decay, unsigned seed);
/* optional fp8 error-feedback residual (r: e4m3 per weight, rsc: ue8m0 per block) used for fp4 weights: w = q4 s + r8 sr */
void nn_wq_residual(const float *w, const void *q, const void *sc, void *r, void *rsc, int co, int ci, int taps, int bits);
void nn_srste24(float *g, const float *w, int co, int ci, int taps, float lambda);
void nn_set_layer_prec(int id, int p);
int nn_set_prec_policy(const char *policy);   /* see nn.cu: block names, enc1.c2-style single convs, "all", fwd:bwd_data:wgrad values */
/* finer precision control (prec 4 = fp16 operands with fp16 group accumulation folded into fp32, needs bf16 storage):
   nn_set_conv tags the conv within the current layer (0 = c1, 1 = c2, -1 = untagged); nn_set_conv_prec sets the
   (forward, backward-data, weight-gradient) precisions of conv sub (-1 = both) of layer id (0 = follow the layer). */
void nn_set_conv(int sub);
int nn_get_layer(void);
int nn_get_conv(void);
void nn_set_conv_prec(int id, int sub, int p_fwd, int p_bwd_data, int p_wgrad);
int nn_get_conv_prec(int id, int sub, int pass);
int nn_cur_prec(void);
int nn_prec_parse(const char *name);     /* "bf16" 1, "fp8" 2, "fp4" 3, "fp16" 4, or a digit; -1 if unknown */
const char *nn_prec_name(int p);
int nn_prec_manifest(char *buf, size_t n);   /* requested per-conv fwd:bwd_data:wgrad policy; returns characters written */
int nn_exec_manifest(char *buf, size_t n);   /* observed compute paths, including storage/shape fallbacks; not storage precision */
void nn_conv3d_fwd_fp8(const float *x, shape5 xs, const float *w, const float *b, int cout, float *y);        /* k=3 stride 1, fp32 tensors */
void nn_conv3d_fwd_fp4(const float *x, shape5 xs, const float *w, const float *b, int cout, float *y);
void nn_conv3d_bwd_weight_fp8(const float *x, shape5 xs, const float *gy, shape5 ys, float *gw, float *gb);
int nn_get_tf32(void);
/* 1: convolution-output GN statistics describe the rounded stored tensor.
   0: preserve the historical unrounded epilogue statistics for old checkpoints. */
void nn_set_gn_stored(int on);
int nn_get_gn_stored(void);
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
/* Per-tensor storage registry. A device buffer registered with dt 8 holds an MX-fp8 tensor: channel-blocked e4m3 bytes
   data[n][blk][voxel][bw] (bw = 16 if C <= 16 else 32) followed by ue8m0 scales sc[n][blk][voxel]; nn_mx8_bytes(shape) bytes.
   Ops that read or write a registered tensor take their MX path (convs then run in fp8 and stage by copy). nn_free forgets.
   dt 4 holds an MX-fp4 tensor: the same blocking with packed e2m1 nibbles, data[n][blk][voxel][bw/2] + the ue8m0 plane
   (nn_mx4_bytes); convs reading it run the fp4 kernel (Ci > 16) or the fp8 kernel with nibble staging. */
void nn_set_storage(const void *p, size_t bytes, int dt);
int nn_storage(const void *p);          /* 8 = MX-fp8, 4 = MX-fp4, 0 = default */
void nn_storage_forget(const void *p);
size_t nn_mx8_bytes(shape5 s);
size_t nn_mx4_bytes(shape5 s);
size_t nn_mx_bytes(shape5 s, int dt);   /* registry dt 8 / 4 */
void nn_f32_to_act(const float *x, shape5 s, void *y);   /* network input -> activation storage (16-bit or MX) */
void nn_h16_to_mx(const void *x, shape5 s, void *y);   /* 16-bit storage-type tensor -> the registered MX (fp8 / fp4) tensor y */
void nn_f32_to_h16(const float *x, size_t n, void *y, float scale);
/* 16-bit type for activation/gradient storage and MMA operands: 0 bf16 (default), 1 fp16 (8x finer mantissa, same tensor-core rate;
   env UFSM_F16=1). With fp16 gradient storage the activation gradients are kept scaled by nn_get_grad_scale() (env UFSM_GSCALE,
   default 1024) and the network divides the parameter gradients by it. The fp8/fp4 kernels require the bf16 setting. */
void nn_set_f16(int on);
int nn_get_f16(void);
void nn_set_grad_scale(float s);
void nn_set_logits_h16(int on);     /* the losses read fp16 logits (unet_logits_h16) */
void nn_set_head_out_h16(int on);   /* the MX head writes fp16 logits (set by unet around its head) */
void nn_set_loss_grad_h16(int on);   /* nn_loss_async writes gl as the 16-bit storage type scaled by the gradient scale (unet_backward_x(.., 1)) */
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
/* Conv with output GN statistics under nn_set_gn_stored's contract (G_in > 0 transforms the input).
   G_out > 0 with null mean/rstd replays the normalization-input store, including FP16 clamping, without reducing statistics. */
int nn_conv3d_fwd_gn_stats(const float *x, shape5 xs, int G_in, const float *gamma, const float *beta, const float *mean, const float *rstd,
                           const float *w, const float *b, int cout, float *y, int G_out, float eps, float *omean, float *orstd);
/* Input-side recompute. The conv input is silu(gn(x)) formed while staging from a stored pre-norm x (gx: GroupNorm
   parameters and statistics; nullptr = x used as is). With x2 the input is the channel concat [x, x2] (c_split = channels
   of x, gx2 the GroupNorm of x2; gx needs gx2, gx2 alone transforms x2 only). k=3 stride 1 (optionally with GN
   statistics of the output for G_out groups), k=3 stride 2 and k=1 (no split). up = 1 (split only, gx nullptr): x is
   stored at half resolution and read as its trilinear 2x upsample (nn_up2_fwd_into values) while staging. Tensor-core
   path only; -1 when unsupported. */
typedef struct { const float *gamma, *beta, *mean, *rstd; int G; } nn_gn_t;
int nn_conv3d_fwd_x(const float *x, const nn_gn_t *gx, const float *x2, const nn_gn_t *gx2, int c_split, int up, shape5 xs,
                    const float *w, const float *b, int cout, int k, int stride, float *y, int G_out, float eps, float *omean, float *orstd);
int nn_conv3d_bwd_weight_x(const float *x, const nn_gn_t *gx, const float *x2, const nn_gn_t *gx2, int c_split, int up, shape5 xs,
                           const float *gy, shape5 ys, int k, int stride, float *gw, float *gb);   /* gw / gb accumulated */
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
int nn_gn_stats(const float *x, shape5 s, int G, float eps, float *mean, float *rstd);   /* stats of the stored fp32, 16-bit or MX tensor; -1 for unsupported groups */
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
void nn_up2_fwd_mx_range(const float *x, shape5 xs, int c0, int nc, float *y);   /* MX: channels [c0, c0 + nc) of x (32-aligned) upsampled into an nc-channel y */
void nn_add_rows(float *dst, size_t dld, const float *src, size_t sld, int rows, size_t cols);   /* dst[r dld + c] += src[r sld + c] */
/* accuracy study: round a stored activation (activation storage type) in place to a narrower block-scaled format:
   1 NVFP4 (e2m1, e4m3 scale per 16 channels, fp32 tensor scale), 2 MXFP4, 3 MXFP6 e2m3, 4 MXFP6 e3m2, 5 MXFP8 e4m3
   (ue8m0 per 32 channels). No-op for MX-stored tensors. */
void nn_fake_quant(void *x, shape5 s, int fmt);
void nn_fake_quant_affine(void *x, shape5 s, int fmt, const float *mean, const float *rstd, int G);   /* quantise (x - mean) * rstd per (n, group), restore; MX formats */
void nn_up2_fwd_gn_into(const float *x, shape5 xs, const nn_gn_t *g, float *y, int ctot, int c0);   /* upsample of silu(gn(x)) (g nullptr: of x) */
void nn_up2_bwd(const float *gy, shape5 xs, float *gx);
void nn_up2_bwd_into(const float *gy, shape5 xs, float *gx, int ctot, int c0);   /* gy: xs.c channels at 2x; gx: channels c0.. of a ctot-channel tensor (SET) */
int nn_conv3d_bwd_data_range(const float *gy, shape5 ys, const float *w, shape5 xs, int c0, int nc, float *gx, float *scratch);   /* input channels [c0, c0 + nc) only; -1 unsupported */                  /* gx SET */

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
/* Offset-tolerant positives on channel 0 (radius r <= 2 voxels, 0 = off; see nn.cu): code is a uint8 [n][S] work buffer. */
void nn_set_loss_tol(int r);
int nn_get_loss_tol(void);
void nn_loss_async_tol(const float *logits, const uint8_t *t, const uint8_t *m, const uint8_t *w, shape5 s, float dice_w, float *gl, float *scratch, uint8_t *code);
/* Affinity loss on band labels (task band_affinity, src/nn_loss.cu): output channels c0 .. c0 + K - 1 are affinities for
   the offsets off (axis 0 z / 1 y / 2 x, distance d); band [n][S] holds band values in 1/18-turn steps mod 252, 255 =
   unknown. Writes those channels of gl (scaled by lambda / K); fin: nn_aff_scratch(K) bytes of per-channel results
   [bce, dice, wdiff, 1/norm, den, num]. Split z: pairs are owned by this GPU's planes, partners may lie in the halo. */
typedef struct { int K; int ax[8]; int d[8]; } aff_offsets_t;
size_t nn_aff_scratch(int K);
void nn_aff_loss_async(const float *logits, shape5 s, int c0, const uint8_t *band, aff_offsets_t off, float dice_w, float lambda, float *gl, float *fin);
void nn_aff_loss_fetch(const float *fin, int K, float *out);
void nn_set_pos_weight(float w);   /* BCE weight of positive targets (class balance), default 1 */
void nn_loss_fetch(const float *scratch, shape5 s, float *out);
void nn_loss_grad_scale(void *g, size_t n, float a);   /* scale a logit gradient (the losses' storage format) */
/* Sparse trilinear samples. Ownership is global z in [own_lo,own_hi), so a
   sample crossing a split boundary is reduced once and its corners get the
   correct local gradients. coords is host [np][3], values host [np][2]. */
void nn_sheet_gather(const float *logits, shape5 s, const float *coords, size_t np,
                     int z0, int own_lo, int own_hi, double *values);
/* Unique local flattened indices; gradients add to the dense loss buffer. */
void nn_sheet_scatter(float *gl, const uint64_t *indices, const float *values, size_t n, int h16);
/* rows[z] = {origin_y-axis_y,origin_x-axis_x,1/pitch,offset} in native voxels */
void nn_sheet_input(void *input,int W,const float *rows,float center,float scale,int h16);
/* In-place absolute q and validity, no additional dense device buffer. */
void nn_sheet_gate(float *residual,const float *surface,const uint8_t *ct,int W,const float *rows);
/* ---- spatial split of one window along z across two GPUs (unet_set_split, src/split.c) ----
   nn_split_cfg (per current device): this GPU's tensors carry lo0 halo planes at the low z end and hi0 at the high end of the
   level-0 local depth D0 (Dg0 = the whole window's level-0 depth); D0 = 0 turns it off. GroupNorm statistics then skip the
   halo planes and, like the GroupNorm backward sums and the loss statistics, are summed across both GPUs by the reduce
   callback (registered once; called on the GPU's own thread with a device buffer of n doubles to sum in place). */
void nn_split_cfg(int lo0, int hi0, int D0, int Dg0);
void nn_split_thread_slot(int side);   /* both halves on one GPU: this thread's split config is side's (-1: the device's) */
void nn_split_set_reduce(void (*fn)(double *, int));
/* esz: bytes per element of a plane-major tensor, 0 = an MX-registered tensor */
void nn_split_zero(const void *t, shape5 s, int esz, int lo, int hi);   /* zero the halo planes (current device) */
size_t nn_split_halo_bytes(shape5 s, int esz);
/* halo exchange: begin for both sides at once (one thread), then end on each side's own thread; slot 0 / 1: two exchanges
   may be in flight, each with its own buffers */
void nn_split_halo_begin(void *const *t, const int *dev, shape5 s, int esz, int h, void *const *sb, void *const *rb, int slot);
void nn_split_halo_end(void *t, int side, shape5 s, int esz, int h, void *rb, int slot);
void nn_split_allreduce(double *const *b, const int *dev, int n, double *const *r);
/* Device-to-device copy across GPUs (peer access when possible). */
void nn_peer_copy(void *dst, int dst_dev, const void *src, int src_dev, size_t bytes);

/* ---- optimizer ---- */
void nn_adamw(float *p, const float *g, float *m, float *v, size_t n, float lr, float b1, float b2, float eps, float wd, int step);
/* Muon step for one [Co][K] weight: nesterov momentum (mom), Newton-Schulz orthogonalisation, p = p (1 - lr wd) - lr sqrt(max(1, Co/K)) O; work >= 2 Co K + 2 Co^2 floats */
void nn_muon(float *p, const float *g, float *mom, int Co, int K, float lr, float beta, float wd, float *work);
/* batched Muon over nconv weights: descs is a device array of {float *p; const float *g; float *mom, *X, *Y, *A, *B; int Co, K;} (X, Y: Co K floats; A, B: Co Co) */
void nn_muon_batch(const void *descs, int nconv, int maxco, int maxk, float lr, float beta, float wd);
/* ANVIL II over nconv weights: descs {float *p; const float *g; float *v0, *X, *Y, *A, *B, *v1, *E, *R; int Co, K;} (E, R: Co floats) */
void nn_anvil_batch(const void *descs, int nconv, int maxco, int maxk, float lr, float beta_fast, float beta_slow, float w_fast, float mu, float beta2, float wd);
void nn_ema(float *ema, const float *p, size_t n, float decay);
double nn_sum(const float *x, size_t n, float *scratch);     /* scratch: >= 4096 floats */
double nn_sumsq(const float *x, size_t n, float *scratch);
void nn_sigmoid(const float *x, size_t n, float *y);
void nn_flip_sign(void *x, size_t n, int h16);   /* x[i] = -x[i] for n fp32 (h16 0) or 16-bit (h16 1) values: a sign-bit flip, exact */
void nn_prob_u8(const float *logits, size_t n, uint8_t *out, float hard);   /* out[i] = round(255 sigmoid(logits[i])); hard > 0: 255 where sigmoid >= hard, else 0 */
/* thin ridge of a uint8 probability volume along the per-voxel direction in channels 1..3 of the network input x ([4][d][h][w],
   fp32 / fp16 / bf16 by xfmt 0 / 1 / 2): out = 255 where some voxel within 1 step along the direction is a ridge (p >= thr and
   p >= p at +-1 and +-2 steps along the direction, nearest-voxel sampling), else 0 */
void nn_ridge_u8(const uint8_t *p, const void *x, int xfmt, int d, int h, int w, int thr, uint8_t *out);
void nn_fill_unlabelled(uint8_t *t, uint8_t *m, const uint8_t *vm, const uint8_t *src, size_t n);   /* where m == 0 and vm != 0: t = src, m = 1 */
/* inference: network input from the uint8 CT window on the device (z-score + radial channels; 16-bit storage when h16),
   and recto probability * 255 (0 where the CT is 0) from the logits */
void nn_pred_input(const uint8_t *ct, int W, float mean, float isd, const float *dyo, const float *dxo, int axis, void *x, int h16);
void nn_pred_output(const float *lg, const uint8_t *ct, size_t n, uint8_t *out);
void nn_pred_place(const float *lg, const uint8_t *ct, int W, int halo, int oz, int oy, int ox, int ez, int ey, int ex, int shard, uint8_t *dsh);   /* interior -> device shard */
void nn_pred_stats(const uint8_t *ct, size_t n, void *scratch, size_t *nz, double *sum, double *sq);   /* exact window stats (scratch: 24 B device) */

#ifdef __cplusplus
}
#endif
