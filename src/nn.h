/* CUDA ops with a C ABI. Tensors are fp32, NCDHW, on the device. Every op is synchronous on the
   default stream unless noted; nn_check() reports the last CUDA error. */
#pragma once
#include <stdlib.h>
#include <string.h>
/* boolean environment switch: set and not "0" (UFSM_X=0 means off) */
static inline int ufsm_env_on(const char *n) { const char *e = getenv(n); return e && *e && strcmp(e, "0") != 0; }
#include <stddef.h>
#include <stdio.h>
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
const char *nn_check_site(void);         /* the call site of the error the last nn_check returned ("" for none; low-precision kernels: their own latch) */
void *nn_malloc(size_t bytes);
void nn_free(void *p);
void nn_zero(void *p, size_t bytes);
void nn_h2d(void *dst, const void *src, size_t bytes);
void nn_d2h(void *dst, const void *src, size_t bytes);
void nn_d2d(void *dst, const void *src, size_t bytes);
void nn_copy_async(void *dst, const void *src, size_t bytes);   /* device / pinned host in any direction, on the compute stream */
void nn_d2d_async(void *dst, const void *src, size_t bytes);   /* on the default stream, no host sync */
void nn_d2d_kernel(void *dst, const void *src, size_t bytes);  /* the same as a copy kernel (16-byte aligned dst, src): no copy engine */
void nn_set_zrange(int zi0, int Di, int first, int last, int Do, int zo);   /* z-range of the next MX forward convs (see nn_bwd.cu): output planes zo + .. of Do, reading input planes zi0 + .. of a depth-Di input; Di 0 = off */
void nn_set_zrange_out(int on);   /* 1: the z-range's MX conv outputs go to planes zo + .. of the depth-Do output itself (no temp; nn_bwd.cu) */
void nn_set_zwin(int zw0, int zw1, int D0, int first, int last);   /* z-window of the next level-0 ops (see nn_bwd.cu): output planes [zw0, zw1) of their whole tensors, in planes of depth D0 (scaled per tensor); D0 0 = off; first / last zero / finalise GroupNorm sums */
void nn_sync(void);
const char *nn_device_error(void);   /* cudaDeviceSynchronize: an earlier kernel fault (sticky) as a message, else nullptr */
void *nn_host_alloc(size_t bytes);       /* pinned host memory */
void *nn_managed_alloc(size_t bytes);    /* CUDA managed memory (may exceed the device: pages migrate on demand); nn_free frees it */
void *nn_malloc_try(size_t bytes);       /* device memory, nullptr (no sticky error) when it does not fit */
void *nn_host_alloc_try(size_t bytes);   /* the same, nullptr when the host cannot pin that much */
void nn_host_free(void *p);
size_t nn_host_bytes(void);            /* live pinned host bytes (nn_host_alloc*) */
void nn_host_report(FILE *f);         /* the same by caller address */
/* pipelined uploads: an async copy on a per-device copy stream (pinned source), events to order it against the compute stream */
void nn_h2d_copy_stream(void *dst, const void *src, size_t bytes);
void nn_d2h_copy_stream(void *dst, const void *src, size_t bytes);   /* device -> pinned host on the copy stream */
void nn_h2d_async(void *dst, const void *src, size_t bytes);         /* pinned host -> device on stream 0 (asynchronous, in stream order) */
void *nn_event_create(void);
void nn_event_record(void *ev, int on_copy_stream);      /* 0: compute stream, 1: copy stream, 2: the aux stream (nn_d2d_aux) */
void nn_stream_wait(int copy_stream_waits, void *ev);    /* make the copy stream (1), the aux stream (2) or the compute stream (0) wait for ev */
void nn_set_conv1_const(int on);   /* fp16 1^3 heads with constant-bank weights (UFSM_CONV1_CONST, default 1; bit-identical): 0 the staged kernel */
void nn_d2d_aux(void *dst, const void *src, size_t bytes);   /* device copy on the per-device aux stream (ordered by events only:
                                                                in-place slab plane copies overlapped with the next slab's conv) */
void nn_event_sync(void *ev);
/* activation offload: copy (pinned host <-> device) on an offload stream once the compute stream reaches this point (ev_start);
   ev_done completes with the copy (nn_stream_wait(0, ev_done) before the compute stream touches dst / src again) */
void nn_offload_copy(void *dst, const void *src, size_t n, int to_host, void *ev_start, void *ev_done);
size_t nn_mem_free(void);
/* train --lat: label planes kept on the device, packed (bits 8 bytes, 4 codes 0..14 / 255, 2 --rv papyrus classes); stream k: 0 compute, 1 copy, 2 aux */
void nn_lab_quant(int k, uint8_t *s, size_t n, int bits);   /* experiment: a cyclic label plane (codes 0..251) to 2^bits levels in place */
void nn_lab_pack(int k, const uint8_t *s, uint8_t *d, size_t n, int bits);
void nn_lab_unpack(int k, const uint8_t *p, uint8_t *d, size_t v0, size_t n, int bits);
/* virtual-memory rings (CUDA VMM): one tensor's whole address range of nreg consecutive regions (reg[r] bytes, slabs of slab[r]), slab
   s of a region backed by physical chunk s mod R of it (kernels address the whole tensor; the schedule keeps fewer than R slabs live).
   nullptr: no VMM, or sizes off the mapping granularity (nn_ring_gran). nn_free releases it; *phys: bytes backed */
void *nn_ring_alloc(int nreg, const size_t *reg, const size_t *slab, int R, size_t *phys);
size_t nn_ring_gran(void);
size_t nn_ring_bytes(void);
/* event profiler (no host syncs): bracket ops, then collect per-category GPU milliseconds */
void nn_prof_begin(int k);
void nn_prof_end(void);
void nn_h2d_stream(int k, void *d, const void *s, size_t n);   /* an asynchronous upload on stream k (0 compute, 1 copy, 2 aux: nn_stream_wait / nn_event_record); s pinned */
int nn_ptick_on(void); void nn_ptick(const char *name); void nn_ptick_report(int steps);   /* UFSM_PTICK=1: synchronised phase timer (nn_core.cu) */
void nn_ptick_fine(const char *name);   /* the same with UFSM_PTICK=2 only (ticks inside a phase) */
void nn_prof_collect(double *out, int nk);

/* Tensor-core TF32 path for the 3^3 stride-1 convolutions (default on; fp32 CUDA-core kernels otherwise). */
void nn_set_tf32(int on);
/* precision of the tensor-core path: 1 bf16 (default), 2 fp8 (e4m3 with MX block scales, fp32 accumulate), 3 fp4 forward +
   backward-data with fp8 weight gradient. nn_set_tf32(0) selects exact fp32; nn_set_tf32(1) restores the last precision. */
void nn_set_prec(int p);
void nn_set_sr(int on);              /* stochastic rounding of the fp8 gradient operands: backward-data input, weight-gradient gy (env UFSM_SR=1) */
void nn_set_sr_step(unsigned step);
/* the per-host-thread op state (layer ids, the stochastic-rounding counter, the loss / logit formats, ...): one host thread per GPU starts
   from the main thread's (capture in the main thread, install in the worker) */
typedef struct { int layer, sub, pass, block; unsigned sr_ctr; float posw; int logits_h16, head_h16, loss_g16, gn_const, actbf; } nn_tls;
void nn_tls_capture(nn_tls *t);
void nn_tls_install(const nn_tls *t);  /* reseed per training step (deterministic per step) */
unsigned nn_sr_ctr(void);            /* the step's seed draw counter (reordered ops replay draws: nn_sr_set_ctr) */
void nn_sr_set_ctr(unsigned c);
unsigned nn_sr_seed(void);           /* one draw of the step's seed sequence (what an op drawing a seed would get) */
int nn_sr_on(void);
void nn_exec_prec_note(int pass, int p);   /* executed precision p of pass (0 fwd, 1 bwd-data, 2 wgrad) of the current layer */
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
void nn_set_block(int blk);   /* 0..15: which of several convs under one layer id (residual blocks); part of the weight-memo key */
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
/* 1 when the MX-fp4 / fp8 paths (packed activations, fp8 / fp4 compute) are on: by default in every build (ARCH=120: the
   block-scaled tensor cores; ARCH=86: the exact software MX kernels, e.g. on an RTX 3090); UFSM_MX_EMU=0 forces them off. Off:
   every fp8 / fp4 precision runs as bf16 and MX storage is 16-bit (one log line). Makes no CUDA call (safe before a fork). */
int nn_mx_ok(void);
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
void nn_storage_dump(const char *tag);   /* debug: prints the storage registry */
void nn_label_warp_ref(int on);   /* tests: the label warps' reference kernels */
void nn_aug_stage_reset(void);     /* before a batch's label / CT uploads: the staging buffer holds no plane of this batch yet */
size_t nn_mx8_bytes(shape5 s);
size_t nn_mx4_bytes(shape5 s);
size_t nn_mx_bytes(shape5 s, int dt);   /* registry dt 8 / 4 */
void nn_f32_to_act(const float *x, shape5 s, void *y);   /* network input -> activation storage (16-bit or MX) */
void nn_h16_to_mx(const void *x, shape5 s, void *y);   /* 16-bit storage-type tensor -> the registered MX (fp8 / fp4) tensor y */
/* train --gpu-input: the network input (z-scored CT, then the radial unit vector through the window's cube symmetry) built on the
   device, bit for bit sample.c's write_x (16-bit) followed by nn_h16_to_mx. ct16: the CT channel [n][S] in the 16-bit storage type
   (fp16 / bf16 as nn_set_f16), the jitter and appearance augmentation applied by the sampler; rec: per sample NN_RAD_REC(D) bytes:
   int32 o1, o2 (the window corner's y / x at its level), D, P, hasax, perm[0] | perm[1] << 2 | perm[2] << 4 | flip[0] << 8 |
   flip[1] << 9 | flip[2] << 10, 0, 0; then float cyz[D], cxz[D] (the axis centre per source z); xs: {n, 4, D, P, P}; y: the
   registered MX input (fp8 / fp4). The 16-bit 4-channel window never exists (6 B per voxel less on the host, PCIe and device).
   Warped windows (sample.c write_spatial, cubes only: D == P): header word 6 = 1 | affine << 1, and after cxz the warp: float
   table[6 P] (spatial_aug_tables), R[9], A[9], Ainv[9] (row major, as floats), scale; the radial vector is then the transported one
   of write_spatial (zero outside the source cube), bit for bit. */
#define NN_RAD_REC(D) ((size_t)(8 + 8 * (D) + 32) * 4)
void nn_input_rad(const void *ct16, shape5 xs, const void *rec, void *y);
/* --half-res pack: the same from the full-resolution CT channel [n][8 d h w] and records of NN_RAD_REC(2 d), into the network's
   8-channel input on the 2x2x2 blocks (xs {n, 8, d, h, h}): bit for bit write_x -> nn_half_input(mode 2) -> nn_h16_to_mx */
void nn_input_rad_pack(const void *ct16, shape5 xs, const void *rec, void *y);
/* train --lat 4 (pool4) with --gpu-input: the 4-channel MX input on the 4^3 cells of the window (xs {n, 4, d, h, h}: the cell grid), each
   channel the cell mean of nn_input_rad's full-resolution value (warp included) from the full-resolution 16-bit CT channel ct16 */
void nn_input_lat(const void *ct16, shape5 xs, const void *rec, void *y);
/* the same with the learned cell transform (latfit.h): xs.c = C + 3 (16 or 32): C quantised coefficients x gain, then the radial means */
void nn_input_latc(const void *ct16, shape5 xs, const void *rec, const float *bas, const float *gain, int C, float step, void *y);
/* predict: the same channels from the full-resolution 4-channel fp16 window x ([4][4d][4h][4w]: CT, radial) into y (fp16 [C + 3][d][h][w]) */
/* predict --latents: the cell channels from a latcomp store's indices q (int8 [d][h][w](C + 1), lc_cli.c layout; raw-domain step) with the window's
   mean / 1 / sd; the radial means from x's channels 1-3 */
void nn_lat_input_q(const void *q, const void *x, int d, int h, int w, const float *gain, int C, float step_raw, float mean, float isd, void *y);
/* predict --latents: the CT window rebuilt from a window of stored cells (nn_lat.cu lat_synth_k: 1..255 in material cells, 0 in air
   cells), and a --lat-basis net's input straight from the cells with the radial channels from the axis offsets (lat_pred_in_q_k) */
void nn_lat_synth_ct(const void *q, int C, const float *bas, float step_raw, int D, int W, uint8_t *ct);
void nn_lat_pred_input_q(const void *q, int D, int W, float mean, float isd, const float *dyo, const float *dxo, int axis,
                         const float *gain, int C, float step_raw, void *y);
/* the guided head (unet_cfg.lat_guide): see nn_lat.cu lat_guide_k */
/* the sub-cell head (unet_cfg.lat_guide 64): see nn_lat.cu lat_shuf_k */
void nn_lat_shuffle(const void *lg, int lh16, int d, int h, int w, int base, int ng, int z0, int nz, void *y);
void nn_lat_shuffle_adj(const void *g, int d, int h, int w, int base, int ng, int z0, int nz, void *gc);   /* gc: fp16 */
void nn_lat_guide4(const void *x, int D, int H, int W, int z0, int nz, void *lg, int base, int c0, int ng, int dir);   /* 4 coefficients per guided channel */
void nn_lat_guide(const void *x, int D, int H, int W, int z0, int nz, void *lg, int base, int c0, int dir);
void nn_lat_input_x(const void *x, int d, int h, int w, const float *bas, const float *gain, int C, float step, void *y);
/* latent-grid logits -> one z-slab of fine logits and back (src/nn_lat.cu; the pool4 chain of nn_lowres_output, B 1): coarse lg
   [C][d][h][w] (fp16 when lh16, else fp32) -> fine planes [z0, z0 + nz) of the 4d x 4h x 4w window, fp16 [C][nz][4h][4w] in y;
   nn_lat_up_adj adds the adjoint of such a slab's fp16 gradient g into the fp32 coarse gradient gc [C][d][h][w] */
void nn_lat_up_slab(const void *lg, int lh16, int C, int d, int h, int w, int z0, int nz, void *y);
/* _c: the cell-grid array holds planes [q0, q0 + qd) only ([C][qd][h][w]); nn_lat_grad_fin / _carry: the streamed head's rolling
   coarse gradient (nn_lat.cu) */
void nn_lat_up_slab_c(const void *lg, int lh16, int C, int d, int h, int w, int z0, int nz, void *y, int q0, int qd);
void nn_lat_up_adj_c(const void *g, int C, int d, int h, int w, int z0, int nz, float *gc, void *scratch, int q0, int qd);
void nn_lat_shuffle_c(const void *lg, int lh16, int d, int h, int w, int base, int ng, int z0, int nz, void *y, int q0, int qd);
void nn_lat_shuffle_adj_c(const void *g, int d, int h, int w, int base, int ng, int z0, int nz, void *gc, int q0, int qd);
void nn_lat_grad_fin(const float *a, int nf, const void *sb, int C, size_t pl, int cap, int i0, int np, void *gc);
/* the head slabs' gradient pass fused for one loss group (nn_lat.cu): upsample + guide + BCE logit gradient + transposed upsample
   without the fp16 slab; part: nn_lat_fuse_part bytes of scratch */
size_t nn_lat_fuse_part(int n, int h, int w, int nz);
void nn_lat_fuse_grad(const void *lg, int lh16, int lq0, int lqd, int d, int h, int w, int z0, int nz, int c0, int n, int guide, int gb, const void *x,
                      const uint8_t *t, const uint8_t *m, const uint8_t *wch, const float *st, float pw, float rec_w, int unit_c, const float *lam,
                      float *part, float *gc, int gq0, int gqd, double *ds);   /* ds: also the group's loss sums 1 .. 4 (after a mask_only statistics pass) */
void nn_lat_fuse_stats(const void *lg, int lh16, int lq0, int lqd, int d, int h, int w, int z0, int nz, int c0, int n, int guide, int gb, const void *x,
                       const uint8_t *t, const uint8_t *m, const uint8_t *wch, float pw, int unit_c, double *ds, int mask_only);   /* nn_loss_stats_acc of the group, fused */
void nn_loss_cfg_get(float *pw, float *rec_w, int *unit7);   /* the BCE's positive weight, --recall weight, 7-channel unit-mask channel */
void nn_lat_grad_carry(float *a, int nf, void *sb, int C, size_t pl, int cap, int from, int n);
void nn_lat_up_adj(const void *g, int C, int d, int h, int w, int z0, int nz, float *gc, void *scratch);   /* scratch: C nz h w fp32 for the separable path (z0, nz multiples of 4), or nullptr */
int nn_lat_set_slow(int on);   /* the per-voxel kernels for every slab (UFSM_LAT_SLOW=1; train's UFSM_LAT_CHECK compares against them); returns the previous setting */
/* predict on a latent-grid net: the network's 4-channel input on the cell grid ([4][D/4][W/4][W/4], 16-bit storage as nn_set_f16) straight
   from a D x W x W uint8 CT window = nn_pred_input_d then nn_lowres_input (div 4, mode 1), without the full-resolution input */
void nn_lat_pred_input(const uint8_t *ct, int D, int W, float mean, float isd, const float *dyo, const float *dxo, int axis, void *y);
/* nn_lat_place_rv for the sub-cell head (lat_guide 64: the side pair's per-voxel outputs at cell channels sb + 64 j + sub; sb < 0: none)
   = nn_lat_up_slab + nn_lat_shuffle + nn_rv_fuse (fp16) + nn_pred_place */
void nn_lat_place_rvs(const void *lg, int lh16, int sb, int d, int h, int w, const uint8_t *ct, const uint32_t *bits, int iso, int hz, int halo,
                      int oz, int oy, int ox, int ez, int ey, int ex, int shard, uint8_t *dsh);
/* nn_lat_place_rv for the guided head (the coefficient planes: cell channels nb .. nb + 4; ctg: the window's CT bytes as read, mean / isd
   its z-score) = nn_lat_up_slab + nn_lat_guide + nn_rv_fuse (fp16) + nn_pred_place */
void nn_lat_place_rvg(const void *lg, int lh16, int nb, int d, int h, int w, const uint8_t *ctg, float mean, float isd, const uint8_t *ct, const uint32_t *bits,
                      int iso, int hz, int halo, int oz, int oy, int ox, int ez, int ey, int ex, int shard, uint8_t *dsh);
void nn_lat_ct_x(const uint8_t *ct, int W, int z0, int nz, float mean, float isd, void *x);   /* planes z0 .. z0 + nz of the fp16 z-scored CT channel [nz][W][W] from the window's bytes */
/* the same for a --lat-basis checkpoint = nn_pred_input_d then nn_lat_input_x (fp16 storage): [C + 3][D/4][W/4][W/4] */
void nn_lat_pred_input_c(const uint8_t *ct, int D, int W, float mean, float isd, const float *dyo, const float *dxo, int axis,
                         const float *bas, const float *gain, int C, float step, void *y);
/* predict --channel rv on a latent-grid net: the two side logits of the cell grid (lg [2][d][h][w], fp16 / fp32) upsampled, fused
   (nn_rv_fuse) and quantised (nn_pred_place) into a device shard: window voxel v of the interior ([hz, 4 d - hz) x [halo, 4 h - halo) x
   [halo, 4 w - halo)) lands at shard voxel o + v when inside [0, e); 0 where the CT (ct >= iso, or ct nullptr: the mask bits) is not set */
void nn_lat_place_rv(const void *lg, int lh16, int d, int h, int w, const uint8_t *ct, const uint32_t *bits, int iso, int hz, int halo,
                     int oz, int oy, int ox, int ez, int ey, int ex, int shard, uint8_t *dsh);
/* train --gpu-aug (src/nn_aug.cu): a window's CT channel built on the device from its source-grid bytes ct8_h (P^3, pinned) and record
   rec_h (scan_augment.h scan_ct_rec: the scanner plan, jitter, normalisation): the scanner ops in 16-bit storage in ct16 (the channel's
   own buffer), the output bytes' z-score, then the symmetry / warp of rad_d (the device copy of the window's NN_RAD_REC record, uploaded
   before on the same stream) with the jitter into ct16. Device scratch: nn_aug_device_bytes(P) (one P^3 byte staging buffer). */
void nn_aug_ct(const uint8_t *ct8_h, const void *rec_h, const void *rad_d, int P, void *ct16, int on_copy_stream);
size_t nn_aug_device_bytes(int P);
void nn_aug_bop_line(int on);   /* tests: 0 = the separable operators through the first kernels (z / y in shared memory, x one row a block) */
void nn_aug_bop_bench(int P, int taps, double *out);   /* tuning: ms per pass, out[6] = z shared / z line / y shared / y line / x one row a block / x rows + weight table */
void nn_aug_cnoise_chunk(int nz);   /* tests: the correlated noise in chunks of nz planes (0: the per-plane kernels, -1: the default chunk) */
void nn_aug_last_bytes(uint8_t *out, int P);   /* tests: the last window's output bytes */
/* train --gpu-aug 2: one warped window's labels warped on the device from the source-grid labels the sampler shipped (sample_cfg.gpu_aug
   2): kind 0 the packed targets / mask a_h -> da (db: the pmask plane, optional), 1 the phase a_h -> da, 2 the winding mod 4 / component a_h, b_h -> da, db; rad_d its
   radial record on the device; cs_h / sn_h sample_phase_cssn's tables. Same stream order as nn_aug_ct (its staging buffer, 2 P^3). */
void nn_label_warp(int kind, const uint8_t *a_h, const uint8_t *b_h, const void *rad_d, int P, uint8_t *da, uint8_t *db, const float *cs_h, const float *sn_h, int on_copy_stream);
size_t nn_label_warp_device_bytes(int P);
/* --rv ink bits: one window's papyrus plane on the device from its CT bytes and known bits (pinned host; sample.h batch.inkb, cutoff it),
   then the window's warp (rad_d, warped) or cube symmetry (symcode: the radial record's h[5]) into da. Same staging / order as nn_label_warp */
void nn_label_sym(const uint8_t *a_h, int symcode, int P, uint8_t *da, int on_copy_stream);   /* a source-grid label plane through the cube symmetry */
/* --gpu-aug 3 runs slots: w4 / comp from the source-grid phase and the window's runs (band_w4dev.h w4runs), warped or through the symmetry */
void nn_w4runs_label(const uint8_t *side_h, const void *runs, const void *rad_d, int warped, int symcode, int P, uint8_t *da, uint8_t *db, const float *cs_h, const float *sn_h, int on_copy_stream);
void nn_ink_bits(const uint8_t *ct8_h, const uint8_t *bits_h, int it, const void *rad_d, int warped, int symcode, int P, uint8_t *da, int on_copy_stream);
/* packed targets (sample.h SAMPLE_TM_*: bit 0 target 0 = 255, bit 1 target 1 = 255, bit 2 mask) of nb samples of p3 voxels in m ->
   t (nb x 2 x p3: 0 / 255) and the 0 / 1 mask in m, in place; on the copy stream (after an nn_h2d_copy_stream of the codes) or the
   compute stream */
void nn_unpack_tm(uint8_t *t, uint8_t *m, int nb, size_t p3, int on_copy_stream);
void nn_f32_to_h16(const float *x, size_t n, void *y, float scale);
/* 16-bit type for activation/gradient storage and MMA operands: 0 bf16 (default), 1 fp16 (8x finer mantissa, same tensor-core rate;
   env UFSM_F16=1). With fp16 gradient storage the activation gradients are kept scaled by nn_get_grad_scale() (env UFSM_GSCALE,
   default 1024) and the network divides the parameter gradients by it. The fp8/fp4 kernels require the bf16 setting. */
void nn_set_f16(int on);
int nn_get_f16(void);
void nn_set_grad_scale(float s);
void nn_set_logits_h16(int on);     /* the losses, nn_prob_u8, nn_pred_output / nn_pred_place* and nn_fit_accum_d read fp16 logits (unet_logits_h16) */
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
int nn_head_bwd_x(const float *x, const nn_gn_t *gx_gn, shape5 xs, const float *gy, shape5 ys, const float *w, float *gw, float *gb, float *gx);
/* the head on voxels [vb, vb + nv) of x (trainer head slabs): fp16 logits / logit gradient as planes of nv voxels; bwd: first = the first
   slab of a backward (draws the SR seed the others reuse). -1: not covered (MX x / gx, 16-bit gradients) */
int nn_head_fwd_z(const float *x, const nn_gn_t *gx_gn, shape5 xs, const float *w, const float *b, int cout, void *y16, size_t vb, size_t nv);
int nn_head_bwd_z(const float *x, const nn_gn_t *gx_gn, shape5 xs, const void *gy16, int cout, const float *w, float *gw, float *gb, float *gx, size_t vb, size_t nv, int first);   /* fused 1^3 head backward (MX x / gx, fp16 gy); -1 = not covered */
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
/* the same for the output x (MX-fp4, 8 / 16 channels) of the block the head reads, gy being the head's input gradient: recomputed per
   voxel from the fp16 logit gradient gl (cout channels, head weights wh [cout][C]) as the head's backward-data with SR seed hs would
   have stored it, gx written in place over x (the bits of the two-op path, the gradient never stored); -1 not covered
   (nn_gn_silu_bwd_head_ok). nn_head_bwd_x_draws: nn_head_bwd_x draws an SR seed before declining such inputs */
int nn_gn_silu_bwd_head_ok(int C, int cout);
int nn_gn_silu_bwd_head(float *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd, const void *gl, int cout,
                        const float *wh, unsigned hs, float *ggamma, float *gbeta, float *scratch);
/* the same over the pseudo-sample shape of slab-local GroupNorm (s.n = samples x ns z-slabs of s.d planes, statistics [n][ns][G]; the
   logit gradient gl stays per sample); ns 1: nn_gn_silu_bwd_head */
int nn_gn_silu_bwd_head_ns(float *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd, const void *gl, int cout,
                           const float *wh, unsigned hs, float *ggamma, float *gbeta, float *scratch, int ns);
int nn_head_bwd_x_draws(void);
/* level-0 normalisation alternatives (unet cfg.l0_norm, src/nn_l0norm.cu): while nn_set_gn_const is on, the GroupNorm backwards
   (nn_gn_bwd, nn_gn_silu_bwd, nn_gn_silu_bwd_head) treat the statistics as constants (no mean / variance gradient terms);
   nn_gn_const_fill writes constant statistics for N samples (run nullptr: mean 0, rstd 1; else run = [mean G | var G]);
   nn_gn_ema_update moves run toward the batch's statistics (mean, rstd per sample; mom 1: replaced) */
void nn_set_gn_const(int on);
/* slab-local GroupNorm (unet cfg.l0_norm slab, kernel path): a statistics array registered here is per (sample, z-slab) [n][ns][G],
   plane z in slab (z + zo) / zs, zhw voxels per plane; the readers given it (GroupNorm + SiLU in their staging) apply each plane's
   slab. mean nullptr: clear the registry; ns <= 0: remove mean's entry. (nn_lp.h: gnp_mk / the gnp_t fields) */
void lp_gn_slab_reg(const float *mean, int zs, int zo, int ns, int zhw);
int nn_get_gn_const(void);
void nn_gn_const_fill(float *mean, float *rstd, const float *run, int N, int G, float eps);
void nn_gn_ema_update(float *run, const float *mean, const float *rstd, int N, int G, float mom, float eps);
/* scaled weight standardisation (cfg.l0_norm 3) of cout output channels with n contiguous fan-in weights each: wh = gain (w - mean) /
   (std sqrt(n) + eps); nn_ws_bwd: g <- g0 + (the gradient w.r.t. w of the part g - g0 w.r.t. wh) */
void nn_ws_fwd(float *wh, const float *w, int cout, int n, float gain, float eps);
void nn_ws_bwd(float *g, const float *g0, const float *w, int cout, int n, float gain, float eps);
/* cfg.l0_norm 4 (slab): an MX tensor's GroupNorm affine in place, x <- (x - mean) rstd per (sample, group) (mean nullptr: 0; exact SR
   keyed by sr for fp4); the fp32 path's permutation NCDHW <-> (n * D / Sz + slab, C, Sz, H, W) (dir 0: to the slab layout, 1: back) */
void nn_l0_affine(float *x, shape5 s, int G, const float *mean, const float *rstd, unsigned sr);
void nn_slab_perm(float *dst, const float *src, shape5 s, int Sz, int dir);
/* GroupNorm forward fused with SiLU (y = silu(gn(x))), and recompute of both g = gn(x) and silu(g). */
int nn_gn_stats(const float *x, shape5 s, int G, float eps, float *mean, float *rstd);   /* stats of the stored fp32, 16-bit or MX tensor; -1 for unsupported groups */
int nn_gn_stats_n(const float *x, shape5 s, int G, float eps, float *mean, float *rstd, int n0, int cnt);   /* samples [n0, n0 + cnt) only (MX), nn_gn_stats's bits */
void nn_gn_fwd_silu(const float *x, shape5 s, int G, float eps, const float *gamma, const float *beta, float *y, float *mean, float *rstd);
void nn_gn_apply_silu(const float *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd, float *g, float *sil);
/* Recompute y = gn(x) from saved mean/rstd (no statistics pass). */
void nn_gn_apply(const float *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd, float *y);

/* ---- elementwise ---- */
void nn_silu_fwd(const float *x, size_t n, float *y);
void nn_silu_bwd(const float *x, const float *gy, size_t n, float *gx);   /* gx SET */
void nn_axpy(float *y, float a, const float *x, size_t n);               /* y += a x */
/* unet cfg.film (src/nn_film.cu): a GroupNorm's effective gamma / beta (offsets gam / bet, C channels) from its K x C pairs A, D at fo and the
   K host codes, written into fw; the backward turns the effective pair's gradient in g into gamma's, A's and D's */
void nn_film_fwd(const float *p, float *fw, size_t gam, size_t bet, size_t fo, int C, int K, const float *codes);
void nn_film_bwd(const float *p, float *g, size_t gam, size_t bet, size_t fo, int C, int K, const float *codes);
/* the Gram of K planes of 16-bit values (bf16 1, else fp16) with row stride ld over N entries: S[i K + j] += sum_n x_i x_j for j >= i (src/nn_gram.cu; train --grad-cos) */
void nn_gram16(const void *x, int bf16, int K, size_t N, size_t ld, double *S);
/* the fine voxel stem (src/nn_fstem.cu, train --fstem): params W1 [C][64] (4^3 taps at stride 2), b1 [C], W2 [nout][C]; fwd adds W2 silu(W1 x + b1)
   to the fp16 logit planes chs[o] (slab planes [z0, z0 + nz) of a D x H x W window, ns voxels a plane) on each 2^3 block; bwd accumulates the
   parameter gradients from fp16 gradient planes of that layout into gp. 1: a shape the kernels do not take (odd sizes, C or nout > 16) */
int nn_fstem_fwd(const void *x, int D, int H, int W, int z0, int nz, const float *p, int C, int nout, const int *chs, void *lg, size_t ns);
int nn_fstem_bwd(const void *x, int D, int H, int W, int z0, int nz, const float *p, int C, int nout, const int *chs, const void *g, size_t ns, float *gp);
/* a latent-grid net's head straight from the cell logits lg [cout][d][h][w] into a device shard (src/nn_lathead.cu, predict --heads): the
   window interior ([hz, 4d - hz) x [halo, 4w - halo)^2) inside the shard [0, e) at offset o; channel c0 (a probability) or the recto / verso
   pair c0, c1 (nn_rv_fuse); sb0 / sb1 the sub-cell outputs' base of each (lat_guide 64: 16 + 64 j for the phase pair, else -1), s0 / s1 the
   fine stem's output row of each (-1 none; stem null: no stem); ct the window's CT bytes (mask CT >= iso, stem taps) */
void nn_lat_head_place(const void *lg, int lh16, int d, int h, int w, int c0, int c1, int sb0, int sb1, int s0, int s1,
                       const float *stem, int C, int nout, const uint8_t *ct, float mean, float isd, int iso,
                       int hz, int halo, int oz, int oy, int ox, int ez, int ey, int ex, int shard, uint8_t *dsh);
/* nh (<= 24) heads at once, head j (c0[j], c1[j], sb0[j], sb1[j], s0[j], s1[j]) into its shard dsh[j]: the stem's features computed once a cell;
   byte-identical to one nn_lat_head_place per head */
void nn_lat_heads_place(const void *lg, int lh16, int d, int h, int w, int nh, const int *c0, const int *c1, const int *sb0, const int *sb1, const int *s0, const int *s1,
                        uint8_t *const *dsh, const float *stem, int C, int nout, const uint8_t *ct, float mean, float isd, int iso,
                        int hz, int halo, int oz, int oy, int ox, int ez, int ey, int ex, int shard);
/* predict's device-resident CT window (src/nn_ctdev.cu): a reader worker's stream + device / host chunks; nn_ctw_put gathers one decoded
   chunk (device substreams dm, the host chunk too when host) into the window [wo, wo + wn) clipped to the array shape sh, and waits */
void *nn_ctw_new(int dev);
void nn_ctw_free(void *w);
uint8_t *nn_ctw_host(void *w);
uint8_t *nn_ctw_dev(void *w);
int nn_ctw_put(void *w, uint32_t dm, int host, const int64_t co[3], const int64_t wo[3], const int64_t wn[3], const int64_t sh[3], uint8_t *dwin);
int nn_ctw_zero(void *w, uint8_t *d, size_t n);
/* one 2x mean pooling step of an n^3 uint8 volume into (n / 2)^3: (sum + 4) >> 3, as the CPU pyramid (ingest.c pool2_mean) */
void nn_pool2_u8(const uint8_t *in, int n, uint8_t *out);
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
void nn_set_loss_unit_ch7(int c);   /* 7-channel losses (task unwrap normal n n^T + density): channel c's mask weight counts as 0 / 1 (the packed weight off); -1 off */
void nn_set_recall(float w);
int nn_get_loss_grad_h16(void);   /* the loss kernels' gradient format (nn_set_loss_grad_h16) */
void nn_pool_targets(const uint8_t *t, const uint8_t *m, int B, int C, int D, int H, int W, int f, uint8_t *to, uint8_t *mo);   /* deep supervision: max-pool masked targets, any-pool the mask (C <= 16) */   /* recto channel: + w (1 - (sum s p + 1) / (sum p + 1)), gradient only (the logged dice excludes it) */
int nn_get_loss_tol(void);
void nn_loss_async_tol(const float *logits, const uint8_t *t, const uint8_t *m, const uint8_t *w, shape5 s, float dice_w, float *gl, float *scratch, uint8_t *code);
/* --half-res pack: the loss of the full-resolution window (s, even sides) from the network's packed logits ([n][8 s.c][half grid],
   fp16 / fp32 as nn_set_logits_h16), the gradient in that layout (gl may be the logits: in place; nullptr: statistics only); bit
   for bit nn_half_output + nn_loss_async_tol (no tolerance band) + nn_half_grad, without full-resolution logits. t nullptr: m holds
   the packed target codes (nn_unpack_tm's input; s.c <= 2), read as nn_unpack_tm would unpack them */
void nn_loss_pack(const void *logits, const uint8_t *t, const uint8_t *m, const uint8_t *w, shape5 s, float dice_w, void *gl, float *scratch);
/* the same loss in parts (head slabs): statistics added to ds (5 doubles per (n, c), caller-zeroed) per slab, finished into scratch
   (s: the whole window), then each slab's gradient (gl may be the logits: in place) */
void nn_loss_stats_acc(const void *logits, const uint8_t *t, const uint8_t *m, const uint8_t *w, shape5 s, double *ds);
void nn_loss_stats_fin(double *ds, const uint8_t *w, shape5 s, float *scratch);
void nn_loss_grad_only(const void *logits, const uint8_t *t, const uint8_t *m, const uint8_t *w, shape5 s, float dice_w, void *gl, const float *scratch);
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
/* train --fib-normal-lambda (src/nn_loss.cu): lambda x mean t^T N t at fibre voxels; N the sign-free normal dyad of the six normal logits
   at lg (planes of ns, the slab's voxels; fp16 / fp32 as nn_set_logits_h16), t the tube axis of the fibre plane fib (prob254, 255 unknown,
   indexed by window voxel; D H W the window) from its gradient structure tensor over (2r + 1)^3, at voxels >= thr whose stencil lies in
   the slab planes [z0, z1), and (nm non-null: planes of ns, the normal loss mask) only where nm is 0. mode 0: acc[0] += valid voxels, acc[1] += t^T N t; mode 1: their per-voxel gradients d(t^T N t) / dlogit into
   list (cap entries, *nl the count: zero it per slab). nn_fibn_apply adds lambda / acc[0] x the list into the gradient gl (the losses'
   storage format, planes of ns: call after the normal loss wrote it) */
size_t nn_fibn_cap(size_t ns);
size_t nn_fibn_list_bytes(size_t cap);
void nn_fibn(const void *lg, size_t ns, const uint8_t *fib, int D, int H, int W, int z0, int z1, int r, int thr, const uint8_t *nm, double *acc, void *list, unsigned *nl, size_t cap, int mode);
void nn_fibn_apply(const void *list, const unsigned *nl, size_t cap, const double *acc, float lambda, void *gl, size_t ns);
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
/* Muon variants over the same table (train --muon-c / --hyperball; see nn_opt.cu). mcd: nullptr (the Muon direction) or a device
   array of nconv {const float *U; float *O; float *EX, *EY; int Co, Ci, k, m, n, tr;} (U = the entry's X, O = its Y; EX, EY:
   nf 2 m n floats each (complex m x n blocks), nf = 14 for k = 3 (1 for k = 1), m = min(Co, Ci), n = max(Co, Ci), tr = Co > Ci)
   with nsd the matching Newton-Schulz table of nconv nf entries, conv-major {.., X = EX + f 2mn, Y = EY + f 2mn, A, B (2 m^2
   floats each), Co = m, K = n}, k the kernel size of every entry (3 or 1), ns_maxm / ns_maxn the largest m / n. hb: nullptr or
   nconv device flags (1: the Hyperball step with radius R[i], eta = hb_lr, no decay); r_init: first set R[i] = |W_i|_F where
   R[i] <= 0. part: >= 32 nconv (2 + nf) doubles of scratch. */
void nn_muon_ext_batch(const void *descs, int nconv, int maxco, int maxk, const void *mcd, const void *nsd, int k, int ns_maxm, int ns_maxn,
                       const int *hb, float *R, int r_init, double *part, float lr, float beta, float wd, float hb_lr);
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
void nn_fill_unlabelled(uint8_t *t, uint8_t *m, const uint8_t *vm, const uint8_t *src, size_t n);
unsigned long long nn_count_nonfinite(const void *p, size_t n, int h16);   /* diagnostics (synchronous): non-finite values among n fp32 (h16 0) / fp16 (1) / bf16 (2) */
void nn_add_into(void *y, const void *x, shape5 s, int h16);   /* y += x in y's storage (MX by registration, else 16-bit or fp32) */
void nn_fakeq_w4_2d(float *w, int Co, int Ci, int T, int B);   /* w[Co][Ci][T] in place to MX-fp4 values, a scale per B x B block and tap
                                                                   (unet_cfg.w4_2d: forward and backward-data then quantise it identically) */
void nn_fakeq_w4(float *w, int Co, int Ci, int T, int Bo, int Bi);   /* the same with Bo x Bi blocks (tests: 1 x 16 / 32 = the forward's weights) */
void nn_add_into_act(void *y, const void *x, shape5 s, int h16);   /* the same for a forward activation: MX-fp4 rounded to nearest (no SR) */
void nn_side_targets(const uint8_t *side, size_t n, uint8_t *t, uint8_t *m);   /* band.h SIDE_* labels -> loss target (recto 255, verso 0) and mask (known) */   /* where m == 0 and vm != 0: t = src, m = 1 */
void nn_phase_targets(const uint8_t *phase, size_t n, uint8_t *t, uint8_t *m);   /* band.h PHASE_* -> targets t[0..n) = 0.5 + 0.5 cos(2 pi phase), t[n..2n) = 0.5 + 0.5 sin (x255), mask (known) */
void nn_rv_targets(const uint8_t *phase, const uint8_t *pap, size_t n, uint8_t *t, uint8_t *m);   /* train --rv: recto / verso side papyrus targets t[0..2n) and mask from the phase + papyrus plane */
void nn_rv_verso_w(const uint8_t *t, uint8_t *m, size_t n, float w);   /* train --rv-verso-w: verso-side papyrus loss weight x w */
void nn_rv_fuse(void *lg, size_t n, int h16);   /* predict --channel rv: [recto | verso] logits -> channel 0 = logit((1 + p_r - p_v) / 2) */
void nn_ink_targets(const uint8_t *ink, size_t n, uint8_t *t, uint8_t *m);   /* train --ink-head: ink prob254 (255 unknown) -> target t (x255) and mask m (known) */
void nn_phase_trunc(const void *lc, const void *ls, const uint8_t *t, uint8_t *m, size_t n, float tau, float kappa);   /* robust phase loss: confident (>= kappa) predictions off their label by > tau turns leave the mask (fp16 logits) */
void nn_phase_trunc_counts(unsigned long long out[2]);   /* labelled voxels / dropped since the last call */
void nn_unwrap_targets(const uint8_t *phase, int d, int h, int w, int s, float dscale, uint8_t *t, uint8_t *m);   /* task unwrap: sign-free normal n n^T (6) and density min(1, dscale rho) (x255, 7 planes) from the phase gradient over +-s voxels; mask */
/* the same with the density code (0: min(1, x), 1: x / (1 + x), x = dscale rho: never saturates) and the mask holding the packed-sheet
   loss weight (1 at a spacing um / rho of packed_um or more, up to packed_w at packed_um / 2; um: the batch's voxel size in um) */
void nn_unwrap_targets2(const uint8_t *phase, int d, int h, int w, int s, float dscale, int code, float um, float packed_um, float packed_w, uint8_t *t, uint8_t *m);
void nn_packed_mask(const uint8_t *phase, int d, int h, int w, int s, float um, float packed_um, float packed_w, uint8_t *m);   /* 0/1 mask x the weight */
void nn_packed_mask_z(const uint8_t *phase, int d, int h, int w, int s, float um, float packed_um, float packed_w, uint8_t *m, int z0, int z1);   /* m: planes [z0, z1) */
/* train --fib-dir: the axis of the H / V label tubes (t: 2 planes, >= 128 = fibre; m: shared mask) -> 7 target planes (n n^T coding of the
   axis, family V) and a 0/1 mask, over the (2r + 1)^3 second moment of the voxel's family tube; lambda_1 / trace >= cmin (src/nn_elem.cu) */
void nn_fib_dir_targets(const uint8_t *t, const uint8_t *m, int d, int h, int w, int r, float cmin, uint8_t *o, uint8_t *om);
/* train --fib-head: fib_dir targets of window planes [z0, z0 + nz) from the H / V tube planes (prob254, 255 unknown; window-indexed, +-r halo planes present) */
void nn_fib_head_dir(const uint8_t *fh, const uint8_t *fv, int d, int h, int w, int z0, int nz, int r, float cmin, uint8_t *o, uint8_t *om);
void nn_unwrap_targets2_z(const uint8_t *phase, int d, int h, int w, int s, float dscale, int code, float um, float packed_um, float packed_w, uint8_t *t, uint8_t *m,
                          int z0, int z1);   /* planes [z0, z1) of the window into planes of (z1 - z0) h w */
/* train --count-lambda: K segments of L voxels along the ground-truth phase gradient; loss |log((sum rho_pred + .05) / (true turns + .05))|
   (rho_pred = exp(logit) / dscale, the x / (1 + x) density code); its gradient times lambda added into dens_grad (the density
   channel of the logit gradient; nullptr: none). acc: d h w floats (zero, left zero); seg: 4 K floats; out2 (device, 21 floats): mean loss,
   valid segments, mean predicted / true turns, then per --count-strata spacing bin b < 9 the sums of predicted and true turns (3 + 2 b, 4 + 2 b) */
/* train --count-strata E1,..,En (um): stratify the count segments by GT sheet spacing: segment k is drawn from bin k mod (n + 1) of the
   edges (converted to voxels with um, the window's voxel size; call before each nn_count_loss); n 0 = off (uniform starts) */
void nn_count_strata(int n, const float *edges_um, float um);
void nn_count_loss(const void *dens_logits, const uint8_t *phase, int d, int h, int w, int s, int K, int L, float dscale, uint64_t seed,
                   float lambda, void *dens_grad, float *acc, float *seg, unsigned *nvalid, float *out2);
/* train --pgrad-lambda: the phase wavenumber loss over a head slab [z0, z1) (src/nn_loss.cu pgrad_k): lc / ls the slab's phase cos / sin
   logits, phase the window's codes, wm the slab's phase mask (weights); sums (2 device doubles: sum w loss, sum w) accumulate; acc
   (2 x slab floats, zero; nullptr: statistics only) the logit gradient, which nn_pgrad_apply adds times lambda / sum w into gc / gs
   (16-bit gradients) and zeroes; lc = ls = nullptr: the pair weights only (sums[1] += sum w, no logits read: the training stats pass) */
void nn_pgrad_slab(const void *lc, const void *ls, const uint8_t *phase, const uint8_t *wm, int d, int h, int w, int z0, int z1, int s, const float *dist,
                   int nd, int angle, int mask, double *sums, float *acc);   /* dist: nd (1..4) partner distances, each its own pairs; angle: angular gradient only; mask: reliable label pairs only */
void nn_pgrad_apply(void *gc, void *gs, float *acc, size_t ns, const double *sums, float lambda);
void nn_cond_targets(const uint8_t *wind, size_t n, uint8_t *t, uint8_t *m);   /* --cond: winding mod 14 (band.h band_winding) -> 4 targets: cos / sin of 2 pi q / 14 and 2 pi q / 2 (x255, 4 planes), mask (known) */
void nn_cond_acc(const void *lg, int lg_h16, const uint8_t *wind, size_t n, unsigned long long *cnt);   /* --cond: known / right-turn counts of the 14-turn head (cnt: 2 device u64, accumulated) */
void nn_pap_targets(const uint8_t *pl, const uint8_t *pm, size_t n, uint8_t *t, uint8_t *m);   /* --pap-head: gap labels (0 / 1 / 255 ignored) -> target, mask (not ignored, inside the scroll) */
/* --task unwrap2: the gauge-free winding mod 4 loss (src/nn_loss.cu). la / lb: the cos / sin logits (fp32, or fp16 under
   nn_set_logits_h16), n voxels; w4 (W mod 4, 63 codes per turn, 255 unknown), comp (component 0..254, 255 none: one gauge each), m
   (mask weight). Per component the soft BCE against the target rotated by the best of the four quarter turns (the gauge); ds:
   nn_wind4_sums() bytes (zeroed by the caller, accumulated over slabs by nn_wind4_stats), fin: nn_wind4_scratch() bytes ([0] loss,
   [1] turn accuracy, [2] voxels, [3] 1 / voxels, [4] components, [5] worst-gauge loss, [8 + c] the gauges). nn_wind4_grad writes
   lambda x the gradient of fin[0] (zero where unsupervised; ga / gb may be la / lb) */
size_t nn_wind4_scratch(void);
size_t nn_wind4_sums(void);   /* bytes of the ds sums (doubles) */
void nn_wind4_stats(const void *la, const void *lb, const uint8_t *w4, const uint8_t *comp, const uint8_t *m, size_t n, double *ds);
void nn_wind4_fin(double *ds, float *fin);
void nn_wind4_grad(const void *la, const void *lb, const uint8_t *w4, const uint8_t *comp, const uint8_t *m, size_t n, const float *fin, float lambda, void *ga, void *gb);
void nn_wind4_loss(const void *la, const void *lb, const uint8_t *w4, const uint8_t *comp, const uint8_t *m, size_t n, double *ds, float *fin, float lambda, void *ga, void *gb);
void nn_cond_gather(const uint8_t *c14, const uint8_t *done, const uint8_t *ct, int W, int oz, int oy, int ox, int bz, int by, int bx, size_t N, void *x, int h16);   /* predict --cond-sweep: condition inputs 4-6 of a window from the box canvas (src/sweep.c) */
void nn_cond_place(const float *lg, const uint8_t *ct, int W, int halo, int oz, int oy, int ox, int bz, int by, int bx, size_t N, int ch14, uint8_t *c14, uint8_t *p1, uint8_t *recto, uint8_t *done);   /* the window interior into the box canvas where not done */
/* train --distill: cosine feature distillation. y the head's fp32 outputs [n][C][S], t the int8 targets in the same layout (a voxel whose
   C target values are all 0 has none); g (fp32, same layout) = w x the gradient of the mean over the voxels with a target of 1 - cos(y, t);
   ds (2 device doubles): the sum of 1 - cos and the voxel count */
void nn_distill_loss(const float *y, const int8_t *t, int n, int C, size_t S, float w, float *g, double *ds);
/* half-resolution networks (unet_cfg.half_res): the network runs on 2x2x2 blocks of the full-resolution window.
   nn_half_input: the 4-channel input x ([n][4][D][H][W], 16-bit storage type when h16, else fp32) -> the network input y at half
   resolution, same type: mode 1 (pool) the block mean of every channel (4 channels); mode 2 (pack) those 4 means plus, from the block's
   8 CT voxels, the first-order Haar details along z, y, x ((upper half - lower half) / 8) and the mean absolute deviation from the
   block mean (8 channels: the stem kernels take at most 8).
   nn_half_output: half-resolution logits lg (fp16 when lh16, else fp32) -> full-resolution fp32 logits y [n][hs.c][2d][2h][2w]: pool
   ([n][hs.c][d][h][w] logits) trilinear 2x (fine voxel f samples (f + 0.5) / 2 - 0.5, edge-clamped); pack ([n][8 hs.c][d][h][w]) channel
   8 c + k is sub-voxel k of channel c. nn_half_grad: the adjoint, full-resolution gradient g -> half-resolution gh (16-bit storage
   type when g16, else fp32). hs: the half-resolution shape with c = the full-resolution output channels. */
void nn_half_input(const void *x, shape5 s, int mode, void *y, int h16);
void nn_half_output(const void *lg, int lh16, shape5 hs, int mode, float *y);
void nn_half_grad(const void *g, int g16, shape5 hs, int mode, void *gh);
/* the same at div 2 or 4 (unet_cfg.half_res 3 / 4: quarter resolution): div 4 first pools 2x (input) / ends with a trilinear 2x
   (logits), mode as above on the remaining 2x. qs: the network's grid, c = the full-resolution output channels. tmp (div 4): the
   intermediate grid (full / 8 voxels) for 4 input channels in the input's type, or qs.c channels of fp32 logits / the gradient type */
void nn_lowres_input(const void *x, shape5 s, int div, int mode, void *tmp, void *y, int h16);
void nn_lowres_output(const void *lg, int lh16, shape5 qs, int div, int mode, float *tmp, float *y);
void nn_lowres_grad(const void *g, int g16, shape5 qs, int div, int mode, void *tmp, void *gh);
/* inference: network input from the uint8 CT window on the device (z-score + radial channels; 16-bit storage when h16),
   and recto probability * 255 (0 where the CT is 0) from the logits */
void nn_pred_input(const uint8_t *ct, int W, float mean, float isd, const float *dyo, const float *dxo, int axis, void *x, int h16);
void nn_aux_input(const uint8_t *a, size_t n, void *x, int h16);   /* the aux input channel: v / 127.5 - 1 in the input format */
void nn_pred_output(const float *lg, const uint8_t *ct, size_t n, uint8_t *out);
void nn_pred_input_mx(const uint8_t *ct, int W, float mean, float isd, const float *dyo, const float *dxo, int axis, int z0, int nz, void *y, int h16, int bits);   /* the same, MX-fp4 (bits 4) or MX-fp8 (8) */
void nn_pred_input_mx8(const uint8_t *ct, int W, float mean, float isd, const float *dyo, const float *dxo, int axis, int z0, int nz, void *y, int h16);   /* nn_pred_input of planes [z0, z0 + nz) straight to MX-fp8 (depth-nz tensor), bit-identical to nn_pred_input + the input conversion */
void nn_pred_place(const float *lg, const uint8_t *ct, int W, int halo, int oz, int oy, int ox, int ez, int ey, int ex, int shard, uint8_t *dsh);   /* interior -> device shard */
/* the same for a D x W x W window (slabs: predict --window-z), interior planes [hz, D - hz); nn_pred_place = D W, hz halo */
void nn_pred_input_d(const uint8_t *ct, int D, int W, float mean, float isd, const float *dyo, const float *dxo, int axis, void *x, int h16);
void nn_pred_place_d(const float *lg, const uint8_t *ct, int D, int W, int hz, int halo, int oz, int oy, int ox, int ez, int ey, int ex, int shard, uint8_t *dsh);
void nn_pred_place_m_d(const float *lg, const uint32_t *bits, int D, int W, int hz, int halo, int oz, int oy, int ox, int ez, int ey, int ex, int shard, uint8_t *dsh);
void nn_pred_stats(const uint8_t *ct, size_t n, void *scratch, size_t *nz, double *sum, double *sq);   /* exact window stats (scratch: 24 B device) */
void nn_pred_ctmask(const uint8_t *ct, size_t n, uint32_t *bits);   /* CT != 0 as bits ((n + 31) / 32 words) */
void nn_pred_ctmask_t(const uint8_t *ct, size_t n, uint32_t *bits, int t);   /* predict --iso: bits of CT >= t */
void nn_set_pred_iso(int t);
int nn_get_pred_iso(void);   /* predict --iso: the CT window path places outputs only where CT >= t */
void nn_pred_place_m(const float *lg, const uint32_t *bits, int W, int halo, int oz, int oy, int ox, int ez, int ey, int ex, int shard, uint8_t *dsh);   /* nn_pred_place with the CT mask as bits */
void nn_pred_place_mz(const float *lg, const uint32_t *bits, int W, int halo, int zlo, int zhi, int zoff, int oz, int oy, int ox, int ez, int ey, int ex, int shard, uint8_t *dsh);   /* the same from a split half: lg / bits hold window planes [zoff, ..), window planes [zlo, zhi) placed */

#ifdef __cplusplus
}
#endif
