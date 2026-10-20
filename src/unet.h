/* Tiny 3-D U-Net on top of nn.h: `nlev` levels with widths w[i]; per level two 3^3 convs each followed
   by GroupNorm(G) + SiLU; stride-2 3^3 conv down; trilinear 2x up + channel concat; 1^3 head.
   Parameters, gradients, Adam moments and the EMA copy live in flat device arrays. */
#pragma once
#include <math.h>
#include "nn.h"
#include <stdint.h>

#define UNET_MAXLEV 8
#define UNET_MAXCONV 160   /* 3^3 convs (encoder, down, decoder, residual blocks) */

typedef struct {
    int nlev;
    int widths[UNET_MAXLEV];
    int cin, cout;
    int G;                 /* GroupNorm groups (min(G, C) is used) */
    int down_norm;         /* 1: GroupNorm + SiLU after each stride-2 down conv (params appended after the head; 0 keeps the old layout) */
    int scale_cond;        /* 1: voxel-size conditioning: the stem conv bias is b + c * v, c = unet_scale_code(input voxel size), v a learned
                              per-channel vector (appended last, zero-initialised: a donor without it is unchanged) */
    int dec_widths[UNET_MAXLEV];   /* decoder output channels per level (0 = widths[i]); <= widths[i] (multiples of 32 for MX storage), level 0 = widths[0]:
                                      a narrower coarse decoder upsamples fewer channels into the level below (dec[i].c1 reads
                                      dec_widths[i+1] + widths[i]) */
    int enc_blocks[UNET_MAXLEV];   /* extra pre-activation residual blocks after encoder level i (levels >= 1; 0 = none): x_k = x_{k-1} +
                                      conv2(silu(gn(conv1(silu(gn(x_{k-1})))))), widths[i] -> widths[i]; the last one's output is the level's
                                      skip and down-conv input. Parameters appended last; a fresh block starts as the identity (zero conv2) */
    int dec_convs[UNET_MAXLEV];   /* convs per decoder level (0 or 2: two; 1: one, nnU-Net / ResEnc style): a one-conv level's output is
                                     silu(gn1(conv1([up | skip]))), written straight into its a2 buffer (n2 = n1); no a0_share / slab aliasing */
    int ds_levels;         /* deep supervision: a 1^3 head (cout outputs) on each decoder level 1..ds_levels (params appended last); a training
                              forward computes their fp32 logits (unet_ds_logits), the trainer's loss gradients come back through
                              unet_set_ds_grad and join each level's gradient before its decoder backward */
    int half_res;          /* the caller runs the network on 2x2x2 blocks of its window (nn_half_*): 1 pool (cin 4, the block means; logits
                              upsampled trilinearly), 2 pack (cin 8: the block means, the CT's sub-voxel gradients and contrast;
                              cout 8 per output, one per sub-voxel); 3 pool4, 4 pack4: the network on 4^3 blocks (a 2x pool / trilinear 2x
                              around pool / pack). The network itself does not use it; train / predict apply the transforms (nn_lowres_*) */
    int w4_2d;             /* consistent fp4 weights (train --w4-2d 1): the stride-1 3^3 conv weights (not the fp8 stem, not the stride-2
                              downs) round to MX-fp4 with one scale per 32 x 32 (output x input channel) block and tap before every forward
                              (nn_fakeq_w4_2d into the forward copy), so the forward's per-input-channel blocks and backward-data's per-
                              output-channel blocks re-quantise them exactly: both passes see the same weights (the NVFP4 report's 2D
                              weight scaling, arXiv 2509.25149); the weight gradient goes to the fp32 master weights unchanged */
    int l0_norm;           /* the level-0 normalisation (enc[0] and dec[0]: their four GroupNorms, the head's input one included; down
                              norms stay GroupNorm): 0 GroupNorm, 1 none (gamma, beta + SiLU: statistics mean 0, rstd 1), 2 ema
                              (BatchNorm-style running statistics per group, constants in forward and backward once unet_set_l0_ema's
                              warm-up on live statistics is over; saved in checkpoints), 3 ws (none + scaled weight standardisation of
                              the four level-0 3^3 convs, per output channel over its fan-in, from the master weights every forward;
                              the convs reading SiLU outputs with the NF-net gain 1 / std(silu(z))), 4 slab (GroupNorm statistics per
                              z-slab of l0_slab planes: no whole-window reduction). Tensor-core and fp32 paths (slab: MX or fp32 storage) */
    int l0_slab;           /* l0_norm 4: planes per slab (0: 64; the level-0 depth a multiple of it) */
} unet_cfg;
static inline int unet_hr_div(int hr) { return hr >= 3 ? 4 : hr ? 2 : 1; }      /* the window voxels per network voxel along each axis */
static inline int unet_hr_mode(int hr) { return hr == 2 || hr == 4 ? 2 : 1; }   /* the last stage: 1 pool, 2 pack (nn_half_*) */

typedef struct unet unet;

unet *unet_create(const unet_cfg *cfg);
/* scale code of an input voxel size (um): log2(um / 4.8), so 2.4 um -> -1, 9.6 um -> +1 */
static inline float unet_scale_code(double um) { return um > 0 ? (float)log2(um / 4.8) : 0.f; }
/* input voxel size of the following forward passes (scale_cond nets; one value per forward, i.e. per batch) */
void unet_set_scale(unet *u, double um);
/* the conditioning code c itself (scale_cond nets used for another scalar, e.g. task papyrus: the cutoff in input units) */
void unet_set_cond(unet *u, float c);
void unet_free(unet *u);
const unet_cfg *unet_cfg_of(const unet *u);
size_t unet_nparams(const unet *u);
void unet_head_offsets(const unet *u, size_t *w, size_t *b);   /* the 1^3 head's weights (cout x its input channels) and biases (cout) in the flat parameters */
const float *unet_live_params(const unet *u);   /* device weights the optimiser trains (p) */
void unet_use_params(unet *u, const float *w);  /* forward with these device weights (unet_nparams floats); nullptr restores */
void unet_grad_head_only(unet *u, int blocks);   /* zero all gradients but the output head's (and the last decoder blocks') */
void unet_init(unet *u, uint64_t seed);                 /* Kaiming-normal convs, GN gamma 1 / beta 0, head bias -2 */
void unet_zero_head(unet *u);   /* zero the output head (live / EMA weights, Adam moments): a fresh residual net outputs its input */

/* Forward for one input shape (activation buffers are (re)allocated when the shape changes).
   train = 1 keeps everything needed for backward. Logits: [n][cout][d][h][w] on the device. */
const float *unet_forward(unet *u, const float *x, shape5 xs, int train);
/* x_h16: x is already the 16-bit storage type (fp16 with nn_set_f16, else bf16; not MX) and is read in place (no copy) */
const float *unet_forward_x(unet *u, const void *x, shape5 xs, int train, int x_h16);
shape5 unet_out_shape(const unet *u, shape5 xs);
/* Backward from d loss / d logits (device). Gradients ACCUMULATE into the flat grad array. */
void unet_backward(unet *u, const float *glogits);
void unet_debug_stats(const unet *u);
void unet_debug_acts(const unet *u);    /* stderr: per-block stored activations, non-finite counts and extremes per sample */   /* stderr: per-block GroupNorm statistics of the last forward and largest |weight| */
/* g_h16: glogits already holds the 16-bit storage type scaled by the gradient scale (nn_set_loss_grad_h16) */
void unet_backward_x(unet *u, const void *glogits, int g_h16);
void unet_zero_grad(unet *u);
/* Flat gradient exchange for data-parallel training (host buffer of unet_nparams floats). */
float *unet_grad_ptr(unet *u);                        /* device pointer of the flat gradient */
void unet_grad_d2h(unet *u, float *host);
void unet_grad_h2d(unet *u, const float *host);
double unet_grad_norm(unet *u);
void unet_clip_grad(unet *u, double max_norm);          /* scales grads if their norm exceeds max_norm */
void unet_muon(unet *u, float lr_muon, float beta, float lr_adam, float b1, float b2, float eps, float wd, int step);   /* Muon on the 3^3 conv weights, AdamW elsewhere */
/* Muon variants for unet_muon: muonc = Muon-C (per-frequency polarisation of the conv kernels), hyperball = fixed weight and
   update norms (eta = hb_mult x the Muon lr, replaces the decay); both 0 = Muon. The radii go into checkpoints (hb_R). */
void unet_set_muon_ext(unet *u, int muonc, int hyperball, float hb_mult);
void unet_anvil(unet *u, float lr, float wd, int step, int steps, float lr_adam, float b1, float b2, float eps, float wd_adam);   /* ANVIL II on the 3^3 conv weights, AdamW elsewhere */
void unet_adamw(unet *u, float lr, float b1, float b2, float eps, float wd, int step);
void unet_ema(unet *u, float decay);
/* Use the EMA weights (1) or the live weights (0) for forward. */
void unet_use_ema(unet *u, int on);
/* Frozen GroupNorm statistics for inference: unet_gn_nstats floats per sample (means and reciprocal standard deviations of every
   GroupNorm in a fixed order); unet_gn_stats_get copies sample 0 of the last forward (tensor-core path); unet_gn_freeze(h) makes
   every later inference forward use h instead of the window's statistics (nullptr: back to the window's). */
size_t unet_gn_nstats(unet *u);
int unet_gn_stats_get(unet *u, float *h);
void unet_gn_freeze(unet *u, const float *h);
void unet_gn_accumulate(unet *u);           /* add the last forward's statistics (sample 0) to a running pool */
long unet_gn_pooled(unet *u, float *h);     /* pooled statistics (unet_gn_nstats floats) of the accumulated forwards; returns their count */
/* cfg.l0_norm 2 (ema): the running statistics move by mom toward each training forward's (the first one replaces them); the first
   warm forwards normalise with their own statistics (GroupNorm), later ones with the running statistics from before the forward.
   Defaults 0.01, 100; unet_l0_updates: training forwards so far (saved in checkpoints with the statistics) */
void unet_set_l0_ema(unet *u, float mom, int warm);
void unet_set_l0_slab_kern(int on);   /* --l0-norm slab on the kernel path (readers apply per-slab statistics; 1) or D's prototype (0); -1: env UFSM_L0_SLAB_KERN */
/* ema: recalibrate the running statistics of the weights in use (the EMA weights' after unet_use_ema(u, 1)) -- every forward from
   unet_l0_recal(u, 1) to unet_l0_recal(u, 0) normalises with its window's statistics, their cumulative mean replaces that set (the
   standard BatchNorm recalibration of averaged weights; the update count and the other set unchanged) */
void unet_l0_recal(unet *u, int on);
long unet_l0_updates(const unet *u);
/* 2:4 structured sparsity of the 3^3 conv weights (groups of 4 input channels keep 2): the forward reads a masked copy
   of the weights (unet_apply_sparse24 runs inside unet_forward); unet_srste24 adds the SR-STE term lambda * w to the
   gradient of the pruned weights so they keep decaying but can be revisited. The flag is saved in checkpoints. */
void unet_set_sparse24(unet *u, int on);
int unet_get_sparse24(const unet *u);
void unet_apply_sparse24(unet *u);
void unet_srste24(unet *u, float lambda);
/* true fp8 / fp4 weights: the 3^3 conv weights live in packed e4m3 / e2m1 storage with MX block scales; AdamW and the EMA update
   the packed values directly (stochastic rounding) and the fp32 arrays only hold dequantized shadows for the kernels. The head,
   biases and GroupNorm parameters stay fp32. Checkpoints store the shadows (exact grid values) plus the flag. */
void unet_set_wq(unet *u, int bits);
int unet_get_wq(const unet *u);
void unet_wquant(unet *u, unsigned seed);

/* Checkpoint: "UFSM" magic, JSON header line, then float32 params, ema, adam m, adam v;
   optional Muon momentum follows when the header's muon_mom flag is set. */
int unet_save(const unet *u, const char *path, int step, const char *extra_json);
/* Loads weights and saved optimizer state from the same cfg; returns the saved step or -1.
   Old checkpoints remain readable and start Muon with zero momentum. */
int unet_load(unet *u, const char *path);
int unet_load_grow(unet *u, const char *path, int keep, float new_bias);   /* warm start into a wider head (src/unet.c) */
int unet_init_from(unet *u, const char *path);   /* partial warm start: shape-matching layers of another net (src/unet.c) */
int unet_init_from_map(unet *u, const char *path, const int *head_rows);   /* the same with an explicit head row map (src/unet.c) */
/* Retain surface response, zero the new scalar input and winding output,
   including EMA/Adam/Muon state for those parameters. */
int unet_start_sheet(unet *u);
/* Read only the cfg from a checkpoint (to construct the net before loading). */
int unet_peek(const char *path, unet_cfg *cfg, int *step);
size_t unet_activation_bytes(const unet *u);
void unet_prof_report(void);   /* with UFSM_PROF=1: print and reset per-op timings */
/* per-conv kernel timing: slot = 2 * layer + conv (layer ids of nn_set_layer for L levels: enc0..enc(L-1) = 0..L-1,
   down0..down(L-2) = L..2L-2, dec(L-2)..dec0 = 2L-1..3L-3, head = 3L-2; L = 4: enc0..3 = 0..3, down0..2 = 4..6, dec2..0 = 7..9,
   head = 10; conv 0 = c1 or the single conv, 1 = c2) */
#define UNET_NSLOT (2 * (3 * UNET_MAXLEV - 1))
void unet_prof_layers_on(int on);          /* same as env UFSM_PROF=layers */
void unet_prof_layers(double out[][3]);    /* [UNET_NSLOT][fwd, bwd_data, bwd_w] ms since the last call (resets) */
void unet_prof_cats(double out[8]);       /* per-category ms (conv_fwd, conv_bwd_data, conv_bwd_w, gn, elementwise, up/concat, ...) of the last unet_prof_layers call */
/* MX-fp8 activation storage (also env UFSM_ACT_MX8=1): every stored activation is channel-blocked e4m3 + ue8m0 scales and
   the convolutions that read them run in fp8; activation gradients keep their storage. Takes effect at the next build. */
void unet_set_act_mx8(int on);
void unet_set_act_mx4(int on);   /* packed fp4 activation storage (env UFSM_ACT_MX4=1); takes precedence over mx8 */
void unet_set_input_mx(int on);  /* quantize the stem input to the activation MX format; mandatory with MX gradients */
int unet_set_input_prec(int bits);  /* 0: automatic; 4/8: MX stem input independent of the rest of the graph; invalid returns -1 */
int unet_input_prec(void);          /* effective stem input MX bits, or 0 for ordinary input storage */
int unet_act_mx4(void);          /* effective FP4 activation storage */
int unet_grad_mx8(void);   /* effective MX-fp8 gradient storage (env or set by train --mem auto) */
void unet_set_grad_mx8(int on);   /* MX-fp8 activation gradients as well (env UFSM_GRAD_MX8=1; requires the MX activations) */
/* recompute mode (env UFSM_RECOMPUTE=1): no stored block outputs silu(gn(a2)) and no stored upsampled decoder inputs; the
   consumers apply GN + SiLU while staging and the upsample is rebuilt into a transient buffer */
void unet_set_recompute(int on);
void unet_set_lean(int on);
void *unet_grad_scratch(unet *u, size_t bytes);
void *unet_batch_scratch(unet *u, size_t head, size_t tail);   /* lean 2: the batch as [head (input) | tail (targets, masks)] */
/* three level-0 buffers (env UFSM_L0_3BUF=1 / unet_set_l0_3buf, -1 the env; training builds of the MX-fp4 production chain with an 8- or
   16-channel level 0): the batch's head in level-0 buffer X2, its tail at the end of X3 (*tailp), nullptr if not this layout / too large;
   unet_set_batch_bytes before the build sizes X3 for the tail; the logit gradient goes over the fp16 logits (unet_logit_grad_scratch
   gives none). unet_l0_3buf: this build uses them; unet_last_dry_l3b: the last unet_train_bytes did */
void unet_set_l0_3buf(int on);
void unet_set_l0_3buf_split(int on);     /* dec[0]'s conv1 weight gradient in an up and a skip part around the skip's copy (UFSM_L0_3BUF_SPLIT, default 0) */
void unet_set_l0_3buf_reorder(int on);   /* enc[0]'s conv2 backward-data before its weight gradient (UFSM_L0_3BUF_REORDER, default 1) */
void unet_set_l3_sched(int on);          /* the host copies in the order they are due, the coarse encoders' conv1 re-run (UFSM_L3_SCHED, default 1;
                                            takes effect at the next build) */
void unet_set_l3_skip_d2d(int on);      /* (l3_sched) the skip left in X2 copied into X3 on the device for down0's weight gradient (UFSM_L3_SKIP_D2D,
                                            default 1; -1 the env / default) */
void *unet_batch_scratch2(unet *u, size_t head, size_t tail, void **tailp);
void unet_set_batch_bytes(unet *u, size_t head, size_t tail);
int unet_l0_3buf(const unet *u);
int unet_l0_3buf_ct(const unet *u);   /* the last training forward (GPU-built input) keeps the CT, not the input, for enc[0]'s backward */
/* an event the backward records on the compute stream once the batch head (unet_batch_scratch2) and the radial records are read for the
   last time (before the stem's weight gradient): the next batch's head may go up from there; nullptr without the three buffers. Never
   recorded before the first backward (a wait on it then returns at once) */
void *unet_l0_3buf_head_free(unet *u);
int unet_last_dry_l3b(void);
/* --mem fast: all training host offloads off (0) or at their defaults (1); the pinned host bytes of the last build (dry or real), each
   copied device -> host and back once per step */
void unet_set_host_offloads(int on);
size_t unet_last_offload_bytes(void);
void *unet_logit_grad_scratch(unet *u, size_t bytes);   /* lean: home of the 16-bit logit gradient (A after the logits under lean 2, else B) */
int unet_lean_nob(const unet *u);   /* lean 2 build: no gradient buffer B (gout[0] is the batch scratch) */
void unet_build(unet *u, shape5 xs, int train);   /* build the buffers for xs now (the first forward would) */
void unet_set_head_channels(unet *u, int c0, int nc);   /* inference: only logit channels [c0, c0 + nc) (nc 0: all) */
void *unet_input_scratch(unet *u, size_t bytes);
void *unet_input_scratch2(unet *u, size_t bytes);   /* inference: enc[0].a2's buffer, free until the forward's input conversion (predict: the uint8 CT) */
void unet_set_infer_alias(int on);
void unet_set_infer_slab(int planes);   /* inference: level-0 convs in place in z-slabs of this many planes (UFSM_INFER_SLAB; 0 off) */   /* inference aliasing of the network input and the logits into level-0 buffers (UFSM_INFER_ALIAS, default 1) */   /* inference build: scratch for the 16-bit network input (or nullptr) */
void unet_set_infer_offload(int on);   /* inference with level-1 slabs: the coarse levels in the tail of the level-0 buffer, which waits on the host (UFSM_INFER_OFFLOAD, default 1) */
void unet_set_ct_input(unet *u, int on);   /* inference input is the uint8 CT window (unet_forward_ct); with slabs the stem builds its input per slab */
int unet_ct_stem(const unet *u);             /* built so: unet_forward_ct applies */
const float *unet_forward_ct(unet *u, const uint8_t *ct, float mean, float isd, const float *dyo, const float *dxo, int axis, shape5 xs, int h16);   /* predict: CT window + axis rows (nn_pred_input's arguments) */
int unet_input_converted(void);
int unet_act_mx(void);   /* 1: MX activation storage (MX-fp8 gradients need it) */   /* 1: the forward copies the 16-bit input into MX storage first (it may then live in scratch) */   /* lean training build: B as scratch for the 16-bit logit gradient (else nullptr) */       /* training: the fp32 logits share the gradient buffer A (env UFSM_LEAN=1; set by train --mem auto for the large-window modes) */
void unet_set_chunk_up(int on);   /* chunked up-part gradient: 0 off, 1 (default) 16-bit storage only, 2 also under MX storage */
void unet_set_wide_up_grad(int on);   /* optional larger level-0 MX8 up-gradient scratch under lean 2; default off, env UFSM_WIDE_UP_GRAD */
int unet_wide_up_grad(void);
void unet_set_share_enc_a1(int on);   /* training, recompute 1: encoder blocks above the bottom keep no a1 (backward re-runs conv1 into the same-level decoder's a1) */
int unet_share_enc_a1(void);
int unet_logits_in_gx(const unet *u);   /* the logits share the head's input-gradient buffer (no in-place logit gradient) */
/* train --gpu-input: the next forwards' 16-bit x is the CT channel alone ([n][S]) and rec (device, n records of NN_RAD_REC(D) bytes, nn.h)
   holds each sample's radial record: the 4 MX input channels are built on the device (nn_input_rad). nullptr: x has all 4 channels */
void unet_set_input_rad(unet *u, const void *rec);
int unet_logits_h16(const unet *u);   /* the logits unet_forward returns are fp16 (lean 2 training: the logit gradient fits after them) */
/* head slabs (training): the forward skips the 1^3 head and allocates no whole-window logits; the caller runs it per z-slab
   (unet_head_fwd_z after the forward), and the backward calls cb(ctx) at the head's step, which runs unet_head_fwd_z, the loss
   gradient and unet_head_bwd_z per slab (first: the first slab). Takes effect at the next build; MX activations and 16-bit gradients
   (the fused head backward); -1 from the slab calls: not covered */
void unet_set_head_ext(unet *u, int on);
void unet_set_wd_nonconv(unet *u, float mul);
const float *unet_ds_logits(const unet *u, int j, shape5 *s);   /* deep supervision: fp32 logits of decoder level j (1..ds_levels) after a training forward */
void unet_set_ds_grad(unet *u, int j, const float *gl);         /* their fp32 logit gradient for the next backward (nullptr: none) */   /* weight decay multiplier for GroupNorm / biases / head under Muon and ANVIL (default 0) */
void unet_blend_ema(unet *u, float a);          /* EMA weights := a x live + (1 - a) x EMA (inference: the tail-average blend) */
void unet_set_head_cb(unet *u, void (*cb)(void *), void *ctx);
int unet_head_fwd_z(unet *u, int z0, int z1, void *y16);
int unet_head_bwd_z(unet *u, int z0, int z1, const void *gy16, int first);
void unet_set_fp16_logits(unet *u, int on);   /* MX activations: fp16 logits where they get their own buffer (half the head output; the readers
                                                 follow unet_logits_h16 / nn_set_logits_h16). Not for half-resolution or sheet nets */
void unet_set_grad_mx4(int on);   /* MX-fp4 activation gradients (with the MX gradient mode; opt-in, env UFSM_GRAD_MX4) */
int unet_grad_mx4(void);
void unet_set_a1_offload(int on);   /* shared encoder a1: pinned host copies instead of the conv1 re-run (default on; UFSM_A1_OFFLOAD) */
int unet_a1_offload(void);
void unet_set_a0_share(int on);   /* lean 2, MX-fp4: each decoder level's A in its dec[i].a2's buffer, the skip on the host meanwhile (default on; UFSM_A0_SHARE) */
void unet_set_coarse_grad_share(int on);   /* lean 2, level-0 A share: the coarse levels' gout in dec[0]'s buffers (default on; UFSM_COARSE_GRAD_SHARE) */
void unet_set_down_offload(int on);   /* with the coarse gradient share: the down-conv outputs in dec[0].a2's buffer, host copy in between (default on; UFSM_DOWN_OFFLOAD) */
void unet_set_skip_offload(int on);   /* with the down-conv output offload: the coarse skips in dec[0].a2's buffer, host copy in between (default on; UFSM_SKIP_OFFLOAD) */
void unet_set_xin_offload(int on);   /* training: the input in dec[0].a2's buffer, pinned host copy in between (default on; UFSM_XIN_OFFLOAD) */
void unet_set_up_wg_chunk(int on);   /* decoder conv1 weight gradient in 32-channel up slices (-1 default: with MX-fp4 gradients) */
size_t unet_train_bytes(unet *u, shape5 xs);   /* training activation + gradient bytes at xs for the current modes (dry build) */
size_t unet_infer_bytes(unet *u, shape5 xs);   /* the same for the inference forward (buffers allocated at build time) */
size_t unet_grad_bytes(const unet *u);   /* the activation-gradient part of unet_activation_bytes */
/* spatial split of one window along z across two GPUs (src/split.h): side 0 holds the low z planes, side 1 the high ones, each
   plus h0 = 2^(nlev-1) halo planes at level 0 on the side facing the other (the input it is given has that depth). fn exchanges
   a tensor's innermost halo plane with the other GPU (split_halo). Loss masks must be 0 on the halo planes. side -1 / fn null:
   off. Parameter gradients are this half's share: sum them across the two GPUs. */
typedef void (*unet_halo_fn)(const void *p, shape5 s, int esz, int h);
void unet_set_split(unet *u, int side, int h0, unet_halo_fn fn);
/* optional: the exchange in two parts (split_halo_begin / split_halo_end) so a weight gradient overlaps it */
void unet_set_split_async(unet *u, unet_halo_fn begin, unet_halo_fn end);
