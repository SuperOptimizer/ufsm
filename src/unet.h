/* Tiny 3-D U-Net on top of nn.h: `nlev` levels with widths w[i]; per level two 3^3 convs each followed
   by GroupNorm(G) + SiLU; stride-2 3^3 conv down; trilinear 2x up + channel concat; 1^3 head.
   Parameters, gradients, Adam moments and the EMA copy live in flat device arrays. */
#pragma once
#include "nn.h"
#include <stdint.h>

#define UNET_MAXLEV 8

typedef struct {
    int nlev;
    int widths[UNET_MAXLEV];
    int cin, cout;
    int G;                 /* GroupNorm groups (min(G, C) is used) */
    int down_norm;         /* 1: GroupNorm + SiLU after each stride-2 down conv (params appended after the head; 0 keeps the old layout) */
} unet_cfg;

typedef struct unet unet;

unet *unet_create(const unet_cfg *cfg);
void unet_free(unet *u);
const unet_cfg *unet_cfg_of(const unet *u);
size_t unet_nparams(const unet *u);
void unet_init(unet *u, uint64_t seed);                 /* Kaiming-normal convs, GN gamma 1 / beta 0, head bias -2 */

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
void unet_anvil(unet *u, float lr, float wd, int step, int steps, float lr_adam, float b1, float b2, float eps, float wd_adam);   /* ANVIL II on the 3^3 conv weights, AdamW elsewhere */
void unet_adamw(unet *u, float lr, float b1, float b2, float eps, float wd, int step);
void unet_ema(unet *u, float decay);
/* Use the EMA weights (1) or the live weights (0) for forward. */
void unet_use_ema(unet *u, int on);
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

/* Checkpoint: "UFSM" magic, JSON header line, then float32 params, ema, adam m, adam v. */
int unet_save(const unet *u, const char *path, int step, const char *extra_json);
/* Loads params/ema/adam from a checkpoint created with the same cfg; returns the saved step or -1. */
int unet_load(unet *u, const char *path);
/* Read only the cfg from a checkpoint (to construct the net before loading). */
int unet_peek(const char *path, unet_cfg *cfg, int *step);
size_t unet_activation_bytes(const unet *u);
void unet_prof_report(void);   /* with UFSM_PROF=1: print and reset per-op timings */
/* per-conv kernel timing: slot = 2 * layer + conv (layer ids of nn_set_layer: enc0..3 = 0..3, down0..2 = 4..6, dec2, dec1,
   dec0 = 7..9, head = 10; conv 0 = c1 or the single conv, 1 = c2) */
#define UNET_NSLOT 22
void unet_prof_layers_on(int on);          /* same as env UFSM_PROF=layers */
void unet_prof_layers(double out[][3]);    /* [UNET_NSLOT][fwd, bwd_data, bwd_w] ms since the last call (resets) */
void unet_prof_cats(double out[8]);       /* per-category ms (conv_fwd, conv_bwd_data, conv_bwd_w, gn, elementwise, up/concat, ...) of the last unet_prof_layers call */
/* MX-fp8 activation storage (also env UFSM_ACT_MX8=1): every stored activation is channel-blocked e4m3 + ue8m0 scales and
   the convolutions that read them run in fp8; activation gradients keep their storage. Takes effect at the next build. */
void unet_set_act_mx8(int on);
void unet_set_grad_mx8(int on);   /* MX-fp8 activation gradients as well (env UFSM_GRAD_MX8=1; requires the MX activations) */
/* recompute mode (env UFSM_RECOMPUTE=1): no stored block outputs silu(gn(a2)) and no stored upsampled decoder inputs; the
   consumers apply GN + SiLU while staging and the upsample is rebuilt into a transient buffer */
void unet_set_recompute(int on);
size_t unet_grad_bytes(const unet *u);   /* the activation-gradient part of unet_activation_bytes */
