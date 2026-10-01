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
shape5 unet_out_shape(const unet *u, shape5 xs);
/* Backward from d loss / d logits (device). Gradients ACCUMULATE into the flat grad array. */
void unet_backward(unet *u, const float *glogits);
void unet_zero_grad(unet *u);
/* Flat gradient exchange for data-parallel training (host buffer of unet_nparams floats). */
float *unet_grad_ptr(unet *u);                        /* device pointer of the flat gradient */
void unet_grad_d2h(unet *u, float *host);
void unet_grad_h2d(unet *u, const float *host);
double unet_grad_norm(unet *u);
void unet_clip_grad(unet *u, double max_norm);          /* scales grads if their norm exceeds max_norm */
void unet_adamw(unet *u, float lr, float b1, float b2, float eps, float wd, int step);
void unet_ema(unet *u, float decay);
/* Use the EMA weights (1) or the live weights (0) for forward. */
void unet_use_ema(unet *u, int on);

/* Checkpoint: "UFSM" magic, JSON header line, then float32 params, ema, adam m, adam v. */
int unet_save(const unet *u, const char *path, int step, const char *extra_json);
/* Loads params/ema/adam from a checkpoint created with the same cfg; returns the saved step or -1. */
int unet_load(unet *u, const char *path);
/* Read only the cfg from a checkpoint (to construct the net before loading). */
int unet_peek(const char *path, unet_cfg *cfg, int *step);
size_t unet_activation_bytes(const unet *u);
void unet_prof_report(void);   /* with UFSM_PROF=1: print and reset per-op timings */
