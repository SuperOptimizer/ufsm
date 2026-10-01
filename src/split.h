/* Spatial (domain) parallelism of one training window across two GPUs, split along z.
   split_run executes job(side) on two host threads, one per GPU, that take turns: only one runs host code at a time (the nn
   and unet modules keep global state), and they hand over at every collective. A collective (halo exchange, sum of a small
   device buffer) is issued for both GPUs by the thread that arrives second, so each side's stream sees it in order. */
#pragma once
#include "nn.h"

typedef struct split_ctx split_ctx;
split_ctx *split_create(int dev0, int dev1);
void split_free(split_ctx *c);
void split_run(split_ctx *c, void (*job)(int side, void *arg), void *arg);   /* returns when both sides finished */
/* inside a job: halo exchange of tensor p (shape s on both sides, h halo planes, esz as nn_split_halo) */
void split_halo(const void *p, shape5 s, int esz, int h);
/* the same in two parts: begin (a collective) starts the exchange, end (this side only) completes it; the compute stream may
   run work that does not read the halo in between. One such exchange in flight at a time (split_halo may run meanwhile). */
void split_halo_begin(const void *p, shape5 s, int esz, int h);
void split_halo_end(const void *p, shape5 s, int esz, int h);
void split_reduce(double *b, int n);   /* registered as the nn reduce callback by split_create */
