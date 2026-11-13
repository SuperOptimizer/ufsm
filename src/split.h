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
void split_halo_k(const void *p, shape5 s, int esz, int h, int k);   /* nn_split_halo_k inside a job (a collective) */
/* generic collectives for the device-side input builders (augz nn_zcomm): stream = a cudaStream_t of this side (nullptr: default) */
void split_xchg(const void *send_d, size_t ns, void *recv_d, size_t nr, void *stream);   /* recv_d (nr bytes) <- the peer's send (its ns = my nr) */
void split_sum(void *d, int n, int type, void *stream);   /* in-place sum over both sides of n values: type 0 u64, 1 f64, 2 u32 */
void split_gather(const void *p, shape5 s, int esz, int h, int D);   /* nn_split_gather inside a job (unet_set_split_top's gather) */
void split_yield(void);   /* a no-op collective: hands the turn to the other side (both sides must call it equally often) */
int split_side(void);   /* this job's side (0 / 1), -1 outside split_run */
void split_reduce(double *b, int n);   /* registered as the nn reduce callback by split_create */
