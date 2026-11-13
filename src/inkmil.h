/* the normal-pooled MIL ink loss (2026-11-11; villa's normal_pooled_3d, src/nn_inkmil.cu): the 2-D label of a ray (inksplat.h inkray)
   supervises the pooled 3-D ink logit along the label pixel's normal. Per ray, samples z_k = trilinear(logit, p + t_k n), t_k = -R + k step
   (k <= 2R / step; samples outside the volume are skipped); pooled zr = tau log((1 / K') sum_k exp(z_k / tau)) (tau > 0: a soft max) or
   max_k z_k (tau = 0); loss_r = w BCE(zr, y'), y' = y (1 - eps) + eps / 2 (label smoothing). grad += scale w (sigmoid(zr) - y') dzr / dz_k
   spread over each sample's 8 trilinear corners. *loss (device) += scale sum_r loss_r. A ray without a sample inside contributes nothing. */
#pragma once
#include "inksplat.h"
#ifdef __cplusplus
extern "C" {
#endif
/* logit / grad: device float D x H x W (z y x); rays: device; loss: device double. 0 ok. */
__attribute__((weak)) int ink_mil_loss_dev(const float *logit, float *grad, int D, int H, int W, const inkray *rays, int nr, float R, float step, float tau,
                                           float eps, float scale, double *loss, void *stream);
__attribute__((weak)) int ink_mil_loss_hostmem(const float *logit, float *grad, int D, int H, int W, const inkray *rays, int nr, float R, float step, float tau,
                                               float eps, float scale, double *loss);   /* (tests: host arrays, copied through the device) */
/* the slab form for train's head_slab_pass (the logits in z slabs, pass 0 then pass 1; fp16 or bf16 logits / gradients):
   acc: device float nr x K (K = ink_mil_samples), zeroed before the first slab. Pass 0, per slab [z0, z0 + nz) of the D x H x W window
   (lg: that slab's logits): fwd_slab adds each sample's corner terms inside the slab (one writer per sample). Between the passes, fin:
   acc -> dL/dz per sample, *loss += the loss (a fixed-order sum). Pass 1, per slab: bwd_slab adds the slab's gradient into gl (that
   slab's 16-bit logit gradient), deterministic (the corner terms sorted by voxel and slot, summed in order; one sync per slab). 0 ok. */
__attribute__((weak)) int ink_mil_samples(float R, float step);
__attribute__((weak)) int ink_mil_fwd_slab(const void *lg, int bf16, int z0, int nz, int D, int H, int W, const inkray *rays, int nr, float R, float step, float *acc, void *stream);
__attribute__((weak)) int ink_mil_fin(float *acc, const inkray *rays, int nr, int D, int H, int W, float R, float step, float tau, float eps, float scale, double *loss, void *stream);
__attribute__((weak)) void ink_mil_set_async(int on);   /* 1: bwd slabs bound their pair counts from fin's plane counts (no per-slab sync); 0: the per-slab readback */
__attribute__((weak)) int ink_mil_bwd_slab(const float *dz, int z0, int nz, int D, int H, int W, const inkray *rays, int nr, float R, float step, void *gl, int bf16, void *stream);
__attribute__((weak)) int ink_mil_slab_hostmem(const float *logit, int bf16, int D, int H, int W, const inkray *rays, int nr, float R, float step, float tau, float eps, float scale,
                                               int nslab, float *grad, double *loss);   /* (tests) */
/* the host reference (tests): the same arithmetic in double */
double ink_mil_loss_host(const float *logit, double *grad, int D, int H, int W, const inkray *rays, int nr, float R, float step, float tau, float eps, float scale);
#ifdef __cplusplus
}
#endif
