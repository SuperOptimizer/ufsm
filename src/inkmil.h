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
/* the host reference (tests): the same arithmetic in double */
double ink_mil_loss_host(const float *logit, double *grad, int D, int H, int W, const inkray *rays, int nr, float R, float step, float tau, float eps, float scale);
#ifdef __cplusplus
}
#endif
