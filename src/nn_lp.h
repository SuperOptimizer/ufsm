/* Internal interface between nn.cu and the low-precision (FP8 / FP4) convolution kernels in nn_fp8.cu. */
#pragma once
#include "nn.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct { const float *gamma, *beta, *mean, *rstd; int G; } gnp_t;   /* G == 0: no transform */
/* channel split: input channels >= c_split come from x2 (channel ci - c_split); output channels >= o_split go to y2 */
typedef struct { const void *x2; int c_split; void *y2; int o_split; int accum; } split_t;   /* accum: y += conv instead of y = conv */
/* k=3 stride 1 forward / weight gradient with the same semantics as the BF16 tensor-core paths in nn.cu
   (conv_fwd_tc / launch_bwd_w_tc): xbf / ybf = activation (and x2 / y2) storage type, 0 fp32, 1 bf16, 2 fp16; the
   forward needs xbf == ybf. Weights, biases and GroupNorm parameters are fp32; the weight gradient accepts fp32 or (with xbf) bf16 gy. */
int lp_conv_fwd_f8(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, gnp_t gp, double *osum, int Go, split_t sp);
int lp_conv_fwd_f4(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, gnp_t gp, double *osum, int Go, split_t sp);
int lp_bwd_w_f8(const void *x, int xbf, shape5 xs, const void *gy, int gybf, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp);   /* gybf: gy is bf16 (needs xbf) */
int lp_conv_fwd_s2_f8(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, shape5 ys);   /* stride 2 */
int lp_bwd_w_s2_f8(const void *x, int xbf, shape5 xs, const void *gy, int gybf, shape5 ys, float *gw, float *gb);   /* stride 2, gw / gb accumulated */
/* MX-fp8 activation storage (lp dtype 3), see nn_fp8.cu: channel-blocked e4m3 bytes + ue8m0 scale plane */
size_t lp_mx8_bytes(int N, int C, size_t S);
void lp_f32_to_mx8(const float *x, int N, int C, size_t S, void *y);
void lp_mx8_to_f32(const void *x, int N, int C, size_t S, float *y);
void lp_gn_silu_apply_mx(const void *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd, void *y);
void lp_up2_fwd_mx(const void *x, shape5 xs, void *y);
void lp_conv1_fwd_mx(const void *x, shape5 xs, const float *w, const float *b, int cout, float *y);
void lp_bwd_w1_mx(const void *x, shape5 xs, const void *gy, int gydt, shape5 ys, float *gw);
void lp_gn_silu_bwd_mx(const void *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd,
                       const void *gy, void *gx, int gdt, double *ds, float *st, float *AB);
void lp_gn_silu_bwd_apply_mx(const void *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd,
                             const void *gy, void *gx, int gdt, const float *AB);
const char *lp_check(void);
void lp_f32_to_bf16(const float *x, size_t n, void *y);   /* test helpers */
void lp_bf16_to_f32(const void *x, size_t n, float *y);
#ifdef __cplusplus
}
#endif
