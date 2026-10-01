/* Internal interface between nn.cu and the low-precision (FP8 / FP4) convolution kernels in nn_fp8.cu. */
#pragma once
#include "nn.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct { const float *gamma, *beta, *mean, *rstd; int G; } gnp_t;   /* G == 0: no transform */
/* channel split: input channels >= c_split come from x2 (channel ci - c_split); output channels >= o_split go to y2 */
/* accum: y += conv instead of y = conv. gp2 (G > 0): GroupNorm + SiLU of the x2 segment (its own channel indices / groups);
   the main gnp_t then applies to the x segment alone. gp2.G == 0 keeps the legacy meaning (gnp_t over all channels). */
/* up (with x2): the x segment is stored at half resolution (D/2 x H/2 x W/2) and read as its trilinear 2x upsample
   (align_corners = false, edge clamp: the same values as nn_up2_fwd_into), so the decoder input is never materialised */
/* sr != 0: stochastic rounding (seed) of this conv's gradient operand when quantised to fp8 (the input of a backward-data
   conv, the gy of a weight gradient); 0 = round to nearest */
/* wkey != 0: id of this conv (layer / conv / pass) for the prepared-weight memo of the fp4 kernel (see lp_wmemo_*) */
typedef struct { const void *x2; int c_split; void *y2; int o_split; int accum; gnp_t gp2; int up; unsigned sr; unsigned wkey; } split_t;
/* k=3 stride 1 forward / weight gradient with the same semantics as the BF16 tensor-core paths in nn.cu
   (conv_fwd_tc / launch_bwd_w_tc): xbf / ybf = activation (and x2 / y2) storage type, 0 fp32, 1 bf16, 2 fp16; the
   forward needs xbf == ybf. Weights, biases and GroupNorm parameters are fp32; the weight gradient accepts fp32 or (with xbf) bf16 gy. */
int lp_conv_fwd_f8(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, gnp_t gp, double *osum, int Go, split_t sp);
int lp_conv_fwd_f4(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, gnp_t gp, double *osum, int Go, split_t sp);
void lp_wmemo_step(unsigned step);   /* fp4 prepared-weight memo: new training step (weights changed) */
void lp_wmemo_clear(void);
void lp_set_w4_2d(int on);         /* 2D (32 x 32 tile) fp4 weight scales, shared by forward and backward-data (UFSM_W4_2D) */           /* weights changed outside a step (checkpoint load, EMA swap) */
int lp_bwd_w_f8(const void *x, int xbf, shape5 xs, const void *gy, int gybf, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp);   /* gybf: gy is bf16 (needs xbf) */
int lp_bwd_w_f4(const void *x, int xbf, shape5 xs, const void *gy, int gybf, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp, int had);   /* fp4 (e2m1, SR on gy when sp.sr); had: bit 0 fixed-sign H32 on both operands, bit 1 SR on x too (diagnostic) */
int lp_conv_fwd_s2_f8(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, shape5 ys, gnp_t gp);   /* stride 2; gp: input gn+silu */
int lp_bwd_w_s2_f8(const void *x, int xbf, shape5 xs, const void *gy, int gybf, shape5 ys, float *gw, float *gb, gnp_t gp);   /* stride 2, gw / gb accumulated */
/* MX-fp8 activation storage (lp dtype 3), see nn_fp8.cu: channel-blocked e4m3 bytes + ue8m0 scale plane */
size_t lp_mx8_bytes(int N, int C, size_t S);
void lp_f32_to_mx8(const float *x, int N, int C, size_t S, void *y);
void lp_h16_to_mx8(const void *x, int dt, int N, int C, size_t S, void *y);   /* dt 1 bf16, 2 fp16 */
void lp_mx8_to_f32(const void *x, int N, int C, size_t S, float *y);
/* MX-fp4 activation storage (lp dtype 4): the same blocking with packed e2m1 nibbles (bw/2 bytes per row) + ue8m0 plane */
size_t lp_mx4_bytes(int N, int C, size_t S);
void lp_f32_to_mx4(const float *x, int N, int C, size_t S, void *y);
void lp_h16_to_mx4(const void *x, int dt, int N, int C, size_t S, void *y);
void lp_mx4_to_f32(const void *x, int N, int C, size_t S, float *y);
double lp_sr_e2m1_mean(float v, size_t n);                        /* test probe: mean of n stochastic e2m1 roundings of v (v in grid units) */
void lp_cvt_e2m1_probe(const float *hv, unsigned char *ho, int n);   /* test probe: raw cvt.rn.satfinite.e2m1x2 nibble of each host value */
/* MX elementwise ops: xdt / ydt = lp dtype of the operand (3 mx8, 4 mx4; gn_silu_apply also takes a plane-major 0 / 1 / 2 input);
   gradients (gy / gx, gdt) are never mx4 */
void lp_gn_silu_apply_mx(const void *x, int xdt, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd, void *y, int ydt);
void lp_up2_fwd_mx(const void *x, int xdt, shape5 xs, void *y, int ydt, gnp_t gp);   /* gp: silu(gn(x)) upsampled */
void lp_conv1_fwd_mx(const void *x, int xdt, shape5 xs, const float *w, const float *b, int cout, float *y, gnp_t gp);
void lp_bwd_w1_mx(const void *x, int xdt, shape5 xs, const void *gy, int gydt, shape5 ys, float *gw, gnp_t gp);
void lp_gn_silu_bwd_mx(const void *x, int xdt, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd,
                       const void *gy, void *gx, int gdt, double *ds, float *st, float *AB);
void lp_gn_silu_bwd_apply_mx(const void *x, int xdt, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd,
                             const void *gy, void *gx, int gdt, const float *AB);
void lp_conv1_to_mx(const void *x, int gdt, int N, int Ci, size_t S, const float *w, int Co, void *y);
void lp_up2_bwd_mx(const void *gy, shape5 xs, void *gx);
void lp_bwd_data_s2_mx(const void *gy, shape5 ys, const float *w, shape5 xs, void *gx, int accum);
const char *lp_check(void);
void lp_f32_to_bf16(const float *x, size_t n, void *y);   /* test helpers */
void lp_bf16_to_f32(const void *x, size_t n, float *y);
int lp_gn_sums_mx(const void *x, int xdt, int N, int C, int G, size_t S, double *sums);   /* GroupNorm (sum, sum sq) per (n, group) of an MX tensor, accumulated; -1 unsupported */
#ifdef __cplusplus
}
#endif
