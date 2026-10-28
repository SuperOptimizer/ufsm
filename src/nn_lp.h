/* Internal interface between nn.cu and the low-precision (FP8 / FP4) convolution kernels in nn_fp8.cu. */
#pragma once
#include "nn.h"
#ifdef __cplusplus
extern "C" {
#endif
/* bf16 / fp32 tensor-core instantiations (verification, legacy --f16 0) are only compiled with -DUFSM_ALL_TYPES
   (make VERIFY=1); otherwise a call routed to one stops with a message instead of running. */
void ufsm_no_verify_types(const char *fn);
#ifdef UFSM_ALL_TYPES
#define VERIFY_ONLY(...) (__VA_ARGS__)
#else
#define VERIFY_ONLY(...) (ufsm_no_verify_types(__func__), 0)
#endif
typedef struct { const float *gamma, *beta, *mean, *rstd; int G; int zs, zo, ns, zhw; } gnp_t;   /* G == 0: no transform */
/* slab-local GroupNorm (ns > 0; unet cfg.l0_norm slab, kernel path): mean / rstd are per (sample, z-slab), [n][ns][G]; plane z of
   the tensor belongs to slab (z + zo) / zs (0 <= zo < zs), zhw = voxels per z plane (pointwise readers). ns == 0: per sample.
   Statistics arrays registered with lp_gn_slab_reg carry this geometry into every gnp_t built by to_gnp / gnp_mk. */
gnp_t gnp_mk(const float *gamma, const float *beta, const float *mean, const float *rstd, int G);
int lp_gn_slab_any(void);   /* any statistics array registered */
/* channel split: input channels >= c_split come from x2 (channel ci - c_split); output channels >= o_split go to y2 */
/* accum: y += conv instead of y = conv. gp2 (G > 0): GroupNorm + SiLU of the x2 segment (its own channel indices / groups);
   the main gnp_t then applies to the x segment alone. gp2.G == 0 keeps the legacy meaning (gnp_t over all channels). */
/* up (with x2): the x segment is stored at half resolution (D/2 x H/2 x W/2) and read as its trilinear 2x upsample
   (align_corners = false, edge clamp: the same values as nn_up2_fwd_into), so the decoder input is never materialised */
/* sr != 0: stochastic rounding (seed) of this conv's gradient operand when quantised to fp8 (the input of a backward-data
   conv, the gy of a weight gradient); 0 = round to nearest */
/* wkey != 0: id of this conv (layer / conv / pass) for the prepared-weight memo of the fp4 kernel (see lp_wmemo_*) */
/* zlo / zhi (spatial split, nn_split_cfg): the GroupNorm statistics of the output (osum) skip the first zlo and the last zhi
   z planes (halo planes owned by the other GPU); 0 / 0 = every plane */
/* stored_stats: output statistics use decoded rounded MX values, including bias exactly once. */
typedef struct { const void *x2; int c_split; void *y2; int o_split; int accum; gnp_t gp2; int up; unsigned sr; unsigned wkey; int zlo, zhi; int stored_stats; unsigned osr; int zi0, Di; int o_pad;
                 float *wpart; int wpart_x0, wpart_nx; } split_t;   /* wpart: deterministic weight-gradient CTA partials; x0/nx retain original spatial indices.
                                      o_pad: the split output's
                                                                     second tensor starts at padded row o_pad (0: o_split rounded up to 32; 16 for two <= 16-channel MX tensors) */   /* osr: exact SR seed of an MX-fp4 output (fp4 gradients); zi0 / Di: z-range (MX forward kernels: output plane z reads input plane
   z + zi0 of a depth-Di input; Di 0 = off) */
/* k=3 stride 1 forward / weight gradient with the same semantics as the BF16 tensor-core paths in nn.cu
   (conv_fwd_tc / launch_bwd_w_tc): xbf / ybf = activation (and x2 / y2) storage type, 0 fp32, 1 bf16, 2 fp16; the
   forward needs xbf == ybf. Weights, biases and GroupNorm parameters are fp32; the weight gradient accepts fp32 or (with xbf) bf16 gy. */
int lp_conv_fwd_f8(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, gnp_t gp, double *osum, int Go, split_t sp);
int lp_conv_fwd_f4(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, gnp_t gp, double *osum, int Go, split_t sp);
void lp_wmemo_step(unsigned step);   /* fp4 prepared-weight memo: new training step (weights changed) */
void lp_wmemo_clear(void);
void lp_set_w4_2d(int on);         /* 2D (32 x 32 tile) fp4 weight scales, shared by forward and backward-data (UFSM_W4_2D) */           /* weights changed outside a step (checkpoint load, EMA swap) */
int lp_bwd_w_f8(const void *x, int xbf, shape5 xs, const void *gy, int gybf, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp);   /* gybf: gy is bf16 (needs xbf) */
void lp_set_f8w_coop(int c);   /* test hook: cooperative MX staging of lp_bwd_w_f8 (bit 0 x, bit 1 gy; default 1) */
int lp_bwd_w_f4(const void *x, int xbf, shape5 xs, const void *gy, int gybf, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp, int had);   /* fp4 (e2m1, SR on gy when sp.sr); had: bit 0 fixed-sign H32 on both operands, bit 1 SR on x too (diagnostic) */
void lp_set_f4w_coop(int on);   /* test hook: cooperative MX8 gradient loads for Co <= 16, reusing the decoded-X tile */
/* the fp4 weight gradient's gy records (gy_pre4_k layout: 136 B per (n, co, z-step, 8-row tile, 16-x tile), the whole layer) written by
   the GN+SiLU backward apply that produced gy; valid until the next apply on this device, matched by gy pointer and shape (lp_bwd_w_f4
   then skips its pre-pass); gb: the per-channel sums of gy (the bias gradient) */
typedef struct { const void *gy; int n, c, d, h, w, valid, dev; uint8_t *rec; size_t cap; float *gb; int gbc; int bw; } lp_gyrec_t;   /* bw: the MX block width of the records' gy */
#ifdef __cplusplus
extern thread_local lp_gyrec_t g_gyrec;   /* per host thread (a GPU's trainer thread; split halves sharing a GPU), its own buffers */
#else
extern _Thread_local lp_gyrec_t g_gyrec;
#endif
int lp_mx_affine(void *x, int bits, shape5 s, int G, const float *mean, const float *rstd, unsigned sr);   /* src/lp_l0slab.cu */
int lp_gn_silu_bwd_apply_rec(const void *x, int xdt, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd,
                             const void *gy, void *gx, const float *AB, unsigned gsr, unsigned rsr);   /* 1: done (gx and the records) */
void lp_set_s2b_f4(int on);  /* the fp4 stride-2 backward-data (MX-fp4 gy): 3 for 32 and 64 channels, 1 for 32 only, 0 bf16 tensor cores, -1 env UFSM_S2B_F4 (default 3) */
void lp_set_s2_f4(int on);   /* the fp4 stride-2 forward (MX-fp4 32-channel input, widths % 16): 1 on, 0 fp8, -1 env UFSM_S2_F4 (default on) */
void lp_set_f4w_gypre_kb(int kb);   /* gy pre-pass of the fp4 weight gradient: slab cap in KiB (0 off, -1 env UFSM_F4W_GYPRE MiB, default 96) */
void lp_set_f4w_ws(int m);          /* warp-specialised fp4 weight gradient: 0 off, 1 on (default), 3 only >= 32 outputs; -1 env UFSM_F4W_WS */
void lp_set_gyrec_narrow(int on);   /* the GN+SiLU backward's gy records also for whole 8 / 16-channel levels (-1: env UFSM_GYREC=2) */
void lp_set_gyrec_off(int off);     /* test hook: 1 no gy records at all (the plain apply), -1 env */
int lp_gyrec_narrow(void);
int lp_conv_fwd_s2_f8(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, shape5 ys, gnp_t gp);   /* stride 2; gp: input gn+silu; 0, or 3 when the fp4 kernel ran */
void lp_set_s2_f4g(int on);   /* the coarse down convs' fp4 forward (conv_fwd_s2_f4_k GEN): 1 on, 0 off, -1 env UFSM_S2_F4G (default off) */
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
void lp_input_rad(const void *ct16, int dt, const void *rec, size_t rec_bytes, int N, int D, int P, void *y, int bits);   /* nn_input_rad */
void lp_fakeq_w4(float *w, int Co, int Ci, int T, int Bo, int Bi);   /* nn_fakeq_w4_2d / nn_fakeq_w4 */
void lp_input_rad_pack(const void *ct16, int dt, const void *rec, size_t rec_bytes, int N, int d, int p, void *y, int bits);   /* nn_input_rad_pack */
void lp_mx4_to_f32(const void *x, int N, int C, size_t S, float *y);
double lp_sr_e2m1_mean(float v, size_t n);                        /* test probe: mean of n stochastic e2m1 roundings of v (v in grid units) */
double lp_sr_e2m1_nib_mean(float v, size_t n);                    /* test probe: the same for the direct-nibble rounding of the fp4 weight gradient (|v| <= 6) */
double lp_sr_e4m3_mean(float v, size_t n);                        /* test probe: mean of n sr_e4m3 + round-to-nearest e4m3 conversions of v */
double lp_sr_e4m3x_mean(float v, size_t n);                       /* test probe: the same for the exact sr_e4m3x of the fp8 weight gradient's gy */
void lp_cvt_e2m1_probe(const float *hv, unsigned char *ho, int n);   /* test probe: raw cvt.rn.satfinite.e2m1x2 nibble of each host value */
/* MX elementwise ops: xdt / ydt = lp dtype of the operand (3 mx8, 4 mx4; gn_silu_apply also takes a plane-major 0 / 1 / 2 input);
   gradients (gy / gx, gdt) are never mx4 */
void lp_gn_silu_apply_mx(const void *x, int xdt, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd, void *y, int ydt);
void lp_add_mx(void *y, const void *x, shape5 s, int dt, unsigned sr);   /* y += x, MX (dt 4 / 8) */
void lp_up2_fwd_mx(const void *x, int xdt, shape5 xs, void *y, int ydt, gnp_t gp);   /* gp: silu(gn(x)) upsampled */
void lp_up2_fwd_mx_blocks(const void *x, int xdt, shape5 xs, int b0, int nbk, void *y, int ydt);   /* 32-channel blocks [b0, b0 + nbk) of x */
void lp_conv1_fwd_mx(const void *x, int xdt, shape5 xs, const float *w, const float *b, int cout, float *y, gnp_t gp);
void lp_conv1_fwd_mx_h16(const void *x, int xdt, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp);   /* fp16 output */
void lp_conv1_fwd_mx_h16_r(const void *x, int xdt, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, size_t vb, size_t nv);   /* voxels [vb, vb + nv) into planes of nv */
void lp_bwd_w1_mx(const void *x, int xdt, shape5 xs, const void *gy, int gydt, shape5 ys, float *gw, gnp_t gp);
int lp_bwd_w1_mx_b(const void *x, int xdt, shape5 xs, const void *gy, int gydt, shape5 ys, float *gw, float *gb, gnp_t gp);   /* also gb when it returns 1 */
int lp_head_bwd_mx_r(const void *x, int xdt, shape5 xs, const void *gy, int gydt, shape5 ys, const float *w, float *gw, float *gb, void *gx, int gxdt, gnp_t gp, unsigned sr,
                     size_t vb, size_t nv);   /* voxels [vb, vb + nv), gy planes of nv voxels */
int lp_head_bwd_mx(const void *x, int xdt, shape5 xs, const void *gy, int gydt, shape5 ys, const float *w, float *gw, float *gb, void *gx, int gxdt, gnp_t gp, unsigned sr);   /* fused head backward: 1 = done, 0 = not covered */
void lp_gn_silu_bwd_mx(const void *x, int xdt, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd,
                       const void *gy, void *gx, int gdt, double *ds, float *st, float *AB);
void lp_gn_silu_bwd_apply_mx(const void *x, int xdt, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd,
                             const void *gy, void *gx, int gdt, const float *AB, unsigned gsr);   /* gsr: SR seed of an MX-fp4 gx (gdt 4) */
/* the same for an MX-fp4 x of 8 / 16 channels whose gy is the head's input gradient recomputed from the fp16 logit gradient gl
   (ci channels, head weights wh [ci][C], the head backward-data's SR seed hs); the apply writes gx in place over x */
int lp_gn_silu_bwd_head_ok(int C, int ci);
void lp_set_gn_full8(int on);        /* GN backward: 8-channel tensors on the NK 8 kernels (1, default) or the general ones (0, test hook) */
void lp_set_gn_persist(int on);      /* GN backward applies: 1 persistent grid-stride (default), 0 a row per thread (test hook), -1 env */
void lp_set_gn_head_exact(int on);   /* the head-recomputing GN backward: 1 exact fp32 gy (default), 0 the two-op path's bits, -1 env */
/* ns > 1: s is the pseudo-sample shape of slab-local GroupNorm (s.n = samples x ns slabs of s.d planes; gl stays [sample][ci][ns s.d planes]) */
void lp_gn_silu_bwd_head_stats(const void *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd,
                               const void *gl, int ci, const float *wh, unsigned hs, double *ds, int ns);
void lp_gn_silu_bwd_head_apply(void *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd,
                               const void *gl, int ci, const float *wh, unsigned hs, const float *AB, unsigned gsr, int ns);
void lp_pred_input_mx(const uint8_t *ct, int W, float mean, float isd, const float *dyo, const float *dxo, int axis, int z0, int nz, void *y, int hd, int bits);   /* the same, MX-fp4 (bits 4) or MX-fp8 (8) */
void lp_pred_input_mx8(const uint8_t *ct, int W, float mean, float isd, const float *dyo, const float *dxo, int axis, int z0, int nz, void *y, int hd);   /* predict input planes [z0, z0 + nz) -> MX-fp8 (hd: 16-bit rounding 2 fp16 / 1 bf16 / 0 none) */
void lp_conv1_to_mx(const void *x, int gdt, int N, int Ci, size_t S, const float *w, int Co, void *y, int ydt, unsigned sr);   /* ydt 3 MX-fp8, 4 MX-fp4 (exact SR keyed by sr) */
void lp_up2_bwd_mx(const void *gy, shape5 xs, void *gx, int gdt, int xdt, unsigned osr);   /* gdt / xdt: MX dt 3 (fp8) or 4 (fp4, gx with exact SR keyed by osr) */
void lp_up2_bwd_mx_slice(const void *gy, shape5 xs, void *gx, int ctot, int c0, int gdt, int xdt, unsigned osr);   /* channels [c0, c0 + xs.c) of an MX-fp8 gx (ctot ch) */
int lp_bwd_data_s2_mx(const void *gy, shape5 ys, const float *w, shape5 xs, void *gx, int accum, int gdt, int xdt, unsigned osr);   /* returns the compute precision (3 fp4, 2 fp8, 1 bf16, 0 fp32) */
const char *lp_check(void);
void lp_f32_to_bf16(const float *x, size_t n, void *y);   /* test helpers */
void lp_bf16_to_f32(const void *x, size_t n, float *y);
int lp_gn_sums_mx(const void *x, int xdt, int N, int C, int G, size_t S, double *sums, size_t v0, size_t v1);   /* GroupNorm (sum, sum sq) per (n, group) of an MX tensor over voxels [v0, v1), accumulated; -1 unsupported */
#ifdef __cplusplus
}
#endif
