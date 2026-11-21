#pragma once
/* The surface-aligned ink head (DESIGN 2026-11-20 "inksurf"): the trunk's level-0 features resampled onto a segment's mesh as a flattened
   patch (K layers along the normal, trilinear), a small fp32 2-D U-Net on it, BCE straight on the 2-D labels. Kernels in src/nn_inksurf.cu,
   the head and the inverse lists in src/inksurf.c. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __CUDACC__
#define ISF_HD __host__ __device__ __forceinline__
#else
#define ISF_HD static inline
#endif
enum { ISF_F32 = 0, ISF_F16 = 1, ISF_BF16 = 2 };
/* a sample: q = {P z, y, x, N z, y, x} in feature voxel coordinates (voxel centres at integers), t along N */
ISF_HD void isf_pos(const float *q, float t, float *z, float *y, float *x) { *z = q[0] + t * q[3]; *y = q[1] + t * q[4]; *x = q[2] + t * q[5]; }
ISF_HD void isf_cell(float v, int *i, float *w) { const float f = floorf(v); *i = (int)f; *w = v - f; }

#ifdef __cplusplus
extern "C" {
#endif
/* out[(c K + j) n + p] = trilinear feat[c] (C x D x H x W, ftype) at P_p + (j - (K - 1) / 2) step N_p; pn [n][6] device */
void nn_isf_resample(const void *feat, int ftype, int C, int D, int H, int W, const float *pn, int n, int K, float step, float *out);
/* the adjoint over inverse lists (isf_lists): g[c S + vox[u]] = (acc ? g : 0) + scale sum_e ew[e] gout[(c K + j_e) n + p_e], e in
   [off[u], off[u + 1]), entry ent[e] = j n + p; voxels not listed are left alone (zero g first) */
void nn_isf_gather(const float *gout, int C, int K, int n, const int *vox, const int *off, const int *ent, const float *ew, int nv,
                    void *g, int gtype, size_t S, float scale, const float *dscale, int acc);   /* vox nullptr: g index u; dscale (device, nullptr 1): x dscale[0] */
/* the compact feature path (train --ink-surf): fc / gc [C][nv] over the touched voxels (vox ascending window-voxel indices) */
void nn_isf_take(const void *feat16, int C, size_t ns, size_t vb, const int *vox, int ua, int ub, float *fc, int nv);
void nn_isf_put(const float *gc, int C, int nv, const int *vox, int ua, int ub, size_t vb, size_t ns, void *rows16);
void nn_isf_resample_c(const float *fc, int C, int nv, const int *fidx, const float *fw, int n, int K, float *out);
/* 2-D fp32 ops, [C][H][W], batch 1; convs pad k / 2, weights [Co][Ci][k][k] */
void nn_isf2_conv(const float *x, int Ci, int H, int W, const float *w, const float *b, int Co, int k, float *y);
void nn_isf2_conv_bd(const float *gy, int Co, int H, int W, const float *w, int Ci, int k, float *gx, int acc);
void nn_isf2_conv_bw(const float *x, int Ci, int H, int W, const float *gy, int Co, int k, float *gw, float *gb, int acc);   /* acc 0: gw, gb set; 1: added */
/* the simple one-thread-per-output kernels the tiled ones above must match bit for bit (test_inksurf, bench_isfhead) */
void nn_isf2_conv_ref(const float *x, int Ci, int H, int W, const float *w, const float *b, int Co, int k, float *y);
void nn_isf2_conv_bd_ref(const float *gy, int Co, int H, int W, const float *w, int Ci, int k, float *gx, int acc);
void nn_isf2_conv_bw_ref(const float *x, int Ci, int H, int W, const float *gy, int Co, int k, float *gw, float *gb, int acc);
void nn_isf2_add(float *y, const float *x, size_t n);   /* y += x */
void nn_isf2_silu(const float *x, size_t n, float *y);
void nn_isf2_silu_bwd(const float *x, const float *gy, size_t n, float *gx);   /* x: the pre-activation */
void nn_isf2_pool(const float *x, int C, int H, int W, float *y);
void nn_isf2_pool_bwd(const float *gy, int C, int H, int W, float *gx, int acc);
void nn_isf2_up(const float *x, int C, int h, int w, float *y);
void nn_isf2_up_bwd(const float *gy, int C, int h, int w, float *gx);
void nn_isf2_bce_grad(const float *z, const float *t, const float *w, size_t n, float inv_wsum, float *g);
void nn_isf2_adamw(float *p, const float *g, float *m, float *v, size_t n, float lr, float b1, float b2, float eps, float wd, int step);

/* ---- host side (src/inksurf.c) ---- */
/* inverse lists of the resample of n samples (pn host [n][6]) x K layers into a D x H x W grid: per touched voxel (vox sorted ascending)
   its entries (j n + p, weight) in increasing j n + p; arrays malloc'd. 0 ok */
typedef struct { int nv, ne; int *vox, *off, *ent; float *ew; } isf_lists;
int isf_lists_build(const float *pn, int n, int K, float step, int D, int H, int W, isf_lists *L);
void isf_lists_free(isf_lists *L);
/* the 2-D head: stem 1x1 (Cin -> 32), a 4-down U-Net (32 48 64 96, bottleneck 2 x 96, decoder 64 48 32 32), 1x1 -> 1 logit; SiLU; plus a
   linear readout (1x1 from the patch to the logit, summed) */
typedef struct isfhead isfhead;
isfhead *isfhead_create(int cin, int Hp, int Wp, uint64_t seed);   /* Hp, Wp multiples of 16; depth 4 */
isfhead *isfhead_create_d(int cin, int Hp, int Wp, int depth, uint64_t seed);   /* depth 1 .. 6 pools (S3: 5); Hp, Wp multiples of 16 and 2^depth */
int isfhead_depth(const isfhead *h);
void isfhead_free(isfhead *h);
size_t isfhead_nparams(const isfhead *h);
int isfhead_nconv(const isfhead *h);
size_t isfhead_conv_span(const isfhead *h, int i, size_t *n);   /* conv i's weights + bias: offset, count */
float *isfhead_params(isfhead *h);   /* device */
float *isfhead_grads(isfhead *h);
const float *isfhead_forward(isfhead *h, const float *x);   /* x [cin][Hp][Wp] device; returns the logits [Hp][Wp] (device) */
const float *isfhead_backward(isfhead *h, const float *glogit);   /* adds to the parameter gradients (isfhead_zero_grad); returns d loss / d x [cin][Hp][Wp] */
void isfhead_zero_grad(isfhead *h);
void isfhead_adamw(isfhead *h, float lr, float wd, int step);
size_t isfhead_bytes(const isfhead *h);   /* device bytes (parameters, moments, activations) */
float *isfhead_moments(isfhead *h, int k);   /* Adam m (0) / v (1), device */

/* ---- train --ink-surf (src/inksurf.c) ---- */
#ifndef __CUDACC__
#include "inksplat.h"
#include "unet.h"
/* one window's patch (host): Hp x Hp px on the label grid every step-th row / col of the unit with the most rays, pn [n][6] window level
   voxels (far outside: no ray / masked), lab, wt (1 labelled, 0 not), the inverse lists + forward table of the K-layer resample */
typedef struct { int Hp, K, np, unit, nouv; float kstep; double frac; float *pn, *lab, *wt; size_t cap; isf_lists L; int *fidx; float *fw; size_t fcap; } isf_win;
int isf_win_build(isf_win *w, const inkray *r, int nr, int Hp, int step, int K, float kstep, int D, int H, int W, unsigned aug, int soft);   /* aug: 1 transpose, 2 / 4 flip rows / cols,
                                                                                                                                   8 reverse the normal (train --isf-aug) */
void isf_win_free(isf_win *w);
int isf_win_alloc(isf_win *w, int Hp, int K, float kstep);   /* pn / lab / wt for Hp x Hp (the caller fills them, sets np) */
int isf_win_lists(isf_win *w, int D, int H, int W);           /* the lists + forward table of the filled pn */
/* the device side: the head, the window's lists, the compact features fc / gradient gc [C][nv], slab scratch */
typedef struct { isfhead *h; int C, Hp, K, nv; float *patch, *lab, *wt, *glog, *zh; int *vox, *off, *ent, *fidx; float *ew, *fw, *fc, *gc; void *feat, *rows;
                 size_t c_vox, c_off, c_ent, c_ew, c_fidx, c_fw, c_fc, c_gc, c_feat, c_rows; } isf_dev;
int isf_dev_init(isf_dev *g, int C, int Hp, int K, uint64_t seed);
int isf_dev_init_d(isf_dev *g, int C, int Hp, int K, int depth, uint64_t seed);
size_t isf_reserve_bytes(int C, int Hp, int K, int depth, int D, int H, int W);   /* the planner's reservation for train --ink-surf */
void isf_dev_free(isf_dev *g);
size_t isf_dev_bytes(const isf_dev *g);
int isf_dev_upload(isf_dev *g, const isf_win *w);                                   /* the window's lists and labels (synchronous) */
int isf_dev_take(isf_dev *g, unet *u, const isf_win *w, int z0, int z1, size_t hw);  /* after a forward, per head slab: the slab's touched features */
/* after every slab's take: resample, head forward, the mean weighted BCE (returned; -1: no labelled px); train: the head's gradients (added)
   and the compact feature gradient x gscale x dscale[0] (device; nullptr 1). auc: the patch AUC (nullptr: not computed) */
double isf_dev_step(isf_dev *g, const isf_win *w, float lambda, int train, float gscale, const float *dscale, double *auc);
int isf_dev_put(isf_dev *g, unet *u, const isf_win *w, int z0, int z1, size_t hw);   /* per gradient-pass slab: sets the unet's feature-gradient rows */
int isf_save(const isf_dev *g, const char *path, int step);
int isf_load(isf_dev *g, const char *path, int *step);
int isf_load_x(isf_dev *g, const char *path, int *step, int *mapped);   /* another depth: the matching convs (role, level, shape), *step 0 */
int isf_peek(const char *path, int *cin, int *depth, int *step);
#endif
#ifdef __cplusplus
}
#endif
