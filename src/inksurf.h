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
                    void *g, int gtype, size_t S, float scale, int acc);
/* 2-D fp32 ops, [C][H][W], batch 1; convs pad k / 2, weights [Co][Ci][k][k] */
void nn_isf2_conv(const float *x, int Ci, int H, int W, const float *w, const float *b, int Co, int k, float *y);
void nn_isf2_conv_bd(const float *gy, int Co, int H, int W, const float *w, int Ci, int k, float *gx, int acc);
void nn_isf2_conv_bw(const float *x, int Ci, int H, int W, const float *gy, int Co, int k, float *gw, float *gb);   /* gw, gb SET */
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
isfhead *isfhead_create(int cin, int Hp, int Wp, uint64_t seed);   /* Hp, Wp multiples of 16 */
void isfhead_free(isfhead *h);
size_t isfhead_nparams(const isfhead *h);
int isfhead_nconv(void);
size_t isfhead_conv_span(const isfhead *h, int i, size_t *n);   /* conv i's weights + bias: offset, count */
float *isfhead_params(isfhead *h);   /* device */
float *isfhead_grads(isfhead *h);
const float *isfhead_forward(isfhead *h, const float *x);   /* x [cin][Hp][Wp] device; returns the logits [Hp][Wp] (device) */
const float *isfhead_backward(isfhead *h, const float *glogit);   /* sets the parameter gradients; returns d loss / d x [cin][Hp][Wp] */
void isfhead_adamw(isfhead *h, float lr, float wd, int step);
size_t isfhead_bytes(const isfhead *h);   /* device bytes (parameters, moments, activations) */
#ifdef __cplusplus
}
#endif
