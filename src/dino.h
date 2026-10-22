/* dino: our own dense embedding network trained from scratch (ufsm dino-train, src/dino.c), kernels in src/nn_dino.cu.
   The U-Net (src/unet.h) with cout = D embedding channels; losses on sampled voxels only:
   - GT sheet identity (primary): supervised contrastive (SupCon) over labelled voxels of a window; positives = same wrap (winding codes,
     round(dq) == 0 anywhere in the window), negatives = other wraps, the adjacent wraps (|dk| == 1) within a radius weighted up (hard);
   - SSL regulariser: DINO-style prototype cross-entropy between the student on one view and the EMA teacher on the other view at the same
     papyrus voxels (teacher centring + sharpening), few prototypes.
   Everything on device, fp32, small point sets (N <= ~16k). */
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { int perm[3], flip[3]; } dn_geom;   /* output axis d reads source axis perm[d], reversed when flip[d] */

/* the network input: x[o] = ((ct[src(o)] - center) / scale) gain + bias + sigma N(0, 1) (hash noise of seed), cubes of the mask grid
   (edge mcube, masked when hash(cube, mseed) < mfrac) set to mval; ct u8 [P^3] on the device, x float [P^3] */
void dn_input(const uint8_t *ct, float *x, int P, dn_geom g, float center, float scale, float gain, float bias, float sigma, uint32_t seed,
              int mcube, float mfrac, uint32_t mseed, float mval);
/* embeddings at points: logits [D][P^3] (fp16 if h16 else fp32), idx [N] voxel indices -> e [N][D] unit vectors, nrm [N] the norms */
void dn_gather(const void *logits, int h16, size_t p3, int D, const int *idx, int N, float *e, float *nrm);
/* SupCon over anchors a (N x D unit, student) with keys = the anchors and N2 extra detached keys t (teacher, may be 0): per-anchor labels
   code (winding code 1..254) and voxel coordinates xyz [N][3] (keys the same, t's after the anchors'); tau temperature; hard negatives
   (|dk| == 1 within hr voxels) weigh hw. Writes ga [N][D] (d loss / d a, mean over anchors with a positive) and returns the loss via
   out[0], positives' / negatives' mean cosine out[1], out[2], anchors with positives out[3] (device float[4]); scratch (2 N + 8) floats */
void dn_supcon(const float *a, const int *code, const int *xyz, int N, const float *t, const int *tcode, const int *txyz, int N2, int D,
               float tau, float hw, float hr, float *ga, float *out, float *scratch);
/* DINO prototype loss: student s [N][D] unit, teacher t [N][D] unit, prototypes W [K][D] (student, raw) and Wt [K][D] (teacher), center [K];
   loss mean_i sum_k -softmax((t_i.ct_k - center_k) / tt) log softmax(s_i.c_k / ts) (c = W / |W|). Writes gs [N][D], gW [K][D], the batch
   mean of the teacher's raw logits into tmean [K]; out[0] loss, out[1] teacher entropy of the batch-mean assignment (nats), out[2] mean
   per-point teacher entropy */
size_t dn_dino_scratch(int N, int K);   /* scratch bytes of dn_dino */
void dn_dino(const float *s, const float *t, int N, const float *W, const float *Wt, const float *center, int K, int D, float ts, float tt,
             float *gs, float *gW, float *tmean, float *out, float *scratch);
/* back through e = y / |y| and into the dense fp32 logit gradient: gl[c][idx[i]] += w (ge_i - e_i (e_i . ge_i)) / nrm_i */
void dn_scatter_grad(const float *e, const float *ge, const float *nrm, const int *idx, int N, int D, size_t p3, float w, float *gl);
/* Adam on a small flat parameter array (prototypes) + EMA copy */
void dn_adam(float *p, const float *g, float *m, float *v, size_t n, float lr, float b1, float b2, float eps, float wd, int step);
void dn_ema(float *pt, const float *p, size_t n, float decay);
void dn_axpby(float *y, const float *x, size_t n, float a, float b);   /* y = a y + b x */
/* per-dimension std of e [N][D], averaged over dimensions (collapse monitor) -> out[0] */
void dn_emb_std(const float *e, int N, int D, float *out);
/* unit int8 embeddings of a tile: q[v][c] = round(127 y / |y|) for interior voxels; logits [D][P^3]; out box-local [nz][ny][nx][D] at
   (oz, oy, ox) receives tile voxels [h, P - h) offset by (tz, ty, tx); masked (ct == 0) voxels zero */
void dn_tile_out(const void *logits, int h16, int P, int D, const uint8_t *ct, int z0, int z1, int y0, int y1, int x0, int x1,
                 int8_t *out, int onz, int ony, int onx, int ooz, int ooy, int oox);

#ifdef __cplusplus
}
#endif
