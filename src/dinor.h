/* dinor: the upstream dinovol (3D DINOv3-style) training recipe for our own network (dino-train --recipe, src/dino.c), kernels in
   src/nn_dinor.cu. Pieces (all fp32, on the default stream):
   - multi-crop: each sample's source crop (u8 [Ps^3]) -> 2 global + several local cubic crops, each with its own cube symmetry
     (axis permutation + flips), offset and intensity augmentation; the exact crop -> source map is kept (dr_xf) for the
     cross-view terms;
   - DINO head: (LayerNorm'd CLS features) -> MLP (GELU) -> bottleneck -> L2 normalise -> weight-normalised prototypes (cosine logits);
   - teacher targets: softmax with centring, or Sinkhorn-Knopp (DINOv2 / v3), temperature tt; student log-softmax at ts;
   - multi-crop cross-entropy over a list of (student row, teacher row) pairs;
   - KoLeo on L2-normalised features (DINOv2 form). */
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* crop view -> source map (upstream random resized crop + cube symmetry): the view is a cube of edge n; view axis d reads the resized
   crop's axis perm[d] (reversed when flip[d]); the resized crop samples the source cube of edge m at offset off[] (source axes z, y, x)
   linearly with voxel centres aligned: source = off + (o + 0.5) m / n - 0.5 (o = resized-crop coordinate); src = the sample whose
   source crop it reads */
typedef struct { int off[3], perm[3], flip[3], n, m, src; } dr_xf;
static inline void dr_map(const dr_xf *t, const double v[3], double s[3]) {   /* view coordinates -> source */
    for (int d = 0; d < 3; d++) { const double o = t->flip[d] ? (double)(t->n - 1) - v[d] : v[d]; s[t->perm[d]] = t->off[t->perm[d]] + (o + 0.5) * t->m / t->n - 0.5; }
}
static inline void dr_unmap(const dr_xf *t, const double s[3], double v[3]) {   /* source -> view */
    for (int d = 0; d < 3; d++) { const double o = (s[t->perm[d]] - t->off[t->perm[d]] + 0.5) * t->n / t->m - 0.5; v[d] = t->flip[d] ? (double)(t->n - 1) - o : o; }
}
/* per-view intensity: y = ((v - center) / scale) (trilinear source value v, borders clamped), then gamma on the [lo, hi] range of the
   normalised values (gamma != 1), contrast about the view mean (contrast != 1: y = mean + c (y - mean); mean passed in), + bias,
   inverted (y = -y) when invert, + sigma N(0, 1) hash noise of seed */
typedef struct { float center, scale, gamma, lo, hi, contrast, mean, bias, sigma; int invert; uint32_t seed; } dr_int;
/* ncrop views of edge n from the source crops src [nsrc][Ps^3]: x [ncrop][n^3]; xf / it on the device */
void dr_crops(const uint8_t *src, int Ps, const dr_xf *xf, const dr_int *it, int ncrop, int n, float *x);

/* iBOT input masking: voxels of the masked token cells (edge ps, mask [G^3] u8 on the device, G = n / ps) of an n^3 view set to val */
void dr_mask_cells(float *x, int n, int ps, const uint8_t *mask, float val);

/* DINO head (upstream DINOHead, nlayers 3, no BN): x [R][Cin] -> GELU(x W1^T + b1) -> GELU(. W2^T + b2) -> . W3^T + b3 = z [R][B] -> zn =
   z / max(|z|, 1e-12) -> logits = zn . (g_k v_k / |v_k|) [R][K] (weight-normalised bias-free last layer; g [K] the weight-norm scales).
   Offsets into one flat array: W1 [H][Cin], b1 [H], W2 [H][H], b2 [H], W3 [B][H], b3 [B], V [K][B], g [K] */
typedef struct { int Cin, H, B, K; size_t w1, b1, w2, b2, w3, b3, v, g; size_t n; } dr_head;
void dr_head_layout(dr_head *h, int Cin, int H, int B, int K);
/* scratch for R rows plus the normalised prototypes */
typedef struct { float *a1, *h1, *a2, *h2, *z, *zn, *zr, *vn, *vr, *dzn, *dh, *dh2, *dvn, *dg;
                 float *zc; float zc_m; } dr_hws;   /* zc [B] (optional, caller-owned): the forward subtracts it from z before the row norm (a constant
                                                       shift: the backward is unchanged); zc_m > 0: the forward first updates zc = m zc + (1 - m) mean_r z */
size_t dr_hws_bytes(const dr_head *h, int R);
void dr_hws_carve(dr_hws *w, const dr_head *h, int R, void *base);
/* x [R][Cin] -> logits [R][K] */
int dr_head_fwd(const dr_head *h, const float *P, const float *x, int R, float *logits, dr_hws *w);
/* d loss / d logits [R][K] -> parameter gradients += into G (last: 0 skips V and g, the frozen last layer), d loss / d x [R][Cin] written */
int dr_head_bwd(const dr_head *h, const float *P, float *G, const float *x, const float *dlogits, int R, float *dx, int last, dr_hws *w);

/* teacher targets q [R][K] from teacher logits [R][K]: Sinkhorn-Knopp with iters iterations at temperature tt (iters > 0), or
   softmax((l - center) / tt) (iters 0, center [K] or nullptr); scratch: dr_sk_scratch(R, K) bytes */
size_t dr_sk_scratch(int R, int K);
void dr_teacher_probs(const float *logits, int R, int K, float tt, int iters, const float *center, float *q, void *scratch);
/* centre update (iters 0 path): center = m center + (1 - m) mean_r logits[r] */
void dr_center_update(float *center, const float *logits, int R, int K, float m);
/* multi-crop cross-entropy: npair pairs (s[i], t[i], weight w[i]) on the device; L = sum_i w_i sum_k -q[t_i][k] log softmax(ls[s_i] / ts)[k];
   dls [Rs][K] (written) = d L / d ls; loss -> out[0] (device); scratch dr_ce_scratch(Rs) bytes */
size_t dr_ce_scratch(int Rs);
void dr_ce(const float *ls, int Rs, int K, float ts, const float *q, const int *ps, const int *pt, const float *pw, int npair, float *dls, float *out, void *scratch);
/* KoLeo (upstream KoLeoLoss): xh = x / max(|x|, 1e-8) over rows [N][D]; nearest neighbour n(i) by the largest dot product among the rows of
   i's group (contiguous groups of gs rows; gs 0: all), i excluded; L = -w mean_i log(|xh_i - xh_n(i) + 1e-8| + 1e-8) (the pairwise distance
   adds eps to the difference); d L / d x added into gx (n(i) fixed); loss -> out[0]. N <= 1024 */
void dr_koleo(const float *x, int N, int D, int gs, float w, float *gx, float *out);
/* the same against M candidate rows (a queue of earlier features, no gradient; batch-1 training): n(i) = the candidate with the largest dot
   product with xh_i; N <= 1024, M >= 1 */
void dr_koleo_q(const float *x, int N, const float *cand, int M, int D, float w, float *gx, float *out);
/* the average pool of token rows [n][S][C] over rows [r0, S) of each sequence -> cls [n][C]; backward: g [n][C] spread into gt (added) */
void dr_pool_fwd(const float *t, int n, int S, int C, int r0, float *cls);
void dr_pool_bwd(const float *g, int n, int S, int C, int r0, float *gt);
/* parameter-free layer norm of R rows of C (the CLS before the DINO head / KoLeo: upstream's final LayerNorm): y = (x - mean) / sqrt(var + 1e-6),
   in place allowed; rstd [R] kept for the backward: dx = rstd (dy - mean(dy) - y mean(dy y)) (dx may be dy) */
void dr_rownorm_fwd(const float *x, int R, int C, float *y, float *rstd);
void dr_rownorm_bwd(const float *dy, const float *y, const float *rstd, int R, int C, float *dx);
/* the argmax prototype of each of R rows of q [R][K] -> out [R] (device ints; the prototype usage monitor) */
void dr_argmax_rows(const float *q, int R, int K, int *out);
/* rows of v [R][N] scaled to unit length in place */
void dr_unit_rows(float *v, int R, int N);
/* object / region terms (upstream ObjectLoss, L_out SupCon with margin-capped positives): anchors S [N][D] (student, unit) vs keys T [N][D]
   (teacher, the same points in the other view, unit); obj [N] (> 0 labelled), vs / vt [N] validity (u8 device). Positives: the same object
   (the self point excluded when another exists), negatives: other labelled objects; per anchor (1 / |P|) sum_P softplus(lse_N(s / t) -
   min(s, margin) / t); the mean over anchors, times w, -> out[0] (+=), anchors -> out[1] (+=); d / d S added into gS. The region term:
   the same with one row per object (object-mean embeddings through the region head) */
void dr_obj(const float *S, const float *T, const int *obj, const uint8_t *vs, const uint8_t *vt, int N, int D, float temp, float margin, float w, float *gS, float *out);
/* the ink term for one anchor family (upstream _ink_loss): anchors A [Na][D] (unit) with point ids aid (or nullptr), pull keys Kp [np][D]
   with point ids kid (-1: queued keys; the anchor's own point left out while another pull key exists), push keys Kn [nn][D]; per anchor
   0.5 (softplus(m_push - m_pull) + softplus(lme_push - lme_pull)), pull logits capped at max(margin, max push sim + T); mean over anchors
   times w -> out[0] (+=), d / d A added into gA. Ink anchors: pull = ink keys, push = no-ink keys; no-ink anchors: the roles swapped */
void dr_ink(const float *A, const int *aid, int Na, const float *Kp, const int *kid, int np, const float *Kn, int nn, int D, float T, float margin, float w, float *gA, float *out);
/* InfoNCE over a key bank (the batch-1 self-supervision: MoCo-style CLS, DenseCL-style cross-view tokens): anchors Q [N][D] (unit), keys
   K [M][D] (unit, no gradient: the EMA teacher's in-batch keys first, then the queue), pos[i] the row of anchor i's positive; per anchor
   loss lse_m(q.k_m / tau) - q.k_pos / tau; out[0] += w mean loss, out[1] += mean top-1 (the positive ranked first); gQ += d (w mean loss) / d Q.
   scratch: dr_nce_scratch(N, M) bytes */
size_t dr_nce_scratch(int N, int M);
void dr_infonce(const float *Q, int N, const float *K, int M, const int *pos, int D, float tau, float w, float *gQ, float *out, void *scratch);
/* the same with a positive per anchor P [N][D] and only shared negatives Kn [M][D] (the queue: keys of other samples), so the same sample's
   other keys never act as negatives (a sheet-identity embedding makes a view's points near duplicates): loss_i = lse(q.p_i, q.n_m) / tau -
   q.p_i / tau; top-1 = the positive above every negative. scratch: dr_nce_scratch(N, M + 1) bytes */
void dr_infonce_p(const float *Q, const float *P, int N, const float *Kn, int M, int D, float tau, float w, float *gQ, float *out, void *scratch);
/* rows by index: dst[i] = src[idx[i]] ([n][C]); scatter: dst[idx[i]] += scale src[i] (indices distinct) */
void dr_rows_gather(const float *src, int C, const int *idx, int n, float *dst);
void dr_rows_scatter_add(const float *src, const int *idx, int n, int C, float scale, float *dst);
/* per-object means of point rows x [N][D] by grp [N] (-1 none) -> m [M][D] and counts; backward adds gm / count to the points' rows */
void dr_gmean(const float *x, const int *grp, int N, int D, int M, float *m, float *cnt);
void dr_gmean_bwd(const float *gm, const float *cnt, const int *grp, int N, int D, float *gx);
/* the region head: Linear (Cin -> H) -> GELU -> Linear (H -> O) -> L2 normalisation; ws: dr_mlp2_ws(R) bytes (the forward's state for the
   backward); bwd adds d / d x into dx */
typedef struct { int Cin, H, O; size_t w1, b1, w2, b2, n; } dr_mlp2;
void dr_mlp2_layout(dr_mlp2 *m, int Cin, int H, int O);
size_t dr_mlp2_ws(const dr_mlp2 *m, int R);
int dr_mlp2_fwd(const dr_mlp2 *m, const float *P, const float *x, int R, float *y, void *ws);
int dr_mlp2_bwd(const dr_mlp2 *m, const float *P, float *G, const float *x, const float *y, const float *dy, int R, float *dx, void *ws);
/* small optimiser pieces for the head: AdamW and EMA over a flat array */
void dr_adamw(float *p, const float *g, float *m, float *v, size_t n, float lr, float b1, float b2, float eps, float wd, int step);
void dr_ema(float *pt, const float *p, size_t n, float mom);
/* Muon (as nn_muon: nesterov, Newton-Schulz 5, sqrt(max(1, Co / K)) scale, decoupled decay) for one [Co][K] matrix, the Newton-Schulz
   products on cuBLASLt GEMMs over the smaller side (large head matrices); work: dr_muon_work bytes */
size_t dr_muon_work(int Co, int K);
void dr_muon(float *p, const float *g, float *mom, int Co, int K, float lr, float beta, float wd, float *work);

#ifdef __cplusplus
}
#endif
