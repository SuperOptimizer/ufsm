/* vwt: our trainable ray-attention winding model (architecture after villa's winding_model.py, trained from scratch on our labels;
   NO villa weights). Shared interface of the four parts:
     src/vwt_net.c      model: parameters, forward, backward, optimiser glue (VW)
     src/nn_vwt.cu      layer kernels: conv3d / LayerNorm / GELU / resize / head / losses, fwd + bwd (reference: VW; speed, bf16 / MX: PF)
     src/vwt_sample.c   ray sampler: slabs + targets (WM: labels / holdouts / exclusions; DA: crush / deformation aug)
     src/vwt_train.c    ufsm vwt-train / vwt-predict: loop, AdamW + EMA, checkpoints (VW; launch scripts: DA)

   SLAB GEOMETRY (as villa volume_slab_extractor.py, see src/villawind.c frame_of): direction d (unit, xyz scan voxels of the CT level),
   ref = unit axis of d's smallest |component|, a = normalise(ref x d), b = d x a; voxel (i, j, k) at o + i a + j b + k d,
   o = ray_origin - (T - 1) / 2 (a + b). Arrays are [T][T][L] = [i][j][k], k (the ray) fastest. Unit spacing at ~9.6 um.

   TENSORS on the device: channels-last fp32 [N][D][H][W][C] with D = i (axis a), H = j (axis b), W = k (ray), C fastest (vt_t).
   Conv weights [co][kd][kh][kw][ci] (ci fastest: one im2col row = the K = kd kh kw ci taps in that order); linears [out][in]. */
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------- sampler (src/vwt_sample.c; WM + DA) ----------------
   One training example = one slab. The sampler owns the host buffers of the sample it returns (valid until the next vws_next call
   of the same thread slot). Everything is in slab order [T][T][L]. */
typedef struct {
    int T, L;
    uint8_t *im;        /* [T][T][L] CT uint8, trilinear (rounded) from the source's ~9.6 um level */
    uint8_t *va;        /* [T][T][L] 1 = inside the volume (villa validity), 0 = outside (im 0 there) */
    float *ph;          /* [T][T][L] target winding phase in turns, unwrapped inside each component (NAN = no label) */
    uint16_t *comp;     /* [T][T][L] unwrap component of ph (0 = no label): the loss is shift-invariant per component, so each component
                           must be internally consistent (no integer slips inside it); split rather than guess */
    float *pw;          /* [T][T][L] per-voxel target weight in [0, 1] (nullptr: 1 where comp != 0); holdout / exclusion tubes = 0 */
    double o[3], a[3], b[3], d[3];   /* frame (xyz, scan voxels of the CT level) */
    int src;            /* source index (sources json order) */
    int flags;          /* bit 0: the ray points inward (toward the umbilicus) */
} vws_sample;

typedef struct vws vws;
/* sources: the JSON list of sources (CT root / key / level, GT phase root / key / level, axis json, weight, holdouts, exclusions);
   train 1: random slabs (random source by weight, random labelled point, radial direction from the axis + random tilt <= tilt_deg and
   azimuth, random sign, the point at a random k); train 0: a fixed validation list (seeded). nthreads: background workers. */
vws *vws_open(const char *sources_json, int T, int L, float tilt_deg, int nthreads, unsigned seed, int train);
int vws_next(vws *s, vws_sample *out);   /* blocking; 0 on success */
/* the trainer's argv, passed through before vws_open (the sampler / aug parse their own flags from it, e.g. vwa_cfg_parse) */
void vws_set_args(int argc, char **argv);
void vws_close(vws *s);

/* ---------------- layer kernels (src/nn_vwt.cu; reference VW, optimisation PF). Default stream, fp32 tensors. ---------------- */
typedef struct { float *p; int n, d, h, w, c; } vt_t;
static inline size_t vt_numel(vt_t t) { return (size_t)t.n * t.d * t.h * t.w * t.c; }
static inline size_t vt_rows(vt_t t) { return (size_t)t.n * t.d * t.h * t.w; }

/* 3D convolution, zero padding k / 2, stride (sd, sh, sw), kernel k^3 (k 1 or 3): y = conv(x) + b; y preallocated with the output shape
   (D = ceil(D / sd) ...). scratch: vt_conv_scratch bytes (im2col chunks). */
size_t vt_conv_scratch(vt_t x, int co, int k, int sd, int sh, int sw);
int vt_conv_fwd(vt_t x, const float *w, const float *b, int k, int sd, int sh, int sw, vt_t y, void *scratch);
/* backward: gx = d/dx (nullptr gx.p: skipped; written, not accumulated), gw += d/dw, gb += d/db (nullptr: skipped) */
int vt_conv_bwd(vt_t x, vt_t gy, const float *w, int k, int sd, int sh, int sw, vt_t gx, float *gw, float *gb, void *scratch);
/* the paths behind vt_conv_* (dispatch: fast unless env UFSM_VWT_REF=1): vtr_* the fp32 reference (im2col + at_gemm), vtf_* bf16 (PF) */
int vtr_conv_fwd(vt_t x, const float *w, const float *b, int k, int sd, int sh, int sw, vt_t y, void *scratch);
int vtr_conv_bwd(vt_t x, vt_t gy, const float *w, int k, int sd, int sh, int sw, vt_t gx, float *gw, float *gb, void *scratch);
size_t vtf_conv_scratch(vt_t x, int co, int k, int sd, int sh, int sw);
int vtf_conv_fwd(vt_t x, const float *w, const float *b, int k, int sd, int sh, int sw, vt_t y, void *scratch);
int vtf_conv_bwd(vt_t x, vt_t gy, const float *w, int k, int sd, int sh, int sw, vt_t gx, float *gw, float *gb, void *scratch);
/* channel LayerNorm over C per row (eps 1e-5) with optional exact GELU: y = act(LN(x) g + b); mean / rstd [rows] kept for the backward.
   Backward (recomputes LN and GELU from x): gx = d/dx (may alias gy), gg / gb += */
void vt_ln_fwd(vt_t x, const float *g, const float *b, int gelu, vt_t y, float *mean, float *rstd);
void vt_ln_bwd(vt_t x, const float *g, const float *b, int gelu, const float *mean, const float *rstd, vt_t gy, vt_t gx, float *gg, float *gb);
/* x2 transverse upsample (D and H doubled, W kept), trilinear align_corners False (= PyTorch interpolate scale (2, 2, 1)); bwd = adjoint */
void vt_up2_fwd(vt_t x, vt_t y);
void vt_up2_bwd(vt_t gy, vt_t gx);
/* channel concat y = [a | b] and its split */
void vt_cat_fwd(vt_t a, vt_t b, vt_t y);
void vt_cat_bwd(vt_t gy, vt_t ga, vt_t gb);
void vt_add(float *y, const float *x, size_t n);   /* y += x */
void vt_input(const uint8_t *im, const uint8_t *va, long long n, float *x);   /* device u8 [n] -> x [n][2] (standardised image, valid) */
void vt_transpose(const float *x, float *y, long long A, long long B, int C);   /* rows [A][B][C] -> [B][A][C] */
/* phase head: logits [N][T/2][T/2][L][4] (channel si 2 + sj) -> inc [N][T][T][L] = softplus (pixel shuffle 2), ph = cumsum_k inc.
   Backward from d/d ph and d/d inc (either nullptr) to d/d logits. */
void vt_phase_fwd(vt_t lg, float *inc, float *ph);
void vt_phase_bwd(vt_t lg, const float *gph, const float *ginc, vt_t glg);
/* losses over one slab (N 1), host-visible scalars; gradients written into gph / ginc (scaled by `scale`):
   phase: Huber(delta_p) of (ph - mean_c ph) - (tg - mean_c tg) per unwrap component c, weighted by pw, normalised by the total weight;
   density: Huber(delta_d) of inc[k] - (tg[k] - tg[k-1]) where comp[k] == comp[k-1] != 0, weight min(pw[k], pw[k-1]). */
typedef struct { double phase, density, wsum, dsum; } vt_loss;
int vt_losses(const float *ph, const float *inc, const float *tg, const uint16_t *comp, const float *pw, int T, int L, float delta_p, float delta_d,
              float lam_p, float lam_d, float scale, float *gph, float *ginc, vt_loss *out);

/* ---------------- model (src/vwt_net.c; VW) ---------------- */
typedef struct {
    int T, L;                  /* slab (128, 384) */
    int ch[3], trunk, stem, blocks, heads, mlp, dec;   /* 32 64 96, 192, 2, 6 (each = ray block + transverse block), 4 (head dim 48), 256, 96 */
    float phase_init;          /* initial increment (0.04) */
} vwt_cfg;
typedef struct vwt vwt;
vwt *vwt_create(const vwt_cfg *c, unsigned seed);
size_t vwt_nparams(const vwt *m);
float *vwt_params(vwt *m);     /* device, flat */
float *vwt_grads(vwt *m);      /* device, flat (accumulated by vwt_backward; zeroed by vwt_zero_grad) */
void vwt_zero_grad(vwt *m);
/* forward of one slab (im / va device uint8 [T][T][L]) -> ph / inc device fp32 [T][T][L]; train 1 keeps the activations */
int vwt_forward(vwt *m, const uint8_t *im, const uint8_t *va, float *ph, float *inc, int train);
int vwt_backward(vwt *m, const float *gph, const float *ginc);   /* after vwt_forward(.., 1): gradients += */
void vwt_free(vwt *m);
int vwt_save(const vwt *m, const char *path, const float *ema, const void *extra, size_t extra_bytes);
vwt *vwt_load(const char *path, int use_ema);

#ifdef __cplusplus
}
#endif
