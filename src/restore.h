/* Task restore (CT super-resolution, train --task restore / predict on a restore checkpoint): the clean window is the target,
   its degraded copy on the same (fine) grid the input of a residual net (out = input + 64 logit, input (x - 128) / 64).

   Degradation families (restore_recipe, built on the GPU from the clean window, src/nn_restore.cu):
     identity  input = target (the net must leave sharp CT alone)
     pyramid   exact 4x block mean (the published zarr pyramid: level 2 of a 2.4 um scan is the 9.6 um rung), rounded to uint8, the
               published codec of that rung (volcomp q 2 on level 2; q 4 on level 1, q 8 on level 0) on the coarse grid, then trilinear
               back onto the fine grid
     scanner   a native coarse scan (8.64 / 9.36 um scans enlarged ~3.6-3.9x): a Gaussian PSF of sigma psf fine voxels sampled on a coarse
               grid of step 3.4-4.4 fine voxels at a random sub-voxel phase, then on the coarse grid a reconstruction tone curve (gain,
               gamma, bias), signal-dependent spatially correlated noise, an optional reconstruction-filter edge term, uint8 rounding and
               the native level-0 codec (volcomp q 8), then trilinear back onto the fine grid
   Every stage is separable or local, so the scratch is ~1.4 bytes per fine voxel (restore_scratch_bytes). The coarse grid of sample i
   (0-based) is centred on fine coordinate (i + 0.5 - phase) step - 0.5; the upsampling inverts the same map, so predict on a real
   coarse volume (phase 0, step = coarse um / fine um) sees exactly the training geometry. */
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

enum { RF_IDENTITY = 0, RF_PYRAMID = 1, RF_SCANNER = 2, RF_N = 3 };
extern const char *const restore_family_names[RF_N];

typedef struct {
    int family;
    int small[3];       /* coarse grid (z, y, x) */
    float step[3];      /* fine voxels per coarse voxel along each axis (fine extent / small) */
    float phase[3];     /* coarse grid offset in coarse voxels, |phase| <= 0.5 (0 for the pyramid) */
    float psf[3];       /* scanner: Gaussian sigma in fine voxels (truncated at 4 sigma); the pyramid uses the exact block mean */
    int box;            /* 1: exact integer block mean of step[] (pyramid) */
    float gain, gamma, bias;                      /* scanner tone: 255 gain (v / 255)^gamma + bias on the coarse grid */
    float noise, sig_var, corr, edge;             /* noise sigma (DN), signal-dependent variance (DN^2 per DN), neighbour correlation
                                                     0..1 (6-neighbour mix), reconstruction edge term v += edge (v - mean of 6) */
    float q; int qphase[3];                       /* codec on the coarse grid: volcomp q (0 = none) and its 16^3 block phase */
    unsigned seed;                                /* noise hash seed */
} restore_recipe;

/* the ranges a family draws from (train --restore-* flags; DESIGN.md "Task restore" records the calibration) */
typedef struct {
    float mix[RF_N];                 /* family probabilities (normalised) */
    float pyr_q;                     /* pyramid codec q (2: the published 9.6 um rung of 2.4 um scans) */
    float sc_step[2];                /* scanner coarse step (fine voxels), uniform */
    float sc_psf[2];                 /* scanner PSF sigma in coarse voxels (x step = fine voxels), uniform */
    float sc_aniso;                  /* per-axis PSF factor exp(U(-a, a)) */
    float sc_noise[2], sc_sigvar[2], sc_corr[2];   /* uniform */
    float sc_edge[2], sc_edge_p;     /* edge term uniform with probability sc_edge_p */
    float sc_gain, sc_gamma, sc_bias;   /* gain, gamma exp(U(-x, x)); bias U(-x, x) DN */
    float sc_q;                      /* scanner codec q (8: the native level-0 codec) */
} restore_ranges;
void restore_ranges_default(restore_ranges *g);
/* parse "pyramid:0.45,scanner:0.45,identity:0.1" into g->mix; 0 on success */
int restore_parse_mix(restore_ranges *g, const char *s);
/* a recipe of family f for a D x H x W fine window from a 64-bit seed (deterministic) */
void restore_recipe_draw(restore_recipe *r, int family, uint64_t seed, int D, int H, int W, const restore_ranges *g);
/* the family drawn by u in [0, 1) under g->mix */
int restore_pick_family(const restore_ranges *g, double u);

/* ---- GPU (src/nn_restore.cu); every pointer below is device memory unless stated ---- */
/* scratch bytes for windows up to D x H x W and coarse steps >= min_step */
size_t restore_scratch_bytes(int D, int H, int W, float min_step);
/* clean (D x H x W uint8, 0 = outside the scan) -> low (same grid, 0 where clean is 0). coarse (optional, small[] uint8): the coarse grid
   after rounding and the codec (the scan as stored). Returns 0, or -1 for a recipe the scratch cannot hold */
int nn_restore_degrade(const uint8_t *clean, uint8_t *low, int D, int H, int W, const restore_recipe *r, void *scratch, size_t scratch_bytes, uint8_t *coarse);
/* coarse uint8 (sd x sh x sw) -> fine (D x H x W) trilinear with the recipe geometry (fine voxel x reads coarse (x + 0.5) / step - 0.5 + phase);
   mask: fine voxels where mask is 0 become 0 (nullptr: where the nearest coarse voxel is 0). Values clamped to [1, 255] elsewhere */
void nn_restore_upsample(const uint8_t *small, int sd, int sh, int sw, uint8_t *out, int D, int H, int W, const float step[3], const float phase[3], const uint8_t *mask);
/* volcomp lossy round trip (16^3 DCT blocks, the encoder's dead-zone quantiser and the decoder's reconstruction offsets) of a d x h x w
   uint8 volume in place; blocks start at -phase (0..15) per axis, partial blocks edge-replicated as in volcomp's chunk padding */
void nn_codec_roundtrip(uint8_t *vol, int d, int h, int w, float q, const int phase[3]);
/* network input: (low - 128) / 64 as two MX-fp8-exact 16-bit channels [hi | lo] (hi = k / 4, lo = (low - 128 - 16 k) / 16,
   k = round((low - 128) / 16); hi + lo / 4 = (low - 128) / 64) when split, else one channel. h16: fp16 (else bf16) */
void nn_restore_input(const uint8_t *low, void *x16, size_t n, int split, int h16);
/* loss over voxels with clean > 0: Charbonnier(e) + mse_w e^2 / 2 + edge_w Charbonnier(e_i - e_j) over face neighbours (mean over pairs),
   e = logit + (low - clean) / 64. st (device, 8 doubles): count, Charbonnier sum, |error| DN sum, error^2 DN sum, baseline error^2 DN
   sum, edge pairs, edge Charbonnier sum. gl (nullptr: stats only): d loss / d logit, 16-bit scaled by the gradient scale when g16
   (fp16 when h16, else bf16), else fp32. lg16: the logits are fp16 (else fp32). Not in place (the edge term reads neighbours) */
void nn_restore_loss(const void *lg, int lg16, const uint8_t *low, const uint8_t *clean, int D, int H, int W, float eps, float mse_w, float edge_w,
                     void *gl, int g16, int h16, float gscale, double *st);
/* output CT: clean-free, low + 64 logit rounded, clamped to [1, 255], 0 where low is 0 */
void nn_restore_output(const void *lg, int lg16, const uint8_t *low, uint8_t *out, size_t n);

#ifdef __cplusplus
}
#endif
