/* ink3d-ext: published dense 3D ink networks run as teachers over a CT box, in C/CUDA (xnet).

   gambiarra = scrollprize/ink_paris4_3d_4.8um_gambiarra (HF, MIT; villa branch ink-paris4-3d-gambiarra,
   vesuvius/ink_detection/paris4_3d_gambiarra): vesuvius NetworkFromConfig "unet", 24.5M parameters, weights straight from
   model.safetensors (src/wload.c). Encoder: 5 stages of BasicBlockD (ResEnc-D), widths 32 64 128 256 320, 1 2 2 2 2 blocks, the
   first block of stages 1-4 strided 2 (conv3 s2; skip AvgPool3d(2) + 1x1 conv (no bias) + InstanceNorm); every conv3 has a bias
   and an affine InstanceNorm (eps 1e-5), LeakyReLU 0.01. Decoder: ConvTranspose3d(k 2, s 2) -> cat(up, skip) -> conv3 + IN + LReLU
   per stage; head task_heads.ink = 1x1 conv -> one logit. No stem, no deep supervision.

   Input (volume.py): a 128^3 level-0 (2.4 um) crop at [2a, 2a + 128) per axis (a = output-grid origin), normalised with p1 / p99
   (np.percentile linear) of its 256^3 parent [2a - 64, 2a + 192): clip, subtract, divide (float64 steps rounded to float32), then
   avg_pool3d(2) -> 64^3 at 4.8 um; output = 64^3 logits of output voxels [a, a + 64) (output j covers level-0 [2j, 2j + 2)).

   dino3d = scrollprize/ink_3d_dino_guided (HF, MIT; ckpt_78k_fullsup.pth, ema_model): the same NetworkFromConfig family with a stem
   (conv3 + IN + LReLU, 1 -> 32), 7 stages, widths 32 64 128 256 320 320 320, 1 3 4 6 6 6 6 blocks, 6 decoder stages; trained on
   256^3 level-0 (2.4 um) patches with percentile_minmax (p1 / p99 of the patch itself: vesuvius A.Normalize('percentile', 1, 99)).
   Here: a 256^3 level-0 crop at [2a, 2a + 256) (its own parent), the network at full 2.4 um, its 256^3 logits avg-pooled 2 -> 128^3
   on the 4.8 um output grid [a, a + 128); Gaussian blend sigma 16 (= 128 / 8). */
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* crop geometry of a model (gb_load sets it): crop = the level-0 crop edge, pool_in = its pooling to the network input (net_in =
   crop / pool_in), pool_out = the pooling of the network's logits to the 4.8 um output (out = net_in / pool_out), parent = the
   normalisation window edge, starting poff level-0 voxels before the crop (crop at [2a, 2a + crop), parent at [2a - poff, ..)) */
typedef struct { int crop, pool_in, net_in, pool_out, out, parent, poff; } i3_geom;

/* kernels (src/nn_ink3dext.cu), on xnet_stream() */
/* u8 crops [B][C^3] + per-crop value tables lut [B][256] -> fp16 [B][(C / pool)^3] (one channel): mean of lut over each pool^3 block */
void i3_prep(const uint8_t *u8, const float *lut, int B, int C, int pool, void *out_half);
/* logits [B][N^3] -> [B][(N / 2)^3]: mean of each 2^3 block */
void i3_pool2(const float *in, int B, int N, float *out);
/* Gaussian blend (sigma O / 8, peak 1 at O / 2) of B crops' logits [B][O^3] into ring accumulators acc / wsum [R][TY][TX]:
   org [B][3] = (global z, y - band y0, x - band x0) of each crop's first output voxel; rows outside [z0, z1) are skipped; ring row = z % R */
void i3_blend(float *acc, float *wsum, int R, int TY, int TX, const float *logits, const int *org_dev, int B, int z0, int z1, int O);
/* rows [z0, z1) (global, z0 even) of the ring: out48 [z1 - z0][TY][TX] prob254 (round(254 sigmoid(acc / wsum)), 255 = no weight),
   out96 [(z1 - z0 + 1) / 2][(TY + 1) / 2][(TX + 1) / 2] = round(254 mean of the known 4.8 sigmoids) (255: none known), hist
   [I3_HBINS] counts of the blended logits (bins of 0.5 over [-16, 16), first / last = under / over); then zeroes the rows.
   Any output may be nullptr. */
enum { I3_HBINS = 66 };
void i3_emit(float *acc, float *wsum, int R, int TY, int TX, int z0, int z1, uint8_t *out48, uint8_t *out96, unsigned long long *hist);

/* the network (src/ink3dext_net.c) */
typedef struct gb_net gb_net;
gb_net *gb_load(const char *weights, const char *model, i3_geom *g);   /* model gambiarra | dino3d; nullptr on failure (message printed); needs xnet_init first */
void gb_free(gb_net *n);
/* B crops (u8 [B][crop^3] and lut [B][256] on the device) -> logits fp32 [B][out^3] on the device (owned by the net, valid until the
   next call). 0 on success. */
int gb_forward(gb_net *n, const uint8_t *u8_dev, const float *lut_dev, int B, const float **logits_dev);
size_t gb_crop_bytes_gpu(const gb_net *n);  /* rough peak device memory per crop of a batch */

/* p1 / p99 of a uint8 histogram of n values exactly as np.percentile(linear) of the float32 values, and the per-value table of
   volume.normalize_reference (clip / subtract / divide in float64 rounded to float32 per step; 0 if hi - lo < 1e-6) */
void i3_percentiles(const uint64_t hist[256], uint64_t n, double *lo, double *hi);
void i3_lut(double lo, double hi, float lut[256]);

int cmd_ink3d_ext(int argc, char **argv);    /* src/ink3dext.c */
int cmd_vol_cmp(int argc, char **argv);      /* src/ink3dext.c: two uint8 volumes over a box (correlations, value mapping) */

#ifdef __cplusplus
}
#endif
