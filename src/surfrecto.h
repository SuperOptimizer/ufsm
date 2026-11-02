/* surfrecto: the published recto-surface 3D U-Net (HF scrollprize/surface_recto_3dunet, MIT; model_name ps256_bs2_msr_default,
   Angelotti et al. arXiv:2606.29085) in C/CUDA on xnet, for ufsm surfrecto-predict (src/surfrecto.c). Weights straight from
   checkpoint_inference_ready.pth (torch zip, entry "model", read by src/wload.c; no conversion step).

   Model (villa vesuvius NetworkFromConfig, @fd97a4b): 1-channel input, per-window z-score (villa VCDataset 'instance_zscore':
   (x - mean) / max(std, 1e-8) over the whole 256^3 window, population std; the checkpoint's normalization_scheme is "zscore" with
   empty intensity_properties). Every conv has a bias, InstanceNorm3d(affine, eps 1e-5), LeakyReLU 0.01.
   - stem: conv3 1 -> 32 + IN + LReLU;
   - encoder: 7 stages of BasicBlockD (ResNet-D), widths 32 64 128 256 320 320 320, blocks 1 3 4 6 6 6 6, the first block of stages
     1..6 strided 2 (conv3 s2; skip AvgPool3d(2) + 1x1 conv (no bias) + IN); block: h = IN(conv2(LReLU(IN(conv1 x)))),
     scSE: h = h sigmoid(fc2(ReLU(fc1(mean h)))) + h sigmoid(conv1x1(h)) (cSE reduction 8 / 8 / 8 / 16 / 24 / 24 / 24), out =
     LReLU(h + skip). mean h over the window = the conv2 IN beta exactly (an affine InstanceNorm output has spatial mean beta), so the
     channel gate is a per-block constant computed at load time.
   - decoder (task_decoders.surface, separate decoder, no deep supervision): 6 stages of ConvTranspose3d(k2 s2, bias) -> cat(up, skip)
     -> conv3 + IN + LReLU; seg_layers.5 (1x1, 32 -> 2) is the only head used at inference. 2-class softmax: p(surface) =
     sigmoid(l1 - l0); the network returns the margin l1 - l0 (fp32).
   Inference (villa vesuvius.models.run.inference + blending + finalize_outputs): 256^3 patches, overlap 0.5, optional mirror TTA (8
   flips, mean of the logits), windows whose CT is constant skipped, Gaussian blend of the logits (sigma P / 8, peak 1 at P / 2),
   softmax, uint8 = rint(255 p); voxels no window covers = background (0). */
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* kernels (src/nn_surfrecto.cu), on xnet_stream(); h / res fp16 [nvox][C] (channels last) */
/* h = LReLU(h (cg[c] + sigmoid(h . sw + sb)) + res), in place (scSE + residual + activation of a BasicBlockD) */
void sr_scse(void *h, const void *res, size_t nvox, int C, const float *sw_dev, float sb, const float *cg_dev);
/* mirror TTA: out = u8 window [P]^3 flipped along the axes in mask (bit 0 x, 1 y, 2 z) */
void sr_flip_u8(const uint8_t *in, uint8_t *out, int P, int mask);
/* acc (+)= flip(m, mask) (first: acc = ...); fp32 [P]^3 */
void sr_flip_acc(float *acc, const float *m, int P, int mask, int first);
void sr_scale(float *p, size_t n, float s);

/* the network (src/surfrecto_net.c) */
typedef struct sr_net sr_net;
sr_net *sr_load(const char *pth);           /* nullptr on failure (message printed); needs xnet_init first */
void sr_free(sr_net *n);
/* one window: u8 [P]^3 on the device (P a multiple of 64), normalised (x - mean) / std. Returns the margin l1 - l0, fp32 [P^3] on the
   device (owned by the net, valid until the next call), nullptr on failure. */
const float *sr_forward(sr_net *n, const uint8_t *u8_dev, int P, float mean, float stdv);
/* --raw parity: the stage outputs of the last forward (fp16 NDHWC, kept when sr_keep(n, 1) was set before it) */
void sr_keep(sr_net *n, int on);
const void *sr_kept(const sr_net *n, int k, int *edge, int *ch);   /* k 0..6 encoder stages, 7..12 decoder stages */

/* exact z-score statistics of a uint8 window from its histogram: mean, population std (np.mean / np.std) */
void sr_zstats(const uint8_t *u, size_t n, double *mean, double *stdv, int *vmin, int *vmax);

int cmd_surfrecto_predict(int argc, char **argv);   /* src/surfrecto.c */

#ifdef __cplusplus
}
#endif
