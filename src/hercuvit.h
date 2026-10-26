/* hercuvit: HercuViT v0 (HF jimmylomro/hercuvit-v0, MIT; code github jimmylomro/hercUNet branch hercuvit, hercunet/hercuvit) in
   C/CUDA, for ufsm hercuvit (src/hercuvit.c). Weights straight from checkpoint_latest.pth (torch zip, entry "model", read by
   src/wload.c; no conversion step).

   Model (recipe full_decoupled, variant "full"): the fine-tuned Dinovol encoder dinovol_ps8_9.6um (src/dinovol_net.c: CNN stem
   16 / 48 / 128 / 192 at strides 1, 2, 2, 2 + 288-wide ViT, 10 all-global blocks of 6 heads of 48, 4 registers, mixed RoPE) ->
   DecoupledPixelEncoder (per stage ConvTranspose3d k2 s2 -> channel LayerNorm (eps 1e-6) -> concat the stem skip -> 2 BasicBlockD
   (conv3 + InstanceNorm + LeakyReLU 0.01, conv3 + InstanceNorm, 1x1 conv + InstanceNorm skip when the width changes, add,
   LeakyReLU); widths 128 @ 1/4, 48 @ 1/2, 32 @ 1/1; LayerNorm -> U1 = 32-d mask features) -> 1x1 surface head (2 classes; the
   margin l1 - l0 is the surface logit) -> DecoupledMaskDecoder (64 learned queries of width 32, 6 Mask2Former layers of 4 heads of 8
   with QK-LayerNorm, masked cross-attention round-robin over trilinear U1 pools at 1/32, 1/16, 1/8 (+ 3D sine PE + level
   embedding), self-attention, GELU MLP 128; mask embedding = LN(MLP3(LN(q))); mask logits = embedding . U1 per voxel).
   Inference (hercunet/hercuvit/infer/model.py forward_meshes): owner = argmax over the 64 queries, membership = softmax max,
   surface gate p > 0.4. The query side of the decoder (64 x 32) runs on the host in fp32; the dense dot products on the GPU. */
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* kernels (src/nn_hercuvit.cu), on xnet_stream(); U: fp16 [S^3][32] (z y x, channels last) */
/* out[q][cell] = max over the cell's (S / side)^3 voxels of me[q] . U[v]  (out pre-filled with -inf by the call) */
void hv_pool_max(const void *U, int S, const float *me_dev, int Q, int side, float *out_dev);
/* per voxel v of the window region [r0, r1) (window coords, 3 axes): q = argmax me . U[v], memb = exp(max - logsumexp);
   destination index (v - off) in a [dd][dh][dw] array: ids (uint32) = base + q + 1 and score = round(255 memb) where margin[v] > lthr
   (ids / score nullptr: skipped); owner (uint8 q) / memb_f (float) for every voxel (nullptr: skipped) */
void hv_final(const void *U, int S, const float *me_dev, int Q, const float *margin_dev, float lthr, const int r0[3], const int r1[3],
              const int off[3], const int dd[3], uint32_t *ids, uint32_t base, uint8_t *score, uint8_t *owner, float *memb_f);
/* Gaussian blend (villa generate_gaussian_map: sigma P / 8, peak at P / 2): acc += margin g, wsum += g over the window placed at
   org (tile coords, may be negative) inside a [dd][dh][dw] tile */
void hv_blend(float *acc, float *wsum, const int dd[3], const float *margin_dev, int P, const int org[3]);
/* u8 = round(255 sigmoid(clamp(acc / wsum, -30, 30))), 0 where wsum == 0 */
void hv_finish(const float *acc, const float *wsum, size_t n, uint8_t *u8);
void hv_fill(float *p, size_t n, float v);

/* the network (src/hercuvit_net.c) */
typedef struct hv_net hv_net;
hv_net *hv_load(const char *pth);           /* nullptr on failure (message printed); needs xnet_init first */
void hv_free(hv_net *n);
int hv_queries(const hv_net *n);            /* 64 */
/* one window: u8 [T]^3 on the device, y = (clamp(x, lo, hi) - center) / scale. Leaves the surface margin (fp32 [T^3] on the device:
   hv_margin) and U1; with inst also the final mask embeddings (hv_embed: fp32 [Q][32], device) for hv_final. 0 on success. */
int hv_forward(hv_net *n, const uint8_t *u8_dev, int T, float lo, float hi, float center, float scale, int inst);
const float *hv_margin(const hv_net *n);
const void *hv_u1(const hv_net *n);
const float *hv_embed(const hv_net *n);     /* device */
const float *hv_embed_host(const hv_net *n);
const void *hv_tokens(const hv_net *n);     /* fp16 [(T/8)^3][288] (parity) */
const void *hv_skip(const hv_net *n, int k, int *edge, int *ch);

/* villa normalize_robust (percentile 1 / 99 clip, median, 1.4826 MAD, fallback ladder) of a uint8 window, exact from its histogram */
void hv_robust_stats(const uint8_t *u, size_t n, double *lo, double *hi, double *center, double *scale);

int cmd_hercuvit(int argc, char **argv);    /* src/hercuvit.c */

#ifdef __cplusplus
}
#endif
