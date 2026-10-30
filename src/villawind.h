/* villawind: the ScrollPrize villa winding model (HF scrollprize/winding_model_9um, ckpt_final.pth = runs/winding_model_3d_large_11
   step 50000; code ScrollPrize/villa vesuvius/src/vesuvius/neural_tracing/winding_models/winding_model.py @37d1897) in C/CUDA, for
   ufsm villawind (src/villawind.c). Weights straight from the torch zip (entry "model", as infer_winding_volume.py loads them).

   Input: a ray-aligned slab [T][T][L] (transverse a, transverse b, ray; T 128, L 384, unit spacing at 9.6 um) of uint8 CT plus a
   validity mask; per-slab standardisation over the valid voxels (population std, + 1e-6), invalid voxels zeroed; channels (image,
   valid). Encoder: 3 stages (32, 64, 96) of conv3 stride (2, 2, 1) + channel LayerNorm + GELU + conv3 + LN + GELU (only the
   transverse axes halve); trunk 1x1 -> 192 + LN + GELU; 2 residual conv3 blocks; 6 pre-LN axial attention blocks (6 heads of 32):
   ray attention over L per column (relative bias, distance clamp 128), transverse attention over the (T / 8)^2 plane per ray position
   (2-D relative bias), GELU MLP 768. Decoder: trilinear x2 (transverse) + stage-2 skip -> 2 x (conv3 96 + LN + GELU); full-resolution
   head: trilinear to the stage-1 grid + stage-1 skip -> 1x1 32 + LN + GELU -> 1x1 4 -> pixel shuffle 2 -> per voxel increment
   softplus(logit) -> phase = cumsum along the ray (turns; monotone; one free offset per slab). */
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* kernels (src/nn_villawind.cu), on xnet_stream() */
void vw_input(const uint8_t *im, const uint8_t *va, float mean, float inv, void *y, long long n);
void vw_ln(const void *x, int x32, const float *w, const float *b, float eps, int gelu, const void *res, void *y, int y32, long long M, int C);
void vw_add_half(float *x, const void *u, long long n);
void vw_h2f(float *x, const void *u, long long n);
void vw_f2h(void *x, const float *u, long long n);
void vw_softmax(const float *S, void *P, long long rows, int L, int mode, int md, int side, const float *bias, float scale);
int vw_gemm(int ta, int tb, int m, int n, int k, const void *A, int lda, long long sa, const void *B, int ldb, long long sb,
            void *D, int ldd, long long sd, int d32, int batch);
void vw_phase(const float *lg, int D, int H, int W, float *ph, float *inc);

typedef struct vw_net vw_net;
vw_net *vw_load(const char *pth);   /* nullptr on failure (message printed); needs xnet_init first */
void vw_free(vw_net *n);
/* one slab: im / va uint8 [T][T][L] on the device (T % 8 == 0, T <= 128), mean / inv = the standardisation (host). phase_dev:
   fp32 [T][T][L] (device). dump (nullptr: none): directory for the parity intermediates (fp32 raw files). 0 on success. */
int vw_forward(vw_net *n, const uint8_t *im, const uint8_t *va, int T, int L, float mean, float inv, float *phase_dev, const char *dump);

int cmd_villawind(int argc, char **argv);   /* src/villawind.c */

#ifdef __cplusplus
}
#endif
