/* wtrunk: the frozen wind96 U-Net run in-slab for the ray model (vwt). The network sees each slab as a window with
   (z, y, x) = (j, i, k) (vws_sample order is [i][j][k]), its CT z-scored over the slab (predict's per-window statistics),
   the radial input channels from the umbilicus (horizontal unit vector from the axis, its a and d components = the
   network's y and x channels; the b component, <= the slab tilt, has no channel). Inference only (stage 1): no stored
   activations, the checkpoint's own MX / fp4 mode, EMA weights.

   OUTPUT (device, bf16, channels-last [n][T][T][L][C] in vws_sample voxel order (i, j, k), k fastest):
     channels 0 .. cout - 1: sigmoid of each head logit = the trained target codes (task unwrap: 0 recto, 1 ink,
       2-3 0.5 + 0.5 cos / sin (2 pi phase), 4-9 n_z^2 n_y^2 n_x^2 0.5 + n_z n_y, 0.5 + n_z n_x, 0.5 + n_y n_x with
       the network's (z, y, x) = slab (j, i, k), 10 density code x / (1 + x));
     channels cout .. cout + widths[0] - 1 (feat 1): dec[0]'s output silu(gn(a2)) as the head reads it (fp4 storage).
   Device buffers are on the current GPU (nn_init first); default stream; calls are synchronous with respect to it. */
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct wtrunk wtrunk;
typedef struct {
    double o[3], a[3], b[3], d[3];   /* slab frame, xyz scan voxels of the CT level (vws_sample) */
    double c[3], u[3];               /* a point on the umbilicus (xyz, e.g. at the slab centre's z) and the axis direction (unit, xyz) */
} wt_frame;

/* ckpt: an embedded-runtime task-unwrap checkpoint (wind96); T, L: slab size (T, L divisible by 2^(nlev-1)); maxbatch: slabs per
   network forward (larger calls are split); feat: 1 = heads + level-0 features, 0 = heads only. nullptr on error (message printed) */
wtrunk *wt_open(const char *ckpt, int T, int L, int maxbatch, int feat);
int wt_channels(const wtrunk *w);       /* C of the output */
int wt_heads(const wtrunk *w);          /* cout: the head channels at the front */
/* im: device uint8 [n][T][T][L] (0 outside the volume, as vws_sample); fr: host n frames; um: the CT level's voxel size (the
   network's scale input); out: device bf16 [n][T][T][L][C]. 0 on success */
int wt_forward(wtrunk *w, const uint8_t *im, const wt_frame *fr, int n, double um, void *out);
/* the same into rows of ld >= C channels (e.g. vwt_xfeat_buf: the features inside the ray model's padded input); channels C .. ld - 1 untouched */
int wt_forward_ld(wtrunk *w, const uint8_t *im, const wt_frame *fr, int n, double um, void *out, int ld);
void wt_close(wtrunk *w);

/* training (stage 2): ckpt = the wind96 checkpoint (fine-tuning start: its EMA weights live, a fresh optimiser) or a wt_save file (resumed
   with its optimiser state, EMA and step). The production training precision (fp16 + gradient scale, MX-fp4 activations / gradients, SR).
   levels > 0: only the head and the last `levels` decoder blocks train. wt_forward_ld on this handle runs a no-grad inference build with
   the trained live weights (copied after each step, on first use); wt_forward_train keeps the activations for wt_backward (n 1);
   wt_backward(g: bf16 d loss / d outputs, rows of ld in output order) accumulates the trunk's gradients (through the sigmoid heads and the
   features); wt_step: clip, Muon (3^3 convs) + AdamW, EMA (1: skipped, non-finite gradient); wt_save: a unet checkpoint wt_open reads */
wtrunk *wt_open_train(const char *ckpt, int T, int L, int levels);
int wt_forward_train(wtrunk *w, const uint8_t *im, const wt_frame *fr, double um, void *out, int ld);
int wt_backward(wtrunk *w, const void *g, int ld);
void wt_zero_grad(wtrunk *w);
int wt_step(wtrunk *w, float lr_muon, float lr_adam, float wd, float clip, float ema);
int wt_step_of(const wtrunk *w);
int wt_save(wtrunk *w, const char *path);
struct unet *wt_unet(wtrunk *w);   /* the trained network (tests) */
const void *wt_y16(wtrunk *w);     /* the last training forward's fp16 head logits + features [C][T][T][L] (network order; tests) */

/* kernels (src/nn_wtrunk.cu). prm: device floats [n][16] = (o - c)[3], a[3], b[3], d[3], u[3], pad; st: device u64 [n][2] scratch;
   x: [n][4][T][T][L] network input, fp16 (h16 1) or bf16 (h16 2); lg: [n][C][T][T][L] fp16 (lh16) or fp32, (z, y, x) = (j, i, k) */
void wt_input(const uint8_t *im, const float *prm, int n, int T, int L, unsigned long long *st, void *x, int h16);
void wt_output(const void *lg, int lh16, int n, int C, int cout, int T, int L, int ld, void *out);   /* rows of ld >= C channels */
#ifdef __cplusplus
}
#endif
