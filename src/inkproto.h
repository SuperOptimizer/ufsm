/* ink-proto (2026-11-27, inkproto): conservative 3-D ink labels from a feature prototype (arXiv 2606.29085, stage 2, with our own features).
   The features are the ink net's own penultimate ones: the trunk's level-0 output as the head reads it (unet_head_feat_z, widths[0]
   channels per voxel). Steps:
   - seed:  predict CK ... --proto-seed OUT.proto [--proto-thr 0.95]: per window, the interior voxels whose own ink probability is
            >= thr add their feature vector (raw and L2-normalised) to the sums; OUT holds the sums and the count (ink_proto_file).
   - cos:   predict CK ... --proto P.proto: the output channel is replaced by (1 + cos(feature, prototype)) / 2 (uint8 255 p, CT-masked as
            any probability output); the prototype is the normalised unit-vector mean of P.
   - label: ufsm ink-proto label OUT --teacher T --cos C --ct ROOT KEY: per chunk min-max of the cosine, times the teacher's probability,
            CT below --air masked; a uint8 score or a prob254 label store (ink_proto_label_chunk). */
#pragma once
#include <stddef.h>
#include <stdint.h>

#define INK_PROTO_MAXC 256
typedef struct { int C; double n; double raw[INK_PROTO_MAXC], unit[INK_PROTO_MAXC]; } ink_proto_file;
int ink_proto_read(const char *path, ink_proto_file *P);          /* 0 ok */
int ink_proto_write(const char *path, const ink_proto_file *P);
/* the prototype: the unit sums' mean (mode 0) or the raw sums' mean (mode 1), L2-normalised into pr[C]; -1 if empty */
int ink_proto_vector(const ink_proto_file *P, int mode, float *pr);

/* one chunk of the label generator (CPU; tests/test_ink_proto.c): n voxels of teacher probability t (uint8, 0 = no output there),
   cosine store c (uint8 255 (1 + cos) / 2, 0 = no output) and CT; the chunk's min / max of c over the voxels with t, c > 0 and CT >= air,
   then s = t / 255 x (c - min) / (max - min) (max == min: s = 0, the chunk has no contrast). out_mode 0: the score, uint8
   round(254 s) + 1 (0 = masked: no t / c or CT < air); 1: a prob254 label, 254 where s > thr, 0 where not, 255 masked. Returns the
   count of voxels labelled ink (s > thr). */
long ink_proto_label_chunk(const uint8_t *t, const uint8_t *c, const uint8_t *ct, size_t n, int air, double thr, int out_mode, uint8_t *out);

/* the device kernels (src/nn_proto.cu). ft: fp16 planes [C][nv] of the window's level-0 slab (unet_head_feat_z); lg: the output
   channel's logits of the same slab (fp16 if h16, else fp32). */
void nn_proto_cos(const void *ft, int C, size_t nv, const float *pr_dev, void *lg, int h16);
/* acc (device, 2 C + 1 doubles: raw sums, unit sums, count) += the voxels of window planes [z0, z0 + nz) (D x H x W window) inside the
   interior (hz, hy, hx margins) with logit >= thr_logit */
void nn_proto_acc(const void *ft, int C, int z0, int nz, int D, int H, int W, int hz, int hy, int hx, const void *lg, int h16, float thr_logit, double *acc_dev);

/* predict's hooks (src/inkproto.c): ink_proto_pred_open after the net is built (C = widths[0]; seed: path to write, else nullptr;
   proto: the prototype file, else nullptr); ink_proto_pred_window after each window's forward with the output channel's logits;
   ink_proto_pred_close at the end (writes the seed file). */
struct unet;
typedef struct ink_proto_pred ink_proto_pred;
ink_proto_pred *ink_proto_pred_open(int C, const char *seed, double thr, const char *proto, int mode);
int ink_proto_pred_window(ink_proto_pred *Q, struct unet *u, void *lg, int h16, int D, int W, int hz, int halo);
int ink_proto_pred_close(ink_proto_pred *Q);
