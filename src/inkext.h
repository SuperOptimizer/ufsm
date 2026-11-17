/* ufsm ink-ext: external ink-detection networks (v8in, Hecate 2.4 / 9.6 um, Reader V3, scrollprize ink_9um) ported to xnet (xnet.h), the
   surface-volume render they read (from a tifxyz and the 2.4 um CT pyramid), their tiling / blending, and the ensemble.

   ie_net: one loaded network. ie_forward runs B tiles of uint8 surface volume (B x D x T x T, layer 0 = the first layer the
   model reads, i.e. after any --reverse) through the model's own input normalisation and returns per tile T x T values:
   the model's final logits (ie_forward(.., logits = 1); Hecate: logit of its upsampled probability) or probabilities. */
#pragma once
#include <stddef.h>
#include <stdint.h>

enum { IE_V8IN = 0, IE_HEC24 = 1, IE_HEC96 = 2, IE_READER3 = 3, IE_INK9UM = 4,
       /* the ink-sdk profiles' further ink_9um-family checkpoints (2026-11-16): same network (vesuvius_unet_3d_stem_2d, 508 tensors), own tile / spacing */
       IE_READER2 = 5, IE_DENSE9 = 6, IE_RECTINA = 7, IE_TRACE3B = 8,
       IE_CANON2 = 9,   /* scrollprize/ink_canonical_2um: Hecate 2.4's ResNet3D-152 + 3-D decoder, a plain depth softmax, 62 layers at 2.399 um */
       IE_STUDIO = 10,  /* Studio softloss-20261023: DeepResidualUNet (GroupNorm + SiLU pre-activation 3-D U-Net), a learned depth softmax, 16 layers */
       IE_TSF = 11,     /* the scrollprize ink TimeSformers (lucidrains 0.4.1; depth / heads from the checkpoint): 26 frames of 64 px at 7.91 um */
       IE_NMODELS = 12 };
/* the koine_machines hybrid 3D-stem / 2D U-Net family (Reader V3 / V2, rectina m4 384 px, scrollprize/ink_9um, dense-native, TRACE v3B
   128 px): one network, koine infer.py tiling (overlap 0.5, hann), normalize_robust (robust_mad, p1 / p99) per tile */
static inline int ie_is_koine(int kind) { return kind == IE_READER3 || kind == IE_INK9UM || (kind >= IE_READER2 && kind <= IE_TRACE3B); }

typedef struct {
    const char *name;     /* v8in | hecate24 | hecate96 | reader3 | ink9um | reader2 | dense9 | rectina | trace3b | canon2 | studio | tsf */
    int D, T;             /* input layers, tile edge (pixels) */
    double px_um;         /* render spacing (pixel and layer step) */
    int stride;           /* default tile stride */
    int center;           /* index of the surface layer within the D layers (offsets i - center) */
} ie_spec;
extern const ie_spec ie_specs[IE_NMODELS];
int ie_model_kind(const char *name);   /* -1 if unknown */

typedef struct ie_net ie_net;
ie_net *ie_load(int kind, const char *weights);   /* nullptr on failure (message printed) */
void ie_free(ie_net *n);
/* tiles: host uint8 [B][D][T][T]; out: host float [B][T][T]. 0 on success. */
int ie_forward(ie_net *n, const uint8_t *tiles, int B, float *out, int logits);
size_t ie_tile_bytes_gpu(int kind);   /* rough peak device memory of one tile in a batch */

/* per-tile input normalisation parameters (host): y = clamp(x, lo, hi) * a + b */
void ie_robust_params(const uint8_t *x, size_t n, float *lo, float *hi, float *a, float *b);   /* Reader: normalize_robust (p1, p99, median / MAD) */
/* Reader / ink9um: use these parameters for every tile instead of the per-tile normalize_robust (on = 0: back to per tile) */
void ie_set_norm(ie_net *n, int on, float lo, float hi, float a, float b);
