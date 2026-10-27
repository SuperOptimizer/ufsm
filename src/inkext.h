/* ufsm ink-ext: external ink-detection networks (v8in, Hecate 2.4 / 9.6 um, Reader V3, scrollprize ink_9um) ported to xnet (xnet.h), the
   surface-volume render they read (from a tifxyz and the 2.4 um CT pyramid), their tiling / blending, and the ensemble.

   ie_net: one loaded network. ie_forward runs B tiles of uint8 surface volume (B x D x T x T, layer 0 = the first layer the
   model reads, i.e. after any --reverse) through the model's own input normalisation and returns per tile T x T values:
   the model's final logits (ie_forward(.., logits = 1); Hecate: logit of its upsampled probability) or probabilities. */
#pragma once
#include <stddef.h>
#include <stdint.h>

enum { IE_V8IN = 0, IE_HEC24 = 1, IE_HEC96 = 2, IE_READER3 = 3, IE_INK9UM = 4, IE_NMODELS = 5 };
/* the koine_machines hybrid 3D-stem / 2D U-Net family (Reader V3 384 px, scrollprize/ink_9um 128 px): one network, koine infer.py
   tiling (overlap 0.5, hann), normalize_robust per tile */
static inline int ie_is_koine(int kind) { return kind == IE_READER3 || kind == IE_INK9UM; }

typedef struct {
    const char *name;     /* v8in | hecate24 | hecate96 | reader3 | ink9um */
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
