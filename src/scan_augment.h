/* Scanner-domain augmentation of the uint8 CT cube on its source grid (z = scan axis, the axis centre known per z),
   after the targets are built and before the cube symmetry / continuous warp, so every op is then oriented randomly
   like the data. Each op has its own probability; strength scales every magnitude (0 = identity, 1 = default, 2 =
   aggressive). All probabilities 0 (the default) draws nothing from the sampler stream. Labels are untouched except:
   seam shifts CT, labels, ignore and the axis together; dropped slices / cutouts are ignored when drop_ignore; voxels
   the op zeroes (chunk dropouts, simulated air masking) lose supervision through the CT > 0 mask. */
#pragma once
#include <stddef.h>
#include <stdint.h>

enum { SA_SEAM, SA_AIR, SA_LOWRES, SA_BLUR, SA_SHARPEN, SA_BIAS, SA_RING, SA_CUPPING, SA_SLICE, SA_NOISE, SA_CNOISE,
       SA_TONE, SA_QUANT, SA_DROPSLICE, SA_CUTOUT, SA_CHUNK, SA_N };
extern const char *const scan_aug_names[SA_N];

typedef struct {
    float strength;   /* magnitude scale, 0..2 */
    float p[SA_N];    /* per-op probability */
    int drop_ignore;  /* 1: dropped slices and cutouts are unsupervised (else the label stays and the net must infer it) */
} scan_aug_cfg;

static inline int scan_aug_enabled(const scan_aug_cfg *c) {
    if (!(c->strength > 0)) return 0;
    for (int i = 0; i < SA_N; i++) if (c->p[i] > 0) return 1;
    return 0;
}

/* In place on ctu (P^3), tgt (nch x P^3 labels, only seam touches them), ign (P^3, 1 = ignore). cy/cx: axis centre per
   source z in level voxels (absolute; patch voxel y is o[1] + y), shifted by seam; hasax 0 puts ring centres at a random
   point outside the patch. chunk: zarr inner chunk edge of this level (chunk dropouts align to it). allow_seam 0 for
   tasks whose labels are not dense rasters. scratch: 5 P^3 bytes. Seed fixes every decision. Returns the bitmask of
   the ops applied. */
unsigned scan_aug_apply(const scan_aug_cfg *c, uint64_t seed, uint8_t *ctu, uint8_t *tgt, int nch, uint8_t *ign, int P,
                        const int64_t o[3], float *cy, float *cx, int hasax, const int chunk[3], int allow_seam, void *scratch);
