/* Conservative appearance augmentation of reconstructed uint8 CT. This module never
   changes labels, radial channels, or the original CT>0 supervision mask. */
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct {
    float lut[256];           /* CT after optional gamma, normalized with the supplied mean/sd */
    int filter_mode;          /* 0 = none, 1 = weak low-pass, 2 = weak unsharp */
    float filter_strength;
    float shading[3];        /* source z/y/x linear field; sum(abs(coeff)) <= .08 */
    float noise_rho;         /* 0 = caller's existing white-noise path */
    float noise_sigma;       /* correlated-noise SD in normalized CT units */
    int gamma_applied;
} ct_aug_plan;

/* Same arguments/seed give exactly the same plan. enabled=0 gives the ordinary
   normalized CT LUT and zero extra operations. P must be positive. */
void ct_aug_make(ct_aug_plan *p, uint64_t seed, int enabled, double mean, double sd, int P);

/* Coordinates are valid source coordinates in [0,P). Clamped neighbors implement
   the 7-point star (.5 center + 1/12 each axial neighbor), not an exact Gaussian
   reconstruction kernel. Inlining keeps the no-filter/no-shading path to one LUT
   lookup. A constant image is preserved exactly by the difference formulation. */
static inline float ct_aug_value(const ct_aug_plan *p, const uint8_t *ctu,
                                 int P, int z, int y, int x) {
    const size_t P2 = (size_t)P * P, q = (size_t)z * P2 + (size_t)y * P + x;
    float v = p->lut[ctu[q]];
    if (p->filter_mode) {
        const size_t zm = z ? q - P2 : q, zp = z + 1 < P ? q + P2 : q;
        const size_t ym = y ? q - P : q, yp = y + 1 < P ? q + P : q;
        const size_t xm = x ? q - 1 : q, xp = x + 1 < P ? q + 1 : q;
        float delta = ((p->lut[ctu[zm]] - v) + (p->lut[ctu[zp]] - v)
                     + (p->lut[ctu[ym]] - v) + (p->lut[ctu[yp]] - v)
                     + (p->lut[ctu[xm]] - v) + (p->lut[ctu[xp]] - v)) * (1.f / 12.f);
        v += (p->filter_mode == 1 ? p->filter_strength : -p->filter_strength) * delta;
    }
    if (P > 1 && (p->shading[0] != 0 || p->shading[1] != 0 || p->shading[2] != 0)) {
        float a = 2.f / (float)(P - 1);
        v += p->shading[0] * (a * z - 1.f)
           + p->shading[1] * (a * y - 1.f)
           + p->shading[2] * (a * x - 1.f);
    }
    return v;
}

/* row initially contains independent N(0,1) draws from the caller. Apply unit-
   variance AR(1) along source X, then Y, then Z. Call once per row in increasing
   z/y order, before any spatial permutation. previous_row has P floats holding
   the Y stage; previous_plane has P*P floats holding the final Z stage. Initial
   contents do not matter: y=0/z=0 overwrite the states at every cube boundary.
   Both state arrays are required when rho>0; rho=0 is an exact no-op and permits
   NULL states. Multiply the result by noise_sigma at the call site. */
void ct_aug_noise_row(const ct_aug_plan *p, float *row, float *previous_row,
                      float *previous_plane, int P, int z, int y);
