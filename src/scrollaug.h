/* Scroll-damage augmentations of stored-phase windows (sample.c --aug-tear / --aug-delam / ...): the geometry and the per-voxel
   kernels, free of the sampler so the tests can drive them on synthetic boxes. Phase codes as band.h (0 .. PHASE_PERIOD - 1, 255
   unknown); boxes zyx, row-major. */
#pragma once
#include <stdint.h>
#include <stddef.h>

/* the dominant sheet normal and the sheet spacing of a phase box (D x H x W): the structure tensor of the circular phase gradient
   (central differences, every stride-th voxel of the central box inset by inset voxels). n zyx unit; *spacing voxels per turn (the
   median gradient's inverse). Returns the number of gradient samples (outputs unchanged under 32). */
int sa_phase_normal(const uint8_t *ph, int D, int H, int W, int inset, int stride, double n[3], double *spacing);

/* --aug-tear: a smooth cut surface h(p) = c . (p - p0) + sum_j a_j sin(k_j . p + f_j) = 0 roughly across the sheets (c within
   max_tilt of perpendicular to the sheet normal), the side h > 0 shifted by dp, the other by dm (integer voxels, dp - dm = the
   displacement: up to ~1+ turns along the normal and optionally along the sheet), and the seam: 0 none, 1 a thin void (CT the
   window's void level, labels unknown), 2 a blended band (the two sides' CT mixed, labels unknown). */
typedef struct {
    double c[3], p0[3], k[3][3], a[3], f[3];
    int dp[3], dm[3];
    int seam; double width;
    double turns_n, turns_t;   /* (diagnostics: the drawn shift along the normal / the sheet, turns) */
} sa_tear;
/* P the window; n the sheet normal, spacing voxels per turn; maxd the largest displacement (voxels, |dp - dm|); seed */
void sa_tear_make(sa_tear *t, uint64_t seed, int P, const double n[3], double spacing, double maxd);
double sa_tear_h(const sa_tear *t, double z, double y, double x);
/* the margin a padded read needs around the window for this tear (max over axes of |dp|, |dm|) */
int sa_tear_margin(const sa_tear *t);
/* the window (P^3) from padded boxes (P + 2 M)^3 whose corner is the window corner - M: ct and ph (codes) always, w4 / wc (nullable:
   the winding mod 4 and its component, sample.c band_wind4_tiled) pulled with the same integer shifts; void_ct the seam's void CT */
void sa_tear_apply(const sa_tear *t, int P, int M, const uint8_t *ct_pad, const uint8_t *ph_pad, const uint8_t *w4_pad, const uint8_t *wc_pad,
                   uint8_t void_ct, uint8_t *ct, uint8_t *ph, uint8_t *w4, uint8_t *wc);
/* the CT value at the q-quantile of a box's nonzero voxels (0 when none) */
uint8_t sa_ct_quantile(const uint8_t *ct, size_t n, double q);
