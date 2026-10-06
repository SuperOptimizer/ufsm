/* Band field (whole-sheet labels) from a winding_mod14 raster: every voxel gets the band of the wrap it lies in, as the
   winding (in 1/18-turn steps mod 252) of the nearest labelled recto surface plus or minus half a turn (outside / inside
   that surface relative to the umbilicus). Between consecutive wraps both bounding rectos give the same value, so the band
   is constant inside a wrap and jumps by one turn exactly at each recto. 255 = unknown. Shared by the sampler (task
   band_affinity) and `ufsm band` (boxes for evaluation and previews). See tools/band_field.py for the python version. */
#pragma once
#include <stdint.h>
#include <stddef.h>

enum { BAND_STEPS = 18, BAND_PERIOD = 252, BAND_HALF = 9, BAND_UNKNOWN = 255 };

typedef struct {
    float radius;      /* flood radius in native voxels (beyond: unknown); about 3/4 of the turn pitch */
    float span;        /* native voxels around a missing-wrap jump that become unknown */
} band_params;

/* codes: winding_mod14 raster on the label grid (one label voxel = 2 native voxels), dims n[3] (z, y, x), origin o[3] in
   label voxels; cy, cx: umbilicus centre (native voxels) at each label z plane. out: band per voxel (0..251 or 255).
   Returns 0, or -1 when out of memory. Neighbourhoods are limited to the array: callers pass a halo of at least
   (radius + span) / 2 + 2 label voxels and crop. */
int band_field(const uint8_t *codes, const int n[3], const int64_t o[3], const double *cy, const double *cx, band_params p, uint8_t *out);

/* modular difference of band / winding steps in (-126, 126] */
static inline int band_diff(int a, int b) { int d = ((a - b + 126) % 252 + 252) % 252; return d - 126; }
