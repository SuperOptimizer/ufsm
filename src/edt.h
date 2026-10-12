/* Exact Euclidean distance transform of a 3-D binary mask (Felzenszwalb & Huttenlocher's separable lower-envelope algorithm):
   float32 squared distances to the nearest feature voxel, one pass per axis, each pass over independent lines on threads.
   4 bytes per voxel plus per-thread line buffers (scipy.ndimage's transform needs ~20 and one core). */
#pragma once
#include <stddef.h>
#include <stdint.h>

/* d (n[0] * n[1] * n[2], z-major) = squared distance from each voxel to the nearest voxel with feat != 0 (in voxels; voxels
   are features at distance 0); EDT_INF where the mask has no feature. Returns 0, -1 on allocation failure. */
#define EDT_INF 1e30f
int edt_sq(const uint8_t *feat, const int64_t n[3], float *d, int nthreads);
