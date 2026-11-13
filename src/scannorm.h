#pragma once
/* Per-scan CT input normalisation (train --scan-norm 1, and predict for nets trained with it, unet_cfg.scan_norm): one (center, scale)
   per scan and level instead of each window's own mean / sd, so a window's input does not depend on its extent or position (the
   2026-11-13 seam measurements: the per-window z-score was one of three seam causes). center = the mean, scale = the sd of the
   nonzero voxels in K = 48 boxes of 32^3 at the working level L, the boxes centred on nonzero voxels of a coarse level
   (the first of L + 2 .. L + 9 with at most 2^26 voxels, read whole; else the smallest, every st-th voxel per axis) picked at fixed ranks: a deterministic function of the volume,
   so the sampler and predict compute the same numbers for the same scan and level. The input is (v - center) / scale. */
#include "zarr3.h"

/* open(ctx, level): the scan's CT level (nullptr: none); the handles are not closed here. 0 ok; < 0: -2 no level L, -3 no coarse
   level, -4 / -6 a read failure, -5 / -7 too little material */
int scan_norm_stats(z3 *(*open)(void *ctx, int level), void *ctx, int L, float *center, float *scale, int nthreads);
