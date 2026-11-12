/* The winding phase computed from the codes on read (2026-11-11): a z3 array (z3_open_derived) whose z3_read(o, n) runs the pair rule
   (band_field_phase_pair, src/band.h; on the GPU through src/bandgpu.h when configured) over the box plus a halo of the pair radius,
   tile by tile, instead of reading a stored phase-raster output. A 4.8 um phase set is ~10x its codes (PHerc1667 A: 46 GB vs 4.5 GB,
   volcomp q0: the phase ramp varies everywhere) and is a deterministic function of them, so a source can carry only the codes.
   Byte-identical to `ufsm phase-raster --pair 1` on the same codes grid (--gs 2, or --gs 4 --codes-grid 1, or the 2x-pooled codes
   written by `ufsm codes-pool2` for the --gs 4 sets): the phase at a voxel depends only on the labelled voxels within rmax of it, and
   the halo (ceil(rmax) + 2, as phase-raster's) covers them. Voxels outside the codes' shape read PHASE_UNKNOWN (the stored fill). */
#pragma once
#include "zarr3.h"
#include <stddef.h>
#include <stdint.h>

/* codes: a winding_mod14 raster (owned by the result); rmax: the pair radius in codes voxels (phase-raster --radius / --gs). */
z3 *phase_onfly_open(z3 *codes, float rmax);
/* process-wide: the pair rule on CUDA device gpu (-1: CPU) with `slots` concurrent tiles of at most budget_mb device bytes each
   (the tile edge follows from it); reserve 1 allocates the slots' device buffers now (before the trainer's memory planner looks). */
int phase_onfly_config(int gpu, int slots, int budget_mb, int reserve);
/* the tile edge (codes voxels, halo included) for rmax under the configured budget */
int phase_onfly_tile_edge(float rmax);
/* counters since the start: derived reads, tiles computed / skipped (no codes), output voxels, tile voxels, seconds in the pair rule
   and in the codes reads (summed over threads) */
typedef struct { uint64_t reads, tiles, skipped, vox_out, vox_tile, gpu_tiles; double s_pair, s_read; } phase_onfly_stats;   /* gpu_tiles: codes decoded on the device */
void phase_onfly_get_stats(phase_onfly_stats *st);
/* tests: the codes of the box o / n decoded on the device (band_pair_gpu_codes with rmax 0) into out; 0 ok, else the GPU path's code */
int phase_onfly_decode_gpu(z3 *codes, const int64_t o[3], const int64_t n[3], uint8_t *out);
