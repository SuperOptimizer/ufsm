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
/* the 2x circular-mean pooling of winding_mod14 codes (phase-raster --gs 4's praster_read, `ufsm codes-pool2`): of a box of the coarse
   grid, and as a derived array (shape ceil(codes / 2), fill 0; codes taken over; nullptr on failure, codes kept). The onfly phase over
   it equals phase-raster --gs 4 (the pair rule on the GPU, the codes pooled on the CPU) */
int codes_pool2_box(z3 *codes, const int64_t ro[3], const int64_t rn[3], uint8_t *out, int nthreads);
uint8_t codes_pool2_block(const uint8_t v8[8], int ref);   /* one 2^3 block (children in z, y, x bit order, 0 outside); ref 1: the reference loop */
extern int codes_pool2_ref;                                /* tests: 1 the reference loop in codes_pool2_box */
z3 *codes_pool2_open(z3 *codes);
/* a stored phase pooled 2x (circular mean of the known codes, unknown under half resultant: downsample --mode phase); ign 1: an "ignore"
   array's pooling (1 where any child is set). phase_pool2_view: the lazy derived array on the 2x coarser grid, borrowing phase (closing
   the view leaves it open) */
int phase_pool2_box(z3 *ph, const int64_t ro[3], const int64_t rn[3], uint8_t *out, int nthreads, int ign);
z3 *phase_pool2_view(z3 *phase, int ign);
/* a phase view's reads so far: coarse voxels with >= 1 known child, and of them known after the pooling (the rest: incoherent blocks) */
int phase_pool2_view_stats(const z3 *view, uint64_t *any, uint64_t *known);
/* several phase sets of one scroll on one grid (each phase_onfly_open'd or stored; all taken over, closed on failure too) fused as
   phase-fuse does (src/fuse.h fuse_phase, --tol turns): byte-identical to `ufsm phase-fuse` over the same sets' rasters */
z3 *phase_onfly_fuse_open(z3 **sets, int n, float tol);
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
