/* Per-voxel fusion of label sets rasterised separately (several published surface sets of one scroll, each in its own winding
   numbering mapped to a common gauge by wind-seed / wind-assign): no union of surfaces into one raster (two traces of one sheet
   a few voxels apart became two or three label lines), but
   - phase (src/band.h PHASE_*: 0..251, 255 unknown; normals and density follow from it in the sampler): where every set that
     knows a voxel agrees within tol turns, their circular mean; where two disagree by more, PHASE_UNKNOWN (ignored by the loss);
   - recto surface lines (binary label rasters, 0 none, > 0 surface): one source per block, never a union: the set whose surface
     voxels lie on the brightest CT (the papyrus ridge) is the block's primary; another set adds only its voxels farther than dup
     voxels from every primary surface voxel (a different sheet, e.g. the next wrap), never a parallel copy of the same sheet. */
#pragma once
#include <stddef.h>
#include <stdint.h>

/* in[k][i] (k < n): phase codes of len voxels; out: fused codes; stats (optional, 4): voxels known by 1 set, fused (>= 2 agreeing),
   conflicts (written unknown), unknown everywhere; hist (optional, PHASE_PERIOD / 2 + 1 bins): at voxels known by >= 2 sets, the
   largest pairwise circular difference in codes (the duplicate-offset statistics that set tol) */
void fuse_phase(const uint8_t *const *in, int n, size_t len, float tol_turns, uint8_t *out, size_t *stats, size_t *hist);

/* Binary recto rasters in[k] and the CT ct on one grid of d0 x d1 x d2 voxels (z-major); blocks of block^3; out: the chosen
   surface (255) or 0. A surface voxel of a set is dropped when another set has a surface voxel within dup voxels whose block
   (block^3) traces the CT ridge better (higher mean CT over that set's surface voxels in the block). stats (optional, 3): voxels
   kept where another set was near (the better trace), voxels kept with no other set near (unique coverage), voxels dropped. */
void fuse_recto(const uint8_t *const *in, int n, const uint8_t *ct, int64_t d0, int64_t d1, int64_t d2, int block, int dup,
                uint8_t *out, size_t *stats);
