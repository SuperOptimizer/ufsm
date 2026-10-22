/* Phase U0 data plumbing (plan ufsm-unsup-40scrolls): a local mirror of one pyramid level (ufsm mirror) and the per-scan prep the
   multi-scroll sampler reads (ufsm scanprep): a coarse object mask on a cell grid and robust CT intensity statistics.

   ufsm mirror <root> <array-key> <out-dir> [--threads 4] [--mbps 0] [--link-from DIR] [--verify N] [--seed 1]
     Copies every shard file of one zarr v3 array (e.g. ".../X.zarr/0" of the volcomp mirror) byte for byte to <out-dir> (zarr.json +
     c/z/y/x), so the local copy keeps the mirror's shard layout (1024^3 shards of 128^3 volcomp chunks). Resumable: a shard whose local
     file has the remote size is skipped; --link-from DIR hard-links an identical-size shard from an older local copy of the same array
     instead of downloading it (same filesystem, no extra space). --mbps caps the download rate (MB/s, all threads). --verify N then
     compares N random present inner chunks decoded from the local copy and from the remote (memcmp + crc32c) and the crc32c of the
     whole shard files they lie in. Exit 0 when every shard is present and verified.

   ufsm scanprep <root> <group-key> --level L --out PREFIX [--um U] [--cell 16] [--close-mm 2] [--window 128] [--min-inside 0.25]
                 [--keep-frac 0.1] [--fill2d 1] [--stats-chunks 256] [--threads 16] [--seed 1]
     Level L of a CT pyramid group (the ~9.6 um level the sampler draws). The cell grid has cells of --cell level-L voxels; each cell's
     mean over its NONZERO voxels and its nonzero count come from the finest of levels L+2, L+3, L+4 present (else L itself). Object
     mask: Otsu threshold of the cell means (cells with any nonzero voxel), morphological closing with a ball of --close-mm, 6-connected
     components smaller than --keep-frac x the largest dropped (sample mounts / holder bits at the z ends; 0 keeps all), holes
     filled (3-D from the grid border, then per z plane with --fill2d), intersected with the cells holding nonzero voxels (the scan's
     own mask). Valid window centres: object cells whose --window^3 (level-L voxels) box around the cell centre is at least
     --min-inside object cells (lower it for thin fragments). Intensity statistics over the nonzero voxels of object cells, from up to
     --stats-chunks random 128^3 inner chunks of level L: 0.5 / 25 / 50 / 75 / 99.5 percentiles, MAD and scale = 1.4826 MAD, mean, sd,
     the voxel Otsu cut, the 256-bin histogram. Outputs: PREFIX.json, PREFIX.mask.raw (uint8 cells z-major: 0 outside, 1 object,
     2 object and a valid window centre) with PREFIX.mask.raw.json {"grid", "cell", "level", "lev"} (lev = log2 of the cell edge in
     level-0 voxels: the sampler's occupancy level), PREFIX.mask.pgm (mid z / mid y planes: cell means | mask). */
#pragma once
#include <stddef.h>
#include <stdint.h>

int cmd_mirror(int argc, char **argv);
int cmd_scanprep(int argc, char **argv);

/* Otsu threshold of a histogram of nb bins (bin index = value): the t in 1..nb-1 maximising the between-class variance of
   [0, t) vs [t, nb); values >= t are foreground. Returns -1 when fewer than two bins are populated. */
int sp_otsu(const double *hist, int nb);
/* Morphological closing of a binary grid (m[k] != 0) with a Euclidean ball of radius r cells (exact EDT): dilate then erode. Cells
   outside the grid count as neither (no erosion from the border). In place; 0 on success, -1 on allocation failure. */
int sp_close(uint8_t *m, const int64_t n[3], float r, int nthreads);
/* Hole filling: background cells not 6-connected to the grid border through background become foreground (3-D); with plane2d,
   then the same per z plane (4-connected): the open-ended core of a scroll. In place; 0 / -1 on allocation failure. */
int sp_fill_holes(uint8_t *m, const int64_t n[3], int plane2d);
/* Drops 6-connected foreground components smaller than frac x the largest (m: 0 / 1, in place). Returns the number kept (*ncomp: found),
   -1 on allocation failure. */
int64_t sp_keep_components(uint8_t *m, const int64_t n[3], double frac, int64_t *ncomp);
/* Valid centres: out[k] = 2 where m[k] and the box of edge w cells centred on cell k (cells k - w/2 .. k - w/2 + w - 1, clipped to the
   grid; outside counts as background, divided by the full w^3) holds at least min_inside * w^3 foreground cells, else 1 where m[k],
   else 0. Returns the number of valid centres, -1 on allocation failure. */
int64_t sp_valid_centres(const uint8_t *m, const int64_t n[3], int w, double min_inside, uint8_t *out);
/* The q-quantile (0..1) of a 256-bin histogram (lowest value whose cumulative count reaches q * total); -1 when empty. */
double sp_hist_quantile(const uint64_t *h, double q);
