/* Checks of a stored band phase (src/band.h PHASE_*: 0..251, 255 unknown) against the convention the fitter and the trainer assume
   (fit.h W, slab/phase-pair): the phase rises outward from the scroll's axis, one period per sheet (2026-10-19: the 10-16 multi-scroll
   labels stored Paris 4 June / July and PHerc0500P2 rising inward, unnoticed until the fine-tune). Used by ufsm phase-check and
   ufsm phase-raster --mask-ref. */
#pragma once
#include <stddef.h>
#include <stdint.h>

/* The radial component of the phase gradient over a box p (n[0] x n[1] x n[2], z-major) whose voxel (0,0,0) sits at y0 / x0 (phase
   voxels; cy[z] / cx[z]: the axis at each z, phase voxels): forward differences along y and x where both voxels are known and differ
   by under 60 codes (mod 252: a sheet's wrap 251 -> 0 is +1), their radial projection |g_r| > 0.5 counted, *up the ones rising
   outward of *tot. The same rule as phcmp/signcheck.py (2026-10-19). */
void phase_outward_count(const uint8_t *p, const int n[3], int64_t y0, int64_t x0, const double *cy, const double *cx, size_t *up, size_t *tot);

/* agreement of two phases on one grid, in turns |d| (circular): bins < 0.05, 0.05..0.1, 0.1..0.2, 0.2..0.4, >= 0.4 */
enum { PHASE_AGREE_BINS = 5 };
static inline int phase_agree_bin(double d) { return d < 0.05 ? 0 : d < 0.1 ? 1 : d < 0.2 ? 2 : d < 0.4 ? 3 : 4; }
/* out (the phase being written) against ref on the same voxels: where both are known, the |difference| histogram (hist, added to)
   and, where it exceeds tol turns, out set to unknown (*masked added to). Returns the voxels known in both. */
size_t phase_mask_ref(uint8_t *out, const uint8_t *ref, size_t n, double tol, size_t hist[PHASE_AGREE_BINS], size_t *masked);
