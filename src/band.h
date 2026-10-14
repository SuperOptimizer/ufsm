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
    float gs;          /* native voxels per grid voxel of the codes (0 = 2: the 4.8 um label grid of a 2.4 um scan; 4: codes pooled to 9.6 um) */
} band_params;

/* codes: winding_mod14 raster on the label grid (one label voxel = gs native voxels, default 2), dims n[3] (z, y, x), origin o[3] in
   label voxels; cy, cx: umbilicus centre (native voxels) at each label z plane. out: band per voxel (0..251 or 255).
   Returns 0, or -1 when out of memory. Neighbourhoods are limited to the array: callers pass a halo of at least
   (radius + span) / gs + 2 label voxels and crop; band_gs(p) is gs with the default applied. */
static inline float band_gs(band_params p) { return p.gs > 0 ? p.gs : 2.f; }
int band_field(const uint8_t *codes, const int n[3], const int64_t o[3], const double *cy, const double *cx, band_params p, uint8_t *out);
/* the same plus the side of every known voxel relative to its nearest labelled recto: SIDE_RECTO outward of it (away from the
   umbilicus, the papyrus behind the recto face and the inner half of the gap beyond it), SIDE_VERSO inward (the outer half of the
   gap before it), SIDE_UNKNOWN where the band is unknown. Going outward the side turns VERSO -> RECTO exactly at each labelled
   recto face and RECTO -> VERSO midway between consecutive rectos: a dense two-class label whose boundaries are the sheets. */
enum { SIDE_VERSO = 0, SIDE_RECTO = 1, SIDE_UNKNOWN = 255 };
int band_field_side(const uint8_t *codes, const int n[3], const int64_t o[3], const double *cy, const double *cx, band_params p, uint8_t *out, uint8_t *side);

/* the same plus the winding phase of every voxel between two labelled consecutive rectos: the fraction of the way outward from
   the recto it lies outward of to the next recto, times PHASE_PERIOD (0 on a recto face, PHASE_PERIOD / 2 midway, rising
   outward and wrapping to 0 at the next recto); PHASE_UNKNOWN elsewhere (beyond the outermost / innermost labelled recto,
   past the end of a trace, where the band is unknown). The recto side is phase < PHASE_PERIOD / 2. spacing (optional): the local
   distance between the two rectos (label voxels, 1..255; 0 where the phase is unknown). */
enum { PHASE_PERIOD = 252, PHASE_UNKNOWN = 255 };
int band_field_phase(const uint8_t *codes, const int n[3], const int64_t o[3], const double *cy, const double *cx, band_params p, uint8_t *out, uint8_t *phase,
                     uint8_t *spacing);

/* The same phase from the two nearest labelled sheets of opposite winding parity (consecutive windings alternate parity): a voxel
   between a sheet of winding k and one of winding k + 1 (the nearest even-winding and nearest odd-winding labelled voxels, on
   opposite sides of it) sits d_k / (d_k + d_k+1) of the way from k to k + 1; labelled voxels 0. No axis: band_field_phase decides
   inside / outside radially, which flips wherever a sheet is oblique to the radial direction (wavy, tilted, folded) and then blanks
   +- span around the false "missing wrap": on a Paris4 radial strip 22% (regular half) / 35% (expanded half) of the block voxels.
   Unknown beyond rmax label voxels from either sheet, where the two windings are not consecutive (a missing wrap) or the sheets are
   on one side. spacing (optional) as band_field_phase. */
int band_field_phase_pair(const uint8_t *codes, const int n[3], float rmax, uint8_t *phase, uint8_t *spacing);

/* modular difference of band / winding steps in (-126, 126] */
static inline int band_diff(int a, int b) { int d = ((a - b + 126) % 252 + 252) % 252; return d - 126; }

/* The winding mod 14 turns of a voxel in BAND_STEPS per turn, in the phase encoding (0..251, PHASE_UNKNOWN unknown), from its
   band (the winding of the recto inward of it + BAND_HALF) and its phase (the fraction of the way to the next recto outward x
   PHASE_PERIOD): (band - BAND_HALF + phase / 14) mod 252. What ufsm phase-raster --winding writes. */
static inline uint8_t band_winding(uint8_t band, uint8_t phase) {
    if (band == BAND_UNKNOWN || phase == PHASE_UNKNOWN) return PHASE_UNKNOWN;
    const int w = (int)band - BAND_HALF + ((int)phase * BAND_STEPS + PHASE_PERIOD / 2) / PHASE_PERIOD;
    return (uint8_t)(((w % BAND_PERIOD) + BAND_PERIOD) % BAND_PERIOD);
}
