/* predict --cond-sweep (train --cond checkpoints): conditioned inference over a box. Windows run one after another in a z, y, x raster
   (interiors tile the box); each gets, as inputs 4-6, the 14-turn winding phasor the earlier windows wrote where they reached (the
   box canvas on the device), so one gauge is carried across the box. The result: q mod 14 per voxel (the 14-turn phasor snapped to
   the 1-turn phase plus dir theta / 2 pi), unwrapped from the seed by a confidence-ordered flood into U (turns), written as
   <out>/U.f32 + sweep.json for ufsm wind-score. */
#pragma once
#include "unet.h"
#include "zarr3.h"
#include "sources.h"

typedef struct {
    unet *u; int ch14;                  /* the network (cin 7, all heads computed); its 14-turn phasor channels ch14, ch14 + 1 */
    z3 *ct; int64_t bo[3], bn[3];       /* the CT level, the box on it */
    double um, scale;                   /* the level's voxel size (um), level voxels per native voxel */
    const axis *ax; const char *axis_path; int axis_flag;   /* the umbilicus, its file, nn_pred_input's radial flag (0: none) */
    int W, halo, h16, nthreads, dir, reverse;   /* window, halo; 16-bit inputs; handedness (+1: q = sheet + theta / 2 pi); reversed sweep order */
    const char *out, *ckpt;
} sweep_cfg;

int cond_sweep(const sweep_cfg *c);
/* the host half (exposed for tests): from the canvas planes (c14 2N, p1 2N, done N; src/nn_elem.cu nn_cond_place) and the axis per box z
   (cy, cx: box-local level voxels, bn[0] entries), q mod 14 and its flood from voxel `seed` into U (NaN where not reached); returns the
   number of voxels reached */
size_t sweep_unwrap(const uint8_t *c14, const uint8_t *p1, const uint8_t *done, const int64_t bn[3], const float *cy, const float *cx, int dir, size_t seed, float *U);
