/* src/phasenear.h: ufsm phase-near (phasenear.c): predicted phase-0 sheets vs GT sheets on the level-0 grid. */
#pragma once
#include "fit.h"
#include <stddef.h>
#include <stdint.h>

/* distances in voxels: pg pred sheet -> nearest GT sheet, gp GT sheet -> nearest pred sheet (none: 1e6); within[k]: shares <= 1 / 2 / 4 */
typedef struct { double known; size_t gt_sheet, pred_sheet; float pg50, pg90, gp50, gp90; double pg_within[3], gp_within[3]; } phase_near_res;
/* the fields' phase (turns in [0, 1)) and confidence at the voxel centres of the box o, n of a grid of voxel size um0 (trilinear phase vectors) */
void phase_near_sample(const fit_fields_buf *F, const int64_t o[3], const int64_t n[3], double um0, float *ph, float *cf);
/* g: the GT phase (band.h: 0 = a labelled sheet, PHASE_UNKNOWN outside the region); predicted sheets: the voxel nearer the integer of each
   6-neighbour pair whose phase wraps, both confidences > cmin; only voxels >= margin in from the faces counted. 0 ok, -1 out of memory */
void phase_near_set_min_comp(long n);   /* --min-comp: drop predicted sheet components (26-connected) of fewer than n voxels before the distances (0 off) */
int phase_near_measure(const uint8_t *g, const float *ph, const float *cf, const int64_t n[3], int margin, float cmin, int nthreads, phase_near_res *r);
