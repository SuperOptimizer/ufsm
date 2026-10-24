/* Second-scan views for scan fusion (train / predict --view-P, unet_cfg.view_P): a Pv^3 crop on a source scan's level-lv grid, centred
   on a training / prediction window, read either from the scan itself (no map: e.g. 2.4 um detail around a 9.6 um window) or from a
   registered partner scan through a ufsm reg map (fine(c) = A^-1 (c + D(c)), src/reg.h; the source scan is the map's coarse scan, its
   level-lv grid the map's coarse level-lv grid; the partner is resampled trilinearly from its level nearest that voxel size, e.g. the
   137 keV scan of Paris 4 on the 78 keV scan's 2.4 um grid). Outside the scan (or the partner) the crop is 0 (the masked scans' void). */
#pragma once
#include "reg.h"
#include "zarr3.h"
#include <stdint.h>

typedef struct sview {
    int lv;               /* the crop's level on the source scan's grid */
    int has_map;          /* 0: the source scan itself (the caller's level-lv array) */
    rmap m; rfield D;     /* the map (has_map) */
    rvol *fv; int lf;     /* the partner scan and its level read */
    double cum, fum;      /* the map's coarse / fine level-0 voxel sizes */
} sview;

/* map: a ufsm reg map JSON (nullptr or "": the source scan itself); lv: the crop level; src_um: the source scan's level-0 voxel (checked
   against the map's coarse scan, 2%); cache: the zarr cache directory (nullptr: none). nullptr on error (message printed) */
sview *sview_open(const char *map, int lv, double src_um, const char *cache);
void sview_free(sview *v);
/* the crop at level-lv voxel origin o (zyx), Pv^3 voxels, into out (z y x). self: the source scan's level-lv array (used without a map).
   0 on success, -1 on a read error */
int sview_read(const sview *v, z3 *self, const int64_t o[3], int Pv, uint8_t *out);
/* the crop origin (level-lv voxels, zyx) centred on a window of n (zyx) level-l voxels at corner o (level-l voxels), l >= lv:
   the window centre (o + n / 2) 2^(l - lv) minus Pv / 2 (level grids nest: level-l voxel k covers level-lv voxels k 2^(l-lv) ..) */
static inline void sview_origin(int l, int lv, const int64_t o[3], const int64_t n[3], int Pv, int64_t vo[3]) {
    for (int d = 0; d < 3; d++) vo[d] = ((2 * o[d] + n[d]) << (l - lv)) / 2 - Pv / 2;
}
