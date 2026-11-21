#pragma once
/* predict --surf (src/surfpred.c; DESIGN 2026-11-20 "inksurf"): the surface ink head on unrolled wraps (tifxyz), maps in the
   ink-wrapscore --maps format */
#include "unet.h"
#include <stdint.h>
typedef struct surfp surfp;
/* wraps: comma-separated tifxyz dirs; isf: the head (train --ink-surf's CKPT.isf); C0 the trunk's level-0 width, Hp the patch side
   (multiple of 16), halo px dropped each side, K layers, s: level-0 voxels per window voxel (= per map pixel) */
surfp *surfp_open(const char *wraps, const char *isf, int C0, int Hp, int halo, int K, double s);
void surfp_free(surfp *S);
/* the windows to run (level origins, extent n, interior margin m): one centred on each patch not yet placed, taking its neighbours that fit;
   *o malloc'd [3 per window]; returns the count. Window t's patches run in surfp_window(t) */
long surfp_plan(surfp *S, const int64_t n[3], const int m[3], int64_t **o);
int surfp_window(surfp *S, unet *u, long t, const int64_t o[3], int D, int H, int W);   /* after window t's forward: its patches into the maps */
int surfp_write(surfp *S, const char *dir);   /* DIR/<wrap>.tif per wrap */
