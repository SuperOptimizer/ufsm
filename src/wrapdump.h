/* Wrap vertices of a z slab (src/wrapdump.c; ufsm wrap-dump, ufsm fiber-judge --wraps). */
#pragma once
#include <stddef.h>
/* Every wNNN.sfc of dir with first <= NNN <= last: its vertices with z in [z0, z1), y in [yx[0], yx[1]), x in [yx[2], yx[3]) (voxels of
   gum; gum 0: the wraps' own voxel size; voxel index = centre), every ev-th row and column, as 5 floats each (z, y, x, id, column / width)
   into *out (malloc'd, *n records). Rows are z, columns run around the turn from theta = +pi. Returns the wrap count or -1. */
int wraps_slab_load(const char *dir, double z0, double z1, const double yx[4], int ev, double gum, int first, int last, float **out, size_t *n);
