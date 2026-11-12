/* Winding codes on the fly from q-assigned surface meshes (2026-11-11): a derived array (zarr3.h z3_open_derived) whose z3_read(o, n)
   rasterises the quads of <dir>/NAME.tifxyz (x / y / z.tif + the per-vertex winding <value>, e.g. q.tif) that reach the box, on the grid of
   CT level `level`: byte for byte the output of `ufsm raster --value q.tif --binary 0 --level L` (src/ingest.c raster_shard), so no codes
   raster is stored (Paris 4 June at 2.4 um: ~170 GB; at 1.129 um ~770 GB; the meshes are 5.8 GB). The rule is order-defined and kept:
   samples run in (mesh by name, 32 x 32 tile, row, column, a, b) order; a voxel takes the first sample's code and becomes 255 when a
   later sample is more than 4 codes (1/4 turn) from it.
   map: a ufsm reg map (reg.json) whose coarse scan is the meshes' scan: every vertex goes through rmap_c2f (level-0 x, y, z) into the
   fine scan first (the 1.129 um twins); nullptr = the meshes' own scan.
   ct_level: an open array of the target level (meta only: shape, chunk / shard grid); taken over (closed on failure too).
   The known extent (z3_derived_extent: the sampler's draw cells) is the chunks the tiles' bounds reach.
   gpu: reserved (the device stamp); the CPU reference runs for now. */
#pragma once
#include "zarr3.h"
z3 *mesh_codes_open(const char *dir, const char *value, const char *map, int level, z3 *ct_level, int gpu);
int cmd_codes_onfly(int argc, char **argv);   /* ufsm codes-onfly: the derived codes over windows, ms and byte diffs vs a stored raster */
