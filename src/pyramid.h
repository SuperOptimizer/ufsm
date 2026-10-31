/* Pyramid helpers shared by ingest / raster / predict: build level l (>= 1) of a volcomp zarr v3 group from
   level l-1 (labels=1: fraction/ignore pooling; labels=2: binary union pooling
   with lossless mask coding; labels=3: known-max pooling, 255 = unknown (ink-splat); otherwise mean pooling), and write the
   OME group file. */
#pragma once
#include <stdint.h>
int pyramid_build_level(const char *group_dir, double um0, int l, const int64_t shape0[3], int shard, float q, int labels, int nthreads, const char *attrs);
/* level l over its shard z-layers [z0, z1) only (z1 < 0: to the end); quiet: no timing line */
int pyramid_build_level_z(const char *group_dir, double um0, int l, const int64_t shape0[3], int shard, float q, int labels, int nthreads, const char *attrs,
                          int64_t z0, int64_t z1, int quiet);
int pyramid_write_group(const char *dir, double um0, int nlev, const char *name, const char *attrs);
