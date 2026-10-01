/* zarr v3 writer in the published volcomp layout: uint8 array, `sharding_indexed` shards of 128^3 inner
   chunks, inner codec volcomp(q) (q = 0 lossless), index at end with crc32c, all-zero chunks omitted.
   Shards are written whole; a pyramid group writes OME multiscales with levels named by voxel size. */
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct z3w z3w;

/* dir is created. shard must be a multiple of 128 (typically 1024, or 128 for small arrays).
   fill: fill_value (chunks entirely equal to it are omitted). attrs_json: attributes object text or nullptr. */
z3w *z3w_create(const char *dir, const int64_t shape[3], int shard, float q, int fill, const char *attrs_json);
/* data: shard^3 bytes in C order (the part outside the array is ignored). Returns 0 on success.
   Thread-safe across different shards. */
int z3w_write_shard(z3w *w, int64_t sz, int64_t sy, int64_t sx, const uint8_t *data, int nthreads);
int z3w_close(z3w *w);
/* Write an OME group zarr.json at dir listing levels[i] with voxel sizes um[i]; names are "%g" of um. */
int z3w_write_group(const char *dir, const double *um, int nlevels, const char *name, const char *attrs_json);
/* Name used for a level: "%.10g" of the voxel size */
void z3w_level_name(double um, char *buf, int n);
uint32_t crc32c(uint32_t crc, const void *buf, size_t len);
const char *z3w_error(void);
