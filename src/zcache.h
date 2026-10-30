/* zcache: a thread-safe LRU of decoded z3 inner chunks shared by sampler threads (e.g. the vwt ray sampler's CT / GT / shell
   reads: overlapping slabs decode each chunk once while it stays in the budget). Reads give the same bytes as z3_read. */
#pragma once
#include "zarr3.h"
#include <stddef.h>
#include <stdint.h>

typedef struct zcache zcache;
zcache *zcache_open(size_t budget_bytes);   /* nullptr on failure */
void zcache_close(zcache *c);
/* the box [o, o + n) (z, y, x) of z into out (C order, esize bytes per voxel) through the cache; parts outside the array are the
   fill value (as z3_read). Arrays with esize != 1 are read directly (no caching). 0 ok, -1 on a decode / I/O failure. */
int zcache_read(zcache *c, z3 *z, const int64_t o[3], const int64_t n[3], uint8_t *out);
void zcache_stats(const zcache *c, uint64_t *hits, uint64_t *misses, size_t *bytes);
