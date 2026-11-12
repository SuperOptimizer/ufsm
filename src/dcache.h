/* src/dcache.h: a chunk cache for derived arrays (2026-11-11). z3_open_cached(inner, ...) returns a derived array whose z3_read is served
   in C^3 cells aligned to the array grid: each cell is looked up in a bounded in-memory LRU, then on disk under DIR/<inputs hash>/, and
   only on a miss read from inner (whose derive computes it) and stored in both. The inputs hash is SHA-256 of the caller's key string,
   which must name everything the values depend on (inputs, their versions, the parameters, a code version); DIR/<hash>/inputs.txt
   holds it in plain text. Deleting DIR (or any part of it) is always safe: missing cells are recomputed. Concurrent readers of one cell
   wait for the first (one compute per cell per process); processes sharing DIR race only on writes, which are atomic renames.
   Meta / chunk map / box tests come from inner (and so from its base). */
#pragma once
#include "zarr3.h"
#include <stddef.h>
#include <stdint.h>

/* inner (owned): any array, typically z3_open_derived. dir: the cache root (nullptr or "" = memory only); key: the inputs description;
   cell: the cell edge (0 = 128); mem_mb: the memory LRU cap (0 = 1024). Returns nullptr (inner closed) on failure. */
z3 *z3_open_cached(z3 *inner, const char *dir, const char *key, int cell, size_t mem_mb);
/* DIR/<24 hex of sha256(key)> (created, inputs.txt written): where a cached array or a companion file (an index) of key lives. 0 ok */
int dcache_key_dir(const char *dir, const char *key, char *out, size_t nout);
/* the default cache root: $UFSM_DERIVED_CACHE ("off" / "0" = none), else $HOME/.cache/ufsm/derived */
const char *dcache_default_dir(void);
typedef struct { uint64_t reads, cells_mem, cells_disk, cells_computed, cells_fill, disk_bytes_written; double s_compute, s_disk; } dcache_stats;
void dcache_get_stats(dcache_stats *st);   /* process-wide sums over every cached array */
