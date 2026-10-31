/* zarr v3 reader for the published volcomp tree: uint8 arrays, `sharding_indexed` shards of 128^3
   inner chunks, inner codecs `volcomp(q)` optionally followed by `zstd`. Reads any (z,y,x) box; missing
   shards / chunks are the fill value (0). Inner chunks fetched from an HTTP store are cached on disk under
   <cache>/<key>/<shard>.<i> so a region is only ever downloaded once. */
#pragma once
#include "store.h"
#include <stdint.h>

typedef struct {
    int64_t shape[3];     /* z, y, x */
    int shard[3];         /* outer chunk (shard) shape; == chunk when unsharded */
    int chunk[3];         /* inner chunk shape */
    int sharded;
    float q;              /* volcomp q; -1 if the codec chain has no volcomp */
    int zstd;             /* 1 if zstd follows volcomp */
    int raw;              /* 1 if bytes codec only (uncompressed) */
    char sep;             /* chunk key separator */
    double scale_um;      /* voxel size from the parent group's OME multiscales, 0 if unknown */
    int fill;             /* fill_value for missing chunks / shards */
    int label_binary;     /* ufsm.encoding=binary: 255 is positive, 0 is background */
    int dtype, esize;     /* element type (volcomp_dtype: 0 uint8, 1 uint16, 2 int16, 3 uint32, 4 int32, 5 float32) and bytes; z3_read
                             fills esize bytes per voxel (native byte order) */
    double fillv;         /* fill_value of a typed array */
} z3_meta;

typedef struct z3 z3;

/* key = path of the ARRAY directory inside the store (e.g. ".../foo.zarr/0"). cache may be nullptr. */
z3 *z3_open(store *s, const char *key, const char *cache_dir);   /* uint8 arrays only */
z3 *z3_open_typed(store *s, const char *key, const char *cache_dir);   /* also uint16 / int16 / uint32 / int32 / float32 (meta.esize bytes per voxel) */
void z3_close(z3 *z);
const z3_meta *z3_meta_of(const z3 *z);
const char *z3_key(const z3 *z);

/* Read the box [o, o+n) (z,y,x) into out (C order, n[0]*n[1]*n[2] bytes). Out-of-range parts use fill_value.
   nthreads <= 0 picks the CPU count. Returns 0 on success, -1 on I/O or decode failure. */
int z3_read(z3 *z, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads);   /* out: n[0] n[1] n[2] * esize bytes */
/* z3_read decoding only the volcomp substreams in submask of each chunk (bit s = substream s: blocks 16 s .. 16 s + 15, i.e. the
   16 z x 32 y x 128 x slab z0 = 16 (s / 4), y0 = 32 (s % 4)); voxels of the other substreams read as 0. The decoded ones equal a full
   decode byte for byte (volcomp_decode_block's guarantee). submask 0 or all ones, or a chunk that is not a lossy / coded volcomp chunk
   (mask, surface, raw, const, typed, other codecs): the whole chunk. */
int z3_read_sub(z3 *z, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads, uint32_t submask);

/* Shard-level presence: 1 if the shard object exists in the store, 0 if not, -1 on error. */
int z3_shard_present(z3 *z, int64_t sz, int64_t sy, int64_t sx);
/* Inner-chunk presence of a sharded array from its shard indices alone (no chunk is read or decoded): bits[k] = 1 when chunk k
   (z-major over the chunk grid ceil(shape / chunk), grid[] filled) is stored; chunks equal to the fill value are not (z3w omits
   them). Returns the number present, -1 on error or an unsharded array. bits: grid[0] * grid[1] * grid[2] bytes (caller frees). */
int64_t z3_chunk_map(z3 *z, int64_t grid[3], uint8_t **bits);

/* OME group: list the level array paths and their voxel sizes. Returns count, fills up to max. */
typedef struct { char path[64]; double um; } z3_level;
int z3_group_levels(store *s, const char *group_key, z3_level *out, int max);

const char *z3_error(void);
void z3_io_stats(uint64_t *cache_hits, uint64_t *store_reads);
/* the shared decoded-substream cache (UFSM_ZCACHE_GB, src/zshm.h) in this process: substream hits, misses, stored */
void z3_shared_cache_stats(uint64_t *hits, uint64_t *misses, uint64_t *stored);   /* chunk reads served from the cache vs fetched from the store */
int z3_prefetch_chunk(z3 *z, int64_t cz, int64_t cy, int64_t cx);   /* fetch one inner chunk into the cache (no decode): 1 fetched, 0 cached/absent, -1 error */
int z3_chunk_raw(z3 *z, int64_t cz, int64_t cy, int64_t cx, uint8_t **buf, size_t *n);   /* encoded inner chunk bytes (after zstd): 1 with *buf (malloc), 0 absent, -1 error */
