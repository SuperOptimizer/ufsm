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
/* A derived array: z3_read(z) calls derive(ctx, base, o, n, out, nthreads) (o / n any box, like z3_read; out n[0] n[1] n[2] bytes);
   its meta is base's (shape, shard / chunk grid) with fill_value fill; z3_chunk_map / z3_box_any / z3_shard_present answer from base
   (the chunks derive can make non-fill). z3_close closes base and calls ctx_free(ctx). Sub-chunk / device reads fail on it. */
typedef int (*z3_derive_fn)(void *ctx, z3 *base, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads);
z3 *z3_open_derived(z3 *base, int fill, z3_derive_fn derive, void *ctx, void (*ctx_free)(void *));
z3 *z3_base_of(const z3 *z);   /* a derived array's base, else nullptr */
void *z3_derived_ctx(const z3 *z);   /* a derived array's ctx, else nullptr */
void z3_derived_borrow(z3 *z);   /* z3_close(z) then leaves its base open (a derived view of an array its owner keeps) */
/* more arrays (same shape, up to 4) whose chunks the derived values also reach (e.g. a fusion of several sets): z3_chunk_map / z3_box_any /
   z3_shard_present answer for the union; taken over (closed with z). 0 or -1 */
int z3_derived_cover(z3 *z, z3 *cover);
/* a derived array on the 2x coarser grid of base (shape ceil(base / 2), the same chunk / shard edges; derive gets coarse boxes): its
   chunk map / box_any / shard presence from the base's chunks (each coarse chunk = 2^3 base chunks). nullptr on failure (base kept) */
z3 *z3_open_derived_pool2(z3 *base, int fill, z3_derive_fn derive, void *ctx, void (*ctx_free)(void *));
/* the derived values' known extent (sparse targets: the sampler's draw cells, occupancy tests): fn marks bits[(cz grid[1] + cy) grid[2] + cx]
   != 0 for every chunk (the base's chunk grid, chunk edge `chunk`) where derive can give a known value; called once, on the first query.
   z3_chunk_map / z3_box_any / z3_shard_present then answer from it (instead of the base and covers). ctx stays the caller's (freed with the
   derive ctx when it is the same). 0 or -1 */
int z3_derived_extent(z3 *z, int (*fn)(void *ctx, const int64_t grid[3], const int chunk[3], uint8_t *bits), void *ctx);
const z3_meta *z3_meta_of(const z3 *z);
const char *z3_key(const z3 *z);

/* Read the box [o, o+n) (z,y,x) into out (C order, n[0]*n[1]*n[2] bytes). Out-of-range parts use fill_value.
   nthreads <= 0 picks the CPU count. Returns 0 on success, -1 on I/O or decode failure. */
int z3_read(z3 *z, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads);   /* out: n[0] n[1] n[2] * esize bytes */
int z3_box_any(z3 *z, const int64_t o[3], const int64_t n[3]);   /* any stored chunk in the box (shard indices only): 1, none 0, error -1 */
/* z3_read decoding only the volcomp substreams in submask of each chunk (bit s = substream s: blocks 16 s .. 16 s + 15, i.e. the
   16 z x 32 y x 128 x slab z0 = 16 (s / 4), y0 = 32 (s % 4)); voxels of the other substreams read as 0. The decoded ones equal a full
   decode byte for byte (volcomp_decode_block's guarantee). submask 0 or all ones, or a chunk that is not a lossy / coded volcomp chunk
   (mask, surface, raw, const, typed, other codecs): the whole chunk. */
int z3_read_sub(z3 *z, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads, uint32_t submask);
/* z3_read_sub of one whole aligned 128^3 chunk with UFSM_GPU_DECODE on: the GPU decodes lossy substreams straight into the device chunk
   dchunk (128^3, chunk layout, on the decoder's device; out keeps zeros there); *dev_mask = those substreams (0: everything went to out) */
int z3_read_sub_dev(z3 *z, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads, uint32_t submask, uint8_t *dchunk, uint32_t *dev_mask);
/* the same, but out holds only the substreams decoded on the host (want & ~*dev_mask; the rest of out is left as it was: no zeroing,
   no copy of the device ones): for callers that upload just those (render --gpu) */
int z3_read_sub_dev_lean(z3 *z, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads, uint32_t submask, uint8_t *dchunk, uint32_t *dev_mask);

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
/* the calling thread's volcomp decodes on the CPU (1) even with the GPU decoder on (predict's device CT windows: CPU workers beside the GPU ones) */
void z3_thread_cpu_decode(int on);
/* the shared decoded-substream cache (UFSM_ZCACHE_GB, src/zshm.h) in this process: substream hits, misses, stored */
void z3_shared_cache_stats(uint64_t *hits, uint64_t *misses, uint64_t *stored);   /* chunk reads served from the cache vs fetched from the store */
int z3_prefetch_chunk(z3 *z, int64_t cz, int64_t cy, int64_t cx);   /* fetch one inner chunk into the cache (no decode): 1 fetched, 0 cached/absent, -1 error */
int z3_chunk_raw(z3 *z, int64_t cz, int64_t cy, int64_t cx, uint8_t **buf, size_t *n);   /* encoded inner chunk bytes (after zstd): 1 with *buf (malloc), 0 absent, -1 error */
uint64_t z3_sink_chunks(void);   /* chunks decoded straight into a request's output (the window sink), process-wide */

