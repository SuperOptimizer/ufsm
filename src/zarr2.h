/* zarr v2 reader (upstream label / CT arrays): 3-D uint8 arrays with blosc, zstd, zlib or raw chunks.
   Same box-read semantics as zarr3; HTTP chunks are cached on disk under <cache>/<key>/<chunk>. */
#pragma once
#include "store.h"
#include <stdint.h>

typedef struct {
    int64_t shape[3];
    int chunk[3];
    char sep;              /* '.' or '/' */
    int comp;              /* 0 raw, 1 blosc, 2 zstd, 3 zlib, 4 gzip */
    int fill;
} z2_meta;

typedef struct z2 z2;

z2 *z2_open(store *s, const char *key, const char *cache_dir);
void z2_close(z2 *z);
const z2_meta *z2_meta_of(const z2 *z);
int z2_read(z2 *z, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads);
/* 1 if the chunk object exists, 0 if missing, -1 on error */
int z2_chunk_present(z2 *z, int64_t cz, int64_t cy, int64_t cx);
/* When set, chunk reads skip the existence probe (HEAD) and GET directly; a 404 then counts as missing. */
void z2_assume_present(z2 *z, int on);
/* Decode one chunk body with compressor code comp (0 raw, 1 blosc, 2 zstd, 3 zlib, 4 gzip) into cv bytes. */
int z2_decode(int comp, const uint8_t *in, size_t n, uint8_t *out, size_t cv);
int z2_comp_of(const char *id);   /* -1 if unsupported */
const char *z2_error(void);
