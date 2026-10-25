/* Reader of the dinovol-embed zarr v3 arrays (src/embz.c): sharded, zstd, channel-last int8 / float16 / float32 */
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

typedef struct { char dir[1024]; int64_t shape[3]; int C, esize, shard, chunk, dt; double vscale; int64_t ish[3]; uint8_t *idx; int64_t csh, cci; uint8_t *cb; int64_t cshard[3]; } zr;
int zr_open(zr *r, const char *dir);                                  /* 0 on success */
void zr_close(zr *r);
const uint8_t *zr_chunk(zr *r, int64_t z, int64_t y, int64_t x);      /* the decoded chunk holding output cell (z, y, x); nullptr absent */
int zr_read(zr *r, const int64_t o[3], const int64_t n[3], float *out);   /* float32 [n0][n1][n2](C), outside / absent 0 */
int zr_read_i8(zr *r, const int64_t o[3], const int64_t n[3], int8_t *out);   /* int8 arrays: the stored values [n0][n1][n2](C) (no value_scale), outside / absent 0; -1 other types */
/* an output with its optional per-vector scale array */
typedef struct { zr a, s; int has_s; } emb;
int emb_open(emb *e, const char *group, const char *name);
void emb_close(emb *e);
int emb_read(emb *e, const int64_t o[3], const int64_t n[3], float *out);
float ez_h2f(uint16_t h);

/* writer of sharded zarr v3 [Z][Y][X](C) arrays (zstd inner chunks; src/embz.c): create, put chunk-aligned blocks (a shard is written
   once: put every block of a shard before zw_flush), flush closes the open shards */
typedef struct { int64_t sz, sy, sx; FILE *f; char path[1200]; uint64_t off, *idx; } zw_shard;
typedef struct { char dir[1024]; int64_t shape[3]; int C, esize, shard, chunk, nc1; zw_shard *open; int nopen, cap; size_t chunk_bytes; uint8_t *cbuf, *zbuf; size_t zcap; } zw;
int zw_create(zw *w, const char *dir, const int64_t shape[3], int C, const char *dtype, int esize, int shard, int chunk, const char *attrs);
int zw_put(zw *w, int64_t oz, int64_t oy, int64_t ox, int nz, int ny, int nx, const void *data);
int zw_flush(zw *w);
void zw_free(zw *w);
extern int zw_nthreads;
