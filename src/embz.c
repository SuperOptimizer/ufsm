/* Reader of the dinovol-embed zarr v3 arrays (sharded, zstd, channel-last; int8 / float16 / float32): any region dequantised to float32
   (value_scale, and the <name>_scale per-vector array when present). Shared by dinovol-read / dinovol-eval (src/dinovol.c) and the
   fitter's embedding evidence (src/fit_emb.c). Not thread-safe per handle (one decoded-chunk cache). */
#include "embz.h"
#include "json.h"
#include "z3w.h"
#include <zstd.h>
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *read_file(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb"); if (!f) return nullptr;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    void *p = malloc(sz > 0 ? (size_t)sz : 1); if (!p || fread(p, 1, (size_t)sz, f) != (size_t)sz) { free(p); fclose(f); return nullptr; }
    fclose(f); *n = (size_t)sz; return p;
}
float ez_h2f(uint16_t h) {
    const uint32_t s = (uint32_t)(h & 0x8000u) << 16; int e = (h >> 10) & 0x1f; uint32_t m = h & 0x3ffu, x;
    if (e == 0) { if (!m) x = s; else { while (!(m & 0x400u)) { m <<= 1; e--; } e++; m &= 0x3ffu; x = s | ((uint32_t)(e + 112) << 23) | (m << 13); } }
    else if (e == 31) x = s | 0x7f800000u | (m << 13);
    else x = s | ((uint32_t)(e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &x, 4); return f;
}
#define h2f ez_h2f

int zr_open(zr *r, const char *dir) {
    memset(r, 0, sizeof *r); snprintf(r->dir, sizeof r->dir, "%s", dir); r->ish[0] = -1; r->csh = -1;
    char p[1200]; snprintf(p, sizeof p, "%s/zarr.json", dir); size_t n = 0; char *t = read_file(p, &n);
    if (!t) return -1;
    json *j = json_parse(t, n); free(t); if (!j) return -1;
    const json *sh = json_get(j, "shape"), *dtj = json_get(j, "data_type"), *cs = json_path(j, "chunk_grid.configuration.chunk_shape"), *ic = json_path(j, "codecs.0.configuration.chunk_shape");
    if (!sh || !dtj || !cs || !ic) { json_free(j); return -1; }
    for (int d = 0; d < 3; d++) r->shape[d] = (int64_t)json_num(json_at(sh, d), 0);
    r->C = sh->n > 3 ? (int)json_num(json_at(sh, 3), 0) : 0;
    r->shard = (int)json_num(json_at(cs, 0), 0); r->chunk = (int)json_num(json_at(ic, 0), 0);
    const char *dt = json_str(dtj, ""); r->dt = !strcmp(dt, "int8") ? 0 : !strcmp(dt, "float16") ? 1 : !strcmp(dt, "float32") ? 2 : -1; r->esize = r->dt == 0 ? 1 : r->dt == 1 ? 2 : 4;
    r->vscale = json_num(json_path(j, "attributes.value_scale"), 1.0);
    json_free(j);
    if (r->dt < 0 || r->chunk <= 0 || r->shard % r->chunk) return -1;
    const size_t cb = (size_t)r->chunk * r->chunk * r->chunk * (r->C ? r->C : 1) * r->esize; r->cb = malloc(cb);
    return r->cb ? 0 : -1;
}
void zr_close(zr *r) { free(r->idx); free(r->cb); }
/* the decoded chunk holding output cell (z, y, x), or nullptr when absent (zeros) */
const uint8_t *zr_chunk(zr *r, int64_t z, int64_t y, int64_t x) {
    const int64_t s[3] = {z / r->shard, y / r->shard, x / r->shard}; const int nc1 = r->shard / r->chunk, nc = nc1 * nc1 * nc1;
    const int64_t ci = ((z % r->shard) / r->chunk * nc1 + (y % r->shard) / r->chunk) * nc1 + (x % r->shard) / r->chunk;
    char p[1300];
    if (r->C) snprintf(p, sizeof p, "%s/c/%lld/%lld/%lld/0", r->dir, (long long)s[0], (long long)s[1], (long long)s[2]);
    else snprintf(p, sizeof p, "%s/c/%lld/%lld/%lld", r->dir, (long long)s[0], (long long)s[1], (long long)s[2]);
    if (memcmp(s, r->ish, sizeof s)) {   /* shard index */
        free(r->idx); r->idx = nullptr; memcpy(r->ish, s, sizeof s); r->csh = -1;
        FILE *f = fopen(p, "rb");
        if (f) { r->idx = malloc((size_t)nc * 16); fseek(f, -(long)(nc * 16 + 4), SEEK_END); if (fread(r->idx, 16, nc, f) != (size_t)nc) { free(r->idx); r->idx = nullptr; } fclose(f); }
    }
    if (!r->idx) return nullptr;
    uint64_t off, len; memcpy(&off, r->idx + ci * 16, 8); memcpy(&len, r->idx + ci * 16 + 8, 8);
    if (off == UINT64_MAX) return nullptr;
    if (r->csh == 0 && r->cci == ci && !memcmp(r->cshard, s, sizeof s)) return r->cb;
    FILE *f = fopen(p, "rb"); if (!f) return nullptr;
    void *zb = malloc(len); fseek(f, (long)off, SEEK_SET);
    const size_t got = fread(zb, 1, len, f); fclose(f);
    const size_t cb = (size_t)r->chunk * r->chunk * r->chunk * (r->C ? r->C : 1) * r->esize;
    if (got != len || ZSTD_decompress(r->cb, cb, zb, len) != cb) { free(zb); return nullptr; }
    free(zb); r->csh = 0; r->cci = ci; memcpy(r->cshard, s, sizeof s);
    return r->cb;
}
/* the region at o (output cells) chunk by chunk (each chunk decoded once): raw (int8 arrays only: the stored values, no value_scale) or
   float32 [n0][n1][n2](C); outside the array or absent: 0 */
static int zr_read_any(zr *r, const int64_t o[3], const int64_t n[3], float *out, int8_t *raw) {
    const int C = r->C ? r->C : 1, c = r->chunk;
    if (raw && r->dt != 0) return -1;
    const size_t es = raw ? 1 : 4;
    memset(raw ? (void *)raw : (void *)out, 0, (size_t)n[0] * n[1] * n[2] * C * es);
    int64_t c0[3], c1[3];
    for (int d = 0; d < 3; d++) {
        const int64_t lo = o[d] < 0 ? 0 : o[d], hi = o[d] + n[d] > r->shape[d] ? r->shape[d] : o[d] + n[d];
        if (hi <= lo) return 0;
        c0[d] = lo / c; c1[d] = (hi - 1) / c;
    }
    for (int64_t cz = c0[0]; cz <= c1[0]; cz++) for (int64_t cy = c0[1]; cy <= c1[1]; cy++) for (int64_t cx = c0[2]; cx <= c1[2]; cx++) {
        const uint8_t *ch = zr_chunk(r, cz * c, cy * c, cx * c);
        if (!ch) continue;
        int64_t a[3], b[3]; const int64_t cc[3] = {cz, cy, cx};
        for (int d = 0; d < 3; d++) {
            a[d] = cc[d] * c; b[d] = a[d] + c;
            if (a[d] < o[d]) a[d] = o[d];
            if (b[d] > o[d] + n[d]) b[d] = o[d] + n[d];
            if (b[d] > r->shape[d]) b[d] = r->shape[d];
        }
        for (int64_t gz = a[0]; gz < b[0]; gz++) for (int64_t gy = a[1]; gy < b[1]; gy++) {
            const size_t e0 = ((size_t)((gz % c) * c + gy % c) * c + a[2] % c) * C, w = (size_t)(b[2] - a[2]) * C;
            const size_t di = ((size_t)((gz - o[0]) * n[1] + (gy - o[1])) * n[2] + (size_t)(a[2] - o[2])) * C;
            if (raw) { memcpy(raw + di, ch + e0, w); continue; }
            float *dst = out + di;
            if (r->dt == 0) for (size_t k = 0; k < w; k++) dst[k] = (float)(((const int8_t *)ch)[e0 + k] * r->vscale);
            else if (r->dt == 1) for (size_t k = 0; k < w; k++) { uint16_t h; memcpy(&h, ch + (e0 + k) * 2, 2); dst[k] = (float)(h2f(h) * r->vscale); }
            else for (size_t k = 0; k < w; k++) { float v; memcpy(&v, ch + (e0 + k) * 4, 4); dst[k] = (float)(v * r->vscale); }
        }
    }
    return 0;
}
int zr_read(zr *r, const int64_t o[3], const int64_t n[3], float *out) { return zr_read_any(r, o, n, out, nullptr); }
int zr_read_i8(zr *r, const int64_t o[3], const int64_t n[3], int8_t *out) { return zr_read_any(r, o, n, nullptr, out); }
/* an output with its optional per-vector scale array */

int emb_open(emb *e, const char *group, const char *name) {
    char d[1200]; snprintf(d, sizeof d, "%s/%s", group, name);
    if (zr_open(&e->a, d)) { fprintf(stderr, "dinovol: cannot open %s\n", d); return -1; }
    snprintf(d, sizeof d, "%s/%s_scale", group, name); e->has_s = zr_open(&e->s, d) == 0;
    return 0;
}
void emb_close(emb *e) { zr_close(&e->a); if (e->has_s) zr_close(&e->s); }
int emb_read(emb *e, const int64_t o[3], const int64_t n[3], float *out) {
    if (zr_read(&e->a, o, n, out)) return -1;
    if (e->has_s) {
        const size_t nv = (size_t)n[0] * n[1] * n[2]; float *s = malloc(nv * 4);
        zr_read(&e->s, o, n, s);
        for (size_t i = 0; i < nv; i++) for (int k = 0; k < e->a.C; k++) out[i * e->a.C + k] *= s[i];
        free(s);
    }
    return 0;
}

/* ---- sharded zarr v3 writer of [Z][Y][X](C) arrays (zstd inner chunks, index at the end with crc32c), shards written incrementally:
   compressed chunks are appended as their cells finish and the index is written when the shard closes ---- */
static int zw_mkdirs(const char *path) {
    char p[1200]; snprintf(p, sizeof p, "%s", path);
    for (char *s = p + 1; *s; s++) if (*s == '/') { *s = 0; if (mkdir(p, 0755) && errno != EEXIST) return -1; *s = '/'; }
    return mkdir(p, 0755) && errno != EEXIST ? -1 : 0;
}
int zw_create(zw *w, const char *dir, const int64_t shape[3], int C, const char *dtype, int esize, int shard, int chunk, const char *attrs) {
    memset(w, 0, sizeof *w);
    { struct rlimit rl; if (!getrlimit(RLIMIT_NOFILE, &rl) && rl.rlim_cur < rl.rlim_max) { rl.rlim_cur = rl.rlim_max; setrlimit(RLIMIT_NOFILE, &rl); } }   /* (one open file per shard being written: a whole-scroll slab holds ~800, the default soft limit is 1024) */
    snprintf(w->dir, sizeof w->dir, "%s", dir); memcpy(w->shape, shape, sizeof w->shape); w->C = C; w->esize = esize; w->shard = shard; w->chunk = chunk;
    w->nc1 = shard / chunk; w->chunk_bytes = (size_t)chunk * chunk * chunk * (C ? C : 1) * esize;
    w->cbuf = malloc(w->chunk_bytes); w->zcap = ZSTD_compressBound(w->chunk_bytes); w->zbuf = malloc(w->zcap);
    if (zw_mkdirs(dir)) { fprintf(stderr, "dinovol: cannot create %s\n", dir); return -1; }
    char path[1200]; snprintf(path, sizeof path, "%s/zarr.json", dir);
    FILE *f = fopen(path, "w"); if (!f) return -1;
    char shp[160], csh[160], ich[160];
    if (C) { snprintf(shp, sizeof shp, "[%lld,%lld,%lld,%d]", (long long)shape[0], (long long)shape[1], (long long)shape[2], C);
             snprintf(csh, sizeof csh, "[%d,%d,%d,%d]", shard, shard, shard, C); snprintf(ich, sizeof ich, "[%d,%d,%d,%d]", chunk, chunk, chunk, C); }
    else { snprintf(shp, sizeof shp, "[%lld,%lld,%lld]", (long long)shape[0], (long long)shape[1], (long long)shape[2]);
           snprintf(csh, sizeof csh, "[%d,%d,%d]", shard, shard, shard); snprintf(ich, sizeof ich, "[%d,%d,%d]", chunk, chunk, chunk); }
    fprintf(f, "{\"zarr_format\":3,\"node_type\":\"array\",\"shape\":%s,\"data_type\":\"%s\",\"chunk_grid\":{\"name\":\"regular\",\"configuration\":{\"chunk_shape\":%s}},"
               "\"chunk_key_encoding\":{\"name\":\"default\",\"configuration\":{\"separator\":\"/\"}},\"fill_value\":0,"
               "\"codecs\":[{\"name\":\"sharding_indexed\",\"configuration\":{\"chunk_shape\":%s,\"codecs\":[{\"name\":\"bytes\",\"configuration\":{\"endian\":\"little\"}},"
               "{\"name\":\"zstd\",\"configuration\":{\"level\":1,\"checksum\":false}}],\"index_codecs\":[{\"name\":\"bytes\",\"configuration\":{\"endian\":\"little\"}},"
               "{\"name\":\"crc32c\"}],\"index_location\":\"end\"}}],\"dimension_names\":%s,\"attributes\":%s}\n",
            shp, dtype, csh, ich, C ? "[\"z\",\"y\",\"x\",\"c\"]" : "[\"z\",\"y\",\"x\"]", attrs ? attrs : "{}");
    return fclose(f) ? -1 : 0;
}
static int zw_close_shard(zw *w, zw_shard *s) {
    const int nc = w->nc1 * w->nc1 * w->nc1; uint8_t *idx = malloc((size_t)nc * 16 + 4);
    for (int i = 0; i < nc; i++) { memcpy(idx + (size_t)i * 16, &s->idx[2 * i], 8); memcpy(idx + (size_t)i * 16 + 8, &s->idx[2 * i + 1], 8); }
    const uint32_t crc = crc32c(0, idx, (size_t)nc * 16); memcpy(idx + (size_t)nc * 16, &crc, 4);
    int rc = fwrite(idx, 1, (size_t)nc * 16 + 4, s->f) != (size_t)nc * 16 + 4;
    rc |= fclose(s->f) != 0; free(idx); free(s->idx);
    char fin[1200]; snprintf(fin, sizeof fin, "%.*s", (int)(strlen(s->path) - 4), s->path);   /* drop ".tmp" */
    if (!rc && rename(s->path, fin)) rc = 1;
    if (rc) fprintf(stderr, "dinovol: cannot write %s\n", fin);
    return rc ? -1 : 0;
}
int zw_flush(zw *w) { int rc = 0; for (int i = 0; i < w->nopen; i++) rc |= zw_close_shard(w, &w->open[i]); w->nopen = 0; return rc; }
static zw_shard *zw_get(zw *w, int64_t sz, int64_t sy, int64_t sx) {
    for (int i = 0; i < w->nopen; i++) if (w->open[i].sz == sz && w->open[i].sy == sy && w->open[i].sx == sx) return &w->open[i];
    if (w->nopen == w->cap) { w->cap = w->cap ? 2 * w->cap : 16; w->open = realloc(w->open, w->cap * sizeof *w->open); }
    zw_shard *s = &w->open[w->nopen];
    char d[1200]; snprintf(d, sizeof d, "%s/c/%lld/%lld", w->dir, (long long)sz, (long long)sy);
    if (zw_mkdirs(d)) { fprintf(stderr, "dinovol: cannot create %s: %s\n", d, strerror(errno)); return nullptr; }
    if (w->C) { snprintf(d, sizeof d, "%s/c/%lld/%lld/%lld", w->dir, (long long)sz, (long long)sy, (long long)sx); if (zw_mkdirs(d)) { fprintf(stderr, "dinovol: cannot create %s: %s\n", d, strerror(errno)); return nullptr; }
                snprintf(s->path, sizeof s->path, "%s/0.tmp", d); }
    else snprintf(s->path, sizeof s->path, "%s/%lld.tmp", d, (long long)sx);
    s->f = fopen(s->path, "wb"); if (!s->f) { fprintf(stderr, "dinovol: cannot write %s: %s (%d shards open)\n", s->path, strerror(errno), w->nopen); return nullptr; }
    s->sz = sz; s->sy = sy; s->sx = sx; s->off = 0;
    const int nc = w->nc1 * w->nc1 * w->nc1; s->idx = malloc((size_t)nc * 16); memset(s->idx, 0xff, (size_t)nc * 16);
    w->nopen++;
    return s;
}
/* data [nz][ny][nx](C) at output cell (oz, oy, ox), chunk aligned, nz / ny / nx multiples of the chunk; chunks compressed by worker threads */
typedef struct { const zw *w; const uint8_t *data; int ny, nx, nch, *co; void **zb; size_t *zn; atomic_int next; } zwjob;
static void *zw_worker(void *a) {
    zwjob *J = a; const zw *w = J->w; const int c = w->chunk; const size_t vb = (size_t)(w->C ? w->C : 1) * w->esize;
    uint8_t *cb = malloc(w->chunk_bytes);
    for (int k; (k = atomic_fetch_add(&J->next, 1)) < J->nch;) {
        const int az = J->co[3 * k], ay = J->co[3 * k + 1], ax = J->co[3 * k + 2]; int any = 0;
        for (int z = 0; z < c; z++) for (int y = 0; y < c; y++) {
            const uint8_t *src = J->data + ((((size_t)(az + z) * J->ny) + ay + y) * J->nx + ax) * vb; uint8_t *dst = cb + (((size_t)z * c) + y) * c * vb;
            memcpy(dst, src, (size_t)c * vb);
            for (size_t i = 0; i < (size_t)c * vb && !any; i++) any = dst[i] != 0;
        }
        J->zn[k] = 0; J->zb[k] = nullptr;
        if (!any) continue;
        if (!(J->zb[k] = malloc(w->zcap))) { J->zn[k] = (size_t)-2; continue; }
        const size_t n = ZSTD_compress(J->zb[k], w->zcap, cb, w->chunk_bytes, 1);   /* level 1: int8 embeddings barely compress further at 3 */
        J->zn[k] = ZSTD_isError(n) ? (size_t)-1 : n;
    }
    free(cb); return nullptr;
}
int zw_nthreads = 8;
int zw_put(zw *w, int64_t oz, int64_t oy, int64_t ox, int nz, int ny, int nx, const void *data) {
    const int c = w->chunk; int nch = 0;
    int *co = malloc((size_t)(nz / c + 1) * (ny / c + 1) * (nx / c + 1) * 3 * sizeof *co);
    for (int az = 0; az < nz; az += c) for (int ay = 0; ay < ny; ay += c) for (int ax = 0; ax < nx; ax += c)
        if (oz + az < w->shape[0] && oy + ay < w->shape[1] && ox + ax < w->shape[2]) { co[3 * nch] = az; co[3 * nch + 1] = ay; co[3 * nch + 2] = ax; nch++; }
    zwjob J = {.w = w, .data = data, .ny = ny, .nx = nx, .nch = nch, .co = co, .zb = calloc(nch + 1, sizeof(void *)), .zn = calloc(nch + 1, sizeof(size_t))};
    atomic_init(&J.next, 0);
    const int nt = nch < zw_nthreads ? nch : zw_nthreads; pthread_t th[64];
    for (int i = 0; i < nt; i++) pthread_create(&th[i], nullptr, zw_worker, &J);
    for (int i = 0; i < nt; i++) pthread_join(th[i], nullptr);
    int rc = 0;
    for (int k = 0; k < nch && !rc; k++) {
        const int64_t gz = oz + co[3 * k], gy = oy + co[3 * k + 1], gx = ox + co[3 * k + 2];
        const size_t n = J.zn[k];
        if (n >= (size_t)-2) { fprintf(stderr, "dinovol: chunk %lld,%lld,%lld: %s\n", (long long)gz, (long long)gy, (long long)gx, n == (size_t)-2 ? "out of memory" : "compression failed"); rc = -1; break; }
        if (!J.zb[k]) continue;   /* (an all-zero chunk: not stored) */
        zw_shard *s = zw_get(w, gz / w->shard, gy / w->shard, gx / w->shard);
        if (!s) { rc = -1; break; }
        if (fwrite(J.zb[k], 1, n, s->f) != n) { fprintf(stderr, "dinovol: chunk write to %s failed: %s\n", s->path, strerror(errno)); rc = -1; break; }
        const int ci = (int)((((gz % w->shard) / c) * w->nc1 + (gy % w->shard) / c) * w->nc1 + (gx % w->shard) / c);
        s->idx[2 * ci] = s->off; s->idx[2 * ci + 1] = n; s->off += n;
    }
    for (int k = 0; k < nch; k++) free(J.zb[k]);
    free(J.zb); free(J.zn); free(co);
    return rc;
}
void zw_free(zw *w) { free(w->open); free(w->cbuf); free(w->zbuf); }
