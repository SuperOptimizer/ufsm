#include "zarr3.h"
#include "json.h"
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zstd.h>
#define VOLCOMP_IMPLEMENTATION
#include "volcomp.h"

static _Thread_local char errbuf[256];
const char *z3_error(void) { return errbuf; }
static int fail(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(errbuf, sizeof errbuf, fmt, ap);
    va_end(ap);
    return -1;
}
#define FAIL(...) fail(__VA_ARGS__)

struct z3 {
    store *s;
    char *key;
    char *cache;   /* nullptr = no cache */
    z3_meta m;
    int nc;        /* inner chunks per shard */
    int cgrid[3];  /* inner chunk grid within a shard */
};

/* ---------- metadata ---------- */

static int parse_codecs(const json *codecs, z3_meta *m) {
    m->q = -1;
    for (size_t i = 0; i < (codecs ? codecs->n : 0); i++) {
        const json *c = json_at(codecs, i);
        const char *name = json_str(json_get(c, "name"), "");
        if (!strcmp(name, "volcomp")) m->q = (float)json_num(json_path(c, "configuration.q"), 8.0);
        else if (!strcmp(name, "zstd")) m->zstd = 1;
        else if (!strcmp(name, "bytes")) m->raw = 1;
        else if (!strcmp(name, "transpose")) return FAIL("unsupported codec %s", name);
        else return FAIL("unsupported codec %s", name);
    }
    if (m->q >= 0) m->raw = 0;
    return 0;
}

static void group_scale(store *s, const char *key, z3_meta *m) {
    const char *slash = strrchr(key, '/');
    if (!slash) return;
    char gk[1024];
    snprintf(gk, sizeof gk, "%.*s/zarr.json", (int)(slash - key), key);
    size_t n;
    uint8_t *txt = store_read_all(s, gk, &n);
    if (!txt) return;
    json *j = json_parse((char *)txt, n);
    free(txt);
    const json *ds = json_path(j, "attributes.ome.multiscales.0.datasets");
    if (!ds) ds = json_path(j, "attributes.multiscales.0.datasets");
    for (size_t i = 0; ds && i < ds->n; i++) {
        const json *d = json_at(ds, i);
        if (!strcmp(json_str(json_get(d, "path"), ""), slash + 1)) {
            const json *ct = json_get(d, "coordinateTransformations");
            for (size_t k = 0; ct && k < ct->n; k++)
                if (!strcmp(json_str(json_get(json_at(ct, k), "type"), ""), "scale"))
                    m->scale_um = json_num(json_path(json_at(ct, k), "scale.0"), 0);
        }
    }
    json_free(j);
}

z3 *z3_open(store *s, const char *key, const char *cache_dir) {
    char mk[1024];
    snprintf(mk, sizeof mk, "%s/zarr.json", key);
    size_t n;
    uint8_t *txt = store_read_all(s, mk, &n);
    if (!txt) { FAIL("cannot read %s", mk); return nullptr; }
    json *j = json_parse((char *)txt, n);
    free(txt);
    if (!j) { FAIL("bad json in %s", mk); return nullptr; }
    z3 *z = calloc(1, sizeof *z);
    z->s = s;
    z->key = strdup(key);
    z->cache = cache_dir && !store_is_local(s) ? strdup(cache_dir) : nullptr;
    z3_meta *m = &z->m;
    m->sep = '/';
    const json *shape = json_get(j, "shape");
    if (!shape || shape->n != 3 || strcmp(json_str(json_get(j, "data_type"), ""), "uint8")) {
        FAIL("%s: need a 3-D uint8 array", key); goto bad;
    }
    for (int i = 0; i < 3; i++) m->shape[i] = (int64_t)json_num(json_at(shape, (size_t)i), 0);
    m->fill = (int)json_num(json_get(j, "fill_value"), 0);
    const json *cs = json_path(j, "chunk_grid.configuration.chunk_shape");
    if (!cs || cs->n != 3) { FAIL("%s: no chunk_shape", key); goto bad; }
    for (int i = 0; i < 3; i++) m->shard[i] = m->chunk[i] = (int)json_num(json_at(cs, (size_t)i), 0);
    const char *sep = json_str(json_path(j, "chunk_key_encoding.configuration.separator"), "/");
    m->sep = sep[0];
    const json *codecs = json_get(j, "codecs");
    const json *c0 = json_at(codecs, 0);
    if (c0 && !strcmp(json_str(json_get(c0, "name"), ""), "sharding_indexed")) {
        m->sharded = 1;
        const json *ics = json_path(c0, "configuration.chunk_shape");
        if (!ics || ics->n != 3) { FAIL("%s: no inner chunk_shape", key); goto bad; }
        for (int i = 0; i < 3; i++) m->chunk[i] = (int)json_num(json_at(ics, (size_t)i), 0);
        if (strcmp(json_str(json_path(c0, "configuration.index_location"), "end"), "end")) {
            FAIL("%s: index_location must be end", key); goto bad;
        }
        if (parse_codecs(json_path(c0, "configuration.codecs"), m)) goto bad;
    } else if (parse_codecs(codecs, m)) goto bad;
    if (m->q >= 0 && (m->chunk[0] != 128 || m->chunk[1] != 128 || m->chunk[2] != 128)) {
        FAIL("%s: volcomp needs 128^3 inner chunks", key); goto bad;
    }
    for (int i = 0; i < 3; i++) {
        if (m->chunk[i] <= 0 || m->shard[i] % m->chunk[i]) { FAIL("%s: shard not a multiple of chunk", key); goto bad; }
        z->cgrid[i] = m->shard[i] / m->chunk[i];
    }
    z->nc = z->cgrid[0] * z->cgrid[1] * z->cgrid[2];
    json_free(j);
    group_scale(s, key, m);
    return z;
bad:
    json_free(j);
    z3_close(z);
    return nullptr;
}

void z3_close(z3 *z) {
    if (!z) return;
    free(z->key);
    free(z->cache);
    free(z);
}

const z3_meta *z3_meta_of(const z3 *z) { return &z->m; }
const char *z3_key(const z3 *z) { return z->key; }

int z3_group_levels(store *s, const char *group_key, z3_level *out, int max) {
    char gk[1024];
    snprintf(gk, sizeof gk, "%s/zarr.json", group_key);
    size_t n;
    uint8_t *txt = store_read_all(s, gk, &n);
    if (!txt) return FAIL("cannot read %s", gk);
    json *j = json_parse((char *)txt, n);
    free(txt);
    const json *ds = json_path(j, "attributes.ome.multiscales.0.datasets");
    if (!ds) ds = json_path(j, "attributes.multiscales.0.datasets");
    int k = 0;
    for (size_t i = 0; ds && i < ds->n && k < max; i++) {
        const json *d = json_at(ds, i);
        snprintf(out[k].path, sizeof out[k].path, "%s", json_str(json_get(d, "path"), ""));
        out[k].um = json_num(json_path(d, "coordinateTransformations.0.scale.0"), 0);
        k++;
    }
    json_free(j);
    return k;
}

/* ---------- object naming and cache ---------- */

static void shard_key(const z3 *z, int64_t sz, int64_t sy, int64_t sx, char *buf, size_t n) {
    char sep = z->m.sep;
    snprintf(buf, n, "%s/c%c%lld%c%lld%c%lld", z->key, sep, (long long)sz, sep, (long long)sy, sep, (long long)sx);
}

static int mkdirs(const char *path) {
    char tmp[1024];
    snprintf(tmp, sizeof tmp, "%s", path);
    for (char *p = tmp + 1; *p; p++)
        if (*p == '/') { *p = 0; if (mkdir(tmp, 0755) && errno != EEXIST) return -1; *p = '/'; }
    return mkdir(tmp, 0755) && errno != EEXIST ? -1 : 0;
}

/* cache file for (shard, item): item = "idx", "missing" or the inner chunk number */
static void cache_path(const z3 *z, const char *skey, const char *item, char *buf, size_t n) {
    snprintf(buf, n, "%s/%s.%s", z->cache, skey, item);
}

static uint8_t *cache_get(const z3 *z, const char *skey, const char *item, size_t *len) {
    if (!z->cache) return nullptr;
    char p[1400];
    cache_path(z, skey, item, p, sizeof p);
    int fd = open(p, O_RDONLY);
    if (fd < 0) return nullptr;
    struct stat st;
    fstat(fd, &st);
    uint8_t *b = malloc((size_t)st.st_size + 1);
    size_t got = 0;
    while (b && got < (size_t)st.st_size) {
        ssize_t r = read(fd, b + got, (size_t)st.st_size - got);
        if (r <= 0) { free(b); b = nullptr; break; }
        got += (size_t)r;
    }
    close(fd);
    if (b) *len = got;
    return b;
}

static void cache_put(const z3 *z, const char *skey, const char *item, const uint8_t *b, size_t len) {
    if (!z->cache) return;
    char p[1400], tmp[1500];
    cache_path(z, skey, item, p, sizeof p);
    char *slash = strrchr(p, '/');
    if (slash) { *slash = 0; mkdirs(p); *slash = '/'; }
    snprintf(tmp, sizeof tmp, "%s.tmp%ld", p, (long)pthread_self());
    int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return;
    size_t off = 0;
    while (off < len) {
        ssize_t w = write(fd, b + off, len - off);
        if (w <= 0) { close(fd); unlink(tmp); return; }
        off += (size_t)w;
    }
    close(fd);
    rename(tmp, p);
}

/* ---------- shard index ---------- */

typedef struct { uint64_t off, len; } ientry;

/* Returns 1 with index filled, 0 if the shard is missing, -1 on error. */
static int load_index(const z3 *z, const char *skey, ientry *idx) {
    size_t ilen = (size_t)z->nc * 16 + 4;
    size_t n;
    uint8_t *b = cache_get(z, skey, "idx", &n);
    if (!b && cache_get(z, skey, "missing", &n)) return 0;
    if (!b) {
        int64_t size = store_size(z->s, skey);
        if (size == -2) { cache_put(z, skey, "missing", (const uint8_t *)"", 0); return 0; }
        if (size < 0) return FAIL("size of %s", skey);
        if ((size_t)size < ilen) return FAIL("%s: shorter than its index", skey);
        b = malloc(ilen);
        if (store_read(z->s, skey, size - (int64_t)ilen, (int64_t)ilen, b) != (int64_t)ilen) {
            free(b);
            return FAIL("read index of %s", skey);
        }
        n = ilen;
        cache_put(z, skey, "idx", b, n);
    }
    if (n != ilen) { free(b); return FAIL("%s: bad index size", skey); }
    for (int i = 0; i < z->nc; i++) {
        memcpy(&idx[i].off, b + (size_t)i * 16, 8);
        memcpy(&idx[i].len, b + (size_t)i * 16 + 8, 8);
    }
    free(b);
    return 1;
}

/* ---------- read ---------- */

typedef struct {
    int64_t sz, sy, sx;      /* shard index */
    int ci;                  /* inner chunk index within shard (C order), or -1 for unsharded */
    int cz, cy, cx;          /* inner chunk coords within shard */
    uint64_t off, len;       /* byte range inside the shard object */
    int present;
} work;

typedef struct {
    z3 *z;
    const int64_t *o, *n;
    uint8_t *out;
    work *items;
    int nitems;
    atomic_int next;
    atomic_int failed;
} job;

static void copy_chunk(const job *jb, const work *w, const uint8_t *chunk) {
    const z3_meta *m = &jb->z->m;
    int64_t c0[3] = {w->sz * m->shard[0] + (int64_t)w->cz * m->chunk[0],
                     w->sy * m->shard[1] + (int64_t)w->cy * m->chunk[1],
                     w->sx * m->shard[2] + (int64_t)w->cx * m->chunk[2]};
    int64_t lo[3], hi[3];
    for (int i = 0; i < 3; i++) {
        lo[i] = jb->o[i] > c0[i] ? jb->o[i] : c0[i];
        int64_t ce = c0[i] + m->chunk[i], oe = jb->o[i] + jb->n[i];
        hi[i] = ce < oe ? ce : oe;
        if (hi[i] <= lo[i]) return;
    }
    size_t row = (size_t)(hi[2] - lo[2]);
    for (int64_t zz = lo[0]; zz < hi[0]; zz++)
        for (int64_t yy = lo[1]; yy < hi[1]; yy++) {
            const uint8_t *src = chunk + ((zz - c0[0]) * m->chunk[1] + (yy - c0[1])) * m->chunk[2] + (lo[2] - c0[2]);
            uint8_t *dst = jb->out + ((zz - jb->o[0]) * jb->n[1] + (yy - jb->o[1])) * jb->n[2] + (lo[2] - jb->o[2]);
            memcpy(dst, src, row);
        }
}

static int decode_into(const z3 *z, const uint8_t *enc, size_t n, uint8_t *dec, ZSTD_DCtx *dctx, uint8_t *scratch) {
    const z3_meta *m = &z->m;
    size_t cv = (size_t)m->chunk[0] * m->chunk[1] * m->chunk[2];
    if (m->zstd) {
        size_t r = ZSTD_decompressDCtx(dctx, scratch, VOLCOMP_ENCODE_BOUND > cv ? VOLCOMP_ENCODE_BOUND : cv, enc, n);
        if (ZSTD_isError(r)) return FAIL("zstd: %s", ZSTD_getErrorName(r));
        enc = scratch;
        n = r;
    }
    if (m->q >= 0) {
        volcomp_status st = volcomp_decode(enc, n, dec, cv);
        if (st != VOLCOMP_OK) return FAIL("volcomp: %s", volcomp_status_string(st));
        return 0;
    }
    if (n != cv) return FAIL("raw chunk size %zu != %zu", n, cv);
    memcpy(dec, enc, cv);
    return 0;
}

static void *worker(void *arg) {
    job *jb = arg;
    z3 *z = jb->z;
    size_t cv = (size_t)z->m.chunk[0] * z->m.chunk[1] * z->m.chunk[2];
    uint8_t *dec = malloc(cv);
    uint8_t *scratch = malloc(VOLCOMP_ENCODE_BOUND > cv ? VOLCOMP_ENCODE_BOUND : cv);
    ZSTD_DCtx *dctx = ZSTD_createDCtx();
    for (;;) {
        int i = atomic_fetch_add(&jb->next, 1);
        if (i >= jb->nitems || atomic_load(&jb->failed)) break;
        work *w = &jb->items[i];
        if (!w->present) continue;
        char skey[1200], item[16];
        shard_key(z, w->sz, w->sy, w->sx, skey, sizeof skey);
        snprintf(item, sizeof item, "%d", w->ci < 0 ? 0 : w->ci);
        size_t n;
        uint8_t *b = cache_get(z, skey, item, &n);
        if (!b) {
            if (w->ci < 0) {
                b = store_read_all(z->s, skey, &n);
                if (!b) continue; /* missing chunk object: fill */
            } else {
                n = (size_t)w->len;
                b = malloc(n);
                if (store_read(z->s, skey, (int64_t)w->off, (int64_t)n, b) != (int64_t)n) {
                    free(b);
                    FAIL("read chunk %d of %s", w->ci, skey);
                    atomic_store(&jb->failed, 1);
                    break;
                }
            }
            cache_put(z, skey, item, b, n);
        }
        int rc = decode_into(z, b, n, dec, dctx, scratch);
        free(b);
        if (rc) { atomic_store(&jb->failed, 1); break; }
        copy_chunk(jb, w, dec);
    }
    ZSTD_freeDCtx(dctx);
    free(dec);
    free(scratch);
    return nullptr;
}

int z3_shard_present(z3 *z, int64_t sz, int64_t sy, int64_t sx) {
    char skey[1200];
    shard_key(z, sz, sy, sx, skey, sizeof skey);
    if (z->m.sharded) {
        ientry *idx = malloc((size_t)z->nc * sizeof *idx);
        int r = load_index(z, skey, idx);
        free(idx);
        return r;
    }
    int64_t s = store_size(z->s, skey);
    return s == -2 ? 0 : s < 0 ? -1 : 1;
}

int z3_read(z3 *z, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads) {
    const z3_meta *m = &z->m;
    memset(out, m->fill, (size_t)n[0] * n[1] * n[2]);
    int64_t lo[3], hi[3];
    for (int i = 0; i < 3; i++) {
        lo[i] = o[i] < 0 ? 0 : o[i];
        hi[i] = o[i] + n[i] > m->shape[i] ? m->shape[i] : o[i] + n[i];
        if (hi[i] <= lo[i]) return 0;
    }
    int64_t s0[3], s1[3];
    for (int i = 0; i < 3; i++) { s0[i] = lo[i] / m->shard[i]; s1[i] = (hi[i] - 1) / m->shard[i]; }
    int cap = 64, nitems = 0;
    work *items = malloc((size_t)cap * sizeof *items);
    ientry *idx = malloc((size_t)z->nc * sizeof *idx);
    int rc = 0;
    for (int64_t sz = s0[0]; sz <= s1[0] && !rc; sz++)
        for (int64_t sy = s0[1]; sy <= s1[1] && !rc; sy++)
            for (int64_t sx = s0[2]; sx <= s1[2] && !rc; sx++) {
                char skey[1200];
                shard_key(z, sz, sy, sx, skey, sizeof skey);
                int present = 1;
                if (m->sharded) {
                    present = load_index(z, skey, idx);
                    if (present < 0) { rc = -1; break; }
                    if (!present) continue;
                }
                /* inner chunk range within this shard that intersects the box */
                int c0[3], c1[3];
                for (int i = 0; i < 3; i++) {
                    int64_t sb = (i == 0 ? sz : i == 1 ? sy : sx) * m->shard[i];
                    int64_t a = lo[i] > sb ? lo[i] - sb : 0;
                    int64_t b = (hi[i] < sb + m->shard[i] ? hi[i] : sb + m->shard[i]) - sb - 1;
                    c0[i] = (int)(a / m->chunk[i]);
                    c1[i] = (int)(b / m->chunk[i]);
                }
                for (int cz = c0[0]; cz <= c1[0]; cz++)
                    for (int cy = c0[1]; cy <= c1[1]; cy++)
                        for (int cx = c0[2]; cx <= c1[2]; cx++) {
                            if (nitems == cap) { cap *= 2; items = realloc(items, (size_t)cap * sizeof *items); }
                            work *w = &items[nitems++];
                            *w = (work){sz, sy, sx, -1, cz, cy, cx, 0, 0, 1};
                            if (m->sharded) {
                                w->ci = (cz * z->cgrid[1] + cy) * z->cgrid[2] + cx;
                                w->off = idx[w->ci].off;
                                w->len = idx[w->ci].len;
                                w->present = !(w->off == UINT64_MAX && w->len == UINT64_MAX) && w->len;
                            }
                        }
            }
    free(idx);
    if (rc) { free(items); return -1; }
    job jb = {z, o, n, out, items, nitems, 0, 0};
    if (nthreads <= 0) nthreads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (nthreads > nitems) nthreads = nitems;
    if (nthreads < 1) nthreads = 1;
    pthread_t th[256];
    if (nthreads > 256) nthreads = 256;
    for (int i = 1; i < nthreads; i++) pthread_create(&th[i], nullptr, worker, &jb);
    worker(&jb);
    for (int i = 1; i < nthreads; i++) pthread_join(th[i], nullptr);
    free(items);
    return atomic_load(&jb.failed) ? -1 : 0;
}
