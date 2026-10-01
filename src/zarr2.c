#include "zarr2.h"
#include "json.h"
#include <blosc.h>
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
#include <zlib.h>
#include <zstd.h>

static _Thread_local char errbuf[256];
const char *z2_error(void) { return errbuf; }
static int fail(const char *fmt, ...) { va_list ap; va_start(ap, fmt); vsnprintf(errbuf, sizeof errbuf, fmt, ap); va_end(ap); return -1; }

struct z2 {
    store *s;
    char *key, *cache;
    z2_meta m;
    int assume;
};
void z2_assume_present(z2 *z, int on) { z->assume = on; }

z2 *z2_open(store *s, const char *key, const char *cache_dir) {
    char mk[1200];
    snprintf(mk, sizeof mk, "%s/.zarray", key);
    size_t n;
    uint8_t *txt = store_read_all(s, mk, &n);
    if (!txt) { fail("cannot read %s", mk); return nullptr; }
    json *j = json_parse((char *)txt, n);
    free(txt);
    if (!j) { fail("bad json in %s", mk); return nullptr; }
    z2 *z = calloc(1, sizeof *z);
    z->s = s;
    z->key = strdup(key);
    z->cache = cache_dir && !store_is_local(s) ? strdup(cache_dir) : nullptr;
    const json *shape = json_get(j, "shape"), *chunks = json_get(j, "chunks");
    const char *dt = json_str(json_get(j, "dtype"), "");
    if (!shape || shape->n != 3 || !chunks || chunks->n != 3 || strcmp(dt, "|u1")) { fail("%s: need 3-D |u1", key); json_free(j); z2_close(z); return nullptr; }
    for (int i = 0; i < 3; i++) { z->m.shape[i] = (int64_t)json_num(json_at(shape, (size_t)i), 0); z->m.chunk[i] = (int)json_num(json_at(chunks, (size_t)i), 0); }
    z->m.sep = json_str(json_get(j, "dimension_separator"), ".")[0];
    z->m.fill = (int)json_num(json_get(j, "fill_value"), 0);
    const json *comp = json_get(j, "compressor");
    const char *id = json_str(json_get(comp, "id"), "");
    if (!comp || comp->type == J_NULL) z->m.comp = 0;
    else if (!strcmp(id, "blosc")) z->m.comp = 1;
    else if (!strcmp(id, "zstd")) z->m.comp = 2;
    else if (!strcmp(id, "zlib")) z->m.comp = 3;
    else if (!strcmp(id, "gzip")) z->m.comp = 4;
    else { fail("%s: unsupported compressor %s", key, id); json_free(j); z2_close(z); return nullptr; }
    const json *filters = json_get(j, "filters");
    if (filters && filters->type == J_ARR && filters->n) { fail("%s: filters unsupported", key); json_free(j); z2_close(z); return nullptr; }
    json_free(j);
    return z;
}

void z2_close(z2 *z) { if (z) { free(z->key); free(z->cache); free(z); } }
const z2_meta *z2_meta_of(const z2 *z) { return &z->m; }

static void chunk_key(const z2 *z, int64_t cz, int64_t cy, int64_t cx, char *buf, size_t n) {
    snprintf(buf, n, "%s/%lld%c%lld%c%lld", z->key, (long long)cz, z->m.sep, (long long)cy, z->m.sep, (long long)cx);
}

static int mkdirs(const char *path) {
    char tmp[1400];
    snprintf(tmp, sizeof tmp, "%s", path);
    for (char *p = tmp + 1; *p; p++) if (*p == '/') { *p = 0; if (mkdir(tmp, 0755) && errno != EEXIST) return -1; *p = '/'; }
    return mkdir(tmp, 0755) && errno != EEXIST ? -1 : 0;
}

/* returns malloc'd bytes, *len; nullptr with *len = 0 if missing; nullptr with *len = 1 on error */
static uint8_t *fetch(const z2 *z, const char *ck, size_t *len) {
    char cp[1500];
    if (z->cache) {
        snprintf(cp, sizeof cp, "%s/%s", z->cache, ck);
        int fd = open(cp, O_RDONLY);
        if (fd >= 0) {
            struct stat st; fstat(fd, &st);
            uint8_t *b = malloc((size_t)st.st_size + 1); size_t got = 0;
            while (got < (size_t)st.st_size) { ssize_t r = read(fd, b + got, (size_t)st.st_size - got); if (r <= 0) break; got += (size_t)r; }
            close(fd);
            *len = got;
            if (got == 0) { free(b); return nullptr; }   /* empty file = negative cache */
            return b;
        }
    }
    int64_t sz = z->assume ? 1 : store_size(z->s, ck);
    uint8_t *b = nullptr;
    if (sz == -2) { *len = 0; }
    else if (sz < 0) { fail("HEAD %s: http %ld", ck, store_last_status()); *len = 1; return nullptr; }
    else { b = store_read_all(z->s, ck, len); if (!b) { if (store_last_status() == 404) { *len = 0; } else { fail("GET %s: http %ld", ck, store_last_status()); *len = 1; return nullptr; } } }
    if (z->cache) {
        char *slash = strrchr(cp, '/');
        if (slash) { *slash = 0; mkdirs(cp); *slash = '/'; }
        char tmp[1600]; snprintf(tmp, sizeof tmp, "%s.tmp%ld", cp, (long)pthread_self());
        FILE *f = fopen(tmp, "wb");
        if (f) { if (b) fwrite(b, 1, *len, f); fclose(f); rename(tmp, cp); }
    }
    return b;
}

int z2_comp_of(const char *id) {
    if (!id || !*id) return 0;
    if (!strcmp(id, "blosc")) return 1;
    if (!strcmp(id, "zstd")) return 2;
    if (!strcmp(id, "zlib")) return 3;
    if (!strcmp(id, "gzip")) return 4;
    return -1;
}

int z2_decode(int comp, const uint8_t *in, size_t n, uint8_t *out, size_t cv) {
    switch (comp) {
    case 0: if (n != cv) return fail("raw chunk size %zu != %zu", n, cv); memcpy(out, in, cv); return 0;
    case 1: { int r = blosc_decompress_ctx(in, out, cv, 1); if (r != (int)cv) return fail("blosc decode %d", r); return 0; }
    case 2: { size_t r = ZSTD_decompress(out, cv, in, n); if (ZSTD_isError(r) || r != cv) return fail("zstd decode"); return 0; }
    case 3: case 4: {
        z_stream st = {0}; st.next_in = (Bytef *)in; st.avail_in = (uInt)n; st.next_out = out; st.avail_out = (uInt)cv;
        if (inflateInit2(&st, comp == 4 ? 16 + MAX_WBITS : MAX_WBITS) != Z_OK) return fail("inflateInit");
        int r = inflate(&st, Z_FINISH); inflateEnd(&st);
        if (r != Z_STREAM_END || st.total_out != cv) return fail("inflate %d", r);
        return 0;
    }
    }
    return fail("bad comp");
}

typedef struct { int64_t cz, cy, cx; } work;
typedef struct { z2 *z; const int64_t *o, *n; uint8_t *out; work *items; int nitems; atomic_int next, failed; } job;

static void *worker(void *arg) {
    job *jb = arg; z2 *z = jb->z; const z2_meta *m = &z->m;
    size_t cv = (size_t)m->chunk[0] * m->chunk[1] * m->chunk[2];
    uint8_t *dec = malloc(cv);
    for (;;) {
        int i = atomic_fetch_add(&jb->next, 1);
        if (i >= jb->nitems || atomic_load(&jb->failed)) break;
        work *w = &jb->items[i];
        char ck[1300]; chunk_key(z, w->cz, w->cy, w->cx, ck, sizeof ck);
        size_t n; uint8_t *b = fetch(z, ck, &n);
        if (!b) { if (n == 1) { atomic_store(&jb->failed, 1); break; } continue; }
        int rc = z2_decode(z->m.comp, b, n, dec, cv); free(b);
        if (rc) { atomic_store(&jb->failed, 1); break; }
        int64_t c0[3] = {w->cz * m->chunk[0], w->cy * m->chunk[1], w->cx * m->chunk[2]}, lo[3], hi[3];
        int skip = 0;
        for (int d = 0; d < 3; d++) {
            lo[d] = jb->o[d] > c0[d] ? jb->o[d] : c0[d];
            int64_t ce = c0[d] + m->chunk[d], oe = jb->o[d] + jb->n[d];
            hi[d] = ce < oe ? ce : oe;
            if (hi[d] <= lo[d]) skip = 1;
        }
        if (skip) continue;
        size_t row = (size_t)(hi[2] - lo[2]);
        for (int64_t zz = lo[0]; zz < hi[0]; zz++) for (int64_t yy = lo[1]; yy < hi[1]; yy++)
            memcpy(jb->out + ((zz - jb->o[0]) * jb->n[1] + (yy - jb->o[1])) * jb->n[2] + (lo[2] - jb->o[2]),
                   dec + ((zz - c0[0]) * m->chunk[1] + (yy - c0[1])) * m->chunk[2] + (lo[2] - c0[2]), row);
    }
    free(dec);
    return nullptr;
}

int z2_read(z2 *z, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads) {
    const z2_meta *m = &z->m;
    memset(out, m->fill, (size_t)n[0] * n[1] * n[2]);
    int64_t lo[3], hi[3], c0[3], c1[3];
    for (int i = 0; i < 3; i++) {
        lo[i] = o[i] < 0 ? 0 : o[i];
        hi[i] = o[i] + n[i] > m->shape[i] ? m->shape[i] : o[i] + n[i];
        if (hi[i] <= lo[i]) return 0;
        c0[i] = lo[i] / m->chunk[i]; c1[i] = (hi[i] - 1) / m->chunk[i];
    }
    int nitems = (int)((c1[0] - c0[0] + 1) * (c1[1] - c0[1] + 1) * (c1[2] - c0[2] + 1)), k = 0;
    work *items = malloc((size_t)nitems * sizeof *items);
    for (int64_t cz = c0[0]; cz <= c1[0]; cz++) for (int64_t cy = c0[1]; cy <= c1[1]; cy++) for (int64_t cx = c0[2]; cx <= c1[2]; cx++) items[k++] = (work){cz, cy, cx};
    job jb = {z, o, n, out, items, nitems, 0, 0};
    if (nthreads <= 0) nthreads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (nthreads > nitems) nthreads = nitems;
    if (nthreads > 256) nthreads = 256;
    if (nthreads < 1) nthreads = 1;
    pthread_t th[256];
    for (int i = 1; i < nthreads; i++) pthread_create(&th[i], nullptr, worker, &jb);
    worker(&jb);
    for (int i = 1; i < nthreads; i++) pthread_join(th[i], nullptr);
    free(items);
    return atomic_load(&jb.failed) ? -1 : 0;
}

int z2_chunk_present(z2 *z, int64_t cz, int64_t cy, int64_t cx) {
    char ck[1300]; chunk_key(z, cz, cy, cx, ck, sizeof ck);
    int64_t s = store_size(z->s, ck);
    return s == -2 ? 0 : s < 0 ? -1 : 1;
}
