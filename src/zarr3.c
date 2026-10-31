#include "zarr3.h"
#include "sysinfo.h"
#include "json.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
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
#include "volcomp_fast.h"
#include "zshm.h"
#include "gpudec.h"
#define XXH_INLINE_ALL
#include <xxhash.h>

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

/* Defaults are enabled; either path can be disabled for comparisons. */
static int option_on(const char *name) { const char *v = getenv(name); return !v || (*v && strcmp(v, "0")); }

typedef struct { uint64_t off, len; } ientry_t;
struct z3 {
    store *s;
    char *key;
    char *cache;   /* nullptr = no cache */
    z3_meta m;
    int nc;        /* inner chunks per shard */
    int cgrid[3];  /* inner chunk grid within a shard */
    uint64_t id;   /* process-unique: keys the decoded-mask cache */
    /* shard indices in memory (key -> nc entries, or absent): an HTTP shard index is fetched once per process */
    pthread_mutex_t imu;
    struct ient_slot { char *key; ientry_t *idx; int present; } *itab;
    size_t icap, icount;
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

static z3 *z3_open_impl(store *s, const char *key, const char *cache_dir, int typed) {
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
    { static atomic_uint_fast64_t next_id = 1; z->id = atomic_fetch_add(&next_id, 1); }
    z->cache = cache_dir && !store_is_local(s) ? strdup(cache_dir) : nullptr;
    pthread_mutex_init(&z->imu, nullptr);
    z3_meta *m = &z->m;
    m->sep = '/';
    const json *shape = json_get(j, "shape");
    static const char *dtn[6] = {"uint8", "uint16", "int16", "uint32", "int32", "float32"};
    const char *dts = json_str(json_get(j, "data_type"), "");
    m->dtype = -1; for (int k = 0; k < 6; k++) if (!strcmp(dts, dtn[k])) m->dtype = k;
    if (!shape || shape->n != 3 || m->dtype < 0 || (m->dtype && !typed)) { FAIL(typed ? "%s: need a 3-D uint8 / uint16 / int16 / uint32 / int32 / float32 array" : "%s: need a 3-D uint8 array", key); goto bad; }
    m->esize = m->dtype == 0 ? 1 : m->dtype <= 2 ? 2 : 4;
    for (int i = 0; i < 3; i++) m->shape[i] = (int64_t)json_num(json_at(shape, (size_t)i), 0);
    m->fill = (int)json_num(json_get(j, "fill_value"), 0); m->fillv = json_num(json_get(j, "fill_value"), 0);
    m->label_binary = !strcmp(json_str(json_path(j, "attributes.ufsm.encoding"), ""), "binary");
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

z3 *z3_open(store *s, const char *key, const char *cache_dir) { return z3_open_impl(s, key, cache_dir, 0); }
z3 *z3_open_typed(store *s, const char *key, const char *cache_dir) { return z3_open_impl(s, key, cache_dir, 1); }

void z3_close(z3 *z) {
    if (!z) return;
    free(z->key);
    free(z->cache);
    for (size_t i = 0; i < z->icap; i++) { free(z->itab[i].key); free(z->itab[i].idx); }
    free(z->itab);
    pthread_mutex_destroy(&z->imu);
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

/* io statistics: chunk reads served from the cache vs fetched from the store (atomic, process-wide) */
static atomic_uint_fast64_t g_z3_cache_hits, g_z3_store_reads;
void z3_shared_cache_stats(uint64_t *hits, uint64_t *misses, uint64_t *stored) { zshm_stats(hits, misses, stored); }

/* src/nn_gpudec.cu (UFSM_GPU_DECODE=1); weak stubs keep CPU-only binaries linking */
__attribute__((weak)) int gpudec_enabled(void) { return 0; }
__attribute__((weak)) int gpudec_threads(int nthreads) { return nthreads; }
__attribute__((weak)) int gpudec_decode(const gd_chunk *c, uint32_t mask, uint8_t *dst) { (void)c; (void)mask; (void)dst; return -1; }
__attribute__((weak)) int gpudec_decode_dev(const gd_chunk *c, uint32_t mask, uint8_t *dchunk) { (void)c; (void)mask; (void)dchunk; return -1; }
__attribute__((weak)) void gpudec_stats(uint64_t *subs, uint64_t *batches, uint64_t *redo, double *gpu_s) { if (subs) *subs = 0; if (batches) *batches = 0; if (redo) *redo = 0; if (gpu_s) *gpu_s = 0; }
/* decode the substreams in need of a parsed lossy chunk: on the GPU when it is on (a substream the GPU cannot decode, or a device
   error: the request on the CPU, the reference status and bytes), else volcomp_fast.h. dsink (z3_read_sub_dev): a device 128^3 chunk
   the GPU writes the substreams into instead of dec; *dev = the substreams that went there (dec untouched for them) */
static volcomp_status z3_decode_subs(const vf_parsed *p, uint32_t need, uint8_t *dec, uint8_t *dsink, uint32_t *dev) {
    *dev = 0;
    if (gpudec_enabled() && !p->strict) {
        gd_chunk g = {p->payload, p->tok_n, p->byp_n, p->off, p->step, {0}, VF_SCAN16};
        for (int m = 0; m < 10; m++) g.dt[m] = (const uint32_t *)p->models[m].dt;
        if (dsink && gpudec_decode_dev(&g, need, dsink) == 0) { *dev = need; return VOLCOMP_OK; }
        if (gpudec_decode(&g, need, dec) == 0) return VOLCOMP_OK;
    }
    return vfx_decode_subs(p, need, dec);
}
void z3_io_stats(uint64_t *cache_hits, uint64_t *store_reads) { *cache_hits = atomic_load(&g_z3_cache_hits); *store_reads = atomic_load(&g_z3_store_reads); }
static uint8_t *cache_get(const z3 *z, const char *skey, const char *item, size_t *len) {
    if (!z->cache) return nullptr;
    char p[1400];
    cache_path(z, skey, item, p, sizeof p);
    int fd = open(p, O_RDONLY);
    if (fd < 0) return nullptr;
    struct stat st;
    if (fstat(fd, &st) || st.st_size < 0 || (uint64_t)st.st_size >= SIZE_MAX) { close(fd); return nullptr; }
    uint8_t *b = malloc((size_t)st.st_size + 1);
    size_t got = 0;
    while (b && got < (size_t)st.st_size) {
        ssize_t r = read(fd, b + got, (size_t)st.st_size - got);
        if (r < 0 && errno == EINTR) continue;
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
    snprintf(tmp, sizeof tmp, "%s.tmp%ld-%lu", p, (long)getpid(), (unsigned long)pthread_self());
    int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return;
    size_t off = 0;
    while (off < len) {
        ssize_t w = write(fd, b + off, len - off);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) { close(fd); unlink(tmp); return; }
        off += (size_t)w;
    }
    close(fd);
    rename(tmp, p);
}

/* ---------- shard index ---------- */

typedef ientry_t ientry;

static uint32_t index_crc_table[256];
static pthread_once_t index_crc_once = PTHREAD_ONCE_INIT;
static void index_crc_init(void) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) c = c & 1 ? 0x82F63B78u ^ (c >> 1) : c >> 1;
        index_crc_table[i] = c;
    }
}
static uint32_t index_crc(const uint8_t *b, size_t n) {
    pthread_once(&index_crc_once, index_crc_init);
    uint32_t crc = UINT32_MAX;
    for (size_t i = 0; i < n; i++) crc = index_crc_table[(crc ^ b[i]) & 255] ^ (crc >> 8);
    return ~crc;
}

/* in-memory index table (open addressing on the shard key; UFSM_Z3_INDEX_MEM=0 disables it) */
static size_t key_hash(const char *k) { size_t h = 1469598103934665603ull; for (; *k; k++) h = (h ^ (uint8_t)*k) * 1099511628211ull; return h; }
static int imem_on(void) { static int on = -1; if (on < 0) on = option_on("UFSM_Z3_INDEX_MEM"); return on; }
static int imem_get(z3 *z, const char *skey, ientry *idx) {   /* -1 not cached, else present (0/1) with idx filled */
    if (!imem_on()) return -1;
    int r = -1;
    pthread_mutex_lock(&z->imu);
    if (z->icap)
        for (size_t i = key_hash(skey) & (z->icap - 1); z->itab[i].key; i = (i + 1) & (z->icap - 1))
            if (!strcmp(z->itab[i].key, skey)) { r = z->itab[i].present; if (r) memcpy(idx, z->itab[i].idx, (size_t)z->nc * sizeof *idx); break; }
    pthread_mutex_unlock(&z->imu);
    return r;
}
static int imem_known(z3 *z, const char *skey) {
    if (!imem_on()) return 0;
    int r = 0;
    pthread_mutex_lock(&z->imu);
    if (z->icap)
        for (size_t i = key_hash(skey) & (z->icap - 1); z->itab[i].key; i = (i + 1) & (z->icap - 1))
            if (!strcmp(z->itab[i].key, skey)) { r = 1; break; }
    pthread_mutex_unlock(&z->imu);
    return r;
}
static void imem_put(z3 *z, const char *skey, const ientry *idx, int present) {
    if (!imem_on()) return;
    pthread_mutex_lock(&z->imu);
    if ((z->icount + 1) * 2 > z->icap) {   /* grow (bounded: 1M shards) */
        size_t nc = z->icap ? z->icap * 2 : 256;
        struct ient_slot *t = nc <= (1u << 21) ? calloc(nc, sizeof *t) : nullptr;
        if (!t) { pthread_mutex_unlock(&z->imu); return; }
        for (size_t i = 0; i < z->icap; i++) if (z->itab[i].key) {
            size_t j = key_hash(z->itab[i].key) & (nc - 1);
            while (t[j].key) j = (j + 1) & (nc - 1);
            t[j] = z->itab[i];
        }
        free(z->itab); z->itab = t; z->icap = nc;
    }
    size_t i = key_hash(skey) & (z->icap - 1);
    while (z->itab[i].key && strcmp(z->itab[i].key, skey)) i = (i + 1) & (z->icap - 1);
    if (!z->itab[i].key) {
        ientry *c = present ? malloc((size_t)z->nc * sizeof *c) : nullptr;
        char *k = strdup(skey);
        if (k && (!present || c)) {
            if (c) memcpy(c, idx, (size_t)z->nc * sizeof *c);
            z->itab[i] = (struct ient_slot){k, c, present}; z->icount++;
        } else { free(k); free(c); }
    }
    pthread_mutex_unlock(&z->imu);
}

/* Returns 1 with index filled, 0 if the shard is missing, -1 on error. */
static int load_index(z3 *z, const char *skey, ientry *idx) {
    const int mem = imem_get(z, skey, idx);
    if (mem >= 0) return mem;
    size_t ilen = (size_t)z->nc * 16 + 4, n = 0;
    int fetched = 0;
    uint64_t payload_size = UINT64_MAX;
    uint8_t *b = cache_get(z, skey, "idx", &n);
    if (!b) {
        uint8_t *missing = cache_get(z, skey, "missing", &n);
        if (missing) { free(missing); imem_put(z, skey, nullptr, 0); return 0; }
        b = malloc(ilen);
        if (!b) return FAIL("index allocation for %s", skey);
        int64_t size = -1, got = store_read_tail(z->s, skey, (int64_t)ilen, b, &size);   /* one suffix-range GET */
        if (got == -2) { free(b); cache_put(z, skey, "missing", (const uint8_t *)"", 0); imem_put(z, skey, nullptr, 0); return 0; }
        if (size >= 0 && (uint64_t)size < ilen) { free(b); return FAIL("%s: shorter than its index", skey); }
        if (got != (int64_t)ilen) { free(b); return FAIL("read index of %s (HTTP %ld)", skey, store_last_status()); }
        payload_size = (uint64_t)size - ilen;
        n = ilen; fetched = 1;
    }
    if (n != ilen) { free(b); return FAIL("%s: bad index size", skey); }
    uint32_t expected;
    memcpy(&expected, b + ilen - 4, 4);
    if (index_crc(b, ilen - 4) != expected) { free(b); return FAIL("%s: index CRC32C mismatch", skey); }
    for (int i = 0; i < z->nc; i++) {
        memcpy(&idx[i].off, b + (size_t)i * 16, 8);
        memcpy(&idx[i].len, b + (size_t)i * 16 + 8, 8);
        ientry e = idx[i];
        if ((e.off == UINT64_MAX && e.len == UINT64_MAX) || !e.len) continue;
        if (e.off > INT64_MAX || e.len > INT64_MAX - e.off ||
            e.off > payload_size || e.len > payload_size - e.off) {
            free(b); return FAIL("%s: chunk %d range outside shard payload", skey, i);
        }
    }
    if (fetched) cache_put(z, skey, "idx", b, n);
    free(b);
    imem_put(z, skey, idx, 1);
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

typedef struct { int first, count; uint64_t off, len; } range_work;

typedef struct {
    z3 *z;
    const int64_t *o, *n;
    uint8_t *out;
    work *items;
    int nitems;
    range_work *ranges; int nranges;
    atomic_int next;
    atomic_int failed;
    char error[256];
    /* batch mode: tasks become ready as the main thread's multi fetch completes them; fetched[task] = bytes of the range */
    int batch;
    uint8_t **fetched;
    int *ready, rq_head, rq_tail, rq_closed;
    pthread_mutex_t rq_mu; pthread_cond_t rq_cv;
    uint32_t submask;   /* z3_read_sub */
    uint8_t *dsink; _Atomic uint32_t dmask;   /* z3_read_sub_dev: the device chunk the GPU decodes into, the substreams it did */
} job;

static void job_failed(job *jb) {
    if (!atomic_exchange(&jb->failed, 1)) snprintf(jb->error, sizeof jb->error, "%s", z3_error());
}

static int object_missing(store *s, const char *key) {
    return store_is_local(s) ? store_size(s, key) == -2 : store_last_status() == 404;
}

/* the volcomp substreams (128^3 chunks: 16 z x 32 y slabs, bit bz * 4 + by) under the request box in chunk w: z3_read decodes only
   those (UFSM_Z3_AUTOSUB=0: whole chunks). A window whose halo reaches 32 voxels into a chunk decodes a quarter of it (predict with
   window 576 halo 32: 3.4x -> 1.9x the decoded voxels of its interior). 0 = all (no saving, or not a 128^3 chunk). */
static uint32_t auto_submask(const job *jb, const work *w) {
    const int on = option_on("UFSM_Z3_AUTOSUB");
    const z3_meta *m = &jb->z->m;
    if (!on || m->chunk[0] != 128 || m->chunk[1] != 128 || m->chunk[2] != 128 || m->q < 0 || m->esize != 1) return 0;
    const int64_t c0[2] = {w->sz * m->shard[0] + (int64_t)w->cz * 128, w->sy * m->shard[1] + (int64_t)w->cy * 128};
    int64_t a[2], e[2];
    for (int i = 0; i < 2; i++) {
        a[i] = jb->o[i] - c0[i]; e[i] = jb->o[i] + jb->n[i] - 1 - c0[i];
        if (a[i] < 0) a[i] = 0;
        if (e[i] > 127) e[i] = 127;
        if (e[i] < a[i]) return 0;
    }
    uint32_t mk = 0;
    for (int64_t bz = a[0] >> 4; bz <= e[0] >> 4; bz++) for (int64_t by = a[1] >> 5; by <= e[1] >> 5; by++) mk |= 1u << (bz * 4 + by);
    return mk == 0xffffffffu ? 0 : mk;
}
static void copy_chunk(const job *jb, const work *w, const uint8_t *chunk) {
    const z3_meta *m = &jb->z->m;
    int64_t c0[3] = {w->sz * m->shard[0] + (int64_t)w->cz * m->chunk[0],
                     w->sy * m->shard[1] + (int64_t)w->cy * m->chunk[1],
                     w->sx * m->shard[2] + (int64_t)w->cx * m->chunk[2]};
    int64_t lo[3], hi[3];
    for (int i = 0; i < 3; i++) {
        lo[i] = jb->o[i] > c0[i] ? jb->o[i] : c0[i];
        int64_t ce = c0[i] + m->chunk[i], oe = jb->o[i] + jb->n[i];
        if (ce > m->shape[i]) ce = m->shape[i];
        hi[i] = ce < oe ? ce : oe;
        if (hi[i] <= lo[i]) return;
    }
    const size_t es = (size_t)m->esize, row = (size_t)(hi[2] - lo[2]) * es;
    for (int64_t zz = lo[0]; zz < hi[0]; zz++)
        for (int64_t yy = lo[1]; yy < hi[1]; yy++) {
            const uint8_t *src = chunk + (((zz - c0[0]) * m->chunk[1] + (yy - c0[1])) * m->chunk[2] + (lo[2] - c0[2])) * es;
            uint8_t *dst = jb->out + (((zz - jb->o[0]) * jb->n[1] + (yy - jb->o[1])) * jb->n[2] + (lo[2] - jb->o[2])) * es;
            memcpy(dst, src, row);
        }
}

static int decode_into(const z3 *z, const uint8_t *enc, size_t n, uint8_t *dec, ZSTD_DCtx *dctx, uint8_t *scratch, int *mask, uint32_t submask, uint8_t *dsink, _Atomic uint32_t *dmask) {
    const z3_meta *m = &z->m;
    size_t cv = (size_t)m->chunk[0] * m->chunk[1] * m->chunk[2];
    if (mask) *mask = 0;
    if (m->zstd) {
        const size_t cap = m->esize > 1 ? (VOLCOMP_TYPED_ENCODE_BOUND > cv * m->esize ? VOLCOMP_TYPED_ENCODE_BOUND : cv * m->esize) : (VOLCOMP_ENCODE_BOUND > cv ? VOLCOMP_ENCODE_BOUND : cv);
        size_t r = ZSTD_decompressDCtx(dctx, scratch, cap, enc, n);
        if (ZSTD_isError(r)) return FAIL("zstd: %s", ZSTD_getErrorName(r));
        enc = scratch;
        n = r;
    }
    if (m->q >= 0 && m->esize > 1) {   /* typed chunk (volcomp modes 7 / 8) */
        volcomp_status st = volcomp_decode_typed(enc, n, dec, cv * m->esize, (volcomp_dtype)m->dtype);
        if (st != VOLCOMP_OK) return FAIL("volcomp typed: %s", volcomp_status_string(st));
        return 0;
    }
    if (m->q >= 0) {
        uint32_t mode;
        if (mask && vll_chunk_mode(enc, n, &mode) == VOLCOMP_OK) *mask = mode == VLL_MODE_MASK || mode == VLL_MODE_MASK_LL;
        volcomp_status st;
        const int sub = submask && submask != 0xffffffffu;
        if (cv == VOLCOMP_CHUNK_VOXELS && vll_chunk_mode(enc, n, &mode) == VOLCOMP_OK && mode == VLL_MODE_LOSSY) {
            /* lossy (CT): the substreams asked for (all for a full read), from the shared cache (zshm.h) or decoded by
               volcomp_fast.h (the reference's bytes); a partial read leaves the rest 0 */
            const uint32_t want = sub ? submask : 0xffffffffu;
            uint32_t need = want;
            XXH128_hash_t hk = {0, 0};
            if (sub) memset(dec, 0, cv);
            if (zshm_on()) {
                hk = XXH3_128bits(enc, n);
                for (uint32_t s = 0; s < VF_NSUB; s++) if (want >> s & 1 && zshm_get(hk.low64, hk.high64, s, dec)) need &= ~(1u << s);
            }
            st = VOLCOMP_OK;
            if (need) {
                vf_parsed p; st = vf_parse(enc, n, &p);
                uint32_t dev = 0;
                if (st == VOLCOMP_OK) st = z3_decode_subs(&p, need, dec, dsink, &dev);
                if (dev) { atomic_fetch_or(dmask, dev); need &= ~dev; }   /* on the device only: not in dec, not for the shared cache */
                if (st == VOLCOMP_OK && zshm_on()) for (uint32_t s = 0; s < VF_NSUB; s++) if (need >> s & 1) zshm_put(hk.low64, hk.high64, s, dec);
            }
        } else if (sub && cv == VOLCOMP_CHUNK_VOXELS && vll_chunk_mode(enc, n, &mode) == VOLCOMP_OK && mode == VLL_MODE_CODED) {
            memset(dec, 0, cv);   /* only the substreams asked for (z3_read_sub) */
            vll_parsed lp; st = vll_parse(enc, n, &lp); for (uint32_t s = 0; s < VF_NSUB && st == VOLCOMP_OK; s++) if (submask >> s & 1) st = vll_decode_sub(&lp, s, dec, -1, NULL);
        } else st = volcomp_decode(enc, n, dec, cv);
        if (st != VOLCOMP_OK) return FAIL("volcomp: %s", volcomp_status_string(st));
        return 0;
    }
    if (n != cv * m->esize) return FAIL("raw chunk size %zu != %zu in %s (q %d, zstd %d)", n, cv * m->esize, z->key, m->q, m->zstd);
    memcpy(dec, enc, cv * m->esize);
    return 0;
}

/* fetch inner chunk (cz, cy, cx) into the cache without decoding: 1 fetched, 0 already cached or absent, -1 error */
static int cache_has(const z3 *z, const char *skey, const char *item) {
    if (!z->cache) return 0;
    char p[1400]; cache_path(z, skey, item, p, sizeof p);
    struct stat st; return stat(p, &st) == 0;
}
int z3_prefetch_chunk(z3 *z, int64_t cz, int64_t cy, int64_t cx) {
    if (!z->cache) return 0;
    int64_t sz = cz / z->cgrid[0], sy = cy / z->cgrid[1], sx = cx / z->cgrid[2];
    int ci = (int)(((cz % z->cgrid[0]) * z->cgrid[1] + (cy % z->cgrid[1])) * z->cgrid[2] + (cx % z->cgrid[2]));
    char skey[1200], item[16];
    shard_key(z, sz, sy, sx, skey, sizeof skey);
    if (!z->m.sharded) {   /* unsharded: the object is the chunk */
        if (cache_has(z, skey, "0")) return 0;
        size_t n; uint8_t *b = store_read_all(z->s, skey, &n);
        if (!b) return object_missing(z->s, skey) ? 0 : FAIL("prefetch object %s", skey);
        cache_put(z, skey, "0", b, n); free(b); return 1;
    }
    ientry *idx = malloc((size_t)z->nc * sizeof *idx);
    if (!idx) return FAIL("index allocation");
    int li = load_index(z, skey, idx);
    if (li <= 0) { free(idx); return li; }
    ientry e = idx[ci]; free(idx);
    if ((e.off == UINT64_MAX && e.len == UINT64_MAX) || !e.len) return 0;   /* chunk absent (fill) */
    snprintf(item, sizeof item, "%d", ci);
    if (cache_has(z, skey, item)) return 0;
    uint8_t *b = malloc(e.len);
    if (!b) return FAIL("prefetch allocation");
    if (store_read(z->s, skey, (int64_t)e.off, (int64_t)e.len, b) != (int64_t)e.len) { free(b); return FAIL("prefetch chunk %d of %s", ci, skey); }
    cache_put(z, skey, item, b, e.len); free(b);
    return 1;
}

/* the encoded bytes of inner chunk (cz, cy, cx) (after zstd when the chain has it): 1 with *buf (malloc) / *n, 0 absent, -1 error */
int z3_chunk_raw(z3 *z, int64_t cz, int64_t cy, int64_t cx, uint8_t **buf, size_t *n) {
    *buf = nullptr; *n = 0;
    int64_t sz = cz / z->cgrid[0], sy = cy / z->cgrid[1], sx = cx / z->cgrid[2];
    int ci = (int)(((cz % z->cgrid[0]) * z->cgrid[1] + (cy % z->cgrid[1])) * z->cgrid[2] + (cx % z->cgrid[2]));
    char skey[1200]; shard_key(z, sz, sy, sx, skey, sizeof skey);
    uint8_t *b = nullptr; size_t len = 0;
    if (!z->m.sharded) {
        if (!(b = store_read_all(z->s, skey, &len))) return object_missing(z->s, skey) ? 0 : FAIL("read object %s", skey);
    } else {
        ientry *idx = malloc((size_t)z->nc * sizeof *idx);
        if (!idx) return FAIL("index allocation");
        int li = load_index(z, skey, idx);
        if (li <= 0) { free(idx); return li; }
        ientry e = idx[ci]; free(idx);
        if ((e.off == UINT64_MAX && e.len == UINT64_MAX) || !e.len) return 0;
        len = (size_t)e.len;
        if (!(b = malloc(len)) || store_read(z->s, skey, (int64_t)e.off, (int64_t)len, b) != (int64_t)len) { free(b); return FAIL("read chunk %d of %s", ci, skey); }
    }
    if (z->m.zstd) {
        const size_t cap = VOLCOMP_TYPED_ENCODE_BOUND; uint8_t *d = malloc(cap);
        size_t r = d ? ZSTD_decompress(d, cap, b, len) : 0; free(b);
        if (!d || ZSTD_isError(r)) { free(d); return FAIL("zstd chunk of %s", skey); }
        b = d; len = r;
    }
    *buf = b; *n = len;
    return 1;
}

typedef struct {
    z3 *z;
    int64_t s0[3], ns[3];
    int *cold, total;
    atomic_int next, failed;
    char error[256];
} index_job;
static void index_key(const index_job *j, int i, char *key, size_t n) {
    int64_t sx = j->s0[2] + i % j->ns[2];
    int64_t sy = j->s0[1] + (i / j->ns[2]) % j->ns[1];
    int64_t sz = j->s0[0] + i / (j->ns[2] * j->ns[1]);
    shard_key(j->z, sz, sy, sx, key, n);
}
static void *index_worker(void *arg) {
    index_job *j = arg;
    ientry *idx = malloc((size_t)j->z->nc * sizeof *idx);
    if (!idx) {
        if (!atomic_exchange(&j->failed, 1)) snprintf(j->error, sizeof j->error, "index allocation");
        return nullptr;
    }
    for (;;) {
        int task = atomic_fetch_add(&j->next, 1);
        if (task >= j->total || atomic_load(&j->failed)) break;
        char key[1200]; index_key(j, j->cold[task], key, sizeof key);
        if (load_index(j->z, key, idx) < 0) {
            if (!atomic_exchange(&j->failed, 1)) snprintf(j->error, sizeof j->error, "%s", z3_error());
            break;
        }
    }
    free(idx);
    return nullptr;
}
static int prefetch_indices(z3 *z, const int64_t *s0, const int64_t *s1, int nthreads) {
    index_job j = {.z = z}; int64_t count = 1;
    for (int d = 0; d < 3; d++) {
        j.s0[d] = s0[d]; j.ns[d] = s1[d] - s0[d] + 1;
        if (j.ns[d] > INT_MAX / count) return FAIL("too many shard indices");
        count *= j.ns[d];
    }
    j.cold = malloc((size_t)count * sizeof *j.cold);
    if (!j.cold) return FAIL("index work allocation");
    for (int i = 0; i < count; i++) {
        char key[1200]; index_key(&j, i, key, sizeof key);
        if (!imem_known(z, key) && !cache_has(z, key, "idx") && !cache_has(z, key, "missing")) j.cold[j.total++] = i;
    }
    if (!j.total) { free(j.cold); return 0; }
    int nt = nthreads > 0 ? nthreads : 16;
    if (nt > j.total) nt = j.total;
    if (nt > 16) nt = 16;
    if (nt < 1) nt = 1;
    pthread_t th[16]; int started = 0;
    for (int i = 1; i < nt; i++) {
        if (pthread_create(&th[started], nullptr, index_worker, &j)) break;
        started++;
    }
    index_worker(&j);
    for (int i = 0; i < started; i++) pthread_join(th[i], nullptr);
    free(j.cold);
    return atomic_load(&j.failed) ? FAIL("parallel shard index: %s", j.error) : 0;
}

/* ---------- decoded mask chunks ----------
   The volcomp mask codec is a serial arithmetic coder (~10 ms per 128^3 chunk, the sampler's largest decode for binary
   labels). Decoded mask chunks with two values are kept in memory as bits, process-wide, the least recently used dropped
   past UFSM_Z3_MASK_CACHE_MB (default 1024; 0 = off). Keys: the z3's id, shard and inner chunk. */
typedef struct mc_entry { uint64_t z, tick; int64_t s[3]; int ci, refs, dead; uint8_t v0, v1; uint8_t *bits; size_t nbytes, h; struct mc_entry *next, *lp, *ln; } mc_entry;
/* lock striping: MC_SH independent shards by key hash (each its mutex, buckets, recency list and 1 / MC_SH of the budget), so the
   sampler's workers do not queue on one process-wide lock (one global lock measured 1.3 ms of waiting per 1.7 s at 18 workers) */
#define MC_NB 8192
#define MC_SH 16
typedef struct { pthread_mutex_t mu; mc_entry *b[MC_NB / MC_SH]; mc_entry *lh, *lt; size_t bytes, n; uint64_t tick; } mc_shard;
static mc_shard g_mc[MC_SH];
static pthread_once_t g_mc_once = PTHREAD_ONCE_INIT;
static void mc_init(void) { for (int k = 0; k < MC_SH; k++) pthread_mutex_init(&g_mc[k].mu, nullptr); }
static long long g_mc_budget = -1;
static void mc_lru_unlink(mc_shard *c, mc_entry *e) {
    if (e->lp) e->lp->ln = e->ln; else c->lh = e->ln;
    if (e->ln) e->ln->lp = e->lp; else c->lt = e->lp;
    e->lp = e->ln = nullptr;
}
static void mc_lru_front(mc_shard *c, mc_entry *e) { e->lp = nullptr; e->ln = c->lh; if (c->lh) c->lh->lp = e; c->lh = e; if (!c->lt) c->lt = e; }
static long long mc_budget(void) {
    if (g_mc_budget < 0) { const char *e = getenv("UFSM_Z3_MASK_CACHE_MB"); g_mc_budget = (e && *e ? atoll(e) : 1024) << 20; }
    return g_mc_budget;
}
static size_t mc_hash(uint64_t z, const work *w) {
    uint64_t h = z * 0x9e3779b97f4a7c15ull ^ (uint64_t)w->sz * 0xc2b2ae3d27d4eb4full ^ (uint64_t)w->sy * 0x165667b19e3779f9ull ^
                 (uint64_t)w->sx * 0x27d4eb2f165667c5ull ^ (uint64_t)(w->ci + 1) * 0x94d049bb133111ebull;
    h ^= h >> 31; return (size_t)(h & (MC_NB - 1));
}
static mc_shard *mc_of(size_t h) { pthread_once(&g_mc_once, mc_init); return &g_mc[h % MC_SH]; }
/* copy the window part of a cached chunk (copy_chunk on the expanded bits) */
static int mc_copy(const job *jb, const work *w) {
    const z3 *z = jb->z; const z3_meta *m = &z->m;
    const size_t h = mc_hash(z->id, w);
    mc_shard *c = mc_of(h);
    pthread_mutex_lock(&c->mu);
    mc_entry *e = c->b[h / MC_SH];
    while (e && !(e->z == z->id && e->s[0] == w->sz && e->s[1] == w->sy && e->s[2] == w->sx && e->ci == w->ci)) e = e->next;
    if (!e) { pthread_mutex_unlock(&c->mu); return 0; }
    e->tick = ++c->tick; e->refs++;   /* copied outside the lock; an evicted entry in use is freed by its last reader */
    mc_lru_unlink(c, e); mc_lru_front(c, e);
    pthread_mutex_unlock(&c->mu);
    int64_t c0[3] = {w->sz * m->shard[0] + (int64_t)w->cz * m->chunk[0], w->sy * m->shard[1] + (int64_t)w->cy * m->chunk[1],
                     w->sx * m->shard[2] + (int64_t)w->cx * m->chunk[2]}, lo[3], hi[3];
    int empty = 0;
    for (int i = 0; i < 3; i++) {
        lo[i] = jb->o[i] > c0[i] ? jb->o[i] : c0[i];
        int64_t ce = c0[i] + m->chunk[i], oe = jb->o[i] + jb->n[i];
        if (ce > m->shape[i]) ce = m->shape[i];
        hi[i] = ce < oe ? ce : oe;
        if (hi[i] <= lo[i]) empty = 1;
    }
    const uint8_t v0 = e->v0, v1 = e->v1, *bits = e->bits;
    for (int64_t zz = lo[0]; !empty && zz < hi[0]; zz++)
        for (int64_t yy = lo[1]; yy < hi[1]; yy++) {
            size_t b = (size_t)(((zz - c0[0]) * m->chunk[1] + (yy - c0[1])) * m->chunk[2] + (lo[2] - c0[2]));
            uint8_t *dst = jb->out + ((zz - jb->o[0]) * jb->n[1] + (yy - jb->o[1])) * jb->n[2] + (lo[2] - jb->o[2]);
            for (int64_t x = 0; x < hi[2] - lo[2]; x++, b++) dst[x] = bits[b >> 3] >> (b & 7) & 1 ? v1 : v0;
        }
    pthread_mutex_lock(&c->mu);
    if (!--e->refs && e->dead) { free(e->bits); free(e); }
    pthread_mutex_unlock(&c->mu);
    return 1;
}
static int mc_has(const z3 *z, const work *w) {
    const size_t h = mc_hash(z->id, w);
    mc_shard *c = mc_of(h);
    pthread_mutex_lock(&c->mu);
    mc_entry *e = c->b[h / MC_SH];
    while (e && !(e->z == z->id && e->s[0] == w->sz && e->s[1] == w->sy && e->s[2] == w->sx && e->ci == w->ci)) e = e->next;
    pthread_mutex_unlock(&c->mu);
    return e != nullptr;
}
static void mc_put(const z3 *z, const work *w, const uint8_t *dec, size_t cv) {
    const long long budget = mc_budget() / MC_SH;   /* each shard its share */
    const size_t nb = (cv + 7) / 8;
    if (budget <= 0 || (long long)nb > budget) return;
    uint8_t v0 = dec[0], v1 = dec[0]; size_t k = 0;
    for (; k < cv && dec[k] == v0; k++) {}
    if (k < cv) v1 = dec[k];
    uint8_t *bits = calloc(nb, 1);
    if (!bits) return;
    for (k = 0; k < cv; k++) {
        if (dec[k] == v1 && v1 != v0) bits[k >> 3] |= (uint8_t)(1u << (k & 7));
        else if (dec[k] != v0) { free(bits); return; }   /* three values: not a mask */
    }
    mc_entry *ne = malloc(sizeof *ne);
    if (!ne) { free(bits); return; }
    const size_t h = mc_hash(z->id, w);
    mc_shard *c = mc_of(h);
    *ne = (mc_entry){z->id, 0, {w->sz, w->sy, w->sx}, w->ci, 0, 0, v0, v1, bits, nb, h, nullptr, nullptr, nullptr};
    pthread_mutex_lock(&c->mu);
    for (mc_entry *e = c->b[h / MC_SH]; e; e = e->next)
        if (e->z == z->id && e->s[0] == w->sz && e->s[1] == w->sy && e->s[2] == w->sx && e->ci == w->ci) { pthread_mutex_unlock(&c->mu); free(bits); free(ne); return; }
    while (c->n && c->lt && (long long)(c->bytes + nb) > budget) {   /* drop the shard's least recently used: its list's tail */
        mc_entry *v = c->lt, **pe = &c->b[v->h / MC_SH];
        while (*pe != v) pe = &(*pe)->next;
        *pe = v->next; mc_lru_unlink(c, v); c->bytes -= v->nbytes; c->n--;
        if (v->refs) v->dead = 1; else { free(v->bits); free(v); }
    }
    ne->tick = ++c->tick; ne->next = c->b[h / MC_SH]; c->b[h / MC_SH] = ne; mc_lru_front(c, ne); c->bytes += nb; c->n++;
    pthread_mutex_unlock(&c->mu);
}

/* the decode buffers and zstd context of a thread, kept between reads (the sampler's workers read a window per draw: a fresh
   2 + 14 MB per read was an mmap / munmap pair, page faults and a TLB shootdown across the process); freed at thread exit */
typedef struct { uint8_t *dec, *scratch; size_t ndec, nscr; ZSTD_DCtx *dctx; } z3_tls;
static pthread_key_t z3_tls_key;
static pthread_once_t z3_tls_once = PTHREAD_ONCE_INIT;
static void z3_tls_free(void *p) { z3_tls *t = p; free(t->dec); free(t->scratch); if (t->dctx) ZSTD_freeDCtx(t->dctx); free(t); }
static void z3_tls_init(void) { pthread_key_create(&z3_tls_key, z3_tls_free); }
static z3_tls *z3_tls_get(size_t ndec, size_t nscr) {
    pthread_once(&z3_tls_once, z3_tls_init);
    z3_tls *t = pthread_getspecific(z3_tls_key);
    if (!t) { if (!(t = calloc(1, sizeof *t))) return nullptr; if (pthread_setspecific(z3_tls_key, t)) { free(t); return nullptr; } }
    if (t->ndec < ndec) { free(t->dec); t->dec = malloc(ndec); t->ndec = t->dec ? ndec : 0; }
    if (t->nscr < nscr) { free(t->scratch); t->scratch = malloc(nscr); t->nscr = t->scratch ? nscr : 0; }
    if (!t->dctx) t->dctx = ZSTD_createDCtx();
    return t->dec && t->scratch && t->dctx ? t : nullptr;
}
/* one task: a coalesced range (fetched: its bytes, or nullptr to read it here) or a single chunk */
static int run_task(job *jb, int task, uint8_t *fetched, uint8_t *dec, uint8_t *scratch, ZSTD_DCtx *dctx, int mc_on) {
    z3 *z = jb->z;
    const size_t cv = (size_t)z->m.chunk[0] * z->m.chunk[1] * z->m.chunk[2];
    int i = jb->ranges ? jb->ranges[task].first : task;
    const range_work *rg = jb->ranges ? &jb->ranges[task] : nullptr;
    if (rg && (rg->count > 1 || fetched)) {
        work *w0 = &jb->items[i]; char skey[1200];
        shard_key(z, w0->sz, w0->sy, w0->sx, skey, sizeof skey);
        uint8_t *group = fetched ? fetched : malloc((size_t)rg->len);
        if (!fetched && (!group || store_read(z->s, skey, (int64_t)rg->off, (int64_t)rg->len, group) != (int64_t)rg->len)) {
            free(group); FAIL("read grouped chunks of %s", skey); return -1;
        }
        int rc = 0;
        for (int j = 0; j < rg->count && !rc; j++) {
            work *w = &jb->items[i + j]; char item[16]; snprintf(item, sizeof item, "%d", w->ci);
            const uint8_t *enc = group + (size_t)(w->off - rg->off); size_t n = (size_t)w->len;
            cache_put(z, skey, item, enc, n); atomic_fetch_add(&g_z3_store_reads, 1);
            int msk;
            if (decode_into(z, enc, n, dec, dctx, scratch, &msk, jb->submask ? jb->submask : auto_submask(jb, w), jb->dsink, &jb->dmask)) { rc = -1; break; }
            if (msk) mc_put(z, w, dec, cv);
            copy_chunk(jb, w, dec);
        }
        free(group);
        return rc;
    }
    work *w = &jb->items[i];
    if (!w->present) return 0;
    if (mc_on && mc_copy(jb, w)) return 0;
    char skey[1200], item[16];
    shard_key(z, w->sz, w->sy, w->sx, skey, sizeof skey);
    snprintf(item, sizeof item, "%d", w->ci < 0 ? 0 : w->ci);
    size_t n;
    uint8_t *b = cache_get(z, skey, item, &n);
    if (b) atomic_fetch_add(&g_z3_cache_hits, 1); else if (z->cache) { atomic_fetch_add(&g_z3_store_reads, 1); if (getenv("UFSM_Z3_TRACE")) fprintf(stderr, "z3 store read: %s item %s (window %lld %lld %lld)\n", skey, item, (long long)jb->o[0], (long long)jb->o[1], (long long)jb->o[2]); }
    if (!b) {
        if (w->ci < 0) {
            b = store_read_all(z->s, skey, &n);
            if (!b) {
                if (object_missing(z->s, skey)) return 0;
                FAIL("read object %s", skey); return -1;
            }
        } else {
            n = (size_t)w->len;
            b = malloc(n);
            if (!b || store_read(z->s, skey, (int64_t)w->off, (int64_t)n, b) != (int64_t)n) {
                free(b);
                FAIL("read chunk %d of %s", w->ci, skey);
                return -1;
            }
        }
        cache_put(z, skey, item, b, n);
    }
    int msk, rc = decode_into(z, b, n, dec, dctx, scratch, &msk, jb->submask ? jb->submask : auto_submask(jb, w), jb->dsink, &jb->dmask);
    free(b);
    if (rc) return -1;
    if (msk && mc_on) mc_put(z, w, dec, cv);
    copy_chunk(jb, w, dec);
    return 0;
}
/* batch mode: the next ready task (blocks), -1 when the queue is closed and drained */
static int rq_pop(job *jb) {
    pthread_mutex_lock(&jb->rq_mu);
    while (jb->rq_head == jb->rq_tail && !jb->rq_closed) pthread_cond_wait(&jb->rq_cv, &jb->rq_mu);
    int t = jb->rq_head < jb->rq_tail ? jb->ready[jb->rq_head++] : -1;
    pthread_mutex_unlock(&jb->rq_mu);
    return t;
}
static void rq_push(job *jb, int task) {
    pthread_mutex_lock(&jb->rq_mu);
    jb->ready[jb->rq_tail++] = task;
    pthread_cond_signal(&jb->rq_cv);
    pthread_mutex_unlock(&jb->rq_mu);
}
static void *worker(void *arg) {
    job *jb = arg;
    z3 *z = jb->z;
    size_t cv = (size_t)z->m.chunk[0] * z->m.chunk[1] * z->m.chunk[2];
    const size_t es = (size_t)z->m.esize, cvb = cv * es, sb = es > 1 ? (VOLCOMP_TYPED_ENCODE_BOUND > cvb ? VOLCOMP_TYPED_ENCODE_BOUND : cvb) : (VOLCOMP_ENCODE_BOUND > cv ? VOLCOMP_ENCODE_BOUND : cv);
    z3_tls *tl = z3_tls_get(cvb, sb);
    uint8_t *dec = tl ? tl->dec : nullptr;
    uint8_t *scratch = tl ? tl->scratch : nullptr;
    ZSTD_DCtx *dctx = tl ? tl->dctx : nullptr;
    const int mc_on = mc_budget() > 0 && es == 1;
    if (!dec || !scratch || !dctx) { FAIL("chunk decode allocation"); job_failed(jb); }
    for (;;) {
        int task;
        if (jb->batch) {
            if ((task = rq_pop(jb)) < 0) break;
            uint8_t *f = jb->fetched[task]; jb->fetched[task] = nullptr;
            if (atomic_load(&jb->failed)) { free(f); continue; }   /* drain: buffers still to free */
            if (run_task(jb, task, f, dec, scratch, dctx, mc_on)) job_failed(jb);
            continue;
        }
        task = atomic_fetch_add(&jb->next, 1);
        if (task >= (jb->ranges ? jb->nranges : jb->nitems) || atomic_load(&jb->failed)) break;
        if (run_task(jb, task, nullptr, dec, scratch, dctx, mc_on)) { job_failed(jb); break; }
    }
    return nullptr;
}

/* batch fetch: a multi request finished on the main thread */
typedef struct { job *jb; int *task_of; store_req *reqs; } batch_ud;
static void batch_done(void *ud, int i) {
    batch_ud *u = ud; job *jb = u->jb; store_req *q = &u->reqs[i]; const int task = u->task_of[i];
    if (q->result != q->len) {
        FAIL("read %s chunks of %s (%lld bytes at %lld)", jb->ranges[task].count > 1 ? "grouped" : "one", q->key, (long long)q->len, (long long)q->off);
        job_failed(jb); free(q->out); q->out = nullptr; return;
    }
    jb->fetched[task] = q->out; q->out = nullptr;
    rq_push(jb, task);
}
/* sort live items by shard, then by byte offset (coalescing merges neighbours in file order) */
static int work_cmp(const void *a, const void *b) {
    const work *x = a, *y = b;
    if (x->sz != y->sz) return x->sz < y->sz ? -1 : 1;
    if (x->sy != y->sy) return x->sy < y->sy ? -1 : 1;
    if (x->sx != y->sx) return x->sx < y->sx ? -1 : 1;
    return x->off < y->off ? -1 : x->off > y->off;
}

int64_t z3_chunk_map(z3 *z, int64_t grid[3], uint8_t **bits) {
    const z3_meta *m = &z->m; *bits = nullptr;
    if (!m->sharded) return FAIL("chunk map: not a sharded array");
    int64_t ns[3];
    for (int i = 0; i < 3; i++) { grid[i] = (m->shape[i] + m->chunk[i] - 1) / m->chunk[i]; ns[i] = (m->shape[i] + m->shard[i] - 1) / m->shard[i]; }
    uint8_t *b = calloc((size_t)(grid[0] * grid[1] * grid[2]), 1); ientry *idx = malloc((size_t)z->nc * sizeof *idx);
    if (!b || !idx) { free(b); free(idx); return FAIL("chunk map allocation"); }
    int64_t n = 0;
    for (int64_t sz = 0; sz < ns[0]; sz++) for (int64_t sy = 0; sy < ns[1]; sy++) for (int64_t sx = 0; sx < ns[2]; sx++) {
        char skey[1200]; shard_key(z, sz, sy, sx, skey, sizeof skey);
        const int pr = load_index(z, skey, idx);
        if (pr < 0) { free(b); free(idx); return -1; }
        if (!pr) continue;
        for (int cz = 0; cz < z->cgrid[0]; cz++) for (int cy = 0; cy < z->cgrid[1]; cy++) for (int cx = 0; cx < z->cgrid[2]; cx++) {
            const int64_t gz = sz * z->cgrid[0] + cz, gy = sy * z->cgrid[1] + cy, gx = sx * z->cgrid[2] + cx;
            if (gz >= grid[0] || gy >= grid[1] || gx >= grid[2]) continue;
            const ientry e = idx[(cz * z->cgrid[1] + cy) * z->cgrid[2] + cx];
            if ((e.off == UINT64_MAX && e.len == UINT64_MAX) || !e.len) continue;
            b[(gz * grid[1] + gy) * grid[2] + gx] = 1; n++;
        }
    }
    free(idx); *bits = b;
    return n;
}
int z3_shard_present(z3 *z, int64_t sz, int64_t sy, int64_t sx) {
    char skey[1200];
    shard_key(z, sz, sy, sx, skey, sizeof skey);
    if (z->m.sharded) {
        ientry *idx = malloc((size_t)z->nc * sizeof *idx);
        if (!idx) return FAIL("index allocation");
        int r = load_index(z, skey, idx);
        free(idx);
        return r;
    }
    int64_t s = store_size(z->s, skey);
    return s == -2 ? 0 : s < 0 ? -1 : 1;
}

/* the fill value over the whole output box */
static void z3_fill(const z3_meta *m, uint8_t *out, size_t nv) {
    if (m->esize == 1) memset(out, m->fill, nv);
    else {   /* typed fill: one element, then copied */
        uint8_t e[4] = {0}; const double f = m->fillv;
        switch (m->dtype) {
        case 1: { uint16_t v = (uint16_t)f; memcpy(e, &v, 2); break; }
        case 2: { int16_t v = (int16_t)f; memcpy(e, &v, 2); break; }
        case 3: { uint32_t v = (uint32_t)f; memcpy(e, &v, 4); break; }
        case 4: { int32_t v = (int32_t)f; memcpy(e, &v, 4); break; }
        case 5: { float v = (float)f; memcpy(e, &v, 4); break; }
        }
        for (size_t i = 0; i < nv; i++) memcpy(out + i * m->esize, e, (size_t)m->esize);
    }
}
static int z3_read_impl(z3 *z, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads, uint32_t submask, uint8_t *dsink, uint32_t *dmask);
int z3_read(z3 *z, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads) { return z3_read_impl(z, o, n, out, nthreads, 0, nullptr, nullptr); }
int z3_read_sub(z3 *z, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads, uint32_t submask) { return z3_read_impl(z, o, n, out, nthreads, submask, nullptr, nullptr); }
int z3_read_sub_dev(z3 *z, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads, uint32_t submask, uint8_t *dchunk, uint32_t *dev_mask) {
    *dev_mask = 0;
    const z3_meta *m = &z->m;   /* one whole, aligned 128^3 uint8 chunk; the shared cache off (it stores host bytes) */
    const int one = dchunk && m->chunk[0] == 128 && m->chunk[1] == 128 && m->chunk[2] == 128 && m->esize == 1 && n[0] == 128 && n[1] == 128 && n[2] == 128 &&
                    !(o[0] % 128) && !(o[1] % 128) && !(o[2] % 128) && !zshm_on() && gpudec_enabled();
    return z3_read_impl(z, o, n, out, nthreads, submask, one ? dchunk : nullptr, dev_mask);
}
static int z3_read_impl(z3 *z, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads, uint32_t submask, uint8_t *dsink, uint32_t *dmask) {
    const z3_meta *m = &z->m;
    const size_t nv = (size_t)n[0] * n[1] * n[2];
    /* a box inside the array whose chunks are all present (sharded) is overwritten entirely by the chunk copies: no fill pass (the
       sampler's P512 CT window: a 128 MB memset per read) */
    int inside = m->sharded;
    for (int i = 0; i < 3; i++) inside &= o[i] >= 0 && o[i] + n[i] <= m->shape[i];
    if (!inside) z3_fill(m, out, nv);
    int64_t lo[3], hi[3];
    for (int i = 0; i < 3; i++) {
        lo[i] = o[i] < 0 ? 0 : o[i];
        hi[i] = o[i] + n[i] > m->shape[i] ? m->shape[i] : o[i] + n[i];
        if (hi[i] <= lo[i]) return 0;
    }
    int64_t s0[3], s1[3];
    for (int i = 0; i < 3; i++) { s0[i] = lo[i] / m->shard[i]; s1[i] = (hi[i] - 1) / m->shard[i]; }
    if (option_on("UFSM_Z3_INDEX_PARALLEL") && m->sharded && !store_is_local(z->s))
        if (prefetch_indices(z, s0, s1, nthreads)) return -1;
    int cap = 64, nitems = 0;
    work *items = malloc((size_t)cap * sizeof *items);
    ientry *idx = malloc((size_t)z->nc * sizeof *idx);
    int rc = 0;
    if (!items || !idx) { free(items); free(idx); return FAIL("read work allocation"); }
    for (int64_t sz = s0[0]; sz <= s1[0] && !rc; sz++)
        for (int64_t sy = s0[1]; sy <= s1[1] && !rc; sy++)
            for (int64_t sx = s0[2]; sx <= s1[2] && !rc; sx++) {
                char skey[1200];
                shard_key(z, sz, sy, sx, skey, sizeof skey);
                int present = 1;
                if (m->sharded) {
                    present = load_index(z, skey, idx);
                    if (present < 0) { rc = -1; break; }
                    if (!present) { if (inside) { z3_fill(m, out, nv); inside = 0; } continue; }   /* (a missing shard: filled) */
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
                            if (nitems == cap) {
                                if (cap > INT_MAX / 2) { free(idx); free(items); return FAIL("too many chunks"); }
                                cap *= 2; work *grown = realloc(items, (size_t)cap * sizeof *items);
                                if (!grown) { free(idx); free(items); return FAIL("read work allocation"); }
                                items = grown;
                            }
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
    if (inside) for (int i = 0; i < nitems; i++) if (!items[i].present) { z3_fill(m, out, nv); inside = 0; break; }   /* (an absent chunk: filled) */
    range_work *ranges = nullptr; int nranges = 0;
    /* HTTP sharded reads go through range tasks: neighbouring cold chunks of one shard (file order, gap <= UFSM_Z3_GAP bytes,
       default 64 KiB; span <= UFSM_Z3_SPAN, default 2 MiB) become one request unless UFSM_Z3_COALESCE=0; with UFSM_HTTP_MULTI=1
       one curl_multi on this thread issues them concurrently while the workers decode (else each worker fetches its own) */
    const int coalesce = option_on("UFSM_Z3_COALESCE"), batch = m->sharded && store_batch_enabled(z->s);
    if ((coalesce || batch) && m->sharded && !store_is_local(z->s)) {
        static int64_t gap = -1, span = -1;
        if (gap < 0) { const char *v = getenv("UFSM_Z3_GAP"); gap = v && *v ? atoll(v) : 64 << 10; }
        if (span < 0) { const char *v = getenv("UFSM_Z3_SPAN"); span = v && *v ? atoll(v) : 2 << 20; }
        /* Absent chunks are already filled. They have no byte range and must not split nearby payloads. */
        int live = 0;
        for (int i = 0; i < nitems; i++) if (items[i].present) items[live++] = items[i];
        nitems = live;
        qsort(items, (size_t)nitems, sizeof *items, work_cmp);
        ranges = malloc((size_t)nitems * sizeof *ranges);
        if (!ranges && nitems) { free(items); return FAIL("range allocation"); }
        for (int i = 0; i < nitems;) {
            work *w = &items[i]; range_work rg = {i, 1, w->off, w->len};
            char skey[1200], item[16];
            shard_key(z, w->sz, w->sy, w->sx, skey, sizeof skey); snprintf(item, sizeof item, "%d", w->ci);
            int cold = coalesce && w->present && w->ci >= 0 && (int64_t)w->len <= span && !cache_has(z, skey, item);
            if (cold) for (int j = i + 1; j < nitems; j++) {
                work *q = &items[j]; uint64_t end = rg.off + rg.len;
                if (!q->present || q->sz != w->sz || q->sy != w->sy || q->sx != w->sx ||
                    q->off < end || (int64_t)(q->off - end) > gap || (int64_t)q->len > span ||
                    (int64_t)(q->off + q->len - rg.off) > span) break;
                snprintf(item, sizeof item, "%d", q->ci);
                if (cache_has(z, skey, item)) break;
                rg.count++; rg.len = q->off + q->len - rg.off;
            }
            ranges[nranges++] = rg; i += rg.count;
        }
        if (getenv("UFSM_Z3_PROF")) {
            size_t wanted = 0, span_b = 0;
            for (int i = 0; i < nitems; i++) if (items[i].present) wanted += (size_t)items[i].len;
            for (int i = 0; i < nranges; i++) if (items[ranges[i].first].present) span_b += (size_t)ranges[i].len;
            fprintf(stderr, "z3 grouped reads: %d chunks, %d tasks, %zu requested bytes, %zu range-span bytes\n",
                    nitems, nranges, wanted, span_b);
        }
    }
    job jb = {.z = z, .o = o, .n = n, .out = out, .items = items, .nitems = nitems,
              .ranges = ranges, .nranges = nranges, .submask = submask, .dsink = dsink};
    if (nthreads <= 0) nthreads = (int)ufsm_ncpu();
    int ntasks = ranges ? nranges : nitems;
    store_req *reqs = nullptr; int *task_of = nullptr; char (*rkeys)[1200] = nullptr; int nreq = 0;
    if (batch && ranges && ntasks) {
        /* which tasks need the network: coalesced groups, and single chunks in neither the disk nor the mask cache */
        const int mc_on = mc_budget() > 0 && m->esize == 1;
        jb.fetched = calloc((size_t)ntasks, sizeof *jb.fetched); jb.ready = malloc((size_t)ntasks * sizeof *jb.ready);
        reqs = malloc((size_t)ntasks * sizeof *reqs); task_of = malloc((size_t)ntasks * sizeof *task_of); rkeys = malloc((size_t)ntasks * sizeof *rkeys);
        if (!jb.fetched || !jb.ready || !reqs || !task_of || !rkeys) {
            free(jb.fetched); free(jb.ready); free(reqs); free(task_of); free(rkeys); free(ranges); free(items); return FAIL("batch allocation");
        }
        jb.batch = 1;
        pthread_mutex_init(&jb.rq_mu, nullptr); pthread_cond_init(&jb.rq_cv, nullptr);
        for (int t = 0; t < ntasks; t++) {
            const range_work *rg = &ranges[t]; work *w = &items[rg->first];
            char item[16]; snprintf(item, sizeof item, "%d", w->ci);
            shard_key(z, w->sz, w->sy, w->sx, rkeys[nreq], sizeof rkeys[nreq]);
            int need = rg->count > 1 || (w->present && !cache_has(z, rkeys[nreq], item) && !(mc_on && mc_has(z, w)));
            if (!need) { jb.ready[jb.rq_tail++] = t; continue; }
            reqs[nreq] = (store_req){rkeys[nreq], (int64_t)rg->off, (int64_t)rg->len, nullptr, -1};
            task_of[nreq++] = t;
        }
    }
    int nworkers = nthreads;
    if (nworkers > ntasks) nworkers = ntasks;
    if (nworkers < 1) nworkers = 1;
    if (nworkers > 256) nworkers = 256;
    pthread_t th[256];
    int started = 0;
    if (jb.batch) {
        /* every worker in its own thread; this thread drives the fetch in slices of at most UFSM_Z3_BATCH_MB (default 256) */
        for (int i = 0; i < nworkers; i++) {
            if (pthread_create(&th[started], nullptr, worker, &jb)) break;
            started++;
        }
        static int64_t slice_cap = -1;
        if (slice_cap < 0) { const char *v = getenv("UFSM_Z3_BATCH_MB"); slice_cap = (v && *v ? atoll(v) : 256) << 20; }
        batch_ud u = {&jb, nullptr, nullptr};
        for (int r0 = 0; r0 < nreq && !atomic_load(&jb.failed);) {
            int r1 = r0; int64_t bytes = 0;
            while (r1 < nreq && (r1 == r0 || bytes + reqs[r1].len <= slice_cap)) bytes += reqs[r1++].len;
            int ok = 1;
            for (int r = r0; r < r1; r++) if (!(reqs[r].out = malloc((size_t)reqs[r].len))) ok = 0;
            if (!ok) { FAIL("batch buffer allocation"); job_failed(&jb); for (int r = r0; r < r1; r++) free(reqs[r].out); break; }
            u.task_of = task_of + r0; u.reqs = reqs + r0;
            if (started) store_read_batch(z->s, reqs + r0, r1 - r0, 0, batch_done, &u);
            for (int r = r0; r < r1; r++) free(reqs[r].out);   /* leftovers of failed requests (passed buffers are nulled) */
            r0 = r1;
        }
        pthread_mutex_lock(&jb.rq_mu); jb.rq_closed = 1; pthread_cond_broadcast(&jb.rq_cv); pthread_mutex_unlock(&jb.rq_mu);
        if (!started) { FAIL("no decode worker thread"); job_failed(&jb); }
        for (int i = 0; i < started; i++) pthread_join(th[i], nullptr);
        for (int t = 0; t < ntasks; t++) free(jb.fetched[t]);
        pthread_mutex_destroy(&jb.rq_mu); pthread_cond_destroy(&jb.rq_cv);
        free(jb.fetched); free(jb.ready); free(reqs); free(task_of); free(rkeys);
    } else {
        for (int i = 1; i < nworkers; i++) {
            if (pthread_create(&th[started], nullptr, worker, &jb)) break;
            started++;
        }
        worker(&jb);
        for (int i = 0; i < started; i++) pthread_join(th[i], nullptr);
    }
    free(ranges); free(items);
    if (dmask) *dmask = atomic_load(&jb.dmask);
    return atomic_load(&jb.failed) ? FAIL("%s", jb.error) : 0;
}
