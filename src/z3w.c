#include "z3w.h"
#include "sysinfo.h"
#include <errno.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "volcomp.h"

static _Thread_local char errbuf[256];
const char *z3w_error(void) { return errbuf; }
static int fail(const char *fmt, ...) { va_list ap; va_start(ap, fmt); vsnprintf(errbuf, sizeof errbuf, fmt, ap); va_end(ap); return -1; }

static uint32_t crc_table[256];
static pthread_once_t crc_once = PTHREAD_ONCE_INIT;
static void crc_init(void) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) c = c & 1 ? 0x82F63B78u ^ (c >> 1) : c >> 1;
        crc_table[i] = c;
    }
}
uint32_t crc32c(uint32_t crc, const void *buf, size_t len) {
    pthread_once(&crc_once, crc_init);
    const uint8_t *p = buf;
    crc = ~crc;
    for (size_t i = 0; i < len; i++) crc = crc_table[(crc ^ p[i]) & 0xff] ^ (crc >> 8);
    return ~crc;
}

struct z3w {
    char *dir;
    int64_t shape[3];
    int shard, cg;      /* inner chunk grid per axis */
    float q;
    int fill;
    int mask_lossless;
    int dtype, esize;   /* typed arrays (volcomp_dtype, element bytes); 0 / 1 = uint8 */
    double fillv;
    int crc;   /* the inner chain ends in crc32c: 4 bytes after each encoded chunk (UFSM_Z3W_CRC=1; readers before 2026-11-14 refuse such stores) */
};
static const char *dtype_name(int dt) { static const char *n[6] = {"uint8", "uint16", "int16", "uint32", "int32", "float32"}; return dt >= 0 && dt < 6 ? n[dt] : "?"; }

static int mkdirs(const char *path) {
    char tmp[1400];
    snprintf(tmp, sizeof tmp, "%s", path);
    for (char *p = tmp + 1; *p; p++) if (*p == '/') { *p = 0; if (mkdir(tmp, 0755) && errno != EEXIST) return -1; *p = '/'; }
    return mkdir(tmp, 0755) && errno != EEXIST ? -1 : 0;
}

/* durable, atomic file writes (2026-11-14 audit mini/zarr 2, 3, 5): a unique temp name per writer (pid + thread: two writers of one path
   never share an inode), every write and the close checked, fsync before the rename (UFSM_Z3W_FSYNC=0: no fsync), the temp unlinked on any
   failure. Before, `<p>.tmp` was shared by every writer, fclose's flush error was ignored and the short file renamed into place, and
   zarr.json was rewritten in place unchecked (a reader could see it half written; a full disk returned success) */
/* per-chunk crc32c (zarr v3 codec, after volcomp): volcomp has no checksum, so a flipped bit in a payload decodes as wrong voxels
   (audit mini/volcomp 1). Opt-in (UFSM_Z3W_CRC=1) because older ufsm binaries reject the codec; the reader checks it whenever present */
static int crc_on(void) { const char *e = getenv("UFSM_Z3W_CRC"); return e && atoi(e) != 0; }
#define CRC_CODEC (crc_on() ? ",{\"name\":\"crc32c\"}" : "")
static int fsync_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("UFSM_Z3W_FSYNC"); on = e ? atoi(e) != 0 : 1; } return on; }
static void tmp_name(char *tmp, size_t n, const char *p) { snprintf(tmp, n, "%s.%d.%lx.tmp", p, (int)getpid(), (unsigned long)pthread_self()); }
static FILE *atomic_open(const char *p, char *tmp, size_t n) { tmp_name(tmp, n, p); FILE *f = fopen(tmp, "wb"); if (!f) fail("cannot write %s: %s", tmp, strerror(errno)); return f; }
static int atomic_close(FILE *f, const char *tmp, const char *p, int rc) {   /* rc: the writes so far (0 ok); publishes p only if all of it held */
    if (!rc && (ferror(f) || fflush(f))) rc = fail("write %s: %s", tmp, strerror(errno));
    if (!rc && fsync_on() && fsync(fileno(f))) rc = fail("fsync %s: %s", tmp, strerror(errno));
    if (fclose(f) && !rc) rc = fail("close %s: %s", tmp, strerror(errno));
    if (!rc && rename(tmp, p)) rc = fail("rename %s -> %s: %s", tmp, p, strerror(errno));
    if (rc) unlink(tmp);
    return rc;
}

void z3w_level_name(double um, char *buf, int n) { snprintf(buf, (size_t)n, "%.10g", um); }

z3w *z3w_create(const char *dir, const int64_t shape[3], int shard, float q, int fill, const char *attrs_json) {
    if (shard % 128 || shard <= 0) { fail("shard must be a multiple of 128"); return nullptr; }
    if (mkdirs(dir)) { fail("mkdir %s: %s", dir, strerror(errno)); return nullptr; }
    char p[1400];
    snprintf(p, sizeof p, "%s/zarr.json", dir);
    char tmp[1500]; FILE *f = atomic_open(p, tmp, sizeof tmp);
    if (!f) return nullptr;
    fprintf(f,
        "{\"zarr_format\":3,\"node_type\":\"array\",\"shape\":[%lld,%lld,%lld],\"data_type\":\"uint8\","
        "\"chunk_grid\":{\"name\":\"regular\",\"configuration\":{\"chunk_shape\":[%d,%d,%d]}},"
        "\"chunk_key_encoding\":{\"name\":\"default\",\"configuration\":{\"separator\":\"/\"}},\"fill_value\":%d,"
        "\"codecs\":[{\"name\":\"sharding_indexed\",\"configuration\":{\"chunk_shape\":[128,128,128],"
        "\"codecs\":[{\"name\":\"volcomp\",\"configuration\":{\"q\":%g}}%s],"
        "\"index_codecs\":[{\"name\":\"bytes\",\"configuration\":{\"endian\":\"little\"}},{\"name\":\"crc32c\"}],\"index_location\":\"end\"}}],"
        "\"attributes\":%s,\"dimension_names\":[\"z\",\"y\",\"x\"]}\n",
        (long long)shape[0], (long long)shape[1], (long long)shape[2], shard, shard, shard, fill, (double)q, CRC_CODEC, attrs_json ? attrs_json : "{}");
    if (atomic_close(f, tmp, p, 0)) return nullptr;
    z3w *w = calloc(1, sizeof *w);
    w->dir = strdup(dir);
    memcpy(w->shape, shape, sizeof w->shape);
    w->shard = shard;
    w->cg = shard / 128;
    w->q = q;
    w->fill = fill;
    w->crc = crc_on();
    return w;
}

z3w *z3w_create_codec(const char *dir, const int64_t shape[3], int shard, const char *codec_json, int fill, const char *attrs_json) {
    if (shard % 128 || shard <= 0) { fail("shard must be a multiple of 128"); return nullptr; }
    if (mkdirs(dir)) { fail("mkdir %s: %s", dir, strerror(errno)); return nullptr; }
    char p[1400];
    snprintf(p, sizeof p, "%s/zarr.json", dir);
    char tmp[1500]; FILE *f = atomic_open(p, tmp, sizeof tmp);
    if (!f) return nullptr;
    fprintf(f,
        "{\"zarr_format\":3,\"node_type\":\"array\",\"shape\":[%lld,%lld,%lld],\"data_type\":\"uint8\","
        "\"chunk_grid\":{\"name\":\"regular\",\"configuration\":{\"chunk_shape\":[%d,%d,%d]}},"
        "\"chunk_key_encoding\":{\"name\":\"default\",\"configuration\":{\"separator\":\"/\"}},\"fill_value\":%d,"
        "\"codecs\":[{\"name\":\"sharding_indexed\",\"configuration\":{\"chunk_shape\":[128,128,128],"
        "\"codecs\":[%s],"
        "\"index_codecs\":[{\"name\":\"bytes\",\"configuration\":{\"endian\":\"little\"}},{\"name\":\"crc32c\"}],\"index_location\":\"end\"}}],"
        "\"attributes\":%s,\"dimension_names\":[\"z\",\"y\",\"x\"]}\n",
        (long long)shape[0], (long long)shape[1], (long long)shape[2], shard, shard, shard, fill, codec_json, attrs_json ? attrs_json : "{}");
    if (atomic_close(f, tmp, p, 0)) return nullptr;
    z3w *w = calloc(1, sizeof *w);
    w->dir = strdup(dir); memcpy(w->shape, shape, sizeof w->shape);
    w->shard = shard; w->cg = shard / 128; w->q = -1; w->fill = fill;
    return w;
}

z3w *z3w_create_typed(const char *dir, const int64_t shape[3], int shard, int dtype, float q, double fill, const char *attrs_json) {
    if (dtype < 1 || dtype > 5) { fail("typed array: dtype %d", dtype); return nullptr; }
    if (shard % 128 || shard <= 0) { fail("shard must be a multiple of 128"); return nullptr; }
    if (mkdirs(dir)) { fail("mkdir %s: %s", dir, strerror(errno)); return nullptr; }
    char p[1400];
    snprintf(p, sizeof p, "%s/zarr.json", dir);
    char tmp[1500]; FILE *f = atomic_open(p, tmp, sizeof tmp);
    if (!f) return nullptr;
    fprintf(f,
        "{\"zarr_format\":3,\"node_type\":\"array\",\"shape\":[%lld,%lld,%lld],\"data_type\":\"%s\","
        "\"chunk_grid\":{\"name\":\"regular\",\"configuration\":{\"chunk_shape\":[%d,%d,%d]}},"
        "\"chunk_key_encoding\":{\"name\":\"default\",\"configuration\":{\"separator\":\"/\"}},\"fill_value\":%.17g,"
        "\"codecs\":[{\"name\":\"sharding_indexed\",\"configuration\":{\"chunk_shape\":[128,128,128],"
        "\"codecs\":[{\"name\":\"volcomp\",\"configuration\":{\"q\":%.9g}}%s],"
        "\"index_codecs\":[{\"name\":\"bytes\",\"configuration\":{\"endian\":\"little\"}},{\"name\":\"crc32c\"}],\"index_location\":\"end\"}}],"
        "\"attributes\":%s,\"dimension_names\":[\"z\",\"y\",\"x\"]}\n",
        (long long)shape[0], (long long)shape[1], (long long)shape[2], dtype_name(dtype), shard, shard, shard, fill, (double)q, CRC_CODEC, attrs_json ? attrs_json : "{}");
    if (atomic_close(f, tmp, p, 0)) return nullptr;
    z3w *w = calloc(1, sizeof *w);
    w->dir = strdup(dir); memcpy(w->shape, shape, sizeof w->shape);
    w->shard = shard; w->cg = shard / 128; w->q = q; w->dtype = dtype; w->esize = (int)volcomp_dtype_size((volcomp_dtype)dtype); w->fillv = fill; w->crc = crc_on();
    return w;
}

z3w *z3w_create_mask(const char *dir, const int64_t shape[3], int shard, const char *attrs_json) {
    z3w *w = z3w_create(dir, shape, shard, 0, 0, attrs_json);
    if (w) w->mask_lossless = 1;
    return w;
}

typedef struct { const z3w *w; const uint8_t *data; uint8_t **enc; size_t *len; int nc; atomic_int next, failed; } job;

static int elem_is_fill(const uint8_t *e, int dtype, double fillv) {   /* one element equals the array's fill value */
    switch (dtype) {
    case 1: { uint16_t v; memcpy(&v, e, 2); return v == (uint16_t)fillv; }
    case 2: { int16_t v; memcpy(&v, e, 2); return v == (int16_t)fillv; }
    case 3: { uint32_t v; memcpy(&v, e, 4); return v == (uint32_t)fillv; }
    case 4: { int32_t v; memcpy(&v, e, 4); return v == (int32_t)fillv; }
    case 5: { float v; memcpy(&v, e, 4); return v == (float)fillv; }
    }
    return 0;
}
static int encode_one(const z3w *w, const uint8_t *chunk, uint8_t *enc, size_t bound, uint8_t **out, size_t *len) {
    size_t n;
    volcomp_status st = w->dtype ? volcomp_encode_typed(chunk, (volcomp_dtype)w->dtype, w->q, enc, bound, &n)
        : w->mask_lossless ? volcomp_mask_encode_lossless(chunk, enc, VOLCOMP_ENCODE_BOUND, &n)
        : volcomp_encode(chunk, w->q, enc, VOLCOMP_ENCODE_BOUND, &n);
    if (st != VOLCOMP_OK) return fail("volcomp encode: %s", volcomp_status_string(st));
    if (!(*out = malloc(n))) return fail("encode allocation");
    memcpy(*out, enc, n); *len = n;
    return 0;
}
static void *worker(void *arg) {
    job *jb = arg;
    const z3w *w = jb->w;
    const int es = w->dtype ? w->esize : 1;
    const size_t bound = w->dtype ? VOLCOMP_TYPED_ENCODE_BOUND : VOLCOMP_ENCODE_BOUND;
    uint8_t *chunk = malloc((size_t)VOLCOMP_CHUNK_VOXELS * es);
    uint8_t *enc = malloc(bound);
    for (;;) {
        int i = atomic_fetch_add(&jb->next, 1);
        if (i >= jb->nc || atomic_load(&jb->failed)) break;
        int cz = i / (w->cg * w->cg), cy = (i / w->cg) % w->cg, cx = i % w->cg;
        int any = 0;
        for (int z = 0; z < 128; z++) for (int y = 0; y < 128; y++) {
            const uint8_t *src = jb->data + (((size_t)(cz * 128 + z) * w->shard + (cy * 128 + y)) * w->shard + cx * 128) * es;
            uint8_t *dst = chunk + ((size_t)z * 128 + y) * 128 * es;
            memcpy(dst, src, (size_t)128 * es);
            if (!any) for (int x = 0; x < 128; x++) if (w->dtype ? !elem_is_fill(src + (size_t)x * es, w->dtype, w->fillv) : src[x] != (uint8_t)w->fill) { any = 1; break; }
        }
        if (!any) { jb->enc[i] = nullptr; jb->len[i] = 0; continue; }
        if (encode_one(w, chunk, enc, bound, &jb->enc[i], &jb->len[i])) { atomic_store(&jb->failed, 1); break; }
    }
    free(chunk); free(enc);
    return nullptr;
}

static int write_file(const z3w *w, int64_t sz, int64_t sy, int64_t sx, uint8_t **enc, size_t *len, int nc);
int z3w_write_shard(z3w *w, int64_t sz, int64_t sy, int64_t sx, const uint8_t *data, int nthreads) {
    int nc = w->cg * w->cg * w->cg;
    job jb = {w, data, calloc((size_t)nc, sizeof(uint8_t *)), calloc((size_t)nc, sizeof(size_t)), nc, 0, 0};
    if (nthreads <= 0) nthreads = (int)ufsm_ncpu();
    if (nthreads > nc) nthreads = nc;
    if (nthreads > 256) nthreads = 256;
    pthread_t th[256]; int made = 1;   /* (only the threads that started are joined: a failed pthread_create left th[i] undefined) */
    for (int i = 1; i < nthreads; i++) { if (pthread_create(&th[made], nullptr, worker, &jb)) break; made++; }
    worker(&jb);
    for (int i = 1; i < made; i++) pthread_join(th[i], nullptr);
    int rc = atomic_load(&jb.failed) ? -1 : 0;
    if (!rc) rc = write_file(w, sz, sy, sx, jb.enc, jb.len, nc);
    for (int i = 0; i < nc; i++) free(jb.enc[i]);
    free(jb.enc); free(jb.len);
    return rc;
}
int z3w_shard_chunks(const z3w *w) { return w->cg * w->cg * w->cg; }
int z3w_encode_chunk(const z3w *w, const uint8_t *chunk, uint8_t **enc, size_t *len) {
    const int es = w->dtype ? w->esize : 1; const size_t nv = (size_t)VOLCOMP_CHUNK_VOXELS;
    *enc = nullptr; *len = 0;
    int any = 0;
    for (size_t i = 0; i < nv && !any; i++) any = w->dtype ? !elem_is_fill(chunk + i * es, w->dtype, w->fillv) : chunk[i] != (uint8_t)w->fill;
    if (!any) return 0;
    const size_t bound = w->dtype ? VOLCOMP_TYPED_ENCODE_BOUND : VOLCOMP_ENCODE_BOUND;
    uint8_t *e = malloc(bound); if (!e) return fail("encode allocation");
    const int rc = encode_one(w, chunk, e, bound, enc, len);
    free(e); return rc;
}
int z3w_write_shard_enc(z3w *w, int64_t sz, int64_t sy, int64_t sx, uint8_t **enc, size_t *len) {
    const int nc = w->cg * w->cg * w->cg;
    const int rc = write_file(w, sz, sy, sx, enc, len, nc);
    for (int i = 0; i < nc; i++) { free(enc[i]); enc[i] = nullptr; len[i] = 0; }
    return rc;
}
/* the shard object from its encoded chunks (none: no object) */
static int write_file(const z3w *w, int64_t sz, int64_t sy, int64_t sx, uint8_t **enc, size_t *len, int nc) {
    int rc = 0, any = 0;
    for (int i = 0; i < nc; i++) any |= len[i] > 0;
    char p[1400], tmp[1500];
    snprintf(p, sizeof p, "%s/c/%lld/%lld/%lld", w->dir, (long long)sz, (long long)sy, (long long)sx);
    if (!any) {   /* an all-fill shard is no object: an older one at this path (a rewrite into an existing store) goes, or it would be read as data */
        if (unlink(p) && errno != ENOENT) return fail("unlink %s: %s", p, strerror(errno));
        return 0;
    }
    { char d[1400]; snprintf(d, sizeof d, "%s/c/%lld/%lld", w->dir, (long long)sz, (long long)sy); if (mkdirs(d)) return fail("mkdir %s: %s", d, strerror(errno)); }
    FILE *f = atomic_open(p, tmp, sizeof tmp);
    if (!f) return -1;
    uint8_t *idx = malloc((size_t)nc * 16 + 4);
    if (!idx) { fclose(f); unlink(tmp); return fail("index allocation"); }
    uint64_t off = 0;
    for (int i = 0; i < nc; i++) {
        const size_t cl = len[i] ? len[i] + (w->crc ? 4 : 0) : 0;   /* (crc32c: after each chunk, whoever encoded it: z3w or predict's GPU coder) */
        uint64_t o = cl ? off : UINT64_MAX, l = cl ? cl : UINT64_MAX;
        memcpy(idx + (size_t)i * 16, &o, 8);
        memcpy(idx + (size_t)i * 16 + 8, &l, 8);
        if (len[i] && !rc) {
            if (fwrite(enc[i], 1, len[i], f) != len[i]) rc = fail("write %s: %s", tmp, strerror(errno));
            if (w->crc && !rc) { const uint32_t c = crc32c(0, enc[i], len[i]); if (fwrite(&c, 1, 4, f) != 4) rc = fail("write %s: %s", tmp, strerror(errno)); }
            off += cl;
        }
    }
    uint32_t crc = crc32c(0, idx, (size_t)nc * 16);
    memcpy(idx + (size_t)nc * 16, &crc, 4);
    if (!rc && fwrite(idx, 1, (size_t)nc * 16 + 4, f) != (size_t)nc * 16 + 4) rc = fail("write %s: %s", tmp, strerror(errno));
    free(idx);
    return atomic_close(f, tmp, p, rc);   /* (checks the final flush: the index tail sat in the stdio buffer, and its ENOSPC was lost) */
}

int z3w_close(z3w *w) { if (w) { free(w->dir); free(w); } return 0; }

int z3w_write_group(const char *dir, const double *um, int n, const char *name, const char *attrs_json) {
    if (mkdirs(dir)) return fail("mkdir %s", dir);
    char p[1400];
    snprintf(p, sizeof p, "%s/zarr.json", dir);
    char tmp[1500]; FILE *f = atomic_open(p, tmp, sizeof tmp);
    if (!f) return -1;
    fprintf(f, "{\"zarr_format\":3,\"node_type\":\"group\",\"attributes\":{\"ome\":{\"version\":\"0.5\",\"multiscales\":[{\"version\":\"0.5\",\"name\":\"%s\","
               "\"axes\":[{\"name\":\"z\",\"type\":\"space\",\"unit\":\"micrometer\"},{\"name\":\"y\",\"type\":\"space\",\"unit\":\"micrometer\"},{\"name\":\"x\",\"type\":\"space\",\"unit\":\"micrometer\"}],\"datasets\":[", name);
    for (int i = 0; i < n; i++) {
        char lv[32]; z3w_level_name(um[i], lv, sizeof lv);
        fprintf(f, "%s{\"path\":\"%s\",\"coordinateTransformations\":[{\"type\":\"scale\",\"scale\":[%.10g,%.10g,%.10g]}]}", i ? "," : "", lv, um[i], um[i], um[i]);
    }
    fprintf(f, "]}]}");
    if (attrs_json && attrs_json[0] == '{' && attrs_json[1] != '}') fprintf(f, ",%s", attrs_json + 1); else fprintf(f, "}");
    fprintf(f, "}\n");
    return atomic_close(f, tmp, p, 0);
}
