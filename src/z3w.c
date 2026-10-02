#include "z3w.h"
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
};

static int mkdirs(const char *path) {
    char tmp[1400];
    snprintf(tmp, sizeof tmp, "%s", path);
    for (char *p = tmp + 1; *p; p++) if (*p == '/') { *p = 0; if (mkdir(tmp, 0755) && errno != EEXIST) return -1; *p = '/'; }
    return mkdir(tmp, 0755) && errno != EEXIST ? -1 : 0;
}

void z3w_level_name(double um, char *buf, int n) { snprintf(buf, (size_t)n, "%.10g", um); }

z3w *z3w_create(const char *dir, const int64_t shape[3], int shard, float q, int fill, const char *attrs_json) {
    if (shard % 128 || shard <= 0) { fail("shard must be a multiple of 128"); return nullptr; }
    if (mkdirs(dir)) { fail("mkdir %s: %s", dir, strerror(errno)); return nullptr; }
    char p[1400];
    snprintf(p, sizeof p, "%s/zarr.json", dir);
    FILE *f = fopen(p, "w");
    if (!f) { fail("cannot write %s", p); return nullptr; }
    fprintf(f,
        "{\"zarr_format\":3,\"node_type\":\"array\",\"shape\":[%lld,%lld,%lld],\"data_type\":\"uint8\","
        "\"chunk_grid\":{\"name\":\"regular\",\"configuration\":{\"chunk_shape\":[%d,%d,%d]}},"
        "\"chunk_key_encoding\":{\"name\":\"default\",\"configuration\":{\"separator\":\"/\"}},\"fill_value\":%d,"
        "\"codecs\":[{\"name\":\"sharding_indexed\",\"configuration\":{\"chunk_shape\":[128,128,128],"
        "\"codecs\":[{\"name\":\"volcomp\",\"configuration\":{\"q\":%g}}],"
        "\"index_codecs\":[{\"name\":\"bytes\",\"configuration\":{\"endian\":\"little\"}},{\"name\":\"crc32c\"}],\"index_location\":\"end\"}}],"
        "\"attributes\":%s,\"dimension_names\":[\"z\",\"y\",\"x\"]}\n",
        (long long)shape[0], (long long)shape[1], (long long)shape[2], shard, shard, shard, fill, (double)q, attrs_json ? attrs_json : "{}");
    fclose(f);
    z3w *w = calloc(1, sizeof *w);
    w->dir = strdup(dir);
    memcpy(w->shape, shape, sizeof w->shape);
    w->shard = shard;
    w->cg = shard / 128;
    w->q = q;
    w->fill = fill;
    return w;
}

typedef struct { const z3w *w; const uint8_t *data; uint8_t **enc; size_t *len; int nc; atomic_int next, failed; } job;

static void *worker(void *arg) {
    job *jb = arg;
    const z3w *w = jb->w;
    uint8_t *chunk = malloc(VOLCOMP_CHUNK_VOXELS);
    uint8_t *enc = malloc(VOLCOMP_ENCODE_BOUND);
    for (;;) {
        int i = atomic_fetch_add(&jb->next, 1);
        if (i >= jb->nc || atomic_load(&jb->failed)) break;
        int cz = i / (w->cg * w->cg), cy = (i / w->cg) % w->cg, cx = i % w->cg;
        int any = 0;
        for (int z = 0; z < 128; z++) for (int y = 0; y < 128; y++) {
            const uint8_t *src = jb->data + ((size_t)(cz * 128 + z) * w->shard + (cy * 128 + y)) * w->shard + cx * 128;
            uint8_t *dst = chunk + ((size_t)z * 128 + y) * 128;
            memcpy(dst, src, 128);
            if (!any) for (int x = 0; x < 128; x++) if (src[x] != (uint8_t)w->fill) { any = 1; break; }
        }
        if (!any) { jb->enc[i] = nullptr; jb->len[i] = 0; continue; }
        size_t n;
        volcomp_status st = volcomp_encode(chunk, w->q, enc, VOLCOMP_ENCODE_BOUND, &n);
        if (st != VOLCOMP_OK) { fail("volcomp encode: %s", volcomp_status_string(st)); atomic_store(&jb->failed, 1); break; }
        jb->enc[i] = malloc(n);
        memcpy(jb->enc[i], enc, n);
        jb->len[i] = n;
    }
    free(chunk); free(enc);
    return nullptr;
}

int z3w_write_shard(z3w *w, int64_t sz, int64_t sy, int64_t sx, const uint8_t *data, int nthreads) {
    int nc = w->cg * w->cg * w->cg;
    job jb = {w, data, calloc((size_t)nc, sizeof(uint8_t *)), calloc((size_t)nc, sizeof(size_t)), nc, 0, 0};
    if (nthreads <= 0) nthreads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (nthreads > nc) nthreads = nc;
    if (nthreads > 256) nthreads = 256;
    pthread_t th[256];
    for (int i = 1; i < nthreads; i++) pthread_create(&th[i], nullptr, worker, &jb);
    worker(&jb);
    for (int i = 1; i < nthreads; i++) pthread_join(th[i], nullptr);
    int rc = atomic_load(&jb.failed) ? -1 : 0;
    if (!rc) {
        int any = 0;
        for (int i = 0; i < nc; i++) any |= jb.len[i] > 0;
        if (any) {
            char p[1400], tmp[1500];
            snprintf(p, sizeof p, "%s/c/%lld/%lld", w->dir, (long long)sz, (long long)sy);
            mkdirs(p);
            snprintf(p, sizeof p, "%s/c/%lld/%lld/%lld", w->dir, (long long)sz, (long long)sy, (long long)sx);
            snprintf(tmp, sizeof tmp, "%s.tmp", p);
            FILE *f = fopen(tmp, "wb");
            if (!f) rc = fail("cannot write %s", tmp);
            else {
                uint8_t *idx = malloc((size_t)nc * 16 + 4);
                uint64_t off = 0;
                for (int i = 0; i < nc; i++) {
                    uint64_t o = jb.len[i] ? off : UINT64_MAX, l = jb.len[i] ? jb.len[i] : UINT64_MAX;
                    memcpy(idx + (size_t)i * 16, &o, 8);
                    memcpy(idx + (size_t)i * 16 + 8, &l, 8);
                    if (jb.len[i]) { if (fwrite(jb.enc[i], 1, jb.len[i], f) != jb.len[i]) rc = fail("write %s", tmp); off += jb.len[i]; }
                }
                uint32_t crc = crc32c(0, idx, (size_t)nc * 16);
                memcpy(idx + (size_t)nc * 16, &crc, 4);
                if (fwrite(idx, 1, (size_t)nc * 16 + 4, f) != (size_t)nc * 16 + 4) rc = fail("write %s", tmp);
                free(idx);
                fclose(f);
                if (!rc && rename(tmp, p)) rc = fail("rename %s", p);
            }
        }
    }
    for (int i = 0; i < nc; i++) free(jb.enc[i]);
    free(jb.enc); free(jb.len);
    return rc;
}

int z3w_close(z3w *w) { if (w) { free(w->dir); free(w); } return 0; }

int z3w_write_group(const char *dir, const double *um, int n, const char *name, const char *attrs_json) {
    if (mkdirs(dir)) return fail("mkdir %s", dir);
    char p[1400];
    snprintf(p, sizeof p, "%s/zarr.json", dir);
    FILE *f = fopen(p, "w");
    if (!f) return fail("cannot write %s", p);
    fprintf(f, "{\"zarr_format\":3,\"node_type\":\"group\",\"attributes\":{\"ome\":{\"version\":\"0.5\",\"multiscales\":[{\"version\":\"0.5\",\"name\":\"%s\","
               "\"axes\":[{\"name\":\"z\",\"type\":\"space\",\"unit\":\"micrometer\"},{\"name\":\"y\",\"type\":\"space\",\"unit\":\"micrometer\"},{\"name\":\"x\",\"type\":\"space\",\"unit\":\"micrometer\"}],\"datasets\":[", name);
    for (int i = 0; i < n; i++) {
        char lv[32]; z3w_level_name(um[i], lv, sizeof lv);
        fprintf(f, "%s{\"path\":\"%s\",\"coordinateTransformations\":[{\"type\":\"scale\",\"scale\":[%.10g,%.10g,%.10g]}]}", i ? "," : "", lv, um[i], um[i], um[i]);
    }
    fprintf(f, "]}]}");
    if (attrs_json && attrs_json[0] == '{' && attrs_json[1] != '}') fprintf(f, ",%s", attrs_json + 1); else fprintf(f, "}");
    fprintf(f, "}\n");
    fclose(f);
    return 0;
}
