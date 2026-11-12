/* src/dcache.h: the derived-array chunk cache (memory LRU + disk, keyed by an inputs hash). */
#include "dcache.h"
#include <errno.h>
#include <openssl/evp.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <zstd.h>

typedef struct dent { int64_t key; uint8_t *data; uint8_t cval; int busy, failed, refs; struct dent *prev, *next, *hnext; } dent;
typedef struct {
    z3 *in; int C, fill; int64_t shape[3], gc[3]; char *dir;   /* dir: ROOT/<hash> or nullptr (memory only) */
    pthread_mutex_t mu; pthread_cond_t cv; dent **hash; size_t nhash; dent *head, *tail; size_t bytes, cap; _Atomic int disk_warned;
} dc;
static struct { _Atomic uint64_t reads, mem, disk, comp, fill, wbytes, ns_comp, ns_disk; } g_dc;
static double dnow(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
void dcache_get_stats(dcache_stats *st) {
    st->reads = g_dc.reads; st->cells_mem = g_dc.mem; st->cells_disk = g_dc.disk; st->cells_computed = g_dc.comp; st->cells_fill = g_dc.fill;
    st->disk_bytes_written = g_dc.wbytes; st->s_compute = 1e-9 * (double)g_dc.ns_comp; st->s_disk = 1e-9 * (double)g_dc.ns_disk;
}
const char *dcache_default_dir(void) {
    static char b[1024]; const char *e = getenv("UFSM_DERIVED_CACHE");
    if (e) return !*e || !strcmp(e, "off") || !strcmp(e, "0") ? nullptr : e;
    const char *h = getenv("HOME"); if (!h || !*h) return nullptr;
    snprintf(b, sizeof b, "%s/.cache/ufsm/derived", h); return b;
}
static int mkdirs(const char *p) {
    char b[1200]; snprintf(b, sizeof b, "%s", p);
    for (char *s = b + 1; *s; s++) if (*s == '/') { *s = 0; if (mkdir(b, 0775) && errno != EEXIST) return -1; *s = '/'; }
    return mkdir(b, 0775) && errno != EEXIST ? -1 : 0;
}
static size_t cbytes(const dc *c) { return (size_t)c->C * c->C * c->C; }
static void unlink_lru(dc *c, dent *e) { if (e->prev) e->prev->next = e->next; else c->head = e->next; if (e->next) e->next->prev = e->prev; else c->tail = e->prev; e->prev = e->next = nullptr; }
static void push_lru(dc *c, dent *e) { e->next = c->head; e->prev = nullptr; if (c->head) c->head->prev = e; c->head = e; if (!c->tail) c->tail = e; }
static size_t hb(const dc *c, int64_t k) { return (size_t)((uint64_t)k * 0x9E3779B97F4A7C15ull >> 32) % c->nhash; }
static void unhash(dc *c, dent *e) { dent **pp = &c->hash[hb(c, e->key)]; while (*pp && *pp != e) pp = &(*pp)->hnext; if (*pp) *pp = e->hnext; }
static void evict(dc *c) {   /* (mu held) drop unreferenced tail entries above the cap */
    for (dent *v = c->tail; v && c->bytes > c->cap;) {
        dent *pv = v->prev;
        if (!v->refs && !v->busy) { unlink_lru(c, v); unhash(c, v); c->bytes -= sizeof *v + (v->data ? cbytes(c) : 0); free(v->data); free(v); }
        v = pv;
    }
}
/* the disk file of a cell: "UDC1", u32 C, u8 kind (0 constant, 1 zstd), u8 value, u16 0, u32 payload bytes, payload */
static void cell_path(const dc *c, int64_t cz, int64_t cy, int64_t cx, char *p, size_t np) { snprintf(p, np, "%s/%lld/%lld.%lld", c->dir, (long long)cz, (long long)cy, (long long)cx); }
static int disk_get(const dc *c, const char *p, uint8_t **data, uint8_t *cval) {
    FILE *f = fopen(p, "rb"); if (!f) return -1;
    uint8_t h[16]; int rc = -1; uint8_t *zb = nullptr, *d = nullptr;
    if (fread(h, 1, 16, f) != 16 || memcmp(h, "UDC1", 4)) goto out;
    uint32_t C, nz; memcpy(&C, h + 4, 4); memcpy(&nz, h + 12, 4);
    if ((int)C != c->C) goto out;
    if (h[8] == 0) { *data = nullptr; *cval = h[9]; rc = 0; goto out; }
    if (h[8] != 1 || !(zb = malloc(nz ? nz : 1)) || fread(zb, 1, nz, f) != nz || !(d = malloc(cbytes(c)))) goto out;
    if (ZSTD_decompress(d, cbytes(c), zb, nz) != cbytes(c)) goto out;
    *data = d; d = nullptr; rc = 0;
out:
    fclose(f); free(zb); free(d); return rc;
}
static int disk_put(dc *c, int64_t cz, const char *p, const uint8_t *data, uint8_t cval) {
    char dd[1200], tmp[1300]; snprintf(dd, sizeof dd, "%s/%lld", c->dir, (long long)cz);
    if (mkdirs(dd)) return -1;
    snprintf(tmp, sizeof tmp, "%s.%d.%lx.tmp", p, (int)getpid(), (unsigned long)pthread_self());
    uint8_t h[16] = {'U', 'D', 'C', '1'}; const uint32_t C = (uint32_t)c->C; memcpy(h + 4, &C, 4);
    uint8_t *zb = nullptr; uint32_t nz = 0;
    if (data) {
        const size_t cap = ZSTD_compressBound(cbytes(c)); zb = malloc(cap); if (!zb) return -1;
        const size_t r = ZSTD_compress(zb, cap, data, cbytes(c), 3); if (ZSTD_isError(r)) { free(zb); return -1; }
        nz = (uint32_t)r; h[8] = 1;
    } else { h[8] = 0; h[9] = cval; }
    memcpy(h + 12, &nz, 4);
    FILE *f = fopen(tmp, "wb"); if (!f) { free(zb); return -1; }
    const int ok = fwrite(h, 1, 16, f) == 16 && (!nz || fwrite(zb, 1, nz, f) == nz); free(zb);
    if (fclose(f) || !ok || rename(tmp, p)) { unlink(tmp); return -1; }
    atomic_fetch_add(&g_dc.wbytes, 16 + nz);
    return 0;
}
/* one cell: a referenced entry (ready), or nullptr on failure */
static dent *cell_get(dc *c, int64_t cz, int64_t cy, int64_t cx, int nthreads) {
    const int64_t key = (cz * c->gc[1] + cy) * c->gc[2] + cx; const size_t h = hb(c, key);
    pthread_mutex_lock(&c->mu);
    dent *e = c->hash[h]; while (e && e->key != key) e = e->hnext;
    if (e) {
        e->refs++; while (e->busy) pthread_cond_wait(&c->cv, &c->mu);
        if (e->failed) { if (!--e->refs) free(e); pthread_mutex_unlock(&c->mu); return nullptr; }
        unlink_lru(c, e); push_lru(c, e); pthread_mutex_unlock(&c->mu); atomic_fetch_add(&g_dc.mem, 1); return e;
    }
    e = calloc(1, sizeof *e); e->key = key; e->busy = 1; e->refs = 1; e->hnext = c->hash[h]; c->hash[h] = e; push_lru(c, e);
    pthread_mutex_unlock(&c->mu);
    char p[1300]; int ok = 0; uint8_t *data = nullptr, cval = 0;
    if (c->dir) { cell_path(c, cz, cy, cx, p, sizeof p); const double t = dnow(); ok = !disk_get(c, p, &data, &cval); if (ok) { atomic_fetch_add(&g_dc.disk, 1); atomic_fetch_add(&g_dc.ns_disk, (uint64_t)((dnow() - t) * 1e9)); } }
    if (!ok) {
        const double t = dnow(); const int64_t o[3] = {cz * c->C, cy * c->C, cx * c->C}, n[3] = {c->C, c->C, c->C};
        data = malloc(cbytes(c));
        if (data && !z3_read(c->in, o, n, data, nthreads)) {
            ok = 1; size_t k = 1; const size_t nb = cbytes(c); while (k < nb && data[k] == data[0]) k++;
            if (k == nb) { cval = data[0]; free(data); data = nullptr; atomic_fetch_add(&g_dc.fill, 1); }   /* constant cells (most: no labels in reach) */
            atomic_fetch_add(&g_dc.comp, 1); atomic_fetch_add(&g_dc.ns_comp, (uint64_t)((dnow() - t) * 1e9));
            if (c->dir && disk_put(c, cz, p, data, cval) && !atomic_exchange(&c->disk_warned, 1)) fprintf(stderr, "dcache: cannot write %s (%s): memory only from here\n", p, strerror(errno));
        } else { free(data); data = nullptr; }
    }
    pthread_mutex_lock(&c->mu);
    e->busy = 0;
    if (!ok) { e->failed = 1; unlink_lru(c, e); unhash(c, e); pthread_cond_broadcast(&c->cv); if (!--e->refs) free(e); pthread_mutex_unlock(&c->mu); return nullptr; }
    e->data = data; e->cval = cval; c->bytes += sizeof *e + (data ? cbytes(c) : 0); evict(c);
    pthread_cond_broadcast(&c->cv); pthread_mutex_unlock(&c->mu);
    return e;
}
static void cell_put(dc *c, dent *e) { pthread_mutex_lock(&c->mu); if (!--e->refs && e->failed) free(e); else evict(c); pthread_mutex_unlock(&c->mu); }
static int dc_read(void *ctx, z3 *base, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads) {
    (void)base; dc *c = ctx; atomic_fetch_add(&g_dc.reads, 1);
    memset(out, c->fill, (size_t)n[0] * n[1] * n[2]);
    int64_t c0[3], c1[3];
    for (int d = 0; d < 3; d++) {
        if (o[d] + n[d] <= 0 || o[d] >= c->shape[d]) return 0;
        c0[d] = o[d] < 0 ? 0 : o[d] / c->C; c1[d] = (o[d] + n[d] - 1) / c->C; if (c1[d] >= c->gc[d]) c1[d] = c->gc[d] - 1;
    }
    const int64_t C = c->C;
    for (int64_t cz = c0[0]; cz <= c1[0]; cz++) for (int64_t cy = c0[1]; cy <= c1[1]; cy++) for (int64_t cx = c0[2]; cx <= c1[2]; cx++) {
        dent *e = cell_get(c, cz, cy, cx, nthreads); if (!e) return -1;
        int64_t a[3], b[3]; const int64_t co[3] = {cz * C, cy * C, cx * C};   /* the overlap [a, b) in array coordinates */
        for (int d = 0; d < 3; d++) { a[d] = co[d] > o[d] ? co[d] : o[d]; b[d] = co[d] + C < o[d] + n[d] ? co[d] + C : o[d] + n[d]; if (b[d] > c->shape[d]) b[d] = c->shape[d]; }
        if (e->data || e->cval != (uint8_t)c->fill)
            for (int64_t z = a[0]; z < b[0]; z++) for (int64_t y = a[1]; y < b[1]; y++) {
                uint8_t *dst = out + ((size_t)(z - o[0]) * n[1] + (y - o[1])) * n[2] + (a[2] - o[2]);
                if (e->data) memcpy(dst, e->data + ((size_t)(z - co[0]) * C + (y - co[1])) * C + (a[2] - co[2]), (size_t)(b[2] - a[2]));
                else memset(dst, e->cval, (size_t)(b[2] - a[2]));
            }
        cell_put(c, e);
    }
    return 0;
}
static void dc_free(void *p) {
    dc *c = p; if (!c) return;
    for (size_t i = 0; i < c->nhash; i++) for (dent *e = c->hash[i], *nx; e; e = nx) { nx = e->hnext; free(e->data); free(e); }
    free(c->hash); free(c->dir); pthread_mutex_destroy(&c->mu); pthread_cond_destroy(&c->cv); free(c);
}
int dcache_key_dir(const char *dir, const char *key, char *out, size_t nout) {
    unsigned char hsh[32]; unsigned nh = 0; char hex[65];
    if (!dir || !*dir || !key || !EVP_Digest(key, strlen(key), hsh, &nh, EVP_sha256(), nullptr) || nh != 32) return -1;
    for (int i = 0; i < 32; i++) snprintf(hex + 2 * i, 3, "%02x", hsh[i]);
    snprintf(out, nout, "%s/%.24s", dir, hex);
    if (mkdirs(out)) { fprintf(stderr, "dcache: cannot create %s (%s): memory only\n", out, strerror(errno)); return -1; }
    char ip[1300]; snprintf(ip, sizeof ip, "%s/inputs.txt", out); struct stat st;
    if (stat(ip, &st)) { char tmp[1400]; snprintf(tmp, sizeof tmp, "%s.%d.tmp", ip, (int)getpid()); FILE *f = fopen(tmp, "wb");
        if (f) { fprintf(f, "%s\n", key); if (fclose(f) || rename(tmp, ip)) unlink(tmp); } }
    return 0;
}
z3 *z3_open_cached(z3 *inner, const char *dir, const char *key, int cell, size_t mem_mb) {
    if (!inner) return nullptr;
    const z3_meta *m = z3_meta_of(inner);
    dc *c = calloc(1, sizeof *c); c->in = inner; c->C = cell > 0 ? cell : 128; c->fill = m->fill;
    for (int d = 0; d < 3; d++) { c->shape[d] = m->shape[d]; c->gc[d] = (m->shape[d] + c->C - 1) / c->C; }
    c->nhash = 1 << 16; c->hash = calloc(c->nhash, sizeof *c->hash); c->cap = (mem_mb ? mem_mb : 1024) << 20;
    pthread_mutex_init(&c->mu, nullptr); pthread_cond_init(&c->cv, nullptr);
    char d[1200];
    if (dir && *dir && key && !dcache_key_dir(dir, key, d, sizeof d)) c->dir = strdup(d);
    z3 *z = z3_open_derived(inner, c->fill, dc_read, c, dc_free);
    if (!z) { dc_free(c); z3_close(inner); }
    return z;
}
