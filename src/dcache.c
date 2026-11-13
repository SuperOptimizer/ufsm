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

struct dc;
typedef struct dent { int64_t key; uint8_t *data; uint8_t cval; int busy, failed, refs; struct dc *own; struct dent *prev, *next, *hnext; } dent;
/* the memory LRU and its cap: one per cached array, or one shared by several (dcache_pool_shared: the on-the-fly ink targets of every
   source under one budget); the pool's mutex guards the LRU and every member array's hash */
struct dcache_pool { pthread_mutex_t mu; pthread_cond_t cv; dent *head, *tail; size_t bytes, cap; int shared; };
typedef struct dc {
    unsigned magic; int batch;   /* batch: the edge (voxels) up to which a read's missed cells are computed as one box (dcache_set_batch) */
    z3 *in; int C, fill; int64_t shape[3], gc[3]; char *dir;   /* dir: ROOT/<hash> or nullptr (memory only) */
    dcache_pool *P; dent **hash; size_t nhash; _Atomic int disk_warned;
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
static void unlink_lru(dcache_pool *c, dent *e) { if (e->prev) e->prev->next = e->next; else c->head = e->next; if (e->next) e->next->prev = e->prev; else c->tail = e->prev; e->prev = e->next = nullptr; }
static void push_lru(dcache_pool *c, dent *e) { e->next = c->head; e->prev = nullptr; if (c->head) c->head->prev = e; c->head = e; if (!c->tail) c->tail = e; }
static size_t hb(const dc *c, int64_t k) { return (size_t)((uint64_t)k * 0x9E3779B97F4A7C15ull >> 32) % c->nhash; }
static void unhash(dc *c, dent *e) { dent **pp = &c->hash[hb(c, e->key)]; while (*pp && *pp != e) pp = &(*pp)->hnext; if (*pp) *pp = e->hnext; }
static void evict(dcache_pool *P) {   /* (mu held) drop unreferenced tail entries above the cap, of any member array */
    for (dent *v = P->tail; v && P->bytes > P->cap;) {
        dent *pv = v->prev;
        if (!v->refs && !v->busy) { unlink_lru(P, v); unhash(v->own, v); P->bytes -= sizeof *v + (v->data ? cbytes(v->own) : 0); free(v->data); free(v); }
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
/* a cell's entry: found (referenced; *mine 0, maybe still busy with another reader) or created busy for the caller (*mine 1) */
static dent *cell_claim(dc *c, int64_t key, int *mine) {
    const size_t h = hb(c, key); dcache_pool *P = c->P;
    pthread_mutex_lock(&P->mu);
    dent *e = c->hash[h]; while (e && e->key != key) e = e->hnext;
    if (e) { e->refs++; *mine = 0; pthread_mutex_unlock(&P->mu); return e; }
    e = calloc(1, sizeof *e); if (!e) { pthread_mutex_unlock(&P->mu); return nullptr; }
    e->key = key; e->own = c; e->busy = 1; e->refs = 1; e->hnext = c->hash[h]; c->hash[h] = e; push_lru(P, e); *mine = 1;
    pthread_mutex_unlock(&P->mu);
    return e;
}
/* another reader's entry ready (the referenced entry, or nullptr: it failed) */
static dent *cell_wait(dc *c, dent *e) {
    dcache_pool *P = c->P; pthread_mutex_lock(&P->mu);
    while (e->busy) pthread_cond_wait(&P->cv, &P->mu);
    if (e->failed) { if (!--e->refs) free(e); pthread_mutex_unlock(&P->mu); return nullptr; }
    unlink_lru(P, e); push_lru(P, e); pthread_mutex_unlock(&P->mu); atomic_fetch_add(&g_dc.mem, 1); return e;
}
static void cell_path_k(const dc *c, int64_t key, char *p, size_t np, int64_t *cz) {
    const int64_t z = key / (c->gc[1] * c->gc[2]), y = key / c->gc[2] % c->gc[1], x = key % c->gc[2]; *cz = z; cell_path(c, z, y, x, p, np);
}
static int cell_disk(dc *c, dent *e) {   /* a claimed cell from disk: 1 found (data / cval set), 0 not */
    if (!c->dir) return 0;
    char p[1300]; int64_t cz; cell_path_k(c, e->key, p, sizeof p, &cz);
    const double t = dnow(); uint8_t *data = nullptr, cval = 0;
    if (disk_get(c, p, &data, &cval)) return 0;
    atomic_fetch_add(&g_dc.disk, 1); atomic_fetch_add(&g_dc.ns_disk, (uint64_t)((dnow() - t) * 1e9));
    e->data = data; e->cval = cval; return 1;
}
/* a computed cell's C^3 values (data owned by the call: kept, or freed when constant) into the entry and onto disk */
static void cell_computed(dc *c, dent *e, uint8_t *data) {
    size_t k = 1; const size_t nb = cbytes(c); uint8_t cval = 0;
    while (k < nb && data[k] == data[0]) k++;
    if (k == nb) { cval = data[0]; free(data); data = nullptr; atomic_fetch_add(&g_dc.fill, 1); }   /* constant cells (most: no labels in reach) */
    atomic_fetch_add(&g_dc.comp, 1);
    char p[1300]; int64_t cz;
    if (c->dir) { cell_path_k(c, e->key, p, sizeof p, &cz);
        if (disk_put(c, cz, p, data, cval) && !atomic_exchange(&c->disk_warned, 1)) fprintf(stderr, "dcache: cannot write %s (%s): memory only from here\n", p, strerror(errno)); }
    e->data = data; e->cval = cval;
}
/* a claimed cell done: ok -> ready (referenced), else failed (nullptr) */
static dent *cell_finish(dc *c, dent *e, int ok) {
    dcache_pool *P = c->P; pthread_mutex_lock(&P->mu);
    e->busy = 0;
    if (!ok) { free(e->data); e->data = nullptr; e->failed = 1; unlink_lru(P, e); unhash(c, e); pthread_cond_broadcast(&P->cv); if (!--e->refs) free(e); pthread_mutex_unlock(&P->mu); return nullptr; }
    P->bytes += sizeof *e + (e->data ? cbytes(c) : 0); evict(P);
    pthread_cond_broadcast(&P->cv); pthread_mutex_unlock(&P->mu);
    return e;
}
/* one cell: a referenced entry (ready), or nullptr on failure */
static dent *cell_get(dc *c, int64_t cz, int64_t cy, int64_t cx, int nthreads) {
    const int64_t key = (cz * c->gc[1] + cy) * c->gc[2] + cx; int mine = 0;
    dent *e = cell_claim(c, key, &mine); if (!e) return nullptr;
    if (!mine) return cell_wait(c, e);
    if (cell_disk(c, e)) return cell_finish(c, e, 1);
    const double t = dnow(); const int64_t o[3] = {cz * c->C, cy * c->C, cx * c->C}, n[3] = {c->C, c->C, c->C};
    uint8_t *data = malloc(cbytes(c)); int ok = 0;
    if (data && !z3_read(c->in, o, n, data, nthreads)) { ok = 1; cell_computed(c, e, data); atomic_fetch_add(&g_dc.ns_comp, (uint64_t)((dnow() - t) * 1e9)); }
    else free(data);
    return cell_finish(c, e, ok);
}
static void cell_put(dc *c, dent *e) { dcache_pool *P = c->P; pthread_mutex_lock(&P->mu); if (!--e->refs && e->failed) free(e); else evict(P); pthread_mutex_unlock(&P->mu); }
static void cell_copy(const dc *c, const dent *e, int64_t cz, int64_t cy, int64_t cx, const int64_t o[3], const int64_t n[3], uint8_t *out) {
    int64_t a[3], b[3]; const int64_t C = c->C, co[3] = {cz * C, cy * C, cx * C};   /* the overlap [a, b) in array coordinates */
    for (int d = 0; d < 3; d++) { a[d] = co[d] > o[d] ? co[d] : o[d]; b[d] = co[d] + C < o[d] + n[d] ? co[d] + C : o[d] + n[d]; if (b[d] > c->shape[d]) b[d] = c->shape[d]; }
    if (e->data || e->cval != (uint8_t)c->fill)
        for (int64_t z = a[0]; z < b[0]; z++) for (int64_t y = a[1]; y < b[1]; y++) {
            uint8_t *dst = out + ((size_t)(z - o[0]) * n[1] + (y - o[1])) * n[2] + (a[2] - o[2]);
            if (e->data) memcpy(dst, e->data + ((size_t)(z - co[0]) * C + (y - co[1])) * C + (a[2] - co[2]), (size_t)(b[2] - a[2]));
            else memset(dst, e->cval, (size_t)(b[2] - a[2]));
        }
}
/* the batched read (2026-11-13, perf2): per block of G^3 cells (G C <= batch), every cell claimed first; the claimed cells not on disk are
   computed as one box (their bounding box, when they fill at least half of it) with one z3_read, then split into cells. The pair rule reads
   its codes with a halo of rmax + 2 around every box it computes: a 128^3 cell at rmax 40 computed 4.5x its voxels, a 3^3-cell box 1.8x
   (and one tile's launches instead of 27). Only for inner arrays whose values do not depend on the box read (the onfly phase: byte-identical
   across tilings, tests/test_phase_onfly.sh). A reader waits for another reader's cells only after finishing all its own claims (no
   wait cycles); a block's references are held until it is copied (at most G^3 cells above the LRU cap). */
static int dc_block(dc *c, const int64_t b0[3], const int64_t b1[3], const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads) {
    enum { MAXB = 512 };
    dent *E[MAXB]; int mine[MAXB], need[MAXB]; int64_t cc[MAXB][3]; int ne = 0, nneed = 0, rc = 0;
    int64_t m0[3] = {INT64_MAX, INT64_MAX, INT64_MAX}, m1[3] = {-1, -1, -1};
    for (int64_t cz = b0[0]; cz <= b1[0]; cz++) for (int64_t cy = b0[1]; cy <= b1[1]; cy++) for (int64_t cx = b0[2]; cx <= b1[2]; cx++) {
        const int64_t key = (cz * c->gc[1] + cy) * c->gc[2] + cx;
        dent *e = cell_claim(c, key, &mine[ne]); if (!e) { rc = -1; break; }
        E[ne] = e; cc[ne][0] = cz; cc[ne][1] = cy; cc[ne][2] = cx; need[ne] = 0;
        if (mine[ne] && !cell_disk(c, e)) { need[ne] = 1; nneed++; const int64_t q[3] = {cz, cy, cx}; for (int d = 0; d < 3; d++) { if (q[d] < m0[d]) m0[d] = q[d]; if (q[d] > m1[d]) m1[d] = q[d]; } }
        ne++;
    }
    if (!rc && nneed) {
        const int64_t C = c->C, bv = (m1[0] - m0[0] + 1) * (m1[1] - m0[1] + 1) * (m1[2] - m0[2] + 1);
        int done = 0;
        if (nneed > 1 && 2 * nneed >= bv) {   /* one box over the missed cells (cells in it already held are recomputed and dropped) */
            const double t = dnow(); int64_t bo[3], bn[3];
            for (int d = 0; d < 3; d++) { bo[d] = m0[d] * C; bn[d] = (m1[d] - m0[d] + 1) * C; }
            uint8_t *box = malloc((size_t)bn[0] * bn[1] * bn[2]);
            if (box && !z3_read(c->in, bo, bn, box, nthreads)) {
                done = 1;
                for (int i = 0; i < ne && done; i++) {
                    if (!need[i]) continue;
                    uint8_t *data = malloc(cbytes(c)); if (!data) { done = 0; break; }
                    const int64_t z0 = (cc[i][0] - m0[0]) * C, y0 = (cc[i][1] - m0[1]) * C, x0 = (cc[i][2] - m0[2]) * C;
                    for (int64_t z = 0; z < C; z++) for (int64_t y = 0; y < C; y++)
                        memcpy(data + ((size_t)z * C + y) * C, box + ((size_t)(z0 + z) * bn[1] + (y0 + y)) * bn[2] + x0, (size_t)C);
                    cell_computed(c, E[i], data); need[i] = 2;
                }
                atomic_fetch_add(&g_dc.ns_comp, (uint64_t)((dnow() - t) * 1e9));
            }
            free(box);
        }
        if (!done)   /* cell by cell (sparse misses, a failed box) */
            for (int i = 0; i < ne; i++) {
                if (need[i] != 1) continue;
                const double t = dnow(); const int64_t co[3] = {cc[i][0] * C, cc[i][1] * C, cc[i][2] * C}, cn[3] = {C, C, C};
                uint8_t *data = malloc(cbytes(c));
                if (data && !z3_read(c->in, co, cn, data, nthreads)) { cell_computed(c, E[i], data); need[i] = 2; atomic_fetch_add(&g_dc.ns_comp, (uint64_t)((dnow() - t) * 1e9)); }
                else { free(data); need[i] = -1; }
            }
    }
    for (int i = 0; i < ne; i++) if (mine[i]) E[i] = cell_finish(c, E[i], need[i] >= 0 && !rc);   /* (all own claims settled before any wait) */
    for (int i = 0; i < ne; i++) if (!mine[i]) E[i] = cell_wait(c, E[i]);
    for (int i = 0; i < ne; i++) {
        if (!E[i]) { rc = -1; continue; }
        if (!rc) cell_copy(c, E[i], cc[i][0], cc[i][1], cc[i][2], o, n, out);
        cell_put(c, E[i]);
    }
    return rc;
}
static int dc_read(void *ctx, z3 *base, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads) {
    (void)base; dc *c = ctx; atomic_fetch_add(&g_dc.reads, 1);
    memset(out, c->fill, (size_t)n[0] * n[1] * n[2]);
    int64_t c0[3], c1[3];
    for (int d = 0; d < 3; d++) {
        if (o[d] + n[d] <= 0 || o[d] >= c->shape[d]) return 0;
        c0[d] = o[d] < 0 ? 0 : o[d] / c->C; c1[d] = (o[d] + n[d] - 1) / c->C; if (c1[d] >= c->gc[d]) c1[d] = c->gc[d] - 1;
    }
    const int G = c->batch > c->C ? (c->batch / c->C < 8 ? c->batch / c->C : 8) : 1;
    if (G > 1) {
        for (int64_t bz = c0[0]; bz <= c1[0]; bz += G) for (int64_t by = c0[1]; by <= c1[1]; by += G) for (int64_t bx = c0[2]; bx <= c1[2]; bx += G) {
            const int64_t b0[3] = {bz, by, bx}, b1[3] = {bz + G - 1 < c1[0] ? bz + G - 1 : c1[0], by + G - 1 < c1[1] ? by + G - 1 : c1[1], bx + G - 1 < c1[2] ? bx + G - 1 : c1[2]};
            if (dc_block(c, b0, b1, o, n, out, nthreads)) return -1;
        }
        return 0;
    }
    for (int64_t cz = c0[0]; cz <= c1[0]; cz++) for (int64_t cy = c0[1]; cy <= c1[1]; cy++) for (int64_t cx = c0[2]; cx <= c1[2]; cx++) {
        dent *e = cell_get(c, cz, cy, cx, nthreads); if (!e) return -1;
        cell_copy(c, e, cz, cy, cx, o, n, out);
        cell_put(c, e);
    }
    return 0;
}
#define DC_MAGIC 0x44434331u
int dcache_set_batch(z3 *z, int edge) {
    dc *c = z ? (dc *)z3_derived_ctx(z) : nullptr;
    if (!c || c->magic != DC_MAGIC) return -1;
    c->batch = edge > 0 ? edge : 0; return 0;
}
static void dc_free(void *p) {
    dc *c = p; if (!c) return;
    dcache_pool *P = c->P;
    if (P) {   /* (a shared pool keeps the other arrays' entries: unlink this array's from its LRU) */
        pthread_mutex_lock(&P->mu);
        for (size_t i = 0; i < c->nhash; i++) for (dent *e = c->hash[i], *nx; e; e = nx) { nx = e->hnext; unlink_lru(P, e); P->bytes -= sizeof *e + (e->data ? cbytes(c) : 0); free(e->data); free(e); }
        pthread_mutex_unlock(&P->mu);
        if (!P->shared) { pthread_mutex_destroy(&P->mu); pthread_cond_destroy(&P->cv); free(P); }
    }
    free(c->hash); free(c->dir); free(c);
}
static dcache_pool *pool_new(size_t cap, int shared) {
    dcache_pool *P = calloc(1, sizeof *P); if (!P) return nullptr;
    pthread_mutex_init(&P->mu, nullptr); pthread_cond_init(&P->cv, nullptr); P->cap = cap; P->shared = shared; return P;
}
dcache_pool *dcache_pool_shared(const char *name, size_t mem_mb) {
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER; static struct { char name[32]; dcache_pool *P; } tab[8]; static int n;
    pthread_mutex_lock(&mu); dcache_pool *P = nullptr;
    for (int i = 0; i < n && !P; i++) if (!strcmp(tab[i].name, name)) P = tab[i].P;
    if (!P && n < 8 && (P = pool_new((mem_mb ? mem_mb : 1024) << 20, 1))) { snprintf(tab[n].name, sizeof tab[n].name, "%s", name); tab[n++].P = P; }
    pthread_mutex_unlock(&mu); return P;   /* (never freed: process lifetime) */
}
size_t dcache_pool_bytes(dcache_pool *P, size_t *cap) {
    if (!P) return 0; pthread_mutex_lock(&P->mu); const size_t b = P->bytes; if (cap) *cap = P->cap; pthread_mutex_unlock(&P->mu); return b;
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
z3 *z3_open_cached_pool(z3 *inner, const char *dir, const char *key, int cell, dcache_pool *pool) {
    if (!inner || !pool) { z3_close(inner); return nullptr; }
    const z3_meta *m = z3_meta_of(inner);
    dc *c = calloc(1, sizeof *c); c->magic = 0x44434331u; c->in = inner; c->C = cell > 0 ? cell : 128; c->fill = m->fill;
    for (int d = 0; d < 3; d++) { c->shape[d] = m->shape[d]; c->gc[d] = (m->shape[d] + c->C - 1) / c->C; }
    c->nhash = 1 << 16; c->hash = calloc(c->nhash, sizeof *c->hash); c->P = pool;
    char d[1200];
    if (dir && *dir && key && !dcache_key_dir(dir, key, d, sizeof d)) c->dir = strdup(d);
    z3 *z = z3_open_derived(inner, c->fill, dc_read, c, dc_free);
    if (!z) { dc_free(c); z3_close(inner); }
    return z;
}
z3 *z3_open_cached(z3 *inner, const char *dir, const char *key, int cell, size_t mem_mb) {
    if (!inner) return nullptr;
    dcache_pool *P = pool_new((mem_mb ? mem_mb : 1024) << 20, 0); if (!P) { z3_close(inner); return nullptr; }
    return z3_open_cached_pool(inner, dir, key, cell, P);   /* (the private pool is freed with the array, or by dc_free on failure) */
}
