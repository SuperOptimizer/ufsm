#include "zcache.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

typedef struct zc_ent {
    uint64_t z; int64_t c[3]; int n[3];    /* the array's z3_id (not its address: a closed array's address can come back; threads audit #7), chunk index, its clipped size */
    uint8_t *buf; size_t bytes;
    int ref, state;                        /* state 0 loading, 1 ready, -1 failed */
    unsigned h;                            /* its bucket */
    struct zc_ent *next;                   /* bucket chain */
    struct zc_ent *lprev, *lnext;          /* the LRU list of ready entries (head: most recently used) */
} zc_ent;

#define ZC_NB 65536
struct zcache {
    pthread_mutex_t mu; pthread_cond_t cv;
    zc_ent *b[ZC_NB];
    size_t budget, bytes;
    zc_ent *lru_head, *lru_tail;           /* ready entries, most recently used first: eviction walks from the tail (was a scan of all 65536 buckets per
                                              eviction under mu: threads audit #2) */
    _Atomic uint64_t hits, misses;
};

zcache *zcache_open(size_t budget) {
    zcache *c = calloc(1, sizeof *c); if (!c) return nullptr;
    pthread_mutex_init(&c->mu, nullptr); pthread_cond_init(&c->cv, nullptr); c->budget = budget;
    return c;
}
void zcache_close(zcache *c) {
    if (!c) return;
    for (int i = 0; i < ZC_NB; i++) for (zc_ent *e = c->b[i], *n; e; e = n) { n = e->next; free(e->buf); free(e); }
    pthread_mutex_destroy(&c->mu); pthread_cond_destroy(&c->cv); free(c);
}
void zcache_stats(const zcache *c, uint64_t *hits, uint64_t *misses, size_t *bytes) {
    if (hits) *hits = c->hits;
    if (misses) *misses = c->misses;
    if (bytes) *bytes = c->bytes;
}
/* (mu held) the LRU list of ready entries */
static void lru_unlink(zcache *c, zc_ent *e) {
    if (e->lprev) e->lprev->lnext = e->lnext; else c->lru_head = e->lnext;
    if (e->lnext) e->lnext->lprev = e->lprev; else c->lru_tail = e->lprev;
    e->lprev = e->lnext = nullptr;
}
static void lru_front(zcache *c, zc_ent *e) { e->lprev = nullptr; e->lnext = c->lru_head; if (c->lru_head) c->lru_head->lprev = e; c->lru_head = e; if (!c->lru_tail) c->lru_tail = e; }
static void bucket_unlink(zcache *c, zc_ent *e) { for (zc_ent **pp = &c->b[e->h]; *pp; pp = &(*pp)->next) if (*pp == e) { *pp = e->next; break; } e->next = nullptr; }
static void zc_free_ready(zcache *c, zc_ent *e) { lru_unlink(c, e); bucket_unlink(c, e); c->bytes -= e->bytes; free(e->buf); free(e); }   /* (an unreferenced ready entry) */
/* the array is about to be closed: its unreferenced ready chunks go (a chunk in use or loading stays; keyed by z3_id it can never be
   served to a later array at the same address) */
void zcache_drop(zcache *c, const z3 *z) {
    if (!c) return;
    const uint64_t id = z3_id(z);
    pthread_mutex_lock(&c->mu);
    for (zc_ent *e = c->lru_head, *n; e; e = n) { n = e->lnext; if (e->z == id && !e->ref) zc_free_ready(c, e); }
    pthread_mutex_unlock(&c->mu);
}
static unsigned zc_hash(uint64_t z, const int64_t c[3]) {
    uint64_t h = z * 0x9e3779b97f4a7c15ull ^ (uint64_t)c[0] * 0xbf58476d1ce4e5b9ull ^ (uint64_t)c[1] * 0x94d049bb133111ebull ^ (uint64_t)c[2] * 0xd6e8feb86659fd93ull;
    return (unsigned)((h ^ (h >> 29)) & (ZC_NB - 1));
}
/* evict unreferenced ready entries, least recently used first, until under budget (mu held): from the LRU tail, skipping the referenced */
static void zc_evict(zcache *c) {
    for (zc_ent *e = c->lru_tail, *p; e && c->bytes > c->budget; e = p) { p = e->lprev; if (!e->ref) zc_free_ready(c, e); }
}
/* the chunk, referenced (zc_put releases); nullptr on failure */
static zc_ent *zc_get(zcache *c, z3 *z, const int64_t ci[3]) {
    const uint64_t id = z3_id(z); const unsigned h = zc_hash(id, ci);
    pthread_mutex_lock(&c->mu);
    for (;;) {
        zc_ent *e = c->b[h];
        while (e && !(e->z == id && e->c[0] == ci[0] && e->c[1] == ci[1] && e->c[2] == ci[2])) e = e->next;
        if (!e) break;
        if (e->state == 0) { pthread_cond_wait(&c->cv, &c->mu); continue; }   /* another thread decodes it */
        if (e->state < 0) { pthread_mutex_unlock(&c->mu); return nullptr; }
        e->ref++; lru_unlink(c, e); lru_front(c, e); pthread_mutex_unlock(&c->mu); atomic_fetch_add(&c->hits, 1); return e;
    }
    const z3_meta *m = z3_meta_of(z);
    zc_ent *e = calloc(1, sizeof *e); if (!e) { pthread_mutex_unlock(&c->mu); return nullptr; }
    e->z = id; e->h = h; memcpy(e->c, ci, sizeof e->c); e->ref = 1; e->state = 0;
    int64_t o[3], nn[3];
    for (int d = 0; d < 3; d++) { o[d] = ci[d] * m->chunk[d]; nn[d] = m->shape[d] - o[d] < m->chunk[d] ? m->shape[d] - o[d] : m->chunk[d]; e->n[d] = (int)nn[d]; }
    e->bytes = (size_t)nn[0] * nn[1] * nn[2];
    e->next = c->b[h]; c->b[h] = e;
    pthread_mutex_unlock(&c->mu);
    atomic_fetch_add(&c->misses, 1);
    uint8_t *buf = malloc(e->bytes);
    const int rc = !buf || z3_read(z, o, nn, buf, 1);
    pthread_mutex_lock(&c->mu);
    if (rc) { free(buf); e->state = -1; e->bytes = 0; }
    else { e->buf = buf; e->state = 1; c->bytes += e->bytes; lru_front(c, e); }
    pthread_cond_broadcast(&c->cv);
    if (rc) {   /* unlink the failed entry once its waiters have seen it: drop it now (later reads retry) */
        for (zc_ent **pp = &c->b[h]; *pp; pp = &(*pp)->next) if (*pp == e) { *pp = e->next; break; }
        e->next = nullptr; e->ref--;
        if (!e->ref) free(e);   /* (a waiter that saw state -1 returned without a reference) */
        pthread_mutex_unlock(&c->mu); return nullptr;
    }
    zc_evict(c);
    pthread_mutex_unlock(&c->mu);
    return e;
}
static void zc_put(zcache *c, zc_ent *e) { pthread_mutex_lock(&c->mu); e->ref--; pthread_mutex_unlock(&c->mu); }

int zcache_read(zcache *c, z3 *z, const int64_t o[3], const int64_t n[3], uint8_t *out) {
    const z3_meta *m = z3_meta_of(z);
    if (!c || m->esize != 1) return z3_read(z, o, n, out, 1);
    const size_t sy = (size_t)n[2], sz = (size_t)n[1] * n[2];
    /* the clipped box; outside: fill */
    int64_t lo[3], hi[3]; int inside = 1;
    for (int d = 0; d < 3; d++) { lo[d] = o[d] < 0 ? 0 : o[d]; hi[d] = o[d] + n[d] > m->shape[d] ? m->shape[d] : o[d] + n[d]; inside &= lo[d] < hi[d]; }
    if (!inside || lo[0] != o[0] || lo[1] != o[1] || lo[2] != o[2] || hi[0] != o[0] + n[0] || hi[1] != o[1] + n[1] || hi[2] != o[2] + n[2])
        memset(out, m->fill, (size_t)n[0] * sz);
    if (!inside) return 0;
    int64_t c0[3], c1[3]; for (int d = 0; d < 3; d++) { c0[d] = lo[d] / m->chunk[d]; c1[d] = (hi[d] - 1) / m->chunk[d]; }
    for (int64_t cz = c0[0]; cz <= c1[0]; cz++) for (int64_t cy = c0[1]; cy <= c1[1]; cy++) for (int64_t cx = c0[2]; cx <= c1[2]; cx++) {
        const int64_t ci[3] = {cz, cy, cx};
        zc_ent *e = zc_get(c, z, ci); if (!e) return -1;
        int64_t a[3], b[3], co[3];
        for (int d = 0; d < 3; d++) { co[d] = ci[d] * m->chunk[d]; a[d] = lo[d] > co[d] ? lo[d] : co[d]; b[d] = hi[d] < co[d] + e->n[d] ? hi[d] : co[d] + e->n[d]; }
        const size_t w = (size_t)(b[2] - a[2]);
        for (int64_t z0 = a[0]; z0 < b[0]; z0++) for (int64_t y0 = a[1]; y0 < b[1]; y0++)
            memcpy(out + (size_t)(z0 - o[0]) * sz + (size_t)(y0 - o[1]) * sy + (size_t)(a[2] - o[2]),
                   e->buf + ((size_t)(z0 - co[0]) * e->n[1] + (size_t)(y0 - co[1])) * e->n[2] + (size_t)(a[2] - co[2]), w);
        zc_put(c, e);
    }
    return 0;
}
