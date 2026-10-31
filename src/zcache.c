#include "zcache.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

typedef struct zc_ent {
    const z3 *z; int64_t c[3]; int n[3];   /* chunk index, its clipped size */
    uint8_t *buf; size_t bytes;
    int ref, state;                        /* state 0 loading, 1 ready, -1 failed */
    uint64_t used;
    struct zc_ent *next;                   /* bucket chain */
} zc_ent;

#define ZC_NB 65536
struct zcache {
    pthread_mutex_t mu; pthread_cond_t cv;
    zc_ent *b[ZC_NB];
    size_t budget, bytes; uint64_t clock;
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
static unsigned zc_hash(const z3 *z, const int64_t c[3]) {
    uint64_t h = (uint64_t)(uintptr_t)z * 0x9e3779b97f4a7c15ull ^ (uint64_t)c[0] * 0xbf58476d1ce4e5b9ull ^ (uint64_t)c[1] * 0x94d049bb133111ebull ^ (uint64_t)c[2] * 0xd6e8feb86659fd93ull;
    return (unsigned)((h ^ (h >> 29)) & (ZC_NB - 1));
}
/* evict unreferenced ready entries, least recently used first, until under budget (mu held) */
static void zc_evict(zcache *c) {
    while (c->bytes > c->budget) {
        zc_ent *best = nullptr, **bp = nullptr;
        for (int i = 0; i < ZC_NB; i++) for (zc_ent **pp = &c->b[i]; *pp; pp = &(*pp)->next) {
            zc_ent *e = *pp; if (e->ref || e->state == 0) continue;
            if (!best || e->used < best->used) { best = e; bp = pp; }
        }
        if (!best) return;
        *bp = best->next; c->bytes -= best->bytes; free(best->buf); free(best);
    }
}
/* the chunk, referenced (zc_put releases); nullptr on failure */
static zc_ent *zc_get(zcache *c, z3 *z, const int64_t ci[3]) {
    const unsigned h = zc_hash(z, ci);
    pthread_mutex_lock(&c->mu);
    for (;;) {
        zc_ent *e = c->b[h];
        while (e && !(e->z == z && e->c[0] == ci[0] && e->c[1] == ci[1] && e->c[2] == ci[2])) e = e->next;
        if (!e) break;
        if (e->state == 0) { pthread_cond_wait(&c->cv, &c->mu); continue; }   /* another thread decodes it */
        if (e->state < 0) { pthread_mutex_unlock(&c->mu); return nullptr; }
        e->ref++; e->used = ++c->clock; pthread_mutex_unlock(&c->mu); atomic_fetch_add(&c->hits, 1); return e;
    }
    const z3_meta *m = z3_meta_of(z);
    zc_ent *e = calloc(1, sizeof *e); if (!e) { pthread_mutex_unlock(&c->mu); return nullptr; }
    e->z = z; memcpy(e->c, ci, sizeof e->c); e->ref = 1; e->state = 0; e->used = ++c->clock;
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
    else { e->buf = buf; e->state = 1; c->bytes += e->bytes; }
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

/* zcache_read with nthreads workers over the box's chunks (each worker gets / copies whole chunks: disjoint output regions); the same
   bytes as zcache_read / z3_read (predict's window reader: overlapping windows decode each chunk once while it stays in the budget) */
typedef struct { zcache *c; z3 *z; const int64_t *o, *n, *lo, *hi, *c0, *cn; uint8_t *out; _Atomic long next; long total; _Atomic int fail; } zc_mt;
static void *zc_mt_worker(void *arg) {
    zc_mt *t = arg; const z3_meta *m = z3_meta_of(t->z);
    const size_t sy = (size_t)t->n[2], sz = (size_t)t->n[1] * t->n[2];
    for (long i; (i = atomic_fetch_add(&t->next, 1)) < t->total && !atomic_load(&t->fail);) {
        const int64_t ci[3] = {t->c0[0] + i / (t->cn[1] * t->cn[2]), t->c0[1] + (i / t->cn[2]) % t->cn[1], t->c0[2] + i % t->cn[2]};
        zc_ent *e = zc_get(t->c, t->z, ci); if (!e) { atomic_store(&t->fail, 1); break; }
        int64_t a[3], b[3], co[3];
        for (int d = 0; d < 3; d++) { co[d] = ci[d] * m->chunk[d]; a[d] = t->lo[d] > co[d] ? t->lo[d] : co[d]; b[d] = t->hi[d] < co[d] + e->n[d] ? t->hi[d] : co[d] + e->n[d]; }
        const size_t w = (size_t)(b[2] - a[2]);
        for (int64_t z0 = a[0]; z0 < b[0]; z0++) for (int64_t y0 = a[1]; y0 < b[1]; y0++)
            memcpy(t->out + (size_t)(z0 - t->o[0]) * sz + (size_t)(y0 - t->o[1]) * sy + (size_t)(a[2] - t->o[2]),
                   e->buf + ((size_t)(z0 - co[0]) * e->n[1] + (size_t)(y0 - co[1])) * e->n[2] + (size_t)(a[2] - co[2]), w);
        zc_put(t->c, e);
    }
    return nullptr;
}
int zcache_read_mt(zcache *c, z3 *z, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads) {
    const z3_meta *m = z3_meta_of(z);
    if (!c || m->esize != 1) return z3_read(z, o, n, out, nthreads);
    int64_t lo[3], hi[3], c0[3], cn[3]; int inside = 1;
    for (int d = 0; d < 3; d++) { lo[d] = o[d] < 0 ? 0 : o[d]; hi[d] = o[d] + n[d] > m->shape[d] ? m->shape[d] : o[d] + n[d]; inside &= lo[d] < hi[d]; }
    if (!inside || lo[0] != o[0] || lo[1] != o[1] || lo[2] != o[2] || hi[0] != o[0] + n[0] || hi[1] != o[1] + n[1] || hi[2] != o[2] + n[2])
        memset(out, m->fill, (size_t)n[0] * n[1] * n[2]);
    if (!inside) return 0;
    for (int d = 0; d < 3; d++) { c0[d] = lo[d] / m->chunk[d]; cn[d] = (hi[d] - 1) / m->chunk[d] - c0[d] + 1; }
    zc_mt t = {c, z, o, n, lo, hi, c0, cn, out, 0, cn[0] * cn[1] * cn[2], 0};
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 64) nthreads = 64;
    pthread_t th[64]; int nt = 0;
    for (int i = 1; i < nthreads; i++) if (!pthread_create(&th[nt], nullptr, zc_mt_worker, &t)) nt++;
    zc_mt_worker(&t);
    for (int i = 0; i < nt; i++) pthread_join(th[i], nullptr);
    return atomic_load(&t.fail) ? -1 : 0;
}
