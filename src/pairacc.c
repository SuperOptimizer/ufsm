/* Gauge-free pair winding accuracy: see pairacc.h */
#include "pairacc.h"
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

static inline uint64_t pa_mix(uint64_t x) { x += 0x9e3779b97f4a7c15ull; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull; x = (x ^ (x >> 27)) * 0x94d049bb133111ebull; return x ^ (x >> 31); }
static void pa_filter(pa_buf *b) { size_t k = 0; for (size_t i = 0; i < b->n; i++) if (b->r[i].h < b->T) b->r[k++] = b->r[i]; b->n = k; }

void pa_init(pa_buf *b, size_t cap) { memset(b, 0, sizeof *b); b->cap = cap < 16 ? 16 : cap; b->r = malloc(b->cap * sizeof *b->r); b->T = UINT32_MAX; }
void pa_free(pa_buf *b) { free(b->r); memset(b, 0, sizeof *b); }
void pa_add(pa_buf *b, float pz, float py, float px, float d, uint8_t q, uint64_t key) {
    const uint32_t h = (uint32_t)(pa_mix(key) >> 32);
    if (h >= b->T || !b->r) return;
    while (b->n == b->cap) { b->T /= 2; pa_filter(b); if (h >= b->T) return; }
    b->r[b->n++] = (pa_rec){{pz, py, px}, d, h, q};
}
void pa_merge(pa_buf *dst, pa_buf *src) {
    if (src->T < dst->T) { dst->T = src->T; pa_filter(dst); }
    if (dst->n + src->n > dst->cap) { dst->cap = dst->n + src->n; dst->r = realloc(dst->r, dst->cap * sizeof *dst->r); }
    for (size_t i = 0; i < src->n; i++) if (src->r[i].h < dst->T) dst->r[dst->n++] = src->r[i];
    src->n = 0;
}

typedef struct { uint64_t key; uint32_t i; } pa_cell;
static inline uint64_t pa_ck(int64_t z, int64_t y, int64_t x) { return ((uint64_t)(z + (1 << 20)) << 42) | ((uint64_t)(y + (1 << 20)) << 21) | (uint64_t)(x + (1 << 20)); }
static inline uint64_t pa_rnd(uint64_t *s) { *s += 0x9e3779b97f4a7c15ull; return pa_mix(*s); }
static inline double pa_u(uint64_t *s) { return (pa_rnd(s) >> 11) * (1.0 / 9007199254740992.0); }
/* c sorted by (key, i): an LSD radix sort on the key (16-bit digits, stable) of cells made in index order = qsort by key then index
   (constant digits skipped); tmp: scratch of N cells; returns the sorted array (c or tmp) */
static pa_cell *pa_sort(pa_cell *c, pa_cell *tmp, size_t N) {
    uint64_t lo = ~0ull, hi = 0; for (size_t i = 0; i < N; i++) { lo &= c[i].key; hi |= c[i].key; }
    const uint64_t vary = lo ^ hi;   /* bits that differ between some keys */
    size_t *cnt = malloc(65536 * sizeof *cnt);
    for (int sh = 0; sh < 64; sh += 16) {
        if (!(vary >> sh & 0xffff)) continue;
        memset(cnt, 0, 65536 * sizeof *cnt);
        for (size_t i = 0; i < N; i++) cnt[c[i].key >> sh & 0xffff]++;
        size_t s = 0; for (int d = 0; d < 65536; d++) { const size_t t = cnt[d]; cnt[d] = s; s += t; }
        for (size_t i = 0; i < N; i++) tmp[cnt[c[i].key >> sh & 0xffff]++] = c[i];
        pa_cell *t = c; c = tmp; tmp = t;
    }
    free(cnt);
    return c;
}
typedef struct { const pa_buf *b; const double *dist; size_t npairs; uint64_t seed; double *acc, *acc_cross; size_t *n, *n_cross; int k; } pa_job;
static void *pa_one(void *a) {   /* distance k: its own cell table and random stream (the same values as the serial loop) */
    const pa_job *J = a; const pa_buf *b = J->b; const size_t N = b->n; const int k = J->k; const size_t npairs = J->npairs;
    J->acc[k] = J->acc_cross[k] = NAN; J->n[k] = J->n_cross[k] = 0;
    if (N < 2 || !(J->dist[k] > 0)) return nullptr;
    pa_cell *c0 = malloc(N * sizeof *c0), *t0 = malloc(N * sizeof *t0);
    if (!c0 || !t0) { free(c0); free(t0); return nullptr; }
    const double D = J->dist[k], cs = D / 2;
    for (size_t i = 0; i < N; i++) c0[i] = (pa_cell){pa_ck((int64_t)floor(b->r[i].p[0] / cs), (int64_t)floor(b->r[i].p[1] / cs), (int64_t)floor(b->r[i].p[2] / cs)), (uint32_t)i};
    const pa_cell *c = pa_sort(c0, t0, N);
    uint64_t s = J->seed ^ pa_mix((uint64_t)k + 1); size_t right = 0, rc = 0, got = 0, gc = 0;
    for (size_t t = 0; t < 20 * npairs && got < npairs; t++) {
        const pa_rec *u = &b->r[pa_rnd(&s) % N];
        const double cz = 2 * pa_u(&s) - 1, ph = 2 * M_PI * pa_u(&s), sz = sqrt(1 - cz * cz);   /* uniform direction */
        const double tp[3] = {u->p[0] + D * cz, u->p[1] + D * sz * cos(ph), u->p[2] + D * sz * sin(ph)};
        const uint64_t key = pa_ck((int64_t)floor(tp[0] / cs), (int64_t)floor(tp[1] / cs), (int64_t)floor(tp[2] / cs));
        size_t lo = 0, hi = N; while (lo < hi) { size_t m = (lo + hi) / 2; if (c[m].key < key) lo = m + 1; else hi = m; }
        size_t e = lo, h2 = N; while (e < h2) { size_t m = (e + h2) / 2; if (c[m].key <= key) e = m + 1; else h2 = m; }
        if (e == lo) continue;
        const pa_rec *v = &b->r[c[lo + pa_rnd(&s) % (e - lo)].i];
        const double dz = v->p[0] - u->p[0], dy = v->p[1] - u->p[1], dx = v->p[2] - u->p[2], r = sqrt(dz * dz + dy * dy + dx * dx);
        if (r < 0.75 * D || r > 1.25 * D) continue;
        double w = u->d - v->d; w -= 14 * round(w / 14); const int ok = fabs(w) < 0.5;
        got++; right += ok; if (u->q != v->q) { gc++; rc += ok; }
    }
    J->n[k] = got; J->n_cross[k] = gc;
    if (got) J->acc[k] = (double)right / got;
    if (gc) J->acc_cross[k] = (double)rc / gc;
    free(c0); free(t0);
    return nullptr;
}
/* the distances on threads (each its own cell table: N x 32 B per distance) */
void pa_score(const pa_buf *b, const double *dist, int nd, size_t npairs, uint64_t seed, double *acc, double *acc_cross, size_t *n, size_t *n_cross) {
    enum { PA_MAXT = 16 };
    pa_job J[PA_MAXT]; pthread_t th[PA_MAXT]; int run[PA_MAXT] = {0};
    for (int k0 = 0; k0 < nd; k0 += PA_MAXT) {
        const int m = nd - k0 < PA_MAXT ? nd - k0 : PA_MAXT;
        for (int q = 0; q < m; q++) { J[q] = (pa_job){b, dist, npairs, seed, acc, acc_cross, n, n_cross, k0 + q}; run[q] = m > 1 && pthread_create(&th[q], nullptr, pa_one, &J[q]) == 0; if (!run[q]) pa_one(&J[q]); }
        for (int q = 0; q < m; q++) if (run[q]) pthread_join(th[q], nullptr);
    }
}
