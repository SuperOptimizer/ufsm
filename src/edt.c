#include "edt.h"
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

/* the lower envelope of the parabolas (q - v)^2 + f[v] over a line: d[q] = min_v (q - v)^2 + f[v]; v / z scratch of n / n + 1 */
static void edt1d(const float *f, int n, float *d, int *v, float *z) {
    int k = 0; v[0] = -1;
    for (int q = 0; q < n; q++) {
        if (f[q] >= EDT_INF) continue;
        if (v[0] < 0) { v[0] = q; z[0] = -INFINITY; z[1] = INFINITY; continue; }
        float s;
        for (;;) {
            const int p = v[k];
            s = ((f[q] + (float)q * q) - (f[p] + (float)p * p)) / (2.f * (q - p));
            if (s <= z[k] && k > 0) k--; else break;
        }
        if (s <= z[k]) { v[k] = q; z[k + 1] = INFINITY; continue; }   /* k == 0: q replaces the only parabola */
        k++; v[k] = q; z[k] = s; z[k + 1] = INFINITY;
    }
    if (v[0] < 0) { for (int q = 0; q < n; q++) d[q] = EDT_INF; return; }
    k = 0;
    for (int q = 0; q < n; q++) {
        while (z[k + 1] < q) k++;
        const float t = (float)(q - v[k]); d[q] = t * t + f[v[k]];
    }
}

typedef struct { const uint8_t *feat; float *d; int64_t n[3]; int axis; _Atomic int64_t next; int64_t nlines; } edt_job;

static void *edt_worker(void *arg) {
    edt_job *J = arg; const int64_t *n = J->n;
    const int64_t len = n[J->axis], stride = J->axis == 2 ? 1 : J->axis == 1 ? n[2] : n[1] * n[2];
    float *f = malloc(len * sizeof(float)), *o = malloc(len * sizeof(float)), *z = malloc((len + 1) * sizeof(float));
    int *v = malloc(len * sizeof(int));
    if (!f || !o || !z || !v) { free(f); free(o); free(z); free(v); return (void *)1; }
    for (int64_t L; (L = atomic_fetch_add(&J->next, 64)) < J->nlines;) {   /* lines in batches of 64 */
        const int64_t L1 = L + 64 < J->nlines ? L + 64 : J->nlines;
        for (int64_t l = L; l < L1; l++) {
            int64_t base;   /* the line's first voxel: lines index the two other axes in order */
            if (J->axis == 2) base = l * n[2];
            else if (J->axis == 1) base = (l / n[2]) * n[1] * n[2] + l % n[2];
            else base = l;
            if (J->axis == 2) {   /* first pass from the mask: 0 at features */
                for (int64_t q = 0; q < len; q++) f[q] = J->feat[base + q] ? 0.f : EDT_INF;
            } else for (int64_t q = 0; q < len; q++) f[q] = J->d[base + q * stride];
            edt1d(f, (int)len, o, v, z);
            for (int64_t q = 0; q < len; q++) J->d[base + q * stride] = o[q];
        }
    }
    free(f); free(o); free(z); free(v);
    return nullptr;
}

int edt_sq(const uint8_t *feat, const int64_t n[3], float *d, int nthreads) {
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 64) nthreads = 64;
    for (int axis = 2; axis >= 0; axis--) {   /* x, then y, then z */
        edt_job J = {feat, d, {n[0], n[1], n[2]}, axis, 0, axis == 2 ? n[0] * n[1] : axis == 1 ? n[0] * n[2] : n[1] * n[2]};
        pthread_t th[64]; int bad = 0;
        for (int t = 0; t < nthreads; t++) pthread_create(&th[t], nullptr, edt_worker, &J);
        for (int t = 0; t < nthreads; t++) { void *r; pthread_join(th[t], &r); bad |= r != nullptr; }
        if (bad) return -1;
    }
    return 0;
}
