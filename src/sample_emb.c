/* train --distill feature targets: see src/sample_emb.h */
#include "sample_emb.h"
#include "embz.h"
#include "json.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void sample_emb_coords(int Pz, int P, int div, const int perm[3], const int flip[3], const spatial_aug *sa, float *V) {
    const int G[3] = {Pz / div, P / div, P / div}, n[3] = {Pz, P, P};
    const int warp = sa && sa->active;
    const double c = .5 * (P - 1), hc = .5 * (div - 1);
    for (int gz = 0; gz < G[0]; gz++) for (int gy = 0; gy < G[1]; gy++) for (int gx = 0; gx < G[2]; gx++) {
        const double u[3] = {div * gz + hc, div * gy + hc, div * gx + hc};
        double U[3];
        if (warp) {   /* spatial_aug_pull at continuous u (its tables hold integer u only) */
            double v[3], w[3];
            for (int d = 0; d < 3; d++) v[d] = u[d] - c - sa->amplitude[d] * sin(sa->frequency * u[d] + sa->phase[d]);
            for (int d = 0; d < 3; d++) w[d] = sa->R[0][d] * v[0] + sa->R[1][d] * v[1] + sa->R[2][d] * v[2];
            for (int d = 0; d < 3; d++) U[d] = c + sa->scale * (sa->affine ? sa->A[d][0] * w[0] + sa->A[d][1] * w[1] + sa->A[d][2] * w[2] : w[d]);
        } else for (int d = 0; d < 3; d++) U[d] = u[d];
        float *o = V + 3 * (((size_t)gz * G[1] + gy) * G[2] + gx);
        for (int d = 0; d < 3; d++) o[perm[d]] = (float)(flip[d] ? n[d] - 1 - U[d] : U[d]);
    }
}

/* per-thread open groups */
typedef struct { const source *s; int k; emb e; int step, ok; } eh_t;
static _Thread_local eh_t *g_eh; static _Thread_local int g_neh;
static eh_t *eh_get(const source *s, int k) {
    for (int i = 0; i < g_neh; i++) if (g_eh[i].s == s && g_eh[i].k == k) return &g_eh[i];
    eh_t *t = realloc(g_eh, (size_t)(g_neh + 1) * sizeof *t); if (!t) return nullptr;
    g_eh = t; eh_t *h = &g_eh[g_neh++]; memset(h, 0, sizeof *h); h->s = s; h->k = k;
    if (emb_open(&h->e, s->emb_grp[k], s->emb_name)) { fprintf(stderr, "distill: %s: cannot open %s/%s\n", s->name, s->emb_grp[k], s->emb_name); return h; }
    char p[1400]; snprintf(p, sizeof p, "%s/%s/zarr.json", s->emb_grp[k], s->emb_name);
    FILE *f = fopen(p, "rb"); char *t8 = nullptr; long n = 0;
    if (f) { fseek(f, 0, SEEK_END); n = ftell(f); fseek(f, 0, SEEK_SET); t8 = malloc(n > 0 ? (size_t)n : 1); if (t8 && fread(t8, 1, (size_t)n, f) != (size_t)n) n = 0; fclose(f); }
    json *j = t8 && n > 0 ? json_parse(t8, (size_t)n) : nullptr; free(t8);
    h->step = j ? (int)json_num(json_at(json_path(j, "attributes.step_zyx"), 0), 0) : 0;
    json_free(j);
    if (h->step <= 0 || h->e.a.dt != 0) { fprintf(stderr, "distill: %s: %s/%s needs int8 values and attributes.step_zyx\n", s->name, s->emb_grp[k], s->emb_name); emb_close(&h->e); return h; }
    h->ok = 1;
    return h;
}

int sample_emb_window(const source *s, int C, int div, int l, const int64_t o[3], int Pz, int P, const int perm[3], const int flip[3], const spatial_aug *sa,
                      int8_t *out) {
    const int G[3] = {Pz / div, P / div, P / div};
    const size_t ng = (size_t)G[0] * G[1] * G[2];
    memset(out, 0, ng * C);
    if (!s->nemb) return 0;
    float *V = malloc(ng * 3 * sizeof *V), *cu = malloc(ng * 3 * sizeof *cu), *acc = malloc((size_t)C * sizeof *acc);
    uint8_t *done = calloc(ng, 1);
    if (!V || !cu || !acc || !done) { free(V); free(cu); free(acc); free(done); return -1; }
    sample_emb_coords(Pz, P, div, perm, flip, sa, V);
    const double f = ldexp(1.0, l - s->emb_level);   /* embedding voxels per level-l voxel */
    int rc = 0;
    for (int pass = 0; pass < 2 && !rc; pass++)   /* whole trilinear stencils first, in any group; then the partial ones (>= half the weight) */
    for (int k = 0; k < s->nemb && !rc; k++) {
        eh_t *h = eh_get(s, k);
        if (!h || !h->ok) { rc = -1; break; }
        if (h->e.a.C != C) { fprintf(stderr, "distill: %s: %s has %d channels, the head %d\n", s->name, s->emb_grp[k], h->e.a.C, C); rc = -1; break; }
        const double st = h->step;
        /* the cell lattice coordinate of every open cell centre (cell i covers embedding voxels [i st, (i + 1) st), centre i st + (st - 1) / 2) */
        int64_t lo[3] = {INT64_MAX, INT64_MAX, INT64_MAX}, hi[3] = {INT64_MIN, INT64_MIN, INT64_MIN};
        for (size_t q = 0; q < ng; q++) {
            if (done[q]) continue;
            for (int d = 0; d < 3; d++) {
                const double e = (o[d] + V[3 * q + d] + .5) * f - .5, c = (e - (st - 1) * .5) / st;
                cu[3 * q + d] = (float)c;
                const int64_t a = (int64_t)floor(c);
                if (a < lo[d]) lo[d] = a;
                if (a + 1 > hi[d]) hi[d] = a + 1;
            }
        }
        if (lo[0] > hi[0]) { pass = 2; break; }   /* every cell done */
        for (int d = 0; d < 3; d++) { if (lo[d] < 0) lo[d] = 0; if (hi[d] >= h->e.a.shape[d]) hi[d] = h->e.a.shape[d] - 1; }
        if (lo[0] > hi[0] || lo[1] > hi[1] || lo[2] > hi[2]) continue;   /* outside this group's array */
        const int64_t rn[3] = {hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1};
        const size_t rb = (size_t)rn[0] * rn[1] * rn[2] * C;
        if (rb > ((size_t)256 << 20)) continue;   /* (a window far larger than the head grid: none) */
        int8_t *reg = malloc(rb);
        if (!reg || zr_read_i8(&h->e.a, lo, rn, reg)) { free(reg); rc = -1; break; }
        for (size_t q = 0; q < ng; q++) {
            if (done[q]) continue;
            int64_t i0[3]; double fr[3];
            for (int d = 0; d < 3; d++) { const double c = cu[3 * q + d]; i0[d] = (int64_t)floor(c); fr[d] = c - (double)i0[d]; }
            for (int k2 = 0; k2 < C; k2++) acc[k2] = 0.f;
            double ws = 0;
            for (int a = 0; a < 8; a++) {
                const int64_t z = i0[0] + (a >> 2) - lo[0], y = i0[1] + ((a >> 1) & 1) - lo[1], x = i0[2] + (a & 1) - lo[2];
                if (z < 0 || y < 0 || x < 0 || z >= rn[0] || y >= rn[1] || x >= rn[2]) continue;
                const double w = ((a >> 2) ? fr[0] : 1 - fr[0]) * (((a >> 1) & 1) ? fr[1] : 1 - fr[1]) * ((a & 1) ? fr[2] : 1 - fr[2]);
                if (w <= 0) continue;
                const int8_t *sv = reg + (((size_t)z * rn[1] + y) * rn[2] + x) * C;
                double l2 = 0; for (int k2 = 0; k2 < C; k2++) l2 += (double)sv[k2] * sv[k2];
                if (!(l2 > 0)) continue;
                const float wn = (float)(w / sqrt(l2));   /* each corner as a unit vector */
                for (int k2 = 0; k2 < C; k2++) acc[k2] += wn * sv[k2];
                ws += w;
            }
            if (ws < (pass ? .5 : .999)) continue;   /* (pass 1: most of the stencil with a vector; else a later group or pass) */
            double l2 = 0; for (int k2 = 0; k2 < C; k2++) l2 += (double)acc[k2] * acc[k2];
            if (!(l2 > 0)) continue;
            const double inv = 127.0 / sqrt(l2); int any = 0;
            for (int k2 = 0; k2 < C; k2++) { const long t = lround(acc[k2] * inv); const int8_t b = (int8_t)(t > 127 ? 127 : t < -127 ? -127 : t); out[(size_t)k2 * ng + q] = b; any |= b; }
            done[q] = any != 0;
        }
        free(reg);
    }
    free(V); free(cu); free(acc); free(done);
    return rc;
}
