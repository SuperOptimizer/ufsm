/* Sheet-identity embeddings as winding evidence: see src/fit_emb.h */
#include "fit_emb.h"
#include "embz.h"
#include "json.h"
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct emb_vol {
    int8_t *v;            /* [nz][ny][nx][C] unit x 127; all zero = no vector */
    int64_t n[3], o[3];   /* cells; embedding voxel of cell 0's low corner */
    int C, step;
    double org[3], scale; /* native box origin (native voxels), embedding voxels per native voxel */
};

static char *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb"); if (!f) return nullptr;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *p = malloc(sz > 0 ? (size_t)sz + 1 : 1); if (!p || fread(p, 1, (size_t)sz, f) != (size_t)sz) { free(p); fclose(f); return nullptr; }
    fclose(f); p[sz] = 0; *n = (size_t)sz; return p;
}

emb_vol *emb_vol_open(const char *group, const char *name, const long long org[3], const long long n[3], double scale, int pad) {
    emb E; if (emb_open(&E, group, name)) return nullptr;
    int step = 0;
    {   char p[1400]; snprintf(p, sizeof p, "%s/%s/zarr.json", group, name); size_t len = 0; char *t = slurp(p, &len);
        json *j = t ? json_parse(t, len) : nullptr; free(t);
        if (j) step = (int)json_num(json_at(json_path(j, "attributes.step_zyx"), 0), 0);
        json_free(j);
    }
    if (step <= 0 || E.a.C <= 0 || !(scale > 0)) { fprintf(stderr, "fit-emb: %s/%s: no step_zyx / channels\n", group, name); emb_close(&E); return nullptr; }
    emb_vol *e = calloc(1, sizeof *e); e->C = E.a.C; e->step = step; e->scale = scale;
    for (int d = 0; d < 3; d++) {
        e->org[d] = (double)org[d];
        int64_t c0 = (int64_t)floor((org[d] - pad) * scale / step), c1 = (int64_t)ceil((org[d] + n[d] + pad) * scale / step) + 1;
        if (c0 < 0) c0 = 0;
        if (c1 > E.a.shape[d]) c1 = E.a.shape[d];
        if (c1 <= c0) { fprintf(stderr, "fit-emb: the box is outside %s/%s\n", group, name); emb_close(&E); free(e); return nullptr; }
        e->o[d] = c0 * step; e->n[d] = c1 - c0;
    }
    const int C = e->C; const size_t plane = (size_t)e->n[1] * e->n[2];
    /* slabs of planes ending on the chunk grid (each chunk decoded once), at most ~256 MB of float32 */
    const int ch = E.a.chunk > 0 ? E.a.chunk : 1; int64_t sl = (int64_t)((256u << 20) / (plane * C * sizeof(float))); sl = sl < 1 ? 1 : sl > ch ? ch : sl;
    e->v = calloc(plane * e->n[0] * C, 1); float *buf = malloc(plane * C * sl * sizeof *buf);
    if (!e->v || !buf) { fprintf(stderr, "fit-emb: out of memory for %lld cells x %d\n", (long long)(plane * e->n[0]), C); free(buf); emb_vol_free(e); emb_close(&E); return nullptr; }
    size_t nv = 0;
    for (int64_t z0 = 0; z0 < e->n[0];) {
        const int64_t gz = e->o[0] / step + z0, end = sl == ch ? (gz / ch + 1) * ch - e->o[0] / step : z0 + sl, z1 = end < e->n[0] ? end : e->n[0];
        const int64_t o3[3] = {gz, e->o[1] / step, e->o[2] / step}, n3[3] = {z1 - z0, e->n[1], e->n[2]};
        if (emb_read(&E, o3, n3, buf)) { fprintf(stderr, "fit-emb: read failed\n"); free(buf); emb_vol_free(e); emb_close(&E); return nullptr; }
        for (size_t i = 0; i < plane * (size_t)(z1 - z0); i++) {
            const float *s = buf + i * C; double q = 0; for (int k = 0; k < C; k++) q += (double)s[k] * s[k];
            if (q <= 0) continue;
            const double inv = 127.0 / sqrt(q); int8_t *d = e->v + ((size_t)z0 * plane + i) * C;
            for (int k = 0; k < C; k++) { const long t = lround(s[k] * inv); d[k] = (int8_t)(t > 127 ? 127 : t < -127 ? -127 : t); }
            nv++;
        }
        z0 = z1;
    }
    free(buf); emb_close(&E);
    fprintf(stderr, "fit-emb: %s/%s: %lldx%lldx%lld cells of %d voxels (C %d, %.2f GB), %zu with vectors\n", group, name, (long long)e->n[0], (long long)e->n[1], (long long)e->n[2],
            step, C, plane * e->n[0] * C / 1e9, nv);
    return e;
}

emb_vol *emb_vol_wrap(const int8_t *v, const int64_t n[3], int C, int step, const int64_t o[3], const long long org[3], double scale) {
    emb_vol *e = calloc(1, sizeof *e); const size_t tot = (size_t)n[0] * n[1] * n[2] * C;
    e->v = malloc(tot); memcpy(e->v, v, tot); e->C = C; e->step = step; e->scale = scale;
    for (int d = 0; d < 3; d++) { e->n[d] = n[d]; e->o[d] = o[d]; e->org[d] = (double)org[d]; }
    return e;
}
void emb_vol_free(emb_vol *e) { if (e) { free(e->v); free(e); } }
int emb_vol_dim(const emb_vol *e) { return e->C; }

int emb_at(const emb_vol *e, const double p[3], float *v) {
    const int C = e->C; int64_t i0[3]; double f[3];
    for (int d = 0; d < 3; d++) {   /* p continuous native (voxel k spans [k, k + 1)) -> embedding voxel index q; cell i covers voxels [o + i step, o + (i + 1) step) */
        const double q = (e->org[d] + p[d]) * e->scale - 0.5, u = (q - e->o[d] - (e->step - 1) * 0.5) / e->step;
        i0[d] = (int64_t)floor(u); f[d] = u - (double)i0[d];
    }
    for (int k = 0; k < C; k++) v[k] = 0.f;
    double ws = 0;
    for (int a = 0; a < 8; a++) {
        const int64_t z = i0[0] + (a >> 2), y = i0[1] + ((a >> 1) & 1), x = i0[2] + (a & 1);
        if (z < 0 || y < 0 || x < 0 || z >= e->n[0] || y >= e->n[1] || x >= e->n[2]) continue;
        const double w = ((a >> 2) ? f[0] : 1 - f[0]) * (((a >> 1) & 1) ? f[1] : 1 - f[1]) * ((a & 1) ? f[2] : 1 - f[2]);
        if (w <= 0) continue;
        const int8_t *s = e->v + (((size_t)z * e->n[1] + y) * e->n[2] + x) * C;
        int any = 0; for (int k = 0; k < C; k++) any |= s[k];
        if (!any) continue;
        for (int k = 0; k < C; k++) v[k] += (float)(w * s[k]);
        ws += w;
    }
    return ws > 0.25;   /* most of the stencil present */
}

/* the mean vector over three samples across the sheet at p (along unit n), unnormalised; 0 if none */
static int sheet_vec(const emb_vol *e, const double p[3], const double n[3], float *acc, float *tmp) {
    const int C = e->C; int got = 0;
    for (int k = 0; k < C; k++) acc[k] = 0;
    for (int s = -1; s <= 1; s++) {
        const double q[3] = {p[0] + s * n[0], p[1] + s * n[1], p[2] + s * n[2]};
        if (!emb_at(e, q, tmp)) continue;
        double l = 0; for (int k = 0; k < C; k++) l += (double)tmp[k] * tmp[k];
        if (l <= 0) continue;
        const double inv = 1 / sqrt(l); for (int k = 0; k < C; k++) acc[k] += (float)(tmp[k] * inv);
        got++;
    }
    return got > 0;
}
static double vcos(const float *a, const float *b, int C) {
    double ab = 0, aa = 0, bb = 0; for (int k = 0; k < C; k++) { ab += (double)a[k] * b[k]; aa += (double)a[k] * a[k]; bb += (double)b[k] * b[k]; }
    return aa > 0 && bb > 0 ? ab / sqrt(aa * bb) : -2;
}

static int vote_buf(const emb_vol *e, const emb_side *a, const emb_side *b, int rmax, double phi, int *r, float *conf, float *buf) {
    const int C = e->C; float *ea = buf, *eb = buf + C, *tmp = buf + 2 * C;
    const double ta = floor(a->W - phi + 0.5) + phi - a->W, L = a->W + ta;   /* turns to a's nearest sheet centre, its level */
    const double pa[3] = {a->p[0] + a->n[0] * ta * a->sp, a->p[1] + a->n[1] * ta * a->sp, a->p[2] + a->n[2] * ta * a->sp};
    if (!sheet_vec(e, pa, a->n, ea, tmp)) return -1;
    double best = -3, second = -3; int rb = 0, nr = 0;
    for (int rr = -rmax; rr <= rmax; rr++) {
        const double tb = L + rr - b->W;   /* b's line: turns from b to its centre at level L + rr */
        const double pb[3] = {b->p[0] + b->n[0] * tb * b->sp, b->p[1] + b->n[1] * tb * b->sp, b->p[2] + b->n[2] * tb * b->sp};
        if (!sheet_vec(e, pb, b->n, eb, tmp)) continue;
        const double c = vcos(ea, eb, C); nr++;
        if (c > best) { second = best; best = c; rb = rr; } else if (c > second) second = c;
    }
    if (nr < 2) return -1;
    *r = rb; *conf = (float)(best - second);
    return 0;
}
int emb_shift_vote(const emb_vol *e, const emb_side *a, const emb_side *b, int rmax, double phi, int *r, float *conf) {
    if (e->C <= 512) { float sb[3 * 512]; return vote_buf(e, a, b, rmax, phi, r, conf, sb); }
    float *buf = malloc(3 * (size_t)e->C * sizeof *buf); const int rc = vote_buf(e, a, b, rmax, phi, r, conf, buf); free(buf); return rc;
}

typedef struct { const emb_vol *e; size_t n; const emb_side *a, *b; int rmax; double phi; int32_t *r; float *conf; _Atomic size_t next; } vjob;
static void *vote_main(void *p) {
    vjob *J = p; float *buf = malloc(3 * (size_t)J->e->C * sizeof *buf);
    for (size_t k; (k = atomic_fetch_add(&J->next, 64)) < J->n;)
        for (size_t i = k; i < k + 64 && i < J->n; i++) {
            int rr = 0; float cf = 0;
            if (vote_buf(J->e, &J->a[i], &J->b[i], J->rmax, J->phi, &rr, &cf, buf)) { J->r[i] = INT32_MIN; J->conf[i] = 0; }
            else { J->r[i] = rr; J->conf[i] = cf; }
        }
    free(buf); return nullptr;
}
void emb_shift_votes(const emb_vol *e, size_t n, const emb_side *a, const emb_side *b, int rmax, double phi, int32_t *r, float *conf, int threads) {
    if (threads <= 0) { const long c = sysconf(_SC_NPROCESSORS_ONLN); threads = c > 0 ? (int)c : 1; }
    if (threads > 64) threads = 64;
    vjob J = {.e = e, .n = n, .a = a, .b = b, .rmax = rmax, .phi = phi, .r = r, .conf = conf}; atomic_init(&J.next, 0);
    pthread_t th[64]; int nt = 0;
    for (int t = 1; t < threads; t++) if (!pthread_create(&th[nt], nullptr, vote_main, &J)) nt++;
    vote_main(&J);
    for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
}
const int8_t *emb_vol_cells(const emb_vol *e, int64_t n[3], int64_t o[3], int *step) {
    for (int d = 0; d < 3; d++) { n[d] = e->n[d]; o[d] = e->o[d]; }
    *step = e->step; return e->v;
}
