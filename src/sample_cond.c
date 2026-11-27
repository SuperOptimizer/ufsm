/* train --cond-fields conditioning planes: see src/sample_cond.h */
#include "sample_cond.h"
#include "sample_emb.h"
#include "defaulted.h"
#include "json.h"
#include "store.h"
#include "zarr3.h"
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *cond_planes[8] = {"pc", "ps", "w", "sp", "dq", "dz", "dy", "dx"};

void sample_cond_decode(const uint8_t v[8], int has_w4, const int perm[3], const int flip[3], float out[COND_K]) {
    out[0] = (v[0] - 128) / 127.f; out[1] = (v[1] - 128) / 127.f; out[2] = v[2] / 255.f; out[3] = v[3] / 255.f;
    for (int k = 4; k < COND_K; k++) out[k] = 0.f;
    if (!has_w4) return;
    const float q = v[4] / 255.f; out[4] = q;
    for (int a = 0; a < 3; a++) {
        const double t = 2 * M_PI * v[5 + perm[a]] / 256.0;
        out[5 + 2 * a] = q * (float)cos(t); out[6 + 2 * a] = (flip[a] ? -q : q) * (float)sin(t);
    }
}

/* per-thread open sets */
typedef struct { const source *s; int k, ok, has_w4, h; double um; int64_t origin[3], off[3], shape[3]; z3 *z[8]; } ch_t;
static _Thread_local ch_t *g_ch; static _Thread_local int g_nch;
static json *read_json(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return nullptr;
    fseek(f, 0, SEEK_END); const long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *t = malloc((size_t)n + 1); if (!t || n <= 0 || fread(t, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(t); return nullptr; }
    fclose(f); t[n] = 0; json *j = json_parse(t, (size_t)n); free(t); return j;
}
static ch_t *ch_get(const source *s, int k) {
    for (int i = 0; i < g_nch; i++) if (g_ch[i].s == s && g_ch[i].k == k) return &g_ch[i];
    ch_t *t = realloc(g_ch, (size_t)(g_nch + 1) * sizeof *t); if (!t) return nullptr;
    g_ch = t; ch_t *h = &g_ch[g_nch++]; memset(h, 0, sizeof *h); h->s = s; h->k = k;
    const char *dir = s->cond_dir[k]; char p[1400]; snprintf(p, sizeof p, "%s/fields.json", dir);
    json *j = read_json(p);
    if (!j || strcmp(json_str(json_get(j, "format"), "raw"), "zarr")) { fprintf(stderr, "cond: %s: %s is not a zarr fit-field set (predict --fit-zarr)\n", s->name, dir); json_free(j); return h; }
    h->h = (int)json_num(json_get(j, "h"), 0); h->um = json_num(json_get(j, "um"), 0);
    for (int d = 0; d < 3; d++) {
        h->origin[d] = (int64_t)json_num(json_at(json_get(j, "origin"), d), 0); h->shape[d] = (int64_t)json_num(json_at(json_get(j, "shape"), d), 0);
        h->off[d] = (int64_t)json_num(json_at(json_get(j, "zarr_offset"), d), 0);
    }
    const int w4d = (int)json_num(json_get(j, "w4_d"), 0); json_free(j);
    if (h->h < 1 || !(h->um > 0) || h->shape[0] < 1 || h->shape[1] < 1 || h->shape[2] < 1) { fprintf(stderr, "cond: %s: %s: fields.json without h / um / shape\n", s->name, dir); return h; }
    h->has_w4 = w4d > 0;
    for (int q = 0; q < (h->has_w4 ? 8 : 4); q++) {
        if (!(h->z[q] = z3_open(store_open(dir), cond_planes[q], nullptr))) { fprintf(stderr, "cond: %s: %s/%s: %s\n", s->name, dir, cond_planes[q], z3_error()); return h; }
    }
    if (!h->has_w4) UFSM_DEFAULTED("cond: %s: %s has no winding mod 4 planes: its channels 4..10 are 0", s->name, dir);
    h->ok = 1;
    return h;
}

static _Atomic unsigned long g_cond_dropped;
int sample_cond_window(const source *s, int div, int l, const int64_t o[3], int Pz, int P, const int perm[3], const int flip[3], const spatial_aug *sa,
                       int8_t *out) {
    const int G[3] = {Pz / div, P / div, P / div};
    const size_t ng = (size_t)G[0] * G[1] * G[2];
    memset(out, 0, ng * COND_K);
    if (!s->ncond) return 0;
    float *V = malloc(ng * 3 * sizeof *V); uint8_t *done = calloc(ng, 1);
    if (!V || !done) { free(V); free(done); return -1; }
    sample_emb_coords(Pz, P, div, perm, flip, sa, V);
    const double uml = s->um * ldexp(1.0, l);   /* the window level's voxel (um) */
    int rc = 0;
    for (int k = 0; k < s->ncond && !rc; k++) {
        ch_t *h = ch_get(s, k);
        if (!h) { rc = -1; break; }
        if (!h->ok) continue;
        /* the field cell index of every coarse cell centre: native um of the set, cell i centred at origin + h (i + 0.5) */
        float *cu = malloc(ng * 3 * sizeof *cu); if (!cu) { rc = -1; break; }
        int64_t lo[3] = {INT64_MAX, INT64_MAX, INT64_MAX}, hi[3] = {INT64_MIN, INT64_MIN, INT64_MIN}; size_t nin = 0;
        for (size_t q = 0; q < ng; q++) {
            int in = 1;
            for (int d = 0; d < 3; d++) {
                const double e = ((o[d] + V[3 * q + d] + 0.5) * uml / h->um - h->origin[d]) / h->h - 0.5;
                cu[3 * q + d] = (float)e;
                if (e < -0.5 || e > h->shape[d] - 0.5) in = 0;
            }
            if (!in || done[q]) { cu[3 * q] = NAN; continue; }
            nin++;
            for (int d = 0; d < 3; d++) { const int64_t a = (int64_t)floor(cu[3 * q + d]); if (a < lo[d]) lo[d] = a; if (a + 1 > hi[d]) hi[d] = a + 1; }
        }
        if (!nin) { free(cu); continue; }
        for (int d = 0; d < 3; d++) { if (lo[d] < 0) lo[d] = 0; if (hi[d] >= h->shape[d]) hi[d] = h->shape[d] - 1; }
        const int64_t rn[3] = {hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1};
        const size_t rv = (size_t)rn[0] * rn[1] * rn[2], np = h->has_w4 ? 8 : 4;
        if (rv * np > ((size_t)512 << 20)) {   /* (a degenerate warp: more than 512 MB of planes) */
            const unsigned long nd = atomic_fetch_add(&g_cond_dropped, 1) + 1;
            if ((nd & (nd - 1)) == 0) UFSM_DEFAULTED("cond: %s: %lu windows without conditioning (field region %zu MB)", s->name, nd, rv * np >> 20);
            free(cu); continue;
        }
        uint8_t *reg = malloc(rv * 8);
        if (!reg) { free(cu); rc = -1; break; }
        for (size_t q = 0; q < np && !rc; q++) {
            const int64_t ro[3] = {lo[0] + h->off[0], lo[1] + h->off[1], lo[2] + h->off[2]};
            if (z3_read(h->z[q], ro, rn, reg + q * rv, 1)) { fprintf(stderr, "cond: %s: %s/%s: read failed\n", s->name, s->cond_dir[k], cond_planes[q]); rc = -1; }
        }
        if (!h->has_w4) { memset(reg + 4 * rv, 0, 4 * rv); }
        for (size_t q = 0; q < ng && !rc; q++) {
            if (isnan(cu[3 * q])) continue;
            int64_t i0[3]; double fr[3];
            for (int d = 0; d < 3; d++) { const double c = cu[3 * q + d]; i0[d] = (int64_t)floor(c); fr[d] = c - (double)i0[d]; }
            float acc[COND_K] = {0};
            for (int a = 0; a < 8; a++) {
                const int64_t z = i0[0] + (a >> 2) - lo[0], y = i0[1] + ((a >> 1) & 1) - lo[1], x = i0[2] + (a & 1) - lo[2];
                if (z < 0 || y < 0 || x < 0 || z >= rn[0] || y >= rn[1] || x >= rn[2]) continue;
                const double w = ((a >> 2) ? fr[0] : 1 - fr[0]) * (((a >> 1) & 1) ? fr[1] : 1 - fr[1]) * ((a & 1) ? fr[2] : 1 - fr[2]);
                if (w <= 0) continue;
                const size_t vi = ((size_t)z * rn[1] + y) * rn[2] + x; uint8_t v8[8]; float dv[COND_K];
                for (int p8 = 0; p8 < 8; p8++) v8[p8] = reg[p8 * rv + vi];
                sample_cond_decode(v8, h->has_w4, perm, flip, dv);
                for (int c = 0; c < COND_K; c++) acc[c] += (float)w * dv[c];
            }
            for (int c = 0; c < COND_K; c++) { const long t = lroundf(acc[c] * 127.f); out[(size_t)c * ng + q] = (int8_t)(t > 127 ? 127 : t < -127 ? -127 : t); }
            done[q] = 1;
        }
        free(reg); free(cu);
    }
    free(V); free(done);
    return rc;
}
