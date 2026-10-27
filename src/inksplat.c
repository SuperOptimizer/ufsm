/* ufsm ink-splat (inksplat.h): published 2D ink labels -> sparse 3D label zarr on the CT grid, and the depth profile of a 3D ink
   prediction along the labelled surface. */
#include "inksplat.h"
#include "imgops.h"
#include "tiffw.h"
#include "json.h"
#include "sources.h"
#include "store.h"
#include "z3w.h"
#include "zarr2.h"
#include "zarr3.h"
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "pyramid.h"

static double snow(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
enum { SH = 512, CH = 128, NCS = SH / CH, TS = 64 };

/* ---------------- surface ---------------- */
typedef struct { char name[256]; int w, h; float *X, *Y, *Z; double scale, vox_um; } ssurf;
static int ssurf_load(ssurf *s, const char *dir) {
    memset(s, 0, sizeof *s);
    const char *b = strrchr(dir, '/'); snprintf(s->name, sizeof s->name, "%s", b && b[1] ? b + 1 : dir);
    { size_t l = strlen(s->name); while (l && s->name[l - 1] == '/') s->name[--l] = 0; }
    char p[1400]; snprintf(p, sizeof p, "%s/meta.json", dir);
    FILE *f = tfx_meta_fopen(p); if (!f) { fprintf(stderr, "ink-splat: cannot read %s\n", p); return -1; }
    char m[65536] = {0}; const size_t mn = fread(m, 1, sizeof m - 1, f); fclose(f);
    json *j = json_parse(m, mn); if (!j) { fprintf(stderr, "ink-splat: bad %s\n", p); return -1; }
    s->scale = json_num(json_at(json_get(j, "scale"), 0), 0); s->vox_um = json_num(json_get(j, "voxel_size_um"), 0);
    json_free(j);
    static const char *nm[3] = {"x", "y", "z"}; float **A[3] = {&s->X, &s->Y, &s->Z};
    for (int k = 0; k < 3; k++) {
        int w, h; snprintf(p, sizeof p, "%s/%s.tif", dir, nm[k]); *A[k] = img_read_f32(p, &w, &h);
        if (!*A[k] || (k && (w != s->w || h != s->h))) { fprintf(stderr, "ink-splat: cannot read %s\n", p); return -1; }
        s->w = w; s->h = h;
    }
    return s->w >= 2 && s->h >= 2 && s->scale > 0 ? 0 : -1;
}
static void ssurf_free(ssurf *s) { free(s->X); free(s->Y); free(s->Z); }
/* point and unit normal du x dv at grid (gu, gv), (z, y, x); 0 when the cell's four vertices are valid */
static inline int ssurf_pn(const ssurf *s, double gu, double gv, double P[3], double N[3]) {
    if (gu < 0 || gv < 0 || gu > s->w - 1 || gv > s->h - 1) return -1;
    int c0 = (int)gu, r0 = (int)gv; if (c0 >= s->w - 1) c0 = s->w - 2; if (r0 >= s->h - 1) r0 = s->h - 2;
    const double fu = gu - c0, fv = gv - r0; const size_t i00 = (size_t)r0 * s->w + c0, i01 = i00 + 1, i10 = i00 + s->w, i11 = i10 + 1;
    if (s->Z[i00] < 0 || s->Z[i01] < 0 || s->Z[i10] < 0 || s->Z[i11] < 0) return -1;
    const float *A[3] = {s->Z, s->Y, s->X}; double du[3], dv[3];
    for (int d = 0; d < 3; d++) {
        const double a = A[d][i00], b = A[d][i01], c = A[d][i10], e = A[d][i11];
        P[d] = (1 - fv) * ((1 - fu) * a + fu * b) + fv * ((1 - fu) * c + fu * e);
        du[d] = (1 - fv) * (b - a) + fv * (e - c); dv[d] = (1 - fu) * (c - a) + fu * (e - b);
    }
    N[0] = du[1] * dv[2] - du[2] * dv[1]; N[1] = du[2] * dv[0] - du[0] * dv[2]; N[2] = du[0] * dv[1] - du[1] * dv[0];
    const double nl = sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]); if (nl < 1e-12) return -1;
    for (int d = 0; d < 3; d++) N[d] /= nl;
    return 0;
}

/* ---------------- 2D label sources ---------------- */
typedef struct {
    int kind;                    /* 1 image, 2 zarr v3 2-D, 3 zarr v2 3-D */
    int64_t H, W;
    uint8_t *img;                /* kind 1 */
    store *s; char key[1024];    /* kinds 2, 3 */
    int64_t shard[2], chunk[2]; int sharded, comp, fill;
    z2 *z; int lz;
    double um;                   /* pixel size from the parent OME group (zarr v3), 0 if unknown */
} lsrc;
static int ends_with(const char *s, const char *e) { size_t a = strlen(s), b = strlen(e); return a >= b && !strcasecmp(s + a - b, e); }
static int lsrc_open(lsrc *L, const char *path, int lz, const char *cache) {
    memset(L, 0, sizeof *L);
    if (ends_with(path, ".tif") || ends_with(path, ".tiff")) {
        int w, h; L->img = img_read_u8(path, &w, &h); if (!L->img) { fprintf(stderr, "ink-splat: cannot read %s (uint8 TIFF)\n", path); return -1; }
        L->kind = 1; L->W = w; L->H = h; return 0;
    }
    if (ends_with(path, ".png")) {
        int w, h; uint8_t *rgb = png_read_rgb(path, &w, &h); if (!rgb) { fprintf(stderr, "ink-splat: cannot read %s\n", path); return -1; }
        L->img = malloc((size_t)w * h); for (size_t k = 0; k < (size_t)w * h; k++) L->img[k] = rgb[3 * k];
        free(rgb); L->kind = 1; L->W = w; L->H = h; return 0;
    }
    char root[1400]; snprintf(root, sizeof root, "%s", path);
    { size_t l = strlen(root); while (l > 1 && root[l - 1] == '/') root[--l] = 0; }
    char *sl = strrchr(root, '/'); if (!sl) { fprintf(stderr, "ink-splat: %s: need a directory path\n", path); return -1; }
    snprintf(L->key, sizeof L->key, "%s", sl + 1); *sl = 0;
    L->s = store_open(root); if (!L->s) { fprintf(stderr, "ink-splat: cannot open %s\n", root); return -1; }
    char mk[1200]; size_t n; snprintf(mk, sizeof mk, "%s/zarr.json", L->key);
    uint8_t *txt = store_read_all(L->s, mk, &n);
    if (!txt) {   /* zarr v2 (depth, H, W): the label layer lz */
        L->z = z2_open(L->s, L->key, cache); if (!L->z) { fprintf(stderr, "ink-splat: %s: %s\n", path, z2_error()); return -1; }
        const z2_meta *m = z2_meta_of(L->z); L->kind = 3; L->H = m->shape[1]; L->W = m->shape[2];
        L->lz = lz >= 0 ? lz : (int)(m->shape[0] / 2);
        if (L->lz >= m->shape[0]) { fprintf(stderr, "ink-splat: %s: layer %d of %lld\n", path, L->lz, (long long)m->shape[0]); return -1; }
        return 0;
    }
    json *j = json_parse((char *)txt, n); free(txt);
    if (!j) { fprintf(stderr, "ink-splat: bad %s\n", mk); return -1; }
    const json *sh = json_get(j, "shape"), *cg = json_path(j, "chunk_grid.configuration.chunk_shape"), *cod = json_get(j, "codecs");
    int rc = -1;
    if (!sh || sh->n != 2 || !cg || cg->n != 2 || strcmp(json_str(json_get(j, "data_type"), ""), "uint8")) { fprintf(stderr, "ink-splat: %s: need a 2-D uint8 array\n", path); goto out; }
    L->H = (int64_t)json_num(json_at(sh, 0), 0); L->W = (int64_t)json_num(json_at(sh, 1), 0);
    L->shard[0] = (int64_t)json_num(json_at(cg, 0), 0); L->shard[1] = (int64_t)json_num(json_at(cg, 1), 0);
    L->fill = (int)json_num(json_get(j, "fill_value"), 0);
    const json *inner = cod;
    if (cod && cod->n && !strcmp(json_str(json_get(json_at(cod, 0), "name"), ""), "sharding_indexed")) {
        const json *cf = json_get(json_at(cod, 0), "configuration"), *ics = json_get(cf, "chunk_shape");
        if (strcmp(json_str(json_get(cf, "index_location"), "end"), "end") || !ics || ics->n != 2) { fprintf(stderr, "ink-splat: %s: unsupported sharding\n", path); goto out; }
        L->sharded = 1; L->chunk[0] = (int64_t)json_num(json_at(ics, 0), 0); L->chunk[1] = (int64_t)json_num(json_at(ics, 1), 0);
        inner = json_get(cf, "codecs");
    } else { L->chunk[0] = L->shard[0]; L->chunk[1] = L->shard[1]; }
    L->comp = 0;
    for (size_t i = 0; inner && i < inner->n; i++) {
        const char *nm = json_str(json_get(json_at(inner, i), "name"), "");
        if (!strcmp(nm, "blosc")) L->comp = 1; else if (!strcmp(nm, "zstd")) L->comp = 2; else if (!strcmp(nm, "gzip")) L->comp = 4;
        else if (strcmp(nm, "bytes")) { fprintf(stderr, "ink-splat: %s: unsupported codec %s\n", path, nm); goto out; }
    }
    L->kind = 2; rc = 0;
    {   /* the pixel size of this level: the parent group's OME multiscales (labels/<group>/<level>) */
        char rr[1400]; snprintf(rr, sizeof rr, "%s", root); char *s2 = strrchr(rr, '/');
        if (s2) {
            char gk[1200]; snprintf(gk, sizeof gk, "%s/zarr.json", s2 + 1); *s2 = 0;
            store *gs = store_open(rr); size_t gn; uint8_t *gt = gs ? store_read_all(gs, gk, &gn) : nullptr;
            json *g = gt ? json_parse((char *)gt, gn) : nullptr; free(gt);
            const json *ds = json_path(g, "attributes.ome.multiscales.0.datasets");
            for (size_t i = 0; ds && i < ds->n; i++) if (!strcmp(json_str(json_get(json_at(ds, i), "path"), ""), L->key))
                L->um = json_num(json_at(json_get(json_at(json_get(json_at(ds, i), "coordinateTransformations"), 0), "scale"), 0), 0);
            json_free(g); if (gs) store_close(gs);
        }
    }
out:
    json_free(j); return rc;
}
static void lsrc_close(lsrc *L) { free(L->img); if (L->z) z2_close(L->z); if (L->s) store_close(L->s); }
/* the box [r0, r0 + h) x [c0, c0 + w) (outside the raster: 0) */
static int lsrc_read(lsrc *L, int64_t r0, int64_t c0, int64_t h, int64_t w, uint8_t *out, int nth) {
    memset(out, 0, (size_t)h * w);
    if (L->kind == 1) {
        for (int64_t r = 0; r < h; r++) { const int64_t rr = r0 + r; if (rr < 0 || rr >= L->H) continue;
            for (int64_t c = 0; c < w; c++) { const int64_t cc = c0 + c; if (cc >= 0 && cc < L->W) out[(size_t)r * w + c] = L->img[(size_t)rr * L->W + cc]; } }
        return 0;
    }
    if (L->kind == 3) { const int64_t o[3] = {L->lz, r0, c0}, n[3] = {1, h, w}; if (z2_read(L->z, o, n, out, nth)) { fprintf(stderr, "ink-splat: label read: %s\n", z2_error()); return -1; } return 0; }
    const int64_t cv = L->chunk[0] * L->chunk[1], ni0 = L->shard[0] / L->chunk[0], ni1 = L->shard[1] / L->chunk[1];
    uint8_t *dec = malloc((size_t)cv); int rc = 0;
    for (int64_t s0 = (r0 > 0 ? r0 : 0) / L->shard[0]; s0 * L->shard[0] < r0 + h && s0 * L->shard[0] < L->H; s0++)
        for (int64_t s1 = (c0 > 0 ? c0 : 0) / L->shard[1]; s1 * L->shard[1] < c0 + w && s1 * L->shard[1] < L->W; s1++) {
            char k[1200]; snprintf(k, sizeof k, "%s/c/%lld/%lld", L->key, (long long)s0, (long long)s1);
            size_t nb; uint8_t *sb = store_read_all(L->s, k, &nb);
            if (!sb) continue;   /* missing shard: fill */
            const size_t ni = L->sharded ? (size_t)(ni0 * ni1) : 1, il = ni * 16 + 4;
            if (L->sharded && nb < il) { free(sb); rc = -1; break; }
            for (int64_t a = 0; a < (L->sharded ? ni0 : 1); a++) for (int64_t b = 0; b < (L->sharded ? ni1 : 1); b++) {
                const int64_t g0 = s0 * L->shard[0] + a * L->chunk[0], g1 = s1 * L->shard[1] + b * L->chunk[1];
                if (g0 >= r0 + h || g0 + L->chunk[0] <= r0 || g1 >= c0 + w || g1 + L->chunk[1] <= c0) continue;
                const uint8_t *src; size_t len;
                if (L->sharded) {
                    uint64_t off, ln; const uint8_t *e = sb + nb - il + (size_t)(a * ni1 + b) * 16; memcpy(&off, e, 8); memcpy(&ln, e + 8, 8);
                    if (off == UINT64_MAX && ln == UINT64_MAX) { memset(dec, L->fill, (size_t)cv); goto copy; }
                    if (off + ln > nb - il) { rc = -1; continue; }
                    src = sb + off; len = (size_t)ln;
                } else { src = sb; len = nb; }
                if (z2_decode(L->comp, src, len, dec, (size_t)cv)) { fprintf(stderr, "ink-splat: label chunk %s (%lld, %lld): %s\n", k, (long long)a, (long long)b, z2_error()); rc = -1; continue; }
            copy:
                for (int64_t r = 0; r < L->chunk[0]; r++) {
                    const int64_t rr = g0 + r - r0; if (rr < 0 || rr >= h || g0 + r >= L->H) continue;
                    for (int64_t c = 0; c < L->chunk[1]; c++) { const int64_t cc = g1 + c - c0; if (cc >= 0 && cc < w && g1 + c < L->W) out[(size_t)rr * w + cc] = dec[(size_t)r * L->chunk[1] + c]; }
                }
            }
            free(sb);
        }
    free(dec);
    if (rc) fprintf(stderr, "ink-splat: label shard read failed\n");
    return rc;
}

/* public: a 2-D label raster window (path as ink-splat's labels: zarr v3 2-D level, zarr v2 3-D (middle layer), uint8 TIFF / PNG);
   *H, *W the raster size; outside the raster 0. malloc'd h x w, nullptr on error. */
uint8_t *inksplat_label_window(const char *path, int64_t r0, int64_t c0, int64_t h, int64_t w, int64_t *H, int64_t *W) {
    lsrc L; if (lsrc_open(&L, path, -1, nullptr)) return nullptr;
    if (H) *H = L.H;
    if (W) *W = L.W;
    uint8_t *o = malloc((size_t)h * w);
    if (!o || lsrc_read(&L, r0, c0, h, w, o, 8)) { free(o); o = nullptr; }
    lsrc_close(&L);
    return o;
}

/* ---------------- volumes (z3 or z2) ---------------- */
typedef struct { z3 *a; z2 *b; int64_t shape[3]; } vol;
static int vol_open(vol *v, const char *root, const char *key, const char *cache) {
    memset(v, 0, sizeof *v); store *s = store_open(root); if (!s) return -1;
    v->a = z3_open(s, key, cache);
    if (v->a) { memcpy(v->shape, z3_meta_of(v->a)->shape, sizeof v->shape); return 0; }
    v->b = z2_open(s, key, cache);
    if (v->b) { memcpy(v->shape, z2_meta_of(v->b)->shape, sizeof v->shape); return 0; }
    fprintf(stderr, "ink-splat: cannot open %s %s: %s / %s\n", root, key, z3_error(), z2_error()); return -1;
}
static int vol_read(vol *v, const int64_t o[3], const int64_t n[3], uint8_t *out, int nth) { return v->a ? z3_read(v->a, o, n, out, nth) : z2_read(v->b, o, n, out, nth); }

/* ---------------- geometry shared by splat and profile ---------------- */
typedef struct {
    const inksplat_cfg *c; const ssurf *S;
    uint8_t *lab, *msk; int64_t lw, lh, lr0, lc0;   /* label / mask window (label pixels) and its origin */
    double F, offr, offc;                              /* canvas pixels per label pixel; canvas offset (--label-offset) */
    double sign, ta, tb;                               /* band in level-0 voxels along sign N */
    int ntiles_x, ntiles_y;
} geo;
static inline int px_sup(const geo *G, int64_t r, int64_t c) { return !G->msk || G->msk[(size_t)r * G->lw + c] >= 128; }
static inline void px_grid(const geo *G, double r, double c, double *gu, double *gv) {   /* label pixel (window coords, may be fractional) -> grid */
    *gu = ((c + G->lc0 + 0.5) * G->F - 0.5 + G->offc) * G->S->scale; *gv = ((r + G->lr0 + 0.5) * G->F - 0.5 + G->offr) * G->S->scale;
}
/* level-0 voxel box of a tile (vertex hull +- reach), 0 if the tile has no supervised pixel on a valid cell */
static int tile_box(const geo *G, int tx, int ty, double reach, double lo[3], double hi[3]) {
    const int64_t r0 = (int64_t)ty * TS, c0 = (int64_t)tx * TS, r1 = r0 + TS < G->lh ? r0 + TS : G->lh, c1 = c0 + TS < G->lw ? c0 + TS : G->lw;
    int any = 0;
    for (int64_t r = r0; r < r1 && !any; r++) for (int64_t c = c0; c < c1; c++) if (px_sup(G, r, c)) { any = 1; break; }
    if (!any) return 0;
    double gu0, gv0, gu1, gv1; px_grid(G, r0 - 0.5, c0 - 0.5, &gu0, &gv0); px_grid(G, r1 - 0.5, c1 - 0.5, &gu1, &gv1);
    const ssurf *S = G->S;
    int u0 = (int)floor(gu0), v0 = (int)floor(gv0), u1 = (int)ceil(gu1) + 1, v1 = (int)ceil(gv1) + 1;
    if (u0 < 0) u0 = 0; if (v0 < 0) v0 = 0; if (u1 > S->w - 1) u1 = S->w - 1; if (v1 > S->h - 1) v1 = S->h - 1;
    for (int d = 0; d < 3; d++) { lo[d] = 1e300; hi[d] = -1e300; }
    int nv = 0;
    for (int v = v0; v <= v1; v++) for (int u = u0; u <= u1; u++) {
        const size_t k = (size_t)v * S->w + u; if (S->Z[k] < 0) continue;
        const double p[3] = {S->Z[k], S->Y[k], S->X[k]};
        for (int d = 0; d < 3; d++) { if (p[d] < lo[d]) lo[d] = p[d]; if (p[d] > hi[d]) hi[d] = p[d]; }
        nv++;
    }
    if (!nv) return 0;
    for (int d = 0; d < 3; d++) { lo[d] -= reach + 1; hi[d] += reach + 1; }
    return 1;
}

/* ---------------- splat ---------------- */
typedef struct { int64_t key; int tile; } stpair;
static int stp_cmp(const void *a, const void *b) { const stpair *p = a, *q = b; return p->key < q->key ? -1 : p->key > q->key ? 1 : p->tile - q->tile; }
typedef struct {
    const geo *G; int L; double V, off; int64_t oshape[3], ns[3];
    stpair *pr; size_t npr; size_t *ustart; size_t nu; _Atomic size_t next;
    z3 *ct; z3w *w; z3 *old; vol *pred;
    double hpx; int nsub, nt; double tstep;
    pthread_mutex_t mu;
    uint64_t n_ink, n_no, n_rej, n_hit, cm[2][2];   /* cm[splat ink][pred ink] over the written known voxels (--pred at this level) */
    uint8_t *chunks_used; int64_t cg[3];            /* 128-chunk occupancy of the written voxels (block origins) */
    int64_t bb_lo[3], bb_hi[3];
    _Atomic int failed;
} sjob;
typedef struct { uint8_t pi[CH * CH * CH], pn[CH * CH * CH], ni[CH * CH * CH], nn[CH * CH * CH]; } acc;
static void *splat_worker(void *arg) {
    sjob *J = arg; const geo *G = J->G; const inksplat_cfg *c = G->c;
    const size_t sv = (size_t)SH * SH * SH, cv = (size_t)CH * CH * CH;
    uint8_t *buf = malloc(sv), *ctb = malloc(cv), *pb = J->pred ? malloc(cv) : nullptr;
    acc **A = calloc(NCS * NCS * NCS, sizeof *A);
    int64_t blo[3] = {INT64_MAX, INT64_MAX, INT64_MAX}, bhi[3] = {-1, -1, -1};
    for (size_t u; (u = atomic_fetch_add(&J->next, 1)) < J->nu && !atomic_load(&J->failed);) {
        const int64_t key = J->pr[J->ustart[u]].key, sz = key / (J->ns[1] * J->ns[2]), sy = (key / J->ns[2]) % J->ns[1], sx = key % J->ns[2];
        const int64_t so[3] = {sz * SH, sy * SH, sx * SH};
        for (size_t q = J->ustart[u]; q < J->ustart[u + 1]; q++) {
            const int t = J->pr[q].tile, tx = t % G->ntiles_x, ty = t / G->ntiles_x;
            const int64_t r0 = (int64_t)ty * TS, c0 = (int64_t)tx * TS, r1 = r0 + TS < G->lh ? r0 + TS : G->lh, c1 = c0 + TS < G->lw ? c0 + TS : G->lw;
            for (int64_t r = r0; r < r1; r++) for (int64_t cc = c0; cc < c1; cc++) {
                if (!px_sup(G, r, cc)) continue;
                const uint8_t lv = G->lab[(size_t)r * G->lw + cc]; const int ink = lv >= 128;
                for (int a = 0; a < J->nsub; a++) for (int b = 0; b < J->nsub; b++) {
                    double gu, gv, P[3], N[3];
                    px_grid(G, r + (a + 0.5) / J->nsub - 0.5, cc + (b + 0.5) / J->nsub - 0.5, &gu, &gv);
                    if (ssurf_pn(G->S, gu, gv, P, N)) continue;
                    for (int k = 0; k < J->nt; k++) {
                        const double tt = (G->ta + k * J->tstep) * G->sign;
                        int64_t vi[3]; int in = 1;
                        for (int d = 0; d < 3; d++) { vi[d] = (int64_t)floor((P[d] + tt * N[d] - J->off) / J->V + 0.5) - so[d]; if (vi[d] < 0 || vi[d] >= SH) { in = 0; break; } }
                        if (!in) continue;
                        const int ci = (int)(((vi[0] >> 7) * NCS + (vi[1] >> 7)) * NCS + (vi[2] >> 7));
                        if (!A[ci]) { A[ci] = calloc(1, sizeof(acc)); if (!A[ci]) { atomic_store(&J->failed, 1); goto done; } }
                        const size_t k2 = (((size_t)(vi[0] & (CH - 1)) * CH) + (vi[1] & (CH - 1))) * CH + (vi[2] & (CH - 1));
                        acc *ac = A[ci];
                        if (ink) { if (ac->ni[k2] < 255) ac->ni[k2]++; if (lv > ac->pi[k2]) ac->pi[k2] = lv; }
                        else { if (ac->nn[k2] < 255) ac->nn[k2]++; if (lv > ac->pn[k2]) ac->pn[k2] = lv; }
                    }
                }
            }
        }
        /* finalize: the old shard (merge) or unknown, then every hit voxel on papyrus */
        int have_old = 0;
        if (J->old && z3_shard_present(J->old, sz, sy, sx) == 1) {
            const int64_t n[3] = {SH, SH, SH};
            if (z3_read(J->old, so, n, buf, 2)) { atomic_store(&J->failed, 1); break; }
            have_old = 1;
        }
        if (!have_old) memset(buf, 255, sv);
        uint64_t ni = 0, nn = 0, nr = 0, nh = 0, cm[2][2] = {{0, 0}, {0, 0}}; int any = 0;
        for (int ci = 0; ci < NCS * NCS * NCS; ci++) {
            acc *ac = A[ci]; if (!ac) continue;
            const int64_t co[3] = {so[0] + (ci / (NCS * NCS)) * CH, so[1] + (ci / NCS % NCS) * CH, so[2] + (ci % NCS) * CH}, cn[3] = {CH, CH, CH};
            if (z3_read(J->ct, co, cn, ctb, 1)) { fprintf(stderr, "ink-splat: CT read: %s\n", z3_error()); atomic_store(&J->failed, 1); break; }
            if (pb && vol_read(J->pred, co, cn, pb, 1)) { fprintf(stderr, "ink-splat: prediction read failed\n"); atomic_store(&J->failed, 1); break; }
            int cany = 0;
            for (size_t k = 0; k < cv; k++) {
                const int h = ac->ni[k] + ac->nn[k]; if (!h) continue;
                nh++;
                const int64_t z = co[0] + (int64_t)(k >> 14), y = co[1] + (int64_t)(k >> 7 & 127), x = co[2] + (int64_t)(k & 127);
                if (z >= J->oshape[0] || y >= J->oshape[1] || x >= J->oshape[2]) continue;
                if (!ctb[k] || ctb[k] < c->iso) { nr++; continue; }
                const int isink = ac->ni[k] && ac->ni[k] >= c->thr * h;
                const uint8_t v = (uint8_t)(((isink ? ac->pi[k] : ac->pn[k]) * 254 + 127) / 255);
                const size_t bi = ((size_t)(z - so[0]) * SH + (size_t)(y - so[1])) * SH + (size_t)(x - so[2]);
                buf[bi] = buf[bi] == 255 ? v : (buf[bi] > v ? buf[bi] : v);
                if (isink) ni++; else nn++;
                if (pb && pb[k] != c->pred_skip) cm[v >= 127][pb[k] >= c->pred_thr]++;
                cany = 1;
                const int64_t p3[3] = {z, y, x};
                for (int d = 0; d < 3; d++) { if (p3[d] < blo[d]) blo[d] = p3[d]; if (p3[d] > bhi[d]) bhi[d] = p3[d]; }
            }
            if (cany) { any = 1; J->chunks_used[((co[0] / CH) * J->cg[1] + co[1] / CH) * J->cg[2] + co[2] / CH] = 1; }
            free(ac); A[ci] = nullptr;
        }
        for (int ci = 0; ci < NCS * NCS * NCS; ci++) { free(A[ci]); A[ci] = nullptr; }
        if (atomic_load(&J->failed)) break;
        if ((any || have_old) && z3w_write_shard(J->w, sz, sy, sx, buf, 2)) { fprintf(stderr, "ink-splat: %s\n", z3w_error()); atomic_store(&J->failed, 1); break; }
        pthread_mutex_lock(&J->mu);
        J->n_ink += ni; J->n_no += nn; J->n_rej += nr; J->n_hit += nh;
        for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) J->cm[a][b] += cm[a][b];
        for (int d = 0; d < 3; d++) { if (blo[d] < J->bb_lo[d]) J->bb_lo[d] = blo[d]; if (bhi[d] > J->bb_hi[d]) J->bb_hi[d] = bhi[d]; }
        pthread_mutex_unlock(&J->mu);
    }
done:
    for (int ci = 0; ci < NCS * NCS * NCS; ci++) free(A[ci]);
    free(A); free(buf); free(ctb); free(pb);
    return nullptr;
}

/* ---------------- profile ---------------- */
enum { PMAX = 64 };
typedef struct { double n[2][2 * PMAX + 1], pap[2][2 * PMAX + 1], pp[2][2 * PMAX + 1], psum[2][2 * PMAX + 1], ctsum[2][2 * PMAX + 1], hist[256]; } prof;
typedef struct { const geo *G; int L; double V, off; z3 *ct; vol *pred; _Atomic int next; int ntile; pthread_mutex_t mu; prof P; _Atomic int failed; } pjob;
static inline float tri_box(const uint8_t *b, const int64_t n[3], double z, double y, double x) {
    if (z < 0 || y < 0 || x < 0 || z > n[0] - 1 || y > n[1] - 1 || x > n[2] - 1) return -1.f;
    int64_t z0 = (int64_t)z, y0 = (int64_t)y, x0 = (int64_t)x; if (z0 >= n[0] - 1) z0 = n[0] - 2; if (y0 >= n[1] - 1) y0 = n[1] - 2; if (x0 >= n[2] - 1) x0 = n[2] - 2;
    if (z0 < 0 || y0 < 0 || x0 < 0) return -1.f;
    const double fz = z - z0, fy = y - y0, fx = x - x0; double s = 0;
    for (int c = 0; c < 8; c++) {
        const int64_t a = z0 + (c >> 2 & 1), bb = y0 + (c >> 1 & 1), e = x0 + (c & 1);
        s += (c >> 2 & 1 ? fz : 1 - fz) * (c >> 1 & 1 ? fy : 1 - fy) * (c & 1 ? fx : 1 - fx) * b[((size_t)a * n[1] + bb) * n[2] + e];
    }
    return (float)s;
}
static void *prof_worker(void *arg) {
    pjob *J = arg; const geo *G = J->G; const inksplat_cfg *c = G->c; const int T = c->profile, st = c->stride > 0 ? c->stride : 1;
    prof *P = calloc(1, sizeof *P);
    for (int t; (t = atomic_fetch_add(&J->next, 1)) < J->ntile && !atomic_load(&J->failed);) {
        const int tx = t % G->ntiles_x, ty = t / G->ntiles_x;
        double lo[3], hi[3]; if (!tile_box(G, tx, ty, T * c->layer_vox, lo, hi)) continue;
        int64_t o[3], n[3]; size_t nv = 1;
        for (int d = 0; d < 3; d++) { o[d] = (int64_t)floor((lo[d] - J->off) / J->V) - 1; n[d] = (int64_t)ceil((hi[d] - J->off) / J->V) + 2 - o[d]; nv *= (size_t)n[d]; }
        if (nv > (size_t)1 << 30) { fprintf(stderr, "ink-splat: profile tile %d box too large, skipped\n", t); continue; }
        uint8_t *cb = malloc(nv), *pb = J->pred ? malloc(nv) : nullptr;
        if (!cb || (J->pred && !pb) || z3_read(J->ct, o, n, cb, 1) || (pb && vol_read(J->pred, o, n, pb, 1))) { fprintf(stderr, "ink-splat: profile read failed\n"); free(cb); free(pb); atomic_store(&J->failed, 1); break; }
        const int64_t r0 = (int64_t)ty * TS, c0 = (int64_t)tx * TS, r1 = r0 + TS < G->lh ? r0 + TS : G->lh, c1 = c0 + TS < G->lw ? c0 + TS : G->lw;
        for (int64_t r = r0; r < r1; r++) {
            if ((r + G->lr0) % st) continue;
            for (int64_t cc = c0; cc < c1; cc++) {
                if ((cc + G->lc0) % st || !px_sup(G, r, cc)) continue;
                const int cls = G->lab[(size_t)r * G->lw + cc] >= 128;
                double gu, gv, Pp[3], N[3]; px_grid(G, (double)r, (double)cc, &gu, &gv);
                if (ssurf_pn(G->S, gu, gv, Pp, N)) continue;
                for (int i = -T; i <= T; i++) {
                    const double tt = i * c->layer_vox * G->sign; double q[3];
                    for (int d = 0; d < 3; d++) q[d] = (Pp[d] + tt * N[d] - J->off) / J->V - o[d];
                    const float cv = tri_box(cb, n, q[0], q[1], q[2]), pv = pb ? tri_box(pb, n, q[0], q[1], q[2]) : 0.f;
                    if (cv < 0) continue;
                    const int k = i + PMAX, pap = cv >= c->iso && cv > 0, pi = pv >= c->pred_thr;
                    P->n[cls][k]++; P->ctsum[cls][k] += cv; P->psum[cls][k] += pv; P->hist[(int)(cv + 0.5f) > 255 ? 255 : (int)(cv + 0.5f)]++;
                    if (pap) { P->pap[cls][k]++; if (pi) P->pp[cls][k]++; }
                }
            }
        }
        free(cb); free(pb);
    }
    pthread_mutex_lock(&J->mu);
    double *a = (double *)&J->P, *b = (double *)P; for (size_t k = 0; k < sizeof *P / sizeof(double); k++) a[k] += b[k];
    pthread_mutex_unlock(&J->mu);
    free(P); return nullptr;
}
/* the best ink band [a, b] (layers) by IoU of splat ink (label ink, papyrus, in the band) vs prediction ink (papyrus, anywhere in
   -T .. T at a supervised pixel); also the agreement inside the band */
static void band_score4(const prof *P, int T, int a, int b, double *iou, double *agree, double *prec, double *rec) {
    double tp = 0, fp = 0, tn = 0, pall = 0, nb = 0;
    for (int i = -T; i <= T; i++) { const int k = i + PMAX; pall += P->pp[0][k] + P->pp[1][k];
        if (i < a || i > b) continue;
        tp += P->pp[1][k]; fp += P->pap[1][k] - P->pp[1][k]; tn += P->pap[0][k] - P->pp[0][k]; nb += P->pap[0][k] + P->pap[1][k]; }
    const double fn = pall - tp;
    *iou = tp + fp + fn > 0 ? tp / (tp + fp + fn) : 0; *agree = nb > 0 ? (tp + tn) / nb : 0;
    *prec = tp + fp > 0 ? tp / (tp + fp) : 0; *rec = pall > 0 ? tp / pall : 0;
}
static void band_score(const prof *P, int T, int a, int b, double *iou, double *agree, double *dice) {
    double pr, rc; band_score4(P, T, a, b, iou, agree, &pr, &rc);
    *dice = *iou > 0 ? 2 * *iou / (1 + *iou) : 0;
}

void inksplat_defaults(inksplat_cfg *c) {
    memset(c, 0, sizeof *c);
    c->label_z = c->mask_z = -1; c->band_a = -4; c->band_b = 4; c->layer_vox = 1; c->thr = 0.5; c->coarse = -1; c->threads = 6;
    c->iso = 1; c->pred_level = -1; c->pred_thr = 128; c->pred_skip = -1; c->stride = 4;
}

static int mkdir_p(const char *path) {
    char tmp[1400]; snprintf(tmp, sizeof tmp, "%s", path);
    for (char *p = tmp + 1; *p; p++) if (*p == '/') { *p = 0; if (mkdir(tmp, 0755) && errno != EEXIST) return -1; *p = '/'; }
    return mkdir(tmp, 0755) && errno != EEXIST ? -1 : 0;
}

static const char *g_attrs = "{\"ufsm\":{\"content\":\"ink-splat\",\"encoding\":\"prob254: p*254, 254 = ink, 0 = no ink, 255 = unknown\"}}";
/* the coarse levels (known-max pooling, L + 1 .. top) of <out> and its OME group (levels L .. top) */
static int splat_pyramid(const inksplat_cfg *c, const int64_t shape0[3]) {
    const int L = c->level; int top = c->coarse >= 0 ? L + c->coarse : (L < 5 ? 5 : L);
    if (top > 9) top = 9;
    double ums[16]; int nl = 0;
    for (int l = L; l <= top; l++) {
        if (l > L && pyramid_build_level(c->out, c->ct_um, l, shape0, SH, 0, 3, 4 * (c->threads > 0 ? c->threads : 1), g_attrs)) { fprintf(stderr, "ink-splat: level %d failed\n", l); return -1; }
        ums[nl++] = c->ct_um * (1 << l);
    }
    if (z3w_write_group(c->out, ums, nl, "ink-splat", g_attrs)) { fprintf(stderr, "ink-splat: group: %s\n", z3w_error()); return -1; }
    return 0;
}
/* --pyramid-only: rebuild the coarse levels and the group of an existing splat (after merging many segments with --coarse 0) */
static int splat_pyramid_only(const inksplat_cfg *c) {
    store *cts = store_open(c->ct_root); z3 *ct0 = cts ? pyramid_open_level(cts, c->ct_key, 0, c->ct_um, c->cache) : nullptr;
    if (!ct0 || !c->out) { fprintf(stderr, "ink-splat: --pyramid-only needs <out> and the CT: %s\n", z3_error()); return 1; }
    int64_t shape0[3]; memcpy(shape0, z3_meta_of(ct0)->shape, sizeof shape0); z3_close(ct0);
    return splat_pyramid(c, shape0) ? 1 : 0;
}

int inksplat_run(const inksplat_cfg *c) {
    const double t0 = snow();
    if (!c->surface || !c->labels || !c->ct_root || !c->ct_key || !(c->ct_um > 0) || c->level < 0 || c->level > 6 || !(c->band_b >= c->band_a) ||
        !(c->layer_vox > 0) || c->profile < 0 || c->profile > PMAX) { fprintf(stderr, "ink-splat: bad arguments\n"); return 2; }
    ssurf S; if (ssurf_load(&S, c->surface)) return 1;
    const char *name = c->name ? c->name : S.name;
    lsrc LL, LM; int rc = 1;
    if (lsrc_open(&LL, c->labels, c->label_z, c->cache)) { ssurf_free(&S); return 1; }
    int has_mask = c->mask != nullptr;
    if (has_mask && lsrc_open(&LM, c->mask, c->mask_z, c->cache)) { lsrc_close(&LL); ssurf_free(&S); return 1; }
    if (has_mask && (LM.H != LL.H || LM.W != LL.W)) { fprintf(stderr, "ink-splat: mask %lld x %lld != labels %lld x %lld\n", (long long)LM.W, (long long)LM.H, (long long)LL.W, (long long)LL.H); goto out0; }
    double F = c->label_scale;
    const double ew = S.w / S.scale, eh = S.h / S.scale;   /* the canvas the tifxyz implies (level-0 pixels) */
    if (!(F > 0)) {
        F = 0.5 * (ew / (double)LL.W + eh / (double)LL.H);   /* (labels on another scan's canvas of the same mesh: the voxel size ratio, e.g. 9.6 / 9.362) */
        const double p2 = pow(2, round(log2(F)));
        if (fabs(F - p2) < 0.002 * p2) F = p2;
        const double mum = S.vox_um > 0 ? S.vox_um : c->ct_um;
        if (LL.um > 0 && mum > 0) {   /* the OME pixel size wins: the canvas sizes are rounded to the grid step */
            fprintf(stderr, "ink-splat: label pixel %g um / mesh voxel %g um = %.5f canvas px per label px (canvas size ratio %.5f)\n", LL.um, mum, LL.um / mum, F);
            F = LL.um / mum;
        }
    }
    fprintf(stderr, "ink-splat: %s: tifxyz %d x %d scale %g (canvas %.0f x %.0f), labels %lld x %lld, %g canvas px per label px%s\n", name, S.w, S.h,
            S.scale, ew, eh, (long long)LL.W, (long long)LL.H, F, fabs(LL.H * F - eh) > 0.03 * eh ? " (WARNING: height does not match)" : "");
    geo G = {0}; G.c = c; G.S = &S; G.F = F; G.sign = c->flip ? -1 : 1; G.ta = c->band_a * c->layer_vox; G.tb = c->band_b * c->layer_vox;
    G.lr0 = 0; G.lc0 = 0; G.lw = LL.W; G.lh = LL.H; G.offr = c->label_off[0]; G.offc = c->label_off[1];
    if (c->crop[2] > 0 && c->crop[3] > 0) { G.lc0 = c->crop[0]; G.lr0 = c->crop[1]; G.lw = c->crop[2]; G.lh = c->crop[3]; }
    if ((double)G.lw * G.lh > 4e9) { fprintf(stderr, "ink-splat: label window too large\n"); goto out0; }
    G.lab = malloc((size_t)G.lw * G.lh); G.msk = has_mask ? malloc((size_t)G.lw * G.lh) : nullptr;
    if (!G.lab || (has_mask && !G.msk)) goto out0;
    if (lsrc_read(&LL, G.lr0, G.lc0, G.lh, G.lw, G.lab, c->threads) || (has_mask && lsrc_read(&LM, G.lr0, G.lc0, G.lh, G.lw, G.msk, c->threads))) goto out1;
    {   /* 0/1 rasters (e.g. scrollprize ink_9um labels and masks) -> 0/255 */
        uint8_t *R[2] = {G.lab, G.msk}; const size_t np = (size_t)G.lw * G.lh;
        for (int k = 0; k < 2; k++) { if (!R[k]) continue; uint8_t mx = 0; for (size_t i = 0; i < np; i++) if (R[k][i] > mx) mx = R[k][i];
            uint64_t h[256] = {0}; for (size_t i = 0; i < np; i++) h[R[k][i]]++;
            char hs[512]; int o = 0, nv = 0; for (int v = 0; v < 256 && o < 400; v++) if (h[v]) { nv++; if (nv <= 8) o += snprintf(hs + o, sizeof hs - o, " %d:%llu", v, (unsigned long long)h[v]); }
            fprintf(stderr, "ink-splat: %s values (%d distinct):%s%s\n", k ? "mask" : "labels", nv, o ? hs : "", nv > 8 ? " .." : "");
            if (mx == 1) { for (size_t i = 0; i < np; i++) R[k][i] *= 255; fprintf(stderr, "ink-splat: %s is 0/1, read as 0/255\n", k ? "mask" : "labels"); } }
    }
    uint64_t nsup = 0, nink = 0;
    for (size_t k = 0; k < (size_t)G.lw * G.lh; k++) if (!G.msk || G.msk[k] >= 128) { nsup++; nink += G.lab[k] >= 128; }
    const double px_um = F * (S.vox_um > 0 ? S.vox_um : c->ct_um), cm2 = 1e-8 * px_um * px_um;
    fprintf(stderr, "ink-splat: %s: %llu supervised label px (%.2f cm2), %llu ink (%.2f cm2, %.1f%%) in %.1fs\n", name, (unsigned long long)nsup, nsup * cm2,
            (unsigned long long)nink, nink * cm2, nsup ? 100.0 * nink / nsup : 0, snow() - t0);
    if (c->preview) {   /* the label window, at most 1024 px: unsupervised black, no ink gray, ink white (box-averaged) */
        const int64_t f = (G.lw > G.lh ? G.lw : G.lh) / 1024 + 1, pw = G.lw / f, ph = G.lh / f;
        uint8_t *rgb = calloc((size_t)pw * ph * 3, 1);
        for (int64_t y = 0; y < ph; y++) for (int64_t x = 0; x < pw; x++) {
            double s = 0, i = 0;
            for (int64_t a = 0; a < f; a++) for (int64_t b = 0; b < f; b++) { const size_t k = (size_t)(y * f + a) * G.lw + x * f + b; if (!G.msk || G.msk[k] >= 128) { s++; i += G.lab[k] >= 128; } }
            uint8_t *o = rgb + (size_t)(y * pw + x) * 3; o[0] = o[1] = o[2] = (uint8_t)(s ? (80 + 175 * i / s) * s / (f * f) : 0);
            if (s && !i) o[2] = 110;
        }
        if (jpeg_write_rgb(c->preview, (int)pw, (int)ph, rgb, 90)) fprintf(stderr, "ink-splat: cannot write %s\n", c->preview);
        else fprintf(stderr, "ink-splat: preview %s: %lld x %lld (1 px = %lld label px)\n", c->preview, (long long)pw, (long long)ph, (long long)f);
        free(rgb);
    }
    G.ntiles_x = (int)((G.lw + TS - 1) / TS); G.ntiles_y = (int)((G.lh + TS - 1) / TS);
    const int ntile = G.ntiles_x * G.ntiles_y;
    store *cts = store_open(c->ct_root);
    z3 *ct0 = pyramid_open_level(cts, c->ct_key, 0, c->ct_um, c->cache);
    if (!ct0) { fprintf(stderr, "ink-splat: CT level 0: %s\n", z3_error()); goto out1; }
    int64_t shape0[3]; memcpy(shape0, z3_meta_of(ct0)->shape, sizeof shape0);

    /* ---- profile ---- */
    char prof_json[65536]; prof_json[0] = 0;
    if (c->profile) {   /* (without --pred: the CT columns only) */
        const int Lp = c->pred_level >= 0 ? c->pred_level : c->level;
        z3 *ctp = pyramid_open_level(cts, c->ct_key, Lp, c->ct_um, c->cache); vol pv;
        if (!ctp || (c->pred_root && vol_open(&pv, c->pred_root, c->pred_key, c->cache))) { fprintf(stderr, "ink-splat: profile volumes\n"); goto out1; }
        pjob PJ = {0}; PJ.G = &G; PJ.L = Lp; PJ.V = (double)(1 << Lp); PJ.off = (PJ.V - 1) / 2; PJ.ct = ctp; PJ.pred = c->pred_root ? &pv : nullptr; PJ.ntile = ntile; pthread_mutex_init(&PJ.mu, nullptr);
        pthread_t th[64]; const int nt = c->threads < 1 ? 1 : c->threads > 64 ? 64 : c->threads;
        for (int i = 0; i < nt; i++) pthread_create(&th[i], nullptr, prof_worker, &PJ);
        for (int i = 0; i < nt; i++) pthread_join(th[i], nullptr);
        if (atomic_load(&PJ.failed)) goto out1;
        const prof *P = &PJ.P; const int T = c->profile;
        printf("# ink-splat profile %s: level %d (%.4g um), sign %+d, iso %d, prediction >= %d, every %d px; t in layers (%g level-0 voxels) along the signed normal\n",
               name, Lp, c->ct_um * PJ.V, (int)G.sign, c->iso, c->pred_thr, c->stride, c->layer_vox);
        printf("#   t   n_ink  pap_ink  pred_ink|pap  meanpred_ink  meanct_ink   n_no  pap_no  pred_ink|pap_no  meanpred_no  meanct_no\n");
        double best = -1, bagree = 0, bdice = 0; int ba = 0, bb = 0, sb = 0; double sbest = -1;
        double mass = 0, msum = 0;
        for (int i = -T; i <= T; i++) {
            const int k = i + PMAX;
            printf("%4d %8.0f %7.3f %8.3f %10.1f %11.1f %8.0f %7.3f %8.3f %12.1f %10.1f\n", i, P->n[1][k], P->n[1][k] ? P->pap[1][k] / P->n[1][k] : 0,
                   P->pap[1][k] ? P->pp[1][k] / P->pap[1][k] : 0, P->n[1][k] ? P->psum[1][k] / P->n[1][k] : 0, P->n[1][k] ? P->ctsum[1][k] / P->n[1][k] : 0,
                   P->n[0][k], P->n[0][k] ? P->pap[0][k] / P->n[0][k] : 0, P->pap[0][k] ? P->pp[0][k] / P->pap[0][k] : 0,
                   P->n[0][k] ? P->psum[0][k] / P->n[0][k] : 0, P->n[0][k] ? P->ctsum[0][k] / P->n[0][k] : 0);
            mass += P->pp[1][k]; msum += i * P->pp[1][k];
        }
        for (int a = -T; a <= T; a++) for (int b = a; b <= T; b++) {
            double iou, ag, di; band_score(P, T, a, b, &iou, &ag, &di);
            if (iou > best) { best = iou; ba = a; bb = b; bagree = ag; bdice = di; }
            if (a == -b && iou > sbest) { sbest = iou; sb = b; }
        }
        double iou4, ag4, di4; band_score(P, T, (int)floor(c->band_a), (int)ceil(c->band_b), &iou4, &ag4, &di4);
        double sag, sdi, siou; band_score(P, T, -sb, sb, &siou, &sag, &sdi);
        printf("# best band [%d, %d] layers: IoU %.4f dice %.4f agreement %.4f; best symmetric +-%d: IoU %.4f agreement %.4f; configured [%g, %g]: IoU %.4f agreement %.4f\n",
               ba, bb, best, bdice, bagree, sb, siou, sag, c->band_a, c->band_b, iou4, ag4);
        printf("# prediction ink at label-ink pixels: mean depth %+.2f layers (signed normal)\n", mass > 0 ? msum / mass : 0);
        {   /* the CT histogram of every sample (CT > 0): its two modes and the valley between them (a papyrus iso candidate) */
            double h[256] = {0}, tot = 0;
            for (int v = 1; v < 256; v++) { for (int d = -3; d <= 3; d++) { const int u = v + d; if (u >= 1 && u < 256) h[v] += P->hist[u]; } tot += P->hist[v]; }
            int m1 = 1; for (int v = 1; v < 256; v++) if (h[v] > h[m1]) m1 = v;
            int m2 = -1; double best2 = 0;   /* the other mode: the highest bin with a dip of >= 10% between it and m1 */
            for (int v = 1; v < 256; v++) { if (abs(v - m1) < 8) continue; double lo = 1e300; for (int u = v < m1 ? v : m1; u <= (v < m1 ? m1 : v); u++) if (h[u] < lo) lo = h[u];
                if (lo < 0.9 * h[v] && h[v] > best2) { best2 = h[v]; m2 = v; } }
            int valley = -1;
            if (m2 > 0) { const int a = m1 < m2 ? m1 : m2, b = m1 < m2 ? m2 : m1; valley = a; for (int u = a; u <= b; u++) if (h[u] < h[valley]) valley = u; }
            double cum = 0; int p10 = 0, p50 = 0; for (int v = 1; v < 256; v++) { cum += P->hist[v]; if (!p10 && cum >= 0.1 * tot) p10 = v; if (!p50 && cum >= 0.5 * tot) p50 = v; }
            printf("# CT histogram (all samples, CT > 0): modes %d and %d, valley %d, p10 %d, median %d, CT == 0 share %.4f\n", m1, m2, valley, p10, p50, P->hist[0] / (tot + P->hist[0] + 1e-30));
        }
        {   /* the sheet side: mean CT (all supervised pixels) over 1 .. T/2 layers on each side of the surface */
            double sm = 0, nm = 0, sp = 0, np = 0;
            for (int i = 1; i <= (T + 1) / 2; i++) for (int cl = 0; cl < 2; cl++) { sm += P->ctsum[cl][PMAX - i]; nm += P->n[cl][PMAX - i]; sp += P->ctsum[cl][PMAX + i]; np += P->n[cl][PMAX + i]; }
            const double am = nm ? sm / nm : 0, ap = np ? sp / np : 0;
            printf("# CT side: mean CT %.1f on -N, %.1f on +N (1 .. %d layers): papyrus bulk on %cN, open face (recto side if the mesh is on the written face) %cN\n",
                   am, ap, (T + 1) / 2, am > ap ? '-' : '+', am > ap ? '+' : '-');
        }
        static const int bl[][2] = {{-2, 2}, {-4, 4}, {-8, 8}, {-12, 12}, {-16, 16}, {0, 4}, {0, 8}, {0, 12}, {0, 16}, {0, 24}, {-4, 0}, {-8, 0}, {-16, 0},
                                    {-4, 8}, {-4, 12}, {-4, 16}, {2, 12}, {4, 12}, {4, 16}, {4, 20}, {8, 16}, {-8, 16}, {-8, 24}};
        printf("# band  IoU  agreement  precision  recall\n");
        for (size_t i = 0; i < sizeof bl / sizeof *bl; i++) {
            if (bl[i][0] < -T || bl[i][1] > T) continue;
            double iou, ag, pr, rcl; band_score4(P, T, bl[i][0], bl[i][1], &iou, &ag, &pr, &rcl);
            printf("# [%3d, %3d] %.4f %.4f %.4f %.4f\n", bl[i][0], bl[i][1], iou, ag, pr, rcl);
        }
        int o = snprintf(prof_json, sizeof prof_json, "\"profile\": {\"level\": %d, \"sign\": %d, \"best_band\": [%d, %d], \"best_iou\": %.5f, \"best_dice\": %.5f, \"best_agreement\": %.5f, "
                         "\"sym_band\": %d, \"sym_iou\": %.5f, \"sym_agreement\": %.5f, \"cfg_iou\": %.5f, \"cfg_agreement\": %.5f, \"ink_depth\": %.3f, \"t\": [",
                         Lp, (int)G.sign, ba, bb, best, bdice, bagree, sb, siou, sag, iou4, ag4, mass > 0 ? msum / mass : 0);
        for (int i = -T; i <= T && o < (int)sizeof prof_json - 200; i++) { const int k = i + PMAX;
            o += snprintf(prof_json + o, sizeof prof_json - o, "%s[%d, %.0f, %.0f, %.0f, %.0f, %.0f, %.0f]", i > -T ? ", " : "", i, P->n[1][k], P->pap[1][k], P->pp[1][k], P->n[0][k], P->pap[0][k], P->pp[0][k]); }
        snprintf(prof_json + o, sizeof prof_json - o, "]}");
        fflush(stdout);
        z3_close(ctp);
        fprintf(stderr, "ink-splat: profile in %.1fs\n", snow() - t0);
    }
    if (!c->out) { rc = 0; goto out2; }

    /* ---- splat ---- */
    {
        const int L = c->level; const double V = (double)(1 << L), um = c->ct_um * V;
        z3 *ct = L ? pyramid_open_level(cts, c->ct_key, L, c->ct_um, c->cache) : ct0;
        if (!ct) { fprintf(stderr, "ink-splat: CT level %d: %s\n", L, z3_error()); goto out2; }
        int64_t oshape[3]; memcpy(oshape, z3_meta_of(ct)->shape, sizeof oshape);
        char lv[64], adir[1400]; z3w_level_name(um, lv, sizeof lv); snprintf(adir, sizeof adir, "%s/%s", c->out, lv);
        if (mkdir_p(c->out)) { fprintf(stderr, "ink-splat: mkdir %s\n", c->out); goto out2; }
        const char *attrs = "{\"ufsm\":{\"content\":\"ink-splat\",\"encoding\":\"prob254: p*254, 254 = ink, 0 = no ink, 255 = unknown\"}}";
        struct stat sst; char zj[1500]; snprintf(zj, sizeof zj, "%s/zarr.json", adir);
        const int exists = !stat(zj, &sst);
        if (exists && !c->merge) { fprintf(stderr, "ink-splat: %s exists (use --merge)\n", adir); goto out2; }
        z3w *w = z3w_create(adir, oshape, SH, 0, 255, attrs);
        if (!w) { fprintf(stderr, "ink-splat: %s\n", z3w_error()); goto out2; }
        store *os = store_open(c->out); z3 *old = exists ? z3_open(os, lv, nullptr) : nullptr;
        vol pv; int have_pred = 0;
        if (c->pred_root && (c->pred_level < 0 || c->pred_level == L)) { if (vol_open(&pv, c->pred_root, c->pred_key, c->cache)) goto out2; have_pred = 1; }
        sjob J = {0}; J.G = &G; J.L = L; J.V = V; J.off = (V - 1) / 2; memcpy(J.oshape, oshape, sizeof oshape); J.ct = ct; J.w = w; J.old = old; J.pred = have_pred ? &pv : nullptr;
        for (int d = 0; d < 3; d++) { J.ns[d] = (oshape[d] + SH - 1) / SH; J.cg[d] = (oshape[d] + CH - 1) / CH; J.bb_lo[d] = INT64_MAX; J.bb_hi[d] = -1; }
        J.hpx = 0.45 * V / F; if (J.hpx > 1) J.hpx = 1; J.nsub = (int)ceil(1 / J.hpx);   /* (canvas pixel = one level-0 voxel nominally) */
        const double span = G.tb - G.ta; J.nt = (int)ceil(span / (0.45 * V)) + 1; J.tstep = J.nt > 1 ? span / (J.nt - 1) : 0;
        if ((double)J.cg[0] * J.cg[1] * J.cg[2] > 4e9) { fprintf(stderr, "ink-splat: grid too large\n"); goto out2; }
        J.chunks_used = calloc((size_t)(J.cg[0] * J.cg[1] * J.cg[2]), 1);
        /* tile -> shard pairs */
        const double reach = fmax(fabs(G.ta), fabs(G.tb));
        size_t cap = 1024, np = 0; stpair *pr = malloc(cap * sizeof *pr);
        for (int t = 0; t < ntile; t++) {
            double lo[3], hi[3]; if (!tile_box(&G, t % G.ntiles_x, t / G.ntiles_x, reach, lo, hi)) continue;
            int64_t s0[3], s1[3]; int ok = 1;
            for (int d = 0; d < 3; d++) {
                int64_t a = (int64_t)floor((lo[d] - J.off) / V), b = (int64_t)ceil((hi[d] - J.off) / V);
                if (a < 0) a = 0; if (b > oshape[d] - 1) b = oshape[d] - 1; if (b < a) { ok = 0; break; }
                s0[d] = a / SH; s1[d] = b / SH;
            }
            if (!ok) continue;
            for (int64_t a = s0[0]; a <= s1[0]; a++) for (int64_t b = s0[1]; b <= s1[1]; b++) for (int64_t e = s0[2]; e <= s1[2]; e++) {
                if (np == cap) { cap *= 2; pr = realloc(pr, cap * sizeof *pr); }
                pr[np++] = (stpair){(a * J.ns[1] + b) * J.ns[2] + e, t};
            }
        }
        qsort(pr, np, sizeof *pr, stp_cmp);
        size_t *us = malloc((np + 1) * sizeof *us), nu = 0;
        for (size_t i = 0; i < np; i++) if (!i || pr[i].key != pr[i - 1].key) us[nu++] = i;
        us[nu] = np;
        J.pr = pr; J.npr = np; J.ustart = us; J.nu = nu; pthread_mutex_init(&J.mu, nullptr);
        fprintf(stderr, "ink-splat: %s: level %d (%g um) grid %lld x %lld x %lld, band [%g, %g] layers x %g voxels, sign %+d, %d x %d sub-samples per label px, %d t samples; %zu shards, %zu tile pairs\n",
                name, L, um, (long long)oshape[0], (long long)oshape[1], (long long)oshape[2], c->band_a, c->band_b, c->layer_vox, (int)G.sign, J.nsub, J.nsub, J.nt, nu, np);
        pthread_t th[64]; const int nt = c->threads < 1 ? 1 : c->threads > 64 ? 64 : c->threads;
        for (int i = 0; i < nt; i++) pthread_create(&th[i], nullptr, splat_worker, &J);
        for (int i = 0; i < nt; i++) pthread_join(th[i], nullptr);
        z3w_close(w);
        if (old) z3_close(old);
        if (atomic_load(&J.failed)) { fprintf(stderr, "ink-splat: failed\n"); goto out2; }
        const double known = (double)(J.n_ink + J.n_no);
        fprintf(stderr, "ink-splat: %s: %llu hit voxels, %llu ink + %llu no-ink written, %llu below iso / masked (%.1f%% on papyrus) in %.1fs\n", name,
                (unsigned long long)J.n_hit, (unsigned long long)J.n_ink, (unsigned long long)J.n_no, (unsigned long long)J.n_rej, J.n_hit ? 100.0 * known / J.n_hit : 0, snow() - t0);
        char cmp_json[512] = "";
        if (have_pred) {
            const double tp = J.cm[1][1], fp = J.cm[1][0], fn = J.cm[0][1], tn = J.cm[0][0], tot = tp + fp + fn + tn;
            printf("# voxelwise vs prediction >= %d over the %.0f written voxels: agreement %.4f, ink IoU %.4f, precision %.4f, recall %.4f (splat ink %.0f, prediction ink %.0f)\n",
                   c->pred_thr, tot, tot ? (tp + tn) / tot : 0, tp + fp + fn ? tp / (tp + fp + fn) : 0, tp + fp ? tp / (tp + fp) : 0, tp + fn ? tp / (tp + fn) : 0, tp + fp, tp + fn);
            snprintf(cmp_json, sizeof cmp_json, "\"voxelwise\": {\"pred_thr\": %d, \"tp\": %.0f, \"fp\": %.0f, \"fn\": %.0f, \"tn\": %.0f, \"agreement\": %.5f, \"iou\": %.5f},",
                     c->pred_thr, tp, fp, fn, tn, tot ? (tp + tn) / tot : 0, tp + fp + fn ? tp / (tp + fp + fn) : 0);
            fflush(stdout);
        }
        /* coarse levels (known-max pooling) and the group */
        if (splat_pyramid(c, shape0)) goto out2;
        /* summary */
        char sp[1400]; if (c->summary) snprintf(sp, sizeof sp, "%s", c->summary); else { snprintf(sp, sizeof sp, "%s/segments", c->out); mkdir_p(sp); snprintf(sp, sizeof sp, "%s/segments/%s.json", c->out, name); }
        FILE *f = fopen(sp, "w");
        if (!f) { fprintf(stderr, "ink-splat: cannot write %s\n", sp); goto out2; }
        int64_t blo[3], bhi[3]; for (int d = 0; d < 3; d++) { blo[d] = J.bb_lo[d] == INT64_MAX ? 0 : J.bb_lo[d] << L; bhi[d] = J.bb_hi[d] < 0 ? 0 : ((J.bb_hi[d] + 1) << L); }
        fprintf(f, "{\"name\": \"%s\", \"surface\": \"%s\", \"labels\": \"%s\", \"mask\": \"%s\", \"ct\": \"%s/%s\", \"ct_um\": %g, \"level\": %d, \"um\": %g, \"iso\": %d,\n"
                   " \"band\": [%g, %g], \"layer_vox\": %g, \"sign\": %d, \"thr\": %g, \"label_px_per_canvas\": %g, \"crop\": [%d, %d, %d, %d],\n"
                   " \"label_px\": %llu, \"ink_px\": %llu, \"label_cm2\": %.4f, \"ink_cm2\": %.4f, \"hit_voxels\": %llu, \"ink_voxels\": %llu, \"noink_voxels\": %llu, \"rejected_voxels\": %llu,\n"
                   " %s%s%s\"bbox\": [%lld, %lld, %lld, %lld, %lld, %lld],\n \"origins\": [",
                name, c->surface, c->labels, c->mask ? c->mask : "", c->ct_root, c->ct_key, c->ct_um, L, um, c->iso, c->band_a, c->band_b, c->layer_vox, (int)G.sign, c->thr, F,
                c->crop[0], c->crop[1], c->crop[2], c->crop[3], (unsigned long long)nsup, (unsigned long long)nink, nsup * cm2, nink * cm2,
                (unsigned long long)J.n_hit, (unsigned long long)J.n_ink, (unsigned long long)J.n_no, (unsigned long long)J.n_rej,
                cmp_json, prof_json[0] ? prof_json : "", prof_json[0] ? ",\n " : "",
                (long long)blo[0], (long long)blo[1], (long long)blo[2], (long long)(bhi[0] - blo[0]), (long long)(bhi[1] - blo[1]), (long long)(bhi[2] - blo[2]));
        /* 1024-blocks (level-0 voxels) holding written voxels */
        const int64_t bs = 1024, cs0 = (int64_t)CH << L; int nb = 0;
        int64_t ng[3]; for (int d = 0; d < 3; d++) ng[d] = (shape0[d] + bs - 1) / bs;
        uint8_t *blk = (double)ng[0] * ng[1] * ng[2] < 1e9 ? calloc((size_t)(ng[0] * ng[1] * ng[2]), 1) : nullptr;
        if (blk) {
            for (int64_t a = 0; a < J.cg[0]; a++) for (int64_t b = 0; b < J.cg[1]; b++) for (int64_t e = 0; e < J.cg[2]; e++) {
                if (!J.chunks_used[(a * J.cg[1] + b) * J.cg[2] + e]) continue;
                const int64_t z0 = a * cs0, y0 = b * cs0, x0 = e * cs0;   /* a chunk may span several blocks when 128 << L > 1024 */
                for (int64_t z = z0 / bs; z <= (z0 + cs0 - 1) / bs && z < ng[0]; z++) for (int64_t y = y0 / bs; y <= (y0 + cs0 - 1) / bs && y < ng[1]; y++)
                    for (int64_t x = x0 / bs; x <= (x0 + cs0 - 1) / bs && x < ng[2]; x++) blk[(z * ng[1] + y) * ng[2] + x] = 1;
            }
            for (int64_t k = 0; k < ng[0] * ng[1] * ng[2]; k++) if (blk[k])
                fprintf(f, "%s[%lld, %lld, %lld]", nb++ ? ", " : "", (long long)(k / (ng[1] * ng[2]) * bs), (long long)(k / ng[2] % ng[1] * bs), (long long)(k % ng[2] * bs));
            free(blk);
        }
        fprintf(f, "], \"block\": 1024}\n"); fclose(f);
        fprintf(stderr, "ink-splat: %s: %d 1024-blocks, summary %s, %.1fs\n", name, nb, sp, snow() - t0);
        free(pr); free(us); free(J.chunks_used);
        if (ct != ct0) z3_close(ct);
        store_close(os);
        rc = 0;
    }
out2:
    z3_close(ct0);
out1:
    free(G.lab); free(G.msk);
out0:
    if (has_mask) lsrc_close(&LM);
    lsrc_close(&LL); ssurf_free(&S);
    return rc;
}

/* --points JSON: 3D point labels (a VC3D point-collection file: collections named "ink" / "noink", points "p": [x, y, z] in level-0
   voxels, integer = voxel centre) rasterised as balls of --radius-um on CT level L: ink 254, no ink 0 (ink wins where balls of both
   meet), CT < iso unknown; merged into <out> like a segment splat. */
typedef struct { int64_t key; uint32_t idx; uint8_t v; } pent;
static int pent_cmp(const void *a, const void *b) { const pent *p = a, *q = b; return p->key != q->key ? (p->key < q->key ? -1 : 1) : p->idx != q->idx ? (p->idx < q->idx ? -1 : 1) : (int)p->v - (int)q->v; }
static int splat_points(const inksplat_cfg *c, const char *path, double radius_um) {
    if (!c->out || !(c->ct_um > 0) || !(radius_um > 0)) { fprintf(stderr, "ink-splat --points: need <out>, --ct-um and --radius-um > 0\n"); return 2; }
    FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr, "ink-splat: cannot read %s\n", path); return 1; }
    fseek(f, 0, SEEK_END); const long fl = ftell(f); fseek(f, 0, SEEK_SET);
    char *txt = malloc((size_t)fl + 1); const size_t got = fread(txt, 1, (size_t)fl, f); fclose(f); txt[got] = 0;
    json *j = json_parse(txt, got); free(txt);
    if (!j) { fprintf(stderr, "ink-splat: bad JSON %s\n", path); return 1; }
    store *cts = store_open(c->ct_root); const int L = c->level; const double V = (double)(1 << L), off = (V - 1) / 2, rr = fmax(0.5, radius_um / (c->ct_um * V));
    z3 *ct = pyramid_open_level(cts, c->ct_key, L, c->ct_um, c->cache), *ct0 = pyramid_open_level(cts, c->ct_key, 0, c->ct_um, c->cache);
    if (!ct || !ct0) { fprintf(stderr, "ink-splat: CT: %s\n", z3_error()); json_free(j); return 1; }
    int64_t oshape[3], shape0[3], ns[3]; memcpy(oshape, z3_meta_of(ct)->shape, sizeof oshape); memcpy(shape0, z3_meta_of(ct0)->shape, sizeof shape0);
    for (int d = 0; d < 3; d++) ns[d] = (oshape[d] + SH - 1) / SH;
    size_t cap = 1 << 16, n = 0; pent *E = malloc(cap * sizeof *E); long npt[2] = {0, 0};
    const json *cols = json_get(j, "collections");
    for (size_t ci = 0; cols && ci < cols->n; ci++) {
        const json *col = cols->items[ci]; const char *nm = json_str(json_get(col, "name"), "");
        const int v = strstr(nm, "noink") || strstr(nm, "no_ink") || strstr(nm, "no-ink") ? 0 : strstr(nm, "ink") ? 254 : -1;
        if (v < 0) { fprintf(stderr, "ink-splat: collection \"%s\" skipped\n", nm); continue; }
        const json *pts = json_get(col, "points");
        for (size_t pi = 0; pts && pi < pts->n; pi++) {
            const json *p = json_get(pts->items[pi], "p"); if (!p || p->n != 3) continue;
            const double q[3] = {(json_num(json_at(p, 2), 0) - off) / V, (json_num(json_at(p, 1), 0) - off) / V, (json_num(json_at(p, 0), 0) - off) / V};
            npt[v ? 1 : 0]++;
            for (int64_t z = (int64_t)ceil(q[0] - rr); z <= (int64_t)floor(q[0] + rr); z++) for (int64_t y = (int64_t)ceil(q[1] - rr); y <= (int64_t)floor(q[1] + rr); y++)
                for (int64_t x = (int64_t)ceil(q[2] - rr); x <= (int64_t)floor(q[2] + rr); x++) {
                    const double dz = z - q[0], dy = y - q[1], dx = x - q[2];
                    if (dz * dz + dy * dy + dx * dx > rr * rr + 1e-9 || z < 0 || y < 0 || x < 0 || z >= oshape[0] || y >= oshape[1] || x >= oshape[2]) continue;
                    if (n == cap) { cap *= 2; E = realloc(E, cap * sizeof *E); }
                    E[n++] = (pent){((z / SH) * ns[1] + y / SH) * ns[2] + x / SH, (uint32_t)(((z % SH) * SH + y % SH) * SH + x % SH), (uint8_t)v};
                }
        }
    }
    json_free(j);
    qsort(E, n, sizeof *E, pent_cmp);
    char lv[64], adir[1400], zj[1500]; z3w_level_name(c->ct_um * V, lv, sizeof lv); snprintf(adir, sizeof adir, "%s/%s", c->out, lv); snprintf(zj, sizeof zj, "%s/zarr.json", adir);
    struct stat sst; const int exists = !stat(zj, &sst);
    if (exists && !c->merge) { fprintf(stderr, "ink-splat: %s exists (use --merge)\n", adir); free(E); return 1; }
    mkdir_p(c->out);
    z3w *w = z3w_create(adir, oshape, SH, 0, 255, g_attrs); store *os = store_open(c->out); z3 *old = exists ? z3_open(os, lv, nullptr) : nullptr;
    uint8_t *buf = malloc((size_t)SH * SH * SH), *ctc[NCS * NCS * NCS] = {0};
    long nink = 0, nno = 0, nrej = 0; int rc = 0, nsh = 0;
    for (size_t i = 0; i < n && !rc;) {
        size_t e = i; while (e < n && E[e].key == E[i].key) e++;
        const int64_t key = E[i].key, so[3] = {key / (ns[1] * ns[2]) * SH, key / ns[2] % ns[1] * SH, key % ns[2] * SH}, sn[3] = {SH, SH, SH};
        if (old && z3_shard_present(old, so[0] / SH, so[1] / SH, so[2] / SH) == 1) { if (z3_read(old, so, sn, buf, 4)) rc = 1; } else memset(buf, 255, (size_t)SH * SH * SH);
        for (size_t k = i; k < e && !rc; k++) {
            if (k + 1 < e && E[k + 1].idx == E[k].idx) continue;   /* the largest value of a voxel is last */
            const uint32_t id = E[k].idx; const int z = (int)(id / (SH * SH)), y = (int)(id / SH % SH), x = (int)(id % SH);
            const int cc = ((z >> 7) * NCS + (y >> 7)) * NCS + (x >> 7);
            if (!ctc[cc]) { ctc[cc] = malloc((size_t)CH * CH * CH); const int64_t co[3] = {so[0] + (z >> 7) * CH, so[1] + (y >> 7) * CH, so[2] + (x >> 7) * CH}, cn[3] = {CH, CH, CH};
                            if (z3_read(ct, co, cn, ctc[cc], 4)) { rc = 1; break; } }
            const uint8_t cv = ctc[cc][(((size_t)(z & 127) * CH) + (y & 127)) * CH + (x & 127)];
            if (!cv || cv < c->iso) { nrej++; continue; }
            buf[id] = buf[id] == 255 ? E[k].v : (buf[id] > E[k].v ? buf[id] : E[k].v);
            if (E[k].v) nink++; else nno++;
        }
        for (int cc = 0; cc < NCS * NCS * NCS; cc++) { free(ctc[cc]); ctc[cc] = nullptr; }
        if (!rc && z3w_write_shard(w, so[0] / SH, so[1] / SH, so[2] / SH, buf, 4)) rc = 1;
        nsh++; i = e;
    }
    z3w_close(w); if (old) z3_close(old); free(buf); free(E); z3_close(ct); z3_close(ct0);
    fprintf(stderr, "ink-splat --points: %ld ink + %ld no-ink points (ball radius %.2f voxels at level %d): %ld ink + %ld no-ink voxels, %ld below iso, %d shards\n",
            npt[1], npt[0], rr, L, nink, nno, nrej, nsh);
    if (rc) { fprintf(stderr, "ink-splat --points: failed: %s %s\n", z3_error(), z3w_error()); return 1; }
    return splat_pyramid(c, shape0) ? 1 : 0;
}

static const char *sopt(int argc, char **argv, const char *k, const char *d) { for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
static int sflag(int argc, char **argv, const char *k) { for (int i = 2; i < argc; i++) if (!strcmp(argv[i], k)) return 1; return 0; }
static int sidx(int argc, char **argv, const char *k) { for (int i = 2; i < argc; i++) if (!strcmp(argv[i], k)) return i; return -1; }

int cmd_ink_splat(int argc, char **argv) {
    inksplat_cfg c; inksplat_defaults(&c);
    const int ci = sidx(argc, argv, "--ct"), pi = sidx(argc, argv, "--pred");
    if (argc < 3 || ci < 0 || ci + 2 >= argc) {
        fprintf(stderr, "usage: ufsm ink-splat <out-group|-> --surface TIFXYZ --labels L [--mask M] --ct ROOT KEY --ct-um U [--level 0] --iso I\n"
                        "         [--band A,B (layers, -4,4)] [--layer-vox 1] [--flip] [--thr 0.5] [--label-z Z] [--label-scale F] [--label-offset r,c] [--crop c0,r0,w,h]\n"
                        "         [--merge] [--coarse N] [--name NAME] [--summary JSON] [--preview JPG] [--pyramid-only]\n         ufsm ink-splat <out-group> --points VC3D.json --radius-um 5 --ct ROOT KEY --ct-um U [--level 0] --iso I [--merge]   (3D point labels) [--threads 6] [--cache DIR]\n"
                        "         [--pred ROOT KEY [--pred-level L] [--pred-thr 128] [--pred-skip V] [--profile T] [--stride 4]]\n"
                        "  L / M: a .tif / .png, or a zarr array dir (v3 2-D, v2 3-D at --label-z), local or https; '-' as out: profile only\n");
        return 2;
    }
    c.out = strcmp(argv[2], "-") ? argv[2] : nullptr;
    c.surface = sopt(argc, argv, "--surface", nullptr); c.labels = sopt(argc, argv, "--labels", nullptr); c.mask = sopt(argc, argv, "--mask", nullptr);
    c.ct_root = argv[ci + 1]; c.ct_key = argv[ci + 2]; c.ct_um = atof(sopt(argc, argv, "--ct-um", "0")); c.level = atoi(sopt(argc, argv, "--level", "0"));
    c.iso = atoi(sopt(argc, argv, "--iso", "1"));
    if (sscanf(sopt(argc, argv, "--band", "-4,4"), "%lf,%lf", &c.band_a, &c.band_b) != 2) { fprintf(stderr, "ink-splat: --band A,B\n"); return 2; }
    c.layer_vox = atof(sopt(argc, argv, "--layer-vox", "1")); c.flip = sflag(argc, argv, "--flip"); c.thr = atof(sopt(argc, argv, "--thr", "0.5"));
    c.label_z = atoi(sopt(argc, argv, "--label-z", "-1")); c.mask_z = atoi(sopt(argc, argv, "--mask-z", "-1")); c.label_scale = atof(sopt(argc, argv, "--label-scale", "0"));
    if (sscanf(sopt(argc, argv, "--label-offset", "0,0"), "%lf,%lf", c.label_off, c.label_off + 1) != 2) { fprintf(stderr, "ink-splat: --label-offset r,c\n"); return 2; }
    const char *cr = sopt(argc, argv, "--crop", nullptr);
    if (cr && sscanf(cr, "%d,%d,%d,%d", c.crop, c.crop + 1, c.crop + 2, c.crop + 3) != 4) { fprintf(stderr, "ink-splat: --crop c0,r0,w,h\n"); return 2; }
    c.merge = sflag(argc, argv, "--merge"); c.coarse = atoi(sopt(argc, argv, "--coarse", "-1")); c.name = sopt(argc, argv, "--name", nullptr); c.summary = sopt(argc, argv, "--summary", nullptr); c.preview = sopt(argc, argv, "--preview", nullptr);
    c.threads = atoi(sopt(argc, argv, "--threads", "6")); c.cache = sopt(argc, argv, "--cache", nullptr);
    if (pi >= 0) { if (pi + 2 >= argc) { fprintf(stderr, "ink-splat: --pred ROOT KEY\n"); return 2; } c.pred_root = argv[pi + 1]; c.pred_key = argv[pi + 2]; }
    c.pred_level = atoi(sopt(argc, argv, "--pred-level", "-1")); c.pred_thr = atoi(sopt(argc, argv, "--pred-thr", "128")); c.pred_skip = atoi(sopt(argc, argv, "--pred-skip", "-1"));
    c.profile = atoi(sopt(argc, argv, "--profile", "0")); c.stride = atoi(sopt(argc, argv, "--stride", "4"));
    store_global_init();
    if (sflag(argc, argv, "--pyramid-only")) return splat_pyramid_only(&c);
    const char *pts = sopt(argc, argv, "--points", nullptr);
    if (pts) return splat_points(&c, pts, atof(sopt(argc, argv, "--radius-um", "5")));
    if (!c.surface || !c.labels) { fprintf(stderr, "ink-splat: need --surface and --labels (or --points / --pyramid-only)\n"); return 2; }
    return inksplat_run(&c);
}

/* ufsm ink-score --surface TIFXYZ --labels L [--mask M] --map NAME=MAP.tif [--map ..] [--label-z K] [--label-scale F] [--surf-um U] [--cache DIR]
   [--threads 8] [--row TAG]
   2D ink maps on the surface's render grid (ufsm render / ink-ext output: pixel = canvas px = tifxyz grid / scale, uint8 p x 255)
   scored against the segment's published 2D ink labels within its supervision mask (labels / mask >= 128, 0/1 rasters read as
   0/255), on the label pixel grid (label px (r, c) = canvas ((c + 0.5) F - 0.5, (r + 0.5) F - 0.5), F canvas px per label px as
   ink-splat: the label OME pixel size / mesh voxel size, else the canvas size ratio). Each label pixel takes the mean of the map
   over its F x F canvas footprint (bilinear when F < 1.5); pixels whose surface cell is invalid are skipped. Per map: supervised
   px, ink fraction, F1 / precision / recall at the F1-optimal threshold and at p >= 0.5 (byte >= 128), ROC AUC.
   --row TAG: one TSV line per map "TAG name n inkfrac f1best thr f1_05 prec05 rec05 auc" on stdout. */
int cmd_ink_score(int argc, char **argv) {
    const char *surf = sopt(argc, argv, "--surface", nullptr), *lab = sopt(argc, argv, "--labels", nullptr), *msk = sopt(argc, argv, "--mask", nullptr);
    const char *cache = sopt(argc, argv, "--cache", nullptr), *row = sopt(argc, argv, "--row", nullptr);
    if (!surf || !lab || sidx(argc, argv, "--map") < 0) {
        fprintf(stderr, "usage: ufsm ink-score --surface TIFXYZ --labels L [--mask M] --map NAME=MAP.tif [--map ..] [--label-z K] [--label-scale F] [--surf-um U] [--cache DIR] [--row TAG]\n"); return 2;
    }
    ssurf S; if (ssurf_load(&S, surf)) return 1;
    { const double su = atof(sopt(argc, argv, "--surf-um", "0")); if (su > 0) S.vox_um = su; }   /* the mesh voxel size when its meta has none */
    lsrc LL, LM; const int lz = atoi(sopt(argc, argv, "--label-z", "-1"));
    if (lsrc_open(&LL, lab, lz, cache)) { ssurf_free(&S); return 1; }
    if (msk && lsrc_open(&LM, msk, lz, cache)) { lsrc_close(&LL); ssurf_free(&S); return 1; }
    if (msk && (LM.H != LL.H || LM.W != LL.W)) { fprintf(stderr, "ink-score: mask %lld x %lld != labels %lld x %lld\n", (long long)LM.W, (long long)LM.H, (long long)LL.W, (long long)LL.H); return 1; }
    double F = atof(sopt(argc, argv, "--label-scale", "0"));
    const double ew = S.w / S.scale, eh = S.h / S.scale;
    if (!(F > 0)) {
        F = 0.5 * (ew / (double)LL.W + eh / (double)LL.H);
        const double p2 = pow(2, round(log2(F))); if (fabs(F - p2) < 0.002 * p2) F = p2;
        if (LL.um > 0 && S.vox_um > 0) F = LL.um / S.vox_um;
    }
    const int nth = atoi(sopt(argc, argv, "--threads", "8"));
    const size_t np = (size_t)LL.W * LL.H;
    uint8_t *L = malloc(np), *M = msk ? malloc(np) : nullptr;
    if (!L || (msk && !M) || lsrc_read(&LL, 0, 0, LL.H, LL.W, L, nth) || (msk && lsrc_read(&LM, 0, 0, LL.H, LL.W, M, nth))) { fprintf(stderr, "ink-score: label read failed\n"); return 1; }
    uint8_t *R[2] = {L, M};
    for (int k = 0; k < 2; k++) { if (!R[k]) continue; uint8_t mx = 0; for (size_t i = 0; i < np; i++) if (R[k][i] > mx) mx = R[k][i]; if (mx == 1) for (size_t i = 0; i < np; i++) R[k][i] *= 255; }
    fprintf(stderr, "ink-score: %s: labels %lld x %lld, %.4g canvas px per label px (canvas %.0f x %.0f)\n", S.name, (long long)LL.W, (long long)LL.H, F, ew, eh);
    int rc = 0;
    for (int a = 2; a + 1 < argc; a++) {
        if (strcmp(argv[a], "--map")) continue;
        char nm[256]; const char *path = argv[a + 1], *eq = strchr(path, '=');
        if (eq && eq - path < (long)sizeof nm) { snprintf(nm, sizeof nm, "%.*s", (int)(eq - path), path); path = eq + 1; } else snprintf(nm, sizeof nm, "%s", path);
        int mw, mh; uint8_t *mp = img_read_u8(path, &mw, &mh);
        if (!mp) { fprintf(stderr, "ink-score: cannot read %s\n", path); rc = 1; continue; }
        uint64_t hi[256] = {0}, hn[256] = {0};
        const int foot = F >= 1.5 ? (int)lround(F) : 0;
        for (int64_t r = 0; r < LL.H; r++) for (int64_t c = 0; c < LL.W; c++) {
            const size_t i = (size_t)r * LL.W + c; if (M && M[i] < 128) continue;
            const double cx = (c + 0.5) * F - 0.5, cy = (r + 0.5) * F - 0.5;
            double P[3], N[3]; if (ssurf_pn(&S, cx * S.scale, cy * S.scale, P, N)) continue;
            double v;
            if (foot) {   /* the label pixel's canvas footprint */
                const int x0 = (int)floor(c * F), y0 = (int)floor(r * F); double s = 0; int n = 0;
                for (int y = y0; y < y0 + foot; y++) for (int x = x0; x < x0 + foot; x++) if (x >= 0 && y >= 0 && x < mw && y < mh) { s += mp[(size_t)y * mw + x]; n++; }
                if (!n) continue;
                v = s / n;
            } else {
                if (cx < 0 || cy < 0 || cx > mw - 1 || cy > mh - 1) continue;
                const int x0 = (int)cx, y0 = (int)cy, x1 = x0 + 1 < mw ? x0 + 1 : x0, y1 = y0 + 1 < mh ? y0 + 1 : y0; const double fx = cx - x0, fy = cy - y0;
                v = (1 - fy) * ((1 - fx) * mp[(size_t)y0 * mw + x0] + fx * mp[(size_t)y0 * mw + x1]) + fy * ((1 - fx) * mp[(size_t)y1 * mw + x0] + fx * mp[(size_t)y1 * mw + x1]);
            }
            const int b = (int)lround(v); if (L[i] >= 128) hi[b]++; else hn[b]++;
        }
        uint64_t ti = 0, tn = 0; for (int b = 0; b < 256; b++) { ti += hi[b]; tn += hn[b]; }
        double best = 0, bthr = 0, f05 = 0, p05 = 0, r05 = 0, wrong = 0; uint64_t tp = 0, fp = 0;
        for (int t = 255; t >= 0; t--) {   /* predict ink where byte >= t */
            wrong += (double)hi[t] * (fp + 0.5 * hn[t]);   /* ink pixels with no-ink ranked above them (ties half) */
            tp += hi[t]; fp += hn[t];
            const double pr = tp + fp ? (double)tp / (tp + fp) : 0, re = ti ? (double)tp / ti : 0, f = pr + re > 0 ? 2 * pr * re / (pr + re) : 0;
            if (f > best) { best = f; bthr = t / 255.0; }
            if (t == 128) { f05 = f; p05 = pr; r05 = re; }
        }
        const double auc = ti && tn ? 1 - wrong / ((double)ti * tn) : 0;
        const uint64_t n = ti + tn;
        fprintf(stderr, "ink-score: %s: %llu px, ink %.3f, F1 best %.4f (p >= %.3f), F1@0.5 %.4f (prec %.3f rec %.3f), AUC %.4f\n", nm,
                (unsigned long long)n, n ? (double)ti / n : 0, best, bthr, f05, p05, r05, auc);
        if (row) printf("%s\t%s\t%llu\t%.4f\t%.4f\t%.3f\t%.4f\t%.4f\t%.4f\t%.4f\n", row, nm, (unsigned long long)n, n ? (double)ti / n : 0, best, bthr, f05, p05, r05, auc);
        fflush(stdout);
        free(mp);
    }
    free(L); free(M); lsrc_close(&LL); if (msk) lsrc_close(&LM); ssurf_free(&S);
    return rc;
}
