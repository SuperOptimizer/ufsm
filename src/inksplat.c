/* ufsm ink-splat (inksplat.h): published 2D ink labels -> sparse 3D label zarr on the CT grid, and the depth profile of a 3D ink
   prediction along the labelled surface. */
#include "inksplat.h"
#include "dcache.h"
#include "inkflygpu.h"
#include "inkmil.h"
#include <stdarg.h>
#include <sys/stat.h>
#include "sysinfo.h"
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
#include "zcache.h"

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
                {   /* the chunk's columns [ca, cb) that fall in the window and the raster (row memcpy: the per-byte loop was half of an index open) */
                    int64_t ca = c0 - g1 > 0 ? c0 - g1 : 0, cb = L->chunk[1]; if (g1 + cb > c0 + w) cb = c0 + w - g1; if (g1 + cb > L->W) cb = L->W - g1;
                    if (cb > ca) for (int64_t r = 0; r < L->chunk[0]; r++) {
                        const int64_t rr = g0 + r - r0; if (rr < 0 || rr >= h || g0 + r >= L->H) continue;
                        memcpy(out + (size_t)rr * w + (g1 + ca - c0), dec + (size_t)r * L->chunk[1] + ca, (size_t)(cb - ca));
                    }
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
    int8_t *face;                                      /* --face: per label pixel the open-face offset (level-0 voxels along sign N), FACE_NONE */
} geo;
enum { FACE_NONE = -128 };
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
    uint64_t n_ink, n_no, n_rej, n_hit, n_excl, cm[2][2], vh[255];   /* cm[splat ink][pred ink] over the written known voxels (--pred at this level) */
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
                const double ta = G->face ? (G->face[(size_t)r * G->lw + cc] == FACE_NONE ? NAN : G->face[(size_t)r * G->lw + cc] + c->face_a) : G->ta;
                if (isnan(ta)) continue;
                const uint8_t lv = G->lab[(size_t)r * G->lw + cc]; const int ink = lv >= 128;
                for (int a = 0; a < J->nsub; a++) for (int b = 0; b < J->nsub; b++) {
                    double gu, gv, P[3], N[3];
                    px_grid(G, r + (a + 0.5) / J->nsub - 0.5, cc + (b + 0.5) / J->nsub - 0.5, &gu, &gv);
                    if (ssurf_pn(G->S, gu, gv, P, N)) continue;
                    for (int k = 0; k < J->nt; k++) {
                        const double tt = (ta + k * J->tstep) * G->sign;
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
        uint64_t ni = 0, nn = 0, nr = 0, nh = 0, nx = 0, cm[2][2] = {{0, 0}, {0, 0}}, vh[255] = {0}; int any = 0;
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
                if (c->nexcl) {   /* --exclude: the voxel's level-0 extent [j V, (j + 1) V) touches a held-out box */
                    const int64_t Vi = (int64_t)J->V; int ex = 0;
                    for (int e = 0; e < c->nexcl && !ex; e++) {
                        const int64_t *b = c->excl[e];
                        ex = (z + 1) * Vi > b[0] && z * Vi < b[0] + b[3] && (y + 1) * Vi > b[1] && y * Vi < b[1] + b[4] && (x + 1) * Vi > b[2] && x * Vi < b[2] + b[5];
                    }
                    if (ex) { nx++; continue; }
                }
                if (!ctb[k] || ctb[k] < c->iso) { nr++; continue; }
                const int isink = ac->ni[k] && ac->ni[k] >= c->thr * h;
                const uint8_t v = (uint8_t)(((isink ? ac->pi[k] : ac->pn[k]) * 254 + 127) / 255);
                const size_t bi = ((size_t)(z - so[0]) * SH + (size_t)(y - so[1])) * SH + (size_t)(x - so[2]);
                buf[bi] = buf[bi] == 255 ? v : (buf[bi] > v ? buf[bi] : v); vh[v]++;
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
        J->n_ink += ni; J->n_no += nn; J->n_rej += nr; J->n_hit += nh; J->n_excl += nx; for (int q = 0; q < 255; q++) J->vh[q] += vh[q];
        for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) J->cm[a][b] += cm[a][b];
        for (int d = 0; d < 3; d++) { if (blo[d] < J->bb_lo[d]) J->bb_lo[d] = blo[d]; if (bhi[d] > J->bb_hi[d]) J->bb_hi[d] = bhi[d]; }
        pthread_mutex_unlock(&J->mu);
    }
done:
    for (int ci = 0; ci < NCS * NCS * NCS; ci++) free(A[ci]);
    free(A); free(buf); free(ctb); free(pb);
    return nullptr;
}

/* ---------------- --face: the open face per label pixel ---------------- */
enum { FBLK = 4 };   /* tiles per face block side (256 label px): one CT box per block, fewer repeated chunk decodes */
typedef struct { const geo *G; double V, off; z3 *ct; int nbx, nby, nblk; _Atomic int next, failed; _Atomic uint64_t nsup, nface; uint64_t hist[256]; pthread_mutex_t mu; } fjob;
static inline float tri_box(const uint8_t *b, const int64_t n[3], double z, double y, double x);
/* the open face of one CT column s[0 .. n) (t = lo + i): the first papyrus run of >= run samples ending at t >= -4 and starting at
   t <= 8, followed by >= gap samples below iso; FACE_NONE when the column has none */
static int face_of(const float *s, int n, int lo, int iso, int run, int gap) {
    for (int i = 0; i < n;) {
        if (s[i] < iso) { i++; continue; }
        int j = i; while (j < n && s[j] >= iso) j++;   /* papyrus run [i, j) */
        const int t0 = lo + i, t1 = lo + j - 1;
        if (j - i >= run && t1 >= -4 && t0 <= 8) {
            int g = 0; while (j + g < n && s[j + g] < iso && g < gap) g++;
            return g >= gap && t1 >= -127 && t1 <= 127 ? t1 : FACE_NONE;
        }
        if (t0 > 8) return FACE_NONE;
        i = j;
    }
    return FACE_NONE;
}
static void *face_worker(void *arg) {
    fjob *J = arg; const geo *G = J->G; const inksplat_cfg *c = G->c;
    const int lo = (int)floor(c->face_lo), hi = (int)ceil(c->face_hi), n = hi - lo + 1;
    float *raw = malloc(n * sizeof *raw), *sm = malloc(n * sizeof *sm); uint64_t h[256] = {0}, ns = 0, nf = 0;
    for (int b; (b = atomic_fetch_add(&J->next, 1)) < J->nblk && !atomic_load(&J->failed);) {
        const int bx = b % J->nbx, by = b / J->nbx;
        double lo3[3] = {1e300, 1e300, 1e300}, hi3[3] = {-1e300, -1e300, -1e300}; int any = 0;
        for (int ty = by * FBLK; ty < (by + 1) * FBLK && ty < G->ntiles_y; ty++) for (int tx = bx * FBLK; tx < (bx + 1) * FBLK && tx < G->ntiles_x; tx++) {
            double a[3], e[3]; if (!tile_box(G, tx, ty, (double)(abs(lo) > abs(hi) ? abs(lo) : abs(hi)) + 1, a, e)) continue;
            for (int d = 0; d < 3; d++) { if (a[d] < lo3[d]) lo3[d] = a[d]; if (e[d] > hi3[d]) hi3[d] = e[d]; } any = 1;
        }
        if (!any) continue;
        int64_t o[3], nb[3]; size_t nv = 1;
        for (int d = 0; d < 3; d++) { o[d] = (int64_t)floor((lo3[d] - J->off) / J->V) - 1; nb[d] = (int64_t)ceil((hi3[d] - J->off) / J->V) + 2 - o[d]; nv *= (size_t)nb[d]; }
        if (nv > (size_t)1 << 31) { fprintf(stderr, "ink-splat: face block %d box too large, skipped\n", b); continue; }
        uint8_t *cb = malloc(nv);
        if (!cb || z3_read(J->ct, o, nb, cb, 1)) { fprintf(stderr, "ink-splat: face CT read failed\n"); free(cb); atomic_store(&J->failed, 1); break; }
        const int64_t r0 = (int64_t)by * FBLK * TS, c0 = (int64_t)bx * FBLK * TS, r1 = r0 + FBLK * TS < G->lh ? r0 + FBLK * TS : G->lh, c1 = c0 + FBLK * TS < G->lw ? c0 + FBLK * TS : G->lw;
        for (int64_t r = r0; r < r1; r++) for (int64_t cc = c0; cc < c1; cc++) {
            if (!px_sup(G, r, cc)) continue;
            double gu, gv, P[3], N[3]; px_grid(G, (double)r, (double)cc, &gu, &gv);
            if (ssurf_pn(G->S, gu, gv, P, N)) continue;
            ns++;
            for (int i = 0; i < n; i++) {
                const double tt = (lo + i) * G->sign; double q[3];
                for (int d = 0; d < 3; d++) q[d] = (P[d] + tt * N[d] - J->off) / J->V - o[d];
                const float v = tri_box(cb, nb, q[0], q[1], q[2]); raw[i] = v < 0 ? 0 : v;
            }
            for (int i = 0; i < n; i++) sm[i] = 0.25f * (raw[i > 0 ? i - 1 : 0] + 2 * raw[i] + raw[i < n - 1 ? i + 1 : n - 1]);
            const int f = face_of(sm, n, lo, c->iso, c->face_run, c->face_gap);
            G->face[(size_t)r * G->lw + cc] = (int8_t)f;
            if (f != FACE_NONE) { nf++; h[f + 128]++; }
        }
        free(cb);
    }
    atomic_fetch_add(&J->nsup, ns); atomic_fetch_add(&J->nface, nf);
    pthread_mutex_lock(&J->mu); for (int k = 0; k < 256; k++) J->hist[k] += h[k]; pthread_mutex_unlock(&J->mu);
    free(raw); free(sm); return nullptr;
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
                const int fo = G->face ? G->face[(size_t)r * G->lw + cc] : 0; if (fo == FACE_NONE) continue;   /* --face: t relative to the open face */
                const int cls = G->lab[(size_t)r * G->lw + cc] >= 128;
                double gu, gv, Pp[3], N[3]; px_grid(G, (double)r, (double)cc, &gu, &gv);
                if (ssurf_pn(G->S, gu, gv, Pp, N)) continue;
                for (int i = -T; i <= T; i++) {
                    const double tt = (fo + i * c->layer_vox) * G->sign; double q[3];
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
    c->face_a = -6; c->face_b = 0; c->face_lo = -6; c->face_hi = 20; c->face_run = 3; c->face_gap = 2;
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

    char face_json[2048] = "";
    if (c->face) {   /* --face: the open face per supervised label pixel from the output level's CT (before the profile, which then runs face-relative) */
        const double V = (double)(1 << c->level); z3 *ct = c->level ? pyramid_open_level(cts, c->ct_key, c->level, c->ct_um, c->cache) : ct0;
        if (!ct) { fprintf(stderr, "ink-splat: CT level %d: %s\n", c->level, z3_error()); goto out2; }
        G.face = malloc((size_t)G.lw * G.lh); if (!G.face) goto out2;
        memset(G.face, (uint8_t)(int8_t)FACE_NONE, (size_t)G.lw * G.lh);
        fjob FJ = {0}; FJ.G = &G; FJ.V = V; FJ.off = (V - 1) / 2; FJ.ct = ct; FJ.nbx = (G.ntiles_x + FBLK - 1) / FBLK; FJ.nby = (G.ntiles_y + FBLK - 1) / FBLK;
        FJ.nblk = FJ.nbx * FJ.nby; pthread_mutex_init(&FJ.mu, nullptr);
        pthread_t th[64]; const int nt = c->threads < 1 ? 1 : c->threads > 64 ? 64 : c->threads;
        for (int i = 0; i < nt; i++) pthread_create(&th[i], nullptr, face_worker, &FJ);
        for (int i = 0; i < nt; i++) pthread_join(th[i], nullptr);
        if (ct != ct0) z3_close(ct);
        if (atomic_load(&FJ.failed)) goto out2;
        const uint64_t ns = atomic_load(&FJ.nsup), nf = atomic_load(&FJ.nface); double cum = 0; int q[5] = {0}; const double qs[5] = {0.1, 0.25, 0.5, 0.75, 0.9};
        for (int k = 0, qi = 0; k < 256 && qi < 5; k++) { cum += FJ.hist[k]; while (qi < 5 && nf && cum >= qs[qi] * nf) q[qi++] = k - 128; }
        fprintf(stderr, "ink-splat: %s: open face found at %llu of %llu supervised px (%.1f%%); face offset p10 / p25 / p50 / p75 / p90 %d / %d / %d / %d / %d level-0 voxels along %+dN; band [f%+g, f%+g]\n",
                name, (unsigned long long)nf, (unsigned long long)ns, ns ? 100.0 * nf / ns : 0, q[0], q[1], q[2], q[3], q[4], (int)G.sign, c->face_a, c->face_b);
        int o = snprintf(face_json, sizeof face_json, "\"face\": {\"band\": [%g, %g], \"search\": [%g, %g], \"run\": %d, \"gap\": %d, \"px\": %llu, \"found\": %llu, \"q10_25_50_75_90\": [%d, %d, %d, %d, %d], \"hist\": {",
                         c->face_a, c->face_b, c->face_lo, c->face_hi, c->face_run, c->face_gap, (unsigned long long)ns, (unsigned long long)nf, q[0], q[1], q[2], q[3], q[4]);
        for (int k = 0, first = 1; k < 256 && o < (int)sizeof face_json - 40; k++) if (FJ.hist[k]) { o += snprintf(face_json + o, sizeof face_json - o, "%s\"%d\": %llu", first ? "" : ", ", k - 128, (unsigned long long)FJ.hist[k]); first = 0; }
        snprintf(face_json + o, sizeof face_json - o, "}},\n ");
        G.ta = c->face_lo + c->face_a; G.tb = c->face_hi + c->face_b;   /* (the reach of the tile -> shard pairs) */
    }
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
        const double span = c->face ? c->face_b - c->face_a : G.tb - G.ta; J.nt = (int)ceil(span / (0.45 * V)) + 1; J.tstep = J.nt > 1 ? span / (J.nt - 1) : 0;
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
        fprintf(stderr, "ink-splat: %s: %llu hit voxels, %llu ink + %llu no-ink written, %llu below iso / masked, %llu excluded (%.1f%% on papyrus) in %.1fs\n", name,
                (unsigned long long)J.n_hit, (unsigned long long)J.n_ink, (unsigned long long)J.n_no, (unsigned long long)J.n_rej, (unsigned long long)J.n_excl, J.n_hit ? 100.0 * known / J.n_hit : 0, snow() - t0);
        char cmp_json[8192] = "";
        if (have_pred) {
            const double tp = J.cm[1][1], fp = J.cm[1][0], fn = J.cm[0][1], tn = J.cm[0][0], tot = tp + fp + fn + tn;
            printf("# voxelwise vs prediction >= %d over the %.0f written voxels: agreement %.4f, ink IoU %.4f, precision %.4f, recall %.4f (splat ink %.0f, prediction ink %.0f)\n",
                   c->pred_thr, tot, tot ? (tp + tn) / tot : 0, tp + fp + fn ? tp / (tp + fp + fn) : 0, tp + fp ? tp / (tp + fp) : 0, tp + fn ? tp / (tp + fn) : 0, tp + fp, tp + fn);
            snprintf(cmp_json, sizeof cmp_json, "\"voxelwise\": {\"pred_thr\": %d, \"tp\": %.0f, \"fp\": %.0f, \"fn\": %.0f, \"tn\": %.0f, \"agreement\": %.5f, \"iou\": %.5f},",
                     c->pred_thr, tp, fp, fn, tn, tot ? (tp + tn) / tot : 0, tp + fp + fn ? tp / (tp + fp + fn) : 0);
            fflush(stdout);
        }
        {   /* the written values of this segment (0 .. 254, before the merge): soft-label percentiles across segments */
            size_t o = strlen(cmp_json); o += (size_t)snprintf(cmp_json + o, sizeof cmp_json - o, "\"value_hist\": [");
            for (int q = 0; q < 255 && o < sizeof cmp_json - 32; q++) o += (size_t)snprintf(cmp_json + o, sizeof cmp_json - o, q ? ", %llu" : "%llu", (unsigned long long)J.vh[q]);
            snprintf(cmp_json + o, sizeof cmp_json - o, "],");
        }
        /* coarse levels (known-max pooling) and the group */
        if (splat_pyramid(c, shape0)) goto out2;
        /* summary */
        char sp[1400]; if (c->summary) snprintf(sp, sizeof sp, "%s", c->summary); else { snprintf(sp, sizeof sp, "%s/segments", c->out); mkdir_p(sp); snprintf(sp, sizeof sp, "%s/segments/%s.json", c->out, name); }
        char cmp_all[12288]; snprintf(cmp_all, sizeof cmp_all, "%s%s", face_json, cmp_json);
        FILE *f = fopen(sp, "w");
        if (!f) { fprintf(stderr, "ink-splat: cannot write %s\n", sp); goto out2; }
        int64_t blo[3], bhi[3]; for (int d = 0; d < 3; d++) { blo[d] = J.bb_lo[d] == INT64_MAX ? 0 : J.bb_lo[d] << L; bhi[d] = J.bb_hi[d] < 0 ? 0 : ((J.bb_hi[d] + 1) << L); }
        fprintf(f, "{\"name\": \"%s\", \"surface\": \"%s\", \"labels\": \"%s\", \"mask\": \"%s\", \"ct\": \"%s/%s\", \"ct_um\": %g, \"level\": %d, \"um\": %g, \"iso\": %d,\n"
                   " \"band\": [%g, %g], \"layer_vox\": %g, \"sign\": %d, \"thr\": %g, \"label_px_per_canvas\": %g, \"crop\": [%d, %d, %d, %d],\n"
                   " \"label_px\": %llu, \"ink_px\": %llu, \"label_cm2\": %.4f, \"ink_cm2\": %.4f, \"hit_voxels\": %llu, \"ink_voxels\": %llu, \"noink_voxels\": %llu, \"rejected_voxels\": %llu, \"excluded_voxels\": %llu,\n"
                   " %s%s%s\"bbox\": [%lld, %lld, %lld, %lld, %lld, %lld],\n \"origins\": [",
                name, c->surface, c->labels, c->mask ? c->mask : "", c->ct_root, c->ct_key, c->ct_um, L, um, c->iso, c->band_a, c->band_b, c->layer_vox, (int)G.sign, c->thr, F,
                c->crop[0], c->crop[1], c->crop[2], c->crop[3], (unsigned long long)nsup, (unsigned long long)nink, nsup * cm2, nink * cm2,
                (unsigned long long)J.n_hit, (unsigned long long)J.n_ink, (unsigned long long)J.n_no, (unsigned long long)J.n_rej, (unsigned long long)J.n_excl,
                cmp_all, prof_json[0] ? prof_json : "", prof_json[0] ? ",\n " : "",
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
    free(G.lab); free(G.msk); free(G.face);
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
                        "         [--merge] [--coarse N] [--name NAME] [--summary JSON] [--preview JPG] [--pyramid-only] [--exclude z,y,x,nz,ny,nx (level-0, repeatable)]\n         ufsm ink-splat <out-group> --points VC3D.json --radius-um 5 --ct ROOT KEY --ct-um U [--level 0] --iso I [--merge]   (3D point labels) [--threads 6] [--cache DIR]\n"
                        "         [--pred ROOT KEY [--pred-level L] [--pred-thr 128] [--pred-skip V] [--profile T] [--stride 4]]\n"
                        "         [--face A,B [--face-search -6,20] [--face-run 3] [--face-gap 2]]   (the band per pixel relative to the open face on +N)\n"
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
    c.threads = ufsm_threads(atoi(sopt(argc, argv, "--threads", "0")), 6); c.cache = sopt(argc, argv, "--cache", nullptr);
    for (int i = 2; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--exclude")) continue;
        long long b[6];
        if (c.nexcl >= 64 || sscanf(argv[i + 1], "%lld,%lld,%lld,%lld,%lld,%lld", b, b + 1, b + 2, b + 3, b + 4, b + 5) != 6) { fprintf(stderr, "ink-splat: --exclude z,y,x,nz,ny,nx (at most 64)\n"); return 2; }
        for (int d = 0; d < 6; d++) c.excl[c.nexcl][d] = b[d];
        c.nexcl++;
    }
    if (pi >= 0) { if (pi + 2 >= argc) { fprintf(stderr, "ink-splat: --pred ROOT KEY\n"); return 2; } c.pred_root = argv[pi + 1]; c.pred_key = argv[pi + 2]; }
    c.pred_level = atoi(sopt(argc, argv, "--pred-level", "-1")); c.pred_thr = atoi(sopt(argc, argv, "--pred-thr", "128")); c.pred_skip = atoi(sopt(argc, argv, "--pred-skip", "-1"));
    { const char *fa = sopt(argc, argv, "--face", nullptr), *fs = sopt(argc, argv, "--face-search", nullptr);
      if (fa && (sscanf(fa, "%lf,%lf", &c.face_a, &c.face_b) != 2 || c.face_b < c.face_a)) { fprintf(stderr, "ink-splat: --face A,B (A <= B)\n"); return 2; }
      if (fs && (sscanf(fs, "%lf,%lf", &c.face_lo, &c.face_hi) != 2 || c.face_lo > -4 || c.face_hi < 8 || c.face_hi - c.face_lo > 250)) { fprintf(stderr, "ink-splat: --face-search LO,HI (LO <= -4, HI >= 8)\n"); return 2; }
      c.face = fa != nullptr; c.face_run = atoi(sopt(argc, argv, "--face-run", "3")); c.face_gap = atoi(sopt(argc, argv, "--face-gap", "2"));
      if (c.face && (c.face_run < 1 || c.face_gap < 1)) { fprintf(stderr, "ink-splat: --face-run / --face-gap >= 1\n"); return 2; } }
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
    const int nth = ufsm_threads(atoi(sopt(argc, argv, "--threads", "0")), 8);
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

/* ufsm ink-vscore: voxelwise score of a 3-D ink prediction (prob254, 255 unknown) against a 3-D reference (an ink-splat zarr) over a box:
   at the voxels known in both, AUC (tie-aware, 255-bin ranks), the best F1 over thresholds, F1 / precision / recall at 127 (p 0.5). */
int cmd_ink_vscore(int argc, char **argv) {
    const int gi = sidx(argc, argv, "--gt"), pi = sidx(argc, argv, "--pred");
    const char *bx = sopt(argc, argv, "--box", nullptr), *row = sopt(argc, argv, "--row", nullptr), *cache = sopt(argc, argv, "--cache", nullptr);
    const int L = atoi(sopt(argc, argv, "--level", "1")), nth = ufsm_threads(atoi(sopt(argc, argv, "--threads", "0")), 8);
    long long b[6];
    if (gi < 0 || pi < 0 || gi + 2 >= argc || pi + 2 >= argc || !bx || sscanf(bx, "%lld,%lld,%lld,%lld,%lld,%lld", b, b + 1, b + 2, b + 3, b + 4, b + 5) != 6 || L < 0 || L > 6) {
        fprintf(stderr, "usage: ufsm ink-vscore --gt ROOT KEY --pred ROOT KEY --box z,y,x,nz,ny,nx (level-0) [--level 1] [--row TAG] [--threads 8] [--cache DIR]\n"
                        "  voxelwise AUC / best F1 / F1 at 0.5 of a prob254 prediction vs a prob254 reference at the voxels known in both (the box on level L)\n"); return 2;
    }
    store *gs = store_open(argv[gi + 1]), *ps = store_open(argv[pi + 1]);
    z3 *g = gs ? z3_open(gs, argv[gi + 2], cache) : nullptr, *p = ps ? z3_open(ps, argv[pi + 2], cache) : nullptr;
    if (!g || !p) { fprintf(stderr, "ink-vscore: cannot open %s: %s\n", !g ? "--gt" : "--pred", z3_error()); return 1; }
    int64_t o[3], n[3]; size_t nv = 1;
    for (int d = 0; d < 3; d++) {
        const int64_t s = z3_meta_of(g)->shape[d] < z3_meta_of(p)->shape[d] ? z3_meta_of(g)->shape[d] : z3_meta_of(p)->shape[d];
        o[d] = b[d] >> L; int64_t e = (b[d] + b[d + 3] + (1 << L) - 1) >> L; if (e > s) e = s; if (o[d] < 0) o[d] = 0;
        n[d] = e - o[d]; if (n[d] <= 0) { fprintf(stderr, "ink-vscore: box outside the arrays\n"); return 1; } nv *= (size_t)n[d];
    }
    uint8_t *A = malloc(nv), *B = malloc(nv);
    if (!A || !B || z3_read(g, o, n, A, nth) || z3_read(p, o, n, B, nth)) { fprintf(stderr, "ink-vscore: read failed: %s\n", z3_error()); return 1; }
    double h[2][255] = {{0}}; uint64_t ngt = 0;
    for (size_t k = 0; k < nv; k++) { if (A[k] == 255) continue; ngt++; if (B[k] == 255) continue; h[A[k] >= 127][B[k]]++; }
    double P = 0, N = 0; for (int v = 0; v < 255; v++) { P += h[1][v]; N += h[0][v]; }
    double auc = 0, below = 0; for (int v = 0; v < 255; v++) { auc += h[1][v] * (below + 0.5 * h[0][v]); below += h[0][v]; }
    auc = P > 0 && N > 0 ? auc / (P * N) : NAN;
    double best = 0; int bt = 0, f5i = 0; double f05 = 0, p05 = 0, r05 = 0, tp = P, fp = N;   /* predicted ink = value >= t */
    for (int t = 0; t < 255; t++) {
        const double f1 = tp + fp + P > 0 ? 2 * tp / (tp + fp + P) : 0;
        if (f1 > best) { best = f1; bt = t; }
        if (t == 127) { f05 = f1; p05 = tp + fp > 0 ? tp / (tp + fp) : 0; r05 = P > 0 ? tp / P : 0; f5i = 1; }
        tp -= h[1][t]; fp -= h[0][t];
    }
    (void)f5i;
    printf("%s\t%.0f\t%.0f\t%.4f\t%.4f\t%.4f\t%d\t%.4f\t%.4f\t%.4f\t%.4f\n", row ? row : "-", P, N, P + N > 0 ? (P + N) / ngt : 0, P / fmax(1, P + N), auc, bt, best, f05, p05, r05);
    fprintf(stderr, "ink-vscore: %s: box %lld,%lld,%lld +%lld,%lld,%lld on level %d: %.0f ink + %.0f no-ink known in both (%.1f%% of the %llu reference voxels): AUC %.4f, best F1 %.4f at >= %d, "
                    "F1 / P / R at 0.5 %.4f / %.4f / %.4f\n", row ? row : "-", (long long)o[0], (long long)o[1], (long long)o[2], (long long)n[0], (long long)n[1], (long long)n[2], L, P, N,
            ngt ? 100.0 * (P + N) / ngt : 0, (unsigned long long)ngt, auc, best, bt, f05, p05, r05);
    free(A); free(B); z3_close(g); z3_close(p); store_close(gs); store_close(ps);
    return 0;
}

/* ---------------- on the fly: the ink-splat --face target as a derived z3 array ---------------- */
/* One unit = one segment's label window (the --crop of a stored run): the same mesh, labels, mask, window, face search and splat
   as inksplat_run, evaluated per 128^3 chunk of the CT level's grid on first read and kept in a bounded LRU. A chunk's value is the
   max over units of the known values (the stored runs' --merge), so the array is byte-identical to the merged splat of the same
   units. Reads may come from any thread. */
typedef struct { char tifxyz[1024], labels[1024], mask[1024]; int crop[4]; } inkfly_seg;
typedef struct { int surf, lab, msk; int64_t lr0, lc0, lw, lh; int ntx, nty; double F; int lab01, msk01; } ifunit;
typedef struct { int32_t lo[3], hi[3]; uint32_t unit, tile; } iftile;
typedef struct {
    inksplat_cfg c; int L; double V, off, reach; int64_t shape[3]; z3 *base_ct; zcache *zc;   /* zc: decoded CT chunks shared by neighbouring chunks' boxes */
    ssurf *S; int nS; char (*Spath)[1024];
    lsrc *Ls; int nL; char (*Lpath)[1024];
    ifunit *U; int nU;
    iftile *T; size_t nT;
    int64_t gcell, gdim[3]; uint32_t *gstart, *glist;   /* tile grid: cells of gcell level-0 voxels -> tile ids (CSR) */
} inkfly;
static struct { _Atomic uint64_t reads, chunks, empty, hits, px, us, ct_us, gpu_chunks, gpu_us; } g_ifs;
static _Atomic int g_ifgpu = -1;   /* -1 undecided (first chunk: $UFSM_INKFLY_CPU / $UFSM_INKFLY_GPU), 0 CPU, 1 GPU */
uint64_t inkfly_gpu_chunks(void) { return atomic_load(&g_ifs.gpu_chunks); }
void inkfly_gpu_config(int dev) { if (dev < 0 || !inkfly_gpu_near) { g_ifgpu = 0; return; } inkfly_gpu_device(dev); g_ifgpu = 1; }
static int ifgpu_on(void) {
    int g = atomic_load(&g_ifgpu); if (g >= 0) return g;
    const char *e = getenv("UFSM_INKFLY_CPU"), *d = getenv("UFSM_INKFLY_GPU");
    if ((e && atoi(e)) || !inkfly_gpu_near) g = 0; else { inkfly_gpu_device(d ? atoi(d) : 0); g = 1; }
    atomic_store(&g_ifgpu, g); return g;
}
static int inkfly_find(char (*paths)[1024], int n, const char *p) { for (int i = 0; i < n; i++) if (!strcmp(paths[i], p)) return i; return -1; }
static void inkfly_free(void *p) {
    inkfly *f = p; if (!f) return;
    if (f->zc) zcache_close(f->zc);
    for (int i = 0; i < f->nS; i++) ssurf_free(&f->S[i]);
    for (int i = 0; i < f->nL; i++) lsrc_close(&f->Ls[i]);
    free(f->S); free(f->Spath); free(f->Ls); free(f->Lpath); free(f->U); free(f->T); free(f->gstart); free(f->glist); free(f);
}
static int box_hit(const int32_t lo[3], const int32_t hi[3], const int64_t a[3], const int64_t b[3]) {   /* [lo, hi] vs [a, b) */
    for (int d = 0; d < 3; d++) if (hi[d] < a[d] || lo[d] >= b[d]) return 0;
    return 1;
}
/* the GPU chunk: 0 done, 1 run the CPU code (a mesh patch fell short), -1 a read failed. A CUDA failure turns the GPU off (CPU from then on). */
static int inkfly_chunk_gpu(inkfly *f, const int64_t o[3], const int64_t n[3], const uint32_t *cand, size_t nc, uint8_t *out, int *nonempty) {
    const inksplat_cfg *c = &f->c; const double V = f->V, slack = f->reach + 3;
    int64_t a0[3], b0[3]; for (int d = 0; d < 3; d++) { a0[d] = o[d] * (int64_t)V; b0[d] = (o[d] + n[d]) * (int64_t)V; }
    size_t nu = 0; for (size_t q = 0; q < nc; q++) nu += !q || f->T[cand[q]].unit != f->T[cand[q - 1]].unit;
    ifg_unit *U = calloc(nu, sizeof *U); int rc = -1; size_t iu = 0;
    uint8_t *ctb = nullptr;
    for (size_t q0 = 0; q0 < nc; iu++) {
        size_t q1 = q0; const ifunit *u = &f->U[f->T[cand[q0]].unit]; while (q1 < nc && f->T[cand[q1]].unit == f->T[cand[q0]].unit) q1++;
        ifg_unit *h = &U[iu]; int64_t rr0 = INT64_MAX, rc0 = INT64_MAX, rr1 = 0, rc1 = 0;
        int32_t *tl = malloc((q1 - q0) * sizeof *tl); h->tiles = tl; h->ntiles = (int)(q1 - q0); h->ntx = u->ntx;
        for (size_t q = q0; q < q1; q++) { const iftile *T = &f->T[cand[q]]; const int t = (int)T->tile, tx = t % u->ntx, ty = t / u->ntx; tl[q - q0] = t;
            const int64_t r0 = (int64_t)ty * TS, c0 = (int64_t)tx * TS, r1 = r0 + TS < u->lh ? r0 + TS : u->lh, c1 = c0 + TS < u->lw ? c0 + TS : u->lw;
            if (r0 < rr0) { rr0 = r0; } if (c0 < rc0) { rc0 = c0; } if (r1 > rr1) { rr1 = r1; } if (c1 > rc1) { rc1 = c1; }
            }
        h->rr0 = rr0; h->rc0 = rc0; h->rh = rr1 - rr0; h->rw = rc1 - rc0; h->lh = u->lh; h->lw = u->lw;
        uint8_t *lab = malloc((size_t)h->rh * h->rw), *msk = u->msk >= 0 ? malloc((size_t)h->rh * h->rw) : nullptr; h->lab = lab; h->msk = msk;
        if (!lab || (u->msk >= 0 && !msk) || lsrc_read(&f->Ls[u->lab], u->lr0 + rr0, u->lc0 + rc0, h->rh, h->rw, lab, 1) ||
            (msk && lsrc_read(&f->Ls[u->msk], u->lr0 + rr0, u->lc0 + rc0, h->rh, h->rw, msk, 1))) goto done;
        if (u->lab01) for (size_t k = 0; k < (size_t)h->rh * h->rw; k++) lab[k] *= 255;
        if (msk && u->msk01) for (size_t k = 0; k < (size_t)h->rh * h->rw; k++) msk[k] *= 255;
        const ssurf *S = &f->S[u->surf];
        geo G = {0}; G.c = c; G.S = S; G.F = u->F; G.lw = h->rw; G.lh = h->rh; G.lr0 = u->lr0 + rr0; G.lc0 = u->lc0 + rc0; G.offr = c->label_off[0]; G.offc = c->label_off[1];
        double gu0, gv0, gu1, gv1; px_grid(&G, -1.0, -1.0, &gu0, &gv0); px_grid(&G, (double)h->rh, (double)h->rw, &gu1, &gv1);   /* the rectangle +- 1 px (sub-samples reach 0.5) */
        int pu0 = (int)floor(gu0) - 1, pv0 = (int)floor(gv0) - 1, pu1 = (int)ceil(gu1) + 2, pv1 = (int)ceil(gv1) + 2;
        if (pu0 < 0) pu0 = 0; if (pv0 < 0) pv0 = 0; if (pu1 > S->w - 1) pu1 = S->w - 1; if (pv1 > S->h - 1) pv1 = S->h - 1;
        if (pu1 < pu0) pu1 = pu0; if (pv1 < pv0) pv1 = pv0;
        h->pu0 = pu0; h->pv0 = pv0; h->pw = pu1 - pu0 + 1; h->ph = pv1 - pv0 + 1; h->mw = S->w; h->mh = S->h; h->scale = S->scale;
        float *X = malloc((size_t)h->pw * h->ph * 4), *Y = malloc((size_t)h->pw * h->ph * 4), *Z = malloc((size_t)h->pw * h->ph * 4); h->X = X; h->Y = Y; h->Z = Z;
        for (int v = 0; v < h->ph; v++) { const size_t so = (size_t)(pv0 + v) * S->w + pu0, dof = (size_t)v * h->pw;
            memcpy(X + dof, S->X + so, (size_t)h->pw * 4); memcpy(Y + dof, S->Y + so, (size_t)h->pw * 4); memcpy(Z + dof, S->Z + so, (size_t)h->pw * 4); }
        h->F = u->F; h->glr0 = G.lr0; h->glc0 = G.lc0;
        double hpx = 0.45 * V / u->F; if (hpx > 1) hpx = 1; h->nsub = (int)ceil(1 / hpx);
        const double span = c->face_b - c->face_a; h->nt = (int)ceil(span / (0.45 * V)) + 1; h->tstep = h->nt > 1 ? span / (h->nt - 1) : 0;
        q0 = q1;
    }
    {
        ifg_cfg C = {0}; C.offr = c->label_off[0]; C.offc = c->label_off[1]; C.sign = c->flip ? -1 : 1; C.face_a = c->face_a; C.thr = c->thr; C.V = V; C.off = f->off; C.slack = slack;
        C.lo = (int)floor(c->face_lo); C.ns = (int)ceil(c->face_hi) - C.lo + 1; C.iso = c->iso; C.run = c->face_run; C.gap = c->face_gap;
        for (int d = 0; d < 3; d++) { C.a0[d] = a0[d]; C.b0[d] = b0[d]; C.shape[d] = f->shape[d]; }
        C.nexcl = c->nexcl; C.excl = (const int64_t (*)[6])c->excl;
        uint64_t npx = 0; int64_t lo[3], hi[3]; double tg = snow();
        int g = inkfly_gpu_near(U, (int)nu, &C, lo, hi, &npx);
        atomic_fetch_add(&g_ifs.gpu_us, (uint64_t)((snow() - tg) * 1e6));
        if (g) { if (g < 0) { fprintf(stderr, "ink onfly: the GPU near pass failed (%d): the CPU from here\n", g); atomic_store(&g_ifgpu, 0); } rc = 1; goto done; }
        if (!npx) { memset(out, 255, (size_t)n[0] * n[1] * n[2]); *nonempty = 0; rc = 0; goto done; }   /* (no CT read: nothing reaches the chunk) */
        /* the CT box: the near pixels' samples (the CPU's box, rounded out) */
        int64_t co[3], cn[3]; size_t ncv = 1;
        for (int d = 0; d < 3; d++) { co[d] = (int64_t)floor((lo[d] - f->off) / V) - 1; cn[d] = (int64_t)ceil((hi[d] - f->off) / V) + 2 - co[d]; ncv *= (size_t)cn[d]; }
        ctb = malloc(ncv); if (!ctb) goto done;
        const double tc = snow();
        if (f->zc ? zcache_read(f->zc, f->base_ct, co, cn, ctb) : z3_read(f->base_ct, co, cn, ctb, 1)) { fprintf(stderr, "ink-splat onfly: CT read: %s\n", z3_error()); goto done; }
        atomic_fetch_add(&g_ifs.ct_us, (uint64_t)((snow() - tc) * 1e6));
        tg = snow(); g = inkfly_gpu_finish(&C, ctb, co, cn, o, n, out, nonempty);
        atomic_fetch_add(&g_ifs.gpu_us, (uint64_t)((snow() - tg) * 1e6));
        if (g < 0) { fprintf(stderr, "ink onfly: the GPU chunk failed (%d): the CPU from here\n", g); atomic_store(&g_ifgpu, 0); rc = 1; }
        else { rc = g; if (!g) { atomic_fetch_add(&g_ifs.px, npx); atomic_fetch_add(&g_ifs.gpu_chunks, 1); } }
    }
done:
    for (size_t i = 0; i < nu; i++) { free((void *)U[i].tiles); free((void *)U[i].lab); free((void *)U[i].msk); free((void *)U[i].X); free((void *)U[i].Y); free((void *)U[i].Z); }
    free(U); free(ctb);
    return rc;
}
/* one chunk [o, o + n) of the level-L grid: candidates -> CT box -> per unit: face + splat into accumulators -> finalize (max merge) */
static int inkfly_chunk(inkfly *f, const int64_t o[3], const int64_t n[3], uint8_t *out, int *nonempty) {
    const inksplat_cfg *c = &f->c; const double V = f->V; const size_t cv = (size_t)n[0] * n[1] * n[2];
    memset(out, 255, cv); *nonempty = 0;
    int64_t a0[3], b0[3]; for (int d = 0; d < 3; d++) { a0[d] = o[d] * (int64_t)V; b0[d] = (o[d] + n[d]) * (int64_t)V; }
    /* candidate tiles (stamped by a per-call bitmap over the touched tile ids) */
    size_t cap = 256, nc = 0; uint32_t *cand = malloc(cap * sizeof *cand);
    int64_t g0[3], g1[3];
    for (int d = 0; d < 3; d++) { g0[d] = a0[d] / f->gcell; g1[d] = (b0[d] - 1) / f->gcell; if (g0[d] < 0) g0[d] = 0; if (g1[d] >= f->gdim[d]) g1[d] = f->gdim[d] - 1; }
    for (int64_t z = g0[0]; z <= g1[0]; z++) for (int64_t y = g0[1]; y <= g1[1]; y++) for (int64_t x = g0[2]; x <= g1[2]; x++) {
        const size_t gi = ((size_t)z * f->gdim[1] + y) * f->gdim[2] + x;
        for (uint32_t k = f->gstart[gi]; k < f->gstart[gi + 1]; k++) {
            const uint32_t t = f->glist[k]; if (!box_hit(f->T[t].lo, f->T[t].hi, a0, b0)) continue;
            int dup = 0; for (size_t q = nc; q-- > 0 && !dup;) dup = cand[q] == t;   /* (a tile is listed in each cell it overlaps) */
            if (dup) continue;
            if (nc == cap) { cap *= 2; cand = realloc(cand, cap * sizeof *cand); }
            cand[nc++] = t;
        }
    }
    if (!nc) { free(cand); return 0; }
    /* sort candidates by unit */
    for (size_t i = 1; i < nc; i++) { uint32_t t = cand[i]; size_t j = i; while (j > 0 && (f->T[cand[j - 1]].unit > f->T[t].unit || (f->T[cand[j - 1]].unit == f->T[t].unit && f->T[cand[j - 1]].tile > f->T[t].tile))) { cand[j] = cand[j - 1]; j--; } cand[j] = t; }
    if (ifgpu_on()) {   /* the GPU: label rectangles, mesh patches and the CT box from here; face, splat, finalize on the device (inkflygpu.h) */
        const int r = inkfly_chunk_gpu(f, o, n, cand, nc, out, nonempty);
        if (r <= 0) { free(cand); return r; }
        memset(out, 255, cv); *nonempty = 0;   /* (1: a mesh patch fell short: the CPU below) */
    }
    /* pass 1 per unit: the label rectangle of its candidate tiles, and the supervised pixels whose surface point is near the chunk */
    typedef struct { const ifunit *u; uint8_t *lab, *msk; int64_t rr0, rc0, rw, rh; size_t np; int64_t *pk; double *PN; } ugrp;
    size_t ng = 0, gcap = 8; ugrp *UG = calloc(gcap, sizeof *UG);
    double lo3[3] = {1e300, 1e300, 1e300}, hi3[3] = {-1e300, -1e300, -1e300}; size_t nnear = 0;
    uint8_t *ctb = nullptr, *ni = nullptr, *nn = nullptr, *pi = nullptr, *pn = nullptr; int rc = -1;
    const double slack = f->reach + 3;
    for (size_t q0 = 0; q0 < nc;) {
        size_t q1 = q0; const ifunit *u = &f->U[f->T[cand[q0]].unit]; while (q1 < nc && f->T[cand[q1]].unit == f->T[cand[q0]].unit) q1++;
        int64_t rr0 = INT64_MAX, rc0 = INT64_MAX, rr1 = 0, rc1 = 0;
        for (size_t q = q0; q < q1; q++) { const int t = (int)f->T[cand[q]].tile, tx = t % u->ntx, ty = t / u->ntx;
            const int64_t r0 = (int64_t)ty * TS, c0 = (int64_t)tx * TS, r1 = r0 + TS < u->lh ? r0 + TS : u->lh, c1 = c0 + TS < u->lw ? c0 + TS : u->lw;
            if (r0 < rr0) { rr0 = r0; } if (c0 < rc0) { rc0 = c0; } if (r1 > rr1) { rr1 = r1; } if (c1 > rc1) { rc1 = c1; } }
        if (ng == gcap) { gcap *= 2; UG = realloc(UG, gcap * sizeof *UG); }
        ugrp *g = &UG[ng++]; memset(g, 0, sizeof *g); g->u = u; g->rr0 = rr0; g->rc0 = rc0; g->rh = rr1 - rr0; g->rw = rc1 - rc0;
        g->lab = malloc((size_t)g->rh * g->rw); g->msk = u->msk >= 0 ? malloc((size_t)g->rh * g->rw) : nullptr;
        if (!g->lab || (u->msk >= 0 && !g->msk) || lsrc_read(&f->Ls[u->lab], u->lr0 + rr0, u->lc0 + rc0, g->rh, g->rw, g->lab, 1) ||
            (g->msk && lsrc_read(&f->Ls[u->msk], u->lr0 + rr0, u->lc0 + rc0, g->rh, g->rw, g->msk, 1))) goto done;
        if (u->lab01) for (size_t k = 0; k < (size_t)g->rh * g->rw; k++) g->lab[k] *= 255;
        if (g->msk && u->msk01) for (size_t k = 0; k < (size_t)g->rh * g->rw; k++) g->msk[k] *= 255;
        geo G = {0}; G.c = c; G.S = &f->S[u->surf]; G.F = u->F; G.lab = g->lab; G.msk = g->msk; G.lw = g->rw; G.lh = g->rh;
        G.lr0 = u->lr0 + rr0; G.lc0 = u->lc0 + rc0; G.offr = c->label_off[0]; G.offc = c->label_off[1];
        size_t pcap = 1024; g->pk = malloc(pcap * sizeof *g->pk); g->PN = malloc(pcap * 6 * sizeof *g->PN);
        for (size_t q = q0; q < q1; q++) {
            const int t = (int)f->T[cand[q]].tile, tx = t % u->ntx, ty = t / u->ntx;
            const int64_t r0 = (int64_t)ty * TS, c0 = (int64_t)tx * TS, r1 = r0 + TS < u->lh ? r0 + TS : u->lh, c1 = c0 + TS < u->lw ? c0 + TS : u->lw;
            for (int64_t r = r0 - rr0; r < r1 - rr0; r++) for (int64_t cc = c0 - rc0; cc < c1 - rc0; cc++) {
                if (!px_sup(&G, r, cc)) continue;
                double gu, gv, P[3], N[3]; px_grid(&G, (double)r, (double)cc, &gu, &gv);
                if (ssurf_pn(G.S, gu, gv, P, N)) continue;
                int far = 0; for (int d = 0; d < 3 && !far; d++) far = P[d] < a0[d] - slack || P[d] > b0[d] + slack;   /* samples cannot reach the chunk */
                if (far) continue;
                if (g->np == pcap) { pcap *= 2; g->pk = realloc(g->pk, pcap * sizeof *g->pk); g->PN = realloc(g->PN, pcap * 6 * sizeof *g->PN); }
                g->pk[g->np] = r * g->rw + cc; memcpy(g->PN + 6 * g->np, P, sizeof P); memcpy(g->PN + 6 * g->np + 3, N, sizeof N); g->np++;
                for (int d = 0; d < 3; d++) { if (P[d] - slack < lo3[d]) lo3[d] = P[d] - slack; if (P[d] + slack > hi3[d]) hi3[d] = P[d] + slack; }
            }
        }
        nnear += g->np; q0 = q1;
    }
    if (!nnear) { rc = 0; goto done; }
    /* the CT box: the near pixels' surface points +- every sample's reach (the values do not depend on the box) */
    int64_t co[3], cn[3]; size_t ncv = 1;
    for (int d = 0; d < 3; d++) { co[d] = (int64_t)floor((lo3[d] - f->off) / V) - 1; cn[d] = (int64_t)ceil((hi3[d] - f->off) / V) + 2 - co[d]; ncv *= (size_t)cn[d]; }
    ctb = malloc(ncv); ni = calloc(cv, 1); nn = calloc(cv, 1); pi = calloc(cv, 1); pn = calloc(cv, 1);
    if (!ctb || !ni || !nn || !pi || !pn) goto done;
    { const double tc = snow();
      if (f->zc ? zcache_read(f->zc, f->base_ct, co, cn, ctb) : z3_read(f->base_ct, co, cn, ctb, 1)) { fprintf(stderr, "ink-splat onfly: CT read: %s\n", z3_error()); goto done; }
      atomic_fetch_add(&g_ifs.ct_us, (uint64_t)((snow() - tc) * 1e6)); }
    const int lo = (int)floor(c->face_lo), hi = (int)ceil(c->face_hi), ns = hi - lo + 1;
    float raw[256], sm[256];
    const double hpx0 = 0.45 * V;
    for (size_t gi = 0; gi < ng; gi++) {
        ugrp *g = &UG[gi]; const ifunit *u = g->u; if (!g->np) continue;
        geo G = {0}; G.c = c; G.S = &f->S[u->surf]; G.F = u->F; G.sign = c->flip ? -1 : 1; G.lab = g->lab; G.msk = g->msk; G.lw = g->rw; G.lh = g->rh;
        G.lr0 = u->lr0 + g->rr0; G.lc0 = u->lc0 + g->rc0; G.offr = c->label_off[0]; G.offc = c->label_off[1];
        double hpx = hpx0 / u->F; if (hpx > 1) hpx = 1; const int nsub = (int)ceil(1 / hpx);
        const double span = c->face_b - c->face_a; const int nt = (int)ceil(span / (0.45 * V)) + 1; const double tstep = nt > 1 ? span / (nt - 1) : 0;
        atomic_fetch_add(&g_ifs.px, g->np);
        for (size_t ip = 0; ip < g->np; ip++) {
            const int64_t pk = g->pk[ip], r = pk / g->rw, cc = pk % g->rw; const double *P = g->PN + 6 * ip, *N = P + 3;
            for (int i = 0; i < ns; i++) {
                const double tt = (lo + i) * G.sign; double qq[3];
                for (int d = 0; d < 3; d++) qq[d] = (P[d] + tt * N[d] - f->off) / V - co[d];
                const float v = tri_box(ctb, cn, qq[0], qq[1], qq[2]); raw[i] = v < 0 ? 0 : v;
            }
            for (int i = 0; i < ns; i++) sm[i] = 0.25f * (raw[i > 0 ? i - 1 : 0] + 2 * raw[i] + raw[i < ns - 1 ? i + 1 : ns - 1]);
            const int fo = face_of(sm, ns, lo, c->iso, c->face_run, c->face_gap);
            if (fo == FACE_NONE) continue;
            const double ta = fo + c->face_a; const uint8_t lv = g->lab[pk]; const int ink = lv >= 128;
            for (int a = 0; a < nsub; a++) for (int b = 0; b < nsub; b++) {
                double gu, gv, P2[3], N2[3];
                px_grid(&G, r + (a + 0.5) / nsub - 0.5, cc + (b + 0.5) / nsub - 0.5, &gu, &gv);
                if (ssurf_pn(G.S, gu, gv, P2, N2)) continue;
                for (int k = 0; k < nt; k++) {
                    const double tt = (ta + k * tstep) * G.sign; int64_t vi[3]; int in = 1;
                    for (int d = 0; d < 3; d++) { vi[d] = (int64_t)floor((P2[d] + tt * N2[d] - f->off) / V + 0.5) - o[d]; if (vi[d] < 0 || vi[d] >= n[d]) { in = 0; break; } }
                    if (!in) continue;
                    const size_t k2 = ((size_t)vi[0] * n[1] + vi[1]) * n[2] + vi[2];
                    if (ink) { if (ni[k2] < 255) ni[k2]++; if (lv > pi[k2]) pi[k2] = lv; }
                    else { if (nn[k2] < 255) nn[k2]++; if (lv > pn[k2]) pn[k2] = lv; }
                }
            }
        }
        /* finalize this unit (as splat_worker) and merge by known max */
        for (size_t k = 0; k < cv; k++) {
            const int h = ni[k] + nn[k]; if (!h) continue;
            const int64_t z = o[0] + (int64_t)(k / ((size_t)n[1] * n[2])), y = o[1] + (int64_t)(k / n[2] % n[1]), x = o[2] + (int64_t)(k % n[2]);
            if (z < f->shape[0] && y < f->shape[1] && x < f->shape[2]) {
                int ex = 0;
                for (int e = 0; e < c->nexcl && !ex; e++) { const int64_t *bx = c->excl[e], Vi = (int64_t)V;
                    ex = (z + 1) * Vi > bx[0] && z * Vi < bx[0] + bx[3] && (y + 1) * Vi > bx[1] && y * Vi < bx[1] + bx[4] && (x + 1) * Vi > bx[2] && x * Vi < bx[2] + bx[5]; }
                const int64_t cz = z - co[0], cy = y - co[1], cx = x - co[2];
                const uint8_t ct = cz >= 0 && cy >= 0 && cx >= 0 && cz < cn[0] && cy < cn[1] && cx < cn[2] ? ctb[((size_t)cz * cn[1] + cy) * cn[2] + cx] : 0;
                if (!ex && ct && ct >= c->iso) {
                    const int isink = ni[k] && ni[k] >= c->thr * h;
                    const uint8_t v = (uint8_t)(((isink ? pi[k] : pn[k]) * 254 + 127) / 255);
                    out[k] = out[k] == 255 ? v : (out[k] > v ? out[k] : v); *nonempty = 1;
                }
            }
            ni[k] = nn[k] = pi[k] = pn[k] = 0;
        }
    }
    rc = 0;
done:
    for (size_t gi = 0; gi < ng; gi++) { free(UG[gi].lab); free(UG[gi].msk); free(UG[gi].pk); free(UG[gi].PN); }
    free(UG); free(cand); free(ctb); free(ni); free(nn); free(pi); free(pn);
    return rc;
}
enum { IFC = 128 };   /* computed chunk edge (level-L voxels): the dcache cell (dcache.h) */
static int inkfly_derive(void *ctx, z3 *base, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads) {
    (void)base; (void)nthreads; inkfly *f = ctx; const double t0 = snow();
    atomic_fetch_add(&g_ifs.reads, 1);
    int aligned = 1; for (int d = 0; d < 3; d++) aligned &= o[d] >= 0 && o[d] % IFC == 0 && n[d] == IFC;
    if (aligned) {   /* (the cache's cell reads) */
        int ne = 0; const int rc = inkfly_chunk(f, o, n, out, &ne);
        atomic_fetch_add(&g_ifs.chunks, 1); if (!ne) atomic_fetch_add(&g_ifs.empty, 1);
        atomic_fetch_add(&g_ifs.us, (uint64_t)((snow() - t0) * 1e6)); return rc;
    }
    memset(out, 255, (size_t)n[0] * n[1] * n[2]);
    const int64_t gc[3] = {(f->shape[0] + IFC - 1) / IFC, (f->shape[1] + IFC - 1) / IFC, (f->shape[2] + IFC - 1) / IFC};
    int64_t c0[3], c1[3];
    for (int d = 0; d < 3; d++) { c0[d] = o[d] < 0 ? 0 : o[d] / IFC; c1[d] = (o[d] + n[d] - 1) / IFC; if (c1[d] >= gc[d]) c1[d] = gc[d] - 1; if (o[d] + n[d] <= 0 || o[d] >= f->shape[d]) return 0; }
    uint8_t *tmp = malloc((size_t)IFC * IFC * IFC); if (!tmp) return -1;
    for (int64_t cz = c0[0]; cz <= c1[0]; cz++) for (int64_t cy = c0[1]; cy <= c1[1]; cy++) for (int64_t cx = c0[2]; cx <= c1[2]; cx++) {
        const int64_t co[3] = {cz * IFC, cy * IFC, cx * IFC}, cn[3] = {IFC, IFC, IFC}; int ne = 0;
        if (inkfly_chunk(f, co, cn, tmp, &ne)) { free(tmp); return -1; }
        atomic_fetch_add(&g_ifs.chunks, 1); if (!ne) { atomic_fetch_add(&g_ifs.empty, 1); continue; }
        for (int64_t z = 0; z < IFC; z++) { const int64_t gz = cz * IFC + z - o[0]; if (gz < 0 || gz >= n[0]) continue;
            for (int64_t y = 0; y < IFC; y++) { const int64_t gy = cy * IFC + y - o[1]; if (gy < 0 || gy >= n[1]) continue;
                const int64_t x0 = cx * IFC - o[2]; int64_t xa = x0 < 0 ? -x0 : 0, xb = x0 + IFC > n[2] ? n[2] - x0 : IFC;
                if (xb > xa) memcpy(out + ((size_t)gz * n[1] + gy) * n[2] + x0 + xa, tmp + ((size_t)z * IFC + y) * IFC + xa, (size_t)(xb - xa)); } }
    }
    free(tmp);
    atomic_fetch_add(&g_ifs.us, (uint64_t)((snow() - t0) * 1e6));
    return 0;
}
/* the coarser level: known-max 2x pooling of the level below (an onfly array), as the stored pyramid */
typedef struct { z3 *fine; int own; } ifpool;
static int inkfly_pool_derive(void *ctx, z3 *base, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads) {
    (void)base; ifpool *p = ctx; const int64_t fo[3] = {2 * o[0], 2 * o[1], 2 * o[2]}, fn[3] = {2 * n[0], 2 * n[1], 2 * n[2]};
    uint8_t *b = malloc((size_t)fn[0] * fn[1] * fn[2]); if (!b) return -1;
    if (z3_read(p->fine, fo, fn, b, nthreads)) { free(b); return -1; }
    const size_t pl = (size_t)fn[1] * fn[2];
    for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) for (int64_t x = 0; x < n[2]; x++) {
        const uint8_t *q = b + ((size_t)(2 * z) * fn[1] + 2 * y) * fn[2] + 2 * x;
        const uint8_t v[8] = {q[0], q[1], q[fn[2]], q[fn[2] + 1], q[pl], q[pl + 1], q[pl + fn[2]], q[pl + fn[2] + 1]};
        int m = -1; for (int i = 0; i < 8; i++) if (v[i] != 255 && v[i] > m) m = v[i];
        out[((size_t)z * n[1] + y) * n[2] + x] = m < 0 ? 255 : (uint8_t)m;
    }
    free(b); return 0;
}
static void inkfly_pool_free(void *p) { ifpool *q = p; if (q && q->own) z3_close(q->fine); free(q); }
z3 *inkfly_open_pooled(z3 *ct_coarse, z3 *fine, int own_fine) {
    ifpool *p = calloc(1, sizeof *p); p->fine = fine; p->own = own_fine;
    z3 *z = z3_open_derived(ct_coarse, 255, inkfly_pool_derive, p, inkfly_pool_free);
    if (!z) { free(p); if (own_fine) z3_close(fine); z3_close(ct_coarse); }
    return z;
}

/* the dcache key: everything a chunk's bytes depend on. Bump INKFLY_CODE when the arithmetic changes. Local inputs add their size and
   mtime (a tifxyz dir: its x / y / z.tif and meta.json; a zarr array dir: its zarr.json and the dir), so edited labels miss the cache. */
#define INKFLY_CODE "inkfly 1 (ink-splat --face, 2026-11-11)"
static void key_stat(char **b, size_t *n, size_t *cap, const char *path, const char *sub) {
    char p[2100]; snprintf(p, sizeof p, "%s%s%s", path, sub ? "/" : "", sub ? sub : ""); struct stat st;
    char l[2400]; const int k = stat(p, &st) ? snprintf(l, sizeof l, " [%s: -]", sub ? sub : ".") :
        snprintf(l, sizeof l, " [%s: %lld %lld.%09ld]", sub ? sub : ".", (long long)st.st_size, (long long)st.st_mtim.tv_sec, st.st_mtim.tv_nsec);
    while (*n + (size_t)k + 1 > *cap) { *cap *= 2; *b = realloc(*b, *cap); }
    memcpy(*b + *n, l, (size_t)k + 1); *n += (size_t)k;
}
static void key_add(char **b, size_t *n, size_t *cap, const char *fmt, ...) {
    va_list ap; va_start(ap, fmt); char l[4096]; int k = vsnprintf(l, sizeof l, fmt, ap); va_end(ap); if (k >= (int)sizeof l) k = (int)sizeof l - 1;
    while (*n + (size_t)k + 1 > *cap) { *cap *= 2; *b = realloc(*b, *cap); }
    memcpy(*b + *n, l, (size_t)k + 1); *n += (size_t)k;
}
static char *inkfly_key(const inkfly *f, const inksplat_cfg *c, const char *segs_txt) {
    size_t n = 0, cap = 4096; char *b = malloc(cap); b[0] = 0;
    key_add(&b, &n, &cap, "%s\nct %s %s level %d shape %lld,%lld,%lld um %.6g\n", INKFLY_CODE, c->ct_root ? c->ct_root : "?", c->ct_key ? c->ct_key : "?", f->L,
            (long long)f->shape[0], (long long)f->shape[1], (long long)f->shape[2], c->ct_um);
    key_add(&b, &n, &cap, "cfg iso %d thr %.9g flip %d face %.9g,%.9g t %.9g,%.9g run %d gap %d label_scale %.9g off %.9g,%.9g\n", c->iso, c->thr, c->flip,
            c->face_a, c->face_b, c->face_lo, c->face_hi, c->face_run, c->face_gap, c->label_scale, c->label_off[0], c->label_off[1]);
    for (int e = 0; e < c->nexcl; e++) key_add(&b, &n, &cap, "exclude %lld,%lld,%lld,%lld,%lld,%lld\n", (long long)c->excl[e][0], (long long)c->excl[e][1],
                                               (long long)c->excl[e][2], (long long)c->excl[e][3], (long long)c->excl[e][4], (long long)c->excl[e][5]);
    for (int i = 0; i < f->nS; i++) { key_add(&b, &n, &cap, "mesh %s", f->Spath[i]);
        static const char *fs[] = {"x.tif", "y.tif", "z.tif", "meta.json"}; for (int k = 0; k < 4; k++) key_stat(&b, &n, &cap, f->Spath[i], fs[k]); key_add(&b, &n, &cap, "\n"); }
    for (int i = 0; i < f->nL; i++) { key_add(&b, &n, &cap, "raster %s", f->Lpath[i]);
        if (strncmp(f->Lpath[i], "http", 4)) { key_stat(&b, &n, &cap, f->Lpath[i], nullptr); key_stat(&b, &n, &cap, f->Lpath[i], "zarr.json"); } key_add(&b, &n, &cap, "\n"); }
    const size_t ls = strlen(segs_txt); while (n + ls + 16 > cap) { cap *= 2; b = realloc(b, cap); }
    memcpy(b + n, "segments ", 9); memcpy(b + n + 9, segs_txt, ls); n += 9 + ls; b[n++] = '\n'; b[n] = 0;
    return b;
}
typedef struct { inkfly *f; const inksplat_cfg *cfg; iftile **tl; size_t *ntl; _Atomic int next, failed; } ifidx;
static void *ifidx_worker(void *p) {
    ifidx *J = p; inkfly *f = J->f; const inksplat_cfg *cfg = J->cfg;
    for (int i; (i = atomic_fetch_add(&J->next, 1)) < f->nU && !atomic_load(&J->failed);) {
        ifunit *u = &f->U[i]; const ssurf *S = &f->S[u->surf];
        uint8_t *lab = malloc((size_t)u->lw * u->lh), *msk = u->msk >= 0 ? malloc((size_t)u->lw * u->lh) : nullptr;
        if (!lab || (u->msk >= 0 && !msk) || lsrc_read(&f->Ls[u->lab], u->lr0, u->lc0, u->lh, u->lw, lab, 1) || (msk && lsrc_read(&f->Ls[u->msk], u->lr0, u->lc0, u->lh, u->lw, msk, 1))) {
            free(lab); free(msk); atomic_store(&J->failed, 1); break; }
        { uint8_t m = 0; for (size_t k = 0; k < (size_t)u->lw * u->lh; k++) if (lab[k] > m) m = lab[k]; u->lab01 = m == 1; }
        if (msk) { uint8_t m = 0; for (size_t k = 0; k < (size_t)u->lw * u->lh; k++) if (msk[k] > m) m = msk[k]; u->msk01 = m == 1; if (u->msk01) for (size_t k = 0; k < (size_t)u->lw * u->lh; k++) msk[k] *= 255; }
        free(lab);
        geo G = {0}; G.c = cfg; G.S = S; G.F = u->F; G.msk = msk; G.lw = u->lw; G.lh = u->lh; G.lr0 = u->lr0; G.lc0 = u->lc0; G.offr = cfg->label_off[0]; G.offc = cfg->label_off[1];
        G.ntiles_x = u->ntx; G.ntiles_y = u->nty;
        size_t cap = 1024, n = 0; iftile *T = malloc(cap * sizeof *T);
        for (int t = 0; t < u->ntx * u->nty; t++) {
            double lo[3], hi[3]; if (!tile_box(&G, t % u->ntx, t / u->ntx, f->reach, lo, hi)) continue;
            if (n == cap) { cap *= 2; T = realloc(T, cap * sizeof *T); }
            iftile *q = &T[n++]; q->unit = (uint32_t)i; q->tile = (uint32_t)t;
            for (int d = 0; d < 3; d++) { q->lo[d] = (int32_t)floor(lo[d]); q->hi[d] = (int32_t)ceil(hi[d]); }
        }
        free(msk); J->tl[i] = T; J->ntl[i] = n;
    }
    return nullptr;
}
/* SEGS: {"segments": [{"tifxyz": DIR, "labels": L, "mask": M, "crop": [c0, r0, w, h]}, ..]}; ct: the CT at level cfg->level (owned) */
z3 *inkfly_open(z3 *ct, const char *segs_path, const inksplat_cfg *cfg, size_t cache_mb, const char *dcache_dir, void **handle) {
    const double t0 = snow();
    FILE *fp = fopen(segs_path, "rb"); if (!fp) { fprintf(stderr, "ink onfly: cannot open %s\n", segs_path); z3_close(ct); return nullptr; }
    fseek(fp, 0, SEEK_END); long sz = ftell(fp); fseek(fp, 0, SEEK_SET); char *txt = malloc((size_t)sz + 1); size_t rd = fread(txt, 1, (size_t)sz, fp); fclose(fp); txt[rd] = 0;
    json *j = json_parse(txt, rd); const json *sg = json_get(j, "segments"); char *sg_txt = txt;   /* (the segments text goes into the cache key) */
    if (!sg || sg->type != J_ARR || !sg->n) { fprintf(stderr, "ink onfly: %s: no \"segments\" array\n", segs_path); json_free(j); free(sg_txt); z3_close(ct); return nullptr; }
    inkfly *f = calloc(1, sizeof *f); f->c = *cfg; f->L = cfg->level; f->V = (double)(1 << f->L); f->off = (f->V - 1) / 2; f->base_ct = ct;
    memcpy(f->shape, z3_meta_of(ct)->shape, sizeof f->shape);
    f->reach = fmax(fmax(fabs(cfg->face_lo), fabs(cfg->face_hi)) + 1, fmax(fabs(cfg->face_lo + cfg->face_a), fabs(cfg->face_hi + cfg->face_b)));
    const int nseg = (int)sg->n; f->S = calloc(nseg, sizeof *f->S); f->Spath = calloc(nseg, sizeof *f->Spath); f->Ls = calloc(2 * nseg, sizeof *f->Ls); f->Lpath = calloc(2 * nseg, sizeof *f->Lpath); f->U = calloc(nseg, sizeof *f->U);
    f->T = nullptr;
    for (int i = 0; i < nseg; i++) {
        const json *e = json_at(sg, (size_t)i); const char *tx = json_str(json_get(e, "tifxyz"), nullptr), *lb = json_str(json_get(e, "labels"), nullptr), *mk = json_str(json_get(e, "mask"), nullptr);
        if (!tx || !lb) { fprintf(stderr, "ink onfly: segment %d: need tifxyz and labels\n", i); goto fail; }
        ifunit *u = &f->U[f->nU];
        if ((u->surf = inkfly_find(f->Spath, f->nS, tx)) < 0) { if (ssurf_load(&f->S[f->nS], tx)) goto fail; snprintf(f->Spath[f->nS], 1024, "%s", tx); u->surf = f->nS++; }
        if ((u->lab = inkfly_find(f->Lpath, f->nL, lb)) < 0) { if (lsrc_open(&f->Ls[f->nL], lb, -1, cfg->cache)) goto fail; snprintf(f->Lpath[f->nL], 1024, "%s", lb); u->lab = f->nL++; }
        u->msk = -1;
        if (mk && (u->msk = inkfly_find(f->Lpath, f->nL, mk)) < 0) { if (lsrc_open(&f->Ls[f->nL], mk, -1, cfg->cache)) goto fail; snprintf(f->Lpath[f->nL], 1024, "%s", mk); u->msk = f->nL++; }
        const ssurf *S = &f->S[u->surf]; const lsrc *LL = &f->Ls[u->lab];
        /* the label scale exactly as inksplat_run */
        double F = cfg->label_scale; const double ew = S->w / S->scale, eh = S->h / S->scale;
        if (!(F > 0)) { F = 0.5 * (ew / (double)LL->W + eh / (double)LL->H); const double p2 = pow(2, round(log2(F))); if (fabs(F - p2) < 0.002 * p2) F = p2;
            const double mum = S->vox_um > 0 ? S->vox_um : cfg->ct_um; if (LL->um > 0 && mum > 0) F = LL->um / mum; }
        u->F = F; u->lr0 = 0; u->lc0 = 0; u->lw = LL->W; u->lh = LL->H;
        const json *cr = json_get(e, "crop");
        if (cr && cr->n == 4 && json_num(json_at(cr, 2), 0) > 0) { u->lc0 = (int64_t)json_num(json_at(cr, 0), 0); u->lr0 = (int64_t)json_num(json_at(cr, 1), 0); u->lw = (int64_t)json_num(json_at(cr, 2), 0); u->lh = (int64_t)json_num(json_at(cr, 3), 0); }
        u->ntx = (int)((u->lw + TS - 1) / TS); u->nty = (int)((u->lh + TS - 1) / TS);
        f->nU++;
    }
    char *key = inkfly_key(f, cfg, sg_txt), kdir[1200] = "", ixp[1300] = ""; int have_ix = 0;
    if (dcache_dir && *dcache_dir && !dcache_key_dir(dcache_dir, key, kdir, sizeof kdir)) {   /* the index cached beside the chunks (same inputs key) */
        snprintf(ixp, sizeof ixp, "%s/index.bin", kdir); FILE *ix = fopen(ixp, "rb");
        if (ix) { char m[4]; uint32_t nu = 0; uint64_t nt = 0;
            if (fread(m, 1, 4, ix) == 4 && !memcmp(m, "UIX1", 4) && fread(&nu, 4, 1, ix) == 1 && fread(&nt, 8, 1, ix) == 1 && nu == (uint32_t)f->nU) {
                uint8_t *fl = malloc(2 * (size_t)nu + 1); f->T = malloc((nt ? nt : 1) * sizeof *f->T);
                if (fread(fl, 1, 2 * (size_t)nu, ix) == 2 * (size_t)nu && fread(f->T, sizeof *f->T, nt, ix) == nt) {
                    for (uint32_t i = 0; i < nu; i++) { f->U[i].lab01 = fl[2 * i]; f->U[i].msk01 = fl[2 * i + 1]; }
                    f->nT = nt; have_ix = 1; }
                else { free(f->T); f->T = nullptr; }
                free(fl); }
            fclose(ix); }
    }
    if (!have_ix) {
    {   /* per window, on worker threads: its rasters once (the 0/1 -> 0/255 rule over the whole window, inksplat_run's) and the supervised
           tiles' boxes; then the tiles concatenated in window order */
        ifidx J = {f, cfg, calloc(f->nU, sizeof *J.tl), calloc(f->nU, sizeof *J.ntl), 0, 0};
        pthread_t th[16]; const int nth = f->nU < 8 ? f->nU : 8;
        for (int i = 0; i < nth; i++) pthread_create(&th[i], nullptr, ifidx_worker, &J);
        for (int i = 0; i < nth; i++) pthread_join(th[i], nullptr);
        size_t tot = 0; for (int i = 0; i < f->nU; i++) tot += J.ntl[i];
        if (!atomic_load(&J.failed)) { free(f->T); f->T = malloc((tot ? tot : 1) * sizeof *f->T); for (int i = 0; i < f->nU; i++) { memcpy(f->T + f->nT, J.tl[i], J.ntl[i] * sizeof *f->T); f->nT += J.ntl[i]; } }
        for (int i = 0; i < f->nU; i++) free(J.tl[i]);
        free(J.tl); free(J.ntl);
        if (atomic_load(&J.failed)) { free(key); goto fail; }
    }
        if (*ixp) {   /* write it for the next process (atomic rename; a failure only costs the next open) */
            char tmp[1400]; snprintf(tmp, sizeof tmp, "%s.%d.tmp", ixp, (int)getpid()); FILE *ix = fopen(tmp, "wb");
            if (ix) { const uint32_t nu = (uint32_t)f->nU; const uint64_t nt = f->nT; uint8_t *fl = malloc(2 * (size_t)nu + 1);
                for (uint32_t i = 0; i < nu; i++) { fl[2 * i] = (uint8_t)f->U[i].lab01; fl[2 * i + 1] = (uint8_t)f->U[i].msk01; }
                const int ok = fwrite("UIX1", 1, 4, ix) == 4 && fwrite(&nu, 4, 1, ix) == 1 && fwrite(&nt, 8, 1, ix) == 1 && fwrite(fl, 1, 2 * (size_t)nu, ix) == 2 * (size_t)nu &&
                               fwrite(f->T, sizeof *f->T, nt, ix) == nt;
                free(fl); if (fclose(ix) || !ok || rename(tmp, ixp)) unlink(tmp); }
        }
    }
    json_free(j); j = nullptr;
    {   /* tile grid: 1024-voxel cells */
        f->gcell = 1024; size_t ng = 1;
        for (int d = 0; d < 3; d++) { f->gdim[d] = (f->shape[d] * (int64_t)f->V + f->gcell - 1) / f->gcell; ng *= (size_t)f->gdim[d]; }
        f->gstart = calloc(ng + 1, sizeof *f->gstart); size_t tot = 0;
        for (int pass = 0; pass < 2; pass++) {
            uint32_t *fill = pass ? calloc(ng, sizeof *fill) : nullptr;
            for (size_t t = 0; t < f->nT; t++) {
                int64_t a[3], b[3]; for (int d = 0; d < 3; d++) { a[d] = f->T[t].lo[d] / f->gcell; b[d] = f->T[t].hi[d] / f->gcell; if (a[d] < 0) a[d] = 0; if (b[d] >= f->gdim[d]) b[d] = f->gdim[d] - 1; }
                for (int64_t z = a[0]; z <= b[0]; z++) for (int64_t y = a[1]; y <= b[1]; y++) for (int64_t x = a[2]; x <= b[2]; x++) {
                    const size_t gi = ((size_t)z * f->gdim[1] + y) * f->gdim[2] + x;
                    if (!pass) { f->gstart[gi + 1]++; tot++; } else f->glist[f->gstart[gi] + fill[gi]++] = (uint32_t)t;
                }
            }
            if (!pass) { for (size_t g = 0; g < ng; g++) f->gstart[g + 1] += f->gstart[g]; f->glist = malloc((tot ? tot : 1) * sizeof *f->glist); }
            free(fill);
        }
    }
    f->zc = zcache_open((size_t)1 << 30);
    z3 *z = z3_open_derived(ct, 255, inkfly_derive, f, inkfly_free);
    if (!z) { free(key); free(sg_txt); inkfly_free(f); z3_close(ct); return nullptr; }
    if (handle) *handle = f;
    z = z3_open_cached(z, dcache_dir, key, IFC, cache_mb ? cache_mb : 2048); free(key);
    free(sg_txt);
    if (!z) return nullptr;
    fprintf(stderr, "ink onfly: %s: %d label windows, %d meshes, %zu supervised tiles indexed in %.1fs%s (level %d, face [%g, %g], iso %d, %d held-out boxes, cache %zu MB, disk %s)\n",
            segs_path, f->nU, f->nS, f->nT, snow() - t0, have_ix ? " (index from the cache)" : "", f->L, cfg->face_a, cfg->face_b, cfg->iso, cfg->nexcl, cache_mb ? cache_mb : 2048, dcache_dir && *dcache_dir ? dcache_dir : "none");
    return z;
fail:
    json_free(j); free(sg_txt); inkfly_free(f); z3_close(ct); return nullptr;
}
void inkfly_stats(uint64_t *reads, uint64_t *chunks, uint64_t *empty, uint64_t *hits, uint64_t *px, double *sec) {
    *reads = atomic_load(&g_ifs.reads); *chunks = atomic_load(&g_ifs.chunks); *empty = atomic_load(&g_ifs.empty); *hits = atomic_load(&g_ifs.hits);
    *px = atomic_load(&g_ifs.px); *sec = atomic_load(&g_ifs.us) * 1e-6;
}
static const char *base_name(const char *k) { static __thread char b[512]; snprintf(b, sizeof b, "%s", k); size_t l = strlen(b); while (l && b[l - 1] == '/') b[--l] = 0; const char *s = strrchr(b, '/'); return s ? s + 1 : b; }
int inkfly_holdouts(const char *path, const char *ct_key, inksplat_cfg *cfg) {
    FILE *fp = fopen(path, "rb"); if (!fp) { fprintf(stderr, "ink onfly: cannot open %s\n", path); return -1; }
    fseek(fp, 0, SEEK_END); long sz = ftell(fp); fseek(fp, 0, SEEK_SET); char *t = malloc((size_t)sz + 1); size_t rd = fread(t, 1, (size_t)sz, fp); fclose(fp); t[rd] = 0;
    json *j = json_parse(t, rd); free(t); const json *bx = json_get(j, "boxes"); char want[512]; snprintf(want, sizeof want, "%s", base_name(ct_key));
    for (size_t i = 0; bx && i < bx->n; i++) {
        const json *e = json_at(bx, i); const char *ck = json_str(json_get(e, "ct"), ""); const json *l0 = json_get(e, "l0");
        if (strcmp(base_name(ck), want) || !l0 || l0->n != 6) continue;
        if (cfg->nexcl >= 64) { fprintf(stderr, "ink onfly: more than 64 held-out boxes for %s\n", want); json_free(j); return -1; }
        for (int d = 0; d < 6; d++) cfg->excl[cfg->nexcl][d] = (int64_t)json_num(json_at(l0, (size_t)d), 0);
        cfg->nexcl++;
    }
    json_free(j); return 0;
}
typedef struct { z3 *z; const int64_t *o, *n; uint8_t *A; _Atomic int64_t next; _Atomic int failed; } sl_t;
static void *slab(void *p) { sl_t *J = p; for (int64_t zs; (zs = atomic_fetch_add(&J->next, 128)) < J->n[0];) {
    const int64_t oo[3] = {J->o[0] + zs, J->o[1], J->o[2]}, nn2[3] = {zs + 128 < J->n[0] ? 128 : J->n[0] - zs, J->n[1], J->n[2]};
    if (z3_read(J->z, oo, nn2, J->A + (size_t)zs * J->n[1] * J->n[2], 1)) atomic_store(&J->failed, 1); } return nullptr; }
/* ufsm ink-onfly SEGS --ct ROOT KEY --ct-um U [--level 1] [--iso 64] [--face -6,0] [--holdout-file H] --box z,y,x,nz,ny,nx [--ref ROOT KEY] [--reads N]
   the derived target over a box (level-0 voxels): chunks computed, time, and with --ref the byte comparison against a stored splat;
   --reads N: also N random 256^3 windows inside the box (ms / window, cold and warm) */
int cmd_ink_onfly(int argc, char **argv) {
    const int ci = sidx(argc, argv, "--ct"), ri = sidx(argc, argv, "--ref"); const char *bx = sopt(argc, argv, "--box", nullptr);
    long long b[6];
    if (argc < 3 || ci < 0 || ci + 2 >= argc || !bx || sscanf(bx, "%lld,%lld,%lld,%lld,%lld,%lld", b, b + 1, b + 2, b + 3, b + 4, b + 5) != 6) {
        fprintf(stderr, "usage: ufsm ink-onfly SEGS.json --ct ROOT KEY --ct-um U [--level 1] [--iso 64] [--face -6,0] [--holdout-file H] --box z,y,x,nz,ny,nx (level-0)\n"
                        "         [--rays (the MIL ray counts)] [--ref ROOT KEY (a stored ink-splat level)] [--threads 8] [--cache DIR] [--cache-mb 2048] [--dcache DIR|off (default $UFSM_DERIVED_CACHE or ~/.cache/ufsm/derived)]\n"
                        "         the on-the-fly ink target over a box\n"); return 2;
    }
    inksplat_cfg c; inksplat_defaults(&c); c.face = 1;
    c.ct_um = atof(sopt(argc, argv, "--ct-um", "0")); c.level = atoi(sopt(argc, argv, "--level", "1")); c.iso = atoi(sopt(argc, argv, "--iso", "64")); c.cache = sopt(argc, argv, "--cache", nullptr);
    if (sscanf(sopt(argc, argv, "--face", "-6,0"), "%lf,%lf", &c.face_a, &c.face_b) != 2) return 2;
    const char *hf = sopt(argc, argv, "--holdout-file", nullptr);
    if (hf && inkfly_holdouts(hf, argv[ci + 2], &c)) return 1;
    const int nth = ufsm_threads(atoi(sopt(argc, argv, "--threads", "0")), 8);
    store *cs = store_open(argv[ci + 1]); z3 *ct = cs ? pyramid_open_level(cs, argv[ci + 2], c.level, c.ct_um, c.cache) : nullptr;
    if (!ct) { fprintf(stderr, "ink-onfly: CT level %d: %s\n", c.level, z3_error()); return 1; }
    c.ct_root = argv[ci + 1]; c.ct_key = argv[ci + 2];
    const char *dcd = sopt(argc, argv, "--dcache", dcache_default_dir());   /* --dcache DIR | off */
    if (dcd && !strcmp(dcd, "off")) dcd = nullptr;
    void *hd = nullptr; z3 *z = inkfly_open(ct, argv[2], &c, (size_t)atol(sopt(argc, argv, "--cache-mb", "2048")), dcd, &hd);
    if (!z) return 1;
    if (sflag(argc, argv, "--rays")) {   /* the MIL rays of the box: every supervised pixel vs those without an open face */
        const int L0 = c.level; int64_t ro[3], rn[3]; for (int d = 0; d < 3; d++) { ro[d] = b[d] >> L0; rn[d] = ((b[d] + b[d + 3] + (1 << L0) - 1) >> L0) - ro[d]; }
        inkray *r0 = nullptr, *r1 = nullptr; size_t n0 = 0, n1 = 0; const double tr = snow();
        if (inkfly_rays(hd, ro, rn, 0, 1, 10, &r0, &n0) || inkfly_rays(hd, ro, rn, 1, 1, 10, &r1, &n1)) { fprintf(stderr, "ink-onfly: rays failed\n"); return 1; }
        size_t i0 = 0, i1 = 0; for (size_t i = 0; i < n0; i++) i0 += r0[i].y; for (size_t i = 0; i < n1; i++) i1 += r1[i].y;
        printf("rays: %zu supervised pixels (%.1f %% ink), %zu without an open face (%.1f %%; %.1f %% ink) in %.1f s\n", n0, 100.0 * i0 / (n0 ? n0 : 1), n1, 100.0 * n1 / (n0 ? n0 : 1),
               100.0 * i1 / (n1 ? n1 : 1), snow() - tr);
        free(r0); free(r1);
    }
    const int L = c.level; int64_t o[3], n[3]; size_t nv = 1;
    for (int d = 0; d < 3; d++) { o[d] = b[d] >> L; n[d] = ((b[d] + b[d + 3] + (1 << L) - 1) >> L) - o[d]; nv *= (size_t)n[d]; }
    uint8_t *A = malloc(nv); const double t0 = snow();
    /* parallel over z slabs of 128 (each slab one read: distinct chunks derive concurrently) */
    sl_t SJ = {z, o, n, A, 0, 0};
    pthread_t th[64]; const int ntt = nth > 64 ? 64 : nth;
    for (int i = 0; i < ntt; i++) pthread_create(&th[i], nullptr, slab, &SJ);
    for (int i = 0; i < ntt; i++) pthread_join(th[i], nullptr);
    const int failed = atomic_load(&SJ.failed);
    const double t1 = snow();
    if (failed) { fprintf(stderr, "ink-onfly: read failed\n"); return 1; }
    uint64_t rd, ch, em, hi, px; double sec; inkfly_stats(&rd, &ch, &em, &hi, &px, &sec);
    uint64_t kn = 0, ink = 0; for (size_t k = 0; k < nv; k++) if (A[k] != 255) { kn++; ink += A[k] >= 127; }
    fprintf(stderr, "ink-onfly: CT reads %.1f s of the derive time (summed); %llu chunks on the GPU, %.1f s in the GPU call (summed, lock waits included)\n",
            atomic_load(&g_ifs.ct_us) * 1e-6, (unsigned long long)atomic_load(&g_ifs.gpu_chunks), atomic_load(&g_ifs.gpu_us) * 1e-6);
    { dcache_stats ds; dcache_get_stats(&ds);
      printf("dcache: %llu cells from memory, %llu from disk (%.2f s), %llu computed (%llu constant, %.1f s), %.1f MB written\n", (unsigned long long)ds.cells_mem,
             (unsigned long long)ds.cells_disk, ds.s_disk, (unsigned long long)ds.cells_computed, (unsigned long long)ds.cells_fill, ds.s_compute, ds.disk_bytes_written / 1048576.0); }
    printf("onfly box %lld,%lld,%lld +%lld,%lld,%lld (level %d): %.1f s wall, %llu chunks computed (%llu empty), %llu label px faced, %.1f s derive (summed); %llu known voxels (%llu ink)\n",
           (long long)o[0], (long long)o[1], (long long)o[2], (long long)n[0], (long long)n[1], (long long)n[2], L, t1 - t0, (unsigned long long)ch, (unsigned long long)em,
           (unsigned long long)px, sec, (unsigned long long)kn, (unsigned long long)ink);
    if (ri >= 0 && ri + 2 < argc) {
        store *rs = store_open(argv[ri + 1]); z3 *r = rs ? z3_open(rs, argv[ri + 2], nullptr) : nullptr;
        uint8_t *B = malloc(nv);
        if (!r || z3_read(r, o, n, B, nth)) { fprintf(stderr, "ink-onfly: --ref: %s\n", z3_error()); return 1; }
        uint64_t diff = 0, onlyA = 0, onlyB = 0, kb = 0; for (size_t k = 0; k < nv; k++) { kb += B[k] != 255; if (A[k] != B[k]) { diff++; onlyA += B[k] == 255; onlyB += A[k] == 255; } }
        printf("vs ref %s/%s: %llu of %zu voxels differ (%llu known only on the fly, %llu known only in the ref; ref known %llu)%s\n", argv[ri + 1], argv[ri + 2], (unsigned long long)diff, nv,
               (unsigned long long)onlyA, (unsigned long long)onlyB, (unsigned long long)kb, diff ? "" : "  BYTE-IDENTICAL");
        free(B); z3_close(r); store_close(rs);
    }
    /* warm re-read timing: 256^3 windows inside the box */
    if (n[0] >= 256 && n[1] >= 256 && n[2] >= 256) {
        uint8_t *W = malloc((size_t)256 * 256 * 256); const int64_t wn[3] = {256, 256, 256}; const double t2 = snow(); int nw = 8;
        for (int i = 0; i < nw; i++) { const int64_t wo[3] = {o[0] + (int64_t)(i * 37 % 8) * (n[0] - 256) / 8, o[1] + (int64_t)(i * 13 % 8) * (n[1] - 256) / 8, o[2] + (int64_t)(i * 5 % 8) * (n[2] - 256) / 8}; z3_read(z, wo, wn, W, 1); }
        printf("warm 256^3 window reads: %.1f ms / window\n", (snow() - t2) * 1e3 / nw); free(W);
    }
    free(A); z3_close(z); store_close(cs);
    return 0;
}
/* the normal-pooled MIL rays of a window (2026-11-11, villa's normal_pooled_3d): per supervised label pixel (every stride-th row / col of
   its label window) whose mesh point lies in the window, the point and the unit normal in the window's level voxels and the 2-D label.
   mode 1: only the pixels without an open face (the ones the --face voxel target leaves unknown; the face search as inkfly_chunk's on
   the level's CT); mode 0: every pixel. Pixels within reach (R level voxels + 1) of a held-out box are left out. */
int inkfly_rays(void *handle, const int64_t o[3], const int64_t n[3], int mode, int stride, float R, inkray **out, size_t *nout) {
    inkfly *f = handle; *out = nullptr; *nout = 0; if (!f) return -1;
    const inksplat_cfg *c = &f->c; const double V = f->V; if (stride < 1) stride = 1;
    int64_t a0[3], b0[3]; for (int d = 0; d < 3; d++) { a0[d] = o[d] * (int64_t)V; b0[d] = (o[d] + n[d]) * (int64_t)V; }
    const double ex_m = (R + 1) * V;
    size_t cap = 256, nc = 0; uint32_t *cand = malloc(cap * sizeof *cand);
    int64_t g0[3], g1[3];
    for (int d = 0; d < 3; d++) { g0[d] = a0[d] / f->gcell; g1[d] = (b0[d] - 1) / f->gcell; if (g0[d] < 0) g0[d] = 0; if (g1[d] >= f->gdim[d]) g1[d] = f->gdim[d] - 1; }
    for (int64_t z = g0[0]; z <= g1[0]; z++) for (int64_t y = g0[1]; y <= g1[1]; y++) for (int64_t x = g0[2]; x <= g1[2]; x++) {
        const size_t gi = ((size_t)z * f->gdim[1] + y) * f->gdim[2] + x;
        for (uint32_t k = f->gstart[gi]; k < f->gstart[gi + 1]; k++) {
            const uint32_t t = f->glist[k]; if (!box_hit(f->T[t].lo, f->T[t].hi, a0, b0)) continue;
            int dup = 0; for (size_t q = nc; q-- > 0 && !dup;) dup = cand[q] == t;
            if (dup) continue;
            if (nc == cap) { cap *= 2; cand = realloc(cand, cap * sizeof *cand); }
            cand[nc++] = t;
        }
    }
    for (size_t i = 1; i < nc; i++) { uint32_t t = cand[i]; size_t j = i; while (j > 0 && (f->T[cand[j - 1]].unit > f->T[t].unit || (f->T[cand[j - 1]].unit == f->T[t].unit && f->T[cand[j - 1]].tile > f->T[t].tile))) { cand[j] = cand[j - 1]; j--; } cand[j] = t; }
    /* candidate pixels: point in the window, not near a held-out box */
    typedef struct { double P[3], N[3]; uint8_t y; } cpx;
    size_t pc = 1024, np = 0; cpx *X = malloc(pc * sizeof *X); int rc = -1;
    double lo3[3] = {1e300, 1e300, 1e300}, hi3[3] = {-1e300, -1e300, -1e300}; const int reach = (int)ceil(fmax(fabs(c->face_lo), fabs(c->face_hi))) + 2;
    for (size_t q0 = 0; q0 < nc;) {
        size_t q1 = q0; const ifunit *u = &f->U[f->T[cand[q0]].unit]; while (q1 < nc && f->T[cand[q1]].unit == f->T[cand[q0]].unit) q1++;
        int64_t rr0 = INT64_MAX, rc0 = INT64_MAX, rr1 = 0, rc1 = 0;
        for (size_t q = q0; q < q1; q++) { const int t = (int)f->T[cand[q]].tile, tx = t % u->ntx, ty = t / u->ntx;
            const int64_t r0 = (int64_t)ty * TS, c0 = (int64_t)tx * TS, r1 = r0 + TS < u->lh ? r0 + TS : u->lh, c1 = c0 + TS < u->lw ? c0 + TS : u->lw;
            if (r0 < rr0) { rr0 = r0; } if (c0 < rc0) { rc0 = c0; } if (r1 > rr1) { rr1 = r1; } if (c1 > rc1) { rc1 = c1; } }
        const int64_t rh = rr1 - rr0, rw = rc1 - rc0;
        uint8_t *lab = malloc((size_t)rh * rw), *msk = u->msk >= 0 ? malloc((size_t)rh * rw) : nullptr;
        if (!lab || (u->msk >= 0 && !msk) || lsrc_read(&f->Ls[u->lab], u->lr0 + rr0, u->lc0 + rc0, rh, rw, lab, 1) || (msk && lsrc_read(&f->Ls[u->msk], u->lr0 + rr0, u->lc0 + rc0, rh, rw, msk, 1))) { free(lab); free(msk); goto done; }
        if (u->lab01) for (size_t k = 0; k < (size_t)rh * rw; k++) lab[k] *= 255;
        if (msk && u->msk01) for (size_t k = 0; k < (size_t)rh * rw; k++) msk[k] *= 255;
        geo G = {0}; G.c = c; G.S = &f->S[u->surf]; G.F = u->F; G.lab = lab; G.msk = msk; G.lw = rw; G.lh = rh; G.lr0 = u->lr0 + rr0; G.lc0 = u->lc0 + rc0; G.offr = c->label_off[0]; G.offc = c->label_off[1];
        for (size_t q = q0; q < q1; q++) {
            const int t = (int)f->T[cand[q]].tile, tx = t % u->ntx, ty = t / u->ntx;
            const int64_t r0 = (int64_t)ty * TS, c0 = (int64_t)tx * TS, r1 = r0 + TS < u->lh ? r0 + TS : u->lh, c1 = c0 + TS < u->lw ? c0 + TS : u->lw;
            for (int64_t r = r0; r < r1; r++) { if ((u->lr0 + r) % stride) continue;
                for (int64_t cc = c0; cc < c1; cc++) { if ((u->lc0 + cc) % stride) continue;
                    if (!px_sup(&G, r - rr0, cc - rc0)) continue;
                    double gu, gv, P[3], N[3]; px_grid(&G, (double)(r - rr0), (double)(cc - rc0), &gu, &gv);
                    if (ssurf_pn(G.S, gu, gv, P, N)) continue;
                    int in = 1; for (int d = 0; d < 3 && in; d++) in = P[d] >= a0[d] && P[d] < b0[d];
                    if (!in) continue;
                    int ex = 0; for (int e = 0; e < c->nexcl && !ex; e++) { const int64_t *bx = c->excl[e]; ex = 1; for (int d = 0; d < 3 && ex; d++) ex = P[d] > bx[d] - ex_m && P[d] < bx[d] + bx[3 + d] + ex_m; }
                    if (ex) continue;
                    if (np == pc) { pc *= 2; X = realloc(X, pc * sizeof *X); }
                    cpx *x = &X[np++]; memcpy(x->P, P, sizeof P); memcpy(x->N, N, sizeof N); x->y = lab[(size_t)(r - rr0) * rw + (cc - rc0)] >= 128;
                    for (int d = 0; d < 3; d++) { if (P[d] - reach < lo3[d]) lo3[d] = P[d] - reach; if (P[d] + reach > hi3[d]) hi3[d] = P[d] + reach; }
                } }
        }
        free(lab); free(msk); q0 = q1;
    }
    uint8_t *ctb = nullptr; int64_t co[3] = {0, 0, 0}, cn[3] = {0, 0, 0};
    if (mode == 1 && np) {   /* the face search on the level's CT, as inkfly_chunk */
        size_t ncv = 1; for (int d = 0; d < 3; d++) { co[d] = (int64_t)floor((lo3[d] - f->off) / V) - 1; cn[d] = (int64_t)ceil((hi3[d] - f->off) / V) + 2 - co[d]; ncv *= (size_t)cn[d]; }
        ctb = malloc(ncv);
        if (!ctb || (f->zc ? zcache_read(f->zc, f->base_ct, co, cn, ctb) : z3_read(f->base_ct, co, cn, ctb, 1))) { free(ctb); goto done; }
    }
    inkray *Rr = malloc((np ? np : 1) * sizeof *Rr); size_t nr = 0;
    const int lo = (int)floor(c->face_lo), hi = (int)ceil(c->face_hi), ns = hi - lo + 1; const double sign = c->flip ? -1 : 1;
    float raw[256], sm[256];
    for (size_t i = 0; i < np; i++) {
        const cpx *x = &X[i];
        if (mode == 1) {
            for (int k = 0; k < ns; k++) { const double tt = (lo + k) * sign; double qq[3];
                for (int d = 0; d < 3; d++) qq[d] = (x->P[d] + tt * x->N[d] - f->off) / V - co[d];
                const float v = tri_box(ctb, cn, qq[0], qq[1], qq[2]); raw[k] = v < 0 ? 0 : v; }
            for (int k = 0; k < ns; k++) sm[k] = 0.25f * (raw[k > 0 ? k - 1 : 0] + 2 * raw[k] + raw[k < ns - 1 ? k + 1 : ns - 1]);
            if (face_of(sm, ns, lo, c->iso, c->face_run, c->face_gap) != FACE_NONE) continue;
        }
        inkray *r = &Rr[nr++];
        for (int d = 0; d < 3; d++) { r->p[d] = (float)((x->P[d] - f->off) / V - o[d]); r->n[d] = (float)(x->N[d] * sign); }
        r->y = x->y; r->w = 1; memset(r->pad, 0, sizeof r->pad);
    }
    free(ctb); *out = Rr; *nout = nr; rc = 0;
done:
    free(X); free(cand); return rc;
}
/* inkmil.h: the MIL loss's host reference (double; the tests compare the GPU against it and it against central differences) */
static double tri_d(const float *v, int D, int H, int W, double z, double y, double x, int c[3], double f[3]) {
    if (!(z >= 0 && y >= 0 && x >= 0 && z <= D - 1 && y <= H - 1 && x <= W - 1)) return NAN;
    int z0 = (int)z, y0 = (int)y, x0 = (int)x; if (z0 > D - 2) z0 = D - 2; if (y0 > H - 2) y0 = H - 2; if (x0 > W - 2) x0 = W - 2;
    f[0] = z - z0; f[1] = y - y0; f[2] = x - x0; c[0] = z0; c[1] = y0; c[2] = x0; double s = 0;
    for (int q = 0; q < 8; q++) s += ((q >> 2 & 1) ? f[0] : 1 - f[0]) * ((q >> 1 & 1) ? f[1] : 1 - f[1]) * ((q & 1) ? f[2] : 1 - f[2]) * v[((size_t)(z0 + (q >> 2 & 1)) * H + y0 + (q >> 1 & 1)) * W + x0 + (q & 1)];
    return s;
}
double ink_mil_loss_host(const float *logit, double *grad, int D, int H, int W, const inkray *rays, int nr, float R, float step, float tau, float eps, float scale) {
    const int K = (int)floorf(2 * R / step + 1e-4f) + 1; double L = 0, *z = malloc((size_t)K * sizeof *z);
    for (int i = 0; i < nr; i++) {
        const inkray *r = &rays[i]; int nv = 0, am = -1; double zm = -INFINITY;
        for (int k = 0; k < K; k++) { const double t = -R + k * (double)step; int c[3]; double f[3];
            z[k] = tri_d(logit, D, H, W, r->p[0] + t * r->n[0], r->p[1] + t * r->n[1], r->p[2] + t * r->n[2], c, f);
            if (z[k] == z[k]) { nv++; if (z[k] > zm) { zm = z[k]; am = k; } } }
        if (!nv) continue;
        double zr, se = 0;
        if (tau > 0) { for (int k = 0; k < K; k++) if (z[k] == z[k]) se += exp((z[k] - zm) / tau); zr = zm + tau * log(se / nv); } else zr = zm;
        const double y = r->y ? 1 - eps / 2.0 : eps / 2.0;
        L += scale * r->w * (fmax(zr, 0) - zr * y + log1p(exp(-fabs(zr))));
        if (!grad) continue;
        const double g = scale * r->w * (1 / (1 + exp(-zr)) - y);
        for (int k = 0; k < K; k++) {
            if (!(z[k] == z[k])) continue;
            const double dk = tau > 0 ? exp((z[k] - zm) / tau) / se : (k == am); if (dk == 0) continue;
            const double t = -R + k * (double)step; int c[3]; double f[3];
            tri_d(logit, D, H, W, r->p[0] + t * r->n[0], r->p[1] + t * r->n[1], r->p[2] + t * r->n[2], c, f);
            for (int q = 0; q < 8; q++) grad[((size_t)(c[0] + (q >> 2 & 1)) * H + c[1] + (q >> 1 & 1)) * W + c[2] + (q & 1)] +=
                g * dk * ((q >> 2 & 1) ? f[0] : 1 - f[0]) * ((q >> 1 & 1) ? f[1] : 1 - f[1]) * ((q & 1) ? f[2] : 1 - f[2]);
        }
    }
    free(z); return L;
}
int inkfly_cells(void *handle, int lev, const int64_t shape[3], uint8_t *bits) {
    const inkfly *f = handle; if (!f) return -1; const int64_t cs = (int64_t)1 << lev; size_t n = 0;
    for (size_t t = 0; t < f->nT; t++) {
        int64_t a[3], b[3]; for (int d = 0; d < 3; d++) { a[d] = f->T[t].lo[d] / cs; b[d] = f->T[t].hi[d] / cs; if (a[d] < 0) a[d] = 0; if (b[d] >= shape[d]) b[d] = shape[d] - 1; }
        for (int64_t z = a[0]; z <= b[0]; z++) for (int64_t y = a[1]; y <= b[1]; y++) for (int64_t x = a[2]; x <= b[2]; x++) { uint8_t *q = &bits[((size_t)z * shape[1] + y) * shape[2] + x]; n += !*q; *q = 1; }
    }
    return (int)(n > 2000000000 ? 2000000000 : n);
}
