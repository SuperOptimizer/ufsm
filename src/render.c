/* ufsm render-sheets: every sheet (tifxyz) of a box rendered in one process -> 5-panel red-ink overlay JPEGs, and the C
   commands that replaced the Python steps of the render pipeline (overlay-red, tifxyz-scale, png2jpg).

   render-sheets renders the CT pass (CT at --ct-div, max over +-ct-depth voxels along the normal) and the ink pass (all
   --ink-depths in one pass: the samples of the largest depth, maxima over each smaller window) for a batch of sheets at once.
   The pixels are cut into 32 x 32 blocks; each block needs the 128^3 volume chunks around its vertices (+- depth + 2); the
   blocks of all sheets are sorted by their first chunk (z, y, x) and rendered a (z, y) chunk row at a time, after decoding the
   chunks that row needs that are not yet in memory (least recently needed ones dropped past the cache budget). Every chunk the
   sheets pass through is so decoded about once per batch instead of once per sheet and output tile. The sampling arithmetic is
   the per-wrap ufsm render's (render_rows / trilin in main.c) on the same float32 coordinates xform_wraps.py wrote, so the
   renders are bit-identical to it; the overlay is overlay_red.py's (imgops.c). */
#include "imgops.h"
#include "store.h"
#include "tiffw.h"
#include "zarr3.h"
#include <dirent.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static const char *ropt(int argc, char **argv, const char *name, const char *def) {
    for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return def;
}
static int rflag(int argc, char **argv, const char *name) { for (int i = 2; i < argc; i++) if (!strcmp(argv[i], name)) return 1; return 0; }
static int rarg(int argc, char **argv, const char *name) { for (int i = 2; i < argc; i++) if (!strcmp(argv[i], name)) return i; return -1; }
static double rnow(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

/* ---- a sheet: its tifxyz grids (as written) and renders ---- */
typedef struct {
    char name[256]; double area;
    int w[2], h[2]; float *X[2], *Y[2], *Z[2]; double scale[2];   /* per pass (0 CT, 1 ink): the grid and its meta scale */
    int ow[2], oh[2]; uint8_t *img[2][4];                          /* renders per pass and depth */
} rsheet;

static int meta_scale_area(const char *dir, double *scale, double *area) {
    char p[1400]; snprintf(p, sizeof p, "%s/meta.json", dir); FILE *f = tfx_meta_fopen(p); if (!f) return -1;
    char b[16384] = {0}; const size_t n = fread(b, 1, sizeof b - 1, f); (void)n; fclose(f);
    const char *q = strstr(b, "\"scale\": ["); if (!q) return -1;
    *scale = atof(q + 10);
    q = strstr(b, "\"area_cm2\":"); *area = q ? atof(q + 11) : 0;
    return 0;
}
/* the grids of pass k: dir/<name>/{x,y,z}.tif divided by div as xform_wraps.py did (div 1: as written); the meta scale x div */
static int sheet_load(rsheet *s, int k, const char *dir, double div) {
    char p[1400]; static const char *nm[3] = {"x", "y", "z"}; float *A[3] = {0}; int w = 0, h = 0;
    snprintf(p, sizeof p, "%s/%s", dir, s->name); double area;
    if (meta_scale_area(p, &s->scale[k], &area)) return -1;
    s->scale[k] *= div;
    for (int d = 0; d < 3; d++) {
        int w2, h2; snprintf(p, sizeof p, "%s/%s/%s.tif", dir, s->name, nm[d]); A[d] = img_read_f32(p, &w2, &h2);
        if (!A[d] || (d && (w2 != w || h2 != h))) { for (int e = 0; e <= d; e++) free(A[e]); return -1; }
        w = w2; h = h2;
    }
    if (w < 2 || h < 2) { for (int d = 0; d < 3; d++) free(A[d]); return -1; }
    if (div != 1) {
        const float df = (float)div; const size_t n = (size_t)w * h;
        for (size_t i = 0; i < n; i++) { const float z = A[2][i]; A[0][i] = tifxyz_scaled(A[0][i], z, df, 0.f); A[1][i] = tifxyz_scaled(A[1][i], z, df, 0.f); A[2][i] = tifxyz_scaled(z, z, df, 0.f); }
    }
    s->X[k] = A[0]; s->Y[k] = A[1]; s->Z[k] = A[2]; s->w[k] = w; s->h[k] = h;
    const double gs = 1 / s->scale[k];   /* render --scale 1: output pixels per grid step = the grid step */
    s->ow[k] = (int)((w - 1) * gs) + 1; s->oh[k] = (int)((h - 1) * gs) + 1;
    return 0;
}

/* ---- decoded volume chunks over a box of chunk indices ---- */
typedef struct {
    z3 *z; int64_t shape[3]; int sh;            /* chunk edge 1 << sh */
    int64_t c0[3], cn[3]; uint8_t **ptr; uint32_t *last;   /* per chunk: decoded data (zero: the shared zero chunk), last group needing it */
    uint8_t *zero; size_t budget, resident; int64_t *res; size_t nres, capres;
    uint64_t ndecoded;
} ctab;
static inline uint8_t ct_voxel(const ctab *T, int64_t z, int64_t y, int64_t x) {
    const int64_t a = (z >> T->sh) - T->c0[0], b = (y >> T->sh) - T->c0[1], c = (x >> T->sh) - T->c0[2];
    if (a < 0 || b < 0 || c < 0 || a >= T->cn[0] || b >= T->cn[1] || c >= T->cn[2]) return 0;
    const uint8_t *p = T->ptr[(a * T->cn[1] + b) * T->cn[2] + c]; const int64_t m = ((int64_t)1 << T->sh) - 1;
    return p ? p[((((z & m) << T->sh) + (y & m)) << T->sh) + (x & m)] : 0;
}
/* main.c trilin() on the global volume: the same weights, products and summation order (coordinates < 0 are 0 there too) */
static inline float ct_trilin(const ctab *T, double z, double y, double x) {
    if (z < 0 || y < 0 || x < 0) return 0.f;
    const int z0 = (int)z, y0 = (int)y, x0 = (int)x; const double fz = z - z0, fy = y - y0, fx = x - x0;
    const int64_t m = ((int64_t)1 << T->sh) - 1; double v[8];
    if ((z0 & m) != m && (y0 & m) != m && (x0 & m) != m) {
        const int64_t a = (z0 >> T->sh) - T->c0[0], b = (y0 >> T->sh) - T->c0[1], c = (x0 >> T->sh) - T->c0[2];
        const uint8_t *p = a >= 0 && b >= 0 && c >= 0 && a < T->cn[0] && b < T->cn[1] && c < T->cn[2] ? T->ptr[(a * T->cn[1] + b) * T->cn[2] + c] : nullptr;
        if (!p) return 0.f;
        p += ((((z0 & m) << T->sh) + (y0 & m)) << T->sh) + (x0 & m);
        const size_t sy = (size_t)1 << T->sh, sz = sy << T->sh;
        v[0] = p[0]; v[1] = p[1]; v[2] = p[sy]; v[3] = p[sy + 1]; v[4] = p[sz]; v[5] = p[sz + 1]; v[6] = p[sz + sy]; v[7] = p[sz + sy + 1];
    } else for (int c = 0; c < 8; c++) v[c] = ct_voxel(T, z0 + (c >> 2 & 1), y0 + (c >> 1 & 1), x0 + (c & 1));
    double s = 0;
    for (int c = 0; c < 8; c++) s += (c >> 2 & 1 ? fz : 1 - fz) * (c >> 1 & 1 ? fy : 1 - fy) * (c & 1 ? fx : 1 - fx) * v[c];
    return (float)s;
}

/* ---- blocks ---- */
enum { RB = 32 };
typedef struct { int s, px, py; int32_t lo[3], hi[3]; int32_t vlo[2], vhi[2]; } rblock;   /* vlo / vhi: the voxel box's z, y (render_wraps_chunked) */
static int blk_cmp(const void *a, const void *b) {
    const rblock *p = a, *q = b;
    for (int d = 0; d < 3; d++) if (p->lo[d] != q->lo[d]) return p->lo[d] < q->lo[d] ? -1 : 1;
    if (p->s != q->s) return p->s < q->s ? -1 : 1;
    if (p->py != q->py) return p->py < q->py ? -1 : 1;
    return (p->px > q->px) - (p->px < q->px);
}

typedef struct {
    ctab *T; rsheet *S; int k, nd, dm; const int *depth;
    const rblock *b; size_t n; _Atomic size_t next;
    const int64_t *todo; size_t ntodo; _Atomic size_t tnext; _Atomic int failed;
} pass_job;
static void *decode_worker(void *arg) {
    pass_job *J = arg; ctab *T = J->T; const int64_t C = (int64_t)1 << T->sh; const size_t cv = (size_t)C * C * C;
    for (size_t t; (t = atomic_fetch_add(&J->tnext, 1)) < J->ntodo;) {
        const int64_t i = J->todo[t], a = i / (T->cn[1] * T->cn[2]), b = i / T->cn[2] % T->cn[1], c = i % T->cn[2];
        const int64_t o[3] = {(T->c0[0] + a) * C, (T->c0[1] + b) * C, (T->c0[2] + c) * C}, n[3] = {C, C, C};
        if (o[0] >= T->shape[0] || o[1] >= T->shape[1] || o[2] >= T->shape[2]) { T->ptr[i] = T->zero; continue; }
        uint8_t *buf = malloc(cv);
        if (!buf || z3_read(T->z, o, n, buf, 1)) { free(buf); atomic_store(&J->failed, 1); T->ptr[i] = T->zero; continue; }
        size_t q = 0; const uint64_t *w = (const uint64_t *)buf; while (q < cv / 8 && !w[q]) q++;
        if (q == cv / 8) { free(buf); T->ptr[i] = T->zero; } else T->ptr[i] = buf;
    }
    return nullptr;
}
static void *block_worker(void *arg) {   /* render_rows of main.c over the block's pixels, every depth at once */
    pass_job *J = arg; const int k = J->k, dm = J->dm;
    for (size_t bi; (bi = atomic_fetch_add(&J->next, 1)) < J->n;) {
        const rblock *B = &J->b[bi]; rsheet *S = &J->S[B->s];
        const int w = S->w[k], h = S->h[k], ow = S->ow[k], oh = S->oh[k]; const double gs = 1 / S->scale[k];
        const float *A[3] = {S->Z[k], S->Y[k], S->X[k]};
        const int py1 = B->py + RB < oh ? B->py + RB : oh, px1 = B->px + RB < ow ? B->px + RB : ow;
        for (int py = B->py; py < py1; py++) for (int px = B->px; px < px1; px++) {
            const double gu = px / gs, gv = py / gs;
            int c0 = (int)gu, r0 = (int)gv; if (c0 >= w - 1) c0 = w - 2; if (r0 >= h - 1) r0 = h - 2; if (c0 < 0 || r0 < 0) continue;
            const double fu = gu - c0, fv = gv - r0;
            const size_t i00 = (size_t)r0 * w + c0, i01 = i00 + 1, i10 = i00 + w, i11 = i10 + 1;
            if (A[0][i00] < 0 || A[0][i01] < 0 || A[0][i10] < 0 || A[0][i11] < 0) continue;
            double P[3], du[3], dv[3];
            for (int d = 0; d < 3; d++) {
                const double a = A[d][i00], b = A[d][i01], c = A[d][i10], e = A[d][i11];
                P[d] = (1 - fv) * ((1 - fu) * a + fu * b) + fv * ((1 - fu) * c + fu * e);
                du[d] = (1 - fv) * (b - a) + fv * (e - c); dv[d] = (1 - fu) * (c - a) + fu * (e - b);
            }
            double N[3] = {du[1] * dv[2] - du[2] * dv[1], du[2] * dv[0] - du[0] * dv[2], du[0] * dv[1] - du[1] * dv[0]};
            const double nl = sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]); if (nl < 1e-12) continue;
            for (int d = 0; d < 3; d++) N[d] /= nl;
            float m[4] = {0, 0, 0, 0};
            for (int t = -dm; t <= dm; t++) {
                const float v = ct_trilin(J->T, P[0] + t * N[0], P[1] + t * N[1], P[2] + t * N[2]); const int at = t < 0 ? -t : t;
                for (int q = 0; q < J->nd; q++) if (at <= J->depth[q] && v > m[q]) m[q] = v;
            }
            for (int q = 0; q < J->nd; q++) S->img[k][q][(size_t)py * ow + px] = (uint8_t)lroundf(m[q]);
        }
    }
    return nullptr;
}

typedef struct { uint32_t last; int64_t i; } lpair;
static int lp_cmp(const void *a, const void *b) { const lpair *p = a, *q = b; return p->last < q->last ? -1 : p->last > q->last; }

/* one pass (k) over the sheets S[0..ns): every pixel of every sheet, depths depth[0..nd) (nd <= 4) */
static int render_pass(z3 *z, rsheet *S, int ns, int k, const int *depth, int nd, size_t budget, int nthreads, int verbose) {

    const z3_meta *zm = z3_meta_of(z); const int C = zm->chunk[0];
    if (C != zm->chunk[1] || C != zm->chunk[2] || (C & (C - 1))) { fprintf(stderr, "render-sheets: chunks must be cubes of a power of 2\n"); return -1; }
    int sh = 0; while ((1 << sh) < C) sh++;
    int dm = 0; for (int q = 0; q < nd; q++) if (depth[q] > dm) dm = depth[q];
    const double t0 = rnow();
    /* blocks and their chunk ranges */
    size_t nb = 0, cap = 1 << 16; rblock *b = malloc(cap * sizeof *b); if (!b) return -1;
    for (int s = 0; s < ns; s++) {
        rsheet *R = &S[s]; const int w = R->w[k], h = R->h[k], ow = R->ow[k], oh = R->oh[k]; const double gs = 1 / R->scale[k];
        for (int q = 0; q < nd; q++) if (!(R->img[k][q] = calloc((size_t)ow * oh, 1))) return -1;
        for (int py = 0; py < oh; py += RB) for (int px = 0; px < ow; px += RB) {
            const int py1 = (py + RB < oh ? py + RB : oh) - 1, px1 = (px + RB < ow ? px + RB : ow) - 1;
            int r0 = (int)(py / gs), r1 = (int)(py1 / gs), c0 = (int)(px / gs), c1 = (int)(px1 / gs);
            r0 = r0 > h - 2 ? h - 2 : r0; r1 = r1 > h - 2 ? h - 2 : r1; c0 = c0 > w - 2 ? w - 2 : c0; c1 = c1 > w - 2 ? w - 2 : c1;
            r1++; c1++;
            double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
            for (int r = r0; r <= r1; r++) for (int c = c0; c <= c1; c++) {
                const size_t i = (size_t)r * w + c; if (R->Z[k][i] < 0) continue;
                const double v[3] = {R->Z[k][i], R->Y[k][i], R->X[k][i]};
                for (int d = 0; d < 3; d++) { if (v[d] < lo[d]) lo[d] = v[d]; if (v[d] > hi[d]) hi[d] = v[d]; }
            }
            if (lo[0] > hi[0]) continue;
            if (nb == cap) { cap *= 2; rblock *g = realloc(b, cap * sizeof *b); if (!g) { free(b); return -1; } b = g; }
            rblock *B = &b[nb++]; B->s = s; B->px = px; B->py = py;
            for (int d = 0; d < 3; d++) {
                double a = floor(lo[d]) - dm - 2, e = ceil(hi[d]) + dm + 2; if (a < 0) a = 0; if (e < 0) e = 0;
                B->lo[d] = (int32_t)((int64_t)a >> sh); B->hi[d] = (int32_t)((int64_t)e >> sh);
            }
        }
    }
    qsort(b, nb, sizeof *b, blk_cmp);
    ctab T = {0}; T.z = z; T.sh = sh; T.budget = budget; for (int d = 0; d < 3; d++) T.shape[d] = zm->shape[d];
    int64_t clo[3] = {INT64_MAX, INT64_MAX, INT64_MAX}, chi[3] = {-1, -1, -1};
    for (size_t i = 0; i < nb; i++) for (int d = 0; d < 3; d++) { if (b[i].lo[d] < clo[d]) clo[d] = b[i].lo[d]; if (b[i].hi[d] > chi[d]) chi[d] = b[i].hi[d]; }
    int rc = 0;
    if (nb) {
        for (int d = 0; d < 3; d++) { T.c0[d] = clo[d]; T.cn[d] = chi[d] - clo[d] + 1; }
        const size_t nc = (size_t)(T.cn[0] * T.cn[1] * T.cn[2]), cv = (size_t)C * C * C;
        T.ptr = calloc(nc, sizeof *T.ptr); T.last = calloc(nc, sizeof *T.last); T.zero = calloc(cv, 1);
        uint32_t *need = calloc(nc, sizeof *need); int64_t *todo = malloc(nc * sizeof *todo); lpair *lp = malloc(nc * sizeof *lp);
        if (!T.ptr || !T.last || !T.zero || !need || !todo || !lp) return -1;
        uint32_t gid = 0; size_t live = 0;   /* decoded non-zero chunks in memory */
        for (size_t g0 = 0; g0 < nb && !rc;) {
            size_t g1 = g0 + 1; while (g1 < nb && b[g1].lo[0] == b[g0].lo[0] && b[g1].lo[1] == b[g0].lo[1]) g1++;
            gid++; size_t nt = 0;
            for (size_t i = g0; i < g1; i++)
                for (int64_t a = b[i].lo[0]; a <= b[i].hi[0]; a++) for (int64_t y = b[i].lo[1]; y <= b[i].hi[1]; y++) for (int64_t x = b[i].lo[2]; x <= b[i].hi[2]; x++) {
                    const size_t ci = (size_t)(((a - T.c0[0]) * T.cn[1] + (y - T.c0[1])) * T.cn[2] + (x - T.c0[2]));
                    if (need[ci] == gid) continue;
                    need[ci] = gid; T.last[ci] = gid;
                    if (!T.ptr[ci]) todo[nt++] = (int64_t)ci;
                }
            if ((live + nt) * cv > budget) {   /* drop the least recently needed decoded chunks this row does not need */
                size_t np = 0;
                for (size_t ci = 0; ci < nc; ci++) if (T.ptr[ci] && T.ptr[ci] != T.zero && need[ci] != gid) lp[np++] = (lpair){T.last[ci], (int64_t)ci};
                qsort(lp, np, sizeof *lp, lp_cmp);
                for (size_t q = 0; q < np && (live + nt) * cv > budget; q++) { free(T.ptr[lp[q].i]); T.ptr[lp[q].i] = nullptr; live--; }
            }
            pass_job J = {&T, S, k, nd, dm, depth, b + g0, g1 - g0, 0, todo, nt, 0, 0};
            const int nth = nthreads < 1 ? 1 : nthreads > 64 ? 64 : nthreads; pthread_t th[64];
            if (nt) {
                for (int t = 0; t < nth; t++) pthread_create(&th[t], nullptr, decode_worker, &J);
                for (int t = 0; t < nth; t++) pthread_join(th[t], nullptr);
                if (atomic_load(&J.failed)) { fprintf(stderr, "render-sheets: chunk read failed: %s\n", z3_error()); rc = -1; }
                for (size_t q = 0; q < nt; q++) if (T.ptr[todo[q]] != T.zero) live++;
                T.ndecoded += nt;
            }
            for (int t = 0; t < nth; t++) pthread_create(&th[t], nullptr, block_worker, &J);
            for (int t = 0; t < nth; t++) pthread_join(th[t], nullptr);
            g0 = g1;
        }
        for (size_t ci = 0; ci < nc; ci++) if (T.ptr[ci] != T.zero) free(T.ptr[ci]);
        free(T.ptr); free(T.last); free(T.zero); free(need); free(todo); free(lp);
    }
    if (verbose) fprintf(stderr, "render-sheets: pass %s: %d sheets, %zu blocks, %llu chunk decodes (%lld x %lld x %lld chunk box), depth %d, %.1fs\n",
                         k ? "ink" : "CT", ns, nb, (unsigned long long)T.ndecoded, (long long)T.cn[0], (long long)T.cn[1], (long long)T.cn[2], dm, rnow() - t0);
    free(b);
    return rc;
}

/* ---- ufsm render of one wrap, chunk-ordered and pipelined (render_wrap_chunked, imgops.h) ----
   The output is cut into RB x RB pixel blocks; each block's chunk ranges (per volume, +- depth + 2 voxels around its vertices) are
   known up front. Blocks are sorted by their first chunk (z, y, x) and grouped by (z, y) chunk row; a pre-pass gives every chunk the
   first and the last group that needs it, so each chunk is decoded exactly once, in first-use order, by decoder threads that run
   ahead of the block renderers (bounded by the budget, which only the earliest unfinished group may exceed), and is freed as soon as
   its last group is done. Renderers take blocks in order once every chunk of the block's group is decoded. The pixel arithmetic is
   main.c render_rows' (corners valid when Z >= 0, bilinear point and normal, max over +- depth of trilin; ct_trilin = trilin on the
   global volume), so the renders equal the tiled / union-box ones byte for byte. */
typedef struct { uint8_t v; int32_t ci; uint32_t first; } wdec;   /* decode list entry: volume, chunk, first group (1-based) */
typedef struct {
    int nv; ctab T[2]; int depth[2]; uint32_t *first[2], *last[2], *smask[2];   /* smask: per chunk the volcomp substreams any block reads */
    int ns; const float *const *Xs, *const *Ys, *const *Zs; int w, h; double gs; uint8_t *const *img; int ow, oh;   /* grid s, volume v: img[s nv + v] */
    rblock *b; int32_t (*blo)[2][3], (*bhi)[2][3]; uint32_t *bgrp; size_t nb; uint32_t ng;
    wdec *dl; size_t ndl; uint8_t *ready; size_t *need;   /* need[g]: decode-list entries with first <= g (g 1-based) */
    size_t *frees; size_t *free0;                         /* per group g: decode entries whose last group is g, frees[free0[g] .. free0[g + 1]) */
    uint32_t *gleft;                                      /* per group: blocks not yet rendered */
    pthread_mutex_t mu; pthread_cond_t cv;
    size_t ready_prefix; uint32_t gdone; size_t resident, budget, peak; uint64_t ndec, nzero;
    _Atomic size_t dnext, bnext; _Atomic int fail;
} wrend;
static void *wr_decoder(void *arg) {
    wrend *R = arg;
    for (size_t e; (e = atomic_fetch_add(&R->dnext, 1)) < R->ndl && !atomic_load(&R->fail);) {
        const wdec *D = &R->dl[e]; ctab *T = &R->T[D->v]; const int64_t C = (int64_t)1 << T->sh; const size_t cv = (size_t)C * C * C;
        pthread_mutex_lock(&R->mu);   /* the budget: only chunks of the earliest unfinished group may exceed it */
        while (R->resident + cv > R->budget && D->first > R->gdone + 1 && !atomic_load(&R->fail)) pthread_cond_wait(&R->cv, &R->mu);
        R->resident += cv; if (R->resident > R->peak) R->peak = R->resident;
        pthread_mutex_unlock(&R->mu);
        const int64_t i = D->ci, a = i / (T->cn[1] * T->cn[2]), bb = i / T->cn[2] % T->cn[1], c = i % T->cn[2];
        const int64_t o[3] = {(T->c0[0] + a) * C, (T->c0[1] + bb) * C, (T->c0[2] + c) * C}, n[3] = {C, C, C};
        uint8_t *buf = nullptr; int zero = 1;
        if (o[0] < T->shape[0] && o[1] < T->shape[1] && o[2] < T->shape[2]) {
            buf = malloc(cv);
            if (!buf || z3_read_sub(T->z, o, n, buf, 1, R->smask[D->v] ? R->smask[D->v][i] : 0)) { fprintf(stderr, "render: chunk read failed: %s\n", z3_error()); free(buf); buf = nullptr; atomic_store(&R->fail, 1); }
            else { size_t q = 0; const uint64_t *w = (const uint64_t *)buf; while (q < cv / 8 && !w[q]) q++; zero = q == cv / 8; }
        }
        if (zero) { free(buf); buf = nullptr; }
        pthread_mutex_lock(&R->mu);
        T->ptr[i] = buf ? buf : T->zero; R->ndec++;
        if (!buf) { R->resident -= cv; R->nzero++; }
        R->ready[e] = 1;
        while (R->ready_prefix < R->ndl && R->ready[R->ready_prefix]) R->ready_prefix++;
        pthread_cond_broadcast(&R->cv);
        pthread_mutex_unlock(&R->mu);
    }
    pthread_mutex_lock(&R->mu); pthread_cond_broadcast(&R->cv); pthread_mutex_unlock(&R->mu);
    return nullptr;
}
static void wr_block(const wrend *R, const rblock *B) {
    const int w = R->w, h = R->h; const double gs = R->gs;
    const int py1 = B->py + RB < R->oh ? B->py + RB : R->oh, px1 = B->px + RB < R->ow ? B->px + RB : R->ow;
    for (int si = 0; si < R->ns; si++) {
    const float *A[3] = {R->Zs[si], R->Ys[si], R->Xs[si]}, *const Zg = R->Zs[si]; uint8_t *const *img = R->img + (size_t)si * R->nv;
    for (int py = B->py; py < py1; py++) for (int px = B->px; px < px1; px++) {
        const double gu = px / gs, gv = py / gs;
        int c0 = (int)gu, r0 = (int)gv; if (c0 >= w - 1) c0 = w - 2; if (r0 >= h - 1) r0 = h - 2; if (c0 < 0 || r0 < 0) continue;
        const double fu = gu - c0, fv = gv - r0;
        const size_t i00 = (size_t)r0 * w + c0, i01 = i00 + 1, i10 = i00 + w, i11 = i10 + 1;
        if (!(Zg[i00] >= 0 && Zg[i01] >= 0 && Zg[i10] >= 0 && Zg[i11] >= 0)) continue;
        double P[3], du[3], dv[3];
        for (int d = 0; d < 3; d++) {
            const double a = A[d][i00], b = A[d][i01], c = A[d][i10], e = A[d][i11];
            P[d] = (1 - fv) * ((1 - fu) * a + fu * b) + fv * ((1 - fu) * c + fu * e);
            du[d] = (1 - fv) * (b - a) + fv * (e - c); dv[d] = (1 - fu) * (c - a) + fu * (e - b);
        }
        double N[3] = {du[1] * dv[2] - du[2] * dv[1], du[2] * dv[0] - du[0] * dv[2], du[0] * dv[1] - du[1] * dv[0]};
        const double nl = sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]); if (nl < 1e-12) continue;
        for (int d = 0; d < 3; d++) N[d] /= nl;
        for (int v = 0; v < R->nv; v++) {
            float m = 0;
            for (int t = -R->depth[v]; t <= R->depth[v]; t++) { const float s = ct_trilin(&R->T[v], P[0] + t * N[0], P[1] + t * N[1], P[2] + t * N[2]); if (s > m) m = s; }
            img[v][(size_t)py * R->ow + px] = (uint8_t)lroundf(m);
        }
    }
    }
}
static void *wr_renderer(void *arg) {
    wrend *R = arg;
    for (size_t k; (k = atomic_fetch_add(&R->bnext, 1)) < R->nb;) {
        const uint32_t g = R->bgrp[k];
        pthread_mutex_lock(&R->mu);
        while (R->ready_prefix < R->need[g] && !atomic_load(&R->fail)) pthread_cond_wait(&R->cv, &R->mu);
        pthread_mutex_unlock(&R->mu);
        if (atomic_load(&R->fail)) break;
        wr_block(R, &R->b[k]);
        pthread_mutex_lock(&R->mu);
        if (!--R->gleft[g] && g == R->gdone + 1) {   /* groups done in order: free the chunks they needed last */
            while (R->gdone < R->ng && !R->gleft[R->gdone + 1]) {
                const uint32_t q = ++R->gdone;
                for (size_t f = R->free0[q]; f < R->free0[q + 1]; f++) {
                    const wdec *D = &R->dl[R->frees[f]]; ctab *T = &R->T[D->v];
                    if (T->ptr[D->ci] && T->ptr[D->ci] != T->zero) { free(T->ptr[D->ci]); R->resident -= (size_t)1 << (3 * T->sh); }
                    T->ptr[D->ci] = nullptr;
                }
            }
            pthread_cond_broadcast(&R->cv);
        }
        pthread_mutex_unlock(&R->mu);
    }
    pthread_mutex_lock(&R->mu); pthread_cond_broadcast(&R->cv); pthread_mutex_unlock(&R->mu);
    return nullptr;
}
static int wr_blk_cmp(const void *a, const void *b) {   /* by the first volume's first chunk (z, y, x), then position */
    const rblock *p = a, *q = b;
    for (int d = 0; d < 3; d++) if (p->lo[d] != q->lo[d]) return p->lo[d] < q->lo[d] ? -1 : 1;
    if (p->py != q->py) return p->py < q->py ? -1 : 1;
    return (p->px > q->px) - (p->px < q->px);
}
int render_wrap_chunked(z3 *const *vz, int nv, const int *depth, const float *X, const float *Y, const float *Z, int w, int h, double gs,
                        uint8_t *const *img, int ow, int oh, const int *crop, size_t budget, int nthreads, int io_threads, int verbose) {
    const float *const Xs[1] = {X}, *const Ys[1] = {Y}, *const Zs[1] = {Z};
    return render_wraps_chunked(vz, nv, depth, 1, Xs, Ys, Zs, w, h, gs, img, ow, oh, crop, budget, nthreads, io_threads, verbose);
}
int render_wraps_chunked(z3 *const *vz, int nv, const int *depth, int ns, const float *const *Xs, const float *const *Ys, const float *const *Zs,
                         int w, int h, double gs, uint8_t *const *img, int ow, int oh, const int *crop, size_t budget, int nthreads, int io_threads, int verbose) {
    if (nv < 1 || nv > 2 || ns < 1 || ns > 8 || w < 2 || h < 2) return -1;
    const double t0 = rnow();
    wrend *R = calloc(1, sizeof *R); if (!R) return -1;
    R->nv = nv; R->ns = ns; R->Xs = Xs; R->Ys = Ys; R->Zs = Zs; R->w = w; R->h = h; R->gs = gs; R->img = img; R->ow = ow; R->oh = oh; R->budget = budget;
    int dm = 0, rc = -1;
    for (int v = 0; v < nv; v++) {
        const z3_meta *zm = z3_meta_of(vz[v]); const int C = zm->chunk[0];
        if (C != zm->chunk[1] || C != zm->chunk[2] || (C & (C - 1))) { fprintf(stderr, "render: chunks must be cubes of a power of 2\n"); free(R); return -1; }
        int sh = 0; while ((1 << sh) < C) sh++;
        R->T[v].z = vz[v]; R->T[v].sh = sh; for (int d = 0; d < 3; d++) R->T[v].shape[d] = zm->shape[d];
        R->depth[v] = depth[v]; if (depth[v] > dm) dm = depth[v];
    }
    /* blocks with their chunk ranges per volume */
    size_t cap = 1 << 16, nb = 0; rblock *b = malloc(cap * sizeof *b); int32_t (*hi1)[3] = malloc(cap * sizeof *hi1), (*lo1)[3] = malloc(cap * sizeof *lo1);
    if (!b || !hi1 || !lo1) goto out0;
    for (int py = 0; py < oh; py += RB) for (int px = 0; px < ow; px += RB) {
        const int py1 = (py + RB < oh ? py + RB : oh) - 1, px1 = (px + RB < ow ? px + RB : ow) - 1;
        if (crop && (px1 < crop[0] || px >= crop[0] + crop[2] || py1 < crop[1] || py >= crop[1] + crop[3])) continue;
        int r0 = (int)(py / gs), r1 = (int)(py1 / gs), c0 = (int)(px / gs), c1 = (int)(px1 / gs);
        r0 = r0 > h - 2 ? h - 2 : r0; r1 = r1 > h - 2 ? h - 2 : r1; c0 = c0 > w - 2 ? w - 2 : c0; c1 = c1 > w - 2 ? w - 2 : c1;
        r1++; c1++;
        double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
        for (int si = 0; si < ns; si++) for (int r = r0; r <= r1; r++) for (int c = c0; c <= c1; c++) {   /* the union of the grids' boxes */
            const float *X = Xs[si], *Y = Ys[si], *Z = Zs[si];
            const size_t i = (size_t)r * w + c; if (!(Z[i] >= 0)) continue;
            const double v3[3] = {Z[i], Y[i], X[i]};
            for (int d = 0; d < 3; d++) { if (v3[d] < lo[d]) lo[d] = v3[d]; if (v3[d] > hi[d]) hi[d] = v3[d]; }
        }
        if (!(lo[0] <= hi[0] && lo[1] <= hi[1] && lo[2] <= hi[2])) continue;
        if (nb == cap) { cap *= 2; rblock *g = realloc(b, cap * sizeof *b); int32_t (*g1)[3] = realloc(hi1, cap * sizeof *hi1), (*g2)[3] = g1 ? realloc(lo1, cap * sizeof *lo1) : nullptr;
                         b = g ? g : b; hi1 = g1 ? g1 : hi1; lo1 = g2 ? g2 : lo1;
                         if (!g || !g1 || !g2) goto out0; }
        rblock *B = &b[nb]; B->s = (int)nb; B->px = px; B->py = py;
        for (int d = 0; d < 3; d++) {
            double a = floor(lo[d]) - dm - 2, e = ceil(hi[d]) + dm + 2; if (a < 0) a = 0; if (e < 0) e = 0;
            a = a > 1e15 ? 1e15 : a; e = e > 1e15 ? 1e15 : e;
            B->lo[d] = (int32_t)((int64_t)a >> R->T[0].sh); B->hi[d] = (int32_t)((int64_t)e >> R->T[0].sh);
            if (d < 2) { B->vlo[d] = (int32_t)a; B->vhi[d] = (int32_t)e; }
            lo1[nb][d] = nv > 1 ? (int32_t)((int64_t)a >> R->T[1].sh) : 0; hi1[nb][d] = nv > 1 ? (int32_t)((int64_t)e >> R->T[1].sh) : 0;
        }
        nb++;
    }
    /* sort (s carries the original index for the second volume's ranges) */
    qsort(b, nb, sizeof *b, wr_blk_cmp);
    R->b = b; R->nb = nb; R->bgrp = malloc((nb + 1) * sizeof *R->bgrp);
    R->blo = malloc((nb + 1) * sizeof *R->blo); R->bhi = malloc((nb + 1) * sizeof *R->bhi);
    if (!R->bgrp || !R->blo || !R->bhi) goto out0;
    for (size_t k = 0; k < nb; k++) {
        const int s = b[k].s;
        for (int d = 0; d < 3; d++) { R->blo[k][0][d] = b[k].lo[d]; R->bhi[k][0][d] = b[k].hi[d]; R->blo[k][1][d] = lo1[s][d]; R->bhi[k][1][d] = hi1[s][d]; }
    }
    free(lo1); free(hi1); lo1 = hi1 = nullptr;
    uint32_t ng = 0;
    for (size_t k = 0; k < nb; k++) { if (!k || b[k].lo[0] != b[k - 1].lo[0] || b[k].lo[1] != b[k - 1].lo[1]) ng++; R->bgrp[k] = ng; }
    R->ng = ng;
    if (nb) {
        /* chunk tables (the box over every block) and the first / last group of each chunk, in first-use order */
        size_t ntot = 0;
        for (int v = 0; v < nv; v++) {
            ctab *T = &R->T[v]; int64_t clo[3] = {INT64_MAX, INT64_MAX, INT64_MAX}, chi[3] = {-1, -1, -1};
            for (size_t k = 0; k < nb; k++) for (int d = 0; d < 3; d++) { if (R->blo[k][v][d] < clo[d]) clo[d] = R->blo[k][v][d]; if (R->bhi[k][v][d] > chi[d]) chi[d] = R->bhi[k][v][d]; }
            for (int d = 0; d < 3; d++) { T->c0[d] = clo[d]; T->cn[d] = chi[d] - clo[d] + 1; }
            const size_t nc = (size_t)(T->cn[0] * T->cn[1] * T->cn[2]); ntot += nc;
            T->ptr = calloc(nc, sizeof *T->ptr); T->zero = calloc((size_t)1 << (3 * T->sh), 1); R->first[v] = calloc(nc, sizeof(uint32_t)); R->last[v] = calloc(nc, sizeof(uint32_t));
            if (!T->ptr || !T->zero || !R->first[v] || !R->last[v]) goto out0;
            /* volcomp substreams (128^3 chunks: 16 z x 32 y slabs): only those under some block's voxel box are decoded; UFSM_RENDER_FULLDECODE=1: all */
            if (T->sh == 7 && !getenv("UFSM_RENDER_FULLDECODE") && !(R->smask[v] = calloc(nc, sizeof(uint32_t)))) goto out0;
        }
        size_t dcap = 1 << 16; R->dl = malloc(dcap * sizeof *R->dl); R->need = calloc(ng + 2, sizeof *R->need);
        if (!R->dl || !R->need) goto out0;
        for (size_t k = 0; k < nb; k++) {
            const uint32_t g = R->bgrp[k];
            for (int v = 0; v < nv; v++) {
                ctab *T = &R->T[v];
                for (int64_t a = R->blo[k][v][0]; a <= R->bhi[k][v][0]; a++) for (int64_t y = R->blo[k][v][1]; y <= R->bhi[k][v][1]; y++) for (int64_t x = R->blo[k][v][2]; x <= R->bhi[k][v][2]; x++) {
                    const size_t ci = (size_t)(((a - T->c0[0]) * T->cn[1] + (y - T->c0[1])) * T->cn[2] + (x - T->c0[2]));
                    if (!R->first[v][ci]) {
                        if (R->ndl == dcap) { dcap *= 2; wdec *q = realloc(R->dl, dcap * sizeof *q); if (!q) goto out0; R->dl = q; }
                        R->first[v][ci] = g; R->dl[R->ndl++] = (wdec){(uint8_t)v, (int32_t)ci, g};
                    }
                    R->last[v][ci] = g;
                    if (R->smask[v]) {   /* the block's z / y voxel range inside this chunk -> substream bits */
                        const int64_t z0 = a << 7, y0 = y << 7;
                        int64_t za = R->b[k].vlo[0] - z0, ze = R->b[k].vhi[0] - z0, ya = R->b[k].vlo[1] - y0, ye = R->b[k].vhi[1] - y0;
                        za = za < 0 ? 0 : za; ze = ze > 127 ? 127 : ze; ya = ya < 0 ? 0 : ya; ye = ye > 127 ? 127 : ye;
                        uint32_t m = 0; for (int64_t bz = za >> 4; bz <= ze >> 4; bz++) for (int64_t by = ya >> 5; by <= ye >> 5; by++) m |= 1u << (bz * 4 + by);
                        R->smask[v][ci] |= m;
                    }
                }
            }
            R->need[g] = R->ndl;
        }
        for (uint32_t g = 1; g <= ng; g++) if (R->need[g] < R->need[g - 1]) R->need[g] = R->need[g - 1];
        /* frees per group (counting sort of the entries by their last group) */
        R->free0 = calloc(ng + 3, sizeof *R->free0); R->frees = malloc((R->ndl + 1) * sizeof *R->frees); R->ready = calloc(R->ndl + 1, 1); R->gleft = calloc(ng + 2, sizeof *R->gleft);
        if (!R->free0 || !R->frees || !R->ready || !R->gleft) goto out0;
        for (size_t e = 0; e < R->ndl; e++) R->free0[R->last[R->dl[e].v][R->dl[e].ci] + 1]++;
        for (uint32_t g = 1; g <= ng + 1; g++) R->free0[g + 1] += R->free0[g];
        { size_t *fill = calloc(ng + 2, sizeof *fill); if (!fill) goto out0;
          for (size_t e = 0; e < R->ndl; e++) { const uint32_t L = R->last[R->dl[e].v][R->dl[e].ci]; R->frees[R->free0[L] + fill[L]++] = e; }
          free(fill); }
        for (size_t k = 0; k < nb; k++) R->gleft[R->bgrp[k]]++;
        pthread_mutex_init(&R->mu, nullptr); pthread_cond_init(&R->cv, nullptr);
        const int nr = nthreads < 1 ? 1 : nthreads > 64 ? 64 : nthreads, nd = io_threads < 1 ? nr : io_threads > 128 ? 128 : io_threads;
        pthread_t th[192];
        for (int t = 0; t < nd; t++) pthread_create(&th[t], nullptr, wr_decoder, R);
        for (int t = 0; t < nr; t++) pthread_create(&th[nd + t], nullptr, wr_renderer, R);
        for (int t = 0; t < nd + nr; t++) pthread_join(th[t], nullptr);
        pthread_mutex_destroy(&R->mu); pthread_cond_destroy(&R->cv);
        rc = atomic_load(&R->fail) ? -1 : 0;
        for (int v = 0; v < nv; v++) { ctab *T = &R->T[v]; const size_t nc = (size_t)(T->cn[0] * T->cn[1] * T->cn[2]); for (size_t i = 0; i < nc; i++) if (T->ptr[i] && T->ptr[i] != T->zero) free(T->ptr[i]); }
        if (verbose) fprintf(stderr, "render: %zu blocks in %u chunk rows, %zu chunks decoded once (%llu empty), peak chunk memory %.2f GB, %.1fs\n",
                             nb, ng, R->ndl, (unsigned long long)R->nzero, R->peak / 1073741824.0, rnow() - t0);
    } else rc = 0;
out0:
    free(lo1); free(hi1);
    for (int v = 0; v < 2; v++) { free(R->T[v].ptr); free(R->T[v].zero); free(R->first[v]); free(R->last[v]); free(R->smask[v]); }
    free(b); free(R->bgrp); free(R->blo); free(R->bhi); free(R->dl); free(R->need); free(R->free0); free(R->frees); free(R->ready); free(R->gleft);
    free(R);
    return rc;
}

static int name_cmp(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }
static int area_cmp(const void *a, const void *b) { const rsheet *p = a, *q = b; return p->area > q->area ? -1 : p->area < q->area; }
static int sname_cmp(const void *a, const void *b) { return strcmp(((const rsheet *)a)->name, ((const rsheet *)b)->name); }
static int parse_ints(const char *s, int *v, int max) { int n = 0; while (n < max && *s) { v[n++] = atoi(s); s = strchr(s, ','); if (!s) break; s++; } return n; }
static void jpeg_name(char *out, size_t n, const char *dir, const char *prefix, const char *name, int strip_w) {
    snprintf(out, n, "%s/%s%s.jpg", dir, prefix, strip_w && name[0] == 'w' ? name + 1 : name);
}
static void sheet_free_pass(rsheet *s, int k) {
    free(s->X[k]); free(s->Y[k]); free(s->Z[k]); s->X[k] = s->Y[k] = s->Z[k] = nullptr;
    for (int q = 0; q < 4; q++) { free(s->img[k][q]); s->img[k][q] = nullptr; }
}

/* ufsm render-sheets <tifxyz-dir> <out-dir> --ct ROOT KEY --ink ROOT KEY [--ct-div 2] [--ink-div 4] [--ink-tifxyz DIR]
   [--ct-depth 3] [--ink-depths 5,3,1] [--wraps a,b,.. | --top N] [--min-cm2 0] [--prefix P] [--strip-w] [--skip-existing]
   [--keep-tif DIR] [--threads 12] [--mem 4] [--cache DIR]
   The sheets of <tifxyz-dir> (2.4 um wraps; --ct-div / --ink-div: their coordinates divided as xform_wraps.py did, e.g. 2 -> the
   4.8 um CT level, 4 -> the 9.6 um ink; div 1 with --ink-tifxyz: copies already on each grid) rendered as ufsm render would
   (--scale 1) and composited as overlay_red.py: CT | CT + red ink | one ink panel per --ink-depths, half-resolution JPEG q90
   <out-dir>/<prefix><sheet>.jpg (none when under 5% of the CT pixels have data). --keep-tif DIR also writes the renders
   (DIR/ct/<sheet>.tif, DIR/ink<depth>/<sheet>.tif). Sheets go in batches that fit --mem GB (renders + grids + decoded chunks). */
int cmd_render_sheets(int argc, char **argv) {
    if (argc < 4 || rarg(argc, argv, "--ct") < 0 || rarg(argc, argv, "--ink") < 0) {
        fprintf(stderr, "usage: ufsm render-sheets <tifxyz-dir> <out-dir> --ct ROOT KEY --ink ROOT KEY [--ct-div 2] [--ink-div 4] [--ink-tifxyz DIR] [--ct-depth 3]\n"
                        "         [--ink-depths 5,3,1] [--wraps a,b | --top N] [--min-cm2 0] [--prefix P] [--strip-w] [--skip-existing] [--keep-tif DIR]\n"
                        "         [--threads 12] [--mem 4] [--cache DIR]\n"); return 2;
    }
    const char *src = argv[2], *out = argv[3], *isrc = ropt(argc, argv, "--ink-tifxyz", src), *prefix = ropt(argc, argv, "--prefix", ""), *keep = ropt(argc, argv, "--keep-tif", nullptr);
    const double cdiv = atof(ropt(argc, argv, "--ct-div", "2")), idiv = atof(ropt(argc, argv, "--ink-div", "4")), mem = atof(ropt(argc, argv, "--mem", "4"));
    int cdepth = atoi(ropt(argc, argv, "--ct-depth", "3")), idepth[3]; const int nink = parse_ints(ropt(argc, argv, "--ink-depths", "5,3,1"), idepth, 3);
    const int nthreads = atoi(ropt(argc, argv, "--threads", "12")), top = atoi(ropt(argc, argv, "--top", "0")), strip_w = rflag(argc, argv, "--strip-w"), skip = rflag(argc, argv, "--skip-existing");
    const double min_cm2 = atof(ropt(argc, argv, "--min-cm2", "0"));
    const int ci = rarg(argc, argv, "--ct"), ii = rarg(argc, argv, "--ink"); const char *cache = ropt(argc, argv, "--cache", nullptr);
    if (ci + 2 >= argc || ii + 2 >= argc || nink < 1 || cdiv <= 0 || idiv <= 0 || cdepth < 0) { fprintf(stderr, "render-sheets: bad options\n"); return 2; }
    z3 *cz = z3_open(store_open(argv[ci + 1]), argv[ci + 2], cache), *iz = z3_open(store_open(argv[ii + 1]), argv[ii + 2], cache);
    if (!cz || !iz) { fprintf(stderr, "render-sheets: %s\n", z3_error()); return 1; }
    char p[1400]; snprintf(p, sizeof p, "mkdir -p '%s'", out); if (system(p)) return 1;
    /* the sheets */
    char *want = nullptr; const char *wl = ropt(argc, argv, "--wraps", nullptr); if (wl) { want = malloc(strlen(wl) + 3); sprintf(want, ",%s,", wl); }
    DIR *dd = opendir(src); if (!dd) { fprintf(stderr, "render-sheets: cannot open %s\n", src); return 1; }
    int ns = 0, cs = 64; rsheet *S = calloc(cs, sizeof *S); struct dirent *de;
    while ((de = readdir(dd))) {
        if (de->d_name[0] == '.' || strlen(de->d_name) >= sizeof S->name) continue;
        if (want) { char k[300]; snprintf(k, sizeof k, ",%s,", de->d_name); if (!strstr(want, k)) continue; }
        double sc, area; snprintf(p, sizeof p, "%s/%s", src, de->d_name); if (meta_scale_area(p, &sc, &area)) continue;
        if (area < min_cm2) continue;
        if (ns == cs) { cs *= 2; S = realloc(S, cs * sizeof *S); memset(S + ns, 0, (cs - ns) * sizeof *S); }
        snprintf(S[ns].name, sizeof S[ns].name, "%s", de->d_name); S[ns].area = area; ns++;
    }
    closedir(dd); free(want);
    if (top > 0) { qsort(S, ns, sizeof *S, area_cmp); if (ns > top) ns = top; }
    qsort(S, ns, sizeof *S, sname_cmp);
    if (skip) { int k = 0; for (int s = 0; s < ns; s++) { jpeg_name(p, sizeof p, out, prefix, S[s].name, strip_w); struct stat st; if (stat(p, &st)) S[k++] = S[s]; } ns = k; }
    if (keep) { snprintf(p, sizeof p, "mkdir -p '%s/ct'", keep); if (system(p)) return 1; for (int q = 0; q < nink; q++) { snprintf(p, sizeof p, "mkdir -p '%s/ink%d'", keep, idepth[q]); if (system(p)) return 1; } }
    /* batches: grids (both passes) + renders within ~60% of --mem, the decoded chunks within the rest */
    const size_t total = (size_t)(mem * (1u << 30)), cbud = total * 2 / 5, bbud = total - cbud;
    const double t0 = rnow(); int nout = 0, nskip = 0, nfail = 0;
    for (int s0 = 0; s0 < ns;) {
        int s1 = s0; size_t used = 0;
        while (s1 < ns) {
            if (sheet_load(&S[s1], 0, src, cdiv) || sheet_load(&S[s1], 1, isrc, idiv)) { fprintf(stderr, "render-sheets: %s: cannot read its tifxyz\n", S[s1].name); sheet_free_pass(&S[s1], 0); sheet_free_pass(&S[s1], 1); S[s1].ow[0] = -1; s1++; nfail++; continue; }
            const size_t need = (size_t)S[s1].w[0] * S[s1].h[0] * 12 + (size_t)S[s1].w[1] * S[s1].h[1] * 12 + (size_t)S[s1].ow[0] * S[s1].oh[0] + (size_t)nink * S[s1].ow[1] * S[s1].oh[1];
            used += need; s1++;
            if (used > bbud) break;
        }
        int nb = 0; rsheet *B = malloc((size_t)(s1 - s0) * sizeof *B); int *map = malloc((size_t)(s1 - s0) * sizeof *map);
        for (int s = s0; s < s1; s++) if (S[s].ow[0] >= 0) { map[nb] = s; B[nb++] = S[s]; }
        if (render_pass(cz, B, nb, 0, &cdepth, 1, cbud, nthreads, 1) || render_pass(iz, B, nb, 1, idepth, nink, cbud, nthreads, 1)) { fprintf(stderr, "render-sheets: render failed\n"); return 1; }
        const double to = rnow();
        for (int s = 0; s < nb; s++) {
            rsheet *R = &B[s];
            for (int k = 0; k < 2; k++) { free(R->X[k]); free(R->Y[k]); free(R->Z[k]); R->X[k] = R->Y[k] = R->Z[k] = nullptr; }
            if (keep) {
                snprintf(p, sizeof p, "%s/ct/%s.tif", keep, R->name); if (tiffw_write(p, R->ow[0], R->oh[0], 1, R->img[0][0], 1)) fprintf(stderr, "render-sheets: cannot write %s\n", p);
                for (int q = 0; q < nink; q++) { snprintf(p, sizeof p, "%s/ink%d/%s.tif", keep, idepth[q], R->name); if (tiffw_write(p, R->ow[1], R->oh[1], 1, R->img[1][q], 1)) fprintf(stderr, "render-sheets: cannot write %s\n", p); }
            }
            if (overlay_red_skip(R->img[0][0], (size_t)R->ow[0] * R->oh[0])) nskip++;
            else {
                int ow, oh; const uint8_t *ink[3] = {R->img[1][0], R->img[1][1], R->img[1][2]};
                uint8_t *rgb = overlay_red_half(R->img[0][0], R->ow[0], R->oh[0], ink, nink, R->ow[1], R->oh[1], &ow, &oh, nthreads);
                jpeg_name(p, sizeof p, out, prefix, R->name, strip_w);
                if (!rgb || jpeg_write_rgb(p, ow, oh, rgb, 90)) { fprintf(stderr, "render-sheets: cannot write %s\n", p); nfail++; } else nout++;
                free(rgb);
            }
            sheet_free_pass(R, 0); sheet_free_pass(R, 1);
        }
        fprintf(stderr, "render-sheets: overlays of %d sheets %.1fs\n", nb, rnow() - to);
        free(B); free(map);
        s0 = s1;
    }
    fprintf(stderr, "render-sheets: %d overlays (%d sheets under 5%% CT, %d failed) of %d sheets in %s, %.1fs\n", nout, nskip, nfail, ns, out, rnow() - t0);
    free(S); z3_close(cz); z3_close(iz);
    return nfail ? 1 : 0;
}

/* ufsm overlay-red <ct-dir> <ink-dir> <out-dir> [<ink2-dir> [<ink3-dir>]] [--strip-w] [--skip-existing] [--threads 12]
   overlay_red.py / overlay_stitched.py: per <name>.tif of ct-dir (not *_ct.tif) the overlay of the same-named ink renders ->
   <out-dir>/<name>.jpg (--strip-w: a leading 'w' dropped); none when under 5% of the CT pixels have data. */
int cmd_overlay_red(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm overlay-red <ct-dir> <ink-dir> <out-dir> [<ink2-dir> [<ink3-dir>]] [--strip-w] [--skip-existing] [--threads 12]\n"); return 2; }
    const char *dirs[3] = {argv[3], nullptr, nullptr}; int nink = 1;
    for (int a = 5; a < argc && nink < 3; a++) { if (argv[a][0] == '-') { if (!strcmp(argv[a], "--threads")) a++; continue; } dirs[nink++] = argv[a]; }
    const int strip_w = rflag(argc, argv, "--strip-w"), skip = rflag(argc, argv, "--skip-existing"), nthreads = atoi(ropt(argc, argv, "--threads", "12"));
    char p[1400]; snprintf(p, sizeof p, "mkdir -p '%s'", argv[4]); if (system(p)) return 1;
    DIR *dd = opendir(argv[2]); if (!dd) { fprintf(stderr, "overlay-red: cannot open %s\n", argv[2]); return 1; }
    int n = 0, cap = 64; char **names = malloc(cap * sizeof *names); struct dirent *de;
    while ((de = readdir(dd))) {
        const size_t l = strlen(de->d_name);
        if (l < 5 || strcmp(de->d_name + l - 4, ".tif") || (l >= 7 && !strcmp(de->d_name + l - 7, "_ct.tif"))) continue;
        if (n == cap) { cap *= 2; names = realloc(names, cap * sizeof *names); }
        names[n++] = strndup(de->d_name, l - 4);
    }
    closedir(dd); qsort(names, n, sizeof *names, name_cmp);
    int nout = 0, rc = 0;
    for (int f = 0; f < n; f++) {
        char jp[1400]; jpeg_name(jp, sizeof jp, argv[4], "", names[f], strip_w); struct stat st;
        if (skip && !stat(jp, &st)) continue;
        int W, H, iw = 0, ih = 0; snprintf(p, sizeof p, "%s/%s.tif", argv[2], names[f]); uint8_t *ct = img_read_u8(p, &W, &H), *ink[3] = {0};
        if (!ct) { fprintf(stderr, "overlay-red: cannot read %s\n", p); rc = 1; continue; }
        if (overlay_red_skip(ct, (size_t)W * H)) { free(ct); continue; }
        int ok = 1;
        for (int k = 0; k < nink; k++) {
            int w2, h2; snprintf(p, sizeof p, "%s/%s.tif", dirs[k], names[f]); ink[k] = img_read_u8(p, &w2, &h2);
            if (!ink[k] || (k && (w2 != iw || h2 != ih))) { fprintf(stderr, "overlay-red: cannot read %s\n", p); ok = 0; break; }
            iw = w2; ih = h2;
        }
        if (ok) {
            int ow, oh; uint8_t *rgb = overlay_red_half(ct, W, H, (const uint8_t *const *)ink, nink, iw, ih, &ow, &oh, nthreads);
            if (!rgb || jpeg_write_rgb(jp, ow, oh, rgb, 90)) { fprintf(stderr, "overlay-red: cannot write %s\n", jp); rc = 1; } else { nout++; if (strip_w) printf("%s\n", jp); }
            free(rgb);
        } else rc = 1;
        free(ct); for (int k = 0; k < 3; k++) free(ink[k]);
    }
    for (int f = 0; f < n; f++) free(names[f]);
    free(names);
    fprintf(stderr, "overlay-red: %d overlays in %s\n", nout, argv[4]);
    return rc;
}

/* ufsm tifxyz-scale <src-dir> <dst-dir> <div> [--offset z,y,x] [--wraps a,b,..] [--normal D [--toward Y,X]]
   xform_wraps.py: the tifxyz <src-dir>/<w> (every subdirectory, or --wraps) copied to <dst-dir>/<w> with coordinates divided
   by div (float32) minus the offset; invalid vertices (z < 0) stay -1; meta scale x div, voxel_size_um x div. --normal D (div 1, no
   offset): instead every vertex moved D voxels along its normal +n = (d/d column) x (d/d row) (tifxyz_shift; one face of a sheet);
   --toward Y,X (the scroll axis at the sheet, native voxels): +n oriented inward first (D > 0 = the inward face). --flip-rows 1: the grid
   rows written in reverse order (Paris 4 wraps of ufsm wraps / flatten render mirrored in z; flipped they read upright). */
int cmd_tifxyz_scale(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm tifxyz-scale <src-dir> <dst-dir> <div> [--offset z,y,x] [--wraps a,b,..] [--normal D [--toward Y,X]] [--flip-rows 1]\n"); return 2; }
    tifxyz_flip_rows = atoi(ropt(argc, argv, "--flip-rows", "0")) != 0;
    const double div = atof(argv[4]); double off[3] = {0, 0, 0};
    if (!(div > 0) || sscanf(ropt(argc, argv, "--offset", "0,0,0"), "%lf,%lf,%lf", off, off + 1, off + 2) != 3) { fprintf(stderr, "tifxyz-scale: bad div / --offset\n"); return 2; }
    const char *wl = ropt(argc, argv, "--wraps", nullptr); char *want = nullptr; if (wl) { want = malloc(strlen(wl) + 3); sprintf(want, ",%s,", wl); }
    DIR *dd = opendir(argv[2]); if (!dd) { fprintf(stderr, "tifxyz-scale: cannot open %s\n", argv[2]); return 1; }
    struct dirent *de; int n = 0, rc = 0;
    while ((de = readdir(dd))) {
        if (de->d_name[0] == '.') continue;
        if (want) { char k[300]; snprintf(k, sizeof k, ",%s,", de->d_name); const size_t kl = strlen(k); if (kl > 5 && !strcmp(k + kl - 5, ".sfc,")) strcpy(k + kl - 5, ","); if (!strstr(want, k)) continue; }
        char s[1400], d[1400], m[1500]; snprintf(s, sizeof s, "%s/%s", argv[2], de->d_name); snprintf(m, sizeof m, "%s/meta.json", s);
        if (!tfx_exists(m)) continue;   /* (a tifxyz dir or a .sfc wrap) */
        snprintf(d, sizeof d, "%s/%s", argv[3], de->d_name); { const size_t dl = strlen(d); if (dl > 4 && !strcmp(d + dl - 4, ".sfc")) d[dl - 4] = 0; }   /* (w.sfc -> w) */
        const char *nd = ropt(argc, argv, "--normal", nullptr);
        double tw[2]; const int htw = sscanf(ropt(argc, argv, "--toward", ""), "%lf,%lf", tw, tw + 1) == 2;
        if (nd ? tifxyz_shift(s, d, atof(nd), htw ? tw : nullptr) : tifxyz_scale(s, d, div, off)) { fprintf(stderr, "tifxyz-scale: %s failed\n", s); rc = 1; } else n++;
    }
    closedir(dd); free(want);
    fprintf(stderr, "tifxyz-scale: %d tifxyz written to %s\n", n, argv[3]);
    return rc;
}

/* ufsm png2jpg <in.png|in.jpg> <out.jpg> [--half] [--max-px N] [--quality 90]: a PNG (8-bit) or a JPEG (by its SOI bytes) as JPEG,
   --half Pillow-BOX halved, --max-px N: the longest side at most N (img_fit_rgb; <out> may be <in>) (png2jpg_half.py,
   sync_big.sh's PNG -> JPEG copies) */
/* ufsm tifxyz-sfc <src-dir> <dst-dir> [--error 0.25] [--every K] [--wraps a,b,..]: every tifxyz wrap of <src-dir> (directories or
   .sfc) as <dst-dir>/<w>.sfc (surface-compressor, joint XYZ within --error voxels, meta.json inside), keeping every K-th vertex in both
   grid directions (K 4 turns a scale-1 wrap into scale 0.25; meta rewritten: scale, area, bbox). Prints the bytes before / after. */
int cmd_tifxyz_sfc(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm tifxyz-sfc <src-dir> <dst-dir> [--error 0.25] [--every K] [--wraps a,b,..]\n"); return 2; }
    const double err = atof(ropt(argc, argv, "--error", "0.25")); const int K = atoi(ropt(argc, argv, "--every", "1"));
    if (!(err > 0) || K < 1 || !tfx_write_hook) { fprintf(stderr, "tifxyz-sfc: --error > 0, --every >= 1 (and a binary with src/tfx.c)\n"); return 2; }
    const char *wl = ropt(argc, argv, "--wraps", nullptr); char *want = nullptr; if (wl) { want = malloc(strlen(wl) + 3); sprintf(want, ",%s,", wl); }
    DIR *dd = opendir(argv[2]); if (!dd) { fprintf(stderr, "tifxyz-sfc: cannot open %s\n", argv[2]); return 1; }
    tifxyz_set_sfc(err);
    struct dirent *de; int n = 0, rc = 0; long long before = 0, after = 0;
    while ((de = readdir(dd))) {
        if (de->d_name[0] == '.') continue;
        char nm[300]; snprintf(nm, sizeof nm, "%s", de->d_name); { const size_t l = strlen(nm); if (l > 4 && !strcmp(nm + l - 4, ".sfc")) nm[l - 4] = 0; }
        if (want) { char k[310]; snprintf(k, sizeof k, ",%s,", nm); if (!strstr(want, k)) continue; }
        char s[1400], p[1500]; snprintf(s, sizeof s, "%s/%s", argv[2], de->d_name); snprintf(p, sizeof p, "%s/meta.json", s);
        if (!tfx_exists(p)) continue;
        FILE *f = tfx_meta_fopen(p); char mb[16384] = {0}; if (f) { const size_t mr = fread(mb, 1, sizeof mb - 1, f); (void)mr; fclose(f); }
        const char *q = strstr(mb, "\"scale\": ["); const double sc = q ? atof(q + 10) : 1; q = strstr(mb, "\"voxel_size_um\":"); const double vum = q ? atof(q + 16) : 1;
        q = strstr(mb, "\"winding\":"); const int wid = q ? atoi(q + 10) : -1;
        static const char *ax[3] = {"x", "y", "z"}; float *A[3] = {0}; int w = 0, h = 0, ok = 1;
        for (int k = 0; k < 3 && ok; k++) { int w2, h2; snprintf(p, sizeof p, "%s/%s.tif", s, ax[k]); A[k] = img_read_f32(p, &w2, &h2); ok = A[k] && (!k || (w2 == w && h2 == h)); w = w2; h = h2; }
        if (ok) {
            const int ow = (w + K - 1) / K, oh = (h + K - 1) / K; float *O[3];
            for (int k = 0; k < 3; k++) { O[k] = malloc((size_t)ow * oh * 4); for (int r = 0; r < oh; r++) for (int c = 0; c < ow; c++) O[k][(size_t)r * ow + c] = A[k][(size_t)r * K * w + (size_t)c * K]; }
            snprintf(p, sizeof p, "%s/%s", argv[3], nm);
            if (tifxyz_write(p, ow, oh, O[0], O[1], O[2], K / (sc > 0 ? sc : 1), vum, wid, "ufsm tifxyz-sfc")) { fprintf(stderr, "tifxyz-sfc: cannot write %s.sfc\n", p); rc = 1; }
            else {
                struct stat st; char sp[1500]; snprintf(sp, sizeof sp, "%s.sfc", p); if (!stat(sp, &st)) after += st.st_size;
                { float *B[3]; int bw = 0, bh = 0; double em = 0; long long bad = 0; static const char *bx[3] = {"x", "y", "z"};   /* read back: the error budget */
                  for (int k = 0; k < 3; k++) { char bp[1600]; snprintf(bp, sizeof bp, "%s/%s.tif", sp, bx[k]); B[k] = img_read_f32(bp, &bw, &bh); }
                  if (B[0] && B[1] && B[2] && bw == ow && bh == oh) for (size_t i = 0; i < (size_t)ow * oh; i++) {
                      if (!(O[2][i] > 0)) { bad += B[2][i] != -1.f; continue; }
                      const double e = sqrt((double)(B[0][i] - O[0][i]) * (B[0][i] - O[0][i]) + (double)(B[1][i] - O[1][i]) * (B[1][i] - O[1][i]) + (double)(B[2][i] - O[2][i]) * (B[2][i] - O[2][i]));
                      if (e > em) em = e; }
                  else bad = -1;
                  printf("%s: %dx%d, max |xyz error| %.4f voxels (budget %g)%s\n", nm, ow, oh, em, err, bad ? " READBACK MISMATCH" : "");
                  if (bad || em > err * 1.0001) rc = 1;
                  for (int k = 0; k < 3; k++) free(B[k]); }
                if (!stat(s, &st) && S_ISREG(st.st_mode)) before += st.st_size;
                else for (int k = 0; k < 4; k++) { snprintf(sp, sizeof sp, "%s/%s", s, k < 3 ? (const char *[]){"x.tif", "y.tif", "z.tif"}[k] : "meta.json"); if (!stat(sp, &st)) before += st.st_size; }
                n++;
            }
            for (int k = 0; k < 3; k++) free(O[k]);
        } else { fprintf(stderr, "tifxyz-sfc: cannot read %s\n", s); rc = 1; }
        for (int k = 0; k < 3; k++) free(A[k]);
    }
    closedir(dd); free(want); tifxyz_set_sfc(0);
    fprintf(stderr, "tifxyz-sfc: %d wraps -> %s: %lld -> %lld bytes (%.1fx)\n", n, argv[3], before, after, after ? (double)before / after : 0);
    return rc;
}
int cmd_png2jpg(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm png2jpg <in.png|in.jpg> <out.jpg> [--half] [--max-px N] [--quality 90]\n"); return 2; }
    int is_jpg = 0; { FILE *f = fopen(argv[2], "rb"); uint8_t m[2] = {0}; if (f) { is_jpg = fread(m, 1, 2, f) == 2 && m[0] == 0xff && m[1] == 0xd8; fclose(f); } }
    int w, h; uint8_t *rgb = is_jpg ? jpeg_read_rgb(argv[2], &w, &h) : png_read_rgb(argv[2], &w, &h); if (!rgb) { fprintf(stderr, "png2jpg: cannot read %s\n", argv[2]); return 1; }
    const int q = atoi(ropt(argc, argv, "--quality", "90"));
    const size_t ol = strlen(argv[3]);
    if (ol > 4 && !strcmp(argv[3] + ol - 4, ".tif")) {   /* <out>.tif: the first channel as a uint8 TIFF (gray PNGs for ink-compare); --crop x,y,w,h */
        int c[4] = {0, 0, w, h}; const char *cr = ropt(argc, argv, "--crop", nullptr);
        if (cr && (sscanf(cr, "%d,%d,%d,%d", c, c + 1, c + 2, c + 3) != 4 || c[0] < 0 || c[1] < 0 || c[0] + c[2] > w || c[1] + c[3] > h)) { free(rgb); return 2; }
        uint8_t *g = malloc((size_t)c[2] * c[3]);
        for (int y = 0; y < c[3]; y++) for (int x = 0; x < c[2]; x++) g[(size_t)y * c[2] + x] = rgb[(((size_t)(c[1] + y)) * w + c[0] + x) * 3];
        const int rc = tiffw_write(argv[3], c[2], c[3], 1, g, 1); free(g); free(rgb);
        return rc ? 1 : 0;
    }
    if (rflag(argc, argv, "--half")) {
        const int ow = w / 2, oh = h / 2; uint8_t *o = ow > 0 && oh > 0 ? malloc((size_t)ow * oh * 3) : nullptr;
        if (!o || pil_resize(rgb, w, h, 3, o, ow, oh, PIL_BOX, 8)) { free(o); free(rgb); return 1; }
        free(rgb); rgb = o; w = ow; h = oh;
    }
    if (img_fit_rgb(&rgb, &w, &h, atoi(ropt(argc, argv, "--max-px", "0")), 8)) { free(rgb); fprintf(stderr, "png2jpg: out of memory\n"); return 1; }
    const int rc = jpeg_write_rgb(argv[3], w, h, rgb, q); free(rgb);
    if (rc) fprintf(stderr, "png2jpg: cannot write %s\n", argv[3]);
    return rc ? 1 : 0;
}

/* ufsm phase-score <pc-dir> <ps-dir> [<sheets-dir>]: the former phase_score.py (phase_eval.sh / phase_eval2.sh): per render pair
   <pc-dir>/X.tif, <ps-dir>/X.tif (ufsm render of the fit-field arrays pc / ps at depth 0) the on-recto score (1 = every vertex
   on a recto face, 0 = unrelated, -1 = on the gaps / verso side; imgops.h phase_score_surface), with <sheets-dir> (stitched
   sheets with seam.tif, render copies named w<sheet>_rR_cC) the same per seam class; then the totals. Analysis only. */
int cmd_phase_score(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm phase-score <pc-dir> <ps-dir> [<sheets-dir>]\n"); return 2; }
    if (phase_score_dirs(argv[2], argv[3], argc > 4 ? argv[4] : nullptr, stdout)) { fprintf(stderr, "phase-score: cannot open %s\n", argv[2]); return 1; }
    return 0;
}

/* ufsm ink-compare <out.jpg> --ct CT.tif --ink LABEL=INK.tif [--ink LABEL=INK.tif ..] [--px-um 2.4] [--rs-um 25] [--rl-um 500]
   [--crop x,y,w,h] [--threads 16]: renders of one surface on one pixel grid (ufsm render: <wrap>_ct.tif and the ink <wrap>.tif of each ink volume)
   -> CT | ink 1 | ink 2 .. half-resolution JPEG (q90; none when under 5% of the CT pixels have data), and per ink over the CT > 0
   pixels one line "ink LABEL: n .. mean .. frac50 .. p50 .. p95 .. p99 .. band .." (ink_stats: band = the contrast between the
   --rs-um and --rl-um box scales) and "corr A B: r" for each pair. Exit 0, 3 when skipped (too little CT).
   --best-crop N: the crop is the N x N window with the most first-ink signal (img_best_window; printed as "crop x,y,w,h");
   --stretch: each ink panel p1 .. p99 contrast stretched over CT > 0 (display only); --full: full-resolution panels;
   --stretch-ct: the CT panel p1 .. p99 stretched as well (img_stretch); --fixed LO,HI: every ink panel on one fixed scale (bytes LO .. HI
   -> 0 .. 255), the same for all sheets, instead of --stretch; --each PREFIX: instead of the combined JPEG every panel alone, unlabelled,
   as PREFIX<label><--each-suffix>.jpg (the CT panel only with --each-ct; <out.jpg> unused); --max-px N: every JPEG's longest side at most N
   (antialiased, img_fit_rgb); --quality Q (90). */
int ink_compare_mem(int argc, char **argv);
int ink_compare_mem(int argc, char **argv) {   /* the in-memory version (inkcompare.c streams; same bytes) */
    const int ci = rarg(argc, argv, "--ct");
    if (argc < 5 || ci < 0 || ci + 1 >= argc || rarg(argc, argv, "--ink") < 0) {
        fprintf(stderr, "usage: ufsm ink-compare <out.jpg> --ct CT.tif --ink LABEL=INK.tif [--ink LABEL=INK.tif ..] [--px-um 2.4] [--rs-um 25] [--rl-um 500] [--crop x,y,w,h | --crop-frac fx,fy,w,h | --best-crop N] [--stretch | --fixed LO,HI] [--stretch-ct] [--full] [--each PREFIX [--each-suffix S] [--each-ct]] [--rgb-agree RECTO,VERSO] [--diff-agree RECTO,VERSO [--diff-gain 1]] [--papyrus INK [--papyrus-gain 1] [--papyrus-floor median|V]] [--cmap viridis] [--no-ct] [--rgb-ct-min 1] [--max-px N] [--quality 90] [--threads 16]\n"); return 2;
    }
    const double px = atof(ropt(argc, argv, "--px-um", "2.4")), rsu = atof(ropt(argc, argv, "--rs-um", "25")), rlu = atof(ropt(argc, argv, "--rl-um", "500"));
    const int nthreads = atoi(ropt(argc, argv, "--threads", "16"));
    if (!(px > 0) || !(rsu > 0) || !(rlu > rsu)) { fprintf(stderr, "ink-compare: need --px-um > 0 and 0 < --rs-um < --rl-um\n"); return 2; }
    enum { MAXI = 8 }; const uint8_t *img[MAXI + 1] = {0}; int isct[MAXI + 1] = {1}; const char *lab[MAXI + 1] = {"ct"}; int n = 1, W, H, rc = 0;
    uint8_t *ct = img_read_u8(argv[ci + 1], &W, &H);
    if (!ct) { fprintf(stderr, "ink-compare: cannot read %s\n", argv[ci + 1]); return 1; }
    img[0] = ct;
    for (int a = 2; a + 1 < argc; a++) {
        if (strcmp(argv[a], "--ink")) continue;
        const char *eq = strchr(argv[a + 1], '=');
        if (!eq || n > MAXI) { fprintf(stderr, "ink-compare: --ink LABEL=PATH (at most %d)\n", MAXI); return 2; }
        int w, h; uint8_t *im = img_read_u8(eq + 1, &w, &h);
        if (!im || w != W || h != H) { fprintf(stderr, "ink-compare: cannot read %s as a %d x %d render\n", eq + 1, W, H); free(im); return 1; }
        lab[n] = strndup(argv[a + 1], (size_t)(eq - argv[a + 1])); isct[n] = 0; img[n++] = im;
    }
    const char *cr = ropt(argc, argv, "--crop", nullptr), *cf = ropt(argc, argv, "--crop-frac", nullptr);
    char crb[96];
    if (!cr && cf) {   /* --crop-frac fx,fy,w,h: the crop's corner at fractions of the render */
        double fx, fy; int cw0, ch0;
        if (sscanf(cf, "%lf,%lf,%d,%d", &fx, &fy, &cw0, &ch0) != 4) { fprintf(stderr, "ink-compare: --crop-frac fx,fy,w,h\n"); return 2; }
        snprintf(crb, sizeof crb, "%d,%d,%d,%d", (int)(fx * W), (int)(fy * H), cw0, ch0); cr = crb;
    }
    const int bwin = atoi(ropt(argc, argv, "--best-crop", "0"));
    if (!cr && bwin > 0 && n > 1) {   /* --best-crop N: the N x N window richest in the first ink (over CT > 0), as --crop */
        uint8_t *mk = malloc((size_t)W * H); if (!mk) return 1;
        for (size_t i = 0; i < (size_t)W * H; i++) mk[i] = ct[i] > 0;
        int bx, by; const int r = img_best_window(img[1], mk, W, H, bwin, &bx, &by); free(mk);
        if (r) { fprintf(stderr, "ink-compare: no %d px window at least half on the surface\n", bwin); return 3; }
        snprintf(crb, sizeof crb, "%d,%d,%d,%d", bx, by, bwin, bwin); cr = crb; printf("crop %s\n", crb);
    }
    if (cr) {   /* --crop x,y,w,h: every render cut to the rectangle (clipped to the image) before the statistics and the panels */
        int cx, cy, cw, ch;
        if (sscanf(cr, "%d,%d,%d,%d", &cx, &cy, &cw, &ch) != 4 || cx < 0 || cy < 0 || cw < 2 || ch < 2 || cx >= W || cy >= H) { fprintf(stderr, "ink-compare: --crop x,y,w,h inside the %d x %d render\n", W, H); return 2; }
        if (cx + cw > W) cw = W - cx;
        if (cy + ch > H) ch = H - cy;
        for (int k = 0; k < n; k++) {
            uint8_t *c = malloc((size_t)cw * ch); if (!c) return 1;
            for (int y = 0; y < ch; y++) memcpy(c + (size_t)y * cw, img[k] + (size_t)(cy + y) * W + cx, (size_t)cw);
            free((void *)img[k]); img[k] = c;
        }
        ct = (uint8_t *)img[0]; W = cw; H = ch;
    }
    const size_t np = (size_t)W * H; uint8_t *mask = malloc(np);
    if (!mask) return 1;
    for (size_t i = 0; i < np; i++) mask[i] = ct[i] > 0;
    const int rs = (int)lround(rsu / px), rl = (int)lround(rlu / px);
    for (int k = 1; k < n; k++) {
        ink_stats_t st;
        if (ink_stats(img[k], mask, W, H, rs < 1 ? 1 : rs, rl < 2 ? 2 : rl, &st)) { fprintf(stderr, "ink-compare: out of memory\n"); return 1; }
        printf("ink %s: n %lld mean %.4f frac50 %.4f p50 %.3f p95 %.3f p99 %.3f band %.4f\n", lab[k], st.n, st.mean, st.frac50, st.p50, st.p95, st.p99, st.band);
        if (getenv("UFSM_INKCMP_FULL")) printf("full %s: %.17g %.17g %.17g\n", lab[k], st.mean, st.frac50, st.band);   /* (tests: every bit) */
    }
    for (int a = 1; a < n; a++) for (int b = a + 1; b < n; b++) { const double r = ink_corr(img[a], img[b], mask, np); printf("corr %s %s: %.4f\n", lab[a], lab[b], r);
                                                                    if (getenv("UFSM_INKCMP_FULL")) printf("full corr: %.17g\n", r); }
    int flo = 0, fhi = 0;
    if (*ropt(argc, argv, "--fixed", "") && (sscanf(ropt(argc, argv, "--fixed", ""), "%d,%d", &flo, &fhi) != 2 || fhi <= flo)) { fprintf(stderr, "ink-compare: --fixed LO,HI (ink bytes, LO < HI)\n"); return 2; }
    if (fhi > flo) for (int k = 1; k < n; k++)   /* display only: one fixed ink scale LO .. HI -> 0 .. 255 for every sheet (no per-sheet stretch) */
        for (size_t i = 0; i < np; i++) { const int v = (img[k][i] - flo) * 255 / (fhi - flo); ((uint8_t *)img[k])[i] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v); }
    else if (rflag(argc, argv, "--stretch")) {   /* display only: each ink stretched from its p1 to its p99 over the CT > 0 pixels */
        for (int k = 1; k < n; k++) {
            uint64_t h[256] = {0}, m = 0; for (size_t i = 0; i < np; i++) if (mask[i]) { h[img[k][i]]++; m++; }
            int lo = 0, hi = 255; uint64_t c = 0; int gl = 0;
            for (int v = 0; v < 256; v++) { c += h[v]; if (!gl && c >= m / 100) { lo = v; gl = 1; } if (c >= m - m / 100) { hi = v; break; } }
            if (hi <= lo) hi = lo + 1;
            uint8_t *s = malloc(np); if (!s) return 1;
            for (size_t i = 0; i < np; i++) { const int v = (img[k][i] - lo) * 255 / (hi - lo); s[i] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v); }
            fprintf(stderr, "ink-compare: %s stretched %d .. %d\n", lab[k], lo, hi);
            free((void *)img[k]); img[k] = s;
        }
    }
    if (overlay_red_skip(ct, np)) { fprintf(stderr, "ink-compare: under 5%% of the CT pixels have data: no %s\n", argv[2]); rc = 3; }
    else {
        if (rflag(argc, argv, "--stretch-ct")) { img_stretch(ct, mask, np, 0.01, 0.99); isct[0] = 0; }   /* the CT panel p1 .. p99 too (scans brighter than the fixed CT contrast) */
        const char *each = ropt(argc, argv, "--each", nullptr);
        const int maxpx = atoi(ropt(argc, argv, "--max-px", "0")), jq = atoi(ropt(argc, argv, "--quality", "90"));   /* JPEG longest side, quality */
        if (each) {   /* --each PREFIX: every panel its own unlabelled JPEG PREFIX<label><--each-suffix>.jpg (the CT only with --each-ct) */
            const char *suf = ropt(argc, argv, "--each-suffix", ""); const int ect = rflag(argc, argv, "--each-ct");
            for (int k = ect ? 0 : 1; k < n; k++) {
                char p[1400]; snprintf(p, sizeof p, "%s%s%s.jpg", each, lab[k], suf);
                int ow, oh; uint8_t *rgb = ink_panels(&img[k], &isct[k], 1, W, H, nullptr, !rflag(argc, argv, "--full"), &ow, &oh, nthreads);
                if (!rgb || img_fit_rgb(&rgb, &ow, &oh, maxpx, nthreads) || jpeg_write_rgb(p, ow, oh, rgb, jq)) { fprintf(stderr, "ink-compare: cannot write %s\n", p); rc = 1; }
                free(rgb);
            }
        } else {
            int ow, oh; uint8_t *rgb = ink_panels(img, isct, n, W, H, lab, !rflag(argc, argv, "--full"), &ow, &oh, nthreads);
            if (!rgb || img_fit_rgb(&rgb, &ow, &oh, maxpx, nthreads) || jpeg_write_rgb(argv[2], ow, oh, rgb, jq)) { fprintf(stderr, "ink-compare: cannot write %s\n", argv[2]); rc = 1; }
            free(rgb);
        }
    }
    fflush(stdout);
    for (int k = 0; k < n; k++) free((void *)img[k]);
    for (int k = 1; k < n; k++) free((void *)lab[k]);
    free(mask);
    return rc;
}
