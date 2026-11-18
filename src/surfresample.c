/* ufsm surf-resample SRC-SURFS SRC-MAPS DST-SURFS OUT [--suffix _reader3.tif] [--max-dist 4] [--scale 1] [--threads 8]
   A map rendered on one set of tifxyz surfaces (SRC-MAPS/<s><suffix>, on render --scale S's canvas of SRC-SURFS/<s>) resampled onto
   another set of surfaces of the same sheet (DST-SURFS/<d> -> OUT/<d><suffix>, the same canvas render --scale S gives <d>), by 3-D
   position: each destination point is projected onto the nearest source surface (bilinear patches of its vertex grid) and the source
   map is sampled there (bilinear). Why (2026-11-18): the unroll's INKMODE=r3 inked the same papyrus twice, once per wrap (the per-wrap
   images) and once per strip stripe (the mosaic), ~440 s of pod GPU each (1667: 37.5k vs 38.0k reader3 tiles, 1187 vs 1189 M px); the
   stripes can take the per-wrap maps instead.
   Destination vertices: the nearest source vertex in a spatial hash (8-voxel cells, the 27 around the point), per source surface; the
   two closest surfaces are refined (Gauss-Newton on the bilinear patch, walking across cells) and the smaller residual wins, if
   <= --max-dist voxels (else the point has no source: 0). Destination pixels (render's grid: pixel / gs, corners valid when Z >= 0):
   the corners' source coordinates interpolated when all four corners have the same source surface, else (seams between source
   surfaces) the pixel's own point projected from each corner's source position. Prints coverage and the residual distribution.
   --geom-only 1 (SRC-MAPS "-"): only the --residual surfaces, from the geometry alone (every source surface counts as mapped; no map read
   or written): the unroll writes them while reader3 still runs on the stripes, so the residual reader3 pass need not wait for the
   resample (pod q2 1667: resample 172 s, then residual reader3 117 s, both after the stripe maps). The residual is the same as a full
   run's whenever every source surface has its map (unroll.sh r3once compares the two and re-inks on a difference). */
#include "imgops.h"
#include "tiffw.h"
#include "json.h"
#include "optcheck.h"
#include "sysinfo.h"
#include <dirent.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

typedef struct { char name[256]; int w, h, mw, mh; float *X, *Y, *Z; uint8_t *M; double gs; } rs_surf;
typedef struct { uint64_t key; uint64_t v; } rs_ent;   /* v: surface << 48 | row << 24 | col */
enum { RS_CELL = 8 };

static double rs_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static const char *rs_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(k))) return argv[i + 1]; return d; }
static uint64_t rs_key(int64_t a, int64_t b, int64_t c) { return ((uint64_t)(a + (1 << 20)) << 42) | ((uint64_t)(b + (1 << 20)) << 21) | (uint64_t)(c + (1 << 20)); }
static int rs_ent_cmp(const void *a, const void *b) { const rs_ent *p = a, *q = b; return p->key < q->key ? -1 : p->key > q->key; }
static double rs_meta_scale(const char *dir) {
    char p[1200]; snprintf(p, sizeof p, "%s/meta.json", dir); FILE *f = fopen(p, "rb"); if (!f) return 0;
    char buf[65536]; const size_t n = fread(buf, 1, sizeof buf - 1, f); fclose(f); buf[n] = 0;
    json *j = json_parse(buf, n); if (!j) return 0; const double s = json_num(json_path(j, "scale.0"), 0); json_free(j); return s;
}
static int rs_load(const char *dir, const char *name, rs_surf *S) {
    char p[1400]; int w2, h2, w3, h3;
    snprintf(S->name, sizeof S->name, "%s", name);
    snprintf(p, sizeof p, "%s/%s/x.tif", dir, name); S->X = img_read_f32(p, &S->w, &S->h);
    snprintf(p, sizeof p, "%s/%s/y.tif", dir, name); S->Y = img_read_f32(p, &w2, &h2);
    snprintf(p, sizeof p, "%s/%s/z.tif", dir, name); S->Z = img_read_f32(p, &w3, &h3);
    if (!S->X || !S->Y || !S->Z || w2 != S->w || w3 != S->w || h2 != S->h || h3 != S->h || S->w < 2 || S->h < 2) { fprintf(stderr, "surf-resample: cannot read the grid %s/%s\n", dir, name); return -1; }
    return 0;
}
static inline int rs_ok(const rs_surf *S, int r, int c) { return r >= 0 && c >= 0 && r < S->h && c < S->w && S->Z[(size_t)r * S->w + c] >= 0; }
/* the closest point of surface S to q on the bilinear patches around (r, c) (start in cell (r0, c0) at (u, v)): residual^2, *fr *fc */
static double rs_project(const rs_surf *S, const double q[3], double fr0, double fc0, double *fr, double *fc) {
    int r0 = (int)floor(fr0), c0 = (int)floor(fc0); double u = fc0 - c0, v = fr0 - r0, best = 1e30;
    if (r0 > S->h - 2) { r0 = S->h - 2; v = 1; } if (c0 > S->w - 2) { c0 = S->w - 2; u = 1; } if (r0 < 0) { r0 = 0; v = 0; } if (c0 < 0) { c0 = 0; u = 0; }
    for (int step = 0; step < 6; step++) {
        if (!rs_ok(S, r0, c0) || !rs_ok(S, r0 + 1, c0) || !rs_ok(S, r0, c0 + 1) || !rs_ok(S, r0 + 1, c0 + 1)) break;
        const size_t i00 = (size_t)r0 * S->w + c0, i01 = i00 + 1, i10 = i00 + S->w, i11 = i10 + 1;
        const float *A[3] = {S->X, S->Y, S->Z};
        double d2 = 0;
        for (int it = 0; it < 6; it++) {   /* Gauss-Newton on (u, v) in [0, 1]^2 */
            double P[3], Pu[3], Pv[3];
            for (int k = 0; k < 3; k++) {
                const double a = A[k][i00], b = A[k][i01], c = A[k][i10], d = A[k][i11];
                P[k] = (1 - v) * ((1 - u) * a + u * b) + v * ((1 - u) * c + u * d); Pu[k] = (1 - v) * (b - a) + v * (d - c); Pv[k] = (1 - u) * (c - a) + u * (d - b);
            }
            const double e[3] = {q[0] - P[0], q[1] - P[1], q[2] - P[2]};
            const double guu = Pu[0] * Pu[0] + Pu[1] * Pu[1] + Pu[2] * Pu[2], gvv = Pv[0] * Pv[0] + Pv[1] * Pv[1] + Pv[2] * Pv[2], guv = Pu[0] * Pv[0] + Pu[1] * Pv[1] + Pu[2] * Pv[2];
            const double bu = Pu[0] * e[0] + Pu[1] * e[1] + Pu[2] * e[2], bv = Pv[0] * e[0] + Pv[1] * e[1] + Pv[2] * e[2], det = guu * gvv - guv * guv;
            d2 = e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
            if (!(det > 1e-12)) break;
            const double du = (gvv * bu - guv * bv) / det, dv = (guu * bv - guv * bu) / det;
            u += du; v += dv; u = u < 0 ? 0 : u > 1 ? 1 : u; v = v < 0 ? 0 : v > 1 ? 1 : v;
            if (fabs(du) + fabs(dv) < 1e-4) break;
        }
        {   double P[3]; for (int k = 0; k < 3; k++) { const double a = A[k][i00], b = A[k][i01], c = A[k][i10], d = A[k][i11]; P[k] = (1 - v) * ((1 - u) * a + u * b) + v * ((1 - u) * c + u * d); }
            d2 = (q[0] - P[0]) * (q[0] - P[0]) + (q[1] - P[1]) * (q[1] - P[1]) + (q[2] - P[2]) * (q[2] - P[2]); }
        if (d2 < best) { best = d2; *fr = r0 + v; *fc = c0 + u; }
        /* on an edge of the cell: continue in the neighbour */
        int nr = r0, nc = c0;
        if (u <= 0 && c0 > 0) { nc--; u = 1; } else if (u >= 1 && c0 < S->w - 2) { nc++; u = 0; }
        if (v <= 0 && r0 > 0) { nr--; v = 1; } else if (v >= 1 && r0 < S->h - 2) { nr++; v = 0; }
        if (nr == r0 && nc == c0) break;
        r0 = nr; c0 = nc;
    }
    return best;
}
static inline uint8_t rs_sample(const rs_surf *S, double fr, double fc) {   /* the map at grid (fr, fc), bilinear */
    double x = fc * S->gs, y = fr * S->gs; int x0 = (int)floor(x), y0 = (int)floor(y);
    if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0; if (x0 > S->mw - 2) x0 = S->mw - 2; if (y0 > S->mh - 2) y0 = S->mh - 2;
    const double a = x - x0, b = y - y0; const uint8_t *m = S->M + (size_t)y0 * S->mw + x0;
    const double v = (1 - b) * ((1 - a) * m[0] + a * m[1]) + b * ((1 - a) * m[S->mw] + a * m[S->mw + 1]);
    return (uint8_t)(v + 0.5);
}

typedef struct {
    rs_surf *S; int ns; rs_ent *E; size_t ne; double maxd2;
    const rs_surf *D; int16_t *vs; float *vr, *vc;     /* per destination vertex: source surface (-1 none), source grid row / col */
    uint8_t *out, *src, *sid; int ow, oh; double gs;   /* sid (--sid-out): the source surface + 1 (mod 255; 0 none) */   /* src: 1 where a pixel was sampled (--src-out) */
    int geom;   /* --geom-only: no maps (out stays 0) */
    _Atomic int next; _Atomic uint64_t nv, nvok, npx, npxok, nseam, hist[8];
} rs_job;
static void rs_lookup(const rs_job *J, const double q[3], int *sid, double *fr, double *fc, double *d2out) {   /* the destination point q's source */
    int bs[2] = {-1, -1}; double bd[2] = {1e30, 1e30}; uint64_t bv[2] = {0, 0};
    const int64_t a = (int64_t)floor(q[0] / RS_CELL), b = (int64_t)floor(q[1] / RS_CELL), c = (int64_t)floor(q[2] / RS_CELL);
    for (int da = -1; da <= 1; da++) for (int db = -1; db <= 1; db++) for (int dc = -1; dc <= 1; dc++) {
        const uint64_t k = rs_key(a + da, b + db, c + dc); size_t lo = 0, hi = J->ne;
        while (lo < hi) { const size_t m = (lo + hi) / 2; if (J->E[m].key < k) lo = m + 1; else hi = m; }
        for (size_t e = lo; e < J->ne && J->E[e].key == k; e++) {
            const int s = (int)(J->E[e].v >> 48), r = (int)(J->E[e].v >> 24 & 0xffffff), cc = (int)(J->E[e].v & 0xffffff);
            const rs_surf *S = &J->S[s]; const size_t i = (size_t)r * S->w + cc;
            const double d = (S->X[i] - q[0]) * (S->X[i] - q[0]) + (S->Y[i] - q[1]) * (S->Y[i] - q[1]) + (S->Z[i] - q[2]) * (S->Z[i] - q[2]);
            if (s == bs[0]) { if (d < bd[0]) { bd[0] = d; bv[0] = J->E[e].v; } }
            else if (s == bs[1]) { if (d < bd[1]) { bd[1] = d; bv[1] = J->E[e].v; } if (bd[1] < bd[0]) { int t = bs[0]; bs[0] = bs[1]; bs[1] = t; double x = bd[0]; bd[0] = bd[1]; bd[1] = x; uint64_t y = bv[0]; bv[0] = bv[1]; bv[1] = y; } }
            else if (d < bd[0]) { bs[1] = bs[0]; bd[1] = bd[0]; bv[1] = bv[0]; bs[0] = s; bd[0] = d; bv[0] = J->E[e].v; }
            else if (d < bd[1]) { bs[1] = s; bd[1] = d; bv[1] = J->E[e].v; }
        }
    }
    *sid = -1; *d2out = 1e30;
    for (int k = 0; k < 2; k++) if (bs[k] >= 0) {
        double r, cc; const double d2 = rs_project(&J->S[bs[k]], q, (double)(bv[k] >> 24 & 0xffffff), (double)(bv[k] & 0xffffff), &r, &cc);
        if (d2 < *d2out) { *d2out = d2; *sid = bs[k]; *fr = r; *fc = cc; }
    }
    if (*d2out > J->maxd2) *sid = -1;
}
static void *rs_vert_main(void *a) {
    rs_job *J = a; const rs_surf *D = J->D;
    uint64_t nv = 0, nvok = 0, hist[5] = {0};   /* (per-thread counts, added once: per-vertex atomics on shared lines serialised the threads) */
    for (int r; (r = atomic_fetch_add(&J->next, 1)) < D->h;) for (int c = 0; c < D->w; c++) {
        const size_t i = (size_t)r * D->w + c; J->vs[i] = -1;
        if (!(D->Z[i] >= 0)) continue;
        nv++;
        const double q[3] = {D->X[i], D->Y[i], D->Z[i]}; int s; double fr = 0, fc = 0, d2;
        rs_lookup(J, q, &s, &fr, &fc, &d2);
        if (s < 0) continue;
        J->vs[i] = (int16_t)s; J->vr[i] = (float)fr; J->vc[i] = (float)fc; nvok++;
        const double d = sqrt(d2); hist[d < 0.25 ? 0 : d < 0.5 ? 1 : d < 1 ? 2 : d < 2 ? 3 : 4]++;
    }
    atomic_fetch_add(&J->nv, nv); atomic_fetch_add(&J->nvok, nvok); for (int k = 0; k < 5; k++) atomic_fetch_add(&J->hist[k], hist[k]);
    return nullptr;
}
static void *rs_px_main(void *a) {
    rs_job *J = a; const rs_surf *D = J->D; uint64_t npx = 0, npxok = 0, nseam = 0;   /* (per-thread counts: 2 contended atomics a pixel were most of the pass) */
    for (int py; (py = atomic_fetch_add(&J->next, 1)) < J->oh;) for (int px = 0; px < J->ow; px++) {
        int r0 = (int)(py / J->gs), c0 = (int)(px / J->gs); r0 = r0 > D->h - 2 ? D->h - 2 : r0; c0 = c0 > D->w - 2 ? D->w - 2 : c0;
        const double v = py / J->gs - r0, u = px / J->gs - c0;
        const size_t i[4] = {(size_t)r0 * D->w + c0, (size_t)r0 * D->w + c0 + 1, (size_t)(r0 + 1) * D->w + c0, (size_t)(r0 + 1) * D->w + c0 + 1};
        if (!(D->Z[i[0]] >= 0 && D->Z[i[1]] >= 0 && D->Z[i[2]] >= 0 && D->Z[i[3]] >= 0)) continue;   /* (render: no surface there) */
        npx++;
        const double w4[4] = {(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v};
        const int s0 = J->vs[i[0]];
        if (s0 >= 0 && J->vs[i[1]] == s0 && J->vs[i[2]] == s0 && J->vs[i[3]] == s0) {
            double fr = 0, fc = 0; for (int k = 0; k < 4; k++) { fr += w4[k] * J->vr[i[k]]; fc += w4[k] * J->vc[i[k]]; }
            J->out[(size_t)py * J->ow + px] = J->geom ? 0 : rs_sample(&J->S[s0], fr, fc); J->src[(size_t)py * J->ow + px] = 1; if (J->sid) J->sid[(size_t)py * J->ow + px] = (uint8_t)(s0 % 255 + 1); npxok++; continue;
        }
        /* a seam (or a corner without a source): the pixel's own point, projected from each corner's source position */
        double q[3] = {0, 0, 0}; const float *A[3] = {D->X, D->Y, D->Z};
        for (int k = 0; k < 4; k++) for (int d = 0; d < 3; d++) q[d] += w4[k] * A[d][i[k]];
        int bs = -1; double bd = 1e30, br = 0, bc = 0;
        for (int k = 0; k < 4; k++) { const int s = J->vs[i[k]]; if (s < 0) continue;
            double fr, fc; const double d2 = rs_project(&J->S[s], q, J->vr[i[k]], J->vc[i[k]], &fr, &fc);
            if (d2 < bd) { bd = d2; bs = s; br = fr; bc = fc; } }
        if (bs < 0 || bd > J->maxd2) { int s; double fr = 0, fc = 0, d2; rs_lookup(J, q, &s, &fr, &fc, &d2); if (s < 0) continue; bs = s; br = fr; bc = fc; }
        J->out[(size_t)py * J->ow + px] = J->geom ? 0 : rs_sample(&J->S[bs], br, bc); J->src[(size_t)py * J->ow + px] = 1; if (J->sid) J->sid[(size_t)py * J->ow + px] = (uint8_t)(bs % 255 + 1); npxok++; nseam++;
    }
    atomic_fetch_add(&J->npx, npx); atomic_fetch_add(&J->npxok, npxok); atomic_fetch_add(&J->nseam, nseam);
    return nullptr;
}
static int rs_names(const char *dir, char ***out) {   /* the surface subdirectories (with x.tif), sorted */
    DIR *d = opendir(dir); if (!d) return -1; int n = 0, cap = 64; char **v = malloc(cap * sizeof *v); char p[1400]; struct stat st;
    for (struct dirent *e; (e = readdir(d));) { if (e->d_name[0] == '.') continue; snprintf(p, sizeof p, "%s/%s/x.tif", dir, e->d_name); if (stat(p, &st)) continue;
        if (n == cap) { cap *= 2; v = realloc(v, cap * sizeof *v); } v[n++] = strdup(e->d_name); }
    closedir(d);
    for (int i = 1; i < n; i++) for (int j = i; j > 0 && strcmp(v[j - 1], v[j]) > 0; j--) { char *t = v[j]; v[j] = v[j - 1]; v[j - 1] = t; }
    *out = v; return n;
}

static void rs_dilate(uint8_t *m, int w, int h, int R) {   /* m (0/1) dilated by R (Chebyshev), separable running counts */
    if (R <= 0) return;
    int *cnt = malloc(sizeof(int) * (size_t)(w > h ? w : h) + sizeof(int)); uint8_t *t = malloc((size_t)w * h); if (!cnt || !t) { free(cnt); free(t); return; }
    for (int y = 0; y < h; y++) { const uint8_t *row = m + (size_t)y * w; int c = 0;
        for (int x = 0; x < w && x < R; x++) c += row[x];
        for (int x = 0; x < w; x++) { if (x + R < w) c += row[x + R]; if (x - R - 1 >= 0) c -= row[x - R - 1]; t[(size_t)y * w + x] = c > 0; } }
    for (int x = 0; x < w; x++) { int c = 0;
        for (int y = 0; y < h && y < R; y++) c += t[(size_t)y * w + x];
        for (int y = 0; y < h; y++) { if (y + R < h) c += t[(size_t)(y + R) * w + x]; if (y - R - 1 >= 0) c -= t[(size_t)(y - R - 1) * w + x]; m[(size_t)y * w + x] = c > 0; } }
    free(cnt); free(t);
}
/* nearest sampled pixel (src) for every pixel by a two-pass chamfer (1 / 1.414): nn[i] its index (-1 none), dist[i] the distance */
static void rs_nearest(const uint8_t *src, int w, int h, int32_t *nn, float *dist) {
    const size_t N = (size_t)w * h;
    for (size_t i = 0; i < N; i++) { nn[i] = src[i] ? (int32_t)i : -1; dist[i] = src[i] ? 0.f : 1e30f; }
    const float d1 = 1.f, d2 = 1.41421356f;
#define RS_RELAX(j, dd) do { if (dist[j] + (dd) < dist[i]) { dist[i] = dist[j] + (dd); nn[i] = nn[j]; } } while (0)
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) { const size_t i = (size_t)y * w + x;
        if (x > 0) RS_RELAX(i - 1, d1);
        if (y > 0) { RS_RELAX(i - w, d1); if (x > 0) RS_RELAX(i - w - 1, d2); if (x + 1 < w) RS_RELAX(i - w + 1, d2); } }
    for (int y = h - 1; y >= 0; y--) for (int x = w - 1; x >= 0; x--) { const size_t i = (size_t)y * w + x;
        if (x + 1 < w) RS_RELAX(i + 1, d1);
        if (y + 1 < h) { RS_RELAX(i + w, d1); if (x + 1 < w) RS_RELAX(i + w + 1, d2); if (x > 0) RS_RELAX(i + w - 1, d2); } }
#undef RS_RELAX
}
static int rs_copy(const char *a, const char *b) {
    FILE *f = fopen(a, "rb"); if (!f) return -1; FILE *g = fopen(b, "wb"); if (!g) { fclose(f); return -1; }
    char buf[65536]; size_t n; int rc = 0; while ((n = fread(buf, 1, sizeof buf, f)) > 0) if (fwrite(buf, 1, n, g) != n) rc = -1;
    fclose(f); if (fclose(g)) rc = -1; return rc;
}
/* --compose OUT RMAPS: OUT/<d><suffix> = OUT/<d>_src.tif ? the resampled map : RMAPS/<d><suffix> (the residual ink-ext's map, same canvas) */
static int rs_compose(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm surf-resample --compose OUT RMAPS [--suffix _reader3.tif]\n"); return 2; }
    const char *od = argv[3], *rd = argc > 4 && argv[4][0] != '-' ? argv[4] : nullptr, *suf = rs_opt(argc, argv, "--suffix", "_reader3.tif");
    DIR *d = opendir(od); if (!d) { fprintf(stderr, "surf-resample --compose: no %s\n", od); return 1; }
    int rc = 0, n = 0; uint64_t nres = 0, nfill = 0, nmiss = 0;
    for (struct dirent *e; (e = readdir(d));) {
        const size_t L = strlen(e->d_name); if (L < 9 || strcmp(e->d_name + L - 8, "_src.tif")) continue;
        char nm[512]; snprintf(nm, sizeof nm, "%.*s", (int)(L - 8), e->d_name);
        char p[1400], q[1400], t[1500]; int w, h, w2, h2, w3 = 0, h3 = 0;
        snprintf(p, sizeof p, "%s/%s%s", od, nm, suf); snprintf(q, sizeof q, "%s/%s", od, e->d_name);
        uint8_t *M = img_read_u8(p, &w, &h), *S = img_read_u8(q, &w2, &h2), *R = nullptr;
        if (!M || !S || w != w2 || h != h2) { fprintf(stderr, "surf-resample --compose: %s: cannot read the map / source mask\n", nm); rc = 1; free(M); free(S); continue; }
        if (rd) { snprintf(q, sizeof q, "%s/%s%s", rd, nm, suf); R = img_read_u8(q, &w3, &h3); if (R && (w3 != w || h3 != h)) { fprintf(stderr, "surf-resample --compose: %s: residual map %d x %d, expected %d x %d\n", q, w3, h3, w, h); rc = 1; free(R); R = nullptr; } }
        for (size_t i = 0; i < (size_t)w * h; i++) if (!S[i]) { if (R) { M[i] = R[i]; nfill += R[i] != 0; } else { nmiss += M[i] != 0; M[i] = 0; } } else nres++;
        snprintf(t, sizeof t, "%s.compose.tmp", p);
        if (tiffw_write(t, w, h, 1, M, 1) || rename(t, p)) { fprintf(stderr, "surf-resample --compose: cannot write %s\n", p); rc = 1; }
        free(M); free(S); free(R); n++;
    }
    closedir(d);
    fprintf(stderr, "surf-resample --compose: %d maps in %s: %llu px resampled, %llu nonzero from the residual maps, %llu nonzero zeroed (no residual map)\n", n, od, (unsigned long long)nres, (unsigned long long)nfill, (unsigned long long)nmiss);
    return rc;
}
int cmd_surf_resample(int argc, char **argv) {
    if (argc >= 3 && !strcmp(argv[2], opt_n("--compose"))) return rs_compose(argc, argv);
    if (argc < 6) { fprintf(stderr, "usage: ufsm surf-resample SRC-SURFS SRC-MAPS DST-SURFS OUT [--suffix _reader3.tif] [--max-dist 4] [--scale 1] [--threads 8] [--src-out 1] [--sid-out 1] [--edge-band PX] [--fill-gap 16] [--residual DIR [--residual-margin 8] [--band-margin 32]] [--geom-only 1]\n"
                    "       ufsm surf-resample --compose OUT RMAPS [--suffix S]   OUT maps = resampled where OUT/<d>_src.tif, else RMAPS/<d><suffix>\n"); return 2; }
    const char *sd = argv[2], *md = argv[3], *dd = argv[4], *od = argv[5], *suf = rs_opt(argc, argv, "--suffix", "_reader3.tif");
    const double maxd = opt_atof("--max-dist", rs_opt(argc, argv, "--max-dist", "4")), rscale = opt_atof("--scale", rs_opt(argc, argv, "--scale", "1"));
    const int band = opt_atoi("--edge-band", rs_opt(argc, argv, "--edge-band", "0"));   /* px around a switch of source surface left unsampled (the residual) */
    const int fgap = opt_atoi("--fill-gap", rs_opt(argc, argv, "--fill-gap", "16"));   /* unsampled surface px within this of a sampled one (not in the band): its value */
    const char *resd = rs_opt(argc, argv, "--residual", nullptr); const int rmargin = opt_atoi("--residual-margin", rs_opt(argc, argv, "--residual-margin", "8")),
              bmargin = opt_atoi("--band-margin", rs_opt(argc, argv, "--band-margin", "32"));
    const int sido = opt_atoi("--sid-out", rs_opt(argc, argv, "--sid-out", "0"));   /* OUT/<d>_sid.tif: each pixel's source surface (sorted order) + 1 */
    const int geom = opt_atoi("--geom-only", rs_opt(argc, argv, "--geom-only", "0"));   /* only the --residual surfaces (no maps) */
    if (geom && !resd) { fprintf(stderr, "surf-resample: --geom-only needs --residual DIR\n"); return 2; }
    const int srco = opt_atoi("--src-out", rs_opt(argc, argv, "--src-out", "0"));   /* OUT/<d>_src.tif: 1 where sampled, 0 where no source (the residual) */
    const int nt0 = ufsm_threads(opt_atoi("--threads", rs_opt(argc, argv, "--threads", "8")), 8), nt = nt0 > 64 ? 64 : nt0 < 1 ? 1 : nt0;
    const double t0 = rs_now();
    /* --watch STOP: DST-SURFS is filled while this runs (the unroll links each wrap in as its flatten lands): every surface is done once,
       new ones are picked up every 2 s, and the run ends when the file STOP exists and a scan after it finds nothing new */
    const char *watch = rs_opt(argc, argv, "--watch", nullptr);
    char **sn, **dn; const int ns = rs_names(sd, &sn); int nd = rs_names(dd, &dn);
    if (nd < 0 && watch) { nd = 0; dn = nullptr; }
    if (ns < 1 || (nd < 1 && !watch) || ns > 32000) { fprintf(stderr, "surf-resample: no surfaces in %s or %s\n", sd, dd); return 1; }
    rs_surf *S = calloc((size_t)ns, sizeof *S); size_t nv = 0; int nm = 0;
    for (int s = 0; s < ns; s++) {
        if (rs_load(sd, sn[s], &S[s])) return 1;
        if (geom) { static const uint8_t one = 0; /* mgpu: const */ S[s].M = (uint8_t *)&one; S[s].mw = S[s].mh = 1; S[s].gs = 1; nm++; for (size_t i = 0; i < (size_t)S[s].w * S[s].h; i++) nv += S[s].Z[i] >= 0; continue; }   /* (M: "mapped"; never sampled) */
        char p[1400]; snprintf(p, sizeof p, "%s/%s%s", md, sn[s], suf); S[s].M = img_read_u8(p, &S[s].mw, &S[s].mh);
        if (!S[s].M) { fprintf(stderr, "surf-resample: no map %s: surface %s skipped\n", p, sn[s]); continue; }
        S[s].gs = (double)(S[s].mw - 1) / (S[s].w - 1);
        if (fabs((double)(S[s].mh - 1) / (S[s].h - 1) - S[s].gs) > 1e-3 * S[s].gs + 1e-9) { fprintf(stderr, "surf-resample: map %s (%d x %d) not proportional to the grid (%d x %d)\n", p, S[s].mw, S[s].mh, S[s].w, S[s].h); return 1; }
        nm++;
        for (size_t i = 0; i < (size_t)S[s].w * S[s].h; i++) nv += S[s].Z[i] >= 0;
    }
    rs_ent *E = malloc((nv ? nv : 1) * sizeof *E); size_t ne = 0;
    if (!E) { fprintf(stderr, "surf-resample: out of memory (%zu vertices)\n", nv); return 1; }
    for (int s = 0; s < ns; s++) { if (!S[s].M) continue;
        for (int r = 0; r < S[s].h; r++) for (int c = 0; c < S[s].w; c++) { const size_t i = (size_t)r * S[s].w + c; if (!(S[s].Z[i] >= 0)) continue;
            E[ne++] = (rs_ent){rs_key((int64_t)floor(S[s].X[i] / RS_CELL), (int64_t)floor(S[s].Y[i] / RS_CELL), (int64_t)floor(S[s].Z[i] / RS_CELL)), (uint64_t)s << 48 | (uint64_t)r << 24 | (uint64_t)c}; } }
    qsort(E, ne, sizeof *E, rs_ent_cmp);
    const double t1 = rs_now();
    fprintf(stderr, "surf-resample: %d source surfaces (%d with a map), %zu vertices indexed, %.1f s\n", ns, nm, ne, t1 - t0);
    if (!geom) mkdir(od, 0777);
    int rc = 0, ndone = 0, ndcap = 0; char **dl = nullptr;   /* (the names done: --watch rescans) */
    for (int stop = 0;;) {
    for (int k = 0; k < nd; k++) {
        int seen = 0; for (int i = 0; i < ndone && !seen; i++) seen = !strcmp(dl[i], dn[k]);
        if (seen) continue;
        if (ndone == ndcap) { ndcap = ndcap ? ndcap * 2 : 64; dl = realloc(dl, (size_t)ndcap * sizeof *dl); }
        dl[ndone++] = strdup(dn[k]);
        rs_surf D = {0}; if (rs_load(dd, dn[k], &D)) { rc = 1; continue; }
        char p[1400]; snprintf(p, sizeof p, "%s/%s", dd, dn[k]); const double ms = rs_meta_scale(p);
        if (!(ms > 0)) { fprintf(stderr, "surf-resample: %s: no scale in meta.json\n", p); rc = 1; continue; }
        rs_job J = {.S = S, .ns = ns, .E = E, .ne = ne, .maxd2 = maxd * maxd, .D = &D, .gs = rscale / ms, .geom = geom};
        J.ow = render_out_px(D.w, J.gs); J.oh = render_out_px(D.h, J.gs);
        const size_t n = (size_t)D.w * D.h;
        J.vs = malloc(n * sizeof *J.vs); J.vr = malloc(n * sizeof *J.vr); J.vc = malloc(n * sizeof *J.vc); J.out = calloc((size_t)J.ow * J.oh, 1); J.src = calloc((size_t)J.ow * J.oh, 1); J.sid = sido || band > 0 ? calloc((size_t)J.ow * J.oh, 1) : nullptr;
        if (!J.vs || !J.vr || !J.vc || !J.out || !J.src) { fprintf(stderr, "surf-resample: out of memory\n"); return 1; }
        const double ta = rs_now(); pthread_t th[64];
        for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, rs_vert_main, &J);
        for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
        const double tb = rs_now(); atomic_store(&J.next, 0);
        for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, rs_px_main, &J);
        for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
        uint64_t nband = 0, nresv = 0; uint8_t *bandpx = nullptr;
        if (band > 0) {   /* the switches between source surfaces (reader3's tile-edge effect at a source's border), widened: unsampled */
            uint8_t *sw = calloc((size_t)J.ow * J.oh, 1); if (!sw) { fprintf(stderr, "surf-resample: out of memory\n"); return 1; }
            for (int y = 0; y < J.oh; y++) for (int x = 0; x < J.ow; x++) { const size_t i = (size_t)y * J.ow + x; const uint8_t a = J.sid[i]; if (!a) continue;
                if ((x + 1 < J.ow && J.sid[i + 1] && J.sid[i + 1] != a) || (y + 1 < J.oh && J.sid[i + J.ow] && J.sid[i + J.ow] != a)) sw[i] = 1; }
            rs_dilate(sw, J.ow, J.oh, band);
            for (size_t i = 0; i < (size_t)J.ow * J.oh; i++) if (sw[i] && J.src[i]) { J.src[i] = 0; nband++; }
            bandpx = sw;   /* (kept: the residual's band seeds get --band-margin) */
        }
        uint64_t nfillg = 0;
        if (fgap > 0) {   /* slivers along the surface's border that the sources miss by a few px: the nearest sampled value (band pixels excluded) */
            uint8_t *bandm = nullptr;
            if (band > 0) { bandm = calloc((size_t)J.ow * J.oh, 1); if (bandm) for (size_t i = 0; i < (size_t)J.ow * J.oh; i++) bandm[i] = J.sid[i] && !J.src[i]; }
            int32_t *nn = malloc((size_t)J.ow * J.oh * sizeof *nn); float *dist = malloc((size_t)J.ow * J.oh * sizeof *dist);
            if (!nn || !dist) { fprintf(stderr, "surf-resample: out of memory\n"); return 1; }
            rs_nearest(J.src, J.ow, J.oh, nn, dist);
            for (int py = 0; py < J.oh; py++) for (int px = 0; px < J.ow; px++) { const size_t i = (size_t)py * J.ow + px;
                if (J.src[i] || nn[i] < 0 || dist[i] > fgap || (bandm && bandm[i])) continue;
                int r0 = (int)(py / J.gs), c0 = (int)(px / J.gs); r0 = r0 > D.h - 2 ? D.h - 2 : r0; c0 = c0 > D.w - 2 ? D.w - 2 : c0;
                const size_t v = (size_t)r0 * D.w + c0;
                if (!(D.Z[v] >= 0 && D.Z[v + 1] >= 0 && D.Z[v + D.w] >= 0 && D.Z[v + D.w + 1] >= 0)) continue;   /* (no surface: stays 0) */
                J.out[i] = J.out[nn[i]]; J.src[i] = 2; nfillg++; }
            for (size_t i = 0; i < (size_t)J.ow * J.oh; i++) if (J.src[i] == 2) J.src[i] = 1;
            free(nn); free(dist); free(bandm);
        }
        if (resd) {   /* the residual surface: the vertices whose pixels are unsampled, + rmargin vertices of context; the rest invalid */
            uint8_t *rv = calloc(n, 1), *rvb = calloc(n, 1); if (!rv || !rvb) { fprintf(stderr, "surf-resample: out of memory\n"); return 1; }
            for (int r = 0; r < D.h; r++) for (int c = 0; c < D.w; c++) { const size_t i = (size_t)r * D.w + c; if (!(D.Z[i] >= 0)) continue;
                int px = (int)lround(c * J.gs), py = (int)lround(r * J.gs); px = px >= J.ow ? J.ow - 1 : px; py = py >= J.oh ? J.oh - 1 : py;
                int r0 = (int)(py / J.gs), c0 = (int)(px / J.gs); r0 = r0 > D.h - 2 ? D.h - 2 : r0; c0 = c0 > D.w - 2 ? D.w - 2 : c0;
                const size_t v = (size_t)r0 * D.w + c0;
                const int surf = D.Z[v] >= 0 && D.Z[v + 1] >= 0 && D.Z[v + D.w] >= 0 && D.Z[v + D.w + 1] >= 0;   /* (render's pixel rule: else no pixel there) */
                const size_t pi = (size_t)py * J.ow + px;
                if (surf && !J.src[pi]) { if (bandpx && bandpx[pi]) rvb[i] = 1; else rv[i] = 1; }   /* (after the band and the sliver fill) */ }
            size_t nr0 = 0; for (size_t i = 0; i < n; i++) nr0 += rv[i] | rvb[i];
            if (nr0) {   /* band seeds get the wider context: a narrow residual strip has reader3 edge effects of its own (11-18: 8 vertices left the seams) */
                rs_dilate(rv, D.w, D.h, rmargin); rs_dilate(rvb, D.w, D.h, bmargin); for (size_t i = 0; i < n; i++) rv[i] |= rvb[i];
                float *RX = malloc(n * 4), *RY = malloc(n * 4), *RZ = malloc(n * 4); if (!RX || !RY || !RZ) { fprintf(stderr, "surf-resample: out of memory\n"); return 1; }
                for (size_t i = 0; i < n; i++) { const int keep = rv[i] && D.Z[i] >= 0; RX[i] = keep ? D.X[i] : -1; RY[i] = keep ? D.Y[i] : -1; RZ[i] = keep ? D.Z[i] : -1; nresv += keep; }
                char q[1400]; mkdir(resd, 0777); snprintf(q, sizeof q, "%s/%s", resd, dn[k]); mkdir(q, 0777);
                snprintf(q, sizeof q, "%s/%s/x.tif", resd, dn[k]); if (tiffw_write(q, D.w, D.h, 3, RX, 1)) rc = 1;
                snprintf(q, sizeof q, "%s/%s/y.tif", resd, dn[k]); if (tiffw_write(q, D.w, D.h, 3, RY, 1)) rc = 1;
                snprintf(q, sizeof q, "%s/%s/z.tif", resd, dn[k]); if (tiffw_write(q, D.w, D.h, 3, RZ, 1)) rc = 1;
                char m0[1400]; snprintf(m0, sizeof m0, "%s/%s/meta.json", dd, dn[k]); snprintf(q, sizeof q, "%s/%s/meta.json", resd, dn[k]); if (rs_copy(m0, q)) rc = 1;
                free(RX); free(RY); free(RZ);
            }
            free(rv); free(rvb);
        }
        free(bandpx);
        const double tc = rs_now();
        snprintf(p, sizeof p, "%s/%s%s", od, dn[k], suf);
        if (!geom && tiffw_write(p, J.ow, J.oh, 1, J.out, 1)) { fprintf(stderr, "surf-resample: cannot write %s\n", p); rc = 1; }
        if (band > 0 || resd || fgap > 0) fprintf(stderr, "surf-resample: %s: %llu px in the %d-px band around source switches, %llu px filled from <= %d px away; residual surface: %llu vertices\n",
                                                  dn[k], (unsigned long long)nband, band, (unsigned long long)nfillg, fgap, (unsigned long long)nresv);
        if (sido && J.sid && !geom) { snprintf(p, sizeof p, "%s/%s_sid.tif", od, dn[k]); if (tiffw_write(p, J.ow, J.oh, 1, J.sid, 1)) rc = 1; }
        if (srco && !geom) { snprintf(p, sizeof p, "%s/%s_src.tif", od, dn[k]); if (tiffw_write(p, J.ow, J.oh, 1, J.src, 1)) { fprintf(stderr, "surf-resample: cannot write %s\n", p); rc = 1; } }
        const uint64_t v = J.nv, vo = J.nvok, h0 = J.hist[0], h1 = J.hist[1], h2 = J.hist[2], h3 = J.hist[3], h4 = J.hist[4];
        fprintf(stderr, "surf-resample: %s: %d x %d px, vertices %llu, with a source %.4f (residual < 0.25 / 0.5 / 1 / 2 / %.3g voxels: %.3f %.3f %.3f %.3f %.3f); "
                        "pixels %llu, sampled %.4f, at seams %llu; %.1f + %.1f s\n", dn[k], J.ow, J.oh, (unsigned long long)v, v ? (double)vo / v : 0, maxd,
                vo ? (double)h0 / vo : 0, vo ? (double)h1 / vo : 0, vo ? (double)h2 / vo : 0, vo ? (double)h3 / vo : 0, vo ? (double)h4 / vo : 0,
                (unsigned long long)J.npx, J.npx ? (double)J.npxok / J.npx : 0, (unsigned long long)J.nseam, tb - ta, tc - tb);
        free(J.vs); free(J.vr); free(J.vc); free(J.out); free(J.src); free(J.sid); free(D.X); free(D.Y); free(D.Z);
    }
    if (!watch || stop) break;
    struct stat st; stop = !stat(watch, &st);   /* (seen before the rescan: whatever landed before STOP is still picked up) */
    if (!stop) { struct timespec ts = {2, 0}; nanosleep(&ts, nullptr); }
    for (int k = 0; k < nd; k++) free(dn[k]); free(dn);
    nd = rs_names(dd, &dn); if (nd < 0) { nd = 0; dn = nullptr; }
    }
    fprintf(stderr, "surf-resample: %d surfaces -> %s, %.1f s\n", ndone, od, rs_now() - t0);
    return rc;
}
