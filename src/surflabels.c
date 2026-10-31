/* ufsm surf-labels WRAPS OUT --axis F --axis-um U0 --dir +-1|auto --grid-um U (--shape Z,Y,X | --like ROOT KEY)
                   [--wrapcheck TSV] [--first K] [--last K] [--drop w1,w2,..] [--band-peaks LO,HI]
                   [--ct ROOT KEY [--min-onpap 0.2] [--ct-level 80] [--ct-range 15]]
                   [--exclude holdouts.json --exclude-ct NAME [--exclude-mm 0.5]]
                   [--radius 20] [--block 128] [--shard 512] [--bake 1] [--gt ROOT KEY] [--tsv OUT] [--threads N]
   Winding labels straight from our whole-scroll wraps (ufsm wraps / wrap-snap .sfc, rows = z, wrap k = the fit's level set W = k
   with W = U - dir theta / 2 pi, theta = atan2 around the axis in (-pi, pi]): every wrap's quads are rasterised on the scan's training
   grid (--grid-um; --like a CT level array of that grid gives the shape) with the sheet coordinate q = k + dir theta / 2 pi (continuous
   along the spiral, +1 turn between radially neighbouring sheets, rising outward; --dir auto: the sense from the wraps themselves, wrap
   k's theta = pi end meeting wrap k + dir's theta = -pi start, voted per row) as winding_mod14 codes (1 + round(18 q) mod 252,
   0 empty, 255 = two surfaces > 1/4 turn apart in one voxel: the ufsm raster --value rule), and the band phase of the pair rule
   (band_field_phase_pair: the nearest sheets of opposite winding parity, consecutive windings only, within --radius grid voxels;
   PHASE_* codes 0..251, 255 unknown: what phase-raster --pair 1 writes and the sampler's stored phase reads) is computed in the same pass,
   in blocks of --block voxels plus a --radius + 2 halo. Outputs (512^3 shards, key = the grid um):
     OUT/codes.zarr   the KEPT wraps' codes (dropped wraps' voxels 0)
     OUT/phase.zarr   the phase (with --bake 1, the default: unknown wherever ignore is set)
     OUT/ignore.zarr  0 = keep; 1 = the voxel's phase comes from a dropped wrap (on it, or between it and its neighbours: the gap
                      between two kept wraps around a dropped one is never labelled, nor are adjacent sheets invented across it),
                      2 = conflicting codes, 3 = masked CT (CT 0, with --ct), 4 = a holdout box
   (the sampler's "phase": {"ignore": ...}: nonzero = not trained on). Dropped wraps take part in the geometry (the pair rule sees
   every sheet) but none of their voxels is labelled. Wraps are dropped by: --wrapcheck (verdict dup / merged), --first / --last
   (ids outside, e.g. strip --last trims), --drop (a list), --band-peaks LO,HI (wrap-check CT peaks per fitted turn, one + 2 x two_plus,
   averaged over +-2 wraps, outside [LO, HI]), --min-onpap (with --ct: the share of the wrap's vertices with CT >= --ct-level within
   +-ct-range um along the ray, every 2nd row / column: a separate pass over the wraps and CT before labelling).
   --exclude: boxes of a holdouts.json whose "ct" ends with --exclude-ct (l0 voxels of the CT, axis-um) grown by --exclude-mm -> 4.
   --gt ROOT KEY (winding_mod14 codes on the same grid, e.g. Paris 4 q-9p6.zarr 9.6): validation. The GT phase (pair rule on the GT
   codes) against ours where both are known (raw and kept), the per-wrap counts (the wrap whose sheet is the voxel's lower winding),
   and the pair winding accuracy (src/pairacc.h, 0.5 / 1 / 2 / 4 mm) of our winding (q of the lower sheet + phase) at the GT-coded kept
   voxels. --tsv: the per-wrap table. Streaming: z bands of one shard row (the wraps' rows meeting it are decoded per band, ~16 B per
   vertex), blocks on --threads threads; deterministic for any thread count. */
#include "band.h"
#include "json.h"
#include "pairacc.h"
#include "phase_check.h"
#include "sources.h"
#include "store.h"
#include "surfcomp.h"
#include "sysinfo.h"
#include "z3w.h"
#include "zarr3.h"
#include <dirent.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { SL_TR = 8, SL_TC = 32 };   /* tile = 8 rows x 32 columns of quads */
enum { IG_DROP = 1, IG_CONFLICT = 2, IG_CT = 3, IG_HOLD = 4 };

static const char *sl_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
static int sl_idx(int argc, char **argv, const char *k) { for (int i = 2; i < argc; i++) if (!strcmp(argv[i], k)) return i; return -1; }
static double sl_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static inline uint8_t sl_code(double q) { long k = lround(q * 18.0) % 252; if (k < 0) k += 252; return (uint8_t)(1 + k); }   /* = ingest.c wind_code */
static inline int sl_cdiff(int a, int b) { int d = (a - b) % 252; if (d < -126) d += 252; if (d >= 126) d -= 252; return d; }
static inline uint64_t sl_mix(uint64_t x) { x += 0x9e3779b97f4a7c15ull; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull; x = (x ^ (x >> 27)) * 0x94d049bb133111ebull; return x ^ (x >> 31); }

typedef struct {
    int id, w, h, keep; char path[1400], why[24];
    sfc_reader *r; sfc_cache *ca; uint32_t ch; double um;
    float *rz0, *rz1;                 /* per row: grid z min / max of its valid vertices (rz0 > rz1: none) */
    float *ra, *rb;                   /* per row (--dir auto): the radius of the valid vertex nearest theta = pi (ra) / -pi (rb) when within 0.15 rad of it, else NAN */
    long long nv;                     /* valid vertices */
    double peaks, onpap; int has_wc; char verdict[16];
    _Atomic long long pap, papn;      /* --min-onpap pass */
    int b0, bn; float *g; uint8_t *v; /* the band's rows: grid z, y, x, q per vertex; valid */
    _Atomic long long kn, kk, gk, g10, g25, kgk, kg10, kg25;   /* phase voxels whose lower sheet is this wrap: known, kept; vs GT: known in both (raw / kept), |d| < .1, < .25 */
} sl_wrap;

typedef struct { int wi, r0, c0; float lo[3], hi[3]; } sl_tile;   /* quads rows r0 .. r0 + SL_TR, cols c0 .. c0 + SL_TC (band-relative rows) */
typedef struct { float p[3], d; uint32_t h; uint8_t q; } sl_rec;
typedef struct { sl_rec *r; size_t n, cap; } sl_recs;

typedef struct {
    sl_wrap *W; int nw; axis *ax; double s0, f, dir;   /* s0: wrap voxel -> axis voxel; f: wrap voxel -> grid voxel */
    int64_t shape[3]; int S, B, Hh; float rmax; int bake;
    z3w *wc, *wp, *wi; z3 *ct, *gt; double gum;
    double (*hold)[6]; int nhold;     /* holdout boxes, grid voxels (z, y, x, nz, ny, nx) */
    /* band */
    int64_t sz; sl_tile *tiles; size_t ntile; size_t *off; uint32_t *refs; int64_t ns[3];
    int *slist; int nsl; int nbs;     /* shards (sy * ns2 + sx) with tiles; blocks per shard */
    uint8_t **bc, **bp, **bi; _Atomic int *pend; pthread_mutex_t mu;
    _Atomic long next; long nitems; _Atomic int fail;
    /* stats */
    _Atomic long long ncode, nconf, nknown, nkept, nig[5], gtk, g10, g25, kgtk, kg10, kg25, nblk, gbins[PHASE_AGREE_BINS], kbins[PHASE_AGREE_BINS];
    sl_recs *recs; uint32_t pa_T; int nt;
} sl_job;

/* ---- wraps ---- */
static int sl_cmp_id(const void *a, const void *b) { const sl_wrap *x = a, *y = b; return (x->id > y->id) - (x->id < y->id); }
static int sl_open(sl_wrap *w) {
    if (sfc_open_file(w->path, &w->r)) { fprintf(stderr, "surf-labels: cannot open %s\n", w->path); return -1; }
    int c = sfc_find_channel(w->r, "x"); if (c < 0) c = 0; w->ch = (uint32_t)c;
    const sfc_channel *ch = sfc_channel_info(w->r, w->ch); w->w = (int)ch->width; w->h = (int)ch->height;
    const uint64_t ml = sfc_metadata_size(w->r); char *meta = calloc(ml + 1, 1);
    if (ml && sfc_read_metadata(w->r, meta, ml)) { free(meta); fprintf(stderr, "surf-labels: no metadata in %s\n", w->path); return -1; }
    const char *q = strstr(meta, "\"voxel_size_um\":"); w->um = q ? atof(q + 16) : 0; free(meta);
    if (!(w->um > 0)) { fprintf(stderr, "surf-labels: %s: no voxel_size_um in its meta\n", w->path); return -1; }
    const size_t budget = (size_t)((w->w + 63) / 64 + 1) * 3 * 64 * 64 * 12 * 2;
    if (sfc_cache_create(w->r, budget, &w->ca)) return -1;
    w->rz0 = malloc((size_t)w->h * 4); w->rz1 = malloc((size_t)w->h * 4); w->ra = malloc((size_t)w->h * 4); w->rb = malloc((size_t)w->h * 4);
    return w->rz0 && w->rz1 && w->ra && w->rb ? 0 : -1;
}
/* pass 0: per row the grid z range and the valid count (rows read 64 at a time) */
typedef struct { sl_job *J; _Atomic int next; _Atomic int fail; } sl_scan;
static void *sl_scan_run(void *arg) {
    sl_scan *S = arg; sl_job *J = S->J;
    for (int k; (k = atomic_fetch_add(&S->next, 1)) < J->nw;) {
        sl_wrap *w = &J->W[k]; const int R = 64; float *xyz = malloc((size_t)R * w->w * 12); uint8_t *v = malloc((size_t)R * w->w);
        const double f = w->um / J->gum;
        for (int r0 = 0; r0 < w->h && xyz && v; r0 += R) {
            const int n = r0 + R <= w->h ? R : w->h - r0; memset(v, 0, (size_t)n * w->w);
            if (sfc_cache_read_xyz_region(w->ca, w->ch, 0, (uint64_t)r0, (uint32_t)w->w, (uint32_t)n, xyz, 12, (size_t)w->w * 12, v)) { atomic_store(&S->fail, 1); break; }
            const double sw = J->s0 * w->um / J->W[0].um;
            for (int r = 0; r < n; r++) {
                float lo = 1e30f, hi = -1e30f; double tmax = -10, tmin = 10, rmx = NAN, rmn = NAN;
                for (int c = 0; c < w->w; c++) { const size_t i = (size_t)r * w->w + c; if (!v[i] || !(xyz[3 * i + 2] >= 0)) continue;
                    const float gz = (float)((xyz[3 * i + 2] + 0.5) * f - 0.5); if (gz < lo) lo = gz; if (gz > hi) hi = gz; w->nv++;
                    double cy, cx; axis_at(J->ax, xyz[3 * i + 2] * sw, &cy, &cx); cy /= sw; cx /= sw;
                    const double dy = xyz[3 * i + 1] - cy, dx = xyz[3 * i] - cx, th = atan2(dy, dx), rr = sqrt(dy * dy + dx * dx);
                    if (th > tmax) { tmax = th; rmx = rr; }
                    if (th < tmin) { tmin = th; rmn = rr; } }
                w->rz0[r0 + r] = lo; w->rz1[r0 + r] = hi;
                w->ra[r0 + r] = tmax > M_PI - 0.15 ? (float)rmx : NAN; w->rb[r0 + r] = tmin < -M_PI + 0.15 ? (float)rmn : NAN;
            }
        }
        sfc_cache_clear(w->ca); free(xyz); free(v);
    }
    return nullptr;
}
/* the rows of wrap w meeting grid z [z0, z1] (plus one row each side: the quads), converted to grid coordinates + q */
static int sl_load_band(sl_job *J, sl_wrap *w, double z0, double z1) {
    free(w->g); free(w->v); w->g = nullptr; w->v = nullptr; w->bn = 0;
    int a = -1, b = -1;
    for (int r = 0; r < w->h; r++) if (w->rz0[r] <= w->rz1[r] && w->rz1[r] >= z0 && w->rz0[r] <= z1) { if (a < 0) a = r; b = r; }
    if (a < 0) return 0;
    a = a > 0 ? a - 1 : 0; b = b + 1 < w->h ? b + 1 : b;
    const int n = b - a + 1; const size_t nv = (size_t)n * w->w;
    float *xyz = malloc(nv * 12); w->g = malloc(nv * 16); w->v = calloc(nv, 1);
    if (!xyz || !w->g || !w->v) { free(xyz); return -1; }
    if (sfc_cache_read_xyz_region(w->ca, w->ch, 0, (uint64_t)a, (uint32_t)w->w, (uint32_t)n, xyz, 12, (size_t)w->w * 12, w->v)) { free(xyz); return -1; }
    const double f = w->um / J->gum, sw = J->s0 * w->um / J->W[0].um;   /* wrap voxels -> grid voxels; -> axis voxels (s0: the first wrap's) */
    for (size_t i = 0; i < nv; i++) {
        if (!w->v[i] || !(xyz[3 * i + 2] >= 0)) { w->v[i] = 0; continue; }
        const double X = xyz[3 * i], Y = xyz[3 * i + 1], Z = xyz[3 * i + 2];
        double cy, cx;
        axis_at(J->ax, Z * sw, &cy, &cx); cy /= sw; cx /= sw;
        const double th = atan2(Y - cy, X - cx);
        float *g = w->g + 4 * i;
        g[0] = (float)((Z + 0.5) * f - 0.5); g[1] = (float)((Y + 0.5) * f - 0.5); g[2] = (float)((X + 0.5) * f - 0.5);
        g[3] = (float)(w->id + J->dir * th / (2 * M_PI));
    }
    free(xyz); sfc_cache_clear(w->ca);
    w->b0 = a; w->bn = n;
    return 0;
}

/* ---- the --min-onpap pass: per wrap, the share of (every 2nd row / column) vertices on papyrus ---- */
typedef struct { z3 *z; int64_t sh[3], g[3]; uint8_t **ch; uint8_t *need; } sl_ctc;   /* 128^3 chunk table */
static inline int sl_ctc_at(const sl_ctc *c, int64_t z, int64_t y, int64_t x) {
    if (z < 0 || y < 0 || x < 0 || z >= c->sh[0] || y >= c->sh[1] || x >= c->sh[2]) return 0;
    const uint8_t *b = c->ch[((size_t)(z >> 7) * c->g[1] + (size_t)(y >> 7)) * c->g[2] + (size_t)(x >> 7)];
    return b ? b[((size_t)(z & 127) * 128 + (size_t)(y & 127)) * 128 + (size_t)(x & 127)] : 0;
}
typedef struct { sl_job *J; sl_ctc *c; int64_t *todo; size_t ntodo; _Atomic size_t next; _Atomic int fail; int mode; double R, lev; _Atomic int wnext; } sl_pap;
static void *sl_pap_run(void *arg) {
    sl_pap *P = arg; sl_job *J = P->J; sl_ctc *c = P->c;
    if (P->mode == 0) {   /* decode the needed chunks */
        for (size_t k; (k = atomic_fetch_add(&P->next, 1)) < P->ntodo;) {
            const int64_t id = P->todo[k], cz = id / (c->g[1] * c->g[2]), cy = id / c->g[2] % c->g[1], cx = id % c->g[2];
            const int64_t o[3] = {cz * 128, cy * 128, cx * 128}, n[3] = {128, 128, 128};
            uint8_t *b = malloc((size_t)128 * 128 * 128);
            if (!b || z3_read(c->z, o, n, b, 1)) { free(b); atomic_store(&P->fail, 1); continue; }
            c->ch[id] = b;
        }
        return nullptr;
    }
    for (int k; (k = atomic_fetch_add(&P->wnext, 1)) < J->nw;) {
        sl_wrap *w = &J->W[k]; long long on = 0, n = 0;
        for (int r = (w->b0 & 1); r < w->bn; r += 2) for (int col = 0; col < w->w; col += 2) {
            const size_t i = (size_t)r * w->w + col; if (!w->v[i]) continue;
            const float *g = w->g + 4 * i;
            const int64_t zi = llround(g[0]);
            if (zi < J->sz * 128 || zi >= (J->sz + 1) * 128) continue;   /* (this 128-row band's vertices only) */
            if (P->mode == 1) {
                for (int dy = -2; dy <= 2; dy += 4) for (int dx = -2; dx <= 2; dx += 4) {
                    const int64_t y = llround(g[1]) + dy, x = llround(g[2]) + dx;
                    if (y < 0 || x < 0 || y >= c->sh[1] || x >= c->sh[2] || zi >= c->sh[0]) continue;
                    c->need[((size_t)(zi >> 7) * c->g[1] + (size_t)(y >> 7)) * c->g[2] + (size_t)(x >> 7)] = 1; }
                continue; }
            /* the ray: radial from the axis at the vertex's z, in grid voxels */
            double cy, cx; const double sw = J->s0 / J->f;   /* grid voxels -> axis voxels */
            axis_at(J->ax, (g[0] + 0.5) * sw - 0.5, &cy, &cx); cy = (cy + 0.5) / sw - 0.5; cx = (cx + 0.5) / sw - 0.5;
            double uy = g[1] - cy, ux = g[2] - cx; const double rr = sqrt(uy * uy + ux * ux); if (rr > 1e-6) { uy /= rr; ux /= rr; } else { uy = 0; ux = 0; }
            int best = 0;
            for (double t = -P->R; t <= P->R + 1e-9; t += 0.5) { const int v = sl_ctc_at(c, zi, llround(g[1] + t * uy), llround(g[2] + t * ux)); if (v > best) best = v; }
            n++; on += best >= P->lev;
        }
        if (P->mode == 2) { atomic_fetch_add(&w->pap, on); atomic_fetch_add(&w->papn, n); }
    }
    return nullptr;
}

/* ---- the labelling pass: blocks ---- */
static int sl_tile_hits(const sl_tile *t, const double lo[3], const double hi[3]) {
    for (int d = 0; d < 3; d++) if (t->hi[d] < lo[d] || t->lo[d] > hi[d]) return 0;
    return 1;
}
static void sl_raster_tile(const sl_job *J, const sl_tile *t, const int64_t o[3], int N, uint8_t *codes, uint8_t *drop, uint16_t *wid) {
    const sl_wrap *w = &J->W[t->wi]; const int W = w->w;
    const int r1 = t->r0 + SL_TR < w->bn - 1 ? t->r0 + SL_TR : w->bn - 1, c1 = t->c0 + SL_TC < W - 1 ? t->c0 + SL_TC : W - 1;
    const double lo[3] = {(double)o[0] - 0.5, (double)o[1] - 0.5, (double)o[2] - 0.5}, hi[3] = {o[0] + N - 0.5, o[1] + N - 0.5, o[2] + N - 0.5};
    for (int r = t->r0; r < r1; r++) for (int c = t->c0; c < c1; c++) {
        const size_t i00 = (size_t)r * W + c, i01 = i00 + 1, i10 = i00 + W, i11 = i10 + 1;
        if (!(w->v[i00] && w->v[i01] && w->v[i10] && w->v[i11])) continue;
        const size_t idx[4] = {i00, i01, i10, i11};
        double p[4][3], q[4], cl[3] = {1e30, 1e30, 1e30}, ch[3] = {-1e30, -1e30, -1e30};
        for (int k = 0; k < 4; k++) {
            const float *g = w->g + 4 * idx[k];
            for (int d = 0; d < 3; d++) { p[k][d] = g[d]; if (g[d] < cl[d]) cl[d] = g[d]; if (g[d] > ch[d]) ch[d] = g[d]; }
            q[k] = g[3]; if (k) q[k] -= round(q[k] - q[0]);   /* (across the theta cut: one turn) */
        }
        int out = 0; for (int d = 0; d < 3; d++) if (cl[d] > hi[d] || ch[d] < lo[d]) out = 1;
        if (out) continue;
        double e1 = 0, e2 = 0;
        for (int d = 0; d < 3; d++) { e1 += (p[1][d] - p[0][d]) * (p[1][d] - p[0][d]); e2 += (p[2][d] - p[0][d]) * (p[2][d] - p[0][d]); }
        int nu = (int)ceil(sqrt(e1) / 0.5) + 1, nv = (int)ceil(sqrt(e2) / 0.5) + 1; if (nu > 256) nu = 256; if (nv > 256) nv = 256;
        for (int a = 0; a < nu; a++) for (int b = 0; b < nv; b++) {
            const double u = nu > 1 ? (double)a / (nu - 1) : 0, v = nv > 1 ? (double)b / (nv - 1) : 0;
            const double w00 = (1 - u) * (1 - v), w01 = u * (1 - v), w10 = (1 - u) * v, w11 = u * v;
            int vox[3], in = 1;
            for (int d = 0; d < 3; d++) {
                const double x = w00 * p[0][d] + w01 * p[1][d] + w10 * p[2][d] + w11 * p[3][d];
                vox[d] = (int)floor(x - o[d] + 0.5); if (vox[d] < 0 || vox[d] >= N) in = 0;
            }
            if (!in) continue;
            const size_t vi = ((size_t)vox[0] * N + vox[1]) * N + vox[2];
            const uint8_t code = sl_code(w00 * q[0] + w01 * q[1] + w10 * q[2] + w11 * q[3]);
            if (!w->keep) drop[vi] = 1;
            if (codes[vi] == 0) { codes[vi] = code; wid[vi] = (uint16_t)(t->wi + 1); }
            else if (codes[vi] != 255 && abs(sl_cdiff(codes[vi], code)) > 4) codes[vi] = 255;
        }
    }
}
typedef struct { uint8_t *codes, *drop, *phase, *gtc, *gph, *ct; uint16_t *wid; int32_t *lo, *hi; uint32_t *tl; size_t tcap; } sl_scratch;
static int sl_block(sl_job *J, sl_scratch *X, int si, int bk, int tid) {
    const int S = J->S, B = J->B, Hh = J->Hh, N = B + 2 * Hh, nb = S / B;
    const int64_t sy = J->slist[si] / J->ns[2], sx = J->slist[si] % J->ns[2];
    const int bz = bk / (nb * nb), by = bk / nb % nb, bx = bk % nb;
    const int64_t b0[3] = {J->sz * S + (int64_t)bz * B, sy * S + (int64_t)by * B, sx * S + (int64_t)bx * B};
    for (int d = 0; d < 3; d++) if (b0[d] >= J->shape[d]) return 0;
    const int64_t o[3] = {b0[0] - Hh, b0[1] - Hh, b0[2] - Hh};
    const double lo[3] = {o[0] - 1.0, o[1] - 1.0, o[2] - 1.0}, hi[3] = {o[0] + N + 0.0, o[1] + N + 0.0, o[2] + N + 0.0};
    const size_t s = (size_t)J->slist[si]; size_t nt = 0;
    for (size_t k = J->off[s]; k < J->off[s + 1]; k++) if (sl_tile_hits(&J->tiles[J->refs[k]], lo, hi)) {
        if (nt == X->tcap) { X->tcap = X->tcap ? 2 * X->tcap : 1024; X->tl = realloc(X->tl, X->tcap * 4); }
        X->tl[nt++] = J->refs[k];
    }
    if (!nt) return 0;
    const size_t NN = (size_t)N * N * N;
    memset(X->codes, 0, NN); memset(X->drop, 0, NN); memset(X->wid, 0, NN * 2);
    for (size_t k = 0; k < nt; k++) sl_raster_tile(J, &J->tiles[X->tl[k]], o, N, X->codes, X->drop, X->wid);
    size_t any = 0; for (size_t i = 0; i < NN && !any; i++) any = X->codes[i] != 0;
    if (!any) return 0;
    const int n3[3] = {N, N, N};
    if (band_field_phase_pair_ex(X->codes, n3, J->rmax, X->phase, nullptr, X->lo, X->hi)) return -1;
    if (J->gt) {
        const int64_t nn[3] = {N, N, N};
        if (z3_read(J->gt, o, nn, X->gtc, 1)) return -1;
        if (band_field_phase_pair(X->gtc, n3, J->rmax, X->gph, nullptr)) return -1;
    }
    if (J->ct) { const int64_t nn[3] = {B, B, B}; if (z3_read(J->ct, b0, nn, X->ct, 1)) return -1; }
    /* the shard's buffers */
    pthread_mutex_lock(&J->mu);
    if (!J->bc[si]) {
        const size_t S3 = (size_t)S * S * S;
        J->bc[si] = calloc(S3, 1); J->bp[si] = malloc(S3); J->bi[si] = calloc(S3, 1);
        if (J->bp[si]) memset(J->bp[si], PHASE_UNKNOWN, S3);
    }
    uint8_t *oc = J->bc[si], *op = J->bp[si], *oi = J->bi[si];
    pthread_mutex_unlock(&J->mu);
    if (!oc || !op || !oi) return -1;
    long long ncode = 0, nconf = 0, nknown = 0, nkept = 0, nig[5] = {0}, gtk = 0, g10 = 0, g25 = 0, kgtk = 0, kg10 = 0, kg25 = 0, gb[PHASE_AGREE_BINS] = {0}, kb[PHASE_AGREE_BINS] = {0};
    sl_recs *R = &J->recs[tid]; const double mm = J->gum * 1e-3;
    for (int z = 0; z < B; z++) for (int y = 0; y < B; y++) for (int x = 0; x < B; x++) {
        const int64_t gz = b0[0] + z, gy = b0[1] + y, gx = b0[2] + x;
        if (gz >= J->shape[0] || gy >= J->shape[1] || gx >= J->shape[2]) continue;
        const size_t i = ((size_t)(z + Hh) * N + (y + Hh)) * N + (x + Hh);
        const int c = X->codes[i]; int ig = 0;
        if (c == 255) { ig = IG_CONFLICT; nconf++; }
        const int ph = X->phase[i];
        if (ph != PHASE_UNKNOWN && (X->drop[X->lo[i]] || X->drop[X->hi[i]])) ig = IG_DROP;
        if (c && c != 255 && X->drop[i]) ig = IG_DROP;
        if (J->ct && !X->ct[((size_t)z * B + y) * B + x]) ig = IG_CT;
        for (int h = 0; h < J->nhold; h++) {
            const double *hb = J->hold[h];
            if (gz >= hb[0] && gz < hb[0] + hb[3] && gy >= hb[1] && gy < hb[1] + hb[4] && gx >= hb[2] && gx < hb[2] + hb[5]) { ig = IG_HOLD; break; }
        }
        const size_t oo = ((size_t)(z + bz * B) * S + (y + by * B)) * S + (x + bx * B);
        oc[oo] = c && c != 255 && !X->drop[i] ? (uint8_t)c : 0; ncode += oc[oo] != 0;
        oi[oo] = (uint8_t)ig;
        op[oo] = (uint8_t)(J->bake && ig ? PHASE_UNKNOWN : ph);
        if (ph == PHASE_UNKNOWN) continue;
        nknown++; nig[ig]++; if (!ig) nkept++;
        const int wi = X->wid[X->lo[i]] - 1; sl_wrap *w = wi >= 0 ? &J->W[wi] : nullptr;
        if (w) { atomic_fetch_add_explicit(&w->kn, 1, memory_order_relaxed); if (!ig) atomic_fetch_add_explicit(&w->kk, 1, memory_order_relaxed); }
        if (!J->gt) continue;
        const int gp = X->gph[i];
        if (gp != PHASE_UNKNOWN) {
            double d = fabs((double)(ph - gp) / PHASE_PERIOD); d -= floor(d + 0.5); d = fabs(d);
            const int bin = phase_agree_bin(d);
            gtk++; g10 += d < 0.1; g25 += d < 0.25; gb[bin]++;
            if (!ig) { kgtk++; kg10 += d < 0.1; kg25 += d < 0.25; kb[bin]++; }
            if (w) { atomic_fetch_add_explicit(&w->gk, 1, memory_order_relaxed); atomic_fetch_add_explicit(&w->g10, d < 0.1, memory_order_relaxed); atomic_fetch_add_explicit(&w->g25, d < 0.25, memory_order_relaxed);
                     if (!ig) { atomic_fetch_add_explicit(&w->kgk, 1, memory_order_relaxed); atomic_fetch_add_explicit(&w->kg10, d < 0.1, memory_order_relaxed); atomic_fetch_add_explicit(&w->kg25, d < 0.25, memory_order_relaxed); } }
        }
        const int gc = X->gtc[i];
        if (!ig && gc >= 1 && gc <= 252) {   /* pair accuracy record: our winding at a GT-coded voxel */
            const uint64_t key = ((uint64_t)gz << 42) ^ ((uint64_t)gy << 21) ^ (uint64_t)gx; const uint32_t h = (uint32_t)(sl_mix(key) >> 32);
            if (h < J->pa_T) {
                const double wo = (X->codes[X->lo[i]] - 1) / 18.0 + (double)ph / PHASE_PERIOD, d = wo - (gc - 1) / 18.0;
                if (R->n == R->cap) { R->cap = R->cap ? 2 * R->cap : 4096; R->r = realloc(R->r, R->cap * sizeof *R->r); }
                R->r[R->n++] = (sl_rec){{(float)(gz * mm), (float)(gy * mm), (float)(gx * mm)}, (float)d, h, (uint8_t)gc};
            }
        }
    }
    atomic_fetch_add(&J->ncode, ncode); atomic_fetch_add(&J->nconf, nconf); atomic_fetch_add(&J->nknown, nknown); atomic_fetch_add(&J->nkept, nkept);
    for (int k = 0; k < 5; k++) atomic_fetch_add(&J->nig[k], nig[k]);
    atomic_fetch_add(&J->gtk, gtk); atomic_fetch_add(&J->g10, g10); atomic_fetch_add(&J->g25, g25);
    atomic_fetch_add(&J->kgtk, kgtk); atomic_fetch_add(&J->kg10, kg10); atomic_fetch_add(&J->kg25, kg25);
    for (int k = 0; k < PHASE_AGREE_BINS; k++) { atomic_fetch_add(&J->gbins[k], gb[k]); atomic_fetch_add(&J->kbins[k], kb[k]); }
    atomic_fetch_add(&J->nblk, 1);
    return 0;
}
static void *sl_block_run(void *arg) {
    void **a = arg; sl_job *J = a[0]; const int tid = (int)(intptr_t)a[1];
    const int N = J->B + 2 * J->Hh; const size_t NN = (size_t)N * N * N;
    sl_scratch X = {0};
    X.codes = malloc(NN); X.drop = malloc(NN); X.phase = malloc(NN); X.wid = malloc(NN * 2); X.lo = malloc(NN * 4); X.hi = malloc(NN * 4);
    X.gtc = calloc(NN, 1); X.gph = malloc(NN); X.ct = malloc((size_t)J->B * J->B * J->B);
    if (!X.codes || !X.drop || !X.phase || !X.wid || !X.lo || !X.hi || !X.gtc || !X.gph || !X.ct) { atomic_store(&J->fail, 1); }
    else for (long it; (it = atomic_fetch_add(&J->next, 1)) < J->nitems && !atomic_load(&J->fail);) {
        const int si = (int)(it / J->nbs), bk = (int)(it % J->nbs);
        if (sl_block(J, &X, si, bk, tid)) { fprintf(stderr, "surf-labels: block failed (memory or read: %s)\n", z3_error()); atomic_store(&J->fail, 1); }
        if (atomic_fetch_sub(&J->pend[si], 1) == 1) {   /* the shard's last block: write it */
            const int64_t sy = J->slist[si] / J->ns[2], sx = J->slist[si] % J->ns[2];
            if (J->bc[si] && (z3w_write_shard(J->wc, J->sz, sy, sx, J->bc[si], 1) || z3w_write_shard(J->wp, J->sz, sy, sx, J->bp[si], 1) || z3w_write_shard(J->wi, J->sz, sy, sx, J->bi[si], 1))) {
                fprintf(stderr, "surf-labels: %s\n", z3w_error()); atomic_store(&J->fail, 1);
            }
            free(J->bc[si]); free(J->bp[si]); free(J->bi[si]); J->bc[si] = J->bp[si] = J->bi[si] = nullptr;
        }
    }
    free(X.codes); free(X.drop); free(X.phase); free(X.wid); free(X.lo); free(X.hi); free(X.gtc); free(X.gph); free(X.ct); free(X.tl);
    return nullptr;
}
typedef struct { sl_job *J; double z0, z1; _Atomic int next, fail; } sl_loadjob;
static void *sl_load_run(void *arg) {
    sl_loadjob *L = arg;
    for (int k; (k = atomic_fetch_add(&L->next, 1)) < L->J->nw;) if (sl_load_band(L->J, &L->J->W[k], L->z0, L->z1)) atomic_store(&L->fail, 1);
    return nullptr;
}
static int sl_load_all(sl_job *J, double z0, double z1, int nt) {
    sl_loadjob L = {J, z0, z1, 0, 0}; pthread_t th[64]; const int m = nt < 64 ? nt : 64;
    for (int q = 0; q < m; q++) pthread_create(&th[q], nullptr, sl_load_run, &L);
    for (int q = 0; q < m; q++) pthread_join(th[q], nullptr);
    return atomic_load(&L.fail) ? -1 : 0;
}
static int sl_cmp_rec(const void *a, const void *b) {
    const sl_rec *x = a, *y = b; if (x->h != y->h) return x->h < y->h ? -1 : 1;
    for (int d = 0; d < 3; d++) if (x->p[d] != y->p[d]) return x->p[d] < y->p[d] ? -1 : 1;
    return 0;
}
static int sl_read_file(const char *p, char **buf) {
    FILE *f = fopen(p, "rb"); if (!f) return -1;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    *buf = malloc((size_t)n + 1); if (!*buf || fread(*buf, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(*buf); return -1; }
    (*buf)[n] = 0; fclose(f); return (int)n;
}

int cmd_surf_labels(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: ufsm surf-labels <wraps-dir> <out-dir> --axis F --axis-um U0 --dir +-1 --grid-um U (--shape Z,Y,X | --like ROOT KEY)\n"
                        "       [--wrapcheck TSV] [--first K] [--last K] [--drop w1,w2] [--band-peaks LO,HI] [--ct ROOT KEY [--min-onpap 0.2] [--ct-level 80] [--ct-range 15]]\n"
                        "       [--exclude holdouts.json --exclude-ct NAME [--exclude-mm 0.5]] [--radius 20] [--block 128] [--shard 512] [--bake 1] [--gt ROOT KEY] [--tsv OUT] [--threads N]\n");
        return 2;
    }
    const char *in = argv[2], *out = argv[3];
    const char *axp = sl_opt(argc, argv, "--axis", nullptr);
    const double axum = atof(sl_opt(argc, argv, "--axis-um", "0")), gum = atof(sl_opt(argc, argv, "--grid-um", "0"));
    const int dir_auto = !strcmp(sl_opt(argc, argv, "--dir", ""), "auto"); double dir = dir_auto ? 1 : atof(sl_opt(argc, argv, "--dir", "0"));
    const int nt = ufsm_threads(atoi(sl_opt(argc, argv, "--threads", "0")), 16), S = atoi(sl_opt(argc, argv, "--shard", "512")), B = atoi(sl_opt(argc, argv, "--block", "128"));
    const double rmax = atof(sl_opt(argc, argv, "--radius", "20"));
    if (!axp || !(axum > 0) || !(gum > 0) || (!dir_auto && dir != 1 && dir != -1) || S < 128 || S % 128 || B < 16 || S % B || !(rmax > 0) || rmax > 120) {
        fprintf(stderr, "surf-labels: need --axis F --axis-um U0 --grid-um U --dir 1|-1|auto; --shard a multiple of 128, --block dividing it, --radius in (0, 120]\n"); return 2;
    }
    sl_job J; memset(&J, 0, sizeof J);
    J.gum = gum; J.dir = dir; J.S = S; J.B = B; J.rmax = (float)rmax; J.Hh = (int)ceil(rmax) + 2; J.bake = atoi(sl_opt(argc, argv, "--bake", "1")); J.nt = nt;
    pthread_mutex_init(&J.mu, nullptr);
    axis ax = {0}; if (axis_load(&ax, axp)) { fprintf(stderr, "surf-labels: cannot load the axis %s\n", axp); return 1; }
    J.ax = &ax;
    /* the grid */
    { const int li = sl_idx(argc, argv, "--like");
      if (li > 0) {
          if (li + 2 >= argc) { fprintf(stderr, "surf-labels: --like ROOT KEY\n"); return 2; }
          z3 *z = z3_open(store_open(argv[li + 1]), argv[li + 2], nullptr); if (!z) { fprintf(stderr, "surf-labels: --like: %s\n", z3_error()); return 1; }
          for (int d = 0; d < 3; d++) J.shape[d] = z3_meta_of(z)->shape[d];
          z3_close(z);
      } else { long long a, b, c; if (sscanf(sl_opt(argc, argv, "--shape", ""), "%lld,%lld,%lld", &a, &b, &c) != 3 || a <= 0 || b <= 0 || c <= 0) { fprintf(stderr, "surf-labels: --shape Z,Y,X or --like ROOT KEY\n"); return 2; }
               J.shape[0] = a; J.shape[1] = b; J.shape[2] = c; } }
    for (int d = 0; d < 3; d++) J.ns[d] = (J.shape[d] + S - 1) / S;
    { const int ci = sl_idx(argc, argv, "--ct"); if (ci > 0) { if (ci + 2 >= argc) return 2; J.ct = z3_open(store_open(argv[ci + 1]), argv[ci + 2], sl_opt(argc, argv, "--cache", nullptr));
        if (!J.ct || memcmp(z3_meta_of(J.ct)->shape, J.shape, sizeof J.shape)) { fprintf(stderr, "surf-labels: --ct %s %s: cannot open or not on the grid\n", argv[ci + 1], argv[ci + 2]); return 1; } } }
    { const int gi = sl_idx(argc, argv, "--gt"); if (gi > 0) { if (gi + 2 >= argc) return 2; J.gt = z3_open(store_open(argv[gi + 1]), argv[gi + 2], nullptr);
        if (!J.gt || memcmp(z3_meta_of(J.gt)->shape, J.shape, sizeof J.shape)) { fprintf(stderr, "surf-labels: --gt %s %s: cannot open or not on the grid\n", argv[gi + 1], argv[gi + 2]); return 1; } } }
    /* the wraps */
    DIR *dd = opendir(in); if (!dd) { fprintf(stderr, "surf-labels: cannot open %s\n", in); return 1; }
    int nw = 0, cap = 64; sl_wrap *W = calloc(cap, sizeof *W); struct dirent *de;
    while ((de = readdir(dd))) {
        int id; char tail[8] = ""; if (sscanf(de->d_name, "w%d.%7s", &id, tail) != 2 || strcmp(tail, "sfc")) continue;
        if (nw == cap) { cap *= 2; W = realloc(W, cap * sizeof *W); }
        memset(&W[nw], 0, sizeof W[nw]); W[nw].id = id; W[nw].keep = 1; snprintf(W[nw].path, sizeof W[nw].path, "%s/%s", in, de->d_name); nw++;
    }
    closedir(dd);
    if (!nw) { fprintf(stderr, "surf-labels: no wNNN.sfc in %s\n", in); return 1; }
    if (nw > 65534) { fprintf(stderr, "surf-labels: too many wraps\n"); return 1; }
    qsort(W, nw, sizeof *W, sl_cmp_id);
    for (int k = 0; k < nw; k++) { if (sl_open(&W[k])) return 1; if (fabs(W[k].um - W[0].um) > 1e-9) { fprintf(stderr, "surf-labels: wraps at different voxel sizes\n"); return 1; } W[k].peaks = NAN; W[k].onpap = NAN; }
    J.W = W; J.nw = nw; J.s0 = W[0].um / axum; J.f = W[0].um / gum;
    /* per-wrap gates */
    const int k0 = atoi(sl_opt(argc, argv, "--first", "-1000000")), k1 = atoi(sl_opt(argc, argv, "--last", "1000000"));
    for (int k = 0; k < nw; k++) if (W[k].id < k0 || W[k].id > k1) { W[k].keep = 0; snprintf(W[k].why, sizeof W[k].why, "trim"); }
    { const char *dl = sl_opt(argc, argv, "--drop", nullptr);
      if (dl) { char *t = strdup(dl), *sv = nullptr; for (char *q = strtok_r(t, ",", &sv); q; q = strtok_r(nullptr, ",", &sv)) { const int id = atoi(q[0] == 'w' ? q + 1 : q);
                for (int k = 0; k < nw; k++) if (W[k].id == id && W[k].keep) { W[k].keep = 0; snprintf(W[k].why, sizeof W[k].why, "list"); } } free(t); } }
    const char *wcp = sl_opt(argc, argv, "--wrapcheck", nullptr);
    if (wcp) {
        FILE *f = fopen(wcp, "r"); if (!f) { fprintf(stderr, "surf-labels: cannot read %s\n", wcp); return 1; }
        char line[1024];
        while (fgets(line, sizeof line, f)) {
            int id, nint; double em, one, two, rct, rnet, gap; char verdict[32] = "";
            if (line[0] != 'w' || sscanf(line, "w%d %d %lf %lf %lf %lf %lf %lf", &id, &nint, &em, &one, &two, &rct, &rnet, &gap) != 8) continue;
            const char *tab = strrchr(line, '\t'); if (tab) sscanf(tab + 1, "%31s", verdict);
            for (int k = 0; k < nw; k++) if (W[k].id == id) { W[k].has_wc = 1; W[k].peaks = one + 2 * two; snprintf(W[k].verdict, sizeof W[k].verdict, "%s", verdict); }
        }
        fclose(f);
        for (int k = 0; k < nw; k++) if (W[k].keep && W[k].has_wc && (!strcmp(W[k].verdict, "dup") || !strcmp(W[k].verdict, "merged"))) { W[k].keep = 0; snprintf(W[k].why, sizeof W[k].why, "%s", W[k].verdict); }
        double plo = 0, phi = 0;
        if (sl_opt(argc, argv, "--band-peaks", nullptr) && sscanf(sl_opt(argc, argv, "--band-peaks", ""), "%lf,%lf", &plo, &phi) == 2 && phi > plo) {
            for (int k = 0; k < nw; k++) {
                double s = 0; int n = 0; for (int j = k - 2; j <= k + 2; j++) if (j >= 0 && j < nw && isfinite(W[j].peaks) && abs(W[j].id - W[k].id) <= 2) { s += W[j].peaks; n++; }
                if (W[k].keep && n && (s / n < plo || s / n > phi)) { W[k].keep = 0; snprintf(W[k].why, sizeof W[k].why, "peaks %.2f", s / n); }
            }
        }
    } else if (sl_opt(argc, argv, "--band-peaks", nullptr)) { fprintf(stderr, "surf-labels: --band-peaks needs --wrapcheck\n"); return 2; }
    /* holdouts */
    const char *hp = sl_opt(argc, argv, "--exclude", nullptr);
    if (hp) {
        const char *hct = sl_opt(argc, argv, "--exclude-ct", nullptr); const double hmm = atof(sl_opt(argc, argv, "--exclude-mm", "0.5"));
        if (!hct) { fprintf(stderr, "surf-labels: --exclude needs --exclude-ct NAME (the CT path's tail in the holdouts' \"ct\")\n"); return 2; }
        char *buf = nullptr; const int n = sl_read_file(hp, &buf); json *j = n > 0 ? json_parse(buf, (size_t)n) : nullptr; free(buf);
        if (!j) { fprintf(stderr, "surf-labels: cannot parse %s\n", hp); return 1; }
        const json *bx = json_get(j, "boxes"); J.hold = calloc(4096, sizeof *J.hold);
        for (size_t k = 0; bx && json_at(bx, k) && J.nhold < 4096; k++) {
            const json *b = json_at(bx, k); const char *c = json_str(json_get(b, "ct"), ""); const size_t lc = strlen(c), lh = strlen(hct);
            if (lc < lh || strcmp(c + lc - lh, hct)) continue;
            const json *l0 = json_get(b, "l0"); const double m = hmm * 1000 / axum, fa = axum / gum;
            double *h = J.hold[J.nhold++];
            for (int d = 0; d < 3; d++) { h[d] = floor((json_num(json_at(l0, d), 0) - m) * fa); h[3 + d] = ceil((json_num(json_at(l0, d), 0) + json_num(json_at(l0, 3 + d), 0) + m) * fa) - h[d]; }
        }
        json_free(j);
        fprintf(stderr, "surf-labels: %d holdout boxes of %s (+%.2f mm)\n", J.nhold, hct, hmm);
    }
    double t0 = sl_now();
    /* pass 0: row z ranges */
    { sl_scan Sc = {&J, 0, 0}; pthread_t th[64]; const int m = nt < 64 ? nt : 64;
      for (int q = 0; q < m; q++) pthread_create(&th[q], nullptr, sl_scan_run, &Sc);
      for (int q = 0; q < m; q++) pthread_join(th[q], nullptr);
      if (Sc.fail) { fprintf(stderr, "surf-labels: wrap read failed\n"); return 1; } }
    long long nvt = 0; for (int k = 0; k < nw; k++) nvt += W[k].nv;
    if (dir_auto) {   /* the spiral sense: wrap k's theta = pi end meets wrap k + dir's theta = -pi start (W = U - dir theta / 2 pi = k) */
        long vp = 0, vm = 0;
        for (int k = 0; k + 1 < nw; k++) {
            const sl_wrap *a = &W[k], *b = &W[k + 1];
            if (b->id != a->id + 1 || a->h != b->h) continue;
            for (int r = 0; r < a->h; r++) {
                const double p = fabs(a->ra[r] - b->rb[r]), m = fabs(b->ra[r] - a->rb[r]);   /* (NAN: no vote) */
                if (isfinite(p) && isfinite(m)) { if (p < m) vp++; else if (m < p) vm++; }
            }
        }
        if (vp + vm < 10 || (vp > vm ? vp : vm) < 0.8 * (vp + vm)) { fprintf(stderr, "surf-labels: --dir auto undecided (votes +1 %ld, -1 %ld): give --dir\n", vp, vm); return 1; }
        dir = vp > vm ? 1 : -1; J.dir = dir;
        fprintf(stderr, "surf-labels: --dir auto: %+g (cut votes +1 %ld, -1 %ld)\n", dir, vp, vm);
    }
    fprintf(stderr, "surf-labels: %d wraps, %lld vertices, %.4g um wraps -> grid %lldx%lldx%lld at %.4g um, dir %+g, radius %.1f, scan %.1fs\n",
            nw, nvt, W[0].um, (long long)J.shape[0], (long long)J.shape[1], (long long)J.shape[2], gum, dir, rmax, sl_now() - t0);
    /* pass 1 (--min-onpap): the wraps' on-papyrus shares */
    const double minpap = atof(sl_opt(argc, argv, "--min-onpap", "0"));
    if (minpap > 0) {
        if (!J.ct) { fprintf(stderr, "surf-labels: --min-onpap needs --ct\n"); return 2; }
        sl_ctc c = {.z = J.ct}; for (int d = 0; d < 3; d++) { c.sh[d] = J.shape[d]; c.g[d] = (J.shape[d] + 127) / 128; }
        const size_t ng = (size_t)c.g[0] * c.g[1] * c.g[2]; c.ch = calloc(ng, sizeof *c.ch); c.need = calloc(ng, 1);
        if (!c.ch || !c.need) return 1;
        const double tp = sl_now();
        for (int64_t b = 0; b < c.g[0]; b++) {
            J.sz = b;
            if (sl_load_all(&J, b * 128.0 - 2, (b + 1) * 128.0 + 1, nt)) { fprintf(stderr, "surf-labels: wrap read failed\n"); return 1; }
            sl_pap P = {.J = &J, .c = &c, .R = atof(sl_opt(argc, argv, "--ct-range", "15")) / gum, .lev = atof(sl_opt(argc, argv, "--ct-level", "80"))};
            pthread_t th[64]; const int m = nt < 64 ? nt : 64;
            P.mode = 1; for (int q = 0; q < m; q++) pthread_create(&th[q], nullptr, sl_pap_run, &P); for (int q = 0; q < m; q++) pthread_join(th[q], nullptr);
            size_t nn = 0; for (size_t k = 0; k < ng; k++) nn += c.need[k];
            P.todo = malloc((nn ? nn : 1) * sizeof *P.todo); for (size_t k = 0; k < ng; k++) if (c.need[k]) { P.todo[P.ntodo++] = (int64_t)k; c.need[k] = 0; }
            P.mode = 0; for (int q = 0; q < m; q++) pthread_create(&th[q], nullptr, sl_pap_run, &P); for (int q = 0; q < m; q++) pthread_join(th[q], nullptr);
            if (P.fail) { fprintf(stderr, "surf-labels: CT read failed: %s\n", z3_error()); return 1; }
            P.mode = 2; atomic_store(&P.wnext, 0); for (int q = 0; q < m; q++) pthread_create(&th[q], nullptr, sl_pap_run, &P); for (int q = 0; q < m; q++) pthread_join(th[q], nullptr);
            for (size_t k = 0; k < P.ntodo; k++) { free(c.ch[P.todo[k]]); c.ch[P.todo[k]] = nullptr; }
            free(P.todo);
        }
        free(c.ch); free(c.need);
        for (int k = 0; k < nw; k++) { W[k].onpap = W[k].papn ? (double)W[k].pap / W[k].papn : NAN;
            if (W[k].keep && W[k].papn && W[k].onpap < minpap) { W[k].keep = 0; snprintf(W[k].why, sizeof W[k].why, "onpap %.2f", W[k].onpap); } }
        fprintf(stderr, "surf-labels: on-papyrus pass %.1fs\n", sl_now() - tp);
    }
    int nkeep = 0; long long vkeep = 0; for (int k = 0; k < nw; k++) if (W[k].keep) { nkeep++; vkeep += W[k].nv; }
    fprintf(stderr, "surf-labels: %d of %d wraps kept (%.3f of the vertices); dropped:", nkeep, nw, nvt ? (double)vkeep / nvt : 0);
    for (int k = 0; k < nw; k++) if (!W[k].keep) fprintf(stderr, " w%03d(%s)", W[k].id, W[k].why);
    fprintf(stderr, "\n");
    /* the outputs */
    char lv[32], p[1600], attrs[1024]; z3w_level_name(gum, lv, sizeof lv);
    snprintf(p, sizeof p, "mkdir -p '%s'", out); if (system(p)) return 1;
    snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"winding\",\"source\":\"surf-labels %s (%d of %d wraps kept, dir %+g)\",\"encoding\":\"winding_mod14\",\"code\":\"1 + (round(q*18) mod 252); 0 empty; 255 conflict (>1/4 turn)\"}}", in, nkeep, nw, dir);
    snprintf(p, sizeof p, "%s/codes.zarr/%s", out, lv); J.wc = z3w_create(p, J.shape, S, 0, 0, attrs);
    snprintf(p, sizeof p, "%s/codes.zarr", out); if (!J.wc || z3w_write_group(p, &gum, 1, "surf-labels-codes", attrs)) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"band field winding phase (src/band.h PHASE_*: 0..251, 255 unknown), pair rule (band_field_phase_pair), radius %g grid voxels%s\",\"source\":\"surf-labels %s\"}}", rmax, J.bake ? ", unknown where ignore is set" : "", in);
    snprintf(p, sizeof p, "%s/phase.zarr/%s", out, lv); J.wp = z3w_create(p, J.shape, S, 0, PHASE_UNKNOWN, attrs);
    snprintf(p, sizeof p, "%s/phase.zarr", out); if (!J.wp || z3w_write_group(p, &gum, 1, "ufsm-phase", attrs)) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"phase ignore mask: 0 keep, 1 dropped wrap, 2 conflict, 3 masked CT, 4 holdout\",\"source\":\"surf-labels %s\"}}", in);
    snprintf(p, sizeof p, "%s/ignore.zarr/%s", out, lv); J.wi = z3w_create(p, J.shape, S, 0, 0, attrs);
    snprintf(p, sizeof p, "%s/ignore.zarr", out); if (!J.wi || z3w_write_group(p, &gum, 1, "surf-labels-ignore", attrs)) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    J.recs = calloc(nt, sizeof *J.recs);
    J.pa_T = (uint32_t)(4294967295.0 * fmin(1.0, atof(sl_opt(argc, argv, "--pa-keep", "0.02"))));
    /* pass 2: z bands of one shard row */
    const double tl = sl_now(); J.nbs = (S / B) * (S / B) * (S / B);
    for (int64_t sz = 0; sz < J.ns[0]; sz++) {
        J.sz = sz;
        const double z0 = (double)sz * S - J.Hh - 2, z1 = (double)(sz + 1) * S + J.Hh + 1;
        if (sl_load_all(&J, z0, z1, nt)) { fprintf(stderr, "surf-labels: wrap read failed\n"); return 1; }
        /* tiles */
        size_t ntile = 0, tcap = 0; sl_tile *tiles = nullptr;
        for (int k = 0; k < nw; k++) {
            sl_wrap *w = &W[k]; if (w->bn < 2) continue;
            for (int r0 = 0; r0 < w->bn - 1; r0 += SL_TR) for (int c0 = 0; c0 < w->w - 1; c0 += SL_TC) {
                sl_tile t = {k, r0, c0, {1e30f, 1e30f, 1e30f}, {-1e30f, -1e30f, -1e30f}}; int any = 0;
                const int r1 = r0 + SL_TR < w->bn - 1 ? r0 + SL_TR : w->bn - 1, c1 = c0 + SL_TC < w->w - 1 ? c0 + SL_TC : w->w - 1;
                for (int r = r0; r <= r1; r++) for (int c = c0; c <= c1; c++) { const size_t i = (size_t)r * w->w + c; if (!w->v[i]) continue; any = 1;
                    for (int d = 0; d < 3; d++) { const float g = w->g[4 * i + d]; if (g < t.lo[d]) t.lo[d] = g; if (g > t.hi[d]) t.hi[d] = g; } }
                if (!any || t.hi[0] < z0 || t.lo[0] > z1) continue;
                if (ntile == tcap) { tcap = tcap ? 2 * tcap : 4096; tiles = realloc(tiles, tcap * sizeof *tiles); }
                tiles[ntile++] = t;
            }
        }
        if (!ntile) { free(tiles); continue; }
        /* tiles -> shards of the row (extended by the halo) */
        const size_t nsh = (size_t)J.ns[1] * J.ns[2]; size_t *cnt = calloc(nsh + 1, sizeof *cnt);
        for (int pass = 0; pass < 2; pass++) {
            if (pass) { size_t s = 0; for (size_t k = 0; k <= nsh; k++) { const size_t t = cnt[k]; cnt[k] = s; s += t; } J.refs = malloc((cnt[nsh] ? cnt[nsh] : 1) * 4); J.off = malloc((nsh + 1) * sizeof *J.off); memcpy(J.off, cnt, (nsh + 1) * sizeof *cnt); }
            for (size_t t = 0; t < ntile; t++) {
                int64_t a[2], b[2];
                for (int d = 1; d < 3; d++) { a[d - 1] = (int64_t)floor((tiles[t].lo[d] - J.Hh - 2) / S); b[d - 1] = (int64_t)floor((tiles[t].hi[d] + J.Hh + 2) / S);
                    if (a[d - 1] < 0) { a[d - 1] = 0; }
                    if (b[d - 1] > J.ns[d] - 1) b[d - 1] = J.ns[d] - 1; }
                for (int64_t y = a[0]; y <= b[0]; y++) for (int64_t x = a[1]; x <= b[1]; x++) { const size_t s = (size_t)(y * J.ns[2] + x); if (pass) J.refs[cnt[s]++] = (uint32_t)t; else cnt[s]++; }
            }
        }
        free(cnt);
        J.tiles = tiles; J.ntile = ntile;
        J.slist = malloc(nsh * sizeof *J.slist); J.nsl = 0;
        for (size_t s = 0; s < nsh; s++) if (J.off[s + 1] > J.off[s]) J.slist[J.nsl++] = (int)s;
        J.bc = calloc(J.nsl, sizeof *J.bc); J.bp = calloc(J.nsl, sizeof *J.bp); J.bi = calloc(J.nsl, sizeof *J.bi); J.pend = malloc(J.nsl * sizeof *J.pend);
        for (int k = 0; k < J.nsl; k++) atomic_store(&J.pend[k], J.nbs);
        J.nitems = (long)J.nsl * J.nbs; atomic_store(&J.next, 0);
        pthread_t th[64]; void *args[64][2]; const int m = nt < 64 ? nt : 64;
        for (int q = 0; q < m; q++) { args[q][0] = &J; args[q][1] = (void *)(intptr_t)q; pthread_create(&th[q], nullptr, sl_block_run, args[q]); }
        for (int q = 0; q < m; q++) pthread_join(th[q], nullptr);
        free(J.bc); free(J.bp); free(J.bi); free((void *)J.pend); free(J.slist); free(J.refs); free(J.off); free(tiles);
        J.tiles = nullptr; J.refs = nullptr; J.off = nullptr;
        if (atomic_load(&J.fail)) return 1;
        fprintf(stderr, "\rsurf-labels: shard row %lld of %lld: %zu tiles, %lld blocks so far, %.0fs   ", (long long)sz + 1, (long long)J.ns[0], ntile, (long long)J.nblk, sl_now() - tl);
    }
    for (int k = 0; k < nw; k++) { free(W[k].g); free(W[k].v); W[k].g = nullptr; W[k].v = nullptr; }
    if (z3w_close(J.wc) | z3w_close(J.wp) | z3w_close(J.wi)) { fprintf(stderr, "surf-labels: %s\n", z3w_error()); return 1; }
    const double v3 = pow(gum * 1e-3, 3);
    fprintf(stderr, "\n");
    printf("surf-labels: %d of %d wraps kept, %lld coded voxels, %lld conflict; phase known %lld (%.1f mm3), kept %lld (%.1f mm3), kept share %.4f; ignored: dropped wrap %lld, conflict %lld, CT %lld, holdout %lld; %.0fs\n",
           nkeep, nw, (long long)J.ncode, (long long)J.nconf, (long long)J.nknown, J.nknown * v3, (long long)J.nkept, J.nkept * v3, J.nknown ? (double)J.nkept / J.nknown : 0,
           (long long)J.nig[1], (long long)J.nig[2], (long long)J.nig[3], (long long)J.nig[4], sl_now() - t0);
    if (J.gt) {
        printf("surf-labels: vs GT phase: raw %lld voxels known in both, |d| < 0.1 turn %.4f, < 0.25 %.4f (bins %.3f %.3f %.3f %.3f %.3f); kept %lld, < 0.1 %.4f, < 0.25 %.4f (bins %.3f %.3f %.3f %.3f %.3f)\n",
               (long long)J.gtk, J.gtk ? (double)J.g10 / J.gtk : 0, J.gtk ? (double)J.g25 / J.gtk : 0,
               J.gtk ? (double)J.gbins[0] / J.gtk : 0, J.gtk ? (double)J.gbins[1] / J.gtk : 0, J.gtk ? (double)J.gbins[2] / J.gtk : 0, J.gtk ? (double)J.gbins[3] / J.gtk : 0, J.gtk ? (double)J.gbins[4] / J.gtk : 0,
               (long long)J.kgtk, J.kgtk ? (double)J.kg10 / J.kgtk : 0, J.kgtk ? (double)J.kg25 / J.kgtk : 0,
               J.kgtk ? (double)J.kbins[0] / J.kgtk : 0, J.kgtk ? (double)J.kbins[1] / J.kgtk : 0, J.kgtk ? (double)J.kbins[2] / J.kgtk : 0, J.kgtk ? (double)J.kbins[3] / J.kgtk : 0, J.kgtk ? (double)J.kbins[4] / J.kgtk : 0);
        size_t n = 0; for (int t = 0; t < nt; t++) n += J.recs[t].n;
        pa_buf all = {0}; all.r = malloc((n ? n : 1) * sizeof *all.r); all.cap = n ? n : 1; all.T = J.pa_T;
        sl_rec *tmp = malloc((n ? n : 1) * sizeof *tmp); size_t m = 0;
        for (int t = 0; t < nt; t++) { memcpy(tmp + m, J.recs[t].r, J.recs[t].n * sizeof *tmp); m += J.recs[t].n; free(J.recs[t].r); }
        qsort(tmp, n, sizeof *tmp, sl_cmp_rec);   /* (a fixed order for any thread split) */
        for (size_t k = 0; k < n; k++) all.r[k] = (pa_rec){{tmp[k].p[0], tmp[k].p[1], tmp[k].p[2]}, tmp[k].d, tmp[k].h, tmp[k].q};
        all.n = n; free(tmp);
        const double mmv[4] = {0.5, 1, 2, 4}; double a[4], ac[4]; size_t na[4], nc[4];
        pa_score(&all, mmv, 4, 200000, 0x5eed, a, ac, na, nc);
        printf("surf-labels: pair winding right vs GT (%zu records):", all.n);
        for (int k = 0; k < 4; k++) printf(" %g mm %.4f cross %.4f (%zu / %zu)%s", mmv[k], a[k], ac[k], na[k], nc[k], k < 3 ? " |" : "\n");
        free(all.r);
    } else { for (int t = 0; t < nt; t++) free(J.recs[t].r); }
    free(J.recs);
    const char *tsv = sl_opt(argc, argv, "--tsv", nullptr);
    if (tsv) {
        FILE *f = fopen(tsv, "w"); if (!f) { fprintf(stderr, "surf-labels: cannot write %s\n", tsv); return 1; }
        fprintf(f, "wrap\tvertices\tkeep\twhy\tverdict\tpeaks\tonpap\tknown\tkept\tgt_both\tgt_lt10\tgt_lt25\tkept_gt_both\tkept_gt_lt10\tkept_gt_lt25\n");
        for (int k = 0; k < nw; k++) fprintf(f, "w%03d\t%lld\t%d\t%s\t%s\t%.4f\t%.4f\t%lld\t%lld\t%lld\t%lld\t%lld\t%lld\t%lld\t%lld\n", W[k].id, W[k].nv, W[k].keep, W[k].why[0] ? W[k].why : "-",
                                             W[k].verdict[0] ? W[k].verdict : "-", W[k].peaks, W[k].onpap, (long long)W[k].kn, (long long)W[k].kk, (long long)W[k].gk, (long long)W[k].g10, (long long)W[k].g25,
                                             (long long)W[k].kgk, (long long)W[k].kg10, (long long)W[k].kg25);
        fclose(f);
    }
    for (int k = 0; k < nw; k++) { free(W[k].rz0); free(W[k].rz1); free(W[k].ra); free(W[k].rb); sfc_cache_destroy(W[k].ca); sfc_close(W[k].r); }
    free(W); free(J.hold);
    if (J.ct) z3_close(J.ct);
    if (J.gt) z3_close(J.gt);
    return 0;
}
