/* ufsm wrap-snap <wraps-in> <wraps-out> --axis FILE --axis-um U0 --ct ROOT KEY --ct-um U [--fields DIR] [--target ct|phase|recto|mix]
                  [--range 60] [--margin 0.1] [--sigma 480,240,120] [--tau 40,25,15] [--iters 2] [--min-prom 6] [--first K] [--last K]
                  [--band 64] [--sfc 0.25] [--threads N] [--dry] [--cache DIR]
   Wraps (ufsm wraps .sfc: rows = z, columns = rays at one angle around the axis, wrap k = the level set W = k) moved ONTO the sheets.
   A fitted W is smoother than the papyrus and its level sets cross the gaps between sheets (Paris 4 strips ~.75-.79 on papyrus).
   Design (ufsm surf-snap moved vertices to the nearest CT ridge one by one: sheet switches and seams, ~10 um off the GT surfaces):
   - each vertex moves only along its RAY (horizontal, radial from the axis at its z): z and the ray angle stay, so the wrap's grid,
     the strip's columns (unrolled angle) and rows (z) keep their parametrisation; the strip stays square and flat;
   - order is kept by construction: on each ray the neighbouring wraps k - 1 and k + 1 (their radii at that row and angle, from
     angle-binned radius tables) bound wrap k to (1/2 - margin) of the gap to each: intervals of adjacent wraps are disjoint, so
     wraps never cross or merge, and two wraps can never take the same sheet (over-counted packed regions keep both wraps apart);
     where a neighbour is missing, the other side's gap (or --range) is used; never more than --range um;
   - evidence along the ray (samples every 1/2 wrap voxel inside the interval): --target ct = CT ridges (local maxima, prominence
     >= --min-prom grey levels), phase = the fields' pc = 128 + 127 conf cos(phase) maxima (sheet centres of the network),
     recto = the fields' r (recto probability) maxima, mix = CT ridges weighted by the network's conf cos(phase) there (CT trusted
     where the net agrees it is a sheet centre); up to 3 candidates per vertex, weighted by strength x |normal . ray|;
   - a smooth per-wrap offset field d(row, col) is fitted to the candidates, coarse to fine (--sigma um, one per scale, --iters
     IRLS rounds each): every vertex takes the candidate nearest the current field (gaussian gate --tau um per scale, x
     strength), then d = normalised 3-pass box (~gaussian) filter of the chosen offsets, falling back to the coarser field where
     there is no evidence: whole regions pick ONE sheet, no per-vertex jumps, no tears;
   - d is clamped to the vertex's interval and to its unmasked CT run (CT 0 = masked: a vertex never moves into or across it).
   Processing: z bands of --band wrap rows over ALL wraps (each CT / fields chunk decoded once per band; the wraps' rows read from
   their .sfc by region), then the per-wrap solves and .sfc writes on --threads. Deterministic for any thread count. --first / --last:
   only those ids are snapped and written (their neighbours still bound them). --dry: the statistics only. */
#include "sources.h"
#include "store.h"
#include "surfcomp.h"
#include "sysinfo.h"
#include "tiffw.h"
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

#define WSN_K 3      /* candidates per vertex */
#define WSN_Q 8.0    /* offsets stored in 1/8 wrap voxel (int8: +-15.9 voxels) */
#define WSN_NB 8192  /* angle bins of the radius tables */

static const char *wsn_opt(int argc, char **argv, const char *name, const char *def) { for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1]; return def; }
static int wsn_flag(int argc, char **argv, const char *name) { for (int i = 2; i < argc; i++) if (!strcmp(argv[i], name)) return 1; return 0; }
static int wsn_idx(int argc, char **argv, const char *name) { for (int i = 2; i < argc; i++) if (!strcmp(argv[i], name)) return i; return -1; }
static double wsn_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

/* ---- a decoded-chunk table over one uint8 array (CT or a fields plane), filled per band ---- */
typedef struct {
    z3 *z; int64_t sh[3]; int cs[3]; int64_t g[3]; uint8_t **ch; _Atomic uint8_t *need; size_t nres;
    double f, org[3]; uint8_t fill;   /* wrap voxel p -> array voxel (p + .5) f - .5 - org */
} wsn_vol;
static int wsn_vol_open(wsn_vol *v, const char *root, const char *key, const char *cache, double f, const double org[3], uint8_t fill) {
    memset(v, 0, sizeof *v); v->z = z3_open(store_open(root), key, cache); if (!v->z) return -1;
    const z3_meta *m = z3_meta_of(v->z);
    for (int d = 0; d < 3; d++) { v->sh[d] = m->shape[d]; v->cs[d] = m->chunk[d] > 0 ? m->chunk[d] : 128; v->g[d] = (m->shape[d] + v->cs[d] - 1) / v->cs[d]; v->org[d] = org[d]; }
    const size_t ng = (size_t)v->g[0] * v->g[1] * v->g[2];
    v->ch = calloc(ng, sizeof *v->ch); v->need = calloc(ng, 1); v->f = f; v->fill = fill;
    return v->ch && v->need ? 0 : -1;
}
static void wsn_vol_close(wsn_vol *v) {
    if (!v->z) return;
    const size_t ng = (size_t)v->g[0] * v->g[1] * v->g[2];
    for (size_t k = 0; k < ng; k++) free(v->ch[k]);
    free(v->ch); free((void *)v->need); z3_close(v->z); memset(v, 0, sizeof *v);
}
static inline void wsn_a(const wsn_vol *v, const double p[3], double a[3]) { for (int d = 0; d < 3; d++) a[d] = (p[d] + 0.5) * v->f - 0.5 - v->org[d]; }
/* mark the chunks of the box spanned by wrap-voxel points p and q (+-1 array voxel) */
static void wsn_mark(wsn_vol *v, const double p[3], const double q[3]) {
    double a[3], b[3]; wsn_a(v, p, a); wsn_a(v, q, b); int64_t lo[3], hi[3];
    for (int d = 0; d < 3; d++) {
        double l = fmin(a[d], b[d]) - 1, h = fmax(a[d], b[d]) + 2;
        if (h < 0 || l > v->sh[d] - 1) return;
        lo[d] = (int64_t)fmax(0, floor(l)) / v->cs[d]; hi[d] = (int64_t)fmin(v->sh[d] - 1, floor(h)) / v->cs[d];
    }
    for (int64_t z = lo[0]; z <= hi[0]; z++) for (int64_t y = lo[1]; y <= hi[1]; y++) for (int64_t x = lo[2]; x <= hi[2]; x++)
        atomic_store_explicit(&v->need[((size_t)z * v->g[1] + y) * v->g[2] + x], 1, memory_order_relaxed);
}
typedef struct { wsn_vol *v; int64_t *todo; size_t n; _Atomic size_t next; _Atomic int err; } wsn_dec;
static void *wsn_dec_run(void *arg) {
    wsn_dec *J = arg; wsn_vol *v = J->v;
    for (size_t k; (k = atomic_fetch_add(&J->next, 1)) < J->n;) {
        const int64_t id = J->todo[k], cz = id / (v->g[1] * v->g[2]), cy = id / v->g[2] % v->g[1], cx = id % v->g[2];
        int64_t o[3] = {cz * v->cs[0], cy * v->cs[1], cx * v->cs[2]}, n[3];
        for (int d = 0; d < 3; d++) n[d] = v->cs[d];   /* (edge chunks: z3_read pads beyond the shape with the fill) */
        uint8_t *b = malloc((size_t)n[0] * n[1] * n[2]);
        if (!b || z3_read(v->z, o, n, b, 1)) { free(b); atomic_store(&J->err, 1); continue; }
        v->ch[id] = b;
    }
    return nullptr;
}
/* the band's chunks: decode the marked ones not held, drop the held ones not marked; clears the marks */
static int wsn_vol_fill(wsn_vol *v, int nt) {
    if (!v->z) return 0;
    const size_t ng = (size_t)v->g[0] * v->g[1] * v->g[2]; int64_t *todo = nullptr; size_t n = 0, cap = 0;
    for (size_t k = 0; k < ng; k++) {
        if (!v->need[k]) { if (v->ch[k]) { free(v->ch[k]); v->ch[k] = nullptr; v->nres--; } continue; }
        v->need[k] = 0; if (v->ch[k]) continue;
        if (n == cap) { cap = cap ? 2 * cap : 1024; todo = realloc(todo, cap * sizeof *todo); }
        todo[n++] = (int64_t)k;
    }
    wsn_dec J = {v, todo, n, 0, 0}; pthread_t th[64]; const int m = nt < 64 ? nt : 64;
    for (int q = 0; q < m; q++) pthread_create(&th[q], nullptr, wsn_dec_run, &J);
    for (int q = 0; q < m; q++) pthread_join(th[q], nullptr);
    free(todo); v->nres += n;
    return atomic_load(&J.err) ? -1 : 0;
}
static inline double wsn_at(const wsn_vol *v, int64_t z, int64_t y, int64_t x) {
    if (z < 0 || y < 0 || x < 0 || z >= v->sh[0] || y >= v->sh[1] || x >= v->sh[2]) return v->fill;
    const uint8_t *b = v->ch[((size_t)(z / v->cs[0]) * v->g[1] + (size_t)(y / v->cs[1])) * v->g[2] + (size_t)(x / v->cs[2])];
    if (!b) return v->fill;
    return b[((size_t)(z % v->cs[0]) * v->cs[1] + (size_t)(y % v->cs[1])) * v->cs[2] + (size_t)(x % v->cs[2])];
}
static double wsn_tri(const wsn_vol *v, const double p[3]) {   /* trilinear at wrap-voxel point p */
    double a[3]; wsn_a(v, p, a);
    const int64_t z0 = (int64_t)floor(a[0]), y0 = (int64_t)floor(a[1]), x0 = (int64_t)floor(a[2]);
    const double fz = a[0] - z0, fy = a[1] - y0, fx = a[2] - x0; double s = 0;
    for (int c = 0; c < 8; c++)
        s += (c >> 2 & 1 ? fz : 1 - fz) * (c >> 1 & 1 ? fy : 1 - fy) * (c & 1 ? fx : 1 - fx) * wsn_at(v, z0 + (c >> 2 & 1), y0 + (c >> 1 & 1), x0 + (c & 1));
    return s;
}
static double wsn_near(const wsn_vol *v, const double p[3]) { double a[3]; wsn_a(v, p, a); return wsn_at(v, llround(a[0]), llround(a[1]), llround(a[2])); }

/* ---- the wraps ---- */
typedef struct {
    int id, w, h, sel; double um, step; char path[1400]; sfc_reader *r; sfc_cache *ca; uint32_t ch; char *meta;
    int8_t *ct, *lo, *hi; uint8_t *cs, *obl;   /* per vertex: WSN_K candidate offsets (1/8 voxel) + strengths; interval; |n . ray| x 255 */
    float *band; uint8_t *bv; int b0, bn;      /* the band's rows b0 .. b0 + bn - 1 (xyz, valid) */
    float *rt;                                 /* the band's radius table: rows x WSN_NB (NAN: none) */
    struct wsn_rs { double ps; long pn, nv, nc, nm; } *rs;   /* per row (one thread per row: deterministic sums): column pitch, counts */
} wsn_wrap;
typedef struct {
    wsn_wrap *W; int nw; axis *ax; double s0;  /* wrap voxels -> axis (level-0) voxels */
    wsn_vol *ct, *pc, *rv; int target; double R, margin, minprom; int B0, BN; long njobs; _Atomic long next; int mark;
} wsn_job;
enum { T_CT, T_PHASE, T_RECTO, T_MIX };

static inline void wsn_axis(const wsn_job *J, double z, double *cy, double *cx) { axis_at(J->ax, z * J->s0, cy, cx); *cy /= J->s0; *cx /= J->s0; }
static inline const float *wsn_bp(const wsn_wrap *w, int r, int c) { return &w->band[((size_t)(r - w->b0) * w->w + c) * 3]; }   /* x y z */
static inline int wsn_bvalid(const wsn_wrap *w, int r, int c) { return r >= w->b0 && r < w->b0 + w->bn && c >= 0 && c < w->w && w->bv[(size_t)(r - w->b0) * w->w + c]; }
static double wsn_rt(const wsn_wrap *w, int r, double th, int B0) {   /* the radius of wrap w at band row r, angle th */
    if (!w || !w->rt) return NAN;
    int b = (int)floor((th + M_PI) / (2 * M_PI) * WSN_NB); b = ((b % WSN_NB) + WSN_NB) % WSN_NB;
    return w->rt[(size_t)(r - B0) * WSN_NB + b];
}
/* the band's radius tables (one job per wrap): mean radius per angle bin, holes up to 16 bins filled linearly (circular) */
static void wsn_table(const wsn_job *J, wsn_wrap *w) {
    const int BN = J->BN; float *t = w->rt; double *sum = calloc(WSN_NB, sizeof(double)); int *cnt = calloc(WSN_NB, sizeof(int));
    for (int r = J->B0; r < J->B0 + BN; r++) {
        memset(sum, 0, WSN_NB * sizeof(double)); memset(cnt, 0, WSN_NB * sizeof(int)); float *row = &t[(size_t)(r - J->B0) * WSN_NB];
        if (r < w->h) for (int c = 0; c < w->w; c++) {
            if (!wsn_bvalid(w, r, c)) continue;
            const float *p = wsn_bp(w, r, c); double cy, cx; wsn_axis(J, p[2], &cy, &cx);
            const double dy = p[1] - cy, dx = p[0] - cx, th = atan2(dy, dx);
            int b = (int)floor((th + M_PI) / (2 * M_PI) * WSN_NB); b = ((b % WSN_NB) + WSN_NB) % WSN_NB;
            sum[b] += sqrt(dy * dy + dx * dx); cnt[b]++;
        }
        int any = 0; for (int b = 0; b < WSN_NB; b++) { row[b] = cnt[b] ? (float)(sum[b] / cnt[b]) : NAN; any |= cnt[b] > 0; }
        if (!any) continue;
        for (int b = 0; b < WSN_NB; b++) {
            if (cnt[b]) continue;
            int l = 1, rr = 1; while (l <= 16 && !cnt[((b - l) % WSN_NB + WSN_NB) % WSN_NB]) l++;
            while (rr <= 16 && !cnt[(b + rr) % WSN_NB]) rr++;
            if (l <= 16 && rr <= 16) { const float a = (float)(sum[((b - l) % WSN_NB + WSN_NB) % WSN_NB] / cnt[((b - l) % WSN_NB + WSN_NB) % WSN_NB]), c = (float)(sum[(b + rr) % WSN_NB] / cnt[(b + rr) % WSN_NB]);
                                       row[b] = a + (c - a) * l / (float)(l + rr); }
        }
    }
    free(sum); free(cnt);
}
static int cmp_cand(const void *a, const void *b) { const float *x = a, *y = b; return x[1] < y[1] ? 1 : x[1] > y[1] ? -1 : (x[0] > y[0]) - (x[0] < y[0]); }
/* one band row of one wrap: intervals, obliquity, candidates (J->mark: only mark the chunks the rays read) */
static void wsn_row(const wsn_job *J, int wi, int r) {
    wsn_wrap *w = &J->W[wi]; if (r >= w->h) return;
    const wsn_wrap *A = wi > 0 && J->W[wi - 1].id == w->id - 1 ? &J->W[wi - 1] : nullptr, *Bw = wi + 1 < J->nw && J->W[wi + 1].id == w->id + 1 ? &J->W[wi + 1] : nullptr;
    const double R = J->R; const int ns = (int)ceil(2 * R / 0.5) + 8; double *prof = malloc(ns * sizeof(double)), *pcp = malloc(ns * sizeof(double));
    for (int c = 0; c < w->w; c++) {
        if (!wsn_bvalid(w, r, c)) continue;
        const size_t i = (size_t)r * w->w + c; const float *p = wsn_bp(w, r, c);
        double cy, cx; wsn_axis(J, p[2], &cy, &cx);
        const double dy = p[1] - cy, dx = p[0] - cx, rad = sqrt(dy * dy + dx * dx); if (!(rad > 1e-6)) continue;
        const double u[3] = {0, dy / rad, dx / rad}, P[3] = {p[2], p[1], p[0]}, th = atan2(dy, dx);   /* (z, y, x) */
        if (J->mark) {
            const double a[3] = {P[0], P[1] - R * u[1], P[2] - R * u[2]}, b[3] = {P[0], P[1] + R * u[1], P[2] + R * u[2]};
            if (J->ct) wsn_mark(J->ct, a, b);
            if (J->pc) wsn_mark(J->pc, a, b);
            if (J->rv) wsn_mark(J->rv, a, b);
            continue;
        }
        /* the normal (grid differences) and the obliquity |n . u| */
        const int cl = wsn_bvalid(w, r, c - 1) ? c - 1 : c, cr = wsn_bvalid(w, r, c + 1) ? c + 1 : c;
        const int ru = wsn_bvalid(w, r - 1, c) ? r - 1 : r, rd = wsn_bvalid(w, r + 1, c) ? r + 1 : r;
        double ob = 0;
        if (cl != cr && ru != rd) {
            const float *a = wsn_bp(w, r, cl), *b = wsn_bp(w, r, cr), *uu = wsn_bp(w, ru, c), *dd = wsn_bp(w, rd, c);
            const double du[3] = {b[2] - a[2], b[1] - a[1], b[0] - a[0]}, dv[3] = {dd[2] - uu[2], dd[1] - uu[1], dd[0] - uu[0]};
            const double n[3] = {du[1] * dv[2] - du[2] * dv[1], du[2] * dv[0] - du[0] * dv[2], du[0] * dv[1] - du[1] * dv[0]}, nl = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            if (nl > 0) ob = fabs(n[1] * u[1] + n[2] * u[2]) / nl;
            if (cr - cl == 2) { w->rs[r].ps += sqrt(du[0] * du[0] + du[1] * du[1] + du[2] * du[2]) / 2; w->rs[r].pn++; }
        }
        w->obl[i] = (uint8_t)lround(fmin(1, ob) * 255);
        w->rs[r].nv++;
        /* the interval: (1/2 - margin) of the radial gap to the neighbouring wraps on this ray */
        const double ra = wsn_rt(A, r, th, J->B0), rb = wsn_rt(Bw, r, th, J->B0);
        double gin = NAN, gout = NAN;   /* gaps inward (smaller radius) and outward */
        const double nb[2] = {ra, rb};
        for (int k = 0; k < 2; k++) if (isfinite(nb[k])) { const double g = nb[k] - rad;
            if (g < 0 && (!isfinite(gin) || -g < gin)) gin = -g; else if (g > 0 && (!isfinite(gout) || g < gout)) gout = g; }
        if (!isfinite(gin) && isfinite(gout)) gin = gout;
        if (!isfinite(gout) && isfinite(gin)) gout = gin;
        const double f = 0.5 - J->margin;
        double tlo = isfinite(gin) ? -f * gin : -R, thi = isfinite(gout) ? f * gout : R;
        tlo = fmax(tlo, -R); thi = fmin(thi, R);
        /* the unmasked CT run around the vertex (CT 0 = masked): never into or across it */
        if (J->ct) {
            double q[3] = {P[0], P[1], P[2]};
            if (wsn_near(J->ct, q) <= 0) { tlo = thi = 0; w->rs[r].nm++; }
            else {
                for (double t = 0.5; t <= thi + 1e-9; t += 0.5) { q[1] = P[1] + t * u[1]; q[2] = P[2] + t * u[2]; if (wsn_near(J->ct, q) <= 0) { thi = t - 0.5; break; } }
                for (double t = 0.5; t <= -tlo + 1e-9; t += 0.5) { q[1] = P[1] - t * u[1]; q[2] = P[2] - t * u[2]; if (wsn_near(J->ct, q) <= 0) { tlo = -(t - 0.5); break; } }
            }
        }
        w->lo[i] = (int8_t)lround(fmax(-127, ceil(tlo * WSN_Q))); w->hi[i] = (int8_t)lround(fmin(127, floor(thi * WSN_Q)));
        if (!(thi - tlo >= 0.5) || ob < 0.2) continue;
        /* the evidence profile over [tlo - 1, thi + 1] every 0.5 voxel */
        const double t0 = tlo - 1; int n = (int)floor((thi + 1 - t0) / 0.5) + 1; if (n > ns) n = ns;
        const wsn_vol *ev = J->target == T_PHASE ? J->pc : J->target == T_RECTO ? J->rv : J->ct;
        for (int s = 0; s < n; s++) {
            const double t = t0 + 0.5 * s, q[3] = {P[0], P[1] + t * u[1], P[2] + t * u[2]};
            prof[s] = wsn_tri(ev, q); if (J->target == T_MIX) pcp[s] = (wsn_tri(J->pc, q) - 128) / 127.0;
        }
        if (getenv("UFSM_WSN_DEBUG") && c % 997 == 0 && r % 61 == 0) {
            fprintf(stderr, "dbg w%03d r%d c%d P %.1f %.1f %.1f [%.2f %.2f]:", w->id, r, c, P[0], P[1], P[2], tlo, thi);
            for (int s = 0; s < n; s++) fprintf(stderr, " %.0f", prof[s]);
            fputc('\n', stderr);
        }
        float cand[64][2]; int nc = 0;
        for (int s = 1; s + 1 < n && nc < 64; s++) {
            const double t = t0 + 0.5 * s; if (t < tlo - 1e-9 || t > thi + 1e-9) continue;
            if (!(prof[s] >= prof[s - 1] && prof[s] > prof[s + 1])) continue;
            double str;
            if (J->target == T_PHASE) { str = (prof[s] - 128) / 127.0; if (str < 0.15) continue; str *= 255; }
            else if (J->target == T_RECTO) { if (prof[s] < 50) continue; str = prof[s]; }
            else {   /* CT: prominence over the lower of the two flanking minima (each up to the next higher sample) */
                double lm = prof[s], rm = prof[s];
                for (int k = s - 1; k >= 0 && prof[k] <= prof[s]; k--) if (prof[k] < lm) lm = prof[k];
                for (int k = s + 1; k < n && prof[k] <= prof[s]; k++) if (prof[k] < rm) rm = prof[k];
                const double pr = prof[s] - fmax(lm, rm); if (pr < J->minprom) continue;
                str = fmin(255, pr * 4);
                if (J->target == T_MIX) str *= fmin(1, fmax(0.1, pcp[s]));
            }
            const double den = prof[s - 1] - 2 * prof[s] + prof[s + 1];
            double dt = den < 0 ? 0.5 * 0.5 * (prof[s - 1] - prof[s + 1]) / den : 0; if (dt > 0.25) dt = 0.25; if (dt < -0.25) dt = -0.25;
            cand[nc][0] = (float)fmin(thi, fmax(tlo, t + dt)); cand[nc][1] = (float)str; nc++;
        }
        if (!nc) continue;
        qsort(cand, nc, sizeof cand[0], cmp_cand);
        for (int k = 0; k < WSN_K && k < nc; k++) { w->ct[i * WSN_K + k] = (int8_t)lround(fmax(-127, fmin(127, cand[k][0] * WSN_Q))); w->cs[i * WSN_K + k] = (uint8_t)fmax(1, fmin(255, lround(cand[k][1]))); }
        w->rs[r].nc++;
    }
    free(prof); free(pcp);
}
static void *wsn_rows(void *arg) {
    wsn_job *J = arg;
    for (long g; (g = atomic_fetch_add(&J->next, 1)) < J->njobs;) {
        const int wi = (int)(g / J->BN), r = J->B0 + (int)(g % J->BN);
        if (J->W[wi].sel && J->W[wi].bn) wsn_row(J, wi, r);
    }
    return nullptr;
}
typedef struct { wsn_job *J; _Atomic int next; int mode; } wsn_wjob;   /* mode 0: read the band rows, 1: radius tables */
static int wsn_read_rows(wsn_wrap *w, int r0, int n) {
    w->b0 = r0; w->bn = n; if (n <= 0) return 0;
    memset(w->bv, 0, (size_t)n * w->w);
    return sfc_cache_read_xyz_region(w->ca, w->ch, 0, (uint64_t)r0, (uint32_t)w->w, (uint32_t)n, w->band, 12, (size_t)w->w * 12, w->bv) ? -1 : 0;
}
static void *wsn_wrun(void *arg) {
    wsn_wjob *Q = arg; wsn_job *J = Q->J;
    for (int k; (k = atomic_fetch_add(&Q->next, 1)) < J->nw;) {
        wsn_wrap *w = &J->W[k];
        if (Q->mode == 0) {
            if (!w->r) { w->bn = 0; continue; }
            int r0 = J->B0 - 1, r1 = J->B0 + J->BN + 1; if (r0 < 0) r0 = 0; if (r1 > w->h) r1 = w->h;
            if (wsn_read_rows(w, r0, r1 - r0)) { fprintf(stderr, "wrap-snap: cannot read rows %d..%d of %s\n", r0, r1, w->path); w->bn = 0; }
            for (size_t i = 0; i < (size_t)w->bn * w->w; i++) if (w->bv[i] && !(w->band[3 * i + 2] >= 0)) w->bv[i] = 0;
        } else if (w->r && w->bn) wsn_table(J, w);
    }
    return nullptr;
}

/* ---- the per-wrap smooth offset field ---- */
/* 3 passes of a normalised box of radius rad along rows (dir 0) or columns (dir 1) of a h x w grid, in place */
static void wsn_box(float *a, int w, int h, int rad, int dir, double *buf, double *tmp) {
    if (rad < 1) return;
    const int L = dir ? h : w, M = dir ? w : h;
    for (int m = 0; m < M; m++) {
        for (int l = 0; l < L; l++) buf[l] = a[dir ? (size_t)l * w + m : (size_t)m * w + l];
        for (int pass = 0; pass < 3; pass++) {
            double s = 0; int lo = 0, hi = -1;
            for (int l = 0; l < L; l++) {
                const int a0 = l - rad < 0 ? 0 : l - rad, a1 = l + rad > L - 1 ? L - 1 : l + rad;
                while (hi < a1) s += buf[++hi];
                while (lo < a0) s -= buf[lo++];
                tmp[l] = s / (2 * rad + 1);
            }
            memcpy(buf, tmp, L * sizeof(double));
        }
        for (int l = 0; l < L; l++) a[dir ? (size_t)l * w + m : (size_t)m * w + l] = (float)buf[l];
    }
}
typedef struct { double sig[8], tau[8]; int nsig, iters; double lam, s0; axis *ax; const char *out; int dry; _Atomic int next, fail; wsn_wrap *W; int nw;
                 _Atomic long long nmov, nclamp, nval; _Atomic long long sumabs_q; } wsn_solve;
static void wsn_solve_one(wsn_solve *S, wsn_wrap *w) {
    const int W = w->w, H = w->h; const size_t nv = (size_t)W * H;
    float *D = calloc(nv, 4), *num = malloc(nv * 4), *den = malloc(nv * 4); const int L = W > H ? W : H; double *buf = malloc(L * sizeof(double)), *tmp = malloc(L * sizeof(double));
    double ps = 0; long pn = 0; for (int r = 0; r < H; r++) { ps += w->rs[r].ps; pn += w->rs[r].pn; }
    const double pitch = pn ? ps / pn : w->step, rowp = w->step;   /* wrap voxels */
    for (int s = 0; s < S->nsig; s++) {
        const double sv = S->sig[s] / w->um, tau = S->tau[s] / w->um;   /* in wrap voxels */
        const int rc = (int)lround(sv / pitch), rr = (int)lround(sv / rowp);
        for (int it = 0; it < S->iters; it++) {
            for (size_t i = 0; i < nv; i++) {
                num[i] = den[i] = 0; if (!w->cs[i * WSN_K]) continue;
                double best = 0, bt = 0;
                for (int k = 0; k < WSN_K && w->cs[i * WSN_K + k]; k++) {
                    const double t = w->ct[i * WSN_K + k] / WSN_Q, e = (t - D[i]) / tau, v = w->cs[i * WSN_K + k] / 255.0 * exp(-0.5 * e * e);
                    if (v > best) { best = v; bt = t; }
                }
                const double wt = best * w->obl[i] / 255.0; num[i] = (float)(wt * bt); den[i] = (float)wt;
            }
            wsn_box(num, W, H, rc, 0, buf, tmp); wsn_box(num, W, H, rr, 1, buf, tmp);
            wsn_box(den, W, H, rc, 0, buf, tmp); wsn_box(den, W, H, rr, 1, buf, tmp);
            for (size_t i = 0; i < nv; i++) D[i] = (float)((num[i] + S->lam * D[i]) / (den[i] + S->lam));
        }
    }
    free(num); free(den); free(buf); free(tmp);
    /* the final offsets, clamped to the intervals; the vertices moved along their rays */
    float *X = malloc(nv * 12); uint8_t *V = malloc(nv);
    if (!X || !V || sfc_cache_read_xyz_region(w->ca, w->ch, 0, 0, (uint32_t)W, (uint32_t)H, X, 12, (size_t)W * 12, V)) { fprintf(stderr, "wrap-snap: cannot read %s\n", w->path); atomic_store(&S->fail, 1); free(X); free(V); free(D); return; }
    float *x = malloc(nv * 4), *y = malloc(nv * 4), *z = malloc(nv * 4); long long mov = 0, cl = 0, val = 0, sq = 0;
    for (size_t i = 0; i < nv; i++) {
        x[i] = y[i] = z[i] = -1.f; if (!V[i] || !(X[3 * i + 2] >= 0)) continue;
        x[i] = X[3 * i]; y[i] = X[3 * i + 1]; z[i] = X[3 * i + 2]; val++;
        double d = D[i]; const double lo = w->lo[i] / WSN_Q, hi = w->hi[i] / WSN_Q;
        if (d < lo) { d = lo; cl++; } else if (d > hi) { d = hi; cl++; }
        if (d == 0) continue;
        double cy, cx; axis_at(S->ax, z[i] * S->s0, &cy, &cx); cy /= S->s0; cx /= S->s0;
        const double dy = y[i] - cy, dx = x[i] - cx, rad = sqrt(dy * dy + dx * dx); if (!(rad > 1e-6)) continue;
        y[i] = (float)(y[i] + d * dy / rad); x[i] = (float)(x[i] + d * dx / rad); mov += fabs(d) >= 0.5 / WSN_Q; sq += llround(fabs(d) * 1000);
    }
    atomic_fetch_add(&S->nmov, mov); atomic_fetch_add(&S->nclamp, cl); atomic_fetch_add(&S->nval, val); atomic_fetch_add(&S->sumabs_q, sq);
    if (!S->dry) {
        char p[1500]; snprintf(p, sizeof p, "%s/w%03d", S->out, w->id);
        int winding = w->id; const char *q = w->meta ? strstr(w->meta, "\"winding\":") : nullptr; if (q) winding = atoi(q + 10);
        if (tifxyz_write(p, W, H, x, y, z, w->step, w->um, winding, "ufsm wrap-snap (wraps moved along their rays onto the sheets)")) { fprintf(stderr, "wrap-snap: cannot write %s\n", p); atomic_store(&S->fail, 1); }
    }
    free(X); free(V); free(x); free(y); free(z); free(D);
}
static void *wsn_solve_run(void *arg) {
    wsn_solve *S = arg;
    for (int k; (k = atomic_fetch_add(&S->next, 1)) < S->nw;) if (S->W[k].sel && S->W[k].r) wsn_solve_one(S, &S->W[k]);
    return nullptr;
}
static int wsn_cmp_id(const void *a, const void *b) { const wsn_wrap *x = a, *y = b; return (x->id > y->id) - (x->id < y->id); }
static int wsn_list(const char *s, double *v, int max) { int n = 0; char *t = strdup(s), *sv = nullptr; for (char *q = strtok_r(t, ",", &sv); q && n < max; q = strtok_r(nullptr, ",", &sv)) v[n++] = atof(q); free(t); return n; }

int cmd_wrap_snap(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm wrap-snap <wraps-in> <wraps-out> --axis FILE --axis-um U0 --ct ROOT KEY --ct-um U [--fields DIR] [--target ct|phase|recto|mix]\n"
                                    "       [--range 60] [--margin 0.1] [--sigma 480,240,120] [--tau 40,25,15] [--iters 2] [--min-prom 6] [--first K] [--last K] [--band 64] [--sfc 0.25] [--threads N] [--dry] [--cache DIR]\n"); return 2; }
    const char *in = argv[2], *out = argv[3], *tg = wsn_opt(argc, argv, "--target", "ct");
    const int target = !strcmp(tg, "ct") ? T_CT : !strcmp(tg, "phase") ? T_PHASE : !strcmp(tg, "recto") ? T_RECTO : !strcmp(tg, "mix") ? T_MIX : -1;
    const int nt = ufsm_threads(atoi(wsn_opt(argc, argv, "--threads", "0")), 16), dry = wsn_flag(argc, argv, "--dry");
    const double ctum = atof(wsn_opt(argc, argv, "--ct-um", "0")), axum = atof(wsn_opt(argc, argv, "--axis-um", "0")), range_um = atof(wsn_opt(argc, argv, "--range", "60"));
    const char *axp = wsn_opt(argc, argv, "--axis", nullptr), *fd = wsn_opt(argc, argv, "--fields", nullptr), *cache = wsn_opt(argc, argv, "--cache", nullptr);
    const int ci = wsn_idx(argc, argv, "--ct"), BR = atoi(wsn_opt(argc, argv, "--band", "64"));
    if (target < 0 || !axp || !(axum > 0) || ci < 0 || ci + 2 >= argc || !(ctum > 0) || BR < 1 || !(range_um > 0)) { fprintf(stderr, "wrap-snap: --target ct|phase|recto|mix, --axis FILE --axis-um U0 (its level-0 um), --ct ROOT KEY --ct-um U, --band >= 1, --range > 0\n"); return 2; }
    if ((target == T_PHASE || target == T_RECTO || target == T_MIX) && !fd) { fprintf(stderr, "wrap-snap: --target %s needs --fields DIR\n", tg); return 2; }
    wsn_solve S; memset(&S, 0, sizeof S);
    S.nsig = wsn_list(wsn_opt(argc, argv, "--sigma", "480,240,120"), S.sig, 8); const int ntau = wsn_list(wsn_opt(argc, argv, "--tau", "40,25,15"), S.tau, 8);
    for (int s = ntau; s < S.nsig; s++) S.tau[s] = ntau ? S.tau[ntau - 1] : 2;
    S.iters = atoi(wsn_opt(argc, argv, "--iters", "2")); S.lam = atof(wsn_opt(argc, argv, "--lambda", "0.02"));
    if (S.nsig < 1 || S.iters < 1) { fprintf(stderr, "wrap-snap: --sigma and --iters\n"); return 2; }
    axis ax = {0}; if (axis_load(&ax, axp)) { fprintf(stderr, "wrap-snap: cannot load the axis %s\n", axp); return 1; }
    const int k0 = atoi(wsn_opt(argc, argv, "--first", "-1000000")), k1 = atoi(wsn_opt(argc, argv, "--last", "1000000"));
    /* the wraps: every wNNN.sfc (the selected ids and their neighbours are opened) */
    DIR *dd = opendir(in); if (!dd) { fprintf(stderr, "wrap-snap: cannot open %s\n", in); return 1; }
    int nw = 0, cap = 64; wsn_wrap *W = calloc(cap, sizeof *W); struct dirent *de;
    while ((de = readdir(dd))) {
        int id; char tail[8] = ""; if (sscanf(de->d_name, "w%d.%7s", &id, tail) != 2 || strcmp(tail, "sfc")) continue;
        if (id < k0 - 1 || id > k1 + 1) continue;
        if (nw == cap) { cap *= 2; W = realloc(W, cap * sizeof *W); }
        memset(&W[nw], 0, sizeof W[nw]); W[nw].id = id; W[nw].sel = id >= k0 && id <= k1; snprintf(W[nw].path, sizeof W[nw].path, "%s/%s", in, de->d_name); nw++;
    }
    closedir(dd);
    if (!nw) { fprintf(stderr, "wrap-snap: no wNNN.sfc wraps in %s\n", in); return 1; }
    qsort(W, nw, sizeof *W, wsn_cmp_id);
    double wum = 0; int H = -1;
    for (int k = 0; k < nw; k++) {
        wsn_wrap *w = &W[k];
        if (sfc_open_file(w->path, &w->r)) { fprintf(stderr, "wrap-snap: cannot open %s\n", w->path); return 1; }
        int c = sfc_find_channel(w->r, "x"); if (c < 0) c = 0; w->ch = (uint32_t)c;
        const sfc_channel *ch = sfc_channel_info(w->r, w->ch); w->w = (int)ch->width; w->h = (int)ch->height;
        const uint64_t ml = sfc_metadata_size(w->r); w->meta = calloc(ml + 1, 1); if (ml && sfc_read_metadata(w->r, w->meta, ml)) { fprintf(stderr, "wrap-snap: no metadata in %s\n", w->path); return 1; }
        const char *q = strstr(w->meta, "\"voxel_size_um\":"); w->um = q ? atof(q + 16) : 0;
        q = strstr(w->meta, "\"scale\":"); if (q) { q = strchr(q, '['); if (q && atof(q + 1) > 0) w->step = 1 / atof(q + 1); }
        if (!(w->um > 0) || !(w->step > 0)) { fprintf(stderr, "wrap-snap: %s: no voxel_size_um / scale in its meta\n", w->path); return 1; }
        if (wum > 0 && fabs(w->um - wum) > 1e-9) { fprintf(stderr, "wrap-snap: wraps at different voxel sizes (%g, %g um)\n", wum, w->um); return 1; }
        if (H >= 0 && w->h != H) { fprintf(stderr, "wrap-snap: %s has %d rows, others %d: wraps of one ufsm wraps run only (rows = the same z)\n", w->path, w->h, H); return 1; }
        wum = w->um; H = w->h;
        const size_t budget = (size_t)((w->w + 63) / 64 + 1) * 3 * 64 * 64 * 12 * 2;
        if (sfc_cache_create(w->r, budget, &w->ca)) { fprintf(stderr, "wrap-snap: no memory\n"); return 1; }
        w->band = malloc((size_t)(BR + 2) * w->w * 12); w->bv = malloc((size_t)(BR + 2) * w->w); w->rt = malloc((size_t)BR * WSN_NB * sizeof(float));
        if (w->sel) { const size_t nv = (size_t)w->w * w->h; w->ct = calloc(nv * WSN_K, 1); w->cs = calloc(nv * WSN_K, 1); w->lo = calloc(nv, 1); w->hi = calloc(nv, 1); w->obl = calloc(nv, 1); w->rs = calloc(w->h, sizeof *w->rs);
                      if (!w->ct || !w->cs || !w->lo || !w->hi || !w->obl || !w->rs) { fprintf(stderr, "wrap-snap: no memory for %s (%zu vertices)\n", w->path, nv); return 1; } }
        if (!w->band || !w->bv || !w->rt) { fprintf(stderr, "wrap-snap: no memory\n"); return 1; }
    }
    const double R = fmin(range_um / wum, 15.0);
    /* the volumes */
    wsn_vol vct, vpc, vr; memset(&vpc, 0, sizeof vpc); memset(&vr, 0, sizeof vr);
    { const double o0[3] = {0, 0, 0}; if (wsn_vol_open(&vct, argv[ci + 1], argv[ci + 2], cache, wum / ctum, o0, 0)) { fprintf(stderr, "wrap-snap: CT: %s\n", z3_error()); return 1; } }
    if (fd && target != T_CT) {
        char jp[1400]; snprintf(jp, sizeof jp, "%s/fields.json", fd); FILE *jf = fopen(jp, "rb"); char jb[4096] = {0};
        if (!jf) { fprintf(stderr, "wrap-snap: no %s\n", jp); return 1; }
        size_t nr = fread(jb, 1, sizeof jb - 1, jf); (void)nr; fclose(jf);
        double org[3] = {0, 0, 0}, fum = 0; const char *q = strstr(jb, "\"origin\":"); if (q) sscanf(strchr(q, '[') + 1, "%lf,%lf,%lf", org, org + 1, org + 2);
        q = strstr(jb, "\"um\":"); if (q) fum = atof(q + 5);
        { const char *hq = strstr(jb, "\"h\":"); const double hh = hq ? atof(hq + 4) : 1; if (hh > 0) { fum *= hh; for (int d = 0; d < 3; d++) org[d] /= hh; } }   /* cells of h native voxels (um = the native um), origin in native voxels */
        if (!(fum > 0)) { fprintf(stderr, "wrap-snap: no um in %s\n", jp); return 1; }
        if ((target == T_PHASE || target == T_MIX) && wsn_vol_open(&vpc, fd, "pc", cache, wum / fum, org, 128)) { fprintf(stderr, "wrap-snap: fields pc: %s\n", z3_error()); return 1; }
        if (target == T_RECTO && wsn_vol_open(&vr, fd, "r", cache, wum / fum, org, 0)) { fprintf(stderr, "wrap-snap: fields r: %s (no recto plane?)\n", z3_error()); return 1; }
    }
    if (!dry) { char cmd[1500]; snprintf(cmd, sizeof cmd, "mkdir -p '%s'", out); if (system(cmd)) return 1; if (atof(wsn_opt(argc, argv, "--sfc", "0.25")) > 0) tifxyz_set_sfc(atof(wsn_opt(argc, argv, "--sfc", "0.25"))); }
    tfx_set_threads(nt);
    double t0 = wsn_now(), tdec = 0; int nsel = 0; for (int k = 0; k < nw; k++) nsel += W[k].sel;
    fprintf(stderr, "wrap-snap: %d wraps (%d snapped), %d rows, %.4g um, target %s, range %.1f voxels, bands of %d rows, %d threads\n", nw, nsel, H, wum, tg, R, BR, nt);
    wsn_job J = {W, nw, &ax, wum / axum, &vct, vpc.z ? &vpc : nullptr, vr.z ? &vr : nullptr, target, R, atof(wsn_opt(argc, argv, "--margin", "0.1")), atof(wsn_opt(argc, argv, "--min-prom", "6")), 0, 0, 0, 0, 0};
    pthread_t th[64]; const int m = nt < 64 ? nt : 64;
    for (int b0 = 0; b0 < H; b0 += BR) {
        J.B0 = b0; J.BN = b0 + BR <= H ? BR : H - b0; J.njobs = (long)nw * J.BN;
        wsn_wjob Q = {&J, 0, 0};
        for (int q = 0; q < m; q++) pthread_create(&th[q], nullptr, wsn_wrun, &Q);
        for (int q = 0; q < m; q++) pthread_join(th[q], nullptr);
        Q.mode = 1; atomic_store(&Q.next, 0);
        for (int q = 0; q < m; q++) pthread_create(&th[q], nullptr, wsn_wrun, &Q);
        for (int q = 0; q < m; q++) pthread_join(th[q], nullptr);
        J.mark = 1; atomic_store(&J.next, 0);
        for (int q = 0; q < m; q++) pthread_create(&th[q], nullptr, wsn_rows, &J);
        for (int q = 0; q < m; q++) pthread_join(th[q], nullptr);
        const double td = wsn_now();
        if (wsn_vol_fill(&vct, nt) || wsn_vol_fill(&vpc, nt) || wsn_vol_fill(&vr, nt)) { fprintf(stderr, "wrap-snap: volume read: %s\n", z3_error()); return 1; }
        tdec += wsn_now() - td;
        J.mark = 0; atomic_store(&J.next, 0);
        for (int q = 0; q < m; q++) pthread_create(&th[q], nullptr, wsn_rows, &J);
        for (int q = 0; q < m; q++) pthread_join(th[q], nullptr);
    }
    wsn_vol_close(&vct); wsn_vol_close(&vpc); wsn_vol_close(&vr);
    long nval = 0, ncand = 0, nmask = 0; for (int k = 0; k < nw; k++) { for (int r = 0; W[k].rs && r < W[k].h; r++) { nval += W[k].rs[r].nv; ncand += W[k].rs[r].nc; nmask += W[k].rs[r].nm; } free(W[k].band); free(W[k].bv); free(W[k].rt); W[k].band = nullptr; W[k].bv = nullptr; W[k].rt = nullptr; }
    const double t1 = wsn_now();
    fprintf(stderr, "wrap-snap: evidence %.1f s (volume decode %.1f s): %ld vertices, %.3f with a candidate, %.4f on masked CT (not moved)\n", t1 - t0, tdec, nval, nval ? (double)ncand / nval : 0, nval ? (double)nmask / nval : 0);
    S.s0 = wum / axum; S.ax = &ax; S.out = out; S.dry = dry; S.W = W; S.nw = nw;
    for (int q = 0; q < m; q++) pthread_create(&th[q], nullptr, wsn_solve_run, &S);
    for (int q = 0; q < m; q++) pthread_join(th[q], nullptr);
    const long long nv = S.nval;
    fprintf(stderr, "wrap-snap: solve + write %.1f s: %d wraps, %lld vertices: moved %.3f, mean |shift| %.2f um, clamped to the interval %.3f -> %s\n", wsn_now() - t1, nsel, nv,
            nv ? (double)S.nmov / nv : 0, nv ? (double)S.sumabs_q / 1000 / nv * wum : 0, nv ? (double)S.nclamp / nv : 0, dry ? "(dry)" : out);
    for (int k = 0; k < nw; k++) { wsn_wrap *w = &W[k]; free(w->ct); free(w->cs); free(w->lo); free(w->hi); free(w->obl); free(w->rs); free(w->meta); if (w->ca) sfc_cache_destroy(w->ca); if (w->r) sfc_close(w->r); }
    free(W);
    return atomic_load(&S.fail) ? 1 : 0;
}
