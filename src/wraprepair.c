/* ufsm wrap-repair <wraps-in> <wraps-out | -> --axis FILE --axis-um U0 --ct ROOT KEY --ct-um U [--codes ROOT KEY] [--nray 720]
                    [--row-step 8] [--sigma 1.2] [--prom 1] [--shift 0.5] [--cell-rows 64] [--cell-deg 10] [--dup-share 0.6]
                    [--merge-share 2] [--min-int 20] [--min-gap 0.05] [--drop-dup-wraps 0] [--first K] [--last K] [--threads N]
                    [--tsv OUT] [--cells OUT] [--sfc 0.25]
   wrap-check on .sfc wraps (the strip's real input, e.g. wraps_snap) plus a repair. The wraps (ufsm wraps: rows = z, wrap k between
   k - 1 and k + 1 along every radial ray from the axis) are sampled on --nray rays per --row-step-th row: per ray the wraps' radii
   (angle-binned radius tables, as wrap-snap), and per pair of consecutive ids k, k + 1 present on the ray, an interval:
     - order: r(k + 1) - r(k) < max(1/2 voxel, --min-gap x the ray's median gap) = crossed (a wrap on or inside its inner neighbour);
     - CT papyrus peaks in the window [r(k), r(k + 1)) shifted inward by --shift x the gap (0.5: snapped wraps sit on their sheets,
       the window then holds the sheet of k in its middle; 0: wrap-check's window, for level-set wraps between the sheets), wrap-check's
       detector (a gaussian of --sigma CT voxels, prominence >= --prom x the local sd within +-24 voxels; CT 0 inside = not material:
       no interval): 0 = dup (two wraps on one sheet), 1 = right, 2+ = merged (a sheet skipped);
     - with --codes (winding_mod14 band codes, KEY = their um): the GT turns between the two crossings (labelled samples within 2).
   Per wrap the shares and wrap-check's verdict rule (dup: an empty share >= 0.5 on either side; merged: a 2+ share >= 0.5) -> --tsv
   (wrap-check's columns + crossed). Region evidence: cells of --cell-rows rows x --cell-deg degrees per pair (k, k + 1) with their
   counts -> --cells (k, row0, sector, n, empty, one, two_plus, gt_n, gt0, gt1, gt2, crossed, peak position f = (peak - r(k)) / gap).
   Repair (an output dir, not "-"): every wrap rewritten with
     - crossed vertices removed (on a ray where the wrap is within --min-gap of a neighbour's radius: the sheet is ambiguous there);
     - dup cells (n >= --min-int, empty share >= --dup-share): the outer wrap k + 1's vertices in the cell removed (one sheet, kept once);
       --drop-dup-wraps 1: a wrap whose verdict is dup removed whole (trim's rule, anywhere in the scroll);
     - merged cells (2+ share >= --merge-share; default 2 = never: on PHerc1667 a 2+ CT peak count is right 1.4 % of the time against
       the human labels): a fragment wrap at the skipped sheet (radius r(k) + f gap, the cell's median f) -> <out>/split/wKKK_m.sfc,
       not part of the numbered spiral.
   Deterministic for any thread count (per-ray results, sums in ray order). CPU only. */
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

#define WR_NB 2048   /* angle bins of the radius tables */

static const char *wr_opt(int argc, char **argv, const char *name, const char *def) { for (int i = 4; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1]; return def; }
static int wr_idx(int argc, char **argv, const char *name) { for (int i = 4; i < argc; i++) if (!strcmp(argv[i], name)) return i; return -1; }
static double wr_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

/* ---- a decoded-chunk table over one uint8 array, filled per band (wrap-snap's wsn_vol) ---- */
typedef struct { z3 *z; int64_t sh[3]; int cs[3]; int64_t g[3]; uint8_t **ch; _Atomic uint8_t *need; double f; uint8_t fill; } wr_vol;
static int wr_vol_open(wr_vol *v, const char *root, const char *key, double f, uint8_t fill) {
    memset(v, 0, sizeof *v); v->z = z3_open(store_open(root), key, nullptr); if (!v->z) return -1;
    const z3_meta *m = z3_meta_of(v->z);
    for (int d = 0; d < 3; d++) { v->sh[d] = m->shape[d]; v->cs[d] = m->chunk[d] > 0 ? m->chunk[d] : 128; v->g[d] = (m->shape[d] + v->cs[d] - 1) / v->cs[d]; }
    const size_t ng = (size_t)v->g[0] * v->g[1] * v->g[2]; v->ch = calloc(ng, sizeof *v->ch); v->need = calloc(ng, 1); v->f = f; v->fill = fill;
    return v->ch && v->need ? 0 : -1;
}
static void wr_vol_close(wr_vol *v) { if (!v->z) return; const size_t ng = (size_t)v->g[0] * v->g[1] * v->g[2]; for (size_t k = 0; k < ng; k++) free(v->ch[k]); free(v->ch); free((void *)v->need); z3_close(v->z); memset(v, 0, sizeof *v); }
static inline void wr_mark1(wr_vol *v, const double p[3]) {   /* the chunks of the 2x2x2 neighbourhood of wrap-voxel point p (z, y, x) */
    for (int c = 0; c < 8; c++) { int64_t a[3];
        for (int d = 0; d < 3; d++) { a[d] = (int64_t)floor((p[d] + 0.5) * v->f - 0.5) + (c >> (2 - d) & 1); if (a[d] < 0 || a[d] >= v->sh[d]) goto next; }
        atomic_store_explicit(&v->need[((size_t)(a[0] / v->cs[0]) * v->g[1] + (size_t)(a[1] / v->cs[1])) * v->g[2] + (size_t)(a[2] / v->cs[2])], 1, memory_order_relaxed);
        next:; }
}
typedef struct { wr_vol *v; int64_t *todo; size_t n; _Atomic size_t next; _Atomic int err; } wr_dec;
static void *wr_dec_run(void *arg) {
    wr_dec *J = arg; wr_vol *v = J->v;
    for (size_t k; (k = atomic_fetch_add(&J->next, 1)) < J->n;) {
        const int64_t id = J->todo[k], cz = id / (v->g[1] * v->g[2]), cy = id / v->g[2] % v->g[1], cx = id % v->g[2];
        int64_t o[3] = {cz * v->cs[0], cy * v->cs[1], cx * v->cs[2]}, n[3] = {v->cs[0], v->cs[1], v->cs[2]};
        uint8_t *b = malloc((size_t)n[0] * n[1] * n[2]);
        if (!b || z3_read(v->z, o, n, b, 1)) { free(b); atomic_store(&J->err, 1); continue; }
        v->ch[id] = b;
    }
    return nullptr;
}
static int wr_vol_fill(wr_vol *v, int nt) {
    if (!v->z) return 0;
    const size_t ng = (size_t)v->g[0] * v->g[1] * v->g[2]; int64_t *todo = nullptr; size_t n = 0, cap = 0;
    for (size_t k = 0; k < ng; k++) {
        if (!v->need[k]) { if (v->ch[k]) { free(v->ch[k]); v->ch[k] = nullptr; } continue; }
        v->need[k] = 0; if (v->ch[k]) continue;
        if (n == cap) { cap = cap ? 2 * cap : 1024; todo = realloc(todo, cap * sizeof *todo); }
        todo[n++] = (int64_t)k;
    }
    wr_dec J = {v, todo, n, 0, 0}; pthread_t th[64]; const int m = nt < 64 ? nt : 64;
    for (int q = 0; q < m; q++) pthread_create(&th[q], nullptr, wr_dec_run, &J);
    for (int q = 0; q < m; q++) pthread_join(th[q], nullptr);
    free(todo); return atomic_load(&J.err) ? -1 : 0;
}
static inline double wr_at(const wr_vol *v, int64_t z, int64_t y, int64_t x) {
    if (z < 0 || y < 0 || x < 0 || z >= v->sh[0] || y >= v->sh[1] || x >= v->sh[2]) return v->fill;
    const uint8_t *b = v->ch[((size_t)(z / v->cs[0]) * v->g[1] + (size_t)(y / v->cs[1])) * v->g[2] + (size_t)(x / v->cs[2])];
    return b ? b[((size_t)(z % v->cs[0]) * v->cs[1] + (size_t)(y % v->cs[1])) * v->cs[2] + (size_t)(x % v->cs[2])] : v->fill;
}
static double wr_tri(const wr_vol *v, const double p[3]) {
    double a[3]; for (int d = 0; d < 3; d++) a[d] = (p[d] + 0.5) * v->f - 0.5;
    const int64_t z0 = (int64_t)floor(a[0]), y0 = (int64_t)floor(a[1]), x0 = (int64_t)floor(a[2]); const double fz = a[0] - z0, fy = a[1] - y0, fx = a[2] - x0; double s = 0;
    for (int c = 0; c < 8; c++) s += (c >> 2 & 1 ? fz : 1 - fz) * (c >> 1 & 1 ? fy : 1 - fy) * (c & 1 ? fx : 1 - fx) * wr_at(v, z0 + (c >> 2 & 1), y0 + (c >> 1 & 1), x0 + (c & 1));
    return s;
}
static int wr_code(const wr_vol *v, const double p[3]) {   /* nearest code voxel (1..252, else 0) */
    const int c = (int)wr_at(v, (int64_t)floor((p[0] + 0.5) * v->f), (int64_t)floor((p[1] + 0.5) * v->f), (int64_t)floor((p[2] + 0.5) * v->f)); return c >= 1 && c <= 252 ? c : 0;
}

/* ---- the wraps ---- */
typedef struct { int id, w, h; double um, step; char path[1400]; sfc_reader *r; sfc_cache *ca; uint32_t ch; char *meta; float *row; uint8_t *rv; } wr_wrap;
static int wr_cmp_id(const void *a, const void *b) { const wr_wrap *x = a, *y = b; return (x->id > y->id) - (x->id < y->id); }
static int wr_bin(double th) { int b = (int)floor((th + M_PI) / (2 * M_PI) * WR_NB); return ((b % WR_NB) + WR_NB) % WR_NB; }
/* one row's radius table of a wrap (mean radius per angle bin, holes up to 8 bins filled linearly); returns the row's mean z or NAN */
static double wr_table(const float *xyz, const uint8_t *val, int w, const axis *ax, double s0, float *t, double *sum, int *cnt) {
    memset(sum, 0, WR_NB * sizeof(double)); memset(cnt, 0, WR_NB * sizeof(int)); double zs = 0; int nz = 0;
    for (int c = 0; c < w; c++) {
        if (!val[c] || !(xyz[3 * c + 2] >= 0)) continue;
        const float *p = &xyz[3 * c]; double cy, cx; axis_at(ax, p[2] * s0, &cy, &cx); cy /= s0; cx /= s0;
        const double dy = p[1] - cy, dx = p[0] - cx; const int b = wr_bin(atan2(dy, dx)); sum[b] += sqrt(dy * dy + dx * dx); cnt[b]++; zs += p[2]; nz++;
    }
    for (int b = 0; b < WR_NB; b++) t[b] = cnt[b] ? (float)(sum[b] / cnt[b]) : NAN;
    if (!nz) return NAN;
    for (int b = 0; b < WR_NB; b++) { if (cnt[b]) continue;
        int l = 1, r = 1; while (l <= 8 && !cnt[((b - l) % WR_NB + WR_NB) % WR_NB]) l++; while (r <= 8 && !cnt[(b + r) % WR_NB]) r++;
        if (l <= 8 && r <= 8) { const int bl = ((b - l) % WR_NB + WR_NB) % WR_NB, br = (b + r) % WR_NB; const double a = sum[bl] / cnt[bl], c = sum[br] / cnt[br]; t[b] = (float)(a + (c - a) * l / (l + r)); } }
    return zs / nz;
}

/* per-cell counts of a pair (k, k + 1) */
typedef struct { int n, e0, e1, e2, gn, g0, g1, g2, cr; float fsum; int fn; } wr_cell;
/* per-ray interval results (one sampled row): for each pair index pi (k = W[pi].id, k + 1 = W[pi + 1].id), per ray j */
typedef struct { int8_t npk, gt; uint8_t ok, cr; float f; } wr_iv;

typedef struct {
    wr_wrap *W; int nw; const axis *ax; double s0; wr_vol *ct, *cd; int nray, nr_s; const int *rows; const double *zrow; float **rt;   /* rt[k][s * WR_NB + b]: sampled row s */
    double sigma, prom, shift, mingap; int mark; wr_iv *iv; _Atomic int next;
} wr_job;
static void wr_ray(wr_job *J, int s, int j, double *prof, double *sm, float *pk, int cap) {
    const double th = -M_PI + 2 * M_PI * (j + 0.5) / J->nray, sn = sin(th), cs = cos(th); const int b = wr_bin(th), nw = J->nw;
    const double z = J->zrow[s]; double cy, cx; axis_at(J->ax, z * J->s0, &cy, &cx); cy /= J->s0; cx /= J->s0;
    double rmin = 1e30, rmax = -1e30; int npres = 0;
    for (int k = 0; k < nw; k++) { const float r = J->rt[k][(size_t)s * WR_NB + b]; if (r == r) { rmin = fmin(rmin, r); rmax = fmax(rmax, r); npres++; } }
    wr_iv *o = &J->iv[((size_t)s * J->nray + j) * nw]; for (int k = 0; k < nw; k++) o[k] = (wr_iv){0, -128, 0, 0, NAN};
    if (npres < 2) return;
    const double r0 = fmax(0, rmin - 40); int n = (int)ceil(rmax + 40 - r0) + 1; if (n > cap) n = cap;
    if (J->mark) {
        for (int i = 0; i < n; i += 2) { const double q[3] = {z, cy + (r0 + i) * sn, cx + (r0 + i) * cs}; wr_mark1(J->ct, q); }
        if (J->cd) for (int k = 0; k < nw; k++) { const float r = J->rt[k][(size_t)s * WR_NB + b]; if (!(r == r)) continue;
            for (int d = -3; d <= 3; d++) { const double q[3] = {z, cy + (r + d) * sn, cx + (r + d) * cs}; wr_mark1(J->cd, q); } }
        return;
    }
    for (int i = 0; i < n; i++) { const double q[3] = {z, cy + (r0 + i) * sn, cx + (r0 + i) * cs}; prof[i] = wr_tri(J->ct, q); }
    /* wrap-check's peak detector (step 1 CT voxel at F = 1; in wrap voxels here) */
    const int R = (int)ceil(3 * J->sigma); double gs = 0, gk[64]; for (int d = -R; d <= R && d + R < 64; d++) { gk[d + R] = exp(-0.5 * (d / J->sigma) * (d / J->sigma)); gs += gk[d + R]; }
    for (int i = 0; i < n; i++) { double a = 0, ws = 0; for (int d = -R; d <= R; d++) { const int k = i + d; if (k < 0 || k >= n) continue; a += gk[d + R] * prof[k]; ws += gk[d + R]; } sm[i] = a / ws; }
    int np = 0;
    for (int i = 1; i + 1 < n && np < cap; i++) {
        if (prof[i] <= 0 || !(sm[i] > sm[i - 1]) || sm[i] < sm[i + 1]) continue;
        const int a = i - 12 < 0 ? 0 : i - 12, bb = i + 13 > n ? n : i + 13; double mu = 0, s2 = 0; for (int k = a; k < bb; k++) { mu += sm[k]; s2 += sm[k] * sm[k]; }
        mu /= bb - a; const double sd = sqrt(fmax(0, s2 / (bb - a) - mu * mu)); if (!(sd > 0)) continue;
        double ml = sm[i], mr = sm[i];
        for (int k = i - 1; k >= 0 && k >= i - 24; k--) { if (sm[k] > sm[i]) break; if (sm[k] < ml) ml = sm[k]; }
        for (int k = i + 1; k < n && k <= i + 24; k++) { if (sm[k] >= sm[i]) break; if (sm[k] < mr) mr = sm[k]; }
        if (sm[i] - fmax(ml, mr) >= J->prom * sd) pk[np++] = (float)(r0 + i);
    }
    /* the pairs of consecutive ids present on the ray; the ray's median gap for the crossing threshold */
    double gaps[512]; int ng = 0;
    for (int k = 0; k + 1 < nw && ng < 512; k++) { if (J->W[k + 1].id != J->W[k].id + 1) continue; const float ra = J->rt[k][(size_t)s * WR_NB + b], rb = J->rt[k + 1][(size_t)s * WR_NB + b]; if (ra == ra && rb == rb && rb > ra) gaps[ng++] = rb - ra; }
    double gmed = 0; if (ng) { for (int a = 1; a < ng; a++) { const double v = gaps[a]; int c = a - 1; while (c >= 0 && gaps[c] > v) { gaps[c + 1] = gaps[c]; c--; } gaps[c + 1] = v; } gmed = gaps[ng / 2]; }
    for (int k = 0; k + 1 < nw; k++) {
        if (J->W[k + 1].id != J->W[k].id + 1) continue;
        const float ra = J->rt[k][(size_t)s * WR_NB + b], rb = J->rt[k + 1][(size_t)s * WR_NB + b]; if (!(ra == ra && rb == rb)) continue;
        wr_iv *v = &o[k];
        if (rb - ra < fmax(0.5, J->mingap * gmed)) { v->ok = 1; v->cr = 1; continue; }   /* (crossed or touching: rb <= ra + max(half a voxel, --min-gap x the median gap)) */
        const double g = rb - ra, wa = ra - J->shift * g, wb = rb - J->shift * g;
        int mat = 1; for (int i = (int)floor(wa - r0); i <= (int)ceil(wb - r0); i++) if (i < 0 || i >= n || prof[i] <= 0) { mat = 0; break; }
        if (!mat) continue;
        int c0 = 0; float best = NAN, bd = 1e30f;
        for (int p = 0; p < np; p++) if (pk[p] >= wa && pk[p] < wb) c0++;
        for (int p = 0; p < np; p++) if (pk[p] > ra + 0.2 * g && pk[p] < rb - 0.2 * g) { const float d = fabsf(pk[p] - (float)(0.5 * (ra + rb))); if (d < bd) { bd = d; best = (float)((pk[p] - ra) / g); } }
        v->ok = 1; v->npk = (int8_t)(c0 > 100 ? 100 : c0); v->f = best;
        if (J->cd) {
            int qa = 0, qb = 0;
            for (int d = 0; d <= 2 && !qa; d++) for (int sg = -1; sg <= 1 && !qa; sg += 2) { const double q[3] = {z, cy + (ra + sg * d) * sn, cx + (ra + sg * d) * cs}; qa = wr_code(J->cd, q); }
            for (int d = 0; d <= 2 && !qb; d++) for (int sg = -1; sg <= 1 && !qb; sg += 2) { const double q[3] = {z, cy + (rb + sg * d) * sn, cx + (rb + sg * d) * cs}; qb = wr_code(J->cd, q); }
            if (qa && qb) { double dq = fmod((qb - qa) / 18.0, 14.0); if (dq < 0) dq += 14; if (dq > 7) dq -= 14; v->gt = (int8_t)(dq < -0.25 ? -1 : dq < 0.5 ? 0 : dq < 1.5 ? 1 : 2); }
        }
    }
}
static void *wr_rays(void *arg) {
    wr_job *J = arg; const int cap = 1 << 15; double *prof = malloc(cap * sizeof(double)), *sm = malloc(cap * sizeof(double)); float *pk = malloc(cap * sizeof(float));
    for (int g; (g = atomic_fetch_add(&J->next, 1)) < J->nr_s * J->nray;) wr_ray(J, g / J->nray, g % J->nray, prof, sm, pk, cap);
    free(prof); free(sm); free(pk); return nullptr;
}

/* ---- the repair: one wrap rewritten (crossed / dup-cell vertices removed), the split fragment of its pair (k, k + 1) ---- */
typedef struct {
    wr_wrap *W; int nw; const axis *ax; double s0, mingap; const char *out; int cr, nsec, ncr; const wr_cell *cells; const uint8_t *dupc, *mrgc, *dropw;
    _Atomic int next, fail; _Atomic long long nval, ncross, ndup, nfrag, nkept;
} wr_rep;
static int wr_load(wr_wrap *w, float **X, uint8_t **V) {
    const size_t nv = (size_t)w->w * w->h; *X = malloc(nv * 12); *V = malloc(nv);
    if (!*X || !*V || sfc_cache_read_xyz_region(w->ca, w->ch, 0, 0, (uint32_t)w->w, (uint32_t)w->h, *X, 12, (size_t)w->w * 12, *V)) { free(*X); free(*V); *X = nullptr; *V = nullptr; return -1; }
    for (size_t i = 0; i < nv; i++) if ((*V)[i] && !((*X)[3 * i + 2] >= 0)) (*V)[i] = 0;
    return 0;
}
static float *wr_full_table(wr_wrap *w, const axis *ax, double s0) {   /* rows x WR_NB radius table of a whole wrap */
    float *X; uint8_t *V; if (wr_load(w, &X, &V)) return nullptr;
    float *t = malloc((size_t)w->h * WR_NB * sizeof(float)); double *sum = malloc(WR_NB * sizeof(double)); int *cnt = malloc(WR_NB * sizeof(int));
    for (int r = 0; r < w->h; r++) wr_table(&X[(size_t)r * w->w * 3], &V[(size_t)r * w->w], w->w, ax, s0, &t[(size_t)r * WR_NB], sum, cnt);
    free(sum); free(cnt); free(X); free(V); return t;
}
static void wr_rep_one(wr_rep *S, int k) {
    wr_wrap *w = &S->W[k]; float *X; uint8_t *V; if (wr_load(w, &X, &V)) { fprintf(stderr, "wrap-repair: cannot read %s\n", w->path); atomic_store(&S->fail, 1); return; }
    const int W = w->w, H = w->h; const size_t nv = (size_t)W * H;
    const int hasA = k > 0 && S->W[k - 1].id == w->id - 1, hasB = k + 1 < S->nw && S->W[k + 1].id == w->id + 1;
    float *ta = hasA ? wr_full_table(&S->W[k - 1], S->ax, S->s0) : nullptr, *tb = hasB ? wr_full_table(&S->W[k + 1], S->ax, S->s0) : nullptr;
    float *x = malloc(nv * 4), *y = malloc(nv * 4), *z = malloc(nv * 4), *fx = nullptr, *fy = nullptr, *fz = nullptr; long long val = 0, ncr = 0, ndp = 0, nfr = 0, kept = 0;
    const int frag = hasB && S->mrgc; if (frag) { fx = malloc(nv * 4); fy = malloc(nv * 4); fz = malloc(nv * 4); }
    const int drop = S->dropw && S->dropw[k];
    for (size_t i = 0; i < nv; i++) {
        x[i] = y[i] = z[i] = -1.f; if (frag) fx[i] = fy[i] = fz[i] = -1.f;
        if (!V[i]) continue;
        val++;
        const float *p = &X[3 * i]; const int r = (int)(i / W); double cy, cx; axis_at(S->ax, p[2] * S->s0, &cy, &cx); cy /= S->s0; cx /= S->s0;
        const double dy = p[1] - cy, dx = p[0] - cx, rad = sqrt(dy * dy + dx * dx), th = atan2(dy, dx); const int b = wr_bin(th);
        const double ra = ta ? ta[(size_t)r * WR_NB + b] : NAN, rb = tb ? tb[(size_t)r * WR_NB + b] : NAN;
        const int cell = (r / S->cr) * S->nsec + (int)((th + M_PI) / (2 * M_PI) * S->nsec) % S->nsec;
        if (drop) { ndp++; continue; }
        /* crossed: within --min-gap x the local gap (the other side's gap where one neighbour is missing) of a neighbour */
        const double gi = ra == ra ? rad - ra : NAN, go = rb == rb ? rb - rad : NAN, gref = gi == gi && go == go ? 0.5 * (fabs(gi) + fabs(go)) : gi == gi ? fabs(gi) : fabs(go);
        if ((gi == gi && gi < fmax(0.5, S->mingap * gref)) || (go == go && go < fmax(0.5, S->mingap * gref))) { ncr++; continue; }
        if (hasA && S->dupc && S->dupc[(size_t)(k - 1) * S->ncr * S->nsec + cell]) { ndp++; continue; }   /* pair (k - 1, k) one sheet: this outer wrap goes there */
        x[i] = p[0]; y[i] = p[1]; z[i] = p[2]; kept++;
        if (frag && S->mrgc[(size_t)k * S->ncr * S->nsec + cell] && rb == rb && rb > rad) {
            const wr_cell *c = &S->cells[(size_t)k * S->ncr * S->nsec + cell]; const double f = c->fn ? c->fsum / c->fn : 0.5, nr = rad + f * (rb - rad);
            fx[i] = (float)(cx + nr * dx / rad); fy[i] = (float)(cy + nr * dy / rad); fz[i] = p[2]; nfr++;
        }
    }
    atomic_fetch_add(&S->nval, val); atomic_fetch_add(&S->ncross, ncr); atomic_fetch_add(&S->ndup, ndp); atomic_fetch_add(&S->nfrag, nfr); atomic_fetch_add(&S->nkept, kept);
    char p[1600]; int winding = w->id; const char *q = w->meta ? strstr(w->meta, "\"winding\":") : nullptr; if (q) winding = atoi(q + 10);
    if (kept) { snprintf(p, sizeof p, "%s/w%03d", S->out, w->id);
        if (tifxyz_write(p, W, H, x, y, z, w->step, w->um, winding, "ufsm wrap-repair (crossed and duplicate-sheet vertices removed)")) { fprintf(stderr, "wrap-repair: cannot write %s\n", p); atomic_store(&S->fail, 1); } }
    if (nfr) { snprintf(p, sizeof p, "%s/split/w%03d_m", S->out, w->id);
        if (tifxyz_write(p, W, H, fx, fy, fz, w->step, w->um, winding, "ufsm wrap-repair (split fragment: the sheet between this wrap and the next)")) { fprintf(stderr, "wrap-repair: cannot write %s\n", p); atomic_store(&S->fail, 1); } }
    free(X); free(V); free(x); free(y); free(z); free(fx); free(fy); free(fz); free(ta); free(tb);
}
static void *wr_rep_run(void *arg) { wr_rep *S = arg; for (int k; (k = atomic_fetch_add(&S->next, 1)) < S->nw;) if (S->W[k].r) wr_rep_one(S, k); return nullptr; }

int cmd_wrap_repair(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm wrap-repair <wraps-in> <wraps-out | -> --axis FILE --axis-um U0 --ct ROOT KEY --ct-um U [--codes ROOT KEY] [--nray 720] [--row-step 8]\n"
                                    "       [--sigma 1.2] [--prom 1] [--shift 0.5] [--cell-rows 64] [--cell-deg 10] [--dup-share 0.6] [--merge-share 2] [--min-int 20] [--min-gap 0.05]\n"
                                    "       [--drop-dup-wraps 0] [--first K] [--last K] [--threads N] [--tsv OUT] [--cells OUT] [--sfc 0.25]\n"
                                    "  wrap-check of .sfc wraps along their rays (CT peaks per interval, GT turns with --codes, crossings) and the repair into wraps-out\n"); return 2; }
    const char *in = argv[2], *out = strcmp(argv[3], "-") ? argv[3] : nullptr, *axp = wr_opt(argc, argv, "--axis", nullptr);
    const double axum = atof(wr_opt(argc, argv, "--axis-um", "0")), ctum = atof(wr_opt(argc, argv, "--ct-um", "0"));
    const int ci = wr_idx(argc, argv, "--ct"), qi = wr_idx(argc, argv, "--codes"), nt0 = ufsm_threads(atoi(wr_opt(argc, argv, "--threads", "0")), 12), nt = nt0 > 64 ? 64 : nt0;
    const int nray = atoi(wr_opt(argc, argv, "--nray", "720")), rstep = atoi(wr_opt(argc, argv, "--row-step", "8")), crow = atoi(wr_opt(argc, argv, "--cell-rows", "64")), mint = atoi(wr_opt(argc, argv, "--min-int", "20"));
    const double cdeg = atof(wr_opt(argc, argv, "--cell-deg", "10")), dupsh = atof(wr_opt(argc, argv, "--dup-share", "0.6")), mrgsh = atof(wr_opt(argc, argv, "--merge-share", "2"));
    const int k0 = atoi(wr_opt(argc, argv, "--first", "-1000000")), k1 = atoi(wr_opt(argc, argv, "--last", "1000000")), dropdw = atoi(wr_opt(argc, argv, "--drop-dup-wraps", "0"));
    if (!axp || !(axum > 0) || ci < 0 || ci + 2 >= argc || !(ctum > 0) || nray < 8 || rstep < 1 || crow < 1 || !(cdeg > 0)) { fprintf(stderr, "wrap-repair: --axis FILE --axis-um U0 --ct ROOT KEY --ct-um U; --nray >= 8, --row-step / --cell-rows >= 1\n"); return 2; }
    axis ax = {0}; if (axis_load(&ax, axp)) { fprintf(stderr, "wrap-repair: cannot load the axis %s\n", axp); return 1; }
    DIR *dd = opendir(in); if (!dd) { fprintf(stderr, "wrap-repair: cannot open %s\n", in); return 1; }
    int nw = 0, cap = 64; wr_wrap *W = calloc(cap, sizeof *W); struct dirent *de;
    while ((de = readdir(dd))) { int id; char tail[8] = ""; if (sscanf(de->d_name, "w%d.%7s", &id, tail) != 2 || strcmp(tail, "sfc") || id < k0 || id > k1) continue;
        if (nw == cap) { cap *= 2; W = realloc(W, cap * sizeof *W); } memset(&W[nw], 0, sizeof W[nw]); W[nw].id = id; snprintf(W[nw].path, sizeof W[nw].path, "%s/%s", in, de->d_name); nw++; }
    closedir(dd);
    if (nw < 2) { fprintf(stderr, "wrap-repair: fewer than 2 wNNN.sfc wraps in %s\n", in); return 1; }
    qsort(W, nw, sizeof *W, wr_cmp_id);
    double wum = 0; int H = -1;
    for (int k = 0; k < nw; k++) {
        wr_wrap *w = &W[k];
        if (sfc_open_file(w->path, &w->r)) { fprintf(stderr, "wrap-repair: cannot open %s\n", w->path); return 1; }
        int c = sfc_find_channel(w->r, "x"); if (c < 0) c = 0; w->ch = (uint32_t)c;
        const sfc_channel *chi = sfc_channel_info(w->r, w->ch); w->w = (int)chi->width; w->h = (int)chi->height;
        const uint64_t ml = sfc_metadata_size(w->r); w->meta = calloc(ml + 1, 1); if (ml && sfc_read_metadata(w->r, w->meta, ml)) { fprintf(stderr, "wrap-repair: no metadata in %s\n", w->path); return 1; }
        const char *q = strstr(w->meta, "\"voxel_size_um\":"); w->um = q ? atof(q + 16) : 0;
        q = strstr(w->meta, "\"scale\":"); if (q) { q = strchr(q, '['); if (q && atof(q + 1) > 0) w->step = 1 / atof(q + 1); }
        if (!(w->um > 0) || !(w->step > 0)) { fprintf(stderr, "wrap-repair: %s: no voxel_size_um / scale\n", w->path); return 1; }
        if ((wum > 0 && fabs(w->um - wum) > 1e-9) || (H >= 0 && w->h != H)) { fprintf(stderr, "wrap-repair: %s: wraps of one ufsm wraps run only (same um and rows)\n", w->path); return 1; }
        wum = w->um; H = w->h;
        if (sfc_cache_create(w->r, (size_t)((w->w + 63) / 64 + 1) * 3 * 64 * 64 * 12 * 2, &w->ca)) { fprintf(stderr, "wrap-repair: no memory\n"); return 1; }
        w->row = malloc((size_t)w->w * 12); w->rv = malloc(w->w);
    }
    const double s0 = wum / axum;
    wr_vol vct, vcd; memset(&vcd, 0, sizeof vcd);
    if (wr_vol_open(&vct, argv[ci + 1], argv[ci + 2], wum / ctum, 0)) { fprintf(stderr, "wrap-repair: CT: %s\n", z3_error()); return 1; }
    if (qi > 0) { if (qi + 2 >= argc || !(atof(argv[qi + 2]) > 0)) { fprintf(stderr, "wrap-repair: --codes ROOT KEY (KEY = the codes' um)\n"); return 2; }
                  if (wr_vol_open(&vcd, argv[qi + 1], argv[qi + 2], wum / atof(argv[qi + 2]), 0)) { fprintf(stderr, "wrap-repair: codes: %s\n", z3_error()); return 1; } }
    /* the sampled rows, in bands of one CT chunk layer (codes: one codes chunk layer) */
    const int nsec = (int)lround(360 / cdeg), ncr = (H + crow - 1) / crow; const size_t ncell = (size_t)nw * ncr * nsec;
    wr_cell *cells = calloc(ncell, sizeof *cells);
    typedef struct { long long n, e0, e1, e2, gn, g0, g1, g2, cr; } wtot; wtot *T = calloc(nw, sizeof *T);
    long long conf[3][4] = {{0}};
    const double bandz = fmin(vct.cs[0] / vct.f, vcd.z ? vcd.cs[0] / vcd.f : 1e30);
    float **rt = malloc(nw * sizeof *rt); double *sum = malloc(WR_NB * sizeof(double)); int *cnt = malloc(WR_NB * sizeof(int));
    const int maxs = H / rstep + 2; int *rows = malloc(maxs * sizeof(int)); double *zrow = malloc(maxs * sizeof(double));
    const double t0 = wr_now(); long long nint = 0;
    fprintf(stderr, "wrap-repair: %d wraps (w%03d..w%03d), %d rows, %.4g um, %d rays every %d rows, cells %d rows x %.0f deg%s\n", nw, W[0].id, W[nw - 1].id, H, wum, nray, rstep, crow, cdeg, vcd.z ? ", GT codes" : "");
    int r = 0;
    while (r < H) {
        /* the band: sampled rows from r while their z stays in one chunk layer of the first row's */
        int ns = 0; double zb = NAN;
        for (int k = 0; k < nw; k++) rt[k] = nullptr;
        float *rtall = malloc((size_t)nw * 64 * WR_NB * sizeof(float));   /* (at most 64 sampled rows per band) */
        for (; r < H && ns < 64; r += rstep) {
            double zr = NAN; float *tb = rtall + (size_t)ns * WR_NB;
            for (int k = 0; k < nw; k++) {
                wr_wrap *w = &W[k]; float *t = rtall + ((size_t)k * 64 + ns) * WR_NB;
                if (sfc_cache_read_xyz_region(w->ca, w->ch, 0, (uint64_t)r, (uint32_t)w->w, 1, w->row, 12, (size_t)w->w * 12, w->rv)) { for (int b = 0; b < WR_NB; b++) t[b] = NAN; continue; }
                const double zz = wr_table(w->row, w->rv, w->w, &ax, s0, t, sum, cnt); if (zz == zz && !(zr == zr)) zr = zz;
            }
            (void)tb;
            if (!(zr == zr)) continue;
            if (zb == zb && floor((zr + 0.5) / bandz) != floor((zb + 0.5) / bandz)) break;   /* (this row starts the next band: re-read there) */
            if (!(zb == zb)) zb = zr;
            rows[ns] = r; zrow[ns] = zr; ns++;
        }
        if (!ns) { free(rtall); continue; }
        /* rt[k] as [s * WR_NB + b]: repack from the per-wrap blocks of 64 rows */
        float *rtp = malloc((size_t)nw * ns * WR_NB * sizeof(float));
        for (int k = 0; k < nw; k++) { rt[k] = rtp + (size_t)k * ns * WR_NB; memcpy(rt[k], rtall + (size_t)k * 64 * WR_NB, (size_t)ns * WR_NB * sizeof(float)); }
        free(rtall);
        wr_iv *iv = malloc((size_t)ns * nray * nw * sizeof *iv);
        wr_job J = {W, nw, &ax, s0, &vct, vcd.z ? &vcd : nullptr, nray, ns, rows, zrow, rt, atof(wr_opt(argc, argv, "--sigma", "1.2")), atof(wr_opt(argc, argv, "--prom", "1")),
                    atof(wr_opt(argc, argv, "--shift", "0.5")), atof(wr_opt(argc, argv, "--min-gap", "0.05")), 1, iv, 0};
        pthread_t th[64];
        for (int q = 0; q < nt; q++) pthread_create(&th[q], nullptr, wr_rays, &J);
        for (int q = 0; q < nt; q++) pthread_join(th[q], nullptr);
        if (wr_vol_fill(&vct, nt) || wr_vol_fill(&vcd, nt)) { fprintf(stderr, "wrap-repair: volume read: %s\n", z3_error()); return 1; }
        J.mark = 0; atomic_store(&J.next, 0);
        for (int q = 0; q < nt; q++) pthread_create(&th[q], nullptr, wr_rays, &J);
        for (int q = 0; q < nt; q++) pthread_join(th[q], nullptr);
        for (int s = 0; s < ns; s++) for (int j = 0; j < nray; j++) for (int k = 0; k + 1 < nw; k++) {
            const wr_iv *v = &iv[((size_t)s * nray + j) * nw + k]; if (!v->ok) continue;
            const double th2 = -M_PI + 2 * M_PI * (j + 0.5) / nray; wr_cell *c = &cells[((size_t)k * ncr + rows[s] / crow) * nsec + (int)((th2 + M_PI) / (2 * M_PI) * nsec) % nsec];
            wtot *t = &T[k]; nint++;
            if (v->cr) { c->cr++; t->cr++; continue; }
            c->n++; t->n++; c->e0 += v->npk == 0; c->e1 += v->npk == 1; c->e2 += v->npk >= 2; t->e0 += v->npk == 0; t->e1 += v->npk == 1; t->e2 += v->npk >= 2;
            if (v->f == v->f) { c->fsum += v->f; c->fn++; }
            if (v->gt != -128) { c->gn++; t->gn++; c->g0 += v->gt == 0; c->g1 += v->gt == 1; c->g2 += v->gt >= 2; t->g0 += v->gt == 0; t->g1 += v->gt == 1; t->g2 += v->gt >= 2;
                                 conf[v->npk > 2 ? 2 : v->npk][v->gt == 0 ? 0 : v->gt == 1 ? 1 : v->gt >= 2 ? 2 : 3]++; }
        }
        free(iv); free(rtp);
        fprintf(stderr, "\rwrap-repair: row %d / %d (%lld intervals, %.0f s)", r, H, nint, wr_now() - t0);
    }
    fprintf(stderr, "\n"); wr_vol_close(&vct); const int hasgt = vcd.z != nullptr; wr_vol_close(&vcd);
    /* totals, the per-wrap verdicts (wrap-check's rule: wrap k from the pairs (k - 1, k) and (k, k + 1)) */
    long long tn = 0, te0 = 0, te2 = 0, tcr = 0, tg = 0, tg0 = 0, tg1 = 0, tg2 = 0;
    for (int k = 0; k < nw; k++) { tn += T[k].n; te0 += T[k].e0; te2 += T[k].e2; tcr += T[k].cr; tg += T[k].gn; tg0 += T[k].g0; tg1 += T[k].g1; tg2 += T[k].g2; }
    printf("wrap-repair: %s: %d wraps, %lld intervals (%.0f s): empty %.4f, 2+ %.4f, crossed %lld (%.5f)%s", in, nw, tn, wr_now() - t0, tn ? (double)te0 / tn : 0, tn ? (double)te2 / tn : 0, tcr, tn + tcr ? (double)tcr / (tn + tcr) : 0, hasgt ? "" : "\n");
    if (hasgt) printf("; GT turns on %lld intervals: 0 %.4f, 1 %.4f, 2+ %.4f\n", tg, tg ? (double)tg0 / tg : 0, tg ? (double)tg1 / tg : 0, tg ? (double)tg2 / tg : 0);
    if (hasgt) { printf("  validation: CT peaks (rows 0 / 1 / 2+) x GT turns (0 / 1 / 2+ / <0)\n"); for (int a = 0; a < 3; a++) printf("    npk %s %9lld %9lld %9lld %9lld\n", a == 0 ? "0 " : a == 1 ? "1 " : "2+", conf[a][0], conf[a][1], conf[a][2], conf[a][3]); }
    uint8_t *dropw = calloc(nw, 1); int ndup = 0, nmrg = 0, nreal = 0; long long minn = 0; for (int k = 0; k < nw; k++) if (T[k].n > minn) minn = T[k].n; minn = minn / 20 > 10 ? minn / 20 : 10;
    FILE *tf = nullptr; const char *tsv = wr_opt(argc, argv, "--tsv", nullptr); if (tsv && !(tf = fopen(tsv, "w"))) { fprintf(stderr, "wrap-repair: cannot write %s\n", tsv); return 1; }
    if (tf) fputs("wrap\tintervals\tempty\tone\ttwo_plus\tcrossed\tgt_n\tgt0\tgt1\tgt2\tverdict\n", tf);
    for (int k = 0; k + 1 < nw; k++) {
        const wtot *s = &T[k], *p = k && W[k - 1].id == W[k].id - 1 ? &T[k - 1] : nullptr; if (s->n < minn) continue;
        const double eo = (double)s->e0 / s->n, ei = p && p->n >= minn ? (double)p->e0 / p->n : 0, mo = (double)s->e2 / s->n, mi = p && p->n >= minn ? (double)p->e2 / p->n : 0;
        const char *vd = fmax(eo, ei) >= 0.5 ? "dup" : fmax(mo, mi) >= 0.5 ? "merged" : "real";
        ndup += vd[0] == 'd'; nmrg += vd[0] == 'm'; nreal += vd[0] == 'r'; if (vd[0] == 'd' && dropdw) dropw[k] = 1;
        if (tf) fprintf(tf, "w%03d\t%lld\t%.4f\t%.4f\t%.4f\t%lld\t%lld\t%.4f\t%.4f\t%.4f\t%s\n", W[k].id, s->n, eo, (double)s->e1 / s->n, mo, s->cr, s->gn, s->gn ? (double)s->g0 / s->gn : 0, s->gn ? (double)s->g1 / s->gn : 0, s->gn ? (double)s->g2 / s->gn : 0, vd);
    }
    if (tf) fclose(tf);
    printf("wrap-repair: verdicts: %d real, %d dup, %d merged (wrap-check's rule, wraps with >= %lld intervals)\n", nreal, ndup, nmrg, minn);
    /* the cells: dup / merged decisions; --cells dump */
    uint8_t *dupc = calloc(ncell, 1), *mrgc = calloc(ncell, 1); long long cd = 0, cm = 0, cdg[3] = {0}, cmg[3] = {0};
    FILE *cf = nullptr; const char *cfp = wr_opt(argc, argv, "--cells", nullptr); if (cfp && !(cf = fopen(cfp, "w"))) { fprintf(stderr, "wrap-repair: cannot write %s\n", cfp); return 1; }
    if (cf) fputs("k\trow0\tsector\tn\tempty\tone\ttwo_plus\tgt_n\tgt0\tgt1\tgt2\tcrossed\tf\n", cf);
    for (int k = 0; k + 1 < nw; k++) for (int a = 0; a < ncr; a++) for (int b = 0; b < nsec; b++) {
        const size_t i = ((size_t)k * ncr + a) * nsec + b; const wr_cell *c = &cells[i]; if (!c->n && !c->cr) continue;
        if (cf) fprintf(cf, "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.3f\n", W[k].id, a * crow, b, c->n, c->e0, c->e1, c->e2, c->gn, c->g0, c->g1, c->g2, c->cr, c->fn ? c->fsum / c->fn : NAN);
        if (c->n < mint || W[k + 1].id != W[k].id + 1) continue;
        if ((double)c->e0 / c->n >= dupsh) { dupc[i] = 1; cd++; cdg[0] += c->g0; cdg[1] += c->g1; cdg[2] += c->g2; }
        if ((double)c->e2 / c->n >= mrgsh) { mrgc[i] = 1; cm++; cmg[0] += c->g0; cmg[1] += c->g1; cmg[2] += c->g2; }
    }
    if (cf) fclose(cf);
    printf("wrap-repair: cells %d rows x %d sectors: dup cells %lld (empty >= %.2f, n >= %d)%s", crow, nsec, cd, dupsh, mint, hasgt ? "" : "");
    if (hasgt) printf(" GT 0 / 1 / 2+ there %lld / %lld / %lld (precision %.3f)", cdg[0], cdg[1], cdg[2], cdg[0] + cdg[1] + cdg[2] ? (double)cdg[0] / (cdg[0] + cdg[1] + cdg[2]) : 0);
    printf("; merged cells %lld (2+ >= %.2f)", cm, mrgsh); if (hasgt) printf(" GT 0 / 1 / 2+ there %lld / %lld / %lld", cmg[0], cmg[1], cmg[2]); printf("\n");
    if (out) {
        char cmd[1700]; snprintf(cmd, sizeof cmd, "mkdir -p '%s/split'", out); if (system(cmd)) return 1;
        if (atof(wr_opt(argc, argv, "--sfc", "0.25")) > 0) tifxyz_set_sfc(atof(wr_opt(argc, argv, "--sfc", "0.25")));
        tfx_set_threads(1);
        wr_rep S = {.W = W, .nw = nw, .ax = &ax, .s0 = s0, .mingap = atof(wr_opt(argc, argv, "--min-gap", "0.05")), .out = out, .cr = crow, .nsec = nsec, .ncr = ncr, .cells = cells,
                    .dupc = cd ? dupc : nullptr, .mrgc = cm ? mrgc : nullptr, .dropw = dropdw ? dropw : nullptr};
        pthread_t th[64]; const double t1 = wr_now();
        for (int q = 0; q < nt; q++) pthread_create(&th[q], nullptr, wr_rep_run, &S);
        for (int q = 0; q < nt; q++) pthread_join(th[q], nullptr);
        const long long nv = S.nval;
        printf("wrap-repair: repair %.0f s: %lld vertices: kept %.5f, crossed removed %lld (%.5f), dup removed %lld (%.5f), split fragment vertices %lld -> %s\n", wr_now() - t1, nv, nv ? (double)S.nkept / nv : 0,
               (long long)S.ncross, nv ? (double)S.ncross / nv : 0, (long long)S.ndup, nv ? (double)S.ndup / nv : 0, (long long)S.nfrag, out);
        if (atomic_load(&S.fail)) return 1;
    }
    for (int k = 0; k < nw; k++) { free(W[k].row); free(W[k].rv); free(W[k].meta); if (W[k].ca) sfc_cache_destroy(W[k].ca); if (W[k].r) sfc_close(W[k].r); }
    free(W); free(cells); free(T); free(rt); free(sum); free(cnt); free(rows); free(zrow); free(dropw); free(dupc); free(mrgc);
    return 0;
}
