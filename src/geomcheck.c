/* ufsm geom-check <wraps-dir> --axis FILE --axis-um U [--out DIR] [--first K] [--last K] [--every 2] [--nb 2048] [--zbins 32]
                  [--threads N] [--max-steep F] [--max-jump F] [--max-merge F] [--max-skip F] [--min-valid V]
   Geometry QA of a ufsm wraps dir (wNNN.sfc: rows = z, columns = rays around the axis; wrap k = the level set W = k). Label-free,
   CT-free: only the surfaces and the axis. Per vertex (wrap voxels; step = the wraps' grid step):
   - column spacing |P(c+1) - P(c)| / step (equidistant sampling = 1 everywhere; ufsm wraps samples uniform theta at the arc step of
     one radius per wrap, so this is r / r_col) and its per-row mean (the circumference per z: rows of one wrap that differ are
     stretched against each other when the strip shares columns across rows);
   - row spacing |P(r+1) - P(r)| / step;
   - steep: the surface normal within 20 degrees of tangential to the cylinder (|n . radial| < 0.34): a wall where the root along
     the rays jumps between sheets (renders: smeared streaks / tents);
   - jump: the radius changes by more than --jump (0.5) local sheet pitches between neighbouring vertices (a tear, sheet switch);
   - neighbour gaps on the same ray (row, angle bin; radius tables of every --every-th row x --nb angle bins): the gap to wrap k +- 1
     over the local pitch (median gap of the pairs k-2 .. k+2 there): merge < 0.3 (two wraps on one sheet / crossing; over-count),
     skip > 1.8 (a sheet left out; under-count);
   - holes: rows of a wrap whose valid share falls below half of the wrap's median row share inside its own z extent.
   Outputs (--out): wraps.tsv (per wrap), zbins.tsv (per wrap x z bin: valid share of the row, mean radius: the inner-end / core
   analysis), cells.tsv (64 rows x 1/64 turn cells with defect share >= 0.25: example locations, centre xyz), summary.txt.
   Guard: --max-steep / --max-jump / --max-merge / --max-skip F (shares of the valid vertices of a wrap): wraps over any are listed
   as "bad" in wraps.tsv and the exit code is 3 when the whole set's shares exceed them (a pipeline stage stops there). */
#include "sources.h"
#include "store.h"
#include "zarr3.h"
#include "surfcomp.h"
#include "sysinfo.h"
#include <dirent.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char *gc_opt(int argc, char **argv, const char *name, const char *def) { for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1]; return def; }
static double gc_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

#define GC_NH 64          /* log2 histogram bins of spacing ratios: 1/16 .. 16 */
#define GC_CR 64          /* cell rows */
#define GC_CT 64          /* cells per turn */
static int gc_hbin(double v) { if (!(v > 0)) return 0; int b = (int)floor((log2(v) + 4) * GC_NH / 8); return b < 0 ? 0 : b >= GC_NH ? GC_NH - 1 : b; }
static double gc_hq(const long long *h, double q) {   /* quantile of a log2 histogram (bin centre) */
    long long n = 0; for (int b = 0; b < GC_NH; b++) n += h[b]; if (!n) return NAN;
    long long c = 0; for (int b = 0; b < GC_NH; b++) { c += h[b]; if (c >= q * n) return exp2((b + 0.5) * 8.0 / GC_NH - 4); }
    return 16;
}

typedef struct {
    int id, w, h; double um, step; char path[1400];
    uint16_t *rt;                 /* radius x 4 per (table row, angle bin), 0 = none */
    /* results */
    long long nv, steep, jump, merge, skip, nboth, holes, rows_in;
    long long hc[GC_NH], hr[GC_NH], hrm[GC_NH];   /* column, row spacing; per-row mean column spacing */
    int row0, row1; double rmed, pitch;
    double *zb_valid, *zb_r, *zb_dx, *zb_dy; long long *zb_n; uint64_t *zb_cov;   /* per z bin: valid share, mean radius, mean offset from the axis (x, y), 1/64-turn coverage */
    int bad; float *rz;   /* rz: per row the z of its first valid vertex (NAN: none) */
} gc_wrap;
typedef struct { int wrap, cr, ct; int n, steep, jump, merge, skip; double cx, cy, cz; } gc_cell;
typedef struct {
    gc_wrap *W; int nw; const axis *ax; double s0; int every, nb, zbins, tr; double jump_t;
    _Atomic int next; _Atomic int fail; int pass; double gpitch; int sgn;
    pthread_mutex_t mu; gc_cell *cells; size_t ncell, ccap;
} gc_job;

static int gc_read(const gc_wrap *w, float **X, uint8_t **V) {
    sfc_reader *r; if (sfc_open_file(w->path, &r)) return -1;
    int c = sfc_find_channel(r, "x"); if (c < 0) c = 0;
    sfc_cache *ca; const size_t budget = (size_t)((w->w + 63) / 64 + 1) * 3 * 64 * 64 * 12 * 2;
    if (sfc_cache_create(r, budget, &ca)) { sfc_close(r); return -1; }
    const size_t nv = (size_t)w->w * w->h; *X = malloc(nv * 12); *V = malloc(nv);
    int rc = !*X || !*V || sfc_cache_read_xyz_region(ca, (uint32_t)c, 0, 0, (uint32_t)w->w, (uint32_t)w->h, *X, 12, (size_t)w->w * 12, *V);
    sfc_cache_destroy(ca); sfc_close(r);
    if (rc) { free(*X); free(*V); *X = nullptr; *V = nullptr; return -1; }
    for (size_t i = 0; i < nv; i++) if ((*V)[i] && !((*X)[3 * i + 2] >= 0 && isfinite((*X)[3 * i]) && isfinite((*X)[3 * i + 1]))) (*V)[i] = 0;
    return 0;
}
static inline void gc_axis(const gc_job *J, double z, double *ay, double *ax) { axis_at(J->ax, z * J->s0, ay, ax); *ay /= J->s0; *ax /= J->s0; }
static inline int gc_bin(const gc_job *J, double th) { int b = (int)floor((th + M_PI) / (2 * M_PI) * J->nb); return b < 0 ? 0 : b >= J->nb ? J->nb - 1 : b; }
static inline double gc_rt(const gc_wrap *w, const gc_job *J, int trow, int b) { if (!w || !w->rt) return NAN; const uint16_t v = w->rt[(size_t)trow * J->nb + b]; return v ? v * 0.25 : NAN; }
static int gc_cmpd(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

static void gc_pass1(gc_job *J, gc_wrap *w) {   /* radius tables, z extent, per z bin valid share and radius */
    float *X; uint8_t *V; if (gc_read(w, &X, &V)) { fprintf(stderr, "geom-check: cannot read %s\n", w->path); atomic_store(&J->fail, 1); return; }
    w->rt = calloc((size_t)J->tr * J->nb, 2);
    w->zb_valid = calloc(J->zbins, 8); w->zb_r = calloc(J->zbins, 8); w->zb_n = calloc(J->zbins, 8); w->zb_dx = calloc(J->zbins, 8); w->zb_dy = calloc(J->zbins, 8); w->zb_cov = calloc(J->zbins, 8);
    w->row0 = -1; w->row1 = -1; w->rz = malloc((size_t)w->h * 4); for (int r = 0; r < w->h; r++) w->rz[r] = NAN;
    double *rs = malloc(sizeof(double) * 4096); int nrs = 0; unsigned long long lcg = 12345;
    for (int r = 0; r < w->h; r++) {
        long long nvr = 0; const int zb = (int)((long long)r * J->zbins / w->h);
        for (int c = 0; c < w->w; c++) {
            const size_t i = (size_t)r * w->w + c; if (!V[i]) continue;
            const double x = X[3 * i], y = X[3 * i + 1], z = X[3 * i + 2]; double ay, ax; gc_axis(J, z, &ay, &ax);
            const double rr = hypot(y - ay, x - ax);
            if (!nvr) w->rz[r] = (float)z;
            nvr++; w->zb_r[zb] += rr; w->zb_dx[zb] += x - ax; w->zb_dy[zb] += y - ay; w->zb_cov[zb] |= 1ULL << (int)fmin(floor((atan2(y - ay, x - ax) + M_PI) / (2 * M_PI) * 64), 63);
            if (r % J->every == 0) { const double q = rr * 4 + 0.5; w->rt[(size_t)(r / J->every) * J->nb + gc_bin(J, atan2(y - ay, x - ax))] = q > 65535 ? 65535 : q < 1 ? 1 : (uint16_t)q; }
            lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
            if (nrs < 4096) rs[nrs++] = rr; else { const unsigned long long j = (lcg >> 33) % (unsigned long long)(w->nv + nvr); if (j < 4096) rs[j] = rr; }
        }
        if (nvr) { if (w->row0 < 0) w->row0 = r; w->row1 = r; }
        w->nv += nvr; w->zb_valid[zb] += (double)nvr / w->w; w->zb_n[zb]++;
    }
    for (int b = 0; b < J->zbins; b++) { if (w->zb_valid[b] > 0) { const double c = w->zb_valid[b] * w->w; w->zb_r[b] /= c; w->zb_dx[b] /= c; w->zb_dy[b] /= c; } if (w->zb_n[b]) w->zb_valid[b] /= w->zb_n[b]; }
    if (nrs) { qsort(rs, nrs, 8, gc_cmpd); w->rmed = rs[nrs / 2]; } else w->rmed = NAN;
    free(rs); free(X); free(V);
}
static double gc_local_pitch(const gc_job *J, int k, int trow, int b) {   /* median |gap| of the pairs (k-2, k-1) .. (k+1, k+2) */
    double g[4]; int n = 0;
    for (int a = k - 2; a <= k + 1; a++) {
        if (a < 0 || a + 1 >= J->nw || J->W[a + 1].id != J->W[a].id + 1) continue;
        const double r0 = gc_rt(&J->W[a], J, trow, b), r1 = gc_rt(&J->W[a + 1], J, trow, b);
        if (isfinite(r0) && isfinite(r1) && fabs(r1 - r0) > 0.25) g[n++] = fabs(r1 - r0);
    }
    if (!n) return J->gpitch;
    for (int i = 1; i < n; i++) for (int j = i; j > 0 && g[j] < g[j - 1]; j--) { const double t = g[j]; g[j] = g[j - 1]; g[j - 1] = t; }
    return n & 1 ? g[n / 2] : 0.5 * (g[n / 2 - 1] + g[n / 2]);
}
static void gc_pass2(gc_job *J, int k) {   /* per vertex metrics */
    gc_wrap *w = &J->W[k]; float *X; uint8_t *V; if (gc_read(w, &X, &V)) { fprintf(stderr, "geom-check: cannot read %s\n", w->path); atomic_store(&J->fail, 1); return; }
    const gc_wrap *wo = k + 1 < J->nw && J->W[k + 1].id == w->id + 1 ? &J->W[k + 1] : nullptr, *wi = k > 0 && J->W[k - 1].id == w->id - 1 ? &J->W[k - 1] : nullptr;
    const int ncr = (w->h + GC_CR - 1) / GC_CR; gc_cell *C = calloc((size_t)ncr * GC_CT, sizeof *C);
    double *rowsh = calloc(w->h, 8);
    for (int r = 0; r < w->h; r++) {
        double ssum = 0; long long sn = 0, nvr = 0; const int trow = (int)fmin(lround((double)r / J->every), J->tr - 1);
        for (int c = 0; c < w->w; c++) {
            const size_t i = (size_t)r * w->w + c; if (!V[i]) continue; nvr++;
            const float *p = X + 3 * i; double ay, ax; gc_axis(J, p[2], &ay, &ax);
            const double uy = p[1] - ay, ux = p[0] - ax, rr = hypot(uy, ux), th = atan2(uy, ux); const int b = gc_bin(J, th);
            const double pitch = gc_local_pitch(J, k, trow, b);
            int st = 0, ju = 0, me = 0, sk = 0;
            const float *pc = c + 1 < w->w && V[i + 1] ? X + 3 * (i + 1) : nullptr, *pr = r + 1 < w->h && V[i + w->w] ? X + 3 * (i + w->w) : nullptr;
            if (pc) { const double d = sqrt((pc[0] - p[0]) * (pc[0] - p[0]) + (pc[1] - p[1]) * (pc[1] - p[1]) + (pc[2] - p[2]) * (pc[2] - p[2])) / w->step; w->hc[gc_hbin(d)]++; ssum += d; sn++;
                      double ay2, ax2; gc_axis(J, pc[2], &ay2, &ax2); if (fabs(hypot(pc[1] - ay2, pc[0] - ax2) - rr) > J->jump_t * pitch) ju = 1; }
            if (pr) { const double d = sqrt((pr[0] - p[0]) * (pr[0] - p[0]) + (pr[1] - p[1]) * (pr[1] - p[1]) + (pr[2] - p[2]) * (pr[2] - p[2])) / w->step; w->hr[gc_hbin(d)]++;
                      double ay2, ax2; gc_axis(J, pr[2], &ay2, &ax2); if (fabs(hypot(pr[1] - ay2, pr[0] - ax2) - rr) > J->jump_t * pitch) ju = 1; }
            if (pc && pr) {
                const double a[3] = {pc[0] - p[0], pc[1] - p[1], pc[2] - p[2]}, e[3] = {pr[0] - p[0], pr[1] - p[1], pr[2] - p[2]};
                const double n[3] = {a[1] * e[2] - a[2] * e[1], a[2] * e[0] - a[0] * e[2], a[0] * e[1] - a[1] * e[0]}, nl = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
                if (nl > 0 && rr > 1e-6 && fabs((n[0] * ux + n[1] * uy) / (nl * rr)) < 0.34) st = 1;
            }
            const double ro = gc_rt(wo, J, trow, b), ri = gc_rt(wi, J, trow, b);
            if (isfinite(ro) || isfinite(ri)) {
                const double go = isfinite(ro) ? J->sgn * (ro - rr) / pitch : NAN, gi = isfinite(ri) ? J->sgn * (rr - ri) / pitch : NAN;
                if ((isfinite(go) && go < 0.3) || (isfinite(gi) && gi < 0.3)) me = 1;
                else if ((isfinite(go) && go > 1.8) || (isfinite(gi) && gi > 1.8)) sk = 1;
                w->nboth++;
            }
            w->steep += st; w->jump += ju; w->merge += me; w->skip += sk;
            gc_cell *q = &C[(size_t)(r / GC_CR) * GC_CT + (int)fmin(floor((th + M_PI) / (2 * M_PI) * GC_CT), GC_CT - 1)];
            q->n++; q->steep += st; q->jump += ju; q->merge += me; q->skip += sk; q->cx += p[0]; q->cy += p[1]; q->cz += p[2];
        }
        if (sn) w->hrm[gc_hbin(ssum / sn)]++;
        rowsh[r] = (double)nvr / w->w;
    }
    /* holes: rows inside the wrap's z extent whose valid share < half the median there */
    if (w->row0 >= 0) {
        const int n = w->row1 - w->row0 + 1; double *s = malloc((size_t)n * 8); memcpy(s, rowsh + w->row0, (size_t)n * 8); qsort(s, n, 8, gc_cmpd);
        const double med = s[n / 2]; w->rows_in = n; for (int r = w->row0; r <= w->row1; r++) w->holes += rowsh[r] < 0.5 * med; free(s);
    }
    pthread_mutex_lock(&J->mu);
    for (int cr = 0; cr < ncr; cr++) for (int ct = 0; ct < GC_CT; ct++) {
        gc_cell *q = &C[(size_t)cr * GC_CT + ct]; if (q->n < 64) continue;
        const int bad = q->steep + q->jump + q->merge + q->skip; if (bad < 0.25 * q->n) continue;
        q->wrap = w->id; q->cr = cr; q->ct = ct; q->cx /= q->n; q->cy /= q->n; q->cz /= q->n;
        if (J->ncell == J->ccap) { J->ccap = J->ccap ? 2 * J->ccap : 4096; J->cells = realloc(J->cells, J->ccap * sizeof *J->cells); }
        J->cells[J->ncell++] = *q;
    }
    pthread_mutex_unlock(&J->mu);
    free(C); free(rowsh); free(X); free(V);
}
static void *gc_worker(void *arg) {
    gc_job *J = arg;
    for (int k; (k = atomic_fetch_add(&J->next, 1)) < J->nw;) { if (J->pass == 1) gc_pass1(J, &J->W[k]); else gc_pass2(J, k); }
    return nullptr;
}
/* ---- --ct: z slip. Per sampled row pair (r, r + D) and wrap and angle bin: the CT's radial motion s of the sheets between the two
   heights (normalised cross-correlation of the two CT profiles along the ray around the wrap's radius, |s| <= 0.6 pitch, parabola
   refined) against the wrap's own radial motion r2 - r1: slip = (r2 - r1) - s. A wrap that follows its sheet has slip ~ 0; one that
   crosses sheets in z (a winding gauge drifting with z: the level set W = k slants through the turns) accumulates one pitch of slip per
   sheet crossed. The CT planes are read once per sampled row. */
typedef struct { gc_job *J; const uint8_t *p1, *p2; int64_t o[3], n[3]; double f, z1, z2; int tr1, tr2, sb; _Atomic int next; double *slip; long long *sn; double *sabs; long long *hs; } gc_slip;
static inline double gc_ct(const uint8_t *pl, const int64_t n[3], const int64_t o[3], double f, double y, double x) {   /* bilinear in one plane, wrap voxels */
    const double cy = (y + 0.5) * f - 0.5 - o[1], cx = (x + 0.5) * f - 0.5 - o[2];
    const int64_t iy = (int64_t)floor(cy), ix = (int64_t)floor(cx); if (iy < 0 || ix < 0 || iy + 1 >= n[1] || ix + 1 >= n[2]) return NAN;
    const double fy = cy - iy, fx = cx - ix; const uint8_t *q = pl + iy * n[2] + ix;
    return (1 - fy) * ((1 - fx) * q[0] + fx * q[1]) + fy * ((1 - fx) * q[n[2]] + fx * q[n[2] + 1]);
}
static void *gc_slip_run(void *arg) {
    gc_slip *S = arg; gc_job *J = S->J; double ay1, ax1, ay2, ax2; gc_axis(J, S->z1, &ay1, &ax1); gc_axis(J, S->z2, &ay2, &ax2);
    double p1[512], p2[512];
    for (int k; (k = atomic_fetch_add(&S->next, 1)) < J->nw;) {
        const gc_wrap *w = &J->W[k]; if (!w->rt) continue;
        const int zb = (int)((long long)S->tr1 * J->every * J->zbins / w->h);
        for (int b = (k * 7) % S->sb; b < J->nb; b += S->sb) {
            const double r1 = gc_rt(w, J, S->tr1, b), r2 = gc_rt(w, J, S->tr2, b); if (!isfinite(r1) || !isfinite(r2)) continue;
            const double P = gc_local_pitch(J, k, S->tr1, b), th = ((b + 0.5) / J->nb) * 2 * M_PI - M_PI, sn = sin(th), cs = cos(th);
            const int Wn = (int)fmin(ceil(1.5 * P / 0.5), 200), Sn = (int)fmin(ceil(0.6 * P / 0.5), 50);
            int ok = 1; double m1 = 0, v1 = 0;
            for (int t = -Wn; t <= Wn && ok; t++) { const double rr = r1 + 0.5 * t; const double v = gc_ct(S->p1, S->n, S->o, S->f, ay1 + rr * sn, ax1 + rr * cs); ok = isfinite(v); p1[t + Wn] = v; m1 += v; }
            for (int t = -Wn - Sn; t <= Wn + Sn && ok; t++) { const double rr = r1 + 0.5 * t; const double v = gc_ct(S->p2, S->n, S->o, S->f, ay2 + rr * sn, ax2 + rr * cs); ok = isfinite(v); p2[t + Wn + Sn] = v; }
            if (!ok) continue;
            const int L = 2 * Wn + 1; m1 /= L; for (int t = 0; t < L; t++) v1 += (p1[t] - m1) * (p1[t] - m1);
            if (v1 < L * 25.0) continue;   /* (flat CT: no sheets to follow) */
            double cc[101]; int best = -1;
            for (int s = -Sn; s <= Sn; s++) {
                double m2 = 0, v2 = 0, c = 0; for (int t = 0; t < L; t++) m2 += p2[t + Sn + s]; m2 /= L;
                for (int t = 0; t < L; t++) { const double d = p2[t + Sn + s] - m2; v2 += d * d; c += (p1[t] - m1) * d; }
                cc[s + Sn] = v2 > 0 ? c / sqrt(v1 * v2) : -1; if (best < 0 || cc[s + Sn] > cc[best]) best = s + Sn;
            }
            if (cc[best] < 0.5 || best == 0 || best == 2 * Sn) continue;
            const double a = cc[best - 1], c0 = cc[best], d = cc[best + 1], den = a - 2 * c0 + d, sub = den < 0 ? 0.5 * (a - d) / den : 0;
            const double s = 0.5 * (best - Sn + sub), slip = ((r2 - r1) - s) / P;
            S->slip[(size_t)k * J->zbins + zb] += slip; S->sabs[(size_t)k * J->zbins + zb] += fabs(slip); S->sn[(size_t)k * J->zbins + zb]++;
            int hb = (int)floor((slip + 1) * 50); hb = hb < 0 ? 0 : hb > 99 ? 99 : hb; S->hs[k * 100 + hb]++;
        }
    }
    return nullptr;
}
static int gc_cmp_id(const void *a, const void *b) { const gc_wrap *x = a, *y = b; return (x->id > y->id) - (x->id < y->id); }
static int gc_cmp_cell(const void *a, const void *b) {
    const gc_cell *x = a, *y = b; const double bx = (double)(x->steep + x->jump + x->merge + x->skip) / x->n, by = (double)(y->steep + y->jump + y->merge + y->skip) / y->n;
    return (bx < by) - (bx > by);
}

int cmd_geom_check(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: ufsm geom-check <wraps-dir> --axis FILE --axis-um U [--out DIR] [--first K] [--last K] [--every 2] [--nb 2048] [--zbins 32] [--jump 0.5]\n"
                                    "       [--threads N] [--max-steep F] [--max-jump F] [--max-merge F] [--max-skip F] [--ct ROOT KEY --ct-um U [--slip-every 128] [--slip-d 2] [--slip-sb 8]]   (src/geomcheck.c)\n"); return 2; }
    const char *in = argv[2], *axp = gc_opt(argc, argv, "--axis", nullptr), *out = gc_opt(argc, argv, "--out", nullptr);
    const double axum = atof(gc_opt(argc, argv, "--axis-um", "0"));
    const int k0 = atoi(gc_opt(argc, argv, "--first", "-1000000")), k1 = atoi(gc_opt(argc, argv, "--last", "1000000"));
    const int nt = ufsm_threads(atoi(gc_opt(argc, argv, "--threads", "0")), 16);
    gc_job J; memset(&J, 0, sizeof J);
    J.every = atoi(gc_opt(argc, argv, "--every", "2")); J.nb = atoi(gc_opt(argc, argv, "--nb", "2048")); J.zbins = atoi(gc_opt(argc, argv, "--zbins", "32")); J.jump_t = atof(gc_opt(argc, argv, "--jump", "0.5"));
    const double mx[4] = {atof(gc_opt(argc, argv, "--max-steep", "0")), atof(gc_opt(argc, argv, "--max-jump", "0")), atof(gc_opt(argc, argv, "--max-merge", "0")), atof(gc_opt(argc, argv, "--max-skip", "0"))};
    if (!axp || !(axum > 0) || J.every < 1 || J.nb < 64 || J.zbins < 1 || !(J.jump_t > 0)) { fprintf(stderr, "geom-check: --axis FILE --axis-um U (its level-0 um), --every >= 1, --nb >= 64, --zbins >= 1\n"); return 2; }
    axis ax = {0}; if (axis_load(&ax, axp)) { fprintf(stderr, "geom-check: cannot load the axis %s\n", axp); return 1; }
    DIR *dd = opendir(in); if (!dd) { fprintf(stderr, "geom-check: cannot open %s\n", in); return 1; }
    int nw = 0, cap = 64; gc_wrap *W = calloc(cap, sizeof *W); struct dirent *de;
    while ((de = readdir(dd))) {
        int id; char tail[8] = ""; if (sscanf(de->d_name, "w%d.%7s", &id, tail) != 2 || strcmp(tail, "sfc") || id < k0 || id > k1) continue;
        if (nw == cap) { cap *= 2; W = realloc(W, cap * sizeof *W); }
        memset(&W[nw], 0, sizeof W[nw]); W[nw].id = id; snprintf(W[nw].path, sizeof W[nw].path, "%s/%s", in, de->d_name); nw++;
    }
    closedir(dd);
    if (!nw) { fprintf(stderr, "geom-check: no wNNN.sfc in %s\n", in); return 1; }
    qsort(W, nw, sizeof *W, gc_cmp_id);
    int H = -1;
    for (int k = 0; k < nw; k++) {
        sfc_reader *r; if (sfc_open_file(W[k].path, &r)) { fprintf(stderr, "geom-check: cannot open %s\n", W[k].path); return 1; }
        int c = sfc_find_channel(r, "x"); if (c < 0) c = 0; const sfc_channel *ch = sfc_channel_info(r, (uint32_t)c); W[k].w = (int)ch->width; W[k].h = (int)ch->height;
        const uint64_t ml = sfc_metadata_size(r); char *meta = calloc(ml + 1, 1); if (ml) sfc_read_metadata(r, meta, ml); sfc_close(r);
        const char *q = strstr(meta, "\"voxel_size_um\":"); W[k].um = q ? atof(q + 16) : 0;
        q = strstr(meta, "\"scale\":"); if (q) { q = strchr(q, '['); if (q && atof(q + 1) > 0) W[k].step = 1 / atof(q + 1); }
        free(meta);
        if (!(W[k].um > 0) || !(W[k].step > 0)) { fprintf(stderr, "geom-check: %s: no voxel_size_um / scale\n", W[k].path); return 1; }
        if (H >= 0 && W[k].h != H) { fprintf(stderr, "geom-check: %s has %d rows, others %d (one ufsm wraps run)\n", W[k].path, W[k].h, H); return 1; }
        H = W[k].h;
    }
    const double um = W[0].um, step = W[0].step;
    J.W = W; J.nw = nw; J.ax = &ax; J.s0 = um / axum; J.tr = (H + J.every - 1) / J.every; pthread_mutex_init(&J.mu, nullptr);
    const double t0 = gc_now();
    fprintf(stderr, "geom-check: %d wraps w%03d..w%03d, %d rows, step %g, %g um, tables %d x %d (%.2f GB), %d threads\n", nw, W[0].id, W[nw - 1].id, H, step, um, J.tr, J.nb, (double)nw * J.tr * J.nb * 2 / 1073741824.0, nt);
    pthread_t th[64]; const int m = nt < 64 ? nt : 64;
    J.pass = 1; for (int t = 0; t < m; t++) pthread_create(&th[t], nullptr, gc_worker, &J); for (int t = 0; t < m; t++) pthread_join(th[t], nullptr);
    if (atomic_load(&J.fail)) return 1;
    /* the sense (radius grows with id: +1) and the global pitch: median of id-adjacent gaps over a sample of table cells */
    { double *g = malloc(sizeof(double) * 200000); int ng = 0; long pos = 0, neg = 0;
      for (int k = 0; k + 1 < nw; k++) { if (W[k + 1].id != W[k].id + 1) continue;
          for (int tr = 0; tr < J.tr; tr += 7) for (int b = 0; b < J.nb; b += 13) { const double a = gc_rt(&W[k], &J, tr, b), c = gc_rt(&W[k + 1], &J, tr, b);
              if (!isfinite(a) || !isfinite(c)) continue;
              if (c > a) pos++; else neg++;
              if (ng < 200000) g[ng++] = fabs(c - a); } }
      J.sgn = pos >= neg ? 1 : -1; if (ng) { qsort(g, ng, 8, gc_cmpd); J.gpitch = g[ng / 2]; } else J.gpitch = 8 * step; free(g);
      fprintf(stderr, "geom-check: radius grows with id: %s (%ld vs %ld table cells); median gap between id-adjacent wraps %.2f voxels (%.1f um); pass 1 %.1f s\n",
              J.sgn > 0 ? "yes" : "no (decreases)", pos, neg, J.gpitch, J.gpitch * um, gc_now() - t0); }
    for (int k = 0; k < nw; k++) {
        double g[64]; int n = 0;
        if (k + 1 < nw && W[k + 1].id == W[k].id + 1) for (int tr = 0; tr < J.tr; tr += J.tr / 8 + 1) for (int b = 0; b < J.nb && n < 64; b += J.nb / 8) { const double a = gc_rt(&W[k], &J, tr, b), c = gc_rt(&W[k + 1], &J, tr, b); if (isfinite(a) && isfinite(c)) g[n++] = fabs(c - a); }
        if (n) { qsort(g, n, 8, gc_cmpd); W[k].pitch = g[n / 2]; } else W[k].pitch = NAN;
    }
    J.pass = 2; atomic_store(&J.next, 0); for (int t = 0; t < m; t++) pthread_create(&th[t], nullptr, gc_worker, &J); for (int t = 0; t < m; t++) pthread_join(th[t], nullptr);
    if (atomic_load(&J.fail)) return 1;
    /* --ct: z slip of the wraps against the CT's sheets */
    double *slip = nullptr, *sabs = nullptr; long long *sn = nullptr, *hs = nullptr; int ci = -1;
    for (int i = 2; i + 2 < argc; i++) if (!strcmp(argv[i], "--ct")) ci = i;
    if (ci > 0) {
        const double ctum = atof(gc_opt(argc, argv, "--ct-um", "0")); const int se = atoi(gc_opt(argc, argv, "--slip-every", "128")), D = atoi(gc_opt(argc, argv, "--slip-d", "2")), sb = atoi(gc_opt(argc, argv, "--slip-sb", "8"));
        if (!(ctum > 0) || se < 1 || D < 1 || D % J.every || sb < 1) { fprintf(stderr, "geom-check: --ct ROOT KEY --ct-um U, --slip-every >= 1, --slip-d a multiple of --every, --slip-sb >= 1\n"); return 2; }
        z3 *cz = z3_open(store_open(argv[ci + 1]), argv[ci + 2], gc_opt(argc, argv, "--cache", nullptr)); if (!cz) { fprintf(stderr, "geom-check: CT: %s\n", z3_error()); return 1; }
        const z3_meta *cm = z3_meta_of(cz); const double f = um / ctum;
        slip = calloc((size_t)nw * J.zbins, 8); sabs = calloc((size_t)nw * J.zbins, 8); sn = calloc((size_t)nw * J.zbins, 8); hs = calloc((size_t)nw * 100, 8);
        const double ts = gc_now(); int nrows = 0;
        for (int r = se / 2 - (se / 2) % J.every; r + D < H; r += se) {
            double z1 = NAN, z2 = NAN, rmax = 0; for (int k = 0; k < nw; k++) { if (!isfinite(z1)) z1 = W[k].rz[r]; if (!isfinite(z2)) z2 = W[k].rz[r + D]; }
            if (!isfinite(z1) || !isfinite(z2)) continue;
            const int tr1 = r / J.every, tr2 = (r + D) / J.every;
            for (int k = 0; k < nw; k++) for (int b = 0; b < J.nb; b += 4) { const double v = gc_rt(&W[k], &J, tr1, b); if (isfinite(v) && v > rmax) rmax = v; }
            if (!(rmax > 0)) continue;
            double ay, ax; gc_axis(&J, z1, &ay, &ax); const double R = rmax + 64;
            gc_slip S = {&J, nullptr, nullptr, {0, 0, 0}, {1, 0, 0}, f, z1, z2, tr1, tr2, sb, 0, slip, sn, sabs, hs};
            int64_t lo[2] = {(int64_t)floor(((ay - R) + 0.5) * f - 0.5), (int64_t)floor(((ax - R) + 0.5) * f - 0.5)}, hi[2] = {(int64_t)ceil(((ay + R) + 0.5) * f + 0.5), (int64_t)ceil(((ax + R) + 0.5) * f + 0.5)};
            for (int d = 0; d < 2; d++) { if (lo[d] < 0) lo[d] = 0; if (hi[d] > cm->shape[d + 1]) hi[d] = cm->shape[d + 1]; }
            if (hi[0] <= lo[0] || hi[1] <= lo[1]) continue;
            S.o[1] = lo[0]; S.o[2] = lo[1]; S.n[1] = hi[0] - lo[0]; S.n[2] = hi[1] - lo[1];
            uint8_t *p1 = malloc((size_t)S.n[1] * S.n[2]), *p2 = malloc((size_t)S.n[1] * S.n[2]);
            int64_t o1[3] = {llround((z1 + 0.5) * f - 0.5), lo[0], lo[1]}, o2[3] = {llround((z2 + 0.5) * f - 0.5), lo[0], lo[1]};
            if (!p1 || !p2 || o2[0] >= cm->shape[0] || z3_read(cz, o1, S.n, p1, nt) || z3_read(cz, o2, S.n, p2, nt)) { free(p1); free(p2); continue; }
            S.p1 = p1; S.p2 = p2;
            for (int t = 0; t < m; t++) pthread_create(&th[t], nullptr, gc_slip_run, &S);
            for (int t = 0; t < m; t++) pthread_join(th[t], nullptr);
            free(p1); free(p2); nrows++;
        }
        z3_close(cz);
        long long hh[100] = {0}, nn = 0; double sa = 0; for (int k = 0; k < nw; k++) for (int b = 0; b < 100; b++) hh[b] += hs[k * 100 + b];
        for (size_t i = 0; i < (size_t)nw * J.zbins; i++) { nn += sn[i]; sa += sabs[i]; }
        long long c = 0; double q[3] = {NAN, NAN, NAN}; const double qq[3] = {.1, .5, .9};
        for (int b = 0, j = 0; b < 100 && j < 3; b++) { c += hh[b]; while (j < 3 && c >= qq[j] * nn) q[j++] = (b + 0.5) / 50 - 1; }
        fprintf(stderr, "geom-check: z slip (CT, %d row pairs %d rows apart): %lld samples, slip / pitch p10 %.3f p50 %.3f p90 %.3f, mean |slip| %.3f pitch per %d rows (%.1f um of z); %.1f s\n",
                nrows, D, nn, q[0], q[1], q[2], nn ? sa / nn : 0, D, D * step * um, gc_now() - ts);
    }
    /* totals */
    long long T[8] = {0}; long long hc[GC_NH] = {0}, hr[GC_NH] = {0}, hrm[GC_NH] = {0}; int nbad = 0;
    for (int k = 0; k < nw; k++) {
        gc_wrap *w = &W[k]; T[0] += w->nv; T[1] += w->steep; T[2] += w->jump; T[3] += w->merge; T[4] += w->skip; T[5] += w->nboth; T[6] += w->holes; T[7] += w->rows_in;
        for (int b = 0; b < GC_NH; b++) { hc[b] += w->hc[b]; hr[b] += w->hr[b]; hrm[b] += w->hrm[b]; }
        const double nv = w->nv ? (double)w->nv : 1, sh[4] = {w->steep / nv, w->jump / nv, w->merge / nv, w->skip / nv};
        for (int a = 0; a < 4; a++) if (mx[a] > 0 && sh[a] > mx[a]) w->bad = 1;
        nbad += w->bad;
    }
    const double N = T[0] ? (double)T[0] : 1, sh[4] = {T[1] / N, T[2] / N, T[3] / N, T[4] / N};
    char sum[4096]; int sl = 0;
    sl += snprintf(sum + sl, sizeof sum - sl, "geom-check: %d wraps, %lld vertices (%.0f cm2), pitch %.2f voxels\n", nw, T[0], T[0] * step * step * um * um * 1e-8, J.gpitch);
    sl += snprintf(sum + sl, sizeof sum - sl, "  steep %.4f  jump %.4f  merge %.4f  skip %.4f (of vertices; merge/skip of %.3f with a neighbour)  hole rows %.4f\n", sh[0], sh[1], sh[2], sh[3], T[5] / N, T[7] ? (double)T[6] / T[7] : 0);
    sl += snprintf(sum + sl, sizeof sum - sl, "  column spacing / step p05 %.3f p50 %.3f p95 %.3f p99 %.3f; per-row mean p05 %.3f p50 %.3f p95 %.3f; row spacing p50 %.3f p95 %.3f p99 %.3f\n",
                   gc_hq(hc, .05), gc_hq(hc, .5), gc_hq(hc, .95), gc_hq(hc, .99), gc_hq(hrm, .05), gc_hq(hrm, .5), gc_hq(hrm, .95), gc_hq(hr, .5), gc_hq(hr, .95), gc_hq(hr, .99));
    if (mx[0] > 0 || mx[1] > 0 || mx[2] > 0 || mx[3] > 0) sl += snprintf(sum + sl, sizeof sum - sl, "  guard: %d of %d wraps over the limits\n", nbad, nw);
    fputs(sum, stdout);
    if (out) {
        char p[1600]; snprintf(p, sizeof p, "mkdir -p '%s'", out); if (system(p)) return 1;
        snprintf(p, sizeof p, "%s/summary.txt", out); FILE *f = fopen(p, "w"); if (f) { fputs(sum, f); fclose(f); }
        snprintf(p, sizeof p, "%s/wraps.tsv", out); f = fopen(p, "w");
        if (f) {
            fprintf(f, "wrap\tcols\tvalid\tcm2\trow0\trow1\tr_med\tpitch\tsteep\tjump\tmerge\tskip\thole_rows\tcolsp_p05\tcolsp_p50\tcolsp_p95\trowmean_p05\trowmean_p95\trowsp_p95\tbad\n");
            for (int k = 0; k < nw; k++) { const gc_wrap *w = &W[k]; const double nv = w->nv ? (double)w->nv : 1;
                fprintf(f, "w%03d\t%d\t%lld\t%.2f\t%d\t%d\t%.1f\t%.2f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%d\n", w->id, w->w, w->nv, w->nv * step * step * um * um * 1e-8,
                        w->row0, w->row1, w->rmed, w->pitch, w->steep / nv, w->jump / nv, w->merge / nv, w->skip / nv, w->rows_in ? (double)w->holes / w->rows_in : 0,
                        gc_hq(w->hc, .05), gc_hq(w->hc, .5), gc_hq(w->hc, .95), gc_hq(w->hrm, .05), gc_hq(w->hrm, .95), gc_hq(w->hr, .95), w->bad); }
            fclose(f);
        }
        snprintf(p, sizeof p, "%s/zbins.tsv", out); f = fopen(p, "w");
        if (f) { fprintf(f, "wrap\tzbin\trow0\tvalid_share\tr_mean\tdx\tdy\tcover\n");
                 for (int k = 0; k < nw; k++) for (int b = 0; b < J.zbins; b++) fprintf(f, "w%03d\t%d\t%d\t%.4f\t%.1f\t%.1f\t%.1f\t%.3f\n", W[k].id, b, (int)((long long)b * H / J.zbins), W[k].zb_valid[b], W[k].zb_r[b], W[k].zb_dx[b], W[k].zb_dy[b], __builtin_popcountll(W[k].zb_cov[b]) / 64.0);
                 fclose(f); }
        if (slip) { snprintf(p, sizeof p, "%s/slip.tsv", out); f = fopen(p, "w");
            if (f) { fprintf(f, "wrap\tzbin\tn\tslip_mean\tslip_abs\n");
                     for (int k = 0; k < nw; k++) for (int b = 0; b < J.zbins; b++) { const size_t i = (size_t)k * J.zbins + b; if (sn[i]) fprintf(f, "w%03d\t%d\t%lld\t%.4f\t%.4f\n", W[k].id, b, sn[i], slip[i] / sn[i], sabs[i] / sn[i]); }
                     fclose(f); } }
        qsort(J.cells, J.ncell, sizeof *J.cells, gc_cmp_cell);
        snprintf(p, sizeof p, "%s/cells.tsv", out); f = fopen(p, "w");
        if (f) { fprintf(f, "wrap\trow0\tturn_frac\tn\tsteep\tjump\tmerge\tskip\tx\ty\tz\n");
                 for (size_t i = 0; i < J.ncell; i++) { const gc_cell *q = &J.cells[i];
                     fprintf(f, "w%03d\t%d\t%.4f\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.1f\t%.1f\t%.1f\n", q->wrap, q->cr * GC_CR, (q->ct + 0.5) / GC_CT, q->n, (double)q->steep / q->n, (double)q->jump / q->n, (double)q->merge / q->n, (double)q->skip / q->n, q->cx, q->cy, q->cz); }
                 fclose(f); }
    }
    fprintf(stderr, "geom-check: %.1f s\n", gc_now() - t0);
    for (int k = 0; k < nw; k++) { free(W[k].rt); free(W[k].zb_valid); free(W[k].zb_r); free(W[k].zb_n); free(W[k].zb_dx); free(W[k].zb_dy); free(W[k].zb_cov); free(W[k].rz); }
    free(slip); free(sabs); free(sn); free(hs);
    free(W); free(J.cells);
    int over = 0; for (int a = 0; a < 4; a++) over |= mx[a] > 0 && sh[a] > mx[a];
    return over ? 3 : 0;
}
