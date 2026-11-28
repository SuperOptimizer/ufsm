/* ufsm count-field IN-FIELDS OUT-FIELDS: winding from the network's sheet COUNTS (density x normal) instead of its wrapped phase.
   The density head counts packed sheets (radial 0.5 mm segments: wind96 density off by >= 0.5 turn on 36% of packed PH1 segments,
   its phase on 92%, the phase skipping sheets), so:
   1. U (unwrapped radius in turns, W = U - dir theta / 2 pi as in src/fit.h) from grad W = rho n: n the principal axis of the
      cell's n n^T oriented outward from the umbilicus, rho = 8 / sp turns per field cell; weighted least squares on a pooled grid
      (cells P x P x P field cells), coarse to fine conjugate gradients, screened weakly toward the label-free prior r / pitch + offset
      (fit_init_prior: also the direction). --method ray: U integrated along umbilicus rays instead (villa-style per-ray counting).
   2. The phase sets the sub-turn position: the residual phi - W (confidence-weighted phasors) box-averaged over +-R field cells is
      added to W (W_out = W + arg <c e^{2 pi i (phi - W)}> / 2 pi); where the phase rotates with W this is the phase, where it stalls
      or skips sheets the average loses the residual ramp and W's counting shows. --R 0: W only (the global gauge from the phase).
   3. The closing error: the circulation of the integrand (rho n + theta term) around L x L squares of the solve grid in the 3 planes
      (turns; 0 for a consistent count field), the confidence of the count field.
   Output: a raw fit-field set (all input planes, pc / ps replaced by W_out's phase), plus wc.f32 (W_out per field cell, turns,
   NaN unknown; fields.json "wcount": true) and cl.raw (closing error x 255 turns, capped). --model DIR: W as a fit model (pitch /
   offset knots of the prior, delta on the solve lattice) for fit --seed-model DIR (the voxel unwrapping takes its integers from W,
   the phase only the sub-turn part: round(W - phi) + phi), its gauge from the phase smoothed over +-Rm solve cells. */
#include "defaulted.h"
#include <complex.h>
#include "optcheck.h"
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fit.h"
#include "sysinfo.h"
#include "z3w.h"
#include "spiralfit.h"
#include "zarr3.h"
#include "store.h"
#define CFGPU_WEAK __attribute__((weak))   /* src/nn_cf.cu: a binary without it runs the CPU code */
#include "cfgpu.h"
#include "afv.h"
#include "l1pot.h"
#include <limits.h>
#include <stdatomic.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>

static const char *cf_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(k))) return argv[i + 1]; return d; }

/* ---- a pthread parallel for over [0, n) in nt contiguous parts; fn(ctx, a, b, t) ---- */
typedef void (*cf_fn)(void *, long, long, int);
typedef struct { cf_fn fn; void *ctx; long a, b; int t; } cf_job;
static void *cf_run(void *p) { cf_job *j = p; j->fn(j->ctx, j->a, j->b, j->t); return nullptr; }
static int cf_nt = 8;
enum { CF_MAXT = 64 };
static void cf_par(long n, cf_fn fn, void *ctx) {
    int nt = cf_nt < 1 ? 1 : cf_nt > CF_MAXT ? CF_MAXT : cf_nt; if (n < nt) nt = n > 0 ? (int)n : 1;
    pthread_t th[CF_MAXT]; cf_job jb[CF_MAXT]; int up[CF_MAXT] = {0};
    for (int t = 0; t < nt; t++) jb[t] = (cf_job){fn, ctx, n * t / nt, n * (t + 1) / nt, t};
    for (int t = 1; t < nt; t++) up[t] = pthread_create(&th[t], nullptr, cf_run, &jb[t]) == 0;
    cf_run(&jb[0]);
    for (int t = 1; t < nt; t++) { if (up[t]) pthread_join(th[t], nullptr); else cf_run(&jb[t]); }
}

static double cf_axis_at(const float *v, int n, double z) { if (z <= 0) return v[0]; if (z >= n - 1) return v[n - 1]; const int i = (int)z; const double t = z - i; return v[i] + t * (v[i + 1] - v[i]); }
static double cf_knot(const float *v, int nk, int dz, double z) { double q = z / dz; if (q <= 0) return v[0]; if (q >= nk - 1) return v[nk - 1]; const int i = (int)q; return v[i] + (q - i) * (v[i + 1] - v[i]); }

/* the principal eigenvector of a cell's n n^T and its eigenvalue (power iteration, as fit-compare) */
static double cf_axis(const fit_fields *f, size_t i, double v[3]) {
    double T[3][3]; const double t[6] = {f->t[0][i] / 255.0, f->t[1][i] / 255.0, f->t[2][i] / 255.0, f->t[3][i] / 255.0 - 0.5, f->t[4][i] / 255.0 - 0.5, f->t[5][i] / 255.0 - 0.5};
    T[0][0] = t[0]; T[1][1] = t[1]; T[2][2] = t[2]; T[0][1] = T[1][0] = t[3]; T[0][2] = T[2][0] = t[4]; T[1][2] = T[2][1] = t[5];
    int k = T[0][0] >= T[1][1] && T[0][0] >= T[2][2] ? 0 : T[1][1] >= T[2][2] ? 1 : 2; v[0] = v[1] = v[2] = 0; v[k] = 1;
    double l = 0;
    for (int it = 0; it < 16; it++) {
        double w[3] = {0, 0, 0}; for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) w[a] += T[a][b] * v[b];
        l = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]); if (l < 1e-12) return 0; for (int a = 0; a < 3; a++) v[a] = w[a] / l;
    }
    const double tr = t[0] + t[1] + t[2]; return tr > 1e-6 ? l / tr : 0;
}
static double cf_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
/* ---- chunk loads one ahead on a thread (the decode overlaps the pooling / the GPU; two chunks held) ---- */
/* (nxt: a used chunk handed back by cf_pre_give (main thread only); cf_pre_start moves it to use, owned by the load thread, whose
   load takes its planes instead of fresh pages: fit_fields_recycle) */
typedef struct { const char *in; int z0, z1, rc, th_on, busy, has_use, has_nxt; fit_fields_buf b, use, nxt; pthread_t th; double t; } cf_pre;
/* --follow FILE (pass 1 behind a running fields predict): lines "z0 z1" of field rows already written (box cells, any order, appended
   as the predict's field slabs are flushed; the driver's streamer writes them). A chunk is loaded once its rows are covered (polled) */
static const char *g_follow;
static int cf_follow_ready(int z0, int z1) {
    FILE *f = fopen(g_follow, "r"); if (!f) return 0;
    enum { NR = 4096 }; static int a[NR], b[NR]; int n = 0, x, y;
    while (n < NR && fscanf(f, "%d %d", &x, &y) == 2) { a[n] = x; b[n] = y; n++; }
    fclose(f);
    for (int z = z0; z < z1;) { int adv = 0; for (int i = 0; i < n; i++) if (a[i] <= z && z < b[i]) { z = b[i]; adv = 1; } if (!adv) return 0; }
    return 1;
}
static void *cf_pre_run(void *a) { cf_pre *p = a;
    if (g_follow) { const double t0 = cf_now(); int said = 0; while (!cf_follow_ready(p->z0, p->z1)) { if (!said && cf_now() - t0 > 600) { fprintf(stderr, "count-field: --follow: rows %d..%d not written after 10 min, still waiting\n", p->z0, p->z1); said = 1; } usleep(500000); } }
    if (p->has_use) { fit_fields_recycle(1); p->rc = fit_fields_load_ex(p->in, p->z0, p->z1, 1, &p->b, 0, nullptr, 0, &p->use); fit_fields_recycle(0); p->has_use = 0; }
    else p->rc = fit_fields_load(p->in, p->z0, p->z1, 1, &p->b);
    return nullptr; }
static void cf_pre_give(cf_pre *p, fit_fields_buf *b) {   /* a chunk done with: its planes for the next cf_pre_start's load */
    if (p->has_nxt || getenv("UFSM_CF_NO_RECYCLE")) { fit_fields_free(b); return; }
    p->nxt = *b; p->has_nxt = 1; memset(b, 0, sizeof *b);
}
static void cf_pre_start(cf_pre *p, const char *in, int z0, int z1, int async) {
    p->in = in; p->z0 = z0; p->z1 = z1; p->rc = 0; p->busy = 1;
    if (p->has_nxt) { p->use = p->nxt; p->has_use = 1; p->has_nxt = 0; memset(&p->nxt, 0, sizeof p->nxt); }
    p->th_on = async && pthread_create(&p->th, nullptr, cf_pre_run, p) == 0;
    if (!p->th_on) cf_pre_run(p);
}
static int cf_pre_take(cf_pre *p, fit_fields_buf *b) {   /* waits for the chunk started last; its buffer moves to b (time waited -> p->t) */
    if (!p->busy) return -1;
    const double t0 = cf_now(); if (p->th_on) pthread_join(p->th, nullptr); p->t += cf_now() - t0; p->busy = 0; p->th_on = 0;
    if (p->rc) { fprintf(stderr, "count-field: cannot load rows %d..%d of %s\n", p->z0, p->z1, p->in); return -1; }
    *b = p->b; memset(&p->b, 0, sizeof p->b); return 0;
}
static double cf_smooth(double x, double a, double b) { const double t = x <= a ? 0 : x >= b ? 1 : (x - a) / (b - a); return t * t * (3 - 2 * t); }
/* ---- the field cells pooled onto a solve grid: S = sum c, G = sum c rho n (turns per field cell) ---- */
typedef struct {
    const fit_fields *f; int zoff, nzl;   /* a chunk of field rows [zoff, zoff + nzl) (f's row 0 = zoff) */
    int H, W, P, hh, w; double h; const float *cy, *cx; int na;   /* the whole box's axis table (box-local) */
    float *S, *G[3];
    double emin;   /* weight floor of the orientation |n . r| ramp */
    double sheet0; /* eigenvalue ratio at which the sheetness weight starts */
    double kph, c0;   /* --phase-w kph: confident network phase gradients join the targets (weight kph x ramp(conf, c0, c0 + 0.3)) */
    int nb;           /* y bands per solve row (jobs = solve rows x bands: disjoint solve cells per job; 0 = 1) */
    float *Sd, *Gd[3];   /* --pool-split 1: the density term alone (S, G hold density + phase); nullptr: none */
    double rs, ra, rb, wa, wb, wd, wp;   /* rs: --rho-scale (the density head's turn count x rs; 0 = 1); ra, rb: --rho-pow A,S0 (x (spacing / S0)^-A, rb = S0 in sp bytes); --wide A,B (sp bytes) --wide-dens wd --wide-phase wp: where the network spacing is wide the density weight x (1 - wd s), the phase weight x (1 + wp s), s = ramp(sp, wa, wb) */
} cf_pool;
static double cf_ph(const fit_fields *f, size_t i, double *c) { const double a = (f->pc[i] - 128) / 127.0, b = (f->ps[i] - 128) / 127.0; *c = f->w[i] ? hypot(a, b) : 0; return atan2(b, a) / (2 * M_PI); }
static double cf_wrap(double d) { return d - round(d); }
static void cf_pool_run(void *a, long j0, long j1, int t_) {   /* jobs [j0, j1): solve row zoff / P + j / nb, y band j % nb */
    cf_pool *p = a; const fit_fields *f = p->f; const size_t HW = (size_t)p->H * p->W;
    const int nb = p->nb > 0 ? p->nb : 1;
    for (long jb = j0; jb < j1; jb++) for (int dz = 0; dz < p->P; dz++) {
        const long Zr = jb / nb; const int bd = (int)(jb % nb), ya = (int)((long)p->hh * bd / nb) * p->P, yb = (int)fmin(p->H, (double)((long)p->hh * (bd + 1) / nb) * p->P);
        const long Z = p->zoff / p->P + Zr; const int z = (int)Z * p->P + dz, zl = z - p->zoff; if (zl < 0) continue; if (zl >= p->nzl) break;
        const double pz = (z + 0.5) * p->h, cy = cf_axis_at(p->cy, p->na, pz), cx = cf_axis_at(p->cx, p->na, pz);
        for (int y = ya; y < yb; y++) for (int x = 0; x < p->W; x++) {
            const size_t i = zl * HW + (size_t)y * p->W + x; if (!f->w[i]) continue;
            if (p->kph > 0 && zl > 0 && zl + 1 < p->nzl && y > 0 && y + 1 < p->H && x > 0 && x + 1 < p->W) {   /* the phase gradient (central wrapped differences) */
                double c0v, cm, cp; const double p0 = cf_ph(f, i, &c0v); const size_t st[3] = {HW, (size_t)p->W, 1}; double gph[3], cmin = c0v;
                for (int k = 0; k < 3; k++) { const double pm = cf_ph(f, i - st[k], &cm), pp = cf_ph(f, i + st[k], &cp); cmin = fmin(cmin, fmin(cm, cp)); gph[k] = 0.5 * (cf_wrap(pp - p0) + cf_wrap(p0 - pm)); }
                const double cw = p->kph * cf_smooth(cmin, p->c0, p->c0 + 0.3) * (p->wp > 0 ? 1 + p->wp * cf_smooth(f->sp[i], p->wa, p->wb) : 1);
                if (cw > 0) { const size_t j = ((size_t)Z * p->hh + y / p->P) * p->w + x / p->P; p->S[j] += (float)cw; for (int k = 0; k < 3; k++) p->G[k][j] += (float)(cw * gph[k]); }
            }
            if (!f->sp[i]) continue;
            double v[3]; const double lr = cf_axis(f, i, v); if (lr <= 0) continue;
            double ry = (y + 0.5) * p->h - cy, rx = (x + 0.5) * p->h - cx; const double rl = hypot(ry, rx); if (rl < 1) continue; ry /= rl; rx /= rl;
            double dot = v[1] * ry + v[2] * rx; if (dot < 0) { v[0] = -v[0]; v[1] = -v[1]; v[2] = -v[2]; dot = -dot; }
            const double c = f->w[i] / 255.0 * cf_smooth(dot, p->emin, p->emin + 0.25) * cf_smooth(lr, p->sheet0, p->sheet0 + 0.3) * (p->wd > 0 ? 1 - p->wd * cf_smooth(f->sp[i], p->wa, p->wb) : 1);
            if (c <= 0) continue;
            const double rho = p->rs * 8.0 / f->sp[i] * (p->ra != 0 ? pow(f->sp[i] / p->rb, -p->ra) : 1);   /* turns per field cell (x --rho-scale, x (sp / S0)^-A: --rho-pow) */
            const size_t j = ((size_t)Z * p->hh + y / p->P) * p->w + x / p->P;
            p->S[j] += (float)c; for (int k = 0; k < 3; k++) p->G[k][j] += (float)(c * rho * v[k]);
            if (p->Sd) { p->Sd[j] += (float)c; for (int k = 0; k < 3; k++) p->Gd[k][j] += (float)(c * rho * v[k]); }
        }
    }
}
/* ---- one solve level: unknown U per cell, edges along +z +y +x ---- */
typedef struct {
    int d, hh, w, P; double h;   /* grid dims, cells per solve cell, native voxels per field cell */
    size_t n; float *wt[3], *tg[3], *b, *diag, *U, *r, *p, *Ap; double lam; const float *cy, *cx; int na, dir;
    const float *pitch, *offset; int nk, knot_dz; double rmin;   /* rmin: no data within this radius (native) of the axis */
    double part[CF_MAXT][2];
    struct cf_pairs *pr;   /* same-sheet pairs (--fibers / --pairs; nullptr: none) on this level's grid */
} cf_lvl;
/* Same-sheet pairs (fibres): rows c . U = t with c = tri(b) - tri(a) (the trilinear weights of the two points on the solve grid, 16
   entries) and t = W(b) - W(a) + dir (theta_b - theta_a) / 2 pi (U = W + dir theta / 2 pi), weight w (Huber-reweighted at level 0). The
   CG operator gathers them per cell through a cell -> (pair, entry) index, so the z-row threads never write a shared cell. */
typedef struct cf_pairs {
    size_t np; uint32_t (*ix)[16]; float (*cw)[16]; float *t, *w0, *w, *q;   /* per pair: cells, coefficients, target, weights, c . v */
    uint8_t *fix;   /* GT anchors (--anchor-pins / --anchor-pairs): kept at w0 by the Huber reweighting */
    size_t *rs; uint32_t *re;   /* cell i's entries re[rs[i] .. rs[i + 1]): pair * 16 + entry */
} cf_pairs;
static int cf_cmpf(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }
static double cf_center(const cf_lvl *L, int i) { return (i + 0.5) * L->P * L->h; }   /* native coordinate of a solve cell centre */
/* the theta term dir grad theta / 2 pi (turns per solve cell) at a solve-cell point (native coordinates) */
static void cf_theta(const cf_lvl *L, double pz, double py, double px, double g[3]) {
    const double cy = cf_axis_at(L->cy, L->na, pz), cx = cf_axis_at(L->cx, L->na, pz), ry = py - cy, rx = px - cx, r2 = ry * ry + rx * rx;
    const double s = L->P * L->h * L->dir / (2 * M_PI) / (r2 > 1 ? r2 : 1);
    g[0] = 0; g[1] = s * rx; g[2] = -s * ry;
}
static double cf_prior(const cf_lvl *L, double pz, double py, double px) {   /* r / pitch + offset */
    const double cy = cf_axis_at(L->cy, L->na, pz), cx = cf_axis_at(L->cx, L->na, pz);
    return hypot(py - cy, px - cx) / cf_knot(L->pitch, L->nk, L->knot_dz, pz) + cf_knot(L->offset, L->nk, L->knot_dz, pz);
}
typedef struct { cf_lvl *L; const float *S, *G[3]; double eps; } cf_build;
static void cf_build_run(void *a, long z0, long z1, int t_) {   /* edge weights / targets out of cell (z, y, x) */
    cf_build *B = a; cf_lvl *L = B->L; const double cn = (double)L->P * L->P * L->P;
    for (long z = z0; z < z1; z++) for (int y = 0; y < L->hh; y++) for (int x = 0; x < L->w; x++) {
        const size_t i = ((size_t)z * L->hh + y) * L->w + x;
        const int c0[3] = {(int)z, y, x}, dm[3] = {L->d, L->hh, L->w}; const size_t st[3] = {(size_t)L->hh * L->w, (size_t)L->w, 1};
        for (int ax = 0; ax < 3; ax++) {
            if (c0[ax] + 1 >= dm[ax]) { L->wt[ax][i] = 0; L->tg[ax][i] = 0; continue; }
            const size_t j = i + st[ax];
            double pm[3] = {cf_center(L, c0[0]), cf_center(L, c0[1]), cf_center(L, c0[2])}; pm[ax] += 0.5 * L->P * L->h;
            const double cy = cf_axis_at(L->cy, L->na, pm[0]), cx = cf_axis_at(L->cx, L->na, pm[0]);
            const int near = hypot(pm[1] - cy, pm[2] - cx) < L->rmin;
            const double si = B->S[i], sj = B->S[j], ss = si + sj;
            double g[3]; cf_theta(L, pm[0], pm[1], pm[2], g);
            double t = g[ax]; if (ss > 0 && !near) t += L->P * (B->G[ax][i] + B->G[ax][j]) / ss;   /* turns per solve cell */
            const double wi = si / cn, wj = sj / cn;
            L->wt[ax][i] = (float)((near ? 0 : fmin(wi, wj)) + B->eps); L->tg[ax][i] = (float)t;
        }
    }
}
typedef struct { cf_lvl *L; } cf_rhs;
static void cf_rhs_run(void *a, long z0, long z1, int t_) {   /* b = sum over edges of +-w t (+ lam U0), diag */
    cf_lvl *L = ((cf_rhs *)a)->L; const size_t st[3] = {(size_t)L->hh * L->w, (size_t)L->w, 1};
    for (long z = z0; z < z1; z++) for (int y = 0; y < L->hh; y++) for (int x = 0; x < L->w; x++) {
        const size_t i = ((size_t)z * L->hh + y) * L->w + x; const int c0[3] = {(int)z, y, x};
        double b = L->lam * cf_prior(L, cf_center(L, (int)z), cf_center(L, y), cf_center(L, x)), dg = L->lam;
        for (int ax = 0; ax < 3; ax++) {
            b -= L->wt[ax][i] * L->tg[ax][i]; dg += L->wt[ax][i];
            if (c0[ax] > 0) { b += L->wt[ax][i - st[ax]] * L->tg[ax][i - st[ax]]; dg += L->wt[ax][i - st[ax]]; }
        }
        if (L->pr) { const cf_pairs *R = L->pr;
            for (size_t e = R->rs[i]; e < R->rs[i + 1]; e++) { const size_t pp = R->re[e] >> 4; const double c = R->cw[pp][R->re[e] & 15]; b += R->w[pp] * c * R->t[pp]; dg += R->w[pp] * c * c; } }
        L->b[i] = (float)b; L->diag[i] = (float)dg;
    }
}
static void cf_apply(cf_lvl *L, const float *u, float *o, long z0, long z1) {
    const size_t st[3] = {(size_t)L->hh * L->w, (size_t)L->w, 1};
    for (long z = z0; z < z1; z++) for (int y = 0; y < L->hh; y++) for (int x = 0; x < L->w; x++) {
        const size_t i = ((size_t)z * L->hh + y) * L->w + x; const int c0[3] = {(int)z, y, x}; const double ui = u[i];
        double s = L->lam * ui;
        for (int ax = 0; ax < 3; ax++) {
            if (c0[ax] + 1 < (ax == 0 ? L->d : ax == 1 ? L->hh : L->w)) s += L->wt[ax][i] * (ui - u[i + st[ax]]);
            if (c0[ax] > 0) s += L->wt[ax][i - st[ax]] * (ui - u[i - st[ax]]);
        }
        if (L->pr) { const cf_pairs *R = L->pr;   /* (R->q = c . u of every pair, computed before this pass: cf_pairs_q) */
            for (size_t e = R->rs[i]; e < R->rs[i + 1]; e++) { const size_t pp = R->re[e] >> 4; s += R->w[pp] * R->cw[pp][R->re[e] & 15] * R->q[pp]; } }
        o[i] = (float)s;
    }
}
/* CG phases (each a parallel pass with per-thread partial sums) */
typedef struct { cf_lvl *L; int ph; double alpha, beta; } cf_cg;
static void cf_cg_run(void *a, long z0, long z1, int t) {
    cf_cg *C = a; cf_lvl *L = C->L; const size_t HW = (size_t)L->hh * L->w, i0 = z0 * HW, i1 = z1 * HW; double s0 = 0, s1 = 0;
    if (C->ph == 0) {   /* r = b - A U; p = r / diag; s0 = r . z */
        cf_apply(L, L->U, L->Ap, z0, z1);
        for (size_t i = i0; i < i1; i++) { L->r[i] = L->b[i] - L->Ap[i]; L->p[i] = L->r[i] / L->diag[i]; s0 += (double)L->r[i] * L->p[i]; s1 += (double)L->b[i] * L->b[i]; }
    } else if (C->ph == 1) {   /* Ap = A p; s0 = p . Ap */
        cf_apply(L, L->p, L->Ap, z0, z1);
        for (size_t i = i0; i < i1; i++) s0 += (double)L->p[i] * L->Ap[i];
    } else if (C->ph == 2) {   /* U += alpha p; r -= alpha Ap; s0 = r . z, s1 = r . r */
        for (size_t i = i0; i < i1; i++) { L->U[i] += (float)(C->alpha * L->p[i]); L->r[i] -= (float)(C->alpha * L->Ap[i]); s0 += (double)L->r[i] * L->r[i] / L->diag[i]; s1 += (double)L->r[i] * L->r[i]; }
    } else {   /* p = z + beta p */
        for (size_t i = i0; i < i1; i++) L->p[i] = (float)(L->r[i] / L->diag[i] + C->beta * L->p[i]);
    }
    L->part[t][0] = s0; L->part[t][1] = s1;
}
/* --cf-robust K (globwind 2026-11-27): the count edges' least squares made robust by K IRLS passes per level. Each edge's weight
   becomes w0 x psi(r) / r for the residual r = U_j - U_i - t (turns per solve cell), Huber: psi = r up to delta, then delta. With a
   small delta (default 0.02 turn, against ~0.1-0.5 turn per edge) this is the L1 integration of the count field (convex; the L1 phase
   unwrapping of Ghiglia & Romero). L2 spreads a merge's missing turns over everything around it; L1 leaves them on the few edges
   through the merge, so the count is carried around it (along z, around the turn) by the consistent majority of edges.
   (A redescending Cauchy weight was tried on the test lens: unstable, it broke the defect-free cases at delta 0.01; not kept.) */
typedef struct { cf_lvl *L; float *w0[3]; double delta; double bad[CF_MAXT], wsum[CF_MAXT]; } cf_rob;
static void cf_rob_run(void *a, long z0, long z1, int t) {
    cf_rob *R = a; cf_lvl *L = R->L; const size_t st[3] = {(size_t)L->hh * L->w, (size_t)L->w, 1}; const int dm[3] = {L->d, L->hh, L->w};
    double bad = 0, ws = 0;
    for (long z = z0; z < z1; z++) for (int y = 0; y < L->hh; y++) for (int x = 0; x < L->w; x++) {
        const size_t i = ((size_t)z * L->hh + y) * L->w + x; const int c0[3] = {(int)z, y, x};
        for (int ax = 0; ax < 3; ax++) {
            if (c0[ax] + 1 >= dm[ax]) continue;
            const double w0 = R->w0[ax][i], r = fabs((double)L->U[i + st[ax]] - L->U[i] - L->tg[ax][i]);
            ws += w0; if (r > 0.5) bad += w0;
            const double f = r > R->delta ? R->delta / r : 1;
            L->wt[ax][i] = (float)(w0 * f);
        }
    }
    R->bad[t] = bad; R->wsum[t] = ws;
}
/* --cf-cut 1 (globwind 2026-11-27, research/graph.md idea 1): a layered min-cut over the solve grid with ordering / spacing arcs.
   Unknowns: an integer turn shift s per solve cell on top of the solved U (U' = U + s). Energy (convex, piecewise linear in the
   differences s_b - s_a, so the PUMA graph-cut moves of src/l1pot reach its exact minimum, as an Ishikawa layered cut would):
   - smoothness: every grid edge votes s_b - s_a = 0 with its count weight (keep the solved shape);
   - spacing arcs: from cell a, the cell b at D solve cells along its outward sheet normal must lie at least D / (f sp_lo) turns
     outward, with sp_lo the median over +-2 blocks (B cells each) of the blocks' q-quantile spacing. A merged patch's
     spacing reads ~1.4-1.75x too wide (pseudolab 11-18), so a count below the neighbourhood's spacing prior is lifted there, by an
     integer jump placed where the cut is cheapest. One-sided (a lower bound: merges hide sheets, the nets add almost none). */
typedef struct { const cf_lvl *L; const float *S, *G[3]; const float *splo; int B, nbz, nby, nbx, D, sub; double f, wsp, K, eps; l1_vote *v[CF_MAXT]; size_t nv[CF_MAXT], cv[CF_MAXT], nlift[CF_MAXT]; } cf_cut;
static void cf_cut_push(cf_cut *C, int t, uint32_t a, uint32_t b, int32_t d, int64_t w, int64_t wn) {
    if (C->nv[t] == C->cv[t]) { C->cv[t] = C->cv[t] ? 2 * C->cv[t] : 65536; C->v[t] = realloc(C->v[t], C->cv[t] * sizeof(l1_vote)); if (!C->v[t]) abort(); }
    C->v[t][C->nv[t]++] = (l1_vote){a, b, d, w, wn};
}
static double cf_cut_sp(const cf_cut *C, size_t i) {   /* sheet spacing at a solve cell (solve cells per turn; 0: no data) */
    const double s = C->S[i]; if (!(s > 0)) return 0;
    const double gz = C->G[0][i] / s, gy = C->G[1][i] / s, gx = C->G[2][i] / s, rho = sqrt(gz * gz + gy * gy + gx * gx) * C->L->P;
    return rho > 1e-6 ? 1 / rho : 0;
}
static void cf_cut_run(void *a, long z0, long z1, int t) {
    cf_cut *C = a; const cf_lvl *L = C->L; const size_t st[3] = {(size_t)L->hh * L->w, (size_t)L->w, 1}; const int dm[3] = {L->d, L->hh, L->w};
    for (long z = z0; z < z1; z++) for (int y = 0; y < L->hh; y++) for (int x = 0; x < L->w; x++) {
        const size_t i = ((size_t)z * L->hh + y) * L->w + x; const int c0[3] = {(int)z, y, x};
        for (int ax = 0; ax < 3; ax++) if (c0[ax] + 1 < dm[ax]) {
            const int64_t w = (int64_t)llround(C->K * L->wt[ax][i]); if (w >= 1) cf_cut_push(C, t, (uint32_t)i, (uint32_t)(i + st[ax]), 0, w, w);
        }
        if ((z % C->sub) || (y % C->sub) || (x % C->sub) || !(C->S[i] > 0)) continue;
        const float sl = C->splo[((size_t)(z / C->B) * C->nby + y / C->B) * C->nbx + x / C->B]; if (!(sl > 0)) continue;
        const double s = C->S[i], gz = C->G[0][i] / s, gy = C->G[1][i] / s, gx = C->G[2][i] / s, gl = sqrt(gz * gz + gy * gy + gx * gx); if (gl < 1e-9) continue;
        const int bz = (int)lround(z + C->D * gz / gl), by = (int)lround(y + C->D * gy / gl), bx = (int)lround(x + C->D * gx / gl);
        if (bz < 0 || by < 0 || bx < 0 || bz >= L->d || by >= L->hh || bx >= L->w) continue;
        const size_t j = ((size_t)bz * L->hh + by) * L->w + bx; if (!(C->S[j] > 0)) continue;
        const double dist = sqrt((double)(bz - z) * (bz - z) + (double)(by - y) * (by - y) + (double)(bx - x) * (bx - x));
        const double need = dist / (C->f * sl) - (L->U[j] - L->U[i]); if (need < 0.5) continue;
        const int64_t wn = (int64_t)llround(C->wsp * C->K * fmin(1, s / ((double)L->P * L->P * L->P))); if (wn < 1) continue;
        cf_cut_push(C, t, (uint32_t)i, (uint32_t)j, (int32_t)lround(need), 1, wn); C->nlift[t]++;
    }
}
static int cf_cut_solve(cf_lvl *L, const float *S, float *const G[3], int D, int B, double q, double f, double wsp, int sub) {
    const double t0 = cf_now(); cf_cut C = {L, S, {G[0], G[1], G[2]}, nullptr, B, (L->d + B - 1) / B, (L->hh + B - 1) / B, (L->w + B - 1) / B, D, sub, f, wsp, 1000, 0, {0}, {0}, {0}, {0}};
    const size_t nb = (size_t)C.nbz * C.nby * C.nbx; float *med = calloc(nb, sizeof(float)), *splo = calloc(nb, sizeof(float)); if (!med || !splo) return -1;
    /* per block: the q-quantile of its cells' spacing, then the min over the +-1 block neighbourhood of those (the local spacing prior) */
    float *buf = malloc((size_t)B * B * B * sizeof(float)); double ssum = 0; size_t sn = 0;
    for (int Z = 0; Z < C.nbz; Z++) for (int Y = 0; Y < C.nby; Y++) for (int X = 0; X < C.nbx; X++) {
        int n = 0;
        for (int z = Z * B; z < (Z + 1) * B && z < L->d; z++) for (int y = Y * B; y < (Y + 1) * B && y < L->hh; y++) for (int x = X * B; x < (X + 1) * B && x < L->w; x++) {
            const double sp = cf_cut_sp(&C, ((size_t)z * L->hh + y) * L->w + x); if (sp > 0) buf[n++] = (float)sp; }
        if (n < B * B) continue;   /* too little data for a prior */
        qsort(buf, n, sizeof(float), cf_cmpf); med[((size_t)Z * C.nby + Y) * C.nbx + X] = buf[(int)(q * (n - 1))]; ssum += buf[n / 2]; sn++;
    }
    free(buf);
    const int Rb = 2; float nbv[(2 * 2 + 1) * (2 * 2 + 1) * (2 * 2 + 1)];
    for (int Z = 0; Z < C.nbz; Z++) for (int Y = 0; Y < C.nby; Y++) for (int X = 0; X < C.nbx; X++) {
        int n = 0;
        for (int a = Z - Rb; a <= Z + Rb; a++) for (int b = Y - Rb; b <= Y + Rb; b++) for (int c = X - Rb; c <= X + Rb; c++) {
            if (a < 0 || b < 0 || c < 0 || a >= C.nbz || b >= C.nby || c >= C.nbx) continue;
            const float v = med[((size_t)a * C.nby + b) * C.nbx + c]; if (v > 0) nbv[n++] = v; }
        if (n) qsort(nbv, n, sizeof(float), cf_cmpf);
        splo[((size_t)Z * C.nby + Y) * C.nbx + X] = n ? nbv[n / 2] : 0;   /* the median over +-2 blocks: a defect patch smaller than half of it cannot set its own prior */
    }
    free(med); C.splo = splo;
    if (D <= 0) D = C.D = (int)lround(3 * (sn ? ssum / sn : 8));   /* default: ~3 median turns along the normal */
    cf_par(L->d, cf_cut_run, &C);
    size_t nv = 0, nl = 0; for (int t = 0; t < CF_MAXT; t++) { nv += C.nv[t]; nl += C.nlift[t]; }
    l1_vote *V = malloc((nv + 1) * sizeof *V); if (!V) return -1; size_t o = 0;
    for (int t = 0; t < CF_MAXT; t++) { if (C.nv[t]) memcpy(V + o, C.v[t], C.nv[t] * sizeof *V); o += C.nv[t]; free(C.v[t]); }
    nv = l1_votes_merge(V, nv);
    int32_t *s = calloc(L->n, sizeof *s); l1_stats stt = {0}; if (!s) return -1;
    const int rc = l1_solve((uint32_t)L->n, V, nv, s, 64, &stt);
    size_t nmov = 0; long smin = 0, smax = 0;
    if (!rc) for (size_t i = 0; i < L->n; i++) { if (s[i]) { nmov++; L->U[i] += (float)s[i]; } if (s[i] < smin) smin = s[i]; if (s[i] > smax) smax = s[i]; }
    fprintf(stderr, "count-field: --cf-cut: D %d solve cells, spacing prior q %.2f x f %.2f (blocks %d), %zu lift arcs, %zu votes; L1 energy %lld -> %lld in %d moves%s; %zu of %zu cells shifted (%ld..%+ld turns), %.1f s\n",
            C.D, q, f, B, nl, nv, (long long)stt.e0, (long long)stt.e1, stt.moves, stt.capped ? " (work budget)" : "", nmov, L->n, smin, smax, cf_now() - t0);
    free(s); free(V); free(splo); return rc;
}
typedef struct { cf_pairs *R; const float *u; } cf_pq;
static void cf_pairs_q_run(void *a, long p0, long p1, int t_) {
    cf_pq *Q = a; cf_pairs *R = Q->R;
    for (long p = p0; p < p1; p++) { double s = 0; for (int k = 0; k < 16; k++) s += R->cw[p][k] * Q->u[R->ix[p][k]]; R->q[p] = (float)s; }
}
static void cf_pairs_q(cf_pairs *R, const float *u) { cf_pq Q = {R, u}; cf_par((long)R->np, cf_pairs_q_run, &Q); }
static void cf_cg_pass(cf_cg *C, double *s0, double *s1) {
    if (C->L->pr && C->ph <= 1) cf_pairs_q(C->L->pr, C->ph == 0 ? C->L->U : C->L->p);
    for (int t = 0; t < CF_MAXT; t++) C->L->part[t][0] = C->L->part[t][1] = 0;
    cf_par(C->L->d, cf_cg_run, C); double a = 0, b = 0; for (int t = 0; t < CF_MAXT; t++) { a += C->L->part[t][0]; b += C->L->part[t][1]; }
    if (s0) *s0 = a;
    if (s1) *s1 = b;
}
static int cf_cg_solve(cf_lvl *L, int iters, double tol, double *relres) {
    cf_cg C = {L, 0, 0, 0}; double rz, bb; cf_cg_pass(&C, &rz, &bb);
    const double bn = sqrt(bb > 0 ? bb : 1); int it = 0; double rr = 0;
    for (; it < iters; it++) {
        double pAp; C.ph = 1; cf_cg_pass(&C, &pAp, nullptr); if (pAp <= 0) break;
        C.ph = 2; C.alpha = rz / pAp; double rz1; cf_cg_pass(&C, &rz1, &rr);
        if (sqrt(rr) / bn < tol) { it++; break; }
        C.ph = 3; C.beta = rz1 / rz; rz = rz1; cf_cg_pass(&C, nullptr, nullptr);
    }
    *relres = sqrt(rr) / bn; return it;
}
static double cf_tri(const float *U, int d, int hh, int w, double qz, double qy, double qx) {   /* trilinear in cell-index coordinates (clamped) */
    const double q[3] = {qz, qy, qx}; const int dm[3] = {d, hh, w}; int c[3]; double t[3];
    for (int k = 0; k < 3; k++) { double a = q[k] < 0 ? 0 : q[k] > dm[k] - 1 ? dm[k] - 1 : q[k]; c[k] = (int)a; if (c[k] > dm[k] - 2) c[k] = dm[k] - 2 < 0 ? 0 : dm[k] - 2; t[k] = dm[k] > 1 ? a - c[k] : 0; }
    double s = 0;
    for (int e = 0; e < 8; e++) {
        const int a = c[0] + (e >> 2 & 1), b = c[1] + (e >> 1 & 1), x = c[2] + (e & 1);
        const double wv = (e >> 2 & 1 ? t[0] : 1 - t[0]) * (e >> 1 & 1 ? t[1] : 1 - t[1]) * (e & 1 ? t[2] : 1 - t[2]);
        if (wv != 0) s += wv * U[((size_t)(a < d ? a : d - 1) * hh + (b < hh ? b : hh - 1)) * w + (x < w ? x : w - 1)];
    }
    return s;
}

/* ---- same-sheet pairs: the point list (box-local native z y x, target t in U), built per level ---- */
typedef struct { float a[3], b[3], t, w; int pin; } cf_pt;   /* w > 0: its own weight (an anchor, fixed); pin: a row U(a) = t (b unused) */
static int cf_data_at(const cf_lvl *L, const float *S, double smin, const float p[3]) {   /* the solve cell of a native point holds data */
    const double cs = (double)L->P * L->h; const int z = (int)floor(p[0] / cs), y = (int)floor(p[1] / cs), x = (int)floor(p[2] / cs);
    return z >= 0 && y >= 0 && x >= 0 && z < L->d && y < L->hh && x < L->w && S[((size_t)z * L->hh + y) * L->w + x] >= smin;
}
static void cf_tri_w(const cf_lvl *L, const float p[3], uint32_t ix[8], float cw[8], float sg) {   /* cf_tri's corners / weights of a native point */
    const double cs = (double)L->P * L->h, q[3] = {p[0] / cs - 0.5, p[1] / cs - 0.5, p[2] / cs - 0.5}; const int dm[3] = {L->d, L->hh, L->w}; int c[3]; double t[3];
    for (int k = 0; k < 3; k++) { double a = q[k] < 0 ? 0 : q[k] > dm[k] - 1 ? dm[k] - 1 : q[k]; c[k] = (int)a; if (c[k] > dm[k] - 2) c[k] = dm[k] - 2 < 0 ? 0 : dm[k] - 2; t[k] = dm[k] > 1 ? a - c[k] : 0; }
    for (int e = 0; e < 8; e++) {
        const int a = c[0] + (e >> 2 & 1), b = c[1] + (e >> 1 & 1), x = c[2] + (e & 1);
        ix[e] = (uint32_t)(((size_t)(a < L->d ? a : L->d - 1) * L->hh + (b < L->hh ? b : L->hh - 1)) * L->w + (x < L->w ? x : L->w - 1));
        cw[e] = sg * (float)((e >> 2 & 1 ? t[0] : 1 - t[0]) * (e >> 1 & 1 ? t[1] : 1 - t[1]) * (e & 1 ? t[2] : 1 - t[2]));
    }
}
static cf_pairs *cf_pairs_build(const cf_lvl *L, const cf_pt *P, size_t np, double w0) {
    cf_pairs *R = calloc(1, sizeof *R); if (!R) return nullptr;
    R->np = np; R->ix = malloc(np * sizeof *R->ix); R->cw = malloc(np * sizeof *R->cw); R->t = malloc(np * 4); R->w0 = malloc(np * 4); R->w = malloc(np * 4); R->q = malloc(np * 4);
    R->rs = calloc(L->n + 1, sizeof(size_t)); R->re = malloc(np * 16 * sizeof(uint32_t)); R->fix = malloc(np ? np : 1);
    if (!R->ix || !R->cw || !R->t || !R->w0 || !R->w || !R->q || !R->rs || !R->re || !R->fix) { fprintf(stderr, "count-field: out of memory (pairs)\n"); exit(1); }
    for (size_t p = 0; p < np; p++) {
        if (P[p].pin) { cf_tri_w(L, P[p].a, R->ix[p], R->cw[p], 1.f); for (int k = 0; k < 8; k++) { R->ix[p][8 + k] = R->ix[p][k]; R->cw[p][8 + k] = 0; } }
        else { cf_tri_w(L, P[p].a, R->ix[p], R->cw[p], -1.f); cf_tri_w(L, P[p].b, R->ix[p] + 8, R->cw[p] + 8, 1.f); }
        R->t[p] = P[p].t; R->w0[p] = R->w[p] = (float)(P[p].w > 0 ? P[p].w : w0); R->fix[p] = P[p].w > 0;
        for (int k = 0; k < 16; k++) R->rs[R->ix[p][k] + 1]++;
    }
    for (size_t i = 0; i < L->n; i++) R->rs[i + 1] += R->rs[i];
    size_t *fill = malloc(L->n * sizeof(size_t)); memcpy(fill, R->rs, L->n * sizeof(size_t));
    for (size_t p = 0; p < np; p++) for (int k = 0; k < 16; k++) R->re[fill[R->ix[p][k]]++] = (uint32_t)(p * 16 + k);
    free(fill); return R;
}
static void cf_pairs_free(cf_pairs *R) { if (!R) return; free(R->fix); free(R->ix); free(R->cw); free(R->t); free(R->w0); free(R->w); free(R->q); free(R->rs); free(R->re); free(R); }
/* the pairs' residuals c . U - t at U: Huber weights w0 min(1, delta / |r|); returns the share with |r| > 0.5 (turns) */
static double cf_pairs_reweight(cf_pairs *R, const float *U, double delta, int set) {
    cf_pairs_q(R, U); size_t nb = 0;
    for (size_t p = 0; p < R->np; p++) { const double r = fabs(R->q[p] - R->t[p]); nb += r > 0.5; if (set && !R->fix[p]) R->w[p] = (float)(R->w0[p] * (r > delta ? delta / r : 1)); }
    return R->np ? (double)nb / R->np : 0;
}

/* ---- the ray variant: U(cell) = prior at the ray's exit (box edge / near the axis) + sum of target . r_hat along the ray ---- */
typedef struct { cf_lvl *L; const float *S, *G[3]; } cf_ray;
static void cf_ray_run(void *a, long z0, long z1, int t_) {
    cf_ray *R = a; cf_lvl *L = R->L; const double cs = L->P * L->h;   /* native voxels per solve cell */
    for (long z = z0; z < z1; z++) {
        const double pz = cf_center(L, (int)z), cy = cf_axis_at(L->cy, L->na, pz), cx = cf_axis_at(L->cx, L->na, pz);
        for (int y = 0; y < L->hh; y++) for (int x = 0; x < L->w; x++) {
            const double py = cf_center(L, y), px = cf_center(L, x), rl = hypot(py - cy, px - cx);
            const size_t i = ((size_t)z * L->hh + y) * L->w + x;
            if (rl < 1) { L->U[i] = (float)cf_prior(L, pz, py, px); continue; }
            const double uy = (py - cy) / rl, ux = (px - cx) / rl; double s = 0, qy = py, qx = px; const double ds = 0.5;   /* solve cells */
            for (;;) {
                const int iy = (int)floor(qy / cs), ix = (int)floor(qx / cs);   /* (floor: (int) truncation took (-cs, 0) as cell 0, audit mini/coords #2) */
                if (iy < 0 || ix < 0 || iy >= L->hh || ix >= L->w || hypot(qy - cy, qx - cx) < fmax(L->rmin, 1)) break;
                const size_t j = ((size_t)z * L->hh + iy) * L->w + ix;
                if (R->S[j] > 0) s += ds * L->P * (R->G[1][j] * uy + R->G[2][j] * ux) / R->S[j];
                qy -= uy * ds * cs; qx -= ux * ds * cs;
            }
            L->U[i] = (float)(cf_prior(L, pz, qy, qx) + s);
        }
    }
}

/* ---- closing error: |circulation| of the targets around L x L squares (3 planes), max over planes ---- */
typedef struct { cf_lvl *L; int Q; float *cl; } cf_close;
static void cf_close_run(void *a, long z0, long z1, int t_) {
    cf_close *C = a; cf_lvl *L = C->L; const int dm[3] = {L->d, L->hh, L->w}, Q = C->Q; const size_t st[3] = {(size_t)L->hh * L->w, (size_t)L->w, 1};
    for (long z = z0; z < z1; z++) for (int y = 0; y < L->hh; y++) for (int x = 0; x < L->w; x++) {
        const int c[3] = {(int)z, y, x}; double best = 0;
        for (int pl = 0; pl < 3; pl++) {
            const int a1 = pl == 0 ? 1 : 0, a2 = pl == 2 ? 1 : 2;   /* the plane's two axes */
            int o[3] = {c[0], c[1], c[2]}; o[a1] -= Q / 2; o[a2] -= Q / 2;
            if (o[a1] < 0 || o[a2] < 0 || o[a1] + Q >= dm[a1] || o[a2] + Q >= dm[a2]) continue;
            double circ = 0;
            for (int k = 0; k < Q; k++) {
                int p0[3] = {o[0], o[1], o[2]}; p0[a1] += k;   /* along a1 at a2 = o: + ; at a2 = o + Q: - */
                size_t i = p0[0] * st[0] + p0[1] * st[1] + p0[2] * st[2]; circ += L->tg[a1][i] - L->tg[a1][i + Q * st[a2]];
                int p1[3] = {o[0], o[1], o[2]}; p1[a2] += k;   /* along a2 at a1 = o + Q: + ; at a1 = o: - */
                i = p1[0] * st[0] + p1[1] * st[1] + p1[2] * st[2]; circ += L->tg[a2][i + Q * st[a1]] - L->tg[a2][i];
            }
            if (fabs(circ) > best) best = fabs(circ);
        }
        C->cl[((size_t)z * L->hh + y) * L->w + x] = (float)best;
    }
}

/* ---- the output: W at the field cells, demodulated against the phase over +-R cells in y / x and +-Rz rows in z (a ring of planes
   inside each chunk of rows; the chunk is loaded with Rz halo rows) ---- */
typedef struct {
    const fit_fields *f; int zoff;   /* the loaded chunk (its row 0 = box row zoff) */
    const cf_lvl *L; int D, H, W, R, na, dir; double h, gauge; const float *cy, *cx;   /* the box's axis table */
    float *re, *im, *cw;   /* this plane: c e^{2 pi i (phi - W)} and c, blurred in y / x */
    float *Wp;             /* W (gauge included) of the plane */
    int z;                 /* box row of the plane */
    const float *gls;      /* --gauge ls: the smooth (vortex-free) gauge field on the solve grid, added to W (nullptr: none) */
} cf_dm;
static double cf_Wcell(const cf_dm *M, int z, int y, int x) {   /* W = U - dir theta / 2 pi at a field cell (box row z) + the gauge */
    const cf_lvl *L = M->L; const double P = L->P, pz = (z + 0.5) * M->h, py = (y + 0.5) * M->h, px = (x + 0.5) * M->h;
    const double U = cf_tri(L->U, L->d, L->hh, L->w, (z + 0.5) / P - 0.5, (y + 0.5) / P - 0.5, (x + 0.5) / P - 0.5) + (M->gls ? cf_tri(M->gls, L->d, L->hh, L->w, (z + 0.5) / P - 0.5, (y + 0.5) / P - 0.5, (x + 0.5) / P - 0.5) : 0);
    const double cy = cf_axis_at(M->cy, M->na, pz), cx = cf_axis_at(M->cx, M->na, pz);
    return U - M->dir * atan2(py - cy, px - cx) / (2 * M_PI) + M->gauge;
}
static void cf_box1(float *v, int n, size_t st, int R, float *tmp) {   /* in-place box sum of radius R along a line (clamped window) */
    double s = 0; for (int k = 0; k <= R && k < n; k++) s += v[k * st];
    for (int k = 0; k < n; k++) { tmp[k] = (float)s; const int ka = k + R + 1, kr = k - R; if (ka < n) s += v[ka * st]; if (kr >= 0) s -= v[kr * st]; }
    for (int k = 0; k < n; k++) v[k * st] = tmp[k];
}
static void cf_plane_run(void *a, long y0, long y1, int t_) {   /* rows [y0, y1) of plane M->z: phasors and W */
    cf_dm *M = a; const fit_fields *f = M->f; const size_t HW = (size_t)M->H * M->W;
    for (long y = y0; y < y1; y++) for (int x = 0; x < M->W; x++) {
        const size_t i = (size_t)(M->z - M->zoff) * HW + (size_t)y * M->W + x, j = (size_t)y * M->W + x;
        const double Wv = cf_Wcell(M, M->z, (int)y, x); M->Wp[j] = (float)Wv;
        const double pc = (f->pc[i] - 128) / 127.0, ps = (f->ps[i] - 128) / 127.0, c = f->w[i] ? hypot(pc, ps) : 0;
        if (c < 1e-3) { M->re[j] = M->im[j] = M->cw[j] = 0; continue; }
        const double e = atan2(ps, pc) - 2 * M_PI * Wv;
        M->re[j] = (float)(c * cos(e)); M->im[j] = (float)(c * sin(e)); M->cw[j] = (float)c;
    }
}
typedef struct { cf_dm *M; } cf_blr;
static void cf_blur_rows(void *a, long r0, long r1, int t_) {   /* y-blur columns [r0, r1) */
    cf_dm *M = ((cf_blr *)a)->M; float *tmp = malloc((size_t)(M->H > M->W ? M->H : M->W) * sizeof(float));
    for (long x = r0; x < r1; x++) { cf_box1(M->re + x, M->H, M->W, M->R, tmp); cf_box1(M->im + x, M->H, M->W, M->R, tmp); cf_box1(M->cw + x, M->H, M->W, M->R, tmp); }
    free(tmp);
}
static void cf_blur_cols(void *a, long r0, long r1, int t_) {   /* x-blur rows [r0, r1) */
    cf_dm *M = ((cf_blr *)a)->M; float *tmp = malloc((size_t)M->W * sizeof(float));
    for (long y = r0; y < r1; y++) { const size_t o = (size_t)y * M->W; cf_box1(M->re + o, M->W, 1, M->R, tmp); cf_box1(M->im + o, M->W, 1, M->R, tmp); cf_box1(M->cw + o, M->W, 1, M->R, tmp); }
    free(tmp);
}

/* ---- --gauge ls: the phase's sub-turn offset to W as a smooth field. Per solve cell the phasor sum z = sum c e^{2 pi i (phi - W)},
   box-smoothed over +-Rg cells; the gauge g minimises sum_edges min(|z_i|, |z_j|) (g_j - g_i - wrap(arg z_j - arg z_i))^2 (least-squares
   phase unwrapping, coarse to fine CG): unlike arg z itself, g has no vortices, so W + g adds no phase dislocations to the output ---- */
typedef struct { const cf_dm *M; float *gr, *gi; } cf_gsum;
static void cf_gsum_run(void *a, long Y0, long Y1, int t_) {   /* solve rows of one plane M->z (bands of P field rows) */
    const cf_gsum *G = a; const cf_dm *M = G->M; const fit_fields *f = M->f; const cf_lvl *L = M->L; const int P = L->P; const size_t HW = (size_t)M->H * M->W;
    for (long y = Y0 * P; y < Y1 * P && y < M->H; y++) for (int x = 0; x < M->W; x++) {
        const size_t i = (size_t)(M->z - M->zoff) * HW + (size_t)y * M->W + x; if (!f->w[i]) continue;
        const double pc = (f->pc[i] - 128) / 127.0, ps = (f->ps[i] - 128) / 127.0, c = hypot(pc, ps); if (c < 1e-3) continue;
        const double e = atan2(ps, pc) - 2 * M_PI * cf_Wcell(M, M->z, (int)y, x); const size_t si = ((size_t)(M->z / P) * L->hh + y / P) * L->w + x / P;
        G->gr[si] += (float)(c * cos(e)); G->gi[si] += (float)(c * sin(e));
    }
}
static void cf_boxsmooth(float *v, int d, int hh, int w, int R) {
    if (R < 1) return;
    const size_t st[3] = {(size_t)hh * w, (size_t)w, 1}; const int dm[3] = {d, hh, w}; const int mx = d > hh ? (d > w ? d : w) : (hh > w ? hh : w);
    float *tmp = malloc((size_t)mx * sizeof(float));
    for (int ax = 0; ax < 3; ax++) { const int a1 = ax == 0 ? 1 : 0, a2 = ax == 2 ? 1 : 2;
        for (int u = 0; u < dm[a1]; u++) for (int v2 = 0; v2 < dm[a2]; v2++) cf_box1(v + u * st[a1] + v2 * st[a2], dm[ax], st[ax], R, tmp); }
    free(tmp);
}
/* threaded line passes (each line's sum as cf_box1: the same numbers) and prolongation */
typedef struct { float *v[2]; int nv, ax, R, dm[3]; size_t st[3]; } cf_bsp;
static void cf_bsp_run(void *a, long u0, long u1, int t_) {
    cf_bsp *B = a; const int a1 = B->ax == 0 ? 1 : 0, a2 = B->ax == 2 ? 1 : 2; float *tmp = malloc((size_t)B->dm[B->ax] * sizeof(float)); if (!tmp) return;
    for (long u = u0; u < u1; u++) for (int v = 0; v < B->dm[a2]; v++) for (int k = 0; k < B->nv; k++) cf_box1(B->v[k] + u * B->st[a1] + v * B->st[a2], B->dm[B->ax], B->st[B->ax], B->R, tmp);
    free(tmp);
}
static void cf_boxsmooth2_par(float *v0, float *v1, int d, int hh, int w, int R) {   /* = cf_box1 along z, y, x of each array, threaded */
    cf_bsp B = {{v0, v1}, v1 ? 2 : 1, 0, R, {d, hh, w}, {(size_t)hh * w, (size_t)w, 1}};
    for (int ax = 0; ax < 3; ax++) { B.ax = ax; cf_par(B.dm[ax == 0 ? 1 : 0], cf_bsp_run, &B); }
}
typedef struct { cf_lvl *L; const cf_lvl *C; } cf_prl;
static void cf_prl_run(void *a, long z0, long z1, int t_) {
    cf_prl *Q = a; cf_lvl *L = Q->L; const cf_lvl *C = Q->C;
    for (long z = z0; z < z1; z++) for (int y = 0; y < L->hh; y++) for (int x = 0; x < L->w; x++)
        L->U[((size_t)z * L->hh + y) * L->w + x] = (float)cf_tri(C->U, C->d, C->hh, C->w, (z + 0.5) / 2 - 0.5, (y + 0.5) / 2 - 0.5, (x + 0.5) / 2 - 0.5);
}
/* gr / gi (level-0 solve grid of L0, smoothed in place) -> the gauge g (turns, malloc'ed, L0's grid) */
static float *cf_gauge_solve(const cf_lvl *L0, float *gr, float *gi, int Rg, int iters) {
    cf_boxsmooth(gr, L0->d, L0->hh, L0->w, Rg); cf_boxsmooth(gi, L0->d, L0->hh, L0->w, Rg);
    int nl = 1; while (nl < 6 && (L0->d >> nl) >= 4 && (L0->hh >> nl) >= 4 && (L0->w >> nl) >= 4) nl++;
    cf_lvl *G = calloc(nl, sizeof *G); float **zr = calloc(nl, sizeof *zr), **zi = calloc(nl, sizeof *zi);
    for (int l = 0; l < nl; l++) {
        cf_lvl *L = &G[l]; L->d = l ? (G[l - 1].d + 1) / 2 : L0->d; L->hh = l ? (G[l - 1].hh + 1) / 2 : L0->hh; L->w = l ? (G[l - 1].w + 1) / 2 : L0->w; L->n = (size_t)L->d * L->hh * L->w;
        if (!l) { zr[0] = gr; zi[0] = gi; continue; }
        zr[l] = calloc(L->n, sizeof(float)); zi[l] = calloc(L->n, sizeof(float)); const cf_lvl *F = &G[l - 1];
        for (int z = 0; z < F->d; z++) for (int y = 0; y < F->hh; y++) for (int x = 0; x < F->w; x++) {
            const size_t i = ((size_t)z * F->hh + y) * F->w + x, j = ((size_t)(z / 2) * L->hh + y / 2) * L->w + x / 2; zr[l][j] += zr[l - 1][i]; zi[l][j] += zi[l - 1][i]; }
    }
    float *g = nullptr;
    for (int l = nl - 1; l >= 0; l--) {
        cf_lvl *L = &G[l]; const size_t st[3] = {(size_t)L->hh * L->w, (size_t)L->w, 1}; const int dm[3] = {L->d, L->hh, L->w};
        double zmax = 0; for (size_t i = 0; i < L->n; i++) zmax = fmax(zmax, hypot(zr[l][i], zi[l][i])); if (zmax <= 0) zmax = 1;
        for (int k = 0; k < 3; k++) { L->wt[k] = malloc(L->n * sizeof(float)); L->tg[k] = malloc(L->n * sizeof(float)); }
        L->b = calloc(L->n, sizeof(float)); L->diag = calloc(L->n, sizeof(float)); L->U = malloc(L->n * sizeof(float));
        L->r = malloc(L->n * sizeof(float)); L->p = malloc(L->n * sizeof(float)); L->Ap = malloc(L->n * sizeof(float)); L->lam = 1e-7;
        for (int z = 0; z < L->d; z++) for (int y = 0; y < L->hh; y++) for (int x = 0; x < L->w; x++) {
            const size_t i = ((size_t)z * L->hh + y) * L->w + x; const int c[3] = {z, y, x};
            for (int ax = 0; ax < 3; ax++) {
                if (c[ax] + 1 >= dm[ax]) { L->wt[ax][i] = L->tg[ax][i] = 0; continue; }
                const size_t j = i + st[ax]; const double ai = hypot(zr[l][i], zi[l][i]), aj = hypot(zr[l][j], zi[l][j]);
                double t = (atan2(zi[l][j], zr[l][j]) - atan2(zi[l][i], zr[l][i])) / (2 * M_PI); t -= round(t);
                L->wt[ax][i] = (float)(fmin(ai, aj) / zmax + 1e-4); L->tg[ax][i] = (float)t;
            }
        }
        for (size_t i = 0; i < L->n; i++) {   /* b, diag (cf_rhs_run without the prior) */
            const int z = (int)(i / st[0]), y = (int)(i / L->w % L->hh), x = (int)(i % L->w); const int c[3] = {z, y, x}; double b = 0, dg = L->lam;
            for (int ax = 0; ax < 3; ax++) { b -= L->wt[ax][i] * L->tg[ax][i]; dg += L->wt[ax][i]; if (c[ax] > 0) { b += L->wt[ax][i - st[ax]] * L->tg[ax][i - st[ax]]; dg += L->wt[ax][i - st[ax]]; } }
            L->b[i] = (float)b; L->diag[i] = (float)dg;
        }
        if (l == nl - 1) for (size_t i = 0; i < L->n; i++) L->U[i] = 0;
        else { const cf_lvl *C = &G[l + 1];
            for (int z = 0; z < L->d; z++) for (int y = 0; y < L->hh; y++) for (int x = 0; x < L->w; x++)
                L->U[((size_t)z * L->hh + y) * L->w + x] = (float)cf_tri(C->U, C->d, C->hh, C->w, (z + 0.5) / 2 - 0.5, (y + 0.5) / 2 - 0.5, (x + 0.5) / 2 - 0.5); }
        double rr; const int it = cf_cg_solve(L, l == nl - 1 ? 4 * iters : iters, 1e-5, &rr);
        if (!l) fprintf(stderr, "count-field: gauge (ls, +-%d solve cells): %d CG iterations at level 0, relative residual %.2e\n", Rg, it, rr);
        free(L->b); free(L->diag); free(L->r); free(L->p); free(L->Ap); for (int k = 0; k < 3; k++) { free(L->wt[k]); free(L->tg[k]); }
        if (l + 1 < nl) free(G[l + 1].U);
        if (l) { free(zr[l]); free(zi[l]); }
    }
    g = G[0].U;
    double complex acc = 0; for (size_t i = 0; i < G[0].n; i++) acc += (gr[i] + I * gi[i]) * cexp(-2 * M_PI * I * g[i]);   /* the constant */
    const double c0 = carg(acc) / (2 * M_PI); for (size_t i = 0; i < G[0].n; i++) g[i] += (float)c0;
    free(G); free(zr); free(zi);
    return g;
}

/* one output plane zo (bands of P rows = one solve row each, in parallel): the ring's blurred phasors -> W_out, confidence, planes;
   the model's phase-gauge sums per solve cell (gr / gi) */
typedef struct {
    const fit_fields *f; int zoff, c0;   /* the chunk (row 0 = box row zoff); the output buffers' row 0 = box row c0 */
    const cf_lvl *L; const float *ring, *Wz, *Sfine, *cl; int zo, D, H, W, Rz, P, cmode, demod, fine, trf; double keep, wsc, t0, t1;
    uint8_t *opc, *ops, *ocl, *osp; const float *gW; float *wc, *gr, *gi; double ag[CF_MAXT][2];   /* osp / gW: --sp-w (sp from |grad W|) */
} cf_out;
static void cf_out_run(void *a, long Y0, long Y1, int t) {
    cf_out *O = a; const fit_fields *f = O->f; const cf_lvl *L = O->L; const size_t HW = (size_t)O->H * O->W; const int Rz = O->Rz, nring = 2 * Rz + 1, P = O->P;
    for (long y = Y0 * P; y < Y1 * P && y < O->H; y++) for (int x = 0; x < O->W; x++) {
        const size_t j = (size_t)y * O->W + x; double re = 0, im = 0, cs = 0;
        if (O->demod) for (int dz = -Rz; dz <= Rz; dz++) { const int zz = O->zo + dz; if (zz < 0 || zz >= O->D) continue; const float *s = O->ring + (size_t)(zz % nring) * 3 * HW; re += s[j]; im += s[HW + j]; cs += s[2 * HW + j]; }
        const size_t i = (size_t)(O->zo - O->zoff) * HW + j, io = (size_t)(O->zo - O->c0) * HW + j, si = ((size_t)(O->zo / P) * L->hh + y / P) * L->w + x / P;
        double Wo = O->Wz[j], A = 0;
        if (O->demod && cs > 0) { A = hypot(re, im) / cs; if (O->demod == 2) { const double th = atan2(im, re); Wo += (O->fine ? A * 2 * (sin(th) - sin(2 * th) / 2 + sin(3 * th) / 3) : im / cs) / (2 * M_PI); } else if (hypot(re, im) > 1e-9) Wo += atan2(im, re) / (2 * M_PI); }   /* 2 (--gauge ls): the bounded continuous A sin(arg) / 2 pi: no new dislocations */
        if (O->keep > 0 && A > 0) {   /* where the phase rotates with W (local agreement A high) it keeps its own sub-turn detail: + A^keep wrap(phi - W_out) */
            const double pcv0 = (f->pc[i] - 128) / 127.0, psv0 = (f->ps[i] - 128) / 127.0;
            if (hypot(pcv0, psv0) > 0.05) { double e = atan2(psv0, pcv0) / (2 * M_PI) - Wo; e -= round(e); Wo += pow(A, O->keep) * e; } }
        if (O->t1 > 0) {   /* --trust: a confident network phase keeps its own position (and count) */
            const double pc1 = (f->pc[i] - 128) / 127.0, ps1 = (f->ps[i] - 128) / 127.0, s = cf_smooth(hypot(pc1, ps1), O->t0, O->t1);
            if (s > 0) { double e = atan2(ps1, pc1) / (2 * M_PI) - Wo; e -= round(e); Wo += O->trf ? s * 2 * (sin(2 * M_PI * e) - sin(4 * M_PI * e) / 2 + sin(6 * M_PI * e) / 3) / (2 * M_PI) : s * e; } }   /* --trust-f 1: a periodic continuous pull (no new dislocations) */
        const double cn = (double)P * P * P, cwt = fmin(1, O->Sfine[si] / cn * 2) * exp(-O->cl[si] * O->cl[si] / (2 * 0.25 * 0.25));
        const double pcv = (f->pc[i] - 128) / 127.0, psv = (f->ps[i] - 128) / 127.0, cph = hypot(pcv, psv);
        if (O->gr && f->w[i] && cph >= 1e-3) { const double e = atan2(psv, pcv) - 2 * M_PI * O->Wz[j]; O->gr[si] += (float)(cph * cos(e)); O->gi[si] += (float)(cph * sin(e)); }
        double c = O->cmode == 0 ? O->wsc * cwt : O->cmode == 1 ? cph : O->cmode == 3 ? fmax(cph, O->wsc * cwt) : fmax(cph * A, O->wsc * cwt);
        if (!f->w[i]) c = 0;
        c = fmin(1, fmax(0, c));
        if (f->w[i] && cs > 0) { O->ag[t][0] += A * cph; O->ag[t][1] += cph; }
        O->opc[io] = (uint8_t)lround(128 + 127 * c * cos(2 * M_PI * Wo)); O->ops[io] = (uint8_t)lround(128 + 127 * c * sin(2 * M_PI * Wo));
        if (O->wc) O->wc[io] = f->w[i] ? (float)Wo : NAN;
        O->ocl[io] = (uint8_t)fmin(255, lround(O->cl[si] * 255));
        if (O->osp) O->osp[io] = f->w[i] && O->gW[si] > 0.03 ? (uint8_t)fmax(1, fmin(255, lround(8 / O->gW[si]))) : f->sp[i];
    }
}

/* zarr output: one block of rows (z shard index sz, nz rows) of the planes pc / ps / cl into 128^3 shards, a job per (plane, sy, sx) */
typedef struct { z3w *const *zw; const uint8_t *pl[4]; int sz, nz, H, W, ny, nx, oy, ox; _Atomic int fail; } cf_shards;   /* oy / ox: the shard grid offset (--into) */
static void cf_shard_run(void *a, long j0, long j1, int t_) {
    cf_shards *S = a; uint8_t *sb = malloc(128 * 128 * 128); static const uint8_t fill[4] = {128, 128, 0, 0};
    for (long j = j0; j < j1 && sb; j++) {
        const int k = (int)(j / ((long)S->ny * S->nx)), sy = (int)(j / S->nx % S->ny), sx = (int)(j % S->nx);
        memset(sb, fill[k], 128 * 128 * 128);
        const int x0 = sx * 128, nx = S->W - x0 < 128 ? S->W - x0 : 128;
        for (int z = 0; z < S->nz; z++) for (int y = 0; y < 128 && sy * 128 + y < S->H; y++)
            memcpy(sb + ((size_t)z * 128 + y) * 128, S->pl[k] + ((size_t)z * S->H + sy * 128 + y) * S->W + x0, (size_t)nx);
        if (z3w_write_shard(S->zw[k], S->sz, S->oy + sy, S->ox + sx, sb, 1)) atomic_store(&S->fail, 1);
    }
    if (!sb) atomic_store(&S->fail, 1);
    free(sb);
}

static int cf_wr(FILE *f, const void *p, size_t sz, size_t n) { return !f || fwrite(p, sz, n, f) != n; }

/* |grad W| per solve cell in turns per field cell (central differences of U, minus the theta term), for --sp-w */
static float *cf_gradW(const cf_lvl *L, const float *cy, const float *cx, int na) {
    float *g = malloc(L->n * sizeof(float)); if (!g) return nullptr;
    const size_t st[3] = {(size_t)L->hh * L->w, (size_t)L->w, 1}; const int dm[3] = {L->d, L->hh, L->w};
    for (int z = 0; z < L->d; z++) for (int y = 0; y < L->hh; y++) for (int x = 0; x < L->w; x++) {
        const int c[3] = {z, y, x}; const size_t i = (size_t)z * st[0] + (size_t)y * st[1] + x; double gu[3];
        for (int a = 0; a < 3; a++) {
            const int lo = c[a] > 0 ? c[a] - 1 : c[a], hi = c[a] + 1 < dm[a] ? c[a] + 1 : c[a];
            gu[a] = hi > lo ? (L->U[i + (hi - c[a]) * (long)st[a]] - L->U[i - (c[a] - lo) * (long)st[a]]) / ((double)(hi - lo) * L->P) : 0;
        }
        double th[3]; cf_theta(L, cf_center(L, z), cf_center(L, y), cf_center(L, x), th);   /* per solve cell */
        for (int a = 0; a < 3; a++) gu[a] -= th[a] / L->P;
        g[i] = (float)sqrt(gu[0] * gu[0] + gu[1] * gu[1] + gu[2] * gu[2]);
    }
    (void)cy; (void)cx; (void)na; return g;
}

/* --model-stride K: the gauge sums exist only on the solve rows of every K-th chunk; the rows between (all-zero planes) take them linearly
   in z from the nearest summed rows above and below, per column (the nearest one at the ends). Before, those rows kept gg = 0 (no local
   gauge: U + the global gauge only) beyond the +-model-R smoothing, so the ref's phase agreement alternated with the chunk stride: 1667
   ref-dm1 (128-row chunks, K 4) coherent on ~256 of every 512 field rows, the low rows 0.03-0.34 (unroll #8 recheck, pub1667 11-14) */
static void cf_gauge_fill_z(float *gr, float *gi, int d, size_t hw) {
    char *has = calloc((size_t)d, 1); if (!has) return;
    for (int z = 0; z < d; z++) { const float *a = gr + (size_t)z * hw, *b = gi + (size_t)z * hw; for (size_t i = 0; i < hw; i++) if (a[i] != 0.f || b[i] != 0.f) { has[z] = 1; break; } }
    int zp = -1, nf = 0;
    for (int z = 0; z < d; z++) {
        if (has[z]) { zp = z; continue; }
        int zn = z + 1; while (zn < d && !has[zn]) zn++;
        if (zp < 0 && zn >= d) break;   /* (no summed row at all) */
        const double t = zp < 0 ? 1 : zn >= d ? 0 : (double)(z - zp) / (zn - zp);
        const float *ap = gr + (size_t)(zp < 0 ? zn : zp) * hw, *bp = gi + (size_t)(zp < 0 ? zn : zp) * hw, *an = gr + (size_t)(zn < d ? zn : zp) * hw, *bn = gi + (size_t)(zn < d ? zn : zp) * hw;
        float *a = gr + (size_t)z * hw, *b = gi + (size_t)z * hw;
        for (size_t i = 0; i < hw; i++) { a[i] = (float)((1 - t) * ap[i] + t * an[i]); b[i] = (float)((1 - t) * bp[i] + t * bn[i]); }
        nf++;
    }
    if (nf) fprintf(stderr, "count-field: --model-stride: the gauge of %d of %d solve rows interpolated in z from the summed rows\n", nf, d);
    free(has);
}
typedef struct { fit_model *m; const cf_lvl *L; const float *gr, *gi, *cy, *cx; int na; double gauge, pitch; } cf_lat;
static void cf_lat_run(void *a_, long a0, long a1, int t_) {   /* model lattice planes [a0, a1): U + the gauges - r / pitch */
    const cf_lat *T = a_; fit_model *m = T->m; const cf_lvl *L = T->L;
    for (long a = a0; a < a1; a++) for (int c = 0; c < m->ly; c++) for (int e = 0; e < m->lx; e++) {
        const double pz = (double)a * m->L, py = (double)c * m->L, px = (double)e * m->L, qz = pz / m->L - 0.5, qy = py / m->L - 0.5, qx = px / m->L - 0.5;   /* solve index */
        const double U = cf_tri(L->U, L->d, L->hh, L->w, qz, qy, qx), re = cf_tri(T->gr, L->d, L->hh, L->w, qz, qy, qx), im = cf_tri(T->gi, L->d, L->hh, L->w, qz, qy, qx);
        const double gg = hypot(re, im) > 1e-9 ? atan2(im, re) / (2 * M_PI) : 0;
        const double cy = cf_axis_at(T->cy, T->na, pz), cx = cf_axis_at(T->cx, T->na, pz);
        m->delta[((size_t)a * m->ly + c) * m->lx + e] = (float)(U + T->gauge + gg - hypot(py - cy, px - cx) / T->pitch);
    }
}

int cmd_count_field(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: ufsm count-field <in-fields> <out-fields> [--method ls|ray] [--pool P (solve cell in field cells, 0 auto)] [--R 10 (phase demod radius, field cells; 0: W only)] [--Rz R]\n"
                        "       [--phase-w 0 (confident network phase gradients in the targets, weight)] [--phase-c0 0.4] [--gauge demod|ls (ls: a smooth LS-unwrapped gauge instead of the +-R demodulation; with --R 0)] [--gauge-R 5 (solve cells)] [--conf w|phase|mix|pmax] [--trust C0,C1 (phase confidence ramp: the network phase kept)] [--conf-w 1 (scale of the count confidence)] [--keep 0 (p: + A^p wrap(phi - W_out), A the local agreement)] [--sp-w 0 (1: sp from |grad W|)] [--dir 0] [--dir-loops 0 (1: no ring vote: the direction from phase loops)] [--dir-default-ok 0 (1: accept dir +1 without evidence; else refused)] [--iters 400] [--lambda 1e-7] [--eps 1e-3] [--loop 8 (closing loop edge, field cells)] [--model DIR] [--model-R 8] [--rmin-um 500] [--emin 0.1] [--sheet0 0.4]\n"
                        "       [--wc 1 (0: no wc.f32)] [--mem-gb 6 (field chunk budget)] [--threads 8] [--cf-gpu 1 (pooling, CG, model pass on the GPU)] [--prefetch auto|0|1] [--read-threads N]\n"
                        "       [--model-only 0 (1: only the --model reference, no count field)] [--model-stride 1]\n"
                        "       [--pool-cache FILE] [--pool-only 1 (pass 1 into the cache, then exit)] [--follow ROWSFILE (pass 1 behind a running predict)] [--follow-split S] [--chunk-rows N]\n");
        return 2;
    }
    const char *in = argv[2], *out = argv[3], *meth = cf_opt(argc, argv, "--method", "ls"), *cmode = cf_opt(argc, argv, "--conf", "mix"), *mdir = cf_opt(argc, argv, "--model", "");
    cf_nt = ufsm_threads(opt_atoi("--threads", cf_opt(argc, argv, "--threads", "0")), 8);
    const int R = opt_atoi("--R", cf_opt(argc, argv, "--R", "10")), Rz = R > 0 ? opt_atoi("--Rz", cf_opt(argc, argv, "--Rz", cf_opt(argc, argv, "--R", "10"))) : 0;
    const int iters = opt_atoi("--iters", cf_opt(argc, argv, "--iters", "400")), Qf = opt_atoi("--loop", cf_opt(argc, argv, "--loop", "8")), Rm = opt_atoi("--model-R", cf_opt(argc, argv, "--model-R", "8")), wwc = opt_atoi("--wc", cf_opt(argc, argv, "--wc", "1"));
    const double lamr = opt_atof("--lambda", cf_opt(argc, argv, "--lambda", "1e-7")), eps = opt_atof("--eps", cf_opt(argc, argv, "--eps", "1e-3")), rmin_um = opt_atof("--rmin-um", cf_opt(argc, argv, "--rmin-um", "500"));
    double memgb = opt_atof("--mem-gb", cf_opt(argc, argv, "--mem-gb", "6"));
    {   /* the chunk budget never above 60% of what this process may still allocate (cgroup aware: pods, capped scopes) */
        const double av = (double)ufsm_mem_available() / (1u << 30);
        if (av > 0 && memgb > 0.6 * av) { fprintf(stderr, "count-field: --mem-gb %.1f lowered to %.1f (60%% of the %.1f GB available)\n", memgb, 0.6 * av, av); memgb = 0.6 * av; }
    }
    /* --model-only 1 (with --model): no output fields (the out directory is not written), only the model; --model-stride K: its phase
       gauge (and the R 0 global gauge) from every K-th field chunk (a whole scroll: the gauge is smoothed over +-model-R solve cells) */
    const int mo = opt_atoi("--model-only", cf_opt(argc, argv, "--model-only", "0")), mstride = opt_atoi("--model-stride", cf_opt(argc, argv, "--model-stride", "1"));
    if (mo && (!*mdir || mstride < 1 || *cf_opt(argc, argv, "--into-shape", ""))) { fprintf(stderr, "count-field: --model-only needs --model DIR, --model-stride >= 1, no --into-*\n"); return 2; }
    /* --cf-gpu 1 (default; the CPU when no CUDA device or nn_cf.o): pass 1's pooling, the CG solves and the --model-only R 0 model
       pass on the GPU (src/nn_cf.cu). --prefetch 1: the next field chunk loads on a thread meanwhile (two chunks in RAM;
       default: when two fit in half the available memory).
       --read-threads N: the zarr decode threads (default min(16, cores)) */
    int gpu = opt_atoi("--cf-gpu", cf_opt(argc, argv, "--cf-gpu", "1")) && cf_gpu_ok && cf_gpu_ok();
    int pref = opt_atoi("--prefetch", cf_opt(argc, argv, "--prefetch", "-1"));
    if (opt_atoi("--read-threads", cf_opt(argc, argv, "--read-threads", "0")) > 0) fit_set_threads(opt_atoi("--read-threads", cf_opt(argc, argv, "--read-threads", "0")));
    const int spiral = !strcmp(meth, "spiral");
    if (strcmp(meth, "ls") && strcmp(meth, "ray") && !spiral) { fprintf(stderr, "count-field: --method ls, ray or spiral\n"); return 2; }
    fit_model *imod = nullptr; if (*cf_opt(argc, argv, "--init-model", "")) { if (!spiral || !(imod = fit_load(cf_opt(argc, argv, "--init-model", "")))) { fprintf(stderr, "count-field: --init-model needs --method spiral and a fit model\n"); return 2; } }
    if (strcmp(cmode, "w") && strcmp(cmode, "phase") && strcmp(cmode, "mix") && strcmp(cmode, "pmax")) { fprintf(stderr, "count-field: --conf w, phase, mix or pmax\n"); return 2; }
    double tr0 = 0, tr1 = 0; if (*cf_opt(argc, argv, "--trust", "") && sscanf(cf_opt(argc, argv, "--trust", ""), "%lf,%lf", &tr0, &tr1) != 2) { fprintf(stderr, "count-field: --trust C0,C1\n"); return 2; }
    if (R < 0 || Rz < 0) { fprintf(stderr, "count-field: --R / --Rz >= 0\n"); return 2; }
    /* --mf-w K (the unroll's MFREF, DESIGN 2026-11-18 "counting around merges"): every field load scales pc ps w by 1 - K mf / 255 (the
       fields' merge-flag plane), so the counts route around merged sheets */
    const double mfw = opt_atof("--mf-w", cf_opt(argc, argv, "--mf-w", "0"));
    if (mfw > 0) { char mp[1400]; snprintf(mp, sizeof mp, "%s/mf/zarr.json", in); if (access(mp, F_OK)) { fprintf(stderr, "count-field: --mf-w needs %s (ufsm merge-flag)\n", mp); return 2; } fit_fields_mf_weight(mfw); }
    {   /* --dv-w K --dv-ref S (2026-11-27 densvar): the density evidence of a cell x min(1, e^(-K (s - S))), s its learned log-variance (predict --fit-dv) */
        const double dvw = opt_atof("--dv-w", cf_opt(argc, argv, "--dv-w", "0")), dvr = opt_atof("--dv-ref", cf_opt(argc, argv, "--dv-ref", "-6"));
        if (dvw < 0) { fprintf(stderr, "count-field: --dv-w K >= 0\n"); return 2; }
        if (dvw > 0) { char mp[1400]; snprintf(mp, sizeof mp, "%s/dv/zarr.json", in); if (access(mp, F_OK)) { fprintf(stderr, "count-field: --dv-w needs %s (predict --fit-dv 1)\n", mp); return 2; }
                       fit_fields_dv_weight(dvw, dvr); fprintf(stderr, "count-field: density weights x min(1, e^(-%g (s - %g))) from the dv plane\n", dvw, dvr); }
    }
    fit_fields_buf g; if (fit_fields_load(in, 0, 0, 0, &g)) { fprintf(stderr, "count-field: cannot load %s\n", in); return 1; }   /* geometry + the axis table */
    const int D = g.shape[0], H = g.shape[1], W = g.shape[2], na = (int)(D * g.h) + 1; const size_t n = (size_t)D * H * W, HW = (size_t)H * W;
    const double h = g.h;
    int P = opt_atoi("--pool", cf_opt(argc, argv, "--pool", "0"));
    if (P <= 0) { P = 1; while ((size_t)((D + P - 1) / P) * ((H + P - 1) / P) * ((W + P - 1) / P) > 48000000ull) P *= 2; if (P < 2 && n > 4000000) P = 2; }
    /* field rows per chunk: ~22 B per cell held (input planes, output buffers) incl. the Rz halo rows */
    int CZ = (int)(memgb * 1e9 / ((mo ? 12.0 : 22.0) * HW)) - 2 * Rz; CZ = CZ / P * P; if (CZ < P) CZ = P; if (CZ > D) CZ = (D + P - 1) / P * P;   /* (--model-only: no output buffers) */
    if (CZ >= 128 && 128 % P == 0) CZ = CZ / 128 * 128;   /* whole 128-row zarr shards per chunk (a shard straddling two chunks is decoded twice) */
    /* --chunk-rows N: the chunk height fixed (a multiple of the pool; the strided model pass and the prior's chunk depend on it, so a
       pass 1 run apart from the solve (--pool-only, --follow) and the solve run pass the same N; default: from --mem-gb as above) */
    /* a fixed N above what --mem-gb holds (the memory rule's CZ): the memory rule's, with a DEFAULTED: line (the rows change the ref;
       unroll.sh passes 512 so runs are reproducible, lead 2026-11-17) */
    /* (the cf-ref recipe on the GPU, --model-only R 0, holds whole chunks only for the prior: its 4 planes, ~4.5 B per cell; the passes
       read shard pieces (below), so a requested --chunk-rows is capped by that, not by the 12 B per cell of whole-chunk reads) */
    const int pieces_ok = (!getenv("UFSM_CF_PIECE_ROWS") || atoi(getenv("UFSM_CF_PIECE_ROWS")) > 0) && gpu && mo && !R && strcmp(cf_opt(argc, argv, "--gauge", "demod"), "ls")
                          && !opt_atoi("--sp-w", cf_opt(argc, argv, "--sp-w", "0")) && !opt_atoi("--pool-split", cf_opt(argc, argv, "--pool-split", "0"))
                          && opt_atof("--rho-solve", cf_opt(argc, argv, "--rho-solve", "1")) == 1 && !*cf_opt(argc, argv, "--rho-map", "") && !*cf_opt(argc, argv, "--rho-r", "");
    if (opt_atoi("--chunk-rows", cf_opt(argc, argv, "--chunk-rows", "0")) > 0) {
        int czp = (int)(memgb * 1e9 / (4.5 * HW)); czp = czp / P * P; if (czp < P) czp = P; if (czp >= 128 && 128 % P == 0) czp = czp / 128 * 128;
        const int req = opt_atoi("--chunk-rows", cf_opt(argc, argv, "--chunk-rows", "0")), czm = pieces_ok && czp > CZ ? czp : CZ;
        CZ = req / P * P; if (CZ < P) CZ = P; if (CZ > D) CZ = (D + P - 1) / P * P;
        if (CZ > czm) { UFSM_DEFAULTED("count-field chunk rows = %d (--chunk-rows %d does not fit --mem-gb %.1f: %d rows)", czm, req, memgb, czm); CZ = czm; }
    }
    if (pref < 0) {   /* --prefetch default: two chunks of ~11 planes in half of what this process may still allocate (cgroup aware: the
                         host's MemAvailable let a 1667 ref inside a 40 GB memcap scope prefetch two 19.5 GB chunks and be OOM-killed, 11-12) */
        const double av = (double)ufsm_mem_available();
        pref = 2 * 11.0 * CZ * HW < 0.5 * av;
    }
    int nl = 1; while (nl < 6) { const int q = P << nl; if ((D + q - 1) / q < 8 || (H + q - 1) / q < 8 || (W + q - 1) / q < 8) break; nl++; }
    cf_lvl *lv = calloc(nl, sizeof *lv); float **Sl = calloc(nl, sizeof *Sl), *(*Gl)[3] = calloc(nl, sizeof *Gl);
    for (int l = 0; l < nl; l++) {
        cf_lvl *L = &lv[l]; const int q = P << l;
        L->P = q; L->h = h; L->d = (D + q - 1) / q; L->hh = (H + q - 1) / q; L->w = (W + q - 1) / q; L->n = (size_t)L->d * L->hh * L->w;
        L->cy = g.cy; L->cx = g.cx; L->na = na; L->rmin = rmin_um / g.um;
        Sl[l] = calloc(L->n, sizeof(float)); for (int k = 0; k < 3; k++) Gl[l][k] = calloc(L->n, sizeof(float));
        if (!Sl[l] || !Gl[l][2]) { fprintf(stderr, "count-field: out of memory (solve grid)\n"); return 1; }
    }
    /* --pool-split 1 (geocount prototype, off by default): the density term's pooled sums kept apart (Sd, Gd; level 0, CPU pooling) so
       the solve can rescale the density count without re-pooling: --rho-solve K (global) and / or --rho-map FILE (float32 per level-0
       solve cell, z-major) give G += (K - 1) Gd, the phase-gradient targets (--phase-w) untouched. With --pool-cache the split sums go
       to / come from FILE.dens. (audit geo-count 11-14: the density head counts 1.2-1.3x the GT sheets on Paris 4 and 1667) */
    const double rho_solve = opt_atof("--rho-solve", cf_opt(argc, argv, "--rho-solve", "1")); const char *rho_map = cf_opt(argc, argv, "--rho-map", "");
    /* --rho-r "R0:K0,R1:K1,...": K by the radius from the umbilicus (mm; piecewise linear, constant past the ends), times --rho-solve
       and --rho-map. (fix0332 11-17: wrap-check's CT peaks per fitted turn by radius band is the label-free K(r); both 0332 fits
       undercount at r 2-4 mm, 1.2-1.45 CT peaks per turn) */
    enum { RHO_RN = 64 }; double rr_r[RHO_RN], rr_k[RHO_RN]; int rr_n = 0;
    { const char *s = cf_opt(argc, argv, "--rho-r", ""); while (*s && rr_n < RHO_RN) { char *e; const double r = strtod(s, &e); if (e == s || *e != ':') break; s = e + 1;
        const double k = strtod(s, &e); if (e == s) break; s = e;
        if (!(k > 0) || (rr_n && !(r > rr_r[rr_n - 1]))) { fprintf(stderr, "count-field: --rho-r: K > 0 and increasing radii, R:K pairs\n"); return 1; }
        rr_r[rr_n] = r; rr_k[rr_n++] = k; if (*s == ',') s++; }
      if (*s) { fprintf(stderr, "count-field: --rho-r: cannot parse \"%s\" (want R0:K0,R1:K1,... in mm)\n", s); return 1; } }
    const int psplit = opt_atoi("--pool-split", cf_opt(argc, argv, "--pool-split", "0")) || rho_solve != 1 || *rho_map || rr_n;
    float *Sd0 = nullptr, *Gd0[3] = {nullptr, nullptr, nullptr};
    if (psplit) { Sd0 = calloc(lv[0].n, sizeof(float)); for (int k = 0; k < 3; k++) Gd0[k] = calloc(lv[0].n, sizeof(float));
                  if (!Sd0 || !Gd0[2]) { fprintf(stderr, "count-field: out of memory (--pool-split)\n"); return 1; } }
    /* pass 1: the pooled sums chunk by chunk; the label-free prior (direction, pitch) from the chunk holding the middle row */
    int dir = opt_atoi("--dir", cf_opt(argc, argv, "--dir", "0")), recto = 0, w4d = 0; double pitch = 8.0 * h;
    int dsrc = dir ? FIT_DIR_GIVEN : -1;   /* where dir came from (fit.h FIT_DIR_*; -1 not yet decided) */
    if (!dir && imod) { dir = imod->dir; dsrc = FIT_DIR_GIVEN; }   /* (the --init-model's dir) */
    /* --pool-cache FILE: the level-0 pooled sums (+ dir, prior pitch, recto, w4d) saved after pass 1 / loaded instead of it (a rerun
       of a whole scroll skips the ~1 h pass; the file is tied to the pooling options, not checked) */
    const char *pcache = cf_opt(argc, argv, "--pool-cache", ""); int cached = 0;
    if (*pcache) { FILE *pf = fopen(pcache, "rb"); int hd[6]; double pv;
        if (pf && fread(hd, sizeof hd, 1, pf) == 1 && fread(&pv, sizeof pv, 1, pf) == 1 && hd[0] == lv[0].d && hd[1] == lv[0].hh && hd[2] == lv[0].w && hd[3] == P) {
            int ok = fread(Sl[0], sizeof(float), lv[0].n, pf) == lv[0].n; for (int k = 0; k < 3 && ok; k++) ok = fread(Gl[0][k], sizeof(float), lv[0].n, pf) == lv[0].n;
            if (ok && psplit) { char dn[PATH_MAX]; snprintf(dn, sizeof dn, "%s.dens", pcache); FILE *df = fopen(dn, "rb"); ok = df && fread(Sd0, sizeof(float), lv[0].n, df) == lv[0].n;
                for (int k = 0; k < 3 && ok; k++) ok = fread(Gd0[k], sizeof(float), lv[0].n, df) == lv[0].n;
                if (df) fclose(df);
                if (!ok) { fprintf(stderr, "count-field: --pool-split: no density sums %s (rerun pass 1 with --pool-split 1)\n", dn); return 1; } }
            if (ok) { cached = 1; if (!dir) { dir = hd[4]; dsrc = (hd[5] >> 16) - 1; if (dsrc < 0) dsrc = FIT_DIR_DEFAULT; }   /* (a cache without the source: unverified) */
                      pitch = pv; recto = hd[5] & 1; w4d = (hd[5] >> 1) & 0x7fff; fprintf(stderr, "count-field: pooled sums from %s\n", pcache); }
            else memset(Sl[0], 0, lv[0].n * sizeof(float));
        }
        if (pf) fclose(pf); }
    double t_pool = 0; const double tst0 = cf_now();
    if (mo) fit_fields_skip_planes(1 | 2);   /* (--model-only: no output planes, the recto / winding mod 4 planes unread) */
    /* the chunks' order: in z; --follow-split S (the second fields part's first row, a multiple of CZ): the two parts' chunks
       alternately (each part's predict writes its rows bottom-up at about the same pace). The pooled sums are per chunk (disjoint solve
       rows) and the prior comes from the chunk holding the middle row, so the order changes nothing */
    const int nch = (D + CZ - 1) / CZ; int *ord = malloc(nch * sizeof(int));
    {   const int fs = opt_atoi("--follow-split", cf_opt(argc, argv, "--follow-split", "0")), ks = fs > 0 && fs % CZ == 0 && fs < D ? fs / CZ : 0; int n = 0;
        if (*cf_opt(argc, argv, "--follow-split", "") && !ks) fprintf(stderr, "count-field: --follow-split %d is not a multiple of the %d-row chunks inside the box: chunks in z order\n", fs, CZ);
        if (ks) for (int i = 0; n < nch; i++) { if (i < ks) ord[n++] = i; if (ks + i < nch) ord[n++] = ks + i; }
        else for (int i = 0; i < nch; i++) ord[i] = i; }
    g_follow = *cf_opt(argc, argv, "--follow", "") ? cf_opt(argc, argv, "--follow", "") : nullptr;
    cf_pool pp0 = {nullptr, 0, 0, H, W, P, lv[0].hh, lv[0].w, h, g.cy, g.cx, na, Sl[0], {Gl[0][0], Gl[0][1], Gl[0][2]}, opt_atof("--emin", cf_opt(argc, argv, "--emin", "0.1")), opt_atof("--sheet0", cf_opt(argc, argv, "--sheet0", "0.4")), opt_atof("--phase-w", cf_opt(argc, argv, "--phase-w", "0")), opt_atof("--phase-c0", cf_opt(argc, argv, "--phase-c0", "0.4"))};
    pp0.rs = opt_atof("--rho-scale", cf_opt(argc, argv, "--rho-scale", "1")); if (!(pp0.rs > 0)) pp0.rs = 1;
    if (psplit) { pp0.Sd = Sd0; for (int k = 0; k < 3; k++) pp0.Gd[k] = Gd0[k]; }
    { double A = 0, S0 = 0; if (*cf_opt(argc, argv, "--rho-pow", "") && sscanf(cf_opt(argc, argv, "--rho-pow", ""), "%lf,%lf", &A, &S0) == 2 && S0 > 0) { pp0.ra = A; pp0.rb = S0 * 8 / (h * g.um); } }
    {   /* --wide A,B (network spacing, um) --wide-dens D (0..1) --wide-phase K: the density weight x (1 - D s), the confident phase weight x (1 + K s), s = ramp(spacing, A, B) */
        double a = 0, bw = 0; if (*cf_opt(argc, argv, "--wide", "") && sscanf(cf_opt(argc, argv, "--wide", ""), "%lf,%lf", &a, &bw) == 2 && bw > a) { pp0.wa = a * 8 / (h * g.um); pp0.wb = bw * 8 / (h * g.um); }
        if (pp0.wb > 0) { pp0.wd = opt_atof("--wide-dens", cf_opt(argc, argv, "--wide-dens", "0")); pp0.wp = opt_atof("--wide-phase", cf_opt(argc, argv, "--wide-phase", "0")); }
    }
    /* pass 1 in shard pieces (memopt 2026-11-18): the GPU pooling reads a chunk's rows RP at a time (RP = UFSM_CF_PIECE_ROWS, default
       128 = the fields' zarr chunk, rounded to the pool; 0: whole chunks), each piece's last solve row deferred to the next piece with
       its P + 1 rows kept (the phase gradient's halo row), the chunk bounds unchanged: the pooled sums of the whole-chunk path bit for
       bit (cf_gpu_pool_rows). The prior's chunk (the middle row's) is read again afterwards without the six normal planes it does not
       use. Host memory: two pieces (prefetch) instead of two chunks: 1667's ref 28.2 GB -> see DESIGN 2026-11-18 */
    int RP = getenv("UFSM_CF_PIECE_ROWS") ? atoi(getenv("UFSM_CF_PIECE_ROWS")) : 128; RP = RP / P * P; if (RP > 0 && RP < 2 * P) RP = 2 * P;
    const int pmode = !cached && gpu && !psplit && !getenv("UFSM_CF_READ_ONLY") && RP > 0 && RP < CZ;
    const cf_gpu_poolcfg gc0 = {H, W, P, lv[0].hh, lv[0].w, na, h, g.cy, g.cx, pp0.emin, pp0.sheet0, pp0.kph, pp0.c0, pp0.rs, pp0.ra, pp0.rb, pp0.wa, pp0.wb, pp0.wd, pp0.wp, pmode ? (getenv("UFSM_CF_STAGE_MB") ? atof(getenv("UFSM_CF_STAGE_MB")) * 1e6 : 512e6) : 0};   /* (pieces: 512 MB upload buffers, 1 GB pinned) */
    cf_pre pre = {0};
    const int gpu0 = gpu; int npc = 0;   /* (a chunk the GPU fails, e.g. short of memory on a shared device: the CPU; the next one tries again) */
    if (pmode) {
        const int ppref = pref || 2 * 11.0 * RP * HW < 0.5 * (double)ufsm_mem_available();   /* (two pieces held: the prefetch nearly always fits) */
        int np = 0; for (int ci = 0; ci < nch; ci++) { const int z0 = ord[ci] * CZ, z1 = z0 + CZ < D ? z0 + CZ : D; np += (z1 - z0 + RP - 1) / RP; }
        int (*pc3)[3] = malloc((size_t)np * sizeof *pc3); np = 0;   /* pieces: first row, end row, the chunk's first row */
        for (int ci = 0; ci < nch; ci++) { const int z0 = ord[ci] * CZ, z1 = z0 + CZ < D ? z0 + CZ : D; for (int a = z0; a < z1; a += RP) { pc3[np][0] = a; pc3[np][1] = a + RP < z1 ? a + RP : z1; pc3[np][2] = z0; np++; } }
        const size_t crn = (size_t)(P + 1) * HW; uint8_t *carry = malloc(10 * crn); if (!carry) { fprintf(stderr, "count-field: out of memory (piece carry)\n"); return 1; }
        int cr0 = 0, cfail = -1;   /* carry: field rows [cr0, cr0 + P + 1) of the previous piece; cfail: a chunk whose GPU pooling failed (redone on the CPU) */
        cf_pre_start(&pre, in, pc3[0][0], pc3[0][1], ppref);
        for (int j = 0; j < np; j++) {
            const int a = pc3[j][0], e = pc3[j][1], z0 = pc3[j][2], z1 = z0 + CZ < D ? z0 + CZ : D; fit_fields_buf b;
            if (cf_pre_take(&pre, &b)) return 1;
            if (ppref && j + 1 < np) cf_pre_start(&pre, in, pc3[j + 1][0], pc3[j + 1][1], 1);
            if (!a) { recto = b.f.recto != nullptr; w4d = b.f.wd[0] ? b.f.w4_d : 0; }
            const double tl1 = cf_now();
            if (cfail != z0) {
                const uint8_t *pl[10] = {b.f.pc, b.f.ps, b.f.t[0], b.f.t[1], b.f.t[2], b.f.t[3], b.f.t[4], b.f.t[5], b.f.sp, b.f.w};
                cf_gpu_rows bk[2]; int nbk = 0;
                if (a > z0) { for (int k = 0; k < 10; k++) bk[nbk].pl[k] = carry + k * crn; bk[nbk].r0 = cr0; bk[nbk].r1 = cr0 + P + 1; nbk++; }
                for (int k = 0; k < 10; k++) bk[nbk].pl[k] = pl[k]; bk[nbk].r0 = a; bk[nbk].r1 = e; nbk++;
                const int Zs0 = a > z0 ? a / P - 1 : a / P, Zs1 = e < z1 ? e / P - 1 : (e + P - 1) / P;
                const int tfail = getenv("UFSM_CF_TEST_PFAIL") && atoi(getenv("UFSM_CF_TEST_PFAIL")) == a;   /* (test: this piece's GPU pooling fails) */
                if (tfail || cf_gpu_pool_rows(&gc0, bk, nbk, z0, z1 - z0, Zs0, Zs1, Sl[0], Gl[0])) { fprintf(stderr, "count-field: GPU pooling failed: rows %d..%d on the CPU\n", z0, z1); cfail = z0; }
                else if (e < z1) { cr0 = e - P - 1; for (int k = 0; k < 10; k++) memcpy(carry + k * crn, pl[k] + (size_t)(cr0 - a) * HW, crn); }
            }
            t_pool += cf_now() - tl1;
            cf_pre_give(&pre, &b);
            if (!ppref && j + 1 < np) cf_pre_start(&pre, in, pc3[j + 1][0], pc3[j + 1][1], 0);
            if (cfail == z0 && e == z1) {   /* the failed chunk whole on the CPU (its solve rows zeroed first) */
                if (pre.busy) { fit_fields_buf nb_; if (cf_pre_take(&pre, &nb_)) return 1; cf_pre_give(&pre, &nb_); }   /* (the next piece's load finished first: one chunk + a piece held) */
                fit_fields_buf cb; if (fit_fields_load(in, z0, z1, 1, &cb)) { fprintf(stderr, "count-field: cannot load rows %d..%d of %s\n", z0, z1, in); return 1; }
                const size_t sa = (size_t)(z0 / P) * lv[0].hh * lv[0].w, se = (size_t)((z1 + P - 1) / P) * lv[0].hh * lv[0].w;
                memset(Sl[0] + sa, 0, (se - sa) * sizeof(float)); for (int k = 0; k < 3; k++) memset(Gl[0][k] + sa, 0, (se - sa) * sizeof(float));
                cf_pool pp = pp0; pp.f = &cb.f; pp.zoff = z0; pp.nzl = z1 - z0; const int nr = (z1 - z0 + P - 1) / P;
                pp.nb = (4 * cf_nt + nr - 1) / nr; if (pp.nb > lv[0].hh) pp.nb = lv[0].hh; if (pp.nb < 1) pp.nb = 1;
                const double tl2 = cf_now(); cf_par((long)nr * pp.nb, cf_pool_run, &pp); t_pool += cf_now() - tl2; npc++;
                fit_fields_free(&cb); cfail = -1;
                if (j + 1 < np) cf_pre_start(&pre, in, pc3[j + 1][0], pc3[j + 1][1], ppref);
            }
        }
        if (pre.has_nxt) { fit_fields_free(&pre.nxt); pre.has_nxt = 0; }
        free(carry); free(pc3);
        {   /* the prior from the middle row's chunk, as the whole-chunk path takes it (pc ps sp w and the optional planes; no normals) */
            const int zm0 = D / 2 / CZ * CZ, zm1 = zm0 + CZ < D ? zm0 + CZ : D; cf_pre pm_ = {0}; fit_fields_buf b;
            fit_fields_skip_planes(mo ? 1 | 2 | 4 : 4); cf_pre_start(&pm_, in, zm0, zm1, 0); const int rc = cf_pre_take(&pm_, &b); fit_fields_skip_planes(mo ? 1 | 2 : 0);
            if (rc) return 1;
            fit_cfg pc = fit_cfg_default(); pc.dir = dir; pc.dir_loops = opt_atoi("--dir-loops", cf_opt(argc, argv, "--dir-loops", "0")); fit_model pm; memset(&pm, 0, sizeof pm);
            if (!fit_init_prior(&b.f, &pc, &pm)) {
                if (!dir) { dir = pm.dir; dsrc = pm.dir_src; }
                float *pk = malloc(pm.nk * sizeof(float)); int npk = 0; for (int k = 0; k < pm.nk; k++) if (pm.pitch[k] > 0) pk[npk++] = pm.pitch[k];
                if (npk) { for (int a = 1; a < npk; a++) for (int c = a; c > 0 && pk[c] < pk[c - 1]; c--) { const float t = pk[c]; pk[c] = pk[c - 1]; pk[c - 1] = t; } pitch = pk[npk / 2]; }
                free(pk); free(pm.pitch); free(pm.offset);
            }
            fit_fields_free(&b);
        }
    }
    if (!cached && !pmode) cf_pre_start(&pre, in, ord[0] * CZ, ord[0] * CZ + CZ < D ? ord[0] * CZ + CZ : D, pref);
    for (int ci = 0; ci < nch && !cached && !pmode; ci++) {
        const int z0 = ord[ci] * CZ, z1 = z0 + CZ < D ? z0 + CZ : D; fit_fields_buf b;
        if (cf_pre_take(&pre, &b)) return 1;
        /* (without --prefetch the next chunk loads after this one is pooled and handed back: one chunk in RAM, not two; before, the
           synchronous load of the next chunk ran here, next to this one: two chunks either way, 39 GB for a 1667 ref at 768 rows) */
        if (pref && ci + 1 < nch) { const int n0 = ord[ci + 1] * CZ; cf_pre_start(&pre, in, n0, n0 + CZ < D ? n0 + CZ : D, pref); }
        if (!z0) { recto = b.f.recto != nullptr; w4d = b.f.wd[0] ? b.f.w4_d : 0; }
        cf_pool pp = pp0; pp.f = &b.f; pp.zoff = z0; pp.nzl = z1 - z0;
        const double tl1 = cf_now();
        if (getenv("UFSM_CF_READ_ONLY")) {   /* (bench: the field reads only; FNV-1a of the chunk's planes for byte-identity checks) */
            const uint8_t *pl[10] = {b.f.pc, b.f.ps, b.f.t[0], b.f.t[1], b.f.t[2], b.f.t[3], b.f.t[4], b.f.t[5], b.f.sp, b.f.w}; uint64_t hs = 1469598103934665603ull;
            if (!strcmp(getenv("UFSM_CF_READ_ONLY"), "h")) for (int k = 0; k < 10; k++) if (pl[k]) for (size_t i = 0; i < (size_t)(z1 - z0) * HW; i += 7) hs = (hs ^ pl[k][i]) * 1099511628211ull;
            fprintf(stderr, "count-field: read only: rows %d..%d hash %016llx\n", z0, z1, (unsigned long long)hs); cf_pre_give(&pre, &b);
            if (!pref && ci + 1 < nch) { const int n0 = ord[ci + 1] * CZ; cf_pre_start(&pre, in, n0, n0 + CZ < D ? n0 + CZ : D, 0); }
            continue; }
        if (gpu && !psplit) {   /* the chunk's solve rows on the GPU (the same sums in the same order; --pool-split: the CPU) */
            const cf_gpu_poolcfg gc = gc0;
            const uint8_t *pl[10] = {b.f.pc, b.f.ps, b.f.t[0], b.f.t[1], b.f.t[2], b.f.t[3], b.f.t[4], b.f.t[5], b.f.sp, b.f.w};
            if (cf_gpu_pool(&gc, pl, z0, z1 - z0, Sl[0], Gl[0])) {
                fprintf(stderr, "count-field: GPU pooling failed: rows %d..%d on the CPU\n", z0, z1); gpu = 0;
                const size_t a = (size_t)(z0 / P) * lv[0].hh * lv[0].w, e = (size_t)((z1 + P - 1) / P) * lv[0].hh * lv[0].w;
                memset(Sl[0] + a, 0, (e - a) * sizeof(float)); for (int k = 0; k < 3; k++) memset(Gl[0][k] + a, 0, (e - a) * sizeof(float));
            }
        }
        if (!gpu || psplit) { const int nr = (z1 - z0 + P - 1) / P; pp.nb = (4 * cf_nt + nr - 1) / nr; if (pp.nb > lv[0].hh) pp.nb = lv[0].hh; if (pp.nb < 1) pp.nb = 1;
          cf_par((long)nr * pp.nb, cf_pool_run, &pp); gpu = gpu0; npc++; }   /* (jobs own whole solve-row y bands: no shared cells; same sum order) */
        t_pool += cf_now() - tl1;
        if (z0 <= D / 2 && D / 2 < z1) {
            fit_cfg pc = fit_cfg_default(); pc.dir = dir; pc.dir_loops = opt_atoi("--dir-loops", cf_opt(argc, argv, "--dir-loops", "0")); fit_model pm; memset(&pm, 0, sizeof pm);
            if (!fit_init_prior(&b.f, &pc, &pm)) {
                if (!dir) { dir = pm.dir; dsrc = pm.dir_src; }
                float *pk = malloc(pm.nk * sizeof(float)); int np = 0; for (int k = 0; k < pm.nk; k++) if (pm.pitch[k] > 0) pk[np++] = pm.pitch[k];
                if (np) { for (int a = 1; a < np; a++) for (int c = a; c > 0 && pk[c] < pk[c - 1]; c--) { const float t = pk[c]; pk[c] = pk[c - 1]; pk[c - 1] = t; } pitch = pk[np / 2]; }
                free(pk); free(pm.pitch); free(pm.offset);
            }
        }
        cf_pre_give(&pre, &b);
        if (!pref && ci + 1 < nch) { const int n0 = ord[ci + 1] * CZ; cf_pre_start(&pre, in, n0, n0 + CZ < D ? n0 + CZ : D, 0); }
    }
    if (pre.has_nxt) { fit_fields_free(&pre.nxt); pre.has_nxt = 0; }
    if (!dir || dsrc == FIT_DIR_DEFAULT || dsrc < 0) {   /* no evidence yet: the label-free gates on 32 middle rows with every plane (the w4 planes
        included, which --model-only skips in pass 1): fit_init_prior's ring vote, phase loops and winding-mod-4 loops (fit_dir_w4); they
        decide only with --dir-loops 1, else they are printed beside the refusal */
        fit_fields_skip_planes(0); fit_fields_buf wb; const int zm = D / 2, z0 = zm - 16 < 0 ? 0 : zm - 16, z1 = zm + 16 > D ? D : zm + 16;
        if (!fit_fields_load(in, z0, z1, 1, &wb)) {
            fit_cfg gc = fit_cfg_default(); gc.dir_loops = opt_atoi("--dir-loops", cf_opt(argc, argv, "--dir-loops", "0")); fit_model gm; memset(&gm, 0, sizeof gm);
            if (!fit_init_prior(&wb.f, &gc, &gm) && gm.dir_src != FIT_DIR_DEFAULT) { dir = gm.dir; dsrc = gm.dir_src; }
            free(gm.pitch); free(gm.offset); fit_fields_free(&wb);
        }
        if (mo) fit_fields_skip_planes(1 | 2);
    }
    if (!dir) { dir = 1; dsrc = FIT_DIR_DEFAULT; }
    if (dsrc < 0) dsrc = FIT_DIR_DEFAULT;
    fprintf(stderr, "count-field: dir %+d (%s)\n", dir, fit_dir_src_name(dsrc));
    if (dsrc == FIT_DIR_DEFAULT && !opt_atoi("--dir-default-ok", cf_opt(argc, argv, "--dir-default-ok", "0")) && !opt_atoi("--pool-only", cf_opt(argc, argv, "--pool-only", "0")) && !getenv("UFSM_CF_READ_ONLY")) {   /* (--pool-only: the cache
           keeps the source; the solve run decides) */
        fprintf(stderr, "count-field: REFUSED: no evidence for the spiral's direction (no ring vote%s). Pass --dir +1 / -1 from a published surface or GT "
                        "(unroll.sh: SPIRALDIR or scripts/unroll/spiral_dir.tsv), or --dir-loops 1 (phase loops), or --dir-default-ok 1 to accept +1. "
                        "A wrong hand keeps distances but hops two sheets per turn (PHerc1667 11-12: sheet identity 0.487 vs 0.762)\n", cached ? ", or a pool cache without one" : "");
        return 2;
    }
    if (getenv("UFSM_CF_READ_ONLY")) { fprintf(stderr, "count-field: read only: waited for field reads %.0f s, wall %.0f s\n", pre.t, cf_now() - tst0); return 0; }
    if (!cached) fprintf(stderr, "count-field: pass 1: waited for field reads %.0f s, pooling %.0f s (%s, %d of %d chunks on the CPU; %d rows per chunk%s), wall %.0f s\n", pre.t, t_pool, gpu ? "GPU" : "CPU", gpu ? npc : (D + CZ - 1) / CZ, (D + CZ - 1) / CZ, CZ, pmode ? ", read in pieces" : "", cf_now() - tst0);
    if (*pcache && !cached) { FILE *pf = fopen(pcache, "wb"); const int hd[6] = {lv[0].d, lv[0].hh, lv[0].w, P, dir, recto | w4d << 1 | (dsrc + 1) << 16};
        int bad = !pf || fwrite(hd, sizeof hd, 1, pf) != 1 || fwrite(&pitch, sizeof pitch, 1, pf) != 1 || fwrite(Sl[0], sizeof(float), lv[0].n, pf) != lv[0].n;
        for (int k = 0; k < 3 && !bad; k++) bad = fwrite(Gl[0][k], sizeof(float), lv[0].n, pf) != lv[0].n;
        if (pf && fclose(pf)) bad = 1;
        if (psplit && !bad) { char dn[PATH_MAX]; snprintf(dn, sizeof dn, "%s.dens", pcache); FILE *df = fopen(dn, "wb"); bad = !df || fwrite(Sd0, sizeof(float), lv[0].n, df) != lv[0].n;
            for (int k = 0; k < 3 && !bad; k++) bad = fwrite(Gd0[k], sizeof(float), lv[0].n, df) != lv[0].n;
            if (df && fclose(df)) bad = 1; }
        fprintf(stderr, bad ? "count-field: cannot write the pool cache %s\n" : "count-field: pooled sums -> %s\n", pcache);
        if (opt_atoi("--pool-only", cf_opt(argc, argv, "--pool-only", "0"))) return bad; }   /* (--pool-only 1: pass 1 into --pool-cache, nothing else) */
    free(ord); g_follow = nullptr;
    if (rho_solve != 1 || *rho_map || rr_n) {   /* --rho-solve K / --rho-map FILE / --rho-r: the density count rescaled per solve cell, G += (K - 1) Gd */
        float *km = nullptr;
        if (*rho_map) { FILE *mf = fopen(rho_map, "rb"); km = malloc(lv[0].n * sizeof(float));
            if (!mf || !km || fread(km, sizeof(float), lv[0].n, mf) != lv[0].n) { fprintf(stderr, "count-field: --rho-map %s: need %zu float32 (the level-0 solve grid %dx%dx%d)\n", rho_map, lv[0].n, lv[0].d, lv[0].hh, lv[0].w); return 1; }
            fclose(mf); }
        if (rr_n) { if (!km) { km = malloc(lv[0].n * sizeof(float)); if (!km) return 1; for (size_t i = 0; i < lv[0].n; i++) km[i] = 1; }
            const cf_lvl *L0 = &lv[0];
            for (int z = 0; z < L0->d; z++) { const double pz = cf_center(L0, z), cy = cf_axis_at(g.cy, na, pz), cx = cf_axis_at(g.cx, na, pz);
                for (int y = 0; y < L0->hh; y++) for (int x = 0; x < L0->w; x++) {
                    const size_t i = ((size_t)z * L0->hh + y) * L0->w + x; const double r = hypot(cf_center(L0, y) - cy, cf_center(L0, x) - cx) * g.um / 1000;
                    double k = rr_k[0]; if (r >= rr_r[rr_n - 1]) k = rr_k[rr_n - 1];
                    else if (r > rr_r[0]) { int j = 1; while (r > rr_r[j]) j++; const double t = (r - rr_r[j - 1]) / (rr_r[j] - rr_r[j - 1]); k = rr_k[j - 1] + t * (rr_k[j] - rr_k[j - 1]); }
                    if (isfinite(km[i]) && km[i] > 0) km[i] = (float)(km[i] * k); else km[i] = (float)k; } } }
        double ks = 0, kw = 0;
        for (size_t i = 0; i < lv[0].n; i++) { if (!(Sd0[i] > 0)) continue; const double K = rho_solve * (km && isfinite(km[i]) && km[i] > 0 ? km[i] : 1);
            for (int k = 0; k < 3; k++) Gl[0][k][i] += (float)((K - 1) * Gd0[k][i]);
            ks += K * Sd0[i]; kw += Sd0[i]; }
        fprintf(stderr, "count-field: density count rescaled (--rho-solve %g%s%s%s%s): weighted mean K %.3f\n", rho_solve, *rho_map ? ", --rho-map " : "", rho_map,
                rr_n ? ", --rho-r " : "", rr_n ? cf_opt(argc, argv, "--rho-r", "") : "", kw > 0 ? ks / kw : 0);
        free(km);
    }
    free(Sd0); for (int k = 0; k < 3; k++) free(Gd0[k]);
    if (getenv("UFSM_CF_ZTILT")) {   /* per solve row: the share of data cells whose target normal tilts >= 30 deg out of the y-x plane (z crush) */
        for (int z = 0; z < lv[0].d; z++) { double nc = 0, nt = 0, sz = 0;
            for (size_t i = (size_t)z * lv[0].hh * lv[0].w; i < (size_t)(z + 1) * lv[0].hh * lv[0].w; i++) { if (!(Sl[0][i] > 0)) continue;
                const double gz = Gl[0][0][i], gl = sqrt(gz * gz + Gl[0][1][i] * Gl[0][1][i] + Gl[0][2][i] * Gl[0][2][i]); if (gl <= 0) continue;
                nc++; nt += fabs(gz) / gl >= 0.5; sz += fabs(gz) / gl; }
            fprintf(stderr, "count-field ztilt: row %d (field rows %d..%d): %.0f cells, tilted >= 30 deg %.3f, mean |n_z| %.3f\n", z, z * P, z * P + P, nc, nc ? nt / nc : 0, nc ? sz / nc : 0); }
    }
    if (*cf_opt(argc, argv, "--ct-cal", "")) {   /* --ct-cal FILE (ufsm ct-cal): the pooled count targets scaled by the CT's sheets per density turn */
        fit_ct_cal *cc = fit_ct_cal_load(cf_opt(argc, argv, "--ct-cal", "")); if (!cc) { fprintf(stderr, "count-field: cannot read --ct-cal %s\n", cf_opt(argc, argv, "--ct-cal", "")); return 1; }
        double ks = 0, kw = 0; const double kmax = opt_atof("--ct-cal-max", cf_opt(argc, argv, "--ct-cal-max", "1")), kmin = opt_atof("--ct-cal-min", cf_opt(argc, argv, "--ct-cal-min", "0"));   /* (--ct-cal-min: a floor, e.g. for a phase calibration where the phase is dead) */   /* (one-sided by default: the CT only lowers counts; Paris 4's CT counts ~5% above its GT) */
        for (int z = 0; z < lv[0].d; z++) {
            const double pz = (z + 0.5) * P * h, ay = cf_axis_at(g.cy, na, pz), ax = cf_axis_at(g.cx, na, pz);
            for (int y = 0; y < lv[0].hh; y++) for (int x = 0; x < lv[0].w; x++) {
                const size_t i = ((size_t)z * lv[0].hh + y) * lv[0].w + x; if (!(Sl[0][i] > 0)) continue;
                const double k = fmax(kmin, fmin(kmax, fit_ct_cal_k(cc, pz, (y + 0.5) * P * h - ay, (x + 0.5) * P * h - ax)));
                for (int q = 0; q < 3; q++) Gl[0][q][i] *= (float)k;
                ks += k * Sl[0][i]; kw += Sl[0][i];
            }
        }
        fprintf(stderr, "count-field: --ct-cal: count targets scaled by the CT's sheets per density turn, weighted mean %.3f\n", kw > 0 ? ks / kw : 0);
        fit_ct_cal_free(cc);
    }
    if (*cf_opt(argc, argv, "--gz-scale", "")) {   /* --gz-scale F: the z component of the pooled count targets times F (0: dU/dz -> 0, the
                                                     winding of a sheet constant along z unless the radial counts force it; the network's n_z
                                                     integrated over the height drifted Paris 3's gauge by up to ~40 turns: geom-check, src/geomcheck.c) */
        const float s = (float)opt_atof("--gz-scale", cf_opt(argc, argv, "--gz-scale", "1")); for (size_t i = 0; i < lv[0].n; i++) Gl[0][0][i] *= s;
        fprintf(stderr, "count-field: --gz-scale %g: z component of the count targets scaled\n", s);
    }
    float kpitch[2] = {(float)pitch, (float)pitch}, koff[2] = {0, 0}; const int kdz = (int)fmax(1, D * h);   /* the prior: r / pitch */
    fprintf(stderr, "count-field: %dx%dx%d cells (h %g, %.2f um), dir %+d, prior pitch %.1f native voxels, solve cells %d, chunks of %d rows (+-%d halo)\n", D, H, W, h, g.um, dir, pitch, P, CZ, Rz);
    for (int l = 1; l < nl; l++) {   /* coarser sums from the finer level */
        const cf_lvl *F = &lv[l - 1], *L = &lv[l];
        for (int z = 0; z < F->d; z++) for (int y = 0; y < F->hh; y++) for (int x = 0; x < F->w; x++) {
            const size_t i = ((size_t)z * F->hh + y) * F->w + x, j = ((size_t)(z / 2) * L->hh + y / 2) * L->w + x / 2;
            Sl[l][j] += Sl[l - 1][i]; for (int k = 0; k < 3; k++) Gl[l][k][j] += Gl[l - 1][k][i];
        }
    }
    /* --fibers F.fib (ufsm fiber-export in this fields box's frame) / --pairs F.f32 (records za ya xa zb yb xb dW, box-local native):
       same-sheet pairs in the solve, weight --cf-fib W (0: off), Huber --cf-fib-huber D turns over --cf-fib-irls K reweighted level-0
       solves. A fibre: resampled at 1 native voxel, pairs every 4th sample (--cf-fib-every N) at the spans --cf-fib-spans (native, default 16,32,64,128),
       dW = 0 (W is constant along a sheet away from the theta cut; the cut is in t through dir * dtheta / 2 pi) */
    cf_pt *pts = nullptr; size_t npt = 0; const double wfib = opt_atof("--cf-fib", cf_opt(argc, argv, "--cf-fib", "0"));
    const char *apin = cf_opt(argc, argv, "--anchor-pins", ""), *apair = cf_opt(argc, argv, "--anchor-pairs", "");
    if (wfib > 0 || *apin || *apair) {
        size_t cap = 0; const double zo = opt_atof("--fib-zoff", cf_opt(argc, argv, "--fib-zoff", "0"));
        #define CF_PUSH(A, B, DW) do { if (npt == cap) { cap = cap ? 2 * cap : 1 << 16; pts = realloc(pts, cap * sizeof *pts); if (!pts) return 1; } \
            cf_pt *e_ = &pts[npt++]; e_->w = 0; e_->pin = 0; for (int k_ = 0; k_ < 3; k_++) { e_->a[k_] = (float)(A)[k_]; e_->b[k_] = (float)(B)[k_]; } \
            double ta_ = atan2((A)[1] - cf_axis_at(g.cy, na, (A)[0]), (A)[2] - cf_axis_at(g.cx, na, (A)[0])), tb_ = atan2((B)[1] - cf_axis_at(g.cy, na, (B)[0]), (B)[2] - cf_axis_at(g.cx, na, (B)[0])); \
            double dt_ = tb_ - ta_; dt_ -= 2 * M_PI * floor((dt_ + M_PI) / (2 * M_PI)); e_->t = (float)((DW) + dir * dt_ / (2 * M_PI)); } while (0)
        if (wfib > 0 && *cf_opt(argc, argv, "--fibers", "")) {
            fib_set fs; if (fib_load(cf_opt(argc, argv, "--fibers", ""), &fs)) { fprintf(stderr, "count-field: cannot read --fibers %s\n", cf_opt(argc, argv, "--fibers", "")); return 1; }
            int sp[8] = {16, 32, 64, 128}, nsp = 4; const int fev = opt_atoi("--cf-fib-every", cf_opt(argc, argv, "--cf-fib-every", "4"));   /* (pairs every fev samples: a whole scroll's AFV gives ~10^8 pairs at 4) */
            if (*cf_opt(argc, argv, "--cf-fib-spans", "")) { nsp = 0; char *t = strdup(cf_opt(argc, argv, "--cf-fib-spans", "")); for (char *q = strtok(t, ","); q && nsp < 8; q = strtok(nullptr, ",")) sp[nsp++] = atoi(q); free(t); }
            double *rq = nullptr; size_t rcap = 0;
            for (int r = 0; r < fs.n; r++) {
                const float *pp = fs.pts + 3 * (size_t)fs.off[r]; const int n = fs.off[r + 1] - fs.off[r]; size_t ns = 0; double acc = 0, next = 0;
                for (int k = 1; k < n; k++) {
                    const double a[3] = {pp[3 * k - 3] - zo, pp[3 * k - 2], pp[3 * k - 1]}, d[3] = {pp[3 * k] - pp[3 * k - 3], pp[3 * k + 1] - pp[3 * k - 2], pp[3 * k + 2] - pp[3 * k - 1]};
                    const double Ls = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                    for (; Ls > 0 && next <= acc + Ls; next += 1) {
                        if (ns == rcap) { rcap = rcap ? 2 * rcap : 4096; rq = realloc(rq, 3 * rcap * sizeof *rq); if (!rq) return 1; }
                        const double t = (next - acc) / Ls; for (int e = 0; e < 3; e++) rq[3 * ns + e] = a[e] + t * d[e]; ns++;
                    }
                    acc += Ls;
                }
                for (size_t b = 0; b < ns; b++) for (int di = 0; di < nsp; di++) {
                    if (b < (size_t)sp[di] || (b - sp[di]) % (size_t)(fev > 0 ? fev : 4)) continue;
                    const double *A = rq + 3 * (b - sp[di]), *B = rq + 3 * b;
                    if (A[0] < 0 || B[0] < 0 || A[0] > D * h || B[0] > D * h) continue;   /* (inside the box's rows) */
                    CF_PUSH(A, B, 0.0);
                }
            }
            free(rq); fprintf(stderr, "count-field: %d fibres -> %zu same-sheet pairs (spans", fs.n, npt); for (int k = 0; k < nsp; k++) fprintf(stderr, " %d", sp[k]); fprintf(stderr, ")\n"); fib_free(&fs);
        }
        if (wfib > 0 && *cf_opt(argc, argv, "--pairs", "")) {
            FILE *pf = fopen(cf_opt(argc, argv, "--pairs", ""), "rb"); float rec[7]; size_t n0 = npt;
            if (!pf) { fprintf(stderr, "count-field: cannot read --pairs %s\n", cf_opt(argc, argv, "--pairs", "")); return 1; }
            while (fread(rec, sizeof rec, 1, pf) == 1) { const double A[3] = {rec[0], rec[1], rec[2]}, B[3] = {rec[3], rec[4], rec[5]}; CF_PUSH(A, B, rec[6]); }
            fclose(pf); fprintf(stderr, "count-field: %zu pairs from --pairs\n", npt - n0);
        }
        /* --anchor-pins F (z y x U' n) / --anchor-pairs F (za ya xa zb yb xb dU n): ground-truth anchors (ufsm gt-anchors, box-local
           native), rows U(a) = U' / U(b) - U(a) = dU + dir dtheta / 2 pi, weight --anchor-w x min(n, --anchor-ncap) (n: anchors merged
           into the record), never Huber-reweighted (the GT is the truth: the data yields) */
        const double aw = opt_atof("--anchor-w", cf_opt(argc, argv, "--anchor-w", "10")), ancap = opt_atof("--anchor-ncap", cf_opt(argc, argv, "--anchor-ncap", "4"));
        for (int kind = 0; kind < 2; kind++) {
            const char *fn = kind ? apair : apin; if (!*fn) continue;
            FILE *af = fopen(fn, "rb"); if (!af) { fprintf(stderr, "count-field: cannot read %s\n", fn); return 1; }
            float rec[8]; const int nr = kind ? 8 : 5; size_t n0 = npt;
            while (fread(rec, 4, nr, af) == (size_t)nr) {
                const double A[3] = {rec[0], rec[1], rec[2]};
                if (kind) { const double B[3] = {rec[3], rec[4], rec[5]}; CF_PUSH(A, B, rec[6]); }
                else { CF_PUSH(A, A, 0.0); pts[npt - 1].pin = 1; pts[npt - 1].t = rec[3]; }
                pts[npt - 1].w = (float)(aw * fmin(rec[nr - 1], ancap));
            }
            fclose(af); fprintf(stderr, "count-field: %zu GT anchor %s from %s (weight %g x min(n, %g))\n", npt - n0, kind ? "pairs" : "pins", fn, aw, ancap);
        }
        #undef CF_PUSH
        if (!npt) fprintf(stderr, "count-field: --cf-fib %g but no pairs (--fibers / --pairs): off\n", wfib);
        /* pairs with an endpoint in a no-data solve cell (pooled weight S < --cf-fib-smin x the median S of the data cells, default 0.1):
           dropped (--cf-fib-nodata keep: kept). In a zeroed (masked) CT W has no data and stays near flat; a pair from there to the same
           sheet's data part pulls that part's radial count down: PHerc1667's published-surface oracle pairs (19 % with an endpoint on
           zero CT) left the unmasked turns 18-30 off by -1 on up to 65 % of their area within 0.8 mm of the mask, 5 % beyond 50 mm
           (geo-outer, DESIGN 2026-11-14). The share is the masked-CT gate's count-field line (> 0.02: a warning) */
        if (npt) {
            const cf_lvl *L0 = &lv[0]; const float *S0 = Sl[0]; size_t nd = 0;
            for (size_t i = 0; i < L0->n; i++) nd += S0[i] > 0;
            float *sv = malloc((nd ? nd : 1) * sizeof(float)); size_t k = 0; for (size_t i = 0; i < L0->n; i++) if (S0[i] > 0) sv[k++] = S0[i];
            qsort(sv, nd, sizeof(float), cf_cmpf); const double smed = nd ? sv[nd / 2] : 0;
            free(sv);
            const double smin = opt_atof("--cf-fib-smin", cf_opt(argc, argv, "--cf-fib-smin", "0.1")) * smed; const int keep = !strcmp(cf_opt(argc, argv, "--cf-fib-nodata", "drop"), "keep");
            size_t nk = 0, nbad = 0;
            for (size_t p = 0; p < npt; p++) {   /* (GT anchors are kept: their point is to set U where the data is weak) */
                const int ok = pts[p].w > 0 || (cf_data_at(L0, S0, smin, pts[p].a) && cf_data_at(L0, S0, smin, pts[p].b)); nbad += !ok; if (ok || keep) pts[nk++] = pts[p]; }
            const double sh = (double)nbad / (double)npt;
            fprintf(stderr, "count-field: no-data: %zu of %zu pairs (%.4f) have an endpoint in a no-data solve cell (S < %.3g, %.2f x the median); %s\n",
                    nbad, npt, sh, smin, opt_atof("--cf-fib-smin", cf_opt(argc, argv, "--cf-fib-smin", "0.1")), keep ? "kept (--cf-fib-nodata keep)" : "dropped");
            if (sh > 0.02) fprintf(stderr, "count-field: WARNING no-data pairs %.4f > 0.02: the pairs reach into masked / empty CT (W has no data there; fields on unmasked CT?)\n", sh);
            npt = nk;
            if (!npt) fprintf(stderr, "count-field: no pairs left on data cells: pairs off\n");
        }
    }
    const double fhub = opt_atof("--cf-fib-huber", cf_opt(argc, argv, "--cf-fib-huber", "0.25")); const int firls = opt_atoi("--cf-fib-irls", cf_opt(argc, argv, "--cf-fib-irls", "3"));
    const int rob_k = opt_atoi("--cf-robust", cf_opt(argc, argv, "--cf-robust", "0")), rob_all = opt_atoi("--cf-robust-all", cf_opt(argc, argv, "--cf-robust-all", "1"));
    const double rob_d = opt_atof("--cf-robust-delta", cf_opt(argc, argv, "--cf-robust-delta", "0.02"));
    if (rob_k > 0 && !(rob_d > 0)) { fprintf(stderr, "count-field: --cf-robust-delta %g: must be > 0\n", rob_d); return 2; }
    float *Sfine = Sl[0], *Gfine[3] = {Gl[0][0], Gl[0][1], Gl[0][2]}; const double tst1 = cf_now();
    for (int l = nl - 1; l >= 0; l--) {
        cf_lvl *L = &lv[l]; const int q = L->P; float *S = Sl[l], **G = Gl[l];
        L->dir = dir; L->pitch = kpitch; L->offset = koff; L->nk = 2; L->knot_dz = kdz;
        for (int k = 0; k < 3; k++) { L->wt[k] = malloc(L->n * sizeof(float)); L->tg[k] = malloc(L->n * sizeof(float)); }
        cf_build B = {L, S, {G[0], G[1], G[2]}, eps}; cf_par(L->d, cf_build_run, &B);
        double ws = 0; size_t nw = 0; for (size_t i = 0; i < L->n; i++) if (L->wt[0][i] > eps) { ws += L->wt[0][i]; nw++; }
        L->lam = lamr * (nw ? ws / nw : 1);
        L->b = malloc(L->n * sizeof(float)); L->diag = malloc(L->n * sizeof(float)); L->U = malloc(L->n * sizeof(float));
        L->r = malloc(L->n * sizeof(float)); L->p = malloc(L->n * sizeof(float)); L->Ap = malloc(L->n * sizeof(float));
        if (!L->b || !L->diag || !L->U || !L->r || !L->p || !L->Ap) { fprintf(stderr, "count-field: out of memory (level %d)\n", l); return 1; }
        if (npt) L->pr = cf_pairs_build(L, pts, npt, wfib);
        cf_rhs RH = {L}; cf_par(L->d, cf_rhs_run, &RH);
        if (l) { free(S); for (int k = 0; k < 3; k++) free(G[k]); }
        if (spiral && imod) {   /* --method spiral --init-model: U from the reference model (no LS solve) */
            if (l == 0) {
                const double oz = (double)(g.origin[0] - imod->origin[0]), oy = (double)(g.origin[1] - imod->origin[1]), ox = (double)(g.origin[2] - imod->origin[2]);
                for (int z = 0; z < L->d; z++) { const double pz = cf_center(L, z), cy = cf_axis_at(g.cy, na, pz), cx = cf_axis_at(g.cx, na, pz);
                    for (int y = 0; y < L->hh; y++) for (int x = 0; x < L->w; x++) { const double py = cf_center(L, y), px = cf_center(L, x);
                        L->U[((size_t)z * L->hh + y) * L->w + x] = (float)(fit_W(imod, pz + oz, py + oy, px + ox) + dir * atan2(py - cy, px - cx) / (2 * M_PI)); } }
            }
            continue;
        }
        if (!strcmp(meth, "ray")) { if (l == 0) { cf_ray RY = {L, Sfine, {Gfine[0], Gfine[1], Gfine[2]}}; cf_par(L->d, cf_ray_run, &RY); } continue; }
        if (l == nl - 1) { for (int z = 0; z < L->d; z++) for (int y = 0; y < L->hh; y++) for (int x = 0; x < L->w; x++) L->U[((size_t)z * L->hh + y) * L->w + x] = (float)cf_prior(L, cf_center(L, z), cf_center(L, y), cf_center(L, x)); }
        else {   /* prolong the coarser level */
            cf_prl Q = {L, &lv[l + 1]}; cf_par(L->d, cf_prl_run, &Q);
        }
        double rr; const double tcg = cf_now(); int it = -1;
        if (gpu && !L->pr) { const float *wt3[3] = {L->wt[0], L->wt[1], L->wt[2]}; it = cf_gpu_cg(L->d, L->hh, L->w, L->lam, wt3, L->b, L->diag, L->U, l == nl - 1 ? 4 * iters : iters, 1e-5, &rr);
                   if (it < 0) fprintf(stderr, "count-field: GPU CG failed: level %d on the CPU\n", l); }
        const int on_gpu = it >= 0;
        if (it < 0) it = cf_cg_solve(L, l == nl - 1 ? 4 * iters : iters, 1e-5, &rr);
        fprintf(stderr, "count-field: level %d (%d field cells): %dx%dx%d, %d CG iterations, relative residual %.2e (%s %.1f s)\n", l, q, L->d, L->hh, L->w, it, rr, on_gpu ? "GPU" : "CPU", cf_now() - tcg);
        if (L->pr && l == 0) {   /* Huber IRLS on the pairs (level 0 only; the coarse levels take them unweighted) */
            fprintf(stderr, "count-field: pairs |c.U - t| > 0.5 turn: %.4f of %zu\n", cf_pairs_reweight(L->pr, L->U, fhub, 0), L->pr->np);
            size_t nfree = 0; for (size_t pp = 0; pp < L->pr->np; pp++) nfree += !L->pr->fix[pp];   /* (anchors only: nothing to reweight) */
            for (int k = 0; k < firls && nfree; k++) {
                const double bad = cf_pairs_reweight(L->pr, L->U, fhub, 1); cf_par(L->d, cf_rhs_run, &RH);
                const int it2 = cf_cg_solve(L, iters, 1e-5, &rr);
                fprintf(stderr, "count-field: IRLS %d (Huber %.2f): before %.4f off by > 0.5; %d CG iterations, residual %.2e -> %.4f\n", k + 1, fhub, bad, it2, rr, cf_pairs_reweight(L->pr, L->U, fhub, 0));
            }
        }
        if (rob_k > 0 && (l == 0 || rob_all)) {   /* --cf-robust: IRLS on the count edges (cf_rob_run) */
            cf_rob RB = {L, {nullptr, nullptr, nullptr}, rob_d, {0}, {0}};
            for (int k = 0; k < 3; k++) { RB.w0[k] = malloc(L->n * sizeof(float)); if (!RB.w0[k]) { fprintf(stderr, "count-field: out of memory (--cf-robust)\n"); return 1; } memcpy(RB.w0[k], L->wt[k], L->n * sizeof(float)); }
            for (int k = 0; k <= rob_k; k++) {
                cf_par(L->d, cf_rob_run, &RB); double bad = 0, ws = 0; for (int t = 0; t < CF_MAXT; t++) { bad += RB.bad[t]; ws += RB.wsum[t]; }
                if (k == rob_k) { fprintf(stderr, "count-field: level %d robust (L1, Huber %.3f): edge weight off by > 0.5 turn %.4f after %d IRLS passes\n", l, rob_d, ws > 0 ? bad / ws : 0, rob_k); break; }
                if (L->pr) cf_pairs_q(L->pr, L->U);
                cf_par(L->d, cf_rhs_run, &RH); int it2 = -1;
                if (gpu && !L->pr) { const float *wt3[3] = {L->wt[0], L->wt[1], L->wt[2]}; it2 = cf_gpu_cg(L->d, L->hh, L->w, L->lam, wt3, L->b, L->diag, L->U, iters, 1e-5, &rr); }
                if (it2 < 0) it2 = cf_cg_solve(L, iters, 1e-5, &rr);
                if (k == 0 || k + 1 == rob_k) fprintf(stderr, "count-field: level %d robust IRLS %d: before %.4f of the edge weight off by > 0.5 turn; %d CG iterations, residual %.2e\n", l, k + 1, ws > 0 ? bad / ws : 0, it2, rr);
            }
            for (int k = 0; k < 3; k++) free(RB.w0[k]);
        }
        if (l + 1 < nl) { cf_lvl *C = &lv[l + 1]; cf_pairs_free(C->pr); C->pr = nullptr; free(C->b); free(C->diag); free(C->r); free(C->p); free(C->Ap); for (int k = 0; k < 3; k++) { free(C->wt[k]); free(C->tg[k]); } C->b = C->diag = C->r = C->p = C->Ap = nullptr; for (int k = 0; k < 3; k++) C->wt[k] = C->tg[k] = nullptr; }
    }
    fprintf(stderr, "count-field: solve %.0f s\n", cf_now() - tst1);
    cf_lvl *L = &lv[0];
    if (opt_atoi("--cf-cut", cf_opt(argc, argv, "--cf-cut", "0")) && !spiral && strcmp(meth, "ray")) {   /* the layered min-cut on top of the solve (cf_cut_solve) */
        if (cf_cut_solve(L, Sfine, Gfine, opt_atoi("--cf-cut-d", cf_opt(argc, argv, "--cf-cut-d", "0")), opt_atoi("--cf-cut-b", cf_opt(argc, argv, "--cf-cut-b", "8")),
                         opt_atof("--cf-cut-q", cf_opt(argc, argv, "--cf-cut-q", "0.25")), opt_atof("--cf-cut-f", cf_opt(argc, argv, "--cf-cut-f", "1.25")),
                         opt_atof("--cf-cut-w", cf_opt(argc, argv, "--cf-cut-w", "4")), opt_atoi("--cf-cut-sub", cf_opt(argc, argv, "--cf-cut-sub", "2")))) { fprintf(stderr, "count-field: --cf-cut failed (memory)\n"); return 1; }
    }
    if (spiral) {   /* the villa spiral model fitted to the level-0 targets, anchored to U (src/nn_spiral.cu); U <- its U' */
        const double cn = (double)P * P * P, rmin = rmin_um / g.um; size_t nd = 0;
        for (size_t i = 0; i < L->n; i++) if (Sfine[i] > 0) nd++;
        sf_cfg sc; memset(&sc, 0, sizeof sc);
        int32_t *iz = malloc(nd * 4), *iy = malloc(nd * 4), *ix = malloc(nd * 4); float *q[7] = {0}; for (int k = 0; k < 6; k++) q[k] = malloc(nd * sizeof(float));
        if (!iz || !iy || !ix || !q[5]) { fprintf(stderr, "count-field: out of memory (spiral data)\n"); return 1; }
        size_t j = 0;
        for (int z = 0; z < L->d; z++) { const double pz = cf_center(L, z), cy = cf_axis_at(g.cy, na, pz), cx = cf_axis_at(g.cx, na, pz);
            for (int y = 0; y < L->hh; y++) for (int x = 0; x < L->w; x++) {
                const size_t i = ((size_t)z * L->hh + y) * L->w + x; if (!(Sfine[i] > 0)) continue;
                const double gz = Gfine[0][i] / Sfine[i], gy = Gfine[1][i] / Sfine[i], gx = Gfine[2][i] / Sfine[i], gl = sqrt(gz * gz + gy * gy + gx * gx);
                if (gl < 1e-6 || hypot(cf_center(L, y) - cy, cf_center(L, x) - cx) < rmin) continue;
                iz[j] = z; iy[j] = y; ix[j] = x; q[0][j] = (float)(gz / gl); q[1][j] = (float)(gy / gl); q[2][j] = (float)(gx / gl);
                q[3][j] = (float)log(gl / h); q[4][j] = (float)fmin(1, Sfine[i] / cn); q[5][j] = isfinite(L->U[i]) ? L->U[i] : -1e38f; j++;
            } }
        /* the anchor confidence per data cell (spiral as a corrector of the reference): released (-> 0) where the reference is
           self-inconsistent: the targets' closing error (cl >= 0.25 turn), the angle between the reference's grad W and the target
           normal (>= 30 deg), or its density along the target normal off by a factor >= 1.4; full (1) elsewhere */
        float *cl0 = malloc(L->n * sizeof(float)); { const int Qs = Qf / P < 2 ? 2 : Qf / P; cf_close CC = {L, Qs, cl0}; cf_par(L->d, cf_close_run, &CC); }
        const int awon = opt_atoi("--sf-release", cf_opt(argc, argv, "--sf-release", "1"));
        q[6] = malloc(nd * sizeof(float)); size_t nrel = 0; double arel = 0;
        for (size_t jj = 0; jj < j; jj++) {
            const int z = iz[jj], y = iy[jj], x = ix[jj]; const size_t i = ((size_t)z * L->hh + y) * L->w + x; const int c3[3] = {z, y, x}, dm[3] = {L->d, L->hh, L->w};
            const size_t st[3] = {(size_t)L->hh * L->w, (size_t)L->w, 1}; double gu[3], th[3]; cf_theta(L, cf_center(L, z), cf_center(L, y), cf_center(L, x), th);
            for (int a = 0; a < 3; a++) { const int lo = c3[a] > 0 ? c3[a] - 1 : c3[a], hi = c3[a] + 1 < dm[a] ? c3[a] + 1 : c3[a];
                gu[a] = hi > lo ? ((L->U[i + (hi - c3[a]) * (long)st[a]] - L->U[i - (c3[a] - lo) * (long)st[a]]) / (hi - lo) - th[a]) / (P * h) : 0; }   /* grad W, turns per native voxel */
            const double gl = sqrt(gu[0] * gu[0] + gu[1] * gu[1] + gu[2] * gu[2]), dn = gu[0] * q[0][jj] + gu[1] * q[1][jj] + gu[2] * q[2][jj];
            const double ang = gl > 1e-9 ? acos(fmin(1, fabs(dn) / gl)) * 180 / M_PI : 90, lr = fabs(log(fmax(fabs(dn), 1e-6)) - q[3][jj]);
            double rel = fmax(cf_smooth(cl0[i], 0.25, 0.75), fmax(cf_smooth(ang, 30, 60), cf_smooth(lr, log(1.4), log(2.0))));
            if (!awon) rel = 0;
            q[6][jj] = (float)(1 - rel); arel += rel; nrel += rel > 0.5;
        }
        free(cl0); sc.aw = q[6];
        fprintf(stderr, "count-field: spiral anchor released (> 0.5) on %.3f of %zu data cells, mean release %.3f\n", j ? (double)nrel / j : 0, j, j ? arel / j : 0);
        sc.nd = j; sc.cz = iz; sc.cy_ = iy; sc.cx_ = ix; sc.gz = q[0]; sc.gy = q[1]; sc.gx = q[2]; sc.lrho = q[3]; sc.c = q[4]; sc.u0 = q[5];
        sc.d = L->d; sc.hh = L->hh; sc.w = L->w; sc.cs = P * h; sc.ay = g.cy; sc.ax = g.cx; sc.na = na; sc.dir = dir; sc.pitch = pitch;
        sc.nst = opt_atoi("--sf-stages", cf_opt(argc, argv, "--sf-stages", "2")); sc.nrk = opt_atoi("--sf-rk", cf_opt(argc, argv, "--sf-rk", "3")); sc.Lf = opt_atoi("--sf-lf", cf_opt(argc, argv, "--sf-lf", "48")); sc.sfac = opt_atoi("--sf-sfac", cf_opt(argc, argv, "--sf-sfac", "6"));
        sc.La = opt_atof("--sf-la", cf_opt(argc, argv, "--sf-la", "96")); sc.Lz = opt_atof("--sf-lz", cf_opt(argc, argv, "--sf-lz", "96")); sc.K = opt_atoi("--sf-K", cf_opt(argc, argv, "--sf-K", "0"));
        sc.steps = opt_atoi("--sf-steps", cf_opt(argc, argv, "--sf-steps", "3000")); sc.nsamp = opt_atoi("--sf-samp", cf_opt(argc, argv, "--sf-samp", "65536")); sc.warm = opt_atoi("--sf-warm", cf_opt(argc, argv, "--sf-warm", "600")); sc.fwd = opt_atoi("--sf-fd-fwd", cf_opt(argc, argv, "--sf-fd-fwd", "1")); sc.lr_dr = opt_atof("--sf-lr-pitch", cf_opt(argc, argv, "--sf-lr-pitch", "0.002"));
        sc.lr_c = opt_atof("--sf-lr-c", cf_opt(argc, argv, "--sf-lr-c", "0.5")); sc.lr_f = opt_atof("--sf-lr-f", cf_opt(argc, argv, "--sf-lr-f", "0.2")); sc.lr_g = opt_atof("--sf-lr-g", cf_opt(argc, argv, "--sf-lr-g", "0.02")); sc.lr_end = opt_atof("--sf-lr-end", cf_opt(argc, argv, "--sf-lr-end", "0.1"));
        sc.w_n = opt_atof("--sf-wn", cf_opt(argc, argv, "--sf-wn", "1")); sc.w_d = opt_atof("--sf-wd", cf_opt(argc, argv, "--sf-wd", "1")); sc.w_sd = opt_atof("--sf-wsd", cf_opt(argc, argv, "--sf-wsd", "0.1"));
        sc.w_a0 = opt_atof("--sf-wa0", cf_opt(argc, argv, "--sf-wa0", "1")); sc.w_a1 = opt_atof("--sf-wa1", cf_opt(argc, argv, "--sf-wa1", "1")); sc.dfloor = opt_atof("--sf-dfloor", cf_opt(argc, argv, "--sf-dfloor", "0.1")); sc.huber = opt_atof("--sf-huber", cf_opt(argc, argv, "--sf-huber", "0.5"));
        sc.eps = opt_atof("--sf-eps", cf_opt(argc, argv, "--sf-eps", "0")); if (sc.eps <= 0) sc.eps = 0.5 * P * h;
        sc.gmin = log(opt_atof("--sf-gap-min", cf_opt(argc, argv, "--sf-gap-min", "0.35"))); sc.gmax = log(opt_atof("--sf-gap-max", cf_opt(argc, argv, "--sf-gap-max", "4")));
        sc.seed = (unsigned)opt_atoi("--sf-seed", cf_opt(argc, argv, "--sf-seed", "1")); sc.gpu = opt_atoi("--gpu", cf_opt(argc, argv, "--gpu", "0")); sc.verbose = opt_atoi("--sf-verbose", cf_opt(argc, argv, "--sf-verbose", "100"));
        /* --sf-pin-codes ROOT --sf-pin-key KEY --sf-pin-rows A:B[,C:D...] (field rows of the box): ground-truth pins (dev experiments:
           villa's human pins stand-in). Labelled voxels of the band codes (winding mod 14, 18 steps a turn) in those rows, kept with
           probability --sf-pin-p, become points of known W: the mod-14 code lifted to the reference's labelling (one gauge = the mode of
           W_ref - q, then the 14-turn period nearest W_ref). Never pin rows that are scored. */
        float *pin[4] = {0}; size_t npin = 0, cpin = 0;
        if (*cf_opt(argc, argv, "--sf-pin-codes", "") && imod) {
            z3 *zc = z3_open(store_open(cf_opt(argc, argv, "--sf-pin-codes", "")), cf_opt(argc, argv, "--sf-pin-key", "9.6"), nullptr);
            if (!zc) { fprintf(stderr, "count-field: %s\n", z3_error()); return 1; }
            const double lcf = opt_atof("--sf-pin-key", cf_opt(argc, argv, "--sf-pin-key", "9.6")) / g.um, pp = opt_atof("--sf-pin-p", cf_opt(argc, argv, "--sf-pin-p", "0.001"));
            const double oz = (double)(g.origin[0] - imod->origin[0]), oy = (double)(g.origin[1] - imod->origin[1]), ox = (double)(g.origin[2] - imod->origin[2]);
            char rows[512]; snprintf(rows, sizeof rows, "%s", cf_opt(argc, argv, "--sf-pin-rows", "")); unsigned rs = 12345;
            for (char *tok = strtok(rows, ","); tok; tok = strtok(nullptr, ",")) {
                int ra, rb; if (sscanf(tok, "%d:%d", &ra, &rb) != 2) { fprintf(stderr, "count-field: --sf-pin-rows A:B,...\n"); return 2; }
                const int64_t la = (int64_t)floor((g.origin[0] + ra * h) / lcf), lb = (int64_t)floor((g.origin[0] + rb * h) / lcf);
                const int64_t ly0 = (int64_t)floor(g.origin[1] / lcf), lx0 = (int64_t)floor(g.origin[2] / lcf), lny = (int64_t)(H * h / lcf), lnx = (int64_t)(W * h / lcf);
                uint8_t *cb = malloc((size_t)8 * lny * lnx); if (!cb) return 1;
                for (int64_t z0 = la; z0 < lb; z0 += 8) {
                    const int64_t nz = z0 + 8 < lb ? 8 : lb - z0, o3[3] = {z0, ly0, lx0}, n3[3] = {nz, lny, lnx};
                    if (z3_read(zc, o3, n3, cb, 16)) { fprintf(stderr, "count-field: %s\n", z3_error()); return 1; }
                    for (size_t i = 0; i < (size_t)nz * lny * lnx; i++) {
                        const uint8_t cd = cb[i]; if (cd < 1 || cd > 252) continue;
                        rs = rs * 1664525u + 1013904223u; if ((rs >> 8) / 16777216.0 >= pp) continue;
                        if (npin == cpin) { cpin = cpin ? 2 * cpin : 1 << 16; for (int k = 0; k < 4; k++) pin[k] = realloc(pin[k], cpin * sizeof(float)); }
                        const int64_t a = z0 + (int64_t)(i / (lny * lnx)), b = ly0 + (int64_t)(i / lnx % lny), e = lx0 + (int64_t)(i % lnx);
                        pin[0][npin] = (float)((a + 0.5) * lcf - g.origin[0]); pin[1][npin] = (float)((b + 0.5) * lcf - g.origin[1]); pin[2][npin] = (float)((e + 0.5) * lcf - g.origin[2]);
                        pin[3][npin] = (cd - 1) / 18.0f; npin++;
                    }
                }
                free(cb);
            }
            z3_close(zc);
            /* the gauge, then each pin's U' target in count-field's convention (W + dir theta / 2 pi) */
            enum { NG = 14 * 36 }; size_t gh[NG] = {0}; double *wr = malloc((npin + 1) * sizeof(double));
            for (size_t k = 0; k < npin; k++) { wr[k] = fit_W(imod, pin[0][k] + oz, pin[1][k] + oy, pin[2][k] + ox); double d = fmod(wr[k] - pin[3][k], 14.0); if (d < 0) d += 14; gh[(int)(d / 14 * NG) % NG]++; }
            int gb = 0; for (int k = 1; k < NG; k++) if (gh[k] > gh[gb]) gb = k; const double gauge = (gb + 0.5) * 14.0 / NG; size_t off1 = 0;
            for (size_t k = 0; k < npin; k++) {
                const double t = pin[3][k] + gauge, Wp = t + 14 * round((wr[k] - t) / 14); off1 += fabs(Wp - wr[k]) >= 0.5;
                const double cy = cf_axis_at(g.cy, na, pin[0][k]), cx = cf_axis_at(g.cx, na, pin[0][k]);
                pin[3][k] = (float)(Wp + dir * atan2(pin[1][k] - cy, pin[2][k] - cx) / (2 * M_PI));
            }
            free(wr);
            fprintf(stderr, "count-field: %zu ground-truth pins (gauge %.3f), the reference off >= 0.5 turn at %.3f of them\n", npin, gauge, npin ? (double)off1 / npin : 0);
            sc.npin = npin; sc.pz = pin[0]; sc.py = pin[1]; sc.px = pin[2]; sc.pu = pin[3]; sc.w_pin = opt_atof("--sf-pin-w", cf_opt(argc, argv, "--sf-pin-w", "5"));
        }
        sc.uout = malloc(L->n * sizeof(float)); if (!sc.uout) { fprintf(stderr, "count-field: out of memory (spiral U)\n"); return 1; }
        if (nn_spiral_fit(&sc)) { fprintf(stderr, "count-field: the spiral fit failed\n"); return 1; }
        const int blend = opt_atoi("--sf-blend", cf_opt(argc, argv, "--sf-blend", "1")), bR = opt_atoi("--sf-blend-R", cf_opt(argc, argv, "--sf-blend-R", "2"));
        if (blend && q[6]) {   /* the output: the reference where it is trusted, the spiral where it was released (anchor confidence, box-smoothed) */
            float *a = malloc(L->n * sizeof(float)); for (size_t i = 0; i < L->n; i++) a[i] = 1;
            for (size_t jj = 0; jj < j; jj++) a[((size_t)iz[jj] * L->hh + iy[jj]) * L->w + ix[jj]] = q[6][jj];
            float *one = malloc(L->n * sizeof(float)); for (size_t i = 0; i < L->n; i++) one[i] = 1;
            cf_boxsmooth(a, L->d, L->hh, L->w, bR); cf_boxsmooth(one, L->d, L->hh, L->w, bR); double ms = 0;
            for (size_t i = 0; i < L->n; i++) { const double t = fmin(1, fmax(0, a[i] / one[i])); ms += 1 - t; if (isfinite(L->U[i]) && isfinite(sc.uout[i])) sc.uout[i] = (float)(t * L->U[i] + (1 - t) * sc.uout[i]); }
            fprintf(stderr, "count-field: spiral blend: mean spiral share %.3f of the solve cells\n", ms / L->n); free(a); free(one);
        }
        free(L->U); L->U = sc.uout; free(iz); free(iy); free(ix); for (int k = 0; k < 7; k++) free(q[k]); for (int k = 0; k < 4; k++) free(pin[k]);
    }
    free(L->b); free(L->diag); free(L->r); free(L->p); free(L->Ap); L->b = L->diag = L->r = L->p = L->Ap = nullptr;
    /* the closing error on the solve grid */
    const int Q = Qf / P < 2 ? 2 : Qf / P;   /* the closing loops: --loop field cells, >= 2 solve cells */
    float *cl = malloc(L->n * sizeof(float)); { cf_close CC = {L, Q, cl}; cf_par(L->d, cf_close_run, &CC); }
    { size_t nc = 0, nb = 0; double sc = 0; for (size_t i = 0; i < L->n; i++) if (Sfine[i] > 0) { nc++; sc += cl[i]; nb += cl[i] > 0.25; }
      fprintf(stderr, "count-field: closing error over %dx%d-solve-cell loops: mean %.3f turns, > 0.25 turn on %.3f of %zu data cells (the targets' non-integrability: "
                     "not used by the solve; it only down-weights the phase gauge, exp(-cl^2 / 0.125); not a quality measure: P4 1.23 at GT ids 0.18, 1667 2.06 at 0.84)\n", Q, Q, nc ? sc / nc : 0, nc ? (double)nb / nc : 0, nc); }
    if (getenv("UFSM_CF_DEBUG")) {   /* closing error, density and |n . r| by radius (1 mm rings) */
        enum { NRB = 40 }; double cn[NRB] = {0}, cb[NRB] = {0}, cr[NRB] = {0}, cs[NRB] = {0}, cg[NRB] = {0};
        for (int z = 0; z < L->d; z++) { const double pz = cf_center(L, z), cy = cf_axis_at(g.cy, na, pz), cx = cf_axis_at(g.cx, na, pz);
            for (int y = 0; y < L->hh; y++) for (int x = 0; x < L->w; x++) {
                const size_t i = ((size_t)z * L->hh + y) * L->w + x; if (!(Sfine[i] > 0)) continue;
                const double ry = cf_center(L, y) - cy, rx = cf_center(L, x) - cx, r = hypot(ry, rx); const int k = (int)fmin(NRB - 1, r * g.um / 1000);
                cn[k]++; cb[k] += cl[i] > 0.25; cs[k] += Sfine[i]; cr[k] += (Gfine[1][i] * ry + Gfine[2][i] * rx) / (r > 1 ? r : 1) / Sfine[i]; cg[k] += sqrt(Gfine[0][i] * Gfine[0][i] + Gfine[1][i] * Gfine[1][i] + Gfine[2][i] * Gfine[2][i]) / Sfine[i]; } }
        for (int k = 0; k < NRB; k++) if (cn[k] > 0) fprintf(stderr, "count-field debug: r %2d mm: %8.0f cells, closing > 0.25 %.3f, mean weight %.3f, radial density %.3f, |g| %.3f turns per field cell\n", k, cn[k], cb[k] / cn[k], cs[k] / cn[k] / ((double)P * P * P), cr[k] / cn[k], cg[k] / cn[k]);
    }
    /* (the level-0 weights / targets and the fine gradient sums are not read past the closing error: 9 floats a solve cell freed before
       the gauge / model passes, whose field pieces peak there; PHerc Paris 4's ref, 246 M solve cells, was OOM-killed at 29.2 GB RSS in a
       31 GB pod, 2026-11-18) */
    for (int k = 0; k < 3; k++) { free(L->wt[k]); free(L->tg[k]); L->wt[k] = L->tg[k] = nullptr; free(Gfine[k]); Gfine[k] = Gl[0][k] = nullptr; }
    cf_dm M = {nullptr, 0, L, D, H, W, R, na, dir, h, 0, g.cy, g.cx, nullptr, nullptr, nullptr, nullptr, 0, nullptr}; double tst2 = cf_now();
    if (mo) fit_fields_skip_planes(1 | 2 | 4);   /* (--model-only from here: pc / ps / sp / w read only) */
    /* --model-only R 0 on the GPU: the global gauge pass and the model pass in one read of the strided chunks. The phasor sums gr / gi
       are taken against W without the global gauge g and rotated by e^{-2 pi i g} afterwards (the CPU's two passes: against W + g in
       fp32; the same sums up to fp32 rounding) */
    float *ggr = nullptr, *ggi = nullptr;
    int gmo = gpu && mo && !R && strcmp(cf_opt(argc, argv, "--gauge", "demod"), "ls") && !opt_atoi("--sp-w", cf_opt(argc, argv, "--sp-w", "0"));
    if (gmo) {
        ggr = calloc(L->n, sizeof(float)); ggi = calloc(L->n, sizeof(float)); if (!ggr || !ggi) { fprintf(stderr, "count-field: out of memory (gauge sums)\n"); return 1; }
        const cf_gpu_gscfg gc = {D, H, W, P, L->d, L->hh, L->w, na, dir, h, g.cy, g.cx, L->U};
        double acc[2] = {0, 0}; int ok = !cf_gpu_gsum_begin(&gc), z0 = 0; cf_pre pr2 = {0};
        /* in shard pieces too (as pass 1; UFSM_CF_PIECE_ROWS) where a piece's solve cells are a multiple of the kernel's 128-cell blocks:
           then every block sums the same cells and acc takes the same block partials in the same order, the whole-chunk result bit for
           bit (1667: 16 solve rows x 196 x 202 = 4949 blocks). Else whole chunks */
        const int R2 = RP > 0 && RP < CZ && (size_t)(RP / P) * lv[0].hh * lv[0].w % 128 == 0 ? RP : CZ;
        const int pref2 = R2 < CZ ? 2 * 5.0 * R2 * HW < 0.5 * (double)ufsm_mem_available() : pref;
        while (z0 < D && (z0 / CZ) % mstride) z0 += CZ;
        if (ok && z0 < D) cf_pre_start(&pr2, in, z0, z0 + R2 < D ? z0 + R2 : D, pref2);
        while (ok && z0 < D) {
            const int zc1 = z0 / CZ * CZ + CZ < D ? z0 / CZ * CZ + CZ : D, z1 = z0 + R2 < zc1 ? z0 + R2 : zc1; fit_fields_buf b; if (cf_pre_take(&pr2, &b)) return 1;
            int zn = z1; if (zn == zc1) while (zn < D && (zn / CZ) % mstride) zn += CZ;
            if (zn < D) { const int zne = zn / CZ * CZ + CZ < D ? zn / CZ * CZ + CZ : D; cf_pre_start(&pr2, in, zn, zn + R2 < zne ? zn + R2 : zne, pref2); }
            ok = !cf_gpu_gsum_piece(b.f.pc, b.f.ps, b.f.w, z0, z1, z0 / CZ * CZ, ggr, ggi, acc);
            fit_fields_free(&b); z0 = zn;
        }
        if (pr2.busy) { fit_fields_buf b; if (!cf_pre_take(&pr2, &b)) fit_fields_free(&b); }
        cf_gpu_gsum_end();
        if (!ok) { fprintf(stderr, "count-field: GPU model pass failed: the CPU passes\n"); free(ggr); free(ggi); ggr = ggi = nullptr; gmo = 0; gpu = 0; }
        else {
            M.gauge = atan2(acc[1], acc[0]) / (2 * M_PI);
            const double cr = cos(2 * M_PI * M.gauge), sr = sin(2 * M_PI * M.gauge);
            for (size_t i = 0; i < L->n; i++) { const double a = ggr[i], b = ggi[i]; ggr[i] = (float)(a * cr + b * sr); ggi[i] = (float)(b * cr - a * sr); }
            fprintf(stderr, "count-field: global gauge + model pass (GPU) %.0f s, waited for field reads %.0f s\n", cf_now() - tst2, pr2.t); tst2 = cf_now();
        }
    }
    if (!R && !gmo) {   /* R 0: the global gauge arg sum c e^{2 pi i (phi - W)} (with R > 0 the demodulation absorbs any constant) */
        double complex acc = 0;
        for (int z0 = 0; z0 < D; z0 += CZ) {
            if (mo && (z0 / CZ) % mstride) continue;   /* (--model-only: every mstride-th chunk) */
            const int z1 = z0 + CZ < D ? z0 + CZ : D; fit_fields_buf b; if (fit_fields_load(in, z0, z1, 1, &b)) return 1;
            for (size_t i = 0; i < (size_t)(z1 - z0) * HW; i += 7) {
                if (!b.f.w[i]) continue;
                const double pcv = (b.f.pc[i] - 128) / 127.0, psv = (b.f.ps[i] - 128) / 127.0, c = hypot(pcv, psv); if (c < 1e-3) continue;
                acc += c * cexp(I * (atan2(psv, pcv) - 2 * M_PI * cf_Wcell(&M, z0 + (int)(i / HW), (int)(i / W % H), (int)(i % W))));
            }
            fit_fields_free(&b);
        }
        M.gauge = carg(acc) / (2 * M_PI);
        fprintf(stderr, "count-field: global gauge pass %.0f s\n", cf_now() - tst2); tst2 = cf_now();
    }
    float *gls = nullptr;
    if (!strcmp(cf_opt(argc, argv, "--gauge", "demod"), "ls")) {   /* the smooth gauge (cf_gauge_solve) instead of the demodulation */
        float *gr2 = calloc(L->n, sizeof(float)), *gi2 = calloc(L->n, sizeof(float)); if (!gr2 || !gi2) { fprintf(stderr, "count-field: out of memory (gauge)\n"); return 1; }
        for (int z0 = 0; z0 < D; z0 += CZ) {
            const int z1 = z0 + CZ < D ? z0 + CZ : D; fit_fields_buf b; if (fit_fields_load(in, z0, z1, 1, &b)) { fprintf(stderr, "count-field: cannot load rows %d..%d of %s\n", z0, z1, in); return 1; }
            M.f = &b.f; M.zoff = z0; cf_gsum GS = {&M, gr2, gi2};
            for (int z = z0; z < z1; z++) { M.z = z; cf_par(L->hh, cf_gsum_run, &GS); }
            fit_fields_free(&b);
        }
        gls = cf_gauge_solve(L, gr2, gi2, opt_atoi("--gauge-R", cf_opt(argc, argv, "--gauge-R", "5")), iters); free(gr2); free(gi2);
        if (!gls) { fprintf(stderr, "count-field: the gauge solve failed\n"); return 1; }
        M.gls = gls;
    }
    /* pass 2: the output, chunk by chunk. raw: rows appended to raw planes (every plane written); zarr (input zarr arrays of the same
       box, no zarr_offset): pc / ps / cl as new zarr arrays (128^3 shards, lossless), the other planes symlinked to the input's */
    char p[1400]; snprintf(p, sizeof p, "mkdir -p '%s'", out); if (!mo && system(p)) { fprintf(stderr, "count-field: cannot create %s\n", out); return 1; }
    static const char *names[15] = {"pc", "ps", "t0", "t1", "t2", "t3", "t4", "t5", "sp", "w", "dz", "dy", "dx", "dq", "r"};
    int zarr = 0; long long ish[3] = {0, 0, 0}, ioff[3] = {0, 0, 0}, irow[2] = {0, 0}; const int into = *cf_opt(argc, argv, "--into-shape", "") != 0;
    if (into) {   /* --into-shape D,H,W --into-off Z,Y,X [--into-rows A,B]: write rows [A, B) of whole-volume pc / ps / cl arrays (global 128^3 shards) */
        if (sscanf(cf_opt(argc, argv, "--into-shape", ""), "%lld,%lld,%lld", &ish[0], &ish[1], &ish[2]) != 3 || sscanf(cf_opt(argc, argv, "--into-off", ""), "%lld,%lld,%lld", &ioff[0], &ioff[1], &ioff[2]) != 3) { fprintf(stderr, "count-field: --into-shape D,H,W and --into-off Z,Y,X\n"); return 2; }
        irow[0] = ioff[0]; irow[1] = ioff[0] + D;
        if (*cf_opt(argc, argv, "--into-rows", "") && sscanf(cf_opt(argc, argv, "--into-rows", ""), "%lld,%lld", &irow[0], &irow[1]) != 2) { fprintf(stderr, "count-field: --into-rows A,B\n"); return 2; }
        if (ioff[1] % 128 || ioff[2] % 128 || irow[0] % 128 || (irow[1] % 128 && irow[1] != ish[0]) || irow[0] < ioff[0] || irow[1] > ioff[0] + D || irow[0] >= irow[1] ||
            ioff[1] + H > ish[1] || ioff[2] + W > ish[2] || *mdir) { fprintf(stderr, "count-field: --into: y / x offsets and the rows must be 128-aligned (B may be the volume end), inside the box and the volume; no --model\n"); return 2; }
    }
    {   const char *fmt = cf_opt(argc, argv, "--format", "auto");
        snprintf(p, sizeof p, "%s/pc/zarr.json", in); FILE *t = fopen(p, "rb"); const int inz = t != nullptr; if (t) fclose(t);
        snprintf(p, sizeof p, "%s/fields.json", in); t = fopen(p, "rb"); int off = 0;
        if (t) { char buf[4096]; const size_t r = fread(buf, 1, sizeof buf - 1, t); buf[r] = 0; off = strstr(buf, "zarr_offset") != nullptr; fclose(t); }
        zarr = into || !strcmp(fmt, "zarr") || (!strcmp(fmt, "auto") && inz && !off);
        if (zarr && (!inz || (off && !into))) { fprintf(stderr, "count-field: --format zarr needs zarr input arrays of the same box (no zarr_offset)\n"); return 2; }
        if (zarr && wwc) fprintf(stderr, "count-field: zarr output: no wc.f32 (raw only)\n");
        if (mo) zarr = 0;
    }
    const int doc = zarr || mo ? 0 : wwc;   /* wc.f32 with raw output only */
    const int BZ = zarr ? 128 : CZ; if (zarr && CZ > BZ) CZ = BZ;
    FILE *fp[15] = {0}, *fwc = nullptr, *fcl = nullptr; z3w *zw[4] = {0};
    const int spw = opt_atoi("--sp-w", cf_opt(argc, argv, "--sp-w", "0")), nzp = spw ? 4 : 3;   /* --sp-w 1: sp from |grad W| (the density term agrees with the phase) */
    if (zarr) {
        for (int k = 2; k < 15; k++) {
            if ((k >= 10 && k < 14 && !w4d) || (k == 14 && !recto) || (k == 8 && spw)) continue;
            char src[PATH_MAX], dst[1400]; snprintf(dst, sizeof dst, "%s/%s", in, names[k]); if (!realpath(dst, src)) { fprintf(stderr, "count-field: %s\n", dst); return 1; }   /* the base array (through crop links) */
            snprintf(dst, sizeof dst, "%s/%s", out, names[k]); struct stat sst; if (into && !lstat(dst, &sst)) continue;
            unlink(dst); if (symlink(src, dst)) { fprintf(stderr, "count-field: cannot link %s\n", dst); return 1; }
        }
        const int64_t zsh[3] = {into ? ish[0] : D, into ? ish[1] : H, into ? ish[2] : W}; static const char *zn[4] = {"pc", "ps", "cl", "sp"}; const int zf[4] = {128, 128, 0, 0};
        for (int k = 0; k < nzp; k++) { snprintf(p, sizeof p, "%s/%s", out, zn[k]); zw[k] = z3w_create(p, zsh, 128, 0, zf[k], "{\"ufsm\": {\"content\": \"ufsm count-field\"}}");
            if (!zw[k]) { fprintf(stderr, "count-field: %s\n", z3w_error()); return 1; } }
    } else if (!mo) {
        for (int k = 0; k < 15; k++) {
            if ((k >= 10 && k < 14 && !w4d) || (k == 14 && !recto)) continue;
            snprintf(p, sizeof p, "%s/%s.raw", out, names[k]); if (!(fp[k] = fopen(p, "wb"))) { fprintf(stderr, "count-field: cannot write %s\n", p); return 1; }
        }
        snprintf(p, sizeof p, "%s/cl.raw", out); fcl = fopen(p, "wb");
        if (doc) { snprintf(p, sizeof p, "%s/wc.f32", out); fwc = fopen(p, "wb"); }
        if (!fcl || (doc && !fwc)) { fprintf(stderr, "count-field: cannot write %s\n", out); return 1; }
    }
    uint8_t *opc = malloc((size_t)BZ * HW), *ops = malloc((size_t)BZ * HW), *ocl = malloc((size_t)BZ * HW); float *wc = doc ? malloc((size_t)BZ * HW * sizeof(float)) : nullptr; uint8_t *osp = spw ? malloc((size_t)BZ * HW) : nullptr;
    float *gW = spw ? cf_gradW(L, g.cy, g.cx, na) : nullptr;
    const int nring = 2 * Rz + 1; float *ring = malloc((size_t)nring * 3 * HW * sizeof(float)), *Wring = malloc((size_t)nring * HW * sizeof(float));
    float *gr = gmo ? ggr : *mdir ? calloc(L->n, sizeof(float)) : nullptr, *gi = gmo ? ggi : *mdir ? calloc(L->n, sizeof(float)) : nullptr;
    if (!opc || !ops || !ocl || (doc && !wc) || !ring || !Wring || (*mdir && (!gr || !gi))) { fprintf(stderr, "count-field: out of memory\n"); return 1; }
    cf_out O = {nullptr, 0, 0, L, ring, nullptr, Sfine, cl, 0, D, H, W, Rz, P, !strcmp(cmode, "w") ? 0 : !strcmp(cmode, "phase") ? 1 : !strcmp(cmode, "pmax") ? 3 : 2, R > 0 ? (gls ? 2 : 1) : 0, opt_atoi("--fine", cf_opt(argc, argv, "--fine", "0")), opt_atoi("--trust-f", cf_opt(argc, argv, "--trust-f", "0")), opt_atof("--keep", cf_opt(argc, argv, "--keep", "0")), opt_atof("--conf-w", cf_opt(argc, argv, "--conf-w", "1")), tr0, tr1, opc, ops, ocl, osp, gW, wc, gr, gi, {{0}}};
    double resid[2][2] = {{0, 0}, {0, 0}};   /* UFSM_CF_RESID: plaquettes, residues (input, output) */
    const int olo = into ? (int)(irow[0] - ioff[0]) : 0, ohi = into ? (int)(irow[1] - ioff[0]) : D;   /* the box rows written */
    for (int b0 = olo; b0 < ohi && !gmo; b0 += BZ) {   /* (gmo: done above) */
        if (mo && ((b0 - olo) / BZ) % mstride) continue;   /* (--model-only: the model gauge from every mstride-th block) */
        const int b1 = b0 + BZ < ohi ? b0 + BZ : ohi; O.c0 = b0;
        if (zarr) { memset(opc, 128, (size_t)BZ * HW); memset(ops, 128, (size_t)BZ * HW); memset(ocl, 0, (size_t)BZ * HW); }
        for (int c0 = b0; c0 < b1; c0 += CZ) {
            const int c1 = c0 + CZ < b1 ? c0 + CZ : b1, a = c0 - Rz > 0 ? c0 - Rz : 0, e = c1 + Rz < D ? c1 + Rz : D;
            fit_fields_buf b; if (fit_fields_load(in, a, e, 1, &b)) { fprintf(stderr, "count-field: cannot load rows %d..%d of %s\n", a, e, in); return 1; }
            M.f = &b.f; M.zoff = a; O.f = &b.f; O.zoff = a;
            for (int zo = a - Rz; zo < c1; zo++) {
                const int zi = zo + Rz;   /* the plane entering the ring */
                if (zi >= a && zi < e) {
                    float *slot = ring + (size_t)(zi % nring) * 3 * HW; M.re = slot; M.im = slot + HW; M.cw = slot + 2 * HW; M.Wp = Wring + (size_t)(zi % nring) * HW; M.z = zi;
                    cf_par(H, cf_plane_run, &M);
                    if (R > 0) { cf_blr BL = {&M}; cf_par(W, cf_blur_rows, &BL); cf_par(H, cf_blur_cols, &BL); }
                }
                if (zo < c0) continue;
                O.zo = zo; O.Wz = Wring + (size_t)(zo % nring) * HW; cf_par(L->hh, cf_out_run, &O);
            }
            if (getenv("UFSM_CF_RESID")) {   /* phase residues on y-x plaquettes (all 4 corners conf >= 0.3): the input vs the output phase */
                for (int zo = c0; zo < c1; zo++) for (int y = 0; y + 1 < H; y++) for (int x = 0; x + 1 < W; x++) {
                    const size_t q[4] = {(size_t)y * W + x, (size_t)y * W + x + 1, (size_t)(y + 1) * W + x + 1, (size_t)(y + 1) * W + x};
                    for (int io = 0; io < 2; io++) {
                        const uint8_t *pc = io ? opc + (size_t)(zo - b0) * HW : b.f.pc + (size_t)(zo - a) * HW, *ps = io ? ops + (size_t)(zo - b0) * HW : b.f.ps + (size_t)(zo - a) * HW;
                        double ph[4]; int ok = 1;
                        for (int k = 0; k < 4 && ok; k++) { const double cc = (pc[q[k]] - 128) / 127.0, ss = (ps[q[k]] - 128) / 127.0; if (hypot(cc, ss) < 0.3) ok = 0; ph[k] = atan2(ss, cc) / (2 * M_PI); }
                        if (!ok) continue;
                        double s = 0; for (int k = 0; k < 4; k++) { double d = ph[(k + 1) % 4] - ph[k]; d -= round(d); s += d; }
                        resid[io][0]++; resid[io][1] += fabs(s) > 0.5;
                    }
                }
            }
            if (!zarr && !mo) {   /* raw: one chunk = one block */
                const size_t nr = (size_t)(c1 - c0) * HW, off = (size_t)(c0 - a) * HW, bo = (size_t)(c0 - b0) * HW;
                int bad = cf_wr(fp[0], opc + bo, 1, nr) || cf_wr(fp[1], ops + bo, 1, nr) || cf_wr(fcl, ocl + bo, 1, nr) || (doc && cf_wr(fwc, wc + bo, sizeof(float), nr));
                for (int k = 2; k < 15 && !bad; k++) {
                    if (!fp[k]) continue;
                    const uint8_t *src = k == 8 && spw ? osp + bo - off : k < 10 ? b.a[k] : k < 14 ? b.f.wd[k - 10] : b.f.recto;
                    bad = !src || cf_wr(fp[k], src + off, 1, nr);
                }
                if (bad) { fprintf(stderr, "count-field: cannot write the rows %d..%d of %s\n", c0, c1, out); return 1; }
            }
            fit_fields_free(&b);
        }
        if (zarr) {   /* the block's shards: (plane, shard y, shard x) jobs on threads */
            cf_shards SJ = {zw, {opc, ops, ocl, osp}, (int)((b0 + ioff[0]) / 128), b1 - b0, H, W, (H + 127) / 128, (W + 127) / 128, (int)(ioff[1] / 128), (int)(ioff[2] / 128), 0};
            cf_par((long)nzp * SJ.ny * SJ.nx, cf_shard_run, &SJ);
            if (SJ.fail) { fprintf(stderr, "count-field: %s\n", z3w_error()); return 1; }
        }
    }
    fprintf(stderr, "count-field: output pass %.0f s\n", cf_now() - tst2);
    if (zarr) { for (int k = 0; k < nzp; k++) if (z3w_close(zw[k])) { fprintf(stderr, "count-field: %s\n", z3w_error()); return 1; } }
    else if (!mo) {
        for (int k = 0; k < 15; k++) if (fp[k] && fclose(fp[k])) { fprintf(stderr, "count-field: write error %s\n", out); return 1; }
        if (fclose(fcl) || (fwc && fclose(fwc))) { fprintf(stderr, "count-field: write error %s\n", out); return 1; }
    }
    if (getenv("UFSM_CF_RESID")) fprintf(stderr, "count-field: phase residues per confident y-x plaquette: input %.5f (%.0f), output %.5f (%.0f)\n", resid[0][0] > 0 ? resid[0][1] / resid[0][0] : 0, resid[0][0], resid[1][0] > 0 ? resid[1][1] / resid[1][0] : 0, resid[1][0]);
    double agree_s = 0, agree_n = 0; for (int t = 0; t < CF_MAXT; t++) { agree_s += O.ag[t][0]; agree_n += O.ag[t][1]; }
    if (agree_n > 0) fprintf(stderr, "count-field: gauge %.3f turn; phase / W agreement (phasor mean over +-%d cells, confidence-weighted) %.3f\n", M.gauge, R, agree_s / agree_n);
    else fprintf(stderr, "count-field: gauge %.3f turn; phase / W agreement: n/a (not measured with --R %d; ufsm cf-demod MODEL FIELDS measures it per z band)\n", M.gauge, R);
    if (into) {   /* the whole volume's fields.json: the base field set's (the directory of the input's pc array), written once */
        struct stat sst; snprintf(p, sizeof p, "%s/fields.json", out);
        if (stat(p, &sst)) {
            char src[PATH_MAX], q[1400]; snprintf(q, sizeof q, "%s/pc", in); if (!realpath(q, src)) { fprintf(stderr, "count-field: %s\n", q); return 1; }
            char *sl = strrchr(src, '/'); if (sl) *sl = 0;
            snprintf(q, sizeof q, "cp '%s/fields.json' '%s.tmp%d' && mv '%s.tmp%d' '%s'", src, p, (int)getpid(), p, (int)getpid(), p);
            if (system(q)) { fprintf(stderr, "count-field: cannot copy %s/fields.json\n", src); return 1; }
        }
    } else if (!mo) {   /* fields.json (fit_fields_save_ex's keys) + "wcount" */
        snprintf(p, sizeof p, "%s/fields.json", out); FILE *fj = fopen(p, "w"); if (!fj) return 1;
        fprintf(fj, "{%s\"origin\": [%lld, %lld, %lld], \"shape\": [%d, %d, %d], \"h\": %d, \"um\": %.9g, ", zarr ? "\"format\": \"zarr\", " : "", (long long)g.origin[0], (long long)g.origin[1], (long long)g.origin[2], D, H, W, g.h, g.um);
        if (w4d > 0) fprintf(fj, "\"w4_d\": %d, ", w4d);
        fprintf(fj, "\"wcount\": %s, \"recto\": %s, \"axis\": [", doc ? "true" : "false", recto ? "true" : "false");
        for (int z = 0; z < na; z++) fprintf(fj, "%s[%.3f, %.3f]", z ? ", " : "", g.cy[z], g.cx[z]);
        fprintf(fj, "]}\n"); if (fclose(fj)) return 1;
    }
    if (*mdir) {   /* W as a fit model on the solve lattice: U + the phase gauge smoothed over +-Rm solve cells */
        const double tlat = cf_now();
        {   /* the model coherence (audit-unroll 3a, the +-1 direction gate): |sum over solve cells of z| / sum |z|, z = sum c e^{2 pi i (phi - W)}
               per cell, before the smoothing. With the right hand phi - W is the gauge plus count errors: one phasor; with the wrong one it rotates
               twice per turn about the axis and the sum cancels. unroll.sh (DIRCOH=1) compares two refs, --dir +1 vs -1, on this line */
            double complex zs = 0; double za = 0; for (size_t i = 0; i < L->n; i++) { zs += gr[i] + I * gi[i]; za += hypot(gr[i], gi[i]); }
            fprintf(stderr, "count-field: model coherence %.4f (dir %+d; |sum z| / sum |z| over %zu solve cells)\n", za > 0 ? cabs(zs) / za : 0, dir, L->n);
        }
        if (mo && mstride > 1) cf_gauge_fill_z(gr, gi, L->d, (size_t)L->hh * L->w);
        if (Rm > 0) cf_boxsmooth2_par(gr, gi, L->d, L->hh, L->w, Rm);
        fit_model m; memset(&m, 0, sizeof m);
        m.dir = dir; m.nk = 2; m.knot_dz = kdz; m.pitch = kpitch; m.offset = koff; m.D = D; m.H = H; m.W = W; m.h = h; m.um = g.um; m.cy = g.cy; m.cx = g.cx;
        for (int d = 0; d < 3; d++) m.origin[d] = g.origin[d];
        m.L = (int)(P * h); m.lz = (int)(D * h / m.L) + 2; m.ly = (int)(H * h / m.L) + 2; m.lx = (int)(W * h / m.L) + 2;
        m.delta = malloc((size_t)m.lz * m.ly * m.lx * sizeof(float));
        { cf_lat LT = {&m, L, gr, gi, g.cy, g.cx, na, M.gauge, pitch}; cf_par(m.lz, cf_lat_run, &LT); }
        fprintf(stderr, "count-field: model lattice %.1f s\n", cf_now() - tlat);
        if (fit_save(&m, mdir)) { fprintf(stderr, "count-field: cannot write the model %s\n", mdir); return 1; }
        fprintf(stderr, "count-field: model (lattice %d native voxels, %dx%dx%d) -> %s\n", m.L, m.lz, m.ly, m.lx, mdir);
        free(m.delta);
    }
    printf("count-field: %s -> %s (method %s, pool %d, R %d / %d, conf %s)\n", in, out, meth, P, R, Rz, cmode);
    free(opc); free(ops); free(ocl); free(osp); free(gW); free(gls); free(wc); free(cl); free(ring); free(Wring); free(gr); free(gi); free(Sfine); for (int k = 0; k < 3; k++) free(Gfine[k]);
    for (int l = 0; l < nl; l++) { cf_lvl *C = &lv[l]; free(C->b); free(C->diag); free(C->U); free(C->r); free(C->p); free(C->Ap); for (int k = 0; k < 3; k++) { free(C->wt[k]); free(C->tg[k]); } }
    free(lv); free(Sl); free(Gl); fit_fields_free(&g);
    return 0;
}

/* ufsm spiral-fit IN --model OUT [--init-model REF] [count-field / --sf-* options]: the villa spiral model as the coarse reference
   = count-field IN - --method spiral --R 0 --pool 8 --phase-w 4 --model-only 1 (the caller's options first: they win) */
int cmd_spiral_fit(int argc, char **argv) {
    if (argc < 3 || !*cf_opt(argc, argv, "--model", "")) {
        fprintf(stderr, "usage: ufsm spiral-fit <in-fields> --model DIR [--init-model REF (anchor: the count-field ref; else its LS U)] [--gpu 0]\n"
                        "       [--sf-steps 3000] [--sf-samp 65536] [--sf-warm 300 (anchor-only steps)] [--sf-wn 1] [--sf-wd 1] [--sf-wsd 0.1] [--sf-wa0 1] [--sf-wa1 0.05]\n"
                        "       [--sf-lf 48 (fine B-spline spacing, native)] [--sf-sfac 6] [--sf-stages 2] [--sf-rk 3] [--sf-la 96] [--sf-lz 96 (gap bins)] [--sf-lr-c 0.5] [--sf-lr-f 0.2] [--sf-lr-g 0.02]\n"
                        "       [--sf-gap-min 0.35] [--sf-gap-max 4] [--sf-eps (cell / 2)] [--sf-verbose 100] + count-field's (--pool, --phase-w, --model-stride, --threads, --mem-gb, ...)\n");
        return 2;
    }
    static const char *def[] = {"--method", "spiral", "--R", "0", "--pool", "8", "--phase-w", "4", "--model-only", "1", "--iters", "1200"};
    const int nd = (int)(sizeof def / sizeof *def), n = argc - 1 + 1 + nd; char **av = calloc(n + 1, sizeof *av);
    av[0] = argv[0]; av[1] = "count-field"; av[2] = argv[2]; av[3] = "/nonexistent-spiral-fit-out"; int k = 4;
    for (int i = 3; i < argc; i++) av[k++] = argv[i];
    for (int i = 0; i < nd; i++) av[k++] = (char *)def[i];
    const int r = cmd_count_field(k, av); free(av); return r;
}

/* ufsm cf-compare MODEL-A MODEL-B [FIELDS] [--step 4] [--threads 8]: two count-field models (the cf ref) of the same box. Lattice:
   max / p99.9 |delta_A - delta_B|. With FIELDS (the box's fit fields): at every step-th field cell (z, y, x) with material (w > 0) and
   a phase (|pc, ps| >= 0.05), the snap integer of the fine stage (fit --ref-snap 1: round(W - phi), phi the network phase in turns)
   under both models, their agreement, and the |W_A - W_B| quantiles */
typedef struct { const fit_model *a, *b; const fit_fields *f; int zoff, z0, z1, step; double h; size_t n[CF_MAXT], same[CF_MAXT], koff[CF_MAXT][17]; uint64_t *hist[CF_MAXT]; } cf_cmp;   /* koff: snap integer differences -8..8 (the gauge-free agreement) */
enum { CF_NH = 4096 };   /* |dW| histogram: bins of 1e-6 turn, log-spaced (bin k = 1e-6 * 10^(k / 512)) */
static int cf_hbin(double d) { if (!(d > 1e-6)) return 0; const int k = (int)(512 * log10(d / 1e-6)) + 1; return k >= CF_NH ? CF_NH - 1 : k; }
static void cf_cmp_run(void *a_, long r0, long r1, int t) {
    cf_cmp *C = a_; const fit_fields *f = C->f; const size_t HW = (size_t)f->H * f->W;
    for (long r = r0; r < r1; r++) { const int z = C->z0 + (int)r * C->step; if (z >= C->z1) break;
        for (int y = 0; y < f->H; y += C->step) for (int x = 0; x < f->W; x += C->step) {
            const size_t i = (size_t)(z - C->zoff) * HW + (size_t)y * f->W + x; if (!f->w[i]) continue;
            const double pc = (f->pc[i] - 128) / 127.0, ps = (f->ps[i] - 128) / 127.0; if (hypot(pc, ps) < 0.05) continue;
            const double ph = atan2(ps, pc) / (2 * M_PI), pz = (z + 0.5) * C->h, py = (y + 0.5) * C->h, px = (x + 0.5) * C->h;
            const double wa = fit_W(C->a, pz, py, px), wb = fit_W(C->b, pz, py, px);
            C->n[t]++; C->same[t] += lround(wa - ph) == lround(wb - ph);
            { const long k = lround(wa - ph) - lround(wb - ph); if (k >= -8 && k <= 8) C->koff[t][k + 8]++; } C->hist[t][cf_hbin(fabs(wa - wb))]++;
        } }
}
static double cf_hq(const uint64_t *h, uint64_t n, double q) { uint64_t s = 0; for (int k = 0; k < CF_NH; k++) { s += h[k]; if (s >= q * n) return k ? 1e-6 * pow(10, k / 512.0) : 0; } return INFINITY; }
int cmd_cf_compare(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm cf-compare <model-a> <model-b> [fields] [--step 4] [--threads 8]\n"); return 2; }
    fit_model *a = fit_load(argv[2]), *b = fit_load(argv[3]); if (!a || !b) { fprintf(stderr, "cf-compare: cannot load the models\n"); return 1; }
    if (a->lz != b->lz || a->ly != b->ly || a->lx != b->lx || a->L != b->L) { fprintf(stderr, "cf-compare: different lattices\n"); return 1; }
    const size_t nl = (size_t)a->lz * a->ly * a->lx; uint64_t *hl = calloc(CF_NH, sizeof *hl); double mx = 0; size_t ni = 0;
    size_t n01 = 0;
    for (size_t i = 0; i < nl; i++) { const double d = fabs((double)a->delta[i] - b->delta[i]); mx = fmax(mx, d); hl[cf_hbin(d)]++; ni += d > 0.5; n01 += d > 0.01; }
    printf("cf-compare: lattice %dx%dx%d (%zu nodes): |delta A - B| max %.3g, p99 %.3g, p99.9 %.3g, p99.99 %.3g turn; > 0.01 turn at %zu nodes, > 0.5 turn at %zu (%.2e)\n",
           a->lz, a->ly, a->lx, nl, mx, cf_hq(hl, nl, 0.99), cf_hq(hl, nl, 0.999), cf_hq(hl, nl, 0.9999), n01, ni, (double)ni / nl);
    if (argc > 4 && argv[4][0] != '-') {
        cf_nt = ufsm_threads(opt_atoi("--threads", cf_opt(argc, argv, "--threads", "0")), 8); const int step = opt_atoi("--step", cf_opt(argc, argv, "--step", "4"));
        fit_fields_buf g; if (fit_fields_load(argv[4], 0, 0, 0, &g)) { fprintf(stderr, "cf-compare: cannot load %s\n", argv[4]); return 1; }
        const int D = g.shape[0]; fit_fields_skip_planes(1 | 2 | 4);
        cf_cmp C = {.a = a, .b = b, .step = step, .h = g.h}; for (int t = 0; t < CF_MAXT; t++) C.hist[t] = calloc(CF_NH, sizeof(uint64_t));
        cf_pre pre = {0}; const int CZ = 128; cf_pre_start(&pre, argv[4], 0, CZ < D ? CZ : D, 1);
        for (int z0 = 0; z0 < D; z0 += CZ) {
            const int z1 = z0 + CZ < D ? z0 + CZ : D; fit_fields_buf fb; if (cf_pre_take(&pre, &fb)) return 1;
            if (z1 < D) cf_pre_start(&pre, argv[4], z1, z1 + CZ < D ? z1 + CZ : D, 1);
            C.f = &fb.f; C.zoff = z0; C.z0 = (z0 + step - 1) / step * step; C.z1 = z1;
            cf_par((z1 - C.z0 + step - 1) / step, cf_cmp_run, &C); fit_fields_free(&fb);
        }
        uint64_t n = 0, s = 0, *hv = calloc(CF_NH, sizeof *hv);
        for (int t = 0; t < CF_MAXT; t++) { n += C.n[t]; s += C.same[t]; for (int k = 0; k < CF_NH; k++) hv[k] += C.hist[t][k]; free(C.hist[t]); }
        printf("cf-compare: %llu field cells (every %d, material with a phase): snap integers round(W - phi) agree %.7f (%llu differ); |W A - B| median %.3g p99 %.3g p99.9 %.3g p99.99 %.3g max <= %.3g turn\n",
               (unsigned long long)n, step, n ? (double)s / n : 0, (unsigned long long)(n - s), cf_hq(hv, n, 0.5), cf_hq(hv, n, 0.99), cf_hq(hv, n, 0.999), cf_hq(hv, n, 0.9999), cf_hq(hv, n, 1.0));
        { uint64_t kk[17] = {0}; int km = 8; for (int t = 0; t < CF_MAXT; t++) for (int k = 0; k < 17; k++) kk[k] += C.koff[t][k];
          for (int k = 0; k < 17; k++) if (kk[k] > kk[km]) km = k;
          printf("cf-compare: gauge-free: snap integers agree %.7f at the modal offset %+d turn(s) (B - A)\n", n ? (double)kk[km] / n : 0, 8 - km); }
        free(hv); fit_fields_free(&g);
    }
    free(hl); fit_free(a); fit_free(b); return 0;
}

/* ufsm cf-demod MODEL FIELDS [--out DIR] [--R 1] [--L N (finer output lattice)] [--band 64] [--step 2] [--cz 32] [--c0 0.1] [--drift-max 0] [--threads N]
   Two label-free checks of a winding model (a count-field ref, ref-cf) against the network's fields, and one optional output.
   1. The z drift (audit ugeom #5: on Paris 4 the joined / ref integers drift 3 turns against the GT over ~5 cm of height while every
      gate passed). Per band of --band field rows: (a) the confidence-weighted phasor of the residual phi - W (phi the network phase,
      W the model) and its angle, unwrapped along z; (b) the mean derivative of W along the sheet's vertical tangent
      u = (e_z - n_z n) / |e_z - n_z n| (n the network normal: a sheet-following W has dW/du = 0), times the band height, summed along
      z. Both are "turns gained against the sheets" from the bottom band up; the excursion (max - min) is the drift. --drift-max D:
      exit 3 when the along-sheet excursion exceeds D turns (0: report only). Measured (ref-cf, --step 8 / 2, --band 128): P4 2.06 turns,
      1667 0.40; the residual phasor's band coherence is low (0.01-0.5) and its excursion (0.93 / 0.73) does not separate them. The
      along-sheet number ranks the two runs as the GT does, but its bands do not line up with the GT's per-band gauge steps: an alarm,
      not a locator.
   2. --out DIR: a demodulated model ("demod-only ref", audit ugeom #2): the model with its lattice delta + arg <c e^{2 pi i (phi - W)}>
      / 2 pi, the phasors pooled per lattice node (nearest node) and box-averaged over +-R nodes: the sub-turn from the phase, the
      integers from the model (the slab fits change no integers on the 1667 / P4 GT boxes, only the sub-turn). */
typedef struct {
    const fit_model *m; const fit_fields *f; int zoff, z0, z1, step, band, nb; double h, c0;
    const int *rows; int nrows;                 /* task k: the field rows rows[k] .. rows[k + 1] - 1 (all nearest the same lattice z node) */
    double (*bacc)[5];                          /* per thread x band: re, im, c, sum w s, sum w */
    float *lre, *lim;                           /* lattice accumulators (nullptr: no --out) */
} cf_dm2;
static void cf_dm2_run(void *a_, long k0, long k1, int t) {
    cf_dm2 *C = a_; const fit_fields *f = C->f; const fit_model *m = C->m; const size_t HW = (size_t)f->H * f->W; const double h = C->h, e = h;
    for (long k = k0; k < k1; k++) for (int z = C->rows[k]; z < C->rows[k + 1]; z++) {
        if ((z - C->z0) % C->step) continue;
        const int b = z / C->band; if (b >= C->nb) continue; double *A = C->bacc[(size_t)t * C->nb + b];
        for (int y = 0; y < f->H; y += C->step) for (int x = 0; x < f->W; x += C->step) {
            const size_t i = (size_t)(z - C->zoff) * HW + (size_t)y * f->W + x; if (!f->w[i]) continue;
            const double pc = (f->pc[i] - 128) / 127.0, ps = (f->ps[i] - 128) / 127.0, c = hypot(pc, ps); if (c < C->c0) continue;
            const double pz = (z + 0.5) * h, py = (y + 0.5) * h, px = (x + 0.5) * h, W0 = fit_W(m, pz, py, px);
            const double r = atan2(ps, pc) / (2 * M_PI) - W0, cr = c * cos(2 * M_PI * r), si = c * sin(2 * M_PI * r);
            A[0] += cr; A[1] += si; A[2] += c;
            double n[3]; const double l = cf_axis(f, i, n);
            if (l > 0.5) {
                const double g[3] = {(fit_W(m, pz + e, py, px) - fit_W(m, pz - e, py, px)) / (2 * e), (fit_W(m, pz, py + e, px) - fit_W(m, pz, py - e, px)) / (2 * e),
                                     (fit_W(m, pz, py, px + e) - fit_W(m, pz, py, px - e)) / (2 * e)};
                double u[3] = {-n[0] * n[0], -n[0] * n[1], -n[0] * n[2]}; u[0] += 1; const double ul = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
                if (ul > 0.3) { const double uz = u[0] / ul, s = (g[0] * u[0] + g[1] * u[1] + g[2] * u[2]) / ul, w = c * l; A[3] += w * s * uz; A[4] += w * uz * uz; }
                /* sum w uz s / sum w uz^2: the weighted mean of s / uz, the W gained per native voxel of height moving along the sheet */
            }
            if (C->lre) {
                const long a = lround(pz / m->L), bb = lround(py / m->L), cc = lround(px / m->L);
                if (a >= 0 && a < m->lz && bb >= 0 && bb < m->ly && cc >= 0 && cc < m->lx) { const size_t j = ((size_t)a * m->ly + bb) * m->lx + cc; C->lre[j] += (float)cr; C->lim[j] += (float)si; }
            }
        }
    }
}
int cmd_cf_demod(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm cf-demod <model> <fields> [--out DIR] [--R 1] [--L N] [--cmax 0.5] [--band 64] [--step 2] [--cz 32] [--c0 0.1] [--drift-max 0] [--threads N]\n"); return 2; }
    fit_model *m = fit_load(argv[2]); if (!m) { fprintf(stderr, "cf-demod: cannot load the model %s\n", argv[2]); return 1; }
    const char *out = cf_opt(argc, argv, "--out", ""); const int R = opt_atoi("--R", cf_opt(argc, argv, "--R", "1")), band = opt_atoi("--band", cf_opt(argc, argv, "--band", "64"));
    const int step = opt_atoi("--step", cf_opt(argc, argv, "--step", "2")), CZ = opt_atoi("--cz", cf_opt(argc, argv, "--cz", "32")); const double c0 = opt_atof("--c0", cf_opt(argc, argv, "--c0", "0.1")), dmax = opt_atof("--drift-max", cf_opt(argc, argv, "--drift-max", "0"));
    if (R < 0 || band < 1 || step < 1 || CZ < 1) { fprintf(stderr, "cf-demod: --R >= 0, --band, --step, --cz >= 1\n"); return 2; }
    cf_nt = ufsm_threads(opt_atoi("--threads", cf_opt(argc, argv, "--threads", "0")), 8);
    fit_fields_skip_planes(1 | 2 | 8);
    fit_fields_buf g; if (fit_fields_load(argv[3], 0, 0, 0, &g)) { fprintf(stderr, "cf-demod: cannot load %s\n", argv[3]); return 1; }
    const int D = g.shape[0]; const double h = g.h; fit_fields_free(&g);
    {   /* --L N: a finer output lattice (N native voxels, a divisor of the model's L): the model's delta resampled exactly (a trilinear
           field sampled at a refinement of its own lattice is reproduced by trilinear interpolation of the samples) */
        const int L2 = opt_atoi("--L", cf_opt(argc, argv, "--L", "0"));
        if (L2 > 0 && L2 != m->L) {
            if (m->L % L2) { fprintf(stderr, "cf-demod: --L %d must divide the model's lattice %d\n", L2, m->L); return 2; }
            const int f = m->L / L2, lz = (m->lz - 1) * f + 1, ly = (m->ly - 1) * f + 1, lx = (m->lx - 1) * f + 1;
            float *nd = malloc((size_t)lz * ly * lx * sizeof(float)); if (!nd) { fprintf(stderr, "cf-demod: out of memory (--L)\n"); return 1; }
            for (int a = 0; a < lz; a++) for (int b = 0; b < ly; b++) for (int c = 0; c < lx; c++) {
                const double qa = (double)a / f, qb = (double)b / f, qc = (double)c / f; const int a0 = a / f < m->lz - 1 ? a / f : m->lz - 2, b0 = b / f < m->ly - 1 ? b / f : m->ly - 2, c0 = c / f < m->lx - 1 ? c / f : m->lx - 2;
                const double ta = qa - a0, tb = qb - b0, tc = qc - c0; double v = 0;
                for (int k = 0; k < 8; k++) { const int da = k >> 2 & 1, db = k >> 1 & 1, dc = k & 1;
                    v += (da ? ta : 1 - ta) * (db ? tb : 1 - tb) * (dc ? tc : 1 - tc) * m->delta[((size_t)(a0 + da) * m->ly + b0 + db) * m->lx + c0 + dc]; }
                nd[((size_t)a * ly + b) * lx + c] = (float)v;
            }
            free(m->delta); m->delta = nd; m->L = L2; m->lz = lz; m->ly = ly; m->lx = lx;
        }
    }
    if (m->D != D || fabs(m->h - h) > 1e-9) fprintf(stderr, "cf-demod: WARNING: model box %d rows h %g vs fields %d rows h %g (a crop? positions are box-local)\n", m->D, m->h, D, h);
    const int nb = (D + band - 1) / band; double (*bacc)[5] = calloc((size_t)CF_MAXT * nb, sizeof *bacc);
    const size_t nl = (size_t)m->lz * m->ly * m->lx; float *lre = *out ? calloc(nl, 4) : nullptr, *lim = *out ? calloc(nl, 4) : nullptr;
    if (!bacc || (*out && (!lre || !lim))) { fprintf(stderr, "cf-demod: out of memory\n"); return 1; }
    int *rows = malloc((size_t)(CZ + 2) * sizeof *rows); const double t0 = cf_now();
    cf_pre pre = {0}; cf_pre_start(&pre, argv[3], 0, CZ < D ? CZ : D, 1);
    for (int z0 = 0; z0 < D; z0 += CZ) {
        const int z1 = z0 + CZ < D ? z0 + CZ : D; fit_fields_buf fb; if (cf_pre_take(&pre, &fb)) return 1;
        if (z1 < D) cf_pre_start(&pre, argv[3], z1, z1 + CZ < D ? z1 + CZ : D, 1);
        int nr = 0; long prev = LONG_MIN;   /* group the rows by their nearest lattice z node: tasks never share a node */
        for (int z = z0; z < z1; z++) { const long a = lround((z + 0.5) * h / m->L); if (a != prev) { rows[nr++] = z; prev = a; } }
        rows[nr] = z1;
        cf_dm2 C = {m, &fb.f, z0, z0, z1, step, band, nb, h, c0, rows, nr, bacc, lre, lim};
        cf_par(nr, cf_dm2_run, &C); cf_pre_give(&pre, &fb);
    }
    if (pre.has_nxt) fit_fields_free(&pre.nxt);
    printf("cf-demod: %s on %s: %d rows in bands of %d, every %d cells, %.1f s\n", argv[2], argv[3], D, band, step, cf_now() - t0);
    printf("cf-demod: band  z0(native)  cells  coherence  resid(turn)  resid_unwrapped  dW/dz_sheet(turn/band)  along_sheet_cum\n");
    /* the excursions count only the bands with >= 1/4 of the fullest band's phase weight: the scroll's end bands (a few cells of a
       cap) swing by turns (P4 band 1: -4.7, 1667 band 0: -1.0) and are no evidence of a drift */
    double smax = 0; for (int b = 0; b < nb; b++) { double c = 0; for (int t = 0; t < CF_MAXT; t++) c += bacc[(size_t)t * nb + b][2]; smax = fmax(smax, c); }
    double prevang = 0, unw = 0, cum = 0, umin = 1e30, umax = -1e30, cmin = 1e30, cmax = -1e30; int first = 1;
    double *bcum = calloc(3 * (size_t)nb, sizeof *bcum), *bdz = bcum + nb, *bfull = bcum + 2 * nb; int bb0 = -1, bb1 = -1;   /* the body: first .. last band with >= 1/2 the fullest band's weight */
    if (!bcum) { fprintf(stderr, "cf-demod: out of memory\n"); return 1; }
    for (int b = 0; b < nb; b++) {
        double s[5] = {0}; for (int t = 0; t < CF_MAXT; t++) for (int k = 0; k < 5; k++) s[k] += bacc[(size_t)t * nb + b][k];
        bcum[b] = cum;
        if (s[2] <= 0) continue;
        if (s[2] >= 0.5 * smax) { if (bb0 < 0) bb0 = b; bb1 = b; }
        const int full = s[2] >= 0.25 * smax;
        const double ang = atan2(s[1], s[0]) / (2 * M_PI), coh = hypot(s[0], s[1]) / s[2];
        if (first) { unw = ang; first = 0; } else { double d = ang - prevang; d -= floor(d + 0.5); unw += d; }
        prevang = ang; const double hb = (b + 1 < nb ? band : D - b * band) * h, dz = s[4] > 0 ? s[3] / s[4] * hb : 0; cum += dz;
        bcum[b] = cum; bdz[b] = dz; bfull[b] = full;
        if (full) { umin = fmin(umin, unw); umax = fmax(umax, unw); cmin = fmin(cmin, cum); cmax = fmax(cmax, cum); }
        printf("cf-demod: %4d  %8.0f  %8.0f  %.3f  %+.3f  %+.3f  %+.4f  %+.3f%s\n", b, b * band * h, s[2], coh, ang, unw, dz, cum, full ? "" : "  (end band: not counted)");
    }
    const double drift = cmax > cmin ? cmax - cmin : 0; if (!(umax > umin)) umin = umax = 0;
    printf("cf-demod: z drift: along-sheet excursion %.2f turns (end %+.2f), residual-phase excursion %.2f turns%s\n", drift, cum, umax - umin,
           dmax > 0 && drift > dmax ? " -- ALARM: above --drift-max" : "");
    /* body vs ends (fix0332 11-17): PHerc0332's 1.9-turn u1 excursion was 1.6 turns in the top two bands, the scroll's broken top end
       (phase coherence 0.10-0.15, a third of a full band's weight), and 0.22-0.34 over the body; the body excursion is the drift of the
       sheet identity where the scroll is whole, the end changes say how much the ends add */
    if (bb0 >= 0) { double bmin = 1e30, bmax = -1e30; for (int b = bb0; b <= bb1; b++) { bmin = fmin(bmin, bcum[b]); bmax = fmax(bmax, bcum[b]); }
        double below = 0, above = 0;   /* (counted bands only, as the excursion: the uncounted end bands swing by turns) */
        for (int b = 0; b < bb0; b++) if (bfull[b]) below += bdz[b];
        for (int b = bb1 + 1; b < nb; b++) if (bfull[b]) above += bdz[b];
        printf("cf-demod: z drift body: along-sheet excursion %.2f turns over bands %d..%d (native z %.0f..%.0f, >= 1/2 the fullest band's weight); "
               "below it %+.2f, above it %+.2f turns\n", bmax - bmin, bb0, bb1, bb0 * band * h, fmin(D, (bb1 + 1) * band) * h, below, above);
        /* the body's global native z range (unroll.sh ZR=auto: the wraps' --zr, so a broken end cap is not unrolled) */
        printf("cf-demod: body z range %.0f,%.0f (global native)\n", m->origin[0] + bb0 * band * h, m->origin[0] + fmin(D, (bb1 + 1) * band) * h); }
    free(bcum);
    int rc = dmax > 0 && drift > dmax ? 3 : 0;
    if (*out) {   /* +-R box average of the node phasors (separable, clamped), then delta += arg / 2 pi */
        float *tmp = malloc((size_t)(m->lz > m->ly ? (m->lz > m->lx ? m->lz : m->lx) : (m->ly > m->lx ? m->ly : m->lx)) * 4);
        const int dm[3] = {m->lz, m->ly, m->lx}; const size_t st[3] = {(size_t)m->ly * m->lx, (size_t)m->lx, 1};
        for (int ax = 0; ax < 3 && R > 0; ax++) for (float *v = lre; v; v = v == lre ? lim : nullptr) {
            const int a1 = ax == 0 ? 1 : 0, a2 = ax == 2 ? 1 : 2;
            for (int p = 0; p < dm[a1]; p++) for (int q = 0; q < dm[a2]; q++) {
                float *b0 = v + p * st[a1] + q * st[a2]; double s = 0; const int n = dm[ax];
                for (int k = 0; k <= R && k < n; k++) s += b0[k * st[ax]];
                for (int k = 0; k < n; k++) { tmp[k] = (float)s; if (k + R + 1 < n) s += b0[(k + R + 1) * st[ax]]; if (k - R >= 0) s -= b0[(k - R) * st[ax]]; }
                for (int k = 0; k < n; k++) b0[k * st[ax]] = tmp[k];
            }
        }
        size_t nset = 0; double sabs = 0;
        const double cmax = opt_atof("--cmax", cf_opt(argc, argv, "--cmax", "0.5"));   /* corrections above cmax turn (the phase half a turn off the ref: ambiguous) are left out */
        for (size_t j = 0; j < nl; j++) if (lre[j] != 0 || lim[j] != 0) { const double c = atan2(lim[j], lre[j]) / (2 * M_PI); if (fabs(c) > cmax) continue; m->delta[j] += (float)c; nset++; sabs += fabs(c); }
        free(tmp);
        if (fit_save(m, out)) { fprintf(stderr, "cf-demod: cannot write %s\n", out); return 1; }
        printf("cf-demod: demodulated model (R %d nodes, lattice %d native) -> %s: %zu of %zu nodes corrected, mean |correction| %.3f turn\n", R, m->L, out, nset, nl, nset ? sabs / nset : 0);
    }
    free(bacc); free(lre); free(lim); free(rows); fit_free(m);
    fit_fields_skip_planes(0);   /* (process-wide: a later load in this process, e.g. a test's count-field, needs sp again) */
    return rc;
}
