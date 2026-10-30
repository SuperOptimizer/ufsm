/* SLIM flattening of tifxyz wraps: see src/flatten.h. CLI: ufsm flatten, ufsm flatten-warp (end of file). */
#include "flatten.h"
#include "imgops.h"
#include "tiffw.h"
#include <dirent.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static double fnow(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

/* ---- a static parallel for: f(ctx, lo, hi, tid) over [0, n) in nt contiguous chunks ---- */
typedef void (*pf_fn)(void *ctx, long lo, long hi, int tid);
typedef struct { pf_fn f; void *ctx; long n; int nt, tid; } pf_arg;
static void *pf_run(void *a) { pf_arg *p = a; p->f(p->ctx, p->n * p->tid / p->nt, p->n * (p->tid + 1) / p->nt, p->tid); return nullptr; }
enum { PF_MAX = 64 };
static void parfor(long n, int nt, pf_fn f, void *ctx) {
    if (nt > PF_MAX) nt = PF_MAX;
    if (nt > n / 16384) nt = (int)(n / 16384);   /* >= 16k items per thread: a CG step is ~1 ms of work, thread start-up must not dominate */
    if (nt <= 1) { f(ctx, 0, n, 0); return; }
    pthread_t th[PF_MAX]; pf_arg a[PF_MAX];
    for (int t = 0; t < nt; t++) a[t] = (pf_arg){f, ctx, n, nt, t};
    for (int t = 1; t < nt; t++) pthread_create(&th[t], nullptr, pf_run, &a[t]);
    pf_run(&a[0]);
    for (int t = 1; t < nt; t++) pthread_join(th[t], nullptr);
}

flat_opts flat_opts_default(void) { flat_opts o = {30, 200, 16, 20000, 1, 3.0, 1e-5, 0}; o.gpu_dev = -1; return o; }

/* ---- per-triangle geometry: gradient coefficients g (6: g0, g1, g2 in the triangle's own 2D frame) and 3D area ---- */
static int tri_frame(const double *p0, const double *p1, const double *p2, double g[6], double *area) {
    double e1[3], e2[3]; for (int d = 0; d < 3; d++) { e1[d] = p1[d] - p0[d]; e2[d] = p2[d] - p0[d]; }
    const double l1 = sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
    const double n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
    const double nl = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (!(l1 > 0) || !(nl > 0)) return -1;
    const double qx = (e2[0] * e1[0] + e2[1] * e1[1] + e2[2] * e1[2]) / l1, qy = nl / l1;   /* e2 in the frame (e1 / l1, n x e1 / |n x e1|) */
    g[2] = 1 / l1; g[3] = -qx / (l1 * qy); g[4] = 0; g[5] = 1 / qy; g[0] = -(g[2] + g[4]); g[1] = -(g[3] + g[5]);
    *area = 0.5 * nl; return 0;
}
static inline void tri_jac(const double *g, const double *x, const int *v, double J[4]) {   /* J = sum_k x_k g_k^T (rows u, v) */
    J[0] = J[1] = J[2] = J[3] = 0;
    for (int k = 0; k < 3; k++) { const double u = x[2 * v[k]], w = x[2 * v[k] + 1]; J[0] += u * g[2 * k]; J[1] += u * g[2 * k + 1]; J[2] += w * g[2 * k]; J[3] += w * g[2 * k + 1]; }
}
static inline double sd_energy(const double J[4]) {   /* symmetric Dirichlet |J|^2 + |J^-1|^2; inf when inverted */
    const double det = J[0] * J[3] - J[1] * J[2], f = J[0] * J[0] + J[1] * J[1] + J[2] * J[2] + J[3] * J[3];
    return det > 0 ? f * (1 + 1 / (det * det)) : INFINITY;
}
static inline void svd2(const double J[4], double *s1, double *s2, double *phi, double *rot) {   /* J = Rot(phi) diag(s1, s2) Rot(theta) */
    const double E = (J[0] + J[3]) / 2, F = (J[0] - J[3]) / 2, G = (J[2] + J[1]) / 2, H = (J[2] - J[1]) / 2;
    const double Q = hypot(E, H), R = hypot(F, G), a1 = atan2(G, F), a2 = atan2(H, E);
    *s1 = Q + R; *s2 = Q - R; *phi = (a2 + a1) / 2; *rot = a2;   /* closest rotation U V^T = Rot(a2) */
}

/* ---- stats ---- */
typedef struct { double w, a; } wa;
/* stable sort by a (LSD radix over the double's order-preserving bits, 4 x 16-bit digits): the order glibc's merge-sort qsort gave
   (stable, ties by index), so the quantiles' running sums are bit-identical; ~10x faster on the 10-20M triangle grids */
static inline uint64_t wa_key(double a) { uint64_t u; memcpy(&u, &a, 8); return u >> 63 ? ~u : u | 1ULL << 63; }
static void wa_sort(wa *v, size_t n) {
    if (n < 2) return;
    wa *t = malloc(n * sizeof *t); size_t *cnt = malloc(65536 * sizeof *cnt);
    if (!t || !cnt) { free(t); free(cnt); for (size_t i = 1; i < n; i++) { const wa x = v[i]; size_t j = i; while (j && wa_key(v[j - 1].a) > wa_key(x.a)) { v[j] = v[j - 1]; j--; } v[j] = x; } return; }
    wa *src = v, *dst = t;
    for (int sh = 0; sh < 64; sh += 16) {
        memset(cnt, 0, 65536 * sizeof *cnt);
        for (size_t i = 0; i < n; i++) cnt[wa_key(src[i].a) >> sh & 0xffff]++;
        if (cnt[wa_key(src[0].a) >> sh & 0xffff] == n) continue;   /* one digit value: the pass changes nothing */
        size_t o = 0; for (int d = 0; d < 65536; d++) { const size_t c = cnt[d]; cnt[d] = o; o += c; }
        for (size_t i = 0; i < n; i++) dst[cnt[wa_key(src[i].a) >> sh & 0xffff]++] = src[i];
        wa *x = src; src = dst; dst = x;
    }
    if (src != v) memcpy(v, src, n * sizeof *v);
    free(t); free(cnt);
}
static double wquant(wa *v, size_t n, double tot, double q) { double c = 0; for (size_t i = 0; i < n; i++) { c += v[i].w; if (c >= q * tot) return v[i].a; } return n ? v[n - 1].a : 0; }
void flat_stats_of(const double *P, const int *tri, long long nt, const double *uv, flat_stats *s) {
    memset(s, 0, sizeof *s);
    wa *an = malloc((size_t)(nt ? nt : 1) * sizeof *an), *mx = malloc((size_t)(nt ? nt : 1) * sizeof *mx), *mn = malloc((size_t)(nt ? nt : 1) * sizeof *mn);
    double A = 0, Auv = 0, E = 0, a12 = 0, s12 = 0; size_t n = 0;
    for (long long t = 0; t < nt; t++) {
        const int *v = tri + 3 * t; double g[6], area;
        if (tri_frame(P + 3 * v[0], P + 3 * v[1], P + 3 * v[2], g, &area)) continue;
        if (!isfinite(uv[2 * v[0]]) || !isfinite(uv[2 * v[1]]) || !isfinite(uv[2 * v[2]])) continue;
        double J[4]; tri_jac(g, uv, v, J);
        const double det = J[0] * J[3] - J[1] * J[2];
        s->nt++; A += area; Auv += det * area;
        if (!(det > 0)) { s->flips++; continue; }
        double s1, s2, ph, ro; svd2(J, &s1, &s2, &ph, &ro);
        const double e = sd_energy(J); E += area * e;
        const double r = s1 / s2; if (r > 1.2) a12 += area; if (s1 > 1.2 || s2 < 1 / 1.2) s12 += area;
        an[n] = (wa){area, r}; mx[n] = (wa){area, s1}; mn[n] = (wa){area, s2}; n++;
    }
    if (A > 0) {
        s->energy = E / A; s->aniso12 = a12 / A; s->stretch12 = s12 / A; s->area_ratio = Auv / A;
        wa_sort(an, n); wa_sort(mx, n); wa_sort(mn, n);
        double An = 0; for (size_t i = 0; i < n; i++) An += an[i].w;
        s->aniso_p50 = wquant(an, n, An, 0.5); s->aniso_p95 = wquant(an, n, An, 0.95); s->aniso_p99 = wquant(an, n, An, 0.99);
        s->smax_p95 = wquant(mx, n, An, 0.95); s->smin_p05 = wquant(mn, n, An, 0.05);
    }
    free(an); free(mx); free(mn);
}
void flat_stats_print(FILE *f, const char *label, const flat_stats *s) {
    fprintf(f, "%s: energy %.4f aniso>1.2 %.2f%% stretch>1.2 %.2f%% aniso p50/p95/p99 %.3f/%.3f/%.3f smax p95 %.3f smin p05 %.3f area-ratio %.4f flips %lld tris %lld\n",
           label, s->energy, 100 * s->aniso12, 100 * s->stretch12, s->aniso_p50, s->aniso_p95, s->aniso_p99, s->smax_p95, s->smin_p05, s->area_ratio, s->flips, s->nt);
}

/* ---- the SLIM solver ---- */
typedef struct {
    int nv, nt, nth; const int *tri;
    double *g, *A;           /* 6 / 1 per triangle */
    int *vo, *vl;            /* per vertex incident (3 t + k), CSR */
    double *M, *MR, *Q;      /* per triangle: weights M = W^T W (3), A M R (4), A M J(p) (4) */
    double lam;
    double *x, *d, *rhs, *diag, *r, *z, *p, *q, *xt;
    double part[PF_MAX][2];
    const double *src; double *dst;   /* matvec operand / result */
    double alpha, beta;
} slim;

static void k_energy(void *c, long lo, long hi, int tid) {   /* part[tid][0] = sum A E at S->src */
    slim *S = c; double e = 0, a = 0;
    for (long t = lo; t < hi; t++) { if (!(S->A[t] > 0)) continue; double J[4]; tri_jac(S->g + 6 * t, S->src, S->tri + 3 * t, J); e += S->A[t] * sd_energy(J); a += S->A[t]; }
    S->part[tid][0] = e; S->part[tid][1] = a;
}
static double energy_at(slim *S, const double *x) {
    for (int t = 0; t < PF_MAX; t++) S->part[t][0] = S->part[t][1] = 0;
    S->src = x; parfor(S->nt, S->nth, k_energy, S);
    double e = 0, a = 0; for (int t = 0; t < PF_MAX; t++) { e += S->part[t][0]; a += S->part[t][1]; }
    return a > 0 ? e / a : 0;
}
static void k_local(void *c, long lo, long hi, int tid) {   /* weights M, the rotation targets A M R */
    slim *S = c;
    for (long t = lo; t < hi; t++) {
        double J[4]; tri_jac(S->g + 6 * t, S->x, S->tri + 3 * t, J);
        double s1, s2, ph, ro; svd2(J, &s1, &s2, &ph, &ro);
        if (!(s2 > 1e-12)) s2 = 1e-12;
        const double w1 = (s1 + 1) * (s1 * s1 + 1) / (s1 * s1 * s1), w2 = (s2 + 1) * (s2 * s2 + 1) / (s2 * s2 * s2);   /* w^2 */
        const double cp = cos(ph), sp = sin(ph);
        double *M = S->M + 3 * t; M[0] = cp * cp * w1 + sp * sp * w2; M[1] = cp * sp * (w1 - w2); M[2] = sp * sp * w1 + cp * cp * w2;
        const double cr = cos(ro), sr = sin(ro), A = S->A[t];   /* R = [[cr, -sr], [sr, cr]] */
        double *B = S->MR + 4 * t;
        B[0] = A * (M[0] * cr + M[1] * sr); B[1] = A * (-M[0] * sr + M[1] * cr); B[2] = A * (M[1] * cr + M[2] * sr); B[3] = A * (-M[1] * sr + M[2] * cr);
    }
}
static void k_triq(void *c, long lo, long hi, int tid) {   /* Q_t = A M J(src) */
    slim *S = c;
    for (long t = lo; t < hi; t++) {
        double J[4]; tri_jac(S->g + 6 * t, S->src, S->tri + 3 * t, J); const double *M = S->M + 3 * t, A = S->A[t]; double *Q = S->Q + 4 * t;
        Q[0] = A * (M[0] * J[0] + M[1] * J[2]); Q[1] = A * (M[0] * J[1] + M[1] * J[3]); Q[2] = A * (M[1] * J[0] + M[2] * J[2]); Q[3] = A * (M[1] * J[1] + M[2] * J[3]);
    }
}
static void k_gather(void *c, long lo, long hi, int tid) {   /* dst = sum over incident triangles Q_t g_k + lam src; part[tid][0] = src . dst */
    slim *S = c; double pq = 0;
    for (long v = lo; v < hi; v++) {
        double a = S->lam * S->src[2 * v], b = S->lam * S->src[2 * v + 1];
        for (int i = S->vo[v]; i < S->vo[v + 1]; i++) {
            const int t = S->vl[i] / 3, k = S->vl[i] % 3; const double *Q = S->Q + 4 * t, *g = S->g + 6 * t + 2 * k;
            a += Q[0] * g[0] + Q[1] * g[1]; b += Q[2] * g[0] + Q[3] * g[1];
        }
        S->dst[2 * v] = a; S->dst[2 * v + 1] = b; pq += S->src[2 * v] * a + S->src[2 * v + 1] * b;
    }
    S->part[tid][0] = pq;
}
static double matvec(slim *S, const double *x, double *y) {   /* y = H x; returns x . y */
    S->src = x; S->dst = y; parfor(S->nt, S->nth, k_triq, S);
    for (int t = 0; t < PF_MAX; t++) S->part[t][0] = 0;
    parfor(S->nv, S->nth, k_gather, S);
    double s = 0; for (int t = 0; t < PF_MAX; t++) s += S->part[t][0]; return s;
}
static void k_rhs(void *c, long lo, long hi, int tid) {   /* rhs = sum A M R g + lam x; diag = sum A M_aa |g|^2 + lam */
    slim *S = c;
    for (long v = lo; v < hi; v++) {
        double a = S->lam * S->x[2 * v], b = S->lam * S->x[2 * v + 1], da = S->lam, db = S->lam;
        for (int i = S->vo[v]; i < S->vo[v + 1]; i++) {
            const int t = S->vl[i] / 3, k = S->vl[i] % 3; const double *B = S->MR + 4 * t, *g = S->g + 6 * t + 2 * k, *M = S->M + 3 * t, gg = g[0] * g[0] + g[1] * g[1];
            a += B[0] * g[0] + B[1] * g[1]; b += B[2] * g[0] + B[3] * g[1]; da += S->A[t] * M[0] * gg; db += S->A[t] * M[2] * gg;
        }
        S->rhs[2 * v] = a; S->rhs[2 * v + 1] = b; S->diag[2 * v] = da; S->diag[2 * v + 1] = db;
    }
}
static void k_update(void *c, long lo, long hi, int tid) {   /* x += alpha p; r -= alpha q; z = r / diag; part = (r . z, r . r) */
    slim *S = c; double rz = 0, rr = 0;
    for (long i = 2 * lo; i < 2 * hi; i++) { S->d[i] += S->alpha * S->p[i]; S->r[i] -= S->alpha * S->q[i]; S->z[i] = S->r[i] / S->diag[i]; rz += S->r[i] * S->z[i]; rr += S->r[i] * S->r[i]; }
    S->part[tid][0] = rz; S->part[tid][1] = rr;
}
static void k_pdir(void *c, long lo, long hi, int tid) { slim *S = c; for (long i = 2 * lo; i < 2 * hi; i++) S->p[i] = S->z[i] + S->beta * S->p[i]; }
typedef struct { slim *S; const double *x, *d; double tmax[PF_MAX]; } stepjob;
static void k_maxstep(void *c, long lo, long hi, int tid) {   /* the first inversion along x + t d, t in (0, inf) */
    stepjob *J = c; const slim *S = J->S; double tm = INFINITY;
    for (long t = lo; t < hi; t++) {
        const int *v = S->tri + 3 * t; const double *x = J->x, *d = J->d;
        const double e1u = x[2 * v[1]] - x[2 * v[0]], e1v = x[2 * v[1] + 1] - x[2 * v[0] + 1], e2u = x[2 * v[2]] - x[2 * v[0]], e2v = x[2 * v[2] + 1] - x[2 * v[0] + 1];
        const double f1u = d[2 * v[1]] - d[2 * v[0]], f1v = d[2 * v[1] + 1] - d[2 * v[0] + 1], f2u = d[2 * v[2]] - d[2 * v[0]], f2v = d[2 * v[2] + 1] - d[2 * v[0] + 1];
        const double cc = e1u * e2v - e1v * e2u, bb = e1u * f2v - e1v * f2u + f1u * e2v - f1v * e2u, aa = f1u * f2v - f1v * f2u;
        double r = INFINITY;
        if (fabs(aa) < 1e-14 * (fabs(bb) + fabs(cc) + 1e-300)) { if (bb < 0) r = -cc / bb; }
        else {
            const double D = bb * bb - 4 * aa * cc;
            if (D >= 0) { const double sq = sqrt(D), q = -0.5 * (bb + (bb >= 0 ? sq : -sq)), r1 = q / aa, r2 = q != 0 ? cc / q : INFINITY;
                          r = fmin(r, r1 > 0 ? r1 : INFINITY); r = fmin(r, r2 > 0 ? r2 : INFINITY); }
        }
        if (r < tm) tm = r;
    }
    J->tmax[tid] = tm;
}
static double max_step(slim *S, const double *x, const double *d) {
    stepjob J = {S, x, d, {0}}; for (int t = 0; t < PF_MAX; t++) J.tmax[t] = INFINITY;
    parfor(S->nt, S->nth, k_maxstep, &J);
    double m = INFINITY; for (int t = 0; t < PF_MAX; t++) if (J.tmax[t] < m) m = J.tmax[t]; return m;
}
static void k_axpy(void *c, long lo, long hi, int tid) { slim *S = c; for (long i = 2 * lo; i < 2 * hi; i++) S->xt[i] = S->x[i] + S->alpha * S->d[i]; }

/* line search from x along d (flip-free cap, then halving until the energy falls); x updated; returns the new energy */
static double line_search(slim *S, double E0, double *step) {
    const double tm = max_step(S, S->x, S->d); double a = fmin(1.0, 0.8 * tm), E = E0;
    for (int k = 0; k < 40 && a > 1e-12; k++, a *= 0.5) {
        S->alpha = a; parfor(S->nv, S->nth, k_axpy, S);
        E = energy_at(S, S->xt);
        if (isfinite(E) && E < E0) { memcpy(S->x, S->xt, (size_t)S->nv * 2 * sizeof(double)); *step = a; return E; }
    }
    *step = 0; return E0;
}

static int slim_init(slim *S, const double *P, int nv, const int *tri, int nt, int nth) {
    memset(S, 0, sizeof *S); S->nv = nv; S->nt = nt; S->tri = tri; S->nth = nth < 1 ? 1 : nth;
    S->g = malloc((size_t)nt * 6 * 8); S->A = malloc((size_t)nt * 8); S->M = malloc((size_t)nt * 3 * 8); S->MR = malloc((size_t)nt * 4 * 8); S->Q = malloc((size_t)nt * 4 * 8);
    S->vo = calloc((size_t)nv + 1, 4); S->vl = malloc((size_t)nt * 3 * 4);
    const size_t n2 = (size_t)nv * 2 * 8;
    S->d = malloc(n2); S->rhs = malloc(n2); S->diag = malloc(n2); S->r = malloc(n2); S->z = malloc(n2); S->p = malloc(n2); S->q = malloc(n2); S->xt = malloc(n2);
    if (!S->g || !S->A || !S->M || !S->MR || !S->Q || !S->vo || !S->vl || !S->d || !S->rhs || !S->diag || !S->r || !S->z || !S->p || !S->q || !S->xt) return -1;
    for (int t = 0; t < nt; t++) {
        const int *v = tri + 3 * t;
        if (tri_frame(P + 3 * v[0], P + 3 * v[1], P + 3 * v[2], S->g + 6 * t, S->A + t)) { memset(S->g + 6 * t, 0, 48); S->A[t] = 0; }
        for (int k = 0; k < 3; k++) S->vo[v[k] + 1]++;
    }
    for (int v = 0; v < nv; v++) S->vo[v + 1] += S->vo[v];
    int *fill = malloc((size_t)nv * 4); if (!fill) return -1; memcpy(fill, S->vo, (size_t)nv * 4);
    for (int t = 0; t < nt; t++) for (int k = 0; k < 3; k++) S->vl[fill[tri[3 * t + k]]++] = 3 * t + k;
    free(fill); return 0;
}
static void slim_free(slim *S) {
    free(S->g); free(S->A); free(S->M); free(S->MR); free(S->Q); free(S->vo); free(S->vl);
    free(S->d); free(S->rhs); free(S->diag); free(S->r); free(S->z); free(S->p); free(S->q); free(S->xt);
}

/* src/nn_flat.cu (weak: binaries without it, e.g. test_flatten, link CPU-only) */
int flat_slim_gpu(const double *g, const double *A, const int *tri, int nt, const int *vo, const int *vl, int nv, double *x,
                  int iters, int cg_iters, double tol, int verbose, const char *tag, int dev, double *Eout) __attribute__((weak));
int flat_gpu_count(void) __attribute__((weak));
int flat_gpu_devices(void) { return flat_gpu_count ? flat_gpu_count() : 0; }

/* SLIM iterations on S (S->x = the current uv, flip-free) */
static double slim_run(slim *S, const flat_opts *o, int iters, const char *tag) {
    if (o->gpu && flat_slim_gpu) { double E; if (!flat_slim_gpu(S->g, S->A, S->tri, S->nt, S->vo, S->vl, S->nv, S->x, iters, o->cg_iters, o->tol, o->verbose, tag, o->gpu_dev, &E)) return E; }
    double E = energy_at(S, S->x); const size_t n2 = (size_t)S->nv * 2;
    if (o->verbose) fprintf(stderr, "flatten: %s %d vertices %d triangles, energy %.5f\n", tag, S->nv, S->nt, E);
    if (!isfinite(E)) return E;
    for (int it = 0; it < iters; it++) {
        const double t0 = fnow();
        parfor(S->nt, S->nth, k_local, S);
        { double md = 0; S->lam = 0; parfor(S->nv, S->nth, k_rhs, S); for (size_t i = 0; i < n2; i++) md += S->diag[i]; S->lam = 1e-6 * md / (double)n2; }
        parfor(S->nv, S->nth, k_rhs, S);
        /* CG on H y = rhs from y = x (warm start); S->d holds y, later turned into the step y - x */
        memcpy(S->d, S->x, n2 * 8);
        matvec(S, S->d, S->q);
        double rz = 0, rr0 = 0, bb = 0;
        for (size_t i = 0; i < n2; i++) { S->r[i] = S->rhs[i] - S->q[i]; S->z[i] = S->r[i] / S->diag[i]; S->p[i] = S->z[i]; rz += S->r[i] * S->z[i]; rr0 += S->r[i] * S->r[i]; bb += S->rhs[i] * S->rhs[i]; }
        int k = 0; double rr = rr0;
        for (; k < o->cg_iters && rr > 1e-14 * bb; k++) {
            const double pq = matvec(S, S->p, S->q); if (!(pq > 0)) break;
            S->alpha = rz / pq;
            for (int t = 0; t < PF_MAX; t++) S->part[t][0] = S->part[t][1] = 0;
            parfor(S->nv, S->nth, k_update, S);
            double rz2 = 0; rr = 0; for (int t = 0; t < PF_MAX; t++) { rz2 += S->part[t][0]; rr += S->part[t][1]; }
            S->beta = rz2 / rz; rz = rz2; parfor(S->nv, S->nth, k_pdir, S);
        }
        for (size_t i = 0; i < n2; i++) S->d[i] -= S->x[i];
        double step; const double E2 = line_search(S, E, &step);
        if (o->verbose) fprintf(stderr, "flatten: %s it %d energy %.5f -> %.5f (step %.3g, cg %d, res %.2g, %.2fs)\n", tag, it, E, E2, step, k, sqrt(rr / (rr0 > 0 ? rr0 : 1)), fnow() - t0);
        const double dE = E - E2; E = E2;
        if (step == 0 || dE < o->tol * E) break;
    }
    return E;
}

double flat_slim(const double *P, int nv, const int *tri, int nt, double *uv, const flat_opts *o) {
    slim S; if (slim_init(&S, P, nv, tri, nt, o->nthreads)) { slim_free(&S); return -1; }
    S.x = uv; const double E = slim_run(&S, o, o->iters, "slim");
    slim_free(&S); return E;
}

/* ---- grid meshes ---- */
typedef struct { int D, lw, lh; int *lv; flat_mesh m; } level;   /* lattice every D-th grid vertex; lv: lattice -> mesh vertex (-1) */
static int uf_find(int *p, int a) { while (p[a] != a) { p[a] = p[p[a]]; a = p[a]; } return a; }
static inline int gvalid(const float *X, const float *Y, const float *Z, size_t i) { return Z[i] >= 0 && isfinite(X[i]) && isfinite(Y[i]) && isfinite(Z[i]); }
static int build_level(const float *X, const float *Y, const float *Z, int w, int h, double step, int D, double max_edge, int all, const double *col_u, level *L) {
    memset(L, 0, sizeof *L); L->D = D; L->lw = (w - 1) / D + 1; L->lh = (h - 1) / D + 1;
    const int lw = L->lw, lh = L->lh; const size_t ln = (size_t)lw * lh;
    if (lw < 2 || lh < 2) return -1;
    L->lv = malloc(ln * 4); int *par = malloc(ln * 4); int *tr = malloc(ln * 2 * 3 * 4);
    if (!L->lv || !par || !tr) { free(par); free(tr); return -1; }
    for (size_t i = 0; i < ln; i++) par[i] = (int)i;
    const double sp = D * step, lmax2 = (max_edge * sp * 1.4143) * (max_edge * sp * 1.4143), amin = 1e-4 * sp * sp;
    long nt = 0;
#define LIDX(i, j) ((size_t)(i) * lw + (j))
#define GIDX(i, j) ((size_t)(i) * D * w + (size_t)(j) * D)
    for (int i = 0; i + 1 < lh; i++) for (int j = 0; j + 1 < lw; j++) {
        const size_t c[4] = {LIDX(i, j), LIDX(i, j + 1), LIDX(i + 1, j), LIDX(i + 1, j + 1)}, gc[4] = {GIDX(i, j), GIDX(i, j + 1), GIDX(i + 1, j), GIDX(i + 1, j + 1)};
        int ok[4], nok = 0; for (int k = 0; k < 4; k++) nok += ok[k] = gvalid(X, Y, Z, gc[k]);
        int cand[2][3], nc = 0;   /* a b c d = 0 1 2 3; positive in (column, row): (a, b, d), (a, d, c); 3 valid: the one triangle */
        if (nok == 4) { cand[0][0] = 0; cand[0][1] = 1; cand[0][2] = 3; cand[1][0] = 0; cand[1][1] = 3; cand[1][2] = 2; nc = 2; }
        else if (nok == 3) {
            static const int one[4][3] = {{1, 3, 2}, {0, 3, 2}, {0, 1, 3}, {0, 1, 2}};
            for (int k = 0; k < 4; k++) if (!ok[k]) { memcpy(cand[0], one[k], sizeof cand[0]); nc = 1; }
        }
        for (int q = 0; q < nc; q++) {
            double p[3][3]; for (int k = 0; k < 3; k++) { const size_t g = gc[cand[q][k]]; p[k][0] = X[g]; p[k][1] = Y[g]; p[k][2] = Z[g]; }
            int bad = 0;
            for (int k = 0; k < 3 && !bad; k++) { const double *a = p[k], *b = p[(k + 1) % 3]; const double l2 = (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]); bad = l2 > lmax2; }
            double g6[6], ar; if (bad || tri_frame(p[0], p[1], p[2], g6, &ar) || ar < amin) continue;
            for (int k = 0; k < 3; k++) tr[3 * nt + k] = (int)c[cand[q][k]];
            const int r0 = uf_find(par, (int)c[cand[q][0]]); for (int k = 1; k < 3; k++) { const int rk = uf_find(par, (int)c[cand[q][k]]); if (rk != r0) par[rk] = r0; }
            nt++;
        }
    }
    /* the largest connected component (by triangles) */
    int *cnt = calloc(ln, 4); if (!cnt) { free(par); free(tr); return -1; }
    for (long t = 0; t < nt; t++) cnt[uf_find(par, tr[3 * t])]++;
    int best = -1; for (size_t i = 0; i < ln; i++) if (cnt[i] > 0 && (best < 0 || cnt[i] > cnt[best])) best = (int)i;
    for (size_t i = 0; i < ln; i++) L->lv[i] = -1;
    if (all && best >= 0) cnt[best] = (int)nt;   /* (every component: the triangle buffer holds them all) */
    flat_mesh *m = &L->m; m->tri = malloc((size_t)(best >= 0 ? cnt[best] : 1) * 3 * 4); int nv = 0, mt = 0;
    for (long t = 0; t < nt && best >= 0; t++) {
        if (!all && uf_find(par, tr[3 * t]) != best) continue;
        for (int k = 0; k < 3; k++) { const int l = tr[3 * t + k]; if (L->lv[l] < 0) L->lv[l] = nv++; m->tri[3 * mt + k] = L->lv[l]; }
        mt++;
    }
    m->nv = nv; m->nt = mt; m->gi = malloc((size_t)(nv ? nv : 1) * 4); m->P = malloc((size_t)(nv ? nv : 1) * 3 * 8);
    m->uv = malloc((size_t)(nv ? nv : 1) * 2 * 8); m->uv0 = malloc((size_t)(nv ? nv : 1) * 2 * 8);
    for (int i = 0; i < lh; i++) for (int j = 0; j < lw; j++) {
        const int v = L->lv[LIDX(i, j)]; if (v < 0) continue; const size_t g = GIDX(i, j);
        m->gi[v] = (int)g; m->P[3 * v] = X[g]; m->P[3 * v + 1] = Y[g]; m->P[3 * v + 2] = Z[g];
        m->uv0[2 * v] = m->uv[2 * v] = col_u ? col_u[(size_t)j * D] : (double)j * D * step; m->uv0[2 * v + 1] = m->uv[2 * v + 1] = (double)i * D * step;
    }
#undef LIDX
#undef GIDX
    free(par); free(tr); free(cnt);
    return nv >= 3 && mt >= 1 ? 0 : -1;
}
void flat_mesh_free(flat_mesh *m) { free(m->gi); free(m->P); free(m->uv); free(m->uv0); free(m->tri); memset(m, 0, sizeof *m); }

static long count_flips(const flat_mesh *m, const double *uv, uint8_t *bad) {   /* triangles with det <= 1e-3 of the grid map's; bad[v] marked */
    long nb = 0;
    for (int t = 0; t < m->nt; t++) {
        const int *v = m->tri + 3 * t; const double *a = uv + 2 * v[0], *b = uv + 2 * v[1], *c = uv + 2 * v[2];
        const double *a0 = m->uv0 + 2 * v[0], *b0 = m->uv0 + 2 * v[1], *c0 = m->uv0 + 2 * v[2];
        const double d = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]), d0 = (b0[0] - a0[0]) * (c0[1] - a0[1]) - (b0[1] - a0[1]) * (c0[0] - a0[0]);
        if (!(d > 1e-3 * d0)) { nb++; if (bad) for (int k = 0; k < 3; k++) bad[v[k]] = 1; }
    }
    return nb;
}
/* fine level uv from the coarse level's solution: displacement (uv - uv0) bilinear over the coarse lattice cell, unknowns filled from
   neighbours, flipped triangles' vertices relaxed to their neighbours' mean; a flip-free line search from the grid map as the fallback */
static void prolongate(const level *C, level *F, const flat_opts *o) {
    flat_mesh *m = &F->m; const int r = C->D / F->D; const size_t n2 = (size_t)m->nv * 2;
    double *dl = malloc(n2 * 8); uint8_t *known = calloc((size_t)m->nv, 1), *bad = calloc((size_t)m->nv, 1); double *acc = malloc(n2 * 8 + 8); int *nn = malloc((size_t)m->nv * 4 + 4);
    for (int i = 0; i < F->lh; i++) for (int j = 0; j < F->lw; j++) {
        const int v = F->lv[(size_t)i * F->lw + j]; if (v < 0) continue;
        /* piecewise linear over the coarse lattice's triangles (a b d / a d c of each cell, as the meshes split them): a fine vertex inside
           a coarse triangle whose map is positively oriented maps through that triangle's affine map (flip-free inside it) */
        const int ci = i / r, cj = j / r; const double fi = (double)(i % r) / r, fj = (double)(j % r) / r;
        const int L4[4][2] = {{ci, cj}, {ci, cj + 1}, {ci + 1, cj}, {ci + 1, cj + 1}}; int cv4[4];
        for (int q = 0; q < 4; q++) cv4[q] = L4[q][0] < C->lh && L4[q][1] < C->lw ? C->lv[(size_t)L4[q][0] * C->lw + L4[q][1]] : -1;
        int tv[3]; double tw[3];
        if (fj >= fi) { tv[0] = cv4[0]; tv[1] = cv4[1]; tv[2] = cv4[3]; tw[0] = 1 - fj; tw[1] = fj - fi; tw[2] = fi; }
        else { tv[0] = cv4[0]; tv[1] = cv4[3]; tv[2] = cv4[2]; tw[0] = 1 - fi; tw[1] = fj; tw[2] = fi - fj; }
        double s[2] = {0, 0}, ws = 0;
        int have = 1; for (int k = 0; k < 3; k++) if (tw[k] > 1e-12 && tv[k] < 0) have = 0;
        if (have) {   /* (on a coarse vertex / edge only the vertices with weight are needed) */
            int pos = 1;
            if (tv[0] >= 0 && tv[1] >= 0 && tv[2] >= 0) { const double *A = C->m.uv + 2 * tv[0], *B = C->m.uv + 2 * tv[1], *Cc = C->m.uv + 2 * tv[2];
                                                          pos = (B[0] - A[0]) * (Cc[1] - A[1]) - (B[1] - A[1]) * (Cc[0] - A[0]) > 0; }
            for (int k = 0; k < 3; k++) if (tw[k] <= 1e-12) { tw[k] = 0; tv[k] = tv[k] < 0 ? 0 : tv[k]; }
            if (pos) {
                for (int k = 0; k < 3; k++) { s[0] += tw[k] * (C->m.uv[2 * tv[k]] - C->m.uv0[2 * tv[k]]); s[1] += tw[k] * (C->m.uv[2 * tv[k] + 1] - C->m.uv0[2 * tv[k] + 1]); }
                ws = 1;
            }
        }
        if (ws > 1e-9) { dl[2 * v] = s[0] / ws; dl[2 * v + 1] = s[1] / ws; known[v] = 1; } else dl[2 * v] = dl[2 * v + 1] = 0;
    }
    /* unknown vertices: neighbour means, repeated */
    for (int pass = 0, prev = -1; pass < 4096; pass++) {
        memset(acc, 0, n2 * 8); memset(nn, 0, (size_t)m->nv * 4); int left = 0;
        for (int t = 0; t < m->nt; t++) { const int *v = m->tri + 3 * t; for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) if (a != b && known[v[b]]) { acc[2 * v[a]] += dl[2 * v[b]]; acc[2 * v[a] + 1] += dl[2 * v[b] + 1]; nn[v[a]]++; } }
        for (int v = 0; v < m->nv; v++) if (!known[v]) { if (nn[v]) { dl[2 * v] = acc[2 * v] / nn[v]; dl[2 * v + 1] = acc[2 * v + 1] / nn[v]; known[v] = 2; } else left++; }
        for (int v = 0; v < m->nv; v++) if (known[v] == 2) known[v] = 1;
        if (!left || left == prev) break;   /* (a component without a coarse vertex stays at the grid map: no progress) */
        prev = left;
    }
    for (size_t i = 0; i < n2; i++) m->uv[i] = m->uv0[i] + dl[i];
    long nb = count_flips(m, m->uv, bad), nb0 = nb; int rounds = 0;
    for (; nb && rounds < 200; rounds++) {   /* relax the vertices of bad triangles (and grow the set) */
        memset(acc, 0, n2 * 8); memset(nn, 0, (size_t)m->nv * 4);
        for (int t = 0; t < m->nt; t++) { const int *v = m->tri + 3 * t; for (int a = 0; a < 3; a++) if (bad[v[a]]) for (int b = 0; b < 3; b++) if (a != b) { acc[2 * v[a]] += dl[2 * v[b]]; acc[2 * v[a] + 1] += dl[2 * v[b] + 1]; nn[v[a]]++; } }
        for (int v = 0; v < m->nv; v++) if (bad[v] && nn[v]) { dl[2 * v] = acc[2 * v] / nn[v]; dl[2 * v + 1] = acc[2 * v + 1] / nn[v]; }
        for (size_t i = 0; i < n2; i++) m->uv[i] = m->uv0[i] + dl[i];
        nb = count_flips(m, m->uv, bad);   /* (the relaxed set only grows: bad marks are kept) */
        if (nb && rounds % 10 == 9) {   /* stuck: grow the relaxed region by one ring */
            uint8_t *g = calloc((size_t)m->nv, 1);
            for (int t = 0; t < m->nt; t++) { const int *v = m->tri + 3 * t; if (bad[v[0]] || bad[v[1]] || bad[v[2]]) g[v[0]] = g[v[1]] = g[v[2]] = 1; }
            memcpy(bad, g, (size_t)m->nv); free(g);
        }
    }
    double alpha = 1; long dropped = 0;
    if (nb && nb * 20 < m->nt) {   /* stuck triangles (mesh borders where the coarse lattice has no cell; folds of crushed inner wraps): dropped, up to 5% of the mesh. Scaling the whole
                                     prolongation down instead (below) undid the coarse levels: P3 inner strip chunks with 0.11% stuck stayed at aniso p50 1.3 */
        int k = 0;
        for (int t = 0; t < m->nt; t++) {
            const int *v = m->tri + 3 * t; const double *a = m->uv + 2 * v[0], *b = m->uv + 2 * v[1], *c = m->uv + 2 * v[2];
            const double *a0 = m->uv0 + 2 * v[0], *b0 = m->uv0 + 2 * v[1], *c0 = m->uv0 + 2 * v[2];
            const double d = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]), d0 = (b0[0] - a0[0]) * (c0[1] - a0[1]) - (b0[1] - a0[1]) * (c0[0] - a0[0]);
            if (!(d > 1e-3 * d0)) { dropped++; continue; }
            if (k != t) memmove(m->tri + 3 * k, m->tri + 3 * t, 3 * sizeof(int));
            k++;
        }
        m->nt = k; nb = 0;
    }
    if (nb) {   /* fallback: from the (flip-free) grid map toward the prolongation, capped before the first inversion */
        slim S; memset(&S, 0, sizeof S); S.nv = m->nv; S.nt = m->nt; S.tri = m->tri; S.nth = o->nthreads;
        alpha = fmin(1.0, 0.8 * max_step(&S, m->uv0, dl));
        for (size_t i = 0; i < n2; i++) m->uv[i] = m->uv0[i] + alpha * dl[i];
    }
    if (o->verbose) fprintf(stderr, "flatten: prolongate D %d -> %d: %ld bad triangles, %d relax rounds, %ld dropped, %ld left, scale %.3f\n", C->D, F->D, nb0, rounds, dropped, nb, alpha);
    free(dl); free(known); free(bad); free(acc); free(nn);
}

int flat_grid(const float *X, const float *Y, const float *Z, int w, int h, double step, const flat_opts *o, flat_mesh *out, flat_stats *before, flat_stats *after) {
    enum { MAXL = 12 }; level L[MAXL]; int nl = 0;
    int D = o->decim < 1 ? 1 : o->decim;
    if (build_level(X, Y, Z, w, h, step, D, o->max_edge, o->all_components, o->col_u, &L[0])) { free(L[0].lv); flat_mesh_free(&L[0].m); return -1; }
    nl = 1;
    while (nl < MAXL && L[nl - 1].m.nv > o->min_verts) {
        D *= 2; if ((w - 1) / D + 1 < 4 || (h - 1) / D + 1 < 4) break;
        if (build_level(X, Y, Z, w, h, step, D, o->max_edge, o->all_components, o->col_u, &L[nl])) { free(L[nl].lv); flat_mesh_free(&L[nl].m); break; }
        nl++;
    }
    if (before) flat_stats_of(L[0].m.P, L[0].m.tri, L[0].m.nt, L[0].m.uv0, before);
    int rc = 0;
    for (int l = nl - 1; l >= 0 && !rc; l--) {
        flat_mesh *m = &L[l].m;
        if (l < nl - 1) prolongate(&L[l + 1], &L[l], o);
        slim S; if (slim_init(&S, m->P, m->nv, m->tri, m->nt, o->nthreads)) { slim_free(&S); rc = -1; break; }
        S.x = m->uv; char tag[64]; snprintf(tag, sizeof tag, "level D%d", L[l].D);
        const double t0 = fnow(); const double E = slim_run(&S, o, l ? 2 * o->iters : o->iters, tag);
        if (o->verbose) fprintf(stderr, "flatten: %s done: energy %.5f in %.1fs\n", tag, E, fnow() - t0);
        slim_free(&S);
        if (!isfinite(E)) rc = -1;
    }
    for (int l = 1; l < nl; l++) { free(L[l].lv); flat_mesh_free(&L[l].m); }
    free(L[0].lv);
    if (rc) { flat_mesh_free(&L[0].m); return rc; }
    flat_mesh *m = &L[0].m;
    if (after) flat_stats_of(m->P, m->tri, m->nt, m->uv, after);
    /* Procrustes onto the grid map (rotation + translation), then the uv box's corner at 0 */
    double ca[2] = {0, 0}, cb[2] = {0, 0};
    for (int v = 0; v < m->nv; v++) for (int k = 0; k < 2; k++) { ca[k] += m->uv[2 * v + k]; cb[k] += m->uv0[2 * v + k]; }
    for (int k = 0; k < 2; k++) { ca[k] /= m->nv; cb[k] /= m->nv; }
    double sc = 0, ss = 0;
    for (int v = 0; v < m->nv; v++) { const double ax = m->uv[2 * v] - ca[0], ay = m->uv[2 * v + 1] - ca[1], bx = m->uv0[2 * v] - cb[0], by = m->uv0[2 * v + 1] - cb[1]; sc += ax * bx + ay * by; ss += ax * by - ay * bx; }
    const double th = atan2(ss, sc), c = cos(th), s = sin(th); double lo[2] = {INFINITY, INFINITY};
    for (int v = 0; v < m->nv; v++) {
        const double ax = m->uv[2 * v] - ca[0], ay = m->uv[2 * v + 1] - ca[1];
        m->uv[2 * v] = c * ax - s * ay; m->uv[2 * v + 1] = s * ax + c * ay;
        for (int k = 0; k < 2; k++) if (m->uv[2 * v + k] < lo[k]) lo[k] = m->uv[2 * v + k];
    }
    for (int v = 0; v < m->nv; v++) for (int k = 0; k < 2; k++) m->uv[2 * v + k] -= lo[k];
    if (o->verbose) fprintf(stderr, "flatten: aligned to the grid map (rotation %.3f deg)\n", th * 180 / M_PI);
    *out = *m;
    return 0;
}

/* ---- resampling on a regular uv grid ---- */
typedef struct { const flat_mesh *m; int w; double os; int ow, oh; float *gu, *gv, *ox, *oy, *oz; } rjob;
static void k_raster(void *c, long lo, long hi, int tid) {
    rjob *R = c; const flat_mesh *m = R->m;
    for (long t = lo; t < hi; t++) {
        const int *v = m->tri + 3 * t; double U[3], V[3];
        for (int k = 0; k < 3; k++) { U[k] = m->uv[2 * v[k]] / R->os; V[k] = m->uv[2 * v[k] + 1] / R->os; }
        const double det = (U[1] - U[0]) * (V[2] - V[0]) - (V[1] - V[0]) * (U[2] - U[0]); if (!(fabs(det) > 1e-300)) continue;
        const int x0 = (int)ceil(fmin(U[0], fmin(U[1], U[2])) - 1e-9), x1 = (int)floor(fmax(U[0], fmax(U[1], U[2])) + 1e-9);
        const int y0 = (int)ceil(fmin(V[0], fmin(V[1], V[2])) - 1e-9), y1 = (int)floor(fmax(V[0], fmax(V[1], V[2])) + 1e-9);
        for (int y = y0 < 0 ? 0 : y0; y <= y1 && y < R->oh; y++) for (int x = x0 < 0 ? 0 : x0; x <= x1 && x < R->ow; x++) {
            const double l1 = ((x - U[0]) * (V[2] - V[0]) - (y - V[0]) * (U[2] - U[0])) / det, l2 = ((U[1] - U[0]) * (y - V[0]) - (V[1] - V[0]) * (x - U[0])) / det, l0 = 1 - l1 - l2;
            const double eps = -1e-7; if (l0 < eps || l1 < eps || l2 < eps) continue;
            const double L[3] = {l0, l1, l2}; double gc = 0, gr = 0, p[3] = {0, 0, 0};
            for (int k = 0; k < 3; k++) { gc += L[k] * (m->gi[v[k]] % R->w); gr += L[k] * (m->gi[v[k]] / R->w); for (int d = 0; d < 3; d++) p[d] += L[k] * m->P[3 * v[k] + d]; }
            const size_t o = (size_t)y * R->ow + x;
            R->gu[o] = (float)gc; R->gv[o] = (float)gr; R->ox[o] = (float)p[0]; R->oy[o] = (float)p[1]; R->oz[o] = (float)p[2];
        }
    }
}
int flat_resample(const flat_mesh *m, const float *X, const float *Y, const float *Z, int w, int h, double ostep, int nthreads,
                  int *ow, int *oh, float **gu, float **gv, float **ox, float **oy, float **oz) {
    double hi[2] = {0, 0}; for (int v = 0; v < m->nv; v++) for (int k = 0; k < 2; k++) if (m->uv[2 * v + k] > hi[k]) hi[k] = m->uv[2 * v + k];
    const int W = (int)floor(hi[0] / ostep) + 1, H = (int)floor(hi[1] / ostep) + 1; const size_t n = (size_t)W * H;
    float *a[5]; for (int k = 0; k < 5; k++) { a[k] = malloc(n * 4); if (!a[k]) { for (int q = 0; q <= k; q++) free(a[q]); return -1; } for (size_t i = 0; i < n; i++) a[k][i] = -1.f; }
    rjob R = {m, w, ostep, W, H, a[0], a[1], a[2], a[3], a[4]};
    parfor(m->nt, nthreads, k_raster, &R);
    for (size_t i = 0; i < n; i++) {   /* the source grid's bilinear surface where the 4 corners are valid (as ufsm render samples it) */
        if (a[0][i] < 0) continue;
        int c0 = (int)a[0][i], r0 = (int)a[1][i]; if (c0 > w - 2) c0 = w - 2; if (r0 > h - 2) r0 = h - 2; if (c0 < 0 || r0 < 0) continue;
        const double fu = a[0][i] - c0, fv = a[1][i] - r0; const size_t g = (size_t)r0 * w + c0;
        if (!gvalid(X, Y, Z, g) || !gvalid(X, Y, Z, g + 1) || !gvalid(X, Y, Z, g + w) || !gvalid(X, Y, Z, g + w + 1)) continue;
        const float *S[3] = {X, Y, Z};
        for (int d = 0; d < 3; d++) a[2 + d][i] = (float)((1 - fv) * ((1 - fu) * S[d][g] + fu * S[d][g + 1]) + fv * ((1 - fu) * S[d][g + w] + fu * S[d][g + w + 1]));
    }
    *ow = W; *oh = H; *gu = a[0]; *gv = a[1]; *ox = a[2]; *oy = a[3]; *oz = a[4];
    return 0;
}

void flat_largest_rect(const uint8_t *valid, int w, int h, int *x0, int *y0, int *rw, int *rh) { flat_largest_rect_aspect(valid, w, h, 0, x0, y0, rw, rh); }
/* the largest all-valid rectangle with max(side) / min(side) <= maxasp (<= 0: any). Every maximal valid rectangle W x H appears as a
   stack pop of the row histogram scan; the best capped rectangle lies inside one of them, and min(W, a H) x min(H, a W) (centred) is
   the largest capped rectangle inside W x H, so clipping each candidate gives the exact optimum. */
void flat_largest_rect_aspect(const uint8_t *valid, int w, int h, double maxasp, int *x0, int *y0, int *rw, int *rh) {
    int *ht = calloc((size_t)w + 1, sizeof(int)), *st = malloc(((size_t)w + 2) * sizeof(int)); long best = 0; *x0 = *y0 = *rw = *rh = 0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) ht[x] = valid[(size_t)y * w + x] ? ht[x] + 1 : 0;
        ht[w] = 0; int sp = 0;
        for (int x = 0; x <= w; x++) {
            while (sp && ht[st[sp - 1]] >= ht[x]) {
                const int hh = ht[st[--sp]], left = sp ? st[sp - 1] + 1 : 0, W = x - left; int cw = W, ch = hh;
                if (maxasp > 0) { if (cw > (int)(maxasp * hh)) cw = (int)(maxasp * hh); if (ch > (int)(maxasp * W)) ch = (int)(maxasp * W); }
                const long a = (long)cw * ch;
                if (a > best) { best = a; *x0 = left + (W - cw) / 2; *rw = cw; *rh = ch; *y0 = y - hh + 1 + (hh - ch) / 2; }
            }
            st[sp++] = x;
        }
    }
    free(ht); free(st);
}

/* stats of a vertex grid's own map (column step, row step): all triangles of the grid (the same edge filter as the mesh) */
static void grid_stats(const float *X, const float *Y, const float *Z, int w, int h, double step, double max_edge, flat_stats *s) {
    level L; if (build_level(X, Y, Z, w, h, step, 1, max_edge, 0, nullptr, &L)) { memset(s, 0, sizeof *s); free(L.lv); flat_mesh_free(&L.m); return; }
    flat_stats_of(L.m.P, L.m.tri, L.m.nt, L.m.uv0, s); free(L.lv); flat_mesh_free(&L.m);
}

/* enclosed holes (4-connected invalid components not touching the grid's border) of at most maxv vertices filled: every array a[0..4]
   (x, y, z, gu, gv; invalid: z < 0) Laplace-interpolated from the hole's rim (Gauss-Seidel from the rim mean); filled[i] = 1 there.
   Returns the filled vertex count. */
long flat_fill_holes(float *a[5], int w, int h, long maxv, uint8_t *filled) {
    const size_t n = (size_t)w * h; memset(filled, 0, n);
    if (maxv <= 0) return 0;
    int8_t *lab = malloc(n); int *q = malloc(n * sizeof(int)); long nf = 0;
    for (size_t i = 0; i < n; i++) lab[i] = a[2][i] >= 0 ? 1 : 0;   /* 1 valid, 0 unvisited invalid, 2 visited */
    for (size_t s = 0; s < n; s++) {
        if (lab[s] != 0) continue;
        long qh = 0, qt = 0; int border = 0; q[qt++] = (int)s; lab[s] = 2;
        while (qh < qt) {
            const int i = q[qh++], x = i % w, y = i / w; if (x == 0 || y == 0 || x == w - 1 || y == h - 1) border = 1;
            const int nb[4] = {x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, y > 0 ? i - w : -1, y < h - 1 ? i + w : -1};
            for (int k = 0; k < 4; k++) if (nb[k] >= 0 && lab[nb[k]] == 0) { lab[nb[k]] = 2; q[qt++] = nb[k]; }
        }
        if (border || qt > maxv) continue;
        double rim[5] = {0}; long nr = 0;   /* start every hole vertex at the rim mean */
        for (long k = 0; k < qt; k++) { const int i = q[k]; const int nb[4] = {i - 1, i + 1, i - w, i + w};
                                        for (int j = 0; j < 4; j++) if (lab[nb[j]] == 1) { for (int c = 0; c < 5; c++) rim[c] += a[c][nb[j]]; nr++; } }
        for (long k = 0; k < qt; k++) for (int c = 0; c < 5; c++) a[c][q[k]] = (float)(rim[c] / (nr ? nr : 1));
        for (int it = 0; it < 5000; it++) {   /* (interior: all 4 neighbours exist; neighbours are valid or this hole) */
            double ch = 0;
            for (long k = 0; k < qt; k++) {
                const int i = q[k]; const int nb[4] = {i - 1, i + 1, i - w, i + w};
                for (int c = 0; c < 5; c++) { const float v = 0.25f * (a[c][nb[0]] + a[c][nb[1]] + a[c][nb[2]] + a[c][nb[3]]); ch = fmax(ch, fabs(v - a[c][i])); a[c][i] = v; }
            }
            if (ch < 1e-3) break;
        }
        for (long k = 0; k < qt; k++) filled[q[k]] = 1;
        nf += qt;
    }
    free(lab); free(q); return nf;
}

/* ---- CLI ---- */
static const char *fopt(int argc, char **argv, const char *name, const char *def) { for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1]; return def; }
static double meta_num(const char *m, const char *key, double def) { const char *q = m ? strstr(m, key) : nullptr; if (!q) return def; q += strlen(key); while (*q == ' ' || *q == ':' || *q == '[') q++; return atof(q); }

/* ufsm flatten <wraps-dir> <out-dir> [--wraps a,b,..] [--decim 1] [--iters 30] [--cg 200] [--min-verts 20000] [--out-step S]
   [--crop rect|none] [--max-aspect A] [--rects N] [--min-cm2 5] [--fill-holes 5000] [--max-edge 3] [--threads 16] [--sfc ERR] [--quiet 1]
   Every tifxyz wrap of <wraps-dir> (directories or .sfc) SLIM-flattened (src/flatten.h), resampled on a regular isometric uv grid of
   pitch --out-step native voxels (default: the input's step) and written as <out-dir>/<wrap> (tifxyz: x / y / z, meta.json, plus gu.tif /
   gv.tif = the source grid column / row of each vertex, for ufsm flatten-warp), cropped (--crop rect, default) to the largest fully valid
   rectangle after enclosed holes of at most --fill-holes vertices (tears: dropped over-stretched triangles, invalid vertices) are
   Laplace-filled (flat_fill_holes); <out-dir>/<wrap>_mask.jpg shows the flat grid (valid grey, filled orange, crop white). ufsm render of that tifxyz is the flattened CT / ink render. Prints per wrap the distortion of the input grid map, of the
   SLIM mesh before / after, and of the written (cropped) grid, also into <out-dir>/<wrap>/flatten.txt.
   --max-aspect A: the rectangle's long side <= A x its short side (largest such, flat_largest_rect_aspect); --rects N: up to N
   non-overlapping rectangles, greedily (each the largest left after the previous are cut out; after the first only >= --min-cm2),
   written as <out-dir>/<wrap>_r0, _r1, .. (N = 1: <out-dir>/<wrap>). --crop none: the whole flat grid is written (<out-dir>/<wrap>,
   invalid vertices z = -1, "= A cm2" = its valid area) and the rectangles go to <out-dir>/<wrap>/rects.tsv (rect x y w h cm2 aniso_p50,
   index boxes on that grid) and as lines "<wrap>_r<i>: rect x,y wxh = A cm2" + "<wrap>_r<i> output grid ..": re-cropping never
   needs a re-flatten. */
int cmd_flatten(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm flatten <wraps-dir> <out-dir> [--wraps a,b,..] [--decim 1] [--iters 30] [--cg 200] [--min-verts 20000] [--out-step S] [--crop rect|none] [--max-aspect A] [--rects N] [--min-cm2 5] [--fill-holes 5000] [--max-edge 3] [--threads 16] [--gpu 1] [--sfc ERR] [--quiet 1]\n"); return 2; }
    flat_opts o = flat_opts_default();
    o.decim = atoi(fopt(argc, argv, "--decim", "1")); o.iters = atoi(fopt(argc, argv, "--iters", "30")); o.cg_iters = atoi(fopt(argc, argv, "--cg", "200"));
    o.min_verts = atoi(fopt(argc, argv, "--min-verts", "20000")); o.max_edge = atof(fopt(argc, argv, "--max-edge", "3")); o.nthreads = atoi(fopt(argc, argv, "--threads", "16"));
    o.verbose = !atoi(fopt(argc, argv, "--quiet", "0")); o.gpu = atoi(fopt(argc, argv, "--gpu", "1"));
    const double ostep_opt = atof(fopt(argc, argv, "--out-step", "0")); const int crop = strcmp(fopt(argc, argv, "--crop", "rect"), "none") != 0;
    const char *wl = fopt(argc, argv, "--wraps", nullptr); char *want = nullptr; if (wl) { want = malloc(strlen(wl) + 3); sprintf(want, ",%s,", wl); }
    const double sfc = atof(fopt(argc, argv, "--sfc", "0")); const long maxhole = atol(fopt(argc, argv, "--fill-holes", "5000"));
    const double maxasp = atof(fopt(argc, argv, "--max-aspect", "0")), mincm2 = atof(fopt(argc, argv, "--min-cm2", "5"));
    int nrect = atoi(fopt(argc, argv, "--rects", "1")); if (nrect < 1) nrect = 1; if (nrect > 16) nrect = 16;
    DIR *dd = opendir(argv[2]); if (!dd) { fprintf(stderr, "flatten: cannot open %s\n", argv[2]); return 1; }
    char cmd[1500]; snprintf(cmd, sizeof cmd, "mkdir -p '%s'", argv[3]); if (system(cmd)) return 1;
    struct dirent *de; int nw = 0, rc = 0;
    while ((de = readdir(dd))) {
        if (de->d_name[0] == '.') continue;
        char nm[300]; snprintf(nm, sizeof nm, "%s", de->d_name); { const size_t l = strlen(nm); if (l > 4 && !strcmp(nm + l - 4, ".sfc")) nm[l - 4] = 0; }
        if (want) { char k[310]; snprintf(k, sizeof k, ",%s,", nm); if (!strstr(want, k)) continue; }
        char s[1400], p[1500]; snprintf(s, sizeof s, "%s/%s", argv[2], de->d_name); snprintf(p, sizeof p, "%s/meta.json", s);
        if (!tfx_exists(p)) continue;
        char *meta = tfx_meta_read(p);
        const double sc = meta_num(meta, "\"scale\"", 1), vum = meta_num(meta, "\"voxel_size_um\"", 1); const int wid = (int)meta_num(meta, "\"winding\"", -1);
        const double step = sc > 0 ? 1 / sc : 1, ostep = ostep_opt > 0 ? ostep_opt : step; free(meta);
        static const char *ax[3] = {"x", "y", "z"}; float *A[3] = {0}; int w = 0, h = 0, ok = 1;
        for (int k = 0; k < 3 && ok; k++) { int w2, h2; snprintf(p, sizeof p, "%s/%s.tif", s, ax[k]); A[k] = img_read_f32(p, &w2, &h2); ok = A[k] && (!k || (w2 == w && h2 == h)); w = w2; h = h2; }
        if (!ok) { fprintf(stderr, "flatten: cannot read %s\n", s); rc = 1; for (int k = 0; k < 3; k++) free(A[k]); continue; }
        const double t0 = fnow();
        fprintf(stderr, "flatten: %s %dx%d step %g\n", nm, w, h, step);
        flat_stats gin, sb, sa, gout; grid_stats(A[0], A[1], A[2], w, h, step, o.max_edge, &gin);
        flat_mesh m; memset(&m, 0, sizeof m);
        if (flat_grid(A[0], A[1], A[2], w, h, step, &o, &m, &sb, &sa)) { fprintf(stderr, "flatten: %s: no mesh / SLIM failed\n", nm); rc = 1; for (int k = 0; k < 3; k++) free(A[k]); continue; }
        int ow, oh; float *gu, *gv, *ox, *oy, *oz;
        if (flat_resample(&m, A[0], A[1], A[2], w, h, ostep, o.nthreads, &ow, &oh, &gu, &gv, &ox, &oy, &oz)) { fprintf(stderr, "flatten: out of memory\n"); return 1; }
        int RX[16], RY[16], RW[16], RH[16], nr = 0; long nvalid = 0;
        float *ga[5] = {ox, oy, oz, gu, gv};
        uint8_t *fm = malloc((size_t)ow * oh); long nfill = flat_fill_holes(ga, ow, oh, maxhole, fm);
        uint8_t *vm = malloc((size_t)ow * oh); for (size_t i = 0; i < (size_t)ow * oh; i++) nvalid += vm[i] = oz[i] >= 0;
        const double cm2px = vum * vum * 1e-8 * ostep * ostep;
        {   /* greedy: the largest (aspect-capped) valid rectangle, taken out of the mask, again; rectangles after the first need --min-cm2 */
            uint8_t *rm = malloc((size_t)ow * oh); memcpy(rm, vm, (size_t)ow * oh);
            for (int k = 0; k < nrect; k++) {
                int x0, y0, rw, rh; flat_largest_rect_aspect(rm, ow, oh, maxasp, &x0, &y0, &rw, &rh);
                if (rw < 2 || rh < 2 || (k > 0 && (double)(rw - 1) * (rh - 1) * cm2px < mincm2)) break;
                RX[nr] = x0; RY[nr] = y0; RW[nr] = rw; RH[nr] = rh; nr++;
                for (int y = y0; y < y0 + rh; y++) memset(rm + (size_t)y * ow + x0, 0, (size_t)rw);
            }
            free(rm);
        }
        {   /* <out>/<wrap>_mask.jpg: the flat grid's valid vertices (grey), the crop rectangles (white), one pixel per vertex */
            uint8_t *rgb = malloc((size_t)ow * oh * 3);
            for (int y = 0; y < oh; y++) for (int x = 0; x < ow; x++) {
                const size_t i = (size_t)y * ow + x; int in = 0;
                for (int k = 0; k < nr && !in; k++) in = x >= RX[k] && x < RX[k] + RW[k] && y >= RY[k] && y < RY[k] + RH[k];
                rgb[3 * i] = rgb[3 * i + 1] = rgb[3 * i + 2] = vm[i] ? (in ? 230 : 110) : 0; if (vm[i] && !in) rgb[3 * i + 2] = 160;
                if (fm[i]) { rgb[3 * i] = 255; rgb[3 * i + 1] = in ? 150 : 90; rgb[3 * i + 2] = 0; }   /* filled hole (orange) */
            }
            for (int k = 0; k < nr; k++) for (int y = RY[k]; y < RY[k] + RH[k]; y++) for (int x = RX[k]; x < RX[k] + RW[k]; x++)   /* rectangle outlines (black) */
                if (y == RY[k] || y == RY[k] + RH[k] - 1 || x == RX[k] || x == RX[k] + RW[k] - 1) { const size_t i = (size_t)y * ow + x; rgb[3 * i] = rgb[3 * i + 1] = rgb[3 * i + 2] = 0; }
            snprintf(p, sizeof p, "%s/%s_mask.jpg", argv[3], nm); if (jpeg_write_rgb(p, ow, oh, rgb, 85)) fprintf(stderr, "flatten: cannot write %s\n", p);
            free(rgb);
        }
        free(vm); free(fm);
        if (!nr && crop) { fprintf(stderr, "flatten: %s: no valid rectangle\n", nm); rc = 1; }
        const int nout = crop ? nr : 1;   /* --crop none: one output, the whole flat grid (invalid vertices z = -1), + rects.tsv */
        for (int ri = 0; ri < nout; ri++) {   /* --rects > 1: <out>/<wrap>_r<i>, else <out>/<wrap> */
            const int cx = crop ? RX[ri] : 0, cy = crop ? RY[ri] : 0, cw = crop ? RW[ri] : ow, ch = crop ? RH[ri] : oh;
            char rn[320]; if (crop && nrect > 1) snprintf(rn, sizeof rn, "%s_r%d", nm, ri); else snprintf(rn, sizeof rn, "%s", nm);
            float *C[5]; float *src[5] = {ox, oy, oz, gu, gv};
            for (int k = 0; k < 5; k++) { C[k] = malloc((size_t)cw * ch * 4); for (int y = 0; y < ch; y++) memcpy(C[k] + (size_t)y * cw, src[k] + (size_t)(cy + y) * ow + cx, (size_t)cw * 4); }
            grid_stats(C[0], C[1], C[2], cw, ch, ostep, o.max_edge, &gout);
            char od[1400]; snprintf(od, sizeof od, "%s/%s", argv[3], rn);
            if (sfc > 0) tifxyz_set_sfc(sfc);
            if (tifxyz_write(od, cw, ch, C[0], C[1], C[2], ostep, vum, wid, "ufsm flatten (SLIM symmetric Dirichlet, isometric uv grid, largest valid rectangle)")) { fprintf(stderr, "flatten: cannot write %s\n", od); rc = 1; }
            tifxyz_set_sfc(0);
            snprintf(cmd, sizeof cmd, "mkdir -p '%s'", od); if (system(cmd)) rc = 1;
            snprintf(p, sizeof p, "%s/gu.tif", od); if (tiffw_write(p, cw, ch, 3, C[3], 1)) rc = 1;
            snprintf(p, sizeof p, "%s/gv.tif", od); if (tiffw_write(p, cw, ch, 3, C[4], 1)) rc = 1;
            const double um2 = vum * vum * 1e-8;
            printf("%s: input %dx%d step %g -> flat %dx%d step %g (%ld valid, %ld in filled holes), crop %d,%d %dx%d = %.2f cm2 (%.1f%% of the flat valid area), %.1fs\n",
                   rn, w, h, step, ow, oh, ostep, nvalid, nfill, cx, cy, cw, ch, crop ? (double)(cw - 1) * (ch - 1) * ostep * ostep * um2 : nvalid * cm2px,
                   crop ? 100.0 * cw * ch / (nvalid ? nvalid : 1) : 100.0, fnow() - t0);
            if (!crop) {   /* the rectangles as index boxes on this grid: <out>/<wrap>/rects.tsv + one line each "<wrap>_r<i>: rect x,y wxh = A cm2" */
                snprintf(p, sizeof p, "%s/rects.tsv", od); FILE *rf = fopen(p, "w");
                if (rf) fprintf(rf, "rect\tx\ty\tw\th\tcm2\taniso_p50\n");
                for (int k = 0; k < nr; k++) {
                    float *R3[3]; for (int c = 0; c < 3; c++) { R3[c] = malloc((size_t)RW[k] * RH[k] * 4); for (int y = 0; y < RH[k]; y++) memcpy(R3[c] + (size_t)y * RW[k], C[c] + (size_t)(RY[k] + y) * ow + RX[k], (size_t)RW[k] * 4); }
                    flat_stats rs; grid_stats(R3[0], R3[1], R3[2], RW[k], RH[k], ostep, o.max_edge, &rs); for (int c = 0; c < 3; c++) free(R3[c]);
                    const double a = (double)(RW[k] - 1) * (RH[k] - 1) * cm2px;
                    if (rf) fprintf(rf, "%s_r%d\t%d\t%d\t%d\t%d\t%.2f\t%.3f\n", nm, k, RX[k], RY[k], RW[k], RH[k], a, rs.aniso_p50);
                    printf("%s_r%d: rect %d,%d %dx%d = %.2f cm2\n", nm, k, RX[k], RY[k], RW[k], RH[k], a);
                    char lb2[400]; snprintf(lb2, sizeof lb2, "%s_r%d output grid", nm, k); flat_stats_print(stdout, lb2, &rs);
                }
                if (rf) fclose(rf);
            }
            char lb[400];
            snprintf(lb, sizeof lb, "%s input grid", rn); flat_stats_print(stdout, lb, &gin);
            snprintf(lb, sizeof lb, "%s mesh before", rn); flat_stats_print(stdout, lb, &sb);
            snprintf(lb, sizeof lb, "%s mesh after", rn); flat_stats_print(stdout, lb, &sa);
            snprintf(lb, sizeof lb, "%s output grid", rn); flat_stats_print(stdout, lb, &gout);
            fflush(stdout);
            snprintf(p, sizeof p, "%s/flatten.txt", od); FILE *f = fopen(p, "w");
            if (f) {
                fprintf(f, "input %dx%d step %g -> flat %dx%d step %g, crop %d,%d %dx%d\n", w, h, step, ow, oh, ostep, cx, cy, cw, ch);
                flat_stats_print(f, "input grid", &gin); flat_stats_print(f, "mesh before", &sb); flat_stats_print(f, "mesh after", &sa); flat_stats_print(f, "output grid", &gout);
                fclose(f);
            }
            for (int k = 0; k < 5; k++) free(C[k]);
            nw++;
        }
        free(gu); free(gv); free(ox); free(oy); free(oz); flat_mesh_free(&m); for (int k = 0; k < 3; k++) free(A[k]);
    }
    closedir(dd); free(want);
    fprintf(stderr, "flatten: %d wraps written to %s\n", nw, argv[3]);
    return rc;
}

/* ufsm flatten-warp <in.jpg|in.tif> <flat-wrap-dir> <out.jpg|out.tif> --src-px P [--scale S] [--quality 90]
   An image rendered on the ORIGINAL wrap's grid (P pixels per source grid step, e.g. render --scale 4 of a step-4 wrap halved by
   ink-compare: P = 8) resampled onto the flattened wrap (ufsm flatten's gu / gv) at S pixels per native voxel (the size ufsm render
   --scale S gives that flat tifxyz), bilinear; 0 outside. For renders whose volume is no longer available (published ink). */
int cmd_flatten_warp(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm flatten-warp <in.jpg|in.tif> <flat-wrap-dir> <out.jpg|out.tif> --src-px P [--scale S] [--quality 90]\n"); return 2; }
    const double P = atof(fopt(argc, argv, "--src-px", "0")), S = atof(fopt(argc, argv, "--scale", "1"));
    if (!(P > 0) || !(S > 0)) { fprintf(stderr, "flatten-warp: --src-px P > 0, --scale S > 0\n"); return 2; }
    const char *in = argv[2]; const size_t il = strlen(in); int iw, ih; uint8_t *im = nullptr;
    if (il > 4 && (!strcmp(in + il - 4, ".jpg") || !strcmp(in + il - 5, ".jpeg"))) {
        uint8_t *rgb = jpeg_read_rgb(in, &iw, &ih); if (rgb) { im = malloc((size_t)iw * ih); for (size_t i = 0; i < (size_t)iw * ih; i++) im[i] = (uint8_t)((rgb[3 * i] + rgb[3 * i + 1] + rgb[3 * i + 2] + 1) / 3); free(rgb); }
    } else im = img_read_u8(in, &iw, &ih);
    if (!im) { fprintf(stderr, "flatten-warp: cannot read %s\n", in); return 1; }
    char p[1500]; int fw, fh, w2, h2;
    snprintf(p, sizeof p, "%s/gu.tif", argv[3]); float *gu = img_read_f32(p, &fw, &fh);
    snprintf(p, sizeof p, "%s/gv.tif", argv[3]); float *gv = img_read_f32(p, &w2, &h2);
    snprintf(p, sizeof p, "%s/meta.json", argv[3]); char *meta = tfx_meta_read(p); const double sc = meta_num(meta, "\"scale\"", 1); free(meta);
    if (!gu || !gv || w2 != fw || h2 != fh || fw < 2 || fh < 2) { fprintf(stderr, "flatten-warp: %s has no gu / gv grids\n", argv[3]); return 1; }
    const double gs = (sc > 0 ? 1 / sc : 1) * S;   /* output pixels per flat grid step */
    const int ow = render_out_px(fw, gs), oh = render_out_px(fh, gs); uint8_t *out = calloc((size_t)ow * oh, 1);
    for (int y = 0; y < oh; y++) for (int x = 0; x < ow; x++) {
        const double fx = x / gs, fy = y / gs; int c0 = (int)fx, r0 = (int)fy; if (c0 > fw - 2) c0 = fw - 2; if (r0 > fh - 2) r0 = fh - 2;
        const double a = fx - c0, b = fy - r0; const size_t g = (size_t)r0 * fw + c0;
        if (gu[g] < 0 || gu[g + 1] < 0 || gu[g + fw] < 0 || gu[g + fw + 1] < 0) continue;
        const double su = P * ((1 - b) * ((1 - a) * gu[g] + a * gu[g + 1]) + b * ((1 - a) * gu[g + fw] + a * gu[g + fw + 1]));
        const double sv = P * ((1 - b) * ((1 - a) * gv[g] + a * gv[g + 1]) + b * ((1 - a) * gv[g + fw] + a * gv[g + fw + 1]));
        if (su < 0 || sv < 0 || su > iw - 1 || sv > ih - 1) continue;
        int u0 = (int)su, v0 = (int)sv; if (u0 > iw - 2) u0 = iw - 2; if (v0 > ih - 2) v0 = ih - 2;
        const double e = su - u0, f = sv - v0; const size_t q = (size_t)v0 * iw + u0;
        out[(size_t)y * ow + x] = (uint8_t)lround((1 - f) * ((1 - e) * im[q] + e * im[q + 1]) + f * ((1 - e) * im[q + iw] + e * im[q + iw + 1]));
    }
    const char *o = argv[4]; const size_t ol = strlen(o); int rc = 0;
    if (ol > 4 && !strcmp(o + ol - 4, ".tif")) rc = tiffw_write(o, ow, oh, 1, out, 1);
    else { uint8_t *rgb = malloc((size_t)ow * oh * 3); for (size_t i = 0; i < (size_t)ow * oh; i++) rgb[3 * i] = rgb[3 * i + 1] = rgb[3 * i + 2] = out[i];
           rc = jpeg_write_rgb(o, ow, oh, rgb, atoi(fopt(argc, argv, "--quality", "90"))); free(rgb); }
    if (rc) fprintf(stderr, "flatten-warp: cannot write %s\n", o); else fprintf(stderr, "flatten-warp: %s %dx%d\n", o, ow, oh);
    free(im); free(gu); free(gv); free(out);
    return rc ? 1 : 0;
}
