/* ufsm fit-join <out-set> <fields-dir> [--blend B] [--min-conf C] [--samples N] [--ref MODEL [--ref-w 1] [--chain-w 1] [--ref-samples N]]
                 <slab-model> <z0> <z1> [<slab-model> <z0> <z1> ...]
   Independent z-slab fits of one field box -> one fit set (src/fit.h fit_set) with a consistent integer gauge, label-free.
   <fields-dir> is the whole box the slabs were cropped from (the same y / x origin and size: mkcrop of z rows only); each slab model
   (ufsm fit --out) covers its own z rows; <z0> <z1> is the slab's interior in box-local native z of <fields-dir> (the interiors
   partition the box, the slab models extend past them: the overlaps). In z order, each slab's integer gauge is matched to the
   previous (already matched) slab on the cells of their overlap that have material and phase confidence >= --min-conf (0.3):
   shift = the mode of round(W_k - W_k-1), then W_k -= shift. Per boundary it prints the shift and the agreement (share of the
   overlap cells whose rounded difference is the mode, and whose difference is within 0.25 turn of it): a slab-to-slab consistency
   measure. The global model of the set is the box's geometry and axis with the slabs' knots (pitch / offset at each knot taken
   from the slab whose interior holds it) and no lattice; W is blended linearly over +- B native voxels (default 16) around each
   interior boundary (fit_set_W). Output: <out-set>/set.json, global/, slabNNN/ (ufsm wraps / fit --model read it).
   --ref MODEL (a model or set over any box holding this one, e.g. a whole-scroll coarse fit or count-field --model: one global
   gauge): the integers come from it instead of the chain. Per slab the histogram of round(W_slab - W_ref - c) over sampled cells of
   the middle (<= 16) field rows of its interior, c the global sub-turn bias (the circular mean of W_slab - W_ref over every slab);
   the gauges S_k minimise ref-w x sum_k -log P_ref,k(S_k) + chain-w x sum_k -log P_overlap,k(S_k - S_k-1) exactly (dynamic
   programming over the ref mode +- 12): a bad boundary cannot shift every slab above it, and --chain-w 0 is the ref alone. Prints
   per slab the ref share at the chosen gauge and per boundary the integer agreement of the overlap cells at the chosen gauges.
   --ref-rows N: the ref histograms over N field rows around the interior's middle (default 16; 0 = the whole interior).
   --global 1 (opt-in): a joint solve over the overlap graph (fit_join_solve): nodes the slabs, edges every pair of slab boxes that
   overlap (consecutive ones, and longer ones where a box reaches past its neighbour), each edge's cost of an offset d the robust
   chain-w x q x min(cap, -log P(d) + log P(mode)) of its overlap histogram, q = min(1, overlap cells / --min-cells) (thin overlaps
   count less), the ref (with --ref) as an anchor prior per slab of the same form (q from its samples x 10). Exact DP over the
   consecutive edges, then a local search over all edges (single-slab and suffix shifts). --cap (nats, default 3) bounds what one
   bad overlap or ref histogram can cost, so the others outvote it. Prints the slabs whose gauges changed against the chain and
   against the default join. */
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fit.h"

static double fj_knot(const float *v, int nk, int dz, double z) {   /* (fit.c knot_lerp) */
    double k = z / dz; if (k < 0) k = 0; if (k > nk - 1) k = nk - 1;
    int k0 = (int)k; if (k0 > nk - 2) k0 = nk - 2 < 0 ? 0 : nk - 2;
    const double t = nk > 1 ? k - k0 : 0;
    return v[k0] + t * (v[nk > 1 ? k0 + 1 : k0] - v[k0]);
}

static int cmp_long(const void *a, const void *b) { const long x = *(const long *)a, y = *(const long *)b; return (x > y) - (x < y); }

/* a histogram of rounded differences: distinct values v (ascending), their counts c and near counts cn (|d - v| < 0.25) */
typedef struct { long *v; long long *c, *cn; int n; long long tot; } fj_hist;
static void fj_hist_free(fj_hist *H) { free(H->v); free(H->c); free(H->cn); memset(H, 0, sizeof *H); }
static long fj_mode(const fj_hist *H) { long m = 0; long long b = 0; for (int i = 0; i < H->n; i++) if (H->c[i] > b) { b = H->c[i]; m = H->v[i]; } return m; }
static long long fj_count(const fj_hist *H, long v, long long *near) {
    int lo = 0, hi = H->n;
    while (lo < hi) { const int m = (lo + hi) / 2; if (H->v[m] < v) lo = m + 1; else hi = m; }
    const int hit = lo < H->n && H->v[lo] == v;
    if (near) *near = hit ? H->cn[lo] : 0;
    return hit ? H->c[lo] : 0;
}

/* the differences at sampled cells, chunks of 4096 samples [c0, c1): boundary (b): W_b - W_a; ref: W_a - W_ref - bias */
typedef struct { const fit_model *a, *b; double za, zb, h; int fz0; size_t HW; int H, W; const size_t *iv; double *dv; size_t n; const fit_set *ref; const double *ro; double bias; } fj_job;
static void fj_rows(void *ctx, int c0, int c1, int t) {
    const fj_job *J = ctx; (void)t;
    for (size_t q = (size_t)c0 * 4096; q < (size_t)c1 * 4096 && q < J->n; q++) {
        const size_t i = J->iv[q]; const int zz = (int)(i / J->HW), yy = (int)((i / J->W) % J->H), xx = (int)(i % J->W);
        const double z = (J->fz0 + zz + 0.5) * J->h, y = (yy + 0.5) * J->h, x = (xx + 0.5) * J->h;
        if (J->b) J->dv[q] = fit_W(J->b, z - J->zb, y, x) - fit_W(J->a, z - J->za, y, x);
        else J->dv[q] = fit_W(J->a, z - J->za, y, x) - fit_set_W(J->ref, z + J->ro[0], y + J->ro[1], x + J->ro[2]) - J->bias;
    }
}

/* sample the cells of field rows [fz0, fz1) with material and phase confidence >= minc (about nsamp of them) and histogram the
   rounded differences into H (nullptr: none); ph / phw (optional): the confidence-weighted phasor sum of the differences and the weight */
static int fj_sample(const char *fdir, int fz0, int fz1, double minc, long long nsamp, const fit_model *a, double za, const fit_model *b, double zb,
                     const fit_set *ref, const double *ro, double bias, double h, fj_hist *H, double complex *ph, double *phw) {
    fit_fields_buf f; if (fit_fields_load(fdir, fz0, fz1, 1, &f)) return -1;
    const size_t HW = (size_t)f.shape[1] * f.shape[2], n = (size_t)(fz1 - fz0) * HW;
    size_t cand = 0; for (size_t i = 0; i < n; i++) cand += f.f.w[i] != 0;
    const size_t stride = cand > (size_t)nsamp ? cand / (size_t)nsamp : 1;
    size_t seen = 0, ni = 0;
    double *dv = malloc(sizeof(double) * (cand / stride + 2)); size_t *iv = malloc(sizeof(size_t) * (cand / stride + 2)); float *cw = malloc(sizeof(float) * (cand / stride + 2));
    for (size_t i = 0; i < n; i++) {   /* the sampled cells, in cell order */
        if (!f.f.w[i]) continue;
        if (seen++ % stride) continue;
        const double c = (f.f.pc[i] - 128) / 127.0, sn = (f.f.ps[i] - 128) / 127.0;
        if (c * c + sn * sn < minc * minc) continue;
        cw[ni] = (float)sqrt(c * c + sn * sn); iv[ni++] = i;
    }
    fj_job J = {a, b, za, zb, h, fz0, HW, f.shape[1], f.shape[2], iv, dv, ni, ref, ro, bias};
    fit_par((int)((ni + 4095) / 4096), fj_rows, &J);   /* (differences on threads, written by position) */
    long *rv = malloc(sizeof(long) * (ni + 1)); size_t nd = 0;
    for (size_t q = 0; q < ni; q++) {
        const double d = dv[q]; if (!isfinite(d)) continue;
        const long r = lround(d); rv[nd++] = 2 * r + (fabs(d - r) < 0.25);   /* (value, near flag) packed: sorts by the value */
        if (ph) { *ph += cw[q] * cexp(2 * M_PI * I * d); *phw += cw[q]; }
    }
    free(iv); free(dv); free(cw); fit_fields_free(&f);
    if (H) {
        qsort(rv, nd, sizeof(long), cmp_long);
        memset(H, 0, sizeof *H); H->v = malloc(sizeof(long) * (nd + 1)); H->c = calloc(nd + 1, sizeof(long long)); H->cn = calloc(nd + 1, sizeof(long long));
        for (size_t i = 0; i < nd; i++) {
            const long v = rv[i] >> 1;   /* (arithmetic shift: floor) */
            if (!H->n || H->v[H->n - 1] != v) { H->v[H->n] = v; H->n++; }
            H->c[H->n - 1]++; H->cn[H->n - 1] += rv[i] & 1;
        }
        H->tot = (long long)nd;
    }
    free(rv);
    return 0;
}

double fj_table_cost(const fj_table *t, long v) {
    int lo = 0, hi = t->nv;
    while (lo < hi) { const int m = (lo + hi) / 2; if (t->v[m] < v) lo = m + 1; else hi = m; }
    return lo < t->nv && t->v[lo] == v ? t->c[lo] : t->other;
}

static double fj_energy(int n, const fj_table *u, int ne, const fj_edge *e, const long *S) {
    double E = 0;
    for (int k = 0; k < n; k++) E += fj_table_cost(&u[k], S[k]);
    for (int i = 0; i < ne; i++) E += fj_table_cost(&e[i].t, S[e[i].b] - S[e[i].a]);
    return E;
}

double fit_join_solve(int n, const fj_table *unary, int ne, const fj_edge *e, int rc, long *S) {
    if (n < 1) return 0;
    const int NC = 2 * rc + 1;
    long *cen = malloc(sizeof(long) * n); memcpy(cen, S, sizeof(long) * n);
    double *cost = malloc(sizeof(double) * n * NC); int *arg = malloc(sizeof(int) * n * NC);
    for (int k = 0; k < n; k++) for (int j = 0; j < NC; j++) {   /* exact over the consecutive-slab edges */
        const long sk = cen[k] + j - rc; const double c = fj_table_cost(&unary[k], sk);
        double best = k ? 1e300 : 0; int ba = -1;
        if (k) for (int i = 0; i < NC; i++) {
            const long sp = cen[k - 1] + i - rc; double v = cost[(size_t)(k - 1) * NC + i];
            for (int q = 0; q < ne; q++) if (e[q].a == k - 1 && e[q].b == k) v += fj_table_cost(&e[q].t, sk - sp);
            if (v < best) { best = v; ba = i; }
        }
        cost[(size_t)k * NC + j] = c + best; arg[(size_t)k * NC + j] = ba;
    }
    int j = 0; for (int i = 1; i < NC; i++) if (cost[(size_t)(n - 1) * NC + i] < cost[(size_t)(n - 1) * NC + j]) j = i;
    for (int k = n - 1; k >= 0; k--) { S[k] = cen[k] + j - rc; if (k) j = arg[(size_t)k * NC + j]; }
    free(cost); free(arg); free(cen);
    /* the local search over every edge (the longer ones too): single-slab and suffix shifts */
    double E = fj_energy(n, unary, ne, e, S);
    static const int dv[4] = {1, -1, 2, -2};
    for (int sweep = 0; sweep < 200; sweep++) {
        int moved = 0;
        for (int mode = 0; mode < 2; mode++) for (int k = mode; k < n; k++) for (int q = 0; q < 4; q++) {
            const int k1 = mode ? n : k + 1;   /* mode 0: slab k alone; mode 1: slabs k.. (k >= 1) */
            for (int i = k; i < k1; i++) S[i] += dv[q];
            const double E2 = fj_energy(n, unary, ne, e, S);
            if (E2 < E - 1e-9) { E = E2; moved = 1; }
            else for (int i = k; i < k1; i++) S[i] -= dv[q];
        }
        if (!moved) break;
    }
    return E;
}

/* the robust cost table of a histogram: w x min(cap, -log P(v) + log P(mode)), P(v) = (count + 0.5) / (tot + 1) */
static fj_table fj_hist_table(const fj_hist *H, double w, double cap) {
    fj_table t = {0};
    if (!H->tot || !(w > 0)) return t;
    long *v = malloc(sizeof(long) * (H->n + 1)); double *c = malloc(sizeof(double) * (H->n + 1));
    const double lm = log((fj_count(H, fj_mode(H), nullptr) + 0.5) / (H->tot + 1.0));
    for (int i = 0; i < H->n; i++) { v[i] = H->v[i]; c[i] = w * fmin(cap, lm - log((H->c[i] + 0.5) / (H->tot + 1.0))); }
    t.v = v; t.c = c; t.nv = H->n; t.other = w * fmin(cap, lm - log(0.5 / (H->tot + 1.0)));
    return t;
}

/* --global: the joint solve over the overlap graph (consecutive overlaps hb[1..ns), longer ones hx), the ref histograms hr as the
   anchor (may be null); S in: the default join's gauges (the DP with --ref, else the chain), out: the solve's. Prints the per-slab
   changes against the chain Sc (up to one constant) and against the default join. */
static void fj_global(int ns, const fj_hist *hb, const fj_hist *hr, int nx, const int *xa, const int *xb, const fj_hist *hx, double refw, double chainw,
                      double cap, double mincells, const long *Sc, long *S) {
    fj_table *u = calloc(ns, sizeof *u); fj_edge *e = calloc(ns + nx, sizeof *e); int ne = 0;
    for (int k = 0; k < ns; k++) if (hr && hr[k].tot) u[k] = fj_hist_table(&hr[k], refw * fmin(1.0, hr[k].tot / (mincells / 10)), cap);   /* (ref samples: 1/10 of the overlap cells') */
    for (int k = 1; k < ns; k++) if (hb[k].tot) { e[ne].a = k - 1; e[ne].b = k; e[ne].t = fj_hist_table(&hb[k], chainw * fmin(1.0, hb[k].tot / mincells), cap); ne++; }
    for (int q = 0; q < nx; q++) if (hx[q].tot) { e[ne].a = xa[q]; e[ne].b = xb[q]; e[ne].t = fj_hist_table(&hx[q], chainw * fmin(1.0, hx[q].tot / mincells), cap); ne++; }
    long *S0 = malloc(sizeof(long) * ns); memcpy(S0, S, sizeof(long) * ns);
    const double E0 = fj_energy(ns, u, ne, e, S0), Ec = fj_energy(ns, u, ne, e, Sc);
    const double E = fit_join_solve(ns, u, ne, e, 12, S);
    /* the chain's constant: the most common S - Sc */
    long c0 = 0; int bc = 0;
    for (int k = 0; k < ns; k++) { int c = 0; for (int i = 0; i < ns; i++) c += S[i] - Sc[i] == S[k] - Sc[k]; if (c > bc) { bc = c; c0 = S[k] - Sc[k]; } }
    int nch = 0, ndef = 0;
    for (int k = 0; k < ns; k++) { nch += S[k] - Sc[k] != c0; ndef += S[k] != S0[k]; }
    printf("fit-join: --global energy %.2f (default join %.2f, chain %.2f); %d of %d slabs off the chain's gauge (constant %+ld), %d moved from the default join\n",
           E, E0, Ec, nch, ns, c0, ndef);
    printf("  slab  global-chain  global-default\n");
    for (int k = 0; k < ns; k++) if (S[k] - Sc[k] != c0 || S[k] != S0[k]) printf("  %4d  %+12ld  %+14ld\n", k, S[k] - Sc[k] - c0, S[k] - S0[k]);
    for (int k = 0; k < ns; k++) { free((void *)u[k].v); free((void *)u[k].c); }
    for (int i = 0; i < ne; i++) { free((void *)e[i].t.v); free((void *)e[i].t.c); }
    free(u); free(e); free(S0);
}

int cmd_fit_join(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: ufsm fit-join <out-set> <fields-dir> [--blend 16] [--min-conf 0.3] [--samples 4000000] [--ref MODEL [--ref-w 1] [--chain-w 1] [--ref-samples 200000] [--ref-rows 16]]\n"
                        "       [--global 1 [--cap 3] [--min-cells 20000]] <slab-model> <z0> <z1> ...\n"
                        "  slabs of one field box (same y / x crop), interiors [z0, z1) in box-local native z, in z order; --ref: the integer gauges from a global model\n");
        return 2;
    }
    const char *out = argv[2], *fdir = argv[3];
    double blend = 16, minc = 0.3, refw = 1, chainw = 1, cap = 3, mincells = 20000; long long nsamp = 4000000, nref = 200000; const char *refdir = nullptr;
    int glob = 0, refrows = 16;
    int ai = 4;
    for (; ai + 1 < argc && !strncmp(argv[ai], "--", 2); ai += 2) {
        if (!strcmp(argv[ai], "--blend")) blend = atof(argv[ai + 1]);
        else if (!strcmp(argv[ai], "--min-conf")) minc = atof(argv[ai + 1]);
        else if (!strcmp(argv[ai], "--samples")) nsamp = atoll(argv[ai + 1]);
        else if (!strcmp(argv[ai], "--ref")) refdir = argv[ai + 1];
        else if (!strcmp(argv[ai], "--ref-w")) refw = atof(argv[ai + 1]);
        else if (!strcmp(argv[ai], "--chain-w")) chainw = atof(argv[ai + 1]);
        else if (!strcmp(argv[ai], "--ref-samples")) nref = atoll(argv[ai + 1]);
        else if (!strcmp(argv[ai], "--ref-rows")) refrows = atoi(argv[ai + 1]);
        else if (!strcmp(argv[ai], "--global")) glob = atoi(argv[ai + 1]);
        else if (!strcmp(argv[ai], "--cap")) cap = atof(argv[ai + 1]);
        else if (!strcmp(argv[ai], "--min-cells")) mincells = atof(argv[ai + 1]);
        else { fprintf(stderr, "fit-join: unknown option %s\n", argv[ai]); return 2; }
    }
    const int ns = (argc - ai) / 3;
    if (ns < 1 || (argc - ai) % 3 || !(blend >= 0) || nsamp < 1 || nref < 1 || !(refw >= 0) || !(chainw >= 0) || refrows < 0 || !(cap > 0) || !(mincells >= 1)) { fprintf(stderr, "fit-join: give <slab-model> <z0> <z1> triples (and --blend >= 0, weights >= 0)\n"); return 2; }
    fit_fields_skip_planes(1 | 2 | 4); fit_fields_ct_planes(0);   /* (the overlaps read pc ps w only) */
    fit_fields_buf g; if (fit_fields_load(fdir, 0, 0, 0, &g)) { fprintf(stderr, "fit-join: cannot read the fields in %s\n", fdir); return 1; }
    const double h = g.h, Z = g.shape[0] * h;
    fit_set *s = calloc(1, sizeof *s); s->n = ns; s->blend = blend;
    s->slab = calloc(ns, sizeof *s->slab); s->z0 = calloc(ns, sizeof(double)); s->z1 = calloc(ns, sizeof(double));
    for (int k = 0; k < ns; k++) {
        const char *md = argv[ai + 3 * k];
        s->slab[k] = fit_load(md); s->z0[k] = atof(argv[ai + 3 * k + 1]); s->z1[k] = atof(argv[ai + 3 * k + 2]);
        const fit_model *m = s->slab[k];
        if (!m) { fprintf(stderr, "fit-join: cannot load the model %s\n", md); return 1; }
        if (m->origin[1] != g.origin[1] || m->origin[2] != g.origin[2] || m->H != g.shape[1] || m->W != g.shape[2] || m->h != h) {
            fprintf(stderr, "fit-join: %s is not a z crop of %s (origin %lld,%lld,%lld size %d,%d h %g vs %lld,%lld size %d,%d h %g)\n", md, fdir,
                    m->origin[0], m->origin[1], m->origin[2], m->H, m->W, m->h, g.origin[1], g.origin[2], g.shape[1], g.shape[2], h);
            return 1;
        }
        const double mz0 = (double)(m->origin[0] - g.origin[0]), mz1 = mz0 + m->D * m->h;
        if (!(s->z0[k] < s->z1[k]) || s->z0[k] < mz0 || s->z1[k] > mz1 || (k && s->z0[k] != s->z1[k - 1]) || (k && m->dir != s->slab[0]->dir)) {
            fprintf(stderr, "fit-join: slab %d (%s): interior [%g, %g) must lie in its box [%g, %g), follow the previous interior, same direction\n", k, md, s->z0[k], s->z1[k], mz0, mz1);
            return 1;
        }
    }
    const double iz0 = s->z0[0], iz1 = s->z1[ns - 1];   /* (the given interiors: the ref samples stay inside them) */
    s->z0[0] = 0; s->z1[ns - 1] = Z;   /* (the set covers the whole box: rows outside the first / last interior use those slabs) */
    printf("fit-join: %d slabs, box %dx%dx%d h %g, blend %g\n", ns, g.shape[0], g.shape[1], g.shape[2], h, blend);
    /* per boundary the histogram of round(W_k - W_k-1) over sampled overlap cells (the slabs' own gauges, nothing shifted yet) */
    fj_hist *hb = calloc(ns, sizeof *hb);
    for (int k = 1; k < ns; k++) {   /* the overlap of slab k - 1 and slab k */
        const fit_model *a = s->slab[k - 1], *b = s->slab[k];
        const double za = (double)(a->origin[0] - g.origin[0]), zb = (double)(b->origin[0] - g.origin[0]);
        const double lo = fmax(za, zb), hi = fmin(za + a->D * h, zb + b->D * h);
        const int fz0 = (int)ceil(lo / h), fz1 = (int)floor(hi / h);
        if (fz1 <= fz0) { fprintf(stderr, "fit-join: slabs %d and %d do not overlap\n", k - 1, k); return 1; }
        if (fj_sample(fdir, fz0, fz1, minc, nsamp, a, za, b, zb, nullptr, nullptr, 0, h, &hb[k], nullptr, nullptr)) { fprintf(stderr, "fit-join: cannot load field rows %d..%d\n", fz0, fz1); return 1; }
    }
    /* --global: the overlaps of slabs further apart (boxes reaching past the next slab's) as extra edges */
    int nx = 0, *xa = nullptr, *xb = nullptr; fj_hist *hx = nullptr;
    if (glob) for (int i = 0; i < ns; i++) for (int j = i + 2; j < ns; j++) {
        const fit_model *a = s->slab[i], *b = s->slab[j];
        const double za = (double)(a->origin[0] - g.origin[0]), zb = (double)(b->origin[0] - g.origin[0]);
        const int fz0 = (int)ceil(fmax(za, zb) / h), fz1 = (int)floor(fmin(za + a->D * h, zb + b->D * h) / h);
        if (fz1 <= fz0) continue;
        xa = realloc(xa, sizeof(int) * (nx + 1)); xb = realloc(xb, sizeof(int) * (nx + 1)); hx = realloc(hx, sizeof *hx * (nx + 1));
        xa[nx] = i; xb[nx] = j;
        if (fj_sample(fdir, fz0, fz1, minc, nsamp, a, za, b, zb, nullptr, nullptr, 0, h, &hx[nx], nullptr, nullptr)) { fprintf(stderr, "fit-join: cannot load field rows %d..%d\n", fz0, fz1); return 1; }
        nx++;
    }
    if (glob) printf("fit-join: --global: %d overlap edges (%d between consecutive slabs, %d longer), cost cap %g nats, full weight from %g cells\n", ns - 1 + nx, ns - 1, nx, cap, mincells);
    /* the gauges S_k (slab k minus S_k joins). Without --ref the chain of overlap modes; with --ref the dynamic programme */
    long *S = calloc(ns, sizeof(long)), *Sc = calloc(ns, sizeof(long));
    for (int k = 1; k < ns; k++) Sc[k] = Sc[k - 1] + fj_mode(&hb[k]);   /* (the chain) */
    fj_hist *hr = nullptr;
    if (!refdir) { memcpy(S, Sc, sizeof(long) * ns); if (glob) fj_global(ns, hb, nullptr, nx, xa, xb, hx, refw, chainw, cap, mincells, Sc, S); }
    else {
        fit_set *ref = fit_set_load(refdir);
        if (!ref) { fprintf(stderr, "fit-join: cannot load the --ref model %s\n", refdir); return 1; }
        if (ref->global->dir != s->slab[0]->dir) { fprintf(stderr, "fit-join: the --ref model's direction %+d differs from the slabs' %+d\n", ref->global->dir, s->slab[0]->dir); return 1; }
        const double ro[3] = {(double)(g.origin[0] - ref->global->origin[0]), (double)(g.origin[1] - ref->global->origin[1]), (double)(g.origin[2] - ref->global->origin[2])};
        hr = calloc(ns, sizeof *hr); double complex cs = 0; double csw = 0, cbias = 0;
        for (int pass = 0; pass < 2; pass++) {   /* pass 0: the sub-turn bias; pass 1: the histograms */
            for (int k = 0; k < ns; k++) {
                const double i0 = k ? s->z0[k] : iz0, i1 = k + 1 < ns ? s->z1[k] : iz1, zc = 0.5 * (i0 + i1);
                int fz0 = refrows ? (int)floor(zc / h) - refrows / 2 : (int)ceil(i0 / h), fz1 = refrows ? fz0 + refrows : (int)floor(i1 / h);
                if (fz0 < (int)ceil(i0 / h)) fz0 = (int)ceil(i0 / h);
                if (fz1 > (int)floor(i1 / h)) fz1 = (int)floor(i1 / h);
                if (fz1 <= fz0) continue;
                const fit_model *m = s->slab[k]; const double zm = (double)(m->origin[0] - g.origin[0]);
                if (fj_sample(fdir, fz0, fz1, minc, pass ? nref : nref / 4 + 1, m, zm, nullptr, 0, ref, ro, cbias, h, pass ? &hr[k] : nullptr, pass ? nullptr : &cs, &csw)) {
                    fprintf(stderr, "fit-join: cannot load field rows %d..%d\n", fz0, fz1); return 1;
                }
            }
            if (!pass) cbias = cabs(cs) > 0 ? carg(cs) / (2 * M_PI) : 0;
        }
        printf("fit-join: --ref %s (box offset %.0f %.0f %.0f native), sub-turn bias of W_slab - W_ref %+.3f turn (coherence %.3f), ref weight %g, chain weight %g\n",
               refdir, ro[0], ro[1], ro[2], cbias, csw > 0 ? cabs(cs) / csw : 0, refw, chainw);
        enum { RC = 12, NC = 2 * RC + 1 };   /* candidates: the ref mode +- RC (a slab without ref cells: the previous slab's centre + its overlap mode) */
        double *cost = malloc(sizeof(double) * ns * NC); int *arg = malloc(sizeof(int) * ns * NC); long *cen = malloc(sizeof(long) * ns);
        for (int k = 0; k < ns; k++) {
            cen[k] = hr[k].tot ? fj_mode(&hr[k]) : k ? cen[k - 1] + fj_mode(&hb[k]) : 0;
            for (int j = 0; j < NC; j++) {
                const long sk = cen[k] + j - RC; const double c = hr[k].tot ? -refw * log((fj_count(&hr[k], sk, nullptr) + 0.5) / (hr[k].tot + 1.0)) : 0;
                double best = k ? 1e300 : 0; int ba = -1;
                if (k) for (int i = 0; i < NC; i++) {
                    const long sp = cen[k - 1] + i - RC;
                    const double v = cost[(size_t)(k - 1) * NC + i] + (hb[k].tot ? -chainw * log((fj_count(&hb[k], sk - sp, nullptr) + 0.5) / (hb[k].tot + 1.0)) : 0);
                    if (v < best) { best = v; ba = i; }
                }
                cost[(size_t)k * NC + j] = c + best; arg[(size_t)k * NC + j] = ba;
            }
        }
        int j = 0; for (int i = 1; i < NC; i++) if (cost[(size_t)(ns - 1) * NC + i] < cost[(size_t)(ns - 1) * NC + j]) j = i;
        for (int k = ns - 1; k >= 0; k--) { S[k] = cen[k] + j - RC; if (k) j = arg[(size_t)k * NC + j]; }
        free(cost); free(arg); free(cen);
        if (glob) fj_global(ns, hb, hr, nx, xa, xb, hx, refw, chainw, cap, mincells, Sc, S);
        printf("  slab  ref-cells  share(ref-mode)  chosen-mode  share(chosen)  share(|d - chosen| < 0.25)\n");
        long long rt = 0, rs = 0;
        for (int k = 0; k < ns; k++) {
            if (!hr[k].tot) { printf("  %4d  %9d  (no cells)\n", k, 0); continue; }
            long long nr = 0; const long md = fj_mode(&hr[k]); const long long cm = fj_count(&hr[k], md, nullptr), cc = fj_count(&hr[k], S[k], &nr);
            printf("  %4d  %9lld  %15.4f  %+11ld  %13.4f  %10.4f\n", k, hr[k].tot, (double)cm / hr[k].tot, S[k] - md, (double)cc / hr[k].tot, (double)nr / hr[k].tot);
            rt += hr[k].tot; rs += cc;
        }
        printf("  mean share of the ref's integer at the chosen gauges, cell weighted: %.4f\n", rt ? (double)rs / rt : 0);
        fit_set_free(ref);
    }
    const int chosen = refdir || glob;
    printf("  boundary  z(native)  overlap-cells  shift  share(mode)  share(|d - mode| < 0.25)%s\n", chosen ? "  chosen-mode  share(chosen)" : "");
    long long tot_n = 0, tot_c = 0; double tot_share = 0;
    for (int k = 1; k < ns; k++) {
        const fj_hist *H = &hb[k]; const long md = fj_mode(H); long long nn = 0; const long long cm = fj_count(H, md, &nn), used = H->tot;
        const double share = used ? (double)cm / used : 0;
        printf("  %3d|%-3d %10.0f %14lld %+6ld  %10.4f  %10.4f", k - 1, k, s->z0[k], used, S[k], share, used ? (double)nn / used : 0);
        if (chosen) { const long long cc = fj_count(H, S[k] - S[k - 1], nullptr); printf("  %+11ld  %12.4f", S[k] - S[k - 1] - md, used ? (double)cc / used : 0); tot_c += cc; }
        printf("\n");
        tot_n += used; tot_share += share * used;
    }
    if (ns > 1) printf("  mean share(mode) over the boundaries, overlap-cell weighted: %.4f\n", tot_n ? tot_share / tot_n : 0);
    if (ns > 1 && chosen) printf("  integer agreement on the overlaps at the chosen gauges, overlap-cell weighted: %.4f\n", tot_n ? (double)tot_c / tot_n : 0);
    for (int k = 0; k < ns; k++) if (S[k]) for (int q = 0; q < s->slab[k]->nk; q++) s->slab[k]->offset[q] -= (float)S[k];
    for (int k = 0; k < ns; k++) { fj_hist_free(&hb[k]); if (hr) fj_hist_free(&hr[k]); }
    for (int q = 0; q < nx; q++) fj_hist_free(&hx[q]);
    free(hb); free(hr); free(S); free(Sc); free(hx); free(xa); free(xb);
    /* the global model: the box geometry and axis, the slabs' knots, no lattice */
    fit_model *G = calloc(1, sizeof *G); const fit_model *s0 = s->slab[0];
    G->dir = s0->dir; G->dir_vote = 1; G->knot_dz = s0->knot_dz; G->D = g.shape[0]; G->H = g.shape[1]; G->W = g.shape[2]; G->h = h; G->um = s0->um;
    for (int d = 0; d < 3; d++) G->origin[d] = g.origin[d];
    G->nk = (int)ceil(Z / G->knot_dz) + 1; if (G->nk < 2) G->nk = 2;
    G->pitch = calloc(G->nk, sizeof(float)); G->offset = calloc(G->nk, sizeof(float));
    for (int q = 0; q < G->nk; q++) {
        const double z = fmin((double)q * G->knot_dz, Z - 1e-6); int k = 0;
        while (k + 1 < ns && z >= s->z1[k]) k++;
        const fit_model *m = s->slab[k]; const double zl = z - (double)(m->origin[0] - g.origin[0]);
        G->pitch[q] = (float)fj_knot(m->pitch, m->nk, m->knot_dz, zl); G->offset[q] = (float)fj_knot(m->offset, m->nk, m->knot_dz, zl);
    }
    G->L = 1 << 20; G->lz = G->ly = G->lx = 2; G->delta = calloc(8, sizeof(float));
    const int na = (int)(G->D * h) + 1; G->cy = malloc(na * sizeof(float)); G->cx = malloc(na * sizeof(float));
    memcpy(G->cy, g.cy, na * sizeof(float)); memcpy(G->cx, g.cx, na * sizeof(float));
    s->global = G;
    if (fit_set_save(s, out)) { fprintf(stderr, "fit-join: cannot save %s\n", out); return 1; }
    printf("fit-join: set written to %s\n", out);
    fit_set_free(s); fit_fields_free(&g);
    return 0;
}
