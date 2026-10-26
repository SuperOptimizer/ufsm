/* ufsm fit-consensus <out-model> <fields-dir> [--min-conf 0.3] [--samples 4000000] <model0> <model1> [<model2> ...]
   Several fits of one field box (perturbed: repair tile offsets, shifted crops, other seeds) -> one model whose W is the per-point
   median of theirs (a majority vote on the winding where the fits slip differently). <fields-dir> is the box <model0> was fitted on
   (its geometry, knots and axis are the output's); the other models may be crops of other origins (their boxes shifted in native
   units: the overlap counts). Each model's integer gauge is matched to model0 on the confident material cells of <fields-dir>
   (mode of round(W_k - W_0), as fit-join); then each lattice node of model0 takes delta = median_k(W_k) - (W_0 - delta_0) over the
   models whose box holds the node (lower median for even counts; a node no other model covers keeps delta_0). Prints the shifts,
   each model's agreement with the consensus (share of nodes within 0.25 turn) and the share of nodes where all agree. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fit.h"

static int cmp_long(const void *a, const void *b) { const long x = *(const long *)a, y = *(const long *)b; return (x > y) - (x < y); }
static int cmp_dbl(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

/* W of model m at a point given in model0's box-local native coordinates (off: m's origin - model0's); NAN outside m's box */
static double fc_W(const fit_model *m, const double off[3], double z, double y, double x) {
    z -= off[0]; y -= off[1]; x -= off[2];
    if (z < 0 || y < 0 || x < 0 || z > m->D * m->h || y > m->H * m->h || x > m->W * m->h) return NAN;
    return fit_W(m, z, y, x);
}

int cmd_fit_consensus(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: ufsm fit-consensus <out-model> <fields-dir> [--min-conf 0.3] [--samples 4000000] <model0> <model1> [<model2> ...]\n"
                        "  per-point median W of several fits of one box (gauges matched to model0); odd counts recommended\n");
        return 2;
    }
    const char *out = argv[2], *fdir = argv[3];
    double minc = 0.3; long long nsamp = 4000000;
    int ai = 4;
    for (; ai + 1 < argc && !strncmp(argv[ai], "--", 2); ai += 2) {
        if (!strcmp(argv[ai], "--min-conf")) minc = atof(argv[ai + 1]);
        else if (!strcmp(argv[ai], "--samples")) nsamp = atoll(argv[ai + 1]);
        else { fprintf(stderr, "fit-consensus: unknown option %s\n", argv[ai]); return 2; }
    }
    const int K = argc - ai;
    if (K < 2 || nsamp < 1) { fprintf(stderr, "fit-consensus: give at least two models\n"); return 2; }
    fit_model **m = calloc(K, sizeof *m); double (*off)[3] = calloc(K, sizeof *off);
    for (int k = 0; k < K; k++) {
        if (!(m[k] = fit_load(argv[ai + k]))) { fprintf(stderr, "fit-consensus: cannot load the model %s\n", argv[ai + k]); return 1; }
        if (m[k]->dir != m[0]->dir) { fprintf(stderr, "fit-consensus: %s has winding direction %d, model0 %d\n", argv[ai + k], m[k]->dir, m[0]->dir); return 1; }
        for (int d = 0; d < 3; d++) off[k][d] = (double)(m[k]->origin[d] - m[0]->origin[d]);
    }
    fit_fields_buf g; if (fit_fields_load(fdir, 0, 0, 1, &g)) { fprintf(stderr, "fit-consensus: cannot read the fields in %s\n", fdir); return 1; }
    const double h = g.h;
    if (g.origin[0] != m[0]->origin[0] || g.origin[1] != m[0]->origin[1] || g.origin[2] != m[0]->origin[2] || g.shape[0] != m[0]->D || g.shape[1] != m[0]->H || g.shape[2] != m[0]->W) {
        fprintf(stderr, "fit-consensus: %s is not the box of model0 %s\n", fdir, argv[ai]); return 1;
    }
    /* gauges: the mode of round(W_k - W_0) on confident material cells */
    const size_t HW = (size_t)g.shape[1] * g.shape[2], n = (size_t)g.shape[0] * HW;
    size_t cand = 0; for (size_t i = 0; i < n; i++) cand += g.f.w[i] != 0;
    const size_t stride = cand > (size_t)nsamp ? cand / (size_t)nsamp : 1;
    long *rv = malloc(sizeof(long) * (cand / stride + 2));
    setvbuf(stdout, nullptr, _IOLBF, 0);
    printf("fit-consensus: %d models, box %dx%dx%d h %g\n", K, g.shape[0], g.shape[1], g.shape[2], h);
    for (int k = 1; k < K; k++) {
        size_t nd = 0, seen = 0;
        for (size_t i = 0; i < n; i++) {
            if (!g.f.w[i] || seen++ % stride) continue;
            const double c = (g.f.pc[i] - 128) / 127.0, sn = (g.f.ps[i] - 128) / 127.0;
            if (c * c + sn * sn < minc * minc) continue;
            const double z = ((double)(i / HW) + 0.5) * h, y = ((double)((i / g.shape[2]) % g.shape[1]) + 0.5) * h, x = ((double)(i % g.shape[2]) + 0.5) * h;
            const double d = fc_W(m[k], off[k], z, y, x) - fit_W(m[0], z, y, x);
            if (isfinite(d)) rv[nd++] = lround(d);
        }
        qsort(rv, nd, sizeof(long), cmp_long);
        long shift = 0; size_t bestn = 0;
        for (size_t i = 0; i < nd;) { size_t j = i; while (j < nd && rv[j] == rv[i]) j++; if (j - i > bestn) { bestn = j - i; shift = rv[i]; } i = j; }
        if (shift) for (int q = 0; q < m[k]->nk; q++) m[k]->offset[q] -= (float)shift;
        printf("  model %d (%s): offset %+.0f,%+.0f,%+.0f, %zu cells, gauge shift %+ld, share(mode) %.4f\n", k, argv[ai + k], off[k][0], off[k][1], off[k][2], nd, shift, nd ? (double)bestn / nd : 0);
    }
    free(rv);
    /* the consensus lattice */
    const fit_model *M = m[0]; const size_t nn = (size_t)M->lz * M->ly * M->lx;
    float *nd = malloc(nn * sizeof *nd); if (!nd) abort();
    size_t *agree = calloc(K, sizeof *agree), all = 0, cov = 0;
    #pragma omp parallel
    {
        double v[64]; size_t ag[64] = {0}, al = 0, cv = 0;
        #pragma omp for schedule(static)
        for (size_t i = 0; i < nn; i++) {
            const int a = (int)(i / ((size_t)M->ly * M->lx)), b = (int)((i / M->lx) % M->ly), c = (int)(i % M->lx);
            const double z = (double)a * M->L, y = (double)b * M->L, x = (double)c * M->L;
            const double w0 = fit_W(M, z, y, x), base0 = w0 - M->delta[i];
            int nv = 0; double wk[64];
            for (int k = 0; k < K && k < 64; k++) { const double w = k ? fc_W(m[k], off[k], z, y, x) : w0; wk[k] = w; if (isfinite(w)) v[nv++] = w; }
            if (nv < 2) { nd[i] = M->delta[i]; continue; }
            qsort(v, nv, sizeof(double), cmp_dbl);
            const double med = v[(nv - 1) / 2];
            nd[i] = (float)(med - base0); cv++;
            int allk = 1;
            for (int k = 0; k < K && k < 64; k++) { if (!isfinite(wk[k])) continue; if (fabs(wk[k] - med) < 0.25) ag[k]++; else allk = 0; }
            al += allk;
        }
        #pragma omp critical
        { for (int k = 0; k < K && k < 64; k++) agree[k] += ag[k]; all += al; cov += cv; }
    }
    for (int k = 0; k < K; k++) printf("  model %d agrees with the consensus (|dW| < 0.25) on %.4f of %zu covered nodes\n", k, cov ? (double)agree[k] / cov : 0, cov);
    printf("  all models agree on %.4f of the covered nodes\n", cov ? (double)all / cov : 0);
    memcpy(m[0]->delta, nd, nn * sizeof *nd); free(nd); free(agree);
    if (fit_save(m[0], out)) { fprintf(stderr, "fit-consensus: cannot save %s\n", out); return 1; }
    printf("fit-consensus: model written to %s\n", out);
    for (int k = 0; k < K; k++) fit_free(m[k]);
    free(m); free(off); fit_fields_free(&g);
    return 0;
}
