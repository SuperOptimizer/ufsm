/* ufsm fit-join <out-set> <fields-dir> [--blend B] [--min-conf C] [--samples N] <slab-model> <z0> <z1> [<slab-model> <z0> <z1> ...]
   Independent z-slab fits of one field box -> one fit set (src/fit.h fit_set) with a consistent integer gauge, label-free.
   <fields-dir> is the whole box the slabs were cropped from (the same y / x origin and size: mkcrop of z rows only); each slab model
   (ufsm fit --out) covers its own z rows; <z0> <z1> is the slab's interior in box-local native z of <fields-dir> (the interiors
   partition the box, the slab models extend past them: the overlaps). In z order, each slab's integer gauge is matched to the
   previous (already matched) slab on the cells of their overlap that have material and phase confidence >= --min-conf (0.3):
   shift = the mode of round(W_k - W_k-1), then W_k -= shift. Per boundary it prints the shift and the agreement (share of the
   overlap cells whose rounded difference is the mode, and whose difference is within 0.25 turn of it): a slab-to-slab consistency
   measure. The global model of the set is the box's geometry and axis with the slabs' knots (pitch / offset at each knot taken
   from the slab whose interior holds it) and no lattice; W is blended linearly over +- B native voxels (default 16) around each
   interior boundary (fit_set_W). Output: <out-set>/set.json, global/, slabNNN/ (ufsm wraps / fit --model read it). */
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

int cmd_fit_join(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: ufsm fit-join <out-set> <fields-dir> [--blend 16] [--min-conf 0.3] [--samples 4000000] <slab-model> <z0> <z1> ...\n"
                        "  slabs of one field box (same y / x crop), interiors [z0, z1) in box-local native z, in z order\n");
        return 2;
    }
    const char *out = argv[2], *fdir = argv[3];
    double blend = 16, minc = 0.3; long long nsamp = 4000000; int tile = 0;
    int ai = 4;
    for (; ai + 1 < argc && !strncmp(argv[ai], "--", 2); ai += 2) {
        if (!strcmp(argv[ai], "--blend")) blend = atof(argv[ai + 1]);
        else if (!strcmp(argv[ai], "--min-conf")) minc = atof(argv[ai + 1]);
        else if (!strcmp(argv[ai], "--samples")) nsamp = atoll(argv[ai + 1]);
        else if (!strcmp(argv[ai], "--tile-shift")) tile = atoi(argv[ai + 1]);
        else { fprintf(stderr, "fit-join: unknown option %s\n", argv[ai]); return 2; }
    }
    const int ns = (argc - ai) / 3;
    if (ns < 1 || (argc - ai) % 3 || !(blend >= 0) || nsamp < 1) { fprintf(stderr, "fit-join: give <slab-model> <z0> <z1> triples (and --blend >= 0)\n"); return 2; }
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
    s->z0[0] = 0; s->z1[ns - 1] = Z;   /* (the set covers the whole box: rows outside the first / last interior use those slabs) */
    printf("fit-join: %d slabs, box %dx%dx%d h %g, blend %g\n", ns, g.shape[0], g.shape[1], g.shape[2], h, blend);
    printf("  boundary  z(native)  overlap-cells  shift  share(mode)  share(|d - mode| < 0.25)\n");
    long long tot_n = 0; double tot_share = 0;
    for (int k = 1; k < ns; k++) {   /* the overlap of slab k - 1 and slab k */
        const fit_model *a = s->slab[k - 1]; fit_model *b = s->slab[k];
        const double za = (double)(a->origin[0] - g.origin[0]), zb = (double)(b->origin[0] - g.origin[0]);
        const double lo = fmax(za, zb), hi = fmin(za + a->D * h, zb + b->D * h);
        const int fz0 = (int)ceil(lo / h), fz1 = (int)floor(hi / h);
        if (fz1 <= fz0) { fprintf(stderr, "fit-join: slabs %d and %d do not overlap\n", k - 1, k); return 1; }
        fit_fields_buf f; if (fit_fields_load(fdir, fz0, fz1, 1, &f)) { fprintf(stderr, "fit-join: cannot load field rows %d..%d\n", fz0, fz1); return 1; }
        const size_t HW = (size_t)f.shape[1] * f.shape[2], n = (size_t)(fz1 - fz0) * HW;
        size_t cand = 0; for (size_t i = 0; i < n; i++) cand += f.f.w[i] != 0;
        const size_t stride = cand > (size_t)nsamp ? cand / (size_t)nsamp : 1;
        long long used = 0, near = 0; size_t seen = 0;   /* (slab gauges are arbitrary: the rounded differences can be any integer) */
        double *dv = malloc(sizeof(double) * (cand / stride + 2)); long *rv = malloc(sizeof(long) * (cand / stride + 2)); size_t nd = 0;
        uint32_t *tv = malloc(sizeof(uint32_t) * (cand / stride + 2));
        const int nty = tile > 0 ? (f.shape[1] + tile - 1) / tile : 0, ntx = tile > 0 ? (f.shape[2] + tile - 1) / tile : 0;
        for (size_t i = 0; i < n; i++) {
            if (!f.f.w[i]) continue;
            if (seen++ % stride) continue;
            const double c = (f.f.pc[i] - 128) / 127.0, sn = (f.f.ps[i] - 128) / 127.0;
            if (c * c + sn * sn < minc * minc) continue;
            const int zz = (int)(i / HW), yy = (int)((i / f.shape[2]) % f.shape[1]), xx = (int)(i % f.shape[2]);
            const double z = (fz0 + zz + 0.5) * h, y = (yy + 0.5) * h, x = (xx + 0.5) * h;
            const double d = fit_W(b, z - zb, y, x) - fit_W(a, z - za, y, x);
            if (!isfinite(d)) continue;
            if (tile > 0) tv[nd] = (uint32_t)((yy / tile) * ntx + xx / tile);
            rv[nd] = lround(d); dv[nd++] = d; used++;
        }
        qsort(rv, nd, sizeof(long), cmp_long);   /* the mode of the rounded differences */
        long shift = 0; size_t bestn = 0;
        for (size_t i = 0; i < nd;) { size_t j = i; while (j < nd && rv[j] == rv[i]) j++; if (j - i > bestn) { bestn = j - i; shift = rv[i]; } i = j; }
        for (size_t i = 0; i < nd; i++) near += fabs(dv[i] - shift) < 0.25;
        if (shift) for (int q = 0; q < b->nk; q++) b->offset[q] -= (float)shift;
        if (tile > 0) {   /* per xy tile: the mode of the rounded difference left after the shift; a tile whose mode is not 0 (>= 100 cells,
                             >= 1.5x the cells at 0) steps slab k's lattice by it over all z (the slabs slipped differently there) */
            const int nt = nty * ntx; int (*hc)[9] = calloc((size_t)nt, sizeof *hc); int *ts = calloc((size_t)nt, sizeof *ts), nsh = 0, nok = 0;
            for (size_t i = 0; i < nd; i++) { const long r = lround(dv[i]) - shift; if (r >= -4 && r <= 4) hc[tv[i]][r + 4]++; }
            for (int t = 0; t < nt; t++) {
                int bm = 4, tot = 0; for (int j = 0; j < 9; j++) { tot += hc[t][j]; if (hc[t][j] > hc[t][bm]) bm = j; }
                if (tot >= 100) nok++;
                if (tot >= 100 && bm != 4 && hc[t][bm] >= 1.5 * hc[t][4]) { ts[t] = bm - 4; nsh++; }
            }
            for (int a = 0; a < b->lz; a++) for (int yb = 0; yb < b->ly; yb++) for (int c = 0; c < b->lx; c++) {
                int ty = (int)((double)yb * b->L / h / tile), tx = (int)((double)c * b->L / h / tile); if (ty >= nty) ty = nty - 1; if (tx >= ntx) tx = ntx - 1;
                b->delta[((size_t)a * b->ly + yb) * b->lx + c] -= (float)ts[ty * ntx + tx];
            }
            printf("  %3d|%-3d tile shifts (%d cells): %d of %d tiles (>= 100 cells) shifted\n", k - 1, k, tile, nsh, nok);
            free(hc); free(ts);
        }
        free(dv); free(rv); free(tv); fit_fields_free(&f);
        const double share = used ? (double)bestn / used : 0;
        printf("  %3d|%-3d %10.0f %14lld %+6ld  %10.4f  %10.4f\n", k - 1, k, s->z0[k], used, shift, share, used ? (double)near / used : 0);
        tot_n += used; tot_share += share * used;
    }
    if (ns > 1) printf("  mean share(mode) over the boundaries, overlap-cell weighted: %.4f\n", tot_n ? tot_share / tot_n : 0);
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
