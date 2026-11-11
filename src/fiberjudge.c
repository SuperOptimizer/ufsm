/* ufsm fiber-judge <fibres.afv | fibres.fib> (--wraps DIR [--first K] [--last K] [--dir +-1] | --phase ROOT KEY | --set FITSET)
       --um U [--level L] [--tr T.json] --slab Z0,NZ [--region Y0,Y1,X0,X1] [--tile 512] [--d-mm 0.3,0.6,1.2,2.4 | --d-vox a,b,..]
       [--hyst 0.75] [--out rows.tsv]
   A label-free judge of a winding labelling: a fibre lies in one sheet, so along a fibre polyline the winding phase of a correct spiral
   stays within a fraction of a turn (DESIGN.md 2026-11-10). Fibres: an AFV (community AutomatedFiberVolumes; --tr maps another scan's AFV
   through the published transform) or a UFIB file whose points are level-0 voxels (z, y, x) of our scan (ufsm fiber-trace). Our winding
   phase at a fibre sample, three sources reduced to one band phase (0..1, unknown):
     --wraps  the final wraps (wNNN.sfc): between the two nearest wrap sheets, nearest vertex per wrap-id parity (d1 the nearer, d2 the
              other), accepted when their along-spiral coordinates q = -dir id + column / width differ by 0.5 .. 1.5:
              phase = frac(sign(q2 - q1) d1 / (d1 + d2));
     --phase  a band-phase raster (surf-labels phase.zarr or a GT phase; 0..251, 255 unknown) on the level-L grid;
     --set    a joined fit set: frac(W) (the theta cut is a unit jump of W).
   Every fibre is sampled each level voxel; a stretch of known phase is unwrapped along the path, median-filtered (9 samples) and
   followed with a hysteresis of --hyst turn: each time it leaves the current sheet by more than that, |k| excursions are counted and
   the baseline moves k turns. Per tile (--tile^2 x NZ level voxels) and family (H, V, U = no family) the TSV holds: labelled arc (m),
   excursions (sum |k|), persistent (|sum k| per stretch: the net sheet change; excursions - persistent = there-and-back), and for each
   pair distance the pairs within half a turn / pairs (pairs every d / 4 samples). The totals are printed. Single thread, CPU. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "afv.h"
#include "fit.h"
#include "store.h"
#include "wrapdump.h"
#include "zarr3.h"

enum { FJ_MAXD = 8, FJ_CELL = 8 };
static const char *fj_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 3; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
static int fj_opti(int argc, char **argv, const char *k) { for (int i = 3; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return i; return 0; }

/* ---- nearest wrap vertex: a cell grid over one tile (+ margin), one per wrap-id parity ---- */
typedef struct { int n[3]; double o[3]; uint32_t *start; uint32_t *idx; const float *v; size_t nv; } fj_grid;
static void fj_grid_build(fj_grid *g, const float *rec, size_t nrec, const uint32_t *sel, size_t nsel, const double lo[3], const double hi[3]) {
    for (int d = 0; d < 3; d++) { g->o[d] = lo[d]; g->n[d] = (int)((hi[d] - lo[d]) / FJ_CELL) + 1; }
    const size_t nc = (size_t)g->n[0] * g->n[1] * g->n[2]; g->start = calloc(nc + 1, sizeof *g->start); g->idx = malloc((nsel + 1) * sizeof *g->idx); g->v = rec; g->nv = nsel;
    uint32_t *cell = malloc((nsel + 1) * sizeof *cell);
    for (size_t i = 0; i < nsel; i++) { const float *p = rec + 5 * (size_t)sel[i]; int c[3];
        for (int d = 0; d < 3; d++) { c[d] = (int)((p[d] - g->o[d]) / FJ_CELL); if (c[d] < 0) c[d] = 0; if (c[d] >= g->n[d]) c[d] = g->n[d] - 1; }
        cell[i] = (uint32_t)(((size_t)c[0] * g->n[1] + c[1]) * g->n[2] + c[2]); g->start[cell[i] + 1]++; }
    for (size_t c = 0; c < nc; c++) g->start[c + 1] += g->start[c];
    uint32_t *fill = malloc(nc * sizeof *fill); memcpy(fill, g->start, nc * sizeof *fill);
    for (size_t i = 0; i < nsel; i++) g->idx[fill[cell[i]]++] = sel[i];
    free(fill); free(cell);
}
static void fj_grid_free(fj_grid *g) { free(g->start); free(g->idx); g->start = nullptr; g->idx = nullptr; }
/* the nearest vertex within R: its record index or -1, *dist its distance. Cells are visited in Chebyshev rings about the query's cell;
   after ring r every unvisited vertex is at least r * FJ_CELL away. */
static long fj_nearest(const fj_grid *g, const double p[3], double R, double *dist) {
    int c[3]; for (int d = 0; d < 3; d++) { c[d] = (int)floor((p[d] - g->o[d]) / FJ_CELL); if (c[d] < -1 || c[d] > g->n[d]) return -1; }
    double best = R * R; long bi = -1; const int rmax = (int)ceil(R / FJ_CELL) + 1;
    for (int r = 0; r <= rmax; r++) {
        if (r > 0 && best <= (double)(r - 1) * FJ_CELL * (r - 1) * FJ_CELL) break;
        for (int a = c[0] - r; a <= c[0] + r; a++) { if (a < 0 || a >= g->n[0]) continue;
            for (int b = c[1] - r; b <= c[1] + r; b++) { if (b < 0 || b >= g->n[1]) continue;
                const int shell = abs(a - c[0]) == r || abs(b - c[1]) == r;
                for (int e = c[2] - r; e <= c[2] + r; e += shell ? 1 : 2 * r > 0 ? 2 * r : 1) { if (e < 0 || e >= g->n[2]) continue;
                    const size_t ci = ((size_t)a * g->n[1] + b) * g->n[2] + e;
                    for (uint32_t k = g->start[ci]; k < g->start[ci + 1]; k++) { const float *v = g->v + 5 * (size_t)g->idx[k];
                        const double dz = v[0] - p[0], dy = v[1] - p[1], dx = v[2] - p[2], dd = dz * dz + dy * dy + dx * dx;
                        if (dd < best) { best = dd; bi = (long)g->idx[k]; } }
                }
            }
        }
    }
    if (bi >= 0) *dist = sqrt(best);
    return bi;
}

typedef struct {
    int mode;                         /* 0 wraps, 1 phase raster, 2 fit set */
    fj_grid g[2]; const float *rec; int wdir; double R;
    const uint8_t *ph; int64_t po[3], pn[3];
    const fit_set *set; double sc;
} fj_src;
/* band phase (0..1) at a level point (global z, y, x), or -1 */
static double fj_phase(const fj_src *S, const double p[3]) {
    if (S->mode == 1) {
        const int64_t q[3] = {llround(p[0]) - S->po[0], llround(p[1]) - S->po[1], llround(p[2]) - S->po[2]};
        if (q[0] < 0 || q[1] < 0 || q[2] < 0 || q[0] >= S->pn[0] || q[1] >= S->pn[1] || q[2] >= S->pn[2]) return -1;
        const int v = S->ph[((size_t)q[0] * S->pn[1] + q[1]) * S->pn[2] + q[2]]; return v <= 251 ? v / 252.0 : -1;
    }
    if (S->mode == 2) {
        const fit_model *m = S->set->global; double n[3];
        for (int d = 0; d < 3; d++) n[d] = (p[d] + 0.5) * S->sc - 0.5 - (double)m->origin[d];
        if (n[0] < 0 || n[1] < 0 || n[2] < 0 || n[0] > m->D * m->h || n[1] > m->H * m->h || n[2] > m->W * m->h) return -1;
        const double w = fit_set_W(S->set, n[0], n[1], n[2]); return isfinite(w) ? w - floor(w) : -1;
    }
    double da, db; const long ia = fj_nearest(&S->g[0], p, S->R, &da), ib = fj_nearest(&S->g[1], p, S->R, &db);
    if (ia < 0 || ib < 0) return -1;
    const float *va = S->rec + 5 * ia, *vb = S->rec + 5 * ib; const double qa = -S->wdir * va[3] + va[4], qb = -S->wdir * vb[3] + vb[4];
    const double d1 = da <= db ? da : db, d2 = da <= db ? db : da, dq = da <= db ? qb - qa : qa - qb;
    if (!(fabs(dq) >= 0.5 && fabs(dq) < 1.5) || !(d1 + d2 > 0)) return -1;
    const double f = (dq > 0 ? 1 : -1) * d1 / (d1 + d2); return f - floor(f);
}

typedef struct { double arc, exc, pers; double pn[FJ_MAXD], pa[FJ_MAXD]; long nfib; } fj_acc;
static int fj_cmp(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
/* one labelled stretch: phases phi[0..n) (turns) -> the accumulators */
static void fj_stretch(const double *phi, int n, double *u, double *um, const int *D, int nd, double hyst, fj_acc *A) {
    u[0] = 0; for (int i = 1; i < n; i++) { const double d = phi[i] - phi[i - 1]; u[i] = u[i - 1] + d - floor(d + 0.5); }
    for (int i = 0; i < n; i++) { double w[9]; for (int k = -4; k <= 4; k++) { const int j = i + k; w[k + 4] = u[j < 0 ? 0 : j >= n ? n - 1 : j]; } qsort(w, 9, sizeof(double), fj_cmp); um[i] = w[4]; }
    double b = um[0], sk = 0, sa = 0;
    for (int i = 0; i < n; i++) if (fabs(um[i] - b) > hyst) { const double k = round(um[i] - b); sa += fabs(k); sk += k; b += k; }
    A->arc += n; A->exc += sa; A->pers += fabs(sk);
    for (int di = 0; di < nd; di++) { const int d = D[di], st = d / 4 > 1 ? d / 4 : 1;
        for (int i = 0; i + d < n; i += st) { A->pn[di]++; A->pa[di] += fabs(um[i + d] - um[i]) < 0.5; } }
}

int cmd_fiber_judge(int argc, char **argv) {
    const int wi = fj_opti(argc, argv, "--wraps"), pi = fj_opti(argc, argv, "--phase"), si = fj_opti(argc, argv, "--set");
    double z0d, nzd; const double um = atof(fj_opt(argc, argv, "--um", "0"));
    if (argc < 5 || !(wi || pi || si) || !(um > 0) || sscanf(fj_opt(argc, argv, "--slab", ""), "%lf,%lf", &z0d, &nzd) != 2) {
        fprintf(stderr, "usage: ufsm fiber-judge <fibres.afv|.fib> (--wraps DIR [--first K --last K --dir 1] | --phase ROOT KEY | --set FITSET) --um U [--level 0] [--tr T.json]\n"
                        "       --slab Z0,NZ [--region Y0,Y1,X0,X1] [--tile 512] [--d-mm 0.3,0.6,1.2,2.4 | --d-vox ..] [--hyst 0.75] [--out rows.tsv]\n");
        return 2;
    }
    const int level = atoi(fj_opt(argc, argv, "--level", "0")), tile = atoi(fj_opt(argc, argv, "--tile", "512")); const double sc = ldexp(1.0, level), lum = um * sc, hyst = atof(fj_opt(argc, argv, "--hyst", "0.75"));
    const int64_t z0 = (int64_t)z0d, nz = (int64_t)nzd; double reg[4] = {0, 0, 0, 0}; const int has_reg = sscanf(fj_opt(argc, argv, "--region", ""), "%lf,%lf,%lf,%lf", &reg[0], &reg[1], &reg[2], &reg[3]) == 4;
    if (tile < 16 || nz < 1 || level < 0 || level > 8) { fprintf(stderr, "fiber-judge: bad --tile / --slab / --level\n"); return 2; }
    int D[FJ_MAXD], nd = 0; double Dmm[FJ_MAXD];
    { const char *dv = fj_opt(argc, argv, "--d-vox", nullptr); char *t = strdup(dv ? dv : fj_opt(argc, argv, "--d-mm", "0.3,0.6,1.2,2.4"));
      for (char *q = strtok(t, ","); q && nd < FJ_MAXD; q = strtok(nullptr, ",")) { D[nd] = dv ? atoi(q) : (int)lround(atof(q) * 1000 / lum); if (D[nd] < 1) D[nd] = 1; Dmm[nd] = D[nd] * lum / 1000; nd++; }
      free(t); }
    /* fibres */
    double M[3][4], Mi[3][3]; const char *trp = fj_opt(argc, argv, "--tr", nullptr); const int has_tr = trp != nullptr;
    if (has_tr && fc_transform(trp, M, Mi)) { fprintf(stderr, "fiber-judge: cannot read the transform %s\n", trp); return 1; }
    afv *A = nullptr; fib_set F = {0}; double (*fbb)[6] = nullptr;
    { FILE *f = fopen(argv[2], "rb"); char mg[4] = ""; if (!f || fread(mg, 1, 4, f) != 4) { fprintf(stderr, "fiber-judge: cannot read %s\n", argv[2]); return 1; } fclose(f);
      if (!memcmp(mg, "UFIB", 4)) {
          if (fib_load(argv[2], &F)) { fprintf(stderr, "fiber-judge: bad UFIB file %s\n", argv[2]); return 1; }
          fbb = malloc(((size_t)F.n + 1) * sizeof *fbb);   /* per run: the level-L bounding box */
          for (int r = 0; r < F.n; r++) { for (int d = 0; d < 3; d++) { fbb[r][d] = 1e30; fbb[r][3 + d] = -1e30; }
              for (int i = F.off[r]; i < F.off[r + 1]; i++) for (int d = 0; d < 3; d++) { const double v = (F.pts[3 * i + d] + 0.5) / sc - 0.5; if (v < fbb[r][d]) fbb[r][d] = v; if (v > fbb[r][3 + d]) fbb[r][3 + d] = v; } }
      } else if (!(A = afv_open(argv[2]))) return 1; }
    /* the winding source */
    fj_src S = {0}; float *rec = nullptr; size_t nrec = 0; z3 *pz = nullptr; fit_set *set = nullptr; int64_t shp[3] = {0, 0, 0};
    const double Mg = 44;   /* vertex margin about a tile (> the search radius 40) */
    S.R = 40; S.sc = sc; S.wdir = atoi(fj_opt(argc, argv, "--dir", "1")) < 0 ? -1 : 1;
    if (wi) {
        const double yx[4] = {has_reg ? reg[0] - Mg : -1e30, has_reg ? reg[1] + Mg : 1e30, has_reg ? reg[2] - Mg : -1e30, has_reg ? reg[3] + Mg : 1e30};
        const int nw = wraps_slab_load(argv[wi + 1], (double)z0 - Mg, (double)(z0 + nz) + Mg, yx, 1, lum, atoi(fj_opt(argc, argv, "--first", "-1000000000")), atoi(fj_opt(argc, argv, "--last", "1000000000")), &rec, &nrec);
        if (nw < 0) return 1;
        printf("fiber-judge: %d wraps, %zu vertices in the slab\n", nw, nrec); S.mode = 0; S.rec = rec;
        if (!has_reg) { reg[0] = reg[2] = 1e30; reg[1] = reg[3] = -1e30;
            for (size_t i = 0; i < nrec; i++) { const float *v = rec + 5 * i; if (v[1] < reg[0]) reg[0] = v[1]; if (v[1] > reg[1]) reg[1] = v[1]; if (v[2] < reg[2]) reg[2] = v[2]; if (v[2] > reg[3]) reg[3] = v[2]; }
            for (int k = 0; k < 4; k += 2) { reg[k] = floor(reg[k] / tile) * tile; reg[k + 1] = ceil(reg[k + 1] / tile) * tile; } }
    } else if (pi) {
        if (pi + 2 >= argc) { fprintf(stderr, "fiber-judge: --phase ROOT KEY\n"); return 2; }
        store *ps = store_open(argv[pi + 1]); pz = ps ? z3_open(ps, argv[pi + 2], nullptr) : nullptr; if (!pz) { fprintf(stderr, "fiber-judge: cannot open the phase %s %s\n", argv[pi + 1], argv[pi + 2]); return 1; }
        for (int d = 0; d < 3; d++) shp[d] = z3_meta_of(pz)->shape[d];
        S.mode = 1; if (!has_reg) { reg[0] = 0; reg[1] = (double)shp[1]; reg[2] = 0; reg[3] = (double)shp[2]; }
    } else {
        set = fit_set_load(argv[si + 1]); if (!set) { fprintf(stderr, "fiber-judge: cannot load the fit set %s\n", argv[si + 1]); return 1; }
        S.mode = 2; S.set = set; const fit_model *m = set->global;
        if (!has_reg) { reg[0] = floor(m->origin[1] / sc / tile) * tile; reg[1] = ceil((m->origin[1] + m->H * m->h) / sc / tile) * tile; reg[2] = floor(m->origin[2] / sc / tile) * tile; reg[3] = ceil((m->origin[2] + m->W * m->h) / sc / tile) * tile; }
    }
    const char *outp = fj_opt(argc, argv, "--out", nullptr); FILE *o = outp ? fopen(outp, "w") : nullptr; if (outp && !o) { fprintf(stderr, "fiber-judge: cannot write %s\n", outp); return 1; }
    if (o) { fprintf(o, "box\truns\tpts\tknown");
        for (int f = 0; f < 3; f++) { fprintf(o, "\t%c_arc_m\t%c_exc\t%c_pers", "HVU"[f], "HVU"[f], "HVU"[f]); for (int di = 0; di < nd; di++) fprintf(o, "\t%c_p%d", "HVU"[f], D[di]); }
        fprintf(o, "\n"); }
    fj_acc T[3]; memset(T, 0, sizeof T); long ntile = 0; double tot_pts = 0, tot_known = 0;
    uint8_t *pbuf = nullptr; uint32_t *sel[2] = {nullptr, nullptr}; size_t cap = 0; double *smp = nullptr, *phi = nullptr, *u = nullptr, *umf = nullptr;
    for (int64_t ty = (int64_t)reg[0]; ty + tile <= (int64_t)reg[1]; ty += tile) for (int64_t tx = (int64_t)reg[2]; tx + tile <= (int64_t)reg[3]; tx += tile) {
        const double lo[3] = {(double)z0, (double)ty, (double)tx}, hi[3] = {(double)(z0 + nz), (double)(ty + tile), (double)(tx + tile)};
        /* the tile's runs: AFV blocks meeting it (query box = the tile's corners in AFV coordinates) or UFIB runs by bounding box */
        afv_run *R = nullptr; int nr = 0;
        if (A) {
            double qlo[3] = {1e30, 1e30, 1e30}, qhi[3] = {-1e30, -1e30, -1e30};
            for (int c = 0; c < 8; c++) { const double lz = (c & 4 ? hi[0] : lo[0]), ly = (c & 2 ? hi[1] : lo[1]), lx = (c & 1 ? hi[2] : lo[2]);
                const double g[3] = {(lx + 0.5) * sc - 0.5, (ly + 0.5) * sc - 0.5, (lz + 0.5) * sc - 0.5}; double a[3];
                for (int d = 0; d < 3; d++) a[d] = has_tr ? M[d][0] * g[0] + M[d][1] * g[1] + M[d][2] * g[2] + M[d][3] : g[d];
                for (int d = 0; d < 3; d++) { if (a[d] < qlo[d]) qlo[d] = a[d]; if (a[d] > qhi[d]) qhi[d] = a[d]; } }
            nr = afv_query(A, qlo, qhi, &R); if (nr < 0) return 1;
        } else {
            int m = 0; for (int r = 0; r < F.n; r++) if (fbb[r][3] >= lo[0] && fbb[r][0] < hi[0] && fbb[r][4] >= lo[1] && fbb[r][1] < hi[1] && fbb[r][5] >= lo[2] && fbb[r][2] < hi[2]) m++;
            R = calloc((size_t)m + 1, sizeof *R);
            for (int r = 0; r < F.n; r++) if (fbb[r][3] >= lo[0] && fbb[r][0] < hi[0] && fbb[r][4] >= lo[1] && fbb[r][1] < hi[1] && fbb[r][5] >= lo[2] && fbb[r][2] < hi[2]) {
                afv_run *q = &R[nr++]; q->fiber = r; q->family = F.fam[r] == 'H' || F.fam[r] == 'V' ? F.fam[r] : '?'; q->n = F.off[r + 1] - F.off[r]; q->xyz = malloc((size_t)q->n * 24 + 8);
                for (int i = 0; i < q->n; i++) { const float *p = F.pts + 3 * (size_t)(F.off[r] + i); q->xyz[3 * i] = p[2]; q->xyz[3 * i + 1] = p[1]; q->xyz[3 * i + 2] = p[0]; } }
        }
        if (nr == 0) { afv_runs_free(R, nr); continue; }
        /* the tile's winding source */
        if (S.mode == 0) {
            size_t ns[2] = {0, 0}; const double glo[3] = {lo[0] - Mg, lo[1] - Mg, lo[2] - Mg}, ghi[3] = {hi[0] + Mg, hi[1] + Mg, hi[2] + Mg};
            for (int pass = 0; pass < 2; pass++) {
                if (pass) { for (int k = 0; k < 2; k++) { free(sel[k]); sel[k] = malloc((ns[k] + 1) * sizeof(uint32_t)); ns[k] = 0; } }
                for (size_t i = 0; i < nrec; i++) { const float *v = rec + 5 * i;
                    if (!(v[4] > 0) || v[0] < glo[0] || v[0] >= ghi[0] || v[1] < glo[1] || v[1] >= ghi[1] || v[2] < glo[2] || v[2] >= ghi[2]) continue;   /* (column 0 is a closing duplicate) */
                    const int k = (int)v[3] & 1; if (pass) sel[k][ns[k]] = (uint32_t)i; ns[k]++; }
            }
            if (ns[0] < 100 || ns[1] < 100) { afv_runs_free(R, nr); continue; }
            for (int k = 0; k < 2; k++) fj_grid_build(&S.g[k], rec, nrec, sel[k], ns[k], glo, ghi);
        } else if (S.mode == 1) {
            for (int d = 0; d < 3; d++) { S.po[d] = (int64_t)lo[d] < 0 ? 0 : (int64_t)lo[d]; const int64_t e = (int64_t)hi[d] > shp[d] ? shp[d] : (int64_t)hi[d]; S.pn[d] = e - S.po[d]; }
            if (S.pn[0] < 1 || S.pn[1] < 1 || S.pn[2] < 1) { afv_runs_free(R, nr); continue; }
            pbuf = realloc(pbuf, (size_t)S.pn[0] * S.pn[1] * S.pn[2]);
            if (!pbuf || z3_read(pz, S.po, S.pn, pbuf, 1)) { fprintf(stderr, "fiber-judge: phase read failed\n"); return 1; }
            S.ph = pbuf;
        }
        fj_acc B[3]; memset(B, 0, sizeof B); long pts = 0, known = 0;
        for (int r = 0; r < nr; r++) {
            const afv_run *q = &R[r]; const int fam = q->family == 'H' ? 0 : q->family == 'V' ? 1 : 2; if (q->n < 2) continue;
            /* the run in level coordinates, resampled every level voxel of arc */
            double *lv = malloc((size_t)q->n * 24), len = 0;
            for (int i = 0; i < q->n; i++) { double g[3] = {q->xyz[3 * i], q->xyz[3 * i + 1], q->xyz[3 * i + 2]};
                if (has_tr) { const double t[3] = {g[0] - M[0][3], g[1] - M[1][3], g[2] - M[2][3]}; for (int d = 0; d < 3; d++) g[d] = Mi[d][0] * t[0] + Mi[d][1] * t[1] + Mi[d][2] * t[2]; }
                for (int d = 0; d < 3; d++) lv[3 * i + d] = (g[2 - d] + 0.5) / sc - 0.5;
                if (i) len += sqrt(pow(lv[3 * i] - lv[3 * i - 3], 2) + pow(lv[3 * i + 1] - lv[3 * i - 2], 2) + pow(lv[3 * i + 2] - lv[3 * i - 1], 2)); }
            const int ns = (int)floor(len) + 1; if (len < 2) { free(lv); continue; }
            if ((size_t)ns > cap) { cap = (size_t)ns * 2; smp = realloc(smp, cap * 24); phi = realloc(phi, cap * 8); u = realloc(u, cap * 8); umf = realloc(umf, cap * 8); }
            { int seg = 1; double s0 = 0, sl = sqrt(pow(lv[3] - lv[0], 2) + pow(lv[4] - lv[1], 2) + pow(lv[5] - lv[2], 2));
              for (int k = 0; k < ns; k++) { const double t = (double)k;
                  while (seg < q->n - 1 && t > s0 + sl) { s0 += sl; seg++; sl = sqrt(pow(lv[3 * seg] - lv[3 * seg - 3], 2) + pow(lv[3 * seg + 1] - lv[3 * seg - 2], 2) + pow(lv[3 * seg + 2] - lv[3 * seg - 1], 2)); }
                  const double w = sl > 0 ? (t - s0) / sl : 0, ww = w < 0 ? 0 : w > 1 ? 1 : w;
                  for (int d = 0; d < 3; d++) smp[3 * k + d] = lv[3 * seg - 3 + d] + ww * (lv[3 * seg + d] - lv[3 * seg - 3 + d]); } }
            free(lv);
            int used = 0, a = -1;
            for (int k = 0; k <= ns; k++) {
                double f = -1;
                if (k < ns) { const double *p = smp + 3 * k; const int in = llround(p[0]) >= (int64_t)lo[0] && llround(p[0]) < (int64_t)hi[0] && llround(p[1]) >= (int64_t)lo[1] && llround(p[1]) < (int64_t)hi[1] && llround(p[2]) >= (int64_t)lo[2] && llround(p[2]) < (int64_t)hi[2];
                    if (in) { pts++; f = fj_phase(&S, p); if (f >= 0) known++; } }
                if (f >= 0) { if (a < 0) a = k; phi[k] = f; }
                else if (a >= 0) { if (k - a >= 16) { fj_stretch(phi + a, k - a, u, umf, D, nd, hyst, &B[fam]); used = 1; } a = -1; }
            }
            B[fam].nfib += used;
        }
        afv_runs_free(R, nr);
        if (S.mode == 0) for (int k = 0; k < 2; k++) fj_grid_free(&S.g[k]);
        if (pts == 0) continue;
        ntile++; tot_pts += (double)pts; tot_known += (double)known;
        if (o) { fprintf(o, "%lld,%lld,%lld,%lld,%d,%d\t%d\t%ld\t%.3f", (long long)z0, (long long)ty, (long long)tx, (long long)nz, tile, tile, nr, pts, (double)known / (double)pts);
            for (int f = 0; f < 3; f++) { fprintf(o, "\t%.4f\t%.0f\t%.0f", B[f].arc * lum * 1e-6, B[f].exc, B[f].pers); for (int di = 0; di < nd; di++) fprintf(o, "\t%.0f:%.0f", B[f].pa[di], B[f].pn[di]); }
            fprintf(o, "\n"); fflush(o); }
        for (int f = 0; f < 3; f++) { T[f].arc += B[f].arc; T[f].exc += B[f].exc; T[f].pers += B[f].pers; T[f].nfib += B[f].nfib; for (int di = 0; di < nd; di++) { T[f].pn[di] += B[f].pn[di]; T[f].pa[di] += B[f].pa[di]; } }
    }
    if (o) fclose(o);
    printf("fiber-judge: %ld tiles, %.0f fibre samples, %.3f with a winding phase; pairs within half a turn / pairs\n", ntile, tot_pts, tot_pts > 0 ? tot_known / tot_pts : 0);
    for (int f = 0; f < 3; f++) { if (T[f].arc == 0) continue; const double m = T[f].arc * lum * 1e-6;
        printf("  %c: arc %.2f m, excursions %.0f = %.1f /m, persistent %.0f = %.1f /m, there-and-back %.1f /m;", "HVU"[f], m, T[f].exc, T[f].exc / m, T[f].pers, T[f].pers / m, (T[f].exc - T[f].pers) / m);
        for (int di = 0; di < nd; di++) printf("  %.2f mm %.0f/%.0f = %.4f", Dmm[di], T[f].pa[di], T[f].pn[di], T[f].pn[di] > 0 ? T[f].pa[di] / T[f].pn[di] : 0);
        printf("\n"); }
    free(pbuf); free(sel[0]); free(sel[1]); free(smp); free(phi); free(u); free(umf); free(rec); free(fbb); if (A) afv_close(A); fib_free(&F); if (pz) z3_close(pz); if (set) fit_set_free(set);
    return 0;
}
