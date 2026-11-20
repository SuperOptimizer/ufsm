/* ufsm fiber-match OURS.fib REF.afv|REF.fib --box Z,Y,X,NZ,NY,NX [--um 2.4] [--map reg.json | --tr T.json] [--r-um 30] [--angle 30]
       [--step-um 10] [--ref-min-mm 0] [--any-family 0] [--out rows.tsv]
   Our traced fibres against a reference set (an AFV, e.g. Paris4_1mm_v2 through the ufsm reg map, or the spiral dataset's native
   fibres; or another UFIB) in one box of level-0 voxels (z, y, x) of our scan (--um per voxel). Both sets are resampled every --step-um
   along their arc and clipped to the box (a reference line leaving the box is split there). Per family (H, V; a known family only
   matches its own unless --any-family 1):
     lens        lines, metres, median, arc-median (half the arc lies in lines at least this long), the arc share in lines >= 1 / 2 mm
                 (ours and the reference's, the continuity measures of fibre_lens.py);
     recall      the share of reference arc with one of our samples within --r-um whose tangent is within --angle degrees;
     precision   the share of our arc with a reference sample so near and aligned;
     runs        along each reference line, its matched samples labelled with the nearest matching line of ours; a run = consecutive
                 samples matched to the same line (an unmatched sample or a switch ends it): the arc-median of the run lengths and the
                 matched arc share in runs >= 1 / 2 mm (how far one of our lines follows one true fibre).
   CPU, one thread (a hash of our samples in --r-um cells). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "afv.h"
#include "optcheck.h"
#include "reg.h"

static const char *fm_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 4; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(k))) return argv[i + 1]; return d; }

typedef struct { int n, cap; float *p; float *t; int *line; char *fam; } fm_smp;   /* samples: point (z, y, x), unit tangent, line id, family */
static void fm_push(fm_smp *S, const double p[3], const double t[3], int line, char fam) {
    if (S->n == S->cap) { S->cap = S->cap ? 2 * S->cap : 1 << 16; S->p = realloc(S->p, (size_t)S->cap * 12); S->t = realloc(S->t, (size_t)S->cap * 12); S->line = realloc(S->line, (size_t)S->cap * 4); S->fam = realloc(S->fam, (size_t)S->cap); }
    for (int a = 0; a < 3; a++) { S->p[3 * (size_t)S->n + a] = (float)p[a]; S->t[3 * (size_t)S->n + a] = (float)t[a]; }
    S->line[S->n] = line; S->fam[S->n] = fam; S->n++;
}
/* lines: per line its samples [ls[l], ls[l + 1]) in S, its length (mm) */
typedef struct { int n, cap; int *s0; double *len; char *fam; } fm_lines;
static void fm_newline(fm_lines *L, int s0, char fam) {
    if (L->n + 1 >= L->cap) { L->cap = L->cap ? 2 * L->cap : 1 << 12; L->s0 = realloc(L->s0, (size_t)(L->cap + 1) * 4); L->len = realloc(L->len, (size_t)L->cap * 8); L->fam = realloc(L->fam, (size_t)L->cap); }
    L->s0[L->n] = s0; L->len[L->n] = 0; L->fam[L->n] = fam; L->n++;
}
/* resample one polyline (zyx level-0 points) every h voxels, clipped to the box [lo, hi): a new line at each entry into the box */
static void fm_add_poly(fm_smp *S, fm_lines *L, const double *pts, int np, char fam, double h, const double lo[3], const double hi[3], double mm_per_vox) {
    int open = 0; double carry = 0;
    for (int k = 0; k + 1 < np; k++) {
        const double *a = pts + 3 * k, *b = a + 3; double d[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}; const double sl = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        if (!(sl > 0)) continue;
        for (int c = 0; c < 3; c++) d[c] /= sl;
        for (double s = carry; s < sl; s += h) {
            const double p[3] = {a[0] + s * d[0], a[1] + s * d[1], a[2] + s * d[2]};
            const int in = p[0] >= lo[0] && p[1] >= lo[1] && p[2] >= lo[2] && p[0] < hi[0] && p[1] < hi[1] && p[2] < hi[2];
            if (!in) { if (open) L->s0[L->n] = S->n; open = 0; continue; }
            if (!open) { fm_newline(L, S->n, fam); open = 1; } else L->len[L->n - 1] += h * mm_per_vox;
            fm_push(S, p, d, L->n - 1, fam); L->s0[L->n] = S->n;
        }
        carry = fmod(carry - sl, h); if (carry < 0) carry += h;
    }
}
static int dcmp(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
/* the lens of a set of lengths (mm): median, arc-median, arc shares >= 1 / 2 mm */
static void fm_lens(double *v, int n, double *med, double *amed, double *s1, double *s2, double *tot) {
    *med = *amed = *s1 = *s2 = *tot = 0; if (!n) return;
    qsort(v, (size_t)n, 8, dcmp); double t = 0, a1 = 0, a2 = 0; for (int i = 0; i < n; i++) { t += v[i]; if (v[i] >= 1) a1 += v[i]; if (v[i] >= 2) a2 += v[i]; }
    *med = v[n / 2]; double c = 0; for (int i = 0; i < n; i++) { c += v[i]; if (c >= t / 2) { *amed = v[i]; break; } }
    *tot = t; *s1 = t > 0 ? a1 / t : 0; *s2 = t > 0 ? a2 / t : 0;
}
/* a spatial hash of samples in cells of r voxels */
typedef struct { int64_t g[3]; double lo[3], r; int *head, *next; } fm_hash;
static int64_t fm_cell(const fm_hash *H, const float *p, int64_t c[3]) { for (int a = 0; a < 3; a++) { c[a] = (int64_t)floor((p[a] - H->lo[a]) / H->r); if (c[a] < 0) c[a] = 0; if (c[a] >= H->g[a]) c[a] = H->g[a] - 1; } return (c[0] * H->g[1] + c[1]) * H->g[2] + c[2]; }
static void fm_hash_build(fm_hash *H, const fm_smp *S, const double lo[3], const double hi[3], double r) {
    H->r = r; for (int a = 0; a < 3; a++) { H->lo[a] = lo[a]; H->g[a] = (int64_t)ceil((hi[a] - lo[a]) / r) + 1; }
    const size_t nc = (size_t)(H->g[0] * H->g[1] * H->g[2]); H->head = malloc(nc * 4); memset(H->head, 0xff, nc * 4); H->next = malloc((size_t)(S->n ? S->n : 1) * 4);
    for (int i = 0; i < S->n; i++) { int64_t c[3]; const int64_t k = fm_cell(H, S->p + 3 * (size_t)i, c); H->next[i] = H->head[k]; H->head[k] = i; }
}
/* the nearest sample of S within r of p, aligned within cmin (|cos|), same family unless anyf: index or -1. pref >= 0: a sample of line
   pref within reach wins over nearer ones (a run continues on its line while that line is still there: parallel duplicates do not cut it) */
static int fm_near(const fm_hash *H, const fm_smp *S, const float *p, const float *t, char fam, double r, double cmin, int anyf, int pref) {
    int64_t c[3]; fm_cell(H, p, c); int best = -1, bp = -1; double bd = r * r, bpd = r * r;
    for (int64_t dz = -1; dz <= 1; dz++) for (int64_t dy = -1; dy <= 1; dy++) for (int64_t dx = -1; dx <= 1; dx++) {
        const int64_t z = c[0] + dz, y = c[1] + dy, x = c[2] + dx; if (z < 0 || y < 0 || x < 0 || z >= H->g[0] || y >= H->g[1] || x >= H->g[2]) continue;
        for (int i = H->head[(z * H->g[1] + y) * H->g[2] + x]; i >= 0; i = H->next[i]) {
            if (!anyf && fam != '?' && S->fam[i] != '?' && S->fam[i] != fam) continue;
            const float *q = S->p + 3 * (size_t)i, *u = S->t + 3 * (size_t)i; const double e[3] = {q[0] - p[0], q[1] - p[1], q[2] - p[2]}, d2 = e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
            if (d2 > bd) continue;
            if (fabs((double)u[0] * t[0] + (double)u[1] * t[1] + (double)u[2] * t[2]) < cmin) continue;
            if (pref >= 0 && S->line[i] == pref && d2 <= bpd) { bpd = d2; bp = i; }
            if (d2 <= bd) { bd = d2; best = i; }
        }
    }
    return bp >= 0 ? bp : best;
}

int cmd_fiber_match(int argc, char **argv) {
    long long b[6];
    if (argc < 5 || sscanf(fm_opt(argc, argv, "--box", ""), "%lld,%lld,%lld,%lld,%lld,%lld", &b[0], &b[1], &b[2], &b[3], &b[4], &b[5]) != 6) {
        fprintf(stderr, "usage: ufsm fiber-match OURS.fib REF.afv|REF.fib --box Z,Y,X,NZ,NY,NX (level-0 voxels) [--um 2.4] [--map reg.json | --tr T.json]\n"
                        "       [--r-um 30] [--angle 30] [--step-um 10] [--ref-min-mm 0] [--any-family 0] [--out rows.tsv]\n");
        return 2;
    }
    const double um = opt_atof("--um", fm_opt(argc, argv, "--um", "2.4")), rum = opt_atof("--r-um", fm_opt(argc, argv, "--r-um", "30")), ang = opt_atof("--angle", fm_opt(argc, argv, "--angle", "30"));
    const double hum = opt_atof("--step-um", fm_opt(argc, argv, "--step-um", "10")), rmin = opt_atof("--ref-min-mm", fm_opt(argc, argv, "--ref-min-mm", "0"));
    const int anyf = opt_atoi("--any-family", fm_opt(argc, argv, "--any-family", "0")); const char *mapp = fm_opt(argc, argv, "--map", nullptr), *trp = fm_opt(argc, argv, "--tr", nullptr), *outp = fm_opt(argc, argv, "--out", nullptr);
    if (!(um > 0) || !(rum > 0) || !(hum > 0) || ang <= 0 || ang > 90) { fprintf(stderr, "fiber-match: bad --um / --r-um / --step-um / --angle\n"); return 2; }
    const double lo[3] = {(double)b[0], (double)b[1], (double)b[2]}, hi[3] = {(double)(b[0] + b[3]), (double)(b[1] + b[4]), (double)(b[2] + b[5])}, h = hum / um, mmv = um / 1000, cmin = cos(ang * M_PI / 180);
    fm_smp O = {0}, R = {0}; fm_lines LO = {0}, LR = {0};
    /* ours */
    { fib_set F; if (fib_load(argv[2], &F)) { fprintf(stderr, "fiber-match: cannot read %s\n", argv[2]); return 1; }
      double *pp = nullptr; size_t cp = 0;
      for (int r = 0; r < F.n; r++) { const int np = F.off[r + 1] - F.off[r]; if ((size_t)np * 3 > cp) { cp = (size_t)np * 3; pp = realloc(pp, cp * 8); }
          for (int k = 0; k < 3 * np; k++) pp[k] = F.pts[3 * (size_t)F.off[r] + k];
          fm_add_poly(&O, &LO, pp, np, F.fam[r], h, lo, hi, mmv); }
      free(pp); fib_free(&F); }
    /* the reference */
    const size_t rl = strlen(argv[3]);
    if (rl > 4 && !strcmp(argv[3] + rl - 4, ".fib")) {
        fib_set F; if (fib_load(argv[3], &F)) { fprintf(stderr, "fiber-match: cannot read %s\n", argv[3]); return 1; }
        double *pp = nullptr; size_t cp = 0;
        for (int r = 0; r < F.n; r++) { const int np = F.off[r + 1] - F.off[r]; if ((size_t)np * 3 > cp) { cp = (size_t)np * 3; pp = realloc(pp, cp * 8); }
            for (int k = 0; k < 3 * np; k++) pp[k] = F.pts[3 * (size_t)F.off[r] + k];
            fm_add_poly(&R, &LR, pp, np, F.fam[r], h, lo, hi, mmv); }
        free(pp); fib_free(&F);
    } else {
        const rmap *M = nullptr; double T[3][4], Ti[3][3];
        if (mapp && trp) { fprintf(stderr, "fiber-match: --map or --tr, not both\n"); return 2; }
        if (mapp && !(M = reg_map_load(mapp))) return 1;
        if (trp && fc_transform(trp, T, Ti)) { fprintf(stderr, "fiber-match: cannot read %s\n", trp); return 1; }
        /* the box in the reference's coordinates (x, y, z): the 8 corners mapped, + a margin */
        double qlo[3] = {1e300, 1e300, 1e300}, qhi[3] = {-1e300, -1e300, -1e300};
        for (int c = 0; c < 8; c++) {
            const double g[3] = {c & 1 ? hi[2] : lo[2], c & 2 ? hi[1] : lo[1], c & 4 ? hi[0] : lo[0]}; double a[3];
            if (M) rmap_f2c(M, g, a); else if (trp) for (int k = 0; k < 3; k++) a[k] = T[k][0] * g[0] + T[k][1] * g[1] + T[k][2] * g[2] + T[k][3]; else for (int k = 0; k < 3; k++) a[k] = g[k];
            for (int k = 0; k < 3; k++) { if (a[k] < qlo[k]) qlo[k] = a[k]; if (a[k] > qhi[k]) qhi[k] = a[k]; }
        }
        for (int k = 0; k < 3; k++) { const double m = 0.05 * (qhi[k] - qlo[k]) + 16; qlo[k] -= m; qhi[k] += m; }
        afv *A = afv_open(argv[3]); if (!A) return 1;
        afv_run *run = nullptr; const int nr = afv_query(A, qlo, qhi, &run); if (nr < 0) { fprintf(stderr, "fiber-match: AFV query failed\n"); return 1; }
        /* runs of one fibre: kept apart (a run is contiguous); points to our level-0 zyx */
        double *pp = nullptr; size_t cp = 0; int kept = 0;
        for (int r = 0; r < nr; r++) {
            if ((size_t)run[r].n * 3 > cp) { cp = (size_t)run[r].n * 3; pp = realloc(pp, cp * 8); }
            for (int k = 0; k < run[r].n; k++) { const double *a = run[r].xyz + 3 * (size_t)k; double g[3];
                if (M) rmap_c2f(M, a, g);
                else if (trp) { const double d[3] = {a[0] - T[0][3], a[1] - T[1][3], a[2] - T[2][3]}; for (int j = 0; j < 3; j++) g[j] = Ti[j][0] * d[0] + Ti[j][1] * d[1] + Ti[j][2] * d[2]; }
                else { g[0] = a[0]; g[1] = a[1]; g[2] = a[2]; }
                pp[3 * k] = g[2]; pp[3 * k + 1] = g[1]; pp[3 * k + 2] = g[0]; }
            const int l0 = LR.n; fm_add_poly(&R, &LR, pp, run[r].n, run[r].family, h, lo, hi, mmv); kept += LR.n > l0;
        }
        free(pp); afv_runs_free(run, nr); afv_close(A);
        fprintf(stderr, "fiber-match: %d reference runs met the box, %d inside\n", nr, kept);
    }
    if (rmin > 0) {   /* --ref-min-mm: reference lines (box pieces) shorter than this are dropped from recall / runs (their samples get family 0) */
        for (int l = 0; l < LR.n; l++) if (LR.len[l] < rmin) for (int i = LR.s0[l]; i < LR.s0[l + 1]; i++) R.fam[i] = 0;
    }
    fm_hash HO, HR; const double r = rum / um; fm_hash_build(&HO, &O, lo, hi, r); fm_hash_build(&HR, &R, lo, hi, r);
    FILE *ot = outp ? fopen(outp, "w") : nullptr; if (ot) fprintf(ot, "family\tset\tlines\tm\tmedian_mm\tarc_median_mm\tge1\tge2\trecall_or_precision\truns_arc_median_mm\truns_ge1\truns_ge2\n");
    printf("fiber-match: %s vs %s, box %lld %lld %lld + %lld %lld %lld at %.3g um, r %.0f um, angle %.0f deg, step %.0f um\n", argv[2], argv[3], b[0], b[1], b[2], b[3], b[4], b[5], um, rum, ang, hum);
    for (int fi = 0; fi < 2; fi++) {
        const char fam = fi ? 'V' : 'H';
        /* lens */
        double med, am, s1, s2, tot; double *v = malloc((size_t)(LO.n + LR.n + 1) * 8); int n = 0;
        for (int l = 0; l < LO.n; l++) if (LO.fam[l] == fam) v[n++] = LO.len[l];
        fm_lens(v, n, &med, &am, &s1, &s2, &tot); const int no = n; const double oo[5] = {tot, med, am, s1, s2};
        n = 0; for (int l = 0; l < LR.n; l++) if (LR.fam[l] == fam && LR.len[l] >= rmin) v[n++] = LR.len[l];
        fm_lens(v, n, &med, &am, &s1, &s2, &tot); const int nrf = n; const double rr[5] = {tot, med, am, s1, s2};
        /* recall + runs (along each reference line) */
        size_t rs = 0, rh = 0; int nrun = 0, caprun = 1024; double *runs = malloc((size_t)caprun * 8);
        for (int l = 0; l < LR.n; l++) {
            if (LR.fam[l] != fam || LR.len[l] < rmin) continue;
            int cur = -1; double cl = 0;
            for (int i = LR.s0[l]; i < LR.s0[l + 1]; i++) {
                const int j = fm_near(&HO, &O, R.p + 3 * (size_t)i, R.t + 3 * (size_t)i, fam, r, cmin, anyf, cur); rs++;
                const int id = j >= 0 ? O.line[j] : -1; rh += j >= 0;
                if (id != cur || id < 0) { if (cur >= 0 && cl > 0) { if (nrun == caprun) runs = realloc(runs, (size_t)(caprun *= 2) * 8); runs[nrun++] = cl; } cur = id; cl = 0; }
                else cl += h * mmv;
            }
            if (cur >= 0 && cl > 0) { if (nrun == caprun) runs = realloc(runs, (size_t)(caprun *= 2) * 8); runs[nrun++] = cl; }
        }
        double rmed, ram, r1, r2, rtot; fm_lens(runs, nrun, &rmed, &ram, &r1, &r2, &rtot);
        /* precision */
        size_t os = 0, oh = 0;
        for (int i = 0; i < O.n; i++) { if (O.fam[i] != fam) continue; os++; oh += fm_near(&HR, &R, O.p + 3 * (size_t)i, O.t + 3 * (size_t)i, fam, r, cmin, anyf, -1) >= 0; }
        printf("  %c ours: %d lines, %.2f m, median %.2f, arc-median %.2f mm, >=1mm %.3f >=2mm %.3f | ref: %d lines, %.2f m, median %.2f, arc-median %.2f, >=1mm %.3f >=2mm %.3f\n",
               fam, no, oo[0] / 1000, oo[1], oo[2], oo[3], oo[4], nrf, rr[0] / 1000, rr[1], rr[2], rr[3], rr[4]);
        printf("  %c recall %.3f (%zu / %zu ref samples), precision %.3f (%zu / %zu); runs on ref lines: %d, %.2f m, arc-median %.2f mm, >=1mm %.3f >=2mm %.3f\n",
               fam, rs ? (double)rh / rs : 0, rh, rs, os ? (double)oh / os : 0, oh, os, nrun, rtot / 1000, ram, r1, r2);
        if (ot) { fprintf(ot, "%c\tours\t%d\t%.4f\t%.3f\t%.3f\t%.4f\t%.4f\t%.4f\t%.3f\t%.4f\t%.4f\n", fam, no, oo[0] / 1000, oo[1], oo[2], oo[3], oo[4], os ? (double)oh / os : 0, ram, r1, r2);
                  fprintf(ot, "%c\tref\t%d\t%.4f\t%.3f\t%.3f\t%.4f\t%.4f\t%.4f\t\t\t\n", fam, nrf, rr[0] / 1000, rr[1], rr[2], rr[3], rr[4], rs ? (double)rh / rs : 0); }
        free(v); free(runs);
    }
    if (ot) fclose(ot);
    free(HO.head); free(HO.next); free(HR.head); free(HR.next);
    free(O.p); free(O.t); free(O.line); free(O.fam); free(R.p); free(R.t); free(R.line); free(R.fam); free(LO.s0); free(LO.len); free(LO.fam); free(LR.s0); free(LR.len); free(LR.fam);
    return 0;
}
