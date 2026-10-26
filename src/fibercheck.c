/* ufsm fiber-check <afv> --tr TRANSFORM.json --boxes BOXES.tsv --names A,B,.. --codes ROOT KEY --phase ROOT KEY
       [--level 2] [--fields-pat DIR/%s --fit-pat DIR/%s/fit] [--d 4,8,16,32,64,128] [--step 1] [--every 2] [--rad 3] [--packed-um 100]
       [--strict 0.1] [--seed 1] [--margin 8]
   Do published fibre polylines (AFV, e.g. community AutomatedFiberVolumes) carry same-sheet information? Field-level, no fitter:
   the fibres of each box (boxes.tsv rows: name scroll z y x nz ny nx level, voxels of --level of the label scan) are mapped from the
   AFV's scan into the label scan's level-0 voxels by the inverse of TRANSFORM (published transform.json: 3x4 "transformation_matrix"
   label-scan L0 xyz -> AFV-scan L0 xyz), resampled every --step level voxels along their arc, and labelled from the GT:
     phase  (band-field phase raster on the level grid: 0..251 = turn x 252 from the labelled recto face outward, 255 unknown),
     codes  (winding_mod14 raster: 1 + round(18 q) mod 252 on the labelled recto faces),
   GT winding at a point W = q(nearest code within --rad voxels) + phi - (phi >= 0.5) (phi < 0.5: outward of that recto).
   Pairs (a, b) at arc distance d on one fibre: the GT says one sheet when round(W_b - W_a) == 0 (the published codes include the
   azimuth: rounded across one sheet both ends move together). Baselines from the same a at the same chord distance r: "tangent"
   (a random direction in the GT sheet's tangent plane, perpendicular to grad phase), "straight" (the fibre's own tangent at a,
   extrapolated straight) and "rand3d" (any direction). Columns: n, P(one sheet) by the GT, and with a fit (wbench fields + fit dir)
   P(fit sheet difference != GT's) on all and on GT-one-sheet pairs (the same-sheet information a fibre would add). Packed: the GT
   spacing at a (1 / |grad phase|) under --packed-um. Strict: both ends within --strict turn of the box's circular-mean fibre phase
   (registration noise baseline). Also the phase of the fibre points by family (H vs V layer: which side of the sheet). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "afv.h"
#include "fit.h"
#include "json.h"
#include "sources.h"
#include "store.h"
#include "zarr3.h"

static const char *fc_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
static int fc_opt2(int argc, char **argv, const char *k) { for (int i = 1; i + 2 < argc; i++) if (!strcmp(argv[i], k)) return i; return 0; }
static uint64_t fc_s;
static double fc_u(void) { uint64_t z = (fc_s += 0x9E3779B97F4A7C15ull); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; z ^= z >> 31; return (z >> 11) * 0x1.0p-53; }
static double wrap1(double v) { return v - floor(v + 0.5); }   /* (-0.5, 0.5] */
static double wrap14(double v) { v = fmod(v + 7, 14); if (v < 0) v += 14; return v - 7; }

typedef struct {
    float p[3];      /* level voxels, global z y x */
    float t[3];      /* unit arc tangent */
    int run, fam;    /* fam 0 H, 1 V, 2 ? */
    int in;          /* inside the box */
    float phi, sp, W, Wf, ph;   /* ph: the network phase (fields pc / ps, turns; NaN unknown or confidence < 0.2) */
    double U; int gap;   /* U: the network phase unwrapped along the run; gap: unknown steps so far */
    /* GT phase (turns, -1 unknown), spacing (voxels, 0 unknown), GT winding (NaN), fit winding (NaN) */
    float g[3];      /* grad phase (turns / voxel) */
} fsamp;

typedef struct { int64_t o[3], n[3]; uint8_t *codes, *phase; } gtbox;

static inline int gt_at(const gtbox *B, const uint8_t *a, int64_t z, int64_t y, int64_t x) {
    z -= B->o[0]; y -= B->o[1]; x -= B->o[2];
    if (z < 0 || y < 0 || x < 0 || z >= B->n[0] || y >= B->n[1] || x >= B->n[2]) return -1;
    return a[((size_t)z * B->n[1] + y) * B->n[2] + x];
}
/* the GT at a level point: phase phi (turns), grad, spacing, winding W; returns 0 when the phase and a code are known */
static int gt_label(const gtbox *B, const float p[3], int rad, float *phi, float g[3], float *sp, float *W) {
    const int64_t c[3] = {lround(p[0]), lround(p[1]), lround(p[2])};
    const int v = gt_at(B, B->phase, c[0], c[1], c[2]); if (v < 0 || v > 251) return -1;
    const double ph = v / 252.0; *phi = (float)ph;
    double gg = 0;
    for (int d = 0; d < 3; d++) {
        int64_t a[3] = {c[0], c[1], c[2]}, b[3] = {c[0], c[1], c[2]}; a[d] -= 2; b[d] += 2;
        const int va = gt_at(B, B->phase, a[0], a[1], a[2]), vb = gt_at(B, B->phase, b[0], b[1], b[2]);
        g[d] = (va >= 0 && va <= 251 && vb >= 0 && vb <= 251) ? (float)(wrap1((vb - va) / 252.0) / 4) : NAN;
        gg += g[d] * g[d];
    }
    *sp = gg == gg && gg > 1e-8 ? (float)(1 / sqrt(gg)) : 0;
    double bd = 1e30; int code = -1;
    for (int dz = -rad; dz <= rad; dz++) for (int dy = -rad; dy <= rad; dy++) for (int dx = -rad; dx <= rad; dx++) {
        const int k = gt_at(B, B->codes, c[0] + dz, c[1] + dy, c[2] + dx); if (k < 1 || k > 252) continue;
        const double dd = dz * dz + dy * dy + dx * dx; if (dd < bd) { bd = dd; code = k; }
    }
    if (code < 0) return -1;
    *W = (float)((code - 1) / 18.0 + ph - (ph >= 0.5));
    return 0;
}

typedef struct { int64_t o[3], n[3]; uint8_t *v; } ctbox;
static double ct_tri(const ctbox *C, double z, double y, double x) {   /* trilinear, NAN outside */
    z -= C->o[0]; y -= C->o[1]; x -= C->o[2];
    const int64_t z0 = (int64_t)floor(z), y0 = (int64_t)floor(y), x0 = (int64_t)floor(x);
    if (z0 < 0 || y0 < 0 || x0 < 0 || z0 + 1 >= C->n[0] || y0 + 1 >= C->n[1] || x0 + 1 >= C->n[2]) return NAN;
    const double fz = z - z0, fy = y - y0, fx = x - x0; double s = 0;
    for (int c = 0; c < 8; c++) { const int a = c >> 2, b = (c >> 1) & 1, d = c & 1;
        s += (a ? fz : 1 - fz) * (b ? fy : 1 - fy) * (d ? fx : 1 - fx) * C->v[((size_t)(z0 + a) * C->n[1] + (y0 + b)) * C->n[2] + x0 + d]; }
    return s;
}
static double ct_score(const ctbox *C, const void *Sv, const size_t *idx, size_t m, const double t[3]);
enum { NSET = 4, MAXD = 12, NPK = 3, NST = 2 };   /* sets: fibre tangent straight rand3d; packed: all packed open; strict: all strict */
static const char *SETN[NSET] = {"fibre", "tangent", "straight", "rand3d"};
typedef struct { long n, same, nf, ferr[2], nfs, ferrs[2], nfd, nfds; } pacc;   /* nfd: fit says another sheet (round dW != 0), nfds: of those the GT says one sheet */   /* ferr per GT sign (+1, -1) */

int cmd_fiber_check(int argc, char **argv) {
    if (argc < 3 || !fc_opt2(argc, argv, "--codes") || !fc_opt2(argc, argv, "--phase")) {
        fprintf(stderr, "usage: ufsm fiber-check <afv> --tr T.json --boxes boxes.tsv --names A,B --codes ROOT KEY --phase ROOT KEY [--level 2] [--fields-pat D/%%s --fit-pat D/%%s/fit] [--d 4,8,16,32,64,128] [--step 1] [--every 2] [--rad 3] [--packed-um 100] [--strict 0.1] [--seed 1] [--samples out.tsv]\n");
        return 2;
    }
    const char *afp = argv[2], *trp = fc_opt(argc, argv, "--tr", nullptr), *btp = fc_opt(argc, argv, "--boxes", nullptr), *names = fc_opt(argc, argv, "--names", nullptr);
    const char *fdp = fc_opt(argc, argv, "--fields-pat", nullptr), *ftp = fc_opt(argc, argv, "--fit-pat", nullptr);
    const int level = atoi(fc_opt(argc, argv, "--level", "2")), every = atoi(fc_opt(argc, argv, "--every", "2")), rad = atoi(fc_opt(argc, argv, "--rad", "3"));
    const double step = atof(fc_opt(argc, argv, "--step", "1")), strict = atof(fc_opt(argc, argv, "--strict", "0.1")), margin = atof(fc_opt(argc, argv, "--margin", "8"));
    fc_s = (uint64_t)atoll(fc_opt(argc, argv, "--seed", "1")) * 0x2545F4914F6CDD1Dull + 7;
    int D[MAXD], nd = 0; { char *t = strdup(fc_opt(argc, argv, "--d", "4,8,16,32,64,128")); for (char *q = strtok(t, ","); q && nd < MAXD; q = strtok(nullptr, ",")) D[nd++] = atoi(q); free(t); }
    if (!trp || !btp || !names || !(step > 0) || every < 1) { fprintf(stderr, "fiber-check: --tr, --boxes, --names needed\n"); return 2; }
    const int ci = fc_opt2(argc, argv, "--codes"), pi = fc_opt2(argc, argv, "--phase");
    const double lum = atof(argv[ci + 2]), packed = atof(fc_opt(argc, argv, "--packed-um", "100")) / (lum > 0 ? lum : 9.6);
    /* the transform: label L0 xyz -> AFV xyz; inverted */
    double M[3][4], Mi[3][3];
    {
        FILE *f = fopen(trp, "rb"); if (!f) { fprintf(stderr, "fiber-check: cannot read %s\n", trp); return 1; }
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); char *t = malloc(n + 1); if (fread(t, 1, n, f) != (size_t)n) n = 0; t[n] = 0; fclose(f);
        json *j = json_parse(t, n); free(t); const json *m = j ? json_get(j, "transformation_matrix") : nullptr;
        if (!m || m->n < 3) { fprintf(stderr, "fiber-check: %s has no transformation_matrix\n", trp); return 1; }
        for (int r = 0; r < 3; r++) for (int c = 0; c < 4; c++) M[r][c] = json_num(json_at(json_at(m, r), c), NAN);
        json_free(j);
        const double a = M[0][0], b = M[0][1], c = M[0][2], d = M[1][0], e = M[1][1], f2 = M[1][2], g = M[2][0], h = M[2][1], k = M[2][2];
        const double det = a * (e * k - f2 * h) - b * (d * k - f2 * g) + c * (d * h - e * g);
        if (!(fabs(det) > 1e-12)) { fprintf(stderr, "fiber-check: singular transform\n"); return 1; }
        Mi[0][0] = (e * k - f2 * h) / det; Mi[0][1] = (c * h - b * k) / det; Mi[0][2] = (b * f2 - c * e) / det;
        Mi[1][0] = (f2 * g - d * k) / det; Mi[1][1] = (a * k - c * g) / det; Mi[1][2] = (c * d - a * f2) / det;
        Mi[2][0] = (d * h - e * g) / det; Mi[2][1] = (b * g - a * h) / det; Mi[2][2] = (a * e - b * d) / det;
    }
    const double sc = ldexp(1.0, level);
    afv *A = afv_open(afp); if (!A) return 1;
    const char *axp = fc_opt(argc, argv, "--axis", nullptr); axis axs = {0}; if (axp && axis_load(&axs, axp)) { fprintf(stderr, "fiber-check: cannot load the axis %s\n", axp); return 1; }
    long facing[3] = {0};
    z3 *ctz = nullptr; const double alR = atof(fc_opt(argc, argv, "--align", "0")); const int alB = atoi(fc_opt(argc, argv, "--align-block", "0"));
    { const int ti = fc_opt2(argc, argv, "--ct"); if (ti) { store *ts = store_open(argv[ti + 1]); ctz = ts ? z3_open(ts, argv[ti + 2], nullptr) : nullptr; if (!ctz) { fprintf(stderr, "fiber-check: cannot open the CT\n"); return 1; } } }
    store *cs = store_open(argv[ci + 1]), *ps = store_open(argv[pi + 1]);
    z3 *cz = cs ? z3_open(cs, argv[ci + 2], nullptr) : nullptr, *pz = ps ? z3_open(ps, argv[pi + 2], nullptr) : nullptr;
    if (!cz || !pz) { fprintf(stderr, "fiber-check: cannot open the codes / phase\n"); return 1; }
    FILE *so = nullptr; { const char *sp = fc_opt(argc, argv, "--samples", nullptr); if (sp && !(so = fopen(sp, "w"))) { fprintf(stderr, "fiber-check: cannot write %s\n", sp); return 1; }
        if (so) fprintf(so, "box\trun\tfam\tz\ty\tx\tphi\tsp\tW\tWf\n"); }

    static pacc T[NSET][MAXD][NPK][NST]; memset(T, 0, sizeof T);
    long hist[3][2][20] = {{{0}}}; double hc[3][2] = {{0}}, hs[3][2] = {{0}}; long hn[3][2] = {{0}};   /* phase of the fibre points by family, all / packed */
    long tot_s = 0, tot_l = 0;
    static long PA[MAXD][NPK][8]; memset(PA, 0, sizeof PA);   /* fibre pairs with a path phase: n, path one sheet, of those GT one, path other & GT one, fit other, fit other & path one, of those GT one, GT one */
    long hv[2][5] = {{0}}; double hvd[2] = {0};   /* H->V nearest same-sheet pairs: n, V outward, |offset| >= 1 voxel, of those outward, V phase larger; mean offset */
    char *nl = strdup(names);
    for (char *bn = strtok(nl, ","); bn; bn = strtok(nullptr, ",")) {
        /* the box row */
        FILE *bf = fopen(btp, "r"); if (!bf) { fprintf(stderr, "fiber-check: cannot read %s\n", btp); return 1; }
        char line[1024]; int64_t bo[3] = {0}, bs[3] = {0}; int found = 0;
        while (fgets(line, sizeof line, bf)) {
            if (line[0] == '#') continue;
            char nm[128], scl[128]; long long z, y, x, nz, ny, nx;
            if (sscanf(line, "%127s %127s %lld %lld %lld %lld %lld %lld", nm, scl, &z, &y, &x, &nz, &ny, &nx) == 8 && !strcmp(nm, bn)) {
                bo[0] = z; bo[1] = y; bo[2] = x; bs[0] = nz; bs[1] = ny; bs[2] = nx; found = 1; break; }
        }
        fclose(bf);
        if (!found) { fprintf(stderr, "fiber-check: no box %s\n", bn); continue; }
        /* the AFV bounding box of the box's corners */
        double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
        for (int c = 0; c < 8; c++) {
            const double pz = sc * (bo[0] + ((c & 1) ? bs[0] : 0) + 0.5) - 0.5, py = sc * (bo[1] + ((c & 2) ? bs[1] : 0) + 0.5) - 0.5, px = sc * (bo[2] + ((c & 4) ? bs[2] : 0) + 0.5) - 0.5;
            const double v[3] = {px, py, pz};
            for (int r = 0; r < 3; r++) { const double q = M[r][0] * v[0] + M[r][1] * v[1] + M[r][2] * v[2] + M[r][3]; if (q < lo[r]) lo[r] = q; if (q > hi[r]) hi[r] = q; }
        }
        for (int r = 0; r < 3; r++) { lo[r] -= margin; hi[r] += margin; }
        afv_run *R; const int nr = afv_query(A, lo, hi, &R); if (nr < 0) return 1;
        /* resample every run into level coordinates */
        size_t ns = 0, cap = 1 << 16; fsamp *S = malloc(cap * sizeof *S);
        long nH = 0, nV = 0;
        for (int i = 0; i < nr; i++) {
            const afv_run *r = &R[i]; if (r->n < 2) continue;
            double prev[3] = {0}, acc = 0, next = 0; const size_t s0 = ns;
            for (int k = 0; k < r->n; k++) {
                const double u[3] = {r->xyz[3 * k] - M[0][3], r->xyz[3 * k + 1] - M[1][3], r->xyz[3 * k + 2] - M[2][3]};
                double l0[3]; for (int a = 0; a < 3; a++) l0[a] = Mi[a][0] * u[0] + Mi[a][1] * u[1] + Mi[a][2] * u[2];   /* x y z */
                const double q[3] = {(l0[2] + 0.5) / sc - 0.5, (l0[1] + 0.5) / sc - 0.5, (l0[0] + 0.5) / sc - 0.5};   /* level z y x */
                if (k == 0) { memcpy(prev, q, sizeof prev); continue; }
                const double dv[3] = {q[0] - prev[0], q[1] - prev[1], q[2] - prev[2]}, L = sqrt(dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]);
                while (L > 0 && next <= acc + L) {
                    const double f = (next - acc) / L;
                    if (ns == cap) { cap *= 2; S = realloc(S, cap * sizeof *S); }
                    fsamp *s = &S[ns++]; memset(s, 0, sizeof *s);
                    for (int a = 0; a < 3; a++) { s->p[a] = (float)(prev[a] + f * dv[a]); s->t[a] = (float)(dv[a] / L); }
                    s->run = i; s->fam = r->family == 'H' ? 0 : r->family == 'V' ? 1 : 2; s->phi = -1; s->W = s->Wf = s->ph = NAN;
                    s->in = s->p[0] >= bo[0] && s->p[1] >= bo[1] && s->p[2] >= bo[2] && s->p[0] < bo[0] + bs[0] && s->p[1] < bo[1] + bs[1] && s->p[2] < bo[2] + bs[2];
                    next += step;
                }
                acc += L; memcpy(prev, q, sizeof prev);
            }
            for (size_t k = s0; k < ns; k++) if (S[k].in) { if (S[k].fam == 0) nH++; else if (S[k].fam == 1) nV++; }
        }
        afv_runs_free(R, nr);
        /* --ct ROOT KEY --align R: a local registration correction from the CT alone (no labels): per block of --align-block level voxels
           (0: the whole box) the translation within +-R level voxels that maximises the mean CT at the block's fibre points (fibres lie in
           papyrus), integer grid then 0.25-voxel steps around the best; every sample is moved by its block's shift */
        if (ctz && alR > 0) {
            ctbox C; const int hc = (int)ceil(alR) + 2;
            for (int d = 0; d < 3; d++) { C.o[d] = bo[d] - hc; C.n[d] = bs[d] + 2 * hc; if (C.o[d] < 0) { C.n[d] += C.o[d]; C.o[d] = 0; } }
            C.v = malloc((size_t)C.n[0] * C.n[1] * C.n[2]);
            if (!C.v || z3_read(ctz, C.o, C.n, C.v, 8)) { fprintf(stderr, "fiber-check: %s: CT read failed\n", bn); return 1; }
            int nb[3]; for (int d = 0; d < 3; d++) nb[d] = alB > 0 ? (int)((bs[d] + alB - 1) / alB) : 1;
            const int NB = nb[0] * nb[1] * nb[2]; float (*shf)[3] = calloc(NB, sizeof *shf);
            size_t *idx = malloc(ns * sizeof *idx);
            for (int b = 0; b < NB; b++) {
                const int bz = b / (nb[1] * nb[2]), by = (b / nb[2]) % nb[1], bx = b % nb[2]; size_t m = 0;
                for (size_t k = 0; k < ns; k++) {
                    if (!S[k].in) continue;
                    const int q[3] = {alB > 0 ? (int)((S[k].p[0] - bo[0]) / alB) : 0, alB > 0 ? (int)((S[k].p[1] - bo[1]) / alB) : 0, alB > 0 ? (int)((S[k].p[2] - bo[2]) / alB) : 0};
                    if (q[0] == bz && q[1] == by && q[2] == bx) idx[m++] = k;
                }
                if (m < 200) continue;
                const size_t stp = m > 30000 ? m / 30000 : 1; size_t mm = 0; for (size_t k = 0; k < m; k += stp) idx[mm++] = idx[k]; m = mm;
                double best = -1, bt[3] = {0, 0, 0}; const double s0 = ct_score(&C, S, idx, m, bt);
                for (int pass = 0; pass < 2; pass++) {
                    const double st = pass ? 0.25 : 1, lim = pass ? 1 : alR, c0[3] = {bt[0], bt[1], bt[2]};
                    for (double dz = -lim; dz <= lim + 1e-9; dz += st) for (double dy = -lim; dy <= lim + 1e-9; dy += st) for (double dx = -lim; dx <= lim + 1e-9; dx += st) {
                        const double t[3] = {c0[0] + dz, c0[1] + dy, c0[2] + dx}; const double s = ct_score(&C, S, idx, m, t);
                        if (s > best) { best = s; bt[0] = t[0]; bt[1] = t[1]; bt[2] = t[2]; }
                    }
                }
                for (int d = 0; d < 3; d++) shf[b][d] = (float)bt[d];
                printf("align %s block %d (%d %d %d): %zu points, shift z %+.2f y %+.2f x %+.2f level voxels, mean CT %.1f -> %.1f\n", bn, b, bz, by, bx, m, bt[0], bt[1], bt[2], s0, best);
            }
            for (size_t k = 0; k < ns; k++) {
                int q[3]; for (int d = 0; d < 3; d++) { q[d] = alB > 0 ? (int)floor((S[k].p[d] - bo[d]) / alB) : 0; if (q[d] < 0) q[d] = 0; if (q[d] >= nb[d]) q[d] = nb[d] - 1; }
                const float *t = shf[(q[0] * nb[1] + q[1]) * nb[2] + q[2]];
                for (int d = 0; d < 3; d++) S[k].p[d] += t[d];
                S[k].in = S[k].p[0] >= bo[0] && S[k].p[1] >= bo[1] && S[k].p[2] >= bo[2] && S[k].p[0] < bo[0] + bs[0] && S[k].p[1] < bo[1] + bs[1] && S[k].p[2] < bo[2] + bs[2];
            }
            free(idx); free(shf); free(C.v);
        }
        /* the GT over the box plus a halo */
        gtbox G; const int hal = rad + 3;
        for (int d = 0; d < 3; d++) { G.o[d] = bo[d] - hal; G.n[d] = bs[d] + 2 * hal; if (G.o[d] < 0) { G.n[d] += G.o[d]; G.o[d] = 0; } }
        const size_t nv = (size_t)G.n[0] * G.n[1] * G.n[2]; G.codes = malloc(nv); G.phase = malloc(nv);
        if (!G.codes || !G.phase || z3_read(cz, G.o, G.n, G.codes, 8) || z3_read(pz, G.o, G.n, G.phase, 8)) { fprintf(stderr, "fiber-check: %s: GT read failed\n", bn); return 1; }
        /* --ct + --axis: which way does the labelled face look? At every 5th labelled voxel the CT 1..2 level voxels outward (away from
           the umbilicus, radially) vs inward: papyrus outward of the label = the labelled (recto) face looks toward the umbilicus */
        if (ctz && axp) {
            ctbox C; C.o[0] = G.o[0]; C.o[1] = G.o[1]; C.o[2] = G.o[2]; C.n[0] = G.n[0]; C.n[1] = G.n[1]; C.n[2] = G.n[2]; C.v = malloc((size_t)C.n[0] * C.n[1] * C.n[2]);
            if (!C.v || z3_read(ctz, C.o, C.n, C.v, 8)) { fprintf(stderr, "fiber-check: %s: CT read failed\n", bn); return 1; }
            long nf = 0, fin = 0, fout = 0, nrd = 0; size_t cnt = 0;
            for (int64_t z = 0; z < G.n[0]; z++) {
                const double zg = z + G.o[0], z0l = sc * (zg + 0.5) - 0.5; double ay, ax2; axis_at(&axs, z0l, &ay, &ax2);
                const double cy = (ay + 0.5) / sc - 0.5, cx = (ax2 + 0.5) / sc - 0.5;
                for (int64_t y = 0; y < G.n[1]; y++) for (int64_t x = 0; x < G.n[2]; x++) {
                    const int c = G.codes[((size_t)z * G.n[1] + y) * G.n[2] + x]; if (c < 1 || c > 252 || (cnt++ % 5)) continue;
                    double ry = y + G.o[1] - cy, rx = x + G.o[2] - cx; const double rl = hypot(ry, rx); if (rl < 50) continue; ry /= rl; rx /= rl;
                    /* the label normal: grad phase (toward the next winding), its radial sign */
                    float gph, gg[3], gsp, gW; const float gp[3] = {(float)zg, (float)(y + G.o[1]), (float)(x + G.o[2])};
                    if (gt_label(&G, gp, 1, &gph, gg, &gsp, &gW)) continue;
                    const double gl = sqrt(gg[0] * gg[0] + gg[1] * gg[1] + gg[2] * gg[2]); if (!(gl > 1e-6)) continue;
                    const double nz = gg[0] / gl, ny = gg[1] / gl, nx = gg[2] / gl, rd = ny * ry + nx * rx;
                    if (fabs(rd) < 0.5) continue;   /* (sheets far from tangential to the axis: radial sense unclear) */
                    double o = 0, in = 0; int ok = 1;
                    for (int s = 1; s <= 2 && ok; s++) {
                        const double vo = ct_tri(&C, zg + s * nz, y + G.o[1] + s * ny, x + G.o[2] + s * nx), vi = ct_tri(&C, zg - s * nz, y + G.o[1] - s * ny, x + G.o[2] - s * nx);
                        if (!(vo == vo) || !(vi == vi)) { ok = 0; break; }
                        o += vo; in += vi;
                    }
                    if (!ok) continue;
                    if (rd < 0) { const double t = o; o = in; in = t; }   /* o: radially outward of the label */
                    nf++; fout += o > in + 4; fin += in > o + 4; nrd += rd > 0;
                }
            }
            printf("facing %s: %ld labelled voxels: papyrus outward of the label (recto faces the umbilicus) %.3f, inward %.3f, unclear %.3f, next winding outward %.3f\n", bn, nf, nf ? (double)fout / nf : NAN, nf ? (double)fin / nf : NAN, nf ? 1 - (double)(fout + fin) / nf : NAN, nf ? (double)nrd / nf : NAN);
            facing[0] += nf; facing[1] += fout; facing[2] += fin;
            free(C.v);
        }
        /* the fit (optional) */
        fit_set *F = nullptr; double fo[3] = {0}, fcf = 1; int fsh[3] = {0}; fit_fields_buf fb; memset(&fb, 0, sizeof fb);
        if (fdp && ftp) {
            char p1[1024], p2[1024]; snprintf(p1, sizeof p1, fdp, bn); snprintf(p2, sizeof p2, ftp, bn);
            memset(&fb, 0, sizeof fb);
            if (!fit_fields_load(p1, 0, 0, 1, &fb) && (F = fit_set_load(p2))) {
                fcf = lum / fb.um; for (int d = 0; d < 3; d++) { fo[d] = (double)fb.origin[d]; fsh[d] = fb.shape[d] * fb.h; }
            } else { fprintf(stderr, "fiber-check: %s: no fit (%s, %s)\n", bn, p1, p2); fit_fields_free(&fb); }
        }
        long nlab = 0; double mc = 0, ms = 0;
        for (size_t k = 0; k < ns; k++) {
            fsamp *s = &S[k]; if (!s->in) continue;
            if (gt_label(&G, s->p, rad, &s->phi, s->g, &s->sp, &s->W)) { s->phi = -1; s->W = NAN; continue; }
            nlab++; mc += cos(2 * M_PI * s->phi); ms += sin(2 * M_PI * s->phi);
            const int pk = s->sp > 0 && s->sp < packed;
            for (int q = 0; q < 2; q++) if (!q || pk) { hist[s->fam][q][(int)(s->phi * 20) % 20]++; hc[s->fam][q] += cos(2 * M_PI * s->phi); hs[s->fam][q] += sin(2 * M_PI * s->phi); hn[s->fam][q]++; }
            if (F) {
                const double l[3] = {s->p[0] * fcf - fo[0], s->p[1] * fcf - fo[1], s->p[2] * fcf - fo[2]};
                if (l[0] >= 0 && l[1] >= 0 && l[2] >= 0 && l[0] < fsh[0] && l[1] < fsh[1] && l[2] < fsh[2]) s->Wf = (float)fit_set_W(F, l[0], l[1], l[2]);
            }
            if (so) fprintf(so, "%s\t%d\t%c\t%.2f\t%.2f\t%.2f\t%.4f\t%.2f\t%.4f\t%.4f\n", bn, s->run, "HV?"[s->fam], s->p[0], s->p[1], s->p[2], s->phi, s->sp, s->W, s->Wf);
        }
        /* the network phase along every run (nearest field cell), unwrapped step by step; unknown steps counted */
        if (F) for (size_t k = 0; k < ns; k++) {
            fsamp *s = &S[k]; const double l[3] = {s->p[0] * fcf - fo[0], s->p[1] * fcf - fo[1], s->p[2] * fcf - fo[2]};
            if (l[0] >= 0 && l[1] >= 0 && l[2] >= 0 && l[0] < fsh[0] && l[1] < fsh[1] && l[2] < fsh[2]) {
                const size_t i = ((size_t)(int)(l[0] / fb.h) * fb.shape[1] + (int)(l[1] / fb.h)) * fb.shape[2] + (int)(l[2] / fb.h);
                const double c = (fb.a[0][i] - 128) / 127.0, sn = (fb.a[1][i] - 128) / 127.0;
                if (hypot(c, sn) >= 0.2) s->ph = (float)(atan2(sn, c) / (2 * M_PI));
            }
            const int cont = k > 0 && S[k - 1].run == s->run;
            s->gap = cont ? S[k - 1].gap : 0; s->U = cont ? S[k - 1].U : 0;
            if (!(s->ph == s->ph)) s->gap++;
            else if (cont) { int j = (int)k - 1; while (j >= 0 && S[j].run == s->run && !(S[j].ph == S[j].ph) && (int)k - j < 4) j--;
                if (j >= 0 && S[j].run == s->run && S[j].ph == S[j].ph) s->U = S[j].U + wrap1(s->ph - S[j].ph); }
        }
        const double pmode = atan2(ms, mc) / (2 * M_PI);
        const double pR = nlab ? sqrt(mc * mc + ms * ms) / nlab : 0;
        /* layer order: each labelled H point and its nearest labelled V point within 4 voxels on the same GT sheet: is V outward of H
           (along grad phase), and is its phase larger? */
        {
            const int cs4 = 4, gd[3] = {(int)(bs[0] / cs4) + 1, (int)(bs[1] / cs4) + 1, (int)(bs[2] / cs4) + 1};
            const size_t ng = (size_t)gd[0] * gd[1] * gd[2]; int64_t *head = malloc(ng * sizeof *head), *nxt = malloc(ns * sizeof *nxt);
            for (size_t i = 0; i < ng; i++) head[i] = -1;
            #define CELL(s, c) int c[3]; for (int d_ = 0; d_ < 3; d_++) { c[d_] = (int)floor(((s)->p[d_] - bo[d_]) / cs4); if (c[d_] < 0) c[d_] = 0; if (c[d_] >= gd[d_]) c[d_] = gd[d_] - 1; }
            for (size_t k = 0; k < ns; k++) { nxt[k] = -1; const fsamp *s = &S[k]; if (s->fam != 1 || !(s->W == s->W)) continue; CELL(s, c); const size_t ci_ = ((size_t)c[0] * gd[1] + c[1]) * gd[2] + c[2]; nxt[k] = head[ci_]; head[ci_] = (int64_t)k; }
            for (size_t k = 0; k < ns; k += every) {
                const fsamp *h = &S[k]; if (h->fam != 0 || !(h->W == h->W)) continue;
                const double gg = h->g[0] * h->g[0] + h->g[1] * h->g[1] + h->g[2] * h->g[2]; if (!(gg > 1e-8)) continue;
                CELL(h, c); double bd = 16.01; int64_t bv = -1;
                for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
                    const int q[3] = {c[0] + dz, c[1] + dy, c[2] + dx}; if (q[0] < 0 || q[1] < 0 || q[2] < 0 || q[0] >= gd[0] || q[1] >= gd[1] || q[2] >= gd[2]) continue;
                    for (int64_t v = head[((size_t)q[0] * gd[1] + q[1]) * gd[2] + q[2]]; v >= 0; v = nxt[v]) {
                        const fsamp *s = &S[v]; if (lround(wrap14(s->W - h->W)) != 0) continue;
                        const double dd = (s->p[0] - h->p[0]) * (s->p[0] - h->p[0]) + (s->p[1] - h->p[1]) * (s->p[1] - h->p[1]) + (s->p[2] - h->p[2]) * (s->p[2] - h->p[2]);
                        if (dd < bd) { bd = dd; bv = v; }
                    }
                }
                if (bv < 0) continue;
                const fsamp *v = &S[bv]; const int pk = h->sp > 0 && h->sp < packed;
                const double out = ((v->p[0] - h->p[0]) * h->g[0] + (v->p[1] - h->p[1]) * h->g[1] + (v->p[2] - h->p[2]) * h->g[2]) / sqrt(gg);
                const double dph = wrap1(v->phi - h->phi);
                for (int q = 0; q < 2; q++) if (!q || pk) { hv[q][0]++; hv[q][1] += out > 0; hv[q][2] += fabs(out) >= 1; hv[q][3] += fabs(out) >= 1 && out > 0; hv[q][4] += dph > 0; hvd[q] += out; }
            }
            #undef CELL
            free(head); free(nxt);
        }
        tot_s += ns; tot_l += nlab;
        /* pairs */
        long bn_[2] = {0}, bs_[2] = {0};   /* fibre pairs at d = 32: n, one sheet (per-box line) */
        for (size_t a = 0; a < ns; a += every) {
            const fsamp *sa = &S[a]; if (!(sa->W == sa->W)) continue;
            const int pka = sa->sp > 0 && sa->sp < packed, sta = fabs(wrap1(sa->phi - pmode)) < strict;
            for (int di = 0; di < nd; di++) {
                const size_t b = a + (size_t)lround(D[di] / step); if (b >= ns || S[b].run != sa->run) continue;
                const fsamp *sb = &S[b]; if (!sb->in) continue;
                const double r = sqrt((double)(sb->p[0] - sa->p[0]) * (sb->p[0] - sa->p[0]) + (double)(sb->p[1] - sa->p[1]) * (sb->p[1] - sa->p[1]) + (double)(sb->p[2] - sa->p[2]) * (sb->p[2] - sa->p[2]));
                for (int set = 0; set < NSET; set++) {
                    float pb[3], phib, gb[3], spb, Wb, Wfb = NAN;
                    if (set == 0) { if (!(sb->W == sb->W)) continue; memcpy(pb, sb->p, sizeof pb); phib = sb->phi; Wb = sb->W; Wfb = sb->Wf; }
                    else {
                        double u[3];
                        if (set == 2) for (int q = 0; q < 3; q++) u[q] = sa->t[q];
                        else {
                            do { for (int q = 0; q < 3; q++) u[q] = 2 * fc_u() - 1; } while (u[0] * u[0] + u[1] * u[1] + u[2] * u[2] > 1 || u[0] * u[0] + u[1] * u[1] + u[2] * u[2] < 1e-4);
                            if (set == 1) {
                                const double gg = sa->g[0] * sa->g[0] + sa->g[1] * sa->g[1] + sa->g[2] * sa->g[2]; if (!(gg > 1e-8)) continue;
                                const double dp = (u[0] * sa->g[0] + u[1] * sa->g[1] + u[2] * sa->g[2]) / gg; for (int q = 0; q < 3; q++) u[q] -= dp * sa->g[q];
                            }
                        }
                        const double ul = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]); if (!(ul > 1e-6)) continue;
                        for (int q = 0; q < 3; q++) pb[q] = (float)(sa->p[q] + r * u[q] / ul);
                        if (gt_label(&G, pb, rad, &phib, gb, &spb, &Wb)) continue;
                        if (F && sa->Wf == sa->Wf) {
                            const double l[3] = {pb[0] * fcf - fo[0], pb[1] * fcf - fo[1], pb[2] * fcf - fo[2]};
                            if (l[0] >= 0 && l[1] >= 0 && l[2] >= 0 && l[0] < fsh[0] && l[1] < fsh[1] && l[2] < fsh[2]) Wfb = (float)fit_set_W(F, l[0], l[1], l[2]);
                        }
                    }
                    const double dg = wrap14(Wb - sa->W); const int same = lround(dg) == 0;
                    const int st = sta && fabs(wrap1(phib - pmode)) < strict;
                    const int hasf = Wfb == Wfb && sa->Wf == sa->Wf; const double dm = Wfb - sa->Wf;
                    for (int q = 0; q < NPK; q++) {
                        if ((q == 1 && !pka) || (q == 2 && pka)) continue;
                        for (int t = 0; t < NST; t++) {
                            if (t == 1 && !st) continue;
                            pacc *P = &T[set][di][q][t]; P->n++; P->same += same;
                            if (hasf) { P->nf++; if (lround(dm) != 0) { P->nfd++; P->nfds += same; } for (int g = 0; g < 2; g++) P->ferr[g] += lround(dm - (g ? -1 : 1) * dg) != 0;
                                if (same) { P->nfs++; for (int g = 0; g < 2; g++) P->ferrs[g] += lround(dm - (g ? -1 : 1) * dg) != 0; } }
                        }
                    }
                    if (set == 0 && hasf && sb->gap - sa->gap <= 3 && sa->ph == sa->ph && sb->ph == sb->ph) {
                        const int po = lround(sb->U - sa->U) == 0, fo_ = lround(dm) != 0;
                        for (int q = 0; q < NPK; q++) { if ((q == 1 && !pka) || (q == 2 && pka)) continue; long *x = PA[di][q];
                            x[0]++; x[1] += po; x[2] += po && same; x[3] += !po && same; x[4] += fo_; x[5] += fo_ && po; x[6] += fo_ && po && same; x[7] += same; }
                    }
                    if (set == 0 && D[di] == 32) { bn_[pka]++; bs_[pka] += same; }
                }
            }
        }
        printf("box %s: %d runs, %zu samples (%ld in-box H, %ld V), %ld labelled; fibre phase circular mean %.3f R %.2f; d32 one-sheet open %ld/%ld packed %ld/%ld%s\n",
               bn, nr, ns, nH, nV, nlab, pmode < 0 ? pmode + 1 : pmode, pR, bs_[0], bn_[0], bs_[1], bn_[1], F ? "" : " (no fit)");
        fflush(stdout);
        free(S); free(G.codes); free(G.phase); if (F) { fit_set_free(F); fit_fields_free(&fb); }
    }
    free(nl);
    if (facing[0]) printf("facing (all boxes): %ld labelled voxels, recto faces the umbilicus %.3f, away %.3f\n", facing[0], (double)facing[1] / facing[0], (double)facing[2] / facing[0]);
    printf("\n%ld samples, %ld labelled (GT phase + code)\n", tot_s, tot_l);
    /* the GT sign (vs the fit): the one under which more fibre pairs agree */
    long ag[2] = {0}; for (int di = 0; di < nd; di++) for (int g = 0; g < 2; g++) ag[g] += T[0][di][0][0].nf - T[0][di][0][0].ferr[g];
    const int g = ag[1] > ag[0];
    printf("fit vs GT sign %+d (agree %ld vs %ld)\n", g ? -1 : 1, ag[g], ag[!g]);
    printf("set\td\tcells\tends\tn\tP_one_sheet\tn_fit\tP_fit_wrong\tn_fit_one\tP_fit_wrong_one\tn_fit_other\tP_one_if_fit_other\n");
    static const char *PK[NPK] = {"all", "packed", "open"}, *ST[NST] = {"all", "strict"};
    for (int set = 0; set < NSET; set++) for (int di = 0; di < nd; di++) for (int q = 0; q < NPK; q++) for (int t = 0; t < NST; t++) {
        const pacc *P = &T[set][di][q][t]; if (!P->n) continue;
        printf("%s\t%d\t%s\t%s\t%ld\t%.4f\t%ld\t%.4f\t%ld\t%.4f\t%ld\t%.4f\n", SETN[set], D[di], PK[q], ST[t], P->n, (double)P->same / P->n,
               P->nf, P->nf ? (double)P->ferr[g] / P->nf : NAN, P->nfs, P->nfs ? (double)P->ferrs[g] / P->nfs : NAN, P->nfd, P->nfd ? (double)P->nfds / P->nfd : NAN);
    }
    printf("\nfibre point phase by family (0 = labelled recto face, rising outward; 20 bins)\n");
    for (int f = 0; f < 3; f++) for (int q = 0; q < 2; q++) {
        if (!hn[f][q]) continue;
        double m = atan2(hs[f][q], hc[f][q]) / (2 * M_PI); if (m < 0) m += 1;
        printf("%c %s n %ld circ-mean %.3f R %.3f |", "HV?"[f], q ? "packed" : "all", hn[f][q], m, sqrt(hc[f][q] * hc[f][q] + hs[f][q] * hs[f][q]) / hn[f][q]);
        for (int b = 0; b < 20; b++) printf(" %.3f", (double)hist[f][q][b] / hn[f][q]);
        printf("\n");
    }
    printf("\nfibre pairs, network phase unwrapped along the fibre (path) vs the fit and the GT\nd\tcells\tn\tP_gt_one\tP_path_one\tP_gt_one_if_path_one\tP_gt_one_if_path_other\tn_fit_other\tP_path_one_if_fit_other\tP_gt_one_if_fit_other_path_one\n");
    for (int di = 0; di < nd; di++) for (int q = 0; q < NPK; q++) { const long *x = PA[di][q]; if (!x[0]) continue;
        printf("%d\t%s\t%ld\t%.4f\t%.4f\t%.4f\t%.4f\t%ld\t%.4f\t%.4f\n", D[di], PK[q], x[0], (double)x[7] / x[0], (double)x[1] / x[0], x[1] ? (double)x[2] / x[1] : NAN,
               x[0] - x[1] ? (double)x[3] / (x[0] - x[1]) : NAN, x[4], x[4] ? (double)x[5] / x[4] : NAN, x[5] ? (double)x[6] / x[5] : NAN); }
    for (int q = 0; q < 2; q++) if (hv[q][0]) printf("layer order (%s): %ld H points with a V point of the same sheet within 4 voxels: V outward of H (along grad phase) %.3f, mean offset %+.2f voxels; |offset| >= 1: %ld, outward %.3f; V phase larger %.3f\n",
        q ? "packed" : "all", hv[q][0], (double)hv[q][1] / hv[q][0], hvd[q] / hv[q][0], hv[q][2], hv[q][2] ? (double)hv[q][3] / hv[q][2] : NAN, (double)hv[q][4] / hv[q][0]);
    if (so) fclose(so);
    afv_close(A); z3_close(cz); z3_close(pz);
    return 0;
}

/* the published transform (label-scan L0 xyz -> AFV xyz) and its inverse */
static int fc_transform(const char *path, double M[3][4], double Mi[3][3]) {
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); char *t = malloc(n + 1); if (fread(t, 1, n, f) != (size_t)n) n = 0; t[n] = 0; fclose(f);
    json *j = json_parse(t, n); free(t); const json *m = j ? json_get(j, "transformation_matrix") : nullptr;
    if (!m || m->n < 3) { json_free(j); return -1; }
    for (int r = 0; r < 3; r++) for (int c = 0; c < 4; c++) M[r][c] = json_num(json_at(json_at(m, r), c), NAN);
    json_free(j);
    const double a = M[0][0], b = M[0][1], c = M[0][2], d = M[1][0], e = M[1][1], f2 = M[1][2], g = M[2][0], h = M[2][1], k = M[2][2];
    const double det = a * (e * k - f2 * h) - b * (d * k - f2 * g) + c * (d * h - e * g);
    if (!(fabs(det) > 1e-12)) return -1;
    Mi[0][0] = (e * k - f2 * h) / det; Mi[0][1] = (c * h - b * k) / det; Mi[0][2] = (b * f2 - c * e) / det;
    Mi[1][0] = (f2 * g - d * k) / det; Mi[1][1] = (a * k - c * g) / det; Mi[1][2] = (c * d - a * f2) / det;
    Mi[2][0] = (d * h - e * g) / det; Mi[2][1] = (b * g - a * h) / det; Mi[2][2] = (a * e - b * d) / det;
    return 0;
}

/* ufsm fiber-export <afv> <fields-dir> <out.fib> --tr T.json [--level 2] [--margin 8] [--shift dz,dy,dx] [--min-pts 4]
   The AFV fibres over a fit-fields box (fields.json origin / shape / h, native = --level voxels of the label scan) in the fields'
   box-local native coordinates, clipped to the box plus --margin native voxels (a fibre leaving and re-entering becomes two runs), for
   fit --fibers. --shift adds a translation (native voxels, e.g. a local registration correction). */
int cmd_fiber_export(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm fiber-export <afv> <fields-dir> <out.fib> --tr T.json [--level 2] [--margin 8] [--shift dz,dy,dx] [--min-pts 4]\n"); return 2; }
    const char *trp = fc_opt(argc, argv, "--tr", nullptr); double M[3][4], Mi[3][3];
    if (!trp || fc_transform(trp, M, Mi)) { fprintf(stderr, "fiber-export: --tr needs a transform.json with a transformation_matrix\n"); return 2; }
    const int level = atoi(fc_opt(argc, argv, "--level", "2")), minp = atoi(fc_opt(argc, argv, "--min-pts", "4")); const double mg = atof(fc_opt(argc, argv, "--margin", "8")), sc = ldexp(1.0, level);
    double sh[3] = {0, 0, 0}; if (*fc_opt(argc, argv, "--shift", "") && sscanf(fc_opt(argc, argv, "--shift", ""), "%lf,%lf,%lf", &sh[0], &sh[1], &sh[2]) != 3) { fprintf(stderr, "fiber-export: --shift dz,dy,dx\n"); return 2; }
    char fp[1200]; snprintf(fp, sizeof fp, "%s/fields.json", argv[3]);
    FILE *jf = fopen(fp, "rb"); if (!jf) { fprintf(stderr, "fiber-export: cannot read %s\n", fp); return 1; }
    fseek(jf, 0, SEEK_END); long jn = ftell(jf); fseek(jf, 0, SEEK_SET); char *jt = malloc(jn + 1); if (fread(jt, 1, jn, jf) != (size_t)jn) jn = 0; jt[jn] = 0; fclose(jf);
    json *J = json_parse(jt, jn); free(jt);
    double org[3], ext[3]; const double h = J ? json_num(json_get(J, "h"), 0) : 0;
    for (int d = 0; d < 3; d++) { org[d] = json_num(json_at(json_get(J, "origin"), d), NAN); ext[d] = json_num(json_at(json_get(J, "shape"), d), NAN) * h; }
    json_free(J);
    if (!(h > 0) || !(org[0] == org[0]) || !(ext[2] > 0)) { fprintf(stderr, "fiber-export: %s lacks origin / shape / h\n", fp); return 1; }
    double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
    for (int c = 0; c < 8; c++) {
        const double pz = sc * (org[0] - sh[0] + ((c & 1) ? ext[0] + mg : -mg) + 0.5) - 0.5, py = sc * (org[1] - sh[1] + ((c & 2) ? ext[1] + mg : -mg) + 0.5) - 0.5, px = sc * (org[2] - sh[2] + ((c & 4) ? ext[2] + mg : -mg) + 0.5) - 0.5;
        const double v[3] = {px, py, pz};
        for (int r = 0; r < 3; r++) { const double q = M[r][0] * v[0] + M[r][1] * v[1] + M[r][2] * v[2] + M[r][3]; if (q < lo[r]) lo[r] = q; if (q > hi[r]) hi[r] = q; }
    }
    afv *A = afv_open(argv[2]); if (!A) return 1;
    afv_run *R; const int nr = afv_query(A, lo, hi, &R); afv_close(A); if (nr < 0) return 1;
    fib_set F = {0}; size_t cap = 1 << 16, rcap = 1024; F.pts = malloc(cap * 3 * sizeof(float)); F.off = malloc((rcap + 1) * sizeof(int)); F.fam = malloc(rcap); F.off[0] = 0;
    long nH = 0, nV = 0;
    for (int i = 0; i < nr; i++) {
        int open = 0;   /* a run is open while its points stay inside */
        for (int k = 0; k < R[i].n; k++) {
            const double u[3] = {R[i].xyz[3 * k] - M[0][3], R[i].xyz[3 * k + 1] - M[1][3], R[i].xyz[3 * k + 2] - M[2][3]};
            double l0[3]; for (int a = 0; a < 3; a++) l0[a] = Mi[a][0] * u[0] + Mi[a][1] * u[1] + Mi[a][2] * u[2];
            const double q[3] = {(l0[2] + 0.5) / sc - 0.5 + sh[0] - org[0], (l0[1] + 0.5) / sc - 0.5 + sh[1] - org[1], (l0[0] + 0.5) / sc - 0.5 + sh[2] - org[2]};
            const int in = q[0] >= -mg && q[1] >= -mg && q[2] >= -mg && q[0] < ext[0] + mg && q[1] < ext[1] + mg && q[2] < ext[2] + mg;
            if (!in) {   /* close the run (dropped when too short) */
                if (open && F.npts - F.off[F.n] < minp) F.npts = F.off[F.n]; else if (open) { F.fam[F.n] = R[i].family; F.n++; F.off[F.n] = F.npts; }
                open = 0; continue;
            }
            if (!open) { if ((size_t)F.n + 1 >= rcap) { rcap *= 2; F.off = realloc(F.off, (rcap + 1) * sizeof(int)); F.fam = realloc(F.fam, rcap); } open = 1; }
            if ((size_t)F.npts == cap) { cap *= 2; F.pts = realloc(F.pts, cap * 3 * sizeof(float)); }
            for (int a = 0; a < 3; a++) F.pts[3 * (size_t)F.npts + a] = (float)q[a];
            F.npts++;
        }
        if (open && F.npts - F.off[F.n] < minp) F.npts = F.off[F.n]; else if (open) { F.fam[F.n] = R[i].family; F.n++; F.off[F.n] = F.npts; }
    }
    for (int r = 0; r < F.n; r++) { if (F.fam[r] == 'H') nH++; else if (F.fam[r] == 'V') nV++; }
    afv_runs_free(R, nr);
    if (fib_save(argv[4], &F)) { fprintf(stderr, "fiber-export: cannot write %s\n", argv[4]); return 1; }
    printf("fiber-export: %d runs (%ld H, %ld V), %d points over %s -> %s\n", F.n, nH, nV, F.npts, argv[3], argv[4]);
    fib_free(&F); return 0;
}
static double ct_score(const ctbox *C, const void *Sv, const size_t *idx, size_t m, const double t[3]) {   /* mean CT at the shifted points */
    const fsamp *S = Sv; double s = 0; size_t n = 0;
    for (size_t k = 0; k < m; k++) { const fsamp *p = &S[idx[k]]; const double v = ct_tri(C, p->p[0] + t[0], p->p[1] + t[1], p->p[2] + t[2]); if (v == v) { s += v; n++; } }
    return n ? s / n : -1;
}
