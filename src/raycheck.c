/* ufsm ray-check WINDINGS.json FIELDS [--zr Z0,Z1] [--step 0.25] [--minconf 0.05] [--tsv OUT]
   A sheet-count benchmark from human relative-winding rays (ScrollPrize spiral-input relative_windings.json: collections of points
   [x, y, z] in the fields' native voxels, each with wind_a = its wrap number; consecutive wraps along a radial line). For each pair of
   points of one collection, consecutive in wind_a, the turns a fit-field set puts between them, against dw = the wind_a difference:
   - phase: the network phase unwrapped along the segment (steps of --step field cells, nearest cell; cells without phase information
     are skipped, and the pair is left out when they are > 20% of the steps);
   - density: the integral of |n . d| / spacing along the segment (n n^T planes and the sheet spacing sp).
   Prints per method the signed error median / mean, the share |err| >= 0.5 turn, and both split by the annotated spacing per turn
   (packed: < 15 native voxels = 144 um at 9.6 um). Only pairs whose segment lies inside the fields box (and z in --zr) count. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fit.h"
#include "json.h"

static const char *rc_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
typedef struct { double a[3], b[3]; double dw, sp; } rc_pair;   /* a, b: z y x native voxels (global) */
static int rc_cmpz(const void *x, const void *y) { const rc_pair *p = x, *q = y; const double a = fmin(p->a[0], p->b[0]), b = fmin(q->a[0], q->b[0]); return (a > b) - (a < b); }
static int rc_cmpw(const void *x, const void *y) { const double a = ((const double *)x)[3], b = ((const double *)y)[3]; return (a > b) - (a < b); }
static int rc_cmpd(const void *x, const void *y) { const double a = *(const double *)x, b = *(const double *)y; return (a > b) - (a < b); }

int cmd_ray_check(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm ray-check WINDINGS.json FIELDS [--zr Z0,Z1] [--step 0.25] [--minconf 0.05] [--tsv OUT]\n"); return 2; }
    FILE *f = fopen(argv[2], "rb"); if (!f) { fprintf(stderr, "ray-check: cannot read %s\n", argv[2]); return 1; }
    fseek(f, 0, SEEK_END); const long L = ftell(f); fseek(f, 0, SEEK_SET); char *txt = malloc(L + 1);
    if (fread(txt, 1, L, f) != (size_t)L) { fclose(f); return 1; } fclose(f); txt[L] = 0;
    json *J = json_parse(txt, L); free(txt); const json *C = J ? json_get(J, "collections") : nullptr;
    if (!C || C->type != J_OBJ) { fprintf(stderr, "ray-check: no \"collections\" in %s\n", argv[2]); return 1; }
    double zr0 = -1e18, zr1 = 1e18; sscanf(rc_opt(argc, argv, "--zr", ""), "%lf,%lf", &zr0, &zr1);
    const double step = atof(rc_opt(argc, argv, "--step", "0.25")), minc = atof(rc_opt(argc, argv, "--minconf", "0.05"));
    size_t np = 0, cap = 1024; rc_pair *P = malloc(cap * sizeof *P);
    for (size_t c = 0; c < C->n; c++) {
        const json *pts = json_get(C->items[c], "points"); if (!pts || pts->type != J_OBJ || pts->n < 2) continue;
        double (*v)[4] = malloc(pts->n * sizeof *v); size_t n = 0;
        for (size_t k = 0; k < pts->n; k++) {
            const json *p = json_get(pts->items[k], "p"), *w = json_get(pts->items[k], "wind_a");
            if (!p || !w || w->type != J_NUM) continue;
            v[n][0] = json_num(json_at(p, 2), 0); v[n][1] = json_num(json_at(p, 1), 0); v[n][2] = json_num(json_at(p, 0), 0); v[n][3] = json_num(w, 0); n++;
        }
        qsort(v, n, sizeof *v, rc_cmpw);
        for (size_t k = 0; k + 1 < n; k++) {
            const double dw = v[k + 1][3] - v[k][3]; if (dw <= 0) continue;
            if (np == cap) { cap *= 2; P = realloc(P, cap * sizeof *P); }
            rc_pair *q = &P[np++]; for (int d = 0; d < 3; d++) { q->a[d] = v[k][d]; q->b[d] = v[k + 1][d]; }
            q->dw = dw; q->sp = sqrt(pow(q->a[0] - q->b[0], 2) + pow(q->a[1] - q->b[1], 2) + pow(q->a[2] - q->b[2], 2)) / dw;
        }
        free(v);
    }
    json_free(J);
    qsort(P, np, sizeof *P, rc_cmpz);
    fit_fields_buf g; if (fit_fields_load(argv[3], 0, 1, 1, &g)) { fprintf(stderr, "ray-check: cannot load %s\n", argv[3]); return 1; }
    const long long o[3] = {g.origin[0], g.origin[1], g.origin[2]}; const int H = g.shape[1], W = g.shape[2]; const double h = g.h; fit_fields_free(&g);
    int D = 0; {   /* the box's rows: fields.json "shape" (a row-range load reports only its rows) */
        char fp[1400]; snprintf(fp, sizeof fp, "%s/fields.json", argv[3]); FILE *jf = fopen(fp, "rb"); if (!jf) return 1;
        fseek(jf, 0, SEEK_END); const long jl = ftell(jf); fseek(jf, 0, SEEK_SET); char *jt = malloc(jl + 1);
        if (fread(jt, 1, jl, jf) != (size_t)jl) { fclose(jf); return 1; } fclose(jf); jt[jl] = 0;
        json *fj = json_parse(jt, jl); free(jt); D = fj ? (int)json_num(json_at(json_get(fj, "shape"), 0), 0) : 0; json_free(fj);
        if (D < 1) { fprintf(stderr, "ray-check: no shape in %s\n", fp); return 1; }
    }
    double *ep = malloc(np * sizeof *ep), *ed = malloc(np * sizeof *ed); uint8_t *pk = malloc(np), *okp = calloc(np, 1), *okd = calloc(np, 1);
    FILE *tsv = *rc_opt(argc, argv, "--tsv", "") ? fopen(rc_opt(argc, argv, "--tsv", ""), "w") : nullptr;
    if (tsv) fprintf(tsv, "z\ty\tx\tdw\tsp_vx\tphase_turns\tdens_turns\n");
    size_t used = 0;
    for (size_t i0 = 0; i0 < np;) {   /* chunks of field rows: the pairs whose rows fall in [r0, r0 + 48) */
        const int r0 = (int)floor((fmin(P[i0].a[0], P[i0].b[0]) - o[0]) / h) - 1, r1 = r0 + 48;
        size_t i1 = i0; while (i1 < np && (int)floor((fmax(P[i1].a[0], P[i1].b[0]) - o[0]) / h) + 1 < r1) i1++;
        if (i1 == i0) i1 = i0 + 1;
        const int z0 = r0 < 0 ? 0 : r0, z1 = r1 > D ? D : r1;
        fit_fields_buf b; const int have = z1 > z0 && !fit_fields_load(argv[3], z0, z1, 1, &b);
        for (size_t i = i0; i < i1; i++) {
            const rc_pair *q = &P[i]; pk[i] = q->sp < 15; ep[i] = ed[i] = 0;
            const double zm = 0.5 * (q->a[0] + q->b[0]); if (!have || zm < zr0 || zm >= zr1) continue;
            double c[2][3]; int in = 1;
            for (int e = 0; e < 2; e++) for (int d = 0; d < 3; d++) { c[e][d] = ((e ? q->b[d] : q->a[d]) - o[d]) / h - 0.5; }
            for (int e = 0; e < 2; e++) { in &= c[e][0] >= z0 && c[e][0] < z1 - 1 && c[e][1] >= 0 && c[e][1] < H - 1 && c[e][2] >= 0 && c[e][2] < W - 1; }
            if (!in) continue;
            const double len = sqrt(pow(c[1][0] - c[0][0], 2) + pow(c[1][1] - c[0][1], 2) + pow(c[1][2] - c[0][2], 2)); if (len < 1e-6) continue;
            const double dd[3] = {(c[1][0] - c[0][0]) / len, (c[1][1] - c[0][1]) / len, (c[1][2] - c[0][2]) / len};
            const int ns = (int)ceil(len / step); double turns = 0, dens = 0, prev = NAN; int skip = 0, dskip = 0;
            for (int s = 0; s <= ns; s++) {
                const double t = (double)s / ns; int ci[3];
                for (int d = 0; d < 3; d++) ci[d] = (int)lround(c[0][d] + t * (c[1][d] - c[0][d]));
                const size_t k = ((size_t)(ci[0] - z0) * H + ci[1]) * W + ci[2];
                const double mc = (b.f.pc[k] - 128) / 127.0, ms = (b.f.ps[k] - 128) / 127.0;
                if (sqrt(mc * mc + ms * ms) < minc || !b.f.w[k]) { skip++; prev = NAN; }
                else { const double ph = atan2(ms, mc) / (2 * M_PI); if (!isnan(prev)) { double dp = ph - prev; dp -= floor(dp + 0.5); turns += dp; } prev = ph; }
                if (s < ns) {
                    const double T[6] = {b.f.t[0][k] / 255.0, b.f.t[1][k] / 255.0, b.f.t[2][k] / 255.0, b.f.t[3][k] / 255.0 - 0.5, b.f.t[4][k] / 255.0 - 0.5, b.f.t[5][k] / 255.0 - 0.5};
                    const double qd = T[0] * dd[0] * dd[0] + T[1] * dd[1] * dd[1] + T[2] * dd[2] * dd[2] + 2 * (T[3] * dd[0] * dd[1] + T[4] * dd[0] * dd[2] + T[5] * dd[1] * dd[2]);
                    if (b.f.sp[k]) dens += sqrt(fmax(0, qd)) * (len / ns) / (b.f.sp[k] / 8.0); else dskip++;
                }
            }
            if (skip <= 0.2 * (ns + 1)) { ep[i] = fabs(turns) - q->dw; okp[i] = 1; }
            if (dskip <= 0.2 * ns) { ed[i] = dens - q->dw; okd[i] = 1; }
            used++;
            if (tsv) fprintf(tsv, "%.0f\t%.0f\t%.0f\t%.0f\t%.1f\t%.3f\t%.3f\n", zm, 0.5 * (q->a[1] + q->b[1]), 0.5 * (q->a[2] + q->b[2]), q->dw, q->sp, okp[i] ? fabs(turns) : NAN, okd[i] ? dens : NAN);
        }
        if (have) fit_fields_free(&b);
        i0 = i1;
    }
    if (tsv) fclose(tsv);
    printf("ray-check: %zu annotated pairs, %zu inside the fields box%s\n", np, used, *rc_opt(argc, argv, "--zr", "") ? " and --zr" : "");
    for (int m = 0; m < 2; m++) for (int sub = 0; sub < 3; sub++) {
        const double *e = m ? ed : ep; const uint8_t *ok = m ? okd : okp; double *v = malloc((np ? np : 1) * sizeof *v); size_t n = 0, off = 0; double s = 0;
        for (size_t i = 0; i < np; i++) if (ok[i] && (sub == 0 || (sub == 1) == (pk[i] != 0))) { v[n++] = e[i]; s += e[i]; off += fabs(e[i]) >= 0.5; }
        qsort(v, n, sizeof *v, rc_cmpd);
        printf("ray-check: %-7s %-6s pairs %5zu  error median %+.3f mean %+.3f turns, |err| >= 0.5 turn %.3f\n", m ? "density" : "phase", sub == 0 ? "all" : sub == 1 ? "packed" : "sparse",
               n, n ? v[n / 2] : 0.0, n ? s / n : 0.0, n ? (double)off / n : 0.0);
        free(v);
    }
    free(P); free(ep); free(ed); free(pk); free(okp); free(okd);
    return 0;
}
