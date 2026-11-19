/* ufsm fiber-contacts: the wgraph contact-crossing idea (villa vesuvius/docs/wgraph.md, fiber_workflows.md: fibre-fibre encounters with
   a unit connector A -> B for ORDER, H / V layer classes and an oriented normal) measured against the human GT, before any solver
   (DESIGN 2026-11-19 "wgraph contacts"). Per GT box (boxes.tsv rows: name scroll z y x nz ny nx, voxels of --level of the label scan):
   the AFV fibres over the box, mapped into the label scan (--map reg.json: ufsm reg, AFV scan = coarse; or --tr transform.json), resampled
   every --step level voxels, each sample labelled from the GT codes (winding_mod14 x 18 + 1 on the labelled faces: the nearest code voxel
   within --rad; W = (code - 1) / 18 turns) with the GT sheet normal there (the minor axis of the covariance of the same-sheet code voxels
   within 5 voxels, signed radially outward from the umbilicus --axis).
   (1) layer class: per fibre the median signed normal offset of its samples from their labelled face (t, level voxels; + outward), by the
       AFV family H / V: does the family say which side of the sheet a fibre is on?
   (2) contacts: every --every-th labelled H sample a and its nearest labelled V sample b within --contact level voxels (one per a);
       GT: dW = round(W_b - W_a) (mod 14, centred); order: the connector c = (b - a) / |b - a| against the radial outward direction
       (s_rad, no GT) and against the GT normal (s_gt, oracle); the GT gap on b's side (from a's face along the GT normal toward b to the
       first code voxel of another sheet, um); the merge flag of a fields set at the midpoint (--mf ROOT KEY --mf-origin z,y,x --mf-h H:
       the mf plane's native grid = --level voxels, cells of H). Tables by gap bucket; every contact to --out (tsv). */
#include "afv.h"
#include "json.h"
#include "optcheck.h"
#include "reg.h"
#include "sources.h"
#include "store.h"
#include "zarr3.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *fx_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(k))) return argv[i + 1]; return d; }
static int fx_opt2(int argc, char **argv, const char *k) { for (int i = 1; i + 2 < argc; i++) if (!strcmp(argv[i], opt_n(k))) return i; return 0; }
static double wrap14(double v) { v = fmod(v + 7, 14); if (v < 0) v += 14; return v - 7; }

typedef struct { float p[3]; int run, fam, in, lab; float W, n[3], t; float q[3]; float u; } xs;   /* u: offset from the sheet's mid-surface (CT run along n through q; NaN unknown) */   /* q: the nearest code voxel; t: offset along n */
typedef struct { int64_t o[3], n[3]; uint8_t *c; } xbox;
static inline int xc(const xbox *B, int64_t z, int64_t y, int64_t x) {
    z -= B->o[0]; y -= B->o[1]; x -= B->o[2];
    if (z < 0 || y < 0 || x < 0 || z >= B->n[0] || y >= B->n[1] || x >= B->n[2]) return 0;
    const int v = B->c[((size_t)z * B->n[1] + y) * B->n[2] + x]; return v >= 1 && v <= 252 ? v : 0;
}
/* the minor eigenvector of a symmetric 3x3 (Jacobi) */
static void minor_ev(double A[3][3], double v[3]) {
    double V[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    for (int sw = 0; sw < 30; sw++) {
        double off = fabs(A[0][1]) + fabs(A[0][2]) + fabs(A[1][2]); if (off < 1e-12) break;
        for (int p = 0; p < 2; p++) for (int q = p + 1; q < 3; q++) {
            if (fabs(A[p][q]) < 1e-15) continue;
            const double th = 0.5 * atan2(2 * A[p][q], A[q][q] - A[p][p]), c = cos(th), s = sin(th);
            for (int k = 0; k < 3; k++) { const double a = A[k][p], b = A[k][q]; A[k][p] = c * a - s * b; A[k][q] = s * a + c * b; }
            for (int k = 0; k < 3; k++) { const double a = A[p][k], b = A[q][k]; A[p][k] = c * a - s * b; A[q][k] = s * a + c * b; }
            for (int k = 0; k < 3; k++) { const double a = V[k][p], b = V[k][q]; V[k][p] = c * a - s * b; V[k][q] = s * a + c * b; }
        }
    }
    int m = 0; for (int k = 1; k < 3; k++) if (A[k][k] < A[m][m]) m = k;
    for (int k = 0; k < 3; k++) v[k] = V[k][m];
}
/* the GT at a sample: nearest code voxel within rad, W, the sheet normal there (signed by r_out: radial outward, z y x). 0 ok */
static int xlabel(const xbox *B, xs *s, int rad, const double rout[3]) {
    const int64_t c[3] = {lround(s->p[0]), lround(s->p[1]), lround(s->p[2])}; double bd = 1e30; int code = 0; int64_t q[3] = {0};
    for (int dz = -rad; dz <= rad; dz++) for (int dy = -rad; dy <= rad; dy++) for (int dx = -rad; dx <= rad; dx++) {
        const int k = xc(B, c[0] + dz, c[1] + dy, c[2] + dx); if (!k) continue;
        const double dd = (c[0] + dz - s->p[0]) * (c[0] + dz - s->p[0]) + (c[1] + dy - s->p[1]) * (c[1] + dy - s->p[1]) + (c[2] + dx - s->p[2]) * (c[2] + dx - s->p[2]);
        if (dd < bd) { bd = dd; code = k; q[0] = c[0] + dz; q[1] = c[1] + dy; q[2] = c[2] + dx; }
    }
    if (!code) return -1;
    const double W = (code - 1) / 18.0; double m[3] = {0}, C[3][3] = {{0}}; int n = 0; const int R = 5;
    for (int dz = -R; dz <= R; dz++) for (int dy = -R; dy <= R; dy++) for (int dx = -R; dx <= R; dx++) {
        if (dz * dz + dy * dy + dx * dx > R * R) continue;
        const int k = xc(B, q[0] + dz, q[1] + dy, q[2] + dx); if (!k || fabs(wrap14((k - 1) / 18.0 - W)) >= 0.5) continue;
        const double d[3] = {dz, dy, dx}; n++; for (int a = 0; a < 3; a++) { m[a] += d[a]; for (int b = 0; b < 3; b++) C[a][b] += d[a] * d[b]; }
    }
    if (n < 12) return -1;
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) C[a][b] = C[a][b] / n - m[a] / n * m[b] / n;
    double v[3]; minor_ev(C, v); const double sg = v[0] * rout[0] + v[1] * rout[1] + v[2] * rout[2] < 0 ? -1 : 1;
    for (int a = 0; a < 3; a++) { s->n[a] = (float)(sg * v[a]); s->q[a] = (float)q[a]; }
    s->W = (float)W; s->t = (float)((s->p[0] - q[0]) * s->n[0] + (s->p[1] - q[1]) * s->n[1] + (s->p[2] - q[2]) * s->n[2]); s->lab = 1;
    return 0;
}
/* from a's face along +-n (dir) to the first code voxel of another sheet within 1 voxel of the line: distance (level voxels), 0 none */
static double xgap(const xbox *B, const xs *a, double dir, int maxd) {
    for (double t = 1; t <= maxd; t += 0.5) {
        const double p[3] = {a->q[0] + dir * t * a->n[0], a->q[1] + dir * t * a->n[1], a->q[2] + dir * t * a->n[2]};
        for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
            const int k = xc(B, lround(p[0]) + dz, lround(p[1]) + dy, lround(p[2]) + dx);
            if (k && fabs(wrap14((k - 1) / 18.0 - a->W)) >= 0.5) return t;
        }
    }
    return 0;
}
/* --ct: the sheet's papyrus run along n through its code voxel q (CT >= iso, trilinear, 0.5-voxel steps, <= 15 voxels each way): the
   sample's offset from the run's middle (level voxels, + outward), NaN when the run is not bounded */
typedef struct { int64_t o[3], n[3]; uint8_t *v; } xct;
static double xtri(const xct *C, double z, double y, double x) {
    z -= C->o[0]; y -= C->o[1]; x -= C->o[2];
    const int64_t z0 = (int64_t)floor(z), y0 = (int64_t)floor(y), x0 = (int64_t)floor(x);
    if (z0 < 0 || y0 < 0 || x0 < 0 || z0 + 1 >= C->n[0] || y0 + 1 >= C->n[1] || x0 + 1 >= C->n[2]) return NAN;
    const double fz = z - z0, fy = y - y0, fx = x - x0; double r = 0;
    for (int c = 0; c < 8; c++) { const int a = c >> 2, b = (c >> 1) & 1, d = c & 1;
        r += (a ? fz : 1 - fz) * (b ? fy : 1 - fy) * (d ? fx : 1 - fx) * C->v[((size_t)(z0 + a) * C->n[1] + (y0 + b)) * C->n[2] + x0 + d]; }
    return r;
}
static float xmid(const xct *C, const xs *s, double iso) {
    double e[2] = {0, 0};
    for (int k = 0; k < 2; k++) { const double dir = k ? 1 : -1; double t = 0; int ok = 0;
        for (double u = 0.5; u <= 15; u += 0.5) { const double v = xtri(C, s->q[0] + dir * u * s->n[0], s->q[1] + dir * u * s->n[1], s->q[2] + dir * u * s->n[2]);
            if (!(v == v)) break; if (v < iso) { ok = 1; break; } t = u; }
        if (!ok) return NAN; e[k] = dir * t; }
    const double mid = 0.5 * (e[0] + e[1]); return (float)(s->t - mid);
}
static int cmp_f(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return x < y ? -1 : x > y; }

enum { NG = 5 };   /* gap buckets: < 50, 50-70, 70-100, 100-150, >= 150 um (and none) */
static const double GB[NG] = {50, 70, 100, 150, 1e30};
int cmd_fiber_contacts(int argc, char **argv) {
    const int ci = fx_opt2(argc, argv, "--codes");
    if (argc < 3 || !ci) {
        fprintf(stderr, "usage: ufsm fiber-contacts <afv> (--map reg.json | --tr T.json) --boxes B.tsv --names A,B --codes ROOT KEY --axis UMB.json\n"
                        "         [--level 2] [--um 9.6] [--step 1] [--every 2] [--rad 3] [--contact 6] [--gap-max 40] [--mf ROOT KEY --mf-origin z,y,x --mf-h 2]\n"
                        "         [--out contacts.tsv] [--fibres fibres.tsv]\n"); return 2; }
    const char *afp = argv[2], *mapp = fx_opt(argc, argv, "--map", nullptr), *trp = fx_opt(argc, argv, "--tr", nullptr), *btp = fx_opt(argc, argv, "--boxes", nullptr);
    const char *names = fx_opt(argc, argv, "--names", nullptr), *axp = fx_opt(argc, argv, "--axis", nullptr);
    const int level = opt_atoi("--level", fx_opt(argc, argv, "--level", "2")), every = opt_atoi("--every", fx_opt(argc, argv, "--every", "2")), rad = opt_atoi("--rad", fx_opt(argc, argv, "--rad", "3"));
    const int gmax = opt_atoi("--gap-max", fx_opt(argc, argv, "--gap-max", "40"));
    const double step = opt_atof("--step", fx_opt(argc, argv, "--step", "1")), cd = opt_atof("--contact", fx_opt(argc, argv, "--contact", "6")), um = opt_atof("--um", fx_opt(argc, argv, "--um", "9.6"));
    if (!btp || !names || !axp || (!mapp && !trp)) { fprintf(stderr, "fiber-contacts: --boxes, --names, --axis and --map or --tr needed\n"); return 2; }
    const rmap *RM = nullptr; double M[3][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}}, Mi[3][3];
    if (mapp) { if (!(RM = reg_map_load(mapp))) { fprintf(stderr, "fiber-contacts: cannot load %s\n", mapp); return 1; } }
    else if (fc_transform(trp, M, Mi)) { fprintf(stderr, "fiber-contacts: cannot read %s\n", trp); return 1; }
    axis ax = {0}; if (axis_load(&ax, axp)) { fprintf(stderr, "fiber-contacts: cannot load the axis %s\n", axp); return 1; }
    store *cs = store_open(argv[ci + 1]); z3 *cz = cs ? z3_open(cs, argv[ci + 2], nullptr) : nullptr; if (!cz) { fprintf(stderr, "fiber-contacts: cannot open the codes\n"); return 1; }
    z3 *mz = nullptr; double mo[3] = {0}; int mh = 2;
    { const int mi = fx_opt2(argc, argv, "--mf"); if (mi) { store *ms = store_open(argv[mi + 1]); mz = ms ? z3_open(ms, argv[mi + 2], nullptr) : nullptr; if (!mz) { fprintf(stderr, "fiber-contacts: cannot open the mf plane\n"); return 1; }
        if (sscanf(fx_opt(argc, argv, "--mf-origin", "0,0,0"), "%lf,%lf,%lf", &mo[0], &mo[1], &mo[2]) != 3) return 2; mh = opt_atoi("--mf-h", fx_opt(argc, argv, "--mf-h", "2")); } }
    FILE *fo = nullptr, *ff = nullptr;
    { const char *p = fx_opt(argc, argv, "--out", nullptr); if (p && !(fo = fopen(p, "w"))) return 1; if (fo) fprintf(fo, "box\trun_h\trun_v\td\tdW\ts_rad\ts_gt\tc_rad\tc_gt\tgap_um\tmf\tu_h\tu_v\tU_h\tU_v\n");
      p = fx_opt(argc, argv, "--fibres", nullptr); if (p && !(ff = fopen(p, "w"))) return 1; if (ff) fprintf(ff, "box\trun\tfam\tn_lab\tt_median\tt_q25\tt_q75\n"); }
    z3 *ctz = nullptr; const double isoarg = opt_atof("--iso", fx_opt(argc, argv, "--iso", "0"));
    { const int ti = fx_opt2(argc, argv, "--ct"); if (ti) { store *ts = store_open(argv[ti + 1]); ctz = ts ? z3_open(ts, argv[ti + 2], nullptr) : nullptr; if (!ctz) { fprintf(stderr, "fiber-contacts: cannot open the CT\n"); return 1; } } }
    const double sc = ldexp(1.0, level);
    afv *A = afv_open(afp); if (!A) return 1;
    /* totals: contacts by gap bucket x (s_rad > 0) x dW in {-1, 0, +1, other}; also by s_gt */
    static long T[2][NG + 1][2][4]; long mfn[NG + 1][2] = {{0}}; long fam_side[2][3] = {{0}}; float *tl[2] = {nullptr, nullptr}; size_t ntl[2] = {0}, ctl[2] = {0};
    char *nl = strdup(names);
    for (char *bn = strtok(nl, ","); bn; bn = strtok(nullptr, ",")) {
        FILE *bf = fopen(btp, "r"); if (!bf) return 1; char line[1024]; int64_t bo[3] = {0}, bs[3] = {0}; int found = 0;
        while (fgets(line, sizeof line, bf)) { char nm[128], scl[128]; long long z, y, x, nz, ny, nx;
            if (line[0] != '#' && sscanf(line, "%127s %127s %lld %lld %lld %lld %lld %lld", nm, scl, &z, &y, &x, &nz, &ny, &nx) == 8 && !strcmp(nm, bn)) { bo[0] = z; bo[1] = y; bo[2] = x; bs[0] = nz; bs[1] = ny; bs[2] = nx; found = 1; break; } }
        fclose(bf); if (!found) { fprintf(stderr, "fiber-contacts: no box %s\n", bn); continue; }
        double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
        for (int c = 0; c < 8; c++) {
            const double v[3] = {sc * (bo[2] + ((c & 4) ? bs[2] : 0) + 0.5) - 0.5, sc * (bo[1] + ((c & 2) ? bs[1] : 0) + 0.5) - 0.5, sc * (bo[0] + ((c & 1) ? bs[0] : 0) + 0.5) - 0.5};
            double q[3]; if (RM) rmap_f2c(RM, v, q); else for (int r = 0; r < 3; r++) q[r] = M[r][0] * v[0] + M[r][1] * v[1] + M[r][2] * v[2] + M[r][3];
            for (int r = 0; r < 3; r++) { if (q[r] < lo[r]) lo[r] = q[r]; if (q[r] > hi[r]) hi[r] = q[r]; }
        }
        for (int r = 0; r < 3; r++) { lo[r] -= 16; hi[r] += 16; }
        afv_run *R; const int nr = afv_query(A, lo, hi, &R); if (nr < 0) return 1;
        size_t ns = 0, cap = 1 << 16; xs *S = malloc(cap * sizeof *S);
        for (int i = 0; i < nr; i++) {
            const afv_run *r = &R[i]; if (r->n < 2 || (r->family != 'H' && r->family != 'V')) continue;
            double prev[3] = {0}, acc = 0, next = 0;
            for (int k = 0; k < r->n; k++) {
                double l0[3];
                if (RM) rmap_c2f(RM, r->xyz + 3 * (size_t)k, l0);
                else { const double u[3] = {r->xyz[3 * k] - M[0][3], r->xyz[3 * k + 1] - M[1][3], r->xyz[3 * k + 2] - M[2][3]}; for (int a = 0; a < 3; a++) l0[a] = Mi[a][0] * u[0] + Mi[a][1] * u[1] + Mi[a][2] * u[2]; }
                const double q[3] = {(l0[2] + 0.5) / sc - 0.5, (l0[1] + 0.5) / sc - 0.5, (l0[0] + 0.5) / sc - 0.5};
                if (k == 0) { memcpy(prev, q, sizeof prev); continue; }
                const double dv[3] = {q[0] - prev[0], q[1] - prev[1], q[2] - prev[2]}, L = sqrt(dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]);
                while (L > 0 && next <= acc + L) {
                    const double f = (next - acc) / L; if (ns == cap) { cap *= 2; S = realloc(S, cap * sizeof *S); }
                    xs *s = &S[ns++]; memset(s, 0, sizeof *s); for (int a = 0; a < 3; a++) s->p[a] = (float)(prev[a] + f * dv[a]);
                    s->run = i; s->fam = r->family == 'H' ? 0 : 1;
                    s->in = s->p[0] >= bo[0] && s->p[1] >= bo[1] && s->p[2] >= bo[2] && s->p[0] < bo[0] + bs[0] && s->p[1] < bo[1] + bs[1] && s->p[2] < bo[2] + bs[2];
                    next += step;
                }
                acc += L; memcpy(prev, q, sizeof prev);
            }
        }
        afv_runs_free(R, nr);
        xbox G; const int hal = rad + gmax + 6;
        for (int d = 0; d < 3; d++) { G.o[d] = bo[d] - hal; G.n[d] = bs[d] + 2 * hal; if (G.o[d] < 0) { G.n[d] += G.o[d]; G.o[d] = 0; } }
        G.c = malloc((size_t)G.n[0] * G.n[1] * G.n[2]); if (!G.c || z3_read(cz, G.o, G.n, G.c, 8)) { fprintf(stderr, "fiber-contacts: %s: codes read failed\n", bn); return 1; }
        xct Cct = {0}; double iso = 0;
        if (ctz) { for (int d = 0; d < 3; d++) { Cct.o[d] = G.o[d]; Cct.n[d] = G.n[d]; } Cct.v = malloc((size_t)Cct.n[0] * Cct.n[1] * Cct.n[2]);
            if (!Cct.v || z3_read(ctz, Cct.o, Cct.n, Cct.v, 8)) { fprintf(stderr, "fiber-contacts: %s: CT read failed\n", bn); return 1; }
            /* iso: half way between the box's 10th CT percentile (air) and the median CT on the labelled voxels (papyrus), unless --iso */
            if (isoarg > 0) iso = isoarg; else { long h0[256] = {0}, h1[256] = {0}; long n0 = 0, n1 = 0; const size_t NV = (size_t)Cct.n[0] * Cct.n[1] * Cct.n[2];
                for (size_t i = 0; i < NV; i += 7) { h0[Cct.v[i]]++; n0++; if (G.c[i] >= 1 && G.c[i] <= 252) { h1[Cct.v[i]]++; n1++; } }
                int p10 = 0; for (long a = 0; p10 < 255 && (a += h0[p10]) < n0 / 10; p10++) {}
                int p50 = 0; for (long a = 0; p50 < 255 && (a += h1[p50]) < n1 / 2; p50++) {}
                iso = 0.5 * (p10 + p50); printf("box %s: CT iso %.1f (air p10 %d, labelled median %d)\n", bn, iso, p10, p50); } }
        size_t nlab = 0;
        for (size_t k = 0; k < ns; k++) {
            xs *s = &S[k]; if (!s->in) continue;
            double ay, axx; axis_at(&ax, sc * (s->p[0] + 0.5) - 0.5, &ay, &axx);
            double r[3] = {0, s->p[1] - ((ay + 0.5) / sc - 0.5), s->p[2] - ((axx + 0.5) / sc - 0.5)}; const double rl = hypot(r[1], r[2]); if (rl < 1) continue; r[1] /= rl; r[2] /= rl;
            s->u = NAN; if (!xlabel(&G, s, rad, r)) { nlab++; if (Cct.v) s->u = xmid(&Cct, s, iso); }
        }
        /* the fibre's face: the median of its samples' mid-surface offsets (a fibre stays in one layer) */
        int maxrun = 0; for (size_t k = 0; k < ns; k++) if (S[k].run > maxrun) maxrun = S[k].run;
        float *fu = malloc(((size_t)maxrun + 1) * sizeof *fu); for (int i = 0; i <= maxrun; i++) fu[i] = NAN;
        for (size_t a = 0; a < ns;) { size_t b = a; while (b < ns && S[b].run == S[a].run) b++;
            float *tv = malloc((b - a) * sizeof *tv); size_t m = 0; for (size_t k = a; k < b; k++) if (S[k].lab && S[k].u == S[k].u) tv[m++] = S[k].u;
            if (m >= 4) { qsort(tv, m, sizeof *tv, cmp_f); fu[S[a].run] = tv[m / 2]; } free(tv); a = b; }
        /* (1) per fibre: the median offset of its labelled samples */
        long bside[2][3] = {{0}};
        for (size_t a = 0; a < ns;) {
            size_t b = a; while (b < ns && S[b].run == S[a].run) b++;
            float *tv = malloc((b - a) * sizeof *tv); size_t m = 0; for (size_t k = a; k < b; k++) if (S[k].lab) tv[m++] = S[k].t;
            if (m >= 8) { qsort(tv, m, sizeof *tv, cmp_f); const float med = tv[m / 2]; const int f = S[a].fam;
                if (ff) fprintf(ff, "%s\t%d\t%c\t%zu\t%.2f\t%.2f\t%.2f\n", bn, S[a].run, "HV"[f], m, med, tv[m / 4], tv[3 * m / 4]);
                bside[f][med < -0.5 ? 0 : med > 0.5 ? 2 : 1]++; fam_side[f][med < -0.5 ? 0 : med > 0.5 ? 2 : 1]++;
                if (ntl[f] == ctl[f]) { ctl[f] = ctl[f] ? 2 * ctl[f] : 4096; tl[f] = realloc(tl[f], ctl[f] * sizeof *tl[f]); } tl[f][ntl[f]++] = med; }
            free(tv); a = b;
        }
        /* (2) contacts: labelled V samples in a hash, each every-th labelled H sample's nearest V within cd */
        const double cs4 = cd > 1 ? cd : 1; int gd[3]; for (int d = 0; d < 3; d++) gd[d] = (int)(bs[d] / cs4) + 1;
        const size_t ngd = (size_t)gd[0] * gd[1] * gd[2]; int64_t *head = malloc(ngd * sizeof *head), *nxt = malloc(ns * sizeof *nxt);
        for (size_t i = 0; i < ngd; i++) head[i] = -1;
        #define XCELL(s, c) int c[3]; for (int d_ = 0; d_ < 3; d_++) { c[d_] = (int)floor(((s)->p[d_] - bo[d_]) / cs4); if (c[d_] < 0) c[d_] = 0; if (c[d_] >= gd[d_]) c[d_] = gd[d_] - 1; }
        for (size_t k = 0; k < ns; k++) { nxt[k] = -1; const xs *s = &S[k]; if (s->fam != 1 || !s->lab) continue; XCELL(s, c); const size_t q = ((size_t)c[0] * gd[1] + c[1]) * gd[2] + c[2]; nxt[k] = head[q]; head[q] = (int64_t)k; }
        long bc[NG + 1][2][4] = {{{0}}}; long nct = 0;
        for (size_t k = 0; k < ns; k += every) {
            const xs *h = &S[k]; if (h->fam != 0 || !h->lab) continue;
            XCELL(h, c); double bd = cd * cd; int64_t bv = -1;
            for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
                const int q[3] = {c[0] + dz, c[1] + dy, c[2] + dx}; if (q[0] < 0 || q[1] < 0 || q[2] < 0 || q[0] >= gd[0] || q[1] >= gd[1] || q[2] >= gd[2]) continue;
                for (int64_t v = head[((size_t)q[0] * gd[1] + q[1]) * gd[2] + q[2]]; v >= 0; v = nxt[v]) {
                    const xs *s = &S[v]; const double dd = (s->p[0] - h->p[0]) * (s->p[0] - h->p[0]) + (s->p[1] - h->p[1]) * (s->p[1] - h->p[1]) + (s->p[2] - h->p[2]) * (s->p[2] - h->p[2]);
                    if (dd < bd) { bd = dd; bv = v; }
                }
            }
            if (bv < 0) continue;
            const xs *v = &S[bv]; const double d = sqrt(bd); if (!(d > 0.25)) continue;   /* (coincident points: no connector) */
            const double cn[3] = {(v->p[0] - h->p[0]) / d, (v->p[1] - h->p[1]) / d, (v->p[2] - h->p[2]) / d};
            double ay, axx; axis_at(&ax, sc * (h->p[0] + 0.5) - 0.5, &ay, &axx);
            double r[3] = {0, h->p[1] - ((ay + 0.5) / sc - 0.5), h->p[2] - ((axx + 0.5) / sc - 0.5)}; const double rl = hypot(r[1], r[2]); r[1] /= rl; r[2] /= rl;
            const double crad = cn[1] * r[1] + cn[2] * r[2], cgt = cn[0] * h->n[0] + cn[1] * h->n[1] + cn[2] * h->n[2];
            const double dw = wrap14(v->W - h->W); const int dW = (int)lround(dw), wi = dW == -1 ? 0 : dW == 0 ? 1 : dW == 1 ? 2 : 3;
            const double gap = xgap(&G, h, cgt >= 0 ? 1 : -1, gmax) * um; int gi = NG; if (gap > 0) { gi = 0; while (gi < NG - 1 && gap >= GB[gi]) gi++; }
            int mf = -1;
            if (mz) { const int64_t o[3] = {(int64_t)floor(((h->p[0] + v->p[0]) / 2 - mo[0]) / mh), (int64_t)floor(((h->p[1] + v->p[1]) / 2 - mo[1]) / mh), (int64_t)floor(((h->p[2] + v->p[2]) / 2 - mo[2]) / mh)}, n1[3] = {1, 1, 1};
                      uint8_t b; if (!z3_read(mz, o, n1, &b, 1)) mf = b; }
            bc[gi][crad > 0][wi]++; T[0][gi][crad > 0][wi]++; T[1][gi][cgt > 0][wi]++; nct++;
            if (mf >= 0) { mfn[gi][0]++; mfn[gi][1] += mf >= 179; }
            if (fo) fprintf(fo, "%s\t%d\t%d\t%.2f\t%d\t%d\t%d\t%.3f\t%.3f\t%.1f\t%d\t%.2f\t%.2f\t%.2f\t%.2f\n", bn, h->run, v->run, d, dW, crad > 0 ? 1 : -1, cgt > 0 ? 1 : -1, crad, cgt, gap, mf,
                            h->u, v->u, fu[h->run], fu[v->run]);
        }
        #undef XCELL
        free(head); free(nxt); free(fu); free(Cct.v);
        printf("box %s: %zu samples, %zu labelled; fibres H on the outward / middle / inward side %ld / %ld / %ld, V %ld / %ld / %ld; %ld contacts\n", bn, ns, nlab,
               bside[0][2], bside[0][1], bside[0][0], bside[1][2], bside[1][1], bside[1][0], nct);
        fflush(stdout); free(S); free(G.c);
    }
    free(nl);
    printf("\n(1) layer class: fibre median offset from its labelled face along the outward GT normal (level voxels): side - / mid / +\n");
    for (int f = 0; f < 2; f++) { const long n = fam_side[f][0] + fam_side[f][1] + fam_side[f][2]; if (!n) continue; qsort(tl[f], ntl[f], sizeof *tl[f], cmp_f);
        printf("%c: %ld fibres: inward (< -0.5) %.3f, mid %.3f, outward (> 0.5) %.3f; median %.2f, q25 %.2f, q75 %.2f\n", "HV"[f], n, (double)fam_side[f][0] / n, (double)fam_side[f][1] / n, (double)fam_side[f][2] / n,
               tl[f][ntl[f] / 2], tl[f][ntl[f] / 4], tl[f][3 * ntl[f] / 4]); free(tl[f]); }
    static const char *GN[NG + 1] = {"<50um", "50-70", "70-100", "100-150", ">=150", "nogap"};
    for (int o = 0; o < 2; o++) {
        printf("\n(2) H -> V contacts by the GT gap on V's side, order by %s: n(V outward / inward) x GT dW (-1 0 +1 other); mf-flagged share\n", o ? "the GT normal (oracle)" : "the radial direction (no GT)");
        printf("gap\tn\tout:-1\tout:0\tout:+1\tout:oth\tin:-1\tin:0\tin:+1\tin:oth\tmf179\n");
        for (int g = 0; g <= NG; g++) { long n = 0; for (int s = 0; s < 2; s++) for (int w = 0; w < 4; w++) n += T[o][g][s][w]; if (!n) continue;
            printf("%s\t%ld", GN[g], n); for (int s = 1; s >= 0; s--) for (int w = 0; w < 4; w++) printf("\t%ld", T[o][g][s][w]);
            if (mfn[g][0]) printf("\t%.3f\n", (double)mfn[g][1] / mfn[g][0]); else printf("\t-\n"); }
    }
    if (fo) fclose(fo);
    if (ff) fclose(ff);
    afv_close(A); z3_close(cz); if (mz) z3_close(mz);
    return 0;
}
