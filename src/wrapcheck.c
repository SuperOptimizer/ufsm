/* ufsm wrap-check <model-or-set> <fields-dir> [--ct ROOT KEY --ct-um U] [--codes ROOT KEY] [--nz 16] [--nth 180] [--zr Z0,Z1]
                   [--sigma 1.2] [--prom 1] [--w-scale 1] [--dup 0.5] [--threads 16] [--cache DIR] [--tsv OUT]
   Duplicate / over-count check of a fitted winding (a joined fit set or one model; its wraps are the level sets W = k, ufsm wraps):
   along radial rays from the umbilicus (--nth angles at --nz z levels evenly over the box, or over native box-local z [Z0, Z1)), every
   pair of consecutive wrap crossings k -> k + 1 (inside material: the fields' weight w > 0 on every sample between them) is an
   interval, measured
     - gap: its normal distance (radial gap x |grad W . r| / |grad W|, um) and that over the network's sheet spacing sp (rnet),
     - with --ct: the papyrus peaks of the CT profile along the ray inside it (npk: the profile smoothed by a gaussian of --sigma native
       voxels, local maxima whose prominence within +-24 native voxels is >= --prom x the local sd over +-12; 1 = one sheet per turn as
       it should be, 0 = two wraps on one sheet (the fit over-counts there: duplicate), 2+ = a sheet skipped (merged)) and the gap over
       the local CT peak spacing (rct: the median spacing of the peaks within +-40 native voxels),
     - with --codes (winding_mod14 band codes, e.g. q-9p6.zarr 9.6; KEY = the codes' um): the ground-truth turns between the two
       crossings, round(dq mod 14) from the labelled samples nearest each crossing (within 2 samples).
   CT at --ct-um U (its level's voxel size; the native grid is the fields'), sampled bilinearly in the CT plane(s) covering each native z.
   Prints: counts per radius band (CT peaks per fitted turn, and the GT turns per fitted turn with --codes), the npk x GT confusion
   (validation), and per wrap k (intervals k -> k + 1) the shares with 0 / 1 / 2+ peaks, the medians of rct, rnet, gap, the GT turn
   shares, and a verdict from the intervals on both sides of k: dup (an empty share >= --dup on either side), merged (a 2+ share
   >= --dup), real otherwise; --tsv OUT writes that table. --w-scale S (a control): the wraps of S x W (2: a fit counting every turn twice).
   Defaults --sigma 1.2 --prom 1: on Paris 4 (9.6 um CT) 1.05 CT peaks per fitted turn where the GT codes give 1.03. Coordinates: the model's box (native voxels, global origin in model.json);
   the fields may be any box holding it at the same native grid. CPU only, rays on --threads threads. */
#include "fit.h"
#include "sysinfo.h"
#include "zarr3.h"
#include "store.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char *wc_opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 3; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}
static int wc_idx(int argc, char **argv, const char *name) { for (int i = 3; i < argc; i++) if (!strcmp(argv[i], name)) return i; return -1; }
static double wc_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

typedef struct { int k; float z, r, gapn, rct, rnet, gtq; uint8_t npk; int8_t gt; } wc_int;   /* the interval from wrap k to k + 1 on a ray; gtq the GT turns, gt their class (< 0.5: 0, < 1.5: 1, 2; < -0.25: -1; -128 none) */

typedef struct {
    const fit_set *fs; const fit_model *m;
    double zl, step, um, h, F, cf, ws; int dirv;
    /* fields row (cells), offset of the model box in the fields box (native) */
    const uint8_t *w, *sp; int FH, FW; double fo[2];
    /* CT plane (float, CT voxels), its origin (CT voxels, global) */
    const float *ct; int64_t co[2], cn[2];
    /* codes plane, its origin (label voxels, global) */
    const uint8_t *cd; int64_t qo[2], qn[2];
    double morg[3];
    int nth; double sigma, prom;
    wc_int **out; int *nout;   /* per ray */
} wc_job;

static double wc_ct(const wc_job *J, double y, double x) {   /* box-local native -> bilinear CT */
    const double cy = (y + J->morg[1] + 0.5) * J->F - 0.5 - J->co[0], cx = (x + J->morg[2] + 0.5) * J->F - 0.5 - J->co[1];
    const int64_t y0 = (int64_t)floor(cy), x0 = (int64_t)floor(cx); const double ty = cy - y0, tx = cx - x0;
    if (y0 < 0 || x0 < 0 || y0 + 1 >= J->cn[0] || x0 + 1 >= J->cn[1]) return NAN;
    const float *p = J->ct + y0 * J->cn[1] + x0;
    return (1 - ty) * ((1 - tx) * p[0] + tx * p[1]) + ty * ((1 - tx) * p[J->cn[1]] + tx * p[J->cn[1] + 1]);
}
static int wc_code(const wc_job *J, double y, double x) {
    const int64_t qy = (int64_t)floor((y + J->morg[1]) / J->cf) - J->qo[0], qx = (int64_t)floor((x + J->morg[2]) / J->cf) - J->qo[1];
    if (qy < 0 || qx < 0 || qy >= J->qn[0] || qx >= J->qn[1]) return 0;
    const int c = J->cd[qy * J->qn[1] + qx]; return c >= 1 && c <= 252 ? c : 0;
}
static int cmp_f(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }

static void wc_rays(void *ctx, int a0, int a1, int t) {
    wc_job *J = ctx; (void)t;
    const fit_model *m = J->m; const int na = (int)(m->D * m->h) + 1; const int iz = (int)fmin(J->zl, na - 1);
    const double cy = m->cy[iz], cx = m->cx[iz], Y = m->H * m->h, X = m->W * m->h, st = J->step;
    for (int j = a0; j < a1; j++) {
        const double th = 2 * M_PI * (j + 0.5) / J->nth, sn = sin(th), cs = cos(th);
        double rmax = 0; for (double r = 0;; r += 16) { const double y = cy + r * sn, x = cx + r * cs; if (r > 0 && (y < 0 || x < 0 || y >= Y || x >= X)) break; rmax = r + 16; if (r > 4 * (Y + X)) break; }
        const int n = (int)(rmax / st) + 1;
        float *W = malloc(sizeof(float) * n), *C = malloc(sizeof(float) * n), *Cs = malloc(sizeof(float) * n), *spn = malloc(sizeof(float) * n);
        uint8_t *mat = calloc(n, 1); short *q = calloc(n, sizeof(short));
        for (int i = 0; i < n; i++) {
            const double r = i * st, y = cy + r * sn, x = cx + r * cs;
            W[i] = NAN; C[i] = NAN; spn[i] = 0;
            if (y < 0 || x < 0 || y >= Y || x >= X) continue;
            const int fy = (int)floor((y + J->fo[0]) / J->h), fx = (int)floor((x + J->fo[1]) / J->h);
            if (fy < 0 || fx < 0 || fy >= J->FH || fx >= J->FW) continue;
            const size_t ci = (size_t)fy * J->FW + fx;
            mat[i] = J->w[ci] != 0; spn[i] = J->sp[ci] * J->h / 8.0f;
            W[i] = (float)(J->ws * fit_set_W(J->fs, J->zl, y, x));
            if (J->ct) C[i] = (float)wc_ct(J, y, x);
            if (J->cd) q[i] = (short)wc_code(J, y, x);
        }
        /* CT peaks: gaussian smoothing, local mean / sd (+-12 native), prominence within +-24 native */
        int npk = 0; float *pk = nullptr;
        if (J->ct) {
            const int R = (int)ceil(3 * J->sigma / st); double *gk = malloc(sizeof(double) * (2 * R + 1)), gs = 0;
            for (int d = -R; d <= R; d++) { gk[d + R] = exp(-0.5 * (d * st / J->sigma) * (d * st / J->sigma)); gs += gk[d + R]; }
            for (int i = 0; i < n; i++) {
                double s = 0, ws = 0;
                for (int d = -R; d <= R; d++) { const int k = i + d; if (k < 0 || k >= n || !(C[k] == C[k])) continue; s += gk[d + R] * C[k]; ws += gk[d + R]; }
                Cs[i] = ws > 0 ? (float)(s / ws) : NAN;
            }
            free(gk);
            const int L = (int)lround(12 / st), P = (int)lround(24 / st);
            double *ps = calloc(n + 1, sizeof(double)), *ps2 = calloc(n + 1, sizeof(double)); int *pc = calloc(n + 1, sizeof(int));
            for (int i = 0; i < n; i++) { const int ok = Cs[i] == Cs[i]; ps[i + 1] = ps[i] + (ok ? Cs[i] : 0); ps2[i + 1] = ps2[i] + (ok ? (double)Cs[i] * Cs[i] : 0); pc[i + 1] = pc[i] + ok; }
            pk = malloc(sizeof(float) * (n / 2 + 1));
            for (int i = 1; i + 1 < n; i++) {
                if (!mat[i] || !(Cs[i] == Cs[i]) || !(Cs[i] > Cs[i - 1]) || Cs[i] < Cs[i + 1]) continue;   /* (plateaus: their first sample) */
                const int a = i - L < 0 ? 0 : i - L, b = i + L + 1 > n ? n : i + L + 1, c = pc[b] - pc[a];
                if (c < 4) continue;
                const double mu = (ps[b] - ps[a]) / c, sd = sqrt(fmax(0, (ps2[b] - ps2[a]) / c - mu * mu));
                if (!(sd > 0)) continue;
                double ml = Cs[i], mr = Cs[i]; int hl = 0, hr = 0;
                for (int k = i - 1; k >= 0 && k >= i - P; k--) { if (!(Cs[k] == Cs[k])) break; if (Cs[k] > Cs[i]) { hl = 1; break; } if (Cs[k] < ml) ml = Cs[k]; }
                for (int k = i + 1; k < n && k <= i + P; k++) { if (!(Cs[k] == Cs[k])) break; if (Cs[k] >= Cs[i]) { hr = 1; break; } if (Cs[k] < mr) mr = Cs[k]; }
                (void)hl; (void)hr;
                const double prom = Cs[i] - fmax(ml, mr);
                if (prom >= J->prom * sd) pk[npk++] = (float)(i * st);
            }
            free(ps); free(ps2); free(pc);
        }
        /* the wrap crossings inside material */
        int nc = 0, cap = 64; double *cr = malloc(sizeof(double) * cap); int *ck = malloc(sizeof(int) * cap), *ci = malloc(sizeof(int) * cap), *cu = malloc(sizeof(int) * cap);
        for (int i = 0; i + 1 < n; i++) {
            if (!mat[i] || !mat[i + 1] || !(W[i] == W[i]) || !(W[i + 1] == W[i + 1])) continue;
            const double fa = floor(W[i]), fb = floor(W[i + 1]);
            if (fa == fb || fabs(W[i + 1] - W[i]) >= 0.5) continue;
            const double k = fmax(fa, fb), t = (k - W[i]) / (W[i + 1] - W[i]);
            if (nc == cap) { cap *= 2; cr = realloc(cr, sizeof(double) * cap); ck = realloc(ck, sizeof(int) * cap); ci = realloc(ci, sizeof(int) * cap); cu = realloc(cu, sizeof(int) * cap); }
            cr[nc] = (i + t) * st; ck[nc] = (int)k; cu[nc] = W[i + 1] > W[i]; ci[nc] = i; nc++;   /* (cu: W increasing outward) */
        }
        wc_int *o = malloc(sizeof(wc_int) * (nc + 1)); int no = 0;
        for (int c = 0; c + 1 < nc; c++) {
            if (!cu[c] || !cu[c + 1] || ck[c + 1] != ck[c] + 1) continue;   /* outward k -> k + 1 only */
            int allm = 1; for (int i = ci[c]; i <= ci[c + 1] + 1 && i < n; i++) if (!mat[i]) { allm = 0; break; }
            if (!allm) continue;
            const double ra = cr[c], rb = cr[c + 1], rm = 0.5 * (ra + rb);
            const double ym = cy + rm * sn, xm = cx + rm * cs; double g[3];
            const double gl = fit_set_gradW(J->fs, J->zl, ym, xm, 0.5, g);
            const double cosr = gl > 0 ? fabs(g[1] * sn + g[2] * cs) / gl : 1;
            wc_int v = {ck[c], (float)J->zl, (float)rm, (float)((rb - ra) * cosr * J->um), NAN, NAN, NAN, 0, -128};
            const int im = (int)lround(rm / st); const double spm = im < n ? spn[im] : 0;
            if (spm > 0) v.rnet = (float)((rb - ra) * cosr / spm);
            if (J->ct) {
                int c0 = 0; for (int p = 0; p < npk; p++) if (pk[p] >= ra && pk[p] < rb) c0++;
                v.npk = (uint8_t)(c0 > 255 ? 255 : c0);
                float dd[64]; int nd = 0;
                for (int p = 0; p + 1 < npk && nd < 64; p++) if (pk[p] >= rm - 40 && pk[p + 1] <= rm + 40) dd[nd++] = pk[p + 1] - pk[p];
                if (nd >= 2) { qsort(dd, nd, sizeof(float), cmp_f); v.rct = (float)((rb - ra) / dd[nd / 2]); }
            }
            if (J->cd) {
                int qa = 0, qb = 0;
                for (int d = 0; d <= 2 && !qa; d++) { const int i1 = (int)lround(ra / st) - d, i2 = (int)lround(ra / st) + d; if (i1 >= 0 && i1 < n && q[i1]) qa = q[i1]; else if (i2 >= 0 && i2 < n && q[i2]) qa = q[i2]; }
                for (int d = 0; d <= 2 && !qb; d++) { const int i1 = (int)lround(rb / st) - d, i2 = (int)lround(rb / st) + d; if (i1 >= 0 && i1 < n && q[i1]) qb = q[i1]; else if (i2 >= 0 && i2 < n && q[i2]) qb = q[i2]; }
                if (qa && qb) { double dq = fmod((qb - qa) / 18.0, 14.0); if (dq < 0) dq += 14; if (dq > 7) dq -= 14; v.gtq = (float)dq; v.gt = dq < -0.25 ? -1 : dq < 0.5 ? 0 : dq < 1.5 ? 1 : 2; }
            }
            o[no++] = v;
        }
        J->out[j] = o; J->nout[j] = no;
        free(W); free(C); free(Cs); free(spn); free(mat); free(q); free(pk); free(cr); free(ck); free(ci); free(cu);
    }
}

static int cmp_int_k(const void *a, const void *b) { const wc_int *x = a, *y = b; return (x->k > y->k) - (x->k < y->k); }
static double wc_median(float *v, int n) { if (!n) return NAN; qsort(v, n, sizeof(float), cmp_f); return v[n / 2]; }

int cmd_wrap_check(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: ufsm wrap-check <model-or-set> <fields-dir> [--ct ROOT KEY --ct-um U] [--codes ROOT KEY] [--nz 16] [--nth 180] [--zr Z0,Z1]\n"
                        "       [--sigma 1.2] [--prom 1] [--w-scale 1] [--dup 0.5] [--threads 16] [--cache DIR] [--tsv OUT]\n"
                        "  per adjacent wrap pair along radial rays: normal gap, CT papyrus peaks between them, GT turns (--codes); per-wrap verdicts\n");
        return 2;
    }
    fit_set *fs = fit_set_load(argv[2]); if (!fs) { fprintf(stderr, "wrap-check: cannot load %s\n", argv[2]); return 1; }
    const fit_model *m = fs->global;
    fit_fields_skip_planes(1 | 2 | 4); fit_fields_ct_planes(0);
    fit_fields_buf g; if (fit_fields_load(argv[3], 0, 0, 0, &g)) { fprintf(stderr, "wrap-check: cannot read the fields in %s\n", argv[3]); return 1; }
    const double h = g.h, um = g.um; long long forg[3]; int fsh[3]; for (int d = 0; d < 3; d++) { forg[d] = g.origin[d]; fsh[d] = g.shape[d]; } fit_fields_free(&g);
    if (m->h != h) { fprintf(stderr, "wrap-check: the model's cell h %g differs from the fields' %g\n", m->h, h); return 1; }
    const int nz = atoi(wc_opt(argc, argv, "--nz", "16")), nth = atoi(wc_opt(argc, argv, "--nth", "180")), nt = ufsm_threads(atoi(wc_opt(argc, argv, "--threads", "0")), 16);
    const double sigma = atof(wc_opt(argc, argv, "--sigma", "1.2")), prom = atof(wc_opt(argc, argv, "--prom", "1")), wsc = atof(wc_opt(argc, argv, "--w-scale", "1")), dup = atof(wc_opt(argc, argv, "--dup", "0.5"));
    const double Z = m->D * m->h; double zr0 = 0.05 * Z, zr1 = 0.95 * Z;
    const char *zr = wc_opt(argc, argv, "--zr", nullptr); if (zr && sscanf(zr, "%lf,%lf", &zr0, &zr1) != 2) { fprintf(stderr, "wrap-check: --zr Z0,Z1\n"); return 2; }
    if (nz < 1 || nth < 4 || !(sigma > 0) || !(zr1 > zr0)) { fprintf(stderr, "wrap-check: bad --nz / --nth / --sigma / --zr\n"); return 2; }
    fit_set_threads(nt);
    z3 *cz = nullptr, *qz = nullptr; double ctum = 0, cf = 1;
    const int ci = wc_idx(argc, argv, "--ct"), qi = wc_idx(argc, argv, "--codes");
    if (ci > 0) {
        if (ci + 2 >= argc) { fprintf(stderr, "wrap-check: --ct ROOT KEY\n"); return 2; }
        ctum = atof(wc_opt(argc, argv, "--ct-um", "0")); if (!(ctum > 0)) { fprintf(stderr, "wrap-check: --ct needs --ct-um\n"); return 2; }
        if (!(cz = z3_open(store_open(argv[ci + 1]), argv[ci + 2], wc_opt(argc, argv, "--cache", nullptr)))) { fprintf(stderr, "wrap-check: %s\n", z3_error()); return 1; }
    }
    if (qi > 0) {
        if (qi + 2 >= argc) { fprintf(stderr, "wrap-check: --codes ROOT KEY\n"); return 2; }
        if (!(qz = z3_open(store_open(argv[qi + 1]), argv[qi + 2], nullptr))) { fprintf(stderr, "wrap-check: %s\n", z3_error()); return 1; }
        cf = atof(argv[qi + 2]) > 0 ? atof(argv[qi + 2]) / um : 1;
    }
    const double F = cz ? um / ctum : 1, step = fmin(1.0, fmax(0.25, 1 / F));
    const double morg[3] = {(double)m->origin[0], (double)m->origin[1], (double)m->origin[2]};
    printf("wrap-check: %s: box %dx%dx%d cells h %g (%.3f um), %d z levels in [%.0f, %.0f) x %d rays, step %.2f native%s%s\n", argv[2], m->D, m->H, m->W, h, um,
           nz, zr0, zr1, nth, step, cz ? ", CT" : "", qz ? ", GT codes" : "");
    const double t0 = wc_now();
    size_t ni = 0, capi = 1 << 16; wc_int *all = malloc(sizeof(wc_int) * capi);
    const double Y = m->H * h, X = m->W * h;
    for (int zi = 0; zi < nz; zi++) {
        const double zl = floor(zr0 + (zi + 0.5) * (zr1 - zr0) / nz) + 0.5, zg = zl + morg[0];
        const int fz = (int)floor((zg - forg[0]) / h);
        if (fz < 0 || fz >= fsh[0]) continue;
        fit_fields_buf b; if (fit_fields_load(argv[3], fz, fz + 1, 1, &b)) { fprintf(stderr, "wrap-check: cannot load field row %d\n", fz); return 1; }
        wc_job J = {.fs = fs, .m = m, .zl = zl, .step = step, .um = um, .h = h, .F = F, .cf = cf, .ws = wsc, .w = b.f.w, .sp = b.f.sp, .FH = fsh[1], .FW = fsh[2],
                    .fo = {morg[1] - forg[1], morg[2] - forg[2]}, .morg = {morg[0], morg[1], morg[2]}, .nth = nth, .sigma = sigma, .prom = prom};
        float *ctp = nullptr; uint8_t *cdp = nullptr;
        if (cz) {   /* the CT rows covering native z [zg - 0.5, zg + 0.5), averaged; the box's y / x range + 2 */
            const z3_meta *zm = z3_meta_of(cz);
            int64_t c0 = (int64_t)floor((zg - 0.5) * F), c1 = (int64_t)ceil((zg + 0.5) * F); if (c1 <= c0) c1 = c0 + 1;
            if (c0 < 0) c0 = 0;
            if (c1 > zm->shape[0]) c1 = zm->shape[0];
            int64_t o3[3] = {c0, (int64_t)floor(morg[1] * F) - 2, (int64_t)floor(morg[2] * F) - 2}, n3[3] = {c1 - c0, (int64_t)ceil(Y * F) + 4, (int64_t)ceil(X * F) + 4};
            for (int d = 1; d < 3; d++) { if (o3[d] < 0) { n3[d] += o3[d]; o3[d] = 0; } if (o3[d] + n3[d] > zm->shape[d]) n3[d] = zm->shape[d] - o3[d]; }
            if (n3[0] > 0 && n3[1] > 1 && n3[2] > 1) {
                uint8_t *raw = malloc((size_t)n3[0] * n3[1] * n3[2]);
                if (!raw || z3_read(cz, o3, n3, raw, nt)) { fprintf(stderr, "wrap-check: CT read failed: %s\n", z3_error()); return 1; }
                const size_t pl = (size_t)n3[1] * n3[2]; ctp = calloc(pl, sizeof(float));
                for (int64_t a = 0; a < n3[0]; a++) for (size_t i = 0; i < pl; i++) ctp[i] += raw[a * pl + i];
                for (size_t i = 0; i < pl; i++) ctp[i] /= (float)n3[0];
                free(raw);
                J.ct = ctp; J.co[0] = o3[1]; J.co[1] = o3[2]; J.cn[0] = n3[1]; J.cn[1] = n3[2];
            }
        }
        if (qz) {
            const z3_meta *zm = z3_meta_of(qz);
            int64_t o3[3] = {(int64_t)floor(zg / cf), (int64_t)floor(morg[1] / cf), (int64_t)floor(morg[2] / cf)}, n3[3] = {1, (int64_t)ceil(Y / cf) + 1, (int64_t)ceil(X / cf) + 1};
            for (int d = 1; d < 3; d++) if (o3[d] + n3[d] > zm->shape[d]) n3[d] = zm->shape[d] - o3[d];
            if (o3[0] < zm->shape[0] && n3[1] > 0 && n3[2] > 0) {
                cdp = malloc((size_t)n3[1] * n3[2]);
                if (z3_read(qz, o3, n3, cdp, nt)) { fprintf(stderr, "wrap-check: code read failed: %s\n", z3_error()); return 1; }
                J.cd = cdp; J.qo[0] = o3[1]; J.qo[1] = o3[2]; J.qn[0] = n3[1]; J.qn[1] = n3[2];
            }
        }
        J.out = calloc(nth, sizeof(wc_int *)); J.nout = calloc(nth, sizeof(int));
        fit_par(nth, wc_rays, &J);
        for (int j = 0; j < nth; j++) {
            if (ni + J.nout[j] > capi) { while (ni + J.nout[j] > capi) capi *= 2; all = realloc(all, sizeof(wc_int) * capi); }
            memcpy(all + ni, J.out[j], sizeof(wc_int) * J.nout[j]); ni += J.nout[j]; free(J.out[j]);
        }
        free(J.out); free(J.nout); free(ctp); free(cdp); fit_fields_free(&b);
        fprintf(stderr, "\rwrap-check: z %d / %d (%zu intervals, %.0f s)", zi + 1, nz, ni, wc_now() - t0);
    }
    fprintf(stderr, "\n");
    if (!ni) { printf("wrap-check: no intervals (no material crossings)\n"); return 1; }
    /* radius bands of 1 mm */
    const double band = 1000 / um; int nb = 0; for (size_t i = 0; i < ni; i++) { const int bi = (int)(all[i].r / band); if (bi + 1 > nb) nb = bi + 1; }
    long long (*bs)[6] = calloc(nb, sizeof *bs);   /* n, sum npk, n with GT, sum GT, empties, 2+ */
    long long conf[3][4] = {{0}};   /* npk 0 / 1 / 2+ x GT 0 / 1 / 2+ / other */
    long long tn = 0, tp = 0, tg = 0, tgn = 0;
    for (size_t i = 0; i < ni; i++) {
        const wc_int *v = &all[i]; const int bi = (int)(v->r / band);
        bs[bi][0]++; bs[bi][1] += v->npk; bs[bi][4] += v->npk == 0; bs[bi][5] += v->npk >= 2; tn++; tp += v->npk;
        if (v->gt != -128) { bs[bi][2]++; bs[bi][3] += lround(1000 * v->gtq); tg += lround(1000 * v->gtq); tgn++; conf[v->npk > 2 ? 2 : v->npk][v->gt == 0 ? 0 : v->gt == 1 ? 1 : v->gt >= 2 ? 2 : 3]++; }
    }
    printf("wrap-check: %zu intervals (consecutive outward crossings in material) in %.0f s\n", ni, wc_now() - t0);
    if (cz) printf("  CT papyrus peaks per fitted turn %.3f (1 = right; < 1: the fit counts more turns than the CT shows sheets)\n", (double)tp / tn);
    if (qz) printf("  GT turns per fitted turn %.3f over %lld intervals with codes at both ends (%.3f of all)\n", tgn ? (double)tg / 1000 / tgn : 0, tgn, (double)tgn / tn);
    printf("  radius(mm)  intervals  CTpeaks/turn  empty  2+  %s\n", qz ? "GT-n  GT/turn" : "");
    for (int bi = 0; bi < nb; bi++) {
        if (bs[bi][0] < 20) continue;
        printf("  %4.0f-%-4.0f  %9lld  %12.3f  %5.3f  %5.3f", bi * band * um / 1000, (bi + 1) * band * um / 1000, bs[bi][0], (double)bs[bi][1] / bs[bi][0],
               (double)bs[bi][4] / bs[bi][0], (double)bs[bi][5] / bs[bi][0]);
        if (qz) printf("  %6lld  %6.3f", bs[bi][2], bs[bi][2] ? (double)bs[bi][3] / 1000 / bs[bi][2] : 0);
        printf("\n");
    }
    if (cz) {   /* per z level (the intervals are in z order) */
        printf("  z(native)  intervals  CTpeaks/turn  empty  2+%s\n", qz ? "  GT-n  GT/turn" : "");
        for (size_t a = 0; a < ni;) {
            size_t b = a; long long s = 0, e0 = 0, e2 = 0, gn = 0, gs = 0;
            for (; b < ni && all[b].z == all[a].z; b++) { s += all[b].npk; e0 += all[b].npk == 0; e2 += all[b].npk >= 2; if (all[b].gt != -128) { gn++; gs += lround(1000 * all[b].gtq); } }
            printf("  %9.0f  %9zu  %12.3f  %5.3f  %5.3f", all[a].z, b - a, (double)s / (b - a), (double)e0 / (b - a), (double)e2 / (b - a));
            if (qz) printf("  %6lld  %6.3f", gn, gn ? gs / 1000.0 / gn : 0);
            printf("\n"); a = b;
        }
    }
    if (qz && cz) {
        printf("  validation: CT peaks in the interval (rows) x GT turns (columns 0 / 1 / 2+ / <0), counts\n");
        for (int a = 0; a < 3; a++) printf("    npk %s  %9lld %9lld %9lld %9lld\n", a == 0 ? "0 " : a == 1 ? "1 " : "2+", conf[a][0], conf[a][1], conf[a][2], conf[a][3]);
        const long long g0 = conf[0][0] + conf[1][0] + conf[2][0], g2 = conf[0][2] + conf[1][2] + conf[2][2], e0 = conf[0][0] + conf[0][1] + conf[0][2], e2 = conf[2][0] + conf[2][1] + conf[2][2];
        printf("    GT-duplicate intervals (GT 0) with no CT peak: recall %.3f, precision %.3f; GT-merged (GT 2+) with 2+ peaks: recall %.3f, precision %.3f\n",
               g0 ? (double)conf[0][0] / g0 : 0, e0 ? (double)conf[0][0] / e0 : 0, g2 ? (double)conf[2][2] / g2 : 0, e2 ? (double)conf[2][2] / e2 : 0);
    }
    /* per wrap: the intervals k -> k + 1 */
    qsort(all, ni, sizeof(wc_int), cmp_int_k);
    int kmin = all[0].k, kmax = all[ni - 1].k, nk = kmax - kmin + 1;
    typedef struct { long long n, e0, e1, e2, gn, g0, g1, g2; double gs, mrct, mrnet, mgap; } wst;
    wst *W = calloc(nk, sizeof *W); float *tmp = malloc(sizeof(float) * ni);
    for (size_t a = 0; a < ni;) {
        size_t b = a; while (b < ni && all[b].k == all[a].k) b++;
        wst *s = &W[all[a].k - kmin]; int nt2 = 0;
        for (size_t i = a; i < b; i++) {
            const wc_int *v = &all[i]; s->n++; s->e0 += v->npk == 0; s->e1 += v->npk == 1; s->e2 += v->npk >= 2;
            if (v->gt != -128) { s->gn++; s->gs += v->gtq; s->g0 += v->gt == 0; s->g1 += v->gt == 1; s->g2 += v->gt >= 2; }
        }
        nt2 = 0; for (size_t i = a; i < b; i++) if (all[i].rct == all[i].rct) tmp[nt2++] = all[i].rct; s->mrct = wc_median(tmp, nt2);
        nt2 = 0; for (size_t i = a; i < b; i++) if (all[i].rnet == all[i].rnet) tmp[nt2++] = all[i].rnet; s->mrnet = wc_median(tmp, nt2);
        nt2 = 0; for (size_t i = a; i < b; i++) tmp[nt2++] = all[i].gapn; s->mgap = wc_median(tmp, nt2);
        a = b;
    }
    FILE *tf = nullptr; const char *tsv = wc_opt(argc, argv, "--tsv", nullptr);
    if (tsv && !(tf = fopen(tsv, "w"))) { fprintf(stderr, "wrap-check: cannot write %s\n", tsv); return 1; }
    const char *hdr = "wrap\tintervals\tempty\tone\ttwo_plus\trct_med\trnet_med\tgap_um_med\tgt_n\tgt_mean\tgt0\tgt1\tgt2\tverdict\n";
    if (tf) fputs(hdr, tf);
    int ndup = 0, nmrg = 0, nreal = 0; long long minn = 0; for (int k = 0; k < nk; k++) if (W[k].n > minn) minn = W[k].n; minn = minn / 20 > 10 ? minn / 20 : 10;
    printf("  per wrap (interval k -> k + 1; verdict from both sides of k, wraps with >= %lld intervals):\n  wrap  intervals  empty   one   2+   rct  rnet  gap(um)%s  verdict\n", minn, qz ? "   gt-n  gt-mean   gt0   gt1   gt2" : "");
    for (int k = 0; k < nk; k++) {
        const wst *s = &W[k], *p = k ? &W[k - 1] : nullptr;
        if (s->n < minn) continue;
        const double e_out = (double)s->e0 / s->n, e_in = p && p->n >= minn ? (double)p->e0 / p->n : 0;
        const double m_out = (double)s->e2 / s->n, m_in = p && p->n >= minn ? (double)p->e2 / p->n : 0;
        const char *vd = !cz ? "-" : fmax(e_out, e_in) >= dup ? "dup" : fmax(m_out, m_in) >= dup ? "merged" : "real";
        if (cz) { ndup += !strcmp(vd, "dup"); nmrg += !strcmp(vd, "merged"); nreal += !strcmp(vd, "real"); }
        printf("  w%03d  %9lld  %5.3f %5.3f %5.3f %5.2f %5.2f %7.1f", kmin + k, s->n, (double)s->e0 / s->n, (double)s->e1 / s->n, (double)s->e2 / s->n, s->mrct, s->mrnet, s->mgap);
        if (qz) printf("  %6lld  %7.3f %5.3f %5.3f %5.3f", s->gn, s->gn ? s->gs / s->gn : 0, s->gn ? (double)s->g0 / s->gn : 0, s->gn ? (double)s->g1 / s->gn : 0, s->gn ? (double)s->g2 / s->gn : 0);
        printf("  %s\n", vd);
        if (tf) fprintf(tf, "w%03d\t%lld\t%.4f\t%.4f\t%.4f\t%.3f\t%.3f\t%.1f\t%lld\t%.3f\t%.4f\t%.4f\t%.4f\t%s\n", kmin + k, s->n, (double)s->e0 / s->n, (double)s->e1 / s->n, (double)s->e2 / s->n,
                        s->mrct, s->mrnet, s->mgap, s->gn, s->gn ? s->gs / s->gn : 0, s->gn ? (double)s->g0 / s->gn : 0, s->gn ? (double)s->g1 / s->gn : 0, s->gn ? (double)s->g2 / s->gn : 0, vd);
    }
    if (cz) printf("wrap-check: verdicts: %d real, %d dup, %d merged (--dup %g)\n", nreal, ndup, nmrg, dup);
    if (tf) fclose(tf);
    free(W); free(tmp); free(all); free(bs); fit_set_free(fs); if (cz) z3_close(cz); if (qz) z3_close(qz);
    return 0;
}
