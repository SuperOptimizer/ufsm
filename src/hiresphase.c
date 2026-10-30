/* ufsm hires-phase CTROOT KEY FIELDS-DIR OUT-DIR --ct-um U [options]
   The sheet phase of a fit-field set measured in a finer CT (e.g. the 2.4 um scan under 9.6 um fields): per field cell the CT is
   sampled along the fields' normal (the top eigenvector of n n^T, oriented away from the axis), averaged over a disc of radius
   --rt-um in the sheet plane, smoothed (Gaussian --sigma-um) and its valleys (the gaps between sheets; --feature peak: the sheets'
   ridges) found with a prominence of at least --prom x the profile's standard deviation (and --prom-abs CT units): shallower ones
   are merged into their neighbours, deepest first kept. The cell centre lies between valleys a <= 0 < b (sub-voxel), so its CT phase
   is -a / (b - a) (one turn per valley interval, increasing outward). Confidence = tanh(min prominence / (2 sd)) x a regularity term
   (the interval against the profile's median interval). The phase's gauge (where in a turn the CT valley sits) comes from a
   reference field set's confident cells: the input fields themselves (default: the network's phase, which fixes the gauge but not
   the count) or --gauge-ref DIR (e.g. a label oracle, for validation), or --gauge PHI.
   Output: OUT-DIR, the input fields with the phase planes replaced where the CT gives a phase (confidence >= --min-conf); elsewhere
   the input's (--fill net) or no information (--fill none).
   --map MAP.json (ufsm reg map: fine L0 xyz -> coarse L0 xyz + field D): a finer scan registered to the fields' scan; without it the
   CT is the fields' scan at another level (voxel c at native (c + 0.5) / F - 0.5, F = fields um / ct um). */
#include <complex.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "fit.h"
#include "store.h"
#include "zarr3.h"
#include "reg.h"
#include "z3w.h"
#include "wind.h"

static const char *hp_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
static double hp_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

enum { HP_MAXN = 1024, HP_MAXM = 256 };
typedef struct {
    const fit_fields *f; int fz0, H, W;        /* slab fields (rows fz0.. of the box) */
    const uint8_t *vol; int64_t vo[3], vn[3];  /* CT box (CT voxels) */
    double F, um, ctum, org[3], ms;            /* ms: fields voxels per map-coarse L0 voxel (the fields on level log2 ms of the map's coarse scan) */
    const rmap *map;                        /* null: same scan, scale F */
    double rt, sigma, prom, prom_abs, rmin, rmax;   /* CT voxels (rmin / rmax: um) */
    int feature, stride;                       /* 0 valley, 1 peak */
    double gabor;                              /* > 0: the phase by demodulation at the fields' spacing over +-gabor periods (no extrema) */
    float *phi, *conf;                         /* whole box (NaN: none) */
    _Atomic int next; _Atomic long long ncell, nphase;
} hp_job;

static float hp_sample(const hp_job *J, double cz, double cy, double cx) {   /* trilinear CT at CT voxel coordinates; -1 outside / no data */
    const double q[3] = {cz - J->vo[0], cy - J->vo[1], cx - J->vo[2]}; int i0[3]; double t[3];
    for (int d = 0; d < 3; d++) { if (!(q[d] >= 0) || q[d] > J->vn[d] - 1) return -1.f; i0[d] = (int)q[d]; if (i0[d] > J->vn[d] - 2) i0[d] = (int)J->vn[d] - 2; if (i0[d] < 0) i0[d] = 0; t[d] = q[d] - i0[d]; }
    double s = 0; int zero = 0;
    for (int c = 0; c < 8; c++) {
        const int a = i0[0] + (c >> 2 & 1), b = i0[1] + (c >> 1 & 1), e = i0[2] + (c & 1);
        const double w = (c >> 2 & 1 ? t[0] : 1 - t[0]) * (c >> 1 & 1 ? t[1] : 1 - t[1]) * (c & 1 ? t[2] : 1 - t[2]);
        const uint8_t v = J->vol[((size_t)(a < J->vn[0] ? a : J->vn[0] - 1) * J->vn[1] + (b < J->vn[1] ? b : J->vn[1] - 1)) * J->vn[2] + (e < J->vn[2] ? e : J->vn[2] - 1)];
        zero |= v == 0 && w > 0.25; s += w * v;
    }
    return zero ? -1.f : (float)s;
}
/* native box-local point -> CT voxel coordinates (z, y, x) */
static void hp_to_ct(const hp_job *J, const double p[3], double c[3]) {
    if (!J->map) { for (int d = 0; d < 3; d++) c[d] = (p[d] + J->org[d] + 0.5) * J->F - 0.5; return; }
    const double s = J->ms > 0 ? J->ms : 1, g[3] = {(p[2] + J->org[2] + 0.5) * s - 0.5, (p[1] + J->org[1] + 0.5) * s - 0.5, (p[0] + J->org[0] + 0.5) * s - 0.5}; double q[3];   /* (the map: x y z, coarse L0) */
    rmap_c2f(J->map, g, q); c[0] = q[2]; c[1] = q[1]; c[2] = q[0];
}

/* valleys (feature 0) or peaks (1) of a smoothed profile p[0..n) with prominence >= thr: candidates are the local extrema; the one with
   the least prominence (depth below the lower of the two maxima between it and its neighbouring candidates) is merged away until all
   pass. pos: sub-voxel positions, pr: prominences; returns the count */
static int hp_extrema(const float *p, int n, int feature, double thr, double *pos, double *pr) {
    static_assert(HP_MAXM <= HP_MAXN, "");
    float q[HP_MAXN]; for (int k = 0; k < n; k++) q[k] = feature ? -p[k] : p[k];   /* minima of q */
    int m[HP_MAXM], nm = 0;
    for (int k = 1; k + 1 < n && nm < HP_MAXM; k++) if (q[k] < q[k - 1] && q[k] <= q[k + 1]) m[nm++] = k;
    float mx[HP_MAXM + 1];   /* mx[j]: the max of q between candidate j-1 and j (mx[0]: start .. m[0], mx[nm]: m[nm-1] .. end) */
    double prom[HP_MAXM];
    for (;;) {
        if (!nm) return 0;
        for (int j = 0; j <= nm; j++) {
            const int a = j ? m[j - 1] : 0, b = j < nm ? m[j] : n - 1; float v = q[a]; for (int k = a; k <= b; k++) if (q[k] > v) v = q[k]; mx[j] = v;
        }
        int worst = -1; double wp = 1e30;
        for (int j = 0; j < nm; j++) {   /* an end interval counts only if the profile rises there (else the extremum is open on that side) */
            const double l = mx[j] - q[m[j]], r = mx[j + 1] - q[m[j]], pj = fmin(l, r); prom[j] = pj;
            if (pj < wp) { wp = pj; worst = j; }
        }
        if (wp >= thr) break;
        memmove(m + worst, m + worst + 1, (size_t)(nm - worst - 1) * sizeof *m); nm--;
    }
    for (int j = 0; j < nm; j++) {
        const int k = m[j]; const double a = q[k - 1], b = q[k], c = q[k + 1], den = a - 2 * b + c;
        pos[j] = k + (den > 1e-9 ? fmax(-0.5, fmin(0.5, 0.5 * (a - c) / den)) : 0); pr[j] = prom[j];
    }
    return nm;
}
static int hp_cmpd(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

static void *hp_rows(void *arg) {
    hp_job *J = arg; const fit_fields *f = J->f; const double h = f->h; const int st = J->stride; const int na = (int)(J->f->D * h) + 1;
    float prof[HP_MAXN], sm[HP_MAXN]; double pos[HP_MAXM], pr[HP_MAXM], iv[HP_MAXM];
    /* the Gaussian kernel and the disc offsets (sheet-plane units of rt) */
    const int gk = (int)ceil(3 * J->sigma); float gw[64]; double gs = 0; const int GK = gk > 31 ? 31 : gk;
    for (int k = -GK; k <= GK; k++) { gw[k + GK] = J->sigma > 0 ? (float)exp(-0.5 * k * k / (J->sigma * J->sigma)) : (k == 0); gs += gw[k + GK]; }
    for (int k = 0; k <= 2 * GK; k++) gw[k] /= (float)gs;
    double da[32], db[32]; int nd = 0;
    da[nd] = 0; db[nd++] = 0;
    if (J->rt > 0) {
        for (int k = 0; k < 6; k++) { da[nd] = 0.5 * cos(k * M_PI / 3); db[nd++] = 0.5 * sin(k * M_PI / 3); }
        for (int k = 0; k < 12; k++) { da[nd] = cos(k * M_PI / 6 + M_PI / 12); db[nd++] = sin(k * M_PI / 6 + M_PI / 12); }
    }
    long long ncell = 0, nph = 0;
    for (;;) {
        const int z = atomic_fetch_add(&J->next, 1); if (z >= f->D) break;
        const int gz = J->fz0 + z; if (gz % st) continue;
        const double pz = (gz + 0.5) * h; const int az = (int)fmin(fmax(pz, 0), na - 1);
        for (int y = 0; y < f->H; y += st) for (int x = 0; x < f->W; x += st) {
            const size_t i = ((size_t)z * f->H + y) * f->W + x, go = ((size_t)gz * J->H + y) * J->W + x;
            if (!f->w[i]) continue;
            const float tzz = f->t[0][i] / 255.f, tyy = f->t[1][i] / 255.f, txx = f->t[2][i] / 255.f, tzy = f->t[3][i] / 255.f - 0.5f, tzx = f->t[4][i] / 255.f - 0.5f, tyx = f->t[5][i] / 255.f - 0.5f;
            if (tzz + tyy + txx < 0.1f) continue;
            ncell++;
            double v[3] = {0.31, 0.57, 0.76};
            for (int it = 0; it < 30; it++) {
                const double u[3] = {tzz * v[0] + tzy * v[1] + tzx * v[2], tzy * v[0] + tyy * v[1] + tyx * v[2], tzx * v[0] + tyx * v[1] + txx * v[2]};
                const double l = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]); if (l < 1e-9) break; for (int d = 0; d < 3; d++) v[d] = u[d] / l;
            }
            const double py = (y + 0.5) * h, px = (x + 0.5) * h, ry = py - f->cy[az], rx = px - f->cx[az], rl = hypot(ry, rx);
            if (rl < 1) continue;
            const double dr = (v[1] * ry + v[2] * rx) / rl; if (fabs(dr) < 0.3) continue;   /* (a normal along the axis: outward undefined) */
            if (dr < 0) for (int d = 0; d < 3; d++) v[d] = -v[d];
            /* the sheet-plane basis */
            double e1[3] = {0, 0, 0}; const int ax = fabs(v[0]) < 0.6 ? 0 : 1; e1[ax] = 1;
            double dt = e1[0] * v[0] + e1[1] * v[1] + e1[2] * v[2]; for (int d = 0; d < 3; d++) e1[d] -= dt * v[d];
            dt = sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]); for (int d = 0; d < 3; d++) e1[d] /= dt;
            const double e2[3] = {v[1] * e1[2] - v[2] * e1[1], v[2] * e1[0] - v[0] * e1[2], v[0] * e1[1] - v[1] * e1[0]};
            /* the profile along v in CT-voxel steps (native step 1 / F), over +-R */
            const double spn = f->sp[i] ? f->sp[i] * h / 8.0 * J->um : 0;
            double Rum = spn > 0 ? 2.5 * spn : J->rmax; Rum = fmin(fmax(Rum, J->rmin), J->rmax);
            int K = (int)(Rum / J->ctum); if (2 * K + 1 > HP_MAXN) K = (HP_MAXN - 1) / 2;
            const int n = 2 * K + 1; const double sn = 1.0 / J->F;   /* native voxels per CT voxel */
            int ok = 1;
            for (int k = 0; k < n && ok; k++) {
                const double t = (k - K) * sn; double s = 0; int cnt = 0;
                for (int j = 0; j < nd; j++) {
                    const double a = da[j] * J->rt * sn, b = db[j] * J->rt * sn;
                    const double p[3] = {pz + t * v[0] + a * e1[0] + b * e2[0], py + t * v[1] + a * e1[1] + b * e2[1], px + t * v[2] + a * e1[2] + b * e2[2]};
                    double c[3]; hp_to_ct(J, p, c); const float sv = hp_sample(J, c[0], c[1], c[2]);
                    if (sv >= 0) { s += sv; cnt++; }
                }
                if (cnt * 2 < nd) { if (abs(k - K) < K / 2) ok = 0; prof[k] = k ? prof[k - 1] : 0; } else prof[k] = (float)(s / cnt);
            }
            if (!ok) continue;
            double mu = 0, s2 = 0;
            for (int k = 0; k < n; k++) {
                double a = 0, ws = 0;
                for (int q = -GK; q <= GK; q++) { const int kk = k + q; if (kk < 0 || kk >= n) continue; a += gw[q + GK] * prof[kk]; ws += gw[q + GK]; }
                sm[k] = (float)(a / ws); mu += sm[k];
            }
            mu /= n; for (int k = 0; k < n; k++) s2 += (sm[k] - mu) * (sm[k] - mu);
            const double sd = sqrt(s2 / n); if (sd < 0.5) continue;
            if (J->gabor > 0) {   /* demodulated at the fields' own sheet spacing (sp: the density head, whose count is unbiased) over +-gabor periods */
                if (!(spn > 0)) continue;
                const double lam = spn / J->ctum; const int G = (int)fmin(K, ceil(J->gabor * lam));
                double complex A = 0; double sw = 0, sa = 0;
                for (int k = -G; k <= G; k++) {
                    const double hw = 0.5 + 0.5 * cos(M_PI * k / (G + 1.0)), dv = sm[K + k] - mu;
                    A += hw * dv * cexp(-I * 2 * M_PI * k / lam); sw += hw; sa += hw * dv * dv;
                }
                if (!(sw > 0) || !(sa > 0)) continue;
                const double amp = 2 * cabs(A) / sw, rms = sqrt(sa / sw), cf = fmin(1, amp / (M_SQRT2 * rms));
                double ph = carg(A) / (2 * M_PI); if (J->feature == 0) ph += 0.5;   /* (a valley at t = 0: cos 2 pi (t / lam + ph) min -> ph = 0.5) */
                ph -= floor(ph);
                for (int a2 = 0; a2 < st && gz + a2 < J->fz0 + f->D; a2++) for (int b2 = 0; b2 < st && y + b2 < f->H; b2++) for (int c2 = 0; c2 < st && x + c2 < f->W; c2++) {
                    const size_t o = ((size_t)(gz + a2) * J->H + y + b2) * J->W + x + c2; J->phi[o] = (float)ph; J->conf[o] = (float)cf;
                }
                nph++; continue;
            }
            const double thr = fmax(J->prom * sd, J->prom_abs);
            const int ne = hp_extrema(sm, n, J->feature, thr, pos, pr);
            int ja = -1; for (int j = 0; j < ne; j++) if (pos[j] <= K) ja = j;
            if (ja < 0 || ja + 1 >= ne) continue;
            const double a = pos[ja] - K, b = pos[ja + 1] - K, L = b - a; if (L < 1) continue;
            for (int j = 0; j + 1 < ne; j++) iv[j] = pos[j + 1] - pos[j];
            double med = L; if (ne >= 3) { qsort(iv, ne - 1, sizeof *iv, hp_cmpd); med = iv[(ne - 1) / 2]; }
            const double lr = log2(L / med), reg = exp(-lr * lr / (2 * 0.5 * 0.5));
            const double cf = tanh(fmin(pr[ja], pr[ja + 1]) / (2 * sd)) * reg;
            J->phi[go] = (float)(-a / L); J->conf[go] = (float)cf; nph++;
            for (int a2 = 0; a2 < st && gz + a2 < J->fz0 + f->D; a2++) for (int b2 = 0; b2 < st && y + b2 < f->H; b2++) for (int c2 = 0; c2 < st && x + c2 < f->W; c2++) {
                const size_t o = ((size_t)(gz + a2) * J->H + y + b2) * J->W + x + c2; J->phi[o] = (float)(-a / L); J->conf[o] = (float)cf;
            }
        }
    }
    atomic_fetch_add(&J->ncell, ncell); atomic_fetch_add(&J->nphase, nph);
    return nullptr;
}

/* the gauge: arg of sum conf_r conf_h exp(i 2 pi (phi_ref - s phi)) over cells where the reference is confident (> cmin); coherence */
/* the same per reference spacing bin (um): phi0 and coherence (does the CT's gauge hold across packing?) */
static void hp_gauge_bins(const fit_fields_buf *R, const float *phi, const float *conf, int s) {
    const double sb[6] = {0, 100, 140, 180, 240, 1e30}; double complex A[5] = {0}; double sw[5] = {0}; size_t nn[5] = {0};
    const size_t n = (size_t)R->shape[0] * R->shape[1] * R->shape[2];
    for (size_t i = 0; i < n; i++) {
        if (!R->f.w[i] || isnan(phi[i]) || !R->f.sp[i]) continue;
        const double c = (R->f.pc[i] - 128) / 127.0, sn = (R->f.ps[i] - 128) / 127.0, cr = hypot(c, sn); if (cr <= 0.3) continue;
        const double sp = R->f.sp[i] * R->h / 8.0 * R->um; int k = 0; while (k < 4 && sp >= sb[k + 1]) k++;
        const double w = cr * conf[i]; A[k] += w * cexp(I * (atan2(sn, c) - s * 2 * M_PI * phi[i])); sw[k] += w; nn[k]++;
    }
    printf("hires-phase: gauge by reference spacing:");
    for (int k = 0; k < 5; k++) { double g = carg(A[k]) / (2 * M_PI); g -= floor(g); printf("  %s%.0f um phi0 %.3f coh %.3f (%zu)", k == 4 ? ">= " : "< ", k == 4 ? sb[4] : sb[k + 1], g, sw[k] > 0 ? cabs(A[k]) / sw[k] : 0, nn[k]); }
    printf("\n");
}
static double hp_gauge(const fit_fields_buf *R, const float *phi, const float *conf, int s, double cmin, double *coh) {
    double complex A = 0; double sw = 0; const size_t n = (size_t)R->shape[0] * R->shape[1] * R->shape[2];
    for (size_t i = 0; i < n; i++) {
        if (!R->f.w[i] || isnan(phi[i])) continue;
        const double c = (R->f.pc[i] - 128) / 127.0, sn = (R->f.ps[i] - 128) / 127.0, cr = hypot(c, sn); if (cr <= cmin) continue;
        const double w = cr * conf[i]; A += w * cexp(I * (atan2(sn, c) - s * 2 * M_PI * phi[i])); sw += w;
    }
    *coh = sw > 0 ? cabs(A) / sw : 0; double g = carg(A) / (2 * M_PI); return g - floor(g);
}

int cmd_hires_phase(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: ufsm hires-phase CTROOT KEY FIELDS-DIR OUT-DIR --ct-um U [--map MAP.json] [--rt-um 15] [--sigma-um 2.4] [--prom 0.5] [--prom-abs 2]\n"
                        "        [--feature valley|peak] [--rmin-um 100] [--rmax-um 250] [--gauge-ref DIR] [--gauge PHI] [--gauge-min 0.4] [--sign 0|1|-1] [--fill net|none]\n"
                        "        [--min-conf 0.05] [--stride 1] [--mem-gb 6] [--threads 16] [--cache DIR] [--dump PREFIX]\n");
        return 2;
    }
    const char *root = argv[2], *key = argv[3], *dir = argv[4], *out = argv[5];
    const double ctum = atof(hp_opt(argc, argv, "--ct-um", "0"));
    if (!(ctum > 0)) { fprintf(stderr, "hires-phase: --ct-um U needed\n"); return 2; }
    const int nt0 = atoi(hp_opt(argc, argv, "--threads", "16")), stride = atoi(hp_opt(argc, argv, "--stride", "1"));
    const int fillnet = strcmp(hp_opt(argc, argv, "--fill", "net"), "none") != 0, feature = !strcmp(hp_opt(argc, argv, "--feature", "valley"), "peak");
    const double mincf = atof(hp_opt(argc, argv, "--min-conf", "0.05")), memgb = atof(hp_opt(argc, argv, "--mem-gb", "6"));
    z3 *cz = z3_open(store_open(root), key, hp_opt(argc, argv, "--cache", nullptr)); if (!cz) { fprintf(stderr, "hires-phase: %s\n", z3_error()); return 1; }
    const z3_meta *zm = z3_meta_of(cz);
    fit_fields_buf g; if (fit_fields_load(dir, 0, 0, 1, &g)) { fprintf(stderr, "hires-phase: cannot load %s\n", dir); return 1; }
    const int D = g.shape[0], H = g.shape[1], W = g.shape[2], h = g.h; const double um = g.um, F = um / ctum;
    const rmap *map = nullptr; const char *mp = hp_opt(argc, argv, "--map", nullptr); double mapS = 1;
    if (mp) {   /* the fine scan registered to the fields' scan (fields at its level 0, KEY = the fine scan's level 0) */
        rvol *fv; double cu, fu; if (!(map = reg_map_open(mp, hp_opt(argc, argv, "--cache", nullptr), &fv, &cu, &fu))) { fprintf(stderr, "hires-phase: cannot open the map %s\n", mp); return 1; } mapS = exp2(lround(log2(um / cu)));
        if (fabs(um / cu - exp2(lround(log2(um / cu)))) > 0.01 * um / cu || fabs(fu - ctum) > 0.01 * ctum) { fprintf(stderr, "hires-phase: the map is %.3f <- %.3f um, the fields %.3f, --ct-um %.3f (KEY must be the fine level 0)\n", cu, fu, um, ctum); return 2; }
    }
    const size_t n = (size_t)D * H * W;
    float *phi = malloc(n * sizeof *phi), *conf = calloc(n, sizeof *conf); for (size_t i = 0; i < n; i++) phi[i] = NAN;
    hp_job J0 = {.F = F, .um = um, .ctum = ctum, .org = {(double)g.origin[0], (double)g.origin[1], (double)g.origin[2]}, .map = map, .ms = mapS,
                 .rt = atof(hp_opt(argc, argv, "--rt-um", "15")) / ctum, .sigma = atof(hp_opt(argc, argv, "--sigma-um", "2.4")) / ctum,
                 .prom = atof(hp_opt(argc, argv, "--prom", "0.5")), .prom_abs = atof(hp_opt(argc, argv, "--prom-abs", "2")),
                 .rmin = atof(hp_opt(argc, argv, "--rmin-um", "100")), .rmax = atof(hp_opt(argc, argv, "--rmax-um", "250")),
                 .gabor = atof(hp_opt(argc, argv, "--gabor", "0")), .feature = feature, .stride = stride < 1 ? 1 : stride, .phi = phi, .conf = conf, .H = H, .W = W};
    const double reach = (J0.rmax / ctum + J0.rt + 3 * J0.sigma) * (map ? 1.05 : 1) + 3;   /* CT voxels */
    /* z slabs of field rows whose CT box fits --mem-gb (the CT box of a slab: the slab's cells mapped to the CT, plus the reach) */
    const double t0 = hp_now(); long long ncell = 0, nph = 0;
    const double cpr = (H * h * F + 2 * reach) * (W * h * F + 2 * reach);   /* CT voxels per CT plane */
    int nzs = (int)((memgb * 1e9 / cpr - 2 * reach) / (h * F)); if (nzs < 1) nzs = 1; nzs -= nzs % J0.stride; if (nzs < J0.stride) nzs = J0.stride;
    for (int fz0 = 0; fz0 < D; fz0 += nzs) {
        const int fz1 = fz0 + nzs < D ? fz0 + nzs : D;
        fit_fields_buf b; if (fit_fields_load(dir, fz0, fz1, 1, &b)) { fprintf(stderr, "hires-phase: cannot load rows %d..%d\n", fz0, fz1); return 1; }
        hp_job *J = malloc(sizeof *J); *J = J0; J->f = &b.f; J->fz0 = fz0; atomic_init(&J->next, 0); atomic_init(&J->ncell, 0); atomic_init(&J->nphase, 0);
        /* the CT box: the corners of the slab's native box mapped (8 corners; a registration map is smooth on this scale) */
        double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
        for (int c = 0; c < 8; c++) {
            const double p[3] = {(c & 4 ? fz1 : fz0) * (double)h, (c & 2 ? H : 0) * (double)h, (c & 1 ? W : 0) * (double)h}; double q[3]; hp_to_ct(J, p, q);
            for (int d = 0; d < 3; d++) { lo[d] = fmin(lo[d], q[d]); hi[d] = fmax(hi[d], q[d]); }
        }
        int ok = 1;
        for (int d = 0; d < 3; d++) { int64_t a = (int64_t)floor(lo[d] - reach), e = (int64_t)ceil(hi[d] + reach) + 1; if (a < 0) a = 0; if (e > zm->shape[d]) e = zm->shape[d]; if (e <= a) ok = 0; J->vo[d] = a; J->vn[d] = e - a; }
        uint8_t *vol = ok ? malloc((size_t)J->vn[0] * J->vn[1] * J->vn[2]) : nullptr;
        if (ok && (!vol || z3_read(cz, J->vo, J->vn, vol, nt0))) { fprintf(stderr, "hires-phase: CT read failed: %s\n", z3_error()); return 1; }
        if (ok) {
            J->vol = vol; const int nt = nt0 < 64 ? (nt0 > 0 ? nt0 : 1) : 64; pthread_t th[64]; int up[64] = {0};
            for (int t = 1; t < nt; t++) up[t] = pthread_create(&th[t], nullptr, hp_rows, J) == 0;
            hp_rows(J);
            for (int t = 1; t < nt; t++) if (up[t]) pthread_join(th[t], nullptr);
            ncell += J->ncell; nph += J->nphase;
        }
        free(vol); free(J); fit_fields_free(&b);
        fprintf(stderr, "\rhires-phase: rows %d / %d (%.0f s)", fz1, D, hp_now() - t0);
    }
    fprintf(stderr, "\n");
    /* the sign and gauge */
    double cp, cm; const double gmin = atof(hp_opt(argc, argv, "--gauge-min", "0.4"));
    const double gp = hp_gauge(&g, phi, conf, 1, gmin, &cp), gm = hp_gauge(&g, phi, conf, -1, gmin, &cm);
    printf("hires-phase: gauge vs the input fields (confidence > %.2f): sign +1 phi0 %.3f coherence %.3f; sign -1 phi0 %.3f coherence %.3f\n", gmin, gp, cp, gm, cm);
    int sign = atoi(hp_opt(argc, argv, "--sign", "1")); double g0 = sign < 0 ? gm : gp;
    const char *gr = hp_opt(argc, argv, "--gauge-ref", nullptr);
    if (gr) {
        fit_fields_buf R; if (fit_fields_load(gr, 0, 0, 1, &R)) { fprintf(stderr, "hires-phase: cannot load %s\n", gr); return 1; }
        if (R.shape[0] != D || R.shape[1] != H || R.shape[2] != W) { fprintf(stderr, "hires-phase: --gauge-ref has another shape\n"); return 2; }
        double rp, rm; const double qp = hp_gauge(&R, phi, conf, 1, 0.3, &rp), qm = hp_gauge(&R, phi, conf, -1, 0.3, &rm);
        printf("hires-phase: gauge vs %s: sign +1 phi0 %.3f coherence %.3f; sign -1 phi0 %.3f coherence %.3f\n", gr, qp, rp, qm, rm);
        if (!sign) sign = rp >= rm ? 1 : -1;
        hp_gauge_bins(&R, phi, conf, sign);
        g0 = sign < 0 ? qm : qp; fit_fields_free(&R);
    } else if (!sign) { sign = cp >= cm ? 1 : -1; g0 = sign < 0 ? gm : gp; }
    if (*hp_opt(argc, argv, "--gauge", "")) g0 = atof(hp_opt(argc, argv, "--gauge", "0"));
    size_t nmat = 0, nset = 0; double csum = 0;
    for (size_t i = 0; i < n; i++) {
        if (g.f.w[i]) nmat++;
        if (!isnan(phi[i]) && conf[i] >= mincf) {
            const double ph = 2 * M_PI * (sign * phi[i] + g0), c = fmin(1, conf[i]);
            g.a[0][i] = (uint8_t)lround(128 + 127 * c * cos(ph)); g.a[1][i] = (uint8_t)lround(128 + 127 * c * sin(ph));
            if (g.f.w[i]) { nset++; csum += c; }
        } else if (!fillnet) g.a[0][i] = g.a[1][i] = 128;
    }
    const char *dump = hp_opt(argc, argv, "--dump", nullptr);
    if (dump) {   /* raw float phi (NaN none) and conf on the field grid */
        char p[1400]; snprintf(p, sizeof p, "%s.phi.f32", dump); FILE *fo = fopen(p, "wb"); if (fo) { fwrite(phi, sizeof *phi, n, fo); fclose(fo); }
        snprintf(p, sizeof p, "%s.conf.f32", dump); fo = fopen(p, "wb"); if (fo) { fwrite(conf, sizeof *conf, n, fo); fclose(fo); }
    }
    /* --labels ROOT --labels-global Z,Y,X [--labels-conf 0.3]: phase training targets (src/band.h PHASE_* codes, 255 unknown) on the field-cell
       grid (h native voxels) of a global array of that shape, 512^3 shards; the box origin must sit on the 512 h grid. Only cells with a CT
       phase of confidence >= --labels-conf and material (w >= --labels-mat x 255). Several boxes fill one array (one shard row each). */
    const char *lroot = hp_opt(argc, argv, "--labels", nullptr);
    if (lroot) {
        long long gv[3]; if (sscanf(hp_opt(argc, argv, "--labels-global", ""), "%lld,%lld,%lld", &gv[0], &gv[1], &gv[2]) != 3) { fprintf(stderr, "hires-phase: --labels needs --labels-global Z,Y,X\n"); return 2; }
        const double lc = atof(hp_opt(argc, argv, "--labels-conf", "0.3")), lm = atof(hp_opt(argc, argv, "--labels-mat", "0.3")); const int BK = 512;
        int64_t so[3]; const int64_t gsh[3] = {gv[0], gv[1], gv[2]};
        for (int d = 0; d < 3; d++) { if (org[d] < 0 || org[d] % ((long long)BK * h)) { fprintf(stderr, "hires-phase: --labels: box origin %lld not on the %d x h grid\n", org[d], BK); return 2; } so[d] = org[d] / ((long long)BK * h); }
        char lv[32], ld[1400], at[400]; z3w_level_name(um * h, lv, sizeof lv); snprintf(ld, sizeof ld, "%s/%s", lroot, lv);
        snprintf(at, sizeof at, "{\"ufsm\":{\"content\":\"stored phase from hires-phase (CT %s at %.3f um, gabor %g, conf >= %g): round(252 frac(phase)) mod 252 (uint8; 255 unknown; src/band.h PHASE_*)\"}}", key, ctum, J0.gabor, lc);
        z3w *zw = z3w_create(ld, gsh, BK, 0.f, 255, at);
        if (!zw || z3w_write_group(lroot, (const double[]){um * h}, 1, "ufsm-phase", at)) { fprintf(stderr, "hires-phase: %s\n", z3w_error()); return 1; }
        uint8_t *sb = malloc((size_t)BK * BK * BK); long long nl = 0;
        for (int64_t sz = 0; sz * BK < D; sz++) for (int64_t sy = 0; sy * BK < H; sy++) for (int64_t sx = 0; sx * BK < W; sx++) {
            memset(sb, 255, (size_t)BK * BK * BK); long long k = 0;
            for (int a = 0; a < BK && sz * BK + a < D; a++) for (int b = 0; b < BK && sy * BK + b < H; b++) for (int c = 0; c < BK && sx * BK + c < W; c++) {
                const size_t i = ((size_t)(sz * BK + a) * H + (sy * BK + b)) * W + (sx * BK + c);
                if (isnan(phi[i]) || conf[i] < lc || g.f.w[i] < lm * 255) continue;
                sb[((size_t)a * BK + b) * BK + c] = fit_phase_code(sign * phi[i] + g0, 1); k++;
            }
            if (k && z3w_write_shard(zw, sz + so[0], sy + so[1], sx + so[2], sb, 8)) { fprintf(stderr, "hires-phase: %s\n", z3w_error()); return 1; }
            nl += k;
        }
        z3w_close(zw); free(sb);
        printf("hires-phase: labels: %lld cells (%.3f of material) -> %s\n", nl, nmat ? (double)nl / nmat : 0, ld);
        if (!strcmp(out, "-")) { free(phi); free(conf); fit_fields_free(&g); z3_close(cz); return 0; }
    }
    uint8_t *a10[10]; for (int k = 0; k < 10; k++) a10[k] = g.a[k];
    const long long org[3] = {g.origin[0], g.origin[1], g.origin[2]}; const int shp[3] = {D, H, W};
    if (fit_fields_save(out, org, shp, h, um, a10, g.cy, g.cx, (int)(D * h) + 1)) { fprintf(stderr, "hires-phase: cannot write %s\n", out); return 1; }
    printf("hires-phase: %dx%dx%d cells (h %d, %.3f um), CT %s at %.3f um%s; %lld cells probed, %lld with a CT phase; set on %zu of %zu material cells (%.3f, mean conf %.3f), sign %+d gauge %.3f, rest %s (%.0f s) -> %s\n",
           D, H, W, h, um, key, ctum, map ? " (registered)" : "", ncell, nph, nset, nmat, nmat ? (double)nset / nmat : 0, nset ? csum / nset : 0, sign, g0,
           fillnet ? "network" : "unknown", hp_now() - t0, out);
    free(phi); free(conf); fit_fields_free(&g); z3_close(cz);
    return 0;
}

/* ufsm hires-count CTROOT KEY REF-FIELDS --ct-um U [--map MAP] [--rt-um 15] [--sigma-um 2.4] [--prom 0.5] [--prom-abs 2] [--feature valley|peak]
   [--margin-um 200] [--threads 16] [--cache DIR] [--mem-gb 6]
   Can the finer CT count the sheets? fit-compare's turn-count segments (0.5 mm radial, in z planes, from every 4th reference cell whose
   steps all carry reference phase; the reference turns = its phase unwrapped along the segment): the CT along the same line (disc
   averaged in the plane of the reference fields' normal at the start cell, smoothed), its valleys (or peaks) with prominence as
   hires-phase, and the CT turns = the valley index interpolated at the segment's end minus at its start (valleys beyond both ends needed:
   --margin-um). Error = CT - reference turns: median, share >= 0.5 turn off, on packed (< 100 um) segments too. */
typedef struct {
    const fit_fields_buf *A; const hp_job *P; int z0, z1;   /* segment start rows [z0, z1) */
    double *err; uint8_t *pk; _Atomic int next; _Atomic size_t nseg, nnoct;
} hc_job;
static void *hc_rows(void *arg) {
    hc_job *C = arg; const fit_fields_buf *A = C->A; const hp_job *J = C->P; const fit_fields *f = &A->f;
    const double h = A->h, um = A->um, L = 500.0, ds = 0.5 * h * um; const int ns = (int)(L / ds), na = (int)(A->shape[0] * h) + 1;
    static thread_local float prof[4096], sm[4096]; static thread_local double pos[512];
    const int GK = (int)fmin(31, ceil(3 * J->sigma)); float gw[64]; double gs = 0;
    for (int k = -GK; k <= GK; k++) { gw[k + GK] = J->sigma > 0 ? (float)exp(-0.5 * k * k / (J->sigma * J->sigma)) : (k == 0); gs += gw[k + GK]; }
    for (int k = 0; k <= 2 * GK; k++) gw[k] /= (float)gs;
    double da[19], db[19]; int nd = 0; da[nd] = 0; db[nd++] = 0;
    if (J->rt > 0) {
        for (int k = 0; k < 6; k++) { da[nd] = 0.5 * cos(k * M_PI / 3); db[nd++] = 0.5 * sin(k * M_PI / 3); }
        for (int k = 0; k < 12; k++) { da[nd] = cos(k * M_PI / 6 + M_PI / 12); db[nd++] = sin(k * M_PI / 6 + M_PI / 12); }
    }
    const int H = A->shape[1], W = A->shape[2];
    for (;;) {
        const int zi = atomic_fetch_add(&C->next, 1), z = C->z0 + 4 * zi; if (z >= C->z1) break;
        for (int y = 0; y < H; y += 4) for (int x = 0; x < W; x += 4) {
            const double nz = h * (z + 0.5); const int az = (int)fmin(fmax(nz, 0), na - 1); const double cyv = f->cy[az], cxv = f->cx[az];
            const double py = h * (y + 0.5), px = h * (x + 0.5); double ry = py - cyv, rx = px - cxv; const double rl = hypot(ry, rx); if (rl < 1) continue; ry /= rl; rx /= rl;
            double ta = 0, la = 0, spsum = 0; int ok = 1;
            for (int k = 0; k <= ns && ok; k++) {   /* (exactly fit-compare's reference count) */
                const double qy = py + ry * k * ds / um, qx = px + rx * k * ds / um; const int iy = (int)(qy / h), ix = (int)(qx / h);
                if (iy < 0 || ix < 0 || iy >= H || ix >= W) { ok = 0; break; }
                const size_t i = ((size_t)z * H + iy) * W + ix; const double ac = (f->pc[i] - 128) / 127.0, as = (f->ps[i] - 128) / 127.0;
                if (!f->w[i] || hypot(ac, as) <= 0.3) { ok = 0; break; }
                const double pa = atan2(as, ac) / (2 * M_PI); spsum += f->sp[i] * h / 8.0;
                if (k) { double d = pa - la; d -= round(d); ta += d; }
                la = pa;
            }
            if (!ok) continue;
            const double ref = fabs(ta); if (ref < 0.5) continue;
            const size_t si = atomic_fetch_add(&C->nseg, 1);
            /* the disc plane: the reference fields' normal at the start cell */
            const size_t i0 = ((size_t)z * H + y) * W + x;
            const float tzz = f->t[0][i0] / 255.f, tyy = f->t[1][i0] / 255.f, txx = f->t[2][i0] / 255.f, tzy = f->t[3][i0] / 255.f - 0.5f, tzx = f->t[4][i0] / 255.f - 0.5f, tyx = f->t[5][i0] / 255.f - 0.5f;
            double v[3] = {0, ry, rx};
            if (tzz + tyy + txx > 0.1f) { v[0] = 0.31; v[1] = 0.57; v[2] = 0.76;
                for (int it = 0; it < 30; it++) { const double u[3] = {tzz * v[0] + tzy * v[1] + tzx * v[2], tzy * v[0] + tyy * v[1] + tyx * v[2], tzx * v[0] + tyx * v[1] + txx * v[2]};
                    const double l = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]); if (l < 1e-9) break; for (int d = 0; d < 3; d++) v[d] = u[d] / l; } }
            double e1[3] = {0, 0, 0}; e1[fabs(v[0]) < 0.6 ? 0 : 1] = 1;
            double dt = e1[0] * v[0] + e1[1] * v[1] + e1[2] * v[2]; for (int d = 0; d < 3; d++) e1[d] -= dt * v[d];
            dt = sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]); for (int d = 0; d < 3; d++) e1[d] /= dt;
            const double e2[3] = {v[1] * e1[2] - v[2] * e1[1], v[2] * e1[0] - v[0] * e1[2], v[0] * e1[1] - v[1] * e1[0]};
            /* the CT along the radial line from -margin to L + margin, one CT voxel per step */
            const double sn = 1.0 / J->F; const int M = (int)(J->rmax / J->ctum), N = (int)(L / J->ctum), n = N + 2 * M + 1;
            if (n > 4096) { C->err[si] = NAN; continue; }
            int bad = 0;
            for (int k = 0; k < n; k++) {
                const double t = (k - M) * sn; double s = 0; int cnt = 0;
                for (int j = 0; j < nd; j++) {
                    const double a = da[j] * J->rt * sn, b = db[j] * J->rt * sn;
                    const double p[3] = {nz + a * e1[0] + b * e2[0], py + t * ry + a * e1[1] + b * e2[1], px + t * rx + a * e1[2] + b * e2[2]};
                    double c[3]; hp_to_ct(J, p, c); const float sv = hp_sample(J, c[0], c[1], c[2]); if (sv >= 0) { s += sv; cnt++; }
                }
                if (cnt * 2 < nd) { bad = 1; break; }
                prof[k] = (float)(s / cnt);
            }
            if (bad) { C->err[si] = NAN; atomic_fetch_add(&C->nnoct, 1); continue; }
            double mu = 0, s2 = 0;
            for (int k = 0; k < n; k++) { double a = 0, w = 0; for (int q = -GK; q <= GK; q++) { const int kk = k + q; if (kk < 0 || kk >= n) continue; a += gw[q + GK] * prof[kk]; w += gw[q + GK]; } sm[k] = (float)(a / w); mu += sm[k]; }
            mu /= n; for (int k = 0; k < n; k++) s2 += (sm[k] - mu) * (sm[k] - mu);
            const double sd = sqrt(s2 / n);
            int ne = 0;
            {   /* extrema in windows of <= 1024 (hp_extrema's limit), overlapping by 64 */
                for (int w0 = 0; w0 < n; w0 += 1024 - 128) {
                    const int wn = n - w0 < 1024 ? n - w0 : 1024; double p2[HP_MAXM], q2[HP_MAXM];
                    const int m = hp_extrema(sm + w0, wn, J->feature, fmax(J->prom * sd, J->prom_abs), p2, q2);
                    for (int j = 0; j < m; j++) { const double ps = p2[j] + w0; if ((w0 && p2[j] < 64) || (w0 + wn < n && p2[j] >= wn - 64)) continue; if (ne < 512) pos[ne++] = ps; }
                    if (w0 + wn >= n) break;
                }
            }
            /* the valley index at t = 0 (k = M) and t = L (k = M + N) */
            double idx[2]; int okk = 1;
            for (int e = 0; e < 2 && okk; e++) {
                const double t = M + (e ? N : 0); int j = -1; for (int q = 0; q < ne; q++) if (pos[q] <= t) j = q;
                if (j < 0 || j + 1 >= ne) { okk = 0; break; }
                idx[e] = j + (t - pos[j]) / (pos[j + 1] - pos[j]);
            }
            C->err[si] = okk ? (idx[1] - idx[0]) - ref : NAN; C->pk[si] = spsum / (ns + 1) * um < 100;
            if (!okk) atomic_fetch_add(&C->nnoct, 1);
        }
    }
    return nullptr;
}
int cmd_hires_count(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm hires-count CTROOT KEY REF-FIELDS --ct-um U [--map MAP.json] [--rt-um 15] [--sigma-um 2.4] [--prom 0.5] [--prom-abs 2] [--feature valley|peak] [--margin-um 200] [--threads 16] [--cache DIR] [--mem-gb 6]\n"); return 2; }
    const char *root = argv[2], *key = argv[3], *dir = argv[4];
    const double ctum = atof(hp_opt(argc, argv, "--ct-um", "0")); if (!(ctum > 0)) { fprintf(stderr, "hires-count: --ct-um U needed\n"); return 2; }
    const int nt0 = atoi(hp_opt(argc, argv, "--threads", "16"));
    z3 *cz = z3_open(store_open(root), key, hp_opt(argc, argv, "--cache", nullptr)); if (!cz) { fprintf(stderr, "hires-count: %s\n", z3_error()); return 1; }
    const z3_meta *zm = z3_meta_of(cz);
    fit_fields_buf A; if (fit_fields_load(dir, 0, 0, 1, &A)) { fprintf(stderr, "hires-count: cannot load %s\n", dir); return 1; }
    const int D = A.shape[0], H = A.shape[1], W = A.shape[2], h = A.h; const double um = A.um, F = um / ctum;
    const rmap *map = nullptr; const char *mp = hp_opt(argc, argv, "--map", nullptr); double mapS = 1;
    if (mp) { rvol *fv; double cu, fu; if (!(map = reg_map_open(mp, hp_opt(argc, argv, "--cache", nullptr), &fv, &cu, &fu))) { fprintf(stderr, "hires-count: cannot open the map %s\n", mp); return 1; } mapS = exp2(lround(log2(um / cu)));
              if (fabs(um / cu - exp2(lround(log2(um / cu)))) > 0.01 * um / cu || fabs(fu - ctum) > 0.01 * ctum) { fprintf(stderr, "hires-count: map %.3f <- %.3f um vs fields %.3f / --ct-um %.3f\n", cu, fu, um, ctum); return 2; } }
    hp_job P = {.F = F, .um = um, .ctum = ctum, .org = {(double)A.origin[0], (double)A.origin[1], (double)A.origin[2]}, .map = map, .ms = mapS,
                .rt = atof(hp_opt(argc, argv, "--rt-um", "15")) / ctum, .sigma = atof(hp_opt(argc, argv, "--sigma-um", "2.4")) / ctum,
                .prom = atof(hp_opt(argc, argv, "--prom", "0.5")), .prom_abs = atof(hp_opt(argc, argv, "--prom-abs", "2")),
                .rmax = atof(hp_opt(argc, argv, "--margin-um", "200")), .feature = !strcmp(hp_opt(argc, argv, "--feature", "valley"), "peak")};
    const size_t maxseg = (size_t)((D + 3) / 4) * ((H + 3) / 4) * ((W + 3) / 4);
    double *err = malloc(maxseg * sizeof *err); uint8_t *pk = calloc(maxseg, 1);
    const double reachxy = (500 + P.rmax) / ctum + P.rt + 3 * P.sigma + 3, reachz = (P.rt + 3 * P.sigma) * (map ? 1.1 : 1) + 3;
    const double cpr = (H * h * F + 2 * reachxy) * (W * h * F + 2 * reachxy), memgb = atof(hp_opt(argc, argv, "--mem-gb", "6"));
    int nzs = (int)((memgb * 1e9 / cpr - 2 * reachz) / (h * F)); nzs -= nzs % 4; if (nzs < 4) nzs = 4;
    hc_job C = {.A = &A, .P = &P, .err = err, .pk = pk}; atomic_init(&C.nseg, 0); atomic_init(&C.nnoct, 0);
    const double t0 = hp_now();
    for (int fz0 = 0; fz0 < D; fz0 += nzs) {
        const int fz1 = fz0 + nzs < D ? fz0 + nzs : D;
        double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
        for (int c = 0; c < 8; c++) {
            const double p[3] = {(c & 4 ? fz1 : fz0) * (double)h, (c & 2 ? H : 0) * (double)h, (c & 1 ? W : 0) * (double)h}; double q[3]; hp_to_ct(&P, p, q);
            for (int d = 0; d < 3; d++) { lo[d] = fmin(lo[d], q[d]); hi[d] = fmax(hi[d], q[d]); }
        }
        int ok = 1;
        for (int d = 0; d < 3; d++) { const double r = d ? reachxy : reachz + (map ? reachxy * 0.02 : 0); int64_t a = (int64_t)floor(lo[d] - r), e = (int64_t)ceil(hi[d] + r) + 1; if (a < 0) a = 0; if (e > zm->shape[d]) e = zm->shape[d]; if (e <= a) ok = 0; P.vo[d] = a; P.vn[d] = e - a; }
        uint8_t *vol = ok ? malloc((size_t)P.vn[0] * P.vn[1] * P.vn[2]) : nullptr;
        if (ok && (!vol || z3_read(cz, P.vo, P.vn, vol, nt0))) { fprintf(stderr, "hires-count: CT read failed: %s\n", z3_error()); return 1; }
        if (ok) {
            P.vol = vol; C.z0 = fz0 + (4 - fz0 % 4) % 4; C.z1 = fz1; atomic_store(&C.next, 0);
            const int nt = nt0 < 64 ? (nt0 > 0 ? nt0 : 1) : 64; pthread_t th[64]; int up[64] = {0};
            for (int t = 1; t < nt; t++) up[t] = pthread_create(&th[t], nullptr, hc_rows, &C) == 0;
            hc_rows(&C);
            for (int t = 1; t < nt; t++) if (up[t]) pthread_join(th[t], nullptr);
        }
        free(vol);
        fprintf(stderr, "\rhires-count: rows %d / %d (%.0f s)", fz1, D, hp_now() - t0);
    }
    fprintf(stderr, "\n");
    const size_t ns = atomic_load(&C.nseg); size_t m = 0, off = 0, pn = 0, poff = 0, pm = 0; double sum = 0; double *v = malloc((ns + 1) * sizeof *v), *vp = malloc((ns + 1) * sizeof *vp);
    for (size_t i = 0; i < ns; i++) { if (isnan(err[i])) continue; v[m++] = err[i]; sum += err[i]; off += fabs(err[i]) >= 0.5; if (pk[i]) { vp[pm++] = err[i]; pn++; poff += fabs(err[i]) >= 0.5; } }
    qsort(v, m, sizeof *v, hp_cmpd); qsort(vp, pm, sizeof *vp, hp_cmpd);
    printf("hires-count: %zu segments (%zu without a CT count), CT %s at %.3f um%s, rt %.1f um sigma %.1f um prom %.2f (abs %.1f) %s\n", ns, (size_t)atomic_load(&C.nnoct), key, ctum, map ? " (registered)" : "",
           P.rt * ctum, P.sigma * ctum, P.prom, P.prom_abs, P.feature ? "peaks" : "valleys");
    if (m) printf("    by CT:      error median %+.2f, mean %+.2f turns, off by >= 0.5 turn %.3f (packed < 100 um: %.3f of %zu, median %+.2f)  [%.0f s]\n",
                  v[m / 2], sum / m, (double)off / m, pn ? (double)poff / pn : 0, pn, pm ? vp[pm / 2] : 0, hp_now() - t0);
    free(v); free(vp); free(err); free(pk); fit_fields_free(&A); z3_close(cz);
    return 0;
}
