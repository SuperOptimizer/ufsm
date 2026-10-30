#include "deform.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t df_next(uint64_t *s) { uint64_t z = (*s += 0x9e3779b97f4a7c15ull); z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull; return z ^ (z >> 31); }
static double df_u(uint64_t *s) { return (double)(df_next(s) >> 11) * 0x1.0p-53; }

deform_cfg deform_cfg_default(void) {
    deform_cfg c = {.crush_p = .5f, .crush = crush_cfg_default(), .elastic_p = .5f, .amp = 8, .lam_lo = 96, .lam_hi = 384, .lip = .45f};
    c.crush.jlo = .05f; c.crush.jhi = .4f; c.crush.loc_amp = 1; c.crush.loc_len = 40; c.crush.loc_jmax = 1.5f;
    c.crush.bend_p = .7f; c.crush.bend_amp = .1f; c.crush.kink_p = .4f; c.crush.klo = 1.3f; c.crush.khi = 2.5f; c.crush.rotate_deg = 0;
    return c;
}
static const char *df_opt(int argc, char **argv, const char *f) { for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], f)) return argv[i + 1]; return nullptr; }
int deform_cfg_parse(deform_cfg *c, int argc, char **argv) {
    if (crush_cfg_parse(&c->crush, argc, argv)) return 1;
    c->crush.rotate_deg = 0;
    const char *v;
    if ((v = df_opt(argc, argv, "--deform-crush"))) c->crush_p = (float)atof(v);
    if ((v = df_opt(argc, argv, "--deform-elastic"))) c->elastic_p = (float)atof(v);
    if ((v = df_opt(argc, argv, "--deform-amp"))) c->amp = (float)atof(v);
    if ((v = df_opt(argc, argv, "--deform-lip"))) c->lip = (float)atof(v);
    if ((v = df_opt(argc, argv, "--deform-lam")) && sscanf(v, "%f,%f", &c->lam_lo, &c->lam_hi) != 2) { fprintf(stderr, "--deform-lam LO,HI\n"); return 1; }
    if (!(c->crush_p >= 0 && c->crush_p <= 1) || !(c->elastic_p >= 0 && c->elastic_p <= 1) || !(c->amp >= 0 && c->amp <= 64) ||
        !(c->lip > 0 && c->lip < .9f) || !(c->lam_lo >= 8 && c->lam_lo <= c->lam_hi)) {
        fprintf(stderr, "--deform-crush / --deform-elastic: 0..1, --deform-amp 0..64, --deform-lip 0..0.9, --deform-lam LO,HI (8 <= LO <= HI)\n"); return 1;
    }
    return 0;
}
void deform_cfg_print(const deform_cfg *c) {
    fprintf(stderr, "deform: crush p %.3g, elastic p %.3g (amp %.3g vx, wavelength %.3g..%.3g, Lipschitz %.3g); ", c->crush_p, c->elastic_p, c->amp, c->lam_lo, c->lam_hi, c->lip);
    crush_cfg_print(&c->crush);
}

int deform_make(deform *D, const deform_cfg *cfg, uint64_t seed, const double c[3], const double n[3], int P, crush_ct_fn ct, void *ud) {
    return deform_make_ext(D, cfg, seed, c, n, P, nullptr, 0, ct, ud);
}
int deform_make_ext(deform *D, const deform_cfg *cfg, uint64_t seed, const double c[3], const double n[3], int P, const double (*pts)[3], int np, crush_ct_fn ct, void *ud) {
    memset(D, 0, sizeof *D); D->P = P;
    for (int d = 0; d < 3; d++) D->o[d] = c[d] - .5 * (P - 1);
    if (cfg->elastic_p > 0 && df_u(&seed) < cfg->elastic_p && cfg->amp > 0) {
        D->ne = 6; double lip = 0;
        for (int k = 0; k < D->ne; k++) {
            const double lam = exp(log(cfg->lam_lo) + df_u(&seed) * (log(cfg->lam_hi) - log(cfg->lam_lo))), kk = 2 * M_PI / lam;
            const double cz = 2 * df_u(&seed) - 1, ph = 2 * M_PI * df_u(&seed), sz = sqrt(fmax(0, 1 - cz * cz));
            D->ek[k][0] = kk * cz; D->ek[k][1] = kk * sz * cos(ph); D->ek[k][2] = kk * sz * sin(ph); D->eph[k] = 2 * M_PI * df_u(&seed);
            double a[3]; for (int d = 0; d < 3; d++) a[d] = 2 * df_u(&seed) - 1;
            const double amp = cfg->amp * df_u(&seed) / sqrt(D->ne), na = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) + 1e-12;
            for (int d = 0; d < 3; d++) D->eA[k][d] = amp * a[d] / na;
            lip += amp * kk;
        }
        if (lip > cfg->lip) for (int k = 0; k < D->ne; k++) for (int d = 0; d < 3; d++) D->eA[k][d] *= cfg->lip / lip;   /* |grad e| < lip < 1 */
    }
    if (cfg->crush_p > 0 && df_u(&seed) < cfg->crush_p) {
        crush_cfg cc = cfg->crush; cc.rotate_deg = 0;
        crush_map_init(&D->m, &cc, df_next(&seed), P, n);
        if (pts && np > 0 && np <= 64) {   /* each point expanded by the elastic stage's bound (+2) along every axis: the hull of the cubes */
            double ea = 2; for (int k = 0; k < D->ne; k++) for (int d = 0; d < 3; d++) ea += fabs(D->eA[k][d]);
            double R[512][3]; int m = 0;
            for (int i = 0; i < np; i++) for (int c8 = 0; c8 < 8; c8++, m++) for (int d = 0; d < 3; d++) R[m][d] = pts[i][d] + ((c8 >> (2 - d) & 1) ? ea : -ea);
            crush_map_extent_pts(&D->m, (const double (*)[3])R, m);
        }
        if (ct) crush_map_columns(&D->m, cc.jlo, cc.jhi, df_next(&seed), ct, ud);   /* (1: too little CT for a split: no collapse, A / B stay) */
        D->ncr = 1;
    }
    return 0;
}
void deform_free(deform *D) { if (D->ncr) crush_map_free(&D->m); D->ncr = 0; }
int deform_on(const deform *D) { return D->ncr || D->ne; }

static void df_elastic(const deform *D, const double u[3], double v[3], double J[3][3]) {
    for (int d = 0; d < 3; d++) { v[d] = u[d]; if (J) for (int e = 0; e < 3; e++) J[d][e] = d == e; }
    for (int k = 0; k < D->ne; k++) {
        const double t = D->ek[k][0] * u[0] + D->ek[k][1] * u[1] + D->ek[k][2] * u[2] + D->eph[k], s = sin(t), c = cos(t);
        for (int d = 0; d < 3; d++) { v[d] += D->eA[k][d] * s; if (J) for (int e = 0; e < 3; e++) J[d][e] += D->eA[k][d] * c * D->ek[k][e]; }
    }
}
static double det3(double J[3][3]) { return J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]); }
void deform_pull(const deform *D, const double out[3], double src[3], double *jdet) {
    double u[3], v[3], Je[3][3], jc = 1;
    for (int d = 0; d < 3; d++) u[d] = out[d] - D->o[d];   /* (window coordinates; E is centred anywhere: a translation-free sine field) */
    df_elastic(D, u, v, jdet ? Je : nullptr);
    if (D->ncr) { double x[3]; crush_pull(&D->m, v, x, jdet ? &jc : nullptr); for (int d = 0; d < 3; d++) v[d] = x[d]; }
    for (int d = 0; d < 3; d++) src[d] = v[d] + D->o[d];
    if (jdet) *jdet = jc * det3(Je);
}
void deform_jacobian(const deform *D, const double out[3], double J[3][3]) {
    const double h = 1e-3;
    for (int j = 0; j < 3; j++) {
        double up[3] = {out[0], out[1], out[2]}, um[3] = {out[0], out[1], out[2]}, xp[3], xm[3];
        up[j] += h; um[j] -= h; deform_pull(D, up, xp, nullptr); deform_pull(D, um, xm, nullptr);
        for (int i = 0; i < 3; i++) J[i][j] = (xp[i] - xm[i]) / (2 * h);
    }
}
void deform_bounds(const deform *D, const double ext[3], double lo[3], double hi[3]) {
    const double c[3] = {D->o[0] + .5 * (D->P - 1), D->o[1] + .5 * (D->P - 1), D->o[2] + .5 * (D->P - 1)};
    for (int d = 0; d < 3; d++) { lo[d] = 1e300; hi[d] = -1e300; }
    const int S = 16;
    for (int f = 0; f < 6; f++) for (int a = 0; a <= S; a++) for (int b = 0; b <= S; b++) {
        const int ax = f / 2; double p[3]; int q = 0;
        for (int d = 0; d < 3; d++) {
            if (d == ax) p[d] = c[d] + (f & 1 ? ext[d] : -ext[d]);
            else { const double t = (q++ ? b : a) / (double)S; p[d] = c[d] - ext[d] + 2 * ext[d] * t; }
        }
        double s[3]; deform_pull(D, p, s, nullptr);
        for (int d = 0; d < 3; d++) { lo[d] = fmin(lo[d], s[d] - 3); hi[d] = fmax(hi[d], s[d] + 3); }
    }
}
