#include "crush.h"
#include "band.h"
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t cr_next(uint64_t *s) { uint64_t z = (*s += 0x9e3779b97f4a7c15ull); z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull; return z ^ (z >> 31); }
static double cr_u(uint64_t *s) { return (double)(cr_next(s) >> 11) * 0x1.0p-53; }
static double cr_logu(uint64_t *s, double lo, double hi) { return exp(log(lo) + cr_u(s) * (log(hi) - log(lo))); }

crush_cfg crush_cfg_default(void) {
    return (crush_cfg){.p = 0, .jlo = .15f, .jhi = .5f, .squash_p = .5f, .klo = 1.5f, .khi = 3.f, .bend_p = .5f, .bend_amp = .06f, .kink_p = .25f, .rotate_deg = 10.f, .fine = 0};
}

static const char *cr_opt(int argc, char **argv, const char *f) { for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], f)) return argv[i + 1]; return nullptr; }
int crush_cfg_parse(crush_cfg *c, int argc, char **argv) {
    const char *v;
#define CR_F(flag, dst) if ((v = cr_opt(argc, argv, flag))) dst = (float)atof(v)
#define CR_R(flag, a, b) if ((v = cr_opt(argc, argv, flag)) && sscanf(v, "%f,%f", &(a), &(b)) != 2) { fprintf(stderr, "%s LO,HI\n", flag); return 1; }
    CR_F("--aug-crush", c->p); CR_R("--aug-crush-void", c->jlo, c->jhi); CR_F("--aug-crush-squash", c->squash_p); CR_R("--aug-crush-k", c->klo, c->khi);
    CR_F("--aug-crush-bend", c->bend_p); CR_F("--aug-crush-bend-amp", c->bend_amp); CR_F("--aug-crush-kink", c->kink_p); CR_F("--aug-crush-rot", c->rotate_deg);
    if ((v = cr_opt(argc, argv, "--aug-crush-fine"))) c->fine = atoi(v);
#undef CR_F
#undef CR_R
    const int bad = !(c->p >= 0 && c->p <= 1) || !(c->jlo > .04f && c->jlo <= c->jhi && c->jhi <= 1) || !(c->squash_p >= 0 && c->squash_p <= 1) ||
                    !(c->klo >= 1 && c->klo <= c->khi && c->khi <= 4) || !(c->bend_p >= 0 && c->bend_p <= 1) || !(c->bend_amp >= 0 && c->bend_amp <= .2f) ||
                    !(c->kink_p >= 0 && c->kink_p <= 1) || !(c->rotate_deg >= 0 && c->rotate_deg <= 45) || c->fine < 0 || c->fine > 3;
    if (bad) { fprintf(stderr, "--aug-crush: p 0..1, void 0.05..1 (lo <= hi), squash p 0..1, k 1..4 (lo <= hi), bend p 0..1 amp 0..0.2, kink p 0..1, rot 0..45, fine 0..3\n"); return 1; }
    return 0;
}
void crush_cfg_print(const crush_cfg *c) {
    fprintf(stderr, "crush augmentation: p %.3g, void ratio %.3g..%.3g, squash p %.3g x%.3g..%.3g, bend p %.3g (%.3g P), kink p %.3g, rotation %.3g deg, CT %d levels finer\n",
            c->p, c->jlo, c->jhi, c->squash_p, c->klo, c->khi, c->bend_p, c->bend_amp, c->kink_p, c->rotate_deg, c->fine);
}
static double dot3(const double a[3], const double b[3]) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
static void unit3(double v[3]) { const double l = sqrt(dot3(v, v)); for (int d = 0; d < 3; d++) v[d] /= l; }
static void cross3(const double a[3], const double b[3], double o[3]) { o[0] = a[1] * b[2] - a[2] * b[1]; o[1] = a[2] * b[0] - a[0] * b[2]; o[2] = a[0] * b[1] - a[1] * b[0]; }

static double cr_bend(const crush_map *m, double a1, double a2) {
    double b = 0;
    for (int i = 0; i < m->nsin; i++) b += m->bA[i] * sin(m->bK[i][0] * a1 + m->bK[i][1] * a2 + m->bP[i]);
    if (m->kink) { const double t = (m->kE[0] * a1 + m->kE[1] * a2 - m->kA0) / m->kW, at = fabs(t); b += m->kS * m->kW * (at + log1p(exp(-2 * at)) - M_LN2); }
    return b;
}

void crush_map_init(crush_map *m, const crush_cfg *cfg, uint64_t seed, int P, const double n[3]) {
    memset(m, 0, sizeof *m); m->P = P; m->c = .5 * (P - 1); m->j0 = 1; m->lat = 2;
    memcpy(m->n, n, sizeof m->n);
    if (!(dot3(m->n, m->n) > 1e-12)) { m->n[0] = 1; m->n[1] = m->n[2] = 0; }
    unit3(m->n);
    {   /* t1, t2: the transverse frame (the axis least aligned with n) */
        int ax = 0; for (int d = 1; d < 3; d++) if (fabs(m->n[d]) < fabs(m->n[ax])) ax = d;
        double e[3] = {0, 0, 0}; e[ax] = 1;
        cross3(m->n, e, m->t1); unit3(m->t1); cross3(m->n, m->t1, m->t2); unit3(m->t2);
    }
    double R[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    if (cfg->rotate_deg > 0) for (int axis = 0; axis < 3; axis++) {   /* (as spatial_aug_make) */
        const double ang = (2 * cr_u(&seed) - 1) * cfg->rotate_deg * (M_PI / 180), c = cos(ang), s = sin(ang);
        const int i = (axis + 1) % 3, j = (axis + 2) % 3;
        for (int k = 0; k < 3; k++) { const double u = R[i][k], v = R[j][k]; R[i][k] = c * u - s * v; R[j][k] = s * u + c * v; }
    }
    double k = 1, d[3] = {1, 0, 0};
    if (cr_u(&seed) < cfg->squash_p && cfg->khi >= cfg->klo && cfg->klo >= 1) {
        k = cr_logu(&seed, cfg->klo, cfg->khi);
        /* in the sheet plane: the cross-section flattens (curvature squashed) while papyrus keeps its thickness; the closing
           along the normal is stage C's (a squash along n thinned the sheets to ~2 voxels a wrap on Paris 4) */
        const double th = 2 * M_PI * cr_u(&seed);
        for (int e = 0; e < 3; e++) d[e] = cos(th) * m->t1[e] + sin(th) * m->t2[e];
    }
    m->sq_k = k; memcpy(m->sq_d, d, sizeof d);
    double S[3][3], Si[3][3];
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) { S[i][j] = (i == j) + (k - 1) * d[i] * d[j]; Si[i][j] = (i == j) + (1 / k - 1) * d[i] * d[j]; }
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
        double a = 0, b = 0;
        for (int l = 0; l < 3; l++) { a += R[i][l] * S[l][j]; b += Si[i][l] * R[j][l]; }   /* M = R S, M^-1 = S^-1 R^T */
        m->M[i][j] = a; m->Minv[i][j] = b;
    }
    m->det = k;
    const double P_ = P;
    if (cr_u(&seed) < cfg->bend_p) {
        m->nsin = 2;
        for (int i = 0; i < 2; i++) {
            const double th = 2 * M_PI * cr_u(&seed), lam = P_ * (.75 + 1.25 * cr_u(&seed));
            m->bA[i] = cfg->bend_amp * P_ * cr_u(&seed); m->bK[i][0] = 2 * M_PI / lam * cos(th); m->bK[i][1] = 2 * M_PI / lam * sin(th); m->bP[i] = 2 * M_PI * cr_u(&seed);
        }
    }
    if (cr_u(&seed) < cfg->kink_p) {
        m->kink = 1; const double th = 2 * M_PI * cr_u(&seed);
        m->kS = (cr_u(&seed) < .5 ? -1 : 1) * (.3 + .7 * cr_u(&seed)); m->kW = P_ * (.02 + .06 * cr_u(&seed));
        m->kE[0] = cos(th); m->kE[1] = sin(th); m->kA0 = (2 * cr_u(&seed) - 1) * .3 * P_;
    }
    /* ranges over the output cube: a and s are linear in u (corners); |b| bounded by its amplitudes and the kink's slope */
    double smn = 1e300, smx = -1e300, T0 = 0, T1 = 0, kmx = 0;
    for (int cz = 0; cz < 2; cz++) for (int cy = 0; cy < 2; cy++) for (int cx = 0; cx < 2; cx++) {
        const double r[3] = {cz ? m->c : -m->c, cy ? m->c : -m->c, cx ? m->c : -m->c}; double q[3];
        for (int i = 0; i < 3; i++) q[i] = m->M[i][0] * r[0] + m->M[i][1] * r[1] + m->M[i][2] * r[2];
        const double a1 = dot3(q, m->t1), a2 = dot3(q, m->t2), s = dot3(q, m->n);
        T0 = fmax(T0, fabs(a1)); T1 = fmax(T1, fabs(a2)); smn = fmin(smn, s); smx = fmax(smx, s);
        if (m->kink) kmx = fmax(kmx, fabs(m->kE[0] * a1 + m->kE[1] * a2 - m->kA0));
    }
    const double B = fabs(m->bA[0]) + fabs(m->bA[1]) + (m->kink ? fabs(m->kS) * kmx : 0);
    m->T[0] = T0 + 1; m->T[1] = T1 + 1; m->srange[0] = smn - B; m->srange[1] = smx + B;
    m->half = 4 * ceil(fmax(fabs(m->srange[0]), fabs(m->srange[1])) + 4) + 24;   /* (columns covering s' at 1 / 4 of the source: papyrus 1, voids >= 0.05) */
}
void crush_footprint(const crush_map *m, double ext[3]) {
    const double sg = m->half + fmax(fabs(m->srange[0]), fabs(m->srange[1])) + 4;   /* (past a column's end F continues with slope 1) */
    for (int d = 0; d < 3; d++) ext[d] = m->T[0] * fabs(m->t1[d]) + m->T[1] * fabs(m->t2[d]) + sg * fabs(m->n[d]) + 3;
}

void crush_map_free(crush_map *m) { free(m->tab); m->tab = nullptr; }

int crush_map_scaled(crush_map *out, const crush_map *m, double t) {
    const double k = m->sq_k, *d = m->sq_d;
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) if (fabs(m->M[i][j] - ((i == j) + (k - 1) * d[i] * d[j])) > 1e-9) return 1;
    *out = *m; out->tab = nullptr;
    const double kt = pow(k, t);
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) { out->M[i][j] = (i == j) + (kt - 1) * d[i] * d[j]; out->Minv[i][j] = (i == j) + (1 / kt - 1) * d[i] * d[j]; }
    out->sq_k = out->det = kt;
    for (int i = 0; i < 2; i++) out->bA[i] = t * m->bA[i];
    out->kS = t * m->kS; out->j0 = 1 - (1 - m->j0) * t;
    if (m->tab) {
        const size_t nc = (size_t)m->G[0] * m->G[1];
        if (!(out->tab = malloc(nc * m->ns * 2 * sizeof *out->tab))) return -1;
        for (size_t c = 0; c < nc; c++) for (int mm = 0; mm < m->ns; mm++) {
            const size_t q = (c * m->ns + mm) * 2; const double s = m->smin + mm;
            out->tab[q] = (float)(s + t * (m->tab[q] - s)); out->tab[q + 1] = (float)(1 + t * (m->tab[q + 1] - 1));
        }
    }
    return 0;
}

/* quadratic B-spline weights of the nodes i0 .. i0 + 2 at node coordinate t (C1, nonnegative, sum 1) */
static void cr_bsp(double t, int *i0, double w[3]) {
    const int mm = (int)floor(t + .5); const double d = t - mm;
    w[0] = .5 * (.5 - d) * (.5 - d); w[1] = .75 - d * d; w[2] = .5 * (.5 + d) * (.5 + d); *i0 = mm - 1;
}
static double cr_F(const crush_map *m, double sp, double a1, double a2, double *dF) {
    if (!m->tab) { if (dF) *dF = 1; return sp; }
    int i0, j0; double w1[3], w2[3];
    cr_bsp((a1 - m->alo[0]) / m->h, &i0, w1); cr_bsp((a2 - m->alo[1]) / m->h, &j0, w2);
    i0 = i0 < 0 ? 0 : i0 > m->G[0] - 3 ? m->G[0] - 3 : i0;
    j0 = j0 < 0 ? 0 : j0 > m->G[1] - 3 ? m->G[1] - 3 : j0;
    const double t = sp - m->smin; int mi = (int)floor(t);
    if (mi < 0 || mi > m->ns - 2) {   /* (outside the tabulated range: slope 1 from the end knot; the range covers the output cube) */
        const int e = mi < 0 ? 0 : m->ns - 1; double se = 0;
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) se += w1[i] * w2[j] * m->tab[(((size_t)(i0 + i) * m->G[1] + j0 + j) * m->ns + e) * 2];
        if (dF) *dF = 1;
        return se + (t - e);
    }
    const double u = t - mi, u2 = u * u, u3 = u2 * u;
    const double h00 = 2 * u3 - 3 * u2 + 1, h10 = u3 - 2 * u2 + u, h01 = -2 * u3 + 3 * u2, h11 = u3 - u2;
    const double g00 = 6 * u2 - 6 * u, g10 = 3 * u2 - 4 * u + 1, g01 = -6 * u2 + 6 * u, g11 = 3 * u2 - 2 * u;
    double s = 0, ds = 0;
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
        const float *c = m->tab + (((size_t)(i0 + i) * m->G[1] + j0 + j) * m->ns + mi) * 2;
        const double w = w1[i] * w2[j];
        s += w * (h00 * c[0] + h10 * c[1] + h01 * c[2] + h11 * c[3]);
        ds += w * (g00 * c[0] + g10 * c[1] + g01 * c[2] + g11 * c[3]);
    }
    if (dF) *dF = ds;
    return s;
}

/* the CT along every column at sigma = s0 .. s1 - 1 (x = c + a1 t1 + a2 t2 + sigma n) into buf[col KX + sigma + HX], swept sigma-chunk
   major so each chunk's tiles are decoded once for all columns */
static void cr_sweep(const crush_map *m, int s0, int s1, int HX, crush_ct_fn ct, void *ud, uint8_t *buf) {
    const size_t KX = (size_t)2 * HX + 1;
    for (int c0 = s0; c0 < s1; c0 += 32) {
        const int c1 = c0 + 32 < s1 ? c0 + 32 : s1;
        for (int i = 0; i < m->G[0]; i++) for (int j = 0; j < m->G[1]; j++) {
            const double a1 = m->alo[0] + i * m->h, a2 = m->alo[1] + j * m->h;
            uint8_t *b = buf + ((size_t)i * m->G[1] + j) * KX + HX;
            for (int sg = c0; sg < c1; sg++) {
                const double x[3] = {m->c + a1 * m->t1[0] + a2 * m->t2[0] + sg * m->n[0], m->c + a1 * m->t1[1] + a2 * m->t2[1] + sg * m->n[1],
                                     m->c + a1 * m->t1[2] + a2 * m->t2[2] + sg * m->n[2]};
                const double v = ct(ud, x); b[sg] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : lrint(v));
            }
        }
    }
}
static double cr_jcol(const crush_map *m, double a1, double a2, double jth, double jk, double jph) {
    const double jc = m->j0 * exp(.4 * sin(jk * (cos(jth) * a1 + sin(jth) * a2) + jph));
    return jc < .05 ? .05 : jc > 1 ? 1 : jc;
}
int crush_map_columns(crush_map *m, double jlo, double jhi, uint64_t seed, crush_ct_fn ct, void *ud) {
    free(m->tab); m->tab = nullptr;
    const int P = m->P;
    m->h = fmax(12, fmax(m->T[0], m->T[1]) / 20);
    for (int d = 0; d < 2; d++) { m->alo[d] = -m->T[d] - 2 * m->h; m->G[d] = (int)ceil(2 * m->T[d] / m->h) + 5; }
    m->smin = floor(m->srange[0]) - 3; m->ns = (int)(ceil(m->srange[1]) + 3 - m->smin) + 1;
    const size_t nc = (size_t)m->G[0] * m->G[1];
    const int S = (int)ceil(fmax(fabs(m->srange[0]), fabs(m->srange[1]))) + 4, HX = (int)m->half;   /* output half range, the column limit */
    const size_t KX = (size_t)2 * HX + 1;
    uint8_t *buf = malloc(nc * KX);
    if (!buf) return 1;
    /* the output's own range first: the Otsu papyrus / void split */
    cr_sweep(m, -S, S + 1, HX, ct, ud, buf);
    size_t hist[256] = {0}, nz = 0;
    for (size_t c = 0; c < nc; c++) for (int k = -S; k <= S; k++) { const uint8_t v = buf[c * KX + HX + k]; if (v) { hist[v]++; nz++; } }
    if (nz < 1000 || nz < nc * (2 * S + 1) / 50) { free(buf); return 1; }
    double tot = 0; for (int b = 1; b < 256; b++) tot += (double)b * hist[b];
    double w0 = 0, s0 = 0, best = -1; int thr = 128;
    for (int b = 1; b < 255; b++) {
        w0 += hist[b]; s0 += (double)b * hist[b];
        const double w1 = (double)nz - w0; if (w0 < 1 || w1 < 1) continue;
        const double m0 = s0 / w0, m1 = (tot - s0) / w1, bv = w0 * w1 * (m0 - m1) * (m0 - m1);
        if (bv > best) { best = bv; thr = b; }
    }
    double lo = 0, nlo = 0, hi = 0, nhi = 0;
    for (int b = 1; b < 256; b++) { if (b <= thr) { lo += (double)b * hist[b]; nlo += hist[b]; } else { hi += (double)b * hist[b]; nhi += hist[b]; } }
    /* voidness a ramp over the middle 30 % between the class means (their midpoint: Otsu's threshold is anywhere in a histogram gap) */
    const double mlo = nlo ? lo / nlo : thr, mhi = nhi ? hi / nhi : thr, wv = fmax(4, .3 * (mhi - mlo)), t = .5 * (mlo + mhi);
    m->j0 = cr_logu(&seed, jlo, jhi);
    const double jth = 2 * M_PI * cr_u(&seed), jk = 2 * M_PI / (P * (1 + cr_u(&seed))), jph = 2 * M_PI * cr_u(&seed);
    float vt[256]; for (int c = 0; c < 256; c++) { const double x = c == 0 ? 1 : .5 + (t - c) / wv; vt[c] = (float)(x < 0 ? 0 : x > 1 ? 1 : x); }
    /* grow both ends in 32-sample chunks until every column's int rho (unblurred) covers the output's s' range (+ 24), HX at most */
    int Hm = S, Hp = S;
    for (int dir = 0; dir < 2; dir++) {
        for (;;) {
            const int H = dir ? Hp : Hm; int need = 0;
            for (int i = 0; i < m->G[0] && !need; i++) for (int j = 0; j < m->G[1] && !need; j++) {
                const double jc = cr_jcol(m, m->alo[0] + i * m->h, m->alo[1] + j * m->h, jth, jk, jph);
                const uint8_t *b = buf + ((size_t)i * m->G[1] + j) * KX + HX; double g = 0;
                for (int k = 1; k <= H; k++) g += 1 - (1 - jc) * vt[b[dir ? k : -k]];
                need = g < S + 24;
            }
            if (!need || H >= HX) break;
            const int H2 = H + 32 < HX ? H + 32 : HX;
            if (dir) { cr_sweep(m, Hp + 1, H2 + 1, HX, ct, ud, buf); Hp = H2; } else { cr_sweep(m, -H2, -Hm, HX, ct, ud, buf); Hm = H2; }
        }
    }
    const int K = Hm + Hp + 1;
    m->tab = malloc(nc * m->ns * 2 * sizeof *m->tab);
    double *rho = malloc((size_t)K * sizeof *rho), *g = malloc((size_t)K * sizeof *g), *vb = malloc((size_t)K * sizeof *vb);
    if (!m->tab || !rho || !g || !vb) { free(buf); free(rho); free(g); free(vb); free(m->tab); m->tab = nullptr; return 1; }
    double gk[11]; { double sw = 0; for (int q = -5; q <= 5; q++) sw += gk[q + 5] = exp(-q * q / (2 * 1.5 * 1.5)); for (int q = 0; q < 11; q++) gk[q] /= sw; }
    for (int i = 0; i < m->G[0]; i++) for (int j = 0; j < m->G[1]; j++) {
        const double jc = cr_jcol(m, m->alo[0] + i * m->h, m->alo[1] + j * m->h, jth, jk, jph);
        const uint8_t *b = buf + ((size_t)i * m->G[1] + j) * KX + HX - Hm;   /* sample k: sigma = k - Hm */
        for (int k = 0; k < K; k++) vb[k] = vt[b[k]];
        for (int k = 0; k < K; k++) {
            double s = 0; for (int q = -5; q <= 5; q++) { const int kk = k + q < 0 ? 0 : k + q >= K ? K - 1 : k + q; s += gk[q + 5] * vb[kk]; }
            const int e = k < K - 1 - k ? k : K - 1 - k; double tp = 1;
            if (e < 16) { const double z = e / 16.0; tp = z * z * (3 - 2 * z); }   /* rho -> 1 at the ends: slope 1 continues C1 */
            rho[k] = 1 - (1 - jc) * s * tp;
        }
        g[0] = 0; for (int k = 0; k + 1 < K; k++) g[k + 1] = g[k] + .5 * (rho[k] + rho[k + 1]);
        const double g0 = g[Hm]; for (int k = 0; k < K; k++) g[k] -= g0;
        float *tc = m->tab + ((size_t)i * m->G[1] + j) * m->ns * 2;
        int k = 0;
        for (int mm = 0; mm < m->ns; mm++) {
            const double s = m->smin + mm; double sg, dv;
            if (s <= g[0]) { sg = -Hm + (s - g[0]); dv = 1; }
            else if (s >= g[K - 1]) { sg = Hp + (s - g[K - 1]); dv = 1; }
            else {
                while (k < K - 2 && g[k + 1] <= s) k++;
                const double r0 = rho[k], dr = rho[k + 1] - rho[k], e = s - g[k];
                const double tt = 2 * e / (r0 + sqrt(fmax(0, r0 * r0 + 2 * dr * e)));   /* g[k] + r0 t + dr t^2 / 2 = s, t in [0, 1] */
                sg = -Hm + k + tt; dv = 1 / (r0 + dr * tt);
            }
            tc[2 * mm] = (float)sg; tc[2 * mm + 1] = (float)dv;
        }
        for (int mm = 0; mm < m->ns; mm++) {   /* derivatives <= 3 x the neighbouring secants: a monotone Hermite (Fritsch-Carlson) */
            double lim = 1e30;
            if (mm > 0) lim = fmin(lim, 3 * (tc[2 * mm] - tc[2 * mm - 2]));
            if (mm + 1 < m->ns) lim = fmin(lim, 3 * (tc[2 * mm + 2] - tc[2 * mm]));
            if (tc[2 * mm + 1] > lim) tc[2 * mm + 1] = (float)lim;
        }
    }
    free(buf); free(rho); free(g); free(vb);
    return 0;
}

void crush_pull(const crush_map *m, const double u[3], double x[3], double *jdet) {
    const double r[3] = {u[0] - m->c, u[1] - m->c, u[2] - m->c}; double q[3];
    for (int i = 0; i < 3; i++) q[i] = m->M[i][0] * r[0] + m->M[i][1] * r[1] + m->M[i][2] * r[2];
    const double a1 = dot3(q, m->t1), a2 = dot3(q, m->t2), s = dot3(q, m->n) + cr_bend(m, a1, a2);
    double dF; const double sg = cr_F(m, s, a1, a2, &dF);
    for (int d = 0; d < 3; d++) x[d] = m->c + a1 * m->t1[d] + a2 * m->t2[d] + sg * m->n[d];
    if (jdet) *jdet = m->det * dF;
}

void crush_pull_window(const crush_map *m, int box, const int u[3], double x[3]) {
    if (m->lat != 2) { const double v[3] = {u[0], u[1], u[2]}; crush_pull(m, v, x, nullptr); return; }
    const int h = box ? 1 : 0; int b[3], o[3];   /* (nodes at u = -h mod 2: cr_block's lattice, its blocks start at even voxels) */
    for (int d = 0; d < 3; d++) { o[d] = (u[d] + h) & 1; b[d] = u[d] - o[d]; }
    const double w = 1.0 / (1 << (o[0] + o[1] + o[2]));
    x[0] = x[1] = x[2] = 0;
    for (int c = 0; c < 8; c++) {
        const int bz = c >> 2 & 1, by = c >> 1 & 1, bx = c & 1;
        if ((bz && !o[0]) || (by && !o[1]) || (bx && !o[2])) continue;
        const double v[3] = {b[0] + 2 * bz, b[1] + 2 * by, b[2] + 2 * bx}; double s[3];
        crush_pull(m, v, s, nullptr);
        for (int d = 0; d < 3; d++) x[d] += w * s[d];
    }
}
void crush_push(const crush_map *m, const double x[3], double u[3]) {
    const double d[3] = {x[0] - m->c, x[1] - m->c, x[2] - m->c};
    const double a1 = dot3(d, m->t1), a2 = dot3(d, m->t2), sg = dot3(d, m->n);
    /* F(., a) is increasing (a monotone Hermite per column, a convex blend of columns): bracket, then bisect */
    double lo = sg, hi = sg;
    for (int i = 0; i < 200 && cr_F(m, lo, a1, a2, nullptr) > sg; i++) lo -= 1 + (sg - lo);
    for (int i = 0; i < 200 && cr_F(m, hi, a1, a2, nullptr) < sg; i++) hi += 1 + (hi - sg);
    for (int i = 0; i < 100 && hi - lo > 1e-11; i++) { const double mid = .5 * (lo + hi); if (cr_F(m, mid, a1, a2, nullptr) < sg) lo = mid; else hi = mid; }
    const double s = .5 * (lo + hi) - cr_bend(m, a1, a2);
    double q[3]; for (int k = 0; k < 3; k++) q[k] = a1 * m->t1[k] + a2 * m->t2[k] + s * m->n[k];
    for (int i = 0; i < 3; i++) u[i] = m->c + m->Minv[i][0] * q[0] + m->Minv[i][1] * q[1] + m->Minv[i][2] * q[2];
}

void crush_jacobian(const crush_map *m, const double u[3], double J[3][3]) {
    const double h = 1e-4;
    for (int j = 0; j < 3; j++) {
        double up[3] = {u[0], u[1], u[2]}, um[3] = {u[0], u[1], u[2]}, xp[3], xm[3];
        up[j] += h; um[j] -= h; crush_pull(m, up, xp, nullptr); crush_pull(m, um, xm, nullptr);
        for (int i = 0; i < 3; i++) J[i][j] = (xp[i] - xm[i]) / (2 * h);
    }
}

void crush_bounds(const crush_map *m, double lo[3], double hi[3]) {
    for (int d = 0; d < 3; d++) { lo[d] = 1e300; hi[d] = -1e300; }
    const int P = m->P, st = P > 64 ? 4 : 2;
    for (int f = 0; f < 6; f++) {   /* the cube's faces (a homeomorphism maps the boundary onto the image's boundary) */
        const int ax = f >> 1, v = (f & 1) ? P - 1 : 0, a1 = (ax + 1) % 3, a2 = (ax + 2) % 3;
        for (int i = 0; ; i += st) {
            if (i > P - 1) i = P - 1;
            for (int j = 0; ; j += st) {
                if (j > P - 1) j = P - 1;
                double u[3], x[3]; u[ax] = v; u[a1] = i; u[a2] = j;
                crush_pull(m, u, x, nullptr);
                for (int d = 0; d < 3; d++) { lo[d] = fmin(lo[d], x[d]); hi[d] = fmax(hi[d], x[d]); }
                if (j == P - 1) break;
            }
            if (i == P - 1) break;
        }
    }
    for (int d = 0; d < 3; d++) { lo[d] -= 2; hi[d] += 2; }
}

/* ---- the tile cache ---- */
enum { TS = 128, TSH = 7, TB = TS * TS * TS, TLB = 256 };
struct crush_slot { const z3 *z; int fill; int64_t t[3]; uint64_t used; };
typedef struct { uint64_t k; int slot; } cr_tlb;

int crush_cache_init(crush_cache *cc, uint8_t *arena, size_t bytes) {
    memset(cc, 0, sizeof *cc);
    cc->arena = arena; cc->nslots = (int)(bytes / TB);
    if (cc->nslots < 16) return -1;
    cc->key = calloc((size_t)cc->nslots + 2 * TLB, sizeof *cc->key);   /* keys, then the direct-mapped (key, slot) memo */
    cc->slot = calloc((size_t)cc->nslots, sizeof *cc->slot);
    if (!cc->key || !cc->slot) { crush_cache_free(cc); return -1; }
    return 0;
}
void crush_cache_free(crush_cache *cc) { free(cc->key); free(cc->slot); cc->key = nullptr; cc->slot = nullptr; }

static uint64_t cr_key(const z3 *z, int fill, int64_t tz, int64_t ty, int64_t tx) {
    uint64_t h = (uint64_t)(uintptr_t)z * 0x9e3779b97f4a7c15ull ^ (uint64_t)(fill + 1) * 0xc2b2ae3d27d4eb4full;
    h ^= (uint64_t)tz * 0x165667b19e3779f9ull; h = (h ^ (h >> 29)) * 0xbf58476d1ce4e5b9ull;
    h ^= (uint64_t)ty * 0x94d049bb133111ebull; h = (h ^ (h >> 31)) * 0x9e3779b97f4a7c15ull;
    h ^= (uint64_t)tx * 0xd6e8feb86659fd93ull; h = (h ^ (h >> 32)) * 0xbf58476d1ce4e5b9ull;
    return (h ^ (h >> 29)) | 1;
}
static int cr_match(const struct crush_slot *s, const z3 *z, int fill, int64_t tz, int64_t ty, int64_t tx) {
    return s->z == z && s->fill == fill && s->t[0] == tz && s->t[1] == ty && s->t[2] == tx;
}
/* the tile's bytes (loaded when absent), marked used at stamp (tiles used at the current stamp are never evicted); nullptr: every
   slot is pinned (cc->err 0) or the read failed (cc->err 1) */
static const uint8_t *cr_tile(crush_cache *cc, const crush_layer *L, int64_t tz, int64_t ty, int64_t tx, uint64_t stamp) {
    const uint64_t k = cr_key(L->z, L->fill, tz, ty, tx);
    uint64_t *tk = cc->key + cc->nslots + 2 * (k & (TLB - 1));
    if (tk[0] == k) {
        const int i = (int)tk[1];
        if (cc->key[i] == k && cr_match(&cc->slot[i], L->z, L->fill, tz, ty, tx)) { cc->slot[i].used = stamp; return cc->arena + (size_t)i * TB; }
    }
    int v = -1; uint64_t best = UINT64_MAX;
    for (int i = 0; i < cc->nslots; i++) {
        if (cc->key[i] == k && cr_match(&cc->slot[i], L->z, L->fill, tz, ty, tx)) { cc->slot[i].used = stamp; tk[0] = k; tk[1] = (uint64_t)i; return cc->arena + (size_t)i * TB; }
        if (!cc->key[i]) { if (best) { v = i; best = 0; } }
        else if (cc->slot[i].used != stamp && cc->slot[i].used < best) { v = i; best = cc->slot[i].used; }
    }
    if (v < 0) return nullptr;
    uint8_t *mem = cc->arena + (size_t)v * TB;
    cc->key[v] = 0;
    const z3_meta *mt = z3_meta_of(L->z);
    const int64_t o[3] = {tz * TS, ty * TS, tx * TS}, n[3] = {TS, TS, TS};
    int any = 1, all = 1;
    for (int d = 0; d < 3; d++) { any &= o[d] < mt->shape[d] && o[d] + TS > 0; all &= o[d] >= 0 && o[d] + TS <= mt->shape[d]; }
    if (!any) memset(mem, L->fill, TB);
    else {
        if (z3_read(L->z, o, n, mem, 1)) { cc->err = 1; return nullptr; }
        cc->decoded++;
        if (!all) for (int z = 0; z < TS; z++) {   /* outside the array: the layer's fill (the array's own fill may be a valid code) */
            const int64_t gz = o[0] + z; uint8_t *pl = mem + (size_t)z * TS * TS;
            if (gz < 0 || gz >= mt->shape[0]) { memset(pl, L->fill, (size_t)TS * TS); continue; }
            for (int y = 0; y < TS; y++) {
                const int64_t gy = o[1] + y; uint8_t *row = pl + (size_t)y * TS;
                if (gy < 0 || gy >= mt->shape[1]) { memset(row, L->fill, TS); continue; }
                const int64_t x0 = o[2] < 0 ? -o[2] : 0, x1 = o[2] + TS > mt->shape[2] ? mt->shape[2] - o[2] : TS;
                if (x0 > 0) memset(row, L->fill, (size_t)(x0 < TS ? x0 : TS));
                if (x1 < TS) memset(row + (x1 > 0 ? x1 : 0), L->fill, (size_t)(TS - (x1 > 0 ? x1 : 0)));
            }
        }
    }
    cc->slot[v] = (struct crush_slot){L->z, L->fill, {tz, ty, tx}, stamp};
    cc->key[v] = k; tk[0] = k; tk[1] = (uint64_t)v;
    return mem;
}
static int cr_point(crush_cache *cc, const crush_layer *L, int64_t z, int64_t y, int64_t x, uint64_t stamp) {
    const uint8_t *t = cr_tile(cc, L, z >> TSH, y >> TSH, x >> TSH, stamp);
    if (!t) return -1;
    return t[((size_t)(z & (TS - 1)) << (2 * TSH)) | ((size_t)(y & (TS - 1)) << TSH) | (size_t)(x & (TS - 1))];
}
double crush_linear_at(crush_cache *cc, const crush_layer *L, const double p[3]) {
    const uint64_t st = ++cc->clock; int64_t k0[3]; double f[3];
    for (int d = 0; d < 3; d++) { const double fl = floor(p[d]); k0[d] = (int64_t)fl; f[d] = p[d] - fl; }
    double s = 0;
    for (int c = 0; c < 8; c++) {
        const int bz = c >> 2 & 1, by = c >> 1 & 1, bx = c & 1;
        const double w = (bz ? f[0] : 1 - f[0]) * (by ? f[1] : 1 - f[1]) * (bx ? f[2] : 1 - f[2]);
        if (w == 0) continue;
        const int v = cr_point(cc, L, k0[0] + bz, k0[1] + by, k0[2] + bx, st);
        if (v < 0) return 0;
        s += w * v;
    }
    return s;
}
int crush_nearest_at(crush_cache *cc, const crush_layer *L, const double p[3]) {
    return cr_point(cc, L, (int64_t)floor(p[0] + .5), (int64_t)floor(p[1] + .5), (int64_t)floor(p[2] + .5), ++cc->clock);
}

void crush_layer_level(crush_layer *L, int finer) {
    L->a = ldexp(1, finer);
    for (int d = 0; d < 3; d++) L->b[d] = .5 * L->a - .5;
}
void crush_layer_phase(crush_layer *L, int shift) {
    const double f = ldexp(1, shift);
    L->a = 1 / f;
    for (int d = 0; d < 3; d++) L->b[d] = (shift - (f - 1) / 2) / f;
}

int crush_phase_normal(crush_cache *cc, const crush_layer *ph, const int64_t o[3], int P, const double *hint, double n[3]) {
    const int st = P / 32 > 2 ? P / 32 : 2; double T[3][3] = {{0}}; int cnt = 0;
    for (int z = P / 4; z < 3 * P / 4; z += st) for (int y = P / 4; y < 3 * P / 4; y += st) for (int x = P / 4; x < 3 * P / 4; x += st) {
        const int u[3] = {z, y, x}; double g[3]; int ok = 1;
        for (int d = 0; d < 3 && ok; d++) {
            int cv[2];
            for (int s = 0; s < 2; s++) {
                double p[3]; for (int e = 0; e < 3; e++) p[e] = ph->a * (double)(o[e] + u[e] + (e == d ? (s ? 2 : -2) : 0)) + ph->b[e];
                cv[s] = crush_nearest_at(cc, ph, p);
                if (cv[s] < 0 || cv[s] >= PHASE_PERIOD) ok = 0;
            }
            if (!ok) break;
            int dd = cv[1] - cv[0]; if (dd > PHASE_PERIOD / 2) dd -= PHASE_PERIOD; if (dd < -PHASE_PERIOD / 2) dd += PHASE_PERIOD;
            g[d] = dd / 4.0;
        }
        if (!ok) continue;
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) T[i][j] += g[i] * g[j];
        cnt++;
    }
    if (cnt < 32) return cnt;
    double v[3] = {hint ? hint[0] : .577, hint ? hint[1] : .577, hint ? hint[2] : .577};
    if (!(dot3(v, v) > 1e-12)) v[0] = v[1] = v[2] = .577;
    unit3(v);
    for (int it = 0; it < 60; it++) {
        double w[3]; for (int i = 0; i < 3; i++) w[i] = T[i][0] * v[0] + T[i][1] * v[1] + T[i][2] * v[2];
        if (!(dot3(w, w) > 1e-30)) return 0;
        unit3(w); memcpy(v, w, sizeof v);
    }
    memcpy(n, v, sizeof v);
    return cnt;
}

/* ---- the window pull ---- */
typedef struct { const uint8_t **p; int64_t t0[3]; int n[3]; } cr_grid;
static inline int cr_get(const cr_grid *g, int64_t z, int64_t y, int64_t x) {
    const uint8_t *t = g->p[(((z >> TSH) - g->t0[0]) * g->n[1] + ((y >> TSH) - g->t0[1])) * g->n[2] + ((x >> TSH) - g->t0[2])];
    return t[((size_t)(z & (TS - 1)) << (2 * TSH)) | ((size_t)(y & (TS - 1)) << TSH) | (size_t)(x & (TS - 1))];
}
/* the 8 corners from one tile (the common case: not on a tile's last plane / row / column) */
static inline const uint8_t *cr_cell(const cr_grid *g, int64_t z, int64_t y, int64_t x) {
    if ((z & (TS - 1)) == TS - 1 || (y & (TS - 1)) == TS - 1 || (x & (TS - 1)) == TS - 1) return nullptr;
    const uint8_t *t = g->p[(((z >> TSH) - g->t0[0]) * g->n[1] + ((y >> TSH) - g->t0[1])) * g->n[2] + ((x >> TSH) - g->t0[2])];
    return t + (((size_t)(z & (TS - 1)) << (2 * TSH)) | ((size_t)(y & (TS - 1)) << TSH) | (size_t)(x & (TS - 1)));
}
static double cr_tri(const cr_grid *g, const double p[3]) {
    const double fz = floor(p[0]), fy = floor(p[1]), fx = floor(p[2]);
    const int64_t z = (int64_t)fz, y = (int64_t)fy, x = (int64_t)fx; const double a = p[0] - fz, b = p[1] - fy, c = p[2] - fx;
    const uint8_t *q = cr_cell(g, z, y, x);
    if (q) {
        enum { DY = TS, DZ = TS * TS };
        const double c00 = q[0] + (q[1] - q[0]) * c, c01 = q[DY] + (q[DY + 1] - q[DY]) * c;
        const double c10 = q[DZ] + (q[DZ + 1] - q[DZ]) * c, c11 = q[DZ + DY] + (q[DZ + DY + 1] - q[DZ + DY]) * c;
        const double d0 = c00 + (c01 - c00) * b, d1 = c10 + (c11 - c10) * b;
        return d0 + (d1 - d0) * a;
    }
    const double c00 = cr_get(g, z, y, x) * (1 - c) + cr_get(g, z, y, x + 1) * c, c01 = cr_get(g, z, y + 1, x) * (1 - c) + cr_get(g, z, y + 1, x + 1) * c;
    const double c10 = cr_get(g, z + 1, y, x) * (1 - c) + cr_get(g, z + 1, y, x + 1) * c, c11 = cr_get(g, z + 1, y + 1, x) * (1 - c) + cr_get(g, z + 1, y + 1, x + 1) * c;
    return (c00 * (1 - b) + c01 * b) * (1 - a) + (c10 * (1 - b) + c11 * b) * a;
}
static float cr_cs[PHASE_PERIOD], cr_sn[PHASE_PERIOD]; static _Atomic int cr_tabs;
/* atan2 to 2e-6 rad (1e-4 phase codes): a degree-11 odd minimax polynomial on [0, 1] plus octant folding */
static inline float cr_atan2(float y, float x) {
    const float ax = fabsf(x), ay = fabsf(y), mx = ax > ay ? ax : ay, mn = ax > ay ? ay : ax;
    if (mx == 0.f) return 0.f;
    const float a = mn / mx, s = a * a;
    float r = ((((-0.0117212f * s + 0.05265332f) * s - 0.11643287f) * s + 0.19354346f) * s - 0.33262347f) * s + 0.99997726f;
    r *= a;
    if (ay > ax) r = 1.57079637f - r;
    if (x < 0.f) r = 3.14159274f - r;
    return y < 0.f ? -r : r;
}
static uint8_t cr_phase(const cr_grid *g, const double p[3]) {
    const double fz = floor(p[0]), fy = floor(p[1]), fx = floor(p[2]);
    const int64_t z = (int64_t)fz, y = (int64_t)fy, x = (int64_t)fx; const float f[3] = {(float)(p[0] - fz), (float)(p[1] - fy), (float)(p[2] - fx)};
    float sx = 0.f, sy = 0.f, wk = 0.f;
    const uint8_t *q = cr_cell(g, z, y, x);
    for (int c = 0; c < 8; c++) {
        const int bz = c >> 2 & 1, by = c >> 1 & 1, bx = c & 1;
        const float w = (bz ? f[0] : 1 - f[0]) * (by ? f[1] : 1 - f[1]) * (bx ? f[2] : 1 - f[2]);
        if (w <= 1e-7f) continue;
        const int ph = q ? q[(bz << (2 * TSH)) | (by << TSH) | bx] : cr_get(g, z + bz, y + by, x + bx);
        if (ph >= PHASE_PERIOD) continue;
        sx += w * cr_cs[ph]; sy += w * cr_sn[ph]; wk += w;
    }
    if (wk < .5f || sx * sx + sy * sy < 1e-8f) return PHASE_UNKNOWN;
    float t = cr_atan2(sy, sx) * (float)(PHASE_PERIOD / (2 * M_PI)); if (t < 0) t += PHASE_PERIOD;
    const int r = (int)lrintf(t); return (uint8_t)(r >= PHASE_PERIOD ? r - PHASE_PERIOD : r);
}

typedef struct { const crush_map *m; crush_cache *cc; const int64_t *o; crush_layer *L; int nl, halo; double *X, *N; const uint8_t **gp; } cr_ctx;

static int cr_block(cr_ctx *C, int z0, int y0, int x0, int B) {
    const crush_map *m = C->m; const int P = m->P, h = C->halo;
    const int nz = z0 + B > P ? P - z0 : B, ny = y0 + B > P ? P - y0 : B, nx = x0 + B > P ? P - x0 : B;
    if (nz <= 0 || ny <= 0 || nx <= 0) return 0;
    const int lz = nz + 2 * h, ly = ny + 2 * h, lx = nx + 2 * h;
    double *X = C->X;
    if (m->lat == 2) {   /* the map at every other voxel (nodes 0, 2, .. covering the block), trilinear between */
        const int mz = lz / 2 + 1, my = ly / 2 + 1, mx = lx / 2 + 1; double *N = C->N;
        for (int z = 0; z < mz; z++) for (int y = 0; y < my; y++) for (int x = 0; x < mx; x++) {
            const double u[3] = {z0 - h + 2 * z, y0 - h + 2 * y, x0 - h + 2 * x};
            crush_pull(m, u, N + (((size_t)z * my + y) * mx + x) * 3, nullptr);
        }
        for (int z = 0; z < lz; z++) for (int y = 0; y < ly; y++) for (int x = 0; x < lx; x++) {
            const int iz = z >> 1, iy = y >> 1, ix = x >> 1, oz = z & 1, oy = y & 1, ox = x & 1;
            double *o = X + (((size_t)z * ly + y) * lx + x) * 3; o[0] = o[1] = o[2] = 0;
            const double w = 1.0 / (1 << (oz + oy + ox));
            for (int c = 0; c < 8; c++) {
                const int bz = c >> 2 & 1, by = c >> 1 & 1, bx = c & 1;
                if ((bz && !oz) || (by && !oy) || (bx && !ox)) continue;
                const double *s = N + (((size_t)(iz + bz) * my + iy + by) * mx + ix + bx) * 3;
                o[0] += w * s[0]; o[1] += w * s[1]; o[2] += w * s[2];
            }
        }
    } else
        for (int z = 0; z < lz; z++) for (int y = 0; y < ly; y++) for (int x = 0; x < lx; x++) {
            const double u[3] = {z0 + z - h, y0 + y - h, x0 + x - h};
            crush_pull(m, u, X + (((size_t)z * ly + y) * lx + x) * 3, nullptr);
        }
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    const size_t nlat = (size_t)lz * ly * lx;
    for (size_t k = 0; k < nlat; k++) for (int d = 0; d < 3; d++) { lo[d] = fmin(lo[d], X[3 * k + d]); hi[d] = fmax(hi[d], X[3 * k + d]); }
    const uint64_t stamp = ++C->cc->clock;
    cr_grid G[8]; int used = 0;
    for (int l = 0; l < C->nl; l++) {
        const crush_layer *L = &C->L[l]; int cnt = 1;
        for (int d = 0; d < 3; d++) {
            const double a = L->a * ((double)C->o[d] + lo[d]) + L->b[d] - 1, b = L->a * ((double)C->o[d] + hi[d]) + L->b[d] + 2;
            G[l].t0[d] = (int64_t)floor(a) >> TSH; G[l].n[d] = (int)(((int64_t)floor(b) >> TSH) - G[l].t0[d] + 1); cnt *= G[l].n[d];
        }
        if (used + cnt > C->cc->nslots || used + cnt > 4096) return 1;
        G[l].p = C->gp + used;
        for (int tz = 0; tz < G[l].n[0]; tz++) for (int ty = 0; ty < G[l].n[1]; ty++) for (int tx = 0; tx < G[l].n[2]; tx++) {
            const uint8_t *t = cr_tile(C->cc, L, G[l].t0[0] + tz, G[l].t0[1] + ty, G[l].t0[2] + tx, stamp);
            if (!t) return C->cc->err ? -1 : 1;
            G[l].p[((size_t)tz * G[l].n[1] + ty) * G[l].n[2] + tx] = t;
        }
        used += cnt;
    }
    for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++) {
        const size_t li = ((size_t)(z + h) * ly + y + h) * lx + x + h, oi = ((size_t)(z0 + z) * P + y0 + y) * P + x0 + x;
        const double *xp = X + 3 * li;
        for (int l = 0; l < C->nl; l++) {
            const crush_layer *L = &C->L[l]; double p[3];
            for (int d = 0; d < 3; d++) p[d] = L->a * ((double)C->o[d] + xp[d]) + L->b[d];
            if (L->mode == CRUSH_NEAREST) { L->out[oi] = (uint8_t)cr_get(&G[l], (int64_t)floor(p[0] + .5), (int64_t)floor(p[1] + .5), (int64_t)floor(p[2] + .5)); continue; }
            if (L->mode == CRUSH_PHASE) { L->out[oi] = cr_phase(&G[l], p); continue; }
            double v;
            if (L->box && h) {   /* the mean of the pulls at the 8 sub-positions u +- 1/4 (Jacobian columns by central differences) */
                double J[3][3]; const size_t sd[3] = {(size_t)ly * lx, (size_t)lx, 1};
                for (int d = 0; d < 3; d++) for (int e = 0; e < 3; e++) J[e][d] = .5 * (X[3 * (li + sd[d]) + e] - X[3 * (li - sd[d]) + e]);
                v = 0;
                for (int c = 0; c < 8; c++) {
                    const double s[3] = {(c >> 2 & 1) ? .25 : -.25, (c >> 1 & 1) ? .25 : -.25, (c & 1) ? .25 : -.25}; double q[3];
                    for (int e = 0; e < 3; e++) q[e] = L->a * ((double)C->o[e] + xp[e] + J[e][0] * s[0] + J[e][1] * s[1] + J[e][2] * s[2]) + L->b[e];
                    v += cr_tri(&G[l], q);
                }
                v *= .125;
            } else v = cr_tri(&G[l], p);
            L->out[oi] = (uint8_t)lrint(v < 0 ? 0 : v > 255 ? 255 : v);
        }
    }
    return 0;
}
static int cr_block_split(cr_ctx *C, int z0, int y0, int x0, int B) {
    const int rc = cr_block(C, z0, y0, x0, B);
    if (rc <= 0) return rc;
    if (B <= 4) { C->cc->err = 2; return -1; }   /* (tiles of one 4^3 block do not fit the cache) */
    const int b = B / 2;
    for (int k = 0; k < 8; k++) { const int r = cr_block_split(C, z0 + ((k >> 2 & 1) ? b : 0), y0 + ((k >> 1 & 1) ? b : 0), x0 + ((k & 1) ? b : 0), b); if (r) return r; }
    return 0;
}
typedef struct { uint64_t key; int i; } cr_ord;
static int cr_ord_cmp(const void *a, const void *b) {
    const cr_ord *x = a, *y = b;
    return x->key < y->key ? -1 : x->key > y->key ? 1 : x->i - y->i;
}
int crush_window(const crush_map *m, crush_cache *cc, const int64_t o[3], crush_layer *L, int nl) {
    if (nl > 8) return -1;
    if (!atomic_load(&cr_tabs)) { for (int k = 0; k < PHASE_PERIOD; k++) { cr_cs[k] = (float)cos(2 * M_PI * k / PHASE_PERIOD); cr_sn[k] = (float)sin(2 * M_PI * k / PHASE_PERIOD); } atomic_store(&cr_tabs, 1); }
    int halo = 0; for (int l = 0; l < nl; l++) halo |= L[l].mode == CRUSH_LINEAR && L[l].box;
    const int B = m->P < 32 ? m->P : 32;
    const size_t nn = (size_t)((B + 2) / 2 + 1) * ((B + 2) / 2 + 1) * ((B + 2) / 2 + 1) * 3;
    cr_ctx C = {m, cc, o, L, nl, halo, malloc((size_t)(B + 2) * (B + 2) * (B + 2) * 3 * sizeof(double)), malloc(nn * sizeof(double)), malloc(4096 * sizeof(uint8_t *))};
    if (!C.X || !C.N || !C.gp) { free(C.X); free(C.N); free(C.gp); return -1; }
    int rc = 0;
    /* blocks in the Morton order of their source centres' tiles: neighbouring source tiles are read back to back (output order
       thrashes the cache when the map squashes / rotates the footprint) */
    const int nb = (m->P + B - 1) / B, nbl = nb * nb * nb;
    cr_ord *ord = malloc((size_t)nbl * sizeof *ord);
    if (!ord) { free(C.X); free(C.N); free(C.gp); return -1; }
    for (int i = 0; i < nbl; i++) {
        const int bz = i / (nb * nb), by = i / nb % nb, bx = i % nb;
        const double u[3] = {bz * B + .5 * (B - 1), by * B + .5 * (B - 1), bx * B + .5 * (B - 1)}; double x[3];
        crush_pull(m, u, x, nullptr);
        uint64_t key = 0;
        for (int d = 0; d < 3; d++) {
            const uint64_t t = (uint64_t)(((int64_t)floor((double)o[d] + x[d]) >> TSH) + (1 << 20)) & 0x1fffff;   /* (21 bits a tile axis) */
            for (int b = 0; b < 21; b++) key |= ((t >> b) & 1) << (3 * b + (2 - d));
        }
        ord[i] = (cr_ord){key, i};
    }
    qsort(ord, (size_t)nbl, sizeof *ord, cr_ord_cmp);
    for (int k = 0; k < nbl && !rc; k++) {
        const int i = ord[k].i, bz = i / (nb * nb), by = i / nb % nb, bx = i % nb;
        rc = cr_block_split(&C, bz * B, by * B, bx * B, B);
    }
    free(ord);
    free(C.X); free(C.N); free(C.gp);
    return rc;
}
