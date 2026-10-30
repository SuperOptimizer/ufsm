#include "vwt_aug.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t va_next(uint64_t *s) { uint64_t z = (*s += 0x9e3779b97f4a7c15ull); z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull; return z ^ (z >> 31); }
static double va_u(uint64_t *s) { return (double)(va_next(s) >> 11) * 0x1.0p-53; }
static double va_gauss(uint64_t *s) { const double u = fmax(va_u(s), 1e-300), v = va_u(s); return sqrt(-2 * log(u)) * cos(2 * M_PI * v); }

vwa_cfg vwa_cfg_default(void) {
    return (vwa_cfg){.on = 1, .geo_p = .5f, .df = deform_cfg_default(), .lattice = 2,
                     .gain_p = .5f, .gamma_p = .3f, .noise_p = .3f, .blur_p = .3f, .lowres_p = .1f, .quant_p = .1f, .strength = 1};
}
static const char *va_opt(int argc, char **argv, const char *f) { for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], f)) return argv[i + 1]; return nullptr; }
int vwa_cfg_parse(vwa_cfg *c, int argc, char **argv) {
    const char *v;
    if (deform_cfg_parse(&c->df, argc, argv)) return 1;
    if ((v = va_opt(argc, argv, "--vaug"))) c->on = atoi(v);
    if ((v = va_opt(argc, argv, "--vaug-geo"))) c->geo_p = (float)atof(v);
    if ((v = va_opt(argc, argv, "--vaug-lattice"))) c->lattice = atoi(v);
    if ((v = va_opt(argc, argv, "--vaug-strength"))) c->strength = (float)atof(v);
    struct { const char *f; float *p; } ps[] = {{"--vaug-gain-p", &c->gain_p}, {"--vaug-gamma-p", &c->gamma_p}, {"--vaug-noise-p", &c->noise_p},
                                               {"--vaug-blur-p", &c->blur_p}, {"--vaug-lowres-p", &c->lowres_p}, {"--vaug-quant-p", &c->quant_p}};
    int bad = c->on < 0 || c->on > 1 || !(c->geo_p >= 0 && c->geo_p <= 1) || c->lattice < 1 || c->lattice > 4 || !(c->strength >= 0 && c->strength <= 3);
    for (size_t i = 0; i < sizeof ps / sizeof ps[0]; i++) { if ((v = va_opt(argc, argv, ps[i].f))) *ps[i].p = (float)atof(v); bad |= !(*ps[i].p >= 0 && *ps[i].p <= 1); }
    if (bad) { fprintf(stderr, "--vaug 0|1, --vaug-geo 0..1, --vaug-lattice 1..4, --vaug-strength 0..3, --vaug-*-p 0..1\n"); return 1; }
    return 0;
}
void vwa_cfg_print(const vwa_cfg *c) {
    if (!c->on) { fprintf(stderr, "vwt aug: off\n"); return; }
    fprintf(stderr, "vwt aug: geometric p %.3g (lattice %d), intensity strength %.3g (gain %.2g gamma %.2g noise %.2g blur %.2g lowres %.2g quant %.2g); ",
            c->geo_p, c->lattice, c->strength, c->gain_p, c->gamma_p, c->noise_p, c->blur_p, c->lowres_p, c->quant_p);
    deform_cfg_print(&c->df);
}

struct vwa { deform D; int lattice; double c[3], ext[3], o[3], a[3], b[3], d[3]; };   /* deform works in zyx; the API in xyz */

typedef struct { double (*ct)(void *, const double *); void *ud; const double *o; } va_ctu;
static double va_ct(void *ud, const double x[3]) {   /* crush window coordinates (zyx) -> scan xyz */
    const va_ctu *u = ud; const double p[3] = {u->o[2] + x[2], u->o[1] + x[1], u->o[0] + x[0]};
    return u->ct(u->ud, p);
}
vwa *vwa_begin(const vwa_cfg *c, uint64_t seed, const vws_sample *s, double (*ct_xyz)(void *ud, const double xyz[3]), void *ud) {
    if (!c->on || !(c->geo_p > 0) || va_u(&seed) >= c->geo_p) return nullptr;
    vwa *w = calloc(1, sizeof *w); if (!w) return nullptr;
    w->lattice = c->lattice;
    const int T = s->T, L = s->L;
    double cx[3];
    for (int d = 0; d < 3; d++) cx[d] = s->o[d] + .5 * (T - 1) * (s->a[d] + s->b[d]) + .5 * (L - 1) * s->d[d];
    for (int d = 0; d < 3; d++) {   /* the slab's xyz half extents (corners about the centre) */
        w->ext[d] = .5 * (T - 1) * (fabs(s->a[d]) + fabs(s->b[d])) + .5 * (L - 1) * fabs(s->d[d]) + 1;
        w->c[d] = cx[d]; w->o[d] = s->o[d]; w->a[d] = s->a[d]; w->b[d] = s->b[d]; w->d[d] = s->d[d];
    }
    const double cz[3] = {cx[2], cx[1], cx[0]}, nz[3] = {s->d[2], s->d[1], s->d[0]};
    const int P = (int)ceil(sqrt((double)L * L + 2.0 * T * T)) + 8;
    const double o[3] = {cz[0] - .5 * (P - 1), cz[1] - .5 * (P - 1), cz[2] - .5 * (P - 1)};
    va_ctu u = {ct_xyz, ud, o};
    deform_cfg dc = c->df; if (!(dc.crush_p > 0 || dc.elastic_p > 0)) { free(w); return nullptr; }
    for (int tries = 0; tries < 4; tries++) {   /* (both stages drawn off: redraw, the slab was chosen for geometric aug) */
        if (deform_make(&w->D, &dc, va_next(&seed), cz, nz, P, ct_xyz ? va_ct : nullptr, &u)) { free(w); return nullptr; }
        if (deform_on(&w->D)) return w;
        deform_free(&w->D);
    }
    free(w); return nullptr;
}
void vwa_src(const vwa *w, const double p[3], double q[3]) {
    const double pz[3] = {p[2], p[1], p[0]}; double qz[3];
    deform_pull(&w->D, pz, qz, nullptr);
    q[0] = qz[2]; q[1] = qz[1]; q[2] = qz[0];
}
double vwa_jdet(const vwa *w, const double p[3]) {
    const double pz[3] = {p[2], p[1], p[0]}; double qz[3], j; deform_pull(&w->D, pz, qz, &j); return j;   /* (zyx <-> xyz is a reflection
                                                                                                           applied on both sides: det unchanged) */
}
/* lattice nodes along one axis: 0, st, 2 st, ..., n - 1 (the last always a node) */
static int va_nodes(int n, int st, int *pos) { int m = 0; for (int x = 0; x < n - 1; x += st) pos[m++] = x; pos[m++] = n - 1; return m; }
void vwa_src_lattice(const vwa *w, int T, int L, float *q) {
    const int st = w->lattice < 1 ? 1 : w->lattice;
    int *pi = malloc(sizeof(int) * (T + 1)), *pk = malloc(sizeof(int) * (L + 1));
    const int ni = va_nodes(T, st, pi), nk = va_nodes(L, st, pk);
    double *nd = malloc(sizeof(double) * 3 * (size_t)ni * ni * nk);
    for (int a = 0; a < ni; a++) for (int b = 0; b < ni; b++) for (int c = 0; c < nk; c++) {
        double p[3]; for (int e = 0; e < 3; e++) p[e] = w->o[e] + pi[a] * w->a[e] + pi[b] * w->b[e] + pk[c] * w->d[e];
        vwa_src(w, p, nd + 3 * (((size_t)a * ni + b) * nk + c));
    }
    int *ci = malloc(sizeof(int) * (T > L ? T : L));
    /* (each voxel: its node cell (lower node index) and fractions) */
    int *ki = malloc(sizeof(int) * L); double *kf = malloc(sizeof(double) * L), *jf = malloc(sizeof(double) * T); int *ji = malloc(sizeof(int) * T);
    for (int x = 0, m = 0; x < T; x++) { while (m + 2 < ni && pi[m + 1] <= x) m++; ji[x] = m; jf[x] = (double)(x - pi[m]) / (pi[m + 1] - pi[m]); }
    for (int x = 0, m = 0; x < L; x++) { while (m + 2 < nk && pk[m + 1] <= x) m++; ki[x] = m; kf[x] = (double)(x - pk[m]) / (pk[m + 1] - pk[m]); }
    for (int i = 0; i < T; i++) for (int j = 0; j < T; j++) for (int k = 0; k < L; k++) {
        const int a = ji[i], b = ji[j], c = ki[k]; const double fa = jf[i], fb = jf[j], fc = kf[k];
        float *o = q + 3 * (((size_t)i * T + j) * L + k);
        for (int e = 0; e < 3; e++) {
#define ND(A, B, C) nd[3 * (((size_t)(A) * ni + (B)) * nk + (C)) + e]
            const double v = (1 - fa) * ((1 - fb) * ((1 - fc) * ND(a, b, c) + fc * ND(a, b, c + 1)) + fb * ((1 - fc) * ND(a, b + 1, c) + fc * ND(a, b + 1, c + 1)))
                           + fa * ((1 - fb) * ((1 - fc) * ND(a + 1, b, c) + fc * ND(a + 1, b, c + 1)) + fb * ((1 - fc) * ND(a + 1, b + 1, c) + fc * ND(a + 1, b + 1, c + 1)));
#undef ND
            o[e] = (float)v;
        }
    }
    free(pi); free(pk); free(nd); free(ci); free(ki); free(kf); free(jf); free(ji);
}
void vwa_box(const vwa *w, int T, int L, double lo[3], double hi[3]) {
    (void)T; (void)L;
    const double ez[3] = {w->ext[2], w->ext[1], w->ext[0]}; double l[3], h[3];
    deform_bounds(&w->D, ez, l, h);
    for (int d = 0; d < 3; d++) { lo[d] = l[2 - d]; hi[d] = h[2 - d]; }
}
void vwa_end(vwa *w) { if (!w) return; deform_free(&w->D); free(w); }

/* ---- intensity (im only, where va) ---- */
static void va_blur1(float *f, int T, int L, int axis, double sg) {   /* separable Gaussian along one slab axis (clamped ends) */
    const int R = (int)ceil(2.5 * sg); double k[32]; double sw = 0; for (int r = -R; r <= R; r++) sw += k[r + R] = exp(-r * r / (2 * sg * sg)); for (int r = 0; r <= 2 * R; r++) k[r] /= sw;
    const int n = axis == 2 ? L : T; const size_t st = axis == 0 ? (size_t)T * L : axis == 1 ? (size_t)L : 1;
    float *line = malloc(sizeof(float) * n), *out = malloc(sizeof(float) * n);
    const int n1 = axis == 0 ? T : T, n2 = axis == 2 ? T : L;   /* the two other axes' sizes */
    for (int u = 0; u < n1; u++) for (int v = 0; v < n2; v++) {
        size_t base = axis == 0 ? (size_t)u * L + v : axis == 1 ? (size_t)u * T * L + v : ((size_t)u * T + v) * L;
        for (int x = 0; x < n; x++) line[x] = f[base + x * st];
        for (int x = 0; x < n; x++) { double s = 0; for (int r = -R; r <= R; r++) { int y = x + r; y = y < 0 ? 0 : y >= n ? n - 1 : y; s += k[r + R] * line[y]; } out[x] = (float)s; }
        for (int x = 0; x < n; x++) f[base + x * st] = out[x];
    }
    free(line); free(out);
}
void vwa_intensity(const vwa_cfg *c, uint64_t seed, uint8_t *im, const uint8_t *va, int T, int L) {
    if (!c->on || !(c->strength > 0)) return;
    const size_t n = (size_t)T * T * L; const double S = c->strength;
    const int gain = va_u(&seed) < c->gain_p, gam = va_u(&seed) < c->gamma_p, noi = va_u(&seed) < c->noise_p, blur = va_u(&seed) < c->blur_p,
              low = va_u(&seed) < c->lowres_p, qu = va_u(&seed) < c->quant_p;
    if (!(gain || gam || noi || blur || low || qu)) return;
    float *f = malloc(sizeof(float) * n); if (!f) return;
    for (size_t i = 0; i < n; i++) f[i] = im[i];
    if (blur) { const double sg = .5 + va_u(&seed) * S; for (int ax = 0; ax < 3; ax++) va_blur1(f, T, L, ax, sg); }
    if (low) {   /* 2^3 box mean, nearest back (a coarser scan) */
        for (int i = 0; i + 1 < T; i += 2) for (int j = 0; j + 1 < T; j += 2) for (int k = 0; k + 1 < L; k += 2) {
            double s = 0; for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) for (int e = 0; e < 2; e++) s += f[((size_t)(i + a) * T + j + b) * L + k + e];
            for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) for (int e = 0; e < 2; e++) f[((size_t)(i + a) * T + j + b) * L + k + e] = (float)(s / 8);
        }
    }
    const double g = gain ? exp((2 * va_u(&seed) - 1) * .3 * S) : 1, off = gain ? (2 * va_u(&seed) - 1) * 20 * S : 0;
    const double ga = gam ? exp((2 * va_u(&seed) - 1) * .4 * S) : 1, sn = noi ? (2 + 8 * va_u(&seed)) * S : 0;
    const int qs = qu ? 2 + (int)(va_u(&seed) * 6 * S) : 1;
    uint64_t ns = va_next(&seed);
    for (size_t i = 0; i < n; i++) {
        if (!va[i]) { im[i] = 0; continue; }
        double v = f[i];
        if (gam) v = 255 * pow(fmax(v, 0) / 255, ga);
        v = g * v + off;
        if (noi) v += sn * va_gauss(&ns);
        if (qu) v = qs * floor(v / qs + .5);
        im[i] = (uint8_t)(v < 1 ? 1 : v > 255 ? 255 : lrint(v));   /* (inside the volume never 0: 0 means outside) */
    }
    free(f);
}
