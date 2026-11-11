/* ufsm sheet-pbr: the papyrus surface of a published segment rendered as a physical object, to see what ink looks like on a sheet
   without an ink model (2026-11-11, DESIGN.md "Physically based sheet rendering").

   ufsm sheet-pbr <tifxyz-dir> <out-dir> --ct ROOT KEY --crop U0,V0,N [--ss 1] [--depth 16] [--step 0.25] [--interp tri|cubic]
                  [--outward 0|1|-1] [--lights 12] [--elev 15] [--relief 1] [--rough 0.35] [--threads 8] [--cache DIR]

   The crop is N x N pixels of the segment's own full-resolution (u, v) frame (the frame of its published 2D renders and ink maps:
   pixel = grid index / meta scale, one level-0 voxel per pixel), rendered at --ss samples per pixel per axis.
   1. Surface: the tifxyz grid interpolated with Catmull-Rom splines (C1: no 48 um grid-cell creases in the normals or heights),
      outward normal (away from the umbilicus; --outward 0 = the crop's own convexity: a turn is concave towards the axis).
   2. Profile per sample along the outward normal, every --step voxels over +- --depth around the sheet (trilinear or tricubic CT),
      the sheet found in two passes (pass 1 on an 8-pixel lattice: the profile maximum near the mesh, median-filtered; pass 2 per
      sample: the maximum within +- 3 voxels of that map). Edges at the half level between the peak and the void minimum on each side,
      linearly interpolated between samples (sub-voxel): recto = the inner face (towards the umbilicus; faces.h convention),
      verso = the outer face.
   3. Maps (float32 .f32, row-major W x W, W = N ss; map.json lists them): recto offset r and verso offset v along the outward normal
      (voxels), thickness, peak CT, near-surface CT (mean over the first 2 voxels inside each face), the flat CT at the mesh, the
      max over +-3 voxels (ufsm render's default), height H = -r (towards a viewer on the recto side), its high-pass (H minus a
      Gaussian, sigma 6 px), local roughness (std of the high-pass, 7 x 7 px), curvature (Laplacian of a Gaussian, sigma 1.5 px).
   4. Shading of the recto height field seen from the recto side (view along -outward): Lambert + GGX microfacet specular (dielectric,
      F0 0.04, --rough alpha), raking lights at --elev degrees from --lights azimuths with cast shadows (height-field ray march),
      horizon-based ambient occlusion (16 directions, 24 px); images: light_k.pgm, rti_max / rti_min / rti_range.pgm, ao.pgm,
      normals.ppm, pbr.pgm (light at 135 deg, flat albedo, x AO), pbr_density.pgm (albedo from the near-surface CT). */
#include "imgops.h"
#include "reg.h"
#include "store.h"
#include "zarr3.h"
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static const char *popt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}

/* ---- rows in parallel: fn(ctx, y) for y in [0, n) on nth threads ---- */
typedef struct { void (*fn)(void *, int); void *ctx; int n; atomic_int next; } prow;
static void *prow_run(void *a) { prow *R = a; for (int y; (y = atomic_fetch_add(&R->next, 1)) < R->n;) R->fn(R->ctx, y); return nullptr; }
static int g_nth = 8;
static void prows(int n, void (*fn)(void *, int), void *ctx) {
    prow R = {fn, ctx, n, 0}; pthread_t th[64]; const int nt = g_nth < 64 ? g_nth : 64;
    for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, prow_run, &R);
    for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
}

/* ---- the surface: Catmull-Rom on the tifxyz grid ---- */
typedef struct { const float *A[3]; int w, h; double gs; } sgrid;   /* A: x, y, z grids (level-0 voxels); gs: pixels per grid step */
static inline void cr_w(double t, double w[4], double d[4]) {
    const double t2 = t * t, t3 = t2 * t;
    w[0] = 0.5 * (-t3 + 2 * t2 - t); w[1] = 0.5 * (3 * t3 - 5 * t2 + 2); w[2] = 0.5 * (-3 * t3 + 4 * t2 + t); w[3] = 0.5 * (t3 - t2);
    d[0] = 0.5 * (-3 * t2 + 4 * t - 1); d[1] = 0.5 * (9 * t2 - 10 * t); d[2] = 0.5 * (-9 * t2 + 8 * t + 1); d[3] = 0.5 * (3 * t2 - 2 * t);
}
/* point P (z, y, x) and the derivatives per pixel along u (columns) and v (rows) at pixel (pu, pv); 0 when a tap is invalid */
static int surf_at(const sgrid *G, double pu, double pv, double P[3], double Du[3], double Dv[3]) {
    const double gu = pu / G->gs, gv = pv / G->gs; const int c = (int)floor(gu), r = (int)floor(gv);
    if (c < 1 || r < 1 || c + 2 >= G->w || r + 2 >= G->h) return 0;
    double wu[4], du[4], wv[4], dv[4]; cr_w(gu - c, wu, du); cr_w(gv - r, wv, dv);
    for (int a = 0; a < 3; a++) { P[a] = Du[a] = Dv[a] = 0; }
    for (int j = 0; j < 4; j++) for (int i = 0; i < 4; i++) {
        const size_t k = (size_t)(r - 1 + j) * G->w + (c - 1 + i);
        if (!(G->A[2][k] >= 0)) return 0;
        for (int a = 0; a < 3; a++) {   /* a: 0 z, 1 y, 2 x */
            const double q = G->A[2 - a][k];
            P[a] += wv[j] * wu[i] * q; Du[a] += wv[j] * du[i] * q / G->gs; Dv[a] += dv[j] * wu[i] * q / G->gs;
        }
    }
    return 1;
}
static void cross3(const double a[3], const double b[3], double c[3]) { c[0] = a[1] * b[2] - a[2] * b[1]; c[1] = a[2] * b[0] - a[0] * b[2]; c[2] = a[0] * b[1] - a[1] * b[0]; }
static void norm3(double a[3]) { const double l = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); if (l > 0) for (int k = 0; k < 3; k++) a[k] /= l; }

/* ---- the CT box ---- */
typedef struct { uint8_t *v; int64_t o[3], n[3]; } cbox;
static inline double cb_at(const cbox *B, int64_t z, int64_t y, int64_t x) {
    z -= B->o[0]; y -= B->o[1]; x -= B->o[2];
    if (z < 0 || y < 0 || x < 0 || z >= B->n[0] || y >= B->n[1] || x >= B->n[2]) return 0;
    return B->v[((size_t)z * B->n[1] + y) * B->n[2] + x];
}
static double cb_tri(const cbox *B, double z, double y, double x) {
    const int64_t z0 = (int64_t)floor(z), y0 = (int64_t)floor(y), x0 = (int64_t)floor(x); const double fz = z - z0, fy = y - y0, fx = x - x0;
    double s = 0;
    for (int c = 0; c < 8; c++) s += (c & 4 ? fz : 1 - fz) * (c & 2 ? fy : 1 - fy) * (c & 1 ? fx : 1 - fx) * cb_at(B, z0 + (c >> 2 & 1), y0 + (c >> 1 & 1), x0 + (c & 1));
    return s;
}
static double cb_cubic(const cbox *B, double z, double y, double x) {
    const int64_t z0 = (int64_t)floor(z), y0 = (int64_t)floor(y), x0 = (int64_t)floor(x); double wz[4], wy[4], wx[4], d[4];
    cr_w(z - z0, wz, d); cr_w(y - y0, wy, d); cr_w(x - x0, wx, d);
    double s = 0;
    for (int a = 0; a < 4; a++) { double sa = 0;
        for (int b = 0; b < 4; b++) { double sb = 0;
            for (int c = 0; c < 4; c++) sb += wx[c] * cb_at(B, z0 - 1 + a, y0 - 1 + b, x0 - 1 + c);
            sa += wy[b] * sb; }
        s += wz[a] * sa; }
    return s;
}

/* ---- the job ---- */
typedef struct {
    sgrid G; cbox B; int cubic, W, ss, K; double step, depth, u0, v0, sgn, win1;
    const rmap *map; double fs;                  /* --reg: the profile scan = the map's fine scan; fs = its voxels per mesh-scan voxel */
    float *tp0, *cr, *cvv;                        /* pass-1 sheet middle, recto and verso offsets per sample (upsampled) */
    float *grr, *grv, *amb;                      /* edge strengths (CT per voxel); runner-up / best of the recto edge */
    float *r, *v, *thk, *pk, *nin, *nout, *flat, *fmax, *valid;
    atomic_int next;
} job;
static double prof(const job *J, const double P[3], const double n[3], double t) {
    const double z = P[0] + t * n[0], y = P[1] + t * n[1], x = P[2] + t * n[2];
    return J->cubic ? cb_cubic(&J->B, z, y, x) : cb_tri(&J->B, z, y, x);
}
static int frame(const job *J, int px, int py, double P[3], double n[3]) {
    double Du[3], Dv[3];
    if (!surf_at(&J->G, J->u0 + (px + 0.5) / J->ss - 0.5, J->v0 + (py + 0.5) / J->ss - 0.5, P, Du, Dv)) return 0;
    cross3(Du, Dv, n); norm3(n); for (int k = 0; k < 3; k++) n[k] *= J->sgn;
    if (J->map) {   /* into the fine scan: the point through the map, the normal through its local Jacobian (finite difference) */
        const double c0[3] = {P[2], P[1], P[0]}, c1[3] = {P[2] + n[2], P[1] + n[1], P[0] + n[0]}; double f0[3], f1[3];
        rmap_c2f(J->map, c0, f0); rmap_c2f(J->map, c1, f1);
        for (int k = 0; k < 3; k++) { P[k] = f0[2 - k]; n[k] = f1[2 - k] - f0[2 - k]; } norm3(n);
    }
    return 1;
}
static void smooth1(const double *f, double *g, int K, double sig) {   /* Gaussian along the profile (sigma in samples) */
    const int R = (int)ceil(3 * sig); double w[64]; double ws = 0; for (int j = -R; j <= R && R < 31; j++) ws += w[j + R] = exp(-0.5 * j * j / (sig * sig));
    for (int k = 0; k < K; k++) { double s = 0, b = 0; for (int j = -R; j <= R; j++) { const int q = k + j; if (q < 0 || q >= K) continue; s += w[j + R] * f[q]; b += w[j + R]; } g[k] = s / b; }
}
/* the sheet around offset c (its middle, from the tracking): edges at the half level between the sheet's peak (within +-win of c)
   and the void minimum on each side, found walking out from c, linearly interpolated (sub-voxel) */
static int sheet(const job *J, const double P[3], const double n[3], double c, double win, double out[8]) {
    const int K = J->K; double f[1024], g[1024]; if (K > 1024) return 0;
    for (int k = 0; k < K; k++) f[k] = prof(J, P, n, c - J->depth + k * J->step);
    smooth1(f, g, K, 0.75 / J->step);
    const int kc = K / 2, kw = (int)(win / J->step + 0.5); int kp = kc;
    for (int k = kc - kw; k <= kc + kw; k++) if (k >= 0 && k < K && g[k] > g[kp]) kp = k;
    const int kv = (int)(0.75 * J->depth / J->step);
    double vin = g[kp], vout = g[kp];
    for (int k = kc; k >= 0 && k >= kc - kv; k--) if (g[k] < vin) vin = g[k];
    for (int k = kc; k < K && k <= kc + kv; k++) if (g[k] < vout) vout = g[k];
    const double hin = 0.5 * (g[kp] + vin), hout = 0.5 * (g[kp] + vout);
    if (g[kc] < fmax(hin, hout)) return 0;   /* the tracked middle is not inside papyrus */
    double ein = NAN, eout = NAN;
    for (int k = kc; k > 0; k--) if (g[k - 1] < hin) { ein = (k - 1) + (hin - g[k - 1]) / (g[k] - g[k - 1] + 1e-12); break; }
    for (int k = kc; k + 1 < K; k++) if (g[k + 1] < hout) { eout = k + (g[k] - hout) / (g[k] - g[k + 1] + 1e-12); break; }
    if (!(ein == ein) || !(eout == eout)) return 0;
    const double r = c - J->depth + ein * J->step, v = c - J->depth + eout * J->step;
    double ni = 0, no = 0; int cn = 0;
    for (double t = 0.25; t <= 2.0 * J->fs; t += 0.25, cn++) { ni += prof(J, P, n, r + t); no += prof(J, P, n, v - t); }
    out[0] = r; out[1] = v; out[2] = v - r; out[3] = g[kp]; out[4] = ni / cn; out[5] = no / cn; out[6] = 0.5 * (r + v);
    return 1;
}
/* pass 2: the faces near their coarse positions (cr, cv: recto / verso offsets from the tracked lattice): the steepest rise of the smoothed
   profile (air -> papyrus, outward) within +-w of cr and the steepest fall within +-w of cv, sub-voxel by a parabola through the
   derivative's extremum */
static int faces(const job *J, const double P[3], const double n[3], double cr, double cv, double w, double out[10]) {
    const double lo = cr - w - 3, hi = cv + w + 3; const int K = (int)((hi - lo) / J->step) + 1; double f[1024], g[1024];
    if (K < 8 || K > 1024) return 0;
    for (int k = 0; k < K; k++) f[k] = prof(J, P, n, lo + k * J->step);
    smooth1(f, g, K, 0.6 / J->step);
    double best = -1e30, worst = 1e30; int kr = -1, kv = -1;
    for (int k = 1; k + 1 < K; k++) { const double t = lo + k * J->step, d = g[k + 1] - g[k - 1];
        if (fabs(t - cr) <= w && d > best) { best = d; kr = k; }
        if (fabs(t - cv) <= w && d < worst) { worst = d; kv = k; } }
    if (kr < 2 || kv < 2 || kr + 2 >= K || kv + 2 >= K || best <= 0 || worst >= 0) return 0;
    double b2 = 0, w2 = 0; const int sep = (int)(1.0 / J->step + 0.5);   /* runner-up extrema >= 1 voxel away: ambiguity */
    for (int k = 2; k + 2 < K; k++) { const double t = lo + k * J->step, d = g[k + 1] - g[k - 1];
        const int lmax = d >= g[k] - g[k - 2] && d >= g[k + 2] - g[k], lmin = d <= g[k] - g[k - 2] && d <= g[k + 2] - g[k];
        if (fabs(t - cr) <= w && abs(k - kr) >= sep && lmax && d > b2) b2 = d;
        if (fabs(t - cv) <= w && abs(k - kv) >= sep && lmin && d < w2) w2 = d; }
    double e[2]; const int kk[2] = {kr, kv};
    for (int s = 0; s < 2; s++) { const int k = kk[s]; const double a = g[k] - g[k - 2], b = g[k + 1] - g[k - 1], c = g[k + 2] - g[k], den = a - 2 * b + c;
        e[s] = lo + (k + (fabs(den) > 1e-9 ? 0.5 * (a - c) / den : 0)) * J->step; }
    if (!(e[1] > e[0])) return 0;
    double ni = 0, no = 0, pk = 0; int cn = 0;
    for (double t = 0.25; t <= 2.0 * J->fs; t += 0.25, cn++) { ni += prof(J, P, n, e[0] + t); no += prof(J, P, n, e[1] - t); }
    for (double t = e[0]; t <= e[1]; t += 0.5) { const double q = prof(J, P, n, t); if (q > pk) pk = q; }
    out[0] = e[0]; out[1] = e[1]; out[2] = e[1] - e[0]; out[3] = pk; out[4] = ni / cn; out[5] = no / cn; out[6] = best; out[7] = -worst; out[8] = b2 / best; out[9] = w2 / worst;
    return 1;
}
/* pass 1: the sheets crossing the normal at one lattice point, as (middle, thickness): runs between void minima of the smoothed profile
   (a minimum counts when it lies below the mean of the profile minus 0.6 of its standard deviation) */
static int sheet_list(const job *J, const double P[3], const double n[3], double *mid, double *gl, double *gr, int maxs) {
    const int K = J->K; double f[1024], g[1024]; if (K > 1024) return 0;
    for (int k = 0; k < K; k++) f[k] = prof(J, P, n, -J->depth + k * J->step);
    smooth1(f, g, K, 1.5 / J->step);
    double mu = 0, s2 = 0; for (int k = 0; k < K; k++) { mu += g[k]; s2 += g[k] * g[k]; } mu /= K; s2 = sqrt(fmax(s2 / K - mu * mu, 0));
    int gap[256], ng = 0; const double thr = mu - 0.6 * s2;
    for (int k = 1; k + 1 < K && ng < 256; k++) if (g[k] < g[k - 1] && g[k] <= g[k + 1] && g[k] < thr) gap[ng++] = k;
    int m = 0;
    for (int i = 0; i + 1 < ng && m < maxs; i++) { const double t = (gap[i + 1] - gap[i]) * J->step; if (t < 6 * J->fs || t > 80 * J->fs) continue; gl[m] = -J->depth + gap[i] * J->step; gr[m] = -J->depth + gap[i + 1] * J->step; mid[m] = 0.5 * (gl[m] + gr[m]); m++; }
    return m;
}
static void *pass2(void *a) {
    job *J = a; const int W = J->W;
    for (int py; (py = atomic_fetch_add(&J->next, 1)) < W;) for (int px = 0; px < W; px++) {
        const size_t i = (size_t)py * W + px; double P[3], n[3], o[10];
        J->valid[i] = 0; J->grr[i] = J->grv[i] = J->amb[i] = NAN; J->r[i] = J->v[i] = J->thk[i] = J->pk[i] = J->nin[i] = J->nout[i] = J->flat[i] = J->fmax[i] = NAN;
        if (!frame(J, px, py, P, n)) continue;
        J->flat[i] = (float)prof(J, P, n, 0); double m = 0; for (double t = -3; t <= 3; t += 0.5) { const double q = prof(J, P, n, t); if (q > m) m = q; } J->fmax[i] = (float)m;
        if (!(J->cr[i] == J->cr[i]) || !faces(J, P, n, J->cr[i], J->cvv[i], fmax(1.0, J->tp0[i]), o)) continue;
        J->grr[i] = (float)(o[6] / (2 * J->step)); J->grv[i] = (float)(o[7] / (2 * J->step)); J->amb[i] = (float)fmax(o[8], o[9]);
        J->r[i] = (float)o[0]; J->v[i] = (float)o[1]; J->thk[i] = (float)o[2]; J->pk[i] = (float)o[3]; J->nin[i] = (float)o[4]; J->nout[i] = (float)o[5]; J->valid[i] = 1;
    }
    return nullptr;
}

/* ---- image helpers ---- */
typedef struct { const float *s; float *d; const double *k; int R, W, pass; } gbctx;
static void gb_row(void *a, int y) {
    const gbctx *C = a; const int W = C->W, R = C->R;
    for (int x = 0; x < W; x++) {
        double s = 0, b = 0;
        for (int j = -R; j <= R; j++) { const int xx = C->pass ? x : x + j, yy = C->pass ? y + j : y; if (xx < 0 || yy < 0 || xx >= W || yy >= W) continue;
            const float q = C->s[(size_t)yy * W + xx]; if (q == q) { s += C->k[j + R] * q; b += C->k[j + R]; } }
        C->d[(size_t)y * W + x] = b > 0 ? (float)(s / b) : NAN;
    }
}
static void gblur(const float *in, float *out, int W, double sigma) {   /* NaN-aware separable Gaussian */
    const int R = (int)ceil(3 * sigma); double *k = malloc((2 * R + 1) * sizeof *k); for (int j = -R; j <= R; j++) k[j + R] = exp(-0.5 * j * j / (sigma * sigma));
    float *t = malloc((size_t)W * W * sizeof *t);
    gbctx a = {in, t, k, R, W, 0}; prows(W, gb_row, &a);
    gbctx b = {t, out, k, R, W, 1}; prows(W, gb_row, &b);
    free(t); free(k);
}
static int fcmp(const void *x, const void *y) { const float p = *(const float *)x, q = *(const float *)y; return (p > q) - (p < q); }
static void pctl(const float *a, size_t n, double lo, double hi, double *plo, double *phi) {
    size_t m = 0; float *s = malloc(n * sizeof *s); for (size_t i = 0; i < n; i += 7) if (a[i] == a[i]) s[m++] = a[i];
    if (!m) { *plo = 0; *phi = 1; free(s); return; }
    qsort(s, m, sizeof *s, fcmp); *plo = s[(size_t)(lo * (m - 1))]; *phi = s[(size_t)(hi * (m - 1))]; if (*phi <= *plo) *phi = *plo + 1e-6; free(s);
}
static int write_pgm(const char *dir, const char *name, const float *a, int W, double lo, double hi) {
    char p[1400]; snprintf(p, sizeof p, "%s/%s.pgm", dir, name); FILE *f = fopen(p, "wb"); if (!f) return -1;
    fprintf(f, "P5\n%d %d\n255\n", W, W); uint8_t *row = malloc(W);
    for (int y = 0; y < W; y++) { for (int x = 0; x < W; x++) { const float q = a[(size_t)y * W + x]; double t = q == q ? (q - lo) / (hi - lo) : 0; row[x] = (uint8_t)(t < 0 ? 0 : t > 1 ? 255 : t * 255 + 0.5); } fwrite(row, 1, W, f); }
    free(row); fclose(f); return 0;
}
static int write_auto(const char *dir, const char *name, const float *a, int W) { double lo, hi; pctl(a, (size_t)W * W, 0.005, 0.995, &lo, &hi); return write_pgm(dir, name, a, W, lo, hi); }
/* shaded images: exposure from the 2 / 98 % points, display gamma 1 / 2.2 */
static int write_shade(const char *dir, const char *name, const float *a, int W) {
    double lo, hi; pctl(a, (size_t)W * W, 0.02, 0.98, &lo, &hi); const size_t n = (size_t)W * W; float *g = malloc(n * sizeof *g);
    for (size_t i = 0; i < n; i++) { const double t = (a[i] - lo) / (hi - lo); g[i] = a[i] == a[i] ? (float)pow(t < 0 ? 0 : t > 1 ? 1 : t, 1 / 2.2) : NAN; }
    const int r = write_pgm(dir, name, g, W, 0, 1); free(g); return r;
}
static int write_f32(const char *dir, const char *name, const float *a, int W, FILE *js, int *first) {
    char p[1400]; snprintf(p, sizeof p, "%s/%s.f32", dir, name); FILE *f = fopen(p, "wb"); if (!f) return -1;
    const size_t n = (size_t)W * W; const int ok = fwrite(a, sizeof *a, n, f) == n; fclose(f);
    fprintf(js, "%s\"%s\"", *first ? "" : ",", name); *first = 0; return ok ? 0 : -1;
}

/* ---- shading ---- */
typedef struct { const float *H; int W; double px; } hfield;   /* H in voxels, px = voxels per pixel */
static inline float hf(const hfield *F, int x, int y) { x = x < 0 ? 0 : x >= F->W ? F->W - 1 : x; y = y < 0 ? 0 : y >= F->W ? F->W - 1 : y; return F->H[(size_t)y * F->W + x]; }
static void hnormal(const hfield *F, int x, int y, double relief, double N[3]) {
    const double dx = (hf(F, x + 1, y) - hf(F, x - 1, y)) / (2 * F->px) * relief, dy = (hf(F, x, y + 1) - hf(F, x, y - 1)) / (2 * F->px) * relief;
    N[0] = -dx; N[1] = -dy; N[2] = 1; norm3(N);
}
static double ggx_spec(const double N[3], const double L[3], double alpha) {   /* view (0, 0, 1); Cook-Torrance with GGX D, Smith G, Schlick F */
    const double V[3] = {0, 0, 1}; double Hh[3] = {L[0] + V[0], L[1] + V[1], L[2] + V[2]}; norm3(Hh);
    const double nl = fmax(N[0] * L[0] + N[1] * L[1] + N[2] * L[2], 0), nv = fmax(N[2], 1e-4), nh = fmax(N[0] * Hh[0] + N[1] * Hh[1] + N[2] * Hh[2], 0), vh = fmax(Hh[2], 0);
    if (nl <= 0) return 0;
    const double a2 = alpha * alpha, dd = nh * nh * (a2 - 1) + 1, D = a2 / (M_PI * dd * dd);
    const double k = alpha / 2, G = (nl / (nl * (1 - k) + k)) * (nv / (nv * (1 - k) + k)), Fr = 0.04 + 0.96 * pow(1 - vh, 5);
    return D * G * Fr / (4 * nl * nv) * nl;
}
static int shadowed(const hfield *F, int x, int y, const double L[3], double relief) {
    const double lxy = hypot(L[0], L[1]); if (lxy < 1e-6) return 0;
    const double tz = L[2] / lxy, h0 = hf(F, x, y) * relief;   /* rise per voxel of horizontal travel */
    for (int s = 1; s < 256; s++) {
        const double qx = x + s * L[0] / lxy, qy = y + s * L[1] / lxy; if (qx < 0 || qy < 0 || qx >= F->W || qy >= F->W) return 0;
        if (hf(F, (int)qx, (int)qy) * relief > h0 + s * F->px * tz) return 1;
    }
    return 0;
}

/* ---- row workers ---- */
/* mesh mode, lattice: the recto face = the steepest rise (air -> papyrus, outward) of the smoothed profile within [ra, rb] voxels of the
   mesh, the verso face = the steepest fall within [recto + 4, recto + 40] */
typedef struct { job *J; float *lr, *lv; int nc, L8; double ra, rb; } mlctx;
static void ml_row(void *a, int b) {
    const mlctx *C = a; job *J = C->J; const double lo = C->ra - 2 * J->fs, hi = C->rb + 44 * J->fs; const int K = (int)((hi - lo) / J->step) + 1;
    for (int i = 0; i < C->nc; i++) { const size_t q = (size_t)b * C->nc + i; C->lr[q] = C->lv[q] = NAN; double P[3], n[3], f[1024], g[1024];
        const int x = i * C->L8 < J->W ? i * C->L8 : J->W - 1, y = b * C->L8 < J->W ? b * C->L8 : J->W - 1;
        if (K > 1024 || !frame(J, x, y, P, n)) continue;
        for (int k = 0; k < K; k++) f[k] = prof(J, P, n, lo + k * J->step);
        smooth1(f, g, K, 1.25 * J->fs / J->step);
        double best = 0; int kr = -1; for (int k = 1; k + 1 < K; k++) { const double t = lo + k * J->step; if (t < C->ra || t > C->rb) continue; const double d = g[k + 1] - g[k - 1]; if (d > best) { best = d; kr = k; } }
        if (kr < 0) continue; const double r = lo + kr * J->step; double worst = 0; int kv = -1;
        for (int k = 1; k + 1 < K; k++) { const double t = lo + k * J->step; if (t < r + 4 * J->fs || t > r + 40 * J->fs) continue; const double d = g[k + 1] - g[k - 1]; if (d < worst) { worst = d; kv = k; } }
        C->lr[q] = (float)r; if (kv >= 0) C->lv[q] = (float)(lo + kv * J->step); }
}
/* robust smoothing of a lattice map: Gaussian (sigma cells) over the kept points, points further than tol from it dropped, 4 rounds */
static void robust_smooth(float *m, int nc, double sigma, double tol) {
    const size_t N = (size_t)nc * nc; float *s = malloc(N * sizeof *s), *w = malloc(N * sizeof *w); uint8_t *keep = malloc(N);
    for (size_t q = 0; q < N; q++) keep[q] = m[q] == m[q];
    const int R = (int)ceil(3 * sigma);
    for (int it = 0; it < 4; it++) {
        for (int b = 0; b < nc; b++) for (int a = 0; a < nc; a++) { double sv = 0, sw = 0;
            for (int j = -R; j <= R; j++) for (int i = -R; i <= R; i++) { const int bb = b + j, aa = a + i; if (bb < 0 || aa < 0 || bb >= nc || aa >= nc) continue; const size_t q = (size_t)bb * nc + aa;
                if (!keep[q]) continue; const double ww = exp(-0.5 * (i * i + j * j) / (sigma * sigma)); sv += ww * m[q]; sw += ww; }
            s[(size_t)b * nc + a] = sw > 0.5 ? (float)(sv / sw) : NAN; w[(size_t)b * nc + a] = (float)sw; }
        int ch = 0; for (size_t q = 0; q < N; q++) { const int k = m[q] == m[q] && s[q] == s[q] && fabsf(m[q] - s[q]) <= tol; if (k != keep[q]) ch = 1; keep[q] = (uint8_t)k; }
        if (!ch) break;
    }
    int nk = 0; for (size_t q = 0; q < N; q++) nk += keep[q];
    fprintf(stderr, "sheet-pbr: robust face: %d of %zu lattice points kept\n", nk, N);
    memcpy(m, s, N * sizeof *m); free(s); free(w); free(keep);
}
typedef struct { job *J; float *cand, *cgl, *cgr; uint8_t *ncand; int nc, L8; } p1ctx;
static void p1_row(void *a, int b) {
    const p1ctx *C = a; job *J = C->J;
    for (int i = 0; i < C->nc; i++) { double P[3], n[3], mid[16], gl[16], gr[16]; const int x = i * C->L8 < J->W ? i * C->L8 : J->W - 1, y = b * C->L8 < J->W ? b * C->L8 : J->W - 1;
        const size_t q = (size_t)b * C->nc + i; const int m = frame(J, x, y, P, n) ? sheet_list(J, P, n, mid, gl, gr, 16) : 0;
        C->ncand[q] = (uint8_t)m; for (int k = 0; k < m; k++) { C->cand[q * 16 + k] = (float)mid[k]; C->cgl[q * 16 + k] = (float)gl[k]; C->cgr[q * 16 + k] = (float)gr[k]; } }
}
typedef struct { const hfield *F; float *ao; double relief; int ss; } aoctx;
static void ao_row(void *a, int y) {
    const aoctx *C = a; const hfield *F = C->F;
    for (int x = 0; x < F->W; x++) {   /* horizon angles over 16 directions, 24 px */
        double s = 0; const double h0 = hf(F, x, y) * C->relief;
        for (int d = 0; d < 16; d++) { const double an = d * M_PI / 8, ca = cos(an), sa = sin(an); double best = 0;
            for (int r = 1; r <= 24 * C->ss; r++) { const double dh = hf(F, (int)(x + r * ca), (int)(y + r * sa)) * C->relief - h0, ang = atan2(dh, r * F->px); if (ang > best) best = ang; }
            s += sin(best); }
        C->ao[(size_t)y * F->W + x] = (float)(1 - s / 16); }
}
typedef struct { const hfield *F; const double *Lv; double relief, alpha, dlo, dhi; int pbrpass; const float *nin, *ao; float *img, *rmax, *rmin, *pbr, *pbd; } ltctx;
static void lt_row(void *a, int y) {
    const ltctx *C = a; const hfield *F = C->F; const double *Lv = C->Lv;
    for (int x = 0; x < F->W; x++) { const size_t i = (size_t)y * F->W + x; double N[3]; hnormal(F, x, y, C->relief, N);
        const double nlv = fmax(N[0] * Lv[0] + N[1] * Lv[1] + N[2] * Lv[2], 0), sh = shadowed(F, x, y, Lv, C->relief) ? 0.15 : 1, sp = ggx_spec(N, Lv, C->alpha);
        const double c = sh * (0.8 * nlv + sp) + 0.05; C->img[i] = (float)c; if (c > C->rmax[i]) C->rmax[i] = (float)c; if (c < C->rmin[i]) C->rmin[i] = (float)c;
        if (C->pbrpass) { C->pbr[i] = (float)(c * C->ao[i]);
            const double alb = C->nin[i] == C->nin[i] ? 0.35 + 0.65 * fmin(fmax((C->nin[i] - C->dlo) / (C->dhi - C->dlo), 0), 1) : 0.35;
            C->pbd[i] = (float)((sh * (alb * nlv + sp) + 0.05 * alb) * C->ao[i]); } }
}

/* a lattice map (nc x nc, L8 pixels apart; NaN = untracked): 5 x 5 median, holes filled from neighbours, bilinear to W x W */
static float *lat_up(const float *c1, int nc, int L8, int W) {
    float *c2 = malloc((size_t)nc * nc * sizeof *c2), *c3 = malloc((size_t)nc * nc * sizeof *c3), *o = malloc((size_t)W * W * sizeof *o);
    for (int b = 0; b < nc; b++) for (int a = 0; a < nc; a++) { float w[25]; int m = 0;
        for (int j = -2; j <= 2; j++) for (int i = -2; i <= 2; i++) { const int bb = b + j, aa = a + i; if (bb < 0 || aa < 0 || bb >= nc || aa >= nc) continue; const float q = c1[(size_t)bb * nc + aa]; if (q == q) w[m++] = q; }
        for (int s = 1; s < m; s++) for (int t = s; t > 0 && w[t - 1] > w[t]; t--) { const float q = w[t]; w[t] = w[t - 1]; w[t - 1] = q; }
        c2[(size_t)b * nc + a] = m >= 5 ? w[m / 2] : NAN; }
    for (int it = 0; it < 2 * nc; it++) { int ch = 0;
        for (int b = 0; b < nc; b++) for (int a = 0; a < nc; a++) { const size_t q = (size_t)b * nc + a; c3[q] = c2[q]; if (c2[q] == c2[q]) continue; double s = 0; int m = 0;
            for (int e = 0; e < 4; e++) { const int aa = a + (e == 0) - (e == 1), bb = b + (e == 2) - (e == 3); if (aa < 0 || bb < 0 || aa >= nc || bb >= nc) continue; const float v = c2[(size_t)bb * nc + aa]; if (v == v) { s += v; m++; } }
            if (m) { c3[q] = (float)(s / m); ch = 1; } }
        memcpy(c2, c3, (size_t)nc * nc * sizeof *c2); if (!ch) break; }
    for (int y = 0; y < W; y++) for (int x = 0; x < W; x++) { const double gx = (double)x / L8, gy = (double)y / L8; const int a = (int)gx, b = (int)gy; const double fx = gx - a, fy = gy - b;
        const float q00 = c2[(size_t)b * nc + a], q01 = c2[(size_t)b * nc + a + 1], q10 = c2[(size_t)(b + 1) * nc + a], q11 = c2[(size_t)(b + 1) * nc + a + 1];
        o[(size_t)y * W + x] = (float)((1 - fy) * ((1 - fx) * q00 + fx * q01) + fy * ((1 - fx) * q10 + fx * q11)); }
    free(c2); free(c3); return o;
}

/* ---- the recto face as one continuous surface: dynamic programming over the offset ----
   Evidence E(pixel, k): the rise of the smoothed profile (air -> papyrus, outward) at t_k = cr + (k - KD) step, normalised per pixel to
   its maximum, cost 1 - E. Row pass: Viterbi along every row, transition cost lam |dt| per pixel with |dt| <= smax voxels (a face cannot
   step more than smax between neighbouring samples); column pass: Viterbi along every column on cost + mu |t - t_row|. */
typedef struct { job *J; float *E; int KD; } dpectx;
static void dpe_row(void *a, int y) {
    const dpectx *C = a; job *J = C->J; const int W = J->W, KE = 2 * C->KD + 1, pad = (int)(2.0 / J->step);
    for (int x = 0; x < W; x++) { const size_t i = (size_t)y * W + x; float *e = C->E + i * KE; double P[3], n[3], f[1024], g[1024];
        for (int k = 0; k < KE; k++) e[k] = 0;
        if (!(J->cr[i] == J->cr[i]) || !frame(J, x, y, P, n)) continue;
        const int K = KE + 2 * pad; const double lo = J->cr[i] - (C->KD + pad) * J->step;
        for (int k = 0; k < K; k++) f[k] = prof(J, P, n, lo + k * J->step);
        smooth1(f, g, K, 0.6 / J->step);
        double mx = 0; for (int k = 0; k < KE; k++) { const int q = k + pad; const double d = g[q + 1] - g[q - 1]; e[k] = (float)(d > 0 ? d : 0); if (d > mx) mx = d; }
        if (mx > 0) for (int k = 0; k < KE; k++) e[k] /= (float)mx; }
}
typedef struct { const job *J; const float *E; int KD; double lam, smax, mu; const float *prior; float *out; int col; } dpctx;
static void dp_line(void *a, int l) {   /* one row (col 0) or column (col 1) */
    const dpctx *C = a; const job *J = C->J; const int W = J->W, KE = 2 * C->KD + 1; const double st = J->step, px = J->fs / J->ss;
    float *D = malloc((size_t)W * KE * sizeof *D); int16_t *B = malloc((size_t)W * KE * sizeof *B);
    for (int j = 0; j < W; j++) { const size_t i = C->col ? (size_t)j * W + l : (size_t)l * W + j; const float *e = C->E + i * KE; const double c0 = J->cr[i];
        const size_t ip = j ? (C->col ? (size_t)(j - 1) * W + l : (size_t)l * W + j - 1) : 0; const double cp = j ? J->cr[ip] : 0;
        for (int k = 0; k < KE; k++) { const double t = c0 + (k - C->KD) * st;
            double u = (c0 == c0) ? 1 - e[k] : 0; if (C->prior && C->prior[i] == C->prior[i]) u += C->mu * fabs(t - C->prior[i]);
            if (!j || !(c0 == c0) || !(cp == cp)) { D[(size_t)j * KE + k] = (float)u; B[(size_t)j * KE + k] = -1; continue; }
            double best = 1e30; int bk = -1;
            for (int q = 0; q < KE; q++) { const double tq = cp + (q - C->KD) * st, dt = fabs(t - tq); if (dt > C->smax * px + 1e-9) continue;
                const double v = D[(size_t)(j - 1) * KE + q] + C->lam * dt / px; if (v < best) { best = v; bk = q; } }
            if (bk < 0) { best = 1e30; for (int q = 0; q < KE; q++) if (D[(size_t)(j - 1) * KE + q] < best) { best = D[(size_t)(j - 1) * KE + q]; bk = q; } best += 4; }   /* a break */
            D[(size_t)j * KE + k] = (float)(best + u); B[(size_t)j * KE + k] = (int16_t)bk; } }
    int k = 0; for (int q = 1; q < KE; q++) if (D[(size_t)(W - 1) * KE + q] < D[(size_t)(W - 1) * KE + k]) k = q;
    for (int j = W - 1; j >= 0; j--) { const size_t i = C->col ? (size_t)j * W + l : (size_t)l * W + j;
        C->out[i] = J->cr[i] == J->cr[i] ? (float)(J->cr[i] + (k - C->KD) * st) : NAN; const int b = B[(size_t)j * KE + k]; if (b >= 0) k = b; }
    free(D); free(B);
}

/* NaN-aware median over a (2R + 1)^2 window, rows in parallel */
typedef struct { const float *in; float *out; int W, R; } mdctx;
static void md_row(void *a, int y) {
    const mdctx *C = a; const int W = C->W, R = C->R; float w[441];
    for (int x = 0; x < W; x++) { int m = 0;
        for (int j = -R; j <= R; j++) for (int i = -R; i <= R; i++) { const int yy = y + j, xx = x + i; if (yy < 0 || xx < 0 || yy >= W || xx >= W) continue; const float q = C->in[(size_t)yy * W + xx]; if (q == q) w[m++] = q; }
        if (!m) { C->out[(size_t)y * W + x] = NAN; continue; }
        const int h = m / 2; int lo = 0, hi = m - 1;   /* quickselect */
        while (lo < hi) { const float pv = w[(lo + hi) / 2]; int i = lo, j = hi; while (i <= j) { while (w[i] < pv) i++; while (w[j] > pv) j--; if (i <= j) { const float t = w[i]; w[i] = w[j]; w[j] = t; i++; j--; } } if (h <= j) hi = j; else if (h >= i) lo = i; else break; }
        C->out[(size_t)y * W + x] = w[h]; }
}
static void median2d(const float *in, float *out, int W, int R) { mdctx C = {in, out, W, R < 10 ? R : 10}; prows(W, md_row, &C); }

static rvol *regfv;   /* --reg: the fine volume */
/* weak: the map loader lives in src/reg_cli.c (the ufsm binary); tests link without it */
__attribute__((weak)) const rmap *reg_map_open(const char *path, const char *cache, rvol **fv, double *coarse_um, double *fine_um);

int cmd_sheet_pbr(int argc, char **argv) {
    const char *ctr = nullptr, *ctk = nullptr; long long cu = -1, cv = -1, cn = 0;
    for (int i = 1; i + 2 < argc; i++) if (!strcmp(argv[i], "--ct")) { ctr = argv[i + 1]; ctk = argv[i + 2]; }
    if (argc < 4 || (!ctr && !*popt(argc, argv, "--reg", "")) || sscanf(popt(argc, argv, "--crop", ""), "%lld,%lld,%lld", &cu, &cv, &cn) != 3 || cn < 16) {
        fprintf(stderr, "usage: ufsm sheet-pbr <tifxyz-dir> <out-dir> --ct ROOT KEY --crop U0,V0,N [--ss 1] [--depth 16] [--step 0.25] [--interp tri|cubic]\n"
                        "       [--reg MAP.json (profiles from the map's fine scan; --ct not needed)] [--outward 0|1|-1] [--lights 12] [--elev 15] [--relief 1] [--rough 0.35] [--threads 8] [--cache DIR]\n");
        return 2;
    }
    const char *src = argv[2], *out = argv[3]; mkdir(out, 0755);
    job J = {0}; J.ss = atoi(popt(argc, argv, "--ss", "1")); J.depth = atof(popt(argc, argv, "--depth", "16")); J.step = atof(popt(argc, argv, "--step", "0.25"));
    J.fs = 1;
    if (*popt(argc, argv, "--reg", "")) {   /* profiles from the map's fine scan (ufsm reg map JSON: coarse = the mesh's scan) */
        rvol *fv = nullptr; double cum = 0, fum = 0; J.map = reg_map_open ? reg_map_open(popt(argc, argv, "--reg", ""), popt(argc, argv, "--cache", nullptr), &fv, &cum, &fum) : nullptr;
        if (!J.map || !(cum > 0) || !(fum > 0)) { fprintf(stderr, "sheet-pbr: cannot open the map %s\n", popt(argc, argv, "--reg", "")); return 1; }
        J.fs = cum / fum; regfv = fv;
        if (!*popt(argc, argv, "--depth", "")) J.depth *= J.fs;
        fprintf(stderr, "sheet-pbr: profiles from the fine scan of %s (%g um; %.4f fine voxels per mesh voxel); offsets in fine voxels\n", popt(argc, argv, "--reg", ""), fum, J.fs);
    }
    J.cubic = !strcmp(popt(argc, argv, "--interp", "tri"), "cubic"); J.win1 = atof(popt(argc, argv, "--win", "6")); J.u0 = cu; J.v0 = cv; J.W = (int)(cn * J.ss); J.K = (int)(2 * J.depth / J.step) + 1;
    const int nth = atoi(popt(argc, argv, "--threads", "8")); g_nth = nth > 0 ? nth : 1; const int nl = atoi(popt(argc, argv, "--lights", "12"));
    const double elev = atof(popt(argc, argv, "--elev", "15")) * M_PI / 180, relief = atof(popt(argc, argv, "--relief", "1")), alpha = atof(popt(argc, argv, "--rough", "0.35"));
    if (J.ss < 1 || J.ss > 8 || !(J.step > 0) || J.K > 1024 || nl < 1 || nl > 64) { fprintf(stderr, "sheet-pbr: bad --ss / --step / --depth / --lights\n"); return 2; }
    /* the grid */
    char p[1400]; static const char *nm[3] = {"x", "y", "z"}; float *A[3]; int gw = 0, gh = 0;
    for (int d = 0; d < 3; d++) { int w, h; snprintf(p, sizeof p, "%s/%s.tif", src, nm[d]); A[d] = img_read_f32(p, &w, &h); if (!A[d] || (d && (w != gw || h != gh))) { fprintf(stderr, "sheet-pbr: cannot read %s\n", p); return 1; } gw = w; gh = h; }
    snprintf(p, sizeof p, "%s/meta.json", src); FILE *mf = fopen(p, "rb"); char mb[8192] = {0}; if (mf) { size_t q = fread(mb, 1, sizeof mb - 1, mf); (void)q; fclose(mf); }
    const char *sq = strstr(mb, "\"scale\""); const char *sb = sq ? strchr(sq, '[') : nullptr; const double scale = sb ? atof(sb + 1) : 0;
    if (!(scale > 0)) { fprintf(stderr, "sheet-pbr: %s/meta.json has no scale\n", src); return 1; }
    J.G = (sgrid){{A[0], A[1], A[2]}, gw, gh, 1 / scale};
    /* outward sign: the crop's convexity (a turn is concave towards the axis: the chord midpoint lies inward) unless given */
    J.sgn = 1; int og = atoi(popt(argc, argv, "--outward", "0"));
    { double C[3] = {0}, Pc[3], nc[3]; int m = 0;
      for (int y = 0; y < J.W; y += J.W / 16) for (int x = 0; x < J.W; x += J.W / 16) { double P[3], n[3]; if (frame(&J, x, y, P, n)) { for (int k = 0; k < 3; k++) C[k] += P[k]; m++; } }
      if (!m || !frame(&J, J.W / 2, J.W / 2, Pc, nc)) { fprintf(stderr, "sheet-pbr: the crop has no surface at its centre\n"); return 1; }
      double s = 0; for (int k = 0; k < 3; k++) s += nc[k] * (Pc[k] - C[k] / m);
      J.sgn = og ? og : (s >= 0 ? 1 : -1);
      fprintf(stderr, "sheet-pbr: crop centre z %.0f y %.0f x %.0f, outward sign %+g (%s; convexity %.2f voxels)\n", Pc[0], Pc[1], Pc[2], J.sgn, og ? "given" : "auto", s); }
    /* the CT box */
    { double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30}; const int st = J.W > 64 ? 4 : 1;
      for (int y = 0; y < J.W + st; y += st) for (int x = 0; x < J.W + st; x += st) { double P[3], n[3];
          if (!frame(&J, x < J.W ? x : J.W - 1, y < J.W ? y : J.W - 1, P, n)) continue;
          for (int s = -1; s <= 1; s += 2) for (int k = 0; k < 3; k++) { const double q = P[k] + s * (J.depth + 14 * J.fs) * n[k]; if (q < lo[k]) lo[k] = q; if (q > hi[k]) hi[k] = q; } }
      for (int k = 0; k < 3; k++) { J.B.o[k] = (int64_t)floor(lo[k]) - 4; J.B.n[k] = (int64_t)ceil(hi[k]) + 5 - J.B.o[k]; if (J.B.o[k] < 0) { J.B.n[k] += J.B.o[k]; J.B.o[k] = 0; } }
      const double gb = (double)J.B.n[0] * J.B.n[1] * J.B.n[2] / 1e9;
      fprintf(stderr, "sheet-pbr: CT box z %lld y %lld x %lld + %lld %lld %lld (%.2f GB)\n", (long long)J.B.o[0], (long long)J.B.o[1], (long long)J.B.o[2], (long long)J.B.n[0], (long long)J.B.n[1], (long long)J.B.n[2], gb);
      if (gb > 16) { fprintf(stderr, "sheet-pbr: the CT box is too large; use a smaller crop\n"); return 1; }
      if (J.map) {
          for (int k = 0; k < 3; k++) { if (J.B.o[k] >= regfv->shape[0][k]) { fprintf(stderr, "sheet-pbr: the crop is outside the fine scan\n"); return 1; }
              if (J.B.o[k] + J.B.n[k] > regfv->shape[0][k]) J.B.n[k] = regfv->shape[0][k] - J.B.o[k]; }
          J.B.v = malloc((size_t)J.B.n[0] * J.B.n[1] * J.B.n[2]);
          if (!J.B.v || rvol_read_nt(regfv, 0, J.B.o, J.B.n, J.B.v, nth)) { fprintf(stderr, "sheet-pbr: fine CT read failed\n"); return 1; }
          size_t nz = 0; const size_t nb = (size_t)J.B.n[0] * J.B.n[1] * J.B.n[2]; for (size_t i = 0; i < nb; i += 97) nz += J.B.v[i] != 0;
          fprintf(stderr, "sheet-pbr: fine CT box: %.1f %% non-zero voxels\n", 100.0 * nz / ((nb + 96) / 97));
      } else {
      z3 *z = z3_open(store_open(ctr), ctk, popt(argc, argv, "--cache", nullptr)); if (!z) { fprintf(stderr, "sheet-pbr: cannot open %s %s: %s\n", ctr, ctk, z3_error()); return 1; }
      const z3_meta *zm = z3_meta_of(z); for (int k = 0; k < 3; k++) if (J.B.o[k] + J.B.n[k] > zm->shape[k]) J.B.n[k] = zm->shape[k] - J.B.o[k];
      J.B.v = malloc((size_t)J.B.n[0] * J.B.n[1] * J.B.n[2]);
      if (!J.B.v || z3_read(z, J.B.o, J.B.n, J.B.v, nth)) { fprintf(stderr, "sheet-pbr: CT read failed: %s\n", z3_error()); return 1; }
      z3_close(z); } }
    if (*popt(argc, argv, "--profiles", "")) {   /* diagnostics: raw profiles on a 16 x 16 lattice (float32 [256][K], t = -depth + k step) */
        FILE *pf = fopen(popt(argc, argv, "--profiles", ""), "wb");
        for (int b = 0; b < 16; b++) for (int a = 0; a < 16; a++) { double P[3], n[3]; float f[1024] = {0};
            if (frame(&J, (a * J.W) / 16 + J.W / 32, (b * J.W) / 16 + J.W / 32, P, n)) for (int k = 0; k < J.K; k++) f[k] = (float)prof(&J, P, n, -J.depth + k * J.step);
            fwrite(f, sizeof(float), J.K, pf); }
        fclose(pf);
    }
    if (*popt(argc, argv, "--xsec", "")) {   /* diagnostics: the cross-section along one output row (float32 [W][K]) and one column */
        char xp[1400]; for (int c = 0; c < 2; c++) { snprintf(xp, sizeof xp, "%s.%s.f32", popt(argc, argv, "--xsec", ""), c ? "col" : "row"); FILE *pf = fopen(xp, "wb");
            for (int a = 0; a < J.W; a++) { double P[3], n[3]; float f[1024] = {0};
                if (frame(&J, c ? J.W / 2 : a, c ? a : J.W / 2, P, n)) for (int k = 0; k < J.K; k++) f[k] = (float)prof(&J, P, n, -J.depth + k * J.step);
                fwrite(f, sizeof(float), J.K, pf); }
            fclose(pf); }
    }
    if (*popt(argc, argv, "--layers", "")) {   /* diagnostics: the CT on parallel layers t = A, A + 1, .., B voxels along the normal (uint8 [nL][W][W]) */
        int la = 0, lb = 0; sscanf(popt(argc, argv, "--layers", ""), "%d,%d", &la, &lb); const int nL = lb - la + 1;
        if (nL > 0 && nL <= 256) { uint8_t *Ls = calloc((size_t)nL * J.W * J.W, 1);
            for (int y = 0; y < J.W; y++) for (int x = 0; x < J.W; x++) { double P[3], n[3]; if (!frame(&J, x, y, P, n)) continue;
                for (int l = 0; l < nL; l++) { const double q = prof(&J, P, n, la + l); Ls[((size_t)l * J.W + y) * J.W + x] = (uint8_t)(q < 0 ? 0 : q > 255 ? 255 : q + 0.5); } }
            snprintf(p, sizeof p, "%s/layers_%d_%d.u8", out, la, lb); FILE *lf = fopen(p, "wb"); fwrite(Ls, 1, (size_t)nL * J.W * J.W, lf); fclose(lf); free(Ls); }
    }
    /* pass 1: the sheet near the mesh on an 8-pixel lattice, median-filtered, bilinear upsampled */
    const int L8 = 8 * J.ss, nc = J.W / L8 + 2; float *c1 = malloc((size_t)nc * nc * sizeof *c1), *c2 = malloc((size_t)nc * nc * sizeof *c2);
    float *cand = malloc((size_t)nc * nc * 16 * sizeof *cand), *cgl = malloc((size_t)nc * nc * 16 * sizeof *cgl), *cgr = malloc((size_t)nc * nc * 16 * sizeof *cgr); uint8_t *ncand = calloc((size_t)nc * nc, 1);
    float *lgl = malloc((size_t)nc * nc * sizeof *lgl), *lgr = malloc((size_t)nc * nc * sizeof *lgr);
    p1ctx P1 = {&J, cand, cgl, cgr, ncand, nc, L8}; prows(nc, p1_row, &P1);
    /* tracking: from the lattice point nearest the crop centre with a sheet, the sheet whose middle is nearest the mesh; breadth-first,
       every lattice point takes the candidate nearest the mean of its assigned neighbours (within 0.4 of the median sheet spacing) */
    { for (size_t q = 0; q < (size_t)nc * nc; q++) c1[q] = lgl[q] = lgr[q] = NAN;
      int *qu = malloc((size_t)nc * nc * sizeof *qu), qh = 0, qt = 0, seed = -1; double best = 1e30;
      for (int b = 0; b < nc; b++) for (int a = 0; a < nc; a++) { const size_t q = (size_t)b * nc + a; if (!ncand[q]) continue;
          const double d = hypot(a - nc / 2.0, b - nc / 2.0); if (d < best) { best = d; seed = (int)q; } }
      double spc = 0; int ns = 0; for (size_t q = 0; q < (size_t)nc * nc; q++) for (int k = 1; k < ncand[q]; k++) { spc += cand[q * 16 + k] - cand[q * 16 + k - 1]; ns++; }
      spc = ns ? spc / ns : 30; const double tol = 0.4 * spc;
      if (seed >= 0) { int bk = 0; for (int k = 1; k < ncand[seed]; k++) if (fabs(cand[(size_t)seed * 16 + k]) < fabs(cand[(size_t)seed * 16 + bk])) bk = k;
          c1[seed] = cand[(size_t)seed * 16 + bk]; lgl[seed] = cgl[(size_t)seed * 16 + bk]; lgr[seed] = cgr[(size_t)seed * 16 + bk]; qu[qt++] = seed; }
      while (qh < qt) { const int q = qu[qh++], a0 = q % nc, b0 = q / nc;
          for (int d = 0; d < 4; d++) { const int a = a0 + (d == 0) - (d == 1), b = b0 + (d == 2) - (d == 3); if (a < 0 || b < 0 || a >= nc || b >= nc) continue;
              const size_t r = (size_t)b * nc + a; if (c1[r] == c1[r] || !ncand[r]) continue;
              double s = 0; int m = 0; for (int e = 0; e < 4; e++) { const int aa = a + (e == 0) - (e == 1), bb = b + (e == 2) - (e == 3); if (aa < 0 || bb < 0 || aa >= nc || bb >= nc) continue;
                  const float v = c1[(size_t)bb * nc + aa]; if (v == v) { s += v; m++; } }
              const double ref = s / m; int bk = -1; double bd = tol;
              for (int k = 0; k < ncand[r]; k++) { const double dd = fabs(cand[r * 16 + k] - ref); if (dd < bd) { bd = dd; bk = k; } }
              if (bk >= 0) { c1[r] = cand[r * 16 + bk]; lgl[r] = cgl[r * 16 + bk]; lgr[r] = cgr[r * 16 + bk]; qu[qt++] = (int)r; } } }
      int nt = 0; for (size_t q = 0; q < (size_t)nc * nc; q++) nt += c1[q] == c1[q];
      fprintf(stderr, "sheet-pbr: tracking: %d of %d lattice points on the sheet, mean sheet spacing %.1f voxels\n", nt, nc * nc, spc);
      free(qu); }
    free(cand); free(cgl); free(cgr); free(ncand);
    const size_t NN = (size_t)J.W * J.W;
    J.tp0 = lat_up(c1, nc, L8, J.W); float *ugl = lat_up(lgl, nc, L8, J.W), *ugr = lat_up(lgr, nc, L8, J.W);
    J.cr = malloc(NN * sizeof(float)); J.cvv = malloc(NN * sizeof(float));
    for (size_t i = 0; i < NN; i++) { const double t = ugr[i] - ugl[i]; J.cr[i] = (float)(ugl[i] + 0.3 * t); J.cvv[i] = (float)(ugr[i] - 0.3 * t); J.tp0[i] = (float)(0.3 * t); }
    if (!strcmp(popt(argc, argv, "--mode", "mesh"), "mesh")) {   /* the faces near the mesh, robustly smoothed (no sheet hopping) */
        double ra = -15, rb = 5; sscanf(popt(argc, argv, "--recto-range", "-15,5"), "%lf,%lf", &ra, &rb); ra *= J.fs; rb *= J.fs;
        float *lr = malloc((size_t)nc * nc * sizeof *lr), *lv = malloc((size_t)nc * nc * sizeof *lv);
        mlctx MC = {&J, lr, lv, nc, L8, ra, rb}; prows(nc, ml_row, &MC);
        const double sg = atof(popt(argc, argv, "--face-sigma", "3")), tl = atof(popt(argc, argv, "--face-tol", "3")) * J.fs;
        robust_smooth(lr, nc, sg, tl); robust_smooth(lv, nc, sg, tl);
        float *ur = lat_up(lr, nc, L8, J.W), *uv = lat_up(lv, nc, L8, J.W);
        const double fw = atof(popt(argc, argv, "--face-win", "3")) * J.fs;
        for (size_t i = 0; i < NN; i++) { J.cr[i] = ur[i]; J.cvv[i] = uv[i]; J.tp0[i] = (float)fw; }
        free(lr); free(lv); free(ur); free(uv);
    }
    free(c1); free(c2); free(lgl); free(lgr); free(ugl); free(ugr);
    /* pass 2 */
    float **M[] = {&J.r, &J.v, &J.thk, &J.pk, &J.nin, &J.nout, &J.flat, &J.fmax, &J.valid, &J.grr, &J.grv, &J.amb};
    for (size_t k = 0; k < sizeof M / sizeof *M; k++) *M[k] = malloc(NN * sizeof(float));
    pthread_t th[64]; const int nt = nth < 64 ? nth : 64;
    for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, pass2, &J);
    for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
    size_t namb = 0, nmov = 0, n2 = 0;   /* pass-2 (unconstrained) statistics */
    for (size_t i = 0; i < NN; i++) if (J.valid[i] > 0) { n2++; namb += J.amb[i] >= 0.7; }
    size_t ndisagree = 0, ndp = 0;
    if (atoi(popt(argc, argv, "--dp", "1"))) {
        const int KD = (int)(atof(popt(argc, argv, "--dp-range", "6")) * J.fs / J.step + 0.5), KE = 2 * KD + 1;
        float *E = malloc(NN * KE * sizeof *E), *rrow = malloc(NN * sizeof *rrow), *rdp = malloc(NN * sizeof *rdp);
        if (!E) { fprintf(stderr, "sheet-pbr: no memory for the DP evidence\n"); return 1; }
        dpectx EC = {&J, E, KD}; prows(J.W, dpe_row, &EC);
        const double lam = atof(popt(argc, argv, "--dp-lambda", "0.5")), smax = atof(popt(argc, argv, "--dp-step", "1")), mu = atof(popt(argc, argv, "--dp-mu", "0.5"));
        dpctx R1 = {&J, E, KD, lam, smax, 0, nullptr, rrow, 0}; prows(J.W, dp_line, &R1);
        dpctx R2 = {&J, E, KD, lam, smax, mu, rrow, rdp, 1}; prows(J.W, dp_line, &R2);
        for (size_t i = 0; i < NN; i++) if (rdp[i] == rdp[i] && J.r[i] == J.r[i]) { ndp++; ndisagree += fabsf(rdp[i] - J.r[i]) > 1; }
        /* the recto face re-found within +-0.5 voxels of the DP surface (sub-voxel), the verso as before around its own median */
        float *mv = malloc(NN * sizeof *mv); median2d(J.v, mv, J.W, 3 * J.ss);
        for (size_t i = 0; i < NN; i++) { J.cr[i] = rdp[i]; J.cvv[i] = mv[i] == mv[i] ? mv[i] : NAN; J.tp0[i] = 0.5f; }
        atomic_store(&J.next, 0);
        for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, pass2, &J);
        for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
        for (size_t i = 0; i < NN; i++) if (!(J.valid[i] > 0) && rdp[i] == rdp[i]) J.r[i] = rdp[i];   /* the DP surface where no verso was found */
        free(E); free(rrow); free(rdp); free(mv);
        fprintf(stderr, "sheet-pbr: DP recto surface: per-sample steepest rise more than 1 voxel from the DP surface at %.1f %% of the samples\n", ndp ? 100.0 * ndisagree / ndp : 0);
    }
    /* pass 3, continuity: every sample re-finds its faces within +-cw voxels of the median of its neighbourhood (pass 2 faces), i.e. a
       face may deviate at most cw from its surroundings; samples that moved by more than cw were discontinuous in pass 2 */
    const double cw = atof(popt(argc, argv, "--cont-win", "1.25")) * J.fs; const int cr2 = (int)(atof(popt(argc, argv, "--cont-r", "3")) * J.ss + 0.5);
    if (cw > 0 && !atoi(popt(argc, argv, "--dp", "1"))) for (int it = 0; it < 2; it++) {
        float *mr = malloc(NN * sizeof *mr), *mv = malloc(NN * sizeof *mv), *r0 = malloc(NN * sizeof *r0);
        median2d(J.r, mr, J.W, cr2); median2d(J.v, mv, J.W, cr2); memcpy(r0, J.r, NN * sizeof *r0);
        for (size_t i = 0; i < NN; i++) { J.cr[i] = mr[i]; J.cvv[i] = mv[i]; J.tp0[i] = (float)cw; }
        atomic_store(&J.next, 0);
        for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, pass2, &J);
        for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
        if (it == 0) for (size_t i = 0; i < NN; i++) nmov += r0[i] == r0[i] && mr[i] == mr[i] && fabsf(r0[i] - mr[i]) > cw;
        free(mr); free(mv); free(r0);
    }
    size_t nv = 0; for (size_t i = 0; i < NN; i++) nv += J.valid[i] > 0;
    fprintf(stderr, "sheet-pbr: faces: %.1f %% of samples found in pass 2; ambiguous recto / verso edge (runner-up >= 0.7 of the best, >= 1 voxel away) %.1f %%; "
                    "pass-2 recto more than %.2f voxels from its neighbourhood median %.1f %%; found after the continuity passes %.1f %%\n",
            100.0 * n2 / NN, n2 ? 100.0 * namb / n2 : 0, cw, n2 ? 100.0 * nmov / n2 : 0, 100.0 * nv / NN);
    /* the height field and diagnostics */
    float *H = malloc(NN * sizeof *H), *Hv = malloc(NN * sizeof *Hv), *g = malloc(NN * sizeof *g), *hp = malloc(NN * sizeof *hp), *rough = malloc(NN * sizeof *rough), *curv = malloc(NN * sizeof *curv), *g2 = malloc(NN * sizeof *g2);
    for (size_t i = 0; i < NN; i++) { H[i] = -J.r[i]; Hv[i] = J.v[i]; }
    gblur(H, g, J.W, 6.0 * J.ss); for (size_t i = 0; i < NN; i++) hp[i] = H[i] - g[i];
    for (size_t i = 0; i < NN; i++) g[i] = hp[i] * hp[i];
    gblur(g, rough, J.W, 2.0 * J.ss); for (size_t i = 0; i < NN; i++) rough[i] = sqrtf(fmaxf(rough[i], 0));
    gblur(H, g2, J.W, 1.5 * J.ss);
    for (int y = 0; y < J.W; y++) for (int x = 0; x < J.W; x++) { const size_t i = (size_t)y * J.W + x;
        curv[i] = (x > 0 && y > 0 && x + 1 < J.W && y + 1 < J.W) ? g2[i - 1] + g2[i + 1] + g2[i - J.W] + g2[i + J.W] - 4 * g2[i] : NAN; }
    /* holes in H (no sheet found): filled from the blurred field for shading only */
    float *Hs = malloc(NN * sizeof *Hs); gblur(H, g, J.W, 3.0 * J.ss); for (size_t i = 0; i < NN; i++) Hs[i] = H[i] == H[i] ? H[i] : (g[i] == g[i] ? g[i] : 0);
    const hfield F = {Hs, J.W, J.fs / J.ss};
    /* shading */
    float *rmax = malloc(NN * sizeof *rmax), *rmin = malloc(NN * sizeof *rmin), *img = malloc(NN * sizeof *img), *ao = malloc(NN * sizeof *ao), *pbr = malloc(NN * sizeof *pbr), *pbd = malloc(NN * sizeof *pbd);
    for (size_t i = 0; i < NN; i++) { rmax[i] = -1e30f; rmin[i] = 1e30f; }
    aoctx AC = {&F, ao, relief, J.ss}; prows(J.W, ao_row, &AC);
    double dlo, dhi; pctl(J.nin, NN, 0.02, 0.98, &dlo, &dhi);
    for (int l = 0; l < nl; l++) {
        const double az = 2 * M_PI * l / nl, Lv[3] = {cos(elev) * cos(az), cos(elev) * sin(az), sin(elev)};
        ltctx LC = {&F, Lv, relief, alpha, dlo, dhi, l == (nl * 3) / 8, J.nin, ao, img, rmax, rmin, pbr, pbd}; prows(J.W, lt_row, &LC);
        char nmb[32]; snprintf(nmb, sizeof nmb, "light_%02d", l); write_shade(out, nmb, img, J.W);
    }
    float *rr = malloc(NN * sizeof *rr); for (size_t i = 0; i < NN; i++) rr[i] = rmax[i] - rmin[i];
    write_shade(out, "rti_max", rmax, J.W); write_shade(out, "rti_min", rmin, J.W); write_auto(out, "rti_range", rr, J.W); write_auto(out, "ao", ao, J.W);
    write_shade(out, "pbr", pbr, J.W); write_shade(out, "pbr_density", pbd, J.W);
    { snprintf(p, sizeof p, "%s/normals.ppm", out); FILE *f = fopen(p, "wb"); fprintf(f, "P6\n%d %d\n255\n", J.W, J.W);
      for (int y = 0; y < J.W; y++) for (int x = 0; x < J.W; x++) { double N[3]; hnormal(&F, x, y, relief * 4, N); const uint8_t c[3] = {(uint8_t)(127.5 + 127.5 * N[0]), (uint8_t)(127.5 + 127.5 * N[1]), (uint8_t)(127.5 + 127.5 * N[2])}; fwrite(c, 1, 3, f); }
      fclose(f); }
    /* the maps */
    snprintf(p, sizeof p, "%s/map.json", out); FILE *js = fopen(p, "w"); int first = 1;
    fprintf(js, "{\"dp_disagree\":%.4f,\"amb\":%.4f,\"discont\":%.4f,\"W\":%d,\"ss\":%d,\"crop\":[%lld,%lld,%lld],\"depth\":%g,\"step\":%g,\"interp\":\"%s\",\"outward\":%g,\"valid\":%.4f,\"maps\":[",
            ndp ? (double)ndisagree / ndp : 0, n2 ? (double)namb / n2 : 0, n2 ? (double)nmov / n2 : 0, J.W, J.ss, cu, cv, cn, J.depth, J.step, J.cubic ? "cubic" : "tri", J.sgn, (double)nv / NN);
    const struct { const char *n; const float *a; } maps[] = {{"recto_off", J.r}, {"verso_off", J.v}, {"thickness", J.thk}, {"peak", J.pk}, {"near_recto", J.nin},
        {"near_verso", J.nout}, {"flat", J.flat}, {"flatmax", J.fmax}, {"height", H}, {"verso_h", Hv}, {"highpass", hp}, {"roughness", rough}, {"curvature", curv},
        {"edge_recto", J.grr}, {"edge_verso", J.grv}, {"ambiguity", J.amb}, {"ao", ao}, {"rti_max", rmax}, {"rti_min", rmin}, {"rti_range", rr}, {"pbr", pbr}, {"pbr_density", pbd}};
    for (size_t k = 0; k < sizeof maps / sizeof *maps; k++) { write_f32(out, maps[k].n, maps[k].a, J.W, js, &first); if (k < 16) write_auto(out, maps[k].n, maps[k].a, J.W); }
    fprintf(js, "]}\n"); fclose(js);
    double ts = 0; for (size_t i = 0; i < NN; i++) if (J.valid[i] > 0) ts += J.thk[i];
    fprintf(stderr, "sheet-pbr: %d x %d samples, sheet found in %.1f %%, mean thickness %.2f voxels -> %s\n", J.W, J.W, 100.0 * nv / NN, nv ? ts / nv : 0, out);
    return 0;
}
