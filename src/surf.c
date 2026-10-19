/* Tifxyz surface tools on top of the winding fitter (src/fit.h, ufsm wraps):
   ufsm refine      each vertex moved along its normal onto the fit fields' own sheet position (the network phase's crossing of the
                    target phase, not a CT threshold), the moves smoothed over the surface (confidence-weighted screened Laplacian)
   ufsm surf-stats  label-free quality of a set of surfaces against the fit fields (phase residual, sheet switches seen in the phase,
                    normal smoothness) and optionally the CT along the normal (on-papyrus share, CT profile)
   Vertex coordinates are in the wraps' voxels (meta.json voxel_size_um); point p there is native ((p + 0.5) wum / um - 0.5) for
   fields of native voxel um (ufsm wraps --out-um maps voxel centres to voxel centres). */
#include "fit.h"
#include "tiff.h"
#include "tiffw.h"
#include "zarr3.h"
#include "store.h"
#include <dirent.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *sopt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}
static int sflag(int argc, char **argv, const char *name) { for (int i = 1; i < argc; i++) if (!strcmp(argv[i], name)) return 1; return 0; }
static float *tif_f32(const char *path, int *w, int *h) {
    tiff *t = tiff_open_file(path); tiff_page pg;
    if (!t || tiff_page_info(t, 0, &pg) || pg.fmt != 3 || pg.bits != 32 || pg.spp != 1) { if (t) tiff_close(t); return nullptr; }
    float *d = malloc((size_t)pg.w * pg.h * 4);
    if (!d || tiff_read_page(t, 0, d)) { free(d); tiff_close(t); return nullptr; }
    *w = pg.w; *h = pg.h; tiff_close(t); return d;
}

/* ---- a tifxyz surface ---- */
typedef struct { int w, h, id; double um, step; float *X, *Y, *Z; char name[256]; } sf_wrap;
static void sf_free(sf_wrap *s) { free(s->X); free(s->Y); free(s->Z); memset(s, 0, sizeof *s); }
static int sf_load(const char *dir, const char *name, sf_wrap *s) {
    memset(s, 0, sizeof *s); snprintf(s->name, sizeof s->name, "%s", name);
    char p[1400]; snprintf(p, sizeof p, "%s/%s/meta.json", dir, name);
    FILE *f = fopen(p, "rb"); char b[4096] = {0}; if (f) { const size_t nr = fread(b, 1, sizeof b - 1, f); (void)nr; fclose(f); }
    const char *q = strstr(b, "\"voxel_size_um\":"); if (q) s->um = atof(q + 16);
    q = strstr(b, "\"scale\":"); if (q && (q = strchr(q, '['))) { const double sc = atof(q + 1); s->step = sc > 0 ? 1 / sc : 0; }
    q = strstr(b, "\"winding\":"); s->id = q ? atoi(q + 10) : atoi(name + 1);
    int w2, h2, w3, h3;
    snprintf(p, sizeof p, "%s/%s/x.tif", dir, name); s->X = tif_f32(p, &s->w, &s->h);
    snprintf(p, sizeof p, "%s/%s/y.tif", dir, name); s->Y = tif_f32(p, &w2, &h2);
    snprintf(p, sizeof p, "%s/%s/z.tif", dir, name); s->Z = tif_f32(p, &w3, &h3);
    if (!s->X || !s->Y || !s->Z || s->w != w2 || s->w != w3 || s->h != h2 || s->h != h3 || !(s->um > 0) || !(s->step > 0) || s->w < 3 || s->h < 3) { sf_free(s); return -1; }
    return 0;
}
static int sf_names(const char *dir, char (**names)[256]) {   /* the wNNN entries of a wraps directory, sorted */
    DIR *d = opendir(dir); if (!d) return -1;
    int n = 0, cap = 64; *names = malloc(cap * sizeof **names); struct dirent *de;
    while ((de = readdir(d))) {
        if (de->d_name[0] != 'w' || strlen(de->d_name) >= 256) continue;
        if (n == cap) { cap *= 2; *names = realloc(*names, cap * sizeof **names); }
        snprintf((*names)[n++], 256, "%s", de->d_name);
    }
    closedir(d);
    qsort(*names, n, 256, (int (*)(const void *, const void *))strcmp);
    return n;
}
/* unit normal of vertex i from its grid neighbours (central differences where they exist); 0 when undefined */
static int sf_normal(const sf_wrap *s, size_t i, double N[3]) {
    const int r = (int)(i / s->w), c = (int)(i % s->w), w = s->w; const float *X = s->X, *Y = s->Y, *Z = s->Z;
    if (Z[i] < 0) return 0;
    const int cl = c > 0 && Z[i - 1] >= 0 ? c - 1 : c, cr = c + 1 < w && Z[i + 1] >= 0 ? c + 1 : c;
    const int ru = r > 0 && Z[i - w] >= 0 ? r - 1 : r, rd = r + 1 < s->h && Z[i + w] >= 0 ? r + 1 : r;
    if (cl == cr || ru == rd) return 0;
    const size_t a = (size_t)r * w + cl, b = (size_t)r * w + cr, u = (size_t)ru * w + c, d = (size_t)rd * w + c;
    const double du[3] = {Z[b] - Z[a], Y[b] - Y[a], X[b] - X[a]}, dv[3] = {Z[d] - Z[u], Y[d] - Y[u], X[d] - X[u]};
    N[0] = du[1] * dv[2] - du[2] * dv[1]; N[1] = du[2] * dv[0] - du[0] * dv[2]; N[2] = du[0] * dv[1] - du[1] * dv[0];
    const double l = sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]); if (!(l > 0)) return 0;
    for (int k = 0; k < 3; k++) N[k] /= l;
    return 1;
}

/* ---- the fit fields sampled at native points ---- */
typedef struct { int D, H, W, na; double h, um, sp_med; long long org[3]; uint8_t *pc, *ps, *sp; float *cy, *cx; } sf_fields;
static void sff_free(sf_fields *F) { free(F->pc); free(F->ps); free(F->sp); free(F->cy); free(F->cx); memset(F, 0, sizeof *F); }
static int sff_load(const char *dir, sf_fields *F) {
    memset(F, 0, sizeof *F);
    fit_fields_buf g; if (fit_fields_load(dir, 0, 0, 0, &g)) return -1;
    F->D = g.shape[0]; F->H = g.shape[1]; F->W = g.shape[2]; F->h = g.h; F->um = g.um; for (int d = 0; d < 3; d++) F->org[d] = g.origin[d];
    F->na = (int)(F->D * F->h) + 1; F->cy = g.cy; F->cx = g.cx; g.cy = g.cx = nullptr; fit_fields_free(&g);
    int shp[3];
    if (!(F->pc = fit_field_array(dir, "pc", 1, shp)) || !(F->ps = fit_field_array(dir, "ps", 1, shp)) || !(F->sp = fit_field_array(dir, "sp", 1, shp))) { sff_free(F); return -1; }
    size_t hist[256] = {0}, nsp = 0; const size_t n = (size_t)F->D * F->H * F->W;
    for (size_t i = 0; i < n; i += 7) if (F->sp[i]) { hist[F->sp[i]]++; nsp++; }
    size_t acc = 0; int med = 0; for (int k = 1; k < 256; k++) { acc += hist[k]; if (2 * acc >= nsp) { med = k; break; } }
    F->sp_med = nsp ? med * F->h / 8 : 0;   /* native voxels */
    return 0;
}
/* the phase vector (confidence x (cos, sin) of 2 pi phase) at box-local native q, trilinear over the cells; 0 outside */
static void sff_phase(const sf_fields *F, const double q[3], double *vc, double *vs) {
    double u[3]; int i0[3]; double t[3]; const int n[3] = {F->D, F->H, F->W};
    *vc = *vs = 0;
    for (int d = 0; d < 3; d++) {
        u[d] = q[d] / F->h - 0.5; if (u[d] < 0 || u[d] > n[d] - 1) { if (u[d] < -0.5 || u[d] > n[d] - 0.5) return; u[d] = u[d] < 0 ? 0 : n[d] - 1; }
        i0[d] = (int)u[d]; if (i0[d] > n[d] - 2) i0[d] = n[d] - 2 < 0 ? 0 : n[d] - 2; t[d] = n[d] > 1 ? u[d] - i0[d] : 0;
    }
    for (int c = 0; c < 8; c++) {
        const int a = i0[0] + (c >> 2 & 1), b = i0[1] + (c >> 1 & 1), e = i0[2] + (c & 1);
        if (a >= n[0] || b >= n[1] || e >= n[2]) continue;
        const double wt = (c >> 2 & 1 ? t[0] : 1 - t[0]) * (c >> 1 & 1 ? t[1] : 1 - t[1]) * (c & 1 ? t[2] : 1 - t[2]);
        const size_t k = ((size_t)a * n[1] + b) * n[2] + e;
        *vc += wt * (F->pc[k] - 128) / 127.0; *vs += wt * (F->ps[k] - 128) / 127.0;
    }
}
static double sff_spacing(const sf_fields *F, const double q[3]) {   /* sheet spacing (native) at q: the cell's sp, else the median */
    int c[3]; const int n[3] = {F->D, F->H, F->W};
    for (int d = 0; d < 3; d++) { c[d] = (int)floor(q[d] / F->h); if (c[d] < 0 || c[d] >= n[d]) return F->sp_med; }
    const uint8_t v = F->sp[((size_t)c[0] * n[1] + c[1]) * n[2] + c[2]];
    return v ? v * F->h / 8 : F->sp_med;
}
static double wrap_turn(double e) { return e - floor(e + 0.5); }   /* into [-0.5, 0.5) */
/* vertex i of s in the fields' box-local native coordinates, and (N non-null) its grid normal times sign */
static int sf_native(const sf_wrap *s, const sf_fields *F, size_t i, double q[3], double N[3], int sign) {
    const double f = s->um / F->um; const float *P[3] = {s->Z, s->Y, s->X};
    for (int d = 0; d < 3; d++) q[d] = (P[d][i] + 0.5) * f - 0.5 - F->org[d];
    if (!N) return 1;
    if (!sf_normal(s, i, N)) return 0;
    if (sign < 0) for (int k = 0; k < 3; k++) N[k] = -N[k];
    return 1;
}
/* the sign that turns the wrap's grid normals towards increasing W: a wrap of ufsm wraps is a graph r(z, theta) around its model's
   axis, so its grid normal keeps one orientation. With the model (m): the majority sign of the normal's radial component from the
   model's axis (exact: W grows away from it); else the majority sign of the fields' phase change across the vertex along the normal
   (the phase is 2 pi W) over confident vertices, else the fields' axis (a local box's fields carry the scroll's axis, not the fit's) */
static int sf_orient(const sf_wrap *s, const sf_fields *F, const fit_model *m) {
    long vote = 0, avote = 0, mvote = 0; const size_t n = (size_t)s->w * s->h; const double dl = F->h / 2;
    const int mna = m ? (int)(m->D * m->h) + 1 : 0;
    for (size_t i = 0; i < n; i += 13) {
        double q[3], N[3]; if (s->Z[i] < 0 || !sf_native(s, F, i, q, N, 1)) continue;
        const double zi = fmin(fmax(q[0], 0), F->na - 1), ry = q[1] - F->cy[(int)zi], rx = q[2] - F->cx[(int)zi];
        avote += N[1] * ry + N[2] * rx > 0 ? 1 : -1;
        if (m) {   /* the vertex in the model's box */
            const double g[3] = {q[0] + F->org[0] - m->origin[0], q[1] + F->org[1] - m->origin[1], q[2] + F->org[2] - m->origin[2]};
            const int mz = (int)fmin(fmax(g[0], 0), mna - 1);
            mvote += N[1] * (g[1] - m->cy[mz]) + N[2] * (g[2] - m->cx[mz]) > 0 ? 1 : -1;
            continue;
        }
        const double a[3] = {q[0] - dl * N[0], q[1] - dl * N[1], q[2] - dl * N[2]}, b[3] = {q[0] + dl * N[0], q[1] + dl * N[1], q[2] + dl * N[2]};
        double ac, as, bc, bs; sff_phase(F, a, &ac, &as); sff_phase(F, b, &bc, &bs);
        if (ac * ac + as * as < 0.09 || bc * bc + bs * bs < 0.09) continue;
        const double d = wrap_turn((atan2(bs, bc) - atan2(as, ac)) / (2 * M_PI)); vote += d > 0 ? 1 : d < 0 ? -1 : 0;
    }
    if (m) return mvote >= 0 ? 1 : -1;
    return vote ? (vote > 0 ? 1 : -1) : (avote >= 0 ? 1 : -1);
}

/* ---- ufsm refine ---- */
typedef struct {
    const sf_wrap *s; const sf_fields *F; double range, target, minconf; float *t, *c, *R; float *N; _Atomic int next; int sign;
} rf_job;
static void *rf_rows(void *arg) {   /* per vertex: the target crossing t (native, along N) nearest the vertex and its confidence */
    rf_job *J = arg; const sf_wrap *s = J->s; const sf_fields *F = J->F;
    for (int r; (r = atomic_fetch_add(&J->next, 1)) < s->h;) for (int c = 0; c < s->w; c++) {
        const size_t i = (size_t)r * s->w + c; J->t[i] = 0; J->c[i] = 0; J->R[i] = 0; J->N[3 * i] = NAN;
        double q[3], N[3]; if (s->Z[i] < 0 || !sf_native(s, F, i, q, N, J->sign)) continue;
        for (int k = 0; k < 3; k++) J->N[3 * i + k] = (float)N[k];
        const double R = J->range * sff_spacing(F, q), st = F->h / 4; J->R[i] = (float)R;
        if (!(R > 0)) continue;
        double best = 1e300, bt = 0, bc = 0, tp = 0, ep = 0, cp = 0; int have = 0;
        for (double t = -R; t <= R + 1e-9; t += st) {
            const double p[3] = {q[0] + t * N[0], q[1] + t * N[1], q[2] + t * N[2]}; double vc, vs;
            sff_phase(F, p, &vc, &vs);
            const double cf = sqrt(vc * vc + vs * vs), e = wrap_turn(atan2(vs, vc) / (2 * M_PI) - J->target);
            if (have && cf > 0.05 && cp > 0.05 && ep < 0 && e >= 0 && e - ep < 0.5) {   /* an upward crossing (not the wrap of e) */
                const double tr = tp - ep * (t - tp) / (e - ep);
                if (fabs(tr) < best) { best = fabs(tr); bt = tr; bc = fmin(cf, cp); }
            }
            tp = t; ep = e; cp = cf; have = 1;
        }
        if (best < 1e300 && bc >= J->minconf) { J->t[i] = (float)bt; J->c[i] = (float)fmin(bc, 1.0); }
    }
    return nullptr;
}
/* conjugate gradients on (C + lam L + eps) d = C t over the valid vertices (L: the grid graph Laplacian of valid 4-neighbours) */
static void rf_apply(const sf_wrap *s, const float *cw, double lam, double eps, const double *x, double *y) {
    const int w = s->w, h = s->h;
    for (int r = 0; r < h; r++) for (int c = 0; c < w; c++) {
        const size_t i = (size_t)r * w + c; if (s->Z[i] < 0) { y[i] = 0; continue; }
        double v = (cw[i] + eps) * x[i];
        if (c > 0 && s->Z[i - 1] >= 0) v += lam * (x[i] - x[i - 1]);
        if (c + 1 < w && s->Z[i + 1] >= 0) v += lam * (x[i] - x[i + 1]);
        if (r > 0 && s->Z[i - w] >= 0) v += lam * (x[i] - x[i - w]);
        if (r + 1 < h && s->Z[i + w] >= 0) v += lam * (x[i] - x[i + w]);
        y[i] = v;
    }
}
static int rf_solve(const sf_wrap *s, const float *cw, const float *t, double lam, double eps, int iters, double *d) {
    const size_t n = (size_t)s->w * s->h; double *r = malloc(n * 8), *p = malloc(n * 8), *Ap = malloc(n * 8);
    double rr = 0, bb = 0;
    for (size_t i = 0; i < n; i++) { d[i] = 0; r[i] = s->Z[i] >= 0 ? (double)cw[i] * t[i] : 0; p[i] = r[i]; rr += r[i] * r[i]; }
    bb = rr; int it = 0;
    for (; it < iters && rr > 1e-12 * bb && rr > 0; it++) {
        rf_apply(s, cw, lam, eps, p, Ap);
        double pAp = 0; for (size_t i = 0; i < n; i++) pAp += p[i] * Ap[i];
        if (!(pAp > 0)) break;
        const double a = rr / pAp; double rn = 0;
        for (size_t i = 0; i < n; i++) { d[i] += a * p[i]; r[i] -= a * Ap[i]; rn += r[i] * r[i]; }
        const double b = rn / rr; rr = rn;
        for (size_t i = 0; i < n; i++) p[i] = r[i] + b * p[i];
    }
    free(r); free(p); free(Ap);
    return it;
}
int cmd_refine(int argc, char **argv) {
    if (argc < 4 || !sopt(argc, argv, "--fields", nullptr)) {
        fprintf(stderr, "usage: ufsm refine <wraps-in> <wraps-out> --fields DIR [--model FIT] [--range 0.33] [--smooth-um 40] [--target 0] [--min-conf 0.2] [--iters 400] [--threads 16]\n"); return 2;
    }
    const double range = atof(sopt(argc, argv, "--range", "0.33")), smooth_um = atof(sopt(argc, argv, "--smooth-um", "40")), target = atof(sopt(argc, argv, "--target", "0"));
    const double minconf = atof(sopt(argc, argv, "--min-conf", "0.2")); const int iters = atoi(sopt(argc, argv, "--iters", "400")), nt = atoi(sopt(argc, argv, "--threads", "16"));
    if (!(range > 0) || range > 0.5 || !(smooth_um >= 0) || nt < 1) { fprintf(stderr, "refine: --range in (0, 0.5] sheet spacings, --smooth-um >= 0\n"); return 2; }
    sf_fields F; if (sff_load(sopt(argc, argv, "--fields", ""), &F)) { fprintf(stderr, "refine: cannot read the fields %s\n", sopt(argc, argv, "--fields", "")); return 1; }
    fit_set *ms = sopt(argc, argv, "--model", nullptr) ? fit_set_load(sopt(argc, argv, "--model", "")) : nullptr;   /* orients the normals */
    if (sopt(argc, argv, "--model", nullptr) && !ms) { fprintf(stderr, "refine: cannot load the model %s\n", sopt(argc, argv, "--model", "")); return 1; }
    char (*names)[256] = nullptr; const int nn = sf_names(argv[2], &names); if (nn <= 0) { fprintf(stderr, "refine: no wraps in %s\n", argv[2]); return 1; }
    size_t tv = 0, tf = 0; double sabs = 0, sraw = 0; int nw = 0;
    for (int k = 0; k < nn; k++) {
        sf_wrap s; if (sf_load(argv[2], names[k], &s)) continue;
        const size_t n = (size_t)s.w * s.h; float *t = malloc(n * 4), *c = malloc(n * 4), *R = malloc(n * 4), *N = malloc(n * 12); double *d = malloc(n * 8);
        rf_job J = {&s, &F, range, target, minconf, t, c, R, N, 0, sf_orient(&s, &F, ms ? ms->global : nullptr)};
        pthread_t th[64]; const int m = nt < 64 ? nt : 64;
        for (int q = 0; q < m; q++) pthread_create(&th[q], nullptr, rf_rows, &J);
        for (int q = 0; q < m; q++) pthread_join(th[q], nullptr);
        /* the smoothing: lam = (smooth length / vertex spacing)^2 per unit confidence; eps keeps the system definite where no vertex
           of a patch has a crossing (those stay put) */
        const double sp_um = s.step * s.um, lam = smooth_um > 0 ? (smooth_um / sp_um) * (smooth_um / sp_um) : 0;
        const int its = lam > 0 ? rf_solve(&s, c, t, lam, 1e-3, iters, d) : 0;
        if (lam <= 0) for (size_t i = 0; i < n; i++) d[i] = c[i] > 0 ? t[i] : 0;
        size_t valid = 0, found = 0; double sa = 0, sr = 0; const double f = s.um / F.um;
        for (size_t i = 0; i < n; i++) {
            if (s.Z[i] < 0) continue;
            valid++; if (c[i] > 0) { found++; sr += fabs(t[i]); }
            if (isnan(N[3 * i])) continue;
            const double dd = fmax(-R[i], fmin(R[i], d[i])); sa += fabs(dd);
            s.Z[i] += (float)(dd * N[3 * i] / f); s.Y[i] += (float)(dd * N[3 * i + 1] / f); s.X[i] += (float)(dd * N[3 * i + 2] / f);
        }
        char od[1400]; snprintf(od, sizeof od, "%s/%s", argv[3], names[k]);
        if (tifxyz_write(od, s.w, s.h, s.X, s.Y, s.Z, s.step, s.um, s.id, "ufsm refine (wraps moved onto the fit fields' sheet phase)")) { fprintf(stderr, "refine: cannot write %s\n", od); return 1; }
        if (sflag(argc, argv, "--verbose")) fprintf(stderr, "refine: %s: %zu vertices, %.3f with a crossing (mean |t| %.1f um), mean |move| %.1f um, CG %d its\n",
                                                    names[k], valid, valid ? (double)found / valid : 0, found ? sr / found * F.um : 0, valid ? sa / valid * F.um : 0, its);
        tv += valid; tf += found; sabs += sa; sraw += sr; nw++;
        free(t); free(c); free(R); free(N); free(d); sf_free(&s);
    }
    printf("refine: %d wraps, %zu vertices, %.3f with a target-phase crossing within %.2f spacings (mean |t| %.1f um), mean |move| after smoothing (%.0f um) %.1f um -> %s\n",
           nw, tv, tv ? (double)tf / tv : 0, range, tf ? sraw / tf * F.um : 0, smooth_um, tv ? sabs / tv * F.um : 0, argv[3]);
    free(names); sff_free(&F); fit_set_free(ms);
    return nw ? 0 : 1;
}

/* ---- ufsm surf-stats ---- */
typedef struct { double path, sw, runs; } ss_stat;
/* sheet switches seen in the phase: along one grid direction, vertices with confidence >= cmin joined into paths (gaps < gap_um);
   along a path the residual e is unwrapped (u += wrap(e_n - e_{n-1})) and round(u) is the sheet the surface is on relative to the
   path's start; runs of one sheet shorter than min_um are dropped and a switch is a change between the remaining runs (as ufsm
   wrapswitch walks the published codes) */
static void ss_walk(const sf_wrap *s, const float *e, const float *cf, double cmin, int along_z, double gap_um, double min_um, ss_stat *st, int *sb, double *lb, float *ub) {
    const int L = along_z ? s->w : s->h, M = along_z ? s->h : s->w;
    for (int l = 0; l < L; l++) {
        int n = 0, last = -1;
        for (int m = 0; m <= M; m++) {
            const size_t i = m < M ? (along_z ? (size_t)m * s->w + l : (size_t)l * s->w + m) : 0;
            const int ok = m < M && s->Z[i] >= 0 && cf[i] >= cmin;
            double dl = 0;
            if (ok && last >= 0) {
                const size_t j = along_z ? (size_t)last * s->w + l : (size_t)l * s->w + last;
                const double dz = s->Z[i] - s->Z[j], dy = s->Y[i] - s->Y[j], dx = s->X[i] - s->X[j]; dl = sqrt(dz * dz + dy * dy + dx * dx) * s->um;
            }
            if (n > 0 && (!ok ? m == M : dl > gap_um)) {
                double run = 0, path = 0; int cur = 0, have = 0;
                for (int k = 0; k < n;) {
                    int q = k; double len = 0; while (q < n && sb[q] == sb[k]) { len += lb[q]; q++; }
                    path += len;
                    if (len >= min_um || (k == 0 && q == n)) { if (have && sb[k] != cur) { st->sw += 1; st->runs += 1; run = 0; } cur = sb[k]; have = 1; run += len; }
                    else run += len;
                    k = q;
                }
                st->path += path; if (run > 0) st->runs += 1;
                n = 0;
            }
            if (!ok) continue;
            if (n == 0) { ub[0] = e[i]; sb[0] = 0; lb[0] = 0; n = 1; }
            else { const size_t j = along_z ? (size_t)last * s->w + l : (size_t)l * s->w + last;
                   ub[n] = (float)(ub[n - 1] + wrap_turn(e[i] - e[j])); sb[n] = (int)lround(ub[n] - ub[0]); lb[n] = dl; n++; }
            last = m;
        }
    }
}
typedef struct {
    const sf_wrap *s; const sf_fields *F; float *e, *cf; float *ct; int nprof; double ctum, ctR, ctstep; const uint8_t *vol; int64_t vo[3], vn[3];
    int r0, r1, c0, c1; _Atomic int next; double ctf; int sign;
} ss_job;
static double vol_tri(const uint8_t *v, const int64_t n[3], double z, double y, double x) {
    if (z < 0 || y < 0 || x < 0 || z > n[0] - 1 || y > n[1] - 1 || x > n[2] - 1) return NAN;
    int a = (int)z, b = (int)y, c = (int)x; if (a > n[0] - 2) a = (int)n[0] - 2; if (b > n[1] - 2) b = (int)n[1] - 2; if (c > n[2] - 2) c = (int)n[2] - 2;
    a = a < 0 ? 0 : a; b = b < 0 ? 0 : b; c = c < 0 ? 0 : c;
    const double fz = z - a, fy = y - b, fx = x - c; double s = 0;
    for (int k = 0; k < 8; k++) {
        const int p = a + (k >> 2 & 1), q = b + (k >> 1 & 1), r = c + (k & 1); if (p >= n[0] || q >= n[1] || r >= n[2]) continue;
        s += (k >> 2 & 1 ? fz : 1 - fz) * (k >> 1 & 1 ? fy : 1 - fy) * (k & 1 ? fx : 1 - fx) * v[((size_t)p * n[1] + q) * n[2] + r];
    }
    return s;
}
static void *ss_rows(void *arg) {   /* phase residual and confidence at each vertex; with a CT tile, the CT profile along the normal */
    ss_job *J = arg; const sf_wrap *s = J->s;
    for (int r; (r = J->r0 + atomic_fetch_add(&J->next, 1)) < J->r1;) for (int c = J->c0; c < J->c1; c++) {
        const size_t i = (size_t)r * s->w + c; if (s->Z[i] < 0) continue;
        double q[3], N[3];
        if (!J->vol) {
            sf_native(s, J->F, i, q, nullptr, 1); double vc, vs; sff_phase(J->F, q, &vc, &vs);
            J->cf[i] = (float)sqrt(vc * vc + vs * vs); J->e[i] = (float)wrap_turn(atan2(vs, vc) / (2 * M_PI)); continue;
        }
        if (!sf_native(s, J->F, i, q, N, J->sign)) continue;
        const double f = J->F->um / J->ctum;   /* native -> CT voxels */
        for (int k = 0; k < J->nprof; k++) {
            const double t = (k - (J->nprof - 1) / 2.0) * J->ctstep;   /* um along the normal (outward = +) */
            const double p[3] = {(q[0] + J->F->org[0] + 0.5) * f - 0.5 + t / J->ctum * N[0] - J->vo[0], (q[1] + J->F->org[1] + 0.5) * f - 0.5 + t / J->ctum * N[1] - J->vo[1],
                                 (q[2] + J->F->org[2] + 0.5) * f - 0.5 + t / J->ctum * N[2] - J->vo[2]};
            J->ct[i * J->nprof + k] = (float)vol_tri(J->vol, J->vn, p[0], p[1], p[2]);
        }
    }
    return nullptr;
}
static int cmp_f32(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }
int cmd_surf_stats(int argc, char **argv) {
    if (argc < 3 || !sopt(argc, argv, "--fields", nullptr)) {
        fprintf(stderr, "usage: ufsm surf-stats <wraps-dir> --fields DIR [--model FIT] [--min-conf 0.3] [--gap 1] [--min-run 100] [--ct ROOT KEY --ct-um U [--ct-level 80] [--ct-range 15] [--cache DIR]] [--threads 16] [--per-wrap]\n"); return 2;
    }
    const double cmin = atof(sopt(argc, argv, "--min-conf", "0.3")), gap_um = atof(sopt(argc, argv, "--gap", "1")) * 1000, min_um = atof(sopt(argc, argv, "--min-run", "100"));
    const int nt = atoi(sopt(argc, argv, "--threads", "16")), per = sflag(argc, argv, "--per-wrap");
    int ci = -1; for (int i = 2; i + 2 < argc; i++) if (!strcmp(argv[i], "--ct")) ci = i;
    const double ctum = atof(sopt(argc, argv, "--ct-um", "0")), ctlev = atof(sopt(argc, argv, "--ct-level", "80")), ctR = atof(sopt(argc, argv, "--ct-range", "15"));
    if (ci > 0 && !(ctum > 0)) { fprintf(stderr, "surf-stats: --ct needs --ct-um (the CT array's voxel um)\n"); return 2; }
    z3 *cz = ci > 0 ? z3_open(store_open(argv[ci + 1]), argv[ci + 2], sopt(argc, argv, "--cache", nullptr)) : nullptr;
    if (ci > 0 && !cz) { fprintf(stderr, "surf-stats: %s\n", z3_error()); return 1; }
    sf_fields F; if (sff_load(sopt(argc, argv, "--fields", ""), &F)) { fprintf(stderr, "surf-stats: cannot read the fields\n"); return 1; }
    fit_set *ms = sopt(argc, argv, "--model", nullptr) ? fit_set_load(sopt(argc, argv, "--model", "")) : nullptr;   /* orients the normals */
    if (sopt(argc, argv, "--model", nullptr) && !ms) { fprintf(stderr, "surf-stats: cannot load the model %s\n", sopt(argc, argv, "--model", "")); return 1; }
    char (*names)[256] = nullptr; const int nn = sf_names(argv[2], &names); if (nn <= 0) { fprintf(stderr, "surf-stats: no wraps in %s\n", argv[2]); return 1; }
    /* the CT profile: offsets -30..30 um at the CT voxel (on-papyrus: the max over |t| <= ctR) */
    const double ctstep = ctum > 0 ? ctum : 1; const int nprof = ctum > 0 ? 2 * (int)ceil(30 / ctstep) + 1 : 0;
    double *prof = calloc(nprof + 1, sizeof(double)); size_t *pcount = calloc(nprof + 1, sizeof(size_t)), on = 0, onn = 0, ctn = 0, ctlow = 0;
    ss_stat A[2] = {{0}}; size_t nv = 0, nconf = 0, nbad = 0, nang = 0, nang30 = 0; double ctsum = 0;
    float *eall = malloc(sizeof(float) * 1024), *aall = malloc(sizeof(float) * 1024), *cfall = malloc(sizeof(float) * 1024); size_t ne = 0, cape = 1024, na = 0, capa = 1024, ncf = 0, capc = 1024;
    if (per) printf("  wrap  vertices  conf>=%.1f  |e|med  |e|>0.25  theta sw/cm  z sw/cm%s\n", cmin, cz ? "  on-papyrus" : "");
    for (int k = 0; k < nn; k++) {
        sf_wrap s; if (sf_load(argv[2], names[k], &s)) continue;
        const size_t n = (size_t)s.w * s.h; float *e = calloc(n, 4), *cf = calloc(n, 4), *ct = nprof ? malloc(n * nprof * 4) : nullptr;
        for (size_t i = 0; i < (size_t)n * nprof; i++) ct[i] = NAN;
        const int sign = sf_orient(&s, &F, ms ? ms->global : nullptr);
        { ss_job J = {&s, &F, e, cf, nullptr, 0, 0, 0, 0, nullptr, {0}, {0}, 0, s.h, 0, s.w, 0, 0, sign};
          pthread_t th[64]; const int m = nt < 64 ? nt : 64; for (int q = 0; q < m; q++) pthread_create(&th[q], nullptr, ss_rows, &J); for (int q = 0; q < m; q++) pthread_join(th[q], nullptr); }
        if (cz) {   /* CT tiles of 128 x 512 vertices: each reads its own box (+ the profile reach) */
            const z3_meta *zm = z3_meta_of(cz); const double f = F.um / ctum, reach = 30 / ctum + 2;
            for (int r0 = 0; r0 < s.h; r0 += 128) for (int c0 = 0; c0 < s.w; c0 += 512) {
                const int r1 = r0 + 128 < s.h ? r0 + 128 : s.h, c1 = c0 + 512 < s.w ? c0 + 512 : s.w;
                double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
                for (int r = r0; r < r1; r++) for (int c = c0; c < c1; c++) {
                    const size_t i = (size_t)r * s.w + c; if (s.Z[i] < 0) continue; double q[3]; sf_native(&s, &F, i, q, nullptr, 1);
                    for (int d = 0; d < 3; d++) { const double v = (q[d] + F.org[d] + 0.5) * f - 0.5; if (v < lo[d]) lo[d] = v; if (v > hi[d]) hi[d] = v; }
                }
                if (lo[0] > hi[0]) continue;
                ss_job J = {&s, &F, e, cf, ct, nprof, ctum, ctR, ctstep, nullptr, {0}, {0}, r0, r1, c0, c1, 0, f, sign}; int ok = 1;
                for (int d = 0; d < 3; d++) { int64_t a = (int64_t)floor(lo[d] - reach), b = (int64_t)ceil(hi[d] + reach) + 1; if (a < 0) a = 0; if (b > zm->shape[d]) b = zm->shape[d]; if (b <= a) ok = 0; J.vo[d] = a; J.vn[d] = b - a; }
                if (!ok || (size_t)J.vn[0] * J.vn[1] * J.vn[2] > ((size_t)3 << 30)) continue;
                uint8_t *vol = malloc((size_t)J.vn[0] * J.vn[1] * J.vn[2]);
                if (!vol || z3_read(cz, J.vo, J.vn, vol, nt)) { free(vol); fprintf(stderr, "surf-stats: %s\n", z3_error()); continue; }
                J.vol = vol;
                pthread_t th[64]; const int m = nt < 64 ? nt : 64; for (int q = 0; q < m; q++) pthread_create(&th[q], nullptr, ss_rows, &J); for (int q = 0; q < m; q++) pthread_join(th[q], nullptr);
                free(vol);
            }
        }
        size_t wv = 0, wc = 0, wb = 0, won = 0, wct = 0; float *we = malloc(n * 4); size_t wne = 0;
        for (size_t i = 0; i < n; i++) {
            if (s.Z[i] < 0) continue;
            wv++;
            if (i % 7 == 0) { if (ncf == capc) { capc *= 2; cfall = realloc(cfall, capc * 4); } cfall[ncf++] = cf[i]; }
            if (cf[i] >= cmin) { wc++; const float a = fabsf(e[i]); we[wne++] = a; wb += a > 0.25f; }
            if (nprof) {
                double mx = -1; int any = 0;
                for (int q = 0; q < nprof; q++) { const float v = ct[i * nprof + q]; if (isnan(v)) continue; prof[q] += v; pcount[q]++; const double t = (q - (nprof - 1) / 2.0) * ctstep; if (fabs(t) <= ctR + 1e-9) { mx = fmax(mx, v); any = 1; } }
                const float v0 = ct[i * nprof + (nprof - 1) / 2];
                if (any) { wct++; won += mx >= ctlev; }
                if (!isnan(v0)) { ctsum += v0; ctn++; ctlow += v0 < ctlev; }
            }
            double N0[3], N1[3];   /* the angle to the right and lower neighbours' normals */
            if (sf_normal(&s, i, N0)) for (int q = 0; q < 2; q++) {
                const int r = (int)(i / s.w), c = (int)(i % s.w); if (q == 0 ? c + 1 >= s.w : r + 1 >= s.h) continue;
                const size_t j = q == 0 ? i + 1 : i + s.w; if (!sf_normal(&s, j, N1)) continue;
                const double dt = fmin(1, fabs(N0[0] * N1[0] + N0[1] * N1[1] + N0[2] * N1[2])), ang = acos(dt) * 180 / M_PI;
                if (na == capa) { capa *= 2; aall = realloc(aall, capa * 4); }
                aall[na++] = (float)ang; nang++; nang30 += ang > 30;
            }
        }
        ss_stat S[2] = {{0}}; const int mx = s.w > s.h ? s.w : s.h; int *sb = malloc((mx + 1) * sizeof(int)); double *lb = malloc((mx + 1) * sizeof(double)); float *ub = malloc((mx + 1) * 4);
        for (int d = 0; d < 2; d++) { ss_walk(&s, e, cf, cmin, d, gap_um, min_um, &S[d], sb, lb, ub); A[d].path += S[d].path; A[d].sw += S[d].sw; A[d].runs += S[d].runs; }
        if (per && wne) { qsort(we, wne, 4, cmp_f32);
            printf("  w%03d %9zu  %9.3f  %6.3f  %8.3f  %11.2f  %7.2f", s.id, wv, wv ? (double)wc / wv : 0, we[wne / 2], (double)wb / wne,
                   S[0].path > 0 ? S[0].sw / (S[0].path * 1e-4) : 0, S[1].path > 0 ? S[1].sw / (S[1].path * 1e-4) : 0);
            if (cz) printf("  %10.3f", wct ? (double)won / wct : 0);
            printf("\n"); }
        for (size_t q = 0; q < wne; q++) { if (ne == cape) { cape *= 2; eall = realloc(eall, cape * 4); } eall[ne++] = we[q]; }
        nv += wv; nconf += wc; nbad += wb; on += won; onn += wct;
        free(we); free(sb); free(lb); free(ub); free(e); free(cf); free(ct); sf_free(&s);
    }
    if (ne) qsort(eall, ne, 4, cmp_f32);
    if (na) qsort(aall, na, 4, cmp_f32);
    if (ncf) { qsort(cfall, ncf, 4, cmp_f32); printf("surf-stats: phase confidence at the vertices: p10 %.2f median %.2f p90 %.2f\n", cfall[ncf / 10], cfall[ncf / 2], cfall[ncf * 9 / 10]); }
    printf("surf-stats: %zu vertices, %.3f with phase confidence >= %.2f; phase residual |e| (turns; 0 = on the fields' sheet phase) median %.3f p90 %.3f, > 0.25 %.3f\n",
           nv, nv ? (double)nconf / nv : 0, cmin, ne ? eall[ne / 2] : 0, ne ? eall[ne * 9 / 10] : 0, ne ? (double)nbad / ne : 0);
    printf("surf-stats: sheet switches in the phase: around the axis %.1f cm, %.0f switches, %.2f per cm, run %.2f mm; along z %.1f cm, %.0f, %.2f per cm, run %.2f mm\n",
           A[0].path * 1e-4, A[0].sw, A[0].path > 0 ? A[0].sw / (A[0].path * 1e-4) : 0, A[0].runs ? A[0].path / A[0].runs * 1e-3 : 0,
           A[1].path * 1e-4, A[1].sw, A[1].path > 0 ? A[1].sw / (A[1].path * 1e-4) : 0, A[1].runs ? A[1].path / A[1].runs * 1e-3 : 0);
    printf("surf-stats: adjacent normals: angle median %.2f p90 %.2f deg, > 30 deg %.5f\n", na ? aall[na / 2] : 0, na ? aall[na * 9 / 10] : 0, nang ? (double)nang30 / nang : 0);
    if (nprof) {
        printf("surf-stats: CT: on papyrus (max CT within +-%.0f um along the normal >= %.0f) %.3f; at the vertex mean %.1f, below %.0f %.3f\n",
               ctR, ctlev, onn ? (double)on / onn : 0, ctn ? ctsum / ctn : 0, ctlev, ctn ? (double)ctlow / ctn : 0);
        printf("surf-stats: CT profile along the normal (um: mean CT; + = outward, larger W):");
        for (int q = 0; q < nprof; q++) printf(" %+.0f:%.1f", (q - (nprof - 1) / 2.0) * ctstep, pcount[q] ? prof[q] / pcount[q] : 0);
        printf("\n");
    }
    free(eall); free(aall); free(cfall); free(prof); free(pcount); free(names); sff_free(&F); fit_set_free(ms); if (cz) z3_close(cz);
    return 0;
}
