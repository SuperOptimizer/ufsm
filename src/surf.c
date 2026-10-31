/* Tifxyz surface tools on top of the winding fitter (src/fit.h, ufsm wraps):
   ufsm refine      each vertex moved along its normal onto the fit fields' own sheet position (the network phase's crossing of the
                    target phase, not a CT threshold), the moves smoothed over the surface (confidence-weighted screened Laplacian)
   ufsm surf-stats  label-free quality of a set of surfaces against the fit fields (phase residual, sheet switches seen in the phase,
                    normal smoothness) and optionally the CT along the normal (on-papyrus share, CT profile)
   Vertex coordinates are in the wraps' voxels (meta.json voxel_size_um); point p there is native ((p + 0.5) wum / um - 0.5) for
   fields of native voxel um (ufsm wraps --out-um maps voxel centres to voxel centres). */
#include "fit.h"
#include "sysinfo.h"
#include "stitch.h"
#include "tiff.h"
#include "tiffw.h"
#include <unistd.h>
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
    if (tfx_axis_hook && access(path, F_OK)) return tfx_axis_hook(path, w, h);   /* a .sfc-backed wrap (src/tfx.c) */
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
    char *bm = tfx_meta_read(p); const char *b = bm ? bm : "";
    const char *q = strstr(b, "\"voxel_size_um\":"); if (q) s->um = atof(q + 16);
    q = strstr(b, "\"scale\":"); if (q && (q = strchr(q, '['))) { const double sc = atof(q + 1); s->step = sc > 0 ? 1 / sc : 0; }
    q = strstr(b, "\"winding\":"); s->id = q ? atoi(q + 10) : atoi(name + 1); free(bm);
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
static int sff_load_geom(const char *dir, sf_fields *F) {   /* the box, the cell size and the axis only (no field planes) */
    memset(F, 0, sizeof *F);
    fit_fields_buf g; if (fit_fields_load(dir, 0, 0, 0, &g)) return -1;
    F->D = g.shape[0]; F->H = g.shape[1]; F->W = g.shape[2]; F->h = g.h; F->um = g.um; for (int d = 0; d < 3; d++) F->org[d] = g.origin[d];
    F->na = (int)(F->D * F->h) + 1; F->cy = g.cy; F->cx = g.cx; g.cy = g.cx = nullptr; fit_fields_free(&g);
    return 0;
}
static int sff_load(const char *dir, sf_fields *F) {
    if (sff_load_geom(dir, F)) return -1;
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
    if (!F->pc) return;   /* (geometry-only fields: surf-stats --ct-only) */
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
    if (!F->sp) return F->sp_med;
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
    const double minconf = atof(sopt(argc, argv, "--min-conf", "0.2")); const int iters = atoi(sopt(argc, argv, "--iters", "400")), nt = ufsm_threads(atoi(sopt(argc, argv, "--threads", "0")), 16);
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
    const fit_set *ms; float *sr;   /* with the model: per vertex the distance to the next wrap outward / the fields' spacing */
    struct ct_cache_ *cc; int mark;  /* the CT from the chunk cache (vol null): mark its chunks (mark 1) or sample */
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
/* The CT profiles' voxels from a cache of decoded inner chunks (surf-stats --ct). Each tile of vertices used to read (decode) its
   whole bounding box: a curved strip of a wrap spans a box many times its own volume, and the next wrap (~30 CT voxels outward)
   decoded the same chunks again, ~most of a 2 cm half box's surf-stats. Now a tile first marks the chunks its samples touch,
   the missing ones are decoded on threads (z3_read of the chunk-aligned box: the same voxels), and the samples read them; the
   chunks stay cached for the next tiles and wraps until the last wrap that reads them is done (ct_wrap; the least recently used
   evicted beyond UFSM_CT_CACHE_MB, default 2048: a 2 cm half box holds at most ~810 chunks). The tile
   box (its bounds make samples outside NaN, a box over 3 GB is skipped) is kept as before: the same profiles, bit for bit. */
typedef struct ct_cache_ { z3 *z; int64_t shape[3]; int cs[3]; int64_t g[3]; uint8_t **ch; uint32_t *stamp; _Atomic uint8_t *need; size_t nres, cap; uint32_t clock; int nt;
                 int32_t *last; size_t maxres; } ct_cache;   /* last: per chunk the last wrap that reads it (-1: none known; the premark pass) */
static _Atomic size_t g_ctc_decoded;   /* chunks decoded (UFSM_CT_CACHE_STATS=1 prints it) */
static int ctc_init(ct_cache *c, z3 *z, int nt) {
    memset(c, 0, sizeof *c); const z3_meta *m = z3_meta_of(z); c->z = z; c->nt = nt < 1 ? 1 : nt;
    for (int d = 0; d < 3; d++) { c->shape[d] = m->shape[d]; c->cs[d] = m->chunk[d] > 0 ? m->chunk[d] : 128; c->g[d] = (m->shape[d] + c->cs[d] - 1) / c->cs[d]; }
    const size_t ng = (size_t)c->g[0] * c->g[1] * c->g[2];
    c->ch = calloc(ng, sizeof *c->ch); c->stamp = calloc(ng, sizeof *c->stamp); c->need = calloc(ng, 1); c->last = malloc(ng * sizeof *c->last);
    if (c->last) for (size_t k = 0; k < ng; k++) c->last[k] = -1;
    const char *e = getenv("UFSM_CT_CACHE_MB"); const double mb = e && atof(e) > 0 ? atof(e) : 2048;
    c->cap = (size_t)(mb * 1048576.0 / ((double)c->cs[0] * c->cs[1] * c->cs[2])); if (c->cap < 8) c->cap = 8;
    return c->ch && c->stamp && c->need && c->last ? 0 : -1;
}
static void ctc_free(ct_cache *c) {
    const size_t ng = (size_t)c->g[0] * c->g[1] * c->g[2];
    if (getenv("UFSM_CT_CACHE_STATS")) { size_t u = 0; for (size_t k = 0; c->stamp && k < ng; k++) u += c->stamp[k] != 0; fprintf(stderr, "surf-stats: CT chunks decoded %zu, distinct %zu, cache %zu chunks (at most %zu held)\n", (size_t)g_ctc_decoded, u, c->cap, c->maxres); }
    for (size_t k = 0; c->ch && k < ng; k++) free(c->ch[k]);
    free(c->ch); free((void *)c->need); free(c->stamp); free(c->last); memset(c, 0, sizeof *c);
}
static inline size_t ctc_idx(const ct_cache *c, int64_t z, int64_t y, int64_t x) { return ((size_t)(z / c->cs[0]) * c->g[1] + (size_t)(y / c->cs[1])) * c->g[2] + (size_t)(x / c->cs[2]); }
static inline uint8_t ctc_at(const ct_cache *c, int64_t z, int64_t y, int64_t x) {
    const uint8_t *b = c->ch[ctc_idx(c, z, y, x)];
    return b[((size_t)(z % c->cs[0]) * c->cs[1] + (size_t)(y % c->cs[1])) * c->cs[2] + (size_t)(x % c->cs[2])];
}
/* vol_tri of the tile box at o (shape n) over the cache: mark (the chunks of the corners it reads) or sample */
static double vol_tri_cc(ct_cache *cc, const int64_t o[3], const int64_t n[3], double z, double y, double x, int mark) {
    if (z < 0 || y < 0 || x < 0 || z > n[0] - 1 || y > n[1] - 1 || x > n[2] - 1) return NAN;
    int a = (int)z, b = (int)y, c = (int)x; if (a > n[0] - 2) a = (int)n[0] - 2; if (b > n[1] - 2) b = (int)n[1] - 2; if (c > n[2] - 2) c = (int)n[2] - 2;
    a = a < 0 ? 0 : a; b = b < 0 ? 0 : b; c = c < 0 ? 0 : c;
    const double fz = z - a, fy = y - b, fx = x - c; double s = 0;
    for (int k = 0; k < 8; k++) {
        const int p = a + (k >> 2 & 1), q = b + (k >> 1 & 1), r = c + (k & 1); if (p >= n[0] || q >= n[1] || r >= n[2]) continue;
        if (mark) { atomic_store_explicit(&cc->need[ctc_idx(cc, o[0] + p, o[1] + q, o[2] + r)], 1, memory_order_relaxed); continue; }
        s += (k >> 2 & 1 ? fz : 1 - fz) * (k >> 1 & 1 ? fy : 1 - fy) * (k & 1 ? fx : 1 - fx) * ctc_at(cc, o[0] + p, o[1] + q, o[2] + r);
    }
    return s;
}
typedef struct { ct_cache *c; const int64_t *todo; size_t n; _Atomic size_t next; _Atomic int err; } ctc_job;
static void *ctc_decode(void *arg) {
    ctc_job *J = arg; ct_cache *c = J->c;
    for (size_t k; (k = atomic_fetch_add(&J->next, 1)) < J->n;) {
        const int64_t id = J->todo[k], cz = id / (c->g[1] * c->g[2]), cy = id / c->g[2] % c->g[1], cx = id % c->g[2];
        const int64_t o[3] = {cz * c->cs[0], cy * c->cs[1], cx * c->cs[2]}, nn[3] = {c->cs[0], c->cs[1], c->cs[2]};
        uint8_t *b = malloc((size_t)nn[0] * nn[1] * nn[2]);
        if (!b || z3_read(c->z, o, nn, b, 1)) { free(b); atomic_store(&J->err, 1); continue; }
        c->ch[id] = b; atomic_fetch_add(&g_ctc_decoded, 1);
    }
    return nullptr;
}
/* the marked chunks: decode the missing ones (threads), stamp them, evict the least recently used beyond the cap; clears the marks */
static int ctc_fill(ct_cache *c) {
    const size_t ng = (size_t)c->g[0] * c->g[1] * c->g[2]; c->clock++;
    int64_t *todo = nullptr; size_t ntodo = 0, captodo = 0;
    for (size_t k = 0; k < ng; k++) if (c->need[k]) {
        c->need[k] = 0; c->stamp[k] = c->clock;
        if (c->ch[k]) continue;
        if (ntodo == captodo) { captodo = captodo ? 2 * captodo : 256; todo = realloc(todo, captodo * sizeof *todo); }
        todo[ntodo++] = (int64_t)k;
    }
    ctc_job J = {c, todo, ntodo, 0, 0};
    pthread_t th[64]; const int m = c->nt < 64 ? c->nt : 64;
    for (int q = 0; q < m; q++) pthread_create(&th[q], nullptr, ctc_decode, &J);
    for (int q = 0; q < m; q++) pthread_join(th[q], nullptr);
    free(todo); c->nres += ntodo; if (c->nres > c->maxres) c->maxres = c->nres;
    if (atomic_load(&J.err)) return -1;
    while (c->nres > c->cap) {   /* evict: the oldest stamps first, never this tile's */
        uint32_t old = c->clock; for (size_t k = 0; k < ng; k++) if (c->ch[k] && c->stamp[k] < old) old = c->stamp[k];
        if (old == c->clock) break;
        for (size_t k = 0; k < ng && c->nres > c->cap; k++) if (c->ch[k] && c->stamp[k] == old) { free(c->ch[k]); c->ch[k] = nullptr; c->nres--; }
    }
    return 0;
}
static void ctc_last(ct_cache *c, int k) {   /* the premark pass: the marked chunks' last wrap is k (wraps in order) */
    const size_t ng = (size_t)c->g[0] * c->g[1] * c->g[2];
    for (size_t q = 0; q < ng; q++) if (c->need[q]) { c->need[q] = 0; c->last[q] = k; }
}
static void ctc_done(ct_cache *c, int k) {   /* wrap k done: drop the chunks no later wrap reads */
    const size_t ng = (size_t)c->g[0] * c->g[1] * c->g[2];
    for (size_t q = 0; q < ng; q++) if (c->ch[q] && c->last[q] >= 0 && c->last[q] <= k) { free(c->ch[q]); c->ch[q] = nullptr; c->nres--; }
}
static void *ss_rows(void *arg) {   /* phase residual and confidence at each vertex; with a CT tile, the CT profile along the normal */
    ss_job *J = arg; const sf_wrap *s = J->s;
    for (int r; (r = J->r0 + atomic_fetch_add(&J->next, 1)) < J->r1;) for (int c = J->c0; c < J->c1; c++) {
        const size_t i = (size_t)r * s->w + c; if (s->Z[i] < 0) continue;
        double q[3], N[3];
        if (!J->vol && !J->cc) {
            sf_native(s, J->F, i, q, nullptr, 1); double vc, vs; sff_phase(J->F, q, &vc, &vs);
            J->cf[i] = (float)sqrt(vc * vc + vs * vs); J->e[i] = (float)wrap_turn(atan2(vs, vc) / (2 * M_PI));
            if (J->ms && sf_native(s, J->F, i, q, N, J->sign)) {   /* the wrap spacing along the normal against the fields' spacing */
                const fit_model *g = J->ms->global; const double sp = sff_spacing(J->F, q);
                const double p0[3] = {q[0] + J->F->org[0] - g->origin[0], q[1] + J->F->org[1] - g->origin[1], q[2] + J->F->org[2] - g->origin[2]};
                const double W0 = fit_set_W(J->ms, p0[0], p0[1], p0[2]); double tp = 0, wp = W0; float ratio = 3.f;
                if (sp > 0) for (double t = sp / 8; t <= 3 * sp + 1e-9; t += sp / 8) {
                    const double w = fit_set_W(J->ms, p0[0] + t * N[0], p0[1] + t * N[1], p0[2] + t * N[2]);
                    if (w >= W0 + 1) { ratio = (float)((tp + (W0 + 1 - wp) / (w - wp) * (t - tp)) / sp); break; }
                    tp = t; wp = w;
                }
                J->sr[i] = sp > 0 ? ratio : NAN;
            }
            continue;
        }
        if (!sf_native(s, J->F, i, q, N, J->sign)) continue;
        const double f = J->F->um / J->ctum;   /* native -> CT voxels */
        for (int k = 0; k < J->nprof; k++) {
            const double t = (k - (J->nprof - 1) / 2.0) * J->ctstep;   /* um along the normal (outward = +) */
            const double p[3] = {(q[0] + J->F->org[0] + 0.5) * f - 0.5 + t / J->ctum * N[0] - J->vo[0], (q[1] + J->F->org[1] + 0.5) * f - 0.5 + t / J->ctum * N[1] - J->vo[1],
                                 (q[2] + J->F->org[2] + 0.5) * f - 0.5 + t / J->ctum * N[2] - J->vo[2]};
            if (J->cc) { const double v = vol_tri_cc(J->cc, J->vo, J->vn, p[0], p[1], p[2], J->mark); if (!J->mark) J->ct[i * J->nprof + k] = (float)v; }
            else J->ct[i * J->nprof + k] = (float)vol_tri(J->vol, J->vn, p[0], p[1], p[2]);
        }
    }
    return nullptr;
}
static int cmp_f32(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }
/* ---- faults of a fit on the field grid: cells (every stride-th per axis, material, known spacing) where |grad W| > ratio x the
   fields' sheet density rho = 8 / (h sp): the fit crams turns there (a fault), counted and kept as a mask for the clean-sheet test ---- */
typedef struct {
    const sf_fields *F; const uint8_t *w; const fit_set *ms; int st, n[3]; double ratio, off[3];
    uint8_t *fault; _Atomic int next; _Atomic long long ncell, nfault; float *rs; _Atomic long long nrs; long long caprs;
} sf_fault_job;
static void *sf_fault_rows(void *arg) {
    sf_fault_job *J = arg; const sf_fields *F = J->F; long long nc = 0, nf = 0;
    for (int a; (a = atomic_fetch_add(&J->next, 1)) < J->n[0];) for (int b = 0; b < J->n[1]; b++) for (int c = 0; c < J->n[2]; c++) {
        const int z = a * J->st, y = b * J->st, x = c * J->st; const size_t i = ((size_t)z * F->H + y) * F->W + x;
        if (!J->w[i] || !F->sp[i]) continue;
        double g[3]; const double rho = 8.0 / (F->h * F->sp[i]);
        const double gl = fit_set_gradW(J->ms, (z + 0.5) * F->h + J->off[0], (y + 0.5) * F->h + J->off[1], (x + 0.5) * F->h + J->off[2], F->h / 2, g);
        nc++; const int bad = gl > J->ratio * rho; nf += bad; J->fault[((size_t)a * J->n[1] + b) * J->n[2] + c] = bad ? 2 : 1;
        const long long k = atomic_fetch_add(&J->nrs, 1); if (k < J->caprs) J->rs[k] = (float)(gl / rho);
    }
    atomic_fetch_add(&J->ncell, nc); atomic_fetch_add(&J->nfault, nf);
    return nullptr;
}
/* the largest 4-connected component of clean vertices (mask 1) of a wrap: its area in cm^2 (quads of the grid, um^2 from the vertex
   coordinates) */
static double sf_largest_clean(const sf_wrap *s, const uint8_t *clean, double *total) {
    const int w = s->w, h = s->h; const size_t n = (size_t)w * h; int *lab = malloc(n * sizeof(int)), *stk = malloc(n * sizeof(int));
    for (size_t i = 0; i < n; i++) lab[i] = -1;
    double best = 0, tot = 0; const double u2 = s->um * s->um * 1e-8;   /* wrap voxel^2 -> cm^2 */
    for (size_t i0 = 0; i0 < n; i0++) {
        if (!clean[i0] || lab[i0] >= 0) continue;
        int sp = 0; stk[sp++] = (int)i0; lab[i0] = (int)i0; double area = 0;
        while (sp) {
            const int i = stk[--sp], r = i / w, c = i % w;
            if (c + 1 < w && r + 1 < h && clean[i + 1] && clean[i + w]) {   /* the quad to the right and down: |du x dv| */
                const double du[3] = {s->Z[i + 1] - s->Z[i], s->Y[i + 1] - s->Y[i], s->X[i + 1] - s->X[i]}, dv[3] = {s->Z[i + w] - s->Z[i], s->Y[i + w] - s->Y[i], s->X[i + w] - s->X[i]};
                const double cx = du[1] * dv[2] - du[2] * dv[1], cy = du[2] * dv[0] - du[0] * dv[2], cz = du[0] * dv[1] - du[1] * dv[0];
                area += sqrt(cx * cx + cy * cy + cz * cz) * u2;
            }
            const int nb[4] = {c > 0 ? i - 1 : -1, c + 1 < w ? i + 1 : -1, r > 0 ? i - w : -1, r + 1 < h ? i + w : -1};
            for (int k = 0; k < 4; k++) if (nb[k] >= 0 && clean[nb[k]] && lab[nb[k]] < 0) { lab[nb[k]] = (int)i0; stk[sp++] = nb[k]; }
        }
        tot += area; if (area > best) best = area;
    }
    free(lab); free(stk); *total = tot;
    return best;
}
/* the CT profiles of one wrap (k: its index) in tiles of 128 x 512 vertices, each with its own box (+ the profile reach) as when each
   tile decoded its box. premark: only record the last wrap whose samples touch each chunk (ct_cache last), so that the main pass drops
   a chunk once its last wrap is done: each chunk decoded once (consecutive wraps ~30 CT voxels apart share most of their chunks; an
   LRU of 384 chunks decoded each ~5 times on a 2 cm half box) */
static int ct_wrap(ct_cache *ctc, z3 *cz, const sf_wrap *s, const sf_fields *F, float *e, float *cf, float *ct, int nprof, double ctum, double ctR, double ctstep,
                   int sign, int nt, int k, int premark) {
    const z3_meta *zm = z3_meta_of(cz); const double f = F->um / ctum, reach = 30 / ctum + 2; int rc = 0;
    for (int r0 = 0; r0 < s->h; r0 += 128) for (int c0 = 0; c0 < s->w; c0 += 512) {
        const int r1 = r0 + 128 < s->h ? r0 + 128 : s->h, c1 = c0 + 512 < s->w ? c0 + 512 : s->w;
        double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
        for (int r = r0; r < r1; r++) for (int c = c0; c < c1; c++) {
            const size_t i = (size_t)r * s->w + c; if (s->Z[i] < 0) continue; double q[3]; sf_native(s, F, i, q, nullptr, 1);
            for (int d = 0; d < 3; d++) { const double v = (q[d] + F->org[d] + 0.5) * f - 0.5; if (v < lo[d]) lo[d] = v; if (v > hi[d]) hi[d] = v; }
        }
        if (lo[0] > hi[0]) continue;
        ss_job J = {s, F, e, cf, ct, nprof, ctum, ctR, ctstep, nullptr, {0}, {0}, r0, r1, c0, c1, 0, f, sign, nullptr, nullptr, nullptr, 0}; int ok = 1;
        for (int d = 0; d < 3; d++) { int64_t a = (int64_t)floor(lo[d] - reach), b = (int64_t)ceil(hi[d] + reach) + 1; if (a < 0) a = 0; if (b > zm->shape[d]) b = zm->shape[d]; if (b <= a) ok = 0; J.vo[d] = a; J.vn[d] = b - a; }
        if (!ok || (size_t)J.vn[0] * J.vn[1] * J.vn[2] > ((size_t)3 << 30)) continue;
        J.cc = ctc; J.mark = 1;   /* the chunks the tile's samples read, then the samples */
        pthread_t th[64]; const int m = nt < 64 ? nt : 64;
        for (int q = 0; q < m; q++) pthread_create(&th[q], nullptr, ss_rows, &J);
        for (int q = 0; q < m; q++) pthread_join(th[q], nullptr);
        if (premark) { ctc_last(ctc, k); continue; }
        if (ctc_fill(ctc)) { rc = -1; continue; }
        J.mark = 0; atomic_store(&J.next, 0);
        for (int q = 0; q < m; q++) pthread_create(&th[q], nullptr, ss_rows, &J);
        for (int q = 0; q < m; q++) pthread_join(th[q], nullptr);
    }
    if (!premark) ctc_done(ctc, k);
    return rc;
}
int cmd_surf_stats(int argc, char **argv) {
    if (argc < 3 || !sopt(argc, argv, "--fields", nullptr)) {
        fprintf(stderr, "usage: ufsm surf-stats <wraps-dir> --fields DIR [--model FIT [--fault-ratio 3] [--fault-stride 2]] [--min-conf 0.3] [--gap 1] [--min-run 100] [--ct ROOT KEY --ct-um U [--ct-level 80] [--ct-range 15] [--cache DIR]] [--threads 16] [--per-wrap]\n"
                        "  with --model: faults (field cells with |grad W| > fault-ratio x the fields' sheet density) and the largest clean sheet: per wrap the largest\n"
                        "  connected patch of vertices that are clean (outside faults, wrap spacing 0.5..1.5 x the fields', phase residual < 0.25 where confident, on papyrus with --ct)\n"
                        "  --ct-only 1 (with --ct): only the fields' geometry is read (no phase / spacing planes: a whole scroll's are 100s of GB), so only the CT\n"
                        "  statistics (on-papyrus share per wrap, CT profile) and the normal angles are measured; the phase columns print 0\n"); return 2;
    }
    const double cmin = atof(sopt(argc, argv, "--min-conf", "0.3")), gap_um = atof(sopt(argc, argv, "--gap", "1")) * 1000, min_um = atof(sopt(argc, argv, "--min-run", "100"));
    const int nt = ufsm_threads(atoi(sopt(argc, argv, "--threads", "0")), 16), per = sflag(argc, argv, "--per-wrap"), ctonly = atoi(sopt(argc, argv, "--ct-only", "0"));
    fit_set_threads(nt);   /* the field loads' core masks */
    int ci = -1; for (int i = 2; i + 2 < argc; i++) if (!strcmp(argv[i], "--ct")) ci = i;
    const double ctum = atof(sopt(argc, argv, "--ct-um", "0")), ctlev = atof(sopt(argc, argv, "--ct-level", "80")), ctR = atof(sopt(argc, argv, "--ct-range", "15"));
    if (ci > 0 && !(ctum > 0)) { fprintf(stderr, "surf-stats: --ct needs --ct-um (the CT array's voxel um)\n"); return 2; }
    z3 *cz = ci > 0 ? z3_open(store_open(argv[ci + 1]), argv[ci + 2], sopt(argc, argv, "--cache", nullptr)) : nullptr;
    if (ci > 0 && !cz) { fprintf(stderr, "surf-stats: %s\n", z3_error()); return 1; }
    ct_cache ctc; memset(&ctc, 0, sizeof ctc); if (cz && ctc_init(&ctc, cz, nt)) { fprintf(stderr, "surf-stats: out of memory for the CT chunk table\n"); return 1; }
    if (ctonly && !cz) { fprintf(stderr, "surf-stats: --ct-only needs --ct\n"); return 2; }
    sf_fields F; if ((ctonly ? sff_load_geom : sff_load)(sopt(argc, argv, "--fields", ""), &F)) { fprintf(stderr, "surf-stats: cannot read the fields\n"); return 1; }
    fit_set *ms = sopt(argc, argv, "--model", nullptr) ? fit_set_load(sopt(argc, argv, "--model", "")) : nullptr;   /* orients the normals */
    if (sopt(argc, argv, "--model", nullptr) && !ms) { fprintf(stderr, "surf-stats: cannot load the model %s\n", sopt(argc, argv, "--model", "")); return 1; }
    char (*names)[256] = nullptr; const int nn = sf_names(argv[2], &names); if (nn <= 0) { fprintf(stderr, "surf-stats: no wraps in %s\n", argv[2]); return 1; }
    /* faults on the field grid (with the model; not with --ct-only: no density plane) */
    sf_fault_job FJ; memset(&FJ, 0, sizeof FJ); uint8_t *wfield = nullptr;
    if (ms && !ctonly) {
        int shp[3]; wfield = fit_field_array(sopt(argc, argv, "--fields", ""), "w", 1, shp);
        if (!wfield) { fprintf(stderr, "surf-stats: cannot read the material plane\n"); return 1; }
        FJ.F = &F; FJ.w = wfield; FJ.ms = ms; FJ.st = atoi(sopt(argc, argv, "--fault-stride", "2")); FJ.ratio = atof(sopt(argc, argv, "--fault-ratio", "3"));
        if (FJ.st < 1) FJ.st = 1;
        for (int d = 0; d < 3; d++) FJ.off[d] = (double)(F.org[d] - ms->global->origin[d]);
        FJ.n[0] = (F.D + FJ.st - 1) / FJ.st; FJ.n[1] = (F.H + FJ.st - 1) / FJ.st; FJ.n[2] = (F.W + FJ.st - 1) / FJ.st;
        FJ.fault = calloc((size_t)FJ.n[0] * FJ.n[1] * FJ.n[2], 1); FJ.caprs = 1 << 22; FJ.rs = malloc(FJ.caprs * sizeof(float));
        pthread_t th[64]; const int m = nt < 64 ? nt : 64; for (int q = 0; q < m; q++) pthread_create(&th[q], nullptr, sf_fault_rows, &FJ); for (int q = 0; q < m; q++) pthread_join(th[q], nullptr);
        free(wfield); wfield = nullptr; FJ.w = nullptr;   /* (only the faults read it: 1 GB of a 2 cm box freed for the CT chunk cache) */
        const long long nr = FJ.nrs < FJ.caprs ? FJ.nrs : FJ.caprs; if (nr) qsort(FJ.rs, nr, 4, cmp_f32);
        printf("surf-stats: faults (field cells every %d with |grad W| > %.1f x the sheet density): %lld of %lld cells, share %.4f; |grad W| / density median %.2f p10 %.2f p90 %.2f\n",
               FJ.st, FJ.ratio, (long long)FJ.nfault, (long long)FJ.ncell, FJ.ncell ? (double)FJ.nfault / FJ.ncell : 0, nr ? FJ.rs[nr / 2] : 0, nr ? FJ.rs[nr / 10] : 0, nr ? FJ.rs[nr * 9 / 10] : 0);
    }
    double clean_best = 0, clean_tot = 0; char clean_name[256] = "";
    /* the CT profile: offsets -30..30 um at the CT voxel (on-papyrus: the max over |t| <= ctR) */
    const double ctstep = ctum > 0 ? ctum : 1; const int nprof = ctum > 0 ? 2 * (int)ceil(30 / ctstep) + 1 : 0;
    double *prof = calloc(nprof + 1, sizeof(double)); size_t *pcount = calloc(nprof + 1, sizeof(size_t)), on = 0, onn = 0, ctn = 0, ctlow = 0;
    ss_stat A[2] = {{0}}; size_t nv = 0, nconf = 0, nbad = 0, nang = 0, nang30 = 0; double ctsum = 0;
    float *eall = malloc(sizeof(float) * 1024), *aall = malloc(sizeof(float) * 1024), *cfall = malloc(sizeof(float) * 1024); size_t ne = 0, cape = 1024, na = 0, capa = 1024, ncf = 0, capc = 1024;
    float *srall = malloc(sizeof(float) * 1024); size_t nsa = 0, capsa = 1024, nsr = 0, srlo = 0, srhi = 0;
    if (per) printf("  wrap  vertices  conf>=%.1f  |e|med  |e|>0.25  theta sw/cm  z sw/cm%s\n", cmin, cz ? "  on-papyrus" : "");
    for (int k = 0; cz && k < nn; k++) {   /* the CT chunks' last wraps */
        sf_wrap s; if (sf_load(argv[2], names[k], &s)) continue;
        ct_wrap(&ctc, cz, &s, &F, nullptr, nullptr, nullptr, nprof, ctum, ctR, ctstep, sf_orient(&s, &F, ms ? ms->global : nullptr), nt, k, 1);
        sf_free(&s);
    }
    for (int k = 0; k < nn; k++) {
        sf_wrap s; if (sf_load(argv[2], names[k], &s)) continue;
        const size_t n = (size_t)s.w * s.h; float *e = calloc(n, 4), *cf = calloc(n, 4), *ct = nprof ? malloc(n * nprof * 4) : nullptr;
        for (size_t i = 0; i < (size_t)n * nprof; i++) ct[i] = NAN;
        const int sign = sf_orient(&s, &F, ms ? ms->global : nullptr);
        float *sr = ms && !ctonly ? malloc(n * 4) : nullptr; if (sr) for (size_t i = 0; i < n; i++) sr[i] = NAN;
        if (!ctonly) { ss_job J = {&s, &F, e, cf, nullptr, 0, 0, 0, 0, nullptr, {0}, {0}, 0, s.h, 0, s.w, 0, 0, sign, ms, sr, nullptr, 0};
          pthread_t th[64]; const int m = nt < 64 ? nt : 64; for (int q = 0; q < m; q++) pthread_create(&th[q], nullptr, ss_rows, &J); for (int q = 0; q < m; q++) pthread_join(th[q], nullptr); }
        if (cz && ct_wrap(&ctc, cz, &s, &F, e, cf, ct, nprof, ctum, ctR, ctstep, sign, nt, k, 0)) { fprintf(stderr, "surf-stats: %s\n", z3_error()); }
        size_t wv = 0, wc = 0, wb = 0, won = 0, wct = 0; float *we = malloc(n * 4); size_t wne = 0;
        uint8_t *clean = ms && !ctonly ? calloc(n, 1) : nullptr;   /* clean vertices (the largest clean sheet) */
        for (size_t i = 0; i < n; i++) {
            if (s.Z[i] < 0) continue;
            wv++;
            int vclean = 1;
            if (clean) {
                if (!isnan(sr[i]) && (sr[i] < 0.5f || sr[i] > 1.5f)) vclean = 0;   /* (no next wrap found: unknown, kept) */
                if (cf[i] >= cmin && fabsf(e[i]) >= 0.25f) vclean = 0;
                double q[3]; sf_native(&s, &F, i, q, nullptr, 1); int fc[3], inb = 1;
                for (int d = 0; d < 3; d++) { fc[d] = (int)floor(q[d] / F.h) / FJ.st; if (fc[d] < 0 || fc[d] >= FJ.n[d]) inb = 0; }
                if (inb && FJ.fault[((size_t)fc[0] * FJ.n[1] + fc[1]) * FJ.n[2] + fc[2]] == 2) vclean = 0;
            }
            if (sr && !isnan(sr[i])) { nsr++; srlo += sr[i] < 0.5f; srhi += sr[i] > 1.5f; if (i % 7 == 0) { if (nsa == capsa) { capsa *= 2; srall = realloc(srall, capsa * 4); } srall[nsa++] = sr[i]; } }
            if (i % 7 == 0) { if (ncf == capc) { capc *= 2; cfall = realloc(cfall, capc * 4); } cfall[ncf++] = cf[i]; }
            if (cf[i] >= cmin) { wc++; const float a = fabsf(e[i]); we[wne++] = a; wb += a > 0.25f; }
            if (nprof) {
                double mx = -1; int any = 0;
                for (int q = 0; q < nprof; q++) { const float v = ct[i * nprof + q]; if (isnan(v)) continue; prof[q] += v; pcount[q]++; const double t = (q - (nprof - 1) / 2.0) * ctstep; if (fabs(t) <= ctR + 1e-9) { mx = fmax(mx, v); any = 1; } }
                const float v0 = ct[i * nprof + (nprof - 1) / 2];
                if (any) { wct++; won += mx >= ctlev; }
                if (!any || mx < ctlev) vclean = 0;
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
            if (clean) clean[i] = (uint8_t)vclean;
        }
        if (clean) {
            double tot; const double best = sf_largest_clean(&s, clean, &tot); clean_tot += tot;
            if (best > clean_best) { clean_best = best; snprintf(clean_name, sizeof clean_name, "%s", s.name); }
            free(clean);
        }
        ss_stat S[2] = {{0}}; const int mx = s.w > s.h ? s.w : s.h; int *sb = malloc((mx + 1) * sizeof(int)); double *lb = malloc((mx + 1) * sizeof(double)); float *ub = malloc((mx + 1) * 4);
        for (int d = 0; d < 2; d++) { ss_walk(&s, e, cf, cmin, d, gap_um, min_um, &S[d], sb, lb, ub); A[d].path += S[d].path; A[d].sw += S[d].sw; A[d].runs += S[d].runs; }
        if (per && (wne || (ctonly && wv))) { if (wne) qsort(we, wne, 4, cmp_f32);
            printf("  w%03d %9zu  %9.3f  %6.3f  %8.3f  %11.2f  %7.2f", s.id, wv, wv ? (double)wc / wv : 0, wne ? we[wne / 2] : 0, wne ? (double)wb / wne : 0,
                   S[0].path > 0 ? S[0].sw / (S[0].path * 1e-4) : 0, S[1].path > 0 ? S[1].sw / (S[1].path * 1e-4) : 0);
            if (cz) printf("  %10.3f", wct ? (double)won / wct : 0);
            printf("\n"); }
        for (size_t q = 0; q < wne; q++) { if (ne == cape) { cape *= 2; eall = realloc(eall, cape * 4); } eall[ne++] = we[q]; }
        nv += wv; nconf += wc; nbad += wb; on += won; onn += wct;
        free(we); free(sb); free(lb); free(ub); free(e); free(cf); free(ct); free(sr); sf_free(&s);
    }
    if (ne) qsort(eall, ne, 4, cmp_f32);
    if (na) qsort(aall, na, 4, cmp_f32);
    if (ncf) { qsort(cfall, ncf, 4, cmp_f32); printf("surf-stats: phase confidence at the vertices: p10 %.2f median %.2f p90 %.2f\n", cfall[ncf / 10], cfall[ncf / 2], cfall[ncf * 9 / 10]); }
    printf("surf-stats: %zu vertices, %.3f with phase confidence >= %.2f; phase residual |e| (turns; 0 = on the fields' sheet phase) median %.3f p90 %.3f, > 0.25 %.3f\n",
           nv, nv ? (double)nconf / nv : 0, cmin, ne ? eall[ne / 2] : 0, ne ? eall[ne * 9 / 10] : 0, ne ? (double)nbad / ne : 0);
    printf("surf-stats: sheet switches in the phase: around the axis %.1f cm, %.0f switches, %.2f per cm, run %.2f mm; along z %.1f cm, %.0f, %.2f per cm, run %.2f mm\n",
           A[0].path * 1e-4, A[0].sw, A[0].path > 0 ? A[0].sw / (A[0].path * 1e-4) : 0, A[0].runs ? A[0].path / A[0].runs * 1e-3 : 0,
           A[1].path * 1e-4, A[1].sw, A[1].path > 0 ? A[1].sw / (A[1].path * 1e-4) : 0, A[1].runs ? A[1].path / A[1].runs * 1e-3 : 0);
    if (nsr) { qsort(srall, nsa, 4, cmp_f32);
        printf("surf-stats: spacing to the next wrap outward / the fields' sheet spacing: median %.2f p10 %.2f p90 %.2f; < 0.5 (wraps bunched: a lost turn) %.4f, > 1.5 (a gained turn) %.4f\n",
               srall[nsa / 2], srall[nsa / 10], srall[nsa * 9 / 10], (double)srlo / nsr, (double)srhi / nsr); }
    printf("surf-stats: adjacent normals: angle median %.2f p90 %.2f deg, > 30 deg %.5f\n", na ? aall[na / 2] : 0, na ? aall[na * 9 / 10] : 0, nang ? (double)nang30 / nang : 0);
    if (ms && !ctonly) printf("surf-stats: clean sheet (outside faults, spacing 0.5..1.5, |e| < 0.25 where confident%s): largest %.3f cm2 (%s), all clean patches %.2f cm2\n",
                   nprof ? ", on papyrus" : "", clean_best, clean_name, clean_tot);
    if (nprof) {
        printf("surf-stats: CT: on papyrus (max CT within +-%.0f um along the normal >= %.0f) %.3f; at the vertex mean %.1f, below %.0f %.3f\n",
               ctR, ctlev, onn ? (double)on / onn : 0, ctn ? ctsum / ctn : 0, ctlev, ctn ? (double)ctlow / ctn : 0);
        printf("surf-stats: CT profile along the normal (um: mean CT; + = outward, larger W):");
        for (int q = 0; q < nprof; q++) printf(" %+.0f:%.1f", (q - (nprof - 1) / 2.0) * ctstep, pcount[q] ? prof[q] / pcount[q] : 0);
        printf("\n");
    }
    free(eall); free(aall); free(cfall); free(srall); free(prof); free(pcount); free(names); sff_free(&F); fit_set_free(ms); if (cz) { ctc_free(&ctc); z3_close(cz); }
    free(wfield); free(FJ.fault); free(FJ.rs);
    return 0;
}

/* ---- ufsm self-gate: the strict stretches of a box fit's wraps (fitter Phase 5, self-training labels) ----
   Per vertex of each wrap (after surf-snap), a gate mask:
     conf   the network phase confidence >= --min-conf and the phase residual |e| < --max-e turns (the wrap sits on the network's sheet)
     pap    with --ct: on papyrus (max CT within +-ct-range um along the normal >= ct-level, as surf-stats)
     fault  the model's |grad W| <= fault-ratio x the fields' sheet density at the vertex, and the spacing to the next wrap outward
            0.5..1.5 x the fields' (no crammed or lost turn)
     back   with --quality (stitch quality output of this box): the block's coherence q >= --min-q and q >= the reversed coherence
     nb     every neighbour fit (--nb FITDIR, repeatable) whose box covers the vertex: W_nb - W within --nb-tol turns of the pair's
            integer offset (the most common rounded difference over all covered vertices); a neighbour that agrees on less than
            --nb-min of its overlap fails every vertex it covers
     excl   inside an --exclude box (global native z0,y0,x0,z1,y1,x1; repeatable: held-out regions)
     small  after --erode passes (a vertex with an invalid / gated 4-neighbour is dropped), 4-connected stretches under --min-area cm2
   The passing vertices are written as tifxyz (the rest -1) with q.tif = the wrap's winding id (ufsm raster --value q.tif) into
   <out-dir>/wNNN (wraps without any passing vertex are skipped). Prints the share of vertices and area failing each gate and the
   gated (kept) share and area; per wrap with --per-wrap. */
enum { SG_CONF = 1, SG_PAP = 2, SG_FAULT = 4, SG_BACK = 8, SG_NB = 16, SG_EXCL = 32, SG_SMALL = 64 };
#define SG_MAXNB 16
typedef struct {
    const sf_wrap *s; const sf_fields *F; const fit_set *ms; const st_box *qb; int sign; double ratio, cmin, emax;
    const float *e, *cf; uint8_t *gate; _Atomic int next;
    int nnb; fit_set *const *nb; const int *nbk; long long *nbhist; double nbtol; const uint8_t *nbbad;   /* nbhist: per nb 64 bins of round(d) + 32 */
    int nex; const double *ex; int pass;   /* pass 0: histograms only */
} sg_job;
static double sg_Wat(const fit_set *f, const double g[3], int *inside) {   /* W of a fit set at global native g; inside: in its box */
    const fit_model *m = f->global; const double l[3] = {g[0] - m->origin[0], g[1] - m->origin[1], g[2] - m->origin[2]};
    *inside = l[0] >= 0 && l[1] >= 0 && l[2] >= 0 && l[0] < m->D * m->h && l[1] < m->H * m->h && l[2] < m->W * m->h;
    return *inside ? fit_set_W(f, l[0], l[1], l[2]) : 0;
}
static void *sg_rows(void *arg) {
    sg_job *J = arg; const sf_wrap *s = J->s; const sf_fields *F = J->F; const fit_model *g0 = J->ms->global;
    for (int r; (r = atomic_fetch_add(&J->next, 1)) < s->h;) for (int c = 0; c < s->w; c++) {
        const size_t i = (size_t)r * s->w + c; if (s->Z[i] < 0) continue;
        double q[3], N[3]; sf_native(s, F, i, q, nullptr, 1);
        const double gp[3] = {q[0] + F->org[0], q[1] + F->org[1], q[2] + F->org[2]};
        int in0; const double W0 = sg_Wat(J->ms, gp, &in0);
        uint8_t gt = J->gate[i];
        for (int k = 0; k < J->nnb; k++) {   /* the neighbours: the difference to the pair's offset */
            int in; const double d = sg_Wat(J->nb[k], gp, &in) - W0; if (!in || !in0) continue;
            if (J->pass == 0) { const long b = lround(d) + 32; if (b >= 0 && b < 64) __atomic_fetch_add(&J->nbhist[k * 64 + b], 1, __ATOMIC_RELAXED); continue; }
            if (J->nbbad[k] || fabs(d - J->nbk[k]) >= J->nbtol) gt |= SG_NB;
        }
        if (J->pass == 0) continue;
        if (!(J->cf[i] >= J->cmin) || !(fabs(J->e[i]) < J->emax)) gt |= SG_CONF;
        for (int k = 0; k < J->nex; k++) { const double *b = J->ex + 6 * k;
            if (gp[0] >= b[0] && gp[1] >= b[1] && gp[2] >= b[2] && gp[0] < b[3] && gp[1] < b[4] && gp[2] < b[5]) gt |= SG_EXCL; }
        if (J->qb && J->qb->q) { int qi[3], ok = 1;
            for (int d = 0; d < 3; d++) { qi[d] = (int)floor((gp[d] - J->qb->lo[d]) / J->qb->qb); if (qi[d] < 0 || qi[d] >= J->qb->qn[d]) ok = 0; }
            const size_t k = ok ? ((size_t)qi[0] * J->qb->qn[1] + qi[1]) * J->qb->qn[2] + qi[2] : 0;
            if (!ok || J->qb->q[k] < (float)J->qb->g || J->qb->qa[k] > J->qb->q[k]) gt |= SG_BACK; }
        if (in0 && sf_native(s, F, i, q, N, J->sign)) {   /* fault: |grad W| against the density; the spacing to the next wrap */
            const double sp = sff_spacing(F, q), lg[3] = {gp[0] - g0->origin[0], gp[1] - g0->origin[1], gp[2] - g0->origin[2]};
            double gr[3]; const double gl = fit_set_gradW(J->ms, lg[0], lg[1], lg[2], F->h / 2, gr);
            if (sp > 0 && gl > J->ratio / sp) gt |= SG_FAULT;
            if (sp > 0) { double tp = 0, wp = W0, ratio = 3;
                for (double t = sp / 8; t <= 3 * sp + 1e-9; t += sp / 8) {
                    const double w = fit_set_W(J->ms, lg[0] + t * N[0], lg[1] + t * N[1], lg[2] + t * N[2]);
                    if (w >= W0 + 1) { ratio = (tp + (W0 + 1 - wp) / (w - wp) * (t - tp)) / sp; break; }
                    tp = t; wp = w; }
                if (ratio < 0.5 || ratio > 1.5) gt |= SG_FAULT; }
        }
        J->gate[i] = gt;
    }
    return nullptr;
}
static double sg_quad_area(const sf_wrap *s, const uint8_t *ok, size_t i) {   /* the quad right / down of vertex i when its corners are ok */
    const int w = s->w, r = (int)(i / w), c = (int)(i % w);
    if (c + 1 >= w || r + 1 >= s->h || !ok[i] || !ok[i + 1] || !ok[i + w]) return 0;
    const double du[3] = {s->Z[i + 1] - s->Z[i], s->Y[i + 1] - s->Y[i], s->X[i + 1] - s->X[i]}, dv[3] = {s->Z[i + w] - s->Z[i], s->Y[i + w] - s->Y[i], s->X[i + w] - s->X[i]};
    const double cx = du[1] * dv[2] - du[2] * dv[1], cy = du[2] * dv[0] - du[0] * dv[2], cz = du[0] * dv[1] - du[1] * dv[0];
    return sqrt(cx * cx + cy * cy + cz * cz) * s->um * s->um * 1e-8;
}
int cmd_self_gate(int argc, char **argv) {
    if (argc < 4 || !sopt(argc, argv, "--fields", nullptr) || !sopt(argc, argv, "--model", nullptr)) {
        fprintf(stderr, "usage: ufsm self-gate <wraps-dir> <out-dir> --fields DIR --model FIT [--ct ROOT KEY --ct-um U [--ct-level 80] [--ct-range 15]] [--min-conf 0.3]\n"
                        "         [--max-e 0.15] [--fault-ratio 3] [--quality Q.json [--min-q 0.5]] [--nb FIT]... [--nb-tol 0.15] [--nb-min 0.6] [--exclude z0,y0,x0,z1,y1,x1]...\n"
                        "         [--min-area 0.02] [--erode 1] [--threads 16] [--per-wrap]\n"
                        "  the strict stretches of a box fit's wraps: on the network's sheet, on papyrus, no fault, not reversed, neighbours agreeing, outside\n"
                        "  excluded boxes -> tifxyz with q.tif = the winding id (training labels: ufsm raster --value q.tif)\n"); return 2;
    }
    const double cmin = atof(sopt(argc, argv, "--min-conf", "0.3")), emax = atof(sopt(argc, argv, "--max-e", "0.15")), ratio = atof(sopt(argc, argv, "--fault-ratio", "3"));
    const double minq = atof(sopt(argc, argv, "--min-q", "0.5")), nbtol = atof(sopt(argc, argv, "--nb-tol", "0.15")), nbmin = atof(sopt(argc, argv, "--nb-min", "0.6"));
    const double minarea = atof(sopt(argc, argv, "--min-area", "0.02")); const int erode = atoi(sopt(argc, argv, "--erode", "1")), nt = ufsm_threads(atoi(sopt(argc, argv, "--threads", "0")), 16), per = sflag(argc, argv, "--per-wrap");
    fit_set_threads(nt);
    int ci = -1; for (int i = 2; i + 2 < argc; i++) if (!strcmp(argv[i], "--ct")) ci = i;
    const double ctum = atof(sopt(argc, argv, "--ct-um", "0")), ctlev = atof(sopt(argc, argv, "--ct-level", "80")), ctR = atof(sopt(argc, argv, "--ct-range", "15"));
    if (ci > 0 && !(ctum > 0)) { fprintf(stderr, "self-gate: --ct needs --ct-um\n"); return 2; }
    z3 *cz = ci > 0 ? z3_open(store_open(argv[ci + 1]), argv[ci + 2], sopt(argc, argv, "--cache", nullptr)) : nullptr;
    if (ci > 0 && !cz) { fprintf(stderr, "self-gate: %s\n", z3_error()); return 1; }
    ct_cache ctc; memset(&ctc, 0, sizeof ctc); if (cz && ctc_init(&ctc, cz, nt)) { fprintf(stderr, "self-gate: out of memory\n"); return 1; }
    sf_fields F; if (sff_load(sopt(argc, argv, "--fields", ""), &F)) { fprintf(stderr, "self-gate: cannot read the fields\n"); return 1; }
    fit_set *ms = fit_set_load(sopt(argc, argv, "--model", "")); if (!ms) { fprintf(stderr, "self-gate: cannot load the model\n"); return 1; }
    st_box *qbx = nullptr; int nqb = 0; st_edge *qe = nullptr; int nqe = 0; double qum;
    if (sopt(argc, argv, "--quality", nullptr)) {
        if (st_load(sopt(argc, argv, "--quality", ""), &qbx, &nqb, &qe, &nqe, &qum) || nqb < 1 || !qbx[0].q) { fprintf(stderr, "self-gate: cannot read the quality map %s\n", sopt(argc, argv, "--quality", "")); return 1; }
        qbx[0].g = minq;   /* (the threshold travels in the box's gauge slot: sg_rows reads it) */
    }
    fit_set *nb[SG_MAXNB]; int nnb = 0; double ex[6 * 64]; int nex = 0;
    for (int i = 2; i + 1 < argc; i++) {
        if (!strcmp(argv[i], "--nb")) { if (nnb == SG_MAXNB || !(nb[nnb] = fit_set_load(argv[i + 1]))) { fprintf(stderr, "self-gate: cannot load the neighbour %s (at most %d)\n", argv[i + 1], SG_MAXNB); return 1; } nnb++; }
        if (!strcmp(argv[i], "--exclude")) { double *b = ex + 6 * nex;
            if (nex == 64 || sscanf(argv[i + 1], "%lf,%lf,%lf,%lf,%lf,%lf", b, b + 1, b + 2, b + 3, b + 4, b + 5) != 6) { fprintf(stderr, "self-gate: --exclude z0,y0,x0,z1,y1,x1 (at most 64)\n"); return 2; } nex++; }
    }
    char (*names)[256] = nullptr; const int nn = sf_names(argv[2], &names); if (nn <= 0) { fprintf(stderr, "self-gate: no wraps in %s\n", argv[2]); return 1; }
    char cmd[1400]; snprintf(cmd, sizeof cmd, "mkdir -p '%s'", argv[3]); if (system(cmd)) return 1;
    /* pass 0: the neighbours' integer offsets */
    long long nbhist[SG_MAXNB * 64] = {0}; int nbk[SG_MAXNB] = {0}; uint8_t nbbad[SG_MAXNB] = {0};
    if (nnb) {
        for (int k = 0; k < nn; k++) {
            sf_wrap s; if (sf_load(argv[2], names[k], &s)) continue;
            uint8_t *gate = calloc((size_t)s.w * s.h, 1);
            sg_job J = {.s = &s, .F = &F, .ms = ms, .gate = gate, .nnb = nnb, .nb = nb, .nbhist = nbhist, .pass = 0};
            pthread_t th[64]; const int m = nt < 64 ? (nt > 0 ? nt : 1) : 64;
            for (int q = 0; q < m; q++) pthread_create(&th[q], nullptr, sg_rows, &J);
            for (int q = 0; q < m; q++) pthread_join(th[q], nullptr);
            free(gate); sf_free(&s);
        }
        for (int k = 0; k < nnb; k++) {
            long long tot = 0, best = -1; for (int b = 0; b < 64; b++) { tot += nbhist[k * 64 + b]; if (best < 0 || nbhist[k * 64 + b] > nbhist[k * 64 + best]) best = b; }
            nbk[k] = (int)best - 32; const double share = tot ? (double)nbhist[k * 64 + best] / tot : 0; nbbad[k] = tot > 0 && share < nbmin;
            printf("self-gate: neighbour %d: %lld covered vertices, offset %+d holds %.3f%s\n", k, tot, nbk[k], share, nbbad[k] ? " (< --nb-min: every covered vertex fails)" : "");
        }
    }
    const double ctstep = ctum > 0 ? ctum : 1; const int nprof = ctum > 0 ? 2 * (int)ceil(30 / ctstep) + 1 : 0;
    for (int k = 0; cz && k < nn; k++) {   /* the CT chunks' last wraps */
        sf_wrap s; if (sf_load(argv[2], names[k], &s)) continue;
        ct_wrap(&ctc, cz, &s, &F, nullptr, nullptr, nullptr, nprof, ctum, ctR, ctstep, sf_orient(&s, &F, ms->global), nt, k, 1);
        sf_free(&s);
    }
    double area_all = 0, area_keep = 0, area_g[7] = {0}; size_t nv_all = 0, nv_keep = 0, nv_g[7] = {0}; int nwr = 0;
    if (per) printf("  wrap  vertices  kept   area_cm2  kept_cm2\n");
    for (int k = 0; k < nn; k++) {
        sf_wrap s; if (sf_load(argv[2], names[k], &s)) continue;
        const size_t n = (size_t)s.w * s.h; float *e = calloc(n, 4), *cf = calloc(n, 4), *ct = nprof ? malloc(n * nprof * 4) : nullptr;
        for (size_t i = 0; i < n * nprof; i++) ct[i] = NAN;
        const int sign = sf_orient(&s, &F, ms->global);
        { ss_job J = {&s, &F, e, cf, nullptr, 0, 0, 0, 0, nullptr, {0}, {0}, 0, s.h, 0, s.w, 0, 0, sign, nullptr, nullptr, nullptr, 0};
          pthread_t th[64]; const int m = nt < 64 ? (nt > 0 ? nt : 1) : 64; for (int q = 0; q < m; q++) pthread_create(&th[q], nullptr, ss_rows, &J); for (int q = 0; q < m; q++) pthread_join(th[q], nullptr); }
        if (cz && ct_wrap(&ctc, cz, &s, &F, e, cf, ct, nprof, ctum, ctR, ctstep, sign, nt, k, 0)) fprintf(stderr, "self-gate: %s\n", z3_error());
        uint8_t *gate = calloc(n, 1), *ok = malloc(n), *valid = malloc(n);
        for (size_t i = 0; i < n; i++) valid[i] = s.Z[i] >= 0;
        if (nprof) for (size_t i = 0; i < n; i++) { if (!valid[i]) continue; double mx = -1; int any = 0;
            for (int q = 0; q < nprof; q++) { const float v = ct[i * nprof + q]; const double t = (q - (nprof - 1) / 2.0) * ctstep; if (!isnan(v) && fabs(t) <= ctR + 1e-9) { mx = fmax(mx, v); any = 1; } }
            if (!any || mx < ctlev) gate[i] |= SG_PAP; }
        { sg_job J = {.s = &s, .F = &F, .ms = ms, .qb = qbx, .sign = sign, .ratio = ratio, .cmin = cmin, .emax = emax, .e = e, .cf = cf, .gate = gate,
                      .nnb = nnb, .nb = nb, .nbk = nbk, .nbtol = nbtol, .nbbad = nbbad, .nex = nex, .ex = ex, .pass = 1};
          pthread_t th[64]; const int m = nt < 64 ? (nt > 0 ? nt : 1) : 64; for (int q = 0; q < m; q++) pthread_create(&th[q], nullptr, sg_rows, &J); for (int q = 0; q < m; q++) pthread_join(th[q], nullptr); }
        for (size_t i = 0; i < n; i++) ok[i] = valid[i] && !gate[i];
        for (int it = 0; it < erode; it++) {   /* drop vertices at a gated / invalid 4-neighbour */
            uint8_t *o2 = malloc(n); memcpy(o2, ok, n);
            for (size_t i = 0; i < n; i++) if (ok[i]) { const int r = (int)(i / s.w), c = (int)(i % s.w);
                if (c == 0 || c + 1 == s.w || r == 0 || r + 1 == s.h || !ok[i - 1] || !ok[i + 1] || !ok[i - s.w] || !ok[i + s.w]) o2[i] = 0; }
            for (size_t i = 0; i < n; i++) if (ok[i] && !o2[i]) gate[i] |= SG_SMALL;
            free(o2); for (size_t i = 0; i < n; i++) ok[i] = valid[i] && !gate[i];
        }
        { int *lab = malloc(n * sizeof(int)), *stk = malloc(n * sizeof(int)); for (size_t i = 0; i < n; i++) lab[i] = -1;
          for (size_t i0 = 0; i0 < n; i0++) {   /* 4-connected stretches; small ones gated */
              if (!ok[i0] || lab[i0] >= 0) continue;
              int sp = 0, ncomp = 0; stk[sp++] = (int)i0; lab[i0] = (int)i0; double area = 0;
              while (sp) { const int i = stk[--sp], r = i / s.w, c = i % s.w; area += sg_quad_area(&s, ok, (size_t)i); stk[n - 1 - ncomp++] = i;
                  const int nbr[4] = {c > 0 ? i - 1 : -1, c + 1 < s.w ? i + 1 : -1, r > 0 ? i - s.w : -1, r + 1 < s.h ? i + s.w : -1};
                  for (int q = 0; q < 4; q++) if (nbr[q] >= 0 && ok[nbr[q]] && lab[nbr[q]] < 0) { lab[nbr[q]] = (int)i0; stk[sp++] = nbr[q]; } }
              if (area < minarea) for (int q = 0; q < ncomp; q++) gate[stk[n - 1 - q]] |= SG_SMALL;
          }
          free(lab); free(stk); }
        /* areas: a quad belongs to its top-left vertex (all corners valid); kept when all corners pass */
        double aw = 0, ak = 0; size_t vw = 0, vk = 0;
        for (size_t i = 0; i < n; i++) ok[i] = valid[i] && !gate[i];
        for (size_t i = 0; i < n; i++) { if (!valid[i]) continue; vw++; vk += ok[i];
            const double a = sg_quad_area(&s, valid, i); aw += a; if (ok[i]) ak += sg_quad_area(&s, ok, i);
            for (int b = 0; b < 7; b++) if (gate[i] & (1 << b)) { nv_g[b]++; area_g[b] += a; } }
        nv_all += vw; nv_keep += vk; area_all += aw; area_keep += ak;
        if (per) printf("  w%03d %9zu  %.3f  %8.3f  %8.3f\n", s.id, vw, vw ? (double)vk / vw : 0, aw, ak);
        if (vk > 0) {
            float *X = malloc(n * 4), *Y = malloc(n * 4), *Z = malloc(n * 4), *Q = malloc(n * 4);
            for (size_t i = 0; i < n; i++) { X[i] = ok[i] ? s.X[i] : -1; Y[i] = ok[i] ? s.Y[i] : -1; Z[i] = ok[i] ? s.Z[i] : -1; Q[i] = ok[i] ? (float)s.id : NAN; }
            char wd[1400], qp[1500]; snprintf(wd, sizeof wd, "%s/%s", argv[3], s.name); snprintf(qp, sizeof qp, "%s/q.tif", wd);
            if (tifxyz_write(wd, s.w, s.h, X, Y, Z, s.step, s.um, s.id, "ufsm self-gate (gated fit wrap)") || tiffw_write(qp, s.w, s.h, 3, Q, 1)) { fprintf(stderr, "self-gate: cannot write %s\n", wd); return 1; }
            nwr++; free(X); free(Y); free(Z); free(Q);
        }
        free(gate); free(ok); free(valid); free(e); free(cf); free(ct); sf_free(&s);
    }
    static const char *gn[7] = {"conf/phase", "papyrus", "fault", "backwards", "neighbours", "excluded", "small/edge"};
    printf("self-gate: failing (vertex share / area cm2, a vertex may fail several):");
    for (int b = 0; b < 7; b++) printf(" %s %.3f / %.3f;", gn[b], nv_all ? (double)nv_g[b] / nv_all : 0, area_g[b]);
    printf("\nself-gate: kept %zu of %zu vertices (%.3f), area %.3f of %.3f cm2 (%.3f), %d wraps written to %s\n", nv_keep, nv_all, nv_all ? (double)nv_keep / nv_all : 0,
           area_keep, area_all, area_all > 0 ? area_keep / area_all : 0, nwr, argv[3]);
    for (int k = 0; k < nnb; k++) fit_set_free(nb[k]);
    for (int i = 0; i < nqb; i++) st_box_free(&qbx[i]);
    free(qbx); free(qe);
    free(names); sff_free(&F); fit_set_free(ms); if (cz) { ctc_free(&ctc); z3_close(cz); }
    return 0;
}
