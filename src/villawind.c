/* ufsm villawind: the villa winding model (src/villawind.h) on ray-aligned slabs of a ~9.6 um CT.

   ufsm villawind --ckpt PTH --raw-in DIR --raw-out DIR [--T 128 --L 384] [--reps 1]
     one slab from DIR/image.u8 + DIR/valid.u8 ([T][T][L]); writes the parity intermediates (tests/test_villawind.py) and timing.
   ufsm villawind ray-check ROOT KEY WINDINGS.json --ckpt PTH [--tsv OUT] [--zr Z0,Z1] [--cache DIR] [--gpu 0] [--threads 8]
     the turn count benchmark of ufsm ray-check (src/raycheck.c) for the model itself: for every pair of consecutive wraps of a
     relative_windings.json collection (points [x, y, z] in the CT's voxels), one slab whose ray runs from a to b, centred on the
     segment's midpoint (as infer_winding_volume.py centres its slabs on the seed); turns = phase(b) - phase(a) on the central ray
     (the mean of the 4 columns around the slab axis, linear along the ray). Pairs longer than L - 8 are skipped.

   Slab geometry (villa volume_slab_extractor.py): direction d (unit, xyz), reference = the unit axis of d's smallest |component|,
   axis_a = normalise(ref x d), axis_b = d x axis_a, voxel (i, j, k) at origin + i axis_a + j axis_b + k d with
   origin = ray_origin - (T - 1) / 2 (axis_a + axis_b); trilinear sampling (rounded to uint8), valid = inside the volume. */
#include "villawind.h"
#include "vwt.h"
#include "wtrunk.h"
#include "nn.h"
#include "attn.h"
#include "json.h"
#include "store.h"
#include "xnet.h"
#include "zarr3.h"
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}
static int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

typedef struct { double o[3], a[3], b[3], d[3]; } vw_frame;   /* xyz */
static void cross3(const double *u, const double *v, double *w) { w[0] = u[1] * v[2] - u[2] * v[1]; w[1] = u[2] * v[0] - u[0] * v[2]; w[2] = u[0] * v[1] - u[1] * v[0]; }
static void frame_of(const double dir[3], const double ray_origin[3], int T, vw_frame *f) {
    const double n = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    for (int c = 0; c < 3; c++) f->d[c] = dir[c] / n;
    double ref[3] = {0, 0, 0}; int m = 0; for (int c = 1; c < 3; c++) if (fabs(f->d[c]) < fabs(f->d[m])) m = c; ref[m] = 1;
    cross3(ref, f->d, f->a); const double na = sqrt(f->a[0] * f->a[0] + f->a[1] * f->a[1] + f->a[2] * f->a[2]); for (int c = 0; c < 3; c++) f->a[c] /= na;
    cross3(f->d, f->a, f->b);
    for (int c = 0; c < 3; c++) f->o[c] = ray_origin[c] - 0.5 * (T - 1) * (f->a[c] + f->b[c]);
}

/* the CT box covering a slab (zyx, clamped to the volume) */
typedef struct { const uint8_t *box; int64_t bo[3], bn[3], V[3]; const vw_frame *f; int T, L, i0, i1; uint8_t *im, *va; double s, s2, cnt; } samp_job;
static void *samp_run(void *arg) {
    samp_job *J = arg; const vw_frame *f = J->f; double s = 0, s2 = 0, cnt = 0;
    for (int i = J->i0; i < J->i1; i++) for (int j = 0; j < J->T; j++) {
        double p[3]; for (int c = 0; c < 3; c++) p[c] = f->o[c] + i * f->a[c] + j * f->b[c];
        for (int k = 0; k < J->L; k++) {
            const double x = p[0] + k * f->d[0], y = p[1] + k * f->d[1], z = p[2] + k * f->d[2];
            const size_t q = ((size_t)i * J->T + j) * J->L + k;
            if (!(x >= 0 && y >= 0 && z >= 0 && x <= J->V[2] - 1 && y <= J->V[1] - 1 && z <= J->V[0] - 1)) { J->im[q] = 0; J->va[q] = 0; continue; }
            const double lz = z - J->bo[0], ly = y - J->bo[1], lx = x - J->bo[2];
            int iz = (int)floor(lz), iy = (int)floor(ly), ix = (int)floor(lx);
            iz = clampi(iz, 0, (int)J->bn[0] - 2); iy = clampi(iy, 0, (int)J->bn[1] - 2); ix = clampi(ix, 0, (int)J->bn[2] - 2);
            const double fz = lz - iz, fy = ly - iy, fx = lx - ix;
            const uint8_t *b = J->box + ((size_t)iz * J->bn[1] + iy) * J->bn[2] + ix; const size_t sy = J->bn[2], sz = (size_t)J->bn[1] * J->bn[2];
            const double v = (1 - fz) * ((1 - fy) * ((1 - fx) * b[0] + fx * b[1]) + fy * ((1 - fx) * b[sy] + fx * b[sy + 1]))
                           + fz * ((1 - fy) * ((1 - fx) * b[sz] + fx * b[sz + 1]) + fy * ((1 - fx) * b[sz + sy] + fx * b[sz + sy + 1]));
            const int u = (int)lround(v); J->im[q] = (uint8_t)(u < 0 ? 0 : u > 255 ? 255 : u); J->va[q] = 1;
            s += J->im[q]; s2 += (double)J->im[q] * J->im[q]; cnt += 1;
        }
    }
    J->s = s; J->s2 = s2; J->cnt = cnt; return nullptr;
}
/* sample one slab into im / va (host, [T][T][L]); mean and 1 / (std + 1e-6) over the valid voxels. -1 on a read error. */
static int sample_slab(z3 *ct, const vw_frame *f, int T, int L, int nth, uint8_t *im, uint8_t *va, uint8_t **buf, size_t *bcap, float *mean, float *inv) {
    const int64_t *V = z3_meta_of(ct)->shape; double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
    for (int c = 0; c < 8; c++) {
        const double ii = (c & 1) ? T - 1 : 0, jj = (c & 2) ? T - 1 : 0, kk = (c & 4) ? L - 1 : 0;
        for (int e = 0; e < 3; e++) { const double v = f->o[e] + ii * f->a[e] + jj * f->b[e] + kk * f->d[e]; lo[e] = fmin(lo[e], v); hi[e] = fmax(hi[e], v); }
    }
    int64_t bo[3], bn[3]; int empty = 0;
    for (int e = 0; e < 3; e++) {   /* zyx from xyz */
        const int x = 2 - e; int64_t a = (int64_t)floor(lo[x]) - 1, b = (int64_t)ceil(hi[x]) + 2;
        a = a < 0 ? 0 : a; b = b > V[e] ? V[e] : b;
        if (b - a < 2) { empty = 1; a = 0; b = 2; }
        bo[e] = a; bn[e] = b - a;
    }
    const size_t need = (size_t)bn[0] * bn[1] * bn[2];
    if (need > *bcap) { free(*buf); *buf = malloc(need); *bcap = need; if (!*buf) return -1; }
    if (empty) memset(*buf, 0, need); else if (z3_read(ct, bo, bn, *buf, nth)) return -1;
    nth = clampi(nth, 1, 64);
    samp_job J[64]; pthread_t th[64]; double s = 0, s2 = 0, cnt = 0;
    for (int t = 0; t < nth; t++) {
        J[t] = (samp_job){*buf, {bo[0], bo[1], bo[2]}, {bn[0], bn[1], bn[2]}, {V[0], V[1], V[2]}, f, T, L, T * t / nth, T * (t + 1) / nth, im, va, 0, 0, 0};
        pthread_create(&th[t], nullptr, samp_run, &J[t]);
    }
    for (int t = 0; t < nth; t++) { pthread_join(th[t], nullptr); s += J[t].s; s2 += J[t].s2; cnt += J[t].cnt; }
    const double m = cnt > 0 ? s / cnt : 0, var = cnt > 0 ? fmax(0, s2 / cnt - m * m) : 0;
    *mean = (float)m; *inv = (float)(1.0 / (sqrt(var) + 1e-6));
    return 0;
}

static int write_file(const char *dir, const char *name, const void *p, size_t n) {
    char path[1400]; snprintf(path, sizeof path, "%s/%s", dir, name); FILE *f = fopen(path, "wb");
    if (!f || fwrite(p, 1, n, f) != n) { fprintf(stderr, "villawind: cannot write %s\n", path); if (f) fclose(f); return -1; }
    fclose(f); return 0;
}

static int raw_mode(vw_net *net, int argc, char **argv) {
    const char *in = opt(argc, argv, "--raw-in", nullptr), *out = opt(argc, argv, "--raw-out", nullptr);
    const int T = atoi(opt(argc, argv, "--T", "128")), L = atoi(opt(argc, argv, "--L", "384")), reps = atoi(opt(argc, argv, "--reps", "1"));
    if (!in || !out) { fprintf(stderr, "villawind --raw-in DIR --raw-out DIR [--T 128 --L 384]\n"); return 2; }
    const size_t nv = (size_t)T * T * L; uint8_t *im = malloc(nv), *va = malloc(nv); char p[1400];
    for (int k = 0; k < 2; k++) {
        snprintf(p, sizeof p, "%s/%s", in, k ? "valid.u8" : "image.u8"); FILE *f = fopen(p, "rb");
        if (!f || fread(k ? va : im, 1, nv, f) != nv) { fprintf(stderr, "villawind: %s is not %zu bytes\n", p, nv); return 1; } fclose(f);
    }
    double s = 0, s2 = 0, c = 0; for (size_t i = 0; i < nv; i++) if (va[i]) { s += im[i]; c += 1; }
    const double m = c > 0 ? s / c : 0; for (size_t i = 0; i < nv; i++) if (va[i]) s2 += (im[i] - m) * (im[i] - m);
    const float mean = (float)m, inv = (float)(1.0 / (sqrt(c > 0 ? s2 / c : 0) + 1e-6));
    { char cmd[1500]; snprintf(cmd, sizeof cmd, "mkdir -p '%s'", out); if (system(cmd)) return 1; }
    uint8_t *imd = xmalloc_dev(nv), *vad = xmalloc_dev(nv); float *ph = xmalloc_dev(nv * 4);
    xh2d(imd, im, nv); xh2d(vad, va, nv);
    if (vw_forward(net, imd, vad, T, L, mean, inv, ph, out)) return 1;
    double best = 1e30;
    for (int r = 0; r < reps; r++) { xnet_sync(); const double t0 = now_s(); if (vw_forward(net, imd, vad, T, L, mean, inv, ph, nullptr)) return 1; xnet_sync(); best = fmin(best, now_s() - t0); }
    if (reps > 0) fprintf(stderr, "villawind: slab %dx%dx%d forward %.1f ms (best of %d), %.1f Mvox/s, xnet conv %.2f TMAC so far\n", T, T, L, best * 1e3, reps, nv / best / 1e6, xnet_macs() / 1e12);
    char st[128]; snprintf(st, sizeof st, "%.9g %.9g\n", mean, inv);
    if (write_file(out, "stats.txt", st, strlen(st))) return 1;
    free(im); free(va); xfree_dev(imd); xfree_dev(vad); xfree_dev(ph);
    return 0;
}

/* ---- ray-check ---- */
typedef struct { double a[3], b[3], dw, sp; } rpair;   /* xyz */
static int rp_cmpw(const void *x, const void *y) { const double a = ((const double *)x)[3], b = ((const double *)y)[3]; return (a > b) - (a < b); }
static int rp_cmpd(const void *x, const void *y) { const double a = *(const double *)x, b = *(const double *)y; return (a > b) - (a < b); }
static rpair *load_pairs(const char *path, size_t *np_out) {
    FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr, "villawind: cannot read %s\n", path); return nullptr; }
    fseek(f, 0, SEEK_END); const long L = ftell(f); fseek(f, 0, SEEK_SET); char *txt = malloc(L + 1);
    if (fread(txt, 1, L, f) != (size_t)L) { fclose(f); free(txt); return nullptr; } fclose(f); txt[L] = 0;
    json *J = json_parse(txt, L); free(txt); const json *C = J ? json_get(J, "collections") : nullptr;
    if (!C || C->type != J_OBJ) { fprintf(stderr, "villawind: no \"collections\" in %s\n", path); return nullptr; }
    size_t np = 0, cap = 1024; rpair *P = malloc(cap * sizeof *P);
    for (size_t c = 0; c < C->n; c++) {   /* as src/raycheck.c: points sorted by wind_a, consecutive pairs with dw > 0 */
        const json *pts = json_get(C->items[c], "points"); if (!pts || pts->type != J_OBJ || pts->n < 2) continue;
        double (*v)[4] = malloc(pts->n * sizeof *v); size_t n = 0;
        for (size_t k = 0; k < pts->n; k++) {
            const json *p = json_get(pts->items[k], "p"), *w = json_get(pts->items[k], "wind_a");
            if (!p || !w || w->type != J_NUM) continue;
            for (int d = 0; d < 3; d++) v[n][d] = json_num(json_at(p, d), 0);
            v[n][3] = json_num(w, 0); n++;
        }
        qsort(v, n, sizeof *v, rp_cmpw);
        for (size_t k = 0; k + 1 < n; k++) {
            const double dw = v[k + 1][3] - v[k][3]; if (dw <= 0) continue;
            if (np == cap) { cap *= 2; P = realloc(P, cap * sizeof *P); }
            rpair *q = &P[np++]; for (int d = 0; d < 3; d++) { q->a[d] = v[k][d]; q->b[d] = v[k + 1][d]; }
            q->dw = dw; q->sp = sqrt(pow(q->a[0] - q->b[0], 2) + pow(q->a[1] - q->b[1], 2) + pow(q->a[2] - q->b[2], 2)) / dw;
        }
        free(v);
    }
    json_free(J); *np_out = np; return P;
}
static double ray_phase(const float *ph, int T, int L, double k) {   /* mean of the 4 central columns, linear along the ray */
    k = k < 0 ? 0 : k > L - 1 ? L - 1 : k;
    const int k0 = (int)floor(k) < L - 1 ? (int)floor(k) : L - 2; const double t = k - k0; double s = 0;
    for (int di = 0; di < 2; di++) for (int dj = 0; dj < 2; dj++) {
        const float *c = ph + ((size_t)(T / 2 - 1 + di) * T + (T / 2 - 1 + dj)) * L;
        s += (1 - t) * c[k0] + t * c[k0 + 1];
    }
    return s / 4;
}
/* trilinear phase at a world point (xyz) of a slab; NAN outside */
static double point_phase(const float *ph, int T, int L, const vw_frame *f, const double p[3]) {
    double u[3] = {0, 0, 0}; const double *ax[3] = {f->a, f->b, f->d};
    for (int e = 0; e < 3; e++) for (int c = 0; c < 3; c++) u[e] += (p[c] - f->o[c]) * ax[e][c];
    const int n[3] = {T, T, L}; int i0[3]; double t[3];
    for (int e = 0; e < 3; e++) { if (u[e] < 0 || u[e] > n[e] - 1) return NAN; i0[e] = (int)floor(u[e]); if (i0[e] > n[e] - 2) i0[e] = n[e] - 2; t[e] = u[e] - i0[e]; }
    double s = 0;
    for (int c = 0; c < 8; c++) {
        const int di = c & 1, dj = c >> 1 & 1, dk = c >> 2 & 1;
        s += (di ? t[0] : 1 - t[0]) * (dj ? t[1] : 1 - t[1]) * (dk ? t[2] : 1 - t[2]) * ph[((size_t)(i0[0] + di) * T + i0[1] + dj) * L + i0[2] + dk];
    }
    return s;
}
/* umbilicus (control points x y z, linear in z, clamped) */
typedef struct { int n; double (*p)[3]; } umb_t;
static int umb_load(const char *path, umb_t *u) {
    FILE *f = fopen(path, "rb"); if (!f) return -1; fseek(f, 0, SEEK_END); const long L = ftell(f); fseek(f, 0, SEEK_SET);
    char *t = malloc(L + 1); if (fread(t, 1, L, f) != (size_t)L) { fclose(f); return -1; } fclose(f); t[L] = 0;
    json *J = json_parse(t, L); free(t); const json *cp = J ? json_get(J, "control_points") : nullptr; if (!cp || !cp->n) return -1;
    u->n = (int)cp->n; u->p = malloc(cp->n * sizeof *u->p);
    for (size_t i = 0; i < cp->n; i++) { const json *q = json_at(cp, i); u->p[i][0] = json_num(json_get(q, "x"), 0); u->p[i][1] = json_num(json_get(q, "y"), 0); u->p[i][2] = json_num(json_get(q, "z"), 0); }
    for (int i = 1; i < u->n; i++) for (int j = i; j > 0 && u->p[j][2] < u->p[j - 1][2]; j--) { double tt[3]; memcpy(tt, u->p[j], 24); memcpy(u->p[j], u->p[j - 1], 24); memcpy(u->p[j - 1], tt, 24); }
    json_free(J); return 0;
}
static void umb_at(const umb_t *u, double z, double *x, double *y) {
    if (z <= u->p[0][2]) { *x = u->p[0][0]; *y = u->p[0][1]; return; }
    for (int i = 1; i < u->n; i++) if (z <= u->p[i][2]) {
        const double t = (z - u->p[i - 1][2]) / fmax(1e-9, u->p[i][2] - u->p[i - 1][2]);
        *x = u->p[i - 1][0] + t * (u->p[i][0] - u->p[i - 1][0]); *y = u->p[i - 1][1] + t * (u->p[i][1] - u->p[i - 1][1]); return; }
    *x = u->p[u->n - 1][0]; *y = u->p[u->n - 1][1];
}

static vwt *g_vwt = nullptr;   /* --vwt: our trained model instead of villa's */
static wtrunk *g_wt = nullptr; static uint16_t *g_xf = nullptr; static int g_xld = 0; static double g_um = 9.6;   /* --trunk: the frozen feature trunk of a cx model */
static int ray_check(vw_net *net, int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm villawind ray-check ROOT KEY WINDINGS.json --ckpt PTH [--tsv OUT] [--zr Z0,Z1] [--cache DIR] [--threads 8] [--prefetch 1] [--geom pair|umb --umb JSON --sz 112 --sr 192 --st 112 --r0 16]\n"); return 2; }
    const int T = atoi(opt(argc, argv, "--T", "128")), L = atoi(opt(argc, argv, "--L", "384")), nth = atoi(opt(argc, argv, "--threads", "8"));
    double zr0 = -1e18, zr1 = 1e18; sscanf(opt(argc, argv, "--zr", ""), "%lf,%lf", &zr0, &zr1);
    const int prefetch = atoi(opt(argc, argv, "--prefetch", "0"));   /* 1: read (and cache) the slabs' CT only, no forward */
    /* --geom pair (default): the slab centred on the pair, ray a -> b (oracle geometry). --geom umb --umb JSON: production tiling, radial
       horizontal rays from the umbilicus on a fixed grid (z bands of --sz, rings r0 = --r0 + m --sr, angle steps --st / r_mid), the
       pair's points wherever they fall in the slab (phase trilinear at the points; pairs not inside one ring's interior are skipped) */
    const int geom_umb = !strcmp(opt(argc, argv, "--geom", "pair"), "umb");
    const double sz = atof(opt(argc, argv, "--sz", "112")), sr = atof(opt(argc, argv, "--sr", "192")), stt = atof(opt(argc, argv, "--st", "112")), r00 = atof(opt(argc, argv, "--r0", "16"));
    umb_t umb = {0}; if ((geom_umb || g_wt) && umb_load(opt(argc, argv, "--umb", ""), &umb)) { fprintf(stderr, "villawind: --geom umb and --trunk need --umb umbilicus.json\n"); return 2; }
    store *st = store_open(argv[2]); z3 *ct = st ? z3_open(st, argv[3], opt(argc, argv, "--cache", nullptr)) : nullptr;
    if (!ct) { fprintf(stderr, "villawind: %s %s: %s\n", argv[2], argv[3], z3_error()); return 1; }
    size_t np = 0; rpair *P = load_pairs(argv[4], &np); if (!P) return 1;
    const size_t nv = (size_t)T * T * L; uint8_t *im = malloc(nv), *va = malloc(nv), *buf = nullptr; size_t bcap = 0;
    float *ph = malloc(nv * 4), *phd = xmalloc_dev(nv * 4); uint8_t *imd = xmalloc_dev(nv), *vad = xmalloc_dev(nv);
    double *err = malloc(np * sizeof *err); uint8_t *ok = calloc(np, 1), *pk = malloc(np);
    FILE *tsv = *opt(argc, argv, "--tsv", "") ? fopen(opt(argc, argv, "--tsv", ""), "w") : nullptr;
    if (tsv) fprintf(tsv, "z\ty\tx\tdw\tsp_vx\tvw_turns\tvalid_frac\n");
    double tsamp = 0, tfwd = 0, t0 = now_s(); size_t nlong = 0, nrun = 0;
    for (size_t i = 0; i < np; i++) {
        const rpair *q = &P[i]; pk[i] = q->sp < 15;
        const double zm = 0.5 * (q->a[2] + q->b[2]); if (zm < zr0 || zm >= zr1) continue;
        double d[3], len = 0; for (int c = 0; c < 3; c++) { d[c] = q->b[c] - q->a[c]; len += d[c] * d[c]; } len = sqrt(len);
        if (len < 1e-6) continue;
        if (len > L - 8) { nlong++; continue; }
        const double mid = 0.5 * (L - 1); double ro[3]; for (int c = 0; c < 3; c++) ro[c] = 0.5 * (q->a[c] + q->b[c]) - mid * d[c] / len;
        vw_frame f; frame_of(d, ro, T, &f);
        if (geom_umb) {   /* production tiling: radial horizontal ray from the umbilicus, slabs on a fixed (z, ring, angle) grid */
            const double pm[3] = {0.5 * (q->a[0] + q->b[0]), 0.5 * (q->a[1] + q->b[1]), 0.5 * (q->a[2] + q->b[2])};
            const double zc = sz * floor(pm[2] / sz) + 0.5 * sz; double ux, uy; umb_at(&umb, zc, &ux, &uy);
            double ra = hypot(q->a[0] - ux, q->a[1] - uy), rb = hypot(q->b[0] - ux, q->b[1] - uy); const double rlo = fmin(ra, rb), rhi = fmax(ra, rb);
            /* ring: r0 = r00 + m sr (ray covers r0 .. r0 + L - 1); the ring whose central part holds both points best */
            int m = (int)floor((0.5 * (rlo + rhi) - r00 - 0.5 * (L - 1)) / sr + 0.5); if (m < 0) m = 0;
            const double r0 = r00 + m * sr, rmid = r0 + 0.5 * (L - 1);
            const double th = atan2(pm[1] - uy, pm[0] - ux), dth = stt / rmid, thc = dth * floor(th / dth + 0.5);
            const double dd[3] = {cos(thc), sin(thc), 0}, o2[3] = {ux + r0 * dd[0], uy + r0 * dd[1], zc};
            frame_of(dd, o2, T, &f);
            if (rlo < r0 + 4 || rhi > r0 + L - 5) { nlong++; continue; }
        }
        double ta = now_s(); float mean, inv;
        if (sample_slab(ct, &f, T, L, nth, im, va, &buf, &bcap, &mean, &inv)) { fprintf(stderr, "villawind: CT read: %s\n", z3_error()); return 1; }
        double tb = now_s(); tsamp += tb - ta;
        if (prefetch) { nrun++; if (nrun % 100 == 0) fprintf(stderr, "villawind: prefetched %zu / %zu\n", nrun, np); continue; }
        xh2d(imd, im, nv); xh2d(vad, va, nv);
        if (g_wt) {   /* the trunk features in-slab, the axis point at the slab centre's z */
            const double cz = f.o[2] + 0.5 * (T - 1) * (f.a[2] + f.b[2]) + 0.5 * (L - 1) * f.d[2];
            double ux, uy; umb_at(&umb, cz, &ux, &uy);
            wt_frame wf; for (int c = 0; c < 3; c++) { wf.o[c] = f.o[c]; wf.a[c] = f.a[c]; wf.b[c] = f.b[c]; wf.d[c] = f.d[c]; wf.u[c] = c == 2; }
            wf.c[0] = ux; wf.c[1] = uy; wf.c[2] = cz;
            if (wt_forward_ld(g_wt, imd, &wf, 1, g_um, g_xf, g_xld)) return 1;
        }
        if (g_vwt ? vwt_forward(g_vwt, imd, vad, phd, nullptr, 0) : vw_forward(net, imd, vad, T, L, mean, inv, phd, nullptr)) return 1;
        xd2h(ph, phd, nv * 4); tfwd += now_s() - tb;
        double turns = ray_phase(ph, T, L, mid + 0.5 * len) - ray_phase(ph, T, L, mid - 0.5 * len);
        if (geom_umb) { turns = fabs(point_phase(ph, T, L, &f, q->b) - point_phase(ph, T, L, &f, q->a)); if (isnan(turns)) { nlong++; continue; } }
        size_t nvld = 0; for (size_t k = 0; k < nv; k++) nvld += va[k];
        err[i] = turns - q->dw; ok[i] = 1; nrun++;
        if (tsv) fprintf(tsv, "%.0f\t%.0f\t%.0f\t%.0f\t%.1f\t%.3f\t%.3f\n", zm, 0.5 * (q->a[1] + q->b[1]), 0.5 * (q->a[0] + q->b[0]), q->dw, q->sp, turns, (double)nvld / nv);
        if (nrun % 100 == 0) fprintf(stderr, "villawind: %zu / %zu pairs, sample %.0f ms, forward %.0f ms per slab\n", nrun, np, tsamp / nrun * 1e3, tfwd / nrun * 1e3);
    }
    if (tsv) fclose(tsv);
    printf("villawind ray-check: %zu annotated pairs, %zu run, %zu longer than %d, %.1f s (sample %.0f ms, forward %.0f ms per slab)\n",
           np, nrun, nlong, L - 8, now_s() - t0, nrun ? tsamp / nrun * 1e3 : 0, nrun ? tfwd / nrun * 1e3 : 0);
    for (int sub = 0; sub < 3; sub++) {
        double *v = malloc((np ? np : 1) * sizeof *v); size_t n = 0, off = 0; double s = 0;
        for (size_t i = 0; i < np; i++) if (ok[i] && (sub == 0 || (sub == 1) == (pk[i] != 0))) { v[n++] = err[i]; s += err[i]; off += fabs(err[i]) >= 0.5; }
        qsort(v, n, sizeof *v, rp_cmpd);
        printf("villawind ray-check: phase   %-6s pairs %5zu  error median %+.3f mean %+.3f turns, |err| >= 0.5 turn %.3f\n", sub == 0 ? "all" : sub == 1 ? "packed" : "sparse",
               n, n ? v[n / 2] : 0.0, n ? s / n : 0.0, n ? (double)off / n : 0.0);
        free(v);
    }
    free(P); free(err); free(ok); free(pk); free(im); free(va); free(buf); free(ph); xfree_dev(phd); xfree_dev(imd); xfree_dev(vad); z3_close(ct);
    return 0;
}

int cmd_villawind(int argc, char **argv) {
    const char *ck = opt(argc, argv, "--ckpt", nullptr), *vk = opt(argc, argv, "--vwt", nullptr);
    if (vk) {   /* ufsm villawind ray-check ... --vwt RUN/step_N.vwt [--ema 1]: the same benchmark for our model (src/vwt.h) */
        if (nn_init(atoi(opt(argc, argv, "--gpu", "0"))) || xnet_init(atoi(opt(argc, argv, "--gpu", "0")))) return 1;
        at_set_core(1); setenv("UFSM_ATTN_TF32", "1", 0); at_set_mx(atoi(opt(argc, argv, "--mx", "16")));
        g_vwt = vwt_load(vk, atoi(opt(argc, argv, "--ema", "1"))); if (!g_vwt) return 1;
        g_xf = vwt_xfeat_buf(g_vwt, &g_xld);
        if (g_xf) {   /* a trunk model (cfg.cx > 0): --trunk CKPT [--trunk-feat 1] [--um 9.6] */
            const char *tk = opt(argc, argv, "--trunk", nullptr); g_um = atof(opt(argc, argv, "--um", "9.6"));
            if (!tk) { fprintf(stderr, "villawind: %s takes trunk features: --trunk CKPT (wind96-wk1-best.ckpt) and --umb\n", vk); return 2; }
            g_wt = wt_open(tk, atoi(opt(argc, argv, "--T", "128")), atoi(opt(argc, argv, "--L", "384")), 1, atoi(opt(argc, argv, "--trunk-feat", "1")));
            if (!g_wt) { fprintf(stderr, "villawind: cannot open the trunk %s\n", tk); return 1; }
        }
        if (argc > 2 && !strcmp(argv[2], "ray-check")) return ray_check(nullptr, argc - 1, argv + 1);
        fprintf(stderr, "villawind --vwt: only ray-check\n"); return 2;
    }
    if (!ck) {
        fprintf(stderr, "usage: ufsm villawind --ckpt ckpt_final.pth --raw-in DIR --raw-out DIR [--T 128 --L 384] [--reps 1]\n"
                        "       ufsm villawind ray-check ROOT KEY WINDINGS.json --ckpt PTH [--tsv OUT] [--zr Z0,Z1] [--cache DIR] [--gpu 0]\n"
                        "       ufsm villawind ray-check ROOT KEY WINDINGS.json --vwt RUN/step_N.vwt [--ema 1] ...   (our ray-attention model, src/vwt.h)\n"
                        "  villa winding model (HF scrollprize/winding_model_9um) on ray-aligned slabs of a 9.6 um CT\n");
        return 2;
    }
    if (xnet_init(atoi(opt(argc, argv, "--gpu", "0")))) return 1;
    if (opt(argc, argv, "--gpu-mem", nullptr)) xnet_set_budget((size_t)(atof(opt(argc, argv, "--gpu-mem", "8")) * 1e9));
    vw_net *net = vw_load(ck); if (!net) return 1;
    int rc;
    if (opt(argc, argv, "--raw-in", nullptr)) rc = raw_mode(net, argc, argv);
    else if (argc > 2 && !strcmp(argv[2], "ray-check")) rc = ray_check(net, argc - 1, argv + 1);
    else { fprintf(stderr, "villawind: unknown mode\n"); rc = 2; }
    vw_free(net);
    return rc;
}
