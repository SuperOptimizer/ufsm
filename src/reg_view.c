/* ufsm reg overlay: a coarse-scan plane next to the fine scan resampled onto it through a map (src/reg.h), as one JPEG:
   coarse | fine resampled / checkerboard | magenta-green composite (coarse magenta, fine green: grey where they agree). */
#include "reg.h"
#include "json.h"
#include "imgops.h"
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}
static char *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb"); if (!f) return nullptr;
    fseek(f, 0, SEEK_END); const long L = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc(L + 1); if (fread(b, 1, L, f) != (size_t)L) { fclose(f); free(b); return nullptr; }
    b[L] = 0; fclose(f); if (n) *n = L; return b;
}
/* the map of a map JSON (reg pair / reg fit output) */
int reg_load_map(const char *path, const char *cache, rvol **cv, rvol **fv, rmap *m, rfield *D) {
    size_t n; char *t = slurp(path, &n); if (!t) { fprintf(stderr, "reg: cannot read %s\n", path); return -1; }
    json *j = json_parse(t, n); free(t); if (!j) return -1;
    const json *A = json_get(j, "A");
    for (int i = 0; i < 3; i++) for (int k = 0; k < 4; k++) m->A.m[i][k] = json_num(json_at(json_at(A, i), k), 0);
    m->Ainv = raff_inv(m->A); m->D = nullptr; memset(D, 0, sizeof *D);
    const json *fd = json_get(j, "field");
    if (fd) {
        char dir[1024], fp[2100]; snprintf(dir, sizeof dir, "%s", path); char *s = strrchr(dir, '/'); if (s) *s = 0; else snprintf(dir, sizeof dir, ".");
        const char *fn = json_str(json_get(fd, "file"), "field.f32");
        if (fn[0] == '/') snprintf(fp, sizeof fp, "%s", fn); else snprintf(fp, sizeof fp, "%s/%s", dir, fn);
        for (int a = 0; a < 3; a++) { D->n[a] = (int)json_num(json_at(json_get(fd, "n"), a), 0); D->o[a] = json_num(json_at(json_get(fd, "o"), a), 0); }
        D->h = json_num(json_get(fd, "h"), 1);
        size_t got; D->d = (float *)slurp(fp, &got);
        if (!D->d || got != (size_t)D->n[0] * D->n[1] * D->n[2] * 3 * sizeof(float)) { fprintf(stderr, "reg: bad field %s\n", fp); json_free(j); return -1; }
        m->D = D;
    }
    const json *c = json_get(j, "coarse"), *f = json_get(j, "fine");
    if (cv) *cv = rvol_open(json_str(json_get(c, "root"), ""), json_str(json_get(c, "key"), ""), cache, json_num(json_get(c, "um"), 0));
    if (fv) *fv = rvol_open(json_str(json_get(f, "root"), ""), json_str(json_get(f, "key"), ""), cache, json_num(json_get(f, "um"), 0));
    json_free(j);
    return (cv && !*cv) || (fv && !*fv) ? -1 : 0;
}

/* plane geometry: axis a (0 z, 1 y, 2 x), level-L plane index p; image rows/cols are the other two axes in z, y, x order */
typedef struct { rvol *cv, *fv; const rmap *m; int L, lf, ss, axis, T; int64_t p, r0, c0, nr, ncol; float *img_c, *img_f; atomic_int next; } ojob;
static void plane_pt(const ojob *j, double r, double c, double pl[3]) {   /* level-L xyz of image (r, c) */
    double zyx[3];
    if (j->axis == 0) { zyx[0] = j->p; zyx[1] = r; zyx[2] = c; }
    else if (j->axis == 1) { zyx[0] = r; zyx[1] = j->p; zyx[2] = c; }
    else { zyx[0] = r; zyx[1] = c; zyx[2] = j->p; }
    pl[0] = zyx[2]; pl[1] = zyx[1]; pl[2] = zyx[0];
}
static void *ov_worker(void *arg) {
    ojob *j = arg; const int T = j->T;
    const int64_t ntr = (j->nr + T - 1) / T, ntc = (j->ncol + T - 1) / T;
    for (;;) {
        const int64_t t = atomic_fetch_add(&j->next, 1); if (t >= ntr * ntc) break;
        const int64_t tr = t / ntc * T, tc = t % ntc * T;
        const int hr = (int)(j->nr - tr < T ? j->nr - tr : T), hc = (int)(j->ncol - tc < T ? j->ncol - tc : T);
        /* tile as a 1-thick box at level L: origin xyz + dims zyx */
        double o0[3], o1[3]; plane_pt(j, j->r0 + tr, j->c0 + tc, o0); plane_pt(j, j->r0 + tr + hr - 1, j->c0 + tc + hc - 1, o1);
        const double org[3] = {fmin(o0[0], o1[0]), fmin(o0[1], o1[1]), fmin(o0[2], o1[2])};
        const int nz = (int)(fabs(o1[2] - o0[2]) + 1), ny = (int)(fabs(o1[1] - o0[1]) + 1), nx = (int)(fabs(o1[0] - o0[0]) + 1), nb[3] = {nz, ny, nx};
        /* coarse */
        const int64_t co[3] = {(int64_t)org[2], (int64_t)org[1], (int64_t)org[0]}, cn[3] = {nz, ny, nx};
        uint8_t *cu = malloc((size_t)nz * ny * nx);
        if (!rvol_read(j->cv, j->L, co, cn, cu))
            for (int r = 0; r < hr; r++) for (int c = 0; c < hc; c++) {
                double pl[3]; plane_pt(j, j->r0 + tr + r, j->c0 + tc + c, pl);
                const int z = (int)(pl[2] - org[2]), y = (int)(pl[1] - org[1]), x = (int)(pl[0] - org[0]);
                j->img_c[(size_t)(tr + r) * j->ncol + tc + c] = cu[((size_t)z * ny + y) * nx + x];
            }
        free(cu);
        /* fine resampled */
        rblk fb; int err = 0; uint8_t *fu = reg_read_fine(j->fv, j->lf, j->m, j->L, org, nb, 0, &fb, &err);
        if (fu) {
            float *g = malloc((size_t)nz * ny * nx * sizeof *g); const double zero[3] = {0, 0, 0};
            reg_resample(&fb, j->m, j->L, j->lf, org, nb, zero, j->ss, g);
            for (int r = 0; r < hr; r++) for (int c = 0; c < hc; c++) {
                double pl[3]; plane_pt(j, j->r0 + tr + r, j->c0 + tc + c, pl);
                const int z = (int)(pl[2] - org[2]), y = (int)(pl[1] - org[1]), x = (int)(pl[0] - org[0]);
                j->img_f[(size_t)(tr + r) * j->ncol + tc + c] = g[((size_t)z * ny + y) * nx + x];
            }
            free(g); free(fu);
        }
    }
    return nullptr;
}
static int fcmp(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return x < y ? -1 : x > y; }
static void stretch(const float *v, size_t n, float *lo, float *hi) {   /* 1st / 99th percentile of the non-zero pixels */
    float *t = malloc(n * sizeof *t); size_t k = 0; for (size_t i = 0; i < n; i++) if (v[i] > 0) t[k++] = v[i];
    if (k < 10) { *lo = 0; *hi = 255; free(t); return; }
    qsort(t, k, sizeof *t, fcmp); *lo = t[k / 100]; *hi = t[k - 1 - k / 100]; if (*hi <= *lo) *hi = *lo + 1; free(t);
}
static inline uint8_t u8(float v, float lo, float hi) { const float x = (v - lo) / (hi - lo) * 255.f; return (uint8_t)(x < 0 ? 0 : x > 255 ? 255 : x); }

int cmd_reg_overlay(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm reg overlay <map.json> <out.jpg> [--level 1] [--axis z|y|x] [--at P (coarse L0 voxels along the axis)] [--crop r0,c0,nr,nc (level voxels) | --size N (centred)]\n"
        "  [--check 32] [--threads 8] [--cache DIR] [--quality 88]\n"); return 2; }
    rvol *cv, *fv; rmap m; rfield D;
    if (reg_load_map(argv[3], opt(argc, argv, "--cache", nullptr), &cv, &fv, &m, &D)) return 1;
    ojob j = {0}; j.cv = cv; j.fv = fv; j.m = &m; j.L = atoi(opt(argc, argv, "--level", "1")); j.T = 128;
    if (j.L >= cv->nl) j.L = cv->nl - 1;
    const char *ax = opt(argc, argv, "--axis", "z"); j.axis = ax[0] == 'z' ? 0 : ax[0] == 'y' ? 1 : 2;
    const double cum = cv->um * (1 << j.L); j.lf = rvol_level_for(fv, cum);
    const double ratio = cum / (fv->um * (1 << j.lf)); j.ss = ratio > 1.3 ? 2 : 1;
    const int64_t *sh = cv->shape[j.L]; const double f = (double)(1 << j.L);
    /* default plane: the middle of the fine volume's extent along the axis */
    double mid[3]; { const double fm[3] = {fv->shape[0][2] / 2.0, fv->shape[0][1] / 2.0, fv->shape[0][0] / 2.0}; rmap_f2c(&m, fm, mid); }
    const double at = atof(opt(argc, argv, "--at", "-1")) >= 0 ? atof(opt(argc, argv, "--at", "-1")) : mid[2 - j.axis];
    j.p = (int64_t)llround((at + 0.5) / f - 0.5);
    const int ra = j.axis == 0 ? 1 : 0, ca = j.axis == 2 ? 1 : 2;
    j.r0 = 0; j.c0 = 0; j.nr = sh[ra]; j.ncol = sh[ca];
    const char *cr = opt(argc, argv, "--crop", nullptr);
    if (cr) { long long a, b, c, d; if (sscanf(cr, "%lld,%lld,%lld,%lld", &a, &b, &c, &d) == 4) { j.r0 = a; j.c0 = b; j.nr = c; j.ncol = d; } }
    else {   /* the fine volume's footprint on the plane */
        double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
        for (int k = 0; k < 8; k++) { const double p[3] = {k & 1 ? fv->shape[0][2] - 1 : 0, k & 2 ? fv->shape[0][1] - 1 : 0, k & 4 ? fv->shape[0][0] - 1 : 0}; double q[3]; rmap_f2c(&m, p, q); for (int d = 0; d < 3; d++) { lo[d] = fmin(lo[d], q[d]); hi[d] = fmax(hi[d], q[d]); } }
        const int64_t r0 = (int64_t)fmax(0, floor((lo[2 - ra] + 0.5) / f - 0.5)), r1 = (int64_t)fmin(sh[ra] - 1, ceil((hi[2 - ra] + 0.5) / f - 0.5));
        const int64_t c0 = (int64_t)fmax(0, floor((lo[2 - ca] + 0.5) / f - 0.5)), c1 = (int64_t)fmin(sh[ca] - 1, ceil((hi[2 - ca] + 0.5) / f - 0.5));
        j.r0 = r0; j.c0 = c0; j.nr = r1 - r0 + 1; j.ncol = c1 - c0 + 1;
        const int64_t sz = atoll(opt(argc, argv, "--size", "0"));   /* a centred sz x sz crop of the footprint */
        if (sz > 0) { if (j.nr > sz) { j.r0 += (j.nr - sz) / 2; j.nr = sz; } if (j.ncol > sz) { j.c0 += (j.ncol - sz) / 2; j.ncol = sz; } }
    }
    if (j.nr <= 0 || j.ncol <= 0 || j.nr * j.ncol > 64000000) { fprintf(stderr, "reg overlay: bad plane size %lld x %lld\n", (long long)j.nr, (long long)j.ncol); return 1; }
    const size_t N = (size_t)j.nr * j.ncol;
    j.img_c = calloc(N, sizeof *j.img_c); j.img_f = calloc(N, sizeof *j.img_f);
    const int nt = atoi(opt(argc, argv, "--threads", "8")); pthread_t th[64];
    for (int t = 0; t < nt && t < 64; t++) pthread_create(&th[t], nullptr, ov_worker, &j);
    for (int t = 0; t < nt && t < 64; t++) pthread_join(th[t], nullptr);
    float lc, hc, lf, hf; stretch(j.img_c, N, &lc, &hc); stretch(j.img_f, N, &lf, &hf);
    const int chk = atoi(opt(argc, argv, "--check", "32")), gap = 8;
    const int W = (int)(2 * j.ncol + gap), H = (int)(2 * j.nr + gap);
    uint8_t *rgb = malloc((size_t)W * H * 3); memset(rgb, 255, (size_t)W * H * 3);
    for (int64_t r = 0; r < j.nr; r++) for (int64_t c = 0; c < j.ncol; c++) {
        const size_t i = (size_t)r * j.ncol + c;
        const uint8_t a = u8(j.img_c[i], lc, hc), b = u8(j.img_f[i], lf, hf), k = ((r / chk + c / chk) & 1) ? a : b;
        uint8_t *p;
        p = rgb + ((size_t)r * W + c) * 3; p[0] = p[1] = p[2] = a;
        p = rgb + ((size_t)r * W + c + j.ncol + gap) * 3; p[0] = p[1] = p[2] = b;
        p = rgb + ((size_t)(r + j.nr + gap) * W + c) * 3; p[0] = p[1] = p[2] = k;
        p = rgb + ((size_t)(r + j.nr + gap) * W + c + j.ncol + gap) * 3; p[0] = a; p[1] = b; p[2] = a;
    }
    const int rc = jpeg_write_rgb(argv[4], W, H, rgb, atoi(opt(argc, argv, "--quality", "88")));
    fprintf(stderr, "reg overlay: %s plane %c = %lld at level %d (%.3f um; fine level %d), %lld x %lld px, panels: coarse | fine / checker %d | magenta coarse + green fine\n",
            argv[4], "zyx"[j.axis], (long long)j.p, j.L, cum, j.lf, (long long)j.nr, (long long)j.ncol, chk);
    free(rgb); free(j.img_c); free(j.img_f); free(D.d); rvol_close(cv); rvol_close(fv);
    return rc;
}

