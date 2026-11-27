/* ufsm umbilicus / axis-diff: the umbilicus by sheet-normal voting (src/umbfind.h; port of villa vc_gen_umbilicus). */
#include "umbfind.h"
#include "defaulted.h"
#include "json.h"
#include "optcheck.h"
#include "store.h"
#include "zarr3.h"
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(name))) return argv[i + 1];
    return dflt;
}
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

/* ---- random numbers: splitmix64, one stream per slice (seed + slice: runs repeat exactly at any thread count) ---- */
static uint64_t sm_next(uint64_t *s) { uint64_t z = (*s += 0x9E3779B97F4A7C15ull); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return z ^ (z >> 31); }
static double sm_unif(uint64_t *s) { return (double)(sm_next(s) >> 11) * (1.0 / 9007199254740992.0); }

void umb_cfg_default(umb_cfg *c, double um0) {
    const double v = 1000.0 / um0;   /* level-0 voxels per mm */
    *c = (umb_cfg){.nsamp = 10000, .ncand = 1000, .near = 0.94 * v, .step0 = 9.6 * v, .step_min = 1.0, .uncrush = 0, .max_moves = 100000};
}

static int cmp_dbl(const void *a, const void *b) { const double p = *(const double *)a, q = *(const double *)b; return (p > q) - (p < q); }
static double median_of(double *v, int n) {   /* villa median(): the mean of the two middle values when n is even (sorts v) */
    if (n <= 0) return NAN;
    qsort(v, (size_t)n, sizeof *v, cmp_dbl);
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/* ---- one slice (villa align_and_extract_umbilicus), in villa's (x, y) naming ---- */
int umb_estimate(const umb_pt *p, int n, const umb_cfg *c, uint64_t seed, double out[2], double *score, int *inside) {
    out[0] = out[1] = NAN; if (score) *score = 0; if (inside) *inside = 0;
    if (n < 1) return -1;
    double *cw = malloc((size_t)n * sizeof *cw), tot = 0;
    double lo_x = INFINITY, lo_y = INFINITY, hi_x = -INFINITY, hi_y = -INFINITY;
    if (!cw) return -1;
    for (int i = 0; i < n; i++) {
        tot += p[i].w; cw[i] = tot;
        lo_x = fmin(lo_x, p[i].x); hi_x = fmax(hi_x, p[i].x); lo_y = fmin(lo_y, p[i].y); hi_y = fmax(hi_y, p[i].y);
    }
    if (!(tot > 0)) { free(cw); return -1; }
    double min_x = lo_x, max_x = hi_x, min_y = lo_y, max_y = hi_y;
    /* un-crush (villa 3b): W = diag((l / l_mean)^(beta / 2)) V^T about the (weighted) mean, from the cell-balanced normal tensor */
    double w00 = 1, w01 = 0, w10 = 0, w11 = 1, mu_x = 0, mu_y = 0; int whiten = 0;
    if (c->uncrush > 0) {
        for (int i = 0; i < n; i++) { mu_x += p[i].w * p[i].x; mu_y += p[i].w * p[i].y; }
        mu_x /= tot; mu_y /= tot;
        enum { K = 32, NB = 12 };
        const double cell = fmax(1.0, fmax(hi_x - lo_x, hi_y - lo_y) / K);
        double *cs = calloc((K + 1) * (K + 1), sizeof *cs);
        if (!cs) { free(cw); return -1; }
#define CELL(i) ((size_t)((int)fmin(K, (p[i].y - lo_y) / cell) * (K + 1) + (int)fmin(K, (p[i].x - lo_x) / cell)))
        for (int i = 0; i < n; i++) cs[CELL(i)] += p[i].w;
        double txx = 0, txy = 0, tyy = 0, bins[NB] = {0};
        for (int i = 0; i < n; i++) {
            const double w = p[i].w / cs[CELL(i)], nx = p[i].nx, ny = p[i].ny;
            txx += w * nx * nx; txy += w * nx * ny; tyy += w * ny * ny;
            double dir = atan2(ny, nx); if (dir < 0) dir += M_PI; if (dir >= M_PI) dir -= M_PI;
            bins[(int)fmin(NB - 1, dir / (M_PI / NB))] += w;
        }
#undef CELL
        free(cs);
        double bt = 0, bmin = INFINITY; for (int b = 0; b < NB; b++) { bt += bins[b]; bmin = fmin(bmin, bins[b]); }
        const int half_turn = bt > 0 && bmin * NB >= 0.01 * bt;
        const double ht = 0.5 * (txx + tyy), r = hypot(0.5 * (txx - tyy), txy), l1 = ht + r, l2 = ht - r;
        if (half_turn && l2 > 0 && l1 >= 1.1 * l2) {
            const double t = 0.5 * atan2(2 * txy, txx - tyy), f1 = pow(l1 / ht, 0.5 * c->uncrush), f2 = pow(l2 / ht, 0.5 * c->uncrush);
            w00 = f1 * cos(t); w01 = f1 * sin(t); w10 = -f2 * sin(t); w11 = f2 * cos(t); whiten = 1;
            min_x = min_y = INFINITY; max_x = max_y = -INFINITY;
            for (int i = 0; i < n; i++) {
                const double dx = p[i].x - mu_x, dy = p[i].y - mu_y, wx = w00 * dx + w01 * dy, wy = w10 * dx + w11 * dy;
                min_x = fmin(min_x, wx); max_x = fmax(max_x, wx); min_y = fmin(min_y, wy); max_y = fmax(max_y, wy);
            }
        }
    }
    const double det = w00 * w11 - w01 * w10;
    const int ns = c->nsamp > 0 ? c->nsamp : 10000;
    double *sx = malloc((size_t)ns * 4 * sizeof *sx), *sy = sx + ns, *snx = sy + ns, *sny = snx + ns;
    if (!sx) { free(cw); return -1; }
    uint64_t rs = seed * 0x2545F4914F6CDD1Dull + 1;
    for (int j = 0; j < ns; j++) {   /* draw by weight (villa: uniform over trace segments) */
        const double u = sm_unif(&rs) * tot;
        int a = 0, b = n - 1; while (a < b) { const int m = (a + b) / 2; if (cw[m] > u) b = m; else a = m + 1; }
        double x = p[a].x, y = p[a].y, nx = p[a].nx, ny = p[a].ny;
        if (whiten) {
            const double dx = x - mu_x, dy = y - mu_y; x = w00 * dx + w01 * dy; y = w10 * dx + w11 * dy;
            const double qx = (w11 * nx - w10 * ny) / det, qy = (-w01 * nx + w00 * ny) / det, ql = hypot(qx, qy);   /* normals go through W^-T */
            nx = ql > 0 ? qx / ql : 0; ny = ql > 0 ? qy / ql : 0;
        }
        sx[j] = x; sy[j] = y; snx[j] = nx; sny[j] = ny;
    }
    free(cw);
    /* RANSAC over candidates: sum cos^2 */
    double bx = 0, by = 0, best = -1;
    for (int i = 0; i < (c->ncand > 0 ? c->ncand : 1000); i++) {
        const double cx = min_x + sm_unif(&rs) * (max_x - min_x), cy = min_y + sm_unif(&rs) * (max_y - min_y);
        double s = 0;
        for (int j = 0; j < ns; j++) {
            const double dx = sx[j] - cx, dy = sy[j] - cy, d2 = dx * dx + dy * dy;
            if (d2 > 0) { const double ca = dx * snx[j] + dy * sny[j]; s += ca * ca / d2; }
        }
        if (s > best) { best = s; bx = cx; by = cy; }
    }
    /* hill climb on the weighted sum cos^2 / max(near, d) */
    const double nearv = c->near > 0 ? c->near : 1;
#define WSCORE(CX, CY, OUT) do { double s_ = 0; for (int j = 0; j < ns; j++) { const double dx = sx[j] - (CX), dy = sy[j] - (CY), d = sqrt(dx * dx + dy * dy); \
        if (d < 1e-6) continue; const double ca = (dx * snx[j] + dy * sny[j]) / d; s_ += ca * ca / fmax(nearv, d); } (OUT) = s_; } while (0)
    double rb; WSCORE(bx, by, rb);
    int moves = 0;
    for (double step = c->step0 > 0 ? c->step0 : 1024; step >= (c->step_min > 0 ? c->step_min : 1) && moves < c->max_moves;) {
        int moved = 0;
        for (int dx = -1; dx <= 1; dx++) for (int dy = -1; dy <= 1; dy++) {
            if (!dx && !dy) continue;
            const double cx = bx + dx * step, cy = by + dy * step; double s; WSCORE(cx, cy, s);
            if (s > rb) { rb = s; bx = cx; by = cy; moved = 1; }
        }
        if (moved) moves++; else step *= 0.5;
    }
#undef WSCORE
    free(sx);
    if (whiten) { const double ux = bx, uy = by; bx = (w11 * ux - w01 * uy) / det + mu_x; by = (-w10 * ux + w00 * uy) / det + mu_y; }
    out[0] = by; out[1] = bx;
    if (score) *score = rb;
    if (inside) *inside = bx >= lo_x && bx <= hi_x && by >= lo_y && by <= hi_y;
    return 0;
}

/* ---- CT slice -> points (2-D structure tensor) ---- */
static void blur(const float *src, float *dst, float *tmp, int H, int W, double sig) {   /* separable Gaussian, clamped edges */
    if (sig <= 0) { memcpy(dst, src, (size_t)H * W * sizeof *dst); return; }
    const int r = (int)ceil(3 * sig); float k[64]; double ks = 0;
    const int rr = r > 31 ? 31 : r;
    for (int i = -rr; i <= rr; i++) { k[i + rr] = (float)exp(-0.5 * i * i / (sig * sig)); ks += k[i + rr]; }
    for (int i = 0; i <= 2 * rr; i++) k[i] = (float)(k[i] / ks);
    for (int y = 0; y < H; y++) {
        const float *s = src + (size_t)y * W; float *t = tmp + (size_t)y * W;
        for (int x = 0; x < W; x++) { float a = 0; for (int i = -rr; i <= rr; i++) { int xx = x + i; xx = xx < 0 ? 0 : xx >= W ? W - 1 : xx; a += k[i + rr] * s[xx]; } t[x] = a; }
    }
    for (int y = 0; y < H; y++) {
        float *d = dst + (size_t)y * W;
        for (int x = 0; x < W; x++) d[x] = 0;
        for (int i = -rr; i <= rr; i++) {
            int yy = y + i; yy = yy < 0 ? 0 : yy >= H ? H - 1 : yy;
            const float *t = tmp + (size_t)yy * W, kk = k[i + rr];
            for (int x = 0; x < W; x++) d[x] += kk * t[x];
        }
    }
}

int umb_ct_points(const float *img, int H, int W, int s, double thr, double sig_d, double sig_t, int stride, double min_coh,
                  umb_pt **out, double cen[2], long long *ncen) {
    *out = nullptr; cen[0] = cen[1] = NAN; *ncen = 0;
    const size_t np = (size_t)H * W;
    float *buf = malloc(np * 5 * sizeof *buf);
    if (!buf) return -1;
    float *is = buf, *tmp = buf + np, *jxx = buf + 2 * np, *jxy = buf + 3 * np, *jyy = buf + 4 * np;
    const double off = (s - 1) / 2.0;
    double cy = 0, cx = 0; long long k = 0;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) if (img[(size_t)y * W + x] > thr) { cy += y; cx += x; k++; }
    *ncen = k;
    if (k) { cen[0] = cy / k * s + off; cen[1] = cx / k * s + off; }
    blur(img, is, tmp, H, W, sig_d);
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const size_t i = (size_t)y * W + x;
        const float gx = 0.5f * (is[(size_t)y * W + (x + 1 < W ? x + 1 : x)] - is[(size_t)y * W + (x > 0 ? x - 1 : x)]);
        const float gy = 0.5f * (is[(size_t)(y + 1 < H ? y + 1 : y) * W + x] - is[(size_t)(y > 0 ? y - 1 : y) * W + x]);
        jxx[i] = gx * gx; jxy[i] = gx * gy; jyy[i] = gy * gy;
    }
    blur(jxx, is, tmp, H, W, sig_t); memcpy(jxx, is, np * sizeof *is);
    blur(jxy, is, tmp, H, W, sig_t); memcpy(jxy, is, np * sizeof *is);
    blur(jyy, is, tmp, H, W, sig_t); memcpy(jyy, is, np * sizeof *is);
    if (stride < 1) stride = 1;
    const size_t cap = ((size_t)(H + stride - 1) / stride) * ((size_t)(W + stride - 1) / stride);
    umb_pt *p = malloc((cap ? cap : 1) * sizeof *p);
    if (!p) { free(buf); return -1; }
    int n = 0;
    for (int y = stride / 2; y < H; y += stride) for (int x = stride / 2; x < W; x += stride) {
        const size_t i = (size_t)y * W + x;
        if (!(img[i] > thr)) continue;
        const double a = jxx[i], b = jxy[i], c = jyy[i], tr = a + c;
        if (!(tr > 1e-12)) continue;
        const double coh = sqrt((a - c) * (a - c) + 4 * b * b) / tr;
        if (coh < min_coh) continue;
        const double t = 0.5 * atan2(2 * b, a - c);
        p[n++] = (umb_pt){.y = y * (double)s + off, .x = x * (double)s + off, .ny = sin(t), .nx = cos(t), .w = coh};
    }
    free(buf);
    *out = p;
    return n;
}

/* ---- along z (villa fit_umbilicus_along_z) ---- */
static double tricube(double u) { u = fmin(fabs(u), 1.0); const double t = 1 - u * u * u; return t * t * t; }
typedef struct { double z, y, x; } zyx;
static int cmp_zyx(const void *a, const void *b) { const double p = ((const zyx *)a)->z, q = ((const zyx *)b)->z; return (p > q) - (p < q); }
static int local_linear(const zyx *e, const double *w, int n, double z0, double hw, double out[2]) {
    int lo = 0, hi = n; while (lo < hi) { const int m = (lo + hi) / 2; if (e[m].z < z0 - hw) lo = m + 1; else hi = m; }
    double sw = 0, swz = 0, swy = 0, swx = 0; int cnt = 0, e1 = lo;
    for (; e1 < n && e[e1].z <= z0 + hw; e1++) {
        const double k = tricube((e[e1].z - z0) / hw) * w[e1];
        if (k > 0) cnt++;
        sw += k; swz += k * e[e1].z; swy += k * e[e1].y; swx += k * e[e1].x;
    }
    if (cnt < 3 || sw <= 0) return 0;
    const double zm = swz / sw, ym = swy / sw, xm = swx / sw;
    double szz = 0, szy = 0, szx = 0;
    for (int i = lo; i < e1; i++) {
        const double k = tricube((e[i].z - z0) / hw) * w[i], dz = e[i].z - zm;
        szz += k * dz * dz; szy += k * dz * (e[i].y - ym); szx += k * dz * (e[i].x - xm);
    }
    if (szz <= 1e-12 * fmax(1.0, sw)) return 0;
    out[0] = ym + szy / szz * (z0 - zm); out[1] = xm + szx / szz * (z0 - zm);
    return 1;
}

int umb_zfit(const double *z, const double *y, const double *x, int n, const double *qz, int nq, double hw, int iters,
             double *oy, double *ox, int *nfall) {
    if (nfall) *nfall = 0;
    for (int q = 0; q < nq; q++) oy[q] = ox[q] = NAN;
    if (n <= 0 || !(hw > 0)) return 0;
    zyx *e = malloc((size_t)n * sizeof *e); double *rw = malloc((size_t)n * 3 * sizeof *rw), *res = rw + n, *fd = res + n;
    if (!e || !rw) { free(e); free(rw); return -1; }
    for (int i = 0; i < n; i++) { e[i] = (zyx){z[i], y[i], x[i]}; rw[i] = 1; }
    qsort(e, (size_t)n, sizeof *e, cmp_zyx);   /* (villa stable_sort; ties among equal z only reorder equal-weight terms) */
    for (int it = 0; it < iters; it++) {
        int nf = 0;
        for (int i = 0; i < n; i++) {
            double f[2];
            if (local_linear(e, rw, n, e[i].z, hw, f)) { res[i] = hypot(e[i].y - f[0], e[i].x - f[1]); fd[nf++] = res[i]; } else res[i] = NAN;
        }
        const double s = nf ? median_of(fd, nf) : 0;
        if (!(s > 0)) break;
        for (int i = 0; i < n; i++) { if (isnan(res[i])) continue; const double u = fmin(res[i] / (6 * s), 1.0); rw[i] = (1 - u * u) * (1 - u * u); }
    }
    for (int q = 0; q < nq; q++) {
        double f[2];
        if (local_linear(e, rw, n, qz[q], hw, f)) { oy[q] = f[0]; ox[q] = f[1]; continue; }
        int m = 0; for (int i = 0; i < n; i++) if (e[i].z == qz[q]) { fd[m] = e[i].y; res[m++] = e[i].x; }
        if (m) { oy[q] = median_of(fd, m); ox[q] = median_of(res, m); if (nfall) (*nfall)++; }
    }
    free(e); free(rw);
    return 0;
}

/* ---- axis files: control_points [{z, y, x}], linear in z, clamped at the ends (sources.c axis_at) ---- */
typedef struct { int n; double *z, *y, *x; } umb_axis;
static char *slurp(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb"); if (!f) return nullptr;
    fseek(f, 0, SEEK_END); const long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *t = n >= 0 ? malloc((size_t)n + 1) : nullptr;
    if (!t || fread(t, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(t); return nullptr; }
    fclose(f); t[n] = 0; *len = (size_t)n; return t;
}
static int ax_load(umb_axis *a, const char *path) {
    memset(a, 0, sizeof *a);
    size_t len; char *t = slurp(path, &len); if (!t) return -1;
    json *j = json_parse(t, len); free(t);
    const json *cp = json_get(j, "control_points");
    if (!cp || !cp->n) { json_free(j); return -1; }
    zyx *e = malloc(cp->n * sizeof *e);
    for (size_t i = 0; i < cp->n; i++) { const json *p = json_at(cp, i); e[i] = (zyx){json_num(json_get(p, "z"), 0), json_num(json_get(p, "y"), 0), json_num(json_get(p, "x"), 0)}; }
    json_mark_read(j);
    qsort(e, cp->n, sizeof *e, cmp_zyx);
    a->n = (int)cp->n; a->z = malloc(3 * cp->n * sizeof(double)); a->y = a->z + cp->n; a->x = a->y + cp->n;
    for (int i = 0; i < a->n; i++) { a->z[i] = e[i].z; a->y[i] = e[i].y; a->x[i] = e[i].x; }
    free(e); json_free(j);
    return 0;
}
static void ax_at(const umb_axis *a, double z, double *y, double *x) {
    if (z <= a->z[0]) { *y = a->y[0]; *x = a->x[0]; return; }
    if (z >= a->z[a->n - 1]) { *y = a->y[a->n - 1]; *x = a->x[a->n - 1]; return; }
    int lo = 0, hi = a->n - 1; while (hi - lo > 1) { const int m = (lo + hi) / 2; if (a->z[m] <= z) lo = m; else hi = m; }
    const double t = a->z[hi] > a->z[lo] ? (z - a->z[lo]) / (a->z[hi] - a->z[lo]) : 0;
    *y = a->y[lo] + t * (a->y[hi] - a->y[lo]); *x = a->x[lo] + t * (a->x[hi] - a->x[lo]);
}
/* distance stats (mm) of the points (z, y, x)[n] against the reference axis, over the points inside the reference's z range */
static int dist_stats(const umb_axis *ref, const double *z, const double *y, const double *x, int n, double um0, double st[4]) {
    double *d = malloc((size_t)(n > 0 ? n : 1) * sizeof *d); int m = 0;
    for (int i = 0; i < n; i++) {
        if (isnan(y[i]) || isnan(x[i]) || z[i] < ref->z[0] || z[i] > ref->z[ref->n - 1]) continue;
        double ry, rx; ax_at(ref, z[i], &ry, &rx); d[m++] = hypot(y[i] - ry, x[i] - rx) * um0 / 1000.0;
    }
    st[0] = m;
    if (!m) { st[1] = st[2] = st[3] = NAN; free(d); return 0; }
    qsort(d, (size_t)m, sizeof *d, cmp_dbl);
    st[1] = m % 2 ? d[m / 2] : 0.5 * (d[m / 2 - 1] + d[m / 2]);
    st[2] = d[(int)fmin(m - 1, ceil(0.9 * m) - 1)];
    st[3] = d[m - 1];
    free(d);
    return m;
}

/* ufsm axis-diff A.json B.json --um0 U [--dz 64] [--z0 Z --z1 Z]: per-z distance of axis A from axis B (both level-0 voxels), sampled every
   dz level-0 voxels over the common z range of their control points (no extrapolation): median / p90 / max in mm */
int cmd_axis_diff(int argc, char **argv) {
    if (argc < 4 || !opt(argc, argv, "--um0", nullptr)) {
        fprintf(stderr, "usage: ufsm axis-diff A.json B.json --um0 UM [--dz 64] [--z0 Z --z1 Z]   per-z distance A vs B in mm (median, p90, max)\n");
        return 2;
    }
    umb_axis a, b;
    if (ax_load(&a, argv[2])) { fprintf(stderr, "axis-diff: cannot load %s\n", argv[2]); return 1; }
    if (ax_load(&b, argv[3])) { fprintf(stderr, "axis-diff: cannot load %s\n", argv[3]); return 1; }
    const double um0 = opt_atof("--um0", opt(argc, argv, "--um0", "0")), dz = opt_atof("--dz", opt(argc, argv, "--dz", "64"));
    double z0 = fmax(a.z[0], b.z[0]), z1 = fmin(a.z[a.n - 1], b.z[b.n - 1]);
    if (opt(argc, argv, "--z0", nullptr)) z0 = fmax(z0, opt_atof("--z0", opt(argc, argv, "--z0", "0")));
    if (opt(argc, argv, "--z1", nullptr)) z1 = fmin(z1, opt_atof("--z1", opt(argc, argv, "--z1", "0")));
    if (!(um0 > 0) || !(dz > 0) || !(z1 >= z0)) { fprintf(stderr, "axis-diff: no common z range (%.0f..%.0f) or bad --um0 / --dz\n", z0, z1); return 1; }
    const int n = (int)floor((z1 - z0) / dz) + 1;
    double *z = malloc((size_t)n * 3 * sizeof *z), *y = z + n, *x = y + n;
    for (int i = 0; i < n; i++) { z[i] = z0 + i * dz; ax_at(&a, z[i], &y[i], &x[i]); }
    double st[4]; dist_stats(&b, z, y, x, n, um0, st);
    printf("axis-diff %s vs %s: z %.0f..%.0f, %d samples: median %.3f mm, p90 %.3f mm, max %.3f mm\n", argv[2], argv[3], z0, z1, (int)st[0], st[1], st[2], st[3]);
    free(z); free(a.z); free(b.z);
    return 0;
}

/* ---- ufsm umbilicus ---- */
typedef struct {
    double z;                   /* level-0 */
    int r0, r1;                 /* input rows (level-L CT rows or field rows) [r0, r1) */
    umb_pt *p; int n;
    double cen[2]; long long ncen;
    double est[2], score; int ok, inside;
} umb_slice;

typedef struct {   /* one input row span shared by the slices of a chunk row */
    int src;                    /* 0 CT, 1 fields */
    const uint8_t *buf[4]; int r0, H, W;   /* CT: buf[0]; fields: t1 t2 t5 w */
    int s; double thr, sig_d, sig_t, min_coh; int stride;   /* CT */
    double org[3], h, f; int pool;   /* fields: origin (native), cell edge (native), native -> level-0 factor, pooling */
    umb_slice *sl; const umb_cfg *cfg; uint64_t seed;
} umb_job;

static int slice_points(const umb_job *J, umb_slice *S) {
    const int nr = S->r1 - S->r0; const size_t pl = (size_t)J->H * J->W;
    if (J->src == 0) {
        float *img = calloc(pl, sizeof *img);
        if (!img) return -1;
        for (int r = S->r0; r < S->r1; r++) { const uint8_t *b = J->buf[0] + (size_t)(r - J->r0) * pl; for (size_t i = 0; i < pl; i++) img[i] += b[i]; }
        for (size_t i = 0; i < pl; i++) img[i] /= (float)nr;
        const int n = umb_ct_points(img, J->H, J->W, J->s, J->thr, J->sig_d, J->sig_t, J->stride, J->min_coh, &S->p, S->cen, &S->ncen);
        free(img);
        if (n < 0) return -1;
        S->n = n;
        return 0;
    }
    /* fields: per pool x pool cells (all rows of the slice) the in-plane 2 x 2 block of the mean n n^T, weighted by w */
    const int P = J->pool, Hp = (J->H + P - 1) / P, Wp = (J->W + P - 1) / P;
    S->p = malloc((size_t)Hp * Wp * sizeof *S->p);
    if (!S->p) return -1;
    double cy = 0, cx = 0; long long kc = 0; int n = 0;
    for (int by = 0; by < Hp; by++) for (int bx = 0; bx < Wp; bx++) {
        double A = 0, B = 0, C = 0, Wt = 0; int cells = 0;
        for (int r = S->r0; r < S->r1; r++) {
            const size_t ro = (size_t)(r - J->r0) * pl;
            for (int yy = by * P; yy < by * P + P && yy < J->H; yy++) for (int xx = bx * P; xx < bx * P + P && xx < J->W; xx++) {
                const size_t i = ro + (size_t)yy * J->W + xx; const double w = J->buf[3][i] / 255.0;
                cells++;
                if (w <= 0) continue;
                A += w * J->buf[0][i] / 255.0; C += w * J->buf[1][i] / 255.0; B += w * (J->buf[2][i] / 255.0 - 0.5); Wt += w;
            }
        }
        if (!(Wt > 0)) continue;
        const double y0 = (J->org[1] + J->h * (by * P + 0.5 * P)) * J->f, x0 = (J->org[2] + J->h * (bx * P + 0.5 * P)) * J->f;
        cy += y0; cx += x0; kc++;
        const double ht = 0.5 * (A + C), rr = hypot(0.5 * (C - A), B), l1 = ht + rr, l2 = ht - rr;   /* (x, y): Jxx = C, Jyy = A, Jxy = B */
        if (!(l1 - l2 > 0) || (l1 - l2) / (l1 + l2) < J->min_coh) continue;
        const double t = 0.5 * atan2(2 * B, C - A);
        S->p[n++] = (umb_pt){.y = y0, .x = x0, .ny = sin(t), .nx = cos(t), .w = (l1 - l2) / cells};
    }
    S->n = n; S->ncen = kc;
    if (kc) { S->cen[0] = cy / kc; S->cen[1] = cx / kc; }
    return 0;
}

typedef struct { umb_job *J; umb_slice *s; int n; atomic_int next, fail; int phase; } umb_pool;
static void *umb_worker(void *arg) {
    umb_pool *P = arg;
    for (int i; (i = atomic_fetch_add(&P->next, 1)) < P->n;) {
        umb_slice *S = &P->s[i];
        if (P->phase == 0) { if (slice_points(P->J, S)) atomic_fetch_add(&P->fail, 1); continue; }
        S->ok = umb_estimate(S->p, S->n, P->J->cfg, P->J->seed + (uint64_t)llround(S->z), S->est, &S->score, &S->inside) == 0;
    }
    return nullptr;
}
static int umb_run(umb_job *J, umb_slice *s, int n, int nt, int phase) {
    umb_pool P = {.J = J, .s = s, .n = n, .phase = phase}; atomic_init(&P.next, 0); atomic_init(&P.fail, 0);
    if (nt < 1) nt = 1;
    pthread_t th[64]; int nth = nt > 64 ? 64 : nt;
    for (int t = 1; t < nth; t++) pthread_create(&th[t], nullptr, umb_worker, &P);
    umb_worker(&P);
    for (int t = 1; t < nth; t++) pthread_join(th[t], nullptr);
    return atomic_load(&P.fail);
}

static void usage_umb(void) {
    fprintf(stderr,
        "usage: ufsm umbilicus OUT.json --um0 UM (--ct-root ROOT --ct-key KEY --level L | --fields DIR) [options]\n"
        "  the umbilicus by sheet-normal voting (villa vc_gen_umbilicus port): per xy slice the point the in-plane sheet normals point away\n"
        "  from, then a robust LOWESS along z. Output: umbilicus json (control_points z, y, x in level-0 voxels of a UM um scan).\n"
        "  --um0 UM          level-0 voxel size of the scan the axis refers to (um; required: sets the mm parameters and the fields' scale)\n"
        "  --ct-root/--ct-key/--level   CT input: the 2-D structure tensor of each slice at level L (--sigma 1 --tsigma 3 px, --stride 4,\n"
        "                    --thr 0: CT > thr is material, --min-coh 0.2)\n"
        "  --fields DIR      fit-field input (zarr planes t1 t2 t5 w; fields.json um / origin / h): the in-plane n n^T pooled --pool 4 cells\n"
        "  --dz-mm 1         slice spacing; --rows 4: input rows averaged per slice; --z0 / --z1 level-0 range\n"
        "  --zfit-mm 4.7     LOWESS half width (villa's measured 4.7 mm; 0: each slice's own estimate)\n"
        "  --uncrush 1.5     villa un-crush beta (their measured setting; vs 0: lower median on 1667 / P4 / Paris3 and 1203's clear core, DESIGN 11-27); --samples 10000 --cands 1000 --near-mm 0.94 --seed 1\n"
        "  --min-pts 200     slices with fewer points are skipped (reported); --threads 6; --csv FILE per-slice table\n"
        "  --ref AXIS.json   also report the per-z distance (mm) of the fitted, raw and CT-centroid axes from AXIS\n");
}

int cmd_umbilicus(int argc, char **argv) {
    const char *out = argc > 2 && argv[2][0] != '-' ? argv[2] : nullptr;
    const char *ctroot = opt(argc, argv, "--ct-root", nullptr), *ctkey = opt(argc, argv, "--ct-key", nullptr), *fdir = opt(argc, argv, "--fields", nullptr);
    const double um0 = opt_atof("--um0", opt(argc, argv, "--um0", "0"));
    if (!out || !(um0 > 0) || (!fdir && !(ctroot && ctkey)) || (fdir && ctroot)) { usage_umb(); return 2; }
    const double v = 1000.0 / um0;   /* level-0 voxels per mm */
    umb_cfg cfg; umb_cfg_default(&cfg, um0);
    cfg.nsamp = opt_atoi("--samples", opt(argc, argv, "--samples", "10000"));
    cfg.ncand = opt_atoi("--cands", opt(argc, argv, "--cands", "1000"));
    cfg.near = opt_atof("--near-mm", opt(argc, argv, "--near-mm", "0.94")) * v;
    cfg.uncrush = opt_atof("--uncrush", opt(argc, argv, "--uncrush", "1.5"));
    if (!(cfg.uncrush >= 0 && cfg.uncrush <= 4)) { fprintf(stderr, "umbilicus: --uncrush must be in [0, 4]\n"); return 2; }
    const double dz = opt_atof("--dz-mm", opt(argc, argv, "--dz-mm", "1")) * v, zfit = opt_atof("--zfit-mm", opt(argc, argv, "--zfit-mm", "4.7")) * v;
    const int rows = opt_atoi("--rows", opt(argc, argv, "--rows", "4")), nt = opt_atoi("--threads", opt(argc, argv, "--threads", "6"));
    const int min_pts = opt_atoi("--min-pts", opt(argc, argv, "--min-pts", "200"));
    const uint64_t seed = (uint64_t)opt_atoll("--seed", opt(argc, argv, "--seed", "1"));
    if (!(dz > 0) || rows < 1 || !(zfit >= 0)) { fprintf(stderr, "umbilicus: bad --dz-mm / --rows / --zfit-mm\n"); return 2; }
    umb_job J = {.cfg = &cfg, .seed = seed};
    J.thr = opt_atof("--thr", opt(argc, argv, "--thr", "0"));
    J.min_coh = opt_atof("--min-coh", opt(argc, argv, "--min-coh", "0.2"));
    J.sig_d = opt_atof("--sigma", opt(argc, argv, "--sigma", "1"));
    J.sig_t = opt_atof("--tsigma", opt(argc, argv, "--tsigma", "3"));
    J.stride = opt_atoi("--stride", opt(argc, argv, "--stride", "4"));
    J.pool = opt_atoi("--pool", opt(argc, argv, "--pool", "4"));
    if (J.pool < 1) J.pool = 1;
    store *st = nullptr; z3 *za[4] = {nullptr};
    int64_t Z, H, W, chunk; double zscale;   /* rows -> level-0 z: row r is centred at zoff0 + (r + 0.5) * zscale */
    double zoff0 = 0; long long fzoff[3] = {0, 0, 0};
    char src_desc[1400];
    if (ctroot) {
        const int L = opt_atoi("--level", opt(argc, argv, "--level", "-1"));
        if (L < 0 || L > 9) { fprintf(stderr, "umbilicus: --level L required with --ct-root\n"); return 2; }
        char k[1200]; snprintf(k, sizeof k, "%s/%d", ctkey, L);
        if (!(st = store_open(ctroot)) || !(za[0] = z3_open(st, k, opt(argc, argv, "--cache", nullptr)))) { fprintf(stderr, "umbilicus: cannot open %s %s: %s\n", ctroot, k, z3_error()); return 1; }
        const z3_meta *m = z3_meta_of(za[0]);
        Z = m->shape[0]; H = m->shape[1]; W = m->shape[2]; chunk = m->chunk[0] > 0 ? m->chunk[0] : 128;
        J.src = 0; J.s = 1 << L; zscale = J.s; zoff0 = -0.5;
        snprintf(src_desc, sizeof src_desc, "CT %s %s level %d (2-D structure tensor, sigma %.2f / %.2f px, stride %d, min coherence %.2f)", ctroot, ctkey, L, J.sig_d, J.sig_t, J.stride, J.min_coh);
    } else {
        char p[1400]; snprintf(p, sizeof p, "%s/fields.json", fdir);
        size_t len; char *t = slurp(p, &len); json *j = t ? json_parse(t, len) : nullptr; free(t);
        if (!j) { fprintf(stderr, "umbilicus: cannot read %s\n", p); return 1; }
        if (strcmp(json_str(json_get(j, "format"), "raw"), "zarr")) { fprintf(stderr, "umbilicus: %s: only zarr fit fields are read\n", p); json_free(j); return 1; }
        const double fum = json_num(json_get(j, "um"), 0);
        if (!(fum > 0)) { fprintf(stderr, "umbilicus: %s has no \"um\"\n", p); json_free(j); return 1; }
        J.h = json_num(json_get(j, "h"), 0); J.f = fum / um0;
        for (int d = 0; d < 3; d++) { J.org[d] = json_num(json_at(json_get(j, "origin"), d), 0); fzoff[d] = (long long)json_num(json_at(json_get(j, "zarr_offset"), d), 0); }
        Z = (int64_t)json_num(json_at(json_get(j, "shape"), 0), 0); H = (int64_t)json_num(json_at(json_get(j, "shape"), 1), 0); W = (int64_t)json_num(json_at(json_get(j, "shape"), 2), 0);
        json_mark_read(j); json_free(j);
        if (!(J.h > 0) || Z <= 0) { fprintf(stderr, "umbilicus: %s: bad h / shape\n", p); return 1; }
        const char *nm[4] = {"t1", "t2", "t5", "w"};
        if (!(st = store_open(fdir))) { fprintf(stderr, "umbilicus: cannot open %s\n", fdir); return 1; }
        for (int k = 0; k < 4; k++) if (!(za[k] = z3_open(st, nm[k], nullptr))) { fprintf(stderr, "umbilicus: %s/%s: %s\n", fdir, nm[k], z3_error()); return 1; }
        chunk = z3_meta_of(za[0])->chunk[0] > 0 ? z3_meta_of(za[0])->chunk[0] : 128;
        J.src = 1; zscale = J.h * J.f; zoff0 = J.org[0] * J.f;
        snprintf(src_desc, sizeof src_desc, "fit fields %s (%.3f um, h %g; in-plane n n^T pooled %d cells, min coherence %.2f)", fdir, fum, J.h, J.pool, J.min_coh);
    }
    J.H = (int)H; J.W = (int)W;
    /* slices: level-0 z every dz from z0 + dz / 2, each the mean of `rows` input rows around it */
    const double zlo = zoff0, zhi = zoff0 + Z * zscale;
    double z0 = fmax(zlo, opt_atof("--z0", opt(argc, argv, "--z0", "-1e18"))), z1 = fmin(zhi, opt_atof("--z1", opt(argc, argv, "--z1", "1e18")));
    const int ns = z1 > z0 ? (int)floor((z1 - z0) / dz) : 0;
    if (ns < 1) { fprintf(stderr, "umbilicus: no slices in z %.0f..%.0f (dz %.1f voxels)\n", z0, z1, dz); return 1; }
    umb_slice *sl = calloc((size_t)ns, sizeof *sl);
    for (int i = 0; i < ns; i++) {
        const double zc = z0 + (i + 0.5) * dz;
        int rc = (int)floor((zc - zoff0) / zscale);   /* the row holding zc */
        int r0 = rc - (rows - 1) / 2; if (r0 < 0) r0 = 0; if (r0 + rows > Z) r0 = (int)Z - rows; if (r0 < 0) r0 = 0;
        sl[i].r0 = r0; sl[i].r1 = r0 + rows > Z ? (int)Z : r0 + rows;
        sl[i].z = zoff0 + 0.5 * (sl[i].r0 + sl[i].r1) * zscale;   /* the mean of the rows' centres zoff0 + (r + 0.5) zscale */
        sl[i].cen[0] = sl[i].cen[1] = sl[i].est[0] = sl[i].est[1] = NAN;
    }
    fprintf(stderr, "umbilicus: %s\n  %d slices z %.0f..%.0f every %.1f level-0 voxels (%.2f mm), %d rows each, %d threads\n", src_desc, ns, sl[0].z, sl[ns - 1].z, dz, dz / v, rows, nt);
    /* read chunk row by chunk row: the slices whose first row lies in it, one span read */
    const double t0 = now_s(); int nfail = 0;
    for (int a = 0; a < ns;) {
        const int64_t cr = sl[a].r0 / chunk; int b = a; int lo = sl[a].r0, hi = sl[a].r1;
        while (b < ns && sl[b].r0 / chunk == cr) { if (sl[b].r1 > hi) hi = sl[b].r1; b++; }
        const size_t pl = (size_t)H * W, nr = (size_t)(hi - lo);
        uint8_t *buf = malloc(nr * pl * (J.src ? 4 : 1));
        if (!buf) { fprintf(stderr, "umbilicus: no memory for %zu rows of %lld x %lld\n", nr, (long long)H, (long long)W); return 1; }
        for (int k = 0; k < (J.src ? 4 : 1); k++) {
            const int64_t o[3] = {lo + fzoff[0], fzoff[1], fzoff[2]}, cnt[3] = {(int64_t)nr, H, W};
            if (z3_read(za[k], o, cnt, buf + k * nr * pl, nt)) { fprintf(stderr, "umbilicus: read rows %d..%d: %s\n", lo, hi, z3_error()); return 1; }
            J.buf[k] = buf + k * nr * pl;
        }
        J.r0 = lo;
        nfail += umb_run(&J, sl + a, b - a, nt, 0);
        free(buf);
        int np = 0; for (int i = a; i < b; i++) np += sl[i].n;
        fprintf(stderr, "  rows %d..%d: %d slices, %d points (%.0f s)\n", lo, hi, b - a, np, now_s() - t0);
        a = b;
    }
    for (int k = 0; k < 4; k++) z3_close(za[k]);
    if (nfail) { fprintf(stderr, "umbilicus: %d slices failed to extract (no memory)\n", nfail); return 1; }
    int nfew = 0; for (int i = 0; i < ns; i++) if (sl[i].n < min_pts) { nfew++; sl[i].n = 0; }
    umb_run(&J, sl, ns, nt, 1);
    int nok = 0, nout = 0;
    double *ez = malloc((size_t)ns * 6 * sizeof *ez), *ey = ez + ns, *ex = ey + ns, *qz = ex + ns, *fy = qz + ns, *fx = fy + ns;
    for (int i = 0; i < ns; i++) {
        qz[i] = sl[i].z;
        if (!sl[i].ok) continue;
        if (!sl[i].inside) { nout++; continue; }   /* a climb that left the section: not used by the fit (counted) */
        ez[nok] = sl[i].z; ey[nok] = sl[i].est[0]; ex[nok] = sl[i].est[1]; nok++;
    }
    fprintf(stderr, "umbilicus: %d of %d slices estimated (%.0f s); %d with < %d points skipped, %d estimates outside their section dropped\n", nok, ns, now_s() - t0, nfew, min_pts, nout);
    if (!nok) { fprintf(stderr, "umbilicus: no slice gave an estimate\n"); return 1; }
    int nfall = 0;
    if (zfit > 0) {
        umb_zfit(ez, ey, ex, nok, qz, ns, zfit, 3, fy, fx, &nfall);
        if (nfall) UFSM_DEFAULTED("umbilicus: %d slices kept their own estimate (fewer than 3 weighted estimates within +-%.1f mm)", nfall, zfit / v);
    } else for (int i = 0; i < ns; i++) { fy[i] = sl[i].ok && sl[i].inside ? sl[i].est[0] : NAN; fx[i] = sl[i].ok && sl[i].inside ? sl[i].est[1] : NAN; }
    FILE *f = fopen(out, "w");
    if (!f) { fprintf(stderr, "umbilicus: cannot write %s\n", out); return 1; }
    fprintf(f, "{\"control_points\":[");
    int nw = 0, nnan = 0;
    for (int i = 0; i < ns; i++) {
        if (isnan(fy[i]) || isnan(fx[i])) { nnan++; continue; }
        json_fprintf(f, "%s{\"z\":%.1f,\"y\":%.2f,\"x\":%.2f,\"raw_y\":%.2f,\"raw_x\":%.2f,\"cen_y\":%.2f,\"cen_x\":%.2f,\"points\":%d,\"inside\":%d}", nw++ ? "," : "", sl[i].z, fy[i], fx[i],
                     sl[i].ok ? sl[i].est[0] : NAN, sl[i].ok ? sl[i].est[1] : NAN, sl[i].cen[0], sl[i].cen[1], sl[i].n, sl[i].inside);
    }
    json_fprintf(f, "],\"method\":\"ufsm umbilicus: sheet-normal voting per slice (villa vc_gen_umbilicus port) + LOWESS along z\",\"source\":\"%s\","
                 "\"um0\":%.4f,\"dz_mm\":%.3f,\"rows\":%d,\"zfit_mm\":%.3f,\"uncrush\":%.3f,\"samples\":%d,\"cands\":%d,\"near_mm\":%.3f,\"seed\":%llu}\n",
                 src_desc, um0, dz / v, rows, zfit / v, cfg.uncrush, cfg.nsamp, cfg.ncand, cfg.near / v, (unsigned long long)seed);
    if (fclose(f)) { fprintf(stderr, "umbilicus: write error %s\n", out); return 1; }
    if (nnan) fprintf(stderr, "umbilicus: %d slices without a fitted point (no estimate within the window) left out\n", nnan);
    fprintf(stderr, "umbilicus: %d control points -> %s\n", nw, out);
    const char *csv = opt(argc, argv, "--csv", nullptr);
    if (csv) {
        FILE *c = fopen(csv, "w");
        if (!c) { fprintf(stderr, "umbilicus: cannot write %s\n", csv); return 1; }
        fprintf(c, "z,fit_y,fit_x,raw_y,raw_x,cen_y,cen_x,points,inside,score\n");
        for (int i = 0; i < ns; i++) fprintf(c, "%.1f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%.6g\n", sl[i].z, fy[i], fx[i], sl[i].est[0], sl[i].est[1], sl[i].cen[0], sl[i].cen[1], sl[i].n, sl[i].inside, sl[i].score);
        fclose(c);
    }
    const char *ref = opt(argc, argv, "--ref", nullptr);
    if (ref) {
        umb_axis ra;
        if (ax_load(&ra, ref)) { fprintf(stderr, "umbilicus: cannot load --ref %s\n", ref); return 1; }
        double *ry = malloc((size_t)ns * 4 * sizeof *ry), *rx = ry + ns, *cy = rx + ns, *cx = cy + ns;
        for (int i = 0; i < ns; i++) {
            const int g = sl[i].ok && sl[i].inside; ry[i] = g ? sl[i].est[0] : NAN; rx[i] = g ? sl[i].est[1] : NAN;
            const int gc = sl[i].ncen > 0; cy[i] = gc ? sl[i].cen[0] : NAN; cx[i] = gc ? sl[i].cen[1] : NAN;
        }
        double s1[4], s2[4], s3[4];
        dist_stats(&ra, qz, fy, fx, ns, um0, s1); dist_stats(&ra, qz, ry, rx, ns, um0, s2); dist_stats(&ra, qz, cy, cx, ns, um0, s3);
        printf("vs %s (z %.0f..%.0f)   n   median mm   p90 mm   max mm\n", ref, ra.z[0], ra.z[ra.n - 1]);
        printf("  vote + zfit        %4d   %8.3f   %7.3f   %7.3f\n", (int)s1[0], s1[1], s1[2], s1[3]);
        printf("  vote per slice     %4d   %8.3f   %7.3f   %7.3f\n", (int)s2[0], s2[1], s2[2], s2[3]);
        printf("  centroid           %4d   %8.3f   %7.3f   %7.3f\n", (int)s3[0], s3[1], s3[2], s3[3]);
        free(ry); free(ra.z);
    }
    for (int i = 0; i < ns; i++) free(sl[i].p);
    free(sl); free(ez);
    store_close(st);
    return 0;
}
