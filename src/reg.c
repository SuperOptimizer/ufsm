/* Scan-to-scan registration core: see reg.h. */
#include "reg.h"
#include "store.h"
#include "zarr3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

const char *const reg_status_name[7] = {"ok", "mask", "fine", "flat", "edge", "read", "large"};

/* ---------- affine ---------- */
raff raff_ident(void) { raff a = {0}; for (int i = 0; i < 3; i++) a.m[i][i] = 1; return a; }
double raff_det(const raff *a) {
    const double (*m)[4] = a->m;
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}
raff raff_inv(raff a) {
    const double (*m)[4] = a.m; raff r = {0};
    const double d = raff_det(&a);
    r.m[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / d; r.m[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / d; r.m[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / d;
    r.m[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / d; r.m[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / d; r.m[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / d;
    r.m[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / d; r.m[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / d; r.m[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / d;
    for (int i = 0; i < 3; i++) r.m[i][3] = -(r.m[i][0] * m[0][3] + r.m[i][1] * m[1][3] + r.m[i][2] * m[2][3]);
    return r;
}
raff raff_mul(raff a, raff b) {
    raff r = {0};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            double s = 0; for (int k = 0; k < 3; k++) s += a.m[i][k] * b.m[k][j];
            r.m[i][j] = s + (j == 3 ? a.m[i][3] : 0);
        }
    }
    return r;
}

/* ---------- field / map ---------- */
void rfield_eval(const rfield *f, const double p[3], double out[3]) {
    out[0] = out[1] = out[2] = 0;
    if (!f || !f->d) return;
    double u[3]; int i0[3];
    for (int a = 0; a < 3; a++) {   /* a: x y z -> grid axis 2 - a */
        const int na = f->n[2 - a];
        double g = (p[a] - f->o[a]) / f->h;
        if (g < 0) g = 0;
        if (g > na - 1) g = na - 1;
        int i = (int)floor(g); if (i > na - 2) i = na - 2; if (i < 0) i = 0;
        i0[a] = i; u[a] = na > 1 ? g - i : 0;
    }
    for (int dz = 0; dz < 2; dz++) for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
        const int z = i0[2] + dz < f->n[0] ? i0[2] + dz : f->n[0] - 1, y = i0[1] + dy < f->n[1] ? i0[1] + dy : f->n[1] - 1, x = i0[0] + dx < f->n[2] ? i0[0] + dx : f->n[2] - 1;
        const double w = (dz ? u[2] : 1 - u[2]) * (dy ? u[1] : 1 - u[1]) * (dx ? u[0] : 1 - u[0]);
        if (w == 0) continue;
        const float *d = f->d + (((size_t)z * f->n[1] + y) * f->n[2] + x) * 3;
        out[0] += w * d[0]; out[1] += w * d[1]; out[2] += w * d[2];
    }
}
void rmap_c2f(const rmap *m, const double c[3], double f[3]) {
    double d[3], q[3]; rfield_eval(m->D, c, d);
    for (int i = 0; i < 3; i++) q[i] = c[i] + d[i];
    raff_apply(&m->Ainv, q, f);
}
void rmap_f2c(const rmap *m, const double f[3], double c[3]) {
    double a[3]; raff_apply(&m->A, f, a);   /* c + D(c) = A f */
    for (int i = 0; i < 3; i++) c[i] = a[i];
    for (int it = 0; it < 20 && m->D; it++) {
        double d[3]; rfield_eval(m->D, c, d); double e = 0;
        for (int i = 0; i < 3; i++) { const double n = a[i] - d[i]; e += fabs(n - c[i]); c[i] = n; }
        if (e < 1e-6) break;
    }
}

/* ---------- volumes ---------- */
static double um_of_key(const char *key) {
    const char *p = strstr(key, "um-"); if (!p) p = strstr(key, "um.");
    if (!p) return 0;
    const char *b = p; while (b > key && (b[-1] == '.' || (b[-1] >= '0' && b[-1] <= '9'))) b--;
    return atof(b);
}
rvol *rvol_open(const char *root, const char *key, const char *cache, double um) {
    store *s = store_open(root); if (!s) return nullptr;
    z3_level lv[10]; const int n = z3_group_levels(s, key, lv, 10);
    if (n <= 0) { fprintf(stderr, "reg: no levels in %s/%s: %s\n", root, key, z3_error()); return nullptr; }
    rvol *v = calloc(1, sizeof *v);
    v->um = um > 0 ? um : um_of_key(key);
    snprintf(v->name, sizeof v->name, "%s", key);
    for (int l = 0; l < n; l++) {
        const double f = lv[l].um / lv[0].um;
        if (fabs(f - (double)(1 << l)) > 1e-3) break;   /* only 2^l pyramids */
        char k[1200]; snprintf(k, sizeof k, "%s/%s", key, lv[l].path);
        z3 *z = z3_open(s, k, cache);
        if (!z) { fprintf(stderr, "reg: %s: %s\n", k, z3_error()); break; }
        v->z[l] = z; const z3_meta *m = z3_meta_of(z);
        for (int a = 0; a < 3; a++) v->shape[l][a] = m->shape[a];
        v->nl = l + 1;
    }
    if (!v->nl || v->um <= 0) { fprintf(stderr, "reg: %s: no usable level or voxel size\n", key); free(v); return nullptr; }
    return v;
}
rvol *rvol_mem(double um, int nl, const int64_t shape0[3], uint8_t *level0) {
    rvol *v = calloc(1, sizeof *v); v->um = um; v->nl = nl; snprintf(v->name, sizeof v->name, "mem");
    v->mem[0] = level0; memcpy(v->shape[0], shape0, sizeof v->shape[0]);
    for (int l = 1; l < nl; l++) {
        const int64_t *p = v->shape[l - 1]; int64_t *s = v->shape[l];
        for (int a = 0; a < 3; a++) s[a] = (p[a] + 1) / 2;
        uint8_t *o = malloc((size_t)s[0] * s[1] * s[2]); const uint8_t *in = v->mem[l - 1];
        for (int64_t z = 0; z < s[0]; z++) for (int64_t y = 0; y < s[1]; y++) for (int64_t x = 0; x < s[2]; x++) {
            int sum = 0, cnt = 0;
            for (int dz = 0; dz < 2; dz++) for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
                const int64_t zz = 2 * z + dz, yy = 2 * y + dy, xx = 2 * x + dx;
                if (zz < p[0] && yy < p[1] && xx < p[2]) { sum += in[((size_t)zz * p[1] + yy) * p[2] + xx]; cnt++; }
            }
            o[((size_t)z * s[1] + y) * s[2] + x] = (uint8_t)((sum + cnt / 2) / cnt);
        }
        v->mem[l] = o;
    }
    return v;
}
int rvol_read(rvol *v, int l, const int64_t o[3], const int64_t n[3], uint8_t *out) { return rvol_read_nt(v, l, o, n, out, 1); }
int rvol_read_nt(rvol *v, int l, const int64_t o[3], const int64_t n[3], uint8_t *out, int nt) {
    if (l < 0 || l >= v->nl) return -1;
    if (v->z[l]) {
        for (int a = 0; a < 5; a++) {   /* the mirror has hiccups: retry after 2, 8, 30, 60 s */
            if (!z3_read(v->z[l], o, n, out, nt)) return 0;
            fprintf(stderr, "reg: read %s level %d (attempt %d): %s\n", v->name, l, a + 1, z3_error());
            static const int wait[4] = {2, 8, 30, 60};
            if (a < 4) sleep(wait[a]);
        }
        return -1;
    }
    const int64_t *s = v->shape[l]; const uint8_t *m = v->mem[l];
    for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) {
        uint8_t *row = out + ((size_t)z * n[1] + y) * n[2];
        const int64_t zz = o[0] + z, yy = o[1] + y;
        if (zz < 0 || zz >= s[0] || yy < 0 || yy >= s[1]) { memset(row, 0, n[2]); continue; }
        for (int64_t x = 0; x < n[2]; x++) { const int64_t xx = o[2] + x; row[x] = xx >= 0 && xx < s[2] ? m[((size_t)zz * s[1] + yy) * s[2] + xx] : 0; }
    }
    return 0;
}
void rvol_close(rvol *v) {
    if (!v) return;
    for (int l = 0; l < v->nl; l++) { if (v->z[l]) z3_close(v->z[l]); free(v->mem[l]); }
    free(v);
}
int rvol_level_for(const rvol *v, double um) {
    int l = 0; while (l + 1 < v->nl && v->um * (1 << (l + 1)) <= um * 1.05) l++;
    return l;
}

/* ---------- sampling ---------- */
float rblk_sample(const rblk *b, double x, double y, double z) {
    x -= b->o[2]; y -= b->o[1]; z -= b->o[0];
    const double fx = floor(x), fy = floor(y), fz = floor(z);
    const int64_t ix = (int64_t)fx, iy = (int64_t)fy, iz = (int64_t)fz;
    const double ux = x - fx, uy = y - fy, uz = z - fz;
    if (ix >= 0 && iy >= 0 && iz >= 0 && ix + 1 < b->n[2] && iy + 1 < b->n[1] && iz + 1 < b->n[0]) {
        const size_t sy = b->n[2], sz = (size_t)b->n[1] * b->n[2];
        const uint8_t *p = b->v + (size_t)iz * sz + (size_t)iy * sy + ix;
        const double c00 = p[0] + ux * (p[1] - p[0]), c01 = p[sy] + ux * (p[sy + 1] - p[sy]);
        const double c10 = p[sz] + ux * (p[sz + 1] - p[sz]), c11 = p[sz + sy] + ux * (p[sz + sy + 1] - p[sz + sy]);
        const double c0 = c00 + uy * (c01 - c00), c1 = c10 + uy * (c11 - c10);
        return (float)(c0 + uz * (c1 - c0));
    }
    double s = 0;
    for (int dz = 0; dz < 2; dz++) for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
        const int64_t zz = iz + dz, yy = iy + dy, xx = ix + dx;
        if (zz < 0 || yy < 0 || xx < 0 || zz >= b->n[0] || yy >= b->n[1] || xx >= b->n[2]) continue;
        s += (dz ? uz : 1 - uz) * (dy ? uy : 1 - uy) * (dx ? ux : 1 - ux) * b->v[((size_t)zz * b->n[1] + yy) * b->n[2] + xx];
    }
    return (float)s;
}

void reg_resample(const rblk *fb, const rmap *m, int lc, int lf, const double org[3], const int n[3], const double sh[3], int ss, float *out) {
    /* the map is exact at every 16th point of a row (and the row end) and linear in between (the field is smooth on that scale) */
    const double fc = (double)(1 << lc), ff = (double)(1 << lf);
    const int ns = ss > 1 ? 2 : 1; const double so[2] = {ns > 1 ? -0.25 : 0, 0.25};
    enum { SEG = 16 };
    for (int k = 0; k < n[0]; k++) for (int j = 0; j < n[1]; j++) {
        float *row = out + ((size_t)k * n[1] + j) * n[2];
        for (int i = 0; i < n[2]; i++) row[i] = 0;
        for (int a = 0; a < ns; a++) for (int b = 0; b < ns; b++) for (int c = 0; c < ns; c++) {
            for (int i0 = 0; i0 < n[2]; i0 += SEG) {
                const int i1 = i0 + SEG < n[2] ? i0 + SEG : n[2] - 1 > i0 ? n[2] - 1 : i0;
                double fa[3], fb2[3];
                for (int e = 0; e < 2; e++) {
                    const int ii = e ? i1 : i0;
                    const double pl[3] = {org[0] + ii + sh[0] + so[c], org[1] + j + sh[1] + so[b], org[2] + k + sh[2] + so[a]};
                    double c0[3], f0[3];
                    for (int d = 0; d < 3; d++) c0[d] = (pl[d] + 0.5) * fc - 0.5;
                    rmap_c2f(m, c0, f0);
                    for (int d = 0; d < 3; d++) (e ? fb2 : fa)[d] = (f0[d] + 0.5) / ff - 0.5;
                }
                const int iend = i0 + SEG < n[2] ? i0 + SEG : n[2];
                const double L = i1 > i0 ? (double)(i1 - i0) : 1;
                for (int i = i0; i < iend; i++) {
                    const double t = (i - i0) / L;
                    row[i] += rblk_sample(fb, fa[0] + t * (fb2[0] - fa[0]), fa[1] + t * (fb2[1] - fa[1]), fa[2] + t * (fb2[2] - fa[2]));
                }
            }
        }
        if (ns > 1) for (int i = 0; i < n[2]; i++) row[i] *= 0.125f;
    }
}

/* ---------- filters ---------- */
/* separable 1D filters along one axis of a [n0][n1][n2] volume. Along x: per row; along y / z: whole rows at a time (contiguous, vectorised).
   Edges clamped. kind 0: kernel k[0 .. 2r] (Gaussian), kind 1: three passes of a (2r + 1) box. */
static void filt_axis(float *v, const int n[3], int ax, int kind, const float *k, int r) {
    const int L = n[ax];
    if (ax == 2) {
        float *line = malloc((L + 2 * r) * sizeof *line), *o = malloc(L * sizeof *o); const float inv = 1.f / (2 * r + 1);
        for (size_t row = 0; row < (size_t)n[0] * n[1]; row++) {
            float *p = v + row * L;
            for (int pass = 0; pass < (kind ? 3 : 1); pass++) {
                for (int i = -r; i < L + r; i++) line[i + r] = p[i < 0 ? 0 : i >= L ? L - 1 : i];
                if (kind) { float acc = 0; for (int t = 0; t <= 2 * r; t++) acc += line[t]; for (int i = 0; i < L; i++) { o[i] = acc * inv; acc += line[i + 2 * r + 1 < L + 2 * r ? i + 2 * r + 1 : L + 2 * r - 1] - line[i]; } }
                else for (int i = 0; i < L; i++) { float acc = 0; for (int t = 0; t <= 2 * r; t++) acc += k[t] * line[i + t]; o[i] = acc; }
                memcpy(p, o, L * sizeof *o);
            }
        }
        free(line); free(o); return;
    }
    /* y (ax 1): rows of length n2 within each z plane; z (ax 0): rows of length n1 n2 */
    const size_t W = ax == 1 ? (size_t)n[2] : (size_t)n[1] * n[2];
    const size_t nblk = ax == 1 ? (size_t)n[0] : 1, bstride = (size_t)n[1] * n[2];
    float *tmp = malloc((size_t)L * W * sizeof *tmp), *acc = malloc(W * sizeof *acc); const float inv = 1.f / (2 * r + 1);
    for (size_t blk = 0; blk < nblk; blk++) {
        float *p = v + blk * bstride;
        for (int pass = 0; pass < (kind ? 3 : 1); pass++) {
            memcpy(tmp, p, (size_t)L * W * sizeof *tmp);
            if (kind) {
                for (size_t x = 0; x < W; x++) acc[x] = 0;
                for (int t = -r; t <= r; t++) { const float *s = tmp + (size_t)(t < 0 ? 0 : t >= L ? L - 1 : t) * W; for (size_t x = 0; x < W; x++) acc[x] += s[x]; }
                for (int i = 0; i < L; i++) {
                    float *d = p + (size_t)i * W; for (size_t x = 0; x < W; x++) d[x] = acc[x] * inv;
                    const int ad = i + r + 1, sb = i - r;
                    const float *sa = tmp + (size_t)(ad >= L ? L - 1 : ad) * W, *ss = tmp + (size_t)(sb < 0 ? 0 : sb) * W;
                    for (size_t x = 0; x < W; x++) acc[x] += sa[x] - ss[x];
                }
            } else {
                for (int i = 0; i < L; i++) {
                    float *d = p + (size_t)i * W; for (size_t x = 0; x < W; x++) d[x] = 0;
                    for (int t = -r; t <= r; t++) { const int q = i + t < 0 ? 0 : i + t >= L ? L - 1 : i + t; const float *s = tmp + (size_t)q * W, kk = k[t + r]; for (size_t x = 0; x < W; x++) d[x] += kk * s[x]; }
                }
            }
        }
    }
    free(tmp); free(acc);
}
/* Gaussian; sigma >= 2: 3 box passes of radius r with r (r + 1) = sigma^2 */
void reg_gauss(float *v, const int n[3], double sigma) {
    if (sigma <= 0) return;
    if (sigma >= 2) {
        int r = (int)lround((sqrt(4 * sigma * sigma + 1) - 1) / 2); if (r < 1) r = 1;
        for (int ax = 0; ax < 3; ax++) filt_axis(v, n, ax, 1, nullptr, r);
        return;
    }
    const int r = (int)ceil(3 * sigma); float *k = malloc((2 * r + 1) * sizeof *k); double s = 0;
    for (int i = -r; i <= r; i++) s += k[i + r] = (float)exp(-0.5 * i * i / (sigma * sigma));
    for (int i = 0; i <= 2 * r; i++) k[i] /= (float)s;
    for (int ax = 0; ax < 3; ax++) filt_axis(v, n, ax, 0, k, r);
    free(k);
}
void reg_filter(float *v, const int n[3], int metric, double hp, double lp) {
    const size_t N = (size_t)n[0] * n[1] * n[2];
    if (lp > 0) reg_gauss(v, n, lp);
    if (metric == REG_GRAD) {
        float *g = malloc(N * sizeof *g);
        for (int z = 0; z < n[0]; z++) for (int y = 0; y < n[1]; y++) for (int x = 0; x < n[2]; x++) {
            const size_t i = ((size_t)z * n[1] + y) * n[2] + x;
            const int xm = x > 0 ? x - 1 : x, xp = x + 1 < n[2] ? x + 1 : x, ym = y > 0 ? y - 1 : y, yp = y + 1 < n[1] ? y + 1 : y, zm = z > 0 ? z - 1 : z, zp = z + 1 < n[0] ? z + 1 : z;
            const float gx = (v[((size_t)z * n[1] + y) * n[2] + xp] - v[((size_t)z * n[1] + y) * n[2] + xm]) / (float)(xp - xm > 0 ? xp - xm : 1);
            const float gy = (v[((size_t)z * n[1] + yp) * n[2] + x] - v[((size_t)z * n[1] + ym) * n[2] + x]) / (float)(yp - ym > 0 ? yp - ym : 1);
            const float gz = (v[((size_t)zp * n[1] + y) * n[2] + x] - v[((size_t)zm * n[1] + y) * n[2] + x]) / (float)(zp - zm > 0 ? zp - zm : 1);
            g[i] = sqrtf(gx * gx + gy * gy + gz * gz);
        }
        memcpy(v, g, N * sizeof *v); free(g);
    }
    if ((metric == REG_DOG || metric == REG_GRAD) && hp > 0) {
        float *b = malloc(N * sizeof *b); memcpy(b, v, N * sizeof *b);
        reg_gauss(b, n, hp);
        for (size_t i = 0; i < N; i++) v[i] -= b[i];
        free(b);
    }
}

/* ---------- NCC ---------- */
/* dot product with 16 independent partial sums (vectorises without -ffast-math) */
static inline float dot16(const float *a, const float *b, int n) {
    float p[16] = {0}; int x = 0;
    for (; x + 16 <= n; x += 16) for (int k = 0; k < 16; k++) p[k] += a[x + k] * b[x + k];
    float s = 0; for (int k = 0; k < 16; k++) s += p[k];
    for (; x < n; x++) s += a[x] * b[x];
    return s;
}
void reg_ncc_map(const float *C, const int n[3], const float *G, int R, float *map) {
    const int m[3] = {n[0] + 2 * R, n[1] + 2 * R, n[2] + 2 * R}, w = 2 * R + 1;
    const size_t N = (size_t)n[0] * n[1] * n[2];
    double mc = 0; for (size_t i = 0; i < N; i++) mc += C[i]; mc /= (double)N;
    float *Cn = malloc(N * sizeof *Cn); double cn = 0;
    for (size_t i = 0; i < N; i++) { Cn[i] = (float)(C[i] - mc); cn += (double)Cn[i] * Cn[i]; }
    cn = sqrt(cn);
    /* integral volumes of G and G^2 (doubles, (m+1)^3) */
    const size_t s1 = m[2] + 1, s0 = (size_t)(m[1] + 1) * s1;
    double *I1 = calloc((size_t)(m[0] + 1) * s0, sizeof *I1), *I2 = calloc((size_t)(m[0] + 1) * s0, sizeof *I2);
    for (int z = 0; z < m[0]; z++) for (int y = 0; y < m[1]; y++) for (int x = 0; x < m[2]; x++) {
        const double g = G[((size_t)z * m[1] + y) * m[2] + x];
        const size_t i = (z + 1) * s0 + (y + 1) * s1 + x + 1;
        I1[i] = g + I1[i - 1] + I1[i - s1] + I1[i - s0] - I1[i - 1 - s1] - I1[i - 1 - s0] - I1[i - s1 - s0] + I1[i - 1 - s1 - s0];
        I2[i] = g * g + I2[i - 1] + I2[i - s1] + I2[i - s0] - I2[i - 1 - s1] - I2[i - 1 - s0] - I2[i - s1 - s0] + I2[i - 1 - s1 - s0];
    }
#define BOXSUM(I, z0, y0, x0) (I[((z0) + n[0]) * s0 + ((y0) + n[1]) * s1 + (x0) + n[2]] - I[(z0) * s0 + ((y0) + n[1]) * s1 + (x0) + n[2]] - I[((z0) + n[0]) * s0 + (y0) * s1 + (x0) + n[2]] \
    - I[((z0) + n[0]) * s0 + ((y0) + n[1]) * s1 + (x0)] + I[(z0) * s0 + (y0) * s1 + (x0) + n[2]] + I[(z0) * s0 + ((y0) + n[1]) * s1 + (x0)] + I[((z0) + n[0]) * s0 + (y0) * s1 + (x0)] - I[(z0) * s0 + (y0) * s1 + (x0)])
    for (int sz = 0; sz < w; sz++) for (int sy = 0; sy < w; sy++) for (int sx = 0; sx < w; sx++) {
        const double sg = BOXSUM(I1, sz, sy, sx), sg2 = BOXSUM(I2, sz, sy, sx);
        const double var = sg2 - sg * sg / (double)N;
        double cr = 0;
        for (int z = 0; z < n[0]; z++) {
            float accz = 0;
            for (int y = 0; y < n[1]; y++) {
                const float *c = Cn + ((size_t)z * n[1] + y) * n[2], *g = G + ((size_t)(z + sz) * m[1] + y + sy) * m[2] + sx;
                accz += dot16(c, g, n[2]);
            }
            cr += accz;
        }
        map[((size_t)sz * w + sy) * w + sx] = var > 1e-9 * N && cn > 0 ? (float)(cr / (cn * sqrt(var))) : -1.f;
    }
#undef BOXSUM
    free(Cn); free(I1); free(I2);
}

/* smallest eigenvalue of a symmetric 3x3 (closed form) */
static double sym3_min_eig(const double a[3][3]) {
    const double p1 = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
    const double q = (a[0][0] + a[1][1] + a[2][2]) / 3;
    if (p1 < 1e-30) { double e = a[0][0]; if (a[1][1] < e) e = a[1][1]; if (a[2][2] < e) e = a[2][2]; return e; }
    const double p2 = (a[0][0] - q) * (a[0][0] - q) + (a[1][1] - q) * (a[1][1] - q) + (a[2][2] - q) * (a[2][2] - q) + 2 * p1;
    const double p = sqrt(p2 / 6); double b[3][3];
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) b[i][j] = (a[i][j] - (i == j ? q : 0)) / p;
    double r = (b[0][0] * (b[1][1] * b[2][2] - b[1][2] * b[2][1]) - b[0][1] * (b[1][0] * b[2][2] - b[1][2] * b[2][0]) + b[0][2] * (b[1][0] * b[2][1] - b[1][1] * b[2][0])) / 2;
    r = r < -1 ? -1 : r > 1 ? 1 : r;
    const double phi = acos(r) / 3;
    return q + 2 * p * cos(phi + 2 * M_PI / 3);
}
int reg_quad_peak(const float f[27], double d[3], double *curv) {
#define F(dx, dy, dz) ((double)f[((dz) + 1) * 9 + ((dy) + 1) * 3 + (dx) + 1])
    double g[3] = {(F(1, 0, 0) - F(-1, 0, 0)) / 2, (F(0, 1, 0) - F(0, -1, 0)) / 2, (F(0, 0, 1) - F(0, 0, -1)) / 2};
    double H[3][3];
    H[0][0] = F(1, 0, 0) + F(-1, 0, 0) - 2 * F(0, 0, 0);
    H[1][1] = F(0, 1, 0) + F(0, -1, 0) - 2 * F(0, 0, 0);
    H[2][2] = F(0, 0, 1) + F(0, 0, -1) - 2 * F(0, 0, 0);
    H[0][1] = H[1][0] = (F(1, 1, 0) - F(1, -1, 0) - F(-1, 1, 0) + F(-1, -1, 0)) / 4;
    H[0][2] = H[2][0] = (F(1, 0, 1) - F(1, 0, -1) - F(-1, 0, 1) + F(-1, 0, -1)) / 4;
    H[1][2] = H[2][1] = (F(0, 1, 1) - F(0, 1, -1) - F(0, -1, 1) + F(0, -1, -1)) / 4;
#undef F
    double nH[3][3]; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) nH[i][j] = -H[i][j];
    const double e = sym3_min_eig(nH); *curv = e;
    if (e <= 1e-12) {   /* not a proper max: step to the best neighbour direction along the gradient */
        for (int i = 0; i < 3; i++) d[i] = g[i] > 0 ? 0.5 : g[i] < 0 ? -0.5 : 0;
        return 1;
    }
    /* solve nH d = g */
    raff t = {0}; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) t.m[i][j] = nH[i][j];
    raff ti = raff_inv(t);
    for (int i = 0; i < 3; i++) { d[i] = ti.m[i][0] * g[0] + ti.m[i][1] * g[1] + ti.m[i][2] * g[2]; d[i] = d[i] > 1 ? 1 : d[i] < -1 ? -1 : d[i]; }
    return 0;
}

/* ---------- one box ---------- */
static double frac_nonzero(const float *v, size_t n) { size_t k = 0; for (size_t i = 0; i < n; i++) k += v[i] > 0; return (double)k / (double)n; }
static double stdev(const float *v, size_t n) { double s = 0, s2 = 0; for (size_t i = 0; i < n; i++) { s += v[i]; s2 += (double)v[i] * v[i]; } s /= n; return sqrt(fmax(0, s2 / n - s * s)); }
static void crop(const float *in, const int ni[3], int p, const int no[3], float *out) {
    for (int z = 0; z < no[0]; z++) for (int y = 0; y < no[1]; y++)
        memcpy(out + ((size_t)z * no[1] + y) * no[2], in + ((size_t)(z + p) * ni[1] + y + p) * ni[2] + p, no[2] * sizeof *out);
}

/* fine block covering the coarse level-lc points org - 1 .. org + n (xyz) shifted by up to |smax| (level-lc voxels) */
uint8_t *reg_read_fine(rvol *fv, int lf, const rmap *m, int lc, const double org[3], const int n[3], double smax, rblk *fb, int *err) {
    const double fc = (double)(1 << lc), ff = (double)(1 << lf);
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) for (int c = 0; c < 3; c++) {   /* 27 points: corners, edge and face centres */
        const int sel[3] = {a, b, c}; double pl[3], c0[3], f0[3];
        for (int d = 0; d < 3; d++) { const int nd = n[2 - d]; pl[d] = org[d] + (sel[d] == 0 ? -1 - smax : sel[d] == 1 ? 0.5 * nd : nd + smax); c0[d] = (pl[d] + 0.5) * fc - 0.5; }
        rmap_c2f(m, c0, f0);
        for (int d = 0; d < 3; d++) { const double fl = (f0[d] + 0.5) / ff - 0.5; if (fl < lo[d]) lo[d] = fl; if (fl > hi[d]) hi[d] = fl; }
    }
    int64_t o[3], nn[3];
    for (int d = 0; d < 3; d++) { const int64_t a = (int64_t)floor(lo[d]) - 3, b = (int64_t)ceil(hi[d]) + 3; o[2 - d] = a; nn[2 - d] = b - a + 1; }
    const double vol = (double)nn[0] * nn[1] * nn[2];
    if (vol > 96e6) { *err = 6; return nullptr; }
    const int64_t *sh = fv->shape[lf];
    for (int d = 0; d < 3; d++) if (o[d] + nn[d] <= 0 || o[d] >= sh[d]) { *err = 2; return nullptr; }
    uint8_t *v = malloc((size_t)vol);
    if (!v || rvol_read(fv, lf, o, nn, v)) { free(v); *err = 5; return nullptr; }
    fb->v = v; memcpy(fb->o, o, sizeof o); memcpy(fb->n, nn, sizeof nn);
    return v;
}

int reg_measure(const rmeas *q, rbox *b) {
    const double fc = (double)(1 << q->lc);
    const int B = q->B, R = q->R, P = (int)ceil(3 * fmax(q->hp, q->lp)) + 1, it = q->iters > 0 ? q->iters : 4;
    rfield_eval(q->m->D, b->c, b->D0);
    b->s[0] = b->s[1] = b->s[2] = 0; b->ncc = b->ncc0 = b->second = b->curv = b->cstd = b->gstd = b->cfrac = b->gfrac = 0;
    double org[3];   /* box origin, level-lc xyz */
    for (int d = 0; d < 3; d++) org[d] = floor((b->c[d] + 0.5) / fc - 0.5 - B / 2.0 + 0.5);
    /* coarse box with filter padding */
    const int nC[3] = {B + 2 * P, B + 2 * P, B + 2 * P}, nB[3] = {B, B, B};
    const size_t NC = (size_t)nC[0] * nC[1] * nC[2], NB = (size_t)B * B * B;
    uint8_t *cu = malloc(NC); float *Cp = malloc(NC * sizeof *Cp), *C = malloc(NB * sizeof *C);
    const int64_t co[3] = {(int64_t)org[2] - P, (int64_t)org[1] - P, (int64_t)org[0] - P}, cn[3] = {nC[0], nC[1], nC[2]};
    int st = 0; uint8_t *fu = nullptr; float *Gp = nullptr, *G = nullptr, *map = nullptr, *Sp = nullptr, *S = nullptr;
    if (rvol_read(q->cv, q->lc, co, cn, cu)) { st = 5; goto done; }
    for (size_t i = 0; i < NC; i++) Cp[i] = cu[i];
    crop(Cp, nC, P, nB, C); b->cfrac = (float)frac_nonzero(C, NB);
    if (b->cfrac < q->minfrac) { st = 1; goto done; }
    reg_filter(Cp, nC, q->metric, q->hp, q->lp); crop(Cp, nC, P, nB, C);
    b->cstd = (float)stdev(C, NB);
    /* fine resampled on the expanded grid */
    rblk fb; const int E = B + 2 * R, nG[3] = {E + 2 * P, E + 2 * P, E + 2 * P}, nE[3] = {E, E, E};
    const double gorg[3] = {org[0] - R - P, org[1] - R - P, org[2] - R - P}, zero[3] = {0, 0, 0};
    fu = reg_read_fine(q->fv, q->lf, q->m, q->lc, gorg, nG, 2.0, &fb, &st);
    if (!fu) goto done;
    const size_t NG = (size_t)nG[0] * nG[1] * nG[2], NE = (size_t)E * E * E;
    Gp = malloc(NG * sizeof *Gp); G = malloc(NE * sizeof *G);
    reg_resample(&fb, q->m, q->lc, q->lf, gorg, nG, zero, q->ss, Gp);
    { const int nI[3] = {B, B, B}; float *t = malloc(NB * sizeof *t); crop(Gp, nG, P + R, nI, t); b->gfrac = (float)frac_nonzero(t, NB); free(t); }
    if (b->gfrac < q->minfrac) { st = 2; goto done; }
    reg_filter(Gp, nG, q->metric, q->hp, q->lp); crop(Gp, nG, P, nE, G);
    { const int nI[3] = {B, B, B}; float *t = malloc(NB * sizeof *t); crop(G, nE, R, nI, t); b->gstd = (float)stdev(t, NB); free(t); }
    if (b->cstd < 1e-3 || b->gstd < 1e-3) { st = 3; goto done; }
    const int w = 2 * R + 1; map = malloc((size_t)w * w * w * sizeof *map);
    reg_ncc_map(C, nB, G, R, map);
    int pk = 0; for (int i = 1; i < w * w * w; i++) if (map[i] > map[pk]) pk = i;
    b->ncc0 = map[((size_t)R * w + R) * w + R];
    const int pz = pk / (w * w), py = pk / w % w, px = pk % w;
    float sec = -1;   /* second local maximum, Chebyshev distance > 2 from the peak */
    for (int z = 0; z < w; z++) for (int y = 0; y < w; y++) for (int x = 0; x < w; x++) {
        if (abs(z - pz) <= 2 && abs(y - py) <= 2 && abs(x - px) <= 2) continue;
        const float v = map[((size_t)z * w + y) * w + x]; if (v <= sec) continue;
        int lm = 1;
        for (int dz = -1; dz <= 1 && lm; dz++) for (int dy = -1; dy <= 1 && lm; dy++) for (int dx = -1; dx <= 1 && lm; dx++) {
            const int zz = z + dz, yy = y + dy, xx = x + dx;
            if ((dz || dy || dx) && zz >= 0 && yy >= 0 && xx >= 0 && zz < w && yy < w && xx < w && map[((size_t)zz * w + yy) * w + xx] > v) lm = 0;
        }
        if (lm) sec = v;
    }
    b->second = sec;
    double s[3] = {px - R, py - R, pz - R};
    if (abs(px - R) == R || abs(py - R) == R || abs(pz - R) == R) { st = 4; b->ncc = map[pk]; for (int d = 0; d < 3; d++) b->s[d] = s[d] * fc; goto done; }
    /* sub-voxel: resample at the current shift, 3^3 NCC, quadratic step */
    const int E1 = B + 2, nS[3] = {E1 + 2 * P, E1 + 2 * P, E1 + 2 * P}, nE1[3] = {E1, E1, E1};
    const size_t NS = (size_t)nS[0] * nS[1] * nS[2];
    Sp = malloc(NS * sizeof *Sp); S = malloc((size_t)E1 * E1 * E1 * sizeof *S);
    const double sorg[3] = {org[0] - 1 - P, org[1] - 1 - P, org[2] - 1 - P};
    float f27[27]; double curv = 0;
    for (int k = 0; k < it; k++) {
        reg_resample(&fb, q->m, q->lc, q->lf, sorg, nS, s, q->ss, Sp);
        reg_filter(Sp, nS, q->metric, q->hp, q->lp); crop(Sp, nS, P, nE1, S);
        reg_ncc_map(C, nB, S, 1, f27);
        double d[3]; reg_quad_peak(f27, d, &curv);
        b->ncc = f27[13];
        double dm = 0; for (int a = 0; a < 3; a++) { s[a] += d[a]; dm = fmax(dm, fabs(d[a])); }
        if (fabs(s[0]) > R + 1 || fabs(s[1]) > R + 1 || fabs(s[2]) > R + 1) { st = 4; break; }
        if (dm < 0.01) break;
    }
    b->curv = (float)curv;
    for (int d = 0; d < 3; d++) b->s[d] = s[d] * fc;
done:
    b->status = st;
    free(cu); free(Cp); free(C); free(fu); free(Gp); free(G); free(map); free(Sp); free(S);
    return st;
}

/* ---------- fits ---------- */
static int solve_n(int n, double *A, double *b) {   /* Gaussian elimination with partial pivoting, A n x n row-major, b -> x */
    for (int c = 0; c < n; c++) {
        int p = c; for (int r = c + 1; r < n; r++) if (fabs(A[r * n + c]) > fabs(A[p * n + c])) p = r;
        if (fabs(A[p * n + c]) < 1e-300) return -1;
        if (p != c) { for (int k = 0; k < n; k++) { const double t = A[c * n + k]; A[c * n + k] = A[p * n + k]; A[p * n + k] = t; } const double t = b[c]; b[c] = b[p]; b[p] = t; }
        for (int r = c + 1; r < n; r++) { const double f = A[r * n + c] / A[c * n + c]; for (int k = c; k < n; k++) A[r * n + k] -= f * A[c * n + k]; b[r] -= f * b[c]; }
    }
    for (int c = n - 1; c >= 0; c--) { double s = b[c]; for (int k = c + 1; k < n; k++) s -= A[c * n + k] * b[k]; b[c] = s / A[c * n + c]; }
    return 0;
}

int reg_fit_affine(int n, const double *c, const double *v, const double *w, double huber, double M[3][4]) {
    double mu[3] = {0}, sw = 0;
    for (int i = 0; i < n; i++) if (w[i] > 0) { for (int d = 0; d < 3; d++) mu[d] += w[i] * c[3 * i + d]; sw += w[i]; }
    if (sw <= 0) return -1;
    for (int d = 0; d < 3; d++) mu[d] /= sw;
    double *rw = malloc(n * sizeof *rw); for (int i = 0; i < n; i++) rw[i] = w[i];
    double X[3][4] = {{0}};
    for (int iter = 0; iter < 12; iter++) {
        double A[16] = {0}, rhs[3][4] = {{0}};
        for (int i = 0; i < n; i++) {
            if (rw[i] <= 0) continue;
            const double p[4] = {c[3 * i] - mu[0], c[3 * i + 1] - mu[1], c[3 * i + 2] - mu[2], 1};
            for (int a = 0; a < 4; a++) { for (int bb = 0; bb < 4; bb++) A[a * 4 + bb] += rw[i] * p[a] * p[bb]; for (int k = 0; k < 3; k++) rhs[k][a] += rw[i] * p[a] * v[3 * i + k]; }
        }
        for (int k = 0; k < 3; k++) { double Ak[16]; memcpy(Ak, A, sizeof Ak); for (int d = 0; d < 4; d++) X[k][d] = rhs[k][d]; if (solve_n(4, Ak, X[k])) { free(rw); return -1; } }
        if (huber <= 0) break;
        for (int i = 0; i < n; i++) {
            if (w[i] <= 0) continue;
            const double p[4] = {c[3 * i] - mu[0], c[3 * i + 1] - mu[1], c[3 * i + 2] - mu[2], 1}; double e = 0;
            for (int k = 0; k < 3; k++) { const double r = v[3 * i + k] - (X[k][0] * p[0] + X[k][1] * p[1] + X[k][2] * p[2] + X[k][3]); e += r * r; }
            e = sqrt(e); rw[i] = w[i] * (e > huber ? huber / e : 1);
        }
    }
    for (int k = 0; k < 3; k++) { for (int d = 0; d < 3; d++) M[k][d] = X[k][d]; M[k][3] = X[k][3] - X[k][0] * mu[0] - X[k][1] * mu[1] - X[k][2] * mu[2]; }
    free(rw);
    return 0;
}

/* trilinear weights of point p on the grid: 8 node indices + weights */
static void tri_w(const rfield *f, const double p[3], size_t idx[8], double wt[8]) {
    double u[3]; int i0[3];
    for (int a = 0; a < 3; a++) {
        const int na = f->n[2 - a]; double g = (p[a] - f->o[a]) / f->h;
        if (g < 0) g = 0;
        if (g > na - 1) g = na - 1;
        int i = (int)floor(g); if (i > na - 2) i = na - 2; if (i < 0) i = 0;
        i0[a] = i; u[a] = g - i;
    }
    int k = 0;
    for (int dz = 0; dz < 2; dz++) for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++, k++) {
        idx[k] = ((size_t)(i0[2] + dz) * f->n[1] + i0[1] + dy) * f->n[2] + i0[0] + dx;
        wt[k] = (dz ? u[2] : 1 - u[2]) * (dy ? u[1] : 1 - u[1]) * (dx ? u[0] : 1 - u[0]);
    }
}
typedef struct { const rfield *f; int n; const size_t *idx; const double *wt; const double *rw; double lam, mu; } fop;
static void fop_apply(const fop *o, const double *x, double *y) {
    const int nz = o->f->n[0], ny = o->f->n[1], nx = o->f->n[2]; const size_t N = (size_t)nz * ny * nx;
    for (size_t i = 0; i < N; i++) y[i] = o->mu * x[i];
    for (int z = 0; z < nz; z++) for (int yy = 0; yy < ny; yy++) for (int xx = 0; xx < nx; xx++) {
        const size_t i = ((size_t)z * ny + yy) * nx + xx; double s = 0;
        if (xx > 0) s += x[i] - x[i - 1];
        if (xx + 1 < nx) s += x[i] - x[i + 1];
        if (yy > 0) s += x[i] - x[i - nx];
        if (yy + 1 < ny) s += x[i] - x[i + nx];
        if (z > 0) s += x[i] - x[i - (size_t)nx * ny];
        if (z + 1 < nz) s += x[i] - x[i + (size_t)nx * ny];
        y[i] += o->lam * s;
    }
    for (int b = 0; b < o->n; b++) {
        if (o->rw[b] <= 0) continue;
        double s = 0; for (int k = 0; k < 8; k++) s += o->wt[8 * b + k] * x[o->idx[8 * b + k]];
        s *= o->rw[b];
        for (int k = 0; k < 8; k++) y[o->idx[8 * b + k]] += o->wt[8 * b + k] * s;
    }
}
static void cg_solve(const fop *o, const double *rhs, double *x, size_t N) {
    double *r = malloc(N * sizeof *r), *p = malloc(N * sizeof *p), *Ap = malloc(N * sizeof *Ap);
    fop_apply(o, x, Ap);
    double rr = 0, bb = 0;
    for (size_t i = 0; i < N; i++) { r[i] = rhs[i] - Ap[i]; p[i] = r[i]; rr += r[i] * r[i]; bb += rhs[i] * rhs[i]; }
    for (int it = 0; it < 5000 && rr > 1e-14 * (bb + 1e-30); it++) {
        fop_apply(o, p, Ap);
        double pAp = 0; for (size_t i = 0; i < N; i++) pAp += p[i] * Ap[i];
        if (pAp <= 0) break;
        const double al = rr / pAp; double rn = 0;
        for (size_t i = 0; i < N; i++) { x[i] += al * p[i]; r[i] -= al * Ap[i]; rn += r[i] * r[i]; }
        const double be = rn / rr; rr = rn;
        for (size_t i = 0; i < N; i++) p[i] = r[i] + be * p[i];
    }
    free(r); free(p); free(Ap);
}
int reg_fit_field(int n, const double *c, const double *v, const double *w, double h, double lam, double huber, int margin, rfield *out) {
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300}; int used = 0;
    for (int i = 0; i < n; i++) if (w[i] > 0) { used++; for (int d = 0; d < 3; d++) { lo[d] = fmin(lo[d], c[3 * i + d]); hi[d] = fmax(hi[d], c[3 * i + d]); } }
    if (!used) return -1;
    rfield *f = out; f->h = h;
    for (int d = 0; d < 3; d++) { f->o[d] = lo[d] - margin * h; f->n[2 - d] = (int)ceil((hi[d] - lo[d]) / h) + 1 + 2 * margin; if (f->n[2 - d] < 2) f->n[2 - d] = 2; }
    const size_t N = (size_t)f->n[0] * f->n[1] * f->n[2];
    f->d = calloc(N * 3, sizeof *f->d);
    size_t *idx = malloc((size_t)n * 8 * sizeof *idx); double *wt = malloc((size_t)n * 8 * sizeof *wt), *rw = malloc(n * sizeof *rw);
    for (int i = 0; i < n; i++) { tri_w(f, c + 3 * i, idx + 8 * i, wt + 8 * i); rw[i] = w[i]; }
    /* data weight per touched node -> the regulariser scale */
    double *nw = calloc(N, sizeof *nw), sw = 0; size_t touched = 0;
    for (int i = 0; i < n; i++) if (w[i] > 0) for (int k = 0; k < 8; k++) nw[idx[8 * i + k]] += w[i] * wt[8 * i + k] * wt[8 * i + k];
    for (size_t i = 0; i < N; i++) if (nw[i] > 0) { sw += nw[i]; touched++; }
    free(nw);
    const double scale = touched ? sw / touched : 1;
    fop o = {f, n, idx, wt, rw, lam * scale, 1e-3 * lam * scale + 1e-9};
    double *x = calloc(N, sizeof *x), *rhs = malloc(N * sizeof *rhs);
    for (int round = 0; round < (huber > 0 ? 5 : 1); round++) {
        for (int k = 0; k < 3; k++) {
            for (size_t i = 0; i < N; i++) { rhs[i] = 0; x[i] = f->d[3 * i + k]; }
            for (int b = 0; b < n; b++) if (rw[b] > 0) for (int j = 0; j < 8; j++) rhs[idx[8 * b + j]] += rw[b] * wt[8 * b + j] * v[3 * b + k];
            cg_solve(&o, rhs, x, N);
            for (size_t i = 0; i < N; i++) f->d[3 * i + k] = (float)x[i];
        }
        if (huber <= 0) break;
        for (int b = 0; b < n; b++) {
            if (w[b] <= 0) continue;
            double p[3]; rfield_eval(f, c + 3 * b, p); double e = 0;
            for (int k = 0; k < 3; k++) e += (v[3 * b + k] - p[k]) * (v[3 * b + k] - p[k]);
            e = sqrt(e); rw[b] = w[b] * (e > huber ? huber / e : 1);
        }
    }
    free(x); free(rhs); free(idx); free(wt); free(rw);
    return 0;
}
