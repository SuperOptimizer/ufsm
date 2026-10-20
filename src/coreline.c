/* Core line of a crushed scroll: detection from the fit fields, signed distance, half-box masks (see coreline.h). */
#include "coreline.h"
#include "json.h"
#include "store.h"
#include "zarr3.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static json *cl_read_json(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return nullptr;
    fseek(f, 0, SEEK_END); const size_t n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    char *t = malloc(n + 1); if (!t || fread(t, 1, n, f) != n) { fclose(f); free(t); return nullptr; }
    fclose(f); t[n] = 0; json *j = json_parse(t, n); free(t); return j;
}

int cl_load(coreline *c, const char *path) {
    memset(c, 0, sizeof *c);
    json *j = cl_read_json(path); if (!j) return -1;
    c->um = json_num(json_get(j, "um"), 2.399);
    const json *S = json_get(j, "slabs"); const int ns = S ? (int)S->n : 0;
    c->s = calloc(ns > 0 ? ns : 1, sizeof *c->s);
    for (int k = 0; k < ns; k++) {
        const json *s = json_at(S, k), *P = json_get(s, "pts"); const int n = P ? (int)P->n : 0;
        if (n < 2) continue;
        cl_slab *d = &c->s[c->ns++];
        d->z0 = json_num(json_get(s, "z0"), 0); d->z1 = json_num(json_get(s, "z1"), 0); d->zc = json_num(json_get(s, "z"), (d->z0 + d->z1) / 2);
        d->n = n; d->y = malloc(n * sizeof(double)); d->x = malloc(n * sizeof(double));
        for (int i = 0; i < n; i++) { d->y[i] = json_num(json_at(json_at(P, i), 0), 0); d->x[i] = json_num(json_at(json_at(P, i), 1), 0); }
    }
    json_free(j);
    for (int a = 1; a < c->ns; a++) for (int b = a; b > 0 && c->s[b].zc < c->s[b - 1].zc; b--) { cl_slab t = c->s[b]; c->s[b] = c->s[b - 1]; c->s[b - 1] = t; }
    return c->ns > 0 ? 0 : -1;
}

int cl_save(const coreline *c, const char *path, const char *extra) {
    FILE *f = fopen(path, "w"); if (!f) return -1;
    fprintf(f, "{\"content\": \"core line (flattened umbilicus) per z-slab: polyline points [y, x], z0 / z1 / z, in voxels of um (continuous, global)\",\n \"um\": %.4f,%s%s\n \"slabs\": [", c->um, extra ? "\n " : "", extra ? extra : "");
    for (int k = 0; k < c->ns; k++) {
        const cl_slab *s = &c->s[k];
        fprintf(f, "%s\n  {\"z0\": %.1f, \"z1\": %.1f, \"z\": %.1f, \"pts\": [", k ? "," : "", s->z0, s->z1, s->zc);
        for (int i = 0; i < s->n; i++) fprintf(f, "%s[%.1f, %.1f]", i ? ", " : "", s->y[i], s->x[i]);
        fprintf(f, "]}");
    }
    fprintf(f, "\n ]}\n");
    return fclose(f) ? -1 : 0;
}

void cl_free(coreline *c) { for (int k = 0; k < c->ns; k++) { free(c->s[k].y); free(c->s[k].x); } free(c->s); memset(c, 0, sizeof *c); }

double cl_poly_sdist(const double *Y, const double *X, int n, double py, double px, double q[2], double nrm[2]) {
    double best = 1e300, qy = Y[0], qx = X[0], ty = 0, tx = 1;
    int bi = -1; double bt = 0;   /* the nearest segment and its parameter: its tangent once, after the scan (the same values as
                                     computing it at every improvement, which took ~60% of the masks' time in hypot) */
    /* Groups of CL_G segments whose bounding box is farther than an upper bound B of the nearest distance (the nearest segment of
       the nearest box) are skipped: their segments cannot be the first nearest one, so the ordered scan of the rest finds the same
       segment, distance and point (the margin covers rounding). */
    enum { CL_G = 8, CL_MAXG = 512 };
    const int ns = n - 1, ng = (ns + CL_G - 1) / CL_G;
    double lb[CL_MAXG]; double B = 1e300;
    if (ns > 2 * CL_G && ng <= CL_MAXG) {
        int gb = 0;
        for (int g = 0; g < ng; g++) {
            const int a = g * CL_G, e = a + CL_G < ns ? a + CL_G : ns;   /* segments [a, e): points a..e */
            double ylo = Y[a], yhi = Y[a], xlo = X[a], xhi = X[a];
            for (int k = a + 1; k <= e; k++) { ylo = Y[k] < ylo ? Y[k] : ylo; yhi = Y[k] > yhi ? Y[k] : yhi; xlo = X[k] < xlo ? X[k] : xlo; xhi = X[k] > xhi ? X[k] : xhi; }
            const double dy = py < ylo ? ylo - py : py > yhi ? py - yhi : 0, dx = px < xlo ? xlo - px : px > xhi ? px - xhi : 0;
            lb[g] = dy * dy + dx * dx; if (lb[g] < lb[gb]) gb = g;
        }
        for (int i = gb * CL_G; i < (gb + 1) * CL_G && i < ns; i++) {
            const double ey = Y[i + 1] - Y[i], ex = X[i + 1] - X[i], l2 = ey * ey + ex * ex; if (l2 <= 0) continue;
            double t = ((py - Y[i]) * ey + (px - X[i]) * ex) / l2; t = t < 0 ? 0 : t > 1 ? 1 : t;
            const double cy = Y[i] + t * ey, cx = X[i] + t * ex, d2 = (py - cy) * (py - cy) + (px - cx) * (px - cx);
            if (d2 < B) B = d2;
        }
        B = B * (1 + 1e-9) + 1e-9;
    }
    for (int i = 0; i + 1 < n; i++) {
        if (B < 1e300 && i % CL_G == 0 && lb[i / CL_G] > B) { i += CL_G - 1; continue; }
        const double ey = Y[i + 1] - Y[i], ex = X[i + 1] - X[i], l2 = ey * ey + ex * ex; if (l2 <= 0) continue;
        double t = ((py - Y[i]) * ey + (px - X[i]) * ex) / l2; t = t < 0 ? 0 : t > 1 ? 1 : t;
        const double cy = Y[i] + t * ey, cx = X[i] + t * ex, d2 = (py - cy) * (py - cy) + (px - cx) * (px - cx);
        if (d2 < best) { best = d2; qy = cy; qx = cx; bi = i; bt = t; }
    }
    if (bi >= 0) {
        const int i = bi; const double t = bt;
        const double ey = Y[i + 1] - Y[i], ex = X[i + 1] - X[i], l2 = ey * ey + ex * ex;
        const double l = sqrt(l2); ty = ey / l; tx = ex / l;
        /* at an interior vertex the tangent is the mean of the two segments' (the sign of a point beyond a corner) */
        const int v = t <= 0 ? i : t >= 1 ? i + 1 : -1;
        if (v > 0 && v < n - 1) {
            const double ay = Y[v] - Y[v - 1], ax = X[v] - X[v - 1], by = Y[v + 1] - Y[v], bx = X[v + 1] - X[v];
            const double la = hypot(ay, ax), lb = hypot(by, bx);
            if (la > 0 && lb > 0) { double my = ay / la + by / lb, mx = ax / la + bx / lb, lm = hypot(my, mx); if (lm > 1e-9) { ty = my / lm; tx = mx / lm; } }
        }
    }
    const double ny = tx, nx = -ty;   /* left normal: n . d = cross(t, d) */
    if (q) { q[0] = qy; q[1] = qx; }
    if (nrm) { nrm[0] = ny; nrm[1] = nx; }
    const double s = (py - qy) * ny + (px - qx) * nx;
    return (s < 0 ? -1 : 1) * sqrt(best);
}

double cl_sdist(const coreline *c, double z, double y, double x, double q[2], double n[2]) {
    if (c->ns <= 0) { if (q) q[0] = q[1] = 0; if (n) { n[0] = 1; n[1] = 0; } return 0; }
    int a = 0; while (a + 1 < c->ns && c->s[a + 1].zc <= z) a++;
    const int b = a + 1 < c->ns && z > c->s[a].zc ? a + 1 : a;
    const double t = b == a ? 0 : (z - c->s[a].zc) / (c->s[b].zc - c->s[a].zc);
    double qa[2], na[2], qb[2], nb[2];
    const double da = cl_poly_sdist(c->s[a].y, c->s[a].x, c->s[a].n, y, x, qa, na);
    const double db = b == a ? da : cl_poly_sdist(c->s[b].y, c->s[b].x, c->s[b].n, y, x, qb, nb);
    if (b == a) { if (q) { q[0] = qa[0]; q[1] = qa[1]; } if (n) { n[0] = na[0]; n[1] = na[1]; } return da; }
    if (q) { q[0] = (1 - t) * qa[0] + t * qb[0]; q[1] = (1 - t) * qa[1] + t * qb[1]; }
    if (n) { double my = (1 - t) * na[0] + t * nb[0], mx = (1 - t) * na[1] + t * nb[1], l = hypot(my, mx); if (l < 1e-9) { my = na[0]; mx = na[1]; l = 1; } n[0] = my / l; n[1] = mx / l; }
    return (1 - t) * da + t * db;
}

/* ---- detection ---- */
cl_params cl_params_default(void) { return (cl_params){1.5, 0.15, 0.5, 0.2, 20, 2, 0.5, 2, 0.35}; }
void cl_line_free(cl_line *l) { free(l->y); free(l->x); memset(l, 0, sizeof *l); }

static void blur1(float *a, int H, int W, double s, float *tmp) {   /* separable Gaussian, zero outside */
    if (s <= 0) return;
    const int r = (int)ceil(3 * s); double *k = malloc((2 * r + 1) * sizeof(double));
    for (int i = -r; i <= r; i++) k[i + r] = exp(-0.5 * i * i / (s * s));
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        double v = 0; for (int i = -r; i <= r; i++) { const int xx = x + i; if (xx >= 0 && xx < W) v += k[i + r] * a[(size_t)y * W + xx]; }
        tmp[(size_t)y * W + x] = (float)v;
    }
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        double v = 0; for (int i = -r; i <= r; i++) { const int yy = y + i; if (yy >= 0 && yy < H) v += k[i + r] * tmp[(size_t)yy * W + x]; }
        a[(size_t)y * W + x] = (float)v;
    }
    double ks = 0; for (int i = -r; i <= r; i++) ks += k[i + r];
    const double nrm = 1 / (ks * ks); for (size_t i = 0; i < (size_t)H * W; i++) a[i] = (float)(a[i] * nrm);
    free(k);
}
static double bil(const float *a, int H, int W, double y, double x) {   /* bilinear at continuous (bin i spans [i, i + 1)), 0 outside */
    y -= 0.5; x -= 0.5; const int i = (int)floor(y), j = (int)floor(x); const double fy = y - i, fx = x - j; double v = 0;
    for (int k = 0; k < 4; k++) {
        const int yy = i + (k >> 1), xx = j + (k & 1); if (yy < 0 || xx < 0 || yy >= H || xx >= W) continue;
        v += (k >> 1 ? fy : 1 - fy) * (k & 1 ? fx : 1 - fx) * a[(size_t)yy * W + xx];
    }
    return v;
}
static int cmp_d(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

int cl_detect(const float *gy0, const float *gx0, const float *w0, int H, int W, const cl_params *p, cl_line *out) {
    memset(out, 0, sizeof *out);
    const size_t N = (size_t)H * W;
    float *gy = malloc(N * 4), *gx = malloc(N * 4), *ws = malloc(N * 4), *tmp = malloc(N * 4), *m = malloc(N * 4), *gm = malloc(N * 4), *ga = malloc(N * 4);
    memcpy(gy, gy0, N * 4); memcpy(gx, gx0, N * 4); memcpy(ws, w0, N * 4);
    blur1(gy, H, W, p->sigma, tmp); blur1(gx, H, W, p->sigma, tmp); blur1(ws, H, W, 1.0, tmp);
    double sw = 0, cy = 0, cx = 0;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) { const size_t i = (size_t)y * W + x; m[i] = ws[i] >= p->wthr; if (m[i]) { sw++; cy += y + 0.5; cx += x + 0.5; } }
    int rc = -1;
    if (sw < 50) goto done;
    cy /= sw; cx /= sw;
    double cyy = 0, cxx = 0, cyx = 0;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) if (m[(size_t)y * W + x]) { const double dy = y + 0.5 - cy, dx = x + 0.5 - cx; cyy += dy * dy; cxx += dx * dx; cyx += dy * dx; }
    const double th = 0.5 * atan2(2 * cyx, cyy - cxx);   /* major axis angle from the y axis */
    const double ma[2] = {cos(th), sin(th)}, mi[2] = {-ma[1], ma[0]};
    out->ma[0] = ma[0]; out->ma[1] = ma[1];
    float *gl = malloc(N * 4);   /* the major-axis component: ~0 along the straight sides, one-signed in the end caps */
    for (size_t i = 0; i < N; i++) { gm[i] = (float)((gy[i] * mi[0] + gx[i] * mi[1]) * m[i]); gl[i] = (float)((gy[i] * ma[0] + gx[i] * ma[1]) * m[i]); ga[i] = (float)(hypot(gy[i], gx[i]) * m[i]); }
    double t0 = 1e300, t1 = -1e300, s0 = 1e300, s1 = -1e300;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) if (m[(size_t)y * W + x]) {
        const double dy = y + 0.5 - cy, dx = x + 0.5 - cx, t = dy * ma[0] + dx * ma[1], s = dy * mi[0] + dx * mi[1];
        t0 = fmin(t0, t); t1 = fmax(t1, t); s0 = fmin(s0, s); s1 = fmax(s1, s);
    }
    s0 -= 2; s1 += 2;
    const int nt = (int)((t1 - t0) / p->dt) + 1, nsm = (int)((s1 - s0) / p->ds) + 2;
    double *sp = malloc(nt * sizeof(double)), *tt = malloc(nt * sizeof(double)), *con = malloc(nt * sizeof(double));
    int *ok = calloc(nt, sizeof(int)); double *g = malloc(nsm * sizeof(double)), *mm = malloc(nsm * sizeof(double));
    double csum = 0, wsum = 0; int nk = 0;
    for (int k = 0; k < nt; k++) {
        const double t = t0 + k * p->dt; tt[k] = t;
        double tot = 0, at = 0, mt = 0, al = 0;
        for (int j = 0; j < nsm; j++) {
            const double s = s0 + j * p->ds, py = cy + t * ma[0] + s * mi[0], px = cx + t * ma[1] + s * mi[1];
            g[j] = bil(gm, H, W, py, px); mm[j] = bil(m, H, W, py, px); tot += g[j]; at += bil(ga, H, W, py, px); mt += mm[j]; al += bil(gl, H, W, py, px);
        }
        double cs = 0, best = -1e300, mb = 0, mcum = 0; int bj = 0;
        for (int j = 0; j < nsm; j++) {   /* split after sample j: -sum_{<= j} g + sum_{> j} g = tot - 2 cs */
            cs += g[j]; mcum += mm[j]; const double sc = tot - 2 * cs; if (sc > best) { best = sc; bj = j; mb = mcum; }
        }
        sp[k] = s0 + (bj + 0.5) * p->ds; con[k] = at > 0 ? best / at : 0;
        const double width = mt * p->ds, side = mt > 0 ? fmin(mb, mt - mb) / mt : 0;
        ok[k] = con[k] >= p->min_contrast && side >= p->min_side && width >= p->min_width && (at > 0 && fabs(al) / at < p->max_along);
    }
    /* the longest run of kept lines (gaps of <= 2 lines bridged) */
    int bs = -1, be = -1, rs = -1, last = -10;
    for (int k = 0; k < nt; k++) if (ok[k]) {
        if (k - last > 3) rs = k;
        last = k;
        if (be - bs < k - rs) { bs = rs; be = k; }
    }
    if (bs < 0 || be - bs < 3) goto done2;
    {
        int n = 0; double *ky = malloc((be - bs + 1) * sizeof(double)), *kt = malloc((be - bs + 1) * sizeof(double));
        for (int k = bs; k <= be; k++) if (ok[k]) { ky[n] = sp[k]; kt[n] = tt[k]; n++; csum += con[k]; wsum++; nk++; }
        double *med = malloc(n * sizeof(double)), win[64];
        for (int i = 0; i < n; i++) {
            int c = 0; for (int d = -p->median; d <= p->median && c < 64; d++) { const int q = i + d; if (q >= 0 && q < n) win[c++] = ky[q]; }
            qsort(win, c, sizeof(double), cmp_d); med[i] = win[c / 2];
        }
        out->n = n; out->y = malloc(n * sizeof(double)); out->x = malloc(n * sizeof(double));
        for (int i = 0; i < n; i++) { out->y[i] = cy + kt[i] * ma[0] + med[i] * mi[0]; out->x[i] = cx + kt[i] * ma[1] + med[i] * mi[1]; }
        free(ky); free(kt); free(med);
        out->contrast = wsum > 0 ? csum / wsum : 0; out->lines = nt; out->kept = nk; out->width = t1 - t0;
        rc = 0;
    }
done2:
    free(sp); free(tt); free(con); free(ok); free(g); free(mm); free(gl);
done:
    free(gy); free(gx); free(ws); free(tmp); free(m); free(gm); free(ga);
    return rc;
}

void cl_accum_row(const uint8_t *pc, const uint8_t *ps, const uint8_t *w, int H, int W, int bin, float *gy, float *gx, float *wb) {
    const int Wb = (W + bin - 1) / bin;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const size_t i = (size_t)y * W + x; const size_t b = (size_t)(y / bin) * Wb + x / bin;
        wb[b] += w[i] / 255.f;
        if (!w[i] || y == 0 || x == 0 || y == H - 1 || x == W - 1) continue;
        const float mr = (pc[i] - 128) / 127.f, mi = (ps[i] - 128) / 127.f;
        const float dyr = (pc[i + W] - pc[i - W]) / 254.f, dyi = (ps[i + W] - ps[i - W]) / 254.f;
        const float dxr = (pc[i + 1] - pc[i - 1]) / 254.f, dxi = (ps[i + 1] - ps[i - 1]) / 254.f;
        gy[b] += mr * dyi - mi * dyr; gx[b] += mr * dxi - mi * dxr;   /* Im(conj(m) dm) */
    }
}

/* ---- half-box masks ---- */
size_t cl_mask_block(const coreline *c, int side, const long long org[3], double h, double um, int z0, int nz, int H, int W,
                     uint8_t *pc, uint8_t *ps, uint8_t *w) {
    const double f = um / c->um;   /* native -> core-line voxels */
    const int B = 4, G = 16, Hg = (H + G - 1) / G, Wg = (W + G - 1) / G; const double thr = 3.0 * h * f, thrg = 12.0 * h * f;   /* > the half diagonals (2.83 / 11.3 cells) */
    size_t nb = 0;
    for (int a = 0; a < nz; a++) {
        const double zg = (org[0] + (z0 + a + 0.5) * h) * f; const size_t ro = (size_t)a * H * W;
        for (int gy = 0; gy < Hg; gy++) for (int gx = 0; gx < Wg; gx++) {   /* 16 x 16 cell groups far from the line: one side */
          const int gylo = gy * G, gxlo = gx * G, gyhi = gylo + G < H ? gylo + G : H, gxhi = gxlo + G < W ? gxlo + G : W;
          const double sg = cl_sdist(c, zg, (org[1] + (gylo + G / 2.0) * h) * f, (org[2] + (gxlo + G / 2.0) * h) * f, nullptr, nullptr);
          if (fabs(sg) > thrg && side * sg >= 0) continue;   /* the kept side: nothing to blank */
          for (int ylo = gylo; ylo < gyhi; ylo += B) for (int xlo = gxlo; xlo < gxhi; xlo += B) {
            const int yhi = ylo + B < H ? ylo + B : H, xhi = xlo + B < W ? xlo + B : W;
            if (w) { int any = 0; for (int y = ylo; y < yhi && !any; y++) for (int x = xlo; x < xhi; x++) if (w[ro + (size_t)y * W + x]) { any = 1; break; } if (!any) continue; }
            const double sd = fabs(sg) > thrg ? sg : cl_sdist(c, zg, (org[1] + (ylo + B / 2.0) * h) * f, (org[2] + (xlo + B / 2.0) * h) * f, nullptr, nullptr);
            const int uniform = fabs(sg) > thrg || fabs(sd) > thr;
            for (int y = ylo; y < yhi; y++) for (int x = xlo; x < xhi; x++) {
                const double s = uniform ? sd : cl_sdist(c, zg, (org[1] + (y + 0.5) * h) * f, (org[2] + (x + 0.5) * h) * f, nullptr, nullptr);
                if (side * s >= 0) continue;
                const size_t i = ro + (size_t)y * W + x;
                if (w) { nb += w[i] != 0; w[i] = 0; } else nb++;
                if (pc) pc[i] = 128;
                if (ps) ps[i] = 128;
            }
          }
        }
    }
    return nb;
}

/* ---- commands ---- */
static const char *copt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}
static int fields_meta(const char *zarr, double *um, int *h, int64_t shp[3]) {
    char p[1400]; snprintf(p, sizeof p, "%s/fields.json", zarr); json *j = cl_read_json(p); if (!j) return -1;
    *um = json_num(json_get(j, "um"), 2.4); *h = (int)json_num(json_get(j, "h"), 1);
    for (int d = 0; d < 3; d++) shp[d] = (int64_t)json_num(json_at(json_get(j, "shape"), d), 0);
    json_free(j); return 0;
}

/* ufsm coreline detect ZARR OUT.json [--slab 64] [--bin 4] [--z0 0] [--z1 D] [--um 2.399] [--threads 8] */
static int cl_cmd_detect(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm coreline detect <fit-field zarr> <out.json> [--slab 64] [--bin 4] [--z0 C] [--z1 C] [--um 2.399] [--threads 8]\n"); return 2; }
    const char *zr = argv[3], *outp = argv[4];
    double um; int h; int64_t shp[3];
    if (fields_meta(zr, &um, &h, shp)) { fprintf(stderr, "coreline: cannot read %s/fields.json\n", zr); return 1; }
    const int slab = atoi(copt(argc, argv, "--slab", "64")), bin = atoi(copt(argc, argv, "--bin", "4")), nt = atoi(copt(argc, argv, "--threads", "8"));
    const int zlo = atoi(copt(argc, argv, "--z0", "0")), zhi = atoi(copt(argc, argv, "--z1", "0")) > 0 ? atoi(copt(argc, argv, "--z1", "0")) : (int)shp[0];
    const double oum = atof(copt(argc, argv, "--um", "2.399")), f = h * um / oum;   /* cells -> output voxels */
    cl_params P = cl_params_default();
    P.sigma = atof(copt(argc, argv, "--sigma", "1.5")); P.min_side = atof(copt(argc, argv, "--min-side", "0.2")); P.min_contrast = atof(copt(argc, argv, "--min-contrast", "0.5")); P.max_along = atof(copt(argc, argv, "--max-along", "0.35"));
    const char *nm[3] = {"pc", "ps", "w"}; z3 *za[3];
    for (int k = 0; k < 3; k++) if (!(za[k] = z3_open(store_open(zr), nm[k], nullptr))) { fprintf(stderr, "coreline: %s\n", z3_error()); return 1; }
    const int H = (int)shp[1], W = (int)shp[2], Hb = (H + bin - 1) / bin, Wb = (W + bin - 1) / bin, RB = 8;
    uint8_t *buf[3]; for (int k = 0; k < 3; k++) buf[k] = malloc((size_t)RB * H * W);
    float *gy = malloc((size_t)Hb * Wb * 4), *gx = malloc((size_t)Hb * Wb * 4), *wb = malloc((size_t)Hb * Wb * 4);
    coreline C = {0}; C.um = oum; C.s = calloc((zhi - zlo) / slab + 2, sizeof(cl_slab));
    double ref[2] = {0, 0};
    for (int z0 = zlo; z0 < zhi; z0 += slab) {
        const int z1 = z0 + slab < zhi ? z0 + slab : zhi;
        memset(gy, 0, (size_t)Hb * Wb * 4); memset(gx, 0, (size_t)Hb * Wb * 4); memset(wb, 0, (size_t)Hb * Wb * 4);
        for (int r = z0; r < z1; r += RB) {
            const int nr = r + RB <= z1 ? RB : z1 - r; const int64_t o[3] = {r, 0, 0}, n[3] = {nr, H, W};
            for (int k = 0; k < 3; k++) if (z3_read(za[k], o, n, buf[k], nt)) { fprintf(stderr, "coreline: %s\n", z3_error()); return 1; }
            for (int a = 0; a < nr; a++) cl_accum_row(buf[0] + (size_t)a * H * W, buf[1] + (size_t)a * H * W, buf[2] + (size_t)a * H * W, H, W, bin, gy, gx, wb);
        }
        const double norm = 1.0 / ((double)(z1 - z0) * bin * bin); for (size_t i = 0; i < (size_t)Hb * Wb; i++) wb[i] = (float)(wb[i] * norm);
        cl_line L;
        if (cl_detect(gy, gx, wb, Hb, Wb, &P, &L)) { fprintf(stderr, "coreline: z cells [%d, %d): no core line\n", z0, z1); continue; }
        double dy = L.y[L.n - 1] - L.y[0], dx = L.x[L.n - 1] - L.x[0];
        if (ref[0] == 0 && ref[1] == 0) { ref[0] = dy; ref[1] = dx; }
        if (dy * ref[0] + dx * ref[1] < 0) {   /* the same orientation as the previous slab */
            for (int i = 0; i < L.n / 2; i++) { double t = L.y[i]; L.y[i] = L.y[L.n - 1 - i]; L.y[L.n - 1 - i] = t; t = L.x[i]; L.x[i] = L.x[L.n - 1 - i]; L.x[L.n - 1 - i] = t; }
            dy = -dy; dx = -dx;
        }
        ref[0] = dy; ref[1] = dx;
        cl_slab *s = &C.s[C.ns++]; s->z0 = z0 * f; s->z1 = z1 * f; s->zc = (z0 + z1) / 2.0 * f; s->n = L.n;
        s->y = malloc(L.n * sizeof(double)); s->x = malloc(L.n * sizeof(double));
        for (int i = 0; i < L.n; i++) { s->y[i] = L.y[i] * bin * f; s->x[i] = L.x[i] * bin * f; }
        fprintf(stderr, "coreline: z cells [%d, %d): %d points, length %.1f mm, contrast %.2f, %d of %d lines kept, ends (y %.0f x %.0f) (y %.0f x %.0f)\n", z0, z1, L.n,
                hypot(dy, dx) * bin * f * oum / 1000, L.contrast, L.kept, L.lines, s->y[0], s->x[0], s->y[L.n - 1], s->x[L.n - 1]);
        cl_line_free(&L);
    }
    char extra[512]; snprintf(extra, sizeof extra, "\"source\": \"%s\", \"slab_cells\": %d, \"bin_cells\": %d, \"cell_um\": %.4f,", zr, slab, bin, h * um);
    const int rc = cl_save(&C, outp, extra);
    fprintf(stderr, "coreline: %d slabs with a core line -> %s\n", C.ns, outp);
    cl_free(&C); for (int k = 0; k < 3; k++) { free(buf[k]); z3_close(za[k]); } free(gy); free(gx); free(wb);
    return rc ? 1 : 0;
}

/* the smoothed core frame of a box: over z rows every 16 native voxels, the core point nearest the box centre and the normal of a
   straight line fitted (principal direction) to the nearest slab's polyline points within R core-line voxels of it, oriented as
   the local polyline normal; both averaged over the rows. (The raw nearest-point normal swings with every zigzag of the detected
   polyline: 2000 native voxels behind the core that moved the axis by up to ~2.5 cm within one box.) */
void cl_box_frame(const coreline *C, const long long org[3], int na, double cyg, double cxg, double f, double R, double q[2], double n[2]) {
    double sq[2] = {0, 0}, sn[2] = {0, 0}; int cnt = 0;
    for (int z = 0; z < na; z += 16) {
        const double zg = (org[0] + z) * f; double qq[2], nn[2]; cl_sdist(C, zg, cyg * f, cxg * f, qq, nn);
        int k = 0; for (int i = 1; i < C->ns; i++) if (fabs(C->s[i].zc - zg) < fabs(C->s[k].zc - zg)) k = i;
        const cl_slab *S = &C->s[k]; double my = 0, mx = 0, syy = 0, sxx = 0, syx = 0; int m = 0;
        for (int i = 0; i < S->n; i++) if (hypot(S->y[i] - qq[0], S->x[i] - qq[1]) <= R) { my += S->y[i]; mx += S->x[i]; m++; }
        double ny = nn[0], nx = nn[1];
        if (m >= 3) {
            my /= m; mx /= m;
            for (int i = 0; i < S->n; i++) if (hypot(S->y[i] - qq[0], S->x[i] - qq[1]) <= R) { const double dy = S->y[i] - my, dx = S->x[i] - mx; syy += dy * dy; sxx += dx * dx; syx += dy * dx; }
            const double th = 0.5 * atan2(2 * syx, syy - sxx), ty = cos(th), tx = sin(th);   /* principal direction (angle from the y axis) */
            ny = tx; nx = -ty; if (ny * nn[0] + nx * nn[1] < 0) { ny = -ny; nx = -nx; }
        }
        sq[0] += qq[0]; sq[1] += qq[1]; sn[0] += ny; sn[1] += nx; cnt++;
    }
    q[0] = sq[0] / cnt; q[1] = sq[1] / cnt; const double l = hypot(sn[0], sn[1]); n[0] = l > 0 ? sn[0] / l : 1; n[1] = l > 0 ? sn[1] / l : 0;
}

/* ufsm coreline box CORE.json --fields ZARR --box Z Y X NZ NY NX --side S [--dist 2000] [--axis const|perz] [--frame-r 2400] --out DIR:
   a half-box fit-field directory (the zarr arrays linked, zarr_offset = the box corner, core_line / core_side for the masks) whose
   axis sits dist native voxels behind the core, on the far side of it (W grows away from the core). --axis const (default): one
   axis for the box, behind the mean nearest core point along the smoothed core normal (cl_box_frame: line fits within frame-r
   core-line voxels, 2400 = 5.8 mm); perz: per native z, behind that z's nearest core point along its raw polyline normal. Prints
   the material share of the box on that side. */
static int cl_cmd_box(int argc, char **argv) {
    const char *cp = argv[3], *zr = copt(argc, argv, "--fields", nullptr), *od = copt(argc, argv, "--out", nullptr);
    int bi = -1; for (int i = 3; i < argc; i++) if (!strcmp(argv[i], "--box")) bi = i;
    if (argc < 4 || !zr || !od || bi < 0 || bi + 6 >= argc) { fprintf(stderr, "usage: ufsm coreline box <core.json> --fields ZARR --box Z Y X NZ NY NX --side +1|-1 [--dist 2000] [--axis const|perz] [--frame-r 2400] --out DIR\n"); return 2; }
    long long bo[3]; int bn[3]; for (int d = 0; d < 3; d++) { bo[d] = atoll(argv[bi + 1 + d]); bn[d] = atoi(argv[bi + 4 + d]); }
    const int side = atoi(copt(argc, argv, "--side", "1")) < 0 ? -1 : 1; const double D = atof(copt(argc, argv, "--dist", "2000"));
    const int perz = !strcmp(copt(argc, argv, "--axis", "const"), "perz"); const double FR = atof(copt(argc, argv, "--frame-r", "2400"));
    coreline C; if (cl_load(&C, cp)) { fprintf(stderr, "coreline: cannot load %s\n", cp); return 1; }
    double um; int h; int64_t shp[3]; if (fields_meta(zr, &um, &h, shp)) { fprintf(stderr, "coreline: cannot read %s/fields.json\n", zr); return 1; }
    const double f = um / C.um;   /* native -> core-line voxels */
    mkdir(od, 0755);
    char rz[PATH_MAX], rc_[PATH_MAX]; if (!realpath(zr, rz) || !realpath(cp, rc_)) { fprintf(stderr, "coreline: bad path\n"); return 1; }
    static const char *nm[10] = {"pc", "ps", "t0", "t1", "t2", "t3", "t4", "t5", "sp", "w"};
    char p[PATH_MAX + 64], t[PATH_MAX + 64];
    for (int k = 0; k < 10; k++) { snprintf(p, sizeof p, "%s/%s", od, nm[k]); snprintf(t, sizeof t, "%s/%s", rz, nm[k]); unlink(p); if (symlink(t, p)) { fprintf(stderr, "coreline: cannot link %s\n", p); return 1; } }
    const long long org[3] = {bo[0] * h, bo[1] * h, bo[2] * h};
    const int na = bn[0] * h + 1;
    snprintf(p, sizeof p, "%s/fields.json", od); FILE *fo = fopen(p, "w"); if (!fo) { fprintf(stderr, "coreline: cannot write %s\n", p); return 1; }
    fprintf(fo, "{\"format\": \"zarr\", \"origin\": [%lld, %lld, %lld], \"shape\": [%d, %d, %d], \"h\": %d, \"um\": %.4f, \"zarr_offset\": [%lld, %lld, %lld],\n \"core_line\": \"%s\", \"core_side\": %d, \"core_dist\": %.1f, \"core_axis\": \"%s\",\n \"axis\": [",
            org[0], org[1], org[2], bn[0], bn[1], bn[2], h, um, bo[0], bo[1], bo[2], rc_, side, D, perz ? "perz" : "const");
    const double cyg = org[1] + bn[1] * h / 2.0, cxg = org[2] + bn[2] * h / 2.0;   /* the box centre (global native) */
    double a0[2] = {0, 0}, a1[2] = {0, 0}, fq[2] = {0, 0}, fn[2] = {1, 0};
    if (!perz) cl_box_frame(&C, org, na, cyg, cxg, f, FR, fq, fn);
    for (int z = 0; z < na; z++) {
        double q[2], n[2];
        if (perz) cl_sdist(&C, (org[0] + z) * f, cyg * f, cxg * f, q, n); else { q[0] = fq[0]; q[1] = fq[1]; n[0] = fn[0]; n[1] = fn[1]; }
        const double ay = q[0] / f - side * D * n[0] - org[1], ax = q[1] / f - side * D * n[1] - org[2];
        fprintf(fo, "%s[%.2f, %.2f]", z ? ", " : "", ay, ax);
        if (z == 0) { a0[0] = ay; a0[1] = ax; } a1[0] = ay; a1[1] = ax;
    }
    fprintf(fo, "]}\n"); if (fclose(fo)) return 1;
    /* the material on this side (every 4th row) */
    z3 *zw = z3_open(store_open(rz), "w", nullptr); if (!zw) { fprintf(stderr, "coreline: %s\n", z3_error()); return 1; }
    uint8_t *row = malloc((size_t)bn[1] * bn[2]); size_t all = 0, kept = 0, cells = 0;
    for (int z = 0; z < bn[0]; z += 4) {
        const int64_t o[3] = {bo[0] + z, bo[1], bo[2]}, n[3] = {1, bn[1], bn[2]};
        if (z3_read(zw, o, n, row, 8)) { fprintf(stderr, "coreline: %s\n", z3_error()); return 1; }
        size_t a = 0; for (size_t i = 0; i < (size_t)bn[1] * bn[2]; i++) a += row[i] != 0;
        all += a; cells += (size_t)bn[1] * bn[2];
        a -= cl_mask_block(&C, side, org, h, um, z, 1, bn[1], bn[2], nullptr, nullptr, row); kept += a;
    }
    free(row); z3_close(zw);
    printf("coreline box: side %+d: material %.2f%% of the box (both sides %.2f%%); axis (box-local native) z0 (y %.0f, x %.0f) .. z1 (y %.0f, x %.0f)\n",
           side, 100.0 * kept / cells, 100.0 * all / cells, a0[0], a0[1], a1[0], a1[1]);
    cl_free(&C);
    return 0;
}

/* ufsm coreline sdist CORE.json Z Y X: signed distance, nearest point and normal (core-line voxels) */
int cmd_coreline(int argc, char **argv) {
    const char *sub = argc > 2 ? argv[2] : "";
    if (!strcmp(sub, "detect")) return cl_cmd_detect(argc, argv);
    if (!strcmp(sub, "box")) return cl_cmd_box(argc, argv);
    if (!strcmp(sub, "sdist") && argc >= 7) {
        coreline C; if (cl_load(&C, argv[3])) { fprintf(stderr, "coreline: cannot load %s\n", argv[3]); return 1; }
        double q[2], n[2]; const double d = cl_sdist(&C, atof(argv[4]), atof(argv[5]), atof(argv[6]), q, n);
        printf("sdist %.2f nearest (y %.2f, x %.2f) normal (%.4f, %.4f)\n", d, q[0], q[1], n[0], n[1]); cl_free(&C); return 0;
    }
    fprintf(stderr, "usage: ufsm coreline detect|box|sdist|check ...\n"
                    "  detect <fit-field zarr> <out.json> [--slab 64] [--bin 4]   the flattened umbilicus per z-slab from the phase gradient\n"
                    "  box <core.json> --fields ZARR --box Z Y X NZ NY NX --side S [--dist 2000] [--axis const|perz] --out DIR   a half-box fit-field dir\n"
                    "  sdist <core.json> Z Y X   signed distance to the core line\n"
                    "  check <core.json> <scratch-dir> <out-dir> ZCELL... [--n 2573]   core-line sanity renders (check.sh's ct / pc / ps / w files)\n");
    return 2;
}
