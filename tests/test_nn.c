/* CPU references + finite-difference gradient checks for every op in nn.h, on tiny tensors. */
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(cond, ...) do { if (!(cond)) { fails++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

static float frand(void) { return (float)rand() / RAND_MAX * 2.f - 1.f; }
static float *randv(size_t n, float s) { float *p = malloc(n * sizeof *p); for (size_t i = 0; i < n; i++) p[i] = s * frand(); return p; }
static float *dev(const float *h, size_t n) { float *d = nn_malloc(n * 4); nn_h2d(d, h, n * 4); return d; }
static void host(float *h, const float *d, size_t n) { nn_d2h(h, d, n * 4); }
static double maxdiff(const float *a, const float *b, size_t n) { double m = 0; for (size_t i = 0; i < n; i++) { double d = fabs(a[i] - b[i]); if (d > m) m = d; } return m; }
static double maxabs(const float *a, size_t n) { double m = 0; for (size_t i = 0; i < n; i++) if (fabs(a[i]) > m) m = fabs(a[i]); return m; }

/* ---- CPU conv ---- */
static void cpu_conv(const float *x, shape5 xs, const float *w, const float *b, int co, int k, int s, float *y, shape5 ys) {
    int pad = k / 2;
    for (int n = 0; n < xs.n; n++) for (int c = 0; c < co; c++)
        for (int oz = 0; oz < ys.d; oz++) for (int oy = 0; oy < ys.h; oy++) for (int ox = 0; ox < ys.w; ox++) {
            double acc = b ? b[c] : 0;
            for (int ci = 0; ci < xs.c; ci++) for (int kz = 0; kz < k; kz++) for (int ky = 0; ky < k; ky++) for (int kx = 0; kx < k; kx++) {
                int iz = oz * s - pad + kz, iy = oy * s - pad + ky, ix = ox * s - pad + kx;
                if (iz < 0 || iz >= xs.d || iy < 0 || iy >= xs.h || ix < 0 || ix >= xs.w) continue;
                acc += (double)x[(((size_t)n * xs.c + ci) * xs.d + iz) * xs.h * xs.w + (size_t)iy * xs.w + ix] * w[(((size_t)c * xs.c + ci) * k + kz) * k * k + ky * k + kx];
            }
            y[(((size_t)n * co + c) * ys.d + oz) * ys.h * ys.w + (size_t)oy * ys.w + ox] = (float)acc;
        }
}

/* generic finite-difference check: L(x) = sum gy * f(x); compares grad against g_gpu */
typedef void (*fwd_fn)(const float *x, float *y, void *ctx);
static double fd_check(fwd_fn f, void *ctx, float *x, size_t nx, const float *gy, size_t ny, const float *g_gpu, float eps, const char *name) {
    float *y = malloc(ny * 4);
    double worst = 0, scale = maxabs(g_gpu, nx) + 1e-6;
    for (size_t i = 0; i < nx; i++) {
        float o = x[i];
        x[i] = o + eps; f(x, y, ctx); double lp = 0; for (size_t j = 0; j < ny; j++) lp += (double)gy[j] * y[j];
        x[i] = o - eps; f(x, y, ctx); double lm = 0; for (size_t j = 0; j < ny; j++) lm += (double)gy[j] * y[j];
        x[i] = o;
        double fd = (lp - lm) / (2 * eps), err = fabs(fd - g_gpu[i]) / scale;
        if (err > worst) worst = err;
    }
    free(y);
    printf("  %-28s fd rel err %.3g\n", name, worst);
    return worst;
}

/* ---- conv test ---- */
typedef struct { shape5 xs, ys; const float *w, *b; int co, k, s; float *dx, *dy, *dw, *db; } conv_ctx;
static void conv_f(const float *x, float *y, void *vc) { conv_ctx *c = vc; nn_h2d(c->dx, x, shape_numel(c->xs) * 4); nn_conv3d_fwd(c->dx, c->xs, c->dw, c->db, c->co, c->k, c->s, c->dy); host(y, c->dy, shape_numel(c->ys)); }
static void conv_fw(const float *w, float *y, void *vc) { conv_ctx *c = vc; nn_h2d(c->dw, w, (size_t)c->co * c->xs.c * c->k * c->k * c->k * 4); nn_conv3d_fwd(c->dx, c->xs, c->dw, c->db, c->co, c->k, c->s, c->dy); host(y, c->dy, shape_numel(c->ys)); }

static void test_conv(int k, int s, shape5 xs, int co) {
    printf("conv k=%d s=%d in %dx%dx%dx%dx%d -> %d\n", k, s, xs.n, xs.c, xs.d, xs.h, xs.w, co);
    shape5 ys = nn_conv3d_out_shape(xs, co, k, s);
    size_t nx = shape_numel(xs), ny = shape_numel(ys), nw = (size_t)co * xs.c * k * k * k;
    float *x = randv(nx, 1), *w = randv(nw, 0.5f), *b = randv(co, 0.5f), *y = malloc(ny * 4), *yr = malloc(ny * 4);
    conv_ctx c = {xs, ys, w, b, co, k, s, dev(x, nx), nn_malloc(ny * 4), dev(w, nw), dev(b, co)};
    nn_conv3d_fwd(c.dx, xs, c.dw, c.db, co, k, s, c.dy);
    host(y, c.dy, ny);
    cpu_conv(x, xs, w, b, co, k, s, yr, ys);
    double d = maxdiff(y, yr, ny);
    printf("  fwd max diff %.3g\n", d);
    CHECK(d < 1e-4, "conv fwd k=%d s=%d", k, s);
    /* backward */
    float *gy = randv(ny, 1), *gx = malloc(nx * 4), *gw = calloc(nw, 4), *gb = calloc(co, 4);
    float *dgy = dev(gy, ny), *dgx = nn_malloc(nx * 4), *dgw = nn_malloc(nw * 4), *dgb = nn_malloc(co * 4), *scr = nn_malloc(nn_conv3d_scratch(xs, co, k));
    nn_zero(dgw, nw * 4); nn_zero(dgb, co * 4);
    nn_conv3d_bwd_data(dgy, ys, c.dw, xs, k, s, dgx, scr);
    nn_conv3d_bwd_weight(c.dx, xs, dgy, ys, k, s, dgw, dgb);
    host(gx, dgx, nx); host(gw, dgw, nw); host(gb, dgb, co);
    const char *e = nn_check(); CHECK(!e, "cuda: %s", e ? e : "");
    CHECK(fd_check(conv_f, &c, x, nx, gy, ny, gx, 1e-2f, "bwd data") < 2e-3, "conv bwd data k=%d s=%d", k, s);
    nn_h2d(c.dx, x, nx * 4);
    CHECK(fd_check(conv_fw, &c, w, nw, gy, ny, gw, 1e-2f, "bwd weight") < 2e-3, "conv bwd weight k=%d s=%d", k, s);
    double gbr[64] = {0};
    for (int n = 0; n < ys.n; n++) for (int cc = 0; cc < co; cc++) for (size_t i = 0; i < shape_spatial(ys); i++) gbr[cc] += gy[((size_t)n * co + cc) * shape_spatial(ys) + i];
    double bd = 0; for (int cc = 0; cc < co; cc++) bd = fmax(bd, fabs(gbr[cc] - gb[cc]));
    printf("  bias grad max diff %.3g\n", bd);
    CHECK(bd < 1e-3, "conv bias grad");
    free(x); free(w); free(b); free(y); free(yr); free(gy); free(gx); free(gw); free(gb);
    nn_free(c.dx); nn_free(c.dy); nn_free(c.dw); nn_free(c.db); nn_free(dgy); nn_free(dgx); nn_free(dgw); nn_free(dgb); nn_free(scr);
}

/* ---- groupnorm ---- */
typedef struct { shape5 s; int G; float *gamma, *beta; float *dx, *dy, *dg, *db, *dm, *dr; } gn_ctx;
static void gn_f(const float *x, float *y, void *vc) { gn_ctx *c = vc; nn_h2d(c->dx, x, shape_numel(c->s) * 4); nn_gn_fwd(c->dx, c->s, c->G, 1e-5f, c->dg, c->db, c->dy, c->dm, c->dr); host(y, c->dy, shape_numel(c->s)); }
static void gn_fg(const float *g, float *y, void *vc) { gn_ctx *c = vc; nn_h2d(c->dg, g, c->s.c * 4); nn_gn_fwd(c->dx, c->s, c->G, 1e-5f, c->dg, c->db, c->dy, c->dm, c->dr); host(y, c->dy, shape_numel(c->s)); }
static void gn_fb(const float *b, float *y, void *vc) { gn_ctx *c = vc; nn_h2d(c->db, b, c->s.c * 4); nn_gn_fwd(c->dx, c->s, c->G, 1e-5f, c->dg, c->db, c->dy, c->dm, c->dr); host(y, c->dy, shape_numel(c->s)); }

static void test_gn(void) {
    shape5 s = {2, 4, 3, 3, 3};
    int G = 2;
    size_t n = shape_numel(s), S = shape_spatial(s);
    printf("groupnorm %dx%d G=%d\n", s.n, s.c, G);
    float *x = randv(n, 1), *gamma = randv(s.c, 1), *beta = randv(s.c, 1), *y = malloc(n * 4), *yr = malloc(n * 4);
    for (int c = 0; c < s.c; c++) gamma[c] += 1.5f;
    gn_ctx c = {s, G, gamma, beta, dev(x, n), nn_malloc(n * 4), dev(gamma, s.c), dev(beta, s.c), nn_malloc(s.n * G * 4), nn_malloc(s.n * G * 4)};
    nn_gn_fwd(c.dx, s, G, 1e-5f, c.dg, c.db, c.dy, c.dm, c.dr);
    host(y, c.dy, n);
    int cpg = s.c / G;
    for (int nn_ = 0; nn_ < s.n; nn_++) for (int g = 0; g < G; g++) {
        double m = 0, v = 0; size_t len = (size_t)cpg * S;
        const float *p = x + ((size_t)nn_ * s.c + (size_t)g * cpg) * S;
        for (size_t i = 0; i < len; i++) m += p[i]; m /= len;
        for (size_t i = 0; i < len; i++) v += (p[i] - m) * (p[i] - m); v /= len;
        double r = 1 / sqrt(v + 1e-5);
        for (int cc = 0; cc < cpg; cc++) for (size_t i = 0; i < S; i++) {
            size_t k = ((size_t)nn_ * s.c + (size_t)g * cpg + cc) * S + i;
            yr[k] = (float)((x[k] - m) * r * gamma[g * cpg + cc] + beta[g * cpg + cc]);
        }
    }
    double d = maxdiff(y, yr, n);
    printf("  fwd max diff %.3g\n", d);
    CHECK(d < 1e-4, "gn fwd");
    float *gy = randv(n, 1), *gx = malloc(n * 4), *gg = calloc(s.c, 4), *gb = calloc(s.c, 4);
    float *dgy = dev(gy, n), *dgx = nn_malloc(n * 4), *dgg = nn_malloc(s.c * 4), *dgb = nn_malloc(s.c * 4), *scr = nn_malloc(nn_gn_scratch(s));
    nn_zero(dgg, s.c * 4); nn_zero(dgb, s.c * 4);
    nn_gn_bwd(c.dx, s, G, c.dg, c.dm, c.dr, dgy, dgx, dgg, dgb, scr);
    host(gx, dgx, n); host(gg, dgg, s.c); host(gb, dgb, s.c);
    CHECK(fd_check(gn_f, &c, x, n, gy, n, gx, 1e-3f, "bwd x") < 5e-3, "gn bwd x");
    nn_h2d(c.dx, x, n * 4);
    CHECK(fd_check(gn_fg, &c, gamma, s.c, gy, n, gg, 1e-3f, "bwd gamma") < 5e-3, "gn bwd gamma");
    nn_h2d(c.dg, gamma, s.c * 4);
    CHECK(fd_check(gn_fb, &c, beta, s.c, gy, n, gb, 1e-3f, "bwd beta") < 5e-3, "gn bwd beta");
}

/* ---- silu ---- */
typedef struct { size_t n; float *dx, *dy; } el_ctx;
static void silu_f(const float *x, float *y, void *vc) { el_ctx *c = vc; nn_h2d(c->dx, x, c->n * 4); nn_silu_fwd(c->dx, c->n, c->dy); host(y, c->dy, c->n); }
static void test_silu(void) {
    size_t n = 100;
    printf("silu\n");
    float *x = randv(n, 3), *gy = randv(n, 1), *gx = malloc(n * 4), *y = malloc(n * 4);
    el_ctx c = {n, dev(x, n), nn_malloc(n * 4)};
    nn_silu_fwd(c.dx, n, c.dy); host(y, c.dy, n);
    double d = 0; for (size_t i = 0; i < n; i++) d = fmax(d, fabs(y[i] - x[i] / (1 + exp(-x[i]))));
    CHECK(d < 1e-5, "silu fwd");
    float *dgy = dev(gy, n), *dgx = nn_malloc(n * 4);
    nn_silu_bwd(c.dx, dgy, n, dgx); host(gx, dgx, n);
    CHECK(fd_check(silu_f, &c, x, n, gy, n, gx, 1e-3f, "bwd") < 1e-3, "silu bwd");
}

/* ---- up2 ---- */
typedef struct { shape5 xs; float *dx, *dy; } up_ctx;
static void up_f(const float *x, float *y, void *vc) { up_ctx *c = vc; nn_h2d(c->dx, x, shape_numel(c->xs) * 4); nn_up2_fwd(c->dx, c->xs, c->dy); host(y, c->dy, shape_numel(c->xs) * 8); }
static void test_up2(void) {
    shape5 xs = {1, 2, 2, 3, 4};
    size_t nx = shape_numel(xs), ny = nx * 8;
    printf("up2 %dx%dx%dx%dx%d\n", xs.n, xs.c, xs.d, xs.h, xs.w);
    float *x = randv(nx, 1), *y = malloc(ny * 4), *yr = malloc(ny * 4);
    up_ctx c = {xs, dev(x, nx), nn_malloc(ny * 4)};
    nn_up2_fwd(c.dx, xs, c.dy); host(y, c.dy, ny);
    /* CPU reference: separable, PyTorch align_corners=False at exact 2x */
    int D = xs.d, H = xs.h, W = xs.w, Do = 2 * D, Ho = 2 * H, Wo = 2 * W;
    for (int nc = 0; nc < xs.n * xs.c; nc++) for (int oz = 0; oz < Do; oz++) for (int oy = 0; oy < Ho; oy++) for (int ox = 0; ox < Wo; ox++) {
        double acc = 0;
        int os[3] = {oz, oy, ox}, ns[3] = {D, H, W};
        int i0[3], i1[3]; double w0[3], w1[3];
        for (int a = 0; a < 3; a++) {
            double src = (os[a] + 0.5) / 2 - 0.5; if (src < 0) src = 0;
            int f = (int)floor(src); double t = src - f; if (f >= ns[a] - 1) { f = ns[a] - 1; t = 0; }
            i0[a] = f; i1[a] = f + (t > 0 ? 1 : 0); w0[a] = 1 - t; w1[a] = t;
        }
        for (int a = 0; a < 2; a++) for (int b2 = 0; b2 < 2; b2++) for (int cc = 0; cc < 2; cc++) { /* 8 corners */
            int z = a ? i1[0] : i0[0], yy = b2 ? i1[1] : i0[1], xx = cc ? i1[2] : i0[2];
            double wt = (a ? w1[0] : w0[0]) * (b2 ? w1[1] : w0[1]) * (cc ? w1[2] : w0[2]);
            acc += wt * x[(((size_t)nc * D + z) * H + yy) * W + xx];
        }
        yr[(((size_t)nc * Do + oz) * Ho + oy) * Wo + ox] = (float)acc;
    }
    double d = maxdiff(y, yr, ny);
    printf("  fwd max diff %.3g\n", d);
    CHECK(d < 1e-5, "up2 fwd");
    float *gy = randv(ny, 1), *gx = malloc(nx * 4), *dgy = dev(gy, ny), *dgx = nn_malloc(nx * 4);
    nn_up2_bwd(dgy, xs, dgx); host(gx, dgx, nx);
    CHECK(fd_check(up_f, &c, x, nx, gy, ny, gx, 1e-2f, "bwd") < 1e-3, "up2 bwd");
}

/* ---- concat ---- */
static void test_concat(void) {
    shape5 s = {2, 0, 2, 2, 3};
    int ca = 2, cb = 3; size_t S = shape_spatial(s), na = (size_t)s.n * ca * S, nb = (size_t)s.n * cb * S;
    printf("concat\n");
    float *a = randv(na, 1), *b = randv(nb, 1), *y = malloc((na + nb) * 4);
    float *da = dev(a, na), *db = dev(b, nb), *dy = nn_malloc((na + nb) * 4);
    nn_concat_fwd(da, ca, db, cb, s, dy); host(y, dy, na + nb);
    double d = 0;
    for (int n = 0; n < s.n; n++) for (int c = 0; c < ca + cb; c++) for (size_t i = 0; i < S; i++) {
        float ref = c < ca ? a[((size_t)n * ca + c) * S + i] : b[((size_t)n * cb + c - ca) * S + i];
        d = fmax(d, fabs(ref - y[((size_t)n * (ca + cb) + c) * S + i]));
    }
    CHECK(d == 0, "concat fwd");
    float *ga = malloc(na * 4), *gb = malloc(nb * 4);
    nn_concat_bwd(dy, ca, cb, s, da, db); host(ga, da, na); host(gb, db, nb);
    CHECK(maxdiff(ga, a, na) == 0 && maxdiff(gb, b, nb) == 0, "concat bwd");
}

/* ---- loss ---- */
typedef struct { shape5 s; uint8_t *t, *m, *w; float *dl, *dg, *scr; uint8_t *dt, *dm, *dw; } loss_ctx;
static double cpu_loss(const float *l, const loss_ctx *c) {
    size_t S = shape_spatial(c->s); int C = c->s.c; double tot = 0; int active = 0;
    for (int nc = 0; nc < c->s.n * C; nc++) {
        if (!c->w[nc]) continue;
        int n = nc / C; double nm = 0, bce = 0, sp = 0, ss = 0, spp = 0;
        for (size_t i = 0; i < S; i++) {
            if (!c->m[(size_t)n * S + i]) continue;
            double x = l[(size_t)nc * S + i], p = c->t[(size_t)nc * S + i] / 255.0, s = 1 / (1 + exp(-x));
            nm++; bce += fmax(x, 0) - x * p + log1p(exp(-fabs(x))); sp += s * p; ss += s; spp += p;
        }
        if (nm < 1) continue;
        active++;
        tot += bce / nm + 0.5 * (1 - (2 * sp + 1) / (ss + spp + 1));
    }
    return active ? tot / active : 0;
}
static void test_loss(void) {
    shape5 s = {2, 2, 3, 3, 3};
    size_t S = shape_spatial(s), n = shape_numel(s);
    printf("loss\n");
    loss_ctx c; c.s = s;
    float *l = randv(n, 2);
    c.t = malloc(n); c.m = malloc((size_t)s.n * S); c.w = malloc((size_t)s.n * s.c);
    for (size_t i = 0; i < n; i++) c.t[i] = (uint8_t)(rand() % 256);
    for (size_t i = 0; i < (size_t)s.n * S; i++) c.m[i] = rand() % 4 != 0;
    c.w[0] = 1; c.w[1] = 0; c.w[2] = 1; c.w[3] = 1;
    c.dl = dev(l, n); c.dg = nn_malloc(n * 4); c.scr = nn_malloc(nn_loss_scratch(s));
    c.dt = nn_malloc(n); nn_h2d(c.dt, c.t, n); c.dm = nn_malloc((size_t)s.n * S); nn_h2d(c.dm, c.m, (size_t)s.n * S); c.dw = nn_malloc(4); nn_h2d(c.dw, c.w, 4);
    float out[8];
    nn_loss(c.dl, c.dt, c.dm, c.dw, s, 0.5f, c.dg, out, c.scr);
    double ref = cpu_loss(l, &c), got = 0;
    /* reconstruct the scalar the same way: mean over active of (bce + 0.5 dice) — recompute from per-channel means */
    got = ((out[0] + 0.5 * out[2]) * 2 + (out[1] + 0.5 * out[3]) * 1) / out[4];   /* w = [1,0,1,1]: channel 0 active twice, channel 1 once */
    printf("  loss cpu %.6f gpu %.6f (active %g)\n", ref, got, out[4]);
    CHECK(fabs(ref - got) < 1e-4, "loss value");
    float *g = malloc(n * 4); host(g, c.dg, n);
    double worst = 0, scale = maxabs(g, n);
    for (size_t i = 0; i < n; i++) {
        float o = l[i]; l[i] = o + 1e-3f; double lp = cpu_loss(l, &c); l[i] = o - 1e-3f; double lm = cpu_loss(l, &c); l[i] = o;
        double fd = (lp - lm) / 2e-3, err = fabs(fd - g[i]) / (scale + 1e-9); if (err > worst) worst = err;
    }
    printf("  grad fd rel err %.3g\n", worst);
    CHECK(worst < 2e-3, "loss grad");
}

static void test_adamw(void) {
    printf("adamw + reductions\n");
    size_t n = 1000;
    float *p = randv(n, 1), *g = randv(n, 1), *m = calloc(n, 4), *v = calloc(n, 4);
    float *dp = dev(p, n), *dg = dev(g, n), *dm = dev(m, n), *dv = dev(v, n), *scr = nn_malloc(4096 * 4);
    double s1 = nn_sum(dp, n, scr), s2 = 0; for (size_t i = 0; i < n; i++) s2 += p[i];
    CHECK(fabs(s1 - s2) < 1e-3, "sum %g vs %g", s1, s2);
    nn_adamw(dp, dg, dm, dv, n, 1e-2f, 0.9f, 0.999f, 1e-8f, 0.01f, 1);
    float *q = malloc(n * 4); host(q, dp, n);
    double d = 0;
    for (size_t i = 0; i < n; i++) {
        double mi = 0.1 * g[i], vi = 0.001 * g[i] * g[i], mh = mi / 0.1, vh = vi / 0.001;
        double ref = p[i] - 1e-2 * (mh / (sqrt(vh) + 1e-8) + 0.01 * p[i]);
        d = fmax(d, fabs(ref - q[i]));
    }
    printf("  adamw max diff %.3g\n", d);
    CHECK(d < 1e-5, "adamw");
}

int main(void) {
    if (nn_init(0)) { printf("no cuda device\n"); return 1; }
    nn_set_tf32(0);   /* finite-difference checks need the exact fp32 kernels */
    srand(1);
    test_conv(3, 1, (shape5){2, 3, 5, 6, 7}, 4);
    test_conv(3, 2, (shape5){2, 3, 6, 7, 8}, 5);
    test_conv(3, 2, (shape5){1, 2, 5, 5, 5}, 3);
    test_conv(1, 1, (shape5){2, 5, 3, 4, 5}, 2);
    test_conv(3, 1, (shape5){1, 17, 3, 4, 5}, 9);
    test_gn();
    test_silu();
    test_up2();
    test_concat();
    test_loss();
    test_adamw();
    const char *e = nn_check();
    CHECK(!e, "cuda error: %s", e ? e : "");
    {   /* 2:4 mask: every group of 4 input channels keeps exactly its two largest magnitudes */
        int co = 3, ci = 8, T = 27; size_t nw = (size_t)co * ci * T; float *h = malloc(nw * 4); for (size_t i = 0; i < nw; i++) h[i] = (float)rand() / RAND_MAX - 0.5f;
        float *dw = nn_malloc(nw * 4), *dm = nn_malloc(nw * 4); nn_h2d(dw, h, nw * 4); nn_mask24(dw, dm, co, ci, T); float *m = malloc(nw * 4); nn_d2h(m, dm, nw * 4);
        int bad = 0;
        for (int c = 0; c < co; c++) for (int g4 = 0; g4 < ci / 4; g4++) for (int t = 0; t < T; t++) {
            size_t base = ((size_t)c * ci + 4 * g4) * T + t; int kept = 0; float mn_kept = 1e9f, mx_drop = 0;
            for (int j = 0; j < 4; j++) { float v = m[base + (size_t)j * T]; if (v != 0) { kept++; if (v != h[base + (size_t)j * T]) bad++; if (fabsf(v) < mn_kept) mn_kept = fabsf(v); } else if (fabsf(h[base + (size_t)j * T]) > mx_drop) mx_drop = fabsf(h[base + (size_t)j * T]); }
            if (kept != 2 || mx_drop > mn_kept) bad++;
        }
        printf("mask24 %s\n", bad ? "FAIL" : "ok"); if (bad) fails++;
        free(h); free(m); nn_free(dw); nn_free(dm);
    }
    printf(fails ? "%d FAILURES\n" : "nn ok\n", fails);
    return fails != 0;
}
