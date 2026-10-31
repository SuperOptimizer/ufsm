/* src/flatten.c: SLIM flattening on synthetic developable sheets with a known exact development (cylinder, cone): from a strongly
   non-isometric grid map the result must be isometric (sigma ~ 1, no flips, area kept), reproduce the development's distances, and its
   isometric resample must have unit, orthogonal 3D vertex spacing; a sphere cap (not developable) must stay flip-free and improve;
   the largest valid rectangle on a hand-made mask. */
#include "flatten.h"
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
static int g_gpu = 0;   /* test_flatten_gpu (-DFLAT_GPU, linked with nn_flat.o): the SLIM iterations on the GPU */
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

/* a w x h grid, parameters t (columns), s (rows) in [0, 1] warped non-uniformly, point = f(t, s) */
typedef void (*surf_fn)(double t, double s, double p[3]);
static double R_cyl = 60, A_cone = 0.5;
static void cyl(double t, double s, double p[3]) { const double th = 1.5 * t; p[0] = R_cyl * cos(th); p[1] = R_cyl * sin(th); p[2] = 60 * s; }
static void cone(double t, double s, double p[3]) { const double ph = 1.2 * t, rho = 40 + 60 * s; p[0] = rho * sin(A_cone) * cos(ph); p[1] = rho * sin(A_cone) * sin(ph); p[2] = rho * cos(A_cone); }
static void sph(double t, double s, double p[3]) { const double th = 0.8 * (t - 0.5), ph = 0.8 * (s - 0.5), R = 50; p[0] = R * cos(ph) * cos(th); p[1] = R * cos(ph) * sin(th); p[2] = R * sin(ph); }
static void make(surf_fn f, int w, int h, float *X, float *Y, float *Z, int holes) {
    for (int r = 0; r < h; r++) for (int c = 0; c < w; c++) {
        const double t0 = (double)c / (w - 1), s0 = (double)r / (h - 1);
        const double t = t0 + 0.12 * sin(2 * M_PI * t0) * (0.6 + 0.4 * s0), s = s0 + 0.08 * sin(2 * M_PI * s0);   /* sheared, stretched grid map */
        double p[3]; f(t, s, p); const size_t i = (size_t)r * w + c;
        X[i] = (float)p[0]; Y[i] = (float)p[1]; Z[i] = (float)p[2] + 100;   /* z > 0: valid */
        if (holes && ((r - h / 2) * (r - h / 2) + (c - w / 3) * (c - w / 3) < 25 || (r < 6 && c > w - 10))) X[i] = Y[i] = Z[i] = -1;
    }
}
static double uvdist(const flat_mesh *m, int w, int r0, int c0, int r1, int c1) {
    int a = -1, b = -1;
    for (int v = 0; v < m->nv; v++) { if (m->gi[v] == r0 * w + c0) a = v; if (m->gi[v] == r1 * w + c1) b = v; }
    if (a < 0 || b < 0) return -1;
    return hypot(m->uv[2 * a] - m->uv[2 * b], m->uv[2 * a + 1] - m->uv[2 * b + 1]);
}

static void run(const char *name, surf_fn f, int holes, int developable) {
    const int w = 121, h = 101; const size_t n = (size_t)w * h;
    float *X = malloc(n * 4), *Y = malloc(n * 4), *Z = malloc(n * 4); make(f, w, h, X, Y, Z, holes);
    flat_opts o = flat_opts_default(); o.min_verts = 800; o.nthreads = 4; o.iters = 60; o.cg_iters = 400; o.tol = 1e-9; o.gpu = g_gpu;
    flat_mesh m; flat_stats b, a;
    CHECK(!flat_grid(X, Y, Z, w, h, 1.0, &o, &m, &b, &a), "%s: flat_grid failed", name);
    char lb[64]; snprintf(lb, sizeof lb, "%s before", name); flat_stats_print(stdout, lb, &b); snprintf(lb, sizeof lb, "%s after", name); flat_stats_print(stdout, lb, &a);
    CHECK(b.aniso12 > 0.2, "%s: the test grid map should be distorted (aniso>1.2 %.3f)", name, b.aniso12);
    CHECK(a.flips == 0, "%s: %lld flips", name, a.flips);
    if (developable) {
        CHECK(a.energy < 4.002, "%s: energy %.5f (isometric 4)", name, a.energy);
        CHECK(a.aniso_p99 < 1.02 && a.smax_p95 < 1.01 && a.smin_p05 > 0.99, "%s: sigma not ~1 (aniso p99 %.4f smax p95 %.4f smin p05 %.4f)", name, a.aniso_p99, a.smax_p95, a.smin_p05);
        CHECK(fabs(a.area_ratio - 1) < 0.002, "%s: area ratio %.5f", name, a.area_ratio);
        /* distances against the exact development (vertices at grid corners: t, s warps vanish at 0 and 1) */
        if (f == cyl) {
            const double d = uvdist(&m, w, h - 1, 0, h - 1, w - 1), ex = 2 * R_cyl * sin(1.5 / 2) * (1.5 * R_cyl) / (2 * R_cyl * sin(1.5 / 2));   /* arc 90 */
            CHECK(d > 0 && fabs(d - ex) / ex < 0.005, "%s: developed width %.3f, exact %.3f", name, d, ex);
            const double dg = uvdist(&m, w, 0, 0, h - 1, w - 1), eg = hypot(1.5 * R_cyl, 60);
            CHECK(dg > 0 && fabs(dg - eg) / eg < 0.005, "%s: developed diagonal %.3f, exact %.3f", name, dg, eg);
        } else if (f == cone) {
            const double d = uvdist(&m, w, h - 1, 0, h - 1, w - 1), ex = 2 * 100 * sin(1.2 * sin(A_cone) / 2);   /* outer arc's chord in the development */
            CHECK(d > 0 && fabs(d - ex) / ex < 0.005, "%s: developed outer chord %.3f, exact %.3f", name, d, ex);
            const double di = uvdist(&m, w, 0, 0, 0, w - 1), ei = 2 * 40 * sin(1.2 * sin(A_cone) / 2);
            CHECK(di > 0 && fabs(di - ei) / ei < 0.005, "%s: developed inner chord %.3f, exact %.3f", name, di, ei);
        }
        /* the isometric resample: unit, orthogonal 3D spacing between neighbouring output vertices */
        int ow, oh; float *gu, *gv, *ox, *oy, *oz;
        CHECK(!flat_resample(&m, X, Y, Z, w, h, 1.0, 4, &ow, &oh, &gu, &gv, &ox, &oy, &oz), "%s: resample", name);
        uint8_t *vm = malloc((size_t)ow * oh), *fm = malloc((size_t)ow * oh); float *ga[5] = {ox, oy, oz, gu, gv};
        const long nf = flat_fill_holes(ga, ow, oh, 5000, fm);
        CHECK(holes ? nf > 5 : nf == 0, "%s: %ld hole vertices filled", name, nf);
        for (size_t i = 0; i < (size_t)ow * oh; i++) vm[i] = oz[i] >= 0;
        int cx, cy, cw, ch; flat_largest_rect(vm, ow, oh, &cx, &cy, &cw, &ch);
        double su = 0, sv = 0, cmax = 0, emax = 0; long k = 0;
        for (int y = cy; y + 1 < cy + ch; y++) for (int x = cx; x + 1 < cx + cw; x++) {
            const size_t i = (size_t)y * ow + x, j = i + 1, l = i + ow;
            const double du[3] = {ox[j] - ox[i], oy[j] - oy[i], oz[j] - oz[i]}, dv[3] = {ox[l] - ox[i], oy[l] - oy[i], oz[l] - oz[i]};
            const double lu = sqrt(du[0] * du[0] + du[1] * du[1] + du[2] * du[2]), lv = sqrt(dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]);
            su += lu; sv += lv; k++; const double cs = fabs(du[0] * dv[0] + du[1] * dv[1] + du[2] * dv[2]) / (lu * lv); if (cs > cmax) cmax = cs;
            emax = fmax(emax, fmax(fabs(lu - 1), fabs(lv - 1)));
        }
        printf("%s resample: %dx%d, rect %d,%d %dx%d, mean spacing %.4f / %.4f, max |spacing - 1| %.4f, max |cos| %.4f\n", name, ow, oh, cx, cy, cw, ch, su / k, sv / k, emax, cmax);
        CHECK(cw * ch > (f == cone ? 0.35 : 0.8) * ow * oh, "%s: rectangle (a cone develops to an annular sector) %dx%d of %dx%d", name, cw, ch, ow, oh);
        CHECK(fabs(su / k - 1) < 0.005 && fabs(sv / k - 1) < 0.005 && emax < 0.06 && cmax < 0.06, "%s: resample not isometric", name);
        free(vm); free(fm); free(gu); free(gv); free(ox); free(oy); free(oz);
    } else {
        CHECK(a.energy < b.energy, "%s: energy %.4f not below %.4f", name, a.energy, b.energy);
        CHECK(a.aniso12 < 0.5 * b.aniso12, "%s: aniso>1.2 %.4f vs %.4f", name, a.aniso12, b.aniso12);
    }
    flat_mesh_free(&m); free(X); free(Y); free(Z);
}

#ifdef FLAT_GPU
int flat_slim_gpu(const double *g, const double *A, const int *tri, int nt, const int *vo, const int *vl, int nv, double *x,
                  int iters, int cg_iters, double tol, int verbose, const char *tag, int dev, double *Eout);
typedef struct { const float *X, *Y, *Z; int w, h; flat_mesh m; int rc; } cg_job;
static void *cg_run(void *a) {
    cg_job *J = a; flat_opts o = flat_opts_default(); o.min_verts = 800; o.nthreads = 2; o.iters = 60; o.cg_iters = 400; o.tol = 1e-9; o.gpu = 1; o.gpu_dev = 0;
    flat_stats b, s; J->rc = flat_grid(J->X, J->Y, J->Z, J->w, J->h, 1.0, &o, &J->m, &b, &s); return nullptr;
}
/* GPU vs CPU on one sheet: the same solution (the summation order differs: uv within 1e-4 of a grid step, energy 1e-7), and two GPU
   runs bit-identical */
static void cmp_gpu(const char *name, surf_fn f, int holes) {
    const int w = 121, h = 101; const size_t n = (size_t)w * h;
    float *X = malloc(n * 4), *Y = malloc(n * 4), *Z = malloc(n * 4); make(f, w, h, X, Y, Z, holes);
    flat_mesh m[3]; flat_stats b, a[3];
    for (int k = 0; k < 3; k++) {
        flat_opts o = flat_opts_default(); o.min_verts = 800; o.nthreads = 4; o.iters = 60; o.cg_iters = 400; o.tol = 1e-9; o.gpu = k > 0;
        CHECK(!flat_grid(X, Y, Z, w, h, 1.0, &o, &m[k], &b, &a[k]), "%s: flat_grid %d failed", name, k);
    }
    CHECK(m[0].nv == m[1].nv && m[1].nv == m[2].nv && m[0].nt == m[1].nt, "%s: meshes differ", name);
    double dmax = 0; for (int i = 0; i < 2 * m[0].nv; i++) dmax = fmax(dmax, fabs(m[0].uv[i] - m[1].uv[i]));
    printf("%s: GPU vs CPU max |duv| %.2e, energy %.9f vs %.9f\n", name, dmax, a[1].energy, a[0].energy);
    CHECK(dmax < 1e-4 && fabs(a[0].energy - a[1].energy) < 1e-7, "%s: GPU solution differs (max |duv| %.3g, energy %.9f vs %.9f)", name, dmax, a[1].energy, a[0].energy);
    CHECK(!memcmp(m[1].uv, m[2].uv, (size_t)m[1].nv * 16), "%s: two GPU runs differ", name);
    {   /* two host threads on one device at once (strip-flatten --jobs; per-thread default streams): the same bits */
        cg_job J[2]; pthread_t th[2];
        for (int k = 0; k < 2; k++) { J[k] = (cg_job){X, Y, Z, w, h, {0}, -1}; pthread_create(&th[k], nullptr, cg_run, &J[k]); }
        for (int k = 0; k < 2; k++) { pthread_join(th[k], nullptr);
            CHECK(!J[k].rc && J[k].m.nv == m[1].nv && !memcmp(J[k].m.uv, m[1].uv, (size_t)m[1].nv * 16), "%s: concurrent GPU run %d differs", name, k);
            if (!J[k].rc) flat_mesh_free(&J[k].m); }
    }
    for (int k = 0; k < 3; k++) flat_mesh_free(&m[k]);
    free(X); free(Y); free(Z);
}
#endif

int main(void) {
#ifdef FLAT_GPU
    {   /* a device? (one triangle) */
        const double g[6] = {-1, -1, 1, 0, 0, 1}, A[1] = {0.5}; const int tri[3] = {0, 1, 2}, vo[4] = {0, 1, 2, 3}, vl[3] = {0, 1, 2}; double x[6] = {0, 0, 1, 0, 0, 1}, E;
        if (flat_slim_gpu(g, A, tri, 1, vo, vl, 3, x, 1, 5, 1e-9, 0, "probe", -1, &E)) { printf("test_flatten_gpu: no GPU, skipped\n"); return 0; }
    }
    g_gpu = 1;
    cmp_gpu("cylinder", cyl, 0); cmp_gpu("cone-holes", cone, 1); cmp_gpu("sphere-cap", sph, 0);
#endif
    run("cylinder", cyl, 0, 1);
    run("cone", cone, 0, 1);
    run("cylinder-holes", cyl, 1, 1);
    run("sphere-cap", sph, 0, 0);
    { /* largest rectangle against brute force on random masks */
        unsigned st = 12345;
        for (int rep = 0; rep < 200; rep++) {
            const int w = 3 + rep % 11, h = 2 + rep % 7; uint8_t m[13 * 8];
            for (int i = 0; i < w * h; i++) { st = st * 1103515245u + 12345u; m[i] = (st >> 16) % 5 != 0; }
            int best = 0;
            for (int y0 = 0; y0 < h; y0++) for (int y1 = y0; y1 < h; y1++) for (int x0 = 0; x0 < w; x0++) for (int x1 = x0; x1 < w; x1++) {
                int ok = 1; for (int y = y0; y <= y1 && ok; y++) for (int x = x0; x <= x1 && ok; x++) ok = m[y * w + x];
                if (ok && (y1 - y0 + 1) * (x1 - x0 + 1) > best) best = (y1 - y0 + 1) * (x1 - x0 + 1);
            }
            int x0, y0, rw, rh; flat_largest_rect(m, w, h, &x0, &y0, &rw, &rh);
            CHECK(rw * rh == best, "largest rect rep %d: %dx%d, brute force area %d", rep, rw, rh, best);
            for (int y = y0; y < y0 + rh; y++) for (int x = x0; x < x0 + rw; x++) CHECK(m[y * w + x], "rect cell %d,%d invalid", x, y);
            for (int ai = 0; ai < 2; ai++) {   /* aspect-capped (flatten --max-aspect) against brute force */
                const double a = ai ? 1.5 : 2; int ba = 0;
                for (int y0b = 0; y0b < h; y0b++) for (int y1 = y0b; y1 < h; y1++) for (int x0b = 0; x0b < w; x0b++) for (int x1 = x0b; x1 < w; x1++) {
                    const int rw2 = x1 - x0b + 1, rh2 = y1 - y0b + 1; if (rw2 > a * rh2 || rh2 > a * rw2) continue;
                    int ok = 1; for (int y = y0b; y <= y1 && ok; y++) for (int x = x0b; x <= x1 && ok; x++) ok = m[y * w + x];
                    if (ok && rw2 * rh2 > ba) ba = rw2 * rh2;
                }
                int ax, ay, aw, ah; flat_largest_rect_aspect(m, w, h, a, &ax, &ay, &aw, &ah);
                CHECK(aw * ah == ba && aw <= a * ah && ah <= a * aw, "aspect %.1f rect rep %d: %dx%d, brute force area %d", a, rep, aw, ah, ba);
                for (int y = ay; y < ay + ah; y++) for (int x = ax; x < ax + aw; x++) CHECK(m[y * w + x], "aspect rect cell %d,%d invalid", x, y);
            }
        }
    }
#ifdef FLAT_GPU
    printf(fails ? "test_flatten_gpu: %d FAILED\n" : "test_flatten_gpu: all passed\n", fails);
#else
    printf(fails ? "test_flatten: %d FAILED\n" : "test_flatten: all passed\n", fails);
#endif
    return fails ? 1 : 0;
}
