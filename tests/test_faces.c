/* ufsm faces: papyrus bands of synthetic CT profiles, and recto (inner) / verso (outer) faces of a synthetic two-sheet cylinder found
   from a wrap sitting in the gap inside the first sheet, whichever way the grid normal points. */
#include "faces.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)

static double S(double u) { return 0.5 * (1 + tanh(u / 0.7)); }
/* sheets [a0, a1] and [b0, b1] (radial, voxels), CT 150 on papyrus over 20 void */
typedef struct { double cy, cx, a0, a1, b0, b1; } cyl;
static double cyl_ct(const cyl *C, double r) { return 20 + 130 * (S(r - C->a0) - S(r - C->a1)) + 130 * (S(r - C->b0) - S(r - C->b1)); }
static double cyl_sample(void *ctx, double z, double y, double x) { const cyl *C = ctx; (void)z; return cyl_ct(C, hypot(y - C->cy, x - C->cx)); }

#ifdef FACES_GPU
/* 3. the GPU path (src/nn_faces.cu) against the CPU path on a chunked uint8 volume (noisy two-sheet cylinder): bit-identical outputs */
#include "faces_core.h"
#include <string.h>
enum { CS = 128 };
typedef struct { int64_t shape[3]; int g[3]; uint8_t **ch; fgpu *G; int put; } tvol;
static double tv_sample(void *ctx, double z, double y, double x) {   /* faces_cli fcc_sample */
    const tvol *T = ctx; const int64_t a = (int64_t)floor(z), b = (int64_t)floor(y), e = (int64_t)floor(x); int v[8];
    for (int k = 0; k < 8; k++) {
        const int64_t zz = a + (k >> 2 & 1), yy = b + (k >> 1 & 1), xx = e + (k & 1);
        if (zz < 0 || yy < 0 || xx < 0 || zz >= T->shape[0] || yy >= T->shape[1] || xx >= T->shape[2]) return NAN;
        v[k] = T->ch[((zz / CS) * T->g[1] + yy / CS) * T->g[2] + xx / CS][((zz % CS) * CS + yy % CS) * CS + xx % CS];
    }
    return fc_trilin(v, z - a, y - b, x - e);
}
static void tv_mark(void *ctx, const double lo[3], const double hi[3]) { (void)ctx; (void)lo; (void)hi; }
static int tv_fill(void *ctx) {   /* every chunk once, in two halves of substreams (the partial-upload path) */
    tvol *T = ctx; if (T->put || !T->G) return 0;
    for (int k = 0; k < T->g[0] * T->g[1] * T->g[2]; k++) { fgpu_put(T->G, k, T->ch[k], 0x0000ffffu); fgpu_put(T->G, k, T->ch[k], 0xffffffffu); }
    T->put = 1; return 0;
}
static int gpu_vs_cpu(void) {
    tvol T = {{128, 384, 384}, {1, 3, 3}, nullptr, nullptr, 0}; const int nch = 9;
    T.ch = malloc(nch * sizeof *T.ch);
    cyl C = {200, 90, 100, 105, 109, 113};
    for (int k = 0; k < nch; k++) {
        T.ch[k] = malloc((size_t)CS * CS * CS); const int cy = k / 3 % 3, cx = k % 3;
        for (int z = 0; z < CS; z++) for (int y = 0; y < CS; y++) for (int x = 0; x < CS; x++) {
            const double Y = cy * CS + y, X = cx * CS + x; uint32_t hsh = (uint32_t)(z * 73856093u ^ (uint32_t)Y * 19349663u ^ (uint32_t)X * 83492791u); hsh ^= hsh >> 13; hsh *= 0x5bd1e995u; hsh ^= hsh >> 15;
            double v = cyl_ct(&C, hypot(Y - C.cy, X - C.cx)) + 0.2 * z + (double)(hsh % 61) - 30; v = v < 0 ? 0 : v > 255 ? 255 : v;
            T.ch[k][((size_t)z * CS + y) * CS + x] = (uint8_t)v;
        }
    }
    const int w = 300, h = 70; const size_t n = (size_t)w * h;   /* an arc reaching past the volume's x / y edges: NaN profiles */
    float *X = malloc(n * 4), *Y = malloc(n * 4), *Z = malloc(n * 4);
    for (int r = 0; r < h; r++) for (int c = 0; c < w; c++) {
        const size_t i = (size_t)r * w + c; const double a = -0.3 + c * 0.012, rad = 98 + 1.5 * sin(0.07 * c + 0.13 * r);
        X[i] = (float)(C.cx + rad * cos(a)); Y[i] = (float)(C.cy + rad * sin(a)); Z[i] = (float)(10 + 1.5 * r);
    }
    for (int k = 0; k < 40; k++) { const size_t i = (size_t)(k * 7919) % n; X[i] = Y[i] = Z[i] = -1; }   /* holes */
    float *out[2][7]; uint8_t *cf[2]; faces_stats st[2]; int rc = 0;
    T.G = fgpu_open(T.shape);
    if (!T.G) { printf("test_faces_gpu: no CUDA device, GPU comparison skipped\n"); return 0; }
    for (int g = 0; g < 2; g++) {
        for (int k = 0; k < 7; k++) out[g][k] = malloc(n * 4);
        cf[g] = malloc(n); memset(&st[g], 0, sizeof st[g]); st[g].ct_thr = 90;
        faces_opt o = {.range = 12, .tmin = 2, .tmax = 12, .min_prom = 8, .rel = 0.1, .tol = 2.5, .inset = 0, .smooth = 4, .nthreads = 4};
        faces_vol V = {&T, tv_sample, tv_mark, tv_fill, g ? T.G : nullptr};
        const int sign = faces_orient(X, Y, Z, w, h, nullptr, 0);
        rc |= faces_wrap(X, Y, Z, w, h, 1.0, 9.6, sign, &o, &V, out[g][0], out[g][1], out[g][2], out[g][3], out[g][4], out[g][5], cf[g], out[g][6], &st[g]);
    }
    CHECK(!rc, "faces_wrap");
    CHECK(!fgpu_broken(T.G), "GPU path fell back to the CPU");
    int diff = 0; for (int k = 0; k < 7; k++) diff += memcmp(out[0][k], out[1][k], n * 4) != 0;
    diff += memcmp(cf[0], cf[1], n) != 0; diff += memcmp(&st[0], &st[1], sizeof st[0]) != 0;
    printf("gpu vs cpu: %d of 9 outputs differ; band %zu picked %zu/%zu, filled %zu, ct_n %zu\n", diff, st[1].nband, st[1].npick, st[1].nv, st[1].nfill, st[1].ct_n);
    CHECK(diff == 0, "GPU and CPU outputs differ (%d)", diff);
    CHECK(st[0].npick > 0.5 * st[0].nv && st[0].ct_n > 100, "the comparison exercises picks (%zu/%zu) and kept profiles (%zu)", st[0].npick, st[0].nv, st[0].ct_n);
    /* the median kernel alone, over a radius the wrap does not use (generic capacity) */
    float *a = malloc(n * 4), *m0 = malloc(n * 4), *m1 = malloc(n * 4);
    for (size_t i = 0; i < n; i++) a[i] = (i * 2654435761u >> 7) % 5 == 0 ? NAN : (float)((i * 40503u) % 1000) / 7.f;
    CHECK(!fgpu_median(T.G, a, m1, w, h, 7, 20), "fgpu_median");
    for (int r = 0; r < h; r++) for (int c = 0; c < w; c++) {   /* the CPU definition */
        float buf[225]; int k = 0;
        for (int y = r - 7; y <= r + 7; y++) for (int x = c - 7; x <= c + 7; x++) if (y >= 0 && y < h && x >= 0 && x < w && !isnan(a[(size_t)y * w + x])) buf[k++] = a[(size_t)y * w + x];
        m0[(size_t)r * w + c] = k >= 20 ? fc_select(buf, k, k / 2) : NAN;
    }
    CHECK(!memcmp(m0, m1, n * 4), "median R 7: GPU differs from the CPU");
    fgpu_close(T.G);
    for (int g = 0; g < 2; g++) { for (int k = 0; k < 7; k++) free(out[g][k]); free(cf[g]); }
    for (int k = 0; k < nch; k++) free(T.ch[k]);
    free(T.ch); free(X); free(Y); free(Z); free(a); free(m0); free(m1);
    return 0;
}
#endif

int main(void) {
    /* 1. one profile: two sheets 5 and 4 voxels thick with a 3 voxel gap; edges at the half levels */
    {
        cyl C = {0, 0, 10, 15, 18, 22}; float f[201]; const double h = 0.1;
        for (int k = 0; k < 201; k++) f[k] = (float)cyl_ct(&C, k * h + 5);   /* t = r - 15 + ... : t0 = 5 */
        fc_band b[4]; const int n = faces_bands(f, 201, 5, h, 8, 0.1, b, 4);
        CHECK(n == 2, "two bands, got %d", n);
        if (n == 2) {
            CHECK(fabs(b[0].near - 10) < 0.15 && fabs(b[0].far - 15) < 0.15, "sheet 1 at 10..15, got %.3f..%.3f", b[0].near, b[0].far);
            CHECK(fabs(b[1].near - 18) < 0.15 && fabs(b[1].far - 22) < 0.15, "sheet 2 at 18..22, got %.3f..%.3f", b[1].near, b[1].far);
            CHECK(b[0].prom > 100, "contrast %.1f", b[0].prom);
        }
        /* touching sheets (no gap: a shallow valley): split at the valley */
        for (int k = 0; k < 201; k++) { const double r = k * h + 5; f[k] = (float)(20 + 130 * (S(r - 10) - S(r - 22)) - 30 * exp(-(r - 15) * (r - 15) / 0.5)); }
        const int m = faces_bands(f, 201, 5, h, 8, 0.1, b, 4);
        CHECK(m == 2, "touching: two bands, got %d", m);
        if (m == 2) CHECK(fabs(b[0].far - 15) < 0.3 && fabs(b[1].near - 15) < 0.3 && fabs(b[0].near - 10) < 0.2 && fabs(b[1].far - 22) < 0.2,
                          "touching: 10..15 | 15..22, got %.2f..%.2f | %.2f..%.2f", b[0].near, b[0].far, b[1].near, b[1].far);
        /* a shallow dip inside one sheet (two fibre layers) keeps one band */
        for (int k = 0; k < 201; k++) { const double r = k * h + 5; f[k] = (float)(20 + 130 * (S(r - 10) - S(r - 16)) - 6 * exp(-(r - 13) * (r - 13) / 0.5)); }
        const int q = faces_bands(f, 201, 5, h, 8, 0.1, b, 4);
        CHECK(q == 1 && fabs(b[0].near - 10) < 0.2 && fabs(b[0].far - 16) < 0.2, "fibre dip: one band 10..16, got %d %.2f..%.2f", q, q ? b[0].near : 0, q ? b[0].far : 0);
    }
    /* 2. a wrap (arc of a cylinder, rows = z) in the gap 2 voxels inside sheet 1 (r 100..105), sheet 2 at 109..113 */
    for (int flip = 0; flip < 2; flip++) {
        cyl C = {300, 300, 100, 105, 109, 113};
        const int w = 160, h = 24; const size_t n = (size_t)w * h;
        float *X = malloc(n * 4), *Y = malloc(n * 4), *Z = malloc(n * 4), *RX = malloc(n * 4), *RY = malloc(n * 4), *RZ = malloc(n * 4), *VX = malloc(n * 4), *VY = malloc(n * 4), *VZ = malloc(n * 4), *th = malloc(n * 4);
        uint8_t *cf = malloc(n);
        for (int r = 0; r < h; r++) for (int c = 0; c < w; c++) {
            const size_t i = (size_t)r * w + c; const double a = (flip ? w - 1 - c : c) * 0.02, rad = 98 + 0.3 * sin(0.1 * c + 0.2 * r);
            X[i] = (float)(C.cx + rad * cos(a)); Y[i] = (float)(C.cy + rad * sin(a)); Z[i] = (float)(50 + 2 * r);
        }
        Z[5 * w + 7] = -1; X[5 * w + 7] = Y[5 * w + 7] = -1;   /* a hole */
        const int sign = faces_orient(X, Y, Z, w, h, nullptr, 0);
        double N[3]; faces_normal(X, Y, Z, w, h, (size_t)10 * w + 80, N);
        const double ry = Y[10 * w + 80] - C.cy, rx = X[10 * w + 80] - C.cx;
        CHECK(sign * (N[1] * ry + N[2] * rx) > 0, "flip %d: orientation %d points inward", flip, sign);
        faces_opt o = {.range = 12, .tmin = 2, .tmax = 12, .min_prom = 8, .rel = 0.1, .tol = 2.5, .inset = 0, .smooth = 3, .nthreads = 4};
        faces_vol V = {&C, cyl_sample, nullptr, nullptr, nullptr};
        faces_stats st = {0}; st.ct_thr = 90;
        CHECK(!faces_wrap(X, Y, Z, w, h, 1.0, 9.6, sign, &o, &V, RX, RY, RZ, VX, VY, VZ, cf, th, &st), "faces_wrap");
        double er = 0, ev = 0, et = 0; int k = 0;
        for (int r = 3; r < h - 3; r++) for (int c = 3; c < w - 3; c++) {
            const size_t i = (size_t)r * w + c; if (Z[i] < 0) continue;
            er = fmax(er, fabs(hypot(RY[i] - C.cy, RX[i] - C.cx) - 100)); ev = fmax(ev, fabs(hypot(VY[i] - C.cy, VX[i] - C.cx) - 105));
            et = fmax(et, fabs(th[i] - 5 * 9.6)); k++;
        }
        printf("flip %d: %d vertices, recto err %.3f verso err %.3f thick err %.2f um, picked %zu/%zu, sign %d\n", flip, k, er, ev, et, st.npick, st.nv, sign);
        CHECK(k > 0 && er < 0.3 && ev < 0.3, "flip %d: recto |r - 100| <= %.3f, verso |r - 105| <= %.3f", flip, er, ev);
        CHECK(et < 4, "flip %d: thickness error %.2f um", flip, et);
        CHECK(RZ[5 * w + 7] < 0 && VZ[5 * w + 7] < 0, "hole stays invalid");
        CHECK(st.npick > 0.9 * st.nv, "flip %d: picked %zu of %zu", flip, st.npick, st.nv);
        free(X); free(Y); free(Z); free(RX); free(RY); free(RZ); free(VX); free(VY); free(VZ); free(th); free(cf);
    }
#ifdef FACES_GPU
    gpu_vs_cpu();
#endif
    if (fails) { fprintf(stderr, "test_faces: %d failures\n", fails); return 1; }
    printf("test_faces: ok\n");
    return 0;
}
