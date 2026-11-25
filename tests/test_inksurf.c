/* The surface-aligned ink head (src/inksurf.h, src/nn_inksurf.cu, src/inksurf.c; DESIGN 2026-11-20 "inksurf"):
   (1) the resample against a host trilinear reference (fp32 and fp16 features);
   (2) its adjoint: <resample(F), G> = <F, gather(G)> over the host-built inverse lists, and the gather bit-identical run to run;
   (3) the 2-D ops against host references (convs 1x1 / 3x3, their data / weight gradients by the adjoint identity, pool, up);
   (4) the head's gradients (parameters and input) against central differences;
   (5) the toy: a curved synthetic sheet whose ink is texture only (extra zero-mean noise in the papyrus under the face, no mean change),
       a small U-Net trunk from scratch + resample + head trained on 2-D labels alone; held-out AUC must pass 0.9. The control: the 3-D
       voxel ink head (the same trunk, splatted labels) on the same volumes; TIS_* (ink / papyrus amplitude, strokes, curvature, stacked
       sheets, mesh error, K, TP, seed; TIS_TOY_ONLY) set the difficulty: DESIGN 2026-11-20 "inksurf" has the sweep.
   UFSM_TEST_QUICK=1: (5) with fewer steps. */
#include "inksurf.h"
#include "nn.h"
#include "unet.h"
#include "nn_lp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "test_gpu.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)
static uint64_t rs = 12345;
static double urand(void) { rs = rs * 6364136223846793005ull + 1442695040888963407ull; return ((rs >> 11) + 0.5) / 9007199254740992.0; }
static float frand(void) { return (float)(2 * urand() - 1); }
static float h2f(uint16_t h) { const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023; float v;
    if (e == 0) v = ldexpf((float)m, -24); else if (e == 31) v = m ? NAN : INFINITY; else v = ldexpf((float)(m | 1024), (int)e - 25); return s ? -v : v; }
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t f2h(float f) { if (f == 0) return 0; const int s = f < 0; f = fabsf(f); int e; float m = frexpf(f, &e); int E = e + 14;
    if (E <= 0) return (uint16_t)(s << 15 | (int)lrintf(ldexpf(f, 24))); int q = (int)lrintf((m * 2 - 1) * 1024); if (q == 1024) { q = 0; E++; } return (uint16_t)(s << 15 | E << 10 | q); }

static float tri_ref(const float *F, int D, int H, int W, float z, float y, float x) {
    int iz, iy, ix; float wz, wy, wx; isf_cell(z, &iz, &wz); isf_cell(y, &iy, &wy); isf_cell(x, &ix, &wx); float s = 0;
    for (int k = 0; k < 8; k++) { const int cz = iz + (k >> 2 & 1), cy = iy + (k >> 1 & 1), cx = ix + (k & 1);
        if (cz < 0 || cy < 0 || cx < 0 || cz >= D || cy >= H || cx >= W) continue;
        s += (k >> 2 & 1 ? wz : 1 - wz) * (k >> 1 & 1 ? wy : 1 - wy) * (k & 1 ? wx : 1 - wx) * F[((size_t)cz * H + cy) * W + cx]; }
    return s;
}
static void conv_ref(const float *x, int Ci, int H, int W, const float *w, const float *b, int Co, int k, float *y) {
    const int r = k / 2;
    for (int co = 0; co < Co; co++) for (int py = 0; py < H; py++) for (int px = 0; px < W; px++) {
        double s = b ? b[co] : 0;
        for (int ci = 0; ci < Ci; ci++) for (int dy = 0; dy < k; dy++) for (int dx = 0; dx < k; dx++) { const int yy = py + dy - r, xx = px + dx - r;
            if (yy >= 0 && yy < H && xx >= 0 && xx < W) s += (double)w[((co * Ci + ci) * k + dy) * k + dx] * x[((size_t)ci * H + yy) * W + xx]; }
        y[((size_t)co * H + py) * W + px] = (float)s;
    }
}
static double dotd(const float *a, const float *b, size_t n) { double s = 0; for (size_t i = 0; i < n; i++) s += (double)a[i] * b[i]; return s; }
static float *dup_h2d(const float *h, size_t n) { float *d = nn_malloc(n * 4); nn_h2d(d, h, n * 4); return d; }

static void test_resample(void) {
    const int C = 3, D = 10, H = 12, W = 14, n = 60, K = 5; const float step = 0.7f; const size_t S = (size_t)D * H * W;
    float *F = malloc(C * S * 4), *pn = malloc(n * 6 * 4), *G = malloc((size_t)C * K * n * 4), *out = malloc((size_t)C * K * n * 4), *ref = malloc((size_t)C * K * n * 4);
    for (size_t i = 0; i < C * S; i++) F[i] = frand();
    for (int p = 0; p < n; p++) { float *q = pn + 6 * p; q[0] = (float)(urand() * (D + 2) - 1.5); q[1] = (float)(urand() * (H + 2) - 1.5); q[2] = (float)(urand() * (W + 2) - 1.5);
        double a = frand(), b = frand(), c = frand(), l = sqrt(a * a + b * b + c * c) + 1e-9; q[3] = (float)(a / l); q[4] = (float)(b / l); q[5] = (float)(c / l); }
    if (n > 3) { pn[0] = 4; pn[1] = 5; pn[2] = 6; pn[3] = 0; pn[4] = 0; pn[5] = 1; }   /* (an axis-aligned sample: exact grid points) */
    for (int c = 0; c < C; c++) for (int j = 0; j < K; j++) for (int p = 0; p < n; p++) { float z, y, x; isf_pos(pn + 6 * p, (j - 0.5f * (K - 1)) * step, &z, &y, &x);
        ref[((size_t)c * K + j) * n + p] = tri_ref(F + c * S, D, H, W, z, y, x); }
    float *dF = dup_h2d(F, C * S), *dpn = dup_h2d(pn, n * 6), *dout = nn_malloc((size_t)C * K * n * 4);
    nn_isf_resample(dF, ISF_F32, C, D, H, W, dpn, n, K, step, dout); nn_d2h(out, dout, (size_t)C * K * n * 4);
    double md = 0; for (size_t i = 0; i < (size_t)C * K * n; i++) md = fmax(md, fabs(out[i] - ref[i]));
    fprintf(stderr, "resample fp32 vs host: max |diff| %.2e\n", md); CHECK(md < 1e-5, "resample fp32 %.2e", md);
    /* fp16 features: the same within the fp16 rounding of F */
    uint16_t *F16 = malloc(C * S * 2); float *Fr = malloc(C * S * 4); for (size_t i = 0; i < C * S; i++) { F16[i] = f2h(F[i]); Fr[i] = h2f(F16[i]); }
    void *dF16 = nn_malloc(C * S * 2); nn_h2d(dF16, F16, C * S * 2); nn_isf_resample(dF16, ISF_F16, C, D, H, W, dpn, n, K, step, dout); nn_d2h(out, dout, (size_t)C * K * n * 4);
    md = 0; for (int c = 0; c < C; c++) for (int j = 0; j < K; j++) for (int p = 0; p < n; p++) { float z, y, x; isf_pos(pn + 6 * p, (j - 0.5f * (K - 1)) * step, &z, &y, &x);
        md = fmax(md, fabs(out[((size_t)c * K + j) * n + p] - tri_ref(Fr + c * S, D, H, W, z, y, x))); }
    fprintf(stderr, "resample fp16 vs host: max |diff| %.2e\n", md); CHECK(md < 1e-5, "resample fp16 %.2e", md);
    /* adjoint: <R F, G> = <F, R^T G> */
    isf_lists L; CHECK(!isf_lists_build(pn, n, K, step, D, H, W, &L), "lists");
    for (size_t i = 0; i < (size_t)C * K * n; i++) G[i] = frand();
    int *dv = nn_malloc((size_t)(L.nv + 1) * 4), *doff = nn_malloc((size_t)(L.nv + 1) * 4), *dent = nn_malloc((size_t)L.ne * 4); float *dew = nn_malloc((size_t)L.ne * 4);
    nn_h2d(dv, L.vox, (size_t)L.nv * 4); nn_h2d(doff, L.off, (size_t)(L.nv + 1) * 4); nn_h2d(dent, L.ent, (size_t)L.ne * 4); nn_h2d(dew, L.ew, (size_t)L.ne * 4);
    float *dG = dup_h2d(G, (size_t)C * K * n), *dg = nn_malloc(C * S * 4), *g1 = malloc(C * S * 4), *g2 = malloc(C * S * 4);
    nn_zero(dg, C * S * 4); nn_isf_gather(dG, C, K, n, dv, doff, dent, dew, L.nv, dg, ISF_F32, S, 1.f, nullptr, 0); nn_d2h(g1, dg, C * S * 4);
    nn_zero(dg, C * S * 4); nn_isf_gather(dG, C, K, n, dv, doff, dent, dew, L.nv, dg, ISF_F32, S, 1.f, nullptr, 0); nn_d2h(g2, dg, C * S * 4);
    const double lhs = dotd(ref, G, (size_t)C * K * n), rhs = dotd(F, g1, C * S);
    fprintf(stderr, "adjoint: <R F, G> %.9g  <F, R^T G> %.9g  (%zu entries over %d voxels)\n", lhs, rhs, (size_t)L.ne, L.nv);
    CHECK(fabs(lhs - rhs) <= 1e-5 * (fabs(lhs) + 1), "adjoint %.9g vs %.9g", lhs, rhs);
    CHECK(!memcmp(g1, g2, C * S * 4), "gather not bit-identical run to run");
    /* fp16 gradient output with scale: g16 = scale * g within fp16 rounding; acc adds */
    void *dg16 = nn_malloc(C * S * 2); nn_zero(dg16, C * S * 2); nn_isf_gather(dG, C, K, n, dv, doff, dent, dew, L.nv, dg16, ISF_F16, S, 8.f, nullptr, 0);
    uint16_t *h16 = malloc(C * S * 2); nn_d2h(h16, dg16, C * S * 2); md = 0; for (size_t i = 0; i < C * S; i++) md = fmax(md, fabs(h2f(h16[i]) - 8 * g1[i]) / (1 + fabs(8 * g1[i])));
    CHECK(md < 2e-3, "gather fp16 rel %.2e", md);
    (void)bf2f;
    isf_lists_free(&L); free(F); free(pn); free(G); free(out); free(ref); free(F16); free(Fr); free(g1); free(g2); free(h16);
    nn_free(dF); nn_free(dpn); nn_free(dout); nn_free(dF16); nn_free(dv); nn_free(doff); nn_free(dent); nn_free(dew); nn_free(dG); nn_free(dg); nn_free(dg16);
}

static void test_2d(void) {
    const int Ci = 5, Co = 4, H = 10, W = 8;
    for (int k = 1; k <= 3; k += 2) {
        const size_t nx = (size_t)Ci * H * W, ny = (size_t)Co * H * W, nw = (size_t)Co * Ci * k * k;
        float *x = malloc(nx * 4), *w = malloc(nw * 4), *b = malloc(Co * 4), *y = malloc(ny * 4), *yr = malloc(ny * 4), *gy = malloc(ny * 4), *gx = malloc(nx * 4), *gw = malloc(nw * 4), *gb = malloc(Co * 4);
        for (size_t i = 0; i < nx; i++) x[i] = frand(); for (size_t i = 0; i < nw; i++) w[i] = frand(); for (int i = 0; i < Co; i++) b[i] = frand(); for (size_t i = 0; i < ny; i++) gy[i] = frand();
        float *dx = dup_h2d(x, nx), *dw = dup_h2d(w, nw), *db = dup_h2d(b, Co), *dy = nn_malloc(ny * 4), *dgy = dup_h2d(gy, ny), *dgx = nn_malloc(nx * 4), *dgw = nn_malloc(nw * 4), *dgb = nn_malloc(Co * 4);
        nn_isf2_conv(dx, Ci, H, W, dw, db, Co, k, dy); nn_d2h(y, dy, ny * 4); conv_ref(x, Ci, H, W, w, b, Co, k, yr);
        double md = 0; for (size_t i = 0; i < ny; i++) md = fmax(md, fabs(y[i] - yr[i])); CHECK(md < 1e-4, "conv k%d fwd %.2e", k, md);
        nn_isf2_conv_bd(dgy, Co, H, W, dw, Ci, k, dgx, 0); nn_d2h(gx, dgx, nx * 4);
        nn_isf2_conv_bw(dx, Ci, H, W, dgy, Co, k, dgw, dgb, 0); nn_d2h(gw, dgw, nw * 4); nn_d2h(gb, dgb, Co * 4);
        conv_ref(x, Ci, H, W, w, nullptr, Co, k, yr);   /* <conv_nobias(x), gy> = <x, gx> = <w, gw> */
        const double a = dotd(yr, gy, ny), bx = dotd(x, gx, nx), bw = dotd(w, gw, nw); double sb = 0; for (size_t i = 0; i < ny; i++) sb += gy[i];
        double gbm = 0; for (int c = 0; c < Co; c++) { double s = 0; for (int q = 0; q < H * W; q++) s += gy[c * H * W + q]; gbm = fmax(gbm, fabs(s - gb[c])); }
        fprintf(stderr, "conv k%d: fwd %.1e, <y, gy> %.6g  <x, gx> %.6g  <w, gw> %.6g, bias %.1e\n", k, md, a, bx, bw, gbm);
        CHECK(fabs(a - bx) < 1e-4 * (fabs(a) + 1) && fabs(a - bw) < 1e-4 * (fabs(a) + 1) && gbm < 1e-4, "conv k%d adjoints", k);
        (void)sb;
        free(x); free(w); free(b); free(y); free(yr); free(gy); free(gx); free(gw); free(gb);
        nn_free(dx); nn_free(dw); nn_free(db); nn_free(dy); nn_free(dgy); nn_free(dgx); nn_free(dgw); nn_free(dgb);
    }
    {   /* pool / up adjoints */
        const int C = 3, H = 8, W = 6; const size_t n = (size_t)C * H * W, m = n / 4;
        float *x = malloc(n * 4), *g = malloc(m * 4), *p = malloc(m * 4), *gx = malloc(n * 4), *u = malloc(n * 4), *gu = malloc(m * 4);
        for (size_t i = 0; i < n; i++) x[i] = frand(); for (size_t i = 0; i < m; i++) g[i] = frand();
        float *dx = dup_h2d(x, n), *dg = dup_h2d(g, m), *dp = nn_malloc(m * 4), *dgx = nn_malloc(n * 4), *du = nn_malloc(n * 4), *dgu = nn_malloc(m * 4);
        nn_isf2_pool(dx, C, H, W, dp); nn_isf2_pool_bwd(dg, C, H, W, dgx, 0); nn_d2h(p, dp, m * 4); nn_d2h(gx, dgx, n * 4);
        CHECK(fabs(dotd(p, g, m) - dotd(x, gx, n)) < 1e-5, "pool adjoint");
        nn_isf2_up(dg, C, H / 2, W / 2, du); nn_isf2_up_bwd(dx, C, H / 2, W / 2, dgu); nn_d2h(u, du, n * 4); nn_d2h(gu, dgu, m * 4);
        CHECK(fabs(dotd(u, x, n) - dotd(g, gu, m)) < 1e-5, "up adjoint");
        free(x); free(g); free(p); free(gx); free(u); free(gu); nn_free(dx); nn_free(dg); nn_free(dp); nn_free(dgx); nn_free(du); nn_free(dgu);
    }
}

/* (3b) the tiled convs against the one-thread-per-output _ref kernels, bit for bit: odd sizes (partial tiles), channel counts off the
   kernels' groups (1, 5, 17, 33 vs 4 / 8 / 16), k 1 / 3, the data gradient's accumulate, the weight gradient's acc (inkperf 2026-11-21) */
static void test_2d_tiled(void) {
    static const int S[][5] = {{5, 4, 3, 10, 8}, {17, 33, 3, 37, 20}, {33, 1, 1, 21, 19}, {8, 16, 1, 16, 16}, {3, 5, 3, 2, 3}, {40, 48, 3, 64, 64}, {96, 17, 1, 33, 47}};
    for (size_t s = 0; s < sizeof S / sizeof S[0]; s++) {
        const int Ci = S[s][0], Co = S[s][1], k = S[s][2], H = S[s][3], W = S[s][4]; const size_t nx = (size_t)Ci * H * W, ny = (size_t)Co * H * W, nw = (size_t)Co * Ci * k * k;
        float *h = malloc((nx > ny ? (nx > nw ? nx : nw) : (ny > nw ? ny : nw)) * 4), *h2 = malloc((nx > ny ? (nx > nw ? nx : nw) : (ny > nw ? ny : nw)) * 4);
        for (size_t i = 0; i < nx; i++) h[i] = frand(); float *dx = dup_h2d(h, nx);
        for (size_t i = 0; i < ny; i++) h[i] = frand(); float *dgy = dup_h2d(h, ny);
        for (size_t i = 0; i < nw; i++) h[i] = 0.3f * frand(); float *dw = dup_h2d(h, nw);
        for (int i = 0; i < Co; i++) h[i] = frand(); float *db = dup_h2d(h, Co);
        float *y0 = nn_malloc(ny * 4), *y1 = nn_malloc(ny * 4), *g0 = nn_malloc(nx * 4), *g1 = nn_malloc(nx * 4), *w0 = nn_malloc(nw * 4), *w1 = nn_malloc(nw * 4), *b0 = nn_malloc(Co * 4), *b1 = nn_malloc(Co * 4);
        int bad[3] = {0, 0, 0};
        for (int acc = 0; acc < 2; acc++) {
            if (!acc) { nn_isf2_conv_ref(dx, Ci, H, W, dw, db, Co, k, y0); nn_isf2_conv(dx, Ci, H, W, dw, db, Co, k, y1);
                        nn_d2h(h, y0, ny * 4); nn_d2h(h2, y1, ny * 4); bad[0] |= memcmp(h, h2, ny * 4) != 0;
                        nn_d2d(g0, dx, nx * 4); nn_d2d(g1, dx, nx * 4); nn_zero(w0, nw * 4); nn_zero(w1, nw * 4); }
            nn_isf2_conv_bd_ref(dgy, Co, H, W, dw, Ci, k, g0, acc); nn_isf2_conv_bd(dgy, Co, H, W, dw, Ci, k, g1, acc);
            nn_d2h(h, g0, nx * 4); nn_d2h(h2, g1, nx * 4); bad[1] |= memcmp(h, h2, nx * 4) != 0;
            nn_isf2_conv_bw_ref(dx, Ci, H, W, dgy, Co, k, w0, b0, acc); nn_isf2_conv_bw(dx, Ci, H, W, dgy, Co, k, w1, b1, acc);
            nn_d2h(h, w0, nw * 4); nn_d2h(h2, w1, nw * 4); bad[2] |= memcmp(h, h2, nw * 4) != 0;
        }
        CHECK(!bad[0] && !bad[1] && !bad[2], "tiled conv %d -> %d k%d %dx%d: forward %s, data grad %s, weight grad %s (vs _ref)", Ci, Co, k, H, W,
              bad[0] ? "DIFFERS" : "=", bad[1] ? "DIFFERS" : "=", bad[2] ? "DIFFERS" : "=");
        free(h); free(h2); nn_free(dx); nn_free(dgy); nn_free(dw); nn_free(db); nn_free(y0); nn_free(y1); nn_free(g0); nn_free(g1); nn_free(w0); nn_free(w1); nn_free(b0); nn_free(b1);
    }
    fprintf(stderr, "tiled convs: bit-identical to the _ref kernels on %zu shapes: %s\n", sizeof S / sizeof S[0], fails ? "NO" : "yes");
}

static double g_labs;   /* sum |r z| of the last evaluation: the scale of its fp32 rounding noise */
static double head_loss(isfhead *h, const float *dx, const float *r, int n) {
    float *z = malloc(n * 4); nn_d2h(z, isfhead_forward(h, dx), n * 4); nn_sync(); double s = 0; g_labs = 0;
    for (int i = 0; i < n; i++) { s += (double)r[i] * z[i]; g_labs += fabs((double)r[i] * z[i]); } free(z); return s;
}
static void test_head_fd(int L) {   /* (depth L: 4 the S1 / S2 head, 5 the S3 one) */
    const int cin = 4, Hp = L > 4 ? 1 << L : 16, Wp = Hp, n = Hp * Wp; const size_t nx = (size_t)cin * n;
    isfhead *h = isfhead_create_d(cin, Hp, Wp, L, 7); CHECK(h, "isfhead_create_d %d", L); if (!h) return;
    float *x = malloc(nx * 4), *r = malloc(n * 4); for (size_t i = 0; i < nx; i++) x[i] = frand(); for (int i = 0; i < n; i++) r[i] = frand();
    { const size_t np = isfhead_nparams(h); float *p = malloc(np * 4); nn_d2h(p, isfhead_params(h), np * 4); for (size_t i = 0; i < np; i++) p[i] += 0.05f * frand();   /* (the output conv off its small init) */
      nn_h2d(isfhead_params(h), p, np * 4); free(p); }
    float *dx = dup_h2d(x, nx), *dr = dup_h2d(r, n);
    isfhead_forward(h, dx); isfhead_zero_grad(h); const float *gxd = isfhead_backward(h, dr); float *gx = malloc(nx * 4); nn_d2h(gx, gxd, nx * 4);
    const size_t np = isfhead_nparams(h); float *gp = malloc(np * 4), *p = malloc(np * 4); nn_d2h(gp, isfhead_grads(h), np * 4); nn_d2h(p, isfhead_params(h), np * 4);
    /* directional derivatives (central differences of fp32 sums are ~1e-6 noisy, so single tiny entries are not a test; a random direction's
       derivative is O(|g|)): the input, and each conv's weights + bias apart */
    double worst = 0; char wn[32] = "";
    for (int t = -1; t < isfhead_nconv(h); t++) {
        size_t o = 0, m = nx; float *v = t < 0 ? x : p, *dv = t < 0 ? dx : isfhead_params(h); const float *g = t < 0 ? gx : gp;
        if (t >= 0) o = isfhead_conv_span(h, t, &m);
        float *d = malloc(m * 4), *v0 = malloc(m * 4); double gd = 0, dn = 0; memcpy(v0, v + o, m * 4);
        for (size_t i = 0; i < m; i++) { d[i] = frand(); dn += (double)d[i] * d[i]; gd += (double)g[o + i] * d[i]; }
        const double e = 1e-2 / sqrt(dn / m), tot = t < 0 ? nx : np;
        double f2[2];   /* steps e and e / 2, Richardson: (4 f(e / 2) - f(e)) / 3 cancels the O(e^2) curvature term */
        for (int q = 0; q < 2; q++) { const double es = q ? e / 2 : e;
            for (size_t i = 0; i < m; i++) v[o + i] = v0[i] + (float)es * d[i]; nn_h2d(dv, v, (size_t)tot * 4); const double lp = head_loss(h, dx, r, n);
            for (size_t i = 0; i < m; i++) v[o + i] = v0[i] - (float)es * d[i]; nn_h2d(dv, v, (size_t)tot * 4); const double lm = head_loss(h, dx, r, n);
            f2[q] = (lp - lm) / (2 * es); }
        memcpy(v + o, v0, m * 4); nn_h2d(dv, v, (size_t)tot * 4);
        /* pass: |fd - g| <= 3e-3 |g| + the differences' noise floor (fp32 logits: ~1e-7 sum |r z| per loss over the step; measured differences 1e-4 .. 3e-4 vs this 7e-4) */
        const double fd = (4 * f2[1] - f2[0]) / 3, nf = 1e-7 * g_labs / e, rel = fabs(fd - gd) / (3e-3 * fabs(gd) + nf);
        if (rel > worst) { worst = rel; if (t < 0) snprintf(wn, sizeof wn, "input"); else snprintf(wn, sizeof wn, "conv %d", t); }
        if (rel > 1 || getenv("TIS_V")) fprintf(stderr, "  %s%d: fd %.6g analytic %.6g (noise floor %.2g)\n", t < 0 ? "input" : "conv ", t < 0 ? 0 : t, fd, gd, nf);
        free(d); free(v0);
    }
    fprintf(stderr, "head gradients (depth %d): worst directional difference %.2f of the tolerance (3e-3 |g| + noise floor; %s; input + %d convs)\n", L, worst, wn, isfhead_nconv(h));
    CHECK(worst < 1, "head gradients (depth %d) %.2f of the tolerance at %s", L, worst, wn);
    free(x); free(r); free(gx); free(gp); free(p); nn_free(dx); nn_free(dr); isfhead_free(h);
}

/* (S3, 2026-11-21) a depth-4 .isf into a depth-5 head (--isf-depth 5 from S2): the 12 convs of the same role / level / shape take the file's
   parameters byte for byte (stem, level-0 conv, encoder 1-4, decoder 3-0, output, readout); the encoder 5, bottleneck and decoder 4 stay
   fresh; a depth-5 file round-trips with ISF2 */
static void test_isf_depth(void) {
    isf_dev a, b, c; const int C = 4, K = 3, Hp = 64; char p4[] = "/tmp/test_isf_depth_XXXXXX"; const int fd = mkstemp(p4); if (fd >= 0) close(fd);
    CHECK(!isf_dev_init_d(&a, C, Hp, K, 4, 3) && !isf_dev_init_d(&b, C, Hp, K, 5, 5), "isf_dev_init_d");
    CHECK(!isf_save(&a, p4, 7), "isf_save depth 4");
    int st = -1, nm = -1, L = 0; CHECK(!isf_peek(p4, nullptr, &L, &st) && L == 4 && st == 7, "isf_peek depth %d step %d", L, st);
    CHECK(!isf_load_x(&b, p4, &st, &nm) && nm == 12 && st == 0, "depth 4 -> 5: %d convs mapped (12), step %d", nm, st);
    const size_t na = isfhead_nparams(a.h), nb = isfhead_nparams(b.h); float *pa = malloc(na * 4), *pb = malloc(nb * 4);
    nn_d2h(pa, isfhead_params(a.h), na * 4); nn_d2h(pb, isfhead_params(b.h), nb * 4);
    static const int ma[12] = {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12}, mb[12] = {0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14}; int same = 0;
    for (int q = 0; q < 12; q++) { size_t n0, n1; const size_t o0 = isfhead_conv_span(a.h, ma[q], &n0), o1 = isfhead_conv_span(b.h, mb[q], &n1);
        same += n0 == n1 && !memcmp(pa + o0, pb + o1, n0 * 4); }
    CHECK(same == 12, "%d of 12 mapped convs byte-identical", same);
    char p5[300]; snprintf(p5, sizeof p5, "%s.5", p4); CHECK(!isf_save(&b, p5, 9), "isf_save depth 5");
    CHECK(!isf_dev_init_d(&c, C, Hp, K, 5, 11) && !isf_load_x(&c, p5, &st, &nm) && nm == 15 && st == 9, "depth 5 round trip: %d convs, step %d", nm, st);
    float *pc = malloc(nb * 4); nn_d2h(pc, isfhead_params(c.h), nb * 4); CHECK(!memcmp(pb, pc, nb * 4), "depth 5 round trip parameters");
    fprintf(stderr, "isf depth: 4 -> 5 mapped %d of 15 convs (12 byte-identical), depth 5 round trip ok\n", 12);
    unlink(p4); unlink(p5); free(pa); free(pb); free(pc); isf_dev_free(&a); isf_dev_free(&b); isf_dev_free(&c);
}

/* ---- (5) the toy ---- */
static int TP = 64;   /* (window and patch side; TIS_TP, a multiple of 16) */
typedef struct { float A, f1, f2, ph1, ph2, th, sp, so, f3, f4, ph3, ph4; } sheet_t;
/* difficulty: the ink's texture amplitude against the papyrus's own (zero-mean uniform noise each), the stroke period range (px) */
static float g_ink = 0.6f, g_bg = 0.15f, g_sp0 = 16, g_sp1 = 32, g_a0 = 3, g_a1 = 6, g_wl0 = 50, g_wl1 = 90, g_gap = 0, g_moff = 0;   /* sheet amplitude / wavelength
   ranges; g_gap > 0: further sheets every g_gap voxels along z (their own ink, unlabelled); g_moff: the mesh's error, a smooth offset
   of +-g_moff voxels along z from the true sheet (the labels' splat and the band read follow the mesh, as in the real data) */
static float envf(const char *k, float d) { const char *e = getenv(k); return e && *e ? (float)atof(e) : d; }
static float sh_z(const sheet_t *s, double y, double x) { return (float)(TP / 2.0 + s->A * sin(s->f1 * x + s->ph1) * cos(s->f2 * y + s->ph2)); }
static float mesh_z(const sheet_t *s, double y, double x) { return sh_z(s, y, x) + (float)(g_moff * sin(s->f3 * x + s->ph3) * cos(s->f4 * y + s->ph4)); }
static int sh_ink(const sheet_t *s, double y, double x) { return sin((x * cos(s->th) + y * sin(s->th)) * s->sp + s->so) > 0.2; }
static void toy_volume(sheet_t *s, float *ct, float *pn, float *lab) {
    s->A = (float)(g_a0 + (g_a1 - g_a0) * urand()); s->f1 = (float)(6.2832 / (g_wl0 + (g_wl1 - g_wl0) * urand())); s->f2 = (float)(6.2832 / (g_wl0 + (g_wl1 - g_wl0) * urand())); s->ph1 = (float)(6.28 * urand()); s->ph2 = (float)(6.28 * urand());
    s->th = (float)(3.1416 * urand()); s->sp = (float)(6.2832 / (g_sp0 + (g_sp1 - g_sp0) * urand())); s->so = (float)(6.28 * urand());
    s->f3 = (float)(6.2832 / (30 + 30 * urand())); s->f4 = (float)(6.2832 / (30 + 30 * urand())); s->ph3 = (float)(6.28 * urand()); s->ph4 = (float)(6.28 * urand());
    for (int z = 0; z < TP; z++) for (int y = 0; y < TP; y++) for (int x = 0; x < TP; x++) {
        float d = z - sh_z(s, y, x), v = 0.05f * frand(); int k = 0;
        if (g_gap > 0) { k = (int)floorf(d / g_gap + 0.5f); d -= k * g_gap; }   /* (sheet k: its own stroke phase) */
        sheet_t sk = *s; sk.so += 2.1f * k; sk.th += 0.7f * k;
        if (fabsf(d) <= 3) { v += 1.f + g_bg * frand(); if (d >= 0 && sh_ink(&sk, y, x)) v += g_ink * frand(); }   /* ink: zero-mean texture in the papyrus's +z half */
        ct[((size_t)z * TP + y) * TP + x] = v;
    }
    for (int y = 0; y < TP; y++) for (int x = 0; x < TP; x++) {
        const float z0 = mesh_z(s, y, x), dzy = (mesh_z(s, y + 0.5, x) - mesh_z(s, y - 0.5, x)), dzx = (mesh_z(s, y, x + 0.5) - mesh_z(s, y, x - 0.5));
        const float l = sqrtf(1 + dzy * dzy + dzx * dzx); float *q = pn + 6 * ((size_t)y * TP + x);
        q[0] = z0; q[1] = (float)y; q[2] = (float)x; q[3] = 1 / l; q[4] = -dzy / l; q[5] = -dzx / l;
        lab[(size_t)y * TP + x] = (float)sh_ink(s, y, x);
    }
}
static const float *g_az;
static int az_cmp(const void *a, const void *b) { const float u = g_az[*(const int *)a], v = g_az[*(const int *)b]; return (u > v) - (u < v); }
static double auc(const float *z, const float *y, int n) {   /* (ties not split: the logits are continuous) */
    double pos = 0, neg = 0, a = 0; int *o = malloc(n * sizeof *o); for (int i = 0; i < n; i++) o[i] = i;
    g_az = z; qsort(o, n, sizeof *o, az_cmp);
    for (int r = 0; r < n; r++) { if (y[o[r]] > 0.5f) { a += neg; pos++; } else neg++; }
    free(o); return pos && neg ? a / (pos * neg) : 0.5;
}
static void test_toy(void) {
    const int quick = getenv("UFSM_TEST_QUICK") && atoi(getenv("UFSM_TEST_QUICK")), steps = quick ? 150 : 400, K = (int)envf("TIS_K", 5), n = TP * TP; const float step = 1.f;
    /* the trunk at the production training precision (wtrunk's: MX-fp4 activations, MX-fp8 activation gradients, fp4 weight gradients, SR) */
    nn_set_tf32(1); nn_set_act_bf16(1); nn_set_grad_bf16(1); nn_set_f16(1); nn_set_grad_scale(1024); nn_set_sr(1); nn_set_gn_stored(1);
    nn_set_prec(3); CHECK(!nn_set_prec_policy("all=fp4:fp4:fp4"), "prec policy"); setenv("UFSM_F4_WGRAD", "1", 1);
    unet_set_act_mx4(1); unet_set_grad_mx8(1); unet_set_grad_mx4(0);
    unet_cfg cfg = {3, {16, 32, 64}, 1, 1, 8, 1, 0};
    unet *u = unet_create(&cfg); unet_init(u, 5); unet_set_fp16_logits(u, 1);
    const int C0 = cfg.widths[0]; const size_t V = (size_t)TP * TP * TP;
    float *hid = calloc((size_t)C0 * (C0 + 1), 4); for (int c = 0; c < C0; c++) hid[(size_t)c * C0 + c] = 1.f;
    float *did = dup_h2d(hid, (size_t)C0 * (C0 + 1)); void *feat = nn_malloc(V * C0 * 2), *gf = nn_malloc(V * C0 * 2), *glz = nn_malloc(V * 2); nn_zero(glz, V * 2);
    const int ft = nn_get_f16() ? ISF_F16 : ISF_BF16;
    const int ctin = (int)envf("TIS_CTIN", 0), cin = C0 * K + (ctin ? K : 0);   /* TIS_CTIN=1: the CT's own K layers as extra patch channels */
    isfhead *h = isfhead_create(cin, TP, TP, 11);
    float *ct = malloc(V * 4), *pn = malloc((size_t)n * 6 * 4), *lab = malloc(n * 4), *w1 = malloc(n * 4), *z = malloc(n * 4);
    for (int i = 0; i < n; i++) w1[i] = 1.f;
    float *dct = nn_malloc(V * 4), *dpn = nn_malloc((size_t)n * 6 * 4), *dlab = nn_malloc(n * 4), *dw = dup_h2d(w1, n), *dpatch = nn_malloc((size_t)cin * n * 4), *dgl = nn_malloc(n * 4);
    const shape5 xs = {1, 1, TP, TP, TP}; sheet_t sh; double t0 = 0; int bad = 0;
    /* the control (the design's): the existing 3-D voxel ink head, the same trunk from the same init, the same volumes and steps, per-voxel
       BCE on the 2-D label splatted into the inked half (0 <= d <= 3 along z from the mesh), read back as its mean logit over that band */
    unet *uv = unet_create(&cfg); unet_init(uv, 5); unet_set_fp16_logits(uv, 1);
    uint16_t *lv16 = malloc(V * 2); float *gv = malloc(V * 4), *lvf = malloc(V * 4), *dgv = nn_malloc(V * 4); void *gv16 = nn_malloc(V * 2);
    for (int it = 1; it <= steps && !bad; it++) {
        toy_volume(&sh, ct, pn, lab); nn_h2d(dct, ct, V * 4); nn_h2d(dpn, pn, (size_t)n * 6 * 4); nn_h2d(dlab, lab, n * 4);
        isf_lists L; if (isf_lists_build(pn, n, K, step, TP, TP, TP, &L)) { bad = 1; break; }
        int *dv = nn_malloc((size_t)(L.nv + 1) * 4), *doff = nn_malloc((size_t)(L.nv + 1) * 4), *dent = nn_malloc((size_t)L.ne * 4); float *dew = nn_malloc((size_t)L.ne * 4);
        nn_h2d(dv, L.vox, (size_t)L.nv * 4); nn_h2d(doff, L.off, (size_t)(L.nv + 1) * 4); nn_h2d(dent, L.ent, (size_t)L.ne * 4); nn_h2d(dew, L.ew, (size_t)L.ne * 4);
        unet_zero_grad(u);
        if (!unet_forward(u, dct, xs, 1) || unet_head_ext_fwd(u, did, did + (size_t)C0 * C0, C0, feat)) { fprintf(stderr, "toy: the trunk's features are not covered\n"); bad = 1; }
        if (!bad) {
            nn_isf_resample(feat, ft, C0, TP, TP, TP, dpn, n, K, step, dpatch); if (ctin) nn_isf_resample(dct, ISF_F32, 1, TP, TP, TP, dpn, n, K, step, dpatch + (size_t)C0 * K * n);
            const float *zl = isfhead_forward(h, dpatch);
            nn_isf2_bce_grad(zl, dlab, dw, n, 1.f / n, dgl);
            if (it % 50 == 0) { nn_d2h(z, zl, n * 4); double l = 0; for (int i = 0; i < n; i++) { const double p = 1 / (1 + exp(-z[i])); l -= lab[i] > 0.5f ? log(p + 1e-9) : log(1 - p + 1e-9); }
                                fprintf(stderr, "toy step %d: loss %.4f, train AUC %.3f\n", it, l / n, auc(z, lab, n)); }
            isfhead_zero_grad(h); const float *gpatch = isfhead_backward(h, dgl);
            nn_zero(gf, V * C0 * 2); nn_isf_gather(gpatch, C0, K, n, dv, doff, dent, dew, L.nv, gf, ft, V, unet_bwd_gscale(), nullptr, 0);
            unet_set_l0_feat_grad(u, gf); unet_backward_x(u, glz, 1); unet_set_l0_feat_grad(u, nullptr);
            if (it == 1) t0 = unet_grad_norm(u);
            unet_muon(u, 0.01f, 0.95f, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.f, it);
            isfhead_adamw(h, 1e-3f, 0.f, it);
            unet_zero_grad(uv); const void *lgv = unet_forward(uv, dct, xs, 1); nn_sync(); nn_d2h(lv16, lgv, V * 2);
            size_t nw = 0; for (size_t q = 0; q < V; q++) { const int x = (int)(q % TP), y = (int)(q / TP % TP), zz = (int)(q / TP / TP); const float d = zz - mesh_z(&sh, y, x); nw += d >= 0 && d <= 3; }
            for (size_t q = 0; q < V; q++) { const int x = (int)(q % TP), y = (int)(q / TP % TP), zz = (int)(q / TP / TP); const float d = zz - mesh_z(&sh, y, x);
                gv[q] = d >= 0 && d <= 3 ? (float)((1 / (1 + exp(-h2f(lv16[q]))) - sh_ink(&sh, y, x)) / (double)nw) : 0.f; }
            nn_h2d(dgv, gv, V * 4); nn_f32_to_h16(dgv, V, gv16, unet_bwd_gscale()); unet_backward_x(uv, gv16, 1);
            unet_muon(uv, 0.01f, 0.95f, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.f, it);
            const char *e = nn_check(); if (e) { fprintf(stderr, "toy: CUDA %s\n", e); bad = 1; }
        }
        nn_free(dv); nn_free(doff); nn_free(dent); nn_free(dew); isf_lists_free(&L);
    }
    CHECK(!bad, "toy run"); CHECK(t0 > 0 && isfinite(t0), "toy: the trunk got no gradient from the head (%.3g)", t0);
    double sa = 0, sm = 0, sv = 0; const int ne = 6;
    for (int e = 0; e < ne && !bad; e++) {   /* held-out volumes */
        toy_volume(&sh, ct, pn, lab); nn_h2d(dct, ct, V * 4); nn_h2d(dpn, pn, (size_t)n * 6 * 4);
        unet_forward(u, dct, xs, 1); unet_head_ext_fwd(u, did, did + (size_t)C0 * C0, C0, feat);
        nn_isf_resample(feat, ft, C0, TP, TP, TP, dpn, n, K, step, dpatch); if (ctin) nn_isf_resample(dct, ISF_F32, 1, TP, TP, TP, dpn, n, K, step, dpatch + (size_t)C0 * K * n);
        nn_d2h(z, isfhead_forward(h, dpatch), n * 4); sa += auc(z, lab, n);
        for (int p = 0; p < n; p++) { double m = 0; for (int j = 0; j <= 3; j++) { float zz, yy, xx; isf_pos(pn + 6 * p, (float)j, &zz, &yy, &xx); m += tri_ref(ct, TP, TP, TP, zz, yy, xx); } z[p] = (float)m; }
        sm += auc(z, lab, n);   /* (the control: the CT's mean over the inked half, a texture-blind reader) */
        nn_d2h(lv16, unet_forward(uv, dct, xs, 1), V * 2); for (size_t q = 0; q < V; q++) lvf[q] = h2f(lv16[q]);
        for (int p = 0; p < n; p++) { double m = 0; for (int j = 0; j <= 3; j++) { float zz, yy, xx; isf_pos(pn + 6 * p, (float)j, &zz, &yy, &xx); m += tri_ref(lvf, TP, TP, TP, zz, yy, xx); } z[p] = (float)m; }
        sv += auc(z, lab, n);
    }
    sm /= ne; sv /= ne;
    fprintf(stderr, "toy (TP %d, ink %.2f, papyrus %.2f, strokes %.0f-%.0f px, sheet A %.0f-%.0f wl %.0f-%.0f gap %.0f, mesh error %.1f, K %d): surface head %.3f, 3-D voxel head %.3f, CT band mean %.3f\n", TP, g_ink, g_bg, g_sp0, g_sp1, g_a0, g_a1, g_wl0, g_wl1, g_gap, g_moff, K, sa / ne, sv, sm);
    sa /= ne; fprintf(stderr, "toy: held-out AUC %.3f after %d steps, the CT band mean (control) %.3f (trunk gradient norm at step 1 %.3g; head %.1f MB on the device)\n", sa, steps, sm, t0, isfhead_bytes(h) / 1e6);
    CHECK(sa > 0.9, "toy: held-out AUC %.3f <= 0.9", sa); CHECK(fabs(sm - 0.5) < 0.1, "toy: the CT band mean reads the ink (%.3f): not texture-only", sm);
    free(hid); free(ct); free(pn); free(lab); free(w1); free(z); nn_free(did); nn_free(feat); nn_free(gf); nn_free(glz);
    nn_free(dct); nn_free(dpn); nn_free(dlab); nn_free(dw); nn_free(dpatch); nn_free(dgl); isfhead_free(h); unet_free(u); unet_free(uv); free(lv16); free(gv); free(lvf); nn_free(dgv); nn_free(gv16);
}

/* (6) the trainer's path: the feature gradient injected per head slab (unet_head_feat_z / unet_set_head_feat_grad_z, the head's rows + the feature
   rows in one fused backward) against the whole-window injection (unet_set_l0_feat_grad), same weights, same random feature gradient, zero
   logit gradient: the trunk's parameter gradients agree (cosine), the slab features equal the whole-window ones (unet_head_ext_fwd) */
typedef struct { unet *u; const void *g; void *zero, *rows; int D, Z, C; size_t hw; } slab_ctx;
static void slab_cb(void *p) {
    slab_ctx *c = p;
    for (int z0 = 0; z0 < c->D; z0 += c->Z) { const int z1 = z0 + c->Z < c->D ? z0 + c->Z : c->D; const size_t ns = (size_t)(z1 - z0) * c->hw;
        for (int k = 0; k < c->C; k++) nn_d2d((char *)c->rows + (size_t)k * ns * 2, (const char *)c->g + ((size_t)k * c->D * c->hw + (size_t)z0 * c->hw) * 2, ns * 2);
        unet_set_head_feat_grad_z(c->u, c->rows);
        if (unet_head_bwd_z(c->u, z0, z1, (const char *)c->zero + (size_t)z0 * c->hw * 2, z0 == 0)) {   /* (cout 1: the logit gradient's slab rows) */ fprintf(stderr, "FAIL: the combined head backward is not covered\n"); fails++; }
        unet_set_head_feat_grad_z(c->u, nullptr); }
}
static void test_slab_inject(void) {
    nn_set_tf32(1); nn_set_act_bf16(1); nn_set_grad_bf16(1); nn_set_f16(1); nn_set_grad_scale(1024); nn_set_sr(1); nn_set_gn_stored(1);
    nn_set_prec(3); CHECK(!nn_set_prec_policy("all=fp4:fp4:fp4"), "prec policy"); setenv("UFSM_F4_WGRAD", "1", 1);
    /* cases: widths[0] 16 (the fixture's: the combined rows) and 32 (sH's: 1 + 32 rows padded to 40, or the fused head rows + feature gradient,
       lp_set_head_fg, 2026-11-25); MX-fp8 / MX-fp4 gradients (the trainer's --fp4 2); zero or random logit gradient (lg: both terms of gx) */
    static const int cs_[][4] = {{16, 8, 0, 0}, {32, 8, 0, 0}, {32, 8, 1, 0}, {32, 8, 0, 1}, {32, 8, 1, 1}, {32, 4, 0, 1}, {32, 4, 1, 1}};
    double cosv[7] = {0}; float *gprev = nullptr; size_t npv = 0; uint64_t rs_pair = 0;
    for (int ci_ = 0; ci_ < 7; ci_++) {
        if (cs_[ci_][3] && !cs_[ci_][2]) rs_pair = rs; else if (cs_[ci_][3]) rs = rs_pair;   /* (a fused case draws its combined-rows partner's data) */
        const int C0 = cs_[ci_][0], gq = cs_[ci_][1], fgm = cs_[ci_][2], lg = cs_[ci_][3], T = 32, Z = 8; unet_cfg cfg = {3, {C0, 32, 64}, 1, 1, 8, 1, 0};
        unet_set_act_mx4(1); unet_set_grad_mx8(1); unet_set_grad_mx4(gq == 4); lp_set_head_fg(fgm);
        const size_t V = (size_t)T * T * T; const shape5 xs = {1, 1, T, T, T};
        float *x = malloc(V * 4); for (size_t i = 0; i < V; i++) x[i] = frand(); float *dx = dup_h2d(x, V);
        uint16_t *g = malloc(V * C0 * 2); for (size_t i = 0; i < V * C0; i++) g[i] = f2h(1024.f * 1e-3f * frand());
        void *dg = nn_malloc(V * C0 * 2); nn_h2d(dg, g, V * C0 * 2); void *zl = nn_malloc(V * 2); nn_zero(zl, V * 2);
        if (lg) { uint16_t *l = malloc(V * 2); for (size_t i = 0; i < V; i++) l[i] = f2h(1024.f * 1e-3f * frand()); nn_h2d(zl, l, V * 2); free(l); }
        unet *a = unet_create(&cfg); unet_init(a, 9); unet_set_fp16_logits(a, 1);
        unet *b = unet_create(&cfg); unet_init(b, 9); unet_set_fp16_logits(b, 1); unet_set_head_ext(b, 1);
        void *zero = nn_malloc(V * 2); nn_d2d(zero, zl, V * 2); void *rows = nn_malloc((size_t)Z * T * T * C0 * 2);
        slab_ctx sc = {b, dg, zero, rows, T, Z, C0, (size_t)T * T}; unet_set_head_cb(b, slab_cb, &sc);
        const size_t np = unet_nparams(a); float *ga = malloc(np * 4), *gb = malloc(np * 4);
        nn_set_sr_step(500); unet_zero_grad(a); unet_forward(a, dx, xs, 1);
        float *hid = calloc((size_t)C0 * (C0 + 1), 4); for (int c = 0; c < C0; c++) hid[(size_t)c * C0 + c] = 1.f; float *did = dup_h2d(hid, (size_t)C0 * (C0 + 1));
        void *fa = nn_malloc(V * C0 * 2), *fb = nn_malloc(V * C0 * 2);
        CHECK(!unet_head_ext_fwd(a, did, did + (size_t)C0 * C0, C0, fa), "head_ext_fwd");
        nn_set_sr_step(600); unet_set_l0_feat_grad(a, dg); unet_backward_x(a, zl, 1); unet_set_l0_feat_grad(a, nullptr); unet_grad_d2h(a, ga);
        nn_set_sr_step(500); unet_zero_grad(b); unet_forward(b, dx, xs, 1);
        for (int z0 = 0; z0 < T; z0 += Z) { void *q = nn_malloc((size_t)Z * T * T * C0 * 2); CHECK(!unet_head_feat_z(b, z0, z0 + Z, q), "head_feat_z");
            for (int k = 0; k < C0; k++) nn_d2d((char *)fb + ((size_t)k * V + (size_t)z0 * T * T) * 2, (char *)q + (size_t)k * Z * T * T * 2, (size_t)Z * T * T * 2); nn_free(q); }
        nn_set_sr_step(600); unet_backward_x(b, zl, 1); unet_grad_d2h(b, gb);
        const char *e = nn_check(); CHECK(!e, "CUDA %s", e ? e : "");
        uint16_t *ha = malloc(V * C0 * 2), *hb = malloc(V * C0 * 2); nn_d2h(ha, fa, V * C0 * 2); nn_d2h(hb, fb, V * C0 * 2);
        double fd = 0; for (size_t i = 0; i < V * C0; i++) fd = fmax(fd, fabs(h2f(ha[i]) - h2f(hb[i])));
        double ab = 0, aa = 0, bb = 0; for (size_t i = 0; i < np; i++) { ab += (double)ga[i] * gb[i]; aa += (double)ga[i] * ga[i]; bb += (double)gb[i] * gb[i]; }
        const double cs = ab / sqrt(aa * bb + 1e-300), rn = sqrt(bb / (aa + 1e-300));
        fprintf(stderr, "slab injection (widths[0] %d, MX-fp%d gradients, %s, %s logit gradient): features max |slab - whole| %.2e; trunk gradients cosine %.5f, norm ratio %.4f\n",
                C0, gq, fgm ? "fused head + feature gradient" : "combined rows", lg ? "random" : "zero", fd, cs, rn);
        cosv[ci_] = cs;
        CHECK(fd < 1e-3, "slab features differ (%.2e)", fd);
        if (!lg) CHECK(cs > 0.995 && fabs(rn - 1) < 0.02 && aa > 0, "slab injection: cosine %.5f norm ratio %.4f", cs, rn);
        else CHECK(cs > 0.85 && fabs(rn - 1) < 0.03 && aa > 0, "slab injection: cosine %.5f norm ratio %.4f", cs, rn);   /* (with a logit gradient the whole-window path's own
                                                                          head backward differs: 0.956 MX-fp8 / 0.891 MX-fp4 on the combined rows, 11-25) */
        if (lg && fgm && gprev && npv == np) {   /* the fused head rows against the combined rows on the same draws: the same gradient up to the rounding of gx */
            double d1 = 0, a1 = 0, b1 = 0; for (size_t i = 0; i < np; i++) { d1 += (double)gprev[i] * gb[i]; a1 += (double)gprev[i] * gprev[i]; b1 += (double)gb[i] * gb[i]; }
            const double c1 = d1 / sqrt(a1 * b1 + 1e-300), r1 = sqrt(b1 / (a1 + 1e-300));
            fprintf(stderr, "  fused vs combined rows (MX-fp%d gradients): parameter gradients cosine %.5f, norm ratio %.4f\n", gq, c1, r1);
            CHECK(c1 > (gq == 4 ? 0.9 : 0.999) && fabs(r1 - 1) < 0.02, "fused vs combined rows MX-fp%d: cosine %.5f norm ratio %.4f", gq, c1, r1);   /* (fp4: SR noise from the first
                                                                          differing gx element on, 0.951 11-25; fp8: 1.00000) */
        }
        free(gprev); gprev = nullptr; if (lg && !fgm) { gprev = malloc(np * 4); memcpy(gprev, gb, np * 4); npv = np; }
        free(x); free(g); free(ga); free(gb); free(hid); free(ha); free(hb); nn_free(dx); nn_free(dg); nn_free(zl); nn_free(zero); nn_free(rows); nn_free(did); nn_free(fa); nn_free(fb);
        unet_free(a); unet_free(b);
    }
    free(gprev); lp_set_head_fg(-1); unet_set_grad_mx8(1); unet_set_grad_mx4(0);
    CHECK(cosv[4] > cosv[3] - 1e-3, "slab injection MX-fp8: fused cosine to the whole window %.5f vs the combined rows' %.5f", cosv[4], cosv[3]);
    CHECK(cosv[6] > cosv[5] - 0.01, "slab injection MX-fp4: fused cosine to the whole window %.5f vs the combined rows' %.5f", cosv[6], cosv[5]);
}

/* (6) isf_rays_crop (inkperf 2026-11-21: the sampler cuts a surf_only window's rays to the patch's): isf_win_build on the cut list ==
   on the whole list, byte for byte (patch normals / positions, labels, weights, step, spacing, unit), on a sheet of 3 units in shuffled
   order with faceless rays, every-step-th and physical (pvox) patches, aug draws, a centre near the patch border of its unit */
static void test_rays_crop(void) {
    const int D = 64, H = 64, W = 64; int checked = 0; const uint64_t rs0 = rs;   /* (restored at the end: the other tests' draws unchanged) */
    for (int cfgi = 0; cfgi < 6; cfgi++) {
        const int Hp = cfgi & 1 ? 32 : 16, step = cfgi < 2 ? 1 : 2; const float pvox = cfgi >= 4 ? 0.5f : 0.f; const double sp = cfgi == 5 ? 0.21 : 0.37;
        const int G = 220; const size_t cap = (size_t)3 * G * G + 5000; inkray *r = malloc(cap * sizeof *r), *c = malloc(cap * sizeof *c); size_t nr = 0;
        const int lr0 = cfgi == 3 ? 40 : -30, i0 = cfgi == 3 ? 6 : G / 2;   /* (3: the window's centre near the low edge of unit 0's grid) */
        for (int u = 0; u < 3; u++) for (int i = 0; i < G; i++) for (int j = 0; j < G; j++) {
            const double y = 32 + (i - i0) * sp + (u ? 70.0 * u : 0), x = 32 + (j - G / 2) * sp, z = 30 + 6 * sin(0.05 * x) + 3 * cos(0.07 * y) + u * 9;
            if (urand() < 0.08) continue;   /* (holes: unsupervised px) */
            inkray q = {{(float)z, (float)y, (float)x}, {1.f, 0.1f * (float)frand(), 0.1f * (float)frand()}, (uint8_t)(urand() < 0.3 ? 255 : (int)(urand() * 254)), 0, (uint16_t)u, 1.f, 1.f, lr0 + i, j - 50};
            if (urand() < 0.03) q.lr = q.lc = -1;   /* (faceless) */
            r[nr++] = q;
        }
        for (size_t k = nr - 1; k > 0; k--) { const size_t m = (size_t)(urand() * (k + 1)); const inkray t = r[k]; r[k] = r[m]; r[m] = t; }
        memcpy(c, r, nr * sizeof *r);
        const int nc = isf_rays_crop(c, (int)nr, Hp, step, D, H, W, pvox);
        for (unsigned aug = 0; aug < 16; aug += 5) {
            isf_win a = {0}, b = {0};
            const int ra = isf_win_build(&a, r, (int)nr, Hp, step, 5, 1.f, D, H, W, aug, 1, pvox), rb = isf_win_build(&b, c, nc, Hp, step, 5, 1.f, D, H, W, aug, 1, pvox);
            const size_t n = (size_t)Hp * Hp;
            const int same = ra == rb && a.np == b.np && a.step == b.step && a.dvox == b.dvox && a.unit == b.unit && a.frac == b.frac && (!a.np ||
                             (!memcmp(a.pn, b.pn, n * 6 * 4) && !memcmp(a.lab, b.lab, n * 4) && !memcmp(a.wt, b.wt, n * 4) && a.L.nv == b.L.nv && a.L.ne == b.L.ne));
            CHECK(same && a.np > Hp, "rays crop cfg %d aug %u: the patch differs (np %d / %d, step %d / %d, unit %d / %d) or is empty", cfgi, aug, a.np, b.np, a.step, b.step, a.unit, b.unit);
            checked += same;
            isf_win_free(&a); isf_win_free(&b);
        }
        fprintf(stderr, "rays crop cfg %d (Hp %d step %d pvox %.1f): %zu -> %d rays\n", cfgi, Hp, step, pvox, nr, nc);
        CHECK(nc > 0 && (size_t)nc * 4 < nr, "rays crop cfg %d: %zu -> %d (no cut)", cfgi, nr, nc);
        free(r); free(c);
    }
    fprintf(stderr, "rays crop: %d patches identical\n", checked);
    rs = rs0;
}
/* --isf-pvox's spacing estimate (11-21): a flat 64 x 64 ray grid 0.25 window voxel per label px whose positions are quantised to whole
   voxels plus a 1e-3 ramp (the up wraps' teacher rays: neighbours nearly coincide in 3-D). The step for pvox 0.5 must be 2 (d ~0.25), not
   the nearest neighbour's ~500 (d 1e-3); exact positions give 2 too */
static void test_pvox_step(void) {
    const int G = 64, D = 64; inkray *r = calloc((size_t)G * G, sizeof *r);
    for (int q = 0; q < 2; q++) {
        for (int i = 0; i < G; i++) for (int j = 0; j < G; j++) { inkray *k = &r[i * G + j];
            const double y = 16 + 0.25 * i, x = 16 + 0.25 * j;
            k->p[0] = 32; k->p[1] = (float)(q ? floor(y) + 1e-3 * i : y); k->p[2] = (float)(q ? floor(x) + 1e-3 * j : x);
            k->n[0] = 1; k->y = (uint8_t)(((i / 8 + j / 8) & 1) * 255); k->unit = 0; k->w = 1; k->lr = i; k->lc = j; }
        isf_win w = {0}; const int np = isf_win_build(&w, r, G * G, 16, 1, 3, 1.f, D, D, D, 0, 0, 0.5f);
        fprintf(stderr, "pvox step (%s positions): step %d, d %.4f, %d patch px\n", q ? "quantised" : "exact", w.step, w.dvox, np);
        CHECK(w.step == 2 && fabsf(w.dvox - 0.25f) < 0.03f, "pvox 0.5 on %s positions: step %d d %.4f (want 2, 0.25)", q ? "quantised" : "exact", w.step, w.dvox);
        isf_win_free(&w);
    }
    free(r);
}
int main(void) {
    test_rays_crop();
    if (nn_init(0)) return 77;
    g_ink = envf("TIS_INK", g_ink); g_bg = envf("TIS_BG", g_bg); g_sp0 = envf("TIS_SP0", g_sp0); g_sp1 = envf("TIS_SP1", g_sp1);
    g_a0 = envf("TIS_A0", g_a0); g_a1 = envf("TIS_A1", g_a1); g_wl0 = envf("TIS_WL0", g_wl0); g_wl1 = envf("TIS_WL1", g_wl1); g_gap = envf("TIS_GAP", g_gap); g_moff = envf("TIS_MOFF", g_moff); TP = (int)envf("TIS_TP", (float)TP);
    if (getenv("TIS_SEED")) rs = (uint64_t)atoll(getenv("TIS_SEED"));
    if (!getenv("TIS_TOY_ONLY")) { test_resample(); test_2d(); test_2d_tiled(); test_head_fd(4); test_head_fd(5); test_isf_depth(); test_pvox_step(); }
    test_toy();
    if (!getenv("TIS_TOY_ONLY")) test_slab_inject();
    if (fails) { fprintf(stderr, "test_inksurf: %d FAILED\n", fails); return 1; }
    fprintf(stderr, "test_inksurf: PASS\n"); return 0;
}
