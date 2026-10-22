/* src/nn_dinor.cu (the DINO-recipe pieces) against double-precision references:
   - views: every voxel of resized / permuted / flipped crops vs a double trilinear reference; dr_map / dr_unmap round trip;
   - DINO head: input and every parameter gradient of L = sum(logits R) vs central differences of a double forward (frozen last layer too);
   - Sinkhorn-Knopp and the centred softmax vs double references;
   - multi-crop cross-entropy: loss and d / d student logits vs central differences;
   - KoLeo: loss and gradient vs central differences (grouped);
   - pooling forward / backward.
   Pass: max |gpu - ref| / rms(ref) < 2e-3 per tensor (5e-3 for the cross-entropy through the 0.1 temperature). */
#include "dinor.h"
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rs = 777;
static double urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (double)(rs >> 11) * 0x1.0p-53; }
static double nrand(void) { double a = urand() + 1e-300, b = urand(); return sqrt(-2 * log(a)) * cos(6.283185307179586 * b); }
static double rf(double v) { return (double)(float)v; }
static float *dev(const double *h, size_t n) { float *f = malloc(n * 4); for (size_t i = 0; i < n; i++) f[i] = (float)h[i]; float *d = nn_malloc(n * 4); nn_h2d(d, f, n * 4); free(f); return d; }
static double *host(const float *d, size_t n) { float *f = malloc(n * 4); nn_d2h(f, d, n * 4); double *h = malloc(n * 8); for (size_t i = 0; i < n; i++) h[i] = f[i]; free(f); return h; }
static int report(const char *name, const double *a, const double *b, int n, double tol) {
    double rms = 0, mx = 0; for (int i = 0; i < n; i++) { rms += b[i] * b[i]; double e = fabs(a[i] - b[i]); if (e > mx) mx = e; }
    rms = sqrt(rms / n); const double rel = mx / (rms > 1e-30 ? rms : 1e-30); const int bad = !(rel < tol);
    printf("  %-10s rel %.2e (rms %.3e, %d pts)%s\n", name, rel, rms, n, bad ? "  FAIL" : "");
    return bad;
}

/* ---- views ---- */
static int test_views(void) {
    const int Ps = 40, n = 16, nc = 6; printf("views: %d crops of %d^3 from 2 sources of %d^3\n", nc, n, Ps);
    uint8_t *src = malloc((size_t)2 * Ps * Ps * Ps); for (size_t i = 0; i < (size_t)2 * Ps * Ps * Ps; i++) src[i] = (uint8_t)(urand() * 256);
    dr_xf xf[6]; dr_int it[6]; int bad = 0;
    for (int c = 0; c < nc; c++) {
        dr_xf *t = &xf[c]; t->n = n; t->m = 10 + c * 5; t->src = c & 1;
        int p[3] = {0, 1, 2}; for (int k = 2; k > 0; k--) { int j = (int)(urand() * (k + 1)) % (k + 1); int q = p[k]; p[k] = p[j]; p[j] = q; }
        for (int d = 0; d < 3; d++) { t->perm[d] = p[d]; t->flip[d] = urand() < 0.5; t->off[d] = (int)(urand() * (Ps - t->m + 4)) - 2; }
        dr_int *a = &it[c]; a->center = 100.f; a->scale = 40.f; a->gamma = c == 2 ? 0.8f : 1.f; a->lo = 0.f; a->hi = 255.f; a->contrast = c == 3 ? 1.2f : 1.f; a->mean = 0.1f;
        a->bias = c == 4 ? 0.3f : 0.f; a->sigma = 0.f; a->invert = c == 5; a->seed = 1;
        double v[3] = {urand() * n, urand() * n, urand() * n}, s[3], w[3]; dr_map(t, v, s); dr_unmap(t, s, w);
        for (int d = 0; d < 3; d++) bad += fabs(w[d] - v[d]) > 1e-9;
    }
    uint8_t *sd = nn_malloc((size_t)2 * Ps * Ps * Ps); nn_h2d(sd, src, (size_t)2 * Ps * Ps * Ps);
    dr_xf *xd = nn_malloc(sizeof xf); nn_h2d(xd, xf, sizeof xf); dr_int *id = nn_malloc(sizeof it); nn_h2d(id, it, sizeof it);
    const size_t n3 = (size_t)n * n * n; float *x = nn_malloc(nc * n3 * 4);
    dr_crops(sd, Ps, xd, id, nc, n, x);
    double *g = host(x, nc * n3), *r = malloc(nc * n3 * 8);
    for (int c = 0; c < nc; c++) for (size_t j = 0; j < n3; j++) {
        const dr_xf *t = &xf[c]; const dr_int *a = &it[c];
        double v[3] = {(double)(j / (n * n)), (double)((j / n) % n), (double)(j % n)}, s[3]; dr_map(t, v, s);
        int i0[3]; double f[3];
        for (int d = 0; d < 3; d++) { double q = fmin(fmax(s[d], 0), Ps - 1); i0[d] = (int)floor(q); if (i0[d] > Ps - 2) i0[d] = Ps - 2; f[d] = q - i0[d]; }
        double val = 0;
        for (int dz = 0; dz < 2; dz++) for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++)
            val += (dz ? f[0] : 1 - f[0]) * (dy ? f[1] : 1 - f[1]) * (dx ? f[2] : 1 - f[2]) * src[(size_t)t->src * Ps * Ps * Ps + ((size_t)(i0[0] + dz) * Ps + i0[1] + dy) * Ps + i0[2] + dx];
        if (a->gamma != 1) val = a->lo + (a->hi - a->lo) * pow(fmin(fmax((val - a->lo) / (a->hi - a->lo), 0), 1), a->gamma);
        double y = (val - a->center) / a->scale; if (a->contrast != 1) y = a->mean + a->contrast * (y - a->mean); y += a->bias; if (a->invert) y = -y;
        r[c * n3 + j] = y;
    }
    printf("  map round trip %s\n", bad ? "FAIL" : "ok");
    bad += report("crops", g, r, (int)(nc * n3), 1e-4);
    free(src); free(g); free(r); nn_free(sd); nn_free(xd); nn_free(id); nn_free(x);
    return bad;
}

/* ---- head ---- */
static double gelu(double v) { return 0.5 * v * (1 + erf(v / sqrt(2.0))); }
static void ref_head(const dr_head *h, const double *P, const double *x, int R, double *lg) {
    const int H = h->H, B = h->B, K = h->K, C = h->Cin;
    double *h1 = malloc((size_t)R * H * 8), *h2 = malloc((size_t)R * H * 8), *z = malloc((size_t)R * B * 8);
    for (int r = 0; r < R; r++) for (int j = 0; j < H; j++) { double a = P[h->b1 + j]; for (int c = 0; c < C; c++) a += x[r * C + c] * P[h->w1 + (size_t)j * C + c]; h1[r * H + j] = gelu(a); }
    for (int r = 0; r < R; r++) for (int j = 0; j < H; j++) { double a = P[h->b2 + j]; for (int c = 0; c < H; c++) a += h1[r * H + c] * P[h->w2 + (size_t)j * H + c]; h2[r * H + j] = gelu(a); }
    for (int r = 0; r < R; r++) { double s = 0; for (int j = 0; j < B; j++) { double a = P[h->b3 + j]; for (int c = 0; c < H; c++) a += h2[r * H + c] * P[h->w3 + (size_t)j * H + c]; z[r * B + j] = a; s += a * a; }
        s = sqrt(s); for (int j = 0; j < B; j++) z[r * B + j] /= fmax(s, 1e-12); }
    for (int k = 0; k < K; k++) { double s = 0; for (int j = 0; j < B; j++) s += P[h->v + (size_t)k * B + j] * P[h->v + (size_t)k * B + j]; s = P[h->g + k] / fmax(sqrt(s), 1e-12);
        for (int r = 0; r < R; r++) { double a = 0; for (int j = 0; j < B; j++) a += z[r * B + j] * P[h->v + (size_t)k * B + j]; lg[(size_t)r * K + k] = a * s; } }
    free(h1); free(h2); free(z);
}
static int test_head(void) {
    dr_head h; dr_head_layout(&h, 24, 32, 16, 300); const int R = 5; printf("head: Cin %d H %d B %d K %d, %d rows\n", h.Cin, h.H, h.B, h.K, R);
    double *P = malloc(h.n * 8), *x = malloc((size_t)R * h.Cin * 8), *Rm = malloc((size_t)R * h.K * 8);
    for (size_t i = 0; i < h.n; i++) P[i] = rf(0.2 * nrand());
    for (int k = 0; k < h.K; k++) P[h.g + k] = rf(1 + 0.2 * nrand());
    for (int i = 0; i < R * h.Cin; i++) x[i] = rf(nrand());
    for (int i = 0; i < R * h.K; i++) Rm[i] = rf(nrand());
    float *Pd = dev(P, h.n), *xd = dev(x, (size_t)R * h.Cin), *Rd = dev(Rm, (size_t)R * h.K), *Gd = nn_malloc(h.n * 4), *lg = nn_malloc((size_t)R * h.K * 4), *dx = nn_malloc((size_t)R * h.Cin * 4);
    void *wb = nn_malloc(dr_hws_bytes(&h, R)); dr_hws w; dr_hws_carve(&w, &h, R, wb);
    int bad = 0;
    for (int last = 1; last >= 0; last--) {
        nn_zero(Gd, h.n * 4);
        dr_head_fwd(&h, Pd, xd, R, lg, &w);
        if (last) { double *a = host(lg, (size_t)R * h.K), *b = malloc((size_t)R * h.K * 8); ref_head(&h, P, x, R, b); bad += report("forward", a, b, R * h.K, 2e-3); free(a); free(b); }
        dr_head_bwd(&h, Pd, Gd, xd, Rd, R, dx, last, &w);
        double *gx = host(dx, (size_t)R * h.Cin), *gP = host(Gd, h.n);
        struct { const char *nm; size_t off, n; } ts[] = {{"x", 0, (size_t)R * h.Cin}, {"w1", h.w1, (size_t)h.H * h.Cin}, {"b1", h.b1, h.H}, {"w2", h.w2, (size_t)h.H * h.H}, {"b2", h.b2, h.H},
            {"w3", h.w3, (size_t)h.B * h.H}, {"b3", h.b3, h.B}, {"v", h.v, (size_t)h.K * h.B}, {"g", h.g, h.K}};
        const int npts = 16; double a[16], r[16]; double *lgr = malloc((size_t)R * h.K * 8);
        for (size_t k = 0; k < sizeof ts / sizeof ts[0]; k++) {
            if (!last && k >= 7) {   /* frozen last layer: no gradient */
                double s = 0; for (size_t i = 0; i < ts[k].n; i++) s += fabs(gP[ts[k].off + i]);
                printf("  %-10s frozen: |g| %.1e%s\n", ts[k].nm, s, s == 0 ? "" : "  FAIL"); bad += s != 0; continue;
            }
            if (!last) continue;
            for (int i = 0; i < npts; i++) {
                size_t j = (size_t)(urand() * ts[k].n) % ts[k].n; double *v = k == 0 ? x + j : P + ts[k].off + j, v0 = *v, e = 1e-5;
                double lp = 0, lm = 0;
                *v = v0 + e; ref_head(&h, P, x, R, lgr); for (int q = 0; q < R * h.K; q++) lp += lgr[q] * Rm[q];
                *v = v0 - e; ref_head(&h, P, x, R, lgr); for (int q = 0; q < R * h.K; q++) lm += lgr[q] * Rm[q];
                *v = v0; r[i] = (lp - lm) / (2 * e); a[i] = k == 0 ? gx[j] : gP[ts[k].off + j];
            }
            bad += report(ts[k].nm, a, r, npts, 2e-3);
        }
        free(lgr); free(gx); free(gP);
    }
    free(P); free(x); free(Rm); nn_free(Pd); nn_free(xd); nn_free(Rd); nn_free(Gd); nn_free(lg); nn_free(dx); nn_free(wb);
    return bad;
}

/* ---- teacher targets ---- */
static int test_targets(void) {
    const int R = 7, K = 500; printf("teacher targets: %d rows x %d prototypes\n", R, K);
    double *l = malloc((size_t)R * K * 8), *c = malloc(K * 8); for (int i = 0; i < R * K; i++) l[i] = rf(nrand() * 0.3); for (int k = 0; k < K; k++) c[k] = rf(0.05 * nrand());
    float *ld = dev(l, (size_t)R * K), *cd = dev(c, K), *q = nn_malloc((size_t)R * K * 4); void *sc = nn_malloc(dr_sk_scratch(R, K));
    const double tt = 0.05; int bad = 0;
    /* Sinkhorn-Knopp, 3 iterations (upstream sinkhorn_knopp_teacher) */
    dr_teacher_probs(ld, R, K, (float)tt, 3, nullptr, q, sc);
    double *Q = malloc((size_t)K * R * 8), mx = -1e300; for (int i = 0; i < R * K; i++) mx = fmax(mx, l[i]);
    double s = 0; for (int r = 0; r < R; r++) for (int k = 0; k < K; k++) { Q[(size_t)k * R + r] = exp((l[(size_t)r * K + k] - mx) / tt); s += Q[(size_t)k * R + r]; }
    for (int i = 0; i < K * R; i++) Q[i] /= s;
    for (int it = 0; it < 3; it++) {
        for (int k = 0; k < K; k++) { double a = 0; for (int r = 0; r < R; r++) a += Q[(size_t)k * R + r]; for (int r = 0; r < R; r++) Q[(size_t)k * R + r] /= a * K; }
        for (int r = 0; r < R; r++) { double a = 0; for (int k = 0; k < K; k++) a += Q[(size_t)k * R + r]; for (int k = 0; k < K; k++) Q[(size_t)k * R + r] /= a * R; }
    }
    double *ref = malloc((size_t)R * K * 8); for (int r = 0; r < R; r++) for (int k = 0; k < K; k++) ref[(size_t)r * K + k] = Q[(size_t)k * R + r] * R;
    double *g = host(q, (size_t)R * K); bad += report("sinkhorn", g, ref, R * K, 2e-3); free(g);
    dr_teacher_probs(ld, R, K, (float)tt, 0, cd, q, sc);
    for (int r = 0; r < R; r++) { double m = -1e300, z = 0; for (int k = 0; k < K; k++) m = fmax(m, (l[(size_t)r * K + k] - c[k]) / tt);
        for (int k = 0; k < K; k++) z += exp((l[(size_t)r * K + k] - c[k]) / tt - m);
        for (int k = 0; k < K; k++) ref[(size_t)r * K + k] = exp((l[(size_t)r * K + k] - c[k]) / tt - m) / z; }
    g = host(q, (size_t)R * K); bad += report("softmax-c", g, ref, R * K, 2e-3); free(g);
    free(l); free(c); free(Q); free(ref); nn_free(ld); nn_free(cd); nn_free(q); nn_free(sc);
    return bad;
}

/* ---- cross-entropy ---- */
static double ref_ce(const double *ls, int K, double ts, const double *q, const int *ps, const int *pt, const double *pw, int np) {
    double L = 0;
    for (int p = 0; p < np; p++) {
        const double *l = ls + (size_t)ps[p] * K; double m = -1e300, z = 0;
        for (int k = 0; k < K; k++) m = fmax(m, l[k] / ts);
        for (int k = 0; k < K; k++) z += exp(l[k] / ts - m);
        const double lse = m + log(z);
        for (int k = 0; k < K; k++) L -= pw[p] * q[(size_t)pt[p] * K + k] * (l[k] / ts - lse);
    }
    return L;
}
static int test_ce(void) {
    const int Rs = 6, Rt = 2, K = 400, np = 10; const double ts = 0.1; printf("cross-entropy: %d student rows, %d teacher rows, %d pairs, K %d\n", Rs, Rt, np, K);
    double *ls = malloc((size_t)Rs * K * 8), *q = malloc((size_t)Rt * K * 8), pw[10]; int ps[10], pt[10];
    for (int i = 0; i < Rs * K; i++) ls[i] = rf(0.3 * nrand());
    for (int r = 0; r < Rt; r++) { double s = 0; for (int k = 0; k < K; k++) { q[(size_t)r * K + k] = exp(2 * nrand()); s += q[(size_t)r * K + k]; } for (int k = 0; k < K; k++) q[(size_t)r * K + k] = rf(q[(size_t)r * K + k] / s); }
    /* the upstream pairing: global 0 <- teacher 1, global 1 <- teacher 0, locals 2..5 <- both; weight 1 / 10 terms */
    int n = 0; ps[n] = 0; pt[n++] = 1; ps[n] = 1; pt[n++] = 0; for (int l = 2; l < 6; l++) for (int t = 0; t < 2; t++) { ps[n] = l; pt[n++] = t; }
    for (int p = 0; p < np; p++) pw[p] = 0.1;
    float *lsd = dev(ls, (size_t)Rs * K), *qd = dev(q, (size_t)Rt * K), *pwd = dev(pw, np), *dls = nn_malloc((size_t)Rs * K * 4), *out = nn_malloc(4);
    int *psd = nn_malloc(np * 4), *ptd = nn_malloc(np * 4); nn_h2d(psd, ps, np * 4); nn_h2d(ptd, pt, np * 4);
    void *sc = nn_malloc(dr_ce_scratch(Rs));
    dr_ce(lsd, Rs, K, (float)ts, qd, psd, ptd, pwd, np, dls, out, sc);
    float lo; nn_d2h(&lo, out, 4); double L = ref_ce(ls, K, ts, q, ps, pt, pw, np);
    double a1 = lo, r1 = L; int bad = report("loss", &a1, &r1, 1, 1e-4);
    double *g = host(dls, (size_t)Rs * K), a[24], r[24];
    for (int i = 0; i < 24; i++) { size_t j = (size_t)(urand() * Rs * K) % ((size_t)Rs * K); double v0 = ls[j], e = 1e-5;
        ls[j] = v0 + e; double lp = ref_ce(ls, K, ts, q, ps, pt, pw, np); ls[j] = v0 - e; double lm = ref_ce(ls, K, ts, q, ps, pt, pw, np); ls[j] = v0;
        r[i] = (lp - lm) / (2 * e); a[i] = g[j]; }
    bad += report("d logits", a, r, 24, 5e-3);
    free(ls); free(q); free(g); nn_free(lsd); nn_free(qd); nn_free(pwd); nn_free(dls); nn_free(out); nn_free(psd); nn_free(ptd); nn_free(sc);
    return bad;
}

/* ---- KoLeo ---- */
static double ref_koleo(const double *x, int N, int D, int gs, double w) {
    double *xh = malloc((size_t)N * D * 8);
    for (int i = 0; i < N; i++) { double s = 0; for (int c = 0; c < D; c++) s += x[i * D + c] * x[i * D + c]; s = fmax(sqrt(s), 1e-8); for (int c = 0; c < D; c++) xh[i * D + c] = x[i * D + c] / s; }
    double L = 0;
    for (int i = 0; i < N; i++) {
        int g0 = gs ? i / gs * gs : 0, g1 = gs ? g0 + gs : N; if (g1 > N) g1 = N;
        double best = -1e300; int bj = -1;
        for (int j = g0; j < g1; j++) {
            if (j == i) continue;
            double d = 0; for (int c = 0; c < D; c++) d += xh[i * D + c] * xh[j * D + c];
            if (d > best) { best = d; bj = j; }
        }
        double s = 0; for (int c = 0; c < D; c++) { double df = xh[i * D + c] - xh[bj * D + c] + 1e-8; s += df * df; }
        L -= log(sqrt(s) + 1e-8);
    }
    free(xh); return w * L / N;
}
static int test_koleo(void) {
    const int N = 12, D = 20, gs = 6; const double w = 0.1; printf("koleo: %d rows (groups of %d), D %d\n", N, gs, D);
    double *x = malloc(N * D * 8); for (int i = 0; i < N * D; i++) x[i] = rf(nrand());
    float *xd = dev(x, N * D), *gx = nn_malloc(N * D * 4), *out = nn_malloc(4); nn_zero(gx, N * D * 4);
    dr_koleo(xd, N, D, gs, (float)w, gx, out);
    float lo; nn_d2h(&lo, out, 4); double a1 = lo, r1 = ref_koleo(x, N, D, gs, w); int bad = report("loss", &a1, &r1, 1, 1e-4);
    double *g = host(gx, N * D), *r = malloc(N * D * 8);
    for (int j = 0; j < N * D; j++) { double v0 = x[j], e = 1e-6; x[j] = v0 + e; double lp = ref_koleo(x, N, D, gs, w); x[j] = v0 - e; double lm = ref_koleo(x, N, D, gs, w); x[j] = v0; r[j] = (lp - lm) / (2 * e); }
    bad += report("d x", g, r, N * D, 2e-3);
    free(x); free(g); free(r); nn_free(xd); nn_free(gx); nn_free(out);
    return bad;
}

static double ref_koleo_q(const double *x, int N, const double *cd, int M, int D, double w) {
    double L = 0;
    for (int i = 0; i < N; i++) {
        double s = 0; for (int c = 0; c < D; c++) s += x[i * D + c] * x[i * D + c]; s = fmax(sqrt(s), 1e-8);
        double best = -1e300; int bj = 0; double *ch = malloc(D * 8), *bc = malloc(D * 8);
        for (int j = 0; j < M; j++) { double t = 0; for (int c = 0; c < D; c++) t += cd[j * D + c] * cd[j * D + c]; t = fmax(sqrt(t), 1e-8);
            double d = 0; for (int c = 0; c < D; c++) { ch[c] = cd[j * D + c] / t; d += x[i * D + c] / s * ch[c]; } if (d > best) { best = d; bj = j; memcpy(bc, ch, D * 8); } }
        (void)bj; double q = 0; for (int c = 0; c < D; c++) { double df = x[i * D + c] / s - bc[c] + 1e-8; q += df * df; }
        L -= log(sqrt(q) + 1e-8); free(ch); free(bc);
    }
    return w * L / N;
}
static int test_koleo_q(void) {
    const int N = 2, M = 40, D = 24; const double w = 0.1; printf("koleo queue: %d rows vs %d candidates, D %d\n", N, M, D);
    double *x = malloc(N * D * 8), *cd = malloc(M * D * 8); for (int i = 0; i < N * D; i++) x[i] = rf(nrand()); for (int i = 0; i < M * D; i++) cd[i] = rf(nrand());
    float *xd = dev(x, N * D), *cdd = dev(cd, M * D), *gx = nn_malloc(N * D * 4), *out = nn_malloc(4); nn_zero(gx, N * D * 4);
    dr_koleo_q(xd, N, cdd, M, D, (float)w, gx, out);
    float lo; nn_d2h(&lo, out, 4); double a1 = lo, r1 = ref_koleo_q(x, N, cd, M, D, w); int bad = report("loss", &a1, &r1, 1, 1e-4);
    double *g = host(gx, N * D), *r = malloc(N * D * 8);
    for (int j = 0; j < N * D; j++) { double v0 = x[j], e = 1e-6; x[j] = v0 + e; double lp = ref_koleo_q(x, N, cd, M, D, w); x[j] = v0 - e; double lm = ref_koleo_q(x, N, cd, M, D, w); x[j] = v0; r[j] = (lp - lm) / (2 * e); }
    bad += report("d x", g, r, N * D, 2e-3);
    free(x); free(cd); free(g); free(r); nn_free(xd); nn_free(cdd); nn_free(gx); nn_free(out);
    return bad;
}
static int test_muon(int Co, int K) {   /* dr_muon (GEMM Newton-Schulz on the smaller side) against nn_muon (the conv path) */
    printf("muon: [%d][%d]\n", Co, K);
    const size_t n = (size_t)Co * K; double *p = malloc(n * 8), *g = malloc(n * 8), *m = malloc(n * 8);
    for (size_t i = 0; i < n; i++) { p[i] = rf(0.05 * nrand()); g[i] = rf(nrand()); m[i] = rf(0.3 * nrand()); }
    float *p1 = dev(p, n), *g1 = dev(g, n), *m1 = dev(m, n), *p2 = dev(p, n), *m2 = dev(m, n);
    float *w1 = nn_malloc(((size_t)2 * n + 2 * (size_t)Co * Co) * 4), *w2 = nn_malloc(dr_muon_work(Co, K));
    nn_muon(p1, g1, m1, Co, K, 0.02f, 0.95f, 0.04f, w1);
    dr_muon(p2, g1, m2, Co, K, 0.02f, 0.95f, 0.04f, w2);
    double *a = host(p1, n), *b = host(p2, n);
    for (size_t i = 0; i < n; i++) { a[i] -= p[i] * (1 - 0.02 * 0.04); b[i] -= p[i] * (1 - 0.02 * 0.04); }   /* compare the orthogonalised steps */
    int bad = report("step", b, a, (int)n, 2e-2);
    free(p); free(g); free(m); free(a); free(b); nn_free(p1); nn_free(g1); nn_free(m1); nn_free(p2); nn_free(m2); nn_free(w1); nn_free(w2);
    return bad;
}
static int test_pool(void) {
    const int n = 2, S = 11, C = 5, r0 = 3; printf("pool: %d x %d tokens (%d registers), C %d\n", n, S, r0, C);
    double *t = malloc(n * S * C * 8), *g = malloc(n * C * 8); for (int i = 0; i < n * S * C; i++) t[i] = rf(nrand()); for (int i = 0; i < n * C; i++) g[i] = rf(nrand());
    float *td = dev(t, n * S * C), *gd = dev(g, n * C), *cl = nn_malloc(n * C * 4), *gt = nn_malloc(n * S * C * 4); nn_zero(gt, n * S * C * 4);
    dr_pool_fwd(td, n, S, C, r0, cl); dr_pool_bwd(gd, n, S, C, r0, gt);
    double *a = host(cl, n * C), *b = host(gt, n * S * C), *ra = malloc(n * C * 8), *rb = malloc(n * S * C * 8);
    for (int s = 0; s < n; s++) for (int c = 0; c < C; c++) { double m = 0; for (int r = r0; r < S; r++) m += t[(s * S + r) * C + c]; ra[s * C + c] = m / (S - r0);
        for (int r = 0; r < S; r++) rb[(s * S + r) * C + c] = r < r0 ? 0 : g[s * C + c] / (S - r0); }
    int bad = report("forward", a, ra, n * C, 1e-5) + report("backward", b, rb, n * S * C, 1e-5);
    free(t); free(g); free(a); free(b); free(ra); free(rb); nn_free(td); nn_free(gd); nn_free(cl); nn_free(gt);
    return bad;
}

int main(void) {
    if (nn_init(0)) { printf("SKIP: no GPU\n"); return 0; }
    int bad = test_views() + test_head() + test_targets() + test_ce() + test_koleo() + test_koleo_q() + test_muon(64, 24) + test_muon(24, 64) + test_pool();
    printf(bad ? "test_dinor: %d FAILED\n" : "test_dinor: all passed\n", bad);
    return bad != 0;
}
