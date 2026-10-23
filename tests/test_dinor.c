/* src/nn_dinor.cu (the DINO-recipe pieces) against double-precision references:
   - views: every voxel of resized / permuted / flipped crops vs a double trilinear reference; dr_map / dr_unmap round trip;
   - DINO head: input and every parameter gradient of L = sum(logits R) vs central differences of a double forward (frozen last layer too);
   - Sinkhorn-Knopp and the centred softmax vs double references; a sharp teacher row stays sharp through the centring at the recipe's sizes;
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

/* sharpness at the recipe's sizes (batch 1: 2 CLS rows, a 62-row queue, 32768 prototypes): teacher rows with one clear prototype (cosine logit
   0.5 over N(0, 0.05) noise, tt 0.04). The centred softmax (DINOv1; centre after 20 EMA updates of such rows) keeps the row sharp; Sinkhorn-Knopp
   over the 64 rows flattens it (the equipartition gives every one of the 32768 columns 1/K of the mass) -- the recipe runs' H(t) = ln K */
static double row_entropy(const float *qd, int r, int K) {
    double *h = host(qd + (size_t)r * K, K), z = 0, H = 0; for (int k = 0; k < K; k++) z += h[k];
    for (int k = 0; k < K; k++) { const double a = h[k] / z; if (a > 0) H -= a * log(a); } free(h); return H;
}
static int test_sharp(void) {
    const int K = 32768, R = 64; const float tt = 0.04f; printf("teacher sharpness: %d rows x %d prototypes, tt %.2f\n", R, K, tt);
    double *l = malloc((size_t)R * K * 8); for (int r = 0; r < R; r++) { for (int k = 0; k < K; k++) l[(size_t)r * K + k] = 0.05 * nrand(); l[(size_t)r * K + (int)(urand() * K)] = 0.5; }
    float *ld = dev(l, (size_t)R * K), *q = nn_malloc((size_t)R * K * 4), *c = nn_malloc((size_t)K * 4); void *sc = nn_malloc(dr_sk_scratch(R, K));
    nn_zero(c, (size_t)K * 4);
    for (int i = 0; i < 20; i++) dr_center_update(c, ld + (size_t)(2 + 2 * i % (R - 2)) * K, 2, K, 0.9f);   /* (the centre from other rows' history) */
    dr_teacher_probs(ld, 2, K, tt, 0, c, q, sc);
    const double Hc = row_entropy(q, 0, K);
    dr_teacher_probs(ld, R, K, tt, 3, nullptr, q, sc);
    const double Hs = row_entropy(q, 0, K), lnK = log((double)K);
    const int bad = !(Hc < 0.25 * lnK);
    printf("  centred softmax H %.2f, Sinkhorn over %d rows H %.2f (ln K %.2f)%s\n", Hc, R, Hs, lnK, bad ? "  FAIL" : "");
    free(l); nn_free(ld); nn_free(q); nn_free(c); nn_free(sc);
    return bad;
}

/* the DINO loop on toy data at batch 1 (the recipe's head, centring, cross-entropy pairing, Muon / AdamW steps and EMA teacher, no trunk):
   16 clusters in 64-d, each step one cluster drawn and two noisy views of it (2 teacher rows); a correctly wired loop has to discover the
   clusters: the teacher's target entropy falls well below ln K and the two views' teacher argmaxes agree. plr: the prototype lr; returns
   the mean H(t) of the last 200 steps (agree: argmax agreement over them) */
static double toy_dino(float plr, int steps, double *agree, int *distinct) {
    const int Cin = 64, HH = 256, HB = 64, K = 4096, NC = 16; const float ts = 0.1f, tt = 0.04f;
    dr_head h; dr_head_layout(&h, Cin, HH, HB, K);
    double *b = calloc(h.n, 8);
    for (size_t i = h.w1; i < h.b1; i++) b[i] = 0.02 * nrand();
    for (size_t i = h.w2; i < h.b2; i++) b[i] = 0.02 * nrand();
    for (size_t i = h.w3; i < h.b3; i++) b[i] = 0.02 * nrand();
    for (size_t i = h.v; i < h.g; i++) b[i] = (2 * urand() - 1) / sqrt((double)HB);
    for (size_t i = h.g; i < h.n; i++) b[i] = 1;
    float *P = dev(b, h.n), *Pt = dev(b, h.n), *G = nn_malloc(h.n * 4), *M = nn_malloc(h.n * 4), *V = nn_malloc(h.n * 4), *MO = nn_malloc(h.n * 4); free(b);
    nn_zero(M, h.n * 4); nn_zero(V, h.n * 4); nn_zero(MO, h.n * 4);
    size_t wb = dr_muon_work(HH, HH); if (dr_muon_work(HH, Cin) > wb) wb = dr_muon_work(HH, Cin); float *work = nn_malloc(wb);
    dr_hws w; void *wbuf = nn_malloc(dr_hws_bytes(&h, 2)); dr_hws_carve(&w, &h, 2, wbuf);
    double *cen = malloc((size_t)NC * Cin * 8); for (int i = 0; i < NC * Cin; i++) cen[i] = nrand();
    float *x = nn_malloc(2 * Cin * 4), *tl = nn_malloc(2 * K * 4), *tq = nn_malloc(2 * K * 4), *sl = nn_malloc(K * 4), *dsl = nn_malloc(K * 4), *c = nn_malloc(K * 4), *out = nn_malloc(64 * 4);
    void *sc = nn_malloc(dr_sk_scratch(2, K)), *ces = nn_malloc(dr_ce_scratch(1)); nn_zero(c, K * 4);
    int *pS = nn_malloc(8), *pT = nn_malloc(8); float *pW = nn_malloc(8);
    double Hsum = 0, ag = 0; int n = 0; uint8_t *seen = calloc(K, 1); *distinct = 0;
    for (int st = 1; st <= steps; st++) {
        const int k = (int)(urand() * NC); double xv[2 * 64];
        for (int v = 0; v < 2; v++) for (int i = 0; i < Cin; i++) xv[v * Cin + i] = cen[k * Cin + i] + 0.3 * nrand();
        float *xd = dev(xv, 2 * Cin); nn_d2d(x, xd, 2 * Cin * 4); nn_free(xd);
        dr_head_fwd(&h, Pt, x, 2, tl, &w); dr_teacher_probs(tl, 2, K, tt, 0, c, tq, sc); dr_center_update(c, tl, 2, K, 0.9f);
        if (st > steps - 200) {
            double *q = host(tq, 2 * K); int a0 = 0, a1 = 0; double H = 0;
            for (int j = 0; j < K; j++) { if (q[j] > 0) H -= q[j] * log(q[j]); if (q[j] > q[a0]) a0 = j; if (q[K + j] > q[K + a1]) a1 = j; }
            Hsum += H; ag += a0 == a1; n++; free(q); if (!seen[a0]) { seen[a0] = 1; (*distinct)++; }
        }
        nn_zero(G, h.n * 4);
        for (int v = 0; v < 2; v++) {   /* student view v against the teacher's other view */
            dr_head_fwd(&h, P, x + v * Cin, 1, sl, &w);
            const int ps = 0, pt = 1 - v; const float pw = 0.5f; nn_h2d(pS, &ps, 4); nn_h2d(pT, &pt, 4); nn_h2d(pW, &pw, 4);
            dr_ce(sl, 1, K, ts, tq, pS, pT, pW, 1, dsl, out, ces);
            dr_head_bwd(&h, P, G, x + v * Cin, dsl, 1, nullptr, 1, &w);
        }
        dr_muon(P + h.w1, G + h.w1, MO + h.w1, HH, Cin, 0.02f, 0.95f, 0.f, work);
        dr_muon(P + h.w2, G + h.w2, MO + h.w2, HH, HH, 0.02f, 0.95f, 0.f, work);
        dr_muon(P + h.w3, G + h.w3, MO + h.w3, HB, HH, 0.02f, 0.95f, 0.f, work);
        dr_adamw(P + h.b1, G + h.b1, M + h.b1, V + h.b1, h.w2 - h.b1, 5e-4f, 0.9f, 0.999f, 1e-8f, 0.f, st);
        dr_adamw(P + h.b2, G + h.b2, M + h.b2, V + h.b2, h.w3 - h.b2, 5e-4f, 0.9f, 0.999f, 1e-8f, 0.f, st);
        dr_adamw(P + h.b3, G + h.b3, M + h.b3, V + h.b3, h.v - h.b3, 5e-4f, 0.9f, 0.999f, 1e-8f, 0.f, st);
        if (plr > 0) dr_adamw(P + h.v, G + h.v, M + h.v, V + h.v, h.g - h.v, plr, 0.9f, 0.999f, 1e-8f, 0.f, st);
        dr_ema(Pt, P, h.n, 0.99f);
    }
    nn_free(P); nn_free(Pt); nn_free(G); nn_free(M); nn_free(V); nn_free(MO); nn_free(work); nn_free(wbuf); nn_free(x); nn_free(tl); nn_free(tq); nn_free(sl); nn_free(dsl);
    nn_free(c); nn_free(out); nn_free(sc); nn_free(ces); nn_free(pS); nn_free(pT); nn_free(pW); free(cen);
    free(seen); *agree = ag / n; return Hsum / n;
}
static int test_toy_dino(void) {
    printf("DINO loop on toy clusters (batch 1, K 4096, centring, ln K %.2f)\n", log(4096.0));
    int bad = 0; const float plrs[3] = {0.f, 4e-6f, 5e-4f};
    for (int i = 0; i < 3; i++) {
        double ag; int nd; const double H = toy_dino(plrs[i], 3000, &ag, &nd);
        const int b = i < 2 && !(H < 0.5 * log(4096.0) && ag > 0.8 && nd >= 8);   /* (no collapse: >= 8 of the 16 clusters on their own prototypes; 5e-4 reported only) */
        printf("  prototype lr %-7g: teacher H %.2f, view argmax agreement %.2f, distinct argmaxes %d (16 clusters)%s\n", plrs[i], H, ag, nd, b ? "  FAIL" : ""); bad += b;
    }
    return bad;
}

/* InfoNCE: loss / accuracy / d Q vs a double reference and central differences */
static double ref_nce(const double *Q, int N, const double *K, int M, const int *pos, int D, double tau, int *hits) {
    double L = 0; *hits = 0;
    for (int i = 0; i < N; i++) { double m = -1e300, z = 0; int am = 0; double s[64];
        for (int k = 0; k < M; k++) { double v = 0; for (int d = 0; d < D; d++) v += Q[i * D + d] * K[k * D + d]; s[k] = v / tau; if (s[k] > m) { m = s[k]; am = k; } }
        for (int k = 0; k < M; k++) z += exp(s[k] - m);
        L += (m + log(z) - s[pos[i]]) / N; *hits += am == pos[i]; }
    return L;
}
static int test_nce(void) {
    const int N = 5, M = 40, D = 16; const double tau = 0.2, w = 0.7; printf("InfoNCE: %d anchors, %d keys, D %d\n", N, M, D); int bad = 0;
    double Q[5 * 16], K[40 * 16]; int pos[5] = {0, 3, 7, 1, 30};
    for (int i = 0; i < N * D; i++) Q[i] = rf(nrand() * 0.3);
    for (int i = 0; i < M * D; i++) K[i] = rf(nrand() * 0.3);
    float *qd = dev(Q, N * D), *kd = dev(K, M * D), *gd = nn_malloc(N * D * 4), *od = nn_malloc(8); int *pd = nn_malloc(sizeof pos); nn_h2d(pd, pos, sizeof pos);
    void *sc = nn_malloc(dr_nce_scratch(N, M)); nn_zero(gd, N * D * 4); nn_zero(od, 8);
    dr_infonce(qd, N, kd, M, pd, D, (float)tau, (float)w, gd, od, sc);
    int hits; const double L = ref_nce(Q, N, K, M, pos, D, tau, &hits);
    double *o = host(od, 2), r2[2] = {w * L, (double)hits / N}; bad += report("loss/acc", o, r2, 2, 2e-3); free(o);
    double gr[5 * 16]; for (int i = 0; i < N * D; i++) { const double e = 1e-5, x = Q[i]; int h; Q[i] = x + e; const double lp = ref_nce(Q, N, K, M, pos, D, tau, &h); Q[i] = x - e; const double lm = ref_nce(Q, N, K, M, pos, D, tau, &h); Q[i] = x; gr[i] = w * (lp - lm) / (2 * e); }
    double *g = host(gd, N * D); bad += report("d Q", g, gr, N * D, 2e-3); free(g);
    nn_free(qd); nn_free(kd); nn_free(gd); nn_free(od); nn_free(pd); nn_free(sc);
    return bad;
}
/* the queue-based self-supervision on toy instances (batch-1 MoCo / dense cross-view InfoNCE): random unit points in 64-d, two views =
   point + N(0, 0.3) noise each; student projection (Linear 64 -> 256 -> GELU -> 64, unit) on view a, the EMA teacher's on view b as the positive,
   negatives = the other anchors' keys + a FIFO queue of earlier teacher keys (Q keys). NB anchors per step (1: CLS-style, 64: dense-style);
   returns the top-1 accuracy over the last 200 steps (chance 1 / (NB + Q)) */
static double toy_nce(int NB, int Q, int steps, double *loss) {
    const int Cin = 64, H = 256, O = 64; dr_mlp2 m; dr_mlp2_layout(&m, Cin, H, O);
    double *b = calloc(m.n, 8); for (size_t i = m.w1; i < m.b1; i++) b[i] = nrand() / sqrt((double)Cin); for (size_t i = m.w2; i < m.b2; i++) b[i] = nrand() / sqrt((double)H);
    float *P = dev(b, m.n), *Pt = dev(b, m.n), *G = nn_malloc(m.n * 4), *Mo = nn_malloc(m.n * 4), *Ma = nn_malloc(m.n * 4), *Va = nn_malloc(m.n * 4); free(b);
    nn_zero(Mo, m.n * 4); nn_zero(Ma, m.n * 4); nn_zero(Va, m.n * 4);
    size_t wb = dr_muon_work(H, Cin); if (dr_muon_work(O, H) > wb) wb = dr_muon_work(O, H); float *work = nn_malloc(wb);
    void *ws = nn_malloc(dr_mlp2_ws(&m, NB)), *wt = nn_malloc(dr_mlp2_ws(&m, NB));
    float *xa = nn_malloc((size_t)NB * Cin * 4), *xb = nn_malloc((size_t)NB * Cin * 4), *q = nn_malloc((size_t)NB * O * 4), *kb = nn_malloc((size_t)(NB + Q) * O * 4), *gq = nn_malloc((size_t)NB * O * 4), *out = nn_malloc(8);
    int *pos = nn_malloc(NB * 4); { int *h = malloc(NB * 4); for (int i = 0; i < NB; i++) h[i] = i; nn_h2d(pos, h, NB * 4); free(h); }
    void *sc = nn_malloc(dr_nce_scratch(NB, NB + Q)); int qf = 0, qp = 0; double acc = 0, ls = 0; int n = 0;
    double *ha = malloc((size_t)NB * Cin * 8), *hb = malloc((size_t)NB * Cin * 8);
    for (int st = 1; st <= steps; st++) {
        for (int i = 0; i < NB; i++) { double c[64], z = 0; for (int d = 0; d < Cin; d++) { c[d] = nrand(); z += c[d] * c[d]; } z = sqrt(z);
            for (int d = 0; d < Cin; d++) { ha[i * Cin + d] = c[d] / z * 4 + 0.3 * nrand(); hb[i * Cin + d] = c[d] / z * 4 + 0.3 * nrand(); } }
        float *t = dev(ha, (size_t)NB * Cin); nn_d2d(xa, t, (size_t)NB * Cin * 4); nn_free(t); t = dev(hb, (size_t)NB * Cin); nn_d2d(xb, t, (size_t)NB * Cin * 4); nn_free(t);
        dr_mlp2_fwd(&m, Pt, xb, NB, kb, wt);   /* in-batch keys first, the queue rows after them */
        dr_mlp2_fwd(&m, P, xa, NB, q, ws);
        nn_zero(gq, (size_t)NB * O * 4); nn_zero(out, 8); nn_zero(G, m.n * 4);
        dr_infonce(q, NB, kb, NB + qf, pos, O, 0.2f, 1.f, gq, out, sc);
        dr_mlp2_bwd(&m, P, G, xa, q, gq, NB, nullptr, ws);
        dr_muon(P + m.w1, G + m.w1, Mo + m.w1, H, Cin, 0.02f, 0.95f, 0.f, work);
        dr_muon(P + m.w2, G + m.w2, Mo + m.w2, O, H, 0.02f, 0.95f, 0.f, work);
        dr_adamw(P + m.b1, G + m.b1, Ma + m.b1, Va + m.b1, H, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.f, st);
        dr_adamw(P + m.b2, G + m.b2, Ma + m.b2, Va + m.b2, O, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.f, st);
        dr_ema(Pt, P, m.n, 0.99f);
        for (int i = 0; i < NB; i++) { nn_d2d(kb + (size_t)(NB + qp) * O, kb + (size_t)i * O, O * 4); qp = (qp + 1) % Q; if (qf < Q) qf++; }   /* (keys enqueued) */
        if (st > steps - 200) { double *o = host(out, 2); ls += o[0]; acc += o[1]; n++; free(o); }
    }
    nn_free(P); nn_free(Pt); nn_free(G); nn_free(Mo); nn_free(Ma); nn_free(Va); nn_free(work); nn_free(ws); nn_free(wt); nn_free(xa); nn_free(xb); nn_free(q); nn_free(kb); nn_free(gq); nn_free(out); nn_free(pos); nn_free(sc); free(ha); free(hb);
    *loss = ls / n; return acc / n;
}
static int test_toy_nce(void) {
    printf("queue InfoNCE on toy instances (tau 0.2)\n"); int bad = 0;
    const int nb[2] = {1, 64}, qn[2] = {4096, 4096};
    for (int k = 0; k < 2; k++) { double L; const double a = toy_nce(nb[k], qn[k], 1500, &L); const double ch = 1.0 / (nb[k] + qn[k]); const int b = !(a > 0.5);
        printf("  %2d anchors / step, queue %d: top-1 %.3f (chance %.5f), loss %.3f (ln(keys) %.2f)%s\n", nb[k], qn[k], a, ch, L, log((double)nb[k] + qn[k]), b ? "  FAIL" : ""); bad += b; }
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
/* double reference of the Muon step (nesterov momentum, Frobenius normalisation, 5 quintic Newton-Schulz iterations on X X^T, the
   sqrt(max(1, Co / K)) scale): the update direction O for a tall [Co][K] matrix (the prototype layer's shape) */
static int test_muon_ref(int Co, int K) {
    printf("muon double reference: [%d][%d]\n", Co, K);
    const size_t n = (size_t)Co * K; double *g = malloc(n * 8), *m = malloc(n * 8), *X = malloc(n * 8), *Y = malloc(n * 8), *A = malloc((size_t)Co * Co * 8), *B = malloc((size_t)Co * Co * 8);
    for (size_t i = 0; i < n; i++) { g[i] = rf(nrand()); m[i] = rf(0.3 * nrand()); }
    double ss = 0; for (size_t i = 0; i < n; i++) { const double mm = 0.95 * m[i] + g[i]; X[i] = g[i] + 0.95 * mm; ss += X[i] * X[i]; }
    for (size_t i = 0; i < n; i++) X[i] /= sqrt(ss) + 1e-7;
    for (int it = 0; it < 5; it++) {
        for (int i = 0; i < Co; i++) for (int j = 0; j < Co; j++) { double a = 0; for (int k = 0; k < K; k++) a += X[(size_t)i * K + k] * X[(size_t)j * K + k]; A[(size_t)i * Co + j] = a; }
        for (int i = 0; i < Co; i++) for (int j = 0; j < Co; j++) { double a = 0; for (int k = 0; k < Co; k++) a += A[(size_t)i * Co + k] * A[(size_t)k * Co + j]; B[(size_t)i * Co + j] = -4.7750 * A[(size_t)i * Co + j] + 2.0315 * a; }
        for (int i = 0; i < Co; i++) for (int k = 0; k < K; k++) { double a = 3.4445 * X[(size_t)i * K + k]; for (int j = 0; j < Co; j++) a += B[(size_t)i * Co + j] * X[(size_t)j * K + k]; Y[(size_t)i * K + k] = a; }
        double *t = X; X = Y; Y = t;
    }
    const double sc = sqrt(fmax(1.0, (double)Co / K)), lr = 0.02;
    double *z = calloc(n, 8); float *pd = dev(z, n), *gd = dev(g, n), *md = dev(m, n), *w = nn_malloc(dr_muon_work(Co, K));
    dr_muon(pd, gd, md, Co, K, (float)lr, 0.95f, 0.f, w);
    double *got = host(pd, n); for (size_t i = 0; i < n; i++) { got[i] /= -lr * sc; }
    int bad = report("direction", got, X, (int)n, 2e-3);
    free(g); free(m); free(X); free(Y); free(A); free(B); free(z); free(got); nn_free(pd); nn_free(gd); nn_free(md); nn_free(w);
    return bad;
}
/* ---- object term ---- */
static double sp(double x) { return x > 30 ? x : log1p(exp(x)); }
static double ref_obj(const double *S, const double *T, const int *obj, const uint8_t *vs, const uint8_t *vt, int N, int D, double t, double mg, double w) {
    double L = 0; int na = 0;
    for (int i = 0; i < N; i++) {
        if (!vs[i] || obj[i] <= 0) continue;
        double s[64]; int np = 0, nps = 0, nn = 0; double m = -1e300;
        for (int j = 0; j < N; j++) { double a = 0; for (int c = 0; c < D; c++) a += S[i * D + c] * T[j * D + c]; s[j] = a; }
        for (int j = 0; j < N; j++) { if (!vt[j] || obj[j] <= 0) continue; if (obj[j] == obj[i]) { np++; if (j != i) nps++; } else { nn++; m = fmax(m, s[j] / t); } }
        if (!np || !nn) continue;
        double z = 0; for (int j = 0; j < N; j++) if (vt[j] && obj[j] > 0 && obj[j] != obj[i]) z += exp(s[j] / t - m);
        const double lse = m + log(z); const int ex = nps > 0; double a = 0; int k = 0;
        for (int j = 0; j < N; j++) { if (!vt[j] || obj[j] != obj[i] || (ex && j == i)) continue; a += sp(lse - fmin(s[j], mg) / t); k++; }
        L += a / k; na++;
    }
    return na ? w * L / na : 0;
}
static int test_obj(void) {
    const int N = 40, D = 16; const double t = 0.15, mg = 0.75, w = 0.1; printf("object term: %d points, D %d\n", N, D);
    double *S = malloc(N * D * 8), *T = malloc(N * D * 8); int obj[40]; uint8_t vs[40], vt[40];
    for (int i = 0; i < N; i++) { obj[i] = i < 36 ? 1 + i % 5 : 0; vs[i] = urand() < 0.9; vt[i] = urand() < 0.9;
        double a = 0, b = 0; for (int c = 0; c < D; c++) { S[i * D + c] = nrand() + (c == obj[i] ? 2 : 0); T[i * D + c] = nrand() + (c == obj[i] ? 2 : 0); a += S[i * D + c] * S[i * D + c]; b += T[i * D + c] * T[i * D + c]; }
        for (int c = 0; c < D; c++) { S[i * D + c] = rf(S[i * D + c] / sqrt(a)); T[i * D + c] = rf(T[i * D + c] / sqrt(b)); } }
    float *Sd = dev(S, N * D), *Td = dev(T, N * D), *gS = nn_malloc(N * D * 4), *out = nn_malloc(8); nn_zero(gS, N * D * 4); nn_zero(out, 8);
    int *od = nn_malloc(N * 4); nn_h2d(od, obj, N * 4); uint8_t *vsd = nn_malloc(N), *vtd = nn_malloc(N); nn_h2d(vsd, vs, N); nn_h2d(vtd, vt, N);
    dr_obj(Sd, Td, od, vsd, vtd, N, D, (float)t, (float)mg, (float)w, gS, out);
    float o[2]; nn_d2h(o, out, 8); double a1 = o[0], r1 = ref_obj(S, T, obj, vs, vt, N, D, t, mg, w); int bad = report("loss", &a1, &r1, 1, 1e-4);
    double *g = host(gS, N * D), *r = malloc(N * D * 8);
    for (int j = 0; j < N * D; j++) { double v0 = S[j], e = 1e-6; S[j] = v0 + e; double lp = ref_obj(S, T, obj, vs, vt, N, D, t, mg, w); S[j] = v0 - e; double lm = ref_obj(S, T, obj, vs, vt, N, D, t, mg, w); S[j] = v0; r[j] = (lp - lm) / (2 * e); }
    bad += report("d S", g, r, N * D, 2e-3);
    free(S); free(T); free(g); free(r); nn_free(Sd); nn_free(Td); nn_free(gS); nn_free(out); nn_free(od); nn_free(vsd); nn_free(vtd);
    return bad;
}
static double ref_ink(const double *A, const int *aid, int Na, const double *Kp, const int *kid, int np, const double *Kn, int nn, int D, double T, double mg, double w) {
    double L = 0;
    for (int i = 0; i < Na; i++) {
        double sn[64], spv[64], mx = -1e300, mn = 0;
        for (int j = 0; j < nn; j++) { double a = 0; for (int c = 0; c < D; c++) a += A[i * D + c] * Kn[j * D + c]; sn[j] = a; mx = fmax(mx, a); mn += a / T; }
        mn /= nn; double z = 0; for (int j = 0; j < nn; j++) z += exp(sn[j] / T - mx / T);
        const double lmen = mx / T + log(z) - log((double)nn), cap = fmax(mg, mx + T);
        int other = 0; for (int j = 0; j < np; j++) other += !(kid[j] == aid[i] && aid[i] >= 0);
        int cnt = 0; double pm = 0, pmx = -1e300;
        for (int j = 0; j < np; j++) { double a = 0; for (int c = 0; c < D; c++) a += A[i * D + c] * Kp[j * D + c]; spv[j] = fmin(a, cap) / T; }
        for (int j = 0; j < np; j++) { if (other && kid[j] == aid[i] && aid[i] >= 0) continue; pm += spv[j]; pmx = fmax(pmx, spv[j]); cnt++; }
        pm /= cnt; double zp = 0; for (int j = 0; j < np; j++) { if (other && kid[j] == aid[i] && aid[i] >= 0) continue; zp += exp(spv[j] - pmx); }
        const double lmep = pmx + log(zp) - log((double)cnt);
        L += 0.5 * (sp(mn - pm) + sp(lmen - lmep));
    }
    return w * L / Na;
}
static int test_ink(void) {
    const int Na = 6, np = 20, nn = 24, D = 12; const double T = 0.15, mg = 0.75, w = 0.1; printf("ink term: %d anchors, %d pull / %d push keys, D %d\n", Na, np, nn, D);
    double *A = malloc(Na * D * 8), *Kp = malloc(np * D * 8), *Kn = malloc(nn * D * 8); int aid[6], kid[20];
#define UNIT(X, n) for (int i_ = 0; i_ < (n); i_++) { double s_ = 0; for (int c = 0; c < D; c++) { X[i_ * D + c] = nrand() + (c == 0 ? 1.5 : 0); s_ += X[i_ * D + c] * X[i_ * D + c]; } for (int c = 0; c < D; c++) X[i_ * D + c] = rf(X[i_ * D + c] / sqrt(s_)); }
    UNIT(A, Na) UNIT(Kp, np) UNIT(Kn, nn)
#undef UNIT
    for (int i = 0; i < Na; i++) aid[i] = i;
    for (int j = 0; j < np; j++) kid[j] = j < 8 ? j : -1;
    float *Ad = dev(A, Na * D), *Kpd = dev(Kp, np * D), *Knd = dev(Kn, nn * D), *gA = nn_malloc(Na * D * 4), *out = nn_malloc(4); nn_zero(gA, Na * D * 4); nn_zero(out, 4);
    int *aidd = nn_malloc(Na * 4), *kidd = nn_malloc(np * 4); nn_h2d(aidd, aid, Na * 4); nn_h2d(kidd, kid, np * 4);
    dr_ink(Ad, aidd, Na, Kpd, kidd, np, Knd, nn, D, (float)T, (float)mg, (float)w, gA, out);
    float lo; nn_d2h(&lo, out, 4); double a1 = lo, r1 = ref_ink(A, aid, Na, Kp, kid, np, Kn, nn, D, T, mg, w); int bad = report("loss", &a1, &r1, 1, 1e-4);
    double *g = host(gA, Na * D), *r = malloc(Na * D * 8);
    for (int j = 0; j < Na * D; j++) { double v0 = A[j], e = 1e-6; A[j] = v0 + e; double lp = ref_ink(A, aid, Na, Kp, kid, np, Kn, nn, D, T, mg, w); A[j] = v0 - e; double lm = ref_ink(A, aid, Na, Kp, kid, np, Kn, nn, D, T, mg, w); A[j] = v0; r[j] = (lp - lm) / (2 * e); }
    bad += report("d A", g, r, Na * D, 2e-3);
    free(A); free(Kp); free(Kn); free(g); free(r); nn_free(Ad); nn_free(Kpd); nn_free(Knd); nn_free(gA); nn_free(out); nn_free(aidd); nn_free(kidd);
    return bad;
}
static void ref_mlp2(const dr_mlp2 *m, const double *P, const double *x, int R, double *y) {
    double *h = malloc((size_t)m->H * 8), *z = malloc((size_t)m->O * 8);
    for (int r = 0; r < R; r++) {
        for (int j = 0; j < m->H; j++) { double a = P[m->b1 + j]; for (int c = 0; c < m->Cin; c++) a += x[r * m->Cin + c] * P[m->w1 + (size_t)j * m->Cin + c]; h[j] = gelu(a); }
        double s = 0; for (int j = 0; j < m->O; j++) { double a = P[m->b2 + j]; for (int c = 0; c < m->H; c++) a += h[c] * P[m->w2 + (size_t)j * m->H + c]; z[j] = a; s += a * a; }
        for (int j = 0; j < m->O; j++) y[r * m->O + j] = z[j] / sqrt(s);
    }
    free(h); free(z);
}
static int test_mlp2(void) {
    dr_mlp2 m; dr_mlp2_layout(&m, 12, 20, 8); const int R = 4; printf("region head: %d -> %d -> %d, %d rows\n", m.Cin, m.H, m.O, R);
    double *P = malloc(m.n * 8), *x = malloc(R * m.Cin * 8), *Rm = malloc(R * m.O * 8), *y = malloc(R * m.O * 8);
    for (size_t i = 0; i < m.n; i++) P[i] = rf(0.3 * nrand());
    for (int i = 0; i < R * m.Cin; i++) x[i] = rf(nrand());
    for (int i = 0; i < R * m.O; i++) Rm[i] = rf(nrand());
    float *Pd = dev(P, m.n), *xd = dev(x, R * m.Cin), *Rd = dev(Rm, R * m.O), *yd = nn_malloc(R * m.O * 4), *G = nn_malloc(m.n * 4), *dx = nn_malloc(R * m.Cin * 4); void *ws = nn_malloc(dr_mlp2_ws(&m, R));
    nn_zero(G, m.n * 4); nn_zero(dx, R * m.Cin * 4);
    dr_mlp2_fwd(&m, Pd, xd, R, yd, ws); dr_mlp2_bwd(&m, Pd, G, xd, yd, Rd, R, dx, ws);
    double *yg = host(yd, R * m.O); ref_mlp2(&m, P, x, R, y); int bad = report("forward", yg, y, R * m.O, 2e-3);
    double *gx = host(dx, R * m.Cin), *gP = host(G, m.n), a[16], r[16];
    struct { const char *nm; size_t off, n; } ts[] = {{"x", 0, (size_t)R * m.Cin}, {"w1", m.w1, (size_t)m.H * m.Cin}, {"b1", m.b1, m.H}, {"w2", m.w2, (size_t)m.O * m.H}, {"b2", m.b2, m.O}};
    for (int k = 0; k < 5; k++) { for (int i = 0; i < 16; i++) { size_t j = (size_t)(urand() * ts[k].n) % ts[k].n; double *v = k == 0 ? x + j : P + ts[k].off + j, v0 = *v, e = 1e-5, lp = 0, lm = 0;
            *v = v0 + e; ref_mlp2(&m, P, x, R, y); for (int q = 0; q < R * m.O; q++) lp += y[q] * Rm[q]; *v = v0 - e; ref_mlp2(&m, P, x, R, y); for (int q = 0; q < R * m.O; q++) lm += y[q] * Rm[q]; *v = v0;
            r[i] = (lp - lm) / (2 * e); a[i] = k == 0 ? gx[j] : gP[ts[k].off + j]; }
        bad += report(ts[k].nm, a, r, 16, 2e-3); }
    free(P); free(x); free(Rm); free(y); free(yg); free(gx); free(gP); nn_free(Pd); nn_free(xd); nn_free(Rd); nn_free(yd); nn_free(G); nn_free(dx); nn_free(ws);
    return bad;
}
/* row layer norm (the CLS before the DINO head): forward vs double, backward of L = sum(w y) vs central differences */
static void ref_rownorm(const double *x, int R, int C, double *y) {
    for (int r = 0; r < R; r++) {
        double m = 0, v = 0;
        for (int c = 0; c < C; c++) m += x[r * C + c];
        m /= C;
        for (int c = 0; c < C; c++) v += (x[r * C + c] - m) * (x[r * C + c] - m);
        v /= C;
        for (int c = 0; c < C; c++) y[r * C + c] = (x[r * C + c] - m) / sqrt(v + 1e-6);
    }
}
static int test_rownorm(void) {
    const int R = 3, C = 192; printf("row layer norm: %d x %d\n", R, C); int bad = 0;
    double *x = malloc(R * C * 8), *w = malloc(R * C * 8), *y = malloc(R * C * 8), *gr = malloc(R * C * 8);
    for (int i = 0; i < R * C; i++) { x[i] = rf(3 + 5 * nrand()); w[i] = rf(nrand()); }
    float *xd = dev(x, R * C), *wd = dev(w, R * C), *yd = nn_malloc(R * C * 4), *rs = nn_malloc(R * 4), *gd = nn_malloc(R * C * 4);
    dr_rownorm_fwd(xd, R, C, yd, rs); ref_rownorm(x, R, C, y);
    double *a = host(yd, R * C); bad += report("forward", a, y, R * C, 2e-3); free(a);
    dr_rownorm_bwd(wd, yd, rs, R, C, gd);
    for (int i = 0; i < R * C; i++) { const double e = 1e-4, o = x[i]; double lp = 0, lm = 0;
        x[i] = o + e; ref_rownorm(x, R, C, y); for (int j = 0; j < R * C; j++) lp += w[j] * y[j];
        x[i] = o - e; ref_rownorm(x, R, C, y); for (int j = 0; j < R * C; j++) lm += w[j] * y[j]; x[i] = o; gr[i] = (lp - lm) / (2 * e); }
    a = host(gd, R * C); bad += report("d x", a, gr, R * C, 2e-3); free(a);
    free(x); free(w); free(y); free(gr); nn_free(xd); nn_free(wd); nn_free(yd); nn_free(rs); nn_free(gd);
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
    int bad = test_views() + test_head() + test_targets() + test_sharp() + test_toy_dino() + test_nce() + test_toy_nce() + test_ce() + test_koleo() + test_koleo_q() + test_muon(64, 24) + test_muon(24, 64) + test_muon_ref(160, 24) + test_obj() + test_ink() + test_mlp2() + test_pool() + test_rownorm();
    printf(bad ? "test_dinor: %d FAILED\n" : "test_dinor: all passed\n", bad);
    return bad != 0;
}
