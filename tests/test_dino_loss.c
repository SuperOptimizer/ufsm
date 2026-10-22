/* dino losses (src/nn_dino.cu): finite-difference checks of dn_supcon (anchor and student-key gradients, hard-negative weights), dn_dino (the
   student embeddings and the raw prototypes through their normalisation) and dn_gather + dn_scatter_grad (the gradient back through
   e = y / |y| into dense logits). fp32 on the GPU; relative error bounds 2e-3 on random small problems. */
#include "dino.h"
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rs = 88172645463325252ull;
static double ru(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 11) * (1.0 / 9007199254740992.0); }
static double rn(void) { return sqrt(-2 * log(ru() + 1e-12)) * cos(2 * M_PI * ru()); }
static void unit(float *v, int n, int D) { for (int i = 0; i < n; i++) { double s = 0; for (int c = 0; c < D; c++) s += v[i * D + c] * v[i * D + c]; s = sqrt(s); for (int c = 0; c < D; c++) v[i * D + c] /= s; } }
static float *up(const void *h, size_t b) { float *d = nn_malloc(b); nn_h2d(d, h, b); return d; }
static int fails = 0;
/* double-precision CPU references of the losses (the finite differences run on them) */
static int wrap_dk(int ci, int cj) { if (ci >= 256 || cj >= 256) return ci == cj ? 0 : 99; int d = ((cj - ci) % 252 + 252) % 252; if (d >= 126) d -= 252; return (int)lrint(d / 18.0); }
static double supcon_ref(const float *a, const int *code, const int *xyz, int N, const float *t, int N2, int D, double tau, double hw, double hr) {
    double L = 0; int na = 0;
    for (int i = 0; i < N; i++) {
        double den = 0, ps = 0; int np = 0;
        for (int j = 0; j < N + N2; j++) {
            if (j == i) continue;
            const float *k = j < N ? a + j * D : t + (j - N) * D; double s = 0; for (int c = 0; c < D; c++) s += (double)a[i * D + c] * k[c];
            const int cj = j < N ? code[j] : code[j - N], dk = wrap_dk(code[i], cj);
            double w = 1;
            if (abs(dk) == 1) { const int *pj = xyz + 3 * (j < N ? j : j - N); double d2 = 0; for (int q = 0; q < 3; q++) d2 += (double)(pj[q] - xyz[3 * i + q]) * (pj[q] - xyz[3 * i + q]); if (d2 <= hr * hr) w = hw; }
            den += w * exp(s / tau);
            if (dk == 0) { ps += s / tau; np++; }
        }
        if (np) { L += log(den) - ps / np; na++; }
    }
    return na ? L / na : 0;
}
static double dino_ref(const float *s, const float *t, int N, const float *W, const float *Wt, const float *cen, int K, int D, double ts, double tt) {
    double L = 0; double *zs = malloc(K * 8), *zt = malloc(K * 8);
    for (int i = 0; i < N; i++) {
        double ms = -1e300, mt = -1e300;
        for (int k = 0; k < K; k++) {
            double a = 0, b = 0, nw = 0, nt = 0;
            for (int c = 0; c < D; c++) { nw += (double)W[k * D + c] * W[k * D + c]; nt += (double)Wt[k * D + c] * Wt[k * D + c]; }
            for (int c = 0; c < D; c++) { a += s[i * D + c] * W[k * D + c] / sqrt(nw); b += t[i * D + c] * Wt[k * D + c] / sqrt(nt); }
            zs[k] = a / ts; zt[k] = (b - cen[k]) / tt; if (zs[k] > ms) ms = zs[k]; if (zt[k] > mt) mt = zt[k];
        }
        double ss = 0, st = 0; for (int k = 0; k < K; k++) { ss += exp(zs[k] - ms); st += exp(zt[k] - mt); }
        for (int k = 0; k < K; k++) { const double p = exp(zt[k] - mt) / st; L -= p * (zs[k] - ms - log(ss)); }
    }
    free(zs); free(zt); return L / N;
}
static void check(const char *what, double fd, double an, double scale) {
    const double err = fabs(fd - an) / (fabs(fd) + fabs(an) + scale);
    if (err > 2e-3) { printf("  %s: fd %.6e analytic %.6e (rel %.2e) FAIL\n", what, fd, an, err); fails++; }
}
int main(void) {
    if (nn_init(0)) { printf("SKIP: no GPU\n"); return 0; }
    const int D = getenv("UFSM_TEST_D") ? atoi(getenv("UFSM_TEST_D")) : 32, N = 96, N2 = 96;   /* embedding width 16 or 32 */
    /* ---- SupCon ---- */
    float *a = malloc(N * D * 4), *t = malloc(N2 * D * 4); int *code = malloc(N * 4), *xyz = malloc(N * 12);
    for (int i = 0; i < N * D; i++) a[i] = (float)rn();
    for (int i = 0; i < N2 * D; i++) t[i] = (float)rn();
    unit(a, N, D); unit(t, N2, D);
    for (int i = 0; i < N; i++) { const int k = (int)(ru() * 4); code[i] = 1 + ((k * 18 + (int)(ru() * 6)) % 252); for (int d = 0; d < 3; d++) xyz[3 * i + d] = (int)(ru() * 100); }
    int *dc = (int *)up(code, N * 4), *dx = (int *)up(xyz, N * 12); float *dt = up(t, N2 * D * 4);
    float *dga = nn_malloc(N * D * 4), *dout = nn_malloc(16), *scr = nn_malloc((2 * N + 16) * 4), *ga = malloc(N * D * 4), out[4];
    const float tau = 0.2f, hw = 3.f, hr = 60.f;
    float *da = up(a, N * D * 4);
    dn_supcon(da, dc, dx, N, dt, dc, dx, N2, D, tau, hw, hr, dga, dout, scr);
    nn_d2h(ga, dga, N * D * 4); nn_d2h(out, dout, 16);
    printf("supcon: loss %.5f, pos cos %.3f, neg cos %.3f, anchors %.0f\n", out[0], out[1], out[2], out[3]);
    for (int r = 0; r < 24; r++) {
        const int i = (int)(ru() * N), c = (int)(ru() * D); const float eps = 1e-3f; float lp, lm, o[4];
        (void)o; a[i * D + c] += eps; lp = (float)0; double Lp = supcon_ref(a, code, xyz, N, t, N2, D, tau, hw, hr);
        a[i * D + c] -= 2 * eps; double Lm = supcon_ref(a, code, xyz, N, t, N2, D, tau, hw, hr); a[i * D + c] += eps; lp = (float)Lp; lm = (float)Lm;
        if (r == 0) { const double L0 = supcon_ref(a, code, xyz, N, t, N2, D, tau, hw, hr); if (fabs(L0 - out[0]) > 1e-4 * (1 + fabs(L0))) { printf("  supcon loss %.6f vs reference %.6f FAIL\n", out[0], L0); fails++; } }
        check("supcon d/da", (Lp - Lm) / (2 * eps), ga[i * D + c], 1e-5); continue;
        check("supcon d/da", (lp - lm) / (2 * eps), ga[i * D + c], 1e-4);
    }
    /* ---- DINO prototypes ---- */
    const int K = 64, NS = 80; const float ts = 0.1f, tt = 0.05f;
    float *s = malloc(NS * D * 4), *tq = malloc(NS * D * 4), *W = malloc(K * D * 4), *Wt = malloc(K * D * 4), *cen = calloc(K, 4);
    for (int i = 0; i < NS * D; i++) { s[i] = (float)rn(); tq[i] = (float)rn(); }
    for (int i = 0; i < K * D; i++) { W[i] = (float)rn(); Wt[i] = W[i] + 0.1f * (float)rn(); }
    for (int k = 0; k < K; k++) cen[k] = 0.05f * (float)rn();
    unit(s, NS, D); unit(tq, NS, D);
    float *ds = up(s, NS * D * 4), *dtq = up(tq, NS * D * 4), *dW = up(W, K * D * 4), *dWt = up(Wt, K * D * 4), *dcen = up(cen, K * 4);
    float *dgs = nn_malloc(NS * D * 4), *dgW = nn_malloc(K * D * 4), *dtm = nn_malloc(K * 4), *dsc = nn_malloc(dn_dino_scratch(NS, K)), *gs = malloc(NS * D * 4), *gW = malloc(K * D * 4);
    dn_dino(ds, dtq, NS, dW, dWt, dcen, K, D, ts, tt, dgs, dgW, dtm, dout, dsc);
    nn_d2h(gs, dgs, NS * D * 4); nn_d2h(gW, dgW, K * D * 4); nn_d2h(out, dout, 12);
    printf("dino: loss %.5f, H(mean) %.3f, H %.3f\n", out[0], out[1], out[2]);
    for (int r = 0; r < 24; r++) {
        const int sel = r & 1, i = (int)(ru() * (sel ? K : NS)), c = (int)(ru() * D); const float eps = 1e-3f; float o[3], lp, lm;
        float *v = sel ? W : s, *dv = sel ? dW : ds; const size_t nb = (sel ? K : NS) * D * 4;
        (void)o; (void)dv; (void)nb; lp = lm = 0;
        v[i * D + c] += eps; const double Lp = dino_ref(s, tq, NS, W, Wt, cen, K, D, ts, tt); v[i * D + c] -= 2 * eps; const double Lm = dino_ref(s, tq, NS, W, Wt, cen, K, D, ts, tt); v[i * D + c] += eps;
        if (r == 0) { const double L0 = dino_ref(s, tq, NS, W, Wt, cen, K, D, ts, tt); if (fabs(L0 - out[0]) > 1e-4 * (1 + fabs(L0))) { printf("  dino loss %.6f vs reference %.6f FAIL\n", out[0], L0); fails++; } }
        check(sel ? "dino d/dW" : "dino d/ds", (Lp - Lm) / (2 * eps), sel ? gW[i * D + c] : gs[i * D + c], 1e-5); continue;
        check(sel ? "dino d/dW" : "dino d/ds", (lp - lm) / (2 * eps), sel ? gW[i * D + c] : gs[i * D + c], 1e-4);
    }
    /* ---- gather + scatter: L = sum_i w_i . e_i with e = y / |y| at points of a dense [D][p3] logit field ---- */
    const size_t p3 = 4096; const int NP = 50;
    float *y = malloc(D * p3 * 4), *wv = malloc(NP * D * 4); int *idx = malloc(NP * 4);
    for (size_t i = 0; i < D * p3; i++) y[i] = (float)rn();
    for (int i = 0; i < NP; i++) idx[i] = (int)(ru() * p3);
    for (int i = 0; i < NP * D; i++) wv[i] = (float)rn();
    float *dy = up(y, D * p3 * 4), *dw = up(wv, NP * D * 4), *de = nn_malloc(NP * D * 4), *dn = nn_malloc(NP * 4), *dgl = nn_malloc(D * p3 * 4), *gl = malloc(D * p3 * 4);
    int *didx = (int *)up(idx, NP * 4);
    dn_gather(dy, 0, p3, D, didx, NP, de, dn); nn_zero(dgl, D * p3 * 4);
    dn_scatter_grad(de, dw, dn, didx, NP, D, p3, 1.f, dgl); nn_d2h(gl, dgl, D * p3 * 4);
    float *e = malloc(NP * D * 4);
    for (int r = 0; r < 20; r++) {
        const int pi = (int)(ru() * NP), c = (int)(ru() * D); const size_t v = (size_t)c * p3 + idx[pi]; const float eps = 1e-3f; double l[2];
        for (int sgn = 0; sgn < 2; sgn++) {
            y[v] += sgn ? -eps : eps; double L = 0;   /* in double on the CPU */
            for (int i = 0; i < NP; i++) { double n2 = 0; for (int q = 0; q < D; q++) n2 += (double)y[(size_t)q * p3 + idx[i]] * y[(size_t)q * p3 + idx[i]];
                for (int q = 0; q < D; q++) L += (double)wv[i * D + q] * y[(size_t)q * p3 + idx[i]] / sqrt(n2); }
            l[sgn] = L; y[v] += sgn ? eps : -eps; (void)e;
        }
        check("scatter d/dy", (l[0] - l[1]) / (2 * eps), gl[v], 1e-4);
    }
    nn_sync();
    printf("%s\n", fails ? "test_dino_loss: FAIL" : "test_dino_loss: OK");
    return fails ? 1 : 0;
}
