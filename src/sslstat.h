/* batch-statistic terms of the SSL batch (dino.c dn_ssl_*), host side, for the gradient-cached (GradCache) virtual batch: the full batch's
   rows X [N][D] are known before any backward (the exact bank), so each row's gradient of the full-batch loss is computed here and fed to
   that row's backward. tests/test_sslstat.c checks every gradient against central differences of the full-batch loss in double. */
#pragma once
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* per-dimension mean / std (sqrt(var_unbiased + eps)) and, when cov is not nullptr, the unbiased covariance [D][D] of X [N][D] */
static inline void ss_stats(const float *X, int N, int D, double eps, double *mu, double *sd, double *cov) {
    for (int c = 0; c < D; c++) { double m = 0; for (int k = 0; k < N; k++) m += X[(size_t)k * D + c]; mu[c] = m / N; }
    for (int c = 0; c < D; c++) { double a = 0; for (int k = 0; k < N; k++) { const double d = X[(size_t)k * D + c] - mu[c]; a += d * d; } sd[c] = sqrt(a / (N - 1) + eps); }
    if (cov) for (int a = 0; a < D; a++) for (int b = a; b < D; b++) {
        double r = 0; for (int k = 0; k < N; k++) r += (X[(size_t)k * D + a] - mu[a]) * (X[(size_t)k * D + b] - mu[b]);
        cov[(size_t)a * D + b] = cov[(size_t)b * D + a] = r / (N - 1);
    }
}
/* VICReg on a batch: L = wv / D sum_c relu(gam - sd_c) + wc / D sum_{a != b} cov_ab^2 (VICReg's per-dimension forms; gam the std target).
   ss_vic_loss: L of X; ss_vic_row_grad: d L / d x (x one of X's rows) from the batch statistics (exact: the mean terms cancel) */
static inline double ss_vic_loss(const double *sd, const double *cov, int D, double gam, double wv, double wc) {
    double l = 0; for (int c = 0; c < D; c++) l += wv / D * (sd[c] < gam ? gam - sd[c] : 0);
    if (cov && wc > 0) for (int a = 0; a < D; a++) for (int b = 0; b < D; b++) if (a != b) l += wc / D * cov[(size_t)a * D + b] * cov[(size_t)a * D + b];
    return l;
}
static inline void ss_vic_row_grad(const float *x, int N, int D, const double *mu, const double *sd, const double *cov, double gam, double wv, double wc, float *g) {
    for (int c = 0; c < D; c++) g[c] = (float)(wv > 0 && sd[c] < gam ? -wv / D * (x[c] - mu[c]) / ((N - 1) * sd[c]) : 0);
    if (cov && wc > 0) for (int a = 0; a < D; a++) {
        double s = 0; for (int b = 0; b < D; b++) if (b != a) s += cov[(size_t)a * D + b] * (x[b] - mu[b]);
        g[a] += (float)(wc * 4.0 / (D * (double)(N - 1)) * s);
    }
}
/* KoLeo on a batch (DINOv2): xh = x / |x|; n(i) = argmax_{j != i} xh_i . xh_j; L = -w / N sum_i log(|xh_i - xh_n(i)| + eps). The gradient of
   L for every row (its own term and the terms where it is another row's nearest neighbour), through the normalisation, into G [N][D]
   (written); returns L. (The neighbour choice is piecewise constant: the gradient of the active pairs.) */
static inline double ss_koleo_all(const float *X, int N, int D, double w, double eps, float *G) {
    double *xh = malloc((size_t)N * D * 8), *nr = malloc((size_t)N * 8), *gh = calloc((size_t)N * D, 8); int *nn = malloc((size_t)N * sizeof(int)); double L = 0;
    for (int i = 0; i < N; i++) { double s = 0; for (int c = 0; c < D; c++) s += (double)X[(size_t)i * D + c] * X[(size_t)i * D + c]; nr[i] = sqrt(s) + 1e-12; for (int c = 0; c < D; c++) xh[(size_t)i * D + c] = X[(size_t)i * D + c] / nr[i]; }
    for (int i = 0; i < N; i++) { double best = -1e300; nn[i] = -1; for (int j = 0; j < N; j++) { if (j == i) continue; double d = 0; for (int c = 0; c < D; c++) d += xh[(size_t)i * D + c] * xh[(size_t)j * D + c]; if (d > best) { best = d; nn[i] = j; } } }
    for (int i = 0; i < N; i++) {
        const int j = nn[i]; double d2 = 0; for (int c = 0; c < D; c++) { const double t = xh[(size_t)i * D + c] - xh[(size_t)j * D + c]; d2 += t * t; }
        const double d = sqrt(d2); L += -w / N * log(d + eps);
        const double k = -w / N / ((d + eps) * (d > 1e-30 ? d : 1e-30));   /* d L / d (xh_i - xh_j) = k (xh_i - xh_j) */
        for (int c = 0; c < D; c++) { const double t = k * (xh[(size_t)i * D + c] - xh[(size_t)j * D + c]); gh[(size_t)i * D + c] += t; gh[(size_t)j * D + c] -= t; }
    }
    for (int i = 0; i < N; i++) {   /* through xh = x / |x|: g = (gh - xh (xh . gh)) / |x| */
        double p = 0; for (int c = 0; c < D; c++) p += xh[(size_t)i * D + c] * gh[(size_t)i * D + c];
        for (int c = 0; c < D; c++) G[(size_t)i * D + c] = (float)((gh[(size_t)i * D + c] - xh[(size_t)i * D + c] * p) / nr[i]);
    }
    free(xh); free(nr); free(gh); free(nn); return L;
}
/* relational (similarity-matrix) distillation: E, T [m][d] (unit rows: student, teacher); L = w / (m (m - 1)) sum_{a != b} (e_a . e_b - t_a . t_b)^2,
   invariant to any rotation of the teacher's code space (dinovol's sheet codes need only agree within a crop). G [m][d] += d L / d E
   (when not nullptr); returns L. */
static inline double ss_relsim(const float *E, const float *T, int m, int d, double w, float *G) {
    if (m < 2) return 0;
    const double c = w / ((double)m * (m - 1)); double L = 0;
    for (int a = 0; a < m; a++) for (int b = a + 1; b < m; b++) {
        double se = 0, st = 0; for (int k = 0; k < d; k++) { se += (double)E[(size_t)a * d + k] * E[(size_t)b * d + k]; st += (double)T[(size_t)a * d + k] * T[(size_t)b * d + k]; }
        const double D = se - st; L += 2 * c * D * D;
        if (G) { const double g = 4 * c * D; for (int k = 0; k < d; k++) { G[(size_t)a * d + k] += (float)(g * E[(size_t)b * d + k]); G[(size_t)b * d + k] += (float)(g * E[(size_t)a * d + k]); } }
    }
    return L;
}
/* teacher-graph SupCon: anchors a of E [m][d] (unit rows), positives P(a) = {b != a : T_a . T_b > thr} (the teacher's cosine graph),
   L = w / |A| sum_{a in A} -1/|P(a)| sum_{p in P(a)} log softmax_{b != a}(E_a . E_b / tau)_p over the anchors A with a positive.
   Only which points the teacher groups together is matched, not its code vectors. G [m][d] += d L / d E (when not nullptr);
   *pf (when not nullptr) = the fraction of pairs that are positive; returns L. */
static inline double ss_tnce(const float *E, const float *T, int m, int d, double thr, double tau, double w, float *G, double *pf) {
    if (m < 2) return 0;
    unsigned char *pos = malloc((size_t)m * m); int *npa = calloc((size_t)m, sizeof *npa), na = 0; long np = 0;
    for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) { double st = 0; if (b != a) for (int k = 0; k < d; k++) st += (double)T[(size_t)a * d + k] * T[(size_t)b * d + k];
        pos[(size_t)a * m + b] = b != a && st > thr; npa[a] += pos[(size_t)a * m + b]; }
    for (int a = 0; a < m; a++) { na += npa[a] > 0; np += npa[a]; }
    if (pf) *pf = (double)np / ((double)m * (m - 1));
    double *s = malloc((size_t)m * 8), L = 0; const double c = na ? w / na : 0;
    for (int a = 0; a < m; a++) {
        if (!npa[a]) continue; const unsigned char *pa = pos + (size_t)a * m; double mx = -1e300;
        for (int b = 0; b < m; b++) { if (b == a) continue; double se = 0; for (int k = 0; k < d; k++) se += (double)E[(size_t)a * d + k] * E[(size_t)b * d + k]; s[b] = se / tau; if (s[b] > mx) mx = s[b]; }
        double Z = 0; for (int b = 0; b < m; b++) if (b != a) Z += exp(s[b] - mx);
        const double lz = mx + log(Z); double la = 0; for (int b = 0; b < m; b++) if (pa[b]) la += lz - s[b];
        L += c * la / npa[a];
        if (G) for (int b = 0; b < m; b++) { if (b == a) continue; const double g = c * (exp(s[b] - lz) - (pa[b] ? 1.0 / npa[a] : 0.0)) / tau;   /* d L / d (E_a . E_b) */
            for (int k = 0; k < d; k++) { G[(size_t)a * d + k] += (float)(g * E[(size_t)b * d + k]); G[(size_t)b * d + k] += (float)(g * E[(size_t)a * d + k]); } }
    }
    free(s); free(pos); free(npa); return L;
}
/* SIGReg (LeJEPA, arXiv 2511.08544): the rows of X [N][D] (times xs) projected on S unit directions A [S][D]; per direction the
   Epps-Pulley distance between the projections' empirical characteristic function and the standard normal's, phi(t) = exp(-t^2 / 2):
   int_{-tmax}^{tmax} phi(t) ((Re ecf(t) - phi(t))^2 + Im ecf(t)^2) dt (an even integrand: 2 x the trapezoid rule on K knots of [0, tmax]),
   times N (the test statistic); L = w x the mean over directions. Teacher-free: pushes the embedding towards an isotropic Gaussian.
   cos / sin of t_k x by the angle-addition recurrence (one sincos per projection). G [N][D] (written, when not nullptr) = d L / d X; returns L. */
static inline double ss_sigreg_all(const float *X, int N, int D, const float *A, int S, int K, double tmax, double xs, double w, float *G) {
    const double dt = tmax / (K - 1); double *ph = malloc((size_t)K * 8), *cw = malloc((size_t)K * 8), *p = malloc((size_t)N * 8), *c1 = malloc((size_t)N * 8), *s1 = malloc((size_t)N * 8);
    double *re = malloc((size_t)K * 8), *im = malloc((size_t)K * 8), *gp = malloc((size_t)N * 8); double L = 0;
    for (int k = 0; k < K; k++) { const double t = k * dt; ph[k] = exp(-0.5 * t * t); cw[k] = 2 * dt * (k == 0 || k == K - 1 ? 0.5 : 1.0) * ph[k]; }
    if (G) memset(G, 0, (size_t)N * D * sizeof *G);
    for (int s = 0; s < S; s++) {
        const float *a = A + (size_t)s * D;
        for (int i = 0; i < N; i++) { double v = 0; for (int c = 0; c < D; c++) v += (double)X[(size_t)i * D + c] * a[c]; p[i] = xs * v; c1[i] = cos(dt * p[i]); s1[i] = sin(dt * p[i]); }
        for (int k = 0; k < K; k++) { re[k] = 0; im[k] = 0; }
        for (int i = 0; i < N; i++) { double ck = 1, sk = 0; for (int k = 0; k < K; k++) { re[k] += ck / N; im[k] += sk / N; const double cn = ck * c1[i] - sk * s1[i]; sk = sk * c1[i] + ck * s1[i]; ck = cn; } }
        double l = 0; for (int k = 0; k < K; k++) l += cw[k] * ((re[k] - ph[k]) * (re[k] - ph[k]) + im[k] * im[k]);
        L += w * N * l / S;
        if (!G) continue;
        /* d l / d p_i = sum_k cw_k 2 ((re_k - ph_k) (-t_k sin(t_k p_i)) + im_k t_k cos(t_k p_i)) / N; d L / d X_i = w N / S xs (d l / d p_i) a */
        for (int i = 0; i < N; i++) { double ck = 1, sk = 0, g = 0;
            for (int k = 0; k < K; k++) { const double t = k * dt; g += cw[k] * 2 * t * (-(re[k] - ph[k]) * sk + im[k] * ck) / N; const double cn = ck * c1[i] - sk * s1[i]; sk = sk * c1[i] + ck * s1[i]; ck = cn; }
            gp[i] = w * N / S * xs * g; }
        for (int i = 0; i < N; i++) for (int c = 0; c < D; c++) G[(size_t)i * D + c] += (float)(gp[i] * a[c]);
    }
    free(ph); free(cw); free(p); free(c1); free(s1); free(re); free(im); free(gp); return L;
}
