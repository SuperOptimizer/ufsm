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
