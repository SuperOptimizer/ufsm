/* src/sslstat.h (the SSL batch's gradient-cached batch-statistic terms): each row's gradient of the FULL-batch loss vs central differences
   of that loss in double -- VICReg (variance hinge + off-diagonal covariance) and KoLeo (own term + the terms where the row is a neighbour).
   This is the GradCache exactness of the virtual big batch: the per-row gradients fed to the per-view backwards equal the full batch's. */
#include "sslstat.h"
#include <stdio.h>

static unsigned long long rs = 99;
static double urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (double)(rs >> 11) * 0x1.0p-53; }
static double vic_full(float *X, int N, int D, double gam, double wv, double wc) {
    double *mu = malloc(D * 8), *sd = malloc(D * 8), *cv = malloc((size_t)D * D * 8); ss_stats(X, N, D, 1e-4, mu, sd, cv);
    const double l = ss_vic_loss(sd, cv, D, gam, wv, wc); free(mu); free(sd); free(cv); return l;
}
int main(void) {
    const int N = 9, D = 7; float X[9 * 7]; int bad = 0;
    for (int i = 0; i < N * D; i++) X[i] = (float)(urand() * 0.6 - 0.3);
    /* VICReg: gam above the stds so every hinge is active; the covariance term on */
    { const double gam = 1.0, wv = 1.3, wc = 0.7; double *mu = malloc(D * 8), *sd = malloc(D * 8), *cv = malloc((size_t)D * D * 8); ss_stats(X, N, D, 1e-4, mu, sd, cv);
      double emax = 0, gmax = 0; float g[7];
      for (int i = 0; i < N; i++) { ss_vic_row_grad(X + (size_t)i * D, N, D, mu, sd, cv, gam, wv, wc, g);
          for (int c = 0; c < D; c++) { const float o = X[i * D + c]; const double e = 1e-3; X[i * D + c] = (float)(o + e); const double lp = vic_full(X, N, D, gam, wv, wc); X[i * D + c] = (float)(o - e); const double lm = vic_full(X, N, D, gam, wv, wc); X[i * D + c] = o;
              const double fd = (lp - lm) / (2 * e); emax = fmax(emax, fabs(fd - g[c])); gmax = fmax(gmax, fabs(fd)); } }
      const int b = !(emax < 2e-3 * gmax); printf("VICReg row gradients vs the full batch: max |err| %.2e of max |grad| %.2e%s\n", emax, gmax, b ? "  FAIL" : ""); bad += b;
      free(mu); free(sd); free(cv); }
    /* KoLeo: every row's gradient (own term + neighbour terms) vs central differences of the full-batch loss (small steps: neighbours fixed) */
    { const double w = 0.8, eps = 1e-8; float G[9 * 7]; ss_koleo_all(X, N, D, w, eps, G); double emax = 0, gmax = 0; float Gd[9 * 7];
      for (int i = 0; i < N * D; i++) { const float o = X[i]; const double e = 1e-4; X[i] = (float)(o + e); const double lp = ss_koleo_all(X, N, D, w, eps, Gd); X[i] = (float)(o - e); const double lm = ss_koleo_all(X, N, D, w, eps, Gd); X[i] = o;
          const double fd = (lp - lm) / (2 * e); emax = fmax(emax, fabs(fd - G[i])); gmax = fmax(gmax, fabs(fd)); }
      const int b = !(emax < 5e-3 * gmax); printf("KoLeo all-row gradients vs the full batch: max |err| %.2e of max |grad| %.2e%s\n", emax, gmax, b ? "  FAIL" : ""); bad += b; }
    /* SIGReg: every row's gradient vs central differences in double; plus a sanity check: Gaussian rows score far lower than collapsed ones */
    { const int S = 5, K = 17, Dg = 7; float A[5 * 7]; for (int s = 0; s < S; s++) { double n = 0; for (int c = 0; c < Dg; c++) { A[s * Dg + c] = (float)(urand() - 0.5); n += A[s * Dg + c] * A[s * Dg + c]; } for (int c = 0; c < Dg; c++) A[s * Dg + c] /= (float)sqrt(n); }
      const double xs = 2.6, w = 0.9; float G[9 * 7], Gd[9 * 7]; ss_sigreg_all(X, N, Dg, A, S, K, 3.0, xs, w, G); double emax = 0, gmax = 0;
      for (int i = 0; i < N * Dg; i++) { const float o = X[i]; const double e = 1e-3; X[i] = (float)(o + e); const double ep = (double)X[i] - o; const double lp = ss_sigreg_all(X, N, Dg, A, S, K, 3.0, xs, w, nullptr);
          X[i] = (float)(o - e); const double em = o - (double)X[i]; const double lm = ss_sigreg_all(X, N, Dg, A, S, K, 3.0, xs, w, nullptr); X[i] = o;
          const double fd = (lp - lm) / (ep + em); emax = fmax(emax, fabs(fd - G[i])); gmax = fmax(gmax, fabs(fd)); (void)Gd; }
      int b = !(emax < 2e-3 * gmax); printf("SIGReg all-row gradients: max |err| %.2e of max |grad| %.2e%s\n", emax, gmax, b ? "  FAIL" : ""); bad += b;
      enum { NG = 512 }; float *Xg = malloc((size_t)NG * Dg * 4), *Xc = malloc((size_t)NG * Dg * 4);
      for (int i = 0; i < NG * Dg; i++) { const double u1 = urand() + 1e-12, u2 = urand(); Xg[i] = (float)(sqrt(-2 * log(u1)) * cos(6.283185307179586 * u2)); Xc[i] = 0.05f * Xg[i] + 0.7f; }
      const double lg = ss_sigreg_all(Xg, NG, Dg, A, S, K, 3.0, 1.0, 1.0, nullptr), lc = ss_sigreg_all(Xc, NG, Dg, A, S, K, 3.0, 1.0, 1.0, nullptr);
      b = !(lg < 2.0 && lc > 50 * lg); printf("SIGReg statistic: gaussian %.3f, collapsed %.3f%s\n", lg, lc, b ? "  FAIL" : ""); bad += b; free(Xg); free(Xc); }
    printf("%s\n", bad ? "test_sslstat: FAIL" : "test_sslstat: OK");
    return bad != 0;
}
