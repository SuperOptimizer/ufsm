/* ufsm dinovol-probe's CPU parts (src/dvprobe.c, 2026-11-24): the weighted ROC AUC against hand-computed cases (hard labels, ties, soft
   labels split into ink / non-ink mass, zero-weight rows ignored) and the weighted logistic probe: on synthetic features of unequal
   scales and offsets with soft labels drawn from a known logistic model, the Newton fit recovers the true weights in the standardised
   space and its gradient vanishes (checked against a double recomputation here). CPU only. */
#include "../src/dinovol.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

static double rnd(unsigned long long *s) { *s = *s * 6364136223846793005ull + 1442695040888963407ull; return ((*s >> 11) + 0.5) / 9007199254740992.0; }
static double gauss(unsigned long long *s) { return sqrt(-2 * log(rnd(s))) * cos(6.283185307179586 * rnd(s)); }

int main(void) {
    {   /* AUC: perfect, one swap, ties, soft labels, zero weights */
        const double s1[4] = {1, 2, 3, 4}; dvf_row R[4] = {{0, 0, 0, 0, 1}, {0, 0, 0, 0, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}};
        CHECK(fabs(dvp_wauc(s1, R, 4) - 1) < 1e-12, "perfect %g", dvp_wauc(s1, R, 4));
        dvf_row R2[4] = {{0, 0, 0, 0, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 0, 1}, {0, 0, 0, 1, 1}};
        CHECK(fabs(dvp_wauc(s1, R2, 4) - 0.75) < 1e-12, "one swap %g", dvp_wauc(s1, R2, 4));
        const double s3[4] = {1, 1, 1, 1}; CHECK(fabs(dvp_wauc(s3, R, 4) - 0.5) < 1e-12, "ties %g", dvp_wauc(s3, R, 4));
        /* soft: rows (s 1, y 0.25, w 4) and (s 2, y 0.75, w 4): ink mass 1 at s1 and 3 at s2, non-ink 3 at s1 and 1 at s2;
           pairs ink > non-ink: ink@2 vs non@1 = 9, ties at equal scores 0.5 (1*3 + 3*1) = 3 -> 12 / (4 * 4) */
        const double s4[3] = {1, 2, 5}; dvf_row R4[3] = {{0, 0, 0, 0.25f, 4}, {0, 0, 0, 0.75f, 4}, {0, 0, 0, 1, 0}};
        CHECK(fabs(dvp_wauc(s4, R4, 3) - 12.0 / 16) < 1e-12, "soft %g", dvp_wauc(s4, R4, 3));
    }
    {   /* the probe */
        enum { N = 40000, F = 6 }; unsigned long long sd0 = 7;
        const double wt[F] = {1.5, -2.0, 0.0, 0.7, 0.0, -0.4}, bt = -1.0, off[F] = {3, -1, 0.5, 100, 0, 7}, scl[F] = {0.1, 2, 1, 30, 1e-3, 5};
        float *X = malloc(sizeof(float) * N * F); dvf_row *R = malloc(sizeof(dvf_row) * N);
        for (int i = 0; i < N; i++) { double t = bt;
            for (int k = 0; k < F; k++) { const double z = gauss(&sd0); X[(size_t)i * F + k] = (float)(off[k] + scl[k] * z); t += wt[k] * z; }
            R[i] = (dvf_row){i / 1000, 0, 0, (float)(1 / (1 + exp(-t))), (float)(1 + (i % 3)) };   /* soft label = the true probability, weights 1..3 */
        }
        double mu[F], sdv[F], w[F + 1];
        CHECK(!dvp_fit(X, R, N, F, 1e-8, 30, 4, mu, sdv, w), "fit failed");
        for (int k = 0; k < F; k++) CHECK(fabs(w[k] * scl[k] / sdv[k] - wt[k]) < 0.06, "w[%d] %g (true %g)", k, w[k] * scl[k] / sdv[k], wt[k]);
        double bias = w[F]; for (int k = 0; k < F; k++) bias -= w[k] * mu[k] / sdv[k];   /* back to raw x: logit = sum w_k (x_k - mu_k) / sd_k + w_F */
        double braw = bt; for (int k = 0; k < F; k++) braw -= wt[k] * off[k] / scl[k];
        CHECK(fabs(bias - braw) < 0.05 * (1 + fabs(braw)), "bias %g (true %g)", bias, braw);
        double g[F + 1] = {0}, W = 0;   /* the objective's gradient at the solution (double, recomputed here) */
        for (int i = 0; i < N; i++) { double z[F + 1], t = w[F]; z[F] = 1;
            for (int k = 0; k < F; k++) { z[k] = (X[(size_t)i * F + k] - mu[k]) / sdv[k]; t += w[k] * z[k]; }
            const double r = R[i].wgt * (1 / (1 + exp(-t)) - R[i].lab); W += R[i].wgt; for (int k = 0; k <= F; k++) g[k] += r * z[k]; }
        double gm = 0; for (int k = 0; k <= F; k++) { const double v = fabs(g[k] / W + (k < F ? 1e-8 * w[k] : 0)); if (v > gm) gm = v; }
        CHECK(gm < 1e-6, "gradient at the solution %g", gm);
        double *s = malloc(sizeof(double) * N), *st = malloc(sizeof(double) * N);
        for (int i = 0; i < N; i++) { double t = w[F], u = bt; for (int k = 0; k < F; k++) { t += w[k] * (X[(size_t)i * F + k] - mu[k]) / sdv[k]; u += wt[k] * (X[(size_t)i * F + k] - off[k]) / scl[k]; } s[i] = t; st[i] = u; }
        CHECK(fabs(dvp_wauc(s, R, N) - dvp_wauc(st, R, N)) < 2e-3, "AUC fit %g vs true model %g", dvp_wauc(s, R, N), dvp_wauc(st, R, N));
        free(s); free(st); free(X); free(R);
    }
    if (fails) { fprintf(stderr, "test_dvprobe: %d FAILED\n", fails); return 1; }
    printf("test_dvprobe: ok\n");
    return 0;
}
