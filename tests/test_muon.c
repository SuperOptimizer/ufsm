/* nn_muon: with p = 0, mom = 0, wd = 0 the update is -lr * scale * O where O is the orthogonalised gradient: check O O^T ~ I */
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
typedef struct { float *p; const float *g; float *mom, *X, *Y, *A, *B; int Co, K; } muon_desc_t;   /* mirrors nn_common.cuh */
/* nn_muon_batch (tiled products, every conv in one launch per stage) == nn_muon (scalar reference) per conv, over three
   steps with momentum and weight decay, for widths of the 4- and 6-level nets and ragged sizes (Co, K not multiples of 64) */
static int batch_matches_reference(void) {
    enum { NC = 6 };
    const int Co[NC] = {16, 40, 96, 128, 192, 160}, K[NC] = {4 * 27, 37 * 27, 224 * 27, 96 * 27, 192 * 27, 352 * 27};
    muon_desc_t h[NC]; float *ref[NC], *refm[NC], *work[NC]; float *hp[NC]; size_t n[NC]; int maxco = 0, maxk = 0;
    unsigned s = 99;
    for (int c = 0; c < NC; c++) {
        n[c] = (size_t)Co[c] * K[c];
        float *w = malloc(n[c] * 4); for (size_t i = 0; i < n[c]; i++) { s = s * 1664525u + 1013904223u; w[i] = ((s >> 8) / 16777216.f - 0.5f) * 0.1f; }
        h[c].p = nn_malloc(n[c] * 4); nn_h2d(h[c].p, w, n[c] * 4); ref[c] = nn_malloc(n[c] * 4); nn_h2d(ref[c], w, n[c] * 4); hp[c] = w;
        h[c].mom = nn_malloc(n[c] * 4); nn_zero(h[c].mom, n[c] * 4); refm[c] = nn_malloc(n[c] * 4); nn_zero(refm[c], n[c] * 4);
        h[c].X = nn_malloc(n[c] * 4); h[c].Y = nn_malloc(n[c] * 4); h[c].A = nn_malloc((size_t)Co[c] * Co[c] * 4); h[c].B = nn_malloc((size_t)Co[c] * Co[c] * 4);
        work[c] = nn_malloc((2 * n[c] + 2 * (size_t)Co[c] * Co[c]) * 4);
        h[c].g = nn_malloc(n[c] * 4); h[c].Co = Co[c]; h[c].K = K[c];
        if (Co[c] > maxco) maxco = Co[c];
        if (K[c] > maxk) maxk = K[c];
    }
    void *dd = nn_malloc(sizeof h); nn_h2d(dd, h, sizeof h);
    for (int step = 0; step < 3; step++) {
        for (int c = 0; c < NC; c++) {
            float *g = malloc(n[c] * 4); for (size_t i = 0; i < n[c]; i++) { s = s * 1664525u + 1013904223u; g[i] = ((s >> 8) / 16777216.f - 0.5f) * (1.f + (i % 5)); }
            nn_h2d((float *)h[c].g, g, n[c] * 4); free(g);
            nn_muon(ref[c], h[c].g, refm[c], Co[c], K[c], 0.02f, 0.95f, 0.01f, work[c]);
        }
        nn_muon_batch(dd, NC, maxco, maxk, 0.02f, 0.95f, 0.01f);
    }
    nn_sync(); const char *e = nn_check(); if (e) { printf("cuda: %s\n", e); return 0; }
    int ok = 1;
    for (int c = 0; c < NC; c++) {
        float *a = malloc(n[c] * 4), *r = malloc(n[c] * 4); nn_d2h(a, h[c].p, n[c] * 4); nn_d2h(r, ref[c], n[c] * 4);
        double num = 0, den = 0; for (size_t i = 0; i < n[c]; i++) { double du = (double)a[i] - hp[c][i], dr = (double)r[i] - hp[c][i]; num += (du - dr) * (du - dr); den += dr * dr; }
        double rel = sqrt(num / (den > 0 ? den : 1));
        printf("  batched == reference, Co %3d K %5d: update rel err %.3g%s\n", Co[c], K[c], rel, rel < 1e-4 ? "" : "  FAIL");
        ok &= rel < 1e-4;
        free(a); free(r); free(hp[c]);
    }
    printf(ok ? "muon batch ok\n" : "muon batch FAIL\n");
    return ok;
}
int main(void) {
    nn_init(0);
    int Co = 48, K = 32 * 27; size_t n = (size_t)Co * K;
    float *g = malloc(n * 4); unsigned s = 1234567; for (size_t i = 0; i < n; i++) { s = s * 1664525u + 1013904223u; g[i] = ((s >> 8) / 16777216.f - 0.5f) * (1.f + (i % 7)); }
    float *dp = nn_malloc(n * 4), *dg = nn_malloc(n * 4), *dm = nn_malloc(n * 4), *work = nn_malloc((2 * n + 2 * (size_t)Co * Co) * 4);
    nn_zero(dp, n * 4); nn_zero(dm, n * 4); nn_h2d(dg, g, n * 4);
    float lr = 1.f, beta = 0.f;
    nn_muon(dp, dg, dm, Co, K, lr, beta, 0.f, work);
    nn_sync(); const char *e = nn_check(); if (e) { printf("cuda: %s\n", e); return 1; }
    float *o = malloc(n * 4); nn_d2h(o, dp, n * 4);
    double scale = sqrt(fmax(1.0, (double)Co / K)), maxoff = 0, maxdiag = 0;
    for (int i = 0; i < Co; i++) for (int j = 0; j < Co; j++) {
        double d = 0; for (int k = 0; k < K; k++) d += (double)o[(size_t)i * K + k] * o[(size_t)j * K + k];
        d /= scale * scale;   /* O = -p / (lr scale) */
        if (i == j) maxdiag = fmax(maxdiag, fabs(d - 1.0)); else maxoff = fmax(maxoff, fabs(d));
    }
    printf("O O^T: max |diag - 1| %.3g, max |offdiag| %.3g\n", maxdiag, maxoff);
    int ok = maxdiag < 0.35 && maxoff < 0.35;   /* 5 Newton-Schulz iterations give singular values in ~[0.7, 1.2] by design */
    printf(ok ? "muon ok\n" : "muon FAIL\n");
    ok &= batch_matches_reference();
    return !ok;
}
