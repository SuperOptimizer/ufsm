#include "latfit.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { LF_CAP = 600000 };   /* reservoir blocks for the rate / distortion table (the covariance uses every sampled cell) */
static double g_cov[64][64], g_m1, g_m2; static long g_n; static float *g_res; static long g_nres;

void latfit_acc(const uint16_t *ct16, int D, int P, int stride, int phase) {
    if (!g_res) g_res = malloc((size_t)LF_CAP * 64 * sizeof(float));
    const _Float16 *x = (const _Float16 *)ct16;
    for (int z = phase % stride; z < D / 4; z += stride) for (int y = (phase / 3) % stride; y < P / 4; y += stride) for (int xx = (phase / 7) % stride; xx < P / 4; xx += stride) {
        float v[64], mn = INFINITY, mx = -INFINITY; double s = 0;
        for (int k = 0; k < 64; k++) {
            v[k] = (float)x[((size_t)(4 * z + (k >> 4)) * P + 4 * y + (k >> 2 & 3)) * P + 4 * xx + (k & 3)];
            s += v[k]; mn = fminf(mn, v[k]); mx = fmaxf(mx, v[k]);
        }
        if (!(mx > mn) || !isfinite(mx) || !isfinite(mn)) continue;   /* air / masked cells: constant */
        const float m = (float)(s / 64);
        float r[64]; for (int k = 0; k < 64; k++) r[k] = v[k] - m;
        for (int a = 0; a < 64; a++) { const double ra = r[a]; for (int b = a; b < 64; b++) g_cov[a][b] += ra * r[b]; }
        g_m1 += 8.0 * m; g_m2 += 64.0 * (double)m * m; g_n++;
        if (g_nres < LF_CAP) memcpy(g_res + (size_t)g_nres++ * 64, v, sizeof v);
        else { static uint64_t rs = 0x9e3779b97f4a7c15ull; rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; const uint64_t j = rs % (uint64_t)g_n; if (j < LF_CAP) memcpy(g_res + (size_t)j * 64, v, sizeof v); }
    }
}

/* cyclic Jacobi: the eigenvalues (w) and eigenvectors (columns of V) of the symmetric A (destroyed) */
static void jacobi(int n, double A[64][64], double V[64][64], double *w) {
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) V[i][j] = i == j;
    for (int sweep = 0; sweep < 100; sweep++) {
        double off = 0; for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
        if (off < 1e-22) break;
        for (int p = 0; p < n; p++) for (int q = p + 1; q < n; q++) {
            if (fabs(A[p][q]) < 1e-300) continue;
            const double th = (A[q][q] - A[p][p]) / (2 * A[p][q]), t = (th >= 0 ? 1 : -1) / (fabs(th) + sqrt(th * th + 1)), c = 1 / sqrt(t * t + 1), s = t * c;
            for (int k = 0; k < n; k++) { const double akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq; }
            for (int k = 0; k < n; k++) { const double apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk; }
            for (int k = 0; k < n; k++) { const double vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
        }
    }
    for (int i = 0; i < n; i++) w[i] = A[i][i];
}

int latfit_finish(const char *path) {
    if (g_n < 1000) { fprintf(stderr, "lat-fit: only %ld cells\n", g_n); return 1; }
    static double A[64][64], V[64][64]; double w[64];
    for (int a = 0; a < 64; a++) for (int b = a; b < 64; b++) A[a][b] = A[b][a] = g_cov[a][b] / (double)g_n;
    jacobi(64, A, V, w);
    int ord[64]; for (int i = 0; i < 64; i++) ord[i] = i;
    for (int i = 0; i < 64; i++) for (int j = i + 1; j < 64; j++) if (w[ord[j]] > w[ord[i]]) { const int t = ord[i]; ord[i] = ord[j]; ord[j] = t; }
    lat_basis *lb = calloc(1, sizeof *lb);
    const double dm = g_m1 / (double)g_n; lb->sigma[0] = (float)sqrt(fmax(g_m2 / (double)g_n - dm * dm, 1e-12));
    for (int k = 0; k < 64; k++) lb->b[0][k] = 0.125f;
    double tot = 0; for (int i = 0; i < 63; i++) tot += fmax(w[ord[i]], 0);   /* (ord[63]: the mean direction, eigenvalue 0) */
    for (int i = 0; i < 63; i++) {
        double sg = 0; for (int k = 0; k < 64; k++) sg += V[k][ord[i]];   /* sign: the larger half positive (deterministic files) */
        double big = 0; for (int k = 0; k < 64; k++) if (fabs(V[k][ord[i]]) > fabs(big)) big = V[k][ord[i]];
        (void)sg; const double sn = big < 0 ? -1 : 1;
        for (int k = 0; k < 64; k++) lb->b[i + 1][k] = (float)(sn * V[k][ord[i]]);
        lb->sigma[i + 1] = (float)sqrt(fmax(w[ord[i]], 0));
    }
    fprintf(stderr, "lat-fit: %ld cells; cell-mean coefficient sd %.3f; AC variance %.4f per cell (%.5f per voxel); AC energy in the first k axes:", g_n, lb->sigma[0], tot, tot / 64);
    { double c = 0; for (int i = 0; i < 63; i++) { c += fmax(w[ord[i]], 0); if (i == 2 || i == 6 || i == 12 || i == 19 || i == 28 || i == 44 || i == 62) fprintf(stderr, " %d: %.4f", i + 1, c / tot); } fprintf(stderr, "\n"); }
    /* rate / distortion on the reservoir: zeroth-order entropy of the quantised coefficients per channel count and step */
    const long nb = g_nres; float *co = malloc((size_t)nb * 64 * sizeof(float));
    double var = 0, mean = 0;
    for (long i = 0; i < nb; i++) { const float *v = g_res + (size_t)i * 64; for (int c = 0; c < 64; c++) { double s = 0; for (int k = 0; k < 64; k++) s += (double)lb->b[c][k] * v[k]; co[(size_t)i * 64 + c] = (float)s; }
                                    for (int k = 0; k < 64; k++) { mean += v[k]; var += (double)v[k] * v[k]; } }
    mean /= (double)nb * 64; var = var / ((double)nb * 64) - mean * mean;
    fprintf(stderr, "lat-fit: rate / distortion on %ld cells (normalised CT, voxel variance %.4f): channels (cell mean + AC axes), step -> bits per voxel, SNR dB, rms error\n", nb, var);
    static const int Cs[] = {4, 8, 13, 21, 29, 45, 64}; static const float Ds[] = {1.f, 0.7f, 0.5f, 0.35f, 0.25f, 0.18f, 0.125f, 0.09f, 0.0625f};
    for (size_t ci = 0; ci < sizeof Cs / sizeof *Cs; ci++) {
        fprintf(stderr, "  C %2d:", Cs[ci]);
        for (size_t di = 0; di < sizeof Ds / sizeof *Ds; di++) {
            const int C = Cs[ci]; const float D = Ds[di]; double bits = 0, mse = 0;
            for (int c = 0; c < 64; c++) {
                if (c >= C) { for (long i = 0; i < nb; i++) { const double x = co[(size_t)i * 64 + c]; mse += x * x; } continue; }
                static long hh[65536]; memset(hh, 0, sizeof hh);   /* (the cell mean: 16-bit range; the AC axes: one byte) */
                for (long i = 0; i < nb; i++) { const float x = co[(size_t)i * 64 + c]; int q = (int)lrintf(x / D); { const int lim = c ? 127 : 32767; q = q < -lim ? -lim : q > lim ? lim : q; } hh[(q + 32768) & 65535]++; const double e = x - q * D; mse += e * e; }
                for (int k = 0; k < 65536; k++) if (hh[k]) { const double p = (double)hh[k] / (double)nb; bits -= p * log2(p); }
            }
            mse /= (double)nb * 64;
            fprintf(stderr, " %.3g: %.3f b %.1f dB %.3f |", D, bits / 64, 10 * log10(var / fmax(mse, 1e-30)), sqrt(mse));
        }
        fprintf(stderr, "\n");
    }
    free(co);
    FILE *f = fopen(path, "wb"); if (!f) { fprintf(stderr, "lat-fit: cannot write %s\n", path); return 1; }
    const uint32_t hd[2] = {0x424c4655u, 1};   /* "UFLB", version */
    const int ok = fwrite(hd, sizeof hd, 1, f) == 1 && fwrite(lb, sizeof *lb, 1, f) == 1; if (fclose(f) || !ok) { fprintf(stderr, "lat-fit: cannot write %s\n", path); return 1; }
    fprintf(stderr, "lat-fit: wrote %s\n", path); free(lb);
    return 0;
}
int lat_basis_load(const char *path, lat_basis *lb) {
    FILE *f = fopen(path, "rb"); if (!f) return 1;
    uint32_t hd[2]; const int ok = fread(hd, sizeof hd, 1, f) == 1 && hd[0] == 0x424c4655u && hd[1] == 1 && fread(lb, sizeof *lb, 1, f) == 1;
    fclose(f); return !ok;
}
