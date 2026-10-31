/* ufsm faces: the per-profile and per-window kernels shared by the CPU (src/faces.c) and the GPU (src/nn_faces.cu) paths. Plain C
   arithmetic in a fixed order, no fused multiply-adds (gcc -std=c23 contracts nothing; nvcc builds nn_faces.cu with -fmad=false and
   IEEE division), so both paths give bit-identical results. */
#pragma once
#include "faces.h"
#include <math.h>

#ifdef __CUDACC__
#define FC_HD __host__ __device__ __forceinline__
#else
#define FC_HD static inline
#endif

/* one profile f[0..m) (smoothed in place) -> bands; ex: scratch of m ints. The algorithm of faces_bands (src/faces.h). */
FC_HD int fc_bands_core(float *f, int *ex, int m, double t0, double h, double min_prom, double rel, fc_band *out, int kmax) {
    if (m < 5 || m > 8192 || kmax < 1) return 0;
    for (int k = 0; k < m; k++) if (!(f[k] == f[k])) return 0;   /* NaN: the profile leaves the volume */
    for (int pass = 0; pass < 2; pass++) {   /* [1 2 1] / 4 twice: trilinear kinks and noise below a voxel */
        float prev = f[0];
        for (int k = 1; k < m - 1; k++) { const float cur = f[k]; f[k] = 0.25f * (prev + 2 * cur + f[k + 1]); prev = cur; }
    }
    float V = f[0]; for (int k = 1; k < m; k++) if (f[k] < V) V = f[k];
    /* alternating extrema min, max, min, ..., min (the profile ends count as minima) */
    int ne = 0; ex[ne++] = 0;
    for (int k = 1; k < m - 1; k++) {
        const int last_max = ne & 1 ? 0 : 1;   /* ex[ne - 1] is a max when ne is even */
        if (!last_max) {   /* looking for a max; a lower min replaces the last min */
            if (f[k] < f[ex[ne - 1]]) ex[ne - 1] = k;
            else if (f[k] > f[k - 1] && f[k] >= f[k + 1]) ex[ne++] = k;
        } else {
            if (f[k] > f[ex[ne - 1]]) ex[ne - 1] = k;
            else if (f[k] < f[k - 1] && f[k] <= f[k + 1]) ex[ne++] = k;
        }
    }
    if (!(ne & 1)) { if (f[m - 1] < f[ex[ne - 1]]) ex[ne++] = m - 1; else ne--; }   /* ends on a max: close with the end, or drop a rise */
    else if (f[m - 1] < f[ex[ne - 1]]) ex[ne - 1] = m - 1;
    /* persistence: drop the shallowest interior dip (and the lower of its two peaks) while it is below the floor */
    for (;;) {
        int bj = -1; double br = 1;
        for (int j = 2; j + 2 < ne; j += 2) {
            const float pa = f[ex[j - 1]], pb = f[ex[j + 1]], lo = pa < pb ? pa : pb, d = lo - f[ex[j]];
            const double thr = fmax(min_prom, rel * (lo - V)), r = d / thr;
            if (r < br) { br = r; bj = j; }
        }
        if (bj < 0) break;
        const int drop = f[ex[bj - 1]] < f[ex[bj + 1]] ? bj - 1 : bj + 1;   /* the lower peak goes with the dip */
        const int a = drop < bj ? drop : bj;
        for (int q = a; q + 2 < ne; q++) ex[q] = ex[q + 2];
        ne -= 2;
    }
    int nb = 0;
    for (int j = 1; j + 1 < ne && nb < kmax; j += 2) {
        const int p = ex[j], vl = ex[j - 1], vr = ex[j + 1]; const float fp = f[p];
        if (fp - V < 2 * min_prom) continue;   /* no contrast over the void */
        const float half = V + 0.5f * (fp - V);
        double e[2] = {0, 0}; int ok = 1;
        for (int sd = 0; sd < 2 && ok; sd++) {
            const int v = sd ? vr : vl, end = v == 0 || v == m - 1;
            if (f[v] > half) {   /* still bright at the base: a contact valley (edge there) or papyrus past the profile end */
                if (end) { ok = 0; break; }
                e[sd] = t0 + h * v; continue;
            }
            const float lev = f[v] + 0.5f * (fp - f[v]);
            if (!sd) { int k = p; while (k > v && f[k] > lev) k--; e[0] = t0 + h * (k + (lev - f[k]) / (f[k + 1] - f[k] != 0 ? f[k + 1] - f[k] : 1)); }
            else { int k = p; while (k < v && f[k] > lev) k++; e[1] = t0 + h * (k - (lev - f[k]) / (f[k - 1] - f[k] != 0 ? f[k - 1] - f[k] : 1)); }
        }
        if (!ok || !(e[1] > e[0])) continue;
        const double di = fmin(1.0, 0.3 * (e[1] - e[0]));   /* just inside each face: its texture (fibres) */
        float cv[2];
        for (int sd = 0; sd < 2; sd++) {
            const double k = ((sd ? e[1] - di : e[0] + di) - t0) / h; int k0 = (int)k; if (k0 > m - 2) k0 = m - 2; if (k0 < 0) k0 = 0;
            cv[sd] = (float)(f[k0] + (k - k0) * (f[k0 + 1] - f[k0]));
        }
        out[nb].near = (float)e[0]; out[nb].far = (float)e[1]; out[nb].peak = fp; out[nb].prom = fp - V; out[nb].cin = cv[0]; out[nb].cout = cv[1];
        nb++;
    }
    return nb;
}

/* k-th smallest of a[0..n) (quickselect, reorders a) */
FC_HD float fc_select(float *a, int n, int k) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        const float pv = a[(lo + hi) / 2]; int i = lo, j = hi;
        while (i <= j) { while (a[i] < pv) i++; while (a[j] > pv) j--; if (i <= j) { const float t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; } }
        if (k <= j) hi = j; else if (k >= i) lo = i; else break;
    }
    return a[k];
}

/* the trilinear CT sample of the cache (faces_cli fcc_sample): corner values v[8] (k = 4 dz + 2 dy + dx), fractions */
FC_HD double fc_trilin(const int *v, double fz, double fy, double fx) {
    double s = 0;
    for (int k = 0; k < 8; k++) s += (k >> 2 & 1 ? fz : 1 - fz) * (k >> 1 & 1 ? fy : 1 - fy) * (k & 1 ? fx : 1 - fx) * v[k];
    return s;
}
