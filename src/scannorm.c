/* per-scan CT normalisation: see scannorm.h */
#include "scannorm.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { SN_K = 48, SN_B = 32 };

int scan_norm_stats(z3 *(*open)(void *ctx, int level), void *ctx, int L, float *center, float *scale, int nthreads) {
    z3 *zl = open(ctx, L);
    if (!zl) return -2;
    const z3_meta *ml = z3_meta_of(zl);
    z3 *zc = nullptr, *zs = nullptr; int Lc = -1, Ls = -1; double ns = 0;
    for (int l = L + 2; l < L + 10 && l < 16; l++) {
        z3 *z = open(ctx, l); if (!z) continue;
        const z3_meta *m = z3_meta_of(z); const double n = (double)m->shape[0] * m->shape[1] * m->shape[2];
        if (n <= (double)(1 << 26)) { zc = z; Lc = l; break; }
        if (!zs || n < ns) { zs = z; Ls = l; ns = n; }
    }
    if (!zc && zs) { zc = zs; Lc = Ls; }   /* (a pyramid that stops early: its coarsest level, read subsampled) */
    if (!zc) return -3;
    const z3_meta *mc = z3_meta_of(zc);
    const int64_t full[3] = {mc->shape[0], mc->shape[1], mc->shape[2]};
    int st = 1; while ((double)full[0] * full[1] * full[2] / ((double)st * st * st) > (double)(1 << 26)) st *= 2;   /* every st-th voxel per axis */
    const int64_t nc[3] = {(full[0] + st - 1) / st, (full[1] + st - 1) / st, (full[2] + st - 1) / st};
    const size_t N = (size_t)nc[0] * nc[1] * nc[2];
    const int64_t sp = st * ((64 + st - 1) / st);   /* slab planes (a multiple of st) */
    uint8_t *cb = malloc(N), *bb = malloc((size_t)SN_B * SN_B * SN_B), *sl = st > 1 ? malloc((size_t)sp * full[1] * full[2]) : nullptr;
    if (!cb || !bb || (st > 1 && !sl)) { free(cb); free(bb); free(sl); return -4; }
    if (st == 1) { const int64_t o0[3] = {0, 0, 0}; if (z3_read(zc, o0, full, cb, nthreads)) { free(cb); free(bb); return -4; } }
    else for (int64_t z0 = 0; z0 < full[0]; z0 += sp) {   /* z slabs, subsampled into the compact grid */
        const int64_t o[3] = {z0, 0, 0}, n[3] = {z0 + sp < full[0] ? sp : full[0] - z0, full[1], full[2]};
        if (z3_read(zc, o, n, sl, nthreads)) { free(cb); free(bb); free(sl); return -4; }
        for (int64_t z = 0; z < n[0]; z += st) for (int64_t y = 0; y < full[1]; y += st) for (int64_t x = 0; x < full[2]; x += st)
            cb[((size_t)((z0 + z) / st) * nc[1] + y / st) * nc[2] + x / st] = sl[((size_t)z * full[1] + y) * full[2] + x];
    }
    free(sl);
    size_t nz = 0; for (size_t k = 0; k < N; k++) nz += cb[k] != 0;
    if (nz < SN_K) { free(cb); free(bb); return -5; }
    size_t want[SN_K]; for (int k = 0; k < SN_K; k++) want[k] = (size_t)(((2.0 * k + 1) * (double)nz) / (2.0 * SN_K));   /* fixed ranks, raster order */
    uint64_t h[256] = {0};
    const double f = (double)(1 << (Lc - L)) * st;
    size_t rank = 0; int k = 0;
    for (size_t q = 0; q < N && k < SN_K; q++) {
        if (!cb[q]) continue;
        if (rank++ != want[k]) continue;
        const int64_t c[3] = {(int64_t)(q / ((size_t)nc[1] * nc[2])), (int64_t)((q / nc[2]) % nc[1]), (int64_t)(q % nc[2])};
        int64_t o[3], n[3];
        for (int d = 0; d < 3; d++) {
            n[d] = ml->shape[d] < SN_B ? ml->shape[d] : SN_B;
            int64_t a = (int64_t)llround((c[d] + 0.5) * f) - n[d] / 2;
            o[d] = a < 0 ? 0 : a + n[d] > ml->shape[d] ? ml->shape[d] - n[d] : a;
        }
        if (z3_read(zl, o, n, bb, nthreads)) { free(cb); free(bb); return -6; }
        const size_t nb = (size_t)n[0] * n[1] * n[2];
        for (size_t i = 0; i < nb; i++) if (bb[i]) h[bb[i]]++;
        k++;
    }
    free(cb); free(bb);
    uint64_t tot = 0; double sum = 0, sq = 0; for (int v = 1; v < 256; v++) { tot += h[v]; sum += (double)v * h[v]; sq += (double)v * v * h[v]; }
    if (tot < 1000) return -7;
    /* mean / sd, not median / MAD: inside the scroll the nonzero voxels are bimodal (air in the mask, papyrus), and the MAD measures the
       dominant mode alone (1667 level 2: median 45, 1.4826 MAD 10.4 against a window sd of ~40): papyrus would sit at +7 */
    const double mean = sum / tot, sd = sqrt(fmax(sq / tot - mean * mean, 0));
    *center = (float)mean; *scale = (float)(sd > 0 ? sd : 1.0);
    return 0;
}
