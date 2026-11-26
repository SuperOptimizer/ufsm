/* train --ssl-views 2 (src/ssl.h, DESIGN 2026-11-26 "SSL v2"): the teacher's view record and the student -> teacher cell map, on the host
   from the windows' radial records (nn.h NN_RAD_REC: header int32[8], cyz[P], cxz[P], then for warped windows table[6 P], R[9], A[9],
   Ainv[9], scale). The pull is nn_aug.cu aug_write_k's (output voxel -> source position) in double, at continuous positions: the elastic
   table linear between its integer entries. */
#include "ssl.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

static const float *rec_warp(const void *rec, int P) { return (const float *)rec + 8 + 2 * (size_t)P; }
int ssl2_pull(const void *rec, int P, const double io[3], double V[3]) {
    const int32_t *h = (const int32_t *)rec; const int sb = h[5], warped = h[6] & 1, affine = h[6] >> 1 & 1;
    const int perm[3] = {sb & 3, sb >> 2 & 3, sb >> 4 & 3}, flip[3] = {sb >> 8 & 1, sb >> 9 & 1, sb >> 10 & 1};
    double U[3];
    if (!warped) for (int d = 0; d < 3; d++) U[d] = io[d];
    else {
        const float *tb = rec_warp(rec, P), *R = tb + 6 * (size_t)P, *A = R + 9; const double sc = R[27], c = 0.5 * (P - 1);
        double vt[3];
        for (int d = 0; d < 3; d++) {
            double u = io[d]; if (u < 0) u = 0; if (u > P - 1) u = P - 1; int i = (int)floor(u); if (i > P - 2) i = P - 2; if (i < 0) i = 0;
            const double f = u - i, t = (1 - f) * tb[d * P + i] + f * tb[d * P + i + (P > 1)];
            vt[d] = io[d] - c - t;
        }
        double w[3]; for (int d = 0; d < 3; d++) w[d] = (double)R[0 * 3 + d] * vt[0] + (double)R[1 * 3 + d] * vt[1] + (double)R[2 * 3 + d] * vt[2];
        if (affine) for (int d = 0; d < 3; d++) U[d] = c + sc * ((double)A[d * 3] * w[0] + (double)A[d * 3 + 1] * w[1] + (double)A[d * 3 + 2] * w[2]);
        else for (int d = 0; d < 3; d++) U[d] = c + sc * w[d];
    }
    for (int d = 0; d < 3; d++) V[perm[d]] = flip[d] ? (P - 1) - U[d] : U[d];
    int in = 1; for (int d = 0; d < 3; d++) in &= V[d] >= 0 && V[d] <= P - 1;
    return in;
}
void ssl2_teacher_rec(const void *srec, void *trec, int P, const double shift[3], double zoom) {
    const size_t nb = (size_t)(8 + 8 * P + 32) * 4;
    memset(trec, 0, nb); memcpy(trec, srec, (8 + 2 * (size_t)P) * 4);   /* (header and the axis centres per source z) */
    int32_t *h = (int32_t *)trec; h[5] = 0 | 1 << 2 | 2 << 4; h[6] = 1;   /* identity symmetry; warped, not affine */
    float *tb = (float *)trec + 8 + 2 * (size_t)P, *R = tb + 6 * (size_t)P;
    for (int d = 0; d < 3; d++) for (int u = 0; u < P; u++) tb[d * P + u] = (float)shift[d];   /* (the derivative half stays 0) */
    for (int k = 0; k < 3; k++) { R[4 * k] = 1.f; R[9 + 4 * k] = 1.f; R[18 + 4 * k] = 1.f; }
    R[27] = (float)zoom;
}
long ssl2_map(const void *srec, const void *trec, int P, int cv, float *map) {
    const int G = P / cv; const double c = 0.5 * (P - 1), cc = 0.5 * (cv - 1);
    const float *tt = rec_warp(trec, P); const double zt = rec_warp(trec, P)[6 * (size_t)P + 27];
    long n = 0;
    for (int z = 0; z < G; z++) for (int y = 0; y < G; y++) for (int x = 0; x < G; x++) {
        float *m = map + 3 * (((size_t)z * G + y) * G + x); const double io[3] = {z * cv + cc, y * cv + cc, x * cv + cc}; double V[3];
        m[0] = m[1] = m[2] = NAN;
        if (!ssl2_pull(srec, P, io, V)) continue;
        double t[3]; int in = 1;
        for (int d = 0; d < 3; d++) { const double it = c + tt[d * P] + (V[d] - c) / zt; in &= it >= 0 && it <= P - 1; t[d] = (it - cc) / cv; }
        if (!in) continue;
        for (int d = 0; d < 3; d++) m[d] = (float)t[d];
        n++;
    }
    return n;
}
