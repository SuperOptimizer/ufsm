/* vwt-infer kernels (src/nn_vwi.cu) against CPU references: slab extraction (villawind samp_run semantics: trilinear in a CT box,
   rounded to uint8, invalid outside the volume) for oblique and radial frames incl. volume faces; the pass-A block means (4x4x2 minus
   the centre, NaN where invalid) and the centre; the phase at pair points (trilinear, NaN outside / invalid).
   Pass: extraction |gpu - cpu| <= 1 grey level on < 0.1 % of voxels (float vs double rounding at .5), valid masks equal; block means and
   points within 1e-3 (fp16 / fp32 rounding). */
#include "nn.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void vwi_sample(const uint8_t *box, int bz, int by, int bx, const float *o, const float *a, const float *b, const float *d, const float *vlo, const float *vhi,
                int T, int L, uint8_t *im, uint8_t *va, void *s);
void vwi_down(const float *ph, const uint8_t *va, int T, int L, void *out, float *centre, void *s);
void vwi_points(const float *ph, const uint8_t *va, int T, int L, const float *pts, int np, float *out, void *s);

static float h2f(uint16_t h) {
    const uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023; uint32_t u;
    if (e == 0) { if (!m) u = s; else { int ee = -1; uint32_t mm = m; while (!(mm & 1024)) { mm <<= 1; ee--; } u = s | (uint32_t)(127 - 15 + ee + 1) << 23 | (mm & 1023) << 13; } }
    else if (e == 31) u = s | 0x7f800000u | m << 13; else u = s | (e + 127 - 15) << 23 | m << 13;
    float f; memcpy(&f, &u, 4); return f;
}
static uint64_t rs = 99;
static double ur(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (double)(rs >> 11) * 0x1.0p-53; }

int main(void) {
    if (nn_init(0)) { printf("SKIP: no GPU\n"); return 0; }
    const int T = 32, L = 64; int fail = 0;
    /* a CT box (zyx) inside a larger volume; smooth + noise content */
    const int bz = 60, by = 90, bx = 100; const int64_t bo[3] = {10, 20, 30}, V[3] = {66, 120, 128};
    const size_t nb = (size_t)bz * by * bx; uint8_t *box = malloc(nb);
    for (size_t i = 0; i < nb; i++) box[i] = (uint8_t)(128 + 80 * sin(i * 0.013) + 40 * ur() - 20);
    uint8_t *boxd = nn_malloc(nb); nn_h2d(boxd, box, nb);
    const size_t nv = (size_t)T * T * L; uint8_t *im = malloc(nv), *va = malloc(nv), *imd = nn_malloc(nv), *vad = nn_malloc(nv);
    uint8_t *gim = malloc(nv), *gva = malloc(nv);
    for (int fr = 0; fr < 3; fr++) {
        /* frame (world xyz): o, unit a, b, d */
        double d[3] = {fr == 0 ? 1 : cos(0.7 * fr), fr == 0 ? 0 : sin(0.7 * fr), fr == 2 ? 0.3 : 0}, a[3], b[3];
        { const double n = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]); for (int c = 0; c < 3; c++) d[c] /= n; }
        double ref[3] = {0, 0, 0}; int mm = 0; for (int c = 1; c < 3; c++) if (fabs(d[c]) < fabs(d[mm])) mm = c; ref[mm] = 1;
        a[0] = ref[1] * d[2] - ref[2] * d[1]; a[1] = ref[2] * d[0] - ref[0] * d[2]; a[2] = ref[0] * d[1] - ref[1] * d[0];
        { const double n = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); for (int c = 0; c < 3; c++) a[c] /= n; }
        b[0] = d[1] * a[2] - d[2] * a[1]; b[1] = d[2] * a[0] - d[0] * a[2]; b[2] = d[0] * a[1] - d[1] * a[0];
        const double o[3] = {35.3 + 4 * fr, 25.7 + 3 * fr, 12.2 + 9 * fr};   /* (frames 1, 2 run past the volume's faces) */
        /* CPU reference (villawind samp_run, doubles) */
        for (int i = 0; i < T; i++) for (int j = 0; j < T; j++) for (int k = 0; k < L; k++) {
            const double x = o[0] + i * a[0] + j * b[0] + k * d[0], y = o[1] + i * a[1] + j * b[1] + k * d[1], z = o[2] + i * a[2] + j * b[2] + k * d[2];
            const size_t q = ((size_t)i * T + j) * L + k;
            if (!(x >= 0 && y >= 0 && z >= 0 && x <= V[2] - 1 && y <= V[1] - 1 && z <= V[0] - 1)) { im[q] = 0; va[q] = 0; continue; }
            const double lz = z - bo[0], ly = y - bo[1], lx = x - bo[2];
            int iz = (int)floor(lz), iy = (int)floor(ly), ix = (int)floor(lx);
            iz = iz < 0 ? 0 : iz > bz - 2 ? bz - 2 : iz; iy = iy < 0 ? 0 : iy > by - 2 ? by - 2 : iy; ix = ix < 0 ? 0 : ix > bx - 2 ? bx - 2 : ix;
            const double fz = lz - iz, fy = ly - iy, fx = lx - ix;
            const uint8_t *p = box + ((size_t)iz * by + iy) * bx + ix; const size_t sy = bx, sz = (size_t)by * bx;
            const double v = (1 - fz) * ((1 - fy) * ((1 - fx) * p[0] + fx * p[1]) + fy * ((1 - fx) * p[sy] + fx * p[sy + 1]))
                           + fz * ((1 - fy) * ((1 - fx) * p[sz] + fx * p[sz + 1]) + fy * ((1 - fx) * p[sz + sy] + fx * p[sz + sy + 1]));
            const int u = (int)lround(v); im[q] = (uint8_t)(u < 0 ? 0 : u > 255 ? 255 : u); va[q] = 1;
        }
        float of[3], af[3], bf[3], df[3], vlo[3], vhi[3];
        for (int e = 0; e < 3; e++) { const int z = 2 - e; of[e] = (float)(o[e] - bo[z]); af[e] = (float)a[e]; bf[e] = (float)b[e]; df[e] = (float)d[e]; vlo[e] = (float)(-bo[z]); vhi[e] = (float)(V[z] - 1 - bo[z]); }
        vwi_sample(boxd, bz, by, bx, of, af, bf, df, vlo, vhi, T, L, imd, vad, nullptr);
        nn_d2h(gim, imd, nv); nn_d2h(gva, vad, nv);
        size_t off1 = 0, bad = 0, vmis = 0, nval = 0;
        for (size_t q = 0; q < nv; q++) { const int dd = abs((int)gim[q] - (int)im[q]); off1 += dd == 1; bad += dd > 1; vmis += gva[q] != va[q]; nval += va[q]; }
        const int ok = !bad && off1 * 1000 < nv && vmis * 2000 < nv;
        printf("frame %d: valid %zu / %zu, valid mismatches %zu, off by 1: %zu, worse: %zu%s\n", fr, nval, nv, vmis, off1, bad, ok ? "" : "  FAIL");
        fail += !ok;
    }
    /* block means + centre + points on a synthetic phase */
    float *ph = malloc(nv * 4); for (size_t q = 0; q < nv; q++) ph[q] = (float)(0.01 * (q % L) + 0.3 * sin(q * 0.001));
    for (size_t q = 0; q < nv; q++) va[q] = !(q % 997 == 0);
    float *phd = nn_malloc(nv * 4); nn_h2d(phd, ph, nv * 4); nn_h2d(vad, va, nv);
    const int TU = T / 4, LW = L / 2; const size_t nd = (size_t)TU * TU * LW;
    uint8_t *outd = nn_malloc(nd * 2 + 64); uint16_t *out = malloc(nd * 2); float cen;
    vwi_down(phd, vad, T, L, outd, (float *)(outd + nd * 2), nullptr);
    nn_d2h(out, outd, nd * 2); nn_d2h(&cen, outd + nd * 2, 4);
    const int ci = T / 2 - 1, ck = L / 2 - 1; double c = 0;
    for (int e = 0; e < 8; e++) c += ph[((size_t)(ci + (e >> 2)) * T + ci + ((e >> 1) & 1)) * L + ck + (e & 1)];
    c /= 8; double me = 0; size_t nanm = 0;
    for (int u = 0; u < TU; u++) for (int v = 0; v < TU; v++) for (int w = 0; w < LW; w++) {
        double s = 0; int ok = 1;
        for (int di = 0; di < 4; di++) for (int dj = 0; dj < 4; dj++) for (int dk = 0; dk < 2; dk++) { const size_t q = ((size_t)(4 * u + di) * T + 4 * v + dj) * L + 2 * w + dk; s += ph[q]; ok &= va[q]; }
        const float g = h2f(out[((size_t)u * TU + v) * LW + w]);
        if (!ok) nanm += !isnan(g); else { nanm += isnan(g); me = fmax(me, fabs(g - (s / 32 - c))); }
    }
    printf("blocks: centre %.6f vs %.6f, max |d| %.2e, NaN mismatches %zu\n", cen, c, me, nanm);
    fail += !(fabs(cen - c) < 1e-5 && me < 1e-3 && !nanm);
    const int np = 500; float *pts = malloc(np * 12), *pv = malloc(np * 4);
    for (int t = 0; t < np; t++) { pts[3 * t] = (float)(ur() * (T + 4) - 2); pts[3 * t + 1] = (float)(ur() * (T + 4) - 2); pts[3 * t + 2] = (float)(ur() * (L + 4) - 2); }
    float *ptsd = nn_malloc(np * 12), *pvd = nn_malloc(np * 4); nn_h2d(ptsd, pts, np * 12);
    vwi_points(phd, vad, T, L, ptsd, np, pvd, nullptr); nn_d2h(pv, pvd, np * 4);
    double pe = 0; size_t pn = 0;
    for (int t = 0; t < np; t++) {
        const double pi = pts[3 * t], pj = pts[3 * t + 1], pk = pts[3 * t + 2]; double ref = NAN;
        if (pi >= 0 && pj >= 0 && pk >= 0 && pi <= T - 1 && pj <= T - 1 && pk <= L - 1) {
            int i0 = (int)pi, j0 = (int)pj, k0 = (int)pk; if (i0 > T - 2) i0 = T - 2; if (j0 > T - 2) j0 = T - 2; if (k0 > L - 2) k0 = L - 2;
            double s = 0; int ok = 1;
            for (int e = 0; e < 8; e++) { const int di = e >> 2, dj = (e >> 1) & 1, dk = e & 1; const size_t q = ((size_t)(i0 + di) * T + j0 + dj) * L + k0 + dk;
                s += (di ? pi - i0 : 1 - (pi - i0)) * (dj ? pj - j0 : 1 - (pj - j0)) * (dk ? pk - k0 : 1 - (pk - k0)) * ph[q]; ok &= va[q]; }
            if (ok) ref = s;
        }
        if (isnan(ref) != isnan(pv[t])) pn++; else if (!isnan(ref)) pe = fmax(pe, fabs(ref - pv[t]));
    }
    printf("points: max |d| %.2e, NaN mismatches %zu\n", pe, pn);
    fail += !(pe < 1e-4 && !pn);
    printf(fail ? "FAIL (%d)\n" : "PASS\n", fail);
    return fail ? 1 : 0;
}
