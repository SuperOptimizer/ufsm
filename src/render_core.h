/* render_core.h - the per-pixel arithmetic of ufsm render's chunk-ordered renderer (src/render.c render_wraps_chunked_fill), shared by
   the CPU path and the GPU path (src/nn_render.cu). The GPU object is built without fast math and with -fmad=false, gcc -std=c23
   contracts nothing: the same operations in the same order give the same bits. Change it here = change both.

   rc_vol: decoded chunks (edge 1 << sh) over a box of chunk indices c0 .. c0 + cn; ptr[(a cn1 + b) cn2 + c] the chunk (nullptr: zero).
   rc_trilin = main.c trilin() on the global volume (coordinates < 0 are 0 there too). rc_pixel: one output pixel (px, py) of a grid
   (w x h, Z < 0 invalid; gs output pixels per grid step): bilinear point and normal, the --fill search on volume 1 (the CT), the max over
   +- depth[v] voxels along the normal for every volume. Returns 0 when the pixel has no surface (nothing written). */
#ifndef UFSM_RENDER_CORE_H
#define UFSM_RENDER_CORE_H
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#ifdef __CUDACC__
#define RC_HD __host__ __device__ static inline
#else
#define RC_HD static inline
#endif

typedef struct { int sh; int64_t c0[3], cn[3]; const uint8_t *const *ptr; } rc_vol;
typedef struct { int n, thr, prefer; } rc_fill;   /* --fill N, --fill-ct, --fill-prefer */
enum { RC_ON = 0, RC_NEG = 50, RC_GRID = 100, RC_EMPTY = 255 };   /* = imgops.h RFILL_* */

RC_HD uint8_t rc_voxel(const rc_vol *T, int64_t z, int64_t y, int64_t x) {
    const int64_t a = (z >> T->sh) - T->c0[0], b = (y >> T->sh) - T->c0[1], c = (x >> T->sh) - T->c0[2];
    if (a < 0 || b < 0 || c < 0 || a >= T->cn[0] || b >= T->cn[1] || c >= T->cn[2]) return 0;
    const uint8_t *p = T->ptr[(a * T->cn[1] + b) * T->cn[2] + c]; const int64_t m = ((int64_t)1 << T->sh) - 1;
    return p ? p[((((z & m) << T->sh) + (y & m)) << T->sh) + (x & m)] : 0;
}
RC_HD float rc_trilin(const rc_vol *T, double z, double y, double x) {
    if (z < 0 || y < 0 || x < 0) return 0.f;
    const int z0 = (int)z, y0 = (int)y, x0 = (int)x; const double fz = z - z0, fy = y - y0, fx = x - x0;
    const int64_t m = ((int64_t)1 << T->sh) - 1; double v[8];
    if ((z0 & m) != m && (y0 & m) != m && (x0 & m) != m) {
        const int64_t a = (z0 >> T->sh) - T->c0[0], b = (y0 >> T->sh) - T->c0[1], c = (x0 >> T->sh) - T->c0[2];
        const uint8_t *p = a >= 0 && b >= 0 && c >= 0 && a < T->cn[0] && b < T->cn[1] && c < T->cn[2] ? T->ptr[(a * T->cn[1] + b) * T->cn[2] + c] : nullptr;
        if (!p) return 0.f;
        p += ((((z0 & m) << T->sh) + (y0 & m)) << T->sh) + (x0 & m);
        const size_t sy = (size_t)1 << T->sh, sz = sy << T->sh;
        v[0] = p[0]; v[1] = p[1]; v[2] = p[sy]; v[3] = p[sy + 1]; v[4] = p[sz]; v[5] = p[sz + 1]; v[6] = p[sz + sy]; v[7] = p[sz + sy + 1];
    } else for (int c = 0; c < 8; c++) v[c] = rc_voxel(T, z0 + (c >> 2 & 1), y0 + (c >> 1 & 1), x0 + (c & 1));
    double s = 0;
    for (int c = 0; c < 8; c++) s += (c >> 2 & 1 ? fz : 1 - fz) * (c >> 1 & 1 ? fy : 1 - fy) * (c & 1 ? fx : 1 - fx) * v[c];
    return (float)s;
}
/* val[v]: the render of volume v; *fl (F only): the --fill flag (RC_ON, +k: k, -k: RC_NEG + k, RC_EMPTY; + RC_GRID when a corner is a
   bridged grid-hole vertex, vgrid nonzero) */
RC_HD int rc_pixel(const float *Zg, const float *Yg, const float *Xg, int w, int h, double gs, int px, int py, const rc_vol *V, int nv,
                   const int *depth, const rc_fill *F, const uint8_t *vgrid, uint8_t *val, uint8_t *fl) {
    const float *A[3] = {Zg, Yg, Xg};
    const double gu = px / gs, gv = py / gs;
    int c0 = (int)gu, r0 = (int)gv; if (c0 >= w - 1) c0 = w - 2; if (r0 >= h - 1) r0 = h - 2; if (c0 < 0 || r0 < 0) return 0;
    const double fu = gu - c0, fv = gv - r0;
    const size_t i00 = (size_t)r0 * w + c0, i01 = i00 + 1, i10 = i00 + w, i11 = i10 + 1;
    if (!(Zg[i00] >= 0 && Zg[i01] >= 0 && Zg[i10] >= 0 && Zg[i11] >= 0)) return 0;
    double P[3], du[3], dv[3];
    for (int d = 0; d < 3; d++) {
        const double a = A[d][i00], b = A[d][i01], c = A[d][i10], e = A[d][i11];
        P[d] = (1 - fv) * ((1 - fu) * a + fu * b) + fv * ((1 - fu) * c + fu * e);
        du[d] = (1 - fv) * (b - a) + fv * (e - c); dv[d] = (1 - fu) * (c - a) + fu * (e - b);
    }
    double N[3] = {du[1] * dv[2] - du[2] * dv[1], du[2] * dv[0] - du[0] * dv[2], du[0] * dv[1] - du[1] * dv[0]};
    const double nl = sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]); if (nl < 1e-12) return 0;
    for (int d = 0; d < 3; d++) N[d] /= nl;
    if (F) {   /* --fill: void CT on the surface -> the nearest papyrus along the normal, every volume centred there */
        const rc_vol *C = &V[1]; int k = 0; uint8_t f = RC_ON;
        if (lroundf(rc_trilin(C, P[0], P[1], P[2])) < F->thr) {
            f = RC_EMPTY;
            for (int j = 1; j <= F->n; j++) {
                const float cp = rc_trilin(C, P[0] + j * N[0], P[1] + j * N[1], P[2] + j * N[2]), cm = rc_trilin(C, P[0] - j * N[0], P[1] - j * N[1], P[2] - j * N[2]);
                const int okp = lroundf(cp) >= F->thr, okm = lroundf(cm) >= F->thr;
                if (!okp && !okm) continue;
                k = okp && okm ? (F->prefer > 0 ? j : F->prefer < 0 ? -j : cm > cp ? -j : j) : okp ? j : -j;
                f = (uint8_t)(k > 0 ? k : RC_NEG - k); break;
            }
            if (k) for (int d = 0; d < 3; d++) P[d] += k * N[d];
        }
        if (vgrid && f != RC_EMPTY && (vgrid[i00] || vgrid[i01] || vgrid[i10] || vgrid[i11])) f += RC_GRID;
        *fl = f;
    }
    for (int v = 0; v < nv; v++) {
        float m = 0;
        for (int t = -depth[v]; t <= depth[v]; t++) { const float s = rc_trilin(&V[v], P[0] + t * N[0], P[1] + t * N[1], P[2] + t * N[2]); if (s > m) m = s; }
        val[v] = (uint8_t)lroundf(m);
    }
    return 1;
}
#endif
