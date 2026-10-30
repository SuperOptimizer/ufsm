/* ufsm spiral-fit (src/spiralfit.h): the villa diffeomorphic spiral model, its losses and their hand-written adjoint. The core is
   templated on the arithmetic type and compiled for host and device: the GPU fit runs it in float, the CPU reference
   (nn_spiral_gradcheck) in double against finite differences. */
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
extern "C" {
#include "spiralfit.h"
}

#define SF_HD __host__ __device__ __forceinline__
#define SF_TWO_PI 6.283185307179586

struct SFM {
    int nst, nrk;
    int n[2][3]; float L[2]; float o[3];   /* lattice l: nodes n[l][z,y,x], spacing L[l]; node i of an axis at o + (i - 1) L */
    const float *lat[2]; float *glat[2];   /* [stage][nz][ny][nx][3] velocity (native voxels per unit time) and its gradient */
    int K, Nz, Nt; float Lz, dr; int dir;  /* gap: windings, z bins, angle bins, z bin size, pitch, sense */
    const float *C; float *Cb;             /* cumulative gaps C[k][zb][tb] (k = 0..K) and their gradient */
    const float *ay, *ax; int na;          /* axis table per native z */
    float uoff;                            /* the label offset added to U' (an integer: the initial U's numbering) */
};

template <class T> SF_HD void sf_acc(float *p, T v) {
#ifdef __CUDA_ARCH__
    atomicAdd(p, (float)v);
#else
    *p += (float)v;
#endif
}
template <class T> SF_HD T sf_tab(const float *v, int n, T z) {
    if (z <= 0) return v[0];
    if (z >= (T)(n - 1)) return v[n - 1];
    const int i = (int)z; const T t = z - (T)i; return v[i] + t * (v[i + 1] - v[i]);
}
template <class T> SF_HD void sf_bsw(T u, T *w, T *dw) {
    const T u2 = u * u, u3 = u2 * u, om = 1 - u;
    w[0] = om * om * om / 6; w[1] = (3 * u3 - 6 * u2 + 4) / 6; w[2] = (-3 * u3 + 3 * u2 + 3 * u + 1) / 6; w[3] = u3 / 6;
    dw[0] = -om * om / 2; dw[1] = (T)1.5 * u2 - 2 * u; dw[2] = (T)-1.5 * u2 + u + (T)0.5; dw[3] = u2 / 2;
}
/* the taps of lattice l along the 3 axes at y: clamped indices, weights, derivative weights (per native voxel) */
template <class T> SF_HD void sf_taps(const SFM &m, int l, const T *y, int (*ix)[4], T (*w)[4], T (*dw)[4]) {
    for (int a = 0; a < 3; a++) {
        const T t = (y[a] - (T)m.o[a]) / (T)m.L[l] + 1; T fl = floor(t); const T u = t - fl; const int i0 = (int)fl;
        sf_bsw(u, w[a], dw[a]);
        for (int q = 0; q < 4; q++) { int i = i0 - 1 + q; i = i < 0 ? 0 : i >= m.n[l][a] ? m.n[l][a] - 1 : i; ix[a][q] = i; dw[a][q] /= (T)m.L[l]; }
    }
}
/* v (+)= velocity of stage s at y, J (+)= dv/dy (row-major [component][axis]) */
template <class T> SF_HD void sf_vel(const SFM &m, int s, const T *y, T *v, T *J) {
    for (int k = 0; k < 3; k++) v[k] = 0;
    if (J) for (int k = 0; k < 9; k++) J[k] = 0;
    for (int l = 0; l < 2; l++) {
        int ix[3][4]; T w[3][4], dw[3][4]; sf_taps(m, l, y, ix, w, dw);
        const float *lat = m.lat[l] + (size_t)s * m.n[l][0] * m.n[l][1] * m.n[l][2] * 3;
        for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) {
            const size_t row = ((size_t)ix[0][a] * m.n[l][1] + ix[1][b]) * m.n[l][2];
            const T wab = w[0][a] * w[1][b], dab0 = dw[0][a] * w[1][b], dab1 = w[0][a] * dw[1][b];
            for (int c = 0; c < 4; c++) {
                const float *p = lat + (row + ix[2][c]) * 3; const T wt = wab * w[2][c];
                const T c0 = p[0], c1 = p[1], c2 = p[2];
                v[0] += wt * c0; v[1] += wt * c1; v[2] += wt * c2;
                if (J) {
                    const T d0 = dab0 * w[2][c], d1 = dab1 * w[2][c], d2 = wab * dw[2][c];
                    J[0] += d0 * c0; J[1] += d1 * c0; J[2] += d2 * c0;
                    J[3] += d0 * c1; J[4] += d1 * c1; J[5] += d2 * c1;
                    J[6] += d0 * c2; J[7] += d1 * c2; J[8] += d2 * c2;
                }
            }
        }
    }
}
template <class T> SF_HD void sf_vel_scatter(const SFM &m, int s, const T *y, const T *vb) {
    for (int l = 0; l < 2; l++) {
        int ix[3][4]; T w[3][4], dw[3][4]; sf_taps(m, l, y, ix, w, dw);
        float *g = m.glat[l] + (size_t)s * m.n[l][0] * m.n[l][1] * m.n[l][2] * 3;
        for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) {
            const size_t row = ((size_t)ix[0][a] * m.n[l][1] + ix[1][b]) * m.n[l][2]; const T wab = w[0][a] * w[1][b];
            for (int c = 0; c < 4; c++) { float *p = g + (row + ix[2][c]) * 3; const T wt = wab * w[2][c]; sf_acc(p, wt * vb[0]); sf_acc(p + 1, wt * vb[1]); sf_acc(p + 2, wt * vb[2]); }
        }
    }
}

/* ---- the gap transform at a deformed point ---- */
struct SFGapIx { int k, z0, z1, t0; };
template <class T> struct SFGap { T th, r, f, D, U, dRk[2], dRk1[2], ut, uz; SFGapIx ix; };   /* dR*: d/dtheta, d/dz */
template <class T> SF_HD T sf_C(const SFM &m, int k, int zi, int ti) {
    return ti < m.Nt ? (T)m.C[((size_t)k * m.Nz + zi) * m.Nt + ti] : (T)m.C[((size_t)(k + 1) * m.Nz + zi) * m.Nt] - (T)m.dr;
}
template <class T> SF_HD T sf_R(const SFM &m, int k, const T th, const SFGap<T> &g, T *dth, T *dz) {
    const T c00 = sf_C<T>(m, k, g.ix.z0, g.ix.t0), c01 = sf_C<T>(m, k, g.ix.z0, g.ix.t0 + 1), c10 = sf_C<T>(m, k, g.ix.z1, g.ix.t0), c11 = sf_C<T>(m, k, g.ix.z1, g.ix.t0 + 1);
    if (dth) { *dth = (T)m.dr / (T)SF_TWO_PI + (T)m.Nt / (T)SF_TWO_PI * ((1 - g.uz) * (c01 - c00) + g.uz * (c11 - c10));
               *dz = g.ix.z1 != g.ix.z0 ? ((1 - g.ut) * (c10 - c00) + g.ut * (c11 - c01)) / (T)m.Lz : 0; }
    return (T)m.dr * th / (T)SF_TWO_PI + (1 - g.uz) * ((1 - g.ut) * c00 + g.ut * c01) + g.uz * ((1 - g.ut) * c10 + g.ut * c11);
}
template <class T> SF_HD void sf_Cb(const SFM &m, int k, const SFGap<T> &g, T v) {   /* d/dC of v R_k */
    const int ti[2] = {g.ix.t0, g.ix.t0 + 1}, zi[2] = {g.ix.z0, g.ix.z1}; const T wt[2] = {1 - g.ut, g.ut}, wz[2] = {1 - g.uz, g.uz};
    for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) {
        const T x = v * wz[a] * wt[b]; if (x == 0) continue;
        const size_t i = ti[b] < m.Nt ? ((size_t)k * m.Nz + zi[a]) * m.Nt + ti[b] : ((size_t)(k + 1) * m.Nz + zi[a]) * m.Nt;
        sf_acc(m.Cb + i, x);
    }
}
template <class T> SF_HD void sf_gap(const SFM &m, const T *q, SFGap<T> &g) {   /* q = deformed (z, Y, X) */
    g.r = sqrt(q[1] * q[1] + q[2] * q[2]); if (g.r < (T)1e-6) g.r = (T)1e-6;
    T th = (T)m.dir * atan2(q[1], q[2]); if (th < 0) th += (T)SF_TWO_PI; if (th >= (T)SF_TWO_PI) th -= (T)SF_TWO_PI; g.th = th;
    const T tb = th / (T)SF_TWO_PI * (T)m.Nt; int t0 = (int)floor(tb); if (t0 < 0) t0 = 0; if (t0 > m.Nt - 1) t0 = m.Nt - 1; g.ut = tb - (T)t0;
    const T zb = q[0] / (T)m.Lz - (T)0.5; int z0, z1; T uz;
    if (zb <= 0) { z0 = z1 = 0; uz = 0; } else if (zb >= (T)(m.Nz - 1)) { z0 = z1 = m.Nz - 1; uz = 0; } else { z0 = (int)zb; z1 = z0 + 1; uz = zb - (T)z0; }
    g.ix.z0 = z0; g.ix.z1 = z1; g.ix.t0 = t0; g.uz = uz;
    int lo = 0, hi = m.K - 2;   /* the largest k <= K - 2 with R_k <= r (0 when r < R_0) */
    while (lo < hi) { const int mid = (lo + hi + 1) / 2; if (sf_R<T>(m, mid, th, g, nullptr, nullptr) <= g.r) lo = mid; else hi = mid - 1; }
    g.ix.k = lo;
    const T Rk = sf_R<T>(m, lo, th, g, &g.dRk[0], &g.dRk[1]), Rk1 = sf_R<T>(m, lo + 1, th, g, &g.dRk1[0], &g.dRk1[1]);
    g.D = Rk1 - Rk; if (g.D < (T)1e-6) g.D = (T)1e-6;
    g.f = (g.r - Rk) / g.D; g.U = (T)lo + th / (T)SF_TWO_PI + g.f;
}

/* ---- the whole map: scroll x (box-local native) -> ideal spiral P; Up = U' (count-field convention) ---- */
template <class T> SF_HD void sf_center(const SFM &m, const T *x, T *c) { c[0] = x[0]; c[1] = x[1] - sf_tab<T>(m.ay, m.na, x[0]); c[2] = x[2] - sf_tab<T>(m.ax, m.na, x[0]); }
template <class T> SF_HD void sf_flow(const SFM &m, T *y, T *store) {
    const T h = (T)1 / (T)m.nrk;
    for (int s = 0; s < m.nst; s++) for (int n = 0; n < m.nrk; n++) {
        if (store) { T *st = store + (s * m.nrk + n) * 3; st[0] = y[0]; st[1] = y[1]; st[2] = y[2]; }
        T k1[3], k2[3], k3[3], k4[3], p[3];
        sf_vel<T>(m, s, y, k1, nullptr); for (int a = 0; a < 3; a++) p[a] = y[a] + h / 2 * k1[a];
        sf_vel<T>(m, s, p, k2, nullptr); for (int a = 0; a < 3; a++) p[a] = y[a] + h / 2 * k2[a];
        sf_vel<T>(m, s, p, k3, nullptr); for (int a = 0; a < 3; a++) p[a] = y[a] + h * k3[a];
        sf_vel<T>(m, s, p, k4, nullptr);
        for (int a = 0; a < 3; a++) y[a] += h / 6 * (k1[a] + 2 * k2[a] + 2 * k3[a] + k4[a]);
    }
}
template <class T> SF_HD void sf_map(const SFM &m, const T *x, T *P, T *Up, SFGap<T> *gout, T *store) {
    T c[3]; sf_center(m, x, c); T y[3] = {c[0], c[1], c[2]};
    sf_flow(m, y, store);
    SFGap<T> g; sf_gap(m, y, g);
    const T s = g.U * (T)m.dr / g.r; P[0] = y[0]; P[1] = s * y[1]; P[2] = s * y[2];
    if (Up) { T dl = atan2(P[1], P[2]) - atan2(c[1], c[2]); dl -= (T)SF_TWO_PI * floor((dl + (T)(SF_TWO_PI / 2)) / (T)SF_TWO_PI); *Up = g.U - (T)m.dir * dl / (T)SF_TWO_PI + (T)m.uoff; }
    if (gout) *gout = g;
}
/* the parameter gradient of Pb . P(x) (lattices into glat, cumulative gaps into Cb) */
template <class T> SF_HD T sf_map_bwd(const SFM &m, const T *x, const T *Pb) {
    T store[8 * 4 * 3]; T c[3]; sf_center(m, x, c); T y[3] = {c[0], c[1], c[2]};
    sf_flow(m, y, store);
    SFGap<T> g; sf_gap(m, y, g);
    const T s = g.U * (T)m.dr / g.r, sb = Pb[1] * y[1] + Pb[2] * y[2];
    T yb[3] = {Pb[0], s * Pb[1], s * Pb[2]};
    const T Ub = sb * (T)m.dr / g.r; T rb = -sb * g.U * (T)m.dr / (g.r * g.r);
    const T ak = -(1 - g.f) / g.D, ak1 = -g.f / g.D;
    rb += Ub / g.D;
    const T thb = Ub * ((T)1 / (T)SF_TWO_PI + ak * g.dRk[0] + ak1 * g.dRk1[0]);
    yb[0] += Ub * (ak * g.dRk[1] + ak1 * g.dRk1[1]);
    sf_Cb<T>(m, g.ix.k, g, Ub * ak); sf_Cb<T>(m, g.ix.k + 1, g, Ub * ak1);
    const T r2 = g.r * g.r;
    yb[1] += thb * (T)m.dir * y[2] / r2 + rb * y[1] / g.r;
    yb[2] += -thb * (T)m.dir * y[1] / r2 + rb * y[2] / g.r;
    const T h = (T)1 / (T)m.nrk;
    for (int s2 = m.nst - 1; s2 >= 0; s2--) for (int n = m.nrk - 1; n >= 0; n--) {
        const T *y0 = store + (s2 * m.nrk + n) * 3; T p[4][3], k[4][3], J[4][9];
        for (int a = 0; a < 3; a++) p[0][a] = y0[a];
        sf_vel<T>(m, s2, p[0], k[0], J[0]); for (int a = 0; a < 3; a++) p[1][a] = y0[a] + h / 2 * k[0][a];
        sf_vel<T>(m, s2, p[1], k[1], J[1]); for (int a = 0; a < 3; a++) p[2][a] = y0[a] + h / 2 * k[1][a];
        sf_vel<T>(m, s2, p[2], k[2], J[2]); for (int a = 0; a < 3; a++) p[3][a] = y0[a] + h * k[2][a];
        sf_vel<T>(m, s2, p[3], k[3], J[3]);
        T kb[4][3], ynew[3]; const T cf[4] = {h / 6, h / 3, h / 3, h / 6}, back[4] = {0, h / 2, h / 2, h};
        for (int i = 0; i < 4; i++) for (int a = 0; a < 3; a++) kb[i][a] = cf[i] * yb[a];
        for (int a = 0; a < 3; a++) ynew[a] = yb[a];
        for (int i = 3; i >= 0; i--) {   /* k_i = v(p_i), p_i = y0 + back_i k_{i-1} */
            sf_vel_scatter<T>(m, s2, p[i], kb[i]);
            T pb[3]; for (int a = 0; a < 3; a++) pb[a] = J[i][0 * 3 + a] * kb[i][0] + J[i][1 * 3 + a] * kb[i][1] + J[i][2 * 3 + a] * kb[i][2];
            for (int a = 0; a < 3; a++) { ynew[a] += pb[a]; if (i) kb[i - 1][a] += back[i] * pb[a]; }
        }
        for (int a = 0; a < 3; a++) yb[a] = ynew[a];
    }
    return sb * (g.U / g.r - 1 / g.D);   /* d/d pitch (P scales with it: C, the winding radii and the gaps are all proportional) */
}

/* ---- one sample's losses; with grad, their gradient (scaled by sc) ---- */
struct SFW { float wn, wd, wsd, wa, eps, huber; int fwd; float dfloor; };   /* fwd: forward differences for J (4 maps per sample, not 7) */
enum { SF_PN, SF_PD, SF_PSD, SF_PA, SF_PC, SF_PDR, SF_NP };   /* SF_PDR: the pitch gradient (summed like a part) */
template <class T> SF_HD void sf_gradW(const SFM &m, const T *P, T *w) {
    const T r2 = P[1] * P[1] + P[2] * P[2] + (T)1e-12, r = sqrt(r2);
    w[0] = 0; w[1] = P[1] / (r * (T)m.dr) - (T)m.dir * P[2] / ((T)SF_TWO_PI * r2); w[2] = P[2] / (r * (T)m.dr) + (T)m.dir * P[1] / ((T)SF_TWO_PI * r2);
}
/* frz (CPU gradcheck only): mode 1 stores P0, mode 2 takes the detached P0 (grad W and the Dirichlet frame) from it */
template <class T> SF_HD void sf_sample(const SFM &m, const SFW &W0, const T *x0, const T *gh, T lrho, T cw, T u0, T aw, int grad, T sc, T *parts, T *frz = nullptr, int fmode = 0) {
    SFW W = W0; { const float ds = W0.dfloor + (1 - W0.dfloor) * (1 - (float)aw); W.wn *= ds; W.wd *= ds; W.wsd *= ds; W.wa *= (float)aw; }   /* the anchor confidence */
    T P[7][3], Up0; const T e = (T)W.eps;
    const int np = W.fwd ? 4 : 7;
    for (int j = 0; j < np; j++) {   /* central: x0, x0 + e e_k (2k + 1), x0 - e e_k (2k + 2); forward: x0, x0 + e e_k (k + 1) */
        T x[3] = {x0[0], x0[1], x0[2]}; if (j) { if (W.fwd) x[j - 1] += e; else x[(j - 1) / 2] += (j & 1) ? e : -e; }
        sf_map<T>(m, x, P[j], j ? nullptr : &Up0, nullptr, nullptr);
    }
    T J[9]; for (int i = 0; i < 3; i++) for (int k = 0; k < 3; k++) J[i * 3 + k] = W.fwd ? (P[k + 1][i] - P[0][i]) / e : (P[2 * k + 1][i] - P[2 * k + 2][i]) / (2 * e);
    if (fmode == 1) for (int a = 0; a < 3; a++) frz[a] = P[0][a];
    const T *p0d = fmode == 2 ? frz : P[0], p0f[2] = {p0d[1], p0d[2]};
    T ws[3]; sf_gradW<T>(m, p0d, ws);
    T mm[3]; for (int k = 0; k < 3; k++) mm[k] = J[0 * 3 + k] * ws[0] + J[1 * 3 + k] * ws[1] + J[2 * 3 + k] * ws[2];
    const T ml = sqrt(mm[0] * mm[0] + mm[1] * mm[1] + mm[2] * mm[2]);
    T Jb[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, Ub = 0, mb[3] = {0, 0, 0};
    if (ml > (T)1e-9) {
        const T d = mm[0] * gh[0] + mm[1] * gh[1] + mm[2] * gh[2], sg = d >= 0 ? (T)1 : (T)-1;
        if (W.wn > 0 && cw > 0) {
            parts[SF_PN] += W.wn * cw * (1 - sg * d / ml);
            for (int k = 0; k < 3; k++) mb[k] += -W.wn * cw * sg * (gh[k] / ml - d * mm[k] / (ml * ml * ml));
        }
        if (W.wd > 0 && cw > 0) {
            const T ed = (T)0.1 * exp(lrho); T f, fp;
            if (d >= ed) { const T q = log(d) - lrho; f = q * q; fp = 2 * q / d; }
            else { const T q = log(ed) - lrho; fp = 2 * q / ed; f = q * q + fp * (d - ed); }
            parts[SF_PD] += W.wd * cw * f; for (int k = 0; k < 3; k++) mb[k] += W.wd * cw * fp * gh[k];
        }
        for (int i = 0; i < 3; i++) for (int k = 0; k < 3; k++) Jb[i * 3 + k] += ws[i] * mb[k];
        {   /* the pitch enters grad_P W = P^ / pitch - (theta part) explicitly: d m / d pitch = -J^T (P^ / pitch) / pitch */
            const T pr = sqrt(p0f[0] * p0f[0] + p0f[1] * p0f[1]) + (T)1e-12; T s = 0;
            for (int k = 0; k < 3; k++) s += mb[k] * (J[1 * 3 + k] * p0f[0] + J[2 * 3 + k] * p0f[1]);
            if (grad) parts[SF_PDR] += -sc * s / (pr * (T)m.dr * (T)m.dr);
        }
        if (W.wsd > 0) {   /* in-sheet symmetric Dirichlet (villa get_symmetric_dirichlet_loss): the spiral frame (z, winding tangent) at P0
                              mapped to the scroll by J^-1; the energy is symmetric in J and J^-1 */
            const T det3 = J[0] * (J[4] * J[8] - J[5] * J[7]) - J[1] * (J[3] * J[8] - J[5] * J[6]) + J[2] * (J[3] * J[7] - J[4] * J[6]);
            if (det3 > (T)1e-12) {
                T Ji[9]; Ji[0] = (J[4] * J[8] - J[5] * J[7]) / det3; Ji[1] = (J[2] * J[7] - J[1] * J[8]) / det3; Ji[2] = (J[1] * J[5] - J[2] * J[4]) / det3;
                Ji[3] = (J[5] * J[6] - J[3] * J[8]) / det3; Ji[4] = (J[0] * J[8] - J[2] * J[6]) / det3; Ji[5] = (J[2] * J[3] - J[0] * J[5]) / det3;
                Ji[6] = (J[3] * J[7] - J[4] * J[6]) / det3; Ji[7] = (J[1] * J[6] - J[0] * J[7]) / det3; Ji[8] = (J[0] * J[4] - J[1] * J[3]) / det3;
                const T pr = sqrt(p0f[0] * p0f[0] + p0f[1] * p0f[1]) + (T)1e-12, t1[3] = {1, 0, 0}, t2[3] = {0, p0f[1] / pr, -p0f[0] / pr};
                T a[3], b[3]; for (int i = 0; i < 3; i++) { a[i] = Ji[i * 3]; b[i] = Ji[i * 3 + 1] * t2[1] + Ji[i * 3 + 2] * t2[2]; }
                const T g11 = a[0] * a[0] + a[1] * a[1] + a[2] * a[2], g22 = b[0] * b[0] + b[1] * b[1] + b[2] * b[2], g12 = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
                const T ie = (T)1e-3, tr = g11 + g22, det = g11 * g22 - g12 * g12, Dn = det + ie * tr + ie * ie, inv = (tr + 2 * ie) / Dn;
                const T E = tr + inv - 4;
                if (E > 0 && E < 100) {
                    parts[SF_PSD] += W.wsd * E;
                    const T dtr = 1 / Dn - (tr + 2 * ie) * ie / (Dn * Dn), ddet = -(tr + 2 * ie) / (Dn * Dn);
                    const T e11 = W.wsd * (1 + dtr + ddet * g22), e22 = W.wsd * (1 + dtr + ddet * g11), e12 = W.wsd * ddet * (-2 * g12);
                    T Ab[9];   /* dE / dJ^-1 */
                    for (int i = 0; i < 3; i++) { const T ab = 2 * e11 * a[i] + e12 * b[i], bb = 2 * e22 * b[i] + e12 * a[i]; for (int k = 0; k < 3; k++) Ab[i * 3 + k] = ab * t1[k] + bb * t2[k]; }
                    T Q[9];    /* J-bar = -J^-T Ab J^-T */
                    for (int i = 0; i < 3; i++) for (int k = 0; k < 3; k++) { T s = 0; for (int q = 0; q < 3; q++) s += Ji[q * 3 + i] * Ab[q * 3 + k]; Q[i * 3 + k] = s; }
                    for (int i = 0; i < 3; i++) for (int k = 0; k < 3; k++) { T s = 0; for (int q = 0; q < 3; q++) s += Q[i * 3 + q] * Ji[k * 3 + q]; Jb[i * 3 + k] -= s; }
                } else if (E >= 100) parts[SF_PSD] += W.wsd * 100;
            }
        }
    }
    if (W.wa > 0 && u0 > (T)-1e30) {
        const T r = Up0 - u0, hd = (T)W.huber, ar = fabs(r);
        parts[SF_PA] += W.wa * (ar < hd ? r * r / 2 : hd * (ar - hd / 2)); Ub = W.wa * (r > hd ? hd : r < -hd ? -hd : r);
    }
    parts[SF_PC] += 1;
    if (!grad) return;
    if (Ub != 0) parts[SF_PDR] += -sc * Ub * sqrt(P[0][1] * P[0][1] + P[0][2] * P[0][2]) / ((T)m.dr * (T)m.dr);   /* U = |P| / pitch explicitly */
    T p0b[3] = {0, sc * Ub * ws[1], sc * Ub * ws[2]};   /* x0's adjoint: the anchor (+ the forward differences' -J-bar / e) */
    for (int k = 0; k < 3; k++) {
        T pb[3]; for (int i = 0; i < 3; i++) pb[i] = sc * Jb[i * 3 + k] / (W.fwd ? e : 2 * e);
        if (pb[0] != 0 || pb[1] != 0 || pb[2] != 0) {
            T x[3] = {x0[0], x0[1], x0[2]}; x[k] += e; parts[SF_PDR] += sf_map_bwd<T>(m, x, pb);
            if (W.fwd) for (int i = 0; i < 3; i++) p0b[i] -= pb[i];
            else { for (int i = 0; i < 3; i++) pb[i] = -pb[i]; x[k] -= 2 * e; parts[SF_PDR] += sf_map_bwd<T>(m, x, pb); }
        }
    }
    if (p0b[0] != 0 || p0b[1] != 0 || p0b[2] != 0) parts[SF_PDR] += sf_map_bwd<T>(m, x0, p0b);
}

/* ---- the gaps: C = cumsum (pitch e^g) over k per (z bin, angle bin); gradient of g from Cb (suffix sums) ---- */
template <class T> SF_HD void sf_cumsum1(const float *g, float *C, int K, size_t nb, size_t b, float dr) {
    T acc = 0; C[b] = 0; for (int k = 0; k < K; k++) { acc += (T)dr * exp((T)g[k * nb + b]); C[(k + 1) * nb + b] = (float)acc; }
}

/* =================================== GPU =================================== */
static double sf_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static unsigned sf_hash(unsigned x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }
__device__ __forceinline__ unsigned sf_hashd(unsigned x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }
struct SFData { size_t nd; const int32_t *cz, *cy, *cx; const float *gz, *gy, *gx, *lr, *c, *u0, *aw; float cs; };
__global__ void sf_loss_k(SFM m, SFW W, SFData D, int ns, unsigned salt, float sc, double *parts, int grad = 1) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    float lp[SF_NP] = {0, 0, 0, 0, 0, 0};
    if (i < ns) {
        const unsigned h1 = sf_hashd(salt ^ (unsigned)i * 0x9E3779B9u), h2 = sf_hashd(h1 + 0x68E31DA4u), h3 = sf_hashd(h2 + 0xB5297A4Du), h4 = sf_hashd(h3 + 0x1B56C4E9u);
        const size_t j = ((size_t)h1 << 16 ^ h2) % D.nd;
        const float x0[3] = {(D.cz[j] + 0.5f + ((h2 & 0xffff) / 65536.f - 0.5f)) * D.cs, (D.cy[j] + 0.5f + ((h3 & 0xffff) / 65536.f - 0.5f)) * D.cs, (D.cx[j] + 0.5f + ((h4 & 0xffff) / 65536.f - 0.5f)) * D.cs};
        const float gh[3] = {D.gz[j], D.gy[j], D.gx[j]};
        sf_sample<float>(m, W, x0, gh, D.lr[j], D.c[j], D.u0[j], D.aw ? D.aw[j] : 1.f, grad, sc, lp);
    }
    for (int k = 0; k < SF_NP; k++) {
        float v = lp[k]; for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if ((threadIdx.x & 31) == 0 && v != 0) atomicAdd(&parts[k], (double)v);
    }
}
__global__ void sf_cumsum_k(const float *g, float *C, int K, size_t nb, float dr) { const size_t b = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (b < nb) sf_cumsum1<float>(g, C, K, nb, b, dr); }
/* the gap gradient (suffix sums of Cb) + Adam + the bounds, one (z bin, angle bin) column per thread; Cb zeroed */
__global__ void sf_gap_step_k(float *g, float *mo, float *ve, float *Cb, int K, size_t nb, int Nt, float dr, float lr, float b1c, float b2c, float gmin, float gmax, float wdec) {
    const size_t b = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (b >= nb) return;
    float S = 0;
    for (int k = K; k >= 1; k--) {
        S += Cb[k * nb + b]; Cb[k * nb + b] = 0;
        const size_t i = (size_t)(k - 1) * nb + b;
        if (k == 1 && b % Nt == 0) { g[i] = 0; continue; }   /* pinned: winding 0 at angle bin 0 (the seam) */
        const float gr = dr * __expf(g[i]) * S + wdec * g[i];
        const float m1 = mo[i] = 0.9f * mo[i] + 0.1f * gr, v1 = ve[i] = 0.999f * ve[i] + 0.001f * gr * gr;
        float x = g[i] - lr * (m1 / b1c) / (sqrtf(v1 / b2c) + 1e-12f);
        g[i] = fminf(fmaxf(x, gmin), gmax);
    }
    Cb[b] = 0;
}
__global__ void sf_adam_k(float *p, float *g, float *mo, float *ve, size_t n, float lr, float b1c, float b2c) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= n) return;
    const float gr = g[i]; g[i] = 0;
    const float m1 = mo[i] = 0.9f * mo[i] + 0.1f * gr, v1 = ve[i] = 0.999f * ve[i] + 0.001f * gr * gr;
    p[i] -= lr * (m1 / b1c) / (sqrtf(v1 / b2c) + 1e-12f);
}
__global__ void sf_out_k(SFM m, int z, int hh, int w, float cs, float *out) {   /* U' at the cell centres of solve row z */
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= hh * w) return;
    const float x[3] = {(z + 0.5f) * cs, (i / w + 0.5f) * cs, (i % w + 0.5f) * cs}; float P[3], Up;
    sf_map<float>(m, x, P, &Up, nullptr, nullptr); out[i] = Up;
}
/* diagnostics over a sample of the data cells: |U' - u0| > 0.5 share, mean normal residual, median-ish density error */
__global__ void sf_diag_k(SFM m, SFData D, int ns, unsigned salt, double *acc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= ns) return;
    const unsigned h1 = sf_hashd(salt ^ (unsigned)i * 0x9E3779B9u), h2 = sf_hashd(h1 + 7);
    const size_t j = ((size_t)h1 << 16 ^ h2) % D.nd;
    const float x[3] = {(D.cz[j] + 0.5f) * D.cs, (D.cy[j] + 0.5f) * D.cs, (D.cx[j] + 0.5f) * D.cs}; float P[3], Up;
    sf_map<float>(m, x, P, &Up, nullptr, nullptr);
    if (D.u0[j] > -1e30f) { const float d = fabsf(Up - D.u0[j]); atomicAdd(&acc[0], 1.0); atomicAdd(&acc[1], (double)d); if (d > 0.5f) atomicAdd(&acc[2], 1.0); }
}

__global__ void sf_pairs_k(SFM m, SFData D, int ns, unsigned salt, float *out) {   /* (U', u0) at sampled data cell centres */
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= ns) return;
    const unsigned h1 = sf_hashd(salt ^ (unsigned)i * 0x9E3779B9u), h2 = sf_hashd(h1 + 7);
    const size_t j = ((size_t)h1 << 16 ^ h2) % D.nd;
    const float x[3] = {(D.cz[j] + 0.5f) * D.cs, (D.cy[j] + 0.5f) * D.cs, (D.cx[j] + 0.5f) * D.cs}; float P[3], Up;
    sf_map<float>(m, x, P, &Up, nullptr, nullptr); out[2 * i] = Up; out[2 * i + 1] = D.u0[j];
}
__global__ void sf_fill_k(float *g, size_t n, size_t nb, int Nt, float v) { const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) g[i] = (i < nb && i % Nt == 0) ? 0.f : v; }
__global__ void sf_pin_k(SFM m, SFW W, const float *pz, const float *py, const float *px, const float *pu, size_t npin, int ns, unsigned salt, float sc, double *parts) {   /* pins: the anchor term at exact points */
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    float lp[SF_NP] = {0, 0, 0, 0, 0, 0};
    if (i < ns) {
        const unsigned h1 = sf_hashd(salt ^ (unsigned)i * 0x9E3779B9u), h2 = sf_hashd(h1 + 0x68E31DA4u); const size_t j = ((size_t)h1 << 16 ^ h2) % npin;
        const float x0[3] = {pz[j], py[j], px[j]}, gh[3] = {0, 1, 0};
        sf_sample<float>(m, W, x0, gh, 0.f, 0.f, pu[j], 1.f, 1, sc, lp);
    }
    float v = lp[SF_PA]; for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0 && v != 0) atomicAdd(&parts[6], (double)v);
    v = lp[SF_PDR]; for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0 && v != 0) atomicAdd(&parts[SF_PDR], (double)v);
}
#define SF_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "spiral-fit: CUDA %s (%s:%d)\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)
template <class X> static X *sf_up(const X *h, size_t n) { X *d = nullptr; if (cudaMalloc(&d, n * sizeof(X) + 16) != cudaSuccess) return nullptr; cudaMemcpy(d, h, n * sizeof(X), cudaMemcpyHostToDevice); return d; }
static float *sf_zeros(size_t n) { float *d = nullptr; if (cudaMalloc(&d, n * sizeof(float) + 16) != cudaSuccess) return nullptr; cudaMemset(d, 0, n * sizeof(float)); return d; }

/* the model geometry from the config (host); returns the parameter counts */
static void sf_geom(const sf_cfg *c, SFM &m, size_t nl[2], size_t &ngap) {
    const double Dn = c->d * c->cs, Hn = c->hh * c->cs, Wn = c->w * c->cs;
    double ylo = 1e30, yhi = -1e30, xlo = 1e30, xhi = -1e30, rmax = 0;
    for (int z = 0; z < c->na; z++) {
        ylo = fmin(ylo, -c->ay[z]); yhi = fmax(yhi, Hn - c->ay[z]); xlo = fmin(xlo, -c->ax[z]); xhi = fmax(xhi, Wn - c->ax[z]);
        const double cy[2] = {-c->ay[z], Hn - c->ay[z]}, cx[2] = {-c->ax[z], Wn - c->ax[z]};
        for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) rmax = fmax(rmax, hypot(cy[a], cx[b]));
    }
    memset(&m, 0, sizeof m); m.nst = c->nst; m.nrk = c->nrk; m.L[1] = (float)c->Lf; m.L[0] = (float)(c->Lf * c->sfac);
    const double mg = m.L[0], lo[3] = {-mg, ylo - mg, xlo - mg}, ext[3] = {Dn + 2 * mg, yhi - ylo + 2 * mg, xhi - xlo + 2 * mg};
    for (int a = 0; a < 3; a++) m.o[a] = (float)lo[a];
    for (int l = 0; l < 2; l++) { for (int a = 0; a < 3; a++) m.n[l][a] = (int)ceil(ext[a] / m.L[l]) + 3; nl[l] = (size_t)c->nst * m.n[l][0] * m.n[l][1] * m.n[l][2] * 3; }
    m.dr = (float)c->pitch; m.dir = c->dir >= 0 ? 1 : -1; m.Lz = (float)c->Lz;
    m.Nz = (int)ceil(Dn / c->Lz); if (m.Nz < 1) m.Nz = 1;
    m.Nt = (int)lround(SF_TWO_PI * rmax / c->La); if (m.Nt < 16) m.Nt = 16; if (m.Nt > 2048) m.Nt = 2048;
    m.K = c->K > 0 ? c->K : (int)ceil(rmax / c->pitch) + 4;
    ngap = (size_t)m.K * m.Nz * m.Nt;
    m.ay = c->ay; m.ax = c->ax; m.na = c->na;
}

extern "C" int nn_spiral_fit(sf_cfg *c) {
    SF_CK(cudaSetDevice(c->gpu));
    SFM m; size_t nl[2], ngap; sf_geom(c, m, nl, ngap); const size_t nb = (size_t)m.Nz * m.Nt;
    fprintf(stderr, "spiral-fit: %zu data cells, lattices %dx%dx%d (%g) + %dx%dx%d (%g) x %d stages, gaps %d windings x %d z x %d angle bins, pitch %.2f, %.2f GB params\n",
            c->nd, m.n[0][0], m.n[0][1], m.n[0][2], m.L[0], m.n[1][0], m.n[1][1], m.n[1][2], m.L[1], m.nst, m.K, m.Nz, m.Nt, m.dr, (4.0 * (nl[0] + nl[1]) + 5.0 * ngap) * 4 / 1e9);
    if (m.nst * m.nrk > 32) { fprintf(stderr, "spiral-fit: stages x RK4 steps <= 32\n"); return 1; }
    float *lat[2], *glat[2], *lm[2], *lv[2];
    for (int l = 0; l < 2; l++) { lat[l] = sf_zeros(nl[l]); glat[l] = sf_zeros(nl[l]); lm[l] = sf_zeros(nl[l]); lv[l] = sf_zeros(nl[l]); if (!lat[l] || !glat[l] || !lm[l] || !lv[l]) { fprintf(stderr, "spiral-fit: out of device memory (lattices)\n"); return 1; } }
    float *g = sf_zeros(ngap), *gm = sf_zeros(ngap), *gv = sf_zeros(ngap), *C = sf_zeros(ngap + nb), *Cb = sf_zeros(ngap + nb);
    float *ay = sf_up(c->ay, c->na), *ax = sf_up(c->ax, c->na);
    if (!g || !gm || !gv || !C || !Cb || !ay || !ax) { fprintf(stderr, "spiral-fit: out of device memory (gaps)\n"); return 1; }
    SFData D; D.nd = c->nd; D.cs = (float)c->cs;
    D.cz = sf_up(c->cz, c->nd); D.cy = sf_up(c->cy_, c->nd); D.cx = sf_up(c->cx_, c->nd); D.gz = sf_up(c->gz, c->nd); D.gy = sf_up(c->gy, c->nd); D.gx = sf_up(c->gx, c->nd);
    D.lr = sf_up(c->lrho, c->nd); D.c = sf_up(c->c, c->nd); D.u0 = sf_up(c->u0, c->nd); D.aw = c->aw ? sf_up(c->aw, c->nd) : nullptr;
    if (!D.cz || !D.cy || !D.cx || !D.gz || !D.gy || !D.gx || !D.lr || !D.c || !D.u0) { fprintf(stderr, "spiral-fit: out of device memory (data)\n"); return 1; }
    m.lat[0] = lat[0]; m.lat[1] = lat[1]; m.glat[0] = glat[0]; m.glat[1] = glat[1]; m.C = C; m.Cb = Cb; m.ay = ay; m.ax = ax;
    double *dparts; SF_CK(cudaMalloc(&dparts, 8 * sizeof(double)));
    float *dpz = c->npin ? sf_up(c->pz, c->npin) : nullptr, *dpy = c->npin ? sf_up(c->py, c->npin) : nullptr, *dpx = c->npin ? sf_up(c->px, c->npin) : nullptr, *dpu = c->npin ? sf_up(c->pu, c->npin) : nullptr;
    const int TPB = 128, ns = c->nsamp; const float sc = 1.0f / ns;
    double acc[8];
    {   /* the initial pitch scale and label offset from the anchor: u0 ~ a U' + b over sampled cells -> gaps e^g = 1 / a, then the
           median of u0 - U' (rounded) as the label offset (the spiral numbers its windings from the axis, the reference does not) */
        const int no = 1 << 20; float *dp; SF_CK(cudaMalloc(&dp, 2 * no * sizeof(float))); float *hp = (float *)malloc(2 * no * sizeof(float));
        for (int pass = getenv("UFSM_SF_SLOPE") ? 0 : 1; pass < 2; pass++) {   /* (the slope pass: opt-in, a uniform rescale hurts crushed sections) */
            sf_cumsum_k<<<(unsigned)((nb + 255) / 256), 256>>>(g, C, m.K, nb, m.dr);
            sf_pairs_k<<<(no + 127) / 128, 128>>>(m, D, no, 777u, dp); SF_CK(cudaMemcpy(hp, dp, 2 * no * sizeof(float), cudaMemcpyDeviceToHost));
            double sx = 0, sy = 0, sxx = 0, sxy = 0; int n = 0; float *df = (float *)malloc(no * sizeof(float));
            for (int i = 0; i < no; i++) { const double x = hp[2 * i], y = hp[2 * i + 1]; if (!(y > -1e30)) continue; sx += x; sy += y; sxx += x * x; sxy += x * y; df[n++] = (float)(y - x); }
            if (n < 100) { free(df); break; }
            if (pass == 0) {
                const double a = (n * sxy - sx * sy) / fmax(1e-9, n * sxx - sx * sx), ga = fmin(fmax(-log(fmax(a, 1e-3)), c->gmin), c->gmax);
                fprintf(stderr, "spiral-fit: initial U' vs the anchor: slope %.4f -> log gap %.4f\n", a, ga);
                sf_fill_k<<<(unsigned)((ngap + 255) / 256), 256>>>(g, ngap, nb, m.Nt, (float)ga);
            } else {
                qsort(df, n, sizeof(float), [](const void *p, const void *q) { const float u = *(const float *)p, v = *(const float *)q; return (u > v) - (u < v); });
                m.uoff = (float)lround(df[n / 2]); double mad = 0; for (int i = 0; i < n; i++) mad += fabs(df[i] - m.uoff); 
                fprintf(stderr, "spiral-fit: label offset %+.0f (median u0 - U' %.3f, mean |residual| %.3f turns)\n", m.uoff, df[n / 2], mad / n);
            }
            free(df);
        }
        cudaFree(dp); free(hp);
    }
    SF_CK(cudaDeviceSynchronize()); double tlast = sf_now(), t0fit = tlast; int itlast = 0; double drm = 0, drv = 0; int ndr = 0;
    for (int it = 0; it <= c->steps; it++) {
        const double fr = (double)it / (c->steps > 0 ? c->steps : 1), lrf = pow(c->lr_end, fr);
        SFW W; W.eps = (float)c->eps; W.huber = (float)c->huber; W.fwd = c->fwd; W.dfloor = (float)c->dfloor;
        const int warm = it < c->warm; W.wn = warm ? 0.f : (float)c->w_n; W.wd = warm ? 0.f : (float)c->w_d; W.wsd = warm ? 0.f : (float)c->w_sd;
        W.wa = warm ? (float)c->w_a0 : (float)(c->w_a0 * pow(c->w_a1 / c->w_a0, (double)(it - c->warm) / fmax(1, c->steps - c->warm)));
        sf_cumsum_k<<<(unsigned)((nb + 255) / 256), 256>>>(g, C, m.K, nb, m.dr);
        if (it == c->steps) break;
        SF_CK(cudaMemset(dparts, 0, 8 * sizeof(double)));
        sf_loss_k<<<(ns + TPB - 1) / TPB, TPB>>>(m, W, D, ns, sf_hash(c->seed * 7919u + (unsigned)it), sc, dparts);
        if (c->npin) { SFW Wp; memset(&Wp, 0, sizeof Wp); Wp.wa = (float)c->w_pin; Wp.eps = W.eps; Wp.huber = W.huber; Wp.fwd = W.fwd; Wp.dfloor = 1; const int pn = 16384;
            sf_pin_k<<<(pn + TPB - 1) / TPB, TPB>>>(m, Wp, dpz, dpy, dpx, dpu, c->npin, pn, sf_hash(c->seed * 31u + (unsigned)it), 1.0f / pn, dparts); }
        const float b1c = (float)(1 - pow(0.9, it + 1)), b2c = (float)(1 - pow(0.999, it + 1));
        if (c->lr_dr > 0 && !warm) {   /* the pitch (villa learns it too): Adam on log pitch, host side (one sync per step) */
            double pd[8]; SF_CK(cudaMemcpy(pd, dparts, 8 * sizeof(double), cudaMemcpyDeviceToHost)); const double gl = pd[SF_PDR] * m.dr;
            drm = 0.9 * drm + 0.1 * gl; drv = 0.999 * drv + 0.001 * gl * gl; ndr++;
            m.dr = (float)(m.dr * exp(-c->lr_dr * lrf * (drm / (1 - pow(0.9, ndr))) / (sqrt(drv / (1 - pow(0.999, ndr))) + 1e-12)));
        }
        sf_gap_step_k<<<(unsigned)((nb + 255) / 256), 256>>>(g, gm, gv, Cb, m.K, nb, m.Nt, m.dr, (float)(c->lr_g * lrf), b1c, b2c, (float)c->gmin, (float)c->gmax, 1e-4f);
        for (int l = 0; l < 2; l++) sf_adam_k<<<(unsigned)((nl[l] + 255) / 256), 256>>>(lat[l], glat[l], lm[l], lv[l], nl[l], (float)((l ? c->lr_f : c->lr_c) * lrf), b1c, b2c);
        if (c->verbose && (it % c->verbose == 0 || it == c->steps - 1)) {
            SF_CK(cudaMemcpy(acc, dparts, 8 * sizeof(double), cudaMemcpyDeviceToHost)); SF_CK(cudaGetLastError());
            const double n = acc[SF_PC] > 0 ? acc[SF_PC] : 1;
            const double tn = sf_now(), ms = (tn - tlast) * 1e3 / fmax(1, it - itlast); tlast = tn; itlast = it;
            fprintf(stderr, "spiral-fit: step %5d  pitch %.3f  normal %.4f  density %.4f  dirichlet %.4f  anchor %.4f (w %.3g)  %.1f ms/step, %.2f M samples/s\n", it, m.dr, acc[SF_PN] / n, acc[SF_PD] / n, acc[SF_PSD] / n, acc[SF_PA] / n, W.wa, ms, ns / ms / 1e3); if (c->npin) fprintf(stderr, "spiral-fit: step %5d  pins %.4f\n", it, acc[6] / 16384);
        }
    }
    SF_CK(cudaDeviceSynchronize());
    fprintf(stderr, "spiral-fit: %d steps in %.1f s (%.1f ms/step)\n", c->steps, sf_now() - t0fit, (sf_now() - t0fit) * 1e3 / fmax(1, c->steps));
    {   /* the data terms at unit weights (no gradient) over a fixed sample: comparable across runs and with the anchor-only fit */
        SFW E; E.wn = 1; E.wd = 1; E.wsd = 1; E.wa = 1; E.eps = (float)c->eps; E.huber = (float)c->huber; E.fwd = c->fwd; E.dfloor = 1; const int ne = 1 << 20;
        SF_CK(cudaMemset(dparts, 0, 8 * sizeof(double))); sf_loss_k<<<(ne + 127) / 128, 128>>>(m, E, D, ne, 4242u, 0.f, dparts, 0);
        SF_CK(cudaMemcpy(acc, dparts, 8 * sizeof(double), cudaMemcpyDeviceToHost)); const double n = acc[SF_PC] > 0 ? acc[SF_PC] : 1;
        fprintf(stderr, "spiral-fit: final terms at unit weights: normal %.4f density %.4f dirichlet %.4f anchor %.4f\n", acc[SF_PN] / n, acc[SF_PD] / n, acc[SF_PSD] / n, acc[SF_PA] / n);
    }
    {   double *dd; SF_CK(cudaMalloc(&dd, 4 * sizeof(double))); SF_CK(cudaMemset(dd, 0, 4 * sizeof(double))); const int nd2 = 1 << 20;
        sf_diag_k<<<(nd2 + 127) / 128, 128>>>(m, D, nd2, 12345u, dd); double h[4]; SF_CK(cudaMemcpy(h, dd, 4 * sizeof(double), cudaMemcpyDeviceToHost)); cudaFree(dd);
        if (h[0] > 0) fprintf(stderr, "spiral-fit: vs the initial U: mean |dU| %.3f turns, > 0.5 turn on %.3f of the data cells\n", h[1] / h[0], h[2] / h[0]); }
    float *orow; const size_t HW = (size_t)c->hh * c->w; SF_CK(cudaMalloc(&orow, HW * sizeof(float)));
    for (int z = 0; z < c->d; z++) { sf_out_k<<<(unsigned)((HW + 127) / 128), 128>>>(m, z, c->hh, c->w, (float)c->cs, orow); SF_CK(cudaMemcpy(c->uout + z * HW, orow, HW * sizeof(float), cudaMemcpyDeviceToHost)); }
    cudaFree(orow); cudaFree(dparts); cudaFree(dpz); cudaFree(dpy); cudaFree(dpx); cudaFree(dpu);
    for (int l = 0; l < 2; l++) { cudaFree(lat[l]); cudaFree(glat[l]); cudaFree(lm[l]); cudaFree(lv[l]); }
    cudaFree(g); cudaFree(gm); cudaFree(gv); cudaFree(C); cudaFree(Cb); cudaFree(ay); cudaFree(ax);
    cudaFree((void *)D.cz); cudaFree((void *)D.cy); cudaFree((void *)D.cx); cudaFree((void *)D.gz); cudaFree((void *)D.gy); cudaFree((void *)D.gx); cudaFree((void *)D.lr); cudaFree((void *)D.c); cudaFree((void *)D.u0); cudaFree((void *)D.aw);
    return 0;
}

/* =================================== CPU reference: gradient check =================================== */
static double *frz = nullptr, g_drb = 0; static int fmode = 0;
static double sf_cpu_loss(const SFM &m, const SFW &W, const double *X, int ns, int grad) {
    fmode = grad ? 1 : 2;
    double parts[SF_NP] = {0, 0, 0, 0, 0, 0};
    for (int i = 0; i < ns; i++) { const double *s = &X[i * 8]; const double x0[3] = {s[0], s[1], s[2]}, gh[3] = {s[3], s[4], s[5]}; sf_sample<double>(m, W, x0, gh, s[6], s[7], i % 3 ? s[1] / 6.5 : -1e38, i % 2 ? 0.7 : 1.0, grad, 1.0, parts, &frz[i * 3], fmode); }
    if (grad) g_drb = parts[SF_PDR];
    return parts[SF_PN] + parts[SF_PD] + parts[SF_PSD] + parts[SF_PA];
}
static void sf_topk(const float *gf, const double *gd, size_t n, int k, size_t *out) {   /* indices of the k largest |gradient| */
    for (int j = 0; j < k; j++) { size_t best = 0; double bv = -1;
        for (size_t i = 0; i < n; i++) { const double v = fabs(gf ? (double)gf[i] : gd[i]); int used = 0; for (int q = 0; q < j; q++) used |= out[q] == i; if (!used && v > bv) { bv = v; best = i; } }
        out[j] = best; }
}
extern "C" int nn_spiral_gradcheck(int verbose) {
    /* a small box: 64 x 96 x 96 native, axis near the centre (tilted), random small lattices and gaps */
    const int na = 65; float ay[65], ax[65]; for (int z = 0; z < na; z++) { ay[z] = 48.f + 0.05f * z; ax[z] = 47.f - 0.03f * z; }
    sf_cfg c; memset(&c, 0, sizeof c); c.d = 4; c.hh = 6; c.w = 6; c.cs = 16; c.ay = ay; c.ax = ax; c.na = na; c.dir = 1; c.pitch = 6.5;
    c.nst = 2; c.nrk = 3; c.Lf = 16; c.sfac = 3; c.La = 12; c.Lz = 20;
    SFM m; size_t nl[2], ngap; sf_geom(&c, m, nl, ngap); const size_t nb = (size_t)m.Nz * m.Nt;
    unsigned s = 1; auto rnd = [&s]() { s = sf_hash(s + 0x9E3779B9u); return (s & 0xffffff) / 16777216.0 - 0.5; };
    float *lat[2], *glat[2]; for (int l = 0; l < 2; l++) { lat[l] = (float *)malloc(nl[l] * 4); glat[l] = (float *)calloc(nl[l], 4); for (size_t i = 0; i < nl[l]; i++) lat[l][i] = (float)((l ? 1.5 : 4.0) * rnd()); }
    float *g = (float *)malloc(ngap * 4), *C = (float *)calloc(ngap + nb, 4), *Cb = (float *)calloc(ngap + nb, 4); for (size_t i = 0; i < ngap; i++) g[i] = (float)(0.4 * rnd());
    for (size_t b = 0; b < nb; b++) if (b % m.Nt == 0) g[b] = 0;
    auto cum = [&]() { for (size_t b = 0; b < nb; b++) sf_cumsum1<double>(g, C, m.K, nb, b, m.dr); };
    cum();
    m.lat[0] = lat[0]; m.lat[1] = lat[1]; m.glat[0] = glat[0]; m.glat[1] = glat[1]; m.C = C; m.Cb = Cb;
    SFW W = {1.f, 0.5f, 0.3f, 0.7f, 3.f, 0.5f, getenv("SF_GC_FWD") ? 1 : 0, 0.2f};
    if (getenv("SF_GC_W")) sscanf(getenv("SF_GC_W"), "%f,%f,%f,%f", &W.wn, &W.wd, &W.wsd, &W.wa);
    const int ns = 24; double X[ns * 8]; frz = (double *)calloc(ns * 3, sizeof(double));
    for (int i = 0; i < ns; i++) {
        double *q = &X[i * 8]; q[0] = 32 + 40 * rnd(); const double ang = SF_TWO_PI * rnd(), rr = 12 + 60 * (rnd() + 0.5); q[1] = 48 + rr * sin(ang); q[2] = 47 + rr * cos(ang);
        double gv[3] = {0.2 * rnd(), sin(ang) + 0.3 * rnd(), cos(ang) + 0.3 * rnd()}; const double gl = sqrt(gv[0] * gv[0] + gv[1] * gv[1] + gv[2] * gv[2]);
        q[3] = gv[0] / gl; q[4] = gv[1] / gl; q[5] = gv[2] / gl; q[6] = log(1 / 6.5) + 0.3 * rnd(); q[7] = 0.5 + rnd();
    }
    sf_cpu_loss(m, W, X, ns, 1);
    /* g's gradient from Cb (suffix sums, as sf_gap_step_k) */
    double *gg = (double *)calloc(ngap, sizeof(double));
    for (size_t b = 0; b < nb; b++) { double S = 0; for (int k = m.K; k >= 1; k--) { S += Cb[k * nb + b]; gg[(k - 1) * nb + b] = m.dr * exp((double)g[(k - 1) * nb + b]) * S; } }
    int fails = 0, checked = 0; const float hstep = 1.f / 512;
    auto check = [&](float *p, double an, const char *what, size_t idx, int isgap) {
        const float p0 = *p; *p = p0 + hstep; if (isgap) cum(); const double lp = sf_cpu_loss(m, W, X, ns, 0);
        *p = p0 - hstep; if (isgap) cum(); const double lm = sf_cpu_loss(m, W, X, ns, 0); *p = p0; if (isgap) cum();
        const double fd = (lp - lm) / (2.0 * hstep), err = fabs(fd - an), tol = 2e-3 + 0.03 * fmax(fabs(fd), fabs(an));
        checked++; if (err > tol) { fails++; fprintf(stderr, "spiral gradcheck FAIL %s[%zu]: analytic %.6g fd %.6g\n", what, idx, an, fd); }
        else if (verbose) fprintf(stderr, "spiral gradcheck %s[%zu]: analytic %.6g fd %.6g\n", what, idx, an, fd);
    };
    size_t idx[16];
    for (int l = 0; l < 2; l++) { sf_topk(glat[l], nullptr, nl[l], 12, idx); for (int k = 0; k < 12; k++) check(&lat[l][idx[k]], glat[l][idx[k]], l ? "fine" : "coarse", idx[k], 0); }
    sf_topk(nullptr, gg, ngap, 16, idx); for (int k = 0; k < 16; k++) check(&g[idx[k]], gg[idx[k]], "gap", idx[k], 1);
    { const double an = g_drb; check(&m.dr, an, "pitch", 0, 1); }
    fprintf(stderr, "spiral gradcheck: %d of %d parameters off\n", fails, checked);
    for (int l = 0; l < 2; l++) { free(lat[l]); free(glat[l]); }
    free(g); free(C); free(Cb); free(gg); free(frz); frz = nullptr;
    return fails;
}
extern "C" int nn_spiral_has_gpu(void) { int n = 0; return cudaGetDeviceCount(&n) == cudaSuccess && n > 0; }
