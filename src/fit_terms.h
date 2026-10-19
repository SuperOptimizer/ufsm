/* The winding fitter's per-voxel data terms (src/fit.h), shared by the CUDA kernels (src/nn_fit.cu) and the CPU backend
   (src/fit.c, fit_cfg cpu): one definition of each loss, so both backends fit the same objective.
   phase: c (1 - cos (2 pi W - phi)), c = w_phase x weight x min(conf, 1)^conf_pow x packed factor; with phase_robust = s > 0 turns
          the Cauchy form S log(1 + (1 - cos) / S), S = 2 pi^2 s^2 (the plain term for residuals << s; a voxel whose phase disagrees
          with the fit by more than ~s pulls with a bounded, decaying force, so a wrong-phase patch cannot drag the level sets onto
          the neighbouring sheet)
   normal: 1 - g^T T g / (|g|^2 tr T); density: (log |g| - log rho)^2; non-degeneracy: max(0, 0.4 rho - |g|)^2 (g = grad W) */
#pragma once
#include "fit.h"
#ifdef __CUDACC__
#define FIT_HD __host__ __device__ __forceinline__
#else
#include <math.h>
#define FIT_HD static inline
#endif
#define FIT_IMIN(a, b) ((a) < (b) ? (a) : (b))

FIT_HD float fit_packed_w(const fit_dev *f, size_t i) {   /* the phase weight factor of packed cells */
    if (f->packed_sp <= 0.f || !f->sp[i]) return 1.f;
    return fminf(1.f, fmaxf(f->packed_floor, ((float)f->sp[i] - 0.6f * f->packed_sp) / (0.4f * f->packed_sp)));
}
/* the phase term of voxel i at winding W: its loss added to *l, dL/dW to *dW */
FIT_HD void fit_phase_term(const fit_dev *f, size_t i, float W, float weight, double *l, float *dW) {
    const float tp = 6.283185307179586f;
    const float mc = ((int)f->pc[i] - 128) * (1.f / 127.f), ms = ((int)f->ps[i] - 128) * (1.f / 127.f), conf = sqrtf(mc * mc + ms * ms);
    if (!(conf > 0.05f && f->w_phase > 0.f)) return;
    float sa, ca; sincosf(tp * W - atan2f(ms, mc), &sa, &ca);
    const float cf = fminf(conf, 1.f), cw = f->conf_pow == 1.f || f->conf_pow <= 0.f ? cf : powf(cf, f->conf_pow);
    const float c = f->w_phase * weight * cw * fit_packed_w(f, i), res = 1.f - ca;
    if (f->phase_robust > 0.f) {
        const float S = 0.5f * tp * tp * f->phase_robust * f->phase_robust;
        *l += c * S * log1pf(res / S); *dW += c * tp * sa / (1.f + res / S);
    } else { *l += c * res; *dW += c * tp * sa; }
}
/* the gradient terms of voxel i at g = grad W: loss parts added to lp[2..4], dL/dg to dg */
FIT_HD void fit_grad_terms(const fit_dev *f, size_t i, const float g[3], float weight, double lp[5], float dg[3]) {
    const float g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2] + 1e-12f, gl = sqrtf(g2);
    const float tzz = f->t[0][i] * (1.f / 255.f), tyy = f->t[1][i] * (1.f / 255.f), txx = f->t[2][i] * (1.f / 255.f);
    const float tzy = f->t[3][i] * (1.f / 255.f) - 0.5f, tzx = f->t[4][i] * (1.f / 255.f) - 0.5f, tyx = f->t[5][i] * (1.f / 255.f) - 0.5f;
    const float tr = tzz + tyy + txx;
    if (tr > 0.1f && f->w_normal > 0.f) {
        const float Tg[3] = {tzz * g[0] + tzy * g[1] + tzx * g[2], tzy * g[0] + tyy * g[1] + tyx * g[2], tzx * g[0] + tyx * g[1] + txx * g[2]};
        const float gTg = g[0] * Tg[0] + g[1] * Tg[1] + g[2] * Tg[2], c = f->w_normal * weight;
        lp[2] += c * (1.f - gTg / (g2 * tr));
        for (int d = 0; d < 3; d++) dg[d] += -c * (2.f * Tg[d] / (g2 * tr) - 2.f * gTg * g[d] / (g2 * g2 * tr));
    }
    if (f->sp[i]) {
        const float rho = 8.f / (f->h * (float)f->sp[i]), lr = logf(gl) - logf(rho);
        if (f->w_density > 0.f) { const float c = f->w_density * weight; lp[3] += c * lr * lr; for (int d = 0; d < 3; d++) dg[d] += c * 2.f * lr * g[d] / g2; }
        const float lo = 0.4f * rho - gl;
        if (lo > 0.f && f->w_nondeg > 0.f) { const float c = f->w_nondeg * weight; lp[4] += c * lo * lo; for (int d = 0; d < 3; d++) dg[d] += -c * 2.f * lo * g[d] / gl; }
    }
}
FIT_HD unsigned fit_hash(unsigned x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }
/* nn_fit_loss2's voxel i (weight wv, data terms x sc): W and grad W analytic at the voxel centre (trilinear delta, the knots' and
   theta's derivatives, the axis slope), the data terms into lp[1..4] (lp[0] their sum) and dL/d delta at the 8 lattice corners
   (indices ic, values gv); returns 0 inside the core (no terms) */
FIT_HD int fit_vox2(const fit_dev *f, size_t i, float wv, float sc, double lp[5], size_t ic[8], float gv[8]) {
    const int z = (int)(i / ((size_t)f->H * f->W)), y = (int)((i / f->W) % f->H), x = (int)(i % f->W);
    const float pz = (z + 0.5f) * f->h, py = (y + 0.5f) * f->h, px = (x + 0.5f) * f->h;
    /* axis and its z slope */
    const float az = fminf(fmaxf(pz, 0.f), (float)(f->naxis - 1)); const int ai = FIT_IMIN((int)az, f->naxis - 2 < 0 ? 0 : f->naxis - 2); const float at = f->naxis > 1 ? az - ai : 0.f;
    const int ai1 = f->naxis > 1 ? ai + 1 : ai;
    const float cy = f->cy[ai] + at * (f->cy[ai1] - f->cy[ai]), cx = f->cx[ai] + at * (f->cx[ai1] - f->cx[ai]), cyz = f->cy[ai1] - f->cy[ai], cxz = f->cx[ai1] - f->cx[ai];
    const float dy = py - cy, dx = px - cx, r2 = dy * dy + dx * dx + 1e-6f, r = sqrtf(r2), th = atan2f(dy, dx);
    /* knots (fixed): value and z slope */
    const float kz = fminf(fmaxf(pz / (float)f->knot_dz, 0.f), (float)(f->nk - 1)); const int k0 = FIT_IMIN((int)kz, f->nk - 2 < 0 ? 0 : f->nk - 2), k1 = f->nk > 1 ? k0 + 1 : k0; const float kt = f->nk > 1 ? kz - k0 : 0.f;
    const float pit = f->pitch[k0] + kt * (f->pitch[k1] - f->pitch[k0]), off = f->offset[k0] + kt * (f->offset[k1] - f->offset[k0]);
    const float pitz = (f->pitch[k1] - f->pitch[k0]) / (float)f->knot_dz, offz = (f->offset[k1] - f->offset[k0]) / (float)f->knot_dz;
    if (r < f->core * pit) return 0;
    /* trilinear delta, value and gradient */
    const float q[3] = {pz / (float)f->L, py / (float)f->L, px / (float)f->L}; const int dim[3] = {f->lz, f->ly, f->lx}; int c0[3]; float ct[3];
    for (int d = 0; d < 3; d++) { const float a = fminf(fmaxf(q[d], 0.f), (float)(dim[d] - 1)); c0[d] = FIT_IMIN((int)a, dim[d] - 2 < 0 ? 0 : dim[d] - 2); ct[d] = dim[d] > 1 ? a - c0[d] : 0.f; }
    float del = 0.f, gdel[3] = {0, 0, 0}, wc[8], gw[8][3];
    const float iL = 1.f / (float)f->L;
    for (int c = 0; c < 8; c++) {
        const int bz = c >> 2 & 1, by = c >> 1 & 1, bx = c & 1;
        const float wz = bz ? ct[0] : 1.f - ct[0], wy = by ? ct[1] : 1.f - ct[1], wx = bx ? ct[2] : 1.f - ct[2];
        wc[c] = wz * wy * wx;
        gw[c][0] = (bz ? iL : -iL) * wy * wx; gw[c][1] = wz * (by ? iL : -iL) * wx; gw[c][2] = wz * wy * (bx ? iL : -iL);
        ic[c] = ((size_t)FIT_IMIN(c0[0] + bz, f->lz - 1) * f->ly + FIT_IMIN(c0[1] + by, f->ly - 1)) * f->lx + FIT_IMIN(c0[2] + bx, f->lx - 1);
        const float v = f->delta[ic[c]];
        del += wc[c] * v; for (int d = 0; d < 3; d++) gdel[d] += gw[c][d] * v;
    }
    const float i2p = 0.15915494309189535f, sdir = (float)f->dir;
    const float drz = -(dy * cyz + dx * cxz) / r;
    float g[3] = {drz / pit - r * pitz / (pit * pit) + offz + gdel[0], dy / (r * pit) + gdel[1], dx / (r * pit) + gdel[2]};
    const float thz = (-dx * cyz + dy * cxz) / r2, thy = dx / r2, thx = -dy / r2;
    g[0] -= sdir * thz * i2p; g[1] -= sdir * thy * i2p; g[2] -= sdir * thx * i2p;
    const float W0 = r / pit + off + del - sdir * th * i2p;
    float dW = 0.f, dg[3] = {0, 0, 0};
    fit_phase_term(f, i, W0, wv * sc, &lp[1], &dW);
    fit_grad_terms(f, i, g, wv * sc, lp, dg);
    lp[0] = lp[1] + lp[2] + lp[3] + lp[4];
    for (int c = 0; c < 8; c++) gv[c] = dW * wc[c] + dg[0] * gw[c][0] + dg[1] * gw[c][1] + dg[2] * gw[c][2];
    return 1;
}
