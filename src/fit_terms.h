/* The winding fitter's per-voxel data terms (src/fit.h), shared by the CUDA kernels (src/nn_fit.cu) and the CPU backend
   (src/fit.c, fit_cfg cpu): one definition of each loss, so both backends fit the same objective.
   phase: c (1 - cos (2 pi W - phi)), c = w_phase x weight x min(conf, 1)^conf_pow x packed factor; with phase_robust = s > 0 turns
          the Cauchy form S log(1 + (1 - cos) / S), S = 2 pi^2 s^2 (the plain term for residuals << s; a voxel whose phase disagrees
          with the fit by more than ~s pulls with a bounded, decaying force, so a wrong-phase patch cannot drag the level sets onto
          the neighbouring sheet)
   normal: 1 - g^T T g / (|g|^2 tr T); density: (log |g| - log rho)^2; non-degeneracy: max(0, 0.4 rho - |g|)^2 (g = grad W)
   recto anchors (fields with an r plane, w_recto > 0): w_recto x weight x r^2 (1 - cos 2 pi W): a confident recto probability puts an
          integer level set there, independently of the phase head
   CT parity (fields with a ct plane, w_ct > 0): c m (1 - cos 2 pi (W - phi_pap)), m in [-1, 1] the cell's papyrus / void parity
          (fit_ct_parity: local CT contrast), c = w_ct x weight x max(ct_floor, 1 - min(conf, 1))^ct_pow: papyrus cells pull frac W to the
          papyrus phase, void cells to its opposite, most where the network phase is weak (it holds the sheets on the CT where the phase
          head cannot)
   CT spacing (fields with a cs plane, w_cs > 0): w_cs x weight x a x (log |g| - log rho_cs)^2 with rho_cs from the CT's own sheet
          spacing, a = 1 where sp is unknown, else ramping from 0 at |log(sp / cs)| = cs_tol to 1 at 2 cs_tol (the network density is
          kept where the two agree)
   winding mod 4 (task unwrap2 fields, nn_fit_loss2 only): per axis a, w_w4 x weight x dq (1 - cos(pi / 2 (W(x + d e_a) - W(x) - delta_a))),
          delta_a the network's W difference over d native voxels mod 4 (dz dy dx planes): counts turns where the phase cannot
   Loss parts: the FIT_P_* slots of src/fit.h (lp[FIT_P_TOTAL] the sum of the data terms; the smoothness slot unused per voxel). */
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
/* the recto anchors of voxel i at winding W */
FIT_HD void fit_recto_term(const fit_dev *f, size_t i, float W, float weight, double *l, float *dW) {
    if (!f->recto || !(f->w_recto > 0.f) || !f->recto[i]) return;
    const float tp = 6.283185307179586f, r = f->recto[i] * (1.f / 255.f), c = f->w_recto * weight * r * r;
    float sa, ca; sincosf(tp * W, &sa, &ca);
    *l += c * (1.f - ca); *dW += c * tp * sa;
}
/* the CT papyrus / void parity of voxel i at winding W */
FIT_HD void fit_ct_term(const fit_dev *f, size_t i, float W, float weight, double *l, float *dW) {
    if (!f->ctm || !(f->w_ct > 0.f) || f->ctm[i] == 128) return;
    const float tp = 6.283185307179586f, m = ((int)f->ctm[i] - 128) * (1.f / 127.f);
    const float mc = ((int)f->pc[i] - 128) * (1.f / 127.f), ms = ((int)f->ps[i] - 128) * (1.f / 127.f), conf = sqrtf(mc * mc + ms * ms);
    float u = 1.f - fminf(conf, 1.f); u = f->ct_pow == 1.f ? u : f->ct_pow <= 0.f ? 1.f : powf(u, f->ct_pow); u = fmaxf(u, f->ct_floor);
    const float c = f->w_ct * weight * u * m;
    float sa, ca; sincosf(tp * (W - f->ct_phi), &sa, &ca);
    *l += c * (1.f - ca); *dW += c * tp * sa;
}
/* the density term against the CT sheet spacing of voxel i at g = grad W (into lp[FIT_P_CSPACE], dL/dg to dg) */
FIT_HD void fit_cspace_term(const fit_dev *f, size_t i, const float g[3], float weight, double lp[FIT_NPARTS], float dg[3]) {
    if (!f->cs || !(f->w_cs > 0.f) || !f->cs[i]) return;
    float a = 1.f;
    if (f->sp[i]) {
        const float d = fabsf(logf((float)f->sp[i] / (float)f->cs[i])), t = f->cs_tol > 0.f ? f->cs_tol : 0.25f;
        a = fminf(1.f, fmaxf(0.f, (d - t) / t)); if (!(a > 0.f)) return;
    }
    const float g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2] + 1e-12f, gl = sqrtf(g2);
    const float rho = 8.f / (f->h * (float)f->cs[i]), lr = logf(gl) - logf(rho), c = f->w_cs * weight * a;
    lp[FIT_P_CSPACE] += c * lr * lr; for (int d = 0; d < 3; d++) dg[d] += c * 2.f * lr * g[d] / g2;
}
/* the gradient terms of voxel i at g = grad W: loss parts added to lp[2..4], dL/dg to dg */
FIT_HD void fit_grad_terms(const fit_dev *f, size_t i, const float g[3], float weight, double lp[FIT_NPARTS], float dg[3]) {
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
        if (f->w_density > 0.f) { const float c = f->w_density * weight * (f->skip ? 1.f + f->w_dskip * f->skip[i] * (1.f / 255.f) : 1.f); lp[3] += c * lr * lr; for (int d = 0; d < 3; d++) dg[d] += c * 2.f * lr * g[d] / g2; }
        const float lo = 0.4f * rho - gl;
        if (lo > 0.f && f->w_nondeg > 0.f) { const float c = f->w_nondeg * weight; lp[4] += c * lo * lo; for (int d = 0; d < 3; d++) dg[d] += -c * 2.f * lo * g[d] / gl; }
    }
}
/* phase-gradient matching (w_pgrad > 0): per axis d the wrapped phase step to the next cell G_d = wrap(phi(i + e_d) - phi(i)) / (2 pi h)
   (turns per native voxel; pooled stages: f->pg, confidence-weighted means over the pool cell) against g_d = dW / dx_d, as the Huber
   function of the relative error r = (g_d - G_d) / rho (rho = 1 / spacing, turns per native voxel) at scale pgrad_delta: quadratic
   below it, L1 above. No integers (it fits in the gradient-only stages too), signed (it knows which way the windings grow, which n n^T
   and the density do not), robust (a dislocation of the phase costs its length, not its square: the turn count follows the
   confident phase and cuts through the low-confidence patches, as L1 phase unwrapping does). The loss into lp[5]. */
FIT_HD void fit_pgrad_term(const fit_dev *f, size_t i, const float g[3], float weight, double lp[FIT_NPARTS], float dg[3]) {
    if (!(f->w_pgrad > 0.f) || !f->sp[i]) return;
    const float rho = 8.f / (f->h * (float)f->sp[i]), dl = f->pgrad_delta > 0.f ? f->pgrad_delta : 0.25f;
    float G[3] = {0, 0, 0}, c[3] = {0, 0, 0};
    const size_t HW = (size_t)f->H * f->W, n = HW * f->D;
    if (f->pg) { for (int d = 0; d < 3; d++) { G[d] = f->pg[d * n + i]; c[d] = f->pg[3 * n + i]; } }
    else {
        const float mc = ((int)f->pc[i] - 128) * (1.f / 127.f), ms = ((int)f->ps[i] - 128) * (1.f / 127.f), c0 = sqrtf(mc * mc + ms * ms);
        if (c0 < 0.05f) return;
        const float a0 = atan2f(ms, mc);
        const int pos[3] = {(int)(i / HW), (int)((i / f->W) % f->H), (int)(i % f->W)}, dim[3] = {f->D, f->H, f->W};
        const size_t st[3] = {HW, (size_t)f->W, 1};
        for (int d = 0; d < 3; d++) {
            if (pos[d] + 1 >= dim[d]) continue;
            const size_t j = i + st[d]; if (!f->w[j]) continue;
            const float nc = ((int)f->pc[j] - 128) * (1.f / 127.f), ns = ((int)f->ps[j] - 128) * (1.f / 127.f), c1 = sqrtf(nc * nc + ns * ns);
            if (c1 < 0.05f) continue;
            float e = (atan2f(ns, nc) - a0) * 0.15915494309189535f; e -= floorf(e + 0.5f);
            G[d] = e / f->h; c[d] = fminf(fminf(c0, c1), 1.f);
        }
    }
    for (int d = 0; d < 3; d++) {
        if (!(c[d] > 0.f)) continue;
        const float r = (g[d] - G[d]) / rho, ar = fabsf(r), cc = f->w_pgrad * weight * c[d];
        if (ar <= dl) { lp[5] += cc * r * r / (2.f * dl); dg[d] += cc * (r / dl) / rho; }
        else { lp[5] += cc * (ar - 0.5f * dl); dg[d] += cc * (r > 0.f ? 1.f : -1.f) / rho; }
    }
}
/* the sheet spacing of a CT profile p[0..n) (n <= 1024, smoothed in place by [1 2 1] / 4), samples step native voxels apart along a
   normal: papyrus peaks are the runs above the profile's mean that start and end with a dip below it inside the profile and reach
   mean + prom sd, each at its maximum; spacing = (last - first) / (peaks - 1) x step, 0 below 2 peaks or for a flat profile (sd < 2) */
FIT_HD float fit_ct_spacing(float *p, int n, float step, float prom) {
    if (n < 5) return 0.f;
    float prev = p[0], mu = 0.f;
    for (int k = 0; k < n; k++) { const float c = p[k], v = 0.25f * prev + 0.5f * c + 0.25f * p[k + 1 < n ? k + 1 : k]; prev = c; p[k] = v; mu += v; }
    mu /= n; float s2 = 0.f; for (int k = 0; k < n; k++) s2 += (p[k] - mu) * (p[k] - mu);
    const float sd = sqrtf(s2 / n); if (sd < 2.f) return 0.f;
    int armed = 0, inrun = 0, first = -1, last = -1, np = 0, am = 0; float mx = 0.f;
    for (int k = 0; k < n; k++) {
        if (p[k] < mu) { if (inrun && armed && mx >= mu + prom * sd) { if (first < 0) first = am; last = am; np++; } inrun = 0; armed = 1; continue; }
        if (!inrun) { inrun = 1; mx = p[k]; am = k; } else if (p[k] > mx) { mx = p[k]; am = k; }
    }
    return np >= 2 ? (float)(last - first) / (float)(np - 1) * step : 0.f;
}
FIT_HD float fit_wrap_pi(float a) {   /* into (-pi, pi] */
    const float tp = 6.283185307179586f;
    return a - tp * floorf((a + 3.14159265358979f) / tp);
}
/* U (fixed knots, trilinear delta) and theta at a box-local native point, with the 8 lattice corners and weights (the winding mod 4
   term's second point) */
FIT_HD float fit_U_at(const fit_dev *f, float pz, float py, float px, float *th, size_t ic[8], float wc[8]) {
    const float az = fminf(fmaxf(pz, 0.f), (float)(f->naxis - 1)); const int ai = FIT_IMIN((int)az, f->naxis - 2 < 0 ? 0 : f->naxis - 2); const float at = f->naxis > 1 ? az - ai : 0.f;
    const int ai1 = f->naxis > 1 ? ai + 1 : ai;
    const float cy = f->cy[ai] + at * (f->cy[ai1] - f->cy[ai]), cx = f->cx[ai] + at * (f->cx[ai1] - f->cx[ai]);
    const float dy = py - cy, dx = px - cx, r = sqrtf(dy * dy + dx * dx + 1e-6f); *th = atan2f(dy, dx);
    const float kz = fminf(fmaxf(pz / (float)f->knot_dz, 0.f), (float)(f->nk - 1)); const int k0 = FIT_IMIN((int)kz, f->nk - 2 < 0 ? 0 : f->nk - 2), k1 = f->nk > 1 ? k0 + 1 : k0; const float kt = f->nk > 1 ? kz - k0 : 0.f;
    const float pit = f->pitch[k0] + kt * (f->pitch[k1] - f->pitch[k0]), off = f->offset[k0] + kt * (f->offset[k1] - f->offset[k0]);
    const float q[3] = {pz / (float)f->L, py / (float)f->L, px / (float)f->L}; const int dim[3] = {f->lz, f->ly, f->lx}; int c0[3]; float ct[3];
    for (int d = 0; d < 3; d++) { const float a = fminf(fmaxf(q[d], 0.f), (float)(dim[d] - 1)); c0[d] = FIT_IMIN((int)a, dim[d] - 2 < 0 ? 0 : dim[d] - 2); ct[d] = dim[d] > 1 ? a - c0[d] : 0.f; }
    float del = 0.f;
    for (int c = 0; c < 8; c++) {
        const int bz = c >> 2 & 1, by = c >> 1 & 1, bx = c & 1;
        wc[c] = (bz ? ct[0] : 1.f - ct[0]) * (by ? ct[1] : 1.f - ct[1]) * (bx ? ct[2] : 1.f - ct[2]);
        ic[c] = ((size_t)FIT_IMIN(c0[0] + bz, f->lz - 1) * f->ly + FIT_IMIN(c0[1] + by, f->ly - 1)) * f->lx + FIT_IMIN(c0[2] + bx, f->lx - 1);
        del += wc[c] * f->delta[ic[c]];
    }
    return r / pit + off + del;
}
FIT_HD unsigned fit_hash(unsigned x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }
/* nn_fit_loss2's voxel i (weight wv, data terms x sc): W and grad W analytic at the voxel centre (trilinear delta, the knots' and
   theta's derivatives, the axis slope), the data terms into lp (FIT_P_*; lp[FIT_P_TOTAL] their sum) and dL/d delta at lattice
   corners (indices ic, values gv: the voxel's 8, then with the winding mod 4 term 8 more per axis for its second points); returns
   the number of corners (0 inside the core: no terms) */
#define FIT_VOX2_MAXC 32
FIT_HD int fit_vox2(const fit_dev *f, size_t i, float wv, float sc, double lp[FIT_NPARTS], size_t ic[FIT_VOX2_MAXC], float gv[FIT_VOX2_MAXC]) {
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
    fit_phase_term(f, i, W0, wv * sc, &lp[FIT_P_PHASE], &dW);
    fit_recto_term(f, i, W0, wv * sc, &lp[FIT_P_RECTO], &dW);
    fit_ct_term(f, i, W0, wv * sc, &lp[FIT_P_CT], &dW);
    fit_grad_terms(f, i, g, wv * sc, lp, dg);
    fit_pgrad_term(f, i, g, wv * sc, lp, dg);
    fit_cspace_term(f, i, g, wv * sc, lp, dg);
    if (f->w_rc > 0.f && f->sp[i]) {   /* radial count: W's radial derivative = the density's radial count rho |n . r| (the density term matches |grad W| = rho
                                        whatever its direction, so a W smoother / more radial than the oblique sheets over-counts along the radius) */
        const float ry = dy / r, rx = dx / r, tyy = f->t[1][i] * (1.f / 255.f), txx = f->t[2][i] * (1.f / 255.f), tyx = f->t[5][i] * (1.f / 255.f) - 0.5f;
        const float tr = f->t[0][i] * (1.f / 255.f) + tyy + txx;
        if (tr > 0.1f) {
            const float nr = sqrtf(fminf(1.f, fmaxf(0.f, (ry * ry * tyy + rx * rx * txx + 2.f * ry * rx * tyx) / tr))), rho = 8.f / (f->h * (float)f->sp[i]) * nr;
            if (nr > 0.3f) {
                const float gr = g[1] * ry + g[2] * rx, e = (gr - rho) / rho, c = f->w_rc * wv * sc;
                lp[FIT_P_DENSITY] += c * e * e; dg[1] += c * 2.f * e / rho * ry; dg[2] += c * 2.f * e / rho * rx;
            }
        }
    }
    int nc = 8;
    const size_t n = (size_t)f->D * f->H * f->W;
    if (f->wd && f->w_w4 > 0.f && f->wd[3 * n + i]) {   /* winding mod 4: per axis 1 - cos(pi / 2 (W(x + d e_a) - W(x) - delta_a)) */
        const float qc = f->wd[3 * n + i] * (1.f / 255.f), cc = f->w_w4 * wv * qc * sc, U0 = r / pit + off + del, kq = 1.5707963267948966f;
        for (int a = 0; a < 3; a++) {
            float th1, wc1[8];
            const float U1 = fit_U_at(f, pz + (a == 0 ? f->w4_d : 0.f), py + (a == 1 ? f->w4_d : 0.f), px + (a == 2 ? f->w4_d : 0.f), &th1, ic + nc, wc1);
            const float dWm = (U1 - U0) - sdir * fit_wrap_pi(th1 - th) * i2p, dl = (float)f->wd[a * n + i] * (1.f / 64.f);
            float sa, ca; sincosf(kq * (dWm - dl), &sa, &ca);
            lp[FIT_P_WIND4] += cc * (1.f - ca);
            const float gW = cc * kq * sa;   /* dL / dW(x + d e_a); dL / dW(x) = -gW */
            dW -= gW;
            for (int c = 0; c < 8; c++) gv[nc + c] = gW * wc1[c];
            nc += 8;
        }
    }
    lp[FIT_P_TOTAL] = lp[FIT_P_PHASE] + lp[FIT_P_NORMAL] + lp[FIT_P_DENSITY] + lp[FIT_P_NONDEG] + lp[FIT_P_PGRAD] + lp[FIT_P_WIND4] + lp[FIT_P_RECTO]
                    + lp[FIT_P_CT] + lp[FIT_P_CSPACE];
    for (int c = 0; c < 8; c++) gv[c] = dW * wc[c] + dg[0] * gw[c][0] + dg[1] * gw[c][1] + dg[2] * gw[c][2];
    return nc;
}
