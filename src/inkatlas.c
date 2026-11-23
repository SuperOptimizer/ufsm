/* ufsm ink-atlas: the pure helpers (inkatlas.h) */
#include "inkatlas.h"
#include <math.h>

void ia_xform(const uint8_t *in, int W, int H, int t, uint8_t *out, int *oW, int *oH) {
    const int tr = t & 4, W2 = tr ? H : W, H2 = tr ? W : H;
    for (int r = 0; r < H2; r++) for (int c = 0; c < W2; c++) {
        const int r1 = t & 2 ? H2 - 1 - r : r, c1 = t & 1 ? W2 - 1 - c : c;
        out[(size_t)r * W2 + c] = tr ? in[(size_t)c1 * W + r1] : in[(size_t)r1 * W + c1];
    }
    *oW = W2; *oH = H2;
}
double ia_pearson(const uint8_t *a, const uint8_t *b, int W, int H, int dy, int dx, int step, size_t *n) {
    double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0; size_t m = 0;
    for (int r = 0; r < H; r += step) { const int rb = r + dy; if (rb < 0 || rb >= H) continue;
        for (int c = 0; c < W; c += step) { const int cb = c + dx; if (cb < 0 || cb >= W) continue;
            const double x = a[(size_t)r * W + c], y = b[(size_t)rb * W + cb]; if (!(x > 0) || !(y > 0)) continue;
            sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y; m++; } }
    if (n) *n = m;
    if (m < 1000) return NAN;
    const double va = saa - sa * sa / m, vb = sbb - sb * sb / m;
    return va > 0 && vb > 0 ? (sab - sa * sb / m) / sqrt(va * vb) : NAN;
}
void ia_shift(const uint8_t *in, int W, int H, int dy, int dx, uint8_t *out) {
    for (int r = 0; r < H; r++) for (int c = 0; c < W; c++) { const int r1 = r + dy, c1 = c + dx;
        out[(size_t)r * W + c] = r1 >= 0 && r1 < H && c1 >= 0 && c1 < W ? in[(size_t)r1 * W + c1] : 0; }
}
int ia_face_order(int fwd, int rev, int min_win, double ratio) {
    if (fwd > rev && fwd >= min_win && fwd >= ratio * rev) return 1;
    if (rev > fwd && rev >= min_win && rev >= ratio * fwd) return -1;
    return 0;
}
void ia_gate(const uint8_t *ens, const uint8_t *const *mod, const double *mean, const double *sd, int nm, const uint8_t *cov, size_t n,
             const ia_gate_cfg *g, uint8_t *lab, uint8_t *msk, size_t *nink, size_t *nbg) {
    size_t ni = 0, nb = 0;
    for (size_t k = 0; k < n; k++) {
        lab[k] = 0; msk[k] = 0; if (!cov[k]) continue;
        const double z = -2 + 6.0 * ens[k] / 255;
        if (z <= g->z_bg) { msk[k] = 255; nb++; continue; }
        if (z < g->z_ink) continue;
        int a = 0; for (int m = 0; m < nm; m++) a += sd[m] > 0 && (mod[m][k] / 255.0 - mean[m]) / sd[m] >= g->model_z;
        if (a >= g->agree) { lab[k] = 255; msk[k] = 255; ni++; }
    }
    *nink = ni; *nbg = nb;
}
