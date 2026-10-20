/* predict --tta K: test-time augmentation of the fit fields. Each window also runs through K - 1 cube symmetries that keep z the
   scroll axis (training draws all 48 cube symmetries, sample.c write_x_aug; the inference input builder only knows a radial
   vector without a z component): flips of z, y, x and the y <-> x swap. Only involutions are used (T = T^-1), so the CT goes in
   through T and the logits come back through T in place: the 8 flips, and the swap with y and x both flipped or neither (x 2 for z),
   12 in all. The radial input follows from the mirrored axis rows (tta_axis), the normal heads n n^T from the signed permutation
   (tta_channels; the off-diagonal heads 0.5 + n_a n_b negate their logits when the sign of n_a n_b flips); phase, recto, ink,
   density and the winding mod 4 heads are scalars (invariant). The untransformed logits of every pass are pooled into the same
   field accumulators, so the fit fields average the passes (the phase as a complex mean: disagreement lowers its confidence). */
#pragma once
#include <stdint.h>

enum { TTA_FZ = 1, TTA_FY = 2, TTA_FX = 4, TTA_SWAP = 8, TTA_MAX = 12 };
/* the pass order: identity, all flipped, the transpose, the anti-transpose, then the rest */
static const int tta_order[TTA_MAX] = {0, 7, 8, 14, 1, 6, 9, 15, 2, 4, 3, 5};

/* T of a voxel of a D x W x W window: flips first, then the swap (an involution for the codes above) */
static inline void tta_map(int code, int D, int W, int z, int y, int x, int *oz, int *oy, int *ox) {
    if (code & TTA_FZ) z = D - 1 - z;
    if (code & TTA_FY) y = W - 1 - y;
    if (code & TTA_FX) x = W - 1 - x;
    *oz = z; if (code & TTA_SWAP) { *oy = x; *ox = y; } else { *oy = y; *ox = x; }
}
/* the axis rows of the transformed window: dyo[z] = window origin y - axis centre y at plane z (predict's rows: the radial input
   at voxel (y, x) of plane z is (y + dyo[z], x + dxo[z])), likewise dxo; the transformed window's rows into dyt, dxt */
static inline void tta_axis(int code, int D, int W, const float *dyo, const float *dxo, float *dyt, float *dxt) {
    for (int zt = 0; zt < D; zt++) {
        const int z = code & TTA_FZ ? D - 1 - zt : zt;
        float ay = -dyo[z], ax = -dxo[z];   /* the axis centre in window voxels */
        if (code & TTA_FY) ay = (float)(W - 1) - ay;
        if (code & TTA_FX) ax = (float)(W - 1) - ax;
        if (code & TTA_SWAP) { const float t = ay; ay = ax; ax = t; }
        dyt[zt] = -ay; dxt[zt] = -ax;
    }
}
/* the logit channels back from the transformed frame: original channel c = sgn[c] x transformed channel src[c] at T(v). Normal heads at
   ch_normal (< 0: none) in the order zz yy xx zy zx yx (unet DESIGN: n_z^2, n_y^2, n_x^2, 0.5 + n_z n_y, 0.5 + n_z n_x, 0.5 + n_y n_x) */
static inline void tta_channels(int code, int ch_normal, int nch, int *src, int *sgn) {
    for (int c = 0; c < nch; c++) { src[c] = c; sgn[c] = 1; }
    if (ch_normal < 0 || ch_normal + 6 > nch) return;
    const int pi[3] = {0, code & TTA_SWAP ? 2 : 1, code & TTA_SWAP ? 1 : 2};   /* n_a = s_a n'_{pi a} */
    const int s[3] = {code & TTA_FZ ? -1 : 1, code & TTA_FY ? -1 : 1, code & TTA_FX ? -1 : 1};
    static const int A[6] = {0, 1, 2, 0, 0, 1}, B[6] = {0, 1, 2, 1, 2, 2};
    for (int c = 0; c < 6; c++) {
        int a = pi[A[c]], b = pi[B[c]]; if (a > b) { const int t = a; a = b; b = t; }
        int k = 0; while (!(A[k] == a && B[k] == b)) k++;
        src[ch_normal + c] = ch_normal + k; sgn[ch_normal + c] = c < 3 ? 1 : s[A[c]] * s[B[c]];
    }
}

/* the device side (src/nn_tta.cu): dst[v] = src[T(v)] of a uint8 D x W x W window; the [nch][D W W] logits (fp16 when h16) back
   through T in place with the channel map */
#ifdef __cplusplus
extern "C" {
#endif
void nn_tta_ct(const uint8_t *src, uint8_t *dst, int D, int W, int code);
void nn_tta_logits(void *lg, int h16, int D, int W, int code, int nch, const int *src, const int *sgn);
#ifdef __cplusplus
}
#endif
