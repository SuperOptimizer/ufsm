/* merge-flag's per-cell arithmetic shared by the CPU (src/mergeflag.c) and the GPU (src/nn_mf.cu), 2026-11-18. Every box filter is a
   sum of integers (exact in any order, so CPU threads, GPU tiles and slab borders all give the same sums); only mf_final's few double
   operations per cell (two divisions, three atan2, a sqrt, an exp) can differ between glibc and the device in the last bit, which moves
   a byte only at an exact .5 of 255 p: the GPU's mf is within 1 LSB of the CPU's (test_mergeflag counts the differing cells).
   The 12 sums of a cell (C box of edge 2 r + 1 over edge-replicated neighbours, i.e. clamped coordinates):
     0 A9   sum amp_q (amp_q = round(65536 |(pc, ps)| / 127 ... as MF_AMPQ units) on material (w > 0) cells, r 4
     1 M9   material cells, r 4                   2 S9  sp, r 4        3 K9  sp > 0, r 4
     4 S5   sp, r 2                               5 K5  sp > 0, r 2
     6..11  CR_a, CI_a for a = z, y, x, r 2: the neighbour product z(g' + e_a) conj z(g') with z = (pc - 128, ps - 128) (the 127^2
            scale cancels in atan2), g' = min(g, N_a - 2) (the last cell repeats the one before)
   amp9 = A9 / (MF_AMPQ M9), sp9 = S9 / (8 K9), sp5 = S5 / (8 K5) (0 where the count is 0), R5 = |(t_z, t_y, t_x)| sp5 with t_a =
   atan2(CI_a, CR_a) / 2 pi, mf = round(255 / (1 + exp(-(b + c_amp amp9 + c_r R5 + c_sp sp9)))) on material cells, 0 elsewhere. */
#pragma once
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __CUDACC__
#define MF_HD __host__ __device__
#else
#define MF_HD
#endif

enum { MF_NS = 12, MF_AMPQ = 65536 };
static inline MF_HD int mf_rad(int c) { return c < 4 ? 4 : 2; }   /* the box radius of sum c */

/* amp_q of a (pc, ps) byte pair: the host table (65536 entries) and the device copy hold this function's values */
static inline int mf_ampq(int pc, int ps) { const double a = (pc - 128) / 127.0, b = (ps - 128) / 127.0; return (int)lrint(MF_AMPQ * sqrt(a * a + b * b)); }

typedef struct { double amp9, r5, sp9; } mf_feat;
static inline MF_HD mf_feat mf_features(const int32_t s[MF_NS]) {
    mf_feat f;
    f.amp9 = s[1] > 0 ? (double)s[0] / ((double)MF_AMPQ * s[1]) : 0;
    f.sp9 = s[3] > 0 ? s[2] / (8.0 * s[3]) : 0;
    const double sp5 = s[5] > 0 ? s[4] / (8.0 * s[5]) : 0;
    double ka = 0;
    for (int a = 0; a < 3; a++) { const double t = atan2((double)s[7 + 2 * a], (double)s[6 + 2 * a]) / (2 * M_PI); ka += t * t; }
    f.r5 = sqrt(ka) * sp5;
    return f;
}
static inline MF_HD uint8_t mf_final(const int32_t s[MF_NS], int material, double b, double c_amp, double c_r, double c_sp) {
    if (!material) return 0;
    const mf_feat f = mf_features(s); const double lg = b + c_amp * f.amp9 + c_r * f.r5 + c_sp * f.sp9;
    return (uint8_t)lrint(255.0 / (1.0 + exp(-lg)));
}

/* the evidence weighting of merge-flag --apply / --weighted / count-field --mf-w: on material cells (w > 0) pc ps toward 128 and w by
   f = 1 - K mf / 255 (cut > 0: f = 0 at mf >= cut, else 1), w kept >= 1; pc / ps may be nullptr; nmat / fsum (may be nullptr) += the
   material cells and the sum of f */
static inline uint8_t mf_scale_phase(uint8_t v, float f) { const float d = (v - 128) * f; return (uint8_t)lrintf(128 + d); }
static inline void mf_weight_rows(uint8_t *pc, uint8_t *ps, uint8_t *w, const uint8_t *mf, size_t n, double K, int cut, size_t *nmat, double *fsum) {
    size_t nm = 0; double fs = 0;
    for (size_t i = 0; i < n; i++) {
        if (!w[i]) continue;
        const float f = cut ? (mf[i] >= cut ? 0.f : 1.f) : fmaxf(0.f, 1.f - (float)K * mf[i] / 255.f);
        if (pc) pc[i] = mf_scale_phase(pc[i], f);
        if (ps) ps[i] = mf_scale_phase(ps[i], f);
        const int v = (int)lrintf(w[i] * f); w[i] = (uint8_t)(v < 1 ? 1 : v); nm++; fs += f;
    }
    if (nmat) *nmat += nm;
    if (fsum) *fsum += fs;
}
