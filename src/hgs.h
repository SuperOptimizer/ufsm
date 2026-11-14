/* the head-gradient plane scale's choice (hgs, 2026-11-14): one definition for the host (train.c's check, tests) and the device (nn_hgs.cu,
   a UFSM_STRICT_UNITS unit: IEEE division, no FMA contraction, no flush; the host C is built -ffp-contract=off), so both give the same
   bits. The choice used to run on the host after an nn_sync + 9 small readbacks per window (audit 11-14, train.md #1 follow-up); the
   device kernel reads the loss scratches in place and writes {s, 1 / s} for the gradient pass. s = 2^k, k = floor(log2(512 / b)) clamped
   to [0, 30] with b the largest per-voxel bound: k from the exponent of 512 / b (frexpf, exact) where floorf(log2f()) could round a
   quotient within an ulp below 2^k up to k (then s b slightly above 2^9; never otherwise different) */
#pragma once
#include <math.h>
#if defined(__CUDACC__)
#define HGS_HD __host__ __device__
#else
#define HGS_HD
#endif
#define HGS_NT 8
typedef struct { const float *p; int C; float lam, dice, pw; } hgs_term;   /* a loss scratch ([C][5] sums, fin [2 C + 2]) and its bound's terms */
typedef struct hgs_args {
    hgs_term t[HGS_NT]; int nt;
    const float *w4; float w4lam;   /* the wind4 fin (w4[3]: 1 / N) and lambda x source weight; w4 nullptr: no wind4 term */
    float milv;                      /* the MIL term (host-known: 8 |lambda| gs / nmil), 0 without */
    float gs, rw, mw;                /* the gradient scale, the recall weight, the largest mask weight x 2 */
} hgs_args;
static inline HGS_HD float hgs_term_bound(const float *h, int C, float lam, float dice, float pw, float rw, float mw, float gs) {
    const float fsc = h[5 * C + 2 * C + 1]; float b = 0.f;
    for (int c = 0; c < C; c++) { const float nm = h[5 * c]; if (nm < 1.f) continue;
        const float v = fabsf(lam) * gs * fsc * ((pw > 1.f ? pw : 1.f) * mw / nm + 1.25f * dice + 0.25f * rw); if (v > b) b = v; }
    return b;
}
/* the bound b over every term; pointers are read as given (device pointers on the device, host copies on the host) */
static inline HGS_HD float hgs_bound_all(const hgs_args *a) {
    float b = 0.f, v;
    for (int i = 0; i < a->nt; i++) { v = hgs_term_bound(a->t[i].p, a->t[i].C, a->t[i].lam, a->t[i].dice, a->t[i].pw, a->rw, a->mw, a->gs); if (v > b) b = v; }
    if (a->w4) { v = 0.5f * a->mw * fabsf(a->w4lam) * a->gs * a->w4[3]; if (v > b) b = v; }   /* wind4: k = mask x 0.5 x (1 / N) x lambda x scale */
    if (a->milv > b) b = a->milv;
    return b;
}
/* the exponent k of s = 2^k from the bound b (0: s 1) */
static inline HGS_HD int hgs_exp(float b) {
    if (!(b > 0.f) || !isfinite(b)) return 0;
    const float q = 512.f / b;
    if (!(q >= 1.f)) return 0;
    if (q >= 1073741824.f) return 30;   /* (2^30; also q inf) */
    int e; (void)frexpf(q, &e);         /* q = m 2^e, m in [0.5, 1): floor(log2 q) = e - 1 */
    return e - 1;
}
