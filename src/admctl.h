/* Adaptive Muon's scalar controllers (2026-11-19, admuon2; DESIGN.md "Fully adaptive Muon, part 2"), shared by the GPU (ad_ctl_k in
   nn_opt.cu, one thread) and the host tests (tests/test_admctl.c): plain C, double precision, no reductions (the callers sum their
   fixed-order partials first, so every data-parallel replica computes the same numbers).
   State: AD_NST doubles (nn_madapt.dog), the first four as before 11-19 ([rbar, G, eta, |x - x0|]).
   - online kappa (adm_kap): the random-walk variance factor of the update directions, kappa = 1 + 2 sum_k rho_k, from a subsample v of
     the direction (the probe): e_t = lam e_t-1 + v_t, c_t = <v_t, e_t-1>, q_t = |v_t|^2, EMAs (bk) of c and q, kappa = 1 + 2 cbar / qbar
     clamped to [1, (1 + beta) / (1 - beta)] (the plain-momentum value: the old fixed kappa, which collapsed DoG's eta to 1.6 % of the
     lr by update 80 on scroll data because Muon's orthogonalised directions decorrelate much faster: probe kappa16 2.2 - 5.5).
     lam = 1 - (1 - beta) / 8 (a taper of ~8 / (1 - beta) updates: on an AR(1) direction with coefficient beta the estimate is
     1 + 2 beta / (1 - lam beta), 17.2 for the true 19 at beta 0.9).
   - DoG (adm_dog): eta_t = rbar_t / sqrt(G_t), G_t = sum_s kappa_s sched_s^2 |v_s|^2 (kappa at the time of the step: the variance of a
     walk whose correlation changes is the sum of the per-step kappa |step|^2), DoWG (w = 1, Khaled, Mishchenko & Jin 2023, arXiv
     2305.16284): eta_t = rbar_t^2 / sqrt(sum_s rbar_s^2 kappa_s sched_s^2 |v_s|^2). Both: first step eta0, cap dmax.
   - autocool (adm_cool): the global lr factor f (x the schedule) x decay (floor) after a stationary block of M windows of N updates.
     From the probe's weight samples theta (before each step) and the applied steps D = theta_t - theta_t+1:
       z = sum (2 (theta - c).D - |D|^2) / sum |D|^2 over the block: the stationarity of |theta - c|^2 (Yaida 2018, arXiv
           1810.00004: at a stationary distribution E[2 (theta - c).D] = E[|D|^2] exactly, for any update rule and any fixed c;
           z > 0 while theta contracts toward c, < 0 while it spreads; the SASA test of Lang, Xiao & Zhang 2019, arXiv 1909.11701,
           on the subsample). c = a slow EMA of theta (horizon N M updates): about c = 0 the noise of theta.D grows as |theta| / |D|,
           i.e. 1 / f after each decay (window z +-5 at f 1/8 on the test quadratic, decisions became chance). The moving centre
           is exact in the sums: y = theta - c, y' = (1 - ck)(y - D) gives E[2 y.D - |D|^2] = -((1 - ck)^-2 - 1) E|y'|^2 at
           stationarity, so z adds that term (uncorrected: z -5.7 at stationarity, 2026-11-19). The block pools M windows' sums;
       r = |theta_end - theta_start|^2 / (N sum |D|^2) per window: the progress fraction (1 a straight drift, ~kappa / N a random
           walk of correlation kappa, ~0 mean-reverting in a basin; r - kappa / N ~ the drift's SNR). Not normalised by the online
           kappa: a drift is long-range correlation itself and drives kappa to its clamp (pk = r N / kappa stayed ~2 on a pure
           slope, 2026-11-19).
     Stationary block: |z| < delta and every window's r < c. Decided at each block end (then a new block); one step between
     windows is not counted (the window's start sample). The GT watchdog (external) stays the arbiter. */
#pragma once
#include <math.h>
#ifdef __CUDACC__
#define ADM_HD __host__ __device__ static inline
#else
#define ADM_HD static inline
#endif

enum {
    AD_RBAR, AD_G, AD_ETA, AD_DIST,             /* DoG (checkpoint adm_dog: the first three) */
    AD_CBAR, AD_QBAR, AD_KAP, AD_KN,            /* online kappa (checkpoint adm_kap) */
    AD_EN, AD_RST,                              /* runtime: e-buffer fills since this process started; 1 = state restored (EMAs wait for e) */
    AD_COOL, AD_STAT, AD_NDEC,                  /* autocool: f, windows in the current block, decays so far (checkpoint adm_cool; the block restarts at a resume) */
    AD_WK, AD_WZ, AD_WQ, AD_BZ, AD_BQ, AD_BR, AD_Z, AD_PK, AD_WS, AD_TV,   /* runtime: window steps and sums, block sums and max r, last z / r, 1 = start a window, 1 = theta_prev valid */
    AD_NST = 32
};

typedef struct { int n, m; double delta, c, decay, floor; } adm_cool_cfg;
ADM_HD double adm_ckeep(adm_cool_cfg c) { return 1.0 / ((double)c.n * c.m); }   /* the centre EMA's rate */
ADM_HD double adm_lam(double beta) { return 1.0 - (1.0 - beta) / 8.0; }
ADM_HD double adm_kmax(double beta) { return (1.0 + beta) / (1.0 - beta); }
/* one update's kappa sample: c = <v, e_prev>, q = |v|^2 (e updated by the caller afterwards) */
ADM_HD void adm_kap(double *S, double c, double q, double beta, double bk) {
    S[AD_EN] += 1.0;
    if (S[AD_RST] != 0.0) { if (S[AD_EN] <= floor(8.0 / (1.0 - beta) + 0.5)) return; S[AD_RST] = 0.0; }   /* resumed: e restarts at 0, keep the saved EMAs until it is filled */
    if (!(q > 0.0)) return;
    S[AD_CBAR] = bk * S[AD_CBAR] + (1.0 - bk) * c; S[AD_QBAR] = bk * S[AD_QBAR] + (1.0 - bk) * q; S[AD_KN] += 1.0;
    S[AD_KAP] = fmin(fmax(1.0 + 2.0 * S[AD_CBAR] / S[AD_QBAR], 1.0), adm_kmax(beta));
}
ADM_HD double adm_kappa(const double *S) { return S[AD_KAP] >= 1.0 ? S[AD_KAP] : 1.0; }
/* DoG / DoWG: kap the factor for this step, sched the lr multiplier on top of eta (schedule x autocool), v2 = sum |v|^2; returns eta */
ADM_HD double adm_dog(double *S, double kap, double sched, double v2, double eta0, double dmax, int w) {
    const double gi = kap * sched * sched * v2;
    if (S[AD_G] <= 0.0 && gi > 0.0) S[AD_RBAR] = fmax(S[AD_RBAR], eta0 * sqrt(gi));
    S[AD_RBAR] = fmax(S[AD_RBAR], S[AD_DIST]);
    if (w) { S[AD_G] += S[AD_RBAR] * S[AD_RBAR] * gi; S[AD_ETA] = S[AD_G] > 0.0 ? fmin(S[AD_RBAR] * S[AD_RBAR] / sqrt(S[AD_G]), dmax) : eta0; }
    else { S[AD_G] += gi; S[AD_ETA] = S[AD_G] > 0.0 ? fmin(S[AD_RBAR] / sqrt(S[AD_G]), dmax) : eta0; }
    return S[AD_ETA];
}
ADM_HD double adm_cool_f(const double *S) { return S[AD_COOL] > 0.0 ? S[AD_COOL] : 1.0; }
/* one update's autocool sample, after the stat kernel: a1 = <theta_prev - c, D>, a2 = |D|^2, yy = |theta - c'|^2 (c' the updated centre;
   all valid when S[AD_TV] was set before the kernel ran), pw = |theta - theta_start|^2 (0 at a window start). Sets AD_WS for the next
   kernel at a window end. */
ADM_HD void adm_cool(double *S, double a1, double a2, double yy, double pw, adm_cool_cfg c) {
    const double ck = adm_ckeep(c), cy = 1.0 / ((1.0 - ck) * (1.0 - ck)) - 1.0;
    if (S[AD_COOL] <= 0.0) S[AD_COOL] = 1.0;
    const int tv = S[AD_TV] != 0.0; S[AD_TV] = 1.0;
    if (!tv) { S[AD_STAT] = S[AD_BZ] = S[AD_BQ] = S[AD_BR] = 0.0; S[AD_WS] = 1.0; return; }   /* (a fresh process: the next kernel starts a window) */
    if (S[AD_WS] != 0.0) { S[AD_WS] = 0.0; S[AD_WK] = S[AD_WZ] = S[AD_WQ] = 0.0; return; }   /* the kernel set theta_start = theta */
    S[AD_WZ] += 2.0 * a1 - a2 + cy * yy; S[AD_WQ] += a2; S[AD_WK] += 1.0;
    if (S[AD_WK] < c.n) return;
    S[AD_WS] = 1.0;
    if (!(S[AD_WQ] > 0.0)) return;
    S[AD_PK] = pw / (S[AD_WK] * S[AD_WQ]); S[AD_Z] = S[AD_WZ] / S[AD_WQ];
    S[AD_BZ] += S[AD_WZ]; S[AD_BQ] += S[AD_WQ]; S[AD_BR] = fmax(S[AD_BR], S[AD_PK]); S[AD_STAT] += 1.0;
    if (S[AD_STAT] < c.m) return;
    const double z = S[AD_BZ] / S[AD_BQ];
    if (fabs(z) < c.delta && S[AD_BR] < c.c) { S[AD_COOL] = fmax(S[AD_COOL] * c.decay, c.floor); S[AD_NDEC] += 1.0; }
    S[AD_STAT] = S[AD_BZ] = S[AD_BQ] = S[AD_BR] = 0.0;
}
