/* admuon2's scalar controllers (src/admctl.h; DESIGN.md "Fully adaptive Muon, part 2"), on the host, with the GPU kernel's sample
   statistics (adm_stat_k: c = <v, e_prev>, q = |v|^2, e = lam e + v; a1 = <theta_prev, D>, a2 = |D|^2, pw = |theta - theta_start|^2)
   computed here in double:
   - online kappa: white-noise directions give ~1, AR(1) directions with coefficient b the tapered value 1 + 2 b / (1 - lam b), a
     straight drift the clamp (1 + beta) / (1 - beta); a resumed state keeps its EMAs until e is refilled;
   - DoG: the first step is eta0 (DoG and DoWG); on momentum-correlated noise (AR(1), beta 0.9) the fixed plain-momentum kappa (19)
     shrinks eta (the 11-14 collapse), kappa = 1 lets it climb, the online kappa keeps it near eta0; a straight drift reaches the cap;
     DoWG climbs on noise (its rbar^2 weighting has no fixed point on an unbounded walk: rbar^2 / sqrt(sum rbar_s^2 g) -> sqrt 2 eta);
   - autocool on a noisy quadratic (heavy-ball momentum, 2000 coordinates, curvatures 0.05 .. 1, optimum away from 0): no decay while
     the loss is still far above its noise floor, decays after it, the final loss below the uncooled floor; on a noisy linear slope
     (a drift that never ends) no decay. ADM_DEBUG=1 prints every window. */
#include "admctl.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { printf("  FAIL: " __VA_ARGS__); printf("\n"); fails++; } } while (0)
static unsigned long long rs = 88172645463325252ull;
static double urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 11) * (1.0 / 9007199254740992.0); }
static double grand(void) { double u = urand(), v = urand(); return sqrt(-2.0 * log(u + 1e-300)) * cos(6.283185307179586 * v); }

enum { D = 2000 };
typedef struct { double e[D], tp[D], ts[D], tc[D]; } probe;   /* the kernel's device state */
/* adm_stat_k on a full sample (stride 1): returns c, q, a1, a2, pw in t[] */
static void stat(probe *p, const double *S, const double *v, const double *th, double lam, double ck, int kap, int cool, double t[6]) {
    memset(t, 0, 6 * sizeof(double)); const int tv = cool && S[AD_TV] != 0.0, ws = cool && S[AD_WS] != 0.0;
    for (int j = 0; j < D; j++) {
        if (kap) { t[0] += v[j] * p->e[j]; t[1] += v[j] * v[j]; p->e[j] = lam * p->e[j] + v[j]; }
        if (cool) {
            if (tv) { const double dd = p->tp[j] - th[j]; t[2] += (p->tp[j] - p->tc[j]) * dd; t[3] += dd * dd; p->tc[j] += ck * (th[j] - p->tc[j]); t[5] += (th[j] - p->tc[j]) * (th[j] - p->tc[j]); }
            else p->tc[j] = th[j];
            p->tp[j] = th[j];
            if (ws) p->ts[j] = th[j]; else { const double w = th[j] - p->ts[j]; t[4] += w * w; }
        }
    }
}

/* AR(1) directions v = b v + sqrt(1 - b^2) xi (unit variance), or a constant drift (b < 0) */
static double kappa_run(double b, double beta, int steps, double *S, probe *p) {
    static double v[D]; double t[6];
    for (int j = 0; j < D; j++) v[j] = b < 0 ? 1.0 : grand();
    for (int it = 0; it < steps; it++) {
        if (b >= 0) for (int j = 0; j < D; j++) v[j] = b * v[j] + sqrt(1 - b * b) * grand();
        stat(p, S, v, v, adm_lam(beta), 0, 1, 0, t); adm_kap(S, t[0], t[1], beta, 0.98);
    }
    return adm_kappa(S);
}
static void test_kappa(void) {
    static probe p; double S[AD_NST]; const double beta = 0.95, lam = adm_lam(beta);
    memset(&p, 0, sizeof p); memset(S, 0, sizeof S); const double k0 = kappa_run(0.0, beta, 2000, S, &p);
    memset(&p, 0, sizeof p); memset(S, 0, sizeof S); const double k5 = kappa_run(0.5, beta, 2000, S, &p);
    memset(&p, 0, sizeof p); memset(S, 0, sizeof S); const double k9 = kappa_run(0.9, beta, 3000, S, &p);
    double Sd[AD_NST]; memset(&p, 0, sizeof p); memset(Sd, 0, sizeof Sd); const double kd = kappa_run(-1.0, beta, 2000, Sd, &p);
    const double e5 = 1 + 2 * 0.5 / (1 - lam * 0.5), e9 = 1 + 2 * 0.9 / (1 - lam * 0.9);
    printf("kappa: white %.3f, AR 0.5 %.3f (tapered %.3f, true 3), AR 0.9 %.3f (tapered %.3f, true 19), drift %.2f (clamp %.1f)\n", k0, k5, e5, k9, e9, kd, adm_kmax(beta));
    CHECK(k0 < 1.1, "white-noise kappa %.3f", k0);
    CHECK(fabs(k5 / e5 - 1) < 0.08, "AR 0.5 kappa %.3f vs %.3f", k5, e5);
    CHECK(fabs(k9 / e9 - 1) < 0.15, "AR 0.9 kappa %.3f vs %.3f", k9, e9);
    CHECK(fabs(kd - adm_kmax(beta)) < 1e-9, "drift kappa %.3f", kd);
    /* resume: the saved EMAs (AR 0.9) stay while e refills from 0 (8 / (1 - beta) = 160 samples), white noise afterwards pulls them down */
    double S2[AD_NST]; memcpy(S2, S, sizeof S2);   /* (S: the AR 0.9 run's state) */ S2[AD_EN] = 0; S2[AD_RST] = 1; memset(&p, 0, sizeof p);
    const double ks = S2[AD_KAP]; kappa_run(0.9, beta, 160, S2, &p);
    CHECK(S2[AD_KAP] == ks && S2[AD_RST] == 1, "resume: kappa moved during the refill (%.3f -> %.3f)", ks, S2[AD_KAP]);
    kappa_run(0.9, beta, 400, S2, &p);
    printf("kappa resume: held %.3f over the refill, %.3f after 400 more AR 0.9 samples\n", ks, S2[AD_KAP]);
    CHECK(S2[AD_RST] == 0 && fabs(S2[AD_KAP] / e9 - 1) < 0.15, "resume: kappa after the refill %.3f", S2[AD_KAP]);
}

/* DoG on directions v (unit-variance AR(1) of coefficient b, or a drift), kappa: > 0 fixed, 0 online; returns eta / eta0 at the end */
static double dog_run(double b, double kfix, int w, int steps, double *first) {
    static double v[D], x[D]; static probe p; double S[AD_NST], t[6]; const double beta = 0.9, eta0 = 0.01;
    memset(S, 0, sizeof S); memset(&p, 0, sizeof p); memset(x, 0, sizeof x);
    for (int j = 0; j < D; j++) v[j] = b < 0 ? 1.0 / sqrt(D) : grand() / sqrt(D);
    for (int it = 0; it < steps; it++) {
        if (b >= 0) for (int j = 0; j < D; j++) v[j] = b * v[j] + sqrt(1 - b * b) * grand() / sqrt(D);
        double v2 = 0; for (int j = 0; j < D; j++) v2 += v[j] * v[j];
        stat(&p, S, v, v, adm_lam(beta), 0, 1, 0, t); adm_kap(S, t[0], t[1], beta, 0.98);
        const double eta = adm_dog(S, kfix > 0 ? kfix : adm_kappa(S), 1.0, v2, eta0, 100 * eta0, w);
        if (it == 0) *first = eta / eta0;
        double d2 = 0; for (int j = 0; j < D; j++) { x[j] -= eta * v[j]; d2 += x[j] * x[j]; }
        S[AD_DIST] = sqrt(d2);
    }
    return S[AD_ETA] / eta0;
}
static void test_dog(void) {
    double f1, f2, f3, f4, f5, f6;
    const double fix = dog_run(0.9, 19.0, 0, 1500, &f1), one = dog_run(0.9, 1.0, 0, 1500, &f2), onl = dog_run(0.9, 0, 0, 1500, &f3);
    const double wonl = dog_run(0.9, 0, 1, 1500, &f4), drift = dog_run(-1, 0, 0, 400, &f5), white = dog_run(0.0, 0, 0, 1500, &f6);
    printf("dog eta / eta0 after 1500 AR 0.9 updates: kappa 19 %.3f, kappa 1 %.3f, online %.3f, DoWG online %.3f; white online %.3f; drift %.1f (cap 100)\n",
           fix, one, onl, wonl, white, drift);
    CHECK(fabs(f1 - 1) < 1e-12 && fabs(f2 - 1) < 1e-12 && fabs(f3 - 1) < 1e-12 && fabs(f4 - 1) < 1e-12, "first step != eta0 (%g %g %g %g)", f1, f2, f3, f4);
    CHECK(fix < 0.6, "fixed kappa 19 should shrink eta on this AR(1) walk: %.3f", fix);
    CHECK(one > onl, "kappa 1 should sit above the online kappa: %.3f vs %.3f", one, onl);
    CHECK(onl > 0.5 && onl < 3.0, "online kappa: eta %.3f x eta0 on noise", onl);
    CHECK(wonl >= onl, "DoWG online: eta %.3f below DoG's %.3f on noise", wonl, onl);   /* (DoWG's time weighting: eta grows ~sqrt 2 per doubling of the walk; documents it) */
    CHECK(white > 0.5 && white < 3.0, "online kappa on white noise: eta %.3f x eta0", white);
    CHECK(drift > 50, "a straight drift should drive eta to the cap: %.1f", drift);
}

/* heavy ball on f = 1/2 sum h (theta - theta*)^2 + slope . theta with gradient noise sigma; autocool or not; returns the mean loss
   (without the slope) of the last 500 steps, the first decay's step and the loss excess (loss / the uncooled floor) at that step */
typedef struct { double loss_end; int first_dec, ndec; double loss_at_dec; double f; } cool_res;
static cool_res cool_run(int cool, double slope, int steps, double floor_ref) {
    static double th[D], ts[D], h[D], m[D]; static probe p; double S[AD_NST], t[6];
    const double beta = 0.9, lr = 0.05, sigma = 1.0; const adm_cool_cfg cfg = {100, 3, 0.2, 0.2, 0.5, 0.01};
    rs = 1234567ull; memset(S, 0, sizeof S); memset(&p, 0, sizeof p); memset(m, 0, sizeof m);
    for (int j = 0; j < D; j++) { h[j] = slope != 0.0 ? 0.0 : 0.002 * pow(100.0, urand()); ts[j] = 3.0 * grand(); th[j] = ts[j] + 30.0 * grand(); }
    cool_res r = {0, -1, 0, 0, 1}; double acc = 0; int nacc = 0;
    for (int it = 0; it < steps; it++) {
        double loss = 0;
        for (int j = 0; j < D; j++) { const double d = th[j] - ts[j]; loss += 0.5 * h[j] * d * d; const double g = h[j] * d + slope + sigma * grand(); m[j] = beta * m[j] + g; }
        loss /= D;
        if (cool) {
            stat(&p, S, m, th, adm_lam(beta), adm_ckeep(cfg), 1, 1, t); adm_kap(S, t[0], t[1], beta, 0.98);
            const double nd = S[AD_NDEC]; adm_cool(S, t[2], t[3], t[5], t[4], cfg);
            if (getenv("ADM_DEBUG") && S[AD_WS] != 0.0) printf("    it %d loss %.4f z %.3f r %.3f kappa %.2f f %.4g dec %g\n", it, loss, S[AD_Z], S[AD_PK], adm_kappa(S), adm_cool_f(S), S[AD_NDEC]);
            if (S[AD_NDEC] > nd && r.first_dec < 0) { r.first_dec = it; r.loss_at_dec = loss / floor_ref; }
        }
        const double f = cool ? adm_cool_f(S) : 1.0;
        for (int j = 0; j < D; j++) th[j] -= lr * f * m[j];
        if (it >= steps - 500) { acc += loss; nacc++; }
    }
    r.loss_end = acc / nacc; r.ndec = (int)S[AD_NDEC]; r.f = cool ? adm_cool_f(S) : 1.0;
    return r;
}
static void test_cool(void) {
    const cool_res base = cool_run(0, 0.0, 8000, 1.0);   /* the uncooled noise floor */
    const cool_res c = cool_run(1, 0.0, 8000, base.loss_end), s = cool_run(1, 0.5, 8000, base.loss_end);
    printf("autocool quadratic: uncooled floor %.4f; cooled: first decay at step %d (loss %.2f x floor), %d decays, f %.4g, final loss %.4f (%.2f x floor)\n",
           base.loss_end, c.first_dec, c.loss_at_dec, c.ndec, c.f, c.loss_end, c.loss_end / base.loss_end);
    printf("autocool slope: %d decays (f %.4g)\n", s.ndec, s.f);
    CHECK(c.first_dec > 0 && c.loss_at_dec < 1.5, "decayed before convergence: step %d at %.2f x the floor", c.first_dec, c.loss_at_dec);
    CHECK(c.ndec >= 2, "only %d decays after convergence", c.ndec);
    CHECK(c.loss_end < 0.6 * base.loss_end, "cooling did not lower the loss: %.4f vs %.4f", c.loss_end, base.loss_end);
    CHECK(s.ndec == 0, "decayed on a never-ending drift: %d", s.ndec);
}

int main(void) {
    test_kappa();
    test_dog();
    test_cool();
    if (fails) { printf("test_admctl: %d FAILED\n", fails); return 1; }
    printf("test_admctl: OK\n");
    return 0;
}
