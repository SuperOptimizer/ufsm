/* Fully adaptive Muon (DESIGN.md 2026-11-14 "Fully adaptive Muon"):
   - --opt-stats: nn_muon_stats changes no weight (bit for bit against a run without it) and its per-matrix |update|^2, |W|^2 equal the
     host's sums of the actual weight change and the weights before the step (plain batched Muon and the ext path with NorMuon);
   - adaptive Muon (nn_muon_ext_adapt): over 5 steps with a varying schedule and decay, the weights, the (b) ratio EMAs and the (c) DoG
     state equal a double-precision reference fed the GPU's own Muon direction (from a twin batch at lr 1, wd 0, weights reset each
     step: O = -dp / s): (b) with the geometric-mean target, (b) with an absolute target that hits the clamp, (c) alone, and NorMuon +
     (b) + (c) together;
   - --muon-scale rms (DESIGN.md 2026-11-15 "Muon update scaling"): from the same state, the rms step equals the kj step times
     r sqrt(max(Co, K)) / sqrt(max(1, Co / K)) per matrix (the same direction O), its element RMS is ~r / lr, and exactly r under
     --muon-fro (O rescaled to the nominal RMS); plain and ext (+ NorMuon);
   - admuon2 (2026-11-19, src/admctl.h): the online kappa, DoG on it and autocool on the GPU equal the host controllers fed the
     device's own probe samples (test_adapt2; the controllers themselves: tests/test_admctl.c). */
#include "nn.h"
#include "admctl.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { float *p; const float *g; float *mom, *X, *Y, *A, *B; int Co, K; } muon_desc_t;   /* mirrors nn_common.cuh */

static int fails;
#define CHECK(c, ...) do { if (!(c)) { printf("  FAIL: " __VA_ARGS__); printf("\n"); fails++; } } while (0)
static unsigned rs = 77;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return (rs >> 8) / 16777216.f - 0.5f; }

enum { N = 5 };
static const int CO[N] = {16, 48, 64, 96, 40}, KK[N] = {108, 16, 64, 864, 1080};
typedef struct { muon_desc_t h[N]; void *dd; int maxco, maxk; double *part; float *rows; int *off; int tot; } batch;
static void batch_new(batch *b, unsigned seed) {
    memset(b, 0, sizeof *b); int off[N + 1], t = 0; rs = seed;
    for (int i = 0; i < N; i++) {
        const size_t nw = (size_t)CO[i] * KK[i]; muon_desc_t *d = &b->h[i];
        d->p = nn_malloc(nw * 4); d->g = nn_malloc(nw * 4); d->mom = nn_malloc(nw * 4); d->X = nn_malloc(nw * 4); d->Y = nn_malloc(nw * 4);
        d->A = nn_malloc((size_t)CO[i] * CO[i] * 4); d->B = nn_malloc((size_t)CO[i] * CO[i] * 4); d->Co = CO[i]; d->K = KK[i];
        float *w = malloc(nw * 4); for (size_t j = 0; j < nw; j++) w[j] = frand() * 0.2f * (1.f + i); nn_h2d(d->p, w, nw * 4); free(w);
        nn_zero(d->mom, nw * 4);
        if (CO[i] > b->maxco) b->maxco = CO[i];
        if (KK[i] > b->maxk) b->maxk = KK[i];
        off[i] = t; t += CO[i];
    }
    off[N] = t; b->tot = t;
    b->dd = nn_malloc(sizeof b->h); nn_h2d(b->dd, b->h, sizeof b->h);
    b->part = nn_malloc((size_t)32 * N * 16 * sizeof(double)); b->rows = nn_malloc((size_t)2 * t * 4); nn_zero(b->rows, (size_t)2 * t * 4);
    b->off = nn_malloc((N + 1) * sizeof(int)); nn_h2d(b->off, off, (N + 1) * sizeof(int));
}
static void batch_free(batch *b) {
    for (int i = 0; i < N; i++) { muon_desc_t *d = &b->h[i]; nn_free(d->p); nn_free((void *)d->g); nn_free(d->mom); nn_free(d->X); nn_free(d->Y); nn_free(d->A); nn_free(d->B); }
    nn_free(b->dd); nn_free(b->part); nn_free(b->rows); nn_free(b->off);
}
static void set_g(batch *a, batch *b, unsigned seed) {
    rs = seed;
    for (int i = 0; i < N; i++) {
        const size_t nw = (size_t)CO[i] * KK[i]; float *g = malloc(nw * 4); for (size_t j = 0; j < nw; j++) g[j] = frand() * (1.f + (j % 7) * 0.3f) * (1.f + (j / KK[i]) % 3);
        nn_h2d((float *)a->h[i].g, g, nw * 4); if (b) nn_h2d((float *)b->h[i].g, g, nw * 4); free(g);
    }
}
static float *get_p(const batch *b, int i) { const size_t nw = (size_t)CO[i] * KK[i]; float *h = malloc(nw * 4); nn_d2h(h, b->h[i].p, nw * 4); return h; }
static void step(batch *b, int ext, int nm, float lr, float wd) {
    if (!ext) nn_muon_batch(b->dd, N, b->maxco, b->maxk, lr, 0.9f, wd);
    else nn_muon_ext_batch(b->dd, N, b->maxco, b->maxk, nullptr, nullptr, 3, 0, 0, nullptr, nullptr, 0, b->part, lr, 0.9f, wd, 0.f, 0, nm ? b->rows : nullptr, b->off, b->tot, 0.9f, nullptr, 0.f);
}

/* stats on A, off on B: identical weights; A's sums equal the host's */
static void test_stats(int ext, int nm) {
    batch a, b; batch_new(&a, 5); batch_new(&b, 5);
    double *st = nn_malloc((size_t)2 * (N + 3) * sizeof(double));
    const float lr = 0.02f, wd = 0.1f; double worst = 0; int same = 1;
    for (int it = 0; it < 3; it++) {
        set_g(&a, &b, 100 + it);
        float *p0[N]; for (int i = 0; i < N; i++) p0[i] = get_p(&a, i);
        nn_zero(st, (size_t)2 * (N + 3) * sizeof(double)); nn_muon_stats(st, 3);   /* base 3: the sums land at [6, 6 + 2 N) */
        step(&a, ext, nm, lr, wd); nn_muon_stats(nullptr, 0); step(&b, ext, nm, lr, wd);
        double h[2 * (N + 3)]; nn_d2h(h, st, sizeof h);
        CHECK(h[0] == 0 && h[5] == 0, "stats outside base");
        for (int i = 0; i < N; i++) {
            const size_t nw = (size_t)CO[i] * KK[i]; float *pa = get_p(&a, i), *pb = get_p(&b, i);
            same &= !memcmp(pa, pb, nw * 4);
            double dd = 0, ww = 0; for (size_t j = 0; j < nw; j++) { const double d = (double)pa[j] - p0[i][j]; dd += d * d; ww += (double)p0[i][j] * p0[i][j]; }
            const double e1 = fabs(h[2 * (3 + i)] - dd) / dd, e2 = fabs(h[2 * (3 + i) + 1] - ww) / ww;
            if (e1 > worst) worst = e1;
            if (e2 > worst) worst = e2;
            free(pa); free(pb); free(p0[i]);
        }
    }
    printf("stats %s%s: weights %s, worst rel err of the sums %.2e\n", ext ? "ext" : "plain", nm ? "+normuon" : "", same ? "identical" : "DIFFER", worst);
    CHECK(same, "stats changed the weights");
    CHECK(worst < 1e-3, "stats sums off: %.3e", worst);
    nn_free(st); batch_free(&a); batch_free(&b);
}

static double mscale(int Co, int K) { return sqrt(fmax(1.0, (double)Co / K)); }
/* lars / dog on batch A against the reference; B gives the direction */
static void test_adapt(int nm, int lars, float tgt, int dog, float kappa) {
    batch a, b; batch_new(&a, 9); batch_new(&b, 9);
    const float lr0 = 0.02f, wd = 0.1f, sched[5] = {1.f, 1.f, 0.7f, 0.4f, 0.1f}, lb = 0.9f, lc = 2.f, dmax = 3.f;
    float *lrm = nn_malloc(N * 4), *ema = nn_malloc(N * 4), *disp[N]; double *dst = nn_malloc(AD_NST * sizeof(double)), *part = nn_malloc((size_t)64 * N * sizeof(double));
    nn_zero(ema, N * 4); nn_zero(dst, AD_NST * sizeof(double));
    for (int i = 0; i < N; i++) { const size_t nw = (size_t)CO[i] * KK[i]; disp[i] = nn_malloc(nw * 4); nn_zero(disp[i], nw * 4); }
    void *ddisp = nn_malloc(sizeof disp); nn_h2d(ddisp, disp, sizeof disp);
    const nn_madapt ad = {.lrm = lrm, .ema = ema, .dog = dst, .disp = (float *const *)ddisp, .part = part, .lr_peak = lr0, .lars_tgt = tgt, .lars_b = lb, .lars_c = lc,
                          .dog_eta0 = lr0, .dog_max = dmax * lr0, .dog_kappa = kappa, .on_lars = lars, .on_dog = dog};
    double *P[N], *Dd[N], rema[N] = {0}, rdog[4] = {0}, worst = 0, worst_s = 0; int clamped = 0;
    for (int i = 0; i < N; i++) { float *p = get_p(&a, i); const size_t nw = (size_t)CO[i] * KK[i]; P[i] = malloc(nw * sizeof(double)); Dd[i] = calloc(nw, sizeof(double)); for (size_t j = 0; j < nw; j++) P[i][j] = p[j]; free(p); }
    for (int it = 0; it < 5; it++) {
        set_g(&a, &b, 300 + it); const float lr = lr0 * sched[it];
        for (int i = 0; i < N; i++) nn_zero(b.h[i].p, (size_t)CO[i] * KK[i] * 4);
        nn_muon_ext_adapt(&ad); step(&a, 1, nm, lr, wd); nn_muon_ext_adapt(nullptr);
        step(&b, 1, nm, 1.f, 0.f);
        double *O[N], so[N], sw[N], lsum = 0, m[N], v2 = 0;
        for (int i = 0; i < N; i++) {
            const size_t nw = (size_t)CO[i] * KK[i]; float *pb = get_p(&b, i); O[i] = malloc(nw * sizeof(double)); so[i] = sw[i] = 0;
            for (size_t j = 0; j < nw; j++) { O[i][j] = -(double)pb[j] / (float)mscale(CO[i], KK[i]); so[i] += O[i][j] * O[i][j]; sw[i] += P[i][j] * P[i][j]; }
            free(pb);
            const double rho = mscale(CO[i], KK[i]) * sqrt(so[i]) / sqrt(sw[i]);
            if (lars) { rema[i] = it ? lb * rema[i] + (1 - lb) * rho : rho; lsum += log(rema[i]); }
        }
        const double t = tgt > 0 ? tgt / lr0 : exp(lsum / N);
        for (int i = 0; i < N; i++) { m[i] = lars ? fmin(fmax(t / rema[i], 1 / (double)lc), lc) : 1; if (lars && (m[i] == lc || m[i] == 1 / (double)lc)) clamped++;
            v2 += m[i] * m[i] * mscale(CO[i], KK[i]) * mscale(CO[i], KK[i]) * so[i]; }
        double f = 1;
        if (dog) { const double sc = sched[it], gi = (double)kappa * sc * sc * v2;
            if (rdog[1] <= 0) rdog[0] = fmax(rdog[0], lr0 * sqrt(gi));
            rdog[1] += gi; rdog[0] = fmax(rdog[0], rdog[3]); rdog[2] = fmin(rdog[0] / sqrt(rdog[1]), dmax * lr0); f = rdog[2] / lr0; }
        double dd = 0, sn[N];
        for (int i = 0; i < N; i++) {
            const size_t nw = (size_t)CO[i] * KK[i]; const double l = lr * m[i] * f, s = mscale(CO[i], KK[i]); sn[i] = 0;
            for (size_t j = 0; j < nw; j++) { const double u = l * s * O[i][j], np = P[i][j] * (1 - l * wd) - u; sn[i] += (np - P[i][j]) * (np - P[i][j]); P[i][j] = np; Dd[i][j] -= u; dd += Dd[i][j] * Dd[i][j]; }
            free(O[i]);
        }
        rdog[3] = sqrt(dd);
        for (int i = 0; i < N; i++) {   /* weights: relative to the step's size (so the check is not diluted by |W|) */
            const size_t nw = (size_t)CO[i] * KK[i]; float *pa = get_p(&a, i); double e = 0, ref = 0;
            for (size_t j = 0; j < nw; j++) { const double d = pa[j] - P[i][j]; e += d * d; } ref = sn[i];
            const double r = sqrt(e / ref); if (r > worst) worst = r;
            for (size_t j = 0; j < nw; j++) P[i][j] = pa[j];   /* continue from the GPU's weights (float rounding does not accumulate) */
            free(pa);
        }
        float he[N]; double hd[4]; nn_d2h(he, ema, sizeof he); nn_d2h(hd, dst, sizeof hd);
        for (int i = 0; i < N && lars; i++) { const double r = fabs(he[i] - rema[i]) / rema[i]; if (r > worst_s) worst_s = r; }
        for (int k = 0; k < 4 && dog; k++) { const double r = fabs(hd[k] - rdog[k]) / fmax(fabs(rdog[k]), 1e-30); if (r > worst_s) worst_s = r; }
        if (dog) { rdog[3] = hd[3]; }   /* (the GPU's displacement continues from its own float buffer) */
        if (it == 4) printf("  step 5: m [%.3f .. %.3f], eta %.4g (lr0 %.3g), rbar %.4g, G %.4g, dist %.4g\n", lars ? fmin(fmin(m[0], m[1]), fmin(m[2], fmin(m[3], m[4]))) : 1.,
                            lars ? fmax(fmax(m[0], m[1]), fmax(m[2], fmax(m[3], m[4]))) : 1., dog ? hd[2] : lr0, lr0, hd[0], hd[1], hd[3]);
    }
    printf("adapt nm %d lars %d (target %g) dog %d (kappa %g): weights worst rel %.2e, state worst rel %.2e%s\n", nm, lars, tgt, dog, kappa, worst, worst_s, clamped ? " (clamp hit)" : "");
    CHECK(worst < 3e-4, "adaptive weights off: %.3e", worst);
    CHECK(worst_s < 1e-4, "adaptive state off: %.3e", worst_s);
    if (tgt > 0) CHECK(clamped, "the absolute target should hit the clamp");
    for (int i = 0; i < N; i++) { free(P[i]); free(Dd[i]); nn_free(disp[i]); }
    nn_free(lrm); nn_free(ema); nn_free(dst); nn_free(part); nn_free(ddisp); batch_free(&a); batch_free(&b);
}

/* admuon2 (src/admctl.h): online kappa + DoG + autocool through adm_stat_k / ad_ctl_k, the probe at stride 1 (so the probe's |v|^2 is
   DoG's v2). Each step the host reads the device's probe state before the step (e, theta_prev, theta_start, centre) and the samples
   after it, forms the six statistics in double and runs the same controllers; the state vectors must agree. Autocool windows of 2
   updates in blocks of 2 (decisions every 5 steps), delta / c huge so the decays fire: the decay path runs on the GPU too. */
static void test_adapt2(int nm) {
    batch a; batch_new(&a, 11);
    const float lr0 = 0.02f, wd = 0.1f, beta = 0.9f;
    long long off[N + 1], M = 0; for (int i = 0; i < N; i++) { off[i] = M; M += (long long)CO[i] * KK[i]; } off[N] = M;
    long long *doff = nn_malloc(sizeof off); nn_h2d(doff, off, sizeof off);
    float *buf = nn_malloc((size_t)2 * M * 4), *ext = nn_malloc((size_t)4 * M * 4), *lrm = nn_malloc(N * 4), *ema = nn_malloc(N * 4), *disp[N];
    nn_zero(ext, (size_t)4 * M * 4); nn_zero(ema, N * 4);
    double *dst = nn_malloc(AD_NST * sizeof(double)), *part = nn_malloc((size_t)64 * N * sizeof(double)), *xpart = nn_malloc(6 * 64 * sizeof(double));
    double S[AD_NST] = {0}; S[AD_COOL] = 1; nn_h2d(dst, S, sizeof S);
    for (int i = 0; i < N; i++) { const size_t nw = (size_t)CO[i] * KK[i]; disp[i] = nn_malloc(nw * 4); nn_zero(disp[i], nw * 4); }
    void *ddisp = nn_malloc(sizeof disp); nn_h2d(ddisp, disp, sizeof disp);
    const adm_cool_cfg cc = {2, 2, 1e9, 1e9, 0.5, 0.2};
    const nn_madapt ad = {.lrm = lrm, .ema = ema, .dog = dst, .disp = (float *const *)ddisp, .part = part, .lr_peak = lr0, .dog_eta0 = lr0, .dog_max = 3 * lr0,
                          .dog_kappa = 19.f, .on_dog = 1, .ext = ext, .xpart = xpart, .kap_b = 0.9f, .cool_delta = (float)cc.delta, .cool_c = (float)cc.c,
                          .cool_decay = (float)cc.decay, .cool_floor = (float)cc.floor, .on_kap = 1, .on_cool = 1, .cool_n = cc.n, .cool_m = cc.m};
    float *pre = malloc((size_t)4 * M * 4), *smp = malloc((size_t)2 * M * 4); double worst = 0; int wi = -1;
    const float ck = (float)adm_ckeep(cc), lam = (float)adm_lam(beta);
    for (int it = 0; it < 12; it++) {
        set_g(&a, nullptr, 500 + it); const float sched = it < 3 ? 0.5f + it * 0.25f : 1.f, lr = lr0 * sched;
        nn_d2h(pre, ext, (size_t)4 * M * 4);
        nn_muon_probe(buf, doff, 1, 0, M); nn_muon_ext_adapt(&ad); step(&a, 1, nm, lr, wd); nn_muon_ext_adapt(nullptr); nn_muon_probe(nullptr, nullptr, 0, 0, 0);
        nn_d2h(smp, buf, (size_t)2 * M * 4);
        const float *E = pre, *TP = pre + M, *TS = pre + 2 * M, *TC = pre + 3 * M, *v = smp, *th = smp + M;
        const int tv = S[AD_TV] != 0.0, ws = S[AD_WS] != 0.0; double t[6] = {0};
        for (long long j = 0; j < M; j++) {
            t[0] += (double)v[j] * E[j]; t[1] += (double)v[j] * v[j];
            if (tv) { const float dd = TP[j] - th[j], cn = TC[j] + ck * (th[j] - TC[j]); t[2] += (double)(TP[j] - TC[j]) * dd; t[3] += (double)dd * dd; t[5] += (double)(th[j] - cn) * (th[j] - cn); }
            if (!ws) { const float w = th[j] - TS[j]; t[4] += (double)w * w; }
        }
        (void)lam;
        adm_kap(S, t[0], t[1], beta, 0.9); const double kap = adm_kappa(S);
        adm_cool(S, t[2], t[3], t[5], t[4], cc); const double cf = adm_cool_f(S);
        adm_dog(S, kap, cf * lr / lr0, t[1], lr0, 3 * lr0, 0);
        double h[AD_NST]; nn_d2h(h, dst, sizeof h);
        S[AD_DIST] = h[AD_DIST];   /* (the displacement: the GPU's own float buffer) */
        for (int k = 0; k < AD_NST; k++) { if (k == AD_DIST) continue; const double r = fabs(h[k] - S[k]) / fmax(fabs(S[k]), 1e-12); if (r > worst) { worst = r; wi = k; } }
        if (it == 11) printf("  step 12: kappa %.3f, eta %.4g, cool f %.3g (%g decays), z %.3g, r %.3g\n", h[AD_KAP], h[AD_ETA], h[AD_COOL], h[AD_NDEC], h[AD_Z], h[AD_PK]);
        for (int k = 0; k < AD_NST; k++) S[k] = h[k];   /* continue from the GPU's state */
    }
    printf("adapt2 nm %d (online kappa + DoG + autocool): state worst rel %.2e (index %d)\n", nm, worst, wi);
    CHECK(worst < 1e-4, "admuon2 state off: %.3e at %d", worst, wi);
    CHECK(S[AD_NDEC] >= 1, "autocool never decayed on the GPU path");
    free(pre); free(smp);
    for (int i = 0; i < N; i++) nn_free(disp[i]);
    nn_free(doff); nn_free(buf); nn_free(ext); nn_free(lrm); nn_free(ema); nn_free(dst); nn_free(part); nn_free(xpart); nn_free(ddisp); batch_free(&a);
}

/* A at kj, B at rms r from the same weights, momentum and gradient: dB = dA * s_rms / s_kj per matrix; RMS(dB) / lr ~ r (exactly r with fro) */
static void test_scale(int ext, int nm, int fro) {
    batch a, b; batch_new(&a, 9); batch_new(&b, 9);
    const float lr = 0.01f, r = 0.2f; double worst = 0, lo = 1e30, hi = 0;
    set_g(&a, &b, 300);
    float *p0[N]; for (int i = 0; i < N; i++) p0[i] = get_p(&a, i);
    nn_muon_ext_fro(fro);
    nn_muon_scale_mode(0); step(&a, ext, nm, lr, 0.f);
    nn_muon_scale_rms(r); nn_muon_scale_mode(2); step(&b, ext, nm, lr, 0.f); nn_muon_scale_mode(0); nn_muon_ext_fro(0);
    for (int i = 0; i < N; i++) {
        const size_t nw = (size_t)CO[i] * KK[i]; float *pa = get_p(&a, i), *pb = get_p(&b, i);
        const double ratio = r * sqrt(fmax(CO[i], KK[i])) / sqrt(fmax(1.0, (double)CO[i] / KK[i]));
        double mx = 0, err = 0, ss = 0;
        for (size_t j = 0; j < nw; j++) { const double da = (double)pa[j] - p0[i][j], db = (double)pb[j] - p0[i][j];
            mx = fmax(mx, fabs(db)); err = fmax(err, fabs(db - da * ratio)); ss += db * db; }
        const double rel = err / mx, rmsr = sqrt(ss / nw) / lr / r;
        if (rel > worst) worst = rel;
        lo = fmin(lo, rmsr); hi = fmax(hi, rmsr);
        free(pa); free(pb); free(p0[i]);
    }
    printf("scale rms %s%s%s: step vs kj x ratio worst rel %.2e, element RMS / (lr r) in [%.4f, %.4f]\n", ext ? "ext" : "plain", nm ? "+normuon" : "", fro ? "+fro" : "", worst, lo, hi);
    CHECK(worst < 2e-3, "rms step is not the kj step rescaled: %.3e", worst);
    if (fro) CHECK(lo > 0.999 && hi < 1.001, "fro: element RMS not r");
    else CHECK(lo > 0.5 && hi < 1.25, "element RMS far from r");
    batch_free(&a); batch_free(&b);
}

int main(void) {
    nn_init(0);
    test_stats(0, 0);
    test_stats(1, 0);
    test_stats(1, 1);
    test_adapt(0, 1, 0.f, 0, 1.f);
    test_adapt(0, 1, 0.05f, 0, 1.f);
    test_adapt(0, 0, 0.f, 1, 1.f);    /* kappa 1: the distance overtakes rbar_0 within the 5 steps (the max branch) */
    test_adapt(1, 1, 0.f, 1, 19.f);   /* unet's kappa at beta 0.9 */
    test_adapt2(0);
    test_adapt2(1);
    test_scale(0, 0, 0);
    test_scale(1, 0, 0);
    test_scale(1, 1, 0);
    test_scale(1, 0, 1);
    if (fails) { printf("test_admuon: %d FAILED\n", fails); return 1; }
    printf("test_admuon: OK\n");
    return 0;
}
