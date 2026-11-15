/* the head-gradient plane scale chosen on the device (hgs.h, nn_hgs_choose; 2026-11-14 hgsdev): the kernel's {s, 1 / s} against the host
   function (hgs.h in host C, -ffp-contract=off: train.c's UFSM_HGS_CHECK reference) bit for bit, on
   (1) random windows: 1 .. 8 loss scratches of 1 / 2 / 7 channels (mask sums with some below 1, fin scales over 2^-40 .. 2^10, signed
       lambdas, dice / recall terms), the wind4 fin's 1 / N, the MIL term, gradient scales 2^0 .. 2^20;
   (2) edge bounds: b = 512 / 2^k exactly and one ulp either side (k 0 .. 30), b 0 / negative / inf / NaN / subnormal / 1e30 (s 1 or 2^30);
   (the producers' and consumers' device-pointer paths: test_hgscale, whose nn_set_head_gscale goes through the same {s, 1 / s} buffer).
   Also counts the windows where the old host formula floorf(log2f(512 / b)) differed (expected: none or a few at an ulp below 2^k).
   Fast math (the default build, DESIGN 2026-11-14 "Fast math everywhere"): nn_hgs.cu is no longer IEEE (div.approx, FMA, flush) and
   the host contracts, so the bound and 512 / b can land an ulp apart: the device's k may be one off the host's where the bound sits
   within ulps of 512 / 2^k (the random windows: on at most 1e-3 of them; the edge cases, which sit there by construction: any), and
   FTZ / DAZ (host src/fpenv.c, device flush) read the subnormal bound 1e-40 as 0 (k 0, not 30). Exact under UFSM_STRICT_MATH.
   Usage: test_hgsdev [windows=4000] */
#include "nn.h"
#include "hgs.h"
#include "fm_tol.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int fails;
static void check(const char *what, int ok) { printf("  %-74s %s\n", what, ok ? "ok" : "FAIL"); if (!ok) fails++; }
static uint64_t rs = 88172645463325252ull;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static float fr(void) { return (float)((rnd() >> 40) / 16777216.0); }   /* [0, 1) */
static int old_exp(float b) {   /* the host formula before hgsdev */
    if (!(b > 0.f) || !isfinite(b)) return 0;
    int k = (int)floorf(log2f(512.f / b)); return k < 0 ? 0 : k > 30 ? 30 : k;
}
#define SCN (5 * 7 + 2 * 7 + 2)
/* one window: host arrays for every term, their device copies, the device choice vs the host's */
static int near_k;   /* (fast math) windows whose device k is one off the host's */
static int run_case(hgs_args *h, float (*hb)[SCN], float *hw4, float **db, float *dw4, float *dout, int *oldiff) {
    hgs_args d = *h;
    for (int i = 0; i < h->nt; i++) { nn_h2d(db[i], hb[i], SCN * 4); d.t[i].p = db[i]; h->t[i].p = hb[i]; }
    if (h->w4) { nn_h2d(dw4, hw4, 8 * 4); d.w4 = dw4; h->w4 = hw4; }
    nn_hgs_choose(&d, dout); nn_sync(); float dv[2]; nn_d2h(dv, dout, sizeof dv);
    const float b = hgs_bound_all(h); const int k = hgs_exp(b); const float s = ldexpf(1.f, k), si = ldexpf(1.f, -k);
    if (old_exp(b) != k) (*oldiff)++;
    int ok = !memcmp(&s, &dv[0], 4) && !memcmp(&si, &dv[1], 4);
    if (!ok && !FM_STRICT && dv[0] > 0.f && dv[0] * dv[1] == 1.f) {   /* {2^kd, 2^-kd} with kd = k +- 1: the ulp edge under fast math */
        int kd; (void)frexpf(dv[0], &kd); kd -= 1;
        if (abs(kd - k) == 1 && ldexpf(1.f, kd) == dv[0]) { near_k++; ok = 2; }
    }
    if (ok != 1) fprintf(stderr, "  mismatch: bound %a host {%a, %a} device {%a, %a}\n", b, s, si, dv[0], dv[1]);
    return ok;
}
int main(int argc, char **argv) {
    const int nwin = argc > 1 ? atoi(argv[1]) : 4000;
    nn_init(0);
    float hb[HGS_NT][SCN], hw4[8]; float *db[HGS_NT], *dw4 = nn_malloc(8 * 4), *dout = nn_malloc(2 * 4);
    for (int i = 0; i < HGS_NT; i++) db[i] = nn_malloc(SCN * 4);
    printf("(1) %d random windows\n", nwin);
    int bad = 0, oldd = 0; int kh[31] = {0};
    for (int w = 0; w < nwin; w++) {
        hgs_args a = {0}; static const int Cs[3] = {1, 2, 7};
        a.nt = 1 + (int)(rnd() % HGS_NT); a.gs = ldexpf(1.f, (int)(rnd() % 21)); a.rw = (rnd() & 1) ? fr() : 0.f; a.mw = 2.f * (1.f + (float)(rnd() % 8));
        for (int i = 0; i < a.nt; i++) {
            const int C = Cs[rnd() % 3];
            a.t[i] = (hgs_term){.C = C, .lam = (fr() - 0.3f) * 4.f, .dice = (rnd() & 3) ? 0.f : fr(), .pw = (rnd() & 1) ? 1.f + 4.f * fr() : 1.f};
            for (int j = 0; j < SCN; j++) hb[i][j] = fr();
            for (int c = 0; c < C; c++) hb[i][5 * c] = (rnd() % 7 == 0) ? fr() : ldexpf(1.f + fr(), (int)(rnd() % 28));   /* the mask sums (some < 1: skipped) */
            hb[i][5 * C + 2 * C + 1] = ldexpf(1.f + fr(), -40 + (int)(rnd() % 51));   /* fin's scale */
        }
        a.w4 = (rnd() % 4) ? hw4 : nullptr; a.w4lam = (fr() - 0.5f) * 2.f;
        for (int j = 0; j < 8; j++) hw4[j] = fr(); hw4[3] = ldexpf(1.f + fr(), -(int)(rnd() % 30));
        a.milv = (rnd() % 3) ? 0.f : ldexpf(1.f + fr(), -20 + (int)(rnd() % 25));
        if (!run_case(&a, hb, hw4, db, dw4, dout, &oldd)) bad++;
        kh[hgs_exp(hgs_bound_all(&a))]++;
    }
    printf("  s exponents seen:"); for (int k = 0; k <= 30; k++) if (kh[k]) printf(" 2^%d:%d", k, kh[k]); printf("\n");
    check("device {s, 1 / s} == host, every window (fast math: k +- 1 on <= 1e-3)", !bad && (size_t)near_k <= fm_allow((size_t)nwin, 1e-3));
    if (near_k) printf("  (device k one off the host's in %d windows: the ulp edge under fast math)\n", near_k);
    printf("  (the old floorf(log2f()) formula differed in %d of %d windows)\n", oldd, nwin);

    printf("(2) edge bounds\n");
    bad = 0; int n2 = 0;
    for (int j = 0; j < SCN; j++) hb[0][j] = 1.f;
    for (int k = 0; k <= 31; k++) for (int u = -1; u <= 1; u++) {   /* one term, one channel, every factor 1 but lambda: the bound is lambda */
        float b = ldexpf(512.f, -k); uint32_t bits; memcpy(&bits, &b, 4); bits += (uint32_t)u; memcpy(&b, &bits, 4);
        hgs_args a = {.nt = 1, .gs = 1.f, .rw = 0.f, .mw = 1.f};
        a.t[0] = (hgs_term){.p = hb[0], .C = 1, .lam = b, .dice = 0.f, .pw = 1.f};
        const float e = hgs_bound_all(&a);
        if (memcmp(&e, &b, 4)) { fprintf(stderr, "  edge setup: bound %a != %a\n", e, b); bad++; }
        bad += !run_case(&a, hb, hw4, db, dw4, dout, &oldd); n2++;
        int want = u > 0 ? k - 1 : k; want = want < 0 ? 0 : want > 30 ? 30 : want;   /* (b one ulp above 512 / 2^k: 512 / b below 2^k) */
        if (hgs_exp(b) != want) { fprintf(stderr, "  edge k %d ulp %+d: exponent %d, want %d\n", k, u, hgs_exp(b), want); bad++; }
    }
    const float specials[] = {0.f, -1.f, INFINITY, NAN, 1e-40f, 1e30f, 1e-30f};
    const int swant[] = {0, 0, 0, 0, FM_STRICT ? 30 : 0, 0, 30};   /* (1e-40: subnormal, read as 0 under FTZ / DAZ) */
    for (int i = 0; i < (int)(sizeof specials / sizeof *specials); i++) {
        hgs_args a = {.nt = 0, .gs = 1.f, .mw = 1.f, .milv = specials[i]};
        bad += !run_case(&a, hb, hw4, db, dw4, dout, &oldd); n2++;
        if (hgs_exp(hgs_bound_all(&a)) != swant[i]) { fprintf(stderr, "  special %g: exponent %d, want %d\n", specials[i], hgs_exp(hgs_bound_all(&a)), swant[i]); bad++; }
    }
    check("device == host and the exponent rule at 512 / 2^k +- 1 ulp, specials", !bad);
    printf("  (%d edge cases)\n", n2);
    printf(fails ? "test_hgsdev: %d FAILED\n" : "test_hgsdev: ok\n", fails);
    return fails ? 1 : 0;
}
