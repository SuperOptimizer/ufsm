/* nn_sumsq (src/nn_opt.cu reduce: per-thread double sums, a double tree per block, one float per block, the 1024 block values summed
   in double on the host) and nn_ema against host references (audit 2026-11-14 #18b). nn_sumsq drives train's gradient clipping and
   its skip: gn = sqrt(unet_grad_norm's nn_sumsq); !isfinite(gn) skips the step and halves the 16-bit gradient scale (train.c, the
   optimizer loop), so a single non-finite gradient element must make the sum non-finite. Checks:
   - 100 M + 7 floats (~381 a thread over 1024 x 256 threads): the sum within 1e-6 relative of the host's double sum;
   - one +inf (element near the end): +inf; one -inf: +inf (squared); one NaN: NaN; an inf and a NaN: NaN; restored: finite again;
   - nn_sum (the same reduce) on the finite data against the host;
   - nn_ema e = d e + (1 - d) p over 10 M + 3 floats at d 0.999 and 0.9: within 2 float ulps (of the larger term) of the double formula; d 0: e = p exactly
     (the EMA's first step), also where e was inf (0 x inf = NaN would leak: reported, the trainer's EMA is never non-finite before it). */
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)
static uint64_t rs = 0x853c49e6748fea9bull;
static float frand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 40) / 16777216.0 - 0.5); }
int main(void) {
    if (nn_init(0)) { printf("SKIP: no GPU\n"); return 0; }
    {
        const size_t n = 100000007; float *h = malloc(n * 4); double ref = 0, rsum = 0;
        for (size_t i = 0; i < n; i++) { h[i] = frand() * (1.f + (float)(i % 13)) * 1e-2f; ref += (double)h[i] * h[i]; rsum += h[i]; }
        float *d = nn_malloc(n * 4), *scr = nn_malloc(4096 * 4); nn_h2d(d, h, n * 4);
        const double s0 = nn_sumsq(d, n, scr), e0 = fabs(s0 - ref) / ref, t0 = nn_sum(d, n, scr), et = fabs(t0 - rsum) / (fabs(rsum) + 1e-3 * sqrt(ref));
        printf("nn_sumsq, %zu floats: %.10g vs host %.10g (rel %.2e); nn_sum %.6g vs %.6g (rel %.2e)\n", n, s0, ref, e0, t0, rsum, et);
        CHECK(e0 < 1e-6, "nn_sumsq: rel error %.3g", e0);
        CHECK(et < 1e-5, "nn_sum: rel error %.3g", et);
        const size_t ki = n - 5, kn = n / 3 + 1;   /* (the inf in the last block's tail, the NaN in the middle) */
        const float pinf = INFINITY, ninf = -INFINITY, qnan = NAN;
        struct { const char *what; float vi, vn; int want; } cs[5] = {{"+inf", pinf, h[kn], 1}, {"-inf", ninf, h[kn], 1}, {"NaN", h[ki], qnan, 2}, {"inf and NaN", pinf, qnan, 2}, {"restored", h[ki], h[kn], 0}};
        for (int c = 0; c < 5; c++) {
            nn_h2d(d + ki, &cs[c].vi, 4); nn_h2d(d + kn, &cs[c].vn, 4);
            const double v = nn_sumsq(d, n, scr); const int got = isnan(v) ? 2 : isinf(v) && v > 0 ? 1 : isfinite(v) ? 0 : 3;
            printf("nn_sumsq with %-11s: %g (%s)\n", cs[c].what, v, got == cs[c].want ? "as train expects" : "WRONG");
            CHECK(got == cs[c].want && (c < 4 || fabs(v - ref) / ref < 1e-6), "nn_sumsq with %s: %g", cs[c].what, v);
            CHECK(c == 4 || !isfinite(sqrt(v)), "nn_sumsq with %s: the gradient norm sqrt %g is finite (train would not skip the step)", cs[c].what, sqrt(v));
        }
        nn_free(d); nn_free(scr); free(h);
    }
    {
        const size_t n = 10000003; float *e = malloc(n * 4), *p = malloc(n * 4), *o = malloc(n * 4); size_t bad = 0;
        for (size_t i = 0; i < n; i++) { e[i] = frand(); p[i] = frand() * 3.f; }
        float *de = nn_malloc(n * 4), *dp = nn_malloc(n * 4); nn_h2d(dp, p, n * 4);
        static const float dk[2] = {0.999f, 0.9f};
        for (int k = 0; k < 2; k++) {
            nn_h2d(de, e, n * 4); nn_ema(de, dp, n, dk[k]); nn_sync(); nn_d2h(o, de, n * 4); bad = 0;
            for (size_t i = 0; i < n; i++) {
                const double a = (double)dk[k] * e[i], b = (1.0 - (double)dk[k]) * p[i], r = a + b, u = fmax(fmax(fabs(a), fabs(b)), fabs(r)) * 0x1p-23;   /* (ulps of the larger term: cancellation) */
                bad += !(fabs(o[i] - r) <= 2 * u + 1e-12);
            }
            printf("nn_ema d %.3f, %zu floats: %zu outside 2 ulps of the double formula\n", dk[k], n, bad);
            CHECK(bad == 0, "nn_ema d %.3f: %zu off", dk[k], bad);
        }
        e[17] = INFINITY; nn_h2d(de, e, n * 4); nn_ema(de, dp, n, 0.f); nn_sync(); nn_d2h(o, de, n * 4);
        bad = 0; for (size_t i = 0; i < n; i++) if (i != 17) bad += memcmp(o + i, p + i, 4) != 0;
        printf("nn_ema d 0: %zu of %zu differ from p (element over an inf EMA: %g)\n", bad, n - 1, o[17]);
        CHECK(bad == 0, "nn_ema d 0: %zu differ from p", bad);
        nn_free(de); nn_free(dp); free(e); free(p); free(o);
    }
    const char *er = nn_check(); CHECK(!er, "cuda %s", er ? er : "");
    if (fails) { fprintf(stderr, "test_sumsq: %d failure(s)\n", fails); return 1; }
    printf("test_sumsq: ok\n"); return 0;
}
