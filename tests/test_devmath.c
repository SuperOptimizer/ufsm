/* the device library's trigonometry by pi (src/nn_devmath.cu) against a long double host reference: sincospi / sinpi / cospi / sincos
   in double, sincospif in float, on special values, random arguments and the FFT twiddles 2 m / P of nn_aug.cu's four_tab_k (P = 64,
   640, 768, 1024). Double: at most 2 ulp (CUDA's documented bound for sincospi / sincos; mcc's are within 1), exact on multiples of
   1/4 (sign of zero included); sincospi equal to sinpi / cospi bit for bit; float: at most 2 ulp. Until mcc e0e23d0 the launch itself
   failed (no double sinpi / cospi in the device library), and the error was only latched, so the caller went on with garbage: first, in a
   child (before this process touches CUDA), a kernel calling a device function nothing defines must stop the program at its launch. */
#include "nn.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static const long double PI_L = 3.14159265358979323846264338327950288L;

/* sin(pi x), cos(pi x) in long double: the exact reduction to f in [-1/4, 1/4] (a - 2 floor(a / 2), then the nearest half), sinl / cosl */
static void ref_scpi(long double x, long double *sp, long double *cp) {
    long double a = fabsl(x), r = a - 2.0L * floorl(a * 0.5L), f, s, c, t;
    const int n = (int)floorl(2.0L * r + 0.5L);
    f = r - 0.5L * n;
    s = f == 0 ? 0 : sinl(PI_L * f); c = cosl(PI_L * f);
    switch (n & 3) {
    case 1: t = s; s = c; c = f == 0 ? 0 : -t; break;
    case 2: s = f == 0 ? 0 : -s; c = -c; break;
    case 3: t = s; s = -c; c = f == 0 ? 0 : t; break;
    }
    *sp = signbit(x) ? -s : s; *cp = c;
}

static int64_t key64(double d) { int64_t i; memcpy(&i, &d, 8); return i < 0 ? INT64_MIN - i : i; }
static int32_t key32(float f) { int32_t i; memcpy(&i, &f, 4); return i < 0 ? INT32_MIN - i : i; }
static int64_t ulp64(double a, double b) { const int64_t d = key64(a) - key64(b); return d < 0 ? -d : d; }
static int64_t ulp32(float a, float b) { const int64_t d = (int64_t)key32(a) - key32(b); return d < 0 ? -d : d; }

int main(void) {
    enum { NMAX = 40000 };
    static double x[NMAX], o[NMAX * 8];
    static const double sp[] = {0.0, -0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, -0.25, -0.5, -1.0, -1.5, -2.0, 3.0, -3.0, 1e-300,
        -1e-300, 1e-10, 0.1, 1.0 / 3.0, 0.4999999999999999, 0.5000000000000001, 123456.75, 0x1p51 + 0.5, 0x1p52 + 1.0, 0x1p53, 1e300};
    int n = 0, nsp = (int)(sizeof sp / sizeof sp[0]);
    for (int i = 0; i < nsp; i++) x[n++] = sp[i];
    static const int Ps[] = {64, 640, 768, 1024};
    for (int p = 0; p < 4; p++) for (int m = 0; m < Ps[p]; m++) x[n++] = 2.0 * (double)m / Ps[p];   /* four_tab_k's arguments */
    uint64_t st = 0x9e3779b97f4a7c15ULL;
    while (n < NMAX) {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        const double u = (double)(st >> 11) * 0x1p-53, sg = st & 1 ? -1.0 : 1.0;
        x[n++] = (st & 6) == 0 ? sg * u * 0x1p20 : (st & 6) == 2 ? sg * u * 0x1p-20 : sg * u * 4.0;
    }
    int undef_ok = 1;
    { const pid_t pid = fork();
      if (pid == 0) { const int r = nn_devmath_undef(); _exit(r == -2 ? 77 : r == 0 ? 0 : 3); }
      int st = 0; waitpid(pid, &st, 0);
      if (WIFEXITED(st) && WEXITSTATUS(st) == 77) printf("undefined device function: not built here (nvcc rejects it at build time)\n");
      else if (WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT) printf("undefined device function: the launch stopped the program (ok)\n");
      else { undef_ok = 0; printf("undefined device function: FAIL: the child %s %d instead of stopping at the launch\n", WIFEXITED(st) ? "exited" : "got signal", WIFEXITED(st) ? WEXITSTATUS(st) : WTERMSIG(st)); } }
    const int rc = nn_devmath_probe(x, n, o);
    if (rc) { printf("test_devmath: FAIL: the kernel did not run (CUDA error %d; before mcc e0e23d0: no double sinpi / cospi on the device)\n", rc); return 1; }
    int64_t w_pi = 0, w_sc = 0, w_f = 0; int bad = 0, incons = 0, nsc = 0;
    for (int i = 0; i < n; i++) {
        const double *r = o + (size_t)i * 8;
        long double rs, rc2; ref_scpi(x[i], &rs, &rc2);
        const double es = (double)rs, ec = (double)rc2;
        const int64_t ds = ulp64(r[0], es), dc = ulp64(r[1], ec);
        if (ds > w_pi) w_pi = ds;
        if (dc > w_pi) w_pi = dc;
        const int quarter = x[i] * 4.0 == floor(x[i] * 4.0) && fabs(x[i]) < 0x1p50;
        const int eq_special = !quarter || (memcmp(&r[0], &es, 8) == 0 && memcmp(&r[1], &ec, 8) == 0)
                               || (fabs(es) == M_SQRT1_2 && ds <= 1 && dc <= 1);   /* (odd quarters: +-sqrt(1/2), not exact) */
        if (memcmp(&r[0], &r[2], 8) || memcmp(&r[1], &r[3], 8)) incons++;
        if (ds > 2 || dc > 2 || !eq_special) { if (bad++ < 8) printf("  x %a: sincospi %a %a, want %a %a (%lld / %lld ulp)\n", x[i], r[0], r[1], es, ec, (long long)ds, (long long)dc); }
        if (fabs(x[i]) < 1e5) {   /* sincos(x) */
            nsc++;
            const int64_t a = ulp64(r[4], (double)sinl((long double)x[i])), b = ulp64(r[5], (double)cosl((long double)x[i]));
            if (a > w_sc) w_sc = a;
            if (b > w_sc) w_sc = b;
            if (a > 2 || b > 2) { if (bad++ < 8) printf("  x %a: sincos %a %a (%lld / %lld ulp)\n", x[i], r[4], r[5], (long long)a, (long long)b); }
        }
        if (fabs(x[i]) < 0x1p20) {   /* sincospif((float)x) */
            long double fs, fc; ref_scpi((long double)(float)x[i], &fs, &fc);
            const int64_t a = ulp32((float)r[6], (float)fs), b = ulp32((float)r[7], (float)fc);
            if (a > w_f) w_f = a;
            if (b > w_f) w_f = b;
            if (a > 2 || b > 2) { if (bad++ < 8) printf("  x %a: sincospif %a %a (%lld / %lld ulp)\n", (double)(float)x[i], r[6], r[7], (long long)a, (long long)b); }
        }
    }
    printf("test_devmath: %d arguments: sincospi worst %lld ulp, sincos worst %lld ulp (%d args), sincospif worst %lld ulp; %d sincospi != sinpi / cospi\n",
           n, (long long)w_pi, (long long)w_sc, nsc, (long long)w_f, incons);
    const int fail = bad || incons || !undef_ok;
    printf(fail ? "test_devmath: FAIL (%d bad)\n" : "test_devmath: ok\n", bad);
    return fail;
}
