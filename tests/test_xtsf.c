/* test_xtsf: the TimeSformer kernels of ink-ext (src/nn_xtsf.cu, xtsf.h) against host references written here from lucidrains
   TimeSformer-pytorch 0.4.1's definitions, in double:
   - xtsf_patches: 'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)' with min(u, clip) / div (exact up to the fp16 rounding);
   - xtsf_attn: divided space-time attention (dim_head 64, q x 64^-0.5; the cls query over all tokens without rotary; a patch token
     (f, p) over the cls token and its frame column (time) or frame row (space), q and those keys rotated: t cos + rotate_every_two(t)
     sin with random tables), 2 samples, 3 heads, 5 frames x 4 patches;
   - xtsf_geglu: x * gelu(g) (exact erf) of the two halves.
   Inputs are fp16-representable, so the bound is the fp16 output rounding plus fp32 accumulation. */
#include "xnet.h"
#include "xtsf.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float h2f(uint16_t h) {
    const int s = h >> 15, e = (h >> 10) & 31, m = h & 1023; float v;
    if (e == 0) v = ldexpf((float)m, -24); else if (e == 31) v = m ? NAN : INFINITY; else v = ldexpf((float)(m | 1024), e - 25);
    return s ? -v : v;
}
static uint16_t f2h(float f) {   /* round to nearest even, normal range only (the test values) */
    if (f == 0.f) return 0;
    const uint16_t s = f < 0 ? 0x8000 : 0; float a = fabsf(f); int e; frexpf(a, &e); e -= 1;
    if (e < -14) { const float q = nearbyintf(a / ldexpf(1.f, -24)); return (uint16_t)(s | (uint16_t)q); }
    const float m = nearbyintf((a / ldexpf(1.f, e) - 1.f) * 1024.f); int mm = (int)m, ee = e;
    if (mm == 1024) { mm = 0; ee++; }
    return (uint16_t)(s | (uint16_t)((ee + 15) << 10) | (uint16_t)mm);
}
static double urand(unsigned *s) { *s = *s * 1664525u + 1013904223u; return (*s >> 8) / 16777216.0; }
static int fails = 0;
static void report(const char *name, double maxerr, double bound) {
    printf("%-14s max |gpu - host| %.3g (bound %.3g)%s\n", name, maxerr, bound, maxerr <= bound ? "" : "  FAIL");
    if (!(maxerr <= bound)) fails++;
}

int main(void) {
    if (xnet_init(0)) { printf("SKIP: no GPU\n"); return 0; }
    unsigned rs = 12345u;
    /* ---- patches ---- */
    {
        const int B = 2, F = 3, S = 32, P = 16, G = S / P; const size_t n = (size_t)B * F * S * S;
        uint8_t *u = malloc(n); for (size_t i = 0; i < n; i++) u[i] = (uint8_t)(urand(&rs) * 256);
        uint8_t *d = xmalloc_dev(n); xh2d(d, u, n);
        xt y = xtsf_patches(d, B, F, S, P, 200.f, 200.f); xnet_sync();
        uint16_t *h = malloc(xt_numel(y) * 2); xd2h(h, y.p, xt_numel(y) * 2);
        double me = 0;
        for (int b = 0; b < B; b++) for (int f = 0; f < F; f++) for (int gy = 0; gy < G; gy++) for (int gx = 0; gx < G; gx++) for (int p1 = 0; p1 < P; p1++) for (int p2 = 0; p2 < P; p2++) {
            const size_t row = ((size_t)b * F + f) * G * G + gy * G + gx;
            const double ref = fmin(u[(((size_t)b * F + f) * S + gy * P + p1) * S + gx * P + p2], 200.0) / 200.0;
            me = fmax(me, fabs(h2f(h[row * P * P + p1 * P + p2]) - ref));
        }
        report("patches", me, 1.0 / 1024);
        free(u); free(h); xfree_dev(d); xt_free(&y);
    }
    /* ---- attention ---- */
    {
        const int B = 2, H = 3, F = 5, NP = 4, N = 1 + F * NP, I = H * 64;
        const size_t nq = (size_t)B * N * 3 * I;
        uint16_t *qh = malloc(nq * 2); float *q = malloc(nq * sizeof *q);
        for (size_t i = 0; i < nq; i++) { qh[i] = f2h((float)(urand(&rs) * 4 - 2)); q[i] = h2f(qh[i]); }
        float ts[5 * 64], tc[5 * 64], ss[4 * 64], sc[4 * 64];
        for (int i = 0; i < 5 * 64; i++) { const double a = urand(&rs) * 6.3; ts[i] = (float)sin(a); tc[i] = (float)cos(a); }
        for (int i = 0; i < 4 * 64; i++) { const double a = urand(&rs) * 6.3; ss[i] = (float)sin(a); sc[i] = (float)cos(a); }
        xt dq = xt_new(B, 1, 1, N, 3 * I); xh2d(dq.p, qh, nq * 2);
        float *dts = xdev_floats(ts, 5 * 64), *dtc = xdev_floats(tc, 5 * 64), *dss = xdev_floats(ss, 4 * 64), *dsc = xdev_floats(sc, 4 * 64);
        for (int time = 0; time < 2; time++) {
            xt o = xt_new(B, 1, 1, N, I); xtsf_attn(dq, o, H, F, NP, time, time ? dts : dss, time ? dtc : dsc); xnet_sync();
            uint16_t *oh = malloc(xt_numel(o) * 2); xd2h(oh, o.p, xt_numel(o) * 2);
            const float *rsn = time ? ts : ss, *rcs = time ? tc : sc;
            double me = 0, mx = 0;
            for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) for (int qi = 0; qi < N; qi++) {
                const float *base = q + (size_t)b * N * 3 * I;
                double qv[64]; int f = 0, p = 0;
                for (int d = 0; d < 64; d++) qv[d] = base[(size_t)qi * 3 * I + h * 64 + d] * 0.125;
                if (qi) {
                    f = (qi - 1) / NP; p = (qi - 1) % NP; const int r = time ? f : p; double t[64];
                    for (int d = 0; d < 64; d++) t[d] = qv[d] * rcs[r * 64 + d] + (d & 1 ? qv[d - 1] : -qv[d + 1]) * rsn[r * 64 + d];
                    memcpy(qv, t, sizeof t);
                }
                const int nk = qi ? 1 + (time ? F : NP) : N; double sc_[32], mxs = -1e300;
                int kts[32], rr[32];
                for (int j = 0; j < nk; j++) {
                    int kt, r = -1;
                    if (!qi) kt = j; else if (!j) kt = 0; else if (time) { kt = 1 + (j - 1) * NP + p; r = j - 1; } else { kt = 1 + f * NP + (j - 1); r = j - 1; }
                    kts[j] = kt; rr[j] = r;
                    double kv[64]; for (int d = 0; d < 64; d++) kv[d] = base[(size_t)kt * 3 * I + I + h * 64 + d];
                    if (r >= 0) { double t[64]; for (int d = 0; d < 64; d++) t[d] = kv[d] * rcs[r * 64 + d] + (d & 1 ? kv[d - 1] : -kv[d + 1]) * rsn[r * 64 + d]; memcpy(kv, t, sizeof t); }
                    double s = 0; for (int d = 0; d < 64; d++) s += qv[d] * kv[d];
                    sc_[j] = s; mxs = fmax(mxs, s);
                }
                double den = 0; for (int j = 0; j < nk; j++) den += exp(sc_[j] - mxs);
                for (int d = 0; d < 64; d++) {
                    double acc = 0; for (int j = 0; j < nk; j++) acc += exp(sc_[j] - mxs) / den * base[(size_t)kts[j] * 3 * I + 2 * I + h * 64 + d];
                    const double g = h2f(oh[((size_t)b * N + qi) * I + h * 64 + d]);
                    me = fmax(me, fabs(g - acc)); mx = fmax(mx, fabs(acc));
                }
                (void)rr;
            }
            report(time ? "attn time" : "attn space", me, 2e-3 * fmax(1, mx));
            free(oh); xt_free(&o);
        }
        free(qh); free(q); xt_free(&dq);
    }
    /* ---- GEGLU ---- */
    {
        const int M = 37, Fh = 48; const size_t n = (size_t)M * 2 * Fh;
        uint16_t *hh = malloc(n * 2); for (size_t i = 0; i < n; i++) hh[i] = f2h((float)(urand(&rs) * 8 - 4));
        xt h = xt_new(1, 1, 1, M, 2 * Fh); xh2d(h.p, hh, n * 2);
        xt y = xtsf_geglu(h); xnet_sync();
        uint16_t *yh = malloc((size_t)M * Fh * 2); xd2h(yh, y.p, (size_t)M * Fh * 2);
        double me = 0;
        for (int r = 0; r < M; r++) for (int c = 0; c < Fh; c++) {
            const double x = h2f(hh[(size_t)r * 2 * Fh + c]), g = h2f(hh[(size_t)r * 2 * Fh + Fh + c]);
            const double ref = x * 0.5 * g * (1 + erf(g / sqrt(2.0)));
            me = fmax(me, fabs(h2f(yh[(size_t)r * Fh + c]) - ref) / fmax(1, fabs(ref)));
        }
        report("geglu", me, 2e-3);
        free(hh); free(yh); xt_free(&h); xt_free(&y);
    }
    if (xnet_check("test_xtsf")) fails++;
    printf("test_xtsf: %s\n", fails ? "FAIL" : "OK");
    return fails ? 1 : 0;
}
