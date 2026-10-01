/* Fused tensor-core paths vs. unfused fp32 reference: gn+silu applied while staging (forward conv and weight
   gradient), channel-split inputs, bf16 activation storage. The finite-difference tests only cover the fp32 kernels. */
#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static float *up(size_t n, float lo, float hi) { float *h = malloc(n * 4); for (size_t i = 0; i < n; i++) h[i] = lo + (hi - lo) * (float)rand() / RAND_MAX; float *d = nn_malloc(n * 4); nn_h2d(d, h, n * 4); free(h); return d; }
static double reldiff(const float *a, const float *b, size_t n) {   /* max |a-b| / max |b| */
    float *ha = malloc(n * 4), *hb = malloc(n * 4); nn_d2h(ha, a, n * 4); nn_d2h(hb, b, n * 4);
    double m = 0, mx = 0; for (size_t i = 0; i < n; i++) { double d = fabs(ha[i] - hb[i]); if (d > m) m = d; if (fabs(hb[i]) > mx) mx = fabs(hb[i]); }
    free(ha); free(hb); return mx > 0 ? m / mx : m;
}
static int fails = 0;
static void check(const char *name, double r, double tol) { printf("  %-44s rel err %.3g %s\n", name, r, r <= tol ? "" : "FAIL"); if (r > tol) fails++; }
static void run(int C, int P, int Co, int split) {
    int G = 8;
    shape5 xs = {2, C, P, P, P}, ys = xs; ys.c = Co;
    size_t n = shape_numel(xs), ny = shape_numel(ys), S = (size_t)P * P * P;
    float *x = up(n, -2, 2), *w = up((size_t)Co * C * 27, -0.2f, 0.2f), *b = up(Co, -0.1f, 0.1f), *gy = up(ny, -1, 1);
    float *gamma = up(C, 0.5f, 1.5f), *beta = up(C, -0.5f, 0.5f), *mean = up(2 * G, -0.3f, 0.3f), *rstd = up(2 * G, 0.8f, 1.2f);
    float *g = nn_malloc(n * 4), *yref = nn_malloc(ny * 4), *y = nn_malloc(ny * 4), *om = nn_malloc(2 * G * 4), *orr = nn_malloc(2 * G * 4);
    float *gwref = nn_malloc((size_t)Co * C * 27 * 4), *gw = nn_malloc((size_t)Co * C * 27 * 4), *gbref = nn_malloc(Co * 4), *gb = nn_malloc(Co * 4);
    printf("C %d P %d Co %d%s\n", C, P, Co, split ? " (split input)" : "");
    nn_set_tf32(0); nn_set_act_bf16(0); nn_set_grad_bf16(0);
    nn_gn_silu_apply(x, xs, G, gamma, beta, mean, rstd, g);
    nn_conv3d_fwd(g, xs, w, b, Co, 3, 1, yref);
    nn_zero(gwref, (size_t)Co * C * 27 * 4); nn_zero(gbref, Co * 4);
    nn_conv3d_bwd_weight(g, xs, gy, ys, 3, 1, gwref, gbref);
    nn_set_tf32(1);
    for (int abf = 0; abf < 2; abf++) {
        nn_set_act_bf16(abf);
        const float *xin = x; void *xb = nullptr;
        if (abf) { xb = nn_malloc(n * 2); nn_f32_to_bf16(x, n, xb); xin = xb; }
        char name[96];
        /* channel split: channels >= c_split come from a second tensor */
        int cs = split ? C / 2 : 0;
        float *x2 = nullptr; const float *xa = xin;
        if (split) {   /* xa = first cs channels of each sample, x2 = the rest (both in the activation storage format) */
            size_t es = abf ? 2 : 4;
            xa = nn_malloc((size_t)2 * cs * S * es); x2 = nn_malloc((size_t)2 * (C - cs) * S * es);
            for (int s = 0; s < 2; s++) {
                nn_d2d((char *)xa + (size_t)s * cs * S * es, (const char *)xin + (size_t)s * C * S * es, (size_t)cs * S * es);
                nn_d2d((char *)x2 + (size_t)s * (C - cs) * S * es, (const char *)xin + ((size_t)s * C + cs) * S * es, (size_t)(C - cs) * S * es);
            }
        }
        float *yo = y; void *yb = nullptr;
        if (abf) { yb = nn_malloc(ny * 2); yo = yb; }
        if (split) nn_conv3d_fwd_split(xa, x2, cs, xs, G, gamma, beta, mean, rstd, w, b, Co, yo, G, 1e-5f, om, orr);
        else nn_conv3d_fwd_gn_stats(xin, xs, G, gamma, beta, mean, rstd, w, b, Co, yo, G, 1e-5f, om, orr);
        if (abf) {   /* compare in fp32: convert the bf16 output back */
            float *hy = malloc(ny * 4); uint16_t *hb = malloc(ny * 2); nn_d2h(hb, yb, ny * 2);
            for (size_t i = 0; i < ny; i++) {
                uint32_t u; if (nn_get_f16()) { uint32_t h = hb[i], sg = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;   /* fp16 -> fp32 */
                    if (e == 0) { if (m == 0) u = sg << 31; else { e = 1; while (!(m & 1024)) { m <<= 1; e--; } m &= 1023; u = (sg << 31) | ((uint32_t)(e + 112) << 23) | (m << 13); } }
                    else if (e == 31) u = (sg << 31) | 0x7f800000u | (m << 13); else u = (sg << 31) | ((uint32_t)(e + 112) << 23) | (m << 13); }
                else u = (uint32_t)hb[i] << 16;
                float f; __builtin_memcpy(&f, &u, 4); hy[i] = f; }
            nn_h2d(y, hy, ny * 4); free(hy); free(hb);
        }
        snprintf(name, sizeof name, "fwd fused gn+silu%s", abf ? " (bf16 act)" : ""); check(name, reldiff(y, yref, ny), abf ? 0.02 : 0.01);
        nn_zero(gw, (size_t)Co * C * 27 * 4); nn_zero(gb, Co * 4);
        if (split) nn_conv3d_bwd_weight_split(xa, x2, cs, xs, G, gamma, beta, mean, rstd, gy, ys, gw, gb);
        else nn_conv3d_bwd_weight_gn(xin, xs, G, gamma, beta, mean, rstd, gy, ys, gw, gb);
        snprintf(name, sizeof name, "bwd_weight fused gn+silu%s", abf ? " (bf16 act)" : ""); check(name, reldiff(gw, gwref, (size_t)Co * C * 27), 0.02);
        snprintf(name, sizeof name, "bias grad%s", abf ? " (bf16 act)" : ""); check(name, reldiff(gb, gbref, Co), 0.02);
        if (split) { nn_free((void *)xa); nn_free(x2); }
        nn_free(xb); nn_free(yb);
    }
    nn_set_act_bf16(1);
    nn_free(x); nn_free(w); nn_free(b); nn_free(gy); nn_free(gamma); nn_free(beta); nn_free(mean); nn_free(rstd); nn_free(g); nn_free(yref); nn_free(y); nn_free(om); nn_free(orr); nn_free(gwref); nn_free(gw); nn_free(gbref); nn_free(gb);
}
int main(void) {
    if (nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0)) { printf("no cuda\n"); return 1; }
    if (getenv("UFSM_PREC")) nn_set_prec(atoi(getenv("UFSM_PREC")));   /* 2: fp8 kernels (expected rel err ~4e-2, reported as FAIL against the bf16 tolerances) */
    if (getenv("UFSM_F16")) nn_set_f16(1);
    run(16, 32, 16, 0);
    run(48, 12, 16, 1);
    run(32, 24, 32, 0);
    run(4, 20, 16, 0);
    const char *e = nn_check(); if (e) { printf("cuda: %s\n", e); return 1; }
    printf(fails ? "fused FAIL (%d)\n" : "fused ok\n", fails);
    return fails != 0;
}
