/* A mixed packed stem must match the same FP8 arithmetic on independently
   decoded inputs, followed by the declared output converter. */
#include "nn.h"
#include "nn_lp.h"
#include <math.h>
#include <stdio.h>

static int bad;
static float *values(size_t n, float scale) {
    float *h = malloc(n * 4), *d = nn_malloc(n * 4);
    for (size_t i = 0; i < n; i++) h[i] = sinf((float)i * 0.173f) * scale;
    nn_h2d(d, h, n * 4); free(h); return d;
}
static void *packed(shape5 s, int bits) {
    size_t n = nn_mx_bytes(s, bits); void *p = nn_malloc(n); nn_set_storage(p, n, bits); return p;
}
static float *decode(const void *p, shape5 s, int bits) {
    float *d = nn_malloc(shape_numel(s) * 4);
    if (bits == 4) lp_mx4_to_f32(p, s.n, s.c, shape_spatial(s), d);
    else lp_mx8_to_f32(p, s.n, s.c, shape_spatial(s), d);
    return d;
}
static void compare(const char *what, const float *a, const float *b, size_t n, double tol) {
    float *ha = malloc(n * 4), *hb = malloc(n * 4); nn_d2h(ha, a, n * 4); nn_d2h(hb, b, n * 4);
    double error = 0, scale = 1e-8; int finite = 1;
    for (size_t i = 0; i < n; i++) { finite &= isfinite(ha[i]) && isfinite(hb[i]); error = fmax(error, fabs((double)ha[i] - hb[i])); scale = fmax(scale, fabs(hb[i])); }
    int ok = finite && error / scale < tol;
    printf("  %s: relative max %.3g %s\n", what, error / scale, ok ? "ok" : "FAIL"); bad += !ok; free(ha); free(hb);
}
static void run(int ci, int co, int input, int output) {
    shape5 xs = {2,ci,5,7,9}, ys = xs; ys.c = co;
    size_t nw = (size_t)ci * co * 27, ny = shape_numel(ys);
    float *xf = values(shape_numel(xs), 1.f), *w = values(nw, .02f), *b = values(co, .1f), *yf = nn_malloc(ny * 4);
    void *x = packed(xs, input), *y = packed(ys, output), *expected = packed(ys, output);
    nn_f32_to_act(xf, xs, x); float *decoded = decode(x, xs, input);
    nn_conv3d_fwd(x, xs, w, b, co, 3, 1, y);
    lp_conv_fwd_f8(decoded, 0, xs, w, b, co, yf, 0, (gnp_t){0}, nullptr, 0, (split_t){0});
    nn_f32_to_act(yf, ys, expected); float *got = decode(y, ys, output), *want = decode(expected, ys, output);
    printf("stem ci%d co%d input%d output%d\n", ci, co, input, output);
    compare("packed forward vs decoded FP8 + converter", got, want, ny, 2e-5);
    float *gyf = values(ny, .01f); void *gy = packed(ys, 8); nn_f32_to_act(gyf, ys, gy); float *gyd = decode(gy, ys, 8);
    float *gw = nn_malloc(nw * 4), *gwr = nn_malloc(nw * 4), *gb = nn_malloc(co * 4), *gbr = nn_malloc(co * 4);
    nn_zero(gw, nw * 4); nn_zero(gwr, nw * 4); nn_zero(gb, co * 4); nn_zero(gbr, co * 4);
    nn_conv3d_bwd_weight(x, xs, gy, ys, 3, 1, gw, gb);
    lp_bwd_w_f8(decoded, 0, xs, gyd, 0, ys, gwr, gbr, (gnp_t){0}, (split_t){0});
    compare("stem weight gradient vs decoded operands", gw, gwr, nw, 2e-4);
    compare("stem bias gradient vs decoded operands", gb, gbr, co, 2e-4);
    nn_free(xf); nn_free(w); nn_free(b); nn_free(yf); nn_free(x); nn_free(y); nn_free(expected); nn_free(decoded); nn_free(got); nn_free(want);
    nn_free(gyf); nn_free(gy); nn_free(gyd); nn_free(gw); nn_free(gwr); nn_free(gb); nn_free(gbr);
}
int main(void) {
    if (nn_init(0)) return 1;
    nn_set_prec(2); nn_set_prec_policy(""); nn_set_f16(1); nn_set_sr(0); nn_set_grad_bf16(0);
    for (int ci = 4; ci <= 8; ci += 4) for (int co = 16; co <= 32; co += 16) { run(ci, co, 8, 4); run(ci, co, 4, 8); }
    nn_sync(); const char *e = nn_check(); if (e) { fprintf(stderr, "%s\n", e); bad++; }
    printf("mixed stem precision: %s\n", bad ? "FAIL" : "ok"); return bad != 0;
}
