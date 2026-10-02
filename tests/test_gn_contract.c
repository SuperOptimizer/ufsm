/* Independent FP64 statistics of the actual stored convolution tensor, including
   constant groups, packed outputs, split inputs and finite FP16 replay. */
#include "nn.h"
#include "nn_lp.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static int fails;
static void *activation(shape5 s, int bits) {
    size_t bytes = bits == 4 || bits == 8 ? nn_mx_bytes(s, bits) : shape_numel(s) * (bits ? 2 : 4);
    void *p = nn_malloc(bytes); nn_zero(p, bytes);
    if (bits == 4 || bits == 8) nn_set_storage(p, bytes, bits);
    return p;
}
static void decode(void *p, shape5 s, int bits, float *h) {
    size_t n = shape_numel(s);
    if (bits == 4 || bits == 8) {
        float *f = nn_malloc(n * 4);
        if (bits == 4) lp_mx4_to_f32(p, s.n, s.c, shape_spatial(s), f);
        else lp_mx8_to_f32(p, s.n, s.c, shape_spatial(s), f);
        nn_d2h(h, f, n * 4); nn_free(f);
    } else if (bits) {
        uint16_t *raw = malloc(n * 2); nn_d2h(raw, p, n * 2);
        for (size_t i = 0; i < n; i++) {
            if (bits == 16) { _Float16 v; memcpy(&v, raw + i, 2); h[i] = (float)v; }
            else { uint32_t v = (uint32_t)raw[i] << 16; memcpy(h + i, &v, 4); }
        }
        free(raw);
    } else nn_d2h(h, p, n * 4);
}
static void backward(void *x, shape5 s, int G, const float *values, float *mean, float *rstd) {
    size_t n = shape_numel(s), S = shape_spatial(s), len = (size_t)(s.c / G) * S;
    float *ref = malloc(n * 4), *got = malloc(n * 4), *ones = malloc(n * 4);
    double *ga = calloc(s.c, sizeof(double)), *gb = calloc(s.c, sizeof(double));
    for (int ng = 0; ng < s.n * G; ng++) {
        size_t start = (size_t)ng * len; double m = 0, var = 0, a = 0, b = 0;
        for (size_t i = 0; i < len; i++) m += values[start + i];
        m /= len;
        for (size_t i = 0; i < len; i++) { double v = values[start + i] - m; var += v * v; }
        double r = 1 / sqrt(var / len + 1e-5f);
        for (size_t i = 0; i < len; i++) {
            double z = (values[start + i] - m) * r, sig = 1 / (1 + exp(-z)), d = sig * (1 + z * (1 - sig));
            ref[start + i] = (float)d; a += d; b += d * z;
            size_t c = ((start + i) / S) % s.c; ga[c] += d * z; gb[c] += d;
        }
        for (size_t i = 0; i < len; i++) { double z = (values[start + i] - m) * r; ref[start + i] = (float)(r * (ref[start + i] - a / len - z * b / len)); }
    }
    for (size_t i = 0; i < n; i++) ones[i] = 1.f;
    float *gy = nn_malloc(n * 4), *gx = nn_malloc(n * 4), *gamma = nn_malloc(s.c * 4), *beta = nn_malloc(s.c * 4);
    float *gg = nn_malloc(s.c * 4), *bg = nn_malloc(s.c * 4), *scratch = nn_malloc(nn_gn_scratch(s));
    nn_h2d(gy, ones, n * 4); nn_h2d(gamma, ones, s.c * 4); nn_zero(beta, s.c * 4); nn_zero(gg, s.c * 4); nn_zero(bg, s.c * 4);
    nn_set_grad_bf16(0); nn_gn_silu_bwd(x, s, G, gamma, beta, mean, rstd, gy, gx, gg, bg, scratch);
    nn_d2h(got, gx, n * 4); double err = 0, scale = 1; int finite = 1;
    for (size_t i = 0; i < n; i++) { finite &= isfinite(got[i]) && isfinite(ref[i]); err = fmax(err, fabs((double)got[i] - ref[i])); scale = fmax(scale, fabs(ref[i])); }
    nn_d2h(got, gg, s.c * 4); double params = 0;
    for (int c = 0; c < s.c; c++) { finite &= isfinite(got[c]); params = fmax(params, fabs(got[c] - ga[c])); }
    nn_d2h(got, bg, s.c * 4);
    for (int c = 0; c < s.c; c++) { finite &= isfinite(got[c]); params = fmax(params, fabs(got[c] - gb[c])); }
    int ok = finite && err / scale < 1e-2 && params / (s.n * S) < 1e-3;
    printf("  stored-tensor GN/SiLU backward: dx %.3g affine/voxel %.3g %s\n", err / scale, params / (s.n * S), ok ? "ok" : "FAIL"); fails += !ok;
    nn_free(gy); nn_free(gx); nn_free(gamma); nn_free(beta); nn_free(gg); nn_free(bg); nn_free(scratch);
    free(ref); free(got); free(ones); free(ga); free(gb);
}
static void run(int bits, int api, int varied) {
    shape5 s = {2,32,5,7,9}, part = s, ys = s; part.c = 16; ys.c = 16;
    const int G = 8, NG = s.n * G; size_t n = shape_numel(s), S = shape_spatial(s), nw = (size_t)ys.c * s.c * 27;
    nn_set_prec(bits == 4 ? 3 : bits == 8 ? 2 : 1); nn_set_prec_policy(""); nn_set_f16(bits == 16);
    nn_set_act_bf16(bits != 0); nn_set_sr(0);
    float *host = malloc(n * 4), *weights = calloc(nw, 4), *bias = malloc(ys.c * 4);
    for (size_t i = 0; i < n; i++) host[i] = varied ? sinf((float)i * 0.17f) : 0.f;
    for (size_t i = 0; i < nw; i++) weights[i] = varied ? sinf((float)i * 0.037f) * 0.02f : 0.f;
    for (int c = 0; c < ys.c; c++) bias[c] = varied ? 0.1f * c : 1.2345f;
    float *xf = nn_malloc(n * 4), *w = nn_malloc(nw * 4), *b = nn_malloc(ys.c * 4);
    nn_h2d(xf, host, n * 4); nn_h2d(w, weights, nw * 4); nn_h2d(b, bias, ys.c * 4);
    void *x = activation(s, bits), *y = activation(ys, bits), *x0 = activation(part, bits), *x1 = activation(part, bits);
    nn_f32_to_act(xf, s, x);
    float *pf = nn_malloc(shape_numel(part) * 4);
    for (int sample = 0; sample < s.n; sample++)
        memcpy(weights + (size_t)sample * 16 * S, host + (size_t)sample * 32 * S, 16 * S * 4);
    nn_h2d(pf, weights, shape_numel(part) * 4); nn_f32_to_act(pf, part, x0);
    for (int sample = 0; sample < s.n; sample++)
        memcpy(weights + (size_t)sample * 16 * S, host + ((size_t)sample * 32 + 16) * S, 16 * S * 4);
    nn_h2d(pf, weights, shape_numel(part) * 4); nn_f32_to_act(pf, part, x1);
    float *mean = nn_malloc(NG * 4), *rstd = nn_malloc(NG * 4), m[16], r[16];
    int rc = api == 0 ? nn_conv3d_fwd_gn_stats(x, s, 0, nullptr, nullptr, nullptr, nullptr, w, b, ys.c, y, G, 1e-5f, mean, rstd)
        : api == 1 ? nn_conv3d_fwd_split(x0, x1, 16, s, 0, nullptr, nullptr, nullptr, nullptr, w, b, ys.c, y, G, 1e-5f, mean, rstd)
        : nn_conv3d_fwd_x(x, nullptr, nullptr, nullptr, 0, 0, s, w, b, ys.c, 3, 1, y, G, 1e-5f, mean, rstd);
    decode(y, ys, bits, host); nn_d2h(m, mean, NG * 4); nn_d2h(r, rstd, NG * 4);
    double worst_m = 0, worst_r = 0; int finite = rc == 0;
    for (int ng = 0; ng < NG; ng++) {
        size_t len = 2 * S; const float *p = host + (size_t)ng * len;
        double avg = 0, var = 0;
        for (size_t i = 0; i < len; i++) { finite &= isfinite(p[i]); avg += p[i]; }
        avg /= len;
        for (size_t i = 0; i < len; i++) { double v = (double)p[i] - avg; var += v * v; }
        double ref = 1 / sqrt(var / len + 1e-5f);
        finite &= isfinite(m[ng]) && isfinite(r[ng]);
        worst_m = fmax(worst_m, fabs(m[ng] - avg)); worst_r = fmax(worst_r, fabs(r[ng] / ref - 1));
    }
    int ok = finite && worst_m < 3e-6 && worst_r < 2e-3;
    printf("stored GN bits%d api%d %s: mean %.3g rstd %.3g %s\n", bits, api, varied ? "varied" : "constant", worst_m, worst_r, ok ? "ok" : "FAIL"); fails += !ok;
    if (api == 0) backward(y, ys, G, host, mean, rstd);
    nn_free(xf); nn_free(pf); nn_free(w); nn_free(b); nn_free(x); nn_free(x0); nn_free(x1); nn_free(y); nn_free(mean); nn_free(rstd);
    free(host); free(weights); free(bias);
}
static void replay(void) {
    shape5 s = {1,16,5,7,9}; const int G = 8; size_t n = shape_numel(s);
    nn_set_prec(1); nn_set_f16(1); nn_set_act_bf16(1);
    void *x = activation(s, 16), *y = activation(s, 16), *again = activation(s, 16);
    float *w = nn_malloc((size_t)16 * 16 * 27 * 4), *b = nn_malloc(16 * 4), bias[16];
    nn_zero(w, (size_t)16 * 16 * 27 * 4); for (int i = 0; i < 16; i++) bias[i] = 70000.f; nn_h2d(b, bias, sizeof bias);
    float *m = nn_malloc(G * 4), *r = nn_malloc(G * 4), *first = malloc(n * 4), *second = malloc(n * 4);
    nn_conv3d_fwd_x(x, nullptr, nullptr, nullptr, 0, 0, s, w, b, 16, 3, 1, y, G, 1e-5f, m, r);
    nn_conv3d_fwd_x(x, nullptr, nullptr, nullptr, 0, 0, s, w, b, 16, 3, 1, again, G, 1e-5f, nullptr, nullptr);
    decode(y, s, 16, first); decode(again, s, 16, second);
    int ok = 1; for (size_t i = 0; i < n; i++) ok &= first[i] == 65504.f && second[i] == first[i];
    printf("FP16 normalization-input saturation replay: %s\n", ok ? "ok" : "FAIL"); fails += !ok;
    nn_free(x); nn_free(y); nn_free(again); nn_free(w); nn_free(b); nn_free(m); nn_free(r); free(first); free(second);
}
int main(void) {
    if (nn_init(0)) return 1;
#ifndef TEST_GN_NO_MODE
    nn_set_gn_stored(1);
#endif
    int bits[] = {0,16,1,8,4};
    for (int k = 0; k < 5; k++) for (int api = 0; api < 3; api++) for (int varied = 0; varied <= 1; varied++) run(bits[k], api, varied);
#ifndef TEST_GN_NO_REPLAY
    replay();
#endif
    nn_sync(); const char *e = nn_check(); if (e) { fprintf(stderr, "%s\n", e); fails++; }
    printf("stored normalization contract: %s\n", fails ? "FAIL" : "ok"); return fails != 0;
}
