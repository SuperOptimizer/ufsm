/* Half-resolution network transforms (nn_half_input / nn_half_output / nn_half_grad, unet_cfg.half_res): pool input = block means,
   pack input = block means + the CT's Haar details and mean absolute deviation; pool output = trilinear 2x against a host reference, pack output = the sub-voxel
   permutation; nn_half_grad is the adjoint of nn_half_output (<U l, g> = <l, U^T g>) in both modes; a half-resolution network
   checkpoint keeps half_res; the quarter-resolution chains (nn_lowres_*, div 4) are adjoint too. */
#include "unet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

static float *dev(const float *h, size_t n) { float *d = nn_malloc(n * 4); nn_h2d(d, h, n * 4); return d; }
static float *rnd(size_t n, unsigned seed) {
    float *h = malloc(n * 4);
    for (size_t i = 0; i < n; i++) h[i] = sinf((float)((i * 2654435761u + seed) % 9973) * 0.013f) + (float)((i * 2246822519u + seed) % 1000) / 1000.f - 0.5f;
    return h;
}
static void taps(int f, int n, int i[2], float w[2]) {   /* 2x linear interpolation, fine voxel f samples (f + 0.5) / 2 - 0.5, edge-clamped */
    i[0] = f >> 1; i[1] = f & 1 ? (i[0] + 1 < n ? i[0] + 1 : n - 1) : (i[0] > 0 ? i[0] - 1 : 0); w[0] = 0.75f; w[1] = 0.25f;
}

int main(void) {
    nn_init(0);
    const int d = 4, h = 3, w = 5, D = 2 * d, H = 2 * h, W = 2 * w, C = 2;
    const size_t hv = (size_t)d * h * w, fv = 8 * hv;
    {   /* input: pool and pack */
        float *xh = rnd(4 * fv, 1), *x = dev(xh, 4 * fv);
        float *y = nn_malloc(16 * hv * 4), *yh = malloc(16 * hv * 4);
        for (int mode = 1; mode <= 2; mode++) {
            nn_half_input(x, (shape5){1, 4, D, H, W}, mode, y, 0);
            const int co = mode == 2 ? 8 : 4;
            nn_d2h(yh, y, (size_t)co * hv * 4);
            double err = 0;
            for (int z = 0; z < d; z++) for (int yy = 0; yy < h; yy++) for (int xx = 0; xx < w; xx++) {
                const size_t v = ((size_t)z * h + yy) * w + xx;
                for (int c = 0; c < 4; c++) {
                    double q[8], s = 0;
                    for (int k = 0; k < 8; k++) { q[k] = xh[(size_t)c * fv + ((size_t)(2 * z + (k >> 2)) * H + 2 * yy + (k >> 1 & 1)) * W + 2 * xx + (k & 1)]; s += q[k]; }
                    err = fmax(err, fabs(yh[(size_t)c * hv + v] - s / 8));
                    if (mode == 2 && c == 0) {
                        double g[3] = {0, 0, 0}, a = 0;
                        for (int k = 0; k < 8; k++) { for (int e = 0; e < 3; e++) g[e] += (k >> (2 - e) & 1) ? q[k] : -q[k]; a += fabs(q[k] - s / 8); }
                        for (int e = 0; e < 3; e++) err = fmax(err, fabs(yh[(size_t)(4 + e) * hv + v] - g[e] / 8));
                        err = fmax(err, fabs(yh[(size_t)7 * hv + v] - a / 8));
                    }
                }
            }
            CHECK(err < 1e-6, "input mode %d: max error %g", mode, err);
        }
        nn_free(x); nn_free(y); free(xh); free(yh);
    }
    for (int mode = 1; mode <= 2; mode++) {   /* output vs host reference, and the adjoint */
        const int cl = mode == 2 ? 8 * C : C;
        float *lh = rnd((size_t)cl * hv, 7), *gh = rnd((size_t)C * fv, 11);
        float *l = dev(lh, (size_t)cl * hv), *g = dev(gh, (size_t)C * fv), *o = nn_malloc((size_t)C * fv * 4), *gl = nn_malloc((size_t)cl * hv * 4);
        float *oh = malloc((size_t)C * fv * 4), *glh = malloc((size_t)cl * hv * 4);
        const shape5 hs = {1, C, d, h, w};
        nn_half_output(l, 0, hs, mode, o); nn_d2h(oh, o, (size_t)C * fv * 4);
        nn_half_grad(g, 0, hs, mode, gl); nn_d2h(glh, gl, (size_t)cl * hv * 4);
        double err = 0;
        for (int c = 0; c < C; c++) for (int z = 0; z < D; z++) for (int yy = 0; yy < H; yy++) for (int xx = 0; xx < W; xx++) {
            double r;
            if (mode == 2) r = lh[((size_t)c * 8 + (z & 1) * 4 + (yy & 1) * 2 + (xx & 1)) * hv + ((size_t)(z >> 1) * h + (yy >> 1)) * w + (xx >> 1)];
            else {
                int iz[2], iy[2], ix[2]; float wz[2], wy[2], wx[2]; taps(z, d, iz, wz); taps(yy, h, iy, wy); taps(xx, w, ix, wx);
                r = 0;
                for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) for (int e = 0; e < 2; e++)
                    r += (double)wz[a] * wy[b] * wx[e] * lh[(size_t)c * hv + ((size_t)iz[a] * h + iy[b]) * w + ix[e]];
            }
            err = fmax(err, fabs(oh[(size_t)c * fv + ((size_t)z * H + yy) * W + xx] - r));
        }
        CHECK(err < 1e-5, "output mode %d: max error %g", mode, err);
        double a = 0, b = 0, m = 0;
        for (size_t i = 0; i < (size_t)C * fv; i++) { a += (double)oh[i] * gh[i]; m += fabs((double)oh[i] * gh[i]); }
        for (size_t i = 0; i < (size_t)cl * hv; i++) b += (double)lh[i] * glh[i];
        CHECK(fabs(a - b) < 1e-5 * m, "adjoint mode %d: <U l, g> %.9g vs <l, U^T g> %.9g", mode, a, b);
        printf("half mode %d: output max error %.2g, adjoint %.9g / %.9g\n", mode, err, a, b);
        nn_free(l); nn_free(g); nn_free(o); nn_free(gl); free(lh); free(gh); free(oh); free(glh);
    }
    for (int mode = 1; mode <= 2; mode++) {   /* quarter resolution (nn_lowres_*, div 4): the adjoint, and the input = two half stages */
        const int q = 2, cl = mode == 2 ? 8 * C : C;
        const size_t qv = (size_t)q * q * q, fv4 = 64 * qv, mv = 8 * qv;
        float *lh = rnd((size_t)cl * qv, 3), *gh = rnd((size_t)C * fv4, 5), *xh = rnd(4 * fv4, 9);
        float *l = dev(lh, (size_t)cl * qv), *g = dev(gh, (size_t)C * fv4), *x = dev(xh, 4 * fv4);
        float *o = nn_malloc((size_t)C * fv4 * 4), *gl = nn_malloc((size_t)cl * qv * 4), *t = nn_malloc(16 * mv * 4), *y = nn_malloc(8 * qv * 4), *y2 = nn_malloc(8 * qv * 4);
        float *oh = malloc((size_t)C * fv4 * 4), *glh = malloc((size_t)cl * qv * 4), *a1 = malloc(8 * qv * 4), *a2 = malloc(8 * qv * 4);
        const shape5 qs = {1, C, q, q, q}, xs4 = {1, 4, 4 * q, 4 * q, 4 * q};
        nn_lowres_output(l, 0, qs, 4, mode, t, o); nn_d2h(oh, o, (size_t)C * fv4 * 4);
        nn_lowres_grad(g, 0, qs, 4, mode, t, gl); nn_d2h(glh, gl, (size_t)cl * qv * 4);
        double a = 0, b = 0, m = 0;
        for (size_t i = 0; i < (size_t)C * fv4; i++) { a += (double)oh[i] * gh[i]; m += fabs((double)oh[i] * gh[i]); }
        for (size_t i = 0; i < (size_t)cl * qv; i++) b += (double)lh[i] * glh[i];
        CHECK(fabs(a - b) < 1e-5 * m, "quarter adjoint mode %d: %.9g vs %.9g", mode, a, b);
        nn_lowres_input(x, xs4, 4, mode, t, y, 0);
        nn_half_input(x, xs4, 1, t, 0); nn_half_input(t, (shape5){1, 4, 2 * q, 2 * q, 2 * q}, mode, y2, 0);
        const int co = mode == 2 ? 8 : 4;
        nn_d2h(a1, y, (size_t)co * qv * 4); nn_d2h(a2, y2, (size_t)co * qv * 4);
        CHECK(!memcmp(a1, a2, (size_t)co * qv * 4), "quarter input mode %d", mode);
        printf("quarter mode %d: adjoint %.9g / %.9g\n", mode, a, b);
        nn_free(l); nn_free(g); nn_free(x); nn_free(o); nn_free(gl); nn_free(t); nn_free(y); nn_free(y2);
        free(lh); free(gh); free(xh); free(oh); free(glh); free(a1); free(a2);
    }
    {   /* the checkpoint keeps half_res */
        unet_cfg cfg = {3, {32, 64, 96}, 8, 16, 8, 1, 0}; cfg.half_res = 2;
        unet *u = unet_create(&cfg); unet_init(u, 1);
        const char *path = "/tmp/ufsm_test_half.ckpt";
        CHECK(!unet_save(u, path, 5, nullptr), "save");
        unet_cfg pc; int st;
        CHECK(!unet_peek(path, &pc, &st) && pc.half_res == 2 && pc.cin == 8 && pc.cout == 16, "peek half_res %d", pc.half_res);
        remove(path); unet_free(u);
    }
    if (fails) { fprintf(stderr, "test_half: %d failure(s)\n", fails); return 1; }
    printf("test_half: OK\n");
    return 0;
}
