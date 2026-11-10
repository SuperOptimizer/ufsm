/* A latent net's first conv under fp4 activations: 32 MX-fp8 input channels (the learned cell channels) into MX-fp4 activations.
   Forward: the general fp8 kernel with MX-fp8 in / MX-fp4 out (lp_conv_fwd_f8) against a host fp32 convolution of the decoded input
   and against the MX-fp8 / MX-fp8 kernel (the stored rows differ only by their rounding).
   Weight gradient: lp_bwd_w_f8 with MX-fp4 gradients (z chunks re-encoded as MX-fp8, lp_f8wgrad.cu bwd_w_f8_mixed) against the host
   sum and against the MX-fp8 / MX-fp8 kernel on the whole tensor with the same (exactly re-encoded) gradient: chunking must not
   change the sums beyond fp32 order. */
#include "nn.h"
#include "nn_lp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rs = 88172645463325252ull;
static float ru(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((double)(rs >> 11) / 9007199254740992.0 * 2 - 1); }

int main(void) {
    nn_init(0);
    int fails = 0, cases = 0;
    static const int dims[][5] = {{32, 96, 12, 16, 16}, {32, 32, 7, 12, 20}, {29, 96, 10, 8, 16}, {32, 64, 16, 16, 16}};   /* cin, cout, D, H, W */
    for (size_t di = 0; di < sizeof dims / sizeof dims[0]; di++) {
        const int ci = dims[di][0], co = dims[di][1], D = dims[di][2], H = dims[di][3], W = dims[di][4]; const size_t S = (size_t)D * H * W, nw = (size_t)co * ci * 27;
        const shape5 xs = {1, ci, D, H, W}, ys = {1, co, D, H, W};
        float *x = malloc(ci * S * 4), *xq = malloc(ci * S * 4), *w = malloc(nw * 4), *b = malloc(co * 4), *g = malloc(co * S * 4), *gq = malloc(co * S * 4);
        for (size_t k = 0; k < ci * S; k++) x[k] = ru() * (1 + 3 * (float)(k / S % 5 == 0));   /* (channels of different scale: the rows' block scale) */
        for (size_t k = 0; k < nw; k++) w[k] = 0.1f * ru();
        for (int k = 0; k < co; k++) b[k] = 0.2f * ru();
        for (size_t k = 0; k < co * S; k++) g[k] = ru() * ru() * exp2f(floorf(5 * ru() * (float)((k % S) % 7 == 0)));   /* (some voxels much larger: the kernel's 32-voxel blocks round the others) */
        float *d_x = nn_malloc(ci * S * 4), *d_w = nn_malloc(nw * 4), *d_b = nn_malloc(co * 4), *d_g = nn_malloc(co * S * 4), *d_o = nn_malloc(co * S * 4), *d_gw = nn_malloc(nw * 4), *d_gb = nn_malloc(co * 4);
        void *xm = nn_malloc(lp_mx8_bytes(1, ci, S)), *y4 = nn_malloc(lp_mx4_bytes(1, co, S)), *y8 = nn_malloc(lp_mx8_bytes(1, co, S)), *g4 = nn_malloc(lp_mx4_bytes(1, co, S)), *g8 = nn_malloc(lp_mx8_bytes(1, co, S));
        nn_h2d(d_x, x, ci * S * 4); nn_h2d(d_w, w, nw * 4); nn_h2d(d_b, b, co * 4); nn_h2d(d_g, g, co * S * 4);
        lp_f32_to_mx8(d_x, 1, ci, S, xm); lp_mx8_to_f32(xm, 1, ci, S, d_x); nn_d2h(xq, d_x, ci * S * 4);          /* the stored input */
        lp_f32_to_mx4(d_g, 1, co, S, g4); lp_mx4_to_f32(g4, 1, co, S, d_g); nn_d2h(gq, d_g, co * S * 4);          /* the stored gradient */
        lp_f32_to_mx8(d_g, 1, co, S, g8);   /* = g4's values (exact) */
        { float *t = malloc(co * S * 4); lp_mx8_to_f32(g8, 1, co, S, d_o); nn_d2h(t, d_o, co * S * 4); size_t nd = 0; for (size_t k = 0; k < co * S; k++) nd += t[k] != gq[k];
          cases++; if (nd) { fprintf(stderr, "FAIL: %dx%d: %zu MX-fp4 gradient values are not exact in MX-fp8\n", ci, co, nd); fails++; } free(t); }
        /* host references */
        float *yr = calloc(co * S, 4), *gwr = calloc(nw, 4);
        for (int o = 0; o < co; o++) for (int c = 0; c < ci; c++) for (int t = 0; t < 27; t++) {
            const int kz = t / 9 - 1, ky = t / 3 % 3 - 1, kx = t % 3 - 1; const float wv = w[((size_t)o * ci + c) * 27 + t]; double gs = 0;
            for (int z = 0; z < D; z++) { const int z2 = z + kz; if (z2 < 0 || z2 >= D) continue;
                for (int yy = 0; yy < H; yy++) { const int y2 = yy + ky; if (y2 < 0 || y2 >= H) continue;
                    const float *xr = xq + ((size_t)c * D + z2) * H * W + (size_t)y2 * W, *gr = gq + ((size_t)o * D + z) * H * W + (size_t)yy * W; float *yo = yr + ((size_t)o * D + z) * H * W + (size_t)yy * W;
                    for (int xx = kx < 0 ? 1 : 0; xx < (kx > 0 ? W - 1 : W); xx++) { yo[xx] += wv * xr[xx + kx]; gs += (double)gr[xx] * xr[xx + kx]; } } }
            gwr[((size_t)o * ci + c) * 27 + t] = (float)gs;
        }
        for (int o = 0; o < co; o++) for (size_t v = 0; v < S; v++) yr[o * S + v] += b[o];
        /* forward */
        float *h4 = malloc(co * S * 4), *h8 = malloc(co * S * 4);
        lp_conv_fwd_f8(xm, 3, xs, d_w, d_b, co, y4, 4, (gnp_t){0}, nullptr, 0, (split_t){0}); lp_mx4_to_f32(y4, 1, co, S, d_o); nn_d2h(h4, d_o, co * S * 4);
        lp_conv_fwd_f8(xm, 3, xs, d_w, d_b, co, y8, 3, (gnp_t){0}, nullptr, 0, (split_t){0}); lp_mx8_to_f32(y8, 1, co, S, d_o); nn_d2h(h8, d_o, co * S * 4);
        double e4 = 0, e8 = 0, e48 = 0, rr = 0, worst = 0;
        for (int blk = 0; blk < (co + 31) / 32; blk++) for (size_t v = 0; v < S; v++) {   /* per stored row: one scale for its (up to) 32 channels */
            double am = 0; for (int c = blk * 32; c < co && c < blk * 32 + 32; c++) am = fmax(am, fabs(yr[c * S + v]));
            for (int c = blk * 32; c < co && c < blk * 32 + 32; c++) { const double a = h4[c * S + v] - yr[c * S + v], q = h8[c * S + v] - yr[c * S + v], d = h4[c * S + v] - h8[c * S + v];
                e4 += a * a; e8 += q * q; e48 += d * d; rr += (double)yr[c * S + v] * yr[c * S + v]; worst = fmax(worst, fabs(a) / (am + 1e-6)); }
        }
        e4 = sqrt(e4 / rr); e8 = sqrt(e8 / rr); e48 = sqrt(e48 / rr); cases++;
        /* fp8 rows: the weights' and the row's e4m3 rounding (a few %); fp4 rows add the e2m1 rounding: at most 1/6 of the row's largest value + the fp8 error */
        if (!(e8 < 0.05) || !(e4 < 0.16) || !(worst < 0.30)) { fprintf(stderr, "FAIL: forward %dx%d %dx%dx%d: rms error / rms fp8 rows %.4f, fp4 rows %.4f (fp4 - fp8 %.4f), worst fp4 error / row max %.3f\n", ci, co, D, H, W, e8, e4, e48, worst); fails++; }
        /* weight gradient: chunked mixed, 1 and several chunks, against the whole-tensor fp8 kernel and the host */
        float *gwa = malloc(nw * 4), *gwb = malloc(nw * 4), *gba = malloc(co * 4), *gbb = malloc(co * 4);
        nn_zero(d_gw, nw * 4); nn_zero(d_gb, co * 4); lp_bwd_w_f8(xm, 3, xs, g8, 3, ys, d_gw, d_gb, (gnp_t){0}, (split_t){0}); nn_d2h(gwb, d_gw, nw * 4); nn_d2h(gbb, d_gb, co * 4);
        nn_zero(d_gw, nw * 4); nn_zero(d_gb, co * 4); lp_bwd_w_f8(xm, 3, xs, g4, 4, ys, d_gw, d_gb, (gnp_t){0}, (split_t){0}); nn_d2h(gwa, d_gw, nw * 4); nn_d2h(gba, d_gb, co * 4);
        double ea = 0, eb = 0, eab = 0, r2 = 0, ebias = 0, rb = 0;
        for (size_t k = 0; k < nw; k++) { const double a = gwa[k] - gwr[k], q = gwb[k] - gwr[k], d = gwa[k] - gwb[k]; ea += a * a; eb += q * q; eab += d * d; r2 += (double)gwr[k] * gwr[k]; }
        for (int o = 0; o < co; o++) { double sref = 0; for (size_t v = 0; v < S; v++) sref += gq[o * S + v]; ebias += (gba[o] - sref) * (gba[o] - sref); rb += sref * sref; }
        ea = sqrt(ea / r2); eb = sqrt(eb / r2); eab = sqrt(eab / r2); ebias = sqrt(ebias / (rb + 1e-12)); cases++;
        if (!(ea < 0.05) || !(eb < 0.05) || !(eab < 2e-4) || !(ebias < 0.02)) { fprintf(stderr, "FAIL: weight gradient %dx%d %dx%dx%d: rms error / rms chunked %.5f, whole fp8 %.5f, chunked - whole %.2e, bias %.5f\n", ci, co, D, H, W, ea, eb, eab, ebias); fails++; }
        printf("test_lat_conv0: %dx%d %dx%dx%d: forward rms error fp8 rows %.4f, fp4 rows %.4f (worst / row max %.3f); weight gradient chunked %.5f, whole fp8 %.5f, chunked - whole %.1e, bias %.1e\n", ci, co, D, H, W, e8, e4, worst, ea, eb, eab, ebias);
        free(x); free(xq); free(w); free(b); free(g); free(gq); free(yr); free(gwr); free(h4); free(h8); free(gwa); free(gwb); free(gba); free(gbb);
        nn_free(d_x); nn_free(d_w); nn_free(d_b); nn_free(d_g); nn_free(d_o); nn_free(d_gw); nn_free(d_gb); nn_free(xm); nn_free(y4); nn_free(y8); nn_free(g4); nn_free(g8);
    }
    const char *e = nn_check(); if (e) { fprintf(stderr, "FAIL: CUDA error %s\n", e); fails++; }
    if (fails) { printf("test_lat_conv0: %d of %d cases FAIL\n", fails, cases); return 1; }
    printf("test_lat_conv0: ok (%d cases)\n", cases);
    return 0;
}
