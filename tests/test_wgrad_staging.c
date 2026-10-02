/* Paired old/new operand gathering: native packed inputs, no giant FP32 temporary.
   Compare every accumulated parameter gradient and bias, with unchanged SR keys.
   Optional arguments P/iterations benchmark full-size shapes; default checks edges, split/up inputs and normalization. */
#include "nn.h"
#include "nn_lp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int failed, input_bits = 4, gradient_scale = 117;
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void checked(void) { nn_sync(); const char *e = nn_check(); if (e) { fprintf(stderr, "%s\n", e); exit(1); } }
static void *packed(shape5 s, int bits, unsigned seed, int scale) {
    size_t bytes = nn_mx_bytes(s, bits), S = shape_spatial(s);
    int bw = s.c <= 8 ? 8 : s.c <= 16 ? 16 : 32, nb = (s.c + bw - 1) / bw;
    size_t data = (size_t)s.n * nb * S * bw * bits / 8, chunk = 16 * 1024 * 1024;
    unsigned char *h = malloc(chunk); void *p = nn_malloc(bytes);
    for (size_t i = 0; i < chunk; i++) {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        h[i] = bits == 4 ? (unsigned char)seed : (unsigned char)((0x20u + (seed & 31u)) | (seed >> 24 & 128u));
    }
    for (size_t off = 0; off < data; off += chunk) nn_h2d((unsigned char *)p + off, h, data - off < chunk ? data - off : chunk);
    memset(h, scale, chunk);
    for (size_t off = data; off < bytes; off += chunk) nn_h2d((unsigned char *)p + off, h, bytes - off < chunk ? bytes - off : chunk);
    free(h); return p;
}
static float *floats(int n, float value) {
    float *h = malloc(n * 4), *p = nn_malloc(n * 4);
    for (int i = 0; i < n; i++) h[i] = value;
    nn_h2d(p, h, n * 4); free(h); return p;
}
static double invoke(int coop, int it, void *x, void *g, shape5 xs, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp) {
    lp_set_f4w_coop(coop); double times[64];
    for (int i = -2; i < it; i++) {
        nn_zero(gw, (size_t)xs.c * ys.c * 27 * 4); nn_zero(gb, ys.c * 4); checked();
        double start = now(); lp_bwd_w_f4(x, input_bits == 4 ? 4 : 3, xs, g, 3, ys, gw, gb, gp, sp, 0); checked();
        if (i >= 0) times[i] = (now() - start) * 1000;
    }
    for (int i = 0; i < it; i++) for (int j = i + 1; j < it; j++) if (times[j] < times[i]) { double t = times[i]; times[i] = times[j]; times[j] = t; }
    return times[it / 2];
}
static void run(int ci, int co, int P, int edge, int split, int up, int gn, unsigned sr, int it) {
    shape5 xs = {edge ? 2 : 1,ci,P,edge ? P + 2 : P,edge ? P + 4 : P}, ys = xs; ys.c = co;
    shape5 as = xs, bs = xs; int cs = split ? ci * 2 / 3 : ci;
    as.c = cs; bs.c = ci - cs;
    if (up) { as.d /= 2; as.h /= 2; as.w /= 2; }
    void *x = packed(as,input_bits,1723,127), *g = packed(ys,8,6281,gradient_scale), *x2 = split ? packed(bs,input_bits,912,127) : NULL;
    size_t nw = (size_t)ci * co * 27;
    float *gw = nn_malloc(nw * 4), *gb = nn_malloc(co * 4), *ref = malloc((nw + co) * 4), *got = malloc((nw + co) * 4);
    float *gamma = floats(ci,1.1f), *beta = floats(ci,0.02f), *mean = floats(xs.n * 8,0.1f), *rstd = floats(xs.n * 8,0.9f);
    gnp_t gp = {gamma,beta,mean,rstd,gn ? 8 : 0};
    split_t sp = {0}; sp.x2 = x2; sp.c_split = cs; sp.up = up; sp.sr = sr;
    if (gn && split) { sp.gp2 = gp; }
    double a = invoke(0,it,x,g,xs,ys,gw,gb,gp,sp);
    nn_d2h(ref,gw,nw * 4); nn_d2h(ref + nw,gb,co * 4);
    double b = invoke(1,it,x,g,xs,ys,gw,gb,gp,sp);
    nn_d2h(got,gw,nw * 4); nn_d2h(got + nw,gb,co * 4);
    double e = 0, norm = 0, maxe = 0, maxv = 0; int finite = 1;
    for (size_t i = 0; i < nw + co; i++) {
        double d = got[i] - ref[i]; finite &= isfinite(got[i]) && isfinite(ref[i]);
        e += d * d; norm += (double)ref[i] * ref[i]; maxe = fmax(maxe,fabs(d)); maxv = fmax(maxv,fabs(ref[i]));
    }
    double relative = sqrt(e / fmax(norm,1e-300));
    int ok = finite && relative < 5e-5 && maxe / fmax(maxv,1e-300) < 1e-4; failed |= !ok;
    double c = invoke(0,it,x,g,xs,ys,gw,gb,gp,sp);
    printf("input_bits=%d gradient_scale=%d ci=%d co=%d shape=%dx%dx%d split=%d up=%d gn=%d sr=%u old=%.3f new=%.3f old2=%.3f ms rel=%.3g max=%.3g %s\n",input_bits,gradient_scale,ci,co,xs.d,xs.h,xs.w,split,up,gn,sr,a,b,c,relative,maxe / fmax(maxv,1e-300),ok ? "ok" : "FAIL"); fflush(stdout);
    nn_free(x); nn_free(x2); nn_free(g); nn_free(gw); nn_free(gb); nn_free(gamma); nn_free(beta); nn_free(mean); nn_free(rstd); free(ref); free(got);
}
int main(int argc,char **argv) {
    if (nn_init(0)) return 1;
    if (argc > 1) {
        int P = atoi(argv[1]), it = argc > 2 ? atoi(argv[2]) : 7;
        if (P <= 0 || it <= 0 || it > 64) return 2;
        run(48,16,P,0,1,1,1,12345,it);
        run(16,16,P,0,0,0,1,12345,it);
        run(32,32,P / 2,0,0,0,1,12345,it);
    } else {
        int dims[][2] = {{4,16},{16,16},{32,32},{48,16},{48,24},{64,48},{16,8}};
        for (int i = 0; i < 7; i++) for (int sr = 0; sr < 2; sr++) run(dims[i][0],dims[i][1],6,1,0,0,i > 0,sr ? 12345 : 0,3);
        run(48,16,6,1,1,0,1,12345,3); run(48,16,6,1,1,1,1,12345,3);
        run(96,32,6,1,1,1,1,12345,3);
        input_bits = 8;
        run(16,16,6,1,0,0,1,12345,3);
        run(48,16,6,1,1,1,1,12345,3);
        input_bits = 4;
        int scales[] = {1,9,10,127,235};
        /* One spatial tile isolates operand arithmetic: multi-block FP32 atomics flush
           subnormal terms differently with different reduction orders, even on repeat. */
        for (int i = 0; i < 5; i++) { gradient_scale = scales[i]; run(16,16,2,0,0,0,1,12345,3); }
    }
    checked(); return failed;
}
