/* FP8 / FP4 vs BF16 vs fp32 for the 3^3 stride-1 convolutions: accuracy (relative L2 and max error against the exact
   fp32 kernels) and kernel throughput, per layer shape. Usage: bench_lp [P cin cout]...  (default: the layer list) */
#include "nn.h"
#include "nn_lp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t rs = 88172645463325252ull;
static double urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 11) * 0x1.0p-53; }
static float nrand(void) { return (float)(sqrt(-2 * log(urand() + 1e-300)) * cos(6.283185307179586 * urand())); }
static void err(const float *a, const float *r, size_t n, double *rel, double *mx) {
    double d2 = 0, r2 = 0, md = 0, mr = 0;
    for (size_t i = 0; i < n; i++) { double d = a[i] - r[i]; d2 += d * d; r2 += (double)r[i] * r[i]; if (fabs(d) > md) md = fabs(d); if (fabs(r[i]) > mr) mr = fabs(r[i]); }
    *rel = sqrt(d2 / (r2 + 1e-300)); *mx = md / (mr + 1e-300);
}
typedef void (*opfn)(void *);
static double timeit(void (*f)(void *), void *a) { f(a); nn_sync(); int it = 0; double t0 = now(), t; do { f(a); it++; nn_sync(); t = now() - t0; } while (t < 0.05 && it < 50); return t / it; }
typedef struct { float *x, *w, *b, *y, *gy, *gx, *gw, *gb, *scr; shape5 xs, ys; int cout; void *xb, *yb; } L;
static void f_fwd_b(void *p) { L *l = p; gnp_t none = {0}; split_t ns = {0}; lp_conv_fwd_f8(l->xb, 1, l->xs, l->w, l->b, l->cout, l->yb, 1, none, nullptr, 0, ns); }
static void f_bwd_w_b(void *p) { L *l = p; gnp_t none = {0}; split_t ns = {0}; lp_bwd_w_f8(l->xb, 1, l->xs, l->gy, 0, l->ys, l->gw, l->gb, none, ns); }
static void f_fwd(void *p) { L *l = p; nn_conv3d_fwd(l->x, l->xs, l->w, l->b, l->cout, 3, 1, l->y); }
static void f_bwd_d(void *p) { L *l = p; nn_conv3d_bwd_data(l->gy, l->ys, l->w, l->xs, 3, 1, l->gx, l->scr); }
static void f_bwd_w(void *p) { L *l = p; nn_conv3d_bwd_weight(l->x, l->xs, l->gy, l->ys, 3, 1, l->gw, l->gb); }
static int NB = 2;
static void run(int P, int cin, int cout, int nprec, const int *precs) {
    L l; l.xs = (shape5){NB, cin, P, P, P}; l.ys = nn_conv3d_out_shape(l.xs, cout, 3, 1); l.cout = cout;
    size_t nx = shape_numel(l.xs), ny = shape_numel(l.ys), nw = (size_t)cout * cin * 27;
    l.x = nn_malloc(nx * 4); l.gx = nn_malloc(nx * 4); l.y = nn_malloc(ny * 4); l.gy = nn_malloc(ny * 4);
    l.w = nn_malloc(nw * 4); l.gw = nn_malloc(nw * 4); l.b = nn_malloc(cout * 4); l.gb = nn_malloc(cout * 4); l.scr = nn_malloc(nn_conv3d_scratch(l.xs, cout, 3));
    float *hx = malloc(nx * 4), *hy = malloc(ny * 4), *hw = malloc(nw * 4), *hb = malloc(cout * 4);
    /* activations ~ silu(N(0,1)) (post GN+SiLU), weights kaiming, gradients tiny (1e-6 scale, as a mean-reduced loss gives) */
    for (size_t i = 0; i < nx; i++) { float v = nrand(); hx[i] = v / (1.f + expf(-v)); }
    float ws = sqrtf(2.f / (cin * 27)); for (size_t i = 0; i < nw; i++) hw[i] = nrand() * ws;
    for (size_t i = 0; i < ny; i++) hy[i] = nrand() * 1e-6f;
    for (int i = 0; i < cout; i++) hb[i] = nrand() * 0.1f;
    nn_h2d(l.x, hx, nx * 4); nn_h2d(l.gy, hy, ny * 4); nn_h2d(l.w, hw, nw * 4); nn_h2d(l.b, hb, cout * 4);
    float *ry = malloc(ny * 4), *rgx = malloc(nx * 4), *rgw = malloc(nw * 4), *rgb = malloc(cout * 4), *ty = malloc(ny * 4), *tgx = malloc(nx * 4), *tgw = malloc(nw * 4), *tgb = malloc(cout * 4);
    nn_set_prec(0);
    f_fwd(&l); nn_d2h(ry, l.y, ny * 4);
    f_bwd_d(&l); nn_d2h(rgx, l.gx, nx * 4);
    nn_zero(l.gw, nw * 4); nn_zero(l.gb, cout * 4); f_bwd_w(&l); nn_d2h(rgw, l.gw, nw * 4); nn_d2h(rgb, l.gb, cout * 4);
    double flop = 2.0 * shape_spatial(l.ys) * NB * cout * cin * 27;
    printf("P %3d cin %3d cout %3d |", P, cin, cout);
    for (int k = 0; k < nprec; k++) {
        int p = precs[k];
        nn_set_prec(p);
        double e1, m1, e2, m2, e3, m3, e4, m4;
        nn_scale(l.y, NAN, ny); nn_scale(l.gx, NAN, nx);   /* poison: an output element the kernel misses shows up as NaN */
        f_fwd(&l); nn_d2h(ty, l.y, ny * 4); err(ty, ry, ny, &e1, &m1);
        f_bwd_d(&l); nn_d2h(tgx, l.gx, nx * 4); err(tgx, rgx, nx, &e2, &m2);
        nn_zero(l.gw, nw * 4); nn_zero(l.gb, cout * 4); f_bwd_w(&l); nn_d2h(tgw, l.gw, nw * 4); nn_d2h(tgb, l.gb, cout * 4); err(tgw, rgw, nw, &e3, &m3); err(tgb, rgb, cout, &e4, &m4);
        double tf = timeit(f_fwd, &l), td = timeit(f_bwd_d, &l), tw = timeit(f_bwd_w, &l);
        printf(" p%d fwd %5.1f TF %.1e | bwd_d %5.1f TF %.1e | bwd_w %5.1f TF %.1e gb %.0e |", p, flop / tf / 1e12, e1, flop / td / 1e12, e2, flop / tw / 1e12, e3, e4);
        (void)m1; (void)m2; (void)m3; (void)m4;
    }
    if (getenv("UFSM_LP_BF16")) {   /* bf16-activation instantiations and accumulate mode of the FP8 kernels */
        gnp_t none = {nullptr, nullptr, nullptr, nullptr, 0};
        split_t ns = {nullptr, 0, nullptr, 0, 0}, acc1 = {nullptr, 0, nullptr, 0, 1};
        void *xb = nn_malloc(nx * 2), *yb = nn_malloc(ny * 2);
        lp_f32_to_bf16(l.x, nx, xb);
        double e1, e2, e3, mm;
        nn_scale(l.y, NAN, ny); lp_f32_to_bf16(l.y, ny, yb);
        lp_conv_fwd_f8(xb, 1, l.xs, l.w, l.b, cout, yb, 1, none, nullptr, 0, ns); lp_bf16_to_f32(yb, ny, l.y); nn_d2h(ty, l.y, ny * 4); err(ty, ry, ny, &e1, &mm);
        nn_zero(l.gw, nw * 4); nn_zero(l.gb, cout * 4); lp_bwd_w_f8(xb, 1, l.xs, l.gy, 0, l.ys, l.gw, l.gb, none, ns); nn_d2h(tgw, l.gw, nw * 4); err(tgw, rgw, nw, &e2, &mm);
        /* accumulate: y = ry (fp32 reference) then y += conv -> compare with 2 * ry */
        nn_h2d(l.y, ry, ny * 4); lp_conv_fwd_f8(l.x, 0, l.xs, l.w, l.b, cout, l.y, 0, none, nullptr, 0, acc1); nn_d2h(ty, l.y, ny * 4);
        for (size_t i = 0; i < ny; i++) ry[i] *= 2; err(ty, ry, ny, &e3, &mm);
        l.xb = xb; l.yb = yb;
        double tfb = timeit(f_fwd_b, &l), twb = timeit(f_bwd_w_b, &l);
        printf(" fp8 bf16-act: fwd %5.1f TF %.1e bwd_w %5.1f TF %.1e | accum fwd %.1e |", flop / tfb / 1e12, e1, flop / twb / 1e12, e2, e3);
        nn_free(xb); nn_free(yb);
    }
    printf("\n"); fflush(stdout);
    const char *e = nn_check(); if (e) { printf("cuda: %s\n", e); exit(1); }
    nn_free(l.x); nn_free(l.gx); nn_free(l.y); nn_free(l.gy); nn_free(l.w); nn_free(l.gw); nn_free(l.b); nn_free(l.gb); nn_free(l.scr);
    free(hx); free(hy); free(hw); free(hb); free(ry); free(rgx); free(rgw); free(rgb); free(ty); free(tgx); free(tgw); free(tgb);
}
int main(int argc, char **argv) {
    nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0);
    nn_set_act_bf16(0); nn_set_grad_bf16(0);   /* the harness feeds fp32 tensors through the public wrappers */
    if (getenv("UFSM_B")) NB = atoi(getenv("UFSM_B"));
    int precs[4] = {1, 2, 3}, np = 2;
    if (getenv("UFSM_PRECS")) { np = 0; for (char *s = getenv("UFSM_PRECS"); *s; s++) if (*s >= '0' && *s <= '9') precs[np++] = *s - '0'; }
    printf("batch %d; columns: TFLOP/s and relative L2 error vs fp32 (fwd, bwd_data, bwd_weight, bias grad)\n", NB);
    if (argc > 3) { for (int i = 1; i + 2 < argc; i += 3) run(atoi(argv[i]), atoi(argv[i + 1]), atoi(argv[i + 2]), np, precs); return 0; }
    /* the layers of the (16,32,64,80) U-Net, then the requested cin sweep at 96^3 / 48^3 */
    static const int net[][3] = {{96, 4, 16}, {96, 16, 16}, {96, 48, 16}, {48, 16, 32}, {48, 32, 32}, {48, 96, 32}, {24, 32, 64}, {24, 64, 64}, {24, 144, 64}, {12, 64, 80}, {12, 80, 80}};
    printf("-- network layers\n");
    for (unsigned i = 0; i < sizeof net / sizeof net[0]; i++) run(net[i][0], net[i][1], net[i][2], np, precs);
    printf("-- cin sweep (cout 16 at 96^3, 32 at 48^3)\n");
    static const int cins[] = {4, 16, 32, 48, 64, 80, 144};
    for (int P = 96; P >= 48; P /= 2) for (unsigned i = 0; i < 7; i++) run(P, cins[i], P == 96 ? 16 : 32, np, precs);
    return 0;
}
