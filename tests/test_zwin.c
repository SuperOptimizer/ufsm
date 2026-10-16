/* z-window launches (level-0 slab streaming plan, stage S2; nn_set_zwin): an op run as several windows of output planes covering a
   tensor, each window with the op's stochastic-rounding seed replayed (nn_sr_set_ctr), gives the bits of one launch over the whole
   tensor (window [0, D)): MX outputs byte for byte, output GroupNorm statistics at double-sum noise, weight gradients at fp32 atomic
   noise. Windows of 8 and 16 planes, a partial last window (depth not a multiple of the window), the production precisions (MX-fp4
   activations and gradients, SR on). Against the launch without a window: equal where the same kernel runs (else the routing, e.g.
   conv_fwd_f4p_k, differs only at the fp4 rounding level). */
#include "nn.h"
#include "nn_lp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int bad;
static uint64_t rs = 777;
static float frand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 11) * 0x1.0p-53 * 2 - 1); }
static float *dev_from(const float *h, size_t n) { float *d = nn_malloc(n * 4); nn_h2d(d, h, n * 4); return d; }
static float *dev_rand(size_t n, float scale) { float *h = malloc(n * 4); for (size_t i = 0; i < n; i++) h[i] = frand() * scale; float *d = dev_from(h, n); free(h); return d; }
static float *dev_zero(size_t n) { float *d = nn_malloc(n * 4); nn_zero(d, n * 4); return d; }
static void *mx4_new(shape5 s) { size_t b = nn_mx4_bytes(s); void *p = nn_malloc(b); nn_zero(p, b); nn_set_storage(p, b, 4); return p; }
static void *mx4_from(const float *x, shape5 s) { void *p = mx4_new(s); lp_f32_to_mx4(x, s.n, s.c, shape_spatial(s), p); return p; }
static void mode_prod(void) {
    nn_set_tf32(1); nn_set_prec(3); nn_set_act_bf16(1); nn_set_grad_bf16(1); nn_set_f16(1); nn_set_grad_scale(1024); nn_set_sr(1);
}
static void mode_off(void) { nn_set_act_bf16(0); nn_set_grad_bf16(0); nn_set_f16(0); nn_set_sr(0); }

/* the windows [z, z + S) covering depth D (the last one partial), in order */
static int windows(int D, int S, int *z0, int *z1) { int k = 0; for (int z = 0; z < D; z += S, k++) { z0[k] = z; z1[k] = z + S < D ? z + S : D; } return k; }

typedef struct { const char *name; size_t nb; void *out; void *out2; size_t nb2; float *st; size_t nst; float *gw; size_t ngw; } res_t;
/* compare result a (one window [0, D)) against b (windows): bytes, statistics, weight gradients */
static void cmp(const char *what, int S, const res_t *a, const res_t *b) {
    char m[160]; snprintf(m, sizeof m, "%s, %d-plane windows", what, S);
    size_t nd = 0;
    if (a->out) { uint8_t *ha = malloc(a->nb), *hb = malloc(a->nb); nn_d2h(ha, a->out, a->nb); nn_d2h(hb, b->out, a->nb);
                  for (size_t i = 0; i < a->nb; i++) nd += ha[i] != hb[i];
                  free(ha); free(hb); }
    if (a->out2) { uint8_t *xa = malloc(a->nb2), *xb = malloc(a->nb2); nn_d2h(xa, a->out2, a->nb2); nn_d2h(xb, b->out2, a->nb2); for (size_t i = 0; i < a->nb2; i++) nd += xa[i] != xb[i]; free(xa); free(xb); }
    double es = 0, eg = 0;
    if (a->st) { float *sa = malloc(a->nst * 4), *sb = malloc(a->nst * 4); nn_d2h(sa, a->st, a->nst * 4); nn_d2h(sb, b->st, a->nst * 4);
                 for (size_t i = 0; i < a->nst; i++) { const double d = fabs((double)sa[i] - sb[i]) / (fabs((double)sa[i]) + 1e-12); if (d > es) es = d; } free(sa); free(sb); }
    if (a->gw) { float *ga = malloc(a->ngw * 4), *gb = malloc(a->ngw * 4); nn_d2h(ga, a->gw, a->ngw * 4); nn_d2h(gb, b->gw, a->ngw * 4);
                 double d2 = 0, r2 = 0; for (size_t i = 0; i < a->ngw; i++) { const double d = (double)ga[i] - gb[i]; d2 += d * d; r2 += (double)ga[i] * ga[i]; }
                 eg = r2 > 0 ? sqrt(d2 / r2) : 0; free(ga); free(gb); }
    const int ok = nd == 0 && es < 1e-5 && eg < 1e-5;
    printf("  %-58s %zu bytes differ, stats %.2g, wgrad %.2g%s\n", m, nd, es, eg, ok ? "" : "  FAIL");
    if (!ok) bad++;
}


/* coverage: a lone window [z0, z1) wrote exactly its planes (equal to the reference there, untouched zero elsewhere) */
static void cover(const char *what, const void *ref, const void *win, shape5 s, int z0, int z1) {
    const size_t n = shape_numel(s), P = (size_t)s.h * s.w;
    float *dr = dev_zero(n), *dw = dev_zero(n), *hr = malloc(n * 4), *hw = malloc(n * 4);
    lp_mx4_to_f32(ref, s.n, s.c, shape_spatial(s), dr); lp_mx4_to_f32(win, s.n, s.c, shape_spatial(s), dw);
    nn_d2h(hr, dr, n * 4); nn_d2h(hw, dw, n * 4);
    size_t in_bad = 0, out_bad = 0, in_nz = 0;
    for (int c = 0; c < s.n * s.c; c++) for (int z = 0; z < s.d; z++) for (size_t v = 0; v < P; v++) {
        const size_t i = ((size_t)c * s.d + z) * P + v;
        if (z >= z0 && z < z1) { in_bad += hw[i] != hr[i]; in_nz += hr[i] != 0; } else out_bad += hw[i] != 0;
    }
    const int ok = !in_bad && !out_bad && in_nz;
    printf("  %-58s window [%d, %d): %zu inside differ, %zu outside written%s\n", what, z0, z1, in_bad, out_bad, ok ? "" : "  FAIL");
    if (!ok) bad++;
    nn_free(dr); nn_free(dw); free(hr); free(hw);
}
/* the launch without a window (rel err: 0 when the same kernel runs) */
static void vs_plain(const char *what, const void *ref, const void *pl, shape5 s, double tol) {
    const size_t n = shape_numel(s); float *da = dev_zero(n), *db = dev_zero(n), *ha = malloc(n * 4), *hb = malloc(n * 4);
    lp_mx4_to_f32(ref, s.n, s.c, shape_spatial(s), da); lp_mx4_to_f32(pl, s.n, s.c, shape_spatial(s), db); nn_d2h(ha, da, n * 4); nn_d2h(hb, db, n * 4);
    double d2 = 0, r2 = 0; for (size_t i = 0; i < n; i++) { const double d = (double)ha[i] - hb[i]; d2 += d * d; r2 += (double)ha[i] * ha[i]; }
    const double e = r2 > 0 ? sqrt(d2 / r2) : 1e9;
    printf("  %-58s vs no window: rel %.3g%s\n", what, e, e <= tol ? "" : "  FAIL");
    if (!(e <= tol)) bad++;
    nn_free(da); nn_free(db); free(ha); free(hb);
}

/* ---- stride-1 3^3 conv forward of silu(gn(x)) with output GroupNorm statistics (MX-fp4 in / out) */
typedef struct { shape5 xs; int cout, Gi, Go; float *gam, *bet, *mean, *rstd, *w, *b; void *x; } fwd_t;
static void fwd_run(const fwd_t *f, int z0, int z1, int D, unsigned ctr, res_t *r, int first, int last) {
    nn_set_zwin(z0, z1, D, first, last); nn_sr_set_ctr(ctr);
    if (nn_conv3d_fwd_gn_stats((const float *)f->x, f->xs, f->Gi, f->gam, f->bet, f->mean, f->rstd, f->w, f->b, f->cout, (float *)r->out, f->Go, 1e-5f, r->st, r->st + f->xs.n * f->Go) < 0) {
        printf("  fwd_gn_stats unsupported  FAIL\n"); bad++;
    }
    nn_set_zwin(0, 0, 0, 0, 0);
}
static void test_fwd(int C, int cout, int D) {
    mode_prod();
    fwd_t f = {{1, C, D, 12, 20}, cout, C / 4, cout / 4};
    const size_t n = shape_numel(f.xs);
    float *x = dev_rand(n, 1.5f); f.x = mx4_from(x, f.xs); nn_free(x);
    f.gam = dev_rand(C, 1.f); f.bet = dev_rand(C, 0.5f); f.mean = dev_rand(f.Gi, 0.2f); f.rstd = dev_rand(f.Gi, 0.3f);
    { float h[64]; nn_d2h(h, f.rstd, f.Gi * 4); for (int i = 0; i < f.Gi; i++) h[i] = 0.8f + fabsf(h[i]); nn_h2d(f.rstd, h, f.Gi * 4); }
    f.w = dev_rand((size_t)cout * C * 27, 0.12f); f.b = dev_rand(cout, 0.1f);
    const shape5 ys = {1, cout, D, 12, 20};
    nn_set_sr_step(31); const unsigned c0 = nn_sr_ctr();
    res_t a = {"", nn_mx4_bytes(ys), mx4_new(ys), nullptr, 0, dev_zero(2 * f.Go), (size_t)2 * f.Go};
    fwd_run(&f, 0, D, D, c0, &a, 1, 1);
    for (int S = 8; S <= 16; S *= 2) {
        res_t b = a; b.out = mx4_new(ys); b.st = dev_zero(2 * f.Go);
        int z0[64], z1[64]; const int k = windows(D, S, z0, z1);
        for (int i = 0; i < k; i++) fwd_run(&f, z0[i], z1[i], D, c0, &b, i == 0, i == k - 1);
        char m[96]; snprintf(m, sizeof m, "fwd gn %2d -> %2d ch, depth %d", C, cout, D); cmp(m, S, &a, &b);
        nn_free(b.out); nn_free(b.st);
    }
    {   char m[96]; snprintf(m, sizeof m, "fwd gn %2d -> %2d ch, depth %d", C, cout, D);
        res_t b = a; b.out = mx4_new(ys); b.st = dev_zero(2 * f.Go);
        fwd_run(&f, 8, 16, D, c0, &b, 1, 1); cover(m, a.out, b.out, ys, 8, 16);
        nn_zero(b.out, a.nb); nn_sr_set_ctr(c0);
        nn_conv3d_fwd_gn_stats((const float *)f.x, f.xs, f.Gi, f.gam, f.bet, f.mean, f.rstd, f.w, f.b, f.cout, (float *)b.out, f.Go, 1e-5f, b.st, b.st + f.Go);
        vs_plain(m, a.out, b.out, ys, C <= 16 ? 0.3 : 0);
        nn_free(b.out); nn_free(b.st); }
    nn_free(a.out); nn_free(a.st); nn_free(f.x); nn_free(f.gam); nn_free(f.bet); nn_free(f.mean); nn_free(f.rstd); nn_free(f.w); nn_free(f.b);
    mode_off();
}

/* ---- stride-1 3^3 backward-data, MX-fp4 gy -> MX-fp4 gx (exact output SR keyed by the op's seed); o_split: split output */
typedef struct { shape5 ys, xs; float *w; void *gy; int o_split; } bwd_t;
static void bwd_run(const bwd_t *f, int z0, int z1, int D, unsigned ctr, res_t *r) {
    nn_set_zwin(z0, z1, D, 1, 1); nn_sr_set_ctr(ctr);
    float *scr = nn_malloc(nn_conv3d_scratch(f->xs, f->ys.c, 3) + 4096);
    if (f->o_split) {
        if (nn_conv3d_bwd_data_split((const float *)f->gy, f->ys, f->w, f->xs, (float *)r->out, (float *)r->out2, f->o_split, scr) < 0) { printf("  bwd_data_split unsupported  FAIL\n"); bad++; }
    } else nn_conv3d_bwd_data((const float *)f->gy, f->ys, f->w, f->xs, 3, 1, (float *)r->out, scr);
    nn_free(scr);
    nn_set_zwin(0, 0, 0, 0, 0);
}
static void test_bwd(int Cy, int Cx, int o_split, int D) {
    mode_prod();
    bwd_t f = {{1, Cy, D, 12, 20}, {1, Cx, D, 12, 20}};
    f.o_split = o_split;
    float *g = dev_rand(shape_numel(f.ys), 1e-2f); f.gy = mx4_from(g, f.ys); nn_free(g);
    f.w = dev_rand((size_t)Cy * Cx * 27, 0.12f);
    const shape5 x1 = {1, o_split ? o_split : Cx, D, 12, 20}, x2 = {1, Cx - o_split, D, 12, 20};
    nn_set_sr_step(57); const unsigned c0 = nn_sr_ctr();
    res_t a = {"", nn_mx4_bytes(x1), mx4_new(x1), o_split ? mx4_new(x2) : nullptr, o_split ? nn_mx4_bytes(x2) : 0};
    bwd_run(&f, 0, D, D, c0, &a);
    for (int S = 8; S <= 16; S *= 2) {
        res_t b = a; b.out = mx4_new(x1); b.out2 = o_split ? mx4_new(x2) : nullptr;
        int z0[64], z1[64]; const int k = windows(D, S, z0, z1);
        for (int i = 0; i < k; i++) bwd_run(&f, z0[i], z1[i], D, c0, &b);
        char m[96]; snprintf(m, sizeof m, "bwd-data %2d -> %2d ch%s, depth %d", Cy, Cx, o_split ? " (16 | 16 split)" : "", D); cmp(m, S, &a, &b);
        nn_free(b.out); if (b.out2) nn_free(b.out2);
    }
    {   char m[96]; snprintf(m, sizeof m, "bwd-data %2d -> %2d ch%s, depth %d", Cy, Cx, o_split ? " (split)" : "", D);
        res_t b = a; b.out = mx4_new(x1); b.out2 = o_split ? mx4_new(x2) : nullptr;
        bwd_run(&f, 16, 24, D, c0, &b); cover(m, a.out, b.out, x1, 16, 24);
        nn_zero(b.out, a.nb); nn_sr_set_ctr(c0);
        float *scr = nn_malloc(nn_conv3d_scratch(f.xs, f.ys.c, 3) + 4096);
        if (o_split) nn_conv3d_bwd_data_split((const float *)f.gy, f.ys, f.w, f.xs, (float *)b.out, (float *)b.out2, o_split, scr);
        else nn_conv3d_bwd_data((const float *)f.gy, f.ys, f.w, f.xs, 3, 1, (float *)b.out, scr);
        nn_free(scr);
        vs_plain(m, a.out, b.out, x1, Cy <= 16 ? 0.3 : 0);
        nn_free(b.out); if (b.out2) nn_free(b.out2); }
    nn_free(a.out); if (a.out2) nn_free(a.out2); nn_free(f.gy); nn_free(f.w);
    mode_off();
}

/* ---- generic: op(ctx, r) under each window; outputs (byte compare), statistics, weight gradients */
typedef void (*op_t)(void *ctx, res_t *r);
static void run_op(op_t op, void *ctx, int z0, int z1, int D, unsigned ctr, res_t *r, int first, int last) {
    nn_set_zwin(z0, z1, D, first, last); nn_sr_set_ctr(ctr); op(ctx, r); nn_set_zwin(0, 0, 0, 0, 0);
}
static res_t res_like(const res_t *a, shape5 so) {   /* fresh zeroed buffers shaped like a's */
    res_t b = *a;
    if (a->out) b.out = mx4_new(so);
    if (a->gw) b.gw = dev_zero(a->ngw);
    if (a->st) b.st = dev_zero(a->nst);
    return b;
}
static void res_free(res_t *r) { if (r->out) nn_free(r->out); if (r->gw) nn_free(r->gw); if (r->st) nn_free(r->st); }
/* windows of S level-0 planes (D0 deep) == one window [0, D0); so: the output shape (coverage: a lone window [8, 16) in its planes) */
static void drive(const char *what, op_t op, void *ctx, int D0, res_t *a, shape5 so, int cover_ok) {
    nn_set_sr_step(91); const unsigned c0 = nn_sr_ctr();
    run_op(op, ctx, 0, D0, D0, c0, a, 1, 1);
    for (int S = 8; S <= 16; S *= 2) {
        res_t b = res_like(a, so);
        int z0[64], z1[64]; const int k = windows(D0, S, z0, z1);
        for (int i = 0; i < k; i++) run_op(op, ctx, z0[i], z1[i], D0, c0, &b, i == 0, i == k - 1);
        cmp(what, S, a, &b); res_free(&b);
    }
    if (cover_ok && a->out) {
        res_t b = res_like(a, so);
        run_op(op, ctx, 8, 16, D0, c0, &b, 1, 1);
        cover(what, a->out, b.out, so, 8 * so.d / D0, 16 * so.d / D0);
        res_free(&b);
    }
}
/* stem: the 4-channel network input (MX-fp4) -> 16 channels with output statistics (fp8 compute) */
typedef struct { shape5 xs; int cout, Go; void *x; float *w, *b; } stem_t;
static void stem_op(void *c, res_t *r) {
    stem_t *f = c;
    if (nn_conv3d_fwd_gn_stats((const float *)f->x, f->xs, 0, nullptr, nullptr, nullptr, nullptr, f->w, f->b, f->cout, (float *)r->out, f->Go, 1e-5f, r->st, r->st + f->Go) < 0) { printf("  stem unsupported  FAIL\n"); bad++; }
}
static void test_stem(int D) {
    mode_prod();
    stem_t f = {{1, 4, D, 12, 20}, 16, 4};
    float *x = dev_rand(shape_numel(f.xs), 1.5f); f.x = mx4_from(x, f.xs); nn_free(x);
    f.w = dev_rand((size_t)16 * 4 * 27, 0.2f); f.b = dev_rand(16, 0.1f);
    const shape5 ys = {1, 16, D, 12, 20};
    res_t a = {"", nn_mx4_bytes(ys), mx4_new(ys), nullptr, 0, dev_zero(2 * f.Go), (size_t)2 * f.Go};
    char m[96]; snprintf(m, sizeof m, "stem 4 -> 16 ch, depth %d", D); drive(m, stem_op, &f, D, &a, ys, 1);
    res_free(&a); nn_free(f.x); nn_free(f.w); nn_free(f.b); mode_off();
}
/* down0: stride-2 forward of silu(gn(x)) (level 0 -> level 1) and its weight gradient */
typedef struct { shape5 xs, ys; nn_gn_t g; void *x, *gy; float *w, *b; } down_t;
static void down_fwd_op(void *c, res_t *r) {
    down_t *f = c;
    if (nn_conv3d_fwd_x((const float *)f->x, &f->g, nullptr, nullptr, 0, 0, f->xs, f->w, f->b, f->ys.c, 3, 2, (float *)r->out, 0, 1e-5f, nullptr, nullptr) < 0) { printf("  down fwd unsupported  FAIL\n"); bad++; }
}
static void down_wgrad_op(void *c, res_t *r) {
    down_t *f = c;
    if (nn_conv3d_bwd_weight_x((const float *)f->x, &f->g, nullptr, nullptr, 0, 0, f->xs, (const float *)f->gy, f->ys, 3, 2, r->gw, r->gw + (size_t)f->ys.c * f->xs.c * 27) < 0) { printf("  down wgrad unsupported  FAIL\n"); bad++; }
}
static void test_down(int C, int Co, int D) {
    mode_prod();
    down_t f = {{1, C, D, 12, 32}, {1, Co, D / 2, 6, 16}};
    float *x = dev_rand(shape_numel(f.xs), 1.5f); f.x = mx4_from(x, f.xs); nn_free(x);
    float *gy = dev_rand(shape_numel(f.ys), 1e-2f); f.gy = mx4_from(gy, f.ys); nn_free(gy);
    const int G = C / 4;
    float *gam = dev_rand(C, 1.f), *bet = dev_rand(C, 0.5f), *mean = dev_rand(G, 0.2f), *rstd = dev_rand(G, 0.3f);
    { float h[64]; nn_d2h(h, rstd, G * 4); for (int i = 0; i < G; i++) h[i] = 0.8f + fabsf(h[i]); nn_h2d(rstd, h, G * 4); }
    f.g = (nn_gn_t){gam, bet, mean, rstd, G};
    f.w = dev_rand((size_t)Co * C * 27, 0.12f); f.b = dev_rand(Co, 0.1f);
    char m[96];
    res_t a = {"", nn_mx4_bytes(f.ys), mx4_new(f.ys)};
    snprintf(m, sizeof m, "down0 fwd %2d -> %2d ch, depth %d", C, Co, D); drive(m, down_fwd_op, &f, D, &a, f.ys, 1); res_free(&a);
    res_t w = {"", 0, nullptr, nullptr, 0, nullptr, 0, dev_zero((size_t)Co * C * 27 + Co), (size_t)Co * C * 27 + Co};
    snprintf(m, sizeof m, "down0 wgrad %2d -> %2d ch, depth %d", C, Co, D); drive(m, down_wgrad_op, &f, D, &w, f.ys, 0); res_free(&w);
    nn_free(f.x); nn_free(f.gy); nn_free(gam); nn_free(bet); nn_free(mean); nn_free(rstd); nn_free(f.w); nn_free(f.b); mode_off();
}

/* stride-1 weight gradients: x (MX-fp4, optionally silu(gn(x))) with an MX-fp4 gy (the fp4 kernel's gy pre-pass; 4-channel stem: fp8) */
typedef struct { shape5 xs, ys; nn_gn_t g; int use_gn; void *x, *gy; } wg_t;
static void wg_op(void *c, res_t *r) {
    wg_t *f = c;
    if (nn_conv3d_bwd_weight_x((const float *)f->x, f->use_gn ? &f->g : nullptr, nullptr, nullptr, 0, 0, f->xs, (const float *)f->gy, f->ys, 3, 1, r->gw, r->gw + (size_t)f->ys.c * f->xs.c * 27) < 0) {
        printf("  wgrad unsupported  FAIL\n"); bad++;
    }
}
static void test_wgrad(int C, int Co, int D, int use_gn) {
    mode_prod();
    wg_t f = {{1, C, D, 12, 20}, {1, Co, D, 12, 20}};
    f.use_gn = use_gn;
    float *x = dev_rand(shape_numel(f.xs), 1.5f); f.x = mx4_from(x, f.xs); nn_free(x);
    float *gy = dev_rand(shape_numel(f.ys), 1e-2f); f.gy = mx4_from(gy, f.ys); nn_free(gy);
    const int G = C >= 4 ? C / 4 : 1;
    float *gam = dev_rand(C, 1.f), *bet = dev_rand(C, 0.5f), *mean = dev_rand(G, 0.2f), *rstd = dev_rand(G, 0.3f);
    { float h[64]; nn_d2h(h, rstd, G * 4); for (int i = 0; i < G; i++) h[i] = 0.8f + fabsf(h[i]); nn_h2d(rstd, h, G * 4); }
    f.g = (nn_gn_t){gam, bet, mean, rstd, G};
    res_t w = {"", 0, nullptr, nullptr, 0, nullptr, 0, dev_zero((size_t)Co * C * 27 + Co), (size_t)Co * C * 27 + Co};
    char m[96]; snprintf(m, sizeof m, "wgrad%s %2d -> %2d ch, depth %d", use_gn ? " gn" : "", C, Co, D);
    drive(m, wg_op, &f, D, &w, f.ys, 0); res_free(&w);
    nn_free(f.x); nn_free(f.gy); nn_free(gam); nn_free(bet); nn_free(mean); nn_free(rstd); mode_off();
}

int main(void) {
    setenv("UFSM_F4_WGRAD", "1", 1);   /* the production fp4 weight gradient (train --fp4 2) */
    nn_init(0);
    printf("z-window launches == one launch (MX-fp4, SR):\n");
    for (int D = 32; D <= 36; D += 4) {   /* 36: a partial last window */
        test_fwd(16, 16, D); test_fwd(32, 32, D);
        test_bwd(16, 16, 0, D); test_bwd(32, 32, 0, D); test_bwd(16, 32, 16, D);
        test_stem(D); test_down(16, 16, D); test_down(32, 32, D);
        test_wgrad(16, 16, D, 1); test_wgrad(32, 32, D, 1); test_wgrad(4, 16, D, 0);
    }
    const char *e = nn_check(); if (e) { printf("  CUDA: %s  FAIL\n", e); bad++; }
    printf("test_zwin: %s\n", bad ? "FAIL" : "PASS");
    return bad != 0;
}
