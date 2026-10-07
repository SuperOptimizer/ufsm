/* The level-0 fp4 3^3 convolutions of training in their storage formats, timed in isolation (min over repetitions) with an
   FNV hash of the output bytes (equal hashes = identical results across builds / variants):
     fwd   MX-fp4 in -> MX-fp4 out, GN+SiLU of the input in staging, GN statistics of the output (enc0.c2 / dec0.c2, dec0.c1)
     bdata MX-fp8 gy -> MX-fp8 gx with stochastic rounding of gy (backward-data), 16 -> 16 and 16 -> 32 + 16 (dec0.c1 split)
   usage: bench_f4conv [P] */
#include "nn.h"
#include "nn_lp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static float *rnd(size_t n, float sc, uint64_t r) {
    float *h = malloc(n * 4);
    for (size_t i = 0; i < n; i++) { r ^= r << 13; r ^= r >> 7; r ^= r << 17; h[i] = ((float)(r >> 40) / 16777216.f - 0.5f) * sc; }
    float *d = nn_malloc(n * 4); nn_h2d(d, h, n * 4); free(h); return d;
}
static void *mx(int bits, int C, size_t S, float sc, uint64_t seed) {
    float *f = rnd(S * C, sc, seed);
    void *q = nn_malloc(bits == 4 ? lp_mx4_bytes(1, C, S) : lp_mx8_bytes(1, C, S));
    if (bits == 4) lp_f32_to_mx4(f, 1, C, S, q); else lp_f32_to_mx8(f, 1, C, S, q);
    nn_free(f); return q;
}
static uint64_t fnv(const void *d, size_t n) {
    uint8_t *h = malloc(n); nn_d2h(h, d, n);
    uint64_t x = 1469598103934665603ull; for (size_t i = 0; i < n; i++) { x ^= h[i]; x *= 1099511628211ull; }
    free(h); return x;
}
typedef struct { const void *x; void *y, *y2; int xb, ci, co, osplit, sr; shape5 xs; const float *w, *b; gnp_t gp; double *osum; int Go; } job;
static void run(job *j, unsigned it) {
    split_t sp = {0};
    if (j->y2) { sp.y2 = j->y2; sp.o_split = j->osplit; }
    if (j->sr) sp.sr = 0x2545f491u + it;
    if (j->osum) nn_zero(j->osum, 2 * 64 * sizeof(double));
    lp_conv_fwd_f4(j->x, j->xb, j->xs, j->w, j->b, j->co, j->y, j->xb, j->gp, j->osum, j->Go, sp);
}
/* BENCH_ONLY: comma list of sections to run (fwd, upfwd, f8fwd, wgrad, up, gnbwd, f8w, w32, s2b, up2; default all); BENCH_WL: one wgrad case index */
static int want(const char *key) {
    const char *o = getenv("BENCH_ONLY");
    if (!o) return 1;
    const size_t k = strlen(key);
    for (const char *p = o; (p = strstr(p, key)); p += k) if ((p == o || p[-1] == ',') && (p[k] == 0 || p[k] == ',')) return 1;
    return 0;
}
int main(int argc, char **argv) {
    const int P = argc > 1 ? atoi(argv[1]) : 384, G = 8;
    nn_init(0); nn_set_f16(1);
    const size_t S = (size_t)P * P * P;
    float *gam = rnd(256, 1.f, 4), *bet = rnd(256, 0.5f, 5), *mean = rnd(G, 0.1f, 6), *rstd = rnd(G, 0.1f, 7);
    double *osum = nn_malloc(2 * 64 * sizeof(double));
    struct { const char *nm; int xb, ci, co, osplit, gn, stats, sr; } L[] = {
        {"fwd 16 -> 16 (gn+silu, stats)", 4, 16, 16, 0, 1, 1, 0}, {"fwd 48 -> 16 (gn+silu, stats)", 4, 48, 16, 0, 1, 1, 0},
        {"bdata 16 -> 16 (mx8, SR)", 3, 16, 16, 0, 0, 0, 1}, {"bdata 16 -> 32 + 16 (mx8, SR)", 3, 16, 48, 32, 0, 0, 1}};
    for (int l = 0; l < 4 && want("fwd"); l++) {
        shape5 xs = {1, L[l].ci, P, P, P};
        job j = {0};
        j.x = mx(L[l].xb == 4 ? 4 : 8, L[l].ci, S, L[l].xb == 4 ? 2.f : 1e-3f, 11 + l); j.xb = L[l].xb; j.ci = L[l].ci; j.co = L[l].co; j.xs = xs; j.sr = L[l].sr;
        const int c1 = L[l].osplit ? L[l].osplit : L[l].co, c2 = L[l].co - c1, bits = L[l].xb == 4 ? 4 : 8;
        const size_t by1 = bits == 4 ? lp_mx4_bytes(1, c1, S) : lp_mx8_bytes(1, c1, S), by2 = c2 ? (bits == 4 ? lp_mx4_bytes(1, c2, S) : lp_mx8_bytes(1, c2, S)) : 0;
        j.y = nn_malloc(by1); j.y2 = c2 ? nn_malloc(by2) : nullptr; j.osplit = L[l].osplit;
        float *w = rnd((size_t)L[l].co * L[l].ci * 27, 0.1f, 2), *b = L[l].osplit || L[l].sr ? nullptr : rnd(L[l].co, 0.1f, 3);
        j.w = w; j.b = b;
        if (L[l].gn) j.gp = (gnp_t){gam, bet, mean, rstd, G};
        if (L[l].stats) { j.osum = osum; j.Go = G; }
        double best = 1e9;
        for (unsigned it = 0; it < 12; it++) {
            nn_sync(); double t0 = now(); run(&j, it); nn_sync(); double t = (now() - t0) * 1e3;
            if (it >= 2 && t < best) best = t;
        }
        run(&j, 0); nn_sync();
        const uint64_t h = fnv(j.y, by1) ^ (c2 ? fnv(j.y2, by2) * 31 : 0);
        double hs[2] = {0, 0}; if (L[l].stats) nn_d2h(hs, osum, sizeof hs);
        printf("%-32s @%d %8.3f ms  out %016llx", L[l].nm, P, best, (unsigned long long)h);
        if (L[l].stats) printf("  stats %.9g %.9g", hs[0], hs[1]);
        printf("\n");
        nn_free((void *)j.x); nn_free(j.y); if (j.y2) nn_free(j.y2); nn_free(w); if (b) nn_free(b);
    }
    if (want("upfwd")) {   /* decoder conv1 forward as dec0.c1 trains: [up2(silu(gn(coarse a2))) | silu(gn(skip a2))] staged in the kernel,
                              MX-fp4 out with GN statistics (32 + 16 -> 16 at P) */
        const int Pc = P / 2; const size_t Sc = (size_t)Pc * Pc * Pc;
        shape5 xs = {1, 48, P, P, P};
        void *xc = mx(4, 32, Sc, 2.f, 91), *xk = mx(4, 16, S, 2.f, 92), *y = nn_malloc(lp_mx4_bytes(1, 16, S));
        float *w = rnd((size_t)16 * 48 * 27, 0.1f, 93), *b = rnd(16, 0.1f, 94);
        double best = 1e9;
        for (int it = 0; it < 10; it++) {
            split_t sp = {0}; sp.up = 1; sp.x2 = xk; sp.c_split = 32; sp.gp2 = (gnp_t){gam, bet, mean, rstd, G}; sp.stored_stats = 1;
            nn_zero(osum, 2 * 64 * sizeof(double));
            nn_sync(); double t0 = now(); lp_conv_fwd_f4(xc, 4, xs, w, b, 16, y, 4, (gnp_t){gam + 64, bet + 64, mean, rstd, G}, osum, G, sp); nn_sync();
            if (it >= 2 && (now() - t0) * 1e3 < best) best = (now() - t0) * 1e3;
        }
        printf("%-32s @%d %8.3f ms  out %016llx\n", "fwd up 32 + 16 -> 16 (dec0.c1)", P, best, (unsigned long long)fnv(y, lp_mx4_bytes(1, 16, S)));
        nn_free(xc); nn_free(xk); nn_free(y); nn_free(w); nn_free(b);
    }
    if (want("f8fwd")) {   /* the same 16 -> 16 / 48 -> 16 forwards on the fp8 kernel (MX-fp4 storage, fp8 compute) */
        for (int ci = 16; ci <= 48; ci += 32) {
            shape5 xs = {1, ci, P, P, P}; void *x = mx(4, ci, S, 2.f, 61 + ci), *y = nn_malloc(lp_mx4_bytes(1, 16, S));
            float *w = rnd((size_t)16 * ci * 27, 0.1f, 62), *b = rnd(16, 0.1f, 63);
            const gnp_t gp = {gam, bet, mean, rstd, G};
            double best = 1e9;
            for (int it = 0; it < 10; it++) {
                nn_zero(osum, 2 * 64 * sizeof(double));
                nn_sync(); double t0 = now(); lp_conv_fwd_f8(x, 4, xs, w, b, 16, y, 4, gp, osum, G, (split_t){0}); nn_sync(); if (it >= 2 && (now() - t0) * 1e3 < best) best = (now() - t0) * 1e3;
            }
            printf("fwd %2d -> 16 fp8 compute (gn+silu, stats) @%d %8.3f ms\n", ci, P, best);
            nn_free(x); nn_free(y); nn_free(w); nn_free(b);
        }
    }
    /* weight gradient (fp4, the --fp4 2 recipe): MX-fp4 x with GN+SiLU in staging, MX-fp8 gy with SR, bias */
    struct { const char *nm; int ci, co, dv; } WL[] = {{"wgrad 16 -> 16 (gn+silu, SR)", 16, 16, 1}, {"wgrad 48 -> 16 (gn+silu, SR)", 48, 16, 1}, {"wgrad 32 -> 32 (gn+silu, SR) @P/2", 32, 32, 2},
                                                  {"wgrad 128 -> 32 (gn+silu, SR) @P/2", 128, 32, 2}, {"wgrad 96 -> 96 (gn+silu, SR) @P/4", 96, 96, 4},
                                                  {"wgrad 224 -> 96 (gn+silu, SR) @P/4", 224, 96, 4}, {"wgrad 128 -> 128 (gn+silu, SR) @P/8", 128, 128, 8}};
    if (getenv("BENCH_GYPRE")) lp_set_f4w_gypre_kb(atoi(getenv("BENCH_GYPRE")));   /* KiB > 0: force the gy pre-pass (also 16 -> 16) */
    for (int l = 0; l < (int)(sizeof WL / sizeof WL[0]) && want("wgrad"); l++) {
        if (getenv("BENCH_WL") && l != atoi(getenv("BENCH_WL"))) continue;
        const int Pl = P / WL[l].dv; const size_t Sl = (size_t)Pl * Pl * Pl;
        shape5 xs = {1, WL[l].ci, Pl, Pl, Pl}, ys = xs; ys.c = WL[l].co;
        void *x = mx(4, WL[l].ci, Sl, 2.f, 21 + l), *gy = mx(8, WL[l].co, Sl, 1e-3f, 31 + l);
        const size_t nw = (size_t)WL[l].co * WL[l].ci * 27;
        float *gw = nn_malloc(nw * 4), *gb = nn_malloc(WL[l].co * 4);
        const gnp_t gp = {gam, bet, mean, rstd, G};
        double best = 1e9;
        for (unsigned it = 0; it < 10; it++) {
            split_t sp = {0}; sp.sr = 0x51ed270bu + it;
            nn_zero(gw, nw * 4); nn_zero(gb, WL[l].co * 4);
            nn_sync(); double t0 = now(); lp_bwd_w_f4(x, 4, xs, gy, 3, ys, gw, gb, gp, sp, 0); nn_sync(); double t = (now() - t0) * 1e3;
            if (it >= 2 && t < best) best = t;
        }
        float *h = malloc(nw * 4); nn_d2h(h, gw, nw * 4); double a = 0; for (size_t i = 0; i < nw; i++) a += fabs(h[i]); free(h);
        printf("%-32s @%d %8.3f ms  |gw| %.9g\n", WL[l].nm, Pl, best, a);
        nn_free(x); nn_free(gy); nn_free(gw); nn_free(gb);
    }
    if (want("up")) {   /* decoder conv1 weight gradients: [up2(coarse s2) | silu(gn(skip))] staged inside the kernel (sp.up), as dec0.c1 / dec1.c1 train */
        struct { const char *nm; int cu, cs, co, dv; } UL[] = {{"wgrad up 32 + 16 -> 16 (dec0.c1)", 32, 16, 16, 1}, {"wgrad up 96 + 32 -> 32 (dec1.c1) @P/2", 96, 32, 32, 2}};
        for (int l = 0; l < 2; l++) {
            const int Pl = P / UL[l].dv; const size_t Sl = (size_t)Pl * Pl * Pl, Sc = Sl / 8;
            shape5 xs = {1, UL[l].cu + UL[l].cs, Pl, Pl, Pl}, ys = xs; ys.c = UL[l].co;
            void *xc = mx(4, UL[l].cu, Sc, 2.f, 71 + l), *xk = mx(4, UL[l].cs, Sl, 2.f, 73 + l), *gy = mx(8, UL[l].co, Sl, 1e-3f, 75 + l);
            const size_t nw = (size_t)UL[l].co * xs.c * 27;
            float *gw = nn_malloc(nw * 4), *gb = nn_malloc(UL[l].co * 4);
            double best = 1e9;
            for (unsigned it = 0; it < 10; it++) {
                split_t sp = {0}; sp.sr = 0x51ed270bu + it; sp.up = 1; sp.x2 = xk; sp.c_split = UL[l].cu; sp.gp2 = (gnp_t){gam, bet, mean, rstd, G};
                nn_zero(gw, nw * 4); nn_zero(gb, UL[l].co * 4);
                nn_sync(); double t0 = now(); lp_bwd_w_f4(xc, 4, xs, gy, 3, ys, gw, gb, (gnp_t){0}, sp, 0); nn_sync(); double t = (now() - t0) * 1e3;
                if (it >= 2 && t < best) best = t;
            }
            float *h = malloc(nw * 4); nn_d2h(h, gw, nw * 4); double a = 0; for (size_t i = 0; i < nw; i++) a += fabs(h[i]); free(h);
            printf("%-32s @%d %8.3f ms  |gw| %.9g\n", UL[l].nm, Pl, best, a);
            nn_free(xc); nn_free(xk); nn_free(gy); nn_free(gw); nn_free(gb);
        }
    }
    if (want("gnbwd")) {   /* GroupNorm+SiLU backward on the training storage: MX-fp4 x, MX-fp8 gy / gx (16 ch at P, 32 at P / 2) */
        for (int k = 0; k < 2; k++) {
            const int C = k ? 32 : 16, Pl = k ? P / 2 : P; const size_t Sl = (size_t)Pl * Pl * Pl;
            shape5 s = {1, C, Pl, Pl, Pl};
            void *x = mx(4, C, Sl, 2.f, 81 + k), *gy = mx(8, C, Sl, 1e-3f, 83 + k);
            const size_t b8 = lp_mx8_bytes(1, C, Sl);
            void *gx = nn_malloc(b8);
            nn_set_storage(x, lp_mx4_bytes(1, C, Sl), 4); nn_set_storage(gy, b8, 8); nn_set_storage(gx, b8, 8);
            float *gm = rnd(C, 1.f, 85), *bt = rnd(C, 0.5f, 86), *gg = nn_malloc(C * 4), *gb2 = nn_malloc(C * 4), *scr = nn_malloc(nn_gn_scratch(s) + 4096);
            nn_set_prec(3);
            double best = 1e9;
            for (int it = 0; it < 10; it++) {
                nn_sync(); double t0 = now(); nn_gn_silu_bwd(x, s, G, gm, bt, mean, rstd, gy, gx, gg, gb2, scr); nn_sync();
                if (it >= 2 && (now() - t0) * 1e3 < best) best = (now() - t0) * 1e3;
            }
            printf("%-32s @%d %8.3f ms  out %016llx\n", k ? "gn+silu bwd 32 ch (mx4 x, mx8 g) @P/2" : "gn+silu bwd 16 ch (mx4 x, mx8 g)", Pl, best, (unsigned long long)fnv(gx, b8));
            nn_storage_forget(x); nn_storage_forget(gy); nn_storage_forget(gx);
            nn_free(x); nn_free(gy); nn_free(gx); nn_free(gm); nn_free(bt); nn_free(gg); nn_free(gb2); nn_free(scr);
        }
    }
    if (want("f8w")) {   /* fp8 weight gradients at level 0: down0 (stride 2, MX-fp4 x level 0 -> MX-fp8 gy level 1, 16 -> 16) and the stem
           (enc0.c1: MX-fp8 4-channel input, MX-fp8 gy, SR) */
        const int Pc = P / 2; const size_t Sc = (size_t)Pc * Pc * Pc;
        shape5 xs = {1, 16, P, P, P}, ys = {1, 16, Pc, Pc, Pc};
        void *x = mx(4, 16, S, 2.f, 51), *gy = mx(8, 16, Sc, 1e-3f, 52);
        float *gw = nn_malloc(16 * 16 * 27 * 4), *gb = nn_malloc(16 * 4);
        double best = 1e9;
        for (int it = 0; it < 10; it++) {
            nn_zero(gw, 16 * 16 * 27 * 4); nn_zero(gb, 64);
            nn_sync(); double t0 = now(); lp_bwd_w_s2_f8(x, 4, xs, gy, 3, ys, gw, gb, (gnp_t){0}); nn_sync(); if (it >= 2 && (now() - t0) * 1e3 < best) best = (now() - t0) * 1e3;
        }
        float h[16 * 16 * 27]; nn_d2h(h, gw, sizeof h); double a = 0; for (int i = 0; i < 16 * 16 * 27; i++) a += fabs(h[i]);
        printf("%-32s @%d %8.3f ms  |gw| %.9g\n", "wgrad s2 down0 16 -> 16 (fp8)", P, best, a);
        {   /* down2-like: 96 -> 96 at P / 4 (MX-fp4 x, MX-fp8 gy) */
            const int Pf = P / 4, Pq = P / 8; const size_t Sf = (size_t)Pf * Pf * Pf, Sq = (size_t)Pq * Pq * Pq, nw = (size_t)96 * 96 * 27;
            shape5 xs2 = {1, 96, Pf, Pf, Pf}, ys2 = {1, 96, Pq, Pq, Pq};
            void *x2 = mx(4, 96, Sf, 2.f, 53), *gy2 = mx(8, 96, Sq, 1e-3f, 54);
            float *gw2 = nn_malloc(nw * 4), *gb2 = nn_malloc(96 * 4);
            double b2 = 1e9;
            for (int it = 0; it < 10; it++) {
                nn_zero(gw2, nw * 4); nn_zero(gb2, 96 * 4);
                nn_sync(); double t0 = now(); lp_bwd_w_s2_f8(x2, 4, xs2, gy2, 3, ys2, gw2, gb2, (gnp_t){0}); nn_sync(); if (it >= 2 && (now() - t0) * 1e3 < b2) b2 = (now() - t0) * 1e3;
            }
            float *h2 = malloc(nw * 4); nn_d2h(h2, gw2, nw * 4); double a2 = 0; for (size_t i = 0; i < nw; i++) a2 += fabs(h2[i]); free(h2);
            printf("%-32s @%d %8.3f ms  |gw| %.9g\n", "wgrad s2 down2 96 -> 96 (fp8)", Pf, b2, a2);
            nn_free(x2); nn_free(gy2); nn_free(gw2); nn_free(gb2);
        }
        nn_free(x); nn_free(gy);
        shape5 x4 = {1, 4, P, P, P}, y4 = {1, 16, P, P, P};
        void *xi = mx(8, 4, S, 2.f, 53), *g4 = mx(8, 16, S, 1e-3f, 54);
        best = 1e9;
        for (unsigned it = 0; it < 10; it++) {
            split_t sp = {0}; sp.sr = 0x6d2b79f5u + it;
            nn_zero(gw, 16 * 4 * 27 * 4); nn_zero(gb, 64);
            nn_sync(); double t0 = now(); lp_bwd_w_f8(xi, 3, x4, g4, 3, y4, gw, gb, (gnp_t){0}, sp); nn_sync(); if (it >= 2 && (now() - t0) * 1e3 < best) best = (now() - t0) * 1e3;
        }
        nn_d2h(h, gw, 16 * 4 * 27 * 4); a = 0; for (int i = 0; i < 16 * 4 * 27; i++) a += fabs(h[i]);
        printf("%-32s @%d %8.3f ms  |gw| %.9g\n", "wgrad stem 4 -> 16 (fp8, SR)", P, best, a);
        nn_free(xi); nn_free(g4); nn_free(gw); nn_free(gb);
    }
    if (want("w32")) {   /* level 0 of the 32,64,... net: stem 4 -> 32 wgrad, down0 32 -> 32 stride-2 wgrad / bdata, head 32 -> 7 wgrad */
        const int Pc = P / 2; const size_t Sc = (size_t)Pc * Pc * Pc;
        double best;
        {
            shape5 x4 = {1, 4, P, P, P}, y32 = {1, 32, P, P, P};
            void *xi = mx(8, 4, S, 2.f, 101), *g = mx(8, 32, S, 1e-3f, 102);
            float *gw = nn_malloc(32 * 4 * 27 * 4), *gb = nn_malloc(32 * 4);
            best = 1e9;
            for (unsigned it = 0; it < 8; it++) {
                split_t sp = {0}; sp.sr = 0x6d2b79f5u + it;
                nn_zero(gw, 32 * 4 * 27 * 4); nn_zero(gb, 32 * 4);
                nn_sync(); double t0 = now(); lp_bwd_w_f8(xi, 3, x4, g, 3, y32, gw, gb, (gnp_t){0}, sp); nn_sync(); if (it >= 2 && (now() - t0) * 1e3 < best) best = (now() - t0) * 1e3;
            }
            printf("%-32s @%d %8.3f ms\n", "wgrad stem 4 -> 32 (fp8, SR)", P, best);
            nn_free(xi); nn_free(g); nn_free(gw); nn_free(gb);
        }
        {
            shape5 xs = {1, 32, P, P, P}, ys = {1, 32, Pc, Pc, Pc};
            void *x = mx(4, 32, S, 2.f, 103), *gy = mx(8, 32, Sc, 1e-3f, 104), *gx = mx(8, 32, S, 1e-3f, 105);
            float *gw = nn_malloc(32 * 32 * 27 * 4), *gb = nn_malloc(32 * 4), *w = rnd((size_t)32 * 32 * 27, 0.1f, 106);
            best = 1e9;
            for (int it = 0; it < 8; it++) {
                nn_zero(gw, 32 * 32 * 27 * 4); nn_zero(gb, 32 * 4);
                nn_sync(); double t0 = now(); lp_bwd_w_s2_f8(x, 4, xs, gy, 3, ys, gw, gb, (gnp_t){0}); nn_sync(); if (it >= 2 && (now() - t0) * 1e3 < best) best = (now() - t0) * 1e3;
            }
            printf("%-32s @%d %8.3f ms\n", "wgrad s2 down0 32 -> 32 (fp8)", P, best);
            best = 1e9;
            for (int it = 0; it < 8; it++) { nn_sync(); double t0 = now(); lp_bwd_data_s2_mx(gy, ys, w, xs, gx, 1); nn_sync(); if (it >= 2 && (now() - t0) * 1e3 < best) best = (now() - t0) * 1e3; }
            nn_free(gx); gx = mx(8, 32, S, 1e-3f, 105); lp_bwd_data_s2_mx(gy, ys, w, xs, gx, 1); nn_sync();
            printf("%-32s @%d %8.3f ms  out %016llx\n", "bdata s2 down0 32 -> 32 (mx8, acc)", P, best, (unsigned long long)fnv(gx, lp_mx8_bytes(1, 32, S)));
            nn_free(x); nn_free(gy); nn_free(gx); nn_free(gw); nn_free(gb); nn_free(w);
        }
        {
            shape5 xs = {1, 32, P, P, P}, ys = {1, 7, P, P, P};
            void *x = mx(4, 32, S, 2.f, 107); float *gyf = rnd(S * 7, 1e-3f, 108); void *gyh = nn_malloc(S * 7 * 2);
            nn_f32_to_h16(gyf, S * 7, gyh, 1.f);
            float *gw = nn_malloc(7 * 32 * 4), *gb = nn_malloc(7 * 4);
            const gnp_t gp = {gam, bet, mean, rstd, G};
            best = 1e9;
            for (int it = 0; it < 8; it++) {
                nn_zero(gw, 7 * 32 * 4); nn_zero(gb, 7 * 4);
                nn_sync(); double t0 = now(); lp_bwd_w1_mx_b(x, 4, xs, gyh, 2, ys, gw, gb, gp); nn_sync(); if (it >= 2 && (now() - t0) * 1e3 < best) best = (now() - t0) * 1e3;
            }
            printf("%-32s @%d %8.3f ms\n", "wgrad head 32 -> 7 (mx4 x, fp16 gy)", P, best);
            nn_free(x); nn_free(gyf); nn_free(gyh); nn_free(gw); nn_free(gb);
        }
    }
    if (want("s2b")) {   /* stride-2 backward-data, accumulating into the level's gradient (down0: 16 -> 16, down1: 32 -> 32 at P / 2) */
        for (int l = 0; l < 2; l++) {
            const int Pf = l ? P / 2 : P, Pc = Pf / 2, C = l ? 32 : 16; const size_t Sf = (size_t)Pf * Pf * Pf, Sc = (size_t)Pc * Pc * Pc;
            shape5 xs = {1, C, Pf, Pf, Pf}, ys = {1, C, Pc, Pc, Pc};
            void *gy = mx(8, C, Sc, 1e-3f, 71 + l), *gx = mx(8, C, Sf, 1e-3f, 73 + l);
            float *w = rnd((size_t)C * C * 27, 0.1f, 75);
            double best = 1e9;
            for (int it = 0; it < 8; it++) { nn_sync(); double t0 = now(); lp_bwd_data_s2_mx(gy, ys, w, xs, gx, 1); nn_sync(); if (it >= 2 && (now() - t0) * 1e3 < best) best = (now() - t0) * 1e3; }
            nn_free(gx); gx = mx(8, C, Sf, 1e-3f, 73 + l); lp_bwd_data_s2_mx(gy, ys, w, xs, gx, 1); nn_sync();
            printf("bdata s2 %s %d -> %d (mx8, acc) @%d %8.3f ms  out %016llx\n", l ? "down1" : "down0", C, C, Pf, best, (unsigned long long)fnv(gx, lp_mx8_bytes(1, C, Sf)));
            nn_free(gy); nn_free(gx); nn_free(w);
        }
    }
    if (want("up2")) {   /* decoder upsample at level 1 -> 0 (32 channels): forward of silu(gn(x)) MX-fp4 -> MX-fp4, backward MX-fp8 -> MX-fp8 */
        const int Pc = P / 2, C = 32; const size_t Sc = (size_t)Pc * Pc * Pc;
        shape5 cs = {1, C, Pc, Pc, Pc};
        void *xc = mx(4, C, Sc, 2.f, 41), *yf = nn_malloc(lp_mx4_bytes(1, C, S)), *gf = mx(8, C, S, 1e-3f, 42), *gc = nn_malloc(lp_mx8_bytes(1, C, Sc));
        float *ug = rnd(C, 1.f, 43), *ub = rnd(C, 0.5f, 44);
        const gnp_t gp = {ug, ub, mean, rstd, G};
        double b0 = 1e9, b1 = 1e9;
        for (int it = 0; it < 10; it++) {
            nn_sync(); double t0 = now(); lp_up2_fwd_mx(xc, 4, cs, yf, 4, gp); nn_sync(); double t1 = now(); lp_up2_bwd_mx(gf, cs, gc); nn_sync(); double t2 = now();
            if (it >= 2) { if ((t1 - t0) * 1e3 < b0) b0 = (t1 - t0) * 1e3; if ((t2 - t1) * 1e3 < b1) b1 = (t2 - t1) * 1e3; }
        }
        printf("%-32s @%d %8.3f ms  out %016llx\n", "up2 fwd 32 ch (gn+silu, mx4)", P, b0, (unsigned long long)fnv(yf, lp_mx4_bytes(1, C, S)));
        { double bp = 1e9; const gnp_t none = {0};
          for (int it = 0; it < 10; it++) { nn_sync(); double t0 = now(); lp_up2_fwd_mx(xc, 4, cs, yf, 4, none); nn_sync(); if (it >= 2 && (now() - t0) * 1e3 < bp) bp = (now() - t0) * 1e3; }
          printf("%-32s @%d %8.3f ms  out %016llx\n", "up2 fwd 32 ch (plain, mx4)", P, bp, (unsigned long long)fnv(yf, lp_mx4_bytes(1, C, S))); }
        printf("%-32s @%d %8.3f ms  out %016llx\n", "up2 bwd 32 ch (mx8)", P, b1, (unsigned long long)fnv(gc, lp_mx8_bytes(1, C, Sc)));
        nn_free(xc); nn_free(yf); nn_free(gf); nn_free(gc); nn_free(ug); nn_free(ub);
    }
    const char *e = nn_check(); if (e) { printf("cuda: %s\n", e); return 1; }
    return 0;
}
