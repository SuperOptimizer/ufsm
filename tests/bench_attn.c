/* bench_attn [C] [D] [nreg] [nseq] [reps] [F] [configs] [rounds]: one transformer block (src/nn_attn.cu) at nseq sequences of
   nreg + D^3 tokens, width C (C / 48 heads), MLP F (0: 8 C / 3 rounded to 32). Times one SSL view's work per block: a teacher forward
   (no grad) + the student's forward + backward (the backward recomputes the forward), reps times per measurement. configs: comma list
   of attention-core[:linear-mx[:keep]] settings (default "0:4,1:4"; keep 1: the backward uses the student forward, no recompute), measured interleaved for rounds rounds (default 5); the minimum ms per
   config is printed (robust on a shared GPU). UFSM_ATTN_PROF=1: the per-phase table of the last config's last round.
   BENCH_RADIAL=mult: radial attention (at_rays, mult rays per plane / D, no registers) around an axis along z through each sample's
   grid (BENCH_AXIS=out: outside it, past a corner); BENCH_ROPE=2 with it: scroll RoPE. */
#include "attn.h"
#include "nn.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rs = 777;
static float nr(void) { double s = 0; for (int i = 0; i < 12; i++) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; s += (double)(rs >> 11) * 0x1.0p-53; } return (float)(s - 6); }
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec * 1e-6; }

int main(int argc, char **argv) {
    const int C = argc > 1 ? atoi(argv[1]) : 288, D = argc > 2 ? atoi(argv[2]) : 16, nreg = argc > 3 ? atoi(argv[3]) : 1, nseq = argc > 4 ? atoi(argv[4]) : 1,
              reps = argc > 5 ? atoi(argv[5]) : 10, F = argc > 6 && atoi(argv[6]) > 0 ? atoi(argv[6]) : (8 * C / 3 + 31) / 32 * 32;
    const char *cfgs = argc > 7 ? argv[7] : "0:4,1:4"; const int rounds = argc > 8 ? atoi(argv[8]) : 5;
    if (nn_init(0)) { printf("no GPU\n"); return 1; }
    int nc = 0, core[16], mx[16], kp[16];
    { char *t = strdup(cfgs); for (char *q = strtok(t, ","); q && nc < 16; q = strtok(nullptr, ",")) { core[nc] = atoi(q); const char *c = strchr(q, ':'); mx[nc] = c ? atoi(c + 1) : 4; const char *c2 = c ? strchr(c + 1, ':') : nullptr; kp[nc++] = c2 ? atoi(c2 + 1) : 0; } free(t); }
    at_dims d = {C, C / AT_HD, F, 0, nreg, 100.f}; at_bp bp; const size_t np = at_block_layout(&d, &bp, 0);
    at_geom g = {nseq, nreg + D * D * D, D, D, D, nullptr, nullptr, nullptr}; const long T = (long)nseq * g.S;
    float *h = malloc(np * 4); for (size_t i = 0; i < np; i++) h[i] = 0.05f * nr();
    for (int c = 0; c < C; c++) h[bp.ln1g + c] = h[bp.ln2g + c] = h[bp.qg + c] = h[bp.kg + c] = 1.f;
    float *P = nn_malloc(np * 4), *G = nn_malloc(np * 4); nn_h2d(P, h, np * 4); nn_zero(G, np * 4); free(h);
    float *hx = malloc(T * C * 4); for (long i = 0; i < T * C; i++) hx[i] = nr();
    float *x = nn_malloc(T * C * 4), *y = nn_malloc(T * C * 4), *gy = nn_malloc(T * C * 4); nn_h2d(x, hx, T * C * 4); free(hx);
    long Tr = 0; at_rays R;
    if (getenv("BENCH_RADIAL")) {   /* the ray plan of an axis along z per sample */
        const float mult = (float)atof(getenv("BENCH_RADIAL")); const int out = getenv("BENCH_AXIS") && !strcmp(getenv("BENCH_AXIS"), "out");
        if (nreg) { printf("BENCH_RADIAL: nreg 0\n"); return 1; }
        int Q, Nt, Nr; at_rays_dims(D, D, D, mult, &Q, &Nt, &Nr); Tr = (long)nseq * Q * Nr;
        float *axr = calloc((size_t)nseq * AT_AXR(D), 4), *ray = malloc((size_t)nseq * Q * AT_RAY_F * 4), *cen = malloc((size_t)nseq * D * 16); int *lens = malloc((size_t)nseq * Q * 4), *ad = malloc((size_t)nseq * 4);
        for (int b = 0; b < nseq; b++) { float *a = axr + (size_t)b * AT_AXR(D); a[1] = 1; for (int k = 0; k < D; k++) { a[2 + 2 * k] = out ? -0.6f * D - 0.05f * k : 0.45f * D + 0.03f * k; a[3 + 2 * k] = out ? 1.3f * D : 0.55f * D - 0.02f * k; } }
        at_rays_plan(nseq, D, D, D, mult, axr, ray, lens, ad, cen);
        long used = 0; for (long q = 0; q < (long)nseq * Q; q++) used += (long)ray[q * AT_RAY_F + 6];
        R = (at_rays){nseq, Q, Nt, Nr, D, D, D, nn_malloc((size_t)nseq * 4), nn_malloc((size_t)nseq * Q * AT_RAY_F * 4), nn_malloc((size_t)nseq * Q * 4), nn_malloc((size_t)Tr * 12), nn_malloc(T * 4)};
        nn_h2d((void *)R.ad, ad, (size_t)nseq * 4); nn_h2d((void *)R.ray, ray, (size_t)nseq * Q * AT_RAY_F * 4); nn_h2d((void *)R.lens, lens, (size_t)nseq * Q * 4);
        at_rays_setup(&R); g.rays = &R; d.rope = getenv("BENCH_ROPE") ? atoi(getenv("BENCH_ROPE")) : 0;
        float *ws = malloc(T * 4); nn_sync(); nn_d2h(ws, R.wsum, T * 4); long cov = 0; for (long i = 0; i < T; i++) cov += ws[i] > 0;
        printf("radial: axis %s, %d rays per plane, Q %d rays x Nr %d slots per sample = %ld ray tokens (%.2f x the grid), %.1f percent used, grid coverage %.3f\n",
               out ? "outside" : "inside", Nt, Q, Nr, Tr, (double)Tr / T, 100.0 * used / Tr, (double)cov / T);
        free(axr); free(ray); free(cen); free(lens); free(ad); free(ws);
    }
    at_ws w; void *wb = nn_malloc(at_ws_bytes_r(&d, T, Tr)); at_ws_carve_r(&w, &d, T, Tr, wb);
    if (getenv("BENCH_GEMM")) {   /* the block's twelve GEMM shapes (bias / residual epilogues as the block): min ms over launches per linear-mx */
        struct { const char *nm; int ta, tb, M, N, K; const float *A, *B; float *Cm; float beta; const float *bias, *res; } gm[] = {
            {"qkv fwd", 0, 1, (int)T, 3 * C, C, w.xn, P + bp.wqkv, w.qkv, 0.f, P + bp.bqkv, nullptr},
            {"o fwd", 0, 1, (int)T, C, C, w.o, P + bp.wo, w.x1, 0.f, P + bp.bo, x},
            {"w1 fwd", 0, 1, (int)T, 2 * F, C, w.xn2, P + bp.w1, w.h, 0.f, P + bp.b1, nullptr},
            {"w2 fwd", 0, 1, (int)T, C, F, w.gg, P + bp.w2, y, 0.f, P + bp.b2, w.x1},
            {"w2 wgrad", 1, 0, C, F, (int)T, gy, w.gg, G + bp.w2, 1.f, nullptr, nullptr},
            {"w2 dgrad", 0, 0, (int)T, F, C, gy, P + bp.w2, w.bB, 0.f, nullptr, nullptr},
            {"w1 wgrad", 1, 0, 2 * F, C, (int)T, w.bA, w.xn2, G + bp.w1, 1.f, nullptr, nullptr},
            {"w1 dgrad", 0, 0, (int)T, C, 2 * F, w.bA, P + bp.w1, w.bC, 0.f, nullptr, nullptr},
            {"o wgrad", 1, 0, C, C, (int)T, w.dx1, w.o, G + bp.wo, 1.f, nullptr, nullptr},
            {"o dgrad", 0, 0, (int)T, C, C, w.dx1, P + bp.wo, w.bC, 0.f, nullptr, nullptr},
            {"qkv wgrad", 1, 0, 3 * C, C, (int)T, w.bB, w.xn, G + bp.wqkv, 1.f, nullptr, nullptr},
            {"qkv dgrad", 0, 0, (int)T, C, 3 * C, w.bB, P + bp.wqkv, w.bC, 0.f, nullptr, nullptr}};
        if (at_block_fwd(&d, P, &bp, x, y, &g, &w)) return 1;
        nn_d2d(gy, x, T * C * 4); at_block_bwd(&d, P, G, &bp, x, gy, &g, &w); nn_d2d(gy, x, T * C * 4);
        for (int k = 0; k < nc; k++) {
            at_set_mx(mx[k]); double tot = 0;
            printf("linear-mx %d GEMMs (min ms):", mx[k]);
            for (size_t j = 0; j < sizeof gm / sizeof gm[0]; j++) {
                double b = 1e30;
                for (int i = 0; i < 4 * reps; i++) {
                    nn_sync(); const double t0 = now_ms();
                    at_gemm_ex(gm[j].ta, gm[j].tb, gm[j].M, gm[j].N, gm[j].K, gm[j].A, gm[j].B, gm[j].Cm, gm[j].beta, gm[j].bias, gm[j].res);
                    nn_sync(); const double t = now_ms() - t0; if (t < b) b = t;
                }
                printf(" %s %.3f", gm[j].nm, b); tot += j < 4 ? 2 * b : b;   /* (forwards: teacher + student) */
            }
            printf(" | per view (2 fwd + bwd) %.2f ms\n", tot);
        }
        return 0;
    }
    if (getenv("BENCH_CORE_ONLY")) {   /* the attention products alone: min over launches of the forward and the backward per core (1, 2) */
        if (at_block_fwd(&d, P, &bp, x, y, &g, &w)) return 1;   /* (qkv) */
        at_fa_pack(w.qkv, P + bp.qg, P + bp.kg, w.qkvh, w.rq, w.rk, w.amx, &g, nreg, C, 0, 100.f);
        nn_d2d(gy, x, T * C * 4);
        for (int k = 0; k < nc; k++) {
            double bf = 1e30, bb = 1e30;
            for (int i = 0; i < 4 * reps; i++) {
                nn_sync(); double t0 = now_ms();
                if (core[k] == 2) at_fa8_fwd(w.qkvh, w.o, w.lse, &g, d.H, C); else at_fa_fwd(w.qkvh, w.o, w.lse, &g, d.H, C);
                nn_sync(); double t1 = now_ms();
                if (core[k] == 2) at_fa8_bwd(w.qkvh, w.doh, w.o, w.lse, gy, w.dv, w.amx, w.bA, &g, d.H, C); else at_fa_bwd(w.qkvh, w.doh, w.o, w.lse, gy, w.dv, w.amx, w.bA, &g, d.H, C);
                nn_sync(); double t2 = now_ms();
                bf = t1 - t0 < bf ? t1 - t0 : bf; bb = t2 - t1 < bb ? t2 - t1 : bb;
            }
            const double A = 4.0 * nseq * (double)g.S * g.S * AT_HD * d.H;
            printf("core %d attention products: forward %.3f ms (%.1f TFLOP/s), backward %.3f ms (%.1f TFLOP/s)\n", core[k], bf, A / bf / 1e9, bb, 3.5 * A / bb / 1e9);
        }
        return 0;
    }
    double best[16]; for (int k = 0; k < nc; k++) best[k] = 1e30;
    for (int r = 0; r < rounds; r++)
        for (int k = 0; k < nc; k++) {
            at_set_core(core[k]); at_set_mx(mx[k]);
            for (int i = 0; i < reps + 1; i++) {
                if (i == 1) at_prof_reset();
                if (at_block_fwd(&d, P, &bp, x, y, &g, &w)) return 1;                  /* teacher */
                if (at_block_fwd(&d, P, &bp, x, y, &g, &w)) return 1;                  /* student */
                w.kept = kp[k];
                nn_d2d_async(gy, x, T * C * 4);
                if (at_block_bwd(&d, P, G, &bp, x, gy, &g, &w)) return 1;
                static double ts; if (i == 0) { nn_sync(); ts = now_ms(); } else if (i == reps) { nn_sync(); const double ms = (now_ms() - ts) / reps; if (ms < best[k]) best[k] = ms; }
            }
        }
    const char *er = nn_check();
    printf("bench_attn: C %d (%d heads) F %d, %d seq x %d tokens, block scratch %.1f MB, %s; ms per view per block (teacher fwd + student fwd + bwd), min of %d rounds:\n",
           C, d.H, F, nseq, g.S, at_ws_bytes_r(&d, T, Tr) / 1e6, er ? er : "ok", rounds);
    for (int k = 0; k < nc; k++) printf("  core %d linear-mx %d keep %d: %.2f ms\n", core[k], mx[k], kp[k], best[k]);
    at_prof_report(stdout); at_prof_reset();
    return 0;
}
