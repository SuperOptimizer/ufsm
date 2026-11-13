/* QK-clip (src/nn_attn_tc.cu at_set_lmax, src/nn_attn.cu at_qk_clip) against a double reference:
   - the fp16 tensor-core forward's per-head max logit q.k / sqrt(48) (QK-norm + gains applied by at_fa_pack) over the valid queries and
     keys: 2 sequences of 200 tokens (partial 64-row tiles; the next sequence's rows reached by a tile must not count), the second one
     with 150 valid keys, 4 heads, gains drawn so the heads' maxima differ (~4..60). Pass: |gpu - ref| <= 2e-3 |ref| + 0.02 (fp16 operands);
   - the clip: heads above tau get qg, kg x sqrt(tau / lmax) exactly (fp32), the others unchanged, last = the max, lmax zeroed;
   - re-running the forward on the clipped gains: every head's max <= tau (+ the fp16 tolerance), the clipped ones at tau;
   - at_set_lmax(nullptr): nothing recorded. */
#include "attn.h"
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rs = 777;
static double urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (double)(rs >> 11) * 0x1.0p-53; }
static double nrand(void) { double a = urand() + 1e-300, b = urand(); return sqrt(-2 * log(a)) * cos(6.283185307179586 * b); }

enum { NSEQ = 2, S = 200, H = 4, C = H * AT_HD, T = NSEQ * S };
static const int LENS[NSEQ] = {S, 150};

static void ref_max(const float *qkv, const float *qg, const float *kg, double *mx) {   /* per head the max logit (double) */
    for (int h = 0; h < H; h++) mx[h] = 0;
    for (int q = 0; q < NSEQ; q++) for (int h = 0; h < H; h++) for (int i = 0; i < S; i++) {
        double qi[AT_HD], s = 0; const float *x = qkv + ((long)q * S + i) * 3 * C + h * AT_HD;
        for (int d = 0; d < AT_HD; d++) s += (double)x[d] * x[d];
        for (int d = 0; d < AT_HD; d++) qi[d] = x[d] / sqrt(s / AT_HD + 1e-6) * qg[h * AT_HD + d];
        for (int j = 0; j < LENS[q]; j++) {
            const float *y = qkv + ((long)q * S + j) * 3 * C + C + h * AT_HD; double t = 0, a = 0;
            for (int d = 0; d < AT_HD; d++) t += (double)y[d] * y[d];
            const double r = 1 / sqrt(t / AT_HD + 1e-6);
            for (int d = 0; d < AT_HD; d++) a += qi[d] * y[d] * r * kg[h * AT_HD + d];
            a /= sqrt((double)AT_HD); if (a > mx[h]) mx[h] = a;
        }
    }
}
static void gpu_max(const float *dqkv, const float *dqg, const float *dkg, void *qkvh, float *rq, float *rk, float *amx, float *o, float *lse,
                    const at_geom *g, float *dl, float *hl, int on) {
    nn_zero(dl, H * 4);
    at_fa_pack(dqkv, dqg, dkg, qkvh, rq, rk, amx, g, 0, C, 0, 100.f);
    at_set_lmax(on ? dl : nullptr); at_fa_fwd(qkvh, o, lse, g, H, C); at_set_lmax(nullptr);
    nn_d2h(hl, dl, H * 4);
}

int main(void) {
    if (nn_init(0)) { printf("SKIP: no GPU\n"); return 0; }
    at_set_core(1);
    float *qkv = malloc((size_t)T * 3 * C * 4), qg[C], kg[C], qg2[C], kg2[C];
    for (long i = 0; i < (long)T * 3 * C; i++) qkv[i] = (float)nrand();
    /* a shared direction in q and k of every head (aligned tokens: large logits), scaled per head */
    for (long t = 0; t < T; t++) for (int h = 0; h < H; h++) for (int d = 0; d < AT_HD; d++) {
        const float u = (float)sin(0.7 * d + h); qkv[t * 3 * C + h * AT_HD + d] += 2.f * u; qkv[t * 3 * C + C + h * AT_HD + d] += 2.f * u;
    }
    const float gs[H] = {0.6f, 1.0f, 1.6f, 2.4f};
    for (int h = 0; h < H; h++) for (int d = 0; d < AT_HD; d++) { qg[h * AT_HD + d] = gs[h] * (float)(0.8 + 0.4 * urand()); kg[h * AT_HD + d] = gs[h] * (float)(0.8 + 0.4 * urand()); }
    const long Tp = T + 64;
    float *dqkv = nn_malloc((size_t)T * 3 * C * 4), *dqg = nn_malloc(C * 4), *dkg = nn_malloc(C * 4), *rq = nn_malloc(T * H * 4), *rk = nn_malloc(T * H * 4);
    float *amx = nn_malloc(16), *o = nn_malloc((size_t)T * C * 4), *lse = nn_malloc((size_t)NSEQ * H * S * 4), *dl = nn_malloc(H * 4), *last = nn_malloc(4);
    void *qkvh = nn_malloc((size_t)3 * H * Tp * AT_HD * 2); int *dlens = (int *)nn_malloc(sizeof LENS);
    nn_h2d(dqkv, qkv, (size_t)T * 3 * C * 4); nn_h2d(dqg, qg, C * 4); nn_h2d(dkg, kg, C * 4); nn_h2d(dlens, LENS, sizeof LENS); nn_zero(last, 4);
    at_geom g = {.nseq = NSEQ, .S = S, .D = 8, .Hh = 5, .W = 5, .lens = dlens};
    int bad = 0; double mx[H], mx2[H]; float hl[H];
    ref_max(qkv, qg, kg, mx);
    gpu_max(dqkv, dqg, dkg, qkvh, rq, rk, amx, o, lse, &g, dl, hl, 1);
    double top = 0;
    for (int h = 0; h < H; h++) {
        const int ok = fabs(hl[h] - mx[h]) <= 2e-3 * fabs(mx[h]) + 0.02; bad += !ok; if (mx[h] > top) top = mx[h];
        printf("head %d: max logit gpu %.4f ref %.4f %s\n", h, hl[h], mx[h], ok ? "ok" : "FAIL");
    }
    float hz[H]; gpu_max(dqkv, dqg, dkg, qkvh, rq, rk, amx, o, lse, &g, dl, hz, 0);
    for (int h = 0; h < H; h++) if (hz[h] != 0.f) { printf("FAIL: lmax off but head %d recorded %g\n", h, hz[h]); bad++; }
    /* clip at tau between the heads' maxima */
    const float tau = (float)(0.5 * (mx[1] + mx[2]));
    gpu_max(dqkv, dqg, dkg, qkvh, rq, rk, amx, o, lse, &g, dl, hl, 1);
    at_qk_clip(dqg, dkg, dl, H, tau, last);
    float hl0[H], lt; nn_d2h(qg2, dqg, C * 4); nn_d2h(kg2, dkg, C * 4); nn_d2h(hl0, dl, H * 4); nn_d2h(&lt, last, 4);
    for (int h = 0; h < H; h++) {
        const float s = hl[h] > tau ? sqrtf(tau / hl[h]) : 1.f; int e = 0;
        for (int d = 0; d < AT_HD; d++) e += qg2[h * AT_HD + d] != qg[h * AT_HD + d] * s || kg2[h * AT_HD + d] != kg[h * AT_HD + d] * s;
        if (e || hl0[h] != 0.f) { printf("FAIL: head %d clip (%d gains off, lmax after %g)\n", h, e, hl0[h]); bad++; }
    }
    if (lt != fmaxf(fmaxf(hl[0], hl[1]), fmaxf(hl[2], hl[3]))) { printf("FAIL: last %g\n", lt); bad++; }
    ref_max(qkv, qg2, kg2, mx2);
    gpu_max(dqkv, dqg, dkg, qkvh, rq, rk, amx, o, lse, &g, dl, hl, 1);
    for (int h = 0; h < H; h++) {
        const int clipped = mx[h] > tau, ok = hl[h] <= tau * 1.002 + 0.02 && (!clipped || fabs(hl[h] - tau) <= 2e-3 * tau + 0.02) && fabs(hl[h] - mx2[h]) <= 2e-3 * mx2[h] + 0.02;
        bad += !ok; printf("after clip tau %.3f: head %d max %.4f (ref %.4f, %s) %s\n", tau, h, hl[h], mx2[h], clipped ? "clipped" : "kept", ok ? "ok" : "FAIL");
    }
    if (top < 20) { printf("FAIL: test logits too small (%.1f)\n", top); bad++; }
    printf(bad ? "FAIL (%d)\n" : "PASS\n", bad);
    return bad != 0;
}
