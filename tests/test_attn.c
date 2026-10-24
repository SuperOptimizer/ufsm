/* src/nn_attn.cu (the U-Net bottleneck transformer blocks) against double-precision references:
   - whole block (LN, QKV, QK-norm, RoPE, attention, out projection, LN, SwiGLU MLP): the GPU input / parameter gradients of
     L = sum(y R) against central finite differences of a double CPU forward, sampled coordinates of every tensor, with and without
     RoPE and register tokens, 66 and 512 tokens, 2 and 4 heads;
   - attention alone at 4100 tokens (4096 grid + 4 registers: partial tiles), 2 heads: forward and dQ / dK / dV against the double
     analytic reference on sampled rows;
   - PEG (depthwise 3^3 conv on the grid tokens) by finite differences; the layout conversions by round trip.
   Pass: max |gpu - ref| / rms(ref) < 2e-3 per tensor. */
#include "attn.h"
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rs = 12345;
static double urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (double)(rs >> 11) * 0x1.0p-53; }
static double nrand(void) { double a = urand() + 1e-300, b = urand(); return sqrt(-2 * log(a)) * cos(6.283185307179586 * b); }

/* ---- double reference ---- */
static void ref_ln(const double *x, const double *g, const double *b, double *y, long T, int C) {
    for (long t = 0; t < T; t++) {
        double m = 0, q = 0; for (int c = 0; c < C; c++) m += x[t * C + c]; m /= C;
        for (int c = 0; c < C; c++) { double d = x[t * C + c] - m; q += d * d; }
        double r = 1 / sqrt(q / C + 1e-5);
        for (int c = 0; c < C; c++) y[t * C + c] = (x[t * C + c] - m) * r * g[c] + b[c];
    }
}
static void ref_lin(const double *x, const double *W, const double *b, double *y, long T, int K, int N, int acc) {   /* y (+)= x W^T + b */
    for (long t = 0; t < T; t++) for (int n = 0; n < N; n++) {
        double a = b[n]; for (int k = 0; k < K; k++) a += x[t * K + k] * W[(long)n * K + k];
        y[t * N + n] = acc ? y[t * N + n] + a : a;
    }
}
static void ref_rope(double *qkn, const at_geom *g, int nreg, int C, double base) {
    const int H = C / AT_HD;
    for (long t = 0; t < (long)g->nseq * g->S; t++) {
        int ts = (int)(t % g->S) - nreg; if (ts < 0) continue;
        int pos[3] = {ts / (g->Hh * g->W), (ts / g->W) % g->Hh, ts % g->W};
        for (int h = 0; h < H; h++) for (int ax = 0; ax < 3; ax++) for (int fi = 0; fi < 8; fi++) {
            double ang = pos[ax] * pow(base, -fi / 8.0), sn = sin(ang), cs = cos(ang);
            for (int p = 0; p < 2; p++) { double *v = qkn + t * 3 * C + p * C + h * AT_HD + ax * 16 + 2 * fi; double a = v[0], b = v[1]; v[0] = a * cs - b * sn; v[1] = a * sn + b * cs; }
        }
    }
}
static void ref_attn(const double *qkn, double *o, double *lse, const at_geom *g, int C) {   /* lse natural log */
    const int H = C / AT_HD, S = g->S; const double sc = 1 / sqrt((double)AT_HD);
    double *s = malloc((size_t)S * 8);
    for (int q = 0; q < g->nseq; q++) for (int h = 0; h < H; h++) for (int i = 0; i < S; i++) {
        const double *qi = qkn + ((long)q * S + i) * 3 * C + h * AT_HD; double m = -1e300;
        for (int j = 0; j < S; j++) { const double *kj = qkn + ((long)q * S + j) * 3 * C + C + h * AT_HD; double a = 0; for (int d = 0; d < AT_HD; d++) a += qi[d] * kj[d]; s[j] = a * sc; if (s[j] > m) m = s[j]; }
        double l = 0; for (int j = 0; j < S; j++) { s[j] = exp(s[j] - m); l += s[j]; }
        double *oi = o + ((long)q * S + i) * C + h * AT_HD;
        for (int d = 0; d < AT_HD; d++) oi[d] = 0;
        for (int j = 0; j < S; j++) { const double *vj = qkn + ((long)q * S + j) * 3 * C + 2 * C + h * AT_HD; for (int d = 0; d < AT_HD; d++) oi[d] += s[j] / l * vj[d]; }
        if (lse) lse[((long)q * H + h) * S + i] = m + log(l);
    }
    free(s);
}
typedef struct { at_dims d; at_bp bp; at_geom g; } blk;
static void ref_block(const blk *B, const double *P, const double *x, double *y) {
    const int C = B->d.C, F = B->d.F, H = B->d.H; const long T = (long)B->g.nseq * B->g.S; const at_bp *bp = &B->bp;
    double *xn = malloc(T * C * 8), *qkv = malloc(T * 3 * C * 8), *o = malloc(T * C * 8), *x1 = malloc(T * C * 8), *h = malloc(T * 2 * F * 8), *gg = malloc(T * F * 8);
    ref_ln(x, P + bp->ln1g, P + bp->ln1b, xn, T, C);
    ref_lin(xn, P + bp->wqkv, P + bp->bqkv, qkv, T, C, 3 * C, 0);
    for (long t = 0; t < T; t++) for (int hh = 0; hh < H; hh++) for (int p = 0; p < 2; p++) {
        double *v = qkv + t * 3 * C + p * C + hh * AT_HD; const double *gs = P + (p ? bp->kg : bp->qg) + hh * AT_HD; double s = 0;
        for (int d = 0; d < AT_HD; d++) s += v[d] * v[d];
        double r = 1 / sqrt(s / AT_HD + 1e-6); for (int d = 0; d < AT_HD; d++) v[d] *= r * gs[d];
    }
    if (B->d.rope) ref_rope(qkv, &B->g, B->d.nreg, C, B->d.rope_base);
    ref_attn(qkv, o, nullptr, &B->g, C);
    memcpy(x1, x, T * C * 8); ref_lin(o, P + bp->wo, P + bp->bo, x1, T, C, C, 1);
    ref_ln(x1, P + bp->ln2g, P + bp->ln2b, xn, T, C);
    ref_lin(xn, P + bp->w1, P + bp->b1, h, T, C, 2 * F, 0);
    for (long t = 0; t < T; t++) for (int f = 0; f < F; f++) { double a = h[t * 2 * F + f], b = h[t * 2 * F + F + f]; gg[t * F + f] = a / (1 + exp(-a)) * b; }
    memcpy(y, x1, T * C * 8); ref_lin(gg, P + bp->w2, P + bp->b2, y, T, F, C, 1);
    free(xn); free(qkv); free(o); free(x1); free(h); free(gg);
}
static double ref_loss(const blk *B, const double *P, const double *x, const double *R) {
    const long n = (long)B->g.nseq * B->g.S * B->d.C; double *y = malloc(n * 8); ref_block(B, P, x, y);
    double L = 0; for (long i = 0; i < n; i++) L += y[i] * R[i];
    free(y); return L;
}

static float *dev(const double *h, size_t n) { float *f = malloc(n * 4); for (size_t i = 0; i < n; i++) f[i] = (float)h[i]; float *d = nn_malloc(n * 4); nn_h2d(d, f, n * 4); free(f); return d; }
static double *host(const float *d, size_t n) { float *f = malloc(n * 4); nn_d2h(f, d, n * 4); double *h = malloc(n * 8); for (size_t i = 0; i < n; i++) h[i] = f[i]; free(f); return h; }
/* max |a - b| / rms(b) < g_tmax and (g_tl2 > 0) the relative L2 error < g_tl2. Defaults 2e-3 / off; the fp16 tensor-core attention
   at 1000-4100 tokens: max 6e-3, L2 2e-3 (fp16 rounding of q / k / v / P: the near-uniform attention's small O magnifies the max norm) */
static double g_tmax = 2e-3, g_tl2 = 0;
static int report(const char *name, const double *a, const double *b, int n) {
    double rms = 0, mx = 0, e2 = 0; for (int i = 0; i < n; i++) { rms += b[i] * b[i]; double e = fabs(a[i] - b[i]); e2 += e * e; if (e > mx) mx = e; }
    const double l2 = sqrt(e2 / (rms > 1e-300 ? rms : 1e-300));
    rms = sqrt(rms / n); double rel = mx / (rms > 1e-30 ? rms : 1e-30);
    int bad = !(rel < g_tmax) || (g_tl2 > 0 && !(l2 < g_tl2));
    printf("  %-8s rel %.2e L2 %.2e (rms %.3e, %d pts)%s\n", name, rel, l2, rms, n, bad ? "  FAIL" : "");
    return bad;
}

/* the float-rounded values both sides use: params and inputs are drawn, rounded to float, kept in double */
static double rf(double v) { return (double)(float)v; }

static int test_block(int nseq, int D, int Hh, int W, int nreg, int C, int F, int rope, int npts) {
    blk B = {0}; B.d.C = C; B.d.H = C / AT_HD; B.d.F = F; B.d.rope = rope; B.d.nreg = nreg; B.d.rope_base = 100.f;
    B.g.nseq = nseq; B.g.D = D; B.g.Hh = Hh; B.g.W = W; B.g.S = nreg + D * Hh * W;
    const size_t np = at_block_layout(&B.d, &B.bp, 0); const long T = (long)nseq * B.g.S;
    printf("block: %d seq x %d tokens (%d registers), C %d (%d heads), F %d, rope %d\n", nseq, B.g.S, nreg, C, B.d.H, F, rope);
    double *P = malloc(np * 8), *x = malloc(T * C * 8), *R = malloc(T * C * 8);
    const at_bp *bp = &B.bp;
    for (size_t i = 0; i < np; i++) P[i] = rf(0.1 * nrand());
    for (int c = 0; c < C; c++) { P[bp->ln1g + c] = rf(1 + 0.2 * nrand()); P[bp->ln2g + c] = rf(1 + 0.2 * nrand()); P[bp->qg + c] = rf(1 + 0.2 * nrand()); P[bp->kg + c] = rf(1 + 0.2 * nrand()); }
    for (size_t i = 0; i < (size_t)3 * C * C; i++) P[bp->wqkv + i] = rf(nrand() / sqrt(C) * 1.5);
    for (size_t i = 0; i < (size_t)C * C; i++) P[bp->wo + i] = rf(nrand() / sqrt(C));
    for (size_t i = 0; i < (size_t)2 * F * C; i++) P[bp->w1 + i] = rf(nrand() / sqrt(C));
    for (size_t i = 0; i < (size_t)C * F; i++) P[bp->w2 + i] = rf(nrand() / sqrt(F));
    for (long i = 0; i < T * C; i++) { x[i] = rf(nrand()); R[i] = rf(nrand()); }
    float *Pd = dev(P, np), *xd = dev(x, T * C), *gd = dev(R, T * C), *yd = nn_malloc(T * C * 4), *Gd = nn_malloc(np * 4); nn_zero(Gd, np * 4);
    at_ws w; void *wb = nn_malloc(at_ws_bytes(&B.d, T)); at_ws_carve(&w, &B.d, T, wb);
    int bad = 0;
    if (at_block_fwd(&B.d, Pd, bp, xd, yd, &B.g, &w)) { printf("  forward failed\n"); return 1; }
    { double *yg = host(yd, T * C), *yr = malloc(T * C * 8); ref_block(&B, P, x, yr); bad += report("forward", yg, yr, (int)(T * C)); free(yg); free(yr); }
    if (at_block_bwd(&B.d, Pd, Gd, bp, xd, gd, &B.g, &w)) { printf("  backward failed\n"); return 1; }
    double *gx = host(gd, T * C), *gP = host(Gd, np);
    struct { const char *nm; size_t off, n; } ts[] = {{"x", 0, (size_t)T * C}, {"ln1g", bp->ln1g, C}, {"ln1b", bp->ln1b, C}, {"wqkv", bp->wqkv, (size_t)3 * C * C}, {"bqkv", bp->bqkv, (size_t)3 * C},
        {"qg", bp->qg, C}, {"kg", bp->kg, C}, {"wo", bp->wo, (size_t)C * C}, {"bo", bp->bo, C}, {"ln2g", bp->ln2g, C}, {"ln2b", bp->ln2b, C},
        {"w1", bp->w1, (size_t)2 * F * C}, {"b1", bp->b1, (size_t)2 * F}, {"w2", bp->w2, (size_t)C * F}, {"b2", bp->b2, C}};
    double *a = malloc(npts * 8), *r = malloc(npts * 8);
    for (size_t k = 0; k < sizeof ts / sizeof ts[0]; k++) {
        for (int i = 0; i < npts; i++) {
            size_t j = (size_t)(urand() * ts[k].n); if (j >= ts[k].n) j = ts[k].n - 1;
            double *v = k == 0 ? x + j : P + ts[k].off + j, v0 = *v, e = 1e-4 * (fabs(v0) > 1 ? fabs(v0) : 1);
            *v = v0 + e; double lp = ref_loss(&B, P, x, R); *v = v0 - e; double lm = ref_loss(&B, P, x, R); *v = v0;
            r[i] = (lp - lm) / (2 * e); a[i] = k == 0 ? gx[j] : gP[ts[k].off + j];
        }
        bad += report(ts[k].nm, a, r, npts);
    }
    free(a); free(r); free(gx); free(gP); free(P); free(x); free(R);
    nn_free(Pd); nn_free(xd); nn_free(gd); nn_free(yd); nn_free(Gd); nn_free(wb);
    return bad;
}

/* core 0: at_attn_fwd / bwd on the given q / k / v. core 1: the fp16 tensor-core path, at_fa_pack (QK-norm with gains, no RoPE) of raw
   qkv rows, at_fa_fwd / bwd; the reference attends over the double-normalised rows and dQ / dK are d/d the normalised q / k.
   gs scales dO (the fp16 path's gradient scaling must make the result scale-free) */
static int test_attn_large(int nseq, int nreg, int D, int C, int core, double gs) {
    at_geom g = {nseq, nreg + D * D * D, D, D, D}; const int S = g.S, H = C / AT_HD; const long T = (long)nseq * S;
    printf("attention (core %d): %d seq x %d tokens, %d heads, dO x %.0e\n", core, nseq, S, H, gs);
    double *qkn = malloc(T * 3 * C * 8), *dO = malloc(T * C * 8), *gn = malloc(2 * C * 8);
    for (long i = 0; i < T * 3 * C; i++) qkn[i] = rf(nrand() * 0.6);
    for (long i = 0; i < T * C; i++) dO[i] = rf(nrand() * gs);
    for (int c = 0; c < 2 * C; c++) gn[c] = rf(1 + 0.2 * nrand());
    float *qd = dev(qkn, T * 3 * C), *gd = dev(dO, T * C), *od = nn_malloc(T * C * 4), *ld = nn_malloc(T * H * 4), *dvd = nn_malloc(T * H * 4), *dq = nn_malloc(T * 3 * C * 4);
    if (core) {
        float *gnd = dev(gn, 2 * C), *rq = nn_malloc(T * H * 4), *rk = nn_malloc(T * H * 4), *amx = nn_malloc(64);
        void *qh = nn_malloc((size_t)3 * C * (T + 64) * 2), *dh = nn_malloc((size_t)C * (T + 64) * 2);
        at_fa_pack(qd, gnd, gnd + C, qh, rq, rk, amx, &g, nreg, C, 0, 100.f);
        if (core == 2) { if (at_fa8_fwd(qh, od, ld, &g, H, C)) printf("  fp8 forward failed\n"); }
        else at_fa_fwd(qh, od, ld, &g, H, C);
        if (core == 2) { if (at_fa8_bwd(qh, dh, od, ld, gd, dvd, amx, dq, &g, H, C)) printf("  fp8 backward failed\n"); }
        else at_fa_bwd(qh, dh, od, ld, gd, dvd, amx, dq, &g, H, C);
        for (long t = 0; t < T; t++) for (int h = 0; h < H; h++) for (int p = 0; p < 2; p++) {   /* the reference's normalised q / k */
            double *v = qkn + t * 3 * C + p * C + h * AT_HD, s = 0; for (int d = 0; d < AT_HD; d++) s += v[d] * v[d];
            const double r = 1 / sqrt(s / AT_HD + 1e-6); for (int d = 0; d < AT_HD; d++) v[d] *= r * gn[p * C + h * AT_HD + d];
        }
        nn_free(gnd); nn_free(rq); nn_free(rk); nn_free(amx); nn_free(qh); nn_free(dh);
    } else {
        at_attn_fwd(qd, od, ld, &g, H, C);
        at_attn_bwd(qd, od, ld, gd, dvd, dq, &g, H, C);
    }
    free(gn);
    double *o = malloc(T * C * 8), *lse = malloc(T * H * 8); ref_attn(qkn, o, lse, &g, C);
    double *og = host(od, T * C), *dg = host(dq, T * 3 * C), *lg = host(ld, T * H);
    int bad = report("O", og, o, (int)(T * C));
    for (long i = 0; i < T * H; i++) lg[i] /= 1.4426950408889634;   /* log2 -> natural */
    bad += report("lse", lg, lse, (int)(T * H));
    /* sampled rows: dQ for 48 query rows, dK / dV for 48 key rows, all heads */
    const int ns = 48; const double sc = 1 / sqrt((double)AT_HD);
    double *aq = malloc(ns * C * 8), *rq = malloc(ns * C * 8), *ak = malloc(ns * C * 8), *rk = malloc(ns * C * 8), *av = malloc(ns * C * 8), *rv = malloc(ns * C * 8);
    double *Dv = malloc(T * H * 8);
    for (long t = 0; t < T; t++) for (int h = 0; h < H; h++) { double a = 0; for (int d = 0; d < AT_HD; d++) a += o[t * C + h * AT_HD + d] * dO[t * C + h * AT_HD + d]; Dv[t * H + h] = a; }
    for (int k = 0; k < ns; k++) {
        const int q = (int)(urand() * nseq) % nseq, i = (int)(urand() * S) % S, j = (int)(urand() * S) % S;
        for (int h = 0; h < H; h++) {
            const double *qi = qkn + ((long)q * S + i) * 3 * C + h * AT_HD, *gi = dO + ((long)q * S + i) * C + h * AT_HD;
            double dqa[AT_HD] = {0}, dka[AT_HD] = {0}, dva[AT_HD] = {0};
            const double li = lse[((long)q * H + h) * S + i], Di = Dv[((long)q * S + i) * H + h];
            for (int jj = 0; jj < S; jj++) {   /* dQ_i */
                const double *kj = qkn + ((long)q * S + jj) * 3 * C + C + h * AT_HD, *vj = kj + C; double s = 0, dp = 0;
                for (int d = 0; d < AT_HD; d++) { s += qi[d] * kj[d]; dp += gi[d] * vj[d]; }
                const double p = exp(s * sc - li), ds = p * (dp - Di);
                for (int d = 0; d < AT_HD; d++) dqa[d] += ds * kj[d] * sc;
            }
            const double *kj = qkn + ((long)q * S + j) * 3 * C + C + h * AT_HD, *vj = kj + C;
            for (int ii = 0; ii < S; ii++) {   /* dK_j, dV_j */
                const double *qq = qkn + ((long)q * S + ii) * 3 * C + h * AT_HD, *gg = dO + ((long)q * S + ii) * C + h * AT_HD; double s = 0, dp = 0;
                for (int d = 0; d < AT_HD; d++) { s += qq[d] * kj[d]; dp += gg[d] * vj[d]; }
                const double p = exp(s * sc - lse[((long)q * H + h) * S + ii]), ds = p * (dp - Dv[((long)q * S + ii) * H + h]);
                for (int d = 0; d < AT_HD; d++) { dka[d] += ds * qq[d] * sc; dva[d] += p * gg[d]; }
            }
            for (int d = 0; d < AT_HD; d++) {
                const int c = h * AT_HD + d;
                rq[k * C + c] = dqa[d]; aq[k * C + c] = dg[((long)q * S + i) * 3 * C + c];
                rk[k * C + c] = dka[d]; ak[k * C + c] = dg[((long)q * S + j) * 3 * C + C + c];
                rv[k * C + c] = dva[d]; av[k * C + c] = dg[((long)q * S + j) * 3 * C + 2 * C + c];
            }
        }
    }
    bad += report("dQ", aq, rq, ns * C); bad += report("dK", ak, rk, ns * C); bad += report("dV", av, rv, ns * C);
    free(qkn); free(dO); free(o); free(lse); free(og); free(dg); free(lg); free(aq); free(rq); free(ak); free(rk); free(av); free(rv); free(Dv);
    nn_free(qd); nn_free(gd); nn_free(od); nn_free(ld); nn_free(dvd); nn_free(dq);
    return bad;
}

static void ref_peg(const double *x, double *y, const double *w, const double *b, const at_geom *g, int nreg, int C) {
    for (long t = 0; t < (long)g->nseq * g->S; t++) {
        int ts = (int)(t % g->S) - nreg; long s0 = t / g->S * g->S;
        for (int c = 0; c < C; c++) {
            if (ts < 0) { y[t * C + c] = x[t * C + c]; continue; }
            int z = ts / (g->Hh * g->W), yy = (ts / g->W) % g->Hh, xx = ts % g->W; double a = x[t * C + c] + b[c];
            for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
                int zz = z + dz, y2 = yy + dy, x2 = xx + dx;
                if (zz < 0 || zz >= g->D || y2 < 0 || y2 >= g->Hh || x2 < 0 || x2 >= g->W) continue;
                a += w[c * 27 + (dz + 1) * 9 + (dy + 1) * 3 + dx + 1] * x[(s0 + nreg + ((long)zz * g->Hh + y2) * g->W + x2) * C + c];
            }
            y[t * C + c] = a;
        }
    }
}
static int test_peg(void) {
    at_geom g = {2, 2 + 3 * 4 * 5, 3, 4, 5}; const int nreg = 2, C = 8; const long T = (long)g.nseq * g.S, n = T * C;
    printf("peg: %d seq x %d tokens, C %d\n", g.nseq, g.S, C);
    double *x = malloc(n * 8), *w = malloc(C * 27 * 8), *b = malloc(C * 8), *R = malloc(n * 8), *y = malloc(n * 8);
    for (long i = 0; i < n; i++) { x[i] = rf(nrand()); R[i] = rf(nrand()); }
    for (int i = 0; i < C * 27; i++) w[i] = rf(0.3 * nrand());
    for (int i = 0; i < C; i++) b[i] = rf(0.3 * nrand());
    float *xd = dev(x, n), *wd = dev(w, C * 27), *bd = dev(b, C), *gy = dev(R, n), *yd = nn_malloc(n * 4), *gx = nn_malloc(n * 4), *gw = nn_malloc(C * 27 * 4), *gb = nn_malloc(C * 4);
    nn_zero(gw, C * 27 * 4); nn_zero(gb, C * 4);
    at_peg_fwd(xd, yd, wd, bd, &g, nreg, C); at_peg_bwd(xd, gy, gx, wd, gw, gb, &g, nreg, C);
    ref_peg(x, y, w, b, &g, nreg, C);
    double *yh = host(yd, n); int bad = report("forward", yh, y, (int)n);
    double *gxh = host(gx, n), *gwh = host(gw, C * 27), *gbh = host(gb, C);
    /* the loss is linear in x, w, b: exact gradients by unit perturbation in double */
    double *rx = malloc(n * 8), *rw = malloc(C * 27 * 8), *rb = malloc(C * 8);
#define FD(v, out) do { double v0 = (v); (v) = v0 + 1; ref_peg(x, y, w, b, &g, nreg, C); double lp = 0; for (long i_ = 0; i_ < n; i_++) lp += y[i_] * R[i_]; \
        (v) = v0 - 1; ref_peg(x, y, w, b, &g, nreg, C); double lm = 0; for (long i_ = 0; i_ < n; i_++) lm += y[i_] * R[i_]; (v) = v0; (out) = (lp - lm) / 2; } while (0)
    for (long i = 0; i < n; i++) FD(x[i], rx[i]);
    for (int i = 0; i < C * 27; i++) FD(w[i], rw[i]);
    for (int i = 0; i < C; i++) FD(b[i], rb[i]);
#undef FD
    bad += report("x", gxh, rx, (int)n); bad += report("w", gwh, rw, C * 27); bad += report("b", gbh, rb, C);
    /* layouts: channel-major -> tokens (registers from reg) -> channel-major; register gradient = sum over sequences */
    const int Sg = 60; float *cs = nn_malloc(2 * C * Sg * 4), *tk = nn_malloc(T * C * 4), *cs2 = nn_malloc(2 * C * Sg * 4), *rg = nn_malloc(nreg * C * 4); nn_zero(rg, nreg * C * 4);
    { float *h = malloc(2 * C * Sg * 4); for (int i = 0; i < 2 * C * Sg; i++) h[i] = (float)i; nn_h2d(cs, h, 2 * C * Sg * 4);
      at_cs_to_tok(cs, tk, wd, 2, C, Sg, nreg); at_tok_to_cs(tk, cs2, 2, C, Sg, nreg); at_reg_grad(tk, rg, 2, C, Sg, nreg);
      float *h2 = malloc(2 * C * Sg * 4), *r2 = malloc(nreg * C * 4), *t2 = malloc(T * C * 4); nn_d2h(h2, cs2, 2 * C * Sg * 4); nn_d2h(r2, rg, nreg * C * 4); nn_d2h(t2, tk, T * C * 4);
      int e = 0; for (int i = 0; i < 2 * C * Sg; i++) e += h2[i] != h[i];
      for (int s = 0; s < 2; s++) for (int t = 0; t < Sg; t++) for (int c = 0; c < C; c++) e += t2[((size_t)s * (Sg + nreg) + nreg + t) * C + c] != h[((size_t)s * C + c) * Sg + t];
      for (int i = 0; i < nreg * C; i++) e += fabsf(r2[i] - 2 * (float)w[i]) > 1e-6f;
      printf("  layouts %s\n", e ? "FAIL" : "ok"); bad += e != 0; free(h); free(h2); free(r2); free(t2); }
    free(x); free(w); free(b); free(R); free(y); free(yh); free(gxh); free(gwh); free(gbh); free(rx); free(rw); free(rb);
    return bad;
}

/* ---- MX GEMM: against an exact double product of the operands quantised as the kernel does (MX blocks of 32 along K, RN) ---- */
static float q_e2m1(float v) {   /* round to nearest even on the e2m1 grid, saturating */
    const float a = fabsf(v); static const float g[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6}; float best = 6; int bi = 7;
    if (a >= 6) return v < 0 ? -6.f : 6.f;
    for (int i = 0; i < 7; i++) if (a >= g[i] && a <= g[i + 1]) { const float d0 = a - g[i], d1 = g[i + 1] - a; if (d0 < d1 || (d0 == d1 && !(i & 1))) { best = g[i]; bi = i; } else { best = g[i + 1]; bi = i + 1; } break; }
    (void)bi; return v < 0 ? -best : best;
}
static float q_e4m3(float v) {
    const float a = fabsf(v); if (a == 0) return 0; if (a >= 448) return v < 0 ? -448.f : 448.f;
    int e; frexpf(a, &e); e -= 1; if (e < -6) e = -6;   /* (subnormals below 2^-6: step 2^-9) */
    const float step = ldexpf(1.f, e - 3); float r = nearbyintf(a / step) * step;   /* (ties to even by the default rounding mode) */
    if (r > 448) r = 448;
    return v < 0 ? -r : r;
}
static int mx_e(float amax, float iq) { float x = amax * iq; uint32_t u; memcpy(&u, &x, 4); int e = (int)((u >> 23) & 0xff) - 127 + ((u & 0x7fffff) != 0); return e < -126 ? -126 : e > 126 ? 126 : e; }
static void mxq_rows(const double *X, int rows, int K, int bits, double *Q) {   /* X [rows][K] -> quantised values */
    for (int r = 0; r < rows; r++) for (int k0 = 0; k0 < K; k0 += 32) {
        float am = 0; for (int k = k0; k < k0 + 32 && k < K; k++) am = fmaxf(am, fabsf((float)X[(size_t)r * K + k]));
        const int e = mx_e(am, bits == 4 ? 1.f / 6.f : 1.f / 448.f); const float m = ldexpf(1.f, -e);
        for (int k = k0; k < k0 + 32 && k < K; k++) { const float v = (float)X[(size_t)r * K + k] * m; Q[(size_t)r * K + k] = ldexp(bits == 4 ? q_e2m1(v) : q_e4m3(v), e); }
    }
}
static int test_mx_gemm(int bits, int ta, int tb, int M, int N, int K) {
    double *A = malloc((size_t)M * K * 8), *B = malloc((size_t)N * K * 8), *Aq = malloc((size_t)M * K * 8), *Bq = malloc((size_t)N * K * 8), *ref = malloc((size_t)M * N * 8), *rf32 = malloc((size_t)M * N * 8);
    for (int i = 0; i < M * K; i++) A[i] = rf(nrand());   /* logical A [M][K], B [N][K] */
    for (int i = 0; i < N * K; i++) B[i] = rf(nrand() * 0.1);
    mxq_rows(A, M, K, bits, Aq); mxq_rows(B, N, K, bits, Bq);
    for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) { double a = 0, b = 0; for (int k = 0; k < K; k++) { a += Aq[(size_t)i * K + k] * Bq[(size_t)j * K + k]; b += A[(size_t)i * K + k] * B[(size_t)j * K + k]; } ref[i * N + j] = a; rf32[i * N + j] = b; }
    /* storage per transpose flag */
    double *As = malloc((size_t)M * K * 8), *Bs = malloc((size_t)N * K * 8);
    for (int i = 0; i < M; i++) for (int k = 0; k < K; k++) As[ta ? (size_t)k * M + i : (size_t)i * K + k] = A[(size_t)i * K + k];
    for (int j = 0; j < N; j++) for (int k = 0; k < K; k++) Bs[tb ? (size_t)j * K + k : (size_t)k * N + j] = B[(size_t)j * K + k];
    float *Ad = dev(As, (size_t)M * K), *Bd = dev(Bs, (size_t)N * K), *Cd = nn_malloc((size_t)M * N * 4); nn_zero(Cd, (size_t)M * N * 4);
    int bad = lp_mx_gemm(bits, ta, tb, M, N, K, Ad, Bd, Cd, 0.f) != 0;
    double *c = host(Cd, (size_t)M * N); char nm[64]; snprintf(nm, sizeof nm, "fp%d %d%d %dx%dx%d%s", bits, ta, tb, M, N, K, M > 256 || N > 256 ? " (packed)" : "");
    printf("mx gemm %s:\n", nm); bad += report("vs quant", c, ref, M * N);
    double e2 = 0, r2 = 0; for (int i = 0; i < M * N; i++) { e2 += (c[i] - rf32[i]) * (c[i] - rf32[i]); r2 += rf32[i] * rf32[i]; }
    printf("    (vs the fp32 product: rel L2 %.3f)\n", sqrt(e2 / r2));
    free(A); free(B); free(Aq); free(Bq); free(ref); free(rf32); free(As); free(Bs); free(c); nn_free(Ad); nn_free(Bd); nn_free(Cd);
    return bad;
}

/* the block's output / gradients with MX linears against fp32 (relative L2; fp4 is coarse by design: the gate is a loose bound) */
static int test_block_mx(int bits, int core, double tol) {   /* bits: the linears (0 fp32), core: the attention core; vs fp32 + core 0 */
    blk B = {0}; B.d.C = 192; B.d.H = 4; B.d.F = 512; B.d.rope = 0; B.d.nreg = 0; B.d.rope_base = 100.f; B.g = (at_geom){1, 512, 8, 8, 8};
    const size_t np = at_block_layout(&B.d, &B.bp, 0); const long T = 512; const int C = 192;
    double *P = malloc(np * 8), *x = malloc(T * C * 8), *R = malloc(T * C * 8);
    for (size_t i = 0; i < np; i++) P[i] = rf(0.05 * nrand());
    for (int c = 0; c < C; c++) { P[B.bp.ln1g + c] = P[B.bp.ln2g + c] = P[B.bp.qg + c] = P[B.bp.kg + c] = 1; }
    for (long i = 0; i < T * C; i++) { x[i] = rf(nrand()); R[i] = rf(nrand()); }
    float *Pd = dev(P, np), *xd = dev(x, T * C), *y[2], *G[2], *gx[2]; at_ws w; void *wb = nn_malloc(at_ws_bytes(&B.d, T)); at_ws_carve(&w, &B.d, T, wb);
    for (int k = 0; k < 2; k++) {
        at_set_mx(k ? bits : 0); at_set_core(k ? core : 0);
        y[k] = nn_malloc(T * C * 4); G[k] = nn_malloc(np * 4); nn_zero(G[k], np * 4); gx[k] = dev(R, T * C);
        at_block_fwd(&B.d, Pd, &B.bp, xd, y[k], &B.g, &w); at_block_bwd(&B.d, Pd, G[k], &B.bp, xd, gx[k], &B.g, &w);
    }
    at_set_mx(0); at_set_core(0);
    double *a = host(y[1], T * C), *b = host(y[0], T * C), *ga = host(G[1], np), *gb = host(G[0], np), *xa = host(gx[1], T * C), *xb = host(gx[0], T * C);
    double e[3] = {0}, r[3] = {0};
    for (long i = 0; i < T * C; i++) { e[0] += (a[i] - b[i]) * (a[i] - b[i]); r[0] += b[i] * b[i]; e[2] += (xa[i] - xb[i]) * (xa[i] - xb[i]); r[2] += xb[i] * xb[i]; }
    for (size_t i = 0; i < np; i++) { e[1] += (ga[i] - gb[i]) * (ga[i] - gb[i]); r[1] += gb[i] * gb[i]; }
    const double ry = sqrt(e[0] / r[0]), rp = sqrt(e[1] / r[1]), rx = sqrt(e[2] / r[2]); const int bad = !(ry < tol && rp < tol && rx < tol);
    printf("block, linears %d bits, attention core %d vs fp32: output rel L2 %.3f, parameter gradient %.3f, input gradient %.3f%s\n", bits ? bits : 32, core, ry, rp, rx, bad ? "  FAIL" : "");
    return bad;
}

int main(void) {
    if (nn_init(0)) { printf("SKIP: no GPU\n"); return 0; }
    int bad = 0;
    for (int core = 0; core < 2; core++) {   /* fp32 CUDA-core attention, fp16 tensor-core attention */
        at_set_core(core); printf("== attention core %d\n", core);
        bad += test_block(2, 4, 4, 4, 1, 96, 128, 0, 24);
        bad += test_block(2, 4, 4, 4, 2, 192, 256, 1, 16);
        bad += test_block(1, 8, 8, 8, 0, 96, 128, 1, 6);
        if (core) { g_tmax = 6e-3; g_tl2 = 2e-3; }
        bad += test_attn_large(core ? 2 : 1, 4, 16, 96, core, 1);
        if (core) bad += test_attn_large(1, 3, 10, 144, core, 1e-7);
        g_tmax = 2e-3; g_tl2 = 0;
    }
    at_set_core(0);
    { printf("== attention core 2 (MX-fp8)\n"); g_tmax = 1e30; g_tl2 = 0.1;   /* (e4m3: measured L2 5-7 percent on these near-uniform attentions) */
      bad += test_attn_large(2, 4, 16, 96, 2, 1); bad += test_attn_large(1, 3, 10, 144, 2, 1e-7); g_tmax = 2e-3; g_tl2 = 0; }
    bad += test_peg();
    for (int b = 4; b <= 8; b += 4) for (int ta = 0; ta < 2; ta++) for (int tb = 0; tb < 2; tb++) { bad += test_mx_gemm(b, ta, tb, 70, 45, 100); bad += test_mx_gemm(b, ta, tb, 300, 520, 100); }
    bad += test_mx_gemm(4, 1, 0, 300, 96, 4100); bad += test_mx_gemm(8, 1, 0, 300, 96, 4100);   /* (split K: the weight-gradient shape) */
    bad += test_block_mx(8, 0, 0.1); bad += test_block_mx(4, 0, 0.35);
    bad += test_block_mx(0, 1, 2e-3); bad += test_block_mx(0, 2, 0.03); bad += test_block_mx(4, 2, 0.35);
    printf(bad ? "test_attn: %d FAILED\n" : "test_attn: all passed\n", bad);
    return bad != 0;
}
