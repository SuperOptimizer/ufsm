/* deformable attention (src/nn_attn_dfm.cu, at_dims.dK > 0) against double-precision references:
   - the kernels alone: the sampled sum o against a double trilinear reference; d V (every level) against the reference's exact adjoint;
     d offsets and d logits against central finite differences of the double loss L = sum(o dO), with sample positions placed just inside
     voxel boundaries (fractions 0.002 / 0.998), partly and wholly outside the level grids (zero padding), and random ones;
   - the whole block (LN, value / offset / logit / finer-level value projections, softmax, sampling, out projection, SwiGLU MLP): its output
     and the input / parameter gradients of L = sum(y R) against central differences of a double CPU forward, 2 and 3 levels, and the kept
     forward (at_ws_set_fwd: no recompute) giving the same gradients;
   - identity at zero init (out projections 0): y == x bit for bit.
   Pass: max |gpu - ref| / rms(ref) < 2e-3 per tensor. */
#include "attn.h"
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rs = 4242;
static double urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (double)(rs >> 11) * 0x1.0p-53; }
static double nrand(void) { double a = urand() + 1e-300, b = urand(); return sqrt(-2 * log(a)) * cos(6.283185307179586 * b); }
static double rf(double v) { return (double)(float)v; }
static float *dev(const double *h, size_t n) { float *f = malloc(n * 4); for (size_t i = 0; i < n; i++) f[i] = (float)h[i]; float *d = nn_malloc(n * 4); nn_h2d(d, f, n * 4); free(f); return d; }
static double *host(const float *d, size_t n) { float *f = malloc(n * 4); nn_d2h(f, d, n * 4); double *h = malloc(n * 8); for (size_t i = 0; i < n; i++) h[i] = f[i]; free(f); return h; }
static int report(const char *name, const double *a, const double *b, long n) {
    double rms = 0, mx = 0; for (long i = 0; i < n; i++) { rms += b[i] * b[i]; double e = fabs(a[i] - b[i]); if (e > mx) mx = e; }
    rms = sqrt(rms / n); const double rel = mx / (rms > 1e-30 ? rms : 1e-30); const int bad = !(rel < 2e-3);
    printf("  %-10s rel %.2e (rms %.3e, %ld pts)%s\n", name, rel, rms, n, bad ? "  FAIL" : "");
    return bad;
}

/* ---- double reference of the sampling ---- */
typedef struct { int n, D, Hh, W, C, H, L, K; } geo;
static long g_S(const geo *G) { return (long)G->D * G->Hh * G->W; }
/* level l value at (s, z, y, x, c), 0 outside */
static double vat(const geo *G, const double *V, int l, int s, int z, int y, int x, int c) {
    const int Dl = G->D << l, Hl = G->Hh << l, Wl = G->W << l;
    if (z < 0 || z >= Dl || y < 0 || y >= Hl || x < 0 || x >= Wl) return 0;
    return V[(((long)s * Dl + z) * Hl + y) * Wl * (long)G->C + (long)x * G->C + c];
}
static void vacc(const geo *G, double *V, int l, int s, int z, int y, int x, int c, double a) {
    const int Dl = G->D << l, Hl = G->Hh << l, Wl = G->W << l;
    if (z < 0 || z >= Dl || y < 0 || y >= Hl || x < 0 || x >= Wl) return;
    V[(((long)s * Dl + z) * Hl + y) * Wl * (long)G->C + (long)x * G->C + c] += a;
}
static void ref_pos(const geo *G, const double *oat, int g, int h, int l, int k, double p[3]) {
    const int z = g / (G->Hh * G->W), y = (g / G->W) % G->Hh, x = g % G->W; const double f = 1 << l, c[3] = {z, y, x};
    const double *o = oat + (((size_t)h * G->L + l) * G->K + k) * 3;
    for (int a = 0; a < 3; a++) p[a] = (double)(float)((float)(c[a] + 0.5) * (float)f - 0.5f) + o[a];
}
/* o = sum softmax-weighted samples; dV (if not null): the adjoint of o -> loss sum(o dO) */
static void ref_dfm(const geo *G, double *const *V, const double *oa, const double *dO, double *o, double *const *dV) {
    const long S = g_S(G), T = G->n * S; const int LK = G->L * G->K, NOA = G->H * LK * 4;
    double *a = malloc(LK * 8);
    for (long t = 0; t < T; t++) for (int h = 0; h < G->H; h++) {
        const double *lg = oa + t * NOA + (size_t)G->H * LK * 3 + (size_t)h * LK; double m = -1e300, sm = 0;
        for (int j = 0; j < LK; j++) if (lg[j] > m) m = lg[j];
        for (int j = 0; j < LK; j++) { a[j] = exp(lg[j] - m); sm += a[j]; }
        for (int j = 0; j < LK; j++) a[j] /= sm;
        for (int d = 0; d < AT_HD; d++) if (o) o[t * G->C + h * AT_HD + d] = 0;
        for (int l = 0; l < G->L; l++) for (int k = 0; k < G->K; k++) {
            double p[3]; ref_pos(G, oa + t * NOA, (int)(t % S), h, l, k, p);
            const int z0 = (int)floor(p[0]), y0 = (int)floor(p[1]), x0 = (int)floor(p[2]); const double fz = p[0] - z0, fy = p[1] - y0, fx = p[2] - x0;
            for (int q = 0; q < 8; q++) {
                const int dz = q >> 2, dy = (q >> 1) & 1, dx = q & 1;
                const double w = (dz ? fz : 1 - fz) * (dy ? fy : 1 - fy) * (dx ? fx : 1 - fx) * a[l * G->K + k];
                for (int d = 0; d < AT_HD; d++) {
                    const int c = h * AT_HD + d;
                    if (o) o[t * G->C + c] += w * vat(G, V[l], l, (int)(t / S), z0 + dz, y0 + dy, x0 + dx, c);
                    if (dV) vacc(G, dV[l], l, (int)(t / S), z0 + dz, y0 + dy, x0 + dx, c, w * dO[t * G->C + c]);
                }
            }
        }
    }
    free(a);
}
static double ref_dfm_loss(const geo *G, double *const *V, const double *oa, const double *dO) {
    const long T = G->n * g_S(G); double *o = malloc(T * G->C * 8); ref_dfm(G, V, oa, nullptr, o, nullptr);
    double s = 0; for (long i = 0; i < T * G->C; i++) s += o[i] * dO[i];
    free(o); return s;
}
static size_t lvn(const geo *G, int l) { return (size_t)G->n * g_S(G) << (3 * l); }

static int test_kernels(int L, int K) {
    geo G = {2, 3, 4, 2, 96, 2, L, K}; const long S = g_S(&G), T = G.n * S; const int C = G.C, LK = L * K, NOA = G.H * LK * 4;
    at_dims d = {0}; d.C = C; d.H = G.H; d.F = 32; d.dK = K; d.dL = L; d.dC[0] = C;
    at_geom g = {G.n, (int)S, G.D, G.Hh, G.W, nullptr, nullptr, nullptr, {0}};
    printf("dfm kernels: %d x %dx%dx%d tokens, C %d, %d levels x %d points\n", G.n, G.D, G.Hh, G.W, C, L, K);
    double *V[AT_DFM_MAXL] = {0}, *dVr[AT_DFM_MAXL] = {0}; float *Vd[AT_DFM_MAXL] = {0}, *dVd[AT_DFM_MAXL] = {0};
    for (int l = 0; l < L; l++) {
        const size_t ne = lvn(&G, l) * C; V[l] = malloc(ne * 8); dVr[l] = calloc(ne, 8);
        for (size_t i = 0; i < ne; i++) V[l][i] = rf(nrand());
        Vd[l] = dev(V[l], ne); dVd[l] = nn_malloc(ne * 4);
    }
    double *oa = malloc(T * NOA * 8), *dO = malloc(T * C * 8);
    int nspec[4] = {0, 0, 0, 0};
    for (long t = 0; t < T; t++) for (int h = 0; h < G.H; h++) for (int l = 0; l < L; l++) for (int k = 0; k < K; k++) {
        double *o = oa + t * NOA + (((size_t)h * L + l) * K + k) * 3; const int kind = (int)((t * 7 + h * 3 + l * 5 + k) % 6);
        for (int a = 0; a < 3; a++) o[a] = rf(2.5 * nrand());
        double p[3]; ref_pos(&G, oa + t * NOA, (int)(t % S), h, l, k, p);
        const int dims[3] = {G.D << l, G.Hh << l, G.W << l};
        for (int a = 0; a < 3; a++) {
            double tgt = p[a];
            if (kind == 0) tgt = floor(p[a]) + 0.002;                  /* just past a voxel boundary */
            else if (kind == 1) tgt = floor(p[a]) + 0.998;             /* just before one */
            else if (kind == 2 && a == 0) tgt = -0.4;                  /* half outside (low side) */
            else if (kind == 3 && a == 1) tgt = dims[1] - 0.6;         /* half outside (high side) */
            else if (kind == 4 && a == 2) tgt = dims[2] + 1.7;         /* wholly outside */
            else { const double fr = tgt - floor(tgt); if (fr < 1e-3 || fr > 1 - 1e-3) tgt += 0.01; }
            o[a] = rf(o[a] + (tgt - p[a]));
        }
        nspec[kind < 2 ? 0 : kind < 4 ? 1 : kind == 4 ? 2 : 3]++;
    }
    for (long t = 0; t < T; t++) for (int j = 0; j < G.H * LK; j++) oa[t * NOA + (size_t)G.H * LK * 3 + j] = rf(nrand());
    for (long i = 0; i < T * C; i++) dO[i] = rf(nrand());
    printf("  positions: %d near voxel boundaries, %d partly outside, %d wholly outside, %d random\n", nspec[0], nspec[1], nspec[2], nspec[3]);
    float *oad = dev(oa, T * NOA), *dOd = dev(dO, T * C), *awd = nn_malloc(T * G.H * LK * 4), *od = nn_malloc(T * C * 4), *doad = nn_malloc(T * NOA * 4);
    at_dfm_softmax(oad, awd, T, &d);
    at_dfm_fwd((const float *const *)Vd, oad, awd, od, &g, &d);
    at_dfm_bwd((const float *const *)Vd, oad, awd, dOd, dVd, doad, &g, &d);
    nn_sync();
    int bad = 0;
    double *orf = malloc(T * C * 8); ref_dfm(&G, V, oa, dO, orf, dVr);
    { double *og = host(od, T * C); bad += report("o", og, orf, T * C); free(og); }
    for (int l = 0; l < L; l++) { char nm[16]; snprintf(nm, sizeof nm, "dV%d", l); double *dg = host(dVd[l], lvn(&G, l) * C); bad += report(nm, dg, dVr[l], (long)lvn(&G, l) * C); free(dg); }
    double *dg = host(doad, T * NOA);
    const long noff = (long)G.H * LK * 3; double *ar = malloc(T * NOA * 8), *rr = malloc(T * NOA * 8); long no = 0, nl = 0;
    double *al = malloc(T * NOA * 8), *rl = malloc(T * NOA * 8);
    for (long t = 0; t < T; t++) for (int j = 0; j < NOA; j++) {
        double *v = oa + t * NOA + j, v0 = *v; const double e = 1e-6;
        *v = v0 + e; const double lp = ref_dfm_loss(&G, V, oa, dO); *v = v0 - e; const double lm = ref_dfm_loss(&G, V, oa, dO); *v = v0;
        if (j < noff) { ar[no] = dg[t * NOA + j]; rr[no++] = (lp - lm) / (2 * e); } else { al[nl] = dg[t * NOA + j]; rl[nl++] = (lp - lm) / (2 * e); }
    }
    bad += report("d offset", ar, rr, no) + report("d logit", al, rl, nl);
    free(ar); free(rr); free(al); free(rl); free(dg); free(orf); free(oa); free(dO);
    for (int l = 0; l < L; l++) { free(V[l]); free(dVr[l]); nn_free(Vd[l]); nn_free(dVd[l]); }
    nn_free(oad); nn_free(dOd); nn_free(awd); nn_free(od); nn_free(doad);
    return bad;
}

/* ---- the whole block ---- */
static void ref_ln(const double *x, const double *g, const double *b, double *y, long T, int C) {
    for (long t = 0; t < T; t++) {
        double m = 0, q = 0; for (int c = 0; c < C; c++) m += x[t * C + c]; m /= C;
        for (int c = 0; c < C; c++) { double e = x[t * C + c] - m; q += e * e; }
        double r = 1 / sqrt(q / C + 1e-5);
        for (int c = 0; c < C; c++) y[t * C + c] = (x[t * C + c] - m) * r * (g ? g[c] : 1) + (b ? b[c] : 0);
    }
}
static void ref_lin(const double *x, const double *W, const double *b, double *y, long T, int K, int N, int acc) {
    for (long t = 0; t < T; t++) for (int n = 0; n < N; n++) {
        double a = b[n]; for (int k = 0; k < K; k++) a += x[t * K + k] * W[(long)n * K + k];
        y[t * N + n] = acc ? y[t * N + n] + a : a;
    }
}
typedef struct { geo G; at_dims d; at_bp bp; double *fx[AT_DFM_MAXL]; } blk;
static void ref_block(const blk *B, const double *P, const double *x, double *y) {
    const geo *G = &B->G; const int C = G->C, F = B->d.F, NOA = at_dfm_noa(&B->d); const long T = G->n * g_S(G); const at_bp *bp = &B->bp;
    double *xn = malloc(T * C * 8), *oa = malloc(T * NOA * 8), *o = malloc(T * C * 8), *x1 = malloc(T * C * 8), *h = malloc(T * 2 * F * 8), *gg = malloc(T * F * 8);
    double *V[AT_DFM_MAXL] = {0};
    ref_ln(x, P + bp->ln1g, P + bp->ln1b, xn, T, C);
    V[0] = malloc(T * C * 8); ref_lin(xn, P + bp->wqkv, P + bp->bqkv, V[0], T, C, C, 0);
    ref_lin(xn, P + bp->woa, P + bp->boa, oa, T, C, NOA, 0);
    for (int l = 1; l < G->L; l++) { V[l] = malloc(lvn(G, l) * C * 8); ref_lin(B->fx[l], P + bp->wvl[l], P + bp->bvl[l], V[l], (long)lvn(G, l), B->d.dC[l], C, 0); }
    ref_dfm(G, V, oa, nullptr, o, nullptr);
    memcpy(x1, x, T * C * 8); ref_lin(o, P + bp->wo, P + bp->bo, x1, T, C, C, 1);
    ref_ln(x1, P + bp->ln2g, P + bp->ln2b, xn, T, C);
    ref_lin(xn, P + bp->w1, P + bp->b1, h, T, C, 2 * F, 0);
    for (long t = 0; t < T; t++) for (int f = 0; f < F; f++) { double a = h[t * 2 * F + f], b = h[t * 2 * F + F + f]; gg[t * F + f] = a / (1 + exp(-a)) * b; }
    memcpy(y, x1, T * C * 8); ref_lin(gg, P + bp->w2, P + bp->b2, y, T, F, C, 1);
    for (int l = 0; l < G->L; l++) free(V[l]);
    free(xn); free(oa); free(o); free(x1); free(h); free(gg);
}
static double ref_loss(const blk *B, const double *P, const double *x, const double *R) {
    const long n = B->G.n * g_S(&B->G) * B->G.C; double *y = malloc(n * 8); ref_block(B, P, x, y);
    double L = 0; for (long i = 0; i < n; i++) L += y[i] * R[i];
    free(y); return L;
}
static int test_block(int L, int K, int npts) {
    blk B; memset(&B, 0, sizeof B);
    B.G = (geo){2, 3, 4, 2, 96, 2, L, K}; const int C = B.G.C, F = 64, Cl[AT_DFM_MAXL] = {96, 40, 24};
    B.d.C = C; B.d.H = B.G.H; B.d.F = F; B.d.dK = K; B.d.dL = L; for (int l = 0; l < L; l++) B.d.dC[l] = Cl[l];
    const long S = g_S(&B.G), T = B.G.n * S; const int NOA = at_dfm_noa(&B.d);
    at_geom g = {B.G.n, (int)S, B.G.D, B.G.Hh, B.G.W, nullptr, nullptr, nullptr, {0}};
    float *fxd[AT_DFM_MAXL] = {0};
    for (int l = 1; l < L; l++) {   /* the finer levels' inputs: LayerNorm'd random rows, as unet feeds them */
        const size_t ne = lvn(&B.G, l) * Cl[l]; double *r = malloc(ne * 8); for (size_t i = 0; i < ne; i++) r[i] = nrand();
        B.fx[l] = malloc(ne * 8); ref_ln(r, nullptr, nullptr, B.fx[l], (long)lvn(&B.G, l), Cl[l]); for (size_t i = 0; i < ne; i++) B.fx[l][i] = rf(B.fx[l][i]);
        fxd[l] = dev(B.fx[l], ne); g.dfx[l] = fxd[l]; free(r);
    }
    const size_t np = at_block_layout(&B.d, &B.bp, 0); const at_bp *bp = &B.bp;
    printf("dfm block: %d levels x %d points, %zu parameters\n", L, K, np);
    double *P = malloc(np * 8), *x = malloc(T * C * 8), *R = malloc(T * C * 8);
    for (size_t i = 0; i < np; i++) P[i] = rf(0.1 * nrand());
    for (int c = 0; c < C; c++) { P[bp->ln1g + c] = rf(1 + 0.2 * nrand()); P[bp->ln2g + c] = rf(1 + 0.2 * nrand()); }
    for (size_t i = 0; i < (size_t)C * C; i++) { P[bp->wqkv + i] = rf(nrand() / sqrt(C)); P[bp->wo + i] = rf(nrand() / sqrt(C)); }
    for (size_t i = 0; i < (size_t)NOA * C; i++) P[bp->woa + i] = rf(nrand() / sqrt(C));   /* offsets ~ +-1 voxel around the prior, logits ~ 1 */
    { float *pr = malloc((size_t)NOA * 4); at_dfm_prior(&B.d, pr); for (int i = 0; i < B.d.H * L * K * 3; i++) P[bp->boa + i] = rf(pr[i] + 0.3 * nrand()); free(pr); }
    for (int l = 1; l < L; l++) for (size_t i = 0; i < (size_t)C * Cl[l]; i++) P[bp->wvl[l] + i] = rf(nrand() / sqrt(Cl[l]));
    for (size_t i = 0; i < (size_t)2 * F * C; i++) P[bp->w1 + i] = rf(nrand() / sqrt(C));
    for (size_t i = 0; i < (size_t)C * F; i++) P[bp->w2 + i] = rf(nrand() / sqrt(F));
    for (long i = 0; i < T * C; i++) { x[i] = rf(nrand()); R[i] = rf(nrand()); }
    float *Pd = dev(P, np), *xd = dev(x, T * C), *gd = dev(R, T * C), *yd = nn_malloc(T * C * 4), *Gd = nn_malloc(np * 4); nn_zero(Gd, np * 4);
    at_ws w; void *wb = nn_malloc(at_ws_bytes(&B.d, T)); at_ws_carve(&w, &B.d, T, wb);
    int bad = 0;
    if (at_block_fwd(&B.d, Pd, bp, xd, yd, &g, &w)) { printf("  forward failed\n"); return 1; }
    { double *yg = host(yd, T * C), *yr = malloc(T * C * 8); ref_block(&B, P, x, yr); bad += report("forward", yg, yr, T * C); free(yg); free(yr); }
    if (at_block_bwd(&B.d, Pd, Gd, bp, xd, gd, &g, &w)) { printf("  backward failed\n"); return 1; }
    double *gx = host(gd, T * C), *gP = host(Gd, np);
    struct { const char *nm; size_t off, n; } ts[8 + AT_DFM_MAXL] = {{"x", 0, (size_t)T * C}, {"ln1g", bp->ln1g, C}, {"wv", bp->wqkv, (size_t)C * C}, {"bv", bp->bqkv, C},
        {"woa", bp->woa, (size_t)NOA * C}, {"boa", bp->boa, (size_t)NOA}, {"wo", bp->wo, (size_t)C * C}, {"w1", bp->w1, (size_t)2 * F * C}};
    int nts = 8;
    for (int l = 1; l < L; l++) { ts[nts].nm = l == 1 ? "wvl1" : "wvl2"; ts[nts].off = bp->wvl[l]; ts[nts++].n = (size_t)C * Cl[l]; }
    double *a = malloc(npts * 8), *r = malloc(npts * 8);
    for (int k = 0; k < nts; k++) {
        for (int i = 0; i < npts; i++) {
            size_t j = (size_t)(urand() * ts[k].n); if (j >= ts[k].n) j = ts[k].n - 1;
            double *v = k == 0 ? x + j : P + ts[k].off + j, v0 = *v; const double e = 1e-6;
            *v = v0 + e; double lp = ref_loss(&B, P, x, R); *v = v0 - e; double lm = ref_loss(&B, P, x, R); *v = v0;
            r[i] = (lp - lm) / (2 * e); a[i] = k == 0 ? gx[j] : gP[ts[k].off + j];
        }
        bad += report(ts[k].nm, a, r, npts);
    }
    { /* the bias of a finer level's values: the sum of its rows' gradients (dense check) */
        if (L > 1) {
            double *ab = malloc(C * 8), *rb = malloc(C * 8);
            for (int c = 0; c < C; c++) {
                double *v = P + bp->bvl[1] + c, v0 = *v; const double e = 1e-6;
                *v = v0 + e; double lp = ref_loss(&B, P, x, R); *v = v0 - e; double lm = ref_loss(&B, P, x, R); *v = v0;
                rb[c] = (lp - lm) / (2 * e); ab[c] = gP[bp->bvl[1] + c];
            }
            bad += report("bvl1", ab, rb, C); free(ab); free(rb);
        }
    }
    /* the kept forward: a forward set of its own (at_ws_set_fwd), kept = 1, the backward without a recompute -> the same gradients */
    {
        void *kb = nn_malloc(at_ws_fwd_bytes(&B.d, T)); at_ws wk = w; at_ws_set_fwd(&wk, &B.d, T, kb);
        float *gd2 = dev(R, T * C), *Gd2 = nn_malloc(np * 4); nn_zero(Gd2, np * 4);
        if (at_block_fwd(&B.d, Pd, bp, xd, yd, &g, &wk)) { printf("  forward failed\n"); return 1; }
        wk.kept = 1;
        if (at_block_bwd(&B.d, Pd, Gd2, bp, xd, gd2, &g, &wk)) { printf("  backward failed\n"); return 1; }
        double *gx2 = host(gd2, T * C), *gP2 = host(Gd2, np);
        bad += report("kept x", gx2, gx, T * C) + report("kept P", gP2, gP, (long)np);
        free(gx2); free(gP2); nn_free(gd2); nn_free(Gd2); nn_free(kb);
    }
    /* identity at zero init: out projections 0 -> y == x exactly */
    {
        double *Pz = malloc(np * 8); memcpy(Pz, P, np * 8);
        for (size_t i = 0; i < (size_t)C * C; i++) Pz[bp->wo + i] = 0;
        for (size_t i = 0; i < (size_t)C * F; i++) Pz[bp->w2 + i] = 0;
        for (int c = 0; c < C; c++) Pz[bp->bo + c] = Pz[bp->b2 + c] = 0;
        float *Pzd = dev(Pz, np); float *xf = malloc(T * C * 4), *yf = malloc(T * C * 4);
        if (at_block_fwd(&B.d, Pzd, bp, xd, yd, &g, &w)) { printf("  forward failed\n"); return 1; }
        nn_d2h(xf, xd, T * C * 4); nn_d2h(yf, yd, T * C * 4);
        const int same = !memcmp(xf, yf, T * C * 4);
        printf("  zero-init identity: %s\n", same ? "y == x bit for bit" : "DIFFERS  FAIL"); bad += !same;
        free(Pz); free(xf); free(yf); nn_free(Pzd);
    }
    free(a); free(r); free(gx); free(gP); free(P); free(x); free(R);
    for (int l = 1; l < L; l++) { free(B.fx[l]); nn_free(fxd[l]); }
    nn_free(Pd); nn_free(xd); nn_free(gd); nn_free(yd); nn_free(Gd); nn_free(wb);
    return bad;
}

int main(void) {
    if (nn_init(0)) { printf("SKIP: no GPU\n"); return 0; }
    at_set_mx(0);
    int bad = 0;
    bad += test_kernels(2, 3);
    bad += test_kernels(3, 2);
    bad += test_block(2, 3, 10);
    bad += test_block(3, 2, 10);
    bad += test_block(1, 4, 10);
    printf(bad ? "test_attn_dfm: %d FAILED\n" : "test_attn_dfm: all passed\n", bad);
    return bad != 0;
}
