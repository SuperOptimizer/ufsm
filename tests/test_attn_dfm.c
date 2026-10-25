/* deformable attention (src/nn_attn_dfm.cu, at_dims.dK > 0) against double-precision references:
   - the kernels alone: the sampled sum o against a double trilinear reference; d V (every level) against the reference's exact adjoint;
     d offsets and d logits against central finite differences of the double loss L = sum(o dO), with sample positions placed just inside
     voxel boundaries (fractions 0.002 / 0.998), partly and wholly outside the level grids (zero padding), and random ones;
     variants: the coarse level (dCo: the 2x average-pooled level-0 values, odd grid dims; d V0 includes the pool's adjoint, the coarse
     gradient against the reference too) and per-token frames (dInit 1: random orthonormal frames rotating the offsets);
   - the whole block (LN, value / offset / logit / finer-level value projections, softmax, sampling, out projection, SwiGLU MLP): its output
     and the input / parameter gradients of L = sum(y R) against central differences of a double CPU forward, 2 and 3 levels, with the
     coarse level and frames, and the kept forward (at_ws_set_fwd: no recompute) giving the same gradients;
   - identity at zero init (out projections 0): y == x bit for bit;
   - default off: dInit 1 with grid-axis frames gives the dInit 0 output and gradients bit for bit (the rotation path is exact for them);
   - at_dfm_pool / at_dfm_frame against CPU references; the normal prior (points on the normal line, heads shifted sideways).
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
/* L: the levels 0..L-1 (2^l finer), co: + the coarse level L; fr: per token frame [T][9] (rows normal, tangent, axis as z y x) or null */
typedef struct { int n, D, Hh, W, C, H, L, K, co; const double *fr; } geo;
static long g_S(const geo *G) { return (long)G->D * G->Hh * G->W; }
static int g_Lt(const geo *G) { return G->L + G->co; }
static void ldims(const geo *G, int l, int d[3], double *f) {
    if (l < G->L) { d[0] = G->D << l; d[1] = G->Hh << l; d[2] = G->W << l; *f = 1 << l; }
    else { d[0] = (G->D + 1) / 2; d[1] = (G->Hh + 1) / 2; d[2] = (G->W + 1) / 2; *f = 0.5; }
}
static size_t lvn(const geo *G, int l) { int d[3]; double f; ldims(G, l, d, &f); return (size_t)G->n * d[0] * d[1] * d[2]; }
/* level l value at (s, z, y, x, c), 0 outside */
static double vat(const geo *G, const double *V, int l, int s, int z, int y, int x, int c) {
    int d[3]; double f; ldims(G, l, d, &f);
    if (z < 0 || z >= d[0] || y < 0 || y >= d[1] || x < 0 || x >= d[2]) return 0;
    return V[((((long)s * d[0] + z) * d[1] + y) * d[2] + x) * (long)G->C + c];
}
static void vacc(const geo *G, double *V, int l, int s, int z, int y, int x, int c, double a) {
    int d[3]; double f; ldims(G, l, d, &f);
    if (z < 0 || z >= d[0] || y < 0 || y >= d[1] || x < 0 || x >= d[2]) return;
    V[((((long)s * d[0] + z) * d[1] + y) * d[2] + x) * (long)G->C + c] += a;
}
/* the coarse level from level 0 (mean of the children inside the grid) and the adjoint (added) */
static void ref_pool(const geo *G, const double *V0, double *Vc, const double *dVc, double *dV0) {
    int d[3]; double f; ldims(G, G->L, d, &f);
    for (int s = 0; s < G->n; s++) for (int zc = 0; zc < d[0]; zc++) for (int yc = 0; yc < d[1]; yc++) for (int xc = 0; xc < d[2]; xc++) for (int c = 0; c < G->C; c++) {
        int cnt = 0; double a = 0;
        for (int q = 0; q < 8; q++) { const int z = 2 * zc + (q >> 2), y = 2 * yc + ((q >> 1) & 1), x = 2 * xc + (q & 1); if (z < G->D && y < G->Hh && x < G->W) { cnt++; a += V0 ? vat(G, V0, 0, s, z, y, x, c) : 0; } }
        const long ic = ((((long)s * d[0] + zc) * d[1] + yc) * d[2] + xc) * G->C + c;
        if (Vc) Vc[ic] = a / cnt;
        if (dV0) for (int q = 0; q < 8; q++) { const int z = 2 * zc + (q >> 2), y = 2 * yc + ((q >> 1) & 1), x = 2 * xc + (q & 1); if (z < G->D && y < G->Hh && x < G->W) vacc(G, dV0, 0, s, z, y, x, c, dVc[ic] / cnt); }
    }
}
static void ref_pos(const geo *G, const double *oat, long t, int h, int l, int k, double p[3]) {
    const int g = (int)(t % g_S(G)), z = g / (G->Hh * G->W), y = (g / G->W) % G->Hh, x = g % G->W; const double c[3] = {z, y, x};
    int d[3]; double f; ldims(G, l, d, &f);
    const double *o = oat + (((size_t)h * g_Lt(G) + l) * G->K + k) * 3;
    for (int a = 0; a < 3; a++) {
        const double off = G->fr ? o[0] * G->fr[t * 9 + a] + o[1] * G->fr[t * 9 + 3 + a] + o[2] * G->fr[t * 9 + 6 + a] : o[a];
        p[a] = (double)(float)((float)(c[a] + 0.5) * (float)f - 0.5f) + off;
    }
}
/* o = sum softmax-weighted samples; dV (if not null): the adjoint of o -> loss sum(o dO) (the coarse level's in dV[L], not pooled back) */
static void ref_dfm(const geo *G, double *const *V, const double *oa, const double *dO, double *o, double *const *dV) {
    const long S = g_S(G), T = G->n * S; const int Lt = g_Lt(G), LK = Lt * G->K, NOA = G->H * LK * 4;
    double *a = malloc(LK * 8);
    for (long t = 0; t < T; t++) for (int h = 0; h < G->H; h++) {
        const double *lg = oa + t * NOA + (size_t)G->H * LK * 3 + (size_t)h * LK; double m = -1e300, sm = 0;
        for (int j = 0; j < LK; j++) if (lg[j] > m) m = lg[j];
        for (int j = 0; j < LK; j++) { a[j] = exp(lg[j] - m); sm += a[j]; }
        for (int j = 0; j < LK; j++) a[j] /= sm;
        for (int d = 0; d < AT_HD; d++) if (o) o[t * G->C + h * AT_HD + d] = 0;
        for (int l = 0; l < Lt; l++) for (int k = 0; k < G->K; k++) {
            double p[3]; ref_pos(G, oa + t * NOA, t, h, l, k, p);
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
/* random orthonormal frames (float-rounded) */
static double *rand_frames(long T) {
    double *fr = malloc(T * 9 * 8);
    for (long t = 0; t < T; t++) {
        double *R = fr + t * 9;
        for (int r = 0; r < 3; r++) {
            for (;;) {
                double v[3] = {nrand(), nrand(), nrand()};
                for (int q = 0; q < r; q++) { const double dp = v[0] * R[3 * q] + v[1] * R[3 * q + 1] + v[2] * R[3 * q + 2]; for (int a = 0; a < 3; a++) v[a] -= dp * R[3 * q + a]; }
                const double nv = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); if (nv < 0.3) continue;
                for (int a = 0; a < 3; a++) R[3 * r + a] = v[a] / nv;
                break;
            }
        }
        for (int j = 0; j < 9; j++) R[j] = rf(R[j]);
    }
    return fr;
}

static int test_kernels(int L, int K, int co, int frames) {
    geo G = {2, 3, 4, 2, 96, 2, L, K, co, nullptr};
    if (co) { G.D = 3; G.Hh = 5; G.W = 2; }   /* (odd dims: partial parents) */
    const long S = g_S(&G), T = G.n * S; const int C = G.C, Lt = g_Lt(&G), LK = Lt * K, NOA = G.H * LK * 4;
    at_dims d = {0}; d.C = C; d.H = G.H; d.F = 32; d.dK = K; d.dL = L; d.dC[0] = C; d.dCo = co; d.dInit = frames;
    at_geom g = {G.n, (int)S, G.D, G.Hh, G.W, nullptr, nullptr, nullptr, {0}, nullptr};
    printf("dfm kernels: %d x %dx%dx%d tokens, C %d, %d levels%s x %d points%s\n", G.n, G.D, G.Hh, G.W, C, L, co ? " + coarse" : "", K, frames ? ", token frames" : "");
    double *frh = frames ? rand_frames(T) : nullptr; float *frd = frames ? dev(frh, T * 9) : nullptr;
    G.fr = frh; g.dfr = frd;
    double *V[AT_DFM_NLV] = {0}, *dVr[AT_DFM_NLV] = {0}; float *Vd[AT_DFM_NLV] = {0}, *dVd[AT_DFM_NLV] = {0};
    for (int l = 0; l < Lt; l++) {
        const size_t ne = lvn(&G, l) * C; V[l] = malloc(ne * 8); dVr[l] = calloc(ne, 8);
        if (l < L) for (size_t i = 0; i < ne; i++) V[l][i] = rf(nrand());
        dVd[l] = nn_malloc(ne * 4);
    }
    for (int l = 0; l < L; l++) Vd[l] = dev(V[l], lvn(&G, l) * C);
    int bad = 0;
    if (co) {   /* the coarse level: at_dfm_pool against the reference; the reference then samples the GPU's pooled values */
        const size_t nc = lvn(&G, L) * C; Vd[L] = nn_malloc(nc * 4);
        at_dfm_pool(Vd[0], Vd[L], &g, C); nn_sync();
        ref_pool(&G, V[0], V[L], nullptr, nullptr);
        double *pg = host(Vd[L], nc); bad += report("pool", pg, V[L], (long)nc);
        for (size_t i = 0; i < nc; i++) V[L][i] = pg[i];
        free(pg);
    }
    double *oa = malloc(T * NOA * 8), *dO = malloc(T * C * 8);
    int nspec[4] = {0, 0, 0, 0};
    for (long t = 0; t < T; t++) for (int h = 0; h < G.H; h++) for (int l = 0; l < Lt; l++) for (int k = 0; k < K; k++) {
        double *o = oa + t * NOA + (((size_t)h * Lt + l) * K + k) * 3; const int kind = (int)((t * 7 + h * 3 + l * 5 + k) % 6);
        for (int a = 0; a < 3; a++) o[a] = rf(2.5 * nrand());
        double p[3], tg[3]; ref_pos(&G, oa + t * NOA, t, h, l, k, p);
        int dims[3]; double f; ldims(&G, l, dims, &f);
        for (int a = 0; a < 3; a++) {
            double tgt = p[a];
            if (kind == 0) tgt = floor(p[a]) + 0.002;                  /* just past a voxel boundary */
            else if (kind == 1) tgt = floor(p[a]) + 0.998;             /* just before one */
            else if (kind == 2 && a == 0) tgt = -0.4;                  /* half outside (low side) */
            else if (kind == 3 && a == 1) tgt = dims[1] - 0.6;         /* half outside (high side) */
            else if (kind == 4 && a == 2) tgt = dims[2] + 1.7;         /* wholly outside */
            else { const double fr = tgt - floor(tgt); if (fr < 1e-3 || fr > 1 - 1e-3) tgt += 0.01; }
            tg[a] = tgt - p[a];   /* the grid-space change */
        }
        if (!frh) for (int a = 0; a < 3; a++) o[a] = rf(o[a] + tg[a]);
        else { const double *R = frh + t * 9; for (int j = 0; j < 3; j++) o[j] = rf(o[j] + R[3 * j] * tg[0] + R[3 * j + 1] * tg[1] + R[3 * j + 2] * tg[2]); }
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
    double *orf = malloc(T * C * 8); ref_dfm(&G, V, oa, dO, orf, dVr);
    if (co) ref_pool(&G, nullptr, nullptr, dVr[L], dVr[0]);   /* d V0 += the pool adjoint of d Vc */
    { double *og = host(od, T * C); bad += report("o", og, orf, T * C); free(og); }
    for (int l = 0; l < Lt; l++) { char nm[16]; snprintf(nm, sizeof nm, l < L ? "dV%d" : "dVcoarse", l); double *dg = host(dVd[l], lvn(&G, l) * C); bad += report(nm, dg, dVr[l], (long)lvn(&G, l) * C); free(dg); }
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
    for (int l = 0; l < Lt; l++) { free(V[l]); free(dVr[l]); nn_free(Vd[l]); nn_free(dVd[l]); }
    nn_free(oad); nn_free(dOd); nn_free(awd); nn_free(od); nn_free(doad);
    free(frh); if (frd) nn_free(frd);
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
    double *V[AT_DFM_NLV] = {0};
    ref_ln(x, P + bp->ln1g, P + bp->ln1b, xn, T, C);
    V[0] = malloc(T * C * 8); ref_lin(xn, P + bp->wqkv, P + bp->bqkv, V[0], T, C, C, 0);
    ref_lin(xn, P + bp->woa, P + bp->boa, oa, T, C, NOA, 0);
    for (int l = 1; l < G->L; l++) { V[l] = malloc(lvn(G, l) * C * 8); ref_lin(B->fx[l], P + bp->wvl[l], P + bp->bvl[l], V[l], (long)lvn(G, l), B->d.dC[l], C, 0); }
    if (G->co) { V[G->L] = malloc(lvn(G, G->L) * C * 8); ref_pool(G, V[0], V[G->L], nullptr, nullptr); }
    ref_dfm(G, V, oa, nullptr, o, nullptr);
    memcpy(x1, x, T * C * 8); ref_lin(o, P + bp->wo, P + bp->bo, x1, T, C, C, 1);
    ref_ln(x1, P + bp->ln2g, P + bp->ln2b, xn, T, C);
    ref_lin(xn, P + bp->w1, P + bp->b1, h, T, C, 2 * F, 0);
    for (long t = 0; t < T; t++) for (int f = 0; f < F; f++) { double a = h[t * 2 * F + f], b = h[t * 2 * F + F + f]; gg[t * F + f] = a / (1 + exp(-a)) * b; }
    memcpy(y, x1, T * C * 8); ref_lin(gg, P + bp->w2, P + bp->b2, y, T, F, C, 1);
    for (int l = 0; l < g_Lt(G); l++) free(V[l]);
    free(xn); free(oa); free(o); free(x1); free(h); free(gg);
}
static double ref_loss(const blk *B, const double *P, const double *x, const double *R) {
    const long n = B->G.n * g_S(&B->G) * B->G.C; double *y = malloc(n * 8); ref_block(B, P, x, y);
    double L = 0; for (long i = 0; i < n; i++) L += y[i] * R[i];
    free(y); return L;
}
static int test_block(int L, int K, int npts, int co, int frames) {
    blk B; memset(&B, 0, sizeof B);
    B.G = (geo){2, 3, 4, 2, 96, 2, L, K, co, nullptr}; const int C = B.G.C, F = 64, Cl[AT_DFM_MAXL] = {96, 40, 24};
    B.d.C = C; B.d.H = B.G.H; B.d.F = F; B.d.dK = K; B.d.dL = L; for (int l = 0; l < L; l++) B.d.dC[l] = Cl[l];
    B.d.dCo = co; B.d.dInit = frames;
    const long S = g_S(&B.G), T = B.G.n * S; const int NOA = at_dfm_noa(&B.d), Lt = g_Lt(&B.G);
    at_geom g = {B.G.n, (int)S, B.G.D, B.G.Hh, B.G.W, nullptr, nullptr, nullptr, {0}, nullptr};
    double *frh = frames ? rand_frames(T) : nullptr; float *frd = frames ? dev(frh, T * 9) : nullptr;
    B.G.fr = frh; g.dfr = frd;
    float *fxd[AT_DFM_MAXL] = {0};
    for (int l = 1; l < L; l++) {   /* the finer levels' inputs: LayerNorm'd random rows, as unet feeds them */
        const size_t ne = lvn(&B.G, l) * Cl[l]; double *r = malloc(ne * 8); for (size_t i = 0; i < ne; i++) r[i] = nrand();
        B.fx[l] = malloc(ne * 8); ref_ln(r, nullptr, nullptr, B.fx[l], (long)lvn(&B.G, l), Cl[l]); for (size_t i = 0; i < ne; i++) B.fx[l][i] = rf(B.fx[l][i]);
        fxd[l] = dev(B.fx[l], ne); g.dfx[l] = fxd[l]; free(r);
    }
    const size_t np = at_block_layout(&B.d, &B.bp, 0); const at_bp *bp = &B.bp;
    printf("dfm block: %d levels%s x %d points%s, %zu parameters\n", L, co ? " + coarse" : "", K, frames ? ", token frames" : "", np);
    double *P = malloc(np * 8), *x = malloc(T * C * 8), *R = malloc(T * C * 8);
    for (size_t i = 0; i < np; i++) P[i] = rf(0.1 * nrand());
    for (int c = 0; c < C; c++) { P[bp->ln1g + c] = rf(1 + 0.2 * nrand()); P[bp->ln2g + c] = rf(1 + 0.2 * nrand()); }
    for (size_t i = 0; i < (size_t)C * C; i++) { P[bp->wqkv + i] = rf(nrand() / sqrt(C)); P[bp->wo + i] = rf(nrand() / sqrt(C)); }
    for (size_t i = 0; i < (size_t)NOA * C; i++) P[bp->woa + i] = rf(nrand() / sqrt(C));   /* offsets ~ +-1 voxel around the prior, logits ~ 1 */
    { float *pr = malloc((size_t)NOA * 4); at_dfm_prior(&B.d, pr); for (int i = 0; i < B.d.H * Lt * K * 3; i++) P[bp->boa + i] = rf(pr[i] + 0.3 * nrand()); free(pr); }
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
    { /* dense checks: the value bias (its gradient sums every level's path, the coarse pool's included) and a finer level's value bias */
        for (int which = 0; which < (L > 1 ? 2 : 1); which++) {
            const size_t boff = which ? bp->bvl[1] : bp->bqkv;
            double *ab = malloc(C * 8), *rb = malloc(C * 8);
            for (int c = 0; c < C; c++) {
                double *v = P + boff + c, v0 = *v; const double e = 1e-6;
                *v = v0 + e; double lp = ref_loss(&B, P, x, R); *v = v0 - e; double lm = ref_loss(&B, P, x, R); *v = v0;
                rb[c] = (lp - lm) / (2 * e); ab[c] = gP[boff + c];
            }
            bad += report(which ? "bvl1" : "bv dense", ab, rb, C); free(ab); free(rb);
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
    free(frh); if (frd) nn_free(frd);
    return bad;
}

/* default off: a block with dInit 1 and grid-axis frames (identity rows) == the dInit 0 block, output and gradients bit for bit */
static int test_frame_identity(void) {
    const int L = 2, K = 3, C = 96, F = 64; const int n = 2, D = 3, Hh = 4, W = 2; const long S = (long)D * Hh * W, T = n * S;
    at_dims d0 = {0}; d0.C = C; d0.H = 2; d0.F = F; d0.dK = K; d0.dL = L; d0.dC[0] = C; d0.dC[1] = 40;
    at_dims d1 = d0; d1.dInit = 1;
    at_bp bp; const size_t np = at_block_layout(&d0, &bp, 0);
    double *P = malloc(np * 8), *x = malloc(T * C * 8), *R = malloc(T * C * 8), *fx = malloc((size_t)T * 8 * 40 * 8), *fr = calloc(T * 9, 8);
    for (size_t i = 0; i < np; i++) P[i] = rf(0.3 * nrand());
    for (long i = 0; i < T * C; i++) { x[i] = rf(nrand()); R[i] = rf(nrand()); }
    for (long i = 0; i < T * 8 * 40; i++) fx[i] = rf(nrand());
    for (long t = 0; t < T; t++) fr[t * 9] = fr[t * 9 + 4] = fr[t * 9 + 8] = 1;
    float *Pd = dev(P, np), *xd = dev(x, T * C), *fxd = dev(fx, (size_t)T * 8 * 40), *frd = dev(fr, T * 9);
    at_geom g = {n, (int)S, D, Hh, W, nullptr, nullptr, nullptr, {0}, nullptr}; g.dfx[1] = fxd;
    float *y[2], *gx[2], *G[2];
    for (int v = 0; v < 2; v++) {
        const at_dims *dd = v ? &d1 : &d0; at_geom gv = g; if (v) gv.dfr = frd;
        at_ws w; void *wb = nn_malloc(at_ws_bytes(dd, T)); at_ws_carve(&w, dd, T, wb);
        float *yd = nn_malloc(T * C * 4), *gd = dev(R, T * C), *Gd = nn_malloc(np * 4); nn_zero(Gd, np * 4);
        at_block_fwd(dd, Pd, &bp, xd, yd, &gv, &w); at_block_bwd(dd, Pd, Gd, &bp, xd, gd, &gv, &w); nn_sync();
        y[v] = malloc(T * C * 4); gx[v] = malloc(T * C * 4); G[v] = malloc(np * 4);
        nn_d2h(y[v], yd, T * C * 4); nn_d2h(gx[v], gd, T * C * 4); nn_d2h(G[v], Gd, np * 4);
        nn_free(yd); nn_free(gd); nn_free(Gd); nn_free(wb);
    }
    /* the block backward scatters d V with atomics (run-to-run summation order), so its gradients compare to a tolerance; the sampling
       backward without d V (no atomics) gives the offset / logit gradients, which must match bit for bit */
    double eg = 0, sg = 0, eP = 0, sP = 0;
    for (long i = 0; i < T * C; i++) { eg = fmax(eg, fabs((double)gx[0][i] - gx[1][i])); sg = fmax(sg, fabs((double)gx[0][i])); }
    for (size_t i = 0; i < np; i++) { eP = fmax(eP, fabs((double)G[0][i] - G[1][i])); sP = fmax(sP, fabs((double)G[0][i])); }
    eg = fmax(eg / sg, eP / sP); sg = 1;
    int same = !memcmp(y[0], y[1], T * C * 4) && eg <= 1e-5;
    {
        const int Lt = L, LK = Lt * K, NOA = at_dfm_noa(&d0);
        double *V0 = malloc(T * C * 8), *V1 = malloc(T * 8 * C * 8), *oa = malloc(T * NOA * 8), *dO = malloc(T * C * 8);
        for (long i = 0; i < T * C; i++) { V0[i] = rf(nrand()); dO[i] = rf(nrand()); }
        for (long i = 0; i < T * 8 * C; i++) V1[i] = rf(nrand());
        for (long i = 0; i < T * NOA; i++) oa[i] = rf(2 * nrand());
        float *Vd[AT_DFM_NLV] = {dev(V0, T * C), dev(V1, T * 8 * C)}, *dvn[AT_DFM_NLV] = {0};
        float *oad = dev(oa, T * NOA), *dOd = dev(dO, T * C), *awd = nn_malloc(T * 2 * LK * 4), *od = nn_malloc(T * C * 4), *doad = nn_malloc(T * NOA * 4);
        float *o2[2], *da2[2];
        for (int v = 0; v < 2; v++) {
            const at_dims *dd = v ? &d1 : &d0; at_geom gv = g; if (v) gv.dfr = frd;
            at_dfm_softmax(oad, awd, T, dd); at_dfm_fwd((const float *const *)Vd, oad, awd, od, &gv, dd);
            at_dfm_bwd((const float *const *)Vd, oad, awd, dOd, dvn, doad, &gv, dd); nn_sync();
            o2[v] = malloc(T * C * 4); da2[v] = malloc(T * NOA * 4); nn_d2h(o2[v], od, T * C * 4); nn_d2h(da2[v], doad, T * NOA * 4);
        }
        same &= !memcmp(o2[0], o2[1], T * C * 4) && !memcmp(da2[0], da2[1], T * NOA * 4);
        for (int v = 0; v < 2; v++) { free(o2[v]); free(da2[v]); }
        free(V0); free(V1); free(oa); free(dO); nn_free(Vd[0]); nn_free(Vd[1]); nn_free(oad); nn_free(dOd); nn_free(awd); nn_free(od); nn_free(doad);
    }
    printf("dfm grid-axis frames: block output, sampled sum and offset / logit gradients == the frameless ones bit for bit, block gradients max rel diff %.1e (atomics)%s\n",
           eg, same ? "" : "  FAIL");
    for (int v = 0; v < 2; v++) { free(y[v]); free(gx[v]); free(G[v]); }
    free(P); free(x); free(R); free(fx); free(fr); nn_free(Pd); nn_free(xd); nn_free(fxd); nn_free(frd);
    return !same;
}
/* at_dfm_frame against a CPU reference: two samples with an axis (axis dims 0 and 2), one without; a token on the axis falls back to the
   grid axes; the normal prior: K points on the normal line spanning +-2 voxels, offsets orthogonal to it the same within a head */
static int test_frames_prior(void) {
    const int n = 3, D = 4, Hh = 5, W = 6, P = 6, S = D * Hh * W; int bad = 0;
    const int adh[3] = {0, 2, 1}; float cen[3 * 6 * 4];
    for (int b = 0; b < n; b++) for (int k = 0; k < P; k++) { float *c = cen + (b * P + k) * 4; c[0] = (float)(1.3 + 0.2 * k); c[1] = (float)(2.0 - 0.1 * k); c[2] = b == 2 ? -1.f : 10.f; c[3] = 0.f; }
    cen[(0 * P + 1) * 4 + 0] = 2.f; cen[(0 * P + 1) * 4 + 1] = 3.f;   /* sample 0, plane z 1: the axis on token (y 2, x 3) */
    int *add = nn_malloc(sizeof adh); nn_h2d(add, adh, sizeof adh); float *cd = nn_malloc(sizeof cen); nn_h2d(cd, cen, sizeof cen);
    float *frd = nn_malloc((size_t)n * S * 9 * 4); at_dfm_frame(n, D, Hh, W, add, cd, frd); nn_sync();
    float *fr = malloc((size_t)n * S * 9 * 4); nn_d2h(fr, frd, (size_t)n * S * 9 * 4);
    double mx = 0; int nax = 0;
    for (int b = 0; b < n; b++) for (int t = 0; t < S; t++) {
        const int z = t / (Hh * W), y = (t / W) % Hh, x = t % W, a = adh[b], k = a == 0 ? z : a == 1 ? y : x, u = a == 0 ? y : z, v = a == 2 ? y : x;
        const int ud = a == 0 ? 1 : 0, vd = a == 2 ? 1 : 2; const float *c = cen + (b * P + k) * 4;
        double R[9] = {0};
        const double du = u - c[0], dv = v - c[1], r = sqrt(du * du + dv * dv);
        if (c[2] < 0 || r <= 1e-3) { R[0] = R[4] = R[8] = 1; nax += c[2] >= 0; }
        else { R[ud] = du / r; R[vd] = dv / r; R[3 + ud] = -dv / r; R[3 + vd] = du / r; R[6 + a] = 1; }
        for (int j = 0; j < 9; j++) { const double e = fabs(R[j] - fr[((size_t)b * S + t) * 9 + j]); if (e > mx) mx = e; }
    }
    printf("dfm frames: max |gpu - ref| %.2e over %d tokens (%d on the axis)%s\n", mx, n * S, nax, mx < 1e-5 && nax == 1 ? "" : "  FAIL"); bad += !(mx < 1e-5 && nax == 1);
    at_dims d = {0}; d.C = 96; d.H = 2; d.dK = 8; d.dL = 2; d.dCo = 1; d.dInit = 1;
    const int Lt = at_dfm_nl(&d); float *pr = malloc((size_t)at_dfm_noa(&d) * 4); at_dfm_prior(&d, pr);
    int okp = 1;
    for (int h = 0; h < d.H; h++) for (int l = 0; l < Lt; l++) {
        const double sc = l < d.dL ? (1 << l) : 1; double lo = 1e9, hi = -1e9;
        for (int k = 0; k < d.dK; k++) {
            const float *o = pr + (((size_t)h * Lt + l) * d.dK + k) * 3, *o0 = pr + (((size_t)h * Lt + l) * d.dK) * 3;
            okp &= o[1] == o0[1] && o[2] == o0[2] && fabs(hypot(o[1], o[2]) - 0.35 * sc) < 1e-5;
            if (o[0] < lo) lo = o[0];
            if (o[0] > hi) hi = o[0];
        }
        okp &= fabs(hi - 2 * sc * (1 - 1.0 / d.dK)) < 1e-5 && fabs(lo + 2 * sc * (1 - 1.0 / d.dK)) < 1e-5;
    }
    printf("dfm normal prior: %s\n", okp ? "points on the normal over +-2 voxels, heads shifted sideways" : "WRONG  FAIL"); bad += !okp;
    free(pr); free(fr); nn_free(frd); nn_free(add); nn_free(cd);
    return bad;
}

int main(void) {
    if (nn_init(0)) { printf("SKIP: no GPU\n"); return 0; }
    at_set_mx(0);
    int bad = 0;
    bad += test_kernels(2, 3, 0, 0);
    bad += test_kernels(3, 2, 0, 0);
    bad += test_kernels(2, 2, 1, 0);
    bad += test_kernels(1, 3, 0, 1);
    bad += test_kernels(2, 2, 1, 1);
    bad += test_block(2, 3, 10, 0, 0);
    bad += test_block(3, 2, 10, 0, 0);
    bad += test_block(1, 4, 10, 0, 0);
    bad += test_block(2, 2, 10, 1, 1);
    bad += test_block(1, 3, 10, 1, 0);
    bad += test_frame_identity();
    bad += test_frames_prior();
    printf(bad ? "test_attn_dfm: %d FAILED\n" : "test_attn_dfm: all passed\n", bad);
    return bad != 0;
}
