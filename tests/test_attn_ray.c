/* radial axial attention and scroll RoPE (src/nn_attn_ray.cu, src/attn.h at_rays / at_dims.rope 2) against double-precision references:
   - the ray plan: every ray sample inside its plane's rectangle, positions (r, arc, k) as the table says;
   - the resampling maps: the GPU gather / scatter (plain and normalised) against a double CPU stencil from the same ray table, and the
     adjoint identities <G x, y> = <x, S y> (plain) and <G_n x, y> = <x, S_n y> (normalised) in double;
   - the whole block with radial attention (gather, QK-norm, scroll RoPE on the ray positions, length-masked attention per ray,
     normalised scatter): input / parameter gradients of L = sum(y R) against central finite differences of a double CPU forward,
     cores 0 (fp32) and 1 (fp16 tensor cores); with an axis inside the grid, outside it, and none (grid rows);
   - the block with full attention and scroll RoPE (at_scroll_pos positions) the same way, and those positions against a host formula.
   Pass: max |gpu - ref| / rms(ref) < 2e-3 per tensor (the adjoint dot products: relative 1e-5). */
#include "attn.h"
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rs = 777;
static double urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (double)(rs >> 11) * 0x1.0p-53; }
static double nrand(void) { double a = urand() + 1e-300, b = urand(); return sqrt(-2 * log(a)) * cos(6.283185307179586 * b); }
static double rf(double v) { return (double)(float)v; }
static float *dev(const double *h, size_t n) { float *f = malloc(n * 4); for (size_t i = 0; i < n; i++) f[i] = (float)h[i]; float *d = nn_malloc(n * 4); nn_h2d(d, f, n * 4); free(f); return d; }
static double *host(const float *d, size_t n) { float *f = malloc(n * 4); nn_d2h(f, d, n * 4); double *h = malloc(n * 8); for (size_t i = 0; i < n; i++) h[i] = f[i]; free(f); return h; }
static double g_tmax = 2e-3;
static int report(const char *name, const double *a, const double *b, long n) {
    double rms = 0, mx = 0; for (long i = 0; i < n; i++) { rms += b[i] * b[i]; double e = fabs(a[i] - b[i]); if (e > mx) mx = e; }
    rms = sqrt(rms / n); const double rel = mx / (rms > 1e-30 ? rms : 1e-30); const int bad = !(rel < g_tmax);
    printf("  %-10s rel %.2e (rms %.3e, %ld pts)%s\n", name, rel, rms, n, bad ? "  FAIL" : "");
    return bad;
}

/* ---- a ray set on the device with its host tables ---- */
typedef struct {
    at_rays R; int n, D, Hh, W, P; long Sg, Tr;
    float *ray_h, *cen_h; int *lens_h, *ad_h;
    float *ray_d, *pos_d, *wsum_d, *cen_d; int *lens_d, *ad_d;
    /* the double stencil: per slot up to 4 (row, weight) */
    long *srow; double *sw; int *sn; double *wsum;
} rays_t;
static void rays_make(rays_t *t, int n, int D, int Hh, int W, float mult, const float *axr) {
    memset(t, 0, sizeof *t); t->n = n; t->D = D; t->Hh = Hh; t->W = W; t->Sg = (long)D * Hh * W;
    int Q, Nt, Nr; at_rays_dims(D, Hh, W, mult, &Q, &Nt, &Nr); const int P = D > Hh ? (D > W ? D : W) : (Hh > W ? Hh : W); t->P = P;
    t->Tr = (long)n * Q * Nr;
    t->ray_h = malloc((size_t)n * Q * AT_RAY_F * 4); t->lens_h = malloc((size_t)n * Q * 4); t->ad_h = malloc((size_t)n * 4); t->cen_h = malloc((size_t)n * P * 16);
    at_rays_plan(n, D, Hh, W, mult, axr, t->ray_h, t->lens_h, t->ad_h, t->cen_h);
    t->ray_d = nn_malloc((size_t)n * Q * AT_RAY_F * 4); t->lens_d = nn_malloc((size_t)n * Q * 4); t->ad_d = nn_malloc((size_t)n * 4);
    t->pos_d = nn_malloc((size_t)t->Tr * 12); t->wsum_d = nn_malloc((size_t)n * t->Sg * 4); t->cen_d = nn_malloc((size_t)n * P * 16);
    nn_h2d(t->ray_d, t->ray_h, (size_t)n * Q * AT_RAY_F * 4); nn_h2d(t->lens_d, t->lens_h, (size_t)n * Q * 4); nn_h2d(t->ad_d, t->ad_h, (size_t)n * 4);
    nn_h2d(t->cen_d, t->cen_h, (size_t)n * P * 16);
    t->R = (at_rays){n, Q, Nt, Nr, D, Hh, W, t->ad_d, t->ray_d, t->lens_d, t->pos_d, t->wsum_d};
    at_rays_setup(&t->R);
    t->srow = malloc((size_t)t->Tr * 4 * sizeof *t->srow); t->sw = malloc((size_t)t->Tr * 4 * 8); t->sn = malloc((size_t)t->Tr * 4);
    t->wsum = calloc((size_t)n * t->Sg, 8);
    for (long s = 0; s < t->Tr; s++) {
        const long rq = s / Nr; const int i = (int)(s % Nr), b = (int)(rq / Q); const float *rr = t->ray_h + rq * AT_RAY_F; const int ad = t->ad_h[b];
        int Np, Nu, Nv; at_ray_plane(ad, D, Hh, W, &Np, &Nu, &Nv);
        int ia[4], ib[4]; float w[4]; const int nc = at_ray_corners_h(rr, i, Nu, Nv, ia, ib, w);
        t->sn[s] = nc;
        for (int c = 0; c < nc; c++) { t->srow[4 * s + c] = (long)b * t->Sg + at_ray_tok(ad, (int)rr[7], ia[c], ib[c], Hh, W); t->sw[4 * s + c] = w[c]; t->wsum[t->srow[4 * s + c]] += w[c]; }
    }
    for (long v = 0; v < n * t->Sg; v++) t->wsum[v] = t->wsum[v] > 1e-6 ? 1 / t->wsum[v] : 0;
}
static void rays_free(rays_t *t) {
    free(t->ray_h); free(t->lens_h); free(t->ad_h); free(t->cen_h); free(t->srow); free(t->sw); free(t->sn); free(t->wsum);
    nn_free(t->ray_d); nn_free(t->lens_d); nn_free(t->ad_d); nn_free(t->pos_d); nn_free(t->wsum_d); nn_free(t->cen_d);
}
static void ref_gather(const rays_t *t, const double *x, double *y, int Cc, int norm) {
    for (long s = 0; s < t->Tr; s++) {
        double *o = y + s * Cc; for (int c = 0; c < Cc; c++) o[c] = 0;
        for (int k = 0; k < t->sn[s]; k++) { const long v = t->srow[4 * s + k]; const double w = t->sw[4 * s + k] * (norm ? t->wsum[v] : 1); for (int c = 0; c < Cc; c++) o[c] += w * x[v * Cc + c]; }
    }
}
static void ref_scatter(const rays_t *t, const double *y, double *x, int Cc, int norm) {
    memset(x, 0, (size_t)t->n * t->Sg * Cc * 8);
    for (long s = 0; s < t->Tr; s++) for (int k = 0; k < t->sn[s]; k++) {
        const long v = t->srow[4 * s + k]; const double w = t->sw[4 * s + k] * (norm ? t->wsum[v] : 1);
        for (int c = 0; c < Cc; c++) x[v * Cc + c] += w * y[s * Cc + c];
    }
}
/* the axis records of the tests: sample 0 an axis inside the plane drifting with k (dim ad0), sample 1 one outside (dim ad1), sample 2
   none (grid rows) */
static float *make_axr(int n, int D, int Hh, int W, const int *ads, const int *kinds) {
    const int P = D > Hh ? (D > W ? D : W) : (Hh > W ? Hh : W); float *a = calloc((size_t)n * AT_AXR(P), 4);
    for (int b = 0; b < n; b++) {
        float *r = a + (size_t)b * AT_AXR(P); int Np, Nu, Nv; at_ray_plane(ads[b], D, Hh, W, &Np, &Nu, &Nv);
        r[0] = (float)ads[b]; r[1] = kinds[b] != 2;
        for (int k = 0; k < Np; k++) {
            if (kinds[b] == 0) { r[2 + 2 * k] = 0.37f * Nu + 0.21f * k; r[3 + 2 * k] = 0.55f * Nv - 0.13f * k; }
            else { r[2 + 2 * k] = -2.3f - 0.1f * k; r[3 + 2 * k] = 1.4f * Nv + 0.6f + 0.05f * k; }
        }
    }
    return a;
}

static int test_maps(int D, int Hh, int W, float mult) {
    const int n = 3, ads[3] = {0, 1, 2}, kinds[3] = {0, 1, 2}, Cc = 8;
    float *axr = make_axr(n, D, Hh, W, ads, kinds);
    rays_t t; rays_make(&t, n, D, Hh, W, mult, axr);
    printf("ray maps: grid %d x %d x %d, %d samples (axis inside / outside / none), Q %d rays x Nr %d\n", D, Hh, W, n, t.R.Q, t.R.Nr);
    int bad = 0;
    {   /* the plan: samples inside the rectangle, positions */
        long out = 0, used = 0; double *pos = host(t.pos_d, (size_t)t.Tr * 3), pe = 0;
        for (long s = 0; s < t.Tr; s++) {
            const long rq = s / t.R.Nr; const int i = (int)(s % t.R.Nr), b = (int)(rq / t.R.Q); const float *rr = t.ray_h + rq * AT_RAY_F;
            int Np, Nu, Nv; at_ray_plane(t.ad_h[b], D, Hh, W, &Np, &Nu, &Nv);
            if (i < (int)rr[6]) { used++; const double pu = rr[0] + i * (double)rr[2], pv = rr[1] + i * (double)rr[3]; out += pu < -0.5 - 1e-4 || pu > Nu - 0.5 + 1e-4 || pv < -0.5 - 1e-4 || pv > Nv - 0.5 + 1e-4; }
            pe = fmax(pe, fabs(pos[3 * s] - (rr[4] + i)) + fabs(pos[3 * s + 1] - rr[5]) + fabs(pos[3 * s + 2] - rr[7]));
        }
        long cov[3] = {0}; double *ws = host(t.wsum_d, (size_t)n * t.Sg);
        for (int b = 0; b < n; b++) for (long v = 0; v < t.Sg; v++) cov[b] += ws[b * t.Sg + v] > 0;
        printf("  plan: %ld of %ld slots used, %ld outside the plane, position error %.1e; coverage %.3f / %.3f / %.3f%s\n", used, t.Tr, out, pe,
               (double)cov[0] / t.Sg, (double)cov[1] / t.Sg, (double)cov[2] / t.Sg, out || pe > 1e-5 ? "  FAIL" : "");
        bad += out || pe > 1e-5;
        bad += report("wsum", ws, t.wsum, n * t.Sg);
        free(pos); free(ws);
    }
    const long ng = (long)n * t.Sg * Cc, nr = t.Tr * Cc;
    double *x = malloc(ng * 8), *y = malloc(nr * 8), *rg = malloc(nr * 8), *rsc = malloc(ng * 8);
    for (long i = 0; i < ng; i++) x[i] = rf(nrand());
    for (long i = 0; i < nr; i++) y[i] = rf(nrand());
    float *xd = dev(x, ng), *yd = dev(y, nr), *gy = nn_malloc(nr * 4), *sx = nn_malloc(ng * 4);
    for (int norm = 0; norm < 2; norm++) {
        at_ray_gather(&t.R, xd, gy, Cc, norm); at_ray_scatter(&t.R, yd, sx, Cc, norm);
        double *G = host(gy, nr), *S = host(sx, ng);
        ref_gather(&t, x, rg, Cc, norm); ref_scatter(&t, y, rsc, Cc, norm);
        bad += report(norm ? "gather n" : "gather", G, rg, nr); bad += report(norm ? "scatter n" : "scatter", S, rsc, ng);
        double a = 0, b2 = 0, ra = 0, rb = 0, sa = 0, sr = 0; for (long i = 0; i < nr; i++) { a += G[i] * y[i]; ra += rg[i] * y[i]; sa += fabs(G[i] * y[i]); sr += fabs(rg[i] * y[i]); } for (long i = 0; i < ng; i++) { b2 += x[i] * S[i]; rb += x[i] * rsc[i]; }
        const double e = fabs(a - b2) / fmax(sa, 1e-30), er = fabs(ra - rb) / fmax(sr, 1e-30);   /* (relative to sum |terms|: the dot products may cancel) */
        printf("  adjoint%s: <Gx, y> %.6e  <x, Sy> %.6e  rel %.1e (double reference %.1e)%s\n", norm ? " (normalised)" : "", a, b2, e, er, e < 1e-5 && er < 1e-12 ? "" : "  FAIL");
        bad += !(e < 1e-5 && er < 1e-12);
        free(G); free(S);
    }
    free(x); free(y); free(rg); free(rsc); free(axr); nn_free(xd); nn_free(yd); nn_free(gy); nn_free(sx); rays_free(&t);
    return bad;
}

/* ---- double block reference (the test_attn.c block with the attention on rays or with positions) ---- */
static void ref_ln(const double *x, const double *g, const double *b, double *y, long T, int C) {
    for (long t = 0; t < T; t++) {
        double m = 0, q = 0; for (int c = 0; c < C; c++) m += x[t * C + c]; m /= C;
        for (int c = 0; c < C; c++) { double d = x[t * C + c] - m; q += d * d; }
        double r = 1 / sqrt(q / C + 1e-5);
        for (int c = 0; c < C; c++) y[t * C + c] = (x[t * C + c] - m) * r * g[c] + b[c];
    }
}
static void ref_lin(const double *x, const double *W, const double *b, double *y, long T, int K, int N, int acc) {
    for (long t = 0; t < T; t++) for (int n = 0; n < N; n++) {
        double a = b[n]; for (int k = 0; k < K; k++) a += x[t * K + k] * W[(long)n * K + k];
        y[t * N + n] = acc ? y[t * N + n] + a : a;
    }
}
/* attention over nseq sequences of S tokens (lens: keys < lens[q]); rope on per-token positions pos (or none) after the QK-norm */
static void ref_attn_seq(double *qkv, double *o, long nseq, int S, const int *lens, const double *pos, int nreg, int C, double base, const double *qg, const double *kg) {
    const int H = C / AT_HD; const long T = nseq * S; const double sc = 1 / sqrt((double)AT_HD);
    for (long t = 0; t < T; t++) for (int hh = 0; hh < H; hh++) for (int p = 0; p < 2; p++) {
        double *v = qkv + t * 3 * C + p * C + hh * AT_HD; const double *gs = (p ? kg : qg) + hh * AT_HD; double s = 0;
        for (int d = 0; d < AT_HD; d++) s += v[d] * v[d];
        double r = 1 / sqrt(s / AT_HD + 1e-6); for (int d = 0; d < AT_HD; d++) v[d] *= r * gs[d];
    }
    if (pos) for (long t = 0; t < T; t++) {
        if ((int)(t % S) < nreg) continue;
        for (int hh = 0; hh < H; hh++) for (int ax = 0; ax < 3; ax++) for (int fi = 0; fi < 8; fi++) {
            const double ang = pos[3 * t + ax] * pow(base, -fi / 8.0), sn = sin(ang), cs = cos(ang);
            for (int p = 0; p < 2; p++) { double *v = qkv + t * 3 * C + p * C + hh * AT_HD + ax * 16 + 2 * fi; double a = v[0], b = v[1]; v[0] = a * cs - b * sn; v[1] = a * sn + b * cs; }
        }
    }
    double *s = malloc((size_t)S * 8);
    for (long q = 0; q < nseq; q++) for (int h = 0; h < H; h++) for (int i = 0; i < S; i++) {
        const int L = lens ? lens[q] : S;
        const double *qi = qkv + (q * S + i) * 3 * C + h * AT_HD; double m = -1e300;
        for (int j = 0; j < L; j++) { const double *kj = qkv + (q * S + j) * 3 * C + C + h * AT_HD; double a = 0; for (int d = 0; d < AT_HD; d++) a += qi[d] * kj[d]; s[j] = a * sc; if (s[j] > m) m = s[j]; }
        double l = 0; for (int j = 0; j < L; j++) { s[j] = exp(s[j] - m); l += s[j]; }
        double *oi = o + (q * S + i) * C + h * AT_HD;
        for (int d = 0; d < AT_HD; d++) oi[d] = 0;
        for (int j = 0; j < L; j++) { const double *vj = qkv + (q * S + j) * 3 * C + 2 * C + h * AT_HD; for (int d = 0; d < AT_HD; d++) oi[d] += s[j] / l * vj[d]; }
    }
    free(s);
}
typedef struct { at_dims d; at_bp bp; at_geom g; const rays_t *rt; const double *rpos, *gpos; } blk;
static void ref_block(const blk *B, const double *P, const double *x, double *y) {
    const int C = B->d.C, F = B->d.F; const long T = (long)B->g.nseq * B->g.S; const at_bp *bp = &B->bp;
    double *xn = malloc(T * C * 8), *qkv = malloc(T * 3 * C * 8), *o = malloc(T * C * 8), *x1 = malloc(T * C * 8), *h = malloc(T * 2 * F * 8), *gg = malloc(T * F * 8);
    ref_ln(x, P + bp->ln1g, P + bp->ln1b, xn, T, C);
    ref_lin(xn, P + bp->wqkv, P + bp->bqkv, qkv, T, C, 3 * C, 0);
    if (B->rt) {
        const rays_t *t = B->rt; double *qr = malloc(t->Tr * 3 * C * 8), *orr = malloc(t->Tr * C * 8);
        ref_gather(t, qkv, qr, 3 * C, 0);
        ref_attn_seq(qr, orr, (long)t->n * t->R.Q, t->R.Nr, t->lens_h, B->d.rope == 2 ? B->rpos : nullptr, 0, C, B->d.rope_base, P + bp->qg, P + bp->kg);
        ref_scatter(t, orr, o, C, 1);
        free(qr); free(orr);
    } else ref_attn_seq(qkv, o, B->g.nseq, B->g.S, nullptr, B->gpos, B->d.nreg, C, B->d.rope_base, P + bp->qg, P + bp->kg);
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
/* radial: 1 (rays, rope 0 / 2), 0 full attention with scroll RoPE (nreg registers) */
static int test_block(int radial, int rope, int nreg, int D, int Hh, int W, float mult, int npts) {
    const int n = 3, ads[3] = {0, 1, 2}, kinds[3] = {0, 1, 2}, C = 96, F = 128;
    float *axr = make_axr(n, D, Hh, W, ads, kinds);
    rays_t t; rays_make(&t, n, D, Hh, W, mult, axr);
    blk B = {0}; B.d.C = C; B.d.H = C / AT_HD; B.d.F = F; B.d.rope = rope; B.d.nreg = radial ? 0 : nreg; B.d.rope_base = 100.f;
    B.g = (at_geom){n, (int)(B.d.nreg + t.Sg), D, Hh, W, nullptr, nullptr, nullptr};
    const long T = (long)n * B.g.S, Tr = radial ? t.Tr : 0;
    float *gpos_d = nullptr; double *gpos = nullptr, *rpos = nullptr;
    int bad = 0;
    if (radial) { B.g.rays = &t.R; B.rt = &t; rpos = host(t.pos_d, (size_t)t.Tr * 3); B.rpos = rpos; }
    else {
        gpos_d = nn_malloc((size_t)T * 12); at_scroll_pos(n, B.d.nreg, D, Hh, W, t.ad_d, t.cen_d, gpos_d); gpos = host(gpos_d, (size_t)T * 3);
        B.g.pos = gpos_d; B.gpos = gpos;
        double pe = 0;   /* the positions against the host formula (cen: cu, cv, rc, phi_m per plane) */
        for (long tt = 0; tt < T; tt++) {
            const int b = (int)(tt / B.g.S), ts = (int)(tt % B.g.S) - B.d.nreg; double r[3] = {0, 0, 0};
            if (ts >= 0) {
                const int z = ts / (Hh * W), yy = (ts / W) % Hh, xx = ts % W, ad = t.ad_h[b], k = ad == 0 ? z : ad == 1 ? yy : xx, u = ad == 0 ? yy : z, v = ad == 2 ? yy : xx;
                const float *c = t.cen_h + ((size_t)b * t.P + k) * 4;
                if (c[2] < 0) { r[0] = v; r[1] = u; r[2] = k; }
                else { const double du = u - c[0], dv = v - c[1], th = c[3] + remainder(atan2(dv, du) - c[3], 2 * M_PI); r[0] = hypot(du, dv); r[1] = th * c[2]; r[2] = k; }
            }
            for (int a = 0; a < 3; a++) pe = fmax(pe, fabs(gpos[3 * tt + a] - r[a]) / fmax(1, fabs(r[a])));
        }
        printf("  scroll positions vs host: max rel error %.1e%s\n", pe, pe < 1e-4 ? "" : "  FAIL"); bad += !(pe < 1e-4);
    }
    printf("block (core %d): %s, rope %d, %d samples of %d x %d x %d tokens%s, %ld ray slots\n", at_get_core(), radial ? "radial attention" : "full attention",
           rope, n, D, Hh, W, B.d.nreg ? " + registers" : "", Tr);
    const size_t np = at_block_layout(&B.d, &B.bp, 0); const at_bp *bp = &B.bp;
    double *P = malloc(np * 8), *x = malloc(T * C * 8), *R = malloc(T * C * 8);
    for (size_t i = 0; i < np; i++) P[i] = rf(0.1 * nrand());
    for (int c = 0; c < C; c++) { P[bp->ln1g + c] = rf(1 + 0.2 * nrand()); P[bp->ln2g + c] = rf(1 + 0.2 * nrand()); P[bp->qg + c] = rf(1 + 0.2 * nrand()); P[bp->kg + c] = rf(1 + 0.2 * nrand()); }
    for (size_t i = 0; i < (size_t)3 * C * C; i++) P[bp->wqkv + i] = rf(nrand() / sqrt(C) * 1.5);
    for (size_t i = 0; i < (size_t)C * C; i++) P[bp->wo + i] = rf(nrand() / sqrt(C));
    for (size_t i = 0; i < (size_t)2 * F * C; i++) P[bp->w1 + i] = rf(nrand() / sqrt(C));
    for (size_t i = 0; i < (size_t)C * F; i++) P[bp->w2 + i] = rf(nrand() / sqrt(F));
    for (long i = 0; i < T * C; i++) { x[i] = rf(nrand()); R[i] = rf(nrand()); }
    float *Pd = dev(P, np), *xd = dev(x, T * C), *gd = dev(R, T * C), *yd = nn_malloc(T * C * 4), *Gd = nn_malloc(np * 4); nn_zero(Gd, np * 4);
    at_ws w; void *wb = nn_malloc(at_ws_bytes_r(&B.d, T, Tr)); at_ws_carve_r(&w, &B.d, T, Tr, wb);
    if (at_block_fwd(&B.d, Pd, bp, xd, yd, &B.g, &w)) { printf("  forward failed\n"); return 1; }
    { double *yg = host(yd, T * C), *yr = malloc(T * C * 8); ref_block(&B, P, x, yr); bad += report("forward", yg, yr, T * C); free(yg); free(yr); }
    if (at_block_bwd(&B.d, Pd, Gd, bp, xd, gd, &B.g, &w)) { printf("  backward failed\n"); return 1; }
    double *gx = host(gd, T * C), *gP = host(Gd, np);
    struct { const char *nm; size_t off, n; } ts[] = {{"x", 0, (size_t)T * C}, {"ln1g", bp->ln1g, C}, {"wqkv", bp->wqkv, (size_t)3 * C * C}, {"bqkv", bp->bqkv, (size_t)3 * C},
        {"qg", bp->qg, C}, {"kg", bp->kg, C}, {"wo", bp->wo, (size_t)C * C}, {"bo", bp->bo, C}, {"w1", bp->w1, (size_t)2 * F * C}, {"w2", bp->w2, (size_t)C * F}};
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
    free(a); free(r); free(gx); free(gP); free(P); free(x); free(R); free(axr); free(rpos); free(gpos);
    nn_free(Pd); nn_free(xd); nn_free(gd); nn_free(yd); nn_free(Gd); nn_free(wb); if (gpos_d) nn_free(gpos_d); rays_free(&t);
    return bad;
}

/* the fp16 core with ray lengths: masked keys scoring far above the valid ones (a padded ray slot against queries whose valid scores are very
   negative: exp2(s - lse) overflowed to inf, x 0 = NaN in the masked rows' dK / dV, then the weight gradients) -> finite, masked rows 0 */
static int test_masked_keys(void) {
    const int nseq = 2, S = 20, C = AT_HD, H = 1, lens_h[2] = {7, 20}; const long T = (long)nseq * S, Tp = T + 64;
    float *qkv = malloc((size_t)T * 3 * C * 4), *gO = malloc((size_t)T * C * 4), *g1 = malloc((size_t)C * 4), *dq = malloc((size_t)T * 3 * C * 4), *o = malloc((size_t)T * C * 4);
    for (int c = 0; c < C; c++) g1[c] = 8.f;
    for (long t = 0; t < T; t++) {
        const int s = (int)(t % S), sq = (int)(t / S), masked = s >= lens_h[sq];
        for (int c = 0; c < C; c++) {
            const float dir = c & 1 ? 1.f : -1.f;   /* queries + masked keys along dir, valid keys against it */
            qkv[t * 3 * C + c] = dir + 0.01f * (float)nrand();
            qkv[t * 3 * C + C + c] = (masked ? dir : -dir) + 0.01f * (float)nrand();
            qkv[t * 3 * C + 2 * C + c] = (float)nrand();
            gO[t * C + c] = (float)nrand();
        }
    }
    float *qkv_d = nn_malloc((size_t)T * 3 * C * 4), *gO_d = nn_malloc((size_t)T * C * 4), *g_d = nn_malloc((size_t)C * 4), *o_d = nn_malloc((size_t)T * C * 4);
    float *lse = nn_malloc((size_t)T * 4), *Dv = nn_malloc((size_t)T * 4), *amx = nn_malloc(16), *rq = nn_malloc((size_t)T * 4), *rk = nn_malloc((size_t)T * 4), *dq_d = nn_malloc((size_t)T * 3 * C * 4);
    int *lens_d = nn_malloc(8); void *qkvh = nn_malloc((size_t)3 * H * Tp * C * 2), *doh = nn_malloc((size_t)H * Tp * C * 2);
    nn_h2d(qkv_d, qkv, (size_t)T * 3 * C * 4); nn_h2d(gO_d, gO, (size_t)T * C * 4); nn_h2d(g_d, g1, (size_t)C * 4); nn_h2d(lens_d, lens_h, 8);
    nn_zero(dq_d, (size_t)T * 3 * C * 4);
    const at_geom g = {nseq, S, 1, 1, S, lens_d, nullptr, nullptr};
    at_fa_pack(qkv_d, g_d, g_d, qkvh, rq, rk, amx, &g, 0, C, 0, 100.f);
    at_fa_fwd(qkvh, o_d, lse, &g, H, C);
    at_fa_bwd(qkvh, doh, o_d, lse, gO_d, Dv, amx, dq_d, &g, H, C);
    nn_sync(); nn_d2h(dq, dq_d, (size_t)T * 3 * C * 4); nn_d2h(o, o_d, (size_t)T * C * 4);
    long nonfin = 0, mnz = 0;
    for (long i = 0; i < T * C; i++) nonfin += !isfinite(o[i]);
    for (long t = 0; t < T; t++) for (int c = 0; c < 3 * C; c++) {
        const float v = dq[t * 3 * C + c]; nonfin += !isfinite(v);
        if (c >= C && (int)(t % S) >= lens_h[t / S] && v != 0.f) mnz++;
    }
    const int bad = nonfin || mnz;
    printf("fp16 core, masked keys above the valid scores: %ld non-finite, %ld non-zero masked key gradients%s\n", nonfin, mnz, bad ? "  FAIL" : "");
    free(qkv); free(gO); free(g1); free(dq); free(o);
    nn_free(qkv_d); nn_free(gO_d); nn_free(g_d); nn_free(o_d); nn_free(lse); nn_free(Dv); nn_free(amx); nn_free(rq); nn_free(rk); nn_free(dq_d); nn_free(lens_d); nn_free(qkvh); nn_free(doh);
    return bad;
}

int main(void) {
    if (nn_init(0)) { printf("SKIP: no GPU\n"); return 0; }
    int bad = 0;
    bad += test_masked_keys();
    bad += test_maps(6, 7, 5, 2.f);
    bad += test_maps(8, 8, 8, 1.f);
    for (int core = 0; core < 2; core++) {
        at_set_core(core); printf("== attention core %d\n", core);
        bad += test_block(1, 0, 0, 4, 5, 3, 1.f, 8);
        bad += test_block(1, 2, 0, 4, 5, 3, 1.f, 8);
        bad += test_block(0, 2, 2, 3, 4, 5, 1.f, 8);
    }
    at_set_core(0);
    printf(bad ? "test_attn_ray: %d FAILED\n" : "test_attn_ray: all passed\n", bad);
    return bad != 0;
}
