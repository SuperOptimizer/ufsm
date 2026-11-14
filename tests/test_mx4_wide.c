/* The MX-fp4 / fp8 conv kernels at the try6 production widths (audit 2026-11-14 kernels.md #8, #9, #10, #12, #13) against host
   references. Exact-operand construction: every operand sits on the e2m1 grid with MX block scales that the stored blocks reproduce
   losslessly (inputs: every 32-channel block of a voxel holds a +-6, so its scale is 1; weights: every (co, tap, 32-ci block) holds a
   +-6; gradients: grid values x 2^-6), so every product and partial sum is exact in fp32 whatever the MMA summation order, and the host
   fp64 reference is the kernel's exact result:
   - fp32 outputs (the verify-only mx4 -> fp32 instantiation of the same kernels): equal to the reference;
   - MX-fp4 outputs rounded to nearest (the production mx4 -> mx4 instantiation): the decoded values bit-identical to the host
     quantiser (the epilogue's mx_exp + e2m1 round-to-nearest-even) applied to the reference;
   - MX-fp4 gradients with exact stochastic rounding: every value the floor or the ceiling e2m1 neighbour of the exact value at the
     block scale of the exact values (sr_e2m1_nib), and a split output bit-identical to the unsplit output under the same seed;
   - GroupNorm + SiLU inputs (tanh.approx: not host-reproducible): the staged operand is read back exactly through an identity conv
     (centre tap 1, fp32 out), checked against the host GN+SiLU (each value the floor / ceiling neighbour of the exact value, the block
     scale 1 by construction), then used as the exact operand of the real conv; the prenorm path (several output tiles,
     gn_prenorm_f4_k) equals the in-kernel staging (one output tile) bit for bit;
   - the fused 2x upsample of the decoder conv1 (coarse rows to e4m3, half2 blend, e2m1 requantisation): exact in fp16 on grid values,
     so the staged up values equal the host blend + quantiser bit for bit;
   - weight gradients: exact sums (atomics in any order) -> equal to the reference; slab splits of the gy pre-pass invariant.
   Not exact (tolerances, justified at the checks): the fused-upsample weight gradient (bf16 tile + fp4 spatial requantisation of the
   interpolated values), the down conv with a GroupNorm input (fp4 output rounding), the MX-fp4 GroupNorm backward (tanh.approx).
   Shapes (try6: P 768 / lat 4 -> grid 192, widths 256,320,384,448,288,288; dec c1 = [up2 | skip]):
   #8  forward 256->256, 256->320, 448->288, 288->288 (12x10x14 ragged, 8x8x16, 6^3): launch_f4<4,2> / <2,2>; GN+SiLU input (prenorm)
       256->256 and 448->288 (+ the 448->448 identity conv); dec c1 up 320 (coarse) + skip 256 -> 256, up 288 + skip 448 -> 448
   #9  backward-data of those dec c1 convs into the split (up | skip) MX-fp4 gradient with output SR (launch_f4<4,2> / <2,4>);
       weight gradient with fp4 gy (warp-specialised conv_bwd_w_f4ws_k<2,2>; launch_bw4<2,2> with the fused upsample) at 576->256,
       448->288, 448->448 (24^3) with one-z-step gy pre-pass slabs, and with a pending (deferred) GN+SiLU backward apply
   #10 down convs 384 / 448 / 288 at 48 / 24 / 12 -> 24 / 12 / 6: forward launch_s2f8<4|2, mx4>, backward-data through the mx4 -> mx8
       recode + s2b_dilate + launch_f8g (accumulating, SR), the FAST stride-2 weight gradient with fp4 gy
   #12 the MX-fp4 GroupNorm + SiLU backward at 256 and 448 channels, G 8 (try6's; 448: 56 channels a group) against host fp64
   #13 (not with UFSM_TEST_QUICK: test_mx4_wide~full) one 256->256 conv at the level-0 size 192^3 (1.8e9 elements, 0.96 GB a tensor):
       forward / backward-data / weight gradient, sampled near the end of the tensors against host references */
#include "nn.h"
#include "nn_lp.h"
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int bad, quick;
static uint64_t rs = 0x9e3779b97f4a7c15ull;
static uint64_t rnd64(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static unsigned rnd(void) { return (unsigned)(rnd64() >> 32); }
static float frand(void) { return (float)((rnd64() >> 40) * 0x1.0p-24); }   /* [0, 1) */
static float urange(float a, float b) { return a + (b - a) * frand(); }
static const float G8[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
static float rgrid(void) { const unsigned r = rnd(); return (r & 8 ? -1.f : 1.f) * G8[r & 7]; }
static void check(int ok, const char *fmt, ...) {
    char m[320]; va_list ap; va_start(ap, fmt); vsnprintf(m, sizeof m, fmt, ap); va_end(ap);
    printf("  %-128s %s\n", m, ok ? "ok" : "FAIL"); fflush(stdout);
    if (!ok) bad++;
}
static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) { fprintf(stderr, "test_mx4_wide: out of host memory (%zu bytes)\n", n); exit(2); } return p; }
static float *hnew(size_t n) { return xmalloc(n * 4); }
static void gpu_ok(const char *nm) {
    nn_sync(); const char *e = nn_check(), *e2 = lp_check();
    check(!e && !e2, "%s: no CUDA error%s%s%s%s", nm, e ? ": " : "", e ? e : "", e2 ? ": " : "", e2 ? e2 : "");
}
/* ---------------- device helpers ---------------- */
static void *dmx4(shape5 s) { const size_t b = nn_mx4_bytes(s); void *p = nn_malloc(b); nn_zero(p, b); nn_set_storage(p, b, 4); return p; }
static float *dput(const float *h, size_t n) { float *d = nn_malloc(n * 4); nn_h2d(d, h, n * 4); return d; }
static float *dzero(size_t n) { float *d = nn_malloc(n * 4); nn_zero(d, n * 4); return d; }
static float *hget(const float *d, size_t n) { float *h = hnew(n); nn_d2h(h, d, n * 4); return h; }
static float *hdeq4(const void *p, shape5 s) {   /* decoded MX-fp4 tensor on the host */
    const size_t n = shape_numel(s); float *d = nn_malloc(n * 4); lp_mx4_to_f32(p, s.n, s.c, shape_spatial(s), d); float *h = hget(d, n); nn_free(d); return h;
}
static void *dmx4_from(const float *h, shape5 s) {   /* MX-fp4 (round to nearest) of a host tensor */
    const size_t n = shape_numel(s); float *d = dput(h, n); void *p = dmx4(s); lp_f32_to_mx4(d, s.n, s.c, shape_spatial(s), p); nn_free(d); return p;
}
static void *dmx4_exact(const float *h, shape5 s, const char *nm) {   /* the same, checked lossless: the kernels read exactly the host values */
    void *p = dmx4_from(h, s); float *r = hdeq4(p, s); const size_t n = shape_numel(s); size_t nd = 0;
    for (size_t i = 0; i < n; i++) nd += r[i] != h[i];
    if (nd) check(0, "%s: MX-fp4 encoding of a grid operand lossless (%zu of %zu values differ)", nm, nd, n);
    free(r); return p;
}
static double rel2(const float *a, const float *r, size_t n) {
    double d2 = 0, r2 = 0; for (size_t i = 0; i < n; i++) { const double d = (double)a[i] - r[i]; d2 += d * d; r2 += (double)r[i] * r[i]; }
    return r2 > 0 ? sqrt(d2 / r2) : (d2 > 0 ? 1e9 : 0);
}
/* ---------------- operands ---------------- */
static float *gen_x(int C, size_t S, int pos6) {   /* grid values; every (32-channel block, voxel) holds a +-6 (+6 with pos6): scale 1 */
    float *x = hnew((size_t)C * S);
    for (size_t i = 0; i < (size_t)C * S; i++) x[i] = rgrid();
    for (int c0 = 0; c0 < C; c0 += 32) for (size_t v = 0; v < S; v++) x[(size_t)(c0 + rnd() % 32) * S + v] = pos6 || (rnd() & 1) ? 6.f : -6.f;
    return x;
}
static float *gen_x6(int C, size_t S) {   /* grid values, channel 5 of every block +6 everywhere (the coarse up input: every blend's amax is 6) */
    float *x = hnew((size_t)C * S);
    for (size_t i = 0; i < (size_t)C * S; i++) x[i] = rgrid();
    for (int c0 = 0; c0 < C; c0 += 32) for (size_t v = 0; v < S; v++) x[(size_t)(c0 + 5) * S + v] = 6.f;
    return x;
}
static float *gen_w(int Co, int Ci) {   /* w[co][ci][27] on the grid, every (co, tap, 32-ci block) with a +-6: prep_w4_k / prep_w8_k lossless */
    float *w = hnew((size_t)Co * Ci * 27);
    for (size_t i = 0; i < (size_t)Co * Ci * 27; i++) w[i] = rgrid();
    for (int co = 0; co < Co; co++) for (int t = 0; t < 27; t++) for (int c0 = 0; c0 < Ci; c0 += 32) w[((size_t)co * Ci + c0 + rnd() % 32) * 27 + t] = rnd() & 1 ? 6.f : -6.f;
    return w;
}
static float *gen_g(size_t n, float sc) { float *g = hnew(n); for (size_t i = 0; i < n; i++) g[i] = rgrid() * sc; return g; }
static float *gen_b(int C) { float *b = hnew(C); for (int c = 0; c < C; c++) b[c] = rgrid() * 0.25f; return b; }
static float *wtrans(const float *w, int Co, int Ci) {   /* wT[ci][co][t] = w[co][ci][t]: the backward-data's weights for the transposed host conv */
    float *t = hnew((size_t)Co * Ci * 27);
    for (int co = 0; co < Co; co++) for (int ci = 0; ci < Ci; ci++) memcpy(t + ((size_t)ci * Co + co) * 27, w + ((size_t)co * Ci + ci) * 27, 27 * 4);
    return t;
}
/* GroupNorm parameters (N 1, G groups): wide 0 keeps GN+SiLU of a +6 in [3.4, 5.5] and of every grid value <= 5.6 (block scale 1) */
typedef struct { float *gam, *bet, *mean, *rstd, *dg, *db, *dm, *dr; int C, G; } gnpar_t;
static gnpar_t gn_make(int C, int G, int wide) {
    gnpar_t p = {hnew(C), hnew(C), hnew(G), hnew(G), nullptr, nullptr, nullptr, nullptr, C, G};
    for (int c = 0; c < C; c++) { p.gam[c] = wide ? urange(0.5f, 1.5f) : urange(0.75f, 0.95f); p.bet[c] = wide ? urange(-0.5f, 0.5f) : urange(-0.05f, 0.05f); }
    for (int g = 0; g < G; g++) { p.mean[g] = wide ? urange(-0.2f, 0.2f) : urange(-0.05f, 0.05f); p.rstd[g] = wide ? urange(0.8f, 1.2f) : urange(0.8f, 0.95f); }
    p.dg = dput(p.gam, C); p.db = dput(p.bet, C); p.dm = dput(p.mean, G); p.dr = dput(p.rstd, G);
    return p;
}
static void gn_free(gnpar_t *p) { free(p->gam); free(p->bet); free(p->mean); free(p->rstd); nn_free(p->dg); nn_free(p->db); nn_free(p->dm); nn_free(p->dr); }
static gnp_t gn_gp(const gnpar_t *p) { gnp_t g = {}; g.gamma = p->dg; g.beta = p->db; g.mean = p->dm; g.rstd = p->dr; g.G = p->G; return g; }
/* the staging's GN+SiLU of channel c (make_chan: a = rstd gamma, b = beta - mean a; act_ab: u = fma(x, a, b)), SiLU in double */
static double gn_silu(const gnpar_t *p, int c, float x) {
    const int g = c / (p->C / p->G); const float a = p->rstd[g] * p->gam[c], b = fmaf(-p->mean[g], a, p->bet[c]);
    const double u = fmaf(x, a, b); return u / (1.0 + exp(-u));
}
/* ---------------- host MX quantisers ---------------- */
static int hmx_exp(float amax, float inv) {   /* lp_common.cuh mx_exp: the exponent of amax * (1 / qmax) rounded up, clamped */
    const float p = amax * inv; uint32_t u; memcpy(&u, &p, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 128;
    const int e = (int)((u >> 23) & 255) - 127 + ((u & 0x7fffffu) != 0);
    return e < -126 ? -126 : e > 126 ? 126 : e;
}
static float e2m1_rne(float a) {   /* a >= 0 in grid units: cvt.rn.satfinite.e2m1 (ties to the even code) */
    if (a >= 6.f) return 6.f;
    int i = 0; while (i < 7 && G8[i + 1] <= a) i++;
    const float lo = G8[i], hi = G8[i + 1], d = a - lo, u = hi - a;
    if (d < u) return lo; if (u < d) return hi;
    return (i & 1) ? hi : lo;
}
static float e4m3_rne(float a) {   /* a >= 0: cvt.rn.satfinite.e4m3 (max 448, binade [2^e, 2^e+1) spacing 2^(e-3), subnormals 2^-9) */
    if (a >= 448.f) return 448.f;
    int ex; frexpf(a, &ex);
    const int e = ex - 1 < -6 ? -6 : ex - 1;
    const double q = ldexp(1.0, e - 3), r = a / q, f = floor(r), d = r - f;
    double k = f; if (d > 0.5 || (d == 0.5 && fmod(f, 2.0) != 0.0)) k += 1.0;
    const double v = k * q; return (float)(v > 448.0 ? 448.0 : v);
}
static void q_rn(const float *v, int n, float *o, int bits) {   /* an MX block rounded to nearest (fp4: e2m1, fp8: e4m3) and decoded */
    float am = 0.f; for (int k = 0; k < n; k++) am = fmaxf(am, fabsf(v[k]));
    const int e = hmx_exp(am, bits == 4 ? 1.f / 6.f : 1.f / 448.f); const float m = ldexpf(1.f, -e), sc = ldexpf(1.f, e);
    for (int k = 0; k < n; k++) { const float a = fabsf(v[k]) * m; o[k] = copysignf((bits == 4 ? e2m1_rne(a) : e4m3_rne(a)) * sc, v[k]); }
}
static int code_of(float a) { for (int k = 0; k < 8; k++) if (G8[k] == a) return k; return -1; }
/* an MX-fp4 block stored with exact stochastic rounding (sr_e2m1_nib): the scale from the amax of the exact values v, every code the
   floor or the ceiling of t = min(2a, a + 2, a / 2 + 4) (a = |v| / scale: the code as a continuous index), the sign kept */
static int sr_ok(const float *v, const float *g, int n) {
    float am = 0.f; for (int k = 0; k < n; k++) am = fmaxf(am, fabsf(v[k]));
    const float m = ldexpf(1.f, -hmx_exp(am, 1.f / 6.f));
    for (int k = 0; k < n; k++) {
        const double a = fabs((double)v[k]) * m, t = fmin(fmin(2 * a, a + 2), fmin(a / 2 + 4, 7.0));
        const int lo = (int)floor(t), hi = t > lo ? lo + 1 : lo, c = code_of(fabsf(g[k]) * m);
        if (c < lo || c > hi) return 0;
        if (c && (g[k] < 0) != (v[k] < 0)) return 0;
    }
    return 1;
}
/* a block rounded to nearest from slightly perturbed values (GN+SiLU through tanh.approx): every code the floor or the ceiling e2m1
   neighbour of the exact value at the exact values' block scale */
static int rn_near_ok(const double *v, const float *g, int n) {
    double am = 0; for (int k = 0; k < n; k++) am = fmax(am, fabs(v[k]));
    const float m = ldexpf(1.f, -hmx_exp((float)am, 1.f / 6.f));
    for (int k = 0; k < n; k++) {
        const double a = fmin(fabs(v[k]) * m, 6.0); int lo = 0; while (lo < 7 && G8[lo + 1] <= a) lo++;
        const int hi = G8[lo] < a ? lo + 1 : lo, c = code_of(fabsf(g[k]) * m);
        if (c < lo || c > hi) return 0;
        if (c && (g[k] < 0) != (v[k] < 0)) return 0;
    }
    return 1;
}
/* ---------------- host convolutions ---------------- */
enum { CV_S1, CV_S2, CV_S1T, CV_S2T };   /* stride 1 / 2 forward (pad 1); their adjoints (backward-data: gx from gy) */
typedef struct { const float *x; int Ci, D, H, W; const float *w; int Co; const float *b; int mode, Do, Ho, Wo; } cref_t;   /* x [Ci][D][H][W], w [Co][Ci][27] */
static void gather(const cref_t *c, int z, int y, int x, float *p) {   /* p[ci * 27 + t]: the input each tap of output voxel (z, y, x) reads */
    const size_t S = (size_t)c->D * c->H * c->W;
    for (int t = 0; t < 27; t++) {
        const int kz = t / 9 - 1, ky = (t / 3) % 3 - 1, kx = t % 3 - 1;
        int iz, iy, ix, ok = 1;
        switch (c->mode) {
        case CV_S1: iz = z + kz; iy = y + ky; ix = x + kx; break;
        case CV_S2: iz = 2 * z + kz; iy = 2 * y + ky; ix = 2 * x + kx; break;
        case CV_S1T: iz = z - kz; iy = y - ky; ix = x - kx; break;
        default: {   /* gx[u] = sum_k w[k] gy[o], u = 2 o + k - 1 */
            const int nz = z - kz, ny = y - ky, nx = x - kx;
            ok = nz >= 0 && ny >= 0 && nx >= 0 && !(nz & 1) && !(ny & 1) && !(nx & 1);
            iz = nz / 2; iy = ny / 2; ix = nx / 2;
        }
        }
        ok = ok && iz >= 0 && iz < c->D && iy >= 0 && iy < c->H && ix >= 0 && ix < c->W;
        const size_t off = ok ? ((size_t)iz * c->H + iy) * c->W + ix : 0;
        for (int ci = 0; ci < c->Ci; ci++) p[(size_t)ci * 27 + t] = ok ? c->x[(size_t)ci * S + off] : 0.f;
    }
}
static double dotw(const float *p, const float *w, size_t n) {   /* (exact operands: the order of the double sums is immaterial) */
    double s[8] = {0, 0, 0, 0, 0, 0, 0, 0}; size_t i = 0;
    for (; i + 8 <= n; i += 8) for (int j = 0; j < 8; j++) s[j] += (double)p[i + j] * w[i + j];
    for (; i < n; i++) s[0] += (double)p[i] * w[i];
    return ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
}
/* reference outputs: every output channel at nv sampled voxels (the 8 corners + random), and the channels cs at every voxel (So <= 4096)
   or at the voxels of 5 z planes; inexact: reference values not representable in fp32 (the exactness premise broken) */
typedef struct { double *r; uint8_t *have; int Co, nv; size_t So, vs[64]; } rset_t;
static rset_t rset_make(const cref_t *c, const int *cs, int ncs, int nrand, size_t *inexact) {
    rset_t r = {}; r.Co = c->Co; r.So = (size_t)c->Do * c->Ho * c->Wo;
    r.r = xmalloc((size_t)r.Co * r.So * 8); r.have = calloc((size_t)r.Co * r.So, 1); if (!r.have) exit(2);
    uint8_t *vm = calloc(r.So, 1); if (!vm) exit(2);
    for (int i = 0; i < 8; i++) r.vs[r.nv++] = ((size_t)((i & 4) ? c->Do - 1 : 0) * c->Ho + ((i & 2) ? c->Ho - 1 : 0)) * c->Wo + ((i & 1) ? c->Wo - 1 : 0);
    for (int i = 0; i < nrand && r.nv < 64; i++) r.vs[r.nv++] = rnd() % r.So;
    for (int i = 0; i < r.nv; i++) vm[r.vs[i]] = 1;
    uint8_t *cm = calloc(r.Co, 1); if (!cm) exit(2);
    for (int i = 0; i < ncs; i++) cm[cs[i]] = 1;
    const size_t K = (size_t)c->Ci * 27; float *p = hnew(K);
    *inexact = 0;
    for (size_t v = 0; v < r.So; v++) {
        const int z = (int)(v / ((size_t)c->Ho * c->Wo)), y = (int)(v / c->Wo % c->Ho), x = (int)(v % c->Wo);
        const int allv = ncs && (r.So <= 4096 || z == 0 || z == 1 || z == c->Do / 2 || z == c->Do - 2 || z == c->Do - 1);
        if (!vm[v] && !allv) continue;
        gather(c, z, y, x, p);
        for (int co = 0; co < c->Co; co++) {
            if (!vm[v] && !cm[co]) continue;
            const double s = (c->b ? (double)c->b[co] : 0.0) + dotw(p, c->w + (size_t)co * K, K);
            r.r[(size_t)co * r.So + v] = s; r.have[(size_t)co * r.So + v] = 1;
            if ((double)(float)s != s) (*inexact)++;
        }
    }
    free(p); free(vm); free(cm);
    return r;
}
static void rset_free(rset_t *r) { free(r->r); free(r->have); }
static int mkcset(int Co, int osplit, int *cs) {   /* tile / block / split edges */
    const int cand[] = {0, 1, 15, 16, 31, 32, 63, 64, 127, Co / 2 + 7, Co - 33, Co - 32, Co - 1, osplit - 1, osplit, osplit + 31};
    int n = 0;
    for (int i = 0; i < (int)(sizeof cand / sizeof cand[0]); i++) {
        const int c = cand[i]; if (c < 0 || c >= Co) continue;
        int dup = 0; for (int j = 0; j < n; j++) dup |= cs[j] == c;
        if (!dup) cs[n++] = c;
    }
    return n;
}
static void cmp_f32(const char *nm, const rset_t *r, const float *g) {   /* every reference value == the fp32 output */
    size_t n = 0, nd = 0;
    for (size_t i = 0; i < (size_t)r->Co * r->So; i++) if (r->have[i]) { n++; nd += (float)r->r[i] != g[i]; }
    check(n && !nd, "%s: fp32 output == exact host reference (%zu values, %zu differ)", nm, n, nd);
}
/* an MX output (decoded g: channels c0 .. c0 + nc of the reference, [nc][So]) at the sampled voxels, block by block: mode 0 round to
   nearest (decoded values bit-identical to the host quantiser), 1 exact SR (e2m1 neighbours) */
static void cmp_mx(const char *nm, const rset_t *r, const float *g, int c0, int nc, int sr) {
    size_t nb = 0, nbad = 0; float v[32], q[32], gg[32];
    for (int j = 0; j < r->nv; j++) for (int b0 = 0; b0 < nc; b0 += 32) {
        const size_t vox = r->vs[j]; const int n = nc - b0 < 32 ? nc - b0 : 32;
        for (int k = 0; k < n; k++) { v[k] = (float)r->r[(size_t)(c0 + b0 + k) * r->So + vox]; gg[k] = g[(size_t)(b0 + k) * r->So + vox]; }
        int ok = 1;
        if (sr) ok = sr_ok(v, gg, n);
        else { q_rn(v, n, q, 4); for (int k = 0; k < n; k++) ok &= q[k] == gg[k]; }
        nb++; nbad += !ok;
    }
    check(nb && !nbad, "%s: %s (%zu blocks at %d voxels, %zu wrong)", nm,
          sr ? "MX-fp4 SR output = e2m1 neighbours of the exact values at the exact block scale" : "MX-fp4 output == host quantiser of the exact values (bit-identical)",
          nb, r->nv, nbad);
}
/* host weight gradient of one (co, ci): o[t] = sum_o gy[o] x[st o + k - 1] (x, gy: that channel's planes) */
static void wg27(const float *xc, int D, int H, int W, const float *gc, int Do, int Ho, int Wo, int st, double *o) {
    for (int t = 0; t < 27; t++) o[t] = 0;
    for (int oz = 0; oz < Do; oz++) for (int oy = 0; oy < Ho; oy++) for (int ox = 0; ox < Wo; ox++) {
        const float g = gc[((size_t)oz * Ho + oy) * Wo + ox]; if (g == 0.f) continue;
        for (int t = 0; t < 27; t++) {
            const int iz = st * oz + t / 9 - 1, iy = st * oy + (t / 3) % 3 - 1, ix = st * ox + t % 3 - 1;
            if (iz < 0 || iz >= D || iy < 0 || iy >= H || ix < 0 || ix >= W) continue;
            o[t] += (double)g * xc[((size_t)iz * H + iy) * W + ix];
        }
    }
}
static int mkpairs(int Co, int Ci, int csplit, size_t So, int (*p)[2]) {   /* (co, ci) pairs: block / split edges, the rest random (~1.2e8 host MACs) */
    int np = (int)(1.2e8 / (27.0 * (double)So)); if (np > 256) np = 256; if (np < 24) np = 24;
    const int co_[4] = {0, 31, 32, Co - 1}, ci_[6] = {0, 31, 32, Ci - 1, csplit > 0 ? csplit - 1 : 1, csplit > 0 ? csplit : 2};
    int k = 0;
    for (int a = 0; a < 4; a++) for (int b = 0; b < 6; b++) { p[k][0] = co_[a]; p[k][1] = ci_[b]; k++; }
    while (k < np) { p[k][0] = (int)(rnd() % Co); p[k][1] = (int)(rnd() % Ci); k++; }
    return k;
}
/* the weight gradient gw [Co][Ci][27] (+ bias gb) against the host on the sampled pairs: exact (tol 0) or rel L2 <= tol */
static void cmp_wg(const char *nm, const float *gw, const float *gb, const float *hx, int Ci, int D, int H, int W, const float *hg, int Co, int Do, int Ho, int Wo,
                   int st, int csplit, double tol) {
    static int pr[256][2]; const size_t S = (size_t)D * H * W, So = (size_t)Do * Ho * Wo;
    const int np = mkpairs(Co, Ci, csplit, So, pr);
    size_t nd = 0, inex = 0; double d2 = 0, r2 = 0, o[27];
    for (int i = 0; i < np; i++) {
        const int co = pr[i][0], ci = pr[i][1];
        wg27(hx + (size_t)ci * S, D, H, W, hg + (size_t)co * So, Do, Ho, Wo, st, o);
        for (int t = 0; t < 27; t++) {
            const float g = gw[((size_t)co * Ci + ci) * 27 + t];
            nd += (float)o[t] != g; inex += (double)(float)o[t] != o[t];
            d2 += ((double)g - o[t]) * ((double)g - o[t]); r2 += o[t] * o[t];
        }
    }
    const double e = r2 > 0 ? sqrt(d2 / r2) : 1e9;
    if (tol == 0) check(!nd && !inex, "%s: weight gradient == exact host reference (%d (co, ci) pairs x 27 taps, %zu differ%s)", nm, np, nd, inex ? ", reference inexact" : "");
    else check(e <= tol, "%s: weight gradient vs host fp64, rel L2 %.3g <= %.2g (%d pairs x 27 taps)", nm, e, tol, np);
    if (gb) {
        size_t nbd = 0;
        for (int co = 0; co < Co; co++) { double s = 0; for (size_t v = 0; v < So; v++) s += hg[(size_t)co * So + v]; nbd += (float)s != gb[co]; }
        check(!nbd, "%s: bias gradient == exact sum of gy (%zu of %d differ)", nm, nbd, Co);
    }
}
/* the staged GN+SiLU operand (read through an identity conv): every value the floor / ceiling e2m1 neighbour of the host GN+SiLU of the
   stored input, block scale 1 (the premise of the exact conv check that follows: every |value| on the scale-1 grid) */
static void check_staged_gn(const char *nm, const float *sf, const float *hx, const gnpar_t *p, int C, size_t S) {
    size_t nb = 0, nbad = 0, offg = 0, same = 0; double v[32];
    for (int b0 = 0; b0 < C; b0 += 32) for (size_t vox = 0; vox < S; vox++) {
        float g[32];
        for (int k = 0; k < 32; k++) {
            const size_t i = (size_t)(b0 + k) * S + vox; v[k] = gn_silu(p, b0 + k, hx[i]); g[k] = sf[i];
            offg += code_of(fabsf(g[k])) < 0;
        }
        float vf[32], q[32]; for (int k = 0; k < 32; k++) vf[k] = (float)v[k];
        q_rn(vf, 32, q, 4); for (int k = 0; k < 32; k++) same += q[k] == g[k];
        nb++; nbad += !rn_near_ok(v, g, 32);
    }
    check(!nbad && !offg, "%s: staged GN+SiLU (G %d, %d channels a group) = e2m1 neighbours of host GN+SiLU, scale 1 (%zu blocks, %zu wrong, %zu off-grid; %.4f%% == host RN)",
          nm, p->G, C / p->G, nb, nbad, offg, 100.0 * same / ((double)nb * 32));
}
static float *up_blend(const float *hc, int C, int D, int H, int W) {   /* fine D x H x W from the coarse D/2 x H/2 x W/2: trilinear 2x, align_corners false, edge clamp */
    const int Dc = D / 2, Hc = H / 2, Wc = W / 2; const size_t S = (size_t)D * H * W, Sc = (size_t)Dc * Hc * Wc;
    float *o = hnew((size_t)C * S); const double wt[2] = {0.75, 0.25};
    for (int c = 0; c < C; c++) for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        int mz[2] = {z >> 1, (z >> 1) + ((z & 1) ? 1 : -1)}, my[2] = {y >> 1, (y >> 1) + ((y & 1) ? 1 : -1)}, mx[2] = {x >> 1, (x >> 1) + ((x & 1) ? 1 : -1)};
        mz[1] = mz[1] < 0 ? 0 : mz[1] > Dc - 1 ? Dc - 1 : mz[1]; my[1] = my[1] < 0 ? 0 : my[1] > Hc - 1 ? Hc - 1 : my[1]; mx[1] = mx[1] < 0 ? 0 : mx[1] > Wc - 1 ? Wc - 1 : mx[1];
        double s = 0;
        for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) for (int e = 0; e < 2; e++) s += wt[a] * wt[b] * wt[e] * hc[(size_t)c * Sc + ((size_t)mz[a] * Hc + my[b]) * Wc + mx[e]];
        o[(size_t)c * S + ((size_t)z * H + y) * W + x] = (float)s;
    }
    return o;
}
static float *wident(int C) { float *w = hnew((size_t)C * C * 27); memset(w, 0, (size_t)C * C * 27 * 4); for (int c = 0; c < C; c++) w[((size_t)c * C + c) * 27 + 13] = 1.f; return w; }
static const gnp_t NOGN = {};

/* ================= #8 forward, stored (copy-staged) input ================= */
static void case_fwd_plain(int Ci, int Co, int D, int H, int W) {
    char nm[96]; snprintf(nm, sizeof nm, "fwd %d -> %d, %dx%dx%d", Ci, Co, D, H, W);
    const shape5 xs = {1, Ci, D, H, W}, ys = {1, Co, D, H, W}; const size_t S = shape_spatial(xs);
    float *hx = gen_x(Ci, S, 0), *hw = gen_w(Co, Ci), *hb = gen_b(Co);
    void *x4 = dmx4_exact(hx, xs, nm), *y4 = dmx4(ys); float *w = dput(hw, (size_t)Co * Ci * 27), *b = dput(hb, Co), *yf = dzero((size_t)Co * S);
    const split_t s0 = {};
    lp_conv_fwd_f4(x4, 4, xs, w, b, Co, yf, 0, NOGN, nullptr, 0, s0);
    lp_conv_fwd_f4(x4, 4, xs, w, b, Co, y4, 4, NOGN, nullptr, 0, s0);
    const cref_t cr = {hx, Ci, D, H, W, hw, Co, hb, CV_S1, D, H, W};
    int cs[16]; const int ncs = mkcset(Co, 0, cs); size_t inex;
    rset_t r = rset_make(&cr, cs, ncs, 16, &inex);
    if (inex) check(0, "%s: exact-operand premise (%zu reference values not representable in fp32)", nm, inex);
    float *g = hget(yf, (size_t)Co * S); cmp_f32(nm, &r, g); free(g);
    g = hdeq4(y4, ys); cmp_mx(nm, &r, g, 0, Co, 0); free(g);
    gpu_ok(nm);
    rset_free(&r); free(hx); free(hw); free(hb); nn_free(x4); nn_free(y4); nn_free(w); nn_free(b); nn_free(yf);
}
/* ================= #8 / #9 forward, GroupNorm + SiLU input (prenorm: gn_prenorm_f4_k, then copy staging) ================= */
static void case_fwd_gn(int Ci, int Co, int D, int H, int W) {
    char nm[96]; snprintf(nm, sizeof nm, "fwd gn+silu %d -> %d, %dx%dx%d", Ci, Co, D, H, W);
    const shape5 xs = {1, Ci, D, H, W}, ys = {1, Co, D, H, W}; const size_t S = shape_spatial(xs);
    float *hx = gen_x(Ci, S, 1); gnpar_t p = gn_make(Ci, 8, 0); const gnp_t gp = gn_gp(&p);
    void *x4 = dmx4_exact(hx, xs, nm);
    const split_t s0 = {};
    /* the staged operand: identity conv (Ci -> Ci, nmt >= 2: the prenorm path) */
    float *hid = wident(Ci), *wid = dput(hid, (size_t)Ci * Ci * 27), *sfd = dzero((size_t)Ci * S); free(hid);
    lp_conv_fwd_f4(x4, 4, xs, wid, nullptr, Ci, sfd, 0, gp, nullptr, 0, s0);
    float *sf = hget(sfd, (size_t)Ci * S); nn_free(sfd); nn_free(wid);
    check_staged_gn(nm, sf, hx, &p, Ci, S);
    /* the real conv on that operand: fp32 and MX-fp4 outputs exact */
    float *hw = gen_w(Co, Ci), *hb = gen_b(Co), *w = dput(hw, (size_t)Co * Ci * 27), *b = dput(hb, Co), *yf = dzero((size_t)Co * S); void *y4 = dmx4(ys);
    lp_conv_fwd_f4(x4, 4, xs, w, b, Co, yf, 0, gp, nullptr, 0, s0);
    lp_conv_fwd_f4(x4, 4, xs, w, b, Co, y4, 4, gp, nullptr, 0, s0);
    const cref_t cr = {sf, Ci, D, H, W, hw, Co, hb, CV_S1, D, H, W};
    int cs[16]; const int ncs = mkcset(Co, 0, cs); size_t inex;
    rset_t r = rset_make(&cr, cs, ncs, 16, &inex);
    if (inex) check(0, "%s: exact-operand premise (%zu reference values not representable in fp32)", nm, inex);
    float *g = hget(yf, (size_t)Co * S); cmp_f32(nm, &r, g);
    float *g4 = hdeq4(y4, ys); cmp_mx(nm, &r, g4, 0, Co, 0); free(g4);
    /* the in-kernel GN staging (64 outputs: one output tile, no prenorm): the same first 64 channels bit for bit */
    float *y64 = dzero((size_t)64 * S);
    lp_conv_fwd_f4(x4, 4, xs, w, b, 64, y64, 0, gp, nullptr, 0, s0);
    float *h64 = hget(y64, (size_t)64 * S);
    check(!memcmp(h64, g, (size_t)64 * S * 4), "%s: prenorm (%d output tiles) == in-kernel GN staging (one tile), first 64 channels bitwise", nm, Co / (Co % 64 ? 32 : 64));
    free(h64); free(g); nn_free(y64);
    gpu_ok(nm);
    rset_free(&r); gn_free(&p); free(hx); free(sf); free(hw); free(hb); nn_free(x4); nn_free(y4); nn_free(w); nn_free(b); nn_free(yf);
}
/* ================= #9 decoder conv1: [up2(coarse) | skip] -> Co, the upsample fused into the staging ================= */
static void case_dec_fwd(int Cx, int Cs, int Co, int D, int H, int W, int gn) {
    char nm[112]; snprintf(nm, sizeof nm, "dec c1 fwd up %d + skip %d%s -> %d, %dx%dx%d", Cx, Cs, gn ? " (gn+silu)" : "", Co, D, H, W);
    const int Ci = Cx + Cs;
    const shape5 fs = {1, Ci, D, H, W}, cs_ = {1, Cx, D / 2, H / 2, W / 2}, ks = {1, Cs, D, H, W}, ys = {1, Co, D, H, W};
    const size_t S = shape_spatial(fs), Sc = shape_spatial(cs_);
    float *hc = gen_x6(Cx, Sc), *hk = gen_x(Cs, S, gn);
    void *c4 = dmx4_exact(hc, cs_, nm), *k4 = dmx4_exact(hk, ks, nm);
    gnpar_t p2 = {}; if (gn) p2 = gn_make(Cs, 8, 0);
    split_t sp = {}; sp.x2 = k4; sp.c_split = Cx; sp.up = 1; if (gn) sp.gp2 = gn_gp(&p2);
    /* the up part as staged: the exact blend (fp16-exact on these values), requantised per (32-channel chunk, voxel) */
    float *hu = up_blend(hc, Cx, D, H, W), *hq = hnew((size_t)Cx * S), v[32], q[32];
    for (int b0 = 0; b0 < Cx; b0 += 32) for (size_t vox = 0; vox < S; vox++) {
        for (int k = 0; k < 32; k++) v[k] = hu[(size_t)(b0 + k) * S + vox];
        q_rn(v, 32, q, 4); for (int k = 0; k < 32; k++) hq[(size_t)(b0 + k) * S + vox] = q[k];
    }
    float *hid = wident(Ci), *wid = dput(hid, (size_t)Ci * Ci * 27), *sfd = dzero((size_t)Ci * S); free(hid);
    lp_conv_fwd_f4(c4, 4, fs, wid, nullptr, Ci, sfd, 0, NOGN, nullptr, 0, sp);
    float *sf = hget(sfd, (size_t)Ci * S); nn_free(sfd); nn_free(wid);
    { size_t nd = 0; for (size_t i = 0; i < (size_t)Cx * S; i++) nd += sf[i] != hq[i];
      check(!nd, "%s: staged up part == host 2x blend + e2m1 requantisation, bit-exact (%zu of %zu differ)", nm, nd, (size_t)Cx * S); }
    if (gn) check_staged_gn(nm, sf + (size_t)Cx * S, hk, &p2, Cs, S);
    else { size_t nd = 0; for (size_t i = 0; i < (size_t)Cs * S; i++) nd += sf[(size_t)Cx * S + i] != hk[i]; check(!nd, "%s: staged skip part == stored skip (%zu differ)", nm, nd); }
    float *hw = gen_w(Co, Ci), *hb = gen_b(Co), *w = dput(hw, (size_t)Co * Ci * 27), *b = dput(hb, Co), *yf = dzero((size_t)Co * S); void *y4 = dmx4(ys);
    lp_conv_fwd_f4(c4, 4, fs, w, b, Co, yf, 0, NOGN, nullptr, 0, sp);
    lp_conv_fwd_f4(c4, 4, fs, w, b, Co, y4, 4, NOGN, nullptr, 0, sp);
    const cref_t cr = {sf, Ci, D, H, W, hw, Co, hb, CV_S1, D, H, W};
    int cs[16]; const int ncs = mkcset(Co, 0, cs); size_t inex;
    rset_t r = rset_make(&cr, cs, ncs, 16, &inex);
    if (inex) check(0, "%s: exact-operand premise (%zu reference values not representable in fp32)", nm, inex);
    float *g = hget(yf, (size_t)Co * S); cmp_f32(nm, &r, g); free(g);
    g = hdeq4(y4, ys); cmp_mx(nm, &r, g, 0, Co, 0); free(g);
    gpu_ok(nm);
    rset_free(&r); if (gn) gn_free(&p2);
    free(hc); free(hk); free(hu); free(hq); free(sf); free(hw); free(hb); nn_free(c4); nn_free(k4); nn_free(w); nn_free(b); nn_free(yf); nn_free(y4);
}
/* ================= #9 decoder conv1 backward-data into the split (up | skip) MX-fp4 gradient, output SR ================= */
static void case_dec_bwd(int Cx, int Cs, int Cg, int D, int H, int W) {
    char nm[112]; snprintf(nm, sizeof nm, "dec c1 bwd_data %d -> up %d | skip %d, %dx%dx%d", Cg, Cx, Cs, D, H, W);
    const int Ci = Cx + Cs;
    const shape5 xs = {1, Ci, D, H, W}, xa = {1, Cx, D, H, W}, xb = {1, Cs, D, H, W}, ys = {1, Cg, D, H, W}; const size_t S = shape_spatial(xs);
    float *hg = gen_g((size_t)Cg * S, 1.f / 64), *hw = gen_w(Cg, Ci);
    void *g4 = dmx4_exact(hg, ys, nm), *ga = dmx4(xa), *gb = dmx4(xb), *gall = dmx4(xs);
    float *w = dput(hw, (size_t)Cg * Ci * 27), *scr = nn_malloc(nn_conv3d_scratch(xs, Cg, 3) + 4096);
    nn_set_sr_step(9001); const int rv = nn_conv3d_bwd_data_split(g4, ys, w, xs, ga, gb, Cx, scr);
    nn_set_sr_step(9001); nn_conv3d_bwd_data(g4, ys, w, xs, 3, 1, gall, scr);
    check(rv == 0, "%s: split backward-data supported", nm);
    float *ha = hdeq4(ga, xa), *hb = hdeq4(gb, xb), *hall = hdeq4(gall, xs);
    check(!memcmp(ha, hall, (size_t)Cx * S * 4) && !memcmp(hb, hall + (size_t)Cx * S, (size_t)Cs * S * 4),
          "%s: split output (up | skip) bit-identical to the unsplit %d-channel output (same SR seed)", nm, Ci);
    float *wT = wtrans(hw, Cg, Ci);
    const cref_t cr = {hg, Cg, D, H, W, wT, Ci, nullptr, CV_S1T, D, H, W};
    size_t inex; rset_t r = rset_make(&cr, nullptr, 0, 24, &inex);
    if (inex) check(0, "%s: exact-operand premise (%zu reference values not representable in fp32)", nm, inex);
    char n2[128];
    snprintf(n2, sizeof n2, "%s, up part", nm); cmp_mx(n2, &r, ha, 0, Cx, 1);
    snprintf(n2, sizeof n2, "%s, skip part", nm); cmp_mx(n2, &r, hb, Cx, Cs, 1);
    gpu_ok(nm);
    rset_free(&r); free(hg); free(hw); free(wT); free(ha); free(hb); free(hall);
    nn_free(g4); nn_free(ga); nn_free(gb); nn_free(gall); nn_free(w); nn_free(scr);
}
/* ================= #9 fp4 weight gradient with MX-fp4 gy (warp-specialised kernel), stored input, forced small gy slabs ================= */
static void case_wgrad(int Cx, int Cs, int Co, int D, int H, int W) {
    char nm[112]; snprintf(nm, sizeof nm, "wgrad fp4 gy %d%s%.0d -> %d, %dx%dx%d", Cx, Cs ? " + " : "", Cs, Co, D, H, W);
    const int Ci = Cx + Cs;
    const shape5 xs = {1, Ci, D, H, W}, xa = {1, Cx, D, H, W}, xb = {1, Cs, D, H, W}, ys = {1, Co, D, H, W}; const size_t S = shape_spatial(xs), nw = (size_t)Co * Ci * 27;
    float *hx = hnew((size_t)Ci * S);
    { float *a = gen_x(Cx, S, 0); memcpy(hx, a, (size_t)Cx * S * 4); free(a); if (Cs) { a = gen_x(Cs, S, 0); memcpy(hx + (size_t)Cx * S, a, (size_t)Cs * S * 4); free(a); } }
    float *hg = gen_g((size_t)Co * S, 1.f / 64);
    void *xa4 = dmx4_exact(hx, xa, nm), *xb4 = Cs ? dmx4_exact(hx + (size_t)Cx * S, xb, nm) : nullptr, *g4 = dmx4_exact(hg, ys, nm);
    split_t sp = {}; sp.sr = 0x2545f491u; if (Cs) { sp.x2 = xb4; sp.c_split = Cx; }
    float *gw1 = dzero(nw), *gb1 = dzero(Co), *gw2 = dzero(nw), *gb2 = dzero(Co);
    lp_set_f4w_gypre_kb(1); lp_bwd_w_f4(xa4, 4, xs, g4, 4, ys, gw1, gb1, NOGN, sp, 0); lp_set_f4w_gypre_kb(-1);   /* 1 KiB: one z-step a slab */
    lp_bwd_w_f4(xa4, 4, xs, g4, 4, ys, gw2, gb2, NOGN, sp, 0);
    float *h1 = hget(gw1, nw), *h2 = hget(gw2, nw), *b1 = hget(gb1, Co);
    check(!memcmp(h1, h2, nw * 4), "%s: one-z-step gy slabs == one slab, bitwise (exact sums)", nm);
    cmp_wg(nm, h1, b1, hx, Ci, D, H, W, hg, Co, D, H, W, 1, Cs ? Cx : 0, 0);
    gpu_ok(nm);
    free(hx); free(hg); free(h1); free(h2); free(b1);
    nn_free(xa4); if (xb4) nn_free(xb4); nn_free(g4); nn_free(gw1); nn_free(gb1); nn_free(gw2); nn_free(gb2);
}
/* ================= #9 the decoder conv1 weight gradient with the fused upsample and the skip's GN+SiLU (launch_bw4) ================= */
static void case_wgrad_up(int Cx, int Cs, int Co, int D, int H, int W) {
    char nm[112]; snprintf(nm, sizeof nm, "dec c1 wgrad fused up %d + skip %d (gn+silu) -> %d, %dx%dx%d", Cx, Cs, Co, D, H, W);
    const int Ci = Cx + Cs;
    const shape5 fs = {1, Ci, D, H, W}, cs_ = {1, Cx, D / 2, H / 2, W / 2}, ks = {1, Cs, D, H, W}, ys = {1, Co, D, H, W};
    const size_t S = shape_spatial(fs), Sc = shape_spatial(cs_), nw = (size_t)Co * Ci * 27;
    float *hc = gen_x(Cx, Sc, 0), *hk = gen_x(Cs, S, 1), *hg = gen_g((size_t)Co * S, 1.f / 64);
    gnpar_t p2 = gn_make(Cs, 8, 0);
    void *c4 = dmx4_exact(hc, cs_, nm), *k4 = dmx4_exact(hk, ks, nm), *g4 = dmx4_exact(hg, ys, nm);
    split_t sp = {}; sp.x2 = k4; sp.c_split = Cx; sp.up = 1; sp.gp2 = gn_gp(&p2); sp.sr = 0x2545f491u;
    float *gwu = dzero(nw), *gbu = dzero(Co), *gwm = dzero(nw), *gbm = dzero(Co);
    lp_set_f4w_gypre_kb(1); lp_bwd_w_f4(c4, 4, fs, g4, 4, ys, gwu, gbu, NOGN, sp, 0); lp_set_f4w_gypre_kb(-1);
    /* the same on the materialised fp32 upsample (+ the fp32 skip, GN+SiLU in staging, fp32 gy): the same values up to the bf16 tile
       rounding of the interpolated values in the fused decode (rare e2m1 flips: test_mx4 measured < 0.03 at 32 + 16) */
    float *hu = up_blend(hc, Cx, D, H, W), *up32 = dput(hu, (size_t)Cx * S), *k32 = dput(hk, (size_t)Cs * S), *g32 = dput(hg, (size_t)Co * S);
    split_t sm = {}; sm.x2 = k32; sm.c_split = Cx; sm.gp2 = gn_gp(&p2); sm.sr = 0x2545f491u;
    lp_bwd_w_f4(up32, 0, fs, g32, 0, ys, gwm, gbm, NOGN, sm, 0);
    float *hgu = hget(gwu, nw), *hgm = hget(gwm, nw), *hbu = hget(gbu, Co);
    const double e = rel2(hgu, hgm, nw);
    check(e < 0.03, "%s: vs the materialised fp32 upsample, rel L2 %.3g < 0.03", nm, e);
    /* host fp64 on the exact interpolated values and the exact GN+SiLU skip: the fp4 requantisation of x along space (round to nearest,
       ~0.1 relative per element) is the kernel's design error, not exact (test_mx4's 0.25) */
    float *hx = hnew((size_t)Ci * S); memcpy(hx, hu, (size_t)Cx * S * 4);
    for (int c = 0; c < Cs; c++) for (size_t v = 0; v < S; v++) hx[(size_t)(Cx + c) * S + v] = (float)gn_silu(&p2, c, hk[(size_t)c * S + v]);
    cmp_wg(nm, hgu, hbu, hx, Ci, D, H, W, hg, Co, D, H, W, 1, Cx, 0.25);
    gpu_ok(nm);
    gn_free(&p2); free(hc); free(hk); free(hg); free(hu); free(hx); free(hgu); free(hgm); free(hbu);
    nn_free(c4); nn_free(k4); nn_free(g4); nn_free(gwu); nn_free(gbu); nn_free(gwm); nn_free(gbm); nn_free(up32); nn_free(k32); nn_free(g32);
}
/* ================= #9 the weight gradient taking over a pending (deferred) GN+SiLU backward apply, slab by slab ================= */
static void case_wgrad_pend(int Cx, int Cs, int Co, int D, int H, int W) {   /* Cs > 0: the dec c1 input (fused up + GN skip); 0: stored x of Cx */
    char nm[128]; snprintf(nm, sizeof nm, "wgrad, pending GN+SiLU bwd apply: %s%d%s%.0d -> %d, %dx%dx%d", Cs ? "fused up " : "", Cx, Cs ? " + gn skip " : "", Cs, Co, D, H, W);
    const int Ci = Cx + Cs, G = 8;
    const shape5 fs = {1, Ci, D, H, W}, cs_ = {1, Cx, D / 2, H / 2, W / 2}, ks = {1, Cs, D, H, W}, xa = {1, Cx, D, H, W}, ys = {1, Co, D, H, W};
    const size_t S = shape_spatial(fs), nw = (size_t)Co * Ci * 27, n = (size_t)Co * S;
    float *h0 = Cs ? gen_x(Cx, shape_spatial(cs_), 0) : gen_x(Cx, S, 0), *hk = Cs ? gen_x(Cs, S, 1) : nullptr;
    void *x4 = dmx4_exact(h0, Cs ? cs_ : xa, nm), *k4 = Cs ? dmx4_exact(hk, ks, nm) : nullptr;
    gnpar_t p2 = {}; if (Cs) p2 = gn_make(Cs, G, 0);
    split_t sp = {}; sp.sr = 0x51ed2701u; if (Cs) { sp.x2 = k4; sp.c_split = Cx; sp.up = 1; sp.gp2 = gn_gp(&p2); }
    /* the conv output (pre-GN activation) and the incoming gradient of its GN+SiLU */
    float *ha = hnew(n), *ho = hnew(n); for (size_t i = 0; i < n; i++) { ha[i] = urange(-3.f, 3.f); ho[i] = urange(-1.f, 1.f) * 1e-3f; }
    void *a4 = dmx4_from(ha, ys), *o4 = dmx4_from(ho, ys), *gxa = dmx4(ys), *gxb = dmx4(ys);
    gnpar_t pa = gn_make(Co, G, 1);
    float *gg = dzero(Co), *gbn = dzero(Co), *scr = nn_malloc(nn_gn_scratch(ys) + 4096);
    float *gwa = dzero(nw), *gba = dzero(Co), *gwb = dzero(nw), *gbb = dzero(Co);
    nn_set_sr_step(1700); nn_gn_silu_bwd(a4, ys, G, pa.dg, pa.db, pa.dm, pa.dr, o4, gxa, gg, gbn, scr);   /* whole-layer gy records */
    lp_bwd_w_f4(x4, 4, fs, gxa, 4, ys, gwa, gba, NOGN, sp, 0);
    lp_gnpend_request(1); nn_set_sr_step(1700); nn_gn_silu_bwd(a4, ys, G, pa.dg, pa.db, pa.dm, pa.dr, o4, gxb, gg, gbn, scr); lp_gnpend_request(0);
    const int pend = lp_gnpend_of(gxb);
    lp_set_f4w_gypre_kb(1); lp_bwd_w_f4(x4, 4, fs, gxb, 4, ys, gwb, gbb, NOGN, sp, 0); lp_set_f4w_gypre_kb(-1);
    const int left = lp_gnpend_of(gxb);
    const size_t by = nn_mx4_bytes(ys); uint8_t *ba = xmalloc(by), *bb = xmalloc(by); nn_d2h(ba, gxa, by); nn_d2h(bb, gxb, by);
    float *wa = hget(gwa, nw), *wb = hget(gwb, nw), *ca = hget(gba, Co), *cb = hget(gbb, Co);
    const double dw = rel2(wb, wa, nw), db = rel2(cb, ca, Co);
    check(pend && !left, "%s: the apply left pending (%d) and taken over by the weight gradient (left %d)", nm, pend, left);
    check(!memcmp(ba, bb, by), "%s: gx written slab by slab == the whole-layer apply's bytes", nm);
    check(dw < 1e-6 && db < 1e-6, "%s: weight / bias gradient == from the whole-layer records (rel %.3g / %.3g < 1e-6: fp32 grouping only)", nm, dw, db);
    gpu_ok(nm);
    if (Cs) gn_free(&p2);
    gn_free(&pa); free(h0); free(hk); free(ha); free(ho); free(ba); free(bb); free(wa); free(wb); free(ca); free(cb);
    nn_free(x4); if (k4) nn_free(k4); nn_free(a4); nn_free(o4); nn_free(gxa); nn_free(gxb); nn_free(gg); nn_free(gbn); nn_free(scr);
    nn_free(gwa); nn_free(gba); nn_free(gwb); nn_free(gbb);
}
/* ================= #10 the down convs (stride 2, C -> C) of levels 2..4 ================= */
static void case_down(int C, int Di) {
    char nm[96]; snprintf(nm, sizeof nm, "down %d, %d^3 -> %d^3", C, Di, Di / 2);
    const int Do = Di / 2;
    const shape5 xs = {1, C, Di, Di, Di}, ys = {1, C, Do, Do, Do}; const size_t Sx = shape_spatial(xs), So = shape_spatial(ys), nw = (size_t)C * C * 27;
    float *hx = gen_x(C, Sx, 0), *hw = gen_w(C, C), *hb = gen_b(C);
    void *x4 = dmx4_exact(hx, xs, nm), *y4 = dmx4(ys); float *w = dput(hw, nw), *b = dput(hb, C);
    /* forward: launch_s2f8 (fp8 MMA, the MX-fp4 input requantised to e4m3 losslessly), MX-fp4 output */
    const int pf = lp_conv_fwd_s2_f8(x4, 4, xs, w, b, C, y4, 4, ys, NOGN);
    check(pf == 0, "%s: forward on the fp8 kernel (launch_s2f8; returned %d)", nm, pf);
    {   const cref_t cr = {hx, C, Di, Di, Di, hw, C, hb, CV_S2, Do, Do, Do};
        size_t inex; rset_t r = rset_make(&cr, nullptr, 0, 24, &inex);
        if (inex) check(0, "%s fwd: exact-operand premise (%zu reference values not representable in fp32)", nm, inex);
        float *g = hdeq4(y4, ys); char n2[112]; snprintf(n2, sizeof n2, "%s fwd", nm); cmp_mx(n2, &r, g, 0, C, 0); free(g); rset_free(&r); }
    /* backward-data, accumulating: gy (MX-fp4) -> mx8 recode, dilated by 2, the MX-fp8 stride-1 conv (launch_f8g) into an MX-fp8
       temp (round to nearest), added to the old gx (recoded to mx8: exact) with one more rounding, recoded to MX-fp4 with exact SR */
    float *hg = gen_g((size_t)C * So, 1.f / 64), *hold = gen_g((size_t)C * Sx, 0.25f);
    void *g4 = dmx4_exact(hg, ys, nm), *gx4 = dmx4_exact(hold, xs, nm);
    const int pb = lp_bwd_data_s2_mx(g4, ys, w, xs, gx4, 1, 4, 4, 0x6d2b79f5u);
    check(pb == 2, "%s: backward-data on the dilated fp8 tensor-core path (returned %d)", nm, pb);
    {   float *wT = wtrans(hw, C, C);
        const cref_t cr = {hg, C, Do, Do, Do, wT, C, nullptr, CV_S2T, Di, Di, Di};
        size_t inex; rset_t r = rset_make(&cr, nullptr, 0, 24, &inex);
        if (inex) check(0, "%s bwd_data: exact-operand premise (%zu reference values not representable in fp32)", nm, inex);
        float *g = hdeq4(gx4, xs); size_t nb = 0, nbad = 0; float t[32], t8[32], s[32], a8[32], gg[32];
        for (int j = 0; j < r.nv; j++) for (int b0 = 0; b0 < C; b0 += 32) {
            const size_t vox = r.vs[j];
            for (int k = 0; k < 32; k++) t[k] = (float)r.r[(size_t)(b0 + k) * Sx + vox];
            q_rn(t, 32, t8, 8);
            for (int k = 0; k < 32; k++) s[k] = hold[(size_t)(b0 + k) * Sx + vox] + t8[k];
            q_rn(s, 32, a8, 8);
            for (int k = 0; k < 32; k++) gg[k] = g[(size_t)(b0 + k) * Sx + vox];
            nb++; nbad += !sr_ok(a8, gg, 32);
        }
        check(nb && !nbad, "%s bwd_data (acc): gx = SR4(RN8(old + RN8(exact))): e2m1 neighbours (%zu blocks at %d voxels, %zu wrong)", nm, nb, r.nv, nbad);
        free(g); free(wT); rset_free(&r); }
    /* weight gradient: the FAST path (MX x, MX-fp4 gy), exact sums */
    float *gw = dzero(nw), *gb = dzero(C);
    lp_bwd_w_s2_f8(x4, 4, xs, g4, 4, ys, gw, gb, NOGN);
    {   float *hgw = hget(gw, nw), *hgb = hget(gb, C); char n2[112]; snprintf(n2, sizeof n2, "%s wgrad (FAST, fp4 gy)", nm);
        cmp_wg(n2, hgw, hgb, hx, C, Di, Di, Di, hg, C, Do, Do, Do, 2, 0, 0); free(hgw); free(hgb); }
    gpu_ok(nm);
    free(hx); free(hw); free(hb); free(hg); free(hold);
    nn_free(x4); nn_free(y4); nn_free(w); nn_free(b); nn_free(g4); nn_free(gx4); nn_free(gw); nn_free(gb);
}
static void case_down_gn(int C, int Di) {   /* the down conv with its input's GN+SiLU in staging (recompute mode): fp4 output, a tolerance */
    char nm[96]; snprintf(nm, sizeof nm, "down gn+silu %d, %d^3 -> %d^3", C, Di, Di / 2);
    const int Do = Di / 2;
    const shape5 xs = {1, C, Di, Di, Di}, ys = {1, C, Do, Do, Do}; const size_t Sx = shape_spatial(xs), nw = (size_t)C * C * 27;
    float *hx = gen_x(C, Sx, 1), *hw = gen_w(C, C), *hb = gen_b(C); gnpar_t p = gn_make(C, 8, 0);
    void *x4 = dmx4_exact(hx, xs, nm), *y4 = dmx4(ys); float *w = dput(hw, nw), *b = dput(hb, C);
    lp_conv_fwd_s2_f8(x4, 4, xs, w, b, C, y4, 4, ys, gn_gp(&p));
    float *ht = hnew((size_t)C * Sx);
    for (int c = 0; c < C; c++) for (size_t v = 0; v < Sx; v++) ht[(size_t)c * Sx + v] = (float)gn_silu(&p, c, hx[(size_t)c * Sx + v]);
    const cref_t cr = {ht, C, Di, Di, Di, hw, C, hb, CV_S2, Do, Do, Do};
    size_t inex; rset_t r = rset_make(&cr, nullptr, 0, 24, &inex);
    float *g = hdeq4(y4, ys); double d2 = 0, r2 = 0;
    for (int j = 0; j < r.nv; j++) for (int c = 0; c < C; c++) { const size_t i = (size_t)c * r.So + r.vs[j]; d2 += (g[i] - r.r[i]) * (g[i] - r.r[i]); r2 += r.r[i] * r.r[i]; }
    /* one e2m1 output rounding (~0.13 relative on such data: test_mx4's 0.16) + the e4m3 staging of the GN output (~0.02); a group or channel
       index error is O(1) */
    const double e = sqrt(d2 / r2);
    check(e < 0.2, "%s: MX-fp4 output vs host fp64, rel L2 %.3g < 0.2 (%d voxels x %d channels)", nm, e, r.nv, C);
    gpu_ok(nm);
    rset_free(&r); gn_free(&p); free(hx); free(hw); free(hb); free(ht); free(g); nn_free(x4); nn_free(y4); nn_free(w); nn_free(b);
}
/* ================= #12 the MX-fp4 GroupNorm + SiLU backward at try6 widths, G 8 ================= */
static void case_gnbwd(int C, int D, int H, int W) {
    char nm[96]; snprintf(nm, sizeof nm, "gn+silu bwd MX-fp4, %d ch (G 8: %d a group), %dx%dx%d", C, C / 8, D, H, W);
    const int G = 8, cpg = C / G, K = 8;
    const shape5 s = {1, C, D, H, W}; const size_t S = shape_spatial(s), n = (size_t)C * S;
    float *hx = hnew(n), *hy = hnew(n); for (size_t i = 0; i < n; i++) { hx[i] = urange(-3.f, 3.f); hy[i] = urange(-1.f, 1.f) * 1e-3f; }
    gnpar_t p = gn_make(C, G, 1);
    void *x4 = dmx4_from(hx, s), *g4 = dmx4_from(hy, s), *gx4 = dmx4(s);
    float *xd = hdeq4(x4, s), *gd = hdeq4(g4, s);
    float *gg = dzero(C), *gb = dzero(C), *scr = nn_malloc(nn_gn_scratch(s) + 4096);
    nn_set_sr_step(4100); nn_gn_silu_bwd(x4, s, G, p.dg, p.db, p.dm, p.dr, g4, gx4, gg, gb, scr);
    /* host fp64: xhat = (x - mean) rstd, y = gamma xhat + beta, g = gy dsilu(y); gx = rstd (gamma g - mean_grp(gamma g) - xhat
       mean_grp(gamma g xhat)), ggamma = sum g xhat, gbeta = sum g */
    double *hgx = xmalloc(n * 8), *sg = calloc(C, 8), *sgx = calloc(C, 8);
    for (int c = 0; c < C; c++) {
        const int q = c / cpg; const double rs_ = p.rstd[q], mu = p.mean[q];
        for (size_t v = 0; v < S; v++) {
            const size_t i = (size_t)c * S + v; const double xh = (xd[i] - mu) * rs_, y = p.gam[c] * xh + p.bet[c], sg_ = 1.0 / (1.0 + exp(-y));
            const double g = gd[i] * sg_ * (1.0 + y * (1.0 - sg_));
            hgx[i] = g; sg[c] += g; sgx[c] += g * xh;
        }
    }
    for (int q = 0; q < G; q++) {
        double A = 0, B = 0; for (int c = q * cpg; c < (q + 1) * cpg; c++) { A += p.gam[c] * sg[c]; B += p.gam[c] * sgx[c]; }
        A /= (double)cpg * S; B /= (double)cpg * S;
        for (int c = q * cpg; c < (q + 1) * cpg; c++) for (size_t v = 0; v < S; v++) {
            const size_t i = (size_t)c * S + v; const double xh = (xd[i] - p.mean[q]) * p.rstd[q];
            hgx[i] = p.rstd[q] * (p.gam[c] * hgx[i] - A - xh * B);
        }
    }
    float *hgg = hget(gg, C), *hgb = hget(gb, C), *rg = hnew(C), *rb = hnew(C), *hr = hnew(n);
    for (int c = 0; c < C; c++) { rg[c] = (float)sgx[c]; rb[c] = (float)sg[c]; }
    for (size_t i = 0; i < n; i++) hr[i] = (float)hgx[i];
    const double eg = rel2(hgg, rg, C), eb = rel2(hgb, rb, C);
    check(eg < 2e-3 && eb < 2e-3, "%s: ggamma / gbeta vs host fp64, rel %.3g / %.3g < 2e-3 (tanh.approx dSiLU)", nm, eg, eb);
    float *h1 = hdeq4(gx4, s), *acc = calloc(n, 4);
    const double one = rel2(h1, hr, n);
    size_t nb = 0, nbad = 0; float vf[32], gg32[32];   /* (the block scale or a code can move one step where the GPU's fp32 value sits on the other side of
                                                          a boundary than the exact one: ~1e-3 relative, a fraction of a percent of the blocks) */
    for (int b0 = 0; b0 < C; b0 += 32) for (size_t v = 0; v < S; v++) {
        for (int k = 0; k < 32; k++) { vf[k] = hr[(size_t)(b0 + k) * S + v]; gg32[k] = h1[(size_t)(b0 + k) * S + v]; }
        nb++; nbad += !sr_ok(vf, gg32, 32);
    }
    for (int k = 0; k < K; k++) {
        nn_zero(gg, C * 4); nn_zero(gb, C * 4); nn_set_sr_step(4200 + k); nn_gn_silu_bwd(x4, s, G, p.dg, p.db, p.dm, p.dr, g4, gx4, gg, gb, scr);
        float *hk = hdeq4(gx4, s); for (size_t i = 0; i < n; i++) acc[i] += hk[i] / K; free(hk);
    }
    const double avg = rel2(acc, hr, n);
    check(one < 0.3, "%s: gx (one SR draw) vs host fp64, rel L2 %.3g < 0.3", nm, one);
    check(nbad * 100 <= nb, "%s: gx blocks = SR neighbours of host fp64 at its block scale (%zu of %zu blocks off, <= 1%%)", nm, nbad, nb);
    check(avg < one * 2.0 / sqrt((double)K) + 0.02, "%s: mean of %d SR draws %.3g (unbiased: ~one / sqrt(%d) + the fp32 error)", nm, K, avg, K);
    gpu_ok(nm);
    gn_free(&p); free(hx); free(hy); free(xd); free(gd); free(hgx); free(sg); free(sgx); free(hgg); free(hgb); free(rg); free(rb); free(hr); free(h1); free(acc);
    nn_free(x4); nn_free(g4); nn_free(gx4); nn_free(gg); nn_free(gb); nn_free(scr);
}
/* ================= #13 the level-0 size (N 1, 256 channels, 192^3): MX-fp4 bytes generated on the host ================= */
static inline float h4at(const uint8_t *t, size_t nb, size_t S, int c, size_t v) {   /* element (c, v) of an N 1 MX-fp4 tensor of 32-channel blocks */
    const size_t ri = (size_t)(c >> 5) * S + v; const unsigned code = (t[ri * 16 + ((c & 31) >> 1)] >> (4 * (c & 1))) & 15u;
    return (code & 8 ? -1.f : 1.f) * G8[code & 7] * ldexpf(1.f, (int)t[nb * S * 16 + ri] - 127);
}
static void d_row4(const void *dev, size_t nb, size_t S, int blk, size_t v, float *o) {   /* one block row of a device MX-fp4 tensor, decoded */
    uint8_t row[16 + 1 + 15]; const size_t ri = (size_t)blk * S + v;
    nn_d2h(row, (const uint8_t *)dev + ri * 16, 16); nn_d2h(row + 16, (const uint8_t *)dev + nb * S * 16 + ri, 1);
    for (int k = 0; k < 32; k++) { const unsigned code = (row[k >> 1] >> (4 * (k & 1))) & 15u; o[k] = (code & 8 ? -1.f : 1.f) * G8[code & 7] * ldexpf(1.f, (int)row[16] - 127); }
}
static void case_layout(void) {   /* the host MX-fp4 byte layout used by case_l0 == lp_mx4_to_f32's */
    const shape5 s = {1, 64, 3, 4, 5}; const size_t S = shape_spatial(s), nb = 2, tot = nn_mx4_bytes(s);
    uint8_t *h = xmalloc(tot); for (size_t i = 0; i < nb * S * 16; i++) h[i] = (uint8_t)rnd(); for (size_t i = 0; i < nb * S; i++) h[nb * S * 16 + i] = (uint8_t)(120 + rnd() % 15);
    void *d = nn_malloc(tot); nn_h2d(d, h, tot); nn_set_storage(d, tot, 4);
    float *g = hdeq4(d, s); size_t nd = 0;
    for (int c = 0; c < 64; c++) for (size_t v = 0; v < S; v++) nd += g[(size_t)c * S + v] != h4at(h, nb, S, c, v);
    check(tot == nb * S * 17 && !nd, "host MX-fp4 layout ([blk][S] rows of 16 B, low nibble = even channel, then the [blk][S] scale bytes) == lp_mx4_to_f32 (%zu differ)", nd);
    free(h); free(g); nn_free(d);
}
static void case_l0(void) {
    const int C = 256, P = 192; const shape5 xs = {1, C, P, P, P};
    const size_t S = (size_t)P * P * P, nb = C / 32, rows = nb * S * 16, tot = rows + nb * S, nw = (size_t)C * C * 27;
    char nm[64]; snprintf(nm, sizeof nm, "level 0 %d -> %d, %d^3", C, C, P);
    printf("%s: %.3g elements a tensor (2^31 = %.3g), %.3f GB as MX-fp4 (nn_mx4_bytes %zu), x / y / gy / gx on the GPU: %.2f GB\n", nm, (double)C * S, 2147483648.0, tot / 1e9,
           nn_mx4_bytes(xs), 4.0 * tot / 1e9);
    check(tot == nn_mx4_bytes(xs), "%s: host byte layout size == nn_mx4_bytes", nm);
    uint8_t *hx = xmalloc(tot), *hg = xmalloc(tot);
    for (size_t i = 0; i < rows; i += 8) { const uint64_t a = rnd64(), b = rnd64(); memcpy(hx + i, &a, 8); memcpy(hg + i, &b, 8); }
    memset(hx + rows, 127, nb * S); memset(hg + rows, 121, nb * S);   /* x: grid values (scale 1); gy: grid x 2^-6 */
    float *hw = gen_w(C, C), *hb = gen_b(C);
    void *dx = nn_malloc(tot), *dg = nn_malloc(tot), *dy = dmx4(xs), *dgx = dmx4(xs);
    nn_h2d(dx, hx, tot); nn_set_storage(dx, tot, 4); nn_h2d(dg, hg, tot); nn_set_storage(dg, tot, 4);
    float *w = dput(hw, nw), *b = dput(hb, C), *scr = nn_malloc(nn_conv3d_scratch(xs, C, 3) + 4096);
    const split_t s0 = {};
    lp_conv_fwd_f4(dx, 4, xs, w, b, C, dy, 4, NOGN, nullptr, 0, s0);
    nn_set_sr_step(31337); nn_conv3d_bwd_data(dg, xs, w, xs, 3, 1, dgx, scr);
    gpu_ok(nm);
    /* sampled voxels: the last planes / rows / columns (the tensor's end: element indices up to 1.81e9, byte offsets up to 0.96e9) + random */
    size_t vs[48]; int nv = 0;
    const int zz[3] = {P - 1, P - 2, P - 9}, yy[3] = {P - 1, P - 8, P / 2}, xx[3] = {P - 1, P - 16, 5};
    for (int a = 0; a < 3; a++) for (int c = 0; c < 3; c++) for (int e = 0; e < 3; e++) vs[nv++] = ((size_t)zz[a] * P + yy[c]) * P + xx[e];
    vs[nv++] = 0; vs[nv++] = S - 1;
    while (nv < 40) vs[nv++] = rnd64() % S;
    const size_t K = (size_t)C * 27; float *p = hnew(K), *wT = wtrans(hw, C, C), v[32], q[32], g[32];
    size_t fb = 0, fbad = 0, bb = 0, bbad = 0, inex = 0; double *yv = xmalloc(C * 8);
    for (int j = 0; j < nv; j++) {
        const size_t vox = vs[j]; const int z = (int)(vox / ((size_t)P * P)), y = (int)(vox / P % P), x = (int)(vox % P);
        for (int mode = 0; mode < 2; mode++) {   /* 0: forward (x, w, bias); 1: backward-data (gy, w transposed) */
            const uint8_t *src = mode ? hg : hx;
            for (int t = 0; t < 27; t++) {
                const int kz = t / 9 - 1, ky = (t / 3) % 3 - 1, kx = t % 3 - 1, iz = mode ? z - kz : z + kz, iy = mode ? y - ky : y + ky, ix = mode ? x - kx : x + kx;
                const int ok = iz >= 0 && iz < P && iy >= 0 && iy < P && ix >= 0 && ix < P; const size_t off = ok ? ((size_t)iz * P + iy) * P + ix : 0;
                for (int c = 0; c < C; c++) p[(size_t)c * 27 + t] = ok ? h4at(src, nb, S, c, off) : 0.f;
            }
            for (int co = 0; co < C; co++) { yv[co] = (mode ? 0.0 : hb[co]) + dotw(p, (mode ? wT : hw) + (size_t)co * K, K); inex += (double)(float)yv[co] != yv[co]; }
            for (int blk = 0; blk < (int)nb; blk++) {
                for (int k = 0; k < 32; k++) v[k] = (float)yv[blk * 32 + k];
                d_row4(mode ? dgx : dy, nb, S, blk, vox, g);
                if (mode) { bb++; bbad += !sr_ok(v, g, 32); }
                else { q_rn(v, 32, q, 4); int ok = 1; for (int k = 0; k < 32; k++) ok &= q[k] == g[k]; fb++; fbad += !ok; }
            }
        }
    }
    check(!inex, "%s: exact-operand premise (%zu reference values not representable in fp32)", nm, inex);
    check(fb && !fbad, "%s fwd: MX-fp4 output == host quantiser of the exact values (%zu blocks at %d voxels near the end, %zu wrong)", nm, fb, nv, fbad);
    check(bb && !bbad, "%s bwd_data: MX-fp4 SR gx = e2m1 neighbours of the exact values (%zu blocks, %zu wrong)", nm, bb, bbad);
    /* weight gradient: gy kept only in planes 0 and P - 1 (73728 voxels: every sum exact in fp32), the rest zero; pre-pass slabs as the
       trainer's (~96 MB), so the last slab ends at the tensor's end */
    for (size_t blk = 0; blk < nb; blk++) memset(hg + (blk * S + (size_t)P * P) * 16, 0, (size_t)(P - 2) * P * P * 16);
    nn_h2d(dg, hg, rows);
    float *gw = dzero(nw), *gb = dzero(C);
    split_t sp = {}; sp.sr = 0x2545f491u;
    lp_bwd_w_f4(dx, 4, xs, dg, 4, xs, gw, gb, NOGN, sp, 0);
    gpu_ok(nm);
    float *hgw = hget(gw, nw), *hgb = hget(gb, C);
    const int pr[16][2] = {{255, 255}, {255, 254}, {254, 255}, {224, 255}, {255, 224}, {0, 0}, {0, 255}, {255, 0}, {31, 32}, {32, 31}, {128, 200}, {200, 128},
                           {(int)(rnd() % C), (int)(rnd() % C)}, {(int)(rnd() % C), (int)(rnd() % C)}, {(int)(rnd() % C), (int)(rnd() % C)}, {(int)(rnd() % C), (int)(rnd() % C)}};
    size_t nd = 0; inex = 0; const int pl[2] = {0, P - 1};
    for (int i = 0; i < 16; i++) {
        const int co = pr[i][0], ci = pr[i][1]; double o[27] = {};
        for (int a = 0; a < 2; a++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
            const size_t vo = ((size_t)pl[a] * P + y) * P + x; const float gv = h4at(hg, nb, S, co, vo); if (gv == 0.f) continue;
            for (int t = 0; t < 27; t++) {
                const int iz = pl[a] + t / 9 - 1, iy = y + (t / 3) % 3 - 1, ix = x + t % 3 - 1;
                if (iz < 0 || iz >= P || iy < 0 || iy >= P || ix < 0 || ix >= P) continue;
                o[t] += (double)gv * h4at(hx, nb, S, ci, ((size_t)iz * P + iy) * P + ix);
            }
        }
        for (int t = 0; t < 27; t++) { nd += (float)o[t] != hgw[((size_t)co * C + ci) * 27 + t]; inex += (double)(float)o[t] != o[t]; }
    }
    size_t nbd = 0;
    for (int co = 0; co < C; co++) { double s = 0; for (int a = 0; a < 2; a++) for (size_t v2 = 0; v2 < (size_t)P * P; v2++) s += h4at(hg, nb, S, co, (size_t)pl[a] * P * P + v2); nbd += (float)s != hgb[co]; }
    check(!nd && !inex, "%s wgrad: == exact host reference (16 (co, ci) pairs x 27 taps incl. channel 255, %zu differ%s)", nm, nd, inex ? ", reference inexact" : "");
    check(!nbd, "%s wgrad: bias gradient == exact sum of gy (%zu of %d differ)", nm, nbd, C);
    free(hx); free(hg); free(hw); free(hb); free(p); free(wT); free(yv); free(hgw); free(hgb);
    nn_free(dx); nn_free(dg); nn_free(dy); nn_free(dgx); nn_free(w); nn_free(b); nn_free(scr); nn_free(gw); nn_free(gb);
}

int main(void) {
    if (nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0)) { printf("no GPU\nFAIL\n"); return 1; }
    quick = getenv("UFSM_TEST_QUICK") && atoi(getenv("UFSM_TEST_QUICK"));
    nn_set_act_bf16(0); nn_set_grad_bf16(0); nn_set_tf32(1);
    case_layout();
    printf("#8 fp4 forward (launch_f4<4,2> at 256-448 outputs, <2,2> at 288), exact operands\n");
    case_fwd_plain(256, 256, 12, 10, 14);   /* MT 4, ragged W: the per-lane MX epilogue */
    case_fwd_plain(256, 320, 8, 8, 16);     /* W 16: the transposed MX epilogue (fwd_epi_mxt) */
    case_fwd_plain(448, 288, 12, 10, 14);   /* Cop 288: MT 2, TZ 2 */
    case_fwd_plain(288, 288, 12, 10, 14);
    case_fwd_plain(288, 288, 6, 6, 6);      /* level 5 */
    case_fwd_gn(256, 256, 12, 10, 14);      /* gn_prenorm_f4_k, 32 channels a group */
    case_fwd_gn(448, 288, 12, 12, 12);      /* 56 channels a group (groups straddle the 32-channel blocks), MT 2 */
    printf("#9 decoder conv1 with the fused upsample, backward-data with the split output, fp4 weight gradient\n");
    case_dec_fwd(320, 256, 256, 12, 10, 14, 0);
    case_dec_fwd(320, 256, 256, 12, 10, 14, 1);   /* dec0.c1 of try6 */
    case_dec_fwd(288, 448, 448, 8, 8, 16, 1);     /* dec3.c1 */
    case_dec_bwd(320, 256, 256, 12, 10, 14);      /* gx 576 rows: MT 4 */
    case_dec_bwd(288, 448, 448, 16, 8, 16);       /* gx 736 rows: MT 2, D 16: TZ 4; W 16: the transposed epilogue */
    case_wgrad(320, 256, 256, 12, 10, 14);        /* dec0.c1 shape, split input (WS kernel) */
    case_wgrad(448, 0, 288, 12, 12, 12);          /* enc4.c1 */
    case_wgrad(448, 0, 448, 24, 24, 24);          /* enc3.c2: more tiles / items */
    case_wgrad_up(320, 256, 256, 12, 10, 14);
    case_wgrad_up(288, 448, 448, 8, 8, 16);
    case_wgrad_pend(320, 256, 256, 12, 16, 16);
    case_wgrad_pend(448, 0, 288, 12, 12, 12);
    printf("#10 down convs of levels 2-4\n");
    case_down(384, 48);
    case_down(448, 24);
    case_down(288, 12);
    case_down_gn(448, 24);
    printf("#12 MX-fp4 GroupNorm + SiLU backward\n");
    case_gnbwd(256, 12, 16, 16);
    case_gnbwd(448, 8, 12, 16);
    if (quick) printf("#13 (UFSM_TEST_QUICK: the level-0 192^3 case runs in test_mx4_wide~full)\n");
    else { printf("#13 the level-0 size\n"); case_l0(); }
    printf(bad ? "test_mx4_wide: FAIL (%d)\n" : "test_mx4_wide: ok\n", bad);
    return bad != 0;
}
