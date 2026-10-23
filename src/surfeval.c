/* ufsm surf-eval: the upstream dinovol downstream test (villa dinovol-refactor dinovol/evals/kaggle_surfaces) in C/CUDA -- a frozen encoder
   (dinovol: its 480-wide patch tokens at 1/8 and the 32 / 64 / 128-channel CNN skips at 1, 1/2, 1/4) under the Vesuvius residual Decoder
   (per stage: 2x2x2 / stride-2 transposed conv of the level below, concat with the skip, 2 BasicBlockD (conv3 + InstanceNorm + LeakyReLU
   0.01, conv3 + InstanceNorm, 1x1 conv + IN projection when the width changes, add, LeakyReLU), 1x1 2-class head at full resolution;
   3.52 M parameters for the 480-wide model), trained with the unmodified DC_SkelREC_and_CE_loss (soft Dice + CE + skeleton recall,
   1 / 1 / 1, foreground only, smooth 1e-5, ignore label 2) on the Kaggle surface-detection volumes; AdamW (lr 1e-4, wd 1e-4), gradient
   norm clip 12. Medial-surface targets after the spatial augmentation: slice-wise (z) Zhang-Suen thinning of the surface label, two 3^3
   dilations, a 3^3 closing, restricted to the surface. Validation: fixed crops per validation case, foreground Dice at p >= 0.5 and
   medial recall over the valid voxels (upstream's crop metrics, not the leaderboard metric).

   Cases: a list file of lines "train|val IMAGE LABEL" (uint8 multi-page TIFF volumes, z pages; label 0 / 1 / 2). */
#include "dinovol.h"
#include "xnet.h"
#include "nn.h"
#include "attn.h"
#include "tiff.h"
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void se_lrelu(const float *x, float *y, long long n, float s);
void se_lrelu_bwd(const float *x, float *g, long long n, float s);
void se_tc_scatter(const float *tmp, const float *bias, float *y, int N, int Co, int d, int h, int w);
void se_tc_gather(const float *gy, float *tmp, int N, int Co, int d, int h, int w);
void se_tc_gbias(const float *tmp, float *gb, int N, int Co, long long V);
void se_h2f_cl(const void *src, float *dst, long long S, int C);
void se_loss_sums(const float *lg, const uint8_t *lab, const uint8_t *sk, int N, long long V, double *sums);
void se_loss_grad(const float *lg, const uint8_t *lab, const uint8_t *sk, int N, int Nmean, long long V, const double *sums, double cnt_all, float sm, float *g);
void se_val_counts(const float *lg, const uint8_t *lab, const uint8_t *sk, int N, long long V, unsigned long long *c);
void se_adamw(float *p, const float *g, float *m, float *v, long long n, float lr, float b1, float b2, float eps, float wd, int step, float gs);
void se_sumsq(const float *g, long long n, double *out);
void se_bias_add(float *y, const float *b, int N, int C, long long V);
void se_bias_grad(const float *g, float *gb, int N, int C, long long V);

static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}
static double se_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t se_rng(uint64_t *s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s; }
static double se_unif(uint64_t *s) { return (se_rng(s) >> 11) * (1.0 / 9007199254740992.0); }

/* ---------------- cases ---------------- */
typedef struct { char img[1024], lab[1024]; int split; int d, h, w; uint8_t *I, *L; double med, scale, p99; int64_t *fg; int nfg; int loaded; } se_case;
static int read_tiff_vol(const char *path, int *d, int *h, int *w, uint8_t **out) {
    tiff *t = tiff_open_file(path); if (!t) { fprintf(stderr, "surf-eval: %s: %s\n", path, tiff_error()); return -1; }
    const int np = tiff_npages(t); tiff_page pi; if (np < 1 || tiff_page_info(t, 0, &pi) || pi.bits != 8 || pi.spp != 1) { fprintf(stderr, "surf-eval: %s: need 8-bit single-channel pages\n", path); tiff_close(t); return -1; }
    *d = np; *h = pi.h; *w = pi.w; *out = malloc((size_t)np * pi.h * pi.w);
    for (int z = 0; z < np; z++) if (tiff_read_page(t, z, *out + (size_t)z * pi.h * pi.w)) { fprintf(stderr, "surf-eval: %s page %d: %s\n", path, z, tiff_error()); tiff_close(t); return -1; }
    tiff_close(t); return 0;
}
static pthread_mutex_t g_case_mu = PTHREAD_MUTEX_INITIALIZER;
static int case_load(se_case *c) {
    pthread_mutex_lock(&g_case_mu);
    if (c->loaded) { pthread_mutex_unlock(&g_case_mu); return 0; }
    int d2, h2, w2; uint8_t *L = nullptr;
    if (read_tiff_vol(c->img, &c->d, &c->h, &c->w, &c->I) || read_tiff_vol(c->lab, &d2, &h2, &w2, &L)) { pthread_mutex_unlock(&g_case_mu); return -1; }
    if (d2 != c->d || h2 != c->h || w2 != c->w) { fprintf(stderr, "surf-eval: %s: label shape differs\n", c->lab); pthread_mutex_unlock(&g_case_mu); return -1; }
    c->L = L;
    int64_t hist[256] = {0}; const size_t V = (size_t)c->d * c->h * c->w;
    for (size_t i = 0; i < V; i += 3) hist[c->I[i]]++;
    if (dv_hist_stats(hist, &c->med, &c->scale, &c->p99)) { c->med = 128; c->scale = 50; c->p99 = 255; }
    int cap = 4096; c->fg = malloc(cap * 8); c->nfg = 0; uint64_t rs = 99;
    size_t seen = 0;
    for (size_t i = 0; i < V; i++) if (L[i] == 1) { seen++; if (c->nfg < cap) c->fg[c->nfg++] = (int64_t)i; else { const size_t k = se_rng(&rs) % seen; if (k < (size_t)cap) c->fg[k] = (int64_t)i; } }
    c->loaded = 1; pthread_mutex_unlock(&g_case_mu); return 0;
}

/* ---------------- medial-surface target ---------------- */
static void thin2d(uint8_t *im, int h, int w, uint8_t *tmp) {   /* Zhang-Suen, im 0/1 in place */
    for (int changed = 1; changed;) {
        changed = 0;
        for (int pass = 0; pass < 2; pass++) {
            memset(tmp, 0, (size_t)h * w);
            for (int y = 1; y < h - 1; y++) for (int x = 1; x < w - 1; x++) {
                if (!im[y * w + x]) continue;
                const int p2 = im[(y - 1) * w + x], p3 = im[(y - 1) * w + x + 1], p4 = im[y * w + x + 1], p5 = im[(y + 1) * w + x + 1], p6 = im[(y + 1) * w + x], p7 = im[(y + 1) * w + x - 1], p8 = im[y * w + x - 1], p9 = im[(y - 1) * w + x - 1];
                const int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9; if (B < 2 || B > 6) continue;
                const int A = (!p2 && p3) + (!p3 && p4) + (!p4 && p5) + (!p5 && p6) + (!p6 && p7) + (!p7 && p8) + (!p8 && p9) + (!p9 && p2); if (A != 1) continue;
                if (pass == 0 ? (p2 && p4 && p6) || (p4 && p6 && p8) : (p2 && p4 && p8) || (p2 && p6 && p8)) continue;
                tmp[y * w + x] = 1;
            }
            for (int i = 0; i < h * w; i++) if (tmp[i]) { im[i] = 0; changed = 1; }
        }
    }
}
static void morph3(const uint8_t *in, uint8_t *out, int D, int H, int W, int dilate) {   /* 3^3 dilation (or erosion) */
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        int r = dilate ? 0 : 1;
        for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
            const int zz = z + dz, yy = y + dy, xx = x + dx;
            const int v = zz >= 0 && zz < D && yy >= 0 && yy < H && xx >= 0 && xx < W ? in[((size_t)zz * H + yy) * W + xx] : 0;
            if (dilate) r |= v; else r &= v;
        }
        out[((size_t)z * H + y) * W + x] = (uint8_t)r;
    }
}
static void medial_target(const uint8_t *L, int P, uint8_t *sk) {
    const size_t V = (size_t)P * P * P; uint8_t *a = malloc(V), *b = malloc(V), *tmp = malloc((size_t)P * P);
    for (size_t i = 0; i < V; i++) a[i] = L[i] == 1;
    for (int z = 0; z < P; z++) thin2d(a + (size_t)z * P * P, P, P, tmp);
    morph3(a, b, P, P, P, 1); morph3(b, a, P, P, P, 1);   /* two dilations */
    morph3(a, b, P, P, P, 1); morph3(b, a, P, P, P, 0);   /* closing */
    for (size_t i = 0; i < V; i++) sk[i] = a[i] && L[i] == 1;
    free(a); free(b); free(tmp);
}

/* crop P^3 of a case at origin o (zero image / ignore label outside), symmetry: perm of axes, flips */
static void crop_case(const se_case *c, const int o[3], int P, const int perm[3], const int flip[3], uint8_t *I, uint8_t *L) {
    const int dims[3] = {c->d, c->h, c->w};
    for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
        const int q[3] = {z, y, x}; int s[3];
        for (int k = 0; k < 3; k++) { int v = q[perm[k]]; if (flip[k]) v = P - 1 - v; s[k] = o[k] + v; }
        const size_t di = ((size_t)z * P + y) * P + x;
        if (s[0] < 0 || s[1] < 0 || s[2] < 0 || s[0] >= dims[0] || s[1] >= dims[1] || s[2] >= dims[2]) { I[di] = 0; L[di] = 2; continue; }
        const size_t si = ((size_t)s[0] * c->h + s[1]) * c->w + s[2]; I[di] = c->I[si]; L[di] = c->L[si];
    }
}

/* ---------------- decoder ---------------- */
typedef struct { size_t w, b; int ci, co, k; } se_conv;           /* offsets into the flat parameter array */
typedef struct { size_t g, b; int c; } se_in;
typedef struct { se_conv c1, c2, pj; se_in n1, n2, np; int ci, co, proj; } se_block;
typedef struct { size_t w, b; int ci, co; } se_tc;                /* w: [8][co][ci] */
typedef struct { se_tc up; se_block bl[2]; int cs; } se_stage;
typedef struct {
    int nst; se_stage st[4]; se_conv seg; size_t np; float *p, *g, *m, *v;
    int nl, ch[5]; int N, P;
} se_dec;
static size_t g_off;
static se_conv mk_conv(int ci, int co, int k, int bias) { se_conv c = {g_off, 0, ci, co, k}; g_off += (size_t)co * ci * k * k * k; if (bias) { c.b = g_off; g_off += co; } else c.b = (size_t)-1; return c; }
static se_in mk_in(int c) { se_in n = {g_off, g_off + c, c}; g_off += 2 * (size_t)c; return n; }
static se_block mk_block(int ci, int co) {
    se_block b; b.ci = ci; b.co = co; b.proj = ci != co;
    b.c1 = mk_conv(ci, co, 3, 1); b.n1 = mk_in(co); b.c2 = mk_conv(co, co, 3, 1); b.n2 = mk_in(co);
    if (b.proj) { b.pj = mk_conv(ci, co, 1, 0); b.np = mk_in(co); }
    return b;
}
static void dec_build(se_dec *D, int nl, const int *ch, int N, int P) {   /* ch[0..nl-1]: the encoder pyramid (full res first) */
    memset(D, 0, sizeof *D); D->nl = nl; memcpy(D->ch, ch, nl * sizeof(int)); D->N = N; D->P = P; g_off = 0;
    D->nst = nl - 1;
    for (int s = 0; s < D->nst; s++) {
        const int below = ch[nl - 1 - s], skip = ch[nl - 2 - s];
        se_stage *S = &D->st[s]; S->cs = skip;
        S->up.ci = below; S->up.co = skip; S->up.w = g_off; g_off += (size_t)8 * below * skip; S->up.b = g_off; g_off += skip;
        S->bl[0] = mk_block(2 * skip, skip); S->bl[1] = mk_block(skip, skip);
    }
    D->seg = mk_conv(ch[0], 2, 1, 1);
    D->np = g_off;
    /* torch defaults: conv / transposed conv weights and biases U(-1/sqrt(fan_in), 1/sqrt(fan_in)) (kaiming_uniform a = sqrt 5); IN 1 / 0 */
    float *h = calloc(D->np, 4); uint64_t rs = 20261023;
    #define UNI(off, n, bound) for (size_t i_ = 0; i_ < (size_t)(n); i_++) h[(off) + i_] = (float)((2 * se_unif(&rs) - 1) * (bound))
    for (int s = 0; s < D->nst; s++) {
        se_stage *S = &D->st[s];
        { const double fi = (double)S->up.co * 8; UNI(S->up.w, (size_t)8 * S->up.ci * S->up.co, 1 / sqrt(fi)); UNI(S->up.b, S->up.co, 1 / sqrt(fi)); }   /* (torch: fan_in of ConvTranspose = weight.size(1) * k^3) */
        for (int k = 0; k < 2; k++) {
            se_block *B = &S->bl[k];
            se_conv *cs[3] = {&B->c1, &B->c2, B->proj ? &B->pj : nullptr}; se_in *ns[3] = {&B->n1, &B->n2, B->proj ? &B->np : nullptr};
            for (int q = 0; q < 3; q++) {
                if (!cs[q]) continue;
                const double fi = (double)cs[q]->ci * cs[q]->k * cs[q]->k * cs[q]->k;
                UNI(cs[q]->w, (size_t)cs[q]->co * cs[q]->ci * cs[q]->k * cs[q]->k * cs[q]->k, 1 / sqrt(fi)); if (cs[q]->b != (size_t)-1) UNI(cs[q]->b, cs[q]->co, 1 / sqrt(fi));
                for (int c = 0; c < ns[q]->c; c++) h[ns[q]->g + c] = 1.f;
            }
        }
    }
    UNI(D->seg.w, 2 * ch[0], 1 / sqrt((double)ch[0])); UNI(D->seg.b, 2, 1 / sqrt((double)ch[0]));
    #undef UNI
    D->p = nn_malloc(D->np * 4); D->g = nn_malloc(D->np * 4); D->m = nn_malloc(D->np * 4); D->v = nn_malloc(D->np * 4);
    nn_h2d(D->p, h, D->np * 4); nn_zero(D->m, D->np * 4); nn_zero(D->v, D->np * 4); free(h);
}

/* activations of one block (allocated once for the fixed batch shape) */
typedef struct { float *x, *h1, *n1, *h2, *o, *y; float *m1, *r1, *m2, *r2, *mp, *rp; shape5 si, so; } se_bact;   /* (a1, n2 and the projection recomputed / folded) */
typedef struct { float *tmp, *up, *cat; se_bact b[2]; shape5 sl, su; } se_sact;
typedef struct { se_sact s[4]; float *lg, *glg; float *gA, *gB, *gC, *gD, *scr, *gnscr; size_t scr_n; } se_acts;
static float *nm(size_t n) { float *p = nn_malloc(n * 4); if (!p) { fprintf(stderr, "surf-eval: out of device memory\n"); exit(1); } return p; }
static void acts_alloc(se_acts *A, const se_dec *D) {
    memset(A, 0, sizeof *A); const int N = D->N; size_t maxe = 0, scr = 0;
    for (int s = 0; s < D->nst; s++) {
        const se_stage *S = &D->st[s]; const int lv = D->nl - 2 - s, e = D->P >> lv;   /* the stage's output level */
        se_sact *a = &A->s[s];
        a->sl = (shape5){N, S->up.ci, e / 2, e / 2, e / 2}; a->su = (shape5){N, S->cs, e, e, e};
        const size_t V = (size_t)e * e * e, Vl = V / 8;
        a->tmp = nm((size_t)N * 8 * S->cs * Vl); a->up = nm((size_t)N * S->cs * V); a->cat = nm((size_t)N * 2 * S->cs * V);
        for (int k = 0; k < 2; k++) {
            se_bact *b = &a->b[k]; const int ci = S->bl[k].ci, co = S->bl[k].co;
            b->si = (shape5){N, ci, e, e, e}; b->so = (shape5){N, co, e, e, e};
            b->h1 = nm((size_t)N * co * V); b->n1 = nm((size_t)N * co * V); b->h2 = nm((size_t)N * co * V);
            b->o = nm((size_t)N * co * V); b->y = nm((size_t)N * co * V);
            if (S->bl[k].proj) { b->mp = nm((size_t)N * co); b->rp = nm((size_t)N * co); }
            b->m1 = nm((size_t)N * co); b->r1 = nm((size_t)N * co); b->m2 = nm((size_t)N * co); b->r2 = nm((size_t)N * co);
            const size_t s1 = nn_conv3d_scratch(b->si, co, 3), s2 = nn_conv3d_scratch(b->so, co, 3), s3 = nn_conv3d_scratch(b->si, co, 1);
            scr = s1 > scr ? s1 : scr; scr = s2 > scr ? s2 : scr; scr = s3 > scr ? s3 : scr;
            const size_t gs = nn_gn_scratch(b->so); scr = gs > scr ? gs : scr;
        }
        const size_t e2 = (size_t)N * 2 * S->cs * V; maxe = e2 > maxe ? e2 : maxe;
        const size_t e3 = (size_t)N * 8 * S->cs * Vl; maxe = e3 > maxe ? e3 : maxe;
        const size_t e4 = (size_t)N * S->up.ci * Vl; maxe = e4 > maxe ? e4 : maxe;
    }
    const size_t V0 = (size_t)D->P * D->P * D->P;
    A->lg = nm((size_t)N * 2 * V0); A->glg = nm((size_t)N * 2 * V0);
    A->gA = nm(maxe); A->gB = nm(maxe); A->gC = nm(maxe); A->gD = nm(maxe);
    A->scr_n = scr; A->scr = nm(scr / 4 + 1); A->gnscr = nm(scr / 4 + 1);
}
/* 1x1 convs as GEMMs (the nn kernels do not cover a k = 1 weight gradient): y_n = W x_n (+ b) */
static void c1_fwd(const float *W, const float *b, const float *x, shape5 xs, int co, float *y) {
    const size_t V = (size_t)xs.d * xs.h * xs.w;
    for (int n = 0; n < xs.n; n++) at_gemm(0, 0, co, (int)V, xs.c, W, x + (size_t)n * xs.c * V, y + (size_t)n * co * V, 0.f);
    if (b) se_bias_add(y, b, xs.n, co, (long long)V);
}
static void c1_bwd(const float *W, float *gW, float *gb, const float *x, shape5 xs, int co, const float *gy, float *gx, int acc) {
    const size_t V = (size_t)xs.d * xs.h * xs.w;
    for (int n = 0; n < xs.n; n++) {
        at_gemm(0, 1, co, xs.c, (int)V, gy + (size_t)n * co * V, x + (size_t)n * xs.c * V, gW, 1.f);
        if (gx) at_gemm(1, 0, xs.c, (int)V, co, W, gy + (size_t)n * co * V, gx + (size_t)n * xs.c * V, acc ? 1.f : 0.f);
    }
    if (gb) se_bias_grad(gy, gb, xs.n, co, (long long)V);
}
/* the nn convs per sample (their batched paths assume one sample) */
static void cv_fwd(const float *x, shape5 xs, const float *w, const float *b, int co, float *y) {
    shape5 s1 = xs; s1.n = 1; const size_t xi = shape_numel(s1), yo = (size_t)co * xs.d * xs.h * xs.w;
    for (int n = 0; n < xs.n; n++) nn_conv3d_fwd(x + n * xi, s1, w, b, co, 3, 1, y + n * yo);
}
static void cv_bwd_w(const float *x, shape5 xs, const float *gy, shape5 ys, float *gw, float *gb) {
    shape5 a = xs, c = ys; a.n = c.n = 1; const size_t xi = shape_numel(a), yo = shape_numel(c);
    for (int n = 0; n < xs.n; n++) nn_conv3d_bwd_weight(x + n * xi, a, gy + n * yo, c, 3, 1, gw, gb);
}
static void cv_bwd_d(const float *gy, shape5 ys, const float *w, shape5 xs, float *gx, float *scr) {
    shape5 a = xs, c = ys; a.n = c.n = 1; const size_t xi = shape_numel(a), yo = shape_numel(c);
    for (int n = 0; n < xs.n; n++) nn_conv3d_bwd_data(gy + n * yo, c, w, a, 3, 1, gx + n * xi, scr);
}
static void in_fwd(const se_dec *D, const se_in *n, const float *x, shape5 s, float *y, float *m, float *r) { nn_gn_fwd(x, s, s.c, 1e-5f, D->p + n->g, D->p + n->b, y, m, r); }
static void block_fwd(const se_dec *D, const se_block *B, se_bact *a, const float *x) {
    const size_t no = shape_numel(a->so);
    a->x = (float *)x;
    cv_fwd(x, a->si, D->p + B->c1.w, D->p + B->c1.b, B->co, a->h1);
    in_fwd(D, &B->n1, a->h1, a->so, a->n1, a->m1, a->r1);
    se_lrelu(a->n1, a->y, (long long)no, 0.01f);   /* (y: a1 scratch) */
    cv_fwd(a->y, a->so, D->p + B->c2.w, D->p + B->c2.b, B->co, a->h2);
    in_fwd(D, &B->n2, a->h2, a->so, a->o, a->m2, a->r2);   /* o = IN(h2) */
    if (B->proj) { c1_fwd(D->p + B->pj.w, nullptr, x, a->si, B->co, a->y); in_fwd(D, &B->np, a->y, a->so, a->y, a->mp, a->rp); nn_axpy(a->o, 1.f, a->y, no); }
    else nn_axpy(a->o, 1.f, x, no);   /* o += residual */
    se_lrelu(a->o, a->y, (long long)no, 0.01f);
}
/* gy (the block output's gradient, modified in place) -> gx (SET); scratch tensors g1, g2 of the output size */
static void block_bwd(se_dec *D, const se_block *B, se_bact *a, float *gy, float *gx, float *g1, float *g2, se_acts *A) {
    const size_t no = shape_numel(a->so);
    se_lrelu_bwd(a->o, gy, (long long)no, 0.01f);   /* gy: d / d o (= d / d n2 = d / d residual) */
    nn_gn_bwd(a->h2, a->so, a->so.c, D->p + B->n2.g, a->m2, a->r2, gy, g1, D->g + B->n2.g, D->g + B->n2.b, A->gnscr);   /* g1 = d / d h2 */
    se_lrelu(a->n1, g2, (long long)no, 0.01f);   /* a1 again */
    cv_bwd_w(g2, a->so, g1, a->so, D->g + B->c2.w, D->g + B->c2.b);
    cv_bwd_d(g1, a->so, D->p + B->c2.w, a->so, g2, A->scr);   /* g2 = d / d a1 */
    se_lrelu_bwd(a->n1, g2, (long long)no, 0.01f);
    nn_gn_bwd(a->h1, a->so, a->so.c, D->p + B->n1.g, a->m1, a->r1, g2, g1, D->g + B->n1.g, D->g + B->n1.b, A->gnscr);   /* g1 = d / d h1 */
    cv_bwd_w(a->x, a->si, g1, a->so, D->g + B->c1.w, D->g + B->c1.b);
    cv_bwd_d(g1, a->so, D->p + B->c1.w, a->si, gx, A->scr);
    if (B->proj) {
        c1_fwd(D->p + B->pj.w, nullptr, a->x, a->si, B->co, g2);   /* (the projection conv output again: the IN backward's input) */
        nn_gn_bwd(g2, a->so, a->so.c, D->p + B->np.g, a->mp, a->rp, gy, g1, D->g + B->np.g, D->g + B->np.b, A->gnscr);
        c1_bwd(D->p + B->pj.w, D->g + B->pj.w, nullptr, a->x, a->si, B->co, g1, gx, 1);
    } else nn_axpy(gx, 1.f, gy, no);
}
static void tc_fwd(const se_dec *D, const se_tc *T, const float *x, shape5 xs, float *tmp, float *y) {
    const size_t V = (size_t)xs.d * xs.h * xs.w;
    for (int n = 0; n < xs.n; n++) for (int t = 0; t < 8; t++)
        at_gemm(0, 0, T->co, (int)V, T->ci, D->p + T->w + (size_t)t * T->co * T->ci, x + (size_t)n * T->ci * V, tmp + ((size_t)n * 8 + t) * T->co * V, 0.f);
    se_tc_scatter(tmp, D->p + T->b, y, xs.n, T->co, xs.d, xs.h, xs.w);
}
static void tc_bwd(se_dec *D, const se_tc *T, const float *x, shape5 xs, const float *gy, float *tmp, float *gx) {
    const size_t V = (size_t)xs.d * xs.h * xs.w;
    se_tc_gather(gy, tmp, xs.n, T->co, xs.d, xs.h, xs.w);
    se_tc_gbias(tmp, D->g + T->b, xs.n, T->co, (long long)V);
    for (int n = 0; n < xs.n; n++) for (int t = 0; t < 8; t++) {
        const float *gt = tmp + ((size_t)n * 8 + t) * T->co * V;
        at_gemm(0, 1, T->co, T->ci, (int)V, gt, x + (size_t)n * T->ci * V, D->g + T->w + (size_t)t * T->co * T->ci, 1.f);
        if (gx) at_gemm(1, 0, T->ci, (int)V, T->co, D->p + T->w + (size_t)t * T->co * T->ci, gt, gx + (size_t)n * T->ci * V, t ? 1.f : 0.f);
    }
}
/* features: f[k] [N][ch[k]][P >> k]^3 (k = 0 full resolution .. nl - 1 bottleneck) -> logits A->lg [N][2][P^3] */
static void dec_fwd(const se_dec *D, se_acts *A, float *const *f) {
    const float *below = f[D->nl - 1];
    for (int s = 0; s < D->nst; s++) {
        const se_stage *S = &D->st[s]; se_sact *a = &A->s[s]; const int lv = D->nl - 2 - s;
        tc_fwd(D, &S->up, below, a->sl, a->tmp, a->up);
        nn_concat_fwd(a->up, S->cs, f[lv], S->cs, a->su, a->cat);
        block_fwd(D, &S->bl[0], &a->b[0], a->cat);
        block_fwd(D, &S->bl[1], &a->b[1], a->b[0].y);
        below = a->b[1].y;
    }
    const se_sact *a = &A->s[D->nst - 1];
    c1_fwd(D->p + D->seg.w, D->p + D->seg.b, a->b[1].y, a->b[1].so, 2, A->lg);
}
static void dec_bwd(se_dec *D, se_acts *A, float *const *f) {
    se_sact *last = &A->s[D->nst - 1]; const shape5 s0 = last->b[1].so;
    float *gcur = A->gA; c1_bwd(D->p + D->seg.w, D->g + D->seg.w, D->g + D->seg.b, last->b[1].y, s0, 2, A->glg, gcur, 0);
    for (int s = D->nst - 1; s >= 0; s--) {
        const se_stage *S = &D->st[s]; se_sact *a = &A->s[s];
        block_bwd(D, &S->bl[1], &a->b[1], gcur, A->gB, A->gC, A->gD, A);   /* gB = d / d b[0].y */
        block_bwd(D, &S->bl[0], &a->b[0], A->gB, A->gA, A->gC, A->gD, A);   /* gA = d / d cat */
        nn_concat_bwd(A->gA, S->cs, S->cs, a->su, A->gB, A->gC);   /* gB = d / d up (gC: the skip's, frozen) */
        float *gl = s > 0 ? A->gD : nullptr;
        tc_bwd(D, &S->up, s > 0 ? A->s[s - 1].b[1].y : f[D->nl - 1], a->sl, A->gB, a->tmp, gl);
        if (gl) { nn_d2d_async(A->gA, gl, shape_numel(a->sl) * 4); gcur = A->gA; }
    }
}

/* ---------------- encoders ---------------- */
typedef struct { dv_net *dv; int nl, ch[5]; } se_enc;
/* one sample's u8 crop (device) -> features of sample n of the batch tensors f[k] */
static int enc_fwd(se_enc *E, const uint8_t *u8d, int P, const se_case *c, float *const *f, int n) {
    nn_sync();   /* (dinovol runs on xnet's non-blocking stream) */
    if (dv_forward(E->dv, u8d, P, (float)c->med, (float)c->scale, (float)c->p99)) return -1;
    xnet_sync();
    for (int k = 0; k < 3; k++) { int e = 0, ch = 0; const void *s = dv_skip(E->dv, k, &e, &ch); se_h2f_cl(s, f[k] + (size_t)n * ch * e * e * e, (long long)e * e * e, ch); }
    const int g = P / 8; se_h2f_cl(dv_tokens(E->dv), f[3] + (size_t)n * E->ch[3] * g * g * g, (long long)g * g * g, E->ch[3]);
    return 0;
}

/* the loss on the current logits (host value, the same formula as the kernels) */
static double se_loss_value(se_acts *A, const uint8_t *dL, const uint8_t *dS, int N, size_t V, double *dsum) {
    se_loss_sums(A->lg, dL, dS, N, (long long)V, dsum); double hs[64]; nn_d2h(hs, dsum, (size_t)N * 64);
    double cnt = 0, ce = 0, dc = 0, rc = 0;
    for (int n = 0; n < N; n++) { cnt += hs[n * 8 + 1]; ce += hs[n * 8]; dc += (2 * hs[n * 8 + 2] + 1e-5) / (hs[n * 8 + 3] + hs[n * 8 + 4] + 1e-5) / N; rc += (hs[n * 8 + 5] + 1e-5) / (hs[n * 8 + 6] + 1e-5) / N; }
    return (cnt > 0 ? ce / cnt : 0) - dc - rc;
}
/* --gradcheck: a small decoder (pyramid 16 / 32 / 32 / 48 channels, 16^3, batch 2) on random features: analytic vs central differences */
static int se_gradcheck(void) {
    const int ch[4] = {16, 32, 32, 48}, N = 1, P = 16; se_dec D; dec_build(&D, 4, ch, N, P); se_acts A; acts_alloc(&A, &D);
    const size_t V = (size_t)P * P * P; uint64_t rs = 5; float *f[4];
    for (int k = 0; k < 4; k++) { const int e = P >> k; const size_t n = (size_t)N * ch[k] * e * e * e; float *h = malloc(n * 4); for (size_t i = 0; i < n; i++) h[i] = (float)(2 * se_unif(&rs) - 1); f[k] = nm(n); nn_h2d(f[k], h, n * 4); free(h); }
    uint8_t *hL = malloc(N * V), *hS = malloc(N * V); for (size_t i = 0; i < N * V; i++) { const double u = se_unif(&rs); hL[i] = u < 0.1 ? 2 : u < 0.4 ? 1 : 0; hS[i] = hL[i] == 1 && se_unif(&rs) < 0.5; }
    uint8_t *dL = nn_malloc(N * V), *dS = nn_malloc(N * V); nn_h2d(dL, hL, N * V); nn_h2d(dS, hS, N * V); double *dsum = nn_malloc((size_t)N * 64);
    dec_fwd(&D, &A, f); const double L0 = se_loss_value(&A, dL, dS, N, V, dsum);
    if (!isfinite(L0)) {   /* where does it go non-finite? */
        #define CHK(nm_, p_, n_) do { const size_t n__ = (n_); float *h__ = malloc(n__ * 4); nn_d2h(h__, (p_), n__ * 4); size_t b__ = 0; double s__ = 0; for (size_t i__ = 0; i__ < n__; i__++) { if (!isfinite(h__[i__])) b__++; else s__ += fabs(h__[i__]); } fprintf(stderr, "  %-12s %zu non-finite of %zu, mean |x| %.3e\n", nm_, b__, n__, s__ / n__); free(h__); } while (0)
        for (int s = 0; s < D.nst; s++) { se_sact *a = &A.s[s]; char t[32];
            snprintf(t, sizeof t, "s%d up", s); CHK(t, a->up, shape_numel(a->su)); snprintf(t, sizeof t, "s%d cat", s); CHK(t, a->cat, 2 * shape_numel(a->su));
            for (int k = 0; k < 2; k++) { snprintf(t, sizeof t, "s%d b%d h1", s, k); CHK(t, a->b[k].h1, shape_numel(a->b[k].so)); snprintf(t, sizeof t, "s%d b%d n1", s, k); CHK(t, a->b[k].n1, shape_numel(a->b[k].so));
                snprintf(t, sizeof t, "s%d b%d h2", s, k); CHK(t, a->b[k].h2, shape_numel(a->b[k].so)); snprintf(t, sizeof t, "s%d b%d y", s, k); CHK(t, a->b[k].y, shape_numel(a->b[k].so)); } }
        CHK("logits", A.lg, (size_t)N * 2 * V);
        #undef CHK
    }
    double hs[64]; nn_d2h(hs, dsum, (size_t)N * 64); double cnt = 0; for (int n = 0; n < N; n++) cnt += hs[n * 8 + 1];
    se_loss_grad(A.lg, dL, dS, N, N, (long long)V, dsum, cnt, 1e-5f, A.glg); nn_zero(D.g, D.np * 4); dec_bwd(&D, &A, f);
    float *g = malloc(D.np * 4), *p = malloc(D.np * 4); nn_d2h(g, D.g, D.np * 4); nn_d2h(p, D.p, D.np * 4);
    int bad = 0; double maxrel = 0;
    for (int t = 0; t < 60; t++) {
        const size_t i = se_rng(&rs) % D.np; const float p0 = p[i], e = 1e-2f * (fabsf(p0) > 0.1f ? fabsf(p0) : 0.1f);
        float v = p0 + e; nn_h2d(D.p + i, &v, 4); dec_fwd(&D, &A, f); const double lp = se_loss_value(&A, dL, dS, N, V, dsum);
        v = p0 - e; nn_h2d(D.p + i, &v, 4); dec_fwd(&D, &A, f); const double lm = se_loss_value(&A, dL, dS, N, V, dsum);
        nn_h2d(D.p + i, &p0, 4);
        const double fd = (lp - lm) / (2 * e), rel = fabs(fd - g[i]) / (fabs(fd) + fabs(g[i]) + 1e-4);
        if (rel > maxrel) maxrel = rel;
        if (rel > 0.05) { bad++; fprintf(stderr, "  param %zu: analytic %.6e fd %.6e\n", i, g[i], fd); }
    }
    fprintf(stderr, "surf-eval gradcheck: loss %.5f, %zu params, 60 samples, max rel err %.3e, %d bad\n", L0, D.np, maxrel, bad);
    return bad > 3;
}
int cmd_surf_eval(int argc, char **argv) {
    if (argc > 2 && !strcmp(argv[2], "--gradcheck")) { if (nn_init(atoi(opt(argc, argv, "--gpu", "0")))) return 1; if (getenv("SE_TF32")) nn_set_tf32(1); return se_gradcheck(); }
    const char *list = opt(argc, argv, "--cases", nullptr), *ck = opt(argc, argv, "--ckpt", nullptr), *out = opt(argc, argv, "--out", nullptr);
    if (!list || !ck || !out) {
        fprintf(stderr, "usage: ufsm surf-eval --cases LIST --ckpt dinovol.pt --out RUN_DIR [--P 128] [--batch 2] [--steps 5000] [--val-every 500] [--val-crops 2]\n"
                        "         [--val-cases 0 (all)] [--lr 1e-4] [--wd 1e-4] [--clip 12] [--seed 42] [--gpu 0] [--gpu-mem 6]\n"
                        "  the upstream kaggle_surfaces frozen-encoder decoder evaluation (see src/surfeval.c); LIST lines: train|val IMAGE.tif LABEL.tif\n");
        return 2;
    }
    const int P = atoi(opt(argc, argv, "--P", "128")), N = 1, NB = atoi(opt(argc, argv, "--batch", "2")), steps = atoi(opt(argc, argv, "--steps", "5000"));
    const int vev = atoi(opt(argc, argv, "--val-every", "500")), vcr = atoi(opt(argc, argv, "--val-crops", "2")), vlim = atoi(opt(argc, argv, "--val-cases", "0"));
    const float lr = (float)atof(opt(argc, argv, "--lr", "1e-4")), wd = (float)atof(opt(argc, argv, "--wd", "1e-4")), clip = (float)atof(opt(argc, argv, "--clip", "12"));
    uint64_t rs = (uint64_t)atoll(opt(argc, argv, "--seed", "42")) * 0x9E3779B97F4A7C15ull + 7;
    const int gpu = atoi(opt(argc, argv, "--gpu", "0"));
    if (P % 16 || NB < 1) { fprintf(stderr, "surf-eval: P a multiple of 16\n"); return 2; }
    if (nn_init(gpu) || xnet_init(gpu)) return 1;
    if (getenv("SE_TF32")) nn_set_tf32(1);   /* (default: the plain fp32 conv kernels; the TF32 path reads 16-bit activations) */
    xnet_set_budget((size_t)(atof(opt(argc, argv, "--gpu-mem", "6")) * 1e9));
    /* cases */
    FILE *lf = fopen(list, "r"); if (!lf) { fprintf(stderr, "surf-eval: cannot read %s\n", list); return 1; }
    int nc = 0, cap = 1024; se_case *C = calloc(cap, sizeof *C); char line[4200];
    while (fgets(line, sizeof line, lf)) {
        char sp[16]; se_case c = {0};
        if (sscanf(line, "%15s %1023s %1023s", sp, c.img, c.lab) != 3) continue;
        c.split = !strcmp(sp, "val");
        if (nc == cap) { cap *= 2; C = realloc(C, cap * sizeof *C); }
        C[nc++] = c;
    }
    fclose(lf);
    int ntr = 0, nva = 0; for (int i = 0; i < nc; i++) { if (C[i].split) nva++; else ntr++; }
    int *tr = malloc(ntr * sizeof *tr), *va = malloc(nva * sizeof *va); ntr = nva = 0;
    for (int i = 0; i < nc; i++) { if (C[i].split) va[nva++] = i; else tr[ntr++] = i; }
    if (!ntr || !nva) { fprintf(stderr, "surf-eval: need train and val cases\n"); return 1; }
    if (vlim > 0 && vlim < nva) nva = vlim;
    /* encoder + decoder */
    se_enc E = {0}; if (!(E.dv = dv_load(ck))) return 1;
    E.nl = 4; E.ch[3] = dv_embed_dim(E.dv);
    {   /* the skip widths from a dry forward */
        uint8_t *z = xmalloc_dev((size_t)P * P * P); { uint8_t *hz = calloc((size_t)P * P * P, 1); xh2d(z, hz, (size_t)P * P * P); free(hz); }
        if (dv_forward(E.dv, z, P, 128, 50, 255)) return 1;
        xnet_sync();
        for (int k = 0; k < 3; k++) { int e, ch; dv_skip(E.dv, k, &e, &ch); E.ch[k] = ch; if (e != P >> k) { fprintf(stderr, "surf-eval: skip %d edge %d, expected %d\n", k, e, P >> k); return 1; } }
        xfree_dev(z);
    }
    se_dec D; dec_build(&D, E.nl, E.ch, N, P);
    se_acts A; acts_alloc(&A, &D);
    float *f[5]; for (int k = 0; k < E.nl; k++) { const int e = P >> k; f[k] = nm((size_t)N * E.ch[k] * e * e * e); }
    const size_t V = (size_t)P * P * P;
    uint8_t *hI = malloc(V), *hL = malloc((size_t)N * V), *hS = malloc((size_t)N * V), *u8d = xmalloc_dev(V), *dL = nn_malloc((size_t)N * V), *dS = nn_malloc((size_t)N * V);
    double *dsum = nn_malloc((size_t)N * 8 * 8), *dsq = nn_malloc(8); unsigned long long *dvc = nn_malloc(5 * 8);
    fprintf(stderr, "surf-eval: %d train / %d val cases; encoder %s (widths %d %d %d | tokens %d), decoder %zu parameters; %d^3 crops, batch %d (accumulated), %d steps\n",
            ntr, nva, ck, E.ch[0], E.ch[1], E.ch[2], E.ch[3], D.np, P, NB, steps);
    { char cmd[1100]; snprintf(cmd, sizeof cmd, "mkdir -p '%s'", out); if (system(cmd)) {} }
    char mp[1200]; snprintf(mp, sizeof mp, "%s/metrics.tsv", out); FILE *mf = fopen(mp, "a");
    if (mf) fprintf(mf, "# step\tkind\tloss\tforeground_dice\tmedial_recall\n");
    double best = -1, t0 = se_now(), lsum = 0; int lcnt = 0;
    for (int step = 1; step <= steps; step++) {
        /* the batch: NB crops, one at a time (the nn convs take one sample), gradients accumulated */
        nn_zero(D.g, D.np * 4);
        double sl = 0;
        for (int nb = 0; nb < NB; nb++) {
            se_case *c = nullptr; int o[3];
            for (int tries = 0; ; tries++) {
                c = &C[tr[se_rng(&rs) % (uint64_t)ntr]]; if (case_load(c)) return 1;
                const int dims[3] = {c->d, c->h, c->w};
                if (se_unif(&rs) < 0.5 && c->nfg) { const int64_t v = c->fg[se_rng(&rs) % (uint64_t)c->nfg]; const int q[3] = {(int)(v / ((int64_t)c->h * c->w)), (int)(v / c->w % c->h), (int)(v % c->w)};
                    for (int k = 0; k < 3; k++) { o[k] = q[k] - P / 2; if (o[k] > dims[k] - P) o[k] = dims[k] - P; if (o[k] < 0) o[k] = dims[k] >= P ? 0 : (dims[k] - P) / 2; } }
                else for (int k = 0; k < 3; k++) o[k] = dims[k] >= P ? (int)(se_unif(&rs) * (dims[k] - P + 1)) : (dims[k] - P) / 2;
                int perm[3] = {0, 1, 2}, flip[3]; for (int k = 2; k > 0; k--) { const int j = (int)(se_rng(&rs) % (uint64_t)(k + 1)), t = perm[k]; perm[k] = perm[j]; perm[j] = t; }
                for (int k = 0; k < 3; k++) flip[k] = (int)(se_rng(&rs) & 1);
                crop_case(c, o, P, perm, flip, hI, hL);
                size_t valid = 0; for (size_t i = 0; i < V; i += 7) valid += hL[i] != 2;
                if (valid * 7 * 100 >= V || tries > 32) break;
            }
            medial_target(hL, P, hS);
            xh2d(u8d, hI, V);
            if (enc_fwd(&E, u8d, P, c, f, 0)) return 1;
            nn_h2d(dL, hL, V); nn_h2d(dS, hS, V);
            dec_fwd(&D, &A, f);
            se_loss_sums(A.lg, dL, dS, 1, (long long)V, dsum);
            double hs[8]; nn_d2h(hs, dsum, 64);
            const double cnt = hs[1], ce = cnt > 0 ? hs[0] / cnt : 0, dc = (2 * hs[2] + 1e-5) / (hs[3] + hs[4] + 1e-5), rc = (hs[5] + 1e-5) / (hs[6] + 1e-5);
            sl += (ce - dc - rc) / NB;
            se_loss_grad(A.lg, dL, dS, 1, NB, (long long)V, dsum, cnt > 0 ? cnt * NB : 1, 1e-5f, A.glg);   /* (CE: each crop's mean / NB) */
            dec_bwd(&D, &A, f);
        }
        lsum += sl; lcnt++;
        se_sumsq(D.g, (long long)D.np, dsq); double gn2; nn_d2h(&gn2, dsq, 8);
        const double gn = sqrt(gn2); const float gs = gn > clip ? (float)(clip / (gn + 1e-6)) : 1.f;
        se_adamw(D.p, D.g, D.m, D.v, (long long)D.np, lr, 0.9f, 0.999f, 1e-8f, wd, step, gs);
        if (step % 50 == 0 || step == 1) {
            fprintf(stderr, "step %d loss %.4f |g| %.3f %.2f s/step\n", step, lsum / lcnt, gn, (se_now() - t0) / step);
            if (mf) { fprintf(mf, "%d\ttrain\t%.5f\t\t\n", step, lsum / lcnt); fflush(mf); }
            lsum = 0; lcnt = 0;
        }
        if (step % vev == 0 || step == steps) {   /* validation: fixed crops per case (no augmentation), batch of 1 per crop */
            unsigned long long tot[5] = {0}; double vl = 0; int vn = 0;
            for (int ci = 0; ci < nva; ci++) {
                se_case *c = &C[va[ci]]; if (case_load(c)) return 1;
                const int dims[3] = {c->d, c->h, c->w};
                for (int k = 0; k < vcr; k++) {
                    int o[3]; uint64_t vr = (uint64_t)(va[ci] + 1) * 1000003ull + (uint64_t)k * 7919 + 1;
                    if (c->nfg) { const int64_t v = c->fg[se_rng(&vr) % (uint64_t)c->nfg]; const int q[3] = {(int)(v / ((int64_t)c->h * c->w)), (int)(v / c->w % c->h), (int)(v % c->w)};
                        for (int a = 0; a < 3; a++) { o[a] = q[a] - P / 2; if (o[a] > dims[a] - P) o[a] = dims[a] - P; if (o[a] < 0) o[a] = dims[a] >= P ? 0 : (dims[a] - P) / 2; } }
                    else for (int a = 0; a < 3; a++) o[a] = dims[a] >= P ? (int)(se_unif(&vr) * (dims[a] - P + 1)) : (dims[a] - P) / 2;
                    const int perm[3] = {0, 1, 2}, flip[3] = {0, 0, 0};
                    for (int n = 0; n < N; n++) {   /* (the batch shape is fixed: the crop repeated) */
                        crop_case(c, o, P, perm, flip, hI, hL + (size_t)n * V);
                        if (n == 0) medial_target(hL, P, hS); else memcpy(hS + (size_t)n * V, hS, V);
                        if (n == 0) { xh2d(u8d, hI, V); if (enc_fwd(&E, u8d, P, c, f, 0)) return 1; }
                        else for (int lv = 0; lv < E.nl; lv++) { const int e = P >> lv; nn_d2d_async(f[lv] + (size_t)n * E.ch[lv] * e * e * e, f[lv], (size_t)E.ch[lv] * e * e * e * 4); }
                    }
                    nn_h2d(dL, hL, (size_t)N * V); nn_h2d(dS, hS, (size_t)N * V);
                    dec_fwd(&D, &A, f);
                    se_loss_sums(A.lg, dL, dS, 1, (long long)V, dsum); double h8[8]; nn_d2h(h8, dsum, 64);
                    if (h8[1] > 0) { vl += h8[0] / h8[1] - (2 * h8[2] + 1e-5) / (h8[3] + h8[4] + 1e-5) - (h8[5] + 1e-5) / (h8[6] + 1e-5); vn++; }
                    nn_zero(dvc, 40); se_val_counts(A.lg, dL, dS, 1, (long long)V, dvc); unsigned long long hc[5]; nn_d2h(hc, dvc, 40);
                    for (int a = 0; a < 5; a++) tot[a] += hc[a];
                }
            }
            const double dice = tot[1] + tot[2] ? 2.0 * tot[0] / (double)(tot[1] + tot[2]) : 1.0, rec = tot[4] ? (double)tot[3] / tot[4] : 0;
            fprintf(stderr, "val step %d: loss %.4f foreground dice %.4f medial recall %.4f (%d crops, %llu foreground, %llu medial voxels)\n", step, vn ? vl / vn : 0, dice, rec, vn, tot[2], tot[4]);
            if (mf) { fprintf(mf, "%d\tval\t%.5f\t%.5f\t%.5f\n", step, vn ? vl / vn : 0, dice, rec); fflush(mf); }
            if (dice > best) best = dice;
        }
    }
    fprintf(stderr, "surf-eval: best validation foreground dice %.4f\n", best);
    if (mf) fclose(mf);
    return 0;
}
