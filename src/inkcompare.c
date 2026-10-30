/* ufsm ink-compare, streamed: the renders are read strip by strip (tiff_open_lazy / tiff_read_strip) and never held whole, so a
   whole wrap (1.5 Gpixel renders) runs in a few hundred MB instead of ~24 GB. The bytes out are those of the in-memory version
   (ink_compare_mem in render.c, still used for --best-crop, non-strip TIFFs and UFSM_INKCMP_MEM=1):
   - pass 1 (strips on threads): the masked histograms (CT > 0) and the integer sums of every pair (corr); means, percentiles, the
     stretch ranges and the 5% rule all come from these exact integer counts;
   - pass 2 (band): the masked box means of radius rs and rl as running integer sums (columns over a ring of rows, then along the
     row; column stripes on threads), each window sum the exact integer the integral images gave, so s / c, the float means and
     their difference are bit-identical; m1 / m2 are summed by one thread in pixel order as before;
   - pass 3 (panels -> JPEG): each thread makes runs of output rows, pulling Pillow's BOX half and BILINEAR fit through two
     rings of horizontally resampled rows (pil_hrow, then the same fixed-point vertical sum as pil_vrow); the panels' contrast is a
     256-entry table per input (the CT's stretch maps 0 to 0, i.e. CT == 0 is the mask), the labels drawn per row with the same font. */
#include "imgops.h"
#include "tiff.h"
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int ink_compare_mem(int argc, char **argv);   /* render.c */

static const char *icopt(int argc, char **argv, const char *name, const char *def) {
    for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return def;
}
static int icflag(int argc, char **argv, const char *name) { for (int i = 2; i < argc; i++) if (!strcmp(argv[i], name)) return 1; return 0; }
static int icarg(int argc, char **argv, const char *name) { for (int i = 2; i < argc; i++) if (!strcmp(argv[i], name)) return i; return -1; }

enum { MAXI = 8, NIMG = MAXI + 1 };
typedef struct {
    int n; tiff *t[NIMG]; int IW, IH, rps, ns;   /* the inputs (0 = CT) and their common strip layout */
    int cx, cy, W, H;                             /* the crop (the whole render by default) */
    int nth; _Atomic int fail;
} icx;
typedef struct { int strip; uint8_t *buf, *scr; size_t cap; } icstrip;   /* one thread's decoded strip of one input */

static const uint8_t *ic_row(icx *X, int k, icstrip *c, int y) {   /* crop row y of input k */
    const int ay = X->cy + y, s = ay / X->rps;
    if (c->strip != s) {
        if (!c->buf && !(c->buf = malloc((size_t)X->rps * X->IW))) { atomic_store(&X->fail, 1); return nullptr; }
        if (tiff_read_strip(X->t[k], 0, s, c->buf, &c->scr, &c->cap)) { fprintf(stderr, "ink-compare: %s\n", tiff_error()); atomic_store(&X->fail, 1); c->strip = -1; return nullptr; }
        c->strip = s;
    }
    return c->buf + (size_t)(ay - s * X->rps) * X->IW + X->cx;
}
static void ic_strips_free(icstrip *c, int n) { for (int k = 0; k < n; k++) { free(c[k].buf); free(c[k].scr); } }

/* ---- pass 1: histograms and pair sums over CT > 0 ---- */
typedef struct {
    icx *X; _Atomic int next; pthread_mutex_t mu;
    uint64_t hist[NIMG][256], m, sab[NIMG][NIMG];   /* masked histograms (CT too), mask count, pair product sums (a < b, inks) */
} p1_job;
static void *p1_worker(void *arg) {
    p1_job *J = arg; icx *X = J->X; const int n = X->n, W = X->W;
    uint64_t (*h)[256] = calloc(NIMG, sizeof *h), m = 0, (*sab)[NIMG] = calloc(NIMG, sizeof *sab);
    icstrip c[NIMG]; for (int k = 0; k < NIMG; k++) c[k] = (icstrip){-1, nullptr, nullptr, 0};
    if (!h || !sab) { atomic_store(&X->fail, 1); free(h); free(sab); return nullptr; }
    const int s0 = X->cy / X->rps, s1 = (X->cy + X->H - 1) / X->rps;
    for (int s; (s = s0 + atomic_fetch_add(&J->next, 1)) <= s1 && !atomic_load(&X->fail);) {
        int y0 = s * X->rps - X->cy, y1 = y0 + X->rps; if (y0 < 0) y0 = 0; if (y1 > X->H) y1 = X->H;
        for (int y = y0; y < y1; y++) {
            const uint8_t *r[NIMG];
            for (int k = 0; k < n; k++) if (!(r[k] = ic_row(X, k, &c[k], y))) goto out;
            for (int x = 0; x < W; x++) {
                if (!r[0][x]) continue;
                m++;
                for (int k = 0; k < n; k++) h[k][r[k][x]]++;
                for (int a = 1; a < n; a++) for (int b = a + 1; b < n; b++) sab[a][b] += (uint64_t)r[a][x] * r[b][x];
            }
        }
    }
out:
    pthread_mutex_lock(&J->mu);
    J->m += m;
    for (int k = 0; k < n; k++) for (int v = 0; v < 256; v++) J->hist[k][v] += h[k][v];
    for (int a = 1; a < n; a++) for (int b = a + 1; b < n; b++) J->sab[a][b] += sab[a][b];
    pthread_mutex_unlock(&J->mu);
    ic_strips_free(c, NIMG); free(h); free(sab);
    return nullptr;
}

/* ---- pass 2: the band statistic of every ink ---- */
typedef struct {
    icx *X; int rs, rl, B, R;            /* radii, rows per block, ring rows */
    uint8_t *ring;                       /* R rows x n inputs x W (row y at y % R) */
    int y0, y1;                          /* the block */
    float *df[2]; int cur;               /* per ink, per block row: (box mean rs - box mean rl) as float, NAN off the mask */
    int nstripe; int *xa, *xb;           /* column stripes */
    int32_t **vs;                        /* per stripe: column sums, per ink (S rs, C rs, S rl, C rl) over the stripe + halo */
    int loaded;                          /* rows [0, loaded) read into the ring */
    _Atomic int next;
    /* loading */
    int ly0, ly1;
} p2_job;
static inline uint8_t *p2_row(p2_job *J, int k, int y) { return J->ring + ((size_t)(y % J->R) * J->X->n + k) * J->X->W; }
static void *p2_load(void *arg) {   /* rows [ly0, ly1) into the ring, a strip per task */
    p2_job *J = arg; icx *X = J->X;
    icstrip c[NIMG]; for (int k = 0; k < NIMG; k++) c[k] = (icstrip){-1, nullptr, nullptr, 0};
    const int s0 = (X->cy + J->ly0) / X->rps, s1 = (X->cy + J->ly1 - 1) / X->rps;
    for (int s; (s = s0 + atomic_fetch_add(&J->next, 1)) <= s1 && !atomic_load(&X->fail);) {
        int y0 = s * X->rps - X->cy, y1 = y0 + X->rps; if (y0 < J->ly0) y0 = J->ly0; if (y1 > J->ly1) y1 = J->ly1;
        for (int y = y0; y < y1; y++) for (int k = 0; k < X->n; k++) {
            const uint8_t *r = ic_row(X, k, &c[k], y); if (!r) goto out;
            memcpy(p2_row(J, k, y), r, (size_t)X->W);
        }
    }
out:
    ic_strips_free(c, NIMG);
    return nullptr;
}
/* stripe t's column sums at the window of row y: add row y + r, drop row y - r - 1 (each radius) */
static void p2_vstep(p2_job *J, int t, int y, int first) {
    icx *X = J->X; const int n = X->n, W = X->W, H = X->H, rl = J->rl;
    const int ha = J->xa[t] - rl < 0 ? 0 : J->xa[t] - rl, hb = J->xb[t] + rl > W ? W : J->xb[t] + rl, hw = hb - ha;
    const int rad[2] = {J->rs, J->rl};
    for (int q = 0; q < 2; q++) {
        const int r = rad[q];
        if (first) for (int j = 0; j < r && j < H; j++) {   /* the window of row 0 without its last row */
            const uint8_t *m = p2_row(J, 0, j) + ha;
            for (int k = 1; k < n; k++) { int32_t *S = J->vs[t] + ((size_t)(k - 1) * 4 + 2 * q) * hw, *C = S + hw; const uint8_t *v = p2_row(J, k, j) + ha;
                                          for (int x = 0; x < hw; x++) if (m[x]) { S[x] += v[x]; C[x]++; } }
        }
        if (y + r < H) { const uint8_t *m = p2_row(J, 0, y + r) + ha;
            for (int k = 1; k < n; k++) { int32_t *S = J->vs[t] + ((size_t)(k - 1) * 4 + 2 * q) * hw, *C = S + hw; const uint8_t *v = p2_row(J, k, y + r) + ha;
                                          for (int x = 0; x < hw; x++) if (m[x]) { S[x] += v[x]; C[x]++; } } }
        if (y - r - 1 >= 0) { const uint8_t *m = p2_row(J, 0, y - r - 1) + ha;
            for (int k = 1; k < n; k++) { int32_t *S = J->vs[t] + ((size_t)(k - 1) * 4 + 2 * q) * hw, *C = S + hw; const uint8_t *v = p2_row(J, k, y - r - 1) + ha;
                                          for (int x = 0; x < hw; x++) if (m[x]) { S[x] -= v[x]; C[x]--; } } }
    }
}
static void *p2_worker(void *arg) {   /* stripe t = next: the block's rows */
    p2_job *J = arg; icx *X = J->X; const int n = X->n, W = X->W;
    for (int t; (t = atomic_fetch_add(&J->next, 1)) < J->nstripe;) {
        const int xa = J->xa[t], xb = J->xb[t], ha = xa - J->rl < 0 ? 0 : xa - J->rl, hb = xb + J->rl > W ? W : xb + J->rl, hw = hb - ha;
        for (int y = J->y0; y < J->y1; y++) {
            p2_vstep(J, t, y, y == 0);
            const uint8_t *m = p2_row(J, 0, y);
            for (int k = 1; k < n; k++) {
                float *o = J->df[J->cur] + ((size_t)(k - 1) * J->B + (y - J->y0)) * W;
                const int32_t *Ss = J->vs[t] + (size_t)(k - 1) * 4 * hw, *Cs = Ss + hw, *Sl = Cs + hw, *Cl = Sl + hw;
                /* horizontal windows [x - r, x + r] clipped, as running sums over the column sums (indices relative to ha) */
                int64_t ss = 0, cs = 0, sl = 0, cl = 0;
                for (int x = xa - J->rs < 0 ? 0 : xa - J->rs; x < (xa + J->rs + 1 < W ? xa + J->rs + 1 : W); x++) { ss += Ss[x - ha]; cs += Cs[x - ha]; }
                for (int x = xa - J->rl < 0 ? 0 : xa - J->rl; x < (xa + J->rl + 1 < W ? xa + J->rl + 1 : W); x++) { sl += Sl[x - ha]; cl += Cl[x - ha]; }
                for (int x = xa; x < xb; x++) {
                    if (x > xa) {
                        if (x + J->rs < W) { ss += Ss[x + J->rs - ha]; cs += Cs[x + J->rs - ha]; }
                        if (x - J->rs - 1 >= 0) { ss -= Ss[x - J->rs - 1 - ha]; cs -= Cs[x - J->rs - 1 - ha]; }
                        if (x + J->rl < W) { sl += Sl[x + J->rl - ha]; cl += Cl[x + J->rl - ha]; }
                        if (x - J->rl - 1 >= 0) { sl -= Sl[x - J->rl - 1 - ha]; cl -= Cl[x - J->rl - 1 - ha]; }
                    }
                    if (!m[x]) { o[x] = NAN; continue; }
                    const float bs = cs > 0 ? (float)((double)ss / (double)cs) : 0.f, bl = cl > 0 ? (float)((double)sl / (double)cl) : 0.f;
                    o[x] = bs - bl;
                }
            }
        }
    }
    return nullptr;
}
static int p2_band(icx *X, int rs, int rl, double *band, const long long *cnt) {
    p2_job J; memset(&J, 0, sizeof J); J.X = X; J.rs = rs; J.rl = rl; J.B = 256; J.R = J.B + 2 * rl + 2;
    const int n = X->n, W = X->W, H = X->H, ni = n - 1;
    J.nstripe = X->nth * 2 < W / 256 ? X->nth * 2 : W / 256 > 0 ? W / 256 : 1;
    J.xa = malloc(J.nstripe * sizeof *J.xa); J.xb = malloc(J.nstripe * sizeof *J.xb); J.vs = calloc(J.nstripe, sizeof *J.vs);
    J.ring = malloc((size_t)J.R * n * W); J.df[0] = malloc((size_t)ni * J.B * W * sizeof(float)); J.df[1] = malloc((size_t)ni * J.B * W * sizeof(float));
    int rc = !J.xa || !J.xb || !J.vs || !J.ring || !J.df[0] || !J.df[1];
    for (int t = 0; t < J.nstripe && !rc; t++) {
        J.xa[t] = (int)((int64_t)W * t / J.nstripe); J.xb[t] = (int)((int64_t)W * (t + 1) / J.nstripe);
        const int ha = J.xa[t] - rl < 0 ? 0 : J.xa[t] - rl, hb = J.xb[t] + rl > W ? W : J.xb[t] + rl;
        if (!(J.vs[t] = calloc((size_t)ni * 4 * (hb - ha), sizeof(int32_t)))) rc = 1;
    }
    double m1[NIMG] = {0}, m2[NIMG] = {0};
    const int nt = X->nth < 1 ? 1 : X->nth > 64 ? 64 : X->nth; pthread_t th[64];
    int pend = -1, pend_rows = 0;   /* the block whose df the main thread sums while the next one is computed */
    for (int y0 = 0; y0 < H && !rc; y0 += J.B) {
        const int y1 = y0 + J.B < H ? y0 + J.B : H, need = y1 - 1 + rl + 1 < H ? y1 + rl : H;
        if (need > J.loaded) {   /* rows [loaded, need) into the ring */
            J.ly0 = J.loaded; J.ly1 = need; atomic_store(&J.next, 0);
            for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, p2_load, &J);
            for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
            if (atomic_load(&X->fail)) { rc = 1; break; }
            J.loaded = need;
        }
        J.y0 = y0; J.y1 = y1; J.cur = (y0 / J.B) & 1; atomic_store(&J.next, 0);
        for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, p2_worker, &J);
        if (pend >= 0) for (int k = 1; k < n; k++) for (int r = 0; r < pend_rows; r++) {   /* the previous block, in pixel order */
            const float *o = J.df[pend] + ((size_t)(k - 1) * J.B + r) * W;
            for (int x = 0; x < W; x++) if (!isnan(o[x])) { const double d = o[x] / 255.0; m1[k] += d; m2[k] += d * d; }
        }
        for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
        pend = J.cur; pend_rows = y1 - y0;
    }
    if (!rc && pend >= 0) for (int k = 1; k < n; k++) for (int r = 0; r < pend_rows; r++) {
        const float *o = J.df[pend] + ((size_t)(k - 1) * J.B + r) * W;
        for (int x = 0; x < W; x++) if (!isnan(o[x])) { const double d = o[x] / 255.0; m1[k] += d; m2[k] += d * d; }
    }
    /* m1 / m2 per ink must be summed in pixel order over the whole image: the per-ink loops above run row by row in order */
    for (int k = 1; k < n && !rc; k++) {
        if (!cnt[k]) { band[k] = 0; continue; }
        const double a = m1[k] / cnt[k], b = m2[k] / cnt[k]; band[k] = sqrt(b - a * a > 0 ? b - a * a : 0);
    }
    for (int t = 0; t < J.nstripe && J.vs; t++) free(J.vs[t]);
    free(J.vs); free(J.xa); free(J.xb); free(J.ring); free(J.df[0]); free(J.df[1]);
    return rc ? -1 : 0;
}

/* ---- pass 3: panels -> half (BOX) -> fit (BILINEAR) -> RGB rows ---- */
typedef struct { int k, *b; int32_t *kk; } pcoef;   /* ksize, bounds, weights */
typedef struct {
    icx *X; int np, first;                 /* panels: inputs first .. first + np - 1 */
    uint8_t lut[NIMG][256]; const char *lab[NIMG]; int sc;
    int TW, H0;                            /* the panel row width, rows */
    int half, fit; int w1, h1, ow, oh;     /* stages: half -> w1 x h1; fit -> ow x oh */
    pcoef ah, av, bh, bv;
    uint8_t *out; int rows; _Atomic int next;
    int rgb[2], ctmin;                     /* rgb[0] >= 0: one agreement panel, R = B = ink rgb[0], G = ink rgb[1]; CT < ctmin dark grey */
} p3_job;
typedef struct { uint8_t *buf; int *tag; int cap; size_t len; } p3ring;
static void p3_full_row(p3_job *J, icstrip *c, int y, uint8_t *row) {   /* ink_panels' row y */
    icx *X = J->X; const int W = X->W;
    memset(row, 255, (size_t)J->TW * 3);
    if (J->rgb[0] >= 0) {   /* recto (magenta) x verso (green): both white, neither black */
        const uint8_t *m = ic_row(X, 0, &c[0], y), *a = ic_row(X, J->rgb[0], &c[J->rgb[0]], y), *b = ic_row(X, J->rgb[1], &c[J->rgb[1]], y);
        if (!m || !a || !b) return;
        const uint8_t *La = J->lut[J->rgb[0]], *Lb = J->lut[J->rgb[1]];
        for (int x = 0; x < W; x++) {
            uint8_t *o = row + 3 * x;
            if (!m[x] || m[x] < J->ctmin) { o[0] = o[1] = o[2] = 40; continue; }
            o[0] = o[2] = La[a[x]]; o[1] = Lb[b[x]];
        }
        const uint8_t yel[3] = {255, 220, 0}; const int sc = J->sc;
        if (y >= 2 * sc && y < 2 * sc + 7 * sc && J->lab[0]) img_text_scaled(row, J->TW, 1, 2 * sc, 2 * sc - y, J->lab[0], sc, yel);
        return;
    }
    for (int p = 0; p < J->np; p++) {
        const int k = J->first + p; const uint8_t *s = ic_row(X, k, &c[k], y), *L = J->lut[k];
        uint8_t *o = row + (size_t)p * (W + 8) * 3;
        if (!s) return;
        for (int x = 0; x < W; x++) o[3 * x] = o[3 * x + 1] = o[3 * x + 2] = L[s[x]];
    }
    const uint8_t yel[3] = {255, 220, 0}; const int sc = J->sc;
    if (y >= 2 * sc && y < 2 * sc + 7 * sc)
        for (int p = 0; p < J->np; p++) if (J->lab[p]) img_text_scaled(row, J->TW, 1, p * (W + 8) + 2 * sc, 2 * sc - y, J->lab[p], sc, yel);
}
static void p3_vsum(uint8_t *const *rows, int n, const int32_t *k, size_t len, uint8_t *out) {   /* pil_vrow over row pointers */
    for (size_t i = 0; i < len; i++) { int ss = 1 << 21; for (int r = 0; r < n; r++) ss += rows[r][i] * k[r]; out[i] = pil_clip8(ss); }
}
static uint8_t *p3_hA(p3_job *J, icstrip *c, p3ring *RA, uint8_t *full, int fr) {   /* full row fr, BOX-resampled horizontally (ring A) */
    const int slot = fr % RA->cap; uint8_t *d = RA->buf + (size_t)slot * RA->len;
    if (RA->tag[slot] != fr) { p3_full_row(J, c, fr, full); pil_hrow(full, 3, d, J->w1, J->ah.k, J->ah.b, J->ah.kk); RA->tag[slot] = fr; }
    return d;
}
static void p3_half_row(p3_job *J, icstrip *c, p3ring *RA, uint8_t *full, int r, uint8_t *out) {   /* stage A output row r */
    if (!J->half) { p3_full_row(J, c, r, out); return; }
    const int b0 = J->av.b[2 * r], nb = J->av.b[2 * r + 1]; uint8_t *rows[64];
    for (int q = 0; q < nb; q++) rows[q] = p3_hA(J, c, RA, full, b0 + q);
    p3_vsum(rows, nb, J->av.kk + (size_t)r * J->av.k, (size_t)J->w1 * 3, out);
}
static void *p3_worker(void *arg) {
    p3_job *J = arg; icx *X = J->X;
    icstrip c[NIMG]; for (int k = 0; k < NIMG; k++) c[k] = (icstrip){-1, nullptr, nullptr, 0};
    p3ring RA = {0}, RB = {0};
    RA.cap = J->half ? J->av.k + 1 : 1; RA.len = (size_t)J->w1 * 3; RA.buf = malloc(RA.cap * RA.len); RA.tag = malloc(RA.cap * sizeof(int));
    RB.cap = J->fit ? J->bv.k + 1 : 1; RB.len = (size_t)J->ow * 3; RB.buf = malloc(RB.cap * RB.len); RB.tag = malloc(RB.cap * sizeof(int));
    uint8_t *full = malloc((size_t)J->TW * 3), *hrow = malloc((size_t)J->w1 * 3);
    if (!RA.buf || !RA.tag || !RB.buf || !RB.tag || !full || !hrow) { atomic_store(&X->fail, 1); goto out; }
    for (int i = 0; i < RA.cap; i++) RA.tag[i] = -1;
    for (int i = 0; i < RB.cap; i++) RB.tag[i] = -1;
    for (int s; (s = atomic_fetch_add(&J->next, 1)) * J->rows < J->oh && !atomic_load(&X->fail);) {
        const int y0 = s * J->rows, y1 = y0 + J->rows < J->oh ? y0 + J->rows : J->oh;
        for (int y = y0; y < y1; y++) {
            uint8_t *o = J->out + (size_t)y * J->ow * 3;
            if (!J->fit) { p3_half_row(J, c, &RA, full, y, o); continue; }
            const int b0 = J->bv.b[2 * y], nb = J->bv.b[2 * y + 1]; uint8_t *rows[64];
            for (int q = 0; q < nb; q++) {
                const int r = b0 + q, slot = r % RB.cap; uint8_t *d = RB.buf + (size_t)slot * RB.len;
                if (RB.tag[slot] != r) { p3_half_row(J, c, &RA, full, r, hrow); pil_hrow(hrow, 3, d, J->ow, J->bh.k, J->bh.b, J->bh.kk); RB.tag[slot] = r; }
                rows[q] = d;
            }
            p3_vsum(rows, nb, J->bv.kk + (size_t)y * J->bv.k, (size_t)J->ow * 3, o);
        }
    }
out:
    ic_strips_free(c, NIMG); free(RA.buf); free(RA.tag); free(RB.buf); free(RB.tag); free(full); free(hrow);
    return nullptr;
}
/* panels first .. first + np - 1 -> RGB *ow x *oh (as ink_panels + img_fit_rgb) */
static uint8_t *p3_panels(icx *X, int first, int np, uint8_t lut[NIMG][256], const char *const *lab, int half, int maxpx, int *owp, int *ohp, const int *rgb, int ctmin) {
    p3_job *J = calloc(1, sizeof *J); if (!J) return nullptr;
    J->X = X; J->np = np; J->first = first; memcpy(J->lut, lut, sizeof J->lut);
    J->rgb[0] = rgb ? rgb[0] : -1; J->rgb[1] = rgb ? rgb[1] : -1; J->ctmin = ctmin;
    for (int p = 0; p < np; p++) J->lab[p] = lab ? lab[p] : nullptr;
    J->TW = np * X->W + 8 * (np - 1); J->H0 = X->H; J->sc = X->H / 200 > 1 ? X->H / 200 : 2;
    if (J->rgb[0] >= 0 && lab && lab[0]) { const int fit = J->TW / (8 * (int)strlen(lab[0]) + 8); if (J->sc > fit) J->sc = fit > 1 ? fit : 1; }   /* the legend fits the one panel */
    J->half = half; J->w1 = half ? J->TW / 2 : J->TW; J->h1 = half ? X->H / 2 : X->H;
    uint8_t *ret = nullptr;
    if (J->w1 < 1 || J->h1 < 1) goto done;
    const int m = J->w1 > J->h1 ? J->w1 : J->h1; J->fit = maxpx > 0 && m > maxpx;
    J->ow = J->w1; J->oh = J->h1;
    if (J->fit) { const double s = (double)maxpx / m; J->ow = (int)lround(J->w1 * s); J->oh = (int)lround(J->h1 * s); if (J->ow < 1) J->ow = 1; if (J->oh < 1) J->oh = 1; }
    if (half && (!(J->ah.k = pil_coeffs(J->TW, J->w1, PIL_BOX, &J->ah.b, &J->ah.kk)) || !(J->av.k = pil_coeffs(X->H, J->h1, PIL_BOX, &J->av.b, &J->av.kk)))) goto done;
    if (J->fit && (!(J->bh.k = pil_coeffs(J->w1, J->ow, PIL_BILINEAR, &J->bh.b, &J->bh.kk)) || !(J->bv.k = pil_coeffs(J->h1, J->oh, PIL_BILINEAR, &J->bv.b, &J->bv.kk)))) goto done;
    if (J->av.k > 64 || J->bv.k > 64) { fprintf(stderr, "ink-compare: resample kernel over 64 rows\n"); goto done; }
    J->out = malloc((size_t)J->ow * J->oh * 3); if (!J->out) goto done;
    J->rows = 16;
    const int nt = X->nth < 1 ? 1 : X->nth > 64 ? 64 : X->nth; pthread_t th[64];
    for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, p3_worker, J);
    for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
    if (atomic_load(&X->fail)) { free(J->out); goto done; }
    ret = J->out; *owp = J->ow; *ohp = J->oh;
done:
    free(J->ah.b); free(J->ah.kk); free(J->av.b); free(J->av.kk); free(J->bh.b); free(J->bh.kk); free(J->bv.b); free(J->bv.kk); free(J);
    return ret;
}

static int stats_from_hist(const uint64_t *hist, ink_stats_t *st) {   /* ink_stats without the band */
    memset(st, 0, sizeof *st); uint64_t sum = 0;
    for (int v = 0; v < 256; v++) { st->n += (long long)hist[v]; sum += (uint64_t)v * hist[v]; }
    if (!st->n) return 0;
    st->mean = (double)sum / st->n / 255.0;
    long long hi = 0; for (int v = 128; v < 256; v++) hi += (long long)hist[v];
    st->frac50 = (double)hi / st->n;
    const double q[3] = {0.50, 0.95, 0.99}; double *qo[3] = {&st->p50, &st->p95, &st->p99};
    for (int k = 0; k < 3; k++) {
        long long c = 0; const double t = q[k] * st->n; int v = 0;
        for (; v < 255; v++) { c += (long long)hist[v]; if (c >= t) break; }
        *qo[k] = v / 255.0;
    }
    return 1;
}

int cmd_ink_compare(int argc, char **argv) {
    const int ci = icarg(argc, argv, "--ct");
    if (argc < 5 || ci < 0 || ci + 1 >= argc || icarg(argc, argv, "--ink") < 0 || icarg(argc, argv, "--best-crop") >= 0 || getenv("UFSM_INKCMP_MEM"))
        return ink_compare_mem(argc, argv);
    const double px = atof(icopt(argc, argv, "--px-um", "2.4")), rsu = atof(icopt(argc, argv, "--rs-um", "25")), rlu = atof(icopt(argc, argv, "--rl-um", "500"));
    if (!(px > 0) || !(rsu > 0) || !(rlu > rsu)) { fprintf(stderr, "ink-compare: need --px-um > 0 and 0 < --rs-um < --rl-um\n"); return 2; }
    icx X; memset(&X, 0, sizeof X); X.nth = atoi(icopt(argc, argv, "--threads", "16"));
    const char *lab[NIMG] = {"ct"}; int rc = 0;
    /* the inputs: strip TIFFs of one size and strip height, else the in-memory version */
    {
        tiff_page pg; int rps, ns;
        X.t[0] = tiff_open_lazy(argv[ci + 1]);
        if (!X.t[0] || tiff_page_info(X.t[0], 0, &pg) || pg.bits != 8 || pg.spp != 1 || tiff_strip_rows(X.t[0], 0, &rps, &ns)) { if (X.t[0]) tiff_close(X.t[0]); return ink_compare_mem(argc, argv); }
        X.IW = pg.w; X.IH = pg.h; X.rps = rps; X.ns = ns; X.n = 1;
        for (int a = 2; a + 1 < argc; a++) {
            if (strcmp(argv[a], "--ink")) continue;
            const char *eq = strchr(argv[a + 1], '=');
            if (!eq || X.n > MAXI) { fprintf(stderr, "ink-compare: --ink LABEL=PATH (at most %d)\n", MAXI); rc = 2; goto end; }
            tiff *t = tiff_open_lazy(eq + 1); tiff_page q;
            if (!t || tiff_page_info(t, 0, &q) || q.bits != 8 || q.spp != 1) { fprintf(stderr, "ink-compare: cannot read %s as a %d x %d render\n", eq + 1, X.IW, X.IH); if (t) tiff_close(t); rc = 1; goto end; }
            if (q.w != X.IW || q.h != X.IH) { fprintf(stderr, "ink-compare: cannot read %s as a %d x %d render\n", eq + 1, X.IW, X.IH); tiff_close(t); rc = 1; goto end; }
            if (tiff_strip_rows(t, 0, &rps, &ns) || rps != X.rps) { tiff_close(t); for (int k = 0; k < X.n; k++) tiff_close(X.t[k]); for (int k = 1; k < X.n; k++) free((void *)lab[k]); return ink_compare_mem(argc, argv); }
            lab[X.n] = strndup(argv[a + 1], (size_t)(eq - argv[a + 1])); X.t[X.n++] = t;
        }
    }
    const int n = X.n, W0 = X.IW, H0 = X.IH;
    X.cx = 0; X.cy = 0; X.W = W0; X.H = H0;
    {
        const char *cr = icopt(argc, argv, "--crop", nullptr), *cf = icopt(argc, argv, "--crop-frac", nullptr); char crb[96];
        if (!cr && cf) {
            double fx, fy; int cw0, ch0;
            if (sscanf(cf, "%lf,%lf,%d,%d", &fx, &fy, &cw0, &ch0) != 4) { fprintf(stderr, "ink-compare: --crop-frac fx,fy,w,h\n"); rc = 2; goto end; }
            snprintf(crb, sizeof crb, "%d,%d,%d,%d", (int)(fx * W0), (int)(fy * H0), cw0, ch0); cr = crb;
        }
        if (cr) {
            int cx, cy, cw, ch;
            if (sscanf(cr, "%d,%d,%d,%d", &cx, &cy, &cw, &ch) != 4 || cx < 0 || cy < 0 || cw < 2 || ch < 2 || cx >= W0 || cy >= H0) { fprintf(stderr, "ink-compare: --crop x,y,w,h inside the %d x %d render\n", W0, H0); rc = 2; goto end; }
            if (cx + cw > W0) cw = W0 - cx;
            if (cy + ch > H0) ch = H0 - cy;
            X.cx = cx; X.cy = cy; X.W = cw; X.H = ch;
        }
    }
    const size_t np = (size_t)X.W * X.H;
    /* pass 1 */
    p1_job *P = calloc(1, sizeof *P); if (!P) { rc = 1; goto end; }
    P->X = &X; pthread_mutex_init(&P->mu, nullptr);
    { const int nt = X.nth < 1 ? 1 : X.nth > 64 ? 64 : X.nth; pthread_t th[64];
      for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, p1_worker, P);
      for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr); }
    pthread_mutex_destroy(&P->mu);
    if (atomic_load(&X.fail)) { free(P); rc = 1; goto end; }
    /* statistics */
    const int rs = (int)lround(rsu / px), rl = (int)lround(rlu / px);
    {
        ink_stats_t st[NIMG]; long long cnt[NIMG] = {0}; double band[NIMG] = {0};
        for (int k = 1; k < n; k++) { stats_from_hist(P->hist[k], &st[k]); cnt[k] = st[k].n; }
        if (n > 1 && p2_band(&X, rs < 1 ? 1 : rs, rl < 2 ? 2 : rl, band, cnt)) { fprintf(stderr, "ink-compare: out of memory\n"); free(P); rc = 1; goto end; }
        for (int k = 1; k < n; k++) { st[k].band = st[k].n ? band[k] : 0;
            printf("ink %s: n %lld mean %.4f frac50 %.4f p50 %.3f p95 %.3f p99 %.3f band %.4f\n", lab[k], st[k].n, st[k].mean, st[k].frac50, st[k].p50, st[k].p95, st[k].p99, st[k].band);
            if (getenv("UFSM_INKCMP_FULL")) printf("full %s: %.17g %.17g %.17g\n", lab[k], st[k].mean, st[k].frac50, st[k].band); }
        for (int a = 1; a < n; a++) for (int b = a + 1; b < n; b++) {   /* ink_corr from the exact integer sums */
            uint64_t sa = 0, sb = 0, saa = 0, sbb = 0; const long long k = (long long)P->m;
            for (int v = 0; v < 256; v++) { sa += (uint64_t)v * P->hist[a][v]; saa += (uint64_t)v * v * P->hist[a][v]; sb += (uint64_t)v * P->hist[b][v]; sbb += (uint64_t)v * v * P->hist[b][v]; }
            double r = 0;
            if (k >= 2) { const double da = (double)sa, db = (double)sb, va = (double)saa - da * da / k, vb = (double)sbb - db * db / k;
                          r = va > 0 && vb > 0 ? ((double)P->sab[a][b] - da * db / k) / sqrt(va * vb) : 0; }
            printf("corr %s %s: %.4f\n", lab[a], lab[b], r);
            if (getenv("UFSM_INKCMP_FULL")) printf("full corr: %.17g\n", r);
        }
    }
    /* the display tables */
    uint8_t lut[NIMG][256]; int isct[NIMG] = {1};
    for (int k = 0; k < n; k++) for (int v = 0; v < 256; v++) lut[k][v] = (uint8_t)v;
    {
        int flo = 0, fhi = 0;
        if (*icopt(argc, argv, "--fixed", "") && (sscanf(icopt(argc, argv, "--fixed", ""), "%d,%d", &flo, &fhi) != 2 || fhi <= flo)) { fprintf(stderr, "ink-compare: --fixed LO,HI (ink bytes, LO < HI)\n"); free(P); rc = 2; goto end; }
        if (fhi > flo) for (int k = 1; k < n; k++) for (int v = 0; v < 256; v++) { const int g = (v - flo) * 255 / (fhi - flo); lut[k][v] = (uint8_t)(g < 0 ? 0 : g > 255 ? 255 : g); }
        else if (icflag(argc, argv, "--stretch")) for (int k = 1; k < n; k++) {
            const uint64_t *h = P->hist[k]; uint64_t m = 0; for (int v = 0; v < 256; v++) m += h[v];
            int lo = 0, hi = 255; uint64_t c = 0; int gl = 0;
            for (int v = 0; v < 256; v++) { c += h[v]; if (!gl && c >= m / 100) { lo = v; gl = 1; } if (c >= m - m / 100) { hi = v; break; } }
            if (hi <= lo) hi = lo + 1;
            for (int v = 0; v < 256; v++) { const int g = (v - lo) * 255 / (hi - lo); lut[k][v] = (uint8_t)(g < 0 ? 0 : g > 255 ? 255 : g); }
            fprintf(stderr, "ink-compare: %s stretched %d .. %d\n", lab[k], lo, hi);
        }
    }
    if ((double)P->m / (double)np < 0.05) { fprintf(stderr, "ink-compare: under 5%% of the CT pixels have data: no %s\n", argv[2]); rc = 3; free(P); goto end; }
    if (icflag(argc, argv, "--stretch-ct")) {   /* img_stretch over CT > 0: CT 0 stays 0 */
        const uint64_t *h = P->hist[0]; long long cnt = (long long)P->m;
        if (cnt) {
            int lo = 0, hi = 255; long long c = 0;
            for (lo = 0; lo < 255; lo++) { c += (long long)h[lo]; if (c >= 0.01 * cnt) break; }
            c = 0; for (hi = 0; hi < 255; hi++) { c += (long long)h[hi]; if (c >= 0.99 * cnt) break; }
            if (hi <= lo) hi = lo + 1;
            for (int v = 0; v < 256; v++) { const double g = (v - lo) * 255.0 / (hi - lo); lut[0][v] = (uint8_t)(g < 0 ? 0 : g > 255 ? 255 : lround(g)); }
            lut[0][0] = 0;
        }
        isct[0] = 0;
    }
    if (isct[0]) for (int v = 0; v < 256; v++) { const int u = lut[0][v]; float g = ((float)u - 40.f) * 2.2f; g = g < 0.f ? 0.f : g > 255.f ? 255.f : g; lut[0][v] = (uint8_t)g; }
    free(P);
    {
        const char *each = icopt(argc, argv, "--each", nullptr);
        const int maxpx = atoi(icopt(argc, argv, "--max-px", "0")), jq = atoi(icopt(argc, argv, "--quality", "90")), half = !icflag(argc, argv, "--full");
        if (each) {
            const char *suf = icopt(argc, argv, "--each-suffix", ""); const int ect = icflag(argc, argv, "--each-ct");
            for (int k = ect ? 0 : 1; k < n; k++) {
                char p[1400]; snprintf(p, sizeof p, "%s%s%s.jpg", each, lab[k], suf);
                int ow, oh; uint8_t *rgb = p3_panels(&X, k, 1, lut, nullptr, half, maxpx, &ow, &oh, nullptr, 0);
                if (!rgb || jpeg_write_rgb(p, ow, oh, rgb, jq)) { fprintf(stderr, "ink-compare: cannot write %s\n", p); rc = 1; }
                free(rgb);
            }
        } else if (icopt(argc, argv, "--rgb-agree", nullptr)) {   /* one RGB panel: recto ink R + B (magenta), verso ink G; agreement white */
            char la[128] = "", lb[128] = ""; int ix[2] = {-1, -1};
            if (sscanf(icopt(argc, argv, "--rgb-agree", ""), "%127[^,],%127s", la, lb) == 2)
                for (int k = 1; k < n; k++) { if (!strcmp(lab[k], la)) ix[0] = k; if (!strcmp(lab[k], lb)) ix[1] = k; }
            if (ix[0] < 0 || ix[1] < 0) { fprintf(stderr, "ink-compare: --rgb-agree RECTO,VERSO (two --ink labels)\n"); rc = 2; goto end; }
            char title[300]; snprintf(title, sizeof title, "%s=magenta %s=green both=white", la, lb); const char *tl[1] = {title};
            int ow, oh; uint8_t *rgb = p3_panels(&X, 0, 1, lut, tl, half, maxpx, &ow, &oh, ix, atoi(icopt(argc, argv, "--rgb-ct-min", "1")));
            if (!rgb || jpeg_write_rgb(argv[2], ow, oh, rgb, jq)) { fprintf(stderr, "ink-compare: cannot write %s\n", argv[2]); rc = 1; }
            free(rgb);
        } else {
            int ow, oh; uint8_t *rgb = p3_panels(&X, 0, n, lut, lab, half, maxpx, &ow, &oh, nullptr, 0);
            if (!rgb || jpeg_write_rgb(argv[2], ow, oh, rgb, jq)) { fprintf(stderr, "ink-compare: cannot write %s\n", argv[2]); rc = 1; }
            free(rgb);
        }
    }
end:
    fflush(stdout);
    for (int k = 0; k < X.n; k++) tiff_close(X.t[k]);
    for (int k = 1; k < X.n; k++) free((void *)lab[k]);
    return rc;
}
