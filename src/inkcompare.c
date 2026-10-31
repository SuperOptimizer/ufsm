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
#include <unistd.h>
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

/* matplotlib viridis (256 entries, round(255 x the float table)) */
static const uint8_t VIRIDIS[256][3] = {
    {68,1,84}, {68,2,86}, {69,4,87}, {69,5,89}, {70,7,90}, {70,8,92}, {70,10,93}, {70,11,94},
    {71,13,96}, {71,14,97}, {71,16,99}, {71,17,100}, {71,19,101}, {72,20,103}, {72,22,104}, {72,23,105},
    {72,24,106}, {72,26,108}, {72,27,109}, {72,28,110}, {72,29,111}, {72,31,112}, {72,32,113}, {72,33,115},
    {72,35,116}, {72,36,117}, {72,37,118}, {72,38,119}, {72,40,120}, {72,41,121}, {71,42,122}, {71,44,122},
    {71,45,123}, {71,46,124}, {71,47,125}, {70,48,126}, {70,50,126}, {70,51,127}, {70,52,128}, {69,53,129},
    {69,55,129}, {69,56,130}, {68,57,131}, {68,58,131}, {68,59,132}, {67,61,132}, {67,62,133}, {66,63,133},
    {66,64,134}, {66,65,134}, {65,66,135}, {65,68,135}, {64,69,136}, {64,70,136}, {63,71,136}, {63,72,137},
    {62,73,137}, {62,74,137}, {62,76,138}, {61,77,138}, {61,78,138}, {60,79,138}, {60,80,139}, {59,81,139},
    {59,82,139}, {58,83,139}, {58,84,140}, {57,85,140}, {57,86,140}, {56,88,140}, {56,89,140}, {55,90,140},
    {55,91,141}, {54,92,141}, {54,93,141}, {53,94,141}, {53,95,141}, {52,96,141}, {52,97,141}, {51,98,141},
    {51,99,141}, {50,100,142}, {50,101,142}, {49,102,142}, {49,103,142}, {49,104,142}, {48,105,142}, {48,106,142},
    {47,107,142}, {47,108,142}, {46,109,142}, {46,110,142}, {46,111,142}, {45,112,142}, {45,113,142}, {44,113,142},
    {44,114,142}, {44,115,142}, {43,116,142}, {43,117,142}, {42,118,142}, {42,119,142}, {42,120,142}, {41,121,142},
    {41,122,142}, {41,123,142}, {40,124,142}, {40,125,142}, {39,126,142}, {39,127,142}, {39,128,142}, {38,129,142},
    {38,130,142}, {38,130,142}, {37,131,142}, {37,132,142}, {37,133,142}, {36,134,142}, {36,135,142}, {35,136,142},
    {35,137,142}, {35,138,141}, {34,139,141}, {34,140,141}, {34,141,141}, {33,142,141}, {33,143,141}, {33,144,141},
    {33,145,140}, {32,146,140}, {32,146,140}, {32,147,140}, {31,148,140}, {31,149,139}, {31,150,139}, {31,151,139},
    {31,152,139}, {31,153,138}, {31,154,138}, {30,155,138}, {30,156,137}, {30,157,137}, {31,158,137}, {31,159,136},
    {31,160,136}, {31,161,136}, {31,161,135}, {31,162,135}, {32,163,134}, {32,164,134}, {33,165,133}, {33,166,133},
    {34,167,133}, {34,168,132}, {35,169,131}, {36,170,131}, {37,171,130}, {37,172,130}, {38,173,129}, {39,173,129},
    {40,174,128}, {41,175,127}, {42,176,127}, {44,177,126}, {45,178,125}, {46,179,124}, {47,180,124}, {49,181,123},
    {50,182,122}, {52,182,121}, {53,183,121}, {55,184,120}, {56,185,119}, {58,186,118}, {59,187,117}, {61,188,116},
    {63,188,115}, {64,189,114}, {66,190,113}, {68,191,112}, {70,192,111}, {72,193,110}, {74,193,109}, {76,194,108},
    {78,195,107}, {80,196,106}, {82,197,105}, {84,197,104}, {86,198,103}, {88,199,101}, {90,200,100}, {92,200,99},
    {94,201,98}, {96,202,96}, {99,203,95}, {101,203,94}, {103,204,92}, {105,205,91}, {108,205,90}, {110,206,88},
    {112,207,87}, {115,208,86}, {117,208,84}, {119,209,83}, {122,209,81}, {124,210,80}, {127,211,78}, {129,211,77},
    {132,212,75}, {134,213,73}, {137,213,72}, {139,214,70}, {142,214,69}, {144,215,67}, {147,215,65}, {149,216,64},
    {152,216,62}, {155,217,60}, {157,217,59}, {160,218,57}, {162,218,55}, {165,219,54}, {168,219,52}, {170,220,50},
    {173,220,48}, {176,221,47}, {178,221,45}, {181,222,43}, {184,222,41}, {186,222,40}, {189,223,38}, {192,223,37},
    {194,223,35}, {197,224,33}, {200,224,32}, {202,225,31}, {205,225,29}, {208,225,28}, {210,226,27}, {213,226,26},
    {216,226,25}, {218,227,25}, {221,227,24}, {223,227,24}, {226,228,24}, {229,228,25}, {231,228,25}, {234,229,26},
    {236,229,27}, {239,229,28}, {241,229,29}, {244,230,30}, {246,230,32}, {248,230,33}, {251,231,35}, {253,231,37}
};

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
    int cmap;                              /* 1: ink panels (k >= 1) through viridis after their table; CT < ctmin dark grey */
    int diff; float gain;                  /* with rgb: viridis(|ink rgb[0] - ink rgb[1]| * gain) instead of magenta/green */
    int pap, pfloor, pbg, pbgv; float pgain; /* pfloor: the CT ramp's floor (stretched CT units); pbg >= 0: off-papyrus neutral grey pbg; else pbgv >= 0: the ramp at pbgv; else the ramp at pfloor */                  /* with rgb: papyrus look, CT through a brown-tan ramp, ink rgb[0] darkens it toward carbon black */
} p3_job;
typedef struct { uint8_t *buf; int *tag; int cap; size_t len; } p3ring;
static void pap_rgb(int v, float d, uint8_t *o) { img_pap_rgb(v, d, o); }   /* --papyrus: imgops.h */
static void p3_full_row(p3_job *J, icstrip *c, int y, uint8_t *row) {   /* ink_panels' row y */
    icx *X = J->X; const int W = X->W;
    memset(row, 255, (size_t)J->TW * 3);
    if (J->rgb[0] >= 0) {   /* recto (magenta) x verso (green): both white, neither black */
        const uint8_t *m = ic_row(X, 0, &c[0], y), *a = ic_row(X, J->rgb[0], &c[J->rgb[0]], y), *b = ic_row(X, J->rgb[1], &c[J->rgb[1]], y);
        if (!m || !a || !b) return;
        const uint8_t *La = J->lut[J->rgb[0]], *Lb = J->lut[J->rgb[1]];
        for (int x = 0; x < W; x++) {
            uint8_t *o = row + 3 * x;
            if (!m[x] || m[x] < J->ctmin) { if (J->pap) { if (J->pbg >= 0) o[0] = o[1] = o[2] = (uint8_t)J->pbg; else pap_rgb(J->pbgv >= 0 ? J->pbgv : J->pfloor, 0.f, o); } else o[0] = o[1] = o[2] = 40; continue; }   /* papyrus: off-papyrus = neutral grey or the ramp at its floor */
            if (J->pap) {   /* papyrus: CT (raised to the floor) -> dark brown .. warm tan, then carbon ink over it by its density */
                const int v = J->lut[0][m[x]];
                float d = (float)La[a[x]] / 255.f * J->pgain; d = d < .12f ? 0.f : d > .9f ? 1.f : (d - .12f) / .78f; d = d * d * (3.f - 2.f * d) * .92f;
                pap_rgb(v > J->pfloor ? v : J->pfloor, d, o);
                continue;
            }
            if (J->diff) {
                const float d = fabsf((float)La[a[x]] - (float)Lb[b[x]]) * J->gain; const uint8_t *v = VIRIDIS[d > 255.f ? 255 : (int)lroundf(d)];
                o[0] = v[0]; o[1] = v[1]; o[2] = v[2]; continue;
            }
            o[0] = o[2] = La[a[x]]; o[1] = Lb[b[x]];
        }
        const uint8_t yel[3] = {255, 220, 0}; const int sc = J->sc;
        if (y >= 2 * sc && y < 2 * sc + 7 * sc && J->lab[0]) img_text_scaled(row, J->TW, 1, 2 * sc, 2 * sc - y, J->lab[0], sc, yel);
        return;
    }
    const uint8_t *m = J->cmap ? ic_row(X, 0, &c[0], y) : nullptr;
    if (J->cmap && !m) return;
    for (int p = 0; p < J->np; p++) {
        const int k = J->first + p; const uint8_t *s = ic_row(X, k, &c[k], y), *L = J->lut[k];
        uint8_t *o = row + (size_t)p * (W + 8) * 3;
        if (!s) return;
        if (J->cmap && k >= 1) {
            for (int x = 0; x < W; x++) { uint8_t *q = o + 3 * x; if (!m[x] || m[x] < J->ctmin) { q[0] = q[1] = q[2] = 40; continue; } const uint8_t *v = VIRIDIS[L[s[x]]]; q[0] = v[0]; q[1] = v[1]; q[2] = v[2]; }
            continue;
        }
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
static int g_cmap = 0, g_cmap_ctmin = 1;   /* --cmap viridis (cmd_ink_compare), its off-papyrus CT floor (--rgb-ct-min) */
static int g_diff = 0; static float g_diff_gain = 1.f;
   /* --diff-agree (cmd_ink_compare), --diff-gain */
static int g_pap = 0, g_pap_floor = 0, g_pap_bg = -1, g_pap_bgv = -1; static float g_pap_gain = 1.f;   /* --papyrus INK (cmd_ink_compare), --papyrus-gain, --papyrus-floor */
static int g_ct_median = 0;   /* the raw CT median over CT > 0 (cmd_ink_compare; --papyrus-floor median) */
static uint8_t *p3_panels(icx *X, int first, int np, uint8_t lut[NIMG][256], const char *const *lab, int half, int maxpx, int *owp, int *ohp, const int *rgb, int ctmin) {
    p3_job *J = calloc(1, sizeof *J); if (!J) return nullptr;
    J->X = X; J->np = np; J->first = first; memcpy(J->lut, lut, sizeof J->lut);
    J->rgb[0] = rgb ? rgb[0] : -1; J->rgb[1] = rgb ? rgb[1] : -1; J->ctmin = ctmin; J->cmap = g_cmap;
    J->diff = g_diff; J->gain = g_diff_gain; J->pap = g_pap; J->pgain = g_pap_gain; J->pfloor = g_pap_floor; J->pbg = g_pap_bg; J->pbgv = g_pap_bgv;
    if (g_cmap && !rgb) J->ctmin = g_cmap_ctmin;
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
    {   /* the raw CT median over CT > 0 */
        const uint64_t *h = P->hist[0]; long long c = 0; g_ct_median = 0;
        for (int v = 1; v < 256; v++) { c += (long long)h[v]; if (2 * c >= (long long)P->m) { g_ct_median = v; break; } }
    }
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
    /* the outputs: OUT [options] of the first segment, then each "--then OUT2 [options]" (output options: --max-px, --quality, --full,
       --cmap, --rgb-ct-min, --no-ct, --each/--each-suffix/--each-ct, --rgb-agree, --diff-agree, --diff-gain, --papyrus, --papyrus-gain, --papyrus-floor, --papyrus-bg; the inputs, crop, statistics
       and contrast come from the first segment and are computed once). A panel image is made once and every --max-px size of it fitted
       in memory (img_fit_rgb: the bytes of the streamed fit). */
    {
        int seg[64], nseg = 0; seg[nseg++] = 1;
        for (int a = 2; a < argc && nseg < 64; a++) if (!strcmp(argv[a], "--then")) { if (a + 1 >= argc) { fprintf(stderr, "ink-compare: --then OUT [options]\n"); rc = 2; goto end; } seg[nseg++] = a; }
        char **sv = malloc((size_t)(argc + 2) * sizeof *sv); if (!sv) { rc = 1; goto end; }
        char ckey[512] = ""; uint8_t *cimg = nullptr; int cw = 0, chh = 0;   /* the last unfitted panel image and its key */
        for (int j = 0; j < nseg && !rc; j++) {
            /* segment j as an argv: [prog, cmd, OUT, options...] */
            const int s0 = j ? seg[j] + 1 : 2, s1 = j + 1 < nseg ? seg[j + 1] : argc; int sc = 0;
            sv[sc++] = argv[0]; sv[sc++] = argv[1]; for (int a = s0; a < s1; a++) sv[sc++] = argv[a]; sv[sc] = nullptr;
            const char *out = sv[2];
            const char *each = icopt(sc, sv, "--each", nullptr);
            const int maxpx = atoi(icopt(sc, sv, "--max-px", "0")), jq = atoi(icopt(sc, sv, "--quality", "90")), half = !icflag(sc, sv, "--full");
            const char *cm = icopt(sc, sv, "--cmap", nullptr);
            if (cm && strcmp(cm, "viridis")) { fprintf(stderr, "ink-compare: --cmap viridis (the only map)\n"); rc = 2; break; }
            g_cmap = cm != nullptr; g_cmap_ctmin = atoi(icopt(sc, sv, "--rgb-ct-min", "1"));
            const int f0 = icflag(sc, sv, "--no-ct") && n > 1 ? 1 : 0;   /* --no-ct: ink panels only */
            /* the jobs of this segment: (first panel, panels, rgb pair or none, title, output) */
            enum { MJ = NIMG }; int jfirst[MJ], jnp[MJ], jrgb[MJ][2], nj = 0, jct = 0; const char *const *jlab[MJ]; char jout[MJ][1400], title[300] = ""; const char *tl[1] = {title};
            g_diff = 0; g_diff_gain = 1.f; g_pap = 0; g_pap_gain = 1.f; g_pap_floor = 0; g_pap_bg = -1; g_pap_bgv = -1;
            if (each) {
                const char *suf = icopt(sc, sv, "--each-suffix", ""); const int ect = icflag(sc, sv, "--each-ct");
                for (int k = ect ? 0 : 1; k < n; k++) { snprintf(jout[nj], sizeof jout[nj], "%s%s%s.jpg", each, lab[k], suf); jfirst[nj] = k; jnp[nj] = 1; jrgb[nj][0] = -1; jlab[nj] = nullptr; nj++; }
            } else if (icopt(sc, sv, "--papyrus", nullptr)) {   /* one panel: CT as papyrus (brown .. tan), the ink as carbon black by density; off-papyrus the ramp's dark brown */
                const char *li = icopt(sc, sv, "--papyrus", ""); int ix = -1;
                for (int k = 1; k < n; k++) if (!strcmp(lab[k], li)) ix = k;
                if (ix < 0) { fprintf(stderr, "ink-compare: --papyrus INK (an --ink label)\n"); rc = 2; break; }
                g_pap = 1; g_pap_gain = (float)atof(icopt(sc, sv, "--papyrus-gain", "1"));
                const char *fl = icopt(sc, sv, "--papyrus-floor", "0");   /* median: CT below the median raised to it; else stretched CT units */
                g_pap_floor = !strcmp(fl, "median") ? lut[0][g_ct_median] : atoi(fl);
                const char *bg = icopt(sc, sv, "--papyrus-bg", "-1");   /* off-papyrus: neutral grey 0..255, "median": the ramp at the CT median, -1: the ramp at the floor */
                if (!strcmp(bg, "median")) g_pap_bgv = lut[0][g_ct_median]; else g_pap_bg = atoi(bg);
                title[0] = 0;
                snprintf(jout[0], sizeof jout[0], "%s", out); jfirst[0] = 0; jnp[0] = 1; jrgb[0][0] = ix; jrgb[0][1] = ix; jlab[0] = nullptr; nj = 1;
                jct = atoi(icopt(sc, sv, "--rgb-ct-min", "1"));
            } else if (icopt(sc, sv, "--rgb-agree", nullptr) || icopt(sc, sv, "--diff-agree", nullptr)) {   /* one panel: recto R + B (magenta), verso G, agreement white; --diff-agree: viridis |recto - verso| */
                const char *pair = icopt(sc, sv, "--diff-agree", nullptr); g_diff = pair != nullptr;
                if (!pair) pair = icopt(sc, sv, "--rgb-agree", "");
                g_diff_gain = (float)atof(icopt(sc, sv, "--diff-gain", "1"));
                char la[128] = "", lb[128] = ""; int ix[2] = {-1, -1};
                if (sscanf(pair, "%127[^,],%127s", la, lb) == 2)
                    for (int k = 1; k < n; k++) { if (!strcmp(lab[k], la)) ix[0] = k; if (!strcmp(lab[k], lb)) ix[1] = k; }
                if (ix[0] < 0 || ix[1] < 0) { fprintf(stderr, "ink-compare: --rgb-agree / --diff-agree RECTO,VERSO (two --ink labels)\n"); rc = 2; break; }
                if (g_diff) snprintf(title, sizeof title, "|%s-%s| viridis 0=agree 255=disagree%s", la, lb, g_diff_gain != 1.f ? " (gain)" : "");
                else snprintf(title, sizeof title, "%s=magenta %s=green both=white", la, lb);
                snprintf(jout[0], sizeof jout[0], "%s", out); jfirst[0] = 0; jnp[0] = 1; jrgb[0][0] = ix[0]; jrgb[0][1] = ix[1]; jlab[0] = tl; nj = 1;
                jct = atoi(icopt(sc, sv, "--rgb-ct-min", "1"));
            } else { snprintf(jout[0], sizeof jout[0], "%s", out); jfirst[0] = f0; jnp[0] = n - f0; jrgb[0][0] = -1; jlab[0] = lab + f0; nj = 1; }
            for (int q = 0; q < nj && !rc; q++) {
                char key[512]; snprintf(key, sizeof key, "%d %d %d %d %d %d %d %d %.9g %d %.9g %d %d %d %s", jfirst[q], jnp[q], jrgb[q][0], jrgb[q][1], jct, g_cmap, g_cmap_ctmin, g_diff, (double)g_diff_gain, g_pap, (double)g_pap_gain, g_pap_floor, g_pap_bg * 1000 + g_pap_bgv, half, jlab[q] == tl ? title : jlab[q] ? "lab" : "-");
                if (!cimg || strcmp(key, ckey)) {
                    free(cimg); cimg = p3_panels(&X, jfirst[q], jnp[q], lut, jlab[q], half, 0, &cw, &chh, jrgb[q][0] >= 0 ? jrgb[q] : nullptr, jct);
                    snprintf(ckey, sizeof ckey, "%s", key);
                    if (!cimg) { fprintf(stderr, "ink-compare: cannot write %s\n", jout[q]); rc = 1; break; }
                }
                int ow = cw, oh = chh; const uint8_t *img = cimg; uint8_t *fit = nullptr;
                if (maxpx > 0 && (cw > chh ? cw : chh) > maxpx) {
                    if (!(fit = malloc((size_t)cw * chh * 3))) { rc = 1; break; }
                    memcpy(fit, cimg, (size_t)cw * chh * 3);
                    if (img_fit_rgb(&fit, &ow, &oh, maxpx, X.nth)) { free(fit); rc = 1; break; }
                    img = fit;
                }
                if (jpeg_write_rgb(jout[q], ow, oh, img, jq)) { fprintf(stderr, "ink-compare: cannot write %s\n", jout[q]); rc = 1; }
                free(fit);
            }
        }
        free(cimg); free(sv);
    }
end:
    fflush(stdout);
    for (int k = 0; k < X.n; k++) tiff_close(X.t[k]);
    for (int k = 1; k < X.n; k++) free((void *)lab[k]);
    return rc;
}

/* ufsm jpeg-fit IN.jpg OUT.jpg [--div D | --max-px N] [--quality 90] [--threads 16]: a finished JPEG shrunk (antialiased bilinear,
   aspect kept) by D per side or to a longest side of N; OUT may be IN (written beside it, then renamed). Already small enough: copied. */
int cmd_jpeg_fit(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm jpeg-fit IN.jpg OUT.jpg [--div 2 | --max-px N] [--quality 90] [--threads 16]\n"); return 2; }
    const double div = atof(icopt(argc, argv, "--div", "0")); int maxpx = atoi(icopt(argc, argv, "--max-px", "0"));
    const int q = atoi(icopt(argc, argv, "--quality", "90")), th = atoi(icopt(argc, argv, "--threads", "16"));
    if ((div > 0) == (maxpx > 0)) { fprintf(stderr, "jpeg-fit: one of --div D (> 0) or --max-px N\n"); return 2; }
    int w, h; uint8_t *rgb = jpeg_read_rgb(argv[2], &w, &h);
    if (!rgb) { fprintf(stderr, "jpeg-fit: cannot read %s\n", argv[2]); return 1; }
    if (div > 0) maxpx = (int)lround((w > h ? w : h) / div);
    const int w0 = w, h0 = h;
    if (img_fit_rgb(&rgb, &w, &h, maxpx, th)) { fprintf(stderr, "jpeg-fit: out of memory\n"); free(rgb); return 1; }
    char tmp[4200]; snprintf(tmp, sizeof tmp, "%s.fit.tmp", argv[3]);
    int rc = jpeg_write_rgb(tmp, w, h, rgb, q) || rename(tmp, argv[3]);
    free(rgb);
    if (rc) { fprintf(stderr, "jpeg-fit: cannot write %s\n", argv[3]); unlink(tmp); return 1; }
    printf("jpeg-fit: %s %dx%d -> %s %dx%d\n", argv[2], w0, h0, argv[3], w, h);
    return 0;
}
