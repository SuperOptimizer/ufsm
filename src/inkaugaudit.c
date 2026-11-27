/* ufsm ink-augaudit (inkaug 11-27): do the scanner augmentations erase the ink signal? On ink-labelled crops (the ink-devcrops layout:
   tifxyz x / y / z.tif + labels.tif / mask.tif + crop.json) every condition (none, one scanner op at p 1, a recipe's mixture) runs the
   training's own CPU executor (scan_augment.c scan_aug_plan / scan_aug_exec) on P^3 tiles of the CT around the labelled surface, then
   reads per lattice cell (st x st label px, ink share >= .75 = ink, 0 = no ink) the window's z-scored CT along the surface normal at
   t = -D .. D voxels, raw and high-passed (HP = f - G_sigma f, 3-D Gaussian). Per scan and condition:
     dHP:  the standardised ink - no-ink difference of the high-pass energy E = mean_{|t| <= 3} HP(t)^2 (the lead's statistic);
     aHP:  the AUC of E (ink above no ink);
     aLDA: the out-of-fold AUC of a ridge LDA on the 2 (2 D + 1) profile features (2 folds by tile parity): how much linearly readable
           ink signal survives the condition (fitted on that condition's own data).
   Conditions: --cond NAME=SPEC (repeatable; SPEC "none", or op:p[:s] joined by '+', "cut=C" for the fourier band edge); without
   --cond the built-in audit list (the ink recipes' ops one at a time at their strength, the recipe, candidate recipes, fourier).
   CPU only (--threads, default 6). Output: one TSV row per scan x condition, then the scans' mean per condition. */
#include "imgops.h"
#include "json.h"
#include "optcheck.h"
#include "scan_augment.h"
#include "store.h"
#include "sysinfo.h"
#include "zarr3.h"
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char *aopt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(name))) return argv[i + 1];
    return dflt;
}
static double anow(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t amix(uint64_t x) { x += 0x9e3779b97f4a7c15ull; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull; x = (x ^ (x >> 27)) * 0x94d049bb133111ebull; return x ^ (x >> 31); }

/* ---- conditions ---- */
#define AA_MAXC 48
typedef struct { char name[48]; int none; scan_aug_cfg cfg; } acond;
static int cond_parse(acond *c, const char *name, const char *spec) {
    memset(c, 0, sizeof *c); snprintf(c->name, sizeof c->name, "%s", name);
    c->cfg.strength = 1; c->cfg.drop_ignore = 0;
    if (!strcmp(spec, "none")) { c->none = 1; return 0; }
    char buf[1024]; snprintf(buf, sizeof buf, "%s", spec);
    for (char *sv = nullptr, *t = strtok_r(buf, "+", &sv); t; t = strtok_r(nullptr, "+", &sv)) {
        if (!strncmp(t, "cut=", 4)) { c->cfg.fourier_cut = (float)atof(t + 4); continue; }
        char op[32] = ""; float p = 0, s = 0; const int n = sscanf(t, "%31[a-z]:%f:%f", op, &p, &s);
        int i = 0; for (; i < SA_N; i++) if (!strcmp(op, scan_aug_names[i])) break;
        if (n < 2 || i == SA_N || !(p >= 0 && p <= 1) || !(s >= 0 && s <= 3)) { fprintf(stderr, "ink-augaudit: bad condition item '%s' in %s\n", t, spec); return -1; }
        c->cfg.p[i] = p; c->cfg.s[i] = n >= 3 ? s : 0;
    }
    return 0;
}
/* the ink recipes' scanner block (R11d / R11P / E0 / R1E / lad base.argv, 11-27): --aug none --aug-strength 1 and these */
#define RECIPE "blur:.7:1.6+lowres:.1:.6+paganin:.2:1+tone:.3+gin:.15+noise:.15+cnoise:.1+sharpen:.03+ring:.1+slice:.1+quant:.1+bias:.15+cupping:.1"
static const char *const g_builtin[][2] = {
    {"none", "none"},
    {"blur", "blur:1:1.6"}, {"blur.8", "blur:1:.8"}, {"blur.4", "blur:1:.4"}, {"lowres", "lowres:1:.6"}, {"paganin", "paganin:1:1"},
    {"paganin.5", "paganin:1:.5"}, {"sharpen", "sharpen:1"}, {"quant", "quant:1"}, {"noise", "noise:1"}, {"cnoise", "cnoise:1"},
    {"gin", "gin:1"}, {"tone", "tone:1"}, {"ring", "ring:1"}, {"slice", "slice:1"}, {"bias", "bias:1"}, {"cupping", "cupping:1"},
    {"recipe", RECIPE},
    {"recipeB", "blur:.3:.8+lowres:.1:.6+paganin:.2:.5+tone:.3+gin:.15+noise:.15+cnoise:.1+sharpen:.03+ring:.1+slice:.1+quant:.1+bias:.15+cupping:.1"},
    {"fourier", "fourier:1"}, {"fourier.06", "fourier:1+cut=.06"}, {"fourier.12", "fourier:1+cut=.12"},
    {"recipeF", RECIPE "+fourier:.3"},
};

/* ---- crops ---- */
typedef struct { int w, h, lw, lh; float *X, *Y, *Z; double scale; uint8_t *lab, *msk; char ct[512]; } acrop;
static void acrop_free(acrop *c) { free(c->X); free(c->Y); free(c->Z); free(c->lab); free(c->msk); }
static int read_json(const char *path, json **j) {
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    fseek(f, 0, SEEK_END); const long n = ftell(f); fseek(f, 0, SEEK_SET); char *t = malloc((size_t)n + 1); const size_t r = fread(t, 1, (size_t)n, f); fclose(f); t[r] = 0;
    *j = json_parse(t, r); free(t); return *j ? 0 : -1;
}
static int acrop_load(acrop *c, const char *dir) {
    memset(c, 0, sizeof *c); char p[1400]; json *j;
    snprintf(p, sizeof p, "%s/crop.json", dir); if (read_json(p, &j)) { fprintf(stderr, "ink-augaudit: cannot read %s\n", p); return -1; }
    snprintf(c->ct, sizeof c->ct, "%s", json_str(json_get(j, "ct"), "")); json_free(j);
    snprintf(p, sizeof p, "%s/meta.json", dir); if (read_json(p, &j)) { fprintf(stderr, "ink-augaudit: cannot read %s\n", p); return -1; }
    c->scale = json_num(json_at(json_get(j, "scale"), 0), 0); json_free(j);
    static const char *const nm[3] = {"x", "y", "z"}; float **A[3] = {&c->X, &c->Y, &c->Z};
    for (int k = 0; k < 3; k++) { int w, h; snprintf(p, sizeof p, "%s/%s.tif", dir, nm[k]); *A[k] = img_read_f32(p, &w, &h);
        if (!*A[k] || (k && (w != c->w || h != c->h))) { fprintf(stderr, "ink-augaudit: cannot read %s\n", p); return -1; } c->w = w; c->h = h; }
    int mw, mh; snprintf(p, sizeof p, "%s/labels.tif", dir); c->lab = img_read_u8(p, &c->lw, &c->lh);
    snprintf(p, sizeof p, "%s/mask.tif", dir); c->msk = img_read_u8(p, &mw, &mh);
    if (!c->lab || !c->msk || mw != c->lw || mh != c->lh || !c->ct[0] || !(c->scale > 0)) { fprintf(stderr, "ink-augaudit: %s: no labels / mask of one size, ct or scale\n", dir); return -1; }
    return 0;
}
static int acrop_point(const acrop *s, double r, double c, double P[3]) {   /* (z, y, x) at label px (r, c); 0 when valid */
    const double gu = c * s->scale, gv = r * s->scale;
    if (gu < 0 || gv < 0 || gu > s->w - 1 || gv > s->h - 1) return -1;
    int c0 = (int)gu, r0 = (int)gv; if (c0 >= s->w - 1) c0 = s->w - 2; if (r0 >= s->h - 1) r0 = s->h - 2;
    const double fu = gu - c0, fv = gv - r0; const size_t i00 = (size_t)r0 * s->w + c0, i01 = i00 + 1, i10 = i00 + s->w, i11 = i10 + 1;
    if (s->Z[i00] < 0 || s->Z[i01] < 0 || s->Z[i10] < 0 || s->Z[i11] < 0) return -1;
    const float *A[3] = {s->Z, s->Y, s->X};
    for (int d = 0; d < 3; d++) P[d] = (1 - fv) * ((1 - fu) * A[d][i00] + fu * A[d][i01]) + fv * ((1 - fu) * A[d][i10] + fu * A[d][i11]);
    return 0;
}

/* ---- points, tiles ---- */
typedef struct { double q[3], n[3]; int ink, tile, grp; } apt;   /* q, n at level L; ink 1 / 0 */
typedef struct { int64_t o[3]; int grp, crop, np0, np; } atile;   /* its points: pts[np0 .. np0 + np) */
typedef struct { char key[512]; z3 *ct; pthread_mutex_t mu; int64_t shape[3]; } agrp;

typedef struct {
    int P, D, nc, nf, seeds; double hps; int level;
    const acond *cond; const atile *tiles; const apt *pts; agrp *grp; int ntile, npt;
    float *feat;   /* [cond][point][nf] */
    int next; pthread_mutex_t mu; double t0; int done;
} actx;

static void gauss3(float *F, int P, double sg, float *line) {   /* in place, separable, replicated edges */
    int r = (int)ceil(3 * sg); if (r > 12) r = 12; float w[25]; double sum = 0;
    for (int t = -r; t <= r; t++) sum += w[t + r] = (float)exp(-.5 * t * t / (sg * sg));
    for (int t = 0; t <= 2 * r; t++) w[t] = (float)(w[t] / sum);
    const size_t P2 = (size_t)P * P;
    for (size_t row = 0; row < P2; row++) {   /* x */
        float *f = F + row * P; memcpy(line, f, P * sizeof *f);
        for (int x = 0; x < P; x++) { float v = 0; for (int t = -r; t <= r; t++) { int j = x + t; j = j < 0 ? 0 : j >= P ? P - 1 : j; v += w[t + r] * line[j]; } f[x] = v; }
    }
    float *pl = malloc((size_t)P * P * sizeof *pl); if (!pl) abort();
    for (int z = 0; z < P; z++) {   /* y: rows of a plane */
        float *f = F + z * P2; memcpy(pl, f, P2 * sizeof *f);
        for (int y = 0; y < P; y++) { float *o = f + (size_t)y * P; memset(o, 0, P * sizeof *o);
            for (int t = -r; t <= r; t++) { int j = y + t; j = j < 0 ? 0 : j >= P ? P - 1 : j; const float *s = pl + (size_t)j * P, ww = w[t + r]; for (int x = 0; x < P; x++) o[x] += ww * s[x]; } }
    }
    for (int y = 0; y < P; y++) {   /* z: one (z, x) slab per y */
        for (int z = 0; z < P; z++) memcpy(pl + (size_t)z * P, F + z * P2 + (size_t)y * P, P * sizeof *pl);
        for (int z = 0; z < P; z++) { float *o = F + z * P2 + (size_t)y * P; memset(o, 0, P * sizeof *o);
            for (int t = -r; t <= r; t++) { int j = z + t; j = j < 0 ? 0 : j >= P ? P - 1 : j; const float *s = pl + (size_t)j * P, ww = w[t + r]; for (int x = 0; x < P; x++) o[x] += ww * s[x]; } }
    }
    free(pl);
}
static float tri(const float *F, int P, double z, double y, double x) {
    if (!(z >= 0 && y >= 0 && x >= 0 && z < P - 1 && y < P - 1 && x < P - 1)) return 0;
    const int iz = (int)z, iy = (int)y, ix = (int)x; const float fz = (float)(z - iz), fy = (float)(y - iy), fx = (float)(x - ix);
    const size_t P2 = (size_t)P * P, k = (size_t)iz * P2 + (size_t)iy * P + ix;
    const float c00 = F[k] + fx * (F[k + 1] - F[k]), c01 = F[k + P] + fx * (F[k + P + 1] - F[k + P]), c10 = F[k + P2] + fx * (F[k + P2 + 1] - F[k + P2]),
                c11 = F[k + P2 + P] + fx * (F[k + P2 + P + 1] - F[k + P2 + P]);
    const float c0 = c00 + fy * (c01 - c00), c1 = c10 + fy * (c11 - c10);
    return c0 + fz * (c1 - c0);
}
static void *aa_worker(void *arg) {
    actx *X = arg; const int P = X->P; const size_t p3 = (size_t)P * P * P;
    uint8_t *u0 = malloc(p3), *u = malloc(p3); void *scr = malloc(5 * p3); float *F = malloc(p3 * sizeof *F), *G = malloc(p3 * sizeof *G), *line = malloc((size_t)P * sizeof *line);
    scan_plan *pl = malloc(sizeof *pl); uint8_t *ign = malloc(p3);
    float *cy = malloc((size_t)P * sizeof *cy), *cx = malloc((size_t)P * sizeof *cx);
    if (!u0 || !u || !scr || !F || !G || !line || !pl || !ign || !cy || !cx) { fprintf(stderr, "ink-augaudit: out of memory (P %d)\n", P); exit(1); }
    for (;;) {
        pthread_mutex_lock(&X->mu); const int ti = X->next++; pthread_mutex_unlock(&X->mu);
        if (ti >= X->ntile) break;
        const atile *T = &X->tiles[ti]; agrp *g = &X->grp[T->grp]; const int64_t n3[3] = {P, P, P};
        pthread_mutex_lock(&g->mu); const int rr = z3_read(g->ct, T->o, n3, u0, 2); pthread_mutex_unlock(&g->mu);
        if (rr) { fprintf(stderr, "ink-augaudit: CT read failed (%s): %s\n", g->key, z3_error()); exit(1); }
        for (int ci = 0; ci < X->nc; ci++) {
            const acond *C = &X->cond[ci]; const int ns = C->none ? 1 : X->seeds;
            for (int si = 0; si < ns; si++) {
                memcpy(u, u0, p3);
                if (!C->none) {
                    memset(ign, 0, p3); for (int z = 0; z < P; z++) { cy[z] = -1e4f; cx[z] = -1e4f; }   /* (no axis: ring centres at a random point outside) */
                    const int chunk[3] = {128, 128, 128};
                    const unsigned on = scan_aug_plan(&C->cfg, amix((uint64_t)ti * 1000003u + (uint64_t)ci * 7919u + (uint64_t)si), u, nullptr, 0, ign, P, T->o, cy, cx, 0, chunk, 0, pl);
                    if (on) scan_aug_exec(pl, u, scr);
                }
                double s1 = 0, s2 = 0; size_t nm = 0;   /* the window z-score over material (train's draw() normalisation) */
                for (size_t k = 0; k < p3; k++) if (u[k]) { s1 += u[k]; s2 += (double)u[k] * u[k]; nm++; }
                const double mu = nm ? s1 / nm : 0, isd = nm ? 1 / sqrt(fmax(s2 / nm - mu * mu, 1e-6)) : 1;
                for (size_t k = 0; k < p3; k++) G[k] = F[k] = (float)((u[k] - mu) * isd);
                gauss3(G, P, X->hps, line);
                for (size_t k = 0; k < p3; k++) G[k] = F[k] - G[k];
                for (int i = 0; i < T->np; i++) {
                    const apt *q = &X->pts[T->np0 + i];
                    const size_t base = (((size_t)ci * X->seeds + si) * (size_t)X->npt + (size_t)(T->np0 + i)) * X->nf;
                    for (int t = -X->D; t <= X->D; t++) {
                        const double z = q->q[0] - T->o[0] + t * q->n[0], y = q->q[1] - T->o[1] + t * q->n[1], x = q->q[2] - T->o[2] + t * q->n[2];
                        X->feat[base + (t + X->D)] = tri(F, P, z, y, x); X->feat[base + (2 * X->D + 1) + (t + X->D)] = tri(G, P, z, y, x);
                    }
                }
            }
        }
        pthread_mutex_lock(&X->mu); X->done++; const int dn = X->done; pthread_mutex_unlock(&X->mu);
        if (dn % 10 == 0 || dn == X->ntile) fprintf(stderr, "ink-augaudit: %d / %d tiles, %.0f s\n", dn, X->ntile, anow() - X->t0);
    }
    free(u0); free(u); free(scr); free(F); free(G); free(line); free(pl); free(ign); free(cy); free(cx);
    return nullptr;
}

/* ---- statistics ---- */
typedef struct { double s; int y; } aitem;
static int ai_cmp(const void *a, const void *b) { const double x = ((const aitem *)a)->s, y = ((const aitem *)b)->s; return x < y ? -1 : x > y; }
static double auc_of(aitem *it, long n) {   /* P(ink score > no-ink score), ties 1/2 */
    qsort(it, (size_t)n, sizeof *it, ai_cmp);
    double n1 = 0, n0 = 0, rs = 0;
    for (long i = 0; i < n;) { long j = i; while (j < n && it[j].s == it[i].s) j++; const double r = .5 * (i + 1 + j);
        for (long k = i; k < j; k++) { if (it[k].y) { rs += r; n1++; } else n0++; } i = j; }
    return n1 > 0 && n0 > 0 ? (rs - n1 * (n1 + 1) / 2) / (n1 * n0) : NAN;
}
static int chol(double *A, double *b, int n) {   /* A = L L^T in place (lower), b <- A^-1 b */
    for (int j = 0; j < n; j++) { double d = A[j * n + j]; for (int k = 0; k < j; k++) d -= A[j * n + k] * A[j * n + k]; if (!(d > 0)) return -1; d = sqrt(d); A[j * n + j] = d;
        for (int i = j + 1; i < n; i++) { double v = A[i * n + j]; for (int k = 0; k < j; k++) v -= A[i * n + k] * A[j * n + k]; A[i * n + j] = v / d; } }
    for (int i = 0; i < n; i++) { double v = b[i]; for (int k = 0; k < i; k++) v -= A[i * n + k] * b[k]; b[i] = v / A[i * n + i]; }
    for (int i = n - 1; i >= 0; i--) { double v = b[i]; for (int k = i + 1; k < n; k++) v -= A[k * n + i] * b[k]; b[i] = v / A[i * n + i]; }
    return 0;
}
/* LDA direction from rows sel (fold f excluded): w = (S + lam tr(S)/n I)^-1 (m1 - m0) */
static int lda_fit(const float *R, const int *sel, const int *ys, const int *fold, long n, int nf, int f, double *w) {
    double *m1 = calloc(nf, sizeof *m1), *m0 = calloc(nf, sizeof *m0), *S = calloc((size_t)nf * nf, sizeof *S); double c1 = 0, c0 = 0;
    for (long i = 0; i < n; i++) { if (fold[i] == f) continue; const float *r = R + (size_t)sel[i] * nf; double *m = ys[i] ? m1 : m0; if (ys[i]) c1++; else c0++; for (int a = 0; a < nf; a++) m[a] += r[a]; }
    if (c1 < 5 || c0 < 5) { free(m1); free(m0); free(S); return -1; }
    for (int a = 0; a < nf; a++) { m1[a] /= c1; m0[a] /= c0; }
    for (long i = 0; i < n; i++) { if (fold[i] == f) continue; const float *r = R + (size_t)sel[i] * nf; const double *m = ys[i] ? m1 : m0; double d[64];
        for (int a = 0; a < nf; a++) d[a] = r[a] - m[a];
        for (int a = 0; a < nf; a++) for (int b = 0; b <= a; b++) S[a * nf + b] += d[a] * d[b]; }
    double tr = 0; for (int a = 0; a < nf; a++) tr += S[a * nf + a];
    for (int a = 0; a < nf; a++) { for (int b = 0; b < a; b++) S[b * nf + a] = S[a * nf + b]; S[a * nf + a] += 1e-3 * tr / nf + 1e-9; }
    for (int a = 0; a < nf; a++) w[a] = m1[a] - m0[a];
    const int rc = chol(S, w, nf);
    free(m1); free(m0); free(S); return rc;
}

int cmd_ink_augaudit(int argc, char **argv) {
    const char *cdir = aopt(argc, argv, "--crops", nullptr), *out = aopt(argc, argv, "--out", nullptr), *root = aopt(argc, argv, "--root", "/vesuvius/ufsm/v2/ct"),
               *list = aopt(argc, argv, "--list", nullptr), *cache = aopt(argc, argv, "--cache", nullptr);
    const int P = opt_atoi("--P", aopt(argc, argv, "--P", "256")), D = opt_atoi("--depth", aopt(argc, argv, "--depth", "8")), st = opt_atoi("--px-stride", aopt(argc, argv, "--px-stride", "4")),
              L = opt_atoi("--level", aopt(argc, argv, "--level", "0")), maxc = opt_atoi("--max-crops", aopt(argc, argv, "--max-crops", "0")),
              every = opt_atoi("--every", aopt(argc, argv, "--every", "1")), mtile = opt_atoi("--tiles", aopt(argc, argv, "--tiles", "2")),
              seeds = opt_atoi("--seeds", aopt(argc, argv, "--seeds", "3")), maxpt = opt_atoi("--max-points", aopt(argc, argv, "--max-points", "1500"));
    const double hps = opt_atof("--hp-sigma", aopt(argc, argv, "--hp-sigma", "2"));
    const int nth = ufsm_threads(opt_atoi("--threads", aopt(argc, argv, "--threads", "0")), 6);
    if (!cdir || !out || P < 32 || P > SA_MAXP || P % 8 || D < 1 || D > 15 || st < 1 || L < 0 || L > 5 || every < 1 || mtile < 1 || seeds < 1 || !(hps > 0)) {
        fprintf(stderr, "usage: ufsm ink-augaudit --crops DIR --out TSV [--list crops.tsv] [--root /vesuvius/ufsm/v2/ct] [--level 0] [--P 256] [--depth 8] [--px-stride 4]\n"
                        "         [--hp-sigma 2] [--tiles 2 (per crop)] [--max-points 1500 (per tile)] [--seeds 3] [--max-crops N] [--every 1] [--threads 6]\n"
                        "         [--cond NAME=SPEC ...]   SPEC: none | op:p[:s][+op:p[:s]...][+cut=C]  (default: the built-in audit list)\n");
        return 2; }
    acond *cond = calloc(AA_MAXC, sizeof *cond); int nc = 0;
    for (int i = 2; i + 1 < argc; i++) { if (strcmp(argv[i], opt_n("--cond"))) continue;
        char nm[48]; const char *eq = strchr(argv[i + 1], '='); if (!eq || eq - argv[i + 1] >= 47 || nc >= AA_MAXC) { fprintf(stderr, "ink-augaudit: --cond NAME=SPEC\n"); return 2; }
        memcpy(nm, argv[i + 1], (size_t)(eq - argv[i + 1])); nm[eq - argv[i + 1]] = 0;
        if (cond_parse(&cond[nc++], nm, eq + 1)) return 2; }
    if (!nc) for (size_t i = 0; i < sizeof g_builtin / sizeof *g_builtin; i++) if (cond_parse(&cond[nc++], g_builtin[i][0], g_builtin[i][1])) return 2;
    char lp[1400]; if (!list) { snprintf(lp, sizeof lp, "%s/crops.tsv", cdir); list = lp; }
    FILE *lf = fopen(list, "r"); if (!lf) { fprintf(stderr, "ink-augaudit: cannot read %s\n", list); return 1; }
    store_global_init(); store *S = store_open(root); if (!S) { fprintf(stderr, "ink-augaudit: cannot open %s\n", root); return 1; }
    agrp *grp = calloc(64, sizeof *grp); int ng = 0;
    int capp = 1 << 16, np = 0, capt = 256, nt = 0; apt *pts = malloc((size_t)capp * sizeof *pts); atile *tiles = malloc((size_t)capt * sizeof *tiles);
    char line[4096]; int li = 0, ncrop = 0; const double sc = 1.0 / (1 << L);
    while (fgets(line, sizeof line, lf)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        if (li++ % every) continue;
        if (maxc > 0 && ncrop >= maxc) break;
        char *tab = strchr(line, '\t'); if (tab) *tab = 0; else line[strcspn(line, "\n")] = 0;
        char d[1400]; snprintf(d, sizeof d, "%s/%s", cdir, line);
        acrop C; if (acrop_load(&C, d)) return 1;
        int gi = 0; for (; gi < ng; gi++) if (!strcmp(grp[gi].key, C.ct)) break;
        if (gi == ng) {
            if (ng == 64) { fprintf(stderr, "ink-augaudit: more than 64 scans\n"); return 1; }
            char ak[600]; snprintf(ak, sizeof ak, "%s/%d", C.ct, L);
            snprintf(grp[ng].key, sizeof grp[ng].key, "%s", C.ct); grp[ng].ct = z3_open(S, ak, cache);
            if (!grp[ng].ct) { fprintf(stderr, "ink-augaudit: cannot open %s/%s: %s\n", root, ak, z3_error()); return 1; }
            const z3_meta *zm = z3_meta_of(grp[ng].ct); for (int k = 0; k < 3; k++) grp[ng].shape[k] = zm->shape[k];
            pthread_mutex_init(&grp[ng].mu, nullptr); ng++;
        }
        /* the crop's lattice cells: ink share >= .75 -> 1, 0 -> 0 (at least half the cell supervised) */
        const int p0 = np;
        for (int a = 0; a + st <= C.lh; a += st) for (int b = 0; b + st <= C.lw; b += st) {
            int sup = 0, ink = 0;
            for (int y = a; y < a + st; y++) for (int x = b; x < b + st; x++) { const size_t k = (size_t)y * C.lw + x; if (C.msk[k] >= 128) { sup++; ink += C.lab[k] >= 128; } }
            if (2 * sup < st * st) continue;
            const double sh = (double)ink / sup; if (sh > 0 && sh < .75) continue;
            const double r = a + .5 * (st - 1), c = b + .5 * (st - 1); double q[3], qu[3], qv[3], qd[3], qe[3];
            if (acrop_point(&C, r, c, q) || acrop_point(&C, r, c + 2, qu) || acrop_point(&C, r, c - 2, qd) || acrop_point(&C, r + 2, c, qv) || acrop_point(&C, r - 2, c, qe)) continue;
            const double du[3] = {qu[0] - qd[0], qu[1] - qd[1], qu[2] - qd[2]}, dv[3] = {qv[0] - qe[0], qv[1] - qe[1], qv[2] - qe[2]};
            double n[3] = {du[1] * dv[2] - du[2] * dv[1], du[2] * dv[0] - du[0] * dv[2], du[0] * dv[1] - du[1] * dv[0]}; const double nn = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            if (!(nn > 1e-9)) continue;
            if (np == capp) { capp *= 2; pts = realloc(pts, (size_t)capp * sizeof *pts); }
            apt *t = &pts[np++]; for (int k = 0; k < 3; k++) { t->q[k] = q[k] * sc; t->n[k] = n[k] / nn; } t->ink = sh >= .75; t->tile = -1; t->grp = gi;
        }
        /* tiles: centred on a pseudo-random unassigned point, keeping the points >= D + 3 sigma + 2 voxels inside */
        const double m = D + 3 * hps + 2; int ntc = 0; uint64_t rs = amix((uint64_t)ncrop + 17);
        while (ntc < mtile) {
            int un = 0; for (int i = p0; i < np; i++) un += pts[i].tile < 0;
            if (!un) break;
            int pick = (int)(amix(rs++) % (uint64_t)un), c0 = -1; for (int i = p0; i < np; i++) if (pts[i].tile < 0 && !pick--) { c0 = i; break; }
            int64_t o[3]; for (int k = 0; k < 3; k++) { int64_t v = (int64_t)floor(pts[c0].q[k]) - P / 2; if (v > grp[gi].shape[k] - P) v = grp[gi].shape[k] - P; if (v < 0) v = 0; o[k] = v; }
            int n1 = 0, n0 = 0;
            for (int i = p0; i < np; i++) { if (pts[i].tile >= 0) continue; int in = 1; for (int k = 0; k < 3; k++) { const double u = pts[i].q[k] - o[k]; in &= u >= m && u <= P - 1 - m; }
                if (in) { pts[i].tile = nt; if (pts[i].ink) n1++; else n0++; } }
            pts[c0].tile = pts[c0].tile < 0 ? -2 : pts[c0].tile;   /* (a centre outside its own tile's margin: never picked again) */
            if (nt == capt) { capt *= 2; tiles = realloc(tiles, (size_t)capt * sizeof *tiles); }
            tiles[nt] = (atile){{o[0], o[1], o[2]}, gi, ncrop, 0, 0};
            if (n1 >= 20 && n0 >= 20) { nt++; ntc++; } else for (int i = p0; i < np; i++) if (pts[i].tile == nt) pts[i].tile = -2;
        }
        acrop_free(&C); ncrop++;
        /* compact: this crop's assigned points sorted by tile, thinned to --max-points per tile */
        int w = p0;
        for (int ti = nt - ntc; ti < nt; ti++) {
            int cnt = 0; for (int i = p0; i < np; i++) cnt += pts[i].tile == ti;
            const int keep = cnt < maxpt ? cnt : maxpt; tiles[ti].np0 = w; int seen = 0, kept = 0;
            apt *tmp = malloc((size_t)(keep ? keep : 1) * sizeof *tmp);
            for (int i = p0; i < np && kept < keep; i++) if (pts[i].tile == ti) { if ((long)seen * keep / cnt == kept) tmp[kept++] = pts[i]; seen++; }
            tiles[ti].np = kept; memcpy(pts + w, tmp, (size_t)kept * sizeof *tmp); w += kept; free(tmp);
        }
        np = w;
    }
    fclose(lf);
    if (!nt) { fprintf(stderr, "ink-augaudit: no tile with >= 20 ink and no-ink cells\n"); return 1; }
    const int nf = 2 * (2 * D + 1);
    fprintf(stderr, "ink-augaudit: %d crops, %d scans, %d tiles of %d^3 (level %d), %d points, %d conditions x %d seeds, %d threads\n", ncrop, ng, nt, P, L, np, nc, seeds, nth);
    float *feat = calloc((size_t)nc * seeds * np * nf, sizeof *feat); if (!feat) { fprintf(stderr, "ink-augaudit: out of memory (features)\n"); return 1; }
    actx X = {P, D, nc, nf, seeds, hps, L, cond, tiles, pts, grp, nt, np, feat, 0, PTHREAD_MUTEX_INITIALIZER, anow(), 0};
    pthread_t th[64]; const int T = nth < 64 ? nth : 64;
    for (int i = 0; i < T; i++) pthread_create(&th[i], nullptr, aa_worker, &X);
    for (int i = 0; i < T; i++) pthread_join(th[i], nullptr);
    FILE *fo = fopen(out, "w"); if (!fo) { fprintf(stderr, "ink-augaudit: cannot write %s\n", out); return 1; }
    fprintf(fo, "# ink-augaudit P %d level %d depth %d hp-sigma %g stride %d seeds %d tiles %d points %d\n# scan\tcond\tn_ink\tn_no\tdHP\taHP\taLDA\n", P, L, D, hps, st, seeds, nt, np);
    double (*sum)[3] = calloc(nc, sizeof *sum); int *sn = calloc(nc, sizeof *sn);
    int *sel = malloc((size_t)np * seeds * sizeof *sel), *ys = malloc((size_t)np * seeds * sizeof *ys), *fold = malloc((size_t)np * seeds * sizeof *fold);
    aitem *it = malloc((size_t)np * seeds * sizeof *it); double *E = malloc((size_t)np * seeds * sizeof *E);
    for (int gi = 0; gi < ng; gi++) for (int ci = 0; ci < nc; ci++) {
        const int ns = cond[ci].none ? 1 : seeds; long n = 0; int n1 = 0, n0 = 0;
        for (int si = 0; si < ns; si++) for (int i = 0; i < np; i++) { if (pts[i].grp != gi || pts[i].tile < 0) continue;
            sel[n] = ((ci * seeds + si) * np + i); ys[n] = pts[i].ink; fold[n] = pts[i].tile & 1; n++; if (si == 0) { if (pts[i].ink) n1++; else n0++; } }
        if (n1 < 20 || n0 < 20) continue;
        { int f0 = 0; for (long i = 0; i < n; i++) f0 += !fold[i];   /* one tile parity only (a scan with one tile): folds by point, alternating */
          if (!f0 || f0 == n) for (long i = 0; i < n; i++) fold[i] = (sel[i] % np) & 1; }
        /* dHP / aHP */
        double a1 = 0, q1 = 0, a0 = 0, q0 = 0; long c1 = 0, c0 = 0;
        for (long i = 0; i < n; i++) { const float *r = feat + (size_t)sel[i] * nf + (2 * D + 1); double e = 0; for (int t = -3; t <= 3; t++) e += (double)r[D + t] * r[D + t]; e /= 7; E[i] = e;
            if (ys[i]) { a1 += e; q1 += e * e; c1++; } else { a0 += e; q0 += e * e; c0++; } it[i] = (aitem){e, ys[i]}; }
        const double m1 = a1 / c1, m0 = a0 / c0, v1 = q1 / c1 - m1 * m1, v0 = q0 / c0 - m0 * m0, dhp = (m1 - m0) / sqrt(fmax(.5 * (v1 + v0), 1e-30));
        const double ahp = auc_of(it, n);
        /* aLDA: out-of-fold scores of both folds pooled */
        double w[64]; int ok = 1;
        for (int f = 0; f < 2 && ok; f++) {
            if (lda_fit(feat, sel, ys, fold, n, nf, f, w)) { ok = 0; break; }
            for (long i = 0; i < n; i++) if (fold[i] == f) { const float *r = feat + (size_t)sel[i] * nf; double s = 0; for (int a = 0; a < nf; a++) s += w[a] * r[a]; it[i] = (aitem){s, ys[i]}; }
        }
        const double alda = ok ? auc_of(it, n) : NAN;
        fprintf(fo, "%s\t%s\t%d\t%d\t%.4f\t%.4f\t%.4f\n", grp[gi].key, cond[ci].name, n1, n0, dhp, ahp, alda);
        sum[ci][0] += dhp; sum[ci][1] += ahp; sum[ci][2] += isfinite(alda) ? alda : 0; sn[ci]++; if (!isfinite(alda)) sn[ci] += 1000;   /* (x1000: a scan without an LDA score) */
    }
    for (int ci = 0; ci < nc; ci++) { const int k = sn[ci] % 1000, kl = k - sn[ci] / 1000; if (k) fprintf(fo, "MEAN\t%s\t%d\t-\t%.4f\t%.4f\t%.4f\n", cond[ci].name, k, sum[ci][0] / k, sum[ci][1] / k, kl ? sum[ci][2] / kl : NAN); }
    fclose(fo);
    fprintf(stderr, "ink-augaudit: %s (%.0f s)\n", out, anow() - X.t0);
    for (int gi = 0; gi < ng; gi++) z3_close(grp[gi].ct);
    free(feat); free(pts); free(tiles); free(grp); free(cond); free(sum); free(sn); free(sel); free(ys); free(fold); free(it); free(E);
    return 0;
}
