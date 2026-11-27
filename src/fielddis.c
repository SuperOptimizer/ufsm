/* ufsm field-disagree (src/fielddis.h): model disagreement on fit fields as a labelling query signal, its check against GT oracle
   fields, and a ranked list of labelling regions. */
#include "fielddis.h"
#include "fit.h"
#include "optcheck.h"
#include "store.h"
#include "sysinfo.h"
#include "z3w.h"
#include "zarr3.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static double fd_wrap(double d) { return d - floor(d + 0.5); }   /* to [-0.5, 0.5) */
static double fd_wrap4(double d) { return d - 4 * floor(d / 4 + 0.5); }   /* to [-2, 2) */

/* the cell coordinate of physical point p (um) in set F; 0 outside its box */
static int fd_cc(const fit_fields_buf *F, const double p[3], double q[3]) {
    for (int d = 0; d < 3; d++) { q[d] = (p[d] / F->um - (double)F->origin[d]) / F->h - 0.5; if (q[d] < -0.5 || q[d] > F->shape[d] - 0.5) return 0; }
    return 1;
}
/* trilinear taps of cell coordinate q: 8 indices and weights (clamped at the faces) */
static void fd_taps(const fit_fields_buf *F, const double q[3], size_t idx[8], double wt[8]) {
    int i0[3]; double w[3];
    for (int d = 0; d < 3; d++) {
        const int lim = F->shape[d]; double c = q[d] < 0 ? 0 : q[d] > lim - 1 ? lim - 1 : q[d];
        i0[d] = (int)floor(c); if (i0[d] >= lim - 1) i0[d] = lim > 1 ? lim - 2 : 0; w[d] = lim > 1 ? c - i0[d] : 0;
    }
    for (int k = 0; k < 8; k++) {
        const int a = i0[0] + (k >> 2 & 1), b = i0[1] + (k >> 1 & 1), e = i0[2] + (k & 1);
        const int a_ = a < F->shape[0] ? a : F->shape[0] - 1, b_ = b < F->shape[1] ? b : F->shape[1] - 1, e_ = e < F->shape[2] ? e : F->shape[2] - 1;
        idx[k] = ((size_t)a_ * F->shape[1] + (size_t)b_) * F->shape[2] + (size_t)e_;
        wt[k] = ((k >> 2 & 1) ? w[0] : 1 - w[0]) * ((k >> 1 & 1) ? w[1] : 1 - w[1]) * ((k & 1) ? w[2] : 1 - w[2]);
    }
}
static size_t fd_near(const fit_fields_buf *F, const double q[3]) {
    int i[3]; for (int d = 0; d < 3; d++) { i[d] = (int)lround(q[d]); if (i[d] < 0) i[d] = 0; if (i[d] >= F->shape[d]) i[d] = F->shape[d] - 1; }
    return ((size_t)i[0] * F->shape[1] + (size_t)i[1]) * F->shape[2] + (size_t)i[2];
}
/* phase (turns) at p: 1 valid (inside, material at the nearest cell, interpolated confidence > cmin) */
static int fd_phase(const fit_fields_buf *F, const double p[3], double cmin, double *ph) {
    double q[3]; if (!fd_cc(F, p, q) || !F->f.w[fd_near(F, q)]) return 0;
    size_t ix[8]; double wt[8]; fd_taps(F, q, ix, wt); double c = 0, s = 0;
    for (int k = 0; k < 8; k++) { if (wt[k] == 0) continue; c += wt[k] * (F->f.pc[ix[k]] - 128) / 127.0; s += wt[k] * (F->f.ps[ix[k]] - 128) / 127.0; }
    if (hypot(c, s) <= cmin) return 0;
    *ph = atan2(s, c) / (2 * M_PI); return 1;
}
/* winding mod 4 relative planes at p: t[a] turns in [-2, 2) over the set's baseline; 1 valid (dq-weighted resultant > 0.1) */
static int fd_w4(const fit_fields_buf *F, const double p[3], double t[3]) {
    if (!F->f.wd[0] || F->f.w4_d <= 0) return 0;
    double q[3]; if (!fd_cc(F, p, q)) return 0;
    size_t ix[8]; double wt[8]; fd_taps(F, q, ix, wt);
    for (int a = 0; a < 3; a++) {
        double c = 0, s = 0, sw = 0;
        for (int k = 0; k < 8; k++) { if (wt[k] == 0) continue; const double g = wt[k] * F->f.wd[3][ix[k]] / 255.0, an = 2 * M_PI * F->f.wd[a][ix[k]] / 256.0; c += g * cos(an); s += g * sin(an); sw += wt[k]; }
        if (sw <= 0 || hypot(c, s) / sw <= 0.1) return 0;
        t[a] = 4 * atan2(s, c) / (2 * M_PI);
    }
    return 1;
}
/* winding mod 4 disagreement of two sets at p (turns, the max over axes; < 0: no axis comparable): the shorter baseline's planes
   chained over the longer baseline Dl (k = Dl / Ds samples Ds apart along the axis, summed; each within [-2, 2) turns, so no aliasing
   at 9.6 um's 307 um baseline) against the longer one's, mod 4. Baselines not in an integer ratio: -2 (the command says DEFAULTED). */
double fdis_w4_pair(const fit_fields_buf *A, const double *ta, const fit_fields_buf *B, const double *tb, const double p[3]) {
    double Da = A->f.w4_d * A->um, Db = B->f.w4_d * B->um;
    if (Da > Db) { const fit_fields_buf *X = A; A = B; B = X; const double *t = ta; ta = tb; tb = t; const double d = Da; Da = Db; Db = d; }
    const int k = (int)lround(Db / Da); if (k < 1 || fabs(k * Da - Db) > 0.02 * Db) return -2;
    double best = -1;
    for (int e = 0; e < 3; e++) {
        double s = ta[e]; int ok = 1;
        for (int m = 1; m < k && ok; m++) { double q[3] = {p[0], p[1], p[2]}, t[3]; q[e] += m * Da; ok = fd_w4(A, q, t); if (ok) s += t[e]; }
        if (ok) best = fmax(best, fabs(fd_wrap4(tb[e] - s)));
    }
    return best;
}
/* the principal axis of the nearest cell's n n^T (power iteration, fit-compare's fc_axis); 0 without a normal */
static int fd_normal(const fit_fields_buf *F, const double p[3], double v[3]) {
    double q[3]; if (!fd_cc(F, p, q)) return 0;
    const size_t i = fd_near(F, q); const fit_fields *f = &F->f;
    const double t[6] = {f->t[0][i] / 255.0, f->t[1][i] / 255.0, f->t[2][i] / 255.0, f->t[3][i] / 255.0 - 0.5, f->t[4][i] / 255.0 - 0.5, f->t[5][i] / 255.0 - 0.5};
    const double T[3][3] = {{t[0], t[3], t[4]}, {t[3], t[1], t[5]}, {t[4], t[5], t[2]}};
    int k = T[0][0] >= T[1][1] && T[0][0] >= T[2][2] ? 0 : T[1][1] >= T[2][2] ? 1 : 2; v[0] = v[1] = v[2] = 0; v[k] = 1;
    for (int it = 0; it < 24; it++) {
        double w[3] = {0, 0, 0}; for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) w[a] += T[a][b] * v[b];
        const double l = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]); if (l < 1e-12) return 0; for (int a = 0; a < 3; a++) v[a] = w[a] / l;
    }
    return 1;
}
/* each set's unwrapped phase change along the ray p +- s v over the common valid stretch (stops at the first step any set fails); du[i];
   returns the stretch length (um) */
static double fd_ray(int n, const fit_fields_buf *const *S, const double *cmin, const double p[3], const double v[3], double R, double st, const double *ph0, double *du) {
    double end[2][FD_NMAX] = {{0}}, len = 0;
    for (int sg = 0; sg < 2; sg++) {
        const double sgn = sg ? -1 : 1; double last[FD_NMAX], u[FD_NMAX];
        for (int i = 0; i < n; i++) { last[i] = ph0[i]; u[i] = 0; }
        const int K = (int)floor(R / st);
        for (int k = 1; k <= K; k++) {
            const double q[3] = {p[0] + sgn * k * st * v[0], p[1] + sgn * k * st * v[1], p[2] + sgn * k * st * v[2]}; double ph[FD_NMAX]; int ok = 1;
            for (int i = 0; i < n && ok; i++) ok = fd_phase(S[i], q, cmin[i], &ph[i]);
            if (!ok) break;
            for (int i = 0; i < n; i++) { u[i] += fd_wrap(ph[i] - last[i]); last[i] = ph[i]; }
            len += st;
        }
        for (int i = 0; i < n; i++) end[sg][i] = u[i];
    }
    for (int i = 0; i < n; i++) du[i] = end[0][i] - end[1][i];
    return len;
}

typedef struct {
    int n; const fit_fields_buf *const *in; const fit_fields_buf *gt; const fdis_cfg *c; fdis_out *o; double st;
    fdis_gtstat *gs; size_t (*cnt)[4];
} fd_job;
static uint8_t fd_u8(double v) { return v <= 0 ? 0 : v >= 255 ? 255 : (uint8_t)lround(v); }
static void fd_rows(void *ctx, int z0, int z1, int t) {
    fd_job *J = ctx; fdis_out *o = J->o; const int n = J->n; const fdis_cfg *c = J->c; fdis_gtstat *G = &J->gs[t];
    const fit_fields_buf *S[FD_NMAX + 1]; double cm[FD_NMAX + 1];
    for (int i = 0; i < n; i++) { S[i] = J->in[i]; cm[i] = c->conf; }
    S[n] = J->gt; cm[n] = c->gt_conf;
    for (int z = z0; z < z1; z++) for (int y = 0; y < o->H; y++) for (int x = 0; x < o->W; x++) {
        const size_t oi = ((size_t)z * o->H + y) * o->W + x; double p[3]; fdis_cell_um(o, z, y, x, p);
        double ph[FD_NMAX + 1]; int ok = 1; uint8_t fl = 0;
        for (int i = 0; i < n && ok; i++) ok = fd_phase(S[i], p, cm[i], &ph[i]);
        o->pos[oi] = o->cnt[oi] = o->w4[oi] = o->dis[oi] = 0;
        if (!ok) { o->flag[oi] = 0; continue; }
        fl |= 1; double dpos = 0;
        for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) dpos = fmax(dpos, fabs(fd_wrap(ph[i] - ph[j])));
        double v[3], du[FD_NMAX + 1], dcnt = 0; int ray = fd_normal(S[0], p, v);
        if (ray) { const double L = fd_ray(n, S, cm, p, v, c->ray_um, J->st, ph, du); ray = L >= c->ray_um - 1e-9; }
        if (ray) { fl |= 2; for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) dcnt = fmax(dcnt, fabs(du[i] - du[j])); }
        double tw[FD_NMAX][3], dw4 = -1; int nw = 0, wk[FD_NMAX];
        for (int i = 0; i < n; i++) if (fd_w4(S[i], p, tw[nw])) wk[nw++] = i;
        for (int a = 0; a < nw; a++) for (int b = a + 1; b < nw; b++) dw4 = fmax(dw4, fdis_w4_pair(S[wk[a]], tw[a], S[wk[b]], tw[b], p));
        const int w4ok = dw4 >= 0; if (w4ok) fl |= 4;
        double m = dpos / 0.25; if (ray) m = fmax(m, dcnt / 0.5); if (w4ok) m = fmax(m, dw4 / 0.5);
        const uint8_t dis = fd_u8(m / 2 * 255);
        o->pos[oi] = fd_u8(dpos * 510); o->cnt[oi] = ray ? fd_u8(dcnt * 64) : 0; o->w4[oi] = w4ok ? fd_u8(dw4 * 128) : 0; o->dis[oi] = dis;
        J->cnt[t][0]++; J->cnt[t][1] += ray != 0; J->cnt[t][2] += w4ok; J->cnt[t][3] += dis >= 128;
        if (J->gt && fd_phase(J->gt, p, c->gt_conf, &ph[n])) {   /* errors against the GT: position at the cell, count on a ray the GT also covers */
            fl |= 8; int wrong = 0, nwr = 0; double dug[FD_NMAX + 1]; int rg = 0;
            if (ray) { const double L = fd_ray(n + 1, S, cm, p, v, c->ray_um, J->st, ph, dug); rg = L >= c->ray_um - 1e-9; }
            for (int i = 0; i < n; i++) {
                const int w = fabs(fd_wrap(ph[i] - ph[n])) > 0.25 || (rg && fabs(dug[i] - dug[n]) >= 0.5);
                if (w) { nwr++; G->wrong_k[i]++; if (i < 4) fl |= (uint8_t)(16 << i); }
                wrong |= w;
            }
            G->n++; G->nwrong += wrong; G->nboth += nwr == n; G->nray += rg;
            const int f[4] = {dpos > 0.25, ray && dcnt >= 0.5, w4ok && dw4 >= 0.5, dis >= 128};
            for (int k = 0; k < 4; k++) { G->flag[k] += f[k]; G->flag_wrong[k] += f[k] && wrong; }
            G->dis_hist[dis >> 4]++; G->dis_hist_wrong[dis >> 4] += wrong;
        }
        o->flag[oi] = fl;
    }
}
void fdis_cell_um(const fdis_out *o, int z, int y, int x, double p[3]) {
    const int ix[3] = {z, y, x};
    for (int d = 0; d < 3; d++) p[d] = ((double)o->origin[d] + o->h * ((double)ix[d] * o->stride + (o->stride - 1) / 2.0 + 0.5)) * o->um;
}
int fdis_run(int n, const fit_fields_buf *const *in, const fit_fields_buf *gt, const fdis_cfg *c, fdis_out *o) {
    memset(o, 0, sizeof *o);
    if (n < 2 || n > FD_NMAX || c->stride < 1 || !(c->ray_um > 0)) return -2;
    const fit_fields_buf *R = in[0]; double mc = 1e9;
    for (int i = 0; i < n; i++) mc = fmin(mc, in[i]->h * in[i]->um);
    const double st = c->step_um > 0 ? c->step_um : fmin(8, fmax(2, mc));
    o->stride = c->stride; o->h = R->h; o->um = R->um;
    for (int d = 0; d < 3; d++) o->origin[d] = R->origin[d];
    o->D = (R->shape[0] + c->stride - 1) / c->stride; o->H = (R->shape[1] + c->stride - 1) / c->stride; o->W = (R->shape[2] + c->stride - 1) / c->stride;
    const size_t N = (size_t)o->D * o->H * o->W;
    o->pos = malloc(N); o->cnt = malloc(N); o->w4 = malloc(N); o->dis = malloc(N); o->flag = malloc(N);
    const int nt = fit_par_threads(o->D);
    fdis_gtstat *gs = calloc((size_t)nt + 1, sizeof *gs); size_t (*cnt)[4] = calloc((size_t)nt + 1, sizeof *cnt);
    if (!o->pos || !o->cnt || !o->w4 || !o->dis || !o->flag || !gs || !cnt) { free(gs); free(cnt); fdis_free(o); return -1; }
    fd_job J = {n, in, gt, c, o, st, gs, cnt};
    const int ntr = fit_par(o->D, fd_rows, &J);
    for (int t = 0; t < ntr; t++) {
        o->nvalid += cnt[t][0]; o->nray += cnt[t][1]; o->nw4 += cnt[t][2]; o->ndis += cnt[t][3];
        fdis_gtstat *g = &gs[t], *G = &o->gt;
        G->n += g->n; G->nwrong += g->nwrong; G->nboth += g->nboth; G->nray += g->nray;
        for (int k = 0; k < FD_NMAX; k++) G->wrong_k[k] += g->wrong_k[k];
        for (int k = 0; k < 4; k++) { G->flag[k] += g->flag[k]; G->flag_wrong[k] += g->flag_wrong[k]; }
        for (int k = 0; k < 16; k++) { G->dis_hist[k] += g->dis_hist[k]; G->dis_hist_wrong[k] += g->dis_hist_wrong[k]; }
    }
    free(gs); free(cnt);
    return 0;
}
void fdis_free(fdis_out *o) { free(o->pos); free(o->cnt); free(o->w4); free(o->dis); free(o->flag); o->pos = o->cnt = o->w4 = o->dis = o->flag = nullptr; }

/* ---- the command ---- */

static const char *fd_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(k))) return argv[i + 1]; return d; }

/* one uint8 plane as a zarr v3 array OUT/NAME (128^3 shards, lossless) */
static int fd_write(const char *out, const char *name, const fdis_out *o, const uint8_t *a, const char *attrs) {
    char p[4096]; snprintf(p, sizeof p, "%s/%s", out, name);
    const int64_t sh[3] = {o->D, o->H, o->W};
    z3w *w = z3w_create(p, sh, 128, 0, 0, attrs); if (!w) { fprintf(stderr, "field-disagree: %s: %s\n", p, z3w_error()); return -1; }
    uint8_t *buf = malloc((size_t)128 * 128 * 128); int rc = buf ? 0 : -1;
    for (int64_t sz = 0; !rc && sz * 128 < sh[0]; sz++) for (int64_t sy = 0; !rc && sy * 128 < sh[1]; sy++) for (int64_t sx = 0; !rc && sx * 128 < sh[2]; sx++) {
        memset(buf, 0, (size_t)128 * 128 * 128);
        for (int64_t z = 0; z < 128 && sz * 128 + z < sh[0]; z++) for (int64_t y = 0; y < 128 && sy * 128 + y < sh[1]; y++) {
            const int64_t nx = sh[2] - sx * 128 < 128 ? sh[2] - sx * 128 : 128;
            memcpy(buf + ((size_t)z * 128 + y) * 128, a + ((size_t)(sz * 128 + z) * sh[1] + (size_t)(sy * 128 + y)) * sh[2] + (size_t)sx * 128, (size_t)nx);
        }
        if (z3w_write_shard(w, sz, sy, sx, buf, 4)) { fprintf(stderr, "field-disagree: %s: shard write: %s\n", p, z3w_error()); rc = -1; }
    }
    free(buf);
    if (z3w_close(w) && !rc) { fprintf(stderr, "field-disagree: %s: %s\n", p, z3w_error()); rc = -1; }
    return rc;
}

/* labelled-area samples per region: a GT phase raster (known: values < 252) read in chunk-aligned tiles over the regions' extent plus one
   region on every side, every st-th raster voxel per axis */
typedef struct { size_t *tot, *kn; int64_t r0[3], nr[3]; } fd_lab;
static int fd_lab_scan(const char *root, const char *key, double lum, double fum, int64_t RS, fd_lab *L) {
    z3 *a = z3_open(store_open(root), key, nullptr); if (!a) { fprintf(stderr, "field-disagree: --labelled %s %s: %s\n", root, key, z3_error()); return -1; }
    const z3_meta *m = z3_meta_of(a); int64_t lo[3], hi[3];
    const double rl = RS * fum / lum;   /* a region in raster voxels */
    const int st = rl / 16 >= 1 ? (int)(rl / 16) : 1;
    for (int d = 0; d < 3; d++) {
        lo[d] = (int64_t)floor(L->r0[d] * rl); hi[d] = (int64_t)ceil((L->r0[d] + L->nr[d]) * rl);
        if (lo[d] < 0) lo[d] = 0; if (hi[d] > m->shape[d]) hi[d] = m->shape[d];
    }
    const int64_t T = 256; uint8_t *buf = malloc((size_t)T * T * T);
    if (!buf) { z3_close(a); return -1; }
    for (int64_t z0 = lo[0] / T * T; z0 < hi[0]; z0 += T) for (int64_t y0 = lo[1] / T * T; y0 < hi[1]; y0 += T) for (int64_t x0 = lo[2] / T * T; x0 < hi[2]; x0 += T) {
        const int64_t o[3] = {z0 > lo[0] ? z0 : lo[0], y0 > lo[1] ? y0 : lo[1], x0 > lo[2] ? x0 : lo[2]};
        int64_t nn[3] = {z0 + T, y0 + T, x0 + T}; for (int d = 0; d < 3; d++) { if (nn[d] > hi[d]) nn[d] = hi[d]; nn[d] -= o[d]; }
        if (nn[0] <= 0 || nn[1] <= 0 || nn[2] <= 0) continue;
        if (z3_read(a, o, nn, buf, 6)) { fprintf(stderr, "field-disagree: --labelled read: %s\n", z3_error()); free(buf); z3_close(a); return -1; }
        for (int64_t z = 0; z < nn[0]; z++) { if ((o[0] + z) % st) continue;
            for (int64_t y = 0; y < nn[1]; y++) { if ((o[1] + y) % st) continue;
                for (int64_t x = 0; x < nn[2]; x++) { if ((o[2] + x) % st) continue;
                    const int64_t g[3] = {o[0] + z, o[1] + y, o[2] + x}; int64_t r[3]; int in = 1;
                    for (int d = 0; d < 3 && in; d++) { r[d] = (int64_t)floor((g[d] + 0.5) / rl) - L->r0[d]; in = r[d] >= 0 && r[d] < L->nr[d]; }
                    if (!in) continue;
                    const size_t ri = ((size_t)r[0] * L->nr[1] + (size_t)r[1]) * L->nr[2] + (size_t)r[2];
                    L->tot[ri]++; L->kn[ri] += buf[((size_t)z * nn[1] + y) * nn[2] + x] < 252;
                } } }
    }
    free(buf); z3_close(a);
    return 0;
}

typedef struct { int64_t r[3]; size_t ncell, nvalid, ndis, npos, ncnt, nw4, ngt, nwrong; double sdis, lab, near, score, cover; } fd_reg;
static int fd_cmp_reg(const void *a, const void *b) { const double x = ((const fd_reg *)a)->score, y = ((const fd_reg *)b)->score; return (x < y) - (x > y); }

int cmd_field_disagree(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: ufsm field-disagree OUT FIELDS1 FIELDS2 [FIELDS3 ..] [--stride 1] [--conf 0.2] [--ray-um 100] [--step-um 0]\n"
                        "         [--gt ORACLE-FIELDS [--gt-conf 0.3]] [--region 512] [--frame-um U] [--min-cover 0.25]\n"
                        "         [--labelled ROOT KEY --lab-um U] [--near 0.1] [--tag T] [--threads 6]\n"
                        "  where fit-field sets of several models disagree (src/fielddis.h): OUT/{dis,pos,cnt,w4,flag} zarr arrays on FIELDS1's cells,\n"
                        "  OUT/regions.tsv (regions of --region frame voxels ranked by disagreement x (1 - labelled) x closeness to labels),\n"
                        "  with --gt: OUT/gt.tsv, does disagreement find where a model is wrong (precision / recall per metric)\n"); return 2;
    }
    const char *out = argv[2]; const char *ins[FD_NMAX]; int n = 0;
    for (int i = 3; i < argc; i++) {
        if (!strncmp(argv[i], "--", 2)) { if (!strcmp(argv[i], opt_n("--labelled"))) i++; i++; continue; }
        if (n >= FD_NMAX) { fprintf(stderr, "field-disagree: at most %d field sets\n", FD_NMAX); return 2; }
        ins[n++] = argv[i];
    }
    if (n < 2) { fprintf(stderr, "field-disagree: two or more field sets\n"); return 2; }
    fdis_cfg c = {.conf = opt_atof("--conf", fd_opt(argc, argv, "--conf", "0.2")), .gt_conf = opt_atof("--gt-conf", fd_opt(argc, argv, "--gt-conf", "0.3")),
                  .ray_um = opt_atof("--ray-um", fd_opt(argc, argv, "--ray-um", "100")), .step_um = opt_atof("--step-um", fd_opt(argc, argv, "--step-um", "0")),
                  .stride = opt_atoi("--stride", fd_opt(argc, argv, "--stride", "1"))};
    const char *gtd = fd_opt(argc, argv, "--gt", nullptr), *tag = fd_opt(argc, argv, "--tag", out);
    const int nth = ufsm_threads(opt_atoi("--threads", fd_opt(argc, argv, "--threads", "6")), 6); fit_set_threads(nth);
    const char *lroot = nullptr, *lkey = nullptr;
    for (int i = 3; i + 2 < argc; i++) if (!strcmp(argv[i], opt_n("--labelled"))) { lroot = argv[i + 1]; lkey = argv[i + 2]; }
    const double lum = opt_atof("--lab-um", fd_opt(argc, argv, "--lab-um", "0")), nearf = opt_atof("--near", fd_opt(argc, argv, "--near", "0.1"));
    if (lroot && !(lum > 0)) { fprintf(stderr, "field-disagree: --labelled needs --lab-um (the raster's voxel size)\n"); return 2; }
    store_global_init();
    static fit_fields_buf B[FD_NMAX], G; const fit_fields_buf *P[FD_NMAX];
    for (int i = 0; i < n; i++) {
        if (fit_fields_load(ins[i], 0, 0, 1, &B[i])) { fprintf(stderr, "field-disagree: cannot load %s\n", ins[i]); return 1; }
        P[i] = &B[i];
        fprintf(stderr, "field-disagree: %s: %d x %d x %d cells of %.2f um%s\n", ins[i], B[i].shape[0], B[i].shape[1], B[i].shape[2], B[i].h * B[i].um,
                B[i].f.wd[0] ? "" : ", no winding mod 4 planes");
    }
    if (gtd && fit_fields_load(gtd, 0, 0, 1, &G)) { fprintf(stderr, "field-disagree: cannot load the GT fields %s\n", gtd); return 1; }
    int nw4 = 0; for (int i = 0; i < n; i++) nw4 += B[i].f.wd[0] != nullptr;
    if (nw4 < 2) printf("DEFAULTED: field-disagree: %d of %d sets carry winding mod 4 planes: no w4 metric\n", nw4, n);
    for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) if (B[i].f.wd[0] && B[j].f.wd[0]) {
        const double a = fmin(B[i].f.w4_d * B[i].um, B[j].f.w4_d * B[j].um), b = fmax(B[i].f.w4_d * B[i].um, B[j].f.w4_d * B[j].um), k = round(b / a);
        if (fabs(k * a - b) > 0.02 * b) printf("DEFAULTED: field-disagree: sets %d and %d: winding mod 4 baselines %.1f / %.1f um not in an integer ratio: that pair has no w4 metric\n", i, j, a, b);
    }
    fdis_out o;
    const int rc = fdis_run(n, P, gtd ? &G : nullptr, &c, &o);
    if (rc) { fprintf(stderr, "field-disagree: %s\n", rc == -1 ? "out of memory" : "bad options"); return 1; }
    mkdir(out, 0775);
    char attrs[512]; snprintf(attrs, sizeof attrs, "{\"ufsm\": {\"origin\": [%lld, %lld, %lld], \"h\": %d, \"stride\": %d, \"um\": %.6g}}", o.origin[0], o.origin[1], o.origin[2], o.h, o.stride, o.um);
    if (fd_write(out, "dis", &o, o.dis, attrs) || fd_write(out, "pos", &o, o.pos, attrs) || fd_write(out, "cnt", &o, o.cnt, attrs) || fd_write(out, "w4", &o, o.w4, attrs) || fd_write(out, "flag", &o, o.flag, attrs)) return 1;
    const size_t N = (size_t)o.D * o.H * o.W;
    printf("field-disagree %s: %zu cells, phase valid %.3f, ray %.3f, w4 %.3f; dis >= 128 on %.3f of the valid\n", tag, N, (double)o.nvalid / N,
           o.nvalid ? (double)o.nray / o.nvalid : 0, o.nvalid ? (double)o.nw4 / o.nvalid : 0, o.nvalid ? (double)o.ndis / o.nvalid : 0);
    char p[4096];
    if (gtd) {
        const fdis_gtstat *g = &o.gt; snprintf(p, sizeof p, "%s/gt.tsv", out); FILE *f = fopen(p, "w");
        if (!f) { fprintf(stderr, "field-disagree: cannot write %s\n", p); return 1; }
        const char *mn[4] = {"pos", "cnt", "w4", "dis"};
        fprintf(f, "tag\tmetric\tcells\tbase_wrong\tflagged\tprecision\trecall\tlift\n");
        const double base = g->n ? (double)g->nwrong / g->n : 0;
        for (int k = 0; k < 4; k++) {
            const double prec = g->flag[k] ? (double)g->flag_wrong[k] / g->flag[k] : 0, rec = g->nwrong ? (double)g->flag_wrong[k] / g->nwrong : 0;
            fprintf(f, "%s\t%s\t%zu\t%.4f\t%.4f\t%.4f\t%.4f\t%.2f\n", tag, mn[k], g->n, base, g->n ? (double)g->flag[k] / g->n : 0, prec, rec, base > 0 ? prec / base : 0);
        }
        for (int b = 0; b < 16; b++) fprintf(f, "%s\tdis_bin%02d\t%zu\t%.4f\t%.4f\t%.4f\t\t\n", tag, b, g->dis_hist[b], base, g->n ? (double)g->dis_hist[b] / g->n : 0, g->dis_hist[b] ? (double)g->dis_hist_wrong[b] / g->dis_hist[b] : 0);
        fclose(f);
        printf("  vs GT: %zu cells (ray %.3f): some model wrong %.3f, every model wrong %.3f;", g->n, g->n ? (double)g->nray / g->n : 0, base, g->n ? (double)g->nboth / g->n : 0);
        for (int i = 0; i < n; i++) printf(" model %d wrong %.3f", i, g->n ? (double)g->wrong_k[i] / g->n : 0);
        printf("\n");
        for (int k = 0; k < 4; k++) printf("  %-3s flagged %.3f: precision %.3f recall %.3f (lift %.2f)\n", mn[k], g->n ? (double)g->flag[k] / g->n : 0,
                                           g->flag[k] ? (double)g->flag_wrong[k] / g->flag[k] : 0, g->nwrong ? (double)g->flag_wrong[k] / g->nwrong : 0,
                                           base > 0 && g->flag[k] ? (double)g->flag_wrong[k] / g->flag[k] / base : 0);
    }
    /* regions: --region frame voxels (--frame-um, default the reference's native voxels) on a grid anchored at 0 */
    const double fum = opt_atof("--frame-um", fd_opt(argc, argv, "--frame-um", "0")) > 0 ? opt_atof("--frame-um", fd_opt(argc, argv, "--frame-um", "0")) : o.um;
    const int64_t RS = opt_atoll("--region", fd_opt(argc, argv, "--region", "512")); const double minc = opt_atof("--min-cover", fd_opt(argc, argv, "--min-cover", "0.25"));
    if (RS < 1) { fprintf(stderr, "field-disagree: --region >= 1\n"); return 2; }
    int64_t r0[3], r1[3];
    for (int d = 0; d < 3; d++) {
        const double a = o.origin[d] * o.um / fum, b = (o.origin[d] + (double)B[0].shape[d] * o.h) * o.um / fum;
        r0[d] = (int64_t)floor(a / RS); r1[d] = (int64_t)ceil(b / RS);
    }
    const int64_t nr[3] = {r1[0] - r0[0], r1[1] - r0[1], r1[2] - r0[2]}; const size_t NR = (size_t)nr[0] * nr[1] * nr[2];
    fd_reg *rg = calloc(NR, sizeof *rg); if (!rg) return 1;
    for (int64_t a = 0; a < nr[0]; a++) for (int64_t b = 0; b < nr[1]; b++) for (int64_t e = 0; e < nr[2]; e++) { fd_reg *r = &rg[((size_t)a * nr[1] + b) * nr[2] + e]; r->r[0] = r0[0] + a; r->r[1] = r0[1] + b; r->r[2] = r0[2] + e; }
    for (int z = 0; z < o.D; z++) for (int y = 0; y < o.H; y++) for (int x = 0; x < o.W; x++) {
        double pu[3]; fdis_cell_um(&o, z, y, x, pu); int64_t ri[3];
        for (int d = 0; d < 3; d++) ri[d] = (int64_t)floor(pu[d] / fum / RS) - r0[d];
        if (ri[0] < 0 || ri[1] < 0 || ri[2] < 0 || ri[0] >= nr[0] || ri[1] >= nr[1] || ri[2] >= nr[2]) continue;
        fd_reg *r = &rg[((size_t)ri[0] * nr[1] + ri[1]) * nr[2] + ri[2]]; const size_t i = ((size_t)z * o.H + y) * o.W + x; const uint8_t fl = o.flag[i];
        r->ncell++; if (!(fl & 1)) continue;
        r->nvalid++; r->ndis += o.dis[i] >= 128; r->sdis += o.dis[i]; r->npos += o.pos[i] > 127; r->ncnt += (fl & 2) && o.cnt[i] >= 32; r->nw4 += (fl & 4) && o.w4[i] >= 64;
        if (fl & 8) { r->ngt++; r->nwrong += (fl >> 4) != 0; }
    }
    const double cellv = pow(o.h * o.stride * o.um / fum, 3), regv = (double)RS * RS * RS;
    fd_lab L; memset(&L, 0, sizeof L);
    size_t *ltot = nullptr, *lkn = nullptr; int64_t lr0[3], lnr[3];
    if (lroot) {   /* the labelled samples over the regions plus one ring */
        for (int d = 0; d < 3; d++) { lr0[d] = r0[d] - 1; lnr[d] = nr[d] + 2; }
        const size_t NL = (size_t)lnr[0] * lnr[1] * lnr[2]; ltot = calloc(NL, sizeof *ltot); lkn = calloc(NL, sizeof *lkn);
        L = (fd_lab){ltot, lkn, {lr0[0], lr0[1], lr0[2]}, {lnr[0], lnr[1], lnr[2]}};
        if (!ltot || !lkn || fd_lab_scan(lroot, lkey, lum, fum, RS, &L)) return 1;
    } else { printf("  (no --labelled: regions ranked by disagreement alone)\n"); }
    size_t nk = 0;
    for (size_t i = 0; i < NR; i++) {
        fd_reg *r = &rg[i]; r->cover = r->ncell * cellv / regv; if (r->cover > 1) r->cover = 1;
        if (!r->ncell || r->cover < minc) continue;
        double lab = 0, near = 0, closeness = 1;
        if (lroot) {
            const int64_t a = r->r[0] - lr0[0], b = r->r[1] - lr0[1], e = r->r[2] - lr0[2]; size_t t0 = 0, k0 = 0, t1 = 0, k1 = 0;
            for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
                const size_t j = ((size_t)(a + dz) * lnr[1] + (size_t)(b + dy)) * lnr[2] + (size_t)(e + dx);
                if (!dz && !dy && !dx) { t0 += ltot[j]; k0 += lkn[j]; } else { t1 += ltot[j]; k1 += lkn[j]; }
            }
            lab = t0 ? (double)k0 / t0 : 0; near = t1 ? (double)k1 / t1 : 0; closeness = 0.5 + 0.5 * fmin(1, near / nearf);
        }
        r->lab = lab; r->near = near;
        r->score = (double)r->ndis / r->ncell * (1 - lab) * closeness;
        rg[nk++] = *r;
    }
    qsort(rg, nk, sizeof *rg, fd_cmp_reg);
    snprintf(p, sizeof p, "%s/regions.tsv", out); FILE *f = fopen(p, "w");
    if (!f) { fprintf(stderr, "field-disagree: cannot write %s\n", p); return 1; }
    fprintf(f, "# field-disagree %s: regions of %lld voxels of %.4g um (box z y x nz ny nx in those voxels); score = dis_share x (1 - lab) x (0.5 + 0.5 min(1, near / %.3g))\n", tag, (long long)RS, fum, nearf);
    fprintf(f, "rank\ttag\tz\ty\tx\tnz\tny\tnx\tcover\tvalid\tdis_share\tdis_mean\tpos_share\tcnt_share\tw4_share\tlab\tnear\tscore\tgt_known\twrong_share\n");
    for (size_t k = 0; k < nk; k++) {
        const fd_reg *r = &rg[k];
        fprintf(f, "%zu\t%s\t%lld\t%lld\t%lld\t%lld\t%lld\t%lld\t%.3f\t%.3f\t%.4f\t%.1f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.5f\t", k + 1, tag, (long long)(r->r[0] * RS), (long long)(r->r[1] * RS), (long long)(r->r[2] * RS),
                (long long)RS, (long long)RS, (long long)RS, r->cover, (double)r->nvalid / r->ncell, (double)r->ndis / r->ncell, r->nvalid ? r->sdis / r->nvalid : 0,
                (double)r->npos / r->ncell, (double)r->ncnt / r->ncell, (double)r->nw4 / r->ncell, r->lab, r->near, r->score);
        if (gtd && r->ngt) fprintf(f, "%.4f\t%.4f\n", (double)r->ngt / r->ncell, (double)r->nwrong / r->ngt); else fprintf(f, "na\tna\n");
    }
    fclose(f);
    printf("  %zu regions (cover >= %.2f) -> %s\n", nk, minc, p);
    free(rg); free(ltot); free(lkn); fdis_free(&o);
    for (int i = 0; i < n; i++) fit_fields_free(&B[i]);
    if (gtd) fit_fields_free(&G);
    return 0;
}
