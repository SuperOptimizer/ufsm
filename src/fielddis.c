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
/* set F's relative winding along axis a over Dl um from p (its planes chained Dl / D times; 0: not an integer ratio or a sample invalid) */
static int fd_chain(const fit_fields_buf *F, const double p[3], int a, double Dl, double *T) {
    const double D = F->f.w4_d * F->um; const int k = (int)lround(Dl / D); if (k < 1 || fabs(k * D - Dl) > 0.02 * Dl) return 0;
    double s = 0;
    for (int m = 0; m < k; m++) { double q[3] = {p[0], p[1], p[2]}, t[3]; q[a] += m * D; if (!fd_w4(F, q, t)) return 0; s += t[a]; }
    *T = s; return 1;
}
/* the GT's unwrapped phase change along axis a over Dl um from p (steps of st; 0: a sample without GT) */
static int fd_gt_axis(const fit_fields_buf *G, double cmin, const double p[3], int a, double Dl, double st, double *U) {
    const int K = (int)ceil(Dl / st); double last = 0, u = 0;
    for (int k = 0; k <= K; k++) {
        double q[3] = {p[0], p[1], p[2]}, ph; q[a] += Dl * k / K;
        if (!fd_phase(G, q, cmin, &ph)) return 0;
        if (k) u += fd_wrap(ph - last);
        last = ph;
    }
    *U = u; return 1;
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
    fdis_gtstat *gs; size_t (*cnt)[4]; size_t dbg;
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
        if (J->dbg && nw >= 2 && (oi * 2654435761u) % J->dbg == 0) {   /* UFSM_FDIS_DEBUG=K: about 1 in K cells' raw values */
            fprintf(stderr, "fdis-debug cell %d %d %d: ph", z, y, x); for (int i = 0; i < n; i++) fprintf(stderr, " %.3f", ph[i]);
            fprintf(stderr, " du"); for (int i = 0; i < n && ray; i++) fprintf(stderr, " %.2f", du[i]);
            for (int a = 0; a < nw; a++) fprintf(stderr, " | w4 set %d D %.0f um: %.2f %.2f %.2f", wk[a], S[wk[a]]->f.w4_d * S[wk[a]]->um, tw[a][0], tw[a][1], tw[a][2]);
            fprintf(stderr, " | dw4 %.2f\n", dw4);
        }
        double m = c->metrics & 1 ? dpos / 0.25 : 0; if (ray && c->metrics & 2) m = fmax(m, dcnt / 0.5); if (w4ok && c->metrics & 4) m = fmax(m, dw4 / 0.5);
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
            if (nw >= 2) {   /* winding mod 4 against the GT's unwrapped phase along each axis over the longest baseline */
                double Dl = 0; for (int a = 0; a < nw; a++) Dl = fmax(Dl, S[wk[a]]->f.w4_d * S[wk[a]]->um);
                int all = 1, any = 0;
                for (int a = 0; a < nw; a++) {
                    double e = -1;
                    for (int ax = 0; ax < 3; ax++) {   /* (all three axes or none: the pair metric then saw the same ones) */
                        double T, U; if (!fd_chain(S[wk[a]], p, ax, Dl, &T) || !fd_gt_axis(J->gt, c->gt_conf, p, ax, Dl, J->st, &U)) { e = -1; break; }
                        e = fmax(e, fabs(fd_wrap4(T - U)));
                    }
                    if (e < 0) { all = 0; continue; }
                    G->w4n_k[wk[a]]++; if (e >= 0.5) { G->w4wrong_k[wk[a]]++; any = 1; }
                }
                if (all) { G->w4n++; G->w4wrong += any; G->w4flag += w4ok && dw4 >= 0.5; G->w4flag_wrong += w4ok && dw4 >= 0.5 && any; }
            }
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
    if (n < 2 || n > FD_NMAX || c->stride < 1 || !(c->ray_um > 0) || !(c->metrics & 7)) return -2;
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
    fd_job J = {n, in, gt, c, o, st, gs, cnt, getenv("UFSM_FDIS_DEBUG") ? (size_t)atoll(getenv("UFSM_FDIS_DEBUG")) : 0};
    const int ntr = fit_par(o->D, fd_rows, &J);
    for (int t = 0; t < ntr; t++) {
        o->nvalid += cnt[t][0]; o->nray += cnt[t][1]; o->nw4 += cnt[t][2]; o->ndis += cnt[t][3];
        fdis_gtstat *g = &gs[t], *G = &o->gt;
        G->n += g->n; G->nwrong += g->nwrong; G->nboth += g->nboth; G->nray += g->nray;
        G->w4n += g->w4n; G->w4wrong += g->w4wrong; G->w4flag += g->w4flag; G->w4flag_wrong += g->w4flag_wrong;
        for (int k = 0; k < FD_NMAX; k++) { G->wrong_k[k] += g->wrong_k[k]; G->w4n_k[k] += g->w4n_k[k]; G->w4wrong_k[k] += g->w4wrong_k[k]; }
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
                        "         [--labelled ROOT KEY --lab-um U] [--near 0.1] [--metrics pos,cnt,w4] [--tag T] [--threads 6]\n"
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
                  .stride = opt_atoi("--stride", fd_opt(argc, argv, "--stride", "1")), .metrics = 0};
    {   const char *ms = fd_opt(argc, argv, "--metrics", "pos,cnt,w4");
        if (strstr(ms, "pos")) c.metrics |= 1;
        if (strstr(ms, "cnt")) c.metrics |= 2;
        if (strstr(ms, "w4")) c.metrics |= 4;
        if (!c.metrics) { fprintf(stderr, "field-disagree: --metrics: a list of pos, cnt, w4\n"); return 2; } }
    const char *gtd = fd_opt(argc, argv, "--gt", nullptr), *tag = fd_opt(argc, argv, "--tag", out);
    const int nth = ufsm_threads(opt_atoi("--threads", fd_opt(argc, argv, "--threads", "6")), 6); fit_set_threads(nth);
    const char *lroot = nullptr, *lkey = nullptr;
    for (int i = 3; i + 2 < argc; i++) if (!strcmp(argv[i], opt_n("--labelled"))) { lroot = argv[i + 1]; lkey = argv[i + 2]; }
    const double lum = opt_atof("--lab-um", fd_opt(argc, argv, "--lab-um", "0")), nearf = opt_atof("--near", fd_opt(argc, argv, "--near", "0.1"));
    if (lroot && !(lum > 0)) { fprintf(stderr, "field-disagree: --labelled needs --lab-um (the raster's voxel size)\n"); return 2; }
    store_global_init();
    fit_fields_buf B[FD_NMAX], G; const fit_fields_buf *P[FD_NMAX]; memset(B, 0, sizeof B); memset(&G, 0, sizeof G);
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
        printf("  vs GT: %zu cells (ray %.3f): some model wrong %.3f, every model wrong %.3f;", g->n, g->n ? (double)g->nray / g->n : 0, base, g->n ? (double)g->nboth / g->n : 0);
        for (int i = 0; i < n; i++) printf(" model %d wrong %.3f", i, g->n ? (double)g->wrong_k[i] / g->n : 0);
        printf("\n");
        if (g->w4n) {
            printf("  winding mod 4 vs the GT's phase along the axes (%zu cells): some model off >= 0.5 turn %.3f;", g->w4n, (double)g->w4wrong / g->w4n);
            for (int i = 0; i < n; i++) if (g->w4n_k[i]) printf(" model %d %.3f", i, (double)g->w4wrong_k[i] / g->w4n_k[i]);
            printf("; w4 flagged %.3f: precision %.3f recall %.3f\n", (double)g->w4flag / g->w4n, g->w4flag ? (double)g->w4flag_wrong / g->w4flag : 0, g->w4wrong ? (double)g->w4flag_wrong / g->w4wrong : 0);
            fprintf(f, "%s\tw4_vs_gtw4\t%zu\t%.4f\t%.4f\t%.4f\t%.4f\t%.2f\n", tag, g->w4n, (double)g->w4wrong / g->w4n, (double)g->w4flag / g->w4n,
                    g->w4flag ? (double)g->w4flag_wrong / g->w4flag : 0, g->w4wrong ? (double)g->w4flag_wrong / g->w4wrong : 0,
                    g->w4wrong && g->w4flag ? (double)g->w4flag_wrong / g->w4flag / ((double)g->w4wrong / g->w4n) : 0);
            for (int i = 0; i < n; i++) if (g->w4n_k[i]) fprintf(f, "%s\tw4_wrong_model%d\t%zu\t%.4f\t\t\t\t\n", tag, i, g->w4n_k[i], (double)g->w4wrong_k[i] / g->w4n_k[i]);
        }
        for (int i = 0; i < n; i++) fprintf(f, "%s\twrong_model%d\t%zu\t%.4f\t\t\t\t\n", tag, i, g->n, g->n ? (double)g->wrong_k[i] / g->n : 0);
        fprintf(f, "%s\tevery_model_wrong\t%zu\t%.4f\t\t\t\t\n", tag, g->n, g->n ? (double)g->nboth / g->n : 0);
        for (int k = 0; k < 4; k++) printf("  %-3s flagged %.3f: precision %.3f recall %.3f (lift %.2f)\n", mn[k], g->n ? (double)g->flag[k] / g->n : 0,
                                           g->flag[k] ? (double)g->flag_wrong[k] / g->flag[k] : 0, g->nwrong ? (double)g->flag_wrong[k] / g->nwrong : 0,
                                           base > 0 && g->flag[k] ? (double)g->flag_wrong[k] / g->flag[k] / base : 0);
        fclose(f);
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

/* ---- ufsm field-label (fielddis.h) ---- */

typedef struct {
    const fit_fields_buf *A, *B, *G; const fdl_cfg *c; float *pa, *pb, *cons, *cw; uint8_t *cls; uint8_t *phase, *conf, *kind; fdl_stat *st;
} fdl_job;
static void fdl_cellp(const fit_fields_buf *A, int z, int y, int x, double p[3]) {
    const int ix[3] = {z, y, x}; for (int d = 0; d < 3; d++) p[d] = ((double)A->origin[d] + A->h * (ix[d] + 0.5)) * A->um;
}
static uint8_t fdl_code(double ph) { double t = ph - floor(ph); int k = (int)lround(t * 252); return (uint8_t)(k >= 252 ? 0 : k); }
static void fdl_pass1(void *ctx, int z0, int z1, int t) {
    fdl_job *J = ctx; const fit_fields_buf *A = J->A; const int H = A->shape[1], W = A->shape[2]; (void)t;
    for (int z = z0; z < z1; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const size_t i = ((size_t)z * H + y) * W + x; double p[3], a, b; fdl_cellp(A, z, y, x, p);
        J->cons[i] = NAN; J->cw[i] = 0; J->cls[i] = 0;
        if (!fd_phase(A, p, J->c->conf, &a) || !fd_phase(J->B, p, J->c->conf, &b)) continue;
        J->pa[i] = (float)a; J->pb[i] = (float)b; const double d = fabs(fd_wrap(a - b));
        if (d <= J->c->agree) {
            const double ca = hypot((A->f.pc[i] - 128) / 127.0, (A->f.ps[i] - 128) / 127.0); double q[3]; fd_cc(J->B, p, q); const size_t j = fd_near(J->B, q);
            const double cb = hypot((J->B->f.pc[j] - 128) / 127.0, (J->B->f.ps[j] - 128) / 127.0);
            J->cons[i] = (float)(atan2(sin(2 * M_PI * a) + sin(2 * M_PI * b), cos(2 * M_PI * a) + cos(2 * M_PI * b)) / (2 * M_PI)); J->cw[i] = (float)fmin(1, fmin(ca, cb)); J->cls[i] = 1;
        } else J->cls[i] = d > 0.25 ? 2 : 5;
    }
}
static void fdl_pass2(void *ctx, int z0, int z1, int t) {
    fdl_job *J = ctx; const fit_fields_buf *A = J->A; const fdl_cfg *c = J->c; const int D = A->shape[0], H = A->shape[1], W = A->shape[2]; fdl_stat *S = &J->st[t];
    const int nr = (c->rmax - c->rmin) / 2 + 1, ring = nr * c->nang;
    for (int z = z0; z < z1; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const size_t i = ((size_t)z * H + y) * W + x; double p[3]; fdl_cellp(A, z, y, x, p);
        J->phase[i] = 255; J->conf[i] = 0; J->kind[i] = J->cls[i] == 1 ? 1 : J->cls[i] == 5 ? 5 : 0;
        double g = 0; const int hg = J->G && fd_phase(J->G, p, 0.3, &g);
        if (J->cls[i] == 1) {
            J->phase[i] = fdl_code(J->cons[i]); J->conf[i] = fd_u8(J->cw[i] * 255);
            if (hg) { const int ok = fabs(fd_wrap(J->cons[i] - g)) <= 0.25, b = (int)fmin(9, J->cw[i] * 10); S->nagree++; S->agree_ok += ok; S->agree_bin_n[b]++; S->agree_bin_ok[b] += ok; }
            continue;
        }
        if (J->cls[i] != 2) continue;
        double v[3], t1[3], t2[3], sc = 0, ss = 0, sw = 0; int ns = 0;
        if (fd_normal(A, p, v)) {
            const int ax = fabs(v[0]) <= fabs(v[1]) && fabs(v[0]) <= fabs(v[2]) ? 0 : fabs(v[1]) <= fabs(v[2]) ? 1 : 2; double e[3] = {0, 0, 0}; e[ax] = 1;
            t1[0] = v[1] * e[2] - v[2] * e[1]; t1[1] = v[2] * e[0] - v[0] * e[2]; t1[2] = v[0] * e[1] - v[1] * e[0];
            const double l = sqrt(t1[0] * t1[0] + t1[1] * t1[1] + t1[2] * t1[2]); for (int d = 0; d < 3; d++) t1[d] /= l;
            t2[0] = v[1] * t1[2] - v[2] * t1[1]; t2[1] = v[2] * t1[0] - v[0] * t1[2]; t2[2] = v[0] * t1[1] - v[1] * t1[0];
            for (int r = c->rmin; r <= c->rmax; r += 2) for (int k = 0; k < c->nang; k++) {
                const double an = 2 * M_PI * k / c->nang, ca = cos(an), sa = sin(an); int q[3]; const int ix[3] = {z, y, x}, lim[3] = {D, H, W}, in = 1;
                int ok = in; for (int d = 0; d < 3 && ok; d++) { q[d] = (int)lround(ix[d] + r * (ca * t1[d] + sa * t2[d])); ok = q[d] >= 0 && q[d] < lim[d]; }
                if (!ok) continue;
                const size_t j = ((size_t)q[0] * H + q[1]) * W + q[2]; if (J->cls[j] != 1) continue;
                sc += J->cw[j] * cos(2 * M_PI * J->cons[j]); ss += J->cw[j] * sin(2 * M_PI * J->cons[j]); sw += J->cw[j]; ns++;
            }
        }
        int pick = -1; double cf = 0;
        if (ns >= c->min_samp && sw > 0) {
            const double R = hypot(sc, ss) / sw, e = atan2(ss, sc) / (2 * M_PI), ea = fabs(fd_wrap(J->pa[i] - e)), eb = fabs(fd_wrap(J->pb[i] - e));
            const double ek = fmin(ea, eb), m = fabs(ea - eb);
            if (R >= 0.3 && ek <= c->max_err && m >= c->min_margin) { pick = ea <= eb ? 0 : 1; cf = R * fmin(1, m / 0.5) * fmin(1, (double)ns / (ring / 2.0)); }
        }
        if (pick < 0) J->kind[i] = 4;
        else { J->kind[i] = (uint8_t)(2 + pick); J->phase[i] = fdl_code(pick ? J->pb[i] : J->pa[i]); J->conf[i] = fd_u8(cf * 255); if (!J->conf[i]) J->conf[i] = 1; }
        if (hg) {
            const int ra = fabs(fd_wrap(J->pa[i] - g)) <= 0.25, rb = fabs(fd_wrap(J->pb[i] - g)) <= 0.25;
            S->ndis++; S->base_a += ra; S->base_b += rb; S->any_right += ra || rb; S->nAwrong_Bright += !ra && rb; S->nAwrong_Bright_pickB += !ra && rb && pick == 1; S->nB_ok += pick == 1 && rb;
            if (pick >= 0) { const int ok = pick ? rb : ra, b = (int)fmin(9, cf * 10); S->nlab++; S->ncorrect += ok; S->nA += !pick; S->nB += pick; S->bin_n[b]++; S->bin_ok[b] += ok; S->bin_a[b] += ra; }
        }
    }
}
int fdis_label(const fit_fields_buf *A, const fit_fields_buf *B, const fit_fields_buf *gt, const fdl_cfg *c, uint8_t *phase, uint8_t *conf, uint8_t *kind, fdl_stat *st) {
    memset(st, 0, sizeof *st);
    if (c->rmin < 1 || c->rmax < c->rmin || c->nang < 4) return -2;
    const size_t N = (size_t)A->shape[0] * A->shape[1] * A->shape[2]; const int nt = fit_par_threads(A->shape[0]);
    fdl_job J = {A, B, gt, c, malloc(N * 4), malloc(N * 4), malloc(N * 4), malloc(N * 4), malloc(N), phase, conf, kind, calloc((size_t)nt + 1, sizeof(fdl_stat))};
    int rc = 0;
    if (!J.pa || !J.pb || !J.cons || !J.cw || !J.cls || !J.st) rc = -1;
    else {
        fit_par(A->shape[0], fdl_pass1, &J); const int n2 = fit_par(A->shape[0], fdl_pass2, &J);
        for (int t = 0; t < n2; t++) {
            const fdl_stat *s = &J.st[t];
            st->ndis += s->ndis; st->nlab += s->nlab; st->ncorrect += s->ncorrect; st->nA += s->nA; st->nB += s->nB; st->nB_ok += s->nB_ok; st->nAwrong_Bright += s->nAwrong_Bright; st->nAwrong_Bright_pickB += s->nAwrong_Bright_pickB; st->base_a += s->base_a; st->base_b += s->base_b; st->any_right += s->any_right;
            st->nagree += s->nagree; st->agree_ok += s->agree_ok;
            for (int b = 0; b < 10; b++) { st->bin_n[b] += s->bin_n[b]; st->bin_ok[b] += s->bin_ok[b]; st->bin_a[b] += s->bin_a[b]; st->agree_bin_n[b] += s->agree_bin_n[b]; st->agree_bin_ok[b] += s->agree_bin_ok[b]; }
        }
    }
    free(J.pa); free(J.pb); free(J.cons); free(J.cw); free(J.cls); free(J.st);
    return rc;
}
static int fdl_write(const char *out, const char *name, const fit_fields_buf *A, const uint8_t *a, const char *attrs) {
    fdis_out o; memset(&o, 0, sizeof o); o.D = A->shape[0]; o.H = A->shape[1]; o.W = A->shape[2];
    return fd_write(out, name, &o, a, attrs);
}
/* ufsm field-label OUT FIELDS-A FIELDS-B [--gt ORACLE] [--conf 0.2] [--agree 0.1] [--rmin 2] [--rmax 8] [--nang 12] [--min-samp 8]
   [--max-err 0.2] [--min-margin 0.1] [--tag T] [--threads 6]: OUT/{phase,conf,kind} zarr arrays on A's cells (fielddis.h); with --gt
   one row per confidence cut on stdout (and OUT/gt.tsv) */
int cmd_field_label(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm field-label OUT FIELDS-A FIELDS-B [--gt ORACLE] [--conf 0.2] [--agree 0.1] [--rmin 2] [--rmax 8] [--nang 12] [--min-samp 8] [--max-err 0.2] [--min-margin 0.1] [--tag T] [--threads 6]\n"
                                    "  labels for the cells where two models disagree, from the sheet continuity of the agreeing cells (src/fielddis.h)\n"); return 2; }
    const char *out = argv[2], *gtd = fd_opt(argc, argv, "--gt", nullptr), *tag = fd_opt(argc, argv, "--tag", out);
    fdl_cfg c = {.conf = opt_atof("--conf", fd_opt(argc, argv, "--conf", "0.2")), .agree = opt_atof("--agree", fd_opt(argc, argv, "--agree", "0.1")),
                 .max_err = opt_atof("--max-err", fd_opt(argc, argv, "--max-err", "0.2")), .min_margin = opt_atof("--min-margin", fd_opt(argc, argv, "--min-margin", "0.1")),
                 .rmin = opt_atoi("--rmin", fd_opt(argc, argv, "--rmin", "2")), .rmax = opt_atoi("--rmax", fd_opt(argc, argv, "--rmax", "8")),
                 .nang = opt_atoi("--nang", fd_opt(argc, argv, "--nang", "12")), .min_samp = opt_atoi("--min-samp", fd_opt(argc, argv, "--min-samp", "8"))};
    fit_set_threads(ufsm_threads(opt_atoi("--threads", fd_opt(argc, argv, "--threads", "6")), 6));
    store_global_init();
    fit_fields_buf A, B, G; memset(&A, 0, sizeof A); memset(&B, 0, sizeof B); memset(&G, 0, sizeof G);
    if (fit_fields_load(argv[3], 0, 0, 1, &A) || fit_fields_load(argv[4], 0, 0, 1, &B)) { fprintf(stderr, "field-label: cannot load the fields\n"); return 1; }
    if (gtd && fit_fields_load(gtd, 0, 0, 1, &G)) { fprintf(stderr, "field-label: cannot load the GT fields %s\n", gtd); return 1; }
    const size_t N = (size_t)A.shape[0] * A.shape[1] * A.shape[2];
    uint8_t *ph = malloc(N), *cf = malloc(N), *kd = malloc(N); fdl_stat s;
    if (!ph || !cf || !kd) { fprintf(stderr, "field-label: out of memory\n"); return 1; }
    const int rc = fdis_label(&A, &B, gtd ? &G : nullptr, &c, ph, cf, kd, &s);
    if (rc) { fprintf(stderr, "field-label: %s\n", rc == -1 ? "out of memory" : "bad options"); return 1; }
    size_t nk[6] = {0}; for (size_t i = 0; i < N; i++) nk[kd[i] < 6 ? kd[i] : 0]++;
    mkdir(out, 0775);
    char attrs[512]; snprintf(attrs, sizeof attrs, "{\"ufsm\": {\"origin\": [%lld, %lld, %lld], \"h\": %d, \"um\": %.6g, \"phase_fill\": 255}}", A.origin[0], A.origin[1], A.origin[2], A.h, A.um);
    if (fdl_write(out, "phase", &A, ph, attrs) || fdl_write(out, "conf", &A, cf, attrs) || fdl_write(out, "kind", &A, kd, attrs)) return 1;
    {   char jp[4096]; snprintf(jp, sizeof jp, "%s/label.json", out); FILE *jf = fopen(jp, "w");
        if (!jf) { fprintf(stderr, "field-label: cannot write %s\n", jp); return 1; }
        fprintf(jf, "{\"origin\": [%lld, %lld, %lld], \"shape\": [%d, %d, %d], \"h\": %d, \"um\": %.9g}\n", A.origin[0], A.origin[1], A.origin[2], A.shape[0], A.shape[1], A.shape[2], A.h, A.um); fclose(jf); }
    printf("field-label %s: %zu cells: agree %zu, disagree labelled A %zu / B %zu, abstained %zu, between %zu\n", tag, N, nk[1], nk[2], nk[3], nk[4], nk[5]);
    if (gtd) {
        char p[4096]; snprintf(p, sizeof p, "%s/gt.tsv", out); FILE *f = fopen(p, "w"); if (!f) { fprintf(stderr, "field-label: cannot write %s\n", p); return 1; }
        fprintf(f, "tag\tkind\tcut\tcells_gt\tlabelled\tcorrect\tabstain\tbase_A\tbase_B\tany_right\tA_on_labelled\n");
        for (int b = 0; b < 10; b++) {   /* cumulative: labels with conf >= b / 10 */
            size_t n = 0, ok = 0, an = 0, aok = 0, la = 0; for (int k = b; k < 10; k++) { n += s.bin_n[k]; ok += s.bin_ok[k]; la += s.bin_a[k]; an += s.agree_bin_n[k]; aok += s.agree_bin_ok[k]; }
            fprintf(f, "%s\tdisagree\t%.1f\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\n", tag, b / 10.0, s.ndis, n, ok, s.ndis - n, s.base_a, s.base_b, s.any_right, la);
            fprintf(f, "%s\tagree\t%.1f\t%zu\t%zu\t%zu\t\t\t\t\t\n", tag, b / 10.0, s.nagree, an, aok);
        }
        fclose(f);
        printf("  vs GT: %zu disagreement cells: labelled %.3f, accuracy %.3f (always A %.3f, always B %.3f, either right %.3f); %zu agree cells: accuracy %.3f\n", s.ndis,
               s.ndis ? (double)s.nlab / s.ndis : 0, s.nlab ? (double)s.ncorrect / s.nlab : 0, s.ndis ? (double)s.base_a / s.ndis : 0, s.ndis ? (double)s.base_b / s.ndis : 0,
               s.ndis ? (double)s.any_right / s.ndis : 0, s.nagree, s.nagree ? (double)s.agree_ok / s.nagree : 0);
        printf("  picks A %zu / B %zu (B right %zu); A wrong + B right on %zu cells, B picked on %zu of them\n", s.nA, s.nB, s.nB_ok, s.nAwrong_Bright, s.nAwrong_Bright_pickB);
    }
    free(ph); free(cf); free(kd); fit_fields_free(&A); fit_fields_free(&B); if (gtd) fit_fields_free(&G);
    return 0;
}

/* ufsm field-label-raster OUT --shape Z,Y,X --um U --regions FILE [--frame-um F] [--cut 0.5] [--agree-cut -1] LABELDIR..: one phase raster
   (OUT/<U>, uint8 PHASE codes 0..251, 255 unknown = the mask; 128^3 shards, lossless; an OME group) of the scan's grid at U um voxels.
   Inside each region of FILE (lines "z0 y0 x0 nz ny nx" in --frame-um voxels, '#' comments) every voxel takes the label of the field-label
   cell holding its centre (the first LABELDIR covering it): disagreement labels with conf >= --cut, agree labels with conf >= --agree-cut
   (< 0: none); everything else unknown. Sources: "phase": {"root": OUT, "key": "<U>", "level": L}. */
typedef struct { long long o[3]; int n[3], h; double um; z3 *ph, *cf, *kd; } fdl_box;
int cmd_field_label_raster(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm field-label-raster OUT --shape Z,Y,X --um U --regions FILE [--frame-um F] [--cut 0.5] [--agree-cut -1] LABELDIR..\n"); return 2; }
    const char *out = argv[2], *shs = fd_opt(argc, argv, "--shape", nullptr), *rf = fd_opt(argc, argv, "--regions", nullptr);
    const double um = opt_atof("--um", fd_opt(argc, argv, "--um", "0")), fum = opt_atof("--frame-um", fd_opt(argc, argv, "--frame-um", "0")) > 0 ? opt_atof("--frame-um", fd_opt(argc, argv, "--frame-um", "0")) : um;
    const double cut = opt_atof("--cut", fd_opt(argc, argv, "--cut", "0.5")), acut = opt_atof("--agree-cut", fd_opt(argc, argv, "--agree-cut", "-1"));
    int64_t sh[3];
    if (!shs || sscanf(shs, "%lld,%lld,%lld", (long long *)&sh[0], (long long *)&sh[1], (long long *)&sh[2]) != 3 || !(um > 0) || !rf) { fprintf(stderr, "field-label-raster: --shape Z,Y,X, --um and --regions\n"); return 2; }
    store_global_init();
    fdl_box *bx = calloc((size_t)argc, sizeof *bx); int nb = 0;
    for (int i = 3; i < argc; i++) {
        if (!strncmp(argv[i], "--", 2)) { i++; continue; }
        char jp[4096]; snprintf(jp, sizeof jp, "%s/label.json", argv[i]); FILE *jf = fopen(jp, "r"); char buf[1024] = {0};
        if (!jf) { fprintf(stderr, "field-label-raster: no %s\n", jp); return 1; }
        size_t nr = fread(buf, 1, sizeof buf - 1, jf); (void)nr; fclose(jf); fdl_box *b = &bx[nb];
        if (sscanf(buf, "{\"origin\": [%lld, %lld, %lld], \"shape\": [%d, %d, %d], \"h\": %d, \"um\": %lf", &b->o[0], &b->o[1], &b->o[2], &b->n[0], &b->n[1], &b->n[2], &b->h, &b->um) != 8) { fprintf(stderr, "field-label-raster: bad %s\n", jp); return 1; }
        store *st = store_open(argv[i]); b->ph = z3_open(st, "phase", nullptr); b->cf = z3_open(st, "conf", nullptr); b->kd = z3_open(st, "kind", nullptr);
        if (!b->ph || !b->cf || !b->kd) { fprintf(stderr, "field-label-raster: %s: %s\n", argv[i], z3_error()); return 1; }
        nb++;
    }
    /* the boxes' arrays in memory (small: 256^3 cells each) */
    uint8_t **P = calloc((size_t)nb, sizeof *P), **C = calloc((size_t)nb, sizeof *C), **K = calloc((size_t)nb, sizeof *K);
    for (int b = 0; b < nb; b++) {
        const size_t n = (size_t)bx[b].n[0] * bx[b].n[1] * bx[b].n[2]; const int64_t o0[3] = {0, 0, 0}, nn[3] = {bx[b].n[0], bx[b].n[1], bx[b].n[2]};
        P[b] = malloc(n); C[b] = malloc(n); K[b] = malloc(n);
        if (!P[b] || !C[b] || !K[b] || z3_read(bx[b].ph, o0, nn, P[b], 4) || z3_read(bx[b].cf, o0, nn, C[b], 4) || z3_read(bx[b].kd, o0, nn, K[b], 4)) { fprintf(stderr, "field-label-raster: read failed: %s\n", z3_error()); return 1; }
        z3_close(bx[b].ph); z3_close(bx[b].cf); z3_close(bx[b].kd);
    }
    /* regions -> raster boxes */
    FILE *f = fopen(rf, "r"); if (!f) { fprintf(stderr, "field-label-raster: cannot read %s\n", rf); return 1; }
    int64_t (*R)[6] = nullptr; int nreg = 0, creg = 0; char ln[1024];
    while (fgets(ln, sizeof ln, f)) {
        char *hsh = strchr(ln, '#'); if (hsh) *hsh = 0; long long v[6];
        if (sscanf(ln, "%lld %lld %lld %lld %lld %lld", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) != 6) continue;
        if (nreg == creg) { creg = creg ? 2 * creg : 64; R = realloc(R, (size_t)creg * sizeof *R); }
        for (int d = 0; d < 3; d++) { R[nreg][d] = (int64_t)floor(v[d] * fum / um + 1e-6); R[nreg][3 + d] = (int64_t)ceil((v[d] + v[3 + d]) * fum / um - 1e-6) - R[nreg][d]; }
        nreg++;
    }
    fclose(f);
    mkdir(out, 0775);
    char key[64]; z3w_level_name(um, key, sizeof key); char ap[4096]; snprintf(ap, sizeof ap, "%s/%s", out, key);
    z3w *w = z3w_create(ap, sh, 128, 0, 255, "{\"ufsm\": {\"encoding\": \"phase252\", \"from\": \"field-label\"}}");
    if (!w || z3w_write_group(out, &um, 1, "field-label", nullptr)) { fprintf(stderr, "field-label-raster: %s\n", z3w_error()); return 1; }
    /* the shards any region touches, each written once */
    const int64_t ns[3] = {(sh[0] + 127) / 128, (sh[1] + 127) / 128, (sh[2] + 127) / 128};
    uint8_t *mark = calloc((size_t)ns[0] * ns[1] * ns[2], 1), *buf = malloc((size_t)128 * 128 * 128); size_t nvox = 0, nlab = 0, nshard = 0;
    for (int r = 0; r < nreg; r++) for (int64_t a = R[r][0] / 128; a * 128 < R[r][0] + R[r][3] && a < ns[0]; a++) for (int64_t b = R[r][1] / 128; b * 128 < R[r][1] + R[r][4] && b < ns[1]; b++)
        for (int64_t e = R[r][2] / 128; e * 128 < R[r][2] + R[r][5] && e < ns[2]; e++) mark[((size_t)a * ns[1] + b) * ns[2] + e] = 1;
    for (int64_t a = 0; a < ns[0]; a++) for (int64_t b = 0; b < ns[1]; b++) for (int64_t e = 0; e < ns[2]; e++) {
        if (!mark[((size_t)a * ns[1] + b) * ns[2] + e]) continue;
        memset(buf, 255, (size_t)128 * 128 * 128);
        for (int r = 0; r < nreg; r++) {
            int64_t lo[3], hi[3]; const int64_t s0[3] = {a * 128, b * 128, e * 128}; int ok = 1;
            for (int d = 0; d < 3; d++) { lo[d] = R[r][d] > s0[d] ? R[r][d] : s0[d]; hi[d] = R[r][d] + R[r][3 + d] < s0[d] + 128 ? R[r][d] + R[r][3 + d] : s0[d] + 128; if (hi[d] > sh[d]) hi[d] = sh[d]; ok &= lo[d] < hi[d]; }
            if (!ok) continue;
            for (int64_t z = lo[0]; z < hi[0]; z++) for (int64_t y = lo[1]; y < hi[1]; y++) for (int64_t x = lo[2]; x < hi[2]; x++) {
                const int64_t g[3] = {z, y, x}; uint8_t v = 255; nvox++;
                for (int k = 0; k < nb && v == 255; k++) {
                    int c[3], in = 1; for (int d = 0; d < 3 && in; d++) { c[d] = (int)floor(((g[d] + 0.5) * um / bx[k].um - bx[k].o[d]) / bx[k].h); in = c[d] >= 0 && c[d] < bx[k].n[d]; }
                    if (!in) continue;
                    const size_t i = ((size_t)c[0] * bx[k].n[1] + c[1]) * bx[k].n[2] + c[2]; const int kd = K[k][i];
                    if (((kd == 2 || kd == 3) && C[k][i] >= cut * 255) || (kd == 1 && acut >= 0 && C[k][i] >= acut * 255)) v = P[k][i];
                    else if (kd) v = 254;   /* (covered, unlabelled: kept apart from 'no box' until the end) */
                }
                if (v < 252) nlab++; else v = 255;
                buf[((size_t)(z - s0[0]) * 128 + (size_t)(y - s0[1])) * 128 + (size_t)(x - s0[2])] = v;
            }
        }
        if (z3w_write_shard(w, a, b, e, buf, 6)) { fprintf(stderr, "field-label-raster: shard write: %s\n", z3w_error()); return 1; }
        nshard++;
    }
    if (z3w_close(w)) { fprintf(stderr, "field-label-raster: %s\n", z3w_error()); return 1; }
    printf("field-label-raster: %d regions, %zu voxels, %zu labelled (%.3f), %zu shards -> %s key %s\n", nreg, nvox, nlab, nvox ? (double)nlab / nvox : 0, nshard, out, key);
    for (int b = 0; b < nb; b++) { free(P[b]); free(C[b]); free(K[b]); }
    free(P); free(C); free(K); free(bx); free(R); free(mark); free(buf);
    return 0;
}
